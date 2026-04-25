//! AV1 decoder shim.
//!
//! Consumes OBU streams and produces reconstructed frames. Phase 7
//! wires single-reference translational inter prediction on top of
//! the full Phase 1-6 intra / transform / LR / film-grain pipeline.
//! AVIF still images and AVIS 2-frame-plus image sequences decode
//! end-to-end; multi-ref / compound / warp / OBMC / inter-intra
//! remain `Error::Unsupported`.

use std::collections::VecDeque;
use std::sync::Arc;

use oxideav_core::frame::{VideoFrame, VideoPlane};
use oxideav_core::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, PixelFormat, Result, TimeBase};

use crate::decode::{decode_tile_group, FrameState};
use crate::dpb::Dpb;
use crate::extradata::Av1CodecConfig;
use crate::frame_header::{
    parse_frame_header_with_dpb, parse_frame_obu_with_dpb, FrameHeader, FrameType,
};
use crate::obu::{iter_obus, ObuType};
use crate::sequence_header::{parse_sequence_header, SequenceHeader};
use crate::tile_group::{
    parse_tile_group_header, split_tile_payloads, tile_decode_unsupported, TilePayload,
};

/// Build the registry-side decoder factory.
pub fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(Av1Decoder::new(params.clone())))
}

pub struct Av1Decoder {
    codec_id: CodecId,
    seq_header: Option<SequenceHeader>,
    last_frame_header: Option<FrameHeader>,
    last_error: Option<Error>,
    seen_frame: bool,
    last_tile_payloads: Vec<TilePayload>,
    /// Previously-emitted frame state. Populated after each frame whose
    /// `refresh_frame_flags != 0`, cleared on `OBU_TEMPORAL_DELIMITER`
    /// when the next frame is a key. Used as the LAST reference for
    /// single-reference translational inter blocks.
    prev_frame: Option<Arc<FrameState>>,
    /// §7.20 DPB OrderHint trail. Refreshed after each successfully-
    /// decoded frame per `refresh_frame_flags`; consulted by
    /// `parse_frame_header_with_dpb` to derive `skipModeAllowed`
    /// (§5.9.21).
    dpb: Dpb,
    /// Queue of decoded frames awaiting `receive_frame()`.
    output_queue: VecDeque<Frame>,
    /// Next pts to associate with a decoded frame.
    next_pts: i64,
    /// Time base to stamp on emitted frames.
    time_base: TimeBase,
}

impl Av1Decoder {
    pub fn new(params: CodecParameters) -> Self {
        let mut me = Self {
            codec_id: params.codec_id.clone(),
            seq_header: None,
            last_frame_header: None,
            last_error: None,
            seen_frame: false,
            last_tile_payloads: Vec::new(),
            prev_frame: None,
            dpb: Dpb::new(),
            output_queue: VecDeque::new(),
            next_pts: 0,
            time_base: TimeBase::new(1, 1),
        };
        // Bootstrap from extradata if present (av1C in MP4, codec private in
        // Matroska/WebM). Failures are recorded but not fatal at construction.
        if !params.extradata.is_empty() {
            match Av1CodecConfig::parse(&params.extradata) {
                Ok(cfg) => {
                    if let Some(sh) = cfg.seq_header {
                        me.seq_header = Some(sh);
                    }
                }
                Err(e) => {
                    me.last_error = Some(e);
                }
            }
        }
        me
    }

    pub fn sequence_header(&self) -> Option<&SequenceHeader> {
        self.seq_header.as_ref()
    }

    pub fn last_frame_header(&self) -> Option<&FrameHeader> {
        self.last_frame_header.as_ref()
    }

    /// Tile byte-boundaries extracted from the most recently ingested
    /// `OBU_FRAME` or `OBU_TILE_GROUP`. Each payload is described as an
    /// `(offset, len)` pair; the offsets are relative to the `OBU_FRAME`
    /// payload (for `OBU_TILE_GROUP` alone the caller owns the payload
    /// buffer and can supply it explicitly).
    pub fn last_tile_payloads(&self) -> &[TilePayload] {
        &self.last_tile_payloads
    }

    /// Walk the OBU stream in `packet.data`, updating internal state. Returns
    /// the first error encountered (if any).
    fn ingest(&mut self, data: &[u8], pts: i64, tb: TimeBase) -> Result<()> {
        self.time_base = tb;
        self.next_pts = pts;
        for obu in iter_obus(data) {
            let obu = obu?;
            match obu.header.obu_type {
                ObuType::TemporalDelimiter => {
                    // Clear the DPB ahead of the next KEY frame. We can't
                    // know the frame_type until we see the next frame OBU,
                    // so defer the clear to the next frame-level branch.
                    // No-op here.
                }
                ObuType::Padding => {}
                ObuType::SequenceHeader => {
                    self.seq_header = Some(parse_sequence_header(obu.payload)?);
                }
                ObuType::FrameHeader | ObuType::RedundantFrameHeader => {
                    let seq = self.seq_header.as_ref().ok_or_else(|| {
                        Error::invalid("av1: frame_header before sequence_header")
                    })?;
                    let fh = parse_frame_header_with_dpb(seq, obu.payload, &self.dpb)?;
                    self.last_frame_header = Some(fh);
                    self.seen_frame = true;
                }
                ObuType::Frame => {
                    let seq = self
                        .seq_header
                        .as_ref()
                        .ok_or_else(|| Error::invalid("av1: frame_obu before sequence_header"))?
                        .clone();
                    let (fh, tg_payload) = parse_frame_obu_with_dpb(&seq, obu.payload, &self.dpb)?;
                    self.seen_frame = true;
                    if fh.frame_type == FrameType::Key {
                        self.prev_frame = None;
                        // §5.9.4 mark_ref_frames: a KEY frame implicitly
                        // resets the DPB so subsequent skip-mode
                        // derivations don't reference stale OrderHints.
                        self.dpb.reset();
                    }
                    let tile_decode_err = if let Some(ti) = fh.tile_info.as_ref() {
                        let tgh = parse_tile_group_header(tg_payload, ti)?;
                        let mut tiles = split_tile_payloads(tg_payload, ti, &tgh)?;
                        let frame_header_len = obu.payload.len() - tg_payload.len();
                        for t in &mut tiles {
                            t.offset += frame_header_len;
                        }
                        self.last_tile_payloads = tiles;
                        let sub_x = if seq.color_config.subsampling_x { 1 } else { 0 };
                        let sub_y = if seq.color_config.subsampling_y { 1 } else { 0 };
                        let mut fs = FrameState::with_bit_depth(
                            fh.frame_width,
                            fh.frame_height,
                            sub_x,
                            sub_y,
                            seq.color_config.num_planes == 1,
                            seq.color_config.bit_depth,
                        );
                        let res = decode_tile_group(
                            &seq,
                            &fh,
                            tg_payload,
                            &mut fs,
                            self.prev_frame.as_ref(),
                            &self.dpb,
                        );
                        if res.is_ok() && fh.refresh_frame_flags != 0 {
                            // §7.20 reference_frame_update_process —
                            // install both the OrderHint *and* the
                            // reconstructed planes into every slot
                            // selected by `refresh_frame_flags`. Round
                            // 14 routes the planes through the DPB so
                            // SKIP_MODE compound MC can fetch from two
                            // independent references; the single-ref
                            // `prev_frame` field is kept as a fallback
                            // for the LAST translational MC path.
                            let arc = Arc::new(clone_frame_state(&fs));
                            self.prev_frame = Some(arc.clone());
                            self.dpb
                                .refresh_with_frame(fh.refresh_frame_flags, fh.order_hint, arc);
                        }
                        if res.is_ok() && fh.show_frame {
                            self.enqueue_video_frame(&fs);
                        }
                        res
                    } else {
                        self.last_tile_payloads.clear();
                        Err(tile_decode_unsupported())
                    };
                    self.last_frame_header = Some(fh);
                    // Round 14: AV1 packets routinely carry multiple
                    // `OBU_FRAME` units (e.g. an Inter frame whose ref
                    // is built then immediately a `show_existing_frame`
                    // pulls a slot for output). Don't `return` on the
                    // first Frame OBU — keep walking the packet so
                    // subsequent frames in the same packet are also
                    // decoded. Surface the first hard error encountered.
                    tile_decode_err?;
                    continue;
                }
                ObuType::TileGroup => {
                    let Some(fh) = self.last_frame_header.as_ref() else {
                        return Err(Error::invalid(
                            "av1: OBU_TILE_GROUP without preceding frame_header",
                        ));
                    };
                    let Some(ti) = fh.tile_info.as_ref() else {
                        return Err(tile_decode_unsupported());
                    };
                    let tgh = parse_tile_group_header(obu.payload, ti)?;
                    let tiles = split_tile_payloads(obu.payload, ti, &tgh)?;
                    self.last_tile_payloads = tiles;
                    let seq = self.seq_header.as_ref().ok_or_else(|| {
                        Error::invalid("av1: OBU_TILE_GROUP before sequence_header")
                    })?;
                    let sub_x = if seq.color_config.subsampling_x { 1 } else { 0 };
                    let sub_y = if seq.color_config.subsampling_y { 1 } else { 0 };
                    let mut fs = FrameState::with_bit_depth(
                        fh.frame_width,
                        fh.frame_height,
                        sub_x,
                        sub_y,
                        seq.color_config.num_planes == 1,
                        seq.color_config.bit_depth,
                    );
                    if fh.frame_type == FrameType::Key {
                        self.prev_frame = None;
                        self.dpb.reset();
                    }
                    let res = decode_tile_group(
                        seq,
                        fh,
                        obu.payload,
                        &mut fs,
                        self.prev_frame.as_ref(),
                        &self.dpb,
                    );
                    if res.is_ok() && fh.refresh_frame_flags != 0 {
                        let arc = Arc::new(clone_frame_state(&fs));
                        self.prev_frame = Some(arc.clone());
                        self.dpb
                            .refresh_with_frame(fh.refresh_frame_flags, fh.order_hint, arc);
                    }
                    if res.is_ok() && fh.show_frame {
                        self.enqueue_video_frame(&fs);
                    }
                    res?;
                    continue;
                }
                ObuType::Metadata | ObuType::TileList => {}
                _ => {}
            }
        }
        Ok(())
    }

    fn enqueue_video_frame(&mut self, fs: &FrameState) {
        let monochrome = fs.monochrome;
        let bd = fs.bit_depth;
        let sub_x = fs.sub_x;
        let sub_y = fs.sub_y;
        let format = if bd == 8 {
            if monochrome {
                PixelFormat::Gray8
            } else if sub_x == 1 && sub_y == 1 {
                PixelFormat::Yuv420P
            } else if sub_x == 1 && sub_y == 0 {
                PixelFormat::Yuv422P
            } else {
                PixelFormat::Yuv444P
            }
        } else {
            // HBD not yet plumbed through VideoFrame (no u16 variant in
            // PixelFormat). Fall back to Yuv420P with narrowed samples.
            PixelFormat::Yuv420P
        };
        let width = fs.width;
        let height = fs.height;
        let mut planes = Vec::new();
        if bd == 8 {
            planes.push(VideoPlane {
                stride: fs.width as usize,
                data: fs.y_plane.clone(),
            });
            if !monochrome {
                planes.push(VideoPlane {
                    stride: fs.uv_width as usize,
                    data: fs.u_plane.clone(),
                });
                planes.push(VideoPlane {
                    stride: fs.uv_width as usize,
                    data: fs.v_plane.clone(),
                });
            }
        } else {
            // Narrow HBD to u8 for the VideoFrame surface by right-
            // shifting (bd - 8). Full HBD surface is a follow-up.
            let shift = bd - 8;
            let narrow = |src: &[u16]| -> Vec<u8> {
                src.iter().map(|&v| (v >> shift).min(255) as u8).collect()
            };
            planes.push(VideoPlane {
                stride: fs.width as usize,
                data: narrow(&fs.y_plane16),
            });
            if !monochrome {
                planes.push(VideoPlane {
                    stride: fs.uv_width as usize,
                    data: narrow(&fs.u_plane16),
                });
                planes.push(VideoPlane {
                    stride: fs.uv_width as usize,
                    data: narrow(&fs.v_plane16),
                });
            }
        }
        let vf = VideoFrame {
            format,
            width,
            height,
            pts: Some(self.next_pts),
            time_base: self.time_base,
            planes,
        };
        self.output_queue.push_back(Frame::Video(vf));
    }
}

/// Clone a frame state cheaply. The pixel buffers are `Vec`-based so
/// this is a simple `.clone()`; lr_unit_info / mi fields clone along.
fn clone_frame_state(fs: &FrameState) -> FrameState {
    FrameState {
        width: fs.width,
        height: fs.height,
        mi_cols: fs.mi_cols,
        mi_rows: fs.mi_rows,
        mi: fs.mi.clone(),
        sub_x: fs.sub_x,
        sub_y: fs.sub_y,
        monochrome: fs.monochrome,
        bit_depth: fs.bit_depth,
        y_plane: fs.y_plane.clone(),
        u_plane: fs.u_plane.clone(),
        v_plane: fs.v_plane.clone(),
        y_plane16: fs.y_plane16.clone(),
        u_plane16: fs.u_plane16.clone(),
        v_plane16: fs.v_plane16.clone(),
        uv_width: fs.uv_width,
        uv_height: fs.uv_height,
        lr_unit_info: fs.lr_unit_info.clone(),
        lr_cols: fs.lr_cols,
        lr_rows: fs.lr_rows,
        lr_unit_size: fs.lr_unit_size,
        cdef_idx: fs.cdef_idx.clone(),
        cdef_sb_cols: fs.cdef_sb_cols,
        cdef_sb_rows: fs.cdef_sb_rows,
    }
}

impl Decoder for Av1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        let pts = packet.pts.unwrap_or(0);
        match self.ingest(&packet.data, pts, packet.time_base) {
            Ok(()) => Ok(()),
            Err(Error::Unsupported(s)) => Err(Error::Unsupported(s)),
            Err(e) => Err(e),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        if let Some(f) = self.output_queue.pop_front() {
            return Ok(f);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.last_frame_header = None;
        self.last_error = None;
        self.seen_frame = false;
        self.last_tile_payloads.clear();
        self.prev_frame = None;
        self.dpb.reset();
        self.output_queue.clear();
        Ok(())
    }
}
