//! AV1 encoder — round 2 (forward range coder + framing).
//!
//! Scope statement (round 1, May 2026 — shipped):
//!
//! - Bitstream framing (OBU header + leb128 size prefix).
//! - Sequence Header OBU body — profile 0 / 8-bit 4:2:0 /
//!   `reduced_still_picture_header = 1`. Single still-picture stream.
//! - Frame Header OBU body — single KEY frame, single tile, single
//!   64×64 superblock, fixed `base_q_idx`, all loop-filter / CDEF /
//!   LR / film-grain features off.
//! - Tile-group OBU body — round-1 PLACEHOLDER (16-byte zero stub).
//!
//! Round 2 (May 2026 — in progress):
//!
//! - **Forward range coder** ([`symbol::SymbolEncoder`]) — bit-exact
//!   inverse of [`crate::symbol::SymbolDecoder`], pinned by
//!   `decode(encode(symbols)) == symbols` roundtrip tests on the
//!   default partition CDF and a representative mix of bool/symbol
//!   streams. Uses a wide-bigint V-tracking representation rather
//!   than the streaming carry-deferral scheme libaom uses; correct
//!   for round-2 single-superblock payloads (<1 KB) but a future
//!   round-3+ optimisation can switch to a precarry-buffer encoder
//!   if memory matters.
//!
//! ## Round 3+ checklist
//!
//! Items 2-6 below were originally scoped for round 2 but the
//! forward range coder turned out to be the biggest engineering
//! risk; round 2 ships only item 1 cleanly. Items 2-6 are deferred
//! with the following acceptance criteria:
//!
//! 2. **Partition emit** — single 64×64 SB, no recursive partitioning.
//!    Mirrors [`crate::decode::superblock::decode_partition_node`]:
//!    write `partition_cdf[bsl_ctx*4 + ctx]` symbol with above/left
//!    context = 0/0 for the first SB. The CDF lives in
//!    [`crate::cdfs::DEFAULT_PARTITION_CDF`].
//! 3. **DC_PRED intra mode emit** — mirror the decoder's neighbour
//!    derivation in
//!    [`crate::decode::superblock::decode_leaf_block`]: read
//!    `mode_ctx_bucket(above_mode)` × `mode_ctx_bucket(left_mode)`
//!    and emit through `kf_y_mode_cdf[a][l]` from
//!    [`crate::cdfs::DEFAULT_KF_Y_MODE_CDF`]. Also requires emitting
//!    `skip` / `segment_id` / `cdef` / `delta_q` / `delta_lf` (most
//!    of these are no-ops because the round-2 frame header turns the
//!    features off).
//! 4. **TX-type emit** — DCT_DCT only. Skipped if `coded_lossless`
//!    forces `Only4x4`; otherwise mirror
//!    [`crate::decode::tile::TileDecoder::decode_tx_depth`] +
//!    `decode_tx_type` reads.
//! 5. **Forward 4×4 DCT** in [`transform`] — currently only
//!    `residual4x4` shipped. Pin against `inverse_2d_spec` for ±1
//!    LSB roundtrip.
//! 6. **Coefficient entropy emit** — `txb_skip` / `eob_pt` /
//!    `coeff_base_*` / signs / Golomb-Rice tail. Mirror
//!    [`crate::decode::coeffs::decode_coefficients`].
//! 7. **dav1d cross-validation** — gate on items 1-6 producing a
//!    decoder-readable stream first.
//!
//! Each numbered item is a substantial chunk of work; round 2
//! intentionally lands the entropy coder first so rounds 3+ can
//! focus on the per-block syntax in isolation.

pub mod bitwriter;
pub mod coeffs;
pub mod frame_header;
pub mod obu;
pub mod quant;
pub mod sequence_header;
pub mod symbol;
pub mod tile;
pub mod transform;

use std::collections::VecDeque;

use oxideav_core::{
    CodecId, CodecParameters, Encoder, Error, Frame, MediaType, Packet, PixelFormat, Result,
    TimeBase, VideoFrame,
};

use crate::encoder::frame_header::{write_frame_header_body, EncFrame};
use crate::encoder::obu::{write_obu, write_temporal_delimiter};
use crate::encoder::sequence_header::{write_sequence_header_payload, EncSequence};
use crate::obu::ObuType;

pub use crate::encoder::bitwriter::BitWriter;
pub use crate::encoder::frame_header::EncFrame as FrameConfig;
pub use crate::encoder::sequence_header::EncSequence as SequenceConfig;

/// Round-1 default `base_q_idx` — chosen as a moderate-quality
/// quantiser (libaom's `crf 30` lands near here for 8-bit content).
pub const DEFAULT_BASE_Q_IDX: u8 = 100;

/// One-shot helper that emits a complete single-frame still-picture
/// AV1 stream:
///
/// ```text
///   OBU_TEMPORAL_DELIMITER
///   OBU_SEQUENCE_HEADER
///   OBU_FRAME (frame_header || tile_group_stub)
/// ```
///
/// Returns the raw byte buffer. Round-1 callers can feed this to
/// [`crate::Av1Decoder`] to confirm the headers parse — the tile
/// group stub will surface an `Error::Unsupported` from the
/// coefficient decoder until round 2 ships.
pub fn write_keyframe_stream(seq: &EncSequence, frame: &EncFrame) -> Vec<u8> {
    let mut out = Vec::new();
    write_temporal_delimiter(&mut out);
    let seq_payload = write_sequence_header_payload(seq);
    write_obu(&mut out, ObuType::SequenceHeader, &seq_payload);
    let mut frame_obu_payload = write_frame_header_body(seq, frame);
    // Round 3: emit a real entropy-coded single-superblock all-skip
    // DC_PRED tile group instead of the round-1 16-byte zero stub.
    let tile_group_body = tile::write_tile_group_skip_intra_64(seq);
    frame_obu_payload.extend_from_slice(&tile_group_body);
    write_obu(&mut out, ObuType::Frame, &frame_obu_payload);
    out
}

/// Build an Encoder factory matching the registry's `EncoderFactory`
/// signature.
pub fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    Ok(Box::new(Av1Encoder::new(params.clone())?))
}

/// Round-1 AV1 encoder. Produces a single-frame OBU stream per
/// `send_frame()` call.
///
/// The encoder enforces the round-1 envelope at construction time:
///
/// - Width / height must each be ≤ 64 (single 64-sample superblock).
/// - Pixel format must be Yuv420P (Profile 0, 4:2:0).
///
/// Frames passed to `send_frame()` are queued for `receive_packet()`
/// to drain. The pixel buffers themselves are not yet consumed —
/// round 1 ships only the OBU framing — but the call validates the
/// frame shape so callers see the right error early.
pub struct Av1Encoder {
    codec_id: CodecId,
    output_params: CodecParameters,
    seq: EncSequence,
    base_q_idx: u8,
    output_queue: VecDeque<Packet>,
    next_pts: i64,
    time_base: TimeBase,
}

impl Av1Encoder {
    pub fn new(params: CodecParameters) -> Result<Self> {
        let width = params
            .width
            .ok_or_else(|| Error::invalid("av1 encoder: CodecParameters::width is required"))?;
        let height = params
            .height
            .ok_or_else(|| Error::invalid("av1 encoder: CodecParameters::height is required"))?;
        if width == 0 || height == 0 {
            return Err(Error::invalid(
                "av1 encoder: width / height must be non-zero",
            ));
        }
        if width % 8 != 0 || height % 8 != 0 {
            return Err(Error::invalid(
                "av1 encoder (round 1): width and height must be multiples of 8",
            ));
        }
        if width > 64 || height > 64 {
            return Err(Error::unsupported(
                "av1 encoder (round 1): single-superblock frames only \
                 (width ≤ 64, height ≤ 64) — multi-SB / multi-tile is \
                 a round 2+ deliverable",
            ));
        }
        if let Some(pf) = params.pixel_format {
            if pf != PixelFormat::Yuv420P {
                return Err(Error::unsupported(format!(
                    "av1 encoder (round 1): pixel_format {pf:?} unsupported — \
                     round 1 emits Profile 0 (8-bit 4:2:0) only"
                )));
            }
        }
        let seq = EncSequence { width, height };
        let base_q_idx = DEFAULT_BASE_Q_IDX;
        let time_base = params
            .frame_rate
            .map_or(TimeBase::new(1, 90_000), |r| TimeBase::new(r.den, r.num));
        let mut output_params = params.clone();
        output_params.codec_id = CodecId::new(crate::CODEC_ID_STR);
        output_params.media_type = MediaType::Video;
        output_params.pixel_format = Some(PixelFormat::Yuv420P);
        Ok(Self {
            codec_id: CodecId::new(crate::CODEC_ID_STR),
            output_params,
            seq,
            base_q_idx,
            output_queue: VecDeque::new(),
            next_pts: 0,
            time_base,
        })
    }

    /// Override the round-1 fixed quantiser. `base_q_idx ∈ 0..=255`.
    pub fn set_base_q_idx(&mut self, q: u8) {
        self.base_q_idx = q;
    }

    fn enqueue_keyframe(&mut self, pts: i64) {
        let bytes = write_keyframe_stream(
            &self.seq,
            &EncFrame {
                base_q_idx: self.base_q_idx,
            },
        );
        let mut pkt = Packet::new(0, self.time_base, bytes);
        pkt.pts = Some(pts);
        pkt.dts = Some(pts);
        pkt.flags.keyframe = true;
        self.output_queue.push_back(pkt);
    }
}

impl Encoder for Av1Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.output_params
    }

    fn send_frame(&mut self, frame: &Frame) -> Result<()> {
        let video = match frame {
            Frame::Video(v) => v,
            _ => {
                return Err(Error::invalid(
                    "av1 encoder: send_frame requires VideoFrame",
                ));
            }
        };
        validate_frame_shape(video, &self.seq)?;
        let pts = video.pts.unwrap_or(self.next_pts);
        self.enqueue_keyframe(pts);
        self.next_pts = pts.saturating_add(1);
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Packet> {
        if let Some(p) = self.output_queue.pop_front() {
            return Ok(p);
        }
        Err(Error::NeedMore)
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

fn validate_frame_shape(v: &VideoFrame, seq: &EncSequence) -> Result<()> {
    if v.planes.len() != 3 {
        return Err(Error::invalid(format!(
            "av1 encoder: expected 3-plane Yuv420P frame, got {} planes",
            v.planes.len()
        )));
    }
    let y_stride = v.planes[0].stride;
    if (y_stride as u32) < seq.width {
        return Err(Error::invalid(format!(
            "av1 encoder: Y plane stride {y_stride} < width {}",
            seq.width
        )));
    }
    let expected_y_len = y_stride * seq.height as usize;
    if v.planes[0].data.len() < expected_y_len {
        return Err(Error::invalid(format!(
            "av1 encoder: Y plane buffer {} bytes < {expected_y_len} required",
            v.planes[0].data.len()
        )));
    }
    let uv_w = (seq.width / 2) as usize;
    let uv_h = (seq.height / 2) as usize;
    for (i, plane) in v.planes[1..3].iter().enumerate() {
        if plane.stride < uv_w {
            return Err(Error::invalid(format!(
                "av1 encoder: chroma plane {} stride {} < {uv_w}",
                i + 1,
                plane.stride
            )));
        }
        if plane.data.len() < plane.stride * uv_h {
            return Err(Error::invalid(format!(
                "av1 encoder: chroma plane {} buffer too small",
                i + 1
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obu::{iter_obus, ObuType};

    #[test]
    fn keyframe_stream_three_obus() {
        let seq = EncSequence {
            width: 64,
            height: 64,
        };
        let bytes = write_keyframe_stream(
            &seq,
            &EncFrame {
                base_q_idx: DEFAULT_BASE_Q_IDX,
            },
        );
        let obus: Vec<_> = iter_obus(&bytes).map(|r| r.unwrap()).collect();
        assert_eq!(obus.len(), 3);
        assert_eq!(obus[0].header.obu_type, ObuType::TemporalDelimiter);
        assert_eq!(obus[1].header.obu_type, ObuType::SequenceHeader);
        assert_eq!(obus[2].header.obu_type, ObuType::Frame);
    }

    #[test]
    fn sequence_header_obu_parses() {
        let seq = EncSequence {
            width: 64,
            height: 64,
        };
        let bytes = write_keyframe_stream(
            &seq,
            &EncFrame {
                base_q_idx: DEFAULT_BASE_Q_IDX,
            },
        );
        let obus: Vec<_> = iter_obus(&bytes).map(|r| r.unwrap()).collect();
        let sh = crate::sequence_header::parse_sequence_header(obus[1].payload).unwrap();
        assert_eq!(sh.max_frame_width, 64);
        assert_eq!(sh.max_frame_height, 64);
    }

    #[test]
    fn frame_obu_parses_with_decoder_dpb() {
        use crate::dpb::Dpb;
        use crate::frame_header::parse_frame_obu_with_dpb;
        let seq = EncSequence {
            width: 32,
            height: 32,
        };
        let bytes = write_keyframe_stream(&seq, &EncFrame { base_q_idx: 80 });
        let obus: Vec<_> = iter_obus(&bytes).map(|r| r.unwrap()).collect();
        let sh = crate::sequence_header::parse_sequence_header(obus[1].payload).unwrap();
        let (fh, tg) = parse_frame_obu_with_dpb(&sh, obus[2].payload, &Dpb::new()).unwrap();
        assert_eq!(fh.frame_width, 32);
        assert_eq!(fh.frame_height, 32);
        assert_eq!(fh.quant.base_q_idx, 80);
        // Round 3: tile group is a real entropy-coded payload — at
        // least 2 bytes (`SymbolDecoder::init_symbol` requirement) but
        // bounded above by ~8 bytes for the few symbols emitted.
        assert!(tg.len() >= 2);
        assert!(tg.len() <= 32);
    }

    fn make_params(width: u32, height: u32) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        p.width = Some(width);
        p.height = Some(height);
        p.pixel_format = Some(PixelFormat::Yuv420P);
        p
    }

    #[test]
    fn av1_encoder_rejects_non_multiple_of_8() {
        assert!(Av1Encoder::new(make_params(17, 16)).is_err());
    }

    #[test]
    fn av1_encoder_rejects_oversized() {
        assert!(Av1Encoder::new(make_params(128, 64)).is_err());
    }

    #[test]
    fn av1_encoder_emits_packet_per_frame() {
        let mut enc = Av1Encoder::new(make_params(32, 32)).unwrap();

        let y = vec![128u8; 32 * 32];
        let u = vec![128u8; 16 * 16];
        let v = vec![128u8; 16 * 16];
        let vf = VideoFrame {
            pts: Some(0),
            planes: vec![
                oxideav_core::frame::VideoPlane {
                    stride: 32,
                    data: y,
                },
                oxideav_core::frame::VideoPlane {
                    stride: 16,
                    data: u,
                },
                oxideav_core::frame::VideoPlane {
                    stride: 16,
                    data: v,
                },
            ],
        };
        enc.send_frame(&Frame::Video(vf)).unwrap();
        let pkt = enc.receive_packet().unwrap();
        assert!(pkt.flags.keyframe);
        assert_eq!(pkt.pts, Some(0));
        // Three OBUs: TD + SH + FRAME.
        let obus: Vec<_> = iter_obus(&pkt.data).map(|r| r.unwrap()).collect();
        assert_eq!(obus.len(), 3);
    }
}
