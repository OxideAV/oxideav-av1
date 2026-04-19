//! AV1 decoder shim.
//!
//! The decoder consumes OBU streams, tracks the sequence header, parses
//! the frame header through `tile_info()` (§5.9.15), and splits the
//! `tile_group_obu()` payload into per-tile byte ranges accessible via
//! `Av1Decoder::last_tile_payloads()`. Pixel reconstruction itself —
//! partition walk + coefficient decode + prediction + transforms +
//! deblock + CDEF + loop restoration — is not yet implemented. See
//! `tile_group::tile_decode_unsupported` for the spec-section list the
//! error message points at.

use oxideav_codec::Decoder;
use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Result};

use crate::decode::{decode_tile_group, FrameState};
use crate::extradata::Av1CodecConfig;
use crate::frame_header::{parse_frame_header, parse_frame_obu, FrameHeader};
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
    fn ingest(&mut self, data: &[u8]) -> Result<()> {
        for obu in iter_obus(data) {
            let obu = obu?;
            match obu.header.obu_type {
                ObuType::TemporalDelimiter | ObuType::Padding => {
                    // Empty / ignored.
                }
                ObuType::SequenceHeader => {
                    self.seq_header = Some(parse_sequence_header(obu.payload)?);
                }
                ObuType::FrameHeader | ObuType::RedundantFrameHeader => {
                    let seq = self.seq_header.as_ref().ok_or_else(|| {
                        Error::invalid("av1: frame_header before sequence_header")
                    })?;
                    let fh = parse_frame_header(seq, obu.payload)?;
                    self.last_frame_header = Some(fh);
                    self.seen_frame = true;
                }
                ObuType::Frame => {
                    let seq = self
                        .seq_header
                        .as_ref()
                        .ok_or_else(|| Error::invalid("av1: frame_obu before sequence_header"))?
                        .clone();
                    // OBU_FRAME = frame_header_obu() + byte_alignment() +
                    // tile_group_obu(). `parse_frame_obu` returns both the
                    // header and the remaining tile-group payload slice.
                    let (fh, tg_payload) = parse_frame_obu(&seq, obu.payload)?;
                    self.seen_frame = true;
                    // Split the tile_group_obu into per-tile byte ranges so
                    // callers can observe how many tiles the frame carries
                    // and where each one lives.
                    let tile_decode_err = if let Some(ti) = fh.tile_info.as_ref() {
                        let tgh = parse_tile_group_header(tg_payload, ti)?;
                        let mut tiles = split_tile_payloads(tg_payload, ti, &tgh)?;
                        // Re-base offsets from the tile_group payload to the
                        // enclosing OBU_FRAME payload for easier downstream
                        // consumption (first-tile offset = frame-header +
                        // tg-header length).
                        let frame_header_len = obu.payload.len() - tg_payload.len();
                        for t in &mut tiles {
                            t.offset += frame_header_len;
                        }
                        self.last_tile_payloads = tiles;
                        // Walk every tile through the Phase 2 mode decoder.
                        // For a well-formed intra frame this reads every
                        // partition + mode symbol and exits with
                        // `Error::Unsupported` at the first non-skip leaf
                        // (§5.11.39 coefficient decode pending).
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
                        decode_tile_group(&seq, &fh, tg_payload, &mut fs)
                    } else {
                        self.last_tile_payloads.clear();
                        Err(tile_decode_unsupported())
                    };
                    self.last_frame_header = Some(fh);
                    // Surface whatever the tile walker returned — for Phase
                    // 2 this is `Error::Unsupported("av1 coefficient decode
                    // pending (§5.11.39)")` on the first non-skip leaf of
                    // any real encoded frame. Fully-skip frames would
                    // succeed here, but that virtually never happens in
                    // practice.
                    return match tile_decode_err {
                        Ok(()) => Ok(()),
                        Err(e) => Err(e),
                    };
                }
                ObuType::TileGroup => {
                    // The frame header for this OBU_TILE_GROUP must have
                    // been supplied by a preceding OBU_FRAME_HEADER. If we
                    // have it (and therefore the tile geometry) we split
                    // per-tile boundaries; if not we surface the historical
                    // message.
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
                    return decode_tile_group(seq, fh, obu.payload, &mut fs);
                }
                ObuType::Metadata | ObuType::TileList => {
                    // Metadata is informational; tile_list is for large-scale
                    // tile coding which we don't decode.
                }
                _ => {
                    // Reserved — ignore.
                }
            }
        }
        Ok(())
    }
}

impl Decoder for Av1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        // Drain ingest errors but don't surface them past the first frame —
        // many callers want to inspect headers even when we can't decode.
        match self.ingest(&packet.data) {
            Ok(()) => Ok(()),
            Err(Error::Unsupported(s)) => {
                // Headers parsed; tile body unsupported. Surface so the caller
                // knows there'll never be frames.
                Err(Error::Unsupported(s))
            }
            Err(e) => Err(e),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        Err(tile_decode_unsupported())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        // Scaffold — parses headers but never produces pixels, so the
        // only cross-packet state worth wiping is the cached last frame
        // header and the last ingest error. The sequence header is
        // stream-level config (mirrors what's in av1C extradata) and is
        // preserved so post-seek frame headers can still be parsed
        // without waiting for another sequence OBU.
        self.last_frame_header = None;
        self.last_error = None;
        self.seen_frame = false;
        self.last_tile_payloads.clear();
        Ok(())
    }
}
