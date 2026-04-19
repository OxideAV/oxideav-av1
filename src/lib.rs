//! AV1 (AOMedia Video 1) — pure-Rust decoder for oxideav.
//!
//! Phase 6 scope (this revision): OBU walk + sequence/frame headers +
//! tile partition walk + mode decode + coefficient decode +
//! dequantisation + the full 4/8/16/32/64-point inverse transform
//! kernel set + every intra predictor (DC/V/H + 6 directional + 3
//! smooth + Paeth + filter-intra + CFL) in both 8-bit and 10/12-bit
//! (native u16) variants + per-frame deblocking (narrow 4-tap) +
//! CDEF (direction search + primary/secondary filter) + **loop
//! restoration** (Wiener + SGR with full §5.11.40-.44 per-unit
//! signalling) + **film grain synthesis** (spec §7.20.2 32×32 tiler
//! on top of AR-shaped 73×73 luma / 38×38 chroma templates). The
//! decoder produces pixel-accurate AVIF still-image output for
//! intra-only clips including LR + film-grain combinations.
//!
//! Still deferred:
//!
//! * Inter prediction (§7.11.3, Phase 7).
//! * Wide deblocking (8/14-tap) drivers — only the narrow filter is
//!   currently invoked; wider widths are applied per-edge-class by a
//!   future pass.
//! * Quantisation matrices (§5.9.12).
//!
//! Every error message includes the precise §ref so callers can see
//! exactly where the decoder stopped.
//!
//! Spec references throughout follow the **AV1 Bitstream & Decoding Process
//! Specification (2019-01-08)**: <https://aomediacodec.github.io/av1-spec/av1-spec.pdf>.

pub mod bitreader;
pub mod cdef;
pub mod cdfs;
pub mod decode;
pub mod decoder;
pub mod extradata;
pub mod filmgrain;
pub mod frame_header;
pub mod frame_header_tail;
pub mod loopfilter;
pub mod lr;
pub mod obu;
pub mod predict;
pub mod quant;
pub mod sequence_header;
pub mod symbol;
pub mod tile_group;
pub mod tile_info;
pub mod transform;

/// Compat re-export — `predict::intra` carries the full Phase-5 intra
/// predictor set. The pre-Phase-5 `crate::intra::{IntraMode,
/// Neighbours, predict}` API is re-exposed here with the same shape,
/// plus new variants that map to the decoder's mode taxonomy.
pub mod intra {
    pub use crate::predict::intra::{predict, IntraMode, Neighbours};
}

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 decoder factory with a codec registry.
///
/// The implementation declares `av1_sw_decode_still` to make the build
/// visible in `oxideav list`-style output: every Phase 6 surface is
/// wired — full intra predictor set + deblocking + CDEF + loop
/// restoration (Wiener + SGR with per-unit signal decode) + film
/// grain synthesis on top of the Phase 1-4 header/OBU/coefficient
/// pipeline. Intra-only AVIF stills should decode end-to-end in
/// 8-bit and 10/12-bit HBD, with or without LR / film-grain.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("av1_sw_decode_still")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .tag(CodecTag::fourcc(b"AV01")),
    );
}

pub use decoder::{make_decoder, Av1Decoder};
pub use extradata::Av1CodecConfig;
pub use frame_header::{parse_frame_header, parse_frame_obu, FrameHeader, FrameType, ParseDepth};
pub use obu::{iter_obus, parse_config_obus, parse_obu_header, read_obu, Obu, ObuHeader, ObuType};
pub use sequence_header::{
    parse_sequence_header, ColorConfig, DecoderModelInfo, OperatingPoint, SequenceHeader,
    TimingInfo,
};
pub use tile_group::{parse_tile_group_header, split_tile_payloads, TileGroupHeader, TilePayload};
pub use tile_info::{mi_cols_rows, parse_tile_info, tile_log2, TileInfo};
