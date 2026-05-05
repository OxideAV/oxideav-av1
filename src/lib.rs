//! AV1 (AOMedia Video 1) — pure-Rust decoder for oxideav.
//!
//! Phase 7 scope (this revision): OBU walk + sequence/frame headers +
//! tile partition walk + mode decode + coefficient decode +
//! dequantisation + the full 4/8/16/32/64-point inverse transform
//! kernel set + every intra predictor (DC/V/H + 6 directional + 3
//! smooth + Paeth + filter-intra + CFL) in both 8-bit and 10/12-bit
//! (native u16) variants + per-frame deblocking (narrow 4-tap) +
//! CDEF (direction search + primary/secondary filter) + loop
//! restoration (Wiener + SGR with full §5.11.40-.44 per-unit
//! signalling) + film grain synthesis (spec §7.20.2 32×32 tiler on
//! top of AR-shaped 73×73 luma / 38×38 chroma templates) + the
//! single-reference translational **inter** path (§7.11.3, 8-tap
//! sub-pel filters, eighth-pel MV decode, single-ref LAST only).
//! The decoder produces pixel-accurate AVIF still-image output and
//! handles the key+inter frame cadence used by AVIS (animated AVIF)
//! image sequences.
//!
//! Still deferred:
//!
//! * Compound inter prediction, global motion, warped motion, OBMC,
//!   inter-intra (§7.11.4-.8).
//! * Multi-reference DPB — the decoder tracks only the immediately
//!   preceding reconstruction as LAST.
//! * Wide deblocking (8/14-tap) drivers — only the narrow filter is
//!   currently invoked.
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
pub mod dpb;
pub mod encoder;
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

use oxideav_core::{CodecCapabilities, CodecId, CodecTag};
use oxideav_core::{CodecInfo, CodecRegistry, RuntimeContext};

pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 decoder factory with a codec registry.
///
/// The implementation declares `av1_sw_decode` to mark it as an
/// in-progress full AV1 decoder: every Phase 6 surface (intra +
/// deblock + CDEF + LR + film grain) plus Phase 7 (single-ref
/// translational inter) is wired. Still-image AVIF and 2-frame-plus
/// AVIS sequences decode end-to-end in 8-bit; HBD follows the same
/// path but is narrowed to u8 for the emitted VideoFrame until a
/// u16-capable PixelFormat variant lands upstream.
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("av1_sw_decode")
        .with_lossy(true)
        .with_max_size(16384, 16384);
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(decoder::make_decoder)
            .encoder(encoder::make_encoder)
            .tag(CodecTag::fourcc(b"AV01")),
    );
}

/// Unified registration entry point: install the AV1 codec factories
/// into the codec sub-registry of a [`RuntimeContext`].
///
/// This is the preferred entry point for new code — it matches the
/// convention every sibling crate now follows. Direct callers that need
/// only the codec sub-registry can keep using [`register_codecs`].
///
/// Also auto-registered into [`oxideav_core::REGISTRARS`] via the
/// [`oxideav_core::register!`] macro below so consumers calling
/// [`oxideav_core::RuntimeContext::with_all_features`] pick AV1 up
/// without any explicit umbrella plumbing.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

oxideav_core::register!("av1", register);

pub use decoder::{make_decoder, Av1Decoder};
pub use encoder::{make_encoder, write_keyframe_stream, Av1Encoder, FrameConfig, SequenceConfig};
pub use extradata::Av1CodecConfig;
pub use frame_header::{parse_frame_header, parse_frame_obu, FrameHeader, FrameType, ParseDepth};
pub use obu::{iter_obus, parse_config_obus, parse_obu_header, read_obu, Obu, ObuHeader, ObuType};
pub use sequence_header::{
    parse_sequence_header, ColorConfig, DecoderModelInfo, OperatingPoint, SequenceHeader,
    TimingInfo,
};
pub use tile_group::{parse_tile_group_header, split_tile_payloads, TileGroupHeader, TilePayload};
pub use tile_info::{mi_cols_rows, parse_tile_info, tile_log2, TileInfo};

#[cfg(test)]
mod register_tests {
    use super::*;
    use oxideav_core::{CodecId, CodecParameters, RuntimeContext};

    #[test]
    fn register_via_runtime_context_installs_codec_factory() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        let dec = ctx
            .codecs
            .first_decoder(&params)
            .expect("av1 decoder factory");
        assert_eq!(dec.codec_id().as_str(), CODEC_ID_STR);
    }
}
