//! AV1 (AOMedia Video 1) — pure-Rust decoder for oxideav.
//!
//! Phase 3 scope (this revision): OBU walk + sequence/frame headers +
//! tile partition walk + mode decode + coefficient decode +
//! dequantisation + inverse transform (4/8/16 DCT+ADST) + clip-add
//! pixel reconstruction. A leaf block ≤ 16×16 with a decode-from-
//! neighbors intra predictor (DC / V / H) completes end-to-end.
//!
//! Still deferred:
//!
//! * Transform sizes 32/64 + WHT + mixed V/H-identity (Phase 4,
//!   §7.7 subclauses).
//! * TX splitting for leaf blocks > 16×16 (§5.11.27, Phase 4).
//! * The other 10 intra predictors — directional / smooth / paeth
//!   (§7.11.2, Phase 5). Currently those modes fall back to DC to
//!   keep the pipeline running; Phase 5 closes the fidelity gap.
//! * Inter prediction (§7.11.3, Phase 6).
//! * Segmentation Q overrides beyond SEG_LVL_ALT_Q; loop filter /
//!   CDEF / LR / film grain (§7.13 – §7.17, §7.20, Phase 7).
//! * Quantisation matrices (§5.9.12) — surfaces Unsupported when
//!   `using_qmatrix` is set in the bitstream.
//! * 10-/12-bit sample reconstruction (Phase 3 wires only 8-bit).
//!
//! Every error message includes the precise §ref so callers can see
//! exactly where the decoder stopped.
//!
//! Spec references throughout follow the **AV1 Bitstream & Decoding Process
//! Specification (2019-01-08)**: <https://aomediacodec.github.io/av1-spec/av1-spec.pdf>.

pub mod bitreader;
pub mod cdfs;
pub mod decode;
pub mod decoder;
pub mod extradata;
pub mod frame_header;
pub mod frame_header_tail;
pub mod intra;
pub mod obu;
pub mod quant;
pub mod sequence_header;
pub mod symbol;
pub mod tile_group;
pub mod tile_info;
pub mod transform;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 decoder factory with a codec registry.
///
/// The implementation declares `av1_sw_phase3` to make the build
/// visible in `oxideav list`-style output: headers + partition walk +
/// mode + coefficient decode + reconstruction for 4/8/16 TX blocks
/// land, but 32/64 TX + TX splitting + the full intra predictor set
/// are still deferred (see crate-level doc comment).
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("av1_sw_phase3")
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
