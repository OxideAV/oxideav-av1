//! AV1 (AOMedia Video 1) — pure-Rust parse-only crate for oxideav.
//!
//! Status: **parses every OBU up to and including `tile_info()` plus the
//! `tile_group_obu()` framing header**, extracts per-tile byte boundaries,
//! and stops before pixel reconstruction. Pixel decode is the bulk of an
//! AV1 decoder (~20 KLOC of CDF + transforms + intra/inter prediction +
//! deblock + CDEF + LR) and lands incrementally. What this crate does
//! today:
//!
//! * Walks the OBU stream (§5.3) and surfaces typed `Obu` values.
//! * Parses sequence header OBUs (§5.5) — full color config, operating
//!   points, all enable flags, optional timing / decoder model info.
//! * Parses frame header OBUs (§5.9) fully including `tile_info()`
//!   (§5.9.15) — frame type, dimensions (with superres), render size,
//!   intrabc, interpolation filter, ref frame indices, tile column / row
//!   boundaries, `TileSizeBytes`.
//! * Parses the `OBU_FRAME` payload as `frame_header_obu() +
//!   byte_alignment() + tile_group_obu()`, extracting per-tile byte
//!   ranges via `split_tile_payloads()` / `parse_tile_group_header()`.
//! * Parses the AV1CodecConfigurationRecord (`av1C`) used by MP4 and
//!   Matroska, including the embedded sequence-header config OBU.
//! * Registers a `Decoder` factory that ingests OBU streams and exposes
//!   header-level state via `Av1Decoder::sequence_header()` /
//!   `last_frame_header()` / `last_tile_payloads()`.
//! * Ships the primitives that pixel reconstruction will call:
//!     - Symbol (range / arithmetic) decoder (§4.10.4 + §9.3).
//!     - Inverse 4×4 and 8×8 DCT-DCT transforms (§7.7).
//!     - `DC_PRED` / `V_PRED` / `H_PRED` intra predictors (§7.11.2).
//!
//! `receive_frame()` returns `Error::Unsupported` because the following
//! clauses are still unimplemented:
//!
//! * Default CDF tables (§9.4.1 / §9.4.2).
//! * Partition quadtree walk + `decode_block()` (§5.11.4 – §5.11.14).
//! * Coefficient decode + dequantisation (§5.11.39 / §7.12).
//! * Transforms 16×16 / 32×32 / 64×64 plus the ADST / flipped-ADST /
//!   identity / WHT / mixed paths (§7.7).
//! * The other 10 intra prediction modes (directional, smooth, paeth)
//!   (§7.11.2).
//! * Inter prediction, compound modes (§7.11.3).
//! * Quantisation / segmentation / loop filter / CDEF / loop restoration
//!   / film grain (§5.9.16 – §5.9.22, §7.13 – §7.17, §7.20).
//!
//! Every error message includes the precise §ref so callers can see
//! exactly where the decoder stopped.
//!
//! Spec references throughout follow the **AV1 Bitstream & Decoding Process
//! Specification (2019-01-08)**: <https://aomediacodec.github.io/av1-spec/av1-spec.pdf>.

pub mod bitreader;
pub mod cdfs;
pub mod decoder;
pub mod extradata;
pub mod frame_header;
pub mod intra;
pub mod obu;
pub mod sequence_header;
pub mod symbol;
pub mod tile_decode;
pub mod tile_group;
pub mod tile_info;
pub mod transform;

use oxideav_codec::{CodecInfo, CodecRegistry};
use oxideav_core::{CodecCapabilities, CodecId, CodecTag};

pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 decoder factory with a codec registry.
///
/// The implementation declares `av1_sw_parse` to make it visible in
/// `oxideav list` style output that this is the parse-and-frame build —
/// headers, tile_info, and tile byte-boundaries are produced, but pixel
/// reconstruction is still out of scope.
pub fn register(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("av1_sw_parse")
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
