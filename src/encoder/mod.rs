//! Encoder side of the crate.
//!
//! Arc 1 (round 206) landed the bit-output plumbing. Arc 2 (round
//! 207) lands the `frame_header_obu()` writer on top.
//!
//! Layers:
//!
//!   * [`bitwriter::BitWriter`] — MSB-first bit-output buffer, the
//!     inverse of [`crate::bitreader::BitReader`] (§8.1 `read_bit`),
//!     plus `write_leb128()` (§4.10.5), `write_uvlc()` (§4.10.3),
//!     `write_su(n)` (§4.10.6), and `write_ns(n)` (§4.10.7) — the
//!     full descriptor-inverse set the §5.5 / §5.9 writers need.
//!
//!   * [`obu`] — Open Bitstream Unit framer per §5.3. Writes the
//!     §5.3.2 one-byte `obu_header`, the optional §5.3.3
//!     `obu_extension_header`, and the optional `leb128()`
//!     `obu_size` size field for the §5.2 low-overhead bytestream
//!     format. Concatenation of multiple OBUs into a temporal unit
//!     is byte-aligned and simply uses [`ObuWriter::write`] N times.
//!
//!   * [`sequence_obu`] — `sequence_header_obu()` writer per §5.5.1
//!     (with §5.5.2 `color_config`, §5.5.3 `timing_info`, §5.5.4
//!     `decoder_model_info`, §5.5.5 `operating_parameters_info`).
//!     The inverse of [`crate::sequence_header::parse_sequence_header`].
//!     Reuses the same [`crate::sequence_header::SequenceHeader`]
//!     struct as the source-of-truth descriptor, so a written
//!     payload immediately round-trips through the parser.
//!
//!   * [`frame_obu`] — `frame_header_obu()` writer per §5.9.1 /
//!     §5.9.2 plus every sub-procedure §5.9.2 calls into. The
//!     inverse of [`crate::frame_header::parse_frame_header`] on the
//!     intra / show-existing-frame / reduced-still paths and on the
//!     inter shared tail above `disable_frame_end_update_cdf`.
//!     Reuses the parser's [`crate::frame_header::FrameHeader`] as
//!     source-of-truth descriptor.
//!
//!   * [`ivf`] — IVF v0 container writer (32-byte file header + 12-
//!     byte per-frame header) for shipping the encoded OBU temporal
//!     units into a playable file. IVF is a trivial public file
//!     format developed for VP8 testing; the byte layout used here
//!     matches the `.ivf` fixtures already in `docs/video/av1/
//!     fixtures/`.
//!
//!   * [`temporal_unit`] — arc 3 (round 208) glue. Wraps the per-OBU
//!     body writers above with the §5.3.4 `trailing_bits()` trailer
//!     and the §5.3.1 `obu_size` size field, then aggregates a
//!     sequence of OBUs into a §7.5 temporal unit (TD prefix +
//!     optional SH + the frame OBUs). The product is a complete
//!     byte-aligned bytestream a downstream parser walks back via
//!     [`crate::obu::ObuIter`].
//!
//!   * [`tile_group_obu`] — arc 4 (round 210) §5.11.1 framing
//!     skeleton on top of the r209 [`symbol_writer`]. Builds the
//!     §5.11.1 `tile_group_obu` body around a caller-supplied
//!     `Vec<TilePayload>` (each `TilePayload` is a finished
//!     `SymbolWriter::finish()` byte run): writes
//!     `tile_start_and_end_present_flag` / `tg_start` / `tg_end`,
//!     byte-aligns, then per-tile `tile_size_minus_1`
//!     (`le(TileSizeBytes)`) + tile bytes for every non-last tile.
//!     The body is the byte-aligned payload `write_obu_with_size`
//!     wraps in an `OBU_TILE_GROUP` (which §5.3.1 explicitly
//!     excludes from the §5.3.4 trailer).
//!
//! Next arc: per-block §5.11 syntax — `skip` / `mode_info` and the
//! first §5.11.39 coefficient-encode primitives so emitted tile
//! payloads decode to real pixel data, plus the §5.11.4 partition
//! decision tree. §5.9.7 `frame_size_with_refs()` inverse + §5.9.24
//! `read_global_param` signed-subexp inverse for the remaining
//! inter-frame paths.

pub mod bitwriter;
pub mod frame_obu;
pub mod ivf;
pub mod obu;
pub mod sequence_obu;
pub mod symbol_writer;
pub mod temporal_unit;
pub mod tile_group_obu;

pub use bitwriter::BitWriter;
pub use frame_obu::write_frame_header_obu;
pub use ivf::IvfWriter;
pub use obu::{
    build_temporal_unit, obu_type_takes_trailing_bits, write_obu_with_size, write_temporal_unit,
    ObuExtensionHeader, ObuFrame, ObuHeader, ObuWriter,
};
pub use sequence_obu::write_sequence_header_obu;
pub use symbol_writer::SymbolWriter;
pub use temporal_unit::{encode_sequence_header_obu, encode_temporal_unit, TemporalUnitPlan};
pub use tile_group_obu::{
    parse_tile_group_obu_body, write_tile_group_obu, ParsedTileGroup, TileGroupObu,
    TileGroupObuWriter, TilePayload,
};
