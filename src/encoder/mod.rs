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
//!   * [`block_mode_info`] — arc 5 (round 211) per-block §5.11 syntax
//!     writers, intra arm only: `write_skip` (§5.11.11),
//!     `write_intra_segment_id` (§5.11.8 + §5.11.9), `write_intra_frame_y_mode`
//!     (§5.11.7 line 13 with the §8.3.2 neighbour-CDF ctx),
//!     `write_y_mode` (§5.11.22 line 3 with the `Size_Group[ MiSize ]`
//!     ctx), and `write_intra_uv_mode` (§5.11.22 line 6 with the
//!     §8.3.2 CFL-allowed selector). Pure stateless: ctx is
//!     caller-supplied (mirroring [`SymbolWriter::write_symbol`]'s
//!     caller-supplied CDF slice pattern); round-trip tests drive the
//!     output back through the matching `PartitionWalker::decode_*`
//!     methods.
//!
//!   * [`coefficients`] — arc 6 (round 212) first slice of the §5.11.39
//!     `coefficients()` writers: `write_txb_skip` (the `all_zero` S()),
//!     `write_eob_pt` (eob_pt_{16..1024} S() + eob_extra S() +
//!     eob_extra_bit L(1) refinement loop) and `write_dc_sign` (the
//!     `c == 0` forward-scan S()). Same stateless surface as
//!     `block_mode_info`; the §8.3.2 ctx values are caller-supplied.
//!
//! Next arc: per-coefficient `coeff_base{_eob}` + `coeff_br` writers
//! (the four- / three-symbol §9.4 alphabets plus the §8.3.2
//! `get_coeff_base_ctx` / `get_coeff_base_eob_ctx` / `get_br_ctx`
//! plumbing) and the `golomb_length_bit` / `golomb_data_bit` tail for
//! coefficient magnitudes above `NUM_BASE_LEVELS + COEFF_BASE_RANGE`,
//! followed by the §5.11.39 driver loop and the §5.11.4 partition
//! decision-tree writer; inter-arm mode_info writers (§5.11.18
//! dispatcher composite). §5.9.7 `frame_size_with_refs()` inverse +
//! §5.9.24 `read_global_param` signed-subexp inverse for the remaining
//! inter-frame paths.

pub mod bitwriter;
pub mod block_mode_info;
pub mod coefficients;
pub mod frame_obu;
pub mod ivf;
pub mod obu;
pub mod sequence_obu;
pub mod symbol_writer;
pub mod temporal_unit;
pub mod tile_group_obu;

pub use bitwriter::BitWriter;
pub use block_mode_info::{
    write_intra_frame_y_mode, write_intra_segment_id, write_intra_uv_mode, write_skip, write_y_mode,
};
pub use coefficients::{write_dc_sign, write_eob_pt, write_txb_skip};
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
