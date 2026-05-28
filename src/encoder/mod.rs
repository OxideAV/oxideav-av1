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
//!     `c == 0` forward-scan S()). Arc 7 (round 213) extends with the
//!     per-coefficient base-level chain: `write_coeff_base_eob` (the
//!     3-symbol §9.4 alphabet at `c == eob - 1`), `write_coeff_base`
//!     (the 4-symbol alphabet at non-EOB positions) and `write_coeff_br`
//!     (one `BR_CDF_SIZE`-symbol §9.4 alphabet S() per `coeff_br` chain
//!     iteration, capped at `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) = 4`
//!     repetitions by the spec's `if (coeff_br < BR_CDF_SIZE - 1) break`
//!     guard). Same stateless surface as `block_mode_info`; the §8.3.2
//!     ctx values are caller-supplied — the existing decoder helpers
//!     [`crate::cdf::get_coeff_base_ctx`] /
//!     [`crate::cdf::get_coeff_base_eob_ctx`] /
//!     [`crate::cdf::get_br_ctx`] derive them from the running `Quant[]`
//!     array on both sides.
//!
//! Arc 8 (round 214) landed the `golomb_length_bit` / `golomb_data_bit`
//! magnitude tail (§5.11.39 lines 84-93) for coefficient magnitudes
//! above `NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`, with the
//! §6.10.34 `length <= 20` conformance bound enforced as a caller-bug
//! reject.
//!
//! Arc 10 (round 216) lands the §5.11.4 [`partition`] decision-tree
//! **symbol writer**: the inverse of the `partition` / `split_or_horz`
//! / `split_or_vert` S() reads inside
//! [`crate::cdf::PartitionWalker::decode_partition`]. Encoder drivers
//! pick a partition ordinal from their RD search, call
//! [`partition::write_partition`] with the chosen partition + the same
//! (`has_rows`, `has_cols`, `ctx`) the decoder will derive on its
//! recursive walk, then recurse on the appropriate `subSize` children.
//! Two predicate helpers ([`partition::partition_none_only`] /
//! [`partition::partition_split_only`]) surface the §5.11.4 first /
//! last conditional so the driver knows when to skip the writer call.
//!
//! Arc 11 (round 217) lands the §5.11.4 recursive **dispatch driver**:
//! [`partition_tree::write_partition_tree`] composes the r211–r216 per-block
//! writers (`write_skip`, `write_intra_segment_id`, `write_y_mode`,
//! `write_intra_uv_mode`, per-plane `write_coefficients`) together with the
//! r216 `write_partition` symbol writer into a complete intra-arm
//! partition-tree walker driven from a caller-supplied
//! [`partition_tree::EncodeNode`] tree. The driver maintains its own
//! `MiSizes[]` grid so the §8.3.2 `partition_ctx_for` lookup observes the
//! same neighbour widths the decoder's parallel
//! [`crate::cdf::PartitionWalker`] observes. Round-trips a leaf or 7-leaf
//! two-level split tree back through `decode_partition` plus manual
//! `decode_block` replay; the encoder is now a true encoder end-to-end
//! for the intra-only path.
//!
//! Arc 12 (round 218) lands the §5.11.36 transform_tree / tx_size
//! **writers**: [`transform_tree::write_block_tx_size`] (the §5.11.15
//! `tx_depth` symbol for the §5.11.16 `else` arm — inverse of
//! [`crate::cdf::PartitionWalker::read_block_tx_size`]) and
//! [`transform_tree::write_var_tx_size`] (the §5.11.17 recursive
//! `txfm_split` chain — inverse of
//! [`crate::cdf::PartitionWalker::read_var_tx_size`]). The
//! variable-transform writer takes a caller-supplied
//! [`transform_tree::VarTxNode`] tree describing the desired
//! `(txfm_split, sub_tx_size)` decisions per node, mirroring the same
//! Leaf/Split shape already used for the §5.11.4 `partition_tree`
//! dispatch.
//!
//! Arc 13 (round 219) bootstraps the **pixel-space encoder** with the
//! forward 4×4 DCT primitive in [`forward_transform`]: 1D
//! [`forward_transform::forward_dct_4`] and 2D
//! [`forward_transform::forward_dct_4x4`]. The kernel is the matrix
//! transpose of the §7.13.2.3 inverse DCT-4 reproduced in
//! [`crate::transform::inverse_dct`] (`n = 2` branch). Round-trip
//! lockstep against the inverse confirms `M^T · M ≈ 2 · I` (exactly
//! diagonal; ≈ 1.999 on even rows and ≈ 2.072 on odd rows because the
//! AV1 cosine constants are integer-rounded approximations of the
//! analytic values). The off-diagonal entries are exactly zero — the
//! basis is mutually orthogonal. This primitive is the bridge between
//! the arc-1..12 syntax-only encoder (consumes pre-decided `Quant[]`)
//! and a real encoder that takes pixel residuals as input.
//!
//! Next arc: forward DCT for sizes 8 / 16 / 32 / 64; forward ADST /
//! FLIPADST / WHT / IDTX; quantization primitive; full pixel-space
//! encoder driver assembling the forward kernels with quant + the
//! r211–r218 syntax writers. §5.11.18 inter-arm `mode_info()`
//! dispatcher; intra angle / palette encode. §5.9.7
//! `frame_size_with_refs()` inverse + §5.9.24 `read_global_param`
//! signed-subexp inverse for the remaining inter-frame paths.

pub mod bitwriter;
pub mod block_mode_info;
pub mod coefficients;
pub mod forward_transform;
pub mod frame_obu;
pub mod ivf;
pub mod obu;
pub mod partition;
pub mod partition_tree;
pub mod sequence_obu;
pub mod symbol_writer;
pub mod temporal_unit;
pub mod tile_group_obu;
pub mod transform_tree;

pub use bitwriter::BitWriter;
pub use block_mode_info::{
    write_intra_frame_y_mode, write_intra_segment_id, write_intra_uv_mode, write_skip, write_y_mode,
};
pub use coefficients::{
    write_coeff_base, write_coeff_base_eob, write_coeff_br, write_coefficients, write_dc_sign,
    write_eob_pt, write_golomb, write_txb_skip, GOLOMB_MAX_LENGTH,
};
pub use forward_transform::{forward_dct_4, forward_dct_4x4};
pub use frame_obu::write_frame_header_obu;
pub use ivf::IvfWriter;
pub use obu::{
    build_temporal_unit, obu_type_takes_trailing_bits, write_obu_with_size, write_temporal_unit,
    ObuExtensionHeader, ObuFrame, ObuHeader, ObuWriter,
};
pub use partition::{partition_none_only, partition_split_only, write_partition};
pub use partition_tree::{
    write_partition_tree, EncodeBlock, EncodeNode, PartitionTreeWriter, PlaneCoefficients,
};
pub use sequence_obu::write_sequence_header_obu;
pub use symbol_writer::SymbolWriter;
pub use temporal_unit::{encode_sequence_header_obu, encode_temporal_unit, TemporalUnitPlan};
pub use tile_group_obu::{
    parse_tile_group_obu_body, write_tile_group_obu, ParsedTileGroup, TileGroupObu,
    TileGroupObuWriter, TilePayload,
};
pub use transform_tree::{write_block_tx_size, write_var_tx_size, VarTxNode, VarTxNodeKind};
