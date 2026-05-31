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
//! diagonal; `≈ 1.99988` on even rows and `≈ 1.99994` on odd rows
//! because the AV1 cosine constants are integer-rounded approximations
//! of the analytic values). The off-diagonal entries are exactly zero
//! — the basis is mutually orthogonal. This primitive is the bridge
//! between the arc-1..12 syntax-only encoder (consumes pre-decided
//! `Quant[]`) and a real encoder that takes pixel residuals as input.
//!
//! Arc 14 (round 220) lands the **forward quantization primitive** —
//! [`forward_quantize::forward_quantize`], the encoder counterpart of
//! [`crate::cdf::dequantize_step1`]. Consumes a post-forward-transform
//! coefficient buffer, the per-frame [`crate::cdf::QuantizerParams`],
//! and the per-block plane / segment / tx-type / qm-level selectors;
//! returns the dense `Tx_Width * Tx_Height` `Quant[]` array the
//! §5.11.39 `coefficients()` writer consumes. Round-half-away-from-
//! zero inversion of the spec's `(|dq| & 0xFF_FFFF) / dqDenom`
//! truncating divide; bit-exact roundtrip on the lossless
//! `q_index = 0` lattice and within one quantization step elsewhere.
//!
//! Arc 17 (round 223) extends the pixel driver to **4:2:0 YUV input**
//! via [`pixel_driver::encode_intra_frame_yuv`]. The chroma walk mirrors
//! the luma side: DC_PRED from the running reconstructed chroma plane,
//! forward WHT (lossless arm), forward quantize, write_coefficients per
//! plane. Per §5.11.5 `HasChroma` at 4:2:0 / BLOCK_4X4 the chroma
//! coefficient pass fires only on luma cells `(1,1), (1,3), (3,1),
//! (3,3)` (the SE corner of each 8×8 luma quadrant), so each chroma
//! 4×4 block is emitted exactly once. The same `(SequenceHeader,
//! FrameHeader)` pair feeds both Y-only and YUV entry points — the
//! tiny-i-only-16x16-prof0 fixture is `monochrome = false` /
//! subsampling_x = subsampling_y = 1 already. Every chroma plane
//! round-trips pixel-for-pixel at `base_q_idx = 0` on arbitrary
//! inputs (lossless WHT chain).
//!
//! Arc 16 (round 222) lands the **forward 4×4 Walsh-Hadamard transform**
//! in [`forward_wht`]: 1D [`forward_wht::forward_wht4`] and 2D
//! [`forward_wht::forward_wht_4x4`]. Derived clean-room by inverting
//! the §7.13.2.10 inverse-WHT body algebraically (the four output cells
//! determine `A = a + b` and `D = d - c` uniquely; the `(A - D) >> 1`
//! halving is shared between forward and inverse so the round-trip is
//! exact regardless of parity). Unlike the integer-approximated DCT,
//! the WHT is a **pure integer butterfly** ⇒ **bit-exact round-trip
//! for any integer input**. [`pixel_driver::encode_intra_frame_y`] now
//! routes through the forward WHT on the §5.9.2 `CodedLossless` arm
//! (`base_q_idx == 0 && DeltaQ?? all zero`), unlocking pixel-perfect
//! roundtrip on arbitrary inputs at `base_q_idx = 0`.
//!
//! Arc 19 (round 226) lands the **forward ADST / FLIPADST / IDTX**
//! primitives in [`forward_adst`] and [`forward_identity`]. Same
//! matrix-cache recipe as r225's forward DCT: the §7.13.2.6/7/8
//! inverse ADST is a fixed integer linear map for `n in 2..=4` and
//! the §7.13.2.11..§7.13.2.14 inverse identity is a per-cell scalar
//! map for `n in 2..=5`; the forward kernels are the algebraic
//! transposes (matrix transpose for ADST, scalar transpose-=-self
//! for the diagonal IDTX). FLIPADST reuses the ADST kernel and
//! applies the involutory reverse on the residual input — per
//! §7.13.3 the flip is purely a coordinate transform on the frame-
//! buffer write, NOT a separate butterfly. This rounds out the menu
//! of forward 1D/2D primitives the encoder needs to consider when
//! picking `tx_type` per block; the §7.13.3-equivalent forward 2D
//! dispatcher with row-/col-shift envelope is the next arc.
//!
//! Arc 20 (round 227) lands the **§7.13.3-equivalent forward 2D
//! transform dispatcher** in [`forward_transform_2d`]:
//! [`forward_transform_2d::forward_transform_2d`] consumes a row-
//! major spatial residual buffer + `(tx_size, plane_tx_type,
//! lossless)` and returns the row-major coefficient buffer the
//! decoder's [`crate::transform::inverse_transform_2d`] would
//! consume. Composes the r219/r222/r225/r226 per-axis forward
//! kernels (DCT / ADST / FLIPADST / IDTX / WHT) into the §7.13.3
//! column-then-row pipeline (transpose of the decoder's row-then-
//! column composition), with the §7.13.3 `colShift` / `rowShift`
//! shift envelope applied as **pre-scales** (`<< colShift` before
//! the column pass, `<< rowShift` before the row pass) so the
//! decoder's `Round2(_, shift)` post-scales cancel exactly. Square
//! sizes only this arc (`TX_4X4`, `TX_8X8`, `TX_16X16`, `TX_32X32`,
//! `TX_64X64`); rectangular sizes (`TX_4X8`, …, `TX_64X16`) deferred
//! to a subsequent arc.
//!
//! Round-trip validation: lossless `TX_4X4` arm is **bit-exact** on
//! arbitrary integer inputs (WHT chain). Lossy arms recover the
//! input scaled by the analytic per-axis kernel norms (DCT-N ≈ N,
//! ADST-N ≈ N/2, IDTX with §7.13.2.11..14 scalars) within a
//! conservative per-cell rounding bound. FLIPADST family pre-flips
//! the spatial residual along the appropriate axis (§7.12.3 step-3
//! mirror) before the plain ADST kernel runs.
//!
//! Next arc: pixel-driver chroma path (needs frame-header build with
//! `monochrome = false`); rectangular block sizes (`TX_4X8` /
//! `TX_8X16` / … / `TX_32X64`) for [`forward_transform_2d`];
//! standalone `decode_av1` entry that wires the existing decoder
//! modules into a full pipeline. §5.11.18 inter-arm `mode_info()`
//! dispatcher; intra angle / palette encode. §5.9.7
//! `frame_size_with_refs()` inverse + §5.9.24 `read_global_param`
//! signed-subexp inverse for the remaining inter-frame paths.

pub mod bitwriter;
pub mod block_mode_info;
pub mod coefficients;
pub mod forward_adst;
pub mod forward_identity;
pub mod forward_quantize;
pub mod forward_transform;
pub mod forward_transform_2d;
pub mod forward_wht;
pub mod frame_obu;
pub mod ivf;
pub mod obu;
pub mod partition;
pub mod partition_tree;
pub mod pixel_driver;
pub mod pixel_driver_dyn;
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
pub use forward_adst::{
    forward_adst_16, forward_adst_16x16, forward_adst_4, forward_adst_4x4, forward_adst_8,
    forward_adst_8x8, forward_flipadst_16, forward_flipadst_16x16, forward_flipadst_4,
    forward_flipadst_4x4, forward_flipadst_8, forward_flipadst_8x8,
};
pub use forward_identity::{
    forward_idtx_16, forward_idtx_16x16, forward_idtx_32, forward_idtx_32x32, forward_idtx_4,
    forward_idtx_4x4, forward_idtx_8, forward_idtx_8x8,
};
pub use forward_quantize::forward_quantize;
pub use forward_transform::{
    forward_dct_16, forward_dct_16x16, forward_dct_32, forward_dct_32x32, forward_dct_4,
    forward_dct_4x4, forward_dct_64, forward_dct_64x64, forward_dct_8, forward_dct_8x8,
};
pub use forward_transform_2d::forward_transform_2d;
pub use forward_wht::{forward_wht4, forward_wht_4x4};
pub use frame_obu::write_frame_header_obu;
pub use ivf::{parse_file_header, IvfFileHeader, IvfFrame, IvfReadError, IvfReader, IvfWriter};
pub use obu::{
    build_temporal_unit, obu_type_takes_trailing_bits, write_obu_with_size, write_temporal_unit,
    ObuExtensionHeader, ObuFrame, ObuHeader, ObuWriter,
};
pub use partition::{partition_none_only, partition_split_only, write_partition};
pub use partition_tree::{
    write_partition_tree, EncodeBlock, EncodeNode, PartitionTreeWriter, PlaneCoefficients,
};
pub use pixel_driver::{
    dispatch_order_cells, encode_intra_frame_y, encode_intra_frame_yuv, CellCoord, EncodedFrame,
    EncodedFrameYuv, Yuv420Frame16x16, CELLS_HIGH, CELLS_WIDE, CHROMA_CELLS_HIGH,
    CHROMA_CELLS_WIDE, CHROMA_HEIGHT, CHROMA_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, MI_COLS, MI_ROWS,
};
pub use pixel_driver_dyn::{
    build_intra_only_yuv420_8bit_fh, build_intra_only_yuv420_8bit_fh_with_q,
    build_intra_only_yuv420_8bit_seq, dispatch_order_leaves, encode_intra_frame_yuv_dyn,
    encode_intra_frame_yuv_dyn_with_q, root_super_block, EncodedFrameDyn, Yuv420Frame, MAX_DIM,
    MIN_DIM,
};
pub use sequence_obu::write_sequence_header_obu;
pub use symbol_writer::SymbolWriter;
pub use temporal_unit::{encode_sequence_header_obu, encode_temporal_unit, TemporalUnitPlan};
pub use tile_group_obu::{
    parse_tile_group_obu_body, write_tile_group_obu, ParsedTileGroup, TileGroupObu,
    TileGroupObuWriter, TilePayload,
};
pub use transform_tree::{write_block_tx_size, write_var_tx_size, VarTxNode, VarTxNodeKind};
