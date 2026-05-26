# Changelog

All notable changes to `oxideav-av1` are recorded here.

## [Unreleased]

### Added

* **Round 147 — §5.11.49 `palette_tokens` per-plane diagonal walker.**
  Lands the §5.11.49 palette-tokens walker (p.101–102) that drives the
  decoded `ColorMap{Y,UV}` from the `SymbolDecoder`-emitted
  `palette_color_idx_{y,uv}` symbols and the §5.11.50
  [`get_palette_color_context`] derivation landed in r146. Surface:
  `palette_tokens_plane(dec, tile_ctx, plane, palette_size, block_w,
  block_h, onscreen_w, onscreen_h, color_index_map, color_map, stride)
  -> Result<(), Error>`, with `PalettePlane::{Y, Uv}` picking the
  `palette_{y,uv}_color_cdf` family. The walker (a) seeds
  `ColorMap{Y,UV}[0][0] = color_index_map_{y,uv}`; (b) runs the
  anti-diagonal walk `for i in 1..H+W-1 { for j in min(i, W-1) ..=
  max(0, i - H + 1) }` decoding one `palette_color_idx_*` per
  `(i - j, j)` against the §5.11.50 colour-context-derived cdf row,
  writing `ColorMap[r][c] = ColorOrder[idx]`; (c) replicates the
  on-screen right edge (`onscreen_width - 1`) into columns
  `onscreen_width..block_width`; and (d) replicates the on-screen
  bottom row (`onscreen_height - 1`) across the full `block_width`
  into rows `onscreen_height..block_height`. The chroma-subsampled UV
  path is the same walker; the §5.11.49 `blockWidth >> subsampling_x`
  and `<4 -> +=2` adjustments are the caller's responsibility (they
  belong to the §5.11.49 outer-control flow, not the walker). Two new
  [`Error`] variants surface caller bugs (palette size out of range /
  buffer too small / on-screen exceeds block / `color_index_map` out
  of palette range) as [`Error::InvalidPaletteWalkArgs`] and the
  unreachable §5.11.50 hash slots (`0`, `1`, `3`, `4`) as
  [`Error::PaletteColorContextUnmapped`]. Tests grow by 11 (cdf
  module): every caller-bug pre-condition is rejected before any
  `read_symbol`; 2x2 on-screen / no border-fill walk writes every
  cell to a value `< palette_size` and preserves the (0,0) seed;
  horizontal border-fill replicates column `onscreen_width - 1` into
  the right columns; vertical border-fill replicates row
  `onscreen_height - 1` into the bottom rows; combined corner-block
  fill (2x2 on-screen / 4x4 block) exercises both fills together;
  rectangular shape sweep over every `(onscreen_w, onscreen_h)` in
  `1..=4 × 1..=4` for a 4x4 block confirms no decoder error; the UV
  plane adapts the `palette_uv_color_cdf` family and leaves the Y
  family untouched; chroma-subsampled UV / Y shape parity at the
  4x4 / palette-2 fixture; edge positions on a 2x2 block use the
  `ColorOrder[idx]` remap correctly with the `[0, 1]` identity
  permutation; the 1-wide-block degenerate shape walks without
  underflow; and bitstream-side `read_symbol` underflow propagates
  as [`Error::UnexpectedEnd`] rather than as an `InvalidPaletteWalkArgs`.
  Tests: 323 -> 334, zero `#[ignore]`.

* **Round 146 — §5.11.50 `get_palette_color_context` derivation.** Lands
  the §5.11.50 palette colour-context function (p.103) that consumes the
  `colorMap` at the §5.11.49 diagonal-walk position `(r, c)` plus the
  decoded `palette_size_y` / `palette_size_uv` and produces the
  `ColorOrder[ PALETTE_COLORS ]` permutation + `ColorContextHash` that
  the §5.11.49 syntax feeds back through [`palette_color_ctx`] to the
  §8.3.2 `palette_color_idx_*` cdf selector. Surface:
  `palette_color_context_from_neighbors(left, above_left, above, n) ->
  Option<PaletteColorContext>` (pure-scoring core taking the three
  optional neighbour palette indices) and
  `get_palette_color_context(color_map, stride, r, c, n) ->
  Option<PaletteColorContext>` (spec-faithful 2-D entry that applies
  the §5.11.50 `r > 0` / `c > 0` boundary guards). The
  `PaletteColorContext { color_order, color_context_hash }` struct
  packages both outputs; the partial selection sort is the §5.11.50
  three-iteration loop that promotes the top-scoring neighbours to the
  head of `ColorOrder` while preserving the runners-up's ascending
  order. Tests grow by 11 (cdf module): spec example with every
  neighbour holding the same palette index (`scores = [5, 0, 0]`, hash
  5, ctx 4); distinct left/above with no above-left (hash 6, ctx 3);
  partial-sort swap with two-of-three neighbours sharing an index
  (hash 6, ctx 3); three distinct neighbours covering the +1 / +2
  weight split (hash 8, ctx 1); the no-neighbour identity case (the
  (0,0) corner the §5.11.49 walk never asks for); palette-size
  rejection at `n ∉ 2..=PALETTE_COLORS`; out-of-range neighbour
  rejection (any value `>= PALETTE_COLORS`); full spec-realisable
  combinatorial sweep across every `(left, above_left, above)` at
  every palette size confirming every reachable hash maps to a
  `Some(_)` ctx and the `-1` entries of `PALETTE_COLOR_CONTEXT`
  (hashes 0, 1, 3, 4) are unreachable; 2-D entry-point equivalence
  with `palette_color_context_from_neighbors` across an interior
  position, the top-left corner, top-row-only, and left-column-only
  positions; 2-D boundary rejection (zero stride / OOB column / OOB
  palette size / OOB row); end-to-end `SymbolDecoder` read through
  the `palette_color_idx_y` default cdf selected by the
  derivation -> `palette_color_ctx` -> `palette_y_color_cdf` chain.
  Tests: 312 -> 323, zero `#[ignore]`.

* **Round 145 — §8.3.2 `split_or_horz` / `split_or_vert` derivations.**
  Lands the two §8.3.2 cdf-derivation helpers that build a 2-symbol
  binary cdf out of the already-selected `partition` cdf (the spec's
  `partitionCdf`) per p.362. Each helper folds the §9.4 partition
  probabilities of the "splittable plus orthogonal-axis" symbols into a
  single `psum`, then emits the §8.2.6 binary cdf `cdf[0] = (1 << 15) -
  psum`, `cdf[1] = 1 << 15`, `cdf[2] = 0`. Per the §8.3.2 note the
  disallowed orthogonal partition's probability is folded into the
  split branch — `split_or_horz` cannot return `PARTITION_VERT` and
  `split_or_vert` cannot return `PARTITION_HORZ`. The
  `b_size != BLOCK_128X128` guard drops the `PARTITION_*_4` term that
  the §9.4 `Default_Partition_W128_Cdf` row has no entry for. The §3 /
  §6.10.4 partition ordinals `PARTITION_NONE` (`= 0`), `PARTITION_HORZ`
  (`= 1`), `PARTITION_VERT` (`= 2`), `PARTITION_SPLIT` (`= 3`),
  `PARTITION_HORZ_A` (`= 4`), `PARTITION_HORZ_B` (`= 5`),
  `PARTITION_VERT_A` (`= 6`), `PARTITION_VERT_B` (`= 7`),
  `PARTITION_HORZ_4` (`= 8`), `PARTITION_VERT_4` (`= 9`) plus
  `EXT_PARTITION_TYPES` (`= 10`) and `BLOCK_128X128` (`= 15`) replace
  the literal indices the §8.3.2 formulas use. Tests grow by 10
  (cdf module): partition ordinal pin against the §6.10.4 p.172 table;
  W{16,32,64,128} partition cdf row-length budget vs the §8.3.2
  indexing reach (`PARTITION_VERT_4` for non-128, `PARTITION_VERT_B`
  for W128); spec-`psum` cross-check for `split_or_horz` (W16, ctx 0)
  and `split_or_vert` (W32, ctx 3) re-derived inline; `PARTITION_*_4`
  omission verified for both helpers under `b_size == BLOCK_128X128`;
  full `b_size`-stratified §8.2.6 well-formedness sweep across every
  `DEFAULT_PARTITION_W{16,32,64,128}_CDF` row; end-to-end `SymbolDecoder`
  reads through both derived cdfs (`BLOCK_64X64` for `split_or_horz`,
  `BLOCK_128X128` for `split_or_vert`) confirming the 2-symbol decode
  lands in `0..2`; out-of-range guard rejecting the `bsl == 1` W8 row
  (which the §8.3.2 note forbids for both helpers) with `None`. Tests:
  302 -> 312, zero `#[ignore]`.

* **Round 144 — §9.4 wedge-index default-CDF.** Lands the §9.4
  `Default_Wedge_Index_Cdf[ BLOCK_SIZES ][ WEDGE_TYPES + 1 ]` table —
  the 16-symbol `wedge_index` element read by §5.11.28
  `read_interintra_mode` (the inter-intra wedge sub-branch, when
  `wedge_interintra == 1`) and §5.11.29 `read_compound_type` (the
  inter-inter `COMPOUND_WEDGE` branch). Transcribed verbatim from the
  §9.4 listing (p.435). Adds the §3 constant `WEDGE_TYPES = 16` (the
  spec text reads *"Number of directions for the wedge mask process"*).
  The `TileCdfContext` grows a `wedge_index` field seeded by
  `new_from_defaults` per §8.3.1 ("`WedgeIndexCdf` is set to a copy of
  `Default_Wedge_Index_Cdf`"). The §8.3.2 selection surfaces a
  `wedge_index_cdf(mi_size) -> Option<&mut [u16]>` selector that
  implements the straight `TileWedgeIndexCdf[ MiSize ]` indexing. The
  table's outer dimension is transcribed full-width per the §9.4
  listing; per its note (p.436) indices 0..2, 10..17, and 20..21 are
  never used in the first dimension (matching the §3
  `Wedge_Bits[ MiSize ] == 0` rows) and carry the placeholder uniform
  CDF `{ 2048, 4096, …, 30720, 32768, 0 }` (step `32768 / WEDGE_TYPES`).
  Tests grow by 6 (cdf module): `WEDGE_TYPES` constant pin; default
  table values (cross-checked against the §3 `Wedge_Bits` table — every
  `Wedge_Bits == 0` row must equal the placeholder uniform CDF, every
  `Wedge_Bits > 0` row must not) and §8.2.6 well-formedness (every row
  trails with `0` after `1 << 15 == 32768`, with strictly increasing
  cumulative frequencies); `init_non_coeff_cdfs` working-copy seeding;
  per-row selector return value with out-of-range rejection;
  working-copy independence from the §9.4 source; end-to-end
  `SymbolDecoder` read through a `wedge_index` row from the reachable
  band (`BLOCK_16X16`, `Wedge_Bits[6] = 4`) confirming the 16-symbol
  decode lands in `0..WEDGE_TYPES`. Tests: 296 -> 302, zero `#[ignore]`.

* **Round 143 — §9.4 inter-intra default-CDF group.** Lands the three
  §9.4 default CDFs read by the §5.11.28 `read_interintra_mode`
  syntax — `Default_Inter_Intra_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ 3 ]`
  (binary `interintra`),
  `Default_Inter_Intra_Mode_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ INTERINTRA_MODES + 1 ]`
  (4-symbol `interintra_mode`), and
  `Default_Wedge_Inter_Intra_Cdf[ BLOCK_SIZES ][ 3 ]` (binary
  `wedge_interintra`) — all transcribed verbatim from the §9.4
  listing. Adds the §3 / §6.10.27 constant `INTERINTRA_MODES = 4`
  (`II_DC_PRED` / `II_V_PRED` / `II_H_PRED` / `II_SMOOTH_PRED`) and
  the `interintra_ctx(mi_size) -> Option<usize>` helper that folds
  the §8.3.2 `ctx = Size_Group[ MiSize ] - 1` mapping into a single
  scalar (returning `None` when `Size_Group[ MiSize ] == 0`, i.e.
  for `MiSize < BLOCK_8X8` — the rows that the §5.11.28 syntax gate
  excludes). The `TileCdfContext` grows the `inter_intra` /
  `inter_intra_mode` / `wedge_inter_intra` fields, seeded by
  `new_from_defaults` per §8.3.1. The §8.3.2 selection surfaces the
  `inter_intra_cdf(ctx)` / `inter_intra_mode_cdf(ctx)` /
  `wedge_inter_intra_cdf(mi_size)` selectors. The wedge table's
  outer dimension is transcribed full-width per the §9.4 listing;
  per its note only first-dimension indices 3..=9 (the
  `BLOCK_8X8`..`BLOCK_32X32` band — the same band the §5.11.28
  syntax gate confines coded blocks to) are reachable, with the
  other rows holding the placeholder `{16384, 32768, 0}` row. Tests
  grow by 8 (cdf module): `INTERINTRA_MODES` constant pin; default
  table values and §8.2.6 well-formedness (every row trails with
  `0` after `1 << 15 == 32768`); `init_non_coeff_cdfs` working-copy
  seeding; `interintra_ctx` matches `Size_Group[ MiSize ] - 1`
  across the entire `BLOCK_SIZES` axis (including `None`-rejection
  on the `Size_Group == 0` rows and on `mi_size >= BLOCK_SIZES`);
  per-row selector return value with out-of-range rejection;
  working-copy independence from the §9.4 sources; end-to-end
  `SymbolDecoder` reads through `interintra` + `interintra_mode`
  default CDFs (`BLOCK_16X16` -> `ctx = 1`) and a separate read
  through a `wedge_interintra` row from the reachable band.
  Tests: 288 -> 296, zero `#[ignore]`.

* **Round 142 — §5.11.40 `compute_tx_type()` derivation.** Lands the
  per-plane / per-block transform-type lookup the tile-content walker
  reads before kicking off coefficient decoding and inverse transform.
  `compute_tx_type(plane, tx_sz, lossless, is_inter, tx_set, mi_row,
  mi_col, block_x, block_y, subsampling_x, subsampling_y, uv_mode,
  tx_types)` implements the full spec function:
  `Lossless || Tx_Size_Sqr_Up[ txSz ] > TX_32X32` short-circuits to
  `DCT_DCT`; `plane == 0` returns the `TxTypes[ blockY ][ blockX ]`
  luma cache entry; `is_inter` chroma reads the cache at
  `(Max(MiRow, blockY << subsampling_y), Max(MiCol, blockX <<
  subsampling_x))` then runs the §5.11.40 `is_tx_type_in_set`
  admission filter; intra chroma reads `Mode_To_Txfm[UVMode]` then
  runs the same filter. The caller supplies the §5.11.40 `txSet`
  (i.e. the already-resolved `inter_tx_type_set` / `intra_tx_type_set`
  result) and a closure over `TxTypes[y][x]` so the helper does not
  bake in a particular storage shape — a dense 2D array, a sparse
  map, or a `MiRow/MiCol`-relative tile-local view all work. The
  closure is only invoked on the luma / inter-chroma branches, never
  on the intra-chroma branch. `is_tx_type_in_set(is_inter, tx_set,
  tx_type)` is a direct read of `Tx_Type_In_Set_Inter` /
  `Tx_Type_In_Set_Intra`; out-of-range `tx_set` / `tx_type` returns
  `false` (the spec's reachable values stay in range, so `false`
  flags a bookkeeping bug). Adds the §6.10.16 size ordinal constants
  `TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` / `TX_64X64`
  (replacing the previously locally-scoped `const TX_16X16 = 2;` /
  `const TX_32X32 = 3;` shadows inside `inter_tx_type_set` /
  `intra_tx_type_set`), the §6.10.19 transform-type ordinals
  `DCT_DCT` through `H_FLIPADST` (16 entries), the
  `TX_SET_TYPES_INTRA = 3` / `TX_SET_TYPES_INTER = 4` row-count
  constants, the `Tx_Size_Sqr_Up[ TX_SIZES_ALL ]` table (`t -> Max(w,
  h)-sided square`), the `Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]`
  chroma-mode -> default-tx-type table, and the
  `Tx_Type_In_Set_Intra[ 3 ][ TX_TYPES ]` /
  `Tx_Type_In_Set_Inter[ 4 ][ TX_TYPES ]` admission tables, all
  transcribed verbatim from the spec. Tests grow by 10 (cdf module):
  `Tx_Size_Sqr_Up` well-formedness (every entry is a square ordinal,
  spot-checks against the spec rows); §6.10.16 / §6.10.19 ordinal
  values match the spec; `Mode_To_Txfm` per-row spot-check + bounds;
  `is_tx_type_in_set` per-row admission flags for all 4 inter sets +
  3 intra sets (with out-of-range fallback); `compute_tx_type`
  lossless / `txSzSqrUp > TX_32X32` short-circuits (covering
  `TX_64X64`, `TX_32X64`, and an out-of-range tx_sz); luma branch
  verbatim cache read (with admission filter intentionally inert);
  inter-chroma `Max(MiCol, blockX << subsampling_x)` lift firing
  through a cache that disambiguates lifted vs unlifted columns,
  admission pass for `IDTX` under `TX_SET_INTER_3`, and admission
  fail for `ADST_DCT` falling back to `DCT_DCT`; intra-chroma
  `Mode_To_Txfm` path with admission pass, admission fail, and
  out-of-range `uv_mode` fallback (with a panic-on-call closure
  proving the luma cache is not read); selector / admission filter
  closed-loop check for a 16×16 inter chroma block. 278 -> 288
  tests, zero `#[ignore]`.

* **Round 141 — §8.3.2 `get_coeff_base_ctx()` / `get_br_ctx()`
  neighbour-derivation helpers.** Lands the per-coefficient `ctx`
  computation that feeds the existing `coeff_base` /
  `coeff_base_eob` / `coeff_br` selectors (the r138–r140 braid). Both
  helpers take the coefficient-magnitude array `Quant` plus scalar
  transform / position state and return the `ctx` index; they own the
  §8.3.2 neighbour scan only — the tile-content walker that produces
  `Quant`, `pos`, `c`, and the `compute_tx_type()` derivation is a
  separate gate. `get_coeff_base_ctx(quant, tx_size, tx_class, pos, c,
  is_eob)` scans `Sig_Ref_Diff_Offset` (5 offsets) accumulating
  `Min(Abs(Quant[(refRow<<bwl)+refCol]), 3)` over in-bounds
  neighbours (`refRow < height && refCol < width`, `width = 1<<bwl`,
  `bwl = Tx_Width_Log2[Adjusted_Tx_Size[txSz]]`), forms
  `ctx = Min((mag+1)>>1, 4)`, then routes through the 2D
  `Coeff_Base_Ctx_Offset[txSz][Min(row,4)][Min(col,4)]` branch (with
  the `row==0 && col==0 -> 0` early return) or the 1D
  `Coeff_Base_Pos_Ctx_Offset[Min(idx,2)]` branch (vertical / horizontal);
  the `isEob` path returns the `SIG_COEF_CONTEXTS-{4,3,2,1}` buckets per
  `c` thresholds. `get_coeff_base_eob_ctx(...)` wraps it with the
  §8.3.2 `- SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB` reduction onto
  `0..SIG_COEF_CONTEXTS_EOB`. `get_br_ctx(quant, tx_size, tx_class,
  pos)` scans `Mag_Ref_Offset_With_Tx_Class` (3 offsets) accumulating
  `Min(Quant[refRow*txw+refCol], COEFF_BASE_RANGE+NUM_BASE_LEVELS+1)`
  (no abs; distinct clamp; bound `refRow < txh && refCol < (1<<bwl)`),
  forms `mag = Min((mag+1)>>1, 6)`, then `pos==0 -> mag`; 2D `+7` if
  `row<2 && col<2` else `+14`; horizontal `+7` if `col==0` else `+14`;
  vertical `+7` if `row==0` else `+14` (result in `0..LEVEL_CONTEXTS`).
  Adds the §3 constants `SIG_COEF_CONTEXTS_2D = 26`,
  `SIG_REF_DIFF_OFFSET_NUM = 5`, `NUM_BASE_LEVELS = 2`,
  `COEFF_BASE_RANGE = 12`, `TX_SIZES_ALL = 19`, the `TX_CLASS_{2D,
  HORIZ, VERT}` tags, the `Adjusted_Tx_Size` / `Tx_Width` / `Tx_Height`
  / `Tx_Width_Log2` size tables, the `Sig_Ref_Diff_Offset` /
  `Mag_Ref_Offset_With_Tx_Class` neighbour-offset tables, and the
  `Coeff_Base_Ctx_Offset` / `Coeff_Base_Pos_Ctx_Offset` offset tables,
  all transcribed verbatim from the spec. A pure `get_tx_class()`
  helper reduces the directional transform-type flags to a class.
  Tests grow by 12 (cdf module): constant + table-shape pin
  (including `Tx_Width == 1 << Tx_Width_Log2` self-consistency);
  `get_tx_class` reduction; the four `is_eob` buckets plus the
  `get_coeff_base_eob_ctx` reduction; the `row==0 && col==0 -> 0` 2D
  early return; a 2D case with a non-trivial offset entry and a
  saturated-neighbourhood `ctx` clamp to 4; vertical and horizontal
  branches with `Min(idx,2)` clamping; `get_br_ctx` `pos==0`, the
  saturated magnitude clamp to 6, and each tx-class branch
  (2D inner/outer, horizontal col==0/else, vertical row==0/else), with
  in-range assertions against the matching `coeff_base` / `coeff_br`
  selector. 266 -> 278 tests, zero `#[ignore]`.

* **Round 140 — §9.4 default CDF table + §8.3.1 `init_coeff_cdfs` /
  §8.3.2 selection (`coeff_br` sub-group).** Lands the LAST member of
  the `coeff_base` / `coeff_base_eob` / `coeff_br` braid; with this
  table all three coeff-CDF braid members are now in tree. Extends
  `cdf` with `Default_Coeff_Br_Cdf`
  (`[COEFF_CDF_Q_CTXS][TX_SIZES][PLANE_TYPES][LEVEL_CONTEXTS][BR_CDF_SIZE + 1]`,
  840 5-entry rows = 8400 bytes; declared `static` rather than `const`
  so `clippy::large_const_arrays` does not flag the per-use copy
  hazard) transcribed verbatim from §9.4. `coeff_br` codes the
  per-coefficient base-range increment used to push a level above
  `NUM_BASE_LEVELS`: each read codes a value in `0..BR_CDF_SIZE = 4`,
  and §5.11.39 stacks `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)` such
  reads per coefficient. New §3 constants
  `LEVEL_CONTEXTS = 21` (number of `coeff_br` contexts, the index
  range of the §8.3.2 `get_br_ctx(...)` result) and `BR_CDF_SIZE = 4`
  (the `coeff_br` alphabet size, mirroring §5.11.39's
  `coeff_br < (BR_CDF_SIZE - 1)` loop guard). The §8.3.1
  `init_coeff_cdfs` grows a `self.coeff_br = DEFAULT_COEFF_BR_CDF[idx]`
  copy on the `base_q_idx`-derived `idx`;
  `TileCdfContext::new_from_defaults` seeds the field from `idx == 0`
  so the value is always well-formed. The §8.3.2 selection surfaces
  `coeff_br_cdf(tx_sz_ctx, ptype, ctx)`, implementing the spec
  selector `TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][ ctx ]`
  with `TX_32X32 = 3` clamping built in (so any `txSzCtx` is
  accepted; only `ptype` / `ctx` axes are bounds-checked). The
  largest `(TX_SIZE = TX_64X64, ptype = chroma)` slice is the flat
  `{8192, 16384, 24576, 32768, 0}` placeholder for every q-context
  and ctx value (mirroring the r138 / r139 placeholder pattern),
  locked down by an exhaustive byte-equality test. With this table
  the §9.4 coeff-CDF braid is feature-complete; the next gate is the
  §8.3.2 `get_coeff_base_ctx()` / `get_br_ctx()` neighbour-derivation
  helpers, deferred to a different round because they need tile-content
  walker state. New constants re-exported at the crate root via the
  existing `cdf` glob. Tests grow by 6 (cdf module): constant pin
  (`LEVEL_CONTEXTS = 21`, `BR_CDF_SIZE = 4`); table dimension audit +
  strict-monotonicity / cdf-shape well-formedness across all
  q-contexts; byte-anchor spot-checks of the §9.4 values (luma +
  chroma first-context rows at q0, the last `LEVEL_CONTEXTS - 1`
  row anchor at q0, and the exhaustive `(tx=4, pt=1)`
  flat-placeholder lock-down across every q-context and ctx value);
  `init_coeff_cdfs` q-context re-selection for the new field with
  mutate-doesn't-touch-source independence; selector in-range
  coverage at the §3 `TX_32X32` boundary with explicit clamp
  verification at `TX_SIZES - 1` and at a far-out-of-spec `txSzCtx`,
  with per-axis `None` returns for `ptype` / `ctx`; one end-to-end
  §8.2 `SymbolDecoder` decode driving the `BR_CDF_SIZE`-symbol
  `TileCoeffBrCdf[ 0 ][ 0 ][ 0 ]` row.
* **Round 139 — §9.4 default CDF table + §8.3.1 `init_coeff_cdfs` /
  §8.3.2 selection (`coeff_base` sub-group).** Lands the second
  member of the `coeff_base` / `coeff_base_eob` / `coeff_br` braid,
  the larger sibling of r138's `coeff_base_eob` table. Extends
  `cdf` with `Default_Coeff_Base_Cdf`
  (`[COEFF_CDF_Q_CTXS][TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS][5]`,
  1680 5-entry rows = 16800 bytes; declared `static` rather than
  `const` so `clippy::large_const_arrays` does not flag the per-use
  copy hazard) transcribed verbatim from §9.4. `coeff_base` codes
  the base level of each non-EOB coefficient — a 4-symbol alphabet
  (`0..3`), so each row carries 4 cumulative frequencies plus the
  §8.3 adaptation counter. New §3 constant `SIG_COEF_CONTEXTS = 42`
  (the §3 partition tag `SIG_COEF_CONTEXTS_2D = 26` splits this
  range between the two-dimensional scan prefix and the 1D
  horizontal- / vertical-only tails, used by the not-yet-implemented
  `get_coeff_base_ctx()` lookup). The §8.3.1 `init_coeff_cdfs`
  grows a `self.coeff_base = DEFAULT_COEFF_BASE_CDF[idx]` copy on
  the `base_q_idx`-derived `idx`; `TileCdfContext::new_from_defaults`
  seeds the field from `idx == 0` so the value is always well-formed.
  The §8.3.2 selection surfaces `coeff_base_cdf(tx_sz_ctx, ptype,
  ctx)`, the three-way `TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]`
  lookup (the `get_coeff_base_ctx()` derivation itself belongs to
  the not-yet-implemented tile-content walk and is deferred). Just
  as in r138, the largest `(TX_SIZE = TX_64X64, ptype = chroma)`
  slice is the flat `{8192, 16384, 24576, 32768, 0}` placeholder
  for every q-context and ctx value — an unreachable-chroma
  sentinel — locked down by an exhaustive byte-equality test. The
  last remaining table of the braid (`Default_Coeff_Br_Cdf`) is
  deferred to a later round. New type / constant re-exported at the
  crate root. Tests grow by 6 (cdf module): `SIG_COEF_CONTEXTS`
  pin; table dimension audit + strict-monotonicity / cdf-shape
  well-formedness across all q-contexts; byte-anchor spot-checks of
  the §9.4 values (luma + chroma first-context rows at q0 and the
  highest TX size, a high-q-context anchor, and the exhaustive
  `(tx=4, pt=1)` flat-placeholder lock-down across every q-context
  and ctx value); `init_coeff_cdfs` q-context re-selection for the
  new field with mutate-doesn't-touch-source independence; selector
  in-range coverage at the §3 `SIG_COEF_CONTEXTS_2D` split point
  and at the highest in-range context, with per-axis out-of-range
  `None` returns; one end-to-end §8.2 `SymbolDecoder` decode
  driving the 4-symbol `TileCoeffBaseCdf[ 0 ][ 0 ][ 0 ]` row.
* **Round 138 — §9.4 default CDF table + §8.3.1 `init_coeff_cdfs` /
  §8.3.2 selection (`coeff_base_eob` sub-group).** Lands the first
  member of the `coeff_base` / `coeff_base_eob` / `coeff_br` braid,
  the next gateway to tile-content decode following the round-136
  coefficient-token entry sub-group. Extends `cdf` with
  `Default_Coeff_Base_Eob_Cdf`
  (`[COEFF_CDF_Q_CTXS][TX_SIZES][PLANE_TYPES][SIG_COEF_CONTEXTS_EOB][4]`)
  transcribed verbatim from §9.4: `coeff_base_eob` codes the base
  level of the last non-zero coefficient (the base level is
  `coeff_base_eob + 1`, restricted to 1, 2, or 3, so only three
  symbols are coded — a 4-entry row of 3 cumulative frequencies plus
  the §8.3 adaptation counter). New §3 constant
  `SIG_COEF_CONTEXTS_EOB = 4`. The §8.3.1 `init_coeff_cdfs` grows a
  `self.coeff_base_eob = DEFAULT_COEFF_BASE_EOB_CDF[idx]` copy on the
  `base_q_idx`-derived `idx`; `TileCdfContext::new_from_defaults`
  seeds the field from `idx == 0` so the value is always well-formed.
  The §8.3.2 selection surfaces `coeff_base_eob_cdf(tx_sz_ctx, ptype,
  ctx)`, the three-way `TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]`
  lookup (the `get_coeff_base_ctx() - SIG_COEF_CONTEXTS +
  SIG_COEF_CONTEXTS_EOB` reduction belongs to the not-yet-implemented
  tile-content walk and is deferred). The two remaining tables of the
  braid (`Default_Coeff_Base_Cdf` and `Default_Coeff_Br_Cdf`) are
  deferred to later rounds. New type / constant re-exported at the
  crate root. Tests grow by 6 (cdf module): `SIG_COEF_CONTEXTS_EOB`
  pin; table dimension audit + strict-monotonicity / cdf-shape
  well-formedness across all q-contexts; byte-anchor spot-checks of
  the §9.4 values (luma + chroma first-context rows, the flat
  `{10923, 21845, 32768, 0}` placeholder padding the largest TX
  size's chroma slice across all q-contexts and ctx values, and a
  high-q interior anchor); `init_coeff_cdfs` q-context re-selection
  for the new field with mutate-doesn't-touch-source independence;
  selector in-range coverage with per-axis out-of-range `None`
  returns; one end-to-end §8.2 `SymbolDecoder` decode driving the
  3-symbol `TileCoeffBaseEobCdf[ 0 ][ 0 ][ 0 ]` row.
* **Round 137 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (intra-frame transform-type subset).** Completes the §6.10.19
  transform-set coverage started in round 21 by extending `cdf` with the
  two intra-frame transform-type default tables transcribed verbatim
  from §9.4: `Default_Intra_Tx_Type_Set1_Cdf[ 2 ][ INTRA_MODES ][ 8 ]`
  — the 7-symbol full intra set for 4x4 / 8x8 intra blocks reaching
  `TX_SET_INTRA_1` (`Tx_Type_Intra_Inv_Set1 = { IDTX, DCT_DCT, V_DCT,
  H_DCT, ADST_ADST, ADST_DCT, DCT_ADST }`); and
  `Default_Intra_Tx_Type_Set2_Cdf[ 3 ][ INTRA_MODES ][ 6 ]` — the
  5-symbol reduced intra set for 4x4 / 8x8 / 16x16 intra blocks
  reaching `TX_SET_INTRA_2` (`Tx_Type_Intra_Inv_Set2 = { IDTX, DCT_DCT,
  ADST_ADST, ADST_DCT, DCT_ADST }`). New §3 constants
  `TX_SET_INTRA_1 = 1`, `TX_SET_INTRA_2 = 2`, `TX_TYPES_INTRA_SET1 = 7`,
  `TX_TYPES_INTRA_SET2 = 5`, `INTRA_TX_TYPE_SET1_SIZES = 2`,
  `INTRA_TX_TYPE_SET2_SIZES = 3`, and the §8.3.2
  `Filter_Intra_Mode_To_Intra_Dir[ INTRA_FILTER_MODES ]` table
  (`{ DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED }`). New
  `TileCdfContext` fields `intra_tx_type_set1` / `intra_tx_type_set2`,
  initialised by `TileCdfContext::new_from_defaults` per §8.3.1
  ("`IntraTxTypeSet1Cdf` is set equal to a copy of
  `Default_Intra_Tx_Type_Set1_Cdf`" and likewise for Set2). The §8.3.2
  selection surfaces `intra_tx_type_cdf(set, tx_size_sqr, intra_dir)`,
  the two-way `TileIntraTxTypeSet{1,2}Cdf` switch indexed on the
  `intraDir` axis (returning `None` for `TX_SET_DCTONLY` per §5.11.47
  and for unreachable `(set, tx_size_sqr, intra_dir)` combinations).
  Two scalar helpers complete the path: `intra_tx_type_set(tx_sz_sqr,
  tx_sz_sqr_up, reduced_tx_set)` mirrors §5.11.48 `get_tx_set()` on the
  `is_inter == 0` branch (differing from the inter counterpart in
  routing `txSzSqrUp == TX_32X32` to `TX_SET_DCTONLY` and
  `txSzSqr == TX_16X16` to `TX_SET_INTRA_2`), and
  `intra_dir(use_filter_intra, y_mode, filter_intra_mode)` derives the
  §8.3.2 `intraDir` axis. All new types / constants / helpers
  re-exported at the crate root. Tests grow by 7 (cdf module): table
  well-formedness + dimension audit against the §3 constants;
  byte-anchor spot-checks plus the explicit `Set2` flat-distribution
  check for sizes 0..=1; §8.3.1 init-copy independence with a
  mutate-doesn't-touch-source assertion; selector two-way coverage
  with row-length assertions and unreachable / out-of-range `None`
  returns; the `intra_tx_type_set` formula walked across every
  reachable `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)` triple; the
  `intra_dir` derivation for the pass-through and filter-intra
  branches; and one end-to-end §8.2 `SymbolDecoder` decode driving
  the 5-symbol `TileIntraTxTypeSet2Cdf[ 2 ][ DC_PRED ]` row.
* **Round 136 — §9.4 default CDF tables + §8.3.1 `init_coeff_cdfs` /
  §8.3.2 selection (coefficient-token entry sub-group).** Extends `cdf`
  with the entry sub-group of the coefficient-token CDFs — the gateway
  to tile-content decode — transcribed verbatim from §9.4: the
  transform-block skip flag `Default_Txb_Skip_Cdf`
  (`[COEFF_CDF_Q_CTXS][TX_SIZES][TXB_SKIP_CONTEXTS][3]`), the
  end-of-block position classes `Default_Eob_Pt_16/32/64/128/256_Cdf`
  (`[COEFF_CDF_Q_CTXS][PLANE_TYPES][2][N]`) plus the no-`isInter`-axis
  `Default_Eob_Pt_512/1024_Cdf` (`[COEFF_CDF_Q_CTXS][PLANE_TYPES][N]`),
  the binary `Default_Eob_Extra_Cdf`
  (`[COEFF_CDF_Q_CTXS][TX_SIZES][PLANE_TYPES][EOB_COEF_CONTEXTS][3]`),
  and the binary `Default_Dc_Sign_Cdf`
  (`[COEFF_CDF_Q_CTXS][PLANE_TYPES][DC_SIGN_CONTEXTS][3]`, in the §9.4
  `128 * N` fixed-point form). New §3 constants `PLANE_TYPES = 2`,
  `COEFF_CDF_Q_CTXS = 4`, `TXB_SKIP_CONTEXTS = 13`,
  `EOB_COEF_CONTEXTS = 9`, `DC_SIGN_CONTEXTS = 3`. Unlike the non-coeff
  CDFs, these are reset by the separate `TileCdfContext::init_coeff_cdfs`,
  which derives the q-context `idx` from `base_q_idx` (via the new
  `coeff_cdf_q_ctx` helper: `<=20→0`, `<=60→1`, `<=120→2`, else `3`) and
  copies `Default_*_Cdf[ idx ]` into the working arrays (so the working
  copy drops the `COEFF_CDF_Q_CTXS` axis). `new_from_defaults` seeds the
  fields from `idx 0` so the value is always well-formed. The §8.3.2
  selection surfaces `txb_skip_cdf` / `eob_pt_{16,32,64,128,256}_cdf` /
  `eob_pt_{512,1024}_cdf` / `eob_extra_cdf` / `dc_sign_cdf`. All new
  types / constants re-exported at the crate root. Tests grow by 7 (cdf
  module): §3-constant pins, table well-formedness + strict-monotonicity
  across all q-contexts, byte-anchor spot-checks of the §9.4 values,
  `coeff_cdf_q_ctx` boundary mapping, `init_coeff_cdfs` q-context
  re-selection with mutate-doesn't-touch-source independence, selector
  row-equality + out-of-range `None` returns, and end-to-end §8.2
  `SymbolDecoder` decodes driving the `all_zero` / `eob_pt_16` /
  `dc_sign` default CDFs. The coeff_base / coeff_base_eob / coeff_br
  braid is deferred to a later round.
* **Round 135 — §9.4 default CDF table + §8.3.1 / §8.3.2 selection
  (angle-delta subset).** Extends `cdf` with the angle-delta default
  table `Default_Angle_Delta_Cdf`
  (`[DIRECTIONAL_MODES][(2 * MAX_ANGLE_DELTA + 1) + 1]`, 8 directional-mode
  rows × 7 cumulative frequencies + adaptation counter; the
  `angle_delta_y` / `angle_delta_uv` directional-prediction angle offset)
  — transcribed verbatim from §9.4. New §3 constants
  `DIRECTIONAL_MODES = 8`, `MAX_ANGLE_DELTA = 3` and the directional-mode
  base `V_PRED = 1`. New `TileCdfContext` field `angle_delta`, initialised
  by `TileCdfContext::new_from_defaults` per §8.3.1 ("`AngleDeltaCdf` is
  set to a copy of `Default_Angle_Delta_Cdf`"). Selection accessor
  `angle_delta_cdf(mode)` lands, indexing `TileAngleDeltaCdf[ mode - V_PRED ]`
  — the §8.3.2 `TileAngleDeltaCdf[ YMode - V_PRED ]` /
  `TileAngleDeltaCdf[ UVMode - V_PRED ]` selection for the luma / chroma
  elements — returning `None` for non-directional modes (below `V_PRED`
  or at/above `V_PRED + DIRECTIONAL_MODES`). All new types / constants
  re-exported at the crate root. Tests grow by 5 (cdf module): table
  well-formedness + strict-monotonicity against §3 constants,
  byte-anchor spot-checks of the §9.4 row values, §8.3.1 init-copy
  independence with mutate-doesn't-touch-source assertions, selector
  row-equality across every directional mode plus non-directional-mode
  `None` returns, and an end-to-end §8.2 `SymbolDecoder` decode driving
  `Default_Angle_Delta_Cdf[2]` selected by `angle_delta_cdf(D45_PRED)`.
* **Round 134 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (inter-frame intra-mode subset).** Extends `cdf` with the three
  inter-frame intra-mode default tables — `Default_Y_Mode_Cdf`
  (`[BLOCK_SIZE_GROUPS][INTRA_MODES + 1]`, 4 block-size-group contexts ×
  13 cumulative frequencies + adaptation counter; the non-keyframe luma
  `y_mode` element, distinct from the keyframe
  `Default_Intra_Frame_Y_Mode_Cdf`), `Default_Uv_Mode_Cfl_Not_Allowed_Cdf`
  (`[INTRA_MODES][UV_INTRA_MODES_CFL_NOT_ALLOWED + 1]`) and
  `Default_Uv_Mode_Cfl_Allowed_Cdf`
  (`[INTRA_MODES][UV_INTRA_MODES_CFL_ALLOWED + 1]`) — transcribed
  verbatim from §9.4. New §3 constants `BLOCK_SIZE_GROUPS = 4`,
  `UV_INTRA_MODES_CFL_NOT_ALLOWED = 13`, `UV_INTRA_MODES_CFL_ALLOWED = 14`
  plus the §8.3.2 `Size_Group[ BLOCK_SIZES ]` table. New `TileCdfContext`
  fields `y_mode` / `uv_mode_cfl_not_allowed` / `uv_mode_cfl_allowed`,
  initialised by `TileCdfContext::new_from_defaults` per §8.3.1
  ("`YModeCdf` / `UVModeCflNotAllowedCdf` / `UVModeCflAllowedCdf` is set
  to a copy of `Default_*`"). Selection accessors land —
  `y_mode_cdf(ctx)` indexing `TileYModeCdf[ Size_Group[ MiSize ] ]`
  (with `size_group()` performing the §8.3.2 mapping), and
  `uv_mode_cdf(cfl_allowed, y_mode)` picking the cfl-allowed /
  cfl-not-allowed variant by the resolved flag (the `Lossless` /
  `get_plane_residual_size` / `Max(Block_Width, Block_Height) <= 32`
  derivation belongs in the future tile walk) then indexing by `YMode`,
  returning `None` out of range. All new types / constants re-exported at
  the crate root. Tests grow by 7 (cdf module): table well-formedness +
  strict-monotonicity against §3 constants, byte-anchor spot-checks of
  the §9.4 row values, the `Size_Group` table pinned byte-for-byte,
  §8.3.1 init-copy independence with mutate-doesn't-touch-source
  assertions, selector row-equality across every context / `YMode` /
  variant plus out-of-range `None` returns, and two end-to-end §8.2
  `SymbolDecoder` decodes driving `Default_Y_Mode_Cdf[3]` and both
  `uv_mode` variants selected by the new helpers.
* **Round 24 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (compound-prediction subset).** Extends `cdf` with the three
  compound-prediction default tables — `Default_Comp_Group_Idx_Cdf`
  (`[COMP_GROUP_IDX_CONTEXTS][3]`), `Default_Compound_Idx_Cdf`
  (`[COMPOUND_IDX_CONTEXTS][3]`) and `Default_Compound_Type_Cdf`
  (`[BLOCK_SIZES][COMPOUND_TYPES + 1]`, 22 block-size rows × 2
  cumulative frequencies + adaptation counter) — transcribed verbatim
  from §9.4 (including the spec-flagged-unreachable
  `Default_Compound_Type_Cdf` rows 0..=2, 10..=17 and 20..=21 which
  carry the flat `{ 16384, 32768, 0 }` placeholder per the §9.4 note).
  New §3 constants `COMPOUND_TYPES = 2`, `COMP_GROUP_IDX_CONTEXTS = 6`,
  `COMPOUND_IDX_CONTEXTS = 6`. New `TileCdfContext` fields
  `comp_group_idx` / `compound_idx` / `compound_type`, initialised by
  `TileCdfContext::new_from_defaults` per §8.3.1 ("`CompGroupIdxCdf` /
  `CompoundIdxCdf` / `CompoundTypeCdf` is set to a copy of
  `Default_*`"). Three §8.3.2 selection accessors land —
  `comp_group_idx_cdf(ctx)` and `compound_idx_cdf(ctx)` (binary,
  taking the precomputed §8.3.2 neighbour-derived context whose
  arithmetic belongs in the future tile walk) plus
  `compound_type_cdf(mi_size)` (a straight `0..BLOCK_SIZES` index per
  the §8.3.2 text "`TileCompoundTypeCdf[ MiSize ]`", returning `None`
  for `mi_size >= BLOCK_SIZES`). All new types / constants re-exported
  at the crate root. Tests grow by 6 (cdf module): table
  well-formedness against §3 constants, byte-anchor spot-checks of the
  §9.4 row values (covering both the spec-flagged-unreachable
  placeholders and the reachable runs), §8.3.1 init-copy independence
  with mutate-doesn't-touch-source assertions, selector row-equality
  across every context / `MiSize` plus out-of-range `None` returns,
  and two end-to-end §8.2 `SymbolDecoder` decodes driving the
  `Default_Compound_Type_Cdf[9]` and `Default_Comp_Group_Idx_Cdf[2]`
  rows selected by the new helpers.

* **Round 23 — §9.4 default CDF table + §8.3.1 / §8.3.2 selection
  (motion-mode subset).** Extends `cdf` with the
  `Default_Motion_Mode_Cdf` default table —
  `[BLOCK_SIZES][MOTION_MODES + 1]` (22 block-size rows × 3 cumulative
  frequencies + adaptation counter), transcribed verbatim from §9.4
  (including the spec-flagged-unreachable rows 0..=2 and 16..=17 which
  initialise to the flat `{ 10923, 21845, 32768, 0 }` placeholder).
  New §3 constant `MOTION_MODES = 3` (per §6.10.26 semantics:
  `SIMPLE = 0`, `OBMC = 1`, `LOCALWARP = 2`). New `TileCdfContext`
  field `motion_mode`, initialised by `TileCdfContext::new_from_defaults`
  per §8.3.1 ("`MotionModeCdf` is set to a copy of
  `Default_Motion_Mode_Cdf`"). One §8.3.2 selection accessor lands —
  `motion_mode_cdf(mi_size)` — a straight `0..BLOCK_SIZES` index (the
  spec's §8.3.2 selection text reads "`TileMotionModeCdf[ MiSize ]`";
  no neighbour-context arithmetic). Bounds-check returns `None` for
  `mi_size >= BLOCK_SIZES`. All new types / constants re-exported at
  the crate root. Tests grow from 211 to 216 (cdf module): table
  well-formedness against §3 constants, byte-anchor spot-checks of the
  §9.4 row values (rows 0/1/2/3/9/15/16/17/21 covering both the
  spec-flagged-unreachable placeholders and the heaviest-bias rows),
  §8.3.1 init-copy independence with mutate-doesn't-touch-source
  assertion, selector row-equality for every `MiSize` plus
  out-of-range `None` returns, and one end-to-end §8.2 `SymbolDecoder`
  decode driving the `Default_Motion_Mode_Cdf[15]` row selected by
  the new helper.

* **Round 22 — §9.4 default CDF table + §8.3.1 / §8.3.2 selection
  (inter-frame interpolation-filter subset).** Extends `cdf` with the
  `Default_Interp_Filter_Cdf` default table —
  `[INTERP_FILTER_CONTEXTS][INTERP_FILTERS + 1]` (16 contexts × 3
  cumulative frequencies + adaptation counter), transcribed verbatim
  from §9.4. New §3 constants `INTERP_FILTERS = 3` and
  `INTERP_FILTER_CONTEXTS = 16`, plus the sentinel
  `INTERP_FILTER_NONE = INTERP_FILTERS` (mirrors the spec's literal `3`
  marker for unavailable / mismatched neighbours). New
  `TileCdfContext::interp_filter` field, initialised by
  `TileCdfContext::new_from_defaults` per §8.3.1. One §8.3.2 selection
  accessor lands — `interp_filter_cdf(ctx)` (with bounds-check return
  of `None` for `ctx >= INTERP_FILTER_CONTEXTS`). The scalar §8.3.2
  helper `interp_filter_ctx(above_type, left_type, dir, is_compound)`
  folds the §8.3.2 four-branch formula
  (`((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4` base, then
  `+ leftType` / `+ aboveType` / `+ INTERP_FILTERS` per the
  match-vs-NONE branches) into a single
  `0..INTERP_FILTER_CONTEXTS` index — the caller supplies the
  already-resolved neighbour-filter values per the spec's
  `RefFrames[..][0|1] == RefFrame[0]` matching predicate (the
  neighbour walk itself lives in the future tile-walk crate). All new
  types / constants / fns re-exported at the crate root. Tests grow
  from 204 to 211 (cdf module): table well-formedness against §3
  constants, byte-anchor spot-checks of the §9.4 row values
  (rows 0/2/7/8/14/15), §8.3.1 init-copy independence with
  mutate-doesn't-touch-source assertion, `interp_filter_ctx` walk
  across all four §8.3.2 branches (match, left-NONE, above-NONE,
  distinct) and across all four `(dir, is_compound)` quadrants, an
  exhaustive coverage walk that hits every reachable
  `0..INTERP_FILTER_CONTEXTS` ctx, selector row-equality for every
  ctx, and one end-to-end §8.2 `SymbolDecoder` decode driving the
  `Default_Interp_Filter_Cdf[2]` row selected by the new helpers.

* **Round 21 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (inter-frame transform-type subset).** Extends `cdf` with three
  new default tables (`Default_Inter_Tx_Type_Set1_Cdf` —
  `[INTER_TX_TYPE_SET1_SIZES][TX_TYPES + 1]` for 4x4 / 8x8 inter
  blocks reaching `TX_SET_INTER_1`; `Default_Inter_Tx_Type_Set2_Cdf`
  — flat `[TX_TYPES_SET2 + 1]` for 16x16 inter blocks reaching
  `TX_SET_INTER_2`; `Default_Inter_Tx_Type_Set3_Cdf` —
  `[INTER_TX_TYPE_SET3_SIZES][TX_TYPES_SET3 + 1]` for 4x4..32x32
  inter blocks reaching the reduced `{ IDTX, DCT_DCT }`
  `TX_SET_INTER_3`) — all transcribed verbatim from §9.4. New §3
  constants `TX_TYPES = 16`, `TX_TYPES_SET2 = 12`, `TX_TYPES_SET3 = 2`,
  `INTER_TX_TYPE_SET1_SIZES = 2`, `INTER_TX_TYPE_SET3_SIZES = 4` and
  the §6.10.19 transform-set tag constants `TX_SET_DCTONLY = 0`,
  `TX_SET_INTER_1 = 1`, `TX_SET_INTER_2 = 2`, `TX_SET_INTER_3 = 3`.
  New `TileCdfContext` fields (`inter_tx_type_set1`,
  `inter_tx_type_set2`, `inter_tx_type_set3`), all initialised by
  `TileCdfContext::new_from_defaults` per §8.3.1. One §8.3.2
  selection accessor lands — `inter_tx_type_cdf(set, tx_size_sqr)`
  (the §8.3.2 three-way `TileInterTxTypeSet{1,2,3}Cdf` switch keyed
  by the §5.11.48 set; `None` for `TX_SET_DCTONLY` per §5.11.47 and
  for unreachable `(set, tx_size_sqr)` combinations). New scalar
  §5.11.48 helper `inter_tx_type_set(tx_sz_sqr, tx_sz_sqr_up,
  reduced_tx_set)` computes the set ∈ `{ TX_SET_DCTONLY,
  TX_SET_INTER_1, TX_SET_INTER_2, TX_SET_INTER_3 }` from the
  `Tx_Size_Sqr` / `Tx_Size_Sqr_Up` / `reduced_tx_set` tuple supplied
  by §5.11.47. All new types / constants / fns re-exported at the
  crate root. Tests grow from 198 to 204 (cdf module): table
  well-formedness + dimensions against §3 constants, byte-anchor
  spot-checks on every transcribed table, §8.3.1 init-copy
  independence with mutate-doesn't-touch-source assertion,
  `inter_tx_type_cdf` three-way selection with row-length
  assertions, `inter_tx_type_set` walk across every reachable
  `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)` triple, and one
  end-to-end §8.2 `SymbolDecoder` decode driving the 2-value
  `TileInterTxTypeSet3Cdf[ 1 ]` row selected by the new helpers.
  The intra counterpart (`Default_Intra_Tx_Type_Set{1,2}_Cdf`,
  with their `[INTRA_MODES][..]` second axis and `intraDir`
  selection) is a mechanical follow-up against the same
  `TileCdfContext` shape.

* **Round 20 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (transform-size subset).** Extends `cdf` with five new default
  tables (`Default_Tx_8x8_Cdf`, `Default_Tx_16x16_Cdf`,
  `Default_Tx_32x32_Cdf`, `Default_Tx_64x64_Cdf`,
  `Default_Txfm_Split_Cdf`) — all transcribed verbatim from §9.4.
  New §3 constants `TX_SIZE_CONTEXTS = 3`, `TX_SIZES = 5`,
  `MAX_TX_DEPTH = 2`, `TXFM_PARTITION_CONTEXTS = 21`. New
  `TileCdfContext` fields (`tx_8x8`, `tx_16x16`, `tx_32x32`,
  `tx_64x64`, `txfm_split`), all initialised by
  `TileCdfContext::new_from_defaults` per §8.3.1. Two §8.3.2
  selection accessors land: `tx_depth_cdf(max_tx_depth, ctx)`
  (returns the right `TileTx*Cdf` row per the §8.3.2 four-way
  `maxTxDepth` switch, `None` when `max_tx_depth == 0`) and
  `txfm_split_cdf(ctx)`. Two new scalar §8.3.2 helpers
  `tx_depth_ctx(above_w, left_h, max_tx_width, max_tx_height)`
  (the `(aboveW >= maxTxWidth) + (leftH >= maxTxHeight)` formula)
  and `txfm_split_ctx(above, left, tx_sz_sqr_up, max_tx_sz)`
  (the `(txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 +
  above + left` formula, returns `None` for unreachable
  combinations that would land outside `0..TXFM_PARTITION_CONTEXTS`).
  All new types / constants / fns re-exported at the crate root.
  Tests grow from 190 to 198 (cdf module): table well-formedness +
  dimensions against §3 constants, byte-anchor spot-checks on
  every transcribed table, §8.3.1 init-copy independence,
  `tx_depth_cdf` four-way selection with row-length assertions,
  `tx_depth_ctx` formula across all neighbour combinations,
  `txfm_split_ctx` formula walked term-by-term + an exhaustive
  in-range sweep, and two end-to-end §8.2 `SymbolDecoder` decodes
  driving the 3-value `TileTx16x16Cdf[ 2 ]` row and the binary
  `TileTxfmSplitCdf[ 2 ]` row selected by the new context helpers.

* **Round 19 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (palette / filter-intra / CFL subset).** Extends `cdf` with the
  filter-intra (`Default_Filter_Intra_Mode_Cdf`,
  `Default_Filter_Intra_Cdf`), palette (`Default_Palette_Y_Mode_Cdf`,
  `Default_Palette_Uv_Mode_Cdf`, `Default_Palette_Y_Size_Cdf`,
  `Default_Palette_Uv_Size_Cdf`, and the fourteen
  `Default_Palette_Size_{2..8}_{Y,Uv}_Color_Cdf` colour-index tables),
  and CFL (`Default_Cfl_Sign_Cdf`, `Default_Cfl_Alpha_Cdf`) default
  tables — all transcribed verbatim from §9.4. New §3 constants
  `BLOCK_SIZES`, `INTRA_FILTER_MODES`, `PALETTE_BLOCK_SIZE_CONTEXTS`,
  `PALETTE_Y_MODE_CONTEXTS`, `PALETTE_UV_MODE_CONTEXTS`,
  `PALETTE_SIZES`, `PALETTE_COLORS`, `PALETTE_COLOR_CONTEXTS`,
  `PALETTE_NUM_NEIGHBORS`, `PALETTE_MAX_COLOR_CONTEXT_HASH`,
  `CFL_JOINT_SIGNS`, `CFL_ALPHABET_SIZE`, `CFL_ALPHA_CONTEXTS`, the
  `PALETTE_COLOR_CONTEXT` / `PALETTE_COLOR_HASH_MULTIPLIERS`
  additional-tables arrays, all listed `DEFAULT_*_CDF` tables, the ten
  `*_cdf` selectors (`filter_intra_cdf`, `filter_intra_mode_cdf`,
  `palette_y_mode_cdf`, `palette_uv_mode_cdf`, `palette_y_size_cdf`,
  `palette_uv_size_cdf`, `palette_y_color_cdf`, `palette_uv_color_cdf`,
  `cfl_sign_cdf`, `cfl_alpha_cdf`), and the five `*_ctx` helpers
  (`palette_y_mode_ctx`, `palette_uv_mode_ctx`, `palette_color_ctx`,
  `cfl_alpha_u_ctx`, `cfl_alpha_v_ctx`). `TileCdfContext::new_from_defaults`
  performs the §8.3.1 init step for every new array. 8 new unit tests
  (190 in src/, up from 182).

* **Round 18 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (inter-mode / reference-frame subset).** Extends `cdf` with the 13
  remaining `Default_*_Cdf` tables that drive every inter-block mode
  and reference syntax: `Default_New_Mv_Cdf`, `Default_Zero_Mv_Cdf`,
  `Default_Ref_Mv_Cdf`, `Default_Drl_Mode_Cdf`, `Default_Is_Inter_Cdf`,
  `Default_Comp_Mode_Cdf`, `Default_Skip_Mode_Cdf`,
  `Default_Comp_Ref_Cdf`, `Default_Comp_Bwd_Ref_Cdf`,
  `Default_Single_Ref_Cdf`, `Default_Compound_Mode_Cdf`,
  `Default_Comp_Ref_Type_Cdf`, `Default_Uni_Comp_Ref_Cdf` — all
  transcribed verbatim from §9.4, plus the §8.3.2
  `Compound_Mode_Ctx_Map[ 3 ][ COMP_NEWMV_CTXS ]` lookup table.
  `TileCdfContext::new_from_defaults` performs the §8.3.1 init step
  ("`*Cdf` is set to a copy of `Default_*_Cdf`") for every new array.
  The §8.3.2 selection surfaces 13 new `&mut [u16]` accessors —
  `new_mv_cdf` / `zero_mv_cdf` / `ref_mv_cdf` / `drl_mode_cdf` /
  `is_inter_cdf` / `comp_mode_cdf` / `skip_mode_cdf` / `comp_ref_cdf` /
  `comp_bwd_ref_cdf` / `single_ref_cdf` / `compound_mode_cdf` /
  `comp_ref_type_cdf` / `uni_comp_ref_cdf` — feeding straight into
  `SymbolDecoder::read_symbol`. Scalar §8.3.2 context helpers
  `is_inter_ctx`, `skip_mode_ctx`, `ref_count_ctx`, and
  `compound_mode_ctx` compute each `ctx` from the neighbour-summary
  inputs the (future) tile walk supplies. New public API:
  `DEFAULT_NEW_MV_CDF`, `DEFAULT_ZERO_MV_CDF`, `DEFAULT_REF_MV_CDF`,
  `DEFAULT_DRL_MODE_CDF`, `DEFAULT_IS_INTER_CDF`, `DEFAULT_COMP_MODE_CDF`,
  `DEFAULT_SKIP_MODE_CDF`, `DEFAULT_COMP_REF_CDF`,
  `DEFAULT_COMP_BWD_REF_CDF`, `DEFAULT_SINGLE_REF_CDF`,
  `DEFAULT_COMPOUND_MODE_CDF`, `DEFAULT_COMP_REF_TYPE_CDF`,
  `DEFAULT_UNI_COMP_REF_CDF`, `COMPOUND_MODE_CTX_MAP`, the 13
  `*_cdf` selectors, the four `*_ctx` helpers, and the §3 constants
  `NEW_MV_CONTEXTS`, `ZERO_MV_CONTEXTS`, `REF_MV_CONTEXTS`,
  `DRL_MODE_CONTEXTS`, `IS_INTER_CONTEXTS`, `COMP_INTER_CONTEXTS`,
  `SKIP_MODE_CONTEXTS`, `REF_CONTEXTS`, `FWD_REFS`, `BWD_REFS`,
  `SINGLE_REFS`, `UNIDIR_COMP_REFS`, `COMP_REF_TYPE_CONTEXTS`,
  `COMPOUND_MODES`, `COMPOUND_MODE_CONTEXTS`, `COMP_NEWMV_CTXS`. The
  remaining ~80 §9.4 tables (y_mode, uv_mode, angle-delta, tx-size,
  coefficient, palette, …) are a mechanical followup against the same
  `TileCdfContext` shape.

  10 new unit tests (172 → 182 in src/): table-dimension audit
  verifying every new `Default_*_Cdf` shape matches the spec literal
  (with the §8.2.6 `cdf[N - 1] == 32768` / `cdf[N] == 0` invariant
  enforced on every row); hand-picked byte-exact spot-checks across
  all 13 tables (every literal that appears at a row boundary read
  back unchanged); §8.3.1 init copies every default into the
  corresponding `Tile*Cdf` slot; §8.3.2 selectors return the right
  default row at every hand-picked `(frame_type, ctx)` tuple — both
  extremes of every `ctx` index for all 13 syntax elements;
  working-copy independence — adapting `new_mv` / `comp_ref` /
  `compound_mode` does not mutate the §9.4 source; §8.3.2
  `is_inter_ctx` branch coverage (all 9 above/left combinations);
  `skip_mode_ctx` (the 4 neighbour-flag pairs); `ref_count_ctx` (the 3
  ordering branches); `compound_mode_ctx` (one spot-check from each of
  the 3 `COMPOUND_MODE_CTX_MAP` rows plus the `Min(.., COMP_NEWMV_CTXS
  - 1)` clamp + the `RefMvContext >> 1` saturation); and an end-to-end
  §8.2 `SymbolDecoder` decode driving the `compound_mode` (8-value)
  default CDF row selected by `compound_mode_ctx(4, 4) = 7`.

* **Round 17 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (motion-vector component subset).** Extends `cdf` with the nine
  `Default_Mv_*_Cdf` tables transcribed verbatim from §9.4
  (`Default_Mv_Joint_Cdf`, `Default_Mv_Sign_Cdf`, `Default_Mv_Class_Cdf`,
  `Default_Mv_Class0_Bit_Cdf`, `Default_Mv_Class0_Fr_Cdf`,
  `Default_Mv_Class0_Hp_Cdf`, `Default_Mv_Bit_Cdf`, `Default_Mv_Fr_Cdf`,
  `Default_Mv_Hp_Cdf` — the `216*128` / `136*128` / … fixed-point
  notation expanded). `TileCdfContext::new_from_defaults` performs the
  §8.3.1 init step ("`Mv*Cdf[ i ]` is set equal to a copy of
  `Default_Mv_*_Cdf` for `i = 0..MV_CONTEXTS-1`"), broadcasting the
  per-`comp` flat defaults to both `comp = 0..1` slots. The §8.3.2
  selection surfaces nine new `&mut [u16]` accessors —
  `mv_joint_cdf(MvCtx)`, `mv_sign_cdf(MvCtx, comp)`,
  `mv_class_cdf(MvCtx, comp)`, `mv_class0_bit_cdf(MvCtx, comp)`,
  `mv_class0_fr_cdf(MvCtx, comp, mv_class0_bit)`,
  `mv_class0_hp_cdf(MvCtx, comp)`, `mv_bit_cdf(MvCtx, comp, i)`,
  `mv_fr_cdf(MvCtx, comp)`, `mv_hp_cdf(MvCtx, comp)` — each yielding
  the row `SymbolDecoder::read_symbol` consumes. The §5.11.31
  `MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0` derivation is exposed
  as the `mv_ctx` helper. New public API: `DEFAULT_MV_*` constants,
  the nine `mv_*_cdf` selectors, `mv_ctx`, and the constants
  `MV_CONTEXTS`, `MV_INTRABC_CONTEXT`, `MV_JOINTS`, `MV_CLASSES`,
  `CLASS0_SIZE`, `MV_OFFSET_BITS`, `MV_COMPS`. The remaining ~90 §9.4
  tables (y_mode, uv_mode, angle-delta, tx-size, coefficient, palette,
  …) are a mechanical followup against the same `TileCdfContext` shape.

  7 new unit tests (165 → 172 in src/): every §9.4 transcribed value
  asserted byte-exact (including the expanded `*128` fixed-point);
  §8.3.1 init copies every default into every `MV_CONTEXTS × MV_COMPS`
  slot with the `cdf[N - 1] == 32768` / `cdf[N] == 0` invariant
  enforced on every row; §5.11.31 `mv_ctx` derivation matches the spec
  (`false → 0`, `true → MV_INTRABC_CONTEXT == 1`); §8.3.2 selectors
  return the right default row for every `(MvCtx, comp, *)` indexing
  variant; working-copy independence — adapting `mv_joint` / `mv_sign`
  / `mv_class0_fr` / `mv_bit` does not mutate `DEFAULT_MV_*`; and two
  end-to-end decodes driving the real `SymbolDecoder` through a
  default CDF — a 4-value `mv_joint` decode that exercises the §8.3
  update path (asserting the counter advances and the §9.4 source is
  left untouched) and a binary `mv_bit` decode with
  `disable_cdf_update == true` (asserting the row stays put in the
  non-adaptive path).

* **Round 16 — §9.4 default CDF tables + §8.3.1 / §8.3.2 selection
  (intra-frame mode / partition subset).** A new `cdf` module
  transcribes the §9.4 `Default_Intra_Frame_Y_Mode_Cdf` (5×5×14), the
  five `Default_Partition_W{8,16,32,64,128}_Cdf` tables (the `partition`
  element), `Default_Skip_Cdf`, and `Default_Segment_Id_Cdf` verbatim,
  every row laid out as the `N + 1` slot
  `[..cumulative.., 1 << 15, 0_counter]` `SymbolDecoder::read_symbol`
  consumes. `TileCdfContext::new_from_defaults` performs the §8.3.1 init
  step ("each `Tile*Cdf` array is set equal to a copy of
  `Default_*_Cdf`"). The §8.3.2 selection surfaces a `&mut [u16]` row
  for each carried element — `intra_frame_y_mode`
  (`[abovemode][leftmode]`), `partition` (array-by-`bsl` /
  row-by-`ctx`), `skip` (`[ctx]`), `segment_id` (`[ctx]`) — fed straight
  to `SymbolDecoder::read_symbol`. Scalar context helpers
  `intra_mode_ctx` / `partition_ctx` / `skip_ctx` / `segment_id_ctx`
  compute the index from the neighbour inputs the (future) tile walk
  supplies. The remaining ~100 §9.4 tables, the `init_coeff_cdfs`
  coefficient set, and the other §8.3.2 selections (`split_or_horz` /
  `split_or_vert` / `tx_depth` / `txfm_split` / motion-vector + uv-mode
  groups) are a mechanical followup against the same `TileCdfContext`
  shape. New public API: `TileCdfContext`, `DEFAULT_*_CDF` constants
  (`DEFAULT_INTRA_FRAME_Y_MODE_CDF`, `DEFAULT_PARTITION_W{8,16,32,64,128}_CDF`,
  `DEFAULT_SKIP_CDF`, `DEFAULT_SEGMENT_ID_CDF`), constants
  (`INTRA_MODES`, `INTRA_MODE_CONTEXTS`, `INTRA_MODE_CONTEXT`,
  `PARTITION_CONTEXTS`, `SKIP_CONTEXTS`, `SEGMENT_ID_CONTEXTS`), and the
  four context-derivation helpers.

  9 new unit tests: §8.3.1 byte-exact copy + the
  `cdf[N - 1] == 32768` / `cdf[N] == 0` invariant on every transcribed
  row; working-copy independence from the immutable §9.4 source;
  `Intra_Mode_Context[]` term-by-term; the `partition_ctx` (`left * 2 +
  above`) / `skip_ctx` (neighbour sum) / `segment_id_ctx` (four-branch)
  formulae; `partition_cdf` selected by `bsl` returning the right row
  lengths and the default-row contents; and two end-to-end decodes
  driving the real `SymbolDecoder` through a default-CDF row (a `skip`
  decode that exercises the §8.3 update path — asserting the counter
  advances and the `Default_*` source is left untouched — and a
  `partition` multisymbol decode with `disable_cdf_update == true`).

* **Round 15 — the §8.2 symbol (arithmetic / msac) decoder.** A new
  standalone `symbol_decoder` module implements the AV1 entropy engine
  end-to-end: §8.2.2 `init_symbol(sz)` (the `numBits = Min(sz*8, 15)`
  window read, `paddedBuf = buf << (15 - numBits)`,
  `SymbolValue = ((1<<15)-1) ^ paddedBuf`, `SymbolRange = 1<<15`,
  `SymbolMaxBits = 8*sz - 15`), §8.2.6 `read_symbol(cdf)` (the
  CDF-adaptive multisymbol search loop using `EC_PROB_SHIFT = 6` /
  `EC_MIN_PROB = 4`, the `SymbolRange = prev - cur` /
  `SymbolValue -= cur` update, and the seven-step renormalisation that
  pulls new bits — or §8.2.2 padding zeros once `SymbolMaxBits` is
  exhausted — via `f(numBits)`), the §8.3 CDF update (the
  `rate = 3 + (cdf[N]>15) + (cdf[N]>31) + Min(FloorLog2(N), 2)`
  adaptive-rate walk plus the `cdf[N]` count-to-32 counter), §8.2.3
  `read_bool()` (the fixed `[1<<14, 1<<15, 0]` boolean CDF, fed fresh
  per call so its adaptation is discarded per the §8.2.3 note), §8.2.5
  `read_literal(n)` (`L(n)`, §4.10.8), `NS(n)` (§4.10.10), the
  arithmetic-coded `decode_subexp_bool(numSyms, k)` (§5.9.28 bool
  variant), and §8.2.4 `exit_symbol()` (the
  `trailingBitPosition = get_position() - Min(15, SymbolMaxBits+15)`
  derivation, the `Max(0, SymbolMaxBits)` trailing-bit advance to the
  byte boundary, returning the `(trailingBitPosition,
  paddingEndPosition)` pair, and rejecting the `SymbolMaxBits < -14`
  conformance violation via a new `Error::SymbolExitUnderflow`).

  The decoder shares the existing MSB-first `BitReader` (§8.1 `f(n)`),
  so its bit-position indicator advances the same `get_position()` the
  rest of the OBU walk uses. Default CDF tables and the §8.3.2
  CDF-selection process are deliberately out of scope — they land with
  the tile-content decode that consumes them. New public API:
  `SymbolDecoder` (`init_symbol` / `read_symbol` / `read_bool` /
  `read_literal` / `read_ns` / `decode_subexp_bool` / `exit_symbol` /
  `position`).

  13 new byte-exact unit tests: §8.2.2 init over a full 15-bit and a
  partial 8-bit window; a hand-traced single §8.2.6 decode against the
  boolean CDF (asserting the decoded symbol, post-decode
  `SymbolValue` / `SymbolRange` / `SymbolMaxBits`, and consumed bit
  position); `read_bool` / `read_literal` composition; the §8.3 update
  computed term-by-term against a 3-symbol CDF; the count-to-32 cap;
  CDF mutation-when-enabled vs untouched-when-`disable_cdf_update`;
  `NS(1)` zero-bit short-circuit; `decode_subexp_bool` immediate
  uniform branch; the §8.2.4 byte-boundary advance + the
  `SymbolMaxBits < -14` underflow rejection; and a past-buffer decode
  that draws §8.2.2 padding zeros instead of erroring.

* **Round 14 — the inter-frame `uncompressed_header()` path.** An
  `INTER_FRAME` / `SWITCH_FRAME` header now parses end-to-end. After
  `refresh_frame_flags` the parser walks the §5.9.2 `else` branch:
  `frame_refs_short_signaling` (`f(1)`, gated on `enable_order_hint`),
  the explicit `ref_frame_idx[]` reads (`f(3)` each, plus the
  `delta_frame_id_minus_1` per-ref reads when frame-id numbering is on),
  the §5.9.7 `frame_size_with_refs()` / §5.9.5 `frame_size()` +
  `render_size()` size selection, `allow_high_precision_mv`, §5.9.10
  `read_interpolation_filter()`, `is_motion_mode_switchable`,
  `use_ref_frame_mvs`, then the shared `disable_frame_end_update_cdf` +
  `tile_info()` + quant / segment / delta-q / delta-lf / loop-filter /
  CDEF / LR / `read_tx_mode()` tail, the inter `frame_reference_mode()`
  (`reference_select` `f(1)`), `skip_mode_params()`,
  `allow_warped_motion`, `reduced_tx_set`, inter `global_motion_params()`,
  and `film_grain_params()`. The §5.9.2 `ref_order_hint` walk
  (error-resilient inter) consumes its bits.

  New: §7.8 `set_frame_refs()` (full ordering: explicit LAST/GOLDEN,
  ALTREF latest-backward, BWDREF/ALTREF2 earliest-backward, the
  `Ref_Frame_List` forward refs, smallest-output-order fallback), §5.9.3
  `get_relative_dist()`, §5.9.7 `frame_size_with_refs()`, and §5.9.22
  `skip_mode_params()`. Backed by a public `RefInfo` cross-frame
  reference state (`RefValid[]` / `RefOrderHint[]` / `RefFrameId[]` +
  per-slot `RefUpscaledWidth[]` / `RefFrameHeight[]` / `RefRenderWidth[]`
  / `RefRenderHeight[]`).

  New public API: `parse_frame_header_with_refs(payload, seq, &RefInfo)`
  (the ref-aware entry point), `RefInfo`, `InterFrameRefs` (surfaced on
  the new `FrameHeader::inter_refs` field —
  `frame_refs_short_signaling` / `last_frame_idx` / `gold_frame_idx` /
  `ref_frame_idx[7]` / `allow_high_precision_mv` / `interpolation_filter`
  / `is_motion_mode_switchable` / `use_ref_frame_mvs`). The existing
  `parse_frame_header()` seeds `RefInfo::default()`.

  Verified byte-exact against the `i-frame-then-p-64x64` fixture's
  `idx=1` `FRAME_HEADER` + `REF_MAP` trace lines (the INTER frame:
  `frame_refs_short_signaling=0`, `ref_frame_idx = [0;7]`,
  `frame_size_override_flag=0` ⇒ `frame_size()`+`render_size()`,
  `order_hint=1`, `base_q_idx=120`, `tx_mode=1`, `reference_select=0`,
  `allow_warped_motion=1`; 134 uncompressed-header bits). Pixel
  reconstruction stays out of scope (`decode_av1` remains
  `Err(NotImplemented)`).

* **Round 13 — the §5.9.2 uncompressed-header tail (`global_motion_params()`
  / `film_grain_params()`) wired into the streaming `parse_frame_header`
  walk.** For intra (`KEY_FRAME` / `INTRA_ONLY_FRAME`) frames the parser
  now descends past `read_tx_mode()` to the end of `uncompressed_header()`:
  `frame_reference_mode()` (§5.9.23), `skip_mode_params()` (§5.9.22), the
  `allow_warped_motion` slot, `reduced_tx_set` (`f(1)`),
  `global_motion_params()` (§5.9.24), and `film_grain_params()` (§5.9.30).

  For an intra frame the §5.9.23 `FrameIsIntra ⇒ reference_select = 0`,
  the §5.9.22 `skipModeAllowed = 0 ⇒ skip_mode_present = 0`, the §5.9.2
  `allow_warped_motion` guard (`FrameIsIntra || error_resilient_mode ||
  !enable_warped_motion`), and the §5.9.24 `FrameIsIntra` identity
  short-circuit all consume no bits; only `reduced_tx_set` (one `f(1)`
  bit) and the `film_grain_params()` block read from the stream.

  New types: `WarpModelType` (a 4-variant §6.8.18 enum
  `Identity/Translation/RotZoom/Affine` with `as_u8()`),
  `GlobalMotionParams` (`gm_type[8]` / `gm_params[8][6]` indexed by
  reference-frame index, `short_circuited`, with an `identity()`
  constructor and `prev_gm_params_default()` helper), `FilmGrainParams`
  (the full §5.9.30 field set — `apply_grain`, `grain_seed`,
  `update_grain`, `film_grain_params_ref_idx`, the Y / Cb / Cr scaling
  points, AR coefficients, `grain_scaling`, `ar_coeff_lag`,
  `ar_coeff_shift`, `grain_scale_shift`, the chroma mult/offset triplets,
  `overlap_flag`, `clip_to_restricted_range`, plus `predicted`, with a
  `reset()` constructor), and `FilmGrainContext` (the §5.5.x / §5.9.2
  inputs). New constants: `REFS_PER_FRAME`, `INTRA_FRAME`, `LAST_FRAME`,
  `ALTREF_FRAME`, `WARPEDMODEL_PREC_BITS`, the six `GM_*` precision/bit
  constants, `MAX_NUM_Y_POINTS`, `MAX_NUM_CHROMA_POINTS`,
  `MAX_AR_COEFFS_Y`, `MAX_AR_COEFFS_UV`.

  The complete §5.9.24/§5.9.25 inter global-motion syntax is implemented
  (`read_global_param` + the §5.9.26–§5.9.29
  `decode_signed_subexp_with_ref` / `decode_unsigned_subexp_with_ref` /
  `decode_subexp` / `inverse_recenter` sub-exponential decoders), exposed
  via the standalone `parse_global_motion_params(payload, frame_is_intra,
  allow_high_precision_mv, prev_gm_params)`; `film_grain_params()` is
  exposed via `parse_film_grain_params(payload, ctx)`. New fields on
  `FrameHeader`: `reference_select` / `skip_mode_present` /
  `allow_warped_motion` / `reduced_tx_set` (`Option<bool>`),
  `global_motion_params: Option<GlobalMotionParams>`,
  `film_grain_params: Option<FilmGrainParams>` (`Some` for intra frames,
  `None` for inter / show-existing replays).

  Validation: 14 new unit tests (`WarpModelType` symbol values; the
  §5.9.24 identity defaults; the intra global-motion no-bits
  short-circuit; an inter all-IDENTITY 7-bit walk; an inter
  single-TRANSLATION subexp decode; global-motion unexpected-end; the
  three §5.9.30 short-circuits — `!present`, hidden frame, `apply_grain =
  0`; the INTER predicted `update_grain = 0` path; the 4:2:0 chroma
  suppression branch; a full luma + chroma + AR-coeff path; film-grain
  unexpected-end). The 16-fixture frame-header integration test now
  asserts the new tail columns (`reference_select` / `skip_mode_present`
  / `allow_warped_motion` / `reduced_tx_set` = 0, global-motion identity)
  on every fixture, and the `film-grain-on` fixture's full 718-byte FRAME
  OBU payload is embedded so its `apply_grain = 1` `film_grain_params()`
  (14 Y points, 8 Cb + 9 Cr points, `ar_coeff_lag = 2`, `seed = 45231`,
  `scaling_minus_8 = 11`, `clip_restricted = 1`) is validated byte-exact
  against the fixture trace. The `parses_tiny_key_frame_prefix`
  `bits_consumed` rises from 71 to 72 (one `reduced_tx_set` bit;
  `film_grain_params_present = 0` ⇒ film grain resets).

* **Round 12 — `read_tx_mode()` (§5.9.21) wired into the streaming
  `parse_frame_header` walk.** For intra (`KEY_FRAME` /
  `INTRA_ONLY_FRAME`) frames the parser now descends past `lr_params()`
  into `read_tx_mode()`. When `CodedLossless == 1` the §5.9.21 first
  branch consumes no bits and forces `TxMode = ONLY_4X4`; otherwise the
  `f(1)` `tx_mode_select` slot selects `TX_MODE_SELECT` (`1`) or
  `TX_MODE_LARGEST` (`0`). `CodedLossless` is the same value already
  derived in-module for the §5.9.11 / §5.9.19 short-circuits.

  New type `TxMode` (a 3-variant enum with §6.8.21 symbol-value
  discriminants `Only4x4 = 0, TxModeLargest = 1, TxModeSelect = 2` plus
  an `as_u8()` accessor). New constant `TX_MODES = 3`. New standalone
  parser entry point `parse_tx_mode(payload, coded_lossless) ->
  (TxMode, usize)`. New field on `FrameHeader`: `tx_mode: Option<TxMode>`
  (`Some` for intra frames, `None` for inter / show-existing replays).
  Wired into both intra paths (reduced-still and non-reduced).

  Validation: 6 new unit tests — the §6.8.21 symbol values + `TX_MODES`
  count, the `CodedLossless == 1 ⇒ ONLY_4X4` no-bits-read path (twice:
  empty buffer and a buffer whose bit is ignored), `tx_mode_select = 1 ⇒
  TX_MODE_SELECT`, `tx_mode_select = 0 ⇒ TX_MODE_LARGEST`, and the
  unexpected-end case. The 16-fixture frame-header integration test
  gains one new asserted trace column (`tx_mode` from each fixture's
  `FRAME_HEADER` trace line, compared against the parsed `TxMode`'s
  §6.8.21 symbol value) plus a `ONLY_4X4 ⇒ CodedLossless` invariant
  (only `lossless-i-only` is CodedLossless). The corpus exercises all
  three values: `tx_mode = 0` (`lossless-i-only`, the no-bits path),
  `tx_mode = 1` (`tiny-i-only-16x16-prof0`, `monochrome-grey-only`,
  `profile-1-yuv444-8bit`, `profile-2-yuv422-12bit`), and `tx_mode = 2`
  (the other 11). The `parses_tiny_key_frame_prefix` `bits_consumed`
  assertion rises from 70 to 71 (one `tx_mode_select` bit for the
  non-lossless `tiny-i-only`).

* **Round 11 — `lr_params()` (§5.9.20) wired into the streaming
  `parse_frame_header` walk.** For intra (`KEY_FRAME` /
  `INTRA_ONLY_FRAME`) frames the parser now descends past
  `cdef_params()` into `lr_params()`. `AllLossless` is derived per the
  §5.9.2 line `AllLossless = CodedLossless && (FrameWidth ==
  UpscaledWidth)` (so a super-resolution-downscaled lossless frame is
  *not* AllLossless and still walks the full LR path). The §5.9.20
  `AllLossless || allow_intrabc || !enable_restoration` short-circuit
  consumes no bits and leaves every plane `RESTORE_NONE` with
  `UsesLr = 0` and zero `LoopRestorationSize[]`. The full path reads
  one `lr_type` (`f(2)`) per plane (`NumPlanes` of them), mapping each
  through `Remap_Lr_Type[4] = { RESTORE_NONE, RESTORE_SWITCHABLE,
  RESTORE_WIENER, RESTORE_SGRPROJ }`; when any plane uses LR, the
  parser then reads `lr_unit_shift` (`f(1)`, post-incremented for
  128×128 superblocks, otherwise extended by `lr_unit_extra_shift`
  `f(1)` when the first bit is set) and — when `subsampling_x &&
  subsampling_y && usesChromaLr` (4:2:0 chroma LR) — `lr_uv_shift`
  (`f(1)`). The three `LoopRestorationSize[]` entries derive from
  `RESTORATION_TILESIZE_MAX = 256` via `>> (2 - lr_unit_shift)` for
  luma and `>> lr_uv_shift` for chroma.

  New types `LrParams` (`frame_restoration_type[3]`, `uses_lr`,
  `uses_chroma_lr`, `lr_unit_shift`, `lr_uv_shift`,
  `loop_restoration_size[3]`, `short_circuited`) and
  `FrameRestorationType` (a 4-variant enum with §6.10.15 symbol-value
  discriminants `None = 0, Wiener = 1, SgrProj = 2, Switchable = 3`
  plus a `remap(lr_type)` constructor that walks `Remap_Lr_Type`). New
  constant `RESTORATION_TILESIZE_MAX = 256`. New standalone parser
  entry point `parse_lr_params`. New field on `FrameHeader`:
  `lr_params: Option<LrParams>` (`Some` for intra frames, `None` for
  inter / show-existing replays). Wired into both intra paths
  (reduced-still and non-reduced).

  Validation: 19 new unit tests — short-circuit on each of the three
  gate conditions (AllLossless / allow_intrabc / !enable_restoration),
  `Remap_Lr_Type` table coverage, the UsesLr=0 path (only types read,
  no shift bits), non-128×128 superblock with `lr_unit_shift` in each
  of {0, 1, 2}, 128×128 superblock post-increment giving shifts {1, 2},
  4:2:0 chroma LR uv-shift read with 0 and 1 outcomes, the
  subsampling-gating short-circuits for 4:4:4 and 4:2:2 chroma LR,
  monochrome (`NumPlanes == 1`) only reading one type, all-three
  distinct types, and two unexpected-end variants (at the first type
  and partway through unit-shift reading). The 16-fixture frame-header
  integration test gains five new asserted trace columns (`y_type`,
  `u_type`, `v_type`, `unit_shift`, `uv_shift` from each fixture's
  `LOOP_RESTORATION idx=0` trace line) plus a `UsesLr` cross-check, a
  short-circuit invariant (only `lossless-i-only` is AllLossless), and
  a `LoopRestorationSize[0]` derivation cross-check. The trace's
  `y_type` / `u_type` / `v_type` columns were empirically confirmed to
  log the **raw bitstream `lr_type`** (`f(2)`, 0..=3) rather than the
  post-`Remap_Lr_Type` `FrameRestorationType` symbol that the
  fixture-doc legend "0=NONE, 1=WIENER, 2=SGRPROJ, 3=SWITCHABLE"
  suggests; the integration test routes the trace value through
  `Remap_Lr_Type` before comparing. The
  `parses_tiny_key_frame_prefix` `bits_consumed` assertion rises from
  64 to 70 (the §5.9.20 full path reads 3 × `f(2)` = 6 bits when all
  three planes resolve to `RESTORE_NONE`, so no shift bits follow).

* **Round 10 — `cdef_params()` (§5.9.19) wired into the streaming
  `parse_frame_header` walk.** For intra (`KEY_FRAME` /
  `INTRA_ONLY_FRAME`) frames the parser now descends past
  `loop_filter_params()` into `cdef_params()`. The §5.9.19
  `CodedLossless || allow_intrabc || !enable_cdef` short-circuit consumes
  no bits and leaves `cdef_bits = 0`, `CdefDamping = 3`, and all four
  strength arrays at their index-0 zero defaults. The full path reads
  `cdef_damping_minus_3` (`f(2)`, `CdefDamping = cdef_damping_minus_3 +
  3`), `cdef_bits` (`f(2)`), and then for each of the `1 << cdef_bits`
  entries `cdef_y_pri_strength[i]` (`f(4)`) / `cdef_y_sec_strength[i]`
  (`f(2)`) and — when `NumPlanes > 1` — `cdef_uv_pri_strength[i]`
  (`f(4)`) / `cdef_uv_sec_strength[i]` (`f(2)`). The §5.9.19 secondary
  `== 3 ⇒ += 1` adjustment (raw `3` stored as `4`) is applied to both Y
  and UV secondary strengths.

  New type `CdefParams` (`cdef_damping`, `cdef_bits`, the four
  `cdef_*_strength` arrays, `short_circuited`) with a `short_circuit()`
  constructor. New constant `CDEF_MAX_STRENGTHS = 8`. New standalone
  parser entry point `parse_cdef_params`. New field on `FrameHeader`:
  `cdef_params: Option<CdefParams>` (`Some` for intra frames, `None` for
  inter / show-existing replays). Wired into both intra paths
  (reduced-still and non-reduced).

  Validation: 8 new unit tests — short-circuit on each of the three gate
  conditions (CodedLossless / allow_intrabc / !enable_cdef), full-path
  single-entry 3-plane decode, the `sec == 3 ⇒ 4` adjustment for both Y
  and UV, monochrome (`NumPlanes == 1`) chroma-skip, the 8-entry
  (`cdef_bits = 3`) loop bound, and unexpected-end. The 16-fixture
  frame-header integration test gains six new asserted trace columns
  (`cdef_damping`, `cdef_bits`, `cdef_y_pri_strength[0]`,
  `cdef_uv_pri_strength[0]`, `cdef_y_sec_strength[0]`,
  `cdef_uv_sec_strength[0]`) sourced from each fixture's `CDEF idx=0`
  trace line, plus a short-circuit invariant (`lossless-i-only`
  CodedLossless and `screen-content-tools` `enable_cdef=0` short-circuit;
  the other 14 take the full path). The `CDEF` trace lines were
  empirically confirmed to log the **raw** pre-adjustment secondary
  strength (values of `3` appear in the trace, which the parser stores
  as `4`); the integration test maps the raw expectation through the
  adjustment. The `parses_tiny_key_frame_prefix` `bits_consumed`
  assertion rises from 48 to 64 (the §5.9.19 full path reads 2 + 2 + 16
  = 16 bits for one 3-plane entry).

* **Round 9 — `loop_filter_params()` (§5.9.11) wired into the streaming
  `parse_frame_header` walk.** For intra (`KEY_FRAME` /
  `INTRA_ONLY_FRAME`) frames the parser now descends past
  `delta_lf_params()` into the §5.9.2 lines that derive `CodedLossless`
  and then `loop_filter_params()`. `CodedLossless` is computed by
  scanning `LosslessArray[]` over the eight per-segment qindexes:
  `get_qindex(1, segmentId)` (the §8.7 quantiser-index function with
  `ignoreDeltaQ == 1`) returns `base_q_idx`, or — when
  `seg_feature_active_idx(segmentId, SEG_LVL_ALT_Q)` is set —
  `Clip3(0, 255, base_q_idx + FeatureData[segmentId][SEG_LVL_ALT_Q])`;
  a segment is lossless when its qindex is 0 and all five §5.9.12
  `DeltaQ?*` offsets are 0. The §5.9.11 `CodedLossless || allow_intrabc`
  short-circuit consumes no bits and resets `loop_filter_ref_deltas` to
  the spec defaults; the full path reads the four `loop_filter_level[]`
  slots (chroma pair `[2]`/`[3]` gated on `NumPlanes > 1 &&
  (loop_filter_level[0] || loop_filter_level[1])`), the `f(3)`
  `loop_filter_sharpness`, and the `loop_filter_delta_enabled` /
  `loop_filter_delta_update` per-slot update walk over
  `TOTAL_REFS_PER_FRAME` ref-deltas + 2 mode-deltas. The
  `loop_filter_params()` routine itself was already implemented
  standalone in round 5 (`parse_loop_filter_params`); this round adds
  the streaming wire-in plus the `compute_coded_lossless` derivation.

  New field on `FrameHeader`: `loop_filter_params: Option<LoopFilterParams>`
  (`Some` for intra frames, `None` for inter / show-existing replays).

  Validation: 6 new unit tests — 5 for `compute_coded_lossless`
  (base_q_idx=0 + no deltas + seg-off ⇒ lossless / base_q_idx≠0 ⇒ not
  lossless / any non-zero `DeltaQ?*` ⇒ not lossless / per-segment
  `SEG_LVL_ALT_Q` clamp to 0 across all 8 segments ⇒ lossless /
  `SEG_LVL_ALT_Q` ignored when `segmentation_enabled == 0`), and 1
  streaming full-path test asserting non-zero `loop_filter_level[0,2,3]`
  + sharpness. The `segmentation_streaming_synthetic_alt_q_active` test
  gains a short-circuit assertion (its `SEG_LVL_ALT_Q = -123` clamps
  every qindex to 0 ⇒ `CodedLossless = 1`). The
  `parses_tiny_key_frame_prefix` `bits_consumed` assertion rises from 31
  to 48 (the §5.9.11 full path reads
  `loop_filter_level[0]`(6) + `[1]`(6) + sharpness(3) +
  delta_enabled(1) + delta_update(1) = 17 bits). The 16-fixture
  frame-header integration test gains five new asserted trace columns
  (`lf_y`, `lf_uv0`, `lf_uv1`, `lf_sharp`, `lf_delta_enabled`) mapped to
  `loop_filter_level[0, 2, 3]` / `loop_filter_sharpness` /
  `loop_filter_delta_enabled` per §6.8.10; the `lossless-i-only` fixture
  (`base_q_idx = 0`, `lf_delta_enabled = 0`) exercises the §5.9.11
  short-circuit and confirms `CodedLossless` is derived correctly,
  while the other 15 fixtures exercise the full bitstream path (several
  with non-zero chroma loop-filter levels, e.g. `film-grain-on`
  `lf_y=4 / lf_uv0=14 / lf_uv1=11`).

  Followups: §5.9.19 `cdef_params()`, §5.9.20 `lr_params()`, §5.9.21
  `read_tx_mode()`, §5.9.23 `frame_reference_mode()`. After those, the
  streaming `parse_frame_header` walk reaches `skip_mode_params()` /
  `global_motion_params()` / `film_grain_params()`.

* **Round 8 — `delta_q_params()` (§5.9.17) + `delta_lf_params()`
  (§5.9.18) wired into the streaming `parse_frame_header` walk.** For
  intra (`KEY_FRAME` / `INTRA_ONLY_FRAME`) frames the parser now
  descends past `segmentation_params()` into `delta_q_params()`: the
  `delta_q_present` `f(1)` slot is read only when `base_q_idx > 0`
  (otherwise the §5.9.17 `delta_q_present = 0` initialiser stands, no
  bit consumed), and `delta_q_res` (`f(2)`) follows only when
  `delta_q_present == 1`. Then `delta_lf_params()`: the whole block is
  gated on `delta_q_present`, the `delta_lf_present` `f(1)` slot is
  suppressed when `allow_intrabc == 1`, and `delta_lf_res` (`f(2)`) +
  `delta_lf_multi` (`f(1)`) follow only when `delta_lf_present == 1`.

  New types `DeltaQParams { delta_q_present, delta_q_res }` and
  `DeltaLfParams { delta_lf_present, delta_lf_res, delta_lf_multi }`.
  Two new fields on `FrameHeader`: `delta_q_params:
  Option<DeltaQParams>` and `delta_lf_params: Option<DeltaLfParams>`
  (both `Some` for intra frames, `None` for inter / show-existing
  replays). New standalone parser entry points
  `parse_delta_q_params(payload, base_q_idx) -> (DeltaQParams, usize)`
  and `parse_delta_lf_params(payload, delta_q_present, allow_intrabc)
  -> (DeltaLfParams, usize)`.

  Validation: 9 new unit tests (3 for `delta_q_params` —
  `base_q_idx == 0` reads nothing / `delta_q_present == 0` 1-bit /
  `delta_q_present == 1` reads `delta_q_res` — plus an unexpected-end;
  5 for `delta_lf_params` — gated off when `delta_q_present == 0` /
  `delta_lf_present == 0` 1-bit / full path reading `delta_lf_res` +
  `delta_lf_multi` / suppressed by `allow_intrabc` / unexpected-end).
  The 16-fixture frame-header integration test gains two new asserted
  trace columns (`delta_q_present`, `delta_lf_present`) plus
  `delta_q_res = 0` / `delta_lf_res = 0` / `delta_lf_multi = false`
  invariant guards (every corpus fixture is `delta_q_present=0` /
  `delta_lf_present=0`; `lossless-i-only` has `base_q_idx=0` so it
  exercises the §5.9.17 no-read branch). The `parses_tiny_key_frame_
  prefix` unit-test bit-count rises from 30 to 31 (one extra
  `delta_q_present` bit for `base_q_idx=120`).

  Followups: §5.9.11 `loop_filter_params()` (full streaming wire-in;
  short-circuit `CodedLossless || allow_intrabc` already modelled
  standalone), §5.9.19 `cdef_params()`, §5.9.20 `lr_params()`,
  §5.9.21 `read_tx_mode()`, §5.9.23 `frame_reference_mode()`.

* **Round 7 — `quantization_params()` (§5.9.12) + `segmentation_params()`
  (§5.9.14) wired into the streaming `parse_frame_header` walk.** For
  intra (`KEY_FRAME` / `INTRA_ONLY_FRAME`) frames the parser now
  descends past `tile_info()` into `quantization_params()` (already
  implemented standalone in round 5) and then into the new
  `segmentation_params()` routine: `segmentation_enabled` (`f(1)`),
  then — when `primary_ref_frame == PRIMARY_REF_NONE` — the three
  update flags collapse to `update_map=1` / `temporal_update=0` /
  `update_data=1` with no bitstream reads; otherwise the three flags
  are read (`update_map` always, `temporal_update` only when
  `update_map=1`, `update_data` always). When `update_data=1` the
  inner loop walks all 8 × 8 = 64 `feature_enabled` bits and, for each
  active feature, reads `su(1 + Segmentation_Feature_Bits[j])` (signed
  features 0..=4) or `f(Segmentation_Feature_Bits[j])` (unsigned
  feature 5) and clips against `Segmentation_Feature_Max[j]`. The
  §5.9.14 trailing `SegIdPreSkip` / `LastActiveSegId` derivations are
  computed.

  New type `SegmentationParams { enabled, update_map, temporal_update,
  update_data, segment_feature_active: [[bool; SEG_LVL_MAX];
  MAX_SEGMENTS], segment_feature_data: [[i16; SEG_LVL_MAX];
  MAX_SEGMENTS], seg_id_pre_skip, last_active_seg_id }`. Two new
  fields on `FrameHeader`: `quantization_params:
  Option<QuantizationParams>` and `segmentation_params:
  Option<SegmentationParams>` (both `Some` for intra frames, `None`
  for inter / show-existing replays). New §3 constants: `MAX_SEGMENTS
  = 8`, `SEG_LVL_MAX = 8`, `SEG_LVL_ALT_Q = 0`, `SEG_LVL_ALT_LF_Y_V =
  1`, `SEG_LVL_ALT_LF_Y_H = 2`, `SEG_LVL_ALT_LF_U = 3`,
  `SEG_LVL_ALT_LF_V = 4`, `SEG_LVL_REF_FRAME = 5`, `SEG_LVL_SKIP = 6`,
  `SEG_LVL_GLOBALMV = 7`, `MAX_LOOP_FILTER = 63`. Three Table 5.9.14
  tables also exposed: `SEGMENTATION_FEATURE_BITS = [8, 6, 6, 6, 6, 3,
  0, 0]`, `SEGMENTATION_FEATURE_SIGNED = [1, 1, 1, 1, 1, 0, 0, 0]`,
  `SEGMENTATION_FEATURE_MAX = [255, 63, 63, 63, 63, 7, 0, 0]`. New
  standalone parser entry point `parse_segmentation_params(payload,
  primary_ref_frame) -> (SegmentationParams, usize)`.

  Validation: 9 new unit tests for the standalone
  `parse_segmentation_params` (disabled / `PRIMARY_REF_NONE` collapse
  with all-inactive features / primary-ref three-bit update walk /
  `update_map=0` skips `temporal_update` / signed-feature `su(9)`
  with `SEG_LVL_ALT_Q` value `-50` / signed-feature clipped at the
  `-255` floor when reading `feature_value = -256` / unsigned
  `SEG_LVL_REF_FRAME` `f(3)=6` setting `seg_id_pre_skip=1` /
  `SEG_LVL_SKIP` zero-width with `last_active_seg_id=3` /
  unexpected-end), 1 new streaming-parser synthetic
  (`segmentation_enabled=1` with `SEG_LVL_ALT_Q` active value `-123`),
  and the 16-fixture frame-header integration test gains two new
  asserted trace columns (`base_q_idx`, `seg_enabled`) plus a
  `SegIdPreSkip = 0` / `LastActiveSegId = 0` invariant guard (every
  corpus fixture is `seg_enabled=0`).

  Followups: §5.9.15 `delta_q_params()`, §5.9.16 `delta_lf_params()`,
  §5.9.11 `loop_filter_params()` (full streaming wire-in;
  short-circuit `CodedLossless || allow_intrabc` already modelled
  standalone), §5.9.17 `cdef_params()`, §5.9.20 `lr_params()`. After
  those, the streaming `parse_frame_header` walk reaches
  `read_tx_mode()`.

* **Round 6 — `allow_intrabc` (§5.9.3) +
  `disable_frame_end_update_cdf` + `tile_info()` (§5.9.15) wired
  into the streaming `parse_frame_header` walk.** For intra
  (`KEY_FRAME` / `INTRA_ONLY_FRAME`) frames whose
  `allow_screen_content_tools && UpscaledWidth == FrameWidth`
  conjunction holds, the parser now consumes the §5.9.3 `f(1)`
  `allow_intrabc` slot — otherwise the §5.9.2 `allow_intrabc = 0`
  initialiser stands. The `disable_frame_end_update_cdf` `f(1)`
  bit is consumed next (gated off `reduced_still_picture_header ||
  disable_cdf_update`). Finally `tile_info()` per §5.9.15 walks
  the per-frame tile layout via either the uniform-spacing path
  (`increment_tile_cols_log2` / `increment_tile_rows_log2` loops
  capped at `tile_log2(1, min(sbCols, MAX_TILE_COLS))` /
  `tile_log2(1, min(sbRows, MAX_TILE_ROWS))`) or the non-uniform
  path (`ns(maxWidth)` / `ns(maxHeight)` `width_in_sbs_minus_1` /
  `height_in_sbs_minus_1` reads). The
  `context_update_tile_id` (`f(TileColsLog2 + TileRowsLog2)`) +
  `tile_size_bytes_minus_1` (`f(2)`) trailing reads are gated by
  `TileColsLog2 > 0 || TileRowsLog2 > 0`. Three new fields on
  `FrameHeader`: `allow_intrabc`, `disable_frame_end_update_cdf`,
  `tile_info: Option<TileInfo>`.

  New module: `tile_info`. New public types:
  `TileInfo { uniform_tile_spacing_flag, tile_cols, tile_rows,
  tile_cols_log2, tile_rows_log2, context_update_tile_id,
  tile_size_bytes, mi_col_starts, mi_row_starts }`. New
  standalone entry point: `parse_tile_info(payload, mi_cols,
  mi_rows, use_128x128_superblock) -> (TileInfo, usize)`. New
  public constants from §3: `MAX_TILE_WIDTH = 4096`,
  `MAX_TILE_AREA = 4096 * 2304`, `MAX_TILE_ROWS = 64`,
  `MAX_TILE_COLS = 64`. New internal bitreader primitive:
  `BitReader::ns(n)` per §4.10.7 — the non-symmetric unsigned
  descriptor used for the non-uniform-spacing
  `width_in_sbs_minus_1` / `height_in_sbs_minus_1` reads.

  Because the §5.9.2 syntax tree carries
  `disable_frame_end_update_cdf` between `allow_intrabc` and
  `tile_info()`, the streaming parser also consumes that bit (and
  the `FrameHeader::disable_frame_end_update_cdf` field is now
  surfaced). For inter frames + show-existing-frame replays the
  parser still stops at `refresh_frame_flags` (the
  `frame_size_with_refs()` / `ref_frame_idx[]` walks remain
  un-modelled), so `tile_info` is `None` in those cases.

  `FrameHeader` is no longer `Copy` (the `TileInfo` arrays make
  it `!Copy`); it remains `Clone + PartialEq + Eq`.

  Validation: 11 new unit tests (7 for `tile_info` standalone
  including `tile_log2` table, 16×16 single-tile uniform / 256×64
  two-column uniform / 64×64 single-superblock / 128×128 with
  use_128x128_superblock=1 / non-uniform two-column / truncated
  payload), 3 for the `BitReader::ns(n)` descriptor (n=1, n=5
  table check, n=power-of-two collapse), and 2 for the
  streaming-parser integration (`allow_intrabc = 1` via the
  screen-content seq, `context_update_tile_id` read when
  `TileColsLog2 + TileRowsLog2 > 0`). The 16-fixture frame-header
  integration test gains four new asserted trace columns
  (`allow_intrabc`, `tile_cols`, `tile_rows`,
  `context_update_tile_id`) plus the `MAX_TILE_COLS` /
  `MAX_TILE_ROWS` conformance guard from §6.8.14. The
  `tile-cols-2-rows-1` fixture exercises a real 2-tile layout
  (`TileColsLog2 = 1`, `TileSizeBytes` read).

* **Round 5 — Uncompressed-header tail sub-syntaxes (§5.9.10 /
  §5.9.11 / §5.9.12 / §5.9.13).** New `uncompressed_header_tail`
  module exposes three standalone parser entry points that take a
  byte slice + the relevant `SequenceHeader`-derived flags and
  return a parsed descriptor:

  * `parse_interpolation_filter(payload) -> (InterpolationFilter,
    usize)` — §5.9.10. Reads `is_filter_switchable` (`f(1)`) +
    optional `interpolation_filter` (`f(2)`), returning the
    `InterpolationFilter` enum (`Eighttap` / `EighttapSmooth` /
    `EighttapSharp` / `Bilinear` / `Switchable`) per §6.8.9.

  * `parse_loop_filter_params(payload, num_planes, coded_lossless,
    allow_intrabc) -> (LoopFilterParams, usize)` — §5.9.11. Honours
    the `(CodedLossless || allow_intrabc)` short-circuit (no bits
    read, `loop_filter_ref_deltas` reset to the spec's literal
    defaults `[INTRA=1, LAST=0, LAST2=0, LAST3=0, GOLDEN=-1,
    BWDREF=0, ALTREF2=-1, ALTREF=-1]`). For the full path: four
    `loop_filter_level[]` `f(6)` slots (with the `NumPlanes > 1 &&
    (loop_filter_level[0] || loop_filter_level[1])` gate on the
    chroma pair), `loop_filter_sharpness` (`f(3)`),
    `loop_filter_delta_enabled` (`f(1)`), `loop_filter_delta_update`
    (`f(1)`), and the per-slot update walk: for each of
    `TOTAL_REFS_PER_FRAME = 8` ref-deltas an `update_ref_delta`
    (`f(1)`) gate that conditionally reads `loop_filter_ref_deltas[i]`
    as `su(7)`, then the same pattern for the 2 mode-deltas.

  * `parse_quantization_params(payload, num_planes,
    separate_uv_delta_q) -> (QuantizationParams, usize)` — §5.9.12
    + §5.9.13. Reads `base_q_idx` (`f(8)`), `DeltaQYDc` via
    `read_delta_q()` (a `delta_coded` `f(1)` gate followed by a
    conditional `su(1+6) = su(7)` signed offset), the chroma block
    (`diff_uv_delta` `f(1)` only when `NumPlanes > 1 &&
    separate_uv_delta_q`, `DeltaQUDc` / `DeltaQUAc` via
    `read_delta_q()` when `NumPlanes > 1`, V mirrors U when
    `diff_uv_delta == 0`), and the qmatrix block (`using_qmatrix`
    `f(1)` plus `qm_y` / `qm_u` / `qm_v` `f(4)` each, where `qm_v`
    is read separately only when `separate_uv_delta_q == 1`).

  New types: `InterpolationFilter` enum + `LoopFilterParams` /
  `QuantizationParams` structs. New constants:
  `TOTAL_REFS_PER_FRAME = 8`, `LOOP_FILTER_REF_DELTAS_DEFAULT`,
  `LOOP_FILTER_MODE_DELTAS_DEFAULT`. New bitreader primitive:
  internal `BitReader::su(n)` per §4.10.6, the signed-integer
  descriptor used by `loop_filter_ref_deltas[i]` /
  `loop_filter_mode_deltas[i]` / the `delta_q` field of
  `read_delta_q()`.

  The three sub-syntaxes are exposed as **standalone** parser
  entry points rather than wired into the streaming
  `parse_frame_header` walk: the intervening §5.9.2 syntax
  (`allow_intrabc`, `disable_frame_end_update_cdf`, `tile_info()`,
  `segmentation_params()`, `delta_q_params()`, `delta_lf_params()`)
  sits between round 4's stop point and these calls. The next
  round can stitch them into the streaming parser as the
  intervening syntaxes land.

  Validation: 18 new unit tests across the three sub-syntaxes —
  switchable + each of the four non-switchable interpolation
  filters + truncated-input + raw-roundtrip for §5.9.10; the
  `CodedLossless` short-circuit + the `allow_intrabc` short-circuit
  + full-path-levels-only + 3-plane chroma-level gating + mono
  skip-plane-2/3 + delta-update walk with sparse updates for
  §5.9.11; mono + 3-plane non-separate + 3-plane separate with
  `diff_uv_delta = 1` + `using_qmatrix` with V-mirrors-U +
  truncated-input for §5.9.12. Plus 3 new `BitReader::su(n)` tests
  (positive / negative / minimum negative). Total bitreader tests
  10 → 13, total crate tests 36 → 57.

* **Round 4 — Frame-size sub-syntax block (§5.9.5–§5.9.9).** The
  `parse_frame_header()` parser is extended past `refresh_frame_flags`
  to consume the four §5.9 frame-size sub-syntaxes in spec order:
  `frame_size()` (§5.9.5) reads `frame_width_minus_1` /
  `frame_height_minus_1` (with bit widths from §5.5.1's
  `frame_width_bits_minus_1` / `frame_height_bits_minus_1`) when
  `frame_size_override_flag == 1`, otherwise it falls back to the
  sequence header's `max_frame_width_minus_1 + 1` /
  `max_frame_height_minus_1 + 1`; `superres_params()` (§5.9.8) reads
  `use_superres` + `coded_denom` (gated by `enable_superres`),
  computes `SuperresDenom = coded_denom + SUPERRES_DENOM_MIN` (or
  `SUPERRES_NUM` when superres is off), assigns
  `UpscaledWidth = FrameWidth`, and applies the rounded-half-up
  downscale `FrameWidth = (UpscaledWidth * SUPERRES_NUM +
  SuperresDenom / 2) / SuperresDenom`; `compute_image_size()` (§5.9.9)
  derives `MiCols = 2 * ((FrameWidth + 7) >> 3)` and
  `MiRows = 2 * ((FrameHeight + 7) >> 3)` (the §3 `MI_SIZE = 4` block
  grid); `render_size()` (§5.9.6) reads
  `render_and_frame_size_different`, optional 16-bit
  `render_width_minus_1` / `render_height_minus_1`, and defaults
  `RenderWidth = UpscaledWidth` / `RenderHeight = FrameHeight` per
  §6.8.5.

  Surfaces a new [`FrameSize`] struct with the eight requested
  fields (`frame_width`, `frame_height`, `render_width`,
  `render_height`, `superres_denom`, `upscaled_width`, `mi_cols`,
  `mi_rows`) plus the three sub-syntax-input fields (`use_superres`,
  `coded_denom`, `render_and_frame_size_different`) and a
  convenience `is_super_resolved()` predicate. [`FrameHeader`] now
  carries an `Option<FrameSize>` populated for every intra (`KEY` /
  `INTRA_ONLY`) frame; inter frames keep `frame_size = None` for
  this round because the §5.9.7 `frame_size_with_refs()`
  `found_ref == 1` branch reads `RefUpscaledWidth[]` /
  `RefFrameHeight[]` / `RefRenderWidth[]` / `RefRenderHeight[]`
  from a reference-frame state table not yet tracked across calls.

  New `SUPERRES_NUM = 8` / `SUPERRES_DENOM_MIN = 9` /
  `SUPERRES_DENOM_BITS = 3` constants from §3 of the AV1
  Bitstream & Decoding Process Specification. New
  `Error::RefOrderHintWalkUnsupported` variant surfaces the §5.9.2
  `error_resilient_mode && enable_order_hint` ref_order_hint walk
  that requires per-slot `RefOrderHint[]` / `RefValid[]` state
  (no fixture in the current corpus exercises it).

  Validation: four new unit tests cover the explicit-render-size
  branch (`render_and_frame_size_different == 1` with non-default
  `render_width` / `render_height`), the
  `frame_size_override_flag == 1` branch (reads
  `frame_width_minus_1` / `frame_height_minus_1` against
  `frame_width_bits_minus_1` / `frame_height_bits_minus_1`), the
  `use_superres == 1` branch with `coded_denom == 3` (asserts
  `SuperresDenom = 12`, post-downscale `FrameWidth = 85`, `MiCols
  = 22` against the spec's rounded-half-up formula), and the
  `enable_superres == 1` + `use_superres == 0` reduced-still
  case. Existing unit tests grow to assert the new
  [`FrameHeader::frame_size`] field on the two real-OBU fixtures
  (tiny-i-only-16x16 / show-existing-frame underlying KEY) and the
  two synthetic reduced-still vectors. The integration test
  (`tests/frame_header_fixtures.rs`) is extended with five new
  trace columns per fixture — `trace_w`, `trace_h`,
  `use_superres`, `coded_denom`, and a derived assertion ladder
  computing the expected `superres_denom` / post-superres
  `frame_width` / `mi_cols` / `mi_rows` against the §5.9.5–§5.9.9
  formulas — so all 16 fixtures cross-validate eight
  [`FrameSize`] fields against the `FRAME_HEADER` trace line, and
  the round 3 12-column assertions still pass byte-exact (now
  17 × 16 = 272 field assertions per run of the integration
  test).

* **Round 3 — Uncompressed-header prefix parse (§5.9.2).** New
  `frame_header` module implements `parse_frame_header()` consuming
  the leading slice of `uncompressed_header()` per §5.9.2 of the AV1
  Bitstream & Decoding Process Specification. The slice covers
  `show_existing_frame` (with the show-existing replay branch fully
  modelled — `frame_to_show_map_idx`, `display_frame_id`), the
  `frame_type` enum (`KEY_FRAME` / `INTER_FRAME` / `INTRA_ONLY_FRAME` /
  `SWITCH_FRAME`) with derived `FrameIsIntra`, `show_frame`,
  `showable_frame` (read vs. KEY-derived), `error_resilient_mode`
  (with the SWITCH / (KEY+show_frame) override), `disable_cdf_update`,
  `allow_screen_content_tools` (with the
  `SELECT_SCREEN_CONTENT_TOOLS` sentinel), `force_integer_mv` (with
  the §5.9.2 `FrameIsIntra ⇒ 1` override), `current_frame_id` (gated
  by §5.5.1's `frame_id_numbers_present_flag` with the §6.8.2
  `idLen <= 16` conformance check), `frame_size_override_flag` (with
  the SWITCH-derives-1 / reduced-still-derives-0 cases),
  `order_hint` (width from §5.5.1's `order_hint_bits`),
  `primary_ref_frame` (with `PRIMARY_REF_NONE = 7` for intra /
  error-resilient frames), and `refresh_frame_flags` (with the
  SWITCH / (KEY+show_frame) → `allFrames = 0xff` derivation). The
  reduced-still-picture-header collapse from §5.9.2 is honoured.
  Returns the typed `FrameHeader` descriptor plus the bit count
  consumed so the next round can resume at exactly the right bit.
  New `FrameType` enum (with `from_raw` / `as_raw` /  `is_intra`),
  `NUM_REF_FRAMES` / `PRIMARY_REF_NONE` constants, and
  `Error::InvalidIdLen` / `Error::TemporalPointInfoUnsupported`
  variants. The `temporal_point_info()` (§5.9.31) call sites are
  stubbed for now; the parser refuses to descend when
  `decoder_model_info_present_flag && !equal_picture_interval`
  (no fixture in the current corpus exercises that path).

  Validation: 7 frame-header unit tests (two real-OBU traces +
  reduced-still / show-existing synthetic vectors + truncated-input
  and FrameType-roundtrip), plus one integration test
  (`tests/frame_header_fixtures.rs`) that re-parses the
  sequence header and the first frame OBU's uncompressed-header
  prefix for all 16 corpus fixtures and asserts on 12 trace columns
  per fixture. All 192 (16 × 12) field assertions pass bit-exact.

* **Round 2 — Sequence header OBU parse (§5.5).** New
  `sequence_header` module implements `sequence_header_obu()` per
  §5.5.1 plus the nested `color_config()` (§5.5.2), `timing_info()`
  (§5.5.3), `decoder_model_info()` (§5.5.4) and
  `operating_parameters_info()` (§5.5.5) sub-syntax tables, returning
  a strongly typed `SequenceHeader` descriptor (`seq_profile`,
  `still_picture`, `reduced_still_picture_header`, timing /
  decoder-model state, operating-point list, frame-size bits,
  `frame_id_numbers_present_flag`, all `enable_*` capability bits,
  `seq_force_*` flags, `order_hint_bits`, full
  `ColorConfig` block, `film_grain_params_present`, and the bit
  count consumed so the §5.3.1 `trailing_bits` accounting can plug in
  next round). New internal `bitreader` module provides the §4.10.2
  `f(n)` and §4.10.3 `uvlc()` primitives over a borrowed byte slice
  per §8.1 (MSB-first). New `Error::ReservedProfile(p)` /
  `Error::ReducedStillRequiresStill` variants surface the two
  §6.4.1 bitstream-conformance failures the parser enforces.

  Validation: 7 bitreader unit tests, 7 sequence-header unit tests
  (incl. real OBU bytes captured from three fixture IVFs and a
  synthetic reduced-still vector), plus one integration test in
  `tests/sequence_header_fixtures.rs` that walks all 16 corpus
  fixtures under `docs/video/av1/fixtures/`, strips IVF framing,
  runs the round-1 OBU walker, and asserts every field of the first
  `SEQUENCE_HEADER` matches the `SEQ_HEADER` line in the fixture's
  `trace.txt`. All 16 fixtures pass byte-exact, covering profiles
  0/1/2, 8/10/12-bit, 4:2:0 / 4:2:2 / 4:4:4 / monochrome, 64×64
  through 256×128, 128×128 superblocks, screen-content tools,
  super-resolution still pictures, film-grain-on, and the
  reduced-still-picture-header still-picture paths.

* **Round 1 — OBU bytestream walker.** First clean-room contribution
  to the rebuild. New `obu` module exposes:
  * `parse_leb128` — `leb128()` per §4.10.5, including the
    `(1 << 32) - 1` conformance cap and the 8-byte length bound.
  * `parse_obu` — `obu_header` (§5.3.2) + optional
    `obu_extension_header` (§5.3.3) + optional `obu_size` (§5.3.1 /
    §6.2.1) decode into an `ObuDescriptor`.
  * `ObuIter` — iterator that walks a concatenation of OBUs in the
    §5.2 low-overhead format.
  * `ObuType` — symbolic enum for the obu_type values listed in
    §6.2.2, preserving the raw byte for reserved values.

  New `Error` variants for OBU-walker failures (`UnexpectedEnd`,
  `ForbiddenBitSet`, `MissingSizeField`, `Leb128Overflow`,
  `Leb128TooLong`, `SizeOverflow`). 12 unit tests covering happy-path
  decode, multi-byte leb128, redundant zero padding, leb128
  overflow/length rejection, extension header decode, iterator over
  concatenated OBUs, forbidden-bit rejection, truncated-payload
  rejection, missing-size-field rejection, and reserved-obu_type
  byte preservation.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained.
