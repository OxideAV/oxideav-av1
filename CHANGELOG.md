# Changelog

All notable changes to `oxideav-av1` are recorded here.

## [Unreleased]

### Added

* **Round 172 — §7.10 `find_mv_stack` spatial-only path.** Lifts the
  `Error::FindMvStackUnsupported` stub the r170 §5.11.23 reader
  surfaced after the §5.11.25 `read_ref_frames` prologue, on the
  spatial-only path (`use_ref_frame_mvs == false`).

  The new `PartitionWalker::find_mv_stack(...)` method composes the
  §7.10.2 driver, §7.10.2.1 setup-global-mv (with the §7.10.2.10
  lower-precision pass), §7.10.2.2 scan-row, §7.10.2.3 scan-col,
  §7.10.2.4 scan-point (both corner positions: `(-1, bw4)` and
  `(-1, -1)`), §7.10.2.7 add-ref-mv-candidate (single + compound
  branches), §7.10.2.8 search-stack, §7.10.2.9 compound-search-stack
  (with the `GLOBAL_GLOBALMV` global-MV substitution), §7.10.2.10
  lower-precision (LSB-strip when high-precision off,
  `(|a|+3)>>3<<3` integer-round when `force_integer_mv == 1`),
  §7.10.2.11 stable descending-by-weight sort with lockstep
  `RefStackMv` swap, §7.10.2.12 extra-search (two-pass partial-match
  scan + global-motion fill for compound), §7.10.2.13
  add-extra-mv-candidate (single + compound branches, with the
  `RefFrameSignBias` MV negation), and §7.10.2.14 context + clamping
  (`DrlCtxStack[]` derivation + per-list `clamp_mv_row` /
  `clamp_mv_col` + `NewMvContext` / `RefMvContext` derivation from
  the three-arm `CloseMatches` ladder).

  The §7.10.2.5 temporal scan + §7.10.2.6 temporal sample
  sub-processes (steps 17 of the §7.10.2 driver) are deferred to a
  subsequent arc — when the caller passes `use_ref_frame_mvs ==
  true`, the new method returns the new
  `Error::TemporalMvScanUnsupported` variant without partial state
  mutation. The temporal scan requires a `MotionFieldMvs[ref][y8][x8]`
  grid populated by §7.9 `motion_field_estimation` from the §5.9.20
  `RefFrameSignBias` chain — that scaffolding lands with the next
  arc. Spatial-only is the conformant path for any frame with
  `use_ref_frame_mvs == 0`.

  The walker gains a new `Mvs[r][c][list][comp]` grid stored as a
  flat `Vec<i16>` with four `i16` slots per `(row, col)` cell (two
  lists × two components). Pre-fill is zero; the §7.10.2.7
  `IsInters[mvRow][mvCol] == 0 ⇒ return` gate ensures the pre-fill is
  unobservable on the conformant path until §5.11.31 `assign_mv`
  (next-arc) writes it. New `mvs()` accessor exposes the grid as a
  flat slice; new internal `mv_at(r, c, list, comp)` helper applies
  the §7.10 unavailable-cell fallback (returns 0 for out-of-grid).

  New aggregate `FindMvStackResult` surfaces the §7.10.2 output:
  `num_mv_found`, `new_mv_count`, `ref_stack_mv[idx][list][comp]`,
  `weight_stack[idx]`, `global_mvs[list]`, `new_mv_context`,
  `ref_mv_context`, `zero_mv_context`, `drl_ctx_stack[idx]`, plus
  the §7.10.2 step 12-14 snapshot values `close_matches`,
  `total_matches`, `num_nearest`, `num_new` for downstream §5.11.23
  consumers.

  20 new §3 / §6.10.22 / §7.10 constants land:
  `MAX_REF_MV_STACK_SIZE = 8`, `REF_CAT_LEVEL = 640`,
  `MV_BORDER = 128`, `WARPEDMODEL_PREC_BITS = 16` (cdf-local twin);
  warp-model discriminants `GM_TYPE_{IDENTITY, TRANSLATION, ROTZOOM,
  AFFINE} = 0..=3`; the §6.10.22 inter Y mode ordinal table
  `MODE_{NEARESTMV = 14, NEARMV = 15, GLOBALMV = 16, NEWMV = 17,
  NEAREST_NEARESTMV = 18, NEAR_NEARMV = 19, NEAREST_NEWMV = 20,
  NEW_NEARESTMV = 21, NEAR_NEWMV = 22, NEW_NEARMV = 23,
  GLOBAL_GLOBALMV = 24, NEW_NEWMV = 25}` plus the `has_newmv(mode)`
  predicate.

  The §5.11.23 `decode_inter_block_mode_info` reader continues to
  return `Error::FindMvStackUnsupported` because the post-MV-stack
  reader cascade (`compound_mode` / `new_mv` / `zero_mv` / `ref_mv` /
  `drl_mode` / `assign_mv` / `read_motion_mode` /
  `read_interintra_mode` / `read_compound_type` /
  `read_interpolation_filter`) remains pending. The wiring of
  `find_mv_stack` into the dispatcher follows once those readers
  land.

  24 new unit tests cover: the §7.10.2.10 lower-precision matrix
  (high-precision passthrough, LSB strip, `force_integer_mv` 3-bit
  round); §5.11.53 / §5.11.54 clamp_mv_row / clamp_mv_col bracket
  derivations; §7.10.2.11 stable sort (descending by weight, equal-
  weight stability); §7.10.2.1 setup-global-mv (INTRA_FRAME =
  0, IDENTITY, TRANSLATION shift); the §7.10.2 driver outcomes —
  empty walker single-pred (NumMvFound = 0), empty walker compound
  (NumMvFound = 2 via global-MV fill), caller-bug guards (sub_size,
  mi_row, ref_frame[0..2] range, isCompound mismatch), temporal-scan
  deferral, one above-neighbour with NEWMV (REF_CAT_LEVEL bonus +
  NewMvContext = 2, RefMvContext = 3), one left-neighbour with
  NEARESTMV (numNew = 0, NewMvContext = 3), both-neighbours case
  (CloseMatches = 2 ⇒ NewMvContext = 4, RefMvContext = 5),
  mismatched-ref handling (close-match excluded but extra-search
  picks up), identical-MV deduplication (weight accumulation),
  single-pred extra-search NumMvFound non-increment, `has_newmv`
  truth table, MV-stack constant values, and DrlCtxStack[]
  derivation. Test count: 627 → 651 (+24). `decode_av1` /
  `encode_av1` continue to return `Error::NotImplemented`.

* **Round 171 — §5.11.46 palette-entries reader
  (`palette_colors_y[]` / `palette_colors_u[]` / `palette_colors_v[]`)
  + §5.11.49 `get_palette_cache(plane)`.** Lifts the
  `Error::PaletteEntriesUnsupported` stub the r169 §5.11.22 reader
  surfaced after `palette_size_y_minus_2` /
  `palette_size_uv_minus_2`. The §5.11.22
  `decode_intra_block_mode_info` reader now threads a new
  `bit_depth: u8` argument (in {8, 10, 12} per §5.5.2; caller-bug
  inputs surface `PartitionWalkOutOfRange` before any bit is
  consumed) and runs the §5.11.46 entry reads to completion on the
  conformant path:

  * The Y luma arm — §5.11.49 `cacheN = get_palette_cache(0)` merge
    of above + left palettes, the cache-coded indices loop reading
    `use_palette_color_cache_y L(1)` per cache slot, the first new
    entry as `L(BitDepth)`, the optional `L(2)
    palette_num_extra_bits_y` and the delta loop with `palette_delta_y
    L(paletteBits)` + the spec's `++` increment + `Clip1` + `range =
    (1 << BitDepth) - palette_colors_y[idx] - 1` paletteBits
    refinement, terminating in the trailing ascending sort.
  * The U chroma arm — mirrors the Y reader minus the `++` step and
    with `range = (1 << BitDepth) - palette_colors_u[idx]` (no
    `- 1`), terminating in the same trailing sort.
  * The V chroma arm — two-arm dispatch on
    `delta_encode_palette_colors_v L(1)`: on 1, signed-delta path
    with `minBits = BitDepth - 4`, `L(2)
    palette_num_extra_bits_v`, `L(BitDepth) palette_colors_v[0]`,
    then per-entry `L(paletteBits) palette_delta_v` + optional
    `L(1) palette_delta_sign_bit_v` sign-flip + modular wrap into
    `[0, maxVal)` + `Clip1`; on 0, direct `PaletteSizeUV`-many
    `L(BitDepth)` literals (no sort either way).

  The walker grows two new grids — `PaletteSizes[plane][r][c]` (a
  `3 * mi_rows * mi_cols` flat `Vec<u8>`) and
  `PaletteColors[plane][r][c][idx]` (a `3 * mi_rows * mi_cols *
  PALETTE_COLORS = 24 * area` flat `Vec<u16>`) — pre-filled at
  walker-construction time to the spec's "no palette here"
  identity. Decoded palette entries are stamped over the block's
  `bh4 * bw4` footprint per plane so the next §5.11.46 call's
  `get_palette_cache(plane)` two-pointer neighbour merge observes
  the propagated values. The §5.11.49 `(MiRow * MI_SIZE) % 64`
  superblock-top gate suppresses the above-neighbour read at 64×64
  superblock boundaries; `AvailL` collapses to `mi_col > 0` for the
  tile-start tests (out-of-grid reads return 0, the spec-correct
  identity).

  Two new public-by-grids accessors land on `PartitionWalker`:
  `palette_sizes()` and `palette_colors()` (each returns a flat
  slice with plane-outermost row-major layout). The
  `get_palette_cache(plane, mi_row, mi_col, &mut [u16; 16])`
  method runs the §5.11.49 merge + dedupe and writes the cache
  into the caller-supplied buffer (max `2 * PALETTE_COLORS = 16`
  entries), returning the count. Two new free helpers cover the
  spec's §4.7 mathematical primitives the §5.11.46 reader needs:
  `clip1_to_bit_depth(x, bd)` (the §4.7 `Clip1` clamp to `[0, (1 <<
  BitDepth) - 1]`) and `ceil_log2_av1(x)` (the §4.7 `CeilLog2`
  defined to return 0 on `x < 2`).

  Three new fields land on `DecodedIntraBlockModeInfo` carrying
  the decoded palette entries per plane: `palette_colors_y:
  Option<[u16; PALETTE_COLORS]>` (sorted ascending),
  `palette_colors_u: Option<[u16; PALETTE_COLORS]>` (sorted
  ascending), and `palette_colors_v: Option<[u16; PALETTE_COLORS]>`
  (source order — the V plane's §5.11.46 body skips the trailing
  sort). The `palette_size_y` / `palette_size_uv` fields now carry
  `Some(decoded_size)` on the conformant path (previously stayed
  `None` because the reader short-circuited before committing the
  size). `Error::PaletteEntriesUnsupported` is retained for ABI
  stability but is no longer constructed by the conformant code
  path (documented as a defensive fallback).

  12 new unit tests cover: the `bit_depth ∈ {8, 10, 12}` caller-bug
  guard; `get_palette_cache` returning empty on a fresh walker; the
  `clip1_to_bit_depth` truth table at 8 / 10 / 12 bits; the
  `ceil_log2_av1` truth table around the `x < 2` boundary;
  `decode_intra_block_mode_info` with rigged CDFs on the
  `has_palette_y == 0` path returning `Ok` with empty palette state;
  `read_palette_entries_y` direct-call with `PaletteSizeY = 2` and a
  zero bitstream yielding `[0, 1]` (the Y `delta + 1` step);
  `read_palette_entries_uv` direct-call with `PaletteSizeUV = 2` and
  a zero bitstream yielding `[0, 0]` on U (no `++` step) and the V
  direct-literal arm; hand-stamped palette grids visible through
  `get_palette_cache` (left-neighbour-only); the §5.11.49 above +
  left merge with a duplicate entry; the §5.11.49 superblock-top
  boundary gate; and the `read_palette_entries_y` cache-coded
  no-bit-read path via the `get_palette_cache` accessor. Total
  lib-test count: 580 → 592 (+12). All four §5.11.22 test setups
  thread the new `bit_depth = 8` argument; no behavioural change on
  any pre-r171 test.

  Followup arcs: §7.10 `find_mv_stack` MV-stack derivation (the
  `Error::FindMvStackUnsupported` gap r170 surfaced); §5.11.30
  `compute_prediction`; §5.11.34 `residual` walker.

* **Round 170 — §5.11.23 `inter_block_mode_info()` prologue +
  §5.11.25 `read_ref_frames()`
  (`PartitionWalker::decode_inter_block_mode_info`).** Lands the
  §5.11.23 inter-arm prologue + the full §5.11.25 reference-frame
  syntax tree, lifting the §5.11.18 `if (is_inter)` arm out of the
  r168 `InterBlockModeInfoUnsupported` stub. The composite runs to
  completion through:

  * Lines 1-2: `PaletteSizeY = 0, PaletteSizeUV = 0` (zero-init).
  * Line 3 / §5.11.25 `read_ref_frames()` — four-arm dispatch:
    * `skip_mode == 1` ⇒ `RefFrame[0..2] = SkipModeFrame[0..2]`
      (no `S()` reads; values from §5.9.22 frame-header derivation
      passed via the new caller argument).
    * `seg_feature_active(SEG_LVL_REF_FRAME)` ⇒
      `RefFrame[0] = FeatureData[seg][SEG_LVL_REF_FRAME]`,
      `RefFrame[1] = NONE` (no `S()` reads).
    * `seg_feature_active(SEG_LVL_SKIP | SEG_LVL_GLOBALMV)` ⇒
      `RefFrame[0] = LAST_FRAME, RefFrame[1] = NONE` (no `S()`
      reads).
    * Default arm: `comp_mode` `S()` (gated on `reference_select &&
      Min(bw4, bh4) >= 2`); on COMPOUND_REFERENCE the
      `comp_ref_type` `S()` splits into the UNIDIR_COMP ladder
      (`uni_comp_ref` / `uni_comp_ref_p1` / `uni_comp_ref_p2`
      against `TileUniCompRefCdf[ctx][p]`) or the BIDIR_COMP ladder
      (`comp_ref` / `comp_ref_p1` / `comp_ref_p2` /
      `comp_bwdref` / `comp_bwdref_p1` against `TileCompRefCdf` /
      `TileCompBwdRefCdf`); on SINGLE_REFERENCE the
      `single_ref_p1..p6` cascade against `TileSingleRefCdf[ctx][p]`
      runs.
  * Line 4: `isCompound = RefFrame[1] > INTRA_FRAME` derivation.
  * Walker grid stamp: `RefFrames[r + y][c + x][0..2]` over the
    block's `bh4 * bw4` footprint, so subsequent §5.11.18
    prologues + §8.3.2 ref-frame ctx walks observe the value.

  The reader then short-circuits at the §7.10.2 `find_mv_stack`
  entry with the new `Error::FindMvStackUnsupported` — the
  motion-vector-stack derivation + every dependent §5.11.23
  reader (`compound_mode`, `new_mv` / `zero_mv` / `ref_mv`,
  `drl_mode`, `assign_mv`, `read_motion_mode`,
  `read_interintra_mode`, `read_compound_type`,
  `read_interpolation_filter`) are subsequent-arc targets. Every
  §5.11.25 `S()` read + the grid stamp commits to state before
  the stub fires.

  New §8.3.2 ref-frame ctx helpers land as free functions:
  * `check_backward(ref)` — `BWDREF_FRAME..=ALTREF_FRAME` predicate.
  * `is_samedir_ref_pair(ref0, ref1)` — same-direction-group test.
  * `count_refs(frame_type, …)` — neighbour-slot tallying.
  * `comp_mode_ctx(…)` — `TileCompModeCdf[ctx]` derivation
    (av1-spec p.366, 9-arm dispatch).
  * `comp_ref_type_ctx(…)` — `TileCompRefTypeCdf[ctx]` derivation
    (av1-spec p.382, three-nested-if dispatch).

  Walker gains a `RefFrames[][][..]` grid (`Vec<i8>`, two slots
  per `(row, col)` cell, pre-filled with `[INTRA_FRAME, NONE]`)
  with the `ref_frames()` view accessor + internal `ref_frame_at`
  neighbour-lookup helper. The §5.11.18 prologue's `LeftRefFrame[..]`
  / `AboveRefFrame[..]` derivations now consult the grid (gated by
  `AvailU` / `AvailL`); the §8.3.2 ref-frame ctx walks observe
  propagated values from prior decoded inter blocks rather than
  the previously-hardcoded fallback.

  Eight new public ref-frame ordinal constants land:
  `LAST2_FRAME = 2`, `LAST3_FRAME = 3`, `GOLDEN_FRAME = 4`,
  `BWDREF_FRAME = 5`, `ALTREF2_FRAME = 6` (alongside the existing
  `INTRA_FRAME = 0`, `LAST_FRAME = 1`, `ALTREF_FRAME = 7`), plus
  `SINGLE_REFERENCE = 0`, `COMPOUND_REFERENCE = 1`,
  `UNIDIR_COMP_REFERENCE = 0`, `BIDIR_COMP_REFERENCE = 1`.

  The §5.11.18 dispatcher signature gains four new arguments
  (`skip_mode_frame: [i32; 2]`, `seg_skip_active: bool`,
  `seg_ref_frame_data: i32`, `reference_select: bool`) threaded
  through to the §5.11.23 reader. New `DecodedInterBlockModeInfo`
  aggregate carries the §5.11.25 output + `is_compound` derivation
  (currently observable only via the walker grid since the reader
  always returns `Err`-path until §7.10 lands).

  Returns the new `Error::FindMvStackUnsupported` on the §7.10
  entry. Caller-bug arguments surface
  `Error::PartitionWalkOutOfRange`.

* **Round 169 — §5.11.22 `intra_block_mode_info()`
  (`PartitionWalker::decode_intra_block_mode_info`).** Lands the
  per-block intra-mode syntax composite reached from the §5.11.18
  `else` arm of `if (is_inter)` (the §5.11.7 keyframe path uses
  `intra_frame_y_mode` instead). Reader composes every §5.11.22 leaf
  in spec order:

  * Line 1: `RefFrame[0] = INTRA_FRAME, RefFrame[1] = NONE`
    (constant `[0, -1]`).
  * Line 2: `y_mode` S() against `TileYModeCdf[ctx = Size_Group[MiSize]]`
    via the existing `size_group` helper + `TileCdfContext::y_mode_cdf`
    accessor. Decoded value in `0..INTRA_MODES = 13`.
  * Line 3 grid-fill: `YModes[r + y][c + x] = YMode` over the
    `bh4 * bw4` footprint, mirroring `decode_intra_frame_y_mode`.
  * §5.11.42 `intra_angle_info_y()` — gated on `MiSize >= BLOCK_8X8 &&
    is_directional(YMode)`. Reads `S()` against
    `TileAngleDeltaCdf[YMode - V_PRED]`; result is biased into
    `-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA = -3..=3`.
  * `if (HasChroma)` arm — `uv_mode` S() with §8.3.2 CFL-allowance
    branch (lossless + post-subsampling 4×4 OR !lossless +
    Max(W,H) ≤ 32 → `TileUVModeCflAllowedCdf[YMode]`, else
    `TileUVModeCflNotAllowedCdf[YMode]`). Decoded value in
    `0..UV_INTRA_MODES_CFL_{ALLOWED,NOT_ALLOWED}`.
  * §5.11.45 `read_cfl_alphas()` — gated on `UVMode == UV_CFL_PRED`.
    Reads `cfl_alpha_signs` S() (8-symbol context-free), derives
    `signU = (signs+1)/3, signV = (signs+1)%3` (CFL_SIGN_ZERO/NEG/POS
    = 0/1/2). Per axis, if sign ≠ ZERO, reads `cfl_alpha_{u,v}` S()
    against `TileCflAlphaCdf[ctx]` (ctx from existing
    `cfl_alpha_{u,v}_ctx` helpers); computes
    `CflAlpha{U,V} = ±(1 + raw)` (negated on SIGN_NEG).
  * §5.11.43 `intra_angle_info_uv()` — mirror of §5.11.42 against
    `TileAngleDeltaCdf[UVMode - V_PRED]`.
  * §5.11.46 `palette_mode_info()` — outer gate
    `MiSize >= BLOCK_8X8 && Block_W <= 64 && Block_H <= 64 &&
    allow_screen_content_tools`. Per-plane:
    * Luma arm (`YMode == DC_PRED`): `has_palette_y` S() against
      `TilePaletteYModeCdf[bsizeCtx][ctx]` (`bsizeCtx =
      Mi_W_Log2 + Mi_H_Log2 - 2`, `ctx = above_pal + left_pal`
      via existing `palette_y_mode_ctx`); on 1, reads
      `palette_size_y_minus_2` S() then surfaces
      `Err(PaletteEntriesUnsupported)` (the §5.11.46 entries L(*)
      reads need parser-scope `BitDepth` + `PaletteCache[]`
      plumbing — subsequent arc).
    * Chroma arm (`HasChroma && UVMode == DC_PRED`): `has_palette_uv`
      S() against `TilePaletteUVModeCdf[ctx = (PaletteSizeY > 0)]`;
      on 1, reads `palette_size_uv_minus_2` S() then surfaces
      `PaletteEntriesUnsupported`.
  * §5.11.24 `filter_intra_mode_info()` — outer gate
    `enable_filter_intra && YMode == DC_PRED && PaletteSizeY == 0 &&
    Max(Block_W, Block_H) <= 32`. Reads `use_filter_intra` S()
    against `TileFilterIntraCdf[MiSize]`; on 1, reads
    `filter_intra_mode` S() against `TileFilterIntraModeCdf`
    (context-free, 5 modes).

  Returns the full `DecodedIntraBlockModeInfo` aggregate carrying
  every decoded value (with `Option<…>` fields for the spec's
  "not read" arms). The §5.11.18 dispatcher stub
  (`IntraBlockModeInfoUnsupported`) remains in place — the new
  reader needs the additional sequence-header arguments
  (`has_chroma`, `allow_screen_content_tools`,
  `enable_filter_intra`, `subsampling_x`, `subsampling_y`,
  `above_palette_y`, `left_palette_y`) that the §5.11.18
  signature doesn't yet thread through; direct callers can invoke
  `decode_intra_block_mode_info` with the missing arguments.

  Two new public free functions land alongside:

  * `is_directional(mode)` — §5.11.44 predicate
    (`V_PRED <= mode <= D67_PRED`).
  * `cfl_allowed_for_uv_mode(lossless, mi_size, sub_x, sub_y)` —
    §8.3.2 `uv_mode` CFL-allowance derivation.

  Three new public constants: `D67_PRED = 8`, `UV_CFL_PRED = 13`,
  and one new error variant: `PaletteEntriesUnsupported`.

  Tests added (15 new direct-call tests):

  * `decode_intra_block_mode_info_rejects_out_of_range` — bounds
    guards (sub_size / mi_row / mi_col) surface
    `PartitionWalkOutOfRange` before any bit read.
  * `decode_intra_block_mode_info_dc_pred_happy_path_all_gates_off`
    — DC_PRED + HasChroma + all gates off, every field at its
    short-circuit value.
  * `decode_intra_block_mode_info_monochrome_skips_chroma_reads` —
    HasChroma = false skips uv_mode / cfl / angle_delta_uv.
  * `decode_intra_block_mode_info_dc_pred_skips_angle_delta_y` —
    §5.11.42 non-directional short-circuit.
  * `decode_intra_block_mode_info_small_block_skips_angle_delta_y`
    — §5.11.42 small-block (BLOCK_4X4) short-circuit.
  * `decode_intra_block_mode_info_palette_gate_off_skips_palette_reads`
    — `allow_screen_content_tools = false` skips §5.11.46.
  * `decode_intra_block_mode_info_non_dc_y_mode_skips_palette_luma_arm`
    — non-DC YMode skips luma palette read.
  * `decode_intra_block_mode_info_filter_intra_disabled_short_circuit`
    — `enable_filter_intra = false` skips §5.11.24.
  * `decode_intra_block_mode_info_non_dc_y_mode_skips_filter_intra`
    — non-DC YMode skips §5.11.24 (gate's `YMode == DC_PRED` term).
  * `decode_intra_block_mode_info_large_block_skips_filter_intra` —
    Max(W, H) > 32 skips §5.11.24.
  * `decode_intra_block_mode_info_ref_frame_always_intra_none` —
    `[INTRA_FRAME, NONE] = [0, -1]` invariant.
  * `decode_intra_block_mode_info_stamps_y_modes_grid` /
    `decode_intra_block_mode_info_stamps_non_dc_y_mode` — §5.11.5
    grid-fill verification.
  * `is_directional_matches_spec_range` — §5.11.44 truth-table.
  * `cfl_allowed_truth_table_matches_spec` — §8.3.2 CFL-allowance
    truth-table (lossless + post-subsampling 4×4, !lossless +
    Max ≤ 32, out-of-range caller-bug guard).

  Test count: 587 → 602 (+15 new).

  Provenance: AV1 Specification (av1-spec.txt) §5.11.22 / §5.11.24
  / §5.11.42 / §5.11.43 / §5.11.44 / §5.11.45 / §5.11.46 + §8.3.2
  ("y_mode", "uv_mode", "angle_delta_y", "angle_delta_uv",
  "has_palette_y", "has_palette_uv", "palette_size_y_minus_2",
  "palette_size_uv_minus_2", "use_filter_intra",
  "filter_intra_mode", "cfl_alpha_signs", "cfl_alpha_u",
  "cfl_alpha_v") + the §9.4 default CDFs already in the crate.

* **Round 168 — §5.11.17 `read_var_tx_size()` + §5.11.18
  `inter_frame_mode_info()` (`PartitionWalker::read_var_tx_size` +
  `decode_inter_frame_mode_info`).** Lands the two missing inter-arm
  syntax composites that bound the §5.11.5 walker's inter side.

  §5.11.17 body transcription (av1-spec p.70):

  * Frame-edge clip: `row >= MiRows || col >= MiCols` short-circuits
    with no read, no stamp.
  * `txSz == TX_4X4 || depth == MAX_VARTX_DEPTH` forces
    `txfm_split = 0` (no S() consumed); otherwise reads `txfm_split`
    against the §8.3.2-selected `TileTxfmSplitCdf[ctx]` row. The ctx
    formula inlines `find_tx_size(size, size)` for the
    `maxTxSz` derivation and routes through the existing
    `txfm_split_ctx` helper.
  * §8.3.2 `get_above_tx_width` / `get_left_tx_height` inlined as
    [`PartitionWalker`] private helpers against the walker's
    `Skips[]` / `IsInters[]` / `MiSizes[]` / `InterTxSizes[]`
    grids; the `row == MiRow` / `col == MiCol` arm gates the
    `Skips && IsInters` early return, and the fall-through reads
    `Tx_Width[ InterTxSizes[ above ] ]` /
    `Tx_Height[ InterTxSizes[ left ] ]`.
  * `txfm_split == 1`: recurse over `(h4 / stepH, w4 / stepW)`
    sub-blocks with `subTxSz = Split_Tx_Size[ txSz ]` and
    `depth + 1`. The recursive descent is bounded by
    `MAX_VARTX_DEPTH = 2`.
  * `txfm_split == 0` (terminal else): stamp
    `InterTxSizes[ row + i ][ col + j ] = txSz` over the `(h4, w4)`
    footprint, set `TxSize = txSz`.

  The §5.11.16 inter-arm now enters `read_var_tx_size` instead of
  surfacing `Error::ReadVarTxSizeUnsupported`. `TxSizes[]` is
  stamped with the last terminal-else's `txSz` over the full block
  footprint per §5.11.5.

  §5.11.18 body transcription (av1-spec p.71): composes every leaf
  in spec order:

  * `use_intrabc = 0`.
  * `LeftRefFrame[..]` / `AboveRefFrame[..]` / `LeftIntra` /
    `AboveIntra` / `LeftSingle` / `AboveSingle` local derivations
    (currently fixed at `[INTRA_FRAME, NONE]` since the walker
    doesn't yet track `RefFrames[][][..]` — the §5.11.23 readers'
    next-round target).
  * `skip = 0`.
  * §5.11.19 `inter_segment_id(1)` via
    [`PartitionWalker::decode_inter_segment_id`].
  * §5.11.10 `read_skip_mode()` via
    [`PartitionWalker::decode_skip_mode`].
  * `if (skip_mode) skip = 1 else read_skip()` — the
    `skip_mode == 1` arm bypasses [`PartitionWalker::decode_skip`]
    and stamps `Skips[][] = 1` directly per the §5.11.5 grid-fill
    invariant.
  * §5.11.19 `inter_segment_id(0)` (post-skip arm, fired only when
    `!SegIdPreSkip`).
  * `Lossless = LosslessArray[segment_id]`.
  * §5.11.56 `read_cdef()` via [`PartitionWalker::decode_cdef`].
  * §5.11.12 `read_delta_qindex()` via
    [`PartitionWalker::decode_delta_qindex`].
  * §5.11.13 `read_delta_lf()` via
    [`PartitionWalker::decode_delta_lf`].
  * `ReadDeltas = 0` (caller-owned per the §6.10.4 derivation).
  * §5.11.20 `read_is_inter()` via
    [`PartitionWalker::decode_is_inter`].
  * Terminal `if (is_inter) inter_block_mode_info() else
    intra_block_mode_info()` short-circuits at two new `Error`
    variants: `Error::InterBlockModeInfoUnsupported` (the §5.11.23
    next-round target — MV stack / ref-frame readers) and
    `Error::IntraBlockModeInfoUnsupported` (the §5.11.22 next-round
    target — per-block intra angle / UV mode readers). All
    pre-dispatch reads commit to the bitstream / grids before the
    stub fires.

  New free function [`cdf::find_tx_size`] (re-exported at the crate
  root) — the §5.11.36 spec helper, a linear scan over
  `TX_SIZES_ALL` returning the first ordinal whose
  `(Tx_Width, Tx_Height)` matches `(w, h)`. Used by the §8.3.2
  `txfm_split` ctx selector's `maxTxSz = find_tx_size(size, size)`
  derivation.

  New `DecodedInterFrameModeInfo` per-block aggregate (publicly
  constructible) carries every §5.11.18 derived value:
  `mi_row` / `mi_col` / `mi_size` / `use_intrabc` / `avail_u` /
  `avail_l` / `left_ref_frame` / `above_ref_frame` / `left_intra`
  / `above_intra` / `left_single` / `above_single` / `skip` /
  `skip_mode` / `segment_id` / `lossless` / `cdef_idx` /
  `current_q_index` / `current_delta_lf` / `is_inter`. Re-exported
  at the crate root.

  Two new `Error` variants: `Error::InterBlockModeInfoUnsupported`
  (§5.11.23) and `Error::IntraBlockModeInfoUnsupported` (§5.11.22).

  The §5.11.5 `decode_block_syntax` walker is unchanged on the
  `frame_is_intra = false` arm — it still short-circuits with
  `Error::DecodeBlockInterFrameUnsupported` (the umbrella stub)
  because the §5.11.18 reader needs additional segmentation-feature
  / skip-mode-present caller state the §5.11.5 driver doesn't yet
  thread through. Direct callers of `decode_inter_frame_mode_info`
  get the full pre-dispatch walk and the §5.11.22 / §5.11.23
  distinction.

  11 new integration tests
  (`tests/decode_block_syntax_walker.rs`):

  * `read_var_tx_size_tx_4x4_no_read` — base case, no S() consumed,
    stamp at anchor.
  * `read_var_tx_size_max_depth_no_read` — depth cap, no S(), full
    footprint stamp.
  * `read_var_tx_size_split_to_max_depth_stamps_tx_4x4` — split path
    with all-1 `txfm_split` returns `TX_4X4` at depth 2 and stamps
    every 1×1 cell.
  * `read_var_tx_size_out_of_frame_short_circuits` — both
    `row >= MiRows` and `col >= MiCols`.
  * `read_var_tx_size_rejects_out_of_range` — four caller-bug
    guards.
  * `read_block_tx_size_inter_arm_no_split_returns_max_tx_size`
    — the §5.11.16 inter-arm now enters `read_var_tx_size` and
    returns `maxTxSz` on the no-split path with the BLOCK_16X16
    (4×4) footprint of `InterTxSizes[]` stamped (replaces the r167
    stub test).
  * `decode_inter_frame_mode_info_reaches_intra_block_stub` — the
    baseline path reaches the §5.11.22 intra stub with `Skips[][] = 0`
    and `IsInters[][] = 0` stamped.
  * `decode_inter_frame_mode_info_reaches_inter_block_stub` — the
    `seg_ref_frame_active + is_inter` arm reaches §5.11.23 with
    `IsInters[][] = 1` stamped.
  * `decode_inter_frame_mode_info_skip_mode_forces_skip_and_inter`
    — `skip_mode = 1` forces `Skips[][] = 1` + `SkipModes[][] = 1`
    + `IsInters[][] = 1` and reaches §5.11.23.
  * `decode_inter_frame_mode_info_seg_globalmv_forces_inter` —
    the §5.11.20 third arm (`seg_globalmv_active`) reaches §5.11.23.
  * `decode_inter_frame_mode_info_rejects_out_of_range` — four
    caller-bug guards.
  * `decoded_inter_frame_mode_info_struct_public_api_smoke` —
    `DecodedInterFrameModeInfo` is publicly constructible with
    every field default-valid; the struct is
    `Debug + Clone + Copy + PartialEq + Eq`.

  Plus 1 new unit test for `find_tx_size` (square and rectangular
  size matches, out-of-range returns `None`).

  The §5.11.5 calls that remain stubbed (each becomes the next-round
  target):

  * §5.11.22 `intra_block_mode_info()` — the §5.11.18 `else` arm of
    `if (is_inter)`. Per-block intra angle / UV mode readers.
  * §5.11.23 `inter_block_mode_info()` — the §5.11.18 `if (is_inter)`
    arm. MV stack / ref-frame readers + `assign_mv` /
    `read_interintra_mode` / `read_motion_mode` / `read_compound_type`
    / interpolation-filter reads.
  * §5.11.30 `compute_prediction()` — the immediate next-round
    target on the intra arm; the walker now reaches this stub after
    the §5.11.16 pass completes (unchanged from r167).
  * §5.11.34 `residual()` — reachable once §5.11.30 lands.

  `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

* **Round 167 — §5.11.16 `read_block_tx_size()` + §5.11.15
  `read_tx_size()` reader (`PartitionWalker::read_block_tx_size`).**
  Lands the per-block transform-size syntax-tree read that the r166
  §5.11.5 walker hits as the first stub on the intra arm. New
  [`PartitionWalker::read_block_tx_size`] method exposes the §5.11.16
  reader standalone; the §5.11.5 walker now invokes it after the
  §5.11.49 `palette_tokens()` no-op and short-circuits at the §5.11.30
  `compute_prediction()` stub instead.

  §5.11.16 body transcription (av1-spec p.70):

  * `bw4 = Num_4x4_Blocks_Wide[ MiSize ]`, `bh4 =
    Num_4x4_Blocks_High[ MiSize ]`.
  * The outer `TX_MODE_SELECT && MiSize > BLOCK_4X4 && is_inter &&
    !skip && !Lossless` gate routes to §5.11.17 `read_var_tx_size`
    (deferred to the next-arc round — currently surfaces a new
    `Error::ReadVarTxSizeUnsupported`). Unreachable from
    `decode_block_syntax` (the inter arm is stubbed upstream at
    `Error::DecodeBlockInterFrameUnsupported`); reachable from
    direct `read_block_tx_size` calls.
  * The `else` arm inlines §5.11.15 `read_tx_size(!skip ||
    !is_inter)` and applies the §5.11.16 `else`-arm grid-fill
    (`InterTxSizes[ row ][ col ] = TxSize` over the block footprint),
    plus the §5.11.5 footer's `TxSizes[ r + y ][ c + x ] = TxSize`.

  §5.11.15 body (av1-spec p.69):

  * `Lossless` short-circuits to `TxSize = TX_4X4` with no S()
    consumed.
  * Otherwise `TxSize = Max_Tx_Size_Rect[ MiSize ]`.
  * When `MiSize > BLOCK_4X4 && allowSelect && TxMode ==
    TX_MODE_SELECT`, read `tx_depth` S() and apply
    `TxSize = Split_Tx_Size[ TxSize ]` for `tx_depth` iterations.

  §8.3.2 `tx_depth` ctx (av1-spec p.363):

  * `aboveW` / `leftH` walk the §8.3.2 neighbour ladder: when
    `AvailU && IsInters[ above ]`, `aboveW = Block_Width[
    MiSizes[ above ] ]`; when `AvailU && !IsInters[ above ]` (the
    fall-through arm of `get_above_tx_width` after the
    `Skips && IsInters` gate fails), `aboveW = Tx_Width[
    InterTxSizes[ above ] ]`; when `!AvailU`, `aboveW = 0`. Mirrored
    for `leftH`.
  * `ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)`,
    consumed by the existing [`tx_depth_ctx`] helper.
  * CDF row selected by `Max_Tx_Depth[ MiSize ]`:
    [`TileCdfContext::tx_depth_cdf`] returns `tx_8x8` /
    `tx_16x16` / `tx_32x32` / `tx_64x64` for `maxTxDepth = 1, 2,
    3, 4` respectively (the existing accessor, used here for the
    first time on the read path).

  New spec tables (`src/cdf.rs`):

  * [`MAX_TX_SIZE_RECT`] — `Max_Tx_Size_Rect[ BLOCK_SIZES ]` from
    av1-spec p.402. Square `BLOCK_NxN → TX_NxN` for the four
    primary square sizes; `TX_64X64` for the 128×* triple
    (`BLOCK_64X128` / `BLOCK_128X64` / `BLOCK_128X128`); the
    matching rectangular entry for every rectangular `BLOCK_*`.
  * [`MAX_TX_DEPTH_TABLE`] — `Max_Tx_Depth[ BLOCK_SIZES ]` from
    av1-spec p.69 (`{ 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4,
    4, 2, 2, 3, 3, 4, 4 }`). Named with the `_TABLE` suffix to
    avoid shadowing the existing [`MAX_TX_DEPTH`] (`= 2`) symbol-cap
    constant.
  * [`SPLIT_TX_SIZE`] — `Split_Tx_Size[ TX_SIZES_ALL ]` from
    av1-spec p.404. Indexed by `TxSize` ordinal; `TX_4X4` is a
    fixed point.
  * [`MAX_VARTX_DEPTH`] — `= 2` per §3 av1-spec p.7.

  New rectangular `TX_*` ordinals (`TX_4X8` through `TX_64X16`,
  values `5..=18` per §6.10.16). Used as entries of
  [`MAX_TX_SIZE_RECT`] / [`SPLIT_TX_SIZE`] and exposed for the
  §5.11.17 reader.

  New [`PartitionWalker`] grids:

  * `tx_sizes: Vec<u8>` — `TxSizes[ row ][ col ]` from §5.11.5.
    `mi_rows * mi_cols` row-major; cells stamped to `TxSize` over
    the block footprint by [`PartitionWalker::read_block_tx_size`].
    Initialised to `TX_4X4` (the §8.3.2 ctx-walk identity).
    Accessor: [`PartitionWalker::tx_sizes`].
  * `inter_tx_sizes: Vec<u8>` — `InterTxSizes[ row ][ col ]` from
    §5.11.16 / §5.11.17. Same shape as `tx_sizes`; on the §5.11.16
    `else` arm both grids carry the same per-cell `TxSize` value.
    Accessor: [`PartitionWalker::inter_tx_sizes`].

  [`DecodedBlock`] gains a `tx_size: u8` field carrying the
  §5.11.16 return on the no-stub path.

  [`PartitionWalker::decode_block_syntax`] gains a `tx_mode_select:
  bool` parameter (threaded from §5.9.21 / §6.8.21 `TxMode ==
  TX_MODE_SELECT`). The walker now invokes `read_block_tx_size`
  after `palette_tokens()`, populates the [`DecodedBlock`]
  aggregate, and short-circuits with the
  `Error::DecodeBlockComputePredictionUnsupported` variant (the
  §5.11.30 target).
  [`PartitionWalker::decode_partition_syntax`] threads the new
  parameter through every recursive call and every leaf-emitting
  `db!` macro expansion.

  New `Error::ReadVarTxSizeUnsupported` variant surfaces the
  §5.11.17 `read_var_tx_size` deferral on the inter `TX_MODE_SELECT
  && !skip && !Lossless` arm.

  10 new integration tests
  (`tests/decode_block_syntax_walker.rs`) cover the §5.11.16
  reader paths:

  * `read_block_tx_size_lossless_forces_tx_4x4` — the §5.11.15
    `Lossless` short-circuit (no S() consumed).
  * `read_block_tx_size_tx_mode_largest_skips_tx_depth_read` — the
    `TX_MODE_LARGEST` path (`maxRectTxSize` returned with no
    `tx_depth` read).
  * `read_block_tx_size_block_4x4_skips_tx_depth_read` — the
    `MiSize > BLOCK_4X4` guard's fall-through for BLOCK_4X4.
  * `read_block_tx_size_tx_mode_select_depth_{zero,one,two}_*` — the
    `Split_Tx_Size` chain `TX_16X16 → TX_8X8 → TX_4X4` for
    BLOCK_16X16 with rigged `tx_depth` values 0, 1, 2.
  * `read_block_tx_size_inter_arm_returns_var_tx_size_stub` — the
    §5.11.17 deferral surface.
  * `read_block_tx_size_inter_skip_else_arm_uses_max_rect_tx_size` —
    the inter+skip path falls through the `else` arm with
    `allowSelect = false`.
  * `read_block_tx_size_rejects_out_of_range` — the three
    caller-bug guards.
  * `decode_block_syntax_with_tx_mode_select_reaches_compute_prediction`
    — the integrated walker's intra-arm reach to the §5.11.30 stub
    after `read_block_tx_size` succeeds.

  Plus 5 new unit tests (`src/cdf.rs`) for
  [`MAX_TX_SIZE_RECT`] square-block identity,
  [`MAX_TX_DEPTH_TABLE`] spec-listing match,
  [`SPLIT_TX_SIZE`] recursive-split contract,
  [`tx_depth_ctx`] combination-table sanity, and
  [`MAX_VARTX_DEPTH`] value.

  Updated existing integration tests for the parameter-list growth
  and the new walker short-circuit error (`DecodeBlockComputePredictionUnsupported`
  in place of `DecodeBlockReadBlockTxSizeUnsupported`).

  The §5.11.5 calls that remain stubbed (each becomes the
  next-round target):

  * §5.11.17 `read_var_tx_size()` — the variable-transform-tree
    recursion the §5.11.16 inter-arm enters.
  * §5.11.18 `inter_frame_mode_info()` — the inter arm of §5.11.6
    `mode_info()`.
  * §5.11.30 `compute_prediction()` — the immediate next-round
    target on the intra arm; the walker now reaches this stub
    after the §5.11.16 pass completes.
  * §5.11.34 `residual()` — reachable once §5.11.30 lands.

  `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

* **Round 166 — §5.11.5 `decode_block()` syntax-walker skeleton
  (`decode_block_syntax` + `decode_partition_syntax` + `DecodedBlock`).**
  Lands the missing §5.11.5 per-block syntax-walker dispatcher that
  the §5.11.4 partition walker recurses into at every leaf. Until
  this round the §5.11.4 `decode_partition` walker had no body to
  invoke at a `PARTITION_NONE` leaf — it stamped `MiSizes[]` and
  emitted a `DecodedBlockRecord` but didn't run the §5.11.5 per-block
  syntax pass. The new
  [`PartitionWalker::decode_block_syntax`] method performs the
  §5.11.5 prologue + §5.11.6 `mode_info()` intra arm + §5.11.49
  `palette_tokens()` (no-op on the no-palette path) and short-
  circuits at the §5.11.16 `read_block_tx_size()` stub — the
  immediate next-round target.

  §5.11.5 prologue implementation (av1-spec p.63 lines 1-25):
  `MiRow = r`, `MiCol = c`, `MiSize = subSize`, `bw4 =
  Num_4x4_Blocks_Wide[subSize]`, `bh4 = Num_4x4_Blocks_High[subSize]`;
  the `HasChroma` three-arm dispatch (`bh4 == 1 && subsampling_y &&
  (MiRow & 1) == 0 ⇒ false`; `bw4 == 1 && subsampling_x && (MiCol &
  1) == 0 ⇒ false`; else `num_planes > 1`); `AvailU = is_inside(r-1,
  c)`, `AvailL = is_inside(r, c-1)` via
  [`TileGeometry::is_inside`]; `AvailUChroma` / `AvailLChroma` with
  the spec's chroma fix-up arm (`subsampling_y && bh4 == 1 ⇒
  is_inside(r-2, c)`; `subsampling_x && bw4 == 1 ⇒ is_inside(r, c-2)`).
  All five derivations surface on the returned [`DecodedBlock`]
  aggregate.

  §5.11.6 `mode_info()` dispatch (av1-spec p.64): `frame_is_intra =
  false` short-circuits with the new
  `Error::DecodeBlockInterFrameUnsupported` (zero bits consumed —
  the §5.11.5 prologue doesn't touch the bit cursor). The implemented
  intra arm composes the §5.11.7 `intra_frame_mode_info` body
  through r161's
  [`PartitionWalker::decode_intra_frame_mode_info_prefix`]
  (lines 1-10: `skip = 0`, `intra_segment_id`, `skip_mode = 0`,
  `read_skip`, post-skip `intra_segment_id`, `read_cdef`,
  `read_delta_qindex`, `read_delta_lf`, `ReadDeltas = 0`, the fixed
  `RefFrame[0..2] = [INTRA_FRAME, NONE]` assignments) + r164's
  [`PartitionWalker::decode_use_intrabc`] + r165's
  [`PartitionWalker::decode_intra_frame_y_mode`] (on the `use_intrabc
  == 0` arm). The `use_intrabc == 1` arm short-circuits to `YMode =
  DC_PRED`, `is_inter = 1` (the find_mv_stack / assign_mv body is a
  next-round target).

  §5.11.49 `palette_tokens()`: gated by the spec's `if
  ( PaletteSize{Y,UV} )` outer guard. On the reachable path
  (`palette_mode_info()` is not yet wired ⇒ `PaletteSize{Y,UV} ==
  0`), the call is a no-op per the spec.

  Stub fire at §5.11.16: the walker emits a [`DecodedBlockRecord`]
  leaf (matching the legacy `decode_partition` leaf emitter so the
  `blocks()` accessor reports the same record either path) and
  returns `Error::DecodeBlockReadBlockTxSizeUnsupported`. The
  §5.11.5 grid-fill writes that already happen inside the composed
  leaves (`Skips[][]`, `SkipModes[][]`, `SegmentIds[][]`,
  `MiSizes[][]`, `YModes[][]`, `cdef_idx[][]`) remain observable on
  the walker's grid accessors after the stub fires.

  Four new `Error` variants surface the §5.11.5 next-round
  boundaries one-to-one:
    * `DecodeBlockInterFrameUnsupported` — §5.11.18
      `inter_frame_mode_info()` (the inter arm of §5.11.6).
    * `DecodeBlockReadBlockTxSizeUnsupported` — §5.11.16
      `read_block_tx_size()`, the immediate next stub the walker
      reaches.
    * `DecodeBlockComputePredictionUnsupported` — §5.11.30
      `compute_prediction()`, reserved for the round that lands
      §5.11.16.
    * `DecodeBlockResidualUnsupported` — §5.11.34 `residual()`,
      reserved for the round that lands §5.11.30.

  New `DecodedBlock` per-block aggregate (publicly constructible)
  carries every value the §5.11.5 walker derives:
    * Prologue: `mi_row` / `mi_col` / `mi_size` / `bw4` / `bh4` /
      `has_chroma` / `avail_u` / `avail_l` / `avail_u_chroma` /
      `avail_l_chroma`.
    * Mode-info pass: every `IntraFrameModeInfoPrefix` field plus
      `use_intrabc` / `is_inter` / `y_mode` / `is_compound`
      (always `false` on the intra arm since `RefFrame[1] = NONE
      < INTRA_FRAME`).

  Partition-walker driver:
  [`PartitionWalker::decode_partition_syntax`] mirrors the §5.11.4
  recursion of [`PartitionWalker::decode_partition`] verbatim
  (identical `partition` / `split_or_horz` / `split_or_vert`
  reads, identical `partition_subsize` dispatch, identical
  edge-of-frame fall-throughs); the only difference is that every
  `decode_block( r, c, sz )` site in the §5.11.4 pseudocode now
  routes through `decode_block_syntax` instead of the leaf-only
  emitter. Stub propagates from the first leaf that fires it; grid
  stamps from earlier leaves remain observable.

  10 new integration tests (`tests/decode_block_syntax_walker.rs`):
    * `decode_block_syntax_reaches_read_block_tx_size_stub_after_intra_mode_info`
      — the baseline keyframe / no-segmentation / no-screen-content
      path reaches the §5.11.16 stub, the bitstream cursor advanced
      past the prologue, one leaf record emitted at (0, 0,
      BLOCK_8X8), and the BLOCK_8X8 footprint of `Skips[]`,
      `SegmentIds[]`, `MiSizes[]` carries the expected stamps.
    * `decode_block_syntax_prologue_has_chroma_three_arm_dispatch` —
      the §5.11.5 `HasChroma` three-arm dispatch on `BLOCK_4X4`
      (subsampling-y arm 1, subsampling-x arm 2, no-subsampling
      fall-through arm 3); each subcase reaches the §5.11.16 stub.
    * `decode_block_syntax_inter_frame_arm_returns_stub` —
      `frame_is_intra = false` ⇒ `Error::DecodeBlockInterFrameUnsupported`
      with zero bits consumed and no leaf record emitted.
    * `decode_block_syntax_intra_pre_skip_arm_reaches_stub` —
      `seg_id_pre_skip = true` + `segmentation_enabled = true` ⇒
      the §5.11.9 `intra_segment_id` read fires before
      `read_skip`, and the §5.11.16 stub is reached after the
      composed reads (with the BLOCK_8X8 footprint stamped).
    * `decode_block_syntax_rejects_out_of_range` — `mi_row >=
      MiRows`, `mi_col >= MiCols`, and `sub_size >= BLOCK_SIZES`
      each surface `Error::PartitionWalkOutOfRange` before any bit
      is read.
    * `decode_partition_syntax_routes_leaf_through_decode_block_syntax`
      — a `BLOCK_4X4` superblock (the `< BLOCK_8X8` short-circuit
      arm) drives the partition walker into exactly one
      `decode_block_syntax` call at (0, 0, BLOCK_4X4), which
      surfaces the §5.11.16 stub; one leaf record emitted.
    * `decode_partition_syntax_out_of_grid_short_circuits` — `r >=
      MiRows` triggers the §5.11.4 line-1 early return with
      `Ok(())` and no leaf records.
    * `decoded_block_struct_public_api_smoke` — `DecodedBlock` is
      publicly constructible with every field default-valid; the
      struct is `Debug + Clone + Copy + PartialEq + Eq`.
    * `decode_block_syntax_block_8x16_grid_fill_footprint` — a
      `BLOCK_8X16` block (bw4 = 2, bh4 = 4) at (0, 0) stamps the
      §5.11.5 grid-fill 2×4 footprint on `MiSizes[]`; cells
      outside the footprint stay at the `BLOCK_INVALID` sentinel.
    * `decode_block_syntax_cdef_bits_two_reaches_stub` — with
      `cdef_bits = 2` the §5.11.56 literal-bits read consumes two
      bits and the walker still reaches the §5.11.16 stub on a
      `BLOCK_16X16` block; the 4×4 footprint of `MiSizes[]` and
      `Skips[]` stamped.

  The §5.11.7 follow-on `else`-arm elements
  (`intra_angle_info_y`, `uv_mode`, `read_cfl_alphas`,
  `intra_angle_info_uv`, `palette_mode_info`,
  `filter_intra_mode_info`) and the `use_intrabc == 1` MV-stack /
  assign-mv body remain bounded leaf targets that can be slotted
  into a future round before or alongside §5.11.16. `decode_av1` /
  `encode_av1` continue to return `Error::NotImplemented`.

* **Round 165 — §5.11.7 / §5.11.22 `intra_frame_y_mode` syntax element
  (`decode_intra_frame_y_mode`).**
  Lands a new [`PartitionWalker::decode_intra_frame_y_mode`] method
  implementing the §5.11.7 `intra_frame_y_mode` syntax element
  (av1-spec p.65) — the per-block luma intra-prediction-mode selector
  read on the §5.11.7 `intra_frame_mode_info()` `else` arm
  (`use_intrabc == 0`), immediately after the `is_inter = 0`
  assignment that r164's [`PartitionWalker::decode_use_intrabc`]
  fall-through arm produced. The spec body is the two-line
  `intra_frame_y_mode S(); YMode = intra_frame_y_mode`; the
  dispatcher reads a single `S()` symbol against the §8.3.2
  `intra_frame_y_mode` ctx-selected CDF and stamps the result over
  the block's `bw4 * bh4` footprint of a new `YModes[][]` grid.

  New `PartitionWalker::y_modes: Vec<u8>` field — a `mi_rows *
  mi_cols` row-major buffer covering the §6.10.4 `YModes[ r ][ c ]`
  grid (av1-spec p.378 / §5.11.5 line `YModes[ r + y ][ c + x ] =
  YMode`). Cells are initialised to `0` (= `DC_PRED`, the §3
  intra-mode enumeration's ordinal-zero value); the initial-zero
  state matches the §8.3.2 `intra_frame_y_mode` ctx walk's
  "neighbour unavailable" arm, where the spec writes `abovemode =
  Intra_Mode_Context[ AvailU ? YModes[ MiRow - 1 ][ MiCol ] :
  DC_PRED ]` (an unavailable-or-pre-write neighbour contributes the
  same `Intra_Mode_Context[ DC_PRED ] = 0` weight). The new
  `y_modes()` read accessor surfaces the grid; a `y_mode_at`
  private helper performs the bounds-clipped neighbour lookup.

  §8.3.2 ctx derivation honours both §5.11.51 tile-bound predicates
  (`AvailU` / `AvailL` via [`TileGeometry::is_inside`]) and the
  §8.3.2 `Intra_Mode_Context[]` mapping (driven through the existing
  [`intra_mode_ctx`] helper). The CDF row is selected via the
  existing [`TileCdfContext::intra_frame_y_mode_cdf`] accessor —
  no new `Default_*` table is added (r127 already transcribed the
  §9.4 `Default_Intra_Frame_Y_Mode_Cdf` array verbatim).

  7 new cdf-module tests (539 → 546):
    * `fresh_walker_y_modes_grid_is_dc_pred` — the constructor's
      pre-fill matches the §5.11.5 `YModes[]` initial-zero (=
      `DC_PRED`) state across every cell.
    * `decode_intra_frame_y_mode_rejects_out_of_range` — `sub_size
      >= BLOCK_SIZES` / `mi_row >= MiRows` / `mi_col >= MiCols`
      all surface `PartitionWalkOutOfRange` before any bit is read.
    * `decode_intra_frame_y_mode_returns_symbol_zero_and_stamps_grid`
      — a rigged CDF forcing symbol 0 (`DC_PRED`) is returned, and
      the value is stamped across the `BLOCK_16X16` footprint
      (4×4 mi cells).
    * `decode_intra_frame_y_mode_returns_symbol_max_and_stamps_grid`
      — a rigged CDF forcing symbol 12 (`PAETH_PRED`, the largest
      valid `YMode`) is returned and stamped across a `BLOCK_8X8`
      footprint at (4,4); a non-footprint cell at (0,0) remains
      `DC_PRED`.
    * `decode_intra_frame_y_mode_corner_uses_dc_pred_neighbours` —
      the (0,0) top-left block routes through ctx `(abovemode=0,
      leftmode=0)` (both neighbours unavailable ⇒ both map to
      `DC_PRED` ⇒ both ctx indices are 0).
    * `decode_intra_frame_y_mode_reads_neighbour_ymodes_grid` —
      after a first block at (0,0) stamps `YMode = V_PRED = 1`
      across its footprint, the next block at (0, bw4) routes
      through ctx `(abovemode=0, leftmode=1)`, observable through
      the rigged distinctive-symbol return.
    * `decode_intra_frame_y_mode_applies_intra_mode_context_mapping`
      — stamping a `YMode = 4` (`D203_PRED`) neighbour selects a
      leftmode_ctx of `4` per the §8.3.2 `Intra_Mode_Context[]`
      table (`[0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0]`).

  Together with r164's [`PartitionWalker::decode_use_intrabc`] this
  closes the first leaf of the §5.11.7 `else` arm and the first
  sub-element of the §5.11.22 `intra_block_mode_info` composite. The
  remaining §5.11.7 `else`-arm elements (`intra_angle_info_y`,
  `uv_mode`, `read_cfl_alphas`, `intra_angle_info_uv`,
  `palette_mode_info`, `filter_intra_mode_info`) remain bounded leaf
  targets for r166+. The `use_intrabc == 1` short-circuit
  (`find_mv_stack(0)` / `assign_mv(0)`) still awaits the
  motion-vector stack walker. `decode_av1` / `encode_av1` continue
  to return `Error::NotImplemented`.

* **Round 164 — §5.11.7 `use_intrabc` syntax element (`decode_use_intrabc`).**
  Lands a new [`PartitionWalker::decode_use_intrabc`] method
  implementing the §5.11.7 `use_intrabc` syntax element (av1-spec
  p.65): the per-block intra-block-copy enable bit read on the
  §5.11.7 `intra_frame_mode_info()` body immediately after the
  `RefFrame[ 0..2 ]` assignments that r161
  [`PartitionWalker::decode_intra_frame_mode_info_prefix`] produced.
  The spec body is the two-arm `if ( allow_intrabc ) { use_intrabc
  S() } else { use_intrabc = 0 }`; the dispatcher routes both
  arms exactly, with no bit consumed on the fall-through.

  New `DEFAULT_INTRABC_CDF: [u16; 3] = [30531, 32768, 0]` table
  verbatim from §9.4 (av1-spec p.430) — a single-row binary CDF
  with no context index (the §8.3.2 selection text is "use_intrabc:
  The cdf for use_intrabc is given by TileIntrabcCdf" with no
  `[ctx]` subscript, mirroring the `Default_Delta_Q_Cdf` /
  `Default_Delta_Lf_Cdf` shape). New `TileCdfContext::intrabc`
  field initialised in `new_from_defaults` from
  `DEFAULT_INTRABC_CDF`, plus the `TileCdfContext::intrabc_cdf()`
  selector implementing the §8.3.2 contextless lookup.
  `DEFAULT_INTRABC_CDF` re-exported at the crate root.

  Unlike the §5.11.5 `decode_skip` / `decode_skip_mode` /
  `decode_is_inter` siblings, `decode_use_intrabc` writes nothing
  to the walker's §5.11.5 grids (`Skips[]` / `SkipModes[]` /
  `IsInters[]` / `SegmentIds[]`) — AV1 has no per-block
  `UseIntrabc[][]` map. The value is consumed locally by the
  §5.11.7 follow-on arm (the `is_inter = 1` / `YMode = DC_PRED` /
  `find_mv_stack(0)` / `assign_mv(0)` short-circuit when
  `use_intrabc == 1`, vs. the `intra_block_mode_info` composite
  when `use_intrabc == 0`); both follow-on arms remain the next
  round's targets.

  7 new cdf-module tests (532 → 539):
    * `decode_use_intrabc_allow_false_no_read` — fall-through arm
      consumes zero bits on a hostile `0xFF` buffer.
    * `decode_use_intrabc_allow_true_returns_symbol_zero` /
      `_returns_symbol_one` — rigged S() arm against
      `force_binary_cdf(0)` / `(1)` returns the forced symbol.
    * `decode_use_intrabc_rejects_out_of_range` — `sub_size >=
      BLOCK_SIZES` / `mi_row >= MiRows` / `mi_col >= MiCols` all
      surface `PartitionWalkOutOfRange`.
    * `decode_use_intrabc_contextless_cdf_selection` — three
      distinct `(mi_row, mi_col, sub_size)` triples all select the
      same `TileIntrabcCdf` row (confirming the §8.3.2 selection
      is contextless).
    * `default_intrabc_cdf_layout_and_accessor` —
      `DEFAULT_INTRABC_CDF == [30531, 32768, 0]` verbatim from
      §9.4, fresh `TileCdfContext::intrabc` matches, and the
      `intrabc_cdf` accessor returns the same row.
    * `decode_use_intrabc_does_not_stamp_grids` — after a
      `decode_use_intrabc` call, all of `Skips[]` / `SkipModes[]` /
      `IsInters[]` stay zeroed (no §5.11.5-style footprint stamp).

  The §5.11.7 follow-on body now divides cleanly into two
  remaining unblocked arms: (i) the `use_intrabc == 1`
  short-circuit (`is_inter = 1`, `YMode = DC_PRED`, `UVMode =
  DC_PRED`, `motion_mode = SIMPLE`, `compound_type =
  COMPOUND_AVERAGE`, palette sizes = 0, `interp_filter[0..2] =
  BILINEAR`, `find_mv_stack(0)`, `assign_mv(0)`) — needing the
  motion-vector stack walker; and (ii) the `intra_block_mode_info`
  composite (`intra_frame_y_mode`, `intra_angle_info_y`, `uv_mode`,
  `read_cfl_alphas`, `intra_angle_info_uv`, `palette_mode_info`,
  `filter_intra_mode_info`) — each a bounded leaf, queued for
  r165+. `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

* **Round 163 — §5.11.21 `get_segment_id()` predicted-segment-id helper.**
  Lands a new free function [`cdf::get_segment_id`] (re-exported at the
  crate root) implementing the §5.11.21 spec body (av1-spec p.72): the
  inter-frame per-block segment-id **prediction** lookup over the
  previous frame's `PrevSegmentIds[][]` map. The function scans the
  `xMis = Min(MiCols - MiCol, bw4)` × `yMis = Min(MiRows - MiRow, bh4)`
  on-screen window covered by the current block and returns the
  `Min` of the cells visited, with the §5.11.21 sentinel `seg = 7`
  (i.e. `MAX_SEGMENTS - 1`) as the upper bound.

  Signature: `get_segment_id(prev_segment_ids: &[i32], prev_mi_rows: u32,
  prev_mi_cols: u32, mi_rows: u32, mi_cols: u32, mi_row: u32, mi_col:
  u32, sub_size: usize) -> Option<i32>`. The previous-frame segmentation
  surface is passed as a row-major `&[i32]` slice (matching the
  `PartitionWalker::segment_ids` layout — `i32` so the §5.11.9 `-1`
  sentinel for an unwritten cell round-trips faithfully). Returns
  `Some(seg)` with `seg` in `-1..=7`; the `-1` value surfaces only if
  a not-yet-decoded cell of a previous walker falls inside the window,
  letting callers detect a malformed reference surface via the existing
  §5.11.19 `predicted_segment_id > last_active_seg_id` range guard.
  Returns `None` for caller-bug arguments: out-of-range `sub_size`,
  anchor outside the current frame, previous-frame extent smaller than
  the current frame's, or `prev_segment_ids.len() != prev_mi_rows *
  prev_mi_cols`.

  The function is **pure** — no walker state, no bitreader, no CDF —
  and complements the r162
  [`cdf::PartitionWalker::decode_inter_segment_id`] caller, which
  takes a pre-computed `predicted_segment_id: u8` so the walker can
  stay inter-frame-state-free. The §6.10 reference-frame walk now
  has a verbatim spec-shaped routine to compute that argument from
  `PrevSegmentIds[]`.

  12 new cdf-module tests (520 → 532): uniform-0 / uniform-7
  reductions over several block sizes; explicit Min-over-2x2-window
  cells with out-of-window decoy values that must not contribute;
  frame-edge `xMis`/`yMis` clipping on a `BLOCK_16X16` anchored at the
  bottom-right of a 4x4 mi-grid; `-1` sentinel round-trip; a
  wider-than-current previous frame exercising the `prev_mi_cols`
  row stride; single-cell `BLOCK_4X4` covering exactly
  `prev[MiRow][MiCol]` with a neighbour-cell decoy; out-of-range
  guards for invalid `sub_size`, anchor past frame extent,
  previous-frame extent smaller than current frame's, and length /
  shape mismatch; an end-to-end composition test feeding
  `get_segment_id`'s result into `decode_inter_segment_id`'s no-read
  `!segmentation_update_map` arm and verifying the predicted id is
  adopted with zero bit reads on a hostile `0xFF` buffer.

  The §5.11.18 `inter_frame_mode_info()` top-level dispatcher, §5.11.7
  `use_intrabc` arm, and §5.11.22 `intra_block_mode_info` composite
  remain the next round's targets. `decode_av1` / `encode_av1`
  continue to return `Error::NotImplemented`.

* **Round 162 — §5.11.19 `inter_segment_id( preSkip )` syntax element.**
  Lands a new
  [`PartitionWalker::decode_inter_segment_id`] method implementing the
  full §5.11.19 spec body (av1-spec p.71): the §5.11.18 inter-frame
  per-block segment-id read, called twice per block (with `preSkip =
  1` before §5.11.11 `read_skip()` and `preSkip = 0` after) so the
  §5.9.14 `SegIdPreSkip` derivation routes the segment-id read to the
  intended position relative to `skip`.

  New `SEGMENT_ID_PREDICTED_CONTEXTS = 3` constant (§9.3) and new
  [`DEFAULT_SEGMENT_ID_PREDICTED_CDF`] table verbatim from §9.4
  (av1-spec p.442 — three uniform `[128 * 128, 32768, 0]` rows, the
  §8.3.1 binary-symbol start). New
  [`TileCdfContext::segment_id_predicted`] field initialised in
  `new_from_defaults` from `DEFAULT_SEGMENT_ID_PREDICTED_CDF`, plus
  the [`TileCdfContext::segment_id_predicted_cdf`] selector
  implementing the §8.3.2 `TileSegmentIdPredictedCdf[ ctx ]` index.

  New persistent `above_seg_pred_context: Vec<u8>` (length `mi_cols`)
  and `left_seg_pred_context: Vec<u8>` (length `mi_rows`) buffers on
  `PartitionWalker` per the §8.3.1 tile-entry initialisation
  (`AboveSegPredContext[i] = 0`, `LeftSegPredContext[i] = 0` for
  every column/row). The §8.3.2 ctx walk is `ctx =
  LeftSegPredContext[ MiRow ] + AboveSegPredContext[ MiCol ]` (each
  in `0..=1`; sum in `0..SEGMENT_ID_PREDICTED_CONTEXTS = 0..3`).
  Public read-only accessors
  [`PartitionWalker::above_seg_pred_context`] /
  [`PartitionWalker::left_seg_pred_context`] surface flat views.

  The dispatcher routes the full §5.11.19 cascade exactly:

  * outer `!segmentation_enabled` → `segment_id = 0`, no read, grid
    stamped, context arrays untouched;
  * inner `!segmentation_update_map` → `segment_id =
    predictedSegmentId`, no read, grid stamped, context arrays
    untouched;
  * `pre_skip && !SegIdPreSkip` early-exit → `segment_id = 0`, no
    read, grid stamped, context arrays untouched;
  * `!pre_skip && skip != 0` post-skip-with-skip arm → context arrays
    zeroed over the `bw4`/`bh4` footprint, then `decode_segment_id`
    is called (the §5.11.9 path short-circuits on `skip`);
  * `segmentation_temporal_update == 1` → reads the binary
    `seg_id_predicted` symbol against the §8.3.2 cdf; on `1` adopts
    `predictedSegmentId` (no further read), on `0` descends into
    `decode_segment_id`; stamps context arrays with the just-read
    flag;
  * `segmentation_temporal_update == 0` fall-through → straight
    `decode_segment_id` call; context arrays untouched (per spec).

  `predicted_segment_id` (§5.11.21 `get_segment_id()` over
  `PrevSegmentIds[]`) is caller-supplied so the walker stays
  inter-frame-state-free — the current-frame `segment_ids[]` grid is
  the only segmentation surface it owns.

  Range guards (`sub_size >= BLOCK_SIZES`, `mi_row >= MiRows`,
  `mi_col >= MiCols`, `last_active_seg_id >= MAX_SEGMENTS`,
  `predicted_segment_id > last_active_seg_id`) fire up-front on every
  arm so the no-symbol paths are total over the same input space as
  the bitstream-reading paths (matching the r160 `decode_intra_segment_id`
  pattern).

  11 new cdf-module tests (509 → 520):

  * `fresh_walker_seg_pred_context_is_zero` — the §8.3.1 tile-entry
    `AboveSegPredContext[]` / `LeftSegPredContext[]` arrays are sized
    `MiCols` / `MiRows` and all-zero.
  * `decode_inter_segment_id_segmentation_disabled_no_read` —
    `!segmentation_enabled` on both `pre_skip = true` and `false`
    yields `segment_id = 0` with no bits consumed on a hostile `0xFF`
    buffer; grid stamped over the BLOCK_8X8 footprint; context
    arrays untouched.
  * `decode_inter_segment_id_no_update_map_adopts_predicted` —
    `!segmentation_update_map` adopts `predictedSegmentId` without
    reading any bits; grid stamped with the predicted id; context
    arrays untouched.
  * `decode_inter_segment_id_pre_skip_with_post_skip_pre_skip_flag_returns_zero`
    — `pre_skip && !SegIdPreSkip` early-exit returns `segment_id = 0`
    with no bits consumed; grid stamped to 0; context arrays
    untouched.
  * `decode_inter_segment_id_post_skip_with_skip_clears_context_and_short_circuits`
    — the `!pre_skip && skip` arm zeroes the context arrays over the
    footprint (verified by poisoning the arrays to `1` first); the
    inner §5.11.9 `decode_segment_id` skip short-circuit fires with
    no `S()` consumed; columns/rows outside the footprint retain
    the poison value.
  * `decode_inter_segment_id_temporal_update_predicted_adopts_predicted_id`
    — `temporal_update == 1` + rigged `seg_id_predicted = 1` adopts
    `predictedSegmentId` (no §5.11.9 descent); context arrays stamped
    to `1` over the footprint; grid stamped with the predicted id.
  * `decode_inter_segment_id_temporal_update_unpredicted_reads_segment_id`
    — `temporal_update == 1` + rigged `seg_id_predicted = 0` descends
    into `decode_segment_id`; context arrays stamped to `0` over the
    footprint (verified by poisoning first); grid stamped with the
    `decode_segment_id` return value.
  * `decode_inter_segment_id_no_temporal_update_reads_segment_id_only`
    — `temporal_update == 0` fall-through reads a literal
    `read_segment_id()` without touching the context arrays (verified
    by poisoning and confirming the poison survives).
  * `decode_inter_segment_id_rejects_out_of_range` — five-way
    out-of-range guard (`mi_row`, `mi_col`, `sub_size`,
    `last_active_seg_id`, and the new `predicted_segment_id >
    last_active_seg_id` invariant).
  * `default_segment_id_predicted_cdf_layout` — the §9.4 table
    transcription matches `[16384, 32768, 0]` per ctx row.
  * `segment_id_predicted_cdf_accessor_round_trip` — the §8.3.2
    selector round-trips through mutation.

  The §5.11.18 `inter_frame_mode_info()` top-level dispatcher
  (`use_intrabc` arm + the `LeftRefFrame` / `AboveRefFrame` /
  `LeftIntra` / `AboveIntra` / `LeftSingle` / `AboveSingle`
  derivations + the §5.11.18 two-call `inter_segment_id` protocol
  composing on top of r152 `read_skip()` / r156 `read_cdef()` /
  r154 `read_delta_qindex()` / r155 `read_delta_lf()` / r158
  `read_is_inter()` / §5.11.22 `intra_block_mode_info` / §5.11.23
  `inter_block_mode_info`) is the next round's architectural
  payoff. `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

  [`DEFAULT_SEGMENT_ID_PREDICTED_CDF`]: crate::cdf::DEFAULT_SEGMENT_ID_PREDICTED_CDF
  [`PartitionWalker::decode_inter_segment_id`]: crate::cdf::PartitionWalker::decode_inter_segment_id
  [`PartitionWalker::above_seg_pred_context`]: crate::cdf::PartitionWalker::above_seg_pred_context
  [`PartitionWalker::left_seg_pred_context`]: crate::cdf::PartitionWalker::left_seg_pred_context
  [`TileCdfContext::segment_id_predicted`]: crate::cdf::TileCdfContext::segment_id_predicted
  [`TileCdfContext::segment_id_predicted_cdf`]: crate::cdf::TileCdfContext::segment_id_predicted_cdf

* **Round 161 — §5.11.7 `intra_frame_mode_info()` prefix dispatcher.**
  Lands a new
  [`PartitionWalker::decode_intra_frame_mode_info_prefix`] method
  composing the first 11 lines of the §5.11.7 spec body
  (av1-spec p.64) into a single walker entry-point: `skip = 0`;
  conditional pre-skip `intra_segment_id()`; `skip_mode = 0`;
  `read_skip()`; conditional post-skip `intra_segment_id()`;
  `read_cdef()`; `read_delta_qindex()`; `read_delta_lf()`; and the
  fixed `RefFrame[0] = INTRA_FRAME` / `RefFrame[1] = NONE` trailing
  assignments. Returns a new public
  [`cdf::IntraFrameModeInfoPrefix`] struct carrying every post-call
  observable: `skip`, `skip_mode` (always `0` on intra-frame walks),
  `segment_id`, `lossless`, `cdef_idx`, `current_q_index`,
  `current_delta_lf`, `ref_frame`.

  The §5.11.7 `SegIdPreSkip` conditional routes the §5.11.8 call
  before or after the §5.11.11 `read_skip()` per the caller-passed
  `seg_id_pre_skip` boolean (the §5.9.14 trailing derivation
  surfaced as
  [`SegmentationParams::seg_id_pre_skip`]). `skip_mode` is fixed at
  `0` because the intra-frame walk never reaches `decode_skip_mode`
  — §5.11.10 short-circuits on `!skip_mode_present` (intra-only
  frames have `skip_mode_present == 0` per §5.9.21). The §6.10.4
  `ReadDeltas = 0` line of §5.11.7 is left to the caller (the
  walker stays stateless about per-superblock first-block
  detection), matching the §6.10.4 pattern existing
  [`PartitionWalker::decode_delta_qindex`] /
  [`PartitionWalker::decode_delta_lf`] call sites already use.

  Range guards (`sub_size >= BLOCK_SIZES`, `mi_row >= MiRows`,
  `mi_col >= MiCols`, `last_active_seg_id >= MAX_SEGMENTS`,
  `cdef_bits > 3`) fire on the dispatcher level before any inner
  read so a caller bug never produces a partial-read.

  8 new cdf-module tests (501 → 509):

  * `decode_intra_frame_mode_info_prefix_minimum_path` — single
    `S()` consumed for `read_skip`; every other field
    short-circuits on its caller-passed gate.
  * `decode_intra_frame_mode_info_prefix_pre_skip_arm_reads_segment_first`
    — `SegIdPreSkip = true` reads `segment_id` (diff = 2 against a
    rigged cdf) before `skip` (forced 1); `cdef_idx = -1` because
    `skip != 0` triggers the §5.11.56 short-circuit.
  * `decode_intra_frame_mode_info_prefix_post_skip_arm_segment_after_skip`
    — `SegIdPreSkip = false` arm: `read_skip` first then
    `intra_segment_id` with `skip = 1` ⇒ §5.11.9 short-circuit
    fires (`segment_id = pred = 0`, no bit consumed).
  * `decode_intra_frame_mode_info_prefix_seg_skip_active_no_skip_bit`
    — `seg_skip_active = true` forces `skip = 1` with zero bits
    consumed on a hostile `0xFF` buffer (also asserts the §5.11.56
    `cdef_idx = -1` sentinel survives).
  * `decode_intra_frame_mode_info_prefix_ref_frame_fixed` —
    `ref_frame = [INTRA_FRAME, NONE] = [0, -1]` regardless of path
    taken.
  * `decode_intra_frame_mode_info_prefix_read_deltas_routes_through`
    — `read_deltas = true` + `delta_lf_present = true` drives both
    delta reads; rigged `delta_q_abs = 0` / `delta_lf_abs = 0` ⇒
    accumulators unchanged but the `S()` reads advance the
    decoder.
  * `decode_intra_frame_mode_info_prefix_rejects_out_of_range` —
    five-way out-of-range guard on `mi_row`, `mi_col`, `sub_size`,
    `last_active_seg_id`, `cdef_bits`.
  * `decode_intra_frame_mode_info_prefix_skip_mode_field_always_zero`
    — `skip_mode = 0` on both pre-skip arms; the walker's
    `SkipModes[]` grid stays at the construction default (the
    dispatcher never calls `decode_skip_mode`).

  The §5.11.7 follow-on body (`use_intrabc` arm + the §5.11.22
  `intra_block_mode_info` composite — `intra_frame_y_mode`,
  `intra_angle_info_y`, `uv_mode`, `intra_angle_info_uv`,
  `palette_mode_info`, `filter_intra_mode_info`) and the
  §5.11.18 `inter_frame_mode_info` / §5.11.19 `inter_segment_id`
  two-call protocol remain the next round's targets. `decode_av1`
  / `encode_av1` continue to return `Error::NotImplemented`.

  [`SegmentationParams::seg_id_pre_skip`]: crate::uncompressed_header_tail::SegmentationParams::seg_id_pre_skip

* **Round 160 — §5.11.8 `intra_segment_id()` syntax element.** Lands
  the intra-frame variant of the per-block segment-id read (av1-spec
  p.66) as a new [`PartitionWalker::decode_intra_segment_id`] method
  built on top of r159's [`PartitionWalker::decode_segment_id`].
  `intra_segment_id` is called from §5.11.7 `intra_frame_mode_info`
  on both the `SegIdPreSkip` pre-skip arm and the `!SegIdPreSkip`
  post-skip arm. The §5.11.8 spec body is short — `if
  (segmentation_enabled) read_segment_id(); else segment_id = 0;
  Lossless = LosslessArray[segment_id]` — but the Lossless lookup
  is the first place the per-segment §6.8.2 `LosslessArray[]` table
  reaches the leaf walk.

  The dispatch is exact:

  * `segmentation_enabled = true` ⇒ descends into the r159
    implementation (which performs the §5.11.9 neighbour cascade,
    the skip / non-skip dispatch, the `S()` read against
    `TileSegmentIdCdf[ctx]`, the `neg_deinterleave` mapping, and
    the §5.11.5 grid-fill).
  * `segmentation_enabled = false` ⇒ forces `segment_id = 0`
    without reading any bits and stamps the `bh4 * bw4` footprint
    to `0` so subsequent §5.11.9 neighbour lookups see a real zero
    rather than the §5.11.9 `-1` sentinel.

  Both arms then resolve `Lossless = lossless_array[segment_id as
  usize]` from the caller-supplied `&[bool; MAX_SEGMENTS]` table
  (the §6.8.2 derivation the frame-header walk computes from
  `qindex = get_qindex(1, segmentId)` plus the five `DeltaQ?Dc` /
  `DeltaQ?Ac` offsets; `compute_coded_lossless` in `frame_header.rs`
  is the frame-wide conjunction, this round's table is the
  per-segment data the walker indexes by `segment_id`). The walker
  stays segmentation-state-free: callers pass
  `segmentation_enabled`, `last_active_seg_id`, and `lossless_array`
  per-call, mirroring the r159 pattern.

  Range guards (out-of-range `sub_size`, `mi_row` / `mi_col` past
  extent, `last_active_seg_id >= MAX_SEGMENTS`) fire on both arms
  so the no-symbol path is total over the same input space as the
  bitstream-reading path.

  Returns `(segment_id, lossless)`. The §5.11.18
  `inter_frame_mode_info` top-level dispatcher (the §5.11.19
  `inter_segment_id(preSkip)` two-call protocol) remains the next
  round's target.

  7 new cdf-module tests (494 → 501):

  * `decode_intra_segment_id_segmentation_disabled_no_read` —
    `!segmentation_enabled` ⇒ no `S()` consumed on a hostile `0xFF`
    buffer; grid stamped to `0`; `Lossless` from
    `lossless_array[0] = true`.
  * `decode_intra_segment_id_segmentation_disabled_lossless_false`
    — same arm with `lossless_array[0] = false` reports `Lossless
    = false`.
  * `decode_intra_segment_id_segmentation_enabled_skip_origin` —
    `skip = 1` at frame origin ⇒ pred = 0, no `S()` consumed, grid
    stamped to `0`, Lossless from slot 0.
  * `decode_intra_segment_id_segmentation_enabled_non_skip_reads_symbol`
    — `skip = 0` with rigged CDF reading `diff = 3` ⇒ `segment_id
    = 3`; Lossless from slot 3 = true; grid-fill verified at the
    BLOCK_8X8 footprint.
  * `decode_intra_segment_id_lossless_indexed_by_segment_id` —
    rig `diff = 5`, set `lossless_array[5] = false` while every
    other slot is `true`; expect `Lossless = false` (proves the
    lookup is per-segment, not frame-wide).
  * `decode_intra_segment_id_segmentation_disabled_grid_fill_clips`
    — `BLOCK_16X16 @ (2, 2)` in a 4×4 frame stamps only the
    in-grid 2×2 quadrant and leaves the rest at `-1`.
  * `decode_intra_segment_id_rejects_out_of_range` — five-way
    out-of-range guard (`mi_row >= mi_rows`, `mi_col >= mi_cols`,
    `sub_size == BLOCK_SIZES`, `last_active_seg_id ==
    MAX_SEGMENTS`, plus a `mi_row` guard on the
    `segmentation_enabled = true` path).

  `decode_av1` / `encode_av1` still return
  [`Error::NotImplemented`].

* **Round 159 — §5.11.9 `read_segment_id()` syntax element.** Lands
  the per-block segment-id reader (av1-spec p.66) as a new
  [`PartitionWalker::decode_segment_id`] method on the r158 walker,
  plus a `segment_ids: Vec<i32>` row-major grid sized `MiRows ×
  MiCols` (pre-filled with the §5.11.9 `-1` sentinel; cells inside a
  decoded block's `bh4 * bw4` footprint then carry the block's
  `segment_id ∈ 0..MAX_SEGMENTS = 0..8`) with a
  [`PartitionWalker::segment_ids`] read accessor. Adds the public
  module-level [`neg_deinterleave`] helper transcribing the §5.11.9
  bijection (`diff ∈ 0..max ↔ segment_id ∈ 0..max` biased toward
  values near `ref`).

  The §5.11.9 neighbour cascade is honoured exactly as spelled out:
  `prevUL` requires both [`TileGeometry::is_inside`]-derived `AvailU`
  AND `AvailL`, `prevU` and `prevL` each gate on their own edge, and
  out-of-grid neighbours fall through to the spec's `-1` sentinel.
  The four-arm `pred` derivation (`prevU == -1 ⇒ prevL/0`; `prevL ==
  -1 ⇒ prevU`; `prevUL == prevU ⇒ prevU`; else `prevL`) is preserved
  verbatim with a `#[allow(clippy::if_same_then_else)]` annotation —
  two arms happen to return `prev_u` but the predicates are
  semantically distinct ("left neighbour unavailable" vs. "above-left
  agrees with above"), and collapsing them would obscure the spec
  correspondence. The §5.11.9 dispatch distinguishes the two paths:

  * `skip != 0` ⇒ `segment_id = pred` (zero bits read; the
    spatially-predicted-on-skip semantics the spec relies on for
    skip-block segment-map continuity).
  * Else: `diff S()` against `TileSegmentIdCdf[ctx]` (ctx from the
    existing [`segment_id_ctx`] helper, which already honours the
    `-1` sentinels), then
    `segment_id = neg_deinterleave(diff, pred, last_active_seg_id +
    1)`.

  The §5.11.5 grid-fill stamps the result over the block's `bw4 *
  bh4` footprint, clipped at the frame's `MiRows` / `MiCols` extent
  so a leaf straddling the bottom or right edge fills only the
  in-grid portion. The walker stays segmentation-state-free: the
  caller-passes `last_active_seg_id` (the §5.9.14 trailing
  derivation `[`SegmentationParams::last_active_seg_id`]`) and the
  `skip` value the §5.11.11 [`PartitionWalker::decode_skip`] just
  returned.

  `decode_segment_id` is read inside both §5.11.8 `intra_segment_id`
  and §5.11.19 `inter_segment_id` (the segmentation-enabled inner
  branch in each); the latter's `preSkip` machinery is the caller's
  job. The eventual §5.11.20 [`PartitionWalker::decode_is_inter`]
  Arm 2 (`FeatureData[segment_id][SEG_LVL_REF_FRAME]`) becomes
  segment-aware once a §5.11.18 caller wires
  `decode_segment_id`'s result into the `seg_ref_frame_is_inter`
  argument the r158 method already accepts.

  Tests grow by 11 (cdf module, 483 → 494): fresh-walker grid all
  `-1`; skip short-circuit at frame origin with empty neighbours
  writes `segment_id = pred = 0` (footprint-stamped, no S() bit
  consumed on a hostile `0xFF` byte buffer); skip short-circuit
  inherits `prev_u` when `prev_l` is unavailable (a non-skip seed at
  origin stamps sid=5 over rows 0..4 cols 0..4 via `diff = 5`,
  `pred = 0`, then a skip at `(4, 0)` reads `pred = prev_u = 5`);
  non-skip path with `pred = 0` returns `diff` unchanged (the
  `if (!ref) return diff` branch); direct `neg_deinterleave` table
  exercises for the `2 * ref < max` upward branch (`pred = 2`, all
  eight `diff` values) and the `2 * ref >= max` downward branch
  (`pred = 5`, all eight `diff` values); edge cases (`ref == 0` ⇒
  identity; `ref == max - 1` ⇒ `max - diff - 1`; smallest
  non-trivial alphabet `max = 2`); ctx-0 origin selection via rigged
  rows; ctx-2 all-neighbours-match selection through three
  walker-stamped seeds; bottom-right edge clip on `BLOCK_16X16 @
  (2, 2)` in a 4×4 frame stamps only the in-grid 2×2 quadrant;
  four-way out-of-range guard (`mi_row` past extent / `mi_col` past
  extent / `sub_size == BLOCK_SIZES` / `last_active_seg_id >=
  MAX_SEGMENTS`) ⇒ `PartitionWalkOutOfRange`.

  The §5.11.5 `decode_block()` body itself (coefficient /
  motion-vector / reconstruction) remains the next round's target.
  [`decode_av1`] / [`encode_av1`] continue to return
  [`Error::NotImplemented`].

* **Round 158 — §5.11.20 `read_is_inter()` syntax element.** Lands
  the per-block intra/inter classifier (av1-spec p.71-72) as a new
  [`PartitionWalker::decode_is_inter`] method on the r157 walker,
  plus an `is_inters: Vec<u8>` row-major grid sized `MiRows ×
  MiCols` (all-zero before any leaf fires) with a
  [`PartitionWalker::is_inters`] read accessor. The four spec arms
  are dispatched in order (first match fires, three short-circuit
  arms read zero bits): (1) `skip_mode == 1` ⇒ `is_inter = 1`; (2)
  `seg_feature_active(SEG_LVL_REF_FRAME)` ⇒
  `FeatureData[segment_id][SEG_LVL_REF_FRAME] != INTRA_FRAME` (the
  caller pre-computes this into `seg_ref_frame_is_inter` so the
  walker stays segmentation-state-free, identical to r154's
  `seg_skip_mode_off` pattern); (3) `seg_feature_active(SEG_LVL_GLOBALMV)`
  ⇒ `is_inter = 1`; (4) fall-through `S()` symbol read against
  `TileIsInterCdf[ctx]` with `ctx` from the existing
  [`is_inter_ctx`] helper.

  The §8.3.2 ctx derivation samples neighbour intra-ness from the
  complement of the walker's `IsInters[]` grid (`intra =
  !is_inter`); an unavailable neighbour (gated by
  [`TileGeometry::is_inside`]) is treated as intra per §5.11.18
  (`LeftRefFrame[0] = AvailL ? RefFrames[..][0] : INTRA_FRAME`).
  The §5.11.5 grid-fill stamps the decoded value over the block's
  `bw4 * bh4` footprint, clipped at the frame's `MiRows` /
  `MiCols` extent so a leaf straddling the bottom or right edge
  fills only the in-grid portion.

  Tests grow by 15 (cdf module, 468 → 483): fresh-walker grid
  all-zero; Arm 1 skip_mode short-circuit (position-invariant on a
  hostile `0xFF` byte buffer); Arm 2 routing to intra
  (`seg_ref_frame_is_inter = false`) and to inter
  (`seg_ref_frame_is_inter = true`), both position-invariant;
  Arm 3 globalmv short-circuit (position-invariant); Arm 1 takes
  precedence over both Arm 2 and Arm 3; Arm 2 takes precedence
  over Arm 3; else-branch `S()` returning symbol 0 and symbol 1
  on rigged binary CDF rows (the `is_inter = 1` arm verifies the
  footprint grid-stamp on BLOCK_16X16 plus the four cells outside
  the footprint staying at the initial 0); ctx-0 selection at the
  frame origin (`AvailU = AvailL = false`); ctx-3 selection through
  two intra-stamping seeds at `(0, 0)` and `(0, 4)` driving the
  block at `(3, 5)`; ctx-1 through one intra + one inter neighbour
  with both available; ctx-2 through a `mi_col_start = 4` tile
  origin clearing `AvailL` plus an above-intra seed; bottom-right
  edge clip on `BLOCK_16X16 @ (2, 2)` in a 4×4 frame stamps only
  the in-grid quadrant; three-way out-of-range guard (`mi_row`
  past extent / `mi_col` past extent / `sub_size == BLOCK_SIZES`)
  ⇒ `PartitionWalkOutOfRange`.

  The §5.11.5 `decode_block()` body itself (coefficient /
  motion-vector / reconstruction) remains the next round's target.
  [`decode_av1`] / [`encode_av1`] continue to return
  [`Error::NotImplemented`].

* **Round 157 — §5.11.56 `read_cdef()` syntax element + §5.11.55
  `clear_cdef()` reset.** Lands the per-leaf CDEF-index read
  (av1-spec p.104) as a new [`PartitionWalker::decode_cdef`] method
  on the r156 walker, plus the per-superblock
  [`PartitionWalker::clear_cdef`] reset called by §5.11.2 at each
  superblock entry. Adds a `cdef_idx: Vec<i8>` row-major grid sized
  `MiRows × MiCols` to `PartitionWalker` (pre-filled with the `-1`
  sentinel from §5.11.55, interpreted as "CDEF disabled for that
  block" per §6.10.40) with a [`PartitionWalker::cdef_idx`] read
  accessor.

  CDEF operates on 64×64 anchor cells: `decode_cdef` masks the
  leaf's `(MiRow, MiCol)` to the anchor at `(MiRow & cdefMask4,
  MiCol & cdefMask4)`, where `cdefMask4 = ~(cdefSize4 - 1)` and
  `cdefSize4 = Num_4x4_Blocks_Wide[ BLOCK_64X64 ] = 16` so the
  low four bits are zeroed. If the anchor still holds the `-1`
  sentinel, an `L(cdef_bits)` literal is read (`cdef_bits ∈ 0..=3`
  per §5.9.19 `f(2)`, so the decoded value is in `0..=7`). The
  literal then stamps across the leaf's `(w4, h4)` footprint at the
  `cdefSize4 = 16` stride so super-64 blocks (`BLOCK_128X128`) reach
  all four anchor cells while sub-64 blocks touch only their
  containing anchor. Subsequent leaves whose `cdefMask4` lands on
  the same anchor short-circuit (no read; the anchor already holds
  the value — `cdef_idx[r][c] != -1` ⇒ outer `if` false).

  The §5.11.56 short-circuit set is honoured: `skip ||
  CodedLossless || !enable_cdef || allow_intrabc` ⇒ no read, the
  anchor's current value (sentinel or prior stamp) returned
  unchanged. `clear_cdef( r, c, use_128x128_superblock )` stamps
  `-1` at the one (64×64 superblock) or four (128×128 superblock)
  anchor cells per §5.11.55; out-of-grid anchors are silently
  skipped so the bottom/right superblock can straddle the frame
  edge without panic. `cdef_bits == 0` yields `L(0) = 0` (no bit
  read) and still transitions the anchor from `-1` to `0`, matching
  the §5.9.19 single-strength case.

  Tests grow by 18 (cdf module): fresh-walker all-`-1` invariant;
  `clear_cdef` 64×64 single-anchor stamp; `clear_cdef` 128×128
  four-anchor stamp; `clear_cdef` out-of-grid silent skip; each of
  the four `skip` / `CodedLossless` / `!enable_cdef` /
  `allow_intrabc` short-circuit gates (separately, with `0xFF`
  byte buffers proving no bit consumed); first-leaf-reads-literal-
  and-stamps-anchor with off-anchor cell stays at sentinel;
  second-leaf-in-anchor-no-read (cross-call position invariant);
  `cdef_bits == 0` zero-bit stamp; `cdef_bits == 3` upper-bound;
  anchor-mask routes `(10, 13)` ⇒ `(0, 0)` (leaf coords are not
  stamp coords); `BLOCK_128X128` stamps all four 64×64 anchor
  cells while off-anchor cells stay at sentinel; grid-fill clips
  at the frame edge (24×24 grid + 128×128 leaf at `(16, 16)`); the
  short-circuit returns the anchor's prior stamp (not `-1`) once a
  prior leaf has written it; `clear_cdef` after a stamp resets the
  anchor; four-way out-of-range guard (`mi_row` past extent /
  `mi_col` past extent / `sub_size == BLOCK_SIZES` / `cdef_bits >
  3`) ⇒ `Error::PartitionWalkOutOfRange`. 450 → 468 tests, zero
  `#[ignore]`.

  `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`; the §5.11.5 `decode_block()` body itself
  (coefficient / motion-vector / reconstruction) remains the next
  round's target.

* **Round 156 — §5.11.13 `read_delta_lf()` syntax element.**
  Lands the per-superblock loop-filter delta read (av1-spec p.68)
  as a new [`PartitionWalker::decode_delta_lf`] method, structurally
  parallel to the §5.11.12 `decode_delta_qindex` walker landed in
  r155 but iterating `frameLfCount` times over a four-slot
  `DeltaLF[ i ]` accumulator and selecting between the §8.3.2
  single-LF (`TileDeltaLFCdf`) and per-edge multi-LF
  (`TileDeltaLFMultiCdf[ i ]`) CDF rows via the `delta_lf_multi`
  argument. Adds a `current_delta_lf: [i32; FRAME_LF_COUNT]`
  accumulator on `PartitionWalker` with
  [`PartitionWalker::current_delta_lf`] read accessor and
  [`PartitionWalker::reset_current_delta_lf`] for the §5.11.2
  tile-entry reset.

  Honours the §5.11.13 superblock-skip short-circuit (identical
  shape to §5.11.12) and the outer `ReadDeltas && delta_lf_present`
  gate (two AND-ed flags — `delta_lf_present` is the §5.9.18
  frame-header bit, accepted as an argument). When the gate
  passes, `frameLfCount` is derived locally:
  `delta_lf_multi == 0 ⇒ 1`;
  `delta_lf_multi == 1 && mono_chrome ⇒ FRAME_LF_COUNT - 2 = 2`;
  otherwise `FRAME_LF_COUNT = 4`. Each iteration reads
  `delta_lf_abs` `S()` against the branch-selected CDF, then either
  the literal value or the §5.11.13 escape ladder
  (`delta_lf_rem_bits` `L(3)` + post-increment + `delta_lf_abs_bits`
  `L(rem_bits + 1)` ⇒ `deltaLfAbs = abs_bits + (1 << n) + 1`); for
  non-zero magnitudes reads `delta_lf_sign_bit` `L(1)` and applies
  `DeltaLF[ i ] = Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER,
  DeltaLF[ i ] + (reducedDeltaLfLevel << delta_lf_res))`.

  New constants [`DELTA_LF_SMALL = 3`], [`FRAME_LF_COUNT = 4`],
  and `cdf::MAX_LOOP_FILTER = 63i32` (distinct from the pre-existing
  `uncompressed_header_tail::MAX_LOOP_FILTER` `i16` twin from §5.9.11).
  New table [`DEFAULT_DELTA_LF_CDF`] transcribed verbatim from §9.4
  p.431 (`[28160, 32120, 32677, 32768, 0]`, identical row to
  `DEFAULT_DELTA_Q_CDF` per the spec listing — preserved as two
  independent constants so adaptation drift on one does not leak
  through the other). New fields [`TileCdfContext::delta_lf`] +
  [`TileCdfContext::delta_lf_multi`] with accessors
  [`TileCdfContext::delta_lf_cdf`] /
  [`TileCdfContext::delta_lf_multi_cdf`]. Tests grow by 17 (cdf
  module): default-CDF literal match (incl. §9.4 equality with
  `DEFAULT_DELTA_Q_CDF`); init-from-defaults invariant for both
  single-LF and all four multi-LF rows; sb-skip short-circuit at
  both `use_128x128_superblock` settings; `ReadDeltas` false
  short-circuit; `delta_lf_present` false short-circuit; single-LF
  branch writes only `DeltaLF[ 0 ]`; multi-LF colour branch writes
  all four slots; multi-LF monochrome branch writes only the two
  Y slots; zero-`delta_lf_abs` no-update; literal-positive with
  shift; Clip3 upper-bound at `MAX_LOOP_FILTER = 63`; Clip3
  lower-bound via hostile seed at `i32::MIN + 1`;
  `DELTA_LF_SMALL` escape ladder minimum value; cross-call
  accumulation; fresh-walker initial accumulator all-zero +
  `reset_current_delta_lf` round-trip; out-of-range guard. 433 ->
  450 tests, zero `#[ignore]`.

* **Round 155 — §5.11.12 `read_delta_qindex()` syntax element.**
  Lands the per-superblock quantiser-index delta read (av1-spec
  p.67) as a new [`PartitionWalker::decode_delta_qindex`] method
  on the r154 walker, plus a `CurrentQIndex` scalar tracked across
  calls with [`PartitionWalker::current_q_index`] /
  [`PartitionWalker::set_current_q_index`] accessors. Honours the
  §5.11.12 superblock-skip short-circuit (`MiSize == sbSize && skip`
  ⇒ early return, `CurrentQIndex` unchanged) with `sbSize` derived
  from the §5.5.1 `use_128x128_superblock` argument, plus the outer
  `ReadDeltas` (§6.10.4) gate. Otherwise an `S()` symbol is decoded
  against `TileDeltaQCdf` (no context index — single-row §8.3.2 CDF
  with length `DELTA_Q_SMALL + 2 = 5`); a decoded value of
  `DELTA_Q_SMALL = 3` triggers the §5.11.12 escape ladder
  (`delta_q_rem_bits` `L(3)` + post-increment + `delta_q_abs_bits`
  `L(rem_bits+1)`), reconstructing the extended absolute value via
  `delta_q_abs = delta_q_abs_bits + (1 << n) + 1` with n ∈ 1..=8
  and the final range `0..=2 ∪ 3..=511`. Non-zero `delta_q_abs`
  reads a `delta_q_sign_bit` `L(1)` and applies the §5.11.12 update
  `CurrentQIndex = Clip3(1, 255, CurrentQIndex +
  (reducedDeltaQIndex << delta_q_res))`. New constant
  [`DELTA_Q_SMALL = 3`], new table [`DEFAULT_DELTA_Q_CDF`]
  transcribed verbatim from §9.4 p.431 (`[28160, 32120, 32677,
  32768, 0]`), new field [`TileCdfContext::delta_q`] +
  [`TileCdfContext::delta_q_cdf`] accessor. Tests grow by 16 (cdf
  module): default-cdf literal match; init-from-defaults
  invariant; sb-skip short-circuit for both `use_128x128_superblock`
  values; `ReadDeltas` false short-circuit; zero-`delta_q_abs`
  no-update; literal-positive no-shift; literal-positive with
  shift; Clip3 lower-bound via hostile seed; Clip3 upper-bound;
  DELTA_Q_SMALL escape ladder minimum value; escape ladder
  in-clip-range path; cross-call accumulation; fresh-walker
  initial `CurrentQIndex = 0`; out-of-range guard; arithmetic
  decoder zero-byte sign-bit observation. 417 -> 433 tests, zero
  `#[ignore]`.

* **Round 154 — §5.11.10 `read_skip_mode()` syntax element.**
  Lands the per-block `skip_mode` read (av1-spec p.67) as a new
  [`PartitionWalker::decode_skip_mode`] method, plus a
  `SkipModes[r][c]` flag grid on the walker (parallel to the
  r152 `Skips[]` and the existing §6.10.4 `MiSizes[]` grids).
  Honours the §5.11.10 short-circuit set (any-true ⇒
  `skip_mode = 0`, no symbol read): `seg_feature_active(
  SEG_LVL_SKIP / REF_FRAME / GLOBALMV )` collapsed into the
  caller-provided `seg_skip_mode_off`; `!skip_mode_present` via
  the §5.9.21 frame-header scalar; and `Block_Width[MiSize] < 8
  || Block_Height[MiSize] < 8` derived locally from `sub_size`
  via the §9.3 [`block_width`] / [`block_height`] tables.
  Otherwise an `S()` symbol is decoded against
  `TileSkipModeCdf[ ctx ]` with the §8.3.2 ctx `ctx = AvailU *
  SkipModes[MiRow-1][MiCol] + AvailL *
  SkipModes[MiRow][MiCol-1]` (av1-spec p.378), routed through
  the existing [`skip_mode_ctx`] helper. The §5.11.5 footer's
  `SkipModes[r+y][c+x] = skip_mode` line is applied literally
  over the block's `bw4 * bh4` footprint (clipped at the frame's
  MiRows / MiCols extent). New
  [`PartitionWalker::skip_modes`] accessor returns a row-major
  view of the grid for downstream §5.11.x consumers.
  `skip_mode` is the inter-frame compound-reference shortcut
  read in [`inter_frame_mode_info`] before the rest of the inter
  mode decode; intra-only frames never call this. Tests grow by
  12 (cdf module): seg short-circuit; `skip_mode_present` false
  short-circuit; `Block_Width < 8` short-circuit (BLOCK_4X8);
  `Block_Height < 8` short-circuit (BLOCK_8X4); else branch
  returns the rigged symbol (0 / 1) on a forced binary CDF;
  else-branch stamps the `SkipModes[]` footprint; origin ctx =
  0; ctx-2 path through two prior `SkipModes=1` neighbours plus
  ctx-1 single-neighbour variants; non-zero tile origin clears
  both AvailU / AvailL; right-edge `bw4` clip; out-of-range
  (`mi_row`, `mi_col`, `sub_size`) ⇒
  `Error::PartitionWalkOutOfRange`; fresh-walker grid is all
  zero. The §5.11.5 `decode_block()` body itself (coefficient /
  motion-vector / reconstruction) remains the next round's
  target; this round adds the `SkipModes[]` grid + the
  `read_skip_mode()` read that §5.11.18 `inter_frame_mode_info`
  consults before the rest of the inter-block decode. 405 ->
  417 tests, zero `#[ignore]`.

* **Round 152 — §5.11.11 `read_skip()` syntax element.** Lands
  the per-block `skip` read (av1-spec p.65) as a new
  [`PartitionWalker::decode_skip`] method, plus a `Skips[r][c]`
  flag grid on the walker (parallel to the existing §6.10.4
  `MiSizes[]` grid). Honours both spec branches: the
  `SegIdPreSkip && seg_feature_active( SEG_LVL_SKIP )`
  short-circuit (no symbol read, `skip = 1`) is taken when the
  caller passes `seg_skip_active = true`; otherwise an `S()`
  symbol is decoded against `TileSkipCdf[ ctx ]` with the §8.3.2
  ctx `ctx = AvailU * Skips[MiRow-1][MiCol] + AvailL *
  Skips[MiRow][MiCol-1]` (av1-spec p.378). The §5.11.5 footer's
  `Skips[r+y][c+x] = skip` line is applied literally over the
  block's `bw4 * bh4` footprint (clipped at the frame's MiRows /
  MiCols extent). New [`PartitionWalker::skips`] accessor returns
  a row-major view of the grid for downstream §5.11.x consumers.
  The walker itself does not track segmentation state (the
  segment id is per-block and is read by a separate §5.11.9
  `intra_segment_id()` call site that lives in the frame parser);
  the combined `SegIdPreSkip && seg_feature_active(SEG_LVL_SKIP)`
  precondition is computed upstream and passed in. Tests grow by
  11 (cdf module): seg short-circuit returns 1 with no symbol
  read; else branch returns the rigged symbol (0 / 1) on a
  forced binary CDF; seg-branch + else-branch both stamp the
  `Skips[]` footprint; origin ctx = 0; ctx-2 path through two
  prior `Skips=1` neighbours; AvailL-false drops the
  left-neighbour contribution; non-zero tile origin clears both
  AvailU / AvailL; right-edge `bw4` clip; out-of-range
  (`mi_row`, `mi_col`, `sub_size`) ⇒
  `Error::PartitionWalkOutOfRange`; fresh-walker grid is all
  zero. The §5.11.5 `decode_block()` body itself (coefficient /
  motion-vector / reconstruction) remains the next round's
  target; this round drops the `Skips[]` grid + the `read_skip()`
  read that everything in §5.11.12 onwards reads against. 394 ->
  405 tests, zero `#[ignore]`.

* **Round 151 — §5.11.4 `decode_partition()` body.** Lands the
  recursive partition-tree walker (av1-spec p.61–62) as a new
  [`PartitionWalker`] type. The walker stitches together every
  partition prerequisite landed in rounds 137–150: the §9.4
  partition-default CDFs (r137–r144), the §8.3.2
  [`split_or_horz_cdf`] / [`split_or_vert_cdf`] binary-CDF
  derivation (r145), the §9.3 [`PARTITION_SUBSIZE`] table +
  [`partition_subsize`] accessor (r150), and the §9.3
  [`MI_WIDTH_LOG2`] / [`MI_HEIGHT_LOG2`] /
  [`NUM_4X4_BLOCKS_WIDE`] / [`NUM_4X4_BLOCKS_HIGH`] tables. The
  walker carries the §6.10.4 `MiSizes[r][c]` grid (filled at every
  leaf via the block's `bh4 * bw4` footprint), and consults it for
  the §8.3.2 [`partition_ctx`] derivation `above = AvailU &&
  (Mi_Width_Log2[ MiSizes[r-1][c] ] < bsl)` / `left = AvailL &&
  (Mi_Height_Log2[ MiSizes[r][c-1] ] < bsl)` on every recursive
  child (av1-spec p.362). The walker emits a
  `Vec<DecodedBlockRecord>` of `(MiRow, MiCol, MiSize)` leaves in
  §5.11.4 syntax order; the actual §5.11.5 `decode_block()` body
  (coefficient / motion-vector / reconstruction) stays out of
  scope. All four §5.11.4 edge-of-frame branches are honoured: the
  `r >= MiRows || c >= MiCols` early return; the `bSize <
  BLOCK_8X8` short-circuit to `PARTITION_NONE` with no symbol
  read; the `hasCols`-alone `split_or_horz` branch; the
  `hasRows`-alone `split_or_vert` branch; the `!hasRows &&
  !hasCols` fall-through to `PARTITION_SPLIT`. All ten partition
  arms (`NONE` / `HORZ` / `VERT` / `SPLIT` / `HORZ_A` / `HORZ_B`
  / `VERT_A` / `VERT_B` / `HORZ_4` / `VERT_4`) dispatch the spec's
  literal `decode_block` / recursive `decode_partition` calls with
  the appropriate `subSize` (`Partition_Subsize[ partition ][
  bSize ]`) or `splitSize` (`Partition_Subsize[ PARTITION_SPLIT
  ][ bSize ]`). The §5.11.4 bottom-right edge clip on the optional
  `HORZ_4` / `VERT_4` fourth leaf is applied literally. New
  [`TileGeometry`] type carries the four §5.11.1 mi-unit tile
  bounds for the §5.11.51 [`TileGeometry::is_inside`] test. New
  [`Error::PartitionWalkOutOfRange`] surfaces caller-bug
  preconditions (bSize out of range, partition out of range,
  unsupported bsl, BLOCK_INVALID lookup). Tests grow by 19 (cdf
  module): TileGeometry boundary cases (zero-origin + non-zero
  origin); §5.11.4 `r >= MiRows || c >= MiCols` early return; the
  `bSize < BLOCK_8X8` PARTITION_NONE short-circuit at BLOCK_4X4
  (no symbol read); §6.10.4 grid-fill at a single BLOCK_4X4 leaf;
  the `!hasRows && !hasCols` PARTITION_SPLIT fallback at a
  2×2-mi-unit frame (no symbol read at the parent); forced
  PARTITION_NONE / HORZ / VERT / HORZ_4 / VERT_4 / HORZ_A / VERT_B
  / SPLIT at BLOCK_16X16 via rigged CDFs (one leaf for NONE; two
  leaves for HORZ / VERT; four leaves for HORZ_4 / VERT_4; three
  leaves for HORZ_A / VERT_B with the documented splitSize /
  subSize pairing; four BLOCK_8X8 leaves for SPLIT in the spec's
  quadrant order); §6.10.4 grid-fill at BLOCK_16X16 with forced
  PARTITION_HORZ (both BLOCK_16X8 leaves' `bh4 * bw4` footprints
  carry `BLOCK_16X8`, surrounding cells stay `BLOCK_INVALID`);
  default-CDF W128 smoke test (non-empty leaves, all in-frame,
  all valid sub_size, total mi area bounded by frame area);
  [`partition_ctx`] derivation at the origin (`above = left =
  false ⇒ ctx = 0`) and after a leaf decode (wide neighbour
  drops the bit per `Mi_Width_Log2[ ... ] < bsl`); construction
  overflow rejection at u32::MAX × u32::MAX dimensions;
  [`PartitionWalker::take_blocks`] drain semantics. 375 -> 394
  tests, zero `#[ignore]`.

* **Round 150 — §9.3 `Partition_Subsize` table + §3 `BLOCK_*`
  enum staging.** Lands [`PARTITION_SUBSIZE`] (`[10][BLOCK_SIZES]`,
  av1-spec p.402–403) plus the typed accessor
  [`partition_subsize(partition, b_size) -> Option<usize>`] that
  folds the [`BLOCK_INVALID`] (`22`) sentinel into `None`.
  Transcription is byte-for-byte against av1-spec p.402–403; the
  spec note at p.401 ("The table will never get accessed for
  rectangular block sizes") is reflected by `BLOCK_INVALID` filling
  every rectangular `bSize` column across all 10 partition rows.
  Alongside the table, 19 named `BLOCK_*` constants land (the
  remaining members of the §3 enumeration at av1-spec p.171–172
  beyond the existing `BLOCK_8X8` (r149) and `BLOCK_128X128`
  (r112)) and the [`BLOCK_INVALID`] sentinel from the §3 constant
  table (p.7) so the table can read as the spec spells it, with no
  bare numeric literals. The [`PARTITION_TYPES_TOTAL`] constant
  (`10`) is added for the table's first dimension. Unblocks the
  §5.11.4 `decode_partition()` body (av1-spec p.61–62), which
  reads both `subSize = Partition_Subsize[ partition ][ bSize ]`
  and `splitSize = Partition_Subsize[ PARTITION_SPLIT ][ bSize ]`
  side by side; the typed accessor's `None` return means the
  recursive descent never silently propagates a sentinel. Tests
  grow by 16 (cdf module): BLOCK_* ordinal pinning (`BLOCK_4X4 =
  0` through `BLOCK_INVALID = 22`); `PARTITION_TYPES_TOTAL` pin;
  table-shape pin; row 0 (PARTITION_NONE) identity on every
  square; row 1 (HORZ) height-halving; row 2 (VERT) width-halving;
  row 3 (SPLIT) both-dimensions halving; rows 4/5 (HORZ_A/B) row
  equality with row 1; rows 6/7 (VERT_A/B) row equality with row
  2; rows 8/9 (HORZ_4/VERT_4) quarter-splits + BLOCK_128X128 drop;
  `BLOCK_4X4` only-resolves-for-`PARTITION_NONE` invariant;
  exhaustive rectangular-`bSize`-is-invalid coverage; every
  resolved subSize in `0..BLOCK_SIZES`; out-of-range guard
  (`partition >= 10`, `b_size >= BLOCK_SIZES`); §5.11.4
  subdivision-shrinks-area invariant (halving / quartering
  partitions strictly shrink the child area); §5.11.4 subSize +
  splitSize pair well-formedness for every reachable HORZ_A /
  HORZ_B parent block. 359 -> 375 tests, zero `#[ignore]`.

* **Round 149 — §5.11.49 caller-side argument derivation.** Lands the
  `palette_tokens_args(mi_size, mi_row, mi_col, mi_rows, mi_cols,
  plane, subsampling_x, subsampling_y) -> Option<PaletteTokensArgs>`
  helper (av1-spec p.101–102) that computes the four
  [`palette_tokens_plane`] size arguments straight from the §5.11.49
  parser-scope variables. Y branch returns the §9.3-driven
  `block_{w,h}` clipped by `Min(.., (MiRows - MiRow) * MI_SIZE)` /
  `Min(.., (MiCols - MiCol) * MI_SIZE)`; UV branch then applies the
  `>> subsampling_{x,y}` shift followed by the §5.11.49 `<4`-bump
  (`block_w += 2; onscreen_w += 2` when post-shift `block_w < 4`,
  ditto height). The new [`PaletteTokensArgs`] struct holds the four
  resolved dimensions and the §5.11.46 palette-gate constant
  [`BLOCK_8X8`] (`3`) is exposed for caller-side gating. The helper
  returns `None` for any §5.11.46 palette-gate violation (`mi_size <
  BLOCK_8X8`, `block_w > 64`, `block_h > 64`, `mi_size >=
  BLOCK_SIZES`), out-of-bounds `mi_row` / `mi_col`, zero mi-grid, or
  out-of-range subsampling flag, so the helper is safe to call
  defensively from a not-yet-gated caller. Walker invariants (`1 <=
  onscreen_* <= block_*`, `block_* <= 64`) proven over every
  palette-eligible `(MiSize, sub_x, sub_y, plane)` combination via
  an exhaustive sweep test. End-to-end shape test feeds the helper's
  output straight into [`palette_tokens_plane`] against the §9.4
  default palette CDFs and confirms `InvalidPaletteWalkArgs` never
  fires, sealing the data-flow gap pinned by the r147 follow-up
  test `palette_tokens_args_from_mi_size_pins_data_flow`.
  `read_block` can now call `palette_tokens` directly once the
  parser surfaces the variables. Tests grow by 15 (cdf module):
  `BLOCK_8X8`-row pinning; Y-plane fully-on-screen / right-edge /
  bottom-edge clipping; UV 4:2:0 minimum block, large block, width-
  `<4`-bump, height-`<4`-bump; UV 4:2:2 shape; UV 4:4:4 = Y
  identity; UV right-edge clip carry-through; the exhaustive
  invariant sweep; palette-gate rejections; caller-bug rejections;
  and the walker-fed end-to-end shape. Tests: 344 -> 359, zero
  `#[ignore]`.

* **Round 148 — §9.3 block-size conversion tables.** Stages the four
  `BLOCK_SIZES`-indexed lookup tables that convert a `MiSize` into
  block dimensions (av1-spec p.400–401): [`MI_WIDTH_LOG2`],
  [`MI_HEIGHT_LOG2`], [`NUM_4X4_BLOCKS_WIDE`], [`NUM_4X4_BLOCKS_HIGH`],
  each transcribed verbatim with the spec ordering — the 16 entries
  for `BLOCK_4X4`..`BLOCK_128X128` followed by the seven `1:4` / `4:1`
  aspect-ratio entries (`BLOCK_4X16` .. `BLOCK_64X16`). Also adds the
  §3 constants [`MI_SIZE`] (`4`) and [`MI_SIZE_LOG2`] (`2`) that the
  §9.3 spec definitions reference (`Block_Width[ x ] = 4 *
  Num_4x4_Blocks_Wide[ x ]` is encoded as `NUM_4X4_BLOCKS_WIDE[ x ] <<
  MI_SIZE_LOG2` so the spec identity is not duplicated as a numeric
  table). Six new `MiSize`-keyed accessors round-out the surface:
  [`block_width`], [`block_height`], [`num_4x4_blocks_wide`],
  [`num_4x4_blocks_high`], [`mi_width_log2`], [`mi_height_log2`]; each
  is `const fn` with a `debug_assert!(mi_size < BLOCK_SIZES)` bound.
  These feed the §5.11.49 [`palette_tokens_plane`] caller (block_w /
  block_h derivation) staged in r147, and unblock the wider §5.x
  reconstruction call sites (`bw4 = Num_4x4_Blocks_Wide[ MiSize ]`)
  that the parser will surface once `read_block` is wired. Tests grow
  by 10 (cdf module): the four §9.3 tables pinned byte-for-byte at
  the expected `BLOCK_SIZES = 22` length; the §3 `MI_SIZE == 1 <<
  MI_SIZE_LOG2` identity; the `Num_4x4_Blocks_{Wide,High} == 1 <<
  Mi_{Width,Height}_Log2` identity per the §9.3 derivation; the
  `Block_{Width,Height} = 4 * Num_4x4_Blocks_{Wide,High}` identity
  for every `MiSize` with the canonical 22-entry expected vectors
  (4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4,
  16, 8, 32, 16, 64 for widths; 4, 8, 4, 8, 16, 8, 16, 32, 16, 32,
  64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16 for heights); the
  square diagonal `BLOCK_4X4` .. `BLOCK_128X128` resolving to the
  expected `n×n` luma sizes; the §5.11.46 `bsizeCtx =
  Mi_Width_Log2[ MiSize ] + Mi_Height_Log2[ MiSize ] - 2` derivation
  staying inside `0..PALETTE_BLOCK_SIZE_CONTEXTS` for every `MiSize`
  inside the §5.11.46 palette syntax gate (`MiSize >= BLOCK_8X8 &&
  Block_Width <= 64 && Block_Height <= 64`); and a §5.11.49 caller
  data-flow pin confirming `block_width(mi_size)` /
  `block_height(mi_size)` are inside `8..=64` at the palette-minimum
  (`BLOCK_8X8`) and palette-maximum (`BLOCK_64X64`) `MiSize` values
  the §5.11.46 gate admits. Tests: 334 -> 344, zero `#[ignore]`.

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
