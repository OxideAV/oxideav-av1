# Changelog

All notable changes to `oxideav-av1` are recorded here.

## [Unreleased]

- encoder r262 (2026-06-09): add §5.11.46 luma-plane palette-entries
  writer (`encoder::block_mode_info::write_palette_entries_y`,
  av1-spec p.97-98) — the inverse of
  `crate::cdf::read_palette_entries_y`. This closes one of the two
  arcs the r261 `write_palette_mode_info` leaf flagged with
  `PaletteEntriesUnsupported`: once the §5.11.22 dispatcher learns
  to thread `has_palette_y = 1`, the §5.11.46 luma palette syntax
  is now expressible on the encoder side.

  The writer mirrors the §5.11.49 cache + L(BitDepth) literal +
  L(2) `palette_num_extra_bits_y` + L(paletteBits) delta loop the
  reader inverts. For each cache slot it emits L(1)=1 iff the next
  palette entry equals the cache value, advancing the entry pointer
  on a hit. After cache exhaustion, the first new entry rides
  L(BitDepth). The L(2) extra-bits choice is derived by simulating
  the reader's per-step `paletteBits = Min(paletteBits,
  CeilLog2(range))` refinement and picking the smallest `extra ∈
  {0, 1, 2, 3}` such that every remaining `(color[idx] - color[idx
  - 1]) - 1` raw value fits in the current `paletteBits` — strictly
  ascending input in `[0, 2^BitDepth)` (the reader's post-sort +
  post-Clip1 invariant) guarantees `extra ≤ 3` always works.

  +13 round-trip tests through `read_palette_entries_y`: size 2 at
  8/10/12 bit-depth, size 3 with non-trivial ascending deltas,
  full PALETTE_COLORS=8 max-size palette exercising the paletteBits
  refinement, full §5.11.49 cache-hit round-trip (both entries
  adopted from a hand-stamped left-neighbour palette via the
  new `#[cfg(test)] PartitionWalker::stamp_palette_for_test`
  helper), partial-cache hit (one cache slot adopted + one new
  literal entry); caller-bug rejections for descending pair,
  duplicate pair, entry above 2^BitDepth, size below 2, size
  above PALETTE_COLORS, and invalid `bit_depth`.

  Followups for r263+:
  * §5.11.46 UV-plane palette-entries writer
    (`write_palette_entries_uv`) — U-plane (no `delta++`, `range`
    without `- 1`) + V-plane two-arm
    (`delta_encode_palette_colors_v == 1` signed-delta with
    L(1) sign bit + modular wrap; `== 0` direct-literal arm; no
    trailing sort).
  * Lift the §5.11.22 dispatcher's `has_palette_y == 0 /
    has_palette_uv == 0` precondition once the UV writer lands and
    the dispatcher can thread the `palette_size_*_minus_2` S() +
    `write_palette_entries_*` invocations.

  Helpers exposed `pub(crate)`: `crate::cdf::clip1_to_bit_depth`
  and `crate::cdf::ceil_log2_av1` — both previously private to
  `cdf.rs` (used by `read_palette_entries_y` and friends), now
  reachable from the encoder module's matching writer logic.

- encoder r261 (2026-06-08): add §5.11.46 no-palette leaf writer
  (`encoder::block_mode_info::write_palette_mode_info`, av1-spec
  p.97-98) and the §5.11.22 `intra_block_mode_info` dispatcher
  (`encoder::block_mode_info::write_intra_block_mode_info`, av1-spec
  p.72) that composes the four §5.11.22 leaves the r211-r260 arc
  produced (`write_y_mode` / `write_intra_uv_mode` /
  `write_intra_angle_info_{y,uv}` / `write_filter_intra_mode_info`)
  with the new §5.11.46 leaf.

  The §5.11.46 leaf covers the no-palette path only — commits
  `has_palette_y == 0` AND `has_palette_uv == 0`, mirroring the
  §5.11.46 reader's outer-gate (`allow_screen_content_tools && MiSize
  >= BLOCK_8X8 && Block_Width <= 64 && Block_Height <= 64`) +
  inner-arm short-circuits. The §8.3.2 CDF selections are
  `TilePaletteYModeCdf[ bsizeCtx ][ palette_y_mode_ctx(above, left) ]`
  for the luma arm and `TilePaletteUVModeCdf[ palette_uv_mode_ctx(0) ]`
  for the chroma arm (with `bsizeCtx = Mi_Width_Log2[ MiSize ] +
  Mi_Height_Log2[ MiSize ] - 2`). Non-zero `has_palette_*` surfaces
  `Error::PaletteEntriesUnsupported` so the dispatcher knows to
  re-route once the §5.11.46 palette-entries arc (cache merge +
  `L(BitDepth)` literal + `paletteBits` delta loop + trailing sort)
  lands in a subsequent round.

  The §5.11.22 dispatcher covers the no-palette + no-CFL +
  `0..=12 uv_mode` path of the spec body (av1-spec p.72 lines 2,
  4, 6, 10, 13-15, 16). It derives the §8.3.2 `size_group(MiSize)`
  ctx for the §5.11.22 line 2 `y_mode` S() in-place + threads the
  `has_chroma`-gated chroma arm into `write_intra_uv_mode` +
  `write_intra_angle_info_uv`. The §5.11.22 line 8 `read_cfl_alphas`
  call is intentionally excluded from this round's arc; callers
  that need the CFL arm compose `write_cfl_alphas` manually until a
  future round folds it into the dispatcher.

  +13 round-trip tests covering the new leaf + dispatcher: gate-off
  (allow_screen_content_tools = false ⇒ no §5.11.46 S() symbol
  emitted vs a baseline `SymbolWriter`), gate-off via `MiSize <
  BLOCK_8X8`, non-DC `YMode` skips the luma arm + still emits the
  chroma arm, has_palette_y/uv == 1 rejection with
  `PaletteEntriesUnsupported`, all out-of-range caller-bug rejections;
  full §5.11.22 dispatcher round-trips through
  `PartitionWalker::decode_intra_block_mode_info` covering DC_PRED,
  directional `YMode = V_PRED` with `angle_delta_y = 2`,
  palette-gate-open / no-palette path, filter-intra arm with
  `use_filter_intra = 1` + `filter_intra_mode = Some(3)`, monochrome
  (`has_chroma = false`); caller-bug rejections for missing
  `uv_mode` when has_chroma, `uv_mode = UV_CFL_PRED = 13`, and
  `Some(uv_mode)` when monochrome.

- encoder r260 (2026-06-08): add three §5.11.22 leaf writers
  toward the §5.11.18 line 23 `if ( !is_inter )` terminal arm
  (`intra_block_mode_info`):
  `encoder::block_mode_info::write_intra_angle_info_y` (§5.11.42
  inverse, av1-spec p.95),
  `encoder::block_mode_info::write_intra_angle_info_uv` (§5.11.43
  inverse, p.95), and
  `encoder::block_mode_info::write_filter_intra_mode_info` (§5.11.24
  inverse, p.74).

  These are the §5.11.22 `intra_block_mode_info` syntax elements
  inlined into [`PartitionWalker::decode_intra_block_mode_info`] at
  the corresponding §5.11.22 call sites (line 4 luma angle delta,
  line 10 chroma angle delta, line 16 filter-intra mode info).
  Each writer is stateless on grid: callers feed the §5.5.2 /
  §5.11.46 frame-header / per-block scalars (`mi_size`, `y_mode` /
  `uv_mode`, `palette_size_y`, `enable_filter_intra`) plus the
  committed §5.11.42 / §5.11.43 / §5.11.24 syntax-element values
  (`angle_delta_y` / `angle_delta_uv` in
  `-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA = -3..=3`; `use_filter_intra`
  in `0..=1` with `filter_intra_mode: Option<u8>` in
  `0..INTRA_FILTER_MODES = 5` on the `use == 1` arm).

  Each writer implements the §5.11.42 / §5.11.43 / §5.11.24
  short-circuits (no §8.2.6 S() symbols emitted on the
  short-circuit arms — verified vs a pristine `SymbolWriter` in
  the unit tests) and the §8.3.2 CDF selections
  (`TileAngleDeltaCdf[ YMode - V_PRED ]` /
  `TileAngleDeltaCdf[ UVMode - V_PRED ]` for the angle deltas;
  `TileFilterIntraCdf[ MiSize ]` + `TileFilterIntraModeCdf` for
  the §5.11.24 inner arm).

  +20 round-trip tests covering the three new writers: (1)
  §5.11.42 round-trip across all seven `AngleDeltaY` values on
  `V_PRED`; (2) §5.11.43 round-trip across all seven `AngleDeltaUV`
  values on `V_PRED`; (3) combined §5.11.42 + §5.11.43 round-trip
  with independent non-zero deltas; (4) §5.11.24 round-trip with
  `use_filter_intra = 0` and `use_filter_intra = 1` across all
  five `INTRA_FILTER_MODES = 5` mode values; plus six
  short-circuit tests (non-directional `YMode`/`UVMode`, `MiSize
  < BLOCK_8X8`, `UV_CFL_PRED`, `!enable_filter_intra`, `y_mode !=
  DC_PRED`, `palette_size_y != 0`, `Max(BW, BH) > 32`) and six
  caller-bug guards (non-zero delta on short-circuit,
  out-of-range delta, non-zero `use_filter_intra` on gate-off,
  `use_filter_intra == 1` without mode, `use_filter_intra == 0`
  with mode, out-of-range mode).

  Library test count moves to 1725 (was 1705).

  Followups for r261+:
  * Compose the three r260 leaf writers into a §5.11.22
    `intra_block_mode_info` dispatcher writer (the §5.11.18 line
    23 `if ( !is_inter )` terminal arm), threading the §5.11.22
    line 5-12 `if ( HasChroma )` arm + the §5.11.45
    `write_cfl_alphas` + a future §5.11.46
    `write_palette_mode_info` leaf writer.
  * Begin §5.11.23 `inter_block_mode_info` writer dispatch
    (composes §5.11.25 `read_ref_frames` + §5.11.26 `assign_mv` +
    §5.11.27 `read_motion_mode` + §5.11.28 `read_interintra_mode`
    + §5.11.29 `read_compound_type` writers, none of which exist
    yet on the encoder side).
  * Then wire the §5.11.18 line 23-26 `if ( is_inter )` terminal
    dispatch (§5.11.22 vs §5.11.23) into the r257
    `write_inter_frame_mode_info_prefix` dispatcher, closing the
    §5.11.18 inter_frame_mode_info() body end-to-end.

- encoder r259 (2026-06-08): compose the r258 §5.11.18 lines 18-20
  leaf writers (`write_cdef` / `write_delta_qindex` / `write_delta_lf`)
  into `encoder::block_mode_info::write_inter_frame_mode_info_prefix`,
  completing the §5.11.18 prefix dispatcher up to the
  `read_is_inter()` call.

  Lines 18-20 inputs are bundled into a new
  `InterFrameDeltaSiteInputs` struct (with a `Default` impl yielding
  the all-short-circuit configuration — `enable_cdef = false`,
  `read_deltas = false`, zero deltas — so callers that target the
  §5.11.18 no-bits-emitted arms only need `&Default::default()`).
  The dispatcher's eight-line prologue is now:

  * §5.11.19 `inter_segment_id( 1 )` — `write_inter_segment_id`.
  * §5.11.10 `read_skip_mode( )` — `write_skip_mode`.
  * §5.11.11 `read_skip( )` (when `skip_mode == 0`) — `write_skip`.
  * §5.11.19 `inter_segment_id( 0 )` (when `!SegIdPreSkip`) —
    `write_inter_segment_id`.
  * §5.11.18 line 17 `Lossless = LosslessArray[ segment_id ]` lookup.
  * §5.11.56 `read_cdef( )` — `write_cdef`.
  * §5.11.12 `read_delta_qindex( )` — `write_delta_qindex`.
  * §5.11.13 `read_delta_lf( )` — `write_delta_lf`.
  * §5.11.18 line 21 `ReadDeltas = 0` (caller-owned state).
  * §5.11.20 `read_is_inter( )` — `write_is_inter`.

  `InterFrameModeInfoPrefix` grows three derived fields surfacing
  the lines 18-20 results: `cdef_idx: Option<i8>` (`None` on the
  §5.11.56 short-circuit or the anchor-already-stamped arm,
  `Some(value)` on the first-leaf-in-anchor write),
  `reduced_delta_q_index: i32` (the caller-passed §5.11.12 signed
  intermediate, surfaced for chaining into the encoder-owned
  `CurrentQIndex` accumulator), and `reduced_delta_lf: [i32;
  FRAME_LF_COUNT]` (likewise for the §5.11.13 row).

  +4 round-trip tests covering the new path: (1) all-short-circuit
  arm with default `InterFrameDeltaSiteInputs` (no bits on any of
  the three lines, `cdef_idx == None`); (2) `write_cdef`
  first-leaf-in-anchor arm round-trip through the dispatcher's full
  byte run against `decode_cdef`, including all eight composed
  decoder reads in order; (3) §5.11.12 literal-branch round-trip
  with `read_deltas = true` and non-zero `reducedDeltaQIndex`,
  verifying the §5.11.12 `Clip3(1, 255, ...)` accumulator update;
  (4) §5.11.13 single-LF round-trip with non-zero `DeltaLF[0]`
  delta, verifying the `Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER,
  ...)` accumulator update.

  All 8 existing dispatcher tests + 4 leaf-writer round-trip
  batteries pass unchanged — the new parameter (`delta_inputs:
  &InterFrameDeltaSiteInputs`) appears as the trailing positional
  arg on every callsite, and the existing tests pass
  `&InterFrameDeltaSiteInputs::default()` so the §5.11.18 lines
  18-20 short-circuit on every prior assertion.

  Library test count moves to 1705 (was 1701).

  Followups for r260+:
  * §5.11.18 line 23 `if ( is_inter )` terminal arms — §5.11.22
    `intra_block_mode_info` writer (the `else` arm) and §5.11.23
    `inter_block_mode_info` writer (the `if` arm), the latter
    composing §5.11.25 `read_ref_frames` + §5.11.26 `assign_mv` +
    §5.11.27 `read_motion_mode` + §5.11.28 `read_interintra_mode` +
    §5.11.29 `read_compound_type` writers.
  * Wire the dispatcher's new `cdef_idx_committed` /
    `reduced_delta_q_index` / `reduced_delta_lf` surfaced fields
    into a downstream `WriteInterBlockMvState` aggregate once the
    §5.11.23 inter-block dispatcher writer lands; the encoder-owned
    `CurrentQIndex` / `DeltaLF[..]` / `cdef_idx[..]` accumulators
    naturally fold in there.

- encoder r258 (2026-06-08): add three §5.11.18 lines 18-20 leaf
  writers — `encoder::block_mode_info::write_cdef` (§5.11.56
  inverse, av1-spec p.104), `encoder::block_mode_info::write_delta_qindex`
  (§5.11.12 inverse, p.68), and `encoder::block_mode_info::write_delta_lf`
  (§5.11.13 inverse, p.68). These are the three syntax elements
  the r257 `write_inter_frame_mode_info_prefix` dispatcher leaves
  out (between the post-skip segment_id arm and the `read_is_inter`
  call); next round will compose them into the dispatcher's body.

  Each writer mirrors its decoder counterpart on
  [`crate::cdf::PartitionWalker`] exactly: same arm structure,
  same short-circuit set, same escape-ladder reach. `write_cdef`
  is the cleanest pure-literal writer (`L(cdef_bits)` on the
  first-leaf-in-anchor arm, no bits otherwise); `write_delta_qindex`
  and `write_delta_lf` share the §5.11.12 / §5.11.13 `S() +
  L(3) + L(n) + L(1)` escape-ladder shape (the analytic inverse
  of the spec's `delta_q_abs = abs_bits + (1 << n) + 1` is
  `n = FloorLog2(abs_value - 1)` for `abs_value >= 3`).

  Stateless on grid state: `write_cdef` does not maintain an
  encoder-side `cdef_idx[]` grid (caller threads its own — or
  uses a parallel `PartitionWalker`, the round-trip-test pattern);
  `write_delta_qindex` / `write_delta_lf` consume the *signed*
  `reducedDeltaQIndex` / `reducedDeltaLfLevel` intermediate so
  the running `CurrentQIndex` / `DeltaLF[ i ]` accumulators stay
  caller-owned (decoder-symmetric — the decoder returns the
  post-`Clip3` value the caller already had to fold in).

  +17 tests covering every arm: short-circuit set (`skip /
  coded_lossless / !enable_cdef / allow_intrabc` for cdef;
  `MiSize == sbSize && skip` + `!ReadDeltas` + `!delta_lf_present`
  for the deltas), first-leaf vs anchor-already-stamped for cdef
  (with prior-stamp verification), literal vs escape branches
  (including the boundary value 3, mid-range 17, and the
  near-max ±511 with `Clip3` clamping), single-LF vs multi-LF
  vs mono_chrome two-lane variants for `write_delta_lf`, plus
  range-guard batteries on every writer (`cdef_bits > 3`,
  `cdef_idx >= (1 << cdef_bits)`, `delta_q_res > 3`, `|reduced|
  > 511`, non-zero on short-circuit arm, non-zero past
  frame_lf_count). Each round-trip test replays the writer's byte
  run through the matching decoder call and asserts every decoded
  scalar (or row) matches the writer's input.

  Library test count moves to 1701 (was 1684).

  Followups for r259+:
  * Bring r257's `write_inter_frame_mode_info_prefix` to a full
    §5.11.18 dispatcher by composing the three writers landed here
    into its body between the post-skip segment_id arm and the
    `read_is_inter` call.
  * §5.11.18 line 23 `if ( is_inter )` terminal arms — §5.11.22
    `intra_block_mode_info` writer (the `else` arm) and §5.11.23
    `inter_block_mode_info` writer (the `if` arm), the latter
    composing §5.11.25 `read_ref_frames` + §5.11.26 `assign_mv` +
    §5.11.27 `read_motion_mode` + §5.11.28 `read_interintra_mode`
    + §5.11.29 `read_compound_type` writers.

- encoder r257 (2026-06-08): add
  `encoder::block_mode_info::write_inter_frame_mode_info_prefix` —
  the §5.11.18 `inter_frame_mode_info()` dispatcher (av1-spec p.71)
  that composes the five sub-writers landed across r253-r256
  (§5.11.19 `inter_segment_id` pre-skip + §5.11.10 `read_skip_mode`
  + §5.11.11 `read_skip` + §5.11.19 `inter_segment_id` post-skip +
  §5.11.20 `read_is_inter`) in the spec body order.

  Scope is the §5.11.18 *prefix* — every value the writer commits
  before the §5.11.18 line 18 `read_cdef( )` call. The line 18-21
  `read_cdef` / `read_delta_qindex` / `read_delta_lf` writers and
  the line 23-26 `if ( is_inter )` terminal dispatch into §5.11.22
  / §5.11.23 are next-round encoder-side targets. The writer
  returns a `InterFrameModeInfoPrefix` aggregate carrying
  `segment_id`, `skip_mode`, `skip`, `is_inter`, and the §5.11.18
  line 17 `Lossless = LosslessArray[ segment_id ]` lookup (no bits)
  so the caller can chain.

  Stateless on purpose (mirrors the rest of
  `encoder::block_mode_info`) — every §8.3.2 context is
  caller-supplied and grid-fill stays decoder-side. The §5.11.18
  pre-skip vs post-skip dispatch handles the segment_id mapping
  internally: when `seg_id_pre_skip == false` the pre-skip call's
  segment_id is forced through the §5.11.19 Arm 1 / Arm 2 / Arm 3
  short-circuit (`0` / `predicted_segment_id` / `0` per arm) so the
  caller need only pass the *final* segment_id once.

  Round-trip coverage: 8 tests covering (a) segmentation disabled
  every (skip × is_inter) combination, (b) `skip_mode == 1`
  compound-reference shortcut, (c) `!seg_id_pre_skip` post-skip
  arm via temporal-update adopt, (d) SEG_LVL_REF_FRAME +
  SEG_LVL_SKIP combined override, plus four range-guard rejection
  cases (`last_active_seg_id >= MAX_SEGMENTS`, `skip > 1`,
  `skip_mode == 1 && skip != 1`, and the `LosslessArray[]` lookup
  surfacing test). Each round-trip test replays the writer's byte
  run through the matching composed `decode_*` calls in the same
  order the §5.11.18 spec body specifies, and asserts every
  decoded scalar matches the writer's input.

  Followups for r258+:
  * §5.11.18 line 18 `read_cdef` writer — composes the §5.11.56
    `cdef_idx` `L(cdef_bits)` bit-fixed read with the §5.11.56
    `skip || coded_lossless || !enable_cdef || allow_intrabc`
    short-circuit set.
  * §5.11.18 lines 19-20 `read_delta_qindex` / `read_delta_lf`
    writers.
  * §5.11.18 line 23 `if ( is_inter )` terminal arms — §5.11.22
    `intra_block_mode_info` writer (the `else` arm) and §5.11.23
    `inter_block_mode_info` writer (the `if` arm), the latter
    composing §5.11.25 `read_ref_frames` + §5.11.26 `assign_mv` +
    §5.11.27 `read_motion_mode` + §5.11.28 `read_interintra_mode`
    + §5.11.29 `read_compound_type` writers.

- encoder r256 (2026-06-08): add
  `encoder::block_mode_info::write_inter_segment_id` — the §5.11.19
  `inter_segment_id( preSkip )` syntax-element writer (av1-spec
  p.71). The encoder mirror of
  [`crate::cdf::PartitionWalker::decode_inter_segment_id`]; lands as
  the natural caller of r255's `write_segment_id` (§5.11.9) and
  composes it on the three arms that descend into
  `read_segment_id()`.

  Transcribes the §5.11.19 six-arm dispatch verbatim:

  1. `!segmentation_enabled` ⇒ `segment_id = 0`, no bits.
  2. `!segmentation_update_map` ⇒ `segment_id =
     predictedSegmentId`, no bits.
  3. `preSkip && !SegIdPreSkip` ⇒ `segment_id = 0`, no bits (the
     pre-skip caller defers to the post-skip call).
  4. `!preSkip && skip` (post-skip arm's skip-block branch) ⇒
     descends to `write_segment_id` with `skip = 1`, which §5.11.9
     short-circuits to `segment_id = pred` with no `S()`. The
     forced `seg_id_predicted = 0` grid stamp is the caller's job
     (the writer is stateless — no encoder-side
     `AboveSegPredContext` / `LeftSegPredContext` arrays).
  5. `segmentation_temporal_update == 1` ⇒ emits one §8.2.6 `S()`
     `seg_id_predicted` symbol against
     `TileSegmentIdPredictedCdf[ ctx ]` where `ctx` is the §8.3.2
     `seg_pred_ctx` (`LeftSegPredContext[ MiRow ] +
     AboveSegPredContext[ MiCol ]`, caller-derived in
     `0..SEGMENT_ID_PREDICTED_CONTEXTS = 3`). Then either adopt
     `predictedSegmentId` (no further bits, `segment_id` MUST equal
     `predicted_segment_id`) or fall through to `write_segment_id`
     with `skip = 0` for the literal id read. The §6.10.9 spec-note
     permits `seg_id_predicted == 0` even when
     `segment_id == predictedSegmentId`; the caller chooses the
     flag.
  6. Fall-through (`segmentation_temporal_update == 0`) ⇒ delegate
     to `write_segment_id` with `skip = 0` for the literal id
     read.

  Stateless on purpose (mirrors the rest of
  `encoder::block_mode_info`) — every §8.3.2 context (`seg_pred_ctx`
  for arm 5, `seg_id_read_ctx` for the inner `write_segment_id`
  `S()` arm) is passed in, and the caller threads its own
  `PartitionWalker` (or equivalent neighbour-array state) for grid
  stamps.

  Range / mismatch guards (all surface as
  `Error::PartitionWalkOutOfRange`):
  `last_active_seg_id >= MAX_SEGMENTS = 8`,
  `segment_id > last_active_seg_id`, `pred > last_active_seg_id`,
  `predicted_segment_id > last_active_seg_id`,
  `seg_pred_ctx >= SEGMENT_ID_PREDICTED_CONTEXTS = 3` on arm 5,
  `seg_id_predicted > 1` on arm 5, and per-arm preconditions
  (`segment_id != 0` on arms 1 and 3,
  `segment_id != predicted_segment_id` on arm 2 and the adopt
  branch of arm 5, `segment_id != pred` on arm 4 via the inner
  `write_segment_id`).

  +19 unit tests: six arm round-trips against
  `PartitionWalker::decode_inter_segment_id` (disabled / no-update-
  map / pre-skip-early-exit no-bits arms, the arm 4 `skip` branch,
  the arm 5 adopt branch exhaustive over `predicted_segment_id
  ∈ 0..8`, the arm 5 literal branch exhaustive over `segment_id
  ∈ 0..8`, the arm 6 fall-through exhaustive over `segment_id ∈
  0..8`), eleven range / arm-mismatch rejections (one per arm plus
  the up-front guards), and one §8.3.2 admissibility sweep across
  every `seg_pred_ctx ∈ 0..SEGMENT_ID_PREDICTED_CONTEXTS = 3`.
  Full library test count moves to 1676 (was 1657).

- encoder r255 (2026-06-08): add
  `encoder::block_mode_info::write_segment_id` — the §5.11.9
  `read_segment_id()` syntax-element writer (av1-spec p.66). This
  is the shared per-block segment-id primitive both the §5.11.8
  `intra_segment_id` and the (pending) §5.11.19 `inter_segment_id`
  writers compose; r254's `write_intra_segment_id` is refactored
  to delegate to it instead of inlining the inverse search.

  Companion change: add `cdf::neg_interleave` — the analytic
  inverse of `cdf::neg_deinterleave` (the §5.11.9 helper). The
  inverse mirrors `neg_deinterleave` arm-by-arm: identity for
  `pred == 0`; `max - segment_id - 1` for `pred >= max - 1`; the
  two bidirectional-fan formulas for the upward (`2 * pred < max`)
  and downward (`2 * pred >= max`) arms, each with an in-fan vs.
  out-of-fan branch. The pair `(neg_interleave,
  neg_deinterleave)` is a bijection on `0..max` for every fixed
  `pred ∈ 0..max`; exhaustive round-trip tests pin both
  directions over the full `max ∈ 1..=8` × `pred ∈ 0..max` × `s
  ∈ 0..max` cube. Replaces an O(8) search inside
  `write_intra_segment_id` with an O(1) inverse.

  Writer two-arm dispatch transcribes §5.11.9 directly:
  `skip != 0` ⇒ `segment_id = pred`, no symbol emitted (caller
  must pass `segment_id == pred`); `skip == 0` ⇒ one §8.2.6 `S()`
  symbol against `TileSegmentIdCdf[ ctx ]`, with `diff =
  neg_interleave(segment_id, pred, last_active_seg_id + 1)`.
  `ctx` is the §8.3.2 derivation via the existing
  `cdf::segment_id_ctx` helper.

  Stateless mirror of the rest of `encoder::block_mode_info` — no
  encoder-side `SegmentIds[]` grid; the caller threads its own
  walk and supplies the §5.11.9 `pred` value pre-computed via the
  neighbour cascade. Range guards
  (`last_active_seg_id >= MAX_SEGMENTS = 8`, `segment_id >= max`,
  `pred >= max`, `ctx >= SEGMENT_ID_CONTEXTS = 3` on the `S()`
  arm) surface as `Error::PartitionWalkOutOfRange`.

  +13 unit tests: 9 against `write_segment_id` (skip arm, origin
  else arm, upward + downward fan exhaustive round-trips via
  writer-decoder pairs, five range-guard rejections), 4 against
  `neg_interleave` (forward + reverse exhaustive bijection
  checks, upward + downward fan worked-example matches). Full
  library test count moves to 1657 (was 1644).

- encoder r254 (2026-06-08): add
  `encoder::block_mode_info::write_skip_mode` — the §5.11.10
  `read_skip_mode()` syntax-element writer (av1-spec p.67). Lands
  beside r253's `write_is_inter` (whose Arm 1 is gated by
  `skip_mode`) and mirrors the existing decoder reader
  [`crate::cdf::PartitionWalker::decode_skip_mode`].

  The writer transcribes the §5.11.10 two-arm dispatch verbatim:

  1. Short-circuit set (any-true ⇒ `skip_mode = 0`, no bits): the
     three `seg_feature_active( SEG_LVL_SKIP / SEG_LVL_REF_FRAME /
     SEG_LVL_GLOBALMV )` predicates (caller collapses into
     `seg_skip_mode_off`), the §5.9.21 `skip_mode_present`
     frame-header scalar (collapses to `!skip_mode_present`), and
     the small-block check `Block_Width[ MiSize ] < 8 ||
     Block_Height[ MiSize ] < 8` (derived locally from `sub_size`
     via the §9.3 [`crate::cdf::block_width`] /
     [`crate::cdf::block_height`] tables).
  2. Fall-through `else`: `skip_mode` is emitted as one §8.2.6
     `S()` symbol against `TileSkipModeCdf[ ctx ]`. `ctx` is the
     §8.3.2 `skip_mode` context the caller pre-computes via
     [`crate::cdf::skip_mode_ctx`] from the §5.11.5 `SkipModes[]`
     neighbour-sum.

  Stateless on purpose (mirrors `write_skip` / `write_is_inter`) —
  does not maintain an encoder-side `SkipModes[]` grid; the caller
  threads whatever grid state the encode-side §8.3.2 ctx derivations
  need. Arm-precondition violations (caller-supplied `skip_mode != 0`
  on any short-circuit arm, `ctx >= SKIP_MODE_CONTEXTS = 3` on the
  fall-through arm, `skip_mode > 1` everywhere,
  `sub_size >= BLOCK_SIZES = 22`) surface as
  `Error::PartitionWalkOutOfRange`.

  +15 unit tests: one per short-circuit arm (seg-active / present-
  false / small-width / small-height / both-small-4x4) each asserting
  the decoder bit position is unchanged across the writer → decoder
  round-trip and the decoder recovers `skip_mode = 0`; two
  fall-through round-trips at the frame origin for `skip_mode ∈ {0,
  1}` with §8.3 CDF adaptation engaged; §8.3.2 ctx admissibility
  across every `(above_sm, left_sm)` combo (4 combos × 2 = 8
  writes); six out-of-range / arm-mismatch rejections; one
  arm-ordering invariant pinning that any short-circuit arm fires
  before the ctx admissibility check. Full lib test count 1629 →
  1644.

- encoder r253 (2026-06-08): add
  `encoder::block_mode_info::write_is_inter` — the §5.11.20
  `read_is_inter()` syntax-element writer (av1-spec p.71). Lands at
  the same dispatcher level as r211's
  `write_skip` / `write_intra_segment_id` / `write_intra_frame_y_mode`
  / `write_y_mode` / `write_intra_uv_mode` / `write_cfl_alphas` — the
  encoder mirror of
  [`crate::cdf::PartitionWalker::decode_is_inter`].

  The writer transcribes the §5.11.20 four-arm dispatch verbatim:

  1. `if ( skip_mode )` ⇒ `is_inter = 1`, no bits written.
  2. `else if ( seg_feature_active( SEG_LVL_REF_FRAME ) )` ⇒
     `is_inter = FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] != INTRA_FRAME`
     (the caller pre-computes the `!= INTRA_FRAME` comparison as
     `seg_ref_frame_is_inter`), no bits written.
  3. `else if ( seg_feature_active( SEG_LVL_GLOBALMV ) )` ⇒
     `is_inter = 1`, no bits written.
  4. Fall-through `else`: `is_inter` is emitted as one §8.2.6 `S()`
     symbol against `TileIsInterCdf[ ctx ]`. `ctx` is the §8.3.2
     `is_inter` context the caller pre-computes via the existing
     public [`crate::cdf::is_inter_ctx`] helper from the neighbour
     `LeftIntra` / `AboveIntra` predicates (or `None` for an
     unavailable neighbour per §5.11.18).

  The writer is stateless on purpose — it does not maintain an
  encoder-side `IsInters[]` grid; the caller is responsible for
  threading whatever grid state the §8.3.2 ctx derivation needs
  (mirroring the §5.11.7 / §5.11.11 / §5.11.22 intra-arm writer
  pattern landed in r211). Out-of-range and arm-precondition
  violations (caller-supplied `is_inter` disagreeing with the
  spec-derived value on Arms 1 / 2 / 3, `ctx >= IS_INTER_CONTEXTS = 4`
  on Arm 4, `is_inter > 1` everywhere) are surfaced as
  [`Error::PartitionWalkOutOfRange`].

  +14 unit tests covering every arm — short-circuit on Arms 1 / 2 /
  3 (no bits, decoder position unchanged), fall-through `S()`
  round-trip at the frame origin for both `is_inter ∈ {0, 1}` with
  §8.3 CDF adaptation engaged on both sides, §8.3.2 ctx
  admissibility across every `(above_intra, left_intra)`
  combination [`is_inter_ctx`] distinguishes (9 combos × 2 values),
  out-of-range guards on `is_inter > 1` and `ctx >= 4`, the three
  arm-mismatch rejections, and the arm-ordering invariant that Arm 1
  short-circuits before Arms 2 / 3 / 4 are consulted.

  Brings encoder mode_info coverage to the same six §5.11 elements
  the decoder reaches at the §5.11.18 `inter_frame_mode_info`
  pre-dispatch step (`skip` + `intra_segment_id` +
  `intra_frame_y_mode` + `y_mode` + `uv_mode` + `cfl_alphas` +
  `is_inter`) and is the first inter-arm scalar the encoder side
  can now emit. The follow-on §5.11.23 `inter_block_mode_info`
  syntax elements (ref-frame pair, inter-mode, MV) remain
  out-of-scope.

- encoder r252 (2026-06-08): close the rectangular TX_SIZE family
  on the encoder's 2D forward-transform dispatcher
  (`encoder::forward_transform_2d::forward_transform_2d`) with the
  mid pair of the `|log2W - log2H| == 2` family — `TX_8X32` and
  `TX_32X8`, the rectangular pair at `min(log2W, log2H) == 3` /
  `max(log2W, log2H) == 5`. Together with r235 / r238 / r241 / r244
  (the eight `|log2W - log2H| == 1` pairs) and r250 / r251 (the
  endpoints of the `|log2W - log2H| == 2` family), all twelve
  §5.11 `TX_SIZES_ALL` rectangular shapes are now dispatcher-
  supported.

  Per §5.11.40 `compute_tx_type()` the gate `txSzSqrUp > TX_32X32`
  is strict, so this pair (`Tx_Size_Sqr_Up == TX_32X32`) escapes
  the `return DCT_DCT` short-circuit. The §5.11.48 `get_tx_set()`
  routing on `txSzSqrUp == TX_32X32` lands intra-coded blocks on
  `TX_SET_DCTONLY` (only `DCT_DCT`) and inter-coded blocks on
  `TX_SET_INTER_3` whose §6.10.19 inversion table
  `Tx_Type_Inter_Inv_Set3 = { IDTX, DCT_DCT }` admits exactly two
  ordinals — so the reachable per-shape tx_type set is
  `{DCT_DCT, IDTX}`. Per-axis kernel reachability matches: both
  axes (`n=3` for length-8 and `n=5` for length-32) are inside
  `forward_dct_dispatch n in 2..=6` and
  `forward_idtx_dispatch n in 2..=5`, so both reachable ordinals
  walk a real per-axis kernel pair.

  Per §7.13.3 the rectangular `× 2896` row-pass step gates
  EXACTLY on `Abs(log2W - log2H) == 1`, so this pair skips it.
  Per §7.12.3 `dqDenom` falls to the default `1` arm — neither
  `TX_8X32` nor `TX_32X8` is in the `dqDenom = 2` set
  `{TX_32X32, TX_16X32, TX_32X16, TX_16X64, TX_64X16}` nor in the
  `dqDenom = 4` set `{TX_64X64, TX_32X64, TX_64X32}` per av1-spec
  p.294; the existing `super::forward_quantize` routing through
  `crate::cdf::dequant_denom` already covers this.

  Analytic per-cell round-trip scale (no rectangular factor):
  `(N_w * N_h) / 2^(row_shift + col_shift)` = `256 / 2^(2 + 4)` =
  `4`. Empirical per-cell round-trip scale is `1` — the `4×`
  constant-DC input-mass factor documented across the rectangular
  family (analytic `4` → empirical `1`).

  Test count: +8 forward_transform_2d tests (86 → 94). The pre-r252
  panic guard `rectangular_tx_size_out_of_arc_panics` is retired
  (no rectangular shape is out of arc as of r252); the replacement
  `tx_size_above_tx_sizes_all_panics` pins the remaining first-line
  dispatcher reject — a `tx_size` at or beyond `TX_SIZES_ALL` (=
  19).

- encoder r251 (2026-06-07): extend the rectangular TX_SIZE family
  arc on the encoder's 2D forward-transform dispatcher
  (`encoder::forward_transform_2d::forward_transform_2d`) by the
  short-side-4 endpoint of the `|log2W - log2H| == 2` family —
  `TX_4X16` and `TX_16X4`, the rectangular pair at
  `min(log2W, log2H) == 2` / `max(log2W, log2H) == 4`. Together
  with the r250 long-side-64 endpoint (`TX_16X64` / `TX_64X16`)
  this lands both endpoints of the `|log2W - log2H| == 2` family;
  only the mid-family `TX_8X32` / `TX_32X8` pair (at `min_log2 ==
  3`) remains.

  Unlike the r250 pair, `TX_4X16` / `TX_16X4` sit at
  `Tx_Size_Sqr_Up == TX_16X16` ≤ `TX_32X32`, so per §5.11.40
  `compute_tx_type()` the `txSzSqrUp > TX_32X32 ⇒ return DCT_DCT`
  short-circuit does NOT fire and the full §5.11.48 / §6.10.19
  `tx_type` matrix is reachable (DCT_DCT, ADST_DCT, DCT_ADST,
  ADST_ADST, FLIPADST_DCT, DCT_FLIPADST, FLIPADST_FLIPADST,
  ADST_FLIPADST, FLIPADST_ADST, IDTX, V_DCT, H_DCT, V_ADST,
  H_ADST, V_FLIPADST, H_FLIPADST — 16 ordinals total). Per-axis
  kernel reachability is complete: both axes (`n=2` and `n=4`)
  fall inside every per-axis dispatcher's range
  (`forward_dct_dispatch n in 2..=6`, `forward_adst_dispatch n
  in 2..=4`, `forward_idtx_dispatch n in 2..=5`).

  Per §7.13.3 the `Round2(T[j] * 2896, 12)` rectangular row-pass
  step gates EXACTLY on `Abs(log2W - log2H) == 1`, so this pair —
  like the r250 pair — skips the rectangular post-scale. Per
  §7.12.3 `dqDenom` falls to the default `1` arm (neither
  `TX_4X16` nor `TX_16X4` is in the `dqDenom = 2` set or the
  `dqDenom = 4` set per av1-spec p.294); the existing
  `super::forward_quantize` routing through
  `crate::cdf::dequant_denom` already covers this.

  Analytic per-cell round-trip scale (no rectangular factor):
  `(N_w * N_h) / 2^(row_shift + col_shift)` = `64 / 2^(1 + 4)` =
  `2`. Empirical per-cell round-trip scale is `1/2` — the `4×`
  constant-DC input-mass factor documented across the rectangular
  family (analytic `2` → empirical `1/2`). A constant input of
  `4` recovers `2` per cell within ±1 LSB.

  +11 forward_transform_2d tests (75 → 86) plus 20 entries added
  to the `zero_input_across_all_supported_square_combinations`
  matrix walk (covering the full 16-ordinal §6.10.19 reachable
  matrix on each of the two new shapes). Tests exercise the
  DCT × DCT arm, ADST × ADST arm, IDTX × IDTX arm, FLIPADST ×
  FLIPADST flip composition (via `check_roundtrip_flip`), mixed
  DCT × IDTX (`V_DCT`) and ADST × IDTX (`H_ADST`) arms,
  constant-DC dominance + recovery, and zero-input ⇒ zero-output
  invariants. Tests use the full ±128 residual envelope — the
  inverse-side 16-bit between-stage `Clip3` has plenty of
  headroom on these small shapes, unlike the length-64-axis r250
  pair's ±2 envelope. The out-of-arc panic guard is re-pinned on
  `TX_8X32` (still in the `|log2W - log2H| == 2` family but
  outside both currently-landed endpoints).

  Spec provenance: `docs/video/av1/av1-spec.txt` §7.13.3
  (av1-spec p.305 — the `Round2(T[j] * 2896, 12)` rectangular
  row-pass scale fires only on `Abs(log2W - log2H) == 1`),
  §5.11.40 `compute_tx_type()` (p.92 — `txSzSqrUp > TX_32X32 ⇒
  return DCT_DCT` short-circuit does NOT fire on this pair),
  §5.11.48 / §6.10.19 (p.178-180 — full reachable `tx_type`
  matrix on `txSzSqrUp ≤ TX_32X32` shapes), §7.12.3 (p.294 —
  `dqDenom = 1` default arm for this pair). Only this crate,
  oxideav-core public API, and `docs/video/av1/` material were
  consulted. No external library source, no web search.

- encoder r250 (2026-06-07): extend the rectangular TX_SIZE family
  arc on the encoder's 2D forward-transform dispatcher
  (`encoder::forward_transform_2d::forward_transform_2d`) by the
  first `|log2W - log2H| == 2` pair — `TX_16X64` and `TX_64X16`,
  the rectangular pair at `min(log2W, log2H) == 4` /
  `max(log2W, log2H) == 6` (short-side-16 / long-side-64). This is
  the smallest spec-aligned delta from r244's `TX_32X64` /
  `TX_64X32` pair (same length-64 axis, short axis 32 ⇒ 16).

  Per §5.11.40 `compute_tx_type()` the `txSzSqrUp > TX_32X32 ⇒
  return DCT_DCT` short-circuit forces this pair to the
  `TX_SET_DCTONLY` set — `DCT_DCT` is the only reachable `tx_type`
  ordinal per §6.10.19. The dispatcher composes `forward_dct_16` ×
  `forward_dct_64` (TX_16X64) / `forward_dct_64` × `forward_dct_16`
  (TX_64X16) in the standard column-then-row order; per-axis kernel
  reachability is `forward_dct_dispatch (n in 2..=6)` covering both
  16 (n=4) and 64 (n=6). The other per-axis selectors
  (`forward_idtx_dispatch n in 2..=5`, `forward_adst_dispatch
  n in 2..=4`) are irrelevant on this pair because §6.10.19 has
  already forced DCT_DCT.

  Per §7.13.3 (av1-spec p.305), the `Round2(T[j] * 2896, 12)`
  rectangular row-pass step gates on EXACTLY `Abs(log2W - log2H) ==
  1` — so the encoder's mirror step (applied after the row kernel)
  also skips on this pair. The dispatcher's existing
  `log2_w.abs_diff(log2_h) == 1` gate already handles this; the
  round 250 change extends only the front-of-function `assert!`
  that accepts the new shape and adds the test surface.

  Analytic per-cell round-trip scale: `(N_w * N_h) /
  2^(row_shift + col_shift)` with NO rectangular factor =
  `(16 * 64) / 2^(2 + 4)` = `1024 / 64` = `16`. Empirical
  constant-DC scale is `4` (input `4` ⇒ recovered `16` within ±2
  LSBs per cell) — the documented `4×` constant-DC input-mass
  factor across the rectangular family. Both shapes pull the
  §7.12.3 `dqDenom = 4` branch (any 64-axis transform), already
  routed by `super::forward_quantize`.

  +9 forward_transform_2d tests (64 → 73): zero-input ⇒ zero-output
  on both shapes (TX_16X64, TX_64X16), pseudorandom DCT_DCT
  round-trips (bound ±2 + max_err = 64 mirrors the TX_64X64 /
  TX_32X64 / TX_64X32 envelopes — length-64 axis drives the 16-bit
  between-stage clamp closer to saturation), constant-DC dominance
  + recovery probes (input 4 ⇒ recovered 16 within ±2 LSBs per
  cell). Zero-input matrix walk gains 2 new cases (TX_16X64,
  TX_64X16 with DCT_DCT only). Out-of-arc panic guard stays on
  TX_4X16 — still a `|log2W - log2H| == 2` shape but with
  `min_log2 = 2`, outside the round-250 (`min_log2 = 4`) slice.

  Spec provenance: `docs/video/av1/av1-spec.txt` §7.13.3
  (av1-spec p.305 — the `Round2(T[j] * 2896, 12)` rectangular
  row-pass scale fires only on `Abs(log2W - log2H) == 1`),
  §5.11.40 `compute_tx_type()` (p.92 — `txSzSqrUp > TX_32X32 ⇒
  return DCT_DCT`), §6.10.19 (p.178 — `TX_SET_DCTONLY` contains
  only DCT_DCT), §7.12.3 (av1-spec p.294 — `dqDenom = 4` for any
  64-axis transform). Only this crate, `oxideav-core` public API,
  and `docs/video/av1/` material were consulted. No external
  library source, no web search.

- public-API r249 (2026-06-07): graduate the dyn-driver 4:2:0 YUV
  intra-only codepath into the top-level public entry point. Before
  this round, `oxideav_av1::encode_av1(pixels, width, height)` was a
  hard-stubbed `Err(Error::NotImplemented)` while
  `encode_intra_frame_yuv_dyn` had been a crate-public dyn encoder
  since r230 (with the §3 13-mode intra picker on chroma graduated in
  r229 / r231 / r232 and frame-size generalisation across `[8, 64]`
  per axis in r230). The public `encode_av1` now interprets `pixels`
  as the planar `Y || U || V` byte layout (Y row-major
  `width × height`, then U + V each `(width/2) × (height/2)`) and
  delegates to `encode_intra_frame_yuv_dyn` for any `(width, height)`
  the dyn driver accepts (multiples of `MIN_DIM = 8`, each axis in
  `[8, MAX_DIM] = [8, 64]`). Out-of-scope inputs surface as
  `Error::PartitionWalkOutOfRange` rather than the historical
  `NotImplemented`.

  The matching public `decode_av1` docstring is brought up to date
  with the actual accepted scope: the §5.5.1 dispatch in
  `decoder::pixel_driver::decode_frame` was already routing through
  the r230 dyn driver (`decode_frame_dyn`) for any 4:2:0 frame
  `(w, h) != (16, 16)`, through the r235 monochrome dyn driver
  (`decode_frame_dyn_y`) for `mono_chrome == true`, and through the
  r207 / r214 multi-super-block grid walk for any axis `> MAX_DIM`.
  All those codepaths — UV_CFL_PRED, frame-size generalisation past
  16×16, partial-SB right edge, multi-SB tiling, and base_q_idx > 0
  lossy quant — are reachable through the public surface; this round
  documents that contract in the `decode_av1` docstring.

  +6 integration tests (67 → 73 in
  `tests/encode_decode_pixel_roundtrip.rs`):
  `public_encode_av1_decode_av1_roundtrip_8x8_corner_lossless` (the
  smallest aligned extent the dyn driver accepts),
  `public_encode_av1_decode_av1_roundtrip_16x16_lossless` (the
  fixed-size `Frame::Yuv420_16x16` arm under the dyn-encoded IVF —
  the decoder routes back to the fixed-size path because the IVF
  frame_width / frame_height match the fixed-size driver's scope),
  `public_encode_av1_decode_av1_roundtrip_32x24_rectangular_lossless`
  (a non-square dyn extent surfacing the `Frame::Yuv420Dyn` arm), and
  three refusal probes pinning the validation propagation —
  `public_encode_av1_rejects_wrong_pixel_buffer_length`,
  `public_encode_av1_rejects_non_multiple_of_8_dim`,
  `public_encode_av1_rejects_above_max_dim`.

  Wider extents (multi-super-block) + lossy quant + monochrome remain
  reachable through the per-driver crate-public entries
  (`encoder::encode_intra_frame_yuv_dyn_multi_sb`,
  `encoder::encode_intra_frame_yuv_dyn_with_q`,
  `encoder::encode_intra_frame_y_dyn`, etc.); a single
  `(pixels, width, height)` public signature would have to multiplex
  pixel format + quantizer + multi-SB through extra parameters, so
  they stay on the per-driver entries until the public surface grows
  an explicit config struct in a later arc.

  Wall respected: only this crate, `oxideav-core` public API, and
  `docs/video/av1/` material were consulted. No external library
  source, no web search.

## [0.1.10](https://github.com/OxideAV/oxideav-av1/compare/v0.1.9...v0.1.10) - 2026-06-07

### Other

- drop release-plz.toml — use release-plz defaults across the workspace
- encoder r244: extend rectangular TX_SIZE arc by TX_32X64 / TX_64X32
- encoder r241: extend rectangular TX_SIZE arc by TX_16X32 / TX_32X16
- encoder r238: extend rectangular TX_SIZE arc by TX_8X16 / TX_16X8
- encoder r235: rectangular TX_SIZE family on forward_transform_2d (TX_4X8 / TX_8X4)
- decoder/encoder r223: §8.2.6 post-renorm invariant probes
- multi-super-block 4:2:0 YUV dyn driver (extends ceiling to 128)
- encoder/decoder r207: multi-super-block tiling on the Y-only dyn driver
- encoder/decoder r235: Y-only (monochrome) on the dyn driver
- encoder/decoder r197/r234: rectangular frame extents on the dyn driver
- encoder/decoder r233: base_q_idx > 0 (lossy quant) on the dyn driver
- encoder/decoder r232: UV_CFL_PRED on the dynamic-extent driver

- encoder r244 (2026-06-07): extend the rectangular TX_SIZE family
  arc on the encoder's 2D forward-transform dispatcher
  (`encoder::forward_transform_2d::forward_transform_2d`) by the
  last `|log2W - log2H| == 1` pair on the chain — the short-side-32
  pair `TX_32X64` and `TX_64X32` (the max-kernel pair on the
  `Abs(log2W - log2H) == 1` family). The dispatcher now accepts the
  five square sizes plus eight rectangular sizes (`TX_4X8` /
  `TX_8X4` from r235 + `TX_8X16` / `TX_16X8` from r238 +
  `TX_16X32` / `TX_32X16` from r241 + `TX_32X64` / `TX_64X32` from
  r244). Same composition shape — column pass first (64-tall
  `forward_dct_64` for TX_32X64 / 32-tall `forward_dct_32` for
  TX_64X32) followed by the row pass plus the per-row
  encoder-mirror `Round2(T[j] * 2896, 12)` rectangular post-scale.
  The per-axis kernel selectors extend without modification:
  `forward_dct_dispatch` (n in 2..=6) covers both 32 and 64;
  `forward_idtx_dispatch` (n in 2..=5) covers 32 only (n=6 /
  length-64 is out of range); `forward_adst_dispatch` (n in 2..=4)
  covers neither, so any ADST × {32|64} tx_type combination is
  forced to DCT by §6.10.19. For TX_32X64 the row selector picks
  IDTX via tx_type = V_DCT (row length 32 = in IDTX range, col
  length 64 = in DCT range); for TX_64X32 the col selector picks
  IDTX via tx_type = H_DCT (col length 32 = in IDTX range, row
  length 64 = in DCT range). The empirical round-trip per-cell
  scale on a constant-DC probe is `8 ×` input (input 2 ⇒ recovered
  16): the larger `N_w * N_h = 2048` kernel norm gains another
  `4×` over the short-side-16 pair's `512` while the row-shift
  envelope stays at `Transform_Row_Shift = 1`, so the full `4×`
  lands in the per-cell scale (`2 × 4 = 8`). Per §7.12.3,
  `dqDenom = 4` for both `TX_32X64` and `TX_64X32` (the 64-axis
  presence promotes the §7.12.3 dqDenom from `2` to `4`); the
  forward quantizer already routes this through
  `crate::cdf::dequant_denom`.

  +8 lib tests (1580 → 1588): 2 zero-input probes (TX_32X64 /
  TX_64X32 ⇒ zero-out), 2 DCT_DCT pseudorandom roundtrips at the
  empirically-derived per-cell scale `8` (input bound `±2` + loose
  `max_err = 64` envelope mirrors the TX_64X64 square test —
  length-64 axis drives the inverse pipeline's 16-bit between-stage
  clamp closer to saturation than the constant-DC probe does, and
  the DCT-64 butterfly graph's 31-step accumulated `Round2(_, 12)`
  floor adds more LSB jitter), 2 constant-DC probes pinning the
  DC-dominance + a recovery bound (input 2 ⇒ recovered 16 within
  ±2 LSBs per cell), 1 V_DCT (TX_32X64, col DCT length 64 + row
  IDTX length 32 — the only IDTX-reachable axis combination), 1
  H_DCT (TX_64X32, row DCT length 64 + col IDTX length 32, by
  transpose). The out-of-arc rectangular-panic guard is updated to
  assert `TX_4X16` panics (a `|log2W - log2H| == 2` size, still out
  of arc) — `TX_32X64` is now in-arc. The zero-input matrix-walk
  test gains 4 new cases (TX_32X64 / TX_64X32 with DCT_DCT and
  V_DCT / H_DCT respectively).

  Out of scope (next arc): the `|log2W - log2H| == 2` family
  (`TX_4X16` / `TX_16X4` / `TX_8X32` / `TX_32X8` / `TX_16X64` /
  `TX_64X16`), which per §7.13.3 av1-spec p.305 does follow a
  different path — the `× 2896` rectangular post-scale fires only
  on the `Abs(log2W - log2H) == 1` branch, so this family stays at
  the bare per-axis kernel composition.

- encoder r241 (2026-06-06): extend the rectangular TX_SIZE family
  arc on the encoder's 2D forward-transform dispatcher
  (`encoder::forward_transform_2d::forward_transform_2d`) by one
  more `|log2W - log2H| == 1` pair — the short-side-16 pair
  `TX_16X32` and `TX_32X16`. The dispatcher now accepts the five
  square sizes plus six rectangular sizes (`TX_4X8` / `TX_8X4`
  from r235 + `TX_8X16` / `TX_16X8` from r238 + `TX_16X32` /
  `TX_32X16` from r241). Same composition shape — column pass
  first (32-tall `forward_dct_32` for TX_16X32 / 16-tall
  `forward_dct_16` for TX_32X16) followed by the row pass plus the
  per-row encoder-mirror `Round2(T[j] * 2896, 12)` rectangular
  post-scale. The per-axis kernel selectors extend without
  modification: `forward_dct_dispatch` (n in 2..=6 covers both 16
  and 32), `forward_idtx_dispatch` (n in 2..=5 covers both 16 and
  32). ADST is only reachable on the length-16 axis (the
  §7.13.2.9 dispatcher caps at n=4), so the §6.10.19 tx_type
  derivation routes ADST × length-32 combinations to DCT — for
  TX_16X32 the row selector picks ADST via tx_type = DCT_ADST
  (row length 16 = in range); for TX_32X16 the col selector picks
  ADST via tx_type = ADST_DCT (col length 16 = in range). The
  empirical round-trip per-cell scale on a constant-DC probe is
  `2 ×` input (input 8 ⇒ recovered 16): the larger N_w × N_h
  kernel norm (`512` vs `128` for TX_8X16) gains another `4×`
  over the short-side-8 pair while the row-shift envelope stays at
  `Transform_Row_Shift = 1`, so the full `4×` lands in the per-cell
  scale (`1/2 × 4 = 2`). Per §7.12.3, `dqDenom = 2` for both
  `TX_16X32` and `TX_32X16` (the 32-axis presence on either side
  promotes the §7.12.3 dqDenom from `1` to `2`).

  +12 lib tests (1568 → 1580): 2 zero-input probes (TX_16X32 /
  TX_32X16 ⇒ zero-out), 2 DCT_DCT pseudorandom roundtrips at the
  empirically-derived per-cell scale `2`, 2 constant-DC probes
  pinning the DC-dominance + a recovery bound (input 8 ⇒ recovered
  16 within ±2 LSBs per cell), 1 DCT × ADST (TX_16X32), 1
  ADST × DCT (TX_32X16) keeping ADST on the length-16 axis, 2
  IDTX nonzero-roundtrip drills (IDTX scale envelope is checked
  separately from DCT), 1 V_DCT (TX_16X32, col DCT length 32 +
  row IDTX length 16), 1 H_DCT (TX_32X16, row DCT length 32 +
  col IDTX length 16). The out-of-arc rectangular-panic guard is
  updated to assert TX_32X64 panics (TX_16X32 is now in-arc).

- encoder r238 (2026-06-05): extend the rectangular TX_SIZE family
  arc on the encoder's 2D forward-transform dispatcher
  (`encoder::forward_transform_2d::forward_transform_2d`) by one
  more `|log2W - log2H| == 1` pair — the short-side-8 pair
  `TX_8X16` and `TX_16X8`. The dispatcher now accepts the five
  square sizes plus four rectangular sizes (`TX_4X8` / `TX_8X4`
  from r235 + `TX_8X16` / `TX_16X8` from r238). Same composition
  shape as r235 — column pass first (16-tall `forward_dct_16` for
  TX_8X16 / 8-tall `forward_dct_8` for TX_16X8) followed by the row
  pass (8-wide for TX_8X16 / 16-wide for TX_16X8) plus the per-row
  encoder-mirror `Round2(T[j] * 2896, 12)` rectangular post-scale.
  The per-axis kernel selectors all extend without modification:
  `forward_dct_dispatch` (n in 2..=6), `forward_adst_dispatch` (n in
  2..=4 — both log2_w = 3 and log2_h = 4 are in range, so ADST × DCT
  / DCT × ADST / ADST × ADST / the FLIPADST family / V_/H_ variants
  all reach), `forward_idtx_dispatch` (n in 2..=5). The empirical
  round-trip per-cell scale on a constant-DC probe is `1/2` of the
  input (vs `1/4` for short-side-4 — input `32` ⇒ recovered `16`):
  the larger N_w × N_h kernel norm (128 vs 32) gains `4×` over the
  short-side-4 pair while the larger row-shift envelope
  (`Transform_Row_Shift[TX_8X16] = 1` vs `0` for TX_4X8) eats back
  `2×`, netting the `2×` larger per-cell round-trip scale. The
  `dequant_denom(TX_8X16) == dequant_denom(TX_16X8) == 1` lookup
  matches the existing TX_4X4 / TX_8X8 / TX_4X8 / TX_8X4 path, so
  the forward-quantize / dequantize round-trip on the new shapes is
  bit-exact at q_index = 0. The remaining 10 rectangular sizes
  (`TX_16X32` / `TX_32X16` / `TX_32X64` / `TX_64X32` / `TX_4X16` /
  `TX_16X4` / `TX_8X32` / `TX_32X8` / `TX_16X64` / `TX_64X16`)
  remain out of arc.

  +13 lib tests (1555 → 1568): 2 zero-input probes (TX_8X16 /
  TX_16X8 ⇒ zero-out), 2 DCT_DCT pseudorandom roundtrips at the
  empirically-derived per-cell scale `1/2`, 2 constant-DC probes
  pinning the DC-dominance + a recovery bound (input 32 ⇒ recovered
  16 within ±2 LSBs per cell), 1 ADST × DCT (TX_8X16), 1 DCT × ADST
  (TX_16X8), 1 IDTX (TX_8X16), 1 FLIPADST × FLIPADST (TX_16X8),
  1 ADST × ADST (TX_8X16) covering the kernel-selector matrix; 2
  forward-quantize / dequantize round-trip drills exercising the
  128-cell rectangular buffer through the existing `dqDenom == 1`
  quantizer path; and 2 entries appended to
  `dequant_denom_per_tx_size` for TX_8X16 / TX_16X8 (both = 1). The
  out-of-arc rectangular-panic guard is updated to assert TX_16X32
  panics (TX_8X16 is now in-arc).

- encoder r235 (2026-06-05): rectangular TX_SIZE family wiring on
  the encoder's 2D forward-transform dispatcher
  (`encoder::forward_transform_2d::forward_transform_2d`). Adds
  the `|log2W - log2H| == 1` short-side-4 pair — `TX_4X8` and
  `TX_8X4` — to the previously square-only dispatcher. The
  composition is still column-pass-first (8-tall `forward_dct_8`
  for TX_4X8 / 4-tall `forward_dct_4` for TX_8X4) followed by the
  row pass (4-wide for TX_4X8 / 8-wide for TX_8X4) plus the new
  encoder-mirror of §7.13.3's `Round2(T[j] * 2896, 12)` rectangular
  scale, applied per row AFTER the row kernel runs (the encoder's
  last pass — the time-reverse of the decoder's pre-row-kernel
  rectangular scale on the same `2896` constant). Both encoder and
  decoder thus contribute one factor of `2896 / 4096`, giving a
  combined rectangular gain of `(2896 / 4096)^2 ≈ 1/2` per
  rectangular axis pair. The per-axis kernel selectors
  (DCT × DCT, ADST × DCT, DCT × ADST, ADST × ADST, FLIPADST
  family, V_/H_ variants, IDTX) all carry over without modification
  — the per-axis log-sizes (`log2_w = 2`, `log2_h = 3` for TX_4X8)
  are both inside the existing forward-DCT and forward-ADST ranges
  (n in 2..=6 for DCT, n in 2..=4 for ADST, n in 2..=5 for IDTX).
  The §7.12.3 step-3 FLIPADST axis-flip carries over verbatim —
  the flip is a spatial-residual reordering that runs before the
  transform on rectangular and square shapes identically. The
  remaining 12 rectangular sizes (`TX_8X16` / `TX_16X8` /
  `TX_16X32` / `TX_32X16` / `TX_32X64` / `TX_64X32` / `TX_4X16` /
  `TX_16X4` / `TX_8X32` / `TX_32X8` / `TX_16X64` / `TX_64X16`)
  remain out of arc — the `|log2W - log2H| == 2` family in
  particular does NOT take the §7.13.3 `× 2896` rectangular
  post-scale and needs its own arc.

  +12 lib tests (1543 → 1555): 2 zero-input probes (TX_4X8 /
  TX_8X4 ⇒ zero-out), 2 DCT_DCT pseudorandom roundtrips at the
  empirically-derived per-cell scale `1/4` (same scale shape as
  TX_4X4 — the short-side-4 kernel norm carries the per-cell
  magnitude), 2 constant-DC probes pinning the DC-dominance + a
  recovery bound (input 64 ⇒ recovered 16 within ±2 LSBs per
  cell), 1 ADST × DCT + 1 DCT × ADST + 1 IDTX + 1
  FLIPADST × FLIPADST coverage for the rectangular family, and 2
  forward-quantize / dequantize round-trip drills exercising the
  32-cell rectangular buffer through the existing TX_4X4-shape
  quantizer (`dqDenom == 1` for the short-side-4 family, so the
  round-trip is bit-exact at q_index = 0). The square-only
  rectangular-panic test is rewritten to assert `TX_8X16` (an
  out-of-arc rectangular size) panics with the new "not supported
  in this arc" message.

- decoder/encoder r223 (2026-06-03): §8.2.6 post-renormalisation
  invariant probes wired into `SymbolDecoder::renormalize` and
  `SymbolWriter::write_symbol`. Both surfaces now check the two §8.2.6
  invariants after every renormalisation step rather than trusting the
  partition arithmetic blindly:
    1. `32768 ≤ SymbolRange ≤ 65535` (the §8.2.6 ordered steps 1 + 2
       restore the range to the top half of the 16-bit window after
       every symbol decode / encode).
    2. `SymbolValue < SymbolRange` (the §8.2.6 symbol-search loop
       terminates the moment `SymbolValue >= cur` lands inside the
       selected symbol's interval; the subsequent `SymbolValue -= cur`
       / `SymbolRange = prev − cur` rewrite plus renormalisation
       preserve this ordering).
  A violation surfaces as a new `Error::SymbolStateInvariantBroken`
  variant rather than panicking on `debug_assert!`; this lets bitstream
  conformance failures and internal CDF / partition-arithmetic bugs be
  reported through the normal error path instead of leaving the codec
  in an undefined state. The two invariants are re-stated as
  cross-implementation oracles in the freshly-staged trace doc
  `docs/video/av1/fixtures/issue_796/msac-trace.md` ("Invariants
  observed across all 256 rows" section, both holding row-for-row
  across the §8.2 256-symbol capture).

  Six new unit tests pin the new surface: two §8.2.6 invariant sweeps
  on the decoder side over four different starting byte windows
  (`post_renorm_range_in_top_half_across_decodes`,
  `post_renorm_value_below_range_across_decodes`), a direct boundary
  test of `check_post_renorm_invariants`
  (`invariant_check_returns_error_on_each_violation` — in-range +
  boundary OK + each of the three out-of-range edges), an adaptive
  4-symbol CDF run with §8.3 updates that mirrors the trace doc's
  "all 256 rows" assertion as closely as a synthetic in-tree input
  allows (`invariants_hold_across_adaptive_multisymbol_run`), a mixed
  bool / multi-symbol encode→decode pair that exercises both writer
  and reader simultaneously
  (`encoder_decoder_state_stays_in_top_half_window`), and a direct
  boundary test of the encoder's `check_range_invariant`
  (`encoder_range_invariant_check_distinguishes_boundaries`).

  Total: 1537 → 1543 lib tests. No public API change beyond the new
  `Error::SymbolStateInvariantBroken` variant; existing happy-path
  tests continue to pass byte-exact, confirming the runtime checks are
  no-ops on every conformant decode produced so far.

- encoder/decoder r214: multi-super-block tiling on the 4:2:0 YUV dyn
  driver — extends the single-SB YUV path's ceiling from `MAX_DIM = 64`
  to a new `MAX_DIM_YUV_MULTI_SB = 128` by walking the §5.11.1 SB grid
  literally (`for r += sbSize4; for c += sbSize4;
  decode_partition(r, c, sbSize)` with `sbSize = BLOCK_64X64`,
  `sbSize4 = 16`), reusing the r207 `sb_grid_origins` /
  `sb_grid_dispatch_order_leaves` helpers. New encoder entries
  `encode_intra_frame_yuv_dyn_multi_sb` / `_with_q` accept a
  `Yuv420FrameMultiSb { width, height, y, u, v }` input at any extent
  `(w, h) ∈ {8..=128} × {8..=128}` aligned to 8; the decoder mirrors
  by lifting `decode_frame_dyn`'s cap from `MAX_DIM` to
  `MAX_DIM_YUV_MULTI_SB` and dispatching to the multi-SB walk when
  either dimension exceeds `MAX_DIM`. Extents ≤ 64 keep the
  single-root behaviour (IVF bytes byte-for-byte identical to prior
  output). §5.11.5 HasChroma composes across SBs by construction
  (`(mi_r & 1) != 0 && (mi_c & 1) != 0` over frame-global mi coords),
  §7.11.5.3 CFL subsampled-luma window already clips at the chroma
  plane edge for any extent, per-SB context resets are vacuous at the
  driver's hard-coded ctx=0 leaves. +9 lib unit tests
  (1528 → 1537: validate sweep over `{8..=128}² ∩ mod 8 = 0`,
  validate rejection on oversized dims + wrong plane lengths,
  flat-grey 96×64 internal recon equality + SH flag carriage, 128×128
  pseudo-random lossless bit-exact recon on Y/U/V, 64×64 multi-SB ↔
  single-SB IVF byte-identity, q-divergence at 96×64, 10-extent
  encode→decode→bit-exact roundtrip on all three planes, q∈{1,32,200}
  lossy self-consistency at 96×64, IVF dimension-write check on 5
  extents). +5 integration tests in `encode_decode_pixel_roundtrip`
  (62 → 67) covering 96×64 + 128×128 lossless, 6-extent rectangular-
  edge sweep, q=32 lossy self-consistency, and IVF v0 dimension
  carriage. Provenance: docs/video/av1/av1-spec.txt §5.11.1 (decode_
  tile SB-grid walk + sbSize/sbSize4 derivation), §5.11.5 (HasChroma
  predicate), §7.11.5.3 (CFL subsampled-luma window — already
  edge-clipping), §5.5.2 (4:2:0 color-config — unchanged), §6.10.2
  (clear_left/above_context — vacuous at ctx=0).
- encoder/decoder r207: multi-super-block tiling on the monochrome
  (Y-only) dyn driver — extends the r235 Y-only path's ceiling from
  `MAX_DIM = 64` to a new `MAX_DIM_Y_MULTI_SB = 128` by walking the
  §5.11.1 SB grid (`for r += sbSize4; for c += sbSize4; decode_partition
  (r, c, sbSize)`) literally over `sbSize = BLOCK_64X64`,
  `sbSize4 = SB_SIZE4_64 = 16`. New encoder entries
  `encode_intra_frame_y_dyn_multi_sb` / `_with_q` accept a
  `MonoYFrameMultiSb { width, height, y: Vec<u8> }` input at any
  rectangular extent `(w, h) ∈ {8..=128} × {8..=128}` aligned to 8;
  the decoder mirrors by lifting `decode_frame_dyn_y`'s cap from
  `MAX_DIM` to `MAX_DIM_Y_MULTI_SB` and dispatching to the multi-SB
  walk when either dimension exceeds `MAX_DIM`. Extents ≤ 64 keep the
  single-root behaviour from r235 (IVF bytes byte-for-byte identical
  to prior output). Each SB is a fresh `BLOCK_64X64`-rooted
  `EncodeNode` tree; edge SBs with `(width % 64) != 0` or
  `(height % 64) != 0` rely on the existing `EncodeNode::dummy_oob` +
  §5.11.4 line-1 OOB short-circuit (introduced in r234 for
  rectangular extents ≤ 64) to swallow partial-coverage quadrants.
  New `sb_grid_origins` / `sb_grid_dispatch_order_leaves` helpers
  expose the §5.11.1 SB-row-major ordering for tests + future
  YUV/CFL multi-SB extensions.
  +12 lib unit tests (1516 → 1528: `sb_grid_origins` shape table over
  single-SB and 2×2 grids, `sb_grid_dispatch_order_leaves` exactly-
  once coverage at 96×64 + 128×128, `MonoYFrameMultiSb::validate`
  acceptance over `{8..=128}² ∩ mod 8 = 0` + oversized + wrong-len
  rejection, encoder flat-grey internal recon at 96×64, pseudo-
  random lossless internal recon at 128×128, multi-SB ↔ legacy
  64×64 byte-for-byte parity, lossy q ≠ 0 byte divergence at 96×64,
  10-extent encode→decode→bit-exact integration roundtrip, and a
  3-qindex lossy self-consistency on 96×64) + 5 new integration
  tests (57 → 62 on `encode_decode_pixel_roundtrip`: lossless
  96×64, lossless 128×128, 6-extent rectangular-edge sweep
  covering all partial-coverage SB shapes, lossy q=32 self-
  consistency, IVF v0 dimension-write check on multi-SB extents).

- encoder/decoder r235: Y-only (monochrome) on the dynamic-extent
  driver — `MonoYFrame { width, height, y: Vec<u8> }` input + new
  `encode_intra_frame_y_dyn` / `encode_intra_frame_y_dyn_with_q`
  entries on the encoder side, mirrored on the decoder by a
  `Frame::YDyn { width, height, y: Vec<u8> }` variant and a new
  `decode_frame_dyn_y` dispatched from `decode_frame` whenever the
  parsed SH carries `mono_chrome = true`. The §5.5.2 mono color-config
  arm (`num_planes = 1`, `subsampling_x = subsampling_y = true` per
  spec) is emitted by a new `build_intra_only_y_8bit_seq` helper; the
  FH builder is shared with the YUV path because the §5.11.5
  `HasChroma` / §5.11.22 line-8 / §5.11.39 walks are already gated on
  `NumPlanes > 1` in the existing writers + readers, so the chroma
  syntax silently vanishes from the bitstream on the mono arm. The
  encoder skips the per-leaf chroma residual + prediction + coefficient
  pass entirely; the decoder mirrors with a streamlined
  `decode_block_leaf_y` that reads only skip + y_mode + luma
  coefficients. Validates at every rectangular extent
  `(w, h) ∈ {8, 16, 24, 32, 40, 48, 56, 64}` × itself
  (1504 → 1516 lib tests; 12 new tests pin: SH mono-flag carriage,
  `MonoYFrame::validate` rejection of mis-sized planes, lossless WHT
  bit-exact internal reconstruction, `with_q(0)` byte equivalence to
  the legacy entry, `q > 0` byte-divergence, and a 14-extent
  encode→decode→pixel-equality integration roundtrip that covers
  every square + rectangular shape the dyn driver admits).

- encoder/decoder r197/r234: rectangular frame extents on the
  dynamic-extent driver — promoted from "works incidentally" to a
  tested invariant. Width and height are now independently bounded by
  `MIN_DIM..=MAX_DIM` (8..=64, multiples of 8); the §5.11.4 partition
  tree's per-quadrant `r >= mi_rows || c >= mi_cols` early return +
  `EncodeNode::dummy_oob` sentinel already let the encoder + decoder
  walk any rectangular `(mi_rows, mi_cols)` rooted in the smallest
  power-of-two super-block covering `max(mi_cols, mi_rows)` — the
  tests pin the property at `{8,16}×{16,8}` (BLOCK_16X16 root),
  `{16,32}×{32,16}` (BLOCK_32X32), `{24,32}×{32,24}` /
  `{32,48}×{48,32}` (BLOCK_32X32 + partial-coverage), and
  `{40,16}×{16,40}` / `{32,64}×{64,32}` (BLOCK_64X64). Scope docs on
  `encoder::pixel_driver_dyn` + `decoder::pixel_driver_dyn` updated
  to reflect that "no rectangular partitions" referred to the
  §3 `TX_4X8` / `TX_8X4` / `TX_8X16` / `TX_16X8` **transform-size**
  family (still out of scope this arc — TX_4X4 leaves everywhere) and
  NOT to the frame extent (which is now in scope). +6 lib tests
  (1498 → 1504; rectangular `dispatch_order_leaves` coverage,
  rectangular `root_super_block` shape table, every-extent
  `validate` acceptance, flat-grey rectangular recon equality,
  every-extent lossy q-grid, and a lib-level rectangular-lossy
  self-decode contract). +22 integration tests
  (35 → 57 in `encode_decode_pixel_roundtrip`): 12 lossless
  pseudorandom rectangular extents (16×32 / 32×16 / 8×16 / 16×8 /
  24×32 / 32×24 / 40×16 / 16×40 / 48×32 / 32×48 / 32×64 / 64×32) +
  9 lossy rectangular roundtrips across q ∈ {1, 16, 32, 64, 128, 200,
  255} + 1 IVF v0 header dimension-write check on six rectangular
  shapes. Rectangular **TX_SIZE** family, multi-super-block tiling,
  §5.11.18 inter mode_info, monochrome / 4:2:2 / 4:4:4 sampling, and
  per-segment / per-block delta_q remain out of scope.
- encoder/decoder r196/r233: `base_q_idx > 0` (lossy quant) on the
  dynamic-extent driver. New public entry
  `encode_intra_frame_yuv_dyn_with_q(input, base_q_idx)` (additive;
  the legacy `encode_intra_frame_yuv_dyn` is now a thin wrapper at
  `base_q_idx = 0` that produces byte-for-byte identical IVF +
  reconstruction output). At `base_q_idx > 0` the encoder routes the
  leaf transform through §7.13.3 forward DCT_DCT + §7.12.3 forward
  quantize and emits the FH with `TxModeLargest` (§5.9.21 requires
  `Only4x4` only under §5.9.2 `CodedLossless`); the decoder reads
  `base_q_idx` from the parsed FH and dispatches
  `inverse_transform_2d`'s `lossless` flag symmetrically. Contract:
  `decode_av1(enc.ivf_bytes)` matches `enc.reconstructed_*`
  byte-for-byte at any qindex. New helper
  `build_intra_only_yuv420_8bit_fh_with_q` + lib re-exports. +9 lib
  tests (1489 → 1498) covering the lossy-FH builder, the q=0
  back-compat invariant, qindex enumeration, and the self-decode
  contract. +6 integration tests (29 → 35 in
  `encode_decode_pixel_roundtrip`) covering q=1 flat/64×64, q=16
  horizontal gradient, q=64 pseudo-random, q=255 stress, and the
  with_q(0) ↔ legacy byte-for-byte equality regression guard. Lossy
  rectangular TX, §5.11.18 inter mode_info, frames > 64×64, and
  §5.11.34 per-block delta_q remain out of scope.
- encoder/decoder r232: UV_CFL_PRED on the dynamic-extent driver
  (§7.11.5.3 chroma-from-luma now wired through
  `encode_intra_frame_yuv_dyn` + `Frame::Yuv420Dyn`). Picks the
  §5.11.45 (αU, αV) over a compact `{±1, ±2, ±4}` grid against the
  §7.11.5.3 subsampled-luma window; full encode → decode → pixel-
  equality stays bit-exact at 32×32 and 64×64 on luma-correlated
  chroma. +3 integration tests (29 → 32 in
  `encode_decode_pixel_roundtrip`).

## [0.1.9](https://github.com/OxideAV/oxideav-av1/compare/v0.1.8...v0.1.9) - 2026-05-29

### Other

- release v0.1.8 ([#11](https://github.com/OxideAV/oxideav-av1/pull/11))
- encoder/decoder r231: UV_CFL_PRED (§7.11.5.3 chroma-from-luma)
- encoder/decoder r230: pixel-driver frame-size generalisation
- encoder r229: 13-mode chroma intra prediction picker (DC/V/H/D-modes/SMOOTH/PAETH)
- encoder r228: 13-mode intra prediction picker (DC/V/H/D-modes/SMOOTH/PAETH)
- round 227 — §7.13.3-equivalent forward 2D transform dispatcher
- encoder r226: forward ADST / FLIPADST / IDTX primitives
- encoder r225: forward DCT for sizes 8 / 16 / 32 / 64
- decoder r224: standalone decode_av1 + first encode→decode pixel roundtrip
- encoder r223: pixel-driver chroma path (4:2:0 YUV)
- encoder r222: forward Walsh-Hadamard transform (lossless milestone)
- encoder r221: pixel-space driver (YUV in → IVF out, milestone arc)
- encoder r220: forward quantization primitive
- encoder r219: scrub stale 2.072 / 1.999 numbers from M^T·M comments
- encoder r219: forward 4×4 DCT primitive (pixel-space bootstrap)
- encoder r218: §5.11.36 transform_tree / tx_size writers
- encoder r217: §5.11.4 recursive dispatch driver (intra-only)
- §5.11.4 partition decision-tree writer (round 216)
- round 215: §5.11.39 coefficients() driver loop end-to-end
- round 214: §5.11.39 golomb magnitude tail writer
- round 213: §5.11.39 coeff_base{_eob} + coeff_br writers
- round 212: first §5.11.39 coefficient-encode primitives
- round 211: first per-block syntax writers — intra arm
- encoder r210: §5.11.1 tile_group_obu framing + entropy composition wrappers
- Round 209 — encoder arc 4: §8.2 entropy encoder (SymbolWriter)
- Round 208 — encoder arc 3: trailing_bits + OBU+size wrapper + temporal_unit aggregator + end-to-end IVF smoke
- Round 207 — encoder arc 2: frame_header_obu() writer
- Round 206 — encoder arc 1: bit-output plumbing (BitWriter / OBU framer / sequence_header_obu writer / IVF v0 container)
- Round 205 — wire §7.10.2.5 temporal scan + §7.10.2.6 temporal sample
- refresh stale doc comments now that §9.5.3 QM is LANDED
- §9.5.3 Quantizer matrix tables + §7.12.3 step-1b QM-active arm
- §7.11.3.1 OBMC motion-mode arm wired into predict_inter
- §7.11.3.1 WARPED_CAUSAL motion-mode arm wired into predict_inter
- §7.11.3.1 step-14 compound arm wired into predict_inter
- §7.16 superres upscaling — per-frame pipeline complete
- r199 — §7.18.3 film grain synthesis (close-out post-processing layer)
- r198 — §7.17.2 / §7.17.3 self-guided projection arm
- r197 — §7.17 loop restoration — driver + Wiener arm
- §7.15 CDEF — driver + direction search + primary/secondary filter
- r195 — §7.14 loop filter (deblocking) — driver + edge-strength + narrow/wide filter bodies
- r194 — §7.11.3.1 predict_inter driver skeleton (SIMPLE single-ref arm)
- r193 — §7.11.3.9-10 OBMC overlap-blend leaves
- r192 — §7.11.3.5-8 WARP motion compensation
- r191 — §7.11.3.11-15 compound bodies
- r190 — `decode_block_syntax` inter-arm wire-up + `InterFrameContext`
- §7.11.3 inter prediction — translational single-ref MC kernel
- r188 — §7.11.2.4 six D-mode directional intra prediction + §7.11.2.{7,9,10,11,12} intra-edge helpers + Mode_To_Angle / Dr_Intra_Derivative tables
- r187 — §7.11.2.2 PAETH + §7.11.2.6 SMOOTH / SMOOTH_V / SMOOTH_H leaves + §7.11.2.1 corner derivation
- r186 — §7.11.2.4 V_PRED + H_PRED leaves + §7.11.2.1 AboveRow/LeftCol derivation
- r185 — §7.12.3 step-3 frame-buffer merge + CurrFrame buffers
- r184 — §7.5 / §5.11.41 get_scan(txSz) table dispatcher
- r183 — §7.12.2 dequant tables + §7.12.3 step-1 + §5.11.47 transform_type
- r182 — §7.13 inverse transform process; lift ResidualReconstructUnsupported
- r181 — §5.11.34 residual() outer dispatch + §5.11.36 transform_tree recursion + per-TU §5.11.39 wiring
- r180 — §5.11.30 / §5.11.33 compute_prediction() dispatcher + §7.11.2.5 DC_PRED leaf
- §5.11.39 coefficients reader (the first §5.11.34 body)
- §5.11.x read_interpolation_filter reader
- §5.11.29 read_compound_type reader
- §5.11.28 read_interintra_mode reader
- §5.11.27 read_motion_mode + §7.10.3/§7.10.4 helpers
- §5.11.31 assign_mv + §5.11.32 read_mv_component cascade
- §5.11.23 post-find_mv_stack reader cascade
- §7.10 find_mv_stack spatial-only path
- av1 r171: §5.11.46 palette-entries reader + §5.11.49 get_palette_cache
- round 170 — §5.11.23 inter_block_mode_info() prologue + §5.11.25 read_ref_frames()
- round 169 — §5.11.22 intra_block_mode_info()
- §5.11.17 read_var_tx_size + §5.11.18 inter_frame_mode_info
- round 167 — §5.11.16 read_block_tx_size() + §5.11.15 read_tx_size
- round 166 — §5.11.5 decode_block() syntax-walker skeleton
- round 165 — §5.11.7 / §5.11.22 intra_frame_y_mode syntax element
- round 164 — §5.11.7 use_intrabc syntax element
- round 163 — §5.11.21 get_segment_id() predicted-segment-id helper
- round 162 — §5.11.19 inter_segment_id(preSkip) syntax element
- round 161 — §5.11.7 intra_frame_mode_info() prefix dispatcher
- round 160 — §5.11.8 intra_segment_id() syntax element
- round 159 — §5.11.9 read_segment_id() syntax element
- round 158 — §5.11.20 read_is_inter() syntax element
- round 157 — §5.11.56 read_cdef() + §5.11.55 clear_cdef()
- round 156 — §5.11.13 read_delta_lf() syntax element
- round 155 — §5.11.12 read_delta_qindex() syntax element
- round 154 — §5.11.10 read_skip_mode() syntax element
- round 152 — §5.11.11 read_skip() syntax element
- round 151 — §5.11.4 decode_partition() recursive walker
- round 150 — §9.3 Partition_Subsize table + §3 BLOCK_* enum staging
- round 149 — §5.11.49 palette_tokens caller-side arg derivation
- round 148 — §9.3 block-size conversion tables
- round 147 — §5.11.49 palette_tokens per-plane diagonal walker
- round 146 — §5.11.50 get_palette_color_context derivation
- round 145 — §8.3.2 split_or_horz / split_or_vert cdf derivation
- round 144 — §9.4 wedge-index default-CDF
- round 143 — §9.4 inter-intra default-CDF group
- round 142 — §5.11.40 compute_tx_type() derivation
- r141 follow-up — re-export §8.3.2 helpers + add §3 constants to crate-root glob + lib.rs round-note + sync test count
- §8.3.2 get_coeff_base_ctx / get_br_ctx neighbour derivation (r141)
- round 140 — §9.4 Default_Coeff_Br_Cdf + §8.3.1 init_coeff_cdfs / §8.3.2 selection (coeff_br sub-group)
- Round 139 — coefficient `coeff_base` sub-group default CDF + selectors
- round 138 — §9.4 Default_Coeff_Base_Eob_Cdf + §8.3.1 init_coeff_cdfs / §8.3.2 selection (coeff_base_eob sub-group)
- round 137 — intra-frame transform-type default CDF group
- round 136 — §9.4 default-CDF + §8.3.1 init_coeff_cdfs / §8.3.2 selection (coefficient-token entry sub-group)
- round 135 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (angle-delta subset)
- round 134 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (inter-frame intra-mode subset)
- round 24 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (compound-prediction subset)
- round 23 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (motion-mode subset)
- round 22 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (inter-frame interpolation-filter subset)
- round 21 — §9.4 default CDFs + §8.3.1 / §8.3.2 selection (inter-frame transform-type subset)
- round 20 — transform-size §9.4 default CDFs + §8.3.2 selection
- round 19 follow-up — rename FILTER_INTRA_MODES to spec name INTRA_FILTER_MODES
- round 19 — §9.4 default CDFs + §8.3.1/§8.3.2 selection (palette/filter-intra/CFL subset)
- round 18 — §9.4 default CDFs + §8.3.1/§8.3.2 selection (inter-mode/ref-frame subset)
- Round 17: §9.4 default CDF tables + §8.3.2 selection (motion-vector
- Round 16: §9.4 default CDF tables + §8.3.1/§8.3.2 selection (intra-mode/partition subset)
- round 15 — §8.2 symbol (arithmetic / msac) decoder
- round 14 — inter-frame uncompressed_header() path (set_frame_refs / frame_size_with_refs / ref_frame_idx)
- round 13 — §5.9.24 global_motion_params + §5.9.30 film_grain_params
- round 12 — read_tx_mode() (§5.9.21) wired into streaming parser
- round 11 — wire lr_params() (§5.9.20) into the streaming parser
- round 10 — wire cdef_params() (§5.9.19) into the streaming parser
- round 9 — wire loop_filter_params() (§5.9.11) + CodedLossless derivation
- wire §5.9.17 delta_q_params + §5.9.18 delta_lf_params into streaming parse (round 8)
- frame header r7: wire §5.9.14 segmentation_params (+ §5.9.12 quantization_params) into streaming parse
- round 6 — §5.9.3 allow_intrabc + §5.9.15 tile_info wired into the streaming parse_frame_header walk
- round 5 — §5.9.10 / §5.9.11 / §5.9.12 uncompressed-header tail sub-syntaxes
- round 4 — §5.9.5–§5.9.9 frame-size sub-syntax block
- round 3 — §5.9.2 uncompressed_header prefix parse
- embed sequence-header fixture corpus instead of reading docs/
- round 2: sequence_header_obu parse (§5.5)
- round 1: OBU bytestream walker (§5.3 / §4.10.5)
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

## [0.1.8](https://github.com/OxideAV/oxideav-av1/compare/v0.1.7...v0.1.8) - 2026-05-28

### Other

- encoder/decoder r231: UV_CFL_PRED (§7.11.5.3 chroma-from-luma)
- encoder/decoder r230: pixel-driver frame-size generalisation
- encoder r229: 13-mode chroma intra prediction picker (DC/V/H/D-modes/SMOOTH/PAETH)
- encoder r228: 13-mode intra prediction picker (DC/V/H/D-modes/SMOOTH/PAETH)
- round 227 — §7.13.3-equivalent forward 2D transform dispatcher
- encoder r226: forward ADST / FLIPADST / IDTX primitives
- encoder r225: forward DCT for sizes 8 / 16 / 32 / 64
- decoder r224: standalone decode_av1 + first encode→decode pixel roundtrip
- encoder r223: pixel-driver chroma path (4:2:0 YUV)
- encoder r222: forward Walsh-Hadamard transform (lossless milestone)
- encoder r221: pixel-space driver (YUV in → IVF out, milestone arc)
- encoder r220: forward quantization primitive
- encoder r219: scrub stale 2.072 / 1.999 numbers from M^T·M comments
- encoder r219: forward 4×4 DCT primitive (pixel-space bootstrap)
- encoder r218: §5.11.36 transform_tree / tx_size writers
- encoder r217: §5.11.4 recursive dispatch driver (intra-only)
- §5.11.4 partition decision-tree writer (round 216)
- round 215: §5.11.39 coefficients() driver loop end-to-end
- round 214: §5.11.39 golomb magnitude tail writer
- round 213: §5.11.39 coeff_base{_eob} + coeff_br writers
- round 212: first §5.11.39 coefficient-encode primitives
- round 211: first per-block syntax writers — intra arm
- encoder r210: §5.11.1 tile_group_obu framing + entropy composition wrappers
- Round 209 — encoder arc 4: §8.2 entropy encoder (SymbolWriter)
- Round 208 — encoder arc 3: trailing_bits + OBU+size wrapper + temporal_unit aggregator + end-to-end IVF smoke
- Round 207 — encoder arc 2: frame_header_obu() writer
- Round 206 — encoder arc 1: bit-output plumbing (BitWriter / OBU framer / sequence_header_obu writer / IVF v0 container)
- Round 205 — wire §7.10.2.5 temporal scan + §7.10.2.6 temporal sample
- refresh stale doc comments now that §9.5.3 QM is LANDED
- §9.5.3 Quantizer matrix tables + §7.12.3 step-1b QM-active arm
- §7.11.3.1 OBMC motion-mode arm wired into predict_inter
- §7.11.3.1 WARPED_CAUSAL motion-mode arm wired into predict_inter
- §7.11.3.1 step-14 compound arm wired into predict_inter
- §7.16 superres upscaling — per-frame pipeline complete
- r199 — §7.18.3 film grain synthesis (close-out post-processing layer)
- r198 — §7.17.2 / §7.17.3 self-guided projection arm
- r197 — §7.17 loop restoration — driver + Wiener arm
- §7.15 CDEF — driver + direction search + primary/secondary filter
- r195 — §7.14 loop filter (deblocking) — driver + edge-strength + narrow/wide filter bodies
- r194 — §7.11.3.1 predict_inter driver skeleton (SIMPLE single-ref arm)
- r193 — §7.11.3.9-10 OBMC overlap-blend leaves
- r192 — §7.11.3.5-8 WARP motion compensation
- r191 — §7.11.3.11-15 compound bodies
- r190 — `decode_block_syntax` inter-arm wire-up + `InterFrameContext`
- §7.11.3 inter prediction — translational single-ref MC kernel
- r188 — §7.11.2.4 six D-mode directional intra prediction + §7.11.2.{7,9,10,11,12} intra-edge helpers + Mode_To_Angle / Dr_Intra_Derivative tables
- r187 — §7.11.2.2 PAETH + §7.11.2.6 SMOOTH / SMOOTH_V / SMOOTH_H leaves + §7.11.2.1 corner derivation
- r186 — §7.11.2.4 V_PRED + H_PRED leaves + §7.11.2.1 AboveRow/LeftCol derivation
- r185 — §7.12.3 step-3 frame-buffer merge + CurrFrame buffers
- r184 — §7.5 / §5.11.41 get_scan(txSz) table dispatcher
- r183 — §7.12.2 dequant tables + §7.12.3 step-1 + §5.11.47 transform_type
- r182 — §7.13 inverse transform process; lift ResidualReconstructUnsupported
- r181 — §5.11.34 residual() outer dispatch + §5.11.36 transform_tree recursion + per-TU §5.11.39 wiring
- r180 — §5.11.30 / §5.11.33 compute_prediction() dispatcher + §7.11.2.5 DC_PRED leaf
- §5.11.39 coefficients reader (the first §5.11.34 body)
- §5.11.x read_interpolation_filter reader
- §5.11.29 read_compound_type reader
- §5.11.28 read_interintra_mode reader
- §5.11.27 read_motion_mode + §7.10.3/§7.10.4 helpers
- §5.11.31 assign_mv + §5.11.32 read_mv_component cascade
- §5.11.23 post-find_mv_stack reader cascade
- §7.10 find_mv_stack spatial-only path
- av1 r171: §5.11.46 palette-entries reader + §5.11.49 get_palette_cache
- round 170 — §5.11.23 inter_block_mode_info() prologue + §5.11.25 read_ref_frames()
- round 169 — §5.11.22 intra_block_mode_info()
- §5.11.17 read_var_tx_size + §5.11.18 inter_frame_mode_info
- round 167 — §5.11.16 read_block_tx_size() + §5.11.15 read_tx_size
- round 166 — §5.11.5 decode_block() syntax-walker skeleton
- round 165 — §5.11.7 / §5.11.22 intra_frame_y_mode syntax element
- round 164 — §5.11.7 use_intrabc syntax element
- round 163 — §5.11.21 get_segment_id() predicted-segment-id helper
- round 162 — §5.11.19 inter_segment_id(preSkip) syntax element
- round 161 — §5.11.7 intra_frame_mode_info() prefix dispatcher
- round 160 — §5.11.8 intra_segment_id() syntax element
- round 159 — §5.11.9 read_segment_id() syntax element
- round 158 — §5.11.20 read_is_inter() syntax element
- round 157 — §5.11.56 read_cdef() + §5.11.55 clear_cdef()
- round 156 — §5.11.13 read_delta_lf() syntax element
- round 155 — §5.11.12 read_delta_qindex() syntax element
- round 154 — §5.11.10 read_skip_mode() syntax element
- round 152 — §5.11.11 read_skip() syntax element
- round 151 — §5.11.4 decode_partition() recursive walker
- round 150 — §9.3 Partition_Subsize table + §3 BLOCK_* enum staging
- round 149 — §5.11.49 palette_tokens caller-side arg derivation
- round 148 — §9.3 block-size conversion tables
- round 147 — §5.11.49 palette_tokens per-plane diagonal walker
- round 146 — §5.11.50 get_palette_color_context derivation
- round 145 — §8.3.2 split_or_horz / split_or_vert cdf derivation
- round 144 — §9.4 wedge-index default-CDF
- round 143 — §9.4 inter-intra default-CDF group
- round 142 — §5.11.40 compute_tx_type() derivation
- r141 follow-up — re-export §8.3.2 helpers + add §3 constants to crate-root glob + lib.rs round-note + sync test count
- §8.3.2 get_coeff_base_ctx / get_br_ctx neighbour derivation (r141)
- round 140 — §9.4 Default_Coeff_Br_Cdf + §8.3.1 init_coeff_cdfs / §8.3.2 selection (coeff_br sub-group)
- Round 139 — coefficient `coeff_base` sub-group default CDF + selectors
- round 138 — §9.4 Default_Coeff_Base_Eob_Cdf + §8.3.1 init_coeff_cdfs / §8.3.2 selection (coeff_base_eob sub-group)
- round 137 — intra-frame transform-type default CDF group
- round 136 — §9.4 default-CDF + §8.3.1 init_coeff_cdfs / §8.3.2 selection (coefficient-token entry sub-group)
- round 135 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (angle-delta subset)
- round 134 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (inter-frame intra-mode subset)
- round 24 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (compound-prediction subset)
- round 23 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (motion-mode subset)
- round 22 — §9.4 default-CDF + §8.3.1 / §8.3.2 selection (inter-frame interpolation-filter subset)
- round 21 — §9.4 default CDFs + §8.3.1 / §8.3.2 selection (inter-frame transform-type subset)
- round 20 — transform-size §9.4 default CDFs + §8.3.2 selection
- round 19 follow-up — rename FILTER_INTRA_MODES to spec name INTRA_FILTER_MODES
- round 19 — §9.4 default CDFs + §8.3.1/§8.3.2 selection (palette/filter-intra/CFL subset)
- round 18 — §9.4 default CDFs + §8.3.1/§8.3.2 selection (inter-mode/ref-frame subset)
- Round 17: §9.4 default CDF tables + §8.3.2 selection (motion-vector
- Round 16: §9.4 default CDF tables + §8.3.1/§8.3.2 selection (intra-mode/partition subset)
- round 15 — §8.2 symbol (arithmetic / msac) decoder
- round 14 — inter-frame uncompressed_header() path (set_frame_refs / frame_size_with_refs / ref_frame_idx)
- round 13 — §5.9.24 global_motion_params + §5.9.30 film_grain_params
- round 12 — read_tx_mode() (§5.9.21) wired into streaming parser
- round 11 — wire lr_params() (§5.9.20) into the streaming parser
- round 10 — wire cdef_params() (§5.9.19) into the streaming parser
- round 9 — wire loop_filter_params() (§5.9.11) + CodedLossless derivation
- wire §5.9.17 delta_q_params + §5.9.18 delta_lf_params into streaming parse (round 8)
- frame header r7: wire §5.9.14 segmentation_params (+ §5.9.12 quantization_params) into streaming parse
- round 6 — §5.9.3 allow_intrabc + §5.9.15 tile_info wired into the streaming parse_frame_header walk
- round 5 — §5.9.10 / §5.9.11 / §5.9.12 uncompressed-header tail sub-syntaxes
- round 4 — §5.9.5–§5.9.9 frame-size sub-syntax block
- round 3 — §5.9.2 uncompressed_header prefix parse
- embed sequence-header fixture corpus instead of reading docs/
- round 2: sequence_header_obu parse (§5.5)
- round 1: OBU bytestream walker (§5.3 / §4.10.5)
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **Round 231 — UV_CFL_PRED (§7.11.5.3 chroma-from-luma) end-to-end.**
  The chroma intra picker in `encoder::pixel_driver::encode_intra_frame_yuv`
  gains a 14th candidate alongside the 13 §6.10.x modes that landed in
  r229: §7.11.5.3 chroma-from-luma over a compact (αU, αV) sign/magnitude
  search grid (`{±1, ±2, ±4}` per channel, single-channel arms
  included, excluding the §6.10.36 forbidden `(0, 0)` joint sign).
  When CFL wins on combined U+V SSD the picker commits the (αU, αV)
  pair as signed §5.11.45 `CflAlpha{U,V}` values; new fields
  `cfl_alpha_u` / `cfl_alpha_v` on `EncodeBlock` thread them into
  `write_encode_block_leaf`, which emits §5.11.45 via the new
  `encoder::block_mode_info::write_cfl_alphas` writer (joint
  `cfl_alpha_signs` over `Default_Cfl_Sign_Cdf` + per-channel
  `cfl_alpha_u` / `cfl_alpha_v` magnitude over
  `Default_Cfl_Alpha_Cdf[ ctx ]`).

  The decoder (`decoder::pixel_driver::decode_block_leaf`) now
  accepts `uv_mode == 13`, reads §5.11.45 via a new
  `read_cfl_alphas` helper, and dispatches both chroma planes
  through `cfl_predict_4x4_for_plane` (DC_PRED base +
  `Round2Signed(α * (L - lumaAvg), 6)` clipped to byte) using a
  §7.11.5.3 subsampled-luma window the encoder and decoder compute
  identically against the running reconstructed luma. The lossless
  WHT chain keeps the full encode → decode → pixel-equality
  contract bit-exact on the new path.

  +14 lib unit tests (1475 → 1489): 7 `write_cfl_alphas` round-trips
  (pos/pos, neg/neg, zero-U-pos-V, neg-U-zero-V, max magnitudes,
  both-zero rejection, out-of-range magnitude rejection) + 2
  `round2_signed` arms + 1 subsampled-luma flat-luma sanity + 3
  `cfl_predict_4x4_for_plane` (α=0 returns dc_pred, positive-α
  tracks luma excess, clipping at byte bounds) + 1 candidate-set
  no-zero-pair guard. +3 integration tests (23 → 26): CFL roundtrip
  bit-exact on a luma-correlated chroma input, encoder picks CFL on
  the same input, committed uv_mode stays in `0..=13`.

* **Round 230 — Pixel-driver frame-size generalisation: 32×32 and 64×64
  4:2:0 YUV encode → decode_av1 roundtrip.** The encoder gains a
  Vec-backed dynamic-extent driver
  (`encoder::pixel_driver_dyn::encode_intra_frame_yuv_dyn`,
  `Yuv420Frame { width, height, y, u, v }`,
  `EncodedFrameDyn`) that accepts any `(width, height)` with both
  dimensions ∈ {8, 16, 24, 32, 40, 48, 56, 64} (multiples of 8 ≤ 64
  per the §5.11.5 4:2:0 chroma cell constraint) and synthesises its
  own SequenceHeader + FrameHeader from the requested dimensions via
  `build_intra_only_yuv420_8bit_seq` / `build_intra_only_yuv420_8bit_fh`
  — callers no longer need to supply a fixture descriptor. The
  recursive `build_partition_tree` helper walks the smallest covering
  power-of-two super-block (one of BLOCK_16X16 / BLOCK_32X32 /
  BLOCK_64X64) and emits `EncodeNode::dummy_oob()` for fully-out-of-
  frame quadrants — the §5.11.4 driver's line-1 `r >= MiRows ||
  c >= MiCols` early return swallows them.

  The decoder side gains a matching dyn driver
  (`decoder::pixel_driver_dyn::decode_frame_dyn`) and a new
  `Frame::Yuv420Dyn { width, height, y, u, v }` variant; the public
  `decode_av1` entry dispatches automatically based on the parsed
  frame extent (the existing fixed-size 16×16 path continues to surface
  `Frame::Yuv420_16x16` byte-for-byte unchanged). The `Frame` enum is
  now `#[non_exhaustive]` so future extents (monochrome / 4:2:2 / 10-bit)
  can land without a SemVer bump.

  Composition of primitives is unchanged from r223+r229: the same
  13-mode intra picker on luma (`pick_best_intra_mode_4x4`) and chroma
  (`pick_best_intra_mode_4x4_chroma_joint`) drives a per-leaf forward-
  WHT + forward-quantize + dequantize + inverse-WHT chain. The lossless
  `base_q_idx = 0` arm makes every leaf a bit-exact reversible step, so
  the full encode → decode_av1 → pixel-equality contract holds for any
  allowed dynamic extent — verified end-to-end on flat-grey + pseudo-
  random + horizontal-gradient inputs at 16×16, 32×32, and 64×64.

  Tests: 13 new lib tests in `encoder::pixel_driver_dyn` (dispatch
  coverage + size validation + flat/pseudo-random encode-only
  roundtrip at 16×16/32×32/64×64), 5 new lib tests in
  `decoder::pixel_driver_dyn` (encode → decode_av1 → pixel-equality at
  16×16 / 32×32 / 64×64 flat + 32×32 + 64×64 pseudo-random), 5 new
  integration tests in `tests/encode_decode_pixel_roundtrip.rs`
  (encode → decode_av1 → pixel-equality at 32×32 flat + 32×32
  pseudo-random + 32×32 horizontal gradient + 64×64 flat + 64×64
  pseudo-random). Total +23 tests.

  Out of scope: frames larger than 64×64 (multi-super-block tiling),
  rectangular partitions, larger TX sizes, `base_q_idx > 0`, monochrome
  / 4:2:2 / 4:4:4 sampling, inter mode_info, and a bit-cost-aware RD
  picker — all follow-ups for subsequent arcs.

* **Round 229 — Encoder picks from all 13 §6.10.x intra prediction
  modes per chroma 4×4 block (mirror of r228's luma picker).** The
  pixel driver (`encoder::pixel_driver::encode_intra_frame_yuv`) now
  runs a combined-U+V-SSD picker over each of the 13 §6.10.x intra
  modes (the §3 INTRA_MODES set — `DC_PRED` through `PAETH_PRED`) on
  every chroma 4×4 block and commits the lowest-SSD pick via
  `write_intra_uv_mode` (r211). Per §5.11.22 one `intra_uv_mode()`
  symbol governs both U and V planes, so the picker minimises joint
  residual energy rather than picking U and V independently. Replaces
  the r223 hardcoded `uv_mode = DC_PRED`. The decoder gains a matching
  §7.11.2.{2..6} dispatch in `decoder::pixel_driver::decode_block_leaf`
  that routes the decoded `uv_mode` through the matching
  sample-generation kernel against a `CHROMA_HEIGHT × CHROMA_WIDTH`
  surface (`predict_intra_chroma_for_mode_4x4` +
  `derive_intra_neighbours_4x4_chroma`).

  Lossless `base_q_idx = 0` arm is preserved: every chroma picker
  selection still routes through the bit-exact forward WHT → forward
  quantize → dequantize → inverse WHT chain so the end-to-end YUV
  roundtrip stays bit-exact regardless of which chroma mode the
  picker selects. `EncodedFrameYuv` gains a
  `committed_uv_modes: Vec<u8>` field surfacing the per-chroma-cell
  picker output (4 entries under the 4:2:0 BLOCK_4X4 walk).
  UV_CFL_PRED (ordinal 13) stays out of scope this arc — the §7.11.5.3
  CFL αU/αV linear predictor path isn't wired yet; the decoder
  hard-rejects `uv_mode >= 13`. New test coverage: 5 lib unit tests +
  7 integration tests (horizontal-chroma-gradient ⇒ picker selects
  non-DC, vertical chroma gradient ⇒ non-DC, flat-128 ⇒ DC_PRED
  tie-break, in-range bounds, end-to-end roundtrip bit-exact on
  chroma gradient / pseudo-random YUV / 0-255 extremes).

  The picker is residual-SSD-only — it does **not** model the
  §5.11.22 `uv_mode` rate cost (the §8.3.2 chroma-CDF row
  probabilities) so non-DC selections can occasionally cost more
  symbol bits than they save in residual energy. Full RD with
  bit-cost weighting is a follow-up arc (same as the luma side).

* **Round 228 — Encoder picks from all 13 §6.10.x Y intra prediction
  modes per BLOCK_4X4 leaf.** The pixel driver
  (`encoder::pixel_driver::encode_intra_frame_y` /
  `encode_intra_frame_yuv`) now runs a residual-SSD picker over each of
  the 13 §6.10.x intra Y modes — `DC_PRED`, `V_PRED`, `H_PRED`,
  `D45_PRED`, `D135_PRED`, `D113_PRED`, `D157_PRED`, `D203_PRED`,
  `D67_PRED`, `SMOOTH_PRED`, `SMOOTH_V_PRED`, `SMOOTH_H_PRED`,
  `PAETH_PRED` — and commits the lowest-SSD pick via `write_y_mode`
  (r211) + `forward_transform_2d` (r227) + `forward_quantize` (r220) +
  `write_coefficients` (r215). Replaces the r221/r223 hardcoded
  `DC_PRED` luma path. The decoder gains a matching dispatch in
  `decoder::pixel_driver::decode_block_leaf` that routes the decoded
  `y_mode` to the matching §7.11.2.{2..6} sample-generation leaf
  (`predict_intra_dc_pred` / `predict_intra_v_pred` /
  `predict_intra_h_pred` / `predict_intra_d_mode` /
  `predict_intra_smooth*_pred` / `predict_intra_paeth_pred`).

  All `AboveRow[]`, `LeftCol[]`, and `AboveRow[-1]` corner samples for
  the picker are derived inline from the running reconstructed luma
  plane via a §7.11.2.1 prologue restricted to the BLOCK_4X4 cell
  extent. The head-extended `above_ext` / `left_ext` buffers (offset 0
  = index -2, offset 1 = index -1, offset 2+k = index k) match the
  `predict_intra_directional` calling convention so the six
  non-degenerate D-modes consume the same neighbour arrays as the
  V / H / SMOOTH / PAETH leaves. Lossless `base_q_idx = 0` arm is
  preserved: every picker selection still routes through the bit-exact
  forward WHT → forward quantize → dequantize → inverse WHT chain so
  the existing end-to-end YUV roundtrip is unchanged. New test
  coverage: 7 lib unit tests + 6 integration tests (incl.
  horizontal-gradient input ⇒ picker selects `V_PRED`, vertical
  gradient ⇒ `H_PRED`, flat-128 ⇒ picker keeps `DC_PRED` tie-break).
  `EncodedFrame` / `EncodedFrameYuv` gain a `committed_y_modes:
  Vec<u8>` field surfacing the per-leaf picker output.

  The picker is residual-SSD-only — it does **not** model the §5.11.22
  `y_mode` rate cost (the §8.3.2 `TileYModeCdf[ Size_Group ]` row
  probabilities) so non-DC selections can occasionally cost more
  symbol bits than they save in residual energy. Full RD with
  bit-cost weighting is a follow-up arc. Chroma still picks
  `uv_mode = DC_PRED`.

* **Round 227 — §7.13.3-equivalent forward 2D transform dispatcher.**
  Lands `encoder::forward_transform_2d(input: &[i64], tx_size: usize,
  plane_tx_type: usize, lossless: bool) -> Vec<i64>` — the encoder
  counterpart of the §7.13.3 inverse-transform dispatcher in
  `transform::inverse_transform_2d`. Composes the r219/r222/r225/r226
  per-axis forward kernels (DCT / ADST / FLIPADST / IDTX / WHT) into
  the §7.13.3 **column-then-row** pipeline (transpose of the decoder's
  row-then-column composition). Per `(tx_size, plane_tx_type)` the
  dispatcher selects the matching row + column kernel via the same
  16-arm decision tree the inverse uses (DCT for `{DCT_DCT, ADST_DCT,
  FLIPADST_DCT, H_DCT}`, ADST for the eight ADST-family arms, identity
  for `{IDTX, V_DCT, V_ADST, V_FLIPADST}` on row pass — mirrored on
  the column pass per §7.13.3 row/col tables on p.306-307). FLIPADST
  family flips the spatial residual along the appropriate axis
  (vertical for `FLIPADST_*` column kernels, horizontal for `*_FLIPADST`
  row kernels) before the plain ADST kernel runs — encoder mirror of
  the decoder's §7.12.3 step-3 post-inverse frame-buffer flip.

  **Square-only scope.** The five square sizes — `TX_4X4`, `TX_8X8`,
  `TX_16X16`, `TX_32X32`, `TX_64X64` — are landed. ADST kernel is
  capped at `TX_16X16` because `inverse_adst` itself routes
  `n in 2..=4`; IDTX is capped at `TX_32X32` because
  `inverse_identity` routes `n in 2..=5`. The §6.10.19 `tx_type`
  derivation forces `DCT_DCT` for the unreachable combinations, so
  the kernel-range panics in the dispatcher document the spec
  invariant rather than fence off encoder-driver work. Rectangular
  sizes (`TX_4X8`, `TX_8X4`, …, `TX_64X16`) are a subsequent arc.

  **Lossless arm.** `lossless = true` (with `tx_size = TX_4X4`)
  routes through r222's bit-exact `forward_wht_4x4`. The WHT integer
  butterfly + pre-shift envelope preserves every input bit:
  `inverse_transform_2d(forward_transform_2d(x), TX_4X4, _, true) ==
  x` exactly for any integer residual.

  **Lossy arm.** Round-trip recovers the input scaled by the per-cell
  factor `N^2 / 2^(rowShift + colShift)` (analytic = empirical):
  `{1/4, 1/2, 1, 4, 4}` for `TX_{4..64}X{4..64}`. The encoder does
  not apply matching `<< colShift` / `<< rowShift` pre-scales because
  doing so pushes the inverse pipeline's intermediate values past the
  decoder's between-stage `Clip3` clamp (16 bits at BD = 8), breaking
  the round-trip catastrophically. A real encoder driver pairing
  this dispatcher with `forward_quantize` recovers bit-correct
  coefficients by dividing through the per-stage gain in the
  quantizer step.

  +26 lib tests (1419 → 1445): bit-exact `TX_4X4` lossless WHT
  round-trip across pseudo-random and extreme-value inputs; lossy
  round-trip per kernel family across square sizes with per-tx-size
  scale calibration; FLIPADST family verified against the flipped
  input (since `inverse_transform_2d` doesn't itself flip — that
  runs externally at §7.12.3 step 3); zero-input → zero-output across
  20 `(tx_size, tx_type)` combinations; panic guards on rectangular
  `tx_size` and lossless-with-non-TX_4X4.

  Subsequent arcs: rectangular block sizes for `forward_transform_2d`;
  pixel-driver wiring (chroma path, intra angle / palette);
  §5.11.18 inter-arm `mode_info()` dispatcher; the inter-frame
  `frame_size_with_refs()` + `read_global_param` writers.

* **Round 226 — forward ADST / FLIPADST / IDTX primitives.** Lands the
  forward 1D and 2D non-DCT transform kernels: forward ADST for sizes
  `4 / 8 / 16` (the spec's full ADST coverage per §7.13.2.6/7/8),
  forward FLIPADST for the same sizes, and forward IDTX (identity)
  for sizes `4 / 8 / 16 / 32`. New API: `encoder::forward_adst_4` /
  `_8` / `_16` (1D) + `forward_adst_4x4` / `_8x8` / `_16x16` (2D);
  the matching `encoder::forward_flipadst_*` and `forward_idtx_*`
  entry points.

  Derivation generalises r225's matrix-cache recipe: the inverse
  ADST is a fixed integer linear map for each `n`, the forward is
  the algebraic transpose built once per size by probing the inverse
  on unit-coefficient basis inputs and caching the response matrix
  in a `OnceLock`. FLIPADST shares the same butterfly kernel as ADST
  per §7.13.3 — the flip is purely a coordinate transform on the
  frame-buffer write — so `forward_flipadst_*` reverses the residual
  before the forward ADST kernel and reverses the output coefficient
  cells back. IDTX is a diagonal scalar map: `forward_idtx_4` is
  `Round2(5793 * in, 12)`, `_8` is `2 * in` exactly, `_16` is
  `Round2(11586 * in, 12)`, `_32` is `4 * in` exactly.

  +44 lib tests (1375 → 1419): zero-input identities for every 1D
  and 2D primitive; per-cell scalar values matched against the spec
  multipliers for IDTX; `inverse(forward(x))` roundtrips on flat-DC
  and LCG-pseudo-random inputs for ADST; the FLIPADST convention
  proven via the `forward_flipadst_N(x) ==
  reverse(forward_adst_N(reverse(x)))` lockstep + the 2D
  `double_reverse` equivalent.

* **Round 225 — forward DCT for sizes 8 / 16 / 32 / 64.** Extends the
  r219 forward 4×4 DCT primitive to every spec-defined square DCT
  block size. New API: 1D primitives `encoder::forward_dct_8` /
  `_16` / `_32` / `_64`, and the matching square 2D primitives
  `encoder::forward_dct_8x8` / `_16x16` / `_32x32` / `_64x64`.

  The derivation generalises r219's matrix-transpose argument: for
  each size `N = 2^n` (`n in 2..=6`) we materialise the response
  matrix `M_inv[n]` by walking `crate::transform::inverse_dct` on
  each of the `N` unit-coefficient basis inputs `[4096, 0, …]`,
  `[0, 4096, …]`, … and cache it in a `OnceLock`. The forward then
  computes `forward(x)[k] = Round2(sum_i M_inv[n][i + N * k] * x[i],
  12)` — the inner product of `x` with the `k`-th column of `M_inv`,
  which is `M_inv^T @ x / 4096`. Same `Round2(_, 12)` rounding
  shape, same `O(N^2)` per 1D pass, same row-then-column composition
  for the 2D primitives.

  +22 lib tests (1353 → 1375): zero-input identities, DC-bin energy
  concentration with bounded off-DC noise, `inverse(forward(x))`
  roundtrips on flat-DC and pseudo-random inputs (the well-posed
  invariant: integer rounding accumulates across the multi-stage
  butterfly so `M_inv^T · M_inv` is only **approximately** diagonal
  for n >= 3, but `inverse(forward(x))` recovers `x` scaled by
  `≈ N/2` within a bounded noise floor).

  Unblocks: encoder rate-distortion search across all spec DCT
  block sizes (previously the encoder was stuck at TX_4X4 only).
  Subsequent arcs: forward ADST / FLIPADST / IDTX, the rectangular
  block sizes (TX_4X8 / TX_8X16 / etc.), and the §7.13.3-equivalent
  forward 2D dispatcher with row-/col-shift envelope.

* **Round 224 — `decode_av1` public entry + first encode→decode pixel
  roundtrip.** The arc that closes the loop: the existing decoder
  modules (`obu::ObuIter` / `parse_sequence_header` /
  `parse_frame_header` / `parse_tile_group_obu_body` / the §5.11.39
  `PartitionWalker::coefficients` reader / §7.12.3 `dequantize_step1` /
  §7.13 `inverse_transform_2d` lossless WHT arm / §7.11.2.5
  `predict_intra_dc_pred`) are now composed into a single public
  pixel-out entry, `decode_av1(&[u8]) -> Result<Vec<Frame>>`, that is
  the inverse of `encoder::encode_intra_frame_yuv` on the lossless
  arc-17 4:2:0 YUV path.

  Supporting additions:

  * **`encoder::ivf::IvfReader`** — the demuxer counterpart of
    `IvfWriter`. Parses the 32-byte v0 file header
    (`parse_file_header` exposed standalone) and iterates per-frame
    `(pts, payload)` records via `read_next_frame` /
    `read_all`. Public `IvfFileHeader` / `IvfFrame` /
    `IvfReadError` aggregates.
  * **`decoder` module** with `decoder::pixel_driver` containing the
    arc-18 dispatch: `decode_av1`, `decode_temporal_unit`,
    `TemporalUnitResult`, and the `Frame::Yuv420_16x16` output
    enum. The dispatch walks the §5.11.4 BLOCK_16X16 →
    BLOCK_8X8 → BLOCK_4X4 SPLIT tree (mirroring the encoder's
    `write_partition_tree`), reads the §5.11.5 per-leaf scalars
    (`read_skip`, `y_mode` over the Size_Group ctx, `uv_mode`
    gated on §5.11.5 `HasChroma`), and feeds the per-plane
    `coefficients()` output through the lossless WHT inverse
    chain plus DC_PRED reconstruction.
  * **`decode_av1` in `lib.rs`** is now wired (was a stub returning
    `Error::NotImplemented`). Returns `Vec<decoder::Frame>`.

  Arc-18 hard scope (matches the encoder pixel driver): 16×16
  frame, 4:2:0 YUV (`monochrome = false`, `subsampling_x =
  subsampling_y = 1`), `base_q_idx = 0` lossless WHT arm,
  intra-only with DC_PRED only, single tile, no in-loop
  post-processing. Streams outside the scope return
  `Error::PartitionWalkOutOfRange`.

  **Milestone**: encoder → `decode_av1` → pixel equality is now
  exercised end-to-end via the public API. Five integration tests
  (`tests/encode_decode_pixel_roundtrip.rs`) cover flat-grey, a
  horizontal-chroma-gradient, pseudo-random YUV (LCG), 0/255 extremes,
  and the intentionally-non-conformant Y-only encode-path rejection.

  13 new lib tests (1340 → 1353): 6 IVF reader tests
  (`parse_file_header_round_trip`,
  `parse_file_header_rejects_short_buffer`,
  `parse_file_header_rejects_bad_magic`,
  `ivf_reader_round_trips_writer_output`,
  `ivf_reader_empty_buffer_returns_none`,
  `ivf_reader_truncated_frame_header_errors`,
  `ivf_reader_truncated_payload_errors`); 6 decoder pixel-driver
  tests (`decode_av1_recovers_flat_grey_yuv`,
  `decode_av1_recovers_non_flat_yuv`,
  `decode_av1_y_only_encoder_path_emits_non_conformant_stream`,
  `decode_av1_rejects_short_buffer`,
  `temporal_unit_result_carries_sh_when_present`,
  `decode_zeroed_yuv420_16x16_factory_zeros_each_plane`).

  Deferred to subsequent arcs (subset of the encoder's deferred list):
  forward DCT for sizes 8 / 16 / 32 / 64; forward ADST / FLIPADST /
  IDTX kernels; intra mode picker beyond DC_PRED; §5.11.18 inter-arm
  `mode_info()` dispatcher; intra angle / palette encode; multi-tile
  frames; in-loop filter (LF / CDEF / superres / LR / film grain)
  exercise (they are spec-correct no-ops on the arc-18 frame's
  parameter set so the decoder skips them, but the chain itself is
  exercised by next-arc test fixtures with non-zero filter levels).

* **Round 223 — pixel-driver chroma (4:2:0 YUV) path.** New
  `encoder::pixel_driver::encode_intra_frame_yuv(input, seq, fh) ->
  EncodedFrameYuv` accepts a 16×16 luma + two 8×8 chroma planes
  (`Yuv420Frame16x16`) and returns IVF bytes plus all three
  reconstructed planes. The chroma walk mirrors the luma side:
  §7.11.2.5 DC_PRED (built from the running reconstructed chroma plane),
  forward WHT (lossless arm), forward quantize (plane = 1 / 2), inverse
  chain for the reconstruction. Per §5.11.5 `HasChroma` at 4:2:0 /
  BLOCK_4X4 with `bw4 == bh4 == 1`, chroma coefficients fire only on
  luma cells whose `MiRow` and `MiCol` are both odd ⇒ exactly four
  chroma 4×4 blocks per 16×16 frame, hung off the SE corner of each
  8×8 luma quadrant.

  The same `(SequenceHeader, FrameHeader)` pair feeds both Y-only and
  YUV entry points — the `tiny-i-only-16x16-prof0` fixture is already
  `monochrome = false` / `subsampling_x = subsampling_y = 1`, so the
  difference between the two paths is entirely in the per-block
  coefficient stream (and the leaf's `uv_mode` slot). End-to-end the
  encoder reconstructs **arbitrary 4:2:0 YUV inputs pixel-for-pixel on
  every plane** at `base_q_idx = 0` (lossless WHT chain on Y / U / V).

  New public surface: `Yuv420Frame16x16`, `EncodedFrameYuv`,
  `encode_intra_frame_yuv`, plus chroma constants `CHROMA_WIDTH`,
  `CHROMA_HEIGHT`, `CHROMA_CELLS_WIDE`, `CHROMA_CELLS_HIGH`. Re-exported
  from `oxideav_av1::encoder`.

  15 new tests (1329 → 1340 lib, 6 → 10 integration): chroma-cell
  dispatch (`has_chroma_cells_are_se_corners_of_each_8x8_quadrant`,
  `cell_to_chroma_block_maps_se_corners_to_chroma_grid`); per-plane
  flat-128 zero-quant + bit-exact reconstruction; non-flat chroma
  lossless roundtrip (flat-64 U + flat-192 V); chroma horizontal
  gradient; pseudo-random YUV roundtrip; YUV tile-group payload
  strictly larger than Y-only for the same luma input (proof the
  chroma syntax reaches the bitstream); YUV-with-flat-chroma luma walk
  equals Y-only-driver luma walk; SH / FH OBU reparse equality on the
  YUV path.

* **Round 222 — forward Walsh-Hadamard transform (lossless milestone).**
  Encoder counterpart of the §7.13.2.10 inverse WHT used by the §7.13.3
  `Lossless` arm. New module `encoder::forward_wht` exposes
  `forward_wht4(t, shift)` (1D length-4 with caller-supplied pre-shift)
  and `forward_wht_4x4(input) -> [i64; 16]` (2D `TX_4X4` with the
  §7.13.3 lossless shift envelope: column pass `shift = 0`, row pass
  `shift = 2`).

  Derived clean-room by inverting the §7.13.2.10 inverse body
  algebraically: the four output cells `(a_out, b_out, c_out, d_out)`
  determine `A = a_out + b_out` and `D = d_out - c_out`, from which the
  pre-shift values `(a0, b0, c0, d0)` are recovered exactly; the
  `(A - D) >> 1` floor-halving is shared between forward and inverse
  so the round-trip is bit-exact regardless of `(A - D)` parity. The
  forward output is multiplied by `1 << shift` to cancel the inverse's
  `>> shift` pre-scaling, keeping the row-pass round-trip lossless.

  `encoder::pixel_driver::encode_intra_frame_y` now routes through the
  forward WHT when the §5.9.2 `CodedLossless` predicate fires
  (`base_q_idx == 0 && DeltaQ?? all zero`, satisfied by the tiny
  fixture's `QuantizerParams::neutral(0, 8)`). Because every WHT
  coefficient is divisible by the lossless `q2 = 4` (the row pass
  multiplies by `1 << 2`), the forward-quantize round-trip is exact;
  the encoder-internal reconstruction (decoder's `dequantize_step1` +
  `inverse_transform_2d` with `lossless = true`) recovers any input
  pixel-for-pixel. **Bit-exact pixel roundtrip on arbitrary input —
  the milestone unlock.**

  13 new lib tests (1316 → 1329): 10 in `encoder::forward_wht`
  (shift = 0 / shift = 2 1D sweeps over the `[-256, 255]` range across
  every slot; mixed-slot round-trip including odd-parity cases; 2D
  zero/flat-DC/unit-impulse/LCG-sweep/linearity round-trips against
  `inverse_transform_2d(lossless = true)`; short-buffer caller-bug
  panic) and 3 in `encoder::pixel_driver` (non-uniform pattern,
  horizontal gradient, pseudo-random LCG — all now bit-exact through
  the lossless WHT path). The pre-existing flat-64 reconstruction
  test is upgraded from a generous `|delta| <= 80` envelope to a
  strict bit-exact equality.

* **Round 221 — pixel-space encoder driver (milestone arc).** First
  end-to-end pixel-in / bytes-out entry point. New module
  `encoder::pixel_driver` exposes
  `encode_intra_frame_y(luma_in, seq, fh) -> EncodedFrame` composing
  r219 `forward_dct_4x4`, r220 `forward_quantize`, r217
  `write_partition_tree`, r210 `write_tile_group_obu`, r208 temporal
  unit aggregation, and r206 IVF — into one call that converts a
  16×16 monochrome luma plane into a complete IVF byte buffer.

  The driver builds its own §7.11.2.5 DC_PRED prediction from the
  running reconstructed neighbour buffer in §5.11.4 dispatch order
  (NW → NE → SW → SE recursion across the BLOCK_16X16 / BLOCK_8X8 /
  BLOCK_4X4 two-level split), then per-leaf computes the residual,
  applies the forward DCT, forward-quantizes at `base_q_idx = 0`, and
  immediately runs the decoder's `dequantize_step1` +
  `inverse_transform_2d` chain to reconstruct the leaf's pixels for
  the next leaf's prediction. Output `EncodedFrame` bundles the IVF
  bytes, the temporal-unit body, the per-leaf committed `Quant[]`
  arrays, and the encoder-internal reconstructed luma plane.

  **Bit-exact pixel roundtrip on flat-128 input.** For a 16×16
  mid-grey luma plane the DC_PRED prediction equals the input at
  every leaf ⇒ residual = 0 ⇒ all coefficients = 0 ⇒ all
  `Quant[]` = 0 ⇒ recovered residual = 0 ⇒ recovered pixels = 128
  exactly. The first end-to-end pixel roundtrip the encoder
  achieves.

  Out of scope (next arc): the forward WHT primitive that unlocks
  the §7.13.3 `Lossless` path (for arbitrary-input pixel-perfect
  roundtrip at `q_index = 0`); chroma path; forward DCT for larger
  sizes / non-DCT row+column kernels.

  14 tests total (8 lib + 6 integration in
  `tests/encoder_pixel_driver_y_only.rs`): dispatch-order coverage
  + Z-curve shape verification, flat-128 zero-quant proof, flat-128
  bit-exact reconstruction, internal-helper vs driver agreement,
  IVF preamble + 4-OBU walk (TD + SH + FrameHeader OBU + TileGroup
  OBU), SH/FH reparse equality, tile-group OBU payload
  non-emptiness, lossy-but-self-consistent uniform-64 input round-
  trip.

* **Round 220 — forward quantization primitive.** Encoder counterpart
  of the §7.12.3 step-1 dequant loop (`cdf::dequantize_step1`). New
  function `encoder::forward_quantize::forward_quantize(coeffs,
  tx_size, plane, segment_id, plane_tx_type, seg_qm_level, quant) ->
  Vec<i32>` consumes a post-forward-transform coefficient buffer plus
  the per-frame `QuantizerParams` and per-block selectors and returns
  the dense `Tx_Width[tx] * Tx_Height[tx]` `Quant[]` array a §5.11.39
  `coefficients()` writer (round 215) consumes.

  Derivation: the spec body
  `dq2 = sign(dq) * (|dq| & 0xFFFFFF) / dqDenom` simplifies (under
  typical `|Quant * q2| < 2^24`) to
  `Dequant = sign(Quant) * (|Quant| * q2 / dqDenom)` with the divide
  truncating toward zero. Solving for `Quant` with round-half-away-
  from-zero gives `Quant = round((|coeff| * dqDenom) / q2)` carrying
  the sign of `coeff`. The `q2`, `dqDenom`, and `qm_active`
  derivations are reused from `cdf` so the encoder reads the same
  §7.12.2 lookups and the same §9.5.3 `Quantizer_Matrix[][][]` the
  decoder reads.

  Bit-exact roundtrip is verified through the decoder's existing
  `dequantize_step1`: for any caller-chosen `Quant[]` whose dequant
  stays in range, `forward_quantize(dequantize_step1(Quant)) == Quant`
  exactly. For arbitrary (off-lattice) coefficients the recovered
  level is within one quantization step of the analytic optimum.

  10 tests: lossless `q_index = 0` TX_4X4 (DC + AC slots), sign
  preservation, lossy half-away rounding behaviour at `q_index = 0`
  (ties round up in magnitude — matches the encoder's half-away-from-
  zero bias against the decoder's toward-zero truncation), higher
  `q_index = 16` aligned-coeff roundtrip, TX_8X8 lossless roundtrip
  (the 8×8 active region), all-zero-coeff identity across multiple
  `q_index` values, full `dequant → forward_quantize` roundtrip
  inversion over four distinct level patterns including magnitudes up
  to 4096, QM-active branch roundtrip (`using_qmatrix = true`,
  `seg_qm_level = 0`, `plane_tx_type = DCT_DCT`), and two caller-bug
  panic tests (short buffer + out-of-range `tx_size`). Test total
  1298 → 1308.

* **Round 219 — pixel-space encoder bootstrap: forward 4×4 DCT.** First
  primitive that takes pixel residuals as input (the rest of the
  encoder arc 1..12 consumed pre-decided `Quant[]` arrays). Two
  stateless functions in `encoder::forward_transform`:

  * `forward_dct_4(t, r)` — 1D DCT-II of length 4. Matrix transpose
    of the §7.13.2.3 inverse DCT-4 reproduced in
    [`crate::transform::inverse_dct`] (`n = 2` branch). Single
    §4.7.2 `Round2(_, 12)` per output coefficient. Coefficients
    derived by walking the inverse on the four 4-dimensional unit-
    coefficient inputs: `M^T = (1 / 4096) * [[2896, 2896, 2896,
    2896], [3784, 1567, -1567, -3784], [2896, -2896, -2896, 2896],
    [1567, -3784, 3784, -1567]]`.
  * `forward_dct_4x4(input) -> [i64; 16]` — 2D DCT-II for `TX_4X4`,
    row-then-column composition of `forward_dct_4`. No `Lossless`
    short-circuit, no rectangular scaling (square block), no row-
    /col-shift envelope (those belong to a §7.13.3-equivalent
    forward 2D dispatcher, the subsequent arc).

  Roundtrip verification: `M^T · M` is exactly diagonal (the basis is
  mutually orthogonal — off-diagonal entries are exact zeros) with
  diagonal `≈ 1.99988` on even rows and `≈ 1.99994` on odd rows
  because the AV1 cosine constants `2896`, `3784`, `1567` are
  integer-rounded approximations of the analytic cosines used by the
  continuous DCT-II. 8 tests: zero-input, DC-only, unit-vector
  matrix-transpose probe, all-basis-coefficient roundtrip (exact
  diagonal + exact zero off-diagonal), linearity at 4096-aligned
  inputs, 2D zero / DC / impulse / all-position bound coverage. Test
  total 1288 → 1298.

* **Round 218 — §5.11.36 transform_tree / tx_size writers.** The
  encoder counterpart of
  [`crate::cdf::PartitionWalker::read_block_tx_size`] and
  [`crate::cdf::PartitionWalker::read_var_tx_size`]. Two stateless
  writers exposed from `encoder::transform_tree`:

  * `write_block_tx_size(writer, cdfs, tx_size, sub_size, lossless,
    allow_select, tx_mode_select, ctx)` — emits the §5.11.15
    `tx_depth` symbol for the §5.11.16 `else` arm. Derives the
    spec's `tx_depth` value by walking `Split_Tx_Size` from
    `Max_Tx_Size_Rect[MiSize]` until reaching the caller's chosen
    `tx_size`. Skips the symbol entirely on the no-symbol arms
    (Lossless, `sub_size <= BLOCK_4X4`, `!allow_select`, or
    `!tx_mode_select`) and asserts the caller's `tx_size` matches
    the spec-forced value on each.
  * `write_var_tx_size(writer, cdfs, root)` — recursive driver
    emitting one `txfm_split S()` per non-terminal node of a
    caller-supplied `VarTxNode` tree. Mirrors the §5.11.17 recursion's
    spec-forced terminal at `txSz == TX_4X4 || depth == MAX_VARTX_DEPTH`
    (no symbol emitted at terminals), enforces the `Split_Tx_Size`
    quadrant-decomposition child-count invariant per
    `(h4 / stepH) * (w4 / stepW)`, and rejects illegal splits at the
    spec-forced terminal as caller bugs.

  New public types: `VarTxNode { tx_size, ctx, kind }` and
  `VarTxNodeKind { Leaf, Split(Vec<VarTxNode>) }`. 21 roundtrip /
  rejection tests (writer → `SymbolDecoder::read_symbol` against a
  parallel `TileCdfContext`).

* **Round 217 — §5.11.4 recursive dispatch driver.** The encoder
  counterpart of [`crate::cdf::PartitionWalker::decode_partition`]'s
  recursive walk: a complete intra-arm partition-tree walker that
  composes the r211–r216 per-block writers (`write_skip`,
  `write_intra_segment_id`, `write_y_mode`, `write_intra_uv_mode`,
  per-plane `write_coefficients`) with the r216 `write_partition`
  symbol writer.

  New public types in `encoder::partition_tree`:

  * `PlaneCoefficients { plane, is_inter, tx_size, tx_class,
    txb_skip_ctx, dc_sign_ctx, scan, quant }` — per-plane §5.11.39
    `coefficients()` input forwarded verbatim to `write_coefficients`.
  * `EncodeBlock { skip, segment_id, segment_pred, y_mode, uv_mode,
    coefficients }` — per-leaf intra-arm scalar bundle.
  * `EncodeNode::Leaf(EncodeBlock)` / `EncodeNode::Split([Box<EncodeNode>; 4])`
    — partition-tree node. `EncodeNode::dummy_oob()` sentinel for
    out-of-frame quadrants.
  * `PartitionTreeWriter::new(mi_rows, mi_cols, geometry,
    segmentation_enabled, last_active_seg_id, lossless, subsampling_x,
    subsampling_y)` — driver state; maintains the §6.10.4 `MiSizes[]`
    grid so the §8.3.2 `partition_ctx_for` lookup observes the same
    neighbour widths the decoder's parallel
    [`crate::cdf::PartitionWalker`] observes.
  * `write_partition_tree(writer, cdfs, state, node, r, c, b_size)` —
    recursive entry. Mirrors the §5.11.4 first conditional precisely:
    `r >= MiRows || c >= MiCols` early return; `bSize < BLOCK_8X8`
    forced-NONE leaf (no symbol); `!hasRows && !hasCols` forced-SPLIT
    corner (no symbol); otherwise emit the
    `partition` / `split_or_horz` / `split_or_vert` symbol via
    `write_partition`, then dispatch.

  Scope: PARTITION_NONE leaves + PARTITION_SPLIT splits only (the two
  recursive shapes the §5.11.4 walk uses for "pure" partition trees).
  The asymmetric partitions (`PARTITION_HORZ` / `PARTITION_VERT` /
  `*_A` / `*_B` / `*_4`) are out of scope for this arc — the
  `write_partition` symbol writer already supports their alphabet, but
  the multi-leaf mixed-size dispatch is a separate piece of work.

  Caller-bug surface (`Error::PartitionWalkOutOfRange`): out-of-range
  `b_size`, a `Leaf` at a forced-SPLIT corner, a `Split` at a
  forced-NONE node, plus any block-syntax / coefficient-write surface
  error propagated from the per-block writers.

  15 new lib tests (1252 → 1267) cover the recursive driver:
  constructor + state guards (zero-extent / oversize / inverted
  geometries), caller-bug shape rejections (Split at forced-NONE,
  Leaf at forced-SPLIT, out-of-range `b_size`, out-of-frame quadrant
  short-circuit), and five round-trip shapes through a mirror walker
  that replays the encoder's exact bit ordering (single PARTITION_NONE
  leaf, forced-NONE `bSize < BLOCK_8X8`, one-level SPLIT recovering
  four BLOCK_8X8 leaves, forced-SPLIT corner with three out-of-frame
  quadrants, two-level SPLIT on 16×16 frame recovering 7 mixed-size
  leaves) plus per-leaf block-syntax recovery (skip, V_PRED y_mode,
  all-zero §5.11.39 coefficients across Y/U/V planes). The mirror
  walker also asserts §8.3 CDF-adaptation lockstep on the encode +
  decode partition rows. With this arc the encoder is a true encoder
  end-to-end on the intra-only path: caller supplies a partition tree
  + per-block scalars + per-plane `Quant[]`, the encoder emits bytes
  the decoder walks back to identical scalars + identical leaf shape.

* **Round 216 — §5.11.4 partition decision-tree writer.** The encoder
  counterpart of [`crate::cdf::PartitionWalker::decode_partition`]'s
  symbol-read portion. Two predicate helpers + one writer:

  * `partition_none_only(b_size) -> bool` — the §5.11.4 first-conditional
    branch `bSize < BLOCK_8X8` ⇒ `PARTITION_NONE` with no symbol emit.
  * `partition_split_only(b_size, has_rows, has_cols) -> bool` — the
    §5.11.4 fall-through `!hasRows && !hasCols` corner-of-frame branch
    ⇒ `PARTITION_SPLIT` with no symbol emit.
  * `write_partition(writer, cdfs, partition, b_size, has_rows,
    has_cols, ctx) -> Result<(), Error>` — emits the right §8.2.6
    symbol(s) for the chosen partition: full N-value alphabet against
    the §8.3.2 `bsl`-selected partition CDF when `has_rows && has_cols`;
    binary `split_or_horz` (HORZ vs SPLIT) over the folded 2-symbol CDF
    from [`split_or_horz_cdf`] when `has_cols` only; binary
    `split_or_vert` (VERT vs SPLIT) over the folded 2-symbol CDF from
    [`split_or_vert_cdf`] when `has_rows` only; nothing in the no-symbol
    branches above. Stateless on purpose, same shape as the
    `block_mode_info` / `coefficients` writers — the encoder driver
    threads its own `PartitionWalker` for the §6.10.4 `MiSizes[]` grid
    and feeds the same (`has_rows`, `has_cols`, `ctx`) the decoder will
    derive.

  Symbol-mode caller bug surface (`Error::PartitionWalkOutOfRange`):
  out-of-range `b_size`, a partition ordinal that doesn't match the
  branch the caller landed in (e.g. `PARTITION_VERT` in the
  `split_or_horz` branch, `PARTITION_HORZ_4` on W128 which drops that
  ordinal from its row), or a CDF-row-too-short surface from the
  `split_or_*` folders.

  15 new lib tests (1237 → 1252) cover the four §5.11.4 branches +
  caller-bug rejects + a multi-block composition that round-trips
  through a parallel decoder using a running adapting `TileCdfContext`:
  predicate match against the spec; no-symbol short-circuit at
  BLOCK_4X4; no-symbol forced-split in the bottom-right corner; full
  10-ordinal alphabet round-trip on W16 × every `PARTITION_CONTEXTS`
  ctx; 8-ordinal alphabet round-trip on W128; W128 horz4/vert4 reject;
  `split_or_horz` 2-symbol round-trip; `split_or_vert` 2-symbol
  round-trip; both directional binary-CDF reject paths; out-of-range
  `b_size` reject; multi-block sequence with §8.3 adaptation lockstep.

  fmt + clippy --all-targets clean. Unblocks multi-block frames on the
  encode side: the bottom of the §5.11 stack is now expressible.

* **Round 215 — §5.11.39 `coefficients()` driver loop (end-to-end).**
  Composes every per-coefficient writer landed in rounds 212–214
  (`txb_skip` / `eob_pt` / `coeff_base_eob` / `coeff_base` /
  `coeff_br` / `dc_sign` / `golomb`) into a single per-transform-block
  encode that round-trips through the §5.11.39
  `PartitionWalker::coefficients` reader.

  New public API in `encoder::coefficients` (re-exported from
  `encoder`):
  * `write_coefficients(writer, cdfs, plane, is_inter, tx_size,
    tx_class, txb_skip_ctx, dc_sign_ctx, scan, quant_in) -> Result<u32,
    Error>` — the full §5.11.39 driver. Takes the caller's
    final-`Quant[]` array (signed, post-decoder values) and emits
    every symbol / literal the reader would consume. Returns the
    written `eob` (0 on the §5.11.39 line-13 all-zero short-circuit;
    otherwise the position past the highest non-zero coefficient).
    Mirrors the reader's caller-bug guards (`tx_size <
    TX_SIZES_ALL`, `tx_class <= TX_CLASS_VERT`, `plane <= 2`,
    `is_inter <= 1`, `txb_skip_ctx < TXB_SKIP_CONTEXTS`,
    `dc_sign_ctx < DC_SIGN_CONTEXTS`, `scan.len() >= seg_eob`,
    `quant_in.len() >= tx_w * tx_h`).

  ## Inverse strategy

  1. Compute `eob` from `quant_in`. `eob == 0` ⇒ `write_txb_skip(1)`
     and return; the §5.11.39 line-13 short-circuit is byte-exact.
  2. Maintain a running magnitude buffer that mirrors the decoder's
     `Quant[]` state as the reverse scan populates it. Both
     `get_coeff_base_*_ctx` and `get_br_ctx` walk this buffer; the
     encoder + decoder agree because both write into the same
     running state in the same order.
  3. **Reverse-scan** for `c = eob - 1 .. 0`: emit
     `coeff_base_eob(min(|q|-1, 2))` (at `c == eob - 1`) or
     `coeff_base(min(|q|, 3))` (otherwise); if the level hits 3,
     chain `coeff_br` with `sym = min(residue, 3)` per iter, capped
     at 4 iters per the §5.11.39 line-66 `COEFF_BASE_RANGE /
     (BR_CDF_SIZE - 1) = 4` bound. Stamp `running[pos] = min(|q|, 15)`.
  4. **Forward-scan** for `c = 0 .. eob`: emit `dc_sign(sign)` at
     `c == 0` (when `|q| > 0`) or `sign_bit` L(1) otherwise; if
     `|q| > 14` (= NUM_BASE_LEVELS + COEFF_BASE_RANGE), emit the
     §5.11.39 lines 84-93 golomb tail via `write_golomb(|q|)`.

  12 new end-to-end round-trip lib tests (1225 → 1237). Every test
  encodes `quant_in` through `write_coefficients`, decodes the bytes
  back through `PartitionWalker::coefficients`, and asserts the
  decoder's `Quant[]` matches `quant_in` cell-for-cell on TX_4X4 luma
  / chroma and TX_8X8 luma:
  - all-zero block (line-13 short-circuit; eob = 0, no further bits)
  - single DC = 1 (smallest non-trivial: eob = 1, one
    `coeff_base_eob`, one `dc_sign`, no chain / no golomb / no
    sign_bit)
  - two coefficients with mixed signs (DC = -1, AC = +2 ⇒ exercises
    `dc_sign` and `sign_bit`)
  - level 3 (chain-entry + immediate-terminate; one `coeff_br = 0`)
  - level 8 (mid-chain terminate; two `coeff_br` iters of 3 + 2)
  - level 14 (boundary; four `coeff_br` iters of 3 + 3 + 3 + 2; no
    golomb)
  - level 15 (chain saturates ⇒ golomb fires with x = 1, length = 1;
    minimal golomb)
  - AC magnitude 30 with `sign_bit` (no `dc_sign` because DC = 0)
  - TX_8X8 dense small pattern (6 non-zero positions, mixed signs)
  - chroma ptype = 1 axis with off-origin `txb_skip_ctx = 5` /
    `dc_sign_ctx = 1`
  - large golomb (200; x = 186, length = 8) composed with a normal
    DC + AC
  - exhaustive caller-bug guard sweep (8 out-of-range arguments)

  fmt + clippy --all-targets clean.

  Next arc: §5.11.4 partition decision-tree writer; §5.11.18
  inter-arm `mode_info()` dispatcher; transform_tree / tx_size
  encode; intra angle / palette encode.

* **Round 214 — §5.11.39 golomb magnitude-tail writer.** Extends
  `encoder::coefficients` with the per-magnitude `golomb_length_bit`
  (do-while unary prefix) + `golomb_data_bit` (MSB-first L(1)
  payload) tail for coefficient magnitudes above
  `NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`. Single
  `write_golomb(value)` writer; one composite call emits both halves.
  `GOLOMB_MAX_LENGTH = 20` constant captures the §6.10.34 conformance
  bound (any `value` that would derive `length > 20` is rejected as a
  caller bug — would overflow the §5.11.39 line-97 `Quant[pos] &
  0xFFFFF` 20-bit clip). 10 new lib tests (1215 → 1225): min magnitude
  (15, length 1), value 16 / 21 / 270, exhaustive sweep 15..=29,
  max representable (1_048_589, length 20), threshold / zero / past-
  conformance rejects, and a `dc_sign` then `golomb` sequence at the
  forward-scan c = 0 composition. fmt + clippy --all-targets clean.

* **Round 213 — §5.11.39 coefficient base-level chain writers.** Extends
  `encoder::coefficients` with the per-coefficient `coeff_base_eob` /
  `coeff_base` / `coeff_br` writers — the reverse-scan body the §5.11.39
  driver loop calls into for every coded coefficient.

  New public API in `encoder::coefficients` (re-exported from
  `encoder`):
  * `write_coeff_base_eob(writer, cdfs, sym, tx_sz_ctx, ptype, ctx)` —
    inverse of the §5.11.39 line-60 `coeff_base_eob S()` against
    `TileCoeffBaseEobCdf[txSzCtx][ptype][ctx]` (§8.3.2 p.376). One
    3-symbol §9.4 alphabet write (`sym ∈ 0..=2`, level = `sym + 1`).
    `ctx` is `0..SIG_COEF_CONTEXTS_EOB = 4` — the §8.3.2 reduction of
    `get_coeff_base_ctx(..., is_eob = true)` already exposed as the
    public helper `crate::cdf::get_coeff_base_eob_ctx`.
  * `write_coeff_base(writer, cdfs, sym, tx_sz_ctx, ptype, ctx)` —
    inverse of the §5.11.39 line-63 `coeff_base S()` against
    `TileCoeffBaseCdf[txSzCtx][ptype][ctx]` (§8.3.2 p.371). One
    4-symbol §9.4 alphabet write (`sym ∈ 0..=3`, level = `sym`). `ctx`
    is `0..SIG_COEF_CONTEXTS = 42` — the public helper
    `crate::cdf::get_coeff_base_ctx` (`is_eob = false`) supplies it
    from the running `Quant[]` array.
  * `write_coeff_br(writer, cdfs, sym, tx_sz_ctx, ptype, ctx)` —
    inverse of one §5.11.39 line-67 `coeff_br S()` against
    `TileCoeffBrCdf[Min(txSzCtx, TX_32X32)][ptype][ctx]` (§8.3.2
    p.378). One `BR_CDF_SIZE = 4`-symbol §9.4 alphabet write (`sym ∈
    0..=3`). `ctx` is `0..LEVEL_CONTEXTS = 21` — the public helper
    `crate::cdf::get_br_ctx` supplies it. One write per call; the
    chain-stacker that bounds at `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)
    = 4` iterations lives in the follow-on arc that lands the full
    §5.11.39 driver loop.

  Stateless on purpose, same pattern as round 212. 14 new lib tests
  (1201 → 1215): per-writer round-trips at the zero / chain-continues /
  upper-ctx edges, the `coeff_br` `Min(txSzCtx, TX_32X32)` clamp,
  out-of-range guards for each axis, and a hand-built two-coefficient
  driver-shape integration test that drives the §8.3.2 helpers on both
  encoder and decoder sides off the same `Quant[]` array
  (`coeff_base_eob = 0` at `c = eob - 1`, then `coeff_base = 3` +
  `coeff_br = 0` at `c = 0`).

  Next arc: the `golomb_length_bit` / `golomb_data_bit` magnitude tail
  for coefficients above `NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`,
  followed by the §5.11.39 driver loop and the §5.11.4 partition
  decision-tree writer.

* **Round 212 — first §5.11.39 coefficient-encode primitives.** New
  `encoder::coefficients` module: the encoder counterparts to the
  three "entry" §8.2.6 `S()` reads inside
  `PartitionWalker::coefficients` (the §5.11.39 `coefficients()`
  reader).

  New public API in `encoder::coefficients` (re-exported from
  `encoder`):
  * `write_txb_skip(writer, cdfs, all_zero, tx_sz_ctx, ctx)` —
    inverse of the §5.11.39 line-13 `all_zero S()` against
    `TileTxbSkipCdf[txSzCtx][ctx]` (§8.3.2 p.371). One binary symbol.
  * `write_eob_pt(writer, cdfs, eob, tx_size, tx_class, plane,
    is_inter)` — inverse of the §5.11.39 lines 19–55 EOB block: the
    `eob_pt_{16,32,64,128,256,512,1024}` selector S() plus the
    optional `eob_extra` S() and the raw `eob_extra_bit` L(1) tail.
    Solves for `eobPt` from the target `eob` via the §5.11.39 line-39
    `(eobPt < 2 ? eobPt : (1 << (eobPt - 2)) + 1)` inverse, then
    emits the `(eobPt - 2)`-bit residue MSB-first against the
    contextual `eob_extra` row + the raw L(1) loop, in lockstep with
    the decoder.
  * `write_dc_sign(writer, cdfs, dc_sign, plane, ctx)` — inverse of
    the §5.11.39 line-564 `dc_sign S()` against
    `TileDcSignCdf[ptype][ctx]` (§8.3.2 p.377). Caller honours the
    `Quant[0] != 0` gate before invoking.

  Stateless on purpose, same pattern as `block_mode_info`: every
  writer takes its §8.3.2 ctx values as inputs. The follow-on arc
  lands `coeff_base{_eob}` + `coeff_br` (the four- / three-symbol
  per-coefficient alphabets) and the `golomb_length_bit` /
  `golomb_data_bit` magnitude tail.

  21 new lib tests (1180 → 1201), all bit-exact round-trips: encode
  through `SymbolWriter`, decode back through small `read_*` shims
  that mirror the §5.11.39 reader's first three `S()` reads (the
  full reader also reads the coeff_base / coeff_br chain we don't
  emit yet). Coverage includes `txb_skip ∈ {0, 1}` at `(0, 0)` and
  the upper grid `(TX_SIZES - 1, TXB_SKIP_CONTEXTS - 1)`, `eob` ∈
  `{1, 2, 3, 8, 16}` on `TX_4X4` / `TX_8X8` (eobMultisize ∈ {0, 1};
  refinement bit-widths 0 → 3), `eob = 5` on `TX_16X16` chroma intra
  with `TX_CLASS_HORIZ` (exercising the `ptype = 1` axis + the
  non-`TX_CLASS_2D` ctx flip), `eob = 17` on `TX_32X32` inter
  (eobMultisize = 3), `dc_sign ∈ {0, 1}` on luma `ctx = 0` and
  chroma `ctx = 2`, plus a sequenced `txb_skip = 0` → `eob_pt(4)`
  round-trip confirming §8.3 CDF adaptation stays in lockstep across
  multiple writer calls. Caller-bug guards (`eob = 0`, `tx_size >=
  TX_SIZES_ALL`, alphabet-overflow `eob`, out-of-range ctx/symbol)
  surface as `Error::PartitionWalkOutOfRange`.

* **Round 211 — first per-block syntax writers (intra arm).** New
  `encoder::block_mode_info` module: pure stateless writers covering
  the §5.11.7 / §5.11.8 / §5.11.11 / §5.11.22 syntax elements an
  intra-only single-block payload needs.

  New public API in `encoder::block_mode_info` (re-exported from
  `encoder`):
  * `write_skip(writer, cdfs, skip, ctx, seg_skip_active)` — inverse
    of `decode_skip` (§5.11.11). `seg_skip_active` arm emits no bits;
    otherwise writes one `S()` against `Default_Skip_Cdf[ctx]`.
  * `write_intra_segment_id(writer, cdfs, segment_id, skip, pred,
    ctx, segmentation_enabled, last_active_seg_id)` — inverse of
    `decode_intra_segment_id` (§5.11.8 + §5.11.9). Disabled / skip /
    pred arms emit no bits; the active arm writes the `diff` whose
    `neg_deinterleave(diff, pred, max) == segment_id` reconstruction
    yields the target id (O(8) inverse search over the §5.11.9
    bijection).
  * `write_intra_frame_y_mode(writer, cdfs, y_mode, abovemode_ctx,
    leftmode_ctx)` — inverse of `decode_intra_frame_y_mode` (§5.11.7
    line 13 with the §8.3.2 neighbour-CDF `TileIntraFrameYModeCdf
    [abovemode][leftmode]` row).
  * `write_y_mode(writer, cdfs, y_mode, size_group_ctx)` — inverse
    of the §5.11.22 line 3 `y_mode` S() with the `Size_Group[MiSize]`
    ctx (the inter-frame intra-block path).
  * `write_intra_uv_mode(writer, cdfs, y_mode, uv_mode, cfl_allowed)`
    — inverse of the §5.11.22 line 6 `uv_mode` S() with the §8.3.2
    CFL-allowed selector (caller passes the resolved boolean, e.g.
    via `cdf::cfl_allowed_for_uv_mode`).

  Stateless on purpose: every writer takes its §8.3.2 ctx as input,
  mirroring `SymbolWriter::write_symbol`'s caller-supplied CDF slice
  pattern. The caller derives ctx through the public §8.3.2 helpers
  (`skip_ctx`, `intra_mode_ctx`, `size_group`, `segment_id_ctx`)
  exactly as the decoder side does.

  13 new lib tests (1167 → 1180), all bit-exact round-trips: encode
  through `SymbolWriter`, then decode back through the matching
  `PartitionWalker::decode_skip` / `decode_intra_segment_id` /
  `decode_intra_frame_y_mode` / `decode_intra_block_mode_info`
  methods on a fresh walker + default CDFs. Coverage includes the
  `seg_skip_active` no-bit arm, the `segmentation_enabled = false`
  no-bit arm, the §5.11.9 skip short-circuit, the segmentation
  else-branch `neg_deinterleave` round-trip, the §5.11.7
  `DC_PRED` / `V_PRED` intra-frame y_mode round-trips, the §5.11.22
  composed `y_mode` + `uv_mode` round-trip on a CFL-allowed block,
  the §5.11.22 monochrome (`has_chroma = false`) y_mode-only path,
  the `seg_skip_active` arm `skip != 1` caller-bug rejection, and
  the composed `{skip, intra_segment_id, y_mode, uv_mode}` smoke
  test in §5.11.7 order.

* **Round 210 — §5.11.1 `tile_group_obu` framing writer + entropy-
  encoder composition wrappers.** The §5.11.1 body framer is the next
  arc on top of the r209 `SymbolWriter`; the composition wrappers
  bring the encoder side to parity with the `SymbolDecoder`'s
  `read_literal` / `read_ns` / `decode_subexp_bool` primitives so the
  per-block §5.11 syntax writers can be composed without further
  scaffolding.

  New public API in `encoder::tile_group_obu`:
  * `TilePayload { bytes }` — one tile's entropy bytes (the raw
    `SymbolWriter::finish` output, including §8.2.4 zero pads).
  * `TileGroupObu { num_tiles, tile_cols_log2, tile_rows_log2,
    tile_size_bytes, tg_start, tg_end, start_and_end_present, tiles }`
    — §5.11.1 descriptor; `TileGroupObu::whole_frame(...)` for the
    §6.10.1 single-tile-group default.
  * `TileGroupObuWriter` / `write_tile_group_obu(&obu)` — emits the
    §5.11.1 body: `tile_start_and_end_present_flag` (`f(1)`, NumTiles
    > 1 only), `tg_start` / `tg_end` (`f(tileBits)` each, flag = 1
    only), `byte_alignment()`, per non-last tile `tile_size_minus_1`
    (`le(TileSizeBytes)`) + tile bytes, last tile bytes (no size).
  * `parse_tile_group_obu_body(body, ..) -> ParsedTileGroup` —
    walker used by the round-trip tests.

  New public API in `encoder::symbol_writer`:
  * `SymbolWriter::write_literal(n, value)` — inverse of §8.2.5
    `read_literal(n)`. MSB-first composition of `write_bool`.
  * `SymbolWriter::write_ns(n, value)` — inverse of §4.10.10 `NS(n)`.
    Values `0..m` written in `w - 1` literal bits; `m..n` written as
    `(value + m)` in `w` literal bits.
  * `SymbolWriter::write_subexp_bool(num_syms, k, value)` — inverse
    of §5.9.28 `decode_subexp_bool`. Walks the `(i, mk)` ladder,
    emitting `subexp_more_bools` advances until the current rung
    covers `value`, then the `NS(num_syms - mk)` uniform tail or the
    `L(b2)` fixed-width tail.

  15 new lib tests (1152 → 1167) — every wrapper round-trips through
  the existing `SymbolDecoder` primitives and every `TileGroupObu`
  shape round-trips through `parse_tile_group_obu_body`.

* **Round 209 — §8.2 entropy *encoder* (`SymbolWriter`).** Inverse of
  `SymbolDecoder` (§8.2.2 `init_symbol` / §8.2.6 `read_symbol` /
  §8.2.4 `exit_symbol`). The foundation every §5.11 tile-content
  encode pass sits on top of: without it nothing inside
  `tile_group_obu()` can be written.

  New public API in `encoder::symbol_writer`:
  * `SymbolWriter::new(disable_cdf_update: bool)` — construct.
  * `SymbolWriter::write_symbol(symbol, cdf)` — inverse of §8.2.6
    `read_symbol(cdf)`. Computes the same `cur(s)` / `prev`
    boundaries the decoder uses, advances `low` to the symbol's
    u-space interval (`offset = range - prev`), shrinks `range`,
    and renormalises by appending `bits = 15 - FloorLog2(range)`
    zero bits to the accumulated `low`. Applies §8.3 `update_cdf`
    in lockstep with the decoder when `!disable_cdf_update`.
  * `SymbolWriter::write_bool(bit)` — inverse of §8.2.3
    `read_bool()`; uses the fresh `[1<<14, 1<<15, 0]` CDF and
    discards the §8.3 adaptation per the spec's note.
  * `SymbolWriter::finish() -> Vec<u8>` — emits accumulated `low`
    bits MSB-first as the encoded bytestream the decoder consumes.

  Implementation: the encoder maintains `low` as a `Vec<bool>` of
  MSB-first bits with explicit big-integer addition; carry-free,
  byte-flush-free, bounded by tile-payload sizes that fit in RAM.
  A future arc that needs to encode multi-megabyte tiles can
  swap in the standard carry-buffer flush without changing the
  external API.

  Helper made `pub(crate)`:
  `symbol_decoder::update_cdf_for_encoder` — the §8.3 update is
  shared between encode and decode (same arithmetic), so both sides
  apply the exact same rule to the same CDF arrays.

  11 new lib tests (1141 → 1152), all bit-exact round-trips through
  the existing `SymbolDecoder`: single bool 0 / 1, mixed-pattern
  bools, multi-symbol N=4 with §8.3 adaptation (encoder/decoder
  CDFs remain equal across the run), heavy- and rare-side skewed
  CDFs, 200-symbol N=8 with §8.3 adaptation, mixed bool +
  multi-symbol on one writer, empty-payload sanity, all-ones and
  all-zeros bool streams.

  Out of scope this arc (next): `read_literal` / `NS` /
  `decode_subexp_bool` inverses for the descriptor wrappers; §5.11
  `tile_group_obu()` writer skeleton on top of `SymbolWriter`; the
  first per-block syntax (skip / mode_info / coefficient encode);
  byte-level flushing during encode.

* **Round 208 — encoder arc 3: §5.3.4 `trailing_bits` + §5.3.1 OBU
  size-field wrapper + §7.5 temporal-unit aggregator.** End-to-end
  glue that takes the existing `write_sequence_header_obu` (r206) +
  `write_frame_header_obu` (r207) payload writers and produces a
  complete, parser-walkable IVF file.

  New `BitWriter` primitives:
  * `BitWriter::trailing_bits(nb_bits)` — §5.3.4 `trailing_bits()`
    syntax (1 `trailing_one_bit` followed by `nb_bits - 1` zero
    `trailing_zero_bit`s).
  * `BitWriter::trailing_bits_to_alignment()` — pads the body up to
    the next byte boundary, emitting a full `0x80` byte when the
    writer is already byte-aligned (the common §5.3.1 case where
    `payloadBits` is a multiple of 8 and the wrapper still needs the
    `trailing_one_bit` slot to mark end-of-body).

  New §5.3.1 OBU framing helpers in `encoder::obu`:
  * `write_obu_with_size(out, header, body_bytes)` — composite that
    appends the §5.3.4 trailer (one `0x80` byte) for the OBU types
    the §5.3.1 `open_bitstream_unit()` tail wraps (every type except
    `OBU_TILE_GROUP` / `OBU_TILE_LIST` / `OBU_FRAME`) and computes
    `obu_size` over the trailer-inclusive body, then writes
    `obu_header` + `leb128(obu_size)` + body.
  * `obu_type_takes_trailing_bits(t)` — the §5.3.1 type-gate.
  * `ObuFrame { header, body }` + `write_temporal_unit(&[ObuFrame])`
    — sequence-of-OBUs aggregator.
  * `build_temporal_unit(seq_payload, &[ObuFrame])` — §7.5 helper
    that prefixes an `OBU_TEMPORAL_DELIMITER` (zero-payload, no
    trailer) and an optional `OBU_SEQUENCE_HEADER` ahead of the
    supplied frame OBUs.

  New typed-descriptor aggregator in `encoder::temporal_unit`:
  * `TemporalUnitPlan<'a> { seq, emit_sequence_header, frames }` —
    one §7.5 temporal unit described in terms of the parser's
    `SequenceHeader` + `FrameHeader[]` rather than raw bytes.
  * `encode_sequence_header_obu(seq) -> Vec<u8>` and
    `encode_frame_header_obu(fh, seq) -> Vec<u8>` — wrap a typed
    descriptor into a complete OBU including the §5.3.4 trailer +
    §5.3.1 size field.
  * `encode_temporal_unit(&TemporalUnitPlan)` — produces the
    complete byte-aligned bytestream for one TU.

  End-to-end smoke fixture (`tests/encoder_end_to_end_ivf.rs`):
  * Builds a `SequenceHeader` + `FrameHeader` from the
    tiny-i-only-16x16-prof0 corpus fixture, encodes a single-TU IVF
    file via the new aggregator + r206 `IvfWriter`, demuxes it back,
    walks the OBUs via the parser-side `ObuIter`, and confirms
    `parse_sequence_header` + `parse_frame_header` round-trip the
    embedded descriptors. A second test exercises the multi-TU /
    "SH only in TU0" pattern.

  15 new lib tests (1126 → 1141): 3 BitWriter (`trailing_bits(5)`,
  byte-aligned-emits-full-byte, partial-byte pad), 7 OBU framer
  (`write_obu_with_size` with SH / Frame / TileGroup / TileList,
  zero-body short-circuit, `write_temporal_unit`,
  `build_temporal_unit` with + without SH,
  `obu_type_takes_trailing_bits` table), 4 typed-descriptor wrappers
  (encode SH OBU, encode FH OBU, encode TU with SH, encode TU without
  SH). Plus 2 integration tests (end-to-end IVF round-trip
  single-frame + multi-frame).

  Next arc: §5.11 `tile_group_obu()` writer (entropy coder +
  coefficient encode + per-block syntax) so the emitted file decodes
  to pixels; §5.9.7 `frame_size_with_refs()` inverse for the inter
  frames that override the seq size; §5.9.24 `read_global_param`
  signed-subexp inverse for non-IDENTITY refs.

* **Round 207 — encoder arc 2: `frame_header_obu()` writer.**
  `encoder::frame_obu::write_frame_header_obu` lands as the encoder
  counterpart to `crate::frame_header::parse_frame_header`. Takes a
  fully-populated `FrameHeader` + the active `SequenceHeader` and
  emits the §5.9.1 `uncompressed_header()` payload bytes the OBU
  framer should wrap with `obu_type == OBU_FRAME_HEADER`.
  Sub-procedures covered:
  * §5.9.2 `uncompressed_header()` — show-existing-frame replay,
    reduced-still-picture-header, intra (KEY / INTRA_ONLY), and the
    inter shared-tail (above `disable_frame_end_update_cdf`) for
    headers where `frame_size_with_refs()` would have bailed.
  * §5.9.3 `allow_intrabc` gate (`allow_scc && UpscaledWidth ==
    FrameWidth`).
  * §5.9.5 `frame_size()` + §5.9.6 `render_size()` + §5.9.8
    `superres_params()` + §5.9.9 `compute_image_size()`.
  * §5.9.10 `read_interpolation_filter()` (inter path).
  * §5.9.11 `loop_filter_params()` (with the `CodedLossless ||
    allow_intrabc` short-circuit + ref/mode-delta update walks).
  * §5.9.12 `quantization_params()` + §5.9.13 `read_delta_q()`.
  * §5.9.14 `segmentation_params()` (with `update_data` feature
    walk over `MAX_SEGMENTS * SEG_LVL_MAX` slots, signed / unsigned
    feature bit widths from `SEGMENTATION_FEATURE_BITS[]` and
    `SEGMENTATION_FEATURE_SIGNED[]`).
  * §5.9.15 `tile_info()` (uniform + non-uniform paths with the
    `width_in_sbs_minus_1` / `height_in_sbs_minus_1` `ns(n)` walks).
  * §5.9.17 `delta_q_params()` + §5.9.18 `delta_lf_params()`.
  * §5.9.19 `cdef_params()` (with the secondary `4 ⇒ 3` invert of
    the §5.9.19 `== 3 ⇒ += 1` adjustment).
  * §5.9.20 `lr_params()` (with the `lr_unit_shift` /
    `lr_unit_extra_shift` / `lr_uv_shift` bit derivations).
  * §5.9.21 `read_tx_mode()` (no bits under `CodedLossless`).
  * §5.9.22 `skip_mode_params()` + §5.9.23 `frame_reference_mode()`
    intra short-circuits.
  * §5.9.24 `global_motion_params()` intra short-circuit + inter
    `IDENTITY`-only emission.
  * §5.9.30 `film_grain_params()` short-circuit + reset path +
    fully-populated apply_grain==true path.

  Three new `BitWriter` primitives land alongside: `write_uvlc`
  (§4.10.3), `write_su` (§4.10.6), `write_ns` (§4.10.7) — the
  inverses of the parser's existing `uvlc()` / `su(n)` / `ns(n)`
  descriptor readers.

  +18 tests (1108 → 1126 lib): 5 new primitive round-trips (uvlc,
  su over a range of n/value pairs, ns over n in 1..=17, ns(1) zero
  bits) plus 13 frame-header round-trips covering the tiny-i-only
  fixture (KEY frame, 72 bits exact match), synthetic intra over
  the same seq, lossless intra (CodedLossless ⇒ all the
  short-circuit paths fire), QM-enabled, non-zero delta-q offsets,
  non-trivial CDEF strengths (with the `4`-invert path), LR with
  Wiener on Y plane, segmentation with active ALT_Q feature, render
  size different from frame size, loop-filter delta updates,
  show-existing-frame replay. The §5.9.1 OBU-payload framing
  (size-field self-counts + trailing bits) and the §5.9.24 inter
  `read_global_param` signed-subexp inverse are next-arc; the
  §5.9.31 `temporal_point_info` round-trip is gated by a debug
  assert that fires the same way the parser refuses to descend.

* **Round 206 — encoder bootstrap (arc 1: bit-output plumbing).**
  New `crate::encoder` module groups three writers covering the
  byte-aligned framing layers above any future coefficient/tile
  encode work:
  * `encoder::bitwriter::BitWriter` — MSB-first bit-output buffer,
    the inverse of `crate::bitreader::BitReader` (§8.1 `read_bit`).
    Primitives: `write_bit`, `write_bits(n, value)` (inverse of
    §4.10.2 `f(n)`), `write_leb128(value)` (inverse of §4.10.5
    `leb128()`, with the same `(1 << 32) - 1` cap), `byte_align`,
    `leb128_size`, `finish`.
  * `encoder::obu` — `ObuHeader` / `ObuExtensionHeader` / `ObuWriter`
    plus the free `write_obu()` function emitting the §5.3.2
    `obu_header`, optional §5.3.3 `obu_extension_header`, and the
    §5.3.1 / §4.10.5 `obu_size` leb128 for the §5.2 low-overhead
    bytestream format. Multiple OBUs concatenate as the byte-aligned
    stream the parser's `ObuIter` walks.
  * `encoder::sequence_obu::write_sequence_header_obu` —
    `sequence_header_obu()` payload writer per §5.5.1 (with §5.5.2
    `color_config`, §5.5.3 `timing_info`, §5.5.4
    `decoder_model_info`, §5.5.5 `operating_parameters_info`).
    Reuses the parser's `crate::sequence_header::SequenceHeader`
    struct as source-of-truth descriptor, so every encoder output
    round-trips through `parse_sequence_header` byte-for-byte
    (modulo the `bits_consumed` field which is parse-side
    bookkeeping).
  * `encoder::ivf` — IVF v0 container writer (`IvfWriter<W: Write>`
    + free `build_file_header` / `build_frame_header` helpers).
    Trivial public file format byte layout (32-byte file header +
    12-byte per-frame header, all little-endian) is documented
    inline at module level. `patch_frame_count` available when the
    sink is `Seek`. Byte-for-byte exact against the
    `tiny-i-only-16x16-prof0/input.ivf` fixture file header.

  +43 tests (1065 → 1108 lib): 8 BitWriter, 7 OBU, 9 IVF, 19
  sequence-OBU (including round-trips through the decoder for
  reduced-still-picture, monochrome, profile-2 4:2:2 12-bit,
  multi-operating-point, timing+decoder-model, color description,
  frame-id-numbers, screen-content-tools toggles). No
  `frame_header_obu` / tile / coefficient encode yet — subsequent
  arcs. `encode_av1` continues to return `Error::NotImplemented`.

* **Round 205 — §7.10.2.5 temporal scan + §7.10.2.6 temporal
  sample wired into `find_mv_stack`; §7.9 motion-field
  estimation helpers landed.** The r172 `TemporalMvScanUnsupported`
  deferral is retired. `find_mv_stack` now accepts a
  `&MotionFieldMvs` grid and, when `use_ref_frame_mvs == true`,
  invokes the §7.10.2.5 `scan_blk` driver at step 17 of the §7.10.2
  outer body. The §7.10.2.5 inner loops cover the per-block 4×4
  step grid (`stepW4 = bw4 >= 16 ? 4 : 2`, `stepH4 = bh4 >= 16 ? 4
  : 2`, capped at `Min(b{w,h}4, 16)`) plus the §7.10.2.5
  `allowExtension` corner-offset table `{{bh4,-2},{bh4,bw4},
  {bh4-2,bw4}}` gated on `BLOCK_8X8 <= block < BLOCK_64X64` and
  the §7.10.2.5 `check_sb_border` 16×16-MI window. The §7.10.2.6
  body reads `MotionFieldMvs[RefFrame[list]][y8][x8]`, short-
  circuits on the `MFMV_INVALID = -1 << 15` sentinel, runs the
  §7.10.2.10 lower-precision pass, then dedupes / appends to the
  stack with weight 2 — single-pred + compound branches both
  honoured. The §7.10.2.6 centre-cell `ZeroMvContext` adjustment
  (set to 1 unconditionally at `(dr=0, dc=0)`; refined to 0 only
  when the MV diverges from `GlobalMvs` by < 16 1/8-sample units
  on a non-sentinel cell) is wired verbatim. New
  `MotionFieldMvs` aggregate (per-ref 7-frame × `h8` × `w8` × 2
  i16 grid sized for `MiRows >> 1 × MiCols >> 1` per §7.9.1) with
  `new_invalid` constructor, `set` / `get` / `set_all_refs`
  accessors. New §7.9.3 `get_mv_projection(mv, num, denom)`
  helper using the `Div_Mult[32]` quantised-inverse table
  (av1-spec p.215) with §4.7.2 `Round2Signed(_, 14)` clamp into
  `±((1 << MV_IN_USE_BITS) - 1)`. New §7.9.4 `get_block_position(
  x8, y8, dst_sign, projMv, mi_rows, mi_cols)` helper returning
  `Option<(u32, u32)>` for the §7.9.4 `posValid` outcome. New §3
  constants `MFMV_STACK_SIZE = 3`, `MAX_OFFSET_WIDTH = 8`,
  `MAX_OFFSET_HEIGHT = 0`, `MV_IN_USE_BITS = 14`, `MFMV_INVALID =
  -1 << 15`, `DIV_MULT[32]` (av1-spec p.215). `InterFrameContext`
  gains a `motion_field_mvs: &'a MotionFieldMvs` field plus an
  `'a` lifetime parameter (and drops `Copy`); `identity_default(
  motion_field_mvs)` takes a borrow. `decode_inter_block_mode_info`
  / `decode_inter_frame_mode_info` gain a trailing
  `motion_field_mvs: &MotionFieldMvs` parameter. The
  `TemporalMvScanUnsupported` error variant is removed. +9 lib
  tests (1056 → 1065), workspace-wide 1119 → 1128.

* **Round 204 — §9.5.3 Quantizer matrix tables + §7.12.3 step-1b
  QM-active arm wired.** Transcribes the normative `Qm_Offset[
  TX_SIZES_ALL ]` (av1-spec p.510) and `Quantizer_Matrix[ 15 ][ 2
  ][ QM_TOTAL_SIZE = 3344 ]` (p.511-553) tables verbatim into a new
  `crate::qmatrix` module (~98 KiB of `.rodata`, observed value
  range 30..=242 fits in u8). Five `Qm_Offset` entries share
  storage per the §9.5.2 derivation: TX_64X64 / TX_32X64 /
  TX_64X32 share TX_32X32's 32×32 region (offset 336); TX_16X64
  shares TX_16X32 (offset 1680); TX_64X16 shares TX_32X16 (offset
  2192). New constants `QM_TOTAL_SIZE`, `QM_LEVELS = 15` (level 15
  is the no-QM short-circuit and is not stored), `QM_PLANES = 2`
  (`0` = luma, `1` = chroma; spec selector `plane > 0`); new helper
  `qmatrix_value( seg_qm_level, plane, tx_size, i, j ) -> u8` with
  the spec's `tw = min(32, Tx_Width[ txSz ])` / `th = min(32,
  Tx_Height[ txSz ])` clamp baked in. The `cdf::dequantize_step1`
  QM-active arm — previously a `debug_assert!`-guarded
  fall-through to `q2 = q` — now evaluates `q2 = Round2( q *
  Quantizer_Matrix[ ... ], 5 ) = (q * qm + 16) >> 5` per §7.12.3
  step-1b + §3 p.13 integer Round2. The §7.12.3 guards
  (`using_qmatrix && PlaneTxType < IDTX && SegQMLevel < 15`) are
  preserved verbatim. +15 lib tests (1041 → 1056), workspace-wide
  1104 → 1119.

* **Round 203 — §7.11.3.1 OBMC motion-mode arm wired into
  `predict_inter`.** The r194 `PredictInterObmcUnsupported` stub is
  retired in favour of an end-to-end OBMC dispatch: the driver now
  performs the §7.11.3.9 `overlapped_motion_compensation` mi-grid
  neighbour walk (above-row + left-column passes per av1-spec p.275
  lines 15301-15346) and runs the §7.11.3.10 overlap blend (p.277-278
  lines 15440-15449) against each qualifying neighbour's translational
  MC. The §7.11.3.9-10 pixel-blend leaves landed in r193
  (`overlap_blending` / `overlap_neighbour_predict_blend` /
  `get_obmc_mask` / the five `OBMC_MASK_*` tables); r203 is the
  driver-side mi-grid walk that consumes them. New public
  `ObmcParams<'a, 'n>` struct plumbs the spec inputs (`MiRow` /
  `MiCol` / `MiCols` / `MiRows` / `Mi_{Width,Height}_Log2[MiSize]` /
  `AvailU` / `AvailL` / `get_plane_residual_size(MiSize, plane) >=
  BLOCK_8X8`) plus two ordered slices of qualifying neighbours
  (`above_neighbours` / `left_neighbours`); new
  `ObmcNeighbour<'a>` bundles the neighbour's `PredictInterRef` (mv +
  ref-frame plane) plus the `Clip3(2, 16,
  Num_4x4_Blocks_{Wide,High}[candSz])` step-4 advance. The
  module-private `obmc_walk_axis` helper drives the spec's per-pass
  `(x4, y4, step4, nLimit)` walk + per-candidate `predict_overlap`
  steps 1-8. `predict_inter`'s signature gains a trailing `obmc:
  Option<&ObmcParams<'_, '_>>` argument before `pred_out`; passing
  `obmc = None` with `motion_mode == MOTION_MODE_OBMC` is a caller
  bug (returns `PartitionWalkOutOfRange`). All non-OBMC arms behave
  identically to r202 when `obmc == None`. The
  `Error::PredictInterObmcUnsupported` variant is removed (no longer
  reachable). **All four motion modes (SIMPLE / WARPED_CAUSAL /
  OBMC + compound) are now driver-side complete.** Test count: 1033
  → 1041 (+8 in lib): empty-list baseline, AvailU/AvailL-off gate,
  identity-blend above-pass, identity-blend left-pass, top-half-only
  modification with non-zero neighbour MV,
  `plane_residual_size_ge_block_8x8` above-pass gate,
  `nLimit = Min(4, Mi_Width_Log2)` cap, and a multi-neighbour
  `step4 += step4` walk on a 16-wide block.

* **Round 202 — §7.11.3.1 WARPED_CAUSAL motion-mode arm wired into
  `predict_inter`.** The r194 `PredictInterWarpUnsupported` stub is
  retired in favour of an end-to-end WARP dispatch: the driver now
  performs the av1-spec p.257 lines 14308-14349 step-2/3/6/7 `useWarp`
  derivation per `refList` and, when `useWarp ∈ {1, 2}`, drives the
  §7.11.3.5 `block_warp` kernel through the step-12 per-8×8 sub-block
  loop (`i8 ∈ 0..((h-1) >> 3)`, `j8 ∈ 0..((w-1) >> 3)`) in lieu of
  the §7.11.3.4 translational kernel. New public
  `WarpDriverParams` struct plumbs the spec inputs (`YMode`,
  `GmType[refFrame]`, `gm_params[refFrame]`, `LocalWarpParams`,
  `LocalValid`, `is_scaled(refFrame)`, `force_integer_mv`); the
  step-7 decision tree is exposed through the module-private
  `derive_use_warp` helper covering all five spec arms (`w/h < 8` ⇒
  0; `force_integer_mv` ⇒ 0; LOCALWARP + LocalValid ⇒ 1;
  GLOBAL_GLOBALMV + `GmType > TRANSLATION` + `!is_scaled` +
  globalValid ⇒ 2; else 0). The step-3 `plane == 0`
  `setup_shear(LocalWarpParams)` re-validation is performed
  internally — a `LocalWarpParams` with a degenerate `[2] == 0`
  diagonal demotes the run to translational fallback even when the
  caller passes `local_valid = true`. `predict_inter`'s signature
  gains a trailing `warp: Option<&WarpDriverParams>` argument before
  `pred_out`; passing `warp = None` with `motion_mode ==
  WARPED_CAUSAL` is a caller bug (returns `PartitionWalkOutOfRange`).
  All non-WARP arms behave identically to r201 when `warp == None`.
  The `Error::PredictInterWarpUnsupported` variant is removed (no
  longer reachable). Test count: 1025 → 1033 (+8 in lib): per-arm
  routing tests for LOCALWARP and GLOBAL_WARP (hand-composed
  `block_warp` equivalence), three fallback tests pinning step-7
  ◦ 1 (`w/h < 8`) / ◦ 2 (`force_integer_mv`) / ◦ 3-negative
  (`!LocalValid`) / ◦ 4-blocked-by-`is_scaled`, one step-3
  re-validation test, and a `derive_use_warp` truth-table test
  covering all seven step-7 paths.

* **Round 201 — §7.11.3.1 step-14 compound arm wired into
  `predict_inter`.** The r194 `PredictInterCompoundUnsupported` stub is
  retired in favour of an end-to-end compound dispatch: when
  `is_compound == true`, the driver iterates `refList ∈ {0, 1}`, runs
  §7.11.3.3 + §7.11.3.4 twice, and combines via one of the five
  `compound_type` mechanisms per av1-spec p.258 lines 14400-14412
  (`COMPOUND_AVERAGE` inline, `COMPOUND_DISTANCE` via
  `compound_distance_blend`, `COMPOUND_WEDGE` / `COMPOUND_DIFFWTD` /
  `COMPOUND_INTRA` via `mask_blend` against the caller-supplied
  luma-grid mask). New public `CompoundParams<'a>` enum carries the
  per-arm side data (`Average`, `Distance(DistanceWeights)`,
  `Wedge { mask, mask_stride }`, `Diffwtd { mask, mask_stride }`,
  `Intra { mask, mask_stride }`). `predict_inter`'s signature gains a
  trailing `compound: Option<CompoundParams<'_>>` argument; passing
  `Some(_)` on the single-ref arm (or `None` on the compound arm) is
  rejected as a caller bug. Per av1-spec p.258 lines 14386-14393 the
  mask is computed once at `plane == 0` and reused for chroma planes
  — the driver does not cache it across calls; the caller is
  responsible for supplying the same buffer on subsequent plane
  invocations. Internal refactor: the per-`refList` steps 5-13 are
  factored into a new private `predict_inter_per_ref` helper so the
  compound arm runs it twice without code duplication. The
  `Error::PredictInterCompoundUnsupported` variant is removed (no
  longer reachable). Test count: 1019 → 1025 (+6 in lib): one test per
  compound-type arm (`COMPOUND_AVERAGE` / `COMPOUND_DISTANCE` /
  `COMPOUND_WEDGE` / `COMPOUND_DIFFWTD` / `COMPOUND_INTRA`) verifying
  driver output equals hand-composed external blend on a `ref0 == ref1`
  setup, plus a caller-bug guard test for the
  `(is_compound, refs.len(), compound)` mismatch matrix.

* **Round 198 — §7.17.2 / §7.17.3 self-guided projection arm.**
  Completes the §7.17 loop-restoration layer that landed in r197
  (driver + Wiener arm) by implementing the previously-stubbed
  self-guided projection path. New public `self_guided_filter`
  reads the per-unit `set = LrSgrSet[plane][unitRow][unitCol]` and
  weights `(w0, w1) = LrSgrXqd[plane][unitRow][unitCol][0..2]` (two
  new `&dyn Fn(...)` closures on `LoopRestorationFrameContext`:
  `lr_sgr_set`, `lr_sgr_xqd`), invokes the new internal `box_filter`
  helper twice with `(r0, eps0)` / `(r1, eps1)` from
  `SGR_PARAMS[set]`, and combines the two outputs against
  `u = UpscaledCdefFrame[plane][y + i][x + j] << SGRPROJ_RST_BITS`
  per the §7.17.2 projection `v = w1 * u + w0 * (flt0 or u) + w2 *
  (flt1 or u)`, `w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1`,
  `LrFrame = Clip1(Round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS))`
  per av1-spec p.330 lines 18253-18272. `box_filter` implements
  §7.17.3 verbatim (p.331-332 lines 18277-18389): builds `A[]/B[]`
  across `[-1, h + 1] × [-1, w + 1]` via the `(2r + 1)²` box-sum
  kernel, normalises by `n2e = n² · eps`, applies the piecewise
  `a2 ∈ {1, ((z << SGR_BITS) + z/2) / (z + 1), 256}` nonlinearity,
  computes `b2 = ((1 << SGR_BITS) - a2) * b * oneOverN`,
  `B = Round2(b2, SGRPROJ_RECIP_BITS)`, and the second pass
  convolves `(A, B)` with the §7.17.3 weighted 3×3 footprint
  (pass-0 odd-row-only `(5, 6, 5, 0, 0, 0)`; pass-1
  centre-plus-cross 4 / corner 3). The 10/12-bit profile widens
  the `2 * (BitDepth - 8)` / `(BitDepth - 8)` Round2 shifts;
  `i32`-overflowing intermediates use a new internal `round2_i64`
  helper. The `r == 0` early-exit branch (sets 10..=13 for `r0`,
  sets 14..=15 for `r1`) substitutes `u` for the corresponding
  `flt`. `loop_restore_block`'s `SgrProj` arm now routes to
  `self_guided_filter` instead of leaving the pre-copied
  `UpscaledCdefFrame` content. Test count 1053 → 1058 (+5 net in
  lib; +6 new SGR tests minus the retired
  `restore_sgrproj_passes_cdef_through` stub-pass-through
  assertion).
* **Round 197 — §7.17 loop restoration — driver + Wiener arm +
  §7.17.5 coefficient reconstruction + §7.17.6 stripe-aware source
  fetch.** New `loop_restoration` module exposes the §7.17 top-level
  driver (`loop_restoration_frame` walking the frame in `MI_SIZE`
  steps), the §7.17.1 per-block driver (`loop_restore_block` deriving
  `(unitRow, unitCol, x, y, w, h, StripeStartY, StripeEndY,
  PlaneEndX, PlaneEndY)`), the §7.17.4 Wiener arm
  (`wiener_filter` — two-pass separable 7-tap convolution with the
  §7.11.3.2 `(InterRound0, InterRound1)` rounding shifts), the
  §7.17.5 coefficient reconstruction (`wiener_coefficients` — DC-gain
  128 constraint), and the §7.17.6 stripe-aware source fetch
  (`get_source_sample` — routes inside-stripe reads to
  `UpscaledCdefFrame` and out-of-stripe to `UpscaledCurrFrame` with
  the cropped `StripeStartY - 2` / `StripeEndY + 2` neighbour-line
  reach). Constant tables `WIENER_TAPS_{MID,MIN,MAX,K}`,
  `SGRPROJ_XQD_{MID,MIN,MAX}`, `SGR_PARAMS[16][4]`, and the bit-
  precision constants `SGRPROJ_{PARAMS,PRJ,RST,MTABLE,RECIP,SGR}_BITS`
  are transcribed verbatim from av1-spec p.74 / p.107 / p.332. The
  §7.17.2 / §7.17.3 self-guided projection arm was stubbed at this
  round (`RESTORE_SGRPROJ` ⇒ pre-copied `UpscaledCdefFrame`); r198
  implements the body. Test count 1040 → 1053 (+13 in lib).
* **Round 196 — §7.15 CDEF (Constrained Directional Enhancement
  Filter) — top-level driver + direction search + primary +
  secondary tap filter.**
  New `cdef` module exposes the full §7.15 de-ringing surface acting
  on the post-§7.14 deblocked `CurrFrame[plane]` samples per av1-spec
  p.318-324: `cdef_frame` (§7.15.1 per-8×8 walk with `step4 = 2`
  stride and `cdef_idx[r & cdefMask4][c & cdefMask4]` anchor lookup),
  `cdef_block` (§7.15.1 per-block driver — `CurrFrame ↦ CdefFrame`
  copy, `idx == -1` / `Skips[]`-all short-circuits, luma `(priStr *
  (4 + varStr) + 8) >> 4` strength rescale, chroma `CDEF_UV_DIR`
  translation), `cdef_direction` (§7.15.2 — 8-direction
  `partial[8][15]` projection + `DIV_TABLE` cost accumulator + `var
  = (bestCost - cost[(yDir + 4) & 7]) >> 10`), `cdef_filter_block`
  (§7.15.3 per-plane filter — `CDEF_PRI_TAPS` along `dir` +
  `CDEF_SEC_TAPS` along `(dir ± 2) & 7`, both filtered through
  `constrain` damping, output `Clip3(min, max, x + ((8 + sum -
  (sum < 0)) >> 4))`), and `constrain` (§7.15.3 primitive — `sign *
  Clip3(0, |diff|, threshold - (|diff| >> dampingAdj))` with
  `dampingAdj = Max(0, damping - FloorLog2(threshold))`).
  Tables `CDEF_PRI_TAPS`, `CDEF_SEC_TAPS`, `CDEF_DIRECTIONS`,
  `DIV_TABLE`, `CDEF_UV_DIR` are transcribed verbatim from av1-spec
  p.320-324. A small standalone `CdefFrameContext` bundles the
  §5.5 / §5.9.5 / §5.9.19 frame-header inputs plus closures for the
  §5.11 decode state (`cdef_idx`, `skip`). The §5.9.19 `CdefParams`
  (r10) and §5.11.56 `cdef_idx[]` walker (r157) are now consumed
  end-to-end. Test count 1026 → 1040 (+14 in lib): 3 `constrain`
  axis tests, 3 table sanity tests, 3 direction-search tests
  (uniform / horizontal / vertical stripes), 3 `cdef_block`
  copy / skip / filter-identity tests, and 2 frame-level
  driver-walk tests covering the chroma sub-sampling path.
* **Round 195 — §7.14 loop filter (deblocking) — top-level driver +
  edge-strength + narrow/wide filter bodies.**
  New `loop_filter` module exposes the full §7.14 deblocking surface
  acting on the post-§7.12.3 reconstructed `CurrFrame[plane]`
  samples: `loop_filter_frame` (§7.14.1 per-plane × per-pass × per-row
  × per-col walk with the `rowStep`/`colStep` chroma stride),
  `loop_filter_edge` (§7.14.2 per-edge driver — `onScreen` + `(xP, yP)`
  + `(prevRow, prevCol)` + `isBlockEdge` + `isTxEdge` + `applyFilter`
  gates), `filter_size` (§7.14.3 — `Min(plane==0?16:8,
  Min(Tx_*[txSz], Tx_*[prevTxSz]))`),
  `adaptive_filter_strength` + `adaptive_filter_strength_selection`
  (§7.14.4 + §7.14.5 — `(lvl, limit, blimit, thresh)` with
  per-segment `SEG_LVL_ALT_LF_*` + `loop_filter_ref_deltas[]` /
  `loop_filter_mode_deltas[]` adjustments, both `Clip3(0,
  MAX_LOOP_FILTER)` clipped), `sample_filtering` (§7.14.6.1 mask →
  narrow/wide dispatch), `filter_mask` (§7.14.6.2 — `hevMask` +
  `filterMask` + `flatMask` + `flatMask2`), `narrow_filter`
  (§7.14.6.3 — 4-pixel-window edge filter with `filter4_clamp` to
  `[-(1<<(BD-1)), (1<<(BD-1))-1]` and the two-tap `Round2(filter1,
  1)` low-energy arm gated on `!hevMask`), and `wide_filter`
  (§7.14.6.4 — 6/8/14-tap low-pass with `n ∈ {2, 3, 6}` and `n2 ∈ {0,
  1}` per `log2Size`/`plane`, `Clip3(-(n+1), n, ...)` edge
  replication, `i64` accumulator for the 14-tap luma path). New
  public types: `LoopFilterFrameContext<'_>` (frame-level inputs +
  closure surface for `is_intra` / `skip` / `ref_frame` / `mode` /
  `segment_id` / `delta_lf` / `seg_feature_active` /
  `seg_feature_data` / `lf_tx_size` / `mi_size`), `PlaneBuffer<'_>`
  (per-plane sample-buffer mutable view with `(rows, cols, samples)`
  triple), `FilterStrength`, `FilterMaskOutput`. New public ordinal
  constants in `loop_filter`: `NEARESTMV` / `GLOBALMV` /
  `GLOBAL_GLOBALMV` (mirroring `cdf::MODE_*` for the §7.14.4 mode-type
  comparison) and `SEG_LVL_ALT_LF_Y_V = 1`. Test count: 1014 → 1026
  (+12 in lib): four §7.14.4/§7.14.5 strength tests (lvl=0 ⇒ skip;
  basic lvl=32 sharp=0 table; intra ref-delta lifts lvl; clipped at
  `MAX_LOOP_FILTER`); one §7.14.3 filter-size cap test; two narrow
  filter tests (uniform-input idempotency; step-edge softening with
  symmetric move about the mid-point); two wide filter
  uniform-preservation tests at `log2Size ∈ {3, 4}`; two §7.14.6.2
  mask tests (flat region allows filter; steep edge vetoes it); one
  §7.14.1 driver iteration-count test on a 2×2 mi-grid.

* **Round 194 — §7.11.3.1 `predict_inter` driver skeleton
  (translational single-ref SIMPLE arm).**
  New free function `inter_pred::predict_inter(plane, x, y, w, h,
  motion_mode, is_compound, is_inter_intra, bit_depth,
  subsampling_x, subsampling_y, frame_width, frame_height,
  interp_filter_x, interp_filter_y, refs: &[PredictInterRef],
  pred_out: &mut [u16]) -> Result<(), Error>` plus the
  `PredictInterRef<'a>` per-list bundle (carries `ref_plane` +
  `ref_stride` + `ref_upscaled_width` + `ref_width` + `ref_height` +
  `mv`). The single-reference translational path
  (`is_compound == false && motion_mode == MOTION_MODE_SIMPLE &&
  is_inter_intra == false`) is wired end-to-end: §7.11.3.2
  `rounding_variables` → §7.11.3.3 `motion_vector_scaling` → §7.11.3.4
  `block_inter_prediction` → §7.11.3.1 single-ref final-clip
  `Clip1(preds[0][i][j])` via `clip1_single_ref`. Three new dedicated
  [`Error`] variants narrow the live remaining gaps:
  `PredictInterCompoundUnsupported` (step-14 compound combine; the
  §7.11.3.11-15 leaves landed in r191), `PredictInterWarpUnsupported`
  (`motion_mode == MOTION_MODE_WARPED_CAUSAL`; the §7.11.3.5-8 kernels
  landed in r192), `PredictInterObmcUnsupported` (`motion_mode ==
  MOTION_MODE_OBMC`; the §7.11.3.9-10 leaves landed in r193). Test
  count: 1011 → 1014 (+3 in lib).

* **Round 193 — §7.11.3.9-10 OBMC (Overlapped Block Motion
  Compensation) overlap-blend leaves.**
  Three new public bodies in `inter_pred` lift the last §7.11.3
  inter-prediction sample-generation body: `overlap_blending`
  (§7.11.3.10 — per-pixel blend `currentPlane[i][j] = Round2(m *
  currentPlane[i][j] + (64 - m) * obmcPred[i][j], 6)` with row-
  vs-column mask indexing), `get_obmc_mask` (§7.11.3.9
  length-to-table dispatch), `overlap_neighbour_predict_blend` (the
  `predict_overlap` step-8 wrapper that runs the lookup +
  blending). Plus the new `OverlapPass` enum (`Above` / `Left`
  matching spec `pass ∈ {0, 1}`) and the five `OBMC_MASK_*` tables
  transcribed verbatim from av1-spec p.277 lines 15406-15418
  (`OBMC_MASK_2` = `[45, 64]`, `OBMC_MASK_4` = `[39, 50, 59, 64]`,
  `OBMC_MASK_8` = `[36, 42, 48, 53, 57, 61, 64, 64]`,
  `OBMC_MASK_16` rising 34→64 + 4×64 saturated tail,
  `OBMC_MASK_32` rising 33→64 + 8×64 saturated tail).
  The §7.11.3.9 outer mi-grid driver (`while ( nCount < nLimit && x4
  < Min(MiCols, MiCol + w4) )` above-row walk + the mirror left-
  column walk + `get_plane_residual_size( MiSize, plane ) >=
  BLOCK_8X8` gate) needs partially-implemented mi-grid /
  `MiSizes[..]` / `RefFrames[..]` / `Mvs[..]` state and lives in
  the §7.11.3.1 driver wiring deferred to the next arc. Test
  count: 998 → 1011 (+13 in lib).

* **Round 192 — §7.11.3.5-8 WARP motion compensation.**
  Four new public bodies in `inter_pred` lift the §7.11.3.1
  driver's affine-warp MC arm (LOCALWARP + GLOBAL_GLOBALMV):
  `block_warp` (§7.11.3.5 — apply the warp to a single 8×8
  sub-section), `setup_shear` (§7.11.3.6 — derive `(α, β, γ, δ,
  warpValid)` from a 6-element warp matrix), `resolve_divisor`
  (§7.11.3.7 — fixed-point inverse via `DIV_LUT`), `warp_estimation`
  (§7.11.3.8 — least-squares fit of `LocalWarpParams` from the
  §7.10.4 `find_warp_samples` candidate list).
  * `block_warp(use_warp, plane, subX, subY, x, y, i8, j8, w, h,
    ref_plane, ref_stride, ref_upscaled_w, ref_h, warp_params,
    InterRound0, InterRound1, pred_stride, &mut pred)` projects
    `(srcX, srcY) = ((x + j8*8 + 4) << subX, (y + i8*8 + 4) << subY)`
    through the affine matrix into the reference plane, computes the
    spec's `intermediate[15][8]` from a horizontal 8-tap pass with
    `Round2(s, InterRound0)`, then writes the (≤8×≤8) sub-section
    into `pred` from the vertical 8-tap with `Round2(s, InterRound1)`.
    Caller chooses `USE_WARP_LOCAL = 1` for LOCALWARP (using
    `LocalWarpParams` from §7.11.3.8) or `USE_WARP_GLOBAL = 2` for
    GLOBAL_GLOBALMV (using `gm_params[RefFrame[refList]]`).
  * `setup_shear(warp_params) -> Option<ShearParams>` returns
    `(alpha, beta, gamma, delta, warp_valid)` per the spec
    factorisation. `alpha0 = clip(warpParams[2] - (1<<16))`, `beta0 =
    clip(warpParams[3])`, `gamma0` / `delta0` use the §7.11.3.7
    divisor of `warpParams[2]`. Each is `Round2Signed(_,
    WARP_PARAM_REDUCE_BITS=6) << 6`. Returns `None` only for the
    `warpParams[2] == 0` divisor-fail. `warp_valid` is `false` when
    `4|α| + 7|β| >= 1<<16` or `4|γ| + 4|δ| >= 1<<16`.
  * `resolve_divisor(d) -> Option<Divisor>` computes `n =
    FloorLog2(|d|)`, `e = |d| - (1<<n)`, `f = Round2(e, n-8)` (or `e
    << (8-n)`) into `DIV_LUT[f]`, negating when `d < 0`. `divShift =
    n + DIV_LUT_PREC_BITS = n + 14`. Returns `None` for `d == 0`.
  * `warp_estimation(cand_list, mi_row, mi_col, w4, h4, mv) ->
    LocalWarp` builds the symmetric 2×2 matrix `A` and length-2 arrays
    `Bx, By` via `ls_product(a, b) = ((a*b) >> 2) + (a + b)` with the
    spec's per-iter `+8 / +4` biases. `det == 0` ⇒ `LocalValid =
    false`. Otherwise the resolved divisor of `det` drives `nondiag`
    (clamped to `±(1<<13) ∓ 1`) and `diag` (clamped to `(1<<16) ± (1<<13)
    ∓ 1`) on the four affine entries; translation `(LocalWarpParams[0],
    LocalWarpParams[1])` clamps to `±(1<<23) ∓ 1`.
  * Two new transcribed tables: `WARPED_FILTERS[193][8]` (av1-spec
    p.268-270; every row sums to 128) and `DIV_LUT[257]` (av1-spec
    p.272; monotonically non-increasing from `16384 = 1 <<
    DIV_LUT_PREC_BITS` to `8192`).
  * New constants: `WARP_WARPEDMODEL_PREC_BITS = 16`,
    `WARP_PARAM_REDUCE_BITS = 6`, `DIV_LUT_PREC_BITS = 14`,
    `DIV_LUT_BITS = 8`, `DIV_LUT_NUM = 257`, `LS_MV_MAX = 256`,
    `WARPEDMODEL_TRANS_CLAMP = 1<<23`,
    `WARPEDMODEL_NONDIAGAFFINE_CLAMP = 1<<13`,
    `WARPEDPIXEL_PREC_SHIFTS = 64`, `WARPEDDIFF_PREC_BITS = 10`,
    `LEAST_SQUARES_SAMPLES_MAX = 8`, `USE_WARP_LOCAL = 1`,
    `USE_WARP_GLOBAL = 2`. New public types `Divisor`, `ShearParams`,
    `WarpSampleCand`, `LocalWarp`. Test count: 983 → 998 (+15).
    `decode_av1` / `encode_av1` still return `Error::NotImplemented`
    pending the §7.11.3.1 predict_inter driver wiring.

* **Round 191 — §7.11.3.11-15 compound bodies.**
  Five new public bodies in `inter_pred` lift the §7.11.3.1
  bi-prediction blending layer: `wedge_mask` (§7.11.3.11 —
  COMPOUND_WEDGE), `difference_weight_mask` (§7.11.3.12 —
  COMPOUND_DIFFWTD), `intra_mode_variant_mask` (§7.11.3.13 —
  COMPOUND_INTRA), `mask_blend` / `mask_blend_interintra` (§7.11.3.14
  — combine two preds via mask), and `distance_weights` /
  `compound_distance_blend` (§7.11.3.15 — COMPOUND_DISTANCE).
  * `wedge_mask(mi_size, num_4x4_w, num_4x4_h, wedge_index,
    wedge_sign, &mut mask)` generates the per-`(bsize, wedge_index)`
    mask via the spec's `initialise_wedge_mask_table` algorithm: a
    lazily-built 24 KiB `MasterMask[6][64][64]` table backs the
    `(yoff, xoff)` slice extraction; the `flipSign` average-of-edge
    correction is honoured. The 1d driver tables
    (`WEDGE_MASTER_OBLIQUE_{ODD,EVEN}`, `WEDGE_MASTER_VERTICAL`),
    the `WEDGE_CODEBOOK[3][16][3]` per-shape codebook, and the
    `WEDGE_BITS[22]` per-block-size flag table are transcribed
    verbatim. New constants: `MASK_MASTER_SIZE = 64`,
    `WEDGE_DIRECTIONS = 6`, and the six `WEDGE_{HORIZONTAL,
    VERTICAL, OBLIQUE27, OBLIQUE63, OBLIQUE117, OBLIQUE153}`
    direction ordinals. `block_shape(num_4x4_wide, num_4x4_high)`
    helper returns the codebook outer dimension.
  * `difference_weight_mask(bit_depth, inter_post_round, mask_type,
    preds0, preds1, w, h, &mut mask)` fills the per-pixel `m =
    Clip3(0, 64, 38 + Round2(|p0-p1|, (BD-8)+IPR) / 16)` with the
    `mask_type` toggle inverting to `64 - m`.
  * `intra_mode_variant_mask(interintra_mode, w, h, &mut mask)`
    dispatches on `II_DC_PRED` / `II_V_PRED` / `II_H_PRED` /
    `II_SMOOTH_PRED` (new constants) into the 128-entry
    `II_WEIGHTS_1D` table (transcribed verbatim from av1-spec
    p.283-284). New constant `MAX_SB_SIZE = 128`.
  * `mask_blend(...)` applies the inter-inter blend `out =
    Clip1(Round2(m*p0 + (64-m)*p1, 6 + IPR))` with the four
    `(sub_x, sub_y)` chroma-mask averaging cases. Companion
    `mask_blend_interintra(...)` handles the interintra arm where
    the destination already holds the intra prediction and `preds0`
    is the pre-Clip inter prediction.
  * `distance_weights(order_hint_bits, current, ref0, ref1) ->
    DistanceWeights { fwd_weight, bck_weight }` derives the
    COMPOUND_DISTANCE weights from the two refs' `OrderHints[]`
    relative distances. New helper `get_relative_dist(a, b, OHB)`
    implements §5.9.3. New constants `MAX_FRAME_DISTANCE = 31`,
    `QUANT_DIST_WEIGHT[4][2]`, `QUANT_DIST_LOOKUP[4][2]` (all
    transcribed verbatim from av1-spec p.285). The early-return
    `d0 == 0 || d1 == 0` branch and the per-`i` search loop with
    its order-dependent break condition are honoured. Companion
    `compound_distance_blend(...)` applies the
    `Clip1(Round2(fwd*p0 + bck*p1, 4 + IPR))` site (av1-spec p.258
    line 14408). New constants `COMPOUND_WEDGE = 0`,
    `COMPOUND_DIFFWTD = 1`, `COMPOUND_AVERAGE = 2`,
    `COMPOUND_INTRA = 3`, `COMPOUND_DISTANCE = 4` (av1-spec p.185
    table) reproduced locally.
  * `Error::ComputePredictionInterUnsupported` display message
    updated to reflect that the §7.11.3.11-15 compound bodies
    have landed (the variant is retained as a defensive fallback
    because the §7.11.3.1 driver wiring all five bodies into a
    `predict_inter` entry point is still pending).
  * Test count: 954 → 983 (+29). New tests cover: master 1d
    arrays / `MasterMask` value-range invariant; the
    HORIZONTAL=VERTICAL^T and OBLIQUE27=OBLIQUE63^T derivation
    symmetries; `block_shape` truth table; per-pixel range +
    sign-flip-inverts invariants for `wedge_mask`; spec degenerate
    cases for `difference_weight_mask` (38 / 26 / saturation);
    per-arm uniformity / symmetry for `intra_mode_variant_mask`;
    `II_WEIGHTS_1D` monotone-non-increasing invariant; endpoint
    cases (`m == 0` / `m == 64`) for both `mask_blend` and
    `mask_blend_interintra`; `(1, 1)` chroma-subsample averaging
    correctness; `get_relative_dist` sign-extension + OHB=0 path;
    `QUANT_DIST_LOOKUP` row-sums-to-16 invariant; four
    `distance_weights` regime tests (equal d / zero d / OHB=0 / far
    forward); symmetric-weight `compound_distance_blend` reduces to
    arithmetic mean. Each body also has a dedicated caller-bug
    rejection test enforcing the `Err(PartitionWalkOutOfRange)`
    contract.

* **Round 190 — `decode_block_syntax` inter-arm wire-up.**
  Integration arc lifting the historical
  `Error::DecodeBlockInterFrameUnsupported` short-circuit from the
  §5.11.5 walker and threading the §5.11.18 dispatcher's MV /
  ref-frame / interp-filter state into the §5.11.33
  `compute_prediction` inter arm (per r189) and the §5.11.34
  `residual` `is_inter && !Lossless && !plane` `transform_tree`
  arm.
  * `PartitionWalker::decode_block_syntax` and
    `PartitionWalker::decode_partition_syntax` grow an
    `inter_ctx: Option<&InterFrameContext>` last argument.
    Passing `None` preserves the historical short-circuit on
    `frame_is_intra == false`; passing `Some(&ctx)` runs the
    §5.11.18 cascade to completion.
  * New public `InterFrameContext` struct bundling the 25+
    per-frame / per-segment scalars
    `decode_inter_frame_mode_info` consumes (segmentation
    overrides, `skip_mode_present` / `skip_mode_frame[]`,
    `reference_select`, §5.9.24 `gm_type` / `gm_params` /
    `ref_frame_sign_bias`, MV-precision flags, motion-mode trio,
    compound-blend quad, interp-filter pair). The
    `identity_default()` constructor seeds every flag off,
    identity warp params, `EIGHTTAP` interp filter.
  * `DecodedInterFrameModeInfo` gains an
    `inter_block: Option<DecodedInterBlockModeInfo>` field. The
    §5.11.18 dispatcher's `Ok(_)` arm now populates it (lifting
    the pre-r190 `Err(InterBlockModeInfoUnsupported)` stub) so the
    §5.11.5 walker can read back the §5.11.23 outputs.
  * Internal `PartitionWalker::decode_block_syntax_inter_arm`
    helper composes §5.11.18 → §5.11.16 → §5.11.33 → §5.11.34 on
    the inter path; returns a populated `DecodedBlock` with
    `is_inter = 1`, `ref_frame[0..2]` carrying the §5.11.23
    `ref_frame` output (cast `i32` → `i8` per the `DecodedBlock`
    field type), `y_mode` carrying the §5.11.23 inter Y-mode
    ordinal, `is_compound` from `RefFrame[1] > INTRA_FRAME`.
  * Test count: 949 → 954 (+5). New tests in
    `tests/decode_block_syntax_walker.rs`:
    `r190_decode_block_syntax_with_inter_ctx_runs_inter_arm_to_completion`
    (end-to-end on the `seg_globalmv_active` path with identity
    MV); `r190_decode_partition_syntax_with_inter_ctx_routes_through_inter_arm`
    (the §5.11.4 driver propagates the context to its `BLOCK_4X4`
    leaf); `r190_decode_block_syntax_inter_arm_without_ctx_keeps_legacy_stub`
    (`None` preserves the historical short-circuit);
    `r190_inter_frame_context_identity_default_matches_spec_identity_warp`
    (constructor matches the §5.9.24 identity-warp defaults);
    `r190_decoded_inter_frame_mode_info_intra_arm_still_stubs`
    (the §5.11.22 stub on the `is_inter == 0` arm is unchanged).
    Two pre-existing `decode_inter_frame_mode_info_*` tests were
    converted from asserting `Err(InterBlockModeInfoUnsupported)`
    to asserting `Ok(_)` with the new `inter_block: Some(_)`
    field populated.

* **Round 189 — §7.11.3 inter prediction process (translational
  single-reference MC kernel) per av1-spec p.257-265.** New
  `inter_pred` module exposing the three §7.11.3.{2,3,4} sample-
  generation leaves as standalone helpers (mirroring the §7.11.2
  intra-leaf pattern from r180+):
  * §7.11.3.2 `rounding_variables(bit_depth, is_compound)` — derives
    the `(InterRound0, InterRound1, InterPostRound)` rounding shifts
    per av1-spec p.259. Six-case spec table verified: `(8, false) →
    (3, 11, 0)`, `(8, true) → (3, 7, 4)`, `(10, false) → (3, 11, 0)`,
    `(10, true) → (3, 7, 4)`, `(12, false) → (5, 9, 0)`, `(12, true)
    → (5, 7, 2)`.
  * §7.11.3.3 `motion_vector_scaling(plane, subX, subY, FW, FH,
    RefUpscaledWidth, RefFrameHeight, x, y, mv)` — returns
    `(startX, startY, stepX, stepY)` in `SCALE_SUBPEL_BITS = 10`
    fractional bits per av1-spec p.260-261. Identity case
    (`RefUpscaledWidth == FrameWidth`, zero MV) yields `stepX = stepY
    = 1024` and `startX = (x << 10) + 32` (the spec's `off` rounding
    offset).
  * §7.11.3.4 `block_inter_prediction(plane, subX, subY, refPlane,
    refStride, refW, refH, startX, startY, stepX, stepY, w, h,
    interpFilterX, interpFilterY, R0, R1, pred)` — the translational
    8-tap horizontal + vertical convolution kernel with
    `Clip3(0, lastX/Y, ...)` boundary clamp and `Round2(s, R0/R1)`
    after each pass per av1-spec p.262-265. Phase-0 integer-aligned
    case verified bit-exact: each `pred[r][c] = ref[start_y + r,
    start_x + c]`. Constant-reference boundary-clamp verified.
  * `SUBPEL_FILTERS[6][16][8]` — av1-spec p.263-265
    `Subpel_Filters[]` table transcribed verbatim. All 96 phase rows
    sum to `128` (sample-conservation invariant per av1-spec p.265
    note); every entry even; phase-0 row of every filter is the
    `(0, 0, 0, 128, 0, 0, 0, 0)` unit copy; small-block 4-tap
    reductions (indices 4, 5) zero taps 0/1/6/7.
  * `select_interp_filter_small_block(f)` — the spec's
    `w <= 4` / `h <= 4` remap (`EIGHTTAP` / `EIGHTTAP_SHARP` →
    `EIGHTTAP_4TAP`, `EIGHTTAP_SMOOTH` → `EIGHTTAP_SMOOTH_4TAP`).
  * `clip1_single_ref(bit_depth, pred, out)` — the §7.11.3.1
    `isCompound == 0, IsInterIntra == 0` post-prediction
    `Clip1(preds[0])` clamp for callers that want sample-domain
    outputs.
  * Constants: `FILTER_BITS = 7` (re-exported as
    `INTER_FILTER_BITS`), `SUBPEL_BITS = 4`, `SUBPEL_MASK = 15`,
    `SCALE_SUBPEL_BITS = 10`, `REF_SCALE_SHIFT = 14`,
    `EIGHTTAP_4TAP = 4`, `EIGHTTAP_SMOOTH_4TAP = 5`.
  * §5.11.33 `compute_prediction` dispatcher's `is_inter == 1` arm
    now emits one `PlanePredictionTask` per `(plane, i4, j4)` 4x4
    sub-block carrying `mode = COMPUTE_PRED_MODE_INTER`, `log2_w =
    log2_h = 2`, `start_x = baseX + j4 * 4`, `start_y = baseY + i4 *
    4` (was: surfaced `Error::ComputePredictionInterUnsupported` on
    the first task). A BLOCK_8X8 luma-only inter block now produces
    four tasks at `(0,0)`, `(4,0)`, `(0,4)`, `(4,4)`.
  * `Error::ComputePredictionInterUnsupported` retained for API
    stability; the variant's docstring + `Display` body updated to
    reflect that the translational MC kernel landed (the variant is
    now defensive / no longer surfaced by the dispatcher on the
    `is_inter == 1` arm).
  * **+14 unit tests** in `inter_pred::tests`; one walker integration
    test rewritten
    (`compute_prediction_inter_arm_emits_per_subblock_tasks`).
  * **Scope (sensibly split — single-ref translational only):**
    deferred to subsequent arcs are §7.11.3.5 `block_warp`
    (LOCALWARP / GLOBAL_GLOBALMV affine warp), §7.11.3.6
    `setup_shear`, §7.11.3.7 `resolve_divisor`, §7.11.3.8
    `warp_estimation`, §7.11.3.9 `overlapped_motion_compensation`
    (OBMC), §7.11.3.10 `overlap_blending`, §7.11.3.11 `wedge_mask`,
    §7.11.3.12 `difference_weight_mask`, §7.11.3.13
    `intra_mode_variant_mask`, §7.11.3.14 `mask_blend`, §7.11.3.15
    `distance_weights`. The §7.11.3.1 driver itself awaits the
    §5.11.5 walker's inter arm reaching `predict_inter` (today
    `decode_block_syntax` short-circuits at
    `Error::DecodeBlockInterFrameUnsupported`).

* **Round 188 — §7.11.2.4 six directional D-mode sample-generation
  leaves (D45 / D135 / D113 / D157 / D203 / D67) + §7.11.2.{7, 9, 10,
  11, 12} intra-edge helpers + `Mode_To_Angle[]` /
  `Dr_Intra_Derivative[ 90 ]` tables.** Closes the §7.11.2 Y intra-
  prediction surface — all 13 §6.10.x intra modes (`DC_PRED` /
  `V_PRED` / `H_PRED` / `D45` / `D135` / `D113` / `D157` / `D203` /
  `D67` / `SMOOTH` / `SMOOTH_V` / `SMOOTH_H` / `PAETH_PRED`) are now
  admitted on the §5.11.33 `compute_prediction` dispatcher's intra
  arm. `Error::ComputePredictionIntraModeUnsupported` is reachable
  only on a caller-bug `mode == UV_CFL_PRED == 13` arriving on the
  luma plane (the §5.11.7 / §5.11.22 readers cap at the legal range,
  so no conformant stream surfaces it).

  * **New `predict_intra_directional(w, h, p_angle, upsample_above,
    upsample_left, above_row, left_col, pred)`** — the unified §7.11.2.4
    sample-generation body (av1-spec p.245-248) for the six
    non-degenerate D-modes. Routes to step 7 (`pAngle < 90`, above-
    only projection), step 8 (`90 < pAngle < 180`, hybrid above/left
    projection), or step 9 (`pAngle > 180`, left-only projection)
    keyed by `pAngle`. Rejects `pAngle == 90` / `pAngle == 180` as
    caller-bug — those route through the dedicated r186
    [`predict_intra_v_pred`] / [`predict_intra_h_pred`] leaves.
  * **New `predict_intra_d_mode(mode, angle_delta, w, h,
    upsample_above, upsample_left, above_row, left_col, pred)`** —
    the §6.10.x ordinal-to-angle wrapper. Looks up
    `MODE_TO_ANGLE[ mode ] + angle_delta * ANGLE_STEP`, then
    delegates to `predict_intra_directional`. Rejects modes outside
    `D45_PRED..=D67_PRED` (the non-degenerate D-mode range).
  * **New `MODE_TO_ANGLE[ INTRA_MODES ]` table** — verbatim
    transcription from av1-spec p.485. Seven directional entries
    (V_PRED → 90, H_PRED → 180, D45 → 45, D135 → 135, D113 → 113,
    D157 → 157, D203 → 203, D67 → 67) + six non-directional entries
    (DC / SMOOTH / SMOOTH_V / SMOOTH_H / PAETH → 0).
  * **New `DR_INTRA_DERIVATIVE[ 90 ]` table** — verbatim
    transcription from av1-spec p.487. 27 non-zero per-angle
    derivatives (at indices 3, 6, 9, 14, 17, 20, 23, 26, 29, 32, 36,
    39, 42, 45, 48, 51, 54, 58, 61, 64, 67, 70, 73, 76, 81, 84, 87)
    used by §7.11.2.4 steps 5 / 6 to derive `dx` / `dy`.
  * **New `ANGLE_STEP = 3`** — per av1-spec p.485. Each
    `angle_delta_y` / `angle_delta_uv` unit moves `pAngle` by 3°.
  * **New `INTRA_EDGE_KERNEL[ 3 ][ 5 ]`** — verbatim transcription
    from av1-spec p.256: `[0, 4, 8, 4, 0]`, `[0, 5, 6, 5, 0]`,
    `[2, 4, 4, 4, 2]`. Each row sums to 16 (DC-preserving under
    `(s + 8) >> 4`).
  * **New `filter_corner(left_col_0, above_left, above_row_0)`** —
    §7.11.2.7 filter corner process (av1-spec p.252). Three-tap
    filter `(5, 6, 5) / 16` producing the new value for the top-left
    corner cell.
  * **New `intra_edge_filter_strength_selection(w, h, filter_type,
    delta)`** — §7.11.2.9 (av1-spec p.253). Returns one of
    `0..=3` driving the §7.11.2.12 filter. The `<= 12` / `<= 16`
    `blk_wh` branches are merged (identical bodies per spec).
  * **New `intra_edge_upsample_selection(w, h, filter_type,
    delta)`** — §7.11.2.10 (av1-spec p.254). Returns `0` or `1`
    deciding whether the §7.11.2.11 2x upsample pre-pass applies.
  * **New `intra_edge_upsample(num_px, bit_depth, buf)`** —
    §7.11.2.11 (av1-spec p.255). In-place 2x upsampling with the
    4-tap `(-1, 9, 9, -1) / 16` kernel; clips per `BitDepth`.
  * **New `intra_edge_filter(sz, strength, buf)`** — §7.11.2.12
    (av1-spec p.255-256). Applies the `Intra_Edge_Kernel[ strength
    - 1 ]` 5-tap filter; strength == 0 is a no-op early-return.
  * **Dispatcher gate widened to the full 13-mode set.** The
    §5.11.33 `PartitionWalker::compute_prediction` intra arm now
    admits every `plane_mode` in `0..INTRA_MODES = 0..13`.
    `ComputePredictionIntraModeUnsupported` is reachable only on
    `mode == UV_CFL_PRED == 13` arriving on the luma plane (a
    caller-bug since §5.11.22 caps luma at `INTRA_MODES`).
  * **New named §6.10.x intra-mode ordinals** — `D45_PRED = 3`,
    `D135_PRED = 4`, `D113_PRED = 5`, `D157_PRED = 6`,
    `D203_PRED = 7`. (`D67_PRED = 8` was already exported via the
    r180 boundary marker.)

  The r188 dispatcher runs the **no-upsample, no-filter** path of
  §7.11.2.4 step 4 (i.e. `enable_intra_edge_filter == 0` short-
  circuit + `useUpsample == 0` short-circuit). The §7.11.2.{9, 10,
  11, 12} helpers are exposed as standalone functions so a future
  arc can stitch the pre-passes into the dispatcher; for the path
  the §5.11.5 walker emits today the un-upsampled / un-filtered
  edges produce the correct shape (lower quality, but spec-conformant
  for the non-`enable_intra_edge_filter` profile).

  Test count: 914 → 935 (+21). New tests: `MODE_TO_ANGLE` length +
  13-cell value pin; `DR_INTRA_DERIVATIVE` length + 27 non-zero
  index/value pins + 63 zero-cell pin; `INTRA_EDGE_KERNEL` three-row
  transcription pin + per-row DC-preserving sum pin; `filter_corner`
  constant-input identity + asymmetric mid-point hand-trace;
  `intra_edge_upsample_selection` truth-table spot-checks across
  both `filter_type` values + boundary conditions; `intra_edge_
  filter_strength_selection` truth-table spot-checks for the
  `blk_wh` < 8 / <= 16 / <= 24 / <= 32 / > 32 branches at both
  `filter_type` values; `intra_edge_filter` strength-0 no-op +
  constant-input DC-preservation across strengths 1 / 2 / 3;
  `intra_edge_upsample` constant-input DC-preservation;
  `predict_intra_directional` D45 / D135 / D203 constant-neighbour
  identity + D45 ramp diagonal-copy hand-trace;
  `predict_intra_d_mode` six-D-mode dispatch via the §6.10.x ordinal
  + caller-bug guards on out-of-range mode / out-of-range
  angle_delta / degenerate V_PRED / H_PRED rejection;
  `predict_intra_directional` caller-bug guards on `pAngle == 90` /
  `pAngle == 180` / out-of-range `pAngle` / `w == 0` / `h > 64` /
  upsample flag > 1 / short pred buffer; `intra_edge_filter` /
  `intra_edge_upsample` caller-bug guards; dispatcher acceptance
  for each of the six D-modes (`3..=8`) on a 3-plane 4:2:0 block;
  dispatcher rejection of `UV_CFL_PRED == 13` on luma; dispatcher
  rejection of out-of-range mode `== 14` as caller-bug.

  `decode_av1` / `encode_av1` still return `Error::NotImplemented`.

* **Round 187 — §7.11.2.2 PAETH + §7.11.2.6 SMOOTH / SMOOTH_V /
  SMOOTH_H sample-generation leaves + §7.11.2.1 `AboveRow[-1]`
  corner-sample derivation.** Extends the §5.11.33
  `compute_prediction` dispatcher's intra arm from
  `{DC_PRED, V_PRED, H_PRED}` (r186) to seven of thirteen Y intra
  modes — the four "smooth" predictors that share neighbour-array
  infrastructure with V_PRED / H_PRED, plus the §7.11.2.2 PAETH
  basic intra prediction that introduces the corner-cell read.

  * **New `derive_above_left(have_above, have_left, x, y, curr_frame,
    curr_frame_cols, bit_depth) -> u16`** — §7.11.2.1 corner-cell
    derivation (av1-spec p.242). Four-arm dispatch on
    `(haveAbove, haveLeft)`: `(1, 1)` ⇒
    `CurrFrame[plane][y-1][x-1]`; `(1, 0)` ⇒
    `CurrFrame[plane][y-1][x]`; `(0, 1)` ⇒
    `CurrFrame[plane][y][x-1]`; `(0, 0)` ⇒ `1 << (BitDepth - 1)`
    (the symmetric mid-grey value, NOT the `±1`-offset asymmetric
    values arms 1 / 2 of `derive_above_row` / `derive_left_col`
    use). The spec sets `LeftCol[-1] = AboveRow[-1]`, so a single
    helper covers both.
  * **New `predict_intra_paeth_pred(w, h, above_row, left_col,
    above_left, pred)`** — §7.11.2.2 basic intra prediction process
    (av1-spec p.243). Per-cell, computes `base = AboveRow[j] +
    LeftCol[i] - AboveRow[-1]` and the three absolute deltas
    `pLeft`, `pTop`, `pTopLeft`; tie-break is
    `LeftCol[i]` (first arm) > `AboveRow[j]` (second arm) >
    `AboveRow[-1]` (third arm). Reachable when `mode == PAETH_PRED
    == 12`.
  * **New `predict_intra_smooth_pred(log2_w, log2_h, w, h,
    above_row, left_col, pred)`** — §7.11.2.6 SMOOTH_PRED arm
    (av1-spec p.250). 4-tap bidirectional blend
    `smoothPred = smWeightsY[i] * AboveRow[j] + (256 -
    smWeightsY[i]) * LeftCol[h-1] + smWeightsX[j] * LeftCol[i] +
    (256 - smWeightsX[j]) * AboveRow[w-1]`, then `Round2(_, 9)`.
    `smWeightsX = Sm_Weights_Tx_{1 << log2_w}x...`, similarly
    `smWeightsY`. Note the spec uses `LeftCol[h-1]` and
    `AboveRow[w-1]` (the LAST entries of the per-edge neighbour
    arrays, NOT the corner cell `AboveRow[-1]`).
  * **New `predict_intra_smooth_v_pred(log2_h, w, h, above_row,
    left_col, pred)`** — §7.11.2.6 SMOOTH_V_PRED arm
    (av1-spec p.250-251). Vertical-only 2-tap blend
    `smWeights[i] * AboveRow[j] + (256 - smWeights[i]) *
    LeftCol[h-1]`, then `Round2(_, 8)`.
  * **New `predict_intra_smooth_h_pred(log2_w, w, h, above_row,
    left_col, pred)`** — §7.11.2.6 SMOOTH_H_PRED arm
    (av1-spec p.251). Horizontal-only 2-tap blend
    `smWeights[j] * LeftCol[i] + (256 - smWeights[j]) *
    AboveRow[w-1]`, then `Round2(_, 8)`.
  * **New `Sm_Weights_Tx_{4x4, 8x8, 16x16, 32x32, 64x64}` tables**
    — verbatim transcription from av1-spec p.508. Five size
    variants cover the §3 transform-size set; indexed by
    `log2_n - 2`.
  * **Dispatcher gate widened.** The §5.11.33
    `PartitionWalker::compute_prediction` intra arm now admits
    `mode ∈ {DC_PRED, V_PRED, H_PRED, SMOOTH_PRED, SMOOTH_V_PRED,
    SMOOTH_H_PRED, PAETH_PRED}` (seven of thirteen Y intra modes).
    `ComputePredictionIntraModeUnsupported` still fires for the
    remaining six directional D-modes (D45_PRED / D135_PRED /
    D113_PRED / D157_PRED / D203_PRED / D67_PRED — ordinals
    `3..=8`) — next-arc targets.
  * **New named §6.10.x intra-mode ordinals** —
    `SMOOTH_PRED = 9`, `SMOOTH_V_PRED = 10`, `SMOOTH_H_PRED = 11`,
    `PAETH_PRED = 12` exported as `pub const`.

  Test count: 885 → 914 (+29). New tests cover: SMOOTH / PAETH
  ordinal pins; `Sm_Weights_Tx_*` boundary-value transcription
  pins; `derive_above_left` four-arm dispatch (top-left
  diagonal / above-only / left-only / no-neighbour mid-grey for 8 /
  10 / 12-bit `BitDepth`); negative + over-range sample clipping
  to `[0, (1 << BitDepth) - 1]`; PAETH on constant-neighbour
  blocks (all three deltas zero ⇒ first-arm wins); PAETH
  degenerating to H_PRED when `AboveRow[j] == AboveRow[-1]`;
  PAETH degenerating to V_PRED when `LeftCol[i] == AboveRow[-1]`;
  PAETH first-arm tie-break (`pLeft == pTop`); PAETH third-arm
  hand-trace where `pTopLeft` is the unique minimum; PAETH
  caller-bug guards; SMOOTH_PRED constant-neighbour produces
  constant block; SMOOTH_PRED corner cell hand-trace (`pred[0][0]
  = (255*100 + 200 + 255*200 + 100 + 256) >> 9 = 150`);
  SMOOTH_PRED bottom-right hand-trace; SMOOTH_PRED 64×64 max-size
  12-bit ceiling round-trip; SMOOTH_V top-row / bottom-row
  weighting asymmetry; SMOOTH_H left-col / right-col weighting
  asymmetry; all-three SMOOTH leaves' caller-bug guards;
  dispatcher acceptance of SMOOTH_PRED / SMOOTH_V_PRED /
  SMOOTH_H_PRED / PAETH_PRED with task-list mode forwarding on a
  3-plane 4:2:0 block; dispatcher rejection of D45 / D135 / D67
  at the D-mode-interval boundaries; end-to-end PAETH and
  SMOOTH_PRED through the §7.11.2.1 derivation helpers against a
  real `CurrFrame[plane]`-shaped buffer.

  The dispatcher itself still returns a per-plane task list rather
  than invoking the prediction kernels directly; the walker
  integration (driving the leaves end-to-end through the walker's
  `CurrFrame[plane]` buffer) lands when the §5.11.5 driver has
  the §5.11.6 / §5.11.5 mode-info read fully wired. The six
  directional D-modes (`D45_PRED` / `D135_PRED` / `D113_PRED` /
  `D157_PRED` / `D203_PRED` / `D67_PRED`) — the §7.11.2.4 non-
  degenerate body with `Dr_Intra_Derivative[]` driven sample
  projection — are the next-arc target.

* **Round 186 — §7.11.2.4 V_PRED + H_PRED sample-generation leaves
  + §7.11.2.1 `AboveRow[]` / `LeftCol[]` neighbour derivation.**
  Extends the §5.11.33 `compute_prediction` dispatcher's intra arm
  from DC_PRED-only (r180) to {DC_PRED, V_PRED, H_PRED} — the two
  cheapest non-DC intra modes, covering the §7.11.2.4 degenerate
  directional cases (`pAngle == 90` for V_PRED, `pAngle == 180` for
  H_PRED) where the directional process collapses to a row /
  column broadcast.

  * **New `predict_intra_v_pred(w, h, above_row, pred)`** —
    §7.11.2.4 step 10 (av1-spec p.247). Fills `pred[i][j] =
    AboveRow[j]` for `i ∈ 0..h`, `j ∈ 0..w` (each row of the
    block is a copy of the top-edge sample at column `j`).
    Reachable when `mode == V_PRED == 1` and `angleDelta == 0`
    (the §5.11.5 walker's default on the
    `decode_intra_block_mode_info` path).
  * **New `predict_intra_h_pred(w, h, left_col, pred)`** —
    §7.11.2.4 step 11. Mirror of V_PRED: `pred[i][j] =
    LeftCol[i]` for `i ∈ 0..h`, `j ∈ 0..w` (each column of the
    block is a copy of the left-edge sample at row `i`).
  * **New `derive_above_row(have_above, have_left,
    have_above_right, x, y, w, h, max_x, curr_frame,
    curr_frame_cols, bit_depth, above_row)`** — §7.11.2.1
    prologue arms 1-3 (av1-spec p.241). Reads `CurrFrame[plane]`
    at the per-TU `(x, y)` top-left to populate the
    `AboveRow[0..w+h-1]` neighbour array, with the spec's
    `haveAboveRight ? 2*w : w` tail extension and the `aboveLimit
    = Min(maxX, x + extend - 1)` frame-right-boundary clamp. On
    the `haveAbove == 0 && haveLeft == 1` arm the array is
    constant-propagated from `CurrFrame[plane][y][x-1]`; on the
    all-unavailable arm it is filled with `(1 << (BitDepth - 1))
    - 1` (mid-grey minus one — the spec's asymmetric
    `AboveRow[]` no-neighbour value).
  * **New `derive_left_col(have_left, have_above,
    have_below_left, x, y, w, h, max_y, curr_frame,
    curr_frame_cols, bit_depth, left_col)`** — §7.11.2.1
    prologue mirror for `LeftCol[0..w+h-1]`. `haveBelowLeft ?
    2*h : h` tail extension, `leftLimit = Min(maxY, y + extend -
    1)` bottom-boundary clamp, `(1 << (BitDepth - 1)) + 1`
    (mid-grey plus one — the spec's asymmetric `LeftCol[]`
    no-neighbour value) all-unavailable fill.
  * **Dispatcher gate narrowed.** The §5.11.33
    `PartitionWalker::compute_prediction` intra arm now admits
    `mode ∈ {DC_PRED, V_PRED, H_PRED}` instead of DC_PRED only.
    `ComputePredictionIntraModeUnsupported` still fires for the
    remaining 10 intra modes (SMOOTH_PRED / SMOOTH_V_PRED /
    SMOOTH_H_PRED / PAETH_PRED / D45_PRED / D135_PRED /
    D113_PRED / D157_PRED / D203_PRED / D67_PRED) — next-arc
    targets.
  * **New named §6.10.x intra-mode ordinals** — `DC_PRED = 0` /
    `H_PRED = 2` exported as `pub const`. `V_PRED = 1` already
    existed via the §8.3.2 angle-delta selector base. Pins the
    constants the dispatcher gate consults so callers can route
    on named values instead of magic numbers.

  Test count: 859 → 885 (+26). The new tests cover V_PRED row
  broadcast on 4×4 / 8×4 / 64×64 blocks; H_PRED column broadcast
  on 4×4 / 4×8 / 64×64 blocks; both leaves ignore neighbour-array
  slots past `w` / `h` (the §7.11.2.1 prologue derives `w + h`
  samples but step-10 / step-11 only consume the first `w` /
  `h`); caller-bug guards on both leaves; `AboveRow[]` arms 1-3
  (top-row read, `haveAboveRight` tail extension, `max_x` clamp,
  left-only propagation, all-unavailable mid-grey-minus-one);
  `LeftCol[]` arms 1-3 (left-col read, `haveBelowLeft` tail
  extension, above-only propagation, all-unavailable
  mid-grey-plus-one for 8-bit and 10-bit `BitDepth`); end-to-end
  `derive_above_row` → `predict_intra_v_pred` and
  `derive_left_col` → `predict_intra_h_pred` round trips through
  the §7.11.2.1 prologue against a real `CurrFrame[plane]`-shaped
  buffer; dispatcher acceptance of V_PRED and H_PRED with task-
  list mode forwarding on a 3-plane 4:2:0 block; dispatcher
  rejection of D45_PRED / SMOOTH_PRED / PAETH_PRED at the
  next-arc-gap boundary; intra-mode ordinal pins.

  The §7.11.2.1 corner-sample derivation (`AboveRow[-1]` /
  `LeftCol[-1]`) is not landed in r186 — V_PRED and H_PRED do not
  consult it. It is the only PAETH_PRED dependency on the
  degenerate-angle subset and lands with the PAETH leaf in a
  later arc. The dispatcher itself still returns a per-plane
  task list rather than invoking the prediction kernels directly:
  the walker integration (driving the leaves end-to-end through
  the walker's `CurrFrame[plane]` buffer) lands when the §5.11.5
  walker starts emitting non-DC_PRED `YMode` cells.

* **Round 185 — §7.12.3 step-3 frame-buffer merge + `CurrFrame`
  per-plane sample buffers.** Lifts the last remaining placeholder
  in the §7.12.3 reconstruct chain: with r185 the per-TU
  inverse-transform residual the r182 walker captured is merged
  into a real per-plane `CurrFrame[plane][y][x]` sample buffer
  with the spec's `flipUD` / `flipLR` destination remap for the
  FLIPADST family and a `Clip1` envelope per `BitDepth`.

  * **New `PartitionWalker` per-plane sample buffers.** Three
    `Option<CurrFramePlane>` slots (Y / Cb / Cr) on the walker,
    allocated lazily on the first §7.12.3 step-3 merge call
    touching a plane. Per-plane dimensions follow §5.9.5 /
    §5.11.34: luma is `MiRows * MI_SIZE × MiCols * MI_SIZE`
    samples; chroma is `>> sub_y` / `>> sub_x` per the §5.11.34
    line 19 `subX` / `subY` derivation (so 4:2:0 halves both
    axes). Initial fill is zero — the spec's `Clip1(0 +
    Residual[i][j])` additive identity before any §7.11.x
    prediction-sample writer fires.
  * **New `PartitionWalker::curr_frame(plane: usize) ->
    Option<&[i32]>`** — row-major sample view (`plane` ∈
    `0..3`). Returns `None` until the first §7.12.3 step-3
    merge call on that plane has allocated the buffer.
  * **New `PartitionWalker::curr_frame_dims(plane: usize) ->
    Option<(u32, u32)>`** — `(rows, cols)` companion to
    `curr_frame`.
  * **`PartitionWalker::transform_block_emit` now applies the
    §7.12.3 step-3 merge on the `!skip && eob > 0` arm** —
    immediately after the r182 §7.13 inverse transform produces
    the per-TU `Residual[][]`, and before recording the residual
    onto `ResidualReadout::residuals`. Driven directly from the
    `ResidualContext`'s `QuantizerParams::bit_depth` and the
    r183 §5.11.40 `compute_tx_type` derivation of `PlaneTxType`.
  * **`ResidualTuTask` gains a `plane_tx_type: u8` field** — the
    per-TU §5.11.40 `compute_tx_type` outcome the step-3 merge
    consults. Back-filled by the dispatcher on the `!skip` arm
    after the §5.11.40 derivation; the `skip == true` arm
    leaves it at the `DCT_DCT` placeholder (no transform fired,
    the step-3 merge is gated to the prediction-only write —
    DCT_DCT's neither-flip identity).
  * **`Clip1(x) = Clip3(0, (1 << BitDepth) - 1, x)`** per §3
    `Clip1`. The merge supports `BitDepth ∈ {8, 10, 12}`; the
    pre-`Clip1` `prediction + Residual[i][j]` sum is `i64`-wide
    so 12-bit-plus-large-residual edge cases don't overflow.
  * **Out-of-buffer overhang is silently clipped.** A TU whose
    `start_x + w > cols` or `start_y + h > rows` (possible at
    the frame boundary per §5.11.34's chunk loop) lands its
    in-buffer slice and drops the rest. Plane-out-of-range and
    wrong-residual-size arguments are defensive no-ops.

  16 new tests (843 → 859): unallocated-on-fresh-walker accessor
  behaviour; `ensure_curr_frame_plane` per-subsampling sizing
  (luma 16×16, 4:2:0 chroma 8×8 on the 4-mi-square fixture);
  DCT_DCT identity write across a 4×4 TU; `Clip1` 8-bit
  saturation at both the upper (200 + 200 → 255) and lower
  (100 - 150 → 0) envelopes; `Clip1` 10-bit saturation (upper
  bound 1023); FLIPADST_DCT row-flip (dst row `r` reads src row
  `3 - r`); H_FLIPADST col-flip (dst col `c` reads src col `3 -
  c`); FLIPADST_FLIPADST 180° rotation; offset top-left
  placement; additive accumulation across two consecutive
  merges; silent overhang clip; plane-out-of-range no-op;
  wrong-residual-size no-op; chroma 4:2:0 plane allocation with
  luma / V left unallocated; `ResidualTuTask::plane_tx_type`
  back-fill on the walker's reachable intra-only path (every TU
  carries DCT_DCT after the §5.11.40 derivation on the
  `YMode = DC_PRED` fixture); skip-arm `plane_tx_type` stays at
  the DCT_DCT placeholder. `decode_av1` / `encode_av1` continue
  to return `Error::NotImplemented`.

* **Round 184 — §7.5 / §5.11.41 `get_scan(txSz)` table dispatch.**
  Replaces the r183 identity-ascending scan placeholder in
  `transform_block_emit`'s call into `coefficients()` with the
  full §5.11.41 dispatcher.

  * New `oxideav_av1::scan` module (`pub mod scan` in `lib.rs`)
    with the full §7.5 scan-table family from av1-spec p.388-399:
    * **Default scans (14 tables):** `DEFAULT_SCAN_4X4`,
      `_4X8`, `_8X4`, `_8X8`, `_8X16`, `_16X8`, `_16X16`,
      `_16X32`, `_32X16`, `_32X32`, `_4X16`, `_16X4`, `_8X32`,
      `_32X8` (16..1024 entries; the zig-zag positions the
      §5.11.39 reader walks for symmetric `PlaneTxType`
      values).
    * **Mcol scans (9 tables):** `MCOL_SCAN_4X4`, `_4X8`,
      `_8X4`, `_8X8`, `_8X16`, `_16X8`, `_16X16`, `_4X16`,
      `_16X4` (column-major rasters for `H_DCT` / `H_ADST` /
      `H_FLIPADST`; only sizes whose `Tx_Size_Sqr_Up <=
      TX_16X16` are admitted by the §5.11.40 / §5.11.48 set
      restrictions, matching the spec enumeration).
    * **Mrow scans (9 tables):** `MROW_SCAN_4X4`, `_4X8`,
      `_8X4`, `_8X8`, `_8X16`, `_16X8`, `_16X16`, `_4X16`,
      `_16X4` (row-major identity rasters for `V_DCT` /
      `V_ADST` / `V_FLIPADST`).
  * New `oxideav_av1::scan::{get_default_scan, get_mcol_scan,
    get_mrow_scan, get_scan}` — the four §5.11.41 dispatchers
    (av1-spec p.93-94). `get_scan( txSz, PlaneTxType )` routes
    per the spec's three short-circuits (`TX_16X64` →
    `Default_Scan_16x32`, `TX_64X16` → `Default_Scan_32x16`,
    `Tx_Size_Sqr_Up[txSz] == TX_64X64` → `Default_Scan_32x32`),
    then the `IDTX` → default arm, then the `preferRow` arm
    (V_DCT / V_ADST / V_FLIPADST → `get_mrow_scan`), the
    `preferCol` arm (H_DCT / H_ADST / H_FLIPADST →
    `get_mcol_scan`), and finally the symmetric tx-type default
    arm.
  * `PartitionWalker::transform_block_emit` (the §5.11.35 per-TU
    driver) now invokes `scan::get_scan(tx_sz, plane_tx_type)`
    instead of constructing a `Vec<u16>` of `0..segEob` per
    iteration. The returned `&'static [u16]` is passed straight
    to `self.coefficients(..., scan, &mut quant)`. The stale
    `seg_eob` derivation in `transform_block_emit` (only used
    by the placeholder) is dropped; the §5.11.39 reader still
    re-derives it internally from `tx_sz`.

  22 new tests (821 → 843): permutation checks for every default
  / Mcol / Mrow table (each scan visits every coefficient
  position exactly once); identity check for every Mrow table;
  `col * h + row → row * w + col` column-major formula check for
  every Mcol table; spec-anchor 4x4 / 8x8 default-scan
  prologue values; `get_default_scan` / `get_mcol_scan` /
  `get_mrow_scan` per-size enumeration including spec-tail
  fallback (TX_32X32 → `DEFAULT_SCAN_32X32`, anything not in
  Mcol's / Mrow's listed sizes → `MCOL_SCAN_16X4` /
  `MROW_SCAN_16X4`); `get_scan` arm-by-arm coverage
  (`TX_16X64` / `TX_64X16` / `Tx_Size_Sqr_Up == TX_64X64` /
  IDTX / preferRow / preferCol / symmetric) under every
  representative `PlaneTxType`; and `scan.len() >= segEob` over
  the §5.11.40-admitted `(tx_sz, plane_tx_type)` reachable
  subset for every transform size in `TX_SIZES_ALL`.

  **Split off explicitly to subsequent arcs (unchanged from r183):**
  (a) §9.5.3 `Quantizer_Matrix[15][2][QM_TOTAL_SIZE]` table
  transcription; (b) §7.12.3 step-3 frame-buffer merge; (c) 12
  non-DC intra-prediction modes; (d) §7.9 inter prediction.

  `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

* **Round 183 — §7.12.2 dequantization-function tables + §7.12.3
  step-1 dequantization loop + §5.11.47 `transform_type` per-TU
  S() reader.** Replaces the r182 placeholder identity dequant +
  hard-coded `DCT_DCT` per-TU in the `residual()` walker path
  with a real per-plane / per-segment quantization pipeline.

  * New `oxideav_av1::{DC_QLOOKUP, AC_QLOOKUP, dc_q, ac_q}` —
    the `Dc_Qlookup[3][256]` / `Ac_Qlookup[3][256]` per-bit-depth
    quantizer index tables (av1-spec p.289-292, verbatim) plus
    the `(BitDepth-8) >> 1` row-select + `Clip3(0, 255, b)`
    column-clamp helpers from §7.12.2.
  * New `oxideav_av1::{QuantizerParams, get_qindex, get_dc_quant,
    get_ac_quant}` — `QuantizerParams` aggregates the §5.9.12
    `base_q_idx` / four `delta_q_*` slots, §5.9.18
    `delta_q_present` / `current_q_index`, §5.9.14
    `SEG_LVL_ALT_Q` `FeatureEnabled[]` / `FeatureData[]` tables,
    §5.9.12 `using_qmatrix`, and §6.4.1 `BitDepth`. The three
    accessors mirror av1-spec p.293-294 (segmentation arm +
    delta_q arm + `base_q_idx` fallback; per-plane DC / AC delta
    routing).
  * New `oxideav_av1::{dequant_denom, dequantize_step1}` — the
    §7.12.3 step-1 loop (av1-spec p.294-295): per-coefficient
    `q = (i,j == 0,0) ? get_dc_quant : get_ac_quant`,
    `dq = Quant * q`, `sign = (dq < 0) ? -1 : 1`,
    `dq2 = sign * (|dq| & 0xFFFFFF) / dqDenom`,
    `Dequant[i][j] = Clip3(-(1<<(7+BitDepth)),
    (1<<(7+BitDepth)) - 1, dq2)`, with `tw = Min(32, w)` /
    `th = Min(32, h)` and `dqDenom = 4 / 2 / 1` per the size.
    The §9.5.3 `Quantizer_Matrix[15][2][QM_TOTAL_SIZE]` table
    is split-off (~30 KB); the `using_qmatrix && plane_tx_type
    < IDTX && seg_qm_level < 15` arm falls through to the no-QM
    identity (`q2 = q`) path with a debug-only assertion. The
    `seg_qm_level == 15` spec sentinel takes the no-QM branch
    directly.
  * New `oxideav_av1::{TX_TYPE_INTRA_INV_SET1,
    TX_TYPE_INTRA_INV_SET2, TX_TYPE_INTER_INV_SET1,
    TX_TYPE_INTER_INV_SET2, TX_TYPE_INTER_INV_SET3}` — the
    §5.11.47 spec inversion tables verbatim (`{ IDTX, DCT_DCT,
    V_DCT, ... }` 16/12/7/5/2 entries).
  * New `PartitionWalker::transform_type(decoder, cdfs, tx_size,
    is_inter, intra_dir, reduced_tx_set, segment_id, &quant)` —
    the §5.11.47 per-TU luma S() read against
    `TileIntraTxTypeSet{1,2}Cdf` / `TileInterTxTypeSet{1,2,3}Cdf`
    (routed via the existing `inter_tx_type_cdf` /
    `intra_tx_type_cdf` selectors) with the spec's `set > 0 &&
    (segmentation_enabled ? get_qindex(1, segment_id) :
    base_q_idx) > 0` guard. Returns a `TransformTypeReadout {
    tx_type, w4, h4 }`.
  * New `PartitionWalker::{tx_types(), tx_type_at(r, c),
    stamp_tx_type(x4, y4, w4, h4, tx_type)}` — the §5.11.39
    / §5.11.47 `TxTypes[][]` luma cache accessors. A new
    walker-level `tx_types: Vec<u8>` grid (one cell per
    4×4-luma-sample / mi unit) tracks the cache; `stamp_tx_type`
    writes a `(w4, h4)` footprint with edge-clipping.
  * New `oxideav_av1::ResidualContext` — per-block aggregate
    (`QuantizerParams`, `reduced_tx_set`, `intra_dir`,
    `segment_id`, `seg_qm_level[3]`, `uv_mode`) the §5.11.34
    `residual()` dispatcher consumes. Use
    `ResidualContext::neutral(base_q_idx, bit_depth)` for the
    "all defaults" path.

  Wired into `PartitionWalker::residual` /
  `transform_block_emit`: the dispatcher now, on the `!skip` arm,
  runs `transform_type` (luma only), `compute_tx_type` (per
  plane), `get_tx_class(plane_tx_type)`, then `dequantize_step1`
  in order — replacing the r182 hard-coded `DCT_DCT` + identity
  dequant. The §7.13 `inverse_transform_2d` call now receives the
  per-plane `PlaneTxType` and `BitDepth` from `ResidualContext`.
  The §5.11.5 `decode_block_syntax` walker synthesises a neutral
  `ResidualContext` (`base_q_idx = 0`, no segmentation,
  `intra_dir = y_mode`, `uv_mode = DC_PRED`, no QM) so the
  reachable walker path exercises the §5.11.47 `q_for_guard ==
  0 ⇒ DCT_DCT` short-circuit — every fixture in the corpus
  remains green.

  **Split-off explicitly:** (a) §9.5.3 `Quantizer_Matrix` table
  transcription; (b) §7.5 `get_scan` table dispatch; (c) §7.12.3
  step-3 frame-buffer merge; (d) 12 non-DC intra-prediction
  modes; (e) §7.9 inter prediction.

  20 new tests (801 → 821); `decode_av1` / `encode_av1` continue
  to return `Error::NotImplemented`.

* **Round 182 — §7.13 inverse transform process.** New
  `oxideav_av1::transform` module reproduces the AV1 §7.13 inverse-
  transform stack from the spec text directly: §7.13.2.1
  `butterfly_b` / `butterfly_h` / `cos128` / `sin128` /
  `brev` primitives driven by the spec's 65-entry
  `Cos128_Lookup[65]` quarter-period table; §7.13.2.2 inverse DCT
  permutation; §7.13.2.3 `inverse_dct(t, n, r)` for `n in 2..=6`
  (sizes 4 / 8 / 16 / 32 / 64; 31-step `B`/`H` butterfly schedule
  reproduced verbatim); §7.13.2.4 / §7.13.2.5 inverse ADST input
  + output Gray-code permutations; §7.13.2.6 `inverse_adst4`
  closed-form with the `SINPI_1_9 = 1321` /  `SINPI_2_9 = 2482` /
  `SINPI_3_9 = 3344` / `SINPI_4_9 = 3803` constants; §7.13.2.7
  `inverse_adst8` 7-step butterfly schedule; §7.13.2.8
  `inverse_adst16` 9-step butterfly schedule; §7.13.2.9
  `inverse_adst` dispatcher; §7.13.2.10 `inverse_wht4` (Lossless
  4-point Walsh-Hadamard); §7.13.2.11..§7.13.2.14 inverse identity
  4 / 8 / 16 / 32 (with the spec's `5793` / `11586` scaling pairs
  and `*2` / `*4` doublings); §7.13.2.15 `inverse_identity`
  dispatcher.

  The §7.13.3 2D dispatcher `inverse_transform_2d(dequant, tx_sz,
  tx_type, bit_depth, lossless)` returns the row-major `w * h`
  `Residual[][]` buffer. Implements the full row-then-column
  composition over the 16-element `PlaneTxType` matrix (DCT_DCT
  through H_FLIPADST per §6.10.19), the `Abs(log2W - log2H) == 1`
  rectangular `Round2(t * 2896, 12)` pre-scale, the
  `Transform_Row_Shift[TX_SIZES_ALL]` per-size row right-shift, the
  between-stage `Clip3` envelope at `colClampRange = Max(BitDepth +
  6, 16)` bits, and the `Lossless` short-circuit through WHT.

  Wired into `PartitionWalker::residual` / `transform_block_emit`:
  on every TU with `eob > 0` the walker invokes
  `inverse_transform_2d` (passing `Quant[]` directly as a
  placeholder identity dequant — the §7.12.3 step-1 true
  dequantization is split-off) and records the per-TU `Residual`
  on a new `ResidualReadout::residuals: Vec<Option<Vec<i64>>>`
  field (aligned to `tasks` length). The walker now returns
  `Ok(DecodedBlock)` cleanly on every §5.11.5 path it reaches —
  **lifting `Error::ResidualReconstructUnsupported`** (variant
  retained as a defensive caller-bug sentinel).

  Split-off explicitly: (1) §7.12.3 step-1 true dequantization
  with `get_dc_quant` / `get_ac_quant` / `Quantizer_Matrix[
  SegQMLevel ][ ... ]`; (2) §7.12.3 step-3 frame-buffer merge
  with `flipLR` / `flipUD` destination remap on
  `CurrFrame[plane][y + yy][x + xx]`; (3) §5.11.40 / §5.11.41
  per-TU `PlaneTxType` read (still defaults to `DCT_DCT`); (4)
  §7.5 `get_scan` table dispatch; (5) `AboveLevelContext` /
  `LeftLevelContext` neighbour grids; (6) 10/12-bit `BitDepth`
  (hardcoded to `8`).

  Public surface added: `transform::inverse_transform_2d`,
  `transform::inverse_dct` / `inverse_dct_permute`,
  `transform::inverse_adst` / `inverse_adst4` / `inverse_adst8` /
  `inverse_adst16` / `inverse_adst_input_permute` /
  `inverse_adst_output_permute`, `transform::inverse_wht4`,
  `transform::inverse_identity` / `inverse_identity4` /
  `inverse_identity8` / `inverse_identity16` / `inverse_identity32`,
  `transform::butterfly_b` / `butterfly_h`, `transform::cos128` /
  `sin128` / `brev` / `round2` / `clip3`,
  `transform::COS128_LOOKUP`, `transform::SINPI_1_9..SINPI_4_9`,
  `transform::TRANSFORM_ROW_SHIFT`. `ResidualReadout` gains
  `residuals: Vec<Option<Vec<i64>>>`.

  33 new tests (768 → 801): twelve on the §7.13.2 1D primitives
  (cos128 anchors at 0/32/64/96/128/160/192/224, sin128 ==
  cos128(a-64) over the full period, brev, round2 sign-handling,
  DCT permute n=2, ADST input permute n=3, zero-input
  zero-preservation for IDCT 4 / 8 / 16 / 32 / 64 and IADST 4 / 8
  / 16, identity 4 / 8 / 16 / 32 scaling with bit-exact
  expectations, identity dispatcher); eight on the §7.13.3 2D
  dispatcher (DCT_DCT 4×4 DC-only bit-exact = `128` per cell,
  DCT_DCT zero-preserves, IDTX 4×4 DC-only bit-exact = `512` at
  origin, Lossless zero-preserves, TX_4X8 rectangular
  zero-preserves, 8×8 / 16×16 / 32×32 / 64×64 zero-preserves,
  FLIPADST_FLIPADST and the six V_ / H_ variants zero-preserve);
  one IDCT4 linearity check; one
  `residual_no_skip_eob_positive_invokes_inverse_transform`
  walker-integration test asserting `Some(Residual)` of length
  `Tx_Width * Tx_Height` per gate-open TU. The seven existing
  `decode_block_syntax` integration tests previously asserting
  `Err(ResidualReconstructUnsupported)` now assert
  `Ok(DecodedBlock)`. `decode_av1` / `encode_av1` continue to
  return `Error::NotImplemented`.

* **Round 181 — §5.11.34 `residual()` outer dispatch + §5.11.36
  `transform_tree` recursion + per-TU §5.11.39 wiring.** Lifts the
  §5.11.5 walker's long-standing `DecodeBlockResidualUnsupported`
  stub by landing the §5.11.34 outer dispatcher per av1-spec p.84-85.
  Exposed as `PartitionWalker::residual(decoder, cdfs, mi_row, mi_col,
  mi_size, has_chroma, subsampling_x, subsampling_y, is_inter,
  lossless, skip, tx_size, use_128x128_superblock)` returning a new
  `ResidualReadout { width_chunks, height_chunks, mi_size_chunk,
  num_planes_visited, tasks, coeffs, skip }` aggregate. Per-TU tasks
  are surfaced through `Vec<ResidualTuTask { plane, start_x, start_y,
  tx_size, chunk_x, chunk_y, from_transform_tree }>` in §5.11.34 spec
  order (`chunkY` outer / `chunkX` next / `plane` next / per-plane
  `(y, x)` direct iteration inside `transform_block`; DFS leaf order
  for the inter-luma `transform_tree` arm). The `tasks.len() ==
  coeffs.len()` invariant holds on the `!skip` arm; on `skip == true`
  the `coeffs` vec is empty (per the §5.11.35 `if ( !skip ) coeffs(
  ... )` gate).

  Implemented bodies:

  * §5.11.34 outer `widthChunks` / `heightChunks` derivation
    (`Max( 1, Block_Width[ MiSize ] >> 6 )` for both axes).
  * §5.11.34 `miSizeChunk = ( widthChunks > 1 || heightChunks > 1 ) ?
    BLOCK_64X64 : MiSize` per av1-spec p.84 line 5.
  * §5.11.34 chunk-loop + per-plane loop bound to
    `1 + HasChroma * 2`.
  * §5.11.37 `get_tx_size(plane, txSz)` per-plane lookup with the
    `Tx_Width[ uvTx ] == 64 || Tx_Height[ uvTx ] == 64` chroma clamp
    arm.
  * §5.11.38 `get_plane_residual_size(miSizeChunk, plane)` for the
    per-plane `planeSz`.
  * §5.11.36 `transform_tree( startX, startY, w, h )` recursive
    split — `w <= lumaW && h <= lumaH` leaf-emit (with `find_tx_size(
    w, h )` resolution), `w > h` / `w < h` / `w == h` per-direction
    splits operating against the walker's `InterTxSizes[ row ][ col ]`
    grid.
  * §5.11.35 `transform_block(plane, baseX, baseY, txSz, x, y)`
    per-TU emit with the `startX = baseX + 4 * x` /
    `startY = baseY + 4 * y` derivation and the `startX >= maxX ||
    startY >= maxY` early-return.
  * §5.11.35 `if ( !skip ) eob = coeffs( ... )` invocation of the
    r179 §5.11.39 `PartitionWalker::coefficients` reader. The
    `if ( eob > 0 ) reconstruct(...)` arm surfaces
    `Error::ResidualReconstructUnsupported`.

  Wired into `decode_block_syntax`, lifting
  `Error::DecodeBlockResidualUnsupported`. The walker now reaches
  `Error::ResidualReconstructUnsupported` (the §7.6 inverse-
  transform + §7.13 dequant + §7.7 reconstruction merge) when the
  rigged-zero bitstream produces a non-`all_zero == 1` TU. On the
  `skip == true` arm and on the `all_zero == 1`-for-every-TU arm the
  walker returns through `decode_block_syntax` cleanly with the
  per-block `DecodedBlock`. The `DecodeBlockResidualUnsupported`
  variant is retained as a defensive caller-bug fallback.

  Explicit splits to subsequent arcs:

  * §5.11.40 `transform_type` per-luma-TU S() read into `TxTypes[]`
    (defaulted to `DCT_DCT`).
  * §5.11.41 `compute_tx_type` full body (defaulted to `DCT_DCT`).
  * §7.5 `get_scan` table dispatch over `Default_Scan_*` /
    `Mrow_Scan_*` / `Mcol_Scan_*` (defaulted to identity scan
    `0..segEob`).
  * `AboveLevelContext` / `LeftLevelContext` / `AboveDcContext` /
    `LeftDcContext` per-plane context arrays (passed as `0`
    clean-state context — exactly matches the first TU of every
    reachable site).
  * §5.11.35 `predict_intra` / `predict_palette` /
    `predict_chroma_from_luma` per-TU intra-prediction kernels.
  * §5.11.35 `reconstruct(plane, startX, startY, txSz)` — the §7.6
    inverse-transform kernel itself (the headline next-arc target).
  * §5.11.35 `LoopfilterTxSizes[ plane ][ row ][ col ] = txSz` /
    `BlockDecoded[ plane ][ row ][ col ] = 1` per-TU stamps.
  * §7.13 `Dequant[][]` per-`(plane, pos)` dequantization.

  New public surface: `PartitionWalker::residual`, `ResidualReadout`,
  `ResidualTuTask`, `get_tx_size`. New `Error` variants:
  `ResidualReconstructUnsupported`,
  `ResidualTransformTreeUnsupported`,
  `ResidualCoefficientsTxSizeUnsupported`.

  13 new tests (755 → 768): `get_tx_size` (plane-0 pass-through, 4:4:4
  chroma residual = luma residual, 4:2:0 BLOCK_64X64 / BLOCK_128X128
  size clamp, caller-bug guards); `residual` (skip-arm task-only
  emission with zero bits read, no-skip all_zero per-TU short-
  circuit, lossless `TX_4X4` forcing per plane, monochrome luma-only
  emission, BLOCK_128X128 4-chunk emission with `BLOCK_64X64`
  miSizeChunk, six caller-bug guards, inter-luma `transform_tree` 4-
  way recursion to `TX_4X4` leaves); `ResidualReadout` Clone / Debug
  / Eq derive smoke. The existing `decode_block_syntax` integration
  tests are updated to expect `Error::ResidualReconstructUnsupported`.
  `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

* **Round 180 — §5.11.30 / §5.11.33 `compute_prediction()` dispatcher
  + §7.11.2.5 DC intra prediction leaf.** Lifts the §5.11.5 walker's
  long-standing `DecodeBlockComputePredictionUnsupported` stub by
  landing the §5.11.33 per-plane prediction-task dispatcher. Exposed
  as `PartitionWalker::compute_prediction(mi_row, mi_col, mi_size,
  has_chroma, avail_u, avail_l, avail_u_chroma, avail_l_chroma,
  subsampling_x, subsampling_y, is_inter, y_mode, uv_mode,
  ref_frame_1_is_intra)` returning a new `ComputePredictionReadout {
  is_inter_intra, is_inter, num_planes_visited, tasks }` aggregate;
  per-plane tasks are surfaced through `Vec<PlanePredictionTask {
  plane, start_x, start_y, log2_w, log2_h, mode, have_left,
  have_above }>` in §5.11.33 spec order. Wired into
  `decode_block_syntax`, so the §5.11.5 walker now advances past
  §5.11.30 and reaches `Error::DecodeBlockResidualUnsupported`
  (§5.11.34 outer dispatch around r179's §5.11.39 inner body).

  Standalone leaf landed alongside: `predict_intra_dc_pred(have_left,
  have_above, log2_w, log2_h, w, h, bit_depth, above_row, left_col,
  pred)` per §7.11.2.5 (av1-spec p.248-249) — the four-arm dispatch
  on `(haveLeft, haveAbove)` for DC intra prediction with `Clip1`
  bit-depth saturation and the `1 << (BitDepth - 1)` mid-grey
  fallback. Operates on caller-supplied `u16` neighbour slices and
  flat row-major output buffer.

  Supporting infra: `SUBSAMPLED_SIZE[ BLOCK_SIZES ][ 2 ][ 2 ]`
  (av1-spec p.88 `Subsampled_Size` table) + `get_plane_residual_size(
  subsize, plane, subsampling_x, subsampling_y) -> Option<usize>`
  (§5.11.38). New error variants:
  `ComputePredictionInterUnsupported` (§7.9 motion-compensated inter
  prediction next-arc); `ComputePredictionInterIntraUnsupported`
  (§7.11.5 inter+intra blend); `ComputePredictionIntraModeUnsupported`
  (§7.11.2.2 / .3 / .4 / .6 / .7+ non-DC intra modes).

  `DecodeBlockComputePredictionUnsupported` is retained as a
  defensive caller-bug fallback (the dispatcher path no longer
  constructs it on the conformant `is_inter == 0`, `y_mode ==
  DC_PRED` arm). 18 new tests (737 → 755): 7 on
  `compute_prediction`, 6 on `predict_intra_dc_pred`, 3 on
  table / helper sanity (`SUBSAMPLED_SIZE` shape, plane-0
  pass-through, `Clip1`-bit-depth saturation at 8 / 10 / 12 bit),
  2 on `get_plane_residual_size` (spot-checks + out-of-range
  guards). Existing `decode_block_syntax` integration tests
  updated to expect `Error::DecodeBlockResidualUnsupported`.
  `decode_av1` / `encode_av1` still return `Error::NotImplemented`.

  **Split off explicitly for the next arc**: §7.9 `predict_inter`
  body; remaining 12 §7.11.2.x intra modes; §7.11.5 inter+intra
  blend; §5.11.34 outer dispatch (`widthChunks` / `heightChunks` /
  `transform_tree` / `transform_block` recursion against a real
  per-plane `CurrFrame` sample buffer).

* **Round 179 — §5.11.39 `coeffs( )` per-TU coefficient reader.** Lands
  the first body of the §5.11.34 `residual()` cascade — the gate to
  the entire transform-coefficient pipeline. Exposed as
  `PartitionWalker::coefficients(decoder, cdfs, plane, is_inter,
  tx_size, tx_class, txb_skip_ctx, dc_sign_ctx, scan, quant)` returning
  a new `CoefficientsReadout { all_zero, eob, cul_level, dc_category }`
  aggregate. The §5.11.34 outer dispatch + §5.11.30
  `compute_prediction()` walker site remain subsequent-arc targets; the
  `PartitionWalker::decode_block_syntax` walker still short-circuits at
  `DecodeBlockComputePredictionUnsupported` upstream of the new reader.

  Implemented surface: §5.11.39 lines 1-7 derived sizes (`txSzCtx` /
  `ptype` / `segEob` with TX_16X64 / TX_64X16 ⇒ 512 special case); the
  line-8 `Quant[ 0..segEob ]` zero-fill; the line-13 `all_zero` S()
  against `TileTxbSkipCdf[ txSzCtx ][ ctx ]` with early-out; the
  lines-19-37 `eob_pt_{16, 32, 64, 128, 256, 512, 1024}` S() dispatch
  via `eobMultisize`; the lines-39-55 `eob` derivation with
  `eob_extra` S() + `eob_extra_bit` L(1) loop; the lines-56-71
  reverse-scan reading `coeff_base_eob` / `coeff_base` against the
  `TileCoeffBase{Eob,}Cdf[ txSzCtx ][ ptype ][ ctx ]` rows (ctx from
  the existing `get_coeff_base_{eob,}ctx` / `get_br_ctx` helpers) with
  the `coeff_br` S() chain when `level > NUM_BASE_LEVELS`; the
  lines-73-100 forward-scan reading `dc_sign` S() at `c == 0` and
  `sign_bit` L(1) otherwise, the §5.11.39 golomb chain
  (`golomb_length_bit` L(1) do-while + `golomb_data_bit` L(1) loop)
  for `Quant[ pos ] > NUM_BASE_LEVELS + COEFF_BASE_RANGE`, the
  `dcCategory` derive, the 0xFFFFF clip, the `culLevel += Quant[ pos ]`
  accumulation, the sign apply, and the line-102 `Min(63, culLevel)`
  clamp.

  Deferred to subsequent arcs:
  - The §7.13 `Dequant[][]` dequantization step (true qmatrix
    dequantization) — the reader produces the raw signed `Quant[]`
    array, which is the §7.13 input.
  - The §5.11.34 outer `widthChunks` / `heightChunks` / `transform_tree`
    / `transform_block` dispatch.
  - The §5.11.40 `transform_type` S() read + `compute_tx_type` lookup
    (caller-supplied via `tx_class`).
  - The §7.5 `get_scan` table lookup (caller-supplied `scan`).
  - The `AboveLevelContext` / `LeftLevelContext` / `AboveDcContext` /
    `LeftDcContext` walker arrays (caller-supplied `txb_skip_ctx` /
    `dc_sign_ctx`; readout returns `cul_level` / `dc_category` for the
    caller to apply).

  New public types: `CoefficientsReadout` (re-exported through
  `oxideav_av1::CoefficientsReadout`). 6 new unit tests cover the
  gate-closed short-circuit, a gate-open smoke test asserting readout
  invariants, a CDF-adaptation cross-check on `txb_skip`, the seven
  caller-bug guards, the TX_16X64 `segEob = 512` boundary, and a
  square-tx-size sweep across all five TX_NxN ordinals. Test count: 731
  → 737 (+6). `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

* **Round 178 — §5.11.x `read_interpolation_filter` reader.** Lands
  the LAST leaf of the §5.11.23 inter cascade. The §5.11.23
  dispatcher now runs the entire `inter_block_mode_info()` body to
  completion (through the `interpolation_filter` arm at av1-spec
  p.74). The §5.11.18 dispatcher's `Ok(_)` arm continues to surface
  `InterBlockModeInfoUnsupported` pending the next-arc §5.11.34
  `residual()` lift and a follow-on refactor that lifts
  `DecodedInterBlockModeInfo` into `DecodedInterFrameModeInfo`.

  The reader composes two paths: `interpolation_filter == SWITCHABLE`
  runs the per-dir loop with the inner `needs_interp_filter( ) ?
  interp_filter S() : EIGHTTAP` branch (and the post-loop
  `!enable_dual_filter` mirror); the else arm forces both slots to
  the frame-header value with zero per-block bits.
  `needs_interp_filter()` is inlined: closes the gate on `skip_mode
  || motion_mode == LOCALWARP`, the two large-block GLOBALMV /
  GLOBAL_GLOBALMV gm_type-TRANSLATION paths, and defaults to `1`.

  The §8.3.2 ctx walk reuses the round-22 `interp_filter_ctx` helper,
  seeding it from a new `InterpFilters[r][c][dir]` walker grid (two
  slots per cell, pre-fill `EIGHTTAP`) stamped over the bh4 * bw4
  footprint on every inter block decode. The `interp_filters()`
  accessor surfaces it.

  Five new §6.8.9 named constants land: `EIGHTTAP = 0`,
  `EIGHTTAP_SMOOTH = 1`, `EIGHTTAP_SHARP = 2`, `BILINEAR = 3`,
  `SWITCHABLE = 4`. New `InterpolationFilterReadout` aggregate
  carries `interp_filter: [u8; 2]` (per-dir resolved ordinal) and
  `read_from_bitstream: [bool; 2]` (per-dir "did this slot fire an
  S()?" flag). `DecodedInterBlockModeInfo` gains an
  `interp_filter: InterpolationFilterReadout` field.
  `decode_inter_block_mode_info` and `decode_inter_frame_mode_info`
  gain two new caller-supplied scalars: `interpolation_filter: u8`
  (§5.9.10 frame-header value) and `enable_dual_filter: bool`
  (§5.5.2 sequence-header bit).

  10 new unit tests cover: §6.8.9 ordinal alignment; the
  non-switchable arm (both slots forced, no bits); the four
  `needs_interp_filter == 0` short-circuit paths (skip_mode /
  LOCALWARP / large GLOBALMV + non-TRANSLATION gm_type), each
  asserting zero state advancement; the `SWITCHABLE` single-dir
  path (one S() ⇒ both slots, `read_from_bitstream = [true, false]`);
  the dual-dir path (two S()s ⇒ `read_from_bitstream = [true, true]`);
  three caller-bug guards (out-of-range `sub_size` /
  `interpolation_filter` / `ref_frame[0]`); and an end-to-end §5.11.23
  cascade test through the dispatcher witnessing the readout +
  `interp_filters` grid stamp on the SWITCHABLE + skip_mode (gate-
  closed needs_interp_filter) path. Test count: 721 → 731 (+10).

* **Round 176 — §5.11.28 `read_interintra_mode` reader.** Wires the
  inter-intra blending triple into the §5.11.23 inter cascade. Reads
  `interintra` against `TileInterIntraCdf[ Size_Group[ MiSize ] - 1 ]`
  when the outer gate (`skip_mode == 0 && enable_interintra_compound
  && !isCompound && BLOCK_8X8 <= MiSize <= BLOCK_32X32`) is open; when
  `interintra == 1` continues with `interintra_mode` against
  `TileInterIntraModeCdf[ ctx ]` (4-way `II_DC_PRED` / `II_V_PRED` /
  `II_H_PRED` / `II_SMOOTH_PRED`), `wedge_interintra` against
  `TileWedgeInterIntraCdf[ MiSize ]`, and on the wedge sub-branch
  `wedge_index` against `TileWedgeIndexCdf[ MiSize ]` (16-symbol shared
  with §5.11.29).

  Inner-arm side-effect: the spec sets `RefFrame[ 1 ] = INTRA_FRAME`
  on the §5.11.28 inner arm; the §5.11.23 dispatcher restamps the
  walker's slot-1 grid over the `bh4 * bw4` footprint so downstream
  neighbour walks observe the override. `AngleDeltaY`, `AngleDeltaUV`,
  and `use_filter_intra` are forced to 0 per the spec but are
  inter-block scalars not currently tracked.

  Four new §6.10.27 named constants: `II_DC_PRED = 0`,
  `II_V_PRED = 1`, `II_H_PRED = 2`, `II_SMOOTH_PRED = 3`. New
  `InterIntraReadout` aggregate carries `interintra: u8` plus
  `Option<u8>` companions for `interintra_mode` / `wedge_interintra` /
  `wedge_index`. `DecodedInterBlockModeInfo` gains an `interintra`
  field of this type.

  `decode_inter_block_mode_info` and `decode_inter_frame_mode_info`
  gain a new `enable_interintra_compound: bool` parameter (the
  caller-supplied §5.5.2 sequence-header bit).

  10 new unit tests cover: II_* ordinal alignment; the five outer-
  gate-closed paths (skip_mode / !enable_interintra_compound /
  isCompound / MiSize < BLOCK_8X8 / MiSize > BLOCK_32X32), each
  asserting zero bits consumed; the `MiSize = BLOCK_SIZES` defensive
  fallback; the gate-open + interintra=0 path (witnessed by
  inter_intra CDF adaptation and untouched inter_intra_mode row);
  the four-symbol reachability path (100-trial property test biased
  toward the inner arm, witnessing the `wedge_index` CDF adaptation);
  and two end-to-end §5.11.23 cascade tests (disabled-compound
  short-circuits, enabled-compound witnesses CDF adaptation and the
  conditional slot-1 grid stamp). Test count: 691 → 703 (+12).

* **Round 174 — §5.11.31 `assign_mv` + §5.11.32 `read_mv_component`
  syntax tree.** Wires the per-block motion-vector decode into the
  §5.11.23 inter cascade. Lifts `Error::AssignMvUnsupported` (now a
  defensive caller-bug fallback only) and surfaces
  `Error::MotionModeUnsupported` (the new §5.11.27 `read_motion_mode`
  next-arc target).

  The §5.11.31 `assign_mv( isCompound )` body iterates over
  `i = 0..1 + isCompound` and resolves each per-list `Mv[ i ]` per
  the four-arm spec dispatch (av1-spec p.78): `compMode = get_mode(i)`
  (the §5.11.30 helper, also lands this round); `PredMv[ i ] =
  GlobalMvs[ i ]` for `GLOBALMV`, else `PredMv[ i ] =
  RefStackMv[ pos ][ i ]` with `pos = (compMode == NEARESTMV) ? 0 :
  RefMvIdx` (forced to `0` when `compMode == NEWMV && NumMvFound <=
  1`); finally `Mv[ i ] = PredMv[ i ] + diffMv` via `read_mv( i )` on
  `NEWMV` or `Mv[ i ] = PredMv[ i ]` otherwise.

  The §5.11.31 `read_mv` body composes one `mv_joint` S() against
  `TileMvJointCdf[ MvCtx ]` (the 4-way `MV_JOINT_ZERO` /
  `MV_JOINT_HNZVZ` / `MV_JOINT_HZVNZ` / `MV_JOINT_HNZVNZ` code), then
  conditionally invokes `read_mv_component( 0 )` and/or
  `read_mv_component( 1 )` per the spec gating, finishing with
  per-component addition to `PredMv`.

  The §5.11.32 `read_mv_component( comp )` body composes the full
  sign-magnitude tree: `mv_sign` + `mv_class` (one of
  `MV_CLASS_0..=MV_CLASS_10`), then either the **MV_CLASS_0 ladder**
  (`mv_class0_bit` + `mv_class0_fr` / `=3` on `force_integer_mv` +
  `mv_class0_hp` / `=1` on `!allow_high_precision_mv`, with `mag =
  ((bit << 3) | (fr << 1) | hp) + 1`) or the **MV_CLASS_K ladder**
  for K >= 1 (per-bit `mv_bit` S() loop yielding `d = sum(mv_bit_i <<
  i)`, then `mv_fr` / `mv_hp` with the same gating, and `mag =
  (CLASS0_SIZE << (mv_class + 2)) + ((d << 3) | (fr << 1) | hp) + 1`).
  Returns `mv_sign ? -mag : mag`; the §6.10.25 `is_mv_valid`
  conformance bound is the caller's responsibility.

  Six new §3 / §6.10.27 / §6.10.28 named constants: `MV_JOINT_ZERO` =
  0, `MV_JOINT_HNZVZ` = 1, `MV_JOINT_HZVNZ` = 2, `MV_JOINT_HNZVNZ` =
  3, `MV_CLASS_0` = 0.

  New `get_mode(y_mode, ref_list)` public helper (§5.11.30) folds a
  YMode + reference-list index into the per-list `compMode` value
  used by §5.11.31.

  The walker's `Mvs[r][c][list][comp]` grid (introduced in r172 as a
  §7.10.2 neighbour-walk feed) is now stamped over the `bh4 * bw4`
  footprint after every `assign_mv` call. `DecodedInterBlockModeInfo`
  gains a `mv: [[i32; 2]; 2]` field carrying the §5.11.31
  `Mv[ 0..2 ]` array (observable only on the `Ok` path; the
  dispatcher still returns `Err(MotionModeUnsupported)` until
  §5.11.27 lands).

  11 new unit tests cover `get_mode` (single-pred identity + compound
  table); `assign_mv` skip_mode + seg_globalmv arms (no MV bits
  read); `read_mv` with `mv_joint = MV_JOINT_ZERO` yielding zero
  diff; direct `read_mv_component` exercises of the MV_CLASS_0
  ladder (3 sub-cases — all-sym-0 + allow_hp ⇒ mag=1; fall-through
  hp=1 ⇒ mag=2; sign=1 ⇒ mag=-2); `force_integer_mv` short-circuiting
  the `mv_class0_fr` read (mag=8); cascade structural smoke (Mv
  values within the §6.10.25 bound); and the `assign_mv` defensive
  guard rejecting `use_intrabc = true` in the inter arm. Test count:
  661 → 672 (+11).

* **Round 173 — §5.11.23 post-`find_mv_stack` reader cascade.** Wires
  `find_mv_stack` into `decode_inter_block_mode_info` and runs every
  bit-consuming leaf in §5.11.23 lines 1-32: YMode dispatch (four
  arms — `skip_mode` / SEG_LVL_SKIP|GLOBALMV / isCompound /
  single-pred), per-mode `RefMvIdx` + `drl_mode` loops, and the
  §7.10.2 stack-summary surfaced through the `DecodedInterBlockModeInfo`
  aggregate.

  Lifts `Error::FindMvStackUnsupported` (now a defensive caller-bug
  fallback only) and surfaces `Error::AssignMvUnsupported` (the new
  §5.11.31 next-arc target — `read_mv` / `read_mv_component` syntax
  tree). The §5.11.23 dispatcher therefore now consumes:

  * **§5.11.25 `read_ref_frames`** — r170 prologue (S() cascade).
  * **§7.10 `find_mv_stack`** — r172 spatial-only path, wired in r173.
  * **§5.11.23 YMode dispatch**: Arm 1 (skip_mode = 1 ⇒
    NEAREST_NEARESTMV); Arm 2 (seg_skip / seg_globalmv ⇒ GLOBALMV);
    Arm 3 (compound) reads `compound_mode` against
    `TileCompoundModeCdf[ctx]` (ctx from
    [`compound_mode_ctx`](src/cdf.rs) per §8.3.2); Arm 4 (single)
    walks `new_mv` ⇒ `zero_mv` ⇒ `ref_mv` ladder against
    `TileNewMvCdf[NewMvContext]` / `TileZeroMvCdf[ZeroMvContext]` /
    `TileRefMvCdf[RefMvContext]`.
  * **§5.11.23 RefMvIdx + drl_mode loops**: on `YMode ∈ {NEWMV,
    NEW_NEWMV}` iterate `idx = 0, 1`; on `has_nearmv(YMode)` true seed
    `RefMvIdx = 1` and iterate `idx = 1, 2`. Each iteration is gated
    on `NumMvFound > idx + 1` and reads against
    `TileDrlModeCdf[DrlCtxStack[idx]]`.

  Six new caller scalars on `decode_inter_block_mode_info` (and
  threaded through `decode_inter_frame_mode_info`): `gm_type[8]`,
  `gm_params[8][6]`, `ref_frame_sign_bias[8]`,
  `allow_high_precision_mv`, `force_integer_mv`, `use_ref_frame_mvs`.

  `DecodedInterBlockModeInfo` extended with six new fields surfaced
  on the (currently unreachable) `Ok` arm: `y_mode` (in
  `MODE_NEARESTMV..=MODE_NEW_NEWMV = 14..=25`), `ref_mv_idx` (in
  `0..=2`), and the §7.10.2 stack-summary snapshot — `num_mv_found`,
  `new_mv_context`, `ref_mv_context`, `zero_mv_context`.

  New `has_nearmv(mode)` predicate (av1-spec p.75) — returns true
  for the four NEARMV-bearing inter Y modes (NEARMV / NEAR_NEARMV /
  NEAR_NEWMV / NEW_NEARMV). Joins the existing `has_newmv`.

  Public re-exports extended with `DecodedInterBlockModeInfo`,
  `FindMvStackResult`, `GM_TYPE_IDENTITY` / `_TRANSLATION` /
  `_ROTZOOM` / `_AFFINE`, `MAX_REF_MV_STACK_SIZE`, `MV_BORDER`,
  `REF_CAT_LEVEL`, and the §6.10.22 inter Y-mode ordinals
  (`MODE_NEARESTMV` through `MODE_NEW_NEWMV`).

  10 new unit tests cover: Arm 1 zero-bit path; Arm 2 zero-bit path;
  Arm 3 compound case (rigged ref-frame cascade landing on
  `[BWDREF, ALTREF]` + `compound_mode` S() firing); Arm 4 three
  terminal modes (NEWMV / GLOBALMV / NEARESTMV); drl_mode loop
  short-circuit on fresh-walker `NumMvFound = 0` (CDF counter stays
  0 ⇒ no read fired); `has_nearmv` truth table over the full inter
  Y-mode set; caller-bug guards unchanged with the new args; and the
  `use_ref_frame_mvs = true` arm surfacing `TemporalMvScanUnsupported`
  ahead of the cascade. Test count: 651 → 661 (+10).

  The §5.11.23 post-`assign_mv` readers (`read_motion_mode` /
  `read_interintra_mode` / `read_compound_type` /
  `read_interpolation_filter`) remain pending; the natural arc
  ordering is `assign_mv` first (the §5.11.27 motion-mode walks
  consume `Mv[..]`). `decode_av1` / `encode_av1` continue to return
  `Error::NotImplemented`.

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
