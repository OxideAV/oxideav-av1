# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec.

## Status — 2026-05-28 (round 179)

**Clean-room rebuild, round 23.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core decoder modules could not be defended against the "no external
library source as reference" rule that governs every crate in this
workspace.

Bitstream parsing currently covers:

* **§5.3 / §4.10.5 — OBU bytestream walker (round 1).**
  `leb128()` (with the `(1 << 32) - 1` conformance cap and the
  8-byte length bound), `obu_header` (forbidden bit, 4-bit
  `obu_type`, extension flag, size flag), `obu_extension_header`
  (`temporal_id`, `spatial_id`, inferred to 0 when no extension —
  §6.2.3), `obu_size` payload framing, and an iterator over a
  concatenation of OBUs in the §5.2 low-overhead format.

* **§5.5 — Sequence header OBU parse (round 2).**
  `sequence_header_obu()` (§5.5.1) plus its nested syntax tables:
  `color_config()` (§5.5.2), `timing_info()` (§5.5.3),
  `decoder_model_info()` (§5.5.4), and
  `operating_parameters_info()` (§5.5.5). Returns a typed
  `SequenceHeader` (profile, still-picture / reduced-still flags,
  timing / decoder-model state, operating-point list, frame-size
  bits, frame-id presence, all `enable_*` capability bits,
  `seq_force_screen_content_tools` / `seq_force_integer_mv`,
  `order_hint_bits`, full `ColorConfig`, `film_grain_params_present`,
  and the bit-count consumed). Backed by a small internal MSB-first
  bit reader implementing §4.10.2 `f(n)` and §4.10.3 `uvlc()` per
  the §8.1 parsing process. Enforces the two §6.4.1
  bitstream-conformance gates (`seq_profile <= 2`,
  `reduced_still_picture_header == 1 ⇒ still_picture == 1`).

* **§5.9.2 — Uncompressed-header prefix parse (round 3).**
  `parse_frame_header()` consumes the leading slice of
  `uncompressed_header()` and returns a typed `FrameHeader`:
  `show_existing_frame` plus the optional `frame_to_show_map_idx`
  and `display_frame_id` for the show-existing replay path,
  `frame_type` (`KEY_FRAME` / `INTER_FRAME` / `INTRA_ONLY_FRAME` /
  `SWITCH_FRAME`), `show_frame`, `showable_frame`,
  `error_resilient_mode`, `disable_cdf_update`,
  `allow_screen_content_tools`, `force_integer_mv` (with the
  §5.9.2 `FrameIsIntra` override applied), `current_frame_id`
  (only when the sequence header opted into frame-id numbering),
  `frame_size_override_flag`, `order_hint` (width derived from
  §5.5.1's `order_hint_bits`), `primary_ref_frame` (with
  `PRIMARY_REF_NONE` for intra / error-resilient frames), and
  `refresh_frame_flags` (with the SWITCH or KEY-with-show_frame
  derivation to `allFrames = 0xff`). The reduced-still-picture
  collapse from §5.9.2 is honoured. Bit-count consumed is reported
  via `FrameHeader::bits_consumed` for the next round.
  `temporal_point_info()` (§5.9.31) call sites — gated by
  `decoder_model_info_present_flag && !equal_picture_interval` —
  are stubbed; the parser returns `Error::TemporalPointInfoUnsupported`
  if it would have to descend, but none of the 16 fixtures
  triggers it.

* **§5.9.5 / §5.9.6 / §5.9.8 / §5.9.9 — Frame-size sub-syntax
  block (round 4).** The same `parse_frame_header()` now drops
  past `refresh_frame_flags` into the four frame-size sub-syntaxes
  and returns a typed `FrameSize`: `frame_width` (post-superres),
  `frame_height`, `render_width`, `render_height`, `superres_denom`
  (in `9..=16` when `use_superres == 1`, otherwise `SUPERRES_NUM
  = 8`), `upscaled_width` (pre-superres), `mi_cols`, `mi_rows`
  (the `MI_SIZE = 4` block grid via `2 * ((dim + 7) >> 3)`), and
  the three sub-syntax-input fields (`use_superres`, `coded_denom`,
  `render_and_frame_size_different`). For super-resolved frames the
  rounded-half-up downscale `FrameWidth = (UpscaledWidth *
  SUPERRES_NUM + SuperresDenom / 2) / SuperresDenom` is applied
  literally per §5.9.8. `FrameHeader::frame_size` is
  `Some(FrameSize)` for every intra (`KEY_FRAME` /
  `INTRA_ONLY_FRAME`) frame and `None` for show-existing-frame
  replays and inter frames (the §5.9.7 `frame_size_with_refs()`
  `found_ref == 1` branch needs ref-frame state — `RefUpscaledWidth[]`
  / `RefFrameHeight[]` / `RefRenderWidth[]` / `RefRenderHeight[]` —
  not yet tracked across calls).

* **§5.9.10 / §5.9.11 / §5.9.12 / §5.9.13 — Uncompressed-header
  tail sub-syntaxes (round 5).** Three standalone parser entry
  points landed in a new `uncompressed_header_tail` module:
  * `parse_interpolation_filter` (§5.9.10) — reads
    `is_filter_switchable` + optional `f(2)` `interpolation_filter`,
    returning a typed `InterpolationFilter` enum (`Eighttap` /
    `EighttapSmooth` / `EighttapSharp` / `Bilinear` / `Switchable`)
    per §6.8.9.
  * `parse_loop_filter_params` (§5.9.11) — honours the
    `CodedLossless || allow_intrabc` short-circuit (no bits read,
    `loop_filter_ref_deltas` reset to the spec's literal defaults
    `INTRA = 1`, `LAST/LAST2/LAST3/BWDREF = 0`, `GOLDEN/ALTREF2/
    ALTREF = -1`), then for the full path reads the four
    `loop_filter_level[]` slots (with the `NumPlanes > 1 &&
    (level[0] || level[1])` gate on the chroma pair), the `f(3)`
    `loop_filter_sharpness`, and the `loop_filter_delta_enabled /
    delta_update / update_ref_delta[i] / update_mode_delta[i]`
    per-slot update walk over `TOTAL_REFS_PER_FRAME = 8`
    ref-deltas + 2 mode-deltas with `su(7)` signed offsets.
  * `parse_quantization_params` (§5.9.12 + §5.9.13) — reads
    `base_q_idx` (`f(8)`), the four `delta_q_*` per-plane offsets
    via `read_delta_q()` (each a `delta_coded` gate followed by a
    `su(7)` signed offset), the `diff_uv_delta` /
    `separate_uv_delta_q` chroma-coupling logic that mirrors V to
    U when `diff_uv_delta == 0`, and the `using_qmatrix` / `qm_y` /
    `qm_u` / `qm_v` quantizer-matrix selection.
  These three calls are **not** wired into the streaming
  `parse_frame_header` walk yet — the intervening §5.9.2 syntax
  (`allow_intrabc`, `disable_frame_end_update_cdf`, `tile_info()`,
  `segmentation_params()`, `delta_q_params()`, `delta_lf_params()`)
  sits between round 4's stop point and these calls. The next
  round can stitch them in as the intervening syntaxes land.

  New types: `InterpolationFilter`, `LoopFilterParams`,
  `QuantizationParams`. New constant: `TOTAL_REFS_PER_FRAME = 8`.
  New bitreader primitive: `BitReader::su(n)` (§4.10.6). 21 new
  unit tests cover all three sub-syntaxes (switchable +
  non-switchable interpolation, short-circuit + full-path
  loop_filter with mono/3-plane gating + delta update walk,
  mono/3-plane quantization with and without `separate_uv_delta_q`
  and with/without `using_qmatrix`) and `su(7)` boundary values.

* **§5.9.3 `allow_intrabc` + §5.9.15 `tile_info()` wired into the
  streaming parser (round 6).** For intra frames whose
  `allow_screen_content_tools && UpscaledWidth == FrameWidth`
  conjunction holds, the `parse_frame_header` walk now consumes
  the §5.9.3 `f(1)` `allow_intrabc` slot; otherwise the §5.9.2
  `allow_intrabc = 0` initialiser stands. The
  `disable_frame_end_update_cdf` bit (gated by
  `!reduced_still_picture_header && !disable_cdf_update`) is
  consumed next, then `tile_info()` per §5.9.15 walks the
  per-frame tile layout: the uniform-spacing path uses
  `increment_tile_cols_log2` / `increment_tile_rows_log2` loops
  capped at `tile_log2(1, min(sbCols, MAX_TILE_COLS))` /
  `tile_log2(1, min(sbRows, MAX_TILE_ROWS))`; the non-uniform
  path uses `ns(maxWidth)` / `ns(maxHeight)` for
  `width_in_sbs_minus_1` / `height_in_sbs_minus_1` via the new
  `BitReader::ns(n)` §4.10.7 primitive. `context_update_tile_id`
  (`f(TileColsLog2 + TileRowsLog2)`) +
  `tile_size_bytes_minus_1` (`f(2)`) are read when at least one
  of the log2 counts is non-zero. New type `TileInfo` exposes
  `uniform_tile_spacing_flag`, `tile_cols`, `tile_rows`,
  `tile_cols_log2`, `tile_rows_log2`, `context_update_tile_id`,
  `tile_size_bytes`, `mi_col_starts` (`MiColStarts[0..=TileCols]`),
  and `mi_row_starts`. New public §3 constants:
  `MAX_TILE_WIDTH = 4096`, `MAX_TILE_AREA = 4096 * 2304`,
  `MAX_TILE_ROWS = MAX_TILE_COLS = 64`. New fields on
  `FrameHeader`: `allow_intrabc`, `disable_frame_end_update_cdf`,
  `tile_info: Option<TileInfo>` (the latter `None` for inter
  frames + show-existing-frame replays). 11 new tests (7 for
  `tile_info` + 3 for `ns(n)` + 2 for the streaming-parser
  integration); the 16-fixture frame-header integration test
  gains 4 new asserted trace columns (`allow_intrabc`,
  `tile_cols`, `tile_rows`, `context_update_tile_id`) plus a
  §6.8.14 `MAX_TILE_COLS` / `MAX_TILE_ROWS` conformance guard.
  The `tile-cols-2-rows-1` fixture exercises a real 2-tile
  layout (`TileColsLog2 = 1`).

* **§5.9.12 `quantization_params()` + §5.9.14
  `segmentation_params()` wired into the streaming parser (round
  7).** After `tile_info()` the parser now consumes the
  quantization_params block (already implemented standalone in
  round 5) — `base_q_idx` (`f(8)`), the per-plane `delta_q_*`
  offsets via `read_delta_q()` (`delta_coded` `f(1)` gate + `su(7)`
  signed offset), the chroma `diff_uv_delta` / V-mirrors-U logic,
  and the `using_qmatrix` / `qm_y` / `qm_u` / `qm_v` quantizer-
  matrix block — and surfaces a typed `QuantizationParams` on
  `FrameHeader::quantization_params`. Then the new
  segmentation_params routine reads `segmentation_enabled` and,
  when enabled, either reads the three update flags or uses the
  §5.9.14 `primary_ref_frame == PRIMARY_REF_NONE` collapse
  (`update_map=1` / `temporal_update=0` / `update_data=1`, no
  bitstream reads). For `update_data=1` the inner loop walks all
  8 segments × 8 features, reading `feature_enabled` and (when
  active) `su(1+bits)` or `f(bits)` per the `Segmentation_Feature_Bits`
  / `Segmentation_Feature_Signed` / `Segmentation_Feature_Max`
  Table 5.9.14 tables. The §5.9.14 trailing `SegIdPreSkip` /
  `LastActiveSegId` derivations are surfaced. New type
  `SegmentationParams` exposing `enabled`, `update_map`,
  `temporal_update`, `update_data`,
  `segment_feature_active: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS]`,
  `segment_feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS]`,
  `seg_id_pre_skip`, `last_active_seg_id`. New public §3 constants:
  `MAX_SEGMENTS = 8`, `SEG_LVL_MAX = 8`, `SEG_LVL_ALT_Q = 0`,
  `SEG_LVL_ALT_LF_Y_V = 1`, `SEG_LVL_ALT_LF_Y_H = 2`,
  `SEG_LVL_ALT_LF_U = 3`, `SEG_LVL_ALT_LF_V = 4`,
  `SEG_LVL_REF_FRAME = 5`, `SEG_LVL_SKIP = 6`,
  `SEG_LVL_GLOBALMV = 7`, `MAX_LOOP_FILTER = 63`. New public Table
  5.9.14 tables: `SEGMENTATION_FEATURE_BITS`,
  `SEGMENTATION_FEATURE_SIGNED`, `SEGMENTATION_FEATURE_MAX`. New
  standalone parser entry point `parse_segmentation_params`. 10
  new tests (9 standalone — disabled / PRIMARY_REF_NONE collapse /
  three-bit update walk / `update_map=0` skips temporal /
  signed `SEG_LVL_ALT_Q` value `-50` / clipped at the `-255` floor
  / unsigned `SEG_LVL_REF_FRAME=6` sets `SegIdPreSkip=1` /
  zero-width `SEG_LVL_SKIP` sets `LastActiveSegId=3` /
  unexpected-end — plus 1 streaming-parser synthetic with
  `SEG_LVL_ALT_Q` active value `-123`). The 16-fixture
  frame-header integration test gains two new asserted trace
  columns (`base_q_idx`, `seg_enabled`) plus a `SegIdPreSkip = 0`
  / `LastActiveSegId = 0` invariant guard.

* **§5.9.17 `delta_q_params()` + §5.9.18 `delta_lf_params()` wired
  into the streaming parser (round 8).** After `segmentation_params()`
  the parser now consumes `delta_q_params()` per §5.9.17 — the
  `delta_q_present` `f(1)` slot is read only when `base_q_idx > 0`
  (otherwise the §5.9.17 `delta_q_present = 0` initialiser stands and
  no bit is consumed), and `delta_q_res` (`f(2)`) follows only when
  `delta_q_present == 1` — then `delta_lf_params()` per §5.9.18: the
  whole block is gated on `delta_q_present`, the `delta_lf_present`
  `f(1)` slot is suppressed when `allow_intrabc == 1`, and
  `delta_lf_res` (`f(2)`) + `delta_lf_multi` (`f(1)`) follow only when
  `delta_lf_present == 1`. New types `DeltaQParams`
  (`delta_q_present`, `delta_q_res`) and `DeltaLfParams`
  (`delta_lf_present`, `delta_lf_res`, `delta_lf_multi`) surface on
  `FrameHeader::delta_q_params` / `FrameHeader::delta_lf_params`
  (`Some` for intra frames, `None` for inter / show-existing-frame
  paths). New standalone parser entry points `parse_delta_q_params` /
  `parse_delta_lf_params`. 9 new unit tests (3 for `delta_q_params` —
  `base_q_idx == 0` no-read / `delta_q_present == 0` 1-bit /
  `delta_q_present == 1` reads `delta_q_res` — plus an
  unexpected-end; 5 for `delta_lf_params` — gated-off when
  `delta_q_present == 0` / `delta_lf_present == 0` 1-bit / full path
  reading `delta_lf_res` + `delta_lf_multi` / suppressed by
  `allow_intrabc` / unexpected-end). The 16-fixture frame-header
  integration test gains two new asserted trace columns
  (`delta_q_present`, `delta_lf_present`) plus `delta_q_res = 0` /
  `delta_lf_res = 0` / `delta_lf_multi = false` invariant guards. The
  `lossless-i-only` fixture (`base_q_idx = 0`) exercises the §5.9.17
  no-read branch; every other fixture reads exactly one
  `delta_q_present` bit (all 0).

* **§5.9.11 `loop_filter_params()` wired into the streaming parser
  (round 9).** After `delta_lf_params()` the parser now derives
  `CodedLossless` per the §5.9.2 lines that scan `LosslessArray[]` over
  the eight per-segment qindexes — `get_qindex(1, segmentId)` (the §8.7
  ignore-delta branch) returns `base_q_idx`, or
  `Clip3(0, 255, base_q_idx + FeatureData[segmentId][SEG_LVL_ALT_Q])`
  when `seg_feature_active_idx(segmentId, SEG_LVL_ALT_Q)` is set; a
  segment is lossless when its qindex is 0 and all five §5.9.12
  `DeltaQ?*` offsets are 0 — and then consumes `loop_filter_params()`
  per §5.9.11. The `CodedLossless || allow_intrabc` short-circuit
  consumes no bits and resets `loop_filter_ref_deltas` to the spec
  defaults; the full path reads the four `loop_filter_level[]` slots
  (chroma `[2]`/`[3]` gated on `NumPlanes > 1 && (level[0] ||
  level[1])`), the `f(3)` `loop_filter_sharpness`, and the
  `loop_filter_delta_enabled` / `delta_update` per-slot update walk. The
  routine itself landed standalone in round 5
  (`parse_loop_filter_params`); this round adds the streaming wire-in
  plus the new `compute_coded_lossless` derivation. New field
  `FrameHeader::loop_filter_params: Option<LoopFilterParams>` (`Some`
  for intra frames, `None` for inter / show-existing replays). 6 new
  unit tests (5 for `compute_coded_lossless` covering the lossless /
  non-lossless / non-zero-delta / `SEG_LVL_ALT_Q`-clamp / seg-disabled
  branches, 1 streaming full-path test with non-zero loop-filter
  levels); the `parses_tiny_key_frame_prefix` bit-count rises 31 → 48.
  The 16-fixture frame-header integration test gains five new asserted
  trace columns (`lf_y`, `lf_uv0`, `lf_uv1`, `lf_sharp`,
  `lf_delta_enabled`) mapped to `loop_filter_level[0, 2, 3]` /
  `loop_filter_sharpness` / `loop_filter_delta_enabled` per §6.8.10. The
  `lossless-i-only` fixture (`base_q_idx = 0`, `lf_delta_enabled = 0`)
  exercises the §5.9.11 short-circuit and validates the `CodedLossless`
  derivation; the other 15 take the full path (several with non-zero
  chroma levels, e.g. `film-grain-on` `lf_y=4 / lf_uv0=14 / lf_uv1=11`).

* **§5.9.19 `cdef_params()` wired into the streaming parser (round
  10).** After `loop_filter_params()` the parser now consumes
  `cdef_params()` per §5.9.19. The `CodedLossless || allow_intrabc ||
  !enable_cdef` short-circuit consumes no bits and leaves `cdef_bits =
  0`, `CdefDamping = 3`, and all four strength arrays zeroed; the full
  path reads `cdef_damping_minus_3` (`f(2)`, `CdefDamping =
  cdef_damping_minus_3 + 3`), `cdef_bits` (`f(2)`), then for each of the
  `1 << cdef_bits` entries the `cdef_y_pri_strength[i]` (`f(4)`) /
  `cdef_y_sec_strength[i]` (`f(2)`) pair plus, when `NumPlanes > 1`, the
  `cdef_uv_pri_strength[i]` (`f(4)`) / `cdef_uv_sec_strength[i]` (`f(2)`)
  pair. The §5.9.19 secondary `== 3 ⇒ += 1` adjustment is applied
  literally (a raw `3` is stored as `4`). New type `CdefParams`
  (`cdef_damping`, `cdef_bits`, the four `cdef_*_strength: [u8; 8]`
  arrays, `short_circuited`). New public §3-derived constant
  `CDEF_MAX_STRENGTHS = 8` (the loop bound `1 << cdef_bits` with
  `cdef_bits` an `f(2)` value is at most 8). New standalone parser entry
  point `parse_cdef_params`. New field `FrameHeader::cdef_params:
  Option<CdefParams>` (`Some` for intra frames, `None` for inter /
  show-existing replays). 8 new unit tests (short-circuit on each of the
  three gate conditions, full-path single-entry 3-plane,
  secondary-`3⇒4` for both Y/UV, monochrome chroma-skip, 8-entry
  `cdef_bits=3` loop bound, unexpected-end); the
  `parses_tiny_key_frame_prefix` bit-count rises 48 → 64. The 16-fixture
  frame-header integration test gains six new asserted trace columns
  (`cdef_damping`, `cdef_bits`, `cdef_y_pri_strength[0]`,
  `cdef_uv_pri_strength[0]`, `cdef_y_sec_strength[0]`,
  `cdef_uv_sec_strength[0]`) sourced from each fixture's `CDEF idx=0`
  trace line, plus a short-circuit invariant. The `CDEF` trace lines were
  empirically confirmed to log the raw pre-adjustment secondary strength
  (a value of `3` appears, which the parser stores as `4`); the test maps
  the raw expectation through the adjustment. `lossless-i-only`
  (CodedLossless) and `screen-content-tools` (`enable_cdef=0`) exercise
  the short-circuit; the other 14 take the full path (e.g.
  `super-resolution` `damping=5 / cdef_y_pri=2 / cdef_uv_pri=14`).

* **§5.9.20 `lr_params()` wired into the streaming parser (round
  11).** After `cdef_params()` the parser now consumes `lr_params()`
  per §5.9.20. `AllLossless = CodedLossless && (FrameWidth ==
  UpscaledWidth)` is derived inline (super-resolution downscaling
  keeps AllLossless 0 even when CodedLossless is 1). The `AllLossless
  || allow_intrabc || !enable_restoration` short-circuit consumes no
  bits and leaves every plane `RESTORE_NONE` with `UsesLr = 0` and
  zero `LoopRestorationSize[]`. The full path reads one `lr_type`
  (`f(2)`) per plane, mapping each through `Remap_Lr_Type[4] = {
  RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ }`;
  when any plane uses LR, the parser reads `lr_unit_shift` (`f(1)`,
  post-incremented for 128×128 superblocks, otherwise extended by
  `lr_unit_extra_shift` `f(1)` when the first bit is set) and — when
  `subsampling_x && subsampling_y && usesChromaLr` — `lr_uv_shift`
  (`f(1)`). The three `LoopRestorationSize[]` entries derive from
  `RESTORATION_TILESIZE_MAX = 256` via `>> (2 - lr_unit_shift)` for
  luma and `>> lr_uv_shift` for chroma. New types `LrParams`
  (`frame_restoration_type[3]`, `uses_lr`, `uses_chroma_lr`,
  `lr_unit_shift`, `lr_uv_shift`, `loop_restoration_size[3]`,
  `short_circuited`) and `FrameRestorationType` (4-variant enum with
  §6.10.15 symbol discriminants 0..=3, plus `remap(lr_type)`). New
  constant `RESTORATION_TILESIZE_MAX = 256`. New standalone parser
  entry point `parse_lr_params`. New field
  `FrameHeader::lr_params: Option<LrParams>`. 19 new unit tests
  (short-circuit on all three gates, the `Remap_Lr_Type` table, UsesLr=0
  no-shift path, non-128 SB with lr_unit_shift each of {0, 1, 2}, 128 SB
  post-increment with each of {1, 2}, 4:2:0 chroma uv-shift {0, 1},
  subsampling-gated uv-shift skip on 4:4:4 and 4:2:2, monochrome
  one-type-only, distinct types per plane, and two unexpected-end
  cases). The `parses_tiny_key_frame_prefix` bit-count rises 64 → 70.
  The 16-fixture frame-header integration test gains five new asserted
  trace columns (`y_type`, `u_type`, `v_type`, `unit_shift`,
  `uv_shift` from each fixture's `LOOP_RESTORATION idx=0` trace line)
  plus a `UsesLr` cross-check, short-circuit invariant (only
  `lossless-i-only` AllLossless short-circuits), and a
  `LoopRestorationSize[0]` derivation cross-check. Empirically
  confirmed: the trace logger writes the **raw bitstream `lr_type`**
  (not the post-`Remap_Lr_Type` symbol) — the four
  `bits_consumed`-traceable fixtures (`i-only-64x64-prof0`,
  `i-frame-then-p-64x64`, `super-resolution`, `superblocks-128`) all
  decode bit-exactly only when the test routes the trace value through
  `Remap_Lr_Type` before comparing. `super-resolution` exercises
  three-plane Wiener LR with unit_shift=2; `superblocks-128`
  exercises two-plane SgrProj LR with unit_shift=2 from a 128×128
  superblock; `i-only-64x64-prof0` and `i-frame-then-p-64x64` each
  exercise V-plane-only Wiener LR (`usesChromaLr=1` and the 4:2:0
  uv-shift read=0); `lossless-i-only` exercises the AllLossless
  short-circuit (the only fixture that does); the other 10 walk the
  full LR path with all three planes RESTORE_NONE / `UsesLr = 0`.

* **§5.9.21 `read_tx_mode()` wired into the streaming parser (round
  12).** After `lr_params()` the parser now consumes `read_tx_mode()`
  per §5.9.21. When `CodedLossless == 1` the first branch fires:
  no bits are read and `TxMode = ONLY_4X4`. Otherwise a single
  `tx_mode_select` (`f(1)`) bit selects `TX_MODE_SELECT` (`1`) or
  `TX_MODE_LARGEST` (`0`). New type `TxMode` (3-variant enum with
  §6.8.21 symbol discriminants `Only4x4 = 0`, `TxModeLargest = 1`,
  `TxModeSelect = 2`). New constant `TX_MODES = 3`. New standalone
  parser entry point `parse_tx_mode`. New field
  `FrameHeader::tx_mode: Option<TxMode>` (`Some` for intra frames,
  `None` for inter / show-existing replays). 6 new unit tests (the
  §6.8.21 symbol values + count, the CodedLossless no-bits path twice,
  `tx_mode_select` set/clear, unexpected-end); the
  `parses_tiny_key_frame_prefix` bit-count rises 70 → 71. The 16-fixture
  frame-header integration test gains one new asserted trace column
  (`tx_mode` from each fixture's `FRAME_HEADER` trace line) plus a
  `ONLY_4X4 ⇒ CodedLossless` invariant. The corpus exercises all three
  values: `lossless-i-only` is the only `tx_mode = 0` (ONLY_4X4,
  CodedLossless, no-bits path); `tiny-i-only-16x16-prof0`,
  `monochrome-grey-only`, `profile-1-yuv444-8bit`, and
  `profile-2-yuv422-12bit` are `tx_mode = 1` (TX_MODE_LARGEST,
  `tx_mode_select = 0`); the other 11 are `tx_mode = 2`
  (TX_MODE_SELECT, `tx_mode_select = 1`).

* **The §5.9.2 uncompressed-header tail wired into the streaming parser
  (round 13).** After `read_tx_mode()` the intra path now walks to the
  end of `uncompressed_header()`: `frame_reference_mode()` (§5.9.23),
  `skip_mode_params()` (§5.9.22), the `allow_warped_motion` slot,
  `reduced_tx_set` (`f(1)`), `global_motion_params()` (§5.9.24), and
  `film_grain_params()` (§5.9.30). For an intra frame all but
  `reduced_tx_set` (one bit) and `film_grain_params()` collapse without
  reading bits. New types `WarpModelType`, `GlobalMotionParams`
  (`gm_type[8]` / `gm_params[8][6]`, identity short-circuit), and
  `FilmGrainParams` (full §5.9.30 field set); new `FrameHeader` fields
  `reference_select` / `skip_mode_present` / `allow_warped_motion` /
  `reduced_tx_set` / `global_motion_params` / `film_grain_params`. The
  full §5.9.24/§5.9.25 inter global-motion syntax — `read_global_param`
  plus the §5.9.26–§5.9.29 `decode_signed_subexp_with_ref` /
  `decode_subexp` / `inverse_recenter` sub-exponential decoders — is
  implemented and exposed as standalone `parse_global_motion_params`;
  `film_grain_params()` is exposed as `parse_film_grain_params`. 14 new
  unit tests; the integration test embeds the `film-grain-on` fixture's
  full 718-byte FRAME OBU payload to validate the `apply_grain = 1` FGS
  block (14 Y points, 8 Cb + 9 Cr points, `ar_coeff_lag = 2`,
  `seed = 45231`) byte-exact against the fixture trace.

* **The inter-frame `uncompressed_header()` path (round 14).** An
  `INTER_FRAME` / `SWITCH_FRAME` header now parses end-to-end. The §5.9.2
  `else` branch reads `frame_refs_short_signaling`, the explicit
  `ref_frame_idx[]` (or computes them via §7.8 `set_frame_refs()`), the
  §5.9.7 `frame_size_with_refs()` / `frame_size()` + `render_size()` size
  selection, `allow_high_precision_mv`, §5.9.10
  `read_interpolation_filter()`, `is_motion_mode_switchable`,
  `use_ref_frame_mvs`, then the shared tile / quant / segment / delta /
  loop-filter / CDEF / LR / tx-mode tail, the inter
  `frame_reference_mode()` (`reference_select`), §5.9.22
  `skip_mode_params()`, `allow_warped_motion`, `reduced_tx_set`, inter
  `global_motion_params()`, and `film_grain_params()`. New §7.8
  `set_frame_refs()`, §5.9.3 `get_relative_dist()`, §5.9.7
  `frame_size_with_refs()`, and §5.9.22 `skip_mode_params()` are backed by
  a new public `RefInfo` cross-frame reference state (`RefValid[]` /
  `RefOrderHint[]` / `RefFrameId[]` + per-slot `RefUpscaledWidth[]` /
  `RefFrameHeight[]` / `RefRenderWidth[]` / `RefRenderHeight[]`). New
  public API: `parse_frame_header_with_refs()`, `RefInfo`,
  `InterFrameRefs` (surfaced on `FrameHeader::inter_refs`). Verified
  byte-exact against the `i-frame-then-p-64x64` `idx=1` `FRAME_HEADER` +
  `REF_MAP` trace lines (134 uncompressed-header bits; `ref_frame_idx =
  [0; 7]`, `base_q_idx = 120`, `tx_mode = 1`, `reference_select = 0`,
  `allow_warped_motion = 1`). 5 new tests (a byte-exact inter-header
  fixture test, a `RefInfo` default contract test, and three
  `set_frame_refs()` / `get_relative_dist()` unit tests).

* **§8.2 — the symbol (arithmetic / msac) decoder (round 15).** A new
  standalone `SymbolDecoder` implements the AV1 entropy engine that
  every tile-content read will sit on: §8.2.2 `init_symbol(sz)`, §8.2.6
  `read_symbol(cdf)` (the CDF-adaptive multisymbol search with
  `EC_PROB_SHIFT`/`EC_MIN_PROB`, the `prev - cur` range update, and the
  seven-step renormalisation drawing new bits — or §8.2.2 padding zeros
  once `SymbolMaxBits` is exhausted), the §8.3 adaptive-rate CDF update
  (`rate = 3 + (cdf[N]>15) + (cdf[N]>31) + Min(FloorLog2(N), 2)` plus the
  count-to-32 counter), §8.2.3 `read_bool()`, §8.2.5 `read_literal(n)`
  (`L(n)`), `NS(n)` (§4.10.10), the arithmetic-coded
  `decode_subexp_bool(numSyms, k)` (§5.9.28 bool variant), and §8.2.4
  `exit_symbol()` (trailing-bit accounting + byte-alignment advance,
  rejecting the `SymbolMaxBits < -14` conformance violation). The
  decoder shares the existing MSB-first `BitReader` (§8.1 `f(n)`) so its
  position indicator advances the same `get_position()` the OBU walk
  uses. Default CDF tables and the §8.3.2 CDF-selection process are out
  of scope — they land with the tile-content decode that consumes them.
  13 byte-exact unit tests (hand-traced single decodes + term-by-term
  §8.3 update checks + padding-zero / underflow edge cases).

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — intra-frame
  mode / partition subset (round 16).** A new `cdf` module transcribes
  the §9.4 `Default_Intra_Frame_Y_Mode_Cdf` (5×5×14), the five
  `Default_Partition_W{8,16,32,64,128}_Cdf` tables, `Default_Skip_Cdf`,
  and `Default_Segment_Id_Cdf` verbatim from the spec — every row is
  length `N + 1` with `row[N-1] == 1 << 15` and `row[N]` the §8.3
  adaptation counter, exactly as `SymbolDecoder::read_symbol` expects.
  `TileCdfContext::new_from_defaults` performs the §8.3.1 init step
  ("each `Tile*Cdf` array is set equal to a copy of `Default_*_Cdf`"),
  producing a mutable per-tile working set. The §8.3.2 selection
  surfaces a `&mut [u16]` row for each element: `intra_frame_y_mode`
  (`[abovemode][leftmode]`), `partition` (array-by-`bsl`, row-by-`ctx`),
  `skip` (`[ctx]`), `segment_id` (`[ctx]`) — passed straight to
  `read_symbol`. Scalar context helpers `intra_mode_ctx` /
  `partition_ctx` / `skip_ctx` / `segment_id_ctx` compute the index from
  the neighbour inputs the (future) tile walk supplies. The remaining
  ~100 §9.4 tables, the `init_coeff_cdfs` coefficient set, and the
  other §8.3.2 selections (`split_or_horz` / `split_or_vert` /
  `tx_depth` / `txfm_split` / the motion-vector + uv-mode groups) are a
  clear followup. 9 new unit tests: §8.3.1 byte-exact copy + the
  `cdf[N-1] == 32768` / `cdf[N] == 0` invariant on every transcribed
  row; working-copy independence from the immutable defaults;
  `Intra_Mode_Context[]` term-by-term; `partition_ctx` / `skip_ctx` /
  `segment_id_ctx` formulae across all branches; `partition_cdf`
  selected by `bsl` returning the right row lengths and the
  default-row contents; and two end-to-end decodes driving the real
  `SymbolDecoder` through a default-CDF row (a `skip` decode that
  exercises the §8.3 update path, and a `partition` multisymbol decode
  with the update disabled).

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — motion-vector
  component subset (round 17).** Extends `cdf` with the nine
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
  `mv_fr_cdf(MvCtx, comp)`, `mv_hp_cdf(MvCtx, comp)` — yielding the
  row `SymbolDecoder::read_symbol` consumes. The §5.11.31
  `MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0` derivation is exposed
  as the `mv_ctx` helper. 7 new unit tests: every §9.4 transcribed
  value asserted byte-exact (including the expanded `*128` fixed-point);
  §8.3.1 init copies the default into every `MV_CONTEXTS × MV_COMPS`
  slot with the `cdf[N - 1] == 32768` / `cdf[N] == 0` invariant on
  every row; the §5.11.31 `mv_ctx` derivation matches the spec; §8.3.2
  selectors return the right default row for every
  `(MvCtx, comp, *)` indexing variant; working-copy independence —
  adapting `mv_joint` / `mv_sign` / `mv_class0_fr` / `mv_bit` does not
  mutate the §9.4 source; and two end-to-end decodes driving the real
  `SymbolDecoder` through a default CDF (a 4-value `mv_joint` decode
  that exercises the §8.3 update path and a binary `mv_bit` decode
  with `disable_cdf_update == true`).

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — inter-mode /
  reference-frame subset (round 18).** Extends `cdf` with the 13
  `Default_*_Cdf` tables that drive every inter-block mode and reference
  syntax: `Default_New_Mv_Cdf`, `Default_Zero_Mv_Cdf`,
  `Default_Ref_Mv_Cdf`, `Default_Drl_Mode_Cdf`, `Default_Is_Inter_Cdf`,
  `Default_Comp_Mode_Cdf`, `Default_Skip_Mode_Cdf`,
  `Default_Comp_Ref_Cdf`, `Default_Comp_Bwd_Ref_Cdf`,
  `Default_Single_Ref_Cdf`, `Default_Compound_Mode_Cdf`,
  `Default_Comp_Ref_Type_Cdf`, `Default_Uni_Comp_Ref_Cdf` —
  all transcribed verbatim from §9.4 with the §3 constants
  `NEW_MV_CONTEXTS = 6`, `ZERO_MV_CONTEXTS = 2`, `REF_MV_CONTEXTS = 6`,
  `DRL_MODE_CONTEXTS = 3`, `IS_INTER_CONTEXTS = 4`,
  `COMP_INTER_CONTEXTS = 5`, `SKIP_MODE_CONTEXTS = 3`, `REF_CONTEXTS = 3`,
  `FWD_REFS = 4`, `BWD_REFS = 3`, `SINGLE_REFS = 7`,
  `UNIDIR_COMP_REFS = 4`, `COMP_REF_TYPE_CONTEXTS = 5`,
  `COMPOUND_MODES = 8`, `COMPOUND_MODE_CONTEXTS = 8`, `COMP_NEWMV_CTXS = 5`
  re-exposed at the crate root. `TileCdfContext::new_from_defaults`
  performs the §8.3.1 init step for every new array. The §8.3.2
  selection surfaces 13 `&mut [u16]` accessors —
  `new_mv_cdf` / `zero_mv_cdf` / `ref_mv_cdf` / `drl_mode_cdf` /
  `is_inter_cdf` / `comp_mode_cdf` / `skip_mode_cdf` / `comp_ref_cdf` /
  `comp_bwd_ref_cdf` / `single_ref_cdf` / `compound_mode_cdf` /
  `comp_ref_type_cdf` / `uni_comp_ref_cdf` — feeding straight into
  `SymbolDecoder::read_symbol`. Scalar §8.3.2 context helpers
  `is_inter_ctx(above_intra, left_intra)` (`AvailU && AvailL` /
  `AvailU XOR AvailL` / neither-avail branches per the spec ladder),
  `skip_mode_ctx(above, left)` (sum of neighbour `SkipModes[]`),
  `ref_count_ctx(c0, c1)` (the `<` / `==` / `>` three-branch ladder
  shared by every `single_ref_p*` / `comp_ref` / `comp_bwdref` /
  `uni_comp_ref_p*` paragraph), and `compound_mode_ctx(ref_mv_ctx,
  new_mv_ctx)` (the `Compound_Mode_Ctx_Map[ RefMvContext >> 1 ][
  Min(NewMvContext, COMP_NEWMV_CTXS - 1) ]` lookup, with the
  `COMPOUND_MODE_CTX_MAP` table itself surfaced as a public constant)
  compute each `ctx` from the neighbour-summary inputs the (future)
  tile walk supplies. 10 new unit tests (182 in src/, up from 172):
  table-dimension audit verifying every new `Default_*_Cdf` shape
  matches the spec literal (with the §8.2.6 `cdf[N - 1] == 32768` /
  `cdf[N] == 0` invariant enforced on every row); hand-picked
  byte-exact spot-checks across all 13 tables (every literal that
  appears at a row boundary read back unchanged); §8.3.1 init copies
  every default into the corresponding `Tile*Cdf` slot; §8.3.2
  selectors return the right default row at every hand-picked
  `(frame_type, ctx)` tuple — both extremes of every `ctx` index for
  all 13 syntax elements; working-copy independence — adapting
  `new_mv` / `comp_ref` / `compound_mode` does not mutate the §9.4
  source; §8.3.2 `is_inter_ctx` branch coverage (all 9 above/left
  combinations); `skip_mode_ctx` (the 4 neighbour-flag pairs);
  `ref_count_ctx` (the 3 ordering branches); `compound_mode_ctx` (one
  spot-check from each of the 3 `COMPOUND_MODE_CTX_MAP` rows plus the
  `Min(.., COMP_NEWMV_CTXS - 1)` clamp + the `RefMvContext >> 1`
  saturation); and an end-to-end §8.2 `SymbolDecoder` decode driving
  the `compound_mode` (8-value) default CDF row selected by
  `compound_mode_ctx(4, 4) = 7`.

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — palette /
  filter-intra / CFL subset (round 19).** Extends `cdf` with the
  filter-intra group (`Default_Filter_Intra_Mode_Cdf` 5-value,
  `Default_Filter_Intra_Cdf[ BLOCK_SIZES ]` binary with the §9.4
  "indices 10–15 / 20–21 never used" filler preserved), the palette
  group (`Default_Palette_Y_Mode_Cdf[ 7 ][ 3 ]`,
  `Default_Palette_Uv_Mode_Cdf[ 2 ]`,
  `Default_Palette_{Y,Uv}_Size_Cdf[ 7 ]` 7-value, and the fourteen
  `Default_Palette_Size_{2..8}_{Y,Uv}_Color_Cdf[ 5 ]` colour-index
  tables whose symbol count grows with `PaletteSize`), and the CFL
  group (`Default_Cfl_Sign_Cdf` 8-value,
  `Default_Cfl_Alpha_Cdf[ 6 ][ 17 ]` 16-value) — all transcribed
  verbatim from §9.4 with the §3 constants `BLOCK_SIZES = 22`,
  `INTRA_FILTER_MODES = 5`, `PALETTE_BLOCK_SIZE_CONTEXTS = 7`,
  `PALETTE_{Y,UV}_MODE_CONTEXTS = 3/2`, `PALETTE_SIZES = 7`,
  `PALETTE_COLORS = 8`, `PALETTE_COLOR_CONTEXTS = 5`,
  `CFL_JOINT_SIGNS = 8`, `CFL_ALPHABET_SIZE = 16`,
  `CFL_ALPHA_CONTEXTS = 6` and the `Palette_Color_Context` /
  `Palette_Color_Hash_Multipliers` additional-tables arrays re-exposed
  at the crate root. `new_from_defaults` performs the §8.3.1 init step
  for every array. The §8.3.2 selection surfaces ten `&mut [u16]`
  accessors — `filter_intra_cdf(MiSize)`, `filter_intra_mode_cdf()`,
  `palette_y_mode_cdf(bsizeCtx, ctx)`, `palette_uv_mode_cdf(ctx)`,
  `palette_{y,uv}_size_cdf(bsizeCtx)`,
  `palette_{y,uv}_color_cdf(PaletteSize, ctx)` (size-keyed, `Option`
  for out-of-range), `cfl_sign_cdf()`, `cfl_alpha_cdf(ctx)`. Scalar
  §8.3.2 helpers `palette_y_mode_ctx(above, left)`,
  `palette_uv_mode_ctx(PaletteSizeY)`,
  `palette_color_ctx(ColorContextHash)` (the `Palette_Color_Context`
  lookup returning `None` for the spec's `-1` sentinels), and
  `cfl_alpha_{u,v}_ctx(signU, signV)` (`(signU-1)*3+signV` /
  `(signV-1)*3+signU`, with the §8.3.2 `ctx == cfl_alpha_signs - 2`
  identity checked) compute each `ctx` from scalar neighbour inputs.
  8 new unit tests (190 in src/, up from 182): full dimension audit of
  every palette / filter-intra / CFL table with the §8.2.6
  `cdf[N-1] == 32768` / `cdf[N] == 0` invariant on every row;
  hand-picked byte-exact spot-checks; §8.3.1 init copy; size-keyed
  colour-CDF selection (row length `K+1` for size `K`, `None` outside
  `2..=8`); the palette/CFL context formulas; the
  `Palette_Color_Context` sentinel map; and an end-to-end §8.2
  `SymbolDecoder` decode driving the 16-value `cfl_alpha_u` default
  CDF row selected by `cfl_alpha_u_ctx(1, 0) = 0`.

* **§9.4 default CDF tables + §8.3.2 selection — transform-size
  subset (round 20).** Extends `cdf` with the five transform-size
  default tables (`Default_Tx_8x8_Cdf[ 3 ][ 3 ]`,
  `Default_Tx_16x16_Cdf[ 3 ][ 4 ]`, `Default_Tx_32x32_Cdf[ 3 ][ 4 ]`,
  `Default_Tx_64x64_Cdf[ 3 ][ 4 ]`, `Default_Txfm_Split_Cdf[ 21 ][ 3 ]`)
  — all transcribed verbatim from §9.4 with the §3 constants
  `TX_SIZE_CONTEXTS = 3`, `TX_SIZES = 5`, `MAX_TX_DEPTH = 2`,
  `TXFM_PARTITION_CONTEXTS = 21`. `new_from_defaults` performs the
  §8.3.1 init step for every array. Two `&mut [u16]` accessors
  surface the §8.3.2 selection: `tx_depth_cdf(maxTxDepth, ctx)` picks
  `TileTx{8x8,16x16,32x32,64x64}Cdf[ ctx ]` per the §8.3.2 paragraph's
  four-way `maxTxDepth ∈ {1, 2, 3, 4}` switch (returning `None` when
  `maxTxDepth == 0`, the syntax-not-read case), and
  `txfm_split_cdf(ctx)` picks `TileTxfmSplitCdf[ ctx ]`. Scalar
  context helpers `tx_depth_ctx(aboveW, leftH, maxTxWidth, maxTxHeight)`
  (the `(aboveW >= maxTxWidth) + (leftH >= maxTxHeight)` formula) and
  `txfm_split_ctx(above, left, txSzSqrUp, maxTxSz)` (the
  `(txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 + above + left`
  formula) compute the `ctx` from scalar inputs the §5.11.15 /
  §5.11.16 syntax supplies. 8 new unit tests (198 in src/, up from
  190): every transform-size table's `cdf[N-1] == 32768` /
  `cdf[N] == 0` invariant, dimension audit against the §3 constants;
  byte-anchor spot-checks on every table's first/last entries;
  §8.3.1 init-copy independence; `tx_depth_cdf` four-way selection
  with row-length assertions; the `tx_depth_ctx` formula across all
  four neighbour combinations; the `txfm_split_ctx` formula
  walked term-by-term plus an exhaustive bounds sweep over the
  reachable `(above, left, maxTxSz, txSzSqrUp)` tuples; and two
  end-to-end §8.2 `SymbolDecoder` decodes — one driving the
  3-symbol `TileTx16x16Cdf[ 2 ]` row selected by
  `tx_depth_ctx(16, 16, 16, 16) = 2`, the other driving the
  binary `TileTxfmSplitCdf[ 2 ]` row selected by
  `txfm_split_ctx(true, true, 4, 4) = 2`.

* **§9.4 default CDF tables + §8.3.2 selection — inter-frame
  transform-type subset (round 21).** Extends `cdf` with the three
  inter-frame transform-type default tables
  (`Default_Inter_Tx_Type_Set1_Cdf[ 2 ][ 17 ]` — the 16-symbol full
  set for 4x4 / 8x8 inter blocks reaching `TX_SET_INTER_1`;
  `Default_Inter_Tx_Type_Set2_Cdf[ 13 ]` — the 12-symbol 16x16-only
  set for `TX_SET_INTER_2`; `Default_Inter_Tx_Type_Set3_Cdf[ 4 ][ 3 ]`
  — the 2-symbol `{ IDTX, DCT_DCT }` reduced set for
  `TX_SET_INTER_3`) — all transcribed verbatim from §9.4 with the §3
  constants `TX_TYPES = 16`, `TX_TYPES_SET2 = 12`, `TX_TYPES_SET3 = 2`,
  `INTER_TX_TYPE_SET1_SIZES = 2`, `INTER_TX_TYPE_SET3_SIZES = 4` and
  the §6.10.19 transform-set tag constants `TX_SET_DCTONLY = 0` /
  `TX_SET_INTER_1 = 1` / `TX_SET_INTER_2 = 2` / `TX_SET_INTER_3 = 3`.
  `new_from_defaults` performs the §8.3.1 init step for every array.
  An `&mut [u16]` accessor `inter_tx_type_cdf(set, tx_size_sqr)`
  surfaces the §8.3.2 selection — the three-way
  `TileInterTxTypeSet{1,2,3}Cdf` switch keyed by the §5.11.48 `set`
  return — yielding `None` for `TX_SET_DCTONLY` (where §5.11.47
  forces `TxType = DCT_DCT` and `inter_tx_type` is not read) and for
  any unreachable `(set, tx_size_sqr)` combination. The scalar
  §5.11.48 helper `inter_tx_type_set(tx_sz_sqr, tx_sz_sqr_up,
  reduced_tx_set)` computes the `set ∈ { TX_SET_DCTONLY,
  TX_SET_INTER_1, TX_SET_INTER_2, TX_SET_INTER_3 }` from the
  `Tx_Size_Sqr[ txSz ]` / `Tx_Size_Sqr_Up[ txSz ]` / `reduced_tx_set`
  inputs the surrounding §5.11.47 syntax supplies. 6 new unit tests
  (204 in src/, up from 198): every inter-tx-type table's
  `cdf[N-1] == 32768` / `cdf[N] == 0` invariant, dimension audit
  against the §3 constants; byte-anchor spot-checks on every table's
  first / last entries; §8.3.1 init-copy independence with a
  mutate-doesn't-touch-source assertion; `inter_tx_type_cdf`
  three-way selection with row-length assertions and unreachable-set
  / out-of-range coverage; the `inter_tx_type_set` formula walked
  across every reachable `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)`
  triple (including the rectangular `TX_4X8` / `TX_16X32` cases
  where `tx_sz_sqr != tx_sz_sqr_up`); and one end-to-end §8.2
  `SymbolDecoder` decode driving the 2-symbol
  `TileInterTxTypeSet3Cdf[ 1 ]` row selected by
  `inter_tx_type_set(1, 1, true) = TX_SET_INTER_3`.

* **§9.4 default CDF tables + §8.3.2 selection — intra-frame
  transform-type subset (round 137).** Completes the §6.10.19
  transform-set coverage started in round 21 with the two intra-frame
  default tables (`Default_Intra_Tx_Type_Set1_Cdf[ 2 ][ INTRA_MODES ][ 8 ]`
  — the 7-symbol full intra set for 4x4 / 8x8 intra blocks reaching
  `TX_SET_INTRA_1`; `Default_Intra_Tx_Type_Set2_Cdf[ 3 ][ INTRA_MODES ][ 6 ]`
  — the 5-symbol reduced intra set for 4x4 / 8x8 / 16x16 intra blocks
  reaching `TX_SET_INTRA_2`) — both transcribed verbatim from §9.4 with
  the §3 constants `TX_SET_INTRA_1 = 1`, `TX_SET_INTRA_2 = 2`,
  `TX_TYPES_INTRA_SET1 = 7`, `TX_TYPES_INTRA_SET2 = 5`,
  `INTRA_TX_TYPE_SET1_SIZES = 2`, `INTRA_TX_TYPE_SET2_SIZES = 3`, and
  the §8.3.2 `Filter_Intra_Mode_To_Intra_Dir[ INTRA_FILTER_MODES ]`
  table mapping each filter mode to a directional anchor
  (`{ DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED }`).
  `new_from_defaults` performs the §8.3.1 init step for both arrays.
  An `&mut [u16]` accessor `intra_tx_type_cdf(set, tx_size_sqr, intra_dir)`
  surfaces the §8.3.2 selection — the two-way
  `TileIntraTxTypeSet{1,2}Cdf` switch keyed by the §5.11.48 `set`
  return and indexed on the `intraDir` axis — yielding `None` for
  `TX_SET_DCTONLY` and for any unreachable `(set, tx_size_sqr,
  intra_dir)` combination. Two scalar §5.11.48 / §8.3.2 helpers
  complete the path: `intra_tx_type_set(tx_sz_sqr, tx_sz_sqr_up,
  reduced_tx_set)` computes the `set ∈ { TX_SET_DCTONLY,
  TX_SET_INTRA_1, TX_SET_INTRA_2 }` from the surrounding §5.11.47
  syntax (differing from the inter branch in two places:
  `txSzSqrUp == TX_32X32` itself routes to `TX_SET_DCTONLY` rather
  than `TX_SET_INTER_3`, and `txSzSqr == TX_16X16` routes to
  `TX_SET_INTRA_2` rather than `TX_SET_INTER_2`), and
  `intra_dir(use_filter_intra, y_mode, filter_intra_mode)` derives
  the `intraDir` axis from the `use_filter_intra` flag plus
  the `YMode` / `filter_intra_mode` pair. 7 new unit tests:
  every intra-tx-type table's `cdf[N-1] == 32768` / `cdf[N] == 0`
  invariant + dimension audit against the §3 constants; byte-anchor
  spot-checks on every table's first / last entries plus the explicit
  flat-distribution check for `Set2` sizes 0..=1; §8.3.1 init-copy
  independence with a mutate-doesn't-touch-source assertion;
  `intra_tx_type_cdf` two-way selection over every reachable
  `(set, tx_size_sqr, intra_dir)` triple with row-length assertions
  and unreachable-set / out-of-range coverage; the
  `intra_tx_type_set` formula walked across every reachable
  `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)` triple (including the
  rectangular `TX_4X8` / `TX_16X32` cases and the `txSzSqrUp == TX_32X32`
  short-circuit specific to the intra branch); the `intra_dir`
  derivation tested for the pass-through (`use_filter_intra == 0`)
  and the `Filter_Intra_Mode_To_Intra_Dir` lookup branches; and one
  end-to-end §8.2 `SymbolDecoder` decode driving the 5-symbol
  `TileIntraTxTypeSet2Cdf[ 2 ][ 0 ]` row selected by
  `intra_tx_type_set(2, 2, false) = TX_SET_INTRA_2` +
  `intra_dir(false, DC_PRED, _) = DC_PRED`. This completes §9.4's
  transform-type coverage (intra + inter); remaining §9.4 work is
  the `init_coeff_cdfs` coefficient set (`Default_Coeff_Base_Eob_Cdf`
  / `Default_Coeff_Base_Cdf` / `Default_Coeff_Br_Cdf`) and the
  inter-intra group.

Validation: all 16 IVF fixtures under
`docs/video/av1/fixtures/` (`tiny-i-only-16x16-prof0`,
`i-only-64x64-prof0`, `profile-1-yuv444-8bit`,
`profile-2-yuv422-10bit`, `profile-2-yuv422-12bit`,
`monochrome-grey-only`, `super-resolution`, `screen-content-tools`,
`film-grain-on`, `superblocks-128`, `tile-cols-2-rows-1`,
`show-existing-frame`, `lossless-i-only`, `i-frame-then-p-64x64`,
`obu-with-extension-headers`, `profile-0-yuv420-8bit`) round-trip
both the first sequence header bit-exact against the `SEQ_HEADER`
line captured in each fixture's `trace.txt`, and the first frame
OBU's leading uncompressed-header slice bit-exact against the
`FRAME_HEADER idx=0` line in the same trace. Round 4 extends the
trace columns asserted per fixture from 12 to 17 (adding `w`, `h`,
`use_superres`, `coded_denom`, plus a derived assertion ladder
computing `superres_denom` / post-superres `frame_width` /
`mi_cols` / `mi_rows` from the §5.9.5 / §5.9.8 / §5.9.9 formulas).
The `super-resolution` fixture exercises the §5.9.8 downscale
(`UpscaledWidth = 128`, `coded_denom = 3` ⇒ `SuperresDenom = 12`,
post-downscale `FrameWidth = (128 * 8 + 6) / 12 = 85`,
`MiCols = 22`); every other fixture is `use_superres == 0` with
`FrameWidth == UpscaledWidth`.

Both the intra and the inter `uncompressed_header()` are now parsed
end-to-end (through `film_grain_params()`). What remains: the
`set_frame_refs()` short-signaling ordering is implemented and unit-
tested but not yet exercised by a corpus fixture (no short-signaling
bitstream exists in `docs/video/av1/fixtures/`); the
`frame_size_with_refs()` `found_ref == 1` branch is implemented but
likewise unexercised by the corpus; the §5.9.2 OrderHints[] /
RefFrameSignBias[] derivation and the §7.20 reference frame update
process (which would *store* a decoded frame's dimensions / hints back
into `RefInfo` across frames) are session-state concerns left to the
decode pipeline; and tile-content decode (motion vectors, transform /
quantisation, in-loop filters, film-grain synthesis) is unstarted. The
§8.2 symbol (arithmetic) decoder — the engine all tile-content reads run
on — now exists as a standalone, byte-exact `SymbolDecoder` (round 15);
round 16 lands the §9.4 default CDF tables and the §8.3.1 / §8.3.2
selection for a bounded **intra-frame mode / partition** syntax group
(`intra_frame_y_mode` / `partition` / `skip` / `segment_id`); round 17
extends the same `TileCdfContext` shape with the **motion-vector
component** subset (`mv_joint` / `mv_sign` / `mv_class` /
`mv_class0_bit` / `mv_class0_fr` / `mv_class0_hp` / `mv_bit` / `mv_fr`
/ `mv_hp`) and the §5.11.31 `MvCtx` derivation; round 18 extends it
again with the **inter-mode / reference-frame** subset (`new_mv` /
`zero_mv` / `ref_mv` / `drl_mode` / `is_inter` / `comp_mode` /
`skip_mode` / `comp_ref{,_p1,_p2}` / `comp_bwdref{,_p1}` /
`single_ref_p{1..6}` / `compound_mode` / `comp_ref_type` /
`uni_comp_ref{,_p1,_p2}`) plus the §8.3.2 context helpers
`is_inter_ctx` / `skip_mode_ctx` / `ref_count_ctx` / `compound_mode_ctx`.
round 20 extends the same `TileCdfContext` shape with the
**transform-size** subset (`tx_depth` over the four
per-`maxTxDepth` `Default_Tx_{8,16,32,64}x{8,16,32,64}_Cdf` arrays
and `txfm_split` over `Default_Txfm_Split_Cdf`) plus the §8.3.2
`tx_depth_ctx` / `txfm_split_ctx` derivations. Round 21 lands the
**inter-frame transform-type** subset (`inter_tx_type` over
`Default_Inter_Tx_Type_Set{1,2,3}_Cdf`) plus the §5.11.48
`inter_tx_type_set` switch driving the §8.3.2 three-way
`TileInterTxTypeSet{1,2,3}Cdf` selection. Round 22 lands the
**inter-frame interpolation-filter** subset (`interp_filter` over
`Default_Interp_Filter_Cdf` — 16 contexts × 3 cumulative
frequencies) plus the §8.3.2 four-branch `interp_filter_ctx`
formula (the `((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4`
base plus the leftType / aboveType / NONE-match folding). Round 23
lands the **motion-mode** subset (`motion_mode` over
`Default_Motion_Mode_Cdf` — 22 block-size rows × 3 cumulative
frequencies) plus its §8.3.2 selection — a straight
`TileMotionModeCdf[ MiSize ]` index with no neighbour-context
arithmetic; the §6.10.26 enumeration `MOTION_MODES = 3`
(`SIMPLE` / `OBMC` / `LOCALWARP`) is added as a new §3 constant.
Round 24 lands the **compound-prediction** subset — the three
default tables `Default_Comp_Group_Idx_Cdf` /
`Default_Compound_Idx_Cdf` (binary, 6 contexts each) and
`Default_Compound_Type_Cdf` (22 block-size rows × 2 cumulative
frequencies) — plus their §8.3.2 selections:
`comp_group_idx_cdf(ctx)` / `compound_idx_cdf(ctx)` take the
precomputed neighbour-derived context (whose arithmetic stays in
the future tile walk) and `compound_type_cdf(mi_size)` is a
straight `TileCompoundTypeCdf[ MiSize ]` index; the §3 constants
`COMPOUND_TYPES = 2`, `COMP_GROUP_IDX_CONTEXTS = 6` and
`COMPOUND_IDX_CONTEXTS = 6` are added.
Round 134 lands the **inter-frame intra-mode** subset — the three
default tables `Default_Y_Mode_Cdf` (4 block-size-group contexts ×
13 cumulative frequencies), `Default_Uv_Mode_Cfl_Not_Allowed_Cdf`
(13 `YMode` rows × 13) and `Default_Uv_Mode_Cfl_Allowed_Cdf`
(13 `YMode` rows × 14) — plus their §8.3.2 selections:
`y_mode_cdf(ctx)` indexes `TileYModeCdf[ Size_Group[ MiSize ] ]`
(the `Size_Group` table + `size_group()` helper land alongside),
and `uv_mode_cdf(cfl_allowed, y_mode)` picks the cfl-allowed /
cfl-not-allowed variant by the resolved flag (the `Lossless` /
`get_plane_residual_size` / `Max(Block_Width, Block_Height) <= 32`
derivation stays in the future tile walk) then indexes by `YMode`;
the §3 constants `BLOCK_SIZE_GROUPS = 4`,
`UV_INTRA_MODES_CFL_NOT_ALLOWED = 13` and
`UV_INTRA_MODES_CFL_ALLOWED = 14` are added.
Round 135 lands the **angle-delta** subset — the default table
`Default_Angle_Delta_Cdf` (8 directional-mode rows × 7 cumulative
frequencies) — plus its §8.3.2 selection: `angle_delta_cdf(mode)`
indexes `TileAngleDeltaCdf[ mode - V_PRED ]` (the
`TileAngleDeltaCdf[ YMode - V_PRED ]` / `[ UVMode - V_PRED ]`
selection shared by `angle_delta_y` / `angle_delta_uv`), returning
`None` for non-directional modes; the §3 constants
`DIRECTIONAL_MODES = 8`, `MAX_ANGLE_DELTA = 3` and the
directional-mode base `V_PRED = 1` are added.
Round 136 lands the **coefficient-token entry sub-group** — the
`init_coeff_cdfs` gateway to tile-content decode: the
transform-block skip flag `Default_Txb_Skip_Cdf` (4 q-contexts ×
5 transform sizes × 13 skip contexts), the end-of-block position
classes `Default_Eob_Pt_{16,32,64,128,256}_Cdf` (per-plane,
per-`isInter`) plus the no-`isInter`-axis
`Default_Eob_Pt_{512,1024}_Cdf`, the binary `Default_Eob_Extra_Cdf`
(per transform size / plane / 9 EOB contexts) and the binary
`Default_Dc_Sign_Cdf` (per plane / 3 contexts, in the §9.4
`128 * N` fixed-point form), all transcribed verbatim from §9.4.
Unlike the non-coeff CDFs these are reset by the separate
`TileCdfContext::init_coeff_cdfs`, which derives the q-context
`idx` from `base_q_idx` (`coeff_cdf_q_ctx`: `<=20→0`, `<=60→1`,
`<=120→2`, else `3`) and copies `Default_*_Cdf[ idx ]` into the
working arrays; the §8.3.2 selectors `txb_skip_cdf` /
`eob_pt_*_cdf` / `eob_extra_cdf` / `dc_sign_cdf` land alongside,
and the §3 constants `PLANE_TYPES = 2`, `COEFF_CDF_Q_CTXS = 4`,
`TXB_SKIP_CONTEXTS = 13`, `EOB_COEF_CONTEXTS = 9`,
`DC_SIGN_CONTEXTS = 3` are added.
Round 137 completes §9.4's transform-type coverage by adding the
**intra-frame transform-type** subset (`intra_tx_type` over
`Default_Intra_Tx_Type_Set{1,2}_Cdf`) plus the §5.11.48
`intra_tx_type_set` switch driving the §8.3.2 two-way
`TileIntraTxTypeSet{1,2}Cdf` selection and the §8.3.2 `intra_dir`
helper that derives the `intraDir` axis from `use_filter_intra` +
`YMode` / `filter_intra_mode` via the
`Filter_Intra_Mode_To_Intra_Dir` table.
Round 138 lands the first member of the `coeff_base` /
`coeff_base_eob` / `coeff_br` braid — the **`coeff_base_eob`
sub-group** — by adding `Default_Coeff_Base_Eob_Cdf`
(`[COEFF_CDF_Q_CTXS=4][TX_SIZES=5][PLANE_TYPES=2][SIG_COEF_CONTEXTS_EOB=4][4]`)
transcribed verbatim from §9.4. `coeff_base_eob` codes the base
level of the last non-zero coefficient (the base level is
`coeff_base_eob + 1`, restricted to 1, 2, or 3, so only three
symbols are coded). `init_coeff_cdfs` grows the new
`self.coeff_base_eob = DEFAULT_COEFF_BASE_EOB_CDF[ idx ]` copy on
the `base_q_idx`-derived `idx`, and the §8.3.2 selector
`coeff_base_eob_cdf(tx_sz_ctx, ptype, ctx)` surfaces the
`TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]` lookup. The §8.3.2
ctx derivation
(`get_coeff_base_ctx() - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`)
belongs to the not-yet-implemented tile-content walk and is
deferred along with `Default_Coeff_Base_Cdf` and
`Default_Coeff_Br_Cdf`; the §3 constant `SIG_COEF_CONTEXTS_EOB = 4`
is added.
Round 139 lands the second member of the braid — the **`coeff_base`
sub-group** — by adding `Default_Coeff_Base_Cdf`
(`[COEFF_CDF_Q_CTXS=4][TX_SIZES=5][PLANE_TYPES=2][SIG_COEF_CONTEXTS=42][5]`,
1680 5-entry rows = 16800 bytes; declared `static` to satisfy
`clippy::large_const_arrays`) transcribed verbatim from §9.4.
`coeff_base` codes the base level of every non-EOB coefficient — the
4-symbol alphabet `0..3`, so each row carries 4 cumulative
frequencies plus the §8.3 adaptation counter. `init_coeff_cdfs`
grows the new `self.coeff_base = DEFAULT_COEFF_BASE_CDF[ idx ]`
copy on the `base_q_idx`-derived `idx`, and the §8.3.2 selector
`coeff_base_cdf(tx_sz_ctx, ptype, ctx)` surfaces the
`TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]` lookup. The §3
constant `SIG_COEF_CONTEXTS = 42` is added (the §3 partition tag
`SIG_COEF_CONTEXTS_2D = 26` splits this range between the
two-dimensional scan prefix and the 1D horizontal- / vertical-only
tails). Just as in r138, the largest `(TX_SIZE = TX_64X64, ptype =
chroma)` slice is the flat `{8192, 16384, 24576, 32768, 0}`
placeholder for every q-context and ctx value — a sentinel for an
unreachable chroma row at the largest TX size — and is locked down
by an exhaustive byte-equality test.
Round 140 lands the LAST member of the braid — the **`coeff_br`
sub-group** — by adding `Default_Coeff_Br_Cdf`
(`[COEFF_CDF_Q_CTXS=4][TX_SIZES=5][PLANE_TYPES=2][LEVEL_CONTEXTS=21][BR_CDF_SIZE + 1 = 5]`,
840 5-entry rows = 8400 bytes; declared `static` to satisfy
`clippy::large_const_arrays`) transcribed verbatim from §9.4. With
this table all three coefficient-CDF braid members are landed.
`coeff_br` codes the per-coefficient base-range increment used to
push a level above `NUM_BASE_LEVELS`: each read codes a value in
`0..BR_CDF_SIZE = 4`, and §5.11.39 stacks
`COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)` such reads per coefficient.
`init_coeff_cdfs` grows the new
`self.coeff_br = DEFAULT_COEFF_BR_CDF[ idx ]` copy on the
`base_q_idx`-derived `idx`, and the §8.3.2 selector
`coeff_br_cdf(tx_sz_ctx, ptype, ctx)` surfaces the
`TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][ ctx ]` lookup
with the `TX_32X32 = 3` clamp built in (so any `txSzCtx` is
accepted; only `ptype` / `ctx` are bounds-checked). New §3
constants `LEVEL_CONTEXTS = 21` and `BR_CDF_SIZE = 4`. Mirroring
r138 / r139, the largest `(TX_SIZE = TX_64X64, ptype = chroma)`
slice is the flat `{8192, 16384, 24576, 32768, 0}` placeholder for
every q-context and ctx value, locked down by an exhaustive
byte-equality test. The next gate is the §8.3.2
`get_coeff_base_ctx()` / `get_br_ctx()` neighbour-derivation helpers,
deferred to a different round because they need tile-content walker
state.
Round 141 lands those **§8.3.2 `get_coeff_base_ctx()` /
`get_br_ctx()` neighbour-derivation helpers** — the per-coefficient
`ctx` computation that feeds the r138–r140 selectors. Both are free
functions that take the coefficient-magnitude array `Quant` plus
scalar transform / position state (`tx_size`, `tx_class`, `pos`,
`c`, `is_eob`) and return the `ctx` index; they own the §8.3.2
neighbour scan only — the tile-content walker that produces `Quant`
and the `compute_tx_type()` derivation is the next gate.
`get_coeff_base_ctx()` scans `Sig_Ref_Diff_Offset`
(`SIG_REF_DIFF_OFFSET_NUM = 5` offsets) accumulating
`Min(Abs(Quant[(refRow<<bwl)+refCol]), 3)` over in-bounds neighbours
(`refRow < height && refCol < width`, `bwl =
Tx_Width_Log2[Adjusted_Tx_Size[txSz]]`), forms `ctx = Min((mag+1)>>1,
4)`, then routes through the 2D `Coeff_Base_Ctx_Offset[txSz][Min(row,
4)][Min(col,4)]` branch (with the `row==0 && col==0 -> 0` early
return) or the 1D `Coeff_Base_Pos_Ctx_Offset[Min(idx,2)]` branch;
the `isEob` path returns the `SIG_COEF_CONTEXTS-{4,3,2,1}` buckets per
`c` thresholds. A `get_coeff_base_eob_ctx()` wrapper applies the
§8.3.2 `- SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB` reduction.
`get_br_ctx()` scans `Mag_Ref_Offset_With_Tx_Class` (3 offsets)
accumulating `Min(Quant[refRow*txw+refCol],
COEFF_BASE_RANGE+NUM_BASE_LEVELS+1)` (no abs, distinct clamp, bound
`refRow < txh && refCol < (1<<bwl)`), forms `mag = Min((mag+1)>>1,
6)`, then `pos==0 -> mag`; 2D `+7/+14` on `(row<2 && col<2)`;
horizontal `+7/+14` on `col==0`; vertical `+7/+14` on `row==0`
(result in `0..LEVEL_CONTEXTS`). Adds the §3 constants
`SIG_COEF_CONTEXTS_2D = 26`, `SIG_REF_DIFF_OFFSET_NUM = 5`,
`NUM_BASE_LEVELS = 2`, `COEFF_BASE_RANGE = 12`, `TX_SIZES_ALL = 19`,
the `TX_CLASS_{2D,HORIZ,VERT}` tags, the `Adjusted_Tx_Size` /
`Tx_Width` / `Tx_Height` / `Tx_Width_Log2` size tables, and the
`Sig_Ref_Diff_Offset` / `Mag_Ref_Offset_With_Tx_Class` /
`Coeff_Base_Ctx_Offset` / `Coeff_Base_Pos_Ctx_Offset` tables, all
transcribed verbatim from the spec; a pure `get_tx_class()` helper
reduces the directional transform-type flags to a class. 12 new unit
tests (266 -> 278) pin each branch with hand-computed `ctx` values.
Round 142 follows up with the **§5.11.40 `compute_tx_type()`
derivation** — `compute_tx_type(plane, tx_sz, lossless, is_inter,
tx_set, mi_row, mi_col, block_x, block_y, subsampling_x,
subsampling_y, uv_mode, tx_types)` implements the full spec function
the tile-content walker reads before kicking off coefficient
decoding. `Lossless || Tx_Size_Sqr_Up[ txSz ] > TX_32X32`
short-circuits to `DCT_DCT`; `plane == 0` returns the
`TxTypes[ blockY ][ blockX ]` luma cache entry; `is_inter` chroma
reads the cache at
`(Max(MiRow, blockY << subsampling_y), Max(MiCol, blockX <<
subsampling_x))` then runs the §5.11.40 `is_tx_type_in_set`
admission filter; intra chroma reads `Mode_To_Txfm[UVMode]` then
runs the same filter. The caller supplies the §5.11.40 `txSet`
(i.e. the already-resolved `inter_tx_type_set` /
`intra_tx_type_set` result) and a closure over `TxTypes[y][x]` so
the helper does not bake in a particular storage shape — a dense 2D
array, a sparse map, or a `MiRow/MiCol`-relative tile-local view
all work. Adds the §6.10.16 size ordinals
`TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` / `TX_64X64`
(replacing the previously locally-scoped `const TX_*` shadows
inside `inter_tx_type_set` / `intra_tx_type_set`), the §6.10.19
transform-type ordinals `DCT_DCT` through `H_FLIPADST` (16
entries), the `TX_SET_TYPES_INTRA = 3` / `TX_SET_TYPES_INTER = 4`
row-count constants, the `Tx_Size_Sqr_Up[ TX_SIZES_ALL ]`,
`Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]`,
`Tx_Type_In_Set_Intra[ 3 ][ TX_TYPES ]`, and
`Tx_Type_In_Set_Inter[ 4 ][ TX_TYPES ]` tables, all transcribed
verbatim from the spec. 10 new unit tests (278 -> 288) pin each
branch (lossless / `txSzSqrUp > TX_32X32` fallback, luma cache
read, inter-chroma `Max` lift + admission pass/fail, intra-chroma
`Mode_To_Txfm` + admission pass/fail + out-of-range, plus a
selector/admission closed-loop check).
Round 143 lands the **inter-intra group** — the three §9.4 default
CDFs read by the §5.11.28 `read_interintra_mode` syntax:
`Default_Inter_Intra_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ 3 ]`,
`Default_Inter_Intra_Mode_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ INTERINTRA_MODES + 1 ]`,
and `Default_Wedge_Inter_Intra_Cdf[ BLOCK_SIZES ][ 3 ]`. Adds the
`INTERINTRA_MODES = 4` §3 constant (per §6.10.27 — `II_DC_PRED` /
`II_V_PRED` / `II_H_PRED` / `II_SMOOTH_PRED`) and the §8.3.2
`interintra_ctx(mi_size) = Size_Group[ MiSize ] - 1` mapping
(returning `None` for the `Size_Group[ MiSize ] == 0` rows that the
§5.11.28 syntax gate excludes — `MiSize < BLOCK_8X8`). The
`TileCdfContext` grows the `inter_intra` / `inter_intra_mode` /
`wedge_inter_intra` fields and gains the
`inter_intra_cdf(ctx)` / `inter_intra_mode_cdf(ctx)` /
`wedge_inter_intra_cdf(mi_size)` selectors. The wedge table's outer
dimension is transcribed full-width per the §9.4 listing; per its
note only indices `3..=9` (the `BLOCK_8X8`..`BLOCK_32X32` band) are
reachable and the other rows hold the placeholder
`{16384, 32768, 0}` row. 8 new unit tests (288 -> 296) pin the table
shapes / well-formedness / `Size_Group - 1` mapping / per-row
selector return value / working-copy independence, plus end-to-end
`SymbolDecoder` reads through `interintra`, `interintra_mode`, and
`wedge_interintra` rows.
Round 144 lands the **wedge-index CDF** — the §9.4
`Default_Wedge_Index_Cdf[ BLOCK_SIZES ][ WEDGE_TYPES + 1 ]` table
(p.435) and the matching §8.3.2 selection. `wedge_index` is the
16-symbol element read by both §5.11.28 `read_interintra_mode` (the
inter-intra wedge sub-branch, when `wedge_interintra == 1`) and
§5.11.29 `read_compound_type` (the inter-inter `COMPOUND_WEDGE`
branch). Adds the §3 constant `WEDGE_TYPES = 16` (the spec text reads
*"Number of directions for the wedge mask process"*). `TileCdfContext`
grows a `wedge_index` field and gains a
`wedge_index_cdf(mi_size) -> Option<&mut [u16]>` selector (straight
`TileWedgeIndexCdf[ MiSize ]` indexing). The table's outer dimension is
transcribed full-width per the §9.4 listing; per its note (p.436)
indices `0..2`, `10..17`, and `20..21` are never used in the first
dimension (matching the §3 `Wedge_Bits[ MiSize ] == 0` rows) and carry
the placeholder uniform CDF `{ 2048, 4096, …, 30720, 32768, 0 }` (step
`32768 / WEDGE_TYPES`). 6 new unit tests (296 -> 302) pin the §3
constant / table shape and values (cross-checked against the §3
`Wedge_Bits` table) / well-formedness / `init_non_coeff_cdfs` seeding /
selector return value with out-of-range rejection / working-copy
independence, plus an end-to-end `SymbolDecoder` read through a
`wedge_index` row from the reachable band.
Round 145 lands the §8.3.2 **`split_or_horz` / `split_or_vert`
cdf-derivation helpers** (p.362–363) — two pure functions that turn an
already-selected `partition` cdf into a 2-symbol binary cdf by folding
the §9.4 partition probabilities of the splittable plus orthogonal-axis
symbols into a single `psum`. Per the §8.3.2 note the disallowed
orthogonal partition's probability is folded into the split branch
(`split_or_horz` cannot return `PARTITION_VERT`; `split_or_vert` cannot
return `PARTITION_HORZ`). The `b_size != BLOCK_128X128` guard drops the
`PARTITION_*_4` term that the §9.4 `Default_Partition_W128_Cdf` row has
no entry for. Adds the §3 / §6.10.4 partition ordinal constants
(`PARTITION_NONE` through `PARTITION_VERT_4` plus `EXT_PARTITION_TYPES =
10`) and the block-size ordinal `BLOCK_128X128 = 15` that the §8.3.2
formulas reference by name. 10 new unit tests (302 -> 312) pin the
§6.10.4 ordinals against the spec table (p.172), validate the
W{16,32,64,128} row-length budget against the §8.3.2 indexing reach,
re-derive both helpers' `psum` inline against a known
`Default_Partition_W{16,32}_Cdf` row, exercise the `PARTITION_*_4`
omission for both helpers under `b_size == BLOCK_128X128`, sweep §8.2.6
well-formedness across every default partition cdf row, drive the §8.2
`SymbolDecoder` through both derived binary cdfs (`BLOCK_64X64` for
`split_or_horz`, `BLOCK_128X128` for `split_or_vert`), and reject the
disallowed `bsl == 1` W8 row with `None`.
Round 146 lands the §5.11.50 **`get_palette_color_context` derivation**
(p.103) — the function the §5.11.49 diagonal-walk reads at each
`palette_color_idx_*` position to derive the `ColorOrder[ PALETTE_COLORS ]`
permutation + `ColorContextHash` that flow back through
[`palette_color_ctx`] into the §8.3.2 cdf selector. Surface:
`palette_color_context_from_neighbors(left, above_left, above, n) ->
Option<PaletteColorContext>` (pure-scoring core taking the three optional
neighbour palette indices) and `get_palette_color_context(color_map,
stride, r, c, n) -> Option<PaletteColorContext>` (spec-faithful 2-D
entry that applies the §5.11.50 `r > 0` / `c > 0` boundary guards). The
partial selection sort is the §5.11.50 three-iteration loop that
promotes the top-scoring neighbours to the head of `ColorOrder` while
preserving the runners-up's ascending order; the hash is the
`Palette_Color_Hash_Multipliers`-weighted sum of the top three sorted
scores. 11 new unit tests (312 -> 323) cover the spec example (all-same
neighbour, hash 5, ctx 4), distinct left/above (hash 6, ctx 3), the
partial-sort swap (two-of-three neighbours sharing an index, hash 6,
ctx 3), three distinct neighbours (hash 8, ctx 1), the no-neighbour
identity, palette-size / neighbour-value rejection, the full
spec-realisable combinatorial sweep across every `(left, above_left,
above)` at every palette size (every reachable hash maps to a
`Some(_)` ctx; the `-1` entries 0/1/3/4 are unreachable), 2-D
entry-point equivalence (interior + top-left corner + top-row-only +
left-column-only positions), 2-D boundary rejection (zero stride / OOB
column / OOB palette size / OOB row), and an end-to-end `SymbolDecoder`
read through the `palette_color_idx_y` default cdf selected by the
derivation -> `palette_color_ctx` -> `palette_y_color_cdf` chain.
The remaining §8.3.2 selections (the tile-content walker plumbing
that wires `get_palette_color_context` into the §5.11.49 diagonal walk,
plus the corresponding wedge / inter / intra walks for the unwalked
syntax elements …) are a mechanical followup against the same
`TileCdfContext` shape.
Round 147 lands the §5.11.49 **`palette_tokens( )` per-plane diagonal
walker** (p.101–102) — the caller-facing entry that drives the §5.11.50
colour-context derivation across an anti-diagonal walk, decodes one
`palette_color_idx_{y,uv}` per `(i - j, j)` against the cdf row picked
by `palette_color_ctx`, remaps through `ColorOrder[idx]`, and replicates
the on-screen last column / last row into the block's border band.
Surface: `palette_tokens_plane(dec, tile_ctx, plane, palette_size,
block_w, block_h, onscreen_w, onscreen_h, color_index_map, color_map,
stride) -> Result<(), Error>` with `PalettePlane::{Y, Uv}` picking
between `palette_y_color_cdf` and `palette_uv_color_cdf`. The
chroma-subsampling adjustments (`blockWidth >> subsampling_x` and the
`< 4` bump) are the caller's responsibility — they belong to the
§5.11.49 outer-control flow, not the walker. Two new `Error` variants
surface caller-bug preconditions (`InvalidPaletteWalkArgs`) and the
§5.11.50 unreachable hash slots (`PaletteColorContextUnmapped`); the
`SymbolDecoder` underflow path still propagates as `UnexpectedEnd`.
11 new unit tests (323 -> 334) cover every caller-bug rejection, a 2x2
on-screen walk that writes every cell with no border-fill, the
horizontal / vertical / combined border-fill paths on a 2x2 / 4x4
shape, a rectangular shape sweep over every `(onscreen_w, onscreen_h)`
in `1..=4 × 1..=4`, the UV plane adapting only the UV cdf family, the
chroma-subsampled UV / Y shape parity, the `ColorOrder[idx]` remap on
the 2x2 edge positions, the degenerate 1-wide-block shape, and
`read_symbol` underflow propagating as `UnexpectedEnd` (not as a
walker-side caller-bug variant).
Round 148 stages the §9.3 **block-size conversion tables** (av1-spec
p.400–401) that the §5.11.49 caller needs to derive `block_w` / `block_h`
from a `MiSize`. The four `BLOCK_SIZES`-indexed lookup tables
(`MI_WIDTH_LOG2`, `MI_HEIGHT_LOG2`, `NUM_4X4_BLOCKS_WIDE`,
`NUM_4X4_BLOCKS_HIGH`) are transcribed verbatim with the spec
ordering (16 square/rectangular entries `BLOCK_4X4` ..
`BLOCK_128X128` followed by the seven `1:4` / `4:1` aspect-ratio
entries `BLOCK_4X16` .. `BLOCK_64X16`). The §3 constants `MI_SIZE = 4`
and `MI_SIZE_LOG2 = 2` land alongside them so the §9.3
`Block_Width[ x ] = 4 * Num_4x4_Blocks_Wide[ x ]` identity is encoded
as `NUM_4X4_BLOCKS_WIDE[ x ] << MI_SIZE_LOG2` rather than duplicated
as a numeric table. Six new `MiSize`-keyed accessors round-out the
surface: `block_width`, `block_height`, `num_4x4_blocks_wide`,
`num_4x4_blocks_high`, `mi_width_log2`, `mi_height_log2` — each a
`const fn` with a `debug_assert!(mi_size < BLOCK_SIZES)` bound. These
feed the §5.11.49 `palette_tokens_plane` caller staged in r147 and
unblock the wider §5.x reconstruction call sites (`bw4 =
Num_4x4_Blocks_Wide[ MiSize ]`) the parser will surface once
`read_block` is wired. 10 new unit tests (334 -> 344) cover the four
§9.3 tables pinned byte-for-byte at `BLOCK_SIZES = 22`, the §3
`MI_SIZE == 1 << MI_SIZE_LOG2` identity, the `Num_4x4_Blocks_* == 1
<< Mi_*_Log2` identity per §9.3, the canonical 22-entry expected
width/height vectors, the square diagonal `BLOCK_4X4` ..
`BLOCK_128X128` resolving to `n×n` luma sizes, the §5.11.46
`bsizeCtx` derivation staying inside `0..PALETTE_BLOCK_SIZE_CONTEXTS`
for every `MiSize` inside the §5.11.46 palette syntax gate
(`MiSize >= BLOCK_8X8 && Block_Width <= 64 && Block_Height <= 64`),
and a §5.11.49 caller data-flow pin confirming `block_width(mi_size)`
/ `block_height(mi_size)` are inside `8..=64` at the palette-minimum
(`BLOCK_8X8`) and palette-maximum (`BLOCK_64X64`) `MiSize` values
the gate admits.

Round 149 wires the §5.11.49 **caller-side argument derivation**
(av1-spec p.101–102) on top of the r148 tables. The new
`palette_tokens_args(mi_size, mi_row, mi_col, mi_rows, mi_cols, plane,
subsampling_x, subsampling_y) -> Option<PaletteTokensArgs>` helper
returns the four `palette_tokens_plane` size arguments
(`block_w`, `block_h`, `onscreen_w`, `onscreen_h`) for one plane, and
a new `BLOCK_8X8` constant (`3`, the §5.11.46 palette gate) sits
alongside it. Y branch returns the §9.3 dimensions clipped by
`Min(.., (MiRows - MiRow) * MI_SIZE)` / `Min(.., (MiCols - MiCol)
* MI_SIZE)`. UV branch applies the §5.11.49 `>> subsampling_{x,y}`
shift and the `<4`-bump (`block_w += 2; onscreen_w += 2` when post-
shift `block_w < 4`, ditto height); the bump preserves the walker's
`onscreen_* <= block_*` invariant because it adds the same `2` to
both. The helper returns `None` for any §5.11.46 palette-gate
violation (sub-`BLOCK_8X8` size, `block_w > 64`, `block_h > 64`,
out-of-table index), out-of-bounds `mi_row` / `mi_col`, zero mi-grid,
or out-of-range subsampling flag — safe to call defensively from a
not-yet-gated caller. 15 new unit tests (344 -> 359) cover
`BLOCK_8X8`-row pinning; Y-plane fully-on-screen / right-edge /
bottom-edge clipping; UV 4:2:0 minimum block + large block;
UV 4:2:0 width-`<4`-bump (`BLOCK_4X16`) and height-`<4`-bump
(`BLOCK_16X4`); UV 4:2:2 + UV 4:4:4 shape; UV right-edge clip carry-
through; an exhaustive sweep proving `1 <= onscreen_* <= block_* <=
64` over every palette-eligible `(MiSize, sub_x, sub_y, plane)`
combination; palette-gate + caller-bug rejection paths; and an end-
to-end shape test that feeds the helper's output straight into
`palette_tokens_plane` against the §9.4 default palette CDFs,
confirming the walker's `InvalidPaletteWalkArgs` guard never fires
on conformant arguments. This closes the data-flow gap pinned by the
r147 follow-up test and leaves `read_block` clear to call
`palette_tokens` once the parser surfaces the variables.
`decode_av1` and `encode_av1` still return `Error::NotImplemented`.

Round 150 stages the §9.3 **`Partition_Subsize[ 10 ][ BLOCK_SIZES ]`**
lookup (av1-spec p.402–403) plus the §3 enumeration of all 22 named
`BLOCK_*` ordinals and the `BLOCK_INVALID = 22` sentinel from the
§3 constant table (p.7). `PARTITION_SUBSIZE` is transcribed verbatim;
every rectangular `bSize` column carries `BLOCK_INVALID` per the
spec p.401 note "The table will never get accessed for rectangular
block sizes". The typed accessor
`partition_subsize(partition, b_size) -> Option<usize>` folds the
sentinel into `None` so the upcoming §5.11.4 `decode_partition()`
body — which reads
`subSize = Partition_Subsize[ partition ][ bSize ]` and
`splitSize = Partition_Subsize[ PARTITION_SPLIT ][ bSize ]` side
by side, then dispatches `decode_block` / `decode_partition` per
the resolved size — never silently hands the `22` sentinel to its
recursive children. 16 new unit tests (359 -> 375) cover BLOCK_*
ordinal pinning (`BLOCK_4X4 = 0` through `BLOCK_INVALID = 22`);
`PARTITION_TYPES_TOTAL` (`10`) pin; table-shape pin; row-0
(PARTITION_NONE) identity on every square; row-1 / row-2 / row-3
halving / quartering identities; row-4..7 (`_A` / `_B`) row equality
with rows 1 / 2; row-8..9 (`_4`) quarter-splits + `BLOCK_128X128`
drop; `BLOCK_4X4`-only-resolves-for-`PARTITION_NONE` column-0
invariant; exhaustive rectangular-`bSize`-is-invalid coverage; every
resolved subSize in `0..BLOCK_SIZES`; out-of-range
(`partition >= 10`, `b_size >= BLOCK_SIZES`) guard; the §5.11.4
subdivision-shrinks-area invariant; and the §5.11.4 `subSize` +
`splitSize` pair well-formedness for every reachable HORZ_A /
HORZ_B parent block. The full `decode_partition` body remains the
next round's target; this round drops the last lookup it needs.
`decode_av1` and `encode_av1` still return `Error::NotImplemented`.

Round 151 lands the §5.11.4 **`decode_partition()` body** (av1-spec
p.61–62) — the recursive partition-tree walker that stitches every
partition prerequisite landed in rounds 137–150 into one
`PartitionWalker` type. The walker carries the §6.10.4
`MiSizes[r][c]` grid (filled at every leaf via the block's `bh4 *
bw4` footprint) and consults it for the §8.3.2 `partition_ctx`
derivation `above = AvailU && (Mi_Width_Log2[ MiSizes[r-1][c] ] <
bsl)` / `left = AvailL && (Mi_Height_Log2[ MiSizes[r][c-1] ] <
bsl)` on every recursive child (av1-spec p.362). It emits a
`Vec<DecodedBlockRecord>` of `(MiRow, MiCol, MiSize)` leaves in
§5.11.4 syntax order; the actual §5.11.5 `decode_block()` body
(coefficient / motion-vector / reconstruction) is the next round's
target. All four §5.11.4 edge-of-frame branches are honoured —
`r >= MiRows || c >= MiCols` early return, `bSize < BLOCK_8X8`
short-circuit to `PARTITION_NONE` with no symbol read,
`hasCols`-alone `split_or_horz`, `hasRows`-alone `split_or_vert`,
`!hasRows && !hasCols` fall-through to `PARTITION_SPLIT`. All ten
partition arms (`NONE` / `HORZ` / `VERT` / `SPLIT` / `HORZ_A` /
`HORZ_B` / `VERT_A` / `VERT_B` / `HORZ_4` / `VERT_4`) dispatch
the spec's literal `decode_block` / recursive `decode_partition`
calls with the appropriate `subSize` (`Partition_Subsize[
partition ][ bSize ]`) or `splitSize` (`Partition_Subsize[
PARTITION_SPLIT ][ bSize ]`). The §5.11.4 bottom-right edge clip
on the optional `HORZ_4` / `VERT_4` fourth leaf is applied
literally. New `TileGeometry` type carries the four §5.11.1
mi-unit tile bounds for the §5.11.51 `is_inside()` test. New
`Error::PartitionWalkOutOfRange` surfaces caller-bug preconditions
(`bSize` / `partition` / `bsl` / `Partition_Subsize` lookup out of
range). 19 new cdf-module tests (375 -> 394): TileGeometry
boundary cases; §5.11.4 early-return; the `bSize < BLOCK_8X8`
no-symbol-read short-circuit; §6.10.4 grid-fill; the `!hasRows &&
!hasCols` corner-case fallback; forced PARTITION_NONE / HORZ /
VERT / HORZ_4 / VERT_4 / HORZ_A / VERT_B / SPLIT at BLOCK_16X16
via rigged CDFs; forced-HORZ grid-fill; default-CDF W128 smoke
test; partition_ctx derivation at origin + after a wide-neighbour
leaf; construction overflow; `take_blocks` drain semantics.
`decode_av1` and `encode_av1` still return `Error::NotImplemented`
— the next round wires the §5.11.5 `decode_block()` body (block
coefficient decode + motion-vector decode + reconstruction)
behind the now-complete partition walker.

Round 152 lands the §5.11.11 **`read_skip()` syntax element**
(av1-spec p.65) as a new `PartitionWalker::decode_skip` method
plus a `Skips[r][c]` flag grid on the walker (parallel to the
§6.10.4 `MiSizes[]` grid landed in r151). Both spec branches are
honoured: the `SegIdPreSkip && seg_feature_active(SEG_LVL_SKIP)`
short-circuit (no symbol read, `skip = 1`) is taken when the
caller passes `seg_skip_active = true`; otherwise an `S()` symbol
is decoded against `TileSkipCdf[ctx]` with the §8.3.2 ctx `ctx =
AvailU * Skips[MiRow-1][MiCol] + AvailL * Skips[MiRow][MiCol-1]`
(av1-spec p.378). The decoded value is stamped over the block's
`bw4 * bh4` footprint of `Skips[]` per the §5.11.5 footer (clipped
at the frame's `MiRows` / `MiCols` extent). New
`PartitionWalker::skips()` accessor surfaces a row-major view of
the grid for downstream consumers. The combined precondition
`SegIdPreSkip && seg_feature_active(SEG_LVL_SKIP)` is computed
upstream by the frame parser — the walker stays segmentation-state
free. 11 new cdf-module tests (394 -> 405): seg short-circuit
returns 1 with no symbol read; else branch returns the rigged
symbol on a forced binary CDF; seg- + else-branch grid-stamp
invariant; origin ctx = 0; ctx-2 path through two prior
`Skips=1` neighbours; AvailL-false drops the left contribution;
non-zero tile origin clears AvailU / AvailL; right-edge `bw4`
clip; out-of-range `mi_row` / `mi_col` / `sub_size` ⇒
`PartitionWalkOutOfRange`; fresh-walker grid all-zero. The §5.11.5
`decode_block()` body itself (coefficient / motion-vector /
reconstruction) remains the next round's target. `decode_av1` and
`encode_av1` still return `Error::NotImplemented`.

Round 154 lands the §5.11.10 **`read_skip_mode()` syntax element**
(av1-spec p.67) as a new `PartitionWalker::decode_skip_mode` method
plus a `SkipModes[r][c]` flag grid on the walker (parallel to the
r152 `Skips[]` and the r151 `MiSizes[]` grids). The §5.11.10
short-circuit set (any-true ⇒ `skip_mode = 0`, no symbol read) is
honoured: `seg_feature_active( SEG_LVL_SKIP / REF_FRAME / GLOBALMV )`
collapsed into the caller-provided `seg_skip_mode_off`;
`!skip_mode_present` via the §5.9.21 frame-header scalar; and
`Block_Width[MiSize] < 8 || Block_Height[MiSize] < 8` derived
locally from `sub_size` via the §9.3 `block_width` / `block_height`
tables. Otherwise an `S()` symbol is decoded against
`TileSkipModeCdf[ctx]` with `ctx = AvailU *
SkipModes[MiRow-1][MiCol] + AvailL * SkipModes[MiRow][MiCol-1]`
(av1-spec p.378), routed through the existing `skip_mode_ctx`
helper. The §5.11.5 footer stamps the value over the block's
`bw4 * bh4` footprint of `SkipModes[]` (clipped at the frame's
`MiRows` / `MiCols` extent). New `PartitionWalker::skip_modes()`
accessor surfaces a row-major view. `skip_mode` is the
inter-frame compound-reference shortcut read in §5.11.18
`inter_frame_mode_info` before the rest of the inter mode decode
(intra-only frames never call this). 12 new cdf-module tests
(405 -> 417): seg short-circuit; `skip_mode_present` false
short-circuit; `Block_Width < 8` short-circuit (BLOCK_4X8);
`Block_Height < 8` short-circuit (BLOCK_8X4); else-branch S()
returning 0 / 1 on a forced binary CDF; footprint grid-stamp;
ctx-0 selection at the frame origin; ctx-1 single-neighbour and
ctx-2 both-neighbour paths; non-zero tile origin clearing AvailU
/ AvailL; right-edge `bw4` clip; out-of-range guards ⇒
`PartitionWalkOutOfRange`; fresh-walker grid all-zero. The
§5.11.5 `decode_block()` body itself (coefficient / motion-vector
/ reconstruction) remains the next round's target. `decode_av1`
and `encode_av1` still return `Error::NotImplemented`.

Round 174 lands the **§5.11.31 `assign_mv` + §5.11.32 `read_mv_component`
syntax tree** — wiring the per-block motion-vector decode into the
§5.11.23 inter cascade so the §5.11.31 / §5.11.32 leaves run
end-to-end. The r173 `Error::AssignMvUnsupported` stub is lifted; the
reader now short-circuits one step later at the new
`Error::MotionModeUnsupported` (§5.11.27 `read_motion_mode` is the
next-arc target — its body composes `use_obmc` / `motion_mode` S()
reads gated on §7.10.4 `find_warp_samples` + the §5.9.5
`force_integer_mv` / global-motion arms).

The §5.11.31 `assign_mv( isCompound )` body iterates over the active
reference lists (`i = 0..1 + isCompound`) and resolves each per-list
`Mv[ i ]` per the four-arm spec dispatch (av1-spec p.78):
`compMode = get_mode(i)` (the §5.11.30 helper, also lands this round);
`PredMv[ i ] = GlobalMvs[i]` when `compMode == GLOBALMV`, else
`PredMv[ i ] = RefStackMv[ pos ][ i ]` with `pos = (compMode ==
NEARESTMV) ? 0 : RefMvIdx` (forced to `0` when `compMode == NEWMV &&
NumMvFound <= 1`); finally `Mv[ i ] = PredMv[ i ] + diffMv` when
`compMode == NEWMV` (via §5.11.31 `read_mv`) or `Mv[ i ] = PredMv[ i ]`
otherwise (no MV bits read).

The §5.11.31 `read_mv( ref )` body composes one `mv_joint` S()
against `TileMvJointCdf[ MvCtx ]` (the 4-way joint code
`MV_JOINT_ZERO` / `MV_JOINT_HNZVZ` / `MV_JOINT_HZVNZ` /
`MV_JOINT_HNZVNZ`), then conditionally invokes
`read_mv_component( 0 )` and/or `read_mv_component( 1 )` per the
spec body's gating (`HZVNZ || HNZVNZ` for `diffMv[0]`,
`HNZVZ || HNZVNZ` for `diffMv[1]`), and finishes with
`Mv[ ref ][ c ] = PredMv[ ref ][ c ] + diffMv[ c ]`. `MvCtx` is
derived per the §5.11.31 `mv_ctx` helper (`MV_INTRABC_CONTEXT = 1`
when `use_intrabc == 1`, `0` otherwise); the inter caller always
passes `use_intrabc = false`.

The §5.11.32 `read_mv_component( comp )` body composes the full
two-branch sign-magnitude tree (av1-spec p.81-82): one `mv_sign` S()
against `TileMvSignCdf[ MvCtx ][ comp ]`, one `mv_class` S() against
`TileMvClassCdf[ MvCtx ][ comp ]` (one of `MV_CLASS_0..=MV_CLASS_10`),
then either the **MV_CLASS_0 ladder** (`mv_class0_bit` S() +
`mv_class0_fr` S() OR `= 3` when `force_integer_mv == 1` + `mv_class0_hp`
S() OR `= 1` when `allow_high_precision_mv == 0`, then
`mag = ((mv_class0_bit << 3) | (mv_class0_fr << 1) | mv_class0_hp) + 1`)
or the **MV_CLASS_K ladder** for `K >= 1` (per-bit `mv_bit` S() loop
`d |= mv_bit_i << i` for `i = 0..K`, then `mv_fr` / `mv_hp` reads with
the same `force_integer_mv` / `allow_high_precision_mv` gating, and
`mag = (CLASS0_SIZE << (mv_class + 2)) + ((d << 3) | (mv_fr << 1) |
mv_hp) + 1`). The signed return is `mv_sign ? -mag : mag`; the
§6.10.25 `is_mv_valid` conformance bound (`|Mv[ i ][ comp ]| < (1 <<
14)`) is the caller's responsibility.

The `Mvs[r][c][list][comp]` grid (introduced in r172 as a §7.10.2
neighbour-walk feed) is now stamped over the bh4 * bw4 footprint
after every `assign_mv` call. The pre-fill `0` value is replaced by
the decoded per-list Mv (cast `i32` → `i16`; the §6.10.25 bound keeps
the value in i16 range). Subsequent blocks' §7.10.2 spatial scans
therefore observe the decoded motion vectors instead of the
fresh-walker pre-fill.

`DecodedInterBlockModeInfo` gains a `mv: [[i32; 2]; 2]` field
carrying the §5.11.31 `Mv[ 0..2 ]` array. Slot `mv[0]` is always
written; `mv[1]` is only meaningful on compound blocks
(`is_compound == true`) — on single-pred blocks slot 1 stays at the
§5.11.5 pre-fill `[0, 0]`. The aggregate remains observable only on
the `Ok` path; the r174 dispatcher always returns `Err`
(`MotionModeUnsupported`) since the §5.11.27 `read_motion_mode` body
is not yet wired.

New `get_mode(y_mode, ref_list)` helper (§5.11.30) folds a YMode +
reference-list index into the per-list `compMode` used by §5.11.31
(one of `MODE_NEWMV` / `MODE_NEARESTMV` / `MODE_NEARMV` /
`MODE_GLOBALMV`). The single-pred branch (`ref_list == 0` AND
`YMode < NEAREST_NEARESTMV`) is the identity; the compound branch
applies the eight-row §6.10.22 → §5.11.30 mapping table.

Six new §3 / §6.10.27 / §6.10.28 named constants land:
`MV_JOINT_ZERO`, `MV_JOINT_HNZVZ`, `MV_JOINT_HZVNZ`,
`MV_JOINT_HNZVNZ` (the four §6.10.27 mv_joint ordinals) plus
`MV_CLASS_0` (the §5.11.32 small-magnitude class). The remaining
`MV_CLASS_K` for `K = 1..=10` reuses the literal `K` value (the spec
only names class 0 explicitly in the syntax body).

11 new unit tests cover: `get_mode` single-pred identity on
`ref_list = 0` (4 modes); `get_mode` compound table (8 modes ×
2 ref_lists); `assign_mv` skip_mode arm with NEAREST_NEARESTMV
producing zero MV bits read; `assign_mv` seg_globalmv arm using
identity GlobalMvs; `read_mv` NEWMV with `mv_joint = MV_JOINT_ZERO`
yielding zero diff (no per-component reads); `read_mv_component`
direct exercise of the MV_CLASS_0 ladder (three sub-cases: all-sym-0
with allow_hp ⇒ mag=1; all-sym-0 with hp=false fall-through ⇒ mag=2;
all-sym-0 with sign=1 ⇒ mag=-2); `force_integer_mv` short-circuiting
the `mv_class0_fr` read (forces fr=3 ⇒ mag=8); cascade structural
smoke (NEWMV cascade with default CDFs producing in-range Mv per
§6.10.25); and an `assign_mv` defensive guard (rejecting
`use_intrabc = true` in the inter arm). Test count: 661 → 672 (+11).
`decode_av1` / `encode_av1` still return `Error::NotImplemented`.

Round 175 lands the **§5.11.27 `read_motion_mode` reader** — with its
§7.10.3 `has_overlappable_candidates` and §7.10.4 `find_warp_samples`
helpers — wiring the per-block motion-mode decode into the §5.11.23
inter cascade. The r174 `Error::MotionModeUnsupported` stub is lifted;
the §5.11.18 dispatcher now reaches its `Ok(_)` arm and surfaces the
next-arc gap (`read_interintra_mode` / `read_compound_type` /
`read_interpolation_filter`) as `InterBlockModeInfoUnsupported`.

The §5.11.27 body composes the five short-circuit arms in spec order
(av1-spec p.79): `skip_mode == 1` ⇒ SIMPLE; `!is_motion_mode_switchable`
⇒ SIMPLE; `Min(Block_Width, Block_Height) < 8` ⇒ SIMPLE;
`!force_integer_mv && YMode ∈ {GLOBALMV, GLOBAL_GLOBALMV} &&
GmType[RefFrame[0]] > TRANSLATION` ⇒ SIMPLE; `isCompound ||
RefFrame[1] == INTRA_FRAME || !has_overlappable_candidates()` ⇒ SIMPLE.
Otherwise the body invokes §7.10.4 `find_warp_samples()` then either
the **`use_obmc` S()** arm (`force_integer_mv || NumSamples == 0 ||
!allow_warped_motion || is_scaled(RefFrame[0])`) returning
SIMPLE / OBMC, or the **`motion_mode` S()** arm returning
SIMPLE / OBMC / WARPED_CAUSAL directly.

The §7.10.3 `has_overlappable_candidates()` helper walks the 8x8-
granular above + left ref-frame grid (`(x4 | 1)` / `(y4 | 1)` probes
into `RefFrames[][][0]`) returning true on the first non-INTRA_FRAME
slot. The §7.10.4 `find_warp_samples()` walk runs the spec's
above-neighbour + left-neighbour + top-left + top-right `add_sample`
invocations, threading the §7.10.4.2 in-tile / written-cell /
matching-ref / single-list gates and the magnitude-validity check
(`Clip3(16, 112, Max(Block_Width, Block_Height))`-bounded
`|mvDiffRow| + |mvDiffCol|`). The §7.10.4.1 "first large MV kept if no
small one matches" special case is honoured (`NumSamples = 1` when all
scanned samples failed the magnitude bound). The §3
`LEAST_SQUARES_SAMPLES_MAX = 8` cap on `NumSamplesScanned` short-
circuits the walk per the §7.10.4.2 first-line guard. The `CandList`
itself is not surfaced (the §7.11.4 `warp_estimation()` consumer is a
downstream arc; the data is recoverable from the `Mvs[]` / `MiSizes[]`
walker grids on demand).

The default `Default_Use_Obmc_Cdf[BLOCK_SIZES][3]` table (§9.4) is
transcribed verbatim, surfaced through the new
`TileCdfContext::use_obmc` field + `use_obmc_cdf(mi_size)` selector.
The §9.4 note (indices `0..=2` and `16..=17` of the first dimension
unreachable) is honoured by the §5.11.27 prologue's `Min(BW, BH) >= 8`
gate. Three new §6.10.26 named constants land: `MOTION_MODE_SIMPLE`,
`MOTION_MODE_OBMC`, `MOTION_MODE_WARPED_CAUSAL`. Two new §3 constants
land: `LEAST_SQUARES_SAMPLES_MAX = 8`, `REF_SCALE_SHIFT = 14`.

Three new §5.11.27 frame-header scalars thread through
`decode_inter_block_mode_info` (and the §5.11.18 dispatcher):
`is_motion_mode_switchable` (§5.9.2 bit), `allow_warped_motion`
(§5.9.2 bit), `is_scaled_per_ref: [bool; 7]` (caller-precomputed per-
reference `is_scaled` lookup; `is_scaled_per_ref[refFrame - LAST_FRAME]`
mirrors the §5.11.27 `is_scaled(RefFrame[0])` predicate without
threading the frame-size / RefInfo arrays into the walker).

`DecodedInterBlockModeInfo` gains two new fields: `motion_mode: u8`
(carrying one of the three `MOTION_MODE_*` ordinals) and
`num_warp_samples: u32` (the §7.10.4 `NumSamples` snapshot at the
§5.11.27 site, in `0..=LEAST_SQUARES_SAMPLES_MAX = 0..=8`). Observable
through the §5.11.23 dispatcher's `Ok(_)` arm — now reached on every
conformant inter block — even though the §5.11.18 dispatcher still
surfaces `InterBlockModeInfoUnsupported` for the pending
post-`read_motion_mode` cascade.

19 new unit tests cover: `DEFAULT_USE_OBMC_CDF` row-shape sanity
(BLOCK_SIZES rows × 3 cumulative frequencies + counter); the
`use_obmc_cdf(mi_size)` selector against the §9.4 source for every
in-range index plus None for out-of-range; the §6.10.26 ordinal
identities; `has_overlappable_candidates` returning false on a fresh
walker and true after stamping an above (resp. left) inter
neighbour; `find_warp_samples` returning 0 on a fresh walker, ≥1 on a
same-ref + matching-MV above neighbour, and 0 on a mismatched-ref
neighbour; the five §5.11.27 SIMPLE short-circuit arms (skip_mode,
!is_motion_mode_switchable, small block, GLOBALMV + non-TRANSLATION
gm_type, compound, no overlappable candidates) each verified to read
zero S() bits; the `use_obmc` arm consuming the §8.3 `update_cdf` of
the use_obmc row while leaving the motion_mode row untouched; the
`motion_mode` arm consuming the `motion_mode` row while leaving
use_obmc untouched; the `is_scaled = true` routing forcing the
`use_obmc` arm; and an end-to-end §5.11.23 dispatcher exercise
showing `skip_mode = 1` surfacing
`Ok(DecodedInterBlockModeInfo { motion_mode: SIMPLE, .. })`. Test
count: 672 → 691 (+19). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 178 lands the **§5.11.x `read_interpolation_filter` reader** —
the LAST leaf of the §5.11.23 inter cascade. The §5.11.23 dispatcher
now runs the entire `inter_block_mode_info()` body to completion;
the §5.11.18 dispatcher continues to surface
`InterBlockModeInfoUnsupported` from its `Ok(_)` arm pending the
next-arc §5.11.34 `residual()` lift and the follow-on refactor that
lifts the `DecodedInterBlockModeInfo` aggregate into the
`DecodedInterFrameModeInfo` return path.

The §5.11.x body composes two paths: the `interpolation_filter ==
SWITCHABLE` arm runs the per-direction `for dir = 0..(enable_dual_filter
? 2 : 1)` loop with the inner `if ( needs_interp_filter( ) )
interp_filter[ dir ] S() against TileInterpFilterCdf[ ctx ] else
interp_filter[ dir ] = EIGHTTAP` branch, and on the
`!enable_dual_filter` post-loop mirror sets `interp_filter[ 1 ] =
interp_filter[ 0 ]`. The `else` arm forces both slots to the
frame-header value (`interpolation_filter`) with no per-block bits.

`needs_interp_filter( )` (av1-spec p.75) is also lifted: it closes
the gate on `skip_mode || motion_mode == LOCALWARP`, plus the two
large-block GLOBALMV / GLOBAL_GLOBALMV paths consult the
caller-supplied `GmType[ RefFrame[ . ] ] == TRANSLATION` predicate;
the default arm returns `1`.

The §8.3.2 `interp_filter` ctx walk consumes the §8.3.2 paragraph
verbatim through `interp_filter_ctx` (already in place since round
22): a folded `((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4` ctx
seed, then the two neighbour-mismatch / match-and-equal / one-mismatch
arms. The match test consults the neighbour cell's
`RefFrames[..][..][0..2]` against the current block's `RefFrame[0]`;
non-matching neighbours collapse to the `aboveType = leftType = 3`
sentinel (`INTERP_FILTER_NONE`).

The walker grows an `InterpFilters[r][c][dir]` grid (two slots per
cell, pre-fill `EIGHTTAP = 0`), stamped over the bh4 * bw4 footprint
on every inter block decode so the next block's ctx walk observes
the values. The `interp_filters()` accessor surfaces it.

New §6.8.9 named ordinals land: `EIGHTTAP = 0`, `EIGHTTAP_SMOOTH = 1`,
`EIGHTTAP_SHARP = 2`, `BILINEAR = 3`, `SWITCHABLE = 4`. New
`InterpolationFilterReadout` aggregate carries `interp_filter: [u8; 2]`
and `read_from_bitstream: [bool; 2]` (the per-dir "did this slot
fire an S()?" flag). `DecodedInterBlockModeInfo` gains an
`interp_filter: InterpolationFilterReadout` field.
`decode_inter_block_mode_info` and `decode_inter_frame_mode_info` gain
two new caller-supplied scalars: `interpolation_filter: u8` (§5.9.10
frame-header value) and `enable_dual_filter: bool` (§5.5.2
sequence-header bit).

10 new unit tests cover: the §6.8.9 ordinal alignment (and the
`INTERP_FILTERS` boundary against `BILINEAR`); the non-switchable
arm (both slots forced to the frame-header value, no bits read);
the four `needs_interp_filter == 0` short-circuits (`skip_mode`,
`motion_mode == LOCALWARP`, large GLOBALMV + non-TRANSLATION
gm_type) all collapsing to EIGHTTAP with no bits; the
`SWITCHABLE && needs_interp_filter` single-dir path (one S() ⇒ both
slots get the symbol, `read_from_bitstream = [true, false]`); the
dual-dir path (two S()s ⇒ `read_from_bitstream = [true, true]`); the
three caller-bug guards (out-of-range `sub_size` /
`interpolation_filter` / `ref_frame[0]` all surface
`PartitionWalkOutOfRange`); and an end-to-end §5.11.23 cascade test
through the dispatcher witnessing the `InterpolationFilterReadout`
on the SWITCHABLE + skip_mode (needs_interp_filter == 0) path with
the walker's `interp_filters` grid stamped EIGHTTAP over the
BLOCK_16X16 footprint. Test count: 721 → 731 (+10). `decode_av1` /
`encode_av1` still return `Error::NotImplemented`.

Round 179 lands the **§5.11.39 `coeffs( plane, startX, startY, txSz )`
per-TU coefficient reader** — the first body of the §5.11.34
`residual()` cascade, the gate to the entire transform / inverse-
transform / reconstruction pipeline. Exposed as
`PartitionWalker::coefficients(decoder, cdfs, plane, is_inter,
tx_size, tx_class, txb_skip_ctx, dc_sign_ctx, scan, quant)` and
surfaced through the new `CoefficientsReadout { all_zero, eob,
cul_level, dc_category }` aggregate. This reader is callable directly
today; the §5.11.34 outer dispatch (the `widthChunks` / `heightChunks`
/ `transform_tree` / `transform_block` recursion) and the §5.11.30
`compute_prediction()` walker-site that gates it remain
subsequent-arc targets — the §5.11.5 walker
([`PartitionWalker::decode_block_syntax`]) still short-circuits at
`Error::DecodeBlockComputePredictionUnsupported` upstream of the new
reader.

The §5.11.39 body composes seven pieces, all wired:

* **Line 3-7 derived sizes.** `txSzCtx` from `Tx_Size_Sqr` (derived
  inline from `Min(Tx_Width, Tx_Height)`'s `trailing_zeros`) +
  `Tx_Size_Sqr_Up` (the existing `TX_SIZE_SQR_UP` table); `ptype =
  plane > 0`; `segEob` honouring the `TX_16X64 || TX_64X16 ⇒ 512`
  special case and the `Min(1024, tx_w * tx_h)` cap.
* **Line 8-9 Quant pre-loop.** Zero-fills `Quant[ 0..segEob ]` in the
  caller's output buffer.
* **Line 13 `all_zero` S().** Against `TileTxbSkipCdf[ txSzCtx ][
  ctx ]` (caller-supplied ctx). On the `all_zero == 1` short-circuit
  returns `CoefficientsReadout { all_zero: true, eob: 0, cul_level:
  0, dc_category: 0 }`. The §5.11.40 luma `TxTypes[ y4 + j ][ x4 + i
  ] = DCT_DCT` stamp on the short-circuit arm is deferred to the
  §5.11.34 caller (the walker's `TxTypes` grid is not yet tracked).
* **Lines 19-37 EOB position read.** `eobMultisize = Min(
  Tx_Width_Log2, 5 ) + Min( Tx_Height_Log2, 5 ) - 4` selects one of
  the seven `eob_pt_{16, 32, 64, 128, 256, 512, 1024}` S() reads
  against `TileEobPt*Cdf[ ptype ][ ctx ]` (with `ctx = 0` for
  `TX_CLASS_2D`, `ctx = 1` for HORIZ / VERT; the 512 / 1024 tables
  have no ctx axis per §8.3.2). `eobPt = sym + 1`.
* **Lines 39-55 `eob` derivation.** `eob = (eobPt < 2) ? eobPt : ((1
  << (eobPt - 2)) + 1)`; `eobShift = Max(-1, eobPt - 3)`. When
  `eobShift >= 0` reads `eob_extra` S() against `TileEobExtraCdf[
  txSzCtx ][ ptype ][ eobPt - 3 ]` plus the `eob_extra_bit` L(1)
  loop adding `1 << eobShift` per set bit.
* **Lines 56-71 reverse-scan base levels.** Walks `c = eob-1 → 0`:
  at `c == eob-1` reads `coeff_base_eob` S() against
  `TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]` with ctx from the
  existing `get_coeff_base_eob_ctx(quant, tx_size, tx_class, pos,
  c)` helper; otherwise reads `coeff_base` S() against
  `TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]` with ctx from the
  existing `get_coeff_base_ctx(quant, tx_size, tx_class, pos, c,
  false)` helper. `level > NUM_BASE_LEVELS = 2` triggers the
  `coeff_br` chain — up to `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) =
  4` S() reads against `TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][
  ptype ][ ctx ]` with ctx from `get_br_ctx`; chain terminates on the
  first symbol `< BR_CDF_SIZE - 1 = 3`. Writes `Quant[ pos ] =
  level` (positive magnitude only at this stage).
* **Lines 73-100 forward-scan signs + golomb.** Walks `c = 0..eob`:
  reads `dc_sign` S() at `c == 0` (against `TileDcSignCdf[ ptype ][
  ctx ]` with caller-supplied ctx), otherwise `sign_bit` L(1).
  `Quant[ pos ] > NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14` triggers
  the §5.11.39 golomb chain — `golomb_length_bit` L(1) do-while loop
  for the magnitude exponent, then `golomb_data_bit` L(1) loop for
  the data bits; `Quant[ pos ] = x + COEFF_BASE_RANGE +
  NUM_BASE_LEVELS`. Updates `dc_category` on the `pos == 0 &&
  Quant[ pos ] > 0` arm (`1` for negative DC, `2` for positive),
  applies the `& 0xFFFFF` 20-bit clip and the `culLevel +=
  Quant[ pos ]` accumulation, then sign-applies `Quant[ pos ] =
  -Quant[ pos ]` when needed. Closes with the `culLevel = Min(63,
  culLevel)` clamp.

Caller-supplied state surfaced through the function signature:
`tx_size` (TX_SIZES_ALL ordinal), `tx_class` (the §8.3.2
`get_tx_class(txType)` result, one of TX_CLASS_2D / TX_CLASS_HORIZ /
TX_CLASS_VERT — the §5.11.40 `compute_tx_type` derivation belongs to
the caller), `plane` / `is_inter`, `txb_skip_ctx` (`all_zero` ctx;
the §8.3.2 derivation needs `AboveLevelContext` / `LeftLevelContext`
/ `AboveDcContext` / `LeftDcContext` neighbour state that the
walker does not yet track), `dc_sign_ctx` (same neighbour-state
dependency), `scan` (caller-derived from `txSz` + `txType` per §7.5
`get_scan` — also deferred), and a `quant: &mut [i32]` output buffer
sized `>= tx_w * tx_h`. Caller-bug guards return
`PartitionWalkOutOfRange` for each out-of-range argument
(`tx_size >= TX_SIZES_ALL`, `tx_class > TX_CLASS_VERT`, `plane > 2`,
`is_inter > 1`, `txb_skip_ctx >= TXB_SKIP_CONTEXTS`,
`dc_sign_ctx >= DC_SIGN_CONTEXTS`, undersized `scan` / `quant`).

The §5.11.39 `Dequant[][]` step (true §7.13 dequantization through
the per-plane qmatrix) is **out of scope for this round**; the
reader produces the §5.11.39 raw signed `Quant[]` array — the same
quantity that feeds the §7.13 dequant — and leaves the
dequantization itself for a subsequent arc. The §5.11.34
`AboveLevelContext` / `LeftLevelContext` / `AboveDcContext` /
`LeftDcContext` stamping at the post-§5.11.39 line 104-115 site is
also deferred to the round that adds those context arrays to the
walker (the readout already carries `cul_level` and `dc_category`
for the caller to apply).

6 new unit tests cover: the gate-closed `all_zero == 1`
short-circuit (no symbol / literal bits past `txb_skip`, returned
readout zeroed, output buffer zero-filled); a gate-open smoke test
confirming the cascade produces a non-`all_zero` readout with
`eob ∈ 1..=segEob`, `cul_level ≤ 63`, `dc_category ∈ {0, 1, 2}`,
and decoder position advanced past at least one symbol bit; a CDF
adaptation cross-check confirming the §8.3 `txb_skip` counter slot
increments on a non-`disable_cdf_update` run (proves the reader is
indexing the right CDF row); the seven caller-bug guards all
returning `PartitionWalkOutOfRange`; a `TX_16X64` `segEob = 512`
boundary check (511-element scan rejected, 512-element accepted);
and a square-tx-size sweep across all five `TX_NxN` ordinals
confirming the internal `txSzCtx = (Tx_Size_Sqr + Tx_Size_Sqr_Up +
1) >> 1` derivation stays within `0..TX_SIZES` for every square
size. Test count: 731 → 737 (+6). `decode_av1` / `encode_av1` still
return `Error::NotImplemented`.

Round 177 lands the **§5.11.29 `read_compound_type` reader** —
wiring the inter-inter compound-blending controls into the §5.11.23
inter cascade. The §5.11.23 dispatcher now runs the full §5.11.23
body through line 36 (`read_compound_type( isCompound )`); the
§5.11.18 dispatcher continues to surface `InterBlockModeInfoUnsupported`
for the still-pending `read_interpolation_filter` post-cascade.

The §5.11.29 body composes five paths: (1) the `skip_mode == 1`
short-circuit (`compound_type = COMPOUND_AVERAGE`, defaults
`comp_group_idx = 0` / `compound_idx = 1`, no bits); (2) the
`!isCompound` else-arm (the §5.11.28 outcome alone selects the type
— `interintra && wedge_interintra ? COMPOUND_WEDGE : COMPOUND_INTRA`
or `COMPOUND_AVERAGE`, no bits); (3) the `isCompound &&
enable_masked_compound` arm (`comp_group_idx` S() against
`TileCompGroupIdxCdf[ ctx ]` with ctx from the §8.3.2 paragraph
combining `CompGroupIdxs` neighbour values, `AvailU` / `AvailL`,
`AboveSingle` / `LeftSingle`, and the `AboveRefFrame[0] / LeftRefFrame[0]
== ALTREF_FRAME` tests, clamped at `Min(5, ctx)`); (4) the
`comp_group_idx == 0 && enable_jnt_comp` sub-arm (`compound_idx` S()
against `TileCompoundIdxCdf[ ctx ]` with ctx seeded by `dist_equal
? 3 : 0` and incremented by the same neighbour walk, then
`compound_type = compound_idx ? COMPOUND_AVERAGE : COMPOUND_DISTANCE`);
(5) the `comp_group_idx == 1` sub-arm (`n = Wedge_Bits[ MiSize ];
n == 0 ? COMPOUND_DIFFWTD : compound_type S() against
TileCompoundTypeCdf[ MiSize ]`). The wedge sub-branch reads
`wedge_index` S() against `TileWedgeIndexCdf[ MiSize ]` (the
16-symbol element shared with the §5.11.28 inter-intra wedge arm)
plus `wedge_sign` L(1); the diffwtd sub-branch reads `mask_type`
L(1) selecting `UNIFORM_45` (0) or `UNIFORM_45_INV` (1) per §6.10.24.

The walker grows two grids — `CompGroupIdxs[r][c]` (pre-fill `0`)
and `CompoundIdxs[r][c]` (pre-fill `1`, matching the §5.11.29
default initialiser) — stamped over the bh4 * bw4 footprint on
every block decode so the next block's §8.3.2 ctx walks observe the
values. `comp_group_idxs()` / `compound_idxs()` accessors surface
both grids for tests / external consumers.

New §3 `WEDGE_BITS` table (`[u8; BLOCK_SIZES]`, av1-spec p.470
verbatim — non-zero only on the `BLOCK_8X8..=BLOCK_32X32` band plus
the `BLOCK_8X16` / `BLOCK_16X8` 2:1 rectangles, matching the §9.4
`Default_Wedge_Index_Cdf` non-placeholder rows) + `wedge_bits()`
accessor. New §6.10.24 named constants land: `COMPOUND_WEDGE = 0`,
`COMPOUND_DIFFWTD = 1`, `COMPOUND_AVERAGE = 2`, `COMPOUND_INTRA = 3`,
`COMPOUND_DISTANCE = 4` (the spec note clarifies the last three are
inferred from other syntax elements rather than coded directly via
the §8.3.2 `compound_type` S()); plus the two `mask_type` ordinals
`UNIFORM_45` / `UNIFORM_45_INV`. New §8.3.2 ctx helpers
`comp_group_idx_ctx` / `compound_idx_ctx` accept the neighbour scalar
inputs (closed forms covered by exhaustive small-input audits).

New `CompoundTypeReadout` aggregate carries `comp_group_idx` (u8) +
`compound_idx` (u8) + `compound_type` (u8) plus three `Option<u8>`
companions (`wedge_index` / `wedge_sign` Some on `COMPOUND_WEDGE`;
`mask_type` Some on `COMPOUND_DIFFWTD`). `DecodedInterBlockModeInfo`
gains a `compound_type: CompoundTypeReadout` field.
`decode_inter_block_mode_info` and `decode_inter_frame_mode_info`
gain three new caller-supplied scalars: `enable_masked_compound: bool`
(§5.5.2 sequence-header bit), `enable_jnt_comp: bool` (§5.5.2),
and `dist_equal: bool` (the `Abs(get_relative_dist(OrderHints[
RefFrame[0]], OrderHint)) == Abs(get_relative_dist(OrderHints[
RefFrame[1]], OrderHint))` outcome the caller computes from
frame-header state).

18 new unit tests cover: the `WEDGE_BITS` table verbatim cross-check
+ `wedge_bits()` helper out-of-range fallback; the §6.10.24
ordinal identities (compound-type + mask-type); the §8.3.2
`comp_group_idx_ctx` formula across every closed-form arm (closed-
both, above-only !Single, above-only Single+ALTREF, both-singles
ALTREF clamp at 5, mixed arms); the §8.3.2 `compound_idx_ctx`
formula across the dist_equal seed + every neighbour arm + a
2^7 = 128-trial exhaustive bound-check; six per-arm reader tests
exercising the `skip_mode` short-circuit (no bits), the three
`!isCompound` else-arms (`interintra=0`, `interintra=1 +
wedge_interintra=0`, `interintra=1 + wedge_interintra=1`, no bits
each), the `isCompound && !enable_masked_compound && !enable_jnt_comp`
no-bits arm, the `enable_masked_compound + comp_group_idx S()⇒0 +
!enable_jnt_comp` one-symbol arm (witnessed by CDF row adaptation),
the wedge-sub-branch path (forced `comp_group_idx⇒1` +
`compound_type⇒COMPOUND_WEDGE`; checks `wedge_index` + `wedge_sign`
both Some + the wedge_index row CDF adaptation), the
diffwtd-sub-branch path (forced `comp_group_idx⇒1` +
`compound_type⇒COMPOUND_DIFFWTD`; checks `mask_type` Some + the
wedge_index row stayed untouched), the `n == 0 ⇒ COMPOUND_DIFFWTD`
shortcut on `BLOCK_4X4` (no `compound_type` S() read; CDF row
unchanged), the `compound_idx S()⇒1 ⇒ COMPOUND_AVERAGE` and
`compound_idx S()⇒0 ⇒ COMPOUND_DISTANCE` arms, the `dist_equal=true`
ctx-3 seed witness, the `mi_size = BLOCK_SIZES` caller-bug defensive
fallback (collapses to the `wedge_bits == 0` shortcut without
hitting `wedge_index_cdf`); and an end-to-end §5.11.23 cascade test
through the dispatcher witnessing the `CompoundTypeReadout` on the
single-pred + closed-§5.11.28-gate path with the walker
`comp_group_idxs` / `compound_idxs` grids stamped. Test count: 703
→ 721 (+18). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 176 lands the **§5.11.28 `read_interintra_mode` reader** —
wiring the inter-intra blending triple into the §5.11.23 inter
cascade. The §5.11.23 dispatcher now runs the full §5.11.23 body
through line 35 (`read_interintra_mode( isCompound )`); the §5.11.18
dispatcher continues to surface `InterBlockModeInfoUnsupported` for
the still-pending §5.11.29 `read_compound_type` /
`read_interpolation_filter` post-cascade.

The §5.11.28 body composes the outer gate (`skip_mode == 0 &&
enable_interintra_compound && !isCompound && BLOCK_8X8 <= MiSize <=
BLOCK_32X32`) and the inner four-symbol arm: `interintra` S() against
`TileInterIntraCdf[ Size_Group[ MiSize ] - 1 ]`; on
`interintra == 1` then `interintra_mode` S() against
`TileInterIntraModeCdf[ ctx ]` (one of the four §6.10.27 `II_*`
modes — `II_DC_PRED` / `II_V_PRED` / `II_H_PRED` / `II_SMOOTH_PRED`),
`wedge_interintra` S() against `TileWedgeInterIntraCdf[ MiSize ]`,
and on `wedge_interintra == 1` then `wedge_index` S() against
`TileWedgeIndexCdf[ MiSize ]` (the 16-symbol element shared with
§5.11.29). On the closed-gate path zero bits are consumed and the
returned `InterIntraReadout` has `interintra = 0` with the per-symbol
`Option` fields at `None`.

Inner-arm side-effect: the spec sets `RefFrame[ 1 ] = INTRA_FRAME` on
the inner arm; the §5.11.23 dispatcher restamps the walker's slot-1
`RefFrames[..][..][1]` grid over the bh4 * bw4 block footprint so
subsequent neighbour walks observe the override. The §5.11.27 readers
covered in r175 consult slot 0 only, so the slot-1 override does not
perturb r175 behaviour; the §5.11.29 reader (next-arc) will consult
slot 1 explicitly via the §8.3.2 `comp_group_idx` ctx. The
imperative `AngleDeltaY = AngleDeltaUV = use_filter_intra = 0` writes
the spec spells out are inter-block scalars not currently tracked.

Four new §6.10.27 named constants land: `II_DC_PRED = 0`,
`II_V_PRED = 1`, `II_H_PRED = 2`, `II_SMOOTH_PRED = 3` — spanning the
`INTERINTRA_MODES = 4` axis. New `InterIntraReadout` aggregate carries
`interintra: u8` plus three `Option<u8>` companions; the inner-arm
arms populate `interintra_mode` + `wedge_interintra` always and
`wedge_index` only when `wedge_interintra == 1`.
`DecodedInterBlockModeInfo` gains an `interintra: InterIntraReadout`
field. `decode_inter_block_mode_info` and
`decode_inter_frame_mode_info` gain a new
`enable_interintra_compound: bool` parameter — the §5.5.2 sequence-
header bit caller-supplied.

10 new unit tests cover: the four §6.10.27 II_* ordinal identities;
five outer-gate-closed paths (`skip_mode = 1`, !enable_interintra_-
compound, `is_compound = true`, `MiSize < BLOCK_8X8` via `BLOCK_4X4`,
`MiSize > BLOCK_32X32` via `BLOCK_64X64`) each verified to read zero
S() bits; the `MiSize = BLOCK_SIZES` defensive fallback collapsing to
the closed-gate path; the gate-open + `interintra = 0` arm
(witnessed by `inter_intra` row CDF adaptation alongside an untouched
`inter_intra_mode` row); the four-symbol path reachability
(100-trial property test biased toward the inner arm, witnessing the
`wedge_index` row CDF adaptation); and two end-to-end §5.11.23
cascade tests — `enable_interintra_compound = false` short-circuiting
through the dispatcher with `interintra = 0` + every Option `None`,
and `enable_interintra_compound = true` witnessing the
`inter_intra` row adaptation through the dispatcher with the
conditional slot-1 grid stamp asserted on the inner-arm trials. Test
count: 691 → 703 (+12). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 173 lands the **§5.11.23 post-find_mv_stack reader cascade** —
wiring `find_mv_stack` into the `decode_inter_block_mode_info`
dispatcher so the inter arm of §5.11.18 runs end-to-end through every
bit-consuming leaf in §5.11.23 lines 1-32. The §5.11.23
`Error::FindMvStackUnsupported` stub is lifted; the reader now
short-circuits one step later at the new
`Error::AssignMvUnsupported` (§5.11.31 `assign_mv` is the next-arc
target — its body composes `read_mv` / `read_mv_component` which is
the large §5.11.32 syntax tree).

The cascade implements the four-arm YMode dispatch in spec order
(av1-spec p.74). Arm 1 (`skip_mode == 1`) forces
`YMode = NEAREST_NEARESTMV` with no S() read. Arm 2
(`seg_feature_active(SEG_LVL_SKIP|GLOBALMV)`) forces `YMode =
GLOBALMV` with no S() read. Arm 3 (compound, `RefFrame[1] >
INTRA_FRAME`) reads one `S()` against `TileCompoundModeCdf[ctx]`
where `ctx = Compound_Mode_Ctx_Map[RefMvContext >> 1][Min(NewMvContext,
COMP_NEWMV_CTXS - 1)]` per §8.3.2, then `YMode = NEAREST_NEARESTMV +
compound_mode`. Arm 4 (single-pred fall-through) walks the
`new_mv` ⇒ `zero_mv` ⇒ `ref_mv` ladder: each is one S() against the
matching `TileNewMvCdf[NewMvContext]` / `TileZeroMvCdf[ZeroMvContext]` /
`TileRefMvCdf[RefMvContext]` row from §7.10.2.14, terminating in one
of NEWMV / GLOBALMV / NEARESTMV / NEARMV.

The §5.11.23 RefMvIdx + drl_mode loops then run (av1-spec p.74-75).
On `YMode ∈ {NEWMV, NEW_NEWMV}` the loop iterates `idx = 0, 1`; on
`has_nearmv(YMode)` true (= one of NEARMV / NEAR_NEARMV / NEAR_NEWMV /
NEW_NEARMV) the loop seeds `RefMvIdx = 1` and iterates `idx = 1, 2`.
Each iteration is gated on `NumMvFound > idx + 1`; when fired it
reads one `S()` against `TileDrlModeCdf[DrlCtxStack[idx]]`, breaking
to `RefMvIdx = idx` on symbol 0 or continuing with `RefMvIdx = idx +
1` on symbol 1. On fresh-walker / `NumMvFound = 0` paths the loop
runs zero iterations and `RefMvIdx = 0`.

The §7.10 `find_mv_stack` call is wired in directly via the r172
spatial-only entry. Six new caller scalars are threaded through:
`gm_type[8]` / `gm_params[8][6]` (§7.10.2.1 setup-global-mv),
`ref_frame_sign_bias[8]` (§7.10.2.13 extra-MV negation arm),
`allow_high_precision_mv` / `force_integer_mv` (§7.10.2.10 lower-
precision LSB strip vs 3-bit integer round), and `use_ref_frame_mvs`
(the §7.10.2.5 temporal-scan deferral — when true, the reader
surfaces `Error::TemporalMvScanUnsupported` without partial state
mutation). The §5.11.18 `decode_inter_frame_mode_info` dispatcher
threads the same six scalars to keep the §5.11.18 → §5.11.23 call
chain end-to-end.

`DecodedInterBlockModeInfo` gains six new fields surfaced for caller
verification: `y_mode` (one of `MODE_NEARESTMV..=MODE_NEW_NEWMV =
14..=25`); `ref_mv_idx` (in `0..=2`); and the §7.10.2 stack-summary
snapshot — `num_mv_found`, `new_mv_context`, `ref_mv_context`,
`zero_mv_context`. These are observable only when the dispatcher
returns `Ok`; the r173 method always returns `Err`
(`AssignMvUnsupported`) since the §5.11.31 body is not yet wired —
direct callers can inspect the stamped walker grids
(`ref_frames()` / `is_inters()`) for the conformant prologue state
that committed before the stub fired.

New `has_nearmv(mode)` predicate (av1-spec p.75) returns true for
NEARMV / NEAR_NEARMV / NEAR_NEWMV / NEW_NEARMV (the four NEARMV-
bearing inter Y modes). Joins the existing `has_newmv` predicate
that already covers the six NEWMV-bearing modes.

10 new unit tests cover: Arm 1 `skip_mode = 1` yielding NEAREST_NEARESTMV
with zero bits consumed past §5.11.25; Arm 2 `seg_skip_active`
yielding GLOBALMV with zero bits consumed; Arm 3 compound case with
rigged `comp_mode` / `comp_ref_type` / `uni_comp_ref` cascade
producing `RefFrame = [BWDREF, ALTREF]` and the `compound_mode` S()
consuming bits; Arm 4 single-pred three terminal outcomes
(`new_mv = 0` ⇒ NEWMV; `new_mv = 1, zero_mv = 0` ⇒ GLOBALMV;
`new_mv = 1, zero_mv = 1, ref_mv = 0` ⇒ NEARESTMV); drl_mode loop
short-circuit on `NumMvFound = 0` (CDF counter stays 0 ⇒ no read
fired); `has_nearmv` truth table over the entire inter Y-mode
enumeration; caller-bug guards unchanged with the new r173 args;
and the `use_ref_frame_mvs == true` arm surfacing
`TemporalMvScanUnsupported` ahead of the cascade. Test count: 651 →
661 (+10). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 172 lands the **§7.10 `find_mv_stack` spatial-only path** —
the motion-vector-stack derivation reached from the §5.11.23
`inter_block_mode_info()` body once the §5.11.25 `read_ref_frames`
prologue commits `RefFrame[0..2]`. The new
`PartitionWalker::find_mv_stack(...)` method composes every
sub-process from §7.10.2.1 setup-global-mv through §7.10.2.14
context + clamping, on the `use_ref_frame_mvs == false` path.

The §7.10.2 driver runs steps 1-16 + 18-36 in spec order:
`NumMvFound = 0`, `NewMvCount = 0`, setup-global-mv for slot 0 (and
slot 1 if compound), then the inner-ring scans
(`scan_row(-1)` / `scan_col(-1)` / `scan_point(-1, bw4)` when
`Max(bw4, bh4) ≤ 16`), the §7.10.2 step 15 `WeightStack[0..numNearest]
+= REF_CAT_LEVEL = 640` close-match bonus, the outer-ring scans
(`scan_point(-1, -1)` / `scan_row(-3)` / `scan_col(-3)` /
`scan_row(-5)` if `bh4 > 1` / `scan_col(-5)` if `bw4 > 1`), the
§7.10.2.11 stable descending-by-weight sort over both segments
(`[0..numNearest)` and `[numNearest..NumMvFound)`), the §7.10.2.12
extra-search when `NumMvFound < 2` (two-pass partial-match scan over
the row above + column to the left, then a `combinedMvs`
fill from `RefIdMvs` + `RefDiffMvs` + `GlobalMvs` for compound, or
direct `RefStackMv` extension for single — though single-pred does
NOT increment `NumMvFound` per the §7.10.2.12 note), and the
§7.10.2.14 context + clamping
(`DrlCtxStack[idx]` derivation off the `WeightStack[idx] >=
REF_CAT_LEVEL` / `WeightStack[idx+1] < REF_CAT_LEVEL` partition,
per-list `clamp_mv_row` / `clamp_mv_col` with
`border = MV_BORDER + (bh * 8) | (bw * 8)`, and the
`NewMvContext` / `RefMvContext` three-arm ladder over `CloseMatches`).

The §7.10.2.2 scan_row applies the `Abs(deltaRow) > 1` neighbour
adjustment (`deltaRow += MiRow & 1; deltaCol = 1 - (MiCol & 1)`), the
`useStep16 = bw4 >= 16` 4-step stride, and the per-neighbour
weighted advance (`len = Min(bw4, Num_4x4_Blocks_Wide[MiSizes[mvRow]
[mvCol]])`, then `Max(2, len)` outer-ring + `Max(4, len)` step16
refinements). §7.10.2.3 scan_col is the vertical mirror. §7.10.2.4
scan_point gates on `RefFrames[mvRow][mvCol][0] != INTRA_FRAME = 0`
(matching the spec's "has been written" check; the walker's pre-fill
`[INTRA_FRAME, NONE]` makes the gate the spec-correct identity).

The §7.10.2.7 add_ref_mv_candidate gate
(`IsInters[mvRow][mvCol] == 1`) then dispatches to §7.10.2.8
search-stack (single-pred, for each `candList` where
`RefFrames[mvRow][mvCol][candList] == RefFrame[0]`) or §7.10.2.9
compound-search-stack (compound, when both
`RefFrames[mvRow][mvCol][0..2]` match `RefFrame[0..2]`). The
§7.10.2.10 lower-precision pass (LSB-strip when
`allow_high_precision_mv == 0`, `(|a| + 3) >> 3 << 3` integer round
when `force_integer_mv == 1`) is applied to every candidate MV
before dedupe. The §7.10.2.8 `(candMode in {GLOBALMV,
GLOBAL_GLOBALMV}) && GmType[RefFrame[0]] > TRANSLATION && large == 1`
global-MV substitution arm is a no-op refinement on the conformant
decode path — §5.11.31 `assign_mv` already stamps
`Mvs[..][refList] = GlobalMvs[refList]` for those modes, so the
"use Mvs[..]" fall-through produces the same bytes.

The §7.10.2.5 temporal scan + §7.10.2.6 temporal sample sub-
processes (step 17 of the §7.10.2 driver) are deferred — when the
caller passes `use_ref_frame_mvs == true`, the new method returns
`Error::TemporalMvScanUnsupported` without partial state mutation.
The temporal scan needs a `MotionFieldMvs[ref][y8][x8]` grid that
§7.9 `motion_field_estimation` is responsible for populating from
the §5.9.20 `RefFrameSignBias` chain — that scaffolding lands with
the next arc.

The walker gains a new `Mvs[r][c][list][comp]` grid stored as a flat
`Vec<i16>` with four `i16` slots per `(row, col)` cell (two lists ×
two components). Pre-fill is zero; the §7.10.2.7
`IsInters[mvRow][mvCol] == 0 ⇒ return` gate ensures the pre-fill
is unobservable on the conformant path until §5.11.31 `assign_mv`
(next-arc) writes it. New `mvs()` accessor exposes the grid.

New aggregate `FindMvStackResult` surfaces every §7.10.2 output:
`num_mv_found`, `new_mv_count`, `ref_stack_mv[idx][list][comp]`,
`weight_stack[idx]`, `global_mvs[list]`, `new_mv_context`,
`ref_mv_context`, `zero_mv_context`, `drl_ctx_stack[idx]` plus
the §7.10.2 step 12-14 snapshot values `close_matches`,
`total_matches`, `num_nearest`, `num_new`.

20 new §3 / §6.10.22 / §7.10 constants land:
`MAX_REF_MV_STACK_SIZE = 8`, `REF_CAT_LEVEL = 640`,
`MV_BORDER = 128`, `WARPEDMODEL_PREC_BITS = 16` (cdf-local twin);
GM type discriminants (`IDENTITY`/`TRANSLATION`/`ROTZOOM`/`AFFINE`
= 0..=3); the §6.10.22 inter Y mode ordinal table
(`MODE_NEARESTMV = 14`..`MODE_NEW_NEWMV = 25`) plus the
`has_newmv(mode)` predicate.

The §5.11.23 `decode_inter_block_mode_info` reader continues to
return `Error::FindMvStackUnsupported` because the post-MV-stack
reader cascade (`compound_mode` / `new_mv` / `zero_mv` / `ref_mv` /
`drl_mode` / `assign_mv` / `read_motion_mode` /
`read_interintra_mode` / `read_compound_type` /
`read_interpolation_filter`) remains pending. The wiring of
`find_mv_stack` into the dispatcher follows once those readers land.

24 new unit tests cover: §7.10.2.10 lower-precision matrix
(high-precision passthrough, LSB strip, `force_integer_mv` 3-bit
round); §5.11.53 / §5.11.54 clamp bracket derivations; §7.10.2.11
stable sort (descending by weight + equal-weight stability);
§7.10.2.1 setup-global-mv (INTRA_FRAME ⇒ 0, IDENTITY ⇒ 0,
TRANSLATION shift); empty-walker single-pred (NumMvFound = 0) and
compound (NumMvFound = 2 global-fill) outcomes; caller-bug guards;
temporal-scan deferral; one above-neighbour with NEWMV
(REF_CAT_LEVEL bonus + NewMvContext = 2, RefMvContext = 3); one
left-neighbour with NEARESTMV (numNew = 0, NewMvContext = 3);
both-neighbours case (CloseMatches = 2 ⇒ NewMvContext = 4,
RefMvContext = 5); mismatched-ref close-match exclusion +
extra-search pickup; identical-MV dedupe; single-pred extra-search
NumMvFound non-increment; `has_newmv` truth table; MV-stack
constant values; DrlCtxStack[] derivation. Test count: 627 → 651
(+24). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 171 lands the **§5.11.46 palette-entries reader**
(`palette_colors_y[]` / `palette_colors_u[]` /
`palette_colors_v[]`) and the **§5.11.49 `get_palette_cache(plane)`**
two-pointer merge of the above + left neighbour palettes —
lifting the `Error::PaletteEntriesUnsupported` stub the r169
§5.11.22 reader surfaced after `palette_size_*_minus_2`. The
§5.11.22 `PartitionWalker::decode_intra_block_mode_info` reader
now threads a `bit_depth: u8` argument (in {8, 10, 12} per
§5.5.2) and runs every leaf to completion on the conformant path.
The Y plane walks the cache-coded indices loop
(`use_palette_color_cache_y L(1)` per cache slot, copying
`PaletteCache[i]`), the first new entry as `L(BitDepth)`, the
optional `L(2) palette_num_extra_bits_y`, and the delta loop with
`palette_delta_y L(paletteBits)` + the spec's `++` increment +
`Clip1(prev + delta)` + `range = (1 << BitDepth) -
palette_colors_y[idx] - 1` paletteBits refinement, terminating in
the trailing ascending sort. The U plane mirrors Y minus the `++`
step and with `range = (1 << BitDepth) - palette_colors_u[idx]`
(no `- 1`). The V plane dispatches on
`delta_encode_palette_colors_v L(1)`: on 1, a signed-delta path
with `minBits = BitDepth - 4`, per-entry `L(paletteBits)
palette_delta_v` + optional `L(1) palette_delta_sign_bit_v` sign
flip + modular wrap into `[0, maxVal)` + `Clip1`; on 0, direct
`PaletteSizeUV`-many `L(BitDepth)` literals. The V plane is not
sorted either way per §5.11.46.

The walker grows two new grids — `PaletteSizes[plane][r][c]` and
`PaletteColors[plane][r][c][idx]`, packed flat with the plane
outermost — pre-filled at construction time to the spec's "no
palette" identity. Decoded entries are stamped over the block's
`bh4 * bw4` footprint per plane so the next §5.11.46 call's
§5.11.49 neighbour merge observes the propagated values. The
§5.11.49 `(MiRow * MI_SIZE) % 64` superblock-top gate suppresses
the above-neighbour read at 64×64 boundaries; out-of-grid reads
return 0, the spec-correct identity. New public accessors:
`palette_sizes()` and `palette_colors()` (flat slice views) plus
`get_palette_cache(plane, mi_row, mi_col, &mut [u16; 16])` (runs
the merge + dedupe and returns the entry count). Two new free
helpers cover the §4.7 mathematical primitives the §5.11.46
reader needs: `clip1_to_bit_depth(x, bd)` (§4.7 `Clip1`) and
`ceil_log2_av1(x)` (§4.7 `CeilLog2`, defined to return 0 on
`x < 2`).

New on `DecodedIntraBlockModeInfo`: three optional palette-entry
arrays — `palette_colors_y / _u / _v: Option<[u16; PALETTE_COLORS]>`
— carrying the decoded entries (Y and U sorted; V in source order
per spec). `palette_size_y` / `palette_size_uv` now report
`Some(decoded_size)` on the conformant path (previously stayed
`None` because the reader short-circuited before committing).
`Error::PaletteEntriesUnsupported` is retained for ABI stability
but is no longer constructed on the conformant code path.

12 new unit tests cover: the `bit_depth ∈ {8, 10, 12}`
caller-bug guard; an empty-cache fresh-walker read;
`clip1_to_bit_depth` / `ceil_log2_av1` truth tables at the
spec-defined boundaries; `decode_intra_block_mode_info` with a
rigged `has_palette_y == 0` path returning `Ok` with empty
palette state; `read_palette_entries_y` direct-call with
`PaletteSizeY = 2` and a zero bitstream yielding `[0, 1]` (the Y
`delta + 1` step); the U reader yielding `[0, 0]` (no `++`); the
V direct-literal arm yielding `[0, 0]`; hand-stamped palette
grids visible through `get_palette_cache` (left-neighbour-only);
the §5.11.49 above + left merge with a duplicate entry; the
§5.11.49 superblock-top boundary gate; and the
`read_palette_entries_y` cache-coded no-bit-read path via the
`get_palette_cache` accessor. Test count: 615 → 627 (+12).
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 169 lands **§5.11.22 `intra_block_mode_info()`** (av1-spec
p.73) — the per-block intra-mode syntax composite reached from the
§5.11.18 `else` arm of `if (is_inter)`. The new
`PartitionWalker::decode_intra_block_mode_info` reader composes every
§5.11.22 leaf in spec order: `RefFrame[0] = INTRA_FRAME, RefFrame[1] =
NONE` (constant `[0, -1]`); `y_mode` S() against `TileYModeCdf[ctx =
Size_Group[MiSize]]` (decoded value in `0..INTRA_MODES = 13`) + the
§5.11.5 `YModes[r + y][c + x] = YMode` grid-fill over the `bh4 * bw4`
footprint; §5.11.42 `intra_angle_info_y()` (gated on `MiSize >=
BLOCK_8X8 && is_directional(YMode)`, reads against
`TileAngleDeltaCdf[YMode - V_PRED]`, biased into
`-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA = -3..=3`); the `if (HasChroma)`
arm with `uv_mode` S() (§8.3.2 CFL-allowance branch routes through the
new `cfl_allowed_for_uv_mode` helper: lossless + post-subsampling 4×4
OR !lossless + Max(W,H) ≤ 32 → `TileUVModeCflAllowedCdf[YMode]`, else
`TileUVModeCflNotAllowedCdf[YMode]`); §5.11.45 `read_cfl_alphas()`
(gated on `UVMode == UV_CFL_PRED == 13`, reads `cfl_alpha_signs` S()
against `TileCflSignCdf` then per-axis `cfl_alpha_{u,v}` S() against
`TileCflAlphaCdf[ctx]` with sign-decomposition `signU =
(signs+1)/3, signV = (signs+1)%3` and `CflAlpha{U,V} = ±(1 + raw)` per
sign); §5.11.43 `intra_angle_info_uv()` (mirror of §5.11.42 against
`TileAngleDeltaCdf[UVMode - V_PRED]`); §5.11.46 `palette_mode_info()`
(outer gate `MiSize >= BLOCK_8X8 && Block_W ≤ 64 && Block_H ≤ 64 &&
allow_screen_content_tools`, per-plane reads `has_palette_*` S() then
on 1 reads `palette_size_*_minus_2` S() and surfaces
`Err(PaletteEntriesUnsupported)` — the §5.11.46 `palette_colors_*[]`
literal + delta reads need parser-scope `BitDepth` + `PaletteCache[]`
plumbing the walker doesn't yet thread through); §5.11.24
`filter_intra_mode_info()` (outer gate `enable_filter_intra && YMode
== DC_PRED && PaletteSizeY == 0 && Max(Block_W, Block_H) ≤ 32`, reads
`use_filter_intra` S() against `TileFilterIntraCdf[MiSize]` then on 1
reads `filter_intra_mode` S() against `TileFilterIntraModeCdf`).

Returns `DecodedIntraBlockModeInfo` carrying every decoded value with
`Option<…>` fields for the spec's "not read" arms (uv_mode /
cfl_alpha_* / angle_delta_uv / palette_size_* / use_filter_intra /
filter_intra_mode). Two new public free functions:
`is_directional(mode)` (§5.11.44, `V_PRED ≤ mode ≤ D67_PRED`) and
`cfl_allowed_for_uv_mode(lossless, mi_size, sub_x, sub_y)`
(§8.3.2 derivation). Three new public constants: `D67_PRED = 8`,
`UV_CFL_PRED = 13`, plus one new `Error` variant
`PaletteEntriesUnsupported`. 15 new unit tests cover the bounds
guards, the DC_PRED all-gates-off happy path, the HasChroma-false
short-circuit, the §5.11.42 non-directional / small-block
short-circuits, the §5.11.46 / §5.11.24 outer-gate short-circuits,
the §5.11.22 RefFrame invariant, the §5.11.5 grid-fill (DC and
non-DC), the §5.11.44 truth-table, and the §8.3.2 CFL-allowance
truth-table.

The §5.11.18 `decode_inter_frame_mode_info` dispatcher still
short-circuits at `Error::IntraBlockModeInfoUnsupported` — wiring the
new reader into the dispatcher needs additional sequence-header
arguments (`has_chroma`, `allow_screen_content_tools`,
`enable_filter_intra`, `subsampling_x`, `subsampling_y`,
`above_palette_y`, `left_palette_y`) that the §5.11.18 signature
doesn't yet thread through. Direct callers can invoke
`decode_intra_block_mode_info` with the missing arguments. Test count:
587 → 602 (+15). `decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 170 lands **§5.11.23 `inter_block_mode_info()` prologue +
§5.11.25 `read_ref_frames()`** (av1-spec p.74-77) — the per-block
inter-mode reference-frame syntax tree reached from the §5.11.18
`if (is_inter)` arm. The new
`PartitionWalker::decode_inter_block_mode_info` reader composes:
lines 1-2 (`PaletteSizeY = 0, PaletteSizeUV = 0`); §5.11.25
`read_ref_frames()` four-arm dispatch — `skip_mode` arm
(`RefFrame[0..2] = SkipModeFrame[0..2]`, no `S()`),
`seg_feature_active(SEG_LVL_REF_FRAME)` arm
(`RefFrame[0] = seg_ref_frame_data, RefFrame[1] = NONE`, no `S()`),
`seg_feature_active(SEG_LVL_SKIP | SEG_LVL_GLOBALMV)` arm
(`RefFrame[0] = LAST_FRAME, RefFrame[1] = NONE`, no `S()`), and the
default syntax-tree arm (`comp_mode` `S()` gated on
`reference_select && Min(bw4, bh4) >= 2`; COMPOUND splits into the
UNIDIR_COMP cascade `uni_comp_ref` / `uni_comp_ref_p1` /
`uni_comp_ref_p2` or the BIDIR_COMP cascade `comp_ref` /
`comp_ref_p1` / `comp_ref_p2` + `comp_bwdref` / `comp_bwdref_p1`;
SINGLE runs the `single_ref_p1..p6` cascade); line 4
(`isCompound = RefFrame[1] > INTRA_FRAME`); and the walker grid stamp
`RefFrames[r + y][c + x][0..2]` over the `bh4 * bw4` footprint.

The reader then short-circuits at the §7.10.2 `find_mv_stack(isCompound)`
entry with the new `Error::FindMvStackUnsupported`. The §7.10
motion-vector-stack derivation (scan-row, scan-col, temporal-scan,
sorting, extra-search, context-and-clamping) and every dependent
§5.11.23 reader (`compound_mode`, `new_mv` / `zero_mv` / `ref_mv`,
`drl_mode`, `assign_mv`, `read_motion_mode`, `read_interintra_mode`,
`read_compound_type`, `read_interpolation_filter`) remain
subsequent-arc targets. Every §5.11.25 `S()` read + the grid stamp
commits to state before the stub fires; the §5.11.18 dispatcher's
`if (is_inter)` arm now routes through the new reader (the
`InterBlockModeInfoUnsupported` variant becomes a defensive
caller-bug fallback no longer fired on the conformant path).

Five new §8.3.2 free functions land for the ref-frame ctx walks:
`check_backward(ref)` (`BWDREF_FRAME..=ALTREF_FRAME`),
`is_samedir_ref_pair(ref0, ref1)`, `count_refs(frame_type, …)`,
`comp_mode_ctx(…)` (the av1-spec p.366 nine-arm dispatch), and
`comp_ref_type_ctx(…)` (the av1-spec p.382 three-nested-if
dispatch).

The walker gains a `RefFrames[][][..]` grid as a flat `Vec<i8>`
with two slots per `(row, col)` cell, pre-filled with the §5.11.18
"unavailable neighbour" identity `[INTRA_FRAME = 0, NONE = -1]`.
View accessor `ref_frames()` exposes it for tests / callers; the
§5.11.18 prologue's `Left/AboveRefFrame[..]` derivations now consult
the grid so subsequent inter blocks observe the propagated values.

Eight new ref-frame ordinal constants land: `LAST2_FRAME = 2`,
`LAST3_FRAME = 3`, `GOLDEN_FRAME = 4`, `BWDREF_FRAME = 5`,
`ALTREF2_FRAME = 6` (plus the existing `INTRA_FRAME = 0`,
`LAST_FRAME = 1`, `ALTREF_FRAME = 7`), and the
`SINGLE_REFERENCE = 0`, `COMPOUND_REFERENCE = 1`,
`UNIDIR_COMP_REFERENCE = 0`, `BIDIR_COMP_REFERENCE = 1` enum
constants. New aggregate `DecodedInterBlockModeInfo` carries the
§5.11.25 output + `is_compound` derivation (observable only via the
walker grid since the reader always returns `Err`-path until §7.10
lands). The §5.11.18 dispatcher signature gains four new arguments
(`skip_mode_frame: [i32; 2]`, `seg_skip_active`, `seg_ref_frame_data`,
`reference_select`).

13 new unit tests cover the §8.3.2 ctx helpers (check_backward /
is_samedir_ref_pair / count_refs truth tables, comp_mode_ctx
nine-arm corners), the §5.11.23 caller-bug guards (out-of-range
mi_row / mi_col / sub_size / seg_ref_frame_data / skip_mode_frame),
the four §5.11.25 arm dispatch paths (skip_mode, seg_ref_frame_active,
seg_skip / seg_globalmv, default), the §5.11.25 small-block
`Min(bw4, bh4) < 2` gate, and the walker `ref_frames()` grid
propagation. Test count: 602 → 615 (+13).
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 168 lands **§5.11.17 `read_var_tx_size()`** + **§5.11.18
`inter_frame_mode_info()`** (av1-spec p.70-71) — the two missing
inter-arm composites that bound the §5.11.5 walker's inter side.

The **§5.11.17 reader** is exposed as a new
`PartitionWalker::read_var_tx_size` method (callable standalone). The
spec body is a frame-edge clip + a recursive `txfm_split` symbol read
that bottoms out at `txSz == TX_4X4` or `depth == MAX_VARTX_DEPTH`,
with each terminal-else stamping `InterTxSizes[ row + i ][ col + j ] =
txSz` over the `(h4, w4)` sub-block footprint. The §8.3.2 `txfm_split`
ctx selector inlines the spec's `get_above_tx_width` /
`get_left_tx_height` helpers against the walker's grids (`Skips[]` /
`IsInters[]` / `MiSizes[]` / `InterTxSizes[]`) — the `row == MiRow` /
`col == MiCol` arm gates the `Skips && IsInters` early return, and
the fall-through reads `Tx_Width[ InterTxSizes[ above ] ]` /
`Tx_Height[ InterTxSizes[ left ] ]`. The ctx formula
`ctx = (txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 +
above + left` already had its `txfm_split_ctx` helper from earlier
rounds; this round wires it into the live recursion. The §5.11.16
`read_block_tx_size` inter-arm now enters `read_var_tx_size` instead
of surfacing `Error::ReadVarTxSizeUnsupported` — `TxSizes[]` is
stamped with the last terminal-else's `txSz` over the full block
footprint per §5.11.5.

New free function `find_tx_size( w, h )` (§5.11.36 spec helper) —
the linear scan over `TX_SIZES_ALL` returning the first ordinal whose
`(Tx_Width, Tx_Height)` matches. Used by the §8.3.2 `txfm_split` ctx
selector's `maxTxSz = find_tx_size( size, size )` derivation.

The **§5.11.18 reader** is exposed as a new
`PartitionWalker::decode_inter_frame_mode_info` method composing every
pre-dispatch leaf in spec order: `use_intrabc = 0`; the §5.11.18
`LeftRefFrame[..]` / `AboveRefFrame[..]` / `LeftIntra` / `AboveIntra`
/ `LeftSingle` / `AboveSingle` local derivations (currently fixed at
`[INTRA_FRAME, NONE]` since the walker doesn't yet track
`RefFrames[][][..]` — the §5.11.23 readers' next-round target);
`skip = 0`; §5.11.19 `inter_segment_id(1)` via
`decode_inter_segment_id`; §5.11.10 `read_skip_mode()` via
`decode_skip_mode`; the `if (skip_mode) skip = 1 else read_skip()`
dispatch (the `skip_mode == 1` arm bypasses `decode_skip` and stamps
`Skips[][] = 1` directly per the §5.11.5 grid-fill invariant);
§5.11.19 `inter_segment_id(0)` (the post-skip arm, fired only when
`!SegIdPreSkip`); `Lossless = LosslessArray[segment_id]`; §5.11.56
`read_cdef()`; §5.11.12 `read_delta_qindex()`; §5.11.13
`read_delta_lf()`; §5.11.20 `read_is_inter()` via `decode_is_inter`.

The §5.11.18 terminal `if (is_inter) inter_block_mode_info() else
intra_block_mode_info()` dispatch short-circuits at two new `Error`
variants: `Error::InterBlockModeInfoUnsupported` (the §5.11.23
`inter_block_mode_info()` next-round target — MV stack / ref-frame
readers) and `Error::IntraBlockModeInfoUnsupported` (the §5.11.22
`intra_block_mode_info()` next-round target — per-block intra angle
/ UV mode readers). All pre-dispatch reads commit to the bitstream
/ grids before the stub fires.

New `DecodedInterFrameModeInfo` per-block aggregate carries every
§5.11.18 derived value (`mi_row` / `mi_col` / `mi_size` / `use_intrabc`
/ `avail_u` / `avail_l` / `left_ref_frame` / `above_ref_frame` /
`left_intra` / `above_intra` / `left_single` / `above_single` / `skip`
/ `skip_mode` / `segment_id` / `lossless` / `cdef_idx` /
`current_q_index` / `current_delta_lf` / `is_inter`).

The §5.11.5 `decode_block_syntax` walker is unchanged on the
`frame_is_intra = false` arm — it still short-circuits with
`Error::DecodeBlockInterFrameUnsupported` (the umbrella stub) because
the §5.11.18 reader needs additional segmentation-feature /
skip-mode-present caller state the §5.11.5 driver doesn't yet
thread through. Direct callers of `decode_inter_frame_mode_info`
get the full pre-dispatch walk and the §5.11.22 / §5.11.23
distinction.

11 new integration tests (`tests/decode_block_syntax_walker.rs`)
cover: the §5.11.17 base case (`TX_4X4`) with no S() read; the
depth cap (`MAX_VARTX_DEPTH`) with no S() read; the split path with
all-`1` `txfm_split` returning `TX_4X4` at depth 2; the frame-edge
clip; the caller-bug guards; the inter-arm `read_block_tx_size` now
enters `read_var_tx_size` and returns the input `maxTxSz` on the
no-split path with the `(4, 4)` footprint of `InterTxSizes[]`
stamped; the §5.11.18 baseline path reaches the §5.11.22 intra stub;
the §5.11.20 segment-override arm reaches the §5.11.23 inter stub;
the `skip_mode = 1` arm forces `Skips[][] = 1` + reaches §5.11.23;
the `seg_globalmv_active` arm reaches §5.11.23; the four
caller-bug guards. Plus 1 new unit test for `find_tx_size` on
square and rectangular sizes.

`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 167 lands **§5.11.16 `read_block_tx_size()`** (av1-spec p.70) —
the per-block transform-size syntax-tree read that the r166 §5.11.5
walker hits as the first stub on the intra arm. Exposed as a new
`PartitionWalker::read_block_tx_size` method (callable standalone) and
wired into `decode_block_syntax` so the walker now reaches §5.11.30
`compute_prediction()` instead.

The §5.11.16 reader transcribes the spec body one-to-one:

* `bw4` / `bh4` from `Num_4x4_Blocks_*[ MiSize ]`.
* The outer `TX_MODE_SELECT && MiSize > BLOCK_4X4 && is_inter && !skip
  && !Lossless` gate routes to §5.11.17 `read_var_tx_size` (deferred
  to the next arc — surfaces `Error::ReadVarTxSizeUnsupported`).
* The `else` arm performs §5.11.15 `read_tx_size(!skip || !is_inter)`
  inline: the `Lossless` short-circuit forces `TxSize = TX_4X4` (no
  S() consumed); otherwise `TxSize` starts at
  `Max_Tx_Size_Rect[ MiSize ]` and is further split `tx_depth` times
  via `Split_Tx_Size[]` when `MiSize > BLOCK_4X4 && allowSelect &&
  TxMode == TX_MODE_SELECT`.
* `tx_depth` is an `S()` against the §8.3.2-selected CDF. The
  selector is `TileTx{8x8,16x16,32x32,64x64}Cdf[ ctx ]` indexed by
  `maxTxDepth = Max_Tx_Depth[ MiSize ] ∈ { 1, 2, 3, 4 }`. The ctx
  derivation `ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)`
  walks the §8.3.2 neighbour ladder: when `AvailU &&
  IsInters[above]`, `aboveW = Block_Width[ MiSizes[above] ]`;
  otherwise `aboveW = Tx_Width[ InterTxSizes[above] ]` per the
  `get_above_tx_width` helper. Mirrored for `leftH`.
* The `else`-arm grid-fill loops over `(row, col) ∈ MiRow + 0..bh4,
  MiCol + 0..bw4` and stamps `InterTxSizes[ row ][ col ] = TxSize`.
  The §5.11.5 outer footer additionally stamps
  `TxSizes[ r + y ][ c + x ] = TxSize` over the same footprint —
  both stamps land in the reader.

New `PartitionWalker` grids: `tx_sizes: Vec<u8>` for the §5.11.5
`TxSizes[][]` writes and `inter_tx_sizes: Vec<u8>` for the §5.11.16
/ §5.11.17 `InterTxSizes[][]` writes. Both initialised to `TX_4X4`
(the §8.3.2 ctx-walk identity for an unavailable neighbour:
`Tx_Width[TX_4X4] = 4`). Public accessors `tx_sizes()` /
`inter_tx_sizes()` surface them after the walk.

New spec tables transcribed in `src/cdf.rs`:

* `MAX_TX_SIZE_RECT[ BLOCK_SIZES ]` — `Max_Tx_Size_Rect[ MiSize ]`
  from av1-spec p.402. The square `BLOCK_NxN → TX_NxN` identity for
  the four primary square sizes, `TX_64X64` cap for the 128×*
  blocks, and the matching rectangular `TX_*` entry for every
  rectangular `BLOCK_*`.
* `MAX_TX_DEPTH_TABLE[ BLOCK_SIZES ]` — `Max_Tx_Depth[ MiSize ]`
  from av1-spec p.69. The four-row listing
  `{ 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4,
  4 }`. Named `MAX_TX_DEPTH_TABLE` to avoid shadowing the existing
  `MAX_TX_DEPTH = 2` symbol-cap constant (per the §3 listing /
  CDF row-length contract).
* `SPLIT_TX_SIZE[ TX_SIZES_ALL ]` — `Split_Tx_Size[ TxSize ]` from
  av1-spec p.404. The recursive-split table that takes a transform
  size to the size obtained by splitting it into four
  sub-transforms; `TX_4X4` is a fixed point.
* `MAX_VARTX_DEPTH = 2` — the §3 constant table value for the
  §5.11.17 recursion depth cap.

New `TX_*` rectangular ordinals (`TX_4X8` through `TX_64X16`, ordinals
5..=18 per §6.10.16), needed for `MAX_TX_SIZE_RECT` and the future
§5.11.17 reader.

`DecodedBlock` gains a `tx_size: u8` field carrying the §5.11.16
return. The struct is the no-stub-path return of the §5.11.5 walker;
it remains publicly constructible.

`decode_block_syntax` gains a `tx_mode_select: bool` parameter
(threaded from the §5.9.21 / §6.8.21 frame-header `TxMode ==
TX_MODE_SELECT` derivation), now invokes `read_block_tx_size` after
the §5.11.49 `palette_tokens()` no-op, and short-circuits with the
new `Error::DecodeBlockComputePredictionUnsupported` (the §5.11.30
target). The §5.11.16 stub variant
`Error::DecodeBlockReadBlockTxSizeUnsupported` is retained for API
stability but no longer reached from the walker.

New `Error::ReadVarTxSizeUnsupported` variant surfaces the §5.11.17
`read_var_tx_size` deferral on the inter `TX_MODE_SELECT && !skip &&
!Lossless` arm. Currently unreachable from `decode_block_syntax`
(the inter arm is stubbed upstream at
`Error::DecodeBlockInterFrameUnsupported`); reachable from direct
`read_block_tx_size` calls with the right (`is_inter = 1`, `skip = 0`,
`lossless = false`, `tx_mode_select = true`) shape.

10 new integration tests
(`tests/decode_block_syntax_walker.rs`): the standalone reader's
`Lossless` short-circuit forces `TX_4X4` with no S() consumed; the
`TX_MODE_LARGEST` arm skips the `tx_depth` read and returns
`maxRectTxSize`; the `BLOCK_4X4` arm skips the read regardless of
TxMode (`MiSize > BLOCK_4X4` is false); `TX_MODE_SELECT` with rigged
`tx_depth = 0` / `1` / `2` walks the `Split_Tx_Size` chain
`TX_16X16 → TX_8X8 → TX_4X4` for a BLOCK_16X16 block; the inter
`TX_MODE_SELECT` arm surfaces the `ReadVarTxSizeUnsupported` stub
without bitstream consumption; the inter `skip` path falls through
the `else` arm with `allowSelect = false` and yields `maxRectTxSize`;
out-of-range guards reject the three caller-bug cases; and the
integrated walker reaches `compute_prediction` after the
`read_block_tx_size` pass stamps `TxSizes[]` / `InterTxSizes[]` over
the BLOCK_16X16 footprint. Plus 5 new unit tests for `MAX_TX_SIZE_RECT`
square-block identity, `MAX_TX_DEPTH_TABLE` spec-listing match,
`SPLIT_TX_SIZE` recursive-split contract, `tx_depth_ctx` combination
table, and `MAX_VARTX_DEPTH` value.

The §5.11.5 calls that remain STUBBED:

* **§5.11.17 `read_var_tx_size()`** — the variable-transform-tree
  recursion the §5.11.16 inter-arm enters; the immediate next-round
  target for the §5.11.16 inter-arm completion (paired with §5.11.18
  inter mode-info).
* **§5.11.18 `inter_frame_mode_info()`** — the inter arm of §5.11.6
  `mode_info()`; the immediate next-round target along the intra arc.
* **§5.11.30 `compute_prediction()`** — reached after §5.11.16. The
  immediate next-round target on the intra arm.
* **§5.11.34 `residual()`** — reachable once §5.11.30 lands.

`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 166 lands the §5.11.5 **`decode_block()` syntax-walker
skeleton** (av1-spec p.63-64) — the missing dispatcher that the
§5.11.4 partition walker recurses into at every leaf. Exposed as a
new `PartitionWalker::decode_block_syntax` method that performs the
§5.11.5 prologue (`MiRow` / `MiCol` / `MiSize` / `bw4` / `bh4` /
`HasChroma` three-arm dispatch / `AvailU` / `AvailL` /
`AvailUChroma` / `AvailLChroma` with the chroma fix-up arms for
sub-sampled 1×1 edge cases) and the §5.11.6 `mode_info()` intra arm
(routed through the implemented `decode_intra_frame_mode_info_prefix`
+ `decode_use_intrabc` + `decode_intra_frame_y_mode` composition,
plus the `use_intrabc == 1` short-circuit's `YMode = DC_PRED` /
`is_inter = 1` fixed assignments). The §5.11.49 `palette_tokens()`
call is a no-op on the no-palette path reachable while
`palette_mode_info()` remains unimplemented (the spec's outer `if
( PaletteSize{Y,UV} )` guard short-circuits). After the mode-info
pass completes the walker emits a `DecodedBlockRecord` leaf (same
as `decode_partition`'s leaf emitter) and short-circuits at the
§5.11.16 `read_block_tx_size()` call with a new
`Error::DecodeBlockReadBlockTxSizeUnsupported` variant — the
immediate next-round target.

Four new `Error` stub variants surface the §5.11.5 next-round
boundaries one-to-one: `DecodeBlockInterFrameUnsupported` (§5.11.18
inter_frame_mode_info, fires on `frame_is_intra = false` with zero
bits consumed); `DecodeBlockReadBlockTxSizeUnsupported` (§5.11.16
read_block_tx_size, the immediate next stub); and the reserved
`DecodeBlockComputePredictionUnsupported` (§5.11.30) and
`DecodeBlockResidualUnsupported` (§5.11.34) variants for the rounds
that land §5.11.16 → §5.11.30 → §5.11.34 in turn.

New `DecodedBlock` per-block aggregate carries the §5.11.5 prologue
+ §5.11.7 mode-info derivations (`mi_row` / `mi_col` / `mi_size` /
`bw4` / `bh4` / `has_chroma` / `avail_u` / `avail_l` /
`avail_u_chroma` / `avail_l_chroma` + the `IntraFrameModeInfoPrefix`
fields + `use_intrabc` / `is_inter` / `y_mode` / `is_compound`).
Constructible publicly so the round that lands §5.11.16 can return
it from the no-stub path.

Partition-walker driver: a new
`PartitionWalker::decode_partition_syntax` method mirrors the
§5.11.4 recursion of `decode_partition` exactly (same partition
symbol reads, same `partition_subsize` dispatch, same edge-of-frame
fall-throughs) but routes every `decode_block( r, c, sz )` leaf
call through `decode_block_syntax` instead of the leaf-only emitter.
On a stub the recursion short-circuits with the stub variant; grid
stamps stamped before the short-circuit remain observable on the
walker's grid accessors.

10 new integration tests
(`tests/decode_block_syntax_walker.rs`): the baseline
keyframe / no-segmentation / no-screen-content path reaches the
§5.11.16 stub after consuming the §5.11.11 skip + §5.11.7
`intra_frame_y_mode` bits + emitting one leaf record with the
expected mi-grid stamps; the §5.11.5 prologue `HasChroma` three-arm
dispatch on `BLOCK_4X4` (subsampling-y / subsampling-x / fall-
through); the §5.11.6 inter-frame arm surfaces the §5.11.18 stub
with zero bits consumed and no leaf record; the §5.11.7
`SegIdPreSkip = true` arm reaches the stub after the pre-skip
`intra_segment_id` + `read_skip` pair fire in spec order;
out-of-range guard symmetry (`mi_row` / `mi_col` / `sub_size`); the
§5.11.4 partition driver routes a `BLOCK_4X4` superblock (the
`< BLOCK_8X8` short-circuit arm) through `decode_block_syntax` and
propagates the stub; the §5.11.4 `r >= MiRows` early return
short-circuits cleanly; `DecodedBlock` public-API constructibility;
`BLOCK_8X16` grid-fill footprint (bw4 = 2, bh4 = 4) at the frame
origin; `BLOCK_16X16` with `cdef_bits = 2` exercises the §5.11.56
literal-bits read path.

The §5.11.5 calls that are STUBBED with `NotImplemented`-class
errors and become the next round's targets:

* **§5.11.16 `read_block_tx_size()`** + its §5.11.15 / §5.11.17
  sub-tree — the immediate next-round target.
* **§5.11.18 `inter_frame_mode_info()`** — the inter arm of
  §5.11.6 `mode_info()`.
* **§5.11.30 `compute_prediction()`** — reachable once §5.11.16
  lands.
* **§5.11.34 `residual()`** — reachable once §5.11.30 lands.

The §5.11.7 follow-on `else`-arm elements (`intra_angle_info_y`,
`uv_mode`, `read_cfl_alphas`, `intra_angle_info_uv`,
`palette_mode_info`, `filter_intra_mode_info`) and the `use_intrabc
== 1` MV-stack / assign-mv body remain bounded leaf targets that
can be slotted into a future round before or alongside §5.11.16.
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 165 lands the §5.11.7 / §5.11.22 **`intra_frame_y_mode` syntax
element** (av1-spec p.65) as a new
`PartitionWalker::decode_intra_frame_y_mode` method — the per-block
luma intra-prediction-mode selector read on the §5.11.7
`intra_frame_mode_info()` `else` arm (`use_intrabc == 0`), immediately
after the `is_inter = 0` assignment that r164's
`decode_use_intrabc` fall-through arm produced. The spec body is
the two-line `intra_frame_y_mode S(); YMode = intra_frame_y_mode`;
the dispatcher reads a single `S()` symbol against the §8.3.2
`intra_frame_y_mode` ctx-selected CDF and stamps the result over the
block's `bw4 * bh4` footprint of a new `YModes[][]` grid. New
`PartitionWalker::y_modes: Vec<u8>` field — a `mi_rows * mi_cols`
row-major buffer covering the §6.10.4 `YModes[ r ][ c ]` grid
(av1-spec p.378 / §5.11.5). Cells are initialised to `0` (=
`DC_PRED`); the initial-zero state matches the §8.3.2 ctx walk's
"neighbour unavailable" arm (`abovemode = Intra_Mode_Context[ AvailU
? YModes[ MiRow - 1 ][ MiCol ] : DC_PRED ]` — an
unavailable-or-pre-write neighbour contributes the same
`Intra_Mode_Context[ DC_PRED ] = 0` weight). New `y_modes()` read
accessor surfaces the grid; a private `y_mode_at` helper performs
the bounds-clipped neighbour lookup. §8.3.2 ctx derivation honours
both §5.11.51 tile-bound predicates (`AvailU` / `AvailL` via
`TileGeometry::is_inside`) and the §8.3.2 `Intra_Mode_Context[]`
mapping (driven through the existing `intra_mode_ctx` helper). The
CDF row is selected via the existing
`TileCdfContext::intra_frame_y_mode_cdf` accessor — no new
`Default_*` table is added (r127 already transcribed the §9.4
`Default_Intra_Frame_Y_Mode_Cdf` verbatim). 7 new cdf-module tests
(539 → 546): fresh-walker `YModes[]` zero-init; three-way
out-of-range guard (`sub_size`, `mi_row`, `mi_col`) with no bits
read; rigged-CDF returns over symbols 0 (`DC_PRED`) and 12
(`PAETH_PRED`, the largest valid `YMode`), each with footprint
grid-stamp verification; (0,0) corner block routes through ctx
`(0, 0)` (both neighbours unavailable ⇒ both map to `DC_PRED`);
neighbour-`YModes`-grid read after a first block stamps `YMode =
V_PRED = 1`, observable via the next block's distinctive-symbol
return; and `Intra_Mode_Context[]` table application (a `YMode = 4`
= `D203_PRED` neighbour maps to ctx 4 per the §8.3.2 table). Closes
the first leaf of the §5.11.7 `else` arm and the first sub-element
of the §5.11.22 `intra_block_mode_info` composite. Remaining
`else`-arm elements (`intra_angle_info_y`, `uv_mode`,
`read_cfl_alphas`, `intra_angle_info_uv`, `palette_mode_info`,
`filter_intra_mode_info`) remain bounded leaf targets for r166+;
the `use_intrabc == 1` short-circuit still awaits the
motion-vector stack walker. `decode_av1` / `encode_av1` continue to
return `Error::NotImplemented`.

Round 164 lands the §5.11.7 **`use_intrabc` syntax element** (av1-spec
p.65) as a new `PartitionWalker::decode_use_intrabc` method —
the per-block intra-block-copy enable bit read on the §5.11.7
`intra_frame_mode_info()` body immediately after the
`RefFrame[ 0..2 ]` assignments r161
`decode_intra_frame_mode_info_prefix` produced. The spec body is the
two-arm `if ( allow_intrabc ) { use_intrabc S() } else { use_intrabc
= 0 }`; the dispatcher routes both arms exactly, with no bit consumed
on the fall-through. New `DEFAULT_INTRABC_CDF: [u16; 3] = [30531,
32768, 0]` verbatim from §9.4 (av1-spec p.430), a single-row binary
CDF with no context index (the §8.3.2 selection text is contextless,
mirroring `Default_Delta_Q_Cdf` / `Default_Delta_Lf_Cdf`). New
`TileCdfContext::intrabc` field + `intrabc_cdf()` selector; constant
re-exported at the crate root. Unlike the §5.11.5
`decode_skip` / `decode_skip_mode` / `decode_is_inter` siblings,
`decode_use_intrabc` writes nothing to the walker's §5.11.5 grids
(`Skips[]` / `SkipModes[]` / `IsInters[]` / `SegmentIds[]`) — AV1
has no per-block `UseIntrabc[][]` map; the value is consumed locally
by the §5.11.7 follow-on arm. 7 new cdf-module tests (532 → 539):
`allow = false` zero-bit fall-through on a hostile `0xFF` buffer;
rigged S() arm forcing symbols 0 / 1; three-way out-of-range guard
(`sub_size`, `mi_row`, `mi_col`); contextless-selection check
across three `(mi_row, mi_col, sub_size)` triples; `DEFAULT_INTRABC_CDF
== [30531, 32768, 0]` layout assertion plus accessor round-trip; and
a no-grid-stamp invariant covering `Skips[]` / `SkipModes[]` /
`IsInters[]` after a `use_intrabc = 1` call. The §5.11.7 follow-on
body now divides into two remaining unblocked arms: (i) the
`use_intrabc == 1` short-circuit (needs the motion-vector
stack walker) and (ii) the `intra_block_mode_info` composite
(`intra_frame_y_mode`, `intra_angle_info_y`, `uv_mode`, etc., each a
bounded leaf, queued for r165+). `decode_av1` / `encode_av1`
continue to return `Error::NotImplemented`.

Round 163 lands the §5.11.21 **`get_segment_id()` predicted-segment-id
helper** (av1-spec p.72) as a new free function
`cdf::get_segment_id` (re-exported at the crate root). The §5.11.21
function is the inter-frame per-block segment-id **prediction**
lookup: it scans the previous frame's `PrevSegmentIds[][]` map over
the on-screen window `xMis = Min(MiCols - MiCol, bw4)` ×
`yMis = Min(MiRows - MiRow, bh4)` and returns the `Min` of the cells
visited, with the §5.11.21 sentinel `seg = 7` (i.e. `MAX_SEGMENTS -
1`) as the initial upper bound. Returns `Some(seg)` with `seg` in
`-1..=7` — the `-1` sentinel surfaces only if a not-yet-decoded
cell of a previous walker falls inside the window, letting callers
detect a malformed reference surface via the existing §5.11.19
`predicted_segment_id > last_active_seg_id` range guard. Returns
`None` for caller-bug arguments: out-of-range `sub_size`, anchor
outside the current frame, previous-frame extent smaller than the
current frame's, or `prev_segment_ids.len() != prev_mi_rows *
prev_mi_cols`. The function is **pure** (no walker state, no
bitreader, no CDF), and complements the r162
`PartitionWalker::decode_inter_segment_id` caller (which takes a
pre-computed `predicted_segment_id: u8` so the walker can stay
inter-frame-state-free) — the §6.10 reference-frame walk now has a
verbatim spec-shaped routine to compute that argument from
`PrevSegmentIds[]`. 12 new cdf-module tests (520 → 532):
uniform-0 / uniform-7 reductions over several block sizes;
Min-over-2x2-window cells with out-of-window decoy values that must
not contribute; frame-edge `xMis`/`yMis` clipping on a `BLOCK_16X16`
anchored at the bottom-right of a 4x4 mi-grid; `-1` sentinel
round-trip; a wider-than-current previous frame exercising the
`prev_mi_cols` row stride; single-cell `BLOCK_4X4` covering exactly
`prev[MiRow][MiCol]` with a neighbour-cell decoy; out-of-range
guards for invalid `sub_size`, anchor past frame extent, previous-
frame extent smaller than current frame's, and length / shape
mismatch; an end-to-end composition test feeding `get_segment_id`'s
result into `decode_inter_segment_id`'s no-read
`!segmentation_update_map` arm and verifying the predicted id is
adopted with zero bit reads on a hostile `0xFF` buffer. The §5.11.18
`inter_frame_mode_info()` top-level dispatcher, §5.11.7 `use_intrabc`
arm, and §5.11.22 `intra_block_mode_info` composite remain the
next round's targets. `decode_av1` / `encode_av1` continue to
return `Error::NotImplemented`.

Round 162 lands the §5.11.19 **`inter_segment_id( preSkip )` syntax
element** (av1-spec p.71) as a new
`PartitionWalker::decode_inter_segment_id` method. `inter_segment_id`
is the §5.11.18 inter-frame variant of the per-block segment-id read,
called twice per block (once with `preSkip = 1` before the §5.11.11
`read_skip()` call and once with `preSkip = 0` after); together with
the §5.9.14 `SegIdPreSkip` derivation, the two calls cover every
combination of "segment-id read before / after skip" the inter-frame
walk encounters. New `SEGMENT_ID_PREDICTED_CONTEXTS = 3` constant and
new `DEFAULT_SEGMENT_ID_PREDICTED_CDF[3][3]` table verbatim from
av1-spec p.442 (each ctx row is the uniform `[128 * 128, 32768, 0]`
binary start). New `TileCdfContext::segment_id_predicted` field plus
`segment_id_predicted_cdf(ctx)` accessor implementing the §8.3.2
selection. New persistent `above_seg_pred_context: Vec<u8>` (length
`mi_cols`) and `left_seg_pred_context: Vec<u8>` (length `mi_rows`)
buffers on `PartitionWalker` per the §8.3.1 tile-entry initialisation
(`AboveSegPredContext[i] = 0`, `LeftSegPredContext[i] = 0`); the
§8.3.2 ctx walk reads `LeftSegPredContext[ MiRow ] +
AboveSegPredContext[ MiCol ]` (each `0..=1`, sum `0..=2`). The
dispatcher routes the full §5.11.19 cascade exactly: outer
`!segmentation_enabled` collapses to `segment_id = 0`; inner
`!segmentation_update_map` adopts `predictedSegmentId` without
reading; the `pre_skip && !SegIdPreSkip` early-exit returns
`segment_id = 0`; the post-skip `skip != 0` arm zeroes the context
arrays then descends into `decode_segment_id` (the §5.11.9 path
short-circuits on `skip`); the `segmentation_temporal_update == 1`
arm reads the binary `seg_id_predicted` symbol, branches to
`predictedSegmentId` adopt or `read_segment_id()`, then stamps the
context arrays; the `temporal_update == 0` fall-through reads
`read_segment_id()` without touching the context arrays. The
`predicted_segment_id` (§5.11.21 `get_segment_id()`) is caller-supplied
from the §6.10 reference-frame walk's `PrevSegmentIds[]` lookup so
the walker stays inter-frame-state-free. Range guards (`sub_size >=
BLOCK_SIZES`, `mi_row`/`mi_col` past extent, `last_active_seg_id >=
MAX_SEGMENTS`, `predicted_segment_id > last_active_seg_id`) fire
up-front. 11 new cdf-module tests (509 → 520): fresh-walker
context-array zeroing; `!segmentation_enabled` no-read on both
pre/post-skip arms; `!segmentation_update_map` adopts predicted-id
without reading; `pre_skip && !SegIdPreSkip` early-exit;
post-skip + `skip != 0` zeroes context arrays then descends into
the §5.11.9 short-circuit (poisoned-context arrays prove the spec
write fires); `temporal_update == 1` + rigged `seg_id_predicted = 1`
adopts `predictedSegmentId` and stamps context arrays to `1`;
`temporal_update == 1` + rigged `seg_id_predicted = 0` descends into
`read_segment_id()` and stamps context arrays to `0`;
`temporal_update == 0` fall-through leaves context arrays untouched;
five-way out-of-range guard; `Default_Segment_Id_Predicted_Cdf`
layout (`[16384, 32768, 0]` per row) and the
`segment_id_predicted_cdf` accessor round-trip. The §5.11.18
`inter_frame_mode_info()` top-level dispatcher (`use_intrabc` arm
+ the `LeftRefFrame`/`AboveRefFrame`/`LeftIntra`/`AboveIntra` /
`LeftSingle`/`AboveSingle` derivations + the §5.11.18 two-call
`inter_segment_id` protocol composing on top of r152
`read_skip()` / r156 `read_cdef()` / r154 `read_delta_qindex()` /
r155 `read_delta_lf()` / r158 `read_is_inter()` /
§5.11.22 `intra_block_mode_info` / §5.11.23 `inter_block_mode_info`)
is the next round's architectural payoff. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 161 lands the §5.11.7 **`intra_frame_mode_info()` prefix
dispatcher** (av1-spec p.64) as a new
`PartitionWalker::decode_intra_frame_mode_info_prefix` method
composing the first 11 lines of the §5.11.7 spec body into a single
walker entry-point. The dispatcher fires the §5.11.7 sequence in
exact spec order: `skip = 0`; conditional pre-skip
`intra_segment_id()` (r160); `skip_mode = 0`; `read_skip()` (r152);
conditional post-skip `intra_segment_id()`; `read_cdef()` (r156);
`read_delta_qindex()` (r154); `read_delta_lf()` (r155); the fixed
`RefFrame[0] = INTRA_FRAME` / `RefFrame[1] = NONE` trailing
assignments. Returns a new public `IntraFrameModeInfoPrefix` struct
carrying every post-call observable: `skip`, `skip_mode`,
`segment_id`, `lossless`, `cdef_idx`, `current_q_index`,
`current_delta_lf`, `ref_frame`. The §5.11.7 `SegIdPreSkip`
conditional routes the §5.11.8 call before or after the §5.11.11
`read_skip()` per the caller-passed `seg_id_pre_skip` boolean (the
§5.9.14 trailing derivation surfaced as
`SegmentationParams::seg_id_pre_skip`). `skip_mode` is fixed at `0`
because the intra-frame walk never calls `decode_skip_mode` —
§5.11.10 short-circuits on `!skip_mode_present` (intra-only frames
have `skip_mode_present == 0` per §5.9.21). The §6.10.4 `ReadDeltas
= 0` assignment on the spec's line 11 is left to the caller (the
walker stays stateless about per-superblock first-block detection),
matching the §6.10.4 pattern existing `decode_delta_qindex` /
`decode_delta_lf` call sites already use. Range guards (`sub_size >=
BLOCK_SIZES`, `mi_row >= MiRows`, `mi_col >= MiCols`,
`last_active_seg_id >= MAX_SEGMENTS`, `cdef_bits > 3`) fire on the
dispatcher level before any inner read so a caller bug never
produces a partial-read. 8 new cdf-module tests (501 → 509):
minimum-bit path (only the §5.11.11 `S()` consumed, every other
field short-circuits); `SegIdPreSkip = true` arm reads `segment_id`
(diff = 2 against rigged cdf) before `skip` (forced 1) and the
§5.11.56 `cdef_idx = -1` sentinel survives the skip short-circuit;
`SegIdPreSkip = false` arm post-skip with `skip = 1` triggers the
§5.11.9 short-circuit (`segment_id = pred = 0`, no bit consumed);
seg-skip-active path forces `skip = 1` with zero bits consumed on a
hostile `0xFF` buffer; `ref_frame = [INTRA_FRAME, NONE] = [0, -1]`
regardless of path taken; `read_deltas = true` + `delta_lf_present
= true` wires through to both delta reads (rigged `delta_q_abs = 0`
/ `delta_lf_abs = 0` ⇒ accumulators unchanged but the `S()` reads
advance the decoder); five-way out-of-range guard on `mi_row`,
`mi_col`, `sub_size`, `last_active_seg_id`, `cdef_bits`;
`skip_mode = 0` on both pre-skip arms (verifies the dispatcher
never calls `decode_skip_mode` and `SkipModes[]` stays untouched).
The §5.11.7 follow-on body (`use_intrabc` arm + the §5.11.22
`intra_block_mode_info` composite — `intra_frame_y_mode`,
`intra_angle_info_y`, `uv_mode`, `intra_angle_info_uv`,
`palette_mode_info`, `filter_intra_mode_info`) and the §5.11.18
`inter_frame_mode_info` / §5.11.19 `inter_segment_id` two-call
protocol remain the next round's targets. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 160 lands the §5.11.8 **`intra_segment_id()` syntax element**
(av1-spec p.66) as a new `PartitionWalker::decode_intra_segment_id`
method built on top of r159's `decode_segment_id`. `intra_segment_id`
is the intra-frame variant of the per-block segment-id read, called
from §5.11.7 `intra_frame_mode_info` on both the `SegIdPreSkip`
pre-skip arm and the `!SegIdPreSkip` post-skip arm. The §5.11.8 spec
body is short — `if (segmentation_enabled) read_segment_id(); else
segment_id = 0; Lossless = LosslessArray[segment_id]` — but the
Lossless lookup is the first place the per-segment §6.8.2
`LosslessArray[]` table reaches the leaf walk. The dispatch is exact:
the `segmentation_enabled = true` arm descends into the r159
implementation (which performs the §5.11.9 neighbour cascade, the
skip / non-skip dispatch, the `S()` read against
`TileSegmentIdCdf[ctx]`, the `neg_deinterleave` mapping, and the
§5.11.5 grid-fill); the `segmentation_enabled = false` arm forces
`segment_id = 0` without reading any bits and stamps the `bh4 * bw4`
footprint to `0` so subsequent §5.11.9 neighbour lookups see a real
zero rather than the `-1` sentinel. Both arms then resolve
`Lossless = lossless_array[segment_id as usize]` from the
caller-supplied `&[bool; MAX_SEGMENTS]` table (the §6.8.2 derivation
the frame-header walk computes from `qindex = get_qindex(1,
segmentId)` plus the five `DeltaQ?Dc` / `DeltaQ?Ac` offsets;
`compute_coded_lossless` in `frame_header.rs` is the frame-wide
conjunction, this round's table is the per-segment data the walker
indexes by `segment_id`). The walker stays segmentation-state-free:
callers pass `segmentation_enabled`, `last_active_seg_id`, and
`lossless_array` per-call, mirroring the r159 pattern. Range guards
(out-of-range `sub_size`, `mi_row` / `mi_col` past extent,
`last_active_seg_id >= MAX_SEGMENTS`) fire on both arms so the
no-symbol path is total over the same input space as the
bitstream-reading path. The §5.11.18 `inter_frame_mode_info`
top-level dispatcher (the §5.11.19 `inter_segment_id(preSkip)`
two-call protocol) remains the next round's target. 7 new cdf-module
tests (494 → 501): `!segmentation_enabled` no-read on a hostile
`0xFF` buffer with `lossless_array[0] = true` and `= false` arms
each; `segmentation_enabled = true, skip = 1` at frame origin
(pred = 0, no `S()` consumed, grid stamped to 0, Lossless from slot
0); `segmentation_enabled = true, skip = 0` reading `diff = 3` on a
rigged CDF (Lossless from slot 3 = true); per-segment Lossless
indexing (rig `diff = 5`, set `lossless_array[5] = false` while
every other slot is `true`, expect `Lossless = false`); bottom-right
edge clip on `BLOCK_16X16 @ (2, 2)` in a 4×4 frame on the
`!segmentation_enabled` arm stamps only the in-grid 2×2 quadrant
and leaves the rest at `-1`; five-way out-of-range guard
(`mi_row >= mi_rows`, `mi_col >= mi_cols`, `sub_size ==
BLOCK_SIZES`, `last_active_seg_id == MAX_SEGMENTS`, and the
`segmentation_enabled = true` path also rejects bad mi-row). The
§5.11.5 `decode_block()` body itself (coefficient / motion-vector /
reconstruction) remains the next round's target. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 159 lands the §5.11.9 **`read_segment_id()` syntax element**
(av1-spec p.66) as a new `PartitionWalker::decode_segment_id`
method on the r158 walker, plus a `SegmentIds[r][c]` grid carried
alongside the r158 `IsInters[]`, r156 `cdef_idx[]`, r154
`SkipModes[]`, r152 `Skips[]`, and the existing §6.10.4 `MiSizes[]`
grids. The grid is pre-filled with the §5.11.9 `-1` sentinel (the
spec's `prevUL = -1` / `prevU = -1` / `prevL = -1` "neighbour
unavailable" marker); cells inside a decoded block's `bh4 * bw4`
footprint then carry the block's `segment_id ∈ 0..MAX_SEGMENTS = 0..8`.
The walker stays segmentation-state-free: the caller passes
`last_active_seg_id` (the §5.9.14 trailing derivation) and the
`skip` value the §5.11.11 `decode_skip` just returned. The §5.11.9
neighbour cascade is honoured exactly as the spec spells it out:
`prevUL` requires both `AvailU` AND `AvailL`; `prevU` and `prevL`
each gate on their own edge; out-of-grid neighbours fall through to
`-1`. The four-arm `pred` derivation (`prevU == -1 ⇒ prevL/0`;
`prevL == -1 ⇒ prevU`; `prevUL == prevU ⇒ prevU`; else `prevL`) is
preserved verbatim — two arms happen to return `prev_u` but the
predicates are semantically distinct ("left neighbour unavailable"
vs. "above-left agrees with above"), and collapsing them would
obscure the spec correspondence. The §5.11.9 dispatch distinguishes
two paths: `skip != 0` ⇒ `segment_id = pred` (zero bits read; the
spatially-predicted-on-skip semantics the spec relies on for
skip-block segment-map continuity); else `diff S()` against
`TileSegmentIdCdf[ctx]` (ctx from the existing `segment_id_ctx`
helper, which already honours the `-1` sentinels), then `segment_id
= neg_deinterleave(diff, pred, last_active_seg_id + 1)`. The
§5.11.5 grid-fill stamps the result over the block's `bw4 * bh4`
footprint, clipped at the frame's extent. A new public module-level
`neg_deinterleave(diff, ref, max)` helper transcribes the §5.11.9
bijection (`diff ∈ 0..max ↔ segment_id ∈ 0..max` biased toward
values near `ref`). `decode_segment_id` is the read inside both
§5.11.8 `intra_segment_id` and §5.11.19 `inter_segment_id` (the
segmentation-enabled inner branch in each); the latter's `preSkip`
machinery and the `segment_id`-aware Arm 2 of §5.11.20 (which
consumes `FeatureData[segment_id][SEG_LVL_REF_FRAME]`) are now both
unblocked at the leaf-walk level. 11 new cdf-module tests (483 →
494): fresh-walker grid all `-1`; skip short-circuit at frame
origin writes `segment_id = pred = 0` (no S() bit consumed on a
hostile `0xFF` byte buffer); skip inherits `prev_u` when `prev_l`
is unavailable; non-skip path with `pred = 0` returns `diff`
unchanged; direct `neg_deinterleave` table exercises for both the
`2 * ref < max` upward branch (`pred = 2`, all eight `diff` values)
and the `2 * ref >= max` downward branch (`pred = 5`, all eight
`diff` values); edge cases (`ref == 0` identity, `ref == max - 1`
inverted, smallest non-trivial alphabet `max = 2`); ctx-0 origin
selection via rigged rows; ctx-2 all-neighbours-match selection
through three walker-stamped seeds; bottom-right edge clip on
`BLOCK_16X16 @ (2, 2)` in a 4×4 frame stamps only the in-grid 2×2
quadrant; four-way out-of-range guard (`mi_row` past extent /
`mi_col` past extent / `sub_size == BLOCK_SIZES` /
`last_active_seg_id >= MAX_SEGMENTS`) ⇒ `PartitionWalkOutOfRange`.
The §5.11.5 `decode_block()` body itself (coefficient /
motion-vector / reconstruction) remains the next round's target.
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 158 lands the §5.11.20 **`read_is_inter()` syntax element**
(av1-spec p.71-72) as a new `PartitionWalker::decode_is_inter`
method on the r157 walker, plus an `IsInters[r][c]` flag grid
carried alongside the r156 `cdef_idx[]`, r154 `SkipModes[]`,
r152 `Skips[]`, and the existing §6.10.4 `MiSizes[]` grids
(`IsInters[ r + y ][ c + x ] = is_inter` per the §5.11.5
footer at av1-spec p.65). All four arms of the §5.11.20 dispatch
are honoured in spec order (first match fires, no read on the
short-circuit arms): Arm 1 — `skip_mode == 1` forces
`is_inter = 1` (a compound-reference skip block is by definition
inter); Arm 2 — `seg_feature_active(SEG_LVL_REF_FRAME) == true`
routes through the caller-pre-computed
`FeatureData[segment_id][SEG_LVL_REF_FRAME] != INTRA_FRAME`
boolean (the walker stays segmentation-state-free, identical to
r154's `seg_skip_mode_off` pattern); Arm 3 —
`seg_feature_active(SEG_LVL_GLOBALMV) == true` forces
`is_inter = 1` (global-MV is intrinsically inter); Arm 4 — `S()`
symbol read against `TileIsInterCdf[ctx]` with `ctx` from the
existing `is_inter_ctx(above_intra, left_intra)` helper. Per
§5.11.18 the §8.3.2 neighbour intra-ness is sampled from
`LeftRefFrame[0] / AboveRefFrame[0]` (`LeftIntra = LeftRefFrame[0]
<= INTRA_FRAME`); the walker derives this from the complement of
its `IsInters[]` grid (`intra = !is_inter`), with an unavailable
neighbour treated as intra per §5.11.18 (`LeftRefFrame[0] = AvailL
? RefFrames[..][0] : INTRA_FRAME` ⇒ `None` to `is_inter_ctx`).
The §5.11.5 grid-fill stamps the decoded value over the block's
`bw4 * bh4` footprint, clipped at the frame's `MiRows` / `MiCols`
extent so a leaf straddling the bottom or right edge fills only
the in-grid portion. New `PartitionWalker::is_inters()` accessor
returns a row-major view; the fresh-walker initial state is
all-zero (a pre-write neighbour weights as intra in the §8.3.2
ctx walk, which is the natural identity for the spec's `LeftIntra
= LeftRefFrame[0] <= INTRA_FRAME` reading and matches an
unavailable neighbour gated by `AvailU` / `AvailL`). `is_inter`
is the per-block intra/inter classifier read inside §5.11.18
`inter_frame_mode_info` (after `inter_segment_id` / `read_skip` /
`read_cdef` / `read_delta_qindex` / `read_delta_lf`) that
dispatches between §5.11.22 `intra_block_mode_info` and §5.11.23
`inter_block_mode_info`; intra-only frames never call it. 15 new
cdf-module tests (468 → 483): fresh-walker grid all-zero; Arm 1
skip_mode short-circuit (position-invariant); Arm 2 routing to
intra (`seg_ref_frame_is_inter = false`) and to inter
(`seg_ref_frame_is_inter = true`), both position-invariant;
Arm 3 globalmv short-circuit; Arm 1 takes precedence over both
Arm 2 and Arm 3; Arm 2 takes precedence over Arm 3; else-branch
S() returning 0 / 1 on a forced binary CDF (the
`is_inter = 1` arm verifies the footprint grid-stamp); ctx-0
selection at the frame origin; ctx-3 selection through two prior
intra-stamping seeds; ctx-1 through one intra + one inter
neighbour with both available; ctx-2 through a non-zero-tile-col
origin clearing AvailL with an above-intra seed; bottom-right
edge clip on `BLOCK_16X16 @ (2, 2)` in a 4×4 frame; three-way
out-of-range guard ⇒ `PartitionWalkOutOfRange`. The §5.11.5
`decode_block()` body itself (coefficient / motion-vector /
reconstruction) remains the next round's target. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 157 lands the §5.11.56 **`read_cdef()` syntax element** plus
the §5.11.55 **`clear_cdef()` reset** (av1-spec p.104) as new
`PartitionWalker::decode_cdef` + `PartitionWalker::clear_cdef`
methods on the r156 walker, alongside a `cdef_idx: Vec<i8>`
row-major grid sized `MiRows × MiCols` (pre-filled with the `-1`
sentinel per §5.11.55, interpreted as "CDEF disabled for that
block" per §6.10.40) with a `cdef_idx()` read accessor. CDEF
operates on 64×64 anchor cells, so `decode_cdef` masks the leaf's
`(MiRow, MiCol)` to the anchor at `(MiRow & cdefMask4, MiCol &
cdefMask4)` (`cdefMask4 = ~(cdefSize4 - 1)`, `cdefSize4 =
Num_4x4_Blocks_Wide[ BLOCK_64X64 ] = 16` so the low four bits are
zeroed). If the anchor still holds the `-1` sentinel, an
`L(cdef_bits)` literal is read (`cdef_bits ∈ 0..=3` per §5.9.19
`f(2)` ⇒ decoded value in `0..=7`) and the grid-fill loop stamps
the value across the leaf's `(w4, h4)` footprint at the `cdefSize4
= 16` stride so super-64 blocks (`BLOCK_128X128`) reach all four
anchor cells while sub-64 blocks touch only their containing
anchor. Subsequent leaves whose `cdefMask4` lands on the same
anchor short-circuit (no read; the anchor already holds the
value). The §5.11.56 short-circuit set is honoured: `skip ||
CodedLossless || !enable_cdef || allow_intrabc` ⇒ no read; the
anchor's current value (sentinel or prior stamp) is returned
unchanged. `clear_cdef( r, c, use_128x128_superblock )` — called
by the §5.11.2 tile walk at each superblock — stamps `-1` at the
one (64×64 superblock) or four (128×128 superblock) anchor cells;
out-of-grid anchors are silently skipped so the bottom/right
superblock can straddle the frame edge without panic. `cdef_bits
== 0` yields `L(0) = 0` (no bit read) and still transitions the
anchor from `-1` to `0`, matching the §5.9.19 single-strength
case. 18 new cdf-module tests (450 → 468): fresh-walker all-`-1`
invariant; `clear_cdef` 64×64 single-anchor stamp; `clear_cdef`
128×128 four-anchor stamp; `clear_cdef` out-of-grid silent skip;
each of the four `skip` / `CodedLossless` / `!enable_cdef` /
`allow_intrabc` short-circuit gates (separately, with `0xFF` byte
buffers proving no bit consumed); first-leaf-reads-literal-
and-stamps-anchor with off-anchor cell stays at sentinel;
second-leaf-in-anchor-no-read; `cdef_bits == 0` zero-bit stamp;
`cdef_bits == 3` upper-bound; anchor-mask routes (10, 13) ⇒ (0,
0); `BLOCK_128X128` stamps all four anchors with off-anchor cells
at sentinel; grid-fill clips at frame edge; short-circuit returns
prior stamp; `clear_cdef` after stamp resets anchor; four-way
out-of-range guard (`mi_row` past extent / `mi_col` past extent /
`sub_size == BLOCK_SIZES` / `cdef_bits > 3`) ⇒
`PartitionWalkOutOfRange`. The §5.11.5 `decode_block()` body
itself (coefficient / motion-vector / reconstruction) remains the
next round's target. `decode_av1` / `encode_av1` continue to
return `Error::NotImplemented`.

Round 156 lands the §5.11.13 **`read_delta_lf()` syntax element**
(av1-spec p.68) as a new `PartitionWalker::decode_delta_lf` method,
structurally parallel to r155's §5.11.12 walker but iterating
`frameLfCount` times over a four-slot `DeltaLF[ i ]` accumulator
and selecting between the §8.3.2 single-LF (`TileDeltaLFCdf`) and
per-edge multi-LF (`TileDeltaLFMultiCdf[ i ]`) CDF rows via the
`delta_lf_multi` argument. Adds a
`current_delta_lf: [i32; FRAME_LF_COUNT]` accumulator on
`PartitionWalker` with `current_delta_lf()` read accessor and
`reset_current_delta_lf()` for the §5.11.2 tile-entry reset.
Honours the §5.11.13 superblock-skip short-circuit (identical to
§5.11.12) and the outer `ReadDeltas && delta_lf_present` gate
(two AND-ed flags — `delta_lf_present` is the §5.9.18 frame-header
bit, accepted as an argument). When the gate passes, `frameLfCount`
is derived locally: `delta_lf_multi == 0 ⇒ 1`;
`delta_lf_multi == 1 && mono_chrome ⇒ FRAME_LF_COUNT - 2 = 2`;
otherwise `FRAME_LF_COUNT = 4`. Each iteration reads
`delta_lf_abs` `S()` against the branch-selected CDF, then either
the literal value or the §5.11.13 escape ladder
(`delta_lf_rem_bits` `L(3)` + post-increment + `delta_lf_abs_bits`
`L(rem_bits + 1)` ⇒ `deltaLfAbs = abs_bits + (1 << n) + 1`); for
non-zero magnitudes reads `delta_lf_sign_bit` `L(1)` and applies
`DeltaLF[ i ] = Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER,
DeltaLF[ i ] + (reducedDeltaLfLevel << delta_lf_res))`. New
constants `DELTA_LF_SMALL = 3`, `FRAME_LF_COUNT = 4`,
`cdf::MAX_LOOP_FILTER = 63i32` (distinct from the §5.9.11
`uncompressed_header_tail::MAX_LOOP_FILTER` `i16` twin); new table
`DEFAULT_DELTA_LF_CDF` transcribed verbatim from §9.4 p.431
(`[28160, 32120, 32677, 32768, 0]`, identical row to
`DEFAULT_DELTA_Q_CDF` per the spec — preserved as two independent
constants so adaptation drift on one does not leak through the
other); new fields `TileCdfContext::delta_lf` +
`TileCdfContext::delta_lf_multi` with `delta_lf_cdf()` /
`delta_lf_multi_cdf(i)` accessors. 17 new cdf-module tests (433 ->
450): default-CDF literal match (incl. §9.4 equality with
`DEFAULT_DELTA_Q_CDF`); init-from-defaults invariant for the
single-LF row and all four multi-LF rows; sb-skip short-circuit
at both `use_128x128_superblock` settings; `ReadDeltas` false
short-circuit; `delta_lf_present` false short-circuit; single-LF
branch writes only `DeltaLF[ 0 ]`; multi-LF colour branch writes
all four slots; multi-LF monochrome branch writes only the two Y
slots; zero-`delta_lf_abs` no-update; literal-positive with
shift; Clip3 upper-bound at `MAX_LOOP_FILTER = 63`; Clip3
lower-bound via hostile seed at `i32::MIN + 1`; `DELTA_LF_SMALL`
escape ladder minimum value; cross-call accumulation;
fresh-walker initial accumulator all-zero + `reset_current_delta_lf`
round-trip; out-of-range guards ⇒ `PartitionWalkOutOfRange`. The
§5.11.5 `decode_block()` body itself (coefficient / motion-vector
/ reconstruction) remains the next round's target. `decode_av1`
/ `encode_av1` continue to return `Error::NotImplemented`.

Round 155 lands the §5.11.12 **`read_delta_qindex()` syntax
element** (av1-spec p.67) as a new
`PartitionWalker::decode_delta_qindex` method on the r154 walker,
plus a `CurrentQIndex` scalar tracked across calls via
`PartitionWalker::current_q_index()` /
`PartitionWalker::set_current_q_index()` accessors. Honours the
§5.11.12 superblock-skip short-circuit (`MiSize == sbSize && skip`
⇒ no symbol read; `sbSize = use_128x128_superblock ?
BLOCK_128X128 : BLOCK_64X64`, the latter via the §5.5.1
sequence-header flag) and the outer `ReadDeltas` (§6.10.4) gate.
Otherwise an `S()` symbol is decoded against `TileDeltaQCdf` —
length `DELTA_Q_SMALL + 2 = 5`, no context index per §8.3.2; the
working CDF is a single row. A decoded value of `DELTA_Q_SMALL`
(`= 3`) triggers the §5.11.12 escape ladder
(`delta_q_rem_bits` `L(3)` + post-increment + `delta_q_abs_bits`
`L(rem_bits + 1)`), reconstructing
`delta_q_abs = delta_q_abs_bits + (1 << n) + 1` over the extended
range `0..=2 ∪ 3..=511`. Non-zero `delta_q_abs` reads
`delta_q_sign_bit` `L(1)` and applies the spec's
`CurrentQIndex = Clip3(1, 255, CurrentQIndex +
(reducedDeltaQIndex << delta_q_res))` update. New constant
`DELTA_Q_SMALL = 3`, new table `DEFAULT_DELTA_Q_CDF` transcribed
verbatim from §9.4 p.431 (`[28160, 32120, 32677, 32768, 0]`), new
field `TileCdfContext::delta_q` + `delta_q_cdf()` accessor. 16
new cdf-module tests (417 -> 433): default-CDF literal match;
init-from-defaults invariant; sb-skip short-circuit for both
`use_128x128_superblock` values; `ReadDeltas` false short-circuit;
zero-`delta_q_abs` no-update; literal-positive no-shift; literal
positive with shift; Clip3 lower-bound via hostile seed; Clip3
upper-bound; DELTA_Q_SMALL escape ladder minimum value; escape
ladder stays in `Clip3(1, 255)` range; cross-call accumulation;
fresh-walker initial `CurrentQIndex = 0`; out-of-range guards ⇒
`PartitionWalkOutOfRange`; arithmetic-decoder zero-byte sign-bit
observation. `decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

## Sources consulted (clean-room wall)

* AV1 Bitstream & Decoding Process Specification — AOMedia, copy at
  `docs/video/av1/av1-spec.txt` / `av1-spec.pdf`. Sections cited in
  module documentation:
  * Round 1: §4.10.5, §5.3.1, §5.3.2, §5.3.3, §6.2.1, §6.2.2,
    §6.2.3.
  * Round 2: §3 (constants — `SELECT_SCREEN_CONTENT_TOOLS`,
    `SELECT_INTEGER_MV`, `CP_UNSPECIFIED`, `TC_UNSPECIFIED`,
    `MC_UNSPECIFIED`, `CSP_UNKNOWN`, `CP_BT_709`, `TC_SRGB`,
    `MC_IDENTITY`), §4.10.2 (`f(n)`), §4.10.3 (`uvlc()`), §5.5.1
    (`sequence_header_obu`), §5.5.2 (`color_config`), §5.5.3
    (`timing_info`), §5.5.4 (`decoder_model_info`), §5.5.5
    (`operating_parameters_info`), §6.4.1 / §6.4.2 (semantics +
    conformance), §8.1 (`read_bit`).
  * Round 3: §3 (constants — `NUM_REF_FRAMES`, `PRIMARY_REF_NONE`),
    §5.9.1 (`frame_header_obu` framing), §5.9.2
    (`uncompressed_header` leading slice — `show_existing_frame`
    through `refresh_frame_flags`), §6.8.1 / §6.8.2 (semantics +
    conformance, including the `idLen <= 16` constraint on
    `display_frame_id`).
  * Round 4: §3 (constants — `SUPERRES_NUM`, `SUPERRES_DENOM_MIN`,
    `SUPERRES_DENOM_BITS`, `MI_SIZE`), §5.9.5 (`frame_size`),
    §5.9.6 (`render_size`), §5.9.7 (`frame_size_with_refs` — the
    `found_ref == 0` branch only), §5.9.8 (`superres_params`),
    §5.9.9 (`compute_image_size`), §6.8.4 / §6.8.5 / §6.8.6 /
    §6.8.7 / §6.8.8 (semantics).
  * Round 5: §3 (constants — `TOTAL_REFS_PER_FRAME`,
    `INTRA_FRAME`, `LAST_FRAME`, `GOLDEN_FRAME`, `BWDREF_FRAME`,
    `ALTREF_FRAME`, `ALTREF2_FRAME`), §4.10.6 (`su(n)`), §5.9.10
    (`read_interpolation_filter`), §5.9.11 (`loop_filter_params`),
    §5.9.12 (`quantization_params`), §5.9.13 (`read_delta_q`),
    §6.8.9 / §6.8.10 / §6.8.11 / §6.8.12 (semantics).
  * Round 6: §3 (constants — `MAX_TILE_WIDTH`, `MAX_TILE_AREA`,
    `MAX_TILE_ROWS`, `MAX_TILE_COLS`), §4.7 (`FloorLog2`),
    §4.10.7 (`ns(n)`), §5.9.2 (the `allow_intrabc` +
    `disable_frame_end_update_cdf` + `tile_info()` placement
    inside the `if (FrameIsIntra)` block / `reduced_still_picture
    header || disable_cdf_update` gate), §5.9.15 (`tile_info`),
    §5.9.16 (`tile_log2`), §6.8.14 (semantics + conformance —
    `TileCols <= MAX_TILE_COLS`, `TileRows <= MAX_TILE_ROWS`,
    `context_update_tile_id < TileCols * TileRows`).
  * Round 7: §3 (constants — `MAX_SEGMENTS`, `SEG_LVL_MAX`,
    `SEG_LVL_ALT_Q`, `SEG_LVL_ALT_LF_Y_V`, `SEG_LVL_ALT_LF_Y_H`,
    `SEG_LVL_ALT_LF_U`, `SEG_LVL_ALT_LF_V`, `SEG_LVL_REF_FRAME`,
    `SEG_LVL_SKIP`, `SEG_LVL_GLOBALMV`, `MAX_LOOP_FILTER`),
    §"Conventions" (`Clip3`), §5.9.2 (the `quantization_params()`
    + `segmentation_params()` placement after `tile_info()` in the
    `if (FrameIsIntra)` block), §5.9.14 (`segmentation_params`),
    §6.8.13 (semantics — `SegIdPreSkip` / `LastActiveSegId`
    derivations).
  * Round 8: §5.9.2 (the `delta_q_params()` + `delta_lf_params()`
    placement after `segmentation_params()` in the
    `if (FrameIsIntra)` block), §5.9.17 (`delta_q_params` — the
    `base_q_idx > 0` gate on `delta_q_present`, the `delta_q_present`
    gate on `delta_q_res`), §5.9.18 (`delta_lf_params` — the
    `delta_q_present` gate on the block, the `!allow_intrabc` gate on
    `delta_lf_present`, the `delta_lf_present` gate on `delta_lf_res`
    / `delta_lf_multi`), §6.8.15 (quantizer-index delta semantics),
    §6.8.16 (loop-filter delta semantics).
  * Round 9: §5.9.2 (the `CodedLossless` / `AllLossless` derivation
    lines and the `loop_filter_params()` placement after
    `delta_lf_params()` in the `if (FrameIsIntra)` block), §5.9.11
    (`loop_filter_params` — the `CodedLossless || allow_intrabc`
    short-circuit, the `NumPlanes > 1 && (level[0] || level[1])` gate on
    the chroma levels, the delta-update walk), §8.7 (`get_qindex`
    `ignoreDeltaQ` branch + the `SEG_LVL_ALT_Q` `Clip3(0, 255, ..)`
    clamp), §5.9.14's `seg_feature_active_idx`, §6.8.10 (loop-filter
    level / sharpness semantics).
  * Round 10: §5.9.2 (the `cdef_params()` placement after
    `loop_filter_params()` in the `if (FrameIsIntra)` block), §5.9.19
    (`cdef_params` — the `CodedLossless || allow_intrabc || !enable_cdef`
    short-circuit, the `CdefDamping = cdef_damping_minus_3 + 3`
    derivation, the `1 << cdef_bits` strength loop, the `NumPlanes > 1`
    gate on the chroma strengths, the secondary `== 3 ⇒ += 1`
    adjustment), §5.5.1 (`enable_cdef`), §6.4 (`enable_cdef`
    semantics), §6.10.14 (CDEF params semantics — `cdef_damping_minus_3`
    / `cdef_bits` / `cdef_*_pri_strength` / `cdef_*_sec_strength`).
  * Round 11: §3 (constant — `RESTORATION_TILESIZE_MAX = 256`),
    §5.9.2 (the `AllLossless = CodedLossless && (FrameWidth ==
    UpscaledWidth)` derivation line and the `lr_params()` placement
    after `cdef_params()` in the `if (FrameIsIntra)` block), §5.9.20
    (`lr_params` — the `AllLossless || allow_intrabc ||
    !enable_restoration` short-circuit, the per-plane `lr_type` `f(2)`
    loop with `Remap_Lr_Type[4] = { RESTORE_NONE, RESTORE_SWITCHABLE,
    RESTORE_WIENER, RESTORE_SGRPROJ }`, the `UsesLr` / `usesChromaLr`
    derivation, the 128×128-superblock-gated `lr_unit_shift`
    post-increment vs the non-128 `lr_unit_extra_shift` extension,
    the 4:2:0-chroma-gated `lr_uv_shift`, and the
    `LoopRestorationSize[plane]` derivation —
    `RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift)` for luma and
    `>> lr_uv_shift` for chroma), §5.5.1 (`enable_restoration`,
    `use_128x128_superblock`), §6.10.15 (Loop restoration params
    semantics — the `FrameRestorationType` symbol-value table).
  * Round 12: §3 (constant — `TX_MODES = 3`), §5.9.2 (the
    `read_tx_mode()` placement after `lr_params()` in the
    `if (FrameIsIntra)` block), §5.9.21 (`read_tx_mode` — the
    `CodedLossless == 1 ⇒ TxMode = ONLY_4X4` no-bits branch and the
    `tx_mode_select` `f(1)` ⇒ `TX_MODE_SELECT` / `TX_MODE_LARGEST`
    selection), §6.8.21 (TX mode semantics — the `TxMode` symbol-value
    table `ONLY_4X4 = 0`, `TX_MODE_LARGEST = 1`, `TX_MODE_SELECT = 2`).
  * Round 13: §3 (constants — `WARPEDMODEL_PREC_BITS = 16`, the `GM_*`
    bit / precision constants, `IDENTITY`/`TRANSLATION`/`ROTZOOM`/
    `AFFINE`, `LAST_FRAME`..`ALTREF_FRAME` reference-frame indices),
    §5.9.2 (the tail ordering after `read_tx_mode()`:
    `frame_reference_mode()` / `skip_mode_params()` / the
    `allow_warped_motion` guard / `reduced_tx_set` `f(1)` /
    `global_motion_params()` / `film_grain_params()`), §5.9.22
    (`skip_mode_params` — the `FrameIsIntra` ⇒ `skipModeAllowed = 0`
    branch), §5.9.23 (`frame_reference_mode` — `FrameIsIntra` ⇒
    `reference_select = 0`), §5.9.24 (`global_motion_params` — the
    identity initialiser, the `FrameIsIntra` early return, and the
    per-ref `is_global` / `is_rot_zoom` / `is_translation` type walk),
    §5.9.25 (`read_global_param` — `absBits` / `precBits` / `precDiff` /
    `round` / `sub` derivation), §5.9.26–§5.9.29
    (`decode_signed_subexp_with_ref` / `decode_unsigned_subexp_with_ref`
    / `decode_subexp` / `inverse_recenter`), §5.9.30 (`film_grain_params`
    — the `reset_grain_params()` short-circuits, the
    `apply_grain` / `grain_seed` / `update_grain` reads, the predicted
    `update_grain == 0` path, the Y / Cb / Cr scaling-point loops, the
    chroma-suppression branch, the AR-coefficient loops, and the
    chroma mult/offset triplets), §6.8.18 (global-motion-type symbol
    values), §6.8.20 (film-grain params semantics).
  * Round 15: §3 (constants — `EC_PROB_SHIFT = 6`, `EC_MIN_PROB = 4`),
    §4.7 (`FloorLog2`), §4.10.8 (`L(n)` descriptor), §4.10.10 (`NS(n)`
    descriptor), §5.9.28 (`decode_subexp_bool` — the bool variant of the
    subexponential code), §8.1 (`f(n)` parsing process — shared bit
    reader), §8.2.2 (`init_symbol` — `numBits` / `paddedBuf` /
    `SymbolValue` / `SymbolRange` / `SymbolMaxBits` init), §8.2.3
    (`read_bool` — the fixed `[1<<14, 1<<15, 0]` CDF), §8.2.4
    (`exit_symbol` — `trailingBitPosition` / `paddingEndPosition`
    derivation, the `Max(0, SymbolMaxBits)` advance, the
    `SymbolMaxBits >= -14` conformance gate), §8.2.5 (`read_literal`),
    §8.2.6 (`read_symbol` — the CDF-adaptive symbol search loop + the
    seven-step renormalisation), §8.3 (the adaptive-rate CDF update).
  * Round 16: §3 / §9.3 (constants — `INTRA_MODES = 13`,
    `INTRA_MODE_CONTEXTS = 5`, `PARTITION_CONTEXTS = 4`,
    `SKIP_CONTEXTS = 3`, `SEGMENT_ID_CONTEXTS = 3`, `MAX_SEGMENTS = 8`),
    §8.3.1 (the "set equal to a copy of `Default_*_Cdf`" init step that
    seeds the per-tile `Tile*Cdf` working set), §8.3.2 (selection
    paragraphs for `intra_frame_y_mode` — including the
    `Intra_Mode_Context[ INTRA_MODES ] = { 0, 1, 2, 3, 4, 4, 4, 4, 3, 0,
    1, 2, 0 }` array — `partition` with the `bsl` / `ctx = left * 2 +
    above` derivation, `skip` with the neighbour-`Skips[]` sum, and
    `segment_id` with the `prevUL / prevU / prevL` branch ladder), §9.4
    (default CDF table values for `Default_Intra_Frame_Y_Mode_Cdf`,
    `Default_Partition_W{8,16,32,64,128}_Cdf`, `Default_Skip_Cdf`,
    `Default_Segment_Id_Cdf`).
  * Round 17: §3 (constants — `MV_CONTEXTS = 2`,
    `MV_INTRABC_CONTEXT = 1`, `MV_JOINTS = 4`, `MV_CLASSES = 11`,
    `CLASS0_SIZE = 2`, `MV_OFFSET_BITS = 10`), §5.11.31 (`read_mv()` —
    the `MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0` derivation),
    §5.11.32 (`read_mv_component()` — the per-`comp` walk through
    `mv_sign` / `mv_class` / `mv_class0_bit` / `mv_class0_fr` /
    `mv_class0_hp` / `mv_bit` / `mv_fr` / `mv_hp`), §8.3.1 (the
    per-`i = 0..MV_CONTEXTS-1` / per-`comp = 0..1` "set equal to a copy
    of `Default_Mv_*_Cdf`" init step for the nine `Mv*Cdf` working
    arrays), §8.3.2 (the selection paragraphs — `mv_joint:
    TileMvJointCdf[ MvCtx ]`, `mv_sign: TileMvSignCdf[ MvCtx ][ comp ]`,
    `mv_class: TileMvClassCdf[ MvCtx ][ comp ]`, `mv_class0_bit:
    TileMvClass0BitCdf[ MvCtx ][ comp ]`, `mv_class0_fr:
    TileMvClass0FrCdf[ MvCtx ][ comp ][ mv_class0_bit ]`,
    `mv_class0_hp: TileMvClass0HpCdf[ MvCtx ][ comp ]`, `mv_fr:
    TileMvFrCdf[ MvCtx ][ comp ]`, `mv_hp: TileMvHpCdf[ MvCtx ][ comp ]`,
    `mv_bit: TileMvBitCdf[ MvCtx ][ comp ][ i ]`), §9.4 (default CDF
    table values for `Default_Mv_Joint_Cdf`, `Default_Mv_Sign_Cdf`,
    `Default_Mv_Class_Cdf`, `Default_Mv_Class0_Bit_Cdf`,
    `Default_Mv_Class0_Fr_Cdf`, `Default_Mv_Class0_Hp_Cdf`,
    `Default_Mv_Bit_Cdf`, `Default_Mv_Fr_Cdf`, `Default_Mv_Hp_Cdf`).
  * Round 18: §3 (constants — `NEW_MV_CONTEXTS = 6`,
    `ZERO_MV_CONTEXTS = 2`, `REF_MV_CONTEXTS = 6`, `DRL_MODE_CONTEXTS = 3`,
    `IS_INTER_CONTEXTS = 4`, `COMP_INTER_CONTEXTS = 5`,
    `SKIP_MODE_CONTEXTS = 3`, `REF_CONTEXTS = 3`, `FWD_REFS = 4`,
    `BWD_REFS = 3`, `SINGLE_REFS = 7`, `UNIDIR_COMP_REFS = 4`,
    `COMP_REF_TYPE_CONTEXTS = 5`, `COMPOUND_MODES = 8`,
    `COMPOUND_MODE_CONTEXTS = 8`, `COMP_NEWMV_CTXS = 5`), §8.3.1 (the
    "set to a copy of `Default_*_Cdf`" init step for `NewMvCdf`,
    `ZeroMvCdf`, `RefMvCdf`, `DrlModeCdf`, `IsInterCdf`, `CompModeCdf`,
    `SkipModeCdf`, `CompRefCdf`, `CompBwdRefCdf`, `SingleRefCdf`,
    `CompoundModeCdf`, `CompRefTypeCdf`, `UniCompRefCdf`), §8.3.2 (the
    selection paragraphs — `new_mv: TileNewMvCdf[ NewMvContext ]`,
    `zero_mv: TileZeroMvCdf[ ZeroMvContext ]`,
    `ref_mv: TileRefMvCdf[ RefMvContext ]`,
    `drl_mode: TileDrlModeCdf[ DrlCtxStack[ idx ] ]`, the
    `is_inter` context ladder over `(AvailU, AvailL) × (AboveIntra,
    LeftIntra)`, `comp_mode: TileCompModeCdf[ ctx ]`, `skip_mode:
    TileSkipModeCdf[ ctx ]` with the neighbour `SkipModes[]` sum,
    `comp_ref{,_p1,_p2}: TileCompRefCdf[ ctx ][ 0..2 ]` with
    `ctx = ref_count_ctx(last12Count, last3GoldCount)` / `..(lastCount,
    last2Count)` / `..(last3Count, goldCount)`, `comp_bwdref{,_p1}:
    TileCompBwdRefCdf[ ctx ][ 0..1 ]` with `ctx = ref_count_ctx(
    brfarf2Count, arfCount)` / `..(brfCount, arf2Count)`,
    `single_ref_p{1..6}: TileSingleRefCdf[ ctx ][ 0..5 ]` (the
    cross-referenced `single_ref_p2` ↔ `comp_bwdref` / `_p3` ↔
    `comp_ref` / `_p4` ↔ `comp_ref_p1` / `_p5` ↔ `comp_ref_p2` / `_p6` ↔
    `comp_bwdref_p1` mappings), `compound_mode: TileCompoundModeCdf[ ctx ]`
    with `ctx = Compound_Mode_Ctx_Map[ RefMvContext >> 1 ][ Min(
    NewMvContext, COMP_NEWMV_CTXS - 1) ]`, `comp_ref_type:
    TileCompRefTypeCdf[ ctx ]` (taken as a precomputed index — the
    nine-branch `aboveCompInter` / `leftCompInter` / `is_samedir_ref_pair`
    ladder is the tile walk's responsibility), `uni_comp_ref{,_p1,_p2}:
    TileUniCompRefCdf[ ctx ][ 0..2 ]`), and §9.4 (default CDF table
    values for `Default_New_Mv_Cdf`, `Default_Zero_Mv_Cdf`,
    `Default_Ref_Mv_Cdf`, `Default_Drl_Mode_Cdf`, `Default_Is_Inter_Cdf`,
    `Default_Comp_Mode_Cdf`, `Default_Skip_Mode_Cdf`,
    `Default_Comp_Ref_Cdf`, `Default_Comp_Bwd_Ref_Cdf`,
    `Default_Single_Ref_Cdf`, `Default_Compound_Mode_Cdf`,
    `Default_Comp_Ref_Type_Cdf`, `Default_Uni_Comp_Ref_Cdf`, plus the
    `Compound_Mode_Ctx_Map[ 3 ][ COMP_NEWMV_CTXS ]` table).
  * Round 20: §3 (constants — `TX_SIZE_CONTEXTS = 3`, `TX_SIZES = 5`,
    `MAX_TX_DEPTH = 2`, `TXFM_PARTITION_CONTEXTS = 21`), §5.11.15
    (`read_tx_size` — the `MiSize > BLOCK_4X4 && allowSelect &&
    TxMode == TX_MODE_SELECT` gate on the `tx_depth` `S()` read +
    the `Max_Tx_Depth[ MiSize ]` driver), §5.11.16 (`read_block_tx_size`
    — the `read_var_tx_size` call site that drives `txfm_split`),
    §8.3.1 (the "set equal to a copy of `Default_*_Cdf`" init step
    for `Tx8x8Cdf`, `Tx16x16Cdf`, `Tx32x32Cdf`, `Tx64x64Cdf`,
    `TxfmSplitCdf`), §8.3.2 (the four-way `tx_depth` selection by
    `maxTxDepth` over `TileTx{8x8,16x16,32x32,64x64}Cdf[ ctx ]`, the
    `ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)` formula,
    the `txfm_split: TileTxfmSplitCdf[ ctx ]` selection with
    `ctx = (txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6
    + above + left`), §9.4 (default CDF table values for
    `Default_Tx_8x8_Cdf`, `Default_Tx_16x16_Cdf`,
    `Default_Tx_32x32_Cdf`, `Default_Tx_64x64_Cdf`,
    `Default_Txfm_Split_Cdf`).
  * Round 21: §3 (constants — `TX_TYPES = 16`, `TX_TYPES_SET2 = 12`,
    `TX_TYPES_SET3 = 2`), §5.11.47 (`transform_type` — the `set > 0
    && qindex > 0 && is_inter` gate on the `inter_tx_type` `S()`
    read + the `Tx_Type_Inter_Inv_Set{1,2,3}` inversion tables),
    §5.11.48 (`get_tx_set` — the `is_inter == 1` branch:
    `txSzSqrUp > TX_32X32 ⇒ TX_SET_DCTONLY` /
    `reduced_tx_set || txSzSqrUp == TX_32X32 ⇒ TX_SET_INTER_3` /
    `txSzSqr == TX_16X16 ⇒ TX_SET_INTER_2` / else `TX_SET_INTER_1`),
    §6.10.19 (`set` tag enumeration — `TX_SET_DCTONLY = 0`,
    `TX_SET_INTRA_1 = 1`, `TX_SET_INTRA_2 = 2`, `TX_SET_INTER_1 = 1`,
    `TX_SET_INTER_2 = 2`, `TX_SET_INTER_3 = 3`), §8.3.1 (the "set
    equal to a copy of `Default_Inter_Tx_Type_Set{1,2,3}_Cdf`" init
    step for `InterTxTypeSet{1,2,3}Cdf`), §8.3.2 (the three-way
    `inter_tx_type` selection by `set` over
    `TileInterTxTypeSet1Cdf[ Tx_Size_Sqr[ txSz ] ]` /
    `TileInterTxTypeSet2Cdf` /
    `TileInterTxTypeSet3Cdf[ Tx_Size_Sqr[ txSz ] ]`), §9.4 (default
    CDF table values for `Default_Inter_Tx_Type_Set1_Cdf`,
    `Default_Inter_Tx_Type_Set2_Cdf`,
    `Default_Inter_Tx_Type_Set3_Cdf`).
  * Round 22: §3 (constants — `INTERP_FILTERS = 3`,
    `INTERP_FILTER_CONTEXTS = 16`), §8.3.1 (the "set equal to a copy
    of `Default_Interp_Filter_Cdf`" init step for `InterpFilterCdf`),
    §8.3.2 (the four-branch `interp_filter` ctx formula —
    `ctx = ((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4` base,
    `leftType = aboveType = 3` initialisers, the
    `RefFrames[..][0|1] == RefFrame[0]` neighbour-matching gate that
    promotes the `InterpFilters[..][dir]` entry, and the
    match / left-NONE / above-NONE / distinct branches that fold
    `leftType` / `aboveType` / `3` into the ctx total; the
    `interp_filter: TileInterpFilterCdf[ ctx ]` selection), §9.4
    (default CDF table values for `Default_Interp_Filter_Cdf`).
  * Round 23: §3 (constants — `MOTION_MODES = 3`), §6.10.26
    (`motion_mode` semantics — `SIMPLE = 0`, `OBMC = 1`,
    `LOCALWARP = 2`), §8.3.1 (the "set equal to a copy of
    `Default_Motion_Mode_Cdf`" init step for `MotionModeCdf`),
    §8.3.2 (the `motion_mode: TileMotionModeCdf[ MiSize ]` selection
    — a straight `0..BLOCK_SIZES` index with no neighbour-context
    arithmetic), §9.4 (default CDF table values for
    `Default_Motion_Mode_Cdf` including the §9.4 note that
    first-dimension indices `0..=2` and `16..=17` are never reached
    by the §5.11.x `read_motion_mode` selection but are still
    transcribed full-width).
  * Round 24: §3 (constants — `COMPOUND_TYPES = 2`,
    `COMP_GROUP_IDX_CONTEXTS = 6`, `COMPOUND_IDX_CONTEXTS = 6`),
    §6.10.24 (`comp_group_idx` / `compound_idx` / `compound_type`
    semantics), §8.3.1 (the "set equal to a copy of
    `Default_Comp_Group_Idx_Cdf` / `Default_Compound_Idx_Cdf` /
    `Default_Compound_Type_Cdf`" init steps), §8.3.2 (the
    `comp_group_idx: TileCompGroupIdxCdf[ ctx ]` paragraph with its
    `ctx = Min(5, ctx)` neighbour clamp, the
    `compound_idx: TileCompoundIdxCdf[ ctx ]` paragraph with its
    `get_relative_dist` fwd/bck seed, and the
    `compound_type: TileCompoundTypeCdf[ MiSize ]` straight index),
    §9.4 (default CDF table values for the three tables including the
    §9.4 note that `Default_Compound_Type_Cdf` first-dimension indices
    `0..=2`, `10..=17` and `20..=21` are never used but are still
    transcribed full-width).
  * **Inter-frame intra-mode CDFs** (round 134): §3 (`BLOCK_SIZE_GROUPS`,
    `UV_INTRA_MODES_CFL_NOT_ALLOWED`, `UV_INTRA_MODES_CFL_ALLOWED`
    constant definitions), §8.3.1 (the "set equal to a copy of
    `Default_Y_Mode_Cdf` / `Default_Uv_Mode_Cfl_Not_Allowed_Cdf` /
    `Default_Uv_Mode_Cfl_Allowed_Cdf`" init steps for `YModeCdf` /
    `UVModeCflNotAllowedCdf` / `UVModeCflAllowedCdf`), §8.3.2 (the
    `y_mode: TileYModeCdf[ Size_Group[ MiSize ] ]` paragraph and the
    `uv_mode` paragraph selecting the cfl-allowed / cfl-not-allowed
    variant by the `Lossless` / `get_plane_residual_size` /
    `Max(Block_Width, Block_Height) <= 32` tests, then indexing by
    `YMode`), §8.3.2 `Size_Group[ BLOCK_SIZES ]` table, §9.4 (default
    CDF table values for the three tables).
  * **Angle-delta CDF** (round 135): §3 (`DIRECTIONAL_MODES`,
    `MAX_ANGLE_DELTA`, `V_PRED`), §8.3.1 (the "`AngleDeltaCdf` is set
    to a copy of `Default_Angle_Delta_Cdf`" init step), §8.3.2 (the
    `TileAngleDeltaCdf[ YMode - V_PRED ]` / `[ UVMode - V_PRED ]`
    selections for `angle_delta_y` / `angle_delta_uv`), §9.4 (default
    CDF table values for `Default_Angle_Delta_Cdf`).
  * **Coefficient-token entry CDFs** (round 136): §3 (`PLANE_TYPES`,
    `COEFF_CDF_Q_CTXS`, `TXB_SKIP_CONTEXTS`, `EOB_COEF_CONTEXTS`,
    `DC_SIGN_CONTEXTS` constant definitions), §8.3.1 `init_coeff_cdfs`
    (the `base_q_idx` → `idx` derivation and the "set to a copy of
    `Default_Txb_Skip_Cdf[ idx ]` / `Default_Eob_Pt_*_Cdf[ idx ]` /
    `Default_Eob_Extra_Cdf[ idx ]` / `Default_Dc_Sign_Cdf[ idx ]`"
    reset steps), §9.4 (default CDF table values for
    `Default_Txb_Skip_Cdf`, `Default_Eob_Pt_{16,32,64,128,256,512,
    1024}_Cdf`, `Default_Eob_Extra_Cdf` and `Default_Dc_Sign_Cdf`,
    the last in the `128 * N` fixed-point form).
  * **Intra-frame transform-type CDFs** (round 137): §3 (constants —
    `TX_SET_INTRA_1 = 1`, `TX_SET_INTRA_2 = 2`,
    `TX_TYPES_INTRA_SET1 = 7`, `TX_TYPES_INTRA_SET2 = 5`,
    `INTRA_TX_TYPE_SET1_SIZES = 2`, `INTRA_TX_TYPE_SET2_SIZES = 3`),
    §5.11.48 `get_tx_set()` (the `is_inter == 0` branch routing
    `txSzSqrUp >= TX_32X32` → `TX_SET_DCTONLY`, `reduced_tx_set ||
    txSzSqr == TX_16X16` → `TX_SET_INTRA_2`, else → `TX_SET_INTRA_1`),
    §6.10.19 (the `Tx_Type_Intra_Inv_Set1` 7-entry / `Tx_Type_Intra_Inv_Set2`
    5-entry transform-type enumeration), §8.3.1 (the "set equal to a
    copy of `Default_Intra_Tx_Type_Set1_Cdf` / `Default_Intra_Tx_Type_Set2_Cdf`"
    init steps for `IntraTxTypeSet1Cdf` / `IntraTxTypeSet2Cdf`), §8.3.2
    (the `intra_tx_type: TileIntraTxTypeSet{1,2}Cdf[ Tx_Size_Sqr[ txSz ]
    ][ intraDir ]` two-way switch and the `intraDir` derivation —
    `Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]` when
    `use_filter_intra == 1`, else `YMode`), §8.3.2
    `Filter_Intra_Mode_To_Intra_Dir[ INTRA_FILTER_MODES ]` table,
    §9.4 (default CDF table values for the two tables).
  * **`coeff_base_eob` CDF** (round 138): §3 (`SIG_COEF_CONTEXTS_EOB = 4`
    constant definition), §5.11 `coeff_base_eob` semantics (the
    "base level is `coeff_base_eob + 1`; only base levels 1, 2, or 3
    can be coded" note constraining the symbol set to three values),
    §8.3.1 `init_coeff_cdfs` (the "`CoeffBaseEobCdf` is set to a
    copy of `Default_Coeff_Base_Eob_Cdf[ idx ]`" reset step), §8.3.2
    (the `coeff_base_eob: TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][
    ctx ]` selection plus the deferred-to-tile-walk
    `ctx = get_coeff_base_ctx(...) - SIG_COEF_CONTEXTS +
    SIG_COEF_CONTEXTS_EOB` derivation), §9.4 (default CDF table
    values for `Default_Coeff_Base_Eob_Cdf`).
  * **`coeff_base` CDF** (round 139): §3 (`SIG_COEF_CONTEXTS = 42`
    constant definition; the `SIG_COEF_CONTEXTS_2D = 26` partition
    tag that splits the 2D-scan prefix from the 1D horizontal- /
    vertical-only tails), §5.11 `coeff_base` semantics (the "level
    is `coeff_base`" assignment for non-EOB coefficients),
    §8.3.1 `init_coeff_cdfs` (the "`CoeffBaseCdf` is set to a copy
    of `Default_Coeff_Base_Cdf[ idx ]`" reset step), §8.3.2 (the
    `coeff_base: TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]`
    selection plus the deferred-to-tile-walk
    `ctx = get_coeff_base_ctx(...)` derivation), §9.4 (default CDF
    table values for `Default_Coeff_Base_Cdf`).
  * **`coeff_br` CDF** (round 140): §3 (`LEVEL_CONTEXTS = 21` and
    `BR_CDF_SIZE = 4` constant definitions), §5.11.39 `coeff_br`
    semantics (the `for idx = 0; idx < COEFF_BASE_RANGE /
    (BR_CDF_SIZE - 1); idx++` per-coefficient stacking loop with
    the `coeff_br < (BR_CDF_SIZE - 1)` early-break),
    §8.3.1 `init_coeff_cdfs` (the "`CoeffBrCdf` is set to a copy of
    `Default_Coeff_Br_Cdf[ idx ]`" reset step), §8.3.2 (the
    `coeff_br: TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][
    ctx ]` selection with the `TX_32X32 = 3` clamp, plus the
    deferred-to-tile-walk `ctx = get_br_ctx(...)` derivation),
    §9.4 (default CDF table values for `Default_Coeff_Br_Cdf`).
  * **`get_coeff_base_ctx` / `get_br_ctx` helpers** (round 141):
    §3 (`TX_SIZES_ALL = 19`, `SIG_COEF_CONTEXTS_2D = 26`,
    `SIG_REF_DIFF_OFFSET_NUM = 5`, `NUM_BASE_LEVELS = 2`,
    `COEFF_BASE_RANGE = 12`, `TX_CLASS_{2D, HORIZ, VERT}` tag
    enumeration with values `{ 0, 1, 2 }`, and the `V_DCT = 10`,
    `H_DCT = 11`, `V_ADST = 12`, `H_ADST = 13`, `V_FLIPADST = 14`,
    `H_FLIPADST = 15` transform-type enumeration used by
    `get_tx_class`), §8.3.2 (the `get_coeff_base_ctx()` and
    `get_br_ctx()` function bodies — `isEob` branch with the
    `(height << bwl) / 8` / `/ 4` boundaries, the
    `Sig_Ref_Diff_Offset` neighbour scan with the
    `Min(Abs(Quant[..]), 3)` per-neighbour clamp, the 2D
    `row == 0 && col == 0 -> 0` short-circuit, the
    `Coeff_Base_Ctx_Offset[txSz][Min(row, 4)][Min(col, 4)]` 2D
    offset, the `Coeff_Base_Pos_Ctx_Offset[Min(idx, 2)]` 1D
    branch, the `Mag_Ref_Offset_With_Tx_Class` three-neighbour
    scan with the `Min(Quant[..], COEFF_BASE_RANGE + NUM_BASE_LEVELS
    + 1)` clamp, the `Min((mag + 1) >> 1, 6)` magnitude bucket,
    the `pos == 0` short-circuit, and the per-class `+7` /
    `+14` offsets — plus the `get_tx_class()` function body and
    the `coeff_base_eob` `ctx = get_coeff_base_ctx(.., 1) -
    SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB` reduction),
    §"Additional tables" (`Tx_Width[TX_SIZES_ALL]`,
    `Tx_Height[TX_SIZES_ALL]`, `Tx_Width_Log2[TX_SIZES_ALL]`,
    `Adjusted_Tx_Size[TX_SIZES_ALL]`,
    `Sig_Ref_Diff_Offset[3][SIG_REF_DIFF_OFFSET_NUM][2]`,
    `Coeff_Base_Ctx_Offset[TX_SIZES_ALL][5][5]`,
    `Coeff_Base_Pos_Ctx_Offset[3]`,
    `Mag_Ref_Offset_With_Tx_Class[3][3][2]`).
  * **Inter-intra CDFs** (round 143): §3 (`INTERINTRA_MODES = 4`),
    §5.11.28 (`read_interintra_mode` — the `interintra` /
    `interintra_mode` / `wedge_interintra` reads, including the
    `BLOCK_8X8 <= MiSize <= BLOCK_32X32` syntax gate),
    §6.10.27 (`II_DC_PRED` / `II_V_PRED` / `II_H_PRED` /
    `II_SMOOTH_PRED` enumeration; `wedge_interintra` semantics),
    §8.3.1 (`init_non_coeff_cdfs` — `InterIntraCdf` /
    `InterIntraModeCdf` / `WedgeInterIntraCdf` "is set equal to a copy
    of `Default_*`"), §8.3.2 (`interintra` /
    `interintra_mode` / `wedge_interintra` paragraphs — the
    `ctx = Size_Group[ MiSize ] - 1` mapping for the first two and
    the `TileWedgeInterIntraCdf[ MiSize ]` straight index for the
    third), §8.3.2 `Size_Group[ BLOCK_SIZES ]` table (from round
    134), §9.4 listings (`Default_Inter_Intra_Cdf`,
    `Default_Inter_Intra_Mode_Cdf`, `Default_Wedge_Inter_Intra_Cdf`
    on pp.434–436 of the spec PDF — including the latter's note that
    only first-dimension indices 3..=9 are used).
  * **`compute_tx_type` derivation** (round 142): §3 / §6.10.16 (the
    `TX_4X4 = 0`, `TX_8X8 = 1`, `TX_16X16 = 2`, `TX_32X32 = 3`,
    `TX_64X64 = 4` `TxSize` ordinals — previously used only as
    locally-scoped `const TX_*` shadows inside the §5.11.48
    helpers — and the `TX_SET_TYPES_INTRA = 3` /
    `TX_SET_TYPES_INTER = 4` row-count constants for the
    `Tx_Type_In_Set_*` tables), §6.10.19 (the `DCT_DCT = 0` through
    `H_FLIPADST = 15` 16-entry transform-type enumeration —
    previously the `V_DCT..H_FLIPADST` tail only), §5.11.40
    (`compute_tx_type()` function body — the `Lossless ||
    Tx_Size_Sqr_Up[ txSz ] > TX_32X32 -> DCT_DCT` short-circuit, the
    `plane == 0 -> TxTypes[ blockY ][ blockX ]` luma branch, the
    `is_inter` chroma `Max(MiRow, blockY << subsampling_y)` /
    `Max(MiCol, blockX << subsampling_x)` lift into the
    `TxTypes[..]` cache, the `is_tx_type_in_set` admission filter
    with the `!is_tx_type_in_set -> DCT_DCT` fallback, the intra
    chroma `Mode_To_Txfm[ UVMode ]` path with the same filter — and
    the `Tx_Type_In_Set_Intra[ 3 ][ TX_TYPES ]` /
    `Tx_Type_In_Set_Inter[ 4 ][ TX_TYPES ]` admission-flag tables
    transcribed verbatim from the spec listing), §"Additional
    tables" (`Tx_Size_Sqr_Up[ TX_SIZES_ALL ]` — `t -> Max(w, h)`-
    sided square — and `Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]`
    — chroma-mode default-tx-type table).
  * **`intra_segment_id` syntax element** (round 160): §3 (the
    `MAX_SEGMENTS = 8` upper bound that sizes the §6.8.2
    `LosslessArray[]` table the §5.11.8 footer indexes), §5.11.7
    (the §5.11.7 `intra_frame_mode_info` caller context — the two
    `SegIdPreSkip` arms that each call `intra_segment_id()`),
    §5.11.8 (the function body itself — the
    `segmentation_enabled ⇒ read_segment_id() : segment_id = 0`
    dispatch and the `Lossless = LosslessArray[ segment_id ]`
    footer), §5.11.9 (the §5.11.9 `read_segment_id` body the
    `segmentation_enabled = true` arm delegates to, including the
    §5.11.5 grid-fill footer line `SegmentIds[ r + y ][ c + x ] =
    segment_id` — accepted as the r159 implementation), and §6.8.2
    (the `LosslessArray[ segmentId ] = qindex == 0 && DeltaQYDc ==
    0 && DeltaQUAc == 0 && DeltaQUDc == 0 && DeltaQVAc == 0 &&
    DeltaQVDc == 0` per-segment derivation that produces the
    eight-entry table the walker indexes by `segment_id` — accepted
    as a `&[bool; MAX_SEGMENTS]` parameter so the walker stays
    segmentation-state-free).
  * **`read_segment_id` syntax element** (round 159): §3 (the
    `MAX_SEGMENTS = 8` upper bound capping the §5.11.9 alphabet
    and the `SEGMENT_ID_CONTEXTS = 3` constant bounding the §8.3.2
    ctx derivation), §5.9.14 (the `LastActiveSegId` trailing
    derivation that supplies `max = LastActiveSegId + 1` to the
    §5.11.9 `neg_deinterleave` call — accepted as a `u8` parameter
    so the walker stays segmentation-state-free), §5.11.5 (the
    per-block grid-fill footer line `SegmentIds[ r + y ][ c + x ] =
    segment_id` the walker stamps over the `bw4 * bh4` footprint),
    §5.11.8 / §5.11.19 (the two caller contexts: intra-frame /
    inter-frame `*_segment_id` syntax, both of which route through
    `read_segment_id` when `segmentation_enabled`), §5.11.9 (the
    function body itself — the three neighbour reads `prevUL` /
    `prevU` / `prevL`, the four-arm `pred` cascade `prevU == -1 ⇒
    prevL / 0`; `prevL == -1 ⇒ prevU`; `prevUL == prevU ⇒ prevU`;
    else `prevL`, the `skip ⇒ segment_id = pred` short-circuit, the
    `S()` read against `TileSegmentIdCdf[ ctx ]` plus
    `neg_deinterleave( diff, pred, max )` reconstruction), and
    §8.3.2 (the `segment_id` ctx formula `prevUL < 0 ⇒ 0; (prevUL
    == prevU && prevUL == prevL) ⇒ 2; (prevUL == prevU || prevUL
    == prevL || prevU == prevL) ⇒ 1; else 0` — already lifted as
    `segment_id_ctx`).
  * **`read_is_inter` syntax element** (round 158): §3 (the
    `IS_INTER_CONTEXTS = 4` constant bounding the §8.3.2 ctx
    derivation, the implicit `INTRA_FRAME = 0` ordinal under which
    the §5.11.20 Arm 2 comparison `FeatureData[..] != INTRA_FRAME`
    encodes "the segmentation override selected an inter reference",
    the `SEG_LVL_REF_FRAME = 5` / `SEG_LVL_GLOBALMV = 7` segment-
    feature indices that gate Arms 2 and 3), §5.11.5 (the per-block
    grid-fill footer line `IsInters[ r + y ][ c + x ] = is_inter`
    that the walker stamps over the `bw4 * bh4` footprint after the
    arm dispatch), §5.11.18 (the §5.11.20 caller context: the
    `LeftRefFrame[0] = AvailL ? RefFrames[..][0] : INTRA_FRAME` /
    `LeftIntra = LeftRefFrame[0] <= INTRA_FRAME` neighbour-
    derivation that maps onto an `Option<bool>` to `is_inter_ctx`,
    and the `read_is_inter()` call-site placement immediately after
    `read_delta_lf()` and before the `is_inter ?
    inter_block_mode_info() : intra_block_mode_info()` dispatch),
    §5.11.20 (the four-arm dispatch body itself —
    `skip_mode` ⇒ 1; `seg_feature_active(SEG_LVL_REF_FRAME)` ⇒
    `FeatureData != INTRA_FRAME`; `seg_feature_active(SEG_LVL_GLOBALMV)`
    ⇒ 1; else `S()` — and the §8.3.2 ctx formula at av1-spec p.365:
    `AvailU && AvailL ⇒ (LeftIntra && AboveIntra) ? 3 : LeftIntra
    || AboveIntra`; `AvailU ^ AvailL ⇒ 2 * (AvailU ? AboveIntra :
    LeftIntra)`; else 0).
  * **`read_cdef` syntax element + `clear_cdef` reset** (round 157):
    §3 (the `BLOCK_64X64 = 12` ordinal that anchors the §5.11.55
    `cdefSize4 = Num_4x4_Blocks_Wide[ BLOCK_64X64 ] = 16` stride; the
    `Num_4x4_Blocks_Wide[ ]` / `Num_4x4_Blocks_High[ ]` tables used
    by the §5.11.56 grid-fill loop, transcribed in round 148 from
    §9.3 p.400), §5.9.19 (the `cdef_bits` `f(2)` upper bound — so
    `cdef_bits ∈ 0..=3` and the decoded `L(cdef_bits)` value fits in
    `i8`; the `CodedLossless || allow_intrabc || !enable_cdef ⇒
    cdef_bits = 0` short-circuit that pairs with §5.11.56's
    short-circuit set), §5.11.2 (the per-superblock `clear_cdef( r,
    c )` call site at the top of the tile-walk inner loop —
    immediately before `decode_partition`), §5.11.55 (the
    `clear_cdef()` function body — the unconditional `cdef_idx[ r ][
    c ] = -1` stamp plus the `use_128x128_superblock` three-extra-
    anchor stamps), §5.11.56 (the `read_cdef()` function body — the
    four-way short-circuit `skip || CodedLossless || !enable_cdef ||
    allow_intrabc`, the `cdefMask4 = ~(cdefSize4 - 1)` anchor mask,
    the `cdef_idx[ r ][ c ] == -1` first-leaf gate that gates the
    `L(cdef_bits)` literal read, and the `cdefSize4`-strided
    grid-fill loop over the leaf's `(w4, h4)` footprint), §6.10.40
    (the `cdef_idx` semantics — "A value of -1 means that CDEF is
    disabled for that block").
  * **`read_delta_lf` syntax element** (round 156): §3 (the
    `DELTA_LF_SMALL = 3` escape sentinel against which `delta_lf_abs`
    is compared to trigger the `delta_lf_rem_bits` /
    `delta_lf_abs_bits` ladder; the `FRAME_LF_COUNT = 4` upper bound
    on the per-superblock iteration loop and on the size of the
    `DeltaLF[ ]` accumulator; the `MAX_LOOP_FILTER = 63` Clip3
    bound — `cdf::MAX_LOOP_FILTER` is an `i32` twin of the
    pre-existing §5.9.11 `uncompressed_header_tail::MAX_LOOP_FILTER`
    `i16` so the signed `DeltaLF[ i ] + (reducedDeltaLfLevel <<
    delta_lf_res)` arithmetic stays in a single integer type), §5.11.2
    (`for ( i = 0; i < FRAME_LF_COUNT; i++ ) DeltaLF[ i ] = 0` — the
    tile-entry reset honoured by `reset_current_delta_lf`), §5.11.13
    (`read_delta_lf()` function body — the `sbSize`-derived skip
    short-circuit identical to §5.11.12, the
    `ReadDeltas && delta_lf_present` outer gate, the
    `frameLfCount = delta_lf_multi ?
        ((NumPlanes > 1) ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2) : 1`
    derivation, the per-iteration `S()` against `TileDeltaLFCdf` or
    `TileDeltaLFMultiCdf[ i ]`, the `DELTA_LF_SMALL` escape ladder,
    the `delta_lf_sign_bit` sign read, and the `Clip3(-MAX_LOOP_FILTER,
    MAX_LOOP_FILTER, ...)` update), §6.10.4 (the `ReadDeltas`
    derivation as `delta_q_present` AND first-block-of-superblock,
    re-used identically for the §5.11.13 outer gate — accepted from
    the caller per the §5.11.12 convention), §8.3.1 (the
    `init_non_coeff_cdfs` steps "`DeltaLFCdf` is set to a copy of
    `Default_Delta_Lf_Cdf`" and "`DeltaLFMultiCdf[ i ]` is set to a
    copy of `Default_Delta_Lf_Cdf` for `i = 0..FRAME_LF_COUNT-1`"),
    §8.3.2 (the `delta_lf_abs` cdf-selection paragraph — `cdf` is
    `TileDeltaLFCdf` when `delta_lf_multi == 0` and
    `TileDeltaLFMultiCdf[ i ]` when `delta_lf_multi == 1`; no
    context index in either case), §9.4 (default CDF table values
    for `Default_Delta_Lf_Cdf[ DELTA_LF_SMALL + 2 ]` on av1-spec
    p.431).
  * **`read_delta_qindex` syntax element** (round 155): §3 (the
    `DELTA_Q_SMALL = 3` constant — the §5.11.12 escape sentinel
    against which `delta_q_abs` is compared to trigger the
    `delta_q_rem_bits` / `delta_q_abs_bits` ladder), §5.11.12
    (`read_delta_qindex()` function body — the `sbSize =
    use_128x128_superblock ? BLOCK_128X128 : BLOCK_64X64` derivation,
    the `MiSize == sbSize && skip` superblock-skip short-circuit,
    the `ReadDeltas` outer gate, the `delta_q_abs` `S()` read with
    the `DELTA_Q_SMALL` escape ladder
    `delta_q_rem_bits = L(3); delta_q_rem_bits++;
    delta_q_abs_bits = L(delta_q_rem_bits); delta_q_abs =
    delta_q_abs_bits + (1 << delta_q_rem_bits) + 1`, the
    `delta_q_sign_bit = L(1)` sign read, the
    `reducedDeltaQIndex = delta_q_sign_bit ? -delta_q_abs :
    delta_q_abs` derivation, and the
    `CurrentQIndex = Clip3(1, 255, CurrentQIndex +
    (reducedDeltaQIndex << delta_q_res))` update), §6.10.4 (the
    `ReadDeltas` derivation as `delta_q_present` AND
    first-block-of-superblock — accepted from the caller as a
    pre-computed boolean rather than re-derived inside the walker),
    §8.3.1 (the `init_non_coeff_cdfs` step "`DeltaQCdf` is set to
    a copy of `Default_Delta_Q_Cdf`"), §8.3.2 (the `delta_q_abs:
    the cdf is given by TileDeltaQCdf` paragraph — single CDF row
    with no context index), §9.4 (default CDF table values for
    `Default_Delta_Q_Cdf[ DELTA_Q_SMALL + 2 ]` on av1-spec p.431).
* Fixtures under `docs/video/av1/fixtures/` (bitstreams + trace
  files emitted by an AV1_TRACE-patched FFmpeg + libdav1d host;
  treated as opaque ground-truth, no source consulted).

No external library source — libaom, dav1d, libgav1, rav1e, SVT-AV1,
FFmpeg AV1 — was consulted. No third-party crate that wraps or
implements the same format was consulted. No web search was
performed.

## License

MIT. See `LICENSE`.
