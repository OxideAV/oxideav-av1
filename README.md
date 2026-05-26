# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec.

## Status — 2026-05-26

**Clean-room rebuild, round 22.** The crate's prior implementation was
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
The remaining §9.4 tables (inter-intra (`Default_Interintra_Cdf`),
and the other §8.3.2 selections (`split_or_horz` / `split_or_vert` /
…) are a mechanical followup against the same `TileCdfContext`
shape. `decode_av1` and `encode_av1` still return
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
* Fixtures under `docs/video/av1/fixtures/` (bitstreams + trace
  files emitted by an AV1_TRACE-patched FFmpeg + libdav1d host;
  treated as opaque ground-truth, no source consulted).

No external library source — libaom, dav1d, libgav1, rav1e, SVT-AV1,
FFmpeg AV1 — was consulted. No third-party crate that wraps or
implements the same format was consulted. No web search was
performed.

## License

MIT. See `LICENSE`.
