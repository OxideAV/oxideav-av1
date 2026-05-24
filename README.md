# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec.

## Status — 2026-05-24

**Clean-room rebuild, round 11.** The crate's prior implementation was
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

Frame decoding past `lr_params()` (the remaining uncompressed-header
tail — `read_tx_mode()`, `frame_reference_mode()` — plus tile-content
decode — motion vectors, transform / quantisation, in-loop filters,
film grain) is **not yet implemented**. `decode_av1` and `encode_av1`
still return `Error::NotImplemented`.

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
* Fixtures under `docs/video/av1/fixtures/` (bitstreams + trace
  files emitted by an AV1_TRACE-patched FFmpeg + libdav1d host;
  treated as opaque ground-truth, no source consulted).

No external library source — libaom, dav1d, libgav1, rav1e, SVT-AV1,
FFmpeg AV1 — was consulted. No third-party crate that wraps or
implements the same format was consulted. No web search was
performed.

## License

MIT. See `LICENSE`.
