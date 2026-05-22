# Changelog

All notable changes to `oxideav-av1` are recorded here.

## [Unreleased]

### Added

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
