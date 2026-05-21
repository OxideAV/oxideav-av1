# Changelog

All notable changes to `oxideav-av1` are recorded here.

## [Unreleased]

### Added

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
