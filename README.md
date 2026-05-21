# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec.

## Status — 2026-05-21

**Clean-room rebuild, round 3.** The crate's prior implementation was
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
`FRAME_HEADER idx=0` line in the same trace (for the 12 columns
this parser owns: `show_existing` / `frame_to_show` / `frame_type` /
`show_frame` / `showable` / `error_resilient` /
`disable_cdf_update` / `allow_screen_content` / `force_integer_mv` /
`order_hint` / `primary_ref_frame` / `refresh_flags`).

Frame decoding past `refresh_frame_flags` (`frame_size()` /
`render_size()` / `tile_info()`, motion vectors, transform /
quantisation, in-loop filters, film grain) is **not yet
implemented**. `decode_av1` and `encode_av1` still return
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
* Fixtures under `docs/video/av1/fixtures/` (bitstreams + trace
  files emitted by an AV1_TRACE-patched FFmpeg + libdav1d host;
  treated as opaque ground-truth, no source consulted).

No external library source — libaom, dav1d, libgav1, rav1e, SVT-AV1,
FFmpeg AV1 — was consulted. No third-party crate that wraps or
implements the same format was consulted. No web search was
performed.

## License

MIT. See `LICENSE`.
