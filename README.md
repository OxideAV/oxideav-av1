# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec.

## Status — 2026-05-20

**Clean-room rebuild, round 1.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core decoder modules could not be defended against the "no external
library source as reference" rule that governs every crate in this
workspace.

Round 1 of the rebuild lands the OBU bytestream walker described in
§5.3 of the AV1 Bitstream & Decoding Process Specification:

* `leb128()` parsing per §4.10.5 (with the conformance cap to
  `(1 << 32) - 1` and the 8-byte length bound enforced).
* OBU header decode per §5.3.2: `obu_forbidden_bit`, `obu_type` (4
  bits), `obu_extension_flag`, `obu_has_size_field`, plus the
  ignored reserved bit.
* Extension header decode per §5.3.3: `temporal_id` (3 bits),
  `spatial_id` (2 bits); inferred to 0 when the extension flag is
  clear (§6.2.3).
* `obu_size` payload framing per §6.2.1.
* Iterator that walks a concatenation of OBUs in the §5.2
  low-overhead bitstream format.

Frame decoding (`sequence_header_obu`, `frame_header_obu`, tile
parsing, motion vectors, transform/quantisation, in-loop filters,
film grain) is **not yet implemented**. `decode_av1` and `encode_av1`
still return `Error::NotImplemented`.

## Sources consulted (clean-room wall)

* AV1 Bitstream & Decoding Process Specification — AOMedia, copy at
  `docs/video/av1/av1-spec.txt` / `av1-spec.pdf`. Sections cited in
  module documentation: §4.10.5, §5.3.1, §5.3.2, §5.3.3, §6.2.1,
  §6.2.2, §6.2.3.

No external library source — libaom, dav1d, rav1e, libgav1, SVT-AV1,
FFmpeg — was consulted. No third-party crate that wraps or implements
the same format was consulted. No web search was performed.

## License

MIT. See `LICENSE`.
