# Changelog

All notable changes to `oxideav-av1` are recorded here.

## [Unreleased]

### Added

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
