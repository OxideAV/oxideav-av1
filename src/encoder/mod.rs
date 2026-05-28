//! Encoder side of the crate.
//!
//! Arc 1 (round 206) landed the bit-output plumbing. Arc 2 (round
//! 207) lands the `frame_header_obu()` writer on top.
//!
//! Layers:
//!
//!   * [`bitwriter::BitWriter`] — MSB-first bit-output buffer, the
//!     inverse of [`crate::bitreader::BitReader`] (§8.1 `read_bit`),
//!     plus `write_leb128()` (§4.10.5), `write_uvlc()` (§4.10.3),
//!     `write_su(n)` (§4.10.6), and `write_ns(n)` (§4.10.7) — the
//!     full descriptor-inverse set the §5.5 / §5.9 writers need.
//!
//!   * [`obu`] — Open Bitstream Unit framer per §5.3. Writes the
//!     §5.3.2 one-byte `obu_header`, the optional §5.3.3
//!     `obu_extension_header`, and the optional `leb128()`
//!     `obu_size` size field for the §5.2 low-overhead bytestream
//!     format. Concatenation of multiple OBUs into a temporal unit
//!     is byte-aligned and simply uses [`ObuWriter::write`] N times.
//!
//!   * [`sequence_obu`] — `sequence_header_obu()` writer per §5.5.1
//!     (with §5.5.2 `color_config`, §5.5.3 `timing_info`, §5.5.4
//!     `decoder_model_info`, §5.5.5 `operating_parameters_info`).
//!     The inverse of [`crate::sequence_header::parse_sequence_header`].
//!     Reuses the same [`crate::sequence_header::SequenceHeader`]
//!     struct as the source-of-truth descriptor, so a written
//!     payload immediately round-trips through the parser.
//!
//!   * [`frame_obu`] — `frame_header_obu()` writer per §5.9.1 /
//!     §5.9.2 plus every sub-procedure §5.9.2 calls into. The
//!     inverse of [`crate::frame_header::parse_frame_header`] on the
//!     intra / show-existing-frame / reduced-still paths and on the
//!     inter shared tail above `disable_frame_end_update_cdf`.
//!     Reuses the parser's [`crate::frame_header::FrameHeader`] as
//!     source-of-truth descriptor.
//!
//!   * [`ivf`] — IVF v0 container writer (32-byte file header + 12-
//!     byte per-frame header) for shipping the encoded OBU temporal
//!     units into a playable file. IVF is a trivial public file
//!     format developed for VP8 testing; the byte layout used here
//!     matches the `.ivf` fixtures already in `docs/video/av1/
//!     fixtures/`.
//!
//!   * [`temporal_unit`] — arc 3 (round 208) glue. Wraps the per-OBU
//!     body writers above with the §5.3.4 `trailing_bits()` trailer
//!     and the §5.3.1 `obu_size` size field, then aggregates a
//!     sequence of OBUs into a §7.5 temporal unit (TD prefix +
//!     optional SH + the frame OBUs). The product is a complete
//!     byte-aligned bytestream a downstream parser walks back via
//!     [`crate::obu::ObuIter`].
//!
//! Next arc: §5.11 / §5.12 `tile_group_obu` writer for actual tile
//! encode (entropy coder + coefficient encode + per-block syntax);
//! §5.9.7 `frame_size_with_refs()` inverse for inter-frame size
//! plumbing; §5.9.24 `read_global_param` signed-subexp inverse for
//! non-IDENTITY refs.

pub mod bitwriter;
pub mod frame_obu;
pub mod ivf;
pub mod obu;
pub mod sequence_obu;
pub mod temporal_unit;

pub use bitwriter::BitWriter;
pub use frame_obu::write_frame_header_obu;
pub use ivf::IvfWriter;
pub use obu::{
    build_temporal_unit, obu_type_takes_trailing_bits, write_obu_with_size, write_temporal_unit,
    ObuExtensionHeader, ObuFrame, ObuHeader, ObuWriter,
};
pub use sequence_obu::write_sequence_header_obu;
pub use temporal_unit::{encode_sequence_header_obu, encode_temporal_unit, TemporalUnitPlan};
