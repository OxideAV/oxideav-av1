//! Decoder side of the crate.
//!
//! One decode surface remains as of r428: the **spec-faithful frame
//! driver** ([`decode_av1_spec`] / [`SpecDecodeSession`]), the
//! conformance-corpus-validated decoder that wires the parsing
//! modules (`obu`, `sequence_header`, `frame_header`,
//! `tile_group_obu`), the §5.11 `PartitionWalker` syntax walk, the
//! §7.11/§7.12/§7.13 reconstruction chain, and the full §7.4 post
//! chain (deblock, CDEF, superres, loop restoration, film grain)
//! into a pixel-out entry point.
//!
//! ## The r428 mirror-path retirement
//!
//! Until r428 a second surface existed: the "encoder-mirror" decode
//! path, the exact writer-inverse of this crate's HISTORICAL
//! constrained intra encoders (fixed 16×16 and dyn-extent drivers
//! from the early encoder arcs). Those encoders emitted
//! NON-conformant streams — their leaves coded `y_mode` with the
//! §5.11.22 non-keyframe CDFs on intra frames — so their streams
//! could only be decoded by inverting the writer bug-for-bug, and
//! [`decode_av1`] tried that mirror arm first. The conformance-grade
//! encoders (r409 onward) made the mirror emit arms redundant:
//! `encode_av1` has produced spec-conformant streams ever since, and
//! every corpus stream rides the spec driver. r428 retires the whole
//! mirror surface — emit arms, mirror decode arm, and the historical
//! `Frame::Yuv420_16x16` / `Frame::Yuv420Dyn` / `Frame::YDyn`
//! variants (the [`Frame`] enum is `#[non_exhaustive]`, so match
//! sites already carried a wildcard arm). Every stream now decodes
//! through the spec driver and surfaces as [`Frame::Spec`].

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod frame_driver;

pub use frame_driver::{decode_av1_spec, SpecFrame};
#[doc(hidden)]
pub use frame_driver::{decode_frame_spec, SpecDecodeSession};

use crate::Error;

/// One decoded (shown) frame surfaced by [`decode_av1`].
///
/// As of the r428 mirror-path retirement every stream decodes through
/// the spec-faithful driver, so [`Frame::Spec`] is the only variant
/// constructed. The enum stays `#[non_exhaustive]` (as it has been
/// since its introduction), so downstream match sites are unaffected.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Frame {
    /// A frame decoded through the spec-faithful driver
    /// ([`decode_av1_spec`]): intra + inter GOPs,
    /// `show_existing_frame`, 4:2:0 / 4:2:2 / 4:4:4 / monochrome
    /// layouts, 8/10/12-bit output, multi-tile frames, film grain,
    /// superres, loop restoration — see [`SpecFrame`] for the plane
    /// layout contract (cropped extents, per-plane dims, 10/12-bit as
    /// little-endian 2-byte samples).
    Spec(SpecFrame),
}

/// Decode an AV1 IVF v0 buffer.
///
/// Every stream rides the spec-faithful driver
/// ([`decode_av1_spec`] — the conformance-corpus-validated decoder);
/// each SHOWN frame surfaces as [`Frame::Spec`]. The historical
/// encoder-mirror acceptance arm was retired in r428 (see the module
/// docs) — the non-conformant streams it accepted could only be
/// produced by this crate's own retired mirror emit arms, never by
/// the conformance-grade encoders behind [`crate::encode_av1`].
///
/// Returns one [`Frame`] per shown frame, in output order.
///
/// ## Errors
///
/// * Buffer ends mid-IVF-header or mid-frame — [`Error::UnexpectedEnd`].
/// * Any spec-driver parse/decode failure — the driver's typed
///   [`Error`].
pub fn decode_av1(input: &[u8]) -> Result<Vec<Frame>, Error> {
    Ok(decode_av1_spec(input)?
        .into_iter()
        .map(Frame::Spec)
        .collect())
}

/// Decode an **Annex B length-delimited** AV1 bitstream (r428).
///
/// The input is the raw Annex B packing (`temporal_unit_size` /
/// `frame_unit_size` / `obu_length` `leb128` nesting per Annex B.2 —
/// no IVF container). Each temporal unit is converted to its §5.2
/// low-overhead equivalent by [`crate::annexb::split_temporal_units`]
/// (enforcing the Annex B.3 consistency + temporal-delimiter
/// placement rules) and decoded through the same
/// [`SpecDecodeSession`] the IVF path drives — reference state and
/// CDF carry work identically, so a stream repacked between the two
/// formats decodes to identical pixels.
///
/// ## Errors
///
/// * [`Error::AnnexBInvalid`] / [`Error::UnexpectedEnd`] — malformed
///   Annex B framing (see [`crate::annexb::split_temporal_units`]).
/// * Any spec-driver parse/decode failure — the driver's typed
///   [`Error`].
pub fn decode_av1_annexb(input: &[u8]) -> Result<Vec<Frame>, Error> {
    let units = crate::annexb::split_temporal_units(input)?;
    let mut session = SpecDecodeSession::new();
    let mut out: Vec<Frame> = Vec::new();
    for unit in &units {
        out.extend(
            session
                .decode_temporal_unit(unit)?
                .into_iter()
                .map(Frame::Spec),
        );
    }
    Ok(out)
}
