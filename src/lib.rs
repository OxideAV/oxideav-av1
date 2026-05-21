//! # oxideav-av1
//!
//! **Status:** orphan-rebuild scaffold (post 2026-05-20 audit), clean
//! room rebuild in progress.
//!
//! The decoder/encoder pipeline is not wired up yet. Bitstream
//! parsing has reached:
//!
//!   * **Round 1.** OBU bytestream walker described in §5.3 of the
//!     AV1 Bitstream & Decoding Process Specification — boundaries
//!     in a low-overhead bitstream plus `obu_type` /
//!     `obu_extension_flag` / `obu_has_size_field` / `temporal_id` /
//!     `spatial_id` / `obu_size` fields and a payload slice for each
//!     unit. See [`obu`].
//!
//!   * **Round 2.** Sequence header OBU parse per §5.5
//!     (`sequence_header_obu`, `color_config`, `timing_info`,
//!     `decoder_model_info`, `operating_parameters_info`). Returns a
//!     strongly typed [`sequence_header::SequenceHeader`] descriptor
//!     plus a bit-position so the trailing-bits accounting from
//!     §5.3.1 can plug in cleanly next round. See [`sequence_header`].
//!
//! Frame decoding (`frame_header_obu`, tile parsing, transform /
//! quantisation, in-loop filters, film grain) is still out of scope.
//! [`decode_av1`] / [`encode_av1`] continue to return
//! [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod bitreader;
pub mod obu;
pub mod sequence_header;

pub use obu::{parse_leb128, parse_obu, ObuDescriptor, ObuIter, ObuType};
pub use sequence_header::{
    parse_sequence_header, ColorConfig, DecoderModelInfo, OperatingParametersInfo, OperatingPoint,
    SequenceHeader, TimingInfo,
};

/// Crate-local error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A high-level API path is still a scaffold pending the
    /// clean-room rebuild.
    NotImplemented,
    /// The input ended in the middle of an OBU header, extension
    /// header, `leb128()` value, or declared payload.
    UnexpectedEnd,
    /// `obu_forbidden_bit` was set, in violation of §6.2.2.
    ForbiddenBitSet,
    /// The OBU header had `obu_has_size_field == 0`; the walker only
    /// accepts the §5.2 low-overhead format with explicit sizes.
    MissingSizeField,
    /// A `leb128()` value exceeded `(1 << 32) - 1`, the §4.10.5
    /// bitstream-conformance cap.
    Leb128Overflow,
    /// A `leb128()` encoding consumed more than 8 bytes — §4.10.5
    /// requires the MSB of the 8th byte to be 0.
    Leb128TooLong,
    /// An `obu_size` value did not fit in `usize` on this target.
    SizeOverflow,
    /// `seq_profile` was greater than 2 — values 3..=7 are reserved
    /// per §6.4.1.
    ReservedProfile(u8),
    /// `reduced_still_picture_header == 1` but `still_picture == 0`,
    /// in violation of the §6.4.1 conformance requirement.
    ReducedStillRequiresStill,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "oxideav-av1: orphan-rebuild scaffold — no decoder/encoder wired up"
            ),
            Self::UnexpectedEnd => write!(f, "oxideav-av1: unexpected end of OBU bytestream"),
            Self::ForbiddenBitSet => {
                write!(f, "oxideav-av1: obu_forbidden_bit was set (§6.2.2)")
            }
            Self::MissingSizeField => write!(
                f,
                "oxideav-av1: obu_has_size_field == 0; only the §5.2 low-overhead format is supported"
            ),
            Self::Leb128Overflow => {
                write!(f, "oxideav-av1: leb128 value exceeded the §4.10.5 cap")
            }
            Self::Leb128TooLong => write!(
                f,
                "oxideav-av1: leb128 encoding used more than 8 bytes (§4.10.5)"
            ),
            Self::SizeOverflow => {
                write!(f, "oxideav-av1: obu_size did not fit in usize on this target")
            }
            Self::ReservedProfile(p) => write!(
                f,
                "oxideav-av1: seq_profile {p} is reserved (only 0..=2 are conformant, §6.4.1)"
            ),
            Self::ReducedStillRequiresStill => write!(
                f,
                "oxideav-av1: reduced_still_picture_header == 1 requires still_picture == 1 (§6.4.1)"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Decode an AV1 elementary stream.
///
/// Still a stub: this round only added the OBU bytestream walker.
pub fn decode_av1(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// Encode YUV data into an AV1 elementary stream.
pub fn encode_av1(_pixels: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the clean-room scaffold does not yet
/// register a working decoder or encoder.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("av1", register);
