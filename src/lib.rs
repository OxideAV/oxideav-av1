//! # oxideav-av1
//!
//! **Status:** orphan-rebuild scaffold (post 2026-05-20 audit), clean
//! room round 1 in progress.
//!
//! The decoder/encoder pipeline is not wired up yet. Round 1 adds the
//! OBU bytestream walker described in §5.3.1 of the AV1 Bitstream &
//! Decoding Process Specification — i.e. it can identify the OBU
//! boundaries in a low-overhead bitstream and surface the
//! `obu_type` / `obu_extension_flag` / `obu_has_size_field` fields
//! plus the `obu_size` (`leb128`) payload length and slice for each
//! unit. Frame decoding remains unimplemented.
//!
//! See [`obu`] for the walker. The legacy [`decode_av1`] /
//! [`encode_av1`] entry points still return [`Error::NotImplemented`]
//! pending the frame-decoding work.

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

pub mod obu;

pub use obu::{parse_leb128, parse_obu, ObuDescriptor, ObuIter, ObuType};

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
