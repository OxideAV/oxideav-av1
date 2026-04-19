//! AV1 transform taxonomy — §6.8.21 + §7.7.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/transform/types.go`
//! (MIT, KarpelesLab/goavif).
//!
//! Two orthogonal axes:
//!
//! - [`TxType`] — one of 16 2D transform pairs (DCT/ADST/flipped
//!   ADST/identity per row and column). The spec lists them in
//!   §6.8.21 Table 13-4.
//! - [`TxSize`] — transform dimensions. The low bits encode
//!   `log2(width)` and the high bits `log2(height)`.
//!
//! Phase 3 only implements 4/8/16-point DCT + ADST (so the callable
//! set is a strict subset of goavif's); every other combination surfaces
//! `Error::Unsupported` at [`crate::transform::inverse_2d`] entry.

use oxideav_core::{Error, Result};

/// AV1 2D transform pair (§6.8.21).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxType {
    DctDct = 0,
    AdstDct = 1,
    DctAdst = 2,
    AdstAdst = 3,
    FlipAdstDct = 4,
    DctFlipAdst = 5,
    FlipAdstFlipAdst = 6,
    AdstFlipAdst = 7,
    FlipAdstAdst = 8,
    IdtIdt = 9,
    VDct = 10,
    HDct = 11,
    VAdst = 12,
    HAdst = 13,
    VFlipAdst = 14,
    HFlipAdst = 15,
}

impl TxType {
    pub fn from_u32(v: u32) -> Result<Self> {
        Ok(match v {
            0 => Self::DctDct,
            1 => Self::AdstDct,
            2 => Self::DctAdst,
            3 => Self::AdstAdst,
            4 => Self::FlipAdstDct,
            5 => Self::DctFlipAdst,
            6 => Self::FlipAdstFlipAdst,
            7 => Self::AdstFlipAdst,
            8 => Self::FlipAdstAdst,
            9 => Self::IdtIdt,
            10 => Self::VDct,
            11 => Self::HDct,
            12 => Self::VAdst,
            13 => Self::HAdst,
            14 => Self::VFlipAdst,
            15 => Self::HFlipAdst,
            _ => return Err(Error::invalid(format!("av1 tx: invalid tx_type {v}"))),
        })
    }

    /// Row-direction kind (first character of the spec name —
    /// e.g. `AdstDct` has DCT as its *row* kind).
    pub fn row_kind(self) -> Kind {
        match self {
            Self::DctDct | Self::AdstDct | Self::FlipAdstDct | Self::VDct => Kind::Dct,
            Self::DctAdst | Self::AdstAdst | Self::FlipAdstAdst | Self::VAdst => Kind::Adst,
            Self::DctFlipAdst
            | Self::AdstFlipAdst
            | Self::FlipAdstFlipAdst
            | Self::VFlipAdst => Kind::FlipAdst,
            Self::IdtIdt | Self::HDct | Self::HAdst | Self::HFlipAdst => Kind::Idtx,
        }
    }

    /// Column-direction kind.
    pub fn col_kind(self) -> Kind {
        match self {
            Self::DctDct | Self::DctAdst | Self::DctFlipAdst | Self::HDct => Kind::Dct,
            Self::AdstDct | Self::AdstAdst | Self::AdstFlipAdst | Self::HAdst => Kind::Adst,
            Self::FlipAdstDct
            | Self::FlipAdstAdst
            | Self::FlipAdstFlipAdst
            | Self::HFlipAdst => Kind::FlipAdst,
            Self::IdtIdt | Self::VDct | Self::VAdst | Self::VFlipAdst => Kind::Idtx,
        }
    }
}

/// 1-D transform kind (DCT / ADST / flipped ADST / identity).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Dct,
    Adst,
    FlipAdst,
    Idtx,
}

/// AV1 transform size — values match the spec's `TxSize` enum.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxSize {
    Tx4x4 = 0,
    Tx8x8 = 1,
    Tx16x16 = 2,
    Tx32x32 = 3,
    Tx64x64 = 4,
    Tx4x8 = 5,
    Tx8x4 = 6,
    Tx8x16 = 7,
    Tx16x8 = 8,
    Tx16x32 = 9,
    Tx32x16 = 10,
    Tx32x64 = 11,
    Tx64x32 = 12,
    Tx4x16 = 13,
    Tx16x4 = 14,
    Tx8x32 = 15,
    Tx32x8 = 16,
    Tx16x64 = 17,
    Tx64x16 = 18,
}

impl TxSize {
    /// 1-D row length in samples.
    pub fn width(self) -> usize {
        match self {
            Self::Tx4x4 | Self::Tx4x8 | Self::Tx4x16 => 4,
            Self::Tx8x8 | Self::Tx8x4 | Self::Tx8x16 | Self::Tx8x32 => 8,
            Self::Tx16x16
            | Self::Tx16x8
            | Self::Tx16x32
            | Self::Tx16x4
            | Self::Tx16x64 => 16,
            Self::Tx32x32 | Self::Tx32x16 | Self::Tx32x64 | Self::Tx32x8 => 32,
            Self::Tx64x64 | Self::Tx64x32 | Self::Tx64x16 => 64,
        }
    }

    /// 1-D column length in samples.
    pub fn height(self) -> usize {
        match self {
            Self::Tx4x4 | Self::Tx8x4 | Self::Tx16x4 => 4,
            Self::Tx8x8 | Self::Tx4x8 | Self::Tx16x8 | Self::Tx32x8 => 8,
            Self::Tx16x16
            | Self::Tx8x16
            | Self::Tx32x16
            | Self::Tx4x16
            | Self::Tx64x16 => 16,
            Self::Tx32x32 | Self::Tx16x32 | Self::Tx64x32 | Self::Tx8x32 => 32,
            Self::Tx64x64 | Self::Tx32x64 | Self::Tx16x64 => 64,
        }
    }

    /// Pick the square `TxSize` for the given side length; returns
    /// `None` if the side isn't a valid AV1 transform dimension.
    pub fn square(side: usize) -> Option<Self> {
        match side {
            4 => Some(Self::Tx4x4),
            8 => Some(Self::Tx8x8),
            16 => Some(Self::Tx16x16),
            32 => Some(Self::Tx32x32),
            64 => Some(Self::Tx64x64),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_for_dct_dct() {
        assert_eq!(TxType::DctDct.row_kind(), Kind::Dct);
        assert_eq!(TxType::DctDct.col_kind(), Kind::Dct);
    }

    #[test]
    fn kind_for_adst_dct() {
        // Spec: "AdstDct" = column=ADST, row=DCT.
        assert_eq!(TxType::AdstDct.row_kind(), Kind::Dct);
        assert_eq!(TxType::AdstDct.col_kind(), Kind::Adst);
    }

    #[test]
    fn tx_size_dimensions_for_square() {
        assert_eq!(TxSize::Tx4x4.width(), 4);
        assert_eq!(TxSize::Tx4x4.height(), 4);
        assert_eq!(TxSize::Tx8x8.width(), 8);
        assert_eq!(TxSize::Tx16x16.height(), 16);
    }

    #[test]
    fn tx_size_square_from_side() {
        assert_eq!(TxSize::square(4), Some(TxSize::Tx4x4));
        assert_eq!(TxSize::square(32), Some(TxSize::Tx32x32));
        assert_eq!(TxSize::square(5), None);
    }
}
