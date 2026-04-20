//! AV1 Loop Restoration filter family — §7.17.
//!
//! Two filter kinds are defined:
//!
//! 1. **Wiener** — a 7×7 separable FIR with symmetric coefficients
//!    (horizontal then vertical 1D passes, 4 unique coefficients per
//!    axis).
//!
//! 2. **Self-guided restoration (SGR)** — a two-pass variance-adaptive
//!    box filter with per-unit edge-preserving weights.
//!
//! Per-restoration-unit signalling (`use_wiener` / `use_sgrproj` flags
//! plus delta-coded coefficient blobs — §5.11.40-.44) lives with the
//! tile decoder in [`crate::decode::lr_unit`]. This module provides
//! the filter primitives plus a full-frame driver that consults a
//! caller-supplied per-unit parameter table (rather than any
//! "default unit params" shortcut).

pub mod frame;
pub mod sgr;
pub mod wiener;

pub use frame::{apply_frame, apply_frame16, Plane, Plane16};
pub use sgr::{
    apply_sgr, apply_sgr16, box_mean, box_var, sgr_sub_filter, sgr_sub_filter16, SgrParams,
};
pub use wiener::{apply_wiener, apply_wiener16, WienerTaps};

/// Filter kind signalled per restoration unit (spec §7.17.1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FilterType {
    /// No restoration — output equals input.
    #[default]
    None,
    /// Wiener filter — `unit_params.wiener_*` carries the 6 decoded
    /// taps (center tap is implicit — see §7.17.5).
    Wiener,
    /// Self-guided — `unit_params.sgr` carries `r0,r1,eps0,eps1,xq`.
    Sgr,
}

/// Per-restoration-unit parameters — one of Wiener, SGR, or None.
/// Filled by the tile decoder from the bitstream (§5.11.40-.44) or by
/// tests / synthetic drivers.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnitParams {
    pub filter: FilterType,
    /// Horizontal + vertical 7-tap Wiener taps (4 unique coefficients
    /// per axis; middle tap is derived so sum == 128).
    pub wiener_horiz: WienerTaps,
    pub wiener_vert: WienerTaps,
    /// SGR filter parameters.
    pub sgr: SgrParams,
}
