//! AV1 deblocking loop filter — §7.14.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/loopfilter` (MIT,
//! KarpelesLab/goavif). Implements the per-8×8-MI-pair narrow/wide
//! deblocking filter that runs after coefficient reconstruction and
//! before CDEF.
//!
//! The core pieces:
//!
//! * [`mask::DeriveThresholds`] — compute `{limit, blimit, thresh}` from
//!   a `filter_level` + `sharpness`.
//! * [`narrow::Filter4`] — 4-tap narrow deblocker.
//! * [`wide::Filter8`] — 8-tap wide deblocker.
//! * [`frame::ApplyFrameNarrow`] — drive the narrow filter over a plane.

pub mod frame;
pub mod mask;
pub mod narrow;
pub mod wide;

pub use frame::{apply_frame_narrow, apply_frame_narrow16, uniform_grid, EdgeGrid, Plane, Plane16};
pub use mask::derive_thresholds;
pub use narrow::{
    apply_horizontal_edge4, apply_horizontal_edge4_16, apply_vertical_edge4,
    apply_vertical_edge4_16, filter4, filter4_16, high_edge_variation, high_edge_variation16,
    narrow_mask, narrow_mask16, scale_thresholds16, Thresholds, Thresholds16,
};
pub use wide::{filter8, flat8_mask};
