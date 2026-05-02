//! AV1 deblocking loop filter — §7.14.
//!
//! Implements the per-8×8-MI-pair narrow/wide deblocking filter that
//! runs after coefficient reconstruction and before CDEF.
//!
//! The core pieces:
//!
//! * [`mask::DeriveThresholds`] — compute `{limit, blimit, thresh}` from
//!   a `filter_level` + `sharpness`.
//! * [`narrow::Filter4`] — 4-tap narrow deblocker.
//! * [`wide::Filter8`] — 8-tap wide deblocker.
//! * [`frame::ApplyFrameNarrow`] — drive the narrow filter over a plane.

pub mod edge;
pub mod frame;
pub mod mask;
pub mod narrow;
pub mod wide;

pub use edge::{
    apply_plane as apply_plane_edges, apply_plane16 as apply_plane_edges16, derive_lvl,
    filter_len_for, filter_size, EdgePlane, EdgePlane16, LfModeType, MiGrid, MiInfo, INTRA_FRAME,
    LAST_FRAME, MAX_LOOP_FILTER as EDGE_MAX_LOOP_FILTER, SEG_LVL_ALT_LF_Y_V,
};
pub use frame::{apply_frame_narrow, apply_frame_narrow16, uniform_grid, EdgeGrid, Plane, Plane16};
pub use mask::derive_thresholds;
pub use narrow::{
    apply_horizontal_edge4, apply_horizontal_edge4_16, apply_vertical_edge4,
    apply_vertical_edge4_16, filter4, filter4_16, high_edge_variation, high_edge_variation16,
    narrow_mask, narrow_mask16, scale_thresholds16, Thresholds, Thresholds16,
};
pub use wide::{filter8, flat8_mask};
