//! Constrained Directional Enhancement Filter (CDEF) — §7.15.
//!
//! Operates on 8×8 blocks of the reconstructed plane, picks a
//! direction via [`direction::find_direction`], then applies
//! [`filter::filter_block`].
//!
//! Call [`frame::apply_frame`] (u8) or [`frame::apply_frame16`] (u16)
//! after deblocking and before loop restoration.

pub mod direction;
pub mod filter;
pub mod frame;

pub use direction::{find_direction, find_direction16, DIRECTIONS};
pub use filter::{
    adjust_pri_strength, constrain, filter_block, filter_block16, filter_block_spec,
    filter_block_spec16, CDEF_PRI_TAPS, CDEF_SEC_TAPS, CDEF_UV_DIR, PRIMARY_TAPS, SECONDARY_TAPS,
};
pub use frame::{
    apply_frame, apply_frame16, apply_frame_per_sb, apply_frame_per_sb16, apply_frame_spec,
    apply_frame_spec16, Plane, Plane16, SbStrengths,
};
