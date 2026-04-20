//! AV1 prediction — §7.11.
//!
//! Phase 5 covers the full intra predictor set (13 luma modes + CFL +
//! filter-intra). Phase 7 adds the 8-tap sub-pel interpolation filters
//! used by the inter path ([`interp`]).

pub mod interp;
pub mod intra;
