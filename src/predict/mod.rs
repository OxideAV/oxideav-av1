//! AV1 prediction — §7.11.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/predict` (MIT,
//! KarpelesLab/goavif). Phase 5 covers the full intra predictor set
//! (13 luma modes + CFL + filter-intra). Inter prediction lands with
//! Phase 6.

pub mod intra;
