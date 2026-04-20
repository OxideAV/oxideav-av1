//! AV1 film grain synthesis — spec §7.20.
//!
//! Implements the spec-correct §7.20.2 32×32 tiler rather than a
//! per-pixel LFSR shortcut.
//!
//! Film grain is a post-processing pass that adds structured noise to
//! decoded pixels, preserving the look of photographic grain after
//! aggressive compression. The grain parameters (seed, scaling curve,
//! AR coefficients) are signaled in the frame header's
//! `film_grain_params` block; application is strictly a synthesis step
//! that never affects reference frames.
//!
//! Pipeline:
//!
//! 1. Generate full 74×74 luma + 38×38 chroma AR-model noise patches
//!    per the `grain_seed` — once per frame ([`patch::new_luma_template`]
//!    / [`patch::new_chroma_template`]).
//! 2. For each 32×32 luma block in the output, extract a pseudo-random
//!    32×32 sub-patch from the template, scale per the per-pixel
//!    intensity LUT (§7.20.3.3), add into the plane, clip
//!    ([`apply::apply_with_template`]).
//! 3. Same for chroma with Cb/Cr chroma-from-luma blending per
//!    `cb_mult` / `cb_luma_mult` / `cb_offset`.

pub mod apply;
pub mod ar;
pub mod patch;
pub mod rng;
pub mod scaling;

pub use apply::{apply_frame, apply_with_template, apply_with_template16};
pub use ar::{apply_ar, generate_grain_template};
pub use patch::{new_chroma_template, new_luma_template, Template};
pub use rng::Rng;
pub use scaling::{build_lut, Point, ScalingLut};

/// Runtime parameters consumed by the grain application step. The
/// fields mirror the subset of `film_grain_params` the synthesis needs
/// at apply time; the parsed [`crate::frame_header_tail::FilmGrainParams`]
/// struct still carries the raw bitstream values.
#[derive(Clone, Debug, Default)]
pub struct Params {
    /// Mixes into the per-block RNG seed. Zero disables grain
    /// application (spec §7.20.1).
    pub grain_seed: u16,
    /// Luma scaling LUT.
    pub scaling_y: ScalingLut,
    /// Cb / Cr scaling LUTs. Callers that want luma-only grain can
    /// leave these as zero LUTs.
    pub scaling_u: ScalingLut,
    pub scaling_v: ScalingLut,
    /// Right-shift applied to the product `(grain * scale)` before it
    /// is added to the pixel. Spec-allowed range is 8..=11; default 8
    /// leaves the grain at full amplitude.
    pub scaling_shift: u8,
    /// Clamp output to broadcast-legal range `[16<<(bd-8), 235<<(bd-8)]`
    /// for luma and `[16<<(bd-8), 240<<(bd-8)]` for chroma when set.
    pub clip_to_restricted_range: bool,
    /// Overlap flag (spec §7.20.3.4). When true, adjacent 32×32 grain
    /// patches cross-fade across a 2-sample rim so block boundaries
    /// aren't visible in the synthesised grain.
    pub overlap_flag: bool,
    /// Chroma-from-luma blending (spec §7.20.3.2). `mult_cb / mult_cr`
    /// are the grain-from-grain weights, `luma_mult_cb / luma_mult_cr`
    /// add the luma average, `offset_cb / offset_cr` are Q0 biases.
    /// All zero means "use the chroma-plane RNG output directly".
    pub cb_mult: i32,
    pub cb_luma_mult: i32,
    pub cb_offset: i32,
    pub cr_mult: i32,
    pub cr_luma_mult: i32,
    pub cr_offset: i32,
}
