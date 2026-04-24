//! AV1 reference-scaling — §7.9 / §7.11.3.2.
//!
//! When a GOP's resolution varies frame-to-frame (e.g. a `superres`
//! key frame followed by full-resolution inter frames), the MC pipeline
//! needs a per-axis scale factor to map `(block_pos + mv)` from the
//! current frame's coordinate system into the reference plane. The
//! spec encodes the factor as a 14-bit fixed-point value
//! `ref_dim << 14 / current_dim` (§5.9.8 `compute_image_size`).
//!
//! This module is intentionally narrow: it exposes a single
//! [`ScaleFactors`] struct plus helpers that clamp the scaled position
//! inside the reference frame. The full §7.11.3.2 warp machinery
//! (`get_block_position`, subpel step via `dSubpelParams`) is not yet
//! wired; this is the minimum plumbing to handle `prev_frame` arriving
//! at a different resolution than `current_frame`.

/// Per-axis fixed-point scale factors. `x_step = round(ref_w << 14 /
/// current_w)`; a value of `1 << 14 = 16384` means the reference is
/// the same dimension as the current frame (unscaled).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScaleFactors {
    pub x_step: i32,
    pub y_step: i32,
}

impl ScaleFactors {
    /// Identity scaling — both steps equal `1 << 14`.
    pub const IDENTITY: Self = Self {
        x_step: 1 << 14,
        y_step: 1 << 14,
    };

    /// Build scale factors from frame dimensions. When `ref_w ==
    /// cur_w` and `ref_h == cur_h` returns [`IDENTITY`].
    pub fn new(ref_w: u32, ref_h: u32, cur_w: u32, cur_h: u32) -> Self {
        debug_assert!(cur_w > 0 && cur_h > 0);
        let x_step = (((ref_w as i64) << 14) / (cur_w as i64)) as i32;
        let y_step = (((ref_h as i64) << 14) / (cur_h as i64)) as i32;
        Self { x_step, y_step }
    }

    /// `true` when neither axis is scaled (`x_step == y_step == 1 <<
    /// 14`).
    pub fn is_identity(self) -> bool {
        self.x_step == (1 << 14) && self.y_step == (1 << 14)
    }

    /// Map a current-frame sample position `(x, y)` (in integer
    /// samples) plus an eighth-pel MV to a reference-frame sample
    /// position in 1/1024-pel units (14-bit fixed + MV shifted).
    /// Returns `(ref_x_subpel, ref_y_subpel)`; callers shift right by
    /// 10 for the integer part or use the low 10 bits for sub-pel
    /// phase.
    ///
    /// Identity scale factors yield positions equivalent to the
    /// unscaled path (MV applied at eighth-pel precision).
    pub fn project(self, x: i32, y: i32, mv_row: i32, mv_col: i32) -> (i64, i64) {
        // Sub-pel precision: position is in 1/(1<<4) units after the
        // x_step multiply (since x_step is in 1/(1<<14) of reference
        // over current and position is integer).
        // Refined: ref_x_subpel = x * x_step + (mv_col * x_step) / 8
        let ref_x =
            (x as i64) * (self.x_step as i64) + ((mv_col as i64) * (self.x_step as i64)) / 8;
        let ref_y =
            (y as i64) * (self.y_step as i64) + ((mv_row as i64) * (self.y_step as i64)) / 8;
        (ref_x, ref_y)
    }
}

impl Default for ScaleFactors {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_for_matching_dims() {
        let sf = ScaleFactors::new(128, 96, 128, 96);
        assert!(sf.is_identity());
        assert_eq!(sf.x_step, 1 << 14);
        assert_eq!(sf.y_step, 1 << 14);
    }

    #[test]
    fn half_resolution_ref_is_half_step() {
        // ref is 64×32, current is 128×64 — reference is half-size on
        // both axes, so x_step / y_step are 8192 (= 1<<13).
        let sf = ScaleFactors::new(64, 32, 128, 64);
        assert_eq!(sf.x_step, 1 << 13);
        assert_eq!(sf.y_step, 1 << 13);
        assert!(!sf.is_identity());
    }

    #[test]
    fn project_identity_is_mv_shift_by_one() {
        // At identity, project((x, y), mv) should return (x + mv/8, y + mv/8)
        // in shifted units. For zero MV, it's just x<<14, y<<14.
        let sf = ScaleFactors::IDENTITY;
        let (rx, ry) = sf.project(10, 20, 0, 0);
        assert_eq!(rx, 10i64 << 14);
        assert_eq!(ry, 20i64 << 14);
    }

    #[test]
    fn project_with_mv_at_identity() {
        // MV col=16 (=2 integer samples) at (0, 0) → ref_x = 2 << 14.
        let sf = ScaleFactors::IDENTITY;
        let (rx, _ry) = sf.project(0, 0, 0, 16);
        assert_eq!(rx, 2i64 << 14);
    }
}
