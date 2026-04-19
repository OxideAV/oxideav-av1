//! AV1 residual-to-pixel reconstruction — §7.7.4 final step.
//!
//! Ported from
//! `github.com/KarpelesLab/goavif/av1/decoder/reconstruct.go` (MIT,
//! KarpelesLab/goavif).
//!
//! After the 2D inverse transform produces a signed `i32` residual
//! block, each coefficient is added to the predictor and clipped to
//! the plane's sample range. Phase 3 wires only the 8-bit path; the
//! 10-/12-bit variant is present for future use but the current
//! pipeline doesn't call it yet.

/// Clip-add an `i32` residual to an 8-bit predictor plane.
///
/// `dst`, `pred`, and `residual` are all length `w * h` in row-major
/// layout. Result is written to `dst`.
pub fn clip_add(dst: &mut [u8], pred: &[u8], residual: &[i32], w: usize, h: usize) {
    let n = w * h;
    debug_assert_eq!(dst.len(), n);
    debug_assert_eq!(pred.len(), n);
    debug_assert_eq!(residual.len(), n);
    for i in 0..n {
        let v = pred[i] as i32 + residual[i];
        dst[i] = v.clamp(0, 255) as u8;
    }
}

/// In-place clip-add where `dst` already holds the predictor. Saves a
/// scratch copy compared to [`clip_add`].
pub fn clip_add_in_place(dst: &mut [u8], residual: &[i32], w: usize, h: usize) {
    let n = w * h;
    debug_assert_eq!(dst.len(), n);
    debug_assert_eq!(residual.len(), n);
    for i in 0..n {
        let v = dst[i] as i32 + residual[i];
        dst[i] = v.clamp(0, 255) as u8;
    }
}

/// Clip-add into a 16-bit destination clamped to `[0, (1<<bit_depth)-1]`.
/// Used by the 10-/12-bit pipelines — Phase 3 does not yet invoke it
/// from the tile walker.
pub fn clip_add16(
    dst: &mut [u16],
    pred: &[u16],
    residual: &[i32],
    w: usize,
    h: usize,
    bit_depth: u32,
) {
    let n = w * h;
    debug_assert_eq!(dst.len(), n);
    debug_assert_eq!(pred.len(), n);
    debug_assert_eq!(residual.len(), n);
    let max = ((1i32) << bit_depth) - 1;
    for i in 0..n {
        let v = pred[i] as i32 + residual[i];
        dst[i] = v.clamp(0, max) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_add_clips_out_of_range_samples() {
        let pred = [100u8, 150, 200, 50];
        let res = [10i32, -20, 100, -100];
        let mut dst = [0u8; 4];
        clip_add(&mut dst, &pred, &res, 2, 2);
        // 200+100 clips to 255; 50-100 clips to 0.
        assert_eq!(dst, [110, 130, 255, 0]);
    }

    #[test]
    fn clip_add_zero_residual_copies_pred() {
        let pred = [42u8; 4];
        let res = [0i32; 4];
        let mut dst = [0u8; 4];
        clip_add(&mut dst, &pred, &res, 2, 2);
        for &v in &dst {
            assert_eq!(v, 42);
        }
    }

    #[test]
    fn clip_add_in_place_equivalent_to_clip_add_when_dst_is_pred() {
        let pred = [100u8, 150, 200, 50];
        let res = [10i32, -20, 100, -100];
        let mut via_copy = [0u8; 4];
        clip_add(&mut via_copy, &pred, &res, 2, 2);
        let mut via_inplace = pred;
        clip_add_in_place(&mut via_inplace, &res, 2, 2);
        assert_eq!(via_copy, via_inplace);
    }

    #[test]
    fn clip_add16_clips_to_12_bit_max() {
        let pred = [1000u16, 100, 0, 4000];
        let res = [50i32, -200, -50, 200];
        let mut dst = [0u16; 4];
        clip_add16(&mut dst, &pred, &res, 2, 2, 12);
        // Expected: 1050, 0, 0, 4095.
        assert_eq!(dst, [1050, 0, 0, 4095]);
    }

    #[test]
    fn clip_add16_respects_10_bit_max() {
        let pred = [1000u16];
        let res = [100i32];
        let mut dst = [0u16; 1];
        clip_add16(&mut dst, &pred, &res, 1, 1, 10);
        assert_eq!(dst[0], 1023);
    }
}
