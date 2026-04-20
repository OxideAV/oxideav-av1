//! Directional intra predictors — §7.11.2.5.
//!
//! Six base angles (45°, 67°, 113°, 135°, 157°, 203°) plus an
//! `angle_delta ∈ {-3..3}` at 3° steps. Per-sample projection is
//! formed along the chosen angle and sampled from the above/left
//! neighbours via sub-pixel interpolation.

use crate::decode::modes::IntraMode;

/// `dr_intra_derivative` from libaom `reconintra.h`. Indexed by an
/// offset derived from the angle; entries that aren't on 3° increments
/// are 0. The (256 / tan(angle)) values are scaled so dx/dy are Q6
/// fixed-point.
pub const DR_INTRA_DERIVATIVE: [i16; 90] = [
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372, 0, 0, 0, 0, 273, 0, 0, 215, 0, 0, 178, 0, 0, 151, 0, 0,
    132, 0, 0, 116, 0, 0, 102, 0, 0, 0, 90, 0, 0, 80, 0, 0, 71, 0, 0, 64, 0, 0, 57, 0, 0, 51, 0, 0,
    45, 0, 0, 0, 40, 0, 0, 35, 0, 0, 31, 0, 0, 27, 0, 0, 23, 0, 0, 19, 0, 0, 15, 0, 0, 0, 0, 11, 0,
    0, 7, 0, 0, 3, 0, 0,
];

/// Base angle (degrees) for each intra mode index; non-directional
/// modes return 0. Matches libaom's `mode_to_angle_map[INTRA_MODES]`.
pub const MODE_TO_ANGLE_MAP: [i32; 13] = [
    0,   // DC_PRED
    90,  // V_PRED
    180, // H_PRED
    45,  // D45_PRED
    135, // D135_PRED
    113, // D113_PRED
    157, // D157_PRED
    203, // D203_PRED
    67,  // D67_PRED
    0, 0, 0, 0, // SMOOTH*, PAETH
];

/// Base angle (in degrees) for `mode`. 0 for non-directional modes.
pub fn mode_to_angle_map(mode: IntraMode) -> i32 {
    let idx = mode as usize;
    if idx < MODE_TO_ANGLE_MAP.len() {
        MODE_TO_ANGLE_MAP[idx]
    } else {
        0
    }
}

/// Horizontal per-row step in Q6 (256/tan). Valid for 0 < angle < 180.
fn get_dx(angle: i32) -> i32 {
    if angle > 0 && angle < 90 {
        DR_INTRA_DERIVATIVE[angle as usize] as i32
    } else if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(180 - angle) as usize] as i32
    } else {
        1
    }
}

/// Vertical per-column step in Q6. Valid for 90 < angle < 270.
fn get_dy(angle: i32) -> i32 {
    if angle > 90 && angle < 180 {
        DR_INTRA_DERIVATIVE[(angle - 90) as usize] as i32
    } else if angle > 180 && angle < 270 {
        DR_INTRA_DERIVATIVE[(270 - angle) as usize] as i32
    } else {
        1
    }
}

/// Directional prediction for an angle in `(0, 270)` degrees. The
/// above/left slices must be "extended" — replicated to at least
/// `(w + h + 1)` samples at each edge so the projection doesn't read
/// out of range. Callers replicate the last sample as needed.
pub fn directional_pred(dst: &mut [u8], w: usize, h: usize, above: &[u8], left: &[u8], angle: i32) {
    if angle > 0 && angle < 90 {
        dr_pred_above_zone(dst, w, h, above, get_dx(angle));
    } else if angle > 180 && angle < 270 {
        dr_pred_left_zone(dst, w, h, left, get_dy(angle));
    } else if angle > 90 && angle < 180 {
        dr_pred_mixed_zone(dst, w, h, above, left, get_dx(angle), get_dy(angle));
    } else if angle == 90 {
        // Vertical — identical to V_PRED on the above row.
        for r in 0..h {
            for c in 0..w {
                dst[r * w + c] = above[c];
            }
        }
    } else if angle == 180 {
        // Horizontal — identical to H_PRED on the left column.
        for r in 0..h {
            let v = left[r];
            for c in 0..w {
                dst[r * w + c] = v;
            }
        }
    }
}

fn dr_pred_above_zone(dst: &mut [u8], w: usize, h: usize, above: &[u8], dx: i32) {
    let max_idx = above.len().saturating_sub(1);
    for r in 0..h {
        let offset = ((r as i32) + 1) * dx;
        for c in 0..w {
            let base = (c as i32) + (offset >> 6);
            let shift = (offset >> 1) & 0x1F;
            let b1 = base.max(0) as usize;
            let b2 = (base + 1).max(0) as usize;
            let b1 = b1.min(max_idx);
            let b2 = b2.min(max_idx);
            let v = ((above[b1] as i32) * (32 - shift) + (above[b2] as i32) * shift + 16) >> 5;
            dst[r * w + c] = v.clamp(0, 255) as u8;
        }
    }
}

fn dr_pred_left_zone(dst: &mut [u8], w: usize, h: usize, left: &[u8], dy: i32) {
    let max_idx = left.len().saturating_sub(1);
    for r in 0..h {
        for c in 0..w {
            let offset = ((c as i32) + 1) * dy;
            let base = (r as i32) + (offset >> 6);
            let shift = (offset >> 1) & 0x1F;
            let b1 = base.max(0) as usize;
            let b2 = (base + 1).max(0) as usize;
            let b1 = b1.min(max_idx);
            let b2 = b2.min(max_idx);
            let v = ((left[b1] as i32) * (32 - shift) + (left[b2] as i32) * shift + 16) >> 5;
            dst[r * w + c] = v.clamp(0, 255) as u8;
        }
    }
}

fn dr_pred_mixed_zone(
    dst: &mut [u8],
    w: usize,
    h: usize,
    above: &[u8],
    left: &[u8],
    dx: i32,
    dy: i32,
) {
    let max_a = above.len().saturating_sub(1);
    let max_l = left.len().saturating_sub(1);
    let inv_dy = dy;
    for r in 0..h {
        for c in 0..w {
            let x_off = -((r as i32) + 1) * dx + ((c as i32) + 1) * 64;
            let y_off = -((c as i32) + 1) * inv_dy + ((r as i32) + 1) * 64;
            if x_off >= 0 {
                let base = x_off >> 6;
                let shift = (x_off >> 1) & 0x1F;
                let b1 = base.max(0) as usize;
                let b2 = (base + 1).max(0) as usize;
                let b1 = b1.min(max_a);
                let b2 = b2.min(max_a);
                let v = ((above[b1] as i32) * (32 - shift) + (above[b2] as i32) * shift + 16) >> 5;
                dst[r * w + c] = v.clamp(0, 255) as u8;
            } else {
                let base = y_off >> 6;
                let shift = (y_off >> 1) & 0x1F;
                let b1 = base.max(0) as usize;
                let b2 = (base + 1).max(0) as usize;
                let b1 = b1.min(max_l);
                let b2 = b2.min(max_l);
                let v = ((left[b1] as i32) * (32 - shift) + (left[b2] as i32) * shift + 16) >> 5;
                dst[r * w + c] = v.clamp(0, 255) as u8;
            }
        }
    }
}

/// 16-bit directional predictor. Output is clipped to
/// `[0, (1 << bitDepth) - 1]`.
pub fn directional_pred16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    above: &[u16],
    left: &[u16],
    angle: i32,
    bit_depth: u32,
) {
    let max_v = ((1i32) << bit_depth) - 1;
    if angle > 0 && angle < 90 {
        dr_pred_above_zone16(dst, w, h, above, get_dx(angle), max_v);
    } else if angle > 180 && angle < 270 {
        dr_pred_left_zone16(dst, w, h, left, get_dy(angle), max_v);
    } else if angle > 90 && angle < 180 {
        dr_pred_mixed_zone16(dst, w, h, above, left, get_dx(angle), get_dy(angle), max_v);
    } else if angle == 90 {
        for r in 0..h {
            for c in 0..w {
                dst[r * w + c] = above[c];
            }
        }
    } else if angle == 180 {
        for r in 0..h {
            let v = left[r];
            for c in 0..w {
                dst[r * w + c] = v;
            }
        }
    }
}

fn dr_pred_above_zone16(dst: &mut [u16], w: usize, h: usize, above: &[u16], dx: i32, max_v: i32) {
    let max_idx = above.len().saturating_sub(1);
    for r in 0..h {
        let offset = ((r as i32) + 1) * dx;
        for c in 0..w {
            let base = (c as i32) + (offset >> 6);
            let shift = (offset >> 1) & 0x1F;
            let b1 = base.max(0) as usize;
            let b2 = (base + 1).max(0) as usize;
            let b1 = b1.min(max_idx);
            let b2 = b2.min(max_idx);
            let v = ((above[b1] as i32) * (32 - shift) + (above[b2] as i32) * shift + 16) >> 5;
            dst[r * w + c] = v.clamp(0, max_v) as u16;
        }
    }
}

fn dr_pred_left_zone16(dst: &mut [u16], w: usize, h: usize, left: &[u16], dy: i32, max_v: i32) {
    let max_idx = left.len().saturating_sub(1);
    for r in 0..h {
        for c in 0..w {
            let offset = ((c as i32) + 1) * dy;
            let base = (r as i32) + (offset >> 6);
            let shift = (offset >> 1) & 0x1F;
            let b1 = base.max(0) as usize;
            let b2 = (base + 1).max(0) as usize;
            let b1 = b1.min(max_idx);
            let b2 = b2.min(max_idx);
            let v = ((left[b1] as i32) * (32 - shift) + (left[b2] as i32) * shift + 16) >> 5;
            dst[r * w + c] = v.clamp(0, max_v) as u16;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dr_pred_mixed_zone16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    above: &[u16],
    left: &[u16],
    dx: i32,
    dy: i32,
    max_v: i32,
) {
    let max_a = above.len().saturating_sub(1);
    let max_l = left.len().saturating_sub(1);
    let inv_dy = dy;
    for r in 0..h {
        for c in 0..w {
            let x_off = -((r as i32) + 1) * dx + ((c as i32) + 1) * 64;
            let y_off = -((c as i32) + 1) * inv_dy + ((r as i32) + 1) * 64;
            if x_off >= 0 {
                let base = x_off >> 6;
                let shift = (x_off >> 1) & 0x1F;
                let b1 = base.max(0) as usize;
                let b2 = (base + 1).max(0) as usize;
                let b1 = b1.min(max_a);
                let b2 = b2.min(max_a);
                let v = ((above[b1] as i32) * (32 - shift) + (above[b2] as i32) * shift + 16) >> 5;
                dst[r * w + c] = v.clamp(0, max_v) as u16;
            } else {
                let base = y_off >> 6;
                let shift = (y_off >> 1) & 0x1F;
                let b1 = base.max(0) as usize;
                let b2 = (base + 1).max(0) as usize;
                let b1 = b1.min(max_l);
                let b2 = b2.min(max_l);
                let v = ((left[b1] as i32) * (32 - shift) + (left[b2] as i32) * shift + 16) >> 5;
                dst[r * w + c] = v.clamp(0, max_v) as u16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_to_angle_base_values() {
        assert_eq!(mode_to_angle_map(IntraMode::D45Pred), 45);
        assert_eq!(mode_to_angle_map(IntraMode::D135Pred), 135);
        assert_eq!(mode_to_angle_map(IntraMode::D113Pred), 113);
        assert_eq!(mode_to_angle_map(IntraMode::D157Pred), 157);
        assert_eq!(mode_to_angle_map(IntraMode::D203Pred), 203);
        assert_eq!(mode_to_angle_map(IntraMode::D67Pred), 67);
    }

    #[test]
    fn derivative_at_45() {
        // tan(45°) = 1 → 256/1 = 256 → Q6 = 64.
        assert_eq!(DR_INTRA_DERIVATIVE[45], 64);
    }

    #[test]
    fn constant_neighbours_yield_constant() {
        let above = [128u8; 16];
        let left = [128u8; 16];
        let mut dst = vec![0u8; 16];
        for angle in [45, 67, 113, 135, 157, 203] {
            for v in dst.iter_mut() {
                *v = 0;
            }
            directional_pred(&mut dst, 4, 4, &above, &left, angle);
            for (i, &v) in dst.iter().enumerate() {
                assert_eq!(v, 128, "angle={angle} dst[{i}]={v} want 128");
            }
        }
    }

    #[test]
    fn pred_45_row_monotonic() {
        let above = [10u8, 20, 30, 40, 50, 60, 70, 80];
        let left = [10u8, 20, 30, 40];
        let mut dst = vec![0u8; 16];
        directional_pred(&mut dst, 4, 4, &above, &left, 45);
        for r in 0..4 {
            let mut prev = 0u8;
            for c in 0..4 {
                let v = dst[r * 4 + c];
                if c > 0 {
                    assert!(
                        v >= prev,
                        "row {r}: dst[{c}]={v} not monotonic after {prev}"
                    );
                }
                prev = v;
            }
        }
    }
}
