//! CDEF primary + secondary filter — §7.15.3.3.
//!
//! Ported from goavif `av1/cdef/filter.go` + `filter16.go`.

use super::direction::DIRECTIONS;

/// Weights used by the primary filter at distance 1 and 2.
pub const PRIMARY_TAPS: [i32; 2] = [4, 2];

/// Weights used by the secondary (perpendicular) filter at distance 1
/// and 2.
pub const SECONDARY_TAPS: [i32; 2] = [2, 1];

/// `constrain()` nonlinearity: `sign(d) * min(|d|, max(0, t - (|d| >> s)))`
/// where `t` is the strength and `s` is the damping shift.
pub fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let mut a = diff.abs();
    let shift = (damping - msb(threshold)).max(0);
    let mut limit = threshold - (a >> shift);
    if limit < 0 {
        limit = 0;
    }
    if a > limit {
        a = limit;
    }
    if diff < 0 {
        -a
    } else {
        a
    }
}

fn msb(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    let mut n = 0;
    let mut v = x;
    while v >= 2 {
        v >>= 1;
        n += 1;
    }
    n
}

/// 8-bit CDEF primary + secondary filter on a single 8×8 block at
/// `(x, y)` in `src`. Writes the filtered block into `dst` (may alias
/// `src` but typical use passes a pre-snapshot copy as `src`).
#[allow(clippy::too_many_arguments)]
pub fn filter_block(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    dir: usize,
    pri_strength: i32,
    sec_strength: i32,
    damping: i32,
) {
    const BS: usize = 8;
    let dir_off = DIRECTIONS[dir];
    let sec_dir_a = DIRECTIONS[(dir + 2) % 8];
    let sec_dir_b = DIRECTIONS[(dir + 6) % 8];

    for r in 0..BS {
        for c in 0..BS {
            let x0 = src[(y + r) * stride + (x + c)] as i32;
            let mut sum = 0i32;
            for i in 0..2 {
                for s in [-1i32, 1] {
                    let nr = (y + r) as i32 + s * dir_off[i][0];
                    let nc = (x + c) as i32 + s * dir_off[i][1];
                    let n = sample_clamped(src, stride, nc, nr);
                    let d = n - x0;
                    sum += PRIMARY_TAPS[i] * constrain(d, pri_strength, damping);
                }
            }
            for so in [sec_dir_a, sec_dir_b].iter() {
                for i in 0..2 {
                    for s in [-1i32, 1] {
                        let nr = (y + r) as i32 + s * so[i][0];
                        let nc = (x + c) as i32 + s * so[i][1];
                        let n = sample_clamped(src, stride, nc, nr);
                        let d = n - x0;
                        sum += SECONDARY_TAPS[i] * constrain(d, sec_strength, damping);
                    }
                }
            }
            let out = x0 + ((8 + sum - i32::from(sum < 0)) >> 4);
            dst[(y + r) * stride + (x + c)] = out.clamp(0, 255) as u8;
        }
    }
}

fn sample_clamped(src: &[u8], stride: usize, col: i32, row: i32) -> i32 {
    // Matches goavif `sampleClamped`: negative coords clamp to 0, but
    // positive out-of-range reads fall through to the `CDEF_VERY_LARGE`
    // sentinel (128 for 8-bit). This preserves goavif's edge-artifact
    // behavior rather than edge-replicating.
    let row = row.max(0) as usize;
    let col = col.max(0) as usize;
    let idx = row.saturating_mul(stride).saturating_add(col);
    if idx >= src.len() {
        return 128;
    }
    src[idx] as i32
}

/// 16-bit CDEF filter. Output is clipped to `(1 << bit_depth) - 1`.
#[allow(clippy::too_many_arguments)]
pub fn filter_block16(
    dst: &mut [u16],
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    dir: usize,
    pri_strength: i32,
    sec_strength: i32,
    damping: i32,
    bit_depth: u32,
) {
    const BS: usize = 8;
    let dir_off = DIRECTIONS[dir];
    let sec_dir_a = DIRECTIONS[(dir + 2) % 8];
    let sec_dir_b = DIRECTIONS[(dir + 6) % 8];
    let max_v = ((1i32) << bit_depth) - 1;

    for r in 0..BS {
        for c in 0..BS {
            let x0 = src[(y + r) * stride + (x + c)] as i32;
            let mut sum = 0i32;
            for i in 0..2 {
                for s in [-1i32, 1] {
                    let nr = (y + r) as i32 + s * dir_off[i][0];
                    let nc = (x + c) as i32 + s * dir_off[i][1];
                    let n = sample_clamped16(src, stride, nc, nr, bit_depth);
                    let d = n - x0;
                    sum += PRIMARY_TAPS[i] * constrain(d, pri_strength, damping);
                }
            }
            for so in [sec_dir_a, sec_dir_b].iter() {
                for i in 0..2 {
                    for s in [-1i32, 1] {
                        let nr = (y + r) as i32 + s * so[i][0];
                        let nc = (x + c) as i32 + s * so[i][1];
                        let n = sample_clamped16(src, stride, nc, nr, bit_depth);
                        let d = n - x0;
                        sum += SECONDARY_TAPS[i] * constrain(d, sec_strength, damping);
                    }
                }
            }
            let out = x0 + ((8 + sum - i32::from(sum < 0)) >> 4);
            dst[(y + r) * stride + (x + c)] = out.clamp(0, max_v) as u16;
        }
    }
}

fn sample_clamped16(src: &[u16], stride: usize, col: i32, row: i32, bit_depth: u32) -> i32 {
    let row = row.max(0) as usize;
    let col = col.max(0) as usize;
    let idx = row.saturating_mul(stride).saturating_add(col);
    if idx >= src.len() {
        return 1i32 << (bit_depth - 1);
    }
    src[idx] as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrain_zero_threshold() {
        assert_eq!(constrain(42, 0, 5), 0);
    }

    #[test]
    fn constrain_small_diff_returned_unchanged() {
        assert_eq!(constrain(3, 10, 6), 3);
        assert_eq!(constrain(-3, 10, 6), -3);
    }

    #[test]
    fn constrain_large_diff_tapers() {
        let got = constrain(1000, 10, 3);
        assert!((-10..=0).contains(&got), "got {got}");
    }

    #[test]
    fn filter_constant_input_is_identity() {
        let stride = 16;
        let src = vec![100u8; 16 * 16];
        let mut dst = src.clone();
        filter_block(&mut dst, &src, stride, 4, 4, 2, 20, 10, 6);
        for r in 4..12 {
            for c in 4..12 {
                assert_eq!(dst[r * stride + c], 100);
            }
        }
    }

    #[test]
    fn filter_zero_strength_is_identity() {
        let stride = 16;
        let mut src = vec![0u8; 16 * 16];
        for (i, v) in src.iter_mut().enumerate() {
            *v = ((i * 7) & 0xFF) as u8;
        }
        let mut dst = vec![0u8; 16 * 16];
        filter_block(&mut dst, &src, stride, 4, 4, 2, 0, 0, 3);
        for r in 4..12 {
            for c in 4..12 {
                assert_eq!(dst[r * stride + c], src[r * stride + c]);
            }
        }
    }
}
