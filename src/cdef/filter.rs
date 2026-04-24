//! CDEF primary + secondary filter — §7.15.3.3.

use super::direction::DIRECTIONS;

/// Legacy single primary tap pair (`[4, 2]`). Retained for back-compat
/// tests; the spec-correct filter selects between `[4, 2]` (strength
/// bit 0) and `[3, 3]` (strength bit 1) via [`CDEF_PRI_TAPS`].
pub const PRIMARY_TAPS: [i32; 2] = [4, 2];

/// Legacy single secondary tap pair (`[2, 1]`). See [`CDEF_SEC_TAPS`]
/// for the spec's 2D table.
pub const SECONDARY_TAPS: [i32; 2] = [2, 1];

/// `Cdef_Pri_Taps[(priStr >> coeffShift) & 1][k]` per spec §7.15.3.
/// Row 0 (`priStr` LSB 0) is `[4, 2]`; row 1 is `[3, 3]`.
pub const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];

/// `Cdef_Sec_Taps[(priStr >> coeffShift) & 1][k]` per spec §7.15.3.
/// Both rows are `[2, 1]` — the spec keeps the 2D shape to mirror
/// `Cdef_Pri_Taps`.
pub const CDEF_SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];

/// `Cdef_Uv_Dir[subX][subY][yDir]` — chroma direction remap per spec
/// §7.15.1. For 4:4:4 (both subs zero) UV copies yDir; the 4:2:0 /
/// 4:2:2 mappings collapse the 8-way ring onto a chroma-appropriate
/// subset.
pub const CDEF_UV_DIR: [[[usize; 8]; 2]; 2] = [
    [
        [0, 1, 2, 3, 4, 5, 6, 7], // subX=0, subY=0 (4:4:4)
        [1, 2, 2, 2, 3, 4, 6, 0], // subX=0, subY=1
    ],
    [
        [7, 0, 2, 4, 5, 6, 6, 6], // subX=1, subY=0 (4:2:2)
        [0, 1, 2, 3, 4, 5, 6, 7], // subX=1, subY=1 (4:2:0)
    ],
];

/// Apply the spec's §7.15.1 variance-strength adjustment to a base
/// primary strength. Given the raw direction-search variance, returns
/// the adjusted `priStr` used by the filter (spec steps 4-5):
///
/// ```text
/// varStr = (var >> 6) ? min(FloorLog2(var >> 6), 12) : 0
/// priStr' = var ? (priStr * (4 + varStr) + 8) >> 4 : 0
/// ```
pub fn adjust_pri_strength(pri_str: i32, var: i32) -> i32 {
    if var == 0 {
        return 0;
    }
    let v6 = var >> 6;
    let var_str = if v6 != 0 {
        floor_log2(v6).min(12)
    } else {
        0
    };
    (pri_str * (4 + var_str) + 8) >> 4
}

fn floor_log2(x: i32) -> i32 {
    if x <= 0 {
        return 0;
    }
    31 - (x as u32).leading_zeros() as i32
}

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

/// Spec-exact §7.15.3 CDEF filter on an 8x8 block. Uses the 2D
/// `Cdef_Pri_Taps / Cdef_Sec_Taps` tables keyed by
/// `(priStr >> coeff_shift) & 1`, applies the spec's min/max sample
/// clipping, and honours the filter-region guard via `available`.
///
/// `available(row, col)` returns `true` iff the sample at the given
/// plane-local coordinate is inside the current filter region (spec's
/// `is_inside_filter_region`). Out-of-region samples are skipped (not
/// folded into min/max).
#[allow(clippy::too_many_arguments)]
pub fn filter_block_spec<A>(
    dst: &mut [u8],
    src: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    dir: usize,
    pri_strength: i32,
    sec_strength: i32,
    damping: i32,
    coeff_shift: u32,
    available: A,
) where
    A: Fn(i32, i32) -> bool,
{
    const BS: usize = 8;
    let pri_idx = ((pri_strength >> coeff_shift) & 1) as usize;
    let pri_taps = CDEF_PRI_TAPS[pri_idx];
    let sec_taps = CDEF_SEC_TAPS[pri_idx];

    let dir_off = DIRECTIONS[dir];
    let sec_dir_a = DIRECTIONS[(dir + 2) % 8];
    let sec_dir_b = DIRECTIONS[(dir + 6) % 8];

    for r in 0..BS {
        for c in 0..BS {
            let x0 = src[(y + r) * stride + (x + c)] as i32;
            let mut sum = 0i32;
            let mut mx = x0;
            let mut mn = x0;
            // Primary taps along `dir`.
            for i in 0..2 {
                for s in [-1i32, 1] {
                    let nr = (y + r) as i32 + s * dir_off[i][0];
                    let nc = (x + c) as i32 + s * dir_off[i][1];
                    if available(nr, nc) {
                        let n = src[nr as usize * stride + nc as usize] as i32;
                        sum += pri_taps[i] * constrain(n - x0, pri_strength, damping);
                        if n > mx {
                            mx = n;
                        }
                        if n < mn {
                            mn = n;
                        }
                    }
                }
            }
            // Secondary taps along `(dir +/- 2) & 7`.
            for so in [sec_dir_a, sec_dir_b].iter() {
                for i in 0..2 {
                    for s in [-1i32, 1] {
                        let nr = (y + r) as i32 + s * so[i][0];
                        let nc = (x + c) as i32 + s * so[i][1];
                        if available(nr, nc) {
                            let n = src[nr as usize * stride + nc as usize] as i32;
                            sum += sec_taps[i] * constrain(n - x0, sec_strength, damping);
                            if n > mx {
                                mx = n;
                            }
                            if n < mn {
                                mn = n;
                            }
                        }
                    }
                }
            }
            let out = x0 + ((8 + sum - i32::from(sum < 0)) >> 4);
            dst[(y + r) * stride + (x + c)] = out.clamp(mn, mx).clamp(0, 255) as u8;
        }
    }
}

/// 16-bit counterpart of [`filter_block_spec`].
#[allow(clippy::too_many_arguments)]
pub fn filter_block_spec16<A>(
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
    coeff_shift: u32,
    available: A,
) where
    A: Fn(i32, i32) -> bool,
{
    const BS: usize = 8;
    let pri_idx = ((pri_strength >> coeff_shift) & 1) as usize;
    let pri_taps = CDEF_PRI_TAPS[pri_idx];
    let sec_taps = CDEF_SEC_TAPS[pri_idx];

    let dir_off = DIRECTIONS[dir];
    let sec_dir_a = DIRECTIONS[(dir + 2) % 8];
    let sec_dir_b = DIRECTIONS[(dir + 6) % 8];
    let max_v = ((1i32) << bit_depth) - 1;

    for r in 0..BS {
        for c in 0..BS {
            let x0 = src[(y + r) * stride + (x + c)] as i32;
            let mut sum = 0i32;
            let mut mx = x0;
            let mut mn = x0;
            for i in 0..2 {
                for s in [-1i32, 1] {
                    let nr = (y + r) as i32 + s * dir_off[i][0];
                    let nc = (x + c) as i32 + s * dir_off[i][1];
                    if available(nr, nc) {
                        let n = src[nr as usize * stride + nc as usize] as i32;
                        sum += pri_taps[i] * constrain(n - x0, pri_strength, damping);
                        if n > mx {
                            mx = n;
                        }
                        if n < mn {
                            mn = n;
                        }
                    }
                }
            }
            for so in [sec_dir_a, sec_dir_b].iter() {
                for i in 0..2 {
                    for s in [-1i32, 1] {
                        let nr = (y + r) as i32 + s * so[i][0];
                        let nc = (x + c) as i32 + s * so[i][1];
                        if available(nr, nc) {
                            let n = src[nr as usize * stride + nc as usize] as i32;
                            sum += sec_taps[i] * constrain(n - x0, sec_strength, damping);
                            if n > mx {
                                mx = n;
                            }
                            if n < mn {
                                mn = n;
                            }
                        }
                    }
                }
            }
            let out = x0 + ((8 + sum - i32::from(sum < 0)) >> 4);
            dst[(y + r) * stride + (x + c)] = out.clamp(mn, mx).clamp(0, max_v) as u16;
        }
    }
}

fn sample_clamped(src: &[u8], stride: usize, col: i32, row: i32) -> i32 {
    // Negative coords clamp to 0, but positive out-of-range reads
    // fall through to the `CDEF_VERY_LARGE` sentinel (128 for 8-bit).
    // This preserves libaom's edge-artifact behavior rather than
    // edge-replicating.
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

    #[test]
    fn adjust_pri_strength_zero_var_kills() {
        assert_eq!(adjust_pri_strength(40, 0), 0);
    }

    #[test]
    fn adjust_pri_strength_varstr_clamp() {
        // var = 1 -> v6 = 0 -> varStr = 0 -> (40 * 4 + 8) >> 4 = 10.
        assert_eq!(adjust_pri_strength(40, 1), 10);
        // var = 64 -> v6 = 1 -> varStr = 0 -> same 10.
        assert_eq!(adjust_pri_strength(40, 64), 10);
        // var = 1 << 20 -> v6 = 16384 -> floor_log2 = 14, clamped to 12.
        // -> (40 * 16 + 8) >> 4 = 40.
        assert_eq!(adjust_pri_strength(40, 1 << 20), 40);
    }

    #[test]
    fn filter_block_spec_out_of_region_samples_skipped() {
        // With all neighbours unavailable the filter leaves the block
        // unchanged (sum stays 0; min=max=x0; out = x0).
        let stride = 16;
        let src: Vec<u8> = (0..16 * 16).map(|i| ((i * 13) & 0xFF) as u8).collect();
        let mut dst = src.clone();
        filter_block_spec(
            &mut dst,
            &src,
            stride,
            4,
            4,
            2,
            20,
            10,
            6,
            0,
            |_, _| false,
        );
        for r in 4..12 {
            for c in 4..12 {
                assert_eq!(dst[r * stride + c], src[r * stride + c]);
            }
        }
    }

    #[test]
    fn filter_block_spec_constant_interior_unchanged() {
        let stride = 24;
        let src = vec![100u8; 24 * 24];
        let mut dst = src.clone();
        filter_block_spec(
            &mut dst,
            &src,
            stride,
            8,
            8,
            2,
            20,
            10,
            6,
            0,
            |r, c| r >= 0 && c >= 0 && (r as usize) < 24 && (c as usize) < 24,
        );
        // Neighbourhood = 100, x0 = 100 -> constrain(0,...) = 0 -> sum=0
        // -> out = 100, clipped to min=max=100.
        for r in 8..16 {
            for c in 8..16 {
                assert_eq!(dst[r * stride + c], 100);
            }
        }
    }

    #[test]
    fn filter_block_spec_clipped_to_neighbourhood() {
        // Single bright sample surrounded by 100s — CDEF cannot pull
        // the centre above the neighbourhood max.
        let stride = 24;
        let mut src = vec![100u8; 24 * 24];
        for r in 8..16 {
            for c in 8..16 {
                src[r * stride + c] = if (r == 12) && (c == 12) { 200 } else { 100 };
            }
        }
        let mut dst = src.clone();
        filter_block_spec(
            &mut dst,
            &src,
            stride,
            8,
            8,
            2,
            60,
            30,
            6,
            0,
            |r, c| r >= 0 && c >= 0 && (r as usize) < 24 && (c as usize) < 24,
        );
        for r in 8..16 {
            for c in 8..16 {
                let v = dst[r * stride + c];
                assert!(v <= 200, "dst[{r},{c}]={v} exceeded neighbourhood max");
                assert!(v >= 100, "dst[{r},{c}]={v} below neighbourhood min");
            }
        }
    }
}
