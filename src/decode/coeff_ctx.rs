//! AV1 coefficient context derivation — §6.10.6 / §6.10.7.
//!
//! Coefficients are decoded in reverse scan order: first the
//! coefficient at position `scan[eob-1]`, then `scan[eob-2]`, and so
//! on down to the DC at `scan[0]`. The `sig_coef_ctx` used to select a
//! CDF for each coefficient's base level is derived from:
//!
//! 1. A "neighbor sum template" over 5 positions adjacent to the
//!    current one (toward higher scan indices), clamped to levels in
//!    0..=3 and summed.
//! 2. A position-dependent offset from `nz_map_ctx_offset[tx_size]`.
//!
//! This file implements the 2D-scan template used by all square
//! blocks and the 4×4 / 8×8 sizes that Phase 3 will exercise. Phase 2
//! does not invoke these helpers yet (the tile decoder exits with
//! `Error::Unsupported` before any coefficient is decoded), but they
//! are ported here so Phase 3 can wire them up unchanged.

/// 5 neighbor positions sampled for a 2D scan's coefficient context:
/// `(dr, dc)` offsets from the current `(r, c)`.
const TEMPLATE_2D_OFFSETS: [(i32, i32); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];

/// `sig_coef_ctx` for a coefficient at block position `(r, c)` in a
/// block of width `w` and height `h`.
///
/// `abs_levels[r*w + c]` holds the partial absolute-level grid decoded
/// so far, clamped to 0..=3. `nz_map_offset[scan_idx]` is the
/// position-specific offset from the per-TX-size table in
/// `cdfs::nz_map_ctx_offset_*`.
pub fn sig_coef_ctx_2d(
    r: i32,
    c: i32,
    w: i32,
    h: i32,
    abs_levels: &[i8],
    nz_map_offset: &[i8],
    scan_idx: usize,
) -> i32 {
    if scan_idx == 0 {
        return 0;
    }
    let mut stats: i32 = 0;
    for (dr, dc) in TEMPLATE_2D_OFFSETS {
        let rr = r + dr;
        let cc = c + dc;
        if rr < h && cc < w {
            let idx = (rr * w + cc) as usize;
            let mut v = abs_levels[idx] as i32;
            if v > 3 {
                v = 3;
            }
            stats += v;
        }
    }
    let mut ctx_base = (stats + 1) >> 1;
    if ctx_base > 4 {
        ctx_base = 4;
    }
    ctx_base + nz_map_offset[scan_idx] as i32
}

/// `coeff_br_ctx` for a coefficient at position `(r, c)` given the
/// partial decoded `abs_levels`. Used when reading additional
/// base-range levels beyond `NUM_BASE_LEVELS` (2).
///
/// Spec §6.10.7 uses a 3-position template; the level bias applied
/// per position is `max(0, min(3, abs - 3))`. The position-based
/// remap the spec adds on top is deferred (full Phase 3 scope).
pub fn level_ctx(r: i32, c: i32, w: i32, h: i32, abs_levels: &[i8]) -> i32 {
    let mut stats: i32 = 0;
    for (dr, dc) in [(0i32, 1i32), (1, 0), (1, 1)] {
        let rr = r + dr;
        let cc = c + dc;
        if rr < h && cc < w {
            let idx = (rr * w + cc) as usize;
            let mut v = abs_levels[idx] as i32;
            if v > 3 {
                v -= 3;
                if v > 3 {
                    v = 3;
                }
                stats += v;
            }
        }
    }
    let mut ctx = (stats + 1) >> 1;
    if ctx > 3 {
        ctx = 3;
    }
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sig_coef_ctx_dc_is_zero() {
        let levels = [0i8; 16];
        let offset = [0i8, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21];
        assert_eq!(sig_coef_ctx_2d(0, 0, 4, 4, &levels, &offset, 0), 0);
    }

    #[test]
    fn sig_coef_ctx_zero_neighbors_uses_position_offset() {
        // For a 4×4 block with all-zero neighbors, stats = 0, ctxBase = 0
        // so the resulting context == nz_map_offset[scan_idx].
        let levels = [0i8; 16];
        let offset = [0i8, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21];
        // scan_idx=1 → position (0,1) → offset 1 → ctx = 1
        assert_eq!(sig_coef_ctx_2d(0, 1, 4, 4, &levels, &offset, 1), 1);
        // scan_idx=7 → offset 21 → ctx = 21
        assert_eq!(sig_coef_ctx_2d(1, 3, 4, 4, &levels, &offset, 7), 21);
    }

    #[test]
    fn sig_coef_ctx_clamps_stats() {
        // Neighbors with huge values should clamp per-neighbor to 3.
        // Three neighbors at 100 → stats = 9 → ctxBase = (9+1)/2 = 5 → clamp to 4
        // → +offset[1]=1 → 5.
        let mut levels = [0i8; 16];
        // Positions (1,1), (0,1), (1,0) in the 4-wide grid.
        levels[5] = 100;
        levels[1] = 100;
        levels[4] = 100;
        let offset = [0i8, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21];
        assert_eq!(sig_coef_ctx_2d(0, 0, 4, 4, &levels, &offset, 1), 5);
    }

    #[test]
    fn level_ctx_returns_zero_for_all_below_base() {
        let levels = [0i8; 16];
        assert_eq!(level_ctx(0, 0, 4, 4, &levels), 0);
    }
}
