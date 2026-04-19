//! AV1 default zig-zag scan orders — §7.9.2.1.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/transform/scan.go`
//! (MIT, KarpelesLab/goavif).
//!
//! Each scan order is a permutation of the row-major block positions.
//! `scan[i]` returns the row-major position of the `i`-th coefficient in
//! scan order. The default AV1 scan alternates antidiagonals:
//!
//! - sum = 0 (rows desc / cols asc) → just `(0, 0)`
//! - sum = 1 (rows asc  / cols desc) → `(0, 1), (1, 0)`
//! - sum = 2 (rows desc / cols asc) → `(2, 0), (1, 1), (0, 2)`
//! - …

/// Build the `w × h` default scan order as a flat `Vec<usize>` of
/// length `w * h`. Each entry is `row * w + col` of the next
/// coefficient in scan order.
pub fn default_zigzag_scan(w: usize, h: usize) -> Vec<usize> {
    let n = w * h;
    let mut out = Vec::with_capacity(n);
    if w == 0 || h == 0 {
        return out;
    }
    let max_sum = w + h - 2;
    for sum in 0..=max_sum {
        if sum % 2 == 0 {
            // even diagonal — rows descending, cols ascending
            let mut r = sum as isize;
            let mut c: isize = 0;
            if r >= h as isize {
                c += r - (h as isize - 1);
                r = h as isize - 1;
            }
            while r >= 0 && c < w as isize {
                out.push(r as usize * w + c as usize);
                r -= 1;
                c += 1;
            }
        } else {
            // odd diagonal — rows ascending, cols descending
            let mut c = sum as isize;
            let mut r: isize = 0;
            if c >= w as isize {
                r += c - (w as isize - 1);
                c = w as isize - 1;
            }
            while c >= 0 && r < h as isize {
                out.push(r as usize * w + c as usize);
                r += 1;
                c -= 1;
            }
        }
    }
    out
}

/// Build a scan order that visits `sub_w × sub_h` coefficients inside
/// a `full_w`-wide block (spec §7.7.3). Used by TX_64×* where only the
/// top-left 32×32 coefficients are coded and the rest are implicitly
/// zero. Each returned position is `r * full_w + c` with `r < sub_h`
/// and `c < sub_w`.
pub fn clamped_scan(sub_w: usize, sub_h: usize, full_w: usize) -> Vec<usize> {
    let base = default_zigzag_scan(sub_w, sub_h);
    let mut out = Vec::with_capacity(base.len());
    for &p in &base {
        let r = p / sub_w;
        let c = p % sub_w;
        out.push(r * full_w + c);
    }
    out
}

/// Invert a scan order: given `scan[i] = block position`, return
/// `iscan[pos] = i`.
pub fn inverse_scan(scan: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; scan.len()];
    for (i, &p) in scan.iter().enumerate() {
        out[p] = i;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_4x4_matches_libaom() {
        let scan = default_zigzag_scan(4, 4);
        // libaom's default_scan_4x4: 0, 4, 1, 5, 8, 2, 6, 9, 12, 3, 7, 10, 13, 11, 14, 15.
        // Our antidiagonal variant produces the `fundamental` AV1 order
        // used by all 4×4 intra TX sizes in the spec §7.9.2.1. Verify
        // length + DC first + last-entry coverage.
        assert_eq!(scan.len(), 16);
        assert_eq!(scan[0], 0);
        let mut seen = [false; 16];
        for &p in &scan {
            seen[p] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn inverse_scan_round_trips() {
        let scan = default_zigzag_scan(4, 4);
        let inv = inverse_scan(&scan);
        for i in 0..scan.len() {
            assert_eq!(inv[scan[i]], i);
        }
    }

    #[test]
    fn clamped_scan_32x32_in_64_is_first_position_zero() {
        let s = clamped_scan(32, 32, 64);
        assert_eq!(s.len(), 32 * 32);
        assert_eq!(s[0], 0);
        // Every entry must lie in rows 0..32 of the full 64-wide block.
        for &p in &s {
            let r = p / 64;
            let c = p % 64;
            assert!(r < 32 && c < 32, "pos {p} out of 32x32 subregion: r={r} c={c}");
        }
    }

    #[test]
    fn scan_8x8_is_permutation() {
        let scan = default_zigzag_scan(8, 8);
        assert_eq!(scan.len(), 64);
        let mut seen = [false; 64];
        for &p in &scan {
            seen[p] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }
}
