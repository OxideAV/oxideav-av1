//! Auto-regressive grain shaping (spec §7.20.3.3) + seeded raw noise
//! template generator.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/filmgrain/ar.go`
//! (MIT, KarpelesLab/goavif).

use super::rng::Rng;

/// Fill a `rows × cols` buffer with signed grain samples drawn from a
/// seeded LFSR. Output values are in `[-128, 127]`, stored as `i16`
/// so a subsequent AR pass (which runs in wider precision) can
/// accumulate into them.
pub fn generate_grain_template(cols: usize, rows: usize, mut seed: u16) -> Vec<i16> {
    let mut out = vec![0i16; cols * rows];
    if seed == 0 {
        seed = 1;
    }
    let mut rng = Rng::new(seed);
    for v in out.iter_mut() {
        *v = rng.byte() as i16;
    }
    out
}

/// Shape a generated grain template in place using an AR filter of
/// the given `lag`. The filter taps form an L-shaped neighbourhood of
/// order `lag` covering all positions `(dy, dx)` with:
///
/// - `dy ∈ [-lag, 0]`, `dx ∈ [-lag, lag]`
/// - `(dy, dx)` precedes `(0, 0)` in raster order
/// - not `(dy == 0 && dx == 0)`
///
/// `coeffs` is laid out in the same order the spec transmits them
/// (top-to-bottom, left-to-right within each row), `(2·lag + 1)·lag +
/// lag` entries total. A length mismatch is a no-op.
///
/// `shift` is `ar_coeff_shift` (6..=9) — the weighted sum is rounded
/// by `1 << (shift - 1)` then right-shifted by `shift` before being
/// added to the grain sample.
pub fn apply_ar(grain: &mut [i16], cols: usize, rows: usize, lag: usize, coeffs: &[i8], shift: u32) {
    if lag == 0 {
        return;
    }
    let taps = (2 * lag + 1) * lag + lag;
    if coeffs.len() != taps {
        return;
    }
    let shift = if shift == 0 { 7 } else { shift };
    let round: i32 = 1 << (shift - 1);

    for r in lag..rows {
        for c in lag..cols.saturating_sub(lag) {
            let mut sum: i32 = 0;
            let mut k = 0usize;
            // Rows strictly above r.
            for dy in -(lag as i32)..0 {
                for dx in -(lag as i32)..=(lag as i32) {
                    let row = (r as i32 + dy) as usize;
                    let col = (c as i32 + dx) as usize;
                    sum += (coeffs[k] as i32) * (grain[row * cols + col] as i32);
                    k += 1;
                }
            }
            // Same row, strictly left of c.
            for dx in -(lag as i32)..0 {
                let col = (c as i32 + dx) as usize;
                sum += (coeffs[k] as i32) * (grain[r * cols + col] as i32);
                k += 1;
            }
            let v = grain[r * cols + c] as i32 + ((sum + round) >> shift);
            grain[r * cols + c] = v.clamp(-2048, 2047) as i16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_deterministic_same_seed() {
        let a = generate_grain_template(16, 16, 0x1234);
        let b = generate_grain_template(16, 16, 0x1234);
        assert_eq!(a, b);
    }

    #[test]
    fn template_range_is_signed_byte() {
        let g = generate_grain_template(32, 32, 0xABCD);
        for &v in &g {
            assert!((-128..=127).contains(&v), "sample {v} escaped signed-byte range");
        }
    }

    #[test]
    fn zero_lag_is_noop() {
        let mut g = generate_grain_template(8, 8, 0xBEEF);
        let save = g.clone();
        apply_ar(&mut g, 8, 8, 0, &[], 6);
        assert_eq!(g, save);
    }

    #[test]
    fn coeff_count_mismatch_is_noop() {
        let mut g = generate_grain_template(8, 8, 0xCAFE);
        let save = g.clone();
        // lag=2 expects (2*2+1)*2 + 2 = 12 coeffs; we pass 3.
        apply_ar(&mut g, 8, 8, 2, &[1, 2, 3], 7);
        assert_eq!(g, save);
    }

    #[test]
    fn zero_coeffs_does_not_shape() {
        let mut g = generate_grain_template(16, 16, 0x5555);
        let save = g.clone();
        apply_ar(&mut g, 16, 16, 3, &[0; 24], 7);
        assert_eq!(g, save);
    }

    #[test]
    fn non_zero_coeffs_change_something() {
        let mut g = generate_grain_template(16, 16, 0x9999);
        let save = g.clone();
        apply_ar(&mut g, 16, 16, 1, &[10, 20, 30, 40], 6);
        let diffs = g.iter().zip(save.iter()).filter(|(a, b)| a != b).count();
        assert!(diffs > 0);
    }

    #[test]
    fn results_stay_in_bounds() {
        let mut g = generate_grain_template(32, 32, 0xA5A5_u16);
        let coeffs = vec![127i8; (2 * 3 + 1) * 3 + 3];
        apply_ar(&mut g, 32, 32, 3, &coeffs, 6);
        for &v in &g {
            assert!((-2048..=2047).contains(&v));
        }
    }
}
