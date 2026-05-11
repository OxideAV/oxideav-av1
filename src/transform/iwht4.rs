//! 4-point inverse Walsh-Hadamard transform — §7.13.2.10.
//!
//! Used by AV1's lossless-only coding path. The §7.13.2.10 process
//! takes a `shift` parameter; the spec's §7.7.4 reconstruction loop
//! invokes the row pass with `shift = 2` and the column pass with
//! `shift = 0`. The `UNIT_QUANT_SHIFT = 2` legacy entry point
//! (`iwht4`) is preserved for the standalone smoke-tests; callers in
//! the live reconstruction pipeline use [`iwht4_with_shift`] so
//! both passes match the spec.
//! libaom: "4-point reversible, orthonormal inverse WHT in 3.5 adds,
//! 0.5 shifts per pixel."

const UNIT_QUANT_SHIFT: u32 = 2;

/// In-place 4-point inverse WHT. `x` must have exactly 4 entries.
///
/// Uses the legacy `UNIT_QUANT_SHIFT = 2` pre-scale; equivalent to
/// [`iwht4_with_shift`]`(x, 2)`.
pub fn iwht4(x: &mut [i32; 4]) {
    iwht4_with_shift(x, UNIT_QUANT_SHIFT);
}

/// In-place 4-point inverse WHT — §7.13.2.10 with a caller-supplied
/// `shift`. Per §7.7.4 the row pass uses `shift = 2` and the column
/// pass uses `shift = 0`.
pub fn iwht4_with_shift(x: &mut [i32; 4], shift: u32) {
    let mut a = x[0] >> shift;
    let mut c = x[1] >> shift;
    let mut d = x[2] >> shift;
    let mut b = x[3] >> shift;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    c = e - c;
    a -= b;
    d += c;
    x[0] = a;
    x[1] = b;
    x[2] = c;
    x[3] = d;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{inverse_2d_spec_lossless, TxSize};

    #[test]
    fn iwht4_zero_is_zero() {
        let mut v = [0i32; 4];
        iwht4(&mut v);
        assert_eq!(v, [0, 0, 0, 0]);
    }

    #[test]
    fn iwht4_dc_constant_reconstruction() {
        // DC only; input is <<2 because UNIT_QUANT_SHIFT shifts right by 2.
        let mut v = [16i32, 0, 0, 0];
        iwht4(&mut v);
        // With a=c=d=b=0 after >>2 of zero entries, only v[0]>>2 = 4 survives:
        // a=4, c=0, d=0, b=0 → a+=c → a=4; d-=b → d=0; e=(4-0)>>1 = 2; b=2-0=2; c=2-0=2;
        // a-=b → a=2; d+=c → d=2. Output: (2, 2, 2, 2).
        assert_eq!(v, [2, 2, 2, 2]);
    }

    /// Round 47 / workspace task #786 — pin the §7.7.4 + §7.13.2.10
    /// 2D WHT (row `shift = 2`, column `shift = 0`) output for the
    /// dequantised coefficient buffer that flows through the Y TU
    /// of `y_plane_divergence_match.avif`'s 1×1 lossless YUV444 KEY
    /// frame. The buffer is derived from entropy-decoded levels
    /// `[3, 1, 0, 0, 2, -1, 0, 0, -1, 2, 0, ...]` after multiplying
    /// by the lossless DC8[0]/AC8[0] dequantiser of 4 (§7.12.2). The
    /// spec-correct residual at position (0, 0) is **2** — matching
    /// what our `inverse_2d_spec_lossless` produces. A future
    /// regression that touches the row/col shifts of the lossless
    /// path would change this number; pinning it here defends the
    /// dispatch against silent magnitude drift even if the entropy
    /// decoder upstream still feeds us a different `level` than
    /// `dav1d`. Hand-trace: row 0 = `[12, 4, 0, 0]` with shift = 2
    /// → `a=3, c=1, d=0, b=0; a+=c=4; d-=b=0; e=(4-0)>>1=2; b=2;
    /// c=2-1=1; a-=b=2; d+=c=1` → row 0 output `[2, 2, 1, 1]`. Row
    /// 1 = `[8, -4, 0, 0]` → `[1, 0, 1, 1]`. Row 2 = `[-4, 8, 0, 0]`
    /// → `[1, 0, -2, -2]`. Row 3 zero. Col 0 = `[2, 1, 1, 0]`
    /// shift=0 → `a=2, c=1, d=1, b=0; a+=c=3; d-=b=1; e=(3-1)>>1=1;
    /// b=1; c=0; a-=b=2; d+=c=1` → col output `[2, 1, 0, 1]`.
    /// Final residual at (0, 0) = `2`.
    #[test]
    fn iwht4_2d_divergence_y_tu_matches_spec() {
        let mut coeffs = [12i32, 4, 0, 0, 8, -4, 0, 0, -4, 8, 0, 0, 0, 0, 0, 0];
        inverse_2d_spec_lossless(&mut coeffs, TxSize::Tx4x4).expect("lossless 4×4 WHT");
        assert_eq!(
            coeffs[0], 2,
            "spec WHT at (0,0): expected residual 2 for the divergence Y TU"
        );
        // Pin the full 4×4 residual buffer so any drift in row/col
        // ordering surfaces, not just the DC sample.
        assert_eq!(
            coeffs,
            [2, 1, 0, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, -1, -1],
            "spec WHT residual buffer drifted vs hand-trace",
        );
    }

    /// Round 47 / workspace task #786 — pin the DC-only spec WHT
    /// path on the trivial input `[4, 0, 0, ..., 0]` (the level=1
    /// case from issue #776's pre-fix probes). Row pass `[4,0,0,0]`
    /// shift=2 → `a=1, b=c=d=0` → `[1, 0, 0, 0]`. Col 0 `[1,0,0,0]`
    /// shift=0 → `a=1, b=c=d=0` → `[1, 0, 0, 0]`. Final residual at
    /// (0, 0) = `1`. Adding to the 128 predictor recovers the
    /// `Y = 129` that round-46 first reported as a non-default
    /// sample.
    #[test]
    fn iwht4_2d_dc_only_level_1_yields_unit_residual() {
        let mut coeffs = [4i32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        inverse_2d_spec_lossless(&mut coeffs, TxSize::Tx4x4).expect("lossless 4×4 WHT");
        assert_eq!(coeffs[0], 1);
        // The other 15 cells stay zero — single-DC input cannot
        // populate AC cells through the WHT row/col shift-pair
        // (row shift=2 collapses the row pass to one DC cell of 1;
        // col shift=0 propagates only down column 0 with the same
        // dispersion `a=1, b=c=d=0` shape).
        for c in coeffs.iter().skip(1) {
            assert_eq!(*c, 0);
        }
    }
}
