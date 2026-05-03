//! Forward-transform scaffolding for the AV1 encoder.
//!
//! Round 1 ships only the *helper* layer (`residual4x4`) so call-site
//! code can subtract a predictor block from a source block to produce
//! the signed residual a forward kernel will eventually consume.
//!
//! The forward 4×4 / 8×8 / 16×16 DCT kernels themselves are deferred
//! to round 2 because they require careful pinning against the
//! decoder's [`crate::transform`] inverse path — getting the per-pass
//! `Round2(., shift)` accounting wrong silently breaks PSNR by 5-10
//! dB. A correct forward kernel is best landed alongside the
//! coefficient encoder so the cascade can be validated against the
//! real `inverse_2d_spec`.
//!
//! Round 2+ checklist:
//!
//! 1. `fdct4` 1-D (transpose of [`crate::transform::idct4`]).
//! 2. `fdct4x4` 2-D row + column with the matching `Round2` shifts.
//! 3. Cascade test `idct4x4(quant(fdct4x4(x))) ≈ x` within ±N LSB.

/// Subtract a `4×4` predictor from a `4×4` source pixel block to
/// produce a 16-element signed residual buffer.
///
/// Both inputs use the standard `(stride, data)` representation. The
/// returned array is row-major: `out[r * 4 + c]` is the residual at
/// pixel `(row=r, col=c)`.
pub fn residual4x4(src: &[u8], src_stride: usize, pred: &[u8], pred_stride: usize) -> [i32; 16] {
    let mut out = [0i32; 16];
    for r in 0..4 {
        for c in 0..4 {
            out[r * 4 + c] = src[r * src_stride + c] as i32 - pred[r * pred_stride + c] as i32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residual4x4_subtracts_predictor() {
        let src = [200u8; 16];
        let pred = [128u8; 16];
        let r = residual4x4(&src, 4, &pred, 4);
        for v in &r {
            assert_eq!(*v, 72);
        }
    }

    #[test]
    fn residual4x4_handles_negative_residual() {
        let src = [50u8; 16];
        let pred = [128u8; 16];
        let r = residual4x4(&src, 4, &pred, 4);
        for v in &r {
            assert_eq!(*v, -78);
        }
    }

    #[test]
    fn residual4x4_respects_strides() {
        // 4×4 source inside an 8-wide buffer.
        let mut src = vec![0u8; 32]; // 4 rows × 8 cols
        for r in 0..4 {
            for c in 0..4 {
                src[r * 8 + c] = (r as u8 * 16) + c as u8;
            }
        }
        let pred = [0u8; 16];
        let r = residual4x4(&src, 8, &pred, 4);
        for row in 0..4 {
            for col in 0..4 {
                assert_eq!(r[row * 4 + col], (row as i32 * 16) + col as i32);
            }
        }
    }
}
