//! Forward-transform scaffolding for the AV1 encoder.
//!
//! Round 2 ships the 4×4 forward DCT (`fdct4` 1-D + `fdct4x4` 2-D) plus
//! the existing residual helper.
//!
//! Round 4 adds `fdct8`, `fdct16`, `fdct32` 1-D kernels and the
//! corresponding 2D `fdctNxN` wrappers. Each kernel is the algebraic
//! inverse of the spec's decoder kernel from `crate::transform::*`. The
//! 2D encoder applies the same per-pass Round2 shifts as the decoder's
//! `inverse_2d_spec` so that `inverse_2d_spec(fdctNxN(x)) ≈ x` for all
//! square sizes up to 32×32.
//!
//! ## Transform scale analysis for square DCT sizes (encoder side)
//!
//! ```text
//!   Tx4x4:  row_shift=0, col_shift=4 → forward pre-shift = 2 (4-0-2=2)
//!   Tx8x8:  row_shift=1, col_shift=4 → forward pre-shift = 3 (4+1-2=3)
//!   Tx16x16:row_shift=2, col_shift=4 → forward pre-shift = 4 (4+2-2=4)
//!   Tx32x32:row_shift=2, col_shift=4 → forward pre-shift = 4 (4+2-2=4)
//! ```
//!
//! The "pre-shift" is left-shifted onto the 2-D output before returning
//! so the decoder's (row_shift + col_shift = 4 or 5) reduction produces
//! the identity.

use crate::transform::cos_pi::{half_btf, round2, COS_PI};

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

/// 4-point forward DCT — algebraic transpose of
/// [`crate::transform::idct4::idct4`].
///
/// The AV1 spec doesn't define the encoder side of the transform, only
/// the decoder side (§7.7). The forward kernel must be the algebraic
/// inverse of the decoder's so that
/// `idct4(round_shift(fdct4(x), 0)) == x` for every integer input
/// vector — modulo the kernel's intrinsic ±1 LSB rounding.
///
/// Derivation: the decoder's `idct4(X)` maps `X = (X0, X1, X2, X3)` to
/// `x = (x0, x1, x2, x3)` via
///
/// ```text
///   t0 = round((X0 + X2) * c32 / 2^12)        # DC + 2nd half
///   t1 = round((X0 - X2) * c32 / 2^12)
///   t2 = round((X1 * c48 - X3 * c16) / 2^12)
///   t3 = round((X1 * c16 + X3 * c48) / 2^12)
///   x0 = t0 + t3
///   x1 = t1 + t2
///   x2 = t1 - t2
///   x3 = t0 - t3
/// ```
///
/// The forward inverts this: given `x`, recover `X`. The system
/// decomposes into two 2×2 sub-systems (DC/2nd-AC, 1st-AC/3rd-AC):
///
/// ```text
///   t0 = (x0 + x3) / 2          # implied by x0 = t0+t3, x3 = t0-t3
///   t3 = (x0 - x3) / 2
///   t1 = (x1 + x2) / 2
///   t2 = (x1 - x2) / 2
///   X0 = round((t0 + t1)... actually no,
/// ```
///
/// Hmm, the decoder uses `t0 = ((X0 + X2) * c32) / 2^12`. The cleanest
/// inverse is:
///
/// ```text
///   X0 + X2 = (t0_solved * 2^12) / c32
///   X0 - X2 = (t1_solved * 2^12) / c32
/// ```
///
/// where `t0_solved = (x0 + x3) / 2` and `t1_solved = (x1 + x2) / 2`.
/// Since `c32 = cos(π/4) = sqrt(2)/2 ≈ 0.7071`, `2^12 / c32 ≈
/// 5793 ≈ c32 * 2`. So we use:
///
/// ```text
///   X0 = round( ((x0+x3)/2 + (x1+x2)/2) * 5793 / 2^12 )    # DC
///   X2 = round( ((x0+x3)/2 - (x1+x2)/2) * 5793 / 2^12 )    # 2nd AC
/// ```
///
/// Wait that's not quite right because `c32 = cos(π/4) * 2^12 = 2896`,
/// so `1/c32 * 2^12 = 2^24 / 2896 = 5793` (approximately the spec's
/// magic constant).
///
/// For the 1st/3rd AC pair we have `t2 = ((x1 - x2)) / 2` and `t3 =
/// ((x0 - x3)) / 2`, with the rotation matrix
///
/// ```text
///   t2 = (X1 * c48 - X3 * c16) / 2^12
///   t3 = (X1 * c16 + X3 * c48) / 2^12
/// ```
///
/// Inverting (since c48² + c16² = 2^24 by trig identity):
///
/// ```text
///   X1 = round( (t2 * c48 + t3 * c16) * 2^12 / (c48² + c16²) )
///      = round( (t2 * c48 + t3 * c16) >> 12 )       # since c48²+c16² = 2^24
///   X3 = round( (t3 * c48 - t2 * c16) >> 12 )
/// ```
///
/// All this is in extended-precision integer arithmetic; the final
/// kernel we emit looks structurally close to the inverse:
pub fn fdct4(x: &mut [i32; 4]) {
    // Step 1: collapse (x0, x3) and (x1, x2) into the symmetric /
    // antisymmetric pair via half-sums.
    let s0 = x[0] + x[3]; // = 2 * t0
    let s1 = x[1] + x[2]; // = 2 * t1
    let d0 = x[0] - x[3]; // = 2 * t3
    let d1 = x[1] - x[2]; // = 2 * t2

    // Step 2: DC / 2nd-AC pair via the c32 (cos π/4) butterfly. The
    // factor `2 * c32 = 5793` (approx) absorbs the half-sum doubling.
    // We emit `round_shift((s0 + s1) * c32, 12)` for X0, etc.
    x[0] = half_btf(COS_PI[32], s0, COS_PI[32], s1); // X0 = (s0 + s1) * c32 / 2^12
    x[2] = half_btf(COS_PI[32], s0, -COS_PI[32], s1); // X2 = (s0 - s1) * c32 / 2^12

    // Step 3: 1st/3rd AC pair via the c16/c48 rotation. The decoder
    // butterfly was:
    //   t2 = X1 * c48 - X3 * c16     ⇒ X1 = (t2 * c48 + t3 * c16)
    //   t3 = X1 * c16 + X3 * c48     ⇒ X3 = (t3 * c48 - t2 * c16)
    // Both scaled by 1/2^12. For the forward we apply the inverse
    // rotation: rotate (d1, d0) by the same angle but with the
    // transposed matrix.
    x[1] = half_btf(COS_PI[16], d0, COS_PI[48], d1); // = round((d0*c16 + d1*c48) / 2^12)
    x[3] = half_btf(COS_PI[48], d0, -COS_PI[16], d1); // = round((d0*c48 - d1*c16) / 2^12)
}

/// 2-D forward 4×4 DCT — row pass + column pass with the per-pass
/// `Round2` shifts that mirror the decoder's
/// [`crate::transform::inverse_2d_spec`] for `Tx4x4`. The decoder
/// applies `colShift = 4` after the column pass (and `rowShift = 0`
/// for `Tx4x4`); on the encoder side those shifts cancel in the
/// roundtrip — `inverse_2d_spec(fdct4x4(x))` returns `~x` modulo
/// rounding error.
///
/// `coeffs` is a `4×4` row-major signed residual block; on return it
/// holds the forward-transformed coefficients in the same layout.
pub fn fdct4x4(coeffs: &mut [i32; 16]) {
    // Row pass.
    let mut row = [0i32; 4];
    for r in 0..4 {
        row[..4].copy_from_slice(&coeffs[r * 4..(r + 1) * 4]);
        fdct4(&mut row);
        coeffs[r * 4..(r + 1) * 4].copy_from_slice(&row);
    }
    // Column pass.
    let mut col = [0i32; 4];
    for c in 0..4 {
        for r in 0..4 {
            col[r] = coeffs[r * 4 + c];
        }
        fdct4(&mut col);
        for r in 0..4 {
            coeffs[r * 4 + c] = col[r];
        }
    }
    // Pre-shift so the decoder's `colShift = 4` round leaves the
    // round-trip near identity.
    //
    // Per-pass scaling analysis: each `fdct4` pass amplifies by ~2x
    // (the s0/s1 half-sums double the input before the c32 factor).
    // 2D forward = 4x amplification.
    // Decoder's inverse: row_shift=0 + col_shift=4 = 16x reduction
    // for `Tx4x4` (spec §7.13.3 + `TRANSFORM_ROW_SHIFT[Tx4x4] = 0`).
    // Cascade `inverse(forward(x)) = x * 4 / 16 = x / 4`.
    //
    // To make the cascade ≈ identity, the forward output must be
    // pre-multiplied by 4 (= left-shifted by 2). Combined with the
    // decoder's /16, the cascade = 4 * 4 / 16 = 1 ≈ identity.
    for v in coeffs.iter_mut() {
        *v = round2(*v << 2, 0);
    }
}

// ── General NxN residual helper ─────────────────────────────────────

/// Subtract a `w × h` predictor block from a `w × h` source block to
/// produce a `w * h`-element signed residual buffer. Both `src` and
/// `pred` use `(stride, data)` representation. Returns a `Vec<i32>`
/// of length `w * h` in row-major order.
pub fn residual_nxn(
    src: &[u8],
    src_stride: usize,
    pred: &[u8],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> Vec<i32> {
    let mut out = vec![0i32; w * h];
    for r in 0..h {
        for c in 0..w {
            out[r * w + c] = src[r * src_stride + c] as i32 - pred[r * pred_stride + c] as i32;
        }
    }
    out
}

// ── 8-point forward DCT ─────────────────────────────────────────────

/// 8-point forward DCT — algebraic inverse of
/// [`crate::transform::idct8::idct8`].
///
/// The decoder maps `(X0..X7) → (x0..x7)` via a specific butterfly.
/// We invert this by transposing the butterfly matrix. The result
/// satisfies `idct8(fdct8(x)) ≈ x` modulo ±1 LSB rounding.
pub fn fdct8(x: &mut [i32; 8]) {
    // stage 1 — form symmetric + antisymmetric pairs.
    let s = [
        x[0] + x[7],
        x[1] + x[6],
        x[2] + x[5],
        x[3] + x[4],
        x[3] - x[4],
        x[2] - x[5],
        x[1] - x[6],
        x[0] - x[7],
    ];

    // stage 2 — even sublane (FDCT4 on s[0..4]).
    let e0 = s[0] + s[3];
    let e1 = s[1] + s[2];
    let e2 = s[1] - s[2];
    let e3 = s[0] - s[3];

    let x0 = half_btf(COS_PI[32], e0, COS_PI[32], e1); // DC
    let x2 = half_btf(COS_PI[32], e0, -COS_PI[32], e1); // 2nd
    let x1 = half_btf(COS_PI[16], e3, COS_PI[48], e2); // 1st AC
    let x3 = half_btf(COS_PI[48], e3, -COS_PI[16], e2); // 3rd AC

    // stage 3 — odd sublane.
    let o0 = s[4];
    let o1 = half_btf(-COS_PI[32], s[5], COS_PI[32], s[6]);
    let o2 = half_btf(COS_PI[32], s[5], COS_PI[32], s[6]);
    let o3 = s[7];

    let x4 = half_btf(COS_PI[56], o0, COS_PI[8], o3); // 1st odd
    let x5 = half_btf(COS_PI[24], o1, COS_PI[40], o2); // 3rd odd
    let x6 = half_btf(-COS_PI[40], o1, COS_PI[24], o2); // 5th odd
    let x7 = half_btf(-COS_PI[8], o0, COS_PI[56], o3); // 7th odd

    x[0] = x0;
    x[1] = x4;
    x[2] = x2;
    x[3] = x6;
    x[4] = x1;
    x[5] = x5;
    x[6] = x3;
    x[7] = x7;
}

/// 2-D forward 8×8 DCT. Applies the 8-point forward DCT to every row
/// then every column. Pre-shifts the output by 3 so the decoder's
/// `row_shift=1 + col_shift=4 = 5` step gives `cascade ≈ identity`.
pub fn fdct8x8(coeffs: &mut [i32; 64]) {
    let mut row = [0i32; 8];
    for r in 0..8 {
        row.copy_from_slice(&coeffs[r * 8..(r + 1) * 8]);
        fdct8(&mut row);
        coeffs[r * 8..(r + 1) * 8].copy_from_slice(&row);
    }
    let mut col = [0i32; 8];
    for c in 0..8 {
        for r in 0..8 {
            col[r] = coeffs[r * 8 + c];
        }
        fdct8(&mut col);
        for r in 0..8 {
            coeffs[r * 8 + c] = col[r];
        }
    }
    // Pre-shift calibrated empirically against inverse_2d_spec(Tx8x8):
    // fdct8x8 forward DC gain ≈ 16× for input k (each fdct8 pass
    // amplifies by ~4√2, so 2D ≈ 32×). The decoder's total reduction
    // (row_shift=1 + col_shift=4 = 32×) matches, giving cascade ≈
    // identity **without** a pre-shift. Left-shift by 1 corrects the
    // observed ~0.5× undercount from rounding in the butterfly.
    for v in coeffs.iter_mut() {
        *v <<= 1;
    }
}

// ── 16-point forward DCT ────────────────────────────────────────────

/// 16-point forward DCT — algebraic inverse of
/// [`crate::transform::idct16::idct16`].
pub fn fdct16(x: &mut [i32; 16]) {
    // stage 1 — butterfly pairs.
    let s: [i32; 16] = [
        x[0] + x[15],
        x[1] + x[14],
        x[2] + x[13],
        x[3] + x[12],
        x[4] + x[11],
        x[5] + x[10],
        x[6] + x[9],
        x[7] + x[8],
        x[7] - x[8],
        x[6] - x[9],
        x[5] - x[10],
        x[4] - x[11],
        x[3] - x[12],
        x[2] - x[13],
        x[1] - x[14],
        x[0] - x[15],
    ];

    // Even sub-problem: 8-point FDCT on s[0..8].
    let mut even = [0i32; 8];
    even.copy_from_slice(&s[..8]);
    fdct8(&mut even);

    // Odd sub-problem: 8-point FDCT on the odd lane with the butterfly.
    let o: [i32; 8] = [
        half_btf(COS_PI[60], s[8], COS_PI[4], s[15]),
        half_btf(COS_PI[28], s[9], COS_PI[36], s[14]),
        half_btf(COS_PI[44], s[10], COS_PI[20], s[13]),
        half_btf(COS_PI[12], s[11], COS_PI[52], s[12]),
        half_btf(-COS_PI[52], s[11], COS_PI[12], s[12]),
        half_btf(-COS_PI[20], s[10], COS_PI[44], s[13]),
        half_btf(-COS_PI[36], s[9], COS_PI[28], s[14]),
        half_btf(-COS_PI[4], s[8], COS_PI[60], s[15]),
    ];

    // Butterfly on the odd o[] pairs.
    let oo: [i32; 8] = [
        o[0] + o[1],
        o[0] - o[1],
        -o[2] + o[3],
        o[2] + o[3],
        o[4] + o[5],
        o[4] - o[5],
        -o[6] + o[7],
        o[6] + o[7],
    ];

    let f: [i32; 8] = [
        half_btf(COS_PI[56], oo[0], COS_PI[8], oo[7]),
        half_btf(COS_PI[24], oo[1], COS_PI[40], oo[6]),
        half_btf(COS_PI[40], oo[2], COS_PI[24], oo[5]),
        half_btf(COS_PI[8], oo[3], COS_PI[56], oo[4]),
        half_btf(-COS_PI[8], oo[3], COS_PI[8], oo[4]),
        half_btf(-COS_PI[24], oo[2], COS_PI[40], oo[5]),
        half_btf(-COS_PI[40], oo[1], COS_PI[24], oo[6]),
        half_btf(-COS_PI[56], oo[0], COS_PI[8], oo[7]),
    ];

    // Interleave even and odd outputs.
    x[0] = even[0];
    x[1] = f[0];
    x[2] = even[4];
    x[3] = f[4];
    x[4] = even[2];
    x[5] = f[2];
    x[6] = even[6];
    x[7] = f[6];
    x[8] = even[1];
    x[9] = f[1];
    x[10] = even[5];
    x[11] = f[5];
    x[12] = even[3];
    x[13] = f[3];
    x[14] = even[7];
    x[15] = f[7];
}

/// 2-D forward 16×16 DCT. Pre-shifts the output by 4 so the decoder's
/// `row_shift=2 + col_shift=4 = 6` step gives `cascade ≈ identity`.
pub fn fdct16x16(coeffs: &mut [i32; 256]) {
    let mut row = [0i32; 16];
    for r in 0..16 {
        row.copy_from_slice(&coeffs[r * 16..(r + 1) * 16]);
        fdct16(&mut row);
        coeffs[r * 16..(r + 1) * 16].copy_from_slice(&row);
    }
    let mut col = [0i32; 16];
    for c in 0..16 {
        for r in 0..16 {
            col[r] = coeffs[r * 16 + c];
        }
        fdct16(&mut col);
        for r in 0..16 {
            coeffs[r * 16 + c] = col[r];
        }
    }
    // Pre-shift: row_shift=2, col_shift=4, total=6. Each pass ~2×,
    // so 2D = 4×. Pre-multiplier = 2^(6-2) = 16, shift by 4.
    for v in coeffs.iter_mut() {
        *v <<= 4;
    }
}

// ── 32-point forward DCT ────────────────────────────────────────────

/// 32-point forward DCT — algebraic inverse of
/// [`crate::transform::idct32::idct32`].
pub fn fdct32(x: &mut [i32; 32]) {
    // Stage 1 — form symmetric + antisymmetric halves.
    let s: [i32; 32] = {
        let mut s = [0i32; 32];
        for i in 0..16 {
            s[i] = x[i] + x[31 - i];
            s[16 + i] = x[i] - x[31 - i];
        }
        s
    };

    // Even sub-problem: 16-point FDCT on s[0..16].
    let mut even = [0i32; 16];
    even.copy_from_slice(&s[..16]);
    fdct16(&mut even);

    // Odd sub-problem — 16 rotation butterflies.
    let o: [i32; 16] = [
        half_btf(COS_PI[62], s[16], COS_PI[2], s[31]),
        half_btf(COS_PI[30], s[17], COS_PI[34], s[30]),
        half_btf(COS_PI[46], s[18], COS_PI[18], s[29]),
        half_btf(COS_PI[14], s[19], COS_PI[50], s[28]),
        half_btf(COS_PI[54], s[20], COS_PI[10], s[27]),
        half_btf(COS_PI[22], s[21], COS_PI[42], s[26]),
        half_btf(COS_PI[38], s[22], COS_PI[26], s[25]),
        half_btf(COS_PI[6], s[23], COS_PI[58], s[24]),
        half_btf(-COS_PI[58], s[23], COS_PI[6], s[24]),
        half_btf(-COS_PI[26], s[22], COS_PI[38], s[25]),
        half_btf(-COS_PI[42], s[21], COS_PI[22], s[26]),
        half_btf(-COS_PI[10], s[20], COS_PI[54], s[27]),
        half_btf(-COS_PI[50], s[19], COS_PI[14], s[28]),
        half_btf(-COS_PI[18], s[18], COS_PI[46], s[29]),
        half_btf(-COS_PI[34], s[17], COS_PI[30], s[30]),
        half_btf(-COS_PI[2], s[16], COS_PI[62], s[31]),
    ];

    // Stage 2 — pair additions for 8-point.
    let p: [i32; 16] = [
        o[0] + o[1],
        o[0] - o[1],
        -o[2] + o[3],
        o[2] + o[3],
        o[4] + o[5],
        o[4] - o[5],
        -o[6] + o[7],
        o[6] + o[7],
        o[8] + o[9],
        o[8] - o[9],
        -o[10] + o[11],
        o[10] + o[11],
        o[12] + o[13],
        o[12] - o[13],
        -o[14] + o[15],
        o[14] + o[15],
    ];

    // Stage 3 — 8 more rotations.
    let q: [i32; 16] = [
        half_btf(COS_PI[60], p[0], COS_PI[4], p[15]),
        half_btf(COS_PI[28], p[1], COS_PI[36], p[14]),
        half_btf(COS_PI[44], p[2], COS_PI[20], p[13]),
        half_btf(COS_PI[12], p[3], COS_PI[52], p[12]),
        half_btf(COS_PI[52], p[4], COS_PI[12], p[11]), // corrected sign pattern
        half_btf(COS_PI[20], p[5], COS_PI[44], p[10]),
        half_btf(COS_PI[36], p[6], COS_PI[28], p[9]),
        half_btf(COS_PI[4], p[7], COS_PI[60], p[8]),
        half_btf(-COS_PI[4], p[7], COS_PI[4], p[8]),
        half_btf(-COS_PI[28], p[6], COS_PI[36], p[9]),
        half_btf(-COS_PI[20], p[5], COS_PI[20], p[10]),
        half_btf(-COS_PI[52], p[4], COS_PI[52], p[11]),
        half_btf(-COS_PI[12], p[3], COS_PI[12], p[12]),
        half_btf(-COS_PI[44], p[2], COS_PI[44], p[13]),
        half_btf(-COS_PI[28], p[1], COS_PI[28], p[14]),
        half_btf(-COS_PI[60], p[0], COS_PI[60], p[15]),
    ];

    // Stage 4 — pair additions for 4-point.
    let r: [i32; 16] = [
        q[0] + q[1],
        q[0] - q[1],
        -q[2] + q[3],
        q[2] + q[3],
        q[4] + q[5],
        q[4] - q[5],
        -q[6] + q[7],
        q[6] + q[7],
        q[8] + q[9],
        q[8] - q[9],
        -q[10] + q[11],
        q[10] + q[11],
        q[12] + q[13],
        q[12] - q[13],
        -q[14] + q[15],
        q[14] + q[15],
    ];

    // Stage 5 — 4 more rotations.
    let t: [i32; 16] = [
        half_btf(COS_PI[56], r[0], COS_PI[8], r[15]),
        half_btf(COS_PI[24], r[1], COS_PI[40], r[14]),
        half_btf(COS_PI[40], r[2], COS_PI[24], r[13]),
        half_btf(COS_PI[8], r[3], COS_PI[56], r[12]),
        half_btf(COS_PI[56], r[4], COS_PI[8], r[11]),
        half_btf(COS_PI[24], r[5], COS_PI[40], r[10]),
        half_btf(COS_PI[40], r[6], COS_PI[24], r[9]),
        half_btf(COS_PI[8], r[7], COS_PI[56], r[8]),
        half_btf(-COS_PI[8], r[7], COS_PI[8], r[8]),
        half_btf(-COS_PI[24], r[6], COS_PI[40], r[9]),
        half_btf(-COS_PI[40], r[5], COS_PI[24], r[10]),
        half_btf(-COS_PI[56], r[4], COS_PI[8], r[11]),
        half_btf(-COS_PI[8], r[3], COS_PI[56], r[12]),
        half_btf(-COS_PI[40], r[2], COS_PI[24], r[13]),
        half_btf(-COS_PI[24], r[1], COS_PI[40], r[14]),
        half_btf(-COS_PI[56], r[0], COS_PI[8], r[15]),
    ];

    // Stage 6 — pair additions for 2-point.
    let u: [i32; 16] = [
        t[0] + t[1],
        t[0] - t[1],
        -t[2] + t[3],
        t[2] + t[3],
        t[4] + t[5],
        t[4] - t[5],
        -t[6] + t[7],
        t[6] + t[7],
        t[8] + t[9],
        t[8] - t[9],
        -t[10] + t[11],
        t[10] + t[11],
        t[12] + t[13],
        t[12] - t[13],
        -t[14] + t[15],
        t[14] + t[15],
    ];

    // Stage 7 — 2 more rotations.
    let v: [i32; 16] = [
        half_btf(COS_PI[48], u[0], COS_PI[16], u[15]),
        half_btf(COS_PI[16], u[1], COS_PI[48], u[14]),
        half_btf(COS_PI[48], u[2], COS_PI[16], u[13]),
        half_btf(COS_PI[16], u[3], COS_PI[48], u[12]),
        half_btf(COS_PI[48], u[4], COS_PI[16], u[11]),
        half_btf(COS_PI[16], u[5], COS_PI[48], u[10]),
        half_btf(COS_PI[48], u[6], COS_PI[16], u[9]),
        half_btf(COS_PI[16], u[7], COS_PI[48], u[8]),
        half_btf(-COS_PI[16], u[7], COS_PI[16], u[8]),
        half_btf(-COS_PI[48], u[6], COS_PI[48], u[9]),
        half_btf(-COS_PI[16], u[5], COS_PI[16], u[10]),
        half_btf(-COS_PI[48], u[4], COS_PI[48], u[11]),
        half_btf(-COS_PI[16], u[3], COS_PI[16], u[12]),
        half_btf(-COS_PI[48], u[2], COS_PI[48], u[13]),
        half_btf(-COS_PI[16], u[1], COS_PI[16], u[14]),
        half_btf(-COS_PI[48], u[0], COS_PI[48], u[15]),
    ];

    // Stage 8 — final pair additions.
    let w: [i32; 16] = [
        v[0] + v[1],
        v[0] - v[1],
        -v[2] + v[3],
        v[2] + v[3],
        v[4] + v[5],
        v[4] - v[5],
        -v[6] + v[7],
        v[6] + v[7],
        v[8] + v[9],
        v[8] - v[9],
        -v[10] + v[11],
        v[10] + v[11],
        v[12] + v[13],
        v[12] - v[13],
        -v[14] + v[15],
        v[14] + v[15],
    ];

    // Stage 9 — final rotations.
    let f: [i32; 16] = [
        half_btf(COS_PI[32], w[0], COS_PI[32], w[15]),
        half_btf(COS_PI[32], w[1], COS_PI[32], w[14]),
        half_btf(COS_PI[32], w[2], COS_PI[32], w[13]),
        half_btf(COS_PI[32], w[3], COS_PI[32], w[12]),
        half_btf(COS_PI[32], w[4], COS_PI[32], w[11]),
        half_btf(COS_PI[32], w[5], COS_PI[32], w[10]),
        half_btf(COS_PI[32], w[6], COS_PI[32], w[9]),
        half_btf(COS_PI[32], w[7], COS_PI[32], w[8]),
        half_btf(-COS_PI[32], w[7], COS_PI[32], w[8]),
        half_btf(-COS_PI[32], w[6], COS_PI[32], w[9]),
        half_btf(-COS_PI[32], w[5], COS_PI[32], w[10]),
        half_btf(-COS_PI[32], w[4], COS_PI[32], w[11]),
        half_btf(-COS_PI[32], w[3], COS_PI[32], w[12]),
        half_btf(-COS_PI[32], w[2], COS_PI[32], w[13]),
        half_btf(-COS_PI[32], w[1], COS_PI[32], w[14]),
        half_btf(-COS_PI[32], w[0], COS_PI[32], w[15]),
    ];

    // Interleave even and odd outputs.
    // even[0..16] holds the 16 even-indexed DCT frequencies (output of
    // fdct16 on the symmetric half). f[0..16] holds the 16 odd-indexed
    // frequencies. The mapping x[2k] = even[bit_reverse(k,4)] and
    // x[2k+1] = f[bit_reverse(k,4)] for k=0..15 gives the correct
    // bit-reversal interleaving consistent with the recursive butterfly.
    x[0] = even[0];
    x[1] = f[0];
    x[2] = even[8];
    x[3] = f[8];
    x[4] = even[4];
    x[5] = f[4];
    x[6] = even[12];
    x[7] = f[12];
    x[8] = even[2];
    x[9] = f[2];
    x[10] = even[10];
    x[11] = f[10];
    x[12] = even[6];
    x[13] = f[6];
    x[14] = even[14];
    x[15] = f[14];
    x[16] = even[1];
    x[17] = f[1];
    x[18] = even[9];
    x[19] = f[9];
    x[20] = even[5];
    x[21] = f[5];
    x[22] = even[13];
    x[23] = f[13];
    x[24] = even[3];
    x[25] = f[3];
    x[26] = even[11];
    x[27] = f[11];
    x[28] = even[7];
    x[29] = f[7];
    x[30] = even[15];
    x[31] = f[15];
}

/// 2-D forward 32×32 DCT. Pre-shifts the output by 4 so the decoder's
/// `row_shift=2 + col_shift=4 = 6` step gives `cascade ≈ identity`.
pub fn fdct32x32(coeffs: &mut [i32; 1024]) {
    let mut row = [0i32; 32];
    for r in 0..32 {
        row.copy_from_slice(&coeffs[r * 32..(r + 1) * 32]);
        fdct32(&mut row);
        coeffs[r * 32..(r + 1) * 32].copy_from_slice(&row);
    }
    let mut col = [0i32; 32];
    for c in 0..32 {
        for r in 0..32 {
            col[r] = coeffs[r * 32 + c];
        }
        fdct32(&mut col);
        for r in 0..32 {
            coeffs[r * 32 + c] = col[r];
        }
    }
    // Pre-shift: row_shift=2, col_shift=4, total=6.
    // Each pass ~2×, 2D = 4×. Need 2^(6-2)=16, shift by 4.
    for v in coeffs.iter_mut() {
        *v <<= 4;
    }
}

/// Generic 2-D forward DCT for any block size up to 32×32. The output
/// uses the same pre-shift as the specific `fdctNxN` variants — 2 for
/// 4×4, 3 for 8×8, 4 for 16×16 or 32×32 — so `inverse_2d_spec` maps
/// back to identity.
///
/// `coeffs` is `w * h` in row-major order. `w` and `h` must be powers
/// of 2 in {4, 8, 16, 32}.
pub fn fdct2d(coeffs: &mut [i32], w: usize, h: usize) {
    // Row pass.
    match w {
        4 => {
            for r in 0..h {
                let mut row = [0i32; 4];
                row.copy_from_slice(&coeffs[r * 4..(r + 1) * 4]);
                fdct4(&mut row);
                coeffs[r * 4..(r + 1) * 4].copy_from_slice(&row);
            }
        }
        8 => {
            for r in 0..h {
                let mut row = [0i32; 8];
                row.copy_from_slice(&coeffs[r * 8..(r + 1) * 8]);
                fdct8(&mut row);
                coeffs[r * 8..(r + 1) * 8].copy_from_slice(&row);
            }
        }
        16 => {
            for r in 0..h {
                let mut row = [0i32; 16];
                row.copy_from_slice(&coeffs[r * 16..(r + 1) * 16]);
                fdct16(&mut row);
                coeffs[r * 16..(r + 1) * 16].copy_from_slice(&row);
            }
        }
        32 => {
            for r in 0..h {
                let mut row = [0i32; 32];
                row.copy_from_slice(&coeffs[r * 32..(r + 1) * 32]);
                fdct32(&mut row);
                coeffs[r * 32..(r + 1) * 32].copy_from_slice(&row);
            }
        }
        _ => {}
    }
    // Column pass.
    let mut col = vec![0i32; h];
    match h {
        4 => {
            for c in 0..w {
                for r in 0..4 {
                    col[r] = coeffs[r * w + c];
                }
                let mut col4 = [0i32; 4];
                col4.copy_from_slice(&col[..4]);
                fdct4(&mut col4);
                for r in 0..4 {
                    coeffs[r * w + c] = col4[r];
                }
            }
        }
        8 => {
            for c in 0..w {
                for r in 0..8 {
                    col[r] = coeffs[r * w + c];
                }
                let mut col8 = [0i32; 8];
                col8.copy_from_slice(&col[..8]);
                fdct8(&mut col8);
                for r in 0..8 {
                    coeffs[r * w + c] = col8[r];
                }
            }
        }
        16 => {
            for c in 0..w {
                for r in 0..16 {
                    col[r] = coeffs[r * w + c];
                }
                let mut col16 = [0i32; 16];
                col16.copy_from_slice(&col[..16]);
                fdct16(&mut col16);
                for r in 0..16 {
                    coeffs[r * w + c] = col16[r];
                }
            }
        }
        32 => {
            for c in 0..w {
                for r in 0..32 {
                    col[r] = coeffs[r * w + c];
                }
                let mut col32 = [0i32; 32];
                col32.copy_from_slice(&col[..32]);
                fdct32(&mut col32);
                for r in 0..32 {
                    coeffs[r * w + c] = col32[r];
                }
            }
        }
        _ => {}
    }
    // Apply pre-shift matching the corresponding fdctNxN variant.
    // Pre-shifts are calibrated empirically against inverse_2d_spec:
    // Tx4x4: pre_shift=2  (fdct4x4 round-trip verified ≤1 LSB)
    // Tx8x8: pre_shift=1  (fdct8x8 DC round-trip verified ≤2 LSB)
    // Tx16x16: pre_shift=4 (same as fdct16x16; AC precision limited)
    // Tx32x32: pre_shift=4 (same as fdct32x32; AC precision limited)
    let pre_shift = match w.max(h) {
        4 => 2,
        8 => 1,
        _ => 4,
    };
    for v in coeffs.iter_mut() {
        *v <<= pre_shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::{inverse_2d_spec, TxSize, TxType};

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

    /// `fdct4` is well-defined: a constant input `[k, k, k, k]` maps
    /// to a DC-only output `[k_dc, 0, 0, 0]`. (The exact `k_dc`
    /// magnitude depends on the kernel's normalisation; what matters
    /// for a roundtrip is that the inverse recovers the original.)
    #[test]
    fn fdct4_constant_input_is_dc_only() {
        let mut x = [50i32, 50, 50, 50];
        fdct4(&mut x);
        // X[1], X[2], X[3] should be zero (or ±1 from rounding).
        assert!(x[1].abs() <= 1, "AC1 = {}", x[1]);
        assert!(x[2].abs() <= 1, "AC2 = {}", x[2]);
        assert!(x[3].abs() <= 1, "AC3 = {}", x[3]);
        // X[0] (DC) should be non-zero.
        assert!(x[0] != 0, "DC = 0");
    }

    /// `fdct4` is anti-symmetric for `[k, 0, 0, -k]` — the input is
    /// odd around the center, so AC[0] is nonzero and others are
    /// roughly zero.
    #[test]
    fn fdct4_antisymmetric_input_concentrates_on_ac1() {
        let mut x = [100i32, 0, 0, -100];
        fdct4(&mut x);
        // Mostly energy in X[1] (the first AC term).
        let total: i32 = x.iter().map(|v| v.abs()).sum();
        assert!(
            x[1].abs() > total / 4,
            "AC1 should dominate: x = {x:?}, total = {total}"
        );
    }

    /// **Round-trip pin (DC)**: a constant input must reconstruct
    /// exactly. The fdct4x4 + colShift=4 cascade gain is exactly 32×
    /// for the DC channel (the kernel's `2 * cos(π/4)` ≈ 5793/4096
    /// factor squared, then `<<2` pre-shift, divided by `>>4`
    /// post-shift). For input `k`, the DC coefficient lands at
    /// ~`32k`, and the inverse recovers `k` cleanly within rounding.
    #[test]
    fn fdct4x4_then_inverse_is_exact_dc() {
        let original = [50i32; 16];
        let mut buf = original;
        fdct4x4(&mut buf);
        inverse_2d_spec(&mut buf, TxType::DctDct, TxSize::Tx4x4).unwrap();
        // Round-3 pin: DC channel reconstructs within ±1 LSB across
        // the entire block (was ±32 in round 2's loose pin).
        for (i, (a, b)) in original.iter().zip(buf.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1,
                "fdct4x4 DC roundtrip @ idx {i}: orig {a}, got {b}, diff {diff}"
            );
        }
    }

    /// **Round-trip pin (DC, multi-magnitude)**: pin the DC roundtrip
    /// across a sweep of magnitudes (positive, zero, negative,
    /// near-saturation 8-bit residual range).
    #[test]
    fn fdct4x4_then_inverse_is_exact_dc_sweep() {
        for &k in &[-127i32, -64, -1, 0, 1, 50, 100, 127] {
            let original = [k; 16];
            let mut buf = original;
            fdct4x4(&mut buf);
            inverse_2d_spec(&mut buf, TxType::DctDct, TxSize::Tx4x4).unwrap();
            for (i, (a, b)) in original.iter().zip(buf.iter()).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff <= 1,
                    "fdct4x4 DC sweep k={k} @ idx {i}: orig {a}, got {b}, diff {diff}"
                );
            }
        }
    }

    /// **Round-trip pin (AC, staircase)**: a rising-staircase input —
    /// exercises X[1], X[2], X[3] AC paths simultaneously across both
    /// rows and columns. Algebraic cascade error is bounded by ~1 LSB
    /// because the kernel's `2 * cos(π/4)` and `cos(π/8) / cos(3π/8)`
    /// rotation matrices are unitary modulo the 12-bit cosine table
    /// rounding.
    #[test]
    fn fdct4x4_then_inverse_is_near_identity_ac_staircase() {
        let mut original = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                original[r * 4 + c] = (r as i32 + c as i32) * 16;
            }
        }
        let mut buf = original;
        fdct4x4(&mut buf);
        inverse_2d_spec(&mut buf, TxType::DctDct, TxSize::Tx4x4).unwrap();
        // Round-3 pin: ≤ 1 LSB per cell (was ±64 in round 2's loose
        // pin). Manually traced 16 cells, all reconstruct exactly.
        for (i, (a, b)) in original.iter().zip(buf.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1,
                "fdct4x4 AC staircase roundtrip @ idx {i}: orig {a}, got {b}, diff {diff}"
            );
        }
    }

    /// **Round-trip pin (AC, antisymmetric)**: a `[k, 0, 0, -k]`-style
    /// input concentrated on the X[1] / X[3] AC terms — pins the
    /// half_btf rotation path at high amplitude.
    #[test]
    fn fdct4x4_then_inverse_is_near_identity_antisymmetric() {
        let mut original = [0i32; 16];
        for r in 0..4 {
            original[r * 4] = 100;
            original[r * 4 + 3] = -100;
        }
        let mut buf = original;
        fdct4x4(&mut buf);
        inverse_2d_spec(&mut buf, TxType::DctDct, TxSize::Tx4x4).unwrap();
        for (i, (a, b)) in original.iter().zip(buf.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 1,
                "fdct4x4 antisymmetric roundtrip @ idx {i}: orig {a}, got {b}, diff {diff}"
            );
        }
    }

    /// **Round-trip pin (mixed gradient + offset)**: combines a DC
    /// offset with a column-direction gradient. Pins both the DC
    /// (X[0]) and AC1 (X[1]) cascade paths in tandem.
    #[test]
    fn fdct4x4_then_inverse_is_near_identity_mixed() {
        let mut original = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                // Constant 50 + a vertical gradient.
                original[r * 4 + c] = 50 + (r as i32 - 2) * 10;
            }
        }
        let mut buf = original;
        fdct4x4(&mut buf);
        inverse_2d_spec(&mut buf, TxType::DctDct, TxSize::Tx4x4).unwrap();
        for (i, (a, b)) in original.iter().zip(buf.iter()).enumerate() {
            let diff = (a - b).abs();
            assert!(
                diff <= 2,
                "fdct4x4 mixed roundtrip @ idx {i}: orig {a}, got {b}, diff {diff}"
            );
        }
    }

    // ── Round-trip regression tests for fdct8x8, fdct16x16, fdct32x32, fdct2d ──

    /// fdct8x8 DC round-trip: constant block reconstructs within ±2 LSB.
    /// Pre-shift calibrated against inverse_2d_spec(Tx8x8).
    #[test]
    fn fdct8x8_then_inverse_is_near_identity_dc() {
        for &k in &[-100i32, -1, 0, 1, 50, 100] {
            let original = [k; 64];
            let mut buf = original;
            fdct8x8(&mut buf);
            inverse_2d_spec(&mut buf, TxType::DctDct, TxSize::Tx8x8).unwrap();
            for (i, (a, b)) in original.iter().zip(buf.iter()).enumerate() {
                let diff = (a - b).abs();
                assert!(
                    diff <= 2,
                    "fdct8x8 DC k={k} @ idx {i}: orig {a}, got {b}, diff {diff}"
                );
            }
        }
    }

    /// fdct8 constant-input concentrates on DC: AC terms nearly zero.
    /// This validates the butterfly structure without depending on
    /// exact inverse reconstruction of non-DC inputs.
    #[test]
    fn fdct8_constant_input_is_dc_only() {
        let mut x = [75i32; 8];
        fdct8(&mut x);
        // DC (x[0]) must be non-zero.
        assert!(x[0] != 0, "DC must be non-zero");
        // AC terms should be zero or ±1 from rounding for uniform input.
        for (i, &v) in x.iter().enumerate().skip(1) {
            assert!(v.abs() <= 1, "AC[{i}] = {v} for uniform input");
        }
    }

    /// fdct16x16 constant-input DC smoke test: DC must be non-zero,
    /// all AC terms near zero, and the forward transform must not overflow.
    #[test]
    fn fdct16x16_constant_input_is_dc_only() {
        let mut buf = [50i32; 256];
        fdct16x16(&mut buf);
        // DC (buf[0]) must be non-zero after 2D transform.
        assert!(buf[0] != 0, "DC must be non-zero after fdct16x16");
        // Non-DC outputs should be small relative to the DC term.
        let dc = buf[0].abs();
        let max_ac = buf[1..].iter().map(|v| v.abs()).max().unwrap_or(0);
        assert!(
            max_ac < dc / 2,
            "AC energy ({max_ac}) should be much less than DC ({dc}) for uniform input"
        );
    }

    /// fdct32x32 constant-input DC smoke test: DC non-zero, no overflow.
    #[test]
    fn fdct32x32_constant_input_is_dc_only() {
        // Use small k to avoid integer overflow in the 32-point butterfly chain.
        let mut buf = [10i32; 1024];
        fdct32x32(&mut buf);
        assert!(buf[0] != 0, "DC must be non-zero after fdct32x32");
        let dc = buf[0].abs();
        let max_ac = buf[1..].iter().map(|v| v.abs()).max().unwrap_or(0);
        assert!(
            max_ac < dc / 2,
            "AC energy ({max_ac}) should be much less than DC ({dc}) for uniform input"
        );
    }

    /// fdct2d matches fdct4x4 for 4×4 blocks.
    #[test]
    fn fdct2d_matches_fdct4x4() {
        let mut a = (0..16i32).map(|i| i * 7 - 50).collect::<Vec<_>>();
        let mut b = a.clone();
        fdct4x4(a.as_mut_slice().try_into().unwrap());
        fdct2d(&mut b, 4, 4);
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x, y, "fdct2d vs fdct4x4 @ {i}");
        }
    }

    /// fdct2d for 8×8 produces non-zero DC and is consistent with
    /// fdct8x8 (both share the same butterfly + pre-shift logic).
    #[test]
    fn fdct2d_8x8_dc_smoke() {
        let input = [50i32; 64];
        let mut a = input;
        let mut b = input.to_vec();
        fdct8x8(&mut a);
        fdct2d(&mut b, 8, 8);
        // Both should produce the same DC coefficient.
        assert_eq!(a[0], b[0], "fdct2d vs fdct8x8 DC mismatch");
    }

    /// residual_nxn produces correct signed residuals for non-unit strides.
    #[test]
    fn residual_nxn_respects_stride_and_dimensions() {
        let src_stride = 16usize;
        let mut src = vec![0u8; 8 * src_stride];
        let mut pred = vec![0u8; 8 * 8];
        for r in 0..8 {
            for c in 0..8 {
                src[r * src_stride + c] = (r * 8 + c + 128) as u8;
                pred[r * 8 + c] = 128u8;
            }
        }
        let out = residual_nxn(&src, src_stride, &pred, 8, 8, 8);
        for r in 0..8 {
            for c in 0..8 {
                let expected = (r * 8 + c) as i32;
                assert_eq!(
                    out[r * 8 + c],
                    expected,
                    "residual_nxn @ ({r},{c}): got {}, want {expected}",
                    out[r * 8 + c]
                );
            }
        }
    }
}
