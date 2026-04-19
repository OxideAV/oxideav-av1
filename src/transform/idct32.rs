//! 32-point inverse DCT — §7.7.2.1.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/transform/idct32.go`
//! (MIT, KarpelesLab/goavif). This is a 9-stage butterfly transcribed
//! directly from libaom's `av1_idct32`. The Go port uses two scratch
//! 32-slot buffers (`out` and `step`) that ping-pong the way libaom
//! swaps `bf0` / `bf1`; the Rust port keeps that structure verbatim so
//! the butterfly constants stay identical and the sign pattern is easy
//! to audit against the reference.

use super::cos_pi::{half_btf, COS_PI};

/// In-place 32-point inverse DCT. `x` must have exactly 32 entries.
pub fn idct32(x: &mut [i32; 32]) {
    // stage 1 — permutation into `out`.
    let in_ = *x;
    let mut out: [i32; 32] = [
        in_[0], in_[16], in_[8], in_[24], in_[4], in_[20], in_[12], in_[28], in_[2], in_[18],
        in_[10], in_[26], in_[6], in_[22], in_[14], in_[30], in_[1], in_[17], in_[9], in_[25],
        in_[5], in_[21], in_[13], in_[29], in_[3], in_[19], in_[11], in_[27], in_[7], in_[23],
        in_[15], in_[31],
    ];

    // stage 2 — rotations on indices 16..31.
    let mut step = [0i32; 32];
    step[..16].copy_from_slice(&out[..16]);
    step[16] = half_btf(COS_PI[62], out[16], -COS_PI[2], out[31]);
    step[17] = half_btf(COS_PI[30], out[17], -COS_PI[34], out[30]);
    step[18] = half_btf(COS_PI[46], out[18], -COS_PI[18], out[29]);
    step[19] = half_btf(COS_PI[14], out[19], -COS_PI[50], out[28]);
    step[20] = half_btf(COS_PI[54], out[20], -COS_PI[10], out[27]);
    step[21] = half_btf(COS_PI[22], out[21], -COS_PI[42], out[26]);
    step[22] = half_btf(COS_PI[38], out[22], -COS_PI[26], out[25]);
    step[23] = half_btf(COS_PI[6], out[23], -COS_PI[58], out[24]);
    step[24] = half_btf(COS_PI[58], out[23], COS_PI[6], out[24]);
    step[25] = half_btf(COS_PI[26], out[22], COS_PI[38], out[25]);
    step[26] = half_btf(COS_PI[42], out[21], COS_PI[22], out[26]);
    step[27] = half_btf(COS_PI[10], out[20], COS_PI[54], out[27]);
    step[28] = half_btf(COS_PI[50], out[19], COS_PI[14], out[28]);
    step[29] = half_btf(COS_PI[18], out[18], COS_PI[46], out[29]);
    step[30] = half_btf(COS_PI[34], out[17], COS_PI[30], out[30]);
    step[31] = half_btf(COS_PI[2], out[16], COS_PI[62], out[31]);

    // stage 3.
    out[..8].copy_from_slice(&step[..8]);
    out[8] = half_btf(COS_PI[60], step[8], -COS_PI[4], step[15]);
    out[9] = half_btf(COS_PI[28], step[9], -COS_PI[36], step[14]);
    out[10] = half_btf(COS_PI[44], step[10], -COS_PI[20], step[13]);
    out[11] = half_btf(COS_PI[12], step[11], -COS_PI[52], step[12]);
    out[12] = half_btf(COS_PI[52], step[11], COS_PI[12], step[12]);
    out[13] = half_btf(COS_PI[20], step[10], COS_PI[44], step[13]);
    out[14] = half_btf(COS_PI[36], step[9], COS_PI[28], step[14]);
    out[15] = half_btf(COS_PI[4], step[8], COS_PI[60], step[15]);
    out[16] = step[16] + step[17];
    out[17] = step[16] - step[17];
    out[18] = -step[18] + step[19];
    out[19] = step[18] + step[19];
    out[20] = step[20] + step[21];
    out[21] = step[20] - step[21];
    out[22] = -step[22] + step[23];
    out[23] = step[22] + step[23];
    out[24] = step[24] + step[25];
    out[25] = step[24] - step[25];
    out[26] = -step[26] + step[27];
    out[27] = step[26] + step[27];
    out[28] = step[28] + step[29];
    out[29] = step[28] - step[29];
    out[30] = -step[30] + step[31];
    out[31] = step[30] + step[31];

    // stage 4.
    step[..4].copy_from_slice(&out[..4]);
    step[4] = half_btf(COS_PI[56], out[4], -COS_PI[8], out[7]);
    step[5] = half_btf(COS_PI[24], out[5], -COS_PI[40], out[6]);
    step[6] = half_btf(COS_PI[40], out[5], COS_PI[24], out[6]);
    step[7] = half_btf(COS_PI[8], out[4], COS_PI[56], out[7]);
    step[8] = out[8] + out[9];
    step[9] = out[8] - out[9];
    step[10] = -out[10] + out[11];
    step[11] = out[10] + out[11];
    step[12] = out[12] + out[13];
    step[13] = out[12] - out[13];
    step[14] = -out[14] + out[15];
    step[15] = out[14] + out[15];
    step[16] = out[16];
    step[17] = half_btf(-COS_PI[8], out[17], COS_PI[56], out[30]);
    step[18] = half_btf(-COS_PI[56], out[18], -COS_PI[8], out[29]);
    step[19] = out[19];
    step[20] = out[20];
    step[21] = half_btf(-COS_PI[40], out[21], COS_PI[24], out[26]);
    step[22] = half_btf(-COS_PI[24], out[22], -COS_PI[40], out[25]);
    step[23] = out[23];
    step[24] = out[24];
    step[25] = half_btf(-COS_PI[40], out[22], COS_PI[24], out[25]);
    step[26] = half_btf(COS_PI[24], out[21], COS_PI[40], out[26]);
    step[27] = out[27];
    step[28] = out[28];
    step[29] = half_btf(-COS_PI[8], out[18], COS_PI[56], out[29]);
    step[30] = half_btf(COS_PI[56], out[17], COS_PI[8], out[30]);
    step[31] = out[31];

    // stage 5.
    out[0] = half_btf(COS_PI[32], step[0], COS_PI[32], step[1]);
    out[1] = half_btf(COS_PI[32], step[0], -COS_PI[32], step[1]);
    out[2] = half_btf(COS_PI[48], step[2], -COS_PI[16], step[3]);
    out[3] = half_btf(COS_PI[16], step[2], COS_PI[48], step[3]);
    out[4] = step[4] + step[5];
    out[5] = step[4] - step[5];
    out[6] = -step[6] + step[7];
    out[7] = step[6] + step[7];
    out[8] = step[8];
    out[9] = half_btf(-COS_PI[16], step[9], COS_PI[48], step[14]);
    out[10] = half_btf(-COS_PI[48], step[10], -COS_PI[16], step[13]);
    out[11] = step[11];
    out[12] = step[12];
    out[13] = half_btf(-COS_PI[16], step[10], COS_PI[48], step[13]);
    out[14] = half_btf(COS_PI[48], step[9], COS_PI[16], step[14]);
    out[15] = step[15];
    out[16] = step[16] + step[19];
    out[17] = step[17] + step[18];
    out[18] = step[17] - step[18];
    out[19] = step[16] - step[19];
    out[20] = -step[20] + step[23];
    out[21] = -step[21] + step[22];
    out[22] = step[21] + step[22];
    out[23] = step[20] + step[23];
    out[24] = step[24] + step[27];
    out[25] = step[25] + step[26];
    out[26] = step[25] - step[26];
    out[27] = step[24] - step[27];
    out[28] = -step[28] + step[31];
    out[29] = -step[29] + step[30];
    out[30] = step[29] + step[30];
    out[31] = step[28] + step[31];

    // stage 6.
    step[0] = out[0] + out[3];
    step[1] = out[1] + out[2];
    step[2] = out[1] - out[2];
    step[3] = out[0] - out[3];
    step[4] = out[4];
    step[5] = half_btf(-COS_PI[32], out[5], COS_PI[32], out[6]);
    step[6] = half_btf(COS_PI[32], out[5], COS_PI[32], out[6]);
    step[7] = out[7];
    step[8] = out[8] + out[11];
    step[9] = out[9] + out[10];
    step[10] = out[9] - out[10];
    step[11] = out[8] - out[11];
    step[12] = -out[12] + out[15];
    step[13] = -out[13] + out[14];
    step[14] = out[13] + out[14];
    step[15] = out[12] + out[15];
    step[16] = out[16];
    step[17] = out[17];
    step[18] = half_btf(-COS_PI[16], out[18], COS_PI[48], out[29]);
    step[19] = half_btf(-COS_PI[16], out[19], COS_PI[48], out[28]);
    step[20] = half_btf(-COS_PI[48], out[20], -COS_PI[16], out[27]);
    step[21] = half_btf(-COS_PI[48], out[21], -COS_PI[16], out[26]);
    step[22] = out[22];
    step[23] = out[23];
    step[24] = out[24];
    step[25] = out[25];
    step[26] = half_btf(-COS_PI[16], out[21], COS_PI[48], out[26]);
    step[27] = half_btf(-COS_PI[16], out[20], COS_PI[48], out[27]);
    step[28] = half_btf(COS_PI[48], out[19], COS_PI[16], out[28]);
    step[29] = half_btf(COS_PI[48], out[18], COS_PI[16], out[29]);
    step[30] = out[30];
    step[31] = out[31];

    // stage 7.
    out[0] = step[0] + step[7];
    out[1] = step[1] + step[6];
    out[2] = step[2] + step[5];
    out[3] = step[3] + step[4];
    out[4] = step[3] - step[4];
    out[5] = step[2] - step[5];
    out[6] = step[1] - step[6];
    out[7] = step[0] - step[7];
    out[8] = step[8];
    out[9] = step[9];
    out[10] = half_btf(-COS_PI[32], step[10], COS_PI[32], step[13]);
    out[11] = half_btf(-COS_PI[32], step[11], COS_PI[32], step[12]);
    out[12] = half_btf(COS_PI[32], step[11], COS_PI[32], step[12]);
    out[13] = half_btf(COS_PI[32], step[10], COS_PI[32], step[13]);
    out[14] = step[14];
    out[15] = step[15];
    out[16] = step[16] + step[23];
    out[17] = step[17] + step[22];
    out[18] = step[18] + step[21];
    out[19] = step[19] + step[20];
    out[20] = step[19] - step[20];
    out[21] = step[18] - step[21];
    out[22] = step[17] - step[22];
    out[23] = step[16] - step[23];
    out[24] = -step[24] + step[31];
    out[25] = -step[25] + step[30];
    out[26] = -step[26] + step[29];
    out[27] = -step[27] + step[28];
    out[28] = step[27] + step[28];
    out[29] = step[26] + step[29];
    out[30] = step[25] + step[30];
    out[31] = step[24] + step[31];

    // stage 8.
    step[0] = out[0] + out[15];
    step[1] = out[1] + out[14];
    step[2] = out[2] + out[13];
    step[3] = out[3] + out[12];
    step[4] = out[4] + out[11];
    step[5] = out[5] + out[10];
    step[6] = out[6] + out[9];
    step[7] = out[7] + out[8];
    step[8] = out[7] - out[8];
    step[9] = out[6] - out[9];
    step[10] = out[5] - out[10];
    step[11] = out[4] - out[11];
    step[12] = out[3] - out[12];
    step[13] = out[2] - out[13];
    step[14] = out[1] - out[14];
    step[15] = out[0] - out[15];
    step[16] = out[16];
    step[17] = out[17];
    step[18] = out[18];
    step[19] = out[19];
    step[20] = half_btf(-COS_PI[32], out[20], COS_PI[32], out[27]);
    step[21] = half_btf(-COS_PI[32], out[21], COS_PI[32], out[26]);
    step[22] = half_btf(-COS_PI[32], out[22], COS_PI[32], out[25]);
    step[23] = half_btf(-COS_PI[32], out[23], COS_PI[32], out[24]);
    step[24] = half_btf(COS_PI[32], out[23], COS_PI[32], out[24]);
    step[25] = half_btf(COS_PI[32], out[22], COS_PI[32], out[25]);
    step[26] = half_btf(COS_PI[32], out[21], COS_PI[32], out[26]);
    step[27] = half_btf(COS_PI[32], out[20], COS_PI[32], out[27]);
    step[28] = out[28];
    step[29] = out[29];
    step[30] = out[30];
    step[31] = out[31];

    // stage 9 — final butterfly.
    x[0] = step[0] + step[31];
    x[1] = step[1] + step[30];
    x[2] = step[2] + step[29];
    x[3] = step[3] + step[28];
    x[4] = step[4] + step[27];
    x[5] = step[5] + step[26];
    x[6] = step[6] + step[25];
    x[7] = step[7] + step[24];
    x[8] = step[8] + step[23];
    x[9] = step[9] + step[22];
    x[10] = step[10] + step[21];
    x[11] = step[11] + step[20];
    x[12] = step[12] + step[19];
    x[13] = step[13] + step[18];
    x[14] = step[14] + step[17];
    x[15] = step[15] + step[16];
    x[16] = step[15] - step[16];
    x[17] = step[14] - step[17];
    x[18] = step[13] - step[18];
    x[19] = step[12] - step[19];
    x[20] = step[11] - step[20];
    x[21] = step[10] - step[21];
    x[22] = step[9] - step[22];
    x[23] = step[8] - step[23];
    x[24] = step[7] - step[24];
    x[25] = step[6] - step[25];
    x[26] = step[5] - step[26];
    x[27] = step[4] - step[27];
    x[28] = step[3] - step[28];
    x[29] = step[2] - step[29];
    x[30] = step[1] - step[30];
    x[31] = step[0] - step[31];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct32_dc_reconstruction_constant() {
        let mut x = [0i32; 32];
        x[0] = 32768;
        idct32(&mut x);
        let first = x[0];
        for (i, v) in x.iter().enumerate().skip(1) {
            assert_eq!(*v, first, "non-constant at {i}");
        }
        assert_ne!(first, 0);
    }

    #[test]
    fn idct32_linearity_within_tolerance() {
        let mut a = [0i32; 32];
        let mut b = [0i32; 32];
        for i in 0..32 {
            a[i] = ((i * 37) as i32 % 200) - 100;
            b[i] = ((i * 23) as i32 % 150) - 75;
        }
        let mut sum = [0i32; 32];
        for i in 0..32 {
            sum[i] = a[i] + b[i];
        }
        let mut ac = a;
        let mut bc = b;
        idct32(&mut ac);
        idct32(&mut bc);
        idct32(&mut sum);
        for i in 0..32 {
            let want = ac[i] + bc[i];
            let diff = (sum[i] - want).abs();
            assert!(diff <= 30, "linearity at {i}: diff {diff}");
        }
    }
}
