//! 16-point inverse ADST — §7.7.2.4.
//!
//! Mirrors libaom's `av1_iadst16`: 9-stage butterfly with the spec's
//! bit-reverse permutation, six rotation stages, three pair add/sub
//! stages, and a final permutation with alternating negations.

use super::cos_pi::{half_btf, COS_PI};

/// In-place 16-point inverse ADST. `x` must have exactly 16 entries.
pub fn iadst16(x: &mut [i32; 16]) {
    // stage 1 — permutation.
    let in_ = *x;
    let mut out = [
        in_[15], in_[0], in_[13], in_[2], in_[11], in_[4], in_[9], in_[6], in_[7], in_[8], in_[5],
        in_[10], in_[3], in_[12], in_[1], in_[14],
    ];

    // stage 2 — rotations.
    let mut step = [0i32; 16];
    step[0] = half_btf(COS_PI[2], out[0], COS_PI[62], out[1]);
    step[1] = half_btf(COS_PI[62], out[0], -COS_PI[2], out[1]);
    step[2] = half_btf(COS_PI[10], out[2], COS_PI[54], out[3]);
    step[3] = half_btf(COS_PI[54], out[2], -COS_PI[10], out[3]);
    step[4] = half_btf(COS_PI[18], out[4], COS_PI[46], out[5]);
    step[5] = half_btf(COS_PI[46], out[4], -COS_PI[18], out[5]);
    step[6] = half_btf(COS_PI[26], out[6], COS_PI[38], out[7]);
    step[7] = half_btf(COS_PI[38], out[6], -COS_PI[26], out[7]);
    step[8] = half_btf(COS_PI[34], out[8], COS_PI[30], out[9]);
    step[9] = half_btf(COS_PI[30], out[8], -COS_PI[34], out[9]);
    step[10] = half_btf(COS_PI[42], out[10], COS_PI[22], out[11]);
    step[11] = half_btf(COS_PI[22], out[10], -COS_PI[42], out[11]);
    step[12] = half_btf(COS_PI[50], out[12], COS_PI[14], out[13]);
    step[13] = half_btf(COS_PI[14], out[12], -COS_PI[50], out[13]);
    step[14] = half_btf(COS_PI[58], out[14], COS_PI[6], out[15]);
    step[15] = half_btf(COS_PI[6], out[14], -COS_PI[58], out[15]);

    // stage 3 — pair add/sub on halves.
    for i in 0..8 {
        out[i] = step[i] + step[i + 8];
        out[i + 8] = step[i] - step[i + 8];
    }

    // stage 4 — rotations on upper half.
    step[..8].copy_from_slice(&out[..8]);
    step[8] = half_btf(COS_PI[8], out[8], COS_PI[56], out[9]);
    step[9] = half_btf(COS_PI[56], out[8], -COS_PI[8], out[9]);
    step[10] = half_btf(COS_PI[40], out[10], COS_PI[24], out[11]);
    step[11] = half_btf(COS_PI[24], out[10], -COS_PI[40], out[11]);
    step[12] = half_btf(-COS_PI[56], out[12], COS_PI[8], out[13]);
    step[13] = half_btf(COS_PI[8], out[12], COS_PI[56], out[13]);
    step[14] = half_btf(-COS_PI[24], out[14], COS_PI[40], out[15]);
    step[15] = half_btf(COS_PI[40], out[14], COS_PI[24], out[15]);

    // stage 5 — pair add/sub on quarters.
    for i in 0..4 {
        out[i] = step[i] + step[i + 4];
        out[i + 4] = step[i] - step[i + 4];
        out[i + 8] = step[i + 8] + step[i + 12];
        out[i + 12] = step[i + 8] - step[i + 12];
    }

    // stage 6 — rotations.
    step[0] = out[0];
    step[1] = out[1];
    step[2] = out[2];
    step[3] = out[3];
    step[4] = half_btf(COS_PI[16], out[4], COS_PI[48], out[5]);
    step[5] = half_btf(COS_PI[48], out[4], -COS_PI[16], out[5]);
    step[6] = half_btf(-COS_PI[48], out[6], COS_PI[16], out[7]);
    step[7] = half_btf(COS_PI[16], out[6], COS_PI[48], out[7]);
    step[8] = out[8];
    step[9] = out[9];
    step[10] = out[10];
    step[11] = out[11];
    step[12] = half_btf(COS_PI[16], out[12], COS_PI[48], out[13]);
    step[13] = half_btf(COS_PI[48], out[12], -COS_PI[16], out[13]);
    step[14] = half_btf(-COS_PI[48], out[14], COS_PI[16], out[15]);
    step[15] = half_btf(COS_PI[16], out[14], COS_PI[48], out[15]);

    // stage 7 — pair add/sub on eighths.
    for i in 0..2 {
        out[i] = step[i] + step[i + 2];
        out[i + 2] = step[i] - step[i + 2];
        out[i + 4] = step[i + 4] + step[i + 6];
        out[i + 6] = step[i + 4] - step[i + 6];
        out[i + 8] = step[i + 8] + step[i + 10];
        out[i + 10] = step[i + 8] - step[i + 10];
        out[i + 12] = step[i + 12] + step[i + 14];
        out[i + 14] = step[i + 12] - step[i + 14];
    }

    // stage 8 — cos_pi[32] crosses on odd pairs.
    step[0] = out[0];
    step[1] = out[1];
    step[2] = half_btf(COS_PI[32], out[2], COS_PI[32], out[3]);
    step[3] = half_btf(COS_PI[32], out[2], -COS_PI[32], out[3]);
    step[4] = out[4];
    step[5] = out[5];
    step[6] = half_btf(COS_PI[32], out[6], COS_PI[32], out[7]);
    step[7] = half_btf(COS_PI[32], out[6], -COS_PI[32], out[7]);
    step[8] = out[8];
    step[9] = out[9];
    step[10] = half_btf(COS_PI[32], out[10], COS_PI[32], out[11]);
    step[11] = half_btf(COS_PI[32], out[10], -COS_PI[32], out[11]);
    step[12] = out[12];
    step[13] = out[13];
    step[14] = half_btf(COS_PI[32], out[14], COS_PI[32], out[15]);
    step[15] = half_btf(COS_PI[32], out[14], -COS_PI[32], out[15]);

    // stage 9 — permutation + alternating negations.
    x[0] = step[0];
    x[1] = -step[8];
    x[2] = step[12];
    x[3] = -step[4];
    x[4] = step[6];
    x[5] = -step[14];
    x[6] = step[10];
    x[7] = -step[2];
    x[8] = step[3];
    x[9] = -step[11];
    x[10] = step[15];
    x[11] = -step[7];
    x[12] = step[5];
    x[13] = -step[13];
    x[14] = step[9];
    x[15] = -step[1];
}

/// In-place 16-point inverse flipped ADST.
pub fn iflipadst16(x: &mut [i32; 16]) {
    iadst16(x);
    for i in 0..8 {
        x.swap(i, 15 - i);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iadst16_linearity_within_tolerance() {
        let a = [
            100i32, 50, -25, 10, 5, -40, 20, 15, 0, -33, 7, 60, -11, 4, -8, 22,
        ];
        let b = [
            -30i32, 40, 5, -12, 22, 7, -18, 3, 10, 5, -2, -15, 33, -4, 9, -7,
        ];
        let mut sum = [0i32; 16];
        for i in 0..16 {
            sum[i] = a[i] + b[i];
        }
        let mut ac = a;
        let mut bc = b;
        iadst16(&mut ac);
        iadst16(&mut bc);
        iadst16(&mut sum);
        for i in 0..16 {
            let want = ac[i] + bc[i];
            let diff = (sum[i] - want).abs();
            assert!(diff <= 16, "linearity at {i}: diff {diff}");
        }
    }
}
