//! 16-point inverse DCT — §7.7.2.1.
//!
//! Follows the spec's 7-stage butterfly decomposition.

use super::cos_pi::{half_btf, COS_PI};

/// In-place 16-point inverse DCT. `x` must have exactly 16 entries.
pub fn idct16(x: &mut [i32; 16]) {
    // stage 1 — permutation.
    let s: [i32; 16] = [
        x[0], x[8], x[4], x[12], x[2], x[10], x[6], x[14], x[1], x[9], x[5], x[13], x[3], x[11],
        x[7], x[15],
    ];

    // stage 2 — odd-lane outer butterflies.
    let t8 = half_btf(COS_PI[60], s[8], -COS_PI[4], s[15]);
    let t15 = half_btf(COS_PI[4], s[8], COS_PI[60], s[15]);
    let t9 = half_btf(COS_PI[28], s[9], -COS_PI[36], s[14]);
    let t14 = half_btf(COS_PI[36], s[9], COS_PI[28], s[14]);
    let t10 = half_btf(COS_PI[44], s[10], -COS_PI[20], s[13]);
    let t13 = half_btf(COS_PI[20], s[10], COS_PI[44], s[13]);
    let t11 = half_btf(COS_PI[12], s[11], -COS_PI[52], s[12]);
    let t12 = half_btf(COS_PI[52], s[11], COS_PI[12], s[12]);

    // stage 3 — even-lane half-butterflies + odd-lane pair combinations.
    let u0 = half_btf(COS_PI[32], s[0], COS_PI[32], s[1]);
    let u1 = half_btf(COS_PI[32], s[0], -COS_PI[32], s[1]);
    let u2 = half_btf(COS_PI[48], s[2], -COS_PI[16], s[3]);
    let u3 = half_btf(COS_PI[16], s[2], COS_PI[48], s[3]);
    let u4 = half_btf(COS_PI[56], s[4], -COS_PI[8], s[7]);
    let u5 = half_btf(COS_PI[24], s[5], -COS_PI[40], s[6]);
    let u6 = half_btf(COS_PI[40], s[5], COS_PI[24], s[6]);
    let u7 = half_btf(COS_PI[8], s[4], COS_PI[56], s[7]);

    let u8 = t8 + t9;
    let u9 = t8 - t9;
    let u10 = -t10 + t11;
    let u11 = t10 + t11;
    let u12 = t12 + t13;
    let u13 = t12 - t13;
    let u14 = -t14 + t15;
    let u15 = t14 + t15;

    // stage 4 — finish IDCT8 structure on even lane + cross on odd.
    let v0 = u0 + u3;
    let v1 = u1 + u2;
    let v2 = u1 - u2;
    let v3 = u0 - u3;
    let v4 = u4 + u5;
    let v5 = u4 - u5;
    let v6 = -u6 + u7;
    let v7 = u6 + u7;
    let v8 = u8;
    let v9 = half_btf(-COS_PI[16], u9, COS_PI[48], u14);
    let v10 = half_btf(-COS_PI[48], u10, -COS_PI[16], u13);
    let v11 = u11;
    let v12 = u12;
    let v13 = half_btf(-COS_PI[16], u10, COS_PI[48], u13);
    let v14 = half_btf(COS_PI[48], u9, COS_PI[16], u14);
    let v15 = u15;

    // stage 5 — even-lane IDCT8 sums + odd-lane pair combinations.
    let w0 = v0 + v7;
    let w1 = v1 + v6;
    let w2 = v2 + v5;
    let w3 = v3 + v4;
    let w4 = v3 - v4;
    let w5 = v2 - v5;
    let w6 = v1 - v6;
    let w7 = v0 - v7;
    let w8 = v8 + v11;
    let w9 = v9 + v10;
    let w10 = v9 - v10;
    let w11 = v8 - v11;
    let w12 = -v12 + v15;
    let w13 = -v13 + v14;
    let w14 = v13 + v14;
    let w15 = v12 + v15;

    // stage 6 — cos_pi[32] cross on odd-lane inners.
    let x8 = w8;
    let x9 = w9;
    let x10 = half_btf(-COS_PI[32], w10, COS_PI[32], w13);
    let x11 = half_btf(-COS_PI[32], w11, COS_PI[32], w12);
    let x12 = half_btf(COS_PI[32], w11, COS_PI[32], w12);
    let x13 = half_btf(COS_PI[32], w10, COS_PI[32], w13);
    let x14 = w14;
    let x15 = w15;

    // stage 7 — final 16-wide butterfly.
    x[0] = w0 + x15;
    x[1] = w1 + x14;
    x[2] = w2 + x13;
    x[3] = w3 + x12;
    x[4] = w4 + x11;
    x[5] = w5 + x10;
    x[6] = w6 + x9;
    x[7] = w7 + x8;
    x[8] = w7 - x8;
    x[9] = w6 - x9;
    x[10] = w5 - x10;
    x[11] = w4 - x11;
    x[12] = w3 - x12;
    x[13] = w2 - x13;
    x[14] = w1 - x14;
    x[15] = w0 - x15;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct16_dc_reconstruction_constant() {
        let mut x = [0i32; 16];
        x[0] = 32768;
        idct16(&mut x);
        let first = x[0];
        for (i, v) in x.iter().enumerate().skip(1) {
            assert_eq!(*v, first, "non-constant at {i}");
        }
        assert_ne!(first, 0);
    }

    #[test]
    fn idct16_linearity_within_tolerance() {
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
        idct16(&mut ac);
        idct16(&mut bc);
        idct16(&mut sum);
        for i in 0..16 {
            let want = ac[i] + bc[i];
            let diff = (sum[i] - want).abs();
            assert!(diff <= 12, "linearity at {i}: diff {diff}");
        }
    }
}
