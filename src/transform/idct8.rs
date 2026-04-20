//! 8-point inverse DCT — §7.7.2.1.

use super::cos_pi::{half_btf, COS_PI};

/// In-place 8-point inverse DCT. `x` must have exactly 8 entries.
pub fn idct8(x: &mut [i32; 8]) {
    // stage 1 — reordered taps.
    let s0 = x[0];
    let s1 = x[4];
    let s2 = x[2];
    let s3 = x[6];
    let s4 = x[1];
    let s5 = x[5];
    let s6 = x[3];
    let s7 = x[7];

    // stage 2.
    let t4 = half_btf(COS_PI[56], s4, -COS_PI[8], s7);
    let t5 = half_btf(COS_PI[24], s5, -COS_PI[40], s6);
    let t6 = half_btf(COS_PI[40], s5, COS_PI[24], s6);
    let t7 = half_btf(COS_PI[8], s4, COS_PI[56], s7);

    // stage 3 — IDCT4 on the even lane.
    let u0 = half_btf(COS_PI[32], s0, COS_PI[32], s1);
    let u1 = half_btf(COS_PI[32], s0, -COS_PI[32], s1);
    let u2 = half_btf(COS_PI[48], s2, -COS_PI[16], s3);
    let u3 = half_btf(COS_PI[16], s2, COS_PI[48], s3);
    let u4 = t4 + t5;
    let u5 = t4 - t5;
    let u6 = -t6 + t7;
    let u7 = t6 + t7;

    // stage 4 — finish IDCT4 cross + mix odd.
    let v0 = u0 + u3;
    let v1 = u1 + u2;
    let v2 = u1 - u2;
    let v3 = u0 - u3;
    let v4 = u4;
    let v5 = half_btf(-COS_PI[32], u5, COS_PI[32], u6);
    let v6 = half_btf(COS_PI[32], u5, COS_PI[32], u6);
    let v7 = u7;

    // stage 5 — final butterfly.
    x[0] = v0 + v7;
    x[1] = v1 + v6;
    x[2] = v2 + v5;
    x[3] = v3 + v4;
    x[4] = v3 - v4;
    x[5] = v2 - v5;
    x[6] = v1 - v6;
    x[7] = v0 - v7;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idct8_dc_reconstruction_constant() {
        let mut x = [0i32; 8];
        x[0] = 16384;
        idct8(&mut x);
        let first = x[0];
        for (i, v) in x.iter().enumerate().skip(1) {
            assert_eq!(*v, first, "non-constant at {i}");
        }
        assert_ne!(first, 0);
    }

    #[test]
    fn idct8_linearity_within_tolerance() {
        let a = [100i32, 50, -25, 10, 5, -40, 20, 15];
        let b = [-30i32, 40, 5, -12, 22, 7, -18, 3];
        let mut sum = [0i32; 8];
        for i in 0..8 {
            sum[i] = a[i] + b[i];
        }
        let mut ac = a;
        let mut bc = b;
        idct8(&mut ac);
        idct8(&mut bc);
        idct8(&mut sum);
        for i in 0..8 {
            let want = ac[i] + bc[i];
            let diff = (sum[i] - want).abs();
            assert!(diff <= 4, "linearity at {i}: diff {diff}");
        }
    }
}
