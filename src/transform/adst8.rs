//! 8-point inverse ADST — §7.7.2.4.
//!
//! Mirrors libaom's `av1_iadst8`: 7-stage butterfly with bit-reverse
//! permutation + final alternating negations.

use super::cos_pi::{half_btf, COS_PI};

/// In-place 8-point inverse ADST. `x` must have exactly 8 entries.
pub fn iadst8(x: &mut [i32; 8]) {
    // stage 1 — permutation.
    let s = [x[7], x[0], x[5], x[2], x[3], x[4], x[1], x[6]];

    // stage 2 — rotations.
    let t = [
        half_btf(COS_PI[4], s[0], COS_PI[60], s[1]),
        half_btf(COS_PI[60], s[0], -COS_PI[4], s[1]),
        half_btf(COS_PI[20], s[2], COS_PI[44], s[3]),
        half_btf(COS_PI[44], s[2], -COS_PI[20], s[3]),
        half_btf(COS_PI[36], s[4], COS_PI[28], s[5]),
        half_btf(COS_PI[28], s[4], -COS_PI[36], s[5]),
        half_btf(COS_PI[52], s[6], COS_PI[12], s[7]),
        half_btf(COS_PI[12], s[6], -COS_PI[52], s[7]),
    ];

    // stage 3 — pair add/subtract.
    let u = [
        t[0] + t[4],
        t[1] + t[5],
        t[2] + t[6],
        t[3] + t[7],
        t[0] - t[4],
        t[1] - t[5],
        t[2] - t[6],
        t[3] - t[7],
    ];

    // stage 4 — selective rotation on u[4..7].
    let v = [
        u[0],
        u[1],
        u[2],
        u[3],
        half_btf(COS_PI[16], u[4], COS_PI[48], u[5]),
        half_btf(COS_PI[48], u[4], -COS_PI[16], u[5]),
        half_btf(-COS_PI[48], u[6], COS_PI[16], u[7]),
        half_btf(COS_PI[16], u[6], COS_PI[48], u[7]),
    ];

    // stage 5 — pair add/subtract.
    let w = [
        v[0] + v[2],
        v[1] + v[3],
        v[0] - v[2],
        v[1] - v[3],
        v[4] + v[6],
        v[5] + v[7],
        v[4] - v[6],
        v[5] - v[7],
    ];

    // stage 6 — rotations on w[2..3] and w[6..7].
    let y = [
        w[0],
        w[1],
        half_btf(COS_PI[32], w[2], COS_PI[32], w[3]),
        half_btf(COS_PI[32], w[2], -COS_PI[32], w[3]),
        w[4],
        w[5],
        half_btf(COS_PI[32], w[6], COS_PI[32], w[7]),
        half_btf(COS_PI[32], w[6], -COS_PI[32], w[7]),
    ];

    // stage 7 — permutation + alternating negations.
    x[0] = y[0];
    x[1] = -y[4];
    x[2] = y[6];
    x[3] = -y[2];
    x[4] = y[3];
    x[5] = -y[7];
    x[6] = y[5];
    x[7] = -y[1];
}

/// In-place 8-point inverse flipped ADST.
pub fn iflipadst8(x: &mut [i32; 8]) {
    iadst8(x);
    x.swap(0, 7);
    x.swap(1, 6);
    x.swap(2, 5);
    x.swap(3, 4);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iadst8_linearity_within_tolerance() {
        let a = [100i32, 50, -25, 10, 5, -40, 20, 15];
        let b = [-30i32, 40, 5, -12, 22, 7, -18, 3];
        let mut sum = [0i32; 8];
        for i in 0..8 {
            sum[i] = a[i] + b[i];
        }
        let mut ac = a;
        let mut bc = b;
        iadst8(&mut ac);
        iadst8(&mut bc);
        iadst8(&mut sum);
        for i in 0..8 {
            let want = ac[i] + bc[i];
            let diff = (sum[i] - want).abs();
            assert!(diff <= 4, "linearity at {i}: diff {diff}");
        }
    }

    #[test]
    fn iflipadst8_reverses_iadst8() {
        let mut a = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let mut af = a;
        iadst8(&mut a);
        iflipadst8(&mut af);
        for i in 0..8 {
            assert_eq!(af[i], a[7 - i]);
        }
    }
}
