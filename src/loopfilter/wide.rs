//! Wide (8-tap) deblocking filter — §7.14.4.3 + §7.14.6.3.
//!
//! Operates on 8 samples straddling the edge (p3..p0 | q0..q3) and
//! replaces the inner 6 (p2..p0, q0..q2) with weighted 8-tap averages.

#[inline]
fn abs_diff(a: u8, b: u8) -> u8 {
    a.abs_diff(b)
}

/// 8-sample "flat" mask: all inner differences within a small (×1)
/// threshold (spec §7.14.4.3).
#[allow(clippy::too_many_arguments)]
pub fn flat8_mask(p3: u8, p2: u8, p1: u8, p0: u8, q0: u8, q1: u8, q2: u8, q3: u8) -> bool {
    const THRESH: u8 = 1;
    abs_diff(p1, p0) <= THRESH
        && abs_diff(q1, q0) <= THRESH
        && abs_diff(p2, p0) <= THRESH
        && abs_diff(q2, q0) <= THRESH
        && abs_diff(p3, p0) <= THRESH
        && abs_diff(q3, q0) <= THRESH
}

#[inline]
fn avg8(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32) -> u8 {
    let s = a + b + c + d + e + f + g + 4;
    ((s >> 3) & 0xFF) as u8
}

/// Wide 8-tap deblocking filter (spec §7.14.6.3). Returns
/// `(new_p2, new_p1, new_p0, new_q0, new_q1, new_q2)`. `p3` and `q3`
/// contribute to the averages but are not themselves updated.
#[allow(clippy::too_many_arguments)]
pub fn filter8(
    p3: u8,
    p2: u8,
    p1: u8,
    p0: u8,
    q0: u8,
    q1: u8,
    q2: u8,
    q3: u8,
) -> (u8, u8, u8, u8, u8, u8) {
    let p3 = p3 as i32;
    let p2 = p2 as i32;
    let p1 = p1 as i32;
    let p0 = p0 as i32;
    let q0 = q0 as i32;
    let q1 = q1 as i32;
    let q2 = q2 as i32;
    let q3 = q3 as i32;
    let np2 = avg8(p3, p3, p3, 2 * p2, p1, p0, q0);
    let np1 = avg8(p3, p3, p2, 2 * p1, p0, q0, q1);
    let np0 = avg8(p3, p2, p1, 2 * p0, q0, q1, q2);
    let nq0 = avg8(p2, p1, p0, 2 * q0, q1, q2, q3);
    let nq1 = avg8(p1, p0, q0, 2 * q1, q2, q3, q3);
    let nq2 = avg8(p0, q0, q1, 2 * q2, q3, q3, q3);
    (np2, np1, np0, nq0, nq1, nq2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_mask_uniform_passes() {
        assert!(flat8_mask(100, 100, 100, 100, 100, 100, 100, 100));
    }

    #[test]
    fn flat_mask_rejects_spike() {
        assert!(!flat8_mask(100, 100, 100, 100, 100, 100, 200, 100));
    }

    #[test]
    fn filter8_flat_input_is_identity() {
        let (p2, p1, p0, q0, q1, q2) = filter8(100, 100, 100, 100, 100, 100, 100, 100);
        assert_eq!(p2, 100);
        assert_eq!(p1, 100);
        assert_eq!(p0, 100);
        assert_eq!(q0, 100);
        assert_eq!(q1, 100);
        assert_eq!(q2, 100);
    }
}
