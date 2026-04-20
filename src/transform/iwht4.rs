//! 4-point inverse Walsh-Hadamard transform — §7.7.2.6.
//!
//! Used by AV1's lossless-only coding path; the `UNIT_QUANT_SHIFT = 2`
//! constant is baked into the first four shifts below. libaom:
//! "4-point reversible, orthonormal inverse WHT in 3.5 adds, 0.5
//! shifts per pixel."

const UNIT_QUANT_SHIFT: u32 = 2;

/// In-place 4-point inverse WHT. `x` must have exactly 4 entries.
pub fn iwht4(x: &mut [i32; 4]) {
    let mut a = x[0] >> UNIT_QUANT_SHIFT;
    let mut c = x[1] >> UNIT_QUANT_SHIFT;
    let mut d = x[2] >> UNIT_QUANT_SHIFT;
    let mut b = x[3] >> UNIT_QUANT_SHIFT;
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
}
