//! Identity inverse transforms — §7.7.2.7.
//!
//! The IDTX kernels leave their input structurally unchanged and apply
//! a size-dependent scale that the spec's txfm_range tables expect:
//!
//! - `idtx4` / `idtx8` / `idtx16` — left-shift by 1.
//! - `idtx32` — no scaling (the range table already accounts for
//!   identity at N = 32).

/// 4-point identity inverse transform.
pub fn idtx4(x: &mut [i32; 4]) {
    for v in x.iter_mut() {
        *v <<= 1;
    }
}

/// 8-point identity inverse transform.
pub fn idtx8(x: &mut [i32; 8]) {
    for v in x.iter_mut() {
        *v <<= 1;
    }
}

/// 16-point identity inverse transform.
pub fn idtx16(x: &mut [i32; 16]) {
    for v in x.iter_mut() {
        *v <<= 1;
    }
}

/// 32-point identity inverse transform — no scale.
pub fn idtx32(_x: &mut [i32; 32]) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idtx4_doubles_each_sample() {
        let mut v = [1i32, -2, 3, -4];
        idtx4(&mut v);
        assert_eq!(v, [2, -4, 6, -8]);
    }

    #[test]
    fn idtx32_is_identity() {
        let mut v = [0i32; 32];
        for (i, cell) in v.iter_mut().enumerate() {
            *cell = i as i32 - 16;
        }
        let before = v;
        idtx32(&mut v);
        assert_eq!(v, before);
    }
}
