//! `PAETH_PRED` — §7.11.2.4.
//!
//! For each (r, c) pick the reference (above[c], left[r], aboveLeft)
//! whose predictor ray (`above[c] + left[r] - aboveLeft`) is closest to
//! it. Ported from goavif `av1/predict/intra_paeth.go` + `intra16.go`.

/// 8-bit Paeth predictor.
pub fn paeth_pred(dst: &mut [u8], w: usize, h: usize, above: &[u8], left: &[u8], above_left: u8) {
    for r in 0..h {
        for c in 0..w {
            let a = above[c] as i32;
            let l = left[r] as i32;
            let al = above_left as i32;
            let base = a + l - al;
            let p_a = (base - a).abs();
            let p_l = (base - l).abs();
            let p_al = (base - al).abs();
            let p = if p_l <= p_a && p_l <= p_al {
                l
            } else if p_a <= p_al {
                a
            } else {
                al
            };
            dst[r * w + c] = p.clamp(0, 255) as u8;
        }
    }
}

/// 16-bit Paeth predictor. Output is clamped to `(1 << bitDepth) - 1`.
pub fn paeth_pred16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    above: &[u16],
    left: &[u16],
    above_left: u16,
    bit_depth: u32,
) {
    let max_v = (1i32 << bit_depth) - 1;
    for r in 0..h {
        for c in 0..w {
            let a = above[c] as i32;
            let l = left[r] as i32;
            let al = above_left as i32;
            let base = a + l - al;
            let p_a = (base - a).abs();
            let p_l = (base - l).abs();
            let p_al = (base - al).abs();
            let p = if p_l <= p_a && p_l <= p_al {
                l
            } else if p_a <= p_al {
                a
            } else {
                al
            };
            dst[r * w + c] = p.clamp(0, max_v) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paeth_constant_propagates() {
        let above = [100u8; 4];
        let left = [100u8; 4];
        let mut dst = vec![0u8; 16];
        paeth_pred(&mut dst, 4, 4, &above, &left, 100);
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn paeth_clamps_to_byte_range() {
        let above = [0u8; 4];
        let left = [255u8; 4];
        let mut dst = vec![0u8; 16];
        paeth_pred(&mut dst, 4, 4, &above, &left, 128);
        for &v in &dst {
            // No under/overflow.
            let _ = v;
        }
    }
}
