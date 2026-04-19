//! `V_PRED` / `H_PRED` — §7.11.2.2. Copy the above row down / left
//! column across, respectively.
//!
//! Ported from goavif `av1/predict/intra_vh.go`.

/// Vertical predictor: each row of `dst` is a copy of `above[..w]`.
pub fn v_pred(dst: &mut [u8], w: usize, h: usize, above: &[u8]) {
    for r in 0..h {
        dst[r * w..(r + 1) * w].copy_from_slice(&above[..w]);
    }
}

/// Horizontal predictor: each column of `dst` replicates `left[r]`.
pub fn h_pred(dst: &mut [u8], w: usize, h: usize, left: &[u8]) {
    for r in 0..h {
        let v = left[r];
        for c in 0..w {
            dst[r * w + c] = v;
        }
    }
}

/// 16-bit vertical predictor.
pub fn v_pred16(dst: &mut [u16], w: usize, h: usize, above: &[u16]) {
    for r in 0..h {
        dst[r * w..(r + 1) * w].copy_from_slice(&above[..w]);
    }
}

/// 16-bit horizontal predictor.
pub fn h_pred16(dst: &mut [u16], w: usize, h: usize, left: &[u16]) {
    for r in 0..h {
        let v = left[r];
        for c in 0..w {
            dst[r * w + c] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v_pred_copies_row() {
        let above = [1, 2, 3, 4];
        let mut dst = vec![0u8; 16];
        v_pred(&mut dst, 4, 4, &above);
        for r in 0..4 {
            assert_eq!(&dst[r * 4..r * 4 + 4], &above[..]);
        }
    }

    #[test]
    fn h_pred_copies_column() {
        let left = [1, 2, 3, 4];
        let mut dst = vec![0u8; 16];
        h_pred(&mut dst, 4, 4, &left);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(dst[r * 4 + c], left[r]);
            }
        }
    }
}
