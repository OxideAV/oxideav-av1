//! `SMOOTH_PRED` / `SMOOTH_V_PRED` / `SMOOTH_H_PRED` — §7.11.2.6.

/// SM_WEIGHTS table for block dimension 4.
pub const SM_WEIGHTS4: [u16; 4] = [255, 149, 85, 64];
/// SM_WEIGHTS table for block dimension 8.
pub const SM_WEIGHTS8: [u16; 8] = [255, 197, 146, 105, 73, 50, 37, 32];
/// SM_WEIGHTS table for block dimension 16.
pub const SM_WEIGHTS16: [u16; 16] = [
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
];
/// SM_WEIGHTS table for block dimension 32.
pub const SM_WEIGHTS32: [u16; 32] = [
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59, 52, 45,
    39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
];
/// SM_WEIGHTS table for block dimension 64.
pub const SM_WEIGHTS64: [u16; 64] = [
    255, 248, 242, 235, 228, 222, 215, 209, 202, 196, 190, 183, 177, 171, 165, 159, 153, 148, 142,
    137, 131, 126, 121, 116, 111, 106, 101, 96, 91, 87, 83, 78, 74, 70, 66, 63, 59, 56, 53, 50, 47,
    44, 41, 39, 36, 34, 32, 30, 28, 26, 24, 23, 21, 20, 18, 17, 16, 15, 14, 13, 12, 11, 10, 10,
];

/// Return the `sm_weights` entry for a given power-of-two dimension in
/// {4, 8, 16, 32, 64}. Panics for other sizes.
pub fn sm_weight_table(n: usize) -> &'static [u16] {
    match n {
        4 => &SM_WEIGHTS4,
        8 => &SM_WEIGHTS8,
        16 => &SM_WEIGHTS16,
        32 => &SM_WEIGHTS32,
        64 => &SM_WEIGHTS64,
        _ => panic!("predict: sm_weights: unsupported block size {n}"),
    }
}

/// SMOOTH_PRED: all 4 edges participate — above row, left column,
/// bottom-left sample, top-right sample (spec §7.11.2.6).
pub fn smooth_pred(dst: &mut [u8], w: usize, h: usize, above: &[u8], left: &[u8]) {
    let wh = sm_weight_table(h);
    let ww = sm_weight_table(w);
    let below_pred = left[h - 1] as u32;
    let right_pred = above[w - 1] as u32;
    for r in 0..h {
        let wr = wh[r] as u32;
        for c in 0..w {
            let wc = ww[c] as u32;
            let pred = wr * (above[c] as u32)
                + (256 - wr) * below_pred
                + wc * (left[r] as u32)
                + (256 - wc) * right_pred;
            dst[r * w + c] = ((pred + 256) >> 9) as u8;
        }
    }
}

/// SMOOTH_V_PRED: vertical interpolation between `above[c]` and
/// `left[h - 1]`.
pub fn smooth_v_pred(dst: &mut [u8], w: usize, h: usize, above: &[u8], left: &[u8]) {
    let wh = sm_weight_table(h);
    let below_pred = left[h - 1] as u32;
    for r in 0..h {
        let wr = wh[r] as u32;
        for c in 0..w {
            let pred = wr * (above[c] as u32) + (256 - wr) * below_pred;
            dst[r * w + c] = ((pred + 128) >> 8) as u8;
        }
    }
}

/// SMOOTH_H_PRED: horizontal interpolation between `left[r]` and
/// `above[w - 1]`.
pub fn smooth_h_pred(dst: &mut [u8], w: usize, h: usize, above: &[u8], left: &[u8]) {
    let ww = sm_weight_table(w);
    let right_pred = above[w - 1] as u32;
    for r in 0..h {
        for c in 0..w {
            let wc = ww[c] as u32;
            let pred = wc * (left[r] as u32) + (256 - wc) * right_pred;
            dst[r * w + c] = ((pred + 128) >> 8) as u8;
        }
    }
}

/// 16-bit SMOOTH_PRED.
pub fn smooth_pred16(dst: &mut [u16], w: usize, h: usize, above: &[u16], left: &[u16]) {
    let wh = sm_weight_table(h);
    let ww = sm_weight_table(w);
    let below_pred = left[h - 1] as u32;
    let right_pred = above[w - 1] as u32;
    for r in 0..h {
        let wr = wh[r] as u32;
        for c in 0..w {
            let wc = ww[c] as u32;
            let pred = wr * (above[c] as u32)
                + (256 - wr) * below_pred
                + wc * (left[r] as u32)
                + (256 - wc) * right_pred;
            dst[r * w + c] = ((pred + 256) >> 9) as u16;
        }
    }
}

/// 16-bit SMOOTH_V_PRED.
pub fn smooth_v_pred16(dst: &mut [u16], w: usize, h: usize, above: &[u16], left: &[u16]) {
    let wh = sm_weight_table(h);
    let below_pred = left[h - 1] as u32;
    for r in 0..h {
        let wr = wh[r] as u32;
        for c in 0..w {
            let pred = wr * (above[c] as u32) + (256 - wr) * below_pred;
            dst[r * w + c] = ((pred + 128) >> 8) as u16;
        }
    }
}

/// 16-bit SMOOTH_H_PRED.
pub fn smooth_h_pred16(dst: &mut [u16], w: usize, h: usize, above: &[u16], left: &[u16]) {
    let ww = sm_weight_table(w);
    let right_pred = above[w - 1] as u32;
    for r in 0..h {
        for c in 0..w {
            let wc = ww[c] as u32;
            let pred = wc * (left[r] as u32) + (256 - wc) * right_pred;
            dst[r * w + c] = ((pred + 128) >> 8) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooth_constant_neighbours_yields_constant() {
        // With all-100 edges the SMOOTH blend reduces to 100 (every term
        // averages to 100).
        let above = [100u8; 4];
        let left = [100u8; 4];
        let mut dst = vec![0u8; 16];
        smooth_pred(&mut dst, 4, 4, &above, &left);
        for (i, &v) in dst.iter().enumerate() {
            assert!((v as i32 - 100).abs() <= 1, "dst[{i}]={v}");
        }
    }

    #[test]
    fn smooth_h_reads_column() {
        let above = [50u8, 60, 70, 80];
        let left = [50u8, 60, 70, 80];
        let mut dst = vec![0u8; 16];
        smooth_h_pred(&mut dst, 4, 4, &above, &left);
        // Leftmost column should closely track `left[r]` (wc=255).
        for r in 0..4 {
            assert_eq!(dst[r * 4], left[r]);
        }
    }
}
