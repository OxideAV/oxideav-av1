//! Chroma-from-luma (CFL) — §7.11.5.
//!
//! CFL reconstructs chroma by combining a per-block DC-style prediction
//! with a scaled, AC-only copy of the co-located reconstructed luma
//! block.
//!
//! Ported from goavif `av1/predict/intra_cfl.go`.

/// Compute the chroma-resolution luma average by block-averaging
/// `recon_luma` per the given subsampling factors. Output `dst` carries
/// `chroma_w * chroma_h` Q3-scaled (×8) averages.
pub fn cfl_subsample(
    dst: &mut [i32],
    recon_luma: &[u8],
    luma_w: usize,
    luma_h: usize,
    sub_x: usize,
    sub_y: usize,
) {
    let chroma_w = luma_w >> sub_x;
    let chroma_h = luma_h >> sub_y;
    let step_x = 1usize << sub_x;
    let step_y = 1usize << sub_y;
    let box_area = (step_x * step_y) as i32;
    for r in 0..chroma_h {
        for c in 0..chroma_w {
            let mut sum: i32 = 0;
            for dy in 0..step_y {
                for dx in 0..step_x {
                    sum += recon_luma[(r * step_y + dy) * luma_w + (c * step_x + dx)] as i32;
                }
            }
            dst[r * chroma_w + c] = sum * 8 / box_area;
        }
    }
}

/// 16-bit `cfl_subsample`.
pub fn cfl_subsample16(
    dst: &mut [i32],
    recon_luma: &[u16],
    luma_w: usize,
    luma_h: usize,
    sub_x: usize,
    sub_y: usize,
) {
    let chroma_w = luma_w >> sub_x;
    let chroma_h = luma_h >> sub_y;
    let step_x = 1usize << sub_x;
    let step_y = 1usize << sub_y;
    let box_area = (step_x * step_y) as i32;
    for r in 0..chroma_h {
        for c in 0..chroma_w {
            let mut sum: i32 = 0;
            for dy in 0..step_y {
                for dx in 0..step_x {
                    sum += recon_luma[(r * step_y + dy) * luma_w + (c * step_x + dx)] as i32;
                }
            }
            dst[r * chroma_w + c] = sum * 8 / box_area;
        }
    }
}

/// CFL prediction: `dst = clip(dc_pred + ((alpha * (luma_q3 - mean) + 32) >> 6))`.
///
/// `alpha` is signed Q6 (-16..=16 magnitude range × sign). Positive
/// alpha copies the luma AC onto chroma; negative inverts.
pub fn cfl_pred(
    dst: &mut [u8],
    w: usize,
    h: usize,
    luma_q3: &[i32],
    dc_pred: &[u8],
    alpha: i32,
) {
    let n = w * h;
    let sum: i64 = luma_q3.iter().take(n).map(|&v| v as i64).sum();
    let half = (n as i64) / 2;
    let avg = ((sum + half) / (n as i64)) as i32;
    for i in 0..n {
        let ac = luma_q3[i] - avg;
        let scaled = ((alpha * ac) + (1 << 5)) >> 6;
        let v = (dc_pred[i] as i32) + scaled;
        dst[i] = v.clamp(0, 255) as u8;
    }
}

/// 16-bit CFL prediction. Output is clipped to `(1 << bitDepth) - 1`.
pub fn cfl_pred16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    luma_q3: &[i32],
    dc_pred: &[u16],
    alpha: i32,
    bit_depth: u32,
) {
    let n = w * h;
    let sum: i64 = luma_q3.iter().take(n).map(|&v| v as i64).sum();
    let half = (n as i64) / 2;
    let avg = ((sum + half) / (n as i64)) as i32;
    let max_v = ((1i32) << bit_depth) - 1;
    for i in 0..n {
        let ac = luma_q3[i] - avg;
        let scaled = ((alpha * ac) + (1 << 5)) >> 6;
        let v = (dc_pred[i] as i32) + scaled;
        dst[i] = v.clamp(0, max_v) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsample_420_flat() {
        let luma = vec![100u8; 16];
        let mut chroma = vec![0i32; 4];
        cfl_subsample(&mut chroma, &luma, 4, 4, 1, 1);
        for &v in &chroma {
            assert_eq!(v, 800);
        }
    }

    #[test]
    fn zero_alpha_returns_dc() {
        let luma = vec![500, 600, 700, 800];
        let dc = vec![50u8, 60, 70, 80];
        let mut dst = vec![0u8; 4];
        cfl_pred(&mut dst, 2, 2, &luma, &dc, 0);
        assert_eq!(&dst, &dc);
    }

    #[test]
    fn positive_alpha_follows_luma_contrast() {
        let mut luma = vec![100i32; 16];
        luma[5] = 2000;
        let dc = vec![128u8; 16];
        let mut dst = vec![0u8; 16];
        cfl_pred(&mut dst, 4, 4, &luma, &dc, 8);
        assert!(dst[5] > dst[0], "alpha>0 should lift sample 5");
    }
}
