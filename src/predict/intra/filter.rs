//! Filter-intra prediction — §7.11.2.7.
//!
//! Block is processed in 4×2 cells; each cell predicts 8 new samples
//! from a row of 5 above samples (including the top-left corner) and 2
//! left samples via an 8×7 coefficient matrix per mode.
//!
//! Ported from goavif `av1/predict/intra_filter.go`.

/// Five learned tap sets from libaom's `av1_filter_intra_taps`. Shape:
/// `[mode][output_idx][input_idx]`. The 8 outputs are the 8 samples of a
/// 4×2 cell; the 7 inputs are the corner/above/left reference samples.
/// The 8th column is unused (padded with 0) to keep the inner loop square.
pub const FILTER_INTRA_TAPS: [[[i8; 8]; 8]; 5] = [
    // Mode 0
    [
        [-6, 10, 0, 0, 0, 12, 0, 0],
        [-5, 2, 10, 0, 0, 9, 0, 0],
        [-3, 1, 1, 10, 0, 7, 0, 0],
        [-3, 1, 1, 2, 10, 5, 0, 0],
        [-4, 6, 0, 0, 0, 2, 12, 0],
        [-3, 2, 6, 0, 0, 2, 9, 0],
        [-3, 2, 2, 6, 0, 2, 7, 0],
        [-3, 1, 2, 2, 6, 3, 5, 0],
    ],
    // Mode 1
    [
        [-10, 16, 0, 0, 0, 10, 0, 0],
        [-6, 0, 16, 0, 0, 6, 0, 0],
        [-4, 0, 0, 16, 0, 4, 0, 0],
        [-2, 0, 0, 0, 16, 2, 0, 0],
        [-10, 16, 0, 0, 0, 0, 10, 0],
        [-6, 0, 16, 0, 0, 0, 6, 0],
        [-4, 0, 0, 16, 0, 0, 4, 0],
        [-2, 0, 0, 0, 16, 0, 2, 0],
    ],
    // Mode 2
    [
        [-8, 8, 0, 0, 0, 16, 0, 0],
        [-8, 0, 8, 0, 0, 16, 0, 0],
        [-8, 0, 0, 8, 0, 16, 0, 0],
        [-8, 0, 0, 0, 8, 16, 0, 0],
        [-4, 4, 0, 0, 0, 0, 16, 0],
        [-4, 0, 4, 0, 0, 0, 16, 0],
        [-4, 0, 0, 4, 0, 0, 16, 0],
        [-4, 0, 0, 0, 4, 0, 16, 0],
    ],
    // Mode 3
    [
        [-2, 8, 0, 0, 0, 10, 0, 0],
        [-1, 3, 8, 0, 0, 6, 0, 0],
        [-1, 2, 3, 8, 0, 4, 0, 0],
        [0, 1, 2, 3, 8, 2, 0, 0],
        [-1, 4, 0, 0, 0, 3, 10, 0],
        [-1, 3, 4, 0, 0, 4, 6, 0],
        [-1, 2, 3, 4, 0, 4, 4, 0],
        [-1, 2, 2, 3, 4, 3, 3, 0],
    ],
    // Mode 4
    [
        [-12, 14, 0, 0, 0, 14, 0, 0],
        [-10, 0, 14, 0, 0, 12, 0, 0],
        [-9, 0, 0, 14, 0, 11, 0, 0],
        [-8, 0, 0, 0, 14, 10, 0, 0],
        [-10, 12, 0, 0, 0, 0, 14, 0],
        [-9, 1, 12, 0, 0, 0, 12, 0],
        [-8, 0, 0, 12, 0, 1, 11, 0],
        [-7, 0, 0, 1, 12, 1, 9, 0],
    ],
];

/// Filter-intra prediction for blocks up to 32×32. `above`/`left` must
/// be extended so the algorithm can read `w+1` above samples starting
/// from index 0 (including the corner) and `h` left samples.
/// `above_left` is the (-1, -1) corner reference.
#[allow(clippy::needless_range_loop)]
pub fn filter_intra_pred(
    dst: &mut [u8],
    w: usize,
    h: usize,
    above: &[u8],
    left: &[u8],
    above_left: u8,
    mode: usize,
) {
    let mode = if mode >= FILTER_INTRA_TAPS.len() {
        0
    } else {
        mode
    };
    let mut buf = [[0u8; 33]; 33];
    buf[0][0] = above_left;
    buf[0][1..=w].copy_from_slice(&above[..w]);
    for r in 0..h {
        buf[r + 1][0] = left[r];
    }

    let mut r = 1usize;
    while r < h + 1 {
        let mut c = 1usize;
        while c < w + 1 {
            let p0 = buf[r - 1][c - 1] as i32;
            let p1 = buf[r - 1][c] as i32;
            let p2 = buf[r - 1][c + 1] as i32;
            let p3 = buf[r - 1][c + 2] as i32;
            let p4 = buf[r - 1][c + 3] as i32;
            let p5 = buf[r][c - 1] as i32;
            let p6 = buf[r + 1][c - 1] as i32;
            for (k, t) in FILTER_INTRA_TAPS[mode].iter().enumerate() {
                let pr = (t[0] as i32) * p0
                    + (t[1] as i32) * p1
                    + (t[2] as i32) * p2
                    + (t[3] as i32) * p3
                    + (t[4] as i32) * p4
                    + (t[5] as i32) * p5
                    + (t[6] as i32) * p6;
                let pr = ((pr + 8) >> 4).clamp(0, 255);
                let dr = r + k / 4;
                let dc = c + k % 4;
                buf[dr][dc] = pr as u8;
            }
            c += 4;
        }
        r += 2;
    }

    for row in 0..h {
        for col in 0..w {
            dst[row * w + col] = buf[row + 1][col + 1];
        }
    }
}

/// 16-bit filter-intra prediction. Output is clipped to
/// `(1 << bitDepth) - 1`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn filter_intra_pred16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    above: &[u16],
    left: &[u16],
    above_left: u16,
    mode: usize,
    bit_depth: u32,
) {
    let mode = if mode >= FILTER_INTRA_TAPS.len() {
        0
    } else {
        mode
    };
    let max_v = ((1i32) << bit_depth) - 1;
    let mut buf = [[0u16; 33]; 33];
    buf[0][0] = above_left;
    buf[0][1..=w].copy_from_slice(&above[..w]);
    for r in 0..h {
        buf[r + 1][0] = left[r];
    }

    let mut r = 1usize;
    while r < h + 1 {
        let mut c = 1usize;
        while c < w + 1 {
            let p0 = buf[r - 1][c - 1] as i32;
            let p1 = buf[r - 1][c] as i32;
            let p2 = buf[r - 1][c + 1] as i32;
            let p3 = buf[r - 1][c + 2] as i32;
            let p4 = buf[r - 1][c + 3] as i32;
            let p5 = buf[r][c - 1] as i32;
            let p6 = buf[r + 1][c - 1] as i32;
            for (k, t) in FILTER_INTRA_TAPS[mode].iter().enumerate() {
                let pr = (t[0] as i32) * p0
                    + (t[1] as i32) * p1
                    + (t[2] as i32) * p2
                    + (t[3] as i32) * p3
                    + (t[4] as i32) * p4
                    + (t[5] as i32) * p5
                    + (t[6] as i32) * p6;
                let pr = ((pr + 8) >> 4).clamp(0, max_v);
                let dr = r + k / 4;
                let dc = c + k % 4;
                buf[dr][dc] = pr as u16;
            }
            c += 4;
        }
        r += 2;
    }

    for row in 0..h {
        for col in 0..w {
            dst[row * w + col] = buf[row + 1][col + 1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_modes_run_without_panic() {
        let above = [100u8; 8];
        let left = [100u8; 8];
        let mut dst = vec![0u8; 16];
        for m in 0..5 {
            for v in dst.iter_mut() {
                *v = 0;
            }
            filter_intra_pred(&mut dst, 4, 4, &above, &left, 100, m);
        }
    }

    #[test]
    fn mode_out_of_range_clamps() {
        let above = [128u8; 8];
        let left = [128u8; 8];
        let mut dst = vec![0u8; 16];
        filter_intra_pred(&mut dst, 4, 4, &above, &left, 128, 99);
    }
}
