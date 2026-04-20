//! `DC_PRED` — §7.11.2.3. Mean of available neighbours, rounded
//! half-up. When neither neighbour is available the half-range value
//! `1 << (bitDepth - 1)` is used (128 for 8-bit).

/// 8-bit DC predictor. `dst` is row-major, tight (`stride == w`).
#[allow(clippy::too_many_arguments)]
pub fn dc_pred(
    dst: &mut [u8],
    w: usize,
    h: usize,
    above: &[u8],
    left: &[u8],
    have_above: bool,
    have_left: bool,
    bit_depth: u32,
) {
    let mut sum: i64 = 0;
    let mut n: i64 = 0;
    if have_above {
        for &v in above.iter().take(w) {
            sum += v as i64;
        }
        n += w as i64;
    }
    if have_left {
        for &v in left.iter().take(h) {
            sum += v as i64;
        }
        n += h as i64;
    }
    let dc: u8 = if n == 0 {
        (1u32 << (bit_depth - 1)).min(255) as u8
    } else {
        ((sum + n / 2) / n) as u8
    };
    for row in 0..h {
        for col in 0..w {
            dst[row * w + col] = dc;
        }
    }
}

/// 16-bit DC predictor. Values are clamped to `(1 << bitDepth) - 1`.
#[allow(clippy::too_many_arguments)]
pub fn dc_pred16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    above: &[u16],
    left: &[u16],
    have_above: bool,
    have_left: bool,
    bit_depth: u32,
) {
    let mut sum: i64 = 0;
    let mut n: i64 = 0;
    if have_above {
        for &v in above.iter().take(w) {
            sum += v as i64;
        }
        n += w as i64;
    }
    if have_left {
        for &v in left.iter().take(h) {
            sum += v as i64;
        }
        n += h as i64;
    }
    let dc: u16 = if n == 0 {
        1u16 << (bit_depth - 1)
    } else {
        ((sum + n / 2) / n) as u16
    };
    for row in 0..h {
        for col in 0..w {
            dst[row * w + col] = dc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_both_neighbours() {
        let mut dst = vec![0u8; 16];
        let above = [10, 20, 30, 40];
        let left = [40, 50, 60, 70];
        dc_pred(&mut dst, 4, 4, &above, &left, true, true, 8);
        for &v in &dst {
            assert_eq!(v, 40);
        }
    }

    #[test]
    fn dc_no_neighbours() {
        let mut dst = vec![0u8; 16];
        dc_pred(&mut dst, 4, 4, &[], &[], false, false, 8);
        for &v in &dst {
            assert_eq!(v, 128);
        }
    }

    #[test]
    fn dc_left_only() {
        let mut dst = vec![0u8; 16];
        let left = [100, 100, 100, 100];
        dc_pred(&mut dst, 4, 4, &[], &left, false, true, 8);
        for &v in &dst {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn dc16_no_neighbours_uses_midrange_10bit() {
        let mut dst = vec![0u16; 16];
        dc_pred16(&mut dst, 4, 4, &[], &[], false, false, 10);
        for &v in &dst {
            assert_eq!(v, 512);
        }
    }
}
