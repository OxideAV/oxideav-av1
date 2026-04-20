//! Wiener 7×7 separable FIR — spec §7.17.3.
//!
//! The 7-element symmetric kernel is materialised from a 4-element
//! tap array as:
//!
//! ```text
//! {taps[0], taps[1], taps[2], taps[3], taps[2], taps[1], taps[0]}
//! ```
//!
//! where `taps[3]` is the center coefficient (strongest) and outer
//! coefficients are typically negative. Each coefficient is in a Q7
//! fixed-point form (values scaled so that the full kernel sums to
//! ≈ 128 — i.e. unity gain after the `+ 64) >> 7` rounding).

/// 4-element Q7 tap array — holds one axis of the symmetric 7-tap
/// Wiener kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WienerTaps(pub [i32; 4]);

impl WienerTaps {
    /// Identity kernel — all weight on the center tap (`128 / 128 == 1`).
    pub const IDENTITY: Self = Self([0, 0, 0, 128]);

    /// Construct from 4 explicit coefficients.
    pub const fn new(a: i32, b: i32, c: i32, d: i32) -> Self {
        Self([a, b, c, d])
    }
}

#[inline]
fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

#[inline]
fn clip_bd(v: i32, bit_depth: u32) -> u16 {
    let max_v = (1i32 << bit_depth) - 1;
    v.clamp(0, max_v) as u16
}

#[inline]
fn tap_coeff(taps: WienerTaps, k: usize) -> i32 {
    match k {
        0 | 6 => taps.0[0],
        1 | 5 => taps.0[1],
        2 | 4 => taps.0[2],
        3 => taps.0[3],
        _ => 0,
    }
}

/// Apply a 1-D 7-tap symmetric Wiener convolution across a row. Source
/// samples are read with edge-clamped indexing.
fn convolve_row(dst: &mut [u8], src: &[u8], w: usize, taps: WienerTaps) {
    for (i, slot) in dst.iter_mut().enumerate().take(w) {
        let mut acc: i32 = 0;
        for k in 0..7 {
            let j = (i as i32 + k as i32 - 3).clamp(0, (w as i32) - 1) as usize;
            acc += tap_coeff(taps, k) * src[j] as i32;
        }
        *slot = clip8((acc + 64) >> 7);
    }
}

fn convolve_row16(dst: &mut [u16], src: &[u16], w: usize, taps: WienerTaps, bit_depth: u32) {
    for (i, slot) in dst.iter_mut().enumerate().take(w) {
        let mut acc: i32 = 0;
        for k in 0..7 {
            let j = (i as i32 + k as i32 - 3).clamp(0, (w as i32) - 1) as usize;
            acc += tap_coeff(taps, k) * src[j] as i32;
        }
        *slot = clip_bd((acc + 64) >> 7, bit_depth);
    }
}

/// Run the 7×7 separable Wiener filter on a `w × h` plane. Horizontal
/// pass is applied first (per row), then the vertical pass (per
/// column). `horiz` and `vert` are independent tap sets.
pub fn apply_wiener(
    dst: &mut [u8],
    src: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    horiz: WienerTaps,
    vert: WienerTaps,
) {
    let mut tmp = vec![0u8; w * h];
    for r in 0..h {
        let src_row = &src[r * stride..r * stride + w];
        convolve_row(&mut tmp[r * w..r * w + w], src_row, w, horiz);
    }
    let mut col = vec![0u8; h];
    let mut out = vec![0u8; h];
    for c in 0..w {
        for (r, col_ref) in col.iter_mut().enumerate().take(h) {
            *col_ref = tmp[r * w + c];
        }
        convolve_row(&mut out, &col, h, vert);
        for r in 0..h {
            dst[r * stride + c] = out[r];
        }
    }
}

/// `uint16` counterpart of [`apply_wiener`]. Output is clipped to
/// `[0, (1 << bit_depth) - 1]`.
#[allow(clippy::too_many_arguments)] // Mirrors the 8-bit apply_wiener signature (dst+src+dims+taps+bit_depth).
pub fn apply_wiener16(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    stride: usize,
    horiz: WienerTaps,
    vert: WienerTaps,
    bit_depth: u32,
) {
    let mut tmp = vec![0u16; w * h];
    for r in 0..h {
        let src_row = &src[r * stride..r * stride + w];
        convolve_row16(&mut tmp[r * w..r * w + w], src_row, w, horiz, bit_depth);
    }
    let mut col = vec![0u16; h];
    let mut out = vec![0u16; h];
    for c in 0..w {
        for (r, col_ref) in col.iter_mut().enumerate().take(h) {
            *col_ref = tmp[r * w + c];
        }
        convolve_row16(&mut out, &col, h, vert, bit_depth);
        for r in 0..h {
            dst[r * stride + c] = out[r];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_preserves_sample(src_byte: u8) {
        let w = 8;
        let h = 8;
        let src: Vec<u8> = (0..w * h)
            .map(|i| ((i * 17) as u8).wrapping_mul(src_byte))
            .collect();
        let mut dst = vec![0u8; w * h];
        apply_wiener(
            &mut dst,
            &src,
            w,
            h,
            w,
            WienerTaps::IDENTITY,
            WienerTaps::IDENTITY,
        );
        assert_eq!(dst, src);
    }

    #[test]
    fn identity_preserves_input() {
        identity_preserves_sample(1);
    }

    #[test]
    fn identity_on_flat_is_flat() {
        let w = 8;
        let h = 8;
        let src = vec![123u8; w * h];
        let mut dst = vec![0u8; w * h];
        apply_wiener(
            &mut dst,
            &src,
            w,
            h,
            w,
            WienerTaps::IDENTITY,
            WienerTaps::IDENTITY,
        );
        for &v in &dst {
            assert_eq!(v, 123);
        }
    }

    #[test]
    fn low_pass_smooths_checkerboard() {
        let w = 8;
        let h = 8;
        let mut src = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                src[r * w + c] = if (r + c) % 2 == 0 { 40 } else { 160 };
            }
        }
        let taps = WienerTaps::new(-1, 2, 5, 116);
        let mut dst = vec![0u8; w * h];
        apply_wiener(&mut dst, &src, w, h, w, taps, taps);
        for r in 2..h - 2 {
            for c in 2..w - 2 {
                let v = dst[r * w + c];
                assert!((20..=180).contains(&v), "dst[{r},{c}]={v}");
            }
        }
    }

    #[test]
    fn wiener16_identity_preserves() {
        let w = 8;
        let h = 8;
        let src: Vec<u16> = (0..w * h).map(|i| (i * 10) as u16).collect();
        let mut dst = vec![0u16; w * h];
        apply_wiener16(
            &mut dst,
            &src,
            w,
            h,
            w,
            WienerTaps::IDENTITY,
            WienerTaps::IDENTITY,
            10,
        );
        assert_eq!(dst, src);
    }

    #[test]
    fn wiener16_clips_to_bit_depth() {
        let w = 16;
        let h = 16;
        let src = vec![4090u16; w * h];
        let mut dst = vec![0u16; w * h];
        // Over-unity taps (sum > 128) — each output should clamp at 4095.
        apply_wiener16(
            &mut dst,
            &src,
            w,
            h,
            w,
            WienerTaps::new(0, 32, 0, 128),
            WienerTaps::new(0, 32, 0, 128),
            12,
        );
        for &v in &dst {
            assert!(v <= 4095, "sample {v} exceeded 12-bit max");
        }
    }
}
