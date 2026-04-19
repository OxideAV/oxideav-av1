//! Self-guided restoration (SGR) — spec §7.17.4.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/lr/{sgr,sgr16}.go`
//! (MIT, KarpelesLab/goavif).
//!
//! SGR runs a variance-adaptive box filter whose output blends toward
//! the local mean in flat regions and toward the input in high-variance
//! regions. The single-pass sub-filter is:
//!
//! ```text
//! n = (2r+1)²                       // window area
//! A = sum over window of pixel²
//! B = sum over window of pixel
//! p = max(0, n·A − B²)
//! z = clamp((p·eps + (1<<19)) >> 20, 0, 255)
//! a = SGR_X_BY_XPLUS1[z]            // 256·z / (z+1)
//! mean = B / n
//! output = (a·pixel + (256−a)·mean + 128) >> 8
//! ```
//!
//! The full AV1 SGR is dual-pass; [`apply_sgr`] implements the dual
//! form blended by per-unit `xq[0,1]`. [`sgr_sub_filter`] exposes a
//! single pass for tests / diagnostics.

/// `x/(x+1)·256` LUT (spec §7.17.4 `av1_x_by_xplus1`). Entry 255
/// saturates at 255.
pub const SGR_X_BY_XPLUS1: [u8; 256] = [
    1, 128, 171, 192, 205, 213, 219, 224, 228, 230, 233, 235, 236, 238, 239, 240, //
    241, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247, 247, 247, 247, 248, 248, //
    248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 250, 251, 251, //
    251, 251, 251, 251, 251, 251, 251, 251, 252, 252, 252, 252, 252, 252, 252, 252, //
    252, 252, 252, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 253, 253, 253, //
    253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, 253, //
    253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, //
    254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 254, 255, 255, //
];

/// Per-unit SGR coefficients. `r0 / r1` are window radii (0 disables
/// the sub-filter). `eps0 / eps1` are edge-preservation thresholds.
/// `xq[0,1]` are signed Q6 blending weights signaled per restoration
/// unit (spec §5.11.44).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SgrParams {
    pub r0: i32,
    pub r1: i32,
    pub eps0: i32,
    pub eps1: i32,
    pub xq: [i32; 2],
}

#[inline]
fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Compute the mean of samples in a `(2r+1)²` window around each
/// pixel. Out-of-range samples are replicated via edge clamp. Result
/// is `w·h` `u16` values row-major.
pub fn box_mean(src: &[u8], w: usize, h: usize, stride: usize, r: i32) -> Vec<u16> {
    let mut out = vec![0u16; w * h];
    let area = (2 * r + 1) * (2 * r + 1);
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            for dy in -r..=r {
                let yy = clamp_i(y as i32 + dy, 0, h as i32 - 1) as usize;
                for dx in -r..=r {
                    let xx = clamp_i(x as i32 + dx, 0, w as i32 - 1) as usize;
                    sum += src[yy * stride + xx] as i32;
                }
            }
            out[y * w + x] = (sum / area) as u16;
        }
    }
    out
}

/// Compute the sample variance (σ²) in the same `(2r+1)²` window.
pub fn box_var(src: &[u8], w: usize, h: usize, stride: usize, r: i32) -> Vec<u32> {
    let mut out = vec![0u32; w * h];
    let area = (2 * r + 1) * (2 * r + 1);
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            let mut sum_sq = 0i32;
            for dy in -r..=r {
                let yy = clamp_i(y as i32 + dy, 0, h as i32 - 1) as usize;
                for dx in -r..=r {
                    let xx = clamp_i(x as i32 + dx, 0, w as i32 - 1) as usize;
                    let s = src[yy * stride + xx] as i32;
                    sum += s;
                    sum_sq += s * s;
                }
            }
            let mean = sum / area;
            let var = sum_sq / area - mean * mean;
            out[y * w + x] = var.max(0) as u32;
        }
    }
    out
}

/// Run one SGR pass with radius `r` and `eps`. `r == 0` copies input
/// to output.
pub fn sgr_sub_filter(
    dst: &mut [u8],
    src: &[u8],
    w: usize,
    h: usize,
    stride: usize,
    r: i32,
    eps: i32,
) {
    if r <= 0 {
        for y in 0..h {
            if dst.as_ptr() != src.as_ptr() {
                let lo = y * stride;
                dst[lo..lo + w].copy_from_slice(&src[lo..lo + w]);
            }
        }
        return;
    }
    let n = (2 * r + 1) * (2 * r + 1);
    let mut tmp = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0i32;
            let mut sum_sq = 0i32;
            for dy in -r..=r {
                let yy = clamp_i(y as i32 + dy, 0, h as i32 - 1) as usize;
                for dx in -r..=r {
                    let xx = clamp_i(x as i32 + dx, 0, w as i32 - 1) as usize;
                    let s = src[yy * stride + xx] as i32;
                    sum += s;
                    sum_sq += s * s;
                }
            }
            let p = (n * sum_sq - sum * sum).max(0);
            let z = (((p * eps) + (1 << 19)) >> 20).clamp(0, 255);
            let a = SGR_X_BY_XPLUS1[z as usize] as i32;
            let mean = sum / n;
            let pix = src[y * stride + x] as i32;
            let out = (a * pix + (256 - a) * mean + 128) >> 8;
            tmp[y * w + x] = out.clamp(0, 255) as u8;
        }
    }
    for y in 0..h {
        dst[y * stride..y * stride + w].copy_from_slice(&tmp[y * w..y * w + w]);
    }
}

/// `uint16` counterpart of [`sgr_sub_filter`]. `eps` should already be
/// scaled for `bit_depth` by the caller.
#[allow(clippy::too_many_arguments)] // Mirrors the goavif SGRSubFilter16 signature.
pub fn sgr_sub_filter16(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    stride: usize,
    r: i32,
    eps: i32,
    bit_depth: u32,
) {
    if r <= 0 {
        if dst.as_ptr() != src.as_ptr() {
            for y in 0..h {
                let lo = y * stride;
                dst[lo..lo + w].copy_from_slice(&src[lo..lo + w]);
            }
        }
        return;
    }
    let n = (2 * r + 1) * (2 * r + 1);
    let n64 = n as i64;
    let shift = 2 * (bit_depth as i64 - 8);
    let max_v: i32 = (1i32 << bit_depth) - 1;
    let mut tmp = vec![0u16; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut sum: i64 = 0;
            let mut sum_sq: i64 = 0;
            for dy in -r..=r {
                let yy = clamp_i(y as i32 + dy, 0, h as i32 - 1) as usize;
                for dx in -r..=r {
                    let xx = clamp_i(x as i32 + dx, 0, w as i32 - 1) as usize;
                    let s = src[yy * stride + xx] as i64;
                    sum += s;
                    sum_sq += s * s;
                }
            }
            let mut p = (n64 * sum_sq - sum * sum).max(0);
            if shift > 0 {
                p >>= shift as u32;
            }
            let z = ((p * eps as i64 + (1 << 19)) >> 20).clamp(0, 255) as usize;
            let a = SGR_X_BY_XPLUS1[z] as i64;
            let mean = sum / n64;
            let pix = src[y * stride + x] as i64;
            let out = (a * pix + (256 - a) * mean + 128) >> 8;
            tmp[y * w + x] = (out as i32).clamp(0, max_v) as u16;
        }
    }
    for y in 0..h {
        dst[y * stride..y * stride + w].copy_from_slice(&tmp[y * w..y * w + w]);
    }
}

/// Run the dual-pass SGR filter and blend the two outputs with the
/// input per spec §7.17.4. When both radii are zero the function
/// copies `src` to `dst`.
pub fn apply_sgr(dst: &mut [u8], src: &[u8], w: usize, h: usize, stride: usize, p: SgrParams) {
    if p.r0 == 0 && p.r1 == 0 {
        if dst.as_ptr() != src.as_ptr() {
            for y in 0..h {
                dst[y * stride..y * stride + w].copy_from_slice(&src[y * stride..y * stride + w]);
            }
        }
        return;
    }
    let mut flt0 = vec![0u8; w * h];
    let mut flt1 = vec![0u8; w * h];
    sgr_sub_filter(&mut flt0, src, w, h, w, p.r0, p.eps0);
    sgr_sub_filter(&mut flt1, src, w, h, w, p.r1, p.eps1);
    for y in 0..h {
        for x in 0..w {
            let pix = src[y * stride + x] as i32;
            let d0 = flt0[y * w + x] as i32 - pix;
            let d1 = flt1[y * w + x] as i32 - pix;
            let v = pix + ((p.xq[0] * d0 + p.xq[1] * d1 + 32) >> 6);
            dst[y * stride + x] = v.clamp(0, 255) as u8;
        }
    }
}

/// `uint16` counterpart of [`apply_sgr`].
pub fn apply_sgr16(
    dst: &mut [u16],
    src: &[u16],
    w: usize,
    h: usize,
    stride: usize,
    p: SgrParams,
    bit_depth: u32,
) {
    if p.r0 == 0 && p.r1 == 0 {
        if dst.as_ptr() != src.as_ptr() {
            for y in 0..h {
                dst[y * stride..y * stride + w].copy_from_slice(&src[y * stride..y * stride + w]);
            }
        }
        return;
    }
    let mut flt0 = vec![0u16; w * h];
    let mut flt1 = vec![0u16; w * h];
    sgr_sub_filter16(&mut flt0, src, w, h, w, p.r0, p.eps0, bit_depth);
    sgr_sub_filter16(&mut flt1, src, w, h, w, p.r1, p.eps1, bit_depth);
    let max_v = (1i32 << bit_depth) - 1;
    for y in 0..h {
        for x in 0..w {
            let pix = src[y * stride + x] as i32;
            let d0 = flt0[y * w + x] as i32 - pix;
            let d1 = flt1[y * w + x] as i32 - pix;
            let v = pix + ((p.xq[0] * d0 + p.xq[1] * d1 + 32) >> 6);
            dst[y * stride + x] = v.clamp(0, max_v) as u16;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_mean_constant_input() {
        let w = 8;
        let h = 8;
        let src = vec![100u8; w * h];
        let got = box_mean(&src, w, h, w, 2);
        for &v in &got {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn box_var_constant_input() {
        let w = 8;
        let h = 8;
        let src = vec![73u8; w * h];
        let got = box_var(&src, w, h, w, 2);
        for &v in &got {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn box_mean_single_spike() {
        let w = 10;
        let h = 10;
        let mut src = vec![0u8; w * h];
        src[5 * w + 5] = 255;
        let got = box_mean(&src, w, h, w, 2);
        assert_eq!(got[5 * w + 5], 10);
    }

    #[test]
    fn apply_sgr_passthrough_on_zero_radii() {
        let w = 8;
        let h = 8;
        let src: Vec<u8> = (0..w * h).map(|i| ((i * 19) & 0xFF) as u8).collect();
        let mut dst = vec![0u8; w * h];
        apply_sgr(&mut dst, &src, w, h, w, SgrParams::default());
        assert_eq!(dst, src);
    }

    #[test]
    fn sgr_sub_filter_flat_input() {
        let w = 8;
        let h = 8;
        let src = vec![77u8; w * h];
        let mut dst = vec![0u8; w * h];
        sgr_sub_filter(&mut dst, &src, w, h, w, 2, 12);
        for &v in &dst {
            assert_eq!(v, 77);
        }
    }

    #[test]
    fn apply_sgr_softens_checkerboard() {
        let w = 8;
        let h = 8;
        let mut src = vec![0u8; w * h];
        for r in 0..h {
            for c in 0..w {
                src[r * w + c] = if (r + c) % 2 == 0 { 40 } else { 160 };
            }
        }
        let mut dst = vec![0u8; w * h];
        let p = SgrParams {
            r0: 1,
            r1: 0,
            eps0: 40,
            eps1: 0,
            xq: [32, 0],
        };
        apply_sgr(&mut dst, &src, w, h, w, p);
        for r in 2..h - 2 {
            for c in 2..w - 2 {
                let v = dst[r * w + c];
                assert!((30..=170).contains(&v), "dst[{r},{c}]={v}");
            }
        }
    }

    #[test]
    fn sgr16_passthrough_zero_radii() {
        let w = 8;
        let h = 8;
        let src: Vec<u16> = (0..w * h).map(|i| (i * 5) as u16).collect();
        let mut dst = vec![0u16; w * h];
        apply_sgr16(&mut dst, &src, w, h, w, SgrParams::default(), 10);
        assert_eq!(dst, src);
    }

    #[test]
    fn sgr16_clips_to_bit_depth() {
        let w = 12;
        let h = 12;
        let mut src = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                src[y * w + x] = 1020 + ((x + y) & 3) as u16;
            }
        }
        let mut dst = vec![0u16; w * h];
        let p = SgrParams {
            r0: 1,
            r1: 0,
            eps0: 12,
            eps1: 0,
            xq: [8, 0],
        };
        apply_sgr16(&mut dst, &src, w, h, w, p, 10);
        for &v in &dst {
            assert!(v <= 1023, "sample {v} exceeds 10-bit max");
        }
    }
}
