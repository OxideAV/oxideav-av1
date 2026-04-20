//! AV1 sub-pel interpolation filters — §7.11.3.4.
//!
//! A block is resampled by applying the horizontal 8-tap filter
//! followed by the vertical 8-tap filter. Motion vectors carry
//! eighth-pel precision; each axis phase is `(mv & 15)` (16 phases ×
//! 8 taps per filter set).
//!
//! Three filter sets are used by AVIF content: REGULAR / SMOOTH /
//! SHARP. The 4th libaom filter (REGULAR_PRESET_BILINEAR) is not
//! coded in AVIF bitstreams. Unknown values degrade to REGULAR.

/// Selects which 8-tap filter set to apply during interpolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpFilter {
    /// libaom's default `av1_sub_pel_filters_8`.
    Regular = 0,
    /// libaom's `av1_sub_pel_filters_8smooth`.
    Smooth = 1,
    /// libaom's `av1_sub_pel_filters_8sharp`.
    Sharp = 2,
}

impl InterpFilter {
    /// Map a raw symbol value to an `InterpFilter`. Unknown values
    /// collapse to `Regular` — we tolerate out-of-range inputs without
    /// erroring so corrupt bitstreams don't hard-fault the decoder.
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::Smooth,
            2 => Self::Sharp,
            _ => Self::Regular,
        }
    }
}

/// `EIGHT_TAP_REGULAR` — libaom `av1_sub_pel_filters_8` default filter.
pub static EIGHT_TAP_REGULAR: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, -6, 126, 8, -2, 0, 0],
    [0, 2, -10, 122, 18, -4, 0, 0],
    [0, 2, -12, 116, 28, -8, 2, 0],
    [0, 2, -14, 110, 38, -10, 2, 0],
    [0, 2, -14, 102, 48, -12, 2, 0],
    [0, 2, -16, 94, 58, -12, 2, 0],
    [0, 2, -14, 84, 66, -12, 2, 0],
    [0, 2, -14, 76, 76, -14, 2, 0],
    [0, 2, -12, 66, 84, -14, 2, 0],
    [0, 2, -12, 58, 94, -16, 2, 0],
    [0, 2, -12, 48, 102, -14, 2, 0],
    [0, 2, -10, 38, 110, -14, 2, 0],
    [0, 2, -8, 28, 116, -12, 2, 0],
    [0, 0, -4, 18, 122, -10, 2, 0],
    [0, 0, -2, 8, 126, -6, 2, 0],
];

/// `EIGHT_TAP_SMOOTH` — libaom `av1_sub_pel_filters_8smooth` filter.
pub static EIGHT_TAP_SMOOTH: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 2, 28, 62, 34, 2, 0, 0],
    [0, 0, 26, 62, 36, 4, 0, 0],
    [0, 0, 22, 62, 40, 4, 0, 0],
    [0, 0, 20, 60, 42, 6, 0, 0],
    [0, 0, 18, 58, 44, 8, 0, 0],
    [0, 0, 16, 56, 46, 10, 0, 0],
    [0, -2, 16, 54, 48, 12, 0, 0],
    [0, -2, 14, 52, 52, 14, -2, 0],
    [0, 0, 12, 48, 54, 16, -2, 0],
    [0, 0, 10, 46, 56, 16, 0, 0],
    [0, 0, 8, 44, 58, 18, 0, 0],
    [0, 0, 6, 42, 60, 20, 0, 0],
    [0, 0, 4, 40, 62, 22, 0, 0],
    [0, 0, 4, 36, 62, 26, 0, 0],
    [0, 0, 2, 34, 62, 28, 2, 0],
];

/// `EIGHT_TAP_SHARP` — libaom `av1_sub_pel_filters_8sharp` filter.
pub static EIGHT_TAP_SHARP: [[i16; 8]; 16] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-2, 2, -6, 126, 8, -2, 2, 0],
    [-2, 6, -12, 124, 16, -6, 4, -2],
    [-2, 8, -18, 120, 26, -10, 6, -2],
    [-4, 10, -22, 116, 38, -14, 6, -2],
    [-4, 10, -22, 108, 48, -18, 8, -2],
    [-4, 10, -24, 100, 60, -20, 8, -2],
    [-4, 10, -24, 90, 70, -22, 10, -2],
    [-4, 12, -24, 80, 80, -24, 12, -4],
    [-2, 10, -22, 70, 90, -24, 10, -4],
    [-2, 8, -20, 60, 100, -24, 10, -4],
    [-2, 8, -18, 48, 108, -22, 10, -4],
    [-2, 6, -14, 38, 116, -22, 10, -4],
    [-2, 6, -10, 26, 120, -18, 8, -2],
    [-2, 4, -6, 16, 124, -12, 6, -2],
    [0, 2, -2, 8, 126, -6, 2, -2],
];

/// Pick the filter coefficient table matching `filt`.
fn filt_table(filt: InterpFilter) -> &'static [[i16; 8]; 16] {
    match filt {
        InterpFilter::Smooth => &EIGHT_TAP_SMOOTH,
        InterpFilter::Sharp => &EIGHT_TAP_SHARP,
        InterpFilter::Regular => &EIGHT_TAP_REGULAR,
    }
}

/// Apply an 8-tap horizontal + vertical interpolation to produce a
/// `w×h` output block from a `(w+7)×(h+7)` reference area.
///
/// `src` is a flat slice indexed as `src[row*src_stride + col]`; `dst`
/// is a `w*h` row-major output slice. `hp` / `vp` are the 1/16-phase
/// indices — typically `(mv_col & 15)` and `(mv_row & 15)`.
#[allow(clippy::too_many_arguments)]
pub fn interp_sub_pel(
    dst: &mut [u8],
    w: usize,
    h: usize,
    src: &[u8],
    src_stride: usize,
    hp: usize,
    vp: usize,
    filt: InterpFilter,
) {
    let table = filt_table(filt);
    let h_filter = table[hp & 15];
    let v_filter = table[vp & 15];
    // Horizontal pass: accumulate into a `w × (h+7)` temp buffer of i32.
    let mut tmp = vec![0i32; w * (h + 7)];
    for r in 0..h + 7 {
        let src_row = r * src_stride;
        let tmp_row = r * w;
        for c in 0..w {
            let s = src_row + c;
            let mut sum = 0i32;
            for k in 0..8 {
                sum += (h_filter[k] as i32) * (src[s + k] as i32);
            }
            tmp[tmp_row + c] = sum;
        }
    }
    // Vertical pass: combine 8 tmp rows into 1 output pixel; round to
    // 14 fractional bits (1<<13 half-up) and clip to u8.
    for r in 0..h {
        let dst_row = r * w;
        for c in 0..w {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += (v_filter[k] as i32) * tmp[(r + k) * w + c];
            }
            let v = (sum + (1 << 13)) >> 14;
            let clipped = v.clamp(0, 255) as u8;
            dst[dst_row + c] = clipped;
        }
    }
}

/// HBD (10/12-bit) counterpart of [`interp_sub_pel`]. Same 8-tap
/// filter + same rounding; output clipped to `(1 << bit_depth) - 1`.
#[allow(clippy::too_many_arguments)]
pub fn interp_sub_pel16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    src: &[u16],
    src_stride: usize,
    hp: usize,
    vp: usize,
    filt: InterpFilter,
    bit_depth: u32,
) {
    let table = filt_table(filt);
    let h_filter = table[hp & 15];
    let v_filter = table[vp & 15];
    let max_v = ((1i32 << bit_depth) - 1).max(0);
    let mut tmp = vec![0i32; w * (h + 7)];
    for r in 0..h + 7 {
        for c in 0..w {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += (h_filter[k] as i32) * (src[r * src_stride + c + k] as i32);
            }
            tmp[r * w + c] = sum;
        }
    }
    for r in 0..h {
        for c in 0..w {
            let mut sum = 0i32;
            for k in 0..8 {
                sum += (v_filter[k] as i32) * tmp[(r + k) * w + c];
            }
            let v = (sum + (1 << 13)) >> 14;
            let clipped = v.clamp(0, max_v) as u16;
            dst[r * w + c] = clipped;
        }
    }
}

/// Integer-pel fast path: copies a `w×h` block from `src` at offset
/// 0, 0 (caller has already adjusted the source pointer).
pub fn interp_integer(dst: &mut [u8], w: usize, h: usize, src: &[u8], src_stride: usize) {
    for r in 0..h {
        dst[r * w..r * w + w].copy_from_slice(&src[r * src_stride..r * src_stride + w]);
    }
}

/// HBD counterpart of [`interp_integer`].
pub fn interp_integer16(dst: &mut [u16], w: usize, h: usize, src: &[u16], src_stride: usize) {
    for r in 0..h {
        dst[r * w..r * w + w].copy_from_slice(&src[r * src_stride..r * src_stride + w]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_phase_passes_input_through() {
        // 15×15 reference for an 8×8 output (7 samples padding each
        // axis). Phase-0 filter weights the single tap at index 3,
        // so dst[r][c] = src[r+3][c+3].
        const SRC_W: usize = 15;
        let mut src = vec![0u8; SRC_W * SRC_W];
        for r in 0..SRC_W {
            for c in 0..SRC_W {
                src[r * SRC_W + c] = ((c as u32) * 16) as u8;
            }
        }
        let mut dst = vec![0u8; 8 * 8];
        interp_sub_pel(&mut dst, 8, 8, &src, SRC_W, 0, 0, InterpFilter::Regular);
        for r in 0..8 {
            for c in 0..8 {
                let want = src[(r + 3) * SRC_W + (c + 3)];
                let got = dst[r * 8 + c];
                assert_eq!(
                    got,
                    want,
                    "dst[{r},{c}] = {got}, want src[{}][{}] = {want}",
                    r + 3,
                    c + 3
                );
            }
        }
    }

    #[test]
    fn half_pel_horizontal_sits_between_neighbours() {
        const SRC_W: usize = 15;
        let mut src = vec![0u8; SRC_W * SRC_W];
        for r in 0..SRC_W {
            for c in 0..SRC_W {
                src[r * SRC_W + c] = ((c as u32) * 16) as u8;
            }
        }
        let mut dst = vec![0u8; 4 * 4];
        // hp=8 (half-pel). Integer samples at r=0 are src[3][3]=48,
        // src[3][4]=64; filter should sit ~56.
        interp_sub_pel(&mut dst, 4, 4, &src, SRC_W, 8, 0, InterpFilter::Regular);
        let v = dst[0];
        assert!(
            (40..=72).contains(&v),
            "half-pel horizontal dst[0,0] = {v}, expected roughly 48..64"
        );
    }

    #[test]
    fn integer_copy_path_matches_source() {
        let mut src = vec![0u8; 10 * 10];
        for (i, v) in src.iter_mut().enumerate() {
            *v = (i & 0xFF) as u8;
        }
        let mut dst = vec![0u8; 4 * 4];
        interp_integer(&mut dst, 4, 4, &src, 10);
        for r in 0..4 {
            for c in 0..4 {
                assert_eq!(dst[r * 4 + c], src[r * 10 + c]);
            }
        }
    }

    #[test]
    fn hbd_zero_phase_passes_through() {
        const SRC_W: usize = 15;
        let mut src = vec![0u16; SRC_W * SRC_W];
        for r in 0..SRC_W {
            for c in 0..SRC_W {
                src[r * SRC_W + c] = (c as u16) * 64;
            }
        }
        let mut dst = vec![0u16; 8 * 8];
        interp_sub_pel16(&mut dst, 8, 8, &src, SRC_W, 0, 0, InterpFilter::Regular, 10);
        for r in 0..8 {
            for c in 0..8 {
                assert_eq!(dst[r * 8 + c], src[(r + 3) * SRC_W + (c + 3)]);
            }
        }
    }
}
