//! Frame-level CDEF driver (8-bit + HBD plane helpers).

use super::direction::{find_direction, find_direction16};
use super::filter::{
    adjust_pri_strength, filter_block, filter_block16, filter_block_spec, filter_block_spec16,
    CDEF_UV_DIR,
};

/// A single reconstructed u8 plane.
pub struct Plane<'a> {
    pub pix: &'a mut [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

/// A single reconstructed u16 plane.
pub struct Plane16<'a> {
    pub pix: &'a mut [u16],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

/// Run CDEF over an entire u8 plane with fixed strengths + damping.
pub fn apply_frame(p: Plane<'_>, pri_strength: i32, sec_strength: i32, damping: i32) {
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }
    let Plane {
        pix,
        stride,
        width,
        height,
    } = p;
    let buf: Vec<u8> = pix.to_vec();
    let mut y = 0usize;
    while y + 8 <= height {
        let mut x = 0usize;
        while x + 8 <= width {
            let (dir, _) = find_direction(&buf, stride, x, y);
            filter_block(
                pix,
                &buf,
                stride,
                x,
                y,
                dir,
                pri_strength,
                sec_strength,
                damping,
            );
            x += 8;
        }
        y += 8;
    }
}

/// Run CDEF over an entire u16 plane.
pub fn apply_frame16(
    p: Plane16<'_>,
    pri_strength: i32,
    sec_strength: i32,
    damping: i32,
    bit_depth: u32,
) {
    if pri_strength == 0 && sec_strength == 0 {
        return;
    }
    let Plane16 {
        pix,
        stride,
        width,
        height,
    } = p;
    let buf: Vec<u16> = pix.to_vec();
    let mut y = 0usize;
    while y + 8 <= height {
        let mut x = 0usize;
        while x + 8 <= width {
            let (dir, _) = find_direction16(&buf, stride, x, y, bit_depth);
            filter_block16(
                pix,
                &buf,
                stride,
                x,
                y,
                dir,
                pri_strength,
                sec_strength,
                damping,
                bit_depth,
            );
            x += 8;
        }
        y += 8;
    }
}

/// Run CDEF per 8×8 block with per-block strengths chosen via
/// `strength_fn(x, y) -> (pri, sec)`. Blocks where both are zero are
/// skipped.
pub fn apply_frame_per_sb<F>(p: Plane<'_>, strength_fn: F, damping: i32)
where
    F: Fn(usize, usize) -> (i32, i32),
{
    let Plane {
        pix,
        stride,
        width,
        height,
    } = p;
    let buf: Vec<u8> = pix.to_vec();
    let mut y = 0usize;
    while y + 8 <= height {
        let mut x = 0usize;
        while x + 8 <= width {
            let (pri, sec) = strength_fn(x, y);
            if pri == 0 && sec == 0 {
                x += 8;
                continue;
            }
            let (dir, _) = find_direction(&buf, stride, x, y);
            filter_block(pix, &buf, stride, x, y, dir, pri, sec, damping);
            x += 8;
        }
        y += 8;
    }
}

/// 16-bit counterpart of [`apply_frame_per_sb`].
pub fn apply_frame_per_sb16<F>(p: Plane16<'_>, strength_fn: F, damping: i32, bit_depth: u32)
where
    F: Fn(usize, usize) -> (i32, i32),
{
    let Plane16 {
        pix,
        stride,
        width,
        height,
    } = p;
    let buf: Vec<u16> = pix.to_vec();
    let mut y = 0usize;
    while y + 8 <= height {
        let mut x = 0usize;
        while x + 8 <= width {
            let (pri, sec) = strength_fn(x, y);
            if pri == 0 && sec == 0 {
                x += 8;
                continue;
            }
            let (dir, _) = find_direction16(&buf, stride, x, y, bit_depth);
            filter_block16(pix, &buf, stride, x, y, dir, pri, sec, damping, bit_depth);
            x += 8;
        }
        y += 8;
    }
}

/// Per-64×64-SB CDEF strengths picked from the frame-header table by
/// the per-SB `cdef_idx`. `pri_y / sec_y / pri_uv / sec_uv` are the
/// raw (unshifted) header values; the driver applies `<< coeff_shift`
/// before invoking [`super::filter::filter_block_spec`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SbStrengths {
    pub pri_y: i32,
    pub sec_y: i32,
    pub pri_uv: i32,
    pub sec_uv: i32,
    /// Base damping (`cdef_damping_minus3 + 3`) before the spec's
    /// `+ coeff_shift` / `- 1` (chroma) adjustment.
    pub damping: i32,
}

/// Spec-exact §7.15 CDEF driver for a single 8-bit plane. Walks 8×8
/// blocks across the plane, computes luma direction + variance for
/// plane 0 (or consumes a precomputed table for chroma), derives per
/// §7.15.1 the effective `priStr / secStr / dir / damping`, then
/// invokes [`super::filter::filter_block_spec`].
///
/// `sb_size` is 64 or 128 luma samples; plane coordinates are assumed
/// already subsampled. `sub_x / sub_y` describe the plane's
/// subsampling relative to luma. `plane == 0` runs the luma direction
/// search; chroma planes read from `dir_var` (filled by the luma pass)
/// via `dir_var[sb_row * sb_cols_luma + sb_col][block_row][block_col]`.
/// Pass a fresh `Vec<Vec<...>>` and let the driver populate it on the
/// luma pass — downstream chroma passes reuse that storage.
///
/// `sb_strengths(sb_col, sb_row) -> Option<SbStrengths>` returns the
/// SB's strengths or `None` to skip filtering (spec `cdef_idx == -1`).
#[allow(clippy::too_many_arguments)]
pub fn apply_frame_spec<F>(
    p: Plane<'_>,
    plane: usize,
    sub_x: u32,
    sub_y: u32,
    sb_size_luma: usize,
    dir_table: &mut Vec<(usize, i32)>,
    sb_cols_luma: usize,
    _sb_rows_luma: usize,
    coeff_shift: u32,
    sb_strengths: F,
) where
    F: Fn(usize, usize) -> Option<SbStrengths>,
{
    let Plane {
        pix,
        stride,
        width,
        height,
    } = p;
    let sb_w_plane = sb_size_luma >> sub_x;
    let sb_h_plane = sb_size_luma >> sub_y;
    // Number of 8x8 blocks per SB in this plane.
    let blocks_per_sb_x = sb_w_plane / 8;
    let blocks_per_sb_y = sb_h_plane / 8;
    let blocks_per_sb = blocks_per_sb_x * blocks_per_sb_y;

    if plane == 0 {
        // Allocate direction/variance storage for the whole luma
        // frame, one entry per 8x8 block. Chroma passes use the same
        // global indexing.
        let luma_sbs_x = width.div_ceil(sb_size_luma);
        let luma_sbs_y = height.div_ceil(sb_size_luma);
        dir_table.clear();
        dir_table.resize(luma_sbs_x * luma_sbs_y * blocks_per_sb, (0, 0));
    }

    let buf: Vec<u8> = pix.to_vec();
    let mut y = 0usize;
    while y + 8 <= height {
        let mut x = 0usize;
        while x + 8 <= width {
            let sb_col = (x << sub_x) / sb_size_luma;
            let sb_row = (y << sub_y) / sb_size_luma;
            let Some(st) = sb_strengths(sb_col, sb_row) else {
                x += 8;
                continue;
            };
            let (pri_raw, sec_raw) = if plane == 0 {
                (st.pri_y, st.sec_y)
            } else {
                (st.pri_uv, st.sec_uv)
            };
            // Global block index: SB index * blocks_per_sb + inner offset.
            let inner_col = (x / 8) % blocks_per_sb_x;
            let inner_row = (y / 8) % blocks_per_sb_y;
            let luma_sbs_x = width.div_ceil(sb_size_luma >> sub_x);
            let gsb = sb_row * luma_sbs_x + sb_col;
            let gblock = gsb * blocks_per_sb + inner_row * blocks_per_sb_x + inner_col;

            let (y_dir, var) = if plane == 0 {
                let dv = find_direction(&buf, stride, x, y);
                if let Some(slot) = dir_table.get_mut(gblock) {
                    *slot = dv;
                }
                dv
            } else {
                dir_table.get(gblock).copied().unwrap_or((0, 0))
            };

            let pri_base = pri_raw << coeff_shift;
            let sec_str = sec_raw << coeff_shift;
            let (pri_str, dir) = if plane == 0 {
                let adjusted = if pri_base == 0 {
                    0
                } else {
                    adjust_pri_strength(pri_base, var)
                };
                let d = if adjusted == 0 { 0 } else { y_dir };
                (adjusted, d)
            } else {
                // Chroma uses the raw base strength (spec §7.15.1 does
                // not apply the variance scaling to chroma; chroma dir
                // is remapped via Cdef_Uv_Dir).
                let d = if pri_base == 0 {
                    0
                } else {
                    CDEF_UV_DIR[sub_x.min(1) as usize][sub_y.min(1) as usize][y_dir]
                };
                (pri_base, d)
            };

            if pri_str != 0 || sec_str != 0 {
                let damping = if plane == 0 {
                    st.damping + coeff_shift as i32
                } else {
                    st.damping + coeff_shift as i32 - 1
                };
                let w_local = width;
                let h_local = height;
                filter_block_spec(
                    pix,
                    &buf,
                    stride,
                    x,
                    y,
                    dir,
                    pri_str,
                    sec_str,
                    damping,
                    coeff_shift,
                    |r, c| r >= 0 && c >= 0 && (r as usize) < h_local && (c as usize) < w_local,
                );
            }
            x += 8;
        }
        y += 8;
    }
    let _ = sb_cols_luma;
}

/// `uint16` counterpart of [`apply_frame_spec`].
#[allow(clippy::too_many_arguments)]
pub fn apply_frame_spec16<F>(
    p: Plane16<'_>,
    plane: usize,
    sub_x: u32,
    sub_y: u32,
    sb_size_luma: usize,
    dir_table: &mut Vec<(usize, i32)>,
    sb_cols_luma: usize,
    _sb_rows_luma: usize,
    bit_depth: u32,
    sb_strengths: F,
) where
    F: Fn(usize, usize) -> Option<SbStrengths>,
{
    let coeff_shift = bit_depth - 8;
    let Plane16 {
        pix,
        stride,
        width,
        height,
    } = p;
    let sb_w_plane = sb_size_luma >> sub_x;
    let sb_h_plane = sb_size_luma >> sub_y;
    let blocks_per_sb_x = sb_w_plane / 8;
    let blocks_per_sb_y = sb_h_plane / 8;
    let blocks_per_sb = blocks_per_sb_x * blocks_per_sb_y;

    if plane == 0 {
        let luma_sbs_x = width.div_ceil(sb_size_luma);
        let luma_sbs_y = height.div_ceil(sb_size_luma);
        dir_table.clear();
        dir_table.resize(luma_sbs_x * luma_sbs_y * blocks_per_sb, (0, 0));
    }

    let buf: Vec<u16> = pix.to_vec();
    let mut y = 0usize;
    while y + 8 <= height {
        let mut x = 0usize;
        while x + 8 <= width {
            let sb_col = (x << sub_x) / sb_size_luma;
            let sb_row = (y << sub_y) / sb_size_luma;
            let Some(st) = sb_strengths(sb_col, sb_row) else {
                x += 8;
                continue;
            };
            let (pri_raw, sec_raw) = if plane == 0 {
                (st.pri_y, st.sec_y)
            } else {
                (st.pri_uv, st.sec_uv)
            };
            let inner_col = (x / 8) % blocks_per_sb_x;
            let inner_row = (y / 8) % blocks_per_sb_y;
            let luma_sbs_x = width.div_ceil(sb_size_luma >> sub_x);
            let gsb = sb_row * luma_sbs_x + sb_col;
            let gblock = gsb * blocks_per_sb + inner_row * blocks_per_sb_x + inner_col;

            let (y_dir, var) = if plane == 0 {
                let dv = find_direction16(&buf, stride, x, y, bit_depth);
                if let Some(slot) = dir_table.get_mut(gblock) {
                    *slot = dv;
                }
                dv
            } else {
                dir_table.get(gblock).copied().unwrap_or((0, 0))
            };

            let pri_base = pri_raw << coeff_shift;
            let sec_str = sec_raw << coeff_shift;
            let (pri_str, dir) = if plane == 0 {
                let adjusted = if pri_base == 0 {
                    0
                } else {
                    adjust_pri_strength(pri_base, var)
                };
                let d = if adjusted == 0 { 0 } else { y_dir };
                (adjusted, d)
            } else {
                let d = if pri_base == 0 {
                    0
                } else {
                    CDEF_UV_DIR[sub_x.min(1) as usize][sub_y.min(1) as usize][y_dir]
                };
                (pri_base, d)
            };

            if pri_str != 0 || sec_str != 0 {
                let damping = if plane == 0 {
                    st.damping + coeff_shift as i32
                } else {
                    st.damping + coeff_shift as i32 - 1
                };
                let w_local = width;
                let h_local = height;
                filter_block_spec16(
                    pix,
                    &buf,
                    stride,
                    x,
                    y,
                    dir,
                    pri_str,
                    sec_str,
                    damping,
                    bit_depth,
                    coeff_shift,
                    |r, c| r >= 0 && c >= 0 && (r as usize) < h_local && (c as usize) < w_local,
                );
            }
            x += 8;
        }
        y += 8;
    }
    let _ = sb_cols_luma;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_frame_zero_strength_is_noop() {
        let mut pix = vec![100u8; 16 * 16];
        let p = Plane {
            pix: &mut pix,
            stride: 16,
            width: 16,
            height: 16,
        };
        apply_frame(p, 0, 0, 3);
        for v in &pix {
            assert_eq!(*v, 100);
        }
    }

    #[test]
    fn apply_frame_flat_interior_preserved() {
        // Flat input yields flat interior samples; edge samples may
        // shift by a few levels due to the CDEF_VERY_LARGE sentinel
        // being substituted for out-of-range reads.
        let mut pix = vec![100u8; 16 * 16];
        let p = Plane {
            pix: &mut pix,
            stride: 16,
            width: 16,
            height: 16,
        };
        apply_frame(p, 20, 10, 6);
        for r in 0..13 {
            for v in pix.iter().skip(r * 16).take(16) {
                assert_eq!(*v, 100, "row {r} should remain flat");
            }
        }
    }

    #[test]
    fn apply_frame_spec_skips_none_strength_sbs() {
        // 64×64 luma frame, one SB, skip cdef_idx = None -> identity.
        let w = 64;
        let h = 64;
        let mut pix: Vec<u8> = (0..w * h).map(|i| (i & 0xFF) as u8).collect();
        let orig = pix.clone();
        let p = Plane {
            pix: &mut pix,
            stride: w,
            width: w,
            height: h,
        };
        let mut dir = Vec::new();
        apply_frame_spec(p, 0, 0, 0, 64, &mut dir, 1, 1, 0, |_, _| None);
        assert_eq!(pix, orig);
    }

    #[test]
    fn apply_frame_spec_flat_interior_unchanged() {
        // Flat 64x64 block -> luma direction search returns var=0 ->
        // priStr collapses to 0 -> no filter applied.
        let w = 64;
        let h = 64;
        let mut pix = vec![100u8; w * h];
        let p = Plane {
            pix: &mut pix,
            stride: w,
            width: w,
            height: h,
        };
        let mut dir = Vec::new();
        apply_frame_spec(p, 0, 0, 0, 64, &mut dir, 1, 1, 0, |_, _| {
            Some(SbStrengths {
                pri_y: 6,
                sec_y: 3,
                pri_uv: 0,
                sec_uv: 0,
                damping: 3,
            })
        });
        for &v in &pix {
            assert_eq!(v, 100);
        }
    }

    #[test]
    fn apply_frame_spec_populates_dir_table_on_luma() {
        let w = 64;
        let h = 64;
        // Inject a horizontal line across the first 8x8 so direction
        // search yields a non-zero var for at least one block.
        let mut pix = vec![100u8; w * h];
        for c in 0..8 {
            pix[3 * w + c] = 200;
        }
        let p = Plane {
            pix: &mut pix,
            stride: w,
            width: w,
            height: h,
        };
        let mut dir_table = Vec::new();
        apply_frame_spec(p, 0, 0, 0, 64, &mut dir_table, 1, 1, 0, |_, _| {
            Some(SbStrengths {
                pri_y: 6,
                sec_y: 3,
                pri_uv: 0,
                sec_uv: 0,
                damping: 3,
            })
        });
        // One SB of 64 blocks of 8x8.
        assert_eq!(dir_table.len(), 64);
        // First block should be direction horizontal (2) with nonzero var.
        assert!(
            dir_table[0].1 > 0,
            "expected nonzero var, got {dir_table:?}"
        );
        assert_eq!(dir_table[0].0, 2);
    }
}
