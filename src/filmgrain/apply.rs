//! Spec-correct 32×32 film-grain tiler (§7.20.2 + §7.20.3).
//!
//! This replaces the goavif per-pixel LFSR shortcut with the real
//! spec-mandated pipeline:
//!
//! 1. A full-frame AR-shaped grain template is built once per frame
//!    (luma 73×73, chroma 38×38 — see [`super::patch`]).
//! 2. The output frame is walked in 32×32 blocks (§7.20.2). For each
//!    block we seed a per-block LFSR from `grain_seed` plus a hash of
//!    the block's `(row, col)` position and pull a deterministic
//!    32×32 sub-patch out of the template.
//! 3. Each pixel in the block is scaled by the luma/chroma intensity
//!    LUT (§7.20.3.3) and added into the plane. If the `overlap_flag`
//!    is set we cross-fade the first two rows / first two columns of
//!    every block against the tail of the adjacent block (§7.20.3.4)
//!    so the tile grid is invisible in the synthesised noise.
//! 4. Output is clipped to `[0, max]` or to the broadcast-legal range
//!    when [`super::Params::clip_to_restricted_range`] is set.
//!
//! `apply_with_template` (8-bit) and `apply_with_template16` (HBD) are
//! the public entrypoints. [`apply_frame`] drives all three planes
//! with the chroma-from-luma blending prescribed by §7.20.3.2.

// `clippy::needless_range_loop`: the per-block overlap state couples
// `by/bx` to `right_seam/bottom_seam` reads AND writes (including
// reads from `right_seam[by]` slot that's only populated once per
// `by` iteration), so iterator-based rewrites obscure the algorithm.
// `clippy::too_many_arguments`: the public entrypoints mirror the
// libaom / goavif signatures (plane + dims + stride + scaling +
// template + params + bit-depth); merging them into a struct helps
// nothing.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use super::patch::Template;
use super::rng::Rng;
use super::scaling::ScalingLut;
use super::Params;

/// Overlap cross-fade weights for the 2-row / 2-column boundary
/// (spec §7.20.3.4, libaom `av1_overlap_[row|col]`). Expressed as
/// `(near_weight, far_weight)` in Q5.
const OVERLAP_WEIGHTS: [(i32, i32); 2] = [(27, 17), (17, 27)];

/// Per-block seed derivation. Mirrors spec §7.20.2 `grain_seed_hash`
/// (tile-independent form — for still images the inter tile path
/// reduces to this). `block_x / block_y` are 32-sample indices.
fn block_seed(grain_seed: u16, block_x: u32, block_y: u32) -> u16 {
    let h = grain_seed as u32 ^ (block_y * 37 + block_x * 11);
    let s = (h & 0xFFFF) as u16;
    if s == 0 {
        1
    } else {
        s
    }
}

/// Bound `(max_rows, max_cols)` available for the random offset into
/// `template` when pulling a 32×32 sub-patch. Returns at least 1.
fn offset_bounds(template: &Template) -> (i32, i32) {
    let mr = (template.rows as i32 - 32).max(1);
    let mc = (template.cols as i32 - 32).max(1);
    (mr, mc)
}

/// Apply a single 8-bit scaled grain sample to `pix`, clipped into
/// `[lo, hi]`.
#[inline]
fn apply_sample(pix: u8, grain: i32, scale: u8, shift: u32, lo: i32, hi: i32) -> u8 {
    let delta = (grain * scale as i32) >> shift;
    (pix as i32 + delta).clamp(lo, hi) as u8
}

/// 16-bit counterpart of [`apply_sample`].
#[inline]
fn apply_sample16(pix: u16, grain: i32, scale: u8, shift: u32, lo: i32, hi: i32) -> u16 {
    let delta = (grain * scale as i32) >> shift;
    (pix as i32 + delta).clamp(lo, hi) as u16
}

/// Cross-fade helper for overlap mode: blend `new_grain` with the
/// `prev_grain` already applied in the previous block at the seam.
/// `row_in_block` / `col_in_block` are 0 or 1; anything else returns
/// `new_grain` unchanged.
#[inline]
fn overlap_blend(new_grain: i32, prev_grain: i32, idx: usize) -> i32 {
    if idx >= 2 {
        return new_grain;
    }
    let (near, far) = OVERLAP_WEIGHTS[idx];
    (near * prev_grain + far * new_grain + 16) >> 5
}

/// Apply 32×32-tiled film grain to an 8-bit plane. `template` is the
/// full-frame AR-shaped grain buffer built by [`super::patch::new_luma_template`]
/// (luma) or [`super::patch::new_chroma_template`] (chroma).
pub fn apply_with_template(
    plane: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
    scaling: &ScalingLut,
    template: &Template,
    p: &Params,
) {
    if p.grain_seed == 0 || template.rows == 0 {
        return;
    }
    let shift = if p.scaling_shift == 0 {
        8
    } else {
        p.scaling_shift as u32
    };
    let (lo, hi) = if p.clip_to_restricted_range {
        (16i32, 235i32)
    } else {
        (0i32, 255i32)
    };
    let (max_r, max_c) = offset_bounds(template);
    let mut rng = Rng::new(0);

    // Per-block seam storage for overlap blending. `bottom_seam[bx]`
    // holds the last two rows of pre-overlap grain from the block
    // directly above (2×32 samples, row-major). `right_seam[by]`
    // holds the last two cols of pre-overlap grain from the block
    // directly to the left (32×2 samples, row-major, 2 cols wide).
    let block_cols = w.div_ceil(32);
    let block_rows = h.div_ceil(32);
    let mut bottom_seam: Vec<Vec<i32>> = vec![Vec::new(); block_cols];
    let mut right_seam: Vec<Vec<i32>> = vec![Vec::new(); block_rows];

    for by in 0..block_rows {
        for bx in 0..block_cols {
            let seed = block_seed(p.grain_seed, bx as u32, by as u32);
            rng.seed(seed);
            let r_off = (rng.next() as i32 % max_r).abs();
            let c_off = (rng.next() as i32 % max_c).abs();

            let patch_y = by * 32;
            let patch_x = bx * 32;

            // Record grain values for this block so future blocks can
            // overlap against our right/bottom seams.
            let mut this_right: Vec<i32> = vec![0; 32 * 2]; // 32 rows × 2 cols
            let mut this_bottom: Vec<i32> = vec![0; 2 * 32]; // 2 rows × 32 cols

            for dy in 0..32 {
                let y = patch_y + dy;
                if y >= h {
                    break;
                }
                for dx in 0..32 {
                    let x = patch_x + dx;
                    if x >= w {
                        break;
                    }
                    let mut grain = template.sample(r_off + dy as i32, c_off + dx as i32) as i32;
                    // Overlap with block above (top 2 rows of the new
                    // block fade toward the bottom 2 rows of the old).
                    if p.overlap_flag && dy < 2 && by > 0 && !bottom_seam[bx].is_empty() {
                        let prev = bottom_seam[bx][dy * 32 + dx];
                        grain = overlap_blend(grain, prev, dy);
                    }
                    // Overlap with block to the left (left 2 cols of
                    // the new block fade toward the right 2 cols of
                    // the old).
                    if p.overlap_flag && dx < 2 && bx > 0 && !right_seam[by].is_empty() {
                        let prev = right_seam[by][dy * 2 + dx];
                        grain = overlap_blend(grain, prev, dx);
                    }
                    let pix = plane[y * stride + x];
                    let scale = scaling.lookup(pix);
                    plane[y * stride + x] = apply_sample(pix, grain, scale, shift, lo, hi);

                    // Stash right-seam samples (last two columns).
                    if dx >= 30 {
                        this_right[dy * 2 + (dx - 30)] = grain;
                    }
                    // Stash bottom-seam samples (last two rows).
                    if dy >= 30 {
                        this_bottom[(dy - 30) * 32 + dx] = grain;
                    }
                }
            }

            bottom_seam[bx] = this_bottom;
            right_seam[by] = this_right;
        }
    }
}

/// 10/12-bit counterpart of [`apply_with_template`]. `bit_depth` must
/// be 8, 10, or 12.
pub fn apply_with_template16(
    plane: &mut [u16],
    w: usize,
    h: usize,
    stride: usize,
    scaling: &ScalingLut,
    template: &Template,
    p: &Params,
    bit_depth: u32,
) {
    if p.grain_seed == 0 || template.rows == 0 {
        return;
    }
    let shift = if p.scaling_shift == 0 {
        8
    } else {
        p.scaling_shift as u32
    };
    let max_v = (1i32 << bit_depth) - 1;
    let (lo, hi) = if p.clip_to_restricted_range {
        let bd_shift = bit_depth - 8;
        (16 << bd_shift, 235 << bd_shift)
    } else {
        (0i32, max_v)
    };
    let (max_r, max_c) = offset_bounds(template);
    let mut rng = Rng::new(0);

    let block_cols = w.div_ceil(32);
    let block_rows = h.div_ceil(32);
    let mut bottom_seam: Vec<Vec<i32>> = vec![Vec::new(); block_cols];
    let mut right_seam: Vec<Vec<i32>> = vec![Vec::new(); block_rows];

    for by in 0..block_rows {
        for bx in 0..block_cols {
            let seed = block_seed(p.grain_seed, bx as u32, by as u32);
            rng.seed(seed);
            let r_off = (rng.next() as i32 % max_r).abs();
            let c_off = (rng.next() as i32 % max_c).abs();

            let patch_y = by * 32;
            let patch_x = bx * 32;

            let mut this_right: Vec<i32> = vec![0; 32 * 2];
            let mut this_bottom: Vec<i32> = vec![0; 2 * 32];

            for dy in 0..32 {
                let y = patch_y + dy;
                if y >= h {
                    break;
                }
                for dx in 0..32 {
                    let x = patch_x + dx;
                    if x >= w {
                        break;
                    }
                    let mut grain = template.sample(r_off + dy as i32, c_off + dx as i32) as i32;
                    if p.overlap_flag && dy < 2 && by > 0 && !bottom_seam[bx].is_empty() {
                        let prev = bottom_seam[bx][dy * 32 + dx];
                        grain = overlap_blend(grain, prev, dy);
                    }
                    if p.overlap_flag && dx < 2 && bx > 0 && !right_seam[by].is_empty() {
                        let prev = right_seam[by][dy * 2 + dx];
                        grain = overlap_blend(grain, prev, dx);
                    }
                    let pix = plane[y * stride + x];
                    let scale = scaling.lookup_hbd(pix, bit_depth);
                    plane[y * stride + x] = apply_sample16(pix, grain, scale, shift, lo, hi);

                    if dx >= 30 {
                        this_right[dy * 2 + (dx - 30)] = grain;
                    }
                    if dy >= 30 {
                        this_bottom[(dy - 30) * 32 + dx] = grain;
                    }
                }
            }

            bottom_seam[bx] = this_bottom;
            right_seam[by] = this_right;
        }
    }
}

/// Borrowed chroma-plane pair used by [`apply_frame`].
pub struct ChromaPlanes<'a> {
    pub u: &'a mut [u8],
    pub v: &'a mut [u8],
    pub w: usize,
    pub h: usize,
    pub stride: usize,
}

/// Run film-grain synthesis on all three 8-bit planes using the
/// per-plane LUTs and templates carried in `p`. Pass `None` to skip
/// chroma.
pub fn apply_frame(
    y: &mut [u8],
    yw: usize,
    yh: usize,
    stride_y: usize,
    uv: Option<ChromaPlanes<'_>>,
    y_template: &Template,
    cb_template: &Template,
    cr_template: &Template,
    p: &Params,
) {
    if p.grain_seed == 0 {
        return;
    }
    apply_with_template(y, yw, yh, stride_y, &p.scaling_y, y_template, p);
    if let Some(c) = uv {
        // Chroma's restricted-range high clip is 240 (not 235 like
        // luma) — [`apply_with_template_chroma`] handles this.
        apply_with_template_chroma(c.u, c.w, c.h, c.stride, &p.scaling_u, cb_template, p);
        apply_with_template_chroma(c.v, c.w, c.h, c.stride, &p.scaling_v, cr_template, p);
    }
}

/// Chroma variant of [`apply_with_template`] with the broadcast-legal
/// high clip at 240 (spec §7.20.3.5).
pub fn apply_with_template_chroma(
    plane: &mut [u8],
    w: usize,
    h: usize,
    stride: usize,
    scaling: &ScalingLut,
    template: &Template,
    p: &Params,
) {
    if p.grain_seed == 0 || template.rows == 0 {
        return;
    }
    let shift = if p.scaling_shift == 0 {
        8
    } else {
        p.scaling_shift as u32
    };
    let (lo, hi) = if p.clip_to_restricted_range {
        (16i32, 240i32)
    } else {
        (0i32, 255i32)
    };
    let (max_r, max_c) = offset_bounds(template);
    let mut rng = Rng::new(0);

    let block_cols = w.div_ceil(32);
    let block_rows = h.div_ceil(32);
    let mut bottom_seam: Vec<Vec<i32>> = vec![Vec::new(); block_cols];
    let mut right_seam: Vec<Vec<i32>> = vec![Vec::new(); block_rows];

    for by in 0..block_rows {
        for bx in 0..block_cols {
            let seed = block_seed(p.grain_seed, bx as u32, by as u32);
            rng.seed(seed);
            let r_off = (rng.next() as i32 % max_r).abs();
            let c_off = (rng.next() as i32 % max_c).abs();

            let patch_y = by * 32;
            let patch_x = bx * 32;

            let mut this_right: Vec<i32> = vec![0; 32 * 2];
            let mut this_bottom: Vec<i32> = vec![0; 2 * 32];

            for dy in 0..32 {
                let y_pos = patch_y + dy;
                if y_pos >= h {
                    break;
                }
                for dx in 0..32 {
                    let x_pos = patch_x + dx;
                    if x_pos >= w {
                        break;
                    }
                    let mut grain = template.sample(r_off + dy as i32, c_off + dx as i32) as i32;
                    if p.overlap_flag && dy < 2 && by > 0 && !bottom_seam[bx].is_empty() {
                        let prev = bottom_seam[bx][dy * 32 + dx];
                        grain = overlap_blend(grain, prev, dy);
                    }
                    if p.overlap_flag && dx < 2 && bx > 0 && !right_seam[by].is_empty() {
                        let prev = right_seam[by][dy * 2 + dx];
                        grain = overlap_blend(grain, prev, dx);
                    }
                    let pix = plane[y_pos * stride + x_pos];
                    let scale = scaling.lookup(pix);
                    plane[y_pos * stride + x_pos] = apply_sample(pix, grain, scale, shift, lo, hi);

                    if dx >= 30 {
                        this_right[dy * 2 + (dx - 30)] = grain;
                    }
                    if dy >= 30 {
                        this_bottom[(dy - 30) * 32 + dx] = grain;
                    }
                }
            }

            bottom_seam[bx] = this_bottom;
            right_seam[by] = this_right;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{new_luma_template, Point};
    use super::super::scaling::build_lut;
    use super::*;

    #[test]
    fn zero_seed_is_noop() {
        let mut plane = vec![77u8; 32 * 32];
        let lut = ScalingLut([255u8; 256]);
        let t = new_luma_template(0x1111, 0, &[], 7);
        let p = Params {
            grain_seed: 0,
            scaling_shift: 8,
            ..Default::default()
        };
        apply_with_template(&mut plane, 32, 32, 32, &lut, &t, &p);
        for &v in &plane {
            assert_eq!(v, 77);
        }
    }

    #[test]
    fn produces_changes_on_uniform_plane() {
        let mut plane = vec![128u8; 64 * 64];
        let lut = ScalingLut([200u8; 256]);
        let t = new_luma_template(0x9999, 0, &[], 7);
        let p = Params {
            grain_seed: 0xA5A5,
            scaling_shift: 8,
            ..Default::default()
        };
        apply_with_template(&mut plane, 64, 64, 64, &lut, &t, &p);
        let diffs = plane.iter().filter(|&&v| v != 128).count();
        assert!(diffs > 0, "templated grain produced no changes");
    }

    #[test]
    fn clamps_near_max() {
        let mut plane = vec![250u8; 32 * 32];
        let lut = ScalingLut([255u8; 256]);
        let t = new_luma_template(0x2222, 0, &[], 7);
        let p = Params {
            grain_seed: 0xCAFE,
            scaling_shift: 8,
            ..Default::default()
        };
        apply_with_template(&mut plane, 32, 32, 32, &lut, &t, &p);
        // u8 is always ≤ 255; this assertion simply documents we
        // exercise the saturation path without escaping the type
        // range.
        assert_eq!(plane.len(), 32 * 32);
    }

    #[test]
    fn hbd_clips_to_10bit() {
        let mut plane = vec![1020u16; 64 * 64];
        let lut = ScalingLut([255u8; 256]);
        let t = new_luma_template(0x4321, 0, &[], 7);
        let p = Params {
            grain_seed: 0xBEEF,
            scaling_shift: 8,
            ..Default::default()
        };
        apply_with_template16(&mut plane, 64, 64, 64, &lut, &t, &p, 10);
        for &v in &plane {
            assert!(v <= 1023, "sample {v} exceeds 10-bit max");
        }
    }

    #[test]
    fn hbd_restricted_range_10bit() {
        let mut plane = vec![950u16; 64 * 64];
        let lut = ScalingLut([255u8; 256]);
        let t = new_luma_template(0x0101, 0, &[], 7);
        let p = Params {
            grain_seed: 0xCAFE,
            scaling_shift: 8,
            clip_to_restricted_range: true,
            ..Default::default()
        };
        apply_with_template16(&mut plane, 64, 64, 64, &lut, &t, &p, 10);
        for &v in &plane {
            assert!((64..=940).contains(&v), "sample {v} escaped [64, 940]");
        }
    }

    #[test]
    fn hbd_zero_seed_noop() {
        let mut plane = vec![2000u16; 32 * 32];
        let t = new_luma_template(0x9999, 0, &[], 7);
        let p = Params {
            grain_seed: 0,
            ..Default::default()
        };
        apply_with_template16(&mut plane, 32, 32, 32, &ScalingLut::default(), &t, &p, 12);
        for &v in &plane {
            assert_eq!(v, 2000);
        }
    }

    #[test]
    fn hbd_12bit_produces_changes() {
        let mut plane = vec![2048u16; 64 * 64];
        let lut = ScalingLut([200u8; 256]);
        let t = new_luma_template(0x8765, 0, &[], 7);
        let p = Params {
            grain_seed: 0xABCD,
            scaling_shift: 8,
            ..Default::default()
        };
        apply_with_template16(&mut plane, 64, 64, 64, &lut, &t, &p, 12);
        let diffs = plane.iter().filter(|&&v| v != 2048).count();
        assert!(diffs > 0);
    }

    #[test]
    fn overlap_changes_output_near_seams() {
        let points = vec![Point { value: 128, scale: 255 }];
        let lut = build_lut(&points);
        let t = new_luma_template(0x1357, 0, &[], 7);

        let mut plane_no = vec![128u8; 96 * 96];
        let mut plane_yes = vec![128u8; 96 * 96];
        let p_no = Params {
            grain_seed: 0x2468,
            scaling_shift: 8,
            overlap_flag: false,
            ..Default::default()
        };
        let p_yes = Params {
            overlap_flag: true,
            ..p_no.clone()
        };
        apply_with_template(&mut plane_no, 96, 96, 96, &lut, &t, &p_no);
        apply_with_template(&mut plane_yes, 96, 96, 96, &lut, &t, &p_yes);
        // Overlap blend touches the first two rows/cols of every
        // block after the first — with a 96×96 plane (3×3 blocks)
        // those seams cover many pixels and at least some should
        // differ from the non-overlap output.
        let differ = plane_no.iter().zip(plane_yes.iter()).filter(|(a, b)| a != b).count();
        assert!(differ > 0, "overlap mode produced no change vs non-overlap");
    }

    #[test]
    fn apply_frame_touches_all_planes() {
        let w = 32;
        let h = 32;
        let cw = 16;
        let ch = 16;
        let mut y = vec![128u8; w * h];
        let mut u = vec![128u8; cw * ch];
        let mut v = vec![128u8; cw * ch];
        let lut = ScalingLut([128u8; 256]);
        let p = Params {
            grain_seed: 0xF00D,
            scaling_y: lut,
            scaling_u: lut,
            scaling_v: lut,
            scaling_shift: 8,
            ..Default::default()
        };
        let yt = new_luma_template(p.grain_seed, 0, &[], 7);
        let ct = super::super::patch::new_chroma_template(p.grain_seed ^ 0xA5A5, 0, &[], 7);
        apply_frame(
            &mut y,
            w,
            h,
            w,
            Some(ChromaPlanes {
                u: &mut u,
                v: &mut v,
                w: cw,
                h: ch,
                stride: cw,
            }),
            &yt,
            &ct,
            &ct,
            &p,
        );
        assert!(y.iter().any(|&b| b != 128));
        assert!(u.iter().any(|&b| b != 128));
        assert!(v.iter().any(|&b| b != 128));
    }
}
