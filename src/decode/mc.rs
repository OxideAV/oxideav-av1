//! AV1 motion compensation — §7.11.3 single-reference translational.
//!
//! Produces a predicted block of samples from a reference plane by
//! combining integer-pel offset + 8-tap sub-pel interpolation, with
//! edge-clamp on source reads so MVs that point past the frame
//! boundary still return valid samples.
//!
//! MV components are in eighth-pel units. Integer MV (`hp == vp == 0`)
//! takes the direct-copy fast path; otherwise we stage a
//! `(w+7)×(h+7)` padded reference region and call the 8-tap filter in
//! [`crate::predict::interp`].
//!
//! Spec §7.11.3.3 `motion_vector_clamping`: MV is clamped so the
//! predicted block stays within `[-MV_BORDER, frame_w + MV_BORDER]` on
//! each axis. `MV_BORDER` is 128 samples (spec §3 constants table).

use crate::predict::interp::{interp_sub_pel, interp_sub_pel16, InterpFilter};

use super::mv::Mv;

/// §7.11.3.3 `MV_BORDER` — 128 luma samples on every side of the
/// reference frame where MVs may still point before the clamp pulls
/// them in. Equivalent constant for chroma planes is divided by the
/// subsampling factor; callers pre-scale the MV before calling into
/// this module.
pub const MV_BORDER: i32 = 128;

/// Clamp the block-relative MV so that `(bx + int_x, by + int_y)` lands
/// inside the reference frame expanded by `MV_BORDER` on each side.
/// Returns a clamped [`Mv`] in eighth-pel units. Spec §7.11.3.3.
///
/// The clamp is applied at integer-pel precision; fractional phases
/// are preserved so sub-pel filtering lands on the same sample.
pub fn clamp_mv_to_frame(
    mv: Mv,
    bx: i32,
    by: i32,
    bw: i32,
    bh: i32,
    frame_w: i32,
    frame_h: i32,
) -> Mv {
    // Separate integer and fractional components (eighth-pel units).
    let int_col = mv.col >> 3;
    let frac_col = mv.col & 7;
    let int_row = mv.row >> 3;
    let frac_row = mv.row & 7;
    // Allowed integer offsets per spec §7.11.3.3.
    let min_col = -bx - MV_BORDER - bw;
    let max_col = frame_w - bx + MV_BORDER;
    let min_row = -by - MV_BORDER - bh;
    let max_row = frame_h - by + MV_BORDER;
    let int_col = int_col.clamp(min_col, max_col);
    let int_row = int_row.clamp(min_row, max_row);
    Mv {
        col: (int_col << 3) | frac_col,
        row: (int_row << 3) | frac_row,
    }
}

/// Produce a `w×h` 8-bit predicted block at `dst` using the reference
/// plane `ref_y` at logical position `(bx + mv.col, by + mv.row)` with
/// MV components in eighth-pel units. The reference is edge-clamped
/// so off-frame MVs are legal.
///
/// `filt` selects the 8-tap filter set. When both sub-pel phases are
/// zero, the integer-pel fast path (direct copy) is used.
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate(
    dst: &mut [u8],
    w: usize,
    h: usize,
    ref_y: &[u8],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    bx: i32,
    by: i32,
    mv: Mv,
    filt: InterpFilter,
) {
    // §7.11.3.3 clamp MV to `[-MV_BORDER, frame + MV_BORDER]`.
    let mv = clamp_mv_to_frame(mv, bx, by, w as i32, h as i32, ref_w as i32, ref_h as i32);
    let int_x = mv.col >> 3;
    let int_y = mv.row >> 3;
    let phase_x = (mv.col & 7) as usize;
    let phase_y = (mv.row & 7) as usize;
    // Spec §7.11.3.4 uses a 16-phase filter table; eighth-pel MVs map
    // to even phases (0, 2, 4, ..., 14). allow_high_precision_mv
    // fills in the odd phases — our decoder pins hp=1 when HP is
    // disabled, which is reflected in `Mv` already.
    let hp = phase_x * 2;
    let vp = phase_y * 2;

    if hp == 0 && vp == 0 {
        integer_copy_clamped(
            dst,
            w,
            h,
            ref_y,
            ref_w,
            ref_h,
            ref_stride,
            bx + int_x,
            by + int_y,
        );
        return;
    }

    let pad_stride = w + 7;
    let pad_len = pad_stride * (h + 7);
    let mut pad = vec![0u8; pad_len];
    for r in 0..h + 7 {
        let sy = (by + int_y + r as i32 - 3).clamp(0, (ref_h as i32) - 1) as usize;
        for c in 0..w + 7 {
            let sx = (bx + int_x + c as i32 - 3).clamp(0, (ref_w as i32) - 1) as usize;
            pad[r * pad_stride + c] = ref_y[sy * ref_stride + sx];
        }
    }
    interp_sub_pel(dst, w, h, &pad, pad_stride, hp, vp, filt);
}

/// HBD counterpart of [`motion_compensate`].
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    ref_y: &[u16],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    bx: i32,
    by: i32,
    mv: Mv,
    filt: InterpFilter,
    bit_depth: u32,
) {
    // §7.11.3.3 clamp MV to `[-MV_BORDER, frame + MV_BORDER]`.
    let mv = clamp_mv_to_frame(mv, bx, by, w as i32, h as i32, ref_w as i32, ref_h as i32);
    let int_x = mv.col >> 3;
    let int_y = mv.row >> 3;
    let phase_x = (mv.col & 7) as usize;
    let phase_y = (mv.row & 7) as usize;
    let hp = phase_x * 2;
    let vp = phase_y * 2;

    if hp == 0 && vp == 0 {
        integer_copy_clamped16(
            dst,
            w,
            h,
            ref_y,
            ref_w,
            ref_h,
            ref_stride,
            bx + int_x,
            by + int_y,
        );
        return;
    }

    let pad_stride = w + 7;
    let pad_len = pad_stride * (h + 7);
    let mut pad = vec![0u16; pad_len];
    for r in 0..h + 7 {
        let sy = (by + int_y + r as i32 - 3).clamp(0, (ref_h as i32) - 1) as usize;
        for c in 0..w + 7 {
            let sx = (bx + int_x + c as i32 - 3).clamp(0, (ref_w as i32) - 1) as usize;
            pad[r * pad_stride + c] = ref_y[sy * ref_stride + sx];
        }
    }
    interp_sub_pel16(dst, w, h, &pad, pad_stride, hp, vp, filt, bit_depth);
}

/// Integer-pel MC with edge clamp. Used directly by the u8 fast path
/// and by HBD when sub-pel phases are zero.
#[allow(clippy::too_many_arguments)]
fn integer_copy_clamped(
    dst: &mut [u8],
    w: usize,
    h: usize,
    src: &[u8],
    src_w: usize,
    src_h: usize,
    src_stride: usize,
    bx: i32,
    by: i32,
) {
    for r in 0..h {
        let sy = (by + r as i32).clamp(0, (src_h as i32) - 1) as usize;
        for c in 0..w {
            let sx = (bx + c as i32).clamp(0, (src_w as i32) - 1) as usize;
            dst[r * w + c] = src[sy * src_stride + sx];
        }
    }
}

/// HBD integer-pel MC with edge clamp.
#[allow(clippy::too_many_arguments)]
fn integer_copy_clamped16(
    dst: &mut [u16],
    w: usize,
    h: usize,
    src: &[u16],
    src_w: usize,
    src_h: usize,
    src_stride: usize,
    bx: i32,
    by: i32,
) {
    for r in 0..h {
        let sy = (by + r as i32).clamp(0, (src_h as i32) - 1) as usize;
        for c in 0..w {
            let sx = (bx + c as i32).clamp(0, (src_w as i32) - 1) as usize;
            dst[r * w + c] = src[sy * src_stride + sx];
        }
    }
}

/// Bi-prediction averaging — spec §7.11.3.9 `AverageMc`.
///
/// Combines two per-block single-reference prediction buffers into
/// one via `(p0 + p1 + 1) >> 1`. Used for compound inter prediction
/// when both references are active and no mask is applied.
pub fn average_mc_u8(p0: &[u8], p1: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(p0.len(), p1.len());
    debug_assert_eq!(p0.len(), dst.len());
    for i in 0..p0.len() {
        dst[i] = (((p0[i] as u16) + (p1[i] as u16) + 1) >> 1) as u8;
    }
}

/// HBD counterpart of [`average_mc_u8`]. `bit_depth` selects the
/// clipping range (unused in the average itself, kept for signature
/// parity with the other HBD calls).
pub fn average_mc_u16(p0: &[u16], p1: &[u16], dst: &mut [u16], _bit_depth: u32) {
    debug_assert_eq!(p0.len(), p1.len());
    debug_assert_eq!(p0.len(), dst.len());
    for i in 0..p0.len() {
        dst[i] = (((p0[i] as u32) + (p1[i] as u32) + 1) >> 1) as u16;
    }
}

/// Masked-compound blend — simplified §7.11.3.9 `BlendMc` path using a
/// raw 0..=64 weight per sample. `mask[i]` is the weight applied to
/// `p0[i]`; `(64 - mask[i])` is applied to `p1[i]`. Output is rounded
/// half-up.
///
/// Full AV1 wedge / diffwt / smooth-ii masks are generated from §7.11.3
/// tables; this helper takes a pre-computed mask so callers can plug
/// it in once the mask-generator is wired.
pub fn blend_mc_u8(p0: &[u8], p1: &[u8], mask: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(p0.len(), p1.len());
    debug_assert_eq!(p0.len(), mask.len());
    debug_assert_eq!(p0.len(), dst.len());
    for i in 0..p0.len() {
        let m = mask[i] as u32;
        let inv = 64 - m;
        let v = ((p0[i] as u32) * m + (p1[i] as u32) * inv + 32) >> 6;
        dst[i] = v.min(255) as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_pel_zero_mv_copies_reference() {
        let w = 16;
        let h = 16;
        let mut r = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                r[y * 64 + x] = ((x + y * 7) & 0xFF) as u8;
            }
        }
        let mut dst = vec![0u8; w * h];
        motion_compensate(
            &mut dst,
            w,
            h,
            &r,
            64,
            64,
            64,
            16,
            16,
            Mv { row: 0, col: 0 },
            InterpFilter::Regular,
        );
        for row in 0..h {
            for col in 0..w {
                let want = r[(16 + row) * 64 + (16 + col)];
                assert_eq!(dst[row * w + col], want, "mismatch at {row},{col}");
            }
        }
    }

    #[test]
    fn shifted_integer_pel_picks_up_offset() {
        let w = 8;
        let h = 8;
        let mut r = vec![0u8; 64 * 64];
        for y in 0..64 {
            for x in 0..64 {
                r[y * 64 + x] = x as u8;
            }
        }
        let mut dst = vec![0u8; w * h];
        // MV eighth-pel: col=16 → +2 integer pel; row=16 → +2.
        motion_compensate(
            &mut dst,
            w,
            h,
            &r,
            64,
            64,
            64,
            10,
            10,
            Mv { row: 16, col: 16 },
            InterpFilter::Regular,
        );
        for row in 0..h {
            for col in 0..w {
                let want = r[(10 + 2 + row) * 64 + (10 + 2 + col)];
                assert_eq!(dst[row * w + col], want, "mismatch at {row},{col}");
            }
        }
    }

    #[test]
    fn clamps_past_edge_returns_edge_pixel() {
        let w = 8;
        let h = 8;
        let r = vec![0x42u8; 16 * 16];
        let mut dst = vec![0u8; w * h];
        // Block at (0, 0), MV ≈ -16 integer pel. All samples clamp to
        // the constant edge value 0x42.
        motion_compensate(
            &mut dst,
            w,
            h,
            &r,
            16,
            16,
            16,
            0,
            0,
            Mv {
                row: -128,
                col: -128,
            },
            InterpFilter::Regular,
        );
        for v in &dst {
            assert_eq!(*v, 0x42);
        }
    }

    #[test]
    fn clamp_mv_keeps_in_bounds_mvs_unchanged() {
        // A small MV well inside the frame is passed through.
        let mv = Mv { row: 40, col: -32 };
        let out = clamp_mv_to_frame(mv, 16, 16, 8, 8, 128, 128);
        assert_eq!(out, mv);
    }

    #[test]
    fn clamp_mv_pulls_far_mvs_to_border() {
        // MV that would land ~4000 samples past the right edge is
        // clipped. Integer part clamps to frame_w + MV_BORDER - bx =
        // 128 + 128 - 0 = 256.
        let mv = Mv {
            row: 0,
            col: 4000 * 8,
        };
        let out = clamp_mv_to_frame(mv, 0, 0, 16, 16, 128, 128);
        assert_eq!(out.col >> 3, 256);
        // Fractional component preserved.
        assert_eq!(out.col & 7, 0);
    }

    #[test]
    fn clamp_mv_preserves_fractional_phase() {
        let mv = Mv {
            row: 0,
            col: -99999,
        };
        let frac = mv.col & 7;
        let out = clamp_mv_to_frame(mv, 0, 0, 16, 16, 64, 64);
        assert_eq!(out.col & 7, frac);
    }

    #[test]
    fn average_mc_u8_round_half_up() {
        let p0 = vec![10u8, 11, 12, 13];
        let p1 = vec![20u8, 21, 22, 23];
        let mut dst = vec![0u8; 4];
        average_mc_u8(&p0, &p1, &mut dst);
        // (10+20+1)>>1=15, (11+21+1)>>1=16, ...
        assert_eq!(dst, vec![15u8, 16, 17, 18]);
    }

    #[test]
    fn average_mc_u16_round_half_up() {
        let p0 = vec![100u16, 200, 300];
        let p1 = vec![120u16, 220, 320];
        let mut dst = vec![0u16; 3];
        average_mc_u16(&p0, &p1, &mut dst, 10);
        assert_eq!(dst, vec![110u16, 210, 310]);
    }

    #[test]
    fn blend_mc_u8_linear_weights() {
        let p0 = vec![0u8, 128, 255];
        let p1 = vec![255u8, 128, 0];
        // All-zero mask: blend = p1.
        let mut dst = vec![0u8; 3];
        blend_mc_u8(&p0, &p1, &[0, 0, 0], &mut dst);
        assert_eq!(dst, vec![255u8, 128, 0]);
        // All-64 mask: blend = p0.
        let mut dst = vec![0u8; 3];
        blend_mc_u8(&p0, &p1, &[64, 64, 64], &mut dst);
        assert_eq!(dst, vec![0u8, 128, 255]);
        // Half mask: mid.
        let mut dst = vec![0u8; 3];
        blend_mc_u8(&p0, &p1, &[32, 32, 32], &mut dst);
        assert_eq!(dst, vec![128u8, 128, 128]);
    }

    #[test]
    fn hbd_integer_pel_copy() {
        let w = 8;
        let h = 8;
        let mut r = vec![0u16; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                r[y * 32 + x] = ((x + y * 17) as u16) & 0x3FF;
            }
        }
        let mut dst = vec![0u16; w * h];
        motion_compensate16(
            &mut dst,
            w,
            h,
            &r,
            32,
            32,
            32,
            10,
            10,
            Mv { row: 0, col: 0 },
            InterpFilter::Regular,
            10,
        );
        for row in 0..h {
            for col in 0..w {
                assert_eq!(dst[row * w + col], r[(10 + row) * 32 + (10 + col)]);
            }
        }
    }
}
