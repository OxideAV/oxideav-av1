//! AV1 motion compensation — §7.11.3 single-reference translational.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/mc.go` +
//! `mc16.go` (MIT, KarpelesLab/goavif). Produces a predicted block of
//! samples from a reference plane by combining integer-pel offset +
//! 8-tap sub-pel interpolation, with edge-clamp on source reads so MVs
//! that point past the frame boundary still return valid samples.
//!
//! MV components are in eighth-pel units. Integer MV (`hp == vp == 0`)
//! takes the direct-copy fast path; otherwise we stage a
//! `(w+7)×(h+7)` padded reference region and call the 8-tap filter in
//! [`crate::predict::interp`].

use crate::predict::interp::{interp_sub_pel, interp_sub_pel16, InterpFilter};

use super::mv::Mv;

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
