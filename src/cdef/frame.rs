//! Frame-level CDEF driver — ported from goavif `av1/cdef/frame.go` +
//! `filter16.go`'s plane helpers.

use super::direction::{find_direction, find_direction16};
use super::filter::{filter_block, filter_block16};

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
        // being substituted for out-of-range reads (matches goavif).
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
}
