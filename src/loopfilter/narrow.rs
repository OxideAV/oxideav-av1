//! Narrow (4-tap) deblocking filter — §7.14.4 + §7.14.6.2.

/// Deblocking threshold triple. All values are expressed for 8-bit
/// depth; [`scale_thresholds16`] scales them for 10/12-bit inputs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Thresholds {
    /// Allowed local variation across the edge.
    pub limit: u8,
    /// Allowed inner-sample variation on each side.
    pub blimit: u8,
    /// High edge variation threshold (HEV detection).
    pub thresh: u8,
}

/// 16-bit-depth deblocking threshold triple. Each field is the 8-bit
/// value left-shifted by `bit_depth - 8`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Thresholds16 {
    pub limit: u16,
    pub blimit: u16,
    pub thresh: u16,
    pub bit_depth: u32,
}

/// Scale an 8-bit [`Thresholds`] up to `bit_depth`. Invalid bit-depths
/// (anything other than 10 or 12) fall through as the 8-bit identity.
pub fn scale_thresholds16(th: Thresholds, bit_depth: u32) -> Thresholds16 {
    let bd = if bit_depth != 10 && bit_depth != 12 {
        8
    } else {
        bit_depth
    };
    let shift = bd - 8;
    Thresholds16 {
        limit: (th.limit as u16) << shift,
        blimit: (th.blimit as u16) << shift,
        thresh: (th.thresh as u16) << shift,
        bit_depth: bd,
    }
}

#[inline]
fn abs_diff(a: u8, b: u8) -> u8 {
    a.abs_diff(b)
}

#[inline]
fn abs_diff16(a: u16, b: u16) -> u16 {
    a.abs_diff(b)
}

/// Report whether the 4-tap narrow filter should be applied given the
/// four samples straddling the edge.
pub fn narrow_mask(p1: u8, p0: u8, q0: u8, q1: u8, th: Thresholds) -> bool {
    if abs_diff(p1, p0) > th.blimit {
        return false;
    }
    if abs_diff(q1, q0) > th.blimit {
        return false;
    }
    // Widen to u16 — `abs_diff(p0, q0) * 2` can otherwise overflow u8
    // (an edge with a 200-step jump produces 400). Spec §7.14.4.2.
    let combined = (abs_diff(p0, q0) as u16) * 2 + (abs_diff(p1, q1) as u16) / 2;
    if combined > th.limit as u16 {
        return false;
    }
    true
}

/// 16-bit narrow mask.
pub fn narrow_mask16(p1: u16, p0: u16, q0: u16, q1: u16, th: Thresholds16) -> bool {
    if abs_diff16(p1, p0) > th.blimit {
        return false;
    }
    if abs_diff16(q1, q0) > th.blimit {
        return false;
    }
    // Widen to u32 to avoid u16 overflow on 12-bit samples
    // (max abs_diff is 4095, so `* 2 = 8190` still fits u16, but a
    // 16-bit-depth scaled limit comparison is safer in u32).
    let combined = (abs_diff16(p0, q0) as u32) * 2 + (abs_diff16(p1, q1) as u32) / 2;
    if combined > th.limit as u32 {
        return false;
    }
    true
}

/// High-edge-variation detector — spec §7.14.5.
pub fn high_edge_variation(p1: u8, p0: u8, q0: u8, q1: u8, thresh: u8) -> bool {
    abs_diff(p1, p0) > thresh || abs_diff(q1, q0) > thresh
}

/// 16-bit HEV detector.
pub fn high_edge_variation16(p1: u16, p0: u16, q0: u16, q1: u16, thresh: u16) -> bool {
    abs_diff16(p1, p0) > thresh || abs_diff16(q1, q0) > thresh
}

#[inline]
fn clip_s8(v: i32) -> i32 {
    v.clamp(-128, 127)
}

#[inline]
fn clip_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// 4-tap narrow deblocking filter (§7.14.6.2). Caller must have
/// verified the mask with [`narrow_mask`] before calling.
pub fn filter4(p1: u8, p0: u8, q0: u8, q1: u8, hev: bool) -> (u8, u8, u8, u8) {
    let a = clip_s8(p1 as i32 - q1 as i32);
    let a = if hev { 0 } else { a };
    let b = clip_s8(3 * (q0 as i32 - p0 as i32) + a);
    let c1 = clip_s8(b + 4) >> 3;
    let c2 = clip_s8(b + 3) >> 3;

    let new_p0 = clip_u8(p0 as i32 + c2);
    let new_q0 = clip_u8(q0 as i32 - c1);
    let (new_p1, new_q1) = if !hev {
        let d = (c1 + 1) >> 1;
        (clip_u8(p1 as i32 + d), clip_u8(q1 as i32 - d))
    } else {
        (p1, q1)
    };
    (new_p1, new_p0, new_q0, new_q1)
}

/// 16-bit narrow filter (libaom `highbd_filter4`).
pub fn filter4_16(
    p1: u16,
    p0: u16,
    q0: u16,
    q1: u16,
    hev: bool,
    bit_depth: u32,
) -> (u16, u16, u16, u16) {
    let shift = bit_depth - 8;
    let clip_limit = 128i32 << shift;
    let max_v = ((1i32) << bit_depth) - 1;

    let clip_s = |v: i32| -> i32 {
        if v < -clip_limit {
            -clip_limit
        } else if v > clip_limit - 1 {
            clip_limit - 1
        } else {
            v
        }
    };
    let clip_u = |v: i32| -> u16 { v.clamp(0, max_v) as u16 };

    let a = clip_s(p1 as i32 - q1 as i32);
    let a = if hev { 0 } else { a };
    let b = clip_s(3 * (q0 as i32 - p0 as i32) + a);
    let c1 = clip_s(b + 4) >> 3;
    let c2 = clip_s(b + 3) >> 3;

    let new_p0 = clip_u(p0 as i32 + c2);
    let new_q0 = clip_u(q0 as i32 - c1);
    let (new_p1, new_q1) = if !hev {
        let d = (c1 + 1) >> 1;
        (clip_u(p1 as i32 + d), clip_u(q1 as i32 - d))
    } else {
        (p1, q1)
    };
    (new_p1, new_p0, new_q0, new_q1)
}

/// Apply [`filter4`] to every row of a vertical edge at column `x`.
pub fn apply_vertical_edge4(img: &mut [u8], stride: usize, x: usize, h: usize, th: Thresholds) {
    for r in 0..h {
        let base = r * stride + x - 2;
        let (p1, p0, q0, q1) = (img[base], img[base + 1], img[base + 2], img[base + 3]);
        if !narrow_mask(p1, p0, q0, q1, th) {
            continue;
        }
        let hev = high_edge_variation(p1, p0, q0, q1, th.thresh);
        let (np1, np0, nq0, nq1) = filter4(p1, p0, q0, q1, hev);
        img[base] = np1;
        img[base + 1] = np0;
        img[base + 2] = nq0;
        img[base + 3] = nq1;
    }
}

/// Apply [`filter4`] to every column of a horizontal edge at row `y`.
pub fn apply_horizontal_edge4(img: &mut [u8], stride: usize, y: usize, w: usize, th: Thresholds) {
    for c in 0..w {
        let p1 = img[(y - 2) * stride + c];
        let p0 = img[(y - 1) * stride + c];
        let q0 = img[y * stride + c];
        let q1 = img[(y + 1) * stride + c];
        if !narrow_mask(p1, p0, q0, q1, th) {
            continue;
        }
        let hev = high_edge_variation(p1, p0, q0, q1, th.thresh);
        let (np1, np0, nq0, nq1) = filter4(p1, p0, q0, q1, hev);
        img[(y - 2) * stride + c] = np1;
        img[(y - 1) * stride + c] = np0;
        img[y * stride + c] = nq0;
        img[(y + 1) * stride + c] = nq1;
    }
}

/// 16-bit counterpart of [`apply_vertical_edge4`].
pub fn apply_vertical_edge4_16(
    img: &mut [u16],
    stride: usize,
    x: usize,
    h: usize,
    th: Thresholds16,
) {
    for r in 0..h {
        let base = r * stride + x - 2;
        let (p1, p0, q0, q1) = (img[base], img[base + 1], img[base + 2], img[base + 3]);
        if !narrow_mask16(p1, p0, q0, q1, th) {
            continue;
        }
        let hev = high_edge_variation16(p1, p0, q0, q1, th.thresh);
        let (np1, np0, nq0, nq1) = filter4_16(p1, p0, q0, q1, hev, th.bit_depth);
        img[base] = np1;
        img[base + 1] = np0;
        img[base + 2] = nq0;
        img[base + 3] = nq1;
    }
}

/// 16-bit counterpart of [`apply_horizontal_edge4`].
pub fn apply_horizontal_edge4_16(
    img: &mut [u16],
    stride: usize,
    y: usize,
    w: usize,
    th: Thresholds16,
) {
    for c in 0..w {
        let p1 = img[(y - 2) * stride + c];
        let p0 = img[(y - 1) * stride + c];
        let q0 = img[y * stride + c];
        let q1 = img[(y + 1) * stride + c];
        if !narrow_mask16(p1, p0, q0, q1, th) {
            continue;
        }
        let hev = high_edge_variation16(p1, p0, q0, q1, th.thresh);
        let (np1, np0, nq0, nq1) = filter4_16(p1, p0, q0, q1, hev, th.bit_depth);
        img[(y - 2) * stride + c] = np1;
        img[(y - 1) * stride + c] = np0;
        img[y * stride + c] = nq0;
        img[(y + 1) * stride + c] = nq1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_flat_edge() {
        let th = Thresholds {
            limit: 20,
            blimit: 10,
            thresh: 8,
        };
        assert!(narrow_mask(100, 100, 100, 100, th));
    }

    #[test]
    fn mask_rejects_big_jump() {
        let th = Thresholds {
            limit: 20,
            blimit: 10,
            thresh: 8,
        };
        assert!(!narrow_mask(100, 100, 200, 200, th));
    }

    #[test]
    fn filter4_softens_block_edge() {
        let th = Thresholds {
            limit: 30,
            blimit: 10,
            thresh: 8,
        };
        let (p1, p0, q0, q1) = (110u8, 110u8, 120u8, 120u8);
        assert!(narrow_mask(p1, p0, q0, q1, th));
        let hev = high_edge_variation(p1, p0, q0, q1, th.thresh);
        let (_np1, np0, nq0, _nq1) = filter4(p1, p0, q0, q1, hev);
        assert!(np0 > p0);
        assert!(nq0 < q0);
    }

    #[test]
    fn apply_vertical_edge4_softens_2row_image() {
        let mut img = vec![110, 110, 120, 120, 110, 110, 120, 120];
        apply_vertical_edge4(
            &mut img,
            4,
            2,
            2,
            Thresholds {
                limit: 30,
                blimit: 10,
                thresh: 8,
            },
        );
        assert!(img[1] > 110);
        assert!(img[2] < 120);
    }
}
