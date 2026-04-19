//! Per-plane loop-restoration frame driver.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/lr/{frame,frame16}.go`
//! (MIT, KarpelesLab/goavif). The goavif driver takes a
//! `UnitFn` closure that provides per-unit parameters by coordinate;
//! we keep that signature so tests can exercise the filter without
//! plumbing the full tile decoder, but the production path is a
//! pre-materialised table of [`super::UnitParams`] fed in from the
//! tile-level per-unit signal decoder (§5.11.40-.44).

use super::sgr::{apply_sgr, apply_sgr16};
use super::wiener::{apply_wiener, apply_wiener16};
use super::{FilterType, UnitParams};

/// 8-bit reconstructed plane.
pub struct Plane<'a> {
    pub pix: &'a mut [u8],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

/// 10/12-bit reconstructed plane.
pub struct Plane16<'a> {
    pub pix: &'a mut [u16],
    pub stride: usize,
    pub width: usize,
    pub height: usize,
}

fn slice_unit(pix: &[u8], stride: usize, x: usize, y: usize, uw: usize, uh: usize) -> Vec<u8> {
    let mut out = vec![0u8; uw * uh];
    for r in 0..uh {
        let off = (y + r) * stride + x;
        out[r * uw..r * uw + uw].copy_from_slice(&pix[off..off + uw]);
    }
    out
}

fn write_unit(
    pix: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    uw: usize,
    uh: usize,
    src: &[u8],
) {
    for r in 0..uh {
        let off = (y + r) * stride + x;
        pix[off..off + uw].copy_from_slice(&src[r * uw..r * uw + uw]);
    }
}

fn slice_unit16(pix: &[u16], stride: usize, x: usize, y: usize, uw: usize, uh: usize) -> Vec<u16> {
    let mut out = vec![0u16; uw * uh];
    for r in 0..uh {
        let off = (y + r) * stride + x;
        out[r * uw..r * uw + uw].copy_from_slice(&pix[off..off + uw]);
    }
    out
}

fn write_unit16(
    pix: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    uw: usize,
    uh: usize,
    src: &[u16],
) {
    for r in 0..uh {
        let off = (y + r) * stride + x;
        pix[off..off + uw].copy_from_slice(&src[r * uw..r * uw + uw]);
    }
}

/// Walk `unit_size × unit_size` restoration units across the plane.
/// For each unit, `params_for(unit_col_index, unit_row_index)` returns
/// the unit's parameters; the unit is then filtered in place according
/// to `params.filter`. Partial units on the right/bottom are clipped
/// to the plane extent.
pub fn apply_frame<F>(p: Plane<'_>, unit_size: usize, mut params_for: F)
where
    F: FnMut(usize, usize) -> UnitParams,
{
    if unit_size == 0 {
        return;
    }
    let mut unit_row = 0usize;
    let mut y = 0usize;
    while y < p.height {
        let uh = (p.height - y).min(unit_size);
        let mut unit_col = 0usize;
        let mut x = 0usize;
        while x < p.width {
            let uw = (p.width - x).min(unit_size);
            let params = params_for(unit_col, unit_row);
            match params.filter {
                FilterType::None => {}
                FilterType::Wiener => {
                    let unit_src = slice_unit(p.pix, p.stride, x, y, uw, uh);
                    let mut unit_dst = vec![0u8; uw * uh];
                    apply_wiener(
                        &mut unit_dst,
                        &unit_src,
                        uw,
                        uh,
                        uw,
                        params.wiener_horiz,
                        params.wiener_vert,
                    );
                    write_unit(p.pix, p.stride, x, y, uw, uh, &unit_dst);
                }
                FilterType::Sgr => {
                    let unit_src = slice_unit(p.pix, p.stride, x, y, uw, uh);
                    let mut unit_dst = vec![0u8; uw * uh];
                    apply_sgr(&mut unit_dst, &unit_src, uw, uh, uw, params.sgr);
                    write_unit(p.pix, p.stride, x, y, uw, uh, &unit_dst);
                }
            }
            x += unit_size;
            unit_col += 1;
        }
        y += unit_size;
        unit_row += 1;
    }
}

/// `uint16` counterpart of [`apply_frame`].
pub fn apply_frame16<F>(p: Plane16<'_>, unit_size: usize, mut params_for: F, bit_depth: u32)
where
    F: FnMut(usize, usize) -> UnitParams,
{
    if unit_size == 0 {
        return;
    }
    let mut unit_row = 0usize;
    let mut y = 0usize;
    while y < p.height {
        let uh = (p.height - y).min(unit_size);
        let mut unit_col = 0usize;
        let mut x = 0usize;
        while x < p.width {
            let uw = (p.width - x).min(unit_size);
            let params = params_for(unit_col, unit_row);
            match params.filter {
                FilterType::None => {}
                FilterType::Wiener => {
                    let unit_src = slice_unit16(p.pix, p.stride, x, y, uw, uh);
                    let mut unit_dst = vec![0u16; uw * uh];
                    apply_wiener16(
                        &mut unit_dst,
                        &unit_src,
                        uw,
                        uh,
                        uw,
                        params.wiener_horiz,
                        params.wiener_vert,
                        bit_depth,
                    );
                    write_unit16(p.pix, p.stride, x, y, uw, uh, &unit_dst);
                }
                FilterType::Sgr => {
                    let unit_src = slice_unit16(p.pix, p.stride, x, y, uw, uh);
                    let mut unit_dst = vec![0u16; uw * uh];
                    apply_sgr16(&mut unit_dst, &unit_src, uw, uh, uw, params.sgr, bit_depth);
                    write_unit16(p.pix, p.stride, x, y, uw, uh, &unit_dst);
                }
            }
            x += unit_size;
            unit_col += 1;
        }
        y += unit_size;
        unit_row += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::WienerTaps;
    use super::*;

    #[test]
    fn filter_none_is_identity() {
        let w = 16;
        let h = 16;
        let mut pix: Vec<u8> = (0..w * h).map(|i| (i & 0xFF) as u8).collect();
        let orig = pix.clone();
        let plane = Plane {
            pix: &mut pix,
            stride: w,
            width: w,
            height: h,
        };
        apply_frame(plane, 8, |_, _| UnitParams {
            filter: FilterType::None,
            ..Default::default()
        });
        assert_eq!(pix, orig);
    }

    #[test]
    fn wiener_identity_preserves() {
        let w = 16;
        let h = 16;
        let mut pix: Vec<u8> = (0..w * h).map(|i| ((i * 17) & 0xFF) as u8).collect();
        let orig = pix.clone();
        let plane = Plane {
            pix: &mut pix,
            stride: w,
            width: w,
            height: h,
        };
        apply_frame(plane, 8, |_, _| UnitParams {
            filter: FilterType::Wiener,
            wiener_horiz: WienerTaps::IDENTITY,
            wiener_vert: WienerTaps::IDENTITY,
            ..Default::default()
        });
        assert_eq!(pix, orig);
    }

    #[test]
    fn sgr_zero_radii_is_identity() {
        let w = 16;
        let h = 16;
        let mut pix: Vec<u8> = (0..w * h).map(|i| ((i * 41) & 0xFF) as u8).collect();
        let orig = pix.clone();
        let plane = Plane {
            pix: &mut pix,
            stride: w,
            width: w,
            height: h,
        };
        apply_frame(plane, 8, |_, _| UnitParams {
            filter: FilterType::Sgr,
            ..Default::default()
        });
        assert_eq!(pix, orig);
    }

    #[test]
    fn hbd_wiener_identity_preserves() {
        let w = 32;
        let h = 32;
        let mut pix = vec![512u16; w * h];
        let orig = pix.clone();
        let plane = Plane16 {
            pix: &mut pix,
            stride: w,
            width: w,
            height: h,
        };
        apply_frame16(
            plane,
            16,
            |_, _| UnitParams {
                filter: FilterType::Wiener,
                wiener_horiz: WienerTaps::IDENTITY,
                wiener_vert: WienerTaps::IDENTITY,
                ..Default::default()
            },
            10,
        );
        assert_eq!(pix, orig);
    }
}
