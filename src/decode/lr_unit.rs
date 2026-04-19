//! Loop-restoration per-unit syntax decode — spec §5.11.40-.44.
//!
//! AV1 sign​als per-restoration-unit filter choice + coefficients in
//! the range-coded tile payload. This module implements that decode
//! so the tile decoder can feed real per-unit parameters into the
//! [`crate::lr`] filter primitives instead of the goavif "default
//! params" shortcut (which effectively disabled Wiener / SGR tuning).
//!
//! Call [`decode_lr_unit`] once per plane×unit at the 64/128/256-aligned
//! super-block boundary that marks the start of each restoration unit
//! (spec §5.11.40 `lr_unit_info()`).
//!
//! CDF source: `libaom/av1/common/entropymode.c` —
//! `default_{switchable,wiener,sgrproj}_restore_cdf`. Constants
//! (`WIENER_FILT_TAP*_{MINV,MAXV,SUBEXP_K}`, `SGRPROJ_PRJ_*`) mirror
//! `libaom/av1/common/restoration.h`.

use crate::cdfs::{
    DEFAULT_SGRPROJ_RESTORE_CDF, DEFAULT_SWITCHABLE_RESTORE_CDF, DEFAULT_WIENER_RESTORE_CDF,
};
use crate::frame_header_tail::{
    RESTORATION_NONE, RESTORATION_SGR, RESTORATION_SWITCHABLE, RESTORATION_WIENER,
};
use crate::lr::{FilterType, SgrParams, UnitParams, WienerTaps};
use crate::symbol::SymbolDecoder;
use oxideav_core::{Error, Result};

/// Wiener tap centering / bit-width constants — spec §5.11.42.
const WIENER_TAP0_MIDV: i32 = 3;
const WIENER_TAP1_MIDV: i32 = -7;
const WIENER_TAP2_MIDV: i32 = 15;
const WIENER_TAP0_BITS: i32 = 4;
const WIENER_TAP1_BITS: i32 = 5;
const WIENER_TAP2_BITS: i32 = 6;
const WIENER_TAP0_SUBEXP_K: u16 = 1;
const WIENER_TAP1_SUBEXP_K: u16 = 2;
const WIENER_TAP2_SUBEXP_K: u16 = 3;

/// SGR projection constants — spec §5.11.44.
const SGRPROJ_PARAMS_BITS: u32 = 4;
const SGRPROJ_PRJ_BITS: i32 = 7;
const SGRPROJ_PRJ_SUBEXP_K: u16 = 4;

fn wiener_tap_min(bits: i32, midv: i32) -> i32 {
    midv - (1 << bits) / 2
}

fn wiener_tap_max(bits: i32, midv: i32) -> i32 {
    midv - 1 + (1 << bits) / 2
}

const fn sgrproj_prj_min0() -> i32 {
    -(1 << SGRPROJ_PRJ_BITS) * 3 / 4
}

const fn sgrproj_prj_min1() -> i32 {
    -(1 << SGRPROJ_PRJ_BITS) / 4
}

const fn sgrproj_prj_max0() -> i32 {
    sgrproj_prj_min0() + (1 << SGRPROJ_PRJ_BITS) - 1
}

const fn sgrproj_prj_max1() -> i32 {
    sgrproj_prj_min1() + (1 << SGRPROJ_PRJ_BITS) - 1
}

/// 16-entry `(r0, r1, eps0, eps1)` lookup — from `libaom/av1/common/restoration.c`
/// `av1_sgr_params`. `-1` in an eps slot means the corresponding
/// sub-filter is disabled.
pub const SGR_PARAMS: [(i32, i32, i32, i32); 16] = [
    (2, 1, 140, 3236),
    (2, 1, 112, 2158),
    (2, 1, 93, 1618),
    (2, 1, 80, 1438),
    (2, 1, 70, 1295),
    (2, 1, 58, 1177),
    (2, 1, 47, 1079),
    (2, 1, 37, 996),
    (2, 1, 30, 925),
    (2, 1, 25, 863),
    (0, 1, -1, 2589),
    (0, 1, -1, 1618),
    (0, 1, -1, 1177),
    (0, 1, -1, 925),
    (2, 0, 56, -1),
    (2, 0, 22, -1),
];

/// Reference Wiener / SGR values carried across restoration units
/// within a single plane. Matches libaom's `WienerInfo*` /
/// `SgrprojInfo*` references used by `aom_read_primitive_refsubexpfin`.
#[derive(Clone, Copy, Debug)]
pub struct LrRef {
    /// Reference Wiener filter — 3 tap deltas (corresponding to
    /// `wiener_info->vfilter / hfilter` at positions 0..=2), per axis.
    pub wiener_v: [i32; 3],
    pub wiener_h: [i32; 3],
    /// Reference SGR `xqd[0,1]`.
    pub sgr_xqd: [i32; 2],
}

impl Default for LrRef {
    fn default() -> Self {
        Self {
            wiener_v: [WIENER_TAP0_MIDV, WIENER_TAP1_MIDV, WIENER_TAP2_MIDV],
            wiener_h: [WIENER_TAP0_MIDV, WIENER_TAP1_MIDV, WIENER_TAP2_MIDV],
            sgr_xqd: [(1 << (SGRPROJ_PRJ_BITS - 1)), 0],
        }
    }
}

/// Decode `n` uncompressed bits from the range-coded payload —
/// spec's `L(n)`. Wrapper around [`SymbolDecoder::read_literal`] for
/// readability.
#[inline]
fn read_literal(sd: &mut SymbolDecoder<'_>, n: u32) -> u32 {
    sd.read_literal(n)
}

/// Spec §5.11.41 `decode_uniform` — read `n - m` bits then optionally
/// one extra bit.
fn decode_uniform(sd: &mut SymbolDecoder<'_>, n: u16) -> u16 {
    if n <= 1 {
        return 0;
    }
    let l = (15 - (n as u32).leading_zeros()) as i32 + 1;
    let m = (1i32 << l) - n as i32;
    let v = read_literal(sd, (l - 1) as u32) as i32;
    if v < m {
        v as u16
    } else {
        ((v << 1) - m + read_literal(sd, 1) as i32) as u16
    }
}

/// Spec §5.11.41 `decode_subexp` — finite subexponential code for a
/// symbol in `[0, n - 1]` with parameter `k`.
fn decode_subexp(sd: &mut SymbolDecoder<'_>, n: u16, k: u16) -> u16 {
    let mut i = 0u32;
    let mut mk = 0u16;
    loop {
        let b = if i == 0 { k } else { k + i as u16 - 1 };
        let a = 1u16 << b;
        if n <= mk.saturating_add(3u16.saturating_mul(a)) {
            return decode_uniform(sd, n - mk) + mk;
        }
        if read_literal(sd, 1) == 0 {
            return read_literal(sd, b as u32) as u16 + mk;
        }
        i += 1;
        mk += a;
    }
}

fn inv_recenter_nonneg(r: u16, v: u16) -> u16 {
    if v > (r << 1) {
        v
    } else if (v & 1) == 0 {
        (v >> 1) + r
    } else {
        r - ((v + 1) >> 1)
    }
}

fn inv_recenter_finite_nonneg(n: u16, r: u16, v: u16) -> u16 {
    if (r << 1) <= n {
        inv_recenter_nonneg(r, v)
    } else {
        n - 1 - inv_recenter_nonneg(n - 1 - r, v)
    }
}

/// Spec §5.11.42 `decode_signed_subexp_with_ref_bool` / libaom's
/// `aom_read_primitive_refsubexpfin`.
fn decode_signed_subexp_with_ref(
    sd: &mut SymbolDecoder<'_>,
    low: i32,
    high: i32,
    r: i32,
    k: u16,
) -> i32 {
    let n = (high - low + 1) as u16;
    let ref_ = (r - low).clamp(0, (n - 1) as i32) as u16;
    let raw = decode_subexp(sd, n, k);
    let v = inv_recenter_finite_nonneg(n, ref_, raw);
    low + v as i32
}

/// Derive runtime SGR `r0 / eps0` and `r1 / eps1` from the 16-entry
/// lookup keyed by `sgr_params_idx ∈ 0..=15`. Sub-filters with
/// `r == 0` are marked disabled by returning zero radius (the
/// caller's pipeline treats `r == 0` as pass-through).
pub fn sgr_params_for(idx: u32) -> (i32, i32, i32, i32) {
    let (r0, r1, e0, e1) = SGR_PARAMS[(idx as usize) & 0xF];
    (r0, r1, e0, e1)
}

/// Decode per-unit Wiener filter taps (§5.11.42).
///
/// `plane` selects tap 0 handling: `plane > 0` uses the 5-tap chroma
/// kernel, so `tap0` is forced to 0. Otherwise all three outer taps
/// are read.
fn read_wiener_filter(
    sd: &mut SymbolDecoder<'_>,
    plane: usize,
    ref_info: &mut LrRef,
) -> WienerTapsPair {
    let is_chroma = plane > 0;

    // Vertical axis.
    let v0 = if is_chroma {
        0
    } else {
        let lo = wiener_tap_min(WIENER_TAP0_BITS, WIENER_TAP0_MIDV);
        let hi = wiener_tap_max(WIENER_TAP0_BITS, WIENER_TAP0_MIDV);
        decode_signed_subexp_with_ref(sd, lo, hi, ref_info.wiener_v[0], WIENER_TAP0_SUBEXP_K)
    };
    let v1 = {
        let lo = wiener_tap_min(WIENER_TAP1_BITS, WIENER_TAP1_MIDV);
        let hi = wiener_tap_max(WIENER_TAP1_BITS, WIENER_TAP1_MIDV);
        decode_signed_subexp_with_ref(sd, lo, hi, ref_info.wiener_v[1], WIENER_TAP1_SUBEXP_K)
    };
    let v2 = {
        let lo = wiener_tap_min(WIENER_TAP2_BITS, WIENER_TAP2_MIDV);
        let hi = wiener_tap_max(WIENER_TAP2_BITS, WIENER_TAP2_MIDV);
        decode_signed_subexp_with_ref(sd, lo, hi, ref_info.wiener_v[2], WIENER_TAP2_SUBEXP_K)
    };

    // Horizontal axis.
    let h0 = if is_chroma {
        0
    } else {
        let lo = wiener_tap_min(WIENER_TAP0_BITS, WIENER_TAP0_MIDV);
        let hi = wiener_tap_max(WIENER_TAP0_BITS, WIENER_TAP0_MIDV);
        decode_signed_subexp_with_ref(sd, lo, hi, ref_info.wiener_h[0], WIENER_TAP0_SUBEXP_K)
    };
    let h1 = {
        let lo = wiener_tap_min(WIENER_TAP1_BITS, WIENER_TAP1_MIDV);
        let hi = wiener_tap_max(WIENER_TAP1_BITS, WIENER_TAP1_MIDV);
        decode_signed_subexp_with_ref(sd, lo, hi, ref_info.wiener_h[1], WIENER_TAP1_SUBEXP_K)
    };
    let h2 = {
        let lo = wiener_tap_min(WIENER_TAP2_BITS, WIENER_TAP2_MIDV);
        let hi = wiener_tap_max(WIENER_TAP2_BITS, WIENER_TAP2_MIDV);
        decode_signed_subexp_with_ref(sd, lo, hi, ref_info.wiener_h[2], WIENER_TAP2_SUBEXP_K)
    };

    ref_info.wiener_v = [v0, v1, v2];
    ref_info.wiener_h = [h0, h1, h2];

    // Compose the 4-element [`WienerTaps`] tuple used by the LR filter
    // primitives: `[taps0, taps1, taps2, taps3]`. The spec derives
    // `taps3` (center tap) so the full 7-element kernel sums to 128
    // (`WIENER_FILT_STEP`): `taps3 = 128 - 2*(taps0 + taps1 + taps2)`.
    let vcenter = 128 - 2 * (v0 + v1 + v2);
    let hcenter = 128 - 2 * (h0 + h1 + h2);
    WienerTapsPair {
        vertical: WienerTaps([v0, v1, v2, vcenter]),
        horizontal: WienerTaps([h0, h1, h2, hcenter]),
    }
}

/// A (vertical, horizontal) pair of [`WienerTaps`].
struct WienerTapsPair {
    vertical: WienerTaps,
    horizontal: WienerTaps,
}

/// Decode per-unit SGR filter parameters (§5.11.44).
fn read_sgrproj_filter(sd: &mut SymbolDecoder<'_>, ref_info: &mut LrRef) -> SgrParams {
    let ep = read_literal(sd, SGRPROJ_PARAMS_BITS);
    let (r0, r1, eps0, eps1) = sgr_params_for(ep);

    let min0 = sgrproj_prj_min0();
    let max0 = sgrproj_prj_max0();
    let min1 = sgrproj_prj_min1();
    let max1 = sgrproj_prj_max1();

    let (xq0, xq1) = if r0 == 0 {
        let xq0 = 0;
        let xq1 =
            decode_signed_subexp_with_ref(sd, min1, max1, ref_info.sgr_xqd[1], SGRPROJ_PRJ_SUBEXP_K);
        (xq0, xq1)
    } else if r1 == 0 {
        let xq0 =
            decode_signed_subexp_with_ref(sd, min0, max0, ref_info.sgr_xqd[0], SGRPROJ_PRJ_SUBEXP_K);
        let xq1 = ((1 << SGRPROJ_PRJ_BITS) - xq0).clamp(min1, max1);
        (xq0, xq1)
    } else {
        let xq0 =
            decode_signed_subexp_with_ref(sd, min0, max0, ref_info.sgr_xqd[0], SGRPROJ_PRJ_SUBEXP_K);
        let xq1 =
            decode_signed_subexp_with_ref(sd, min1, max1, ref_info.sgr_xqd[1], SGRPROJ_PRJ_SUBEXP_K);
        (xq0, xq1)
    };

    ref_info.sgr_xqd = [xq0, xq1];

    SgrParams {
        r0,
        r1,
        eps0,
        eps1,
        xq: [xq0, xq1],
    }
}

/// Decode one restoration unit's syntax — spec §5.11.40
/// `lr_unit_info()`. `frame_restoration_type` is the per-plane
/// `FrameRestorationType` carried in the frame header (§5.9.20).
///
/// CDFs are borrowed mutably so the caller can initialise them to
/// defaults at frame start and the decoder can adapt them in place.
pub fn decode_lr_unit(
    sd: &mut SymbolDecoder<'_>,
    frame_restoration_type: u8,
    plane: usize,
    ref_info: &mut LrRef,
    switchable_cdf: &mut [u16],
    wiener_cdf: &mut [u16],
    sgrproj_cdf: &mut [u16],
) -> Result<UnitParams> {
    match frame_restoration_type {
        RESTORATION_NONE => Ok(UnitParams::default()),
        RESTORATION_SWITCHABLE => {
            let kind = sd.decode_symbol(switchable_cdf)?;
            match kind {
                0 => Ok(UnitParams::default()),
                1 => {
                    let taps = read_wiener_filter(sd, plane, ref_info);
                    Ok(UnitParams {
                        filter: FilterType::Wiener,
                        wiener_horiz: taps.horizontal,
                        wiener_vert: taps.vertical,
                        sgr: SgrParams::default(),
                    })
                }
                2 => {
                    let sgr = read_sgrproj_filter(sd, ref_info);
                    Ok(UnitParams {
                        filter: FilterType::Sgr,
                        wiener_horiz: WienerTaps::default(),
                        wiener_vert: WienerTaps::default(),
                        sgr,
                    })
                }
                _ => Err(Error::invalid(format!(
                    "av1 lr_unit_info: invalid switchable type {kind} (§5.11.40)"
                ))),
            }
        }
        RESTORATION_WIENER => {
            let use_wiener = sd.decode_symbol(wiener_cdf)?;
            if use_wiener != 0 {
                let taps = read_wiener_filter(sd, plane, ref_info);
                Ok(UnitParams {
                    filter: FilterType::Wiener,
                    wiener_horiz: taps.horizontal,
                    wiener_vert: taps.vertical,
                    sgr: SgrParams::default(),
                })
            } else {
                Ok(UnitParams::default())
            }
        }
        RESTORATION_SGR => {
            let use_sgr = sd.decode_symbol(sgrproj_cdf)?;
            if use_sgr != 0 {
                let sgr = read_sgrproj_filter(sd, ref_info);
                Ok(UnitParams {
                    filter: FilterType::Sgr,
                    wiener_horiz: WienerTaps::default(),
                    wiener_vert: WienerTaps::default(),
                    sgr,
                })
            } else {
                Ok(UnitParams::default())
            }
        }
        other => Err(Error::invalid(format!(
            "av1 lr_unit_info: unknown restoration type {other} (§5.9.20)"
        ))),
    }
}

/// Start-of-frame default CDFs — copied owned. Matches the lifetime
/// rules used by the tile-level CDF bank in [`crate::decode::tile`].
pub fn default_switchable_cdf() -> Vec<u16> {
    DEFAULT_SWITCHABLE_RESTORE_CDF.to_vec()
}

pub fn default_wiener_cdf() -> Vec<u16> {
    DEFAULT_WIENER_RESTORE_CDF.to_vec()
}

pub fn default_sgrproj_cdf() -> Vec<u16> {
    DEFAULT_SGRPROJ_RESTORE_CDF.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_ref_defaults() {
        let r = LrRef::default();
        assert_eq!(r.wiener_v, [3, -7, 15]);
        assert_eq!(r.wiener_h, [3, -7, 15]);
        assert_eq!(r.sgr_xqd, [64, 0]);
    }

    #[test]
    fn sgr_params_lookup() {
        assert_eq!(sgr_params_for(0), (2, 1, 140, 3236));
        assert_eq!(sgr_params_for(10), (0, 1, -1, 2589));
        assert_eq!(sgr_params_for(14), (2, 0, 56, -1));
        // Wrap around.
        assert_eq!(sgr_params_for(16), SGR_PARAMS[0]);
    }

    #[test]
    fn inv_recenter_finite_nonneg_is_bijection_on_small_n() {
        let n = 12;
        for r in 0..n {
            let mut seen = std::collections::HashSet::new();
            for v in 0..n {
                let inv = inv_recenter_finite_nonneg(n, r, v);
                assert!(inv < n);
                seen.insert(inv);
            }
            assert_eq!(seen.len(), n as usize);
        }
    }

    #[test]
    fn wiener_tap_minv_maxv_span() {
        assert_eq!(wiener_tap_min(WIENER_TAP0_BITS, WIENER_TAP0_MIDV), -5);
        assert_eq!(wiener_tap_max(WIENER_TAP0_BITS, WIENER_TAP0_MIDV), 10);
        assert_eq!(wiener_tap_min(WIENER_TAP1_BITS, WIENER_TAP1_MIDV), -23);
        assert_eq!(wiener_tap_max(WIENER_TAP1_BITS, WIENER_TAP1_MIDV), 8);
        assert_eq!(wiener_tap_min(WIENER_TAP2_BITS, WIENER_TAP2_MIDV), -17);
        assert_eq!(wiener_tap_max(WIENER_TAP2_BITS, WIENER_TAP2_MIDV), 46);
    }

    #[test]
    fn sgrproj_prj_bounds() {
        assert_eq!(sgrproj_prj_min0(), -96);
        assert_eq!(sgrproj_prj_max0(), 31);
        assert_eq!(sgrproj_prj_min1(), -32);
        assert_eq!(sgrproj_prj_max1(), 95);
    }

    #[test]
    fn decode_lr_unit_returns_none_when_frame_is_none() {
        // RESTORATION_NONE bypasses all bitstream reads — the decoder
        // never consumes from `sd`. Pass a minimal non-empty payload.
        let data = [0u8; 4];
        let mut sd = SymbolDecoder::new(&data, data.len(), false).expect("sd");
        let mut refs = LrRef::default();
        let mut sw = default_switchable_cdf();
        let mut we = default_wiener_cdf();
        let mut sg = default_sgrproj_cdf();
        let got = decode_lr_unit(&mut sd, RESTORATION_NONE, 0, &mut refs, &mut sw, &mut we, &mut sg)
            .expect("decode");
        assert_eq!(got.filter, FilterType::None);
    }
}
