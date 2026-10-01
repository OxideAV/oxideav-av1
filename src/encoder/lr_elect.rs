//! r429 — encoder-side §5.9.20 / §5.11.57 / §7.17 loop-restoration
//! election (encoder ladder item 4).
//!
//! Loop restoration is the LAST in-loop stage (§7.4 order: deblock →
//! CDEF → superres → LR; this encoder codes deblock level 0, so on a
//! flat-width frame LR input is exactly the post-CDEF
//! reconstruction). Per restoration unit the bitstream carries a
//! filter selection and (for Wiener / self-guided) its coefficients;
//! the decoder filters `UpscaledCdefFrame` into `LrFrame`, which
//! becomes the §7.20 reference store.
//!
//! r444 — on a `use_superres = 1` frame ([`LrElectInput::
//! use_superres`]) the election runs at the UPSCALED extent, exactly
//! where §7.17 operates: the caller feeds the §7.16-upscaled
//! pre-CDEF and post-CDEF planes plus the ORIGINAL (full-width)
//! source as the fit target, and the §5.11.57 write window maps
//! superblock columns through the §5.9.8 `SuperresDenom` ratio.
//!
//! ## Election structure
//!
//! Per plane, per 64×64-sample unit (raster order):
//!
//! * **Wiener** — a separable 7-tap symmetric filter with 3
//!   transmitted taps per pass. The fit is free encoder engineering:
//!   alternating least squares (fit the horizontal taps against the
//!   source through the current vertical taps, then vice versa) in
//!   f64, quantised to the §5.11.58 `Wiener_Taps_Min/Max` ranges —
//!   then EVALUATED exactly through the decoder's own
//!   [`crate::loop_restoration::loop_restore_block`] Wiener kernel.
//! * **Self-guided** — for each of the 16 §7.17.3 `Sgr_Params` sets,
//!   the two projection weights are fitted by least squares on the
//!   EXACT per-pixel filter bases: probe runs at `xqd = (128, 0)` /
//!   `(0, 128)` recover `flt0 - dgd` / `flt1 - dgd` exactly
//!   (`(128·a + 64) >> 7 = a`, modulo the output pixel clip), the
//!   2×2 normal equations solve for the weights, radius-0 sets pin
//!   the §5.11.58 derived component — then the quantised candidate
//!   is evaluated exactly.
//! * The unit elects `argmin D + λ·R` over { none, best Wiener, best
//!   self-guided }, `R` priced by running the §5.11.58 writer
//!   ([`super::loop_restoration_write::write_lr_unit`]) against a
//!   counting [`SymbolWriter`] with the running subexp reference
//!   state — the recentred-subexp coefficient costs are exact bits.
//!
//! Per plane, the §5.9.20 `FrameRestorationType` collapses to NONE /
//! WIENER / SGRPROJ when the elected units are uniform, else
//! SWITCHABLE. The CALLER re-emits the tile with the §5.11.57
//! `write_lr` interleave (the LR symbols live inside the tile) and
//! settles LR-on vs LR-off on EXACT realized bytes, then applies the
//! plan through the decoder's own §7.17 frame driver so the stored
//! reference planes equal the decoder's byte-for-byte.
//!
//! Unit-size scope (r429; r460 elects `lr_unit_shift` 0 / 1 / 2 from the caller): `lr_uv_shift = 0` —
//! 64×64-sample units on every plane (the finest §5.9.20 grid; the
//! size election is left open).

use crate::cdf::TileCdfContext;
use crate::cdf::{LrUnit, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER};
use crate::encoder::loop_restoration_write::{write_lr_unit, LrWriteState};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::yuv_frame::YuvFrame;
use crate::loop_filter::PlaneBuffer;
use crate::loop_restoration::{
    box_filter, count_units_in_frame, loop_restore_rect, sgr_project, stripe_unit_rects,
    LoopRestorationFrameContext, LrBlockGeometry, SGRPROJ_RST_BITS, SGRPROJ_XQD_MAX,
    SGRPROJ_XQD_MIN, SGR_PARAMS, WIENER_COEFFS, WIENER_TAPS_MAX, WIENER_TAPS_MID, WIENER_TAPS_MIN,
};
use crate::uncompressed_header_tail::{FrameRestorationType, LrParams as HeaderLrParams};

/// The elected loop-restoration configuration.
pub(crate) struct LrPlan {
    /// §5.9.20 header block (goes into `fh.lr_params`).
    pub header: HeaderLrParams,
    /// The §5.11.57 write-side parameter bundle for
    /// [`super::loop_restoration_write::write_lr`].
    pub write_params: crate::cdf::LrParams,
    /// Every unit of every ACTIVE plane, keyed `(plane, unitRow,
    /// unitCol)` — including `RESTORE_NONE` units (the §5.11.57
    /// window fires `read_lr_unit` for each of them).
    pub units: Vec<((usize, u32, u32), LrUnit)>,
    /// Whole-frame SSD vs the source AFTER the plan is applied
    /// (exact — the §7.17 per-unit outputs are disjoint and depend
    /// only on their own unit's coefficients).
    pub d: u64,
    /// Whole-frame SSD vs the source BEFORE LR (the post-CDEF
    /// reconstruction) — the no-LR arm of the caller's exact-bytes
    /// settlement.
    pub d_pre: u64,
}

/// Election inputs (see [`elect_lr`]).
pub(crate) struct LrElectInput<'a> {
    pub input: &'a YuvFrame,
    /// Pre-CDEF reconstruction (§7.17 reads `CurrFrame` across stripe
    /// boundaries) — whole planes or their stripe-boundary rows.
    pub curr_y: LrCurr<'a>,
    pub curr_u: LrCurr<'a>,
    pub curr_v: LrCurr<'a>,
    /// Post-CDEF reconstruction (the LR input planes).
    pub cdef_y: &'a [u16],
    pub cdef_u: &'a [u16],
    pub cdef_v: &'a [u16],
    pub width: usize,
    pub height: usize,
    pub chroma_w: usize,
    pub chroma_h: usize,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub num_planes: u8,
    pub mi_rows: u32,
    pub mi_cols: u32,
    /// λ on the 1/256-bit `score256` convention.
    pub lambda: u64,
    /// CDF state for the §5.11.58 selection-symbol pricing (the
    /// frame-start state; the exact position-dependent cost is
    /// settled by the caller's re-emission).
    pub price_cdfs: &'a TileCdfContext,
    pub disable_cdf_update: bool,
    /// r444 — §5.9.8 pairing: `true` when this frame codes
    /// `use_superres = 1`. The election then operates at the
    /// UPSCALED extent (`width` / `chroma_w` and every plane slice
    /// are the §7.16 outputs; `mi_rows` / `mi_cols` stay the CODED
    /// grid), and the §5.11.57 write-side window rides the
    /// superres column mapping through
    /// [`crate::cdf::LrParams::use_superres`].
    /// r460 — the CODED frame extent (`UpscaledWidth` / `FrameHeight`,
    /// the §7.17 unit-grid and stripe geometry) when it differs from
    /// the plane buffers' extent (`width` / `height` — the §5.9.5 mi
    /// grid a non-multiple-of-8 picture is padded to).
    pub frame_width: usize,
    /// See [`Self::frame_width`].
    pub frame_height: usize,
    /// r460 — every `sgr_step`-th §7.17 self-guided set is trialled
    /// (`1` = all).
    pub sgr_step: usize,
    /// r460 — §5.9.20 `lr_unit_shift` (0 / 1 / 2 → 64 / 128 / 256 px
    /// units; under 128×128 superblocks the header codes 1 or 2
    /// only). Larger units amortise the per-unit Wiener / SGR
    /// signalling over more samples.
    pub unit_shift: u8,
    pub use_superres: bool,
    /// The §5.9.8 `SuperresDenom` (`SUPERRES_NUM` when
    /// `use_superres` is `false`).
    pub superres_denom: u32,
    /// r464 — worker threads for the per-unit distortion search
    /// (`1` = sequential; the election is identical either way).
    pub threads: usize,
    /// r464 — alternating-least-squares rounds of the Wiener fit.
    pub wiener_rounds: u8,
}

/// r464 — the pre-CDEF samples §7.17 reads across stripe boundaries
/// (§7.17.6 routes `y < StripeStartY` / `y > StripeEndY` to
/// `UpscaledCurrFrame`, at most 2 rows past the stripe edge for the
/// box filter and 3 for the Wiener taps), kept as the 4 rows above and
/// 4 rows below every stripe boundary instead of a frame-sized copy
/// of the pre-CDEF reconstruction (36 MB at 12 MP; 4.5 MB here).
pub(crate) struct StripeRows {
    width: usize,
    /// `(first plane row, offset into `rows`)` per stored band.
    bands: Vec<(usize, usize)>,
    rows: Vec<u16>,
    band_h: usize,
}

impl StripeRows {
    /// Capture the boundary rows of `plane` (`width × height`,
    /// `sub_y` the plane's vertical subsampling).
    pub(crate) fn capture(plane: &[u16], width: usize, height: usize, sub_y: u8) -> Self {
        let band_h = 8usize;
        let mut bands = Vec::new();
        let mut rows = Vec::new();
        let mut n = 1usize;
        loop {
            let boundary = (64 * n - 8) >> sub_y;
            if boundary >= height {
                break;
            }
            let y0 = boundary.saturating_sub(band_h / 2);
            let y1 = (boundary + band_h / 2).min(height);
            bands.push((y0, rows.len()));
            rows.extend_from_slice(&plane[y0 * width..y1 * width]);
            // Keep every band `band_h` rows long in the index: a
            // clipped bottom band is padded by repeating its last row.
            for _ in y1..y0 + band_h {
                let last = rows.len() - width;
                rows.extend_from_within(last..);
            }
            n += 1;
        }
        Self {
            width,
            bands,
            rows,
            band_h,
        }
    }

    /// The stored sample at `(y, x)`, `None` outside the kept rows.
    #[inline]
    fn get(&self, y: usize, x: usize) -> Option<u16> {
        // Bands are sorted and disjoint (64-row pitch, 8-row bands).
        let i = self.bands.partition_point(|&(y0, _)| y0 + self.band_h <= y);
        let &(y0, off) = self.bands.get(i)?;
        if y < y0 {
            return None;
        }
        Some(self.rows[off + (y - y0) * self.width + x])
    }
}

/// A pre-CDEF plane for the §7.17 election / application: the whole
/// plane, or just its stripe-boundary rows ([`StripeRows`]).
#[derive(Clone, Copy)]
pub(crate) enum LrCurr<'a> {
    Full(&'a [u16]),
    Rows(&'a StripeRows),
}

impl LrCurr<'_> {
    /// The pre-CDEF sample at `(y, x)` (`stride` = the plane width),
    /// or `fallback` where only boundary rows are kept — those rows
    /// are the only ones §7.17 ever reads from this plane.
    #[inline]
    fn at(&self, y: usize, x: usize, stride: usize, fallback: u16) -> u16 {
        match self {
            LrCurr::Full(p) => p[y * stride + x],
            LrCurr::Rows(r) => r.get(y, x).unwrap_or(fallback),
        }
    }
}

/// r464 — one (stripe, unit) rectangle of a plane lifted into a
/// LOCAL window: the pre-CDEF (`curr`) and post-CDEF (`cdef`) samples
/// of the rectangle plus a [`LOCAL_MARGIN`]-sample apron on every
/// side (edge-replicated past the plane, exactly what the §7.17.6
/// clamp returns), with the §7.17.1 geometry translated into window
/// coordinates. Every kernel reads at most 3 samples past the
/// rectangle (Wiener taps / box radius 2 + the A/B apron), so the
/// window is self-contained and the filters evaluate sample-exact
/// without a frame-sized `i32` copy of any plane.
struct LocalRect {
    /// Geometry in window coordinates (`x = y = LOCAL_MARGIN`).
    geom: LrBlockGeometry,
    /// Rectangle origin in plane coordinates.
    abs_x: usize,
    abs_y: usize,
    rows: usize,
    cols: usize,
    curr: Vec<i32>,
    cdef: Vec<i32>,
}

const LOCAL_MARGIN: usize = 4;

impl LocalRect {
    fn build(
        geom: &LrBlockGeometry,
        curr: LrCurr<'_>,
        cdef: &[u16],
        plane_w: usize,
        plane_h: usize,
    ) -> Self {
        let m = LOCAL_MARGIN;
        let (w, h) = (geom.w as usize, geom.h as usize);
        let (rows, cols) = (h + 2 * m, w + 2 * m);
        let (abs_x, abs_y) = (geom.x as usize, geom.y as usize);
        let mut lc = vec![0i32; rows * cols];
        let mut ld = vec![0i32; rows * cols];
        for ly in 0..rows {
            let ay = (abs_y as i64 + ly as i64 - m as i64).clamp(0, plane_h as i64 - 1) as usize;
            for lx in 0..cols {
                let ax =
                    (abs_x as i64 + lx as i64 - m as i64).clamp(0, plane_w as i64 - 1) as usize;
                let cd = cdef[ay * plane_w + ax];
                lc[ly * cols + lx] = i32::from(curr.at(ay, ax, plane_w, cd));
                ld[ly * cols + lx] = i32::from(cd);
            }
        }
        let (ox, oy) = (abs_x as i32 - m as i32, abs_y as i32 - m as i32);
        Self {
            geom: LrBlockGeometry {
                unit_row: geom.unit_row,
                unit_col: geom.unit_col,
                x: m as u32,
                y: m as u32,
                w: geom.w,
                h: geom.h,
                stripe_start_y: geom.stripe_start_y - oy,
                stripe_end_y: geom.stripe_end_y - oy,
                plane_end_x: geom.plane_end_x - ox,
                plane_end_y: geom.plane_end_y - oy,
            },
            abs_x,
            abs_y,
            rows,
            cols,
            curr: lc,
            cdef: ld,
        }
    }

    fn bufs(&mut self) -> (PlaneBuffer<'_>, PlaneBuffer<'_>) {
        (
            PlaneBuffer {
                rows: self.rows as u32,
                cols: self.cols as u32,
                samples: &mut self.curr,
            },
            PlaneBuffer {
                rows: self.rows as u32,
                cols: self.cols as u32,
                samples: &mut self.cdef,
            },
        )
    }

    /// SSD of `out` (window layout) against the source over the
    /// rectangle.
    fn ssd(&self, out: &[i32], src: &[u16], src_stride: usize) -> u64 {
        let m = LOCAL_MARGIN;
        let mut ssd = 0u64;
        for i in 0..self.geom.h as usize {
            let orow =
                &out[(m + i) * self.cols + m..(m + i) * self.cols + m + self.geom.w as usize];
            let srow = &src[(self.abs_y + i) * src_stride + self.abs_x..];
            for (o, s) in orow.iter().zip(srow) {
                let d = i64::from(*o) - i64::from(*s);
                ssd += (d * d) as u64;
            }
        }
        ssd
    }
}

/// Run one restoration candidate over a local rectangle into `out`
/// (window layout; samples outside the rectangle are untouched) and
/// return the rectangle's SSD vs the source.
fn eval_rect(
    lr: &mut LocalRect,
    unit: &LrUnit,
    plane: usize,
    bit_depth: u8,
    out: &mut Vec<i32>,
    src: &[u16],
    src_stride: usize,
) -> u64 {
    out.clear();
    out.extend_from_slice(&lr.cdef);
    if unit.restoration_type != RESTORE_NONE {
        let rt = match unit.restoration_type {
            RESTORE_WIENER => FrameRestorationType::Wiener,
            RESTORE_SGRPROJ => FrameRestorationType::SgrProj,
            _ => FrameRestorationType::None,
        };
        let geom = lr.geom;
        let (rows, cols) = (lr.rows as u32, lr.cols as u32);
        let wiener = unit.wiener;
        let sgr_set = unit.sgr_set;
        let sgr_xqd = unit.sgr_xqd;
        let lrp = header_shape([FrameRestorationType::Switchable; 3], 0);
        let ctx = LoopRestorationFrameContext {
            mi_rows: 0,
            mi_cols: 0,
            num_planes: 1,
            bit_depth,
            subsampling_x: 0,
            subsampling_y: 0,
            frame_height: rows,
            upscaled_width: cols,
            lr_params: &lrp,
            lr_type: &move |_, _, _| rt,
            lr_wiener: &move |_, _, _, pass, i| wiener[pass as usize][i],
            lr_sgr_set: &move |_, _, _| sgr_set as u8,
            lr_sgr_xqd: &move |_, _, _, i| sgr_xqd[i],
        };
        let (curr_b, cdef_b) = lr.bufs();
        let mut out_b = [PlaneBuffer {
            rows,
            cols,
            samples: out,
        }];
        // The window is presented as plane 0 (the kernels only ever
        // index the plane they are asked to restore; the plane's
        // subsampling is already folded into the geometry).
        let _ = plane;
        loop_restore_rect(&ctx, &[curr_b], &[cdef_b], &mut out_b, 0, &geom);
    }
    lr.ssd(out, src, src_stride)
}

/// Exact §7.17.2 self-guided output of `(set, xqd)` from
/// precomputed box-filter bases (window layout).
fn sgr_apply_bases(
    lr: &LocalRect,
    flt0: &[i32],
    flt1: &[i32],
    set: usize,
    xqd: [i32; 2],
    bit_depth: u8,
    out: &mut [i32],
) {
    let (r0, r1) = (SGR_PARAMS[set][0], SGR_PARAMS[set][2]);
    let m = LOCAL_MARGIN;
    let (w, h) = (lr.geom.w as usize, lr.geom.h as usize);
    for i in 0..h {
        for j in 0..w {
            let u = lr.cdef[(m + i) * lr.cols + m + j] << SGRPROJ_RST_BITS;
            out[(m + i) * lr.cols + m + j] = sgr_project(
                u,
                flt0[i * w + j],
                flt1[i * w + j],
                xqd[0],
                xqd[1],
                r0,
                r1,
                bit_depth,
            );
        }
    }
}

/// Box-filter bases `flt` (rectangle layout `h × w`) for one
/// `(r, eps, pass)` over a local rectangle.
fn sgr_bases(lr: &mut LocalRect, r: i32, eps: i32, pass: u8, bit_depth: u8) -> Vec<i32> {
    let geom = lr.geom;
    let (curr_b, cdef_b) = lr.bufs();
    box_filter(&[curr_b], &[cdef_b], 0, &geom, bit_depth, r, eps, pass)
}

/// Cached box-filter bases per `(r, eps, pass)`: one `h × w` table per
/// rectangle of the unit.
type BasesCache = Vec<((i32, i32, u8), Vec<Vec<i32>>)>;

/// The distortion side of one unit's search (no pricing): the
/// unfiltered SSD, the best Wiener fit and the best self-guided fit
/// with their exact SSDs.
struct UnitSearch {
    d_none: u64,
    wiener: (LrUnit, u64),
    sgr: Option<(LrUnit, u64)>,
}

/// r464 — the per-unit search: Wiener alternating least squares and,
/// per trialled §7.17.3 set, a least-squares projection fit on the
/// EXACT box-filter bases (`flt0 - flt1`, `u - flt1`) followed by the
/// exact kernel evaluation of the quantised weights. Independent of
/// every other unit, so the caller runs the units in parallel.
fn search_unit(
    rects: &mut [LocalRect],
    plane: usize,
    src: &[u16],
    src_stride: usize,
    bit_depth: u8,
    sgr_step: usize,
    wiener_rounds: u8,
) -> UnitSearch {
    let first_coeff = usize::from(plane != 0);
    let mut out: Vec<i32> = Vec::new();
    let mut d_none = 0u64;
    for lr in rects.iter_mut() {
        d_none += eval_rect(
            lr,
            &LrUnit::NONE,
            plane,
            bit_depth,
            &mut out,
            src,
            src_stride,
        );
    }
    // Wiener.
    let taps = fit_wiener(rects, src, src_stride, first_coeff, wiener_rounds);
    let wiener_unit = LrUnit {
        restoration_type: RESTORE_WIENER,
        wiener: taps,
        sgr_set: 0,
        sgr_xqd: [0; 2],
    };
    let mut d_wiener = 0u64;
    for lr in rects.iter_mut() {
        d_wiener += eval_rect(
            lr,
            &wiener_unit,
            plane,
            bit_depth,
            &mut out,
            src,
            src_stride,
        );
    }
    // Self-guided: bases per (r, eps, pass) are cached across the set
    // list (several sets share a pass).
    let mut best_sgr: Option<(LrUnit, u64)> = None;
    let mut cache: BasesCache = Vec::new();
    fn bases(
        cache: &mut BasesCache,
        rects: &mut [LocalRect],
        r: i32,
        eps: i32,
        pass: u8,
        bit_depth: u8,
    ) -> usize {
        if let Some(k) = cache.iter().position(|(key, _)| *key == (r, eps, pass)) {
            return k;
        }
        let v: Vec<Vec<i32>> = rects
            .iter_mut()
            .map(|lr| sgr_bases(lr, r, eps, pass, bit_depth))
            .collect();
        cache.push(((r, eps, pass), v));
        cache.len() - 1
    }
    for set in (0..SGR_PARAMS.len()).step_by(sgr_step.max(1)) {
        let params = SGR_PARAMS[set];
        let (r0, eps0, r1, eps1) = (params[0], params[1], params[2], params[3]);
        let k0 = bases(&mut cache, rects, r0, eps0, 0, bit_depth);
        let k1 = bases(&mut cache, rects, r1, eps1, 1, bit_depth);
        // Least squares: 128·((src << 4) - f1) ≈ w0·(f0 - f1) + w1·(u -
        // f1), with a zero-radius pass substituting `u` for its `f`.
        let (mut s00, mut s01, mut s11, mut b0, mut b1) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for (ri, lr) in rects.iter().enumerate() {
            let f0 = &cache[k0].1[ri];
            let f1 = &cache[k1].1[ri];
            let m = LOCAL_MARGIN;
            let (w, h) = (lr.geom.w as usize, lr.geom.h as usize);
            for i in 0..h {
                for j in 0..w {
                    let u = lr.cdef[(m + i) * lr.cols + m + j] << SGRPROJ_RST_BITS;
                    let a0 = if r0 != 0 { f0[i * w + j] } else { u };
                    let a1 = if r1 != 0 { f1[i * w + j] } else { u };
                    let s = i32::from(src[(lr.abs_y + i) * src_stride + lr.abs_x + j])
                        << SGRPROJ_RST_BITS;
                    let x0 = f64::from(a0 - a1);
                    let x1 = f64::from(u - a1);
                    let t = 128.0 * f64::from(s - a1);
                    s00 += x0 * x0;
                    s01 += x0 * x1;
                    s11 += x1 * x1;
                    b0 += x0 * t;
                    b1 += x1 * t;
                }
            }
        }
        let (mut xq0, mut xq1) = (0i32, 0i32);
        if r0 != 0 && r1 != 0 {
            let det = s00 * s11 - s01 * s01;
            if det.abs() > 1e-6 {
                xq0 = ((s11 * b0 - s01 * b1) / det).round() as i32;
                xq1 = ((s00 * b1 - s01 * b0) / det).round() as i32;
            }
        } else if r0 != 0 {
            // r1 == 0: `u - f1 == 0`, the projection is w0 alone.
            if s00 > 1e-6 {
                xq0 = (b0 / s00).round() as i32;
            }
        } else if r1 != 0 {
            // r0 == 0: `f0 - f1 == u - f1`, the projection is w1 alone
            // (§5.11.58 forces xqd[0] = 0).
            if s11 > 1e-6 {
                xq1 = (b1 / s11).round() as i32;
            }
        }
        // §5.11.58 constraints: clamp, derive radius-0 components.
        xq0 = xq0.clamp(SGRPROJ_XQD_MIN[0], SGRPROJ_XQD_MAX[0]);
        xq1 = xq1.clamp(SGRPROJ_XQD_MIN[1], SGRPROJ_XQD_MAX[1]);
        if r0 == 0 {
            xq0 = 0;
        }
        if r1 == 0 {
            xq1 = (128 - xq0).clamp(SGRPROJ_XQD_MIN[1], SGRPROJ_XQD_MAX[1]);
        }
        let cand = LrUnit {
            restoration_type: RESTORE_SGRPROJ,
            wiener: [[0; WIENER_COEFFS]; 2],
            sgr_set: set,
            sgr_xqd: [xq0, xq1],
        };
        let mut d = 0u64;
        for (ri, lr) in rects.iter().enumerate() {
            out.clear();
            out.extend_from_slice(&lr.cdef);
            sgr_apply_bases(
                lr,
                &cache[k0].1[ri],
                &cache[k1].1[ri],
                set,
                [xq0, xq1],
                bit_depth,
                &mut out,
            );
            d += lr.ssd(&out, src, src_stride);
        }
        if best_sgr.as_ref().map(|(_, bd)| d < *bd).unwrap_or(true) {
            best_sgr = Some((cand, d));
        }
    }
    UnitSearch {
        d_none,
        wiener: (wiener_unit, d_wiener),
        sgr: best_sgr,
    }
}

/// The §5.9.20 header block this election codes: `64 << unit_shift`
/// units on every plane (`lr_uv_shift = 0`), restoration types as
/// given.
fn header_shape(frt: [FrameRestorationType; 3], unit_shift: u8) -> HeaderLrParams {
    let uses_lr = frt.iter().any(|&t| t != FrameRestorationType::None);
    let uses_chroma_lr = frt[1..].iter().any(|&t| t != FrameRestorationType::None);
    let size = 64u32 << unit_shift.min(2);
    HeaderLrParams {
        frame_restoration_type: frt,
        uses_lr,
        uses_chroma_lr,
        lr_unit_shift: unit_shift.min(2),
        lr_uv_shift: 0,
        loop_restoration_size: if uses_lr { [size; 3] } else { [0, 0, 0] },
        short_circuited: false,
    }
}

/// Alternating-least-squares Wiener tap fit for one unit (free
/// encoder engineering; the exact §7.17.4 kernel evaluates the
/// quantised result). `first_coeff = 1` on chroma (`taps[pass][0]`
/// is forced 0 by §5.11.58). Reads the post-CDEF samples from the
/// unit's local windows (edge-replicated aprons).
fn fit_wiener(
    rects: &[LocalRect],
    src: &[u16],
    src_stride: usize,
    first_coeff: usize,
    rounds: u8,
) -> [[i32; WIENER_COEFFS]; 2] {
    let taps7 = |t: &[f64; 3]| -> [f64; 7] {
        let c = 128.0 - 2.0 * (t[0] + t[1] + t[2]);
        [t[0], t[1], t[2], c, t[2], t[1], t[0]]
    };
    let mut vt = [
        f64::from(WIENER_TAPS_MID[0]),
        f64::from(WIENER_TAPS_MID[1]),
        f64::from(WIENER_TAPS_MID[2]),
    ];
    let mut ht = vt;
    if first_coeff == 1 {
        vt[0] = 0.0;
        ht[0] = 0.0;
    }
    let m = LOCAL_MARGIN as i64;
    for _round in 0..rounds.max(1) {
        for dir in 0..2usize {
            // dir 0: fit horizontal (pass 1) through the vertical
            // taps; dir 1: fit vertical (pass 0) through the
            // horizontal taps.
            let fixed = taps7(if dir == 0 { &vt } else { &ht });
            let nvar = 3 - first_coeff;
            let mut ata = [[0f64; 3]; 3];
            let mut atb = [0f64; 3];
            for lr in rects {
                let cols = lr.cols as i64;
                let at = |x: i64, y: i64| -> f64 { f64::from(lr.cdef[(y * cols + x) as usize]) };
                // r464 — the fixed-direction pass is separable: filter
                // the window once along the fixed axis (`pre`, the
                // same per-sample sum in the same order as the
                // per-pixel evaluation it replaces — bit-identical
                // taps), then the 7 free-axis offsets read it.
                let (pw, ph) = (lr.cols, lr.rows);
                let mut pre = vec![0f64; pw * ph];
                for yy in 0..ph as i64 {
                    for xx in 0..pw as i64 {
                        let mut acc = 0f64;
                        for (k, fk) in fixed.iter().enumerate() {
                            let foff = k as i64 - 3;
                            let (sx, sy) = if dir == 0 {
                                (xx, yy + foff)
                            } else {
                                (xx + foff, yy)
                            };
                            if sx < 0 || sy < 0 || sx >= pw as i64 || sy >= ph as i64 {
                                continue;
                            }
                            acc += fk * at(sx, sy);
                        }
                        pre[(yy * pw as i64 + xx) as usize] = acc;
                    }
                }
                for y in 0..lr.geom.h as i64 {
                    for x in 0..lr.geom.w as i64 {
                        let (lx, ly) = (x + m, y + m);
                        let mut mm = [0f64; 7];
                        for (j, mj) in mm.iter_mut().enumerate() {
                            let off = j as i64 - 3;
                            let (sx, sy) = if dir == 0 {
                                (lx + off, ly)
                            } else {
                                (lx, ly + off)
                            };
                            *mj = pre[(sy * pw as i64 + sx) as usize] / 128.0;
                        }
                        let target = f64::from(
                            src[(lr.abs_y + y as usize) * src_stride + lr.abs_x + x as usize],
                        ) - mm[3];
                        let mut basis = [0f64; 3];
                        for (i, b) in basis.iter_mut().enumerate() {
                            *b = (mm[i] + mm[6 - i] - 2.0 * mm[3]) / 128.0;
                        }
                        for i in first_coeff..3 {
                            for j in first_coeff..3 {
                                ata[i][j] += basis[i] * basis[j];
                            }
                            atb[i] += basis[i] * target;
                        }
                    }
                }
            }
            // Tiny Gaussian elimination; keep the previous taps on a
            // singular fit.
            let mut a = [[0f64; 4]; 3];
            for i in 0..nvar {
                for j in 0..nvar {
                    a[i][j] = ata[first_coeff + i][first_coeff + j];
                }
                a[i][nvar] = atb[first_coeff + i];
            }
            let mut ok = true;
            for i in 0..nvar {
                let mut piv = i;
                for r in i + 1..nvar {
                    if a[r][i].abs() > a[piv][i].abs() {
                        piv = r;
                    }
                }
                a.swap(i, piv);
                if a[i][i].abs() < 1e-9 {
                    ok = false;
                    break;
                }
                for r in i + 1..nvar {
                    let f = a[r][i] / a[i][i];
                    #[allow(clippy::needless_range_loop)]
                    for c in i..=nvar {
                        a[r][c] -= f * a[i][c];
                    }
                }
            }
            if ok {
                let mut sol = [0f64; 3];
                for i in (0..nvar).rev() {
                    let mut v = a[i][nvar];
                    for j in i + 1..nvar {
                        v -= a[i][j] * sol[j];
                    }
                    sol[i] = v / a[i][i];
                }
                let out = if dir == 0 { &mut ht } else { &mut vt };
                for i in first_coeff..3 {
                    out[i] = sol[i - first_coeff]
                        .clamp(f64::from(WIENER_TAPS_MIN[i]), f64::from(WIENER_TAPS_MAX[i]));
                }
            }
        }
    }
    let quant = |t: &[f64; 3]| -> [i32; WIENER_COEFFS] {
        let mut q = [0i32; WIENER_COEFFS];
        for i in 0..WIENER_COEFFS {
            q[i] = (t[i].round() as i32).clamp(WIENER_TAPS_MIN[i], WIENER_TAPS_MAX[i]);
        }
        if first_coeff == 1 {
            q[0] = 0;
        }
        q
    };
    [quant(&vt), quant(&ht)]
}

/// The frame-level + per-unit loop-restoration election. Returns the
/// winning plan — NOT yet applied, NOT yet settled — or `None` when
/// no unit elected a filter. The caller re-emits the tile with the
/// §5.11.57 interleave, settles LR-on vs LR-off on exact realized
/// bytes, and applies via [`apply_lr_plan`].
///
/// r464 — the distortion side (Wiener fit, self-guided fits on the
/// exact box-filter bases, exact kernel SSDs) runs per unit on local
/// stripe-rectangle windows, `threads`-wide; the `D + λ·R` election
/// then walks the units in §5.11.57 order with the running subexp
/// reference state, exactly as before.
pub(crate) fn elect_lr(inp: &LrElectInput<'_>) -> Option<LrPlan> {
    let num_planes = inp.num_planes.min(3) as usize;
    let dims: Vec<(usize, usize)> = (0..num_planes)
        .map(|p| {
            if p == 0 {
                (inp.width, inp.height)
            } else {
                (inp.chroma_w, inp.chroma_h)
            }
        })
        .collect();
    // Eval-side header: every plane SWITCHABLE so the per-unit
    // closure decides (the real header collapses below).
    let eval_lrp = header_shape([FrameRestorationType::Switchable; 3], inp.unit_shift);
    let geom_ctx = LoopRestorationFrameContext {
        mi_rows: inp.mi_rows,
        mi_cols: inp.mi_cols,
        num_planes: num_planes as u8,
        bit_depth: inp.bit_depth,
        subsampling_x: inp.subsampling_x,
        subsampling_y: inp.subsampling_y,
        frame_height: inp.frame_height as u32,
        upscaled_width: inp.frame_width as u32,
        lr_params: &eval_lrp,
        lr_type: &|_, _, _| FrameRestorationType::None,
        lr_wiener: &|_, _, _, _, _| 0,
        lr_sgr_set: &|_, _, _| 0,
        lr_sgr_xqd: &|_, _, _, _| 0,
    };
    let currs: [LrCurr<'_>; 3] = [inp.curr_y, inp.curr_u, inp.curr_v];
    let cdefs: [&[u16]; 3] = [inp.cdef_y, inp.cdef_u, inp.cdef_v];
    let srcs: [&[u16]; 3] = [&inp.input.y, &inp.input.u, &inp.input.v];

    // Per plane: the unit grid + every unit's stripe rectangles (the
    // decoder's own §7.17.1 geometry, one rectangle per stripe).
    let mut per_plane_grid: Vec<(u32, u32)> = Vec::new();
    let mut tasks: Vec<(usize, u32, u32, Vec<LrBlockGeometry>)> = Vec::new();
    for plane in 0..num_planes {
        let (sub_x, sub_y) = if plane == 0 {
            (0u32, 0u32)
        } else {
            (u32::from(inp.subsampling_x), u32::from(inp.subsampling_y))
        };
        let unit_size = 64u32 << inp.unit_shift.min(2);
        let unit_rows = count_units_in_frame(unit_size, (inp.frame_height as u32 + sub_y) >> sub_y);
        let unit_cols = count_units_in_frame(unit_size, (inp.frame_width as u32 + sub_x) >> sub_x);
        per_plane_grid.push((unit_rows, unit_cols));
        let mut per_unit: Vec<Vec<LrBlockGeometry>> =
            (0..unit_rows * unit_cols).map(|_| Vec::new()).collect();
        for g in stripe_unit_rects(&geom_ctx, plane as u8) {
            per_unit[(g.unit_row * unit_cols + g.unit_col) as usize].push(g);
        }
        for ur in 0..unit_rows {
            for uc in 0..unit_cols {
                let rects = std::mem::take(&mut per_unit[(ur * unit_cols + uc) as usize]);
                tasks.push((plane, ur, uc, rects));
            }
        }
    }

    // Phase 1 — the distortion-side search, `threads`-wide over the
    // (plane, unit) list.
    let run = |t: &(usize, u32, u32, Vec<LrBlockGeometry>)| -> Option<UnitSearch> {
        let (plane, _, _, rects) = t;
        if rects.is_empty() {
            return None;
        }
        let (pw, ph) = dims[*plane];
        let mut local: Vec<LocalRect> = rects
            .iter()
            .map(|g| LocalRect::build(g, currs[*plane], cdefs[*plane], pw, ph))
            .collect();
        Some(search_unit(
            &mut local,
            *plane,
            srcs[*plane],
            pw,
            inp.bit_depth,
            inp.sgr_step,
            inp.wiener_rounds,
        ))
    };
    let threads = inp.threads.max(1).min(tasks.len().max(1));
    let searched: Vec<Option<UnitSearch>> = if threads <= 1 {
        tasks.iter().map(run).collect()
    } else {
        let next = core::sync::atomic::AtomicUsize::new(0);
        let mut slots: Vec<Option<UnitSearch>> = (0..tasks.len()).map(|_| None).collect();
        let results: Vec<Vec<(usize, Option<UnitSearch>)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let (next, tasks, run) = (&next, &tasks, &run);
                    scope.spawn(move || {
                        let mut outs = Vec::new();
                        loop {
                            let k = next.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
                            if k >= tasks.len() {
                                break;
                            }
                            outs.push((k, run(&tasks[k])));
                        }
                        outs
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("lr unit search thread panicked"))
                .collect()
        });
        for outs in results {
            for (k, o) in outs {
                slots[k] = o;
            }
        }
        slots
    };

    // Phase 2 — exact §5.11.58 pricing in unit order with the running
    // subexp reference state, `argmin D + λ·R` per unit.
    let price_unit = |state: &LrWriteState, plane: usize, unit: &LrUnit| -> u64 {
        let mut w = SymbolWriter::new_counting(inp.disable_cdf_update, 0x8000);
        let mut cdfs = inp.price_cdfs.clone();
        let mut st = state.clone();
        let _ = write_lr_unit(&mut w, &mut cdfs, &mut st, plane, RESTORE_SWITCHABLE, unit);
        w.cost_bits256()
    };
    let mut plan_units: Vec<((usize, u32, u32), LrUnit)> = Vec::new();
    let mut frt = [FrameRestorationType::None; 3];
    let mut d_total = 0u64;
    let mut d_pre_total = 0u64;
    let mut lr_state = LrWriteState::new();
    let mut task_idx = 0usize;
    for plane in 0..num_planes {
        let (unit_rows, unit_cols) = per_plane_grid[plane];
        let mut plane_kinds = (false, false); // (any wiener, any sgr)
        let mut plane_units: Vec<((usize, u32, u32), LrUnit)> = Vec::new();
        let mut plane_d = 0u64;
        for ur in 0..unit_rows {
            for uc in 0..unit_cols {
                let searched_unit = searched[task_idx].as_ref();
                task_idx += 1;
                let Some(su) = searched_unit else {
                    plane_units.push(((plane, ur, uc), LrUnit::NONE));
                    continue;
                };
                d_pre_total += su.d_none;
                let r_none = price_unit(&lr_state, plane, &LrUnit::NONE);
                let r_wiener = price_unit(&lr_state, plane, &su.wiener.0);
                let mut best = (
                    LrUnit::NONE,
                    su.d_none,
                    su.d_none * 256 + inp.lambda * r_none,
                );
                let s_wiener = su.wiener.1 * 256 + inp.lambda * r_wiener;
                if s_wiener < best.2 {
                    best = (su.wiener.0, su.wiener.1, s_wiener);
                }
                if let Some((sgr_unit, d_sgr)) = su.sgr {
                    let r_sgr = price_unit(&lr_state, plane, &sgr_unit);
                    let s_sgr = d_sgr * 256 + inp.lambda * r_sgr;
                    if s_sgr < best.2 {
                        best = (sgr_unit, d_sgr, s_sgr);
                    }
                }
                // Advance the running subexp reference state with the
                // committed unit.
                {
                    let mut w = SymbolWriter::new_counting(inp.disable_cdf_update, 0x8000);
                    let mut cdfs = inp.price_cdfs.clone();
                    let _ = write_lr_unit(
                        &mut w,
                        &mut cdfs,
                        &mut lr_state,
                        plane,
                        RESTORE_SWITCHABLE,
                        &best.0,
                    );
                }
                match best.0.restoration_type {
                    RESTORE_WIENER => plane_kinds.0 = true,
                    RESTORE_SGRPROJ => plane_kinds.1 = true,
                    _ => {}
                }
                plane_d += best.1;
                plane_units.push(((plane, ur, uc), best.0));
            }
        }
        frt[plane] = match plane_kinds {
            (false, false) => FrameRestorationType::None,
            (true, false) => FrameRestorationType::Wiener,
            (false, true) => FrameRestorationType::SgrProj,
            (true, true) => FrameRestorationType::Switchable,
        };
        // An inactive plane's units elected all-NONE, so `plane_d`
        // already equals its unfiltered SSD; its unit list is dropped
        // (the §5.11.57 window never fires for a RESTORE_NONE plane).
        d_total += plane_d;
        if frt[plane] != FrameRestorationType::None {
            plan_units.extend(plane_units);
        }
    }
    if std::env::var_os("OXIDEAV_AV1_LR_DEBUG").is_some() {
        let mut counts = [[0u32; 4]; 3];
        for ((plane, _, _), u) in &plan_units {
            counts[*plane][usize::from(u.restoration_type.min(3))] += 1;
        }
        eprintln!(
            "lr-elect: frt {frt:?} d_pre {d_pre_total} d {d_total} lambda {} unit-counts {counts:?}",
            inp.lambda
        );
    }
    if frt.iter().all(|&t| t == FrameRestorationType::None) {
        return None;
    }

    let header = header_shape(frt, inp.unit_shift);
    let write_params = crate::cdf::LrParams {
        num_planes,
        frame_restoration_type: [
            frt_ordinal(frt[0]),
            frt_ordinal(frt[1]),
            frt_ordinal(frt[2]),
        ],
        loop_restoration_size: header.loop_restoration_size,
        subsampling_x: inp.subsampling_x,
        subsampling_y: inp.subsampling_y,
        frame_height: inp.frame_height as u32,
        upscaled_width: inp.frame_width as u32,
        use_superres: inp.use_superres,
        superres_denom: inp.superres_denom,
        allow_intrabc: false,
    };
    Some(LrPlan {
        header,
        write_params,
        units: plan_units,
        d: d_total,
        d_pre: d_pre_total,
    })
}

fn frt_ordinal(t: FrameRestorationType) -> u8 {
    match t {
        FrameRestorationType::None => RESTORE_NONE,
        FrameRestorationType::Switchable => RESTORE_SWITCHABLE,
        FrameRestorationType::Wiener => RESTORE_WIENER,
        FrameRestorationType::SgrProj => RESTORE_SGRPROJ,
    }
}

/// Apply an elected plan: the §7.17 restoration of every plane over
/// the plan's unit grids — `curr` (pre-CDEF) and the current `recon`
/// (post-CDEF) in, the restored planes written back over `recon_*`
/// (the §7.20 reference store). Returns the applied whole-frame SSD
/// vs the source (callers `debug_assert` it equals `plan.d`).
///
/// r464 — runs rectangle by rectangle on local windows
/// ([`LocalRect`]) into one plane-sized output at a time (no
/// frame-sized `i32` copies); the kernels are the decoder's own, so
/// the stored planes equal the decoder's byte for byte.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_lr_plan(
    plan: &LrPlan,
    input: &YuvFrame,
    curr_y: LrCurr<'_>,
    curr_u: LrCurr<'_>,
    curr_v: LrCurr<'_>,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    width: usize,
    height: usize,
    chroma_w: usize,
    chroma_h: usize,
    bit_depth: u8,
    subsampling_x: u8,
    subsampling_y: u8,
    num_planes: u8,
    mi_rows: u32,
    mi_cols: u32,
    frame_width: usize,
    frame_height: usize,
) -> u64 {
    let num_planes = num_planes.min(3) as usize;
    let dims: Vec<(usize, usize)> = (0..num_planes)
        .map(|p| {
            if p == 0 {
                (width, height)
            } else {
                (chroma_w, chroma_h)
            }
        })
        .collect();
    let find = |plane: usize, ur: u32, uc: u32| -> LrUnit {
        plan.units
            .iter()
            .find(|(k, _)| *k == (plane, ur, uc))
            .map(|(_, u)| *u)
            .unwrap_or(LrUnit::NONE)
    };
    let geom_ctx = LoopRestorationFrameContext {
        mi_rows,
        mi_cols,
        num_planes: num_planes as u8,
        bit_depth,
        subsampling_x,
        subsampling_y,
        frame_height: frame_height as u32,
        upscaled_width: frame_width as u32,
        lr_params: &plan.header,
        lr_type: &|_, _, _| FrameRestorationType::None,
        lr_wiener: &|_, _, _, _, _| 0,
        lr_sgr_set: &|_, _, _| 0,
        lr_sgr_xqd: &|_, _, _, _| 0,
    };
    let currs: [LrCurr<'_>; 3] = [curr_y, curr_u, curr_v];
    let srcs: [&[u16]; 3] = [&input.y, &input.u, &input.v];
    let mut recons: [&mut [u16]; 3] = [recon_y, recon_u, recon_v];
    let mut ssd = 0u64;
    let mut out: Vec<i32> = Vec::new();
    for plane in 0..num_planes {
        let (pw, ph) = dims[plane];
        let (fw, fh) = if plane == 0 {
            (frame_width, frame_height)
        } else {
            (
                (frame_width + usize::from(subsampling_x)) >> subsampling_x,
                (frame_height + usize::from(subsampling_y)) >> subsampling_y,
            )
        };
        let recon = &mut recons[plane];
        if plan.header.frame_restoration_type[plane] != FrameRestorationType::None {
            // Restore into a fresh plane (later rectangles read their
            // neighbours' PRE-restoration samples), then swap in.
            let mut restored: Vec<u16> = recon.to_vec();
            for g in stripe_unit_rects(&geom_ctx, plane as u8) {
                let unit = find(plane, g.unit_row, g.unit_col);
                if unit.restoration_type == RESTORE_NONE {
                    continue;
                }
                let mut lr = LocalRect::build(&g, currs[plane], recon, pw, ph);
                eval_rect(&mut lr, &unit, plane, bit_depth, &mut out, srcs[plane], pw);
                let m = LOCAL_MARGIN;
                for i in 0..g.h as usize {
                    let orow = &out[(m + i) * lr.cols + m..(m + i) * lr.cols + m + g.w as usize];
                    let drow = &mut restored[(lr.abs_y + i) * pw + lr.abs_x..][..g.w as usize];
                    for (d, o) in drow.iter_mut().zip(orow) {
                        *d = (*o).max(0) as u16;
                    }
                }
            }
            recon.copy_from_slice(&restored);
        }
        // r460 — the SSD is measured over the CODED extent only (the
        // election's per-unit rects never reach into the mi-grid
        // padding of a non-multiple-of-8 picture; §7.17 leaves it
        // untouched).
        for y in 0..fh {
            for x in 0..fw {
                let d = i64::from(recon[y * pw + x]) - i64::from(srcs[plane][y * pw + x]);
                ssd += (d * d) as u64;
            }
        }
    }
    ssd
}
