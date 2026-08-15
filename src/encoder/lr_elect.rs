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
//! Unit-size scope (r429): `lr_unit_shift = 0`, `lr_uv_shift = 0` —
//! 64×64-sample units on every plane (the finest §5.9.20 grid; the
//! size election is left open).

use crate::cdf::TileCdfContext;
use crate::cdf::{LrUnit, RESTORE_NONE, RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER};
use crate::encoder::loop_restoration_write::{write_lr_unit, LrWriteState};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::yuv_frame::YuvFrame;
use crate::loop_filter::PlaneBuffer;
use crate::loop_restoration::{
    count_units_in_frame, derive_block_geometry, loop_restoration_frame, loop_restore_block,
    LoopRestorationFrameContext, SGRPROJ_XQD_MAX, SGRPROJ_XQD_MIN, SGR_PARAMS, WIENER_COEFFS,
    WIENER_TAPS_MAX, WIENER_TAPS_MID, WIENER_TAPS_MIN,
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
    /// boundaries).
    pub curr_y: &'a [u16],
    pub curr_u: &'a [u16],
    pub curr_v: &'a [u16],
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
    pub use_superres: bool,
    /// The §5.9.8 `SuperresDenom` (`SUPERRES_NUM` when
    /// `use_superres` is `false`).
    pub superres_denom: u32,
}

/// One unit's block list + covered rects (plane-local coordinates).
struct UnitBlocks {
    /// `(row, col)` mi coordinates fed to `loop_restore_block`.
    blocks: Vec<(u32, u32)>,
    /// `(x, y, w, h)` output rects (disjoint, union = the unit's
    /// §7.17.1 footprint).
    rects: Vec<(u32, u32, u32, u32)>,
}

/// Shared geometry + plane data for the per-unit evaluator.
struct EvalCtx<'a> {
    input: &'a YuvFrame,
    dims: Vec<(usize, usize)>,
    num_planes: usize,
    bit_depth: u8,
    subsampling_x: u8,
    subsampling_y: u8,
    mi_rows: u32,
    mi_cols: u32,
    frame_height: u32,
    upscaled_width: u32,
    eval_lrp: HeaderLrParams,
}

impl EvalCtx<'_> {
    fn src_plane(&self, plane: usize) -> &[u16] {
        match plane {
            0 => &self.input.y,
            1 => &self.input.u,
            _ => &self.input.v,
        }
    }
}

/// SSD between an i32 plane and a u16 plane over a rect list.
fn rect_ssd(a: &[i32], b: &[u16], stride: usize, rects: &[(u32, u32, u32, u32)]) -> u64 {
    let mut ssd = 0u64;
    for &(x, y, w, h) in rects {
        for row in y..y + h {
            let base = row as usize * stride;
            for col in x..x + w {
                let d = i64::from(a[base + col as usize]) - i64::from(b[base + col as usize]);
                ssd += (d * d) as u64;
            }
        }
    }
    ssd
}

/// Wrap owned plane vectors as the kernel's `PlaneBuffer` views.
fn make_bufs<'a>(owned: &'a mut [Vec<i32>], dims: &[(usize, usize)]) -> Vec<PlaneBuffer<'a>> {
    owned
        .iter_mut()
        .zip(dims.iter())
        .map(|(buf, &(w, h))| PlaneBuffer {
            rows: h as u32,
            cols: w as u32,
            samples: buf,
        })
        .collect()
}

/// The §5.9.20 header block this election codes: 64×64 units on
/// every plane, restoration types as given.
fn header_shape(frt: [FrameRestorationType; 3]) -> HeaderLrParams {
    let uses_lr = frt.iter().any(|&t| t != FrameRestorationType::None);
    let uses_chroma_lr = frt[1..].iter().any(|&t| t != FrameRestorationType::None);
    HeaderLrParams {
        frame_restoration_type: frt,
        uses_lr,
        uses_chroma_lr,
        lr_unit_shift: 0,
        lr_uv_shift: 0,
        loop_restoration_size: if uses_lr { [64, 64, 64] } else { [0, 0, 0] },
        short_circuited: false,
    }
}

/// Fill the unit's rects of `lr[plane]` from `cdef[plane]`, run the
/// decoder's §7.17.1 block driver for the unit's blocks under the
/// candidate payload, and return the unit's SSD vs the source. The
/// filtered samples stay in `lr[plane]` (the SGR probes read them
/// back).
#[allow(clippy::too_many_arguments)]
fn eval_unit(
    ec: &EvalCtx<'_>,
    curr: &mut [Vec<i32>],
    cdef: &mut [Vec<i32>],
    lr: &mut [Vec<i32>],
    plane: usize,
    ur: u32,
    uc: u32,
    unit: &LrUnit,
    ub: &UnitBlocks,
) -> u64 {
    let stride = ec.dims[plane].0;
    for &(x, y, w, h) in &ub.rects {
        for row in y..y + h {
            let base = row as usize * stride;
            for col in x..x + w {
                lr[plane][base + col as usize] = cdef[plane][base + col as usize];
            }
        }
    }
    if unit.restoration_type != RESTORE_NONE {
        let rt = match unit.restoration_type {
            RESTORE_WIENER => FrameRestorationType::Wiener,
            RESTORE_SGRPROJ => FrameRestorationType::SgrProj,
            _ => FrameRestorationType::None,
        };
        let wiener = unit.wiener;
        let sgr_set = unit.sgr_set;
        let sgr_xqd = unit.sgr_xqd;
        let ctx = LoopRestorationFrameContext {
            mi_rows: ec.mi_rows,
            mi_cols: ec.mi_cols,
            num_planes: ec.num_planes as u8,
            bit_depth: ec.bit_depth,
            subsampling_x: ec.subsampling_x,
            subsampling_y: ec.subsampling_y,
            frame_height: ec.frame_height,
            upscaled_width: ec.upscaled_width,
            lr_params: &ec.eval_lrp,
            lr_type: &move |p, r, c| {
                if p as usize == plane && r == ur && c == uc {
                    rt
                } else {
                    FrameRestorationType::None
                }
            },
            lr_wiener: &move |_, _, _, pass, i| wiener[pass as usize][i],
            lr_sgr_set: &move |_, _, _| sgr_set as u8,
            lr_sgr_xqd: &move |_, _, _, i| sgr_xqd[i],
        };
        let curr_bufs = make_bufs(curr, &ec.dims);
        let cdef_bufs = make_bufs(cdef, &ec.dims);
        let mut lr_bufs = make_bufs(lr, &ec.dims);
        for &(row, col) in &ub.blocks {
            loop_restore_block(
                &ctx,
                &curr_bufs,
                &cdef_bufs,
                &mut lr_bufs,
                plane as u8,
                row,
                col,
            );
        }
    }
    rect_ssd(&lr[plane], ec.src_plane(plane), stride, &ub.rects)
}

/// Alternating-least-squares Wiener tap fit for one unit (free
/// encoder engineering; the exact §7.17.4 kernel evaluates the
/// quantised result). `first_coeff = 1` on chroma (`taps[pass][0]`
/// is forced 0 by §5.11.58).
fn fit_wiener(
    cdef: &[i32],
    src: &[u16],
    plane_w: usize,
    plane_h: usize,
    rects: &[(u32, u32, u32, u32)],
    first_coeff: usize,
) -> [[i32; WIENER_COEFFS]; 2] {
    let at = |x: i64, y: i64| -> f64 {
        let xc = x.clamp(0, plane_w as i64 - 1) as usize;
        let yc = y.clamp(0, plane_h as i64 - 1) as usize;
        f64::from(cdef[yc * plane_w + xc])
    };
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
    for _round in 0..2 {
        for dir in 0..2usize {
            // dir 0: fit horizontal (pass 1) through the vertical
            // taps; dir 1: fit vertical (pass 0) through the
            // horizontal taps.
            let fixed = taps7(if dir == 0 { &vt } else { &ht });
            let nvar = 3 - first_coeff;
            let mut ata = [[0f64; 3]; 3];
            let mut atb = [0f64; 3];
            for &(rx, ry, rw, rh) in rects {
                for y in ry..ry + rh {
                    for x in rx..rx + rw {
                        let mut m = [0f64; 7];
                        for (j, mj) in m.iter_mut().enumerate() {
                            let off = j as i64 - 3;
                            let mut acc = 0f64;
                            for (k, fk) in fixed.iter().enumerate() {
                                let foff = k as i64 - 3;
                                let (sx, sy) = if dir == 0 {
                                    (x as i64 + off, y as i64 + foff)
                                } else {
                                    (x as i64 + foff, y as i64 + off)
                                };
                                acc += fk * at(sx, sy);
                            }
                            *mj = acc / 128.0;
                        }
                        let target = f64::from(src[y as usize * plane_w + x as usize]) - m[3];
                        let mut basis = [0f64; 3];
                        for (i, b) in basis.iter_mut().enumerate() {
                            *b = (m[i] + m[6 - i] - 2.0 * m[3]) / 128.0;
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
    let ec = EvalCtx {
        input: inp.input,
        dims: dims.clone(),
        num_planes,
        bit_depth: inp.bit_depth,
        subsampling_x: inp.subsampling_x,
        subsampling_y: inp.subsampling_y,
        mi_rows: inp.mi_rows,
        mi_cols: inp.mi_cols,
        frame_height: inp.height as u32,
        upscaled_width: inp.width as u32,
        // Eval-side header: every plane SWITCHABLE so the per-unit
        // closure decides (the real header collapses below).
        eval_lrp: header_shape([FrameRestorationType::Switchable; 3]),
    };

    let to_i32 = |p: &[u16]| -> Vec<i32> { p.iter().map(|&v| i32::from(v)).collect() };
    let mut curr_owned: Vec<Vec<i32>> = vec![to_i32(inp.curr_y)];
    let mut cdef_owned: Vec<Vec<i32>> = vec![to_i32(inp.cdef_y)];
    if num_planes > 1 {
        curr_owned.push(to_i32(inp.curr_u));
        curr_owned.push(to_i32(inp.curr_v));
        cdef_owned.push(to_i32(inp.cdef_u));
        cdef_owned.push(to_i32(inp.cdef_v));
    }
    let mut lr_owned: Vec<Vec<i32>> = cdef_owned.clone();

    // Per-plane unit maps via the decoder's own §7.17.1 geometry
    // (the `(lumaY + 8)` stripe shift makes unit membership
    // non-obvious — derive it exactly).
    let mut per_plane_units: Vec<Vec<UnitBlocks>> = Vec::new();
    let mut per_plane_grid: Vec<(u32, u32)> = Vec::new();
    {
        let geom_ctx = LoopRestorationFrameContext {
            mi_rows: ec.mi_rows,
            mi_cols: ec.mi_cols,
            num_planes: num_planes as u8,
            bit_depth: ec.bit_depth,
            subsampling_x: ec.subsampling_x,
            subsampling_y: ec.subsampling_y,
            frame_height: ec.frame_height,
            upscaled_width: ec.upscaled_width,
            lr_params: &ec.eval_lrp,
            lr_type: &|_, _, _| FrameRestorationType::None,
            lr_wiener: &|_, _, _, _, _| 0,
            lr_sgr_set: &|_, _, _| 0,
            lr_sgr_xqd: &|_, _, _, _| 0,
        };
        for plane in 0..num_planes {
            let (sub_x, sub_y) = if plane == 0 {
                (0u32, 0u32)
            } else {
                (u32::from(inp.subsampling_x), u32::from(inp.subsampling_y))
            };
            let unit_rows = count_units_in_frame(64, (ec.frame_height + sub_y) >> sub_y);
            let unit_cols = count_units_in_frame(64, (ec.upscaled_width + sub_x) >> sub_x);
            let mut units: Vec<UnitBlocks> = (0..unit_rows * unit_cols)
                .map(|_| UnitBlocks {
                    blocks: Vec::new(),
                    rects: Vec::new(),
                })
                .collect();
            let mut y = 0u32;
            while y < ec.frame_height {
                let mut x = 0u32;
                while x < ec.upscaled_width {
                    let (row, col) = (y >> 2, x >> 2);
                    let g = derive_block_geometry(&geom_ctx, plane as u8, row, col);
                    if g.w > 0 && g.h > 0 {
                        let k = (g.unit_row * unit_cols + g.unit_col) as usize;
                        units[k].blocks.push((row, col));
                        units[k].rects.push((g.x, g.y, g.w, g.h));
                    }
                    x += 4;
                }
                y += 4;
            }
            per_plane_units.push(units);
            per_plane_grid.push((unit_rows, unit_cols));
        }
    }

    // Exact §5.11.58 bits for one unit from the given reference state
    // (selection symbol priced from the frame-start CDFs under the
    // SWITCHABLE arm).
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
    for plane in 0..num_planes {
        let (unit_rows, unit_cols) = per_plane_grid[plane];
        let (plane_w, plane_h) = dims[plane];
        let first_coeff = usize::from(plane != 0);
        let mut plane_kinds = (false, false); // (any wiener, any sgr)
        let mut plane_units: Vec<((usize, u32, u32), LrUnit)> = Vec::new();
        let mut plane_d = 0u64;
        for ur in 0..unit_rows {
            for uc in 0..unit_cols {
                let ub = &per_plane_units[plane][(ur * unit_cols + uc) as usize];
                if ub.blocks.is_empty() {
                    plane_units.push(((plane, ur, uc), LrUnit::NONE));
                    continue;
                }
                let d_none = eval_unit(
                    &ec,
                    &mut curr_owned,
                    &mut cdef_owned,
                    &mut lr_owned,
                    plane,
                    ur,
                    uc,
                    &LrUnit::NONE,
                    ub,
                );
                d_pre_total += d_none;

                // Wiener candidate.
                let taps = fit_wiener(
                    &cdef_owned[plane],
                    ec.src_plane(plane),
                    plane_w,
                    plane_h,
                    &ub.rects,
                    first_coeff,
                );
                let wiener_unit = LrUnit {
                    restoration_type: RESTORE_WIENER,
                    wiener: taps,
                    sgr_set: 0,
                    sgr_xqd: [0; 2],
                };
                let d_wiener = eval_unit(
                    &ec,
                    &mut curr_owned,
                    &mut cdef_owned,
                    &mut lr_owned,
                    plane,
                    ur,
                    uc,
                    &wiener_unit,
                    ub,
                );

                // Self-guided candidate: probe-fit each set, keep the
                // best by exact SSD.
                let mut best_sgr: Option<(LrUnit, u64)> = None;
                for (set, params) in SGR_PARAMS.iter().enumerate() {
                    let (r0, r1) = (params[0], params[2]);
                    let read_delta = |lr: &[Vec<i32>], cdef: &[Vec<i32>]| -> Vec<i32> {
                        let mut out = Vec::new();
                        for &(x, y, w, h) in &ub.rects {
                            for row in y..y + h {
                                let base = row as usize * plane_w;
                                for col in x..x + w {
                                    out.push(
                                        lr[plane][base + col as usize]
                                            - cdef[plane][base + col as usize],
                                    );
                                }
                            }
                        }
                        out
                    };
                    let mut probe = |xqd: [i32; 2]| -> Vec<i32> {
                        let u = LrUnit {
                            restoration_type: RESTORE_SGRPROJ,
                            wiener: [[0; WIENER_COEFFS]; 2],
                            sgr_set: set,
                            sgr_xqd: xqd,
                        };
                        let _ = eval_unit(
                            &ec,
                            &mut curr_owned,
                            &mut cdef_owned,
                            &mut lr_owned,
                            plane,
                            ur,
                            uc,
                            &u,
                            ub,
                        );
                        read_delta(&lr_owned, &cdef_owned)
                    };
                    let a0 = (r0 != 0).then(|| probe([128, 0]));
                    let a1 = (r1 != 0).then(|| probe([0, 128]));
                    // Targets: 128·(src - dgd) over the same traversal.
                    let mut t: Vec<f64> = Vec::new();
                    {
                        let s = ec.src_plane(plane);
                        for &(x, y, w, h) in &ub.rects {
                            for row in y..y + h {
                                let base = row as usize * plane_w;
                                for col in x..x + w {
                                    t.push(
                                        128.0
                                            * (f64::from(s[base + col as usize])
                                                - f64::from(
                                                    cdef_owned[plane][base + col as usize],
                                                )),
                                    );
                                }
                            }
                        }
                    }
                    // Least squares on 128·out_delta = xq0·a0 + xq1·a1.
                    let (mut xq0, mut xq1) = (0i32, 0i32);
                    match (&a0, &a1) {
                        (Some(a0), Some(a1)) => {
                            let (mut s00, mut s01, mut s11, mut b0, mut b1) =
                                (0f64, 0f64, 0f64, 0f64, 0f64);
                            for i in 0..t.len() {
                                let (x0, x1) = (f64::from(a0[i]), f64::from(a1[i]));
                                s00 += x0 * x0;
                                s01 += x0 * x1;
                                s11 += x1 * x1;
                                b0 += x0 * t[i];
                                b1 += x1 * t[i];
                            }
                            let det = s00 * s11 - s01 * s01;
                            if det.abs() > 1e-6 {
                                xq0 = ((s11 * b0 - s01 * b1) / det).round() as i32;
                                xq1 = ((s00 * b1 - s01 * b0) / det).round() as i32;
                            }
                        }
                        (Some(a0), None) => {
                            let (mut s00, mut b0) = (0f64, 0f64);
                            for i in 0..t.len() {
                                let x0 = f64::from(a0[i]);
                                s00 += x0 * x0;
                                b0 += x0 * t[i];
                            }
                            if s00 > 1e-6 {
                                xq0 = (b0 / s00).round() as i32;
                            }
                        }
                        (None, Some(a1)) => {
                            let (mut s11, mut b1) = (0f64, 0f64);
                            for i in 0..t.len() {
                                let x1 = f64::from(a1[i]);
                                s11 += x1 * x1;
                                b1 += x1 * t[i];
                            }
                            if s11 > 1e-6 {
                                xq1 = (b1 / s11).round() as i32;
                            }
                        }
                        (None, None) => {}
                    }
                    // §5.11.58 constraints: clamp, derive radius-0
                    // components.
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
                    let d = eval_unit(
                        &ec,
                        &mut curr_owned,
                        &mut cdef_owned,
                        &mut lr_owned,
                        plane,
                        ur,
                        uc,
                        &cand,
                        ub,
                    );
                    if best_sgr.as_ref().map(|(_, bd)| d < *bd).unwrap_or(true) {
                        best_sgr = Some((cand, d));
                    }
                }

                // Elect argmin D + λ·R.
                let r_none = price_unit(&lr_state, plane, &LrUnit::NONE);
                let r_wiener = price_unit(&lr_state, plane, &wiener_unit);
                let mut best = (LrUnit::NONE, d_none, d_none * 256 + inp.lambda * r_none);
                let s_wiener = d_wiener * 256 + inp.lambda * r_wiener;
                if s_wiener < best.2 {
                    best = (wiener_unit, d_wiener, s_wiener);
                }
                if let Some((sgr_unit, d_sgr)) = best_sgr {
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

    let header = header_shape(frt);
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
        frame_height: ec.frame_height,
        upscaled_width: ec.upscaled_width,
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

/// Apply an elected plan: one §7.17 run through the decoder's own
/// frame driver over the plan's unit grids — `curr` (pre-CDEF) and
/// the current `recon` (post-CDEF) in, the restored planes written
/// back over `recon_*` (the §7.20 reference store). Returns the
/// applied whole-frame SSD vs the source (callers `debug_assert` it
/// equals `plan.d`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_lr_plan(
    plan: &LrPlan,
    input: &YuvFrame,
    curr_y: &[u16],
    curr_u: &[u16],
    curr_v: &[u16],
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
    let to_i32 = |p: &[u16]| -> Vec<i32> { p.iter().map(|&v| i32::from(v)).collect() };
    let mut curr_owned: Vec<Vec<i32>> = vec![to_i32(curr_y)];
    let mut cdef_owned: Vec<Vec<i32>> = vec![to_i32(recon_y)];
    if num_planes > 1 {
        curr_owned.push(to_i32(curr_u));
        curr_owned.push(to_i32(curr_v));
        cdef_owned.push(to_i32(recon_u));
        cdef_owned.push(to_i32(recon_v));
    }
    let mut lr_owned: Vec<Vec<i32>> = cdef_owned.clone();
    let find = |plane: u8, ur: u32, uc: u32| -> LrUnit {
        plan.units
            .iter()
            .find(|(k, _)| *k == (plane as usize, ur, uc))
            .map(|(_, u)| *u)
            .unwrap_or(LrUnit::NONE)
    };
    {
        let ctx = LoopRestorationFrameContext {
            mi_rows,
            mi_cols,
            num_planes: num_planes as u8,
            bit_depth,
            subsampling_x,
            subsampling_y,
            frame_height: height as u32,
            upscaled_width: width as u32,
            lr_params: &plan.header,
            lr_type: &|p, r, c| match find(p, r, c).restoration_type {
                RESTORE_WIENER => FrameRestorationType::Wiener,
                RESTORE_SGRPROJ => FrameRestorationType::SgrProj,
                _ => FrameRestorationType::None,
            },
            lr_wiener: &|p, r, c, pass, i| find(p, r, c).wiener[pass as usize][i],
            lr_sgr_set: &|p, r, c| find(p, r, c).sgr_set as u8,
            lr_sgr_xqd: &|p, r, c, i| find(p, r, c).sgr_xqd[i],
        };
        let curr_bufs = make_bufs(&mut curr_owned, &dims);
        let cdef_bufs = make_bufs(&mut cdef_owned, &dims);
        let mut lr_bufs = make_bufs(&mut lr_owned, &dims);
        loop_restoration_frame(&ctx, &curr_bufs, &cdef_bufs, &mut lr_bufs);
    }
    let mut ssd = 0u64;
    let srcs: [&[u16]; 3] = [&input.y, &input.u, &input.v];
    let recons: [&mut [u16]; 3] = [recon_y, recon_u, recon_v];
    for (p, recon) in recons.into_iter().enumerate().take(num_planes) {
        for (dst, (&out, &s)) in recon.iter_mut().zip(lr_owned[p].iter().zip(srcs[p].iter())) {
            *dst = out.max(0) as u16;
            let d = i64::from(out) - i64::from(s);
            ssd += (d * d) as u64;
        }
    }
    ssd
}
