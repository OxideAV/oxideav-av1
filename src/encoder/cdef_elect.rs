//! r428/r429 — encoder-side §5.9.19 / §7.15 CDEF election (encoder
//! ladder item 3: r428 frame-level arm, r429 per-64×64-unit arm).
//!
//! The decoder's CDEF is corpus-complete; this module mirrors it on
//! the ENCODER's reconstruction path: after the tile is committed the
//! frame's pre-CDEF reconstruction is filtered through the decoder's
//! own §7.15 driver ([`crate::cdef::cdef_frame`]) over the write
//! mirror's committed grids (the §5.11.56 `cdef_idx[]` anchors and
//! the §7.15.1 `Skips[]` conjunction — exactly the state the decoder
//! derives from the emitted tile), a bounded strength search scores
//! each candidate against the SOURCE, and the winner (when it beats
//! the unfiltered frame) is stamped into the header and applied to
//! the reconstruction — so the stored reference planes equal the
//! decoder's §7.20 store byte-for-byte, like every other stage of
//! this encoder.
//!
//! ## The two arms
//!
//! * **Frame-level** (`cdef_bits = 0`, r428): one strength set for
//!   the whole frame — the §5.11.56 `cdef_idx` literal is `L(0)`,
//!   ZERO tile bits, so the arm is pure distortion.
//! * **Per-unit** (`cdef_bits ∈ 1..=3`, r429): up to `1 << cdef_bits`
//!   §5.9.19 strength sets in the header, each 64×64 unit electing
//!   its id through the `L(cdef_bits)` literal of §5.11.56. §7.15
//!   reads only PRE-CDEF samples (`CurrFrame` in, `CdefFrame` out),
//!   so a unit's filtered output depends on its own id alone and the
//!   election decomposes exactly into per-unit SSD tables. Rate is
//!   exact by construction: an `L(n)` literal through the §8.2.6
//!   bool coder costs exactly `n` bits (equiprobable halving), and
//!   the §5.9.19 header grows by exactly 6 (+6 with chroma) bits per
//!   extra strength set — both priced against λ on the same
//!   1/256-bit scale the twin-priced ladders use. The caller
//!   RE-EMITS the tile with the elected per-leaf ids (the literal
//!   perturbs the arithmetic coder state, so the whole tile is
//!   rewritten from the committed trees — the established
//!   exact-replay machinery).
//!
//! The search strategy (coarse-then-refine primary sweep, secondary
//! set, damping sweep, greedy set-list growth) is free encoder
//! engineering; every candidate is evaluated through the real §7.15
//! kernels.

use crate::cdef::{cdef_block_dir, cdef_direction, CdefFrameContext};
use crate::cdf::PartitionWalker;
use crate::encoder::yuv_frame::YuvFrame;
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::CdefParams;

/// One plane set's strength candidate.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Strength {
    pri: u8,
    sec: u8,
}

const ZERO: Strength = Strength { pri: 0, sec: 0 };

/// The elected CDEF configuration: header params + the per-64×64-unit
/// §5.11.56 strength ids (raster order over `ceil(MiCols / 16)`
/// units per row; `-1` = the unit codes no idx — every block in it
/// is skip — and the decoder's copy path applies). `d` is the plan's
/// EXACT whole-frame SSD against the source under the real §7.15
/// kernels (the per-unit decomposition is exact — §7.15 reads only
/// pre-CDEF samples).
pub(crate) struct CdefPlan {
    pub params: CdefParams,
    pub unit_idx: Vec<i8>,
    pub d: u64,
}

/// [`elect_cdef`]'s result: the estimate-best plan plus what the
/// caller needs for the FINAL exact-realized-bytes election when
/// `best` is a per-unit plan (the plan-stage rate model prices the
/// `L(cdef_bits)` literals and the §5.9.19 header growth exactly in
/// bits, but the emitted tile and header are byte-aligned — the
/// caller re-emits and settles per-unit vs frame-level vs unfiltered
/// on real byte counts, the same doctrine as the hp / temporal-seg /
/// primary-ref elections).
pub(crate) struct CdefElection {
    /// The `D + λ·R` winner on the plan-stage (exact-bits) scale.
    pub best: CdefPlan,
    /// The frame-level alternative (`cdef_bits = 0`), present iff it
    /// beats the unfiltered frame — the caller's fallback arm when
    /// `best` is per-unit and loses the exact-bytes settlement.
    pub frame_level: Option<CdefPlan>,
    /// The unfiltered frame's SSD (the no-election arm).
    pub base_d: u64,
}

/// Election inputs (see [`elect_cdef`]).
pub(crate) struct CdefElectInput<'a> {
    pub mirror: &'a PartitionWalker,
    pub input: &'a YuvFrame,
    pub recon_y: &'a [u16],
    pub recon_u: &'a [u16],
    pub recon_v: &'a [u16],
    pub width: usize,
    pub height: usize,
    pub chroma_w: usize,
    pub chroma_h: usize,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub num_planes: u8,
    /// λ on the [`super::rate_twin::score256`] 1/256-bit convention
    /// (the frame quantiser's [`super::key_frame::lambda_for`]).
    pub lambda: u64,
    /// Highest §5.9.19 `cdef_bits` the election may propose
    /// (`0` = frame-level only — the r428 shape; spec cap is 3).
    pub max_bits: u8,
    /// r464 — the reduced sweep: the coarse strength ladder is
    /// scored on a quarter of the units (every other unit row and
    /// column), the three best luma / chroma strengths per plane
    /// set at the better damping are then evaluated on every unit
    /// for the per-unit arm. `false` = the full ladder on every unit
    /// (the r429 election, unchanged).
    pub fast: bool,
    /// r464 — worker threads for the per-unit evaluation (`1` =
    /// sequential; the election is identical either way).
    pub threads: usize,
}

/// Per-unit SSD tables for one plane set at one damping: `cands[0]`
/// is always the zero strength (per-unit SSD = the unfiltered base).
struct SetTables {
    cands: Vec<Strength>,
    /// `ssd[cand][unit]`.
    ssd: Vec<Vec<u64>>,
    /// The §5.9.19 damping the rows were evaluated at.
    damping: u8,
}

impl SetTables {
    fn total(&self, cand: usize) -> u64 {
        self.ssd[cand].iter().sum()
    }
    fn best_by_total(&self) -> usize {
        (0..self.cands.len())
            .min_by_key(|&i| self.total(i))
            .unwrap_or(0)
    }
}

/// Best total of a table (the zero strength included).
fn t_best(t: &SetTables) -> u64 {
    t.total(t.best_by_total())
}

/// r464 — the per-unit evaluation engine: every 64×64 unit is lifted
/// into a local window per plane (the unit plus an 8-luma-sample
/// apron of TRUE neighbour samples where the frame continues, the
/// frame edge where it ends — exactly the §7.15.3 `CdefAvailable`
/// footprint, since no tap reaches further than 2 samples), the
/// §7.15.2 directions of its filtered 8×8 blocks are searched once
/// and cached, and every candidate schedule runs the decoder's own
/// §7.15.3 kernel over the window. Per-unit SSDs are exact (§7.15
/// reads only pre-CDEF samples, so a unit's output depends on its
/// own window alone); no frame-sized `i32` copy is ever made, and
/// units are independent so they are evaluated `threads`-wide.
/// One filtered 8×8 block of a unit: `(r, c)` in frame mi
/// coordinates and its cached §7.15.2 `(yDir, var)`.
type DirBlock = (u32, u32, (i32, i32));

/// One plane's filtered rectangle: `(x0, y0, w, h, samples)` in plane
/// coordinates, samples row-major `h × w`.
type FilteredRect = (usize, usize, usize, usize, Vec<i32>);

struct UnitEngine<'a> {
    inp: &'a CdefElectInput<'a>,
    sb_rows: usize,
    sb_cols: usize,
    coded: &'a [bool],
    /// Per unit: the filtered blocks `(r, c)` in FRAME mi coordinates
    /// with their `(yDir, var)`; empty for uncoded / all-skip units.
    blocks: Vec<Vec<DirBlock>>,
}

/// One plane's window of one unit.
struct Window {
    /// Window origin in plane samples (a multiple of the plane's
    /// 8×8-block pitch).
    ox: usize,
    oy: usize,
    rows: usize,
    cols: usize,
    /// The unit's rectangle inside the window.
    ux0: usize,
    uy0: usize,
    ux1: usize,
    uy1: usize,
    src: Vec<i32>,
    dst: Vec<i32>,
}

impl<'a> UnitEngine<'a> {
    fn new(inp: &'a CdefElectInput<'a>, coded: &'a [bool]) -> Self {
        let mi_rows = inp.mirror.mi_rows();
        let mi_cols = inp.mirror.mi_cols();
        let sb_rows = mi_rows.div_ceil(16) as usize;
        let sb_cols = mi_cols.div_ceil(16) as usize;
        let mut engine = Self {
            inp,
            sb_rows,
            sb_cols,
            coded,
            blocks: Vec::new(),
        };
        // Directions: one pass over the units (parallel), luma only.
        let n_units = sb_rows * sb_cols;
        let dirs: Vec<Vec<DirBlock>> = engine.map_units(
            (0..n_units).collect::<Vec<_>>().as_slice(),
            |eng, k, scratch| eng.search_block_dirs(k, scratch),
        );
        engine.blocks = dirs;
        engine
    }

    fn plane_dims(&self, plane: usize) -> (usize, usize, u8, u8) {
        if plane == 0 {
            (self.inp.width, self.inp.height, 0, 0)
        } else {
            (
                self.inp.chroma_w,
                self.inp.chroma_h,
                self.inp.subsampling_x,
                self.inp.subsampling_y,
            )
        }
    }

    fn recon_plane(&self, plane: usize) -> &[u16] {
        match plane {
            0 => self.inp.recon_y,
            1 => self.inp.recon_u,
            _ => self.inp.recon_v,
        }
    }

    fn src_plane(&self, plane: usize) -> &[u16] {
        match plane {
            0 => &self.inp.input.y,
            1 => &self.inp.input.u,
            _ => &self.inp.input.v,
        }
    }

    /// Build the window of unit `k` on `plane`.
    fn window(&self, k: usize, plane: usize) -> Window {
        let (pw, ph, ssx, ssy) = self.plane_dims(plane);
        let (ur, uc) = (k / self.sb_cols, k % self.sb_cols);
        let (unit_w, unit_h) = (64usize >> ssx, 64usize >> ssy);
        let (mg_x, mg_y) = (8usize >> ssx, 8usize >> ssy);
        let ux0 = uc * unit_w;
        let uy0 = ur * unit_h;
        let ux1 = (ux0 + unit_w).min(pw);
        let uy1 = (uy0 + unit_h).min(ph);
        let ox = ux0.saturating_sub(mg_x);
        let oy = uy0.saturating_sub(mg_y);
        let wx1 = (ux1 + mg_x).min(pw);
        let wy1 = (uy1 + mg_y).min(ph);
        let (cols, rows) = (wx1 - ox, wy1 - oy);
        let recon = self.recon_plane(plane);
        let mut src = vec![0i32; rows * cols];
        for y in 0..rows {
            let srow = &recon[(oy + y) * pw + ox..(oy + y) * pw + ox + cols];
            for (d, &v) in src[y * cols..(y + 1) * cols].iter_mut().zip(srow) {
                *d = i32::from(v);
            }
        }
        let dst = src.clone();
        Window {
            ox,
            oy,
            rows,
            cols,
            ux0: ux0 - ox,
            uy0: uy0 - oy,
            ux1: ux1 - ox,
            uy1: uy1 - oy,
            src,
            dst,
        }
    }

    /// Context for the kernels over unit `k`'s windows: mi coordinates
    /// are window-local (the luma window origin is a multiple of 8,
    /// so block `(r, c)` maps to frame block `(r + r_off, c + c_off)`).
    fn offsets(&self, k: usize) -> (u32, u32) {
        let (ur, uc) = (k / self.sb_cols, k % self.sb_cols);
        let oy = (ur * 64).saturating_sub(8);
        let ox = (uc * 64).saturating_sub(8);
        ((oy / 4) as u32, (ox / 4) as u32)
    }

    /// The §7.15.2 directions of unit `k`'s filtered blocks.
    fn search_block_dirs(&self, k: usize, _scratch: &mut Scratch) -> Vec<DirBlock> {
        let mut out = Vec::new();
        if !self.coded[k] {
            return out;
        }
        let (r_off, c_off) = self.offsets(k);
        let mi_rows = self.inp.mirror.mi_rows();
        let mi_cols = self.inp.mirror.mi_cols();
        let (ur, uc) = (k / self.sb_cols, k % self.sb_cols);
        let win = self.window(k, 0);
        let params = CdefParams::short_circuit();
        let ctx = CdefFrameContext {
            mi_rows: mi_rows - r_off,
            mi_cols: mi_cols - c_off,
            num_planes: 1,
            bit_depth: self.inp.bit_depth,
            subsampling_x: self.inp.subsampling_x,
            subsampling_y: self.inp.subsampling_y,
            cdef_params: &params,
            cdef_idx: &|_, _| 0,
            skip: &|r, c| self.inp.mirror.skip_at_mi(r + r_off, c + c_off),
        };
        let src = [PlaneBuffer {
            rows: win.rows as u32,
            cols: win.cols as u32,
            samples: &mut win.src.clone(),
        }];
        let (r0, c0) = ((ur * 16) as u32, (uc * 16) as u32);
        let (r1, c1) = ((r0 + 16).min(mi_rows), (c0 + 16).min(mi_cols));
        let mut r = r0;
        while r < r1 {
            let mut c = c0;
            while c < c1 {
                // §7.15.1 skip conjunction over the four 4×4 cells.
                let skip = (ctx.skip)(r - r_off, c - c_off)
                    && (r + 1 >= mi_rows || (ctx.skip)(r + 1 - r_off, c - c_off))
                    && (c + 1 >= mi_cols || (ctx.skip)(r - r_off, c + 1 - c_off))
                    && (r + 1 >= mi_rows
                        || c + 1 >= mi_cols
                        || (ctx.skip)(r + 1 - r_off, c + 1 - c_off));
                if !skip {
                    let dir = cdef_direction(&ctx, &src, r - r_off, c - c_off);
                    out.push((r, c, dir));
                }
                c += 2;
            }
            r += 2;
        }
        out
    }

    /// Unfiltered per-unit SSD (luma, chroma).
    fn base_units(&self) -> (Vec<u64>, Vec<u64>) {
        let n = self.sb_rows * self.sb_cols;
        let mut y = vec![0u64; n];
        let mut uv = vec![0u64; n];
        for k in 0..n {
            let (ur, uc) = (k / self.sb_cols, k % self.sb_cols);
            for plane in 0..self.inp.num_planes as usize {
                let (pw, ph, ssx, ssy) = self.plane_dims(plane);
                let (unit_w, unit_h) = (64usize >> ssx, 64usize >> ssy);
                let (x0, y0) = (uc * unit_w, ur * unit_h);
                let (x1, y1) = ((x0 + unit_w).min(pw), (y0 + unit_h).min(ph));
                let (rec, src) = (self.recon_plane(plane), self.src_plane(plane));
                let mut ssd = 0u64;
                for yy in y0..y1 {
                    for (a, b) in rec[yy * pw + x0..yy * pw + x1]
                        .iter()
                        .zip(&src[yy * pw + x0..yy * pw + x1])
                    {
                        let d = i64::from(*a) - i64::from(*b);
                        ssd += (d * d) as u64;
                    }
                }
                if plane == 0 {
                    y[k] += ssd;
                } else {
                    uv[k] += ssd;
                }
            }
        }
        (y, uv)
    }

    /// The Fast subsample: every other unit row and column.
    fn in_subsample(&self, k: usize) -> bool {
        let (ur, uc) = (k / self.sb_cols, k % self.sb_cols);
        ur % 2 == 0 && uc % 2 == 0
    }

    /// Restrict a per-unit row to the subsample (other units zeroed —
    /// totals then compare the subsample only).
    fn subsample_row(&self, row: &[u64]) -> Vec<u64> {
        row.iter()
            .enumerate()
            .map(|(k, &v)| if self.in_subsample(k) { v } else { 0 })
            .collect()
    }

    /// Run `f` over the listed units, `threads`-wide, results in unit
    /// order.
    fn map_units<T: Send>(
        &self,
        units: &[usize],
        f: impl Fn(&Self, usize, &mut Scratch) -> T + Sync,
    ) -> Vec<T> {
        let threads = self.inp.threads.max(1).min(units.len().max(1));
        if threads <= 1 {
            let mut scratch = Scratch::default();
            return units.iter().map(|&k| f(self, k, &mut scratch)).collect();
        }
        let next = core::sync::atomic::AtomicUsize::new(0);
        let mut slots: Vec<Option<T>> = (0..units.len()).map(|_| None).collect();
        let results: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    let (next, f) = (&next, &f);
                    scope.spawn(move || {
                        let mut outs = Vec::new();
                        let mut scratch = Scratch::default();
                        loop {
                            let i = next.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
                            if i >= units.len() {
                                break;
                            }
                            outs.push((i, f(self, units[i], &mut scratch)));
                        }
                        outs
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("cdef unit thread panicked"))
                .collect()
        });
        for outs in results {
            for (i, v) in outs {
                slots[i] = Some(v);
            }
        }
        slots
            .into_iter()
            .map(|s| s.expect("every unit evaluated"))
            .collect()
    }

    /// Filter unit `k` under `params` / `idx` on every plane; returns
    /// per plane `(x0, y0, w, h, samples)` — the unit's rectangle in
    /// plane coordinates and its filtered samples (row-major `h × w`).
    fn filter_unit(&self, k: usize, params: &CdefParams, idx: i8) -> Vec<FilteredRect> {
        let (r_off, c_off) = self.offsets(k);
        let mi_rows = self.inp.mirror.mi_rows();
        let mi_cols = self.inp.mirror.mi_cols();
        let num_planes = self.inp.num_planes;
        let mut wins: Vec<Window> = (0..num_planes as usize)
            .map(|p| self.window(k, p))
            .collect();
        let ctx = CdefFrameContext {
            mi_rows: mi_rows - r_off,
            mi_cols: mi_cols - c_off,
            num_planes,
            bit_depth: self.inp.bit_depth,
            subsampling_x: self.inp.subsampling_x,
            subsampling_y: self.inp.subsampling_y,
            cdef_params: params,
            cdef_idx: &|_, _| idx,
            skip: &|r, c| self.inp.mirror.skip_at_mi(r + r_off, c + c_off),
        };
        {
            let mut srcs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(3);
            let mut dsts: Vec<PlaneBuffer<'_>> = Vec::with_capacity(3);
            for w in wins.iter_mut() {
                srcs.push(PlaneBuffer {
                    rows: w.rows as u32,
                    cols: w.cols as u32,
                    samples: &mut w.src,
                });
                dsts.push(PlaneBuffer {
                    rows: w.rows as u32,
                    cols: w.cols as u32,
                    samples: &mut w.dst,
                });
            }
            for &(r, c, dir) in &self.blocks[k] {
                cdef_block_dir(
                    &ctx,
                    &srcs,
                    &mut dsts,
                    num_planes,
                    r - r_off,
                    c - c_off,
                    idx,
                    dir,
                );
            }
        }
        wins.iter()
            .map(|w| {
                let (wu, hu) = (w.ux1 - w.ux0, w.uy1 - w.uy0);
                let mut out = Vec::with_capacity(wu * hu);
                for y in w.uy0..w.uy1 {
                    out.extend_from_slice(&w.dst[y * w.cols + w.ux0..y * w.cols + w.ux1]);
                }
                (w.ox + w.ux0, w.oy + w.uy0, wu, hu, out)
            })
            .collect()
    }

    /// Per-unit SSD rows for every candidate schedule: `which` 0 =
    /// luma set (luma SSD), 1 = chroma set (U + V SSD), 2 = both
    /// (luma + chroma SSD). With `subsample` only the Fast subsample
    /// units are evaluated (others read 0).
    fn evaluate(&self, cands: &[CdefParams], which: u8, subsample: bool) -> Vec<Vec<u64>> {
        let n_units = self.sb_rows * self.sb_cols;
        let units: Vec<usize> = (0..n_units)
            .filter(|&k| self.coded[k] && !self.blocks[k].is_empty())
            .filter(|&k| !subsample || self.in_subsample(k))
            .collect();
        let (base_y, base_uv) = self.base_units();
        let per_unit: Vec<Vec<u64>> = self.map_units(&units, |eng, k, scratch| {
            eng.evaluate_unit(k, cands, which, scratch)
        });
        let mut rows: Vec<Vec<u64>> = (0..cands.len())
            .map(|_| {
                (0..n_units)
                    .map(|k| {
                        if subsample && !self.in_subsample(k) {
                            0
                        } else {
                            match which {
                                0 => base_y[k],
                                1 => base_uv[k],
                                _ => base_y[k] + base_uv[k],
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        for (ui, &k) in units.iter().enumerate() {
            for (ci, row) in rows.iter_mut().enumerate() {
                row[k] = per_unit[ui][ci];
            }
        }
        rows
    }

    /// SSD of unit `k` under every candidate for plane set `which`.
    fn evaluate_unit(
        &self,
        k: usize,
        cands: &[CdefParams],
        which: u8,
        _scratch: &mut Scratch,
    ) -> Vec<u64> {
        let (r_off, c_off) = self.offsets(k);
        let mi_rows = self.inp.mirror.mi_rows();
        let mi_cols = self.inp.mirror.mi_cols();
        let planes: Vec<usize> = match which {
            0 => vec![0],
            1 => vec![1, 2],
            _ => (0..self.inp.num_planes as usize).collect(),
        };
        let num_planes = if which == 0 { 1 } else { self.inp.num_planes };
        let mut wins: Vec<Window> = (0..3)
            .map(|p| {
                if planes.contains(&p) {
                    self.window(k, p)
                } else {
                    Window {
                        ox: 0,
                        oy: 0,
                        rows: 0,
                        cols: 0,
                        ux0: 0,
                        uy0: 0,
                        ux1: 0,
                        uy1: 0,
                        src: Vec::new(),
                        dst: Vec::new(),
                    }
                }
            })
            .collect();
        let mut out = Vec::with_capacity(cands.len());
        for params in cands {
            // Reset the evaluated planes' unit rectangles (zero
            // strengths leave blocks unwritten).
            for &p in &planes {
                let w = &mut wins[p];
                for y in w.uy0..w.uy1 {
                    let (a, b) = (y * w.cols + w.ux0, y * w.cols + w.ux1);
                    w.dst[a..b].copy_from_slice(&w.src[a..b]);
                }
            }
            let ctx = CdefFrameContext {
                mi_rows: mi_rows - r_off,
                mi_cols: mi_cols - c_off,
                num_planes,
                bit_depth: self.inp.bit_depth,
                subsampling_x: self.inp.subsampling_x,
                subsampling_y: self.inp.subsampling_y,
                cdef_params: params,
                cdef_idx: &|_, _| 0,
                skip: &|r, c| self.inp.mirror.skip_at_mi(r + r_off, c + c_off),
            };
            {
                let mut srcs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(3);
                let mut dsts: Vec<PlaneBuffer<'_>> = Vec::with_capacity(3);
                for w in wins.iter_mut() {
                    srcs.push(PlaneBuffer {
                        rows: w.rows as u32,
                        cols: w.cols as u32,
                        samples: &mut w.src,
                    });
                    dsts.push(PlaneBuffer {
                        rows: w.rows as u32,
                        cols: w.cols as u32,
                        samples: &mut w.dst,
                    });
                }
                for &(r, c, dir) in &self.blocks[k] {
                    cdef_block_dir(
                        &ctx,
                        &srcs,
                        &mut dsts,
                        num_planes,
                        r - r_off,
                        c - c_off,
                        0,
                        dir,
                    );
                }
            }
            let mut ssd = 0u64;
            for &p in &planes {
                let w = &wins[p];
                let (pw, _, _, _) = self.plane_dims(p);
                let src = self.src_plane(p);
                for y in w.uy0..w.uy1 {
                    let drow = &w.dst[y * w.cols + w.ux0..y * w.cols + w.ux1];
                    let srow = &src[(w.oy + y) * pw + w.ox + w.ux0..(w.oy + y) * pw + w.ox + w.ux1];
                    for (a, b) in drow.iter().zip(srow) {
                        let d = i64::from(*a) - i64::from(*b);
                        ssd += (d * d) as u64;
                    }
                }
            }
            out.push(ssd);
        }
        out
    }
}

/// Per-thread scratch (reserved).
#[derive(Default)]
struct Scratch {}

/// The frame-level + per-unit CDEF election. `recon_*` are the
/// committed pre-CDEF reconstruction planes (the §7.14 deblock levels
/// this encoder codes are 0, so the reconstruction IS the CDEF
/// input). Returns the winning plan — NOT yet applied — or `None`
/// when nothing beat the unfiltered frame under `D + λ·R`. The
/// caller applies via [`apply_cdef_plan`] (and, for
/// `params.cdef_bits > 0`, first re-emits the tile with the plan's
/// per-leaf ids and settles the final arm on exact realized bytes —
/// see [`CdefElection`]).
pub(crate) fn elect_cdef(inp: &CdefElectInput<'_>) -> Option<CdefElection> {
    let mi_rows = inp.mirror.mi_rows();
    let mi_cols = inp.mirror.mi_cols();
    let sb_rows = mi_rows.div_ceil(16) as usize;
    let sb_cols = mi_cols.div_ceil(16) as usize;
    let n_units = sb_rows * sb_cols;

    // Which units carry a §5.11.56 idx on the wire: the committed
    // grid stamped the anchor (any non-skip block exists) — `-1`
    // anchors are all-skip and stay on the decoder's copy path.
    let coded: Vec<bool> = {
        let grid = inp.mirror.cdef_idx();
        (0..n_units)
            .map(|k| {
                let (ur, uc) = ((k / sb_cols) as u32, (k % sb_cols) as u32);
                grid[((ur * 16) * mi_cols + uc * 16) as usize] != -1
            })
            .collect()
    };
    let coded_count = coded.iter().filter(|&&c| c).count() as u64;
    if coded_count == 0 {
        return None;
    }

    let params_for = |damping: u8, y: Strength, uv: Strength| -> CdefParams {
        let mut p = CdefParams::short_circuit();
        p.short_circuited = false;
        p.cdef_damping = damping;
        p.cdef_bits = 0;
        p.cdef_y_pri_strength[0] = y.pri;
        p.cdef_y_sec_strength[0] = y.sec;
        p.cdef_uv_pri_strength[0] = uv.pri;
        p.cdef_uv_sec_strength[0] = uv.sec;
        p
    };
    let engine = UnitEngine::new(inp, &coded);

    // Unfiltered per-unit baselines.
    let (base_y_units, base_uv_units) = engine.base_units();
    let base_total: u64 = base_y_units.iter().sum::<u64>() + base_uv_units.iter().sum::<u64>();

    // The coarse ladder per plane set (`which` 0 = luma, 1 = chroma).
    let coarse: Vec<Strength> = {
        let mut v = Vec::new();
        for pri in [1u8, 2, 3, 4, 6, 9, 12, 15] {
            for sec in [0u8, 2] {
                v.push(Strength { pri, sec });
            }
        }
        // Secondary-only candidates (legal stored sec ∈ {1, 2, 4}).
        for sec in [1u8, 2, 4] {
            v.push(Strength { pri: 0, sec });
        }
        v
    };
    let refine_of = |center: Strength| -> Vec<Strength> {
        let mut v = Vec::new();
        for pri in center.pri.saturating_sub(1)..=(center.pri + 1).min(15) {
            for sec in [0u8, 1, 2, 4] {
                v.push(Strength { pri, sec });
            }
        }
        v
    };
    let cand_params = |damping: u8, which: u8, s: Strength| -> CdefParams {
        if which == 0 {
            params_for(damping, s, ZERO)
        } else {
            params_for(damping, ZERO, s)
        }
    };
    // Per-damping tables + the r428 frame-level winner.
    let dampings = [3u8, 5];
    let mut tables: Vec<(SetTables, SetTables)> = Vec::new();
    let has_chroma = inp.num_planes > 1;
    // Append the per-unit SSD rows of `cands` (at `damping`, plane
    // set `which`) to `t`, skipping strengths already tabled.
    let extend_table =
        |t: &mut SetTables, damping: u8, which: u8, cands: &[Strength], subsample: bool| {
            let fresh: Vec<Strength> = cands.iter().copied().filter(|s| !t.cands.contains(s)).fold(
                Vec::new(),
                |mut acc, s| {
                    if !acc.contains(&s) {
                        acc.push(s);
                    }
                    acc
                },
            );
            if fresh.is_empty() {
                return;
            }
            let params: Vec<CdefParams> = fresh
                .iter()
                .map(|&s| cand_params(damping, which, s))
                .collect();
            let rows = engine.evaluate(&params, which, subsample);
            for (s, row) in fresh.into_iter().zip(rows) {
                t.cands.push(s);
                t.ssd.push(row);
            }
        };
    let empty_uv = |damping: u8| SetTables {
        cands: vec![ZERO],
        ssd: vec![base_uv_units.clone()],
        damping,
    };
    if !inp.fast {
        // The r429 election: coarse-then-refine on every unit, both
        // dampings.
        for &d in &dampings {
            let mut ty = SetTables {
                cands: vec![ZERO],
                ssd: vec![base_y_units.clone()],
                damping: d,
            };
            extend_table(&mut ty, d, 0, &coarse, false);
            let center = ty.cands[ty.best_by_total()];
            extend_table(&mut ty, d, 0, &refine_of(center), false);
            let mut tuv = empty_uv(d);
            if has_chroma {
                extend_table(&mut tuv, d, 1, &coarse, false);
                let center = tuv.cands[tuv.best_by_total()];
                extend_table(&mut tuv, d, 1, &refine_of(center), false);
            }
            tables.push((ty, tuv));
        }
    } else {
        // r464 Fast: coarse + refine on the unit subsample at both
        // dampings, then the best damping's top-3 per plane set on
        // every unit (one table).
        let sub_base_y: Vec<u64> = engine.subsample_row(&base_y_units);
        let sub_base_uv: Vec<u64> = engine.subsample_row(&base_uv_units);
        let mut best: Option<(u8, Vec<Strength>, Vec<Strength>)> = None;
        let mut best_total = u64::MAX;
        for &d in &dampings {
            let mut ty = SetTables {
                cands: vec![ZERO],
                ssd: vec![sub_base_y.clone()],
                damping: d,
            };
            extend_table(&mut ty, d, 0, &coarse, true);
            let center = ty.cands[ty.best_by_total()];
            extend_table(&mut ty, d, 0, &refine_of(center), true);
            let mut tuv = SetTables {
                cands: vec![ZERO],
                ssd: vec![sub_base_uv.clone()],
                damping: d,
            };
            if has_chroma {
                extend_table(&mut tuv, d, 1, &coarse, true);
                let center = tuv.cands[tuv.best_by_total()];
                extend_table(&mut tuv, d, 1, &refine_of(center), true);
            }
            let top = |t: &SetTables| -> Vec<Strength> {
                let mut order: Vec<usize> = (1..t.cands.len()).collect();
                order.sort_by_key(|&i| t.total(i));
                order.into_iter().take(3).map(|i| t.cands[i]).collect()
            };
            let total = t_best(&ty) + t_best(&tuv);
            if total < best_total {
                best_total = total;
                best = Some((d, top(&ty), top(&tuv)));
            }
        }
        let (d, top_y, top_uv) = best.expect("two dampings swept");
        let mut ty = SetTables {
            cands: vec![ZERO],
            ssd: vec![base_y_units.clone()],
            damping: d,
        };
        extend_table(&mut ty, d, 0, &top_y, false);
        let mut tuv = empty_uv(d);
        if has_chroma {
            extend_table(&mut tuv, d, 1, &top_uv, false);
        }
        tables.push((ty, tuv));
    }
    let dampings: Vec<u8> = tables.iter().map(|(ty, _)| ty.damping).collect();

    // Frame-level arm: best (y, uv) per damping by totals, then the
    // damping refinement {4, 6} on the winning strengths (full-frame
    // totals — the arm is pure distortion, R = 0).
    let (mut fl_damping, mut fl_y, mut fl_uv, mut fl_total) = (3u8, ZERO, ZERO, u64::MAX);
    for (di, &d) in dampings.iter().enumerate() {
        let (ty, tuv) = &tables[di];
        let (yi, uvi) = (ty.best_by_total(), tuv.best_by_total());
        let total = ty.total(yi) + tuv.total(uvi);
        if total < fl_total {
            (fl_damping, fl_y, fl_uv, fl_total) = (d, ty.cands[yi], tuv.cands[uvi], total);
        }
    }
    if (fl_y != ZERO || fl_uv != ZERO) && !inp.fast {
        let params: Vec<CdefParams> = [4u8, 6]
            .iter()
            .map(|&d| params_for(d, fl_y, fl_uv))
            .collect();
        let rows = engine.evaluate(&params, 2, false);
        for (&d, row) in [4u8, 6].iter().zip(rows) {
            let t: u64 = row.iter().sum();
            if t < fl_total {
                fl_total = t;
                fl_damping = d;
            }
        }
    }

    // Candidate plans: (D, R256, damping, set list, per-unit set
    // choice). Set = (y index, uv index) into the damping's tables.
    struct Plan {
        d: u64,
        r256: u64,
        damping: u8,
        sets: Vec<(usize, usize)>,
        choice: Vec<usize>,
        bits: u8,
        table: usize,
    }
    let mut plans: Vec<Plan> = Vec::new();

    // The unfiltered baseline and the frame-level arm as plans.
    plans.push(Plan {
        d: base_total,
        r256: 0,
        damping: 3,
        sets: Vec::new(),
        choice: Vec::new(),
        bits: 0,
        table: usize::MAX,
    });
    if fl_total < u64::MAX && (fl_y != ZERO || fl_uv != ZERO) {
        plans.push(Plan {
            d: fl_total,
            r256: 0,
            damping: fl_damping,
            sets: Vec::new(),
            choice: Vec::new(),
            bits: 0,
            table: usize::MAX - 1,
        });
    }

    // Per-unit arms: greedy set-list growth over the (y, uv) product
    // space at each swept damping; assignment = per-unit argmin;
    // exact rate = `cdef_bits` per coded unit + 6 (+6 chroma) header
    // bits per extra set.
    let set_hdr_bits: u64 = if inp.num_planes > 1 { 12 } else { 6 };
    for (di, &d) in dampings.iter().enumerate() {
        if inp.max_bits == 0 {
            break;
        }
        let (ty, tuv) = &tables[di];
        let unit_cost = |set: (usize, usize), k: usize| ty.ssd[set.0][k] + tuv.ssd[set.1][k];
        // Greedy seed: the best single set.
        let mut best_single = (0usize, 0usize);
        let mut best_single_d = u64::MAX;
        for yi in 0..ty.cands.len() {
            for uvi in 0..tuv.cands.len() {
                let t = ty.total(yi) + tuv.total(uvi);
                if t < best_single_d {
                    best_single_d = t;
                    best_single = (yi, uvi);
                }
            }
        }
        let mut sets = vec![best_single];
        let mut cur: Vec<u64> = (0..n_units).map(|k| unit_cost(best_single, k)).collect();
        for bits in 1..=inp.max_bits.min(3) {
            let want = 1usize << bits;
            while sets.len() < want {
                // The set whose addition reduces the assigned total most.
                let mut best_gain = 0u64;
                let mut best_set: Option<(usize, usize)> = None;
                for yi in 0..ty.cands.len() {
                    for uvi in 0..tuv.cands.len() {
                        if sets.contains(&(yi, uvi)) {
                            continue;
                        }
                        let gain: u64 = (0..n_units)
                            .filter(|&k| coded[k])
                            .map(|k| cur[k].saturating_sub(unit_cost((yi, uvi), k)))
                            .sum();
                        if gain > best_gain {
                            best_gain = gain;
                            best_set = Some((yi, uvi));
                        }
                    }
                }
                match best_set {
                    Some(s) => {
                        for (k, c) in cur.iter_mut().enumerate() {
                            if coded[k] {
                                *c = (*c).min(unit_cost(s, k));
                            }
                        }
                        sets.push(s);
                    }
                    // No remaining set helps: pad with the seed (the
                    // duplicate costs header bits and will lose to
                    // the smaller-bits plan on R).
                    None => sets.push(best_single),
                }
            }
            let choice: Vec<usize> = (0..n_units)
                .map(|k| {
                    if !coded[k] {
                        return 0;
                    }
                    (0..sets.len())
                        .min_by_key(|&s| unit_cost(sets[s], k))
                        .unwrap_or(0)
                })
                .collect();
            let dtot: u64 = (0..n_units)
                .map(|k| {
                    if coded[k] {
                        unit_cost(sets[choice[k]], k)
                    } else {
                        base_y_units[k] + base_uv_units[k]
                    }
                })
                .sum();
            let r256 = 256 * (u64::from(bits) * coded_count + ((1u64 << bits) - 1) * set_hdr_bits);
            plans.push(Plan {
                d: dtot,
                r256,
                damping: d,
                sets: sets.clone(),
                choice,
                bits,
                table: di,
            });
        }
    }

    // `D·256 + λ·R256` — the twin ladders' exact scale.
    if std::env::var_os("OXIDEAV_AV1_CDEF_DEBUG").is_some() {
        eprintln!(
            "cdef-elect: units {n_units} ({coded_count} coded), lambda {}, base D {base_total}",
            inp.lambda
        );
        for p in &plans {
            let tag = match p.table {
                usize::MAX => "base".to_string(),
                t if t == usize::MAX - 1 => {
                    format!(
                        "frame-level d{fl_damping} y({},{}) uv({},{})",
                        fl_y.pri, fl_y.sec, fl_uv.pri, fl_uv.sec
                    )
                }
                t => {
                    let (ty, tuv) = &tables[t];
                    let sets: Vec<String> = p
                        .sets
                        .iter()
                        .map(|&(yi, uvi)| {
                            format!(
                                "y({},{})uv({},{})",
                                ty.cands[yi].pri,
                                ty.cands[yi].sec,
                                tuv.cands[uvi].pri,
                                tuv.cands[uvi].sec
                            )
                        })
                        .collect();
                    format!("bits{} d{} [{}]", p.bits, p.damping, sets.join(" "))
                }
            };
            eprintln!(
                "cdef-elect:   D {} R256 {} score {} <- {tag}",
                p.d,
                p.r256,
                crate::encoder::rate_twin::score256(p.d, inp.lambda, p.r256)
            );
        }
    }
    let winner = plans
        .into_iter()
        .min_by_key(|p| crate::encoder::rate_twin::score256(p.d, inp.lambda, p.r256))?;
    if winner.table == usize::MAX {
        return None; // the unfiltered baseline won
    }
    // The frame-level arm as a plan (present iff it beats the
    // unfiltered frame — the r428 election condition).
    let frame_level: Option<CdefPlan> = if (fl_y != ZERO || fl_uv != ZERO) && fl_total < base_total
    {
        Some(CdefPlan {
            params: params_for(fl_damping, fl_y, fl_uv),
            unit_idx: coded.iter().map(|&c| if c { 0 } else { -1 }).collect(),
            d: fl_total,
        })
    } else {
        None
    };
    if winner.table == usize::MAX - 1 {
        // Frame-level arm won outright — no exact-bytes settlement
        // needed (zero tile bits, header size equals the default's).
        return frame_level.map(|best| CdefElection {
            best,
            frame_level: None,
            base_d: base_total,
        });
    }
    // Per-unit arm.
    let (ty, tuv) = &tables[winner.table];
    let mut params = CdefParams::short_circuit();
    params.short_circuited = false;
    params.cdef_damping = winner.damping;
    params.cdef_bits = winner.bits;
    for (i, &(yi, uvi)) in winner.sets.iter().enumerate() {
        params.cdef_y_pri_strength[i] = ty.cands[yi].pri;
        params.cdef_y_sec_strength[i] = ty.cands[yi].sec;
        params.cdef_uv_pri_strength[i] = tuv.cands[uvi].pri;
        params.cdef_uv_sec_strength[i] = tuv.cands[uvi].sec;
    }
    let unit_idx: Vec<i8> = (0..n_units)
        .map(|k| if coded[k] { winner.choice[k] as i8 } else { -1 })
        .collect();
    Some(CdefElection {
        best: CdefPlan {
            params,
            unit_idx,
            d: winner.d,
        },
        frame_level,
        base_d: base_total,
    })
}

/// Apply an elected plan to the reconstruction (the §7.20 reference
/// store the decoder will hold after decoding this frame): the §7.15
/// filter over the plan's per-unit grid through the decoder's own
/// kernels. The caller must have re-emitted the tile first when
/// `params.cdef_bits > 0` (the write mirror's grid then equals
/// `plan.unit_idx` — asserted by the callers).
///
/// r464 — runs unit by unit on local windows ([`UnitEngine`]),
/// `threads`-wide, writing each unit's filtered rectangle into one
/// plane-sized output at a time (no frame-sized `i32` copies).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_cdef_plan(
    mirror: &PartitionWalker,
    plan: &CdefPlan,
    input: &YuvFrame,
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
    threads: usize,
) {
    let inp = CdefElectInput {
        mirror,
        input,
        recon_y,
        recon_u,
        recon_v,
        width,
        height,
        chroma_w,
        chroma_h,
        bit_depth,
        subsampling_x,
        subsampling_y,
        num_planes,
        lambda: 0,
        max_bits: 0,
        fast: false,
        threads,
    };
    let n_units = mirror.mi_rows().div_ceil(16) as usize * mirror.mi_cols().div_ceil(16) as usize;
    let coded: Vec<bool> = (0..n_units).map(|k| plan.unit_idx[k] >= 0).collect();
    let engine = UnitEngine::new(&inp, &coded);
    let units: Vec<usize> = (0..n_units)
        .filter(|&k| coded[k] && !engine.blocks[k].is_empty())
        .collect();
    // Per unit: the filtered rectangles of every plane (window
    // layout + the rectangle's plane-space origin).
    let filtered: Vec<Vec<FilteredRect>> = engine.map_units(&units, |eng, k, _| {
        eng.filter_unit(k, &plan.params, plan.unit_idx[k])
    });
    drop(engine);
    let outs: [&mut [u16]; 3] = [recon_y, recon_u, recon_v];
    let widths = [width, chroma_w, chroma_w];
    for (ui, _) in units.iter().enumerate() {
        for (p, (x0, y0, w, h, samples)) in filtered[ui].iter().enumerate() {
            if p >= num_planes as usize {
                break;
            }
            let pw = widths[p];
            for yy in 0..*h {
                let dst = &mut outs[p][(y0 + yy) * pw + x0..(y0 + yy) * pw + x0 + w];
                for (d, &v) in dst.iter_mut().zip(&samples[yy * w..(yy + 1) * w]) {
                    *d = v.max(0) as u16;
                }
            }
        }
    }
}
