//! r481 — the §7.4 in-loop passes (§7.14 deblocking, §7.15 CDEF,
//! §7.17 loop restoration) run 64-luma-row band by band BEHIND the tile
//! walk, and the §7.20 store rows published as soon as they are final,
//! so a later frame (frame-parallel decode) can motion-compensate from
//! the top of this frame while its bottom is still being decoded.
//!
//! Every pass computes each sample from exactly the inputs the
//! whole-frame order gives it. With band `b` = luma rows `[ 64 b,
//! 64 b + 64 )` (chroma `>> subY`) and `walked` the bands the walk has
//! completed (all of them once the frame's walk ends):
//!
//! * §7.14 pass 0 of band `b` (vertical edges, which move samples only
//!   within their own rows) runs once band `b + 1` is walked: the walk
//!   reads `CurrFrame` only at the current superblock and its left /
//!   above / above-right neighbours (§7.11.2 edges, the row just above
//!   a superblock row), so no later block reads band `b` once band
//!   `b + 1` is reconstructed.
//! * §7.14 pass 1 of band `b` (horizontal edges `y` in the band) reads
//!   rows `y - 7 ..= y + 6` (§7.14.6.2, after pass 0) and writes rows
//!   `y - 6 ..= y + 5` (§7.14.6.4), so it runs once pass 0 covers band
//!   `b + 1`; edges of one pass never interact (see
//!   `loop_filter_rows_window`), so the band order is free. A row is
//!   deblocked for good once pass 1 has run on its band and the band
//!   below, except for the 6 rows above that band's first edge: rows
//!   below `64 p1 - 8` are final after `p1` bands.
//! * The §7.17.6 pre-CDEF boundary rows of stripe boundary `s` (luma
//!   rows `64 s - 10 ..= 64 s - 7`, chroma `>> subY`, inside band
//!   `s - 1`) are captured once deblocked for good and before CDEF
//!   writes band `s - 1`.
//! * §7.15 CDEF of band `b` reads deblocked rows `64 b - 2 .. 64 b + 66`
//!   (§7.15.3 taps reach 2) through its rolling window and writes only
//!   band `b`, so it runs once rows below `64 b + 66` are deblocked for
//!   good (and band `b + 1` walked) and boundary `b + 1` is captured.
//! * §7.17 stripe `s` (luma rows `64 s - 8 .. 64 s + 56`) reads CDEF
//!   samples inside the stripe and the captured boundary rows `s`,
//!   `s + 1`, and writes inside the stripe, so it runs once CDEF covers
//!   band `s` (the window of band `s` already holds the unfiltered rows
//!   it shares with the stripe).
//! * Rows are final below the least of: the walked rows, `64 p1 - 8`
//!   (deblocking on), `64 c` after `c` CDEF bands (CDEF on), `64 l - 8`
//!   after `l` stripes (restoration on).

use super::ref_planes::RefPlanesWriter;
use crate::Error;
use crate::cdef::CdefInPlace;
use crate::cdf::PartitionWalker;
use crate::loop_filter::{LfEdgeTable, LfLevelParams, PlaneBuffer, loop_filter_rows_window_table};
use crate::loop_restoration::{
    PlaneStripe, StripeBoundaryRows, plane_stripes, restore_stripe_in_place,
};
use crate::uncompressed_header_tail::{CdefParams, LoopFilterParams, LrParams, SegmentationParams};

/// Mi rows per band (64 luma rows).
const BAND_MI: u32 = 16;
/// Luma rows per band.
const BAND_ROWS: u32 = 64;
/// "Every row" in the final-row bounds.
const ALL: u32 = u32::MAX;

struct LfStage {
    table: LfEdgeTable,
    lp: LfLevelParams,
    /// Mi rows of `table` filled from the walker's grids.
    filled: u32,
}

struct LrStage<'a> {
    params: &'a LrParams,
    /// Per plane: the stripes carrying restoration (lazily built).
    stripes: Option<Vec<Vec<PlaneStripe>>>,
    boundary: Vec<StripeBoundaryRows>,
    plane_end_y: Vec<i32>,
    /// Per plane: its stripe-boundary count.
    boundaries: Vec<usize>,
    /// Stripe count (over every plane).
    stripe_count: usize,
    copy: Vec<i32>,
}

/// The in-loop passes and §7.20 store of one frame, advanced as its
/// walk completes rows.
pub(super) struct InLoopPipeline<'a> {
    num_planes: usize,
    bit_depth: u8,
    sub_x: u8,
    sub_y: u8,
    mi_rows: u32,
    mi_cols: u32,
    frame_height: u32,
    upscaled_width: u32,
    bands: u32,
    lf: Option<LfStage>,
    cdef: Option<(&'a CdefParams, CdefInPlace)>,
    lr: Option<LrStage<'a>>,
    walked: u32,
    p0: u32,
    p1: u32,
    captured: usize,
    cdef_done: u32,
    lr_done: usize,
    /// Luma rows stored and published.
    stored: u32,
    writer: RefPlanesWriter,
}

impl<'a> InLoopPipeline<'a> {
    /// `lf` / `cdef` / `lr` are the passes the frame runs (`None` =
    /// off); `writer` receives the cropped `UpscaledWidth ×
    /// FrameHeight` store (no superres on this path).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        num_planes: usize,
        bit_depth: u8,
        sub_x: u8,
        sub_y: u8,
        mi_rows: u32,
        mi_cols: u32,
        frame_width: u32,
        frame_height: u32,
        lf: Option<(&LoopFilterParams, &SegmentationParams, bool)>,
        cdef: Option<&'a CdefParams>,
        lr: Option<&'a LrParams>,
        writer: RefPlanesWriter,
    ) -> Self {
        let lf = lf.map(|(lf, sp, delta_lf_multi)| LfStage {
            table: LfEdgeTable::with_geometry(
                mi_rows,
                mi_cols,
                bit_depth,
                sub_x,
                sub_y,
                frame_width,
                frame_height,
                lf.loop_filter_level,
                lf.loop_filter_sharpness,
            ),
            lp: PartitionWalker::lf_level_params(lf, sp, delta_lf_multi),
            filled: 0,
        });
        let lr = lr.map(|params| {
            let plane_end_y: Vec<i32> = (0..num_planes)
                .map(|p| {
                    let sy = if p == 0 { 0 } else { u32::from(sub_y) };
                    ((frame_height + sy) >> sy) as i32 - 1
                })
                .collect();
            let boundaries = plane_end_y
                .iter()
                .enumerate()
                .map(|(p, &end)| {
                    StripeBoundaryRows::boundary_count(end, if p == 0 { 0 } else { sub_y })
                })
                .collect();
            LrStage {
                params,
                stripes: None,
                boundary: vec![StripeBoundaryRows::default(); num_planes],
                plane_end_y,
                boundaries,
                stripe_count: 0,
                copy: Vec::new(),
            }
        });
        Self {
            num_planes,
            bit_depth,
            sub_x,
            sub_y,
            mi_rows,
            mi_cols,
            frame_height,
            upscaled_width: frame_width,
            bands: mi_rows.div_ceil(BAND_MI),
            lf,
            cdef: cdef.map(|c| (c, CdefInPlace::new())),
            lr,
            walked: 0,
            p0: 0,
            p1: 0,
            captured: 0,
            cdef_done: 0,
            lr_done: 0,
            stored: 0,
            writer,
        }
    }

    /// The walk has completed every mi row below `mi_rows_done`:
    /// run every pass that became possible and publish the rows that
    /// became final.
    pub(super) fn advance(
        &mut self,
        walker: &mut PartitionWalker,
        mi_rows_done: u32,
    ) -> Result<(), Error> {
        let walked = if mi_rows_done >= self.mi_rows {
            self.bands
        } else {
            mi_rows_done / BAND_MI
        };
        if walked <= self.walked && walked < self.bands {
            return Ok(());
        }
        self.walked = self.walked.max(walked);
        walker.with_curr_frame_planes(self.num_planes, |w, planes| self.run(w, planes))
    }

    /// The planes this pipeline publishes.
    pub(super) fn planes(&self) -> &std::sync::Arc<super::ref_planes::RefPlanes> {
        self.writer.planes()
    }

    /// The walk is complete: run every remaining pass, store and
    /// publish the whole frame.
    pub(super) fn finish(mut self, walker: &mut PartitionWalker) -> Result<(), Error> {
        self.advance(walker, self.mi_rows)?;
        if self.stored != ALL {
            return Err(Error::PartitionWalkOutOfRange);
        }
        self.writer.finish();
        Ok(())
    }

    fn sy(&self, plane: usize) -> u32 {
        if plane == 0 { 0 } else { u32::from(self.sub_y) }
    }

    fn run(&mut self, w: &PartitionWalker, planes: &mut [PlaneBuffer<'_>]) -> Result<(), Error> {
        if planes.len() < self.num_planes {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let nb = self.bands;
        let all_walked = self.walked == nb;
        let walked_rows = if all_walked {
            ALL
        } else {
            self.walked * BAND_ROWS
        };
        // §7.14 deblocking.
        let db_rows = if let Some(lf) = self.lf.as_mut() {
            let rows = (self.walked * BAND_MI).min(self.mi_rows);
            if lf.filled < rows {
                w.fill_lf_edge_table(&mut lf.table, &lf.lp, lf.filled, rows);
                lf.filled = rows;
            }
            let p0_to = if all_walked {
                nb
            } else {
                self.walked.saturating_sub(1)
            };
            while self.p0 < p0_to {
                self.deblock_band(planes, 0, self.p0);
                self.p0 += 1;
            }
            let p1_to = if self.p0 == nb {
                nb
            } else {
                self.p0.saturating_sub(1)
            };
            while self.p1 < p1_to {
                self.deblock_band(planes, 1, self.p1);
                self.p1 += 1;
            }
            if self.p1 == nb {
                ALL
            } else {
                (self.p1 * BAND_ROWS).saturating_sub(8)
            }
        } else {
            walked_rows
        };
        // §7.17.6 boundary rows, ahead of CDEF.
        let mut boundary_limit = usize::MAX;
        if let Some(lr) = self.lr.as_mut() {
            let max_bnd = lr.boundaries.iter().copied().max().unwrap_or(0);
            while self.captured < max_bnd {
                let s = self.captured as u32 + 1;
                if db_rows != ALL && db_rows < (BAND_ROWS * s).saturating_sub(4) {
                    break;
                }
                for (p, bnd) in lr.boundary.iter_mut().enumerate() {
                    if (s as usize) <= lr.boundaries[p] {
                        let sy = if p == 0 { 0 } else { self.sub_y };
                        bnd.push_next(&planes[p], lr.plane_end_y[p], sy);
                    }
                }
                self.captured += 1;
            }
            if self.captured < max_bnd {
                boundary_limit = self.captured;
            }
        }
        // §7.15 CDEF.
        let cdef_rows = if let Some((params, stepper)) = self.cdef.as_mut() {
            let mut to = if self.lf.is_some() {
                if self.p1 == nb {
                    nb
                } else {
                    self.p1.saturating_sub(2)
                }
            } else if all_walked {
                nb
            } else {
                self.walked.saturating_sub(2)
            };
            to = to.min(boundary_limit.min(nb as usize) as u32);
            while self.cdef_done < to {
                let b = self.cdef_done;
                let (r0, r1) = (b * BAND_MI, ((b + 1) * BAND_MI).min(self.mi_rows));
                w.with_cdef_ctx(
                    params,
                    self.num_planes as u8,
                    self.bit_depth,
                    self.sub_x,
                    self.sub_y,
                    |ctx| stepper.band(ctx, planes, r0, r1),
                );
                self.cdef_done += 1;
            }
            if self.cdef_done == nb {
                ALL
            } else {
                self.cdef_done * BAND_ROWS
            }
        } else {
            db_rows
        };
        // §7.17 loop restoration, stripe by stripe.
        let lr_rows = if let Some(lr) = self.lr.as_mut() {
            if lr.stripes.is_none() {
                let mut stripes = Vec::with_capacity(self.num_planes);
                w.with_lr_ctx(
                    lr.params,
                    self.num_planes as u8,
                    self.bit_depth,
                    self.sub_x,
                    self.sub_y,
                    self.mi_rows,
                    self.mi_cols,
                    self.frame_height,
                    self.upscaled_width,
                    |ctx| {
                        for p in 0..self.num_planes {
                            stripes.push(plane_stripes(ctx, p as u8));
                        }
                    },
                );
                lr.stripe_count = stripes
                    .iter()
                    .flat_map(|v| v.iter().map(|s| s.stripe + 1))
                    .max()
                    .unwrap_or(0);
                lr.stripes = Some(stripes);
            }
            let max_bnd = lr.boundaries.iter().copied().max().unwrap_or(0);
            while self.lr_done < lr.stripe_count {
                let s = self.lr_done;
                let source_ready = if self.cdef.is_some() {
                    cdef_rows == ALL || self.cdef_done as usize > s
                } else {
                    db_rows == ALL || db_rows >= BAND_ROWS * s as u32 + 56
                };
                if !source_ready || self.captured < (s + 1).min(max_bnd) {
                    break;
                }
                let stripes = lr.stripes.as_ref().expect("stripes built");
                let copy = &mut lr.copy;
                let boundary = &lr.boundary;
                w.with_lr_ctx(
                    lr.params,
                    self.num_planes as u8,
                    self.bit_depth,
                    self.sub_x,
                    self.sub_y,
                    self.mi_rows,
                    self.mi_cols,
                    self.frame_height,
                    self.upscaled_width,
                    |ctx| {
                        for (p, plane_stripes) in stripes.iter().enumerate() {
                            for stripe in plane_stripes.iter().filter(|st| st.stripe == s) {
                                restore_stripe_in_place(
                                    ctx,
                                    p as u8,
                                    stripe,
                                    &boundary[p],
                                    &mut planes[p],
                                    0,
                                    copy,
                                );
                            }
                        }
                    },
                );
                self.lr_done += 1;
            }
            if self.lr_done == lr.stripe_count {
                ALL
            } else {
                (self.lr_done as u32 * BAND_ROWS).saturating_sub(8)
            }
        } else {
            cdef_rows
        };
        let final_rows = walked_rows.min(db_rows).min(cdef_rows).min(lr_rows);
        self.store(planes, final_rows);
        Ok(())
    }

    fn deblock_band(&self, planes: &mut [PlaneBuffer<'_>], pass: u8, band: u32) {
        let Some(lf) = self.lf.as_ref() else {
            return;
        };
        let (r0, r1) = (band * BAND_MI, ((band + 1) * BAND_MI).min(self.mi_rows));
        for (plane, pb) in planes.iter_mut().enumerate().take(self.num_planes) {
            loop_filter_rows_window_table(&lf.table, plane as u8, pass, pb, 0, r0, r1);
        }
    }

    /// Copy the rows `[ stored, final_rows )` (cropped) into the store
    /// and publish them.
    fn store(&mut self, planes: &[PlaneBuffer<'_>], final_rows: u32) {
        // Below the frame's last luma row unless everything is final:
        // `published >> subY` must not claim the last chroma row of an
        // odd-height frame before it is.
        let luma_h = self.writer.planes().luma_height();
        let final_rows = if final_rows == ALL {
            ALL
        } else {
            final_rows.min(luma_h.saturating_sub(1))
        };
        if final_rows <= self.stored && final_rows != ALL {
            return;
        }
        if self.stored == ALL {
            return;
        }
        let dims: Vec<(u32, u32)> = self.writer.planes().dims().to_vec();
        for (p, &(pw, ph)) in dims.iter().enumerate().take(self.num_planes) {
            let sy = self.sy(p);
            let y0 = ((self.stored >> sy) as usize).min(ph as usize);
            let y1 = if final_rows == ALL {
                ph as usize
            } else {
                ((final_rows >> sy) as usize).min(ph as usize)
            };
            if y1 <= y0 {
                continue;
            }
            let src = &planes[p];
            let cols = src.cols as usize;
            let pw = pw as usize;
            let dst = self.writer.rows_mut(p, y0, y1);
            for (y, row) in (y0..y1).zip(dst.chunks_exact_mut(pw)) {
                for (d, &v) in row.iter_mut().zip(&src.samples[y * cols..y * cols + pw]) {
                    *d = v.max(0) as u16;
                }
            }
        }
        self.stored = final_rows;
        self.writer.publish(final_rows);
    }
}
