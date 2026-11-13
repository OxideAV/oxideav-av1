//! r481 — the §7.4 in-loop passes (§7.14 deblocking, §7.15 CDEF,
//! §7.17 loop restoration) run 64-luma-row band by band BEHIND the tile
//! walk, and the §7.20 store rows published as soon as they are final,
//! so a later frame (frame-parallel decode) can motion-compensate from
//! the top of this frame while its bottom is still being decoded.
//!
//! Every pass computes each sample from exactly the inputs the
//! whole-frame order gives it. Work proceeds in units of two mi rows
//! (8 luma rows; chroma `>> subY`). With `W` the luma rows the walk
//! has completed (every row once the frame's walk ends):
//!
//! * The walk reads `CurrFrame` only at the current superblock and its
//!   left / above / above-right neighbours (the §7.11.2 edges: the row
//!   just above a superblock row, `W - 1`; for chroma the plane row
//!   above, luma rows `W - 2 ..= W - 1`). So a unit of rows that ends
//!   at least 4 luma rows above `W` is never read by the walk again.
//! * §7.14 pass 0 (vertical edges) moves samples only within their own
//!   rows: a unit is filtered once it ends 4 rows above `W`.
//! * §7.14 pass 1 (horizontal edges `y`) reads rows `y - 7 ..= y + 6`
//!   (§7.14.6.2, after pass 0; chroma 4 plane rows each side) and
//!   writes rows `y - 6 ..= y + 5` (§7.14.6.4; chroma 3 plane rows):
//!   a unit is filtered once pass 0 covers 12 rows below its first
//!   edge row. Edges of one pass never interact (see
//!   `loop_filter_rows_window`), so the order is free. After pass 1 of
//!   the units below row `P`, rows below `P - 8` are deblocked for
//!   good (no later edge writes more than 6 luma rows above itself).
//! * The §7.17.6 pre-CDEF boundary rows of stripe boundary `s` (luma
//!   rows `64 s - 10 ..= 64 s - 7`, chroma plane rows `32 s - 6 ..=
//!   32 s - 3` at 4:2:0) are captured once deblocked for good and
//!   before CDEF writes them.
//! * §7.15 CDEF of a unit (one row of 8×8 luma blocks and their chroma
//!   blocks) reads deblocked samples 2 plane rows around it through
//!   its rolling window and writes only its own rows: it runs once
//!   rows to 8 luma rows below it are deblocked for good and every
//!   boundary inside it is captured. The window keeps the unfiltered
//!   rows the next unit shares with this one.
//! * §7.17 stripe `s` (luma rows `64 s - 8 .. 64 s + 56`) reads CDEF
//!   samples inside the stripe and the captured boundary rows `s`,
//!   `s + 1`, and writes inside the stripe: it runs once CDEF covers
//!   the stripe (the CDEF window below the stripe was built before).
//! * Rows are final below the least of: `W`, the deblocked-for-good
//!   bound (deblocking on), the CDEF bound (CDEF on), `64 l - 8` after
//!   `l` stripes (restoration on).

use super::ref_planes::RefPlanesWriter;
use crate::Error;
use crate::cdef::CdefInPlace;
use crate::cdf::PartitionWalker;
use crate::loop_filter::{LfEdgeTable, LfLevelParams, PlaneBuffer, loop_filter_rows_window_table};
use crate::loop_restoration::{
    PlaneStripe, StripeBoundaryRows, plane_stripes, restore_stripe_in_place,
};
use crate::uncompressed_header_tail::{CdefParams, LoopFilterParams, LrParams, SegmentationParams};

/// Mi rows per work unit (8 luma rows).
const UNIT_MI: u32 = 2;
/// Luma rows per §7.17 stripe step.
const STRIPE_ROWS: u32 = 64;
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
    boundary: Vec<StripeBoundaryRows<u16>>,
    plane_end_y: Vec<i32>,
    /// Per plane: its stripe-boundary count.
    boundaries: Vec<usize>,
    /// Stripe count (over every plane).
    stripe_count: usize,
    copy: Vec<u16>,
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
    lf: Option<LfStage>,
    cdef: Option<(&'a CdefParams, CdefInPlace<u16>)>,
    lr: Option<LrStage<'a>>,
    /// Mi rows walked / through §7.14 pass 0 / pass 1 / §7.15.
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
                    StripeBoundaryRows::<u16>::boundary_count(end, if p == 0 { 0 } else { sub_y })
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
        let walked = mi_rows_done.min(self.mi_rows);
        if walked <= self.walked && walked < self.mi_rows {
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

    fn run(
        &mut self,
        w: &PartitionWalker,
        planes: &mut [PlaneBuffer<'_, u16>],
    ) -> Result<(), Error> {
        if planes.len() < self.num_planes {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let mi_rows = self.mi_rows;
        let all_walked = self.walked == mi_rows;
        let walked_rows = if all_walked { ALL } else { self.walked * 4 };
        // §7.14 deblocking.
        let db_rows = if self.lf.is_some() {
            if let Some(lf) = self.lf.as_mut()
                && lf.filled < self.walked
            {
                w.fill_lf_edge_table(&mut lf.table, &lf.lp, lf.filled, self.walked);
                lf.filled = self.walked;
            }
            // Pass 0: units ending at least 4 rows above the walk.
            let p0_to = if all_walked {
                mi_rows
            } else {
                self.walked.saturating_sub(1) / UNIT_MI * UNIT_MI
            };
            while self.p0 < p0_to {
                let end = (self.p0 + UNIT_MI).min(mi_rows);
                self.deblock_rows(planes, 0, self.p0, end);
                self.p0 = end;
            }
            // Pass 1: units whose edges' reach (12 rows below the
            // unit's first edge row) pass 0 covers.
            while self.p1 < self.p0 && (self.p0 == mi_rows || self.p1 + UNIT_MI < self.p0) {
                let end = (self.p1 + UNIT_MI).min(mi_rows);
                self.deblock_rows(planes, 1, self.p1, end);
                self.p1 = end;
            }
            if self.p1 == mi_rows {
                ALL
            } else {
                (self.p1 * 4).saturating_sub(8)
            }
        } else {
            walked_rows
        };
        // §7.17.6 boundary rows, ahead of CDEF.
        let mut max_bnd = 0usize;
        if let Some(lr) = self.lr.as_mut() {
            max_bnd = lr.boundaries.iter().copied().max().unwrap_or(0);
            while self.captured < max_bnd {
                let s = self.captured as u32 + 1;
                if db_rows != ALL && db_rows < (STRIPE_ROWS * s).saturating_sub(4) {
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
        }
        // §7.15 CDEF, one row of 8×8 blocks at a time.
        let cdef_rows = if let Some((params, stepper)) = self.cdef.as_mut() {
            loop {
                let r0 = self.cdef_done;
                if r0 >= mi_rows {
                    break;
                }
                let r1 = (r0 + UNIT_MI).min(mi_rows);
                let src_ready = db_rows == ALL || db_rows >= r1 * 4 + 8;
                // Every boundary whose rows (luma 64 s - 10 ..= 64 s - 7,
                // chroma down to 64 s - 12 in luma rows) meet the unit.
                let bnd_needed = (((r1 * 4 + 11) / STRIPE_ROWS) as usize).min(max_bnd);
                if !src_ready || self.captured < bnd_needed {
                    break;
                }
                w.with_cdef_ctx(
                    params,
                    self.num_planes as u8,
                    self.bit_depth,
                    self.sub_x,
                    self.sub_y,
                    |ctx| stepper.band(ctx, planes, r0, r1),
                );
                self.cdef_done = r1;
            }
            if self.cdef_done == mi_rows {
                ALL
            } else {
                self.cdef_done * 4
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
            while self.lr_done < lr.stripe_count {
                let s = self.lr_done as u32;
                let stripe_end = STRIPE_ROWS * s + 56;
                if (cdef_rows != ALL && cdef_rows < stripe_end)
                    || self.captured < (s as usize + 1).min(max_bnd)
                {
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
                            for stripe in plane_stripes.iter().filter(|st| st.stripe == s as usize)
                            {
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
                (self.lr_done as u32 * STRIPE_ROWS).saturating_sub(8)
            }
        } else {
            cdef_rows
        };
        let final_rows = walked_rows.min(db_rows).min(cdef_rows).min(lr_rows);
        self.store(planes, final_rows);
        Ok(())
    }

    fn deblock_rows(&self, planes: &mut [PlaneBuffer<'_, u16>], pass: u8, r0: u32, r1: u32) {
        let Some(lf) = self.lf.as_ref() else {
            return;
        };
        for (plane, pb) in planes.iter_mut().enumerate().take(self.num_planes) {
            loop_filter_rows_window_table(&lf.table, plane as u8, pass, pb, 0, r0, r1);
        }
    }

    /// Copy the rows `[ stored, final_rows )` (cropped) into the store
    /// and publish them.
    fn store(&mut self, planes: &[PlaneBuffer<'_, u16>], final_rows: u32) {
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
                row.copy_from_slice(&src.samples[y * cols..y * cols + pw]);
            }
        }
        self.stored = final_rows;
        self.writer.publish(final_rows);
    }
}
