//! r421 — the search-side **rate twin**: true bit-accounting rate
//! costs for the encoder's `D + λ·R` elections.
//!
//! Every RD ladder in this encoder historically priced rate with
//! magnitude heuristics ([`super::key_frame::leaf_rate`] and friends).
//! This module replaces those proxies with the real thing: a shadow of
//! the tile's adaptive symbol-coder state — the §8.3.1 working CDFs
//! ([`TileCdfContext`]), the §5.11 neighbour-context mirror
//! ([`PartitionSyntaxWriter`]), and the §8.2.6 arithmetic-coder
//! `range` — that the search can run candidate symbol sequences
//! through WITHOUT emitting, reading off the exact fractional bit
//! cost each candidate would add to the tile payload.
//!
//! ## Why it can never desync from the writer
//!
//! The twin does not re-implement any syntax: pricing and committing
//! run the very same [`write_partition_tree_syntax`] /
//! [`write_block_syntax`] / [`write_partition_symbol`] functions the
//! emitting pass runs, only with a counting
//! [`SymbolWriter`] ([`SymbolWriter::new_counting`]) that tracks the
//! §8.2.6 `range` trajectory and renormalisation-bit count but keeps
//! no `low` accumulator. CDF adaptation (§8.3), context stamping
//! (§5.11.5 grid fills) and arm gating (§5.11.4 forced arms) are
//! therefore bit-for-bit the writer's own. The search threads one
//! twin per superblock — snapshotted from the LIVE writer state just
//! before the superblock's search — commits each decision as it is
//! made, and the driver `debug_assert!`s the committed twin equals
//! the writer's state after the superblock's real emission
//! ([`RateTwin::matches`]).
//!
//! ## Cost units
//!
//! All prices are in **1/256-bit** fixed point
//! ([`SymbolWriter::cost_bits256`]): exact renormalisation bits plus
//! the fractional `log2(range)` drift, i.e. exactly the bits the
//! emitting writer would append for the same symbols at the same
//! stream position. `D + λ·R` comparisons scale distortion by 256 to
//! match (see [`score256`]).

use core::cell::RefCell;

use crate::cdf::{TileCdfContext, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE};
use crate::encoder::partition_tree::{
    write_block_syntax, write_partition_symbol, write_partition_tree_syntax, PartitionSyntaxWriter,
    SyntaxBlock, SyntaxFrameParams, SyntaxNode, WriterScopeSnapshot,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::Error;

/// r421 — rate-model selector for the RD ladders, kept so the sweep
/// harnesses can measure twin-priced vs heuristic-priced elections on
/// the same inputs. Production entry points always use
/// [`RateModel::Twin`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateModel {
    /// Pre-r421 magnitude heuristics (`leaf_rate` / `tree_rate` /
    /// `p_leaf_rate`).
    Heuristic,
    /// Exact bit accounting through the [`RateTwin`].
    Twin,
}

/// `D + λ·R` in 1/256-bit-consistent units: distortion is scaled by
/// 256 so `rate_bits256` (from [`SymbolWriter::cost_bits256`]) keeps
/// its sub-bit precision under the SAME `λ` calibration the heuristic
/// integer-rate scores used.
#[inline]
pub(crate) fn score256(distortion: u64, lambda: u64, rate_bits256: u64) -> u64 {
    distortion * 256 + lambda * rate_bits256
}

/// The search-side shadow of one tile's live write state (see the
/// module docs for the desync argument): the rate twin: a private copy of the tile's live
/// entropy state (`cdfs` + writer mirror + arithmetic range) that
/// candidate symbol sequences are priced against.
///
/// r460 — the live state sits behind a `RefCell` so `&self` pricing
/// entries can trial-write a candidate onto it and roll the block's
/// scope back through
/// [`PartitionSyntaxWriter::snapshot_price_scope`] — O(block) per
/// candidate instead of the pre-r460 whole-frame clone (O(frame) per
/// priced transform unit, quadratic over a picture: a 640×480 KEY
/// frame spent two thirds of its wall clock in that clone).
/// Deep [`Clone`] is still available for callers that branch the
/// whole twin.
#[derive(Debug, Clone)]
pub(crate) struct RateTwin {
    inner: RefCell<TwinInner>,
    disable_cdf_update: bool,
}

#[derive(Debug, Clone)]
struct TwinInner {
    cdfs: TileCdfContext,
    state: PartitionSyntaxWriter,
    range: u32,
}

/// r460 — a frozen block-scoped branch of a [`RateTwin`]: the writer
/// scope, the adapted CDF tables and the arithmetic range after (or
/// before) a trial commit. [`RateTwin::restore`] moves the live twin
/// onto it in O(block) + one CDF-table copy.
#[derive(Debug, Clone)]
pub(crate) struct TwinScope {
    writer: WriterScopeSnapshot,
    cdfs: TileCdfContext,
    range: u32,
}

impl RateTwin {
    /// Fork the live tile state into a twin.
    pub fn snapshot(
        cdfs: &TileCdfContext,
        state: &PartitionSyntaxWriter,
        writer: &SymbolWriter,
    ) -> Self {
        Self {
            inner: RefCell::new(TwinInner {
                cdfs: cdfs.clone(),
                state: state.clone(),
                range: writer.range(),
            }),
            disable_cdf_update: writer.disable_cdf_update(),
        }
    }

    /// Re-arm the §5.11.2 `ReadDeltas` write-side twin (superblock
    /// entry).
    pub fn arm_read_deltas(&mut self) {
        self.inner.get_mut().state.arm_read_deltas();
    }

    /// Whether the next block coded on this twin carries the §5.11.12
    /// / §5.11.13 delta syntax.
    pub fn deltas_pending(&self) -> bool {
        self.inner.borrow().state.deltas_pending()
    }

    /// r460 — capture the block scope `(r, c, b_size)` of the live
    /// twin (see [`TwinScope`]).
    pub fn scope(&self, r: u32, c: u32, b_size: usize, params: &SyntaxFrameParams) -> TwinScope {
        let g = self.inner.borrow();
        TwinScope {
            writer: g.state.snapshot_price_scope(
                r,
                c,
                NUM_4X4_BLOCKS_WIDE[b_size] as u32,
                NUM_4X4_BLOCKS_HIGH[b_size] as u32,
                params,
            ),
            cdfs: g.cdfs.clone(),
            range: g.range,
        }
    }

    /// r460 — move the live twin onto a captured [`TwinScope`].
    pub fn restore(&mut self, scope: &TwinScope) {
        let g = self.inner.get_mut();
        g.state.restore_price_scope(&scope.writer);
        g.cdfs.clone_from(&scope.cdfs);
        g.range = scope.range;
    }

    /// Commit a whole subtree's symbols (partition symbols + every
    /// leaf) and return the exact bits it costs.
    pub fn commit_subtree(
        &mut self,
        node: &SyntaxNode,
        r: u32,
        c: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
    ) -> Result<u64, Error> {
        let g = self.inner.get_mut();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, g.range);
        write_partition_tree_syntax(
            &mut w,
            &mut g.cdfs,
            &mut g.state,
            node,
            r,
            c,
            b_size,
            params,
        )?;
        g.range = w.range();
        Ok(w.cost_bits256())
    }

    /// Commit one §5.11.4 `partition` symbol.
    pub fn commit_partition_symbol(
        &mut self,
        partition: usize,
        r: u32,
        c: u32,
        b_size: usize,
    ) -> Result<u64, Error> {
        let g = self.inner.get_mut();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, g.range);
        write_partition_symbol(&mut w, &mut g.cdfs, &g.state, partition, r, c, b_size)?;
        g.range = w.range();
        Ok(w.cost_bits256())
    }

    /// Commit one leaf block's symbols.
    pub fn commit_block(
        &mut self,
        block: &SyntaxBlock,
        r: u32,
        c: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
    ) -> Result<u64, Error> {
        let g = self.inner.get_mut();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, g.range);
        write_block_syntax(
            &mut w,
            &mut g.cdfs,
            &mut g.state,
            block,
            r,
            c,
            b_size,
            params,
        )?;
        g.range = w.range();
        Ok(w.cost_bits256())
    }

    /// Price one leaf block WITHOUT committing it: the exact bits the
    /// emitting writer would append for it at the current state.
    pub fn price_block(
        &self,
        block: &SyntaxBlock,
        r: u32,
        c: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
    ) -> Result<u64, Error> {
        let mut g = self.inner.borrow_mut();
        let g = &mut *g;
        let scope = g.state.snapshot_price_scope(
            r,
            c,
            NUM_4X4_BLOCKS_WIDE[b_size] as u32,
            NUM_4X4_BLOCKS_HIGH[b_size] as u32,
            params,
        );
        let cdfs = g.cdfs.clone();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, g.range);
        let res = write_block_syntax(
            &mut w,
            &mut g.cdfs,
            &mut g.state,
            block,
            r,
            c,
            b_size,
            params,
        );
        g.state.restore_price_scope(&scope);
        g.cdfs = cdfs;
        res?;
        Ok(w.cost_bits256())
    }

    /// Price a §5.11.27 `motion_mode` symbol at the current CDF state.
    #[allow(clippy::too_many_arguments)]
    pub fn price_motion_mode(
        &self,
        motion_mode: u8,
        mi_size: usize,
        is_compound: bool,
        ref_frame: [i32; 2],
        y_mode: u8,
        num_samples: u32,
        is_motion_mode_switchable: bool,
        allow_warped_motion: bool,
        force_integer_mv: bool,
        gm_type: [i32; 8],
        is_scaled_per_ref: [bool; 7],
        has_overlappable: bool,
    ) -> Result<u64, Error> {
        let g = self.inner.borrow();
        let mut cdfs = g.cdfs.clone();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, g.range);
        crate::encoder::block_mode_info::write_motion_mode(
            &mut w,
            &mut cdfs,
            motion_mode,
            mi_size,
            0,
            is_compound,
            ref_frame,
            y_mode,
            num_samples,
            is_motion_mode_switchable,
            allow_warped_motion,
            force_integer_mv,
            gm_type,
            is_scaled_per_ref,
            has_overlappable,
        )?;
        Ok(w.cost_bits256())
    }

    /// Price the §5.11.23 inter-mode + MV prefix at the current CDF
    /// state.
    #[allow(clippy::too_many_arguments)]
    pub fn price_inter_mode(
        &self,
        ref_frame: [i32; 2],
        y_mode: u8,
        mv: [[i32; 2]; 2],
        ref_mv_idx: u32,
        mv_stack: &crate::cdf::FindMvStackResult,
        mi_size: usize,
        reference_select: bool,
        avail_u: bool,
        avail_l: bool,
        above_ref_frame: [i32; 2],
        left_ref_frame: [i32; 2],
        force_integer_mv: bool,
        allow_high_precision_mv: bool,
    ) -> Result<u64, Error> {
        let g = self.inner.borrow();
        let mut cdfs = g.cdfs.clone();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, g.range);
        crate::encoder::block_mode_info::write_inter_mode_mv_prefix(
            &mut w,
            &mut cdfs,
            ref_frame,
            y_mode,
            mv,
            ref_mv_idx,
            mv_stack,
            mi_size,
            /* skip_mode = */ 0,
            [0, 0],
            /* seg_ref_frame_active = */ false,
            0,
            /* seg_skip_active = */ false,
            /* seg_globalmv_active = */ false,
            reference_select,
            avail_u,
            avail_l,
            above_ref_frame[1] <= 0,
            left_ref_frame[1] <= 0,
            above_ref_frame[0] <= 0,
            left_ref_frame[0] <= 0,
            above_ref_frame,
            left_ref_frame,
            force_integer_mv,
            allow_high_precision_mv,
        )?;
        Ok(w.cost_bits256())
    }

    /// §5.11.9 spatial segment-id prediction at `(mi_row, mi_col)`.
    pub fn spatial_segment_pred(&self, mi_row: u32, mi_col: u32) -> u8 {
        self.inner.borrow().state.segment_pred_ctx(mi_row, mi_col).0
    }

    /// Parity check against the emitting writer's live state.
    pub fn matches(&self, cdfs: &TileCdfContext, writer: &SymbolWriter) -> bool {
        let g = self.inner.borrow();
        g.range == writer.range() && g.cdfs == *cdfs
    }

    /// r424 — open a per-transform-unit fork: TU decisions commit onto
    /// the fork progressively (so later TUs are priced with earlier
    /// ones' §5.11.39 contexts in place) and the whole fork is rolled
    /// back when it drops. r460 — the fork lives on this twin's state
    /// under a block scope instead of a whole-frame clone.
    pub fn tu_fork(&self) -> TuFork<'_> {
        TuFork {
            twin: self,
            origin: RefCell::new(None),
        }
    }
}

/// r424 — the running per-TU fork of a [`RateTwin`] (see
/// [`RateTwin::tu_fork`]).
pub(crate) struct TuFork<'a> {
    twin: &'a RateTwin,
    /// The block scope captured before the fork's first write; restored
    /// when the fork drops. `None` until the first price / commit.
    origin: RefCell<Option<TwinScope>>,
}

/// The per-leaf inputs a TU price needs (the §5.11.39 residual
/// facade's block-level fields).
pub(crate) struct TuCtx<'a> {
    pub params: &'a SyntaxFrameParams,
    pub mi_row: u32,
    pub mi_col: u32,
    pub mi_size: usize,
    pub base_x: u32,
    pub base_y: u32,
    pub is_inter: bool,
    pub segment_id: u8,
    pub y_mode: u8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: Option<u8>,
}

impl TuCtx<'_> {
    fn facade(&self, quant: &[i32], tx_type: u8) -> SyntaxBlock {
        let mut b = SyntaxBlock::skip_leaf(self.y_mode, None);
        b.segment_id = self.segment_id;
        b.use_filter_intra = u8::from(self.use_filter_intra);
        b.filter_intra_mode = self.filter_intra_mode;
        b.residual_quant = vec![quant.to_vec()];
        b.residual_tx_type = vec![tx_type];
        b
    }
}

impl TuFork<'_> {
    fn ensure_origin(&self, ctx: &TuCtx<'_>) {
        let mut o = self.origin.borrow_mut();
        if o.is_none() {
            *o = Some(
                self.twin
                    .scope(ctx.mi_row, ctx.mi_col, ctx.mi_size, ctx.params),
            );
        }
    }

    /// Price one luma TU at the fork's current state (earlier
    /// committed TUs' contexts in place) without committing it.
    pub fn price_luma_tu(
        &self,
        ctx: &TuCtx<'_>,
        tx_sz: usize,
        x: u32,
        y: u32,
        quant: &[i32],
        tx_type: u8,
    ) -> Result<u64, Error> {
        self.ensure_origin(ctx);
        let mut g = self.twin.inner.borrow_mut();
        let g = &mut *g;
        let scope = g.state.snapshot_price_scope(
            ctx.mi_row,
            ctx.mi_col,
            NUM_4X4_BLOCKS_WIDE[ctx.mi_size] as u32,
            NUM_4X4_BLOCKS_HIGH[ctx.mi_size] as u32,
            ctx.params,
        );
        let cdfs = g.cdfs.clone();
        let mut w = SymbolWriter::new_counting(self.twin.disable_cdf_update, g.range);
        let res = crate::encoder::partition_tree::write_single_transform_block(
            &mut w,
            &mut g.cdfs,
            &mut g.state,
            &ctx.facade(quant, tx_type),
            ctx.params,
            /* plane = */ 0,
            ctx.base_x,
            ctx.base_y,
            tx_sz,
            x,
            y,
            ctx.mi_row,
            ctx.mi_col,
            ctx.mi_size,
            ctx.is_inter,
        );
        g.state.restore_price_scope(&scope);
        g.cdfs = cdfs;
        res?;
        Ok(w.cost_bits256())
    }

    /// Commit one luma TU onto the fork.
    pub fn commit_luma_tu(
        &mut self,
        ctx: &TuCtx<'_>,
        tx_sz: usize,
        x: u32,
        y: u32,
        quant: &[i32],
        tx_type: u8,
    ) -> Result<(), Error> {
        self.ensure_origin(ctx);
        let mut g = self.twin.inner.borrow_mut();
        let g = &mut *g;
        let mut w = SymbolWriter::new_counting(self.twin.disable_cdf_update, g.range);
        crate::encoder::partition_tree::write_single_transform_block(
            &mut w,
            &mut g.cdfs,
            &mut g.state,
            &ctx.facade(quant, tx_type),
            ctx.params,
            /* plane = */ 0,
            ctx.base_x,
            ctx.base_y,
            tx_sz,
            x,
            y,
            ctx.mi_row,
            ctx.mi_col,
            ctx.mi_size,
            ctx.is_inter,
        )?;
        g.range = w.range();
        Ok(())
    }
}

impl Drop for TuFork<'_> {
    fn drop(&mut self) {
        if let Some(origin) = self.origin.get_mut().take() {
            let mut g = self.twin.inner.borrow_mut();
            g.state.restore_price_scope(&origin.writer);
            g.cdfs = origin.cdfs;
            g.range = origin.range;
        }
    }
}
