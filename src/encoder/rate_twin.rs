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

use crate::cdf::TileCdfContext;
use crate::encoder::partition_tree::{
    write_block_syntax, write_partition_symbol, write_partition_tree_syntax, PartitionSyntaxWriter,
    SyntaxBlock, SyntaxFrameParams, SyntaxNode,
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

/// The search-side shadow of one tile's live write state. See the
/// module docs for the desync argument.
#[derive(Debug, Clone)]
pub(crate) struct RateTwin {
    cdfs: TileCdfContext,
    state: PartitionSyntaxWriter,
    range: u32,
    disable_cdf_update: bool,
}

impl RateTwin {
    /// Snapshot the live writer state (call at superblock entry,
    /// BEFORE the superblock's search).
    pub fn snapshot(
        cdfs: &TileCdfContext,
        state: &PartitionSyntaxWriter,
        writer: &SymbolWriter,
    ) -> Self {
        Self {
            cdfs: cdfs.clone(),
            state: state.clone(),
            range: writer.range(),
            disable_cdf_update: writer.disable_cdf_update(),
        }
    }

    /// Mirror of [`PartitionSyntaxWriter::arm_read_deltas`] — call
    /// where the driver arms the real state (superblock entry) so the
    /// twin's §5.11.2 delta lifecycle stays in step.
    pub fn arm_read_deltas(&mut self) {
        self.state.arm_read_deltas();
    }

    /// r428 — the fork's §5.11.2 delta lifecycle bit: `true` while
    /// the next block committed into THIS fork is the one that codes
    /// the §5.11.13 deltas. The search's leaf builders consult it so
    /// exactly that block carries the superblock's delta value.
    pub fn deltas_pending(&self) -> bool {
        self.state.deltas_pending()
    }

    /// Commit the subtree rooted at `(r, c, b_size)` into the twin —
    /// advancing CDFs, neighbour mirror and `range` exactly as the
    /// emitting pass will — and return its exact cost in 1/256-bit
    /// units.
    pub fn commit_subtree(
        &mut self,
        node: &SyntaxNode,
        r: u32,
        c: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
    ) -> Result<u64, Error> {
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, self.range);
        write_partition_tree_syntax(
            &mut w,
            &mut self.cdfs,
            &mut self.state,
            node,
            r,
            c,
            b_size,
            params,
        )?;
        self.range = w.range();
        Ok(w.cost_bits256())
    }

    /// Commit ONLY the §5.11.4 partition symbol for `(r, c, b_size)`
    /// (possibly a zero-bit forced arm) — used by the split search
    /// path so child searches see the post-arm state — and return its
    /// exact cost.
    pub fn commit_partition_symbol(
        &mut self,
        partition: usize,
        r: u32,
        c: u32,
        b_size: usize,
    ) -> Result<u64, Error> {
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, self.range);
        write_partition_symbol(&mut w, &mut self.cdfs, &self.state, partition, r, c, b_size)?;
        self.range = w.range();
        Ok(w.cost_bits256())
    }

    /// Commit one leaf block's §5.11.5 syntax at `(r, c, b_size)` —
    /// block only, NO partition symbol. The multi-block partition
    /// shapes (HORZ / VERT / T-shapes / 4-strips) thread a running
    /// fork through this: each later block's search AND validation
    /// then see the earlier siblings' §5.11.5 mirror stamps, exactly
    /// like the writer (the §5.11.27 cascade in particular reads the
    /// committed neighbour grid).
    pub fn commit_block(
        &mut self,
        block: &SyntaxBlock,
        r: u32,
        c: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
    ) -> Result<u64, Error> {
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, self.range);
        write_block_syntax(
            &mut w,
            &mut self.cdfs,
            &mut self.state,
            block,
            r,
            c,
            b_size,
            params,
        )?;
        self.range = w.range();
        Ok(w.cost_bits256())
    }

    /// Price one leaf block's §5.11.5 syntax at `(r, c, b_size)`
    /// without touching the twin. Excludes the partition symbol
    /// (constant across the candidates of one leaf-level election, so
    /// it cancels in every comparison this price feeds).
    pub fn price_block(
        &self,
        block: &SyntaxBlock,
        r: u32,
        c: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
    ) -> Result<u64, Error> {
        let mut twin = self.clone();
        let mut w = SymbolWriter::new_counting(twin.disable_cdf_update, twin.range);
        write_block_syntax(
            &mut w,
            &mut twin.cdfs,
            &mut twin.state,
            block,
            r,
            c,
            b_size,
            params,
        )?;
        Ok(w.cost_bits256())
    }

    /// r421 — exact price (1/256-bit units) of the §5.11.27
    /// `motion_mode` arm for one candidate ordinal, through the
    /// writer's own arm derivation
    /// ([`crate::encoder::block_mode_info::write_motion_mode`] — the
    /// single source of truth for the short-circuit cascade and the
    /// arm-A/arm-B dispatch), against this twin's CURRENT adaptive
    /// CDFs. Zero for every forced-SIMPLE configuration, the exact
    /// `use_obmc` / `motion_mode` S() cost otherwise.
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
        let mut cdfs = self.cdfs.clone();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, self.range);
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

    /// r422 — exact price (1/256-bit units) of the §5.11.23 mode + MV
    /// prefix (the §5.11.25 reference cascade, the four-arm `YMode`
    /// dispatch, the `drl_mode` loop and the NEWMV `read_mv`
    /// differences) for one `(RefFrame, YMode, Mv, RefMvIdx)`
    /// candidate — through the writer's own emission path
    /// ([`crate::encoder::block_mode_info::write_inter_mode_mv_prefix`],
    /// the single body the committing pass also runs), against this
    /// twin's CURRENT adaptive CDFs. Replaces the pre-r422 constant
    /// per-mode bit proxies in the leaf's candidate election.
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
        let mut cdfs = self.cdfs.clone();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, self.range);
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

    /// r423 — the §5.11.9 spatial segment-id `pred` cascade at
    /// `(mi_row, mi_col)` over THIS twin's write-state mirror — the
    /// exact value the write path's own derivation will produce for
    /// this block given the symbols committed so far. A `skip == 1`
    /// leaf on a segmented inter frame MUST carry it as its
    /// `segment_id` (§5.11.19 arm 4 / §5.11.9 skip short-circuit), so
    /// candidate builders consult the twin BEFORE pricing.
    pub fn spatial_segment_pred(&self, mi_row: u32, mi_col: u32) -> u8 {
        self.state.segment_pred_ctx(mi_row, mi_col).0
    }

    /// Anti-desync check: after the driver's REAL emission of the
    /// superblock the search committed, the twin must hold the
    /// identical CDF state and coder `range`. A mismatch means the
    /// search committed a different symbol sequence than the writer
    /// emitted — a bug, never a tolerable approximation.
    pub fn matches(&self, cdfs: &TileCdfContext, writer: &SymbolWriter) -> bool {
        self.range == writer.range() && self.cdfs == *cdfs
    }

    /// r424 — fork the twin's state into a running per-TU pricing
    /// fork for one leaf's residual chain (see [`TuFork`]).
    pub fn tu_fork(&self) -> TuFork {
        TuFork {
            cdfs: self.cdfs.clone(),
            state: self.state.clone(),
            range: self.range,
            disable_cdf_update: self.disable_cdf_update,
        }
    }
}

/// r424 — the running per-TU twin fork for one leaf's residual chain:
/// snapshot the leaf-entry twin ([`RateTwin::tu_fork`]), then price
/// each §5.11.47 tx-type candidate's ACTUAL §5.11.39 coefficient
/// chain (the `all_zero` symbol at its true neighbour context, the
/// `inter_tx_type` / `intra_tx_type` S() against the CURRENT adaptive
/// CDFs, `eob_pt` / `coeff_base` / `coeff_br` / `dc_sign` / golomb
/// tails) through the writer's own one-TU body
/// ([`write_single_transform_block`] — the same
/// `write_transform_block` the emitting pass runs), and COMMIT the
/// winner so the next TU's candidates see its CDF adaptation and
/// §6.10.2 level-context stamps exactly as the emitting pass will.
///
/// Exactness note: the fork prices luma TUs from the leaf-entry state
/// — the block's mode-info prefix (coded between the snapshot and the
/// first TU in the real stream) touches no coefficient CDF row and no
/// level-context cell, and chroma TUs (coded after every luma TU of a
/// ≤ 64-sample-wide block) touch only the chroma context rows
/// (disjoint `txb_skip` context indices, `ptype = 1` tables), so the
/// per-candidate prices differ from the true in-stream costs only by
/// the shared §8.2.6 range fraction — identical across the candidates
/// of one election, which is all an argmin consumes.
pub(crate) struct TuFork {
    cdfs: TileCdfContext,
    state: PartitionSyntaxWriter,
    range: u32,
    disable_cdf_update: bool,
}

/// The leaf-constant inputs of one [`TuFork`] pricing/commit call —
/// everything the §5.11.47 / §5.11.39 one-TU write reads besides the
/// candidate itself.
pub(crate) struct TuCtx<'a> {
    pub params: &'a SyntaxFrameParams,
    /// Leaf position / size (mi units + §3 block-size ordinal).
    pub mi_row: u32,
    pub mi_col: u32,
    pub mi_size: usize,
    /// Leaf luma origin in pixels (`MiCol * MI_SIZE`, `MiRow * MI_SIZE`).
    pub base_x: u32,
    pub base_y: u32,
    /// §5.11.47 arm selector (`inter_tx_type` vs `intra_tx_type`).
    pub is_inter: bool,
    /// §5.11.47 quantiser-guard segment (`get_qindex( 1, segment_id )`).
    pub segment_id: u8,
    /// §8.3.2 `intra_dir` axis inputs (intra arm only).
    pub y_mode: u8,
    pub use_filter_intra: bool,
    pub filter_intra_mode: Option<u8>,
}

impl TuCtx<'_> {
    /// The facade [`SyntaxBlock`] carrying one TU's commitment at
    /// vector index 0.
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

impl TuFork {
    /// Exact price (1/256-bit units) of one LUMA TU candidate —
    /// `Quant[]` array + committed §5.11.47 `TxType` at the TU whose
    /// origin is `(x, y)` in 4-sample units from the leaf's luma
    /// origin — against this fork's CURRENT state. No commit.
    pub fn price_luma_tu(
        &self,
        ctx: &TuCtx<'_>,
        tx_sz: usize,
        x: u32,
        y: u32,
        quant: &[i32],
        tx_type: u8,
    ) -> Result<u64, Error> {
        let mut cdfs = self.cdfs.clone();
        let mut state = self.state.clone();
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, self.range);
        crate::encoder::partition_tree::write_single_transform_block(
            &mut w,
            &mut cdfs,
            &mut state,
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
        Ok(w.cost_bits256())
    }

    /// Commit the elected LUMA TU into the fork — advancing the CDFs,
    /// the §6.10.2 level-context mirror and the coder range exactly
    /// as the emitting pass will for the same TU.
    pub fn commit_luma_tu(
        &mut self,
        ctx: &TuCtx<'_>,
        tx_sz: usize,
        x: u32,
        y: u32,
        quant: &[i32],
        tx_type: u8,
    ) -> Result<(), Error> {
        let mut w = SymbolWriter::new_counting(self.disable_cdf_update, self.range);
        crate::encoder::partition_tree::write_single_transform_block(
            &mut w,
            &mut self.cdfs,
            &mut self.state,
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
        self.range = w.range();
        Ok(())
    }
}
