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

    /// Anti-desync check: after the driver's REAL emission of the
    /// superblock the search committed, the twin must hold the
    /// identical CDF state and coder `range`. A mismatch means the
    /// search committed a different symbol sequence than the writer
    /// emitted — a bug, never a tolerable approximation.
    pub fn matches(&self, cdfs: &TileCdfContext, writer: &SymbolWriter) -> bool {
        self.range == writer.range() && self.cdfs == *cdfs
    }
}
