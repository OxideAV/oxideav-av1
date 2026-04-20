//! AV1 inter-frame syntax reader — narrow single-reference subset.
//!
//! Implements the narrow subset required by AVIF image-sequence (AVIS)
//! content:
//!
//! - `is_inter` per-block flag.
//! - Single-reference ref-frame pick (LAST only).
//! - 4-way inter mode decode (NEWMV / GLOBALMV / NEARESTMV / NEARMV).
//! - MV diff decode (via [`super::mv::MvDecoder`]).
//! - Per-block interpolation filter symbol (only emitted for
//!   switchable frames — callers short-circuit when the frame header
//!   pins REGULAR).
//! - `is_inter`-context skip flag.
//! - Intra-within-inter `y_mode` (inter-frame Y-mode CDF).
//!
//! Compound-ref prediction, global motion, warped motion,
//! inter-intra, OBMC, and the full ref-MV-list machinery are
//! intentionally unimplemented: their entrypoints surface
//! `Error::Unsupported("av1 inter-compound / warp / obmc pending")`.

use oxideav_core::{Error, Result};

use crate::cdfs;
use crate::predict::interp::InterpFilter;
use crate::symbol::SymbolDecoder;

use super::modes::IntraMode;
use super::mv::{Mv, MvDecoder};

/// Inter mode after compound has been ruled out (spec §6.10.22).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterMode {
    NewMv = 0,
    GlobalMv = 1,
    NearestMv = 2,
    NearMv = 3,
}

/// Per-tile inter decoder state. Holds mutable copies of every CDF
/// the narrow inter path consumes. Neighbor-context tracking is
/// handled by the superblock walker; this struct supplies only the
/// symbol-level primitives.
pub struct InterDecoder {
    /// MV diff reader — owns its own CDF bank.
    pub mv: MvDecoder,
    is_inter_cdf: [Vec<u16>; 4],
    new_mv_cdf: [Vec<u16>; 6],
    zero_mv_cdf: [Vec<u16>; 2],
    ref_mv_cdf: [Vec<u16>; 6],
    single_ref_cdf: [[Vec<u16>; 6]; 3],
    interp_filter_cdf: [Vec<u16>; 16],
    y_mode_cdf: [Vec<u16>; 4],
    /// The skip CDF is shared with the intra path — we own a private
    /// copy so adaptive updates during inter decode don't corrupt the
    /// intra CDFs.
    skip_cdf: [Vec<u16>; 3],
}

impl InterDecoder {
    /// Fresh `InterDecoder` primed from libaom default CDFs.
    pub fn new(allow_high_precision_mv: bool) -> Self {
        let mv = MvDecoder::new(allow_high_precision_mv);
        let is_inter_cdf = [
            cdfs::DEFAULT_IS_INTER_CDF[0].to_vec(),
            cdfs::DEFAULT_IS_INTER_CDF[1].to_vec(),
            cdfs::DEFAULT_IS_INTER_CDF[2].to_vec(),
            cdfs::DEFAULT_IS_INTER_CDF[3].to_vec(),
        ];
        let new_mv_cdf = [
            cdfs::DEFAULT_NEW_MV_CDF[0].to_vec(),
            cdfs::DEFAULT_NEW_MV_CDF[1].to_vec(),
            cdfs::DEFAULT_NEW_MV_CDF[2].to_vec(),
            cdfs::DEFAULT_NEW_MV_CDF[3].to_vec(),
            cdfs::DEFAULT_NEW_MV_CDF[4].to_vec(),
            cdfs::DEFAULT_NEW_MV_CDF[5].to_vec(),
        ];
        let zero_mv_cdf = [
            cdfs::DEFAULT_ZERO_MV_CDF[0].to_vec(),
            cdfs::DEFAULT_ZERO_MV_CDF[1].to_vec(),
        ];
        let ref_mv_cdf = [
            cdfs::DEFAULT_REF_MV_CDF[0].to_vec(),
            cdfs::DEFAULT_REF_MV_CDF[1].to_vec(),
            cdfs::DEFAULT_REF_MV_CDF[2].to_vec(),
            cdfs::DEFAULT_REF_MV_CDF[3].to_vec(),
            cdfs::DEFAULT_REF_MV_CDF[4].to_vec(),
            cdfs::DEFAULT_REF_MV_CDF[5].to_vec(),
        ];
        fn make_ref_row(row: &[&[u16]; 6]) -> [Vec<u16>; 6] {
            [
                row[0].to_vec(),
                row[1].to_vec(),
                row[2].to_vec(),
                row[3].to_vec(),
                row[4].to_vec(),
                row[5].to_vec(),
            ]
        }
        let single_ref_cdf = [
            make_ref_row(&cdfs::DEFAULT_SINGLE_REF_CDF[0]),
            make_ref_row(&cdfs::DEFAULT_SINGLE_REF_CDF[1]),
            make_ref_row(&cdfs::DEFAULT_SINGLE_REF_CDF[2]),
        ];
        let mut interp_filter_cdf: [Vec<u16>; 16] = Default::default();
        for (i, dst) in interp_filter_cdf.iter_mut().enumerate() {
            *dst = cdfs::DEFAULT_INTERP_FILTER_CDF[i].to_vec();
        }
        let y_mode_cdf = [
            cdfs::DEFAULT_Y_MODE_CDF[0].to_vec(),
            cdfs::DEFAULT_Y_MODE_CDF[1].to_vec(),
            cdfs::DEFAULT_Y_MODE_CDF[2].to_vec(),
            cdfs::DEFAULT_Y_MODE_CDF[3].to_vec(),
        ];
        let skip_cdf = [
            cdfs::DEFAULT_SKIP_CDF[0].to_vec(),
            cdfs::DEFAULT_SKIP_CDF[1].to_vec(),
            cdfs::DEFAULT_SKIP_CDF[2].to_vec(),
        ];

        Self {
            mv,
            is_inter_cdf,
            new_mv_cdf,
            zero_mv_cdf,
            ref_mv_cdf,
            single_ref_cdf,
            interp_filter_cdf,
            y_mode_cdf,
            skip_cdf,
        }
    }

    /// Read the `is_inter` flag. Context is computed from the
    /// above/left `is_inter` neighbors: both → 3, either → 1, else 0.
    pub fn read_is_inter(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        above_is_inter: bool,
        left_is_inter: bool,
    ) -> Result<bool> {
        let ctx = if above_is_inter && left_is_inter {
            3
        } else if above_is_inter || left_is_inter {
            1
        } else {
            0
        };
        let raw = sym.decode_symbol(&mut self.is_inter_cdf[ctx])?;
        Ok(raw == 1)
    }

    /// Read the single-ref selector tree per §5.11.24. The narrow
    /// Phase 7 decoder only carries the LAST reference, so every
    /// selection on non-LAST paths is mapped to LAST (the bits are
    /// still consumed so the bitstream stays framed).
    ///
    /// Returns the logical reference-frame index in `0..=6` (LAST=0,
    /// LAST2=1, LAST3=2, GOLDEN=3, BWDREF=4, ALTREF2=5, ALTREF=6) for
    /// bookkeeping — all values are collapsed to LAST by the caller.
    pub fn read_single_ref_frame(&mut self, sym: &mut SymbolDecoder<'_>) -> Result<u32> {
        let ctx = 1usize;
        // ref[0]: LAST group vs BWD group.
        let b0 = sym.decode_symbol(&mut self.single_ref_cdf[ctx][0])?;
        if b0 == 0 {
            // LAST group
            let b1 = sym.decode_symbol(&mut self.single_ref_cdf[ctx][1])?;
            if b1 == 0 {
                let b2 = sym.decode_symbol(&mut self.single_ref_cdf[ctx][2])?;
                Ok(if b2 == 0 { 0 } else { 1 }) // LAST / LAST2
            } else {
                let b3 = sym.decode_symbol(&mut self.single_ref_cdf[ctx][3])?;
                Ok(if b3 == 0 { 2 } else { 3 }) // LAST3 / GOLDEN
            }
        } else {
            // BWD group
            let b4 = sym.decode_symbol(&mut self.single_ref_cdf[ctx][4])?;
            if b4 == 0 {
                Ok(4) // BWDREF
            } else {
                let b5 = sym.decode_symbol(&mut self.single_ref_cdf[ctx][5])?;
                Ok(if b5 == 0 { 5 } else { 6 }) // ALTREF2 / ALTREF
            }
        }
    }

    /// Read the 4-way inter mode. Context arguments select which of
    /// the 6 `new_mv_cdf` / `ref_mv_cdf` entries or 2 `zero_mv_cdf`
    /// entries are consulted.
    pub fn read_inter_mode(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        new_mv_ctx: usize,
        zero_mv_ctx: usize,
        ref_mv_ctx: usize,
    ) -> Result<InterMode> {
        let ctx_n = new_mv_ctx.min(5);
        let ctx_z = zero_mv_ctx.min(1);
        let ctx_r = ref_mv_ctx.min(5);
        if sym.decode_symbol(&mut self.new_mv_cdf[ctx_n])? == 0 {
            return Ok(InterMode::NewMv);
        }
        if sym.decode_symbol(&mut self.zero_mv_cdf[ctx_z])? == 0 {
            return Ok(InterMode::GlobalMv);
        }
        if sym.decode_symbol(&mut self.ref_mv_cdf[ctx_r])? == 0 {
            return Ok(InterMode::NearestMv);
        }
        Ok(InterMode::NearMv)
    }

    /// Decode one MV diff via the inner [`MvDecoder`].
    pub fn read_mv(&mut self, sym: &mut SymbolDecoder<'_>) -> Result<Mv> {
        self.mv.read_mv(sym)
    }

    /// Decode a per-block interpolation filter symbol when the frame
    /// header has `is_filter_switchable` set. `ctx` is the 16-way
    /// above/left-context bucket (clamped).
    pub fn read_interp_filter(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        ctx: usize,
    ) -> Result<InterpFilter> {
        let ctx = ctx.min(15);
        let raw = sym.decode_symbol(&mut self.interp_filter_cdf[ctx])?;
        Ok(InterpFilter::from_u32(raw))
    }

    /// Decode an inter-frame intra Y-mode for an intra-within-inter
    /// block. `block_size_group` is the 4-bucket area group.
    pub fn read_y_mode(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        block_size_group: usize,
    ) -> Result<IntraMode> {
        let g = block_size_group.min(3);
        let raw = sym.decode_symbol(&mut self.y_mode_cdf[g])?;
        IntraMode::from_u32(raw).ok_or_else(|| {
            Error::invalid(format!(
                "av1 inter y_mode: invalid intra mode {raw} (§5.11.22)"
            ))
        })
    }

    /// Decode the skip_txfm flag on an inter block. `ctx` is
    /// 0..=2 (clamped).
    pub fn read_skip(&mut self, sym: &mut SymbolDecoder<'_>, ctx: usize) -> Result<bool> {
        let c = ctx.min(2);
        Ok(sym.decode_symbol(&mut self.skip_cdf[c])? != 0)
    }
}

/// Map a luma block `(w, h)` to the 4-bucket block-size group used by
/// the inter-frame Y-mode CDF: 0 for area ≤ 8×8, 1 for ≤ 16×16, 2 for
/// ≤ 32×32, 3 for larger.
pub fn block_size_group(w: usize, h: usize) -> usize {
    let area = w * h;
    if area <= 8 * 8 {
        0
    } else if area <= 16 * 16 {
        1
    } else if area <= 32 * 32 {
        2
    } else {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_size_group_boundaries() {
        assert_eq!(block_size_group(4, 4), 0);
        assert_eq!(block_size_group(8, 8), 0);
        assert_eq!(block_size_group(16, 8), 1);
        assert_eq!(block_size_group(16, 16), 1);
        assert_eq!(block_size_group(32, 16), 2);
        assert_eq!(block_size_group(32, 32), 2);
        assert_eq!(block_size_group(64, 64), 3);
        assert_eq!(block_size_group(128, 128), 3);
    }

    #[test]
    fn new_sets_every_cdf_bank() {
        let id = InterDecoder::new(false);
        assert_eq!(id.is_inter_cdf.len(), 4);
        assert_eq!(id.new_mv_cdf.len(), 6);
        assert_eq!(id.zero_mv_cdf.len(), 2);
        assert_eq!(id.ref_mv_cdf.len(), 6);
        assert_eq!(id.single_ref_cdf.len(), 3);
        assert_eq!(id.interp_filter_cdf.len(), 16);
        assert_eq!(id.y_mode_cdf.len(), 4);
        assert_eq!(id.skip_cdf.len(), 3);
    }
}
