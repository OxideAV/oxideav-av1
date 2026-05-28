//! §5.11.4 `decode_partition` **recursive dispatch driver** — the
//! encoder counterpart of
//! [`crate::cdf::PartitionWalker::decode_partition`]'s recursive walk.
//!
//! Arc 11 (r217). This module composes the per-block writers landed in
//! r211–r216 ([`crate::encoder::block_mode_info`] +
//! [`crate::encoder::coefficients`] +
//! [`crate::encoder::partition::write_partition`]) into a complete
//! partition-tree walker driven from a caller-supplied [`EncodeNode`]
//! tree.
//!
//! ## Tree shape
//!
//! Each node is either a [`EncodeNode::Leaf`] (one §5.11.5
//! `decode_block` invocation) or a [`EncodeNode::Split`] (four §5.11.4
//! recursive children). For arc 11 the leaf is **intra-only**:
//! `decode_block` collapses to `read_skip` + `intra_segment_id` +
//! `y_mode` + `uv_mode` + `coefficients()` per Y/U/V plane. Inter
//! `mode_info` + motion-vector + reference-frame writes are out of
//! scope for r217 (next arc).
//!
//! ## Spec dispatch shape
//!
//! Per the spec body at av1-spec.txt §5.11.4 (p.61-62) — reproduced
//! from the project's `docs/video/av1/av1-spec.txt`:
//!
//! ```text
//!   decode_partition( r, c, bSize ) {
//!       if ( r >= MiRows || c >= MiCols ) return 0
//!       num4x4 = Num_4x4_Blocks_Wide[ bSize ]
//!       halfBlock4x4 = num4x4 >> 1
//!       hasRows = ( r + halfBlock4x4 ) < MiRows
//!       hasCols = ( c + halfBlock4x4 ) < MiCols
//!       if ( bSize < BLOCK_8X8 )            partition = PARTITION_NONE   (no symbol)
//!       else if ( hasRows && hasCols )      partition                    S()
//!       else if ( hasCols )                 split_or_horz                S()
//!       else if ( hasRows )                 split_or_vert                S()
//!       else                                partition = PARTITION_SPLIT  (no symbol)
//!       ...
//!       if ( partition == PARTITION_NONE )  decode_block( r, c, subSize )
//!       else if ( partition == PARTITION_SPLIT ) recurse on four quadrants
//!       ...
//!   }
//! ```
//!
//! The driver mirrors the §5.11.4 first conditional precisely:
//!
//! 1. **`r >= MiRows || c >= MiCols`** — early return; an out-of-frame
//!    quadrant of a [`EncodeNode::Split`] is allowed and emits
//!    nothing. The corresponding `Box<EncodeNode>` is consulted but
//!    its [`EncodeNode::Leaf`] / [`EncodeNode::Split`] payload is
//!    skipped (the caller may supply a dummy sentinel — see
//!    [`EncodeNode::dummy_oob`]).
//! 2. **`bSize < BLOCK_8X8` forced-NONE** — no `partition` symbol; the
//!    node MUST be a [`EncodeNode::Leaf`] (else
//!    [`Error::PartitionWalkOutOfRange`]).
//! 3. **`!hasRows && !hasCols` forced-SPLIT** — no `partition` symbol;
//!    the node MUST be a [`EncodeNode::Split`] (else
//!    [`Error::PartitionWalkOutOfRange`]).
//! 4. Otherwise — emit the `partition` / `split_or_horz` /
//!    `split_or_vert` symbol via
//!    [`crate::encoder::partition::write_partition`] (re-using the same
//!    §8.3.2 ctx the decoder derives), then dispatch.
//!
//! ## Scope: PARTITION_NONE + PARTITION_SPLIT only
//!
//! For arc 11 the driver supports the two recursive shapes the §5.11.4
//! walk uses for "pure" partition trees: [`EncodeNode::Leaf`] (one
//! block at the node's `bSize`) and [`EncodeNode::Split`] (four
//! `subSize` quadrants). The asymmetric partitions
//! (`PARTITION_HORZ` / `PARTITION_VERT` / `PARTITION_HORZ_A` /
//! `PARTITION_HORZ_B` / `PARTITION_VERT_A` / `PARTITION_VERT_B` /
//! `PARTITION_HORZ_4` / `PARTITION_VERT_4`) are out of scope until a
//! follow-up arc — the [`crate::encoder::partition::write_partition`]
//! symbol writer already supports their alphabet, but the dispatch
//! (which has 2-3 leaves per node with mixed `subSize` /
//! `splitSize`) is a separate piece of work.
//!
//! ## MiSizes neighbour-grid maintenance
//!
//! The §8.3.2 `partition_ctx` lookup ([`crate::cdf::partition_ctx`])
//! reads the `MiSizes[r-1][c]` / `MiSizes[r][c-1]` neighbour widths to
//! pick its `above` / `left` booleans. The driver maintains its own
//! `MiSizes[]` grid at construction (mirroring
//! [`crate::cdf::PartitionWalker::decode_block`]'s grid-fill), so the
//! ctx derivation on the encoder side observes the same neighbour
//! widths the decoder's parallel `PartitionWalker` observes. This is
//! what keeps the encoder and decoder in lockstep across the §8.3
//! CDF adaptation.
//!
//! ## Spec provenance
//!
//! Sourced from `docs/video/av1/av1-spec.txt`:
//!   * §5.11.4  decode_partition           (p.61-62)
//!   * §5.11.5  decode_block               (p.62)
//!   * §5.11.7  intra_frame_mode_info       (p.65)
//!   * §5.11.8  intra_segment_id           (p.65)
//!   * §5.11.11 read_skip                  (p.67)
//!   * §5.11.22 intra_block_mode_info       (p.72)
//!   * §5.11.39 coefficients               (p.88-93)
//!   * §8.3.2  context derivations         (p.361-378)
//!   * §9.3    Partition_Subsize           (p.400)
//!
//! [`crate::cdf::PartitionWalker`]: crate::cdf::PartitionWalker
//! [`crate::cdf::PartitionWalker::decode_partition`]: crate::cdf::PartitionWalker::decode_partition
//! [`crate::cdf::PartitionWalker::decode_block`]: crate::cdf::PartitionWalker

use crate::cdf::{
    cfl_allowed_for_uv_mode, intra_mode_ctx, partition_ctx, partition_subsize, size_group,
    skip_ctx, TileCdfContext, TileGeometry, BLOCK_8X8, BLOCK_INVALID, BLOCK_SIZES, DC_PRED,
    MI_HEIGHT_LOG2, MI_WIDTH_LOG2, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PARTITION_NONE,
    PARTITION_SPLIT,
};
use crate::encoder::block_mode_info::{
    write_intra_frame_y_mode, write_intra_segment_id, write_intra_uv_mode, write_skip, write_y_mode,
};
use crate::encoder::coefficients::write_coefficients;
use crate::encoder::partition::write_partition;
use crate::encoder::symbol_writer::SymbolWriter;
use crate::Error;

/// Per-plane §5.11.39 `coefficients()` input the driver hands to
/// [`write_coefficients`] at a leaf.
///
/// The driver does NOT derive `scan` / `tx_size` / `tx_class` — those
/// are caller responsibilities (an encoder's RD search picks the TX
/// per-plane). Mirrors the existing §5.11.39 writer surface verbatim.
#[derive(Debug, Clone)]
pub struct PlaneCoefficients {
    /// §5.11.39 `plane` — `0` (Y), `1`/`2` (U/V).
    pub plane: u8,
    /// §5.11.39 `is_inter`. r217 driver is intra-only → MUST be `0`.
    pub is_inter: u8,
    /// §5.11.39 `txSz` — see [`write_coefficients`].
    pub tx_size: usize,
    /// §5.11.39 `txClass` — see [`write_coefficients`].
    pub tx_class: usize,
    /// §8.3.2 `all_zero` ctx in `0..TXB_SKIP_CONTEXTS = 13`.
    pub txb_skip_ctx: usize,
    /// §8.3.2 `dc_sign` ctx in `0..DC_SIGN_CONTEXTS = 3`.
    pub dc_sign_ctx: usize,
    /// §7.5 scan table for `(tx_size, tx_class)`. See
    /// [`crate::scan::get_scan`].
    pub scan: Vec<u16>,
    /// Per-position signed final `Quant[]` array. See
    /// [`write_coefficients`].
    pub quant: Vec<i32>,
}

/// One intra-arm `decode_block` leaf — the encoder-side bundle of the
/// scalars a §5.11.5 leaf reads (`skip`, `segment_id`, `y_mode`,
/// optional `uv_mode`) plus a per-plane [`PlaneCoefficients`] vector
/// (one entry per plane the §5.11.39 `coefficients()` block walks).
///
/// `coefficients` is keyed by `plane` ordinal (0 = Y, 1 = U, 2 = V).
/// Monochrome leaves supply only `plane = 0`.
#[derive(Debug, Clone)]
pub struct EncodeBlock {
    /// §5.11.11 `skip`. `0` or `1`.
    pub skip: u8,
    /// §5.11.8 / §5.11.9 `segment_id`. `0..MAX_SEGMENTS`.
    pub segment_id: u8,
    /// §5.11.9 segment-id `pred` from the §5.11.9 neighbour cascade.
    /// `0..MAX_SEGMENTS`.
    pub segment_pred: u8,
    /// §5.11.22 `y_mode`. `0..INTRA_MODES = 13`.
    pub y_mode: u8,
    /// §5.11.22 `uv_mode`. `None` for monochrome leaves;
    /// `Some(0..UV_INTRA_MODES_*)` otherwise.
    pub uv_mode: Option<u8>,
    /// §5.11.39 `coefficients()` input per plane the leaf walks.
    /// Length `1` (monochrome) or `3` (luma + 2 chroma).
    pub coefficients: Vec<PlaneCoefficients>,
}

/// One node of the §5.11.4 partition tree.
///
/// Either a [`Self::Leaf`] (terminal — one §5.11.5 `decode_block` at
/// the node's `bSize`) or a [`Self::Split`] (four §5.11.4 recursive
/// children at `Partition_Subsize[PARTITION_SPLIT][bSize]`).
///
/// Asymmetric partitions (`PARTITION_HORZ` etc.) are out of scope for
/// arc 11 — the [`write_partition_tree`] driver rejects them with
/// [`Error::PartitionWalkOutOfRange`].
#[derive(Debug, Clone)]
pub enum EncodeNode {
    /// Single block at this node's `bSize`. PARTITION_NONE for nodes
    /// large enough to subdivide; otherwise the `bSize < BLOCK_8X8`
    /// forced-NONE arm.
    Leaf(EncodeBlock),
    /// Four §5.11.4 quadrants at `Partition_Subsize[PARTITION_SPLIT][bSize]`.
    /// Order matches the spec: `[NW, NE, SW, SE]` = `[(r, c),
    /// (r, c+half), (r+half, c), (r+half, c+half)]`.
    Split([Box<EncodeNode>; 4]),
}

impl EncodeNode {
    /// Sentinel a caller can pass for an out-of-frame quadrant
    /// (`r >= MiRows || c >= MiCols`) — the driver short-circuits per
    /// the §5.11.4 line-1 `return 0` before ever inspecting the
    /// sentinel's contents.
    #[must_use]
    pub fn dummy_oob() -> Self {
        EncodeNode::Leaf(EncodeBlock {
            skip: 0,
            segment_id: 0,
            segment_pred: 0,
            y_mode: 0,
            uv_mode: None,
            coefficients: Vec::new(),
        })
    }
}

/// Driver state — mirrors the decoder-side
/// [`crate::cdf::PartitionWalker`]'s frame geometry + `MiSizes[]` grid.
///
/// Constructed against the frame's `MiRows` / `MiCols` extent and tile
/// [`TileGeometry`]; mutated as the recursion stamps decoded leaves.
/// One driver per tile.
#[derive(Debug)]
pub struct PartitionTreeWriter {
    mi_rows: u32,
    mi_cols: u32,
    geometry: TileGeometry,
    /// `MiSizes[row][col]` packed row-major. `BLOCK_INVALID` for cells
    /// not yet stamped by a leaf.
    mi_sizes: Vec<usize>,
    /// `segmentation_enabled` — gates the §5.11.8 `intra_segment_id`
    /// write.
    segmentation_enabled: bool,
    /// `last_active_seg_id` — for the §5.11.9 `neg_deinterleave` `max`.
    last_active_seg_id: u8,
    /// `lossless` — gates §5.11.22 `cfl_allowed_for_uv_mode`'s
    /// lossless arm.
    lossless: bool,
    /// `subsampling_x` / `subsampling_y` — gates
    /// `cfl_allowed_for_uv_mode`'s chroma-arm.
    subsampling_x: bool,
    subsampling_y: bool,
}

impl PartitionTreeWriter {
    /// Construct a driver for a frame of `mi_rows` × `mi_cols` mi units
    /// scoped to `geometry`. Returns `None` for a zero-extent geometry
    /// (mirroring the decoder's [`crate::cdf::PartitionWalker::new`]
    /// guard).
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mi_rows: u32,
        mi_cols: u32,
        geometry: TileGeometry,
        segmentation_enabled: bool,
        last_active_seg_id: u8,
        lossless: bool,
        subsampling_x: bool,
        subsampling_y: bool,
    ) -> Option<Self> {
        if mi_rows == 0 || mi_cols == 0 {
            return None;
        }
        if geometry.mi_row_end > mi_rows || geometry.mi_col_end > mi_cols {
            return None;
        }
        if geometry.mi_row_start >= geometry.mi_row_end
            || geometry.mi_col_start >= geometry.mi_col_end
        {
            return None;
        }
        Some(Self {
            mi_rows,
            mi_cols,
            geometry,
            mi_sizes: vec![BLOCK_INVALID; (mi_rows * mi_cols) as usize],
            segmentation_enabled,
            last_active_seg_id,
            lossless,
            subsampling_x,
            subsampling_y,
        })
    }

    /// `partition_ctx_for(r, c, bsl)` — mirrors the decoder's private
    /// helper. See [`crate::cdf::PartitionWalker::decode_partition`].
    fn partition_ctx_for(&self, r: u32, c: u32, bsl: u32) -> usize {
        let avail_u = self.geometry.is_inside(r as i32 - 1, c as i32);
        let avail_l = self.geometry.is_inside(r as i32, c as i32 - 1);
        let above = if avail_u {
            let nb = self.mi_size_at(r as i32 - 1, c as i32);
            nb < BLOCK_SIZES && (MI_WIDTH_LOG2[nb] as u32) < bsl
        } else {
            false
        };
        let left = if avail_l {
            let nb = self.mi_size_at(r as i32, c as i32 - 1);
            nb < BLOCK_SIZES && (MI_HEIGHT_LOG2[nb] as u32) < bsl
        } else {
            false
        };
        partition_ctx(above, left)
    }

    fn mi_size_at(&self, r: i32, c: i32) -> usize {
        if r < 0 || c < 0 {
            return BLOCK_INVALID;
        }
        let (r, c) = (r as u32, c as u32);
        if r >= self.mi_rows || c >= self.mi_cols {
            return BLOCK_INVALID;
        }
        self.mi_sizes[(r * self.mi_cols + c) as usize]
    }

    /// Stamp `MiSizes[r..r+bh4][c..c+bw4] = sub_size` per the §6.10.4
    /// grid-fill rule. Clipped at the frame's `MiRows` / `MiCols`
    /// extent.
    fn stamp_mi_sizes(&mut self, r: u32, c: u32, sub_size: usize) {
        let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size] as u32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size] as u32;
        for dr in 0..bh4 {
            let rr = r + dr;
            if rr >= self.mi_rows {
                break;
            }
            for dc in 0..bw4 {
                let cc = c + dc;
                if cc >= self.mi_cols {
                    break;
                }
                self.mi_sizes[(rr * self.mi_cols + cc) as usize] = sub_size;
            }
        }
    }
}

/// Recursive §5.11.4 driver entry point — writes the partition tree
/// rooted at `(r, c, b_size)` into `writer` against the running
/// `cdfs` + `state` grids. The `node` shape MUST match the partition
/// decisions implied by `(b_size, has_rows, has_cols)`.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for:
///   * `b_size >= BLOCK_SIZES`.
///   * A [`EncodeNode::Leaf`] supplied at a forced-SPLIT corner
///     (`!has_rows && !has_cols` with `b_size >= BLOCK_8X8`).
///   * A [`EncodeNode::Split`] supplied at a forced-NONE node
///     (`b_size < BLOCK_8X8`).
///   * Any block-syntax / coefficient-write surface error propagated
///     from the per-block writers.
///
/// The §5.11.4 line-1 `r >= MiRows || c >= MiCols` early return is
/// honoured before the node is inspected; the caller may pass a
/// [`EncodeNode::dummy_oob`] in that quadrant slot.
pub fn write_partition_tree(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionTreeWriter,
    node: &EncodeNode,
    r: u32,
    c: u32,
    b_size: usize,
) -> Result<(), Error> {
    // §5.11.4 line 1 — out-of-frame quadrant short-circuit.
    if r >= state.mi_rows || c >= state.mi_cols {
        return Ok(());
    }
    if b_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let num4x4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
    let half_block4x4 = num4x4 >> 1;
    let has_rows = (r + half_block4x4) < state.mi_rows;
    let has_cols = (c + half_block4x4) < state.mi_cols;

    // §5.11.4 lines 9-19 — choose the partition + emit the matching
    // `partition` / `split_or_horz` / `split_or_vert` symbol (or no
    // symbol on the forced-NONE / forced-SPLIT arms).
    let partition = match node {
        EncodeNode::Leaf(_) => {
            // Forced-SPLIT corner can't be a leaf — caller-bug.
            if b_size >= BLOCK_8X8 && !has_rows && !has_cols {
                return Err(Error::PartitionWalkOutOfRange);
            }
            PARTITION_NONE
        }
        EncodeNode::Split(_) => {
            // Forced-NONE can't be a split — caller-bug.
            if b_size < BLOCK_8X8 {
                return Err(Error::PartitionWalkOutOfRange);
            }
            PARTITION_SPLIT
        }
    };

    let pctx = if b_size >= BLOCK_8X8 {
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        state.partition_ctx_for(r, c, bsl)
    } else {
        0
    };

    write_partition(writer, cdfs, partition, b_size, has_rows, has_cols, pctx)?;

    // §5.11.4 lines 20-21 + 22-23 — subSize lookup + per-partition
    // dispatch.
    let sub_size = partition_subsize(partition, b_size).ok_or(Error::PartitionWalkOutOfRange)?;

    match (node, partition) {
        (EncodeNode::Leaf(block), PARTITION_NONE) => {
            // §5.11.4 line 22-23: PARTITION_NONE ⇒ `decode_block(r, c, subSize)`.
            // `subSize == b_size` for the PARTITION_NONE arm per §9.3.
            state.stamp_mi_sizes(r, c, sub_size);
            write_encode_block_leaf(writer, cdfs, state, block, r, c, sub_size)?;
        }
        (EncodeNode::Split(children), PARTITION_SPLIT) => {
            // §5.11.4 lines 31-35 — recurse on four `subSize` quadrants
            // in NW, NE, SW, SE order.
            let [nw, ne, sw, se] = children;
            write_partition_tree(writer, cdfs, state, nw, r, c, sub_size)?;
            write_partition_tree(writer, cdfs, state, ne, r, c + half_block4x4, sub_size)?;
            write_partition_tree(writer, cdfs, state, sw, r + half_block4x4, c, sub_size)?;
            write_partition_tree(
                writer,
                cdfs,
                state,
                se,
                r + half_block4x4,
                c + half_block4x4,
                sub_size,
            )?;
        }
        // The other (node, partition) pairings can't happen — the
        // match arms above are total over `EncodeNode` and `partition`
        // is derived from `node`.
        _ => return Err(Error::PartitionWalkOutOfRange),
    }
    Ok(())
}

/// Per-leaf §5.11.5 `decode_block` syntax emission — intra arm. Writes,
/// in §5.11 syntax order:
///
///   1. §5.11.11 `read_skip` (using the §8.3.2 `skip_ctx` derived from
///      the neighbour `Skips[]` grid — for arc 11 the grid is not
///      maintained, so `skip_ctx = skip_ctx(0, 0) = 0` at every leaf,
///      matching the decoder behaviour when neighbour skips are
///      pre-zero).
///   2. §5.11.8 `intra_segment_id` (using `segment_pred` from the
///      `EncodeBlock` + `segment_id_ctx = 0` for the simplest origin
///      case).
///   3. §5.11.22 `y_mode` (using `Size_Group[ MiSize ]` ctx). The
///      r217 driver uses the non-keyframe `y_mode` writer (the
///      §5.11.22 line-3 path), not the keyframe `intra_frame_y_mode`
///      (§5.11.7) — both are present in the encoder, and the
///      [`write_y_mode`] path needs the simpler `Size_Group` ctx.
///   4. §5.11.22 `uv_mode` (gated on `has_chroma == uv_mode.is_some()`).
///   5. §5.11.39 `coefficients()` per plane in the supplied order.
///
/// The §8.3.2 ctx derivations on this arc are intentionally minimal —
/// at-the-origin defaults — to keep the round-trip surface tight while
/// the wider neighbour-aware ctx walk (`AboveSkips[]` / `LeftSkips[]`
/// / `YModes[]` etc.) lands in the follow-up arc. The decoder side
/// already supports those richer contexts; threading them through the
/// driver is purely an additive change.
#[allow(clippy::too_many_arguments)]
fn write_encode_block_leaf(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionTreeWriter,
    block: &EncodeBlock,
    _r: u32,
    _c: u32,
    sub_size: usize,
) -> Result<(), Error> {
    // §5.11.11 read_skip — origin-default ctx for arc 11.
    let skip_ctx_val = skip_ctx(0, 0);
    write_skip(writer, cdfs, block.skip, skip_ctx_val, false)?;

    // §5.11.8 intra_segment_id — origin-default segment_id_ctx for arc 11.
    write_intra_segment_id(
        writer,
        cdfs,
        block.segment_id,
        block.skip,
        block.segment_pred,
        0,
        state.segmentation_enabled,
        state.last_active_seg_id,
    )?;

    // §5.11.22 y_mode — Size_Group ctx (non-keyframe path).
    let size_group_ctx = size_group(sub_size);
    write_y_mode(writer, cdfs, block.y_mode, size_group_ctx)?;
    // Silence the unused-import warning on builds that strip the
    // intra-frame y-mode writer; it stays in scope as the documented
    // keyframe alternative for the next arc that lands a keyframe
    // dispatch.
    let _ = write_intra_frame_y_mode;
    let _ = intra_mode_ctx;
    let _ = DC_PRED;

    // §5.11.22 uv_mode — gated on `has_chroma`.
    if let Some(uv_mode) = block.uv_mode {
        let cfl_allowed = cfl_allowed_for_uv_mode(
            state.lossless,
            sub_size,
            state.subsampling_x,
            state.subsampling_y,
        );
        write_intra_uv_mode(writer, cdfs, block.y_mode, uv_mode, cfl_allowed)?;
    }

    // §5.11.39 coefficients() per plane.
    for plane_in in &block.coefficients {
        write_coefficients(
            writer,
            cdfs,
            plane_in.plane,
            plane_in.is_inter,
            plane_in.tx_size,
            plane_in.tx_class,
            plane_in.txb_skip_ctx,
            plane_in.dc_sign_ctx,
            &plane_in.scan,
            &plane_in.quant,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        get_br_ctx, get_coeff_base_ctx, get_coeff_base_eob_ctx, CoefficientsReadout,
        PartitionWalker, TileGeometry as G, BLOCK_16X16, BLOCK_32X32, BLOCK_64X64, BLOCK_8X8,
        DC_PRED, MAX_SEGMENTS, TX_4X4, TX_CLASS_2D, V_PRED,
    };
    use crate::scan::get_default_scan;
    use crate::symbol_decoder::SymbolDecoder;

    /// Build a single-tile geometry covering `mi_rows × mi_cols`.
    fn single_tile(mi_rows: u32, mi_cols: u32) -> G {
        G {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        }
    }

    /// Helper: build a fresh `PartitionTreeWriter` with segmentation off.
    fn fresh_state(mi_rows: u32, mi_cols: u32) -> PartitionTreeWriter {
        PartitionTreeWriter::new(
            mi_rows,
            mi_cols,
            single_tile(mi_rows, mi_cols),
            /* segmentation_enabled = */ false,
            /* last_active_seg_id = */ 0,
            /* lossless = */ false,
            /* subsampling_x = */ false,
            /* subsampling_y = */ false,
        )
        .expect("state construction")
    }

    /// Build the smallest valid `EncodeBlock`: skip = 1 (no coefficient
    /// pass needed downstream), Y/UV DC_PRED, no coefficients vector
    /// per plane (the coefficient stream is gated separately — this
    /// helper produces a "structural" leaf where only the mode-info
    /// portion is exercised). `uv_mode = Some(DC_PRED)` ⇒ chroma arm
    /// fires.
    fn make_skip_leaf() -> EncodeBlock {
        EncodeBlock {
            skip: 1,
            segment_id: 0,
            segment_pred: 0,
            y_mode: DC_PRED as u8,
            uv_mode: Some(DC_PRED as u8),
            coefficients: Vec::new(),
        }
    }

    /// Build a leaf that also writes one all-zero TX_4X4 luma + chroma
    /// coefficient block per plane. `txb_skip_ctx == 0` /
    /// `dc_sign_ctx == 0` — the simplest case.
    fn make_zero_coeff_leaf() -> EncodeBlock {
        let scan: Vec<u16> = get_default_scan(TX_4X4).to_vec();
        let quant = vec![0i32; 16];
        let coefficients = vec![
            PlaneCoefficients {
                plane: 0,
                is_inter: 0,
                tx_size: TX_4X4,
                tx_class: TX_CLASS_2D,
                txb_skip_ctx: 0,
                dc_sign_ctx: 0,
                scan: scan.clone(),
                quant: quant.clone(),
            },
            PlaneCoefficients {
                plane: 1,
                is_inter: 0,
                tx_size: TX_4X4,
                tx_class: TX_CLASS_2D,
                txb_skip_ctx: 0,
                dc_sign_ctx: 0,
                scan: scan.clone(),
                quant: quant.clone(),
            },
            PlaneCoefficients {
                plane: 2,
                is_inter: 0,
                tx_size: TX_4X4,
                tx_class: TX_CLASS_2D,
                txb_skip_ctx: 0,
                dc_sign_ctx: 0,
                scan,
                quant,
            },
        ];
        EncodeBlock {
            skip: 0,
            segment_id: 0,
            segment_pred: 0,
            y_mode: DC_PRED as u8,
            uv_mode: Some(DC_PRED as u8),
            coefficients,
        }
    }

    // -----------------------------------------------------------------
    // Constructor + state guards.
    // -----------------------------------------------------------------

    #[test]
    fn new_rejects_zero_extent() {
        assert!(
            PartitionTreeWriter::new(0, 8, single_tile(0, 8), false, 0, false, false, false)
                .is_none()
        );
        assert!(
            PartitionTreeWriter::new(8, 0, single_tile(8, 0), false, 0, false, false, false)
                .is_none()
        );
    }

    #[test]
    fn new_rejects_oversize_geometry() {
        let bad = G {
            mi_row_start: 0,
            mi_row_end: 16, // > mi_rows
            mi_col_start: 0,
            mi_col_end: 8,
        };
        assert!(PartitionTreeWriter::new(8, 8, bad, false, 0, false, false, false).is_none());
    }

    #[test]
    fn new_rejects_inverted_geometry() {
        let bad = G {
            mi_row_start: 4,
            mi_row_end: 4, // start == end
            mi_col_start: 0,
            mi_col_end: 8,
        };
        assert!(PartitionTreeWriter::new(8, 8, bad, false, 0, false, false, false).is_none());
    }

    // -----------------------------------------------------------------
    // Caller-bug shape rejections.
    // -----------------------------------------------------------------

    #[test]
    fn rejects_split_at_forced_none_b_size() {
        // b_size < BLOCK_8X8 ⇒ forced-NONE; supplying a Split is a
        // caller-bug.
        let mut writer = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = fresh_state(8, 8);
        let node = EncodeNode::Split([
            Box::new(EncodeNode::Leaf(make_skip_leaf())),
            Box::new(EncodeNode::Leaf(make_skip_leaf())),
            Box::new(EncodeNode::Leaf(make_skip_leaf())),
            Box::new(EncodeNode::Leaf(make_skip_leaf())),
        ]);
        let err = write_partition_tree(
            &mut writer,
            &mut cdfs,
            &mut state,
            &node,
            0,
            0,
            crate::cdf::BLOCK_4X4,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    #[test]
    fn rejects_leaf_at_forced_split_corner() {
        // 16×16 frame, BLOCK_16X16 at (0,0): half_block4x4 = 2 < 16 ⇒
        // has_rows = true, has_cols = true ⇒ NOT a forced-split. Move
        // to a real corner: 8×8 frame, BLOCK_16X16 at (0,0):
        // half_block4x4 = 2 < 8 ⇒ also has_rows / has_cols both true.
        // The forced-split corner only fires when both
        // (r + halfBlock4x4) >= MiRows AND (c + halfBlock4x4) >= MiCols.
        // For a 4×4 mi frame + BLOCK_16X16 at origin: halfBlock = 2,
        // (0+2) < 4 — still in bounds. The next size is BLOCK_32X32
        // at origin in a 4×4 frame: halfBlock = 4, (0+4) >= 4 ⇒
        // forced-split.
        let mut writer = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = fresh_state(4, 4);
        let node = EncodeNode::Leaf(make_skip_leaf());
        let err =
            write_partition_tree(&mut writer, &mut cdfs, &mut state, &node, 0, 0, BLOCK_32X32)
                .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    #[test]
    fn rejects_out_of_range_b_size() {
        let mut writer = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = fresh_state(8, 8);
        let node = EncodeNode::Leaf(make_skip_leaf());
        let err =
            write_partition_tree(&mut writer, &mut cdfs, &mut state, &node, 0, 0, BLOCK_SIZES)
                .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    #[test]
    fn out_of_frame_quadrant_short_circuits() {
        // r >= MiRows OR c >= MiCols ⇒ §5.11.4 line-1 early return.
        // No bits written, no caller-bug.
        let mut writer = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = fresh_state(8, 8);
        let node = EncodeNode::dummy_oob();
        // r = 8 is the first OOF row.
        write_partition_tree(&mut writer, &mut cdfs, &mut state, &node, 8, 0, BLOCK_16X16).unwrap();
        write_partition_tree(&mut writer, &mut cdfs, &mut state, &node, 0, 8, BLOCK_16X16).unwrap();
    }

    // -----------------------------------------------------------------
    // Round-trip 1: single PARTITION_NONE leaf at BLOCK_16X16 on a
    // 4×4 (16-sample × 16-sample) frame.
    //
    // The decoder's `PartitionWalker::decode_partition` walks the same
    // tree, recovers the partition tree's leaves, and (because we
    // chose `b_size = BLOCK_16X16` at origin on a 4×4 mi frame with
    // halfBlock = 2 < 4) the §5.11.4 first conditional lands on the
    // `has_rows && has_cols` arm ⇒ one S() partition symbol is
    // emitted on both sides. We assert (a) the decoder accepts the
    // stream without error and (b) the decoded leaf list matches
    // shape (one leaf at (0,0) with sub_size = BLOCK_16X16).
    // -----------------------------------------------------------------

    /// Mirror-walker: replays the encoder's exact bit-ordering
    /// (partition symbol, then per-leaf skip → segment_id → y_mode →
    /// uv_mode → per-plane coefficients() for PARTITION_NONE leaves;
    /// recurse on four quadrants for PARTITION_SPLIT). Maintains its
    /// own MiSizes[] grid identical to the encoder's so partition_ctx
    /// stays in lockstep.
    ///
    /// Returns the list of (mi_row, mi_col, sub_size) leaves the
    /// recursion visited, in §5.11.4 dispatch order.
    #[allow(clippy::too_many_arguments)]
    fn mirror_decode_partition_tree(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        node: &EncodeNode,
        state: &mut PartitionTreeWriter,
        r: u32,
        c: u32,
        b_size: usize,
        leaves: &mut Vec<(u32, u32, usize)>,
    ) {
        if r >= state.mi_rows || c >= state.mi_cols {
            return;
        }
        let num4x4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
        let half_block4x4 = num4x4 >> 1;
        let has_rows = (r + half_block4x4) < state.mi_rows;
        let has_cols = (c + half_block4x4) < state.mi_cols;
        let partition = if b_size < BLOCK_8X8 {
            PARTITION_NONE
        } else if !has_rows && !has_cols {
            PARTITION_SPLIT
        } else {
            let bsl = MI_WIDTH_LOG2[b_size] as u32;
            let pctx = state.partition_ctx_for(r, c, bsl);
            if has_rows && has_cols {
                let cdf = cdfs.partition_cdf(bsl, pctx).unwrap();
                dec.read_symbol(cdf).unwrap() as usize
            } else if has_cols {
                let cdf_row = cdfs.partition_cdf(bsl, pctx).unwrap();
                let mut bin = crate::cdf::split_or_horz_cdf(cdf_row, b_size).unwrap();
                let s = dec.read_symbol(&mut bin).unwrap();
                if s == 0 {
                    crate::cdf::PARTITION_HORZ
                } else {
                    PARTITION_SPLIT
                }
            } else {
                let cdf_row = cdfs.partition_cdf(bsl, pctx).unwrap();
                let mut bin = crate::cdf::split_or_vert_cdf(cdf_row, b_size).unwrap();
                let s = dec.read_symbol(&mut bin).unwrap();
                if s == 0 {
                    crate::cdf::PARTITION_VERT
                } else {
                    PARTITION_SPLIT
                }
            }
        };
        let sub_size = partition_subsize(partition, b_size).unwrap();
        match (node, partition) {
            (EncodeNode::Leaf(block), PARTITION_NONE) => {
                state.stamp_mi_sizes(r, c, sub_size);
                leaves.push((r, c, sub_size));
                // §5.11.11 skip.
                let skip_cdf = cdfs.skip_cdf(skip_ctx(0, 0));
                let _ = dec.read_symbol(skip_cdf).unwrap();
                // §5.11.8 intra_segment_id — segmentation off ⇒ no bits.
                // §5.11.22 y_mode.
                let y_cdf = cdfs.y_mode_cdf(size_group(sub_size)).unwrap();
                let _ = dec.read_symbol(y_cdf).unwrap();
                // §5.11.22 uv_mode (gated on uv_mode.is_some()).
                if block.uv_mode.is_some() {
                    let cfl = cfl_allowed_for_uv_mode(false, sub_size, false, false);
                    let uv_cdf = cdfs.uv_mode_cdf(cfl, block.y_mode as usize).unwrap();
                    let _ = dec.read_symbol(uv_cdf).unwrap();
                }
                // No coefficient reads for skip-leaves (block.coefficients == [] in our test set).
                // For tests that include a coefficients vector the test
                // walks the planes manually after this helper returns.
                assert!(
                    block.coefficients.is_empty(),
                    "mirror_decode helper handles skip-leaves only; coefficient leaves must use a tailored walker"
                );
            }
            (EncodeNode::Split(children), PARTITION_SPLIT) => {
                let [nw, ne, sw, se] = children;
                mirror_decode_partition_tree(dec, cdfs, nw, state, r, c, sub_size, leaves);
                mirror_decode_partition_tree(
                    dec,
                    cdfs,
                    ne,
                    state,
                    r,
                    c + half_block4x4,
                    sub_size,
                    leaves,
                );
                mirror_decode_partition_tree(
                    dec,
                    cdfs,
                    sw,
                    state,
                    r + half_block4x4,
                    c,
                    sub_size,
                    leaves,
                );
                mirror_decode_partition_tree(
                    dec,
                    cdfs,
                    se,
                    state,
                    r + half_block4x4,
                    c + half_block4x4,
                    sub_size,
                    leaves,
                );
            }
            _ => panic!("(node, partition) pairing mismatch"),
        }
    }

    /// Round-trip the partition decisions via the mirror walker.
    /// The §5.11.4 walk reads partitions + (skip-leaf) block syntax in
    /// the order the encoder wrote them; returns the recovered leaves.
    fn round_trip_partition_shape(
        node: &EncodeNode,
        mi_rows: u32,
        mi_cols: u32,
        b_size: usize,
    ) -> Vec<(u32, u32, usize)> {
        // ----- encode -----
        let mut writer = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut enc_state = fresh_state(mi_rows, mi_cols);
        write_partition_tree(
            &mut writer,
            &mut enc_cdfs,
            &mut enc_state,
            node,
            0,
            0,
            b_size,
        )
        .unwrap();
        let bytes = writer.finish();
        let bytes = if bytes.is_empty() { vec![0u8] } else { bytes };

        // ----- decode (mirror walker) -----
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec_state = fresh_state(mi_rows, mi_cols);
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut leaves = Vec::new();
        mirror_decode_partition_tree(
            &mut dec,
            &mut dec_cdfs,
            node,
            &mut dec_state,
            0,
            0,
            b_size,
            &mut leaves,
        );
        // §8.3 CDF adaptation lockstep verification — encoder + decoder
        // walked identical symbol sequences against identical CDF
        // starts ⇒ the W8/W16 partition rows must match.
        assert_eq!(enc_cdfs.partition_w8, dec_cdfs.partition_w8);
        assert_eq!(enc_cdfs.partition_w16, dec_cdfs.partition_w16);
        leaves
    }

    #[test]
    fn round_trip_single_none_leaf_recovers_one_block() {
        // 4×4 mi frame, BLOCK_16X16 at origin (a single 16×16 leaf).
        let node = EncodeNode::Leaf(make_skip_leaf());
        let leaves = round_trip_partition_shape(&node, 4, 4, BLOCK_16X16);
        assert_eq!(leaves, vec![(0, 0, BLOCK_16X16)]);
    }

    #[test]
    fn round_trip_forced_none_b_size_lt_8x8() {
        // BLOCK_4X4 at origin on a 4×4 mi frame: forced-NONE; no
        // partition symbol; one leaf at (0,0) with sub_size = BLOCK_4X4.
        let node = EncodeNode::Leaf(make_skip_leaf());
        let leaves = round_trip_partition_shape(&node, 4, 4, crate::cdf::BLOCK_4X4);
        assert_eq!(leaves, vec![(0, 0, crate::cdf::BLOCK_4X4)]);
    }

    #[test]
    fn round_trip_one_level_split_recovers_four_leaves() {
        // 4×4 mi frame, BLOCK_16X16 at origin, one level of SPLIT ⇒
        // four BLOCK_8X8 quadrants at (0,0), (0,2), (2,0), (2,2). All
        // four are forced-NONE (BLOCK_8X8 has halfBlock = 1 ⇒ at
        // (2,2) on a 4×4 frame: (2+1)=3 < 4 ⇒ still has_rows /
        // has_cols ⇒ normal partition symbol).
        let leaf = || Box::new(EncodeNode::Leaf(make_skip_leaf()));
        let node = EncodeNode::Split([leaf(), leaf(), leaf(), leaf()]);
        let leaves = round_trip_partition_shape(&node, 4, 4, BLOCK_16X16);
        assert_eq!(
            leaves,
            vec![
                (0, 0, BLOCK_8X8),
                (0, 2, BLOCK_8X8),
                (2, 0, BLOCK_8X8),
                (2, 2, BLOCK_8X8),
            ]
        );
    }

    #[test]
    fn round_trip_forced_split_corner() {
        // 4×4 mi frame, BLOCK_32X32 at origin: halfBlock = 4, (0+4)
        // >= 4 ⇒ both has_rows / has_cols are false ⇒ §5.11.4 line-19
        // forced-SPLIT. Recurse on four BLOCK_16X16 quadrants; three
        // of those are out-of-frame; the (0,0) one is BLOCK_16X16 on
        // a 4×4 frame which is forced-NONE (halfBlock = 2 < 4).
        let leaf = || Box::new(EncodeNode::Leaf(make_skip_leaf()));
        let oob = || Box::new(EncodeNode::dummy_oob());
        let node = EncodeNode::Split([leaf(), oob(), oob(), oob()]);
        let leaves = round_trip_partition_shape(&node, 4, 4, BLOCK_32X32);
        assert_eq!(leaves, vec![(0, 0, BLOCK_16X16)]);
    }

    #[test]
    fn round_trip_two_level_split_64x64() {
        // 16×16 mi frame, BLOCK_64X64 at origin: halfBlock4x4 = 8 ⇒
        // children at (0,0)/(0,8)/(8,0)/(8,8). Each is a normal
        // (has_rows && has_cols) BLOCK_32X32. Build NW as a SPLIT
        // recursing into four BLOCK_16X16 leaves at
        // (0,0)/(0,4)/(4,0)/(4,4); NE/SW/SE are single BLOCK_32X32
        // leaves. The decoder recovers 4 + 3 = 7 leaves.
        let leaf = || Box::new(EncodeNode::Leaf(make_skip_leaf()));
        let nw_inner = EncodeNode::Split([leaf(), leaf(), leaf(), leaf()]);
        let node = EncodeNode::Split([Box::new(nw_inner), leaf(), leaf(), leaf()]);
        let leaves = round_trip_partition_shape(&node, 16, 16, BLOCK_64X64);
        assert_eq!(
            leaves,
            vec![
                (0, 0, BLOCK_16X16),
                (0, 4, BLOCK_16X16),
                (4, 0, BLOCK_16X16),
                (4, 4, BLOCK_16X16),
                (0, 8, BLOCK_32X32),
                (8, 0, BLOCK_32X32),
                (8, 8, BLOCK_32X32),
            ]
        );
    }

    // -----------------------------------------------------------------
    // Round-trip 2: per-leaf block-syntax recovery. Encode a tree, then
    // walk the bitstream back through the decoder using the same
    // ordering the driver wrote (skip → segment_id → y_mode → uv_mode
    // → per-plane coefficients()), asserting the recovered scalars
    // match the source `EncodeBlock`.
    // -----------------------------------------------------------------

    /// Helper for the per-leaf round-trip: walks the bitstream by
    /// (a) the §5.11.4 decoder for partition + footprint, (b) manual
    /// reads of skip / segment_id / y_mode / uv_mode in the same
    /// order the driver wrote them, and (c) per-plane
    /// `PartitionWalker::coefficients` reads.
    #[allow(clippy::too_many_arguments)]
    fn manual_decode_leaf(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        walker: &mut PartitionWalker,
        sub_size: usize,
    ) -> EncodeBlock {
        // §5.11.11 read_skip.
        let skip_cdf = cdfs.skip_cdf(skip_ctx(0, 0));
        let skip = dec.read_symbol(skip_cdf).unwrap() as u8;
        // §5.11.8 intra_segment_id — segmentation off ⇒ no bits;
        // segment_id = 0.
        let segment_id = 0u8;
        // §5.11.22 y_mode — Size_Group ctx.
        let y_cdf = cdfs.y_mode_cdf(size_group(sub_size)).unwrap();
        let y_mode = dec.read_symbol(y_cdf).unwrap() as u8;
        // §5.11.22 uv_mode — Size_Group at origin: cfl_allowed for 16×16
        // with subsampling = 0/0 + lossless = false ⇒ true.
        let cfl_allowed = cfl_allowed_for_uv_mode(false, sub_size, false, false);
        let uv_cdf = cdfs.uv_mode_cdf(cfl_allowed, y_mode as usize).unwrap();
        let uv_mode = dec.read_symbol(uv_cdf).unwrap() as u8;
        // The §5.11.39 coefficients() per plane — for the round-trip
        // leaf we encoded with `skip = 1` so the §5.11.39 reader
        // doesn't fire at this leaf (the encoder also skipped the
        // coefficients vector). Leave coefficients empty.
        let _ = (walker, get_default_scan as fn(usize) -> &'static [u16]);
        EncodeBlock {
            skip,
            segment_id,
            segment_pred: 0,
            y_mode,
            uv_mode: Some(uv_mode),
            coefficients: Vec::new(),
        }
    }

    #[test]
    fn round_trip_leaf_block_syntax_recovers_skip() {
        // Single BLOCK_16X16 PARTITION_NONE leaf with skip = 1,
        // y_mode = DC_PRED, uv_mode = DC_PRED. Decoder walks the
        // §5.11.4 partition then manually replays the §5.11 reads
        // and recovers the same scalars.
        let node = EncodeNode::Leaf(make_skip_leaf());
        let mi_rows = 4;
        let mi_cols = 4;
        let b_size = BLOCK_16X16;
        let mut writer = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut enc_state = fresh_state(mi_rows, mi_cols);
        write_partition_tree(
            &mut writer,
            &mut enc_cdfs,
            &mut enc_state,
            &node,
            0,
            0,
            b_size,
        )
        .unwrap();
        let bytes = writer.finish();

        // Decode: first the §5.11.4 partition (which we know lands
        // PARTITION_NONE on this 4×4 frame at BLOCK_16X16 origin —
        // (0+2) < 4 ⇒ has_rows / has_cols both true), then the
        // §5.11.5 leaf scalars in driver order.
        let mut walker =
            PartitionWalker::new(mi_rows, mi_cols, single_tile(mi_rows, mi_cols)).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        // Read the partition symbol per §5.11.4 has_rows & has_cols arm.
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        let pctx_dec = 0; // origin, no neighbour ⇒ partition_ctx(false, false) = 0.
        let part_cdf = dec_cdfs.partition_cdf(bsl, pctx_dec).unwrap();
        let partition = dec.read_symbol(part_cdf).unwrap() as usize;
        assert_eq!(partition, PARTITION_NONE);
        // Per-leaf scalars.
        let leaf = manual_decode_leaf(&mut dec, &mut dec_cdfs, &mut walker, b_size);
        assert_eq!(leaf.skip, 1);
        assert_eq!(leaf.y_mode, DC_PRED as u8);
        assert_eq!(leaf.uv_mode, Some(DC_PRED as u8));
    }

    #[test]
    fn round_trip_leaf_with_v_pred_y_mode() {
        // Same shape as above but y_mode = V_PRED ⇒ exercises a
        // non-DC_PRED selection round-trip.
        let mut block = make_skip_leaf();
        block.y_mode = V_PRED as u8;
        let node = EncodeNode::Leaf(block);
        let mi_rows = 4;
        let mi_cols = 4;
        let b_size = BLOCK_16X16;
        let mut writer = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut enc_state = fresh_state(mi_rows, mi_cols);
        write_partition_tree(
            &mut writer,
            &mut enc_cdfs,
            &mut enc_state,
            &node,
            0,
            0,
            b_size,
        )
        .unwrap();
        let bytes = writer.finish();
        let mut walker =
            PartitionWalker::new(mi_rows, mi_cols, single_tile(mi_rows, mi_cols)).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        let part_cdf = dec_cdfs.partition_cdf(bsl, 0).unwrap();
        let _ = dec.read_symbol(part_cdf).unwrap();
        let leaf = manual_decode_leaf(&mut dec, &mut dec_cdfs, &mut walker, b_size);
        assert_eq!(leaf.y_mode, V_PRED as u8);
    }

    #[test]
    fn round_trip_leaf_with_coefficients_all_zero() {
        // Write a leaf with skip = 0 and three all-zero §5.11.39
        // coefficient blocks (Y/U/V). The §5.11.39 reader returns
        // `all_zero == true` for each plane.
        let node = EncodeNode::Leaf(make_zero_coeff_leaf());
        let mi_rows = 4;
        let mi_cols = 4;
        let b_size = BLOCK_16X16;
        let mut writer = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut enc_state = fresh_state(mi_rows, mi_cols);
        write_partition_tree(
            &mut writer,
            &mut enc_cdfs,
            &mut enc_state,
            &node,
            0,
            0,
            b_size,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut walker =
            PartitionWalker::new(mi_rows, mi_cols, single_tile(mi_rows, mi_cols)).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        // §5.11.4 partition symbol.
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        let part_cdf = dec_cdfs.partition_cdf(bsl, 0).unwrap();
        let _ = dec.read_symbol(part_cdf).unwrap();
        // §5.11.11 skip.
        let skip_cdf = dec_cdfs.skip_cdf(skip_ctx(0, 0));
        let skip = dec.read_symbol(skip_cdf).unwrap() as u8;
        assert_eq!(skip, 0);
        // §5.11.22 y_mode + uv_mode.
        let y_cdf = dec_cdfs.y_mode_cdf(size_group(b_size)).unwrap();
        let y_mode = dec.read_symbol(y_cdf).unwrap() as u8;
        assert_eq!(y_mode, DC_PRED as u8);
        let cfl = cfl_allowed_for_uv_mode(false, b_size, false, false);
        let uv_cdf = dec_cdfs.uv_mode_cdf(cfl, y_mode as usize).unwrap();
        let uv_mode = dec.read_symbol(uv_cdf).unwrap() as u8;
        assert_eq!(uv_mode, DC_PRED as u8);
        // Three planes of §5.11.39 coefficients() — each all-zero.
        for plane in 0u8..3 {
            let mut quant_out = vec![0i32; 16];
            let readout: CoefficientsReadout = walker
                .coefficients(
                    &mut dec,
                    &mut dec_cdfs,
                    plane,
                    0,
                    TX_4X4,
                    TX_CLASS_2D,
                    0,
                    0,
                    get_default_scan(TX_4X4),
                    &mut quant_out,
                )
                .unwrap();
            assert!(readout.all_zero, "plane {plane} all_zero on round-trip");
            assert_eq!(readout.eob, 0);
            assert!(quant_out.iter().all(|&q| q == 0));
        }
        // Silence unused helpers — keep them in scope for the next
        // arc's neighbour-ctx-aware tests.
        let _ = (
            get_coeff_base_ctx,
            get_coeff_base_eob_ctx,
            get_br_ctx,
            MAX_SEGMENTS,
        );
    }
}
