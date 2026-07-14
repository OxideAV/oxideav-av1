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
    block_height, block_width, cfl_allowed_for_uv_mode, compute_tx_type, find_tx_size,
    get_plane_residual_size, get_tx_size, inter_tx_type_set, intra_mode_ctx, intra_tx_type_set,
    is_inter_ctx, palette_tokens_args, partition_ctx, partition_subsize, segment_id_ctx,
    size_group, skip_ctx, skip_mode_ctx, tx_size_sqr_index, EncoderBlockSyntaxStamp,
    FrameInterOrderHints, InterpolationFilterReadout, MotionFieldMvs, PalettePlane,
    PartitionWalker, TileCdfContext, TileGeometry, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8,
    BLOCK_INVALID, BLOCK_SIZES, DCT_DCT, DC_PRED, EIGHTTAP, FRAME_LF_COUNT, GM_TYPE_IDENTITY,
    GM_TYPE_TRANSLATION, H_ADST, H_DCT, H_FLIPADST, MAX_SEGMENTS, MAX_TX_SIZE_RECT,
    MAX_VARTX_DEPTH, MI_HEIGHT_LOG2, MI_SIZE, MI_SIZE_LOG2, MI_WIDTH_LOG2, MODE_GLOBALMV,
    MODE_GLOBAL_GLOBALMV, MODE_NEARESTMV, MODE_NEAREST_NEARESTMV, MODE_NEARMV, MODE_NEWMV,
    MODE_NEW_NEWMV, MOTION_MODE_SIMPLE, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PALETTE_COLORS,
    PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, SPLIT_TX_SIZE, SWITCHABLE,
    TX_4X4, TX_CLASS_2D, TX_CLASS_HORIZ, TX_CLASS_VERT, TX_HEIGHT, TX_SIZES_ALL, TX_SIZE_SQR_UP,
    TX_WIDTH, UV_CFL_PRED, V_ADST, V_DCT, V_FLIPADST, WARPEDMODEL_PREC_BITS,
};
use crate::encoder::block_mode_info::{
    assign_mv_pred_mv, write_cdef, write_cfl_alphas, write_delta_lf, write_delta_qindex,
    write_inter_block_mode_info, write_inter_frame_mode_info_prefix,
    write_intra_block_mode_info_with_palette, write_intra_frame_else_arm,
    write_intra_frame_intrabc_arm, write_intra_frame_y_mode, write_intra_segment_id,
    write_intra_uv_mode, write_palette_tokens_plane, write_skip, write_y_mode,
    InterBlockModeInfoTail, InterFrameDeltaSiteInputs, IntrabcArmInputs,
};
use crate::encoder::coefficients::write_coefficients;
use crate::encoder::partition::write_partition;
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::transform_tree::write_block_tx_size;
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
    /// §5.11.45 `CflAlphaU` — the signed U-channel CFL alpha. `None`
    /// when `uv_mode != UV_CFL_PRED` (the §5.11.22 gate skips the
    /// §5.11.45 reader entirely). When `Some`, the value MUST be in
    /// `-16..=-1 | 0 | 1..=16` per §5.11.45's `CflAlpha = ±(1 +
    /// cfl_alpha_*)` derivation (`0` corresponds to `signU == CFL_SIGN_ZERO`
    /// — no `cfl_alpha_u` symbol emitted). Both alphas being `0` is
    /// prohibited by §6.10.36 (the joint-sign `cfl_alpha_signs` table
    /// excludes the `(ZERO, ZERO)` combination as redundant with
    /// `UV_DC_PRED`), so when `Some` at least one of `cfl_alpha_u` /
    /// `cfl_alpha_v` is non-zero. Added r231.
    pub cfl_alpha_u: Option<i8>,
    /// §5.11.45 `CflAlphaV` — mirror of [`Self::cfl_alpha_u`] for the
    /// V channel. Added r231.
    pub cfl_alpha_v: Option<i8>,
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
            cfl_alpha_u: None,
            cfl_alpha_v: None,
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
        // §5.11.22 line 8: `if ( UVMode == UV_CFL_PRED ) read_cfl_alphas()`
        // — r231. The encoder's CFL picker (in
        // [`crate::encoder::pixel_driver`]) committed to a specific
        // (CflAlphaU, CflAlphaV) when it chose UV_CFL_PRED; replay it
        // here through `write_cfl_alphas` so the decoder reads the
        // same scalars.
        if uv_mode as usize == UV_CFL_PRED {
            let alpha_u = block.cfl_alpha_u.ok_or(Error::PartitionWalkOutOfRange)?;
            let alpha_v = block.cfl_alpha_v.ok_or(Error::PartitionWalkOutOfRange)?;
            write_cfl_alphas(writer, cdfs, alpha_u, alpha_v)?;
        }
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

// ---------------------------------------------------------------------
// r282 — full §5.11.7 block-syntax threading. The write twin of the
// r281 decode-side composition: the §5.11.4 partition recursion below
// emits, at every leaf, the complete §5.11.7 `intra_frame_mode_info()`
// body (prefix → `use_intrabc` dispatch → both arms → §5.11.49
// `palette_tokens()`), deriving every §8.3.2 neighbour context from a
// [`PartitionWalker`] mirror it stamps in lockstep with the
// [`crate::cdf::PartitionWalker::decode_block_syntax`] walker's own
// grid-fill. Feeding the produced bytes into
// [`crate::cdf::PartitionWalker::decode_partition_syntax`] consumes
// exactly the emitted bits (sync-sentinel lockstep).
// ---------------------------------------------------------------------

/// §5.11.46 / §5.11.49 palette inputs for one [`SyntaxBlock`] leaf —
/// the encoder commitments the §5.11.7 else-arm writer and the
/// §5.11.49 `palette_tokens()` writer replay.
///
/// `size_y == 0` / `size_uv == 0` mean "no palette on that plane"
/// (the §5.11.46 `has_palette_{y,uv} = 0` arms); the corresponding
/// colour / map fields are then ignored.
#[derive(Debug, Clone, Default)]
pub struct SyntaxPalette {
    /// §5.11.46 `PaletteSizeY` — `0` or `2..=PALETTE_COLORS`.
    pub size_y: u8,
    /// §5.11.46 `palette_colors_y[ 0..PaletteSizeY ]` (sorted
    /// ascending per the §5.11.46 cache-merge invariant).
    pub colors_y: [u16; PALETTE_COLORS],
    /// §5.11.49 target `ColorMapY` — row-major, stride
    /// `Block_Width[ MiSize ]`, on-screen samples `< size_y`.
    pub color_map_y: Vec<u8>,
    /// §5.11.46 `PaletteSizeUV` — `0` or `2..=PALETTE_COLORS`.
    pub size_uv: u8,
    /// §5.11.46 `palette_colors_u[ 0..PaletteSizeUV ]`.
    pub colors_u: [u16; PALETTE_COLORS],
    /// §5.11.46 `palette_colors_v[ 0..PaletteSizeUV ]`.
    pub colors_v: [u16; PALETTE_COLORS],
    /// §5.11.46 `delta_encode_palette_colors_v` arm selector.
    pub delta_encode_v: bool,
    /// §5.11.49 target `ColorMapUV` — row-major with the §5.11.49
    /// subsampled (+`<4`-bumped) stride; see
    /// [`crate::cdf::palette_tokens_args`].
    pub color_map_uv: Vec<u8>,
}

/// One §5.11.5 leaf of the full-syntax write tree — the per-block
/// §5.11.7 scalars the encoder committed and the
/// [`write_partition_tree_syntax`] driver replays bit-for-bit.
///
/// ## Scope (r283)
///
/// * `skip == 0` is supported on the §5.11.7 `else` (intra) arm as of
///   r283: the §5.11.34 `residual()` write side threads the per-TU
///   §5.11.39 coefficient emission (with the §5.11.47 / §5.11.40
///   tx-type mirror) in decode-walker dispatch order, consuming
///   [`Self::residual_quant`]. The intra-block-copy arm
///   ([`Self::intrabc_mv`]` == Some`) still requires `skip == 1` —
///   the §5.11.36 inter-luma `transform_tree` write recursion is the
///   follow-up arc.
/// * The §5.11.16 `read_block_tx_size()` call is bit-silent on the
///   supported configuration (`tx_mode_select == false`); see
///   [`SyntaxFrameParams::tx_mode_select`].
#[derive(Debug, Clone)]
pub struct SyntaxBlock {
    /// §5.11.11 `skip`. `0` or `1` (`0` requires the intra `else` arm
    /// + [`Self::residual_quant`]; see scope note).
    pub skip: u8,
    /// §5.11.8 `segment_id` in `0..=last_active_seg_id` (and `0` when
    /// segmentation is disabled). On the post-skip arm (`SegIdPreSkip
    /// == 0`) the §5.11.9 skip short-circuit forces `segment_id ==
    /// pred` (the writer's mirror derives `pred`; a mismatch is a
    /// caller bug).
    pub segment_id: u8,
    /// §5.11.56 `cdef_idx` to commit when this leaf opens a 64×64
    /// anchor. Unreachable while `skip == 1` (the §5.11.56
    /// short-circuit) — kept for parameter-surface completeness.
    pub cdef_idx: i8,
    /// §5.11.12 signed `reducedDeltaQIndex` (`0` ⇒ no delta coded).
    pub reduced_delta_q_index: i32,
    /// §5.11.13 signed `reducedDeltaLfLevel` per LF index.
    pub reduced_delta_lf: [i32; FRAME_LF_COUNT],
    /// §5.11.7 `use_intrabc` arm: `Some(Mv[ 0 ])` (`[ row, col ]`,
    /// 1/8-luma-sample units, integer-aligned per `force_integer_mv`)
    /// selects the intra-block-copy arm; `None` the `else` arm.
    /// Requires [`SyntaxFrameParams::allow_intrabc`].
    pub intrabc_mv: Option<[i32; 2]>,
    /// §5.11.7 `intra_frame_y_mode` (`0..INTRA_MODES`). Ignored on the
    /// intrabc arm (forced `DC_PRED`).
    pub y_mode: u8,
    /// §5.11.22-shaped `uv_mode` — `Some` exactly when the §5.11.5
    /// prologue derives `HasChroma` for this leaf, `None` otherwise.
    pub uv_mode: Option<u8>,
    /// §5.11.42 `AngleDeltaY` (`-3..=3`; `0` for non-directional).
    pub angle_delta_y: i8,
    /// §5.11.43 `AngleDeltaUV`.
    pub angle_delta_uv: i8,
    /// §5.11.45 `CflAlphaU` — `Some` iff `uv_mode == UV_CFL_PRED`.
    pub cfl_alpha_u: Option<i8>,
    /// §5.11.45 `CflAlphaV` — `Some` iff `uv_mode == UV_CFL_PRED`.
    pub cfl_alpha_v: Option<i8>,
    /// §5.11.24 `use_filter_intra` (only consulted when the §5.11.24
    /// outer gate opens).
    pub use_filter_intra: u8,
    /// §5.11.24 `filter_intra_mode` — `Some` iff `use_filter_intra == 1`.
    pub filter_intra_mode: Option<u8>,
    /// §5.11.46 / §5.11.49 palette commitments.
    pub palette: SyntaxPalette,
    /// §5.11.34 `residual()` per-TU quantized-coefficient commitments
    /// — one `Quant[]` array (row-major, length `Tx_Width[txSz] *
    /// Tx_Height[txSz]`) per §5.11.35 `transform_block` the §5.11.34
    /// dispatch visits, in dispatch order (chunk-row → chunk-col →
    /// plane → TU-row → TU-col). TUs clipped by the §5.11.35
    /// `startX >= maxX || startY >= maxY` early return are NOT
    /// represented (the reader never visits them). MUST be empty when
    /// `skip == 1` (the §5.11.35 `!skip` gate never fires) and have
    /// exactly the visited-TU count when `skip == 0` — both mismatches
    /// are caller bugs ([`crate::Error::PartitionWalkOutOfRange`]).
    /// Added r283.
    pub residual_quant: Vec<Vec<i32>>,
    /// §5.11.15 `TxSize` commitment for the §5.11.16 `else` arm
    /// (r285). `None` ⇒ the spec-forced default (`TX_4X4` when the
    /// segment is lossless, `Max_Tx_Size_Rect[ MiSize ]` otherwise).
    /// `Some(t)` selects `t` when the §5.11.15 `tx_depth` S() fires
    /// (`TxMode == TX_MODE_SELECT && MiSize > BLOCK_4X4 &&
    /// allowSelect && !Lossless`); `t` must be reachable from
    /// `Max_Tx_Size_Rect[ MiSize ]` within `MAX_TX_DEPTH = 2`
    /// `Split_Tx_Size` steps. On a bit-silent arm a `Some(t)` that
    /// differs from the spec-forced value is a caller bug. MUST be
    /// `None` when the §5.11.16 var-tx arm fires (the recursion's
    /// last terminal-else `txSz` is the block's `TxSize` there).
    pub tx_size: Option<u8>,
    /// §5.11.47 per-luma-TU `TxType` commitment (r286) — one §3 `TxType`
    /// ordinal per §5.11.35 luma (`plane == 0`) `transform_block` the
    /// §5.11.34 dispatch visits on the `!skip` arm, in dispatch order
    /// (the SAME visit order + clipping rules as [`Self::residual_quant`],
    /// but **luma TUs only** — chroma derives its `PlaneTxType` from the
    /// §5.11.40 `Mode_To_Txfm[UVMode]` / `TxTypes[]` fallback without a
    /// per-TU symbol). The §5.11.47 write side emits the
    /// `intra_tx_type` / `inter_tx_type` S() for this `TxType` when the
    /// `set > 0 && qIdx > 0` guard opens, and stamps it into the
    /// mirror's `TxTypes[]`. When the guard is closed (the §5.11.48
    /// `TX_SET_DCTONLY` set, or `base_q_idx == 0`) the committed value
    /// MUST be `DCT_DCT` — a mismatch is a caller bug
    /// ([`Error::PartitionWalkOutOfRange`]). The committed `TxType` MUST
    /// be admissible in the resolved §5.11.48 `set` (the
    /// `is_tx_type_in_set` predicate) on the guard-open arm. MUST be
    /// empty when `skip == 1`. Empty also leaves every luma TU at
    /// `DCT_DCT` (the back-compat default before per-TU tx-type
    /// commitments existed).
    pub residual_tx_type: Vec<u8>,
    /// §5.11.17 variable-transform split-decision trees for the
    /// §5.11.16 var-tx arm (`TxMode == TX_MODE_SELECT && MiSize >
    /// BLOCK_4X4 && is_inter && !skip && !Lossless`), one
    /// [`VarTxSyntaxTree`] per `(txH4, txW4)` sub-rectangle of the
    /// block footprint in §5.11.16 loop order (row-major). Every
    /// loop position gets an entry, including positions clipped by
    /// the §5.11.17 `row >= MiRows || col >= MiCols` early return
    /// (those must be [`VarTxSyntaxTree::Leaf`] — the reader emits
    /// no symbol and stamps nothing there). MUST be empty when the
    /// arm does not fire. Added r285.
    pub var_tx_trees: Vec<VarTxSyntaxTree>,
    /// §5.11.23 inter-leaf commitments (r411). `Some` selects the
    /// §5.11.18 `is_inter == 1` arm on an inter-frame walk
    /// ([`SyntaxFrameParams::inter`]` == Some`); `None` there selects
    /// the §5.11.22 `intra_block_mode_info()` arm (the intra fields
    /// above are then consumed exactly like the §5.11.7 else arm,
    /// with the `y_mode` S() against `TileYModeCdf[ Size_Group ]`).
    /// MUST be `None` on an intra-frame walk.
    pub inter: Option<SyntaxInterBlock>,
}

/// §5.11.23 / §5.11.31 commitments for one inter leaf of the write
/// tree (r411) — the scalars the §5.11.18 `is_inter == 1` arm of
/// [`write_block_syntax`] replays bit-for-bit.
///
/// ## r411 scope
///
/// Single-reference prediction only (`ref_frame[1] == NONE = -1`):
/// the four §5.11.24 single-pred modes. Non-NEWMV modes carry no MV
/// bits — the reader derives `Mv[0] = PredMv[0]` from the §7.10.2
/// stack, and the writer cross-checks [`Self::mv`] against the same
/// derivation (`GlobalMvs[0]` for `GLOBALMV`, `RefStackMv[RefMvIdx][0]`
/// for `NEARESTMV` / `NEARMV`); a mismatch is a caller bug — the
/// encoder would have reconstructed with a vector the decoder never
/// sees.
#[derive(Debug, Clone)]
pub struct SyntaxInterBlock {
    /// §5.11.25 target `RefFrame[0..2]` — slot 0 in
    /// `LAST_FRAME..=ALTREF_FRAME = 1..=7`, slot 1 fixed at
    /// `NONE = -1` (r411 single-reference scope).
    pub ref_frame: [i8; 2],
    /// §5.11.23 `YMode` — [`MODE_NEARESTMV`] / [`MODE_NEARMV`] /
    /// [`MODE_GLOBALMV`] / [`MODE_NEWMV`].
    pub y_mode: u8,
    /// §5.11.31 `Mv[ list ][ 0..2 ]` (`[ row, col ]`, 1/8-luma-sample
    /// units, `Abs < 1 << 14` per §6.10.27). Only list 0 is consulted
    /// on the single-reference scope.
    pub mv: [[i32; 2]; 2],
    /// §5.11.23 `RefMvIdx` — threaded to `write_drl_mode` +
    /// `assign_mv_pred_mv`.
    pub ref_mv_idx: u32,
    /// r412 — the committed §5.11.x `interp_filter[ 0..2 ]` pair AS
    /// THE READER DERIVES IT: on a non-SWITCHABLE frame filter both
    /// slots MUST equal the frame value; on SWITCHABLE with
    /// `needs_interp_filter( )` false both slots MUST be `EIGHTTAP`;
    /// otherwise slot 0 (and slot 1 iff `enable_dual_filter`) carries
    /// the coded choice in `EIGHTTAP..=EIGHTTAP_SHARP` (with
    /// `!enable_dual_filter` mirroring slot 0 into slot 1). Any other
    /// shape is rejected by the §5.11.x writer.
    pub interp_filter: [u8; 2],
    /// r413 — §5.11.10 `skip_mode` (0 or 1). `1` selects the
    /// §5.11.18/§5.11.23 skip-mode arm: requires
    /// [`SyntaxInterFrameParams::skip_mode_present`], a block with
    /// both dimensions >= 8, `skip == 1`, [`Self::ref_frame`] equal to
    /// the frame's `SkipModeFrame[]` pair, [`Self::y_mode`] ==
    /// `NEAREST_NEARESTMV`, `ref_mv_idx == 0`, and (per
    /// `needs_interp_filter( ) == 0`) an `[EIGHTTAP, EIGHTTAP]`
    /// committed filter pair. The §5.11.10 `skip_mode` S() against the
    /// §8.3.2 neighbour ctx is the ONLY symbol the mode-info body
    /// codes for such a block.
    pub skip_mode: u8,
}

/// Frame-scope inter state for a §5.11.18 write walk (r411) — the
/// write-side bundle of [`crate::cdf::InterFrameContext`]'s
/// syntax-relevant scalars (field names and contracts match). `Some`
/// on [`SyntaxFrameParams::inter`] switches [`write_block_syntax`]
/// from the §5.11.7 `intra_frame_mode_info()` arm to the §5.11.18
/// `inter_frame_mode_info()` arm.
#[derive(Debug, Clone)]
pub struct SyntaxInterFrameParams {
    /// §5.9.22 `skip_mode_present`. r413: `true` opens the per-block
    /// §5.11.10 `skip_mode` S() (committed via
    /// [`SyntaxInterBlock::skip_mode`]) on every >= 8×8 leaf; the
    /// short-circuit arms still force `skip_mode = 0`.
    pub skip_mode_present: bool,
    /// §5.9.22 `SkipModeFrame[ 0..2 ]`.
    pub skip_mode_frame: [i8; 2],
    /// §5.9.23 `reference_select`.
    pub reference_select: bool,
    /// §5.9.24 `GmType[ 0..8 ]`.
    pub gm_type: [i32; 8],
    /// §5.9.24 `gm_params[ 0..8 ][ 0..6 ]`.
    pub gm_params: [[i32; 6]; 8],
    /// §7.8 `RefFrameSignBias[ 0..8 ]`.
    pub ref_frame_sign_bias: [i32; 8],
    /// §5.9.2 `allow_high_precision_mv`.
    pub allow_high_precision_mv: bool,
    /// §5.9.2 `force_integer_mv` (post-override value).
    pub force_integer_mv: bool,
    /// §5.9.2 `use_ref_frame_mvs`.
    pub use_ref_frame_mvs: bool,
    /// §5.9.2 `is_motion_mode_switchable`.
    pub is_motion_mode_switchable: bool,
    /// §5.9.14 `segmentation_update_map` (r413) — with
    /// [`SyntaxFrameParams::segmentation_enabled`] set, `true` codes
    /// the per-block §5.11.20 `read_segment_id()` S() (the
    /// `PRIMARY_REF_NONE` configuration forces it to 1); `false`
    /// adopts `predictedSegmentId` with no bits.
    pub segmentation_update_map: bool,
    /// §5.9.14 `segmentation_temporal_update` (r413) — MUST be
    /// `false` on the current scope (the §5.11.19 `seg_id_predicted`
    /// S() + `PrevSegmentIds` threading is a follow-up arc).
    pub segmentation_temporal_update: bool,
    /// §5.9.2 `allow_warped_motion`.
    pub allow_warped_motion: bool,
    /// §5.11.27 `is_scaled( LAST_FRAME + i )` per reference.
    pub is_scaled_per_ref: [bool; 7],
    /// §5.5.2 `enable_interintra_compound`.
    pub enable_interintra_compound: bool,
    /// §5.5.2 `enable_masked_compound`.
    pub enable_masked_compound: bool,
    /// §5.5.2 `enable_jnt_comp`.
    pub enable_jnt_comp: bool,
    /// §5.9.2 `OrderHint` / `OrderHints[]` bundle.
    pub order_hints: FrameInterOrderHints,
    /// §5.9.10 frame-header `interpolation_filter` in
    /// `EIGHTTAP..=SWITCHABLE = 0..=4`. r411 scope: non-SWITCHABLE
    /// only (the §5.11.x filter loop stays bit-silent).
    pub interpolation_filter: u8,
    /// §5.5.2 `enable_dual_filter`.
    pub enable_dual_filter: bool,
    /// §7.9 motion-field grid for the §7.10.2.5 temporal scan —
    /// all-invalid when `use_ref_frame_mvs == false`.
    pub motion_field_mvs: MotionFieldMvs,
}

impl SyntaxInterFrameParams {
    /// r411 P-frame baseline: single reference, identity global
    /// motion, every optional tool gated shut, `EIGHTTAP` frame
    /// filter, no order hints, no temporal MVs.
    #[must_use]
    pub fn single_ref_baseline(mi_rows: u32, mi_cols: u32, force_integer_mv: bool) -> Self {
        let mut gm_params = [[0i32; 6]; 8];
        for row in gm_params.iter_mut() {
            row[2] = 1 << WARPEDMODEL_PREC_BITS;
            row[5] = 1 << WARPEDMODEL_PREC_BITS;
        }
        SyntaxInterFrameParams {
            skip_mode_present: false,
            skip_mode_frame: [0, 0],
            reference_select: false,
            gm_type: [GM_TYPE_IDENTITY; 8],
            gm_params,
            ref_frame_sign_bias: [0; 8],
            allow_high_precision_mv: false,
            force_integer_mv,
            use_ref_frame_mvs: false,
            is_motion_mode_switchable: false,
            segmentation_update_map: false,
            segmentation_temporal_update: false,
            allow_warped_motion: false,
            is_scaled_per_ref: [false; 7],
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_jnt_comp: false,
            order_hints: FrameInterOrderHints::IDENTITY,
            interpolation_filter: EIGHTTAP,
            enable_dual_filter: false,
            motion_field_mvs: MotionFieldMvs::new_invalid(mi_rows, mi_cols),
        }
    }
}

/// One node of a §5.11.17 `read_var_tx_size` split-decision tree on
/// the write side (r285). Unlike
/// [`crate::encoder::transform_tree::VarTxNode`] (the stateless
/// caller-supplies-ctx surface), this shape carries no ctx — the
/// [`write_block_syntax`] driver derives every node's §8.3.2
/// `txfm_split` ctx live from its mirror walker (the ctx is a
/// function of the running `InterTxSizes[]` state, which earlier
/// leaves of the same recursion feed) and stamps the mirror in
/// §5.11.17 visit order, exactly like the decode walker.
#[derive(Debug, Clone)]
pub enum VarTxSyntaxTree {
    /// §5.11.17 terminal `else` arm: `txfm_split = 0` (the S() is
    /// emitted only when the node is not at the spec-forced terminal
    /// `txSz == TX_4X4 || depth == MAX_VARTX_DEPTH`), then
    /// `InterTxSizes[..] = txSz` over the node's footprint.
    Leaf,
    /// §5.11.17 `if ( txfm_split )` arm: `txfm_split = 1` plus the
    /// `Split_Tx_Size[ txSz ]` recursion. The child count must match
    /// the spec loop's `(h4 / stepH) * (w4 / stepW)` visit count
    /// (4 for square ordinals, 2 for rectangular ones), in row-major
    /// `(i, j)` order.
    Split(Vec<VarTxSyntaxTree>),
}

impl SyntaxBlock {
    /// Baseline leaf: `skip = 1`, segment 0, the §5.11.7 `else` arm
    /// with the given Y/UV modes, no angle deltas, no CFL, no
    /// filter-intra, no palette, no deltas.
    #[must_use]
    pub fn skip_leaf(y_mode: u8, uv_mode: Option<u8>) -> Self {
        SyntaxBlock {
            skip: 1,
            segment_id: 0,
            cdef_idx: 0,
            reduced_delta_q_index: 0,
            reduced_delta_lf: [0; FRAME_LF_COUNT],
            intrabc_mv: None,
            y_mode,
            uv_mode,
            angle_delta_y: 0,
            angle_delta_uv: 0,
            cfl_alpha_u: None,
            cfl_alpha_v: None,
            use_filter_intra: 0,
            filter_intra_mode: None,
            palette: SyntaxPalette::default(),
            residual_quant: Vec::new(),
            tx_size: None,
            residual_tx_type: Vec::new(),
            var_tx_trees: Vec::new(),
            inter: None,
        }
    }
}

/// One node of the full-syntax §5.11.4 write tree. Same recursive
/// shape as [`EncodeNode`] (PARTITION_NONE leaves + PARTITION_SPLIT
/// quadrants; the asymmetric partitions remain the [`write_partition`]
/// alphabet's follow-up dispatch work).
#[derive(Debug, Clone)]
pub enum SyntaxNode {
    /// Single §5.11.5 block at this node's `bSize`.
    Leaf(Box<SyntaxBlock>),
    /// Four §5.11.4 quadrants at `Partition_Subsize[PARTITION_SPLIT][bSize]`
    /// in `[NW, NE, SW, SE]` order.
    Split([Box<SyntaxNode>; 4]),
    /// r412 — §5.11.4 `PARTITION_HORZ`: two `decode_block( )` leaves
    /// at `Partition_Subsize[ PARTITION_HORZ ][ bSize ]` in
    /// `[ top, bottom ]` order. Scope: fully-in-frame nodes only
    /// (`hasRows && hasCols` — the `split_or_horz` edge arm remains
    /// the forced-SPLIT driver's territory).
    Horz([Box<SyntaxBlock>; 2]),
    /// r412 — §5.11.4 `PARTITION_VERT`: two leaves in
    /// `[ left, right ]` order (same scope as [`SyntaxNode::Horz`]).
    Vert([Box<SyntaxBlock>; 2]),
}

impl SyntaxNode {
    /// Sentinel for an out-of-frame quadrant (`r >= MiRows || c >=
    /// MiCols`) — short-circuited per §5.11.4 line 1 before
    /// inspection.
    #[must_use]
    pub fn dummy_oob() -> Self {
        SyntaxNode::Leaf(Box::new(SyntaxBlock::skip_leaf(DC_PRED as u8, None)))
    }
}

/// Frame- / sequence-level scalars the §5.11.5 syntax walk threads
/// into every leaf — the write-side bundle of
/// [`crate::cdf::PartitionWalker::decode_block_syntax`]'s parameter
/// list (field names and contracts match one-to-one).
#[derive(Debug, Clone)]
pub struct SyntaxFrameParams {
    /// §5.5.2 `subsampling_x` (0 or 1).
    pub subsampling_x: u8,
    /// §5.5.2 `subsampling_y` (0 or 1).
    pub subsampling_y: u8,
    /// §5.5.2 `NumPlanes` (1 or 3).
    pub num_planes: u8,
    /// §5.9.14 `SegIdPreSkip`.
    pub seg_id_pre_skip: bool,
    /// §5.9.14 `segmentation_enabled`.
    pub segmentation_enabled: bool,
    /// §5.11.11 `SegIdPreSkip && seg_feature_active( SEG_LVL_SKIP )`.
    pub seg_skip_active: bool,
    /// §5.9.14 `LastActiveSegId`.
    pub last_active_seg_id: u8,
    /// §6.8.2 per-segment `LosslessArray[]`.
    pub lossless_array: [bool; MAX_SEGMENTS],
    /// §6.8.2 `CodedLossless`.
    pub coded_lossless: bool,
    /// §5.5.2 `enable_cdef`.
    pub enable_cdef: bool,
    /// §5.9.20 `allow_intrabc`.
    pub allow_intrabc: bool,
    /// §5.9.19 `cdef_bits` (`0..=3`).
    pub cdef_bits: u32,
    /// §6.10.4 `ReadDeltas`.
    pub read_deltas: bool,
    /// §5.5.1 `use_128x128_superblock`.
    pub use_128x128_superblock: bool,
    /// §5.9.17 `delta_q_res` (`0..=3`).
    pub delta_q_res: u8,
    /// §5.9.18 `delta_lf_present`.
    pub delta_lf_present: bool,
    /// §5.9.18 `delta_lf_multi`.
    pub delta_lf_multi: bool,
    /// §5.5.2 `mono_chrome`.
    pub mono_chrome: bool,
    /// §5.9.18 `delta_lf_res` (`0..=3`).
    pub delta_lf_res: u8,
    /// §5.9.5 `allow_screen_content_tools` — the §5.11.46 outer gate.
    pub allow_screen_content_tools: bool,
    /// §5.5.2 `enable_filter_intra` — the §5.11.24 outer gate.
    pub enable_filter_intra: bool,
    /// §5.5.2 `BitDepth` (8 / 10 / 12).
    pub bit_depth: u8,
    /// §5.9.21 `TxMode == TX_MODE_SELECT`. Threaded into the
    /// §5.11.16 write driver (r285): on the intra / `skip == 1` arms
    /// it opens the §5.11.15 `tx_depth` S() (committed via
    /// [`SyntaxBlock::tx_size`]); on the inter `skip == 0` arm it
    /// opens the §5.11.17 `txfm_split` recursion (committed via
    /// [`SyntaxBlock::var_tx_trees`]). With `TX_MODE_SELECT` off the
    /// §5.11.16 call is bit-silent on both sides.
    pub tx_mode_select: bool,
    /// §5.9.12 per-frame quantiser state — the §7.12.2 `get_qindex`
    /// driver the §5.11.47 `transform_type()` write side reads for its
    /// `(segmentation_enabled ? get_qindex(1, segment_id) : base_q_idx)
    /// > 0` symbol-coding guard (r286). With the `base_q_idx == 0`
    /// neutral default the guard is always false and the §5.11.47 path
    /// stays bit-silent (`TxType == DCT_DCT`), matching the prior
    /// hard-coded behaviour exactly. Threading a non-zero `base_q_idx`
    /// opens the per-luma-TU `intra_tx_type` / `inter_tx_type` S().
    pub quant: crate::cdf::QuantizerParams,
    /// §5.9.21 `reduced_tx_set` — the §5.11.48 `get_tx_set()` reduction
    /// flag (`reducedTxSet`) the §5.11.47 write side feeds into
    /// [`crate::cdf::intra_tx_type_set`] / [`crate::cdf::inter_tx_type_set`]
    /// so the chosen `set` (and therefore the `intra_tx_type` /
    /// `inter_tx_type` symbol alphabet) agrees with the decode walker.
    /// Added r286.
    pub reduced_tx_set: bool,
    /// §5.9.4 `FrameIsIntra == 0` selector (r411): `Some` switches the
    /// §5.11.6 `mode_info()` dispatch to the §5.11.18
    /// `inter_frame_mode_info()` arm and carries the frame-scope inter
    /// state; `None` keeps the §5.11.7 intra-frame arm.
    pub inter: Option<SyntaxInterFrameParams>,
}

impl SyntaxFrameParams {
    /// 8-bit 4:4:4 intra-only baseline: 3 planes, no subsampling,
    /// segmentation / deltas / intrabc / screen-content / filter-intra
    /// off, CDEF formally enabled with `cdef_bits = 0`.
    #[must_use]
    pub fn intra_8bit_baseline() -> Self {
        SyntaxFrameParams {
            subsampling_x: 0,
            subsampling_y: 0,
            num_planes: 3,
            seg_id_pre_skip: false,
            segmentation_enabled: false,
            seg_skip_active: false,
            last_active_seg_id: 0,
            lossless_array: [false; MAX_SEGMENTS],
            coded_lossless: false,
            enable_cdef: true,
            allow_intrabc: false,
            cdef_bits: 0,
            read_deltas: false,
            use_128x128_superblock: false,
            delta_q_res: 0,
            delta_lf_present: false,
            delta_lf_multi: false,
            mono_chrome: false,
            delta_lf_res: 0,
            allow_screen_content_tools: false,
            enable_filter_intra: false,
            bit_depth: 8,
            tx_mode_select: false,
            quant: crate::cdf::QuantizerParams::neutral(0, 8),
            reduced_tx_set: false,
            inter: None,
        }
    }
}

/// Full-syntax driver state — frame geometry plus the
/// [`PartitionWalker`] neighbour-grid mirror every §8.3.2 context is
/// derived from. One driver per tile, mirroring the decode side's
/// one-walker-per-tile shape.
#[derive(Debug)]
pub struct PartitionSyntaxWriter {
    mi_rows: u32,
    mi_cols: u32,
    geometry: TileGeometry,
    mirror: PartitionWalker,
    /// §5.11.2 / §5.11.7 `ReadDeltas` lifecycle bit, write-side twin
    /// of the decode walker's — only the FIRST block of a superblock
    /// codes the §5.11.12 / §5.11.13 delta syntax (§5.11.7 line 11
    /// `ReadDeltas = 0`). Re-armed via [`Self::arm_read_deltas`] at
    /// every superblock entry.
    write_deltas_pending: bool,
}

impl PartitionSyntaxWriter {
    /// Construct a driver for a frame of `mi_rows × mi_cols` mi units
    /// scoped to `geometry`. `None` on a zero/inverted/oversize
    /// geometry (the [`PartitionWalker::new`] guards).
    #[must_use]
    pub fn new(mi_rows: u32, mi_cols: u32, geometry: TileGeometry) -> Option<Self> {
        if geometry.mi_row_end > mi_rows || geometry.mi_col_end > mi_cols {
            return None;
        }
        let mirror = PartitionWalker::new(mi_rows, mi_cols, geometry)?;
        Some(Self {
            mi_rows,
            mi_cols,
            geometry,
            mirror,
            write_deltas_pending: true,
        })
    }

    /// §5.11.2 `ReadDeltas = delta_q_present` — re-arm the write-side
    /// delta lifecycle at a superblock entry (the §5.11.7 line-11
    /// clear happens inside [`write_block_syntax`] after the first
    /// block's delta pair is coded).
    pub fn arm_read_deltas(&mut self) {
        self.write_deltas_pending = true;
    }

    /// The encoder-side neighbour-grid mirror (read-only view, e.g.
    /// for asserting stamp parity in tests).
    #[must_use]
    pub fn mirror(&self) -> &PartitionWalker {
        &self.mirror
    }

    /// §8.3.2 `partition` ctx from the mirror's `MiSizes[]` grid —
    /// the same derivation
    /// [`crate::cdf::PartitionWalker::decode_partition_syntax`]
    /// performs on the decode side.
    fn partition_ctx_for(&self, r: u32, c: u32, bsl: u32) -> usize {
        let sizes = self.mirror.mi_sizes();
        let avail_u = self.geometry.is_inside(r as i32 - 1, c as i32);
        let avail_l = self.geometry.is_inside(r as i32, c as i32 - 1);
        let above = avail_u && {
            let nb = sizes[((r - 1) * self.mi_cols + c) as usize];
            nb < BLOCK_SIZES && (MI_WIDTH_LOG2[nb] as u32) < bsl
        };
        let left = avail_l && {
            let nb = sizes[(r * self.mi_cols + c - 1) as usize];
            nb < BLOCK_SIZES && (MI_HEIGHT_LOG2[nb] as u32) < bsl
        };
        partition_ctx(above, left)
    }

    /// §5.11.9 neighbour cascade + §8.3.2 `segment_id` ctx from the
    /// mirror's `SegmentIds[]` grid — the write-side twin of
    /// [`crate::cdf::PartitionWalker::decode_segment_id`]'s
    /// derivation (out-of-grid / unvisited cells carry the `-1`
    /// sentinel).
    fn segment_pred_ctx(&self, mi_row: u32, mi_col: u32) -> (u8, usize) {
        let avail_u = self.geometry.is_inside(mi_row as i32 - 1, mi_col as i32);
        let avail_l = self.geometry.is_inside(mi_row as i32, mi_col as i32 - 1);
        let at = |r: i32, c: i32| -> i32 {
            if r < 0 || c < 0 {
                return -1;
            }
            let (r, c) = (r as u32, c as u32);
            if r >= self.mi_rows || c >= self.mi_cols {
                return -1;
            }
            self.mirror.segment_ids()[(r * self.mi_cols + c) as usize]
        };
        let prev_ul = if avail_u && avail_l {
            at(mi_row as i32 - 1, mi_col as i32 - 1)
        } else {
            -1
        };
        let prev_u = if avail_u {
            at(mi_row as i32 - 1, mi_col as i32)
        } else {
            -1
        };
        let prev_l = if avail_l {
            at(mi_row as i32, mi_col as i32 - 1)
        } else {
            -1
        };
        // §5.11.9 `pred` cascade — transcribed branch-for-branch (two
        // arms intentionally return `prev_u`; see the decode twin).
        #[allow(clippy::if_same_then_else)]
        let pred: i32 = if prev_u == -1 {
            if prev_l == -1 {
                0
            } else {
                prev_l
            }
        } else if prev_l == -1 {
            prev_u
        } else if prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        let to_opt = |v: i32| if v < 0 { None } else { Some(v) };
        let ctx = segment_id_ctx(to_opt(prev_ul), to_opt(prev_u), to_opt(prev_l));
        (pred as u8, ctx)
    }
}

/// Recursive §5.11.4 full-syntax driver — the write twin of
/// [`crate::cdf::PartitionWalker::decode_partition_syntax`]. Emits the
/// partition tree rooted at `(r, c, b_size)` with the COMPLETE
/// §5.11.7 `intra_frame_mode_info()` body (+ §5.11.49
/// `palette_tokens()`) at every leaf, via [`write_block_syntax`].
///
/// Same dispatch contract as [`write_partition_tree`]
/// (PARTITION_NONE leaves / PARTITION_SPLIT quadrants / forced arms /
/// out-of-frame short-circuit); same
/// [`Error::PartitionWalkOutOfRange`] caller-bug surface plus the
/// [`write_block_syntax`] scope rejects.
#[allow(clippy::too_many_arguments)]
pub fn write_partition_tree_syntax(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    node: &SyntaxNode,
    r: u32,
    c: u32,
    b_size: usize,
    params: &SyntaxFrameParams,
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

    let partition = match node {
        SyntaxNode::Leaf(_) => {
            if b_size >= BLOCK_8X8 && !has_rows && !has_cols {
                return Err(Error::PartitionWalkOutOfRange);
            }
            PARTITION_NONE
        }
        SyntaxNode::Split(_) => {
            if b_size < BLOCK_8X8 {
                return Err(Error::PartitionWalkOutOfRange);
            }
            PARTITION_SPLIT
        }
        SyntaxNode::Horz(_) => {
            // r412 scope: the general S() arm only (fully in frame).
            if b_size < BLOCK_8X8 || !has_rows || !has_cols {
                return Err(Error::PartitionWalkOutOfRange);
            }
            PARTITION_HORZ
        }
        SyntaxNode::Vert(_) => {
            if b_size < BLOCK_8X8 || !has_rows || !has_cols {
                return Err(Error::PartitionWalkOutOfRange);
            }
            PARTITION_VERT
        }
    };

    let pctx = if b_size >= BLOCK_8X8 {
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        state.partition_ctx_for(r, c, bsl)
    } else {
        0
    };
    write_partition(writer, cdfs, partition, b_size, has_rows, has_cols, pctx)?;

    let sub_size = partition_subsize(partition, b_size).ok_or(Error::PartitionWalkOutOfRange)?;

    match (node, partition) {
        (SyntaxNode::Leaf(block), PARTITION_NONE) => {
            write_block_syntax(writer, cdfs, state, block, r, c, sub_size, params)?;
        }
        (SyntaxNode::Horz(blocks), PARTITION_HORZ) => {
            // §5.11.4: `decode_block( r, c, subSize )` then
            // `decode_block( r + halfBlock4x4, c, subSize )` (the
            // `hasRows` gate held by the scope check above).
            let [top, bottom] = blocks;
            write_block_syntax(writer, cdfs, state, top, r, c, sub_size, params)?;
            write_block_syntax(
                writer,
                cdfs,
                state,
                bottom,
                r + half_block4x4,
                c,
                sub_size,
                params,
            )?;
        }
        (SyntaxNode::Vert(blocks), PARTITION_VERT) => {
            // §5.11.4: `decode_block( r, c, subSize )` then
            // `decode_block( r, c + halfBlock4x4, subSize )`.
            let [left, right] = blocks;
            write_block_syntax(writer, cdfs, state, left, r, c, sub_size, params)?;
            write_block_syntax(
                writer,
                cdfs,
                state,
                right,
                r,
                c + half_block4x4,
                sub_size,
                params,
            )?;
        }
        (SyntaxNode::Split(children), PARTITION_SPLIT) => {
            let [nw, ne, sw, se] = children;
            write_partition_tree_syntax(writer, cdfs, state, nw, r, c, sub_size, params)?;
            write_partition_tree_syntax(
                writer,
                cdfs,
                state,
                ne,
                r,
                c + half_block4x4,
                sub_size,
                params,
            )?;
            write_partition_tree_syntax(
                writer,
                cdfs,
                state,
                sw,
                r + half_block4x4,
                c,
                sub_size,
                params,
            )?;
            write_partition_tree_syntax(
                writer,
                cdfs,
                state,
                se,
                r + half_block4x4,
                c + half_block4x4,
                sub_size,
                params,
            )?;
        }
        _ => return Err(Error::PartitionWalkOutOfRange),
    }
    Ok(())
}

/// Full §5.11.5 / §5.11.7 leaf emission — the write twin of
/// [`crate::cdf::PartitionWalker::decode_block_syntax`]'s intra-frame
/// arm. Writes, in spec order:
///
/// 1. §5.11.7 lines 1-10 (the prefix): `intra_segment_id()` on the
///    `SegIdPreSkip` arm, `read_skip()`, `intra_segment_id()` on the
///    post-skip arm, `read_cdef()`, `read_delta_qindex()`,
///    `read_delta_lf()` — every §8.3.2 ctx (`skip`, the §5.11.9
///    cascade) derived from the driver's mirror grids.
/// 2. The §5.11.7 `use_intrabc` dispatch via
///    [`write_intra_frame_intrabc_arm`] — on the `Some` arm the
///    §7.10.2 `find_mv_stack( 0 )` runs against the mirror (identity
///    global motion + all-invalid `MotionFieldMvs`, per the §7.10.2.1
///    `ref == INTRA_FRAME ⇒ mv = 0` identity) and the §5.11.26
///    `assign_mv( 0 )` `PredMv` chain + §5.11.31 `read_mv( 0 )` write
///    fire under `MvCtx = MV_INTRABC_CONTEXT` / `force_integer_mv = 1`.
/// 3. The §5.11.7 `else` arm via [`write_intra_frame_else_arm`] —
///    `intra_frame_y_mode` (neighbour-mode ctx pair from the mirror's
///    `YModes[]`), `intra_angle_info_y()`, the `HasChroma` arm
///    (`uv_mode` / §5.11.45 `read_cfl_alphas()` /
///    `intra_angle_info_uv()`), §5.11.46 `palette_mode_info()` with
///    entries (the `has_palette_y` ctx + §5.11.49 cache from the
///    mirror's `PaletteSizes[]` / `PaletteColors[]`), and §5.11.24
///    `filter_intra_mode_info()`.
/// 4. §5.11.49 `palette_tokens()` via [`write_palette_tokens_plane`]
///    (per plane with `PaletteSize{Y,UV} > 0`).
/// 5. The mirror grid stamps
///    ([`PartitionWalker::stamp_encoder_block_syntax`]) so subsequent
///    leaves derive their contexts from the same state the decode
///    walker observes — including the §5.11.16 else-arm `TxSizes[]` /
///    `InterTxSizes[]` fill (bit-silent on the supported
///    `TxMode != TX_MODE_SELECT` configuration).
/// 6. §5.11.34 `residual()` via `write_residual_intra` (r283) —
///    on `skip == 0` the per-TU §5.11.35 / §5.11.39 coefficient write
///    composition (with the §5.11.47 / §5.11.40 tx-type mirror +
///    §5.11.38 per-plane residual sizing) runs in decode-walker
///    dispatch order, consuming [`SyntaxBlock::residual_quant`]; on
///    `skip == 1` only the decode walker's bit-silent
///    `TxTypes[] = DCT_DCT` pre-stamp is mirrored.
///
/// ## r283 scope rejects (all [`Error::PartitionWalkOutOfRange`])
///
/// * `skip == 0` on the intra-block-copy arm — the §5.11.36
///   inter-luma `transform_tree` write recursion is the follow-up arc.
/// * `params.tx_mode_select == true` — §5.11.16 `tx_depth` write
///   threading is the follow-up arc.
/// * [`SyntaxBlock::residual_quant`] count/length mismatches against
///   the §5.11.34 visited-TU enumeration.
/// * The leaf-writer caller-bug surface of the composed writers
///   (shape mismatches between `uv_mode` and `HasChroma`, CFL alphas,
///   palette bounds, non-integer intrabc MV difference, …).
#[allow(clippy::too_many_arguments)]
pub fn write_block_syntax(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    block: &SyntaxBlock,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    params: &SyntaxFrameParams,
) -> Result<(), Error> {
    // §5.11.5 / §5.5.2 caller-bug + scope guards (no bits before any
    // reject below this group fires).
    if sub_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if mi_row >= state.mi_rows || mi_col >= state.mi_cols {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if !matches!(params.bit_depth, 8 | 10 | 12) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if block.skip > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if block.skip == 1 && !block.residual_quant.is_empty() {
        // §5.11.35 `!skip` never fires on a skip leaf — supplying TU
        // coefficient arrays there is a caller bug (they would be
        // silently dropped otherwise).
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (block.segment_id as usize) >= MAX_SEGMENTS {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.6 `mode_info()` dispatch (r411): an inter-frame walk
    // (`FrameIsIntra == 0`) routes through the §5.11.18 arm.
    if params.inter.is_some() {
        return write_block_syntax_inter_frame(
            writer, cdfs, state, block, mi_row, mi_col, sub_size, params,
        );
    }
    // Inter commitments on an intra-frame walk are a caller bug.
    if block.inter.is_some() {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.5 prologue — `bw4` / `bh4` / `HasChroma` / `AvailU` /
    // `AvailL` (the chroma-availability fix-ups feed only the
    // §5.11.33 prediction pass, not any syntax write).
    let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size] as u32;
    let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size] as u32;
    let chroma_y_edge_case = bh4 == 1 && params.subsampling_y != 0 && (mi_row & 1) == 0;
    let chroma_x_edge_case = bw4 == 1 && params.subsampling_x != 0 && (mi_col & 1) == 0;
    let has_chroma = if chroma_y_edge_case || chroma_x_edge_case {
        false
    } else {
        params.num_planes > 1
    };
    let avail_u = state.geometry.is_inside(mi_row as i32 - 1, mi_col as i32);
    let avail_l = state.geometry.is_inside(mi_row as i32, mi_col as i32 - 1);

    // §5.11.7 lines 2-3: `if ( SegIdPreSkip ) intra_segment_id( )` —
    // called with the spec's `skip = 0` initialiser.
    let (seg_pred, seg_ctx) = state.segment_pred_ctx(mi_row, mi_col);
    if params.seg_id_pre_skip {
        write_intra_segment_id(
            writer,
            cdfs,
            block.segment_id,
            /* skip = */ 0,
            seg_pred,
            seg_ctx,
            params.segmentation_enabled,
            params.last_active_seg_id,
        )?;
    }

    // §5.11.7 line 5: `read_skip( )` — §8.3.2 ctx from the mirror's
    // `Skips[]` neighbours.
    let above_skip = if avail_u {
        state.mirror.skips()[((mi_row - 1) * state.mi_cols + mi_col) as usize]
    } else {
        0
    };
    let left_skip = if avail_l {
        state.mirror.skips()[(mi_row * state.mi_cols + mi_col - 1) as usize]
    } else {
        0
    };
    write_skip(
        writer,
        cdfs,
        block.skip,
        skip_ctx(above_skip, left_skip),
        params.seg_skip_active,
    )?;

    // §5.11.7 lines 6-7: `if ( !SegIdPreSkip ) intra_segment_id( )` —
    // post-skip arm sees the committed `skip`.
    if !params.seg_id_pre_skip {
        write_intra_segment_id(
            writer,
            cdfs,
            block.segment_id,
            block.skip,
            seg_pred,
            seg_ctx,
            params.segmentation_enabled,
            params.last_active_seg_id,
        )?;
    }

    // §5.11.7 line 8: `read_cdef( )` — anchor state from the mirror's
    // `cdef_idx[]` grid. (Bit-silent while `skip == 1`, but composed
    // for §5.11.56 ordering fidelity.)
    let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64] as u32;
    let cdef_mask: u32 = !(cdef_size4 - 1);
    let anchor_cell = ((mi_row & cdef_mask) * state.mi_cols + (mi_col & cdef_mask)) as usize;
    let anchor_prior = state.mirror.cdef_idx()[anchor_cell];
    let anchor_already_stamped = anchor_prior != -1;
    let cdef_value = if anchor_already_stamped {
        anchor_prior
    } else {
        block.cdef_idx
    };
    let cdef_committed = write_cdef(
        writer,
        cdef_value,
        anchor_prior,
        params.cdef_bits,
        block.skip,
        params.coded_lossless,
        params.enable_cdef,
        params.allow_intrabc,
        anchor_already_stamped,
    )?;

    // §5.11.7 line 9: `read_delta_qindex( )` — gated by the §5.11.2
    // per-superblock `ReadDeltas` lifecycle bit (only the FIRST block
    // of a superblock codes deltas; §5.11.7 line 11 clears the bit).
    let block_write_deltas = params.read_deltas && state.write_deltas_pending;
    write_delta_qindex(
        writer,
        cdfs,
        sub_size,
        block.reduced_delta_q_index,
        block.skip,
        block_write_deltas,
        params.use_128x128_superblock,
        params.delta_q_res,
    )?;

    // §5.11.7 line 10: `read_delta_lf( )`.
    write_delta_lf(
        writer,
        cdfs,
        sub_size,
        &block.reduced_delta_lf,
        block.skip,
        block_write_deltas,
        params.delta_lf_present,
        params.delta_lf_multi,
        params.mono_chrome,
        params.use_128x128_superblock,
        params.delta_lf_res,
    )?;
    // §5.11.7 line 11: `ReadDeltas = 0`.
    state.write_deltas_pending = false;

    // §6.8.2 per-segment `Lossless` + the §5.11.15 spec-forced
    // default `TxSize = Lossless ? TX_4X4 : Max_Tx_Size_Rect[ MiSize ]`.
    // This pre-fills the footer stamp's `TxSizes[]` / `InterTxSizes[]`
    // grids; the §5.11.16 write driver (r285) re-stamps both grids
    // with the committed value at the spec's `read_block_tx_size( )`
    // position before any grid read can observe the pre-fill.
    let lossless = params.lossless_array[block.segment_id as usize];
    let tx_size_blk = if lossless {
        TX_4X4
    } else {
        MAX_TX_SIZE_RECT[sub_size]
    };

    // §5.11.7 `use_intrabc` dispatch + `use_intrabc == 1` arm.
    if let Some(mv) = block.intrabc_mv {
        // §7.10.2 `find_mv_stack( 0 )` with `RefFrame = [ INTRA_FRAME,
        // NONE ]` against the mirror grids — identity global motion
        // (§7.10.2.1: `ref == INTRA_FRAME ⇒ mv = 0`) and an
        // all-invalid `MotionFieldMvs` (temporal projection is gated
        // off on intra frames), matching the decode walker's intrabc
        // arm setup verbatim.
        let mut gm_params = [[0i32; 6]; 8];
        for row in gm_params.iter_mut() {
            row[2] = 1 << WARPEDMODEL_PREC_BITS;
            row[5] = 1 << WARPEDMODEL_PREC_BITS;
        }
        let mfmv = MotionFieldMvs::new_invalid(state.mi_rows, state.mi_cols);
        let stack = state.mirror.find_mv_stack(
            mi_row,
            mi_col,
            sub_size,
            /* ref_frame = */ [0, -1],
            /* is_compound = */ false,
            /* use_ref_frame_mvs = */ false,
            [GM_TYPE_IDENTITY; 8],
            gm_params,
            [0; 8],
            /* allow_high_precision_mv = */ false,
            /* force_integer_mv = */ true,
            &mfmv,
        )?;
        let inputs = IntrabcArmInputs {
            mv,
            mv_stack: &stack,
            use_128x128_superblock: params.use_128x128_superblock,
            mi_row,
            mi_row_start: state.geometry.mi_row_start,
        };
        let info =
            write_intra_frame_intrabc_arm(writer, cdfs, params.allow_intrabc, Some(&inputs))?
                .ok_or(Error::PartitionWalkOutOfRange)?;

        // §5.11.5 footer stamps for the intra-block-copy block.
        state
            .mirror
            .stamp_encoder_block_syntax(&EncoderBlockSyntaxStamp {
                mi_row,
                mi_col,
                sub_size,
                skip: block.skip,
                skip_mode: 0,
                segment_id: block.segment_id,
                is_inter: 1,
                y_mode: info.y_mode,
                ref_frame: [0, -1],
                mv: info.mv,
                mv2: [0, 0],
                interp_filter: info.interp_filter,
                motion_mode: crate::cdf::MOTION_MODE_SIMPLE,
                palette_size_y: 0,
                palette_colors_y: &[],
                palette_size_uv: 0,
                palette_colors_u: &[],
                palette_colors_v: &[],
                cdef: cdef_committed,
                tx_size: tx_size_blk as u8,
            });
        // §5.11.49 `palette_tokens( )` — `PaletteSize{Y,UV} = 0` on
        // the intrabc arm ⇒ no-op.

        // §5.11.5 line `read_block_tx_size( )` — §5.11.16 write
        // driver (r285). The intrabc arm has `is_inter = 1`: with
        // `skip == 0 && TxMode == TX_MODE_SELECT && MiSize >
        // BLOCK_4X4 && !Lossless` the §5.11.17 `txfm_split` var-tx
        // recursion fires ([`SyntaxBlock::var_tx_trees`]); otherwise
        // the §5.11.15 else arm is bit-silent (`allowSelect = !skip
        // || !is_inter` is false on a `skip == 1` inter leaf).
        let tx_size_committed = write_block_tx_size_syntax(
            writer, cdfs, state, block, mi_row, mi_col, sub_size, lossless,
            /* is_inter = */ true, params,
        )?;

        if block.skip == 0 {
            // §5.11.34 `residual()` — the intra-block-copy arm has
            // `is_inter = 1`, so the luma plane routes through the
            // §5.11.36 `transform_tree` recursion (r284, var-tx-aware
            // since r285: the mirror's `InterTxSizes[]` now carries
            // the per-leaf §5.11.17 stamps) and the §5.11.39 writers
            // take the `is_inter` CDF axes. Chroma sizing follows the
            // §5.11.16-committed `TxSize` (the recursion's last
            // terminal-else `txSz`), matching the decode walker's
            // `residual( tx_size = read_block_tx_size( ) )`.
            write_residual(
                writer,
                cdfs,
                state,
                block,
                mi_row,
                mi_col,
                sub_size,
                params,
                has_chroma,
                lossless,
                tx_size_committed,
                /* is_inter = */ true,
            )?;
        } else {
            // §5.11.34 `residual()` skip-arm mirror (r283): the decode
            // walker still walks the per-TU §5.11.36 / §5.11.34
            // dispatch bit-silently on `skip == 1` and pre-stamps
            // `TxTypes[] = DCT_DCT` over every luma TU footprint (the
            // §5.11.39 gate-closed invariant the §5.11.40 chroma
            // lookup relies on). The TU footprints tile the block's
            // luma extent, so a single clipped block-footprint stamp
            // is observably identical.
            state
                .mirror
                .stamp_tx_type(mi_col, mi_row, bw4, bh4, DCT_DCT as u8);
            // §5.11.5 `if ( skip ) reset_block_context( bw4, bh4 )` —
            // the decode walker resets on the common path, intrabc
            // arm included (r284).
            state.mirror.reset_txb_block_context(
                mi_row,
                mi_col,
                bw4,
                bh4,
                has_chroma,
                params.subsampling_x,
                params.subsampling_y,
            );
        }
        return Ok(());
    }

    // §5.11.7 `use_intrabc = 0` element (S() when `allow_intrabc`,
    // silent otherwise), then the `else` arm composite.
    write_intra_frame_intrabc_arm(writer, cdfs, params.allow_intrabc, None)?;

    // §8.3.2 `intra_frame_y_mode` neighbour-mode ctx pair from the
    // mirror's `YModes[]` grid (`DC_PRED` fallback when unavailable).
    let above_mode = if avail_u {
        state.mirror.y_modes()[((mi_row - 1) * state.mi_cols + mi_col) as usize] as usize
    } else {
        DC_PRED
    };
    let left_mode = if avail_l {
        state.mirror.y_modes()[(mi_row * state.mi_cols + mi_col - 1) as usize] as usize
    } else {
        DC_PRED
    };
    let abovemode_ctx = intra_mode_ctx(above_mode);
    let leftmode_ctx = intra_mode_ctx(left_mode);

    // §8.3.2 `has_palette_y` neighbour ctx from the mirror's
    // `PaletteSizes[ 0 ]` grid (mirrors `decode_block_syntax`).
    let above_palette_y = avail_u
        && mi_row > 0
        && state.mirror.palette_sizes()[((mi_row - 1) * state.mi_cols + mi_col) as usize] > 0;
    let left_palette_y = avail_l
        && mi_col > 0
        && state.mirror.palette_sizes()[(mi_row * state.mi_cols + mi_col - 1) as usize] > 0;

    let cfl_allowed = cfl_allowed_for_uv_mode(
        lossless,
        sub_size,
        params.subsampling_x != 0,
        params.subsampling_y != 0,
    );
    let pal = &block.palette;
    let has_palette_y = u8::from(pal.size_y > 0);
    let has_palette_uv = u8::from(pal.size_uv > 0);

    write_intra_frame_else_arm(
        writer,
        cdfs,
        sub_size,
        block.y_mode,
        block.uv_mode,
        block.angle_delta_y,
        block.angle_delta_uv,
        cfl_allowed,
        has_chroma,
        params.allow_screen_content_tools,
        params.enable_filter_intra,
        block.use_filter_intra,
        block.filter_intra_mode,
        above_palette_y,
        left_palette_y,
        params.bit_depth,
        has_palette_y,
        pal.size_y as usize,
        &pal.colors_y,
        has_palette_uv,
        pal.size_uv as usize,
        &pal.colors_u,
        &pal.colors_v,
        pal.delta_encode_v,
        block.cfl_alpha_u,
        block.cfl_alpha_v,
        &state.mirror,
        mi_row,
        mi_col,
        abovemode_ctx,
        leftmode_ctx,
    )?;

    // §5.11.5 line `palette_tokens( )` — §5.11.49 per-plane writes,
    // gated by `PaletteSize{Y,UV} > 0` exactly like the reader.
    if pal.size_y > 0 {
        let args = palette_tokens_args(
            sub_size,
            mi_row as usize,
            mi_col as usize,
            state.mi_rows as usize,
            state.mi_cols as usize,
            PalettePlane::Y,
            0,
            0,
        )
        .ok_or(Error::PartitionWalkOutOfRange)?;
        write_palette_tokens_plane(
            writer,
            cdfs,
            PalettePlane::Y,
            pal.size_y as usize,
            args.block_w,
            args.block_h,
            args.onscreen_w,
            args.onscreen_h,
            &pal.color_map_y,
            args.block_w,
        )?;
    }
    if pal.size_uv > 0 {
        let args = palette_tokens_args(
            sub_size,
            mi_row as usize,
            mi_col as usize,
            state.mi_rows as usize,
            state.mi_cols as usize,
            PalettePlane::Uv,
            params.subsampling_x as usize,
            params.subsampling_y as usize,
        )
        .ok_or(Error::PartitionWalkOutOfRange)?;
        write_palette_tokens_plane(
            writer,
            cdfs,
            PalettePlane::Uv,
            pal.size_uv as usize,
            args.block_w,
            args.block_h,
            args.onscreen_w,
            args.onscreen_h,
            &pal.color_map_uv,
            args.block_w,
        )?;
    }

    // §5.11.5 footer stamps for the `else`-arm block.
    state
        .mirror
        .stamp_encoder_block_syntax(&EncoderBlockSyntaxStamp {
            mi_row,
            mi_col,
            sub_size,
            skip: block.skip,
            skip_mode: 0,
            segment_id: block.segment_id,
            is_inter: 0,
            y_mode: block.y_mode,
            ref_frame: [0, -1],
            mv: [0, 0],
            mv2: [0, 0],
            interp_filter: [0, 0],
            motion_mode: crate::cdf::MOTION_MODE_SIMPLE,
            palette_size_y: pal.size_y,
            palette_colors_y: &pal.colors_y,
            palette_size_uv: pal.size_uv,
            palette_colors_u: &pal.colors_u,
            palette_colors_v: &pal.colors_v,
            cdef: cdef_committed,
            tx_size: tx_size_blk as u8,
        });

    // §5.11.5 line `read_block_tx_size( )` — §5.11.16 write driver
    // (r285). The intra arm always takes the §5.11.15 else arm
    // (`is_inter = 0` ⇒ `allowSelect = 1`): one `tx_depth` S() when
    // `TxMode == TX_MODE_SELECT && MiSize > BLOCK_4X4 && !Lossless`
    // (committed via [`SyntaxBlock::tx_size`]), bit-silent
    // otherwise.
    let tx_size_committed = write_block_tx_size_syntax(
        writer, cdfs, state, block, mi_row, mi_col, sub_size, lossless,
        /* is_inter = */ false, params,
    )?;

    // §5.11.5 line `residual( )` — §5.11.34 write dispatcher (r283).
    // Follows the §5.11.16 `read_block_tx_size( )` +
    // §5.11.33 `compute_prediction( )` call sites in §5.11.5 order. On
    // `skip == 1` the decode walker's per-TU walk reads no bits and
    // only pre-stamps `TxTypes[] = DCT_DCT` over the luma TU
    // footprints (which tile the block's luma extent) — mirror with
    // one clipped block-footprint stamp. On `skip == 0` the full
    // per-TU §5.11.39 coefficient write composition runs against the
    // §5.11.16-committed `TxSize` (r285).
    if block.skip == 0 {
        write_residual(
            writer,
            cdfs,
            state,
            block,
            mi_row,
            mi_col,
            sub_size,
            params,
            has_chroma,
            lossless,
            tx_size_committed,
            /* is_inter = */ false,
        )?;
    } else {
        state
            .mirror
            .stamp_tx_type(mi_col, mi_row, bw4, bh4, DCT_DCT as u8);
        // §5.11.5 line `if ( skip ) reset_block_context( bw4, bh4 )`
        // — §5.11.42 zeroes the §6.10.2 `Above{Level,Dc}Context` /
        // `Left{Level,Dc}Context` arrays over the footprint; the
        // decode walker performs the same reset, so the mirror must
        // track it for the §8.3.2 `all_zero` / `dc_sign` ctx walks of
        // subsequent leaves to stay in lockstep.
        state.mirror.reset_txb_block_context(
            mi_row,
            mi_col,
            bw4,
            bh4,
            has_chroma,
            params.subsampling_x,
            params.subsampling_y,
        );
    }
    Ok(())
}

/// §5.11.18 `inter_frame_mode_info()` full-syntax leaf emission (r411)
/// — the write twin of the decode walker's
/// [`crate::cdf::PartitionWalker::decode_block_syntax`] inter arm.
/// Writes, in spec order:
///
/// 1. The §5.11.18 prologue via [`write_inter_frame_mode_info_prefix`]
///    (`inter_segment_id(1)`, `read_skip_mode()`, `read_skip()`,
///    `inter_segment_id(0)`, `read_cdef()`, `read_delta_qindex()`,
///    `read_delta_lf()`, `read_is_inter()`) — every §8.3.2 ctx
///    (`skip`, `is_inter`, the §5.11.9 cascade) derived from the
///    driver's mirror grids.
/// 2. On the `is_inter == 1` arm ([`SyntaxBlock::inter`]` == Some`):
///    the §5.11.23 body via [`write_inter_block_mode_info`] — the
///    §5.11.25 reference cascade, §7.10.2 `find_mv_stack` against the
///    mirror, the §5.11.24 single-pred mode cascade, the `drl_mode`
///    loop, the §5.11.31 `assign_mv` MV write on NEWMV, and the
///    (bit-silent on the r411 frame configuration) §5.11.27-§5.11.x
///    tail.
/// 3. On the `is_inter == 0` arm: the §5.11.22
///    `intra_block_mode_info()` composite via
///    [`write_intra_block_mode_info_with_palette`] (the `y_mode` S()
///    against `TileYModeCdf[ Size_Group[ MiSize ] ]`) plus §5.11.49
///    `palette_tokens()`.
/// 4. The §5.11.5 footer stamps, the §5.11.16 `read_block_tx_size()`
///    write driver, and the §5.11.34 `residual()` dispatch (inter luma
///    routes through the §5.11.36 `transform_tree` recursion) — or the
///    `skip == 1` `TxTypes[] = DCT_DCT` pre-stamp + §5.11.42 reset.
///
/// ## Scope rejects (all [`Error::PartitionWalkOutOfRange`])
///
/// * (r413) spatial segmentation (`segmentation_update_map == 1`,
///   `segmentation_temporal_update == 0`) is SUPPORTED — the
///   §5.11.19/§5.11.20 spatial arms ride the mirror's `SegmentIds[]`
///   cascade; the temporal-update arm (`seg_id_predicted` S() over
///   `PrevSegmentIds`) remains a follow-up reject.
/// * (r413) `ip.skip_mode_present` is SUPPORTED: the §5.11.10
///   `skip_mode` S() rides the §8.3.2 neighbour ctx from the mirror's
///   `SkipModes[]` grid, and a `skip_mode == 1` leaf must commit the
///   §5.11.18/§5.11.23 pure-derivation shape (see
///   [`SyntaxInterBlock::skip_mode`]).
/// * A committed MV that differs from the §5.11.26 `PredMv`
///   derivation on any non-NEWMV list (see [`SyntaxInterBlock`];
///   r412 lands compound pairs — both lists are §5.11.26-checked
///   through the shared `assign_mv_pred_mv`).
/// * A committed `interp_filter` pair inconsistent with the §5.11.x
///   loop's gates (r412 lands the SWITCHABLE frame-filter arm: the
///   per-block filter S() with the §8.3.2 neighbour ctx from the
///   mirror's `InterpFilters[]` / `RefFrames[]` grids).
#[allow(clippy::too_many_arguments)]
fn write_block_syntax_inter_frame(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    block: &SyntaxBlock,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    params: &SyntaxFrameParams,
) -> Result<(), Error> {
    let ip = params
        .inter
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    // §5.9.20: `allow_intrabc` is reachable only on intra frames.
    if block.intrabc_mv.is_some() || params.allow_intrabc {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // r413 scope reject: the §5.11.19 temporal-update arm
    // (`seg_id_predicted` S() + `PrevSegmentIds`) is a follow-up arc;
    // spatial segmentation (`segmentation_update_map == 1`,
    // `segmentation_temporal_update == 0`) is fully threaded.
    if params.segmentation_enabled
        && (ip.segmentation_temporal_update || !ip.segmentation_update_map)
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // r413 — §5.11.10 skip-mode commitments. `skip_mode == 1` is a
    // pure-derivation block: every §5.11.18/§5.11.23 value the reader
    // infers must be pre-committed identically by the caller.
    let skip_mode: u8 = block.inter.as_ref().map_or(0, |ib| ib.skip_mode);
    if skip_mode > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if skip_mode == 1 {
        let ib = block.inter.as_ref().expect("skip_mode implies inter");
        // §5.11.10 gates: frame-level presence + both dims >= 8.
        if !ip.skip_mode_present
            || block_width(sub_size) < 8
            || block_height(sub_size) < 8
            // §5.11.18 line 13: `if ( skip_mode ) skip = 1`.
            || block.skip != 1
            // §5.11.25 arm 1: `RefFrame[] = SkipModeFrame[]`.
            || ib.ref_frame != ip.skip_mode_frame
            // §5.11.23 arm 1: `YMode = NEAREST_NEARESTMV`, `RefMvIdx = 0`.
            || ib.y_mode != MODE_NEAREST_NEARESTMV
            || ib.ref_mv_idx != 0
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    // §5.11.5 prologue (mirrors the intra arm).
    let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size] as u32;
    let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size] as u32;
    let chroma_y_edge_case = bh4 == 1 && params.subsampling_y != 0 && (mi_row & 1) == 0;
    let chroma_x_edge_case = bw4 == 1 && params.subsampling_x != 0 && (mi_col & 1) == 0;
    let has_chroma = if chroma_y_edge_case || chroma_x_edge_case {
        false
    } else {
        params.num_planes > 1
    };
    let avail_u = state.geometry.is_inside(mi_row as i32 - 1, mi_col as i32);
    let avail_l = state.geometry.is_inside(mi_row as i32, mi_col as i32 - 1);

    // §5.11.19 neighbour cascade + §8.3.2 segment ctx (bit-silent with
    // segmentation off — derived for surface completeness).
    let (seg_pred, seg_ctx) = state.segment_pred_ctx(mi_row, mi_col);

    // §8.3.2 `skip` ctx from the mirror's `Skips[]` neighbours.
    let above_skip = if avail_u {
        state.mirror.skips()[((mi_row - 1) * state.mi_cols + mi_col) as usize]
    } else {
        0
    };
    let left_skip = if avail_l {
        state.mirror.skips()[(mi_row * state.mi_cols + mi_col - 1) as usize]
    } else {
        0
    };
    let skip_ctx_v = skip_ctx(above_skip, left_skip);

    // r413 — §8.3.2 `skip_mode` ctx from the mirror's `SkipModes[]`
    // neighbours (sum of the two flags, §5.11.10).
    let above_skip_mode = if avail_u {
        state.mirror.skip_modes()[((mi_row - 1) * state.mi_cols + mi_col) as usize]
    } else {
        0
    };
    let left_skip_mode = if avail_l {
        state.mirror.skip_modes()[(mi_row * state.mi_cols + mi_col - 1) as usize]
    } else {
        0
    };
    let skip_mode_ctx_v = skip_mode_ctx(above_skip_mode, left_skip_mode);

    // §8.3.2 `is_inter` ctx from the mirror's `IsInters[]` neighbours
    // (an unavailable neighbour reads as intra per §5.11.18).
    let above_intra_opt: Option<bool> = if avail_u {
        Some(state.mirror.is_inters()[((mi_row - 1) * state.mi_cols + mi_col) as usize] == 0)
    } else {
        None
    };
    let left_intra_opt: Option<bool> = if avail_l {
        Some(state.mirror.is_inters()[(mi_row * state.mi_cols + mi_col - 1) as usize] == 0)
    } else {
        None
    };
    let is_inter_ctx_v = is_inter_ctx(above_intra_opt, left_intra_opt);

    // §5.11.56 anchor state from the mirror's `cdef_idx[]` grid.
    let cdef_size4 = NUM_4X4_BLOCKS_WIDE[BLOCK_64X64] as u32;
    let cdef_mask: u32 = !(cdef_size4 - 1);
    let anchor_cell = ((mi_row & cdef_mask) * state.mi_cols + (mi_col & cdef_mask)) as usize;
    let anchor_prior = state.mirror.cdef_idx()[anchor_cell];
    let anchor_already_stamped = anchor_prior != -1;
    let cdef_value = if anchor_already_stamped {
        anchor_prior
    } else {
        block.cdef_idx
    };

    // §5.11.18 lines 11-22 via the r253-r258 prefix composition.
    let block_write_deltas = params.read_deltas && state.write_deltas_pending;
    let delta_inputs = InterFrameDeltaSiteInputs {
        cdef_idx: cdef_value,
        cdef_idx_prior_stamp: anchor_prior,
        cdef_bits: params.cdef_bits,
        coded_lossless: params.coded_lossless,
        enable_cdef: params.enable_cdef,
        allow_intrabc: params.allow_intrabc,
        anchor_already_stamped,
        reduced_delta_q_index: block.reduced_delta_q_index,
        reduced_delta_lf: block.reduced_delta_lf,
        read_deltas: block_write_deltas,
        use_128x128_superblock: params.use_128x128_superblock,
        delta_q_res: params.delta_q_res,
        delta_lf_present: params.delta_lf_present,
        delta_lf_multi: params.delta_lf_multi,
        mono_chrome: params.mono_chrome,
        delta_lf_res: params.delta_lf_res,
    };
    let is_inter_flag = u8::from(block.inter.is_some());
    let prefix = write_inter_frame_mode_info_prefix(
        writer,
        cdfs,
        sub_size,
        block.segment_id,
        skip_mode,
        block.skip,
        is_inter_flag,
        seg_pred,
        /* seg_id_predicted = */ 0,
        skip_mode_ctx_v,
        skip_ctx_v,
        /* seg_id_read_ctx = */ seg_ctx,
        /* seg_pred_ctx = */ 0,
        is_inter_ctx_v,
        /* seg_skip_mode_off = */ false,
        /* seg_skip_active = */ params.seg_skip_active,
        /* seg_ref_frame_active = */ false,
        /* seg_ref_frame_is_inter = */ false,
        /* seg_globalmv_active = */ false,
        params.segmentation_enabled,
        ip.segmentation_update_map,
        ip.segmentation_temporal_update,
        params.seg_id_pre_skip,
        /* predicted_segment_id = */ 0,
        params.last_active_seg_id,
        ip.skip_mode_present,
        &params.lossless_array,
        &delta_inputs,
    )?;
    // §5.11.18 line 21: `ReadDeltas = 0`.
    state.write_deltas_pending = false;
    let cdef_committed = prefix.cdef_idx;
    let lossless = prefix.lossless;
    let tx_pre = if lossless {
        TX_4X4
    } else {
        MAX_TX_SIZE_RECT[sub_size]
    };

    if let Some(ib) = block.inter.as_ref() {
        // ---- §5.11.18 `if ( is_inter ) inter_block_mode_info()`. ----
        // r412 scope: single reference OR a compound pair (the
        // §5.11.25 cascade validates codable pairs); COMPOUND_AVERAGE
        // only (`enable_masked_compound == enable_jnt_comp == false`
        // keeps the §5.11.29 tail bit-silent).
        let is_compound = ib.ref_frame[1] > 0;
        if !(1..=7).contains(&ib.ref_frame[0])
            || !(ib.ref_frame[1] == -1 || (1..=7).contains(&ib.ref_frame[1]))
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let mode_ok = if is_compound {
            (MODE_NEAREST_NEARESTMV..=MODE_NEW_NEWMV).contains(&ib.y_mode)
        } else {
            matches!(
                ib.y_mode,
                MODE_NEARESTMV | MODE_NEARMV | MODE_GLOBALMV | MODE_NEWMV
            )
        };
        if !mode_ok {
            return Err(Error::PartitionWalkOutOfRange);
        }
        // §6.10.27 conformance bound `Abs( Mv ) < (1 << 14)` on every
        // consulted list.
        let list_count: usize = if is_compound { 2 } else { 1 };
        for i in 0..list_count {
            if ib.mv[i][0].unsigned_abs() >= (1 << 14) || ib.mv[i][1].unsigned_abs() >= (1 << 14) {
                return Err(Error::PartitionWalkOutOfRange);
            }
        }
        if ip.interpolation_filter > SWITCHABLE {
            return Err(Error::PartitionWalkOutOfRange);
        }

        // §5.11.18 lines 2-9 — neighbour ref-frame derivations from
        // the mirror's `RefFrames[]` grid.
        let rf = |r: u32, c: u32, slot: usize| -> i32 {
            i32::from(state.mirror.ref_frames()[((r * state.mi_cols + c) as usize) * 2 + slot])
        };
        let left_ref_frame: [i32; 2] = if avail_l {
            [rf(mi_row, mi_col - 1, 0), rf(mi_row, mi_col - 1, 1)]
        } else {
            [0, -1]
        };
        let above_ref_frame: [i32; 2] = if avail_u {
            [rf(mi_row - 1, mi_col, 0), rf(mi_row - 1, mi_col, 1)]
        } else {
            [0, -1]
        };
        let left_intra = left_ref_frame[0] <= 0;
        let above_intra = above_ref_frame[0] <= 0;
        let left_single = left_ref_frame[1] <= 0;
        let above_single = above_ref_frame[1] <= 0;

        // §7.10.2 `find_mv_stack( 0 )` against the mirror grids —
        // the identical scan the decode walker runs at this leaf.
        let stack = state.mirror.find_mv_stack(
            mi_row,
            mi_col,
            sub_size,
            [i32::from(ib.ref_frame[0]), i32::from(ib.ref_frame[1])],
            is_compound,
            ip.use_ref_frame_mvs,
            ip.gm_type,
            ip.gm_params,
            ip.ref_frame_sign_bias,
            ip.allow_high_precision_mv,
            ip.force_integer_mv,
            &ip.motion_field_mvs,
        )?;

        // §5.11.26: non-NEWMV lists inherit `PredMv[ i ]` with no
        // bits — every committed MV must equal the reader's
        // derivation (the shared `assign_mv_pred_mv`, per list via
        // §get_mode).
        for i in 0..list_count {
            if crate::cdf::get_mode(ib.y_mode, i) != MODE_NEWMV {
                let d = assign_mv_pred_mv(&stack, ib.y_mode, i as u8, ib.ref_mv_idx)?;
                if d != ib.mv[i] {
                    return Err(Error::PartitionWalkOutOfRange);
                }
            }
        }

        // §5.11.23 tail — every optional tool gated shut on the r411
        // frame configuration (the writers still validate the spec
        // preconditions); r412 adds the SWITCHABLE frame-filter arm:
        // the §5.11.x loop codes `interp_filter[ 0 ]` (slot 1 mirrors
        // it on `!enable_dual_filter`) whenever `needs_interp_filter()`
        // holds, else derives EIGHTTAP with no bits.
        let mut tail = InterBlockModeInfoTail::bit_silent();
        tail.interpolation_filter = ip.interpolation_filter;
        tail.enable_dual_filter = ip.enable_dual_filter;
        tail.gm_type = ip.gm_type;
        tail.is_motion_mode_switchable = ip.is_motion_mode_switchable;
        tail.allow_warped_motion = ip.allow_warped_motion;
        tail.is_scaled_per_ref = ip.is_scaled_per_ref;
        tail.enable_interintra_compound = ip.enable_interintra_compound;
        tail.enable_masked_compound = ip.enable_masked_compound;
        tail.enable_jnt_comp = ip.enable_jnt_comp;
        tail.interp_filter = if ip.interpolation_filter != SWITCHABLE {
            // §5.11.x else-arm: both slots read the frame filter with
            // no bits — the committed pair must carry exactly that.
            if ib.interp_filter != [ip.interpolation_filter; 2] {
                return Err(Error::PartitionWalkOutOfRange);
            }
            InterpolationFilterReadout {
                interp_filter: [ip.interpolation_filter; 2],
                read_from_bitstream: [false, false],
            }
        } else {
            // `needs_interp_filter( )` per av1-spec p.75 on this arm's
            // fixed scalars (SIMPLE motion mode —
            // the r411 configuration): only the GLOBALMV-on-large
            // branch can gate the read off.
            let large = core::cmp::min(
                NUM_4X4_BLOCKS_WIDE[sub_size] * MI_SIZE,
                NUM_4X4_BLOCKS_HIGH[sub_size] * MI_SIZE,
            ) >= 8;
            let needs = if skip_mode != 0 {
                // §5.11.23 `needs_interp_filter( )`: `skip_mode` (and
                // LOCALWARP, unreachable here) return 0 — the reader
                // derives EIGHTTAP on both slots with no bits.
                false
            } else if large && ib.y_mode == MODE_GLOBALMV {
                ip.gm_type[ib.ref_frame[0] as usize] == GM_TYPE_TRANSLATION
            } else if large && ib.y_mode == MODE_GLOBAL_GLOBALMV {
                ip.gm_type[ib.ref_frame[0] as usize] == GM_TYPE_TRANSLATION
                    || ip.gm_type[ib.ref_frame[1] as usize] == GM_TYPE_TRANSLATION
            } else {
                true
            };
            let mut read = [false, false];
            if needs {
                read[0] = true;
                if ip.enable_dual_filter {
                    read[1] = true;
                }
            }
            InterpolationFilterReadout {
                interp_filter: ib.interp_filter,
                read_from_bitstream: read,
            }
        };
        if avail_u {
            let cell = (((mi_row - 1) * state.mi_cols + mi_col) as usize) * 2;
            let f = state.mirror.interp_filters();
            tail.above_interp_filters = [f[cell], f[cell + 1]];
        }
        if avail_l {
            let cell = ((mi_row * state.mi_cols + mi_col - 1) as usize) * 2;
            let f = state.mirror.interp_filters();
            tail.left_interp_filters = [f[cell], f[cell + 1]];
        }
        let committed_filters = tail.interp_filter.interp_filter;

        write_inter_block_mode_info(
            writer,
            cdfs,
            [i32::from(ib.ref_frame[0]), i32::from(ib.ref_frame[1])],
            ib.y_mode,
            ib.mv,
            ib.ref_mv_idx,
            &stack,
            sub_size,
            skip_mode,
            ip.skip_mode_frame,
            /* seg_ref_frame_active = */ false,
            /* seg_ref_frame_data = */ 0,
            /* seg_skip_active = */ false,
            /* seg_globalmv_active = */ false,
            ip.reference_select,
            avail_u,
            avail_l,
            above_single,
            left_single,
            above_intra,
            left_intra,
            above_ref_frame,
            left_ref_frame,
            ip.force_integer_mv,
            ip.allow_high_precision_mv,
            &tail,
        )?;

        // §5.11.5 footer stamps for the inter leaf.
        state
            .mirror
            .stamp_encoder_block_syntax(&EncoderBlockSyntaxStamp {
                mi_row,
                mi_col,
                sub_size,
                skip: block.skip,
                skip_mode,
                segment_id: block.segment_id,
                is_inter: 1,
                y_mode: ib.y_mode,
                ref_frame: ib.ref_frame,
                mv: ib.mv[0],
                mv2: ib.mv[1],
                interp_filter: committed_filters,
                motion_mode: MOTION_MODE_SIMPLE,
                palette_size_y: 0,
                palette_colors_y: &[],
                palette_size_uv: 0,
                palette_colors_u: &[],
                palette_colors_v: &[],
                cdef: cdef_committed,
                tx_size: tx_pre as u8,
            });

        // §5.11.16 + §5.11.34 (mirrors the intrabc arm: inter luma
        // routes through the §5.11.36 `transform_tree` recursion).
        let tx_size_committed = write_block_tx_size_syntax(
            writer, cdfs, state, block, mi_row, mi_col, sub_size, lossless,
            /* is_inter = */ true, params,
        )?;
        if block.skip == 0 {
            write_residual(
                writer,
                cdfs,
                state,
                block,
                mi_row,
                mi_col,
                sub_size,
                params,
                has_chroma,
                lossless,
                tx_size_committed,
                /* is_inter = */ true,
            )?;
        } else {
            state
                .mirror
                .stamp_tx_type(mi_col, mi_row, bw4, bh4, DCT_DCT as u8);
            state.mirror.reset_txb_block_context(
                mi_row,
                mi_col,
                bw4,
                bh4,
                has_chroma,
                params.subsampling_x,
                params.subsampling_y,
            );
        }
        return Ok(());
    }

    // ---- §5.11.18 `else intra_block_mode_info()` — §5.11.22. ----
    // §8.3.2 `has_palette_y` neighbour ctx from the mirror's
    // `PaletteSizes[ 0 ]` grid (mirrors the decode walker's
    // intra-in-inter tail).
    let above_palette_y = avail_u
        && mi_row > 0
        && state.mirror.palette_sizes()[((mi_row - 1) * state.mi_cols + mi_col) as usize] > 0;
    let left_palette_y = avail_l
        && mi_col > 0
        && state.mirror.palette_sizes()[(mi_row * state.mi_cols + mi_col - 1) as usize] > 0;
    let cfl_allowed = cfl_allowed_for_uv_mode(
        lossless,
        sub_size,
        params.subsampling_x != 0,
        params.subsampling_y != 0,
    );
    let pal = &block.palette;
    let has_palette_y = u8::from(pal.size_y > 0);
    let has_palette_uv = u8::from(pal.size_uv > 0);
    write_intra_block_mode_info_with_palette(
        writer,
        cdfs,
        sub_size,
        block.y_mode,
        block.uv_mode,
        block.angle_delta_y,
        block.angle_delta_uv,
        cfl_allowed,
        has_chroma,
        params.allow_screen_content_tools,
        params.enable_filter_intra,
        block.use_filter_intra,
        block.filter_intra_mode,
        above_palette_y,
        left_palette_y,
        params.bit_depth,
        has_palette_y,
        pal.size_y as usize,
        &pal.colors_y,
        has_palette_uv,
        pal.size_uv as usize,
        &pal.colors_u,
        &pal.colors_v,
        pal.delta_encode_v,
        block.cfl_alpha_u,
        block.cfl_alpha_v,
        &state.mirror,
        mi_row,
        mi_col,
    )?;

    // §5.11.5 line `palette_tokens( )` — §5.11.49 per-plane writes.
    if pal.size_y > 0 {
        let args = palette_tokens_args(
            sub_size,
            mi_row as usize,
            mi_col as usize,
            state.mi_rows as usize,
            state.mi_cols as usize,
            PalettePlane::Y,
            0,
            0,
        )
        .ok_or(Error::PartitionWalkOutOfRange)?;
        write_palette_tokens_plane(
            writer,
            cdfs,
            PalettePlane::Y,
            pal.size_y as usize,
            args.block_w,
            args.block_h,
            args.onscreen_w,
            args.onscreen_h,
            &pal.color_map_y,
            args.block_w,
        )?;
    }
    if pal.size_uv > 0 {
        let args = palette_tokens_args(
            sub_size,
            mi_row as usize,
            mi_col as usize,
            state.mi_rows as usize,
            state.mi_cols as usize,
            PalettePlane::Uv,
            params.subsampling_x as usize,
            params.subsampling_y as usize,
        )
        .ok_or(Error::PartitionWalkOutOfRange)?;
        write_palette_tokens_plane(
            writer,
            cdfs,
            PalettePlane::Uv,
            pal.size_uv as usize,
            args.block_w,
            args.block_h,
            args.onscreen_w,
            args.onscreen_h,
            &pal.color_map_uv,
            args.block_w,
        )?;
    }

    // §5.11.5 footer stamps — `RefFrames[ .. ] = [ INTRA_FRAME, NONE ]`
    // / `Mvs[ .. ] = 0` collapse to the mirror's pre-fill (the decode
    // walker relies on the same collapse); `IsInters[ .. ] = 0` rides
    // the stamp.
    state
        .mirror
        .stamp_encoder_block_syntax(&EncoderBlockSyntaxStamp {
            mi_row,
            mi_col,
            sub_size,
            skip: block.skip,
            skip_mode: 0,
            segment_id: block.segment_id,
            is_inter: 0,
            y_mode: block.y_mode,
            ref_frame: [0, -1],
            mv: [0, 0],
            mv2: [0, 0],
            interp_filter: [0, 0],
            motion_mode: MOTION_MODE_SIMPLE,
            palette_size_y: pal.size_y,
            palette_colors_y: &pal.colors_y,
            palette_size_uv: pal.size_uv,
            palette_colors_u: &pal.colors_u,
            palette_colors_v: &pal.colors_v,
            cdef: cdef_committed,
            tx_size: tx_pre as u8,
        });

    // §5.11.16 — intra ⇒ the §5.11.15 else arm.
    let tx_size_committed = write_block_tx_size_syntax(
        writer, cdfs, state, block, mi_row, mi_col, sub_size, lossless,
        /* is_inter = */ false, params,
    )?;

    // §5.11.34 `residual()` — intra dispatch.
    if block.skip == 0 {
        write_residual(
            writer,
            cdfs,
            state,
            block,
            mi_row,
            mi_col,
            sub_size,
            params,
            has_chroma,
            lossless,
            tx_size_committed,
            /* is_inter = */ false,
        )?;
    } else {
        state
            .mirror
            .stamp_tx_type(mi_col, mi_row, bw4, bh4, DCT_DCT as u8);
        state.mirror.reset_txb_block_context(
            mi_row,
            mi_col,
            bw4,
            bh4,
            has_chroma,
            params.subsampling_x,
            params.subsampling_y,
        );
    }
    Ok(())
}

/// §5.11.16 `read_block_tx_size( )` write driver (r285) — the write
/// twin of [`PartitionWalker::read_block_tx_size`], invoked from
/// [`write_block_syntax`] at the §5.11.5 `read_block_tx_size( )`
/// position (after `palette_tokens( )`, before `residual( )`).
///
/// Mirrors the §5.11.16 dispatch line-for-line:
///
/// * **Var-tx arm** (`TxMode == TX_MODE_SELECT && MiSize > BLOCK_4X4
///   && is_inter && !skip && !Lossless`) — walks the `(txH4, txW4)`
///   sub-rectangles of the block footprint in spec loop order,
///   consuming one [`VarTxSyntaxTree`] from
///   [`SyntaxBlock::var_tx_trees`] per position and emitting its
///   §5.11.17 recursion via [`write_var_tx_size_syntax`]; then the
///   §5.11.5 outer `TxSizes[]` footprint fill with the last
///   terminal-else `txSz` (the same
///   [`PartitionWalker::stamp_tx_sizes_footprint`] the reader runs).
///   [`SyntaxBlock::tx_size`] must be `None` (the block's `TxSize`
///   is the recursion's last terminal-else `txSz`).
///
/// * **`else` arm** — `read_tx_size( !skip || !is_inter )` per
///   §5.11.15. The committed `TxSize` is [`SyntaxBlock::tx_size`]
///   (or the spec-forced default when `None`); the §8.3.2 `tx_depth`
///   ctx comes from the mirror's
///   [`PartitionWalker::tx_depth_block_ctx`] (the identical
///   derivation the reader runs), the symbol from the stateless
///   [`write_block_tx_size`] (which also validates the spec-forced
///   values on every bit-silent sub-arm), and the grid fill from
///   [`PartitionWalker::stamp_block_tx_size_grids`].
///   [`SyntaxBlock::var_tx_trees`] must be empty.
///
/// Returns the committed `TxSize` ordinal — the value the decode
/// walker's `read_block_tx_size` returns and threads into §5.11.34
/// `residual( )`.
#[allow(clippy::too_many_arguments)]
fn write_block_tx_size_syntax(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    block: &SyntaxBlock,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    lossless: bool,
    is_inter: bool,
    params: &SyntaxFrameParams,
) -> Result<usize, Error> {
    // §5.11.16 lines 1-2: `bw4` / `bh4`.
    let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size] as u32;
    let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size] as u32;
    let skip = block.skip != 0;

    // §5.11.16 outer dispatch — the var-tx arm.
    if params.tx_mode_select && sub_size > BLOCK_4X4 && is_inter && !skip && !lossless {
        if block.tx_size.is_some() {
            // The block's `TxSize` on this arm is the §5.11.17
            // recursion's last terminal-else `txSz` — a scalar
            // commitment alongside the trees is a caller bug.
            return Err(Error::PartitionWalkOutOfRange);
        }
        // §5.11.16 lines 5-7: `maxTxSz` / `txW4` / `txH4`.
        let max_tx_sz = MAX_TX_SIZE_RECT[sub_size];
        let tx_w4 = (TX_WIDTH[max_tx_sz] / MI_SIZE) as u32;
        let tx_h4 = (TX_HEIGHT[max_tx_sz] / MI_SIZE) as u32;
        // §5.11.16 inner loops — one caller-supplied tree per
        // `read_var_tx_size( row, col, maxTxSz, 0 )` position.
        let mut tree_idx = 0usize;
        let mut last_tx_size = max_tx_sz as u8;
        let mut row = mi_row;
        while row < mi_row + bh4 {
            let mut col = mi_col;
            while col < mi_col + bw4 {
                let tree = block
                    .var_tx_trees
                    .get(tree_idx)
                    .ok_or(Error::PartitionWalkOutOfRange)?;
                tree_idx += 1;
                last_tx_size = write_var_tx_size_syntax(
                    writer, cdfs, state, tree, mi_row, mi_col, sub_size, row, col, max_tx_sz,
                    /* depth = */ 0,
                )?;
                col += tx_w4;
            }
            row += tx_h4;
        }
        // A surplus tree is the same caller bug as a shortfall.
        if tree_idx != block.var_tx_trees.len() {
            return Err(Error::PartitionWalkOutOfRange);
        }
        // §5.11.5 outer `TxSizes[]` fill with the last terminal-else
        // `txSz` (same helper as the reader).
        state
            .mirror
            .stamp_tx_sizes_footprint(mi_row, mi_col, sub_size, last_tx_size);
        return Ok(last_tx_size as usize);
    }

    // §5.11.16 `else` arm: `read_tx_size( !skip || !is_inter )`.
    if !block.var_tx_trees.is_empty() {
        // Var-tx trees on a non-var-tx arm are a caller bug.
        return Err(Error::PartitionWalkOutOfRange);
    }
    let allow_select = !skip || !is_inter;
    // §5.11.15 committed `TxSize`: the caller's choice where the
    // `tx_depth` S() fires, the spec-forced default otherwise. The
    // stateless writer below validates both (a `Some(t)` differing
    // from the forced value on a bit-silent arm rejects there).
    let default_tx = if lossless {
        TX_4X4
    } else {
        MAX_TX_SIZE_RECT[sub_size]
    };
    let tx_size = block.tx_size.map_or(default_tx, |t| t as usize);
    // §8.3.2 `tx_depth` ctx from the mirror walker — the identical
    // derivation `read_block_tx_size` performs. Side-effect free, so
    // deriving it on the bit-silent arms is harmless.
    let ctx = state.mirror.tx_depth_block_ctx(mi_row, mi_col, sub_size);
    write_block_tx_size(
        writer,
        cdfs,
        tx_size,
        sub_size,
        lossless,
        allow_select,
        params.tx_mode_select,
        ctx,
    )?;
    // §5.11.16 `else`-arm grid fill (same helper as the reader).
    state
        .mirror
        .stamp_block_tx_size_grids(mi_row, mi_col, sub_size, tx_size as u8);
    Ok(tx_size)
}

/// §5.11.17 `read_var_tx_size( row, col, txSz, depth )` write driver
/// (r285) — the write twin of [`PartitionWalker::read_var_tx_size`],
/// driven by a caller-supplied [`VarTxSyntaxTree`] split-decision
/// tree instead of decoded `txfm_split` symbols. Every other line
/// mirrors the reader:
///
/// * the `row >= MiRows || col >= MiCols` early return (no symbol,
///   no stamp — the supplied node must be a [`VarTxSyntaxTree::Leaf`]);
/// * the `txSz == TX_4X4 || depth == MAX_VARTX_DEPTH` terminal
///   conditions (forced `txfm_split = 0`, no S() on either side; a
///   [`VarTxSyntaxTree::Split`] there is a caller bug);
/// * the §8.3.2 ctx via the mirror's
///   [`PartitionWalker::txfm_split_node_ctx`] — derived live in
///   §5.11.17 visit order, because earlier leaves' `InterTxSizes[]`
///   stamps feed later nodes' `get_above_tx_width` /
///   `get_left_tx_height` walks on BOTH sides;
/// * the terminal-else [`PartitionWalker::stamp_var_tx_leaf`] stamp;
/// * the `Split_Tx_Size` recursion in row-major `(i, j)` order with
///   the last terminal-else `txSz` propagated upward.
#[allow(clippy::too_many_arguments)]
fn write_var_tx_size_syntax(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    tree: &VarTxSyntaxTree,
    mi_row_b: u32,
    mi_col_b: u32,
    sub_size: usize,
    row: u32,
    col: u32,
    tx_sz: usize,
    depth: u32,
) -> Result<u8, Error> {
    // §5.11.17 lines 1-2: frame-edge early return — the reader emits
    // nothing and stamps nothing.
    if row >= state.mi_rows || col >= state.mi_cols {
        return match tree {
            VarTxSyntaxTree::Leaf => Ok(tx_sz as u8),
            // A Split here would describe symbols the reader never
            // reads — caller bug.
            VarTxSyntaxTree::Split(_) => Err(Error::PartitionWalkOutOfRange),
        };
    }
    // §5.11.17 lines 3-7: the two terminal conditions force
    // `txfm_split = 0` with no S() on either side.
    let terminal = tx_sz == TX_4X4 || depth == MAX_VARTX_DEPTH;
    match tree {
        VarTxSyntaxTree::Leaf => {
            if !terminal {
                // `txfm_split = 0` S() against the §8.3.2 ctx.
                let ctx = state
                    .mirror
                    .txfm_split_node_ctx(mi_row_b, mi_col_b, sub_size, row, col, tx_sz)?;
                let cdf = cdfs.txfm_split_cdf(ctx);
                writer.write_symbol(0, cdf)?;
            }
            // §5.11.17 terminal-else `InterTxSizes[]` stamp (same
            // helper as the reader).
            state.mirror.stamp_var_tx_leaf(row, col, tx_sz);
            Ok(tx_sz as u8)
        }
        VarTxSyntaxTree::Split(children) => {
            if terminal {
                return Err(Error::PartitionWalkOutOfRange);
            }
            // `txfm_split = 1` S().
            let ctx = state
                .mirror
                .txfm_split_node_ctx(mi_row_b, mi_col_b, sub_size, row, col, tx_sz)?;
            let cdf = cdfs.txfm_split_cdf(ctx);
            writer.write_symbol(1, cdf)?;
            // §5.11.17 lines 11-16: the `Split_Tx_Size` recursion.
            let sub_tx_sz = SPLIT_TX_SIZE[tx_sz];
            let w4 = (TX_WIDTH[tx_sz] / MI_SIZE) as u32;
            let h4 = (TX_HEIGHT[tx_sz] / MI_SIZE) as u32;
            let step_w = (TX_WIDTH[sub_tx_sz] / MI_SIZE) as u32;
            let step_h = (TX_HEIGHT[sub_tx_sz] / MI_SIZE) as u32;
            debug_assert!(
                step_w > 0 && step_h > 0,
                "Split_Tx_Size step must advance the §5.11.17 loop"
            );
            // One child per spec loop visit, row-major.
            let expected = ((h4 / step_h) * (w4 / step_w)) as usize;
            if children.len() != expected {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let mut last_tx_sz = sub_tx_sz as u8;
            let mut k = 0usize;
            let mut i = 0u32;
            while i < h4 {
                let mut j = 0u32;
                while j < w4 {
                    last_tx_sz = write_var_tx_size_syntax(
                        writer,
                        cdfs,
                        state,
                        &children[k],
                        mi_row_b,
                        mi_col_b,
                        sub_size,
                        row + i,
                        col + j,
                        sub_tx_sz,
                        depth + 1,
                    )?;
                    k += 1;
                    j += step_w;
                }
                i += step_h;
            }
            Ok(last_tx_sz)
        }
    }
}

/// §5.11.34 `residual()` write dispatcher — the write twin of
/// [`PartitionWalker::residual`]'s intra (`is_inter == 0`) arm,
/// invoked from [`write_block_syntax`] on a `skip == 0` else-arm leaf.
///
/// Mirrors the §5.11.34 body line-for-line on the reachable path:
///
/// ```text
///   widthChunks  = Max( 1, Block_Width[ MiSize ] >> 6 )
///   heightChunks = Max( 1, Block_Height[ MiSize ] >> 6 )
///   miSizeChunk  = ( widthChunks > 1 || heightChunks > 1 )
///                      ? BLOCK_64X64 : MiSize
///   for ( chunkY ... ) for ( chunkX ... )
///       for ( plane = 0; plane < 1 + HasChroma * 2; plane++ ) {
///           txSz    = Lossless ? TX_4X4 : get_tx_size( plane, TxSize )
///           stepX/Y = Tx_Width/Height[ txSz ] >> 2
///           planeSz = get_plane_residual_size( miSizeChunk, plane )
///           num4x4W/H, subX/Y, baseXBlock/baseYBlock
///           for ( y ... += stepY ) for ( x ... += stepX )
///               transform_block( plane, baseXBlock, baseYBlock, txSz,
///                                x + ((chunkX << 4) >> subX),
///                                y + ((chunkY << 4) >> subY) )
///       }
/// ```
///
/// On `is_inter == 1` (the §5.11.7 `use_intrabc == 1` arm) the
/// `is_inter && !Lossless && !plane` branch routes the luma plane
/// through the §5.11.36 `transform_tree` recursion
/// ([`write_transform_tree`], r284) instead of the direct `stepX` /
/// `stepY` iteration; chroma planes always take the direct loop.
///
/// Each visited TU consumes the next [`SyntaxBlock::residual_quant`]
/// entry; a count mismatch in either direction is a caller bug
/// ([`Error::PartitionWalkOutOfRange`]).
#[allow(clippy::too_many_arguments)]
fn write_residual(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    block: &SyntaxBlock,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    params: &SyntaxFrameParams,
    has_chroma: bool,
    lossless: bool,
    tx_size_blk: usize,
    is_inter: bool,
) -> Result<(), Error> {
    // Caller-bug guards (mirror `PartitionWalker::residual`).
    if params.subsampling_x > 1 || params.subsampling_y > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.34 lines 3-5: `widthChunks` / `heightChunks` /
    // `miSizeChunk`. `Block_Width = Num_4x4_Blocks_Wide * 4`.
    let block_w = NUM_4X4_BLOCKS_WIDE[sub_size] * 4;
    let block_h = NUM_4X4_BLOCKS_HIGH[sub_size] * 4;
    let width_chunks = core::cmp::max(1, block_w >> 6);
    let height_chunks = core::cmp::max(1, block_h >> 6);
    let mi_size_chunk = if width_chunks > 1 || height_chunks > 1 {
        BLOCK_64X64
    } else {
        sub_size
    };

    // §5.11.34: `1 + HasChroma * 2` planes.
    let num_planes: u8 = if has_chroma { 3 } else { 1 };

    let mut tu_idx = 0usize;
    // §5.11.47 per-luma-TU `TxType` commitment cursor (r286) — advanced
    // once per luma (`plane == 0`) `transform_block` the §5.11.34
    // dispatch visits, independent of the `tu_idx` `residual_quant`
    // cursor (which counts every plane's TUs).
    let mut luma_tx_idx = 0usize;
    for chunk_y in 0..height_chunks {
        for chunk_x in 0..width_chunks {
            for plane in 0..num_planes {
                // §5.11.34 line 12: `txSz`.
                let tx_sz = if lossless {
                    TX_4X4
                } else {
                    get_tx_size(
                        plane,
                        tx_size_blk,
                        sub_size,
                        params.subsampling_x,
                        params.subsampling_y,
                    )
                    .ok_or(Error::PartitionWalkOutOfRange)?
                };
                // §5.11.34 lines 13-14: `stepX` / `stepY` (≥ 1 — the
                // smallest §6.10.16 transform side is 4).
                let step_x = TX_WIDTH[tx_sz] >> 2;
                let step_y = TX_HEIGHT[tx_sz] >> 2;
                // §5.11.34 line 15: `planeSz` — a `None` for chroma is
                // the implicit "no chroma residual" path; caller-bug
                // for luma (same split as the decode walker).
                let plane_sz = match get_plane_residual_size(
                    mi_size_chunk,
                    plane,
                    params.subsampling_x,
                    params.subsampling_y,
                ) {
                    Some(s) => s,
                    None => {
                        if plane == 0 {
                            return Err(Error::PartitionWalkOutOfRange);
                        }
                        continue;
                    }
                };
                // §5.11.34 lines 16-19.
                let num4x4_w = NUM_4X4_BLOCKS_WIDE[plane_sz];
                let num4x4_h = NUM_4X4_BLOCKS_HIGH[plane_sz];
                let sub_x = if plane > 0 { params.subsampling_x } else { 0 };
                let sub_y = if plane > 0 { params.subsampling_y } else { 0 };
                // §5.11.34 lines 23-24: the inter-luma plane routes
                // through the §5.11.36 `transform_tree` recursion
                // anchored at the CHUNK-offset base (the write twin of
                // the decode walker's `residual_transform_tree`).
                if is_inter && !lossless && plane == 0 {
                    let mi_row_chunk = mi_row + ((chunk_y as u32) << 4);
                    let mi_col_chunk = mi_col + ((chunk_x as u32) << 4);
                    let base_x = (mi_col_chunk >> sub_x) * (MI_SIZE as u32);
                    let base_y = (mi_row_chunk >> sub_y) * (MI_SIZE as u32);
                    write_transform_tree(
                        writer,
                        cdfs,
                        state,
                        block,
                        params,
                        base_x,
                        base_y,
                        (num4x4_w * 4) as u32,
                        (num4x4_h * 4) as u32,
                        mi_row,
                        mi_col,
                        sub_size,
                        lossless,
                        &mut tu_idx,
                        &mut luma_tx_idx,
                    )?;
                    continue;
                }
                // §5.11.34 lines 26-27: `baseXBlock` / `baseYBlock`
                // (the original `MiCol` / `MiRow`, NOT the
                // chunk-offset ones).
                let base_x_block = (mi_col >> sub_x) * (MI_SIZE as u32);
                let base_y_block = (mi_row >> sub_y) * (MI_SIZE as u32);
                // §5.11.34 lines 28-30: per-TU iteration.
                let mut y = 0usize;
                while y < num4x4_h {
                    let mut x = 0usize;
                    while x < num4x4_w {
                        let x_arg = (x as u32) + (((chunk_x as u32) << 4) >> sub_x);
                        let y_arg = (y as u32) + (((chunk_y as u32) << 4) >> sub_y);
                        write_transform_block(
                            writer,
                            cdfs,
                            state,
                            block,
                            params,
                            plane,
                            base_x_block,
                            base_y_block,
                            tx_sz,
                            x_arg,
                            y_arg,
                            sub_x,
                            sub_y,
                            mi_row,
                            mi_col,
                            sub_size,
                            lossless,
                            is_inter,
                            &mut tu_idx,
                            &mut luma_tx_idx,
                        )?;
                        x += step_x;
                    }
                    y += step_y;
                }
            }
        }
    }
    // Every supplied TU array must have been consumed — a surplus is
    // the same caller bug as a shortfall (caught inside the per-TU
    // writer).
    if tu_idx != block.residual_quant.len() {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // r286 — the §5.11.47 per-luma-TU `TxType` commitment vector, when
    // supplied, must be EXACTLY the visited luma-TU count (a shortfall
    // silently defaults the tail to `DCT_DCT`; a surplus is a caller
    // bug). An empty vector is the explicit `all-DCT_DCT` opt-out and
    // is always allowed.
    if !block.residual_tx_type.is_empty() && luma_tx_idx != block.residual_tx_type.len() {
        return Err(Error::PartitionWalkOutOfRange);
    }
    Ok(())
}

/// §5.11.35 `transform_block()` write twin — the bit-emitting subset
/// of the decode walker's per-TU dispatch
/// (`PartitionWalker::transform_block_emit`) on the intra `!skip` arm:
///
/// 1. The §5.11.35 line-13 `startX >= maxX || startY >= maxY`
///    early return (clipped TUs consume no `residual_quant` entry —
///    the reader never visits them).
/// 2. The §5.11.47 `transform_type()` mirror on luma. The decode
///    walker invokes its reader against the neutral per-block
///    [`crate::cdf::ResidualContext`] (`base_q_idx = 0`, segmentation
///    off), so the §5.11.47 `set > 0 && qIdx > 0` guard is always
///    false there: no S() symbol on either side and
///    `TxType = DCT_DCT`, stamped over the TU's 4×4 footprint into
///    the mirror's `TxTypes[]`. (The richer caller-supplied-quantiser
///    variant lands with the §5.9.12 quantizer-params threading arc.)
/// 3. The §5.11.40 `compute_tx_type()` derivation (luma reads the
///    just-stamped `TxTypes[]`; intra chroma reads
///    `Mode_To_Txfm[ UVMode ]` filtered by the §5.11.48 set
///    admission) → §8.3.2 `get_tx_class` reduction → §7.5
///    [`crate::scan::get_scan`] selection.
/// 4. The §5.11.39 [`write_coefficients`] emission with the §8.3.2
///    `all_zero` / `dc_sign` contexts derived from the mirror
///    walker's §6.10.2 `Above{Level,Dc}Context` /
///    `Left{Level,Dc}Context` arrays (r284) — the same
///    [`PartitionWalker::txb_skip_ctx`] / [`PartitionWalker::dc_sign_ctx`]
///    derivations the decode walker performs — followed by the
///    §5.11.39 tail stamps of the TU's `culLevel` / `dcCategory`
///    into the mirror.
#[allow(clippy::too_many_arguments)]
fn write_transform_block(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    block: &SyntaxBlock,
    params: &SyntaxFrameParams,
    plane: u8,
    base_x: u32,
    base_y: u32,
    tx_sz: usize,
    x: u32,
    y: u32,
    sub_x: u8,
    sub_y: u8,
    mi_row: u32,
    mi_col: u32,
    mi_size: usize,
    lossless: bool,
    is_inter: bool,
    tu_idx: &mut usize,
    luma_tx_idx: &mut usize,
) -> Result<(), Error> {
    // §5.11.35 lines 1-2 + 10-13.
    let start_x = base_x + 4 * x;
    let start_y = base_y + 4 * y;
    let max_x = (state.mi_cols * (MI_SIZE as u32)) >> sub_x;
    let max_y = (state.mi_rows * (MI_SIZE as u32)) >> sub_y;
    if start_x >= max_x || start_y >= max_y {
        return Ok(());
    }

    let tx_w = TX_WIDTH[tx_sz];
    let tx_h = TX_HEIGHT[tx_sz];
    let x4 = start_x >> 2;
    let y4 = start_y >> 2;

    // §5.11.48 `get_tx_set( txSz )` — the per-set transform-type
    // alphabet. `reduced_tx_set` now flows from the frame params
    // (r286), matching the decode walker's `ctx.reduced_tx_set` exactly.
    let tx_sz_sqr = tx_size_sqr_index(tx_sz);
    let tx_set = if is_inter {
        inter_tx_type_set(
            tx_sz_sqr as u32,
            TX_SIZE_SQR_UP[tx_sz] as u32,
            params.reduced_tx_set,
        )
    } else {
        intra_tx_type_set(
            tx_sz_sqr as u32,
            TX_SIZE_SQR_UP[tx_sz] as u32,
            params.reduced_tx_set,
        )
    };

    // §5.11.39 `coeffs()` — consume the next caller-committed `Quant[]`
    // array up front: the §5.11.39 line-13 `all_zero` write must precede
    // the §5.11.47 `transform_type()` emission (r384 fix — the spec's
    // else-arm ordering), and `all_zero` is a property of the committed
    // coefficients.
    let quant = block
        .residual_quant
        .get(*tu_idx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    *tu_idx += 1;
    if quant.len() != tx_w * tx_h {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let all_zero = quant.iter().all(|&q| q == 0);
    // §8.3.2 `all_zero` / `dc_sign` ctx — the same neighbour-array
    // derivations the decode walker's `transform_block_emit` performs,
    // computed against the mirror's §6.10.2 context arrays (r284).
    let txb_skip_ctx = state
        .mirror
        .txb_skip_ctx(plane, tx_sz, mi_size, x4, y4, sub_x, sub_y)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let dc_sign_ctx = state.mirror.dc_sign_ctx(plane, tx_sz, x4, y4, sub_x, sub_y);
    // §5.11.39 line 13: `all_zero` S() — the FIRST coefficient symbol,
    // written ahead of the §5.11.47 `transform_type()` mirror below.
    {
        let tx_sz_sqr_min = {
            let side = core::cmp::min(tx_w, tx_h);
            (side.trailing_zeros() as usize) - 2
        };
        let tx_sz_ctx = (tx_sz_sqr_min + TX_SIZE_SQR_UP[tx_sz] + 1) >> 1;
        crate::encoder::coefficients::write_txb_skip(
            writer,
            cdfs,
            u8::from(all_zero),
            tx_sz_ctx,
            txb_skip_ctx,
        )?;
    }
    if all_zero {
        // §5.11.39 `if ( all_zero )` arm: no transform_type symbol, no
        // further coefficient writes. The luma plane stamps DCT_DCT
        // over the TU footprint; the commitment vector still advances
        // (one committed TxType per luma TU), and a non-DCT_DCT
        // commitment for an all-zero TU is a caller bug (the decode
        // walker reads DCT_DCT regardless).
        if plane == 0 {
            let committed = block
                .residual_tx_type
                .get(*luma_tx_idx)
                .copied()
                .unwrap_or(DCT_DCT as u8);
            *luma_tx_idx += 1;
            if committed != DCT_DCT as u8 {
                return Err(Error::PartitionWalkOutOfRange);
            }
            state.mirror.stamp_tx_type(
                x4,
                y4,
                (tx_w >> 2) as u32,
                (tx_h >> 2) as u32,
                DCT_DCT as u8,
            );
        }
        // §5.11.39 tail — culLevel = 0 / dcCategory = 0 stamps fire on
        // the gate-closed arm too.
        state
            .mirror
            .stamp_txb_level_context(plane, tx_sz, x4, y4, 0, 0);
        return Ok(());
    }

    // §5.11.47 `transform_type()` write side — luma only (r286). The
    // decode walker invokes its `transform_type()` reader on every luma
    // (`plane == 0`) `all_zero == 0` TU; the §5.11.47 `set > 0 &&
    // (segmentation_enabled ? get_qindex(1, segment_id) : base_q_idx) >
    // 0` guard decides whether an `intra_tx_type` / `inter_tx_type` S()
    // is coded. When closed the stamp is `DCT_DCT` (no symbol); when
    // open we emit the symbol for the caller-committed `TxType` and
    // stamp it. Both sides then stamp the TU's 4×4 footprint into the
    // `TxTypes[]` grid, so the §5.11.40 `compute_tx_type` derivation
    // below (and on the decode side) reads the same value.
    if plane == 0 {
        // §5.11.47 `(segmentation_enabled ? get_qindex(1, segment_id) :
        // base_q_idx) > 0` guard (`ignoreDeltaQ = 1` in the §7.12.2
        // lookup — delta-q is intentionally ignored here).
        let q_for_guard = if params.quant.segmentation_enabled {
            crate::cdf::get_qindex(&params.quant, true, block.segment_id)
        } else {
            params.quant.base_q_idx as i32
        };
        let codes_symbol = tx_set > 0 && q_for_guard > 0;

        // Caller-committed `TxType` for this luma TU (in §5.11.34
        // dispatch order). When the commitment vector is exhausted /
        // empty the value defaults to `DCT_DCT` (back-compat).
        let committed = block
            .residual_tx_type
            .get(*luma_tx_idx)
            .copied()
            .unwrap_or(DCT_DCT as u8);
        *luma_tx_idx += 1;

        let tx_type: u8 = if codes_symbol {
            // §8.3.2 `intra_dir` axis for the `intra_tx_type` CDF (the
            // §5.11.42 `use_filter_intra` → `Filter_Intra_Mode_To_Intra_Dir`
            // remap; unused on the inter path).
            let intra_dir = crate::cdf::intra_dir(
                block.use_filter_intra == 1,
                block.y_mode as usize,
                block.filter_intra_mode.unwrap_or(0) as usize,
            )
            .unwrap_or(0);
            // §5.11.47 forward symbol map — the inverse of the read
            // path's `Tx_Type_*_Inv_Set*[ symbol ]` lookup. A committed
            // `TxType` not admissible in `set` is a caller bug.
            let symbol = crate::cdf::tx_type_to_symbol(tx_set, is_inter, committed as usize)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            if is_inter {
                let cdf = cdfs
                    .inter_tx_type_cdf(tx_set, tx_sz_sqr as u32)
                    .ok_or(Error::PartitionWalkOutOfRange)?;
                writer.write_symbol(symbol, cdf)?;
            } else {
                let cdf = cdfs
                    .intra_tx_type_cdf(tx_set, tx_sz_sqr as u32, intra_dir)
                    .ok_or(Error::PartitionWalkOutOfRange)?;
                writer.write_symbol(symbol, cdf)?;
            }
            committed
        } else {
            // Guard closed: no S(), `TxType == DCT_DCT`. A non-DCT_DCT
            // commitment here is a caller bug (the decode walker would
            // read `DCT_DCT` regardless).
            if committed != DCT_DCT as u8 {
                return Err(Error::PartitionWalkOutOfRange);
            }
            DCT_DCT as u8
        };

        state
            .mirror
            .stamp_tx_type(x4, y4, (tx_w >> 2) as u32, (tx_h >> 2) as u32, tx_type);
    }

    // §5.11.40 `compute_tx_type( plane, txSz, x4, y4 )` against the
    // mirror's `TxTypes[]` grid — same closure shape as the decode
    // walker's.
    let plane_tx_type = {
        let tx_types_grid = state.mirror.tx_types();
        let mi_rows = state.mi_rows;
        let mi_cols = state.mi_cols;
        compute_tx_type(
            plane as usize,
            tx_sz,
            lossless,
            is_inter,
            tx_set,
            mi_row,
            mi_col,
            x4,
            y4,
            sub_x as u32,
            sub_y as u32,
            block.uv_mode.unwrap_or(0) as usize,
            |yy, xx| {
                if yy >= mi_rows || xx >= mi_cols {
                    DCT_DCT
                } else {
                    tx_types_grid[(yy * mi_cols + xx) as usize] as usize
                }
            },
        )
    };

    // §8.3.2 `get_tx_class` reduction.
    let tx_class = match plane_tx_type {
        V_DCT | V_ADST | V_FLIPADST => TX_CLASS_VERT,
        H_DCT | H_ADST | H_FLIPADST => TX_CLASS_HORIZ,
        _ => TX_CLASS_2D,
    };

    // §7.5 / §5.11.41 scan selection.
    let scan = crate::scan::get_scan(tx_sz, plane_tx_type);

    // §5.11.39 gate-open body — the `all_zero` symbol was already
    // written above, ahead of the §5.11.47 emission.
    let wout = crate::encoder::coefficients::write_coefficients_gate_open(
        writer,
        cdfs,
        plane,
        u8::from(is_inter),
        tx_sz,
        tx_class,
        dc_sign_ctx,
        scan,
        quant,
    )?;
    // §5.11.39 tail — stamp the TU's `culLevel` / `dcCategory` into
    // the mirror so the next TU's ctx walk sees what the decode
    // walker's will.
    state
        .mirror
        .stamp_txb_level_context(plane, tx_sz, x4, y4, wout.cul_level, wout.dc_category);
    Ok(())
}

/// §5.11.36 `transform_tree( startX, startY, w, h )` write twin — the
/// bit-emitting mirror of the decode walker's §5.11.36 recursion
/// (`residual_transform_tree`), invoked from [`write_residual`] on the
/// `is_inter && !Lossless && plane == 0` arm (reachable through the
/// §5.11.7 `use_intrabc == 1` + `skip == 0` leaf):
///
/// ```text
///   transform_tree( startX, startY, w, h ) {
///       maxX = MiCols * MI_SIZE ; maxY = MiRows * MI_SIZE
///       if ( startX >= maxX || startY >= maxY ) return
///       row = startY >> MI_SIZE_LOG2 ; col = startX >> MI_SIZE_LOG2
///       lumaTxSz = InterTxSizes[ row ][ col ]
///       lumaW = Tx_Width[ lumaTxSz ] ; lumaH = Tx_Height[ lumaTxSz ]
///       if ( w <= lumaW && h <= lumaH ) {
///           txSz = find_tx_size( w, h )
///           transform_block( 0, startX, startY, txSz, 0, 0 )
///       } else { ... per-direction halving recursion ... }
///   }
/// ```
///
/// The `InterTxSizes[]` lookup reads the encoder mirror's grid — the
/// §5.11.16 write driver ([`write_block_tx_size_syntax`], r285)
/// mirrors the decode walker's `read_block_tx_size` fills on both
/// arms (the else-arm uniform stamp AND the §5.11.17 per-leaf var-tx
/// stamps), so the recursion shape agrees on both sides by
/// construction — including genuinely variable trees.
#[allow(clippy::too_many_arguments)]
fn write_transform_tree(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut PartitionSyntaxWriter,
    block: &SyntaxBlock,
    params: &SyntaxFrameParams,
    start_x: u32,
    start_y: u32,
    w: u32,
    h: u32,
    mi_row: u32,
    mi_col: u32,
    mi_size: usize,
    lossless: bool,
    tu_idx: &mut usize,
    luma_tx_idx: &mut usize,
) -> Result<(), Error> {
    // §5.11.36 lines 1-3: frame-extent early return.
    let max_x = state.mi_cols * (MI_SIZE as u32);
    let max_y = state.mi_rows * (MI_SIZE as u32);
    if start_x >= max_x || start_y >= max_y {
        return Ok(());
    }
    // §5.11.36 lines 4-6: `InterTxSizes[ row ][ col ]` via the mirror
    // (in-grid by the early return above).
    let row = (start_y >> (MI_SIZE_LOG2 as u32)) as usize;
    let col = (start_x >> (MI_SIZE_LOG2 as u32)) as usize;
    let luma_tx_sz = state.mirror.inter_tx_sizes()[row * state.mi_cols as usize + col] as usize;
    if luma_tx_sz >= TX_SIZES_ALL {
        return Err(Error::ResidualTransformTreeUnsupported);
    }
    let luma_w = TX_WIDTH[luma_tx_sz] as u32;
    let luma_h = TX_HEIGHT[luma_tx_sz] as u32;

    // §5.11.36 lines 7-10: leaf emit.
    if w <= luma_w && h <= luma_h {
        let tx_sz =
            find_tx_size(w as usize, h as usize).ok_or(Error::ResidualTransformTreeUnsupported)?;
        return write_transform_block(
            writer,
            cdfs,
            state,
            block,
            params,
            /* plane = */ 0,
            /* base_x = */ start_x,
            /* base_y = */ start_y,
            tx_sz,
            /* x = */ 0,
            /* y = */ 0,
            /* sub_x = */ 0,
            /* sub_y = */ 0,
            mi_row,
            mi_col,
            mi_size,
            lossless,
            /* is_inter = */ true,
            tu_idx,
            luma_tx_idx,
        );
    }
    // §5.11.36 lines 12-25: per-direction halving recursion (the same
    // visit order as the decode twin).
    let halves: &[(u32, u32, u32, u32)] = if w > h {
        &[(0, 0, 1, 0), (1, 0, 1, 0)]
    } else if w < h {
        &[(0, 0, 0, 1), (0, 1, 0, 1)]
    } else {
        &[(0, 0, 1, 1), (1, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]
    };
    for &(ox, oy, hw, hh) in halves {
        let sub_w = if hw != 0 { w / 2 } else { w };
        let sub_h = if hh != 0 { h / 2 } else { h };
        write_transform_tree(
            writer,
            cdfs,
            state,
            block,
            params,
            start_x + ox * (w / 2),
            start_y + oy * (h / 2),
            sub_w,
            sub_h,
            mi_row,
            mi_col,
            mi_size,
            lossless,
            tu_idx,
            luma_tx_idx,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        get_br_ctx, get_coeff_base_ctx, get_coeff_base_eob_ctx, CoefficientsReadout,
        PartitionWalker, TileGeometry as G, BLOCK_16X16, BLOCK_16X32, BLOCK_32X16, BLOCK_32X32,
        BLOCK_64X64, BLOCK_8X8, DC_PRED, MAX_SEGMENTS, TX_16X16, TX_4X4, TX_8X8, TX_CLASS_2D,
        V_PRED,
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
            cfl_alpha_u: None,
            cfl_alpha_v: None,
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
            cfl_alpha_u: None,
            cfl_alpha_v: None,
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
            cfl_alpha_u: None,
            cfl_alpha_v: None,
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

    // -----------------------------------------------------------------
    // r282 — full §5.11.7 block-syntax threading round-trips. Every
    // test below encodes a partition tree through
    // `write_partition_tree_syntax`, appends an 8-bit sync sentinel,
    // and decodes the bytes through the §5.11.4
    // `PartitionWalker::decode_partition_syntax` walker with the SAME
    // frame params. The sentinel landing intact proves the encoder
    // emitted exactly the bits the §5.11.5 walker consumes; the
    // whole-`TileCdfContext` equality proves §8.3 adaptation lockstep
    // symbol-for-symbol; the mirror-vs-walker grid parity proves the
    // encoder's neighbour-context state matches the decoder's.
    // -----------------------------------------------------------------

    use crate::cdf::PALETTE_COLORS;
    use crate::encoder::partition_tree::{
        write_partition_tree_syntax, PartitionSyntaxWriter, SyntaxBlock, SyntaxFrameParams,
        SyntaxNode, SyntaxPalette,
    };

    /// Encode `node`, append `sentinel`, decode through
    /// `decode_partition_syntax`, assert the sentinel + full-CDF
    /// lockstep + grid parity, and return the pieces for per-test
    /// asserts.
    fn syntax_round_trip(
        node: &SyntaxNode,
        mi_rows: u32,
        mi_cols: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
        sentinel: u8,
    ) -> (PartitionSyntaxWriter, PartitionWalker) {
        // ----- encode -----
        let mut writer = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut enc_state =
            PartitionSyntaxWriter::new(mi_rows, mi_cols, single_tile(mi_rows, mi_cols))
                .expect("syntax writer construction");
        write_partition_tree_syntax(
            &mut writer,
            &mut enc_cdfs,
            &mut enc_state,
            node,
            0,
            0,
            b_size,
            params,
        )
        .expect("write_partition_tree_syntax");
        writer.write_literal(8, u32::from(sentinel)).unwrap();
        let bytes = writer.finish();

        // ----- decode (the §5.11.4 / §5.11.5 syntax walker) -----
        let mut walker =
            PartitionWalker::new(mi_rows, mi_cols, single_tile(mi_rows, mi_cols)).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        walker
            .decode_partition_syntax(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                b_size,
                /* frame_is_intra = */ true,
                params.subsampling_x,
                params.subsampling_y,
                params.num_planes,
                params.seg_id_pre_skip,
                params.segmentation_enabled,
                params.seg_skip_active,
                params.last_active_seg_id,
                &params.lossless_array,
                params.coded_lossless,
                params.enable_cdef,
                params.allow_intrabc,
                params.cdef_bits,
                params.read_deltas,
                params.use_128x128_superblock,
                params.delta_q_res,
                params.delta_lf_present,
                params.delta_lf_multi,
                params.mono_chrome,
                params.delta_lf_res,
                params.allow_screen_content_tools,
                params.enable_filter_intra,
                params.bit_depth,
                params.tx_mode_select,
                /* inter_ctx = */ None,
                &params.quant,
                params.reduced_tx_set,
            )
            .expect("decode_partition_syntax must consume the full tree");

        // Sync sentinel — the walker consumed exactly the emitted bits.
        assert_eq!(
            dec.read_literal(8).unwrap(),
            u32::from(sentinel),
            "decoder must be positioned at the sentinel after the tree"
        );
        // §8.3 adaptation lockstep over the ENTIRE working context.
        assert_eq!(
            enc_cdfs, dec_cdfs,
            "encoder and decoder CDF contexts must adapt identically"
        );
        // Mirror-vs-walker grid parity — the encoder derives the next
        // block's contexts from the same state the decoder observes.
        let m = enc_state.mirror();
        assert_eq!(m.mi_sizes(), walker.mi_sizes(), "MiSizes parity");
        assert_eq!(m.skips(), walker.skips(), "Skips parity");
        assert_eq!(m.segment_ids(), walker.segment_ids(), "SegmentIds parity");
        assert_eq!(m.y_modes(), walker.y_modes(), "YModes parity");
        assert_eq!(m.is_inters(), walker.is_inters(), "IsInters parity");
        assert_eq!(m.ref_frames(), walker.ref_frames(), "RefFrames parity");
        assert_eq!(m.mvs(), walker.mvs(), "Mvs parity");
        assert_eq!(
            m.interp_filters(),
            walker.interp_filters(),
            "InterpFilters parity"
        );
        assert_eq!(
            m.palette_sizes(),
            walker.palette_sizes(),
            "PaletteSizes parity"
        );
        assert_eq!(
            m.palette_colors(),
            walker.palette_colors(),
            "PaletteColors parity"
        );
        assert_eq!(m.cdef_idx(), walker.cdef_idx(), "cdef_idx parity");
        // r283 — §5.11.16 / §5.11.47 mirror grids: the residual write
        // threading keeps the encoder's TxSizes / InterTxSizes /
        // TxTypes stamps in decode-walker parity.
        assert_eq!(m.tx_sizes(), walker.tx_sizes(), "TxSizes parity");
        assert_eq!(
            m.inter_tx_sizes(),
            walker.inter_tx_sizes(),
            "InterTxSizes parity"
        );
        assert_eq!(m.tx_types(), walker.tx_types(), "TxTypes parity");
        // r284 — §6.10.2 / §8.3.2 coefficient-level context arrays:
        // the per-TU §5.11.39 tail stamps (and §5.11.42 skip resets)
        // must agree cell-for-cell, pinning write↔decode `all_zero` /
        // `dc_sign` ctx-derivation lockstep for every TU in the tree.
        assert_eq!(
            m.above_level_context(),
            walker.above_level_context(),
            "AboveLevelContext parity"
        );
        assert_eq!(
            m.above_dc_context(),
            walker.above_dc_context(),
            "AboveDcContext parity"
        );
        assert_eq!(
            m.left_level_context(),
            walker.left_level_context(),
            "LeftLevelContext parity"
        );
        assert_eq!(
            m.left_dc_context(),
            walker.left_dc_context(),
            "LeftDcContext parity"
        );
        (enc_state, walker)
    }

    /// Four §5.11.7 else-arm leaves under one SPLIT — the 2nd/3rd/4th
    /// leaves' `intra_frame_y_mode` ctx pairs come from the previously
    /// stamped `YModes[]` neighbours, so the sentinel landing proves
    /// the encoder's mirror threading, not just per-leaf writers.
    /// Covers directional modes + angle deltas + the §5.11.24
    /// filter-intra pair.
    #[test]
    fn r282_syntax_round_trip_split_else_arm_neighbour_ctx() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.enable_filter_intra = true;

        // NW: V_PRED with angle deltas on both planes.
        let mut nw = SyntaxBlock::skip_leaf(V_PRED as u8, Some(V_PRED as u8));
        nw.angle_delta_y = 2;
        nw.angle_delta_uv = -1;
        // NE: DC_PRED with the §5.11.24 filter-intra pair.
        let mut ne = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        ne.use_filter_intra = 1;
        ne.filter_intra_mode = Some(2);
        // SW: H_PRED (mode 2) with a negative luma angle delta.
        let mut sw = SyntaxBlock::skip_leaf(2, Some(DC_PRED as u8));
        sw.angle_delta_y = -2;
        // SE: plain DC.
        let se = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0xA5);

        // Decoded YModes[] footprints match the committed modes.
        let cols = 4usize;
        for (r0, c0, want) in [(0, 0, V_PRED as u8), (0, 2, 0u8), (2, 0, 2u8), (2, 2, 0u8)] {
            for dr in 0..2usize {
                for dc in 0..2usize {
                    assert_eq!(
                        walker.y_modes()[(r0 + dr) * cols + (c0 + dc)],
                        want,
                        "YModes stamp at ({}, {})",
                        r0 + dr,
                        c0 + dc
                    );
                }
            }
        }
        assert!(
            walker.skips().iter().all(|&s| s == 1),
            "every leaf committed skip = 1"
        );
    }

    /// Two intra-block-copy leaves in one SPLIT: the NE leaf's §5.11.26
    /// `PredMv[ 0 ]` comes from the NW leaf's `Mvs[]` stamps via the
    /// §7.10.2 `find_mv_stack` neighbour scan — on BOTH sides. A mirror
    /// divergence would change the coded MV difference and break the
    /// sentinel.
    #[test]
    fn r282_syntax_round_trip_intrabc_pair_threads_mv_stack() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.allow_intrabc = true;

        // NW at the origin: empty stack ⇒ §5.11.26 fallback predictor
        // [ 0, -(sbSize4 * MI_SIZE + INTRABC_DELAY_PIXELS) * 8 ] =
        // [ 0, -2560 ] (64×64 superblocks); code the predictor itself.
        let mut nw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        nw.intrabc_mv = Some([0, -2560]);
        // NE at (0, 2): the left-neighbour scan finds NW's stamped
        // [0, -2560]; code a different block vector ⇒ non-zero MV
        // difference bits.
        let mut ne = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        ne.intrabc_mv = Some([-8, -2560]);
        // SW / SE: plain else-arm leaves after the intrabc pair.
        let sw = SyntaxBlock::skip_leaf(V_PRED as u8, Some(DC_PRED as u8));
        let se = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x5A);

        let cols = 4usize;
        for (r0, c0, want) in [(0usize, 0usize, [0i16, -2560]), (0, 2, [-8, -2560])] {
            for dr in 0..2usize {
                for dc in 0..2usize {
                    let cell = (r0 + dr) * cols + (c0 + dc);
                    assert_eq!(walker.is_inters()[cell], 1, "intrabc IsInters stamp");
                    assert_eq!(
                        [
                            walker.mvs()[(cell * 2) * 2],
                            walker.mvs()[(cell * 2) * 2 + 1]
                        ],
                        want,
                        "decoded Mv[0] at ({}, {})",
                        r0 + dr,
                        c0 + dc
                    );
                }
            }
        }
        // The else-arm leaves stayed intra.
        assert_eq!(walker.is_inters()[2 * cols], 0, "SW else-arm IsInters");
    }

    /// Three palette leaves whose §5.11.46 entries + §8.3.2
    /// `has_palette_y` ctx + §5.11.49 palette cache all come from
    /// neighbour state: NE merges the cache from NW (left neighbour),
    /// SW from NW (above neighbour). Each leaf then runs the full
    /// §5.11.49 `palette_tokens()` colour-index walk; NW additionally
    /// carries a UV palette (entries + UV token walk).
    #[test]
    fn r282_syntax_round_trip_palette_neighbours_share_cache() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.allow_screen_content_tools = true;

        // 8×8 checkerboard over palette indices {0, 1}.
        let checker: Vec<u8> = (0..64).map(|i| (((i / 8) + (i % 8)) & 1) as u8).collect();
        // Vertical stripes over {0, 1}.
        let stripes: Vec<u8> = (0..64).map(|i| u8::from((i % 8) >= 4)).collect();
        // Three-colour diagonal bands over {0, 1, 2}.
        let bands: Vec<u8> = (0..64).map(|i| (((i / 8) + (i % 8)) % 3) as u8).collect();

        let colors = |vals: &[u16]| {
            let mut a = [0u16; PALETTE_COLORS];
            a[..vals.len()].copy_from_slice(vals);
            a
        };

        // NW: luma palette [10, 200] + chroma palette ([20, 30] /
        // [40, 50], direct-literal V arm).
        let mut nw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        nw.palette = SyntaxPalette {
            size_y: 2,
            colors_y: colors(&[10, 200]),
            color_map_y: checker.clone(),
            size_uv: 2,
            colors_u: colors(&[20, 30]),
            colors_v: colors(&[40, 50]),
            delta_encode_v: false,
            color_map_uv: stripes.clone(),
        };
        // NE: luma palette [10, 100] — `10` is a §5.11.49 cache hit
        // against NW's left-neighbour palette, `100` a new literal.
        let mut ne = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        ne.palette = SyntaxPalette {
            size_y: 2,
            colors_y: colors(&[10, 100]),
            color_map_y: stripes.clone(),
            ..SyntaxPalette::default()
        };
        // SW: luma palette [5, 60, 200] — `200` hits NW's
        // above-neighbour cache.
        let mut sw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        sw.palette = SyntaxPalette {
            size_y: 3,
            colors_y: colors(&[5, 60, 200]),
            color_map_y: bands.clone(),
            ..SyntaxPalette::default()
        };
        // SE: no palette (its `has_palette_y` ctx sees both palette
        // neighbours).
        let se = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0xC3);

        let cols = 4usize;
        let area = 16usize;
        // Luma PaletteSizes / PaletteColors stamps.
        for (r0, c0, size, first_two) in [
            (0usize, 0usize, 2u8, [10u16, 200]),
            (0, 2, 2, [10, 100]),
            (2, 0, 3, [5, 60]),
            (2, 2, 0, [0, 0]),
        ] {
            let cell = r0 * cols + c0;
            assert_eq!(walker.palette_sizes()[cell], size, "PaletteSizes[0] stamp");
            if size > 0 {
                assert_eq!(
                    walker.palette_colors()[cell * PALETTE_COLORS],
                    first_two[0],
                    "PaletteColors[0][..][0]"
                );
                assert_eq!(
                    walker.palette_colors()[cell * PALETTE_COLORS + 1],
                    first_two[1],
                    "PaletteColors[0][..][1]"
                );
            }
        }
        // Chroma palette stamps for NW (planes 1 + 2).
        assert_eq!(walker.palette_sizes()[area], 2, "PaletteSizes[1] NW");
        assert_eq!(walker.palette_sizes()[2 * area], 2, "PaletteSizes[2] NW");
        assert_eq!(
            walker.palette_colors()[area * PALETTE_COLORS],
            20,
            "PaletteColors[1][NW][0]"
        );
        assert_eq!(
            walker.palette_colors()[2 * area * PALETTE_COLORS],
            40,
            "PaletteColors[2][NW][0]"
        );
    }

    /// §5.11.7 prefix threading: the `SegIdPreSkip` segment-id arm
    /// (S() per leaf with the §5.11.9 neighbour cascade from the
    /// mirror) plus §5.11.12 / §5.11.13 delta-q / delta-lf writes,
    /// accumulated by the decode walker.
    #[test]
    fn r282_syntax_round_trip_segmentation_and_deltas() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.segmentation_enabled = true;
        params.seg_id_pre_skip = true;
        params.last_active_seg_id = 3;
        params.read_deltas = true;
        params.delta_q_res = 2;
        params.delta_lf_present = true;
        params.delta_lf_res = 1;

        // §5.11.7 line 11 `ReadDeltas = 0`: only the superblock's
        // FIRST block codes the §5.11.12/§5.11.13 delta syntax — the
        // later leaves MUST carry zero deltas (the write side rejects
        // a non-zero delta it can no longer code).
        let mut nw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        nw.segment_id = 2;
        nw.reduced_delta_q_index = 3;
        nw.reduced_delta_lf = [1, 0, 0, 0];
        let mut ne = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        ne.segment_id = 1;
        let mut sw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        sw.segment_id = 0;
        let mut se = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        se.segment_id = 3;

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x96);

        let cols = 4usize;
        for (r0, c0, want) in [(0usize, 0usize, 2i32), (0, 2, 1), (2, 0, 0), (2, 2, 3)] {
            assert_eq!(
                walker.segment_ids()[r0 * cols + c0],
                want,
                "SegmentIds stamp at ({r0}, {c0})"
            );
        }
        // §5.11.12 accumulator: 0 + (3 << 2) ⇒ Clip3(1, 255, 12) = 12
        // (only the first leaf's delta is coded — §5.11.7 line 11).
        assert_eq!(walker.current_q_index(), 12, "CurrentQIndex accumulation");
        // §5.11.13 accumulator: 0 + (1 << 1) = 2 on LF index 0.
        assert_eq!(walker.current_delta_lf()[0], 2, "DeltaLF[0] accumulation");
    }

    /// r283/r284 scope guards: §5.11.16 (`tx_mode_select`) write
    /// threading is a follow-up arc; an intrabc leaf without
    /// `allow_intrabc` and every `residual_quant` count/length
    /// mismatch (including on the now-live `skip == 0` intra-block-
    /// copy §5.11.36 arm) are caller bugs.
    #[test]
    fn r283_write_block_syntax_scope_rejects() {
        let params = SyntaxFrameParams::intra_8bit_baseline();
        let mut writer = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();

        // skip = 0 with NO residual_quant arrays ⇒ §5.11.34 TU
        // shortfall (BLOCK_8X8 4:4:4 visits 3 TUs).
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.skip = 0;
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // skip = 0 with a wrong-length Quant[] array (TX_8X8 needs 64).
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.skip = 0;
        block.residual_quant = vec![vec![0i32; 16], vec![0i32; 64], vec![0i32; 64]];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // skip = 0 with SURPLUS arrays (4 supplied, 3 visited).
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.skip = 0;
        block.residual_quant = vec![vec![0i32; 64]; 4];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // skip = 1 with residual_quant arrays ⇒ caller bug (the
        // §5.11.35 `!skip` gate never consumes them).
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.residual_quant = vec![vec![0i32; 64]; 3];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // `tx_size = Some(t)` differing from the §5.11.15 spec-forced
        // value on a bit-silent arm (`TxMode != TX_MODE_SELECT` ⇒ no
        // `tx_depth` symbol ⇒ TxSize must be `Max_Tx_Size_Rect[
        // BLOCK_8X8 ] = TX_8X8`) is a caller bug (r285 — this
        // sub-case previously asserted the pre-r285 wholesale
        // `tx_mode_select` reject, whose premise the §5.11.16 write
        // threading retired).
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.tx_size = Some(TX_4X4 as u8);
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // intrabc leaf with allow_intrabc = false ⇒ caller bug (the
        // reader's §5.11.7 fall-through forces use_intrabc = 0).
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // skip = 0 on the intra-block-copy arm (r284: the §5.11.36
        // inter-luma transform-tree write threading is LIVE) with a
        // TU-count shortfall (no residual_quant arrays supplied for
        // the 3 visited TUs) ⇒ caller bug.
        let mut params_bc = params.clone();
        params_bc.allow_intrabc = true;
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        block.skip = 0;
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params_bc,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // r283 — §5.11.34 residual() write threading round-trips. Same
    // sentinel + whole-CDF + grid-parity harness as the r282 tests;
    // every tree below contains `skip == 0` leaves whose per-TU
    // §5.11.39 coefficient writes the §5.11.5 decode walker consumes
    // through its §5.11.34 dispatcher.
    // -----------------------------------------------------------------

    /// Build a `Quant[]` array of `len` zeros with `(pos, val)`
    /// overrides.
    fn quant_with(len: usize, entries: &[(usize, i32)]) -> Vec<i32> {
        let mut q = vec![0i32; len];
        for &(pos, val) in entries {
            q[pos] = val;
        }
        q
    }

    /// Single BLOCK_16X16 leaf, 4:4:4: three TX_16X16 TUs (Y/U/V) with
    /// distinct DC + AC commitments (positive, negative, and a
    /// larger-magnitude value through the §5.11.39 coeff_br + golomb
    /// tail).
    #[test]
    fn r283_syntax_round_trip_residual_single_16x16_leaf() {
        let params = SyntaxFrameParams::intra_8bit_baseline();
        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.skip = 0;
        leaf.residual_quant = vec![
            // Y: DC + two low-scan AC taps.
            quant_with(256, &[(0, 5), (1, -2), (16, 1)]),
            // U: negative DC only.
            quant_with(256, &[(0, -7)]),
            // V: a magnitude past the §5.11.39 base-range cap (golomb
            // tail) at DC plus a mid-scan tap.
            quant_with(256, &[(0, 40), (2, 3)]),
        ];
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0xA7);
        assert!(
            walker.skips().iter().all(|&s| s == 0),
            "decoded Skips[] = 0 over the leaf footprint"
        );
        // §5.11.16 else-arm TxSizes stamp: TX_16X16 over the block.
        assert!(
            walker
                .tx_sizes()
                .iter()
                .all(|&t| t as usize == crate::cdf::TX_16X16),
            "TxSizes[] = TX_16X16 over the 16×16 leaf"
        );
    }

    /// SPLIT into four BLOCK_8X8 leaves mixing skip / non-skip and
    /// all-zero / non-zero commitments: the later leaves' §8.3.2
    /// `skip` ctx comes from the earlier leaves' `Skips[]` stamps on
    /// BOTH sides, and the §8.3 coefficient-CDF adaptation from the
    /// earlier TUs feeds the later TUs' writes.
    #[test]
    fn r283_syntax_round_trip_residual_split_mixed_skip() {
        let params = SyntaxFrameParams::intra_8bit_baseline();

        // NW: non-zero coefficients on all three TX_8X8 TUs.
        let mut nw = SyntaxBlock::skip_leaf(V_PRED as u8, Some(DC_PRED as u8));
        nw.skip = 0;
        nw.residual_quant = vec![
            quant_with(64, &[(0, 3), (1, 1)]),
            quant_with(64, &[(0, -1)]),
            quant_with(64, &[(8, 2)]),
        ];
        // NE: skip leaf (no TUs).
        let ne = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        // SW: non-skip with ALL-ZERO commitments — every plane writes
        // `all_zero = 1`.
        let mut sw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        sw.skip = 0;
        sw.residual_quant = vec![vec![0i32; 64]; 3];
        // SE: single negative DC on luma, zeros on chroma.
        let mut se = SyntaxBlock::skip_leaf(2, Some(DC_PRED as u8));
        se.skip = 0;
        se.residual_quant = vec![quant_with(64, &[(0, -4)]), vec![0i32; 64], vec![0i32; 64]];

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x3C);

        let cols = 4usize;
        for (r0, c0, want) in [(0usize, 0usize, 0u8), (0, 2, 1), (2, 0, 0), (2, 2, 0)] {
            assert_eq!(
                walker.skips()[r0 * cols + c0],
                want,
                "Skips stamp at ({r0}, {c0})"
            );
        }
    }

    /// Lossless leaf: the §5.11.15 `Lossless ⇒ TX_4X4` short-circuit
    /// fans a BLOCK_8X8 4:4:4 leaf into 12 TX_4X4 TUs (4 per plane),
    /// exercising the §5.11.34 stepX/stepY iteration order across
    /// multiple TUs per plane.
    #[test]
    fn r283_syntax_round_trip_residual_lossless_tx4x4_fan_out() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.lossless_array = [true; MAX_SEGMENTS];
        params.coded_lossless = true;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.skip = 0;
        // 12 TUs in dispatch order: plane 0 rows (y=0: x=0,1; y=1:
        // x=0,1) then planes 1, 2. Give the first TU of each plane a
        // distinct DC; leave the rest all-zero.
        let mut tus = Vec::new();
        for plane in 0..3 {
            for tu in 0..4 {
                tus.push(if tu == 0 {
                    quant_with(16, &[(0, plane + 1)])
                } else {
                    vec![0i32; 16]
                });
            }
        }
        leaf.residual_quant = tus;
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 2, 2, BLOCK_8X8, &params, 0xE1);
        assert!(
            walker.tx_sizes().iter().all(|&t| t as usize == TX_4X4),
            "lossless TxSizes[] = TX_4X4"
        );
    }

    /// 4:2:0 subsampled leaf: BLOCK_16X16 luma TX_16X16 + two TX_8X8
    /// chroma TUs through the §5.11.37 / §5.11.38 chroma-size
    /// derivations (`get_tx_size` / `get_plane_residual_size` with
    /// `subX = subY = 1`).
    #[test]
    fn r283_syntax_round_trip_residual_420_chroma_sizing() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.subsampling_x = 1;
        params.subsampling_y = 1;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.skip = 0;
        leaf.residual_quant = vec![
            quant_with(256, &[(0, 2), (1, 1)]),
            quant_with(64, &[(0, -3)]),
            quant_with(64, &[(0, 6), (8, -1)]),
        ];
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let _ = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x4B);
    }

    /// Monochrome leaf: `NumPlanes == 1` ⇒ the §5.11.34 plane loop
    /// visits luma only (one TX_16X16 TU).
    #[test]
    fn r283_syntax_round_trip_residual_monochrome_luma_only() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.num_planes = 1;
        params.mono_chrome = true;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, None);
        leaf.skip = 0;
        leaf.residual_quant = vec![quant_with(256, &[(0, 9), (3, -2)])];
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let _ = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0xD2);
    }

    /// Directional chroma mode (`UVMode = V_PRED`): the §5.11.40
    /// `Mode_To_Txfm[ UVMode ]` chroma derivation departs from
    /// DCT_DCT, changing the chroma TUs' §7.5 scan + §8.3.2 tx-class
    /// axes on both sides.
    #[test]
    fn r283_syntax_round_trip_residual_chroma_mode_to_txfm() {
        let params = SyntaxFrameParams::intra_8bit_baseline();
        let mut leaf = SyntaxBlock::skip_leaf(V_PRED as u8, Some(V_PRED as u8));
        leaf.skip = 0;
        leaf.residual_quant = vec![
            quant_with(256, &[(0, 1)]),
            quant_with(256, &[(0, 2), (16, 1)]),
            quant_with(256, &[(0, -2)]),
        ];
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let _ = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x78);
    }

    // -----------------------------------------------------------------
    // r284 — §8.3.2 `all_zero` / `dc_sign` ctx derivation lockstep.
    // The harness now asserts the four §6.10.2 context arrays
    // (`Above{Level,Dc}Context` / `Left{Level,Dc}Context`) cell-for-
    // cell after every tree, so each round-trip below pins that every
    // TU's coefficient symbols used the same neighbour-derived CDF
    // rows on both sides.
    // -----------------------------------------------------------------

    /// SPLIT into four BLOCK_8X8 leaves where the §5.11.39 tail stamps
    /// and §5.11.42 skip resets interleave:
    ///
    /// * NW (`skip = 0`, negative Y DC) stamps `culLevel`/`dcCategory`
    ///   over its columns/rows — NE's and SW's ctx walks observe them.
    /// * NE (`skip = 1`) §5.11.42-resets its columns AND rows 0..1
    ///   (wiping NW's left stamps).
    /// * SW (`skip = 0`) derives its luma `all_zero` ctx from NW's
    ///   above stamps (`top != 0` arm) and its `dc_sign` ctx from
    ///   NW's `dcCategory = 1` (negative census ⇒ ctx 1), then
    ///   overwrites columns 0..1.
    /// * SE (`skip = 0`, positive Y DC) sees NE's reset above (0) and
    ///   SW's left stamps — the asymmetric `top == 0 || left == 0`
    ///   luma arm.
    ///
    /// The end-state array asserts pin the §5.11.39 stamp / §5.11.42
    /// reset choreography beyond the harness parity check.
    #[test]
    fn r284_syntax_round_trip_txb_level_context_stamps_and_skip_reset() {
        let params = SyntaxFrameParams::intra_8bit_baseline();

        // NW: negative Y DC (dcCategory = 1), culLevel = 3 + 1 = 4.
        let mut nw = SyntaxBlock::skip_leaf(V_PRED as u8, Some(DC_PRED as u8));
        nw.skip = 0;
        nw.residual_quant = vec![
            quant_with(64, &[(0, -3), (1, 1)]),
            quant_with(64, &[(0, -1)]),
            quant_with(64, &[(8, 2)]),
        ];
        // NE: skip ⇒ §5.11.42 reset over columns 2..3 + rows 0..1.
        let ne = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        // SW: negative Y DC (dcCategory = 1), culLevel = 5 + 2 + 1 = 8.
        let mut sw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        sw.skip = 0;
        sw.residual_quant = vec![
            quant_with(64, &[(0, -5), (1, 2), (8, 1)]),
            vec![0i32; 64],
            quant_with(64, &[(0, 4)]),
        ];
        // SE: positive Y DC (dcCategory = 2), culLevel = 6 + 1 = 7.
        let mut se = SyntaxBlock::skip_leaf(2, Some(DC_PRED as u8));
        se.skip = 0;
        se.residual_quant = vec![
            quant_with(64, &[(0, 6), (2, 1)]),
            vec![0i32; 64],
            vec![0i32; 64],
        ];

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x5D);

        // Luma plane (flat base 0; 4 columns / 4 rows). Decode order
        // NW → NE → SW → SE leaves:
        //   above cols 0..1 = SW's stamps (culLevel 8, dcCategory 1),
        //   above cols 2..3 = SE's stamps (culLevel 7, dcCategory 2),
        //   left rows 0..1 = 0 (NE's §5.11.42 reset wiped NW's),
        //   left rows 2..3 = SE's stamps (overwriting SW's).
        let alc = walker.above_level_context();
        let adc = walker.above_dc_context();
        let llc = walker.left_level_context();
        let ldc = walker.left_dc_context();
        assert_eq!(&alc[0..4], &[8, 8, 7, 7], "luma AboveLevelContext");
        assert_eq!(&adc[0..4], &[1, 1, 2, 2], "luma AboveDcContext");
        assert_eq!(&llc[0..4], &[0, 0, 7, 7], "luma LeftLevelContext");
        assert_eq!(&ldc[0..4], &[0, 0, 2, 2], "luma LeftDcContext");
    }

    /// Lossless 4:2:0 BLOCK_16X16 leaf — 16 luma TX_4X4 TUs + 4 TUs
    /// per chroma plane over the BLOCK_8X8 chroma residual size. The
    /// chroma TUs take the §8.3.2 chroma `all_zero` arm with
    /// `bw * bh = 64 > w * h = 16` (the `ctx += 3` adjustment, rows
    /// 10..=12 of `TileTxbSkipCdf`), and the second/later TUs of every
    /// plane fold the earlier TUs' stamps through the OR-census
    /// (`above != 0` / `left != 0`) arms.
    #[test]
    fn r284_syntax_round_trip_txb_ctx_lossless_420_chroma_bsize_arm() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.subsampling_x = 1;
        params.subsampling_y = 1;
        params.lossless_array = [true; MAX_SEGMENTS];
        params.coded_lossless = true;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.skip = 0;
        // 24 TUs in §5.11.34 dispatch order: 16 luma (4×4 grid of
        // TX_4X4), then 4 per chroma plane (2×2 grid). Alternate
        // signed DCs + an AC tap so the running census flips arms.
        let mut tus = Vec::new();
        for tu in 0..16 {
            tus.push(match tu % 3 {
                0 => quant_with(16, &[(0, 2)]),
                1 => quant_with(16, &[(0, -1), (1, 1)]),
                _ => vec![0i32; 16],
            });
        }
        for plane in 0..2 {
            for tu in 0..4 {
                tus.push(if (tu + plane) % 2 == 0 {
                    quant_with(16, &[(0, if plane == 0 { -4 } else { 4 })])
                } else {
                    vec![0i32; 16]
                });
            }
        }
        leaf.residual_quant = tus;
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x9E);

        // The chroma planes stamped their 4-sample columns: plane 1
        // base = mi_cols = 4, plane 2 base = 8 (subsampled x4 ∈ 0..2).
        let alc = walker.above_level_context();
        let adc = walker.above_dc_context();
        // Plane 1 row of chroma TUs decoded last for each column:
        // TU (y=1, x=0) all-zero (tu=2 ⇒ (2+0)%2=0? — see commitment
        // map below); pin the exact stamps instead of recomputing:
        // plane 1 TUs in order (y0x0, y0x1, y1x0, y1x1) carry DCs
        // (-4, 0, -4, 0) ⇒ final column stamps (4, 0) levels with
        // dcCategory (1, 0). Plane 2 TUs carry (0, 4, 0, 4) ⇒ final
        // column stamps (0, 4) / (0, 2).
        assert_eq!(&alc[4..6], &[4, 0], "plane-1 AboveLevelContext");
        assert_eq!(&adc[4..6], &[1, 0], "plane-1 AboveDcContext");
        assert_eq!(&alc[8..10], &[0, 4], "plane-2 AboveLevelContext");
        assert_eq!(&adc[8..10], &[0, 2], "plane-2 AboveDcContext");
    }

    /// r284 — §5.11.36 inter-arm `transform_tree` write side: a
    /// `skip == 0` intra-block-copy leaf (`is_inter = 1`) routes its
    /// luma plane through the [`write_transform_tree`] recursion (one
    /// TX_8X8 leaf here — `InterTxSizes` carries the §5.11.16
    /// else-arm `Max_Tx_Size_Rect` stamp) while chroma takes the
    /// direct §5.11.34 loop, and every §5.11.39 writer call takes the
    /// `is_inter = 1` CDF axes (`eob_pt_*[ 1 ]`). The decode walker
    /// consumes the same TUs through its §5.11.36 recursion — the
    /// sentinel + whole-CDF + context-array parity pin the lockstep.
    #[test]
    fn r284_syntax_round_trip_intrabc_skip0_transform_tree() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.allow_intrabc = true;

        // NW: intra-block-copy leaf with residual — negative luma DC
        // + chroma taps so the §8.3.2 stamps engage on the inter arm.
        let mut nw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        nw.intrabc_mv = Some([0, -2560]);
        nw.skip = 0;
        nw.residual_quant = vec![
            quant_with(64, &[(0, -3), (1, 2)]),
            quant_with(64, &[(0, 4)]),
            quant_with(64, &[(2, 1)]),
        ];
        // NE: skip intrabc leaf (the r282 shape — §5.11.42 reset).
        let mut ne = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        ne.intrabc_mv = Some([-8, -2560]);
        // SW: plain else-arm non-skip leaf — its `all_zero` /
        // `dc_sign` ctx walks read NW's inter-arm stamps from above.
        let mut sw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        sw.skip = 0;
        sw.residual_quant = vec![
            quant_with(64, &[(0, 7)]),
            vec![0i32; 64],
            quant_with(64, &[(0, -2)]),
        ];
        // SE: plain skip leaf.
        let se = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0xB6);

        // NW decoded as an inter (intrabc) block with residual.
        assert_eq!(walker.is_inters()[0], 1, "NW IsInters");
        assert_eq!(walker.skips()[0], 0, "NW skip = 0");
        // Final luma context arrays: NW's TX_8X8 stamps on columns
        // 0..1 were overwritten by SW (culLevel 7, positive DC ⇒
        // dcCategory 2); NE + SE skip resets keep columns 2..3 at 0.
        let alc = walker.above_level_context();
        let adc = walker.above_dc_context();
        assert_eq!(&alc[0..4], &[7, 7, 0, 0], "luma AboveLevelContext");
        assert_eq!(&adc[0..4], &[2, 2, 0, 0], "luma AboveDcContext");
    }

    /// §5.11.15 `tx_depth` write threading under `TX_MODE_SELECT`
    /// (r285): a single BLOCK_16X16 non-skip intra leaf commits
    /// `TxSize = TX_8X8` (`tx_depth = 1`), fanning the §5.11.34 luma
    /// plane into four TX_8X8 TUs while 4:4:4 chroma keeps its
    /// §5.11.37 TX_16X16 sizing (chroma ignores the luma `TxSize`).
    #[test]
    fn r285_syntax_round_trip_tx_depth_intra_16x16() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.tx_mode_select = true;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.skip = 0;
        leaf.tx_size = Some(TX_8X8 as u8);
        leaf.residual_quant = vec![
            // Y: four TX_8X8 TUs in row-major §5.11.34 step order.
            quant_with(64, &[(0, 5), (1, -2)]),
            quant_with(64, &[(0, 1)]),
            quant_with(64, &[(3, -4)]),
            quant_with(64, &[(0, 2), (8, 1)]),
            // U / V: one TX_16X16 TU each.
            quant_with(256, &[(0, -7)]),
            quant_with(256, &[(0, 40), (2, 3)]),
        ];
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0xC1);
        assert!(
            walker.tx_sizes().iter().all(|&t| t as usize == TX_8X8),
            "committed TxSize = TX_8X8 over the leaf footprint"
        );
        assert!(
            walker
                .inter_tx_sizes()
                .iter()
                .all(|&t| t as usize == TX_8X8),
            "§5.11.16 else-arm InterTxSizes fill follows the committed TxSize"
        );
    }

    /// SPLIT into four BLOCK_8X8 leaves under `TX_MODE_SELECT`: mixed
    /// `tx_depth` commitments (0 / 1 / 1 / silent), so later leaves'
    /// §8.3.2 `tx_depth` ctx reads the earlier leaves' `InterTxSizes`
    /// stamps on BOTH sides and the `Tx8x8Cdf` rows adapt across the
    /// tree. The SE leaf is an intra-block-copy skip leaf —
    /// `allowSelect = !skip || !is_inter = 0` keeps its §5.11.15 body
    /// bit-silent (the explicit `Some(TX_8X8)` commitment must match
    /// the spec-forced `Max_Tx_Size_Rect[ BLOCK_8X8 ]`).
    #[test]
    fn r285_syntax_round_trip_tx_depth_split_neighbour_ctx_and_silent_arm() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.tx_mode_select = true;
        params.allow_intrabc = true;

        // NW: `tx_size = None` ⇒ maxRect default (tx_depth = 0).
        let nw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        // NE: tx_depth = 1 (TX_8X8 → TX_4X4).
        let mut ne = SyntaxBlock::skip_leaf(V_PRED as u8, Some(DC_PRED as u8));
        ne.tx_size = Some(TX_4X4 as u8);
        // SW: tx_depth = 1 — its above-ctx reads NW's stamps.
        let mut sw = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        sw.tx_size = Some(TX_4X4 as u8);
        // SE: intrabc skip leaf — bit-silent §5.11.15.
        let mut se = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        se.intrabc_mv = Some([-8, -2560]);
        se.tx_size = Some(TX_8X8 as u8);

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0xC2);
        for r in 0..4usize {
            for c in 0..4usize {
                let want = match (r >= 2, c >= 2) {
                    (false, false) | (true, true) => TX_8X8,
                    _ => TX_4X4,
                };
                assert_eq!(
                    walker.inter_tx_sizes()[r * 4 + c] as usize,
                    want,
                    "InterTxSizes[{r}][{c}]"
                );
            }
        }
    }

    /// §5.11.16 var-tx arm round-trip (r285): a BLOCK_32X32
    /// intra-block-copy `skip == 0` leaf under `TX_MODE_SELECT`
    /// commits a genuinely variable §5.11.17 tree — the TX_32X32 root
    /// splits into four TX_16X16, and the NE quadrant splits again
    /// into four TX_8X8 (depth 2 = MAX_VARTX_DEPTH ⇒ spec-forced
    /// terminals, no S() on either side). The §5.11.36 write
    /// recursion then follows the per-leaf `InterTxSizes[]` stamps
    /// (one 16×16 TU, four 8×8 TUs, two more 16×16 TUs), pinning the
    /// transform-tree write side beyond uniform grids; chroma sizes
    /// off the §5.11.37 derivation (one TX_32X32 TU per plane).
    #[test]
    fn r285_syntax_round_trip_var_tx_intrabc_32x32() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.tx_mode_select = true;
        params.allow_intrabc = true;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.intrabc_mv = Some([0, -2560]);
        leaf.skip = 0;
        leaf.var_tx_trees = vec![VarTxSyntaxTree::Split(vec![
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Split(vec![
                VarTxSyntaxTree::Leaf,
                VarTxSyntaxTree::Leaf,
                VarTxSyntaxTree::Leaf,
                VarTxSyntaxTree::Leaf,
            ]),
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Leaf,
        ])];
        leaf.residual_quant = vec![
            // Y, §5.11.36 visit order: NW 16×16 leaf, NE quad of 8×8
            // leaves, SW 16×16, SE 16×16.
            quant_with(256, &[(0, 9), (1, -1)]),
            quant_with(64, &[(0, -3)]),
            quant_with(64, &[(1, 2)]),
            quant_with(64, &[(0, 1)]),
            vec![0i32; 64], // all_zero NE-SE 8×8 TU
            quant_with(256, &[(0, -5)]),
            quant_with(256, &[(2, 4)]),
            // U / V: one TX_32X32 TU each.
            quant_with(1024, &[(0, 6)]),
            quant_with(1024, &[(0, -2), (1, 1)]),
        ];
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 8, 8, BLOCK_32X32, &params, 0xC3);

        // Per-leaf §5.11.17 InterTxSizes stamps: TX_8X8 over the NE
        // quadrant, TX_16X16 elsewhere.
        for r in 0..8usize {
            for c in 0..8usize {
                let want = if r < 4 && c >= 4 { TX_8X8 } else { TX_16X16 };
                assert_eq!(
                    walker.inter_tx_sizes()[r * 8 + c] as usize,
                    want,
                    "InterTxSizes[{r}][{c}]"
                );
            }
        }
        // §5.11.5 outer TxSizes fill = the recursion's LAST
        // terminal-else txSz (the SE TX_16X16 leaf).
        assert!(
            walker.tx_sizes().iter().all(|&t| t as usize == TX_16X16),
            "TxSizes[] = last terminal-else txSz over the footprint"
        );
        assert_eq!(walker.is_inters()[0], 1, "intrabc leaf is inter");
        assert_eq!(walker.skips()[0], 0, "skip = 0");
    }

    /// §5.11.17 frame-edge clip (r285): a BLOCK_32X32 var-tx leaf on
    /// a 6-mi-row frame. The SW TX_16X16 quadrant splits into four
    /// TX_8X8 children whose bottom pair sits at `row = 6 >= MiRows`
    /// — the §5.11.17 early return emits nothing and stamps nothing
    /// for them on either side (they are spec-forced terminals at
    /// depth 2, and their footprints are entirely off-frame). The
    /// §5.11.36 luma walk clips the matching TUs via the §5.11.35
    /// `startY >= maxY` early return, so they consume no
    /// `residual_quant` entries.
    #[test]
    fn r285_syntax_round_trip_var_tx_frame_edge_clip() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.tx_mode_select = true;
        params.allow_intrabc = true;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.intrabc_mv = Some([0, -2560]);
        leaf.skip = 0;
        leaf.var_tx_trees = vec![VarTxSyntaxTree::Split(vec![
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Split(vec![
                VarTxSyntaxTree::Leaf,
                VarTxSyntaxTree::Leaf,
                // (6, 0) / (6, 2): off-frame — no symbol, no stamp.
                VarTxSyntaxTree::Leaf,
                VarTxSyntaxTree::Leaf,
            ]),
            VarTxSyntaxTree::Leaf,
        ])];
        leaf.residual_quant = vec![
            // Y: NW 16×16, NE 16×16, SW 8×8 ×2 (bottom pair clipped),
            // SE 16×16.
            quant_with(256, &[(0, 3)]),
            quant_with(256, &[(1, -2)]),
            quant_with(64, &[(0, 4)]),
            quant_with(64, &[(0, -1)]),
            quant_with(256, &[(0, 2)]),
            // U / V: one TX_32X32 TU each.
            quant_with(1024, &[(0, -6)]),
            quant_with(1024, &[(0, 1)]),
        ];
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 6, 8, BLOCK_32X32, &params, 0xC5);

        // InterTxSizes: TX_8X8 over the on-frame SW band (rows 4-5,
        // cols 0-3), TX_16X16 elsewhere.
        for r in 0..6usize {
            for c in 0..8usize {
                let want = if r >= 4 && c < 4 { TX_8X8 } else { TX_16X16 };
                assert_eq!(
                    walker.inter_tx_sizes()[r * 8 + c] as usize,
                    want,
                    "InterTxSizes[{r}][{c}]"
                );
            }
        }
        assert!(
            walker.tx_sizes().iter().all(|&t| t as usize == TX_16X16),
            "TxSizes[] = last terminal-else txSz"
        );
    }

    /// Lossless segment under `TX_MODE_SELECT` (r285): the §5.11.15
    /// `Lossless` short-circuit forces `TxSize = TX_4X4` with no
    /// symbol on either side.
    #[test]
    fn r285_syntax_round_trip_tx_mode_select_lossless_silent() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.tx_mode_select = true;
        params.lossless_array[0] = true;
        let leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 2, 2, BLOCK_8X8, &params, 0xC4);
        assert!(
            walker.tx_sizes().iter().all(|&t| t as usize == TX_4X4),
            "Lossless forces TX_4X4 with no tx_depth symbol"
        );
    }

    /// §5.11.16 write-driver caller-bug battery (r285).
    #[test]
    fn r285_write_block_tx_size_scope_rejects() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.tx_mode_select = true;
        params.allow_intrabc = true;
        let mut writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();

        // (1) intra leaf: `tx_size` unreachable from maxRectTxSize
        // within MAX_TX_DEPTH = 2 splits (TX_64X64 → TX_4X4 needs 4).
        let mut state = PartitionSyntaxWriter::new(16, 16, single_tile(16, 16)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.tx_size = Some(TX_4X4 as u8);
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_64X64,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (2) var_tx_trees supplied on the intra (else) arm.
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.var_tx_trees = vec![VarTxSyntaxTree::Leaf];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_16X16,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (3) scalar `tx_size` commitment on the var-tx arm.
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        block.skip = 0;
        block.tx_size = Some(TX_16X16 as u8);
        block.var_tx_trees = vec![VarTxSyntaxTree::Leaf];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_16X16,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (4) var-tx arm tree shortfall (no trees supplied).
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        block.skip = 0;
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_16X16,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (5) var-tx arm tree surplus (two trees on a one-position
        // BLOCK_16X16 footprint).
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        block.skip = 0;
        block.var_tx_trees = vec![VarTxSyntaxTree::Leaf, VarTxSyntaxTree::Leaf];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_16X16,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (6) Split at a spec-forced terminal (`txSz == TX_4X4`).
        let mut state = PartitionSyntaxWriter::new(2, 2, single_tile(2, 2)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        block.skip = 0;
        block.var_tx_trees = vec![VarTxSyntaxTree::Split(vec![
            VarTxSyntaxTree::Split(Vec::new()),
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Leaf,
        ])];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_8X8,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (7) Split child-count mismatch (3 children on a square
        // quad decomposition).
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        block.skip = 0;
        block.var_tx_trees = vec![VarTxSyntaxTree::Split(vec![
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Leaf,
            VarTxSyntaxTree::Leaf,
        ])];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_16X16,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (8) var_tx_trees on the intrabc `skip == 1` leaf — the
        // §5.11.16 else arm (`allowSelect = 0`) takes no trees.
        let mut state = PartitionSyntaxWriter::new(4, 4, single_tile(4, 4)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.intrabc_mv = Some([0, -2560]);
        block.var_tx_trees = vec![VarTxSyntaxTree::Leaf];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_16X16,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// r286 — §5.11.47 `transform_type` write side with a non-zero
    /// `base_q_idx`: a single BLOCK_8X8 4:4:4 leaf whose TX_8X8 luma TU
    /// commits `ADST_ADST`. With `base_q_idx > 0` the §5.11.47 `set > 0
    /// && qIdx > 0` guard opens, so the encoder emits an `intra_tx_type`
    /// S() (TX_8X8 intra → `TX_SET_INTRA_1`, 7 symbols) and the decoder
    /// reads it back, stamping the same `ADST_ADST` over the luma TU's
    /// `TxTypes[]` footprint. The `syntax_round_trip` `TxTypes parity`
    /// assertion + sentinel sync pin the write↔read lockstep.
    #[test]
    fn r286_syntax_round_trip_transform_type_intra_8x8_adst() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.quant.base_q_idx = 64;
        params.quant.current_q_index = 64;

        let mut leaf = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf.skip = 0;
        // One luma TX_8X8 TU + two chroma TX_8X8 TUs (4:4:4).
        leaf.residual_quant = vec![
            quant_with(64, &[(0, 5), (1, -2), (8, 1)]),
            quant_with(64, &[(0, -3)]),
            quant_with(64, &[(0, 4), (2, 1)]),
        ];
        // §5.11.47 luma-only `TxType` commitment — ADST_ADST is in
        // `Tx_Type_Intra_Inv_Set1`.
        leaf.residual_tx_type = vec![crate::cdf::ADST_ADST as u8];

        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, walker) = syntax_round_trip(&node, 2, 2, BLOCK_8X8, &params, 0xD5);
        // The decoded `TxTypes[]` carries ADST_ADST over the luma TU's
        // 2×2 (8×8 ⇒ 2 mi) footprint.
        assert!(
            walker
                .tx_types()
                .iter()
                .all(|&t| t as usize == crate::cdf::ADST_ADST),
            "decoded TxTypes[] = ADST_ADST over the 8×8 luma footprint"
        );
    }

    /// r286 — §5.11.47 mixed-set fan-out: a BLOCK_16X16 leaf under
    /// `TX_MODE_SELECT` committed to TX_8X8 luma TUs (four of them) with
    /// distinct per-TU `TxType`s drawn from `TX_SET_INTRA_1`. Each
    /// luma TU codes its own `intra_tx_type` S(); the
    /// CDF adapts across the four reads on both sides, and the
    /// `TxTypes parity` assertion checks every TU stamped the committed
    /// value. The `V_PRED` Y-mode exercises a non-`DC_PRED` §8.3.2
    /// `intra_dir` CDF axis.
    #[test]
    fn r286_syntax_round_trip_transform_type_per_tu_fanout() {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.tx_mode_select = true;
        params.quant.base_q_idx = 110;
        params.quant.current_q_index = 110;

        let mut leaf = SyntaxBlock::skip_leaf(V_PRED as u8, Some(DC_PRED as u8));
        leaf.skip = 0;
        leaf.tx_size = Some(TX_8X8 as u8); // tx_depth = 1: 16×16 → four TX_8X8
        leaf.residual_quant = vec![
            // Y: four TX_8X8 TUs.
            quant_with(64, &[(0, 3), (1, 1)]),
            quant_with(64, &[(0, -2)]),
            quant_with(64, &[(0, 1), (8, 2)]),
            quant_with(64, &[(2, -1)]),
            // U / V: one TX_16X16 each (TX_SET routing differs but no
            // luma S()).
            quant_with(256, &[(0, -4)]),
            quant_with(256, &[(0, 6)]),
        ];
        // Four luma `TxType`s, all admissible in `TX_SET_INTRA_1`.
        leaf.residual_tx_type = vec![
            crate::cdf::ADST_DCT as u8,
            crate::cdf::DCT_ADST as u8,
            crate::cdf::ADST_ADST as u8,
            DCT_DCT as u8,
        ];

        let node = SyntaxNode::Leaf(Box::new(leaf));
        let (_enc, _walker) = syntax_round_trip(&node, 4, 4, BLOCK_16X16, &params, 0x9E);
    }

    /// r286 — §5.11.47 guard-closed paths stay bit-silent: (a)
    /// `base_q_idx == 0` (neutral default) forces every luma TU to
    /// `DCT_DCT` with no symbol; (b) a TX_32X32 luma TU routes to
    /// `TX_SET_DCTONLY` (`set == 0`) so no `intra_tx_type` symbol is
    /// coded even with `base_q_idx > 0`. An empty `residual_tx_type`
    /// leaves every luma TU at the `DCT_DCT` default.
    #[test]
    fn r286_syntax_round_trip_transform_type_dctonly_silent_paths() {
        // (a) base_q_idx = 0 — guard closed regardless of commitment.
        let params0 = SyntaxFrameParams::intra_8bit_baseline();
        let mut leaf0 = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf0.skip = 0;
        leaf0.residual_quant = vec![quant_with(256, &[(0, 2)]); 3];
        // Empty commitment ⇒ DCT_DCT default; bit-silent.
        let node0 = SyntaxNode::Leaf(Box::new(leaf0));
        let (_e0, w0) = syntax_round_trip(&node0, 4, 4, BLOCK_16X16, &params0, 0x11);
        assert!(
            w0.tx_types().iter().all(|&t| t as usize == DCT_DCT),
            "base_q_idx = 0 ⇒ TxTypes[] all DCT_DCT"
        );

        // (b) base_q_idx > 0 but TX_32X32 luma ⇒ TX_SET_DCTONLY (set ==
        // 0): the §5.11.48 reduction short-circuits the symbol. A
        // BLOCK_32X32 leaf with the default TxSize is one TX_32X32 luma
        // TU.
        let mut params1 = SyntaxFrameParams::intra_8bit_baseline();
        params1.quant.base_q_idx = 200;
        params1.quant.current_q_index = 200;
        let mut leaf1 = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        leaf1.skip = 0;
        leaf1.residual_quant = vec![
            quant_with(1024, &[(0, 3)]),
            quant_with(1024, &[(0, -1)]),
            quant_with(1024, &[(0, 2)]),
        ];
        // Commitment MUST be DCT_DCT on the DCTONLY path (a non-DCT_DCT
        // value here is a caller bug — exercised in the reject battery).
        leaf1.residual_tx_type = vec![DCT_DCT as u8];
        let node1 = SyntaxNode::Leaf(Box::new(leaf1));
        let (_e1, w1) = syntax_round_trip(&node1, 8, 8, BLOCK_32X32, &params1, 0x22);
        assert!(
            w1.tx_types().iter().all(|&t| t as usize == DCT_DCT),
            "TX_32X32 ⇒ TX_SET_DCTONLY ⇒ TxTypes[] all DCT_DCT"
        );
    }

    /// r286 — §5.11.47 write-side caller-bug battery: a non-`DCT_DCT`
    /// `TxType` committed on a guard-closed (`TX_SET_DCTONLY`) luma TU,
    /// and a `TxType` not admissible in the resolved `set`, both reject
    /// with [`Error::PartitionWalkOutOfRange`] before any bit is
    /// committed past the point of detection.
    #[test]
    fn r286_transform_type_scope_rejects() {
        // (1) Guard closed (TX_32X32 ⇒ DCTONLY) but commitment is
        // non-DCT_DCT.
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.quant.base_q_idx = 200;
        params.quant.current_q_index = 200;
        let mut writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = PartitionSyntaxWriter::new(8, 8, single_tile(8, 8)).unwrap();
        let mut block = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block.skip = 0;
        block.residual_quant = vec![
            quant_with(1024, &[(0, 3)]),
            quant_with(1024, &[(0, -1)]),
            quant_with(1024, &[(0, 2)]),
        ];
        block.residual_tx_type = vec![crate::cdf::ADST_ADST as u8];
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_32X32,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (2) Guard open (TX_8X8, base_q_idx > 0) but the committed
        // `TxType` is V_DCT — admissible in `TX_SET_INTRA_1`, so swap to
        // a value that ISN'T in any intra set: `FLIPADST_DCT` only
        // appears in the inter sets, never the intra ones.
        let mut params2 = SyntaxFrameParams::intra_8bit_baseline();
        params2.quant.base_q_idx = 64;
        params2.quant.current_q_index = 64;
        let mut writer2 = SymbolWriter::new(false);
        let mut cdfs2 = TileCdfContext::new_from_defaults();
        let mut state2 = PartitionSyntaxWriter::new(2, 2, single_tile(2, 2)).unwrap();
        let mut block2 = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block2.skip = 0;
        block2.residual_quant = vec![quant_with(64, &[(0, 1)]); 3];
        block2.residual_tx_type = vec![crate::cdf::FLIPADST_DCT as u8];
        let err2 = super::write_block_syntax(
            &mut writer2,
            &mut cdfs2,
            &mut state2,
            &block2,
            0,
            0,
            BLOCK_8X8,
            &params2,
        )
        .unwrap_err();
        assert!(matches!(err2, Error::PartitionWalkOutOfRange));

        // (3) A non-empty `residual_tx_type` with a surplus entry (more
        // committed luma `TxType`s than the block has luma TUs) is the
        // same caller bug as a `residual_quant` surplus. BLOCK_8X8
        // 4:4:4 with base_q_idx = 0 has one TX_16X16... no — one TX_8X8
        // luma TU; supplying two commitments is a surplus.
        let mut params3 = SyntaxFrameParams::intra_8bit_baseline();
        params3.quant.base_q_idx = 64;
        params3.quant.current_q_index = 64;
        let mut writer3 = SymbolWriter::new(false);
        let mut cdfs3 = TileCdfContext::new_from_defaults();
        let mut state3 = PartitionSyntaxWriter::new(2, 2, single_tile(2, 2)).unwrap();
        let mut block3 = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));
        block3.skip = 0;
        block3.residual_quant = vec![quant_with(64, &[(0, 1)]); 3];
        block3.residual_tx_type = vec![DCT_DCT as u8, DCT_DCT as u8];
        let err3 = super::write_block_syntax(
            &mut writer3,
            &mut cdfs3,
            &mut state3,
            &block3,
            0,
            0,
            BLOCK_8X8,
            &params3,
        )
        .unwrap_err();
        assert!(matches!(err3, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // r411 — §5.11.18 inter-frame leaf round-trips. Same
    // encode → sentinel → decode → CDF/grid-parity harness as the
    // r282 intra tests, with `frame_is_intra = false` and a §5.11.18
    // `InterFrameContext` mirroring the write-side
    // `SyntaxInterFrameParams` field-for-field.
    // -----------------------------------------------------------------

    use crate::cdf::{InterFrameContext, MODE_GLOBALMV, MODE_NEARESTMV, MODE_NEWMV};
    use crate::encoder::partition_tree::{SyntaxInterBlock, SyntaxInterFrameParams};

    /// Decode-side twin of a [`SyntaxInterFrameParams`] bundle.
    fn inter_ctx_for<'a>(ipp: &'a SyntaxInterFrameParams) -> InterFrameContext<'a> {
        let mut c = InterFrameContext::identity_default(&ipp.motion_field_mvs);
        c.skip_mode_present = ipp.skip_mode_present;
        c.skip_mode_frame = [
            i32::from(ipp.skip_mode_frame[0]),
            i32::from(ipp.skip_mode_frame[1]),
        ];
        c.reference_select = ipp.reference_select;
        c.gm_type = ipp.gm_type;
        c.gm_params = ipp.gm_params;
        c.ref_frame_sign_bias = ipp.ref_frame_sign_bias;
        c.allow_high_precision_mv = ipp.allow_high_precision_mv;
        c.force_integer_mv = ipp.force_integer_mv;
        c.use_ref_frame_mvs = ipp.use_ref_frame_mvs;
        c.is_motion_mode_switchable = ipp.is_motion_mode_switchable;
        c.allow_warped_motion = ipp.allow_warped_motion;
        c.is_scaled_per_ref = ipp.is_scaled_per_ref;
        c.enable_interintra_compound = ipp.enable_interintra_compound;
        c.enable_masked_compound = ipp.enable_masked_compound;
        c.enable_jnt_comp = ipp.enable_jnt_comp;
        c.order_hints = ipp.order_hints;
        c.interpolation_filter = ipp.interpolation_filter;
        c.enable_dual_filter = ipp.enable_dual_filter;
        c
    }

    /// Inter-frame twin of [`syntax_round_trip`].
    fn syntax_round_trip_inter(
        node: &SyntaxNode,
        mi_rows: u32,
        mi_cols: u32,
        b_size: usize,
        params: &SyntaxFrameParams,
        sentinel: u8,
    ) -> (PartitionSyntaxWriter, PartitionWalker) {
        let ipp = params.inter.as_ref().expect("inter params required");
        // ----- encode -----
        let mut writer = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut enc_state =
            PartitionSyntaxWriter::new(mi_rows, mi_cols, single_tile(mi_rows, mi_cols))
                .expect("syntax writer construction");
        write_partition_tree_syntax(
            &mut writer,
            &mut enc_cdfs,
            &mut enc_state,
            node,
            0,
            0,
            b_size,
            params,
        )
        .expect("write_partition_tree_syntax (inter)");
        writer.write_literal(8, u32::from(sentinel)).unwrap();
        let bytes = writer.finish();

        // ----- decode (the §5.11.4 / §5.11.5 / §5.11.18 walker) -----
        let ictx = inter_ctx_for(ipp);
        let mut walker =
            PartitionWalker::new(mi_rows, mi_cols, single_tile(mi_rows, mi_cols)).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        walker
            .decode_partition_syntax(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                b_size,
                /* frame_is_intra = */ false,
                params.subsampling_x,
                params.subsampling_y,
                params.num_planes,
                params.seg_id_pre_skip,
                params.segmentation_enabled,
                params.seg_skip_active,
                params.last_active_seg_id,
                &params.lossless_array,
                params.coded_lossless,
                params.enable_cdef,
                params.allow_intrabc,
                params.cdef_bits,
                params.read_deltas,
                params.use_128x128_superblock,
                params.delta_q_res,
                params.delta_lf_present,
                params.delta_lf_multi,
                params.mono_chrome,
                params.delta_lf_res,
                params.allow_screen_content_tools,
                params.enable_filter_intra,
                params.bit_depth,
                params.tx_mode_select,
                Some(&ictx),
                &params.quant,
                params.reduced_tx_set,
            )
            .expect("decode_partition_syntax must consume the full inter tree");

        assert_eq!(
            dec.read_literal(8).unwrap(),
            u32::from(sentinel),
            "decoder must be positioned at the sentinel after the inter tree"
        );
        assert_eq!(
            enc_cdfs, dec_cdfs,
            "encoder and decoder CDF contexts must adapt identically"
        );
        let m = enc_state.mirror();
        assert_eq!(m.mi_sizes(), walker.mi_sizes(), "MiSizes parity");
        assert_eq!(m.skips(), walker.skips(), "Skips parity");
        assert_eq!(m.segment_ids(), walker.segment_ids(), "SegmentIds parity");
        assert_eq!(m.y_modes(), walker.y_modes(), "YModes parity");
        assert_eq!(m.is_inters(), walker.is_inters(), "IsInters parity");
        assert_eq!(m.ref_frames(), walker.ref_frames(), "RefFrames parity");
        assert_eq!(m.mvs(), walker.mvs(), "Mvs parity");
        assert_eq!(
            m.interp_filters(),
            walker.interp_filters(),
            "InterpFilters parity"
        );
        assert_eq!(
            m.motion_modes(),
            walker.motion_modes(),
            "MotionModes parity"
        );
        assert_eq!(m.cdef_idx(), walker.cdef_idx(), "cdef_idx parity");
        assert_eq!(m.tx_sizes(), walker.tx_sizes(), "TxSizes parity");
        assert_eq!(
            m.inter_tx_sizes(),
            walker.inter_tx_sizes(),
            "InterTxSizes parity"
        );
        assert_eq!(m.tx_types(), walker.tx_types(), "TxTypes parity");
        assert_eq!(
            m.above_level_context(),
            walker.above_level_context(),
            "AboveLevelContext parity"
        );
        assert_eq!(
            m.above_dc_context(),
            walker.above_dc_context(),
            "AboveDcContext parity"
        );
        assert_eq!(
            m.left_level_context(),
            walker.left_level_context(),
            "LeftLevelContext parity"
        );
        assert_eq!(
            m.left_dc_context(),
            walker.left_dc_context(),
            "LeftDcContext parity"
        );
        (enc_state, walker)
    }

    /// Baseline inter-frame parameter bundle: 4:2:0 8-bit, single
    /// reference, everything optional gated shut.
    fn inter_params_420(mi_rows: u32, mi_cols: u32, lossless: bool) -> SyntaxFrameParams {
        let mut params = SyntaxFrameParams::intra_8bit_baseline();
        params.subsampling_x = 1;
        params.subsampling_y = 1;
        params.enable_cdef = false;
        params.coded_lossless = lossless;
        params.lossless_array = [lossless; MAX_SEGMENTS];
        if !lossless {
            params.quant = crate::cdf::QuantizerParams::neutral(100, 8);
        }
        params.inter = Some(SyntaxInterFrameParams::single_ref_baseline(
            mi_rows, mi_cols, /* force_integer_mv = */ false,
        ));
        params
    }

    /// Inter skip leaf helper: LAST_FRAME single-ref with the given
    /// mode / MV.
    fn inter_skip_leaf(y_mode: u8, mv: [i32; 2]) -> SyntaxBlock {
        let mut b = SyntaxBlock::skip_leaf(DC_PRED as u8, None);
        b.inter = Some(SyntaxInterBlock {
            ref_frame: [1, -1],
            y_mode,
            mv: [mv, [0, 0]],
            ref_mv_idx: 0,
            interp_filter: [EIGHTTAP; 2],
            skip_mode: 0,
        });
        b
    }

    /// Four-leaf SPLIT mixing NEWMV / GLOBALMV / a §5.11.22 intra
    /// block / a NEWMV residual leaf on the lossless arm. The 2nd-4th
    /// leaves' §8.3.2 `is_inter` / ref-frame ctx walks and §7.10.2
    /// stacks read the previously stamped mirror grids, so the
    /// sentinel landing proves the full §5.11.18 mirror threading.
    #[test]
    fn r411_inter_syntax_round_trip_newmv_globalmv_intra_mix() {
        let params = inter_params_420(16, 16, /* lossless = */ true);

        // NW: NEWMV, quarter-pel MV, skip = 1.
        let nw = inter_skip_leaf(MODE_NEWMV, [10, -14]);
        // NE: GLOBALMV — identity global motion ⇒ Mv = [0, 0], no MV
        // bits (the §5.11.26 PredMv cross-check must accept it).
        let ne = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);
        // SW: §5.11.22 intra block inside the inter frame (V_PRED with
        // an angle delta so the §5.11.42 arm fires on the y_mode CDF).
        let mut sw = SyntaxBlock::skip_leaf(V_PRED as u8, Some(DC_PRED as u8));
        sw.angle_delta_y = 1;
        // SE: NEWMV with a coded (all-zero) residual — skip = 0 walks
        // the §5.11.34 lossless TX_4X4 fan-out on the inter arm:
        // 8×8 luma TUs + 4×4 chroma TUs per plane.
        let mut se = inter_skip_leaf(MODE_NEWMV, [-8, 24]);
        se.skip = 0;
        se.residual_quant = vec![vec![0i32; 16]; 64 + 16 + 16];

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip_inter(&node, 16, 16, BLOCK_64X64, &params, 0xA7);

        // Spot-check the decoded grids: NW footprint carries the NEWMV
        // vector, NE the zero GLOBALMV, SW is intra.
        let cell = |r: u32, c: u32| (r * 16 + c) as usize;
        assert_eq!(walker.is_inters()[cell(0, 0)], 1);
        assert_eq!(
            [
                walker.mvs()[cell(0, 0) * 4],
                walker.mvs()[cell(0, 0) * 4 + 1]
            ],
            [10i16, -14]
        );
        assert_eq!(walker.is_inters()[cell(0, 8)], 1);
        assert_eq!(
            [
                walker.mvs()[cell(0, 8) * 4],
                walker.mvs()[cell(0, 8) * 4 + 1]
            ],
            [0i16, 0]
        );
        assert_eq!(walker.is_inters()[cell(8, 0)], 0);
        assert_eq!(walker.y_modes()[cell(8, 0)], V_PRED as u8);
        assert_eq!(walker.ref_frames()[cell(8, 8) * 2], 1);
        assert_eq!(walker.ref_frames()[cell(8, 8) * 2 + 1], -1);
    }

    /// r412 — SWITCHABLE frame-filter round trip: four leaves mixing
    /// per-block `interp_filter` choices (EIGHTTAP / SMOOTH / SHARP on
    /// coded-filter leaves, the forced no-bit EIGHTTAP derivation on a
    /// GLOBALMV leaf), replayed bit-for-bit by the §5.11.18 decode
    /// walker. The 4th leaf's §8.3.2 filter ctx walk reads the 2nd/3rd
    /// leaves' stamped `InterpFilters[]` / `RefFrames[]` neighbours,
    /// so the sentinel landing proves the ctx threading, and the
    /// decoded grids prove the per-block filter values.
    #[test]
    fn r412_inter_syntax_round_trip_switchable_filters() {
        let mut params = inter_params_420(16, 16, /* lossless = */ true);
        params
            .inter
            .as_mut()
            .expect("inter params")
            .interpolation_filter = SWITCHABLE;

        use crate::inter_pred::{EIGHTTAP_SHARP, EIGHTTAP_SMOOTH};
        // NW: NEWMV coded with EIGHTTAP_SMOOTH.
        let mut nw = inter_skip_leaf(MODE_NEWMV, [10, -14]);
        nw.inter.as_mut().unwrap().interp_filter = [EIGHTTAP_SMOOTH; 2];
        // NE: GLOBALMV — needs_interp_filter() == 0 on the identity
        // warp, so the pair MUST be the derived EIGHTTAP (no bits).
        let ne = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);
        // SW: NEWMV coded with EIGHTTAP_SHARP.
        let mut sw = inter_skip_leaf(MODE_NEWMV, [-8, 24]);
        sw.inter.as_mut().unwrap().interp_filter = [EIGHTTAP_SHARP; 2];
        // SE: NEWMV coded with EIGHTTAP — its §8.3.2 ctx sees the NE
        // (above) and SW (left) neighbours' stamped filters.
        let se = inter_skip_leaf(MODE_NEWMV, [2, 2]);

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip_inter(&node, 16, 16, BLOCK_64X64, &params, 0x5C);

        let cell = |r: u32, c: u32| (r * 16 + c) as usize;
        let f = walker.interp_filters();
        assert_eq!(
            [f[cell(0, 0) * 2], f[cell(0, 0) * 2 + 1]],
            [EIGHTTAP_SMOOTH; 2],
            "NW leaf filter pair"
        );
        assert_eq!(
            [f[cell(0, 8) * 2], f[cell(0, 8) * 2 + 1]],
            [EIGHTTAP; 2],
            "NE GLOBALMV leaf derives EIGHTTAP with no bits"
        );
        assert_eq!(
            [f[cell(8, 0) * 2], f[cell(8, 0) * 2 + 1]],
            [EIGHTTAP_SHARP; 2],
            "SW leaf filter pair"
        );
        assert_eq!(
            [f[cell(8, 8) * 2], f[cell(8, 8) * 2 + 1]],
            [EIGHTTAP; 2],
            "SE leaf filter pair"
        );
    }

    /// r412 — §5.11.4 PARTITION_HORZ / PARTITION_VERT write dispatch:
    /// a HORZ pair of BLOCK_32X16 inter leaves and a VERT pair of
    /// BLOCK_16X32 leaves inside one SPLIT, replayed bit-for-bit by
    /// the decode walker (rect `MiSizes[]` / `Mvs[]` grid stamps
    /// checked; the second half's §7.10.2 stack and §8.3.2 ctx walks
    /// consume the first half's stamps).
    #[test]
    fn r412_inter_syntax_round_trip_horz_vert_partitions() {
        let params = inter_params_420(16, 16, /* lossless = */ true);

        // NW 32x32 node: HORZ — two 32x16 halves with distinct MVs.
        let top = inter_skip_leaf(MODE_NEWMV, [8, 16]);
        let bottom = inter_skip_leaf(MODE_NEWMV, [-8, -16]);
        // NE 32x32 node: VERT — two 16x32 halves.
        let left = inter_skip_leaf(MODE_NEWMV, [4, -24]);
        let right = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);
        // SW / SE: plain leaves.
        let sw = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);
        let se = SyntaxBlock::skip_leaf(DC_PRED as u8, Some(DC_PRED as u8));

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Horz([Box::new(top), Box::new(bottom)])),
            Box::new(SyntaxNode::Vert([Box::new(left), Box::new(right)])),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip_inter(&node, 16, 16, BLOCK_64X64, &params, 0x9D);

        let cell = |r: u32, c: u32| (r * 16 + c) as usize;
        // HORZ: top half rows 0-3, bottom half rows 4-7 (mi units).
        assert_eq!(walker.mi_sizes()[cell(0, 0)], BLOCK_32X16);
        assert_eq!(walker.mi_sizes()[cell(4, 0)], BLOCK_32X16);
        assert_eq!(
            [
                walker.mvs()[cell(0, 0) * 4],
                walker.mvs()[cell(0, 0) * 4 + 1]
            ],
            [8i16, 16]
        );
        assert_eq!(
            [
                walker.mvs()[cell(4, 0) * 4],
                walker.mvs()[cell(4, 0) * 4 + 1]
            ],
            [-8i16, -16]
        );
        // VERT: left half cols 8-11, right half cols 12-15.
        assert_eq!(walker.mi_sizes()[cell(0, 8)], BLOCK_16X32);
        assert_eq!(walker.mi_sizes()[cell(0, 12)], BLOCK_16X32);
        assert_eq!(
            [
                walker.mvs()[cell(0, 8) * 4],
                walker.mvs()[cell(0, 8) * 4 + 1]
            ],
            [4i16, -24]
        );
        assert_eq!(
            [
                walker.mvs()[cell(0, 12) * 4],
                walker.mvs()[cell(0, 12) * 4 + 1]
            ],
            [0i16, 0]
        );
    }

    /// r412 — COMPOUND_AVERAGE round trip: a SPLIT mixing a
    /// NEW_NEWMV compound leaf (unidirectional { LAST, GOLDEN } pair,
    /// two §5.11.31 MV differences), a NEAREST_NEARESTMV compound
    /// leaf (both lists §5.11.26-derived from the compound §7.10.2
    /// stack the previous leaf seeded), a GLOBAL_GLOBALMV leaf (the
    /// forced-EIGHTTAP `needs_interp_filter( ) == 0` arm), and a
    /// single-ref leaf (whose §5.11.25 cascade now codes the
    /// `comp_mode` bit under `reference_select = 1`) — replayed
    /// bit-for-bit by the decode walker with both MV lists checked
    /// on the grids.
    #[test]
    fn r412_inter_syntax_round_trip_compound_average() {
        let mut params = inter_params_420(16, 16, /* lossless = */ true);
        {
            let ipp = params.inter.as_mut().expect("inter params");
            ipp.interpolation_filter = SWITCHABLE;
            ipp.reference_select = true;
        }
        let comp_leaf = |y_mode: u8, mv: [[i32; 2]; 2], ref_mv_idx: u32| -> SyntaxBlock {
            let mut b = SyntaxBlock::skip_leaf(DC_PRED as u8, None);
            b.inter = Some(SyntaxInterBlock {
                ref_frame: [1, 4],
                y_mode,
                mv,
                ref_mv_idx,
                interp_filter: [EIGHTTAP; 2],
                skip_mode: 0,
            });
            b
        };
        // NW: NEW_NEWMV with two coded MV differences (filter coded
        // under SWITCHABLE).
        let mut nw = comp_leaf(MODE_NEW_NEWMV, [[8, 16], [-4, 24]], 0);
        nw.inter.as_mut().unwrap().interp_filter = [crate::inter_pred::EIGHTTAP_SMOOTH; 2];
        // NE: NEAREST_NEARESTMV — §5.11.26 derives both lists from
        // the §7.10.2 stack; the NW neighbour seeds RefStackMv[0]
        // with ([8, 16], [-4, 24]).
        let ne = comp_leaf(MODE_NEAREST_NEARESTMV, [[8, 16], [-4, 24]], 0);
        // SW: GLOBAL_GLOBALMV — identity warp ⇒ both lists [0, 0],
        // needs_interp_filter() == 0 ⇒ EIGHTTAP, no filter bits.
        let sw = comp_leaf(MODE_GLOBAL_GLOBALMV, [[0, 0], [0, 0]], 0);
        // SE: single-ref GLOBALMV under reference_select = 1 (the
        // §5.11.25 comp_mode bit fires on the SINGLE arm).
        let se = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);

        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip_inter(&node, 16, 16, BLOCK_64X64, &params, 0xE4);

        let cell = |r: u32, c: u32| (r * 16 + c) as usize;
        // NW grids: compound pair + both MV lists.
        assert_eq!(walker.ref_frames()[cell(0, 0) * 2], 1);
        assert_eq!(walker.ref_frames()[cell(0, 0) * 2 + 1], 4);
        assert_eq!(
            [
                walker.mvs()[cell(0, 0) * 4],
                walker.mvs()[cell(0, 0) * 4 + 1],
                walker.mvs()[cell(0, 0) * 4 + 2],
                walker.mvs()[cell(0, 0) * 4 + 3]
            ],
            [8i16, 16, -4, 24]
        );
        let f = walker.interp_filters();
        assert_eq!(
            [f[cell(0, 0) * 2], f[cell(0, 0) * 2 + 1]],
            [crate::inter_pred::EIGHTTAP_SMOOTH; 2]
        );
        // NE: NEAREST_NEARESTMV derived pair equals NW's committed.
        assert_eq!(
            [
                walker.mvs()[cell(0, 8) * 4],
                walker.mvs()[cell(0, 8) * 4 + 1],
                walker.mvs()[cell(0, 8) * 4 + 2],
                walker.mvs()[cell(0, 8) * 4 + 3]
            ],
            [8i16, 16, -4, 24]
        );
        // SW: GLOBAL_GLOBALMV zero pair + forced EIGHTTAP.
        assert_eq!(walker.ref_frames()[cell(8, 0) * 2 + 1], 4);
        assert_eq!(
            [
                walker.mvs()[cell(8, 0) * 4],
                walker.mvs()[cell(8, 0) * 4 + 2]
            ],
            [0i16, 0]
        );
        assert_eq!(f[cell(8, 0) * 2], EIGHTTAP);
        // SE: single-ref under reference_select.
        assert_eq!(walker.ref_frames()[cell(8, 8) * 2], 1);
        assert_eq!(walker.ref_frames()[cell(8, 8) * 2 + 1], -1);
    }

    /// r412 — a committed filter pair inconsistent with the §5.11.x
    /// gates is a caller bug: a GLOBALMV leaf claiming a coded SHARP
    /// filter on the identity-warp configuration must be rejected
    /// (the reader can only derive EIGHTTAP there).
    #[test]
    fn r412_inter_syntax_rejects_filter_on_globalmv_identity_leaf() {
        let mut params = inter_params_420(16, 16, /* lossless = */ true);
        params
            .inter
            .as_mut()
            .expect("inter params")
            .interpolation_filter = SWITCHABLE;
        let mut leaf = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);
        leaf.inter.as_mut().unwrap().interp_filter = [crate::inter_pred::EIGHTTAP_SHARP; 2];

        let mut writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = PartitionSyntaxWriter::new(16, 16, single_tile(16, 16)).unwrap();
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let err = write_partition_tree_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &node,
            0,
            0,
            BLOCK_64X64,
            &params,
        );
        assert!(matches!(err, Err(Error::PartitionWalkOutOfRange)));
    }

    /// NEARESTMV threading: the second leaf's no-bit MV must equal the
    /// §7.10.2 stack head the decoder derives from the first leaf's
    /// stamped grids. The expected vector is pre-computed on a scratch
    /// walker stamped exactly like the write mirror.
    #[test]
    fn r411_inter_syntax_round_trip_nearestmv_from_neighbour_stack() {
        let params = inter_params_420(16, 16, /* lossless = */ true);
        let ipp = params.inter.as_ref().unwrap();

        let nw = inter_skip_leaf(MODE_NEWMV, [16, 8]);

        // Pre-compute NE's §7.10.2 stack against a scratch walker
        // carrying NW's footer stamp.
        let mut scratch = PartitionWalker::new(16, 16, single_tile(16, 16)).unwrap();
        scratch.stamp_encoder_block_syntax(&EncoderBlockSyntaxStamp {
            mi_row: 0,
            mi_col: 0,
            sub_size: BLOCK_32X32,
            skip: 1,
            skip_mode: 0,
            segment_id: 0,
            is_inter: 1,
            y_mode: MODE_NEWMV,
            ref_frame: [1, -1],
            mv: [16, 8],
            mv2: [0, 0],
            interp_filter: [ipp.interpolation_filter; 2],
            motion_mode: crate::cdf::MOTION_MODE_SIMPLE,
            palette_size_y: 0,
            palette_colors_y: &[],
            palette_size_uv: 0,
            palette_colors_u: &[],
            palette_colors_v: &[],
            cdef: None,
            tx_size: TX_4X4 as u8,
        });
        let stack = scratch
            .find_mv_stack(
                0,
                8,
                BLOCK_32X32,
                [1, -1],
                false,
                ipp.use_ref_frame_mvs,
                ipp.gm_type,
                ipp.gm_params,
                ipp.ref_frame_sign_bias,
                ipp.allow_high_precision_mv,
                ipp.force_integer_mv,
                &ipp.motion_field_mvs,
            )
            .unwrap();
        assert!(stack.num_mv_found >= 1, "NW must seed NE's stack");
        let near_mv = stack.ref_stack_mv[0][0];

        let ne = inter_skip_leaf(MODE_NEARESTMV, near_mv);
        let sw = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);
        let se = inter_skip_leaf(MODE_NEWMV, [0, 0]); // zero-diff NEWMV
        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        let (_enc, walker) = syntax_round_trip_inter(&node, 16, 16, BLOCK_64X64, &params, 0x5C);
        let cell = |r: u32, c: u32| (r * 16 + c) as usize;
        assert_eq!(
            [
                i32::from(walker.mvs()[cell(0, 8) * 4]),
                i32::from(walker.mvs()[cell(0, 8) * 4 + 1])
            ],
            near_mv,
            "decoded NEARESTMV must equal the committed stack head"
        );
    }

    /// Lossy `TX_MODE_LARGEST` inter residual: one 32×32 NEWMV leaf
    /// with non-zero DC in every TU (TX_32X32 luma via the §5.11.36
    /// transform-tree write, TX_16X16 chroma), plus a second leaf
    /// whose coefficient ctx walks read the first leaf's §6.10.2
    /// stamps.
    #[test]
    fn r411_inter_syntax_round_trip_lossy_largest_tx_residual() {
        let params = inter_params_420(16, 16, /* lossless = */ false);

        let mut nw = inter_skip_leaf(MODE_NEWMV, [4, 6]);
        nw.skip = 0;
        let mut luma = vec![0i32; 32 * 32];
        luma[0] = 5;
        luma[1] = -2;
        let mut chroma = vec![0i32; 16 * 16];
        chroma[0] = 3;
        nw.residual_quant = vec![luma, chroma.clone(), chroma];

        let mut ne = inter_skip_leaf(MODE_NEWMV, [-4, 2]);
        ne.skip = 0;
        let mut luma2 = vec![0i32; 32 * 32];
        luma2[0] = 1;
        ne.residual_quant = vec![luma2, vec![0i32; 16 * 16], vec![0i32; 16 * 16]];

        let sw = inter_skip_leaf(MODE_GLOBALMV, [0, 0]);
        let se = inter_skip_leaf(MODE_NEWMV, [8, 8]);
        let node = SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ]);
        syntax_round_trip_inter(&node, 16, 16, BLOCK_64X64, &params, 0x3E);
    }

    /// Caller-bug surface: inter commitments on an intra walk, a
    /// compound reference pair, and a GLOBALMV MV that contradicts the
    /// identity global-motion derivation all reject without emitting.
    #[test]
    fn r411_inter_syntax_scope_rejects() {
        // (1) inter leaf on an intra-frame walk.
        let params_intra = SyntaxFrameParams::intra_8bit_baseline();
        let mut writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut state = PartitionSyntaxWriter::new(16, 16, single_tile(16, 16)).unwrap();
        let block = inter_skip_leaf(MODE_NEWMV, [0, 0]);
        let err = super::write_block_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &block,
            0,
            0,
            BLOCK_32X32,
            &params_intra,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // (2) compound pair on the r411 single-ref scope.
        let params = inter_params_420(16, 16, true);
        let mut b2 = inter_skip_leaf(MODE_NEWMV, [0, 0]);
        b2.inter.as_mut().unwrap().ref_frame = [1, 7];
        let mut writer2 = SymbolWriter::new(false);
        let mut cdfs2 = TileCdfContext::new_from_defaults();
        let mut state2 = PartitionSyntaxWriter::new(16, 16, single_tile(16, 16)).unwrap();
        let err2 = super::write_block_syntax(
            &mut writer2,
            &mut cdfs2,
            &mut state2,
            &b2,
            0,
            0,
            BLOCK_32X32,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err2, Error::PartitionWalkOutOfRange));

        // (3) GLOBALMV whose committed MV contradicts the identity
        // global-motion derivation (the reader would reconstruct with
        // [0, 0]).
        let b3 = inter_skip_leaf(MODE_GLOBALMV, [8, 0]);
        let mut writer3 = SymbolWriter::new(false);
        let mut cdfs3 = TileCdfContext::new_from_defaults();
        let mut state3 = PartitionSyntaxWriter::new(16, 16, single_tile(16, 16)).unwrap();
        let err3 = super::write_block_syntax(
            &mut writer3,
            &mut cdfs3,
            &mut state3,
            &b3,
            0,
            0,
            BLOCK_32X32,
            &params,
        )
        .unwrap_err();
        assert!(matches!(err3, Error::PartitionWalkOutOfRange));
    }
}
