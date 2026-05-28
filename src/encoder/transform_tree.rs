//! §5.11.36 `transform_tree` / §5.11.16 `read_block_tx_size` /
//! §5.11.15 `read_tx_size` / §5.11.17 `read_var_tx_size` **writer** —
//! the encoder counterpart of the matching readers in
//! [`crate::cdf::PartitionWalker::read_block_tx_size`] and
//! [`crate::cdf::PartitionWalker::read_var_tx_size`].
//!
//! Scope of this arc (r218): the *symbol-emission* portion of the
//! per-block transform-size syntax tree. The decoder body reads at
//! most one `tx_depth S()` (the §5.11.15 path) **OR** a recursive
//! `txfm_split S()` chain (the §5.11.17 variable-transform path) per
//! block; this module supplies the inverse for both.
//!
//! ## Two arms, same dispatch as §5.11.16
//!
//! The §5.11.16 reader chooses between two arms (av1-spec p.70):
//!
//! * `TX_MODE_SELECT && MiSize > BLOCK_4X4 && is_inter && !skip &&
//!   !Lossless` — enters the §5.11.17 `read_var_tx_size` recursion
//!   across the `(txH4, txW4)` sub-rectangles of the block footprint.
//!   Each sub-rectangle's recursion emits zero or more `txfm_split`
//!   `S()` bits (one per non-terminal node) and stamps the leaf
//!   `txSz` into the encoder's local `InterTxSizes[]` mirror.
//!
//! * Otherwise — the §5.11.15 `read_tx_size(allowSelect)` body. When
//!   `Lossless` the spec forces `TxSize = TX_4X4` with no symbol read;
//!   when `MiSize > BLOCK_4X4 && allowSelect && TxMode ==
//!   TX_MODE_SELECT` the body emits a single `tx_depth S()`. Otherwise
//!   no symbol fires (the decoder picks `TxSize = maxRectTxSize`).
//!
//! ## Surface
//!
//! Two stateless writers, mirroring the decoder's two methods:
//!
//! * [`write_block_tx_size`] — the §5.11.16 outer dispatch +
//!   §5.11.15 `tx_depth` symbol. Given the chosen `tx_size`, derives
//!   the §5.11.15 `tx_depth` value via repeated `Split_Tx_Size`
//!   lookup against `Max_Tx_Size_Rect[MiSize]`, then emits the right
//!   §8.3.2 `Tile{Tx8x8,Tx16x16,Tx32x32,Tx64x64}Cdf[ctx]` symbol.
//!
//! * [`write_var_tx_size`] — the §5.11.17 recursion. Given the
//!   `(root_tx_size, leaf_tx_sizes)` description of the desired
//!   variable-transform tree, emits one `txfm_split S()` per
//!   non-terminal node (any node where the chosen split chose to
//!   descend) and recurses on the four sub-tx-sizes per
//!   `Split_Tx_Size`.
//!
//! ## Ctx threading
//!
//! Like every other writer in this arc, both functions take the §8.3.2
//! ctx values as caller-supplied arguments (the encode driver maintains
//! its own neighbour grids in parallel with the decoder's
//! `PartitionWalker`, so the booleans + ctxs are derivable locally and
//! cheaply). The matching ctx helpers
//! [`crate::cdf::tx_depth_ctx`] / [`crate::cdf::txfm_split_ctx`] are
//! already public and re-used by the decoder; encoders can call them
//! directly with the same inputs.
//!
//! ## Out of scope this arc
//!
//! The §5.11.16 dispatch itself (the `if ( TX_MODE_SELECT && ... &&
//! !Lossless ) { ... } else { ... }` branch + the outer grid-fill
//! stamps) is *not* re-implemented here — it's owned by the encode
//! driver, which has the full `(mi_row, mi_col, sub_size,
//! is_inter, skip, lossless, tx_mode_select)` in scope at the
//! `decode_block` call site and threads the chosen `tx_size` /
//! `(root_tx_size, leaves)` into this module.
//!
//! Same separation as [`crate::encoder::partition`] (symbol-only;
//! [`crate::encoder::partition_tree`] owns the dispatch).
//!
//! ## Spec provenance
//!
//! Sourced from `docs/video/av1/av1-spec.txt`:
//!   * §5.11.15 `read_tx_size`             (p.69)
//!   * §5.11.16 `read_block_tx_size`       (p.70)
//!   * §5.11.17 `read_var_tx_size`         (p.70)
//!   * §8.3.2   `tx_depth` / `txfm_split` ctx + CDF selection (p.363-364, p.376)
//!   * §3       `MAX_VARTX_DEPTH = 2`      (p.7)

use crate::cdf::{
    TileCdfContext, BLOCK_4X4, BLOCK_SIZES, MAX_TX_DEPTH, MAX_TX_DEPTH_TABLE, MAX_TX_SIZE_RECT,
    MAX_VARTX_DEPTH, MI_SIZE, SPLIT_TX_SIZE, TXFM_PARTITION_CONTEXTS, TX_4X4, TX_HEIGHT,
    TX_SIZES_ALL, TX_SIZE_CONTEXTS, TX_WIDTH,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::Error;

/// `read_block_tx_size` / `read_tx_size` **writer** — emits the
/// §5.11.15 `tx_depth` symbol for the §5.11.16 `else` arm.
///
/// This is the encoder counterpart of the `else` branch in
/// [`crate::cdf::PartitionWalker::read_block_tx_size`]. Use it when
/// the encode driver picked the non-variable-transform arm (i.e. the
/// §5.11.16 outer guard `TX_MODE_SELECT && MiSize > BLOCK_4X4 &&
/// is_inter && !skip && !Lossless` is FALSE). For the variable-
/// transform arm use [`write_var_tx_size`] across each
/// `(txH4, txW4)` sub-rectangle instead.
///
/// ## Inverse derivation
///
/// The §5.11.15 body the decoder runs is:
///
/// ```text
///   if ( Lossless ) { TxSize = TX_4X4; return }
///   maxRectTxSize = Max_Tx_Size_Rect[ MiSize ]
///   maxTxDepth    = Max_Tx_Depth[ MiSize ]
///   TxSize        = maxRectTxSize
///   if ( MiSize > BLOCK_4X4 && allowSelect && TxMode == TX_MODE_SELECT ) {
///       tx_depth                                                  S()
///       for ( i = 0; i < tx_depth; i++ ) TxSize = Split_Tx_Size[ TxSize ]
///   }
/// ```
///
/// Given the chosen `tx_size`, the writer derives `tx_depth` by walking
/// `Split_Tx_Size` repeatedly from `maxRectTxSize` and counting steps
/// until the chain hits `tx_size`. The `Default_Tx_*Cdf` row length
/// caps `tx_depth` at [`MAX_TX_DEPTH`] (= `2`); any `tx_size`
/// unreachable from `maxRectTxSize` within `MAX_TX_DEPTH` splits is
/// rejected as a caller bug.
///
/// ## Caller-supplied state
///
/// * `tx_size` — the chosen §6.10.16 `TxSize` ordinal in
///   `0..TX_SIZES_ALL`. The writer derives `tx_depth` to match.
/// * `sub_size` — the §5.11.5 `MiSize` ordinal in `0..BLOCK_SIZES`.
/// * `lossless` — `LosslessArray[ segment_id ]`. When `true` the
///   writer emits nothing (the decoder skips the symbol and forces
///   `TxSize = TX_4X4`); `tx_size` MUST be `TX_4X4`.
/// * `allow_select` — the §5.11.16 caller's `!skip || !is_inter`. For
///   the intra arm this is always `true`; for the inter `skip == 1`
///   arm this is also `true`.
/// * `tx_mode_select` — `true` ⇔ §5.9.21 / §6.8.21 `TxMode ==
///   TX_MODE_SELECT`. When `false` the decoder doesn't read the symbol;
///   `tx_size` MUST be `maxRectTxSize` (the §5.11.15 default).
/// * `ctx` — the §8.3.2 `tx_depth` ctx in `0..TX_SIZE_CONTEXTS`,
///   derived by the encode driver via
///   [`crate::cdf::tx_depth_ctx`].
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any caller-supplied
/// inconsistency: an out-of-range `tx_size` / `sub_size` / `ctx`, a
/// `tx_size` that doesn't match the spec-forced value on the no-symbol
/// arms (Lossless / sub_size == BLOCK_4X4 / !allow_select / !tx_mode_select),
/// or a `tx_size` unreachable from `maxRectTxSize` within
/// `MAX_TX_DEPTH` splits.
#[allow(clippy::too_many_arguments)]
pub fn write_block_tx_size(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    tx_size: usize,
    sub_size: usize,
    lossless: bool,
    allow_select: bool,
    tx_mode_select: bool,
    ctx: usize,
) -> Result<(), Error> {
    // ---------------- caller-bug guards ----------------
    if sub_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_size >= TX_SIZES_ALL {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ctx >= TX_SIZE_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.15 first line: `if ( Lossless ) { TxSize = TX_4X4; return }`.
    // No symbol is read in the decoder; the writer mirrors that and
    // asserts the caller's `tx_size` matches the spec-forced value.
    if lossless {
        if tx_size != TX_4X4 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.15 lines 2-3: `maxRectTxSize = Max_Tx_Size_Rect[ MiSize ]`,
    // `maxTxDepth = Max_Tx_Depth[ MiSize ]`. The §5.11.15 default
    // (when the `tx_depth` symbol does not fire) is `TxSize =
    // maxRectTxSize`.
    let max_rect_tx_size = MAX_TX_SIZE_RECT[sub_size];
    let max_tx_depth = MAX_TX_DEPTH_TABLE[sub_size];

    // §5.11.15 line 6 guard: `MiSize > BLOCK_4X4 && allowSelect &&
    // TxMode == TX_MODE_SELECT`. When FALSE the decoder reads no
    // symbol and `TxSize = maxRectTxSize`.
    if !(sub_size > BLOCK_4X4 && allow_select && tx_mode_select) {
        if tx_size != max_rect_tx_size {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.15 line 7-8 inverse: walk `Split_Tx_Size` from
    // `maxRectTxSize` and count steps until reaching `tx_size`. The
    // §5.11.15 spec note bounds `tx_depth` to `0..=MAX_TX_DEPTH = 0..=2`
    // (the `Default_Tx_*Cdf` row length caps the S() output).
    let mut depth: u32 = 0;
    let mut walker = max_rect_tx_size;
    while walker != tx_size {
        if depth as usize >= MAX_TX_DEPTH {
            // The §5.11.15 reader caps `tx_depth` at MAX_TX_DEPTH;
            // a deeper-required `tx_size` is unreachable from the
            // syntax and surfaces as a caller bug.
            return Err(Error::PartitionWalkOutOfRange);
        }
        let next = SPLIT_TX_SIZE[walker];
        if next == walker {
            // `Split_Tx_Size[TX_4X4] == TX_4X4` is the bottom of the
            // chain; if we haven't matched `tx_size` by then it's
            // unreachable.
            return Err(Error::PartitionWalkOutOfRange);
        }
        walker = next;
        depth += 1;
    }

    // §8.3.2 CDF selection per `max_tx_depth`. `MiSize > BLOCK_4X4`
    // already excludes the only `max_tx_depth == 0` row of
    // [`MAX_TX_DEPTH_TABLE`] (BLOCK_4X4 alone), so the selector returns
    // `Some(_)`.
    let cdf = cdfs
        .tx_depth_cdf(max_tx_depth, ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(depth, cdf)
}

/// `read_var_tx_size` **writer** — emits the §5.11.17 `txfm_split`
/// recursion for one `(txH4, txW4)` sub-rectangle of a §5.11.16
/// variable-transform-arm block.
///
/// This is the encoder counterpart of
/// [`crate::cdf::PartitionWalker::read_var_tx_size`]. Use it when
/// the encode driver picked the §5.11.16 outer `TX_MODE_SELECT &&
/// MiSize > BLOCK_4X4 && is_inter && !skip && !Lossless` arm and is
/// emitting one sub-rectangle of the block's
/// `(MiRow .. MiRow + bh4, MiCol .. MiCol + bw4)` footprint.
///
/// ## Inverse derivation
///
/// The §5.11.17 reader recursion is:
///
/// ```text
///   read_var_tx_size( row, col, txSz, depth ) {
///       if ( row >= MiRows || col >= MiCols ) return
///       if ( txSz == TX_4X4 || depth == MAX_VARTX_DEPTH ) txfm_split = 0
///       else                                              txfm_split S()
///       if ( txfm_split ) recurse on Split_Tx_Size[txSz]
///       else              stamp InterTxSizes[..] = txSz
///   }
/// ```
///
/// The writer receives the desired leaf-tx-size description as a
/// caller-supplied [`VarTxNode`] tree: each non-leaf carries one
/// `txfm_split = 1` decision plus four children (one per quadrant of
/// the `Split_Tx_Size[txSz]` subdivision); each leaf forces
/// `txfm_split = 0` and ends the recursion.
///
/// ## Ctx threading
///
/// Like every other writer in this arc, the §8.3.2 `txfm_split` ctx is
/// caller-supplied per node. The driver derives it via
/// [`crate::cdf::txfm_split_ctx`] using the same inputs the decoder
/// uses (`above_w < TX_WIDTH[txSz]`, `left_h < TX_HEIGHT[txSz]`,
/// `Tx_Size_Sqr_Up[txSz]`, `find_tx_size(size, size)`).
///
/// ## Caller-supplied state
///
/// * `root` — the [`VarTxNode`] tree rooted at this sub-rectangle.
///   `root.tx_size` is the §5.11.17 `txSz` argument (= `maxTxSz` =
///   `Max_Tx_Size_Rect[MiSize]` for the §5.11.16 outer call,
///   matching the reader's `read_var_tx_size( ..., maxTxSz, 0 )`).
/// * The recursion depth starts at `0`; the writer enforces the
///   §3 `MAX_VARTX_DEPTH = 2` bound matching the reader.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any caller-supplied
/// tree malformation: a leaf at a non-terminal node, an internal
/// `Split` at the §5.11.17 terminal `txSz == TX_4X4 || depth ==
/// MAX_VARTX_DEPTH` node, an out-of-range `tx_size` / `ctx`, or a
/// `Split` whose child count doesn't match the `Split_Tx_Size`
/// quadrant decomposition.
pub fn write_var_tx_size(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    root: &VarTxNode,
) -> Result<(), Error> {
    write_var_tx_size_recursive(writer, cdfs, root, 0)
}

/// Internal recursive driver — does not re-validate `depth` between
/// calls (the caller-facing entry-point starts at `0` and the
/// recursion itself increments).
fn write_var_tx_size_recursive(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    node: &VarTxNode,
    depth: u32,
) -> Result<(), Error> {
    if node.tx_size >= TX_SIZES_ALL {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if depth > MAX_VARTX_DEPTH {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let terminal = node.tx_size == TX_4X4 || depth == MAX_VARTX_DEPTH;

    match &node.kind {
        VarTxNodeKind::Leaf => {
            // §5.11.17 terminal arm: `txfm_split = 0`. Emits a symbol
            // only when the node is NOT at the spec-forced terminal
            // (otherwise no S() fires on the reader side).
            if !terminal {
                if node.ctx >= TXFM_PARTITION_CONTEXTS {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                let cdf = cdfs.txfm_split_cdf(node.ctx);
                writer.write_symbol(0, cdf)?;
            }
            Ok(())
        }
        VarTxNodeKind::Split(children) => {
            // The §5.11.17 spec forbids a split at the terminal `txSz
            // == TX_4X4 || depth == MAX_VARTX_DEPTH` node — the
            // reader's `if ( ... ) txfm_split = 0` line short-circuits
            // any further descent. Reject the caller bug.
            if terminal {
                return Err(Error::PartitionWalkOutOfRange);
            }

            // §5.11.17 `txfm_split S()`. Emit the `1` symbol against
            // the caller-supplied ctx.
            if node.ctx >= TXFM_PARTITION_CONTEXTS {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let cdf = cdfs.txfm_split_cdf(node.ctx);
            writer.write_symbol(1, cdf)?;

            // §5.11.17 lines 11-16: recursion. `subTxSz =
            // Split_Tx_Size[txSz]`; the recursion increments depth.
            // The spec body loops `i in 0..h4 step stepH` /
            // `j in 0..w4 step stepW`, so the number of children is
            // `(h4 / stepH) * (w4 / stepW)`.
            let sub_tx_sz = SPLIT_TX_SIZE[node.tx_size];
            let w4 = TX_WIDTH[node.tx_size] / MI_SIZE;
            let h4 = TX_HEIGHT[node.tx_size] / MI_SIZE;
            let step_w = TX_WIDTH[sub_tx_sz] / MI_SIZE;
            let step_h = TX_HEIGHT[sub_tx_sz] / MI_SIZE;
            if step_w == 0 || step_h == 0 {
                // Defensive: `Split_Tx_Size` always yields a non-zero
                // step for any `txSz >= TX_4X4` reachable by the
                // recursion; surface a caller bug otherwise.
                return Err(Error::PartitionWalkOutOfRange);
            }
            let expected_children = (h4 / step_h) * (w4 / step_w);
            if children.len() != expected_children {
                return Err(Error::PartitionWalkOutOfRange);
            }

            for child in children {
                // Sub-children must carry the `subTxSz`.
                if child.tx_size != sub_tx_sz {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                write_var_tx_size_recursive(writer, cdfs, child, depth + 1)?;
            }
            Ok(())
        }
    }
}

/// Caller-supplied description of one node of the §5.11.17
/// `read_var_tx_size` recursion tree.
///
/// Each node carries:
///
/// * `tx_size` — the §5.11.17 `txSz` argument the reader sees at this
///   recursion level. Root nodes carry `Max_Tx_Size_Rect[ MiSize ]`
///   (the §5.11.16 outer caller's `maxTxSz`); sub-nodes carry
///   `Split_Tx_Size[parent.tx_size]`.
///
/// * `ctx` — the §8.3.2 `txfm_split` ctx derived by the encode driver
///   for THIS node (via [`crate::cdf::txfm_split_ctx`]). Ignored for
///   terminal-arm leaves (where the reader emits no S() and so the
///   writer emits none either).
///
/// * `kind` — [`VarTxNodeKind::Leaf`] (terminate the recursion;
///   `txfm_split = 0`) or [`VarTxNodeKind::Split`] (descend; `txfm_split
///   = 1` plus four child nodes per the `Split_Tx_Size` quadrant
///   decomposition).
#[derive(Debug, Clone)]
pub struct VarTxNode {
    pub tx_size: usize,
    pub ctx: usize,
    pub kind: VarTxNodeKind,
}

/// Whether a [`VarTxNode`] terminates (Leaf, `txfm_split = 0`) or
/// descends (Split, `txfm_split = 1` + children).
#[derive(Debug, Clone)]
pub enum VarTxNodeKind {
    /// §5.11.17 `else` arm: `InterTxSizes[..] = txSz`; no further
    /// recursion. Equivalent to `txfm_split = 0` on the reader side.
    /// At the spec-forced terminal (`txSz == TX_4X4` or `depth ==
    /// MAX_VARTX_DEPTH`) the writer also takes this kind but emits no
    /// symbol.
    Leaf,
    /// §5.11.17 `if (txfm_split)` arm: recurse on `Split_Tx_Size[txSz]`.
    /// The number of children must match `(h4 / stepH) * (w4 / stepW)`.
    /// For square `TX_NxN` ordinals this is always `4`; for rectangular
    /// ordinals it can be `2` (one axis already at the smaller-edge
    /// minimum after one split).
    Split(Vec<VarTxNode>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{BLOCK_32X32, BLOCK_64X64, BLOCK_8X8, TX_16X16, TX_32X32, TX_64X64, TX_8X8};
    use crate::symbol_decoder::SymbolDecoder;

    /// Bridge helper: build a fresh `(SymbolWriter, TileCdfContext)`,
    /// run `f`, finish, replay the bytes through a `SymbolDecoder`
    /// against a parallel `TileCdfContext` running the matching
    /// `read_symbol` calls — returning the recovered `tx_depth` value.
    fn roundtrip_tx_depth(
        tx_size: usize,
        sub_size: usize,
        lossless: bool,
        allow_select: bool,
        tx_mode_select: bool,
        ctx: usize,
    ) -> u32 {
        let mut w = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        write_block_tx_size(
            &mut w,
            &mut enc_cdfs,
            tx_size,
            sub_size,
            lossless,
            allow_select,
            tx_mode_select,
            ctx,
        )
        .unwrap();
        let bytes = w.finish();
        // Pad to at least one byte so `init_symbol` can read its
        // 15-bit window.
        let bytes = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        // The decoder-side §5.11.15 `tx_depth` symbol read fires only
        // when `MiSize > BLOCK_4X4 && allowSelect && TxMode ==
        // TX_MODE_SELECT && !Lossless`; otherwise no symbol is read.
        if !lossless && sub_size > BLOCK_4X4 && allow_select && tx_mode_select {
            let max_tx_depth = MAX_TX_DEPTH_TABLE[sub_size];
            let cdf = dec_cdfs.tx_depth_cdf(max_tx_depth, ctx).unwrap();
            d.read_symbol(cdf).unwrap()
        } else {
            0
        }
    }

    /// `tx_size == max_rect_tx_size` on a sub-BLOCK_4X4 (here:
    /// BLOCK_8X8) block where the §5.11.15 guard fires emits
    /// `tx_depth = 0` and round-trips.
    #[test]
    fn write_block_tx_size_emits_zero_for_max_rect() {
        let depth = roundtrip_tx_depth(TX_8X8, BLOCK_8X8, false, true, true, 0);
        assert_eq!(depth, 0);
    }

    /// `tx_size == Split_Tx_Size[max_rect_tx_size]` on BLOCK_8X8 emits
    /// `tx_depth = 1`.
    #[test]
    fn write_block_tx_size_emits_one_for_one_split() {
        // BLOCK_8X8 ⇒ max_rect = TX_8X8 ⇒ Split_Tx_Size[TX_8X8] = TX_4X4.
        let depth = roundtrip_tx_depth(TX_4X4, BLOCK_8X8, false, true, true, 1);
        assert_eq!(depth, 1);
    }

    /// `tx_size` two splits down from `max_rect_tx_size` emits
    /// `tx_depth = 2` (the `Default_Tx_*Cdf` row length cap).
    #[test]
    fn write_block_tx_size_emits_two_for_two_splits() {
        // BLOCK_32X32 ⇒ max_rect = TX_32X32 ⇒ Split[TX_32X32] = TX_16X16
        // ⇒ Split[TX_16X16] = TX_8X8. `tx_depth = 2`.
        let depth = roundtrip_tx_depth(TX_8X8, BLOCK_32X32, false, true, true, 2);
        assert_eq!(depth, 2);
    }

    /// On the Lossless arm the writer emits NO symbol and the
    /// caller's `tx_size` MUST be TX_4X4.
    #[test]
    fn write_block_tx_size_lossless_emits_no_symbol() {
        let depth = roundtrip_tx_depth(TX_4X4, BLOCK_32X32, true, true, true, 0);
        assert_eq!(depth, 0);
    }

    /// On the `!tx_mode_select` arm the writer emits NO symbol and
    /// the caller's `tx_size` MUST be `max_rect_tx_size`.
    #[test]
    fn write_block_tx_size_no_select_emits_no_symbol() {
        let depth = roundtrip_tx_depth(TX_64X64, BLOCK_64X64, false, true, false, 0);
        assert_eq!(depth, 0);
    }

    /// `Lossless` with a non-TX_4X4 `tx_size` is a caller bug.
    #[test]
    fn write_block_tx_size_lossless_rejects_non_4x4() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_block_tx_size(&mut w, &mut cdfs, TX_8X8, BLOCK_32X32, true, true, true, 0)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `!tx_mode_select` with a `tx_size != max_rect_tx_size` is a
    /// caller bug — the decoder's §5.11.15 default forces
    /// `TxSize = maxRectTxSize`.
    #[test]
    fn write_block_tx_size_no_select_rejects_non_max() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_block_tx_size(
            &mut w,
            &mut cdfs,
            TX_8X8,
            BLOCK_32X32,
            false,
            true,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// A `tx_size` unreachable from `max_rect_tx_size` within
    /// `MAX_TX_DEPTH` splits is a caller bug. BLOCK_8X8 has
    /// `max_rect = TX_8X8` and the chain TX_8X8 → TX_4X4 stops there
    /// at depth 1; asking for TX_64X64 (which is larger than
    /// TX_8X8 and not in the chain) is rejected.
    #[test]
    fn write_block_tx_size_rejects_unreachable() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_block_tx_size(&mut w, &mut cdfs, TX_64X64, BLOCK_8X8, false, true, true, 0)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `sub_size` is a caller bug.
    #[test]
    fn write_block_tx_size_rejects_out_of_range_sub_size() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_block_tx_size(&mut w, &mut cdfs, TX_4X4, BLOCK_SIZES, false, true, true, 0)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `tx_size` is a caller bug.
    #[test]
    fn write_block_tx_size_rejects_out_of_range_tx_size() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_block_tx_size(
            &mut w,
            &mut cdfs,
            TX_SIZES_ALL,
            BLOCK_8X8,
            false,
            true,
            true,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `ctx` is a caller bug.
    #[test]
    fn write_block_tx_size_rejects_out_of_range_ctx() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_block_tx_size(
            &mut w,
            &mut cdfs,
            TX_8X8,
            BLOCK_8X8,
            false,
            true,
            true,
            TX_SIZE_CONTEXTS,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Bridge helper for the §5.11.17 var-tx recursion: emit a
    /// caller-supplied [`VarTxNode`] tree, finish, then replay the
    /// bytes through a parallel `SymbolDecoder` running the matching
    /// recursive `txfm_split S()` reads. Returns the sequence of
    /// `txfm_split` bits the decoder consumed (one entry per
    /// non-terminal node visited in spec-recursion order).
    fn roundtrip_var_tx(root: &VarTxNode) -> Vec<u32> {
        let mut w = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        write_var_tx_size(&mut w, &mut enc_cdfs, root).unwrap();
        let bytes = w.finish();
        let bytes = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut splits = Vec::new();
        decode_var_tx_replay(&mut d, &mut dec_cdfs, root, 0, &mut splits);
        splits
    }

    /// Decoder-side §5.11.17 replay: walks the same tree shape the
    /// writer emitted and consumes one `txfm_split S()` per non-
    /// terminal node, in matching spec-recursion order. This is the
    /// minimum the writer needs to roundtrip against — the decoder's
    /// full method needs frame / tile geometry which we don't have in
    /// a unit test.
    fn decode_var_tx_replay(
        d: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        node: &VarTxNode,
        depth: u32,
        out: &mut Vec<u32>,
    ) {
        let terminal = node.tx_size == TX_4X4 || depth == MAX_VARTX_DEPTH;
        match &node.kind {
            VarTxNodeKind::Leaf => {
                if !terminal {
                    let cdf = cdfs.txfm_split_cdf(node.ctx);
                    let s = d.read_symbol(cdf).unwrap();
                    out.push(s);
                    assert_eq!(s, 0, "leaf must round-trip txfm_split = 0");
                }
            }
            VarTxNodeKind::Split(children) => {
                let cdf = cdfs.txfm_split_cdf(node.ctx);
                let s = d.read_symbol(cdf).unwrap();
                out.push(s);
                assert_eq!(s, 1, "split must round-trip txfm_split = 1");
                for child in children {
                    decode_var_tx_replay(d, cdfs, child, depth + 1, out);
                }
            }
        }
    }

    /// A single-leaf root at TX_16X16 (non-terminal) emits one
    /// `txfm_split = 0` symbol; decoder recovers `0`.
    #[test]
    fn var_tx_size_leaf_emits_zero() {
        let root = VarTxNode {
            tx_size: TX_16X16,
            ctx: 3,
            kind: VarTxNodeKind::Leaf,
        };
        let splits = roundtrip_var_tx(&root);
        assert_eq!(splits, vec![0]);
    }

    /// A leaf at the spec-forced terminal (`tx_size == TX_4X4`) emits
    /// NO symbol (decoder reads nothing too).
    #[test]
    fn var_tx_size_terminal_4x4_emits_no_symbol() {
        let root = VarTxNode {
            tx_size: TX_4X4,
            ctx: 0,
            kind: VarTxNodeKind::Leaf,
        };
        let splits = roundtrip_var_tx(&root);
        assert!(splits.is_empty());
    }

    /// A split at TX_16X16 → 4× TX_8X8 leaves emits `1` for the root
    /// split followed by four leaf `0`s (one per child since none of
    /// the children is at the spec-forced terminal yet — TX_8X8 is
    /// not TX_4X4 and depth = 1 < MAX_VARTX_DEPTH).
    #[test]
    fn var_tx_size_split_then_leaves() {
        let leaf_child = |ctx| VarTxNode {
            tx_size: TX_8X8,
            ctx,
            kind: VarTxNodeKind::Leaf,
        };
        let root = VarTxNode {
            tx_size: TX_16X16,
            ctx: 5,
            kind: VarTxNodeKind::Split(vec![
                leaf_child(0),
                leaf_child(1),
                leaf_child(2),
                leaf_child(3),
            ]),
        };
        let splits = roundtrip_var_tx(&root);
        assert_eq!(splits, vec![1, 0, 0, 0, 0]);
    }

    /// Two-level split: TX_32X32 → 4× TX_16X16, each of which splits
    /// to 4× TX_8X8 leaves. depth-2 (MAX_VARTX_DEPTH) leaves at
    /// TX_8X8 emit no symbol (spec-forced terminal).
    #[test]
    fn var_tx_size_two_level_split() {
        let leaf_8x8 = |ctx| VarTxNode {
            tx_size: TX_8X8,
            ctx,
            kind: VarTxNodeKind::Leaf,
        };
        let inner_16x16 = |ctx| VarTxNode {
            tx_size: TX_16X16,
            ctx,
            kind: VarTxNodeKind::Split(vec![leaf_8x8(0), leaf_8x8(0), leaf_8x8(0), leaf_8x8(0)]),
        };
        let root = VarTxNode {
            tx_size: TX_32X32,
            ctx: 7,
            kind: VarTxNodeKind::Split(vec![
                inner_16x16(1),
                inner_16x16(2),
                inner_16x16(3),
                inner_16x16(4),
            ]),
        };
        let splits = roundtrip_var_tx(&root);
        // Root split (1) + four inner splits (1 each).
        // Per-inner: 1 + four depth-2 leaves (each at TX_8X8 with
        // depth == MAX_VARTX_DEPTH, so terminal — no symbol).
        // Total = 1 + 4 * 1 = 5 ones, no zeros.
        assert_eq!(splits, vec![1, 1, 1, 1, 1]);
    }

    /// A `Split` at the spec-forced terminal `tx_size == TX_4X4` node
    /// is a caller bug (the reader can't read `txfm_split` there).
    #[test]
    fn var_tx_size_split_at_4x4_terminal_rejected() {
        let root = VarTxNode {
            tx_size: TX_4X4,
            ctx: 0,
            kind: VarTxNodeKind::Split(vec![VarTxNode {
                tx_size: TX_4X4,
                ctx: 0,
                kind: VarTxNodeKind::Leaf,
            }]),
        };
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_var_tx_size(&mut w, &mut cdfs, &root).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// A `Split` at `depth == MAX_VARTX_DEPTH` (achieved on a two-
    /// level walk with all-splits) is a caller bug — the spec forces
    /// `txfm_split = 0` at that depth.
    #[test]
    fn var_tx_size_split_at_max_depth_rejected() {
        // Build TX_64X64 → 4× TX_32X32 (depth 1) → 4× TX_16X16 (depth
        // 2, MAX_VARTX_DEPTH) → 4× TX_8X8. The depth-2 split is
        // illegal.
        let illegal_inner = VarTxNode {
            tx_size: TX_16X16,
            ctx: 0,
            kind: VarTxNodeKind::Split(vec![
                VarTxNode {
                    tx_size: TX_8X8,
                    ctx: 0,
                    kind: VarTxNodeKind::Leaf,
                };
                4
            ]),
        };
        let mid_32 = VarTxNode {
            tx_size: TX_32X32,
            ctx: 0,
            kind: VarTxNodeKind::Split(vec![illegal_inner; 4]),
        };
        let root = VarTxNode {
            tx_size: TX_64X64,
            ctx: 0,
            kind: VarTxNodeKind::Split(vec![mid_32; 4]),
        };
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_var_tx_size(&mut w, &mut cdfs, &root).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// A `Split` whose child count doesn't match the §5.11.17 quadrant
    /// decomposition is a caller bug.
    #[test]
    fn var_tx_size_wrong_child_count_rejected() {
        // TX_16X16 splits into 4× TX_8X8 — supplying 3 children is
        // wrong.
        let root = VarTxNode {
            tx_size: TX_16X16,
            ctx: 0,
            kind: VarTxNodeKind::Split(vec![
                VarTxNode {
                    tx_size: TX_8X8,
                    ctx: 0,
                    kind: VarTxNodeKind::Leaf,
                };
                3
            ]),
        };
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_var_tx_size(&mut w, &mut cdfs, &root).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Child tx_size that doesn't match `Split_Tx_Size[parent]` is a
    /// caller bug.
    #[test]
    fn var_tx_size_wrong_child_size_rejected() {
        let root = VarTxNode {
            tx_size: TX_16X16,
            ctx: 0,
            kind: VarTxNodeKind::Split(vec![
                VarTxNode {
                    // Should be TX_8X8 (Split_Tx_Size[TX_16X16]); pass
                    // TX_4X4 instead.
                    tx_size: TX_4X4,
                    ctx: 0,
                    kind: VarTxNodeKind::Leaf,
                };
                4
            ]),
        };
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_var_tx_size(&mut w, &mut cdfs, &root).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `tx_size` on a node is a caller bug.
    #[test]
    fn var_tx_size_rejects_out_of_range_tx_size() {
        let root = VarTxNode {
            tx_size: TX_SIZES_ALL,
            ctx: 0,
            kind: VarTxNodeKind::Leaf,
        };
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_var_tx_size(&mut w, &mut cdfs, &root).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `ctx` on a non-terminal leaf is a caller bug.
    #[test]
    fn var_tx_size_rejects_out_of_range_ctx() {
        let root = VarTxNode {
            tx_size: TX_16X16,
            ctx: TXFM_PARTITION_CONTEXTS,
            kind: VarTxNodeKind::Leaf,
        };
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_var_tx_size(&mut w, &mut cdfs, &root).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }
}
