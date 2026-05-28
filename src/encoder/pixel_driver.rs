//! Pixel-space encoder driver — arc 15 (round 221).
//!
//! The first end-to-end **pixel-in / bytes-out** entry point. Composes
//! every encoder primitive landed between r206–r220 into a single call
//! that takes a frame's luma samples and returns the IVF bytes the
//! decoder pipeline can walk back:
//!
//!   * §7.11.2.5 DC_PRED neighbour-derived prediction
//!     ([`crate::cdf::predict_intra_dc_pred`]).
//!   * Pixel residual = input − prediction.
//!   * Forward 4×4 DCT primitive ([`crate::encoder::forward_dct_4x4`],
//!     r219) — pixel residual → DCT-II coefficients.
//!   * Forward quantization ([`crate::encoder::forward_quantize`], r220)
//!     — coefficients → `Quant[]` levels for the §5.11.39 writers.
//!   * §5.11.4 recursive partition writer
//!     ([`crate::encoder::write_partition_tree`], r217) — emits
//!     PARTITION_NONE leaves at BLOCK_4X4 per 4×4 cell of the frame.
//!   * §5.11.39 coefficient writer
//!     ([`crate::encoder::write_coefficients`], r212–r215) — one Y-plane
//!     transform block per leaf.
//!   * §5.11.1 tile-group OBU framing
//!     ([`crate::encoder::write_tile_group_obu`], r210) — wraps the
//!     entropy bytes plus the byte-aligned tile-group header.
//!   * §7.5 temporal unit aggregation
//!     ([`crate::encoder::encode_temporal_unit`], r208) — TD + optional
//!     SH + per-frame OBU sequence (FrameHeader OBU + TileGroup OBU).
//!   * IVF v0 container ([`crate::encoder::IvfWriter`], r206).
//!
//! ## Scope (arc 15)
//!
//! Hard scope per the implementer round: a single 16×16 **monochrome**
//! intra-only frame at `base_q_idx = 0`, BLOCK_4X4 leaves, TX_4X4
//! DCT_DCT, default scan, no segmentation, no QM, no chroma. The
//! 16-cell luma is fed in row-major as `[[u8; 16]; 16]`.
//!
//! The encoder builds its own DC_PRED prediction by walking the
//! 16 BLOCK_4X4 leaves in §5.11.4 dispatch order (NW, NE, SW, SE
//! recursion ⇒ the leaves at `(r, c)` are visited in the order
//! `(0,0), (0,1), (1,0), (1,1), (0,2), (0,3), …`). At each leaf the
//! prediction is computed from the **already-reconstructed** neighbour
//! pixels — exactly what the decoder would observe on its parallel
//! walk. The reconstruction the encoder uses is the same
//! `inverse_transform_2d(dequantize_step1(Quant))` chain the decoder
//! runs at this leaf; this is the encoder's correctness contract.
//!
//! ## Verification
//!
//! Internal pixel-roundtrip helpers in [`internal_roundtrip`] exercise
//! the encoder-side inverse chain (dequantize + inverse transform + add
//! prediction) and surface the recovered pixels alongside the encoded
//! bytes. Integration tests assert:
//!
//!   1. The IVF bytes parse back as TD + SH + FrameHeader OBU + TileGroup
//!      OBU through [`crate::obu::ObuIter`] (structural roundtrip).
//!   2. The encoder-internal reconstructed pixels equal the input when
//!      the input pixels match the running DC_PRED prediction at every
//!      block — i.e. for the uniform mid-grey input (every sample =
//!      128). At `base_q_idx = 0`, DCT_DCT, DC_PRED on a flat-128
//!      input ⇒ zero residual ⇒ zero coefficients ⇒ zero `Quant[]` ⇒
//!      zero recovered residual ⇒ recovered pixels = 128 (bit-exact).
//!   3. For non-flat inputs the encoder-internal recovered pixels are
//!      within one quantization step of the input — the §7.12.3 step-1
//!      truncating divide bounds the per-coefficient error by one
//!      lattice step.
//!
//! ## What this arc does NOT do
//!
//!   * Drive the existing decoder back through `parse_obu` →
//!     PartitionWalker → coefficients → inverse transform → pixels. The
//!     workspace decoder entry [`crate::decode_av1`] is still a stub;
//!     reaching pixel roundtrip via the decoder is a separate arc.
//!   * Chroma planes (U/V). The monochrome path covers Y only; adding
//!     chroma needs a frame-header build with `monochrome = true` (the
//!     tiny fixture in `docs/video/av1/fixtures/` is monochrome).
//!   * Anything beyond TX_4X4 DCT_DCT. Forward transforms for larger
//!     sizes / non-DCT kernels are subsequent arcs.
//!
//! ## Spec provenance
//!
//! `docs/video/av1/av1-spec.txt`:
//!   * §5.11.39 — `coefficients()` (p.88–93).
//!   * §5.11.4  — `decode_partition` (p.61–62).
//!   * §5.11.5  — `decode_block` (p.62).
//!   * §7.11.2.5 — DC_PRED sample generation (p.249).
//!   * §7.12.3   — Reconstruction (`dequantize_step1`, p.293–295).
//!   * §7.13.3   — Inverse transform 2D (p.305–307).

use crate::cdf::{
    dequantize_step1, predict_intra_dc_pred, QuantizerParams, TileCdfContext, TileGeometry,
    BLOCK_16X16, DCT_DCT, DC_PRED, TX_4X4, TX_CLASS_2D,
};
use crate::encoder::forward_quantize::forward_quantize;
use crate::encoder::forward_transform::forward_dct_4x4;
use crate::encoder::forward_wht::forward_wht_4x4;
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::obu::ObuFrame;
use crate::encoder::partition_tree::{
    write_partition_tree, EncodeBlock, EncodeNode, PartitionTreeWriter, PlaneCoefficients,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::temporal_unit::{encode_temporal_unit, TemporalUnitPlan};
use crate::encoder::tile_group_obu::{write_tile_group_obu, TileGroupObu, TilePayload};
use crate::frame_header::FrameHeader;
use crate::obu::ObuType;
use crate::scan::get_default_scan;
use crate::sequence_header::SequenceHeader;
use crate::transform::inverse_transform_2d;
use crate::Error;

/// Frame extent the driver supports this arc: a 16×16 monochrome
/// luma surface (matches the
/// `docs/video/av1/fixtures/tiny-i-only-16x16-prof0` sequence header
/// already used by the IVF round-trip test).
pub const FRAME_WIDTH: usize = 16;
/// Frame height — see [`FRAME_WIDTH`].
pub const FRAME_HEIGHT: usize = 16;
/// `MiCols` derived per §5.9.9: `2 * ((16 + 7) >> 3) = 4`.
pub const MI_COLS: u32 = 4;
/// `MiRows` derived per §5.9.9: `2 * ((16 + 7) >> 3) = 4`.
pub const MI_ROWS: u32 = 4;
/// Number of 4×4 cells across the luma plane: `FRAME_WIDTH / 4 = 4`.
pub const CELLS_WIDE: usize = FRAME_WIDTH / 4;
/// Number of 4×4 cells down the luma plane.
pub const CELLS_HIGH: usize = FRAME_HEIGHT / 4;

/// Result of [`encode_intra_frame_y`] — bundles the IVF bytes with the
/// encoder's internal reconstructed pixels (for the verification path).
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// IVF bytes — file header + one IVF frame with the §7.5 temporal
    /// unit body (TD + optional SH + FrameHeader OBU + TileGroup OBU).
    pub ivf_bytes: Vec<u8>,
    /// §7.5 temporal unit bytes (the IVF frame payload).
    pub temporal_unit_bytes: Vec<u8>,
    /// Encoder-internal **reconstructed pixels** the decoder would
    /// produce on a parallel walk. Bit-exact when the input matches the
    /// running DC_PRED prediction at every block; within one quantizer
    /// step otherwise.
    pub reconstructed_y: [[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    /// Per-block-position `Quant[]` arrays the encoder committed to,
    /// indexed in §5.11.4 dispatch order (NW, NE, SW, SE recursion ⇒
    /// `[(0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3),
    ///  (2,0), (2,1), (3,0), (3,1), (2,2), (2,3), (3,2), (3,3)]` in
    ///  (cell_row, cell_col) coordinates).
    pub committed_quants: Vec<Vec<i32>>,
}

/// Single 4×4 cell of the luma plane the encoder visits per leaf. Used
/// inside the per-block driver; exported so callers can predict the
/// dispatch order through [`dispatch_order_cells`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellCoord {
    /// Cell row index in `0..CELLS_HIGH`. Multiply by `4` to get the
    /// pixel-row index in the luma plane.
    pub cell_row: usize,
    /// Cell column index in `0..CELLS_WIDE`. Multiply by `4` to get
    /// the pixel-column index.
    pub cell_col: usize,
}

impl CellCoord {
    /// Top-left pixel position of this cell in the luma plane.
    pub fn pixel_origin(self) -> (usize, usize) {
        (self.cell_row * 4, self.cell_col * 4)
    }
}

/// §5.11.4 dispatch-order cell sequence for a `CELLS_WIDE × CELLS_HIGH`
/// grid rooted at a single BLOCK_16X16 PARTITION_SPLIT. The §5.11.4
/// recursion is NW → NE → SW → SE, so for a 4×4 cell grid (16 cells)
/// the order is the Z-curve over the two-level split tree.
///
/// Returns the 16 cell coordinates in the order
/// `(0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3),
///  (2,0), (2,1), (3,0), (3,1), (2,2), (2,3), (3,2), (3,3)`.
#[must_use]
pub fn dispatch_order_cells() -> Vec<CellCoord> {
    let mut out = Vec::with_capacity(CELLS_WIDE * CELLS_HIGH);
    // Outer BLOCK_16X16 SPLIT ⇒ 4 BLOCK_8X8 quadrants at (0,0), (0,2),
    // (2,0), (2,2) (in mi units). Each BLOCK_8X8 SPLIT ⇒ 4 BLOCK_4X4
    // leaves at (mi+0,mi+0), (mi+0,mi+1), (mi+1,mi+0), (mi+1,mi+1).
    // Because the cell index equals the mi index for BLOCK_4X4 leaves
    // (NUM_4X4_BLOCKS_WIDE[BLOCK_4X4] = 1), the cell-coord walk lines
    // up directly with the §5.11.4 recursion.
    for &(quad_r, quad_c) in &[(0u32, 0u32), (0, 2), (2, 0), (2, 2)] {
        for dr in 0..2u32 {
            for dc in 0..2u32 {
                out.push(CellCoord {
                    cell_row: (quad_r + dr) as usize,
                    cell_col: (quad_c + dc) as usize,
                });
            }
        }
    }
    out
}

/// Compute the DC_PRED prediction for the 4×4 cell at `(cell_row,
/// cell_col)` against the running `reconstructed` luma buffer. Mirrors
/// the §7.11.2.5 four-arm dispatch on `(haveLeft, haveAbove)` via
/// [`predict_intra_dc_pred`] in a contained-extent form.
///
/// `bit_depth` must be `8`; this driver runs at `BitDepth = 8` per
/// §5.5.2 (the tiny fixture's sequence header).
fn dc_pred_for_cell_y(
    reconstructed: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    cell_row: usize,
    cell_col: usize,
) -> [u8; 16] {
    let row0 = cell_row * 4;
    let col0 = cell_col * 4;
    let have_above = (cell_row > 0) as u8;
    let have_left = (cell_col > 0) as u8;
    let mut above_row: [u16; 4] = [0; 4];
    let mut left_col: [u16; 4] = [0; 4];
    if have_above != 0 {
        for j in 0..4 {
            above_row[j] = reconstructed[row0 - 1][col0 + j] as u16;
        }
    }
    if have_left != 0 {
        for i in 0..4 {
            left_col[i] = reconstructed[row0 + i][col0 - 1] as u16;
        }
    }
    let mut pred16 = [0u16; 16];
    predict_intra_dc_pred(
        have_left,
        have_above,
        2,
        2,
        4,
        4,
        8,
        &above_row,
        &left_col,
        &mut pred16,
    )
    .expect("oxideav-av1 pixel_driver: DC_PRED arguments in range");
    let mut pred8 = [0u8; 16];
    for (slot, v) in pred8.iter_mut().zip(pred16.iter().copied()) {
        *slot = v as u8;
    }
    pred8
}

/// Encode a 16×16 monochrome Y-only intra-only frame at `base_q_idx = 0`
/// against the tiny-fixture-derived sequence + frame headers.
///
/// Returns the IVF bytes (one frame), the temporal-unit body bytes, and
/// the encoder-internal reconstructed pixel plane. See [`EncodedFrame`].
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] for any internal partition-tree
/// or coefficient-writer overflow (none expected for the in-range
/// 16×16 monochrome frame, but propagated for diagnostic visibility).
pub fn encode_intra_frame_y(
    luma_in: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    seq: &SequenceHeader,
    fh: &FrameHeader,
) -> Result<EncodedFrame, Error> {
    // Quantizer state matching the tiny fixture's `base_q_idx = 0`
    // header: dc_q = ac_q = 4, dqDenom = 1, no QM.
    let qp = QuantizerParams::neutral(0, 8);
    let scan = get_default_scan(TX_4X4).to_vec();
    // §5.9.2 `CodedLossless` predicate, simplified for the
    // neutral-quantizer / no-segmentation path this driver runs on:
    // `base_q_idx == 0 && all DeltaQ?? == 0` (the `QuantizerParams::neutral`
    // constructor zeros every delta). When true, the per-block §6.10.1
    // `Lossless` flag is `1` for every block in the frame ⇒ §5.11.5
    // forces `tx_size = TX_4X4` and §7.13.3 routes through the
    // §7.13.2.10 inverse WHT path. The encoder mirrors this with the
    // forward WHT instead of forward DCT, which makes the
    // `(forward_wht, forward_quantize) → (dequantize_step1, inverse_WHT)`
    // chain bit-exact on arbitrary inputs.
    let lossless = qp.base_q_idx == 0
        && qp.delta_q_y_dc == 0
        && qp.delta_q_u_dc == 0
        && qp.delta_q_u_ac == 0
        && qp.delta_q_v_dc == 0
        && qp.delta_q_v_ac == 0;

    // Encoder-side state: a running reconstructed-pixel buffer and a
    // per-cell list of committed `Quant[]` arrays in dispatch order.
    let mut reconstructed: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
    let mut committed_quants: Vec<Vec<i32>> = Vec::with_capacity(CELLS_WIDE * CELLS_HIGH);

    // Build the §5.11.4 dispatch-ordered list of (cell, EncodeBlock)
    // pairs. For each cell, the leaf carries the per-plane
    // `coefficients` from `forward_quantize(forward_dct_4x4(residual))`.
    // The DC_PRED prediction is computed from the running
    // `reconstructed` buffer — this keeps the encoder's predictor in
    // lockstep with what the decoder would compute on its parallel walk.
    let cells = dispatch_order_cells();
    let mut leaves: Vec<EncodeBlock> = Vec::with_capacity(cells.len());
    for cc in &cells {
        let (row0, col0) = cc.pixel_origin();
        let pred = dc_pred_for_cell_y(&reconstructed, cc.cell_row, cc.cell_col);
        // Residual = input - prediction.
        let mut residual = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let p = luma_in[row0 + i][col0 + j] as i64;
                let q = pred[i * 4 + j] as i64;
                residual[i * 4 + j] = p - q;
            }
        }
        // Forward transform. On the lossless arm route through the
        // forward 4×4 WHT (the bit-exact integer inverse of the
        // §7.13.2.10 inverse WHT used by §7.13.3's `Lossless` branch);
        // otherwise the forward DCT (lossy at q_index > 0).
        let coeffs = if lossless {
            forward_wht_4x4(&residual).to_vec()
        } else {
            forward_dct_4x4(&residual).to_vec()
        };
        // Forward quantize at q_index = 0, DCT_DCT, no QM (seg_qm_level
        // = 15 takes the identity branch). On the lossless arm, every
        // `forward_wht_4x4` output is divisible by `4 = q2 / dq_denom`
        // (the row pass multiplies by `1 << 2`), so the forward
        // quantizer round-trip is bit-exact.
        let quant = forward_quantize(&coeffs, TX_4X4, /* plane = */ 0, 0, DCT_DCT, 15, &qp);
        committed_quants.push(quant.clone());

        // Encoder-internal pixel reconstruction: decoder walk of these
        // same `quant` values. Pass the `lossless` flag through to the
        // §7.13.3 dispatcher so the inverse WHT path matches the
        // encoder's forward WHT choice.
        let dequant = dequantize_step1(&quant, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let residual_back = inverse_transform_2d(
            &dequant, TX_4X4, DCT_DCT, /* bit_depth = */ 8, lossless,
        );
        for i in 0..4 {
            for j in 0..4 {
                let p = pred[i * 4 + j] as i64 + residual_back[i * 4 + j];
                let clamped = p.clamp(0, 255) as u8;
                reconstructed[row0 + i][col0 + j] = clamped;
            }
        }

        // Build the §5.11.5 leaf for the §5.11.4 driver. Y-only ⇒
        // `uv_mode = None`.
        let plane_y = PlaneCoefficients {
            plane: 0,
            is_inter: 0,
            tx_size: TX_4X4,
            tx_class: TX_CLASS_2D,
            txb_skip_ctx: 0,
            dc_sign_ctx: 0,
            scan: scan.clone(),
            quant: quant.clone(),
        };
        leaves.push(EncodeBlock {
            skip: 0,
            segment_id: 0,
            segment_pred: 0,
            y_mode: DC_PRED as u8,
            uv_mode: None,
            coefficients: vec![plane_y],
        });
    }

    // Assemble the §5.11.4 partition tree: BLOCK_16X16 root SPLIT into
    // 4 BLOCK_8X8 SPLITs, each into 4 BLOCK_4X4 leaves.
    let leaf_iter = leaves.into_iter().map(EncodeNode::Leaf);
    let leaves_arr: Vec<EncodeNode> = leaf_iter.collect();
    debug_assert_eq!(leaves_arr.len(), 16);
    // Build innermost BLOCK_8X8 SPLIT nodes (4 of them, one per quad).
    let mut leaf_iter = leaves_arr.into_iter();
    let mut quads: Vec<EncodeNode> = Vec::with_capacity(4);
    for _ in 0..4 {
        let a = Box::new(leaf_iter.next().unwrap());
        let b = Box::new(leaf_iter.next().unwrap());
        let c = Box::new(leaf_iter.next().unwrap());
        let d = Box::new(leaf_iter.next().unwrap());
        quads.push(EncodeNode::Split([a, b, c, d]));
    }
    let nw = Box::new(quads.remove(0));
    let ne = Box::new(quads.remove(0));
    let sw = Box::new(quads.remove(0));
    let se = Box::new(quads.remove(0));
    let root = EncodeNode::Split([nw, ne, sw, se]);

    // §5.11 entropy-coder run for the single tile covering the frame.
    let mut writer = SymbolWriter::new(false);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut state = PartitionTreeWriter::new(
        MI_ROWS,
        MI_COLS,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: MI_ROWS,
            mi_col_start: 0,
            mi_col_end: MI_COLS,
        },
        /* segmentation_enabled = */ false,
        /* last_active_seg_id = */ 0,
        /* lossless = */ false,
        /* subsampling_x = */ false,
        /* subsampling_y = */ false,
    )
    .expect("oxideav-av1 pixel_driver: partition state construction");
    write_partition_tree(&mut writer, &mut cdfs, &mut state, &root, 0, 0, BLOCK_16X16)?;
    let tile_bytes = writer.finish();

    // §5.11.1 tile-group OBU body.
    let tile_group = TileGroupObu {
        num_tiles: 1,
        tile_cols_log2: 0,
        tile_rows_log2: 0,
        tile_size_bytes: 1,
        tg_start: 0,
        tg_end: 0,
        start_and_end_present: false,
        tiles: vec![TilePayload::new(tile_bytes)],
    };
    let tile_group_body = write_tile_group_obu(&tile_group)?;

    // Build the per-frame OBU sequence: FrameHeader OBU + TileGroup OBU.
    // `encode_temporal_unit` handles the SH-emission gate.
    // We hand-build the frame_obus list because the temporal-unit
    // helper only knows about FrameHeaders; tile-group OBUs are
    // appended after each frame's FH.
    let fh_body = crate::encoder::frame_obu::write_frame_header_obu(fh, seq);
    let frame_obus: Vec<ObuFrame> = vec![
        ObuFrame::new(ObuType::FrameHeader, fh_body),
        ObuFrame::new(ObuType::TileGroup, tile_group_body),
    ];

    // §7.5 temporal unit: TD + SH + (FH + TileGroup).
    let temporal_unit_bytes = {
        use crate::encoder::obu::build_temporal_unit;
        use crate::encoder::sequence_obu::write_sequence_header_obu;
        let sh_body = write_sequence_header_obu(seq);
        build_temporal_unit(Some(&sh_body), &frame_obus)
    };
    // `TemporalUnitPlan` is the public entry; we shell out to
    // `build_temporal_unit` directly because `TemporalUnitPlan` only
    // emits one OBU per frame. Keep `TemporalUnitPlan` in scope for
    // the next arc that lands a `frames: &[FrameAndTiles]` extension.
    let _ = TemporalUnitPlan {
        seq,
        emit_sequence_header: true,
        frames: &[],
    };
    let _ = encode_temporal_unit;

    // IVF v0 container — one frame at pts = 0.
    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(
            cursor,
            FOURCC_AV01,
            FRAME_WIDTH as u16,
            FRAME_HEIGHT as u16,
            25,
            1,
        )
        .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.write_frame(&temporal_unit_bytes, 0)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedFrame {
        ivf_bytes,
        temporal_unit_bytes,
        reconstructed_y: reconstructed,
        committed_quants,
    })
}

/// Internal-only helpers re-exposed for the integration tests; not part
/// of the public encoder API.
pub mod internal_roundtrip {
    use super::*;

    /// Re-derive the encoder's reconstructed plane from a list of
    /// committed `Quant[]` arrays + the original input plane (for the
    /// prediction baseline). Mirrors the per-leaf reconstruction loop
    /// inside [`super::encode_intra_frame_y`]; provided so test code can
    /// independently verify the per-leaf pixel reconstruction without
    /// re-running the encoder driver.
    pub fn reconstruct_from_quants(quants: &[Vec<i32>]) -> [[u8; FRAME_WIDTH]; FRAME_HEIGHT] {
        let qp = QuantizerParams::neutral(0, 8);
        let lossless = qp.base_q_idx == 0
            && qp.delta_q_y_dc == 0
            && qp.delta_q_u_dc == 0
            && qp.delta_q_u_ac == 0
            && qp.delta_q_v_dc == 0
            && qp.delta_q_v_ac == 0;
        let mut reconstructed: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] =
            [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let cells = dispatch_order_cells();
        assert_eq!(quants.len(), cells.len());
        for (cc, quant) in cells.iter().zip(quants.iter()) {
            let (row0, col0) = cc.pixel_origin();
            let pred = dc_pred_for_cell_y(&reconstructed, cc.cell_row, cc.cell_col);
            let dequant = dequantize_step1(quant, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, lossless);
            for i in 0..4 {
                for j in 0..4 {
                    let p = pred[i * 4 + j] as i64 + residual[i * 4 + j];
                    reconstructed[row0 + i][col0 + j] = p.clamp(0, 255) as u8;
                }
            }
        }
        reconstructed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_header::parse_frame_header;
    use crate::obu::{ObuIter, ObuType};
    use crate::sequence_header::parse_sequence_header;

    // Tiny-i-only-16x16-prof0 fixture payloads (also used by the IVF
    // end-to-end test).
    const TINY_SEQ_PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
    const TINY_FRAME_PAYLOAD: &[u8] = &[
        0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67,
        0x6c, 0xc7, 0xee, 0x51, 0x80,
    ];

    fn tiny_seq() -> SequenceHeader {
        parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap()
    }

    fn tiny_fh(seq: &SequenceHeader) -> FrameHeader {
        parse_frame_header(TINY_FRAME_PAYLOAD, seq).unwrap()
    }

    #[test]
    fn dispatch_order_cells_visits_each_cell_exactly_once() {
        let cells = dispatch_order_cells();
        assert_eq!(cells.len(), CELLS_WIDE * CELLS_HIGH);
        let mut seen = [[false; CELLS_WIDE]; CELLS_HIGH];
        for cc in &cells {
            assert!(cc.cell_row < CELLS_HIGH);
            assert!(cc.cell_col < CELLS_WIDE);
            assert!(
                !seen[cc.cell_row][cc.cell_col],
                "duplicate cell ({}, {})",
                cc.cell_row, cc.cell_col
            );
            seen[cc.cell_row][cc.cell_col] = true;
        }
        for row in &seen {
            for &s in row {
                assert!(s);
            }
        }
    }

    #[test]
    fn dispatch_order_cells_first_quad_is_z_curve() {
        // First four cells = NW quadrant of NW outer split = the four
        // BLOCK_4X4 leaves of the BLOCK_8X8 at mi (0, 0):
        // (0,0), (0,1), (1,0), (1,1) in cell coords.
        let cells = dispatch_order_cells();
        assert_eq!(
            cells[..4],
            [
                CellCoord {
                    cell_row: 0,
                    cell_col: 0
                },
                CellCoord {
                    cell_row: 0,
                    cell_col: 1
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 0
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 1
                },
            ]
        );
        // Cells 4..8 = NE outer quadrant = mi (0, 2) BLOCK_8X8: (0,2),
        // (0,3), (1,2), (1,3).
        assert_eq!(
            cells[4..8],
            [
                CellCoord {
                    cell_row: 0,
                    cell_col: 2
                },
                CellCoord {
                    cell_row: 0,
                    cell_col: 3
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 2
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 3
                },
            ]
        );
    }

    #[test]
    fn encode_uniform_128_input_produces_zero_quant_levels() {
        // Flat mid-grey input ⇒ DC_PRED at (0,0) is the no-neighbour
        // mid-grey 128 ⇒ residual = 0 ⇒ coefficients = 0 ⇒ quants = 0.
        // Subsequent leaves see reconstructed-128 neighbours, so DC_PRED
        // again yields 128 ⇒ residual = 0 chain repeats.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(result.committed_quants.len(), CELLS_WIDE * CELLS_HIGH);
        for (idx, quant) in result.committed_quants.iter().enumerate() {
            assert_eq!(quant.len(), 16);
            for (k, &q) in quant.iter().enumerate() {
                assert_eq!(q, 0, "cell {idx} quant[{k}] = {q}, expected 0");
            }
        }
    }

    #[test]
    fn encode_uniform_128_input_reconstructs_bit_exact() {
        // The encoder's internal reconstruction must match the input
        // exactly for the flat-DC case: zero residual ⇒ recovered pixels
        // = DC_PRED prediction = 128 everywhere.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        for i in 0..FRAME_HEIGHT {
            for j in 0..FRAME_WIDTH {
                assert_eq!(
                    result.reconstructed_y[i][j], 128,
                    "reconstructed_y[{i}][{j}] = {} != 128",
                    result.reconstructed_y[i][j]
                );
            }
        }
    }

    #[test]
    fn encode_uniform_128_internal_roundtrip_matches_driver() {
        // The `internal_roundtrip::reconstruct_from_quants` helper must
        // produce the same pixels as the driver's inline reconstruction
        // when fed the same committed `Quant[]` arrays.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let recon = internal_roundtrip::reconstruct_from_quants(&result.committed_quants);
        assert_eq!(recon, result.reconstructed_y);
    }

    #[test]
    fn encode_uniform_64_input_internal_inverse_chain_is_bit_exact() {
        // Flat input ≠ mid-grey ⇒ first-block residual is `-64` (every
        // pixel minus the no-neighbour DC_PRED = 128 default). The
        // r222 forward WHT routes the q_index = 0 path through the
        // §7.13.3 `Lossless` arm, so the encoder's reconstruction is
        // pixel-perfect bit-exact — no per-pixel error envelope this
        // time (subsequent blocks predict from the recovered 64 pixels
        // ⇒ zero residual chain).
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[64u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let recon = internal_roundtrip::reconstruct_from_quants(&result.committed_quants);
        assert_eq!(recon, result.reconstructed_y);
        for i in 0..FRAME_HEIGHT {
            for j in 0..FRAME_WIDTH {
                assert_eq!(
                    result.reconstructed_y[i][j], 64,
                    "[{i}][{j}]: recon = {} != 64 (lossless WHT path)",
                    result.reconstructed_y[i][j]
                );
            }
        }
    }

    /// Bit-exact pixel roundtrip on a non-uniform input — the r222
    /// milestone unlock. At `base_q_idx = 0` the §7.13.3 `Lossless`
    /// arm routes through the §7.13.2.10 WHT, which is a pure
    /// integer-reversible butterfly. The encoder mirrors with
    /// `forward_wht_4x4`, and every coefficient ends up divisible by
    /// the lossless q2 = 4 (the WHT row pass multiplies by `1 << 2`),
    /// so the quantize → dequantize chain is exact. End-to-end the
    /// encoder's reconstruction must equal the input pixel-for-pixel.
    #[test]
    fn encode_non_uniform_input_roundtrips_bit_exact_lossless() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        // Deterministic non-uniform pattern: pixel = (16 * row + col)
        // mod 256, which exercises a wide range of inter-cell deltas
        // and forces non-zero residuals at every block.
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for (i, row) in luma.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = ((16 * i + j) & 0xFF) as u8;
            }
        }
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(
            result.reconstructed_y, luma,
            "lossless WHT roundtrip failed on non-uniform input"
        );
    }

    /// Bit-exact pixel roundtrip on a horizontal-gradient input —
    /// stresses the row-direction WHT path. The gradient covers the
    /// full `[0, 255]` u8 range across the frame width.
    #[test]
    fn encode_horizontal_gradient_roundtrips_bit_exact_lossless() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for row in luma.iter_mut() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (j * 16) as u8;
            }
        }
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(result.reconstructed_y, luma);
    }

    /// Bit-exact pixel roundtrip on a pseudo-random input — the
    /// definitive "any pixels round-trip exactly" test. Deterministic
    /// LCG so failures replay identically.
    #[test]
    fn encode_pseudorandom_input_roundtrips_bit_exact_lossless() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        for row in luma.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 56) as u8;
            }
        }
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(
            result.reconstructed_y, luma,
            "lossless WHT roundtrip failed on pseudo-random input"
        );
    }

    #[test]
    fn encoded_ivf_bytes_parse_back_as_td_sh_fh_tg() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");

        // IVF preamble.
        assert_eq!(&result.ivf_bytes[0..4], b"DKIF");
        // Frame count == 1.
        let frame_count = u32::from_le_bytes([
            result.ivf_bytes[24],
            result.ivf_bytes[25],
            result.ivf_bytes[26],
            result.ivf_bytes[27],
        ]);
        assert_eq!(frame_count, 1);

        // Walk the temporal-unit OBUs: TD + SH + FH + TG.
        let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(descs.len(), 4, "TD + SH + FH + TG");
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
        assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
        assert_eq!(descs[3].obu_type, ObuType::TileGroup);
    }

    #[test]
    fn encoded_ivf_sh_fh_round_trip_through_parser() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        // SH parses back equal (up to bits_consumed).
        let reparsed_seq = parse_sequence_header(descs[1].payload).unwrap();
        let mut expected_seq = seq.clone();
        expected_seq.bits_consumed = reparsed_seq.bits_consumed;
        assert_eq!(reparsed_seq, expected_seq);
        // FH parses back equal (up to bits_consumed).
        let reparsed_fh = parse_frame_header(descs[2].payload, &seq).unwrap();
        let mut expected_fh = fh.clone();
        expected_fh.bits_consumed = reparsed_fh.bits_consumed;
        assert_eq!(reparsed_fh, expected_fh);
    }
}
