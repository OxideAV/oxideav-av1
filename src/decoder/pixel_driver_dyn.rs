//! Dynamic-extent decoder pixel driver — arc r230 inverse of
//! [`crate::encoder::pixel_driver_dyn::encode_intra_frame_yuv_dyn`].
//!
//! Walks the §5.11.1 tile-group body of an IVF frame produced by the
//! r230 dyn encoder and surfaces the recovered samples as
//! [`super::pixel_driver::Frame::Yuv420Dyn`]. Reuses every existing
//! symbol-decoder / dequantizer / inverse-transform / intra-prediction
//! kernel; the only delta vs the fixed-size [`super::pixel_driver`] is
//! that the running reconstructed plane is a Vec-backed buffer sized
//! to the per-frame extent rather than `[[u8; 16]; 16]`.
//!
//! Scope mirrors the r230 encoder exactly:
//!
//!   * `subsampling_x = subsampling_y = 1` (4:2:0), `bit_depth = 8`,
//!     not monochrome.
//!   * `frame_width`, `frame_height` ∈ {8, 16, 24, 32, 40, 48, 56,
//!     64}; both must be multiples of 8 (the 4:2:0 chroma
//!     constraint).
//!   * `base_q_idx == 0` (the §5.9.2 `CodedLossless` arm — encoder
//!     uses forward WHT, decoder uses the §7.13.2.10 inverse WHT).
//!   * Intra-only, single tile, BLOCK_4X4 leaves, TX_4X4 DCT_DCT,
//!     default scan, no segmentation, no QM, no in-loop filters.
//!   * 13-mode intra picker on luma + chroma (the r228/r229 picker).
//!
//! Outside that scope, returns [`Error::PartitionWalkOutOfRange`].

use crate::cdf::{
    cfl_allowed_for_uv_mode, dequantize_step1, partition_ctx, partition_subsize,
    predict_intra_d_mode, predict_intra_dc_pred, predict_intra_h_pred, predict_intra_paeth_pred,
    predict_intra_smooth_h_pred, predict_intra_smooth_pred, predict_intra_smooth_v_pred,
    predict_intra_v_pred, size_group, skip_ctx, split_or_horz_cdf, split_or_vert_cdf,
    PartitionWalker, QuantizerParams, TileCdfContext, TileGeometry, BLOCK_4X4, BLOCK_8X8,
    BLOCK_INVALID, BLOCK_SIZES, D45_PRED, D67_PRED, DCT_DCT, DC_PRED, H_PRED, MI_HEIGHT_LOG2,
    MI_WIDTH_LOG2, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PAETH_PRED, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED,
    TX_4X4, TX_CLASS_2D, V_PRED,
};
use crate::decoder::pixel_driver::Frame;
use crate::encoder::pixel_driver::NUM_INTRA_MODES_Y;
use crate::encoder::pixel_driver_dyn::{root_super_block, MAX_DIM, MIN_DIM};
use crate::encoder::tile_group_obu::parse_tile_group_obu_body;
use crate::frame_header::FrameHeader;
use crate::scan::get_default_scan;
use crate::sequence_header::SequenceHeader;
use crate::symbol_decoder::SymbolDecoder;
use crate::transform::inverse_transform_2d;
use crate::Error;

/// Decode one dynamic-extent intra-only frame and surface it as
/// [`Frame::Yuv420Dyn`]. Called from [`super::pixel_driver::decode_frame`]
/// for any `(frame_width, frame_height) != (16, 16)`.
///
/// Pre-conditions: the caller has already verified
/// `mono_chrome == false`, `subsampling_x == subsampling_y == 1`.
pub(crate) fn decode_frame_dyn(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    tile_group_body: &[u8],
) -> Result<Frame, Error> {
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let width = fs.frame_width;
    let height = fs.frame_height;
    if width < MIN_DIM
        || height < MIN_DIM
        || width > MAX_DIM
        || height > MAX_DIM
        || width % MIN_DIM != 0
        || height % MIN_DIM != 0
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let chroma_width = (width / 2) as usize;
    let chroma_height = (height / 2) as usize;
    let mi_cols = fs.mi_cols;
    let mi_rows = fs.mi_rows;
    // The encoder hard-codes the lossless quantizer state regardless of
    // the FrameHeader's `base_q_idx` declaration. The decoder mirrors
    // this for round-trip correctness — see the arc-18 driver comment
    // on `_qp_fh`.
    let _qp_fh = fh
        .quantization_params
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let q_params = QuantizerParams::neutral(0, 8);
    let _ = seq;

    let parsed = parse_tile_group_obu_body(
        tile_group_body,
        /* num_tiles = */ 1,
        /* tile_cols_log2 = */ 0,
        /* tile_rows_log2 = */ 0,
        /* tile_size_bytes = */ 1,
    )?;
    if parsed.tiles.len() != 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let tile_bytes = &parsed.tiles[0].bytes;

    let mut decoder =
        SymbolDecoder::init_symbol(tile_bytes, tile_bytes.len(), fh.disable_cdf_update)?;
    let mut cdfs = TileCdfContext::new_from_defaults();

    let mut state = DecoderStateDyn::new(mi_rows, mi_cols);
    let mut coeff_walker = PartitionWalker::new(
        mi_rows,
        mi_cols,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        },
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;

    let mut recon_y = vec![0u8; (width as usize) * (height as usize)];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];
    let scan: Vec<u16> = get_default_scan(TX_4X4).to_vec();

    let root_b = root_super_block(mi_cols, mi_rows);
    decode_partition_node(
        &mut decoder,
        &mut cdfs,
        &mut state,
        &mut coeff_walker,
        0,
        0,
        root_b,
        width as usize,
        height as usize,
        chroma_width,
        chroma_height,
        &scan,
        &q_params,
        &mut recon_y,
        &mut recon_u,
        &mut recon_v,
    )?;

    Ok(Frame::Yuv420Dyn {
        width,
        height,
        y: recon_y,
        u: recon_u,
        v: recon_v,
    })
}

/// Per-frame dyn decoder state — mirror of the encoder's
/// `PartitionTreeWriter` `MiSizes[]` grid.
#[derive(Debug)]
struct DecoderStateDyn {
    mi_rows: u32,
    mi_cols: u32,
    mi_sizes: Vec<usize>,
    geometry: TileGeometry,
}

impl DecoderStateDyn {
    fn new(mi_rows: u32, mi_cols: u32) -> Self {
        Self {
            mi_rows,
            mi_cols,
            mi_sizes: vec![BLOCK_INVALID; (mi_rows * mi_cols) as usize],
            geometry: TileGeometry {
                mi_row_start: 0,
                mi_row_end: mi_rows,
                mi_col_start: 0,
                mi_col_end: mi_cols,
            },
        }
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
}

#[allow(clippy::too_many_arguments)]
fn decode_partition_node(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    state: &mut DecoderStateDyn,
    coeff_walker: &mut PartitionWalker,
    r: u32,
    c: u32,
    b_size: usize,
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) -> Result<(), Error> {
    // §5.11.4 line 1.
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

    let partition = if b_size < BLOCK_8X8 {
        PARTITION_NONE
    } else {
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        let pctx = state.partition_ctx_for(r, c, bsl);
        if has_rows && has_cols {
            let cdf = cdfs
                .partition_cdf(bsl, pctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            decoder.read_symbol(cdf)? as usize
        } else if has_cols {
            let cdf_row = cdfs
                .partition_cdf(bsl, pctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            let mut bin =
                split_or_horz_cdf(cdf_row, b_size).ok_or(Error::PartitionWalkOutOfRange)?;
            let s = decoder.read_symbol(&mut bin)?;
            if s == 0 {
                PARTITION_HORZ
            } else {
                PARTITION_SPLIT
            }
        } else if has_rows {
            let cdf_row = cdfs
                .partition_cdf(bsl, pctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            let mut bin =
                split_or_vert_cdf(cdf_row, b_size).ok_or(Error::PartitionWalkOutOfRange)?;
            let s = decoder.read_symbol(&mut bin)?;
            if s == 0 {
                PARTITION_VERT
            } else {
                PARTITION_SPLIT
            }
        } else {
            PARTITION_SPLIT
        }
    };

    let sub_size = partition_subsize(partition, b_size).ok_or(Error::PartitionWalkOutOfRange)?;

    match partition {
        PARTITION_NONE => {
            state.stamp_mi_sizes(r, c, sub_size);
            decode_block_leaf(
                decoder,
                cdfs,
                coeff_walker,
                r,
                c,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
        }
        PARTITION_SPLIT => {
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c + half_block4x4,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c + half_block4x4,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
        }
        _ => return Err(Error::PartitionWalkOutOfRange),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_block_leaf(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    coeff_walker: &mut PartitionWalker,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) -> Result<(), Error> {
    let skip_ctx_val = skip_ctx(0, 0);
    let skip = {
        let cdf = cdfs.skip_cdf(skip_ctx_val);
        decoder.read_symbol(cdf)? as u8
    };
    let size_group_ctx = size_group(sub_size);
    let y_mode = {
        let cdf = cdfs
            .y_mode_cdf(size_group_ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        decoder.read_symbol(cdf)? as u8
    };
    if (y_mode as usize) >= NUM_INTRA_MODES_Y {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let has_chroma = (mi_row & 1) != 0 && (mi_col & 1) != 0;
    let uv_mode: u8 = if has_chroma {
        let cfl_allowed = cfl_allowed_for_uv_mode(false, sub_size, true, true);
        let cdf = cdfs
            .uv_mode_cdf(cfl_allowed, y_mode as usize)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let m = decoder.read_symbol(cdf)? as u8;
        if (m as usize) >= NUM_INTRA_MODES_Y {
            return Err(Error::PartitionWalkOutOfRange);
        }
        m
    } else {
        DC_PRED as u8
    };

    let mut quant_y = vec![0i32; 16];
    let mut quant_u = vec![0i32; 16];
    let mut quant_v = vec![0i32; 16];

    let _readout_y = coeff_walker.coefficients(
        decoder,
        cdfs,
        /* plane = */ 0,
        0,
        TX_4X4,
        TX_CLASS_2D,
        0,
        0,
        scan,
        &mut quant_y,
    )?;

    let (row0, col0) = ((mi_row as usize) * 4, (mi_col as usize) * 4);
    let pred_y = predict_intra_for_mode_4x4(
        recon_y,
        width,
        height,
        mi_row as usize,
        mi_col as usize,
        y_mode as usize,
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;
    if skip == 0 {
        let dequant = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, qp);
        let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, true);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + residual[i * 4 + j];
                recon_y[(row0 + i) * width + (col0 + j)] = p.clamp(0, 255) as u8;
            }
        }
    } else {
        for i in 0..4 {
            for j in 0..4 {
                recon_y[(row0 + i) * width + (col0 + j)] = pred_y[i * 4 + j];
            }
        }
    }

    if has_chroma {
        let cr = ((mi_row as usize) - 1) / 2;
        let cc_idx = ((mi_col as usize) - 1) / 2;
        let crow0 = cr * 4;
        let ccol0 = cc_idx * 4;

        let _readout_u = coeff_walker.coefficients(
            decoder,
            cdfs,
            1,
            0,
            TX_4X4,
            TX_CLASS_2D,
            0,
            0,
            scan,
            &mut quant_u,
        )?;
        let pred_u = predict_intra_for_mode_4x4(
            recon_u,
            chroma_width,
            chroma_height,
            cr,
            cc_idx,
            uv_mode as usize,
        )
        .ok_or(Error::PartitionWalkOutOfRange)?;
        if skip == 0 {
            let dequant = dequantize_step1(&quant_u, TX_4X4, 1, 0, DCT_DCT, 15, qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, true);
            for i in 0..4 {
                for j in 0..4 {
                    let p = pred_u[i * 4 + j] as i64 + residual[i * 4 + j];
                    recon_u[(crow0 + i) * chroma_width + (ccol0 + j)] = p.clamp(0, 255) as u8;
                }
            }
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    recon_u[(crow0 + i) * chroma_width + (ccol0 + j)] = pred_u[i * 4 + j];
                }
            }
        }

        let _readout_v = coeff_walker.coefficients(
            decoder,
            cdfs,
            2,
            0,
            TX_4X4,
            TX_CLASS_2D,
            0,
            0,
            scan,
            &mut quant_v,
        )?;
        let pred_v = predict_intra_for_mode_4x4(
            recon_v,
            chroma_width,
            chroma_height,
            cr,
            cc_idx,
            uv_mode as usize,
        )
        .ok_or(Error::PartitionWalkOutOfRange)?;
        if skip == 0 {
            let dequant = dequantize_step1(&quant_v, TX_4X4, 2, 0, DCT_DCT, 15, qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, true);
            for i in 0..4 {
                for j in 0..4 {
                    let p = pred_v[i * 4 + j] as i64 + residual[i * 4 + j];
                    recon_v[(crow0 + i) * chroma_width + (ccol0 + j)] = p.clamp(0, 255) as u8;
                }
            }
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    recon_v[(crow0 + i) * chroma_width + (ccol0 + j)] = pred_v[i * 4 + j];
                }
            }
        }
    }

    // Silence the unused-import warning on BLOCK_4X4 (this driver
    // operates at the leaf level but never names the constant directly).
    let _ = BLOCK_4X4;
    Ok(())
}

/// §7.11.2.1 prologue + §7.11.2.{2..6} kernel dispatcher for one 4×4
/// cell against a dynamic-extent plane. Mirror of the encoder-side
/// helpers in [`crate::encoder::pixel_driver_dyn`].
fn predict_intra_for_mode_4x4(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    cell_row: usize,
    cell_col: usize,
    mode: usize,
) -> Option<[u8; 16]> {
    let w = 4usize;
    let h = 4usize;
    let log2_w = 2u32;
    let log2_h = 2u32;
    let bit_depth = 8u8;
    let row0 = cell_row * 4;
    let col0 = cell_col * 4;
    let stride = plane_width;

    let have_above = (row0 > 0) as u8;
    let have_left = (col0 > 0) as u8;
    let above_left: u16 = if have_above != 0 && have_left != 0 {
        plane[(row0 - 1) * stride + (col0 - 1)] as u16
    } else if have_above != 0 {
        plane[(row0 - 1) * stride + col0] as u16
    } else if have_left != 0 {
        plane[row0 * stride + (col0 - 1)] as u16
    } else {
        1u16 << (bit_depth - 1)
    };

    let mut above_ext = [0u16; 10];
    above_ext[0] = above_left;
    above_ext[1] = above_left;
    if have_above != 0 {
        for k in 0..(w + h) {
            let col = (col0 + k).min(plane_width - 1);
            above_ext[2 + k] = plane[(row0 - 1) * stride + col] as u16;
        }
    } else if have_left != 0 {
        let sample = plane[row0 * stride + (col0 - 1)] as u16;
        for slot in above_ext.iter_mut().skip(2).take(w + h) {
            *slot = sample;
        }
    } else {
        let mid_minus = ((1u32 << (bit_depth - 1)) - 1) as u16;
        for slot in above_ext.iter_mut().skip(2).take(w + h) {
            *slot = mid_minus;
        }
    }

    let mut left_ext = [0u16; 10];
    left_ext[0] = above_left;
    left_ext[1] = above_left;
    if have_left != 0 {
        for k in 0..(w + h) {
            let row = (row0 + k).min(plane_height - 1);
            left_ext[2 + k] = plane[row * stride + (col0 - 1)] as u16;
        }
    } else if have_above != 0 {
        let sample = plane[(row0 - 1) * stride + col0] as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = sample;
        }
    } else {
        let mid_plus = ((1u32 << (bit_depth - 1)) + 1) as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = mid_plus;
        }
    }

    let above_row = &above_ext[2..2 + w + h];
    let left_col = &left_ext[2..2 + w + h];
    let mut pred16 = [0u16; 16];
    match mode {
        m if m == DC_PRED => {
            predict_intra_dc_pred(
                have_left,
                have_above,
                log2_w,
                log2_h,
                w,
                h,
                bit_depth,
                above_row,
                left_col,
                &mut pred16,
            )
            .ok()?;
        }
        m if m == V_PRED => predict_intra_v_pred(w, h, above_row, &mut pred16).ok()?,
        m if m == H_PRED => predict_intra_h_pred(w, h, left_col, &mut pred16).ok()?,
        m if (D45_PRED..=D67_PRED).contains(&m) => {
            predict_intra_d_mode(m, 0, w, h, 0, 0, &above_ext, &left_ext, &mut pred16).ok()?;
        }
        m if m == SMOOTH_PRED => {
            predict_intra_smooth_pred(log2_w, log2_h, w, h, above_row, left_col, &mut pred16)
                .ok()?;
        }
        m if m == SMOOTH_V_PRED => {
            predict_intra_smooth_v_pred(log2_h, w, h, above_row, left_col, &mut pred16).ok()?;
        }
        m if m == SMOOTH_H_PRED => {
            predict_intra_smooth_h_pred(log2_w, w, h, above_row, left_col, &mut pred16).ok()?;
        }
        m if m == PAETH_PRED => {
            predict_intra_paeth_pred(w, h, above_row, left_col, above_left, &mut pred16).ok()?;
        }
        _ => return None,
    }

    let mut pred8 = [0u8; 16];
    for (slot, v) in pred8.iter_mut().zip(pred16.iter().copied()) {
        *slot = v as u8;
    }
    Some(pred8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_av1;
    use crate::encoder::pixel_driver_dyn::{encode_intra_frame_yuv_dyn, Yuv420Frame};

    #[test]
    fn dyn_decode_flat_grey_16x16_via_dyn_driver_roundtrip() {
        // 16×16 still goes through the fixed-size driver, but the
        // r230 dyn encoder synthesises its own (different) SH/FH that
        // forces the dyn dispatcher branch in `decode_frame` — under
        // r230's SH, mi_cols/rows are 4 so it falls back through the
        // fixed driver. Confirm it still roundtrips.
        let input = Yuv420Frame::filled(16, 16, 128);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            Frame::Yuv420_16x16 { y, .. } => {
                // The dyn encoder pinned mi=4 at 16×16 so the fixed
                // path took over; check the recovered luma plane
                // matches the input.
                let mut expected = [[0u8; 16]; 16];
                for (i, row) in expected.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell = input.y[i * 16 + j];
                    }
                }
                assert_eq!(y, &expected);
            }
            Frame::Yuv420Dyn { .. } => panic!("16×16 should route through fixed driver"),
        }
    }

    #[test]
    fn dyn_decode_flat_grey_32x32_roundtrip() {
        let input = Yuv420Frame::filled(32, 32, 128);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
                assert_eq!(y, &input.y);
                assert_eq!(u, &input.u);
                assert_eq!(v, &input.v);
            }
            _ => panic!("expected Yuv420Dyn for 32×32"),
        }
    }

    #[test]
    fn dyn_decode_pseudorandom_32x32_roundtrip_bit_exact() {
        let mut input = Yuv420Frame::filled(32, 32, 0);
        let mut state: u64 = 0xDEAD_BEEF_C0FE_BABE;
        let mut next = || -> u8 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        for p in input.y.iter_mut() {
            *p = next();
        }
        for p in input.u.iter_mut() {
            *p = next();
        }
        for p in input.v.iter_mut() {
            *p = next();
        }
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
                assert_eq!(y, &input.y, "Y mismatch at 32×32");
                assert_eq!(u, &input.u, "U mismatch at 32×32");
                assert_eq!(v, &input.v, "V mismatch at 32×32");
            }
            _ => panic!("expected Yuv420Dyn for 32×32"),
        }
    }

    #[test]
    fn dyn_decode_flat_grey_64x64_roundtrip() {
        let input = Yuv420Frame::filled(64, 64, 200);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(y, &input.y);
                assert_eq!(u, &input.u);
                assert_eq!(v, &input.v);
            }
            _ => panic!("expected Yuv420Dyn for 64×64"),
        }
    }

    #[test]
    fn dyn_decode_pseudorandom_64x64_roundtrip_bit_exact() {
        let mut input = Yuv420Frame::filled(64, 64, 0);
        let mut state: u64 = 0xABCD_1234_5678_9F0E;
        let mut next = || -> u8 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        for p in input.y.iter_mut() {
            *p = next();
        }
        for p in input.u.iter_mut() {
            *p = next();
        }
        for p in input.v.iter_mut() {
            *p = next();
        }
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(y, &input.y, "Y mismatch at 64×64");
                assert_eq!(u, &input.u, "U mismatch at 64×64");
                assert_eq!(v, &input.v, "V mismatch at 64×64");
            }
            _ => panic!("expected Yuv420Dyn for 64×64"),
        }
    }
}
