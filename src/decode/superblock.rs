//! AV1 superblock + leaf-block mode decoder + reconstruction —
//! §5.11.4 + §5.11.8 + §5.11.18 + §5.11.39 + §7.7 + §7.11.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/superblock.go`
//! (MIT, KarpelesLab/goavif). Phase 5 wires the full intra predictor
//! set (DC/V/H + 6 directional + 3 smooth + Paeth + CFL) and native
//! 10/12-bit HBD paths; the pre-Phase-5 DC_PRED fallback is gone.

use oxideav_core::{Error, Result};

use crate::predict::intra::{
    cfl_pred, cfl_pred16, cfl_subsample, cfl_subsample16, dc_pred, dc_pred16, directional_pred,
    directional_pred16, h_pred, h_pred16, mode_to_angle_map, paeth_pred, paeth_pred16,
    smooth_h_pred, smooth_h_pred16, smooth_pred, smooth_pred16, smooth_v_pred, smooth_v_pred16,
    v_pred, v_pred16,
};
use crate::quant;
use crate::transform::{clamped_scan, default_zigzag_scan, inverse_2d, TxSize, TxType};

use super::block::{
    block_size_log, half_below_size, horz4_size, quarter_size, vert4_size, BlockSize, PartitionType,
};
use super::coeffs::{decode_coefficients, nz_map_ctx_offset, tx_size_idx};
use super::frame_state::FrameState;
use super::inter_block::{
    decode_inter_block_syntax, mc_chroma_u16, mc_chroma_u8, mc_luma_u16, mc_luma_u8,
};
use super::modes::{mode_ctx_bucket, IntraMode};
use super::mv::Mv;
use super::reconstruct::{clip_add_in_place, clip_add_in_place16};
use super::tile::{cfl_alpha_ctx, cfl_signs, segment_id_ctx, TileDecoder};
use crate::frame_header::FrameType;

/// Decode one superblock at luma-sample position `(sb_x, sb_y)`.
///
/// Matches libaom's `decode_partition()` entry: before reading any
/// partition / mode symbols we honour the spec §5.11.40 per-LR-unit
/// signalling for each plane whose `FrameRestorationType != NONE`,
/// consulting the tile-scoped `switchable_restore_cdf` /
/// `wiener_restore_cdf` / `sgrproj_restore_cdf`.
pub fn decode_superblock(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    sb_x: u32,
    sb_y: u32,
) -> Result<()> {
    let sb_bs = if td.sb_size == 128 {
        BlockSize::Block128x128
    } else {
        BlockSize::Block64x64
    };
    read_lr_unit_coeffs_for_sb(td, fs, sb_x, sb_y)?;
    decode_partition_node(td, fs, sb_x, sb_y, sb_bs)
}

/// For each plane, decode the per-unit LR parameters for every
/// restoration unit whose top-left corner lies inside the `sb_size`
/// superblock anchored at `(sb_x, sb_y)`. Mirrors libaom's
/// `av1_loop_restoration_corners_in_sb` loop (§5.11.40).
fn read_lr_unit_coeffs_for_sb(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    sb_x: u32,
    sb_y: u32,
) -> Result<()> {
    let num_planes = td.seq.color_config.num_planes as usize;
    let sb_size = td.sb_size;
    for plane in 0..num_planes {
        let rt = td.frame.lr.frame_restoration_type[plane];
        if rt == crate::frame_header_tail::RESTORATION_NONE {
            continue;
        }
        let unit_size = fs.lr_unit_size[plane];
        if unit_size == 0 {
            continue;
        }
        // Convert SB luma extent into plane-local samples — chroma is
        // subsampled via the plane's `sub_x / sub_y`.
        let (sub_x, sub_y) = if plane == 0 {
            (0u32, 0u32)
        } else {
            (fs.sub_x, fs.sub_y)
        };
        let plane_x_start = sb_x >> sub_x;
        let plane_y_start = sb_y >> sub_y;
        let plane_x_end = ((sb_x + sb_size) >> sub_x).min(plane_width(fs, plane));
        let plane_y_end = ((sb_y + sb_size) >> sub_y).min(plane_height(fs, plane));
        if plane_x_start >= plane_x_end || plane_y_start >= plane_y_end {
            continue;
        }
        // Compute the LR-unit column/row range whose top-left corner
        // is inside the SB's plane rectangle. Since unit corners land
        // on multiples of `unit_size`, we pick all units `u` such that
        // `u * unit_size ∈ [plane_start, plane_end)` — i.e. the
        // half-open range `[div_ceil_start, div_end)`.
        let col_start = div_ceil_or_zero(plane_x_start, unit_size);
        let col_end = div_ceil_or_zero(plane_x_end, unit_size);
        let row_start = div_ceil_or_zero(plane_y_start, unit_size);
        let row_end = div_ceil_or_zero(plane_y_end, unit_size);
        let col_end = col_end.min(fs.lr_cols[plane]);
        let row_end = row_end.min(fs.lr_rows[plane]);
        for rrow in row_start..row_end {
            for rcol in col_start..col_end {
                let params = td.decode_lr_unit(plane)?;
                *fs.lr_unit_mut(plane, rcol, rrow) = params;
            }
        }
    }
    Ok(())
}

fn plane_width(fs: &FrameState, plane: usize) -> u32 {
    if plane == 0 {
        fs.width
    } else {
        fs.uv_width
    }
}

fn plane_height(fs: &FrameState, plane: usize) -> u32 {
    if plane == 0 {
        fs.height
    } else {
        fs.uv_height
    }
}

/// Ceiling division `⌈a / b⌉`, returning 0 when `b == 0`.
fn div_ceil_or_zero(a: u32, b: u32) -> u32 {
    if b == 0 {
        0
    } else {
        a.div_ceil(b)
    }
}

/// Recursively decode a partition node at `(x, y)` of size `bs`.
pub fn decode_partition_node(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bs: BlockSize,
) -> Result<()> {
    if x >= fs.width || y >= fs.height {
        return Ok(());
    }
    if bs == BlockSize::Block4x4 {
        return decode_leaf_block(td, fs, x, y, bs);
    }
    if !bs.is_square() {
        return decode_leaf_block(td, fs, x, y, bs);
    }

    let w = bs.width();
    let h = bs.height();
    let hw = w / 2;
    let hh = h / 2;

    let bsl = block_size_log(bs);
    let above_ctx = 0u32;
    let left_ctx = 0u32;
    let pt_raw = td.decode_partition(bsl, above_ctx * 2 + left_ctx)?;
    let pt = PartitionType::from_u32(pt_raw).ok_or_else(|| {
        Error::invalid(format!("av1 partition: invalid symbol {pt_raw} (§5.11.4)"))
    })?;

    match pt {
        PartitionType::None => decode_leaf_block(td, fs, x, y, bs),
        PartitionType::Horz => {
            let top = half_below_size(bs, true);
            decode_leaf_block(td, fs, x, y, top)?;
            if y + hh < fs.height {
                let bot = half_below_size(bs, true);
                decode_leaf_block(td, fs, x, y + hh, bot)?;
            }
            Ok(())
        }
        PartitionType::Vert => {
            let left = half_below_size(bs, false);
            decode_leaf_block(td, fs, x, y, left)?;
            if x + hw < fs.width {
                let right = half_below_size(bs, false);
                decode_leaf_block(td, fs, x + hw, y, right)?;
            }
            Ok(())
        }
        PartitionType::Split => {
            let sub = quarter_size(bs);
            decode_partition_node(td, fs, x, y, sub)?;
            decode_partition_node(td, fs, x + hw, y, sub)?;
            decode_partition_node(td, fs, x, y + hh, sub)?;
            decode_partition_node(td, fs, x + hw, y + hh, sub)
        }
        PartitionType::HorzA => {
            let sub = quarter_size(bs);
            let bot = half_below_size(bs, true);
            decode_leaf_block(td, fs, x, y, sub)?;
            decode_leaf_block(td, fs, x + hw, y, sub)?;
            decode_leaf_block(td, fs, x, y + hh, bot)
        }
        PartitionType::HorzB => {
            let top = half_below_size(bs, true);
            let sub = quarter_size(bs);
            decode_leaf_block(td, fs, x, y, top)?;
            decode_leaf_block(td, fs, x, y + hh, sub)?;
            decode_leaf_block(td, fs, x + hw, y + hh, sub)
        }
        PartitionType::VertA => {
            let sub = quarter_size(bs);
            let right = half_below_size(bs, false);
            decode_leaf_block(td, fs, x, y, sub)?;
            decode_leaf_block(td, fs, x, y + hh, sub)?;
            decode_leaf_block(td, fs, x + hw, y, right)
        }
        PartitionType::VertB => {
            let left = half_below_size(bs, false);
            let sub = quarter_size(bs);
            decode_leaf_block(td, fs, x, y, left)?;
            decode_leaf_block(td, fs, x + hw, y, sub)?;
            decode_leaf_block(td, fs, x + hw, y + hh, sub)
        }
        PartitionType::Horz4 => {
            let qh = h / 4;
            let row_bs = horz4_size(bs);
            for i in 0..4 {
                let yy = y + i * qh;
                if yy >= fs.height {
                    break;
                }
                decode_leaf_block(td, fs, x, yy, row_bs)?;
            }
            Ok(())
        }
        PartitionType::Vert4 => {
            let qw = w / 4;
            let col_bs = vert4_size(bs);
            for i in 0..4 {
                let xx = x + i * qw;
                if xx >= fs.width {
                    break;
                }
                decode_leaf_block(td, fs, xx, y, col_bs)?;
            }
            Ok(())
        }
    }
}

/// Decode a single leaf coding block — §5.11.8 `decode_block`.
pub fn decode_leaf_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bs: BlockSize,
) -> Result<()> {
    let w = bs.width();
    let h = bs.height();
    let bw = w.min(fs.width.saturating_sub(x));
    let bh = h.min(fs.height.saturating_sub(y));
    if bw == 0 || bh == 0 {
        return Ok(());
    }

    // Inter frames route through the inter leaf decoder; intra / key /
    // intra-only frames use the original intra-only path below.
    if matches!(td.frame.frame_type, FrameType::Inter | FrameType::Switch) {
        return decode_inter_leaf_block(td, fs, x, y, bw, bh);
    }

    let mi_col = x >> 2;
    let mi_row = y >> 2;

    let have_above_mi = mi_row > 0 && mi_row - 1 < fs.mi_rows && mi_col < fs.mi_cols;
    let have_left_mi = mi_col > 0 && mi_col - 1 < fs.mi_cols && mi_row < fs.mi_rows;

    let above_mode = if have_above_mi {
        fs.mi_at(mi_col, mi_row - 1)
            .mode
            .unwrap_or(IntraMode::DcPred)
    } else {
        IntraMode::DcPred
    };
    let left_mode = if have_left_mi {
        fs.mi_at(mi_col - 1, mi_row)
            .mode
            .unwrap_or(IntraMode::DcPred)
    } else {
        IntraMode::DcPred
    };
    let above_seg = if have_above_mi {
        fs.mi_at(mi_col, mi_row - 1).segment_id
    } else {
        0
    };
    let left_seg = if have_left_mi {
        fs.mi_at(mi_col - 1, mi_row).segment_id
    } else {
        0
    };

    let segment_id = if td.frame.segmentation.enabled && td.frame.segmentation.update_map {
        let ctx = segment_id_ctx(above_seg, left_seg, have_above_mi, have_left_mi);
        td.decode_segment_id(ctx)?
    } else {
        0
    };

    let y_mode = td.decode_intra_y_mode(mode_ctx_bucket(above_mode), mode_ctx_bucket(left_mode))?;
    let angle_delta_y = if y_mode.is_directional() {
        let dir_idx = (y_mode as u32) - (IntraMode::D45Pred as u32);
        td.decode_angle_delta(dir_idx)? as i8
    } else {
        0
    };

    let skip = td.decode_skip(0)?;

    let num_planes = td.seq.color_config.num_planes;
    let mut uv_mode = y_mode;
    let mut angle_delta_uv: i8 = 0;
    let mut cfl_alpha_u: i32 = 0;
    let mut cfl_alpha_v: i32 = 0;
    if !fs.monochrome && num_planes >= 3 {
        let cfl_allowed = true;
        uv_mode = td.decode_uv_mode(y_mode, cfl_allowed)?;
        if uv_mode.is_directional() {
            let dir_idx = (uv_mode as u32) - (IntraMode::D45Pred as u32);
            angle_delta_uv = td.decode_angle_delta(dir_idx)? as i8;
        }
        if uv_mode == IntraMode::CflPred {
            let joint = td.decode_cfl_sign()?;
            let (su, sv) = cfl_signs(joint);
            let mag_u = if su != 0 {
                (td.decode_cfl_alpha(cfl_alpha_ctx(joint, 0))? as i32) + 1
            } else {
                0
            };
            let mag_v = if sv != 0 {
                (td.decode_cfl_alpha(cfl_alpha_ctx(joint, 1))? as i32) + 1
            } else {
                0
            };
            cfl_alpha_u = su * mag_u;
            cfl_alpha_v = sv * mag_v;
        }
    }

    // Propagate mode info to every MI cell the block covers.
    let mi_w = (bw + 3) >> 2;
    let mi_h = (bh + 3) >> 2;
    let stored_uv = if fs.monochrome { None } else { Some(uv_mode) };
    for mr in 0..mi_h {
        for mc in 0..mi_w {
            let cell_col = mi_col + mc;
            let cell_row = mi_row + mr;
            if cell_col >= fs.mi_cols || cell_row >= fs.mi_rows {
                continue;
            }
            let mi = fs.mi_mut(cell_col, cell_row);
            mi.mode = Some(y_mode);
            mi.uv_mode = stored_uv;
            mi.skip = skip;
            mi.segment_id = segment_id;
            mi.angle_delta = angle_delta_y;
            mi.angle_delta_uv = angle_delta_uv;
            mi.cfl_alpha_u = cfl_alpha_u;
            mi.cfl_alpha_v = cfl_alpha_v;
        }
    }

    if td.frame.quant.using_qmatrix {
        return Err(Error::unsupported(
            "av1 quantization matrices (§5.9.12) pending",
        ));
    }

    // Luma path.
    reconstruct_luma_block(
        td,
        fs,
        x,
        y,
        bw,
        bh,
        y_mode,
        angle_delta_y,
        skip,
        segment_id,
    )?;

    // Chroma path.
    if !fs.monochrome && num_planes >= 3 {
        reconstruct_chroma_block(
            td,
            fs,
            x,
            y,
            bw,
            bh,
            uv_mode,
            angle_delta_uv,
            skip,
            cfl_alpha_u,
            cfl_alpha_v,
            segment_id,
        )?;
    }

    Ok(())
}

/// Inter-frame leaf decode — reads block-level inter syntax, runs MC
/// into a prediction buffer, then performs the existing residual /
/// clip-add path on top. Intra-within-inter blocks surface
/// `Error::Unsupported` in the narrow Phase 7 scope.
fn decode_inter_leaf_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bw: u32,
    bh: u32,
) -> Result<()> {
    let info = decode_inter_block_syntax(td, fs, x, y, bw, bh)?;
    let w = bw as usize;
    let h = bh as usize;
    let mi_col = x >> 2;
    let mi_row = y >> 2;

    if !info.is_inter {
        // Intra-within-inter: propagate mode info and reconstruct via
        // the same intra pipeline used on key frames, with DC_PRED
        // fallback for chroma (no angle_delta / CFL signalling).
        let mi_w = (bw + 3) >> 2;
        let mi_h = (bh + 3) >> 2;
        let stored_uv = if fs.monochrome {
            None
        } else {
            Some(IntraMode::DcPred)
        };
        for mr in 0..mi_h {
            for mc in 0..mi_w {
                let cell_col = mi_col + mc;
                let cell_row = mi_row + mr;
                if cell_col >= fs.mi_cols || cell_row >= fs.mi_rows {
                    continue;
                }
                let mi = fs.mi_mut(cell_col, cell_row);
                mi.mode = Some(info.intra_y_mode);
                mi.uv_mode = stored_uv;
                mi.skip = info.skip;
                mi.segment_id = 0;
                mi.angle_delta = 0;
                mi.angle_delta_uv = 0;
                mi.cfl_alpha_u = 0;
                mi.cfl_alpha_v = 0;
                mi.is_inter = false;
                mi.mv_row = 0;
                mi.mv_col = 0;
            }
        }
        reconstruct_luma_block(td, fs, x, y, bw, bh, info.intra_y_mode, 0, info.skip, 0)?;
        if !fs.monochrome && td.seq.color_config.num_planes >= 3 {
            reconstruct_chroma_block(
                td,
                fs,
                x,
                y,
                bw,
                bh,
                IntraMode::DcPred,
                0,
                info.skip,
                0,
                0,
                0,
            )?;
        }
        return Ok(());
    }

    if td.prev_frame.is_none() {
        return Err(Error::unsupported(
            "av1 inter: missing reference frame for LAST-ref translational MC (§7.11.3)",
        ));
    }

    // Record mode info on MI grid so neighbor contexts for subsequent
    // blocks are accurate.
    let mi_w = (bw + 3) >> 2;
    let mi_h = (bh + 3) >> 2;
    for mr in 0..mi_h {
        for mc in 0..mi_w {
            let cell_col = mi_col + mc;
            let cell_row = mi_row + mr;
            if cell_col >= fs.mi_cols || cell_row >= fs.mi_rows {
                continue;
            }
            let mi = fs.mi_mut(cell_col, cell_row);
            mi.mode = None;
            mi.uv_mode = None;
            mi.skip = info.skip;
            mi.segment_id = 0;
            mi.angle_delta = 0;
            mi.angle_delta_uv = 0;
            mi.cfl_alpha_u = 0;
            mi.cfl_alpha_v = 0;
            mi.is_inter = true;
            mi.mv_row = info.mv.row;
            mi.mv_col = info.mv.col;
        }
    }

    reconstruct_inter_luma_block(td, fs, x, y, w, h, info.mv, info.skip, info.interp_filter)?;

    if !fs.monochrome && td.seq.color_config.num_planes >= 3 {
        reconstruct_inter_chroma_block(td, fs, x, y, w, h, info.mv, info.skip, info.interp_filter)?;
    }
    Ok(())
}

/// Motion-compensate + residual-add for the luma plane on an inter
/// leaf block.
#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_luma_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    w: usize,
    h: usize,
    mv: Mv,
    skip: bool,
    filt: crate::predict::interp::InterpFilter,
) -> Result<()> {
    let stride = fs.width as usize;
    let prev = td
        .prev_frame
        .as_ref()
        .ok_or_else(|| Error::invalid("av1 inter: prev_frame checked earlier"))?;
    if fs.bit_depth == 8 {
        let pred = mc_luma_u8(
            &prev.y_plane,
            prev.width as usize,
            prev.height as usize,
            prev.width as usize,
            x as i32,
            y as i32,
            w,
            h,
            mv,
            filt,
        );
        paste_block(&mut fs.y_plane, stride, x as usize, y as usize, &pred, w, h);
    } else {
        let pred = mc_luma_u16(
            &prev.y_plane16,
            prev.width as usize,
            prev.height as usize,
            prev.width as usize,
            x as i32,
            y as i32,
            w,
            h,
            mv,
            filt,
            fs.bit_depth,
        );
        paste_block16(
            &mut fs.y_plane16,
            stride,
            x as usize,
            y as usize,
            &pred,
            w,
            h,
        );
    }

    if skip {
        return Ok(());
    }

    let (sz, num_coeffs, scan) = select_square_tx(w, h)?;
    let tx_idx = tx_size_idx(w, h)?;
    let nz = nz_map_ctx_offset(w, h)?;
    let mut coeffs = decode_coefficients(
        &mut td.symbol,
        &mut td.coeff_bank,
        tx_idx,
        0,
        num_coeffs,
        &scan,
        nz,
        w,
        h,
    )?;
    let qv = quant::Params {
        base_q_idx: td.frame.quant.base_q_idx as i32,
        delta_q_y_dc: td.frame.quant.delta_q_y_dc as i32,
        delta_q_u_dc: td.frame.quant.delta_q_u_dc as i32,
        delta_q_u_ac: td.frame.quant.delta_q_u_ac as i32,
        delta_q_v_dc: td.frame.quant.delta_q_v_dc as i32,
        delta_q_v_ac: td.frame.quant.delta_q_v_ac as i32,
        bit_depth: fs.bit_depth,
    }
    .compute(quant::Plane::Y)?;
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c = dequant_coeff(*c, i, qv);
    }
    inverse_2d(&mut coeffs, TxType::DctDct, sz)?;
    let shift = residual_shift(w, h);
    for v in coeffs.iter_mut() {
        *v = round_shift(*v, shift);
    }
    if fs.bit_depth == 8 {
        let mut block = vec![0u8; w * h];
        extract_block(
            &fs.y_plane,
            stride,
            x as usize,
            y as usize,
            w,
            h,
            &mut block,
        );
        clip_add_in_place(&mut block, &coeffs, w, h);
        paste_block(
            &mut fs.y_plane,
            stride,
            x as usize,
            y as usize,
            &block,
            w,
            h,
        );
    } else {
        let mut block = vec![0u16; w * h];
        extract_block16(
            &fs.y_plane16,
            stride,
            x as usize,
            y as usize,
            w,
            h,
            &mut block,
        );
        clip_add_in_place16(&mut block, &coeffs, w, h, fs.bit_depth);
        paste_block16(
            &mut fs.y_plane16,
            stride,
            x as usize,
            y as usize,
            &block,
            w,
            h,
        );
    }
    Ok(())
}

/// MC + residual for the chroma planes on an inter leaf block.
#[allow(clippy::too_many_arguments)]
fn reconstruct_inter_chroma_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    w: usize,
    h: usize,
    mv: Mv,
    skip: bool,
    filt: crate::predict::interp::InterpFilter,
) -> Result<()> {
    let sub_x = fs.sub_x;
    let sub_y = fs.sub_y;
    let cx = (x >> sub_x) as usize;
    let cy = (y >> sub_y) as usize;
    let cw = ((w as u32 >> sub_x).max(1)) as usize;
    let ch = ((h as u32 >> sub_y).max(1)) as usize;
    let uvw = fs.uv_width as usize;
    let uvh = fs.uv_height as usize;
    let cw_clip = cw.min(uvw.saturating_sub(cx));
    let ch_clip = ch.min(uvh.saturating_sub(cy));
    if cw_clip == 0 || ch_clip == 0 {
        return Ok(());
    }

    let prev = td
        .prev_frame
        .as_ref()
        .ok_or_else(|| Error::invalid("av1 inter: prev_frame checked earlier"))?;
    let stride = uvw;

    for plane_idx in 0..2u32 {
        let pred = if fs.bit_depth == 8 {
            let ref_plane = if plane_idx == 0 {
                &prev.u_plane
            } else {
                &prev.v_plane
            };
            Some(mc_chroma_u8(
                ref_plane,
                prev.uv_width as usize,
                prev.uv_height as usize,
                prev.uv_width as usize,
                cx as i32,
                cy as i32,
                cw_clip,
                ch_clip,
                mv,
                sub_x,
                sub_y,
                filt,
            ))
        } else {
            None
        };
        let pred16 = if fs.bit_depth > 8 {
            let ref_plane = if plane_idx == 0 {
                &prev.u_plane16
            } else {
                &prev.v_plane16
            };
            Some(mc_chroma_u16(
                ref_plane,
                prev.uv_width as usize,
                prev.uv_height as usize,
                prev.uv_width as usize,
                cx as i32,
                cy as i32,
                cw_clip,
                ch_clip,
                mv,
                sub_x,
                sub_y,
                filt,
                fs.bit_depth,
            ))
        } else {
            None
        };

        if fs.bit_depth == 8 {
            let buf = pred.expect("u8 chroma predicted");
            if plane_idx == 0 {
                paste_block(&mut fs.u_plane, stride, cx, cy, &buf, cw_clip, ch_clip);
            } else {
                paste_block(&mut fs.v_plane, stride, cx, cy, &buf, cw_clip, ch_clip);
            }
        } else {
            let buf = pred16.expect("u16 chroma predicted");
            if plane_idx == 0 {
                paste_block16(&mut fs.u_plane16, stride, cx, cy, &buf, cw_clip, ch_clip);
            } else {
                paste_block16(&mut fs.v_plane16, stride, cx, cy, &buf, cw_clip, ch_clip);
            }
        }

        if skip {
            continue;
        }

        // Chroma residual mirrors the intra chroma path.
        let (sz, num_coeffs, scan) = select_square_tx(cw_clip, ch_clip)?;
        let tx_idx = tx_size_idx(cw_clip, ch_clip)?;
        let nz = nz_map_ctx_offset(cw_clip, ch_clip)?;
        let mut coeffs = decode_coefficients(
            &mut td.symbol,
            &mut td.coeff_bank,
            tx_idx,
            1,
            num_coeffs,
            &scan,
            nz,
            cw_clip,
            ch_clip,
        )?;
        let plane = if plane_idx == 0 {
            quant::Plane::U
        } else {
            quant::Plane::V
        };
        let qv = quant::Params {
            base_q_idx: td.frame.quant.base_q_idx as i32,
            delta_q_y_dc: td.frame.quant.delta_q_y_dc as i32,
            delta_q_u_dc: td.frame.quant.delta_q_u_dc as i32,
            delta_q_u_ac: td.frame.quant.delta_q_u_ac as i32,
            delta_q_v_dc: td.frame.quant.delta_q_v_dc as i32,
            delta_q_v_ac: td.frame.quant.delta_q_v_ac as i32,
            bit_depth: fs.bit_depth,
        }
        .compute(plane)?;
        for (i, c) in coeffs.iter_mut().enumerate() {
            *c = dequant_coeff(*c, i, qv);
        }
        inverse_2d(&mut coeffs, TxType::DctDct, sz)?;
        let shift = residual_shift(cw_clip, ch_clip);
        for v in coeffs.iter_mut() {
            *v = round_shift(*v, shift);
        }
        if fs.bit_depth == 8 {
            let mut block = vec![0u8; cw_clip * ch_clip];
            let plane_buf = if plane_idx == 0 {
                &fs.u_plane
            } else {
                &fs.v_plane
            };
            extract_block(plane_buf, stride, cx, cy, cw_clip, ch_clip, &mut block);
            clip_add_in_place(&mut block, &coeffs, cw_clip, ch_clip);
            if plane_idx == 0 {
                paste_block(&mut fs.u_plane, stride, cx, cy, &block, cw_clip, ch_clip);
            } else {
                paste_block(&mut fs.v_plane, stride, cx, cy, &block, cw_clip, ch_clip);
            }
        } else {
            let mut block = vec![0u16; cw_clip * ch_clip];
            let plane_buf = if plane_idx == 0 {
                &fs.u_plane16
            } else {
                &fs.v_plane16
            };
            extract_block16(plane_buf, stride, cx, cy, cw_clip, ch_clip, &mut block);
            clip_add_in_place16(&mut block, &coeffs, cw_clip, ch_clip, fs.bit_depth);
            if plane_idx == 0 {
                paste_block16(&mut fs.u_plane16, stride, cx, cy, &block, cw_clip, ch_clip);
            } else {
                paste_block16(&mut fs.v_plane16, stride, cx, cy, &block, cw_clip, ch_clip);
            }
        }
    }
    Ok(())
}

/// Reconstruct the Y plane for a single leaf block: predict → (if not
/// skip) decode residual → inverse transform → clip-add into `fs.y_plane`.
///
/// AV1's max TX size is 64×64 (§5.11.27, Table `Max_Tx_Size_Rect`), so
/// blocks of 128×128 / 128×64 / 64×128 are split into a 2×2 / 2×1 / 1×2
/// grid of ≤ 64×64 TX units. Intra prediction — including neighbour
/// gathering and SMOOTH / PAETH tables — also runs at TX-unit
/// granularity (libaom `build_intra_predictors` uses tx_size dims, not
/// block dims; there are no SMOOTH weight tables beyond 64 samples).
/// Smaller blocks produce a single iteration, matching the pre-split
/// behaviour.
#[allow(clippy::too_many_arguments)]
fn reconstruct_luma_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bw: u32,
    bh: u32,
    y_mode: IntraMode,
    angle_delta: i8,
    skip: bool,
    segment_id: u8,
) -> Result<()> {
    let w = bw as usize;
    let h = bh as usize;

    // Block-level tx_type (§6.10.15): for blocks with area > 32×32 the
    // ext-tx set is 0 (implicit DCT_DCT, no CDF read), so decoding the
    // type once here covers every TX unit below without disturbing the
    // symbol stream. For blocks with area ≤ 32×32 there is exactly one
    // TX unit (since the block already fits inside a 64×64 TX), so
    // reading the type once remains spec-correct.
    let tx_type = if skip {
        // No coefficients to transform — the tx_type is irrelevant and
        // the symbol stream doesn't carry one anyway.
        TxType::DctDct
    } else {
        td.decode_intra_tx_type(w, h, y_mode)?
    };

    let (tx_w, tx_h) = tx_unit_dims(w, h);
    let cols = w / tx_w;
    let rows = h / tx_h;
    for ty in 0..rows {
        for tx in 0..cols {
            let ux = x as usize + tx * tx_w;
            let uy = y as usize + ty * tx_h;
            reconstruct_one_luma_tx_unit(
                td,
                fs,
                ux,
                uy,
                tx_w,
                tx_h,
                y_mode,
                angle_delta,
                skip,
                tx_type,
                segment_id,
            )?;
        }
    }
    Ok(())
}

/// Predict + (optional) decode residual for a single luma TX unit at
/// plane coordinates `(ux, uy)`, dimensions `tx_w × tx_h ≤ 64×64`. The
/// predictor reads from the already-reconstructed samples above / left
/// of this TX unit, so earlier TX units within the same block feed the
/// later ones' neighbours — matching libaom's incremental
/// `av1_predict_intra_block` per TX-size behaviour.
#[allow(clippy::too_many_arguments)]
fn reconstruct_one_luma_tx_unit(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    ux: usize,
    uy: usize,
    tx_w: usize,
    tx_h: usize,
    y_mode: IntraMode,
    angle_delta: i8,
    skip: bool,
    tx_type: TxType,
    segment_id: u8,
) -> Result<()> {
    let stride = fs.width as usize;

    // Predict + paste.
    if fs.bit_depth == 8 {
        let pred = run_intra_prediction_u8(
            &fs.y_plane,
            stride,
            fs.height as usize,
            ux,
            uy,
            tx_w,
            tx_h,
            y_mode,
            angle_delta,
            128,
        );
        paste_block(&mut fs.y_plane, stride, ux, uy, &pred, tx_w, tx_h);
    } else {
        let pred = run_intra_prediction_u16(
            &fs.y_plane16,
            stride,
            fs.height as usize,
            ux,
            uy,
            tx_w,
            tx_h,
            y_mode,
            angle_delta,
            1u16 << (fs.bit_depth - 1),
            fs.bit_depth,
        );
        paste_block16(&mut fs.y_plane16, stride, ux, uy, &pred, tx_w, tx_h);
    }

    if skip {
        return Ok(());
    }

    let (sz, num_coeffs, scan) = select_square_tx(tx_w, tx_h)?;
    let tx_idx = tx_size_idx(tx_w, tx_h)?;
    let nz = nz_map_ctx_offset(tx_w, tx_h)?;

    let mut coeffs = decode_coefficients(
        &mut td.symbol,
        &mut td.coeff_bank,
        tx_idx,
        0, /* luma */
        num_coeffs,
        &scan,
        nz,
        tx_w,
        tx_h,
    )?;

    let base_q = segmented_base_q(td, segment_id);
    let qv = quant::Params {
        base_q_idx: base_q,
        delta_q_y_dc: td.frame.quant.delta_q_y_dc as i32,
        delta_q_u_dc: td.frame.quant.delta_q_u_dc as i32,
        delta_q_u_ac: td.frame.quant.delta_q_u_ac as i32,
        delta_q_v_dc: td.frame.quant.delta_q_v_dc as i32,
        delta_q_v_ac: td.frame.quant.delta_q_v_ac as i32,
        bit_depth: fs.bit_depth,
    }
    .compute(quant::Plane::Y)?;
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c = dequant_coeff(*c, i, qv);
    }

    if let Err(e) = inverse_2d(&mut coeffs, tx_type, sz) {
        if matches!(e, Error::Unsupported(_)) {
            inverse_2d(&mut coeffs, TxType::DctDct, sz)?;
        } else {
            return Err(e);
        }
    }

    let shift = residual_shift(tx_w, tx_h);
    for v in coeffs.iter_mut() {
        *v = round_shift(*v, shift);
    }
    if fs.bit_depth == 8 {
        let mut block = vec![0u8; tx_w * tx_h];
        extract_block(&fs.y_plane, stride, ux, uy, tx_w, tx_h, &mut block);
        clip_add_in_place(&mut block, &coeffs, tx_w, tx_h);
        paste_block(&mut fs.y_plane, stride, ux, uy, &block, tx_w, tx_h);
    } else {
        let mut block = vec![0u16; tx_w * tx_h];
        extract_block16(&fs.y_plane16, stride, ux, uy, tx_w, tx_h, &mut block);
        clip_add_in_place16(&mut block, &coeffs, tx_w, tx_h, fs.bit_depth);
        paste_block16(&mut fs.y_plane16, stride, ux, uy, &block, tx_w, tx_h);
    }
    Ok(())
}

/// TX-unit dimensions for a leaf block of size `w × h` — per AV1's
/// `Max_Tx_Size_Rect` rule any dimension larger than 64 samples splits
/// into 64-wide (resp. 64-tall) TX units. Returns `(tx_w, tx_h)` such
/// that `w % tx_w == 0` and `h % tx_h == 0`.
#[inline]
fn tx_unit_dims(w: usize, h: usize) -> (usize, usize) {
    (w.min(64), h.min(64))
}

/// Pick the `(TxSize, num_coeffs, scan)` triple for the given block
/// dimensions.
fn select_square_tx(w: usize, h: usize) -> Result<(TxSize, usize, Vec<usize>)> {
    match (w, h) {
        (4, 4) => Ok((TxSize::Tx4x4, 16, default_zigzag_scan(4, 4))),
        (8, 8) => Ok((TxSize::Tx8x8, 64, default_zigzag_scan(8, 8))),
        (16, 16) => Ok((TxSize::Tx16x16, 256, default_zigzag_scan(16, 16))),
        (32, 32) => Ok((TxSize::Tx32x32, 1024, default_zigzag_scan(32, 32))),
        (64, 64) => Ok((TxSize::Tx64x64, 1024, clamped_scan(32, 32, 64))),
        (4, 8) => Ok((TxSize::Tx4x8, 32, default_zigzag_scan(4, 8))),
        (8, 4) => Ok((TxSize::Tx8x4, 32, default_zigzag_scan(8, 4))),
        (8, 16) => Ok((TxSize::Tx8x16, 128, default_zigzag_scan(8, 16))),
        (16, 8) => Ok((TxSize::Tx16x8, 128, default_zigzag_scan(16, 8))),
        (16, 32) => Ok((TxSize::Tx16x32, 512, default_zigzag_scan(16, 32))),
        (32, 16) => Ok((TxSize::Tx32x16, 512, default_zigzag_scan(32, 16))),
        (32, 64) => Ok((TxSize::Tx32x64, 1024, default_zigzag_scan(32, 32))),
        (64, 32) => Ok((TxSize::Tx64x32, 1024, clamped_scan(32, 32, 64))),
        (4, 16) => Ok((TxSize::Tx4x16, 64, default_zigzag_scan(4, 16))),
        (16, 4) => Ok((TxSize::Tx16x4, 64, default_zigzag_scan(16, 4))),
        (8, 32) => Ok((TxSize::Tx8x32, 256, default_zigzag_scan(8, 32))),
        (32, 8) => Ok((TxSize::Tx32x8, 256, default_zigzag_scan(32, 8))),
        (16, 64) => Ok((TxSize::Tx16x64, 512, clamped_scan(16, 32, 16))),
        (64, 16) => Ok((TxSize::Tx64x16, 512, clamped_scan(32, 16, 64))),
        _ => Err(Error::unsupported(format!(
            "av1 superblock: TX {w}×{h} not in the AV1 set (§5.11.27)"
        ))),
    }
}

/// Chroma counterpart of [`reconstruct_luma_block`] covering both U
/// and V. Per-TX-unit prediction + residual mirrors libaom's
/// `av1_predict_intra_block` so SMOOTH / PAETH tables (capped at 64
/// samples) work for 128-wide blocks under 4:4:4 as well.
#[allow(clippy::too_many_arguments)]
fn reconstruct_chroma_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bw: u32,
    bh: u32,
    uv_mode: IntraMode,
    angle_delta_uv: i8,
    skip: bool,
    cfl_alpha_u: i32,
    cfl_alpha_v: i32,
    segment_id: u8,
) -> Result<()> {
    let sub_x = fs.sub_x;
    let sub_y = fs.sub_y;
    let cx = (x >> sub_x) as usize;
    let cy = (y >> sub_y) as usize;
    let cw = ((bw >> sub_x).max(1)) as usize;
    let ch = ((bh >> sub_y).max(1)) as usize;
    let uvw = fs.uv_width as usize;
    let uvh = fs.uv_height as usize;
    let cw_clip = cw.min(uvw.saturating_sub(cx));
    let ch_clip = ch.min(uvh.saturating_sub(cy));
    if cw_clip == 0 || ch_clip == 0 {
        return Ok(());
    }

    // For CFL we need the reconstructed luma block as the AC template.
    // Subsampling runs over the full block; per-TX slicing happens below.
    let cfl_luma_q3: Option<Vec<i32>> = if uv_mode == IntraMode::CflPred {
        let luma_w = bw as usize;
        let luma_h = bh as usize;
        let mut luma_tight = vec![0i32; cw_clip * ch_clip];
        if fs.bit_depth == 8 {
            let mut luma = vec![0u8; luma_w * luma_h];
            extract_block(
                &fs.y_plane,
                fs.width as usize,
                x as usize,
                y as usize,
                luma_w,
                luma_h,
                &mut luma,
            );
            cfl_subsample(
                &mut luma_tight,
                &luma,
                luma_w,
                luma_h,
                sub_x as usize,
                sub_y as usize,
            );
        } else {
            let mut luma = vec![0u16; luma_w * luma_h];
            extract_block16(
                &fs.y_plane16,
                fs.width as usize,
                x as usize,
                y as usize,
                luma_w,
                luma_h,
                &mut luma,
            );
            cfl_subsample16(
                &mut luma_tight,
                &luma,
                luma_w,
                luma_h,
                sub_x as usize,
                sub_y as usize,
            );
        }
        Some(luma_tight)
    } else {
        None
    };

    // For CFL, the base prediction is DC; then the luma AC is overlaid.
    // For other modes, run the predictor directly.
    let base_mode = if uv_mode == IntraMode::CflPred {
        IntraMode::DcPred
    } else {
        uv_mode
    };

    let (tx_w, tx_h) = tx_unit_dims(cw_clip, ch_clip);
    let cols = cw_clip / tx_w;
    let rows = ch_clip / tx_h;

    for plane_idx in 0..2u32 {
        let alpha = if plane_idx == 0 {
            cfl_alpha_u
        } else {
            cfl_alpha_v
        };

        for ty in 0..rows {
            for txi in 0..cols {
                let ux = cx + txi * tx_w;
                let uy = cy + ty * tx_h;
                // Extract the CFL luma slice corresponding to this TX
                // unit so the overlay only mixes in the aligned AC band.
                let cfl_tx_q3 = cfl_luma_q3.as_ref().map(|full| {
                    let mut tile = vec![0i32; tx_w * tx_h];
                    let col_off = txi * tx_w;
                    let row_off = ty * tx_h;
                    for r in 0..tx_h {
                        let src_row = (row_off + r) * cw_clip + col_off;
                        tile[r * tx_w..(r + 1) * tx_w]
                            .copy_from_slice(&full[src_row..src_row + tx_w]);
                    }
                    tile
                });
                reconstruct_one_chroma_tx_unit(
                    td,
                    fs,
                    plane_idx,
                    ux,
                    uy,
                    tx_w,
                    tx_h,
                    base_mode,
                    angle_delta_uv,
                    skip,
                    cfl_tx_q3.as_deref(),
                    alpha,
                    segment_id,
                )?;
            }
        }
    }

    Ok(())
}

/// Predict + (optional) decode residual for a single chroma TX unit at
/// plane coordinates `(ux, uy)`, dimensions `tx_w × tx_h ≤ 64×64`.
/// `plane_idx = 0` → U, `plane_idx = 1` → V. If `cfl_q3` is present it
/// carries the subsampled-luma AC template for this exact TX unit and
/// the predictor is CFL (base DC + luma overlay). Chroma residuals use
/// implicit `DCT_DCT` per §6.10.15.
#[allow(clippy::too_many_arguments)]
fn reconstruct_one_chroma_tx_unit(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    plane_idx: u32,
    ux: usize,
    uy: usize,
    tx_w: usize,
    tx_h: usize,
    base_mode: IntraMode,
    angle_delta_uv: i8,
    skip: bool,
    cfl_q3: Option<&[i32]>,
    alpha: i32,
    segment_id: u8,
) -> Result<()> {
    let stride = fs.uv_width as usize;
    let uvh = fs.uv_height as usize;

    // Predict + (CFL overlay) + paste.
    if fs.bit_depth == 8 {
        let plane_ref: &[u8] = if plane_idx == 0 {
            &fs.u_plane
        } else {
            &fs.v_plane
        };
        let mut pred = run_intra_prediction_u8(
            plane_ref,
            stride,
            uvh,
            ux,
            uy,
            tx_w,
            tx_h,
            base_mode,
            angle_delta_uv,
            128,
        );
        if let Some(q3) = cfl_q3 {
            let dc_copy = pred.clone();
            cfl_pred(&mut pred, tx_w, tx_h, q3, &dc_copy, alpha);
        }
        if plane_idx == 0 {
            paste_block(&mut fs.u_plane, stride, ux, uy, &pred, tx_w, tx_h);
        } else {
            paste_block(&mut fs.v_plane, stride, ux, uy, &pred, tx_w, tx_h);
        }
    } else {
        let plane_ref: &[u16] = if plane_idx == 0 {
            &fs.u_plane16
        } else {
            &fs.v_plane16
        };
        let mid = 1u16 << (fs.bit_depth - 1);
        let mut pred = run_intra_prediction_u16(
            plane_ref,
            stride,
            uvh,
            ux,
            uy,
            tx_w,
            tx_h,
            base_mode,
            angle_delta_uv,
            mid,
            fs.bit_depth,
        );
        if let Some(q3) = cfl_q3 {
            let dc_copy = pred.clone();
            cfl_pred16(&mut pred, tx_w, tx_h, q3, &dc_copy, alpha, fs.bit_depth);
        }
        if plane_idx == 0 {
            paste_block16(&mut fs.u_plane16, stride, ux, uy, &pred, tx_w, tx_h);
        } else {
            paste_block16(&mut fs.v_plane16, stride, ux, uy, &pred, tx_w, tx_h);
        }
    }

    if skip {
        return Ok(());
    }

    let (sz, num_coeffs, scan) = select_square_tx(tx_w, tx_h)?;
    let tx_idx = tx_size_idx(tx_w, tx_h)?;
    let nz = nz_map_ctx_offset(tx_w, tx_h)?;

    let mut coeffs = decode_coefficients(
        &mut td.symbol,
        &mut td.coeff_bank,
        tx_idx,
        1, /* chroma */
        num_coeffs,
        &scan,
        nz,
        tx_w,
        tx_h,
    )?;

    let base_q = segmented_base_q(td, segment_id);
    let plane = if plane_idx == 0 {
        quant::Plane::U
    } else {
        quant::Plane::V
    };
    let qv = quant::Params {
        base_q_idx: base_q,
        delta_q_y_dc: td.frame.quant.delta_q_y_dc as i32,
        delta_q_u_dc: td.frame.quant.delta_q_u_dc as i32,
        delta_q_u_ac: td.frame.quant.delta_q_u_ac as i32,
        delta_q_v_dc: td.frame.quant.delta_q_v_dc as i32,
        delta_q_v_ac: td.frame.quant.delta_q_v_ac as i32,
        bit_depth: fs.bit_depth,
    }
    .compute(plane)?;
    for (i, c) in coeffs.iter_mut().enumerate() {
        *c = dequant_coeff(*c, i, qv);
    }

    inverse_2d(&mut coeffs, TxType::DctDct, sz)?;

    let shift = residual_shift(tx_w, tx_h);
    for v in coeffs.iter_mut() {
        *v = round_shift(*v, shift);
    }

    if fs.bit_depth == 8 {
        let mut block = vec![0u8; tx_w * tx_h];
        let plane_buf = if plane_idx == 0 {
            &fs.u_plane
        } else {
            &fs.v_plane
        };
        extract_block(plane_buf, stride, ux, uy, tx_w, tx_h, &mut block);
        clip_add_in_place(&mut block, &coeffs, tx_w, tx_h);
        if plane_idx == 0 {
            paste_block(&mut fs.u_plane, stride, ux, uy, &block, tx_w, tx_h);
        } else {
            paste_block(&mut fs.v_plane, stride, ux, uy, &block, tx_w, tx_h);
        }
    } else {
        let mut block = vec![0u16; tx_w * tx_h];
        let plane_buf = if plane_idx == 0 {
            &fs.u_plane16
        } else {
            &fs.v_plane16
        };
        extract_block16(plane_buf, stride, ux, uy, tx_w, tx_h, &mut block);
        clip_add_in_place16(&mut block, &coeffs, tx_w, tx_h, fs.bit_depth);
        if plane_idx == 0 {
            paste_block16(&mut fs.u_plane16, stride, ux, uy, &block, tx_w, tx_h);
        } else {
            paste_block16(&mut fs.v_plane16, stride, ux, uy, &block, tx_w, tx_h);
        }
    }
    Ok(())
}

/// Run the u8 intra predictor against the reconstructed plane. Returns
/// a tight `w*h` buffer of predicted samples.
#[allow(clippy::too_many_arguments)]
fn run_intra_prediction_u8(
    plane: &[u8],
    plane_stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mode: IntraMode,
    angle_delta: i8,
    fallback: u8,
) -> Vec<u8> {
    // Extended neighbours for directional / filter-intra reads. Length
    // covers the longest projection: above + right of block.
    let ext_len = w + h + 4;
    let (above_raw, left_raw, above_left) =
        gather_neighbors_u8(plane, plane_stride, plane_height, x, y, ext_len, fallback);
    let have_above = y > 0;
    let have_left = x > 0;
    let above = &above_raw[..];
    let left = &left_raw[..];

    let mut dst = vec![0u8; w * h];
    match mode {
        IntraMode::DcPred | IntraMode::CflPred => {
            dc_pred(&mut dst, w, h, above, left, have_above, have_left, 8);
        }
        IntraMode::VPred => {
            v_pred(&mut dst, w, h, above);
        }
        IntraMode::HPred => {
            h_pred(&mut dst, w, h, left);
        }
        IntraMode::D45Pred
        | IntraMode::D67Pred
        | IntraMode::D113Pred
        | IntraMode::D135Pred
        | IntraMode::D157Pred
        | IntraMode::D203Pred => {
            let base = mode_to_angle_map(mode);
            let angle = base + (angle_delta as i32) * 3;
            directional_pred(&mut dst, w, h, above, left, angle);
        }
        IntraMode::SmoothPred => {
            smooth_pred(&mut dst, w, h, above, left);
        }
        IntraMode::SmoothVPred => {
            smooth_v_pred(&mut dst, w, h, above, left);
        }
        IntraMode::SmoothHPred => {
            smooth_h_pred(&mut dst, w, h, above, left);
        }
        IntraMode::PaethPred => {
            paeth_pred(&mut dst, w, h, above, left, above_left);
        }
    }
    dst
}

/// u16 counterpart of [`run_intra_prediction_u8`].
#[allow(clippy::too_many_arguments)]
fn run_intra_prediction_u16(
    plane: &[u16],
    plane_stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    mode: IntraMode,
    angle_delta: i8,
    fallback: u16,
    bit_depth: u32,
) -> Vec<u16> {
    let ext_len = w + h + 4;
    let (above_raw, left_raw, above_left) =
        gather_neighbors_u16(plane, plane_stride, plane_height, x, y, ext_len, fallback);
    let have_above = y > 0;
    let have_left = x > 0;
    let above = &above_raw[..];
    let left = &left_raw[..];

    let mut dst = vec![0u16; w * h];
    match mode {
        IntraMode::DcPred | IntraMode::CflPred => {
            dc_pred16(
                &mut dst, w, h, above, left, have_above, have_left, bit_depth,
            );
        }
        IntraMode::VPred => {
            v_pred16(&mut dst, w, h, above);
        }
        IntraMode::HPred => {
            h_pred16(&mut dst, w, h, left);
        }
        IntraMode::D45Pred
        | IntraMode::D67Pred
        | IntraMode::D113Pred
        | IntraMode::D135Pred
        | IntraMode::D157Pred
        | IntraMode::D203Pred => {
            let base = mode_to_angle_map(mode);
            let angle = base + (angle_delta as i32) * 3;
            directional_pred16(&mut dst, w, h, above, left, angle, bit_depth);
        }
        IntraMode::SmoothPred => {
            smooth_pred16(&mut dst, w, h, above, left);
        }
        IntraMode::SmoothVPred => {
            smooth_v_pred16(&mut dst, w, h, above, left);
        }
        IntraMode::SmoothHPred => {
            smooth_h_pred16(&mut dst, w, h, above, left);
        }
        IntraMode::PaethPred => {
            paeth_pred16(&mut dst, w, h, above, left, above_left, bit_depth);
        }
    }
    dst
}

/// Gather extended neighbour samples from a u8 plane. Returns
/// `(above, left, above_left)` where `above` has `ext_len` samples
/// (edge-replicated beyond the plane) and `left` has `ext_len` samples.
fn gather_neighbors_u8(
    plane: &[u8],
    plane_stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    ext_len: usize,
    fallback: u8,
) -> (Vec<u8>, Vec<u8>, u8) {
    let mut above = vec![fallback; ext_len];
    let mut left = vec![fallback; ext_len];
    if y > 0 {
        let row = (y - 1) * plane_stride;
        let max_x = plane_stride.saturating_sub(1);
        for (c, slot) in above.iter_mut().enumerate() {
            let sx = (x + c).min(max_x);
            *slot = plane[row + sx];
        }
    }
    if x > 0 {
        let max_y = plane_height.saturating_sub(1);
        for (r, slot) in left.iter_mut().enumerate() {
            let sy = (y + r).min(max_y);
            *slot = plane[sy * plane_stride + (x - 1)];
        }
    }
    let above_left = if x > 0 && y > 0 {
        plane[(y - 1) * plane_stride + (x - 1)]
    } else if y > 0 {
        plane[(y - 1) * plane_stride + x]
    } else if x > 0 {
        plane[y * plane_stride + (x - 1)]
    } else {
        fallback
    };
    (above, left, above_left)
}

/// u16 counterpart of [`gather_neighbors_u8`].
fn gather_neighbors_u16(
    plane: &[u16],
    plane_stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    ext_len: usize,
    fallback: u16,
) -> (Vec<u16>, Vec<u16>, u16) {
    let mut above = vec![fallback; ext_len];
    let mut left = vec![fallback; ext_len];
    if y > 0 {
        let row = (y - 1) * plane_stride;
        let max_x = plane_stride.saturating_sub(1);
        for (c, slot) in above.iter_mut().enumerate() {
            let sx = (x + c).min(max_x);
            *slot = plane[row + sx];
        }
    }
    if x > 0 {
        let max_y = plane_height.saturating_sub(1);
        for (r, slot) in left.iter_mut().enumerate() {
            let sy = (y + r).min(max_y);
            *slot = plane[sy * plane_stride + (x - 1)];
        }
    }
    let above_left = if x > 0 && y > 0 {
        plane[(y - 1) * plane_stride + (x - 1)]
    } else if y > 0 {
        plane[(y - 1) * plane_stride + x]
    } else if x > 0 {
        plane[y * plane_stride + (x - 1)]
    } else {
        fallback
    };
    (above, left, above_left)
}

/// Copy a `w×h` block into a plane at `(x, y)`.
fn paste_block(
    plane: &mut [u8],
    stride: usize,
    x: usize,
    y: usize,
    src: &[u8],
    w: usize,
    h: usize,
) {
    for r in 0..h {
        let dst_off = (y + r) * stride + x;
        plane[dst_off..dst_off + w].copy_from_slice(&src[r * w..(r + 1) * w]);
    }
}

/// Copy a `w×h` block out of a plane at `(x, y)`.
fn extract_block(
    plane: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    dst: &mut [u8],
) {
    for r in 0..h {
        let src_off = (y + r) * stride + x;
        dst[r * w..(r + 1) * w].copy_from_slice(&plane[src_off..src_off + w]);
    }
}

/// 16-bit counterpart of [`paste_block`].
fn paste_block16(
    plane: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    src: &[u16],
    w: usize,
    h: usize,
) {
    for r in 0..h {
        let dst_off = (y + r) * stride + x;
        plane[dst_off..dst_off + w].copy_from_slice(&src[r * w..(r + 1) * w]);
    }
}

/// 16-bit counterpart of [`extract_block`].
fn extract_block16(
    plane: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    dst: &mut [u16],
) {
    for r in 0..h {
        let src_off = (y + r) * stride + x;
        dst[r * w..(r + 1) * w].copy_from_slice(&plane[src_off..src_off + w]);
    }
}

/// Dequantise a single coefficient — DC for `pos == 0`, AC otherwise.
#[inline]
fn dequant_coeff(level: i32, pos: usize, qv: quant::Values) -> i32 {
    let q = if pos == 0 { qv.dc } else { qv.ac };
    if q == 0 {
        return 0;
    }
    level.saturating_mul(q as i32)
}

/// Round-shift used to scale the residual back to the pixel domain.
fn residual_shift(w: usize, h: usize) -> u32 {
    crate::transform::inverse_shift(w, h)
}

#[inline]
fn round_shift(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// Compute the segment-adjusted `base_q_index`.
fn segmented_base_q(td: &TileDecoder<'_>, segment_id: u8) -> i32 {
    let base = td.frame.quant.base_q_idx as i32;
    if !td.frame.segmentation.enabled {
        return base;
    }
    let sid = segment_id as usize;
    if sid >= td.frame.segmentation.feature_enabled.len() {
        return base;
    }
    if !td.frame.segmentation.feature_enabled[sid][0] {
        return base;
    }
    let q = base + td.frame.segmentation.feature_data[sid][0] as i32;
    q.clamp(0, 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    // TX-unit sizing covers the cases spelled out by §5.11.27 / Table
    // `Max_Tx_Size_Rect`: blocks up to 64 on either axis use a single
    // TX, blocks wider/taller than 64 split into ≤ 64-wide / ≤ 64-tall
    // TX units. Every block dim tested here is a multiple of the
    // resulting TX dim, so the loop in reconstruct_luma_block cleanly
    // iterates `(w / tx_w) × (h / tx_h)` units without remainders.
    #[test]
    fn tx_unit_dims_caps_at_64_samples() {
        // Small blocks → single TX, unchanged.
        assert_eq!(tx_unit_dims(4, 4), (4, 4));
        assert_eq!(tx_unit_dims(8, 16), (8, 16));
        assert_eq!(tx_unit_dims(32, 32), (32, 32));
        assert_eq!(tx_unit_dims(64, 64), (64, 64));
        assert_eq!(tx_unit_dims(64, 32), (64, 32));
        assert_eq!(tx_unit_dims(16, 64), (16, 64));

        // 128-wide / 128-tall blocks split along the oversized axis.
        assert_eq!(tx_unit_dims(128, 128), (64, 64));
        assert_eq!(tx_unit_dims(128, 64), (64, 64));
        assert_eq!(tx_unit_dims(64, 128), (64, 64));

        // The grid dimensions line up with the spec's Table
        // `Max_Tx_Size_Rect` expectation of 2×2 / 2×1 / 1×2 TX units.
        let (tw, th) = tx_unit_dims(128, 128);
        assert_eq!((128 / tw, 128 / th), (2, 2));
        let (tw, th) = tx_unit_dims(128, 64);
        assert_eq!((128 / tw, 64 / th), (2, 1));
        let (tw, th) = tx_unit_dims(64, 128);
        assert_eq!((64 / tw, 128 / th), (1, 2));
    }

    // Sanity: every spec-valid 64×64-or-less TX size that the intra
    // path may feed `select_square_tx` must succeed. 128×N does not
    // and must still be rejected — callers are expected to loop over
    // TX units instead of passing oversized dimensions directly.
    #[test]
    fn select_square_tx_rejects_oversized_blocks() {
        for &(w, h) in &[
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
            (4, 8),
            (8, 4),
            (64, 32),
            (32, 64),
            (16, 64),
            (64, 16),
        ] {
            assert!(
                select_square_tx(w, h).is_ok(),
                "expected TX {w}×{h} to be in the AV1 set"
            );
        }
        // 128 on either axis is outside the spec's TX set.
        assert!(select_square_tx(128, 128).is_err());
        assert!(select_square_tx(128, 64).is_err());
        assert!(select_square_tx(64, 128).is_err());
    }
}
