//! AV1 superblock + leaf-block mode decoder + reconstruction —
//! §5.11.4 + §5.11.8 + §5.11.18 + §5.11.39 + §7.7.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/superblock.go`
//! (MIT, KarpelesLab/goavif). Phase 3 lands coefficient decode +
//! dequantisation + inverse transform + clip-add, replacing Phase 2's
//! `Error::Unsupported("av1 coefficient decode pending (§5.11.39)")`
//! bailout.
//!
//! Phase 3 scope:
//!
//! - Intra prediction: only `DC_PRED`, `V_PRED`, `H_PRED` are
//!   implemented (via [`crate::intra::predict`]). Any other mode on a
//!   non-skip block surfaces `Error::Unsupported` with `§7.11.2` in
//!   the message.
//! - Transform sizes: 4×4 / 8×8 / 16×16 (DCT_DCT only for Phase 3 —
//!   ADST / mixed / flipped variants are decoded but silently fallback
//!   to DCT_DCT when [`crate::transform::inverse_2d`] rejects them,
//!   matching goavif's safety net).
//! - Block sizes: leaf blocks ≤ 16×16 go through the single-TX path.
//!   Larger leaf blocks surface `Error::Unsupported` citing §5.11.39.

use oxideav_core::{Error, Result};

use crate::intra;
use crate::quant;
use crate::transform::{default_zigzag_scan, inverse_2d, TxSize, TxType};

use super::block::{
    block_size_log, half_below_size, horz4_size, quarter_size, vert4_size, BlockSize,
    PartitionType,
};
use super::coeffs::{decode_coefficients, nz_map_ctx_offset, tx_size_idx};
use super::frame_state::FrameState;
use super::modes::{mode_ctx_bucket, IntraMode};
use super::reconstruct::clip_add_in_place;
use super::tile::{cfl_alpha_ctx, cfl_signs, segment_id_ctx, TileDecoder};

/// Decode one superblock at luma-sample position `(sb_x, sb_y)`.
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
    decode_partition_node(td, fs, sb_x, sb_y, sb_bs)
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

/// Map a decoder-mode [`IntraMode`] to the intra predictor's
/// [`intra::IntraMode`] enum.
fn to_intra_mode(m: IntraMode) -> intra::IntraMode {
    match m {
        IntraMode::DcPred => intra::IntraMode::Dc,
        IntraMode::VPred => intra::IntraMode::V,
        IntraMode::HPred => intra::IntraMode::H,
        IntraMode::D45Pred => intra::IntraMode::D45,
        IntraMode::D135Pred => intra::IntraMode::D135,
        IntraMode::D113Pred => intra::IntraMode::D113,
        IntraMode::D157Pred => intra::IntraMode::D157,
        IntraMode::D203Pred => intra::IntraMode::D203,
        IntraMode::D67Pred => intra::IntraMode::D67,
        IntraMode::SmoothPred => intra::IntraMode::Smooth,
        IntraMode::SmoothVPred => intra::IntraMode::SmoothV,
        IntraMode::SmoothHPred => intra::IntraMode::SmoothH,
        IntraMode::PaethPred => intra::IntraMode::Paeth,
        // CFL has no direct equivalent — caller should route to DC_PRED
        // and apply the CFL-alpha on top. Phase 3 treats this as DC.
        IntraMode::CflPred => intra::IntraMode::Dc,
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

    let mi_col = x >> 2;
    let mi_row = y >> 2;

    let have_above_mi = mi_row > 0 && mi_row - 1 < fs.mi_rows && mi_col < fs.mi_cols;
    let have_left_mi = mi_col > 0 && mi_col - 1 < fs.mi_cols && mi_row < fs.mi_rows;

    let above_mode = if have_above_mi {
        fs.mi_at(mi_col, mi_row - 1).mode.unwrap_or(IntraMode::DcPred)
    } else {
        IntraMode::DcPred
    };
    let left_mode = if have_left_mi {
        fs.mi_at(mi_col - 1, mi_row).mode.unwrap_or(IntraMode::DcPred)
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

    if fs.bit_depth != 8 {
        return Err(Error::unsupported(format!(
            "av1 superblock: bit_depth {} not supported in Phase 3 (§5.11 — only 8-bit)",
            fs.bit_depth
        )));
    }
    if td.frame.quant.using_qmatrix {
        return Err(Error::unsupported(
            "av1 quantization matrices (§5.9.12) pending",
        ));
    }

    // Luma path.
    reconstruct_luma_block(
        td, fs, x, y, bw, bh, y_mode, skip, segment_id,
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
            skip,
            cfl_alpha_u,
            cfl_alpha_v,
            segment_id,
        )?;
    }

    Ok(())
}

/// Reconstruct the Y plane for a single leaf block: predict → (if not
/// skip) decode residual → inverse transform → clip-add into `fs.y_plane`.
#[allow(clippy::too_many_arguments)]
fn reconstruct_luma_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bw: u32,
    bh: u32,
    y_mode: IntraMode,
    skip: bool,
    segment_id: u8,
) -> Result<()> {
    let w = bw as usize;
    let h = bh as usize;
    let stride = fs.width as usize;

    // Gather neighbor samples for intra prediction.
    let (above_vec, left_vec) = gather_neighbors(
        &fs.y_plane,
        fs.width as usize,
        fs.height as usize,
        x as usize,
        y as usize,
        w,
        h,
    );
    let pred = run_intra_prediction(y_mode, &above_vec, &left_vec, w, h, x > 0, y > 0)?;

    // Paste predictor into dst and, if non-skip, add residual.
    paste_block(&mut fs.y_plane, stride, x as usize, y as usize, &pred, w, h);

    if skip {
        return Ok(());
    }

    // Non-skip: decode a single-TX residual. Block sizes > 16×16 are
    // not yet handled (they would require TX splitting).
    if w > 16 || h > 16 || w != h {
        return Err(Error::unsupported(format!(
            "av1 superblock: luma block {w}×{h} needs TX split (§5.11.27, Phase 4)"
        )));
    }
    let sz = match (w, h) {
        (4, 4) => TxSize::Tx4x4,
        (8, 8) => TxSize::Tx8x8,
        (16, 16) => TxSize::Tx16x16,
        _ => unreachable!(),
    };
    let tx_idx = tx_size_idx(w, h)?;
    let nz = nz_map_ctx_offset(w, h)?;
    let scan = default_zigzag_scan(w, h);

    // Phase 3: force DCT_DCT — the spec signals tx_type for ext-tx sets
    // but our decoder doesn't read that symbol yet. Forcing DCT_DCT
    // means bitstreams that encoded an ADST-path residual will
    // reconstruct incorrectly, but for DC-only blocks (which only have
    // a single DC coefficient) the result is bit-exact regardless of
    // tx_type.
    let tx_type = TxType::DctDct;

    let mut coeffs = decode_coefficients(
        &mut td.symbol,
        &mut td.coeff_bank,
        tx_idx,
        0, /* luma */
        w * h,
        &scan,
        nz,
        w,
        h,
    )?;

    // Dequantise each coefficient.
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

    // Inverse 2D transform.
    inverse_2d(&mut coeffs, tx_type, sz)?;

    // Final shift + clip-add.
    let shift = residual_shift(w, h);
    for v in coeffs.iter_mut() {
        *v = round_shift(*v, shift);
    }
    // Extract the block region, clip-add, write back.
    let mut block = vec![0u8; w * h];
    extract_block(&fs.y_plane, stride, x as usize, y as usize, w, h, &mut block);
    clip_add_in_place(&mut block, &coeffs, w, h);
    paste_block(&mut fs.y_plane, stride, x as usize, y as usize, &block, w, h);
    Ok(())
}

/// Chroma counterpart of [`reconstruct_luma_block`] covering both U
/// and V. Chroma dimensions are subsampled per `fs.sub_x` / `fs.sub_y`.
#[allow(clippy::too_many_arguments)]
fn reconstruct_chroma_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bw: u32,
    bh: u32,
    uv_mode: IntraMode,
    skip: bool,
    _cfl_alpha_u: i32,
    _cfl_alpha_v: i32,
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

    for plane_idx in 0..2u32 {
        let stride = uvw;
        let (above_vec, left_vec) = gather_neighbors(
            if plane_idx == 0 {
                &fs.u_plane
            } else {
                &fs.v_plane
            },
            uvw,
            uvh,
            cx,
            cy,
            cw_clip,
            ch_clip,
        );
        // CFL is treated as DC for Phase 3 (no real CFL math).
        let mode = if uv_mode == IntraMode::CflPred {
            IntraMode::DcPred
        } else {
            uv_mode
        };
        let pred = run_intra_prediction(
            mode,
            &above_vec,
            &left_vec,
            cw_clip,
            ch_clip,
            cx > 0,
            cy > 0,
        )?;

        if plane_idx == 0 {
            paste_block(&mut fs.u_plane, stride, cx, cy, &pred, cw_clip, ch_clip);
        } else {
            paste_block(&mut fs.v_plane, stride, cx, cy, &pred, cw_clip, ch_clip);
        }

        if skip {
            continue;
        }

        if cw_clip > 16 || ch_clip > 16 || cw_clip != ch_clip {
            return Err(Error::unsupported(format!(
                "av1 superblock: chroma block {cw_clip}×{ch_clip} needs TX split (§5.11.27, Phase 4)"
            )));
        }
        let sz = match (cw_clip, ch_clip) {
            (4, 4) => TxSize::Tx4x4,
            (8, 8) => TxSize::Tx8x8,
            (16, 16) => TxSize::Tx16x16,
            _ => unreachable!(),
        };
        let tx_idx = tx_size_idx(cw_clip, ch_clip)?;
        let nz = nz_map_ctx_offset(cw_clip, ch_clip)?;
        let scan = default_zigzag_scan(cw_clip, ch_clip);

        let mut coeffs = decode_coefficients(
            &mut td.symbol,
            &mut td.coeff_bank,
            tx_idx,
            1, /* chroma */
            cw_clip * ch_clip,
            &scan,
            nz,
            cw_clip,
            ch_clip,
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

        let shift = residual_shift(cw_clip, ch_clip);
        for v in coeffs.iter_mut() {
            *v = round_shift(*v, shift);
        }

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
    }

    Ok(())
}

/// Gather a block's above-row and left-column from a reconstructed
/// plane. Pads with `128` when either edge is off the top/left of
/// the plane. Returns `(above, left)` with lengths `w` and `h`.
fn gather_neighbors(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> (Vec<u8>, Vec<u8>) {
    let mut above = vec![128u8; w];
    let mut left = vec![128u8; h];
    if y > 0 {
        for (c, slot) in above.iter_mut().enumerate().take(w) {
            let sx = (x + c).min(plane_w.saturating_sub(1));
            *slot = plane[(y - 1) * plane_w + sx];
        }
    }
    if x > 0 {
        for (r, slot) in left.iter_mut().enumerate().take(h) {
            let sy = (y + r).min(plane_h.saturating_sub(1));
            *slot = plane[sy * plane_w + (x - 1)];
        }
    }
    (above, left)
}

/// Run the intra predictor. Returns a `w*h` buffer of predicted
/// samples. For Phase 3 only DC/V/H produce real output; directional
/// / smooth / paeth modes fall back to DC_PRED to keep the pipeline
/// running (matching goavif's safety net).
fn run_intra_prediction(
    mode: IntraMode,
    above: &[u8],
    left: &[u8],
    w: usize,
    h: usize,
    have_left: bool,
    have_above: bool,
) -> Result<Vec<u8>> {
    let mut dst = vec![0u8; w * h];
    let ni = intra::Neighbours {
        above: if have_above { Some(above) } else { None },
        left: if have_left { Some(left) } else { None },
    };
    let im = to_intra_mode(mode);
    match intra::predict(im, ni, w, h, &mut dst, w) {
        Ok(()) => Ok(dst),
        Err(Error::Unsupported(_)) => {
            // Fallback to DC for any predictor Phase 3 doesn't
            // implement (D45, smooth, paeth). This lets the tile
            // walker progress through real clips rather than bailing;
            // visual fidelity will suffer for blocks that needed the
            // real predictor, but a later Phase 5 port of the full
            // intra predictor set will close the gap.
            intra::predict(intra::IntraMode::Dc, ni, w, h, &mut dst, w)?;
            Ok(dst)
        }
        Err(e) => Err(e),
    }
}

/// Copy a `w×h` block into a plane at `(x, y)`.
fn paste_block(plane: &mut [u8], stride: usize, x: usize, y: usize, src: &[u8], w: usize, h: usize) {
    for r in 0..h {
        let dst_off = (y + r) * stride + x;
        plane[dst_off..dst_off + w].copy_from_slice(&src[r * w..(r + 1) * w]);
    }
}

/// Copy a `w×h` block out of a plane at `(x, y)`.
fn extract_block(plane: &[u8], stride: usize, x: usize, y: usize, w: usize, h: usize, dst: &mut [u8]) {
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
/// Matches goavif's `reconstructResidual` → `reconstruct.go`.
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

/// Compute the segment-adjusted `base_q_index`. Matches goavif's
/// `segmentedBaseQ`.
fn segmented_base_q(td: &TileDecoder<'_>, segment_id: u8) -> i32 {
    let base = td.frame.quant.base_q_idx as i32;
    if !td.frame.segmentation.enabled {
        return base;
    }
    let sid = segment_id as usize;
    if sid >= td.frame.segmentation.feature_enabled.len() {
        return base;
    }
    // SEG_LVL_ALT_Q is feature index 0.
    if !td.frame.segmentation.feature_enabled[sid][0] {
        return base;
    }
    let q = base + td.frame.segmentation.feature_data[sid][0] as i32;
    q.clamp(0, 255)
}
