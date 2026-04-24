//! AV1 superblock + leaf-block mode decoder + reconstruction —
//! §5.11.4 + §5.11.8 + §5.11.18 + §5.11.39 + §7.7 + §7.11.
//!
//! Phase 5 wires the full intra predictor set (DC/V/H + 6 directional +
//! 3 smooth + Paeth + CFL) and native 10/12-bit HBD paths; the
//! pre-Phase-5 DC_PRED fallback is gone.
//!
//! # Round 8 status
//!
//! Round 8 closes three syntax-desync items flagged at the end of
//! round 7:
//!
//! - **`palette_mode_info()` (§5.11.46)** — now invoked from both the
//!   key-frame and intra-within-inter paths on palette-eligible
//!   blocks. When the encoder activates a Y or UV palette we surface
//!   `Error::Unsupported` (the palette-token + color-decode pipeline
//!   §7.11.4 isn't wired yet), mirroring the Round 7 `use_intrabc`
//!   treatment. Bitstreams with `allow_screen_content_tools = 0`
//!   (all non-screen-content aomenc / avifenc presets) skip the read
//!   entirely. The two `has_palette_y` / `has_palette_uv` CDFs are
//!   added to [`crate::cdfs::extra`].
//! - **Inter `read_skip` ordering (§5.11.18)** — the intra-within-inter
//!   path in [`super::inter_block::decode_inter_block_syntax`] now
//!   reads `skip` BEFORE `is_inter` and mode info, matching the spec
//!   sequence. Mirrors the round-6 intra fix; gray P-frame luma PSNR
//!   jumps from 39.34 → 43.10 dB as a direct result.
//! - **Var-tx inter `read_var_tx_size()` (§5.11.17)** — wired into
//!   the inter leaf decoder via `read_inter_block_tx_size`. The
//!   `txfm_split` CDF is added to [`crate::cdfs::extra`]; the
//!   recursion stamps `InterTxSizes[][]` onto the MI grid so
//!   neighbour `tx_depth` / `txfm_split` contexts read back the right
//!   sizes on subsequent blocks. Residual decode still operates on
//!   the block footprint — the coefficient path isn't yet per-TU.
//!
//! Round 7's earlier additions remain in place:
//!
//! - `read_delta_qindex()` (§5.11.12) / `read_delta_lf()` (§5.11.13) —
//!   called per leaf block with the spec-mandated
//!   `MiSize == sbSize && skip` short-circuit. Round 9: the decoded
//!   magnitudes now update `TileDecoder::current_q_index` and
//!   `TileDecoder::delta_lf[]`, and the dequant path indexes off
//!   `CurrentQIndex` so per-SB QP deltas land in reconstruction.
//! - `use_intrabc` (§5.11.7) — Unsupported on activation.
//! - `filter_intra_mode_info()` (§5.11.24) — Round 9 wires the
//!   §7.11.2.3 recursive filter-intra predictor: DC_PRED luma blocks
//!   ≤ 32×32 with `use_filter_intra == 1` now take the 4×2 cell
//!   recursive path with the five spec-table tap sets instead of the
//!   DC_PRED fallback.
//!
//! # Remaining syntax-level gaps (Round 9+)
//!
//! - Palette color decode (§7.11.4) — currently surfaces Unsupported;
//!   a full implementation needs `palette_tokens()` + `PaletteCache`
//!   + color-context CDFs.
//! - `DeltaLF[]` plumbing into the loop filter — the per-tile running
//!   totals are tracked but `apply_deblocking` still reads frame-level
//!   levels only; plumbing per-SB LF deltas needs a per-block LF level
//!   grid in FrameState.
//! - Var-tx residual decode — `read_var_tx_size` is wired, but the
//!   coefficient path still decodes one tile per block footprint
//!   rather than per TU stamped in `InterTxSizes[][]`.
//! - `read_skip_mode()` (§5.11.10) + `inter_segment_id()`
//!   (§5.11.19) — not yet read; currently no-ops in practice because
//!   `skip_mode_present = 0` on all our aomenc fixtures.

use oxideav_core::{Error, Result};

use crate::predict::intra::{
    cfl_pred, cfl_pred16, cfl_subsample, cfl_subsample16, dc_pred, dc_pred16,
    directional_pred16_ext, directional_pred_ext,
    edge::{
        edge_filter, edge_filter16, edge_filter_strength, edge_upsample, edge_upsample16,
        edge_use_upsample,
    },
    filter_intra_pred, filter_intra_pred16, h_pred, h_pred16, mode_to_angle_map, paeth_pred,
    paeth_pred16, smooth_h_pred, smooth_h_pred16, smooth_pred, smooth_pred16, smooth_v_pred,
    smooth_v_pred16, v_pred, v_pred16,
};
use crate::quant;
use crate::transform::{clamped_scan, default_zigzag_scan, inverse_2d, TxSize, TxType};

use super::block::{
    block_size_from_wh, block_size_log, half_below_size, horz4_size, quarter_size, vert4_size,
    BlockSize, PartitionType,
};
use super::coeffs::{decode_coefficients, nz_map_ctx_offset, tx_size_idx};
use super::frame_state::FrameState;
use super::inter_block::{
    decode_inter_block_syntax, mc_chroma_u16, mc_chroma_u8, mc_luma_u16, mc_luma_u8_scaled,
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
    // Spec §5.11.4 top-of-SB: `ReadDeltas = delta_q_present`. The
    // per-leaf `intra_frame_mode_info()` / `inter_frame_mode_info()`
    // clears it back to `false` once a non-degenerate block has
    // consumed the delta symbols.
    td.read_deltas = td.frame.delta_q_present;
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

/// Spec §5.11.56 `read_cdef()`. Called per leaf block just after the
/// `skip` flag is decoded. If the 64×64 luma region enclosing `(x, y)`
/// has not yet had `cdef_idx` signalled (`-1` sentinel) and CDEF is
/// active for the frame, consume `cdef_bits` literal bits and stamp
/// the result over the enclosing 64×64 block.
fn read_cdef(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    skip: bool,
) -> Result<()> {
    if skip
        || !td.seq.enable_cdef
        || td.frame.allow_intrabc
        || crate::frame_header_tail::coded_lossless_hint(&td.frame.quant)
    {
        return Ok(());
    }
    let cdef_bits = td.frame.cdef.cdef_bits as u32;
    let sb_col = x >> 6;
    let sb_row = y >> 6;
    if *fs.cdef_idx_mut(x, y) != -1 {
        return Ok(());
    }
    let idx = if cdef_bits == 0 {
        0
    } else {
        td.symbol.read_literal(cdef_bits) as i8
    };
    // Stamp across the 64×64 region (single entry per 64×64 in storage).
    let cols = fs.cdef_sb_cols.max(1) as usize;
    let rows = fs.cdef_sb_rows.max(1) as usize;
    let sc = (sb_col as usize).min(cols - 1);
    let sr = (sb_row as usize).min(rows - 1);
    fs.cdef_idx[sr * cols + sc] = idx;
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

    // Spec §5.11.7 `intra_frame_mode_info()` ordering — issue the
    // reads in the exact sequence the bitstream expects so the
    // symbol stream stays aligned:
    //   1. intra_segment_id  (if SegIdPreSkip)
    //   2. read_skip
    //   3. intra_segment_id  (if !SegIdPreSkip)
    //   4. read_cdef
    //   5. read_delta_qindex  (gated on ReadDeltas=delta_q_present)
    //   6. read_delta_lf      (gated on delta_lf_present)
    //   7. use_intrabc        (only when allow_intrabc)
    //   8. intra_frame_y_mode + intra_angle_info_y
    //   9. uv_mode + intra_angle_info_uv + read_cfl_alphas
    //  10. palette_mode_info()  (still skipped — Round 8+)
    //  11. filter_intra_mode_info()
    // Then §5.11.5 `decode_block()` calls `read_block_tx_size()` to
    // pull the TX_SIZE symbol.

    let seg_enabled = td.frame.segmentation.enabled && td.frame.segmentation.update_map;
    let seg_pre_skip = td.frame.segmentation.seg_id_pre_skip;
    let mut segment_id: u8 = 0;
    if seg_enabled && seg_pre_skip {
        let ctx = segment_id_ctx(above_seg, left_seg, have_above_mi, have_left_mi);
        segment_id = td.decode_segment_id(ctx)?;
    }

    let skip = td.decode_skip(0)?;

    if seg_enabled && !seg_pre_skip {
        let ctx = segment_id_ctx(above_seg, left_seg, have_above_mi, have_left_mi);
        segment_id = td.decode_segment_id(ctx)?;
    }

    // Spec §5.11.56 read_cdef(): at the first non-skip block in each
    // 64×64 luma region, read `cdef_bits` literal bits of cdef_idx and
    // stamp that value over the entire 64×64 region.
    read_cdef(td, fs, x, y, skip)?;

    // Spec §5.11.12/§5.11.13 — per-block delta Q / LF. Both reads are
    // no-ops when the corresponding frame-level flag is off (which is
    // the common case for AVIF / low-CPU `aomenc` output), but we must
    // still perform the `MiSize == sbSize && skip` guard from the spec.
    // `ReadDeltas` is cleared right after the LF delta read so later
    // blocks in the same SB don't re-consume the symbols.
    read_delta_qindex_for_block(td, bs, skip)?;
    read_delta_lf_for_block(td, bs, skip)?;
    td.read_deltas = false;

    // Spec §5.11.7 — `use_intrabc` fires only for intra frames that
    // opted into screen-content tools. Until the IntraBC motion path
    // is implemented (§7.11.3 / §7.9.1) we surface an
    // `Error::Unsupported` rather than try to reconstruct the block.
    if td.frame.allow_intrabc {
        let use_intrabc = td.decode_use_intrabc()?;
        if use_intrabc {
            return Err(Error::unsupported(
                "av1 intra-block-copy (§5.11.7 use_intrabc) pending",
            ));
        }
    }

    let y_mode = td.decode_intra_y_mode(mode_ctx_bucket(above_mode), mode_ctx_bucket(left_mode))?;
    let angle_delta_y = if y_mode.is_directional() {
        let dir_idx = (y_mode as u32) - (IntraMode::D45Pred as u32);
        td.decode_angle_delta(dir_idx)? as i8
    } else {
        0
    };

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

    // Spec §5.11.46 `palette_mode_info()` — gated on
    // `allow_screen_content_tools && MiSize >= BLOCK_8X8 &&
    // Block_Width <= 64 && Block_Height <= 64`. The helper consumes
    // the `has_palette_y` / `has_palette_uv` bits and surfaces
    // `Error::Unsupported("palette pending")` the moment either fires
    // — the palette color decode (§7.11.4) isn't wired yet. When the
    // block is palette-eligible but the encoder declined, the bits are
    // still consumed so downstream syntax stays aligned.
    read_palette_mode_info(td, fs, bs, bw, bh, y_mode, uv_mode, x, y)?;

    // Spec §5.11.24 `filter_intra_mode_info()` — gated on
    // `enable_filter_intra && YMode == DC_PRED && PaletteSizeY == 0 &&
    // max(block_w, block_h) <= 32`. Our palette path always returns
    // with `PaletteSizeY == 0` (any non-zero size above would have
    // short-circuited into Error::Unsupported), so this branch can
    // ignore the palette gate. Round 9: when `use_filter_intra == 1`
    // the recursive §7.11.2.3 predictor replaces DC_PRED for the luma
    // TX units (handled below).
    let (use_filter_intra, filter_intra_mode) =
        read_filter_intra_mode_info(td, bs, y_mode, bw, bh)?;

    // Spec §5.11.5 `decode_block()` line "read_block_tx_size()".
    // For intra frames the `is_inter` branch of §5.11.16 collapses to
    // `read_tx_size(!skip || !is_inter)` = `read_tx_size(true)`; see
    // `read_intra_block_tx_size` below for the symbol read.
    let tx_size = read_intra_block_tx_size(td, fs, bs, mi_col, mi_row, skip)?;

    // Propagate mode info to every MI cell the block covers.
    let mi_w = (bw + 3) >> 2;
    let mi_h = (bh + 3) >> 2;
    let stored_uv = if fs.monochrome { None } else { Some(uv_mode) };
    let mi_size_idx = bs as u8;
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
            mi.tx_size = Some(tx_size);
            mi.mi_size_idx = mi_size_idx;
            mi.use_filter_intra = use_filter_intra;
            mi.filter_intra_mode = filter_intra_mode;
        }
    }

    if td.frame.quant.using_qmatrix {
        return Err(Error::unsupported(
            "av1 quantization matrices (§5.9.12) pending",
        ));
    }

    // Luma path. Route filter-intra only when the block activated it
    // per §5.11.24 (`use_filter_intra` ∧ `YMode == DC_PRED`). Chroma
    // never takes the §7.11.2.3 branch, so `reconstruct_chroma_block`
    // keeps its existing mode dispatch.
    let fi_opt = if use_filter_intra && y_mode == IntraMode::DcPred {
        Some(filter_intra_mode)
    } else {
        None
    };
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
        Some(tx_size),
        fi_opt,
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

/// §5.11.16 `read_block_tx_size()` — intra branch. For intra-key /
/// intra-only frames `is_inter == 0`, so the var-tx sub-path is
/// unreachable and we always take the `read_tx_size(!skip ||
/// !is_inter)` = `read_tx_size(true)` leg.
///
/// The call is load-bearing even when the decoder still recomputes the
/// TX dims from the block footprint downstream: reading the `tx_depth`
/// symbol keeps the range coder aligned with the encoder's bitstream
/// (spec §5.11.16 is the single biggest desync driver flagged by
/// Round 5).
///
/// Returns the decoded `TxSize` — the value the spec stores as
/// `TxSize` / `InterTxSizes` — so subsequent blocks can derive the
/// `tx_depth` neighbour context.
fn read_intra_block_tx_size(
    td: &mut TileDecoder<'_>,
    fs: &FrameState,
    bs: BlockSize,
    mi_col: u32,
    mi_row: u32,
    skip: bool,
) -> Result<TxSize> {
    // §5.11.15: Lossless path pins TxSize to TX_4X4, no symbol.
    let lossless = crate::frame_header_tail::coded_lossless_hint(&td.frame.quant);
    if lossless {
        return Ok(TxSize::Tx4x4);
    }
    let max_rect_tx_size = bs.max_tx_size_rect().unwrap_or(TxSize::Tx4x4);
    let max_tx_depth = bs.max_tx_depth();
    // For intra (is_inter = 0), allowSelect = !skip || !is_inter = true.
    let allow_select = true;
    let tx_mode_select = matches!(td.frame.tx_mode, crate::frame_header_tail::TxMode::Select);
    if bs as u32 > BlockSize::Block4x4 as u32 && allow_select && tx_mode_select && max_tx_depth > 0
    {
        let ctx = tx_depth_ctx(fs, mi_col, mi_row, max_rect_tx_size);
        let raw = td.decode_tx_depth(max_tx_depth, ctx)?;
        // The `tx_depth` symbol ranges over 0..=(max_encoded_depth),
        // where encoded depth maxes out at 2 per the spec note.
        let mut tx = max_rect_tx_size;
        for _ in 0..raw {
            tx = tx.split();
        }
        Ok(tx)
    } else {
        // `skip` intentionally unused here — §5.11.16 still assigns
        // `TxSize = maxRectTxSize` when `tx_depth` isn't read.
        let _ = skip;
        Ok(max_rect_tx_size)
    }
}

/// §5.11.16 `tx_depth` CDF context. Per spec:
///   ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)
/// where `aboveW` and `leftH` come from the helpers
/// `get_above_tx_width` / `get_left_tx_height` (spec §9.4.8).
///
/// For intra key frames `IsInters[...]` is always 0, so the
/// `IsInters`-gated branches in those helpers collapse to their
/// non-inter leg (use `Tx_Width/Tx_Height[InterTxSizes[...]]`).
fn tx_depth_ctx(fs: &FrameState, mi_col: u32, mi_row: u32, max_rect_tx_size: TxSize) -> u32 {
    let max_tx_w = max_rect_tx_size.width() as u32;
    let max_tx_h = max_rect_tx_size.height() as u32;

    let above_w = if mi_row > 0 && mi_col < fs.mi_cols {
        // Row just above is present.
        let above = fs.mi_at(mi_col, mi_row - 1);
        if let Some(tx) = above.tx_size {
            tx.width() as u32
        } else {
            // Unknown → treat as 64 (max), matching `!AvailU` in spec.
            64
        }
    } else {
        // !AvailU → get_above_tx_width returns 64.
        64
    };

    let left_h = if mi_col > 0 && mi_row < fs.mi_rows {
        let left = fs.mi_at(mi_col - 1, mi_row);
        if let Some(tx) = left.tx_size {
            tx.height() as u32
        } else {
            64
        }
    } else {
        64
    };

    let a = if above_w >= max_tx_w { 1 } else { 0 };
    let l = if left_h >= max_tx_h { 1 } else { 0 };
    a + l
}

/// Spec §5.11.12 `read_delta_qindex()` — tile-level wrapper that
/// honours the `MiSize == sbSize && skip` short-circuit and the
/// `ReadDeltas` guard (itself set to `delta_q_present` at SB top,
/// cleared after the first block). Applies the decoded magnitude via
/// `CurrentQIndex = Clip3(1, 255, CurrentQIndex + (reducedDeltaQIndex
/// << delta_q_res))`, so downstream dequant picks up the per-SB Q
/// deltas.
fn read_delta_qindex_for_block(td: &mut TileDecoder<'_>, bs: BlockSize, skip: bool) -> Result<()> {
    if !td.frame.delta_q_present {
        return Ok(());
    }
    let sb_bs = if td.seq.use_128x128_superblock {
        BlockSize::Block128x128
    } else {
        BlockSize::Block64x64
    };
    if bs == sb_bs && skip {
        return Ok(());
    }
    if !td.read_deltas {
        return Ok(());
    }
    let reduced = td.decode_delta_qindex()?;
    if reduced != 0 {
        let res = td.frame.delta_q_res as i32;
        let shifted = reduced.wrapping_shl(res as u32);
        let updated = td.current_q_index.saturating_add(shifted);
        td.current_q_index = updated.clamp(1, 255);
    }
    Ok(())
}

/// Spec §5.11.13 `read_delta_lf()` — mirrors
/// [`read_delta_qindex_for_block`] but honours the `delta_lf_multi`
/// flag (which selects between 1 and `FRAME_LF_COUNT` symbols). Note
/// that `read_delta_lf()`'s outer guard is `ReadDeltas` — not
/// `delta_lf_present` — and `ReadDeltas` gets cleared right after
/// the two delta calls per §5.11.7, so this fires at most once per
/// SB regardless of `delta_lf_present`.
fn read_delta_lf_for_block(td: &mut TileDecoder<'_>, bs: BlockSize, skip: bool) -> Result<()> {
    if !td.frame.delta_q_present {
        // ReadDeltas is derived from delta_q_present, so when that
        // flag is 0 neither `read_delta_qindex` nor `read_delta_lf`
        // ever consumes bits.
        return Ok(());
    }
    let sb_bs = if td.seq.use_128x128_superblock {
        BlockSize::Block128x128
    } else {
        BlockSize::Block64x64
    };
    if bs == sb_bs && skip {
        return Ok(());
    }
    if !td.read_deltas {
        return Ok(());
    }
    if !td.frame.delta_lf_present {
        return Ok(());
    }
    let num_planes = td.seq.color_config.num_planes;
    let lf_count = if td.frame.delta_lf_multi {
        if num_planes > 1 {
            super::tile::FRAME_LF_COUNT
        } else {
            super::tile::FRAME_LF_COUNT - 2
        }
    } else {
        1
    };
    // §5.11.13: DeltaLF[i] = Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER,
    //                              DeltaLF[i] + (reducedDeltaLfLevel << delta_lf_res))
    // where MAX_LOOP_FILTER == 63 per §9.3.
    const MAX_LOOP_FILTER: i32 = 63;
    let res = td.frame.delta_lf_res as i32;
    for i in 0..lf_count {
        let reduced = td.decode_delta_lf_abs(i)?;
        if reduced != 0 {
            let shifted = reduced.wrapping_shl(res as u32);
            let updated = td.delta_lf[i].saturating_add(shifted);
            td.delta_lf[i] = updated.clamp(-MAX_LOOP_FILTER, MAX_LOOP_FILTER);
        }
    }
    Ok(())
}

/// Spec §5.11.46 `palette_mode_info()` — consume the
/// `has_palette_y` / `has_palette_uv` CDF flags on blocks that match
/// the palette-eligibility predicate. Round 8 scope: detect palette
/// activation and surface `Error::Unsupported("palette pending")`
/// rather than silently desync when the palette color decode path
/// isn't wired. Non-activating blocks consume 0 extra bits once the
/// eligibility check passes.
///
/// Eligibility (§5.11.22 / §5.11.23):
///   - `allow_screen_content_tools` frame header bit set
///   - `MiSize >= BLOCK_8X8`
///   - `Block_Width <= 64 && Block_Height <= 64`
///
/// The Y read fires only when `YMode == DC_PRED`; the UV read fires
/// only when `HasChroma && UVMode == DC_PRED`.
#[allow(clippy::too_many_arguments)]
fn read_palette_mode_info(
    td: &mut TileDecoder<'_>,
    fs: &FrameState,
    bs: BlockSize,
    bw: u32,
    bh: u32,
    y_mode: IntraMode,
    uv_mode: IntraMode,
    x: u32,
    y: u32,
) -> Result<()> {
    // Frame-level gate: per §5.11.22/.23 the palette block is only
    // called when `allow_screen_content_tools != 0`.
    if td.frame.allow_screen_content_tools == 0 {
        return Ok(());
    }
    // Block-size gate: `MiSize >= BLOCK_8X8 && Block_Width <= 64 &&
    // Block_Height <= 64`.
    if (bs as u32) < (BlockSize::Block8x8 as u32) || bw > 64 || bh > 64 {
        return Ok(());
    }
    // §5.11.46 `bsizeCtx = Mi_Width_Log2 + Mi_Height_Log2 - 2`. Our
    // Mi_Width_Log2 is `ilog2(mi_width)`: 8×8 → 1+1-2=0, 8×16 or 16×8
    // → 1+2-2=1, 16×16 → 2+2-2=2, 16×32 or 32×16 → 3, 32×32 → 4,
    // 32×64 or 64×32 → 5, 64×64 → 6. Run the computation directly off
    // `bw/bh` to cover rectangular shapes the spec permits.
    let mi_w = (bw >> 2).max(1);
    let mi_h = (bh >> 2).max(1);
    let mw_log2 = 31 - mi_w.leading_zeros();
    let mh_log2 = 31 - mi_h.leading_zeros();
    let bsize_ctx = (mw_log2 + mh_log2).saturating_sub(2) as usize;

    // Spec §9.4.6 neighbor context for `has_palette_y`:
    //   ctx = 0
    //   if (AvailU && PaletteSizes[0][MiRow-1][MiCol] > 0) ctx += 1
    //   if (AvailL && PaletteSizes[0][MiRow][MiCol-1] > 0) ctx += 1
    // We track neighbor palette activation on the MI grid once we
    // start emitting palette blocks; until then (and we bail on any
    // activation) a neighbour can never have had a palette set, so
    // ctx is structurally 0.
    let mi_col = x >> 2;
    let mi_row = y >> 2;
    let have_above = mi_row > 0 && mi_col < fs.mi_cols;
    let have_left = mi_col > 0 && mi_row < fs.mi_rows;
    let mut y_ctx = 0usize;
    if have_above {
        let above = fs.mi_at(mi_col, mi_row - 1);
        if above.palette_size_y > 0 {
            y_ctx += 1;
        }
    }
    if have_left {
        let left = fs.mi_at(mi_col - 1, mi_row);
        if left.palette_size_y > 0 {
            y_ctx += 1;
        }
    }

    if y_mode == IntraMode::DcPred {
        let has_palette_y = td.decode_has_palette_y(bsize_ctx, y_ctx)?;
        if has_palette_y {
            return Err(Error::unsupported(
                "av1 palette_mode_info luma (§5.11.46) pending",
            ));
        }
    }

    // Chroma gate: `HasChroma && UVMode == DC_PRED`.
    let has_chroma = !fs.monochrome && td.seq.color_config.num_planes >= 3;
    if has_chroma && uv_mode == IntraMode::DcPred {
        // §9.4.6 `has_palette_uv` ctx = `(PaletteSizeY > 0) ? 1 : 0`.
        // PaletteSizeY is 0 at this point (we would have bailed
        // otherwise).
        let uv_ctx = 0usize;
        let has_palette_uv = td.decode_has_palette_uv(uv_ctx)?;
        if has_palette_uv {
            return Err(Error::unsupported(
                "av1 palette_mode_info chroma (§5.11.46) pending",
            ));
        }
    }

    Ok(())
}

/// Spec §5.11.24 `filter_intra_mode_info()`. Returns
/// `(use_filter_intra, filter_intra_mode)`; both zero when the
/// block is ineligible or the encoder declined filter-intra. Per the
/// spec the read fires only when:
///   - `enable_filter_intra` (sequence header)
///   - `YMode == DC_PRED`
///   - `PaletteSizeY == 0` (palette path not decoded — always true
///     here until Round 8)
///   - `max(Block_Width, Block_Height) <= 32`
fn read_filter_intra_mode_info(
    td: &mut TileDecoder<'_>,
    bs: BlockSize,
    y_mode: IntraMode,
    bw: u32,
    bh: u32,
) -> Result<(bool, u8)> {
    if !td.seq.enable_filter_intra || y_mode != IntraMode::DcPred || bw.max(bh) > 32 {
        return Ok((false, 0));
    }
    let bs_idx = bs as usize;
    let use_fi = td.decode_use_filter_intra(bs_idx)?;
    if !use_fi {
        return Ok((false, 0));
    }
    let mode_raw = td.decode_filter_intra_mode()?;
    // The CDF emits 5 symbols (FILTER_DC_PRED..FILTER_PAETH_PRED).
    // Clamp defensively so an out-of-range value doesn't propagate
    // into the predictor tables.
    let mode = mode_raw.min(4) as u8;
    Ok((true, mode))
}

/// Spec §5.11.17 `read_var_tx_size(row, col, txSz, depth)` — consume
/// the `txfm_split` symbol(s) for a var-tx inter transform unit. The
/// spec decoder uses the resulting tree to assign `InterTxSizes[][]`
/// across the block; Round 8 keeps the recursion but does not yet
/// propagate the per-TU transform size into residual decode (inter
/// residual is still all-or-nothing at the block footprint).
///
/// Parameters mirror the spec directly:
///   - `row / col`: MI-grid coordinates of the TU corner.
///   - `tx_sz`: spec's `txSz` — the candidate transform size.
///   - `depth`: recursion depth; spec caps at `MAX_VARTX_DEPTH = 2`.
///   - `max_tx_sz`: spec's `maxTxSz = Max_Tx_Size_Rect[MiSize]`, held
///     across the recursion so the CDF context formula can index by
///     `(TX_SIZES - 1 - maxTxSz)` regardless of the current depth.
#[allow(clippy::too_many_arguments)]
fn read_var_tx_size(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    row: u32,
    col: u32,
    tx_sz: TxSize,
    depth: u32,
    max_tx_sz: TxSize,
) -> Result<()> {
    // §5.11.17 `if (row >= MiRows || col >= MiCols) return`.
    if row >= fs.mi_rows || col >= fs.mi_cols {
        return Ok(());
    }

    // §5.11.17: `if (txSz == TX_4X4 || depth == MAX_VARTX_DEPTH)
    // txfm_split = 0` — no symbol is read.
    const MAX_VARTX_DEPTH: u32 = 2;
    let forced_nosplit = matches!(tx_sz, TxSize::Tx4x4) || depth == MAX_VARTX_DEPTH;

    let txfm_split = if forced_nosplit {
        false
    } else {
        let ctx = txfm_split_ctx(fs, row, col, tx_sz, max_tx_sz);
        td.decode_txfm_split(ctx)?
    };

    let w4 = (tx_sz.width() / 4) as u32;
    let h4 = (tx_sz.height() / 4) as u32;

    if txfm_split {
        let sub = tx_sz.split();
        let step_w = (sub.width() / 4).max(1) as u32;
        let step_h = (sub.height() / 4).max(1) as u32;
        let mut i = 0u32;
        while i < h4 {
            let mut j = 0u32;
            while j < w4 {
                read_var_tx_size(td, fs, row + i, col + j, sub, depth + 1, max_tx_sz)?;
                j += step_w;
            }
            i += step_h;
        }
    } else {
        // Stamp `InterTxSizes[row+i][col+j] = txSz` for every MI cell
        // in the TU footprint. Preserving this so the §9.4.8 above /
        // left TX-size contexts read back the right size on
        // neighbouring blocks.
        for i in 0..h4 {
            for j in 0..w4 {
                let cc = col + j;
                let cr = row + i;
                if cc < fs.mi_cols && cr < fs.mi_rows {
                    fs.mi_mut(cc, cr).tx_size = Some(tx_sz);
                }
            }
        }
    }
    Ok(())
}

/// §5.11.17 / §9.4.8 `txfm_split` context. Matches the spec formula
///   above = get_above_tx_width(row, col) < Tx_Width[txSz]
///   left  = get_left_tx_height(row, col) < Tx_Height[txSz]
///   size  = Min(64, Max(Block_Width[MiSize], Block_Height[MiSize]))
///   maxTxSz = find_tx_size(size, size)
///   txSzSqrUp = Tx_Size_Sqr_Up[txSz]
///   ctx = (txSzSqrUp != maxTxSz) * 3
///       + (TX_SIZES - 1 - maxTxSz) * 6
///       + above + left
/// We approximate `maxTxSz` with the block's spec-derived
/// `Max_Tx_Size_Rect` mapped to the square-up category the CDF table
/// expects. The above/left TX-size reads fall back to 64-sample when
/// the neighbour hasn't stored an `InterTxSizes` value yet.
fn txfm_split_ctx(fs: &FrameState, row: u32, col: u32, tx_sz: TxSize, max_tx_sz: TxSize) -> usize {
    // Above width lookup (spec get_above_tx_width).
    let above_w = if row > 0 && col < fs.mi_cols {
        fs.mi_at(col, row - 1)
            .tx_size
            .map(|t| t.width() as u32)
            .unwrap_or(64)
    } else {
        64
    };
    // Left height lookup (spec get_left_tx_height).
    let left_h = if col > 0 && row < fs.mi_rows {
        fs.mi_at(col - 1, row)
            .tx_size
            .map(|t| t.height() as u32)
            .unwrap_or(64)
    } else {
        64
    };
    let above = (above_w < tx_sz.width() as u32) as usize;
    let left = (left_h < tx_sz.height() as u32) as usize;

    // `tx_size_sqr_up_cat` maps any TxSize to the square-up category
    // 0..=4 (TX_4X4..TX_64X64) the CDF formula expects.
    let sqr_up = tx_sqr_up_cat(tx_sz);
    let max_cat = tx_sqr_up_cat(max_tx_sz);
    let not_max = (sqr_up != max_cat) as usize;
    const TX_SIZES: usize = 5; // TX_4X4, TX_8X8, TX_16X16, TX_32X32, TX_64X64
    let base = not_max * 3 + (TX_SIZES - 1 - max_cat) * 6;
    base + above + left
}

/// Map a TxSize to `Tx_Size_Sqr_Up` (§Additional tables): 4×4=0,
/// 8×8=1, 16×16=2, 32×32=3, 64×64=4. Rectangular sizes collapse to
/// the larger square category.
fn tx_sqr_up_cat(t: TxSize) -> usize {
    let s = t.width().max(t.height());
    match s {
        4 => 0,
        8 => 1,
        16 => 2,
        32 => 3,
        _ => 4,
    }
}

/// Spec §5.11.16 `read_block_tx_size()` — inter-frame branch. Wraps
/// the var-tx tree (`read_var_tx_size`, §5.11.17) for inter blocks
/// that opt into `TX_MODE_SELECT`, and the fallback `read_tx_size`
/// path (`tx_depth` CDF) for everything else. Returns `Ok(())`;
/// per-TU TxSize values are stamped on the MI grid so neighbour
/// context lookups pick them up on the next block.
#[allow(clippy::too_many_arguments)]
fn read_inter_block_tx_size(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    bs: BlockSize,
    mi_col: u32,
    mi_row: u32,
    skip: bool,
    is_inter: bool,
) -> Result<()> {
    let lossless = crate::frame_header_tail::coded_lossless_hint(&td.frame.quant);
    if lossless {
        // §5.11.15 lossless path: TxSize pinned to TX_4X4.
        return Ok(());
    }
    let max_tx = match bs.max_tx_size_rect() {
        Some(t) => t,
        None => return Ok(()),
    };
    let tx_mode_select = matches!(td.frame.tx_mode, crate::frame_header_tail::TxMode::Select);
    let bw4 = (bs.width() >> 2).max(1);
    let bh4 = (bs.height() >> 2).max(1);

    // Spec §5.11.16 first leg: var-tx read fires only for inter
    // blocks bigger than 4×4 whose residual survives (not skipped)
    // and that opted into `TX_MODE_SELECT`.
    if tx_mode_select && bs != BlockSize::Block4x4 && is_inter && !skip {
        let tx_w4 = (max_tx.width() as u32 / 4).max(1);
        let tx_h4 = (max_tx.height() as u32 / 4).max(1);
        let mut r = mi_row;
        while r < mi_row + bh4 {
            let mut c = mi_col;
            while c < mi_col + bw4 {
                read_var_tx_size(td, fs, r, c, max_tx, 0, max_tx)?;
                c += tx_w4;
            }
            r += tx_h4;
        }
    } else {
        // Spec §5.11.16 else branch — `read_tx_size(!skip || !is_inter)`.
        // `allowSelect` collapses to `!skip` for inter and `true` for
        // intra-within-inter.
        let allow_select = !skip || !is_inter;
        if bs as u32 > BlockSize::Block4x4 as u32
            && allow_select
            && tx_mode_select
            && bs.max_tx_depth() > 0
        {
            let ctx = tx_depth_ctx(fs, mi_col, mi_row, max_tx);
            let raw = td.decode_tx_depth(bs.max_tx_depth(), ctx)?;
            let mut tx = max_tx;
            for _ in 0..raw {
                tx = tx.split();
            }
            for r in mi_row..mi_row + bh4 {
                for c in mi_col..mi_col + bw4 {
                    if c < fs.mi_cols && r < fs.mi_rows {
                        fs.mi_mut(c, r).tx_size = Some(tx);
                    }
                }
            }
        } else {
            // No symbol — `TxSize = maxRectTxSize`.
            for r in mi_row..mi_row + bh4 {
                for c in mi_col..mi_col + bw4 {
                    if c < fs.mi_cols && r < fs.mi_rows {
                        fs.mi_mut(c, r).tx_size = Some(max_tx);
                    }
                }
            }
        }
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
    // Spec §5.11.56 read_cdef() — also applies to inter leaves.
    read_cdef(td, fs, x, y, info.skip)?;

    // Spec §5.11.16 `read_block_tx_size()` — inter branch. For inter
    // blocks with `TxMode == TX_MODE_SELECT && MiSize > BLOCK_4X4 &&
    // !skip && !Lossless` we descend the var-tx tree; otherwise we
    // consult `read_tx_size`. Either path consumes the relevant
    // symbol bits so the range coder stays aligned with the encoder.
    let mi_col_blk = x >> 2;
    let mi_row_blk = y >> 2;
    let bs_inter = block_size_from_wh(bw, bh);
    if bs_inter != BlockSize::Invalid {
        read_inter_block_tx_size(
            td,
            fs,
            bs_inter,
            mi_col_blk,
            mi_row_blk,
            info.skip,
            info.is_inter,
        )?;
    }
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
        // Inter-within-inter intra blocks don't read `tx_depth` yet
        // (the var-tx inter path is out of scope for Round 6), so pass
        // `None` and let reconstruct_luma_block use its pre-Round-6
        // 64-sample cap fallback.
        reconstruct_luma_block(
            td,
            fs,
            x,
            y,
            bw,
            bh,
            info.intra_y_mode,
            0,
            info.skip,
            0,
            None,
            None,
        )?;
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
        // §7.9 reference scaling: if the previous frame's resolution
        // differs from the current frame's, route through the scaled
        // MC helper which projects the block origin + MV into the
        // reference coordinate system. Matching-dim case collapses to
        // the plain `mc_luma_u8` path inside the helper.
        let pred = mc_luma_u8_scaled(
            &prev.y_plane,
            prev.width as usize,
            prev.height as usize,
            prev.width as usize,
            fs.width,
            fs.height,
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
    // §5.11.12 — dequant indexes by `CurrentQIndex`, not the frame-level
    // `base_q_idx`, so per-SB delta_q updates propagate here.
    let qv = quant::Params {
        base_q_idx: td.current_q_index,
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
        // §5.11.12: indexing tables by `CurrentQIndex` so per-SB
        // delta_q updates affect the inter-chroma dequant too.
        let qv = quant::Params {
            base_q_idx: td.current_q_index,
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
    decoded_tx_size: Option<TxSize>,
    filter_intra: Option<u8>,
) -> Result<()> {
    let w = bw as usize;
    let h = bh as usize;

    // §5.11.16 provided a `TxSize` for this block. When present, use it
    // to partition the block into TX units — this replaces the legacy
    // `tx_unit_dims(w, h)` 64-sample cap with the spec's `(tx_depth,
    // maxRectTxSize)` pairing. When absent (inter paths not yet wired
    // to the var-tx decoder) we fall back to the pre-Round-6 behaviour.
    let (tx_w, tx_h) = match decoded_tx_size {
        Some(ts) => (ts.width().min(w), ts.height().min(h)),
        None => tx_unit_dims(w, h),
    };
    // Guard against degenerate dims — the block footprint must divide
    // the TX dimensions cleanly; when it doesn't we fall back to the
    // block size (effectively one TX unit) to avoid a divide-by-zero /
    // partial-coverage walk.
    let (tx_w, tx_h) = if tx_w == 0 || tx_h == 0 || w % tx_w != 0 || h % tx_h != 0 {
        tx_unit_dims(w, h)
    } else {
        (tx_w, tx_h)
    };

    // Block-level tx_type (§6.10.15): for blocks with area > 32×32 the
    // ext-tx set is 0 (implicit DCT_DCT, no CDF read), so decoding the
    // type once here covers every TX unit below without disturbing the
    // symbol stream. For blocks with area ≤ 32×32 there is exactly one
    // TX unit (since the block already fits inside a 64×64 TX), so
    // reading the type once remains spec-correct.
    //
    // Per §5.11.47 the `intra_tx_type` symbol is read **per TX unit**,
    // not per block — we approximate by reading once per block when the
    // block has only a single TX unit, and once per TX unit otherwise.
    let single_tx_unit = (w == tx_w) && (h == tx_h);
    let block_tx_type = if skip || !single_tx_unit {
        TxType::DctDct
    } else {
        td.decode_intra_tx_type(tx_w, tx_h, y_mode)?
    };

    let cols = w / tx_w;
    let rows = h / tx_h;
    for ty in 0..rows {
        for tx in 0..cols {
            let ux = x as usize + tx * tx_w;
            let uy = y as usize + ty * tx_h;
            let tx_type = if skip {
                TxType::DctDct
            } else if single_tx_unit {
                block_tx_type
            } else {
                td.decode_intra_tx_type(tx_w, tx_h, y_mode)?
            };
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
                filter_intra,
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
    filter_intra: Option<u8>,
) -> Result<()> {
    let stride = fs.width as usize;
    // §7.11.2.8: filterType depends on the above/left MI's prediction
    // mode. MiRow/MiCol here are in luma-MI units (4×4 samples).
    let mi_col_u = (ux >> 2) as u32;
    let mi_row_u = (uy >> 2) as u32;
    let enable_edge_filter = td.seq.enable_intra_edge_filter;
    let filter_type = if enable_edge_filter {
        get_filter_type(fs, mi_row_u, mi_col_u, 0)
    } else {
        0
    };

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
            enable_edge_filter,
            filter_type,
            filter_intra,
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
            enable_edge_filter,
            filter_type,
            filter_intra,
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
    // §7.11.2.8 applies per-plane. Convert chroma plane coords back to
    // luma-MI units via sub_x / sub_y.
    let mi_col_u = ((ux as u32) << fs.sub_x) >> 2;
    let mi_row_u = ((uy as u32) << fs.sub_y) >> 2;
    let enable_edge_filter = td.seq.enable_intra_edge_filter;
    let filter_type = if enable_edge_filter {
        get_filter_type(fs, mi_row_u, mi_col_u, plane_idx + 1)
    } else {
        0
    };

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
            enable_edge_filter,
            filter_type,
            None,
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
            enable_edge_filter,
            filter_type,
            None,
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

/// Spec §7.11.2.8 `get_filter_type`: 1 if either the block immediately
/// above or to the left uses one of the three SMOOTH_* intra modes.
/// Returns 0 otherwise (including when the neighbour is unavailable or
/// was inter-coded).
///
/// `plane_idx` is 0 for Y, 1/2 for chroma. Chroma planes shift the MI
/// lookup by subsampling per the spec so 4:2:0 blocks read the correct
/// 4×4 MI cell.
fn get_filter_type(fs: &FrameState, mi_row: u32, mi_col: u32, plane_idx: u32) -> u32 {
    let is_smooth = |mode_opt: Option<IntraMode>| -> bool {
        matches!(
            mode_opt,
            Some(IntraMode::SmoothPred | IntraMode::SmoothVPred | IntraMode::SmoothHPred)
        )
    };
    let sub_x = fs.sub_x;
    let sub_y = fs.sub_y;
    let mut above_smooth = false;
    let mut left_smooth = false;
    if mi_row > 0 {
        let mut r = mi_row - 1;
        let mut c = mi_col;
        if plane_idx > 0 {
            if sub_x == 1 && (mi_col & 1) == 0 && c + 1 < fs.mi_cols {
                c += 1;
            }
            if sub_y == 1 && (mi_row & 1) == 1 && r > 0 {
                r -= 1;
            }
        }
        if r < fs.mi_rows && c < fs.mi_cols {
            let mi = fs.mi_at(c, r);
            // §7.11.2.8 is_smooth: non-luma inter blocks return 0.
            let mode = if plane_idx == 0 {
                mi.mode
            } else if mi.is_inter {
                None
            } else {
                mi.uv_mode
            };
            above_smooth = is_smooth(mode);
        }
    }
    if mi_col > 0 {
        let mut r = mi_row;
        let mut c = mi_col - 1;
        if plane_idx > 0 {
            if sub_x == 1 && (mi_col & 1) == 1 && c > 0 {
                c -= 1;
            }
            if sub_y == 1 && (mi_row & 1) == 0 && r + 1 < fs.mi_rows {
                r += 1;
            }
        }
        if r < fs.mi_rows && c < fs.mi_cols {
            let mi = fs.mi_at(c, r);
            let mode = if plane_idx == 0 {
                mi.mode
            } else if mi.is_inter {
                None
            } else {
                mi.uv_mode
            };
            left_smooth = is_smooth(mode);
        }
    }
    if above_smooth || left_smooth {
        1
    } else {
        0
    }
}

/// Run the u8 intra predictor against the reconstructed plane. Returns
/// a tight `w*h` buffer of predicted samples.
///
/// `enable_edge_filter` / `filter_type` come from §7.11.2.8; when
/// enabled and the mode is directional (other than 90°/180°), the
/// above/left edges are pre-processed via §7.11.2.9–.12.
///
/// When `filter_intra` is `Some(mode_idx)`, the recursive §7.11.2.3
/// filter-intra predictor replaces the normal DC_PRED path. The spec
/// only activates this branch on DC_PRED luma blocks ≤ 32×32; the
/// caller is responsible for enforcing those preconditions.
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
    enable_edge_filter: bool,
    filter_type: u32,
    filter_intra: Option<u8>,
) -> Vec<u8> {
    // Extended neighbours for directional / filter-intra reads. Length
    // covers the longest projection: above + right of block.
    let ext_len = w + h + 4;
    let _ = fallback;
    let (above_raw, left_raw, above_left) =
        gather_neighbors_u8(plane, plane_stride, plane_height, x, y, w, h, ext_len, 8);
    let have_above = y > 0;
    let have_left = x > 0;

    let mut dst = vec![0u8; w * h];
    // §7.11.2.3 recursive filter-intra — overrides DC_PRED when the
    // block elected it (gated by the caller to DC_PRED ≤ 32×32). The
    // predictor reads `w+1` above samples (including the corner) and
    // `h` left samples.
    if let Some(fi_mode) = filter_intra {
        filter_intra_pred(
            &mut dst,
            w,
            h,
            &above_raw,
            &left_raw,
            above_left,
            fi_mode as usize,
        );
        return dst;
    }
    match mode {
        IntraMode::DcPred | IntraMode::CflPred => {
            dc_pred(
                &mut dst, w, h, &above_raw, &left_raw, have_above, have_left, 8,
            );
        }
        IntraMode::VPred => {
            v_pred(&mut dst, w, h, &above_raw);
        }
        IntraMode::HPred => {
            h_pred(&mut dst, w, h, &left_raw);
        }
        IntraMode::D45Pred
        | IntraMode::D67Pred
        | IntraMode::D113Pred
        | IntraMode::D135Pred
        | IntraMode::D157Pred
        | IntraMode::D203Pred => {
            let base = mode_to_angle_map(mode);
            let p_angle = base + (angle_delta as i32) * 3;
            let above_delta = p_angle - 90;
            let left_delta = p_angle - 180;
            // Clone into mutable buffers so edge_filter can operate in place.
            let mut above = above_raw.clone();
            let mut left = left_raw.clone();
            let mut upsample_above = false;
            let mut upsample_left = false;
            if enable_edge_filter && p_angle != 90 && p_angle != 180 {
                if have_above {
                    let strength =
                        edge_filter_strength(w as u32, h as u32, filter_type, above_delta);
                    edge_filter(&mut above, strength);
                }
                if have_left {
                    let strength =
                        edge_filter_strength(w as u32, h as u32, filter_type, left_delta);
                    edge_filter(&mut left, strength);
                }
                if edge_use_upsample(w as u32, h as u32, filter_type, above_delta) {
                    above = upsample_edge_u8(&above, w, h, p_angle < 90);
                    upsample_above = true;
                }
                if edge_use_upsample(w as u32, h as u32, filter_type, left_delta) {
                    left = upsample_edge_u8(&left, h, w, p_angle > 180);
                    upsample_left = true;
                }
            }
            directional_pred_ext(
                &mut dst,
                w,
                h,
                &above,
                &left,
                p_angle,
                upsample_above,
                upsample_left,
            );
        }
        IntraMode::SmoothPred => {
            smooth_pred(&mut dst, w, h, &above_raw, &left_raw);
        }
        IntraMode::SmoothVPred => {
            smooth_v_pred(&mut dst, w, h, &above_raw, &left_raw);
        }
        IntraMode::SmoothHPred => {
            smooth_h_pred(&mut dst, w, h, &above_raw, &left_raw);
        }
        IntraMode::PaethPred => {
            paeth_pred(&mut dst, w, h, &above_raw, &left_raw, above_left);
        }
    }
    dst
}

/// Run [`edge_upsample`] against a caller-gathered extended edge slice
/// and return the upsampled buffer laid out so that `out[0]` maps to
/// spec `AboveRow[0]` (or `LeftCol[0]`) in the 2×-density frame —
/// i.e. the sample immediately after the corner.
///
/// `primary` is the block dimension along the edge (w for above, h for
/// left); `secondary` is the other dimension. `extends_into` is `true`
/// when `pAngle < 90` (for above) or `pAngle > 180` (for left), meaning
/// the projection reads past the block's immediate edge.
fn upsample_edge_u8(edge: &[u8], primary: usize, secondary: usize, extends_into: bool) -> Vec<u8> {
    // §7.11.2.6: numPx for above is `w + (pAngle < 90 ? h : 0)`.
    let num_px = primary + if extends_into { secondary } else { 0 };
    // edge_upsample's convention: buf[0] = spec index -2, buf[1] =
    // spec index -1 (corner), buf[2..=num_px+1] = spec 0..num_px-1.
    // Our gathered `edge` has `edge[0]` = spec AboveRow[0]; the corner
    // sample is not directly available here (callers in the intra path
    // store it separately), but the upsampler reads spec -1 only for
    // its `dup[0]` fallback — replicating `edge[0]` there is a safe
    // edge-extension behaviour consistent with §7.11.2.11.
    let mut buf = vec![0u8; 2 * num_px + 2];
    buf[1] = edge[0];
    for i in 0..num_px {
        let src = edge
            .get(i)
            .copied()
            .unwrap_or_else(|| *edge.last().expect("upsample_edge_u8: empty edge slice"));
        buf[i + 2] = src;
    }
    edge_upsample(&mut buf, num_px);
    // After upsample, buf[0]=spec -2, buf[1]=spec -1 (corner),
    // buf[2]=spec 0. Strip the two leading slots so out[0] = spec 0
    // in the 2×-density frame.
    buf.drain(..2);
    buf
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
    enable_edge_filter: bool,
    filter_type: u32,
    filter_intra: Option<u8>,
) -> Vec<u16> {
    let ext_len = w + h + 4;
    let _ = fallback;
    let (above_raw, left_raw, above_left) = gather_neighbors_u16(
        plane,
        plane_stride,
        plane_height,
        x,
        y,
        w,
        h,
        ext_len,
        bit_depth,
    );
    let have_above = y > 0;
    let have_left = x > 0;

    let mut dst = vec![0u16; w * h];
    // §7.11.2.3 recursive filter-intra (HBD path).
    if let Some(fi_mode) = filter_intra {
        filter_intra_pred16(
            &mut dst,
            w,
            h,
            &above_raw,
            &left_raw,
            above_left,
            fi_mode as usize,
            bit_depth,
        );
        return dst;
    }
    match mode {
        IntraMode::DcPred | IntraMode::CflPred => {
            dc_pred16(
                &mut dst, w, h, &above_raw, &left_raw, have_above, have_left, bit_depth,
            );
        }
        IntraMode::VPred => {
            v_pred16(&mut dst, w, h, &above_raw);
        }
        IntraMode::HPred => {
            h_pred16(&mut dst, w, h, &left_raw);
        }
        IntraMode::D45Pred
        | IntraMode::D67Pred
        | IntraMode::D113Pred
        | IntraMode::D135Pred
        | IntraMode::D157Pred
        | IntraMode::D203Pred => {
            let base = mode_to_angle_map(mode);
            let p_angle = base + (angle_delta as i32) * 3;
            let above_delta = p_angle - 90;
            let left_delta = p_angle - 180;
            let mut above = above_raw.clone();
            let mut left = left_raw.clone();
            let mut upsample_above = false;
            let mut upsample_left = false;
            if enable_edge_filter && p_angle != 90 && p_angle != 180 {
                if have_above {
                    let strength =
                        edge_filter_strength(w as u32, h as u32, filter_type, above_delta);
                    edge_filter16(&mut above, strength, bit_depth);
                }
                if have_left {
                    let strength =
                        edge_filter_strength(w as u32, h as u32, filter_type, left_delta);
                    edge_filter16(&mut left, strength, bit_depth);
                }
                if edge_use_upsample(w as u32, h as u32, filter_type, above_delta) {
                    above = upsample_edge_u16(&above, w, h, p_angle < 90, bit_depth);
                    upsample_above = true;
                }
                if edge_use_upsample(w as u32, h as u32, filter_type, left_delta) {
                    left = upsample_edge_u16(&left, h, w, p_angle > 180, bit_depth);
                    upsample_left = true;
                }
            }
            directional_pred16_ext(
                &mut dst,
                w,
                h,
                &above,
                &left,
                p_angle,
                bit_depth,
                upsample_above,
                upsample_left,
            );
        }
        IntraMode::SmoothPred => {
            smooth_pred16(&mut dst, w, h, &above_raw, &left_raw);
        }
        IntraMode::SmoothVPred => {
            smooth_v_pred16(&mut dst, w, h, &above_raw, &left_raw);
        }
        IntraMode::SmoothHPred => {
            smooth_h_pred16(&mut dst, w, h, &above_raw, &left_raw);
        }
        IntraMode::PaethPred => {
            paeth_pred16(&mut dst, w, h, &above_raw, &left_raw, above_left, bit_depth);
        }
    }
    dst
}

/// HBD twin of [`upsample_edge_u8`].
fn upsample_edge_u16(
    edge: &[u16],
    primary: usize,
    secondary: usize,
    extends_into: bool,
    bit_depth: u32,
) -> Vec<u16> {
    let num_px = primary + if extends_into { secondary } else { 0 };
    let mut buf = vec![0u16; 2 * num_px + 2];
    buf[1] = edge[0];
    for i in 0..num_px {
        let src = edge
            .get(i)
            .copied()
            .unwrap_or_else(|| *edge.last().expect("upsample_edge_u16: empty edge slice"));
        buf[i + 2] = src;
    }
    edge_upsample16(&mut buf, num_px, bit_depth);
    buf.drain(..2);
    buf
}

/// Gather extended neighbour samples from a u8 plane per AV1
/// §7.11.2.1. Returns `(above, left, above_left)` where `above` and
/// `left` each have `ext_len` samples (typically `w + h`).
///
/// `w` and `h` are the transform-block dimensions driving the
/// `aboveLimit`/`leftLimit` clamping from the spec: without knowledge
/// of `haveAboveRight`/`haveBelowLeft` we use the conservative form
/// (`aboveLimit = x + w - 1`, `leftLimit = y + h - 1`) so indices past
/// the block's right/below edges replicate the last valid sample —
/// never read into un-reconstructed territory.
///
/// Unavailability substitution matches the spec:
/// - `haveAbove=0, haveLeft=1` → `AboveRow[i] = CurrFrame[y][x-1]`
/// - `haveAbove=1, haveLeft=0` → `LeftCol[i]  = CurrFrame[y-1][x]`
/// - neither → `AboveRow = (1<<(bd-1)) - 1`, `LeftCol = (1<<(bd-1)) + 1`
/// - Corner `AboveRow[-1]` follows the 4-case derivation in §7.11.2.1.
#[allow(clippy::too_many_arguments)]
fn gather_neighbors_u8(
    plane: &[u8],
    plane_stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    ext_len: usize,
    bit_depth: u32,
) -> (Vec<u8>, Vec<u8>, u8) {
    let have_above = y > 0;
    let have_left = x > 0;
    let max_x = plane_stride.saturating_sub(1);
    let max_y = plane_height.saturating_sub(1);
    let mid_above: u8 = ((1u32 << (bit_depth - 1)).saturating_sub(1)).min(255) as u8;
    let mid_left: u8 = ((1u32 << (bit_depth - 1)) + 1).min(255) as u8;

    let mut above = vec![mid_above; ext_len];
    let mut left = vec![mid_left; ext_len];

    if !have_above && have_left {
        let v = plane[y * plane_stride + (x - 1)];
        for slot in above.iter_mut() {
            *slot = v;
        }
    } else if have_above {
        let above_limit = max_x.min(x + w.saturating_sub(1));
        let row = (y - 1) * plane_stride;
        for (c, slot) in above.iter_mut().enumerate() {
            let sx = (x + c).min(above_limit);
            *slot = plane[row + sx];
        }
    }

    if !have_left && have_above {
        let v = plane[(y - 1) * plane_stride + x];
        for slot in left.iter_mut() {
            *slot = v;
        }
    } else if have_left {
        let left_limit = max_y.min(y + h.saturating_sub(1));
        for (r, slot) in left.iter_mut().enumerate() {
            let sy = (y + r).min(left_limit);
            *slot = plane[sy * plane_stride + (x - 1)];
        }
    }

    let above_left = if have_above && have_left {
        plane[(y - 1) * plane_stride + (x - 1)]
    } else if have_above {
        plane[(y - 1) * plane_stride + x]
    } else if have_left {
        plane[y * plane_stride + (x - 1)]
    } else {
        (1u32 << (bit_depth - 1)).min(255) as u8
    };
    (above, left, above_left)
}

/// u16 counterpart of [`gather_neighbors_u8`].
#[allow(clippy::too_many_arguments)]
fn gather_neighbors_u16(
    plane: &[u16],
    plane_stride: usize,
    plane_height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    ext_len: usize,
    bit_depth: u32,
) -> (Vec<u16>, Vec<u16>, u16) {
    let have_above = y > 0;
    let have_left = x > 0;
    let max_x = plane_stride.saturating_sub(1);
    let max_y = plane_height.saturating_sub(1);
    let mid_above: u16 = (1u16 << (bit_depth - 1)).saturating_sub(1);
    let max_sample: u16 = ((1u32 << bit_depth) - 1) as u16;
    let mid_left: u16 = ((1u16 << (bit_depth - 1)) + 1).min(max_sample);

    let mut above = vec![mid_above; ext_len];
    let mut left = vec![mid_left; ext_len];

    if !have_above && have_left {
        let v = plane[y * plane_stride + (x - 1)];
        for slot in above.iter_mut() {
            *slot = v;
        }
    } else if have_above {
        let above_limit = max_x.min(x + w.saturating_sub(1));
        let row = (y - 1) * plane_stride;
        for (c, slot) in above.iter_mut().enumerate() {
            let sx = (x + c).min(above_limit);
            *slot = plane[row + sx];
        }
    }
    if !have_left && have_above {
        let v = plane[(y - 1) * plane_stride + x];
        for slot in left.iter_mut() {
            *slot = v;
        }
    } else if have_left {
        let left_limit = max_y.min(y + h.saturating_sub(1));
        for (r, slot) in left.iter_mut().enumerate() {
            let sy = (y + r).min(left_limit);
            *slot = plane[sy * plane_stride + (x - 1)];
        }
    }
    let above_left = if have_above && have_left {
        plane[(y - 1) * plane_stride + (x - 1)]
    } else if have_above {
        plane[(y - 1) * plane_stride + x]
    } else if have_left {
        plane[y * plane_stride + (x - 1)]
    } else {
        1u16 << (bit_depth - 1)
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

/// Compute the segment-adjusted base quantizer index used by per-block
/// dequant. Per spec §5.11.12 the quantizer used is `CurrentQIndex`
/// (which rides on top of `base_q_idx` via per-SB delta_q), and per
/// §5.11.14 the segmentation feature `SEG_LVL_ALT_Q` re-anchors the
/// value when enabled for the block's segment.
fn segmented_base_q(td: &TileDecoder<'_>, segment_id: u8) -> i32 {
    // Per §5.11.12 `CurrentQIndex` is what the dequantizer tables are
    // indexed by. When `delta_q_present == 0` it stays pinned to
    // `base_q_idx` per §5.11.1.
    let base = td.current_q_index;
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
    // SEG_LVL_ALT_Q semantics: when segmentation_update_data's
    // `segmentation_abs_or_delta_update` is 1 the feature_data value
    // replaces `base_q_idx`; when 0 it adds. Spec requires applying
    // segmentation to `base_q_idx` first, then adding the delta_q
    // running total. Our frame header flattens that already, so fall
    // back to adding the feature_data delta to `CurrentQIndex`.
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

    // §7.11.2.8: filterType should be 1 when either MI neighbour is a
    // smooth-family intra mode, 0 otherwise (and 0 when no neighbours
    // exist at all).
    #[test]
    fn get_filter_type_detects_smooth_above() {
        let mut fs = FrameState::new(32, 32, 1, 1, false);
        // Mark the MI cell directly above (0, 0) as SMOOTH_PRED.
        fs.mi_mut(0, 0).mode = Some(IntraMode::SmoothPred);
        // Query for MI at (col=0, row=1) — above is row 0.
        assert_eq!(get_filter_type(&fs, 1, 0, 0), 1);
    }

    #[test]
    fn get_filter_type_detects_smooth_left() {
        let mut fs = FrameState::new(32, 32, 1, 1, false);
        fs.mi_mut(0, 0).mode = Some(IntraMode::SmoothHPred);
        // Query for MI at (col=1, row=0) — left is col 0.
        assert_eq!(get_filter_type(&fs, 0, 1, 0), 1);
    }

    #[test]
    fn get_filter_type_zero_for_non_smooth() {
        let mut fs = FrameState::new(32, 32, 1, 1, false);
        fs.mi_mut(0, 0).mode = Some(IntraMode::DcPred);
        fs.mi_mut(1, 0).mode = Some(IntraMode::D45Pred);
        assert_eq!(get_filter_type(&fs, 1, 1, 0), 0);
    }

    #[test]
    fn get_filter_type_frame_edge_returns_zero() {
        let fs = FrameState::new(32, 32, 1, 1, false);
        // At the top-left corner neither neighbour exists.
        assert_eq!(get_filter_type(&fs, 0, 0, 0), 0);
    }

    // Sanity that `run_intra_prediction_u8` with edge filter enabled on
    // a flat neighbour pattern yields a constant output — the low-pass
    // kernels and upsample polyphase are both normalised so uniform
    // input must pass through unchanged.
    #[test]
    fn run_intra_prediction_u8_constant_edges_passthrough_with_edge_filter() {
        // 8×8 plane filled with constant 100 surrounds a 4×4 block at
        // (0, 4). With y > 0 and x == 0, only the above edge feeds the
        // predictor; D67 projects into it.
        let plane = vec![100u8; 8 * 8];
        let got = run_intra_prediction_u8(
            &plane,
            8, /* stride */
            8, /* height */
            0, /* x */
            4, /* y */
            4, /* w */
            4, /* h */
            IntraMode::D67Pred,
            0,    /* angle_delta */
            128,  /* fallback */
            true, /* enable_edge_filter */
            0,    /* filter_type */
            None, /* filter_intra */
        );
        for &v in &got {
            assert_eq!(v, 100, "edge-filtered constant D67 drifted: {:?}", got);
        }
    }

    // §7.11.2.3 recursive filter-intra: when `filter_intra: Some(_)`
    // the predictor bypasses the DC_PRED branch. A flat 100-sample
    // neighbourhood lets every tap set in Table 7.11.2.3 emit 100
    // (the taps sum to 16, so `(100*16 + 8) >> 4 = 100`). Verifies
    // the dispatch plumbing reaches the new code path.
    #[test]
    fn run_intra_prediction_u8_filter_intra_constant_passthrough() {
        let plane = vec![100u8; 8 * 8];
        for mode in 0..5u8 {
            let got = run_intra_prediction_u8(
                &plane,
                8,
                8,
                0,
                4,
                4,
                4,
                IntraMode::DcPred,
                0,
                128,
                false, /* enable_edge_filter */
                0,
                Some(mode),
            );
            for &v in &got {
                assert_eq!(
                    v, 100,
                    "filter-intra mode {mode} drifted on constant neighbours: {got:?}",
                );
            }
        }
    }

    // §5.11.16 / §9.4.8: `tx_depth` context is
    //   (aboveW >= maxTxWidth) + (leftH >= maxTxHeight).
    // Without neighbours both `aboveW` and `leftH` clamp to 64, so for
    // any `maxRectTxSize` ≤ 64×64 the ctx is 2. A smaller neighbour
    // TX drops the corresponding term to 0.
    #[test]
    fn tx_depth_ctx_edges_without_neighbours_collapse_to_two() {
        let fs = FrameState::new(32, 32, 1, 1, false);
        // Top-left corner — both AvailU and AvailL are false.
        assert_eq!(tx_depth_ctx(&fs, 0, 0, TxSize::Tx32x32), 2);
    }

    #[test]
    fn tx_depth_ctx_small_above_neighbour_drops_above_term() {
        let mut fs = FrameState::new(32, 32, 1, 1, false);
        // Stamp a 4×4 TX on (0, 0); querying (0, 1) uses that as above.
        fs.mi_mut(0, 0).tx_size = Some(TxSize::Tx4x4);
        // For maxRectTxSize = Tx16x16 the above term is 0 (4 < 16),
        // but left clamps to 64 → ctx = 0 + 1 = 1.
        assert_eq!(tx_depth_ctx(&fs, 0, 1, TxSize::Tx16x16), 1);
    }

    #[test]
    fn tx_depth_ctx_large_neighbours_give_two() {
        let mut fs = FrameState::new(64, 64, 1, 1, false);
        fs.mi_mut(0, 0).tx_size = Some(TxSize::Tx32x32);
        fs.mi_mut(1, 0).tx_size = Some(TxSize::Tx32x32);
        fs.mi_mut(0, 1).tx_size = Some(TxSize::Tx32x32);
        // Query (1, 1) — above is (1, 0), left is (0, 1). Both 32 ≥ 16,
        // so ctx = 1 + 1 = 2.
        assert_eq!(tx_depth_ctx(&fs, 1, 1, TxSize::Tx16x16), 2);
    }
}
