//! AV1 superblock + leaf-block mode decoder — §5.11.4 + §5.11.8 +
//! §5.11.18.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/superblock.go`
//! (MIT, KarpelesLab/goavif), minus the pixel-reconstruction and
//! inter-frame branches. Only the partition walk + the intra mode
//! symbols (Y mode, UV mode, angle_delta, segment_id, skip, CFL
//! sign/alpha) are read. At the first non-skip leaf the decoder
//! returns `Error::Unsupported("av1 coefficient decode pending
//! (§5.11.39)")`.
//!
//! Spec references:
//!
//! - §5.11.4 `decode_partition` — partition-tree recursion.
//! - §5.11.8 `decode_block` — per-leaf mode read.
//! - §5.11.18 `intra_mode_info` — Y/UV modes + `angle_delta`.
//! - §5.11.21 `palette_mode_info` — **not read** (we depend on the
//!   encoder not signalling palette; real support is Phase 3+).
//! - §5.11.27 `read_tx_size` — not read (Phase 3).
//! - §5.11.39 `residual` — not read (Phase 3).

use oxideav_core::{Error, Result};

use super::block::{
    block_size_log, half_below_size, horz4_size, quarter_size, vert4_size, BlockSize,
    PartitionType,
};
use super::frame_state::FrameState;
use super::modes::{mode_ctx_bucket, IntraMode};
use super::tile::{cfl_alpha_ctx, cfl_signs, segment_id_ctx, TileDecoder};

/// Decode one superblock at luma-sample position `(sb_x, sb_y)`.
///
/// Per spec §5.11.3, AV1 superblocks are 64×64 or 128×128 luma
/// samples. The partition-tree recursion in [`decode_partition_node`]
/// walks the tree rooted at this SB and decodes every leaf's mode
/// info. Phase 2 does NOT consume `cdef_idx` — that signal lives in
/// §5.11.10 and requires coefficient state we don't have yet.
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
/// Leaf blocks dispatch to [`decode_leaf_block`]; inner nodes read a
/// partition symbol and recurse.
///
/// Spec §5.11.4.
pub fn decode_partition_node(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bs: BlockSize,
) -> Result<()> {
    // Skip blocks that fall entirely outside the frame.
    if x >= fs.width || y >= fs.height {
        return Ok(());
    }

    // At minimum block size there's no further split.
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

    // BSL context + left/above partition context — goavif simplifies
    // the per-neighbor computation to ctx=0 unconditionally. We
    // mirror that to stay bit-compatible with goavif-written streams
    // (AV1 still decodes with the default ctx=0 because the CDF
    // adapts after each symbol).
    let bsl = block_size_log(bs);
    let above_ctx = 0u32;
    let left_ctx = 0u32;
    let pt_raw = td.decode_partition(bsl, above_ctx * 2 + left_ctx)?;
    let pt = PartitionType::from_u32(pt_raw).ok_or_else(|| {
        Error::invalid(format!(
            "av1 partition: invalid symbol {pt_raw} (§5.11.4)"
        ))
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

/// Decode a single leaf coding block — §5.11.8 `decode_block` plus
/// §5.11.18 `intra_mode_info`. Reads segment_id (if enabled), Y mode,
/// skip, UV mode (+ CFL joint sign / per-plane alpha), and
/// angle_delta for directional modes. Writes the per-MI-unit
/// [`super::frame_state::ModeInfo`] for every 4×4 cell the block
/// covers.
///
/// On a non-skip block the function returns
/// `Error::Unsupported("av1 coefficient decode pending (§5.11.39)")`
/// after the mode info has been written, because Phase 2 doesn't
/// implement coefficient decode.
pub fn decode_leaf_block(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    bs: BlockSize,
) -> Result<()> {
    let w = bs.width();
    let h = bs.height();

    // Clip to frame boundaries.
    let bw = w.min(fs.width.saturating_sub(x));
    let bh = h.min(fs.height.saturating_sub(y));
    if bw == 0 || bh == 0 {
        return Ok(());
    }

    let mi_col = x >> 2;
    let mi_row = y >> 2;

    // Neighbor mode + segment_id state for the context reads. MI
    // grid is (0,0)-origin: "above" is mi_row-1, "left" is
    // mi_col-1.
    let have_above = mi_row > 0 && mi_row - 1 < fs.mi_rows && mi_col < fs.mi_cols;
    let have_left = mi_col > 0 && mi_col - 1 < fs.mi_cols && mi_row < fs.mi_rows;

    let above_mode = if have_above {
        fs.mi_at(mi_col, mi_row - 1).mode.unwrap_or(IntraMode::DcPred)
    } else {
        IntraMode::DcPred
    };
    let left_mode = if have_left {
        fs.mi_at(mi_col - 1, mi_row).mode.unwrap_or(IntraMode::DcPred)
    } else {
        IntraMode::DcPred
    };
    let above_seg = if have_above {
        fs.mi_at(mi_col, mi_row - 1).segment_id
    } else {
        0
    };
    let left_seg = if have_left {
        fs.mi_at(mi_col - 1, mi_row).segment_id
    } else {
        0
    };

    // §5.11.9 segment_id — only signalled when segmentation is on and
    // update_map is set. Phase 2 matches goavif's conservative path.
    let segment_id =
        if td.frame.segmentation.enabled && td.frame.segmentation.update_map {
            let ctx = segment_id_ctx(above_seg, left_seg, have_above, have_left);
            td.decode_segment_id(ctx)?
        } else {
            0
        };

    // §5.11.18 Y intra mode.
    let y_mode = td.decode_intra_y_mode(
        mode_ctx_bucket(above_mode),
        mode_ctx_bucket(left_mode),
    )?;

    // Angle-delta is read right after the mode for directional
    // modes. The goavif port interleaves differently but the §5.11.18
    // spec order is `y_mode` → `angle_delta_y`. We follow the spec.
    let angle_delta_y = if y_mode.is_directional() {
        let dir_idx = (y_mode as u32) - (IntraMode::D45Pred as u32);
        td.decode_angle_delta(dir_idx)? as i8
    } else {
        0
    };

    // §5.11.13 skip flag. Goavif uses ctx=0; we do too.
    let skip = td.decode_skip(0)?;

    // UV mode + CFL. For monochrome frames the UV path is elided.
    let num_planes = td.seq.color_config.num_planes;
    let mut uv_mode = y_mode;
    let mut angle_delta_uv: i8 = 0;
    let mut cfl_alpha_u: i32 = 0;
    let mut cfl_alpha_v: i32 = 0;
    if !fs.monochrome && num_planes >= 3 {
        // Goavif enables CFL for every block; the spec has a set of
        // size-gated allowed/denied rules but the bitstream signals
        // an out-of-band CFL=0 anyway when the block is ineligible.
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

    // Phase 2 boundary: every non-skip block would feed the
    // coefficient decoder (§5.11.39). That reader is Phase 3; for
    // now we surface a precise Unsupported at the first non-skip
    // leaf so callers see exactly where we stopped.
    if !skip {
        return Err(Error::unsupported(
            "av1 coefficient decode pending (§5.11.39)",
        ));
    }
    Ok(())
}
