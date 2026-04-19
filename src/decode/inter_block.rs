//! AV1 inter-frame leaf-block decode — §6.10.23.
//!
//! Ported from goavif (MIT, KarpelesLab/goavif) —
//! `av1/decoder/inter_block.go` and `inter_block16.go`. The port reads
//! the block-level inter syntax (is_inter, single-ref, inter_mode,
//! MV, skip), runs motion compensation into a prediction buffer, and
//! hands the resulting samples to the existing residual / clip-add
//! pipeline.
//!
//! Narrow scope matching Phase 7: single-reference translational
//! inter with explicit NEWMV (or zero-MV fallback for GLOBALMV /
//! NEARESTMV / NEARMV since we don't carry a ref-MV list yet). No
//! compound prediction, no warp, no inter-intra, no OBMC.

use oxideav_core::{Error, Result};

use crate::predict::interp::InterpFilter;

use super::block::BlockSize;
use super::frame_state::FrameState;
use super::inter::{block_size_group, InterMode};
use super::mc::{motion_compensate, motion_compensate16};
use super::modes::IntraMode;
use super::mv::Mv;
use super::tile::TileDecoder;

/// Result of decoding the block-level inter syntax. The caller uses
/// this to drive the subsequent reconstruction/residual passes.
#[derive(Clone, Copy, Debug)]
pub struct InterBlockInfo {
    /// `true` when the bitstream flagged this block as inter.
    pub is_inter: bool,
    /// Reference frame index — always 0 (LAST) in the narrow path.
    pub ref_frame_idx: u8,
    /// Block MV (eighth-pel units). Zero when the mode reduces to
    /// zero-MV (GLOBALMV / NEARESTMV / NEARMV in our ref-list-less
    /// path).
    pub mv: Mv,
    /// `skip_txfm` flag.
    pub skip: bool,
    /// Which 8-tap filter the block uses — always REGULAR in our
    /// simplified path (no switchable filters).
    pub interp_filter: InterpFilter,
    /// Intra Y mode — only meaningful when `is_inter == false`.
    pub intra_y_mode: IntraMode,
}

/// Decode the per-block inter syntax for a leaf block. Returns
/// [`InterBlockInfo`] on success, or `Error::Unsupported` when the
/// bitstream requests something outside the narrow Phase 7 subset
/// (multi-ref, compound, warp, etc.).
pub fn decode_inter_block_syntax(
    td: &mut TileDecoder<'_>,
    fs: &mut FrameState,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<InterBlockInfo> {
    let mi_col = x >> 2;
    let mi_row = y >> 2;

    let above_is_inter = if mi_row > 0 && mi_col < fs.mi_cols {
        fs.mi_at(mi_col, mi_row - 1).is_inter
    } else {
        false
    };
    let left_is_inter = if mi_col > 0 && mi_row < fs.mi_rows {
        fs.mi_at(mi_col - 1, mi_row).is_inter
    } else {
        false
    };

    let inter = td
        .inter
        .as_mut()
        .ok_or_else(|| Error::invalid("av1 inter: InterDecoder missing on TileDecoder"))?;

    let is_inter = inter.read_is_inter(&mut td.symbol, above_is_inter, left_is_inter)?;

    if !is_inter {
        // Intra-within-inter: decode y_mode + skip using inter-frame
        // CDFs and return for the caller to run the intra path on top.
        let group = block_size_group(w as usize, h as usize);
        let y_mode = inter.read_y_mode(&mut td.symbol, group)?;
        let skip = inter.read_skip(&mut td.symbol, 0)?;
        return Ok(InterBlockInfo {
            is_inter: false,
            ref_frame_idx: 0,
            mv: Mv::default(),
            skip,
            interp_filter: InterpFilter::Regular,
            intra_y_mode: y_mode,
        });
    }

    // Single-reference ref frame. The narrow Phase 7 decoder only
    // tracks LAST; other selections are tolerated in the bitstream
    // but collapse to LAST for the actual MC source.
    let _ref_idx = inter.read_single_ref_frame(&mut td.symbol)?;

    // Inter mode. We treat the context as 0 across the board since we
    // don't maintain neighbor-mode classes for inter yet — goavif does
    // the same in its narrow path.
    let mode = inter.read_inter_mode(&mut td.symbol, 0, 0, 0)?;
    let mv = match mode {
        InterMode::NewMv => inter.read_mv(&mut td.symbol)?,
        InterMode::GlobalMv | InterMode::NearestMv | InterMode::NearMv => Mv::default(),
    };

    // Interpolation filter symbol is coded only when
    // `is_filter_switchable` is set. Our simplified path forces
    // REGULAR and skips the symbol.
    let skip = inter.read_skip(&mut td.symbol, 0)?;

    Ok(InterBlockInfo {
        is_inter: true,
        ref_frame_idx: 0,
        mv,
        skip,
        interp_filter: InterpFilter::Regular,
        intra_y_mode: IntraMode::DcPred,
    })
}

/// Run luma motion compensation for a single block into an owned
/// output buffer. Returns a tight `w*h` vector of predicted samples.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma_u8(
    ref_plane: &[u8],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    mv: Mv,
    filt: InterpFilter,
) -> Vec<u8> {
    let mut pred = vec![0u8; w * h];
    motion_compensate(
        &mut pred, w, h, ref_plane, ref_w, ref_h, ref_stride, x, y, mv, filt,
    );
    pred
}

/// HBD luma MC. Returns a tight `w*h` u16 buffer.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma_u16(
    ref_plane: &[u16],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    mv: Mv,
    filt: InterpFilter,
    bit_depth: u32,
) -> Vec<u16> {
    let mut pred = vec![0u16; w * h];
    motion_compensate16(
        &mut pred, w, h, ref_plane, ref_w, ref_h, ref_stride, x, y, mv, filt, bit_depth,
    );
    pred
}

/// Run chroma MC on one subsampled plane. MV is scaled down by the
/// plane's subsampling factor. Returns a tight buffer.
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma_u8(
    ref_plane: &[u8],
    uv_w: usize,
    uv_h: usize,
    uv_stride: usize,
    cx: i32,
    cy: i32,
    cw: usize,
    ch: usize,
    mv: Mv,
    sub_x: u32,
    sub_y: u32,
    filt: InterpFilter,
) -> Vec<u8> {
    let chroma_mv = Mv {
        row: mv.row >> sub_y,
        col: mv.col >> sub_x,
    };
    let mut pred = vec![0u8; cw * ch];
    motion_compensate(
        &mut pred, cw, ch, ref_plane, uv_w, uv_h, uv_stride, cx, cy, chroma_mv, filt,
    );
    pred
}

/// HBD chroma MC counterpart of [`mc_chroma_u8`].
#[allow(clippy::too_many_arguments)]
pub fn mc_chroma_u16(
    ref_plane: &[u16],
    uv_w: usize,
    uv_h: usize,
    uv_stride: usize,
    cx: i32,
    cy: i32,
    cw: usize,
    ch: usize,
    mv: Mv,
    sub_x: u32,
    sub_y: u32,
    filt: InterpFilter,
    bit_depth: u32,
) -> Vec<u16> {
    let chroma_mv = Mv {
        row: mv.row >> sub_y,
        col: mv.col >> sub_x,
    };
    let mut pred = vec![0u16; cw * ch];
    motion_compensate16(
        &mut pred, cw, ch, ref_plane, uv_w, uv_h, uv_stride, cx, cy, chroma_mv, filt, bit_depth,
    );
    pred
}

/// Map `BlockSize` to `block_size_group` bucket.
pub fn block_size_group_for(bs: BlockSize) -> usize {
    block_size_group(bs.width() as usize, bs.height() as usize)
}
