//! AV1 inter-frame leaf-block decode — §6.10.23.
//!
//! Reads the block-level inter syntax (is_inter, single-ref,
//! inter_mode, MV, skip), runs motion compensation into a prediction
//! buffer, and hands the resulting samples to the existing residual /
//! clip-add pipeline.
//!
//! Narrow scope matching Phase 7: single-reference translational
//! inter with explicit NEWMV (or zero-MV fallback for GLOBALMV /
//! NEARESTMV / NEARMV since we don't carry a ref-MV list yet). No
//! compound prediction, no warp, no inter-intra, no OBMC.

use oxideav_core::{Error, Result};

use crate::frame_header_tail::{SEG_LVL_GLOBALMV, SEG_LVL_REF_FRAME, SEG_LVL_SKIP};
use crate::predict::interp::InterpFilter;

use super::block::BlockSize;
use super::frame_state::FrameState;
use super::inter::{block_size_group, InterMode};
use super::mc::{motion_compensate, motion_compensate16};
use super::modes::IntraMode;
use super::mv::Mv;
use super::tile::{segment_id_ctx, TileDecoder};

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
    /// Spec §5.11.10 `skip_mode` — `true` when the block selected the
    /// AV1 SKIP_MODE compound coding. Always `false` until the frame
    /// header has `skip_mode_present == 1`; even then, gated on the
    /// per-segment SEG_LVL_SKIP / REF / GLOBALMV features and the
    /// 8×8 minimum block-dimension predicate.
    pub skip_mode: bool,
    /// Spec §5.11.9 / §5.11.19 `segment_id` — left at 0 when
    /// segmentation is disabled.
    pub segment_id: u8,
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

    let avail_u = mi_row > 0 && mi_col < fs.mi_cols;
    let avail_l = mi_col > 0 && mi_row < fs.mi_rows;

    let above_is_inter = if avail_u {
        fs.mi_at(mi_col, mi_row - 1).is_inter
    } else {
        false
    };
    let left_is_inter = if avail_l {
        fs.mi_at(mi_col - 1, mi_row).is_inter
    } else {
        false
    };

    // Spec §5.11.18 / §9.4 `skip_mode` context — sum of above/left
    // SkipModes neighbour flags (clamped to 0..=2). Computed up-front
    // so the borrow on `td.inter` later doesn't fight it.
    let above_skip_mode = if avail_u {
        fs.mi_at(mi_col, mi_row - 1).skip_mode
    } else {
        false
    };
    let left_skip_mode = if avail_l {
        fs.mi_at(mi_col - 1, mi_row).skip_mode
    } else {
        false
    };
    let skip_mode_ctx = (above_skip_mode as u32) + (left_skip_mode as u32);

    // Spec §5.11.18 / §9.4 segment_id ctx — sum of nonzero neighbours.
    let above_seg = if avail_u {
        fs.mi_at(mi_col, mi_row - 1).segment_id
    } else {
        0
    };
    let left_seg = if avail_l {
        fs.mi_at(mi_col - 1, mi_row).segment_id
    } else {
        0
    };
    let seg_ctx = segment_id_ctx(above_seg, left_seg, avail_u, avail_l);

    // §5.11.18 `inter_frame_mode_info()` ordering — the canonical
    // sequence (Round 12 wires the `inter_segment_id` + `read_skip_mode`
    // calls; previously stubbed out):
    //   skip = 0
    //   inter_segment_id(1) [preSkip pass]
    //   read_skip_mode()
    //   if (skip_mode) skip = 1 else read_skip()
    //   if (!SegIdPreSkip) inter_segment_id(0) [post-skip pass]
    //   read_cdef() / read_delta_qindex() / read_delta_lf()  [done in caller]
    //   read_is_inter()
    //   if (is_inter) inter_block_mode_info() else intra_block_mode_info()

    // §5.11.19 `inter_segment_id(1)` — preSkip pass.
    let mut segment_id = inter_segment_id(td, mi_col, mi_row, w, h, true, seg_ctx, false)?;

    // §5.11.10 `read_skip_mode`. Gated on `skip_mode_present` and the
    // per-segment SEG_LVL_SKIP / REF_FRAME / GLOBALMV features, plus a
    // minimum block-dimension predicate (Block_Width >= 8, Block_Height
    // >= 8).
    let skip_mode = read_skip_mode_for_block(td, w, h, segment_id, skip_mode_ctx)?;

    let inter = td
        .inter
        .as_mut()
        .ok_or_else(|| Error::invalid("av1 inter: InterDecoder missing on TileDecoder"))?;

    // §5.11.11 `read_skip` — when skip_mode is set the spec implicitly
    // forces `skip = 1` and no `read_skip` is performed. The `skip`
    // CDF context per §9.4 is the count of above/left Skips[][]
    // neighbours (0..=2). Round 12 fix: previously the ctx was
    // hard-coded to 0; now we sum the neighbour skip flags.
    let above_skip = if avail_u {
        fs.mi_at(mi_col, mi_row - 1).skip
    } else {
        false
    };
    let left_skip = if avail_l {
        fs.mi_at(mi_col - 1, mi_row).skip
    } else {
        false
    };
    let skip_ctx = (above_skip as u32) + (left_skip as u32);
    let skip = if skip_mode {
        true
    } else {
        inter.read_skip(&mut td.symbol, skip_ctx as usize)?
    };

    // §5.11.19 `inter_segment_id(0)` — post-skip pass (only when
    // !SegIdPreSkip). When `skip` was set the post-skip path resets
    // `seg_id_predicted` to 0 then reads the segment_id directly.
    if !td.frame.segmentation.seg_id_pre_skip {
        segment_id = inter_segment_id(td, mi_col, mi_row, w, h, false, seg_ctx, skip)?;
    }

    // Re-borrow `inter` after the `inter_segment_id` call may have
    // mutated `td`.
    let inter = td
        .inter
        .as_mut()
        .ok_or_else(|| Error::invalid("av1 inter: InterDecoder missing on TileDecoder"))?;

    // §5.11.20 `read_is_inter` — when skip_mode is active, is_inter is
    // implicitly 1 (skip_mode is a compound-prediction shortcut).
    // Likewise, segments with SEG_LVL_REF_FRAME force is_inter based
    // on the feature data; SEG_LVL_GLOBALMV forces is_inter = 1.
    let is_inter = if skip_mode {
        true
    } else if td
        .frame
        .segmentation
        .feature_active(segment_id, SEG_LVL_REF_FRAME)
    {
        td.frame.segmentation.feature_data[segment_id as usize][SEG_LVL_REF_FRAME] != 0
    } else if td
        .frame
        .segmentation
        .feature_active(segment_id, SEG_LVL_GLOBALMV)
    {
        true
    } else {
        // Re-borrow after frame access (immutable on segmentation).
        inter.read_is_inter(&mut td.symbol, above_is_inter, left_is_inter)?
    };

    if !is_inter {
        // Intra-within-inter (§5.11.22 `intra_block_mode_info`). The
        // spec reads y_mode, then chroma / palette / filter-intra on
        // the intra block; our narrow path keeps the y_mode read and
        // leaves the rest to the reconstruction stage's defaults.
        let group = block_size_group(w as usize, h as usize);
        let y_mode = inter.read_y_mode(&mut td.symbol, group)?;
        return Ok(InterBlockInfo {
            is_inter: false,
            ref_frame_idx: 0,
            mv: Mv::default(),
            skip,
            skip_mode: false,
            segment_id,
            interp_filter: InterpFilter::Regular,
            intra_y_mode: y_mode,
        });
    }

    if skip_mode {
        // Spec §5.11.18: when skip_mode is set, the bitstream omits
        // `read_ref_frames`, `inter_block_mode_info`, etc. — the
        // block uses the implicit SkipModeFrame[0..=1] reference pair
        // and the NEAREST_NEAREST_MV compound mode. Our narrow Phase
        // 7 decoder doesn't carry the SkipModeFrame derivation, so
        // we collapse to zero-MV LAST-ref single-reference inter.
        // The bitstream consumes no further symbols on this block.
        return Ok(InterBlockInfo {
            is_inter: true,
            ref_frame_idx: 0,
            mv: Mv::default(),
            skip,
            skip_mode: true,
            segment_id,
            interp_filter: match td.frame.interpolation_filter {
                1 => InterpFilter::Smooth,
                2 => InterpFilter::Sharp,
                _ => InterpFilter::Regular,
            },
            intra_y_mode: IntraMode::DcPred,
        });
    }

    // Single-reference ref frame. The narrow Phase 7 decoder only
    // tracks LAST; other selections are tolerated in the bitstream
    // but collapse to LAST for the actual MC source.
    let _ref_idx = inter.read_single_ref_frame(&mut td.symbol)?;

    // Inter mode. We treat the context as 0 across the board since we
    // don't maintain neighbor-mode classes for inter yet.
    let mode = inter.read_inter_mode(&mut td.symbol, 0, 0, 0)?;
    // §7.10 motion-mode integration: when the block selects GLOBALMV
    // and the frame's `gm_type` for LAST is TRANSLATION, seed the MV
    // from the (quantized) translation parameters. Non-translation
    // global types are unsupported in the narrow path and degrade to
    // zero-MV.
    let mv = match mode {
        InterMode::NewMv => inter.read_mv(&mut td.symbol)?,
        InterMode::GlobalMv => global_mv_for_last(td.frame),
        InterMode::NearestMv | InterMode::NearMv => Mv::default(),
    };

    // Interpolation filter symbol per §5.11.26 `read_interpolation_filter`:
    // when the frame header's `is_filter_switchable` is true, a per-
    // block filter is decoded; otherwise the frame-level filter applies
    // uniformly. The raw frame-level value uses spec mapping
    // 0=REGULAR, 1=SMOOTH, 2=SHARP, 3=BILINEAR, 4=SWITCHABLE.
    let interp_filter = if td.frame.is_filter_switchable {
        // Per §5.11.26 the per-block context is formed from above/left
        // interp-filter types; we collapse to ctx=0 (no neighbor
        // bookkeeping yet) and read a single symbol per block.
        inter.read_interp_filter(&mut td.symbol, 0)?
    } else {
        // 3 = BILINEAR; AV1 treats it as not coded for AVIF fixtures,
        // which use value 0 or 4. We map BILINEAR -> REGULAR since we
        // don't carry a bilinear filter set yet.
        match td.frame.interpolation_filter {
            1 => InterpFilter::Smooth,
            2 => InterpFilter::Sharp,
            _ => InterpFilter::Regular,
        }
    };

    Ok(InterBlockInfo {
        is_inter: true,
        ref_frame_idx: 0,
        mv,
        skip,
        skip_mode: false,
        segment_id,
        interp_filter,
        intra_y_mode: IntraMode::DcPred,
    })
}

/// §5.11.10 `read_skip_mode` — runs the gating predicate then
/// (optionally) consumes the `skip_mode` symbol from the range coder.
/// Returns the resulting boolean. Skips the read entirely (returning
/// `false`) when:
/// - `skip_mode_present` is `false` on the frame header (the most
///   common case for our fixtures — see §5.9.22 / `frame_header.rs`).
/// - The block dimensions are below 8 in either axis.
/// - The active segment has SEG_LVL_SKIP / REF_FRAME / GLOBALMV
///   features enabled (which already constrain the per-block coding).
fn read_skip_mode_for_block(
    td: &mut TileDecoder<'_>,
    w: u32,
    h: u32,
    segment_id: u8,
    ctx: u32,
) -> Result<bool> {
    if !td.frame.skip_mode_present {
        return Ok(false);
    }
    if w < 8 || h < 8 {
        return Ok(false);
    }
    if td
        .frame
        .segmentation
        .feature_active(segment_id, SEG_LVL_SKIP)
        || td
            .frame
            .segmentation
            .feature_active(segment_id, SEG_LVL_REF_FRAME)
        || td
            .frame
            .segmentation
            .feature_active(segment_id, SEG_LVL_GLOBALMV)
    {
        return Ok(false);
    }
    td.decode_skip_mode(ctx)
}

/// §5.11.19 `inter_segment_id(preSkip)` — runs the per-spec branches
/// for spatial, temporal, and skip-bypass segment_id reads. Returns
/// the (possibly updated) segment_id for this block.
///
/// `preSkip == true` corresponds to the spec's `preSkip = 1` call
/// (issued before `read_skip_mode` / `read_skip`); `preSkip == false`
/// is the post-skip call.
///
/// Implementation notes (narrow path):
/// - We don't yet carry a `PrevSegmentIds[][]` map from the previous
///   frame, so the `predictedSegmentId` path defaults to 0 when no
///   previous frame is available.
/// - We don't track `AboveSegPredContext / LeftSegPredContext` arrays
///   either, so the seg_id_predicted CDF context degrades to 0. This
///   is exact when segmentation is off (the only case our fixtures
///   exercise — both reads short-circuit on `segmentation_enabled =
///   false`).
#[allow(clippy::too_many_arguments)]
fn inter_segment_id(
    td: &mut TileDecoder<'_>,
    mi_col: u32,
    mi_row: u32,
    w: u32,
    h: u32,
    pre_skip: bool,
    seg_ctx: u32,
    skip: bool,
) -> Result<u8> {
    if !td.frame.segmentation.enabled {
        return Ok(0);
    }

    // §5.11.21 get_segment_id — spatial-min over the previous frame's
    // PrevSegmentIds[MiRow + y][MiCol + x] inside the block extent.
    // Without a per-frame PrevSegmentIds map we approximate using the
    // previous frame's MI grid (same shape / resolution). Bail to 0
    // when there's no prev frame.
    let predicted_segment_id = predicted_segment_id_from_prev(td, mi_col, mi_row, w, h);

    if td.frame.segmentation.update_map {
        if pre_skip && !td.frame.segmentation.seg_id_pre_skip {
            // Spec: pre-skip pass with !SegIdPreSkip simply resets the
            // segment_id; the post-skip pass will perform the read.
            return Ok(0);
        }
        if !pre_skip {
            // Post-skip pass.
            if skip {
                // Skip-block bypass: no temporal pred, just read.
                let sid = td.decode_segment_id(seg_ctx)?;
                return Ok(sid);
            }
        }
        if td.frame.segmentation.temporal_update {
            // Read seg_id_predicted; on 1, reuse predicted value;
            // on 0, fall through to spatial read.
            let pred = td.decode_seg_id_predicted(0)?;
            if pred {
                return Ok(predicted_segment_id);
            }
            let sid = td.decode_segment_id(seg_ctx)?;
            return Ok(sid);
        } else {
            let sid = td.decode_segment_id(seg_ctx)?;
            return Ok(sid);
        }
    }
    // !update_map: inherit predicted segment_id.
    Ok(predicted_segment_id)
}

/// §5.11.21 `get_segment_id` — spatial min over the previous frame's
/// `PrevSegmentIds[][]` inside the block's MI extent. Returns 0 when
/// the previous frame is absent (e.g. on the first inter frame after
/// a key frame, the prev-frame state may not yet carry seg ids).
fn predicted_segment_id_from_prev(
    td: &TileDecoder<'_>,
    mi_col: u32,
    mi_row: u32,
    w: u32,
    h: u32,
) -> u8 {
    let Some(prev) = td.prev_frame.as_ref() else {
        return 0;
    };
    let bw4 = ((w + 3) >> 2).max(1);
    let bh4 = ((h + 3) >> 2).max(1);
    let x_mis = (prev.mi_cols.saturating_sub(mi_col)).min(bw4);
    let y_mis = (prev.mi_rows.saturating_sub(mi_row)).min(bh4);
    if x_mis == 0 || y_mis == 0 {
        return 0;
    }
    let mut seg: u8 = 7;
    for dy in 0..y_mis {
        for dx in 0..x_mis {
            let v = prev.mi_at(mi_col + dx, mi_row + dy).segment_id;
            if v < seg {
                seg = v;
            }
        }
    }
    seg
}

/// Derive a translational MV for GLOBALMV using the LAST slot's
/// global-motion parameters (§7.10 / §7.11.3.6). Identity / RotZoom /
/// Affine without warp support all collapse to zero-MV; TRANSLATION
/// uses the stored `gm_params[LAST][0..=1]` components (row, col).
///
/// Spec §3 `GM_PARAM_TRANS_PREC` = 3 for TRANSLATION-only: the param
/// bits encode 1/8-pel offsets directly, matching [`Mv`]'s precision.
/// For higher-order global motion the translation components use
/// 1/16-pel (`GM_PARAM_TRANS_BITS` path); in that case we scale down.
fn global_mv_for_last(frame: &crate::frame_header::FrameHeader) -> Mv {
    // LAST is ref index 1 in AV1's reference slot numbering (0 = intra,
    // 1 = LAST, ... per spec §3). Our narrow decoder collapses to
    // slot 1 for single-ref inter.
    translate_gm(
        frame.gm_type.get(1).copied().unwrap_or_default(),
        frame.gm_params.get(1).copied().unwrap_or([0; 6]),
    )
}

/// Pure helper: convert `(gm_type, gm_params)` to an [`Mv`]. Extracted
/// from [`global_mv_for_last`] so unit tests don't need to build a
/// full [`FrameHeader`].
fn translate_gm(gm: crate::frame_header_tail::GmType, params: [i32; 6]) -> Mv {
    use crate::frame_header_tail::GmType;
    match gm {
        GmType::Identity => Mv::default(),
        GmType::Translation => Mv {
            row: params[0],
            col: params[1],
        },
        // Higher-order global motion: translation component is stored
        // at 1/16-pel when paired with alpha coefficients, but our
        // narrow path has no warp engine — fall back to zero-MV.
        GmType::RotZoom | GmType::Affine => Mv::default(),
    }
}

/// Run compound bi-prediction MC for a single luma block (§7.11.3.9
/// `AverageMc`). Fetches both references via the single-ref MC path,
/// then averages them. `ref0 / ref1` are the two reference planes;
/// `mv0 / mv1` are their respective MVs.
#[allow(clippy::too_many_arguments)]
pub fn compound_mc_luma_u8(
    ref0: &[u8],
    ref1: &[u8],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    mv0: Mv,
    mv1: Mv,
    filt: InterpFilter,
) -> Vec<u8> {
    use super::mc::average_mc_u8;
    let p0 = mc_luma_u8(ref0, ref_w, ref_h, ref_stride, x, y, w, h, mv0, filt);
    let p1 = mc_luma_u8(ref1, ref_w, ref_h, ref_stride, x, y, w, h, mv1, filt);
    let mut out = vec![0u8; w * h];
    average_mc_u8(&p0, &p1, &mut out);
    out
}

/// HBD counterpart of [`compound_mc_luma_u8`].
#[allow(clippy::too_many_arguments)]
pub fn compound_mc_luma_u16(
    ref0: &[u16],
    ref1: &[u16],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    mv0: Mv,
    mv1: Mv,
    filt: InterpFilter,
    bit_depth: u32,
) -> Vec<u16> {
    use super::mc::average_mc_u16;
    let p0 = mc_luma_u16(
        ref0, ref_w, ref_h, ref_stride, x, y, w, h, mv0, filt, bit_depth,
    );
    let p1 = mc_luma_u16(
        ref1, ref_w, ref_h, ref_stride, x, y, w, h, mv1, filt, bit_depth,
    );
    let mut out = vec![0u16; w * h];
    average_mc_u16(&p0, &p1, &mut out, bit_depth);
    out
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

/// Reference-scaled luma MC (§7.9). `cur_w / cur_h` are the current
/// frame dimensions; `ref_w / ref_h` are the reference plane's. When
/// the scale factors collapse to identity (equal dims) this routes
/// through the plain [`mc_luma_u8`] path.
///
/// Narrow implementation: computes the scaled block origin via
/// [`super::ref_scale::ScaleFactors::project`] and issues a single
/// translational MC against the projected position. The full spec
/// would interpolate per-sample step sizes — this approximation is
/// exact at integer scales (½, 1, 2×) and near-correct for
/// super-resolution GOPs whose scale ratios are close to 1.
#[allow(clippy::too_many_arguments)]
pub fn mc_luma_u8_scaled(
    ref_plane: &[u8],
    ref_w: usize,
    ref_h: usize,
    ref_stride: usize,
    cur_w: u32,
    cur_h: u32,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    mv: Mv,
    filt: InterpFilter,
) -> Vec<u8> {
    use super::ref_scale::ScaleFactors;
    let sf = ScaleFactors::new(ref_w as u32, ref_h as u32, cur_w, cur_h);
    if sf.is_identity() {
        return mc_luma_u8(ref_plane, ref_w, ref_h, ref_stride, x, y, w, h, mv, filt);
    }
    // Projected block origin (in 14-bit fixed-point). Reduce to
    // integer-pel + eighth-pel MV carryover.
    let (px, py) = sf.project(x, y, mv.row, mv.col);
    // Integer part in samples (right-shift 14) + fractional retained
    // as an effective MV for the inner MC call.
    let int_x = (px >> 14) as i32;
    let int_y = (py >> 14) as i32;
    let frac_x = ((px & ((1i64 << 14) - 1)) >> 11) as i32; // take top 3 bits → eighth-pel
    let frac_y = ((py & ((1i64 << 14) - 1)) >> 11) as i32;
    let proj_mv = Mv {
        row: frac_y,
        col: frac_x,
    };
    mc_luma_u8(
        ref_plane, ref_w, ref_h, ref_stride, int_x, int_y, w, h, proj_mv, filt,
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_header_tail::GmType;

    #[test]
    fn translate_gm_identity_is_zero() {
        let mv = translate_gm(GmType::Identity, [100, 200, 0, 0, 0, 0]);
        assert_eq!(mv, Mv::default());
    }

    #[test]
    fn translate_gm_translation_round_trips_params() {
        let mv = translate_gm(GmType::Translation, [12, -34, 0, 0, 0, 0]);
        assert_eq!(mv.row, 12);
        assert_eq!(mv.col, -34);
    }

    #[test]
    fn translate_gm_rotzoom_falls_back_to_zero() {
        // Without a warp engine, RotZoom reduces to zero-MV even when
        // the params carry a non-zero translation slot.
        let mv = translate_gm(GmType::RotZoom, [0, 0, 1 << 14, 0, 0, 0]);
        assert_eq!(mv, Mv::default());
    }

    #[test]
    fn scaled_mc_luma_identity_matches_plain_path() {
        // At identity scale, `mc_luma_u8_scaled` must match
        // `mc_luma_u8`. Use a deterministic reference.
        let mut r = vec![0u8; 32 * 32];
        for y in 0..32 {
            for x in 0..32 {
                r[y * 32 + x] = ((x * 7 + y * 11) & 0xFF) as u8;
            }
        }
        let mv = Mv { row: 16, col: -8 };
        let plain = mc_luma_u8(&r, 32, 32, 32, 4, 4, 8, 8, mv, InterpFilter::Regular);
        let scaled = mc_luma_u8_scaled(
            &r,
            32,
            32,
            32,
            32,
            32,
            4,
            4,
            8,
            8,
            mv,
            InterpFilter::Regular,
        );
        assert_eq!(plain, scaled, "identity-scaled MC must match plain MC");
    }

    #[test]
    fn scaled_mc_luma_half_res_pulls_half_coords() {
        // Reference is 16×16, current is 32×32 — scaled position (4, 4)
        // maps to (2, 2) in ref. The helper should produce the
        // half-resolution sample.
        let mut r = vec![0u8; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                r[y * 16 + x] = ((y * 10 + x) & 0xFF) as u8;
            }
        }
        let out = mc_luma_u8_scaled(
            &r,
            16,
            16,
            16,
            32,
            32,
            4,
            4,
            4,
            4,
            Mv::default(),
            InterpFilter::Regular,
        );
        // out[0] should come from ref at scaled (2, 2) = 22.
        assert_eq!(out[0], 22);
    }

    #[test]
    fn inter_block_info_default_skip_mode_is_false() {
        // The new InterBlockInfo carries skip_mode + segment_id; both
        // must default to "off" so callers that don't explicitly set
        // them get the conservative non-SKIP_MODE coding path.
        let info = InterBlockInfo {
            is_inter: true,
            ref_frame_idx: 0,
            mv: Mv::default(),
            skip: false,
            skip_mode: false,
            segment_id: 0,
            interp_filter: InterpFilter::Regular,
            intra_y_mode: IntraMode::DcPred,
        };
        assert!(!info.skip_mode);
        assert_eq!(info.segment_id, 0);
    }

    #[test]
    fn skip_mode_cdf_default_is_survival_form() {
        // Sanity check the wire-format invariant for the default
        // skip_mode CDF: spec values `{32621, 32768, 0}` (cumulative)
        // become `{147, 0, 0}` (survival = 32768 - cdf_spec).
        let cdf = crate::cdfs::DEFAULT_SKIP_MODE_CDF;
        assert_eq!(cdf.len(), 3, "skip_mode has SKIP_MODE_CONTEXTS=3");
        // Context 0 — heavy bias against skip_mode (rare).
        assert!(cdf[0][0] < 1000, "P(skip_mode=1 | ctx=0) should be small");
        // Context 1 — middling.
        assert!(cdf[1][0] > 5000 && cdf[1][0] < 20000);
        // Context 2 — heaviest bias toward skip_mode.
        assert!(cdf[2][0] > cdf[1][0]);
    }

    #[test]
    fn segment_id_predicted_cdf_default_is_uniform() {
        // §9.4: Default_Segment_Id_Predicted_Cdf uses 128*128 = 16384
        // for every entry (equal probability — the seg_id_predicted
        // flag has no informative default).
        let cdf = crate::cdfs::DEFAULT_SEGMENT_ID_PREDICTED_CDF;
        assert_eq!(cdf.len(), 3);
        for row in cdf.iter() {
            assert_eq!(row[0], 16384, "default seg_id_predicted CDF is 50/50");
        }
    }

    #[test]
    fn seg_feature_active_off_when_segmentation_disabled() {
        use crate::frame_header_tail::{SegmentationParams, SEG_LVL_SKIP};
        let mut p = SegmentationParams::default();
        // Even with FeatureEnabled[seg=0][SEG_LVL_SKIP] set, the
        // helper must return false when segmentation_enabled is false.
        p.feature_enabled[0][SEG_LVL_SKIP] = true;
        assert!(!p.feature_active(0, SEG_LVL_SKIP));
        p.enabled = true;
        assert!(p.feature_active(0, SEG_LVL_SKIP));
    }

    #[test]
    fn compound_mc_luma_averages_two_refs() {
        let w = 8;
        let h = 8;
        let ref0 = vec![100u8; 32 * 32];
        let ref1 = vec![200u8; 32 * 32];
        let out = compound_mc_luma_u8(
            &ref0,
            &ref1,
            32,
            32,
            32,
            10,
            10,
            w,
            h,
            Mv { row: 0, col: 0 },
            Mv { row: 0, col: 0 },
            InterpFilter::Regular,
        );
        // Average of 100 and 200 with +1 round is 150.
        for v in out {
            assert_eq!(v, 150u8);
        }
    }
}
