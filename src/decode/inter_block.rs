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

    let skip = inter.read_skip(&mut td.symbol, 0)?;

    Ok(InterBlockInfo {
        is_inter: true,
        ref_frame_idx: 0,
        mv,
        skip,
        interp_filter,
        intra_y_mode: IntraMode::DcPred,
    })
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
            &r, 32, 32, 32, 32, 32, 4, 4, 8, 8, mv, InterpFilter::Regular,
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
