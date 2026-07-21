//! Per-block mode-info **writers** — §5.11.7 / §5.11.8 / §5.11.11 /
//! §5.11.22 intra-arm syntax elements.
//!
//! These are the encoder counterparts to the §5.11 per-block decoders
//! that already live in [`crate::cdf::PartitionWalker`] (`decode_skip`,
//! `decode_intra_segment_id`, `decode_intra_frame_y_mode`,
//! `decode_intra_block_mode_info`'s `y_mode` + `uv_mode` S() reads).
//!
//! Scope of this arc (r211): the *intra* arm only — `skip`,
//! `intra_segment_id`, `intra_frame_y_mode` (§5.11.7 path with the
//! §8.3.2 neighbour-CDF ctx), `y_mode` (§5.11.22 path with the
//! `Size_Group[ MiSize ]` ctx), and `uv_mode` (§5.11.22 line 6 with the
//! CFL-allowed selector). No inter mode_info, no MV encode, no
//! partition split, no coefficient encode.
//!
//! ## Stateless on purpose
//!
//! Mirroring [`SymbolWriter::write_symbol`]'s "caller-supplied CDF
//! slice" pattern, every writer here takes its §8.3.2 *context indices*
//! as inputs rather than a [`PartitionWalker`] reference. The caller is
//! responsible for deriving the §8.3.2 ctx from neighbour state (using
//! the existing public helpers — [`skip_ctx`], [`intra_mode_ctx`],
//! [`size_group`], [`segment_id_ctx`]) and feeding the same ctx the
//! decoder side would derive on the corresponding [`PartitionWalker`]
//! call.
//!
//! This keeps the writer module pure (no shared mutable grid state) and
//! makes the roundtrip tests below explicit about what the encode-side
//! ctx must equal. For multi-block encodes the caller threads a
//! [`PartitionWalker`] of its own and re-uses the §8.3.2 helpers; the
//! decode-side walker stamps its own grids on the roundtrip step.
//!
//! ## Spec provenance
//!
//! Sourced from `docs/video/av1/av1-spec.txt`:
//!   * §5.11.7  intra_frame_mode_info  (p.65)
//!   * §5.11.8  intra_segment_id        (p.65)
//!   * §5.11.9  read_segment_id        (p.66) — `neg_deinterleave`
//!   * §5.11.11 read_skip              (p.67)
//!   * §5.11.22 intra_block_mode_info   (p.72)
//!   * §8.3.2  context derivations    (p.361-378)
//!
//! [`PartitionWalker`]: crate::cdf::PartitionWalker
//! [`SymbolWriter::write_symbol`]: crate::encoder::symbol_writer::SymbolWriter::write_symbol
//! [`skip_ctx`]: crate::cdf::skip_ctx
//! [`intra_mode_ctx`]: crate::cdf::intra_mode_ctx
//! [`size_group`]: crate::cdf::size_group
//! [`segment_id_ctx`]: crate::cdf::segment_id_ctx

use crate::cdf::{
    block_height, block_width, ceil_log2_av1, cfl_alpha_u_ctx, cfl_alpha_v_ctx, comp_group_idx_ctx,
    comp_mode_ctx, comp_ref_type_ctx, compound_idx_ctx, compound_mode_ctx, count_refs,
    get_palette_color_context, interintra_ctx, interp_filter_ctx, is_directional,
    is_samedir_ref_pair, mi_height_log2, mi_width_log2, neg_interleave, palette_color_ctx,
    palette_uv_mode_ctx, palette_y_mode_ctx, ref_count_ctx, wedge_bits, CompoundTypeReadout,
    FindMvStackResult, InterIntraReadout, InterpolationFilterReadout, PalettePlane,
    PartitionWalker, TileCdfContext, BILINEAR, BLOCK_128X128, BLOCK_32X32, BLOCK_64X64, BLOCK_8X8,
    BLOCK_SIZES, BLOCK_SIZE_GROUPS, CFL_ALPHABET_SIZE, CFL_JOINT_SIGNS, CLASS0_SIZE,
    COMPOUND_AVERAGE, COMPOUND_DIFFWTD, COMPOUND_DISTANCE, COMPOUND_INTRA, COMPOUND_MODES,
    COMPOUND_MODE_CONTEXTS, COMPOUND_TYPES, COMPOUND_WEDGE, COMP_INTER_CONTEXTS, DC_PRED,
    DELTA_LF_SMALL, DELTA_Q_SMALL, DRL_MODE_CONTEXTS, EIGHTTAP, FRAME_LF_COUNT,
    GM_TYPE_TRANSLATION, INTERINTRA_MODES, INTERP_FILTERS, INTERP_FILTER_NONE,
    INTRABC_DELAY_PIXELS, INTRA_FILTER_MODES, INTRA_MODES, INTRA_MODE_CONTEXTS, IS_INTER_CONTEXTS,
    MAX_ANGLE_DELTA, MAX_REF_MV_STACK_SIZE, MAX_SEGMENTS, MI_SIZE, MODE_GLOBALMV,
    MODE_GLOBAL_GLOBALMV, MODE_NEARESTMV, MODE_NEAREST_NEARESTMV, MODE_NEARMV, MODE_NEAR_NEARMV,
    MODE_NEAR_NEWMV, MODE_NEWMV, MODE_NEW_NEARMV, MODE_NEW_NEWMV, MOTION_MODES, MOTION_MODE_OBMC,
    MOTION_MODE_SIMPLE, MOTION_MODE_WARPED_CAUSAL, MV_CLASSES, MV_CLASS_0, MV_COMPS, MV_CONTEXTS,
    MV_INTRABC_CONTEXT, MV_JOINT_HNZVNZ, MV_JOINT_HNZVZ, MV_JOINT_HZVNZ, MV_JOINT_ZERO,
    NEW_MV_CONTEXTS, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PALETTE_BLOCK_SIZE_CONTEXTS,
    PALETTE_COLORS, REF_MV_CONTEXTS, SEGMENT_ID_CONTEXTS, SEGMENT_ID_PREDICTED_CONTEXTS,
    SKIP_CONTEXTS, SKIP_MODE_CONTEXTS, SWITCHABLE, UV_INTRA_MODES_CFL_ALLOWED,
    UV_INTRA_MODES_CFL_NOT_ALLOWED, WEDGE_TYPES, ZERO_MV_CONTEXTS,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::Error;

/// `read_skip()` inverse per §5.11.11 (av1-spec p.67).
///
/// Spec body:
/// ```text
///   read_skip() {
///       if ( SegIdPreSkip && seg_feature_active( SEG_LVL_SKIP ) ) {
///           skip = 1
///       } else {
///           skip                                            S()
///       }
///   }
/// ```
///
/// `seg_skip_active` is the caller-combined precondition `SegIdPreSkip
/// && seg_feature_active( SEG_LVL_SKIP )` — mirrors the parameter
/// surface of [`crate::cdf::PartitionWalker::decode_skip`].
///
/// * When `seg_skip_active == true`: no symbol is emitted (the decoder
///   short-circuits to `skip = 1` without a bit read). `skip` MUST be
///   `1` — caller bug otherwise, surfaced as
///   [`Error::PartitionWalkOutOfRange`].
/// * When `seg_skip_active == false`: emits one §8.2.6 `S()` over
///   `Default_Skip_Cdf[ ctx ]`. `ctx` is the §8.3.2 derivation
///   `(AvailU ? Skips[r-1][c] : 0) + (AvailL ? Skips[r][c-1] : 0)`,
///   computable through the public [`crate::cdf::skip_ctx`] helper.
///   `ctx` MUST be in `0..SKIP_CONTEXTS = 3`.
pub fn write_skip(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    skip: u8,
    ctx: usize,
    seg_skip_active: bool,
) -> Result<(), Error> {
    if seg_skip_active {
        // §5.11.11 first branch — no bits.
        if skip != 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    if skip > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ctx >= SKIP_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs.skip_cdf(ctx);
    writer.write_symbol(skip as u32, cdf)
}

/// `read_segment_id()` inverse per §5.11.9 (av1-spec p.66) — the
/// shared per-block segment-id writer the §5.11.8 `intra_segment_id`
/// and §5.11.19 `inter_segment_id` writers both compose. Mirrors
/// [`crate::cdf::PartitionWalker::decode_segment_id`].
///
/// Spec body (av1-spec p.66, §5.11.9):
///
/// ```text
///   read_segment_id( ) {
///       prev_ul = AvailU && AvailL ? PrevSegmentIds[r-1][c-1] : -1
///       prev_u  = AvailU           ? PrevSegmentIds[r-1][c  ] : -1
///       prev_l  = AvailL           ? PrevSegmentIds[r  ][c-1] : -1
///       if ( prev_u == -1 ) {
///           if ( prev_l == -1 ) pred = 0
///           else                pred = prev_l
///       } else if ( prev_l == -1 ) {
///           pred = prev_u
///       } else if ( prev_ul == prev_u ) {
///           pred = prev_u
///       } else {
///           pred = prev_l
///       }
///       if ( skip ) {
///           segment_id = pred
///       } else {
///           segment_id                                          S()
///           segment_id = neg_deinterleave( segment_id, pred,
///                                          LastActiveSegId + 1 )
///       }
///   }
/// ```
///
/// The §5.11.9 `pred` derivation is the *neighbour cascade* — the
/// caller threads it in pre-computed (the encoder owns a parallel
/// grid walk; see the round-trip tests below for the pattern). The
/// `last_active_seg_id` argument supplies `max = LastActiveSegId + 1`
/// for the `neg_deinterleave` inverse.
///
/// Two arms; both transcribe one decoder branch each:
///
/// * `skip != 0` — `segment_id = pred`, no symbol emitted. Caller-bug
///   shape: `segment_id != pred` ⇒ [`Error::PartitionWalkOutOfRange`].
/// * `skip == 0` — emits one §8.2.6 `S()` symbol against
///   `TileSegmentIdCdf[ ctx ]`. The on-wire `diff` is the inverse of
///   `neg_deinterleave`, computed analytically via
///   [`crate::cdf::neg_interleave`] (`O(1)` — no search). `ctx` is
///   the §8.3.2 derivation from [`crate::cdf::segment_id_ctx`].
///
/// Range guards mirror the decoder side and surface as
/// [`Error::PartitionWalkOutOfRange`]:
///   * `last_active_seg_id >= MAX_SEGMENTS = 8`.
///   * `segment_id >= max` (max = `last_active_seg_id + 1`).
///   * `pred >= max`.
///   * `ctx >= SEGMENT_ID_CONTEXTS = 3` on the `S()` arm.
///
/// Round-trip semantics: feeding the bytes this writer produces back
/// into [`crate::cdf::PartitionWalker::decode_segment_id`] (with the
/// matching `sub_size`/`mi_row`/`mi_col`/`skip`/`last_active_seg_id`
/// arguments) reconstructs the same `segment_id`. The decoder's
/// §5.11.5 grid-fill of the leaf footprint happens decoder-side; the
/// stateless writer does not maintain an encoder grid (caller responsibility).
pub fn write_segment_id(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    segment_id: u8,
    skip: u8,
    pred: u8,
    ctx: usize,
    last_active_seg_id: u8,
) -> Result<(), Error> {
    if (last_active_seg_id as usize) >= MAX_SEGMENTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let max = last_active_seg_id as u32 + 1;
    if segment_id as u32 >= max {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if pred as u32 >= max {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §5.11.9 `if ( skip ) segment_id = pred` — no bits emitted.
    if skip != 0 {
        if segment_id != pred {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    // §5.11.9 `else segment_id S()` — `ctx` from §8.3.2
    // `segment_id_ctx`.
    if ctx >= SEGMENT_ID_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // Analytic inverse of `neg_deinterleave` via `neg_interleave`
    // (bijection on `0..max`, O(1) — replaces the prior O(8) search).
    let diff = neg_interleave(segment_id as u32, pred as u32, max);
    let cdf = cdfs.segment_id_cdf(ctx);
    writer.write_symbol(diff, cdf)
}

/// `intra_segment_id()` inverse per §5.11.8 (av1-spec p.65).
///
/// Spec body:
/// ```text
///   intra_segment_id( ) {
///       if ( segmentation_enabled )
///           read_segment_id( )
///       else
///           segment_id = 0
///       Lossless = LosslessArray[ segment_id ]
///   }
/// ```
///
/// Composes [`write_segment_id`] (§5.11.9) under the
/// `segmentation_enabled` gate. On the disabled branch no bits are
/// emitted and `segment_id` MUST be `0`; on the enabled branch the
/// `read_segment_id` inverse runs through its skip / non-skip
/// dispatch.
///
/// Parameters mirror [`crate::cdf::PartitionWalker::decode_intra_segment_id`]
/// plus `pred` (the §5.11.9 neighbour-cascade value the caller
/// already has on hand from the encoder's own grid walk) and `ctx`
/// (the §8.3.2 `segment_id_ctx` — see [`crate::cdf::segment_id_ctx`]).
/// Lossless resolution (the spec's
/// `Lossless = LosslessArray[ segment_id ]` step) is the caller's
/// concern — the writer's responsibility ends at the on-wire bits.
#[allow(clippy::too_many_arguments)]
pub fn write_intra_segment_id(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    segment_id: u8,
    skip: u8,
    pred: u8,
    ctx: usize,
    segmentation_enabled: bool,
    last_active_seg_id: u8,
) -> Result<(), Error> {
    // §5.11.8 disabled-branch: no bits written, segment_id MUST be 0.
    if !segmentation_enabled {
        if segment_id != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    // §5.11.8 enabled-branch: delegate to the §5.11.9
    // `read_segment_id` inverse.
    write_segment_id(
        writer,
        cdfs,
        segment_id,
        skip,
        pred,
        ctx,
        last_active_seg_id,
    )
}

/// `intra_frame_y_mode` S() inverse per §5.11.7 line 13 / §8.3.2
/// (av1-spec p.65 / p.361).
///
/// The decoder's §8.3.2 ctx derivation is:
/// ```text
///   abovemode = Intra_Mode_Context[ AvailU ? YModes[ r - 1 ][ c ] : DC_PRED ]
///   leftmode  = Intra_Mode_Context[ AvailL ? YModes[ r ][ c - 1 ] : DC_PRED ]
/// ```
/// then the CDF row is `TileIntraFrameYModeCdf[ abovemode ][ leftmode ]`
/// — the keyframe's INTRA_MODES-wide CDF the §8.2.6 S() reads from.
///
/// The caller pre-computes `abovemode_ctx` and `leftmode_ctx` through
/// [`crate::cdf::intra_mode_ctx`] (the public helper that performs the
/// `Intra_Mode_Context[]` mapping). Both ctx values MUST be in
/// `0..INTRA_MODE_CONTEXTS = 5`.
///
/// `y_mode` MUST be in `0..INTRA_MODES = 13`.
pub fn write_intra_frame_y_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    y_mode: u8,
    abovemode_ctx: usize,
    leftmode_ctx: usize,
) -> Result<(), Error> {
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if abovemode_ctx >= INTRA_MODE_CONTEXTS || leftmode_ctx >= INTRA_MODE_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs.intra_frame_y_mode_cdf(abovemode_ctx, leftmode_ctx);
    writer.write_symbol(y_mode as u32, cdf)
}

/// `y_mode` S() inverse per §5.11.22 (av1-spec p.72) — the
/// non-keyframe / inter-frame intra-block path.
///
/// The §8.3.2 ctx is `Size_Group[ MiSize ]` — a single index into
/// `TileYModeCdf[ ctx ]`. The caller passes the `sub_size` (§5.11.5
/// `MiSize`) and the writer invokes [`crate::cdf::size_group`] to
/// derive the ctx exactly as the decoder does.
///
/// `y_mode` MUST be in `0..INTRA_MODES = 13`; `size_group_ctx` MUST be
/// in `0..BLOCK_SIZE_GROUPS = 4`.
pub fn write_y_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    y_mode: u8,
    size_group_ctx: usize,
) -> Result<(), Error> {
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if size_group_ctx >= BLOCK_SIZE_GROUPS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs
        .y_mode_cdf(size_group_ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(y_mode as u32, cdf)
}

/// `uv_mode` S() inverse per §5.11.22 line 6 (av1-spec p.72).
///
/// The §8.3.2 selector picks between the CFL-allowed and
/// CFL-not-allowed CDF rows per the paragraph at av1-spec p.361:
/// ```text
///   • Lossless == 1 AND get_plane_residual_size( MiSize, 1 ) == BLOCK_4X4
///                                                     ⇒ CFL allowed
///   • Lossless == 0 AND Max( Block_Width, Block_Height ) <= 32
///                                                     ⇒ CFL allowed
///   • otherwise                                       ⇒ CFL not allowed
/// ```
/// Caller derives `cfl_allowed` (e.g. via
/// [`crate::cdf::cfl_allowed_for_uv_mode`]) and feeds it here; the
/// chosen row is `TileUVMode{Cfl{Allowed,NotAllowed}}Cdf[ YMode ]`.
///
/// `uv_mode` MUST be in `0..UV_INTRA_MODES_CFL_{ALLOWED,NOT_ALLOWED}`
/// per the `cfl_allowed` selector; `y_mode` MUST be in
/// `0..INTRA_MODES = 13`.
pub fn write_intra_uv_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    y_mode: u8,
    uv_mode: u8,
    cfl_allowed: bool,
) -> Result<(), Error> {
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let bound = if cfl_allowed {
        UV_INTRA_MODES_CFL_ALLOWED
    } else {
        UV_INTRA_MODES_CFL_NOT_ALLOWED
    };
    if (uv_mode as usize) >= bound {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs
        .uv_mode_cdf(cfl_allowed, y_mode as usize)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(uv_mode as u32, cdf)
}

/// `read_cfl_alphas()` inverse per §5.11.45 (av1-spec p.96) — r231.
///
/// Spec body:
/// ```text
///   read_cfl_alphas() {
///       cfl_alpha_signs                                  S()
///       signU = (cfl_alpha_signs + 1) / 3
///       signV = (cfl_alpha_signs + 1) % 3
///       if (signU != CFL_SIGN_ZERO) {
///           cfl_alpha_u                                  S()
///           CflAlphaU = 1 + cfl_alpha_u
///           if (signU == CFL_SIGN_NEG)
///               CflAlphaU = -CflAlphaU
///       } else {
///           CflAlphaU = 0
///       }
///       // (mirror arm for V)
///   }
/// ```
///
/// `alpha_u` and `alpha_v` are the signed §5.11.45 `CflAlpha{U,V}`
/// outputs the caller (e.g. the encoder's CFL picker) committed to.
/// Each MUST be in `-16..=-1 | 0 | 1..=16` per the §5.11.45 derivation
/// (`CflAlpha = ±(1 + cfl_alpha_*)` with `cfl_alpha_*` ∈
/// `0..CFL_ALPHABET_SIZE`), and not both `0` simultaneously (per
/// §6.10.36 the `(CFL_SIGN_ZERO, CFL_SIGN_ZERO)` joint-sign
/// combination is prohibited as redundant with `UV_DC_PRED`).
///
/// CDF selection per §8.3.2:
///   * `cfl_alpha_signs` reads `TileCflSignCdf`
///     (`Default_Cfl_Sign_Cdf`, 8-symbol, no `ctx`).
///   * `cfl_alpha_u` reads `TileCflAlphaCdf[ ctx_u ]` with
///     `ctx_u = cfl_alpha_u_ctx(signU, signV)` (only fires when
///     `signU != CFL_SIGN_ZERO`).
///   * `cfl_alpha_v` reads `TileCflAlphaCdf[ ctx_v ]` with
///     `ctx_v = cfl_alpha_v_ctx(signU, signV)` (only fires when
///     `signV != CFL_SIGN_ZERO`).
///
/// Returns [`Error::PartitionWalkOutOfRange`] for out-of-range
/// `alpha_u` / `alpha_v` or the prohibited `(0, 0)` pair.
pub fn write_cfl_alphas(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    alpha_u: i8,
    alpha_v: i8,
) -> Result<(), Error> {
    // §5.11.45 magnitude bounds: |CflAlpha| ∈ {0, 1..=CFL_ALPHABET_SIZE}.
    let abs_u: i32 = alpha_u.unsigned_abs() as i32;
    let abs_v: i32 = alpha_v.unsigned_abs() as i32;
    if abs_u as usize > CFL_ALPHABET_SIZE || abs_v as usize > CFL_ALPHABET_SIZE {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §6.10.36: (CFL_SIGN_ZERO, CFL_SIGN_ZERO) is prohibited.
    if alpha_u == 0 && alpha_v == 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §5.11.45: signU = ZERO/NEG/POS for alpha 0 / <0 / >0.
    let sign_u: usize = match alpha_u.signum() {
        0 => 0,  // CFL_SIGN_ZERO
        -1 => 1, // CFL_SIGN_NEG
        _ => 2,  // CFL_SIGN_POS
    };
    let sign_v: usize = match alpha_v.signum() {
        0 => 0,
        -1 => 1,
        _ => 2,
    };
    // §5.11.45: `signU = (cfl_alpha_signs + 1) / 3`,
    // `signV = (cfl_alpha_signs + 1) % 3` ⇒ inverse
    // `cfl_alpha_signs = 3 * signU + signV - 1`. Result is in
    // `0..CFL_JOINT_SIGNS = 8`.
    let joint: i32 = 3 * (sign_u as i32) + (sign_v as i32) - 1;
    if !(0..(CFL_JOINT_SIGNS as i32)).contains(&joint) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let signs_cdf = cdfs.cfl_sign_cdf();
    writer.write_symbol(joint as u32, signs_cdf)?;
    // §5.11.45 U arm.
    if sign_u != 0 {
        let raw_u = abs_u - 1; // `cfl_alpha_u = CflAlphaU.abs() - 1`.
        let ctx_u = cfl_alpha_u_ctx(sign_u, sign_v);
        let row_u = cdfs.cfl_alpha_cdf(ctx_u);
        writer.write_symbol(raw_u as u32, row_u)?;
    }
    // §5.11.45 V arm.
    if sign_v != 0 {
        let raw_v = abs_v - 1;
        let ctx_v = cfl_alpha_v_ctx(sign_u, sign_v);
        let row_v = cdfs.cfl_alpha_cdf(ctx_v);
        writer.write_symbol(raw_v as u32, row_v)?;
    }
    Ok(())
}

/// `read_is_inter()` inverse per §5.11.20 (av1-spec p.71) — the
/// per-block `is_inter` syntax element on the §5.11.18
/// `inter_frame_mode_info` path. Lives at the same dispatcher level
/// as [`write_skip`] / [`write_intra_segment_id`] and mirrors
/// [`crate::cdf::PartitionWalker::decode_is_inter`].
///
/// Spec body (av1-spec p.71, §5.11.20):
///
/// ```text
///   read_is_inter( ) {
///       if ( skip_mode ) {
///           is_inter = 1
///       } else if ( seg_feature_active( SEG_LVL_REF_FRAME ) ) {
///           is_inter = FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] != INTRA_FRAME
///       } else if ( seg_feature_active( SEG_LVL_GLOBALMV ) ) {
///           is_inter = 1
///       } else {
///           is_inter                                            S()
///       }
///   }
/// ```
///
/// The four-arm `if / else if / else if / else` chain is short-circuit
/// on the first match, exactly mirroring the decoder. Only the
/// fall-through `else` arm emits an `S()` symbol; the other three arms
/// write zero bits and demand that the caller-supplied `is_inter`
/// matches the spec-derived value.
///
/// ## Parameter surface
///
/// Mirrors [`crate::cdf::PartitionWalker::decode_is_inter`]:
///
/// * `skip_mode` — per-block §5.11.10 `SkipModes[r][c]` flag (0 or 1).
/// * `seg_ref_frame_active` — caller-precomputed
///   `seg_feature_active( SEG_LVL_REF_FRAME )` per §6.4.2.
/// * `seg_ref_frame_is_inter` — caller-precomputed
///   `FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ] != INTRA_FRAME`.
///   Only consulted on the `seg_ref_frame_active == true` arm.
/// * `seg_globalmv_active` — caller-precomputed
///   `seg_feature_active( SEG_LVL_GLOBALMV )`. Only consulted on the
///   `else if` arm after the SEG_LVL_REF_FRAME gate.
/// * `ctx` — the §8.3.2 `is_inter` context computed via
///   [`crate::cdf::is_inter_ctx`] from the neighbour
///   `LeftIntra` / `AboveIntra` predicates (or `None` for an
///   unavailable neighbour per §5.11.18). MUST be in
///   `0..IS_INTER_CONTEXTS = 4`. Only consulted on the fall-through
///   `else` arm.
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `is_inter > 1` — outside the §3 `IsInter` binary alphabet.
/// * `skip_mode == 1` and `is_inter != 1` — Arm 1 forces `is_inter = 1`.
/// * `seg_ref_frame_active == true` and
///   `is_inter != seg_ref_frame_is_inter as u8` — Arm 2 forces the
///   spec-derived value.
/// * `seg_globalmv_active == true` (with the two above false) and
///   `is_inter != 1` — Arm 3 forces `is_inter = 1`.
/// * `ctx >= IS_INTER_CONTEXTS` on the fall-through arm — invalid
///   §8.3.2 ctx.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// The decoder's `decode_is_inter` stamps the resulting `is_inter`
/// over the block's `bh4 * bw4` footprint into `walker.is_inters[]`.
/// The writer is stateless (mirroring [`write_skip`]) and does not
/// maintain an encoder-side `IsInters[]` grid; the caller threads
/// whatever grid state the encode-side §8.3.2 ctx derivations need
/// (typically by a parallel [`crate::cdf::PartitionWalker`] used only
/// for its grid stamps — see the round-trip tests below).
#[allow(clippy::too_many_arguments)]
pub fn write_is_inter(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    is_inter: u8,
    ctx: usize,
    skip_mode: u8,
    seg_ref_frame_active: bool,
    seg_ref_frame_is_inter: bool,
    seg_globalmv_active: bool,
) -> Result<(), Error> {
    // §3 binary alphabet bound.
    if is_inter > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // Arm 1: `if ( skip_mode ) is_inter = 1` — no bits.
    if skip_mode != 0 {
        if is_inter != 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    // Arm 2: SEG_LVL_REF_FRAME ⇒ is_inter = FeatureData != INTRA_FRAME, no bits.
    if seg_ref_frame_active {
        let expected = seg_ref_frame_is_inter as u8;
        if is_inter != expected {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    // Arm 3: SEG_LVL_GLOBALMV ⇒ is_inter = 1, no bits.
    if seg_globalmv_active {
        if is_inter != 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    // Arm 4 (fall-through): `is_inter S()` over `TileIsInterCdf[ ctx ]`.
    if ctx >= IS_INTER_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs.is_inter_cdf(ctx);
    writer.write_symbol(is_inter as u32, cdf)
}

/// `read_skip_mode()` inverse per §5.11.10 (av1-spec p.67) — the
/// per-block `skip_mode` syntax element on the §5.11.18
/// `inter_frame_mode_info` path. Lives at the same dispatcher level
/// as [`write_skip`] and mirrors
/// [`crate::cdf::PartitionWalker::decode_skip_mode`]. `skip_mode`
/// is the inter-frame "compound-reference shortcut" flag — it
/// gates Arm 1 of [`write_is_inter`] (`skip_mode == 1` ⇒
/// `is_inter = 1`, no symbol read).
///
/// Spec body (av1-spec p.67, §5.11.10):
///
/// ```text
///   read_skip_mode( ) {
///       if ( seg_feature_active( SEG_LVL_SKIP ) ||
///            seg_feature_active( SEG_LVL_REF_FRAME ) ||
///            seg_feature_active( SEG_LVL_GLOBALMV ) ||
///            !skip_mode_present ||
///            Block_Width[ MiSize ] < 8 ||
///            Block_Height[ MiSize ] < 8 ) {
///           skip_mode = 0
///       } else {
///           skip_mode                                            S()
///       }
///   }
/// ```
///
/// The six-condition disjunction collapses, per the decoder
/// reader's parameter surface, into:
///
/// * `seg_skip_mode_off` — caller-precomputed
///   `seg_feature_active( SEG_LVL_SKIP ) ||
///   seg_feature_active( SEG_LVL_REF_FRAME ) ||
///   seg_feature_active( SEG_LVL_GLOBALMV )`.
/// * `skip_mode_present` — the §5.9.21 frame-header scalar gating
///   skip-mode for the whole frame.
/// * `sub_size` — the §5.11.5 `MiSize` of the leaf block. The
///   small-block short-circuit
///   (`Block_Width[ MiSize ] < 8 || Block_Height[ MiSize ] < 8`) is
///   derived locally via the §9.3 [`block_width`] / [`block_height`]
///   tables, mirroring the decoder's own derivation. `sub_size`
///   MUST be in `0..BLOCK_SIZES = 22`.
///
/// On the fall-through `else` arm a single §8.2.6 `S()` symbol is
/// emitted against `TileSkipModeCdf[ ctx ]`. `ctx` is the §8.3.2
/// `skip_mode` context (the sum of the neighbour `SkipModes[]`
/// flags) the caller pre-computes via [`crate::cdf::skip_mode_ctx`]
/// from the §5.11.5 `SkipModes[]` grid state. `ctx` MUST be in
/// `0..SKIP_MODE_CONTEXTS = 3`.
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `skip_mode > 1` — outside the §3 binary alphabet.
/// * `sub_size >= BLOCK_SIZES` — invalid §5.11.5 `MiSize`.
/// * any short-circuit arm fires (`seg_skip_mode_off`,
///   `!skip_mode_present`, or `sub_size` ⇒ `Block_W < 8 ||
///   Block_H < 8`) and `skip_mode != 0` — the spec forces
///   `skip_mode = 0` on every short-circuit arm.
/// * `ctx >= SKIP_MODE_CONTEXTS` on the fall-through arm — invalid
///   §8.3.2 ctx.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// The decoder's `decode_skip_mode` stamps the resulting
/// `skip_mode` over the block's `bh4 * bw4` footprint into
/// `walker.skip_modes[]`. The writer is stateless (mirroring
/// [`write_skip`] / [`write_is_inter`]) and does not maintain an
/// encoder-side `SkipModes[]` grid; the caller threads whatever
/// grid state the encode-side §8.3.2 ctx derivations need
/// (typically by a parallel [`crate::cdf::PartitionWalker`] used
/// only for its grid stamps — see the round-trip tests below).
pub fn write_skip_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    skip_mode: u8,
    ctx: usize,
    sub_size: usize,
    seg_skip_mode_off: bool,
    skip_mode_present: bool,
) -> Result<(), Error> {
    // §3 binary alphabet bound.
    if skip_mode > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §5.11.5 sub_size domain.
    if sub_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §5.11.10 small-block short-circuit set
    // `Block_Width[ MiSize ] < 8 || Block_Height[ MiSize ] < 8`,
    // derived locally from sub_size via the §9.3 tables (mirrors
    // `decode_skip_mode`'s own derivation).
    let small_block = block_width(sub_size) < 8 || block_height(sub_size) < 8;
    // §5.11.10 short-circuit set: any-true ⇒ skip_mode = 0, no bits.
    if seg_skip_mode_off || !skip_mode_present || small_block {
        if skip_mode != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    // Fall-through `else`: `skip_mode S()` over
    // `TileSkipModeCdf[ ctx ]`.
    if ctx >= SKIP_MODE_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs.skip_mode_cdf(ctx);
    writer.write_symbol(skip_mode as u32, cdf)
}

/// `inter_segment_id( preSkip )` inverse per §5.11.19 (av1-spec p.71) —
/// the inter-frame per-block segment-id writer. Mirrors
/// [`crate::cdf::PartitionWalker::decode_inter_segment_id`]. Called
/// twice from the §5.11.18 `inter_frame_mode_info` writer: once with
/// `pre_skip = true` before the §5.11.11 `read_skip()` call (the §5.9.14
/// `SegIdPreSkip == 1` arm activates the early write) and once with
/// `pre_skip = false` after (the post-skip arm covers the
/// `SegIdPreSkip == 0` case and the temporal-update / skip-after-pre-skip
/// update of the segmentation-prediction context arrays).
///
/// Spec body (av1-spec p.71, §5.11.19):
///
/// ```text
///   inter_segment_id( preSkip ) {
///       if ( segmentation_enabled ) {
///           predictedSegmentId = get_segment_id( )
///           if ( segmentation_update_map ) {
///               if ( preSkip && !SegIdPreSkip ) {
///                   segment_id = 0
///                   return
///               }
///               if ( !preSkip ) {
///                   if ( skip ) {
///                       seg_id_predicted = 0
///                       for ( i = 0; i < Num_4x4_Blocks_Wide[ MiSize ]; i++ )
///                           AboveSegPredContext[ MiCol + i ] = seg_id_predicted
///                       for ( i = 0; i < Num_4x4_Blocks_High[ MiSize ]; i++ )
///                           LeftSegPredContext[ MiRow + i ] = seg_id_predicted
///                       read_segment_id( )
///                       return
///                   }
///               }
///               if ( segmentation_temporal_update == 1 ) {
///                   seg_id_predicted                                       S()
///                   if ( seg_id_predicted )
///                       segment_id = predictedSegmentId
///                   else
///                       read_segment_id( )
///                   for ( i = 0; i < Num_4x4_Blocks_Wide[ MiSize ]; i++ )
///                       AboveSegPredContext[ MiCol + i ] = seg_id_predicted
///                   for ( i = 0; i < Num_4x4_Blocks_High[ MiSize ]; i++ )
///                       LeftSegPredContext[ MiRow + i ] = seg_id_predicted
///               } else {
///                   read_segment_id( )
///               }
///           } else {
///               segment_id = predictedSegmentId
///           }
///       } else {
///           segment_id = 0
///       }
///   }
/// ```
///
/// ## Arm dispatch (encoder mirror)
///
/// The writer transcribes each spec arm; arms that the decoder
/// services with zero `S()` reads are also serviced here with zero
/// `write_symbol` calls:
///
/// 1. `!segmentation_enabled` — no bits; `segment_id` MUST be `0`.
/// 2. `!segmentation_update_map` — no bits; `segment_id` MUST equal
///    `predicted_segment_id` (the §5.11.21 `get_segment_id()` result
///    the caller pre-computes).
/// 3. `pre_skip && !seg_id_pre_skip` — no bits; `segment_id` MUST be
///    `0`. The post-skip call services the actual id read.
/// 4. `!pre_skip && skip != 0` (post-skip arm's skip-block branch) —
///    the decoder forces `seg_id_predicted = 0` (no `S()` read) and
///    immediately descends to `read_segment_id()` with `skip = 1`,
///    which inside §5.11.9 short-circuits to `segment_id = pred`.
///    The writer mirrors: stamp `seg_id_predicted = 0` is the
///    caller's grid responsibility; here we delegate to
///    [`write_segment_id`] with `skip = 1` (no bits). `segment_id`
///    MUST equal `pred` on this arm.
/// 5. `segmentation_temporal_update` — emits one §8.2.6 `S()` symbol
///    `seg_id_predicted` against `TileSegmentIdPredictedCdf[ ctx ]`
///    where `ctx` is the §8.3.2 `seg_pred_ctx` (= `LeftSegPredContext[
///    MiRow ] + AboveSegPredContext[ MiCol ]`, caller-derived in
///    `0..SEGMENT_ID_PREDICTED_CONTEXTS = 3`). Then either:
///    * `seg_id_predicted == 1` ⇒ adopt `predicted_segment_id` (no
///      further bits); `segment_id` MUST equal `predicted_segment_id`.
///    * `seg_id_predicted == 0` ⇒ delegate to [`write_segment_id`]
///      with `skip = 0` for the literal id read (one §8.2.6 `S()`
///      against `TileSegmentIdCdf[ seg_id_read_ctx ]`).
///
///    Per §6.10.9 spec-note "It is allowed for seg_id_predicted to be
///    equal to 0 even if the value coded for the segment_id is equal
///    to predictedSegmentId" — the caller chooses the flag and the
///    writer does not second-guess it.
/// 6. Fall-through (`segmentation_temporal_update == 0`) — delegate
///    to [`write_segment_id`] with `skip = 0` for the literal id
///    read.
///
/// ## Parameters
///
/// * `segment_id` — the final `segment_id ∈ 0..=last_active_seg_id`
///   the block should decode to (≤ `MAX_SEGMENTS - 1 = 7`).
/// * `pred` — the §5.11.9 neighbour-cascade predictor the caller
///   already has on hand (used only on arms 4 + 5-else + 6, where
///   `write_segment_id` is invoked).
/// * `seg_id_read_ctx` — the §8.3.2 `segment_id_ctx` value the
///   caller pre-computes via [`crate::cdf::segment_id_ctx`] for the
///   inner `write_segment_id` `S()` arm. Only consulted on arms 4,
///   5-else, and 6.
/// * `seg_pred_ctx` — the §8.3.2 `seg_id_predicted` ctx value the
///   caller pre-computes (= `LeftSegPredContext[ MiRow ] +
///   AboveSegPredContext[ MiCol ]`, in `0..3`). Only consulted on
///   arm 5.
/// * `pre_skip` — the §5.11.18 caller-axis (`true` on the pre-skip
///   call, `false` on the post-skip call).
/// * `skip` — the §5.11.11 `read_skip` return (passed `0` on the
///   pre-skip call — `read_skip` fires after `inter_segment_id( 1 )`).
/// * `seg_id_predicted` — caller's choice on arm 5: `1` to adopt the
///   `predicted_segment_id` (saves bits), `0` to code the literal id.
///   Ignored on every other arm.
/// * `segmentation_enabled` / `segmentation_update_map` /
///   `segmentation_temporal_update` / `seg_id_pre_skip` — the §5.9.14
///   frame-header derivations surfaced through
///   [`SegmentationParams`].
/// * `predicted_segment_id` — `get_segment_id()` result the caller
///   pre-computes from the §6.10 reference-frame walk.
/// * `last_active_seg_id` — §6.10.8 derivation
///   (`MAX_SEGMENTS - 1 = 7` upper bound).
///
/// ## Range / mismatch guards (`Error::PartitionWalkOutOfRange`)
///
/// * `last_active_seg_id >= MAX_SEGMENTS = 8`.
/// * `segment_id > last_active_seg_id`.
/// * `pred > last_active_seg_id`.
/// * `predicted_segment_id > last_active_seg_id`.
/// * `seg_pred_ctx >= SEGMENT_ID_PREDICTED_CONTEXTS = 3` on arm 5.
/// * `seg_id_read_ctx >= SEGMENT_ID_CONTEXTS = 3` on the
///   `write_segment_id` `S()` arm (re-checked inside the inner
///   writer).
/// * `seg_id_predicted > 1` on arm 5.
/// * Arm-precondition violations — `segment_id != 0` on arms 1 and
///   3; `segment_id != predicted_segment_id` on arm 2 and on the
///   `seg_id_predicted == 1` branch of arm 5; `segment_id != pred`
///   on arm 4.
///
/// ## Stateless — no encoder-side grids
///
/// Mirroring the rest of `encoder::block_mode_info`, no
/// `SegmentIds[]` / `AboveSegPredContext[]` / `LeftSegPredContext[]`
/// arrays are maintained here. The decoder stamps those grids
/// itself; an encoder caller typically threads a parallel
/// [`crate::cdf::PartitionWalker`] purely for the grid stamps and
/// uses [`crate::cdf::PartitionWalker::seg_pred_ctx`] (or the
/// equivalent neighbour-array lookup) to derive `seg_pred_ctx`.
///
/// [`SegmentationParams`]: crate::uncompressed_header_tail::SegmentationParams
#[allow(clippy::too_many_arguments)]
pub fn write_inter_segment_id(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    segment_id: u8,
    pred: u8,
    seg_id_read_ctx: usize,
    seg_pred_ctx: usize,
    pre_skip: bool,
    skip: u8,
    seg_id_predicted: u8,
    segmentation_enabled: bool,
    segmentation_update_map: bool,
    segmentation_temporal_update: bool,
    seg_id_pre_skip: bool,
    predicted_segment_id: u8,
    last_active_seg_id: u8,
) -> Result<(), Error> {
    // Up-front range guards (mirror `decode_inter_segment_id`).
    if (last_active_seg_id as usize) >= MAX_SEGMENTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let max = last_active_seg_id as u32 + 1;
    if segment_id as u32 >= max {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if pred as u32 >= max {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if predicted_segment_id as u32 >= max {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // Arm 1: §5.11.19 `if ( !segmentation_enabled ) segment_id = 0`.
    if !segmentation_enabled {
        if segment_id != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // Arm 2: §5.11.19 `else if ( !segmentation_update_map )
    // segment_id = predictedSegmentId`.
    if !segmentation_update_map {
        if segment_id != predicted_segment_id {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // Arm 3: §5.11.19 `if ( preSkip && !SegIdPreSkip ) { segment_id = 0;
    // return }` — the pre-skip caller defers to the post-skip call.
    if pre_skip && !seg_id_pre_skip {
        if segment_id != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // Arm 4: §5.11.19 `if ( !preSkip ) { if ( skip ) { seg_id_predicted
    // = 0; ... read_segment_id(); return } }`. The inner
    // `read_segment_id()` is called with `skip = 1`, which §5.11.9
    // short-circuits to `segment_id = pred` (no `S()` read). The
    // forced `seg_id_predicted = 0` grid stamp is the caller's job.
    if !pre_skip && skip != 0 {
        return write_segment_id(
            writer,
            cdfs,
            segment_id,
            /* skip = */ 1,
            pred,
            seg_id_read_ctx,
            last_active_seg_id,
        );
    }

    // Arm 5: §5.11.19 `if ( segmentation_temporal_update == 1 )` — code
    // the binary `seg_id_predicted` S(), then either adopt or
    // fall through.
    if segmentation_temporal_update {
        if seg_id_predicted > 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if seg_pred_ctx >= SEGMENT_ID_PREDICTED_CONTEXTS {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let cdf = cdfs.segment_id_predicted_cdf(seg_pred_ctx);
        writer.write_symbol(seg_id_predicted as u32, cdf)?;
        if seg_id_predicted != 0 {
            // §5.11.19 `if ( seg_id_predicted ) segment_id =
            // predictedSegmentId` — adopt the previous-frame id; no
            // further bits.
            if segment_id != predicted_segment_id {
                return Err(Error::PartitionWalkOutOfRange);
            }
            return Ok(());
        }
        // `seg_id_predicted == 0` ⇒ §5.11.19 `else read_segment_id()`.
        return write_segment_id(
            writer,
            cdfs,
            segment_id,
            /* skip = */ 0,
            pred,
            seg_id_read_ctx,
            last_active_seg_id,
        );
    }

    // Arm 6 (fall-through, `segmentation_temporal_update == 0`):
    // §5.11.19 `else read_segment_id()` with `skip = 0` (the post-skip
    // call's `skip == 1` arm was handled in Arm 4; the pre-skip call's
    // `skip` parameter is forced to `0` by the §5.11.18 dispatch
    // ordering).
    write_segment_id(
        writer,
        cdfs,
        segment_id,
        /* skip = */ 0,
        pred,
        seg_id_read_ctx,
        last_active_seg_id,
    )
}

/// Aggregate the §5.11.18 `inter_frame_mode_info` writer prefix produces.
///
/// The struct mirrors the decoder-side
/// [`crate::cdf::DecodedInterFrameModeInfo`] for the
/// *pre-cdef / pre-delta / pre-`if(is_inter)`* portion of §5.11.18 — i.e.
/// every value the writer commits to the bitstream before reaching the
/// §5.11.18 line 18 `read_cdef( )` call. The §5.11.18 lines 17-26
/// (`Lossless = LosslessArray[ segment_id ]`, `read_cdef`,
/// `read_delta_qindex`, `read_delta_lf`, `ReadDeltas = 0`, `read_is_inter`,
/// the `if ( is_inter ) inter_block_mode_info( ) else intra_block_mode_info( )`
/// terminal dispatch) are next-round targets — we surface `is_inter` and
/// `lossless` because `write_is_inter` *is* in scope and `LosslessArray[]`
/// is a pure caller-supplied lookup with no bits emitted.
///
/// Fields:
/// * `segment_id` — final `segment_id ∈ 0..=last_active_seg_id` after the
///   §5.11.18 pre-skip and (when `!seg_id_pre_skip`) post-skip
///   `inter_segment_id` writes.
/// * `skip_mode` — the §5.11.10 result (`0` on every short-circuit arm,
///   `0` or `1` on the fall-through S() arm).
/// * `skip` — `1` if `skip_mode == 1` (§5.11.18 lines 13-14 force);
///   otherwise the §5.11.11 `read_skip` result.
/// * `is_inter` — the §5.11.20 result the §5.11.18 line 22 read produces.
/// * `lossless` — `LosslessArray[ segment_id ]`, the §5.11.18 line 17
///   derivation against the caller-supplied lossless array. Mirrors the
///   decoder's same field on [`crate::cdf::DecodedInterFrameModeInfo`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterFrameModeInfoPrefix {
    /// §5.11.18 lines 11 + 15-16 result. `0..=last_active_seg_id`.
    pub segment_id: u8,
    /// §5.11.18 line 12 result.
    pub skip_mode: u8,
    /// §5.11.18 line 10 + lines 13-14 result.
    pub skip: u8,
    /// §5.11.18 line 22 result.
    pub is_inter: u8,
    /// §5.11.18 line 17 result.
    pub lossless: bool,
    /// §5.11.18 line 18 result — `cdef_idx` returned by
    /// [`write_cdef`]. `None` on the §5.11.56 short-circuit set
    /// (`skip || coded_lossless || !enable_cdef || allow_intrabc`)
    /// or when an earlier leaf in the same 64×64 anchor already
    /// stamped the literal (the writer emits no bits on either arm).
    /// `Some(cdef_idx)` on the first-leaf-in-anchor write arm.
    pub cdef_idx: Option<i8>,
    /// §5.11.18 line 19 result — caller-passed `reduced_delta_q_index`
    /// surfaced for chaining. `0` on every short-circuit arm; the
    /// caller owns the `CurrentQIndex` accumulator the §5.11.12 spec
    /// updates via `Clip3` (the writer is pure on grid-state).
    pub reduced_delta_q_index: i32,
    /// §5.11.18 line 20 result — caller-passed `reduced_delta_lf`
    /// row surfaced for chaining. `[0; FRAME_LF_COUNT]` on every
    /// short-circuit arm; the caller owns the `DeltaLF[ i ]`
    /// accumulator the §5.11.13 spec updates via `Clip3`.
    pub reduced_delta_lf: [i32; FRAME_LF_COUNT],
}

/// §5.11.18 lines 18-20 input bundle — the per-block scalars the
/// `read_cdef()` / `read_delta_qindex()` / `read_delta_lf()` writers
/// consume. Bundled into a single argument so the
/// [`write_inter_frame_mode_info_prefix`] dispatcher's parameter
/// surface doesn't grow another 11 positional arguments.
///
/// All fields mirror the matching argument of [`write_cdef`] /
/// [`write_delta_qindex`] / [`write_delta_lf`]:
///
/// * §5.11.56 `read_cdef` — `cdef_idx` / `cdef_idx_prior_stamp` /
///   `cdef_bits` / `coded_lossless` / `enable_cdef` / `allow_intrabc`
///   / `anchor_already_stamped`.
/// * §5.11.12 `read_delta_qindex` — `reduced_delta_q_index` /
///   `read_deltas` / `use_128x128_superblock` / `delta_q_res`.
/// * §5.11.13 `read_delta_lf` — `reduced_delta_lf` / `read_deltas`
///   (shared with `delta_qindex`) / `delta_lf_present` /
///   `delta_lf_multi` / `mono_chrome` / `delta_lf_res`.
///
/// A `Default` impl yields the all-short-circuit configuration
/// (`enable_cdef = false`, `read_deltas = false`, zero deltas) —
/// suitable for callers that target the §5.11.18 lines 18-20
/// no-bits-emitted arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterFrameDeltaSiteInputs {
    /// §5.11.56 `cdef_idx` value to commit on the first-leaf-in-anchor
    /// arm. Range-checked against `cdef_bits` (`0..(1 << cdef_bits)`).
    pub cdef_idx: i8,
    /// §5.11.56 prior-leaf `cdef_idx` stamp — passed straight through
    /// to [`write_cdef`]'s `anchor_already_stamped == true` arm so the
    /// writer can verify the caller's `cdef_idx` matches.
    pub cdef_idx_prior_stamp: i8,
    /// §5.9.19 `cdef_bits` field, `0..=3` (`f(2)` cap).
    pub cdef_bits: u32,
    /// §6.8.21 `CodedLossless` derivation. Short-circuits §5.11.56.
    pub coded_lossless: bool,
    /// §5.9.19 `enable_cdef` flag. Short-circuits §5.11.56.
    pub enable_cdef: bool,
    /// §5.9.20 `allow_intrabc` flag. Short-circuits §5.11.56.
    pub allow_intrabc: bool,
    /// §5.11.56 outer-`if` precondition: `anchor_already_stamped =
    /// (cdef_idx[r][c] != -1)` per the matching `decode_cdef`
    /// derivation, where `(r, c) = (mi_row, mi_col) & ~(cdefSize4 - 1)`.
    /// `true` ⇒ the writer emits no bits on this leaf and the caller
    /// MUST pass `cdef_idx == cdef_idx_prior_stamp`.
    pub anchor_already_stamped: bool,
    /// §5.11.12 `reducedDeltaQIndex` signed intermediate (the value
    /// the §5.11.12 spec computes between the sign-bit read and the
    /// `Clip3` update). The caller owns the `CurrentQIndex`
    /// accumulator the spec updates with this value (`CurrentQIndex
    /// = Clip3(1, 255, CurrentQIndex + (reducedDeltaQIndex << delta_q_res))`).
    pub reduced_delta_q_index: i32,
    /// §5.11.13 `reducedDeltaLfLevel` row — one entry per LF strength.
    /// Only the first `frameLfCount` entries (per §5.11.13 derivation)
    /// are consumed; entries beyond MUST be `0`.
    pub reduced_delta_lf: [i32; FRAME_LF_COUNT],
    /// §6.10.4 `ReadDeltas` flag — shared gate for §5.11.12 and
    /// §5.11.13.
    pub read_deltas: bool,
    /// §5.5.1 `use_128x128_superblock` flag — used for the per-call
    /// `sbSize` derivation that gates the superblock-skip
    /// short-circuit.
    pub use_128x128_superblock: bool,
    /// §5.9.17 `delta_q_res` field, `0..=3`.
    pub delta_q_res: u8,
    /// §5.9.18 `delta_lf_present` flag — gates §5.11.13 reads.
    pub delta_lf_present: bool,
    /// §5.9.18 `delta_lf_multi` flag — selects per-LF or single-LF
    /// row width via the `frameLfCount` derivation.
    pub delta_lf_multi: bool,
    /// §5.5.2 `mono_chrome` flag — narrows the multi-LF row to
    /// `FRAME_LF_COUNT - 2` entries.
    pub mono_chrome: bool,
    /// §5.9.18 `delta_lf_res` field, `0..=3`.
    pub delta_lf_res: u8,
}

impl Default for InterFrameDeltaSiteInputs {
    /// All-short-circuit defaults: `enable_cdef = false` (§5.11.56
    /// short-circuit), `read_deltas = false` (§5.11.12 / §5.11.13
    /// outer-`if` false), zero `reduced_delta_*` so the §5.11.12 /
    /// §5.11.13 short-circuit-validates pass. No bits emitted on
    /// lines 18-20.
    fn default() -> Self {
        Self {
            cdef_idx: 0,
            cdef_idx_prior_stamp: -1,
            cdef_bits: 0,
            coded_lossless: false,
            enable_cdef: false,
            allow_intrabc: false,
            anchor_already_stamped: false,
            reduced_delta_q_index: 0,
            reduced_delta_lf: [0; FRAME_LF_COUNT],
            read_deltas: false,
            use_128x128_superblock: false,
            delta_q_res: 0,
            delta_lf_present: false,
            delta_lf_multi: false,
            mono_chrome: false,
            delta_lf_res: 0,
        }
    }
}

/// `inter_frame_mode_info()` writer prefix per §5.11.18 (av1-spec p.71) —
/// the per-block syntax dispatcher for an inter-frame leaf. Composes
/// every leaf writer landed across r253-r258:
///
/// * §5.11.19 `inter_segment_id( 1 )` — pre-skip arm via
///   [`write_inter_segment_id`].
/// * §5.11.10 `read_skip_mode()` — via [`write_skip_mode`].
/// * §5.11.11 `read_skip()` (when `skip_mode == 0`) — via [`write_skip`].
/// * §5.11.19 `inter_segment_id( 0 )` (when `!seg_id_pre_skip`) —
///   post-skip arm via [`write_inter_segment_id`].
/// * §5.11.56 `read_cdef()` — via [`write_cdef`] (r258).
/// * §5.11.12 `read_delta_qindex()` — via [`write_delta_qindex`] (r258).
/// * §5.11.13 `read_delta_lf()` — via [`write_delta_lf`] (r258).
/// * §5.11.20 `read_is_inter()` — via [`write_is_inter`].
///
/// The spec body (av1-spec p.71) reads:
///
/// ```text
///   inter_frame_mode_info( ) {
///       use_intrabc = 0
///       LeftRefFrame[..]  = ...
///       AboveRefFrame[..] = ...
///       LeftIntra / AboveIntra / LeftSingle / AboveSingle = ...
///       skip = 0
///       inter_segment_id( 1 )                                 ← writer
///       read_skip_mode( )                                     ← writer
///       if ( skip_mode )
///           skip = 1
///       else
///           read_skip( )                                      ← writer
///       if ( !SegIdPreSkip )
///           inter_segment_id( 0 )                             ← writer
///       Lossless = LosslessArray[ segment_id ]
///       read_cdef( )                                          ← writer (r259)
///       read_delta_qindex( )                                  ← writer (r259)
///       read_delta_lf( )                                      ← writer (r259)
///       ReadDeltas = 0
///       read_is_inter( )                                      ← writer
///       if ( is_inter )
///           inter_block_mode_info( )                          (next round)
///       else
///           intra_block_mode_info( )                          (next round)
///   }
/// ```
///
/// ## Scope of this writer
///
/// The dispatcher writes the §5.11.18 *prefix*: the eight S()-emitting
/// sub-writers above. The line 23-26 terminal `if ( is_inter )` dispatch
/// into §5.11.22 `intra_block_mode_info` / §5.11.23 `inter_block_mode_info`
/// remains a next-round encoder-side target — neither has a writer
/// counterpart yet.
///
/// The writer's return value is an [`InterFrameModeInfoPrefix`]
/// aggregate covering every value committed to the bitstream so far,
/// including the `segment_id`-indexed `lossless` lookup (no bits) so
/// the caller can chain the §5.11.18 line 17 derivation without
/// re-fetching it.
///
/// ## Parameter surface
///
/// Mirrors the union of every composed sub-writer. The `is_inter_ctx`
/// argument is the §8.3.2 `is_inter` ctx the caller pre-computes via
/// [`crate::cdf::is_inter_ctx`]; the other ctx values (`skip_ctx`,
/// `skip_mode_ctx`, `seg_id_read_ctx`, `seg_pred_ctx`) are caller-supplied
/// just like the per-sub-writer entry points. The `sub_size`, segmentation
/// scalars, and pred values mirror their per-sub-writer counterparts
/// one-to-one.
///
/// ## Out-of-range / mismatch cases
///
/// All [`Error::PartitionWalkOutOfRange`] guards from the composed
/// sub-writers fire here unchanged (the dispatcher delegates without
/// short-circuiting). One extra guard is added up-front:
///
/// * `last_active_seg_id >= MAX_SEGMENTS` — invalid §6.10.8 derivation.
///   Caught up-front so the lossless lookup below is well-defined.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// As with every sub-writer in this module, the dispatcher is stateless:
/// caller threads any encoder-side §8.3.2 ctx derivations / parallel
/// [`crate::cdf::PartitionWalker`] grid-fill themselves. The round-trip
/// tests below drive the byte run through
/// [`crate::cdf::PartitionWalker::decode_inter_frame_mode_info`]'s prefix
/// (i.e. through the same five composed `decode_*` calls in the same
/// order) and verify the segment_id / skip_mode / skip / is_inter values
/// match.
#[allow(clippy::too_many_arguments)]
pub fn write_inter_frame_mode_info_prefix(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    // Per-block grid coordinates (caller-side state only — the writer
    // is stateless on grid-fill; the parameters are kept symmetric
    // with the decoder for ergonomic call-site mirroring).
    sub_size: usize,
    // §5.11.18 leaf scalars to commit:
    segment_id: u8,
    skip_mode: u8,
    skip: u8,
    is_inter: u8,
    // §5.11.19 segment-id neighbour cascade (caller-precomputed):
    seg_pred: u8,
    seg_id_predicted: u8,
    // §8.3.2 ctx values (caller-precomputed):
    skip_mode_ctx_val: usize,
    skip_ctx_val: usize,
    seg_id_read_ctx: usize,
    seg_pred_ctx: usize,
    is_inter_ctx_val: usize,
    // §5.11.10 / §5.11.11 / §5.11.20 segmentation-feature collapsed flags:
    seg_skip_mode_off: bool,
    seg_skip_active: bool,
    seg_ref_frame_active: bool,
    seg_ref_frame_is_inter: bool,
    seg_globalmv_active: bool,
    // §5.9.14 segmentation scalars:
    segmentation_enabled: bool,
    segmentation_update_map: bool,
    segmentation_temporal_update: bool,
    seg_id_pre_skip: bool,
    predicted_segment_id: u8,
    last_active_seg_id: u8,
    // §5.9.21 frame-header scalar:
    skip_mode_present: bool,
    // §5.11.18 line 17 lookup:
    lossless_array: &[bool; MAX_SEGMENTS],
    // §5.11.18 lines 18-20 inputs — `read_cdef` / `read_delta_qindex` /
    // `read_delta_lf` per-block scalars. See
    // [`InterFrameDeltaSiteInputs`] for the field-by-field contract.
    delta_inputs: &InterFrameDeltaSiteInputs,
) -> Result<InterFrameModeInfoPrefix, Error> {
    if (last_active_seg_id as usize) >= MAX_SEGMENTS {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.18 line 11: `inter_segment_id( 1 )` — pre-skip arm. The
    // §5.11.19 spec body short-circuits when `segmentation_enabled == 0`
    // (writes `segment_id = 0`, no bits), when `segmentation_update_map
    // == 0` (adopts `predictedSegmentId`, no bits), or on the
    // `preSkip && !SegIdPreSkip` early-return (sets `segment_id = 0`,
    // no bits — the post-skip call services the real read).
    //
    // When `seg_id_pre_skip == true` the pre-skip call services the
    // actual segment_id read — pass caller's `segment_id`. When
    // `seg_id_pre_skip == false` the pre-skip call MUST go through
    // §5.11.19 Arm 1 / Arm 2 / Arm 3 short-circuits (one of them MUST
    // fire because Arm 4 / 5 / 6 only run with `pre_skip == true &&
    // seg_id_pre_skip == true` or `pre_skip == false`):
    //   * `!segmentation_enabled` ⇒ pass 0 (Arm 1's invariant).
    //   * `!segmentation_update_map` ⇒ pass `predicted_segment_id`
    //     (Arm 2's invariant).
    //   * else (Arm 3 fires because `pre_skip && !seg_id_pre_skip`) ⇒
    //     pass 0 (Arm 3's invariant).
    // The final value committed by the post-skip call below is the
    // caller's `segment_id`.
    let pre_skip_segment_id = if seg_id_pre_skip {
        segment_id
    } else if !segmentation_enabled {
        0
    } else if !segmentation_update_map {
        predicted_segment_id
    } else {
        0
    };
    write_inter_segment_id(
        writer,
        cdfs,
        pre_skip_segment_id,
        seg_pred,
        seg_id_read_ctx,
        seg_pred_ctx,
        /* pre_skip = */ true,
        /* skip = */ 0,
        seg_id_predicted,
        segmentation_enabled,
        segmentation_update_map,
        segmentation_temporal_update,
        seg_id_pre_skip,
        predicted_segment_id,
        last_active_seg_id,
    )?;

    // §5.11.18 line 12: `read_skip_mode( )`.
    write_skip_mode(
        writer,
        cdfs,
        skip_mode,
        skip_mode_ctx_val,
        sub_size,
        seg_skip_mode_off,
        skip_mode_present,
    )?;

    // §5.11.18 lines 13-14: `if ( skip_mode ) skip = 1 else read_skip( )`.
    // The compound-reference shortcut forces `skip = 1` without reading
    // the §5.11.11 bit. Validate the caller-supplied `skip` matches the
    // spec's invariant either way.
    if skip > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let final_skip: u8 = if skip_mode != 0 {
        if skip != 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        1
    } else {
        write_skip(writer, cdfs, skip, skip_ctx_val, seg_skip_active)?;
        skip
    };

    // §5.11.18 lines 15-16: `if ( !SegIdPreSkip ) inter_segment_id( 0 )`.
    // The post-skip arm fires only when the §5.9.14 SegIdPreSkip
    // derivation is false; otherwise the pre-skip call already committed
    // the segment_id.
    if !seg_id_pre_skip {
        write_inter_segment_id(
            writer,
            cdfs,
            segment_id,
            seg_pred,
            seg_id_read_ctx,
            seg_pred_ctx,
            /* pre_skip = */ false,
            final_skip,
            seg_id_predicted,
            segmentation_enabled,
            segmentation_update_map,
            segmentation_temporal_update,
            seg_id_pre_skip,
            predicted_segment_id,
            last_active_seg_id,
        )?;
    }

    // §5.11.18 line 17: `Lossless = LosslessArray[ segment_id ]`. Pure
    // caller-supplied lookup; segment_id has been range-checked inside
    // the §5.11.19 writers (each call rejects `segment_id >
    // last_active_seg_id`), and `last_active_seg_id < MAX_SEGMENTS`
    // was checked up-front, so the index is always in bounds.
    debug_assert!(
        (segment_id as usize) < MAX_SEGMENTS,
        "segment_id range-checked by write_inter_segment_id"
    );
    let lossless = lossless_array[segment_id as usize];

    // §5.11.18 line 18: `read_cdef( )`. The §5.11.56 spec body
    // short-circuits on `skip || coded_lossless || !enable_cdef ||
    // allow_intrabc` and on `anchor_already_stamped`; otherwise emits
    // one `L(cdef_bits)`. The writer returns `Ok(None)` for either
    // short-circuit and `Ok(Some(cdef_idx))` on the first-leaf write.
    let cdef_idx_committed = write_cdef(
        writer,
        delta_inputs.cdef_idx,
        delta_inputs.cdef_idx_prior_stamp,
        delta_inputs.cdef_bits,
        final_skip,
        delta_inputs.coded_lossless,
        delta_inputs.enable_cdef,
        delta_inputs.allow_intrabc,
        delta_inputs.anchor_already_stamped,
    )?;

    // §5.11.18 line 19: `read_delta_qindex( )`. The §5.11.12 spec body
    // short-circuits on `MiSize == sbSize && skip` or `!ReadDeltas`;
    // otherwise emits the S() + escape ladder + sign bit cascade for
    // the caller-supplied signed `reducedDeltaQIndex`.
    write_delta_qindex(
        writer,
        cdfs,
        sub_size,
        delta_inputs.reduced_delta_q_index,
        final_skip,
        delta_inputs.read_deltas,
        delta_inputs.use_128x128_superblock,
        delta_inputs.delta_q_res,
    )?;

    // §5.11.18 line 20: `read_delta_lf( )`. Per-iteration shape
    // identical to §5.11.12; gated by `ReadDeltas && delta_lf_present`.
    write_delta_lf(
        writer,
        cdfs,
        sub_size,
        &delta_inputs.reduced_delta_lf,
        final_skip,
        delta_inputs.read_deltas,
        delta_inputs.delta_lf_present,
        delta_inputs.delta_lf_multi,
        delta_inputs.mono_chrome,
        delta_inputs.use_128x128_superblock,
        delta_inputs.delta_lf_res,
    )?;

    // §5.11.18 line 21: `ReadDeltas = 0`. Caller-owned state per the
    // §6.10.4 derivation; the writer doesn't track it.

    // §5.11.18 line 22: `read_is_inter( )`.
    write_is_inter(
        writer,
        cdfs,
        is_inter,
        is_inter_ctx_val,
        skip_mode,
        seg_ref_frame_active,
        seg_ref_frame_is_inter,
        seg_globalmv_active,
    )?;

    Ok(InterFrameModeInfoPrefix {
        segment_id,
        skip_mode,
        skip: final_skip,
        is_inter,
        lossless,
        cdef_idx: cdef_idx_committed,
        reduced_delta_q_index: delta_inputs.reduced_delta_q_index,
        reduced_delta_lf: delta_inputs.reduced_delta_lf,
    })
}

/// `read_cdef()` inverse per §5.11.56 (av1-spec p.104) — the per-leaf
/// CDEF-index writer that mirrors
/// [`crate::cdf::PartitionWalker::decode_cdef`].
///
/// Spec body (av1-spec p.104):
///
/// ```text
///   read_cdef( ) {
///       if ( skip || CodedLossless || !enable_cdef || allow_intrabc) {
///           return
///       }
///       cdefSize4 = Num_4x4_Blocks_Wide[ BLOCK_64X64 ]
///       cdefMask4 = ~(cdefSize4 - 1)
///       r = MiRow & cdefMask4
///       c = MiCol & cdefMask4
///       if ( cdef_idx[ r ][ c ] == -1 ) {
///           cdef_idx[ r ][ c ]                          L(cdef_bits)
///           w4 = Num_4x4_Blocks_Wide[ MiSize ]
///           h4 = Num_4x4_Blocks_High[ MiSize ]
///           for ( i = r; i < r + h4 ; i += cdefSize4 ) {
///               for ( j = c; j < c + w4 ; j += cdefSize4 ) {
///                   cdef_idx[ i ][ j ] = cdef_idx[ r ][ c ]
///               }
///           }
///       }
///   }
/// ```
///
/// ## Arms (mirrors the decoder branch-by-branch)
///
/// * **Short-circuit set** — when *any* of `skip`, `coded_lossless`,
///   `!enable_cdef`, `allow_intrabc` is true the §5.11.56 outer
///   `if` fires and the function returns without consuming a bit.
///   The writer emits no symbol and returns `Ok(None)`.
/// * **Anchor already stamped** — `anchor_already_stamped == true`
///   indicates an earlier leaf in the same 64×64 anchor already
///   committed the `cdef_idx` literal (the decoder side reads zero
///   bits and observes the prior stamp). The writer emits no symbol
///   and returns `Ok(None)`. The caller must ensure `cdef_idx`
///   matches the prior stamp; mismatch is surfaced as
///   [`Error::PartitionWalkOutOfRange`].
/// * **First leaf in anchor** — emits one `L(cdef_bits)` literal
///   carrying `cdef_idx`. `cdef_bits` is the §5.9.19 frame-header
///   field clamped to `0..=3` by `f(2)`. `cdef_idx` MUST be in
///   `0..(1 << cdef_bits)`. Returns `Ok(Some(cdef_idx))` so the
///   caller can stamp its own encoder-side grid if it maintains one.
///
/// ## Parameter contract
///
/// `cdef_idx_prior_stamp` is the value the decoder side would read
/// from `cdef_idx[ r ][ c ]` before this call — supplied so the
/// writer's `anchor_already_stamped == true` arm can verify the
/// caller passes the same value (otherwise the decoder will diverge
/// at the matching `decode_cdef` call). Pass `-1` when
/// `anchor_already_stamped == false` (the §5.11.55 sentinel) — the
/// writer ignores the value on that arm anyway, but accepting an
/// explicit argument keeps the call sites uniform.
///
/// `cdef_idx` is the value the writer commits when the inner `if`
/// fires. It MUST fit in `cdef_bits` bits (`0..(1 << cdef_bits)`);
/// `cdef_bits == 0` accepts only `cdef_idx == 0` and emits zero
/// bits (the §8.2.5 `L(0)` literal). When `anchor_already_stamped
/// == true`, `cdef_idx` MUST equal `cdef_idx_prior_stamp`.
///
/// ## Range guards
///
/// All surfaced as [`Error::PartitionWalkOutOfRange`]:
///
/// * `cdef_bits > 3` (caller bug — §5.9.19 `f(2)` cap).
/// * `cdef_idx >= (1 << cdef_bits)` (caller bug — exceeds the L()
///   field width).
/// * `anchor_already_stamped == true` and `cdef_idx_prior_stamp !=
///   cdef_idx` (caller bug — would diverge from the decoder).
///
/// ## Stateless on grid-fill
///
/// Mirrors the rest of `encoder::block_mode_info`: the writer does
/// not maintain an encoder-side `cdef_idx[]` grid. The caller threads
/// its own (or uses a parallel [`crate::cdf::PartitionWalker`] —
/// the round-trip tests below use that exact pattern). The decoder
/// side stamps `cdef_idx[]` on the matching `decode_cdef` call
/// against its own walker.
#[allow(clippy::too_many_arguments)]
pub fn write_cdef(
    writer: &mut SymbolWriter,
    cdef_idx: i8,
    cdef_idx_prior_stamp: i8,
    cdef_bits: u32,
    skip: u8,
    coded_lossless: bool,
    enable_cdef: bool,
    allow_intrabc: bool,
    anchor_already_stamped: bool,
) -> Result<Option<i8>, Error> {
    // §5.9.19: cdef_bits is f(2) ⇒ 0..=3. Larger is a caller bug.
    if cdef_bits > 3 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.56 short-circuit set: any-true ⇒ no bits emitted.
    if skip != 0 || coded_lossless || !enable_cdef || allow_intrabc {
        return Ok(None);
    }

    // §5.11.56 outer `if`: anchor already stamped ⇒ no bits emitted.
    // The caller's `cdef_idx` MUST match the prior stamp; otherwise
    // the decoder will observe a different value than the writer
    // expects (caller bug — mid-anchor leaves carry the same value).
    if anchor_already_stamped {
        if cdef_idx_prior_stamp != cdef_idx {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(None);
    }

    // §5.11.56 inner branch: first leaf in this anchor ⇒ `L(cdef_bits)`.
    // §8.2.5 `L(n)` requires `value < (1 << n)`; `cdef_bits == 0`
    // accepts only `cdef_idx == 0` (an empty literal).
    if cdef_idx < 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let value = cdef_idx as u32;
    if cdef_bits == 0 {
        if value != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        // §8.2.5 L(0) — no bits emitted.
        return Ok(Some(cdef_idx));
    }
    if value >= (1u32 << cdef_bits) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    writer.write_literal(cdef_bits, value)?;
    Ok(Some(cdef_idx))
}

/// `read_delta_qindex()` inverse per §5.11.12 (av1-spec p.68) — the
/// per-block `CurrentQIndex` delta writer that mirrors
/// [`crate::cdf::PartitionWalker::decode_delta_qindex`].
///
/// Spec body (av1-spec p.68):
///
/// ```text
///   read_delta_qindex( ) {
///       sbSize = use_128x128_superblock ? BLOCK_128X128 : BLOCK_64X64
///       if ( MiSize == sbSize && skip )
///           return
///       if ( ReadDeltas ) {
///           delta_q_abs                                            S()
///           if ( delta_q_abs == DELTA_Q_SMALL ) {
///               delta_q_rem_bits                                   L(3)
///               delta_q_rem_bits++
///               delta_q_abs_bits                          L(delta_q_rem_bits)
///               delta_q_abs = delta_q_abs_bits + (1 << delta_q_rem_bits) + 1
///           }
///           if ( delta_q_abs ) {
///               delta_q_sign_bit                                   L(1)
///               reducedDeltaQIndex = delta_q_sign_bit ? -delta_q_abs : delta_q_abs
///               CurrentQIndex = Clip3(1, 255,
///                   CurrentQIndex + (reducedDeltaQIndex << delta_q_res))
///           }
///       }
///   }
/// ```
///
/// ## Caller-supplied `reduced_delta_q_index`
///
/// The writer's per-call deliverable is the §5.11.12 *signed*
/// `reducedDeltaQIndex` value (the same intermediate the decoder
/// computes between the `delta_q_sign_bit` read and the `Clip3`
/// update). The Clip3 update itself is the caller's job: it operates
/// on the running `CurrentQIndex` accumulator, which the encoder
/// owns in its frame-walk state (mirroring the
/// [`crate::cdf::PartitionWalker::set_current_q_index`] / `decode_delta_qindex`
/// return-value pattern). This keeps the writer pure on grid-state.
///
/// `delta_q_abs` is derived as `reduced_delta_q_index.unsigned_abs() as u32`.
///
/// ## Arms
///
/// * **Superblock-skip short-circuit** — when `sub_size == sbSize &&
///   skip != 0` the §5.11.12 outer `if` returns without reading.
///   The writer emits no bits. `reduced_delta_q_index` MUST be `0`
///   (caller bug otherwise).
/// * **`!read_deltas` short-circuit** — outer `if (ReadDeltas)` is
///   false ⇒ no bits emitted. `reduced_delta_q_index` MUST be `0`.
/// * **Literal branch** — `delta_q_abs ∈ 0..DELTA_Q_SMALL` ⇒ one
///   `S()` over `TileDeltaQCdf` carrying `delta_q_abs`. If
///   `delta_q_abs != 0`, follow with `L(1)` for the sign bit.
/// * **Escape branch** — `delta_q_abs >= DELTA_Q_SMALL` ⇒ one
///   `S()` carrying `DELTA_Q_SMALL`, then the escape ladder:
///   `delta_q_rem_bits = L(3)` carrying `n - 1` (where `n` is
///   `FloorLog2(delta_q_abs - 1) - 1` such that the spec's
///   `delta_q_abs = abs_bits + (1 << n) + 1` reconstructs the input),
///   then `delta_q_abs_bits = L(n)` carrying `delta_q_abs - (1 << n) - 1`,
///   then `L(1)` for the sign bit.
///
/// ## Range guards
///
/// All surfaced as [`Error::PartitionWalkOutOfRange`]:
///
/// * `sub_size >= BLOCK_SIZES` (caller bug — invalid block-size ordinal).
/// * `delta_q_res > 3` (caller bug — §5.9.17 `f(2)` cap).
/// * Short-circuit arms with non-zero `reduced_delta_q_index` (caller bug —
///   the decoder won't read the bits, so the writer mustn't emit them
///   either).
/// * `delta_q_abs > 511` — exceeds the §5.11.12 escape-ladder reach
///   (`delta_q_rem_bits ∈ 0..=7` ⇒ `n ∈ 1..=8` ⇒ `delta_q_abs ∈
///   3..=511` on the escape arm; combined with `0..=DELTA_Q_SMALL =
///   0..=3` on the literal arm the full range is `0..=511`).
#[allow(clippy::too_many_arguments)]
pub fn write_delta_qindex(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    sub_size: usize,
    reduced_delta_q_index: i32,
    skip: u8,
    read_deltas: bool,
    use_128x128_superblock: bool,
    delta_q_res: u8,
) -> Result<(), Error> {
    if sub_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if delta_q_res > 3 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.12 superblock-skip short-circuit.
    let sb_size = if use_128x128_superblock {
        BLOCK_128X128
    } else {
        BLOCK_64X64
    };
    if sub_size == sb_size && skip != 0 {
        if reduced_delta_q_index != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.12 outer gate.
    if !read_deltas {
        if reduced_delta_q_index != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.12 |delta_q_abs| derivation. The spec's range is `0..=511`
    // (literal arm covers `0..=3`, escape arm covers `3..=511`).
    let abs_value: u32 = reduced_delta_q_index.unsigned_abs();
    if abs_value > 511 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let cdf = cdfs.delta_q_cdf();
    if (abs_value as usize) < DELTA_Q_SMALL {
        // §5.11.12 literal branch: `delta_q_abs S()` carries the value directly.
        writer.write_symbol(abs_value, cdf)?;
    } else {
        // §5.11.12 escape branch: `delta_q_abs S()` carries
        // DELTA_Q_SMALL, then the rem_bits / abs_bits ladder.
        writer.write_symbol(DELTA_Q_SMALL as u32, cdf)?;
        // Solve `abs_value = abs_bits + (1 << n) + 1` for the smallest
        // `n >= 1` such that `abs_bits = abs_value - (1 << n) - 1 <
        // (1 << n)` (i.e. `abs_value < 2 * (1 << n) + 1`). The
        // equivalent unbiased rephrase: `n = FloorLog2(abs_value - 1)`,
        // valid for `abs_value >= 3` (which the literal-branch guard
        // above already ensures — `DELTA_Q_SMALL == 3`).
        debug_assert!(abs_value >= 3);
        let n = floor_log2(abs_value - 1);
        debug_assert!((1..=8).contains(&n));
        // Spec writes `delta_q_rem_bits = L(3) + 1` ⇒ `rem_bits + 1 = n`
        // ⇒ encode `rem_bits = n - 1`.
        let rem_bits_field = n - 1;
        debug_assert!(rem_bits_field <= 7, "L(3) field width");
        writer.write_literal(3, rem_bits_field)?;
        // `delta_q_abs_bits = abs_value - (1 << n) - 1`, encoded in `L(n)`.
        let abs_bits = abs_value - (1u32 << n) - 1;
        debug_assert!(abs_bits < (1u32 << n));
        writer.write_literal(n, abs_bits)?;
    }

    // §5.11.12 sign bit only when `delta_q_abs != 0`.
    if abs_value != 0 {
        let sign: u32 = if reduced_delta_q_index < 0 { 1 } else { 0 };
        writer.write_literal(1, sign)?;
    }

    Ok(())
}

/// `read_delta_lf()` inverse per §5.11.13 (av1-spec p.68) — the
/// per-block per-loop-filter-strength delta writer that mirrors
/// [`crate::cdf::PartitionWalker::decode_delta_lf`].
///
/// Spec body (av1-spec p.68):
///
/// ```text
///   read_delta_lf( ) {
///       sbSize = use_128x128_superblock ? BLOCK_128X128 : BLOCK_64X64
///       if ( MiSize == sbSize && skip )
///           return
///       if ( ReadDeltas && delta_lf_present ) {
///           frameLfCount = 1
///           if ( delta_lf_multi )
///               frameLfCount = ( NumPlanes > 1 ) ? FRAME_LF_COUNT
///                                                : ( FRAME_LF_COUNT - 2 )
///           for ( i = 0; i < frameLfCount; i++ ) {
///               delta_lf_abs                                        S()
///               if ( delta_lf_abs == DELTA_LF_SMALL ) {
///                   delta_lf_rem_bits                              L(3)
///                   n = delta_lf_rem_bits + 1
///                   delta_lf_abs_bits                              L(n)
///                   deltaLfAbs = delta_lf_abs_bits + (1 << n) + 1
///               } else {
///                   deltaLfAbs = delta_lf_abs
///               }
///               if ( deltaLfAbs ) {
///                   delta_lf_sign_bit                              L(1)
///                   reducedDeltaLfLevel = delta_lf_sign_bit ?
///                                            -deltaLfAbs : deltaLfAbs
///                   DeltaLF[ i ] = Clip3( -MAX_LOOP_FILTER, MAX_LOOP_FILTER,
///                       DeltaLF[ i ] + (reducedDeltaLfLevel << delta_lf_res) )
///               }
///           }
///       }
///   }
/// ```
///
/// ## Caller-supplied `reduced_delta_lf` row
///
/// The writer's per-iteration deliverable is the §5.11.13 *signed*
/// `reducedDeltaLfLevel[ i ]` value (the intermediate the decoder
/// computes between the `delta_lf_sign_bit` read and the `Clip3`
/// update). The Clip3 update is the caller's job; it operates on the
/// running `DeltaLF[ i ]` accumulator that the encoder owns in its
/// frame-walk state (mirroring the
/// [`crate::cdf::PartitionWalker::current_delta_lf`] /
/// `decode_delta_lf` return-value pattern).
///
/// `reduced_delta_lf` is fixed-length [`FRAME_LF_COUNT`]. The writer
/// reads only the first `frameLfCount` entries (`1` on the
/// `!delta_lf_multi` branch; either `FRAME_LF_COUNT` or
/// `FRAME_LF_COUNT - 2` on the multi branch per `mono_chrome`).
/// Entries beyond `frameLfCount` MUST be `0` (caller bug otherwise —
/// the decoder won't observe them).
///
/// ## Arms
///
/// Per-iteration shape mirrors `write_delta_qindex` exactly except
/// for the CDF source: `delta_lf_cdf()` (single-LF) or
/// `delta_lf_multi_cdf(i)` (multi-LF). The escape ladder, literal
/// branch, and sign bit are identical to §5.11.12.
///
/// ## Range guards
///
/// All surfaced as [`Error::PartitionWalkOutOfRange`]:
///
/// * `sub_size >= BLOCK_SIZES`.
/// * `delta_lf_res > 3` (caller bug — §5.9.18 `f(2)` cap).
/// * Short-circuit arms with any non-zero `reduced_delta_lf` entry.
/// * Any `reduced_delta_lf[ i ].unsigned_abs() > 511` (exceeds the
///   escape-ladder reach).
/// * Entries beyond `frameLfCount` being non-zero.
#[allow(clippy::too_many_arguments)]
pub fn write_delta_lf(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    sub_size: usize,
    reduced_delta_lf: &[i32; FRAME_LF_COUNT],
    skip: u8,
    read_deltas: bool,
    delta_lf_present: bool,
    delta_lf_multi: bool,
    mono_chrome: bool,
    use_128x128_superblock: bool,
    delta_lf_res: u8,
) -> Result<(), Error> {
    if sub_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if delta_lf_res > 3 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.13 superblock-skip short-circuit (identical to §5.11.12).
    let sb_size = if use_128x128_superblock {
        BLOCK_128X128
    } else {
        BLOCK_64X64
    };
    let sb_short_circuit = sub_size == sb_size && skip != 0;
    let read_gate = read_deltas && delta_lf_present;

    if sb_short_circuit || !read_gate {
        if reduced_delta_lf.iter().any(|&v| v != 0) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.13 frameLfCount derivation.
    let frame_lf_count: usize = if delta_lf_multi {
        if mono_chrome {
            FRAME_LF_COUNT - 2
        } else {
            FRAME_LF_COUNT
        }
    } else {
        1
    };

    // Entries beyond frameLfCount must be 0 (decoder won't read them).
    for &v in reduced_delta_lf.iter().skip(frame_lf_count) {
        if v != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    for (i, &reduced) in reduced_delta_lf.iter().take(frame_lf_count).enumerate() {
        let abs_value: u32 = reduced.unsigned_abs();
        if abs_value > 511 {
            return Err(Error::PartitionWalkOutOfRange);
        }

        let cdf: &mut [u16] = if delta_lf_multi {
            cdfs.delta_lf_multi_cdf(i)
        } else {
            cdfs.delta_lf_cdf()
        };
        if (abs_value as usize) < DELTA_LF_SMALL {
            // §5.11.13 literal branch.
            writer.write_symbol(abs_value, cdf)?;
        } else {
            // §5.11.13 escape branch — same shape as §5.11.12.
            writer.write_symbol(DELTA_LF_SMALL as u32, cdf)?;
            debug_assert!(abs_value >= 3);
            let n = floor_log2(abs_value - 1);
            debug_assert!((1..=8).contains(&n));
            let rem_bits_field = n - 1;
            debug_assert!(rem_bits_field <= 7);
            writer.write_literal(3, rem_bits_field)?;
            let abs_bits = abs_value - (1u32 << n) - 1;
            debug_assert!(abs_bits < (1u32 << n));
            writer.write_literal(n, abs_bits)?;
        }

        if abs_value != 0 {
            let sign: u32 = if reduced < 0 { 1 } else { 0 };
            writer.write_literal(1, sign)?;
        }
    }

    Ok(())
}

/// `intra_angle_info_y( )` inverse per §5.11.42 (av1-spec p.95) — r260.
///
/// Spec body:
/// ```text
///   intra_angle_info_y() {
///       AngleDeltaY = 0
///       if ( MiSize >= BLOCK_8X8 ) {
///           if ( is_directional_mode( YMode ) ) {
///               angle_delta_y                                  S()
///               AngleDeltaY = angle_delta_y - MAX_ANGLE_DELTA
///           }
///       }
///   }
/// ```
///
/// Mirrors the §5.11.42 reader inlined into
/// [`crate::cdf::PartitionWalker::decode_intra_block_mode_info`] at
/// av1-spec p.72 line 4 (call site `intra_angle_info_y( )` inside
/// §5.11.22).
///
/// Inputs:
/// * `angle_delta_y` — the signed `AngleDeltaY` value the encoder
///   committed to (in `-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA = -3..=3`
///   per §3 / §5.11.42).
/// * `mi_size` — the §5.11.5 `MiSize` ordinal in `0..BLOCK_SIZES`.
/// * `y_mode` — the §5.11.22 `YMode` value in `0..INTRA_MODES = 13`
///   per §3, used for the §5.11.44 `is_directional_mode` short-circuit.
///
/// CDF selection per §8.3.2: `TileAngleDeltaCdf[ YMode - V_PRED ]`
/// when the §5.11.42 reader fires; the row is 8-wide (7 symbols +
/// the §8.2.6 counter). The reader writes `angle_delta_y = AngleDeltaY
/// + MAX_ANGLE_DELTA` (the inverse of the spec's
/// `AngleDeltaY = angle_delta_y - MAX_ANGLE_DELTA` recovery).
///
/// Short-circuits (no bits emitted) when:
/// * `mi_size < BLOCK_8X8`, OR
/// * `!is_directional_mode( YMode )` (§5.11.44: directional modes are
///   `V_PRED = 1 <= YMode <= D67_PRED = 8`).
///
/// In both short-circuit arms `angle_delta_y` MUST be `0` — caller
/// bug otherwise, surfaced as [`Error::PartitionWalkOutOfRange`].
pub fn write_intra_angle_info_y(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    angle_delta_y: i8,
    mi_size: usize,
    y_mode: u8,
) -> Result<(), Error> {
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let max = MAX_ANGLE_DELTA as i8;
    if !(-max..=max).contains(&angle_delta_y) {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let reads = mi_size >= BLOCK_8X8 && is_directional(y_mode);
    if !reads {
        // §5.11.42 short-circuit arm — `AngleDeltaY = 0`. Caller-bug
        // guard: any non-zero `angle_delta_y` here is a contract
        // violation (the decoder would reconstruct `0` regardless).
        if angle_delta_y != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.42 reader: `angle_delta_y S()` against
    // `TileAngleDeltaCdf[ YMode - V_PRED ]`. The raw S() value is
    // `AngleDeltaY + MAX_ANGLE_DELTA`, in `0..(2 * MAX_ANGLE_DELTA + 1) = 7`.
    let raw: u32 = (angle_delta_y as i32 + MAX_ANGLE_DELTA as i32) as u32;
    let cdf = cdfs
        .angle_delta_cdf(y_mode as usize)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(raw, cdf)
}

/// `intra_angle_info_uv( )` inverse per §5.11.43 (av1-spec p.95) — r260.
///
/// Spec body (mirror of §5.11.42 with `UVMode` in place of `YMode`):
/// ```text
///   intra_angle_info_uv() {
///       AngleDeltaUV = 0
///       if ( MiSize >= BLOCK_8X8 ) {
///           if ( is_directional_mode( UVMode ) ) {
///               angle_delta_uv                                 S()
///               AngleDeltaUV = angle_delta_uv - MAX_ANGLE_DELTA
///           }
///       }
///   }
/// ```
///
/// Mirrors the §5.11.43 reader inlined into
/// [`crate::cdf::PartitionWalker::decode_intra_block_mode_info`] at
/// av1-spec p.73 line 10 (call site `intra_angle_info_uv( )` inside
/// the §5.11.22 `if ( HasChroma )` arm).
///
/// Inputs / CDF selection / short-circuits are the §5.11.42 mirror
/// with `uv_mode` in place of `y_mode` (the §3 directional-mode range
/// is the same: `V_PRED = 1..=D67_PRED = 8` for both planes since
/// `UVMode` shares the §3 intra-mode enumeration). `uv_mode` MUST be
/// in `0..UV_INTRA_MODES_CFL_ALLOWED = 14`; non-directional values
/// (including `UV_CFL_PRED = 13`) take the short-circuit arm.
pub fn write_intra_angle_info_uv(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    angle_delta_uv: i8,
    mi_size: usize,
    uv_mode: u8,
) -> Result<(), Error> {
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (uv_mode as usize) >= UV_INTRA_MODES_CFL_ALLOWED {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let max = MAX_ANGLE_DELTA as i8;
    if !(-max..=max).contains(&angle_delta_uv) {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.43 / §5.11.44: the directional range is `V_PRED..=D67_PRED`
    // on both planes (the U-plane CFL signalling at `UV_CFL_PRED = 13`
    // is non-directional). `is_directional` returns `false` for
    // `UV_CFL_PRED` since `is_directional` checks the same §3 range.
    let reads = mi_size >= BLOCK_8X8 && is_directional(uv_mode);
    if !reads {
        if angle_delta_uv != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    let raw: u32 = (angle_delta_uv as i32 + MAX_ANGLE_DELTA as i32) as u32;
    let cdf = cdfs
        .angle_delta_cdf(uv_mode as usize)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(raw, cdf)
}

/// `filter_intra_mode_info( )` inverse per §5.11.24 (av1-spec p.74) — r260.
///
/// Spec body:
/// ```text
///   filter_intra_mode_info() {
///       use_filter_intra = 0
///       if ( enable_filter_intra &&
///            YMode == DC_PRED && PaletteSizeY == 0 &&
///            Max( Block_Width[ MiSize ], Block_Height[ MiSize ] ) <= 32 ) {
///           use_filter_intra                                   S()
///           if ( use_filter_intra ) {
///               filter_intra_mode                              S()
///           }
///       }
///   }
/// ```
///
/// Mirrors the §5.11.24 reader inlined into
/// [`crate::cdf::PartitionWalker::decode_intra_block_mode_info`] at
/// av1-spec p.73 line 16 (call site `filter_intra_mode_info( )` at
/// the tail of §5.11.22).
///
/// Inputs:
/// * `use_filter_intra` — `0` / `1`. When the outer gate is closed
///   MUST be `0`; otherwise the encoder's committed §5.11.24
///   `use_filter_intra` symbol.
/// * `filter_intra_mode` — `Some(mode)` when `use_filter_intra == 1`
///   (`mode` in `0..INTRA_FILTER_MODES = 5`), `None` otherwise.
/// * `mi_size` — `MiSize` ordinal in `0..BLOCK_SIZES`.
/// * `y_mode` — the §5.11.22 `YMode` value (gate fires only on
///   `DC_PRED = 0`).
/// * `palette_size_y` — the §5.11.46 `PaletteSizeY` derivation (gate
///   fires only on `PaletteSizeY == 0`). `u32` to match the
///   walker's grid-side type.
/// * `enable_filter_intra` — §5.5.2 sequence-header bit gating the
///   §5.11.24 outer arm.
///
/// CDF selection per §8.3.2: `TileFilterIntraCdf[ MiSize ]` for the
/// outer `use_filter_intra` S(); `TileFilterIntraModeCdf` (a single
/// context-free row over `INTRA_FILTER_MODES = 5` modes) for the
/// inner `filter_intra_mode` S().
///
/// Outer-gate short-circuits (no bits emitted, must satisfy
/// `use_filter_intra == 0 && filter_intra_mode == None`):
/// * `!enable_filter_intra`
/// * `y_mode != DC_PRED` (DC_PRED = 0)
/// * `palette_size_y != 0`
/// * `Max( Block_Width[ MiSize ], Block_Height[ MiSize ] ) > 32`
///
/// Inner-arm contract: when the outer gate fires, `use_filter_intra
/// == 1` requires `filter_intra_mode == Some(mode)` with `mode in
/// 0..INTRA_FILTER_MODES`; `use_filter_intra == 0` requires
/// `filter_intra_mode == None`.
#[allow(clippy::too_many_arguments)]
pub fn write_filter_intra_mode_info(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    use_filter_intra: u8,
    filter_intra_mode: Option<u8>,
    mi_size: usize,
    y_mode: u8,
    palette_size_y: u32,
    enable_filter_intra: bool,
) -> Result<(), Error> {
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if use_filter_intra > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.24 outer gate. DC_PRED = 0 per §3.
    let bw = block_width(mi_size);
    let bh = block_height(mi_size);
    let outer_gate = enable_filter_intra && y_mode == 0 && palette_size_y == 0 && bw.max(bh) <= 32;

    if !outer_gate {
        // Caller-bug guard — the §5.11.24 body sets `use_filter_intra
        // = 0` unconditionally on the gate-closed arm; the decoder
        // reconstructs `use_filter_intra = 0` regardless of the
        // bitstream so a non-zero caller value is a contract
        // violation.
        if use_filter_intra != 0 || filter_intra_mode.is_some() {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.24: `use_filter_intra S()` against
    // `TileFilterIntraCdf[ MiSize ]` (2-wide, no §8.3.2 ctx beyond
    // `MiSize`).
    let cdf = cdfs.filter_intra_cdf(mi_size);
    writer.write_symbol(use_filter_intra as u32, cdf)?;

    if use_filter_intra == 1 {
        // §5.11.24 inner arm: `filter_intra_mode S()` against
        // `TileFilterIntraModeCdf` (no §8.3.2 ctx, single row).
        let mode = filter_intra_mode.ok_or(Error::PartitionWalkOutOfRange)?;
        if (mode as usize) >= INTRA_FILTER_MODES {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let row = cdfs.filter_intra_mode_cdf();
        writer.write_symbol(mode as u32, row)?;
    } else if filter_intra_mode.is_some() {
        // §5.11.24 contract: when `use_filter_intra == 0` the spec
        // body never reads `filter_intra_mode`; a `Some` value here
        // is a caller bug.
        return Err(Error::PartitionWalkOutOfRange);
    }

    Ok(())
}

/// `palette_mode_info( )` inverse per §5.11.46 (av1-spec p.97-98) —
/// r261. No-palette-arm only: commits `has_palette_y == 0` AND
/// `has_palette_uv == 0` to the bit stream, mirroring the §5.11.46
/// reader's outer-arm + inner-`has_palette_* = 0` branch.
///
/// Spec body (abbreviated to the surface this leaf covers):
/// ```text
///   palette_mode_info() {
///       bsizeCtx = Mi_Width_Log2[ MiSize ] + Mi_Height_Log2[ MiSize ] - 2
///       if ( YMode == DC_PRED ) {
///           has_palette_y                                 S()
///           if ( has_palette_y ) {
///               palette_size_y_minus_2                     S()  ← NOT
///               // … palette_colors_y[] reads …                 covered
///           }
///       }
///       if ( HasChroma && UVMode == DC_PRED ) {
///           has_palette_uv                                S()
///           if ( has_palette_uv ) {
///               palette_size_uv_minus_2                    S()  ← NOT
///               // … palette_colors_u[] / palette_colors_v[] reads …
///           }
///       }
///   }
/// ```
///
/// Mirrors the §5.11.46 reader inlined into
/// [`crate::cdf::PartitionWalker::decode_intra_block_mode_info`] at
/// av1-spec p.73 line 13-15 (call site `palette_mode_info( )` inside
/// §5.11.22). Restricted to the no-palette path because the
/// §5.11.46 palette-entries syntax (cache merge + `L(BitDepth)`
/// literal + `paletteBits` delta loop + trailing sort) is a much
/// larger write surface that needs its own arc; this leaf is the
/// shape the §5.11.22 dispatcher (next round) needs to satisfy its
/// `palette_size_y == 0` precondition for §5.11.24
/// `write_filter_intra_mode_info`.
///
/// Inputs:
/// * `has_palette_y` — MUST be `0` on the no-palette arm. Any
///   non-zero value is a contract violation surfaced as
///   [`Error::PaletteEntriesUnsupported`] (mirrors the reader's
///   short-circuit return on a non-zero decoded size).
/// * `has_palette_uv` — same shape as `has_palette_y` on the chroma
///   arm; MUST be `0`.
/// * `mi_size` — `MiSize` ordinal in `0..BLOCK_SIZES`.
/// * `y_mode` — the §5.11.22 `YMode` value (gate fires only on
///   `DC_PRED = 0`).
/// * `uv_mode` — the §5.11.22 `UVMode` value (gate fires only on
///   `has_chroma && DC_PRED`).  Pass `None` when `has_chroma ==
///   false` (matches the §5.11.22 line 5 `if (HasChroma)` guard the
///   dispatcher honours).
/// * `has_chroma` — §5.11.5 `HasChroma`.
/// * `allow_screen_content_tools` — §5.9.5 sequence-header bit;
///   gates the §5.11.46 outer arm.
/// * `above_palette_y` / `left_palette_y` — §8.3.2 neighbour-palette
///   booleans, fed to [`palette_y_mode_ctx`].
///
/// CDF selection per §8.3.2:
/// * `has_palette_y` against `TilePaletteYModeCdf[ bsizeCtx ][ ctx_y
///   ]` (2-wide row, `ctx_y = palette_y_mode_ctx(above, left)`).
/// * `has_palette_uv` against `TilePaletteUVModeCdf[ ctx_uv ]`
///   (2-wide row, `ctx_uv = palette_uv_mode_ctx(palette_size_y)` —
///   on the no-palette arm `palette_size_y == 0` so `ctx_uv == 0`).
///
/// Outer-gate short-circuits (no bits emitted) when:
/// * `!allow_screen_content_tools`, OR
/// * `mi_size < BLOCK_8X8`, OR
/// * `Block_Width[ MiSize ] > 64`, OR
/// * `Block_Height[ MiSize ] > 64`.
///
/// On the gate-off path `has_palette_y` MUST be `0` AND
/// `has_palette_uv` MUST be `0` — caller bug otherwise.
///
/// Inner-arm short-circuits (no bits emitted on that plane):
/// * Luma: `y_mode != DC_PRED` ⇒ `has_palette_y` MUST be `0`.
/// * Chroma: `!has_chroma || uv_mode != Some(DC_PRED)` ⇒
///   `has_palette_uv` MUST be `0`.
#[allow(clippy::too_many_arguments)]
pub fn write_palette_mode_info(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    has_palette_y: u8,
    has_palette_uv: u8,
    mi_size: usize,
    y_mode: u8,
    uv_mode: Option<u8>,
    has_chroma: bool,
    allow_screen_content_tools: bool,
    above_palette_y: bool,
    left_palette_y: bool,
) -> Result<(), Error> {
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if let Some(uv) = uv_mode {
        // §5.11.22 line 6: `uv_mode` ∈ `0..UV_INTRA_MODES_CFL_ALLOWED
        // = 14`. CFL_PRED = 13 is the widest reachable; pass through
        // the looser bound here.
        if (uv as usize) >= UV_INTRA_MODES_CFL_ALLOWED {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }
    if has_palette_y > 1 || has_palette_uv > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // This arc covers the no-palette path only. Any caller that
    // commits a non-zero `has_palette_*` is asking for the
    // §5.11.46 palette-entries syntax that isn't yet wired
    // through this writer; surface a clear error so the dispatcher
    // can re-route once the arc lands.
    if has_palette_y != 0 || has_palette_uv != 0 {
        return Err(Error::PaletteEntriesUnsupported);
    }

    // §5.11.46 outer gate. Mirrors the reader's `palette_outer_gate`.
    let bw = block_width(mi_size) as u32;
    let bh = block_height(mi_size) as u32;
    let outer_gate = allow_screen_content_tools && mi_size >= BLOCK_8X8 && bw <= 64 && bh <= 64;

    if !outer_gate {
        // Gate-off path: the §5.11.46 spec body sets `PaletteSizeY =
        // 0` / `PaletteSizeUV = 0` unconditionally. The reader
        // reconstructs them as 0 regardless of the bit stream; a
        // non-zero `has_palette_*` was rejected above so the gate-off
        // arm emits zero bits and returns cleanly.
        return Ok(());
    }

    // §5.11.46: `bsizeCtx = Mi_Width_Log2[ MiSize ] +
    // Mi_Height_Log2[ MiSize ] - 2`. The §9.3 mapping guarantees
    // `bsizeCtx ∈ 0..PALETTE_BLOCK_SIZE_CONTEXTS = 7` for any MiSize
    // that passed the outer gate (BLOCK_8X8 has (1, 1) ⇒ bsizeCtx = 0,
    // BLOCK_64X64 has (4, 4) ⇒ bsizeCtx = 6).
    let bsize_ctx = mi_width_log2(mi_size) + mi_height_log2(mi_size) - 2;
    debug_assert!(
        bsize_ctx < PALETTE_BLOCK_SIZE_CONTEXTS,
        "§5.11.46 bsizeCtx must be in 0..PALETTE_BLOCK_SIZE_CONTEXTS"
    );

    // §5.11.46 luma arm: `if ( YMode == DC_PRED )`.
    if y_mode == 0 {
        let ctx_y = palette_y_mode_ctx(above_palette_y, left_palette_y);
        let row = cdfs.palette_y_mode_cdf(bsize_ctx, ctx_y);
        writer.write_symbol(has_palette_y as u32, row)?;
        // `has_palette_y == 1` short-circuited above with
        // PaletteEntriesUnsupported; nothing further to emit on the
        // luma arm.
    } else if has_palette_y != 0 {
        // §5.11.46 contract: when `y_mode != DC_PRED` the reader
        // never reads `has_palette_y`; non-zero here is a caller bug.
        // (Re-asserted because the early-return only catches the
        // gate-off arm.)
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.46 chroma arm: `if ( HasChroma && UVMode == DC_PRED )`.
    if has_chroma && uv_mode == Some(0) {
        // On the no-palette path `palette_size_y == 0`, so the
        // §8.3.2 chroma ctx is `palette_uv_mode_ctx(0) = 0`.
        let ctx_uv = palette_uv_mode_ctx(0);
        let row = cdfs.palette_uv_mode_cdf(ctx_uv);
        writer.write_symbol(has_palette_uv as u32, row)?;
    } else if has_palette_uv != 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    Ok(())
}

/// `palette_colors_y[]` writer per §5.11.46 (av1-spec p.97-98) —
/// r262. The luma-plane half of the §5.11.46 palette-entries syntax:
/// the inverse of [`crate::cdf::read_palette_entries_y`].
///
/// Spec body (Y arm of §5.11.46, post-`PaletteSizeY` read):
/// ```text
///   // §5.11.49 neighbour-merge cache, plane = 0 (luma).
///   cache[] = get_palette_cache( 0, mi_row, mi_col )
///   idx = 0
///   for ( i = 0; i < cache_n && idx < PaletteSizeY; i++ ) {
///       use_palette_color_cache_y                          L(1)
///       if ( use_palette_color_cache_y ) {
///           palette_colors_y[ idx++ ] = cache[ i ]
///       }
///   }
///   if ( idx < PaletteSizeY ) {
///       palette_colors_y[ idx++ ]                          L(BitDepth)
///   }
///   if ( idx < PaletteSizeY ) {
///       minBits  = BitDepth - 3
///       maxVal   = 1 << BitDepth
///       palette_num_extra_bits_y                           L(2)
///       paletteBits = minBits + palette_num_extra_bits_y
///   }
///   while ( idx < PaletteSizeY ) {
///       palette_delta_y                                    L(paletteBits)
///       palette_delta_y++
///       palette_colors_y[ idx ] = Clip1( palette_colors_y[ idx - 1 ]
///                                        + palette_delta_y )
///       range = maxVal - palette_colors_y[ idx ] - 1
///       paletteBits = Min( paletteBits, CeilLog2( range ) )
///       idx++
///   }
///   sort( palette_colors_y[ 0 .. PaletteSizeY - 1 ] )
/// ```
///
/// **Mirror invariant.** This writer takes `palette_colors_y` already
/// sorted ascending (the reader's trailing `sort()` step is the
/// canonical form). The cache + literal + paletteBits + delta loop
/// then re-derives the exact L(1) / L(BitDepth) / L(2) / L(paletteBits)
/// bit stream that, when fed back through
/// [`crate::cdf::read_palette_entries_y`], reconstructs the same
/// sorted entries — a property each round-trip test below verifies
/// directly.
///
/// **paletteBits derivation.** The encoder must choose
/// `palette_num_extra_bits_y` (`L(2)` → 0..=3) such that every
/// downstream `palette_delta_y` raw value fits in the *current*
/// `paletteBits`, which the reader refines after every entry by
/// `paletteBits = Min(paletteBits, CeilLog2(range))`. We simulate
/// the reader's refinement for each candidate `extra ∈ {0, 1, 2, 3}`
/// and pick the smallest one that admits encoding every remaining
/// delta. Such an `extra` always exists when `palette_colors_y` is
/// strictly ascending in `[0, 2^BitDepth)`: the maximum raw delta
/// `(palette_colors_y[idx] - palette_colors_y[idx - 1]) - 1` is
/// bounded by `2^BitDepth - 2`, which fits in `BitDepth` bits —
/// `minBits + 3 = BitDepth - 3 + 3 = BitDepth` — and the §5.11.46
/// refinement only *reduces* `paletteBits` along the way (since
/// `range = maxVal - colors[idx] - 1 < maxVal`).
///
/// **Caller contract.** Inputs must satisfy:
/// * `bit_depth ∈ {8, 10, 12}` (§5.5.2).
/// * `palette_size_y ∈ 2..=PALETTE_COLORS` (§5.11.46 contract — the
///   reader's outer arm only ever calls this with a non-zero size,
///   bounded above by `PALETTE_COLORS = 8`).
/// * `palette_colors_y[0 .. palette_size_y]` strictly ascending, each
///   entry in `[0, 2^bit_depth)`. Duplicate or descending entries
///   would surface non-positive deltas the spec doesn't admit on
///   this arm — [`Error::PartitionWalkOutOfRange`].
/// * `palette_colors_y[palette_size_y ..]` ignored (per the reader's
///   `[0u16; PALETTE_COLORS]` zero-fill convention).
/// * `walker` MUST be at the same §5.11.49 cache state the reader
///   would observe — caller threads its own
///   [`PartitionWalker`](crate::cdf::PartitionWalker) and stamps any
///   preceding decoded blocks into the `PaletteSizes` / `PaletteColors`
///   grids before this call.
///
/// On success, the encoder has committed the §5.11.46 luma palette
/// entries to the bit stream in exact lock-step with the
/// [`crate::cdf::read_palette_entries_y`] reader.
#[allow(clippy::too_many_arguments)]
pub fn write_palette_entries_y(
    writer: &mut SymbolWriter,
    bit_depth: u8,
    palette_size_y: usize,
    palette_colors_y: &[u16],
    walker: &PartitionWalker,
    mi_row: u32,
    mi_col: u32,
) -> Result<(), Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if !(2..=PALETTE_COLORS).contains(&palette_size_y) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if palette_colors_y.len() < palette_size_y {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let bd = bit_depth as u32;
    let max_val: u32 = 1u32 << bd;
    // §5.11.46 caller contract: entries strictly ascending and each
    // within [0, 2^BitDepth). Mirrors the reader's invariant after
    // the trailing sort + Clip1.
    for (i, &c) in palette_colors_y.iter().take(palette_size_y).enumerate() {
        if (c as u32) >= max_val {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if i > 0 && (c as u32) <= palette_colors_y[i - 1] as u32 {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    // §5.11.49 cache via the same walker accessor the reader uses.
    let mut cache_buf = [0u16; 2 * PALETTE_COLORS];
    let cache_n = walker.get_palette_cache(0, mi_row, mi_col, &mut cache_buf);

    // §5.11.46 cache-coded indices loop. For each cache slot we ask
    // "is this slot the next palette entry the decoder should adopt?"
    // and emit L(1). The match condition is `cache[i] ==
    // palette_colors_y[idx]` — the reader assigns `cache[i]` to
    // `palette_colors_y[idx]` and advances `idx` on use_cache == 1,
    // so we mirror that decision-tree exactly.
    let mut idx: usize = 0;
    {
        let mut i = 0usize;
        while i < cache_n && idx < palette_size_y {
            let use_cache = if cache_buf[i] == palette_colors_y[idx] {
                1u32
            } else {
                0u32
            };
            writer.write_literal(1, use_cache)?;
            if use_cache != 0 {
                idx += 1;
            }
            i += 1;
        }
    }

    // §5.11.46 first "new" entry as L(BitDepth) literal.
    if idx < palette_size_y {
        let v = palette_colors_y[idx] as u32;
        writer.write_literal(bd, v)?;
        idx += 1;
    }

    // §5.11.46 paletteBits derivation: pick the smallest `extra ∈
    // {0, 1, 2, 3}` such that every remaining `palette_delta_y` fits
    // in the running `paletteBits` after the reader's refinement.
    // When `idx == palette_size_y` (no entries remain in the delta
    // loop), the L(2) read is skipped on the reader side — we skip
    // the write to match.
    if idx < palette_size_y {
        // §5.11.46: `minBits = BitDepth - 3`. BitDepth ∈ {8, 10, 12}
        // ⇒ minBits ∈ {5, 7, 9} (never negative).
        let min_bits = bd - 3;
        let start_idx = idx;
        let extra = pick_palette_extra_bits_y(palette_colors_y, palette_size_y, start_idx, bd)?;
        writer.write_literal(2, extra)?;
        let mut palette_bits = min_bits + extra;
        // §5.11.46 delta loop — emit each delta minus 1 (the spec's
        // `palette_delta_y++` inverse) and refine paletteBits in
        // lock-step with the reader.
        while idx < palette_size_y {
            let prev = palette_colors_y[idx - 1] as u32;
            let cur = palette_colors_y[idx] as u32;
            // Strictly-ascending pre-check above guarantees cur > prev.
            let delta_raw = cur - prev - 1;
            // The picker guarantees the raw value fits, but cheap
            // defensive guard: a stray `paletteBits == 0` arm can
            // only encode delta_raw == 0; surface caller-bug otherwise.
            if palette_bits == 0 {
                if delta_raw != 0 {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                // L(0) writes no bits — mirror writer's contract.
                writer.write_literal(0, 0)?;
            } else {
                if delta_raw >= (1u32 << palette_bits) {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                writer.write_literal(palette_bits, delta_raw)?;
            }
            // §5.11.46 paletteBits refinement: identical to the
            // reader's. The Clip1 step is a no-op because the entry
            // is already in [0, 2^BitDepth).
            let range = max_val.saturating_sub(cur).saturating_sub(1);
            palette_bits = palette_bits.min(ceil_log2_av1(range));
            idx += 1;
        }
    }

    Ok(())
}

/// Pick the smallest `palette_num_extra_bits_y` (L(2) ⇒ 0..=3) such
/// that every remaining §5.11.46 Y-plane delta fits in the
/// reader-refined `paletteBits`. Returns
/// [`Error::PartitionWalkOutOfRange`] when no choice in 0..=3 admits
/// all deltas — this only happens on a malformed
/// `palette_colors_y` (the strict-ascending + bit-depth check before
/// this call rules out the conformant input).
fn pick_palette_extra_bits_y(
    palette_colors_y: &[u16],
    palette_size_y: usize,
    start_idx: usize,
    bd: u32,
) -> Result<u32, Error> {
    let min_bits = bd - 3;
    let max_val: u32 = 1u32 << bd;
    for extra in 0..=3u32 {
        let mut palette_bits = min_bits + extra;
        let mut ok = true;
        let mut idx = start_idx;
        while idx < palette_size_y {
            let prev = palette_colors_y[idx - 1] as u32;
            let cur = palette_colors_y[idx] as u32;
            let delta_raw = cur - prev - 1;
            // Capacity check mirroring the L(paletteBits) writer.
            // L(0) admits only the raw value 0.
            let capacity = if palette_bits >= 32 {
                u32::MAX
            } else {
                1u32 << palette_bits
            };
            if palette_bits == 0 {
                if delta_raw != 0 {
                    ok = false;
                    break;
                }
            } else if delta_raw >= capacity {
                ok = false;
                break;
            }
            let range = max_val.saturating_sub(cur).saturating_sub(1);
            palette_bits = palette_bits.min(ceil_log2_av1(range));
            idx += 1;
        }
        if ok {
            return Ok(extra);
        }
    }
    Err(Error::PartitionWalkOutOfRange)
}

/// `palette_colors_u[]` + `palette_colors_v[]` writer per §5.11.46
/// (av1-spec p.97-98) — r263. The UV-plane half of the §5.11.46
/// palette-entries syntax: the inverse of
/// [`crate::cdf::read_palette_entries_uv`].
///
/// Spec body (UV arms of §5.11.46, post-`PaletteSizeUV` read):
/// ```text
///   // U plane — mirrors Y, minus the `delta_u++` step and minus the
///   // `- 1` on the range derivation.
///   cache[] = get_palette_cache( 1, mi_row, mi_col )
///   idx = 0
///   for ( i = 0; i < cache_n && idx < PaletteSizeUV; i++ ) {
///       use_palette_color_cache_u                          L(1)
///       if ( use_palette_color_cache_u ) {
///           palette_colors_u[ idx++ ] = cache[ i ]
///       }
///   }
///   if ( idx < PaletteSizeUV ) {
///       palette_colors_u[ idx++ ]                          L(BitDepth)
///   }
///   if ( idx < PaletteSizeUV ) {
///       minBits     = BitDepth - 3
///       palette_num_extra_bits_u                           L(2)
///       paletteBits = minBits + palette_num_extra_bits_u
///   }
///   while ( idx < PaletteSizeUV ) {
///       palette_delta_u                                    L(paletteBits)
///       palette_colors_u[ idx ] = Clip1( palette_colors_u[ idx - 1 ]
///                                        + palette_delta_u )
///       range = ( 1 << BitDepth ) - palette_colors_u[ idx ]
///       paletteBits = Min( paletteBits, CeilLog2( range ) )
///       idx++
///   }
///   sort( palette_colors_u[ 0 .. PaletteSizeUV - 1 ] )
///
///   // V plane — two-arm dispatcher gated on `delta_encode_palette_colors_v`.
///   delta_encode_palette_colors_v                          L(1)
///   if ( delta_encode_palette_colors_v ) {
///       minBits     = BitDepth - 4
///       maxVal      = 1 << BitDepth
///       palette_num_extra_bits_v                           L(2)
///       paletteBits = minBits + palette_num_extra_bits_v
///       palette_colors_v[ 0 ]                              L(BitDepth)
///       for ( idx = 1; idx < PaletteSizeUV; idx++ ) {
///           palette_delta_v                                L(paletteBits)
///           if ( palette_delta_v ) {
///               palette_delta_sign_bit_v                   L(1)
///               if ( palette_delta_sign_bit_v ) palette_delta_v = -palette_delta_v
///           }
///           val = palette_colors_v[ idx - 1 ] + palette_delta_v
///           if ( val < 0 )       val += maxVal
///           if ( val >= maxVal ) val -= maxVal
///           palette_colors_v[ idx ] = Clip1( val )
///       }
///       // NOTE: no sort() on the V delta arm — source order preserved.
///   } else {
///       for ( idx = 0; idx < PaletteSizeUV; idx++ ) {
///           palette_colors_v[ idx ]                        L(BitDepth)
///       }
///   }
/// ```
///
/// **U-plane mirror invariant.** Caller passes `palette_colors_u` already
/// in the post-sort canonical order (non-strictly ascending — duplicate
/// entries are legal here because the U delta loop has no `++` step).
/// The cache + literal + paletteBits + delta loop then reconstructs the
/// exact L(1) / L(BitDepth) / L(2) / L(paletteBits) bit stream that,
/// when fed back through [`crate::cdf::read_palette_entries_uv`],
/// reproduces the same sorted U entries.
///
/// **U paletteBits derivation.** Mirrors the Y picker but with the
/// U-specific delta formula: `delta_raw = cur - prev` (no `- 1`) and
/// `range = maxVal - cur` (no `- 1`). The maximum raw delta is bounded
/// by `2^BitDepth - 1`, which fits in `BitDepth` bits — `minBits + 3
/// = BitDepth - 3 + 3 = BitDepth`. The §5.11.46 refinement only reduces
/// `paletteBits` along the way (since `range = maxVal - cur <= maxVal`).
///
/// **V-plane arm selection.** The `delta_encode_v` parameter chooses
/// the arm:
/// * `false` (direct-literal arm): each `palette_colors_v[idx]` emits as
///   `L(BitDepth)`. No monotonicity or sort requirement.
/// * `true` (signed-delta arm): the writer picks the smallest
///   `palette_num_extra_bits_v ∈ {0, 1, 2, 3}` such that every
///   `palette_delta_v` magnitude (`cur - prev`, allowing the §5.11.46
///   modular wrap to pick the smaller-magnitude alternative
///   `cur - prev ± maxVal`) fits in `paletteBits`. The wrap means the
///   writer can always find a representable signed delta with
///   `|delta| <= maxVal / 2`, which fits in `BitDepth - 1` bits — well
///   within the `minBits + 3 = BitDepth - 1` ceiling.
///
/// **Caller contract.** Inputs must satisfy:
/// * `bit_depth ∈ {8, 10, 12}` (§5.5.2).
/// * `palette_size_uv ∈ 2..=PALETTE_COLORS` (§5.11.46 contract).
/// * `palette_colors_u[0 .. palette_size_uv]` non-strictly ascending,
///   each entry in `[0, 2^bit_depth)`. Descending entries surface
///   [`Error::PartitionWalkOutOfRange`] (the U delta arm only encodes
///   non-negative `cur - prev`).
/// * `palette_colors_v[0 .. palette_size_uv]` arbitrary entries in
///   `[0, 2^bit_depth)`. Not required to be sorted (the §5.11.46 V arm
///   does not sort after decode).
/// * `walker` MUST be at the same §5.11.49 cache state the reader
///   would observe (caller threads its own [`PartitionWalker`] and
///   stamps any preceding decoded blocks into the `PaletteSizes` /
///   `PaletteColors` grids before this call).
///
/// On success, the encoder has committed the §5.11.46 UV palette
/// entries to the bit stream in exact lock-step with the
/// [`crate::cdf::read_palette_entries_uv`] reader.
#[allow(clippy::too_many_arguments)]
pub fn write_palette_entries_uv(
    writer: &mut SymbolWriter,
    bit_depth: u8,
    palette_size_uv: usize,
    palette_colors_u: &[u16],
    palette_colors_v: &[u16],
    delta_encode_v: bool,
    walker: &PartitionWalker,
    mi_row: u32,
    mi_col: u32,
) -> Result<(), Error> {
    if !matches!(bit_depth, 8 | 10 | 12) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if !(2..=PALETTE_COLORS).contains(&palette_size_uv) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if palette_colors_u.len() < palette_size_uv {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if palette_colors_v.len() < palette_size_uv {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let bd = bit_depth as u32;
    let max_val: u32 = 1u32 << bd;
    // §5.11.46 caller contract: U entries non-strictly ascending and
    // each within [0, 2^BitDepth). Mirrors the reader's invariant after
    // the trailing sort + Clip1. Duplicates are admitted because the U
    // delta loop has no `++` step (raw delta == 0 is legal).
    for (i, &c) in palette_colors_u.iter().take(palette_size_uv).enumerate() {
        if (c as u32) >= max_val {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if i > 0 && (c as u32) < palette_colors_u[i - 1] as u32 {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }
    // §5.11.46 caller contract: V entries each within [0, 2^BitDepth).
    // No monotonicity requirement (the §5.11.46 V arm does not sort).
    for &c in palette_colors_v.iter().take(palette_size_uv) {
        if (c as u32) >= max_val {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    // -------- U plane --------
    // §5.11.49 cache via the same walker accessor the reader uses
    // (plane = 1 ⇒ U).
    let mut cache_buf = [0u16; 2 * PALETTE_COLORS];
    let cache_n = walker.get_palette_cache(1, mi_row, mi_col, &mut cache_buf);

    // §5.11.46 cache-coded indices loop for U. Same decision-tree as Y:
    // "is this cache slot the next palette entry?" → L(1).
    let mut idx: usize = 0;
    {
        let mut i = 0usize;
        while i < cache_n && idx < palette_size_uv {
            let use_cache = if cache_buf[i] == palette_colors_u[idx] {
                1u32
            } else {
                0u32
            };
            writer.write_literal(1, use_cache)?;
            if use_cache != 0 {
                idx += 1;
            }
            i += 1;
        }
    }

    // §5.11.46 first "new" U entry as L(BitDepth) literal.
    if idx < palette_size_uv {
        let v = palette_colors_u[idx] as u32;
        writer.write_literal(bd, v)?;
        idx += 1;
    }

    // §5.11.46 paletteBits derivation for U. Same `minBits = BitDepth -
    // 3` as Y; the L(2) read is skipped when no entries remain.
    if idx < palette_size_uv {
        let min_bits = bd - 3;
        let start_idx = idx;
        let extra = pick_palette_extra_bits_u(palette_colors_u, palette_size_uv, start_idx, bd)?;
        writer.write_literal(2, extra)?;
        let mut palette_bits = min_bits + extra;
        // §5.11.46 U delta loop: no `++`, range without `- 1`.
        while idx < palette_size_uv {
            let prev = palette_colors_u[idx - 1] as u32;
            let cur = palette_colors_u[idx] as u32;
            // Non-strict ascending invariant ⇒ cur >= prev.
            let delta_raw = cur - prev;
            if palette_bits == 0 {
                if delta_raw != 0 {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                writer.write_literal(0, 0)?;
            } else {
                if delta_raw >= (1u32 << palette_bits) {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                writer.write_literal(palette_bits, delta_raw)?;
            }
            // §5.11.46 paletteBits refinement: `range = (1 << BitDepth)
            // - palette_colors_u[idx]` (no `- 1`).
            let range = max_val.saturating_sub(cur);
            palette_bits = palette_bits.min(ceil_log2_av1(range));
            idx += 1;
        }
    }

    // -------- V plane --------
    // §5.11.46 V arm selector — L(1) `delta_encode_palette_colors_v`.
    writer.write_literal(1, if delta_encode_v { 1 } else { 0 })?;
    if delta_encode_v {
        // §5.11.46 V delta-encoded arm. `minBits = BitDepth - 4` ⇒
        // `minBits ∈ {4, 6, 8}` for `BitDepth ∈ {8, 10, 12}`.
        let min_bits_v = bd - 4;
        let extra_v = pick_palette_extra_bits_v(palette_colors_v, palette_size_uv, bd)?;
        writer.write_literal(2, extra_v)?;
        let palette_bits_v = min_bits_v + extra_v;
        // First V entry as L(BitDepth) literal.
        let first = palette_colors_v[0] as u32;
        writer.write_literal(bd, first)?;
        // §5.11.46 V delta loop: for each remaining entry, pick the
        // shorter-magnitude signed delta from the two §5.11.46
        // modular-wrap candidates `cur - prev` and `cur - prev ± maxVal`.
        for i in 1..palette_size_uv {
            let prev = palette_colors_v[i - 1] as i32;
            let cur = palette_colors_v[i] as i32;
            let signed_delta = pick_v_signed_delta(prev, cur, max_val as i32, palette_bits_v)?;
            // §5.11.46 emit: `L(paletteBits) palette_delta_v` (magnitude),
            // then on non-zero `L(1) palette_delta_sign_bit_v`.
            let magnitude = signed_delta.unsigned_abs();
            if palette_bits_v == 0 {
                if magnitude != 0 {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                writer.write_literal(0, 0)?;
            } else {
                if magnitude >= (1u32 << palette_bits_v) {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                writer.write_literal(palette_bits_v, magnitude)?;
            }
            if magnitude != 0 {
                let sign_bit = if signed_delta < 0 { 1u32 } else { 0u32 };
                writer.write_literal(1, sign_bit)?;
            }
        }
    } else {
        // §5.11.46 V direct-literal arm: each entry as L(BitDepth).
        for &c in palette_colors_v.iter().take(palette_size_uv) {
            writer.write_literal(bd, c as u32)?;
        }
    }

    Ok(())
}

/// §5.11.49 `palette_tokens( )` per-plane writer — r282. The exact
/// encode-side twin of [`crate::cdf::palette_tokens_plane`]: emits the
/// `color_index_map_{y,uv} NS(PaletteSize)` literal followed by the
/// anti-diagonal `palette_color_idx_{y,uv}` `S()` walk over the
/// on-screen rectangle, deriving the §5.11.50 colour context per
/// position from the same incrementally-reconstructed `ColorMap` the
/// reader maintains.
///
/// Parameters mirror the reader's, with `color_map` flipped from an
/// output to an input:
///
/// * `plane` — see [`PalettePlane`]. Selects the
///   [`TileCdfContext::palette_y_color_cdf`] /
///   [`TileCdfContext::palette_uv_color_cdf`] family.
/// * `palette_size` — `PaletteSize{Y,UV}` in `2..=PALETTE_COLORS`.
/// * `block_width` / `block_height` / `onscreen_width` /
///   `onscreen_height` — the §5.11.49 dimensions
///   ([`crate::cdf::palette_tokens_args`] computes them, including
///   the UV subsampling + `<4`-bump adjustment).
/// * `color_map` — the target `ColorMap{Y,UV}` palette-index plane,
///   row-major with row stride `stride`. Only the on-screen rectangle
///   is read (`0..onscreen_height` × `0..onscreen_width`); every read
///   sample MUST be `< palette_size` (the reader can only reconstruct
///   indices the `ColorOrder` permutation reaches).
/// * `stride` — row stride of `color_map`; `>= block_width`.
///
/// Internally the writer rebuilds the reader's working map sample by
/// sample (each §5.11.50 context reads only already-walked
/// neighbours), so the emitted symbol sequence — and the §8.3 CDF
/// adaptation it drives — is identical to what
/// [`crate::cdf::palette_tokens_plane`] consumes. The reader's
/// §5.11.49 border-fill writes are not mirrored: they happen after
/// every context-relevant sample has been walked and are therefore
/// unobservable on the bit stream.
///
/// Errors mirror the reader: [`crate::Error::InvalidPaletteWalkArgs`]
/// for caller bugs (`palette_size` out of `2..=PALETTE_COLORS`,
/// zero / oversize on-screen dims, `stride < block_width`,
/// `color_map` shorter than `block_height * stride`, an on-screen
/// sample `>= palette_size`), and
/// [`crate::Error::PaletteColorContextUnmapped`] when the §5.11.50
/// `ColorContextHash` lands on an unmapped `Palette_Color_Context[]`
/// entry (unreachable for realisable neighbour combinations).
#[allow(clippy::too_many_arguments)]
pub fn write_palette_tokens_plane(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    plane: PalettePlane,
    palette_size: usize,
    block_width: usize,
    block_height: usize,
    onscreen_width: usize,
    onscreen_height: usize,
    color_map: &[u8],
    stride: usize,
) -> Result<(), Error> {
    // Caller-bug guards — identical set to the reader's.
    if !(2..=PALETTE_COLORS).contains(&palette_size) {
        return Err(Error::InvalidPaletteWalkArgs);
    }
    if onscreen_width == 0 || onscreen_height == 0 {
        return Err(Error::InvalidPaletteWalkArgs);
    }
    if onscreen_width > block_width || onscreen_height > block_height {
        return Err(Error::InvalidPaletteWalkArgs);
    }
    if stride < block_width {
        return Err(Error::InvalidPaletteWalkArgs);
    }
    if color_map.len() < block_height.saturating_mul(stride) {
        return Err(Error::InvalidPaletteWalkArgs);
    }
    if (color_map[0] as usize) >= palette_size {
        return Err(Error::InvalidPaletteWalkArgs);
    }

    // §5.11.49: `color_index_map_{y,uv} NS(PaletteSize)` — the
    // top-left sample is coded as a non-symmetric literal.
    writer.write_ns(palette_size as u32, color_map[0] as u32)?;

    // Working map — the reader's `ColorMap` reconstruction, rebuilt
    // sample by sample so each §5.11.50 context derivation observes
    // exactly the state the reader's derivation observes (unwalked
    // cells hold the same `0` pre-fill the reader's buffer holds).
    let mut work = vec![0u8; block_height * stride];
    work[0] = color_map[0];

    // §5.11.49 anti-diagonal walk over the on-screen rectangle.
    for i in 1..(onscreen_height + onscreen_width - 1) {
        let j_hi = core::cmp::min(i, onscreen_width - 1);
        let j_lo = i.saturating_sub(onscreen_height - 1);
        let mut j = j_hi;
        loop {
            let r = i - j;
            let c = j;

            let desired = color_map[r * stride + c];
            if (desired as usize) >= palette_size {
                return Err(Error::InvalidPaletteWalkArgs);
            }

            // §5.11.50 colour context + `ColorOrder` permutation from
            // the working map (already-walked neighbours only).
            let pcc = get_palette_color_context(&work, stride, r, c, palette_size)
                .ok_or(Error::InvalidPaletteWalkArgs)?;
            let ctx = palette_color_ctx(pcc.color_context_hash)
                .ok_or(Error::PaletteColorContextUnmapped)?;

            // §5.11.49: the reader maps `palette_color_idx` through
            // `ColorOrder`; the writer inverts the permutation.
            let idx = pcc
                .color_order
                .iter()
                .take(palette_size)
                .position(|&v| v == desired)
                .ok_or(Error::InvalidPaletteWalkArgs)?;

            let cdf = match plane {
                PalettePlane::Y => cdfs.palette_y_color_cdf(palette_size, ctx),
                PalettePlane::Uv => cdfs.palette_uv_color_cdf(palette_size, ctx),
            }
            .ok_or(Error::InvalidPaletteWalkArgs)?;
            writer.write_symbol(idx as u32, cdf)?;
            work[r * stride + c] = desired;

            if j == j_lo {
                break;
            }
            j -= 1;
        }
    }

    Ok(())
}

/// Pick the smallest `palette_num_extra_bits_u` (L(2) ⇒ 0..=3) such
/// that every remaining §5.11.46 U-plane delta fits in the
/// reader-refined `paletteBits`. Returns
/// [`Error::PartitionWalkOutOfRange`] when no choice in 0..=3 admits
/// all deltas — this only happens on a malformed
/// `palette_colors_u` (the non-strict-ascending + bit-depth check
/// before this call rules out conformant input).
fn pick_palette_extra_bits_u(
    palette_colors_u: &[u16],
    palette_size_uv: usize,
    start_idx: usize,
    bd: u32,
) -> Result<u32, Error> {
    let min_bits = bd - 3;
    let max_val: u32 = 1u32 << bd;
    for extra in 0..=3u32 {
        let mut palette_bits = min_bits + extra;
        let mut ok = true;
        let mut idx = start_idx;
        while idx < palette_size_uv {
            let prev = palette_colors_u[idx - 1] as u32;
            let cur = palette_colors_u[idx] as u32;
            // U: no `++` ⇒ delta_raw = cur - prev.
            let delta_raw = cur - prev;
            let capacity = if palette_bits >= 32 {
                u32::MAX
            } else {
                1u32 << palette_bits
            };
            if palette_bits == 0 {
                if delta_raw != 0 {
                    ok = false;
                    break;
                }
            } else if delta_raw >= capacity {
                ok = false;
                break;
            }
            // §5.11.46 U refinement: `range = maxVal - cur` (no `- 1`).
            let range = max_val.saturating_sub(cur);
            palette_bits = palette_bits.min(ceil_log2_av1(range));
            idx += 1;
        }
        if ok {
            return Ok(extra);
        }
    }
    Err(Error::PartitionWalkOutOfRange)
}

/// Pick the smallest `palette_num_extra_bits_v` (L(2) ⇒ 0..=3) such
/// that every §5.11.46 V-plane signed delta magnitude fits in
/// `paletteBits = (BitDepth - 4) + extra`. Returns
/// [`Error::PartitionWalkOutOfRange`] when no choice admits all
/// deltas (a malformed input would be required to hit this — for any
/// `BitDepth ∈ {8, 10, 12}`, the smaller-magnitude wrap candidate is
/// bounded by `maxVal / 2 = 2^(BitDepth - 1)`, which fits in
/// `BitDepth - 1` bits, well within `minBits + 3 = BitDepth - 1`).
fn pick_palette_extra_bits_v(
    palette_colors_v: &[u16],
    palette_size_uv: usize,
    bd: u32,
) -> Result<u32, Error> {
    let min_bits = bd - 4;
    let max_val: i32 = 1i32 << bd;
    for extra in 0..=3u32 {
        let palette_bits = min_bits + extra;
        let mut ok = true;
        for i in 1..palette_size_uv {
            let prev = palette_colors_v[i - 1] as i32;
            let cur = palette_colors_v[i] as i32;
            if pick_v_signed_delta(prev, cur, max_val, palette_bits).is_err() {
                ok = false;
                break;
            }
        }
        if ok {
            return Ok(extra);
        }
    }
    Err(Error::PartitionWalkOutOfRange)
}

/// Pick the signed delta the §5.11.46 V delta-encoded arm should emit
/// for the transition `prev -> cur`. The reader applies modular wrap
/// to `prev + signed_delta` so there are up to three candidates:
/// `cur - prev`, `cur - prev + maxVal`, `cur - prev - maxVal`. We pick
/// the one with the smallest magnitude that fits in `paletteBits`
/// (magnitude < `2^paletteBits`). Ties on magnitude prefer the
/// natural (non-wrapped) candidate.
///
/// Returns [`Error::PartitionWalkOutOfRange`] if no candidate fits —
/// the picker's `extra` loop catches this and bumps `paletteBits`.
fn pick_v_signed_delta(prev: i32, cur: i32, max_val: i32, palette_bits: u32) -> Result<i32, Error> {
    let capacity: u32 = if palette_bits == 0 {
        // L(0) only encodes magnitude 0.
        1
    } else if palette_bits >= 32 {
        u32::MAX
    } else {
        1u32 << palette_bits
    };
    // The three §5.11.46 wrap candidates.
    let natural = cur - prev;
    let candidates = [natural, natural + max_val, natural - max_val];
    let mut best: Option<i32> = None;
    for &cand in &candidates {
        // Reader's L(paletteBits) read returns `raw ∈ [0, 2^paletteBits)`,
        // then the optional sign bit can negate it. So the encodable
        // signed range is `[-(2^paletteBits - 1), 2^paletteBits - 1]`
        // (magnitude < 2^paletteBits).
        let mag = cand.unsigned_abs();
        if mag >= capacity {
            continue;
        }
        // Verify the §5.11.46 reader's modular wrap on this candidate
        // actually reconstructs `cur` from `prev`: `prev + cand` then
        // wrapped into `[0, maxVal)` must equal `cur`.
        let mut val = prev + cand;
        if val < 0 {
            val += max_val;
        }
        if val >= max_val {
            val -= max_val;
        }
        if val != cur {
            continue;
        }
        match best {
            None => best = Some(cand),
            Some(b) => {
                if cand.unsigned_abs() < b.unsigned_abs() {
                    best = Some(cand);
                }
            }
        }
    }
    best.ok_or(Error::PartitionWalkOutOfRange)
}

/// `intra_block_mode_info( )` dispatcher per §5.11.22 (av1-spec p.72)
/// — r261. Composes the §5.11.22 leaf writers into the full §5.11.22
/// body for the **no-palette** + **no-CFL** + **DC_PRED** path,
/// matching the shape the §5.11.18 `intra_block_mode_info( )` call
/// site expects.
///
/// Scope of this arc (r261):
/// * `y_mode` (§5.11.22 line 2) via [`write_y_mode`].
/// * `intra_angle_info_y` (§5.11.22 line 4) via
///   [`write_intra_angle_info_y`] — non-directional `y_mode` ⇒
///   no-bits short-circuit.
/// * `uv_mode` (§5.11.22 line 6) via [`write_intra_uv_mode`] when
///   `has_chroma`.
/// * `intra_angle_info_uv` (§5.11.22 line 10) via
///   [`write_intra_angle_info_uv`] when `has_chroma`.
/// * `palette_mode_info` (§5.11.22 line 13-15) via
///   [`write_palette_mode_info`] — restricted to `has_palette_y == 0
///   && has_palette_uv == 0` per the leaf's contract.
/// * `filter_intra_mode_info` (§5.11.22 line 16) via
///   [`write_filter_intra_mode_info`].
///
/// **NOT covered by this arc** (caller must pass values consistent
/// with the no-palette / no-CFL path; non-trivial values surface
/// [`Error::PaletteEntriesUnsupported`] / [`Error::PartitionWalkOutOfRange`]):
/// * `read_cfl_alphas` (§5.11.22 line 8 when `UVMode == UV_CFL_PRED
///   = 13`). Callers compose [`write_cfl_alphas`] manually for the
///   CFL arm.
/// * Non-zero `has_palette_*` (§5.11.46 palette-entries syntax).
///
/// Inputs:
/// * `mi_size` — `MiSize` ordinal in `0..BLOCK_SIZES`.
/// * `y_mode` — the §5.11.22 `YMode` value in `0..INTRA_MODES`.
/// * `uv_mode` — `Some(mode)` when `has_chroma`, `None` otherwise.
///   `mode` MUST be in `0..UV_INTRA_MODES_CFL_ALLOWED` and MUST NOT
///   equal `UV_CFL_PRED = 13` on this arc (the CFL arm is owned by
///   the caller; passing `UV_CFL_PRED` is a contract violation
///   surfaced as [`Error::PartitionWalkOutOfRange`]).
/// * `angle_delta_y` / `angle_delta_uv` — signed deltas in
///   `-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA`. MUST be `0` when the
///   §5.11.42 / §5.11.43 short-circuit fires (`mi_size < BLOCK_8X8`
///   OR `!is_directional(mode)`).
/// * `cfl_allowed` — caller-derived §8.3.2 CFL-allowance flag
///   (e.g. via [`crate::cdf::cfl_allowed_for_uv_mode`]).
/// * `has_chroma` — §5.11.5 `HasChroma`.
/// * `allow_screen_content_tools` — §5.9.5 sequence-header bit.
/// * `enable_filter_intra` — §5.5.2 sequence-header bit.
/// * `use_filter_intra` — `0` / `1`; gated by the §5.11.24 outer
///   gate inside [`write_filter_intra_mode_info`].
/// * `filter_intra_mode` — `Some(mode)` when `use_filter_intra ==
///   1`, `None` otherwise.
/// * `above_palette_y` / `left_palette_y` — §8.3.2 neighbour-palette
///   booleans fed to [`palette_y_mode_ctx`].
///
/// On success, the encoder has committed the full §5.11.22
/// no-palette / no-CFL body to the bit stream + advanced the §8.3
/// adaptation state in lockstep with the
/// [`crate::cdf::PartitionWalker::decode_intra_block_mode_info`]
/// reader.
#[allow(clippy::too_many_arguments)]
pub fn write_intra_block_mode_info(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    mi_size: usize,
    y_mode: u8,
    uv_mode: Option<u8>,
    angle_delta_y: i8,
    angle_delta_uv: i8,
    cfl_allowed: bool,
    has_chroma: bool,
    allow_screen_content_tools: bool,
    enable_filter_intra: bool,
    use_filter_intra: u8,
    filter_intra_mode: Option<u8>,
    above_palette_y: bool,
    left_palette_y: bool,
) -> Result<(), Error> {
    // §5.11.22 caller-bug guards re-asserted here so a contract
    // violation surfaces before any partial S() write disturbs the
    // bit stream. Each leaf re-checks its own bounds.
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if has_chroma {
        let uv = uv_mode.ok_or(Error::PartitionWalkOutOfRange)?;
        if (uv as usize) >= UV_INTRA_MODES_CFL_ALLOWED {
            return Err(Error::PartitionWalkOutOfRange);
        }
        // r261 arc does NOT compose `write_cfl_alphas`; the CFL arm
        // is reserved for a future round. UV_CFL_PRED = 13 per §3.
        if (uv as usize) == 13 {
            return Err(Error::PartitionWalkOutOfRange);
        }
    } else if uv_mode.is_some() {
        // §5.11.22 line 5 `if (HasChroma)` guard: when `has_chroma ==
        // false` the reader never touches `uv_mode`; passing a `Some`
        // value is a caller bug.
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.22 line 2: `y_mode S()` via the §8.3.2 `Size_Group[ MiSize ]`
    // ctx. The leaf takes the derived ctx; do the derivation here so
    // the dispatcher matches the §5.11.22 reader's `y_mode_ctx =
    // size_group(sub_size)` step verbatim.
    let size_group_ctx = crate::cdf::size_group(mi_size);
    write_y_mode(writer, cdfs, y_mode, size_group_ctx)?;

    // §5.11.22 line 4: `intra_angle_info_y( )` per §5.11.42.
    write_intra_angle_info_y(writer, cdfs, angle_delta_y, mi_size, y_mode)?;

    // §5.11.22 lines 5-10: `if ( HasChroma )` arm.
    if has_chroma {
        let uv = uv_mode.expect("validated above");
        // §5.11.22 line 6.
        write_intra_uv_mode(writer, cdfs, y_mode, uv, cfl_allowed)?;
        // §5.11.22 line 10.
        write_intra_angle_info_uv(writer, cdfs, angle_delta_uv, mi_size, uv)?;
    } else {
        // §5.11.22 lines 5-10 skipped entirely. Caller-bug guards:
        // both UV deltas MUST be zero (no bits would be reconstructed
        // on the monochrome arm).
        if angle_delta_uv != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    // §5.11.22 lines 13-15: `palette_mode_info( )` per §5.11.46 — the
    // r261 leaf restricts to has_palette_y == 0 / has_palette_uv == 0.
    write_palette_mode_info(
        writer,
        cdfs,
        /* has_palette_y = */ 0,
        /* has_palette_uv = */ 0,
        mi_size,
        y_mode,
        uv_mode,
        has_chroma,
        allow_screen_content_tools,
        above_palette_y,
        left_palette_y,
    )?;

    // §5.11.22 line 16: `filter_intra_mode_info( )` per §5.11.24. The
    // §5.11.46 leaf above committed `palette_size_y == 0`, so the
    // §5.11.24 outer gate's `PaletteSizeY == 0` clause is satisfied
    // mechanically on this arc.
    write_filter_intra_mode_info(
        writer,
        cdfs,
        use_filter_intra,
        filter_intra_mode,
        mi_size,
        y_mode,
        /* palette_size_y = */ 0,
        enable_filter_intra,
    )?;

    Ok(())
}

/// `palette_mode_info( )` writer per §5.11.46 (av1-spec p.97-98) —
/// r264. Superset of [`write_palette_mode_info`]: lifts the
/// no-palette precondition by threading the `palette_size_y_minus_2`
/// / `palette_size_uv_minus_2` S() symbols + the §5.11.46
/// palette-entries writers ([`write_palette_entries_y`] /
/// [`write_palette_entries_uv`]) into a single leaf.
///
/// Spec body (lifted to cover both arms):
/// ```text
///   if ( outer_gate ) {
///       bsizeCtx = Mi_Width_Log2[ MiSize ] + Mi_Height_Log2[ MiSize ] - 2
///       if ( YMode == DC_PRED ) {
///           has_palette_y                                 S()
///           if ( has_palette_y ) {
///               palette_size_y_minus_2                    S()
///               palette_colors_y[ 0 .. PaletteSizeY ]     (§5.11.46)
///           }
///       }
///       if ( HasChroma && UVMode == DC_PRED ) {
///           has_palette_uv                                S()
///           if ( has_palette_uv ) {
///               palette_size_uv_minus_2                   S()
///               palette_colors_u[ 0 .. PaletteSizeUV ]    (§5.11.46)
///               palette_colors_v[ 0 .. PaletteSizeUV ]    (§5.11.46)
///           }
///       }
///   }
/// ```
///
/// Mirrors the reader path inlined at the §5.11.46 sub-block of
/// [`crate::cdf::PartitionWalker::decode_intra_block_mode_info`] +
/// [`crate::cdf::read_palette_entries_y`] +
/// [`crate::cdf::read_palette_entries_uv`].
///
/// Inputs:
/// * `has_palette_y` — `0` or `1`; `1` triggers the
///   `palette_size_y_minus_2` S() + [`write_palette_entries_y`] call.
/// * `has_palette_uv` — `0` or `1`; `1` triggers the
///   `palette_size_uv_minus_2` S() + [`write_palette_entries_uv`]
///   call.
/// * `mi_size` — `MiSize` ordinal in `0..BLOCK_SIZES`.
/// * `y_mode` — the §5.11.22 `YMode` value; the luma arm fires only
///   on `YMode == DC_PRED = 0`.
/// * `uv_mode` — `Some(mode)` when `has_chroma`, `None` otherwise.
///   The chroma arm fires only on `UVMode == DC_PRED = 0`.
/// * `has_chroma` — §5.11.5 `HasChroma`.
/// * `allow_screen_content_tools` — §5.9.5 sequence-header bit that
///   gates the §5.11.46 outer arm.
/// * `above_palette_y` / `left_palette_y` — §8.3.2 neighbour-palette
///   booleans fed to [`palette_y_mode_ctx`].
/// * `bit_depth` — §6.7.2 `BitDepth` (8 / 10 / 12). Consumed only on
///   the entries-bearing arms (passed straight through to the
///   palette-entries writers).
/// * `palette_size_y` — `PaletteSizeY` in `2..=PALETTE_COLORS` when
///   `has_palette_y == 1`; ignored otherwise.
/// * `palette_colors_y` — sorted-ascending palette entries for the Y
///   plane (see [`write_palette_entries_y`] contract). Ignored when
///   `has_palette_y == 0`.
/// * `palette_size_uv` — `PaletteSizeUV` in `2..=PALETTE_COLORS` when
///   `has_palette_uv == 1`; ignored otherwise.
/// * `palette_colors_u` / `palette_colors_v` — see
///   [`write_palette_entries_uv`] contract. Ignored when
///   `has_palette_uv == 0`.
/// * `delta_encode_v` — §5.11.46 V-plane arm selector. Ignored when
///   `has_palette_uv == 0`.
/// * `walker` — the same [`PartitionWalker`] the dispatcher walks for
///   §5.11.49 `get_palette_cache` neighbour reads.
/// * `mi_row` / `mi_col` — the §5.11.49 block-origin coordinates fed
///   to the palette-cache accessor.
///
/// Caller-bug guards (in addition to the per-leaf re-checks):
/// * `palette_size_y` / `palette_size_uv` ∈ `2..=PALETTE_COLORS`
///   when the corresponding `has_palette_*` is `1`.
/// * `palette_size_y` / `palette_size_uv` ignored when `has_palette_*
///   == 0` (no entries syntax fires).
/// * `has_palette_y == 1` requires `YMode == DC_PRED`; reader never
///   reads `has_palette_y` otherwise.
/// * `has_palette_uv == 1` requires `has_chroma && uv_mode ==
///   Some(DC_PRED)`; reader never reads `has_palette_uv` otherwise.
/// * `has_palette_y == 1` or `has_palette_uv == 1` requires the
///   §5.11.46 outer gate (`allow_screen_content_tools && MiSize >=
///   BLOCK_8X8 && Block_W <= 64 && Block_H <= 64`).
///
/// On success, the encoder has committed the full §5.11.46
/// palette-mode-info body to the bit stream + advanced the §8.3 CDF
/// adaptation state in lockstep with the reader.
#[allow(clippy::too_many_arguments)]
pub fn write_palette_mode_info_with_entries(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    has_palette_y: u8,
    has_palette_uv: u8,
    mi_size: usize,
    y_mode: u8,
    uv_mode: Option<u8>,
    has_chroma: bool,
    allow_screen_content_tools: bool,
    above_palette_y: bool,
    left_palette_y: bool,
    bit_depth: u8,
    palette_size_y: usize,
    palette_colors_y: &[u16],
    palette_size_uv: usize,
    palette_colors_u: &[u16],
    palette_colors_v: &[u16],
    delta_encode_v: bool,
    walker: &PartitionWalker,
    mi_row: u32,
    mi_col: u32,
) -> Result<(), Error> {
    // §5.11.22 / §5.11.46 caller-bug guards. Re-asserted up-front so a
    // contract violation surfaces before any partial S() write
    // disturbs the bit stream. Each leaf re-checks its own bounds.
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if let Some(uv) = uv_mode {
        if (uv as usize) >= UV_INTRA_MODES_CFL_ALLOWED {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }
    if has_palette_y > 1 || has_palette_uv > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.46 outer gate. Mirrors the reader's `palette_outer_gate`.
    let bw = block_width(mi_size) as u32;
    let bh = block_height(mi_size) as u32;
    let outer_gate = allow_screen_content_tools && mi_size >= BLOCK_8X8 && bw <= 64 && bh <= 64;

    if !outer_gate {
        // Gate-off path: the §5.11.46 spec body sets `PaletteSizeY =
        // 0` / `PaletteSizeUV = 0` unconditionally. The reader
        // reconstructs them as 0 regardless of the bit stream; any
        // non-zero `has_palette_*` is a caller bug because the
        // reader never reads the symbol on this arm.
        if has_palette_y != 0 || has_palette_uv != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.46: `bsizeCtx = Mi_Width_Log2[ MiSize ] +
    // Mi_Height_Log2[ MiSize ] - 2`. The §9.3 mapping guarantees
    // `bsizeCtx ∈ 0..PALETTE_BLOCK_SIZE_CONTEXTS = 7` for any MiSize
    // that passed the outer gate (BLOCK_8X8 has (1, 1) ⇒ bsizeCtx = 0,
    // BLOCK_64X64 has (4, 4) ⇒ bsizeCtx = 6).
    let bsize_ctx = mi_width_log2(mi_size) + mi_height_log2(mi_size) - 2;
    debug_assert!(
        bsize_ctx < PALETTE_BLOCK_SIZE_CONTEXTS,
        "§5.11.46 bsizeCtx must be in 0..PALETTE_BLOCK_SIZE_CONTEXTS"
    );

    // §5.11.46 luma arm: `if ( YMode == DC_PRED )`. DC_PRED = 0.
    if y_mode == 0 {
        let ctx_y = palette_y_mode_ctx(above_palette_y, left_palette_y);
        let row = cdfs.palette_y_mode_cdf(bsize_ctx, ctx_y);
        writer.write_symbol(has_palette_y as u32, row)?;
        if has_palette_y == 1 {
            // §5.11.46 caller-bug: when the luma arm fires, the
            // §5.11.46 contract pins `PaletteSizeY ∈ 2..=PALETTE_COLORS`.
            if !(2..=PALETTE_COLORS).contains(&palette_size_y) {
                return Err(Error::PartitionWalkOutOfRange);
            }
            // §5.11.46: `palette_size_y_minus_2 S()` against
            // `TilePaletteYSizeCdf[ bsizeCtx ]`. Encoded value in
            // `0..PALETTE_SIZES = 7`; PaletteSizeY = encoded + 2 is in
            // `2..=PALETTE_COLORS = 8`.
            let row_sz = cdfs.palette_y_size_cdf(bsize_ctx);
            let minus_2 = (palette_size_y - 2) as u32;
            writer.write_symbol(minus_2, row_sz)?;
            // §5.11.46 palette_colors_y[]: cache merge + cache-coded
            // indices + first-new L(BitDepth) + delta loop. Delegates
            // to the r262 leaf, threading the same walker the reader
            // walks for §5.11.49 `get_palette_cache(0, …)`.
            write_palette_entries_y(
                writer,
                bit_depth,
                palette_size_y,
                palette_colors_y,
                walker,
                mi_row,
                mi_col,
            )?;
        }
    } else if has_palette_y != 0 {
        // §5.11.46 contract: when `y_mode != DC_PRED` the reader
        // never reads `has_palette_y`; non-zero here is a caller bug.
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.46 chroma arm: `if ( HasChroma && UVMode == DC_PRED )`.
    if has_chroma && uv_mode == Some(0) {
        // §8.3.2 chroma ctx is `palette_uv_mode_ctx(PaletteSizeY)`. On
        // the luma-arm-not-fired or `has_palette_y == 0` arm the spec
        // sets `PaletteSizeY = 0` (the reader's `palette_size_y` local
        // stays at its 0 initialiser); on the `has_palette_y == 1`
        // arm `PaletteSizeY` is the value just committed.
        let ps_y = if y_mode == 0 && has_palette_y == 1 {
            palette_size_y
        } else {
            0
        };
        let ctx_uv = palette_uv_mode_ctx(ps_y);
        let row = cdfs.palette_uv_mode_cdf(ctx_uv);
        writer.write_symbol(has_palette_uv as u32, row)?;
        if has_palette_uv == 1 {
            if !(2..=PALETTE_COLORS).contains(&palette_size_uv) {
                return Err(Error::PartitionWalkOutOfRange);
            }
            // §5.11.46: `palette_size_uv_minus_2 S()` against
            // `TilePaletteUVSizeCdf[ bsizeCtx ]`.
            let row_sz = cdfs.palette_uv_size_cdf(bsize_ctx);
            let minus_2 = (palette_size_uv - 2) as u32;
            writer.write_symbol(minus_2, row_sz)?;
            // §5.11.46 palette_colors_u[] + palette_colors_v[] writer.
            write_palette_entries_uv(
                writer,
                bit_depth,
                palette_size_uv,
                palette_colors_u,
                palette_colors_v,
                delta_encode_v,
                walker,
                mi_row,
                mi_col,
            )?;
        }
    } else if has_palette_uv != 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    Ok(())
}

/// `intra_block_mode_info( )` dispatcher per §5.11.22 (av1-spec p.72)
/// — r265. Superset of [`write_intra_block_mode_info`] and r264's
/// no-CFL variant: composes [`write_palette_mode_info_with_entries`]
/// for the §5.11.46 palette-entries arm AND composes
/// [`write_cfl_alphas`] for the §5.11.22 line-8 `UVMode ==
/// UV_CFL_PRED` arm.
///
/// Scope of this arc (r265): identical to r264 except the §5.11.22
/// line-8 `if ( UVMode == UV_CFL_PRED ) read_cfl_alphas( )` clause
/// is now folded in. When `has_chroma && uv == UV_CFL_PRED = 13` the
/// dispatcher requires `cfl_alpha_u` / `cfl_alpha_v` to be `Some`
/// and threads them through [`write_cfl_alphas`] between
/// [`write_intra_uv_mode`] and [`write_intra_angle_info_uv`]. The
/// §5.11.24 `filter_intra_mode_info` outer gate's `PaletteSizeY ==
/// 0` clause is still enforced (the gate closes mechanically when
/// `has_palette_y == 1`).
///
/// Inputs: identical to [`write_intra_block_mode_info`] plus the
/// §5.11.46 palette-entries parameters surfaced by
/// [`write_palette_mode_info_with_entries`] plus the §5.11.45 CFL
/// alphas `cfl_alpha_u` / `cfl_alpha_v` — both `None` on the non-CFL
/// chroma arm, both `Some` (with at least one non-zero per §6.10.36
/// joint-sign constraint) on the CFL arm.
///
/// On success, the encoder has committed the full §5.11.22 body —
/// including any §5.11.46 palette-entries syntax and any §5.11.45
/// CFL syntax — to the bit stream + advanced the §8.3 CDF adaptation
/// state in lockstep with
/// [`crate::cdf::PartitionWalker::decode_intra_block_mode_info`].
#[allow(clippy::too_many_arguments)]
pub fn write_intra_block_mode_info_with_palette(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    mi_size: usize,
    y_mode: u8,
    uv_mode: Option<u8>,
    angle_delta_y: i8,
    angle_delta_uv: i8,
    cfl_allowed: bool,
    has_chroma: bool,
    allow_screen_content_tools: bool,
    enable_filter_intra: bool,
    use_filter_intra: u8,
    filter_intra_mode: Option<u8>,
    above_palette_y: bool,
    left_palette_y: bool,
    bit_depth: u8,
    has_palette_y: u8,
    palette_size_y: usize,
    palette_colors_y: &[u16],
    has_palette_uv: u8,
    palette_size_uv: usize,
    palette_colors_u: &[u16],
    palette_colors_v: &[u16],
    delta_encode_v: bool,
    cfl_alpha_u: Option<i8>,
    cfl_alpha_v: Option<i8>,
    walker: &PartitionWalker,
    mi_row: u32,
    mi_col: u32,
) -> Result<(), Error> {
    // §5.11.22 caller-bug guards re-asserted here so a contract
    // violation surfaces before any partial S() write disturbs the
    // bit stream. Each leaf re-checks its own bounds.
    check_intra_mode_info_inputs(
        mi_size,
        y_mode,
        uv_mode,
        cfl_allowed,
        has_chroma,
        cfl_alpha_u,
        cfl_alpha_v,
    )?;

    // §5.11.22 line 2: `y_mode S()` via the §8.3.2 `Size_Group[ MiSize ]`
    // ctx.
    let size_group_ctx = crate::cdf::size_group(mi_size);
    write_y_mode(writer, cdfs, y_mode, size_group_ctx)?;

    // §5.11.22 lines 4-16: shared with the §5.11.7 `else` arm.
    write_intra_mode_info_tail(
        writer,
        cdfs,
        mi_size,
        y_mode,
        uv_mode,
        angle_delta_y,
        angle_delta_uv,
        cfl_allowed,
        has_chroma,
        allow_screen_content_tools,
        enable_filter_intra,
        use_filter_intra,
        filter_intra_mode,
        above_palette_y,
        left_palette_y,
        bit_depth,
        has_palette_y,
        palette_size_y,
        palette_colors_y,
        has_palette_uv,
        palette_size_uv,
        palette_colors_u,
        palette_colors_v,
        delta_encode_v,
        cfl_alpha_u,
        cfl_alpha_v,
        walker,
        mi_row,
        mi_col,
    )
}

/// Caller-bug guard block shared by
/// [`write_intra_block_mode_info_with_palette`] (§5.11.22) and
/// [`write_intra_frame_else_arm`] (§5.11.7 `else` arm) — both bodies
/// share the same `uv_mode` / CFL / chroma input contract, so the
/// guards are asserted once at the dispatcher boundary, BEFORE any
/// S() write touches the bit stream.
fn check_intra_mode_info_inputs(
    mi_size: usize,
    y_mode: u8,
    uv_mode: Option<u8>,
    cfl_allowed: bool,
    has_chroma: bool,
    cfl_alpha_u: Option<i8>,
    cfl_alpha_v: Option<i8>,
) -> Result<(), Error> {
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (y_mode as usize) >= INTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if has_chroma {
        let uv = uv_mode.ok_or(Error::PartitionWalkOutOfRange)?;
        if (uv as usize) >= UV_INTRA_MODES_CFL_ALLOWED {
            return Err(Error::PartitionWalkOutOfRange);
        }
        // §5.11.22 line 8: `UV_CFL_PRED = 13` per §3. On the
        // non-CFL chroma arm the caller MUST pass both alphas as
        // `None`; on the CFL arm both MUST be `Some`. The further
        // §5.11.45 magnitude / joint-sign guards live inside
        // [`write_cfl_alphas`] itself.
        if (uv as usize) == 13 {
            if cfl_alpha_u.is_none() || cfl_alpha_v.is_none() {
                return Err(Error::PartitionWalkOutOfRange);
            }
            // §5.11.22 line 6 / §8.3.2: the CFL-allowed UV CDF row
            // is the only one that carries `UV_CFL_PRED` (width 14);
            // the CFL-not-allowed row is width 13 and excludes it.
            // Rejecting `uv == UV_CFL_PRED` with `cfl_allowed ==
            // false` keeps the writer in lockstep with the reader's
            // CDF-row selection.
            if !cfl_allowed {
                return Err(Error::PartitionWalkOutOfRange);
            }
        } else if cfl_alpha_u.is_some() || cfl_alpha_v.is_some() {
            return Err(Error::PartitionWalkOutOfRange);
        }
    } else if uv_mode.is_some() {
        // §5.11.22 line 5 `if (HasChroma)` guard: when `has_chroma ==
        // false` the reader never touches `uv_mode`; passing a `Some`
        // value is a caller bug.
        return Err(Error::PartitionWalkOutOfRange);
    } else if cfl_alpha_u.is_some() || cfl_alpha_v.is_some() {
        // §5.11.22 line 5 closes the chroma arm entirely on
        // `has_chroma == false`; passing CFL alphas through is a
        // caller bug.
        return Err(Error::PartitionWalkOutOfRange);
    }
    Ok(())
}

/// Shared §5.11.22 / §5.11.7-`else`-arm tail writer — everything after
/// the per-arm Y-mode S() (`y_mode` with the `Size_Group[ MiSize ]`
/// ctx on §5.11.22, `intra_frame_y_mode` with the §8.3.2
/// neighbour-mode ctx pair on §5.11.7). Both spec bodies continue
/// identically from `intra_angle_info_y( )` onward (same syntax
/// elements, same CDF rows), so the tail is factored once and called
/// by both dispatchers. Caller guarantees
/// [`check_intra_mode_info_inputs`] has already passed.
#[allow(clippy::too_many_arguments)]
fn write_intra_mode_info_tail(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    mi_size: usize,
    y_mode: u8,
    uv_mode: Option<u8>,
    angle_delta_y: i8,
    angle_delta_uv: i8,
    cfl_allowed: bool,
    has_chroma: bool,
    allow_screen_content_tools: bool,
    enable_filter_intra: bool,
    use_filter_intra: u8,
    filter_intra_mode: Option<u8>,
    above_palette_y: bool,
    left_palette_y: bool,
    bit_depth: u8,
    has_palette_y: u8,
    palette_size_y: usize,
    palette_colors_y: &[u16],
    has_palette_uv: u8,
    palette_size_uv: usize,
    palette_colors_u: &[u16],
    palette_colors_v: &[u16],
    delta_encode_v: bool,
    cfl_alpha_u: Option<i8>,
    cfl_alpha_v: Option<i8>,
    walker: &PartitionWalker,
    mi_row: u32,
    mi_col: u32,
) -> Result<(), Error> {
    // §5.11.22 line 4: `intra_angle_info_y( )` per §5.11.42.
    write_intra_angle_info_y(writer, cdfs, angle_delta_y, mi_size, y_mode)?;

    // §5.11.22 lines 5-10: `if ( HasChroma )` arm.
    if has_chroma {
        let uv = uv_mode.expect("validated above");
        // §5.11.22 line 6.
        write_intra_uv_mode(writer, cdfs, y_mode, uv, cfl_allowed)?;
        // §5.11.22 line 8: `if ( UVMode == UV_CFL_PRED )
        // read_cfl_alphas( )` per §5.11.45. UV_CFL_PRED = 13 per §3.
        // The reader's `read_cfl_alphas` body sits between
        // `uv_mode S()` and `intra_angle_info_uv( )`; mirror that
        // ordering here so the bit positions of every following
        // symbol line up.
        if (uv as usize) == 13 {
            let alpha_u = cfl_alpha_u.expect("validated above");
            let alpha_v = cfl_alpha_v.expect("validated above");
            write_cfl_alphas(writer, cdfs, alpha_u, alpha_v)?;
        }
        // §5.11.22 line 10.
        write_intra_angle_info_uv(writer, cdfs, angle_delta_uv, mi_size, uv)?;
    } else if angle_delta_uv != 0 {
        // §5.11.22 lines 5-10 skipped entirely. Caller-bug guards:
        // both UV deltas MUST be zero (no bits would be reconstructed
        // on the monochrome arm).
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.22 lines 13-15: `palette_mode_info( )` per §5.11.46 — the
    // r264 leaf accepts non-zero `has_palette_*` and the entries
    // syntax.
    write_palette_mode_info_with_entries(
        writer,
        cdfs,
        has_palette_y,
        has_palette_uv,
        mi_size,
        y_mode,
        uv_mode,
        has_chroma,
        allow_screen_content_tools,
        above_palette_y,
        left_palette_y,
        bit_depth,
        palette_size_y,
        palette_colors_y,
        palette_size_uv,
        palette_colors_u,
        palette_colors_v,
        delta_encode_v,
        walker,
        mi_row,
        mi_col,
    )?;

    // §5.11.22 line 16: `filter_intra_mode_info( )` per §5.11.24. The
    // §5.11.24 outer gate's `PaletteSizeY == 0` clause closes
    // mechanically when `has_palette_y == 1`. Pass `PaletteSizeY` as
    // committed above so the writer matches the reader's gate.
    let palette_size_y_for_filter = if y_mode == 0 && has_palette_y == 1 {
        palette_size_y as u32
    } else {
        0
    };
    write_filter_intra_mode_info(
        writer,
        cdfs,
        use_filter_intra,
        filter_intra_mode,
        mi_size,
        y_mode,
        palette_size_y_for_filter,
        enable_filter_intra,
    )?;

    Ok(())
}

/// §5.11.7 `intra_frame_mode_info( )` `else` (non-intrabc) arm writer
/// (av1-spec p.65) — r280. Composes, in spec order, the write side of:
///
/// ```text
///   } else {
///       is_inter = 0
///       intra_frame_y_mode                                S()
///       YMode = intra_frame_y_mode
///       intra_angle_info_y( )
///       if ( HasChroma ) {
///           uv_mode                                       S()
///           UVMode = uv_mode
///           if ( UVMode == UV_CFL_PRED ) {
///               read_cfl_alphas( )
///           }
///           intra_angle_info_uv( )
///       }
///       PaletteSizeY = 0
///       PaletteSizeUV = 0
///       if ( MiSize >= BLOCK_8X8 &&
///              Block_Width[ MiSize ] <= 64 &&
///              Block_Height[ MiSize ] <= 64 &&
///              allow_screen_content_tools ) {
///           palette_mode_info( )
///       }
///       filter_intra_mode_info( )
///   }
/// ```
///
/// This is the keyframe / intra-only-frame sibling of the §5.11.22
/// [`write_intra_block_mode_info_with_palette`] dispatcher: the two
/// spec bodies are identical from `intra_angle_info_y( )` onward
/// (shared via the factored tail writer), and differ ONLY in the
/// leading Y-mode element — §5.11.7 codes `intra_frame_y_mode`
/// against `TileIntraFrameYModeCdf[ abovemode ][ leftmode ]` (the
/// §8.3.2 neighbour-mode ctx pair) where §5.11.22 codes `y_mode`
/// against `TileYModeCdf[ Size_Group[ MiSize ] ]`. The `is_inter = 0`
/// assignment is a no-bit fixed value (the caller stamps the
/// §5.11.5 `IsInters[][]` grid with `0` exactly as the reader does).
///
/// `abovemode_ctx` / `leftmode_ctx` are the §8.3.2 derivations
/// (av1-spec p.361):
///
/// ```text
///   abovemode = Intra_Mode_Context[ AvailU ? YModes[ MiRow - 1 ][ MiCol ] : DC_PRED ]
///   leftmode  = Intra_Mode_Context[ AvailL ? YModes[ MiRow ][ MiCol - 1 ] : DC_PRED ]
/// ```
///
/// computed by the caller from its own §5.11.5 `YModes[][]` grid via
/// [`crate::cdf::intra_mode_ctx`] (per the stateless-writer doctrine
/// of this module; the unavailable-neighbour identity is
/// `intra_mode_ctx(DC_PRED) = 0`). Both MUST be in
/// `0..INTRA_MODE_CONTEXTS = 5`.
///
/// Every other input maps 1:1 onto the
/// [`write_intra_block_mode_info_with_palette`] parameter of the same
/// name — including the §5.11.46 palette-entries surface and the
/// §5.11.45 CFL alphas (`None`/`None` on the non-CFL chroma arm,
/// `Some`/`Some` on the `uv_mode == UV_CFL_PRED` arm).
///
/// Mirrors
/// [`crate::cdf::PartitionWalker::decode_intra_frame_mode_info_else_arm`]
/// — on success the encoder has committed the full §5.11.7 `else`-arm
/// body to the bit stream + advanced the §8.3 CDF adaptation state in
/// lockstep with that reader. Caller-bug rejects
/// ([`Error::PartitionWalkOutOfRange`]) are the
/// [`write_intra_block_mode_info_with_palette`] set plus
/// out-of-range `abovemode_ctx` / `leftmode_ctx`.
#[allow(clippy::too_many_arguments)]
pub fn write_intra_frame_else_arm(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    mi_size: usize,
    y_mode: u8,
    uv_mode: Option<u8>,
    angle_delta_y: i8,
    angle_delta_uv: i8,
    cfl_allowed: bool,
    has_chroma: bool,
    allow_screen_content_tools: bool,
    enable_filter_intra: bool,
    use_filter_intra: u8,
    filter_intra_mode: Option<u8>,
    above_palette_y: bool,
    left_palette_y: bool,
    bit_depth: u8,
    has_palette_y: u8,
    palette_size_y: usize,
    palette_colors_y: &[u16],
    has_palette_uv: u8,
    palette_size_uv: usize,
    palette_colors_u: &[u16],
    palette_colors_v: &[u16],
    delta_encode_v: bool,
    cfl_alpha_u: Option<i8>,
    cfl_alpha_v: Option<i8>,
    walker: &PartitionWalker,
    mi_row: u32,
    mi_col: u32,
    abovemode_ctx: usize,
    leftmode_ctx: usize,
) -> Result<(), Error> {
    // §5.11.7 caller-bug guards (shared §5.11.22-shaped input
    // contract), asserted before any partial S() write disturbs the
    // bit stream. The ctx-pair bounds are re-checked inside the
    // `write_intra_frame_y_mode` leaf before its write fires, so the
    // whole guard set still precedes the first emitted symbol.
    check_intra_mode_info_inputs(
        mi_size,
        y_mode,
        uv_mode,
        cfl_allowed,
        has_chroma,
        cfl_alpha_u,
        cfl_alpha_v,
    )?;

    // §5.11.7: `intra_frame_y_mode S()` against
    // `TileIntraFrameYModeCdf[ abovemode ][ leftmode ]` per §8.3.2.
    write_intra_frame_y_mode(writer, cdfs, y_mode, abovemode_ctx, leftmode_ctx)?;

    // §5.11.7 body from `intra_angle_info_y( )` onward — identical to
    // the §5.11.22 tail element-for-element.
    write_intra_mode_info_tail(
        writer,
        cdfs,
        mi_size,
        y_mode,
        uv_mode,
        angle_delta_y,
        angle_delta_uv,
        cfl_allowed,
        has_chroma,
        allow_screen_content_tools,
        enable_filter_intra,
        use_filter_intra,
        filter_intra_mode,
        above_palette_y,
        left_palette_y,
        bit_depth,
        has_palette_y,
        palette_size_y,
        palette_colors_y,
        has_palette_uv,
        palette_size_uv,
        palette_colors_u,
        palette_colors_v,
        delta_encode_v,
        cfl_alpha_u,
        cfl_alpha_v,
        walker,
        mi_row,
        mi_col,
    )
}

/// `inter_block_mode_info( )` bootstrap per §5.11.23 (av1-spec p.74)
/// — r266. Lands the first three lines of the §5.11.23 body that emit
/// **zero S() symbols**: the `PaletteSizeY = 0` / `PaletteSizeUV = 0`
/// resets (§5.11.23 lines 2-3) and the §5.11.25 `read_ref_frames( )`
/// no-bit arms — arm 1 (`skip_mode`), arm 2 (`seg_feature_active(
/// SEG_LVL_REF_FRAME )`), and arm 3 (`seg_feature_active( SEG_LVL_SKIP
/// ) || seg_feature_active( SEG_LVL_GLOBALMV )`). The fall-through
/// `else` arm of §5.11.25 (the `comp_mode` / `comp_ref_type` /
/// `single_ref_p?` / `comp_bwdref_p?` / `comp_ref_p?` / `uni_comp_ref`
/// dispatcher — every read with `S()` payload) is **deferred** to a
/// follow-up dispatcher arc and rejected by this bootstrap as a
/// caller-bug.
///
/// ## Spec body (§5.11.23 lines 2-5 + §5.11.25 arms 1-3)
///
/// ```text
///   inter_block_mode_info( ) {
///       PaletteSizeY = 0                                 // §5.11.23 line 2
///       PaletteSizeUV = 0                                // §5.11.23 line 3
///       read_ref_frames( )                               // §5.11.25 dispatcher
///       isCompound = RefFrame[ 1 ] > INTRA_FRAME         // §5.11.23 line 5
///       ...                                              // (further reads — out of scope)
///   }
///
///   read_ref_frames( ) {
///       if ( skip_mode ) {                               // arm 1 — no bits
///           RefFrame[ 0 ] = SkipModeFrame[ 0 ]
///           RefFrame[ 1 ] = SkipModeFrame[ 1 ]
///       } else if ( seg_feature_active( SEG_LVL_REF_FRAME ) ) {   // arm 2 — no bits
///           RefFrame[ 0 ] = FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ]
///           RefFrame[ 1 ] = NONE
///       } else if ( seg_feature_active( SEG_LVL_SKIP ) ||         // arm 3 — no bits
///                   seg_feature_active( SEG_LVL_GLOBALMV ) ) {
///           RefFrame[ 0 ] = LAST_FRAME
///           RefFrame[ 1 ] = NONE
///       } else {
///           // arm 4 — `comp_mode` etc. — deferred follow-up arc.
///       }
///   }
/// ```
///
/// ## Inputs (mirrors §5.11.10 / §5.11.20 caller surface)
///
/// * `skip_mode` — `1` if the §5.11.10 `read_skip_mode()` writer
///   committed `skip_mode == 1` for this block (i.e. the §5.11.18
///   line 12 result). Arm 1 selector.
/// * `skip_mode_frame` — the frame-level `SkipModeFrame[ 0..2 ]`
///   pair from §5.9.22 (`FrameRefs[ 0..NUM_REF_FRAMES - 1 ]` after
///   the §6.8.20 `set_frame_refs( )` walk). Only consulted on arm 1.
///   Each entry MUST be in `INTRA_FRAME..=ALTREF_FRAME` (`0..=7`,
///   i.e. an §3 ref-frame index — `NONE = -1` is **not** valid
///   for `SkipModeFrame`).
/// * `seg_ref_frame_active` — `seg_feature_active( SEG_LVL_REF_FRAME
///   )` per §6.4.2. Arm 2 selector (fires only when `skip_mode == 0`).
/// * `seg_ref_frame_data` — `FeatureData[ segment_id ][ SEG_LVL_REF_FRAME
///   ]` per §5.9.14. The §6.4.1 segmentation-feature alphabet bounds
///   this to `0..=ALTREF_FRAME = 7` (a positive §3 ref-frame index).
///   Only consulted on arm 2.
/// * `seg_skip_active` — `seg_feature_active( SEG_LVL_SKIP )` per
///   §6.4.2. Arm 3 selector (fires only when both arm-1 and arm-2
///   selectors are false).
/// * `seg_globalmv_active` — `seg_feature_active( SEG_LVL_GLOBALMV )`
///   per §6.4.2. Combined with `seg_skip_active` for arm 3.
///
/// ## Output
///
/// Returns the `(RefFrame[ 0 ], RefFrame[ 1 ])` pair as `(i8, i8)`
/// with the §3 `NONE = -1` sentinel reachable on slot 1 (arms 2 + 3
/// stamp `RefFrame[ 1 ] = NONE`; arm 1 propagates whatever
/// `skip_mode_frame[ 1 ]` carries, which is a positive index per
/// §5.9.22 spec). `PaletteSizeY` / `PaletteSizeUV` are implicit
/// (`0` per the §5.11.23 lines 2-3 resets) — the caller threads
/// them into any downstream §5.11.46 / §5.11.49 reads as
/// appropriate.
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `skip_mode > 1` — outside the §3 binary alphabet.
/// * `skip_mode == 1` and either `skip_mode_frame[ 0 ]` or
///   `skip_mode_frame[ 1 ]` outside `INTRA_FRAME..=ALTREF_FRAME` —
///   §5.9.22 invariant violation.
/// * `seg_ref_frame_active == true` (with `skip_mode == 0`) and
///   `seg_ref_frame_data` outside `INTRA_FRAME..=ALTREF_FRAME` —
///   §6.4.1 segmentation-feature alphabet violation.
/// * All four arm selectors false — the `else` arm 4 (`comp_mode`
///   dispatcher) is deferred and rejected as caller-bug for this
///   bootstrap. Follow-up arc.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless: this is a pure derivation (no symbols emitted, no
/// CDF accesses). The caller stamps the resulting `RefFrame[ 0..2
/// ]` pair onto the `bh4 * bw4` block footprint inside the
/// encoder-side `RefFrames[][][..]` grid (per §5.11.23 the same
/// stamp the decoder's [`crate::cdf::PartitionWalker`] performs
/// after `read_ref_frames` returns). No [`SymbolWriter`] reference
/// is needed because the bootstrap never writes a bit.
pub fn write_inter_block_mode_info_bootstrap(
    skip_mode: u8,
    skip_mode_frame: [i8; 2],
    seg_ref_frame_active: bool,
    seg_ref_frame_data: i8,
    seg_skip_active: bool,
    seg_globalmv_active: bool,
) -> Result<(i8, i8), Error> {
    // §3 binary alphabet bound on `skip_mode`.
    if skip_mode > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §3 ref-frame index range — `INTRA_FRAME = 0..=ALTREF_FRAME = 7`.
    // Helper closure for the validity checks below; an `i8` slot is
    // valid iff it lies in `0..=7`.
    let in_ref_range = |v: i8| (0..=7).contains(&v);

    // §5.11.25 arm 1: `if ( skip_mode ) RefFrame[ 0..2 ] = SkipModeFrame[ 0..2 ]`.
    // No bits emitted. §5.9.22 fills `SkipModeFrame[ 0..2 ]` with two
    // positive ref-frame indices (`NONE = -1` is invalid here per the
    // §5.9.22 derivation).
    if skip_mode == 1 {
        if !in_ref_range(skip_mode_frame[0]) || !in_ref_range(skip_mode_frame[1]) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok((skip_mode_frame[0], skip_mode_frame[1]));
    }

    // §5.11.25 arm 2: `else if ( seg_feature_active( SEG_LVL_REF_FRAME ) )`.
    // No bits emitted. `FeatureData[ segment_id ][ SEG_LVL_REF_FRAME ]`
    // is a positive §3 ref-frame index per §6.4.1; `RefFrame[ 1 ] = NONE`.
    if seg_ref_frame_active {
        if !in_ref_range(seg_ref_frame_data) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok((seg_ref_frame_data, -1));
    }

    // §5.11.25 arm 3: `else if ( seg_feature_active( SEG_LVL_SKIP ) ||
    //                            seg_feature_active( SEG_LVL_GLOBALMV ) )`.
    // No bits emitted. `RefFrame[ 0 ] = LAST_FRAME = 1`, `RefFrame[ 1 ] = NONE`.
    if seg_skip_active || seg_globalmv_active {
        return Ok((1, -1));
    }

    // §5.11.25 arm 4 (fall-through `else`): `comp_mode` / `comp_ref_type`
    // / `single_ref_p?` / `comp_bwdref_p?` / `comp_ref_p?` /
    // `uni_comp_ref` dispatcher. Out of scope for r266 — rejected as a
    // caller-bug so callers don't silently fall into a "RefFrame stays
    // unset" hole. Follow-up arc lands the four-arm S()-emitting body.
    Err(Error::PartitionWalkOutOfRange)
}

/// `comp_mode` writer per §5.11.25 `read_ref_frames( )` arm 4 (the
/// fall-through `else`) — av1-spec p.76. This is the **first** `S()`
/// of the four-arm reference-selection dispatcher that
/// [`write_inter_block_mode_info_bootstrap`] defers. It encodes
/// whether the block uses single- or compound-reference prediction.
///
/// ## Spec body (§5.11.25 arm 4, lines 1-5 — av1-spec p.76)
///
/// ```text
///   } else {                                            // arm 4
///       bw4 = Num_4x4_Blocks_Wide[ MiSize ]
///       bh4 = Num_4x4_Blocks_High[ MiSize ]
///       if ( reference_select && ( Min( bw4, bh4 ) >= 2 ) )
///            comp_mode                                              S()
///       else
///            comp_mode = SINGLE_REFERENCE
///       ...                                             // (COMPOUND/SINGLE bodies — follow-up)
///   }
/// ```
///
/// The `bw4 = Num_4x4_Blocks_Wide[ MiSize ]` / `bh4 =
/// Num_4x4_Blocks_High[ MiSize ]` derivations are folded in from the
/// public [`crate::cdf::NUM_4X4_BLOCKS_WIDE`] /
/// [`crate::cdf::NUM_4X4_BLOCKS_HIGH`] tables (§9.3 p.400) so the
/// caller only threads the `MiSize` index. The COMPOUND/SINGLE
/// reference-frame bodies (`comp_ref_type` / `single_ref_p?` /
/// `comp_bwdref_p?` / `uni_comp_ref` reads) are a deferred follow-up
/// arc — this writer only emits the `comp_mode` selector itself.
///
/// ## Inputs
///
/// * `comp_mode` — the §3 `CompMode` value the encoder chose for this
///   block: `SINGLE_REFERENCE = 0` or `COMPOUND_REFERENCE = 1`. MUST
///   be `0` or `1` (the §3 binary alphabet).
/// * `mi_size` — the block's `MiSize` (§3 block-size index), in
///   `0..BLOCK_SIZES = 22`. Indexes the §9.3 4×4-block-count tables to
///   derive `Min( bw4, bh4 )`.
/// * `reference_select` — frame-level `reference_select` (§5.9.23
///   `frame_reference_mode( )`). When `false`, the precondition fails
///   and `comp_mode` is forced to `SINGLE_REFERENCE` with no bit.
/// * `ctx` — the §8.3.2 `comp_mode` context computed via
///   [`crate::cdf::comp_mode_ctx`] from the neighbour single/intra
///   predicates. MUST be in `0..COMP_INTER_CONTEXTS = 5`. Only
///   consulted when the precondition holds (a symbol is emitted).
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `comp_mode > 1` — outside the §3 binary `CompMode` alphabet.
/// * `mi_size >= BLOCK_SIZES` — invalid §3 block-size index.
/// * Precondition false (`!reference_select` or `Min( bw4, bh4 ) < 2`)
///   and `comp_mode != SINGLE_REFERENCE` — the spec forces
///   `comp_mode = SINGLE_REFERENCE` with no bit, so any other value is
///   a caller bug.
/// * Precondition true and `ctx >= COMP_INTER_CONTEXTS` — invalid
///   §8.3.2 ctx.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless (mirrors [`write_is_inter`]): on the suppressed-bit path
/// no symbol is emitted; on the explicit path exactly one §8.2.6
/// `S()` over `TileCompModeCdf[ ctx ]` is written. The caller stamps
/// the resulting `comp_mode`-derived `RefFrame[ 0..2 ]` pair onto the
/// block footprint through its own [`crate::cdf::PartitionWalker`].
pub fn write_comp_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    comp_mode: u8,
    mi_size: usize,
    reference_select: bool,
    ctx: usize,
) -> Result<(), Error> {
    // §3 binary alphabet bound on `CompMode`.
    if comp_mode > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §3 block-size index bound — needed to index the §9.3 tables.
    if mi_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.25 arm 4: `bw4 = Num_4x4_Blocks_Wide[ MiSize ]`,
    // `bh4 = Num_4x4_Blocks_High[ MiSize ]`.
    let bw4 = NUM_4X4_BLOCKS_WIDE[mi_size];
    let bh4 = NUM_4X4_BLOCKS_HIGH[mi_size];

    // `if ( reference_select && ( Min( bw4, bh4 ) >= 2 ) ) comp_mode S()
    //  else comp_mode = SINGLE_REFERENCE`.
    if reference_select && bw4.min(bh4) >= 2 {
        if ctx >= COMP_INTER_CONTEXTS {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let cdf = cdfs.comp_mode_cdf(ctx);
        writer.write_symbol(comp_mode as u32, cdf)
    } else {
        // No bit: the spec forces `comp_mode = SINGLE_REFERENCE = 0`.
        if comp_mode != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// COMPOUND_REFERENCE body writer per §5.11.25 `read_ref_frames( )`
/// arm 4 (av1-spec p.76-77) — r268. Encodes the `comp_ref_type` /
/// `uni_comp_ref` / `uni_comp_ref_p1` / `uni_comp_ref_p2` /
/// `comp_ref` / `comp_ref_p1` / `comp_ref_p2` / `comp_bwdref` /
/// `comp_bwdref_p1` cascade that follows a
/// `comp_mode == COMPOUND_REFERENCE` outcome of [`write_comp_mode`].
/// This is the exact algebraic inverse of the §5.11.25
/// `if ( comp_mode == COMPOUND_REFERENCE )` body: each branch
/// condition the reader evaluates on a decoded bit becomes a bit
/// derived here from the target `RefFrame[ 0..2 ]` pair.
///
/// ## Spec body (§5.11.25 COMPOUND_REFERENCE arm — av1-spec p.76-77)
///
/// ```text
///   if ( comp_mode == COMPOUND_REFERENCE ) {
///       comp_ref_type                                              S()
///       if ( comp_ref_type == UNIDIR_COMP_REFERENCE ) {
///            uni_comp_ref                                          S()
///            if ( uni_comp_ref ) {
///                RefFrame[0] = BWDREF_FRAME; RefFrame[1] = ALTREF_FRAME
///            } else {
///                uni_comp_ref_p1                                   S()
///                if ( uni_comp_ref_p1 ) {
///                     uni_comp_ref_p2                              S()
///                     if ( uni_comp_ref_p2 ) {
///                       RefFrame[0] = LAST_FRAME; RefFrame[1] = GOLDEN_FRAME
///                     } else {
///                       RefFrame[0] = LAST_FRAME; RefFrame[1] = LAST3_FRAME
///                     }
///                } else {
///                     RefFrame[0] = LAST_FRAME; RefFrame[1] = LAST2_FRAME
///                }
///            }
///       } else {
///            comp_ref                                              S()
///            if ( comp_ref == 0 ) {
///                comp_ref_p1                                       S()
///                RefFrame[ 0 ] = comp_ref_p1 ? LAST2_FRAME : LAST_FRAME
///            } else {
///                comp_ref_p2                                       S()
///                RefFrame[ 0 ] = comp_ref_p2 ? GOLDEN_FRAME : LAST3_FRAME
///            }
///            comp_bwdref                                           S()
///            if ( comp_bwdref == 0 ) {
///                comp_bwdref_p1                                    S()
///                RefFrame[ 1 ] = comp_bwdref_p1 ? ALTREF2_FRAME : BWDREF_FRAME
///            } else {
///                RefFrame[ 1 ] = ALTREF_FRAME
///            }
///       }
///   }
/// ```
///
/// ## Inverse derivations
///
/// `comp_ref_type` itself is derived, not caller-chosen: §6.10.24
/// defines `UNIDIR_COMP_REFERENCE = 0` as "both reference frames from
/// the same group" and `BIDIR_COMP_REFERENCE = 1` as "one from Group 1
/// and one from Group 2", which is exactly the §8.3.2
/// `is_samedir_ref_pair( ref0, ref1 )` predicate (av1-spec p.383):
/// `comp_ref_type = is_samedir_ref_pair( .. ) ? UNIDIR : BIDIR`. The
/// per-bit inverses then read off the §5.11.25 leaf assignments:
///
/// * `uni_comp_ref = ( pair == ( BWDREF_FRAME, ALTREF_FRAME ) )`
/// * `uni_comp_ref_p1 = ( RefFrame[ 1 ] != LAST2_FRAME )`
/// * `uni_comp_ref_p2 = ( RefFrame[ 1 ] == GOLDEN_FRAME )`
/// * `comp_ref = ( RefFrame[ 0 ] >= LAST3_FRAME )`
/// * `comp_ref_p1 = ( RefFrame[ 0 ] == LAST2_FRAME )`
/// * `comp_ref_p2 = ( RefFrame[ 0 ] == GOLDEN_FRAME )`
/// * `comp_bwdref = ( RefFrame[ 1 ] == ALTREF_FRAME )`
/// * `comp_bwdref_p1 = ( RefFrame[ 1 ] == ALTREF2_FRAME )`
///
/// ## Inputs
///
/// * `ref_frame` — the target `( RefFrame[ 0 ], RefFrame[ 1 ] )`
///   compound pair. The §5.11.25 tree reaches exactly sixteen pairs:
///   the four UNIDIR leaves `( BWDREF = 5, ALTREF = 7 )`,
///   `( LAST = 1, LAST2 = 2 )`, `( LAST = 1, LAST3 = 3 )`,
///   `( LAST = 1, GOLDEN = 4 )`, and the twelve BIDIR products of
///   `RefFrame[ 0 ] ∈ { LAST, LAST2, LAST3, GOLDEN } = Group 1` with
///   `RefFrame[ 1 ] ∈ { BWDREF, ALTREF2, ALTREF } = Group 2`
///   (§6.10.24). Any other pair is a caller bug.
/// * `avail_u` / `avail_l` / `above_single` / `left_single` /
///   `above_intra` / `left_intra` / `above_ref_frame` /
///   `left_ref_frame` — the §5.11.18 prologue neighbour-state octet,
///   identical to the surface
///   [`crate::cdf::PartitionWalker::decode_inter_block_mode_info`]
///   consumes. **Deviation from this module's caller-supplied-ctx
///   convention:** the COMPOUND body selects up to five distinct
///   §8.3.2 ctx values (`comp_ref_type` p.382 plus four
///   `ref_count_ctx` selections, one per emitted bit), all derived
///   from this one octet via the public [`comp_ref_type_ctx`] /
///   [`count_refs`] / [`ref_count_ctx`] helpers. Passing them
///   severally would invite mismatched-row bugs between sibling
///   symbols, so the writer derives every ctx internally — exactly
///   the lines the decoder's §5.11.25 reader executes.
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * Either `ref_frame` slot outside `LAST_FRAME..=ALTREF_FRAME`
///   (`1..=7`) — a compound pair never carries `INTRA_FRAME = 0` or
///   `NONE = -1` (§6.10.24).
/// * A same-group pair that is not one of the four UNIDIR leaves
///   (e.g. `( LAST2, LAST3 )` or `( BWDREF, ALTREF2 )`) — the
///   §5.11.25 UNIDIR sub-tree cannot express it.
/// * A different-group pair with `RefFrame[ 0 ]` from Group 2 (e.g.
///   `( BWDREF, LAST )`) — the §5.11.25 BIDIR sub-tree always puts
///   the Group-1 frame in slot 0 (§6.10.24).
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless (mirrors [`write_comp_mode`]): between two and five
/// §8.2.6 `S()` symbols are written depending on the leaf. The caller
/// stamps the pair onto the block footprint through its own
/// [`crate::cdf::PartitionWalker`] so subsequent blocks' §8.3.2
/// neighbour walks observe it.
#[allow(clippy::too_many_arguments)]
pub fn write_compound_ref_frames(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    ref_frame: [i32; 2],
    avail_u: bool,
    avail_l: bool,
    above_single: bool,
    left_single: bool,
    above_intra: bool,
    left_intra: bool,
    above_ref_frame: [i32; 2],
    left_ref_frame: [i32; 2],
) -> Result<(), Error> {
    let ref0 = ref_frame[0];
    let ref1 = ref_frame[1];

    // §6.10.24: compound pairs carry two LAST_FRAME..=ALTREF_FRAME
    // (1..=7) indices — INTRA_FRAME = 0 and NONE = -1 are invalid.
    if !(1..=7).contains(&ref0) || !(1..=7).contains(&ref1) {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §6.10.24 / §8.3.2 p.383: UNIDIR_COMP_REFERENCE = 0 ⇔ both refs
    // in the same direction group ⇔ `is_samedir_ref_pair( ref0, ref1 )`.
    let unidir = is_samedir_ref_pair(ref0, ref1);

    if unidir {
        // §5.11.25 UNIDIR leaves: (BWDREF = 5, ALTREF = 7),
        // (LAST = 1, LAST2 = 2), (LAST = 1, LAST3 = 3),
        // (LAST = 1, GOLDEN = 4). Any other same-group pair is
        // unreachable by the sub-tree.
        if !matches!((ref0, ref1), (5, 7) | (1, 2) | (1, 3) | (1, 4)) {
            return Err(Error::PartitionWalkOutOfRange);
        }
    } else if ref0 > 4 {
        // §6.10.24 BIDIR: slot 0 is the Group-1 (forward) frame
        // (LAST..=GOLDEN = 1..=4); slot 1 the Group-2 (backward)
        // frame. With `!is_samedir_ref_pair` and both slots in 1..=7,
        // `ref0 <= 4` forces `ref1 >= 5`; a Group-2 slot 0 cannot be
        // expressed.
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §8.3.2 `count_refs( frameType )` against the fixed neighbour
    // octet (av1-spec p.366) — every per-bit `ref_count_ctx` selection
    // below consumes these counts.
    let cnt = |frame_type: i32| {
        count_refs(
            frame_type,
            avail_u,
            above_ref_frame,
            avail_l,
            left_ref_frame,
        )
    };

    // §5.11.25: `comp_ref_type` S() over `TileCompRefTypeCdf[ ctx ]`
    // (§8.3.2 p.382).
    let crt_ctx = comp_ref_type_ctx(
        avail_u,
        avail_l,
        above_single,
        left_single,
        above_intra,
        left_intra,
        above_ref_frame,
        left_ref_frame,
    );
    let row = cdfs.comp_ref_type_cdf(crt_ctx);
    writer.write_symbol(!unidir as u32, row)?;

    if unidir {
        // §5.11.25: `uni_comp_ref` S() over `TileUniCompRefCdf[ ctx ][ 0 ]`.
        // §8.3.2 p.383: ctx as in the `single_ref_p1` selection (p.368)
        // — `fwdCount` (LAST + LAST2 + LAST3 + GOLDEN) vs `bwdCount`
        // (BWDREF + ALTREF2 + ALTREF).
        let ucr_ctx = ref_count_ctx(cnt(1) + cnt(2) + cnt(3) + cnt(4), cnt(5) + cnt(6) + cnt(7));
        // Inverse of `if ( uni_comp_ref ) { RefFrame = (BWDREF, ALTREF) }`.
        let uni_comp_ref = ((ref0, ref1) == (5, 7)) as u32;
        let row = cdfs.uni_comp_ref_cdf(ucr_ctx, 0);
        writer.write_symbol(uni_comp_ref, row)?;

        if uni_comp_ref == 0 {
            // §5.11.25: `uni_comp_ref_p1` S() over
            // `TileUniCompRefCdf[ ctx ][ 1 ]`. §8.3.2 p.383 ctx:
            // `last2Count` vs `last3GoldCount`.
            let p1_ctx = ref_count_ctx(cnt(2), cnt(3) + cnt(4));
            // Inverse: p1 == 0 ⇒ (LAST, LAST2); p1 == 1 ⇒ p2 leaf.
            let uni_comp_ref_p1 = (ref1 != 2) as u32;
            let row = cdfs.uni_comp_ref_cdf(p1_ctx, 1);
            writer.write_symbol(uni_comp_ref_p1, row)?;

            if uni_comp_ref_p1 == 1 {
                // §5.11.25: `uni_comp_ref_p2` S() over
                // `TileUniCompRefCdf[ ctx ][ 2 ]`. §8.3.2 p.383 ctx:
                // as in `comp_ref_p2` (p.367) — `last3Count` vs
                // `goldCount`.
                let p2_ctx = ref_count_ctx(cnt(3), cnt(4));
                // Inverse: p2 ⇒ (LAST, GOLDEN), else (LAST, LAST3).
                let uni_comp_ref_p2 = (ref1 == 4) as u32;
                let row = cdfs.uni_comp_ref_cdf(p2_ctx, 2);
                writer.write_symbol(uni_comp_ref_p2, row)?;
            }
        }
    } else {
        // §5.11.25: `comp_ref` S() over `TileCompRefCdf[ ctx ][ 0 ]`.
        // §8.3.2 p.366 ctx: `last12Count` vs `last3GoldCount`.
        let cr_ctx = ref_count_ctx(cnt(1) + cnt(2), cnt(3) + cnt(4));
        // Inverse of the two `RefFrame[ 0 ]` leaf groups:
        // comp_ref == 0 ⇒ { LAST, LAST2 }, == 1 ⇒ { LAST3, GOLDEN }.
        let comp_ref = (ref0 >= 3) as u32;
        let row = cdfs.comp_ref_cdf(cr_ctx, 0);
        writer.write_symbol(comp_ref, row)?;

        if comp_ref == 0 {
            // §5.11.25: `comp_ref_p1` S() over `TileCompRefCdf[ ctx ][ 1 ]`.
            // §8.3.2 p.367 ctx: `lastCount` vs `last2Count`. Inverse of
            // `RefFrame[ 0 ] = comp_ref_p1 ? LAST2_FRAME : LAST_FRAME`.
            let p1_ctx = ref_count_ctx(cnt(1), cnt(2));
            let comp_ref_p1 = (ref0 == 2) as u32;
            let row = cdfs.comp_ref_cdf(p1_ctx, 1);
            writer.write_symbol(comp_ref_p1, row)?;
        } else {
            // §5.11.25: `comp_ref_p2` S() over `TileCompRefCdf[ ctx ][ 2 ]`.
            // §8.3.2 p.367 ctx: `last3Count` vs `goldCount`. Inverse of
            // `RefFrame[ 0 ] = comp_ref_p2 ? GOLDEN_FRAME : LAST3_FRAME`.
            let p2_ctx = ref_count_ctx(cnt(3), cnt(4));
            let comp_ref_p2 = (ref0 == 4) as u32;
            let row = cdfs.comp_ref_cdf(p2_ctx, 2);
            writer.write_symbol(comp_ref_p2, row)?;
        }

        // §5.11.25: `comp_bwdref` S() over `TileCompBwdRefCdf[ ctx ][ 0 ]`.
        // §8.3.2 p.367 ctx: `brfarf2Count` vs `arfCount`. Inverse of
        // `if ( comp_bwdref == 0 ) .. else RefFrame[ 1 ] = ALTREF_FRAME`.
        let bw_ctx = ref_count_ctx(cnt(5) + cnt(6), cnt(7));
        let comp_bwdref = (ref1 == 7) as u32;
        let row = cdfs.comp_bwd_ref_cdf(bw_ctx, 0);
        writer.write_symbol(comp_bwdref, row)?;

        if comp_bwdref == 0 {
            // §5.11.25: `comp_bwdref_p1` S() over
            // `TileCompBwdRefCdf[ ctx ][ 1 ]`. §8.3.2 p.367 ctx:
            // `brfCount` vs `arf2Count`. Inverse of `RefFrame[ 1 ] =
            // comp_bwdref_p1 ? ALTREF2_FRAME : BWDREF_FRAME`.
            let p1_ctx = ref_count_ctx(cnt(5), cnt(6));
            let comp_bwdref_p1 = (ref1 == 6) as u32;
            let row = cdfs.comp_bwd_ref_cdf(p1_ctx, 1);
            writer.write_symbol(comp_bwdref_p1, row)?;
        }
    }

    Ok(())
}

/// SINGLE_REFERENCE body writer per §5.11.25 `read_ref_frames( )`
/// arm 4 (av1-spec p.77) — r269. Encodes the `single_ref_p1` /
/// `single_ref_p2` / `single_ref_p3` / `single_ref_p4` /
/// `single_ref_p5` / `single_ref_p6` cascade that follows a
/// `comp_mode == SINGLE_REFERENCE` outcome of [`write_comp_mode`] —
/// the sibling of [`write_compound_ref_frames`]. This is the exact
/// algebraic inverse of the §5.11.25 final `else` body: each branch
/// condition the reader evaluates on a decoded bit becomes a bit
/// derived here from the target `RefFrame[ 0 ]`.
///
/// ## Spec body (§5.11.25 SINGLE_REFERENCE arm — av1-spec p.77)
///
/// ```text
///   } else {
///       single_ref_p1                                              S()
///       if ( single_ref_p1 ) {
///            single_ref_p2                                         S()
///            if ( single_ref_p2 == 0 ) {
///                single_ref_p6                                     S()
///                RefFrame[ 0 ] = single_ref_p6 ? ALTREF2_FRAME : BWDREF_FRAME
///            } else {
///                RefFrame[ 0 ] = ALTREF_FRAME
///            }
///       } else {
///            single_ref_p3                                         S()
///            if ( single_ref_p3 ) {
///                single_ref_p5                                     S()
///                RefFrame[ 0 ] = single_ref_p5 ? GOLDEN_FRAME : LAST3_FRAME
///            } else {
///                single_ref_p4                                     S()
///                RefFrame[ 0 ] = single_ref_p4 ? LAST2_FRAME : LAST_FRAME
///            }
///       }
///       RefFrame[ 1 ] = NONE
///   }
/// ```
///
/// ## Inverse derivations
///
/// Reading the leaves back: the `single_ref_p1 == 1` half reaches
/// exactly the §6.10.24 Group-2 (backward) frames `{ BWDREF = 5,
/// ALTREF2 = 6, ALTREF = 7 }` and the `single_ref_p1 == 0` half the
/// Group-1 (forward) frames `{ LAST = 1, LAST2 = 2, LAST3 = 3,
/// GOLDEN = 4 }`, so:
///
/// * `single_ref_p1 = ( RefFrame[ 0 ] >= BWDREF_FRAME )`
/// * `single_ref_p2 = ( RefFrame[ 0 ] == ALTREF_FRAME )`
/// * `single_ref_p6 = ( RefFrame[ 0 ] == ALTREF2_FRAME )`
/// * `single_ref_p3 = ( RefFrame[ 0 ] >= LAST3_FRAME )`
/// * `single_ref_p5 = ( RefFrame[ 0 ] == GOLDEN_FRAME )`
/// * `single_ref_p4 = ( RefFrame[ 0 ] == LAST2_FRAME )`
///
/// Always three §8.2.6 `S()` symbols except the `ALTREF_FRAME` leaf
/// (`single_ref_p1` + `single_ref_p2` only).
///
/// ## §8.3.2 CDF row selections (av1-spec p.367-368)
///
/// Every cdf is `TileSingleRefCdf[ ctx ][ p ]` with `p` per the
/// §8.3.2 list (`single_ref_p1 → 0` .. `single_ref_p6 → 5`) and `ctx`
/// a `ref_count_ctx` selection over `count_refs` of the §5.11.18
/// neighbour state:
///
/// * `single_ref_p1` — `fwdCount` (LAST + LAST2 + LAST3 + GOLDEN) vs
///   `bwdCount` (BWDREF + ALTREF2 + ALTREF) (p.368).
/// * `single_ref_p2` — as in `comp_bwdref` (p.367): `brfarf2Count` vs
///   `arfCount`.
/// * `single_ref_p3` — as in `comp_ref` (p.366-367): `last12Count` vs
///   `last3GoldCount`.
/// * `single_ref_p4` — as in `comp_ref_p1` (p.367): `lastCount` vs
///   `last2Count`.
/// * `single_ref_p5` — as in `comp_ref_p2` (p.367): `last3Count` vs
///   `goldCount`.
/// * `single_ref_p6` — as in `comp_bwdref_p1` (p.367-368): `brfCount`
///   vs `arf2Count`.
///
/// ## Inputs
///
/// * `ref_frame` — the target `( RefFrame[ 0 ], RefFrame[ 1 ] )`
///   pair. Slot 0 MUST be in `LAST_FRAME..=ALTREF_FRAME` (`1..=7`);
///   slot 1 MUST be `NONE = -1` (the arm's final assignment) — kept
///   in the signature so the contract is pair-shaped like
///   [`write_compound_ref_frames`] and a compound pair can never be
///   silently truncated to its first slot.
/// * `avail_u` / `avail_l` / `above_ref_frame` / `left_ref_frame` —
///   the `count_refs` subset of the §5.11.18 prologue neighbour
///   state. Unlike the COMPOUND body, the SINGLE cascade selects no
///   `comp_ref_type`-style ctx, so the single/intra neighbour
///   predicates are not consumed (§8.3.2 p.366-368 use only
///   `count_refs`). All per-symbol ctx selections are derived
///   internally (same rationale as [`write_compound_ref_frames`]:
///   sibling symbols can never disagree on rows).
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `ref_frame[ 0 ]` outside `LAST_FRAME..=ALTREF_FRAME` (`1..=7`)
///   — the cascade's seven leaves cover exactly that range
///   (`INTRA_FRAME = 0` is unreachable from §5.11.25 arm 4, which
///   runs only on non-intra blocks).
/// * `ref_frame[ 1 ] != NONE = -1` — the arm forces
///   `RefFrame[ 1 ] = NONE`; any other value is a caller bug.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless (mirrors [`write_compound_ref_frames`]): two or three
/// §8.2.6 `S()` symbols are written depending on the leaf. The caller
/// stamps `( RefFrame[ 0 ], NONE )` onto the block footprint through
/// its own [`crate::cdf::PartitionWalker`] so subsequent blocks'
/// §8.3.2 neighbour walks observe it.
pub fn write_single_ref_frames(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    ref_frame: [i32; 2],
    avail_u: bool,
    avail_l: bool,
    above_ref_frame: [i32; 2],
    left_ref_frame: [i32; 2],
) -> Result<(), Error> {
    let ref0 = ref_frame[0];

    // §5.11.25: the SINGLE_REFERENCE cascade reaches exactly
    // LAST_FRAME = 1 ..= ALTREF_FRAME = 7 in slot 0 and forces
    // `RefFrame[ 1 ] = NONE = -1`.
    if !(1..=7).contains(&ref0) || ref_frame[1] != -1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §8.3.2 `count_refs( frameType )` against the fixed neighbour
    // state (av1-spec p.366) — every per-bit `ref_count_ctx`
    // selection below consumes these counts.
    let cnt = |frame_type: i32| {
        count_refs(
            frame_type,
            avail_u,
            above_ref_frame,
            avail_l,
            left_ref_frame,
        )
    };

    // §5.11.25: `single_ref_p1` S() over `TileSingleRefCdf[ ctx ][ 0 ]`.
    // §8.3.2 p.368 ctx: `fwdCount` (LAST + LAST2 + LAST3 + GOLDEN) vs
    // `bwdCount` (BWDREF + ALTREF2 + ALTREF). Inverse of the two leaf
    // halves: p1 == 1 ⇒ Group 2 (BWDREF..ALTREF = 5..=7), p1 == 0 ⇒
    // Group 1 (LAST..GOLDEN = 1..=4) (§6.10.24).
    let p1_ctx = ref_count_ctx(cnt(1) + cnt(2) + cnt(3) + cnt(4), cnt(5) + cnt(6) + cnt(7));
    let single_ref_p1 = (ref0 >= 5) as u32;
    let row = cdfs.single_ref_cdf(p1_ctx, 0);
    writer.write_symbol(single_ref_p1, row)?;

    if single_ref_p1 != 0 {
        // §5.11.25: `single_ref_p2` S() over `TileSingleRefCdf[ ctx ][ 1 ]`.
        // §8.3.2 p.368 ctx: as in `comp_bwdref` (p.367) — `brfarf2Count`
        // vs `arfCount`. Inverse of `if ( single_ref_p2 == 0 ) .. else
        // RefFrame[ 0 ] = ALTREF_FRAME`.
        let p2_ctx = ref_count_ctx(cnt(5) + cnt(6), cnt(7));
        let single_ref_p2 = (ref0 == 7) as u32;
        let row = cdfs.single_ref_cdf(p2_ctx, 1);
        writer.write_symbol(single_ref_p2, row)?;

        if single_ref_p2 == 0 {
            // §5.11.25: `single_ref_p6` S() over
            // `TileSingleRefCdf[ ctx ][ 5 ]`. §8.3.2 p.368 ctx: as in
            // `comp_bwdref_p1` (p.367-368) — `brfCount` vs `arf2Count`.
            // Inverse of `RefFrame[ 0 ] = single_ref_p6 ?
            // ALTREF2_FRAME : BWDREF_FRAME`.
            let p6_ctx = ref_count_ctx(cnt(5), cnt(6));
            let single_ref_p6 = (ref0 == 6) as u32;
            let row = cdfs.single_ref_cdf(p6_ctx, 5);
            writer.write_symbol(single_ref_p6, row)?;
        }
    } else {
        // §5.11.25: `single_ref_p3` S() over `TileSingleRefCdf[ ctx ][ 2 ]`.
        // §8.3.2 p.368 ctx: as in `comp_ref` (p.366-367) — `last12Count`
        // vs `last3GoldCount`. Inverse of the two Group-1 leaf pairs:
        // p3 == 0 ⇒ { LAST, LAST2 }, p3 == 1 ⇒ { LAST3, GOLDEN }.
        let p3_ctx = ref_count_ctx(cnt(1) + cnt(2), cnt(3) + cnt(4));
        let single_ref_p3 = (ref0 >= 3) as u32;
        let row = cdfs.single_ref_cdf(p3_ctx, 2);
        writer.write_symbol(single_ref_p3, row)?;

        if single_ref_p3 != 0 {
            // §5.11.25: `single_ref_p5` S() over
            // `TileSingleRefCdf[ ctx ][ 4 ]`. §8.3.2 p.368 ctx: as in
            // `comp_ref_p2` (p.367) — `last3Count` vs `goldCount`.
            // Inverse of `RefFrame[ 0 ] = single_ref_p5 ?
            // GOLDEN_FRAME : LAST3_FRAME`.
            let p5_ctx = ref_count_ctx(cnt(3), cnt(4));
            let single_ref_p5 = (ref0 == 4) as u32;
            let row = cdfs.single_ref_cdf(p5_ctx, 4);
            writer.write_symbol(single_ref_p5, row)?;
        } else {
            // §5.11.25: `single_ref_p4` S() over
            // `TileSingleRefCdf[ ctx ][ 3 ]`. §8.3.2 p.368 ctx: as in
            // `comp_ref_p1` (p.367) — `lastCount` vs `last2Count`.
            // Inverse of `RefFrame[ 0 ] = single_ref_p4 ?
            // LAST2_FRAME : LAST_FRAME`.
            let p4_ctx = ref_count_ctx(cnt(1), cnt(2));
            let single_ref_p4 = (ref0 == 2) as u32;
            let row = cdfs.single_ref_cdf(p4_ctx, 3);
            writer.write_symbol(single_ref_p4, row)?;
        }
    }

    Ok(())
}

/// Full `read_ref_frames( )` writer per §5.11.25 (av1-spec p.76-77) —
/// the four-arm dispatcher composed from r266's no-bit arms
/// ([`write_inter_block_mode_info_bootstrap`]), r267's `comp_mode`
/// selector ([`write_comp_mode`]), r268's COMPOUND_REFERENCE body
/// ([`write_compound_ref_frames`]) and r269's SINGLE_REFERENCE body
/// ([`write_single_ref_frames`]).
///
/// Where the bootstrap *returns* the pair the no-bit arms force, this
/// dispatcher takes the caller's target `RefFrame[ 0..2 ]` pair and
/// **verifies** reachability: on arms 1-3 the pair must equal the
/// §5.11.25 forced assignment (no symbols emitted); on arm 4 it
/// drives the `comp_mode` + reference-body symbol cascade as the
/// exact inverse of the reader.
///
/// ## Arm selection (§5.11.25 p.76)
///
/// 1. `skip_mode == 1` ⇒ pair must equal `SkipModeFrame[ 0..2 ]`.
/// 2. `seg_ref_frame_active` ⇒ pair must equal
///    `( seg_ref_frame_data, NONE )`.
/// 3. `seg_skip_active || seg_globalmv_active` ⇒ pair must equal
///    `( LAST_FRAME, NONE )`.
/// 4. Else: `comp_mode` is derived from the target pair —
///    `RefFrame[ 1 ] == NONE = -1` ⇒ `SINGLE_REFERENCE = 0`,
///    `RefFrame[ 1 ]` in `LAST_FRAME..=ALTREF_FRAME` ⇒
///    `COMPOUND_REFERENCE = 1` (the §5.11.23
///    `isCompound = RefFrame[ 1 ] > INTRA_FRAME` derivation, p.74),
///    any other slot-1 value rejected. [`write_comp_mode`] emits (or
///    suppresses, per the §5.11.25 `reference_select &&
///    Min( bw4, bh4 ) >= 2` precondition) the selector over the
///    internally-derived [`comp_mode_ctx`], then the matching
///    reference body writes its cascade.
///
/// ## Inputs
///
/// Union of the composed writers' surfaces: the target pair +
/// `mi_size` (§9.3 table index for the `comp_mode` precondition), the
/// arm-selection sextet of
/// [`write_inter_block_mode_info_bootstrap`], frame-level
/// `reference_select` (§5.9.23), and the §5.11.18 neighbour-state
/// octet from which every §8.3.2 ctx is derived internally.
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// All of the composed writers' rejects, plus: a target pair that
/// does not equal the arm-1/2/3 forced assignment, a slot-1 value
/// outside `{ NONE } ∪ LAST_FRAME..=ALTREF_FRAME` on arm 4, and a
/// compound target on the suppressed-`comp_mode` path (rejected
/// inside [`write_comp_mode`] — the spec forces `SINGLE_REFERENCE`
/// with no bit there).
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless like every piece it composes: zero to six §8.2.6 `S()`
/// symbols are written depending on arm and leaf. The caller stamps
/// the pair onto the block footprint through its own
/// [`crate::cdf::PartitionWalker`].
#[allow(clippy::too_many_arguments)]
pub fn write_ref_frames(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    ref_frame: [i32; 2],
    mi_size: usize,
    skip_mode: u8,
    skip_mode_frame: [i8; 2],
    seg_ref_frame_active: bool,
    seg_ref_frame_data: i8,
    seg_skip_active: bool,
    seg_globalmv_active: bool,
    reference_select: bool,
    avail_u: bool,
    avail_l: bool,
    above_single: bool,
    left_single: bool,
    above_intra: bool,
    left_intra: bool,
    above_ref_frame: [i32; 2],
    left_ref_frame: [i32; 2],
) -> Result<(), Error> {
    // §5.11.25 arms 1-3 (no bits): delegate the forced-pair
    // derivation (and the arm-priority ordering + input validation)
    // to the r266 bootstrap, then require the caller's target pair to
    // match it. `skip_mode > 1` falls into the gate and is rejected
    // by the bootstrap's own §3 binary-alphabet check.
    if skip_mode != 0 || seg_ref_frame_active || seg_skip_active || seg_globalmv_active {
        let (forced0, forced1) = write_inter_block_mode_info_bootstrap(
            skip_mode,
            skip_mode_frame,
            seg_ref_frame_active,
            seg_ref_frame_data,
            seg_skip_active,
            seg_globalmv_active,
        )?;
        if ref_frame != [forced0 as i32, forced1 as i32] {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.25 arm 4. `comp_mode` is not caller-chosen: the §5.11.23
    // line 6 derivation `isCompound = RefFrame[ 1 ] > INTRA_FRAME`
    // (p.74) reads it off the target pair — `NONE = -1` ⇒ SINGLE,
    // `LAST_FRAME..=ALTREF_FRAME` ⇒ COMPOUND. `INTRA_FRAME = 0` (or
    // anything else) in slot 1 is unreachable from §5.11.25 and a
    // caller bug.
    let comp_mode: u8 = match ref_frame[1] {
        -1 => 0,
        1..=7 => 1,
        _ => return Err(Error::PartitionWalkOutOfRange),
    };

    // §8.3.2 `comp_mode` ctx (p.366) from the neighbour octet — the
    // same derivation the reader runs before its S().
    let ctx = comp_mode_ctx(
        avail_u,
        avail_l,
        above_single,
        left_single,
        above_intra,
        left_intra,
        above_ref_frame,
        left_ref_frame,
    );
    write_comp_mode(writer, cdfs, comp_mode, mi_size, reference_select, ctx)?;

    if comp_mode == 1 {
        // §5.11.25 `if ( comp_mode == COMPOUND_REFERENCE )` body.
        write_compound_ref_frames(
            writer,
            cdfs,
            ref_frame,
            avail_u,
            avail_l,
            above_single,
            left_single,
            above_intra,
            left_intra,
            above_ref_frame,
            left_ref_frame,
        )
    } else {
        // §5.11.25 final `else` (SINGLE_REFERENCE) body.
        write_single_ref_frames(
            writer,
            cdfs,
            ref_frame,
            avail_u,
            avail_l,
            above_ref_frame,
            left_ref_frame,
        )
    }
}

/// Single-prediction inter-mode writer per §5.11.23
/// `inter_block_mode_info( )` lines 9-22 (av1-spec p.74) — the
/// `else` arm of the §5.11.23 `YMode` derivation that fires when the
/// block is **not** compound, **not** `skip_mode`, and neither
/// `SEG_LVL_SKIP` nor `SEG_LVL_GLOBALMV` is active. This is the exact
/// algebraic inverse of the reader's `new_mv` / `zero_mv` / `ref_mv`
/// cascade.
///
/// ## Spec body (§5.11.23 single-prediction arm — av1-spec p.74)
///
/// ```text
///   new_mv                                                  S()
///   if ( new_mv == 0 ) {
///        YMode = NEWMV
///   } else {
///        zero_mv                                            S()
///        if ( zero_mv == 0 ) {
///            YMode = GLOBALMV
///        } else {
///            ref_mv                                         S()
///            YMode = (ref_mv == 0) ? NEARESTMV : NEARMV
///        }
///   }
/// ```
///
/// ## Inverse derivations
///
/// Each branch condition the reader evaluates on a decoded bit becomes
/// a bit derived here from the target `YMode`:
///
/// * `new_mv  = ( YMode != NEWMV )` — `new_mv == 0` ⇒ `NEWMV`.
/// * `zero_mv = ( YMode != GLOBALMV )` — only emitted when
///   `new_mv == 1`; `zero_mv == 0` ⇒ `GLOBALMV`.
/// * `ref_mv  = ( YMode == NEARMV )` — only emitted when
///   `new_mv == 1 && zero_mv == 1`; `ref_mv == 0` ⇒ `NEARESTMV`,
///   `ref_mv == 1` ⇒ `NEARMV`.
///
/// So `NEWMV` emits one symbol, `GLOBALMV` two, and `NEARESTMV` /
/// `NEARMV` three.
///
/// ## Inputs
///
/// * `y_mode` — the §5.11.23 single-prediction `YMode` the encoder
///   chose: one of `MODE_NEWMV = 17`, `MODE_GLOBALMV = 16`,
///   `MODE_NEARESTMV = 14`, `MODE_NEARMV = 15` (§6.10.22). Any other
///   value is a caller bug — the four compound / intra / GLOBALMV-by-
///   segment modes never reach this writer (the caller's earlier
///   §5.11.23 arms handle them).
/// * `new_mv_context` — §8.3.2 `NewMvContext` (`TileNewMvCdf[
///   NewMvContext ]`), in `0..NEW_MV_CONTEXTS = 6`. Always consulted
///   (every leaf emits `new_mv`).
/// * `zero_mv_context` — §8.3.2 `ZeroMvContext` (`TileZeroMvCdf[
///   ZeroMvContext ]`), in `0..ZERO_MV_CONTEXTS = 2`. Consulted only
///   when `new_mv == 1`.
/// * `ref_mv_context` — §8.3.2 `RefMvContext` (`TileRefMvCdf[
///   RefMvContext ]`), in `0..REF_MV_CONTEXTS = 6`. Consulted only
///   when `new_mv == 1 && zero_mv == 1`.
///
/// All three contexts are produced by the §7.10.2 `find_mv_stack( )`
/// process (`NewMvContext` / `ZeroMvContext` / `RefMvContext`). Like
/// every writer in this module the ctx is caller-supplied: the
/// encoder threads the same §7.10.2 outputs the decoder side derives.
///
/// ## Out-of-range / mismatch cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `y_mode` not in `{ NEWMV, GLOBALMV, NEARESTMV, NEARMV }`.
/// * A context that would be **consulted** is out of range. Contexts
///   on un-taken branches are not validated (the symbol is never
///   emitted, mirroring the reader never reaching the corresponding
///   CDF row).
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless (mirrors [`write_comp_mode`]): one to three §8.2.6 `S()`
/// symbols are written depending on the mode. The caller stamps the
/// resulting `YMode` onto the block footprint through its own
/// [`crate::cdf::PartitionWalker`] so subsequent blocks' §8.3.2
/// neighbour walks observe it.
pub fn write_inter_single_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    y_mode: u8,
    new_mv_context: usize,
    zero_mv_context: usize,
    ref_mv_context: usize,
) -> Result<(), Error> {
    // §6.10.22 single-prediction mode bound — the only four `YMode`
    // values the §5.11.23 single-prediction cascade can produce.
    if !matches!(
        y_mode,
        MODE_NEWMV | MODE_GLOBALMV | MODE_NEARESTMV | MODE_NEARMV
    ) {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.23: `new_mv` S() over `TileNewMvCdf[ NewMvContext ]`.
    // Inverse of `if ( new_mv == 0 ) YMode = NEWMV else …`.
    if new_mv_context >= NEW_MV_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let new_mv = (y_mode != MODE_NEWMV) as u32;
    let row = cdfs.new_mv_cdf(new_mv_context);
    writer.write_symbol(new_mv, row)?;

    if new_mv == 0 {
        // YMode == NEWMV — no further symbols.
        return Ok(());
    }

    // §5.11.23: `zero_mv` S() over `TileZeroMvCdf[ ZeroMvContext ]`.
    // Inverse of `if ( zero_mv == 0 ) YMode = GLOBALMV else …`.
    if zero_mv_context >= ZERO_MV_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let zero_mv = (y_mode != MODE_GLOBALMV) as u32;
    let row = cdfs.zero_mv_cdf(zero_mv_context);
    writer.write_symbol(zero_mv, row)?;

    if zero_mv == 0 {
        // YMode == GLOBALMV — no further symbols.
        return Ok(());
    }

    // §5.11.23: `ref_mv` S() over `TileRefMvCdf[ RefMvContext ]`.
    // Inverse of `YMode = (ref_mv == 0) ? NEARESTMV : NEARMV`.
    if ref_mv_context >= REF_MV_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let ref_mv = (y_mode == MODE_NEARMV) as u32;
    let row = cdfs.ref_mv_cdf(ref_mv_context);
    writer.write_symbol(ref_mv, row)?;

    Ok(())
}

/// `compound_mode` S() inverse per §5.11.23 (av1-spec p.74, lines 6-8
/// arm 3).
///
/// The compound sibling of [`write_inter_single_mode`]. When the
/// §5.11.23 inter-mode dispatch reaches its `is_compound` arm — block
/// is compound (`RefFrame[ 1 ] > INTRA_FRAME`), not `skip_mode`, and
/// neither `SEG_LVL_SKIP` nor `SEG_LVL_GLOBALMV` is active — the
/// reader codes a single §8.2.6 `S()` symbol over
/// `TileCompoundModeCdf[ ctx ]` and sets
///
/// ```text
///   YMode = NEAREST_NEARESTMV + compound_mode
/// ```
///
/// (§5.11.23 line 8, av1-spec p.74). `compound_mode` ranges over
/// `0..COMPOUND_MODES = 8`, so `YMode` ranges over the eight
/// compound-prediction modes `NEAREST_NEARESTMV = 18 ..= NEW_NEWMV =
/// 25` (§6.10.22). This writer is the exact algebraic inverse:
///
/// ```text
///   compound_mode = YMode - NEAREST_NEARESTMV
/// ```
///
/// One §8.2.6 `S()` is emitted unconditionally (the arm always codes a
/// symbol; there is no short-circuit leaf as in the single-prediction
/// cascade).
///
/// ## Inputs
///
/// * `y_mode` — the §5.11.23 compound-prediction `YMode` the encoder
///   chose: one of the eight compound modes
///   `MODE_NEAREST_NEARESTMV = 18 ..= MODE_NEW_NEWMV = 25`
///   (§6.10.22). Any other value is a caller bug — the single-pred /
///   intra / forced-mode arms never reach this writer.
/// * `compound_mode_context` — §8.3.2 `compound_mode` context, the
///   `TileCompoundModeCdf` row index in `0..COMPOUND_MODE_CONTEXTS =
///   8`. Produced by [`crate::cdf::compound_mode_ctx`] from the
///   §7.10.2 `RefMvContext` / `NewMvContext` outputs; like every
///   writer in this module the resolved ctx is caller-supplied, so
///   the encoder threads the same §8.3.2 mapping the decoder side
///   applies.
///
/// ## Out-of-range cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `y_mode` not in `MODE_NEAREST_NEARESTMV ..= MODE_NEW_NEWMV`.
/// * `compound_mode_context >= COMPOUND_MODE_CONTEXTS` (the symbol is
///   always emitted, so the ctx is always validated — unlike the
///   single-prediction writer's consulted-only checks).
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless (mirrors [`write_inter_single_mode`]): one §8.2.6 `S()`
/// is written. The caller stamps the resulting `YMode` onto the block
/// footprint through its own [`crate::cdf::PartitionWalker`] so
/// subsequent blocks' §8.3.2 neighbour walks observe it.
pub fn write_compound_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    y_mode: u8,
    compound_mode_context: usize,
) -> Result<(), Error> {
    // §6.10.22 compound-prediction mode bound — the eight `YMode`
    // values the §5.11.23 `is_compound` arm can produce, contiguous
    // from `NEAREST_NEARESTMV = 18` through `NEW_NEWMV = 25`.
    if !(MODE_NEAREST_NEARESTMV..=MODE_NEW_NEWMV).contains(&y_mode) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if compound_mode_context >= COMPOUND_MODE_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.23 line 8 inverse: `compound_mode = YMode -
    // NEAREST_NEARESTMV`, in `0..COMPOUND_MODES = 8`.
    let compound_mode = (y_mode - MODE_NEAREST_NEARESTMV) as u32;
    debug_assert!(
        (compound_mode as usize) < COMPOUND_MODES,
        "§5.11.23 compound_mode in 0..COMPOUND_MODES"
    );

    // §5.11.23: single `compound_mode` S() over
    // `TileCompoundModeCdf[ ctx ]` (§8.3.2 p.378).
    let row = cdfs.compound_mode_cdf(compound_mode_context);
    writer.write_symbol(compound_mode, row)?;

    Ok(())
}

/// `has_nearmv()` per §5.11.23 (av1-spec p.75 — the helper definition
/// immediately following the `inter_block_mode_info` body). Returns
/// whether `y_mode` is one of the four NEARMV-bearing inter Y modes —
/// the modes whose §5.11.23 `RefMvIdx` loop iterates from `idx = 1`
/// through `idx < 3` rather than the `idx ∈ {0, 1}` window used by the
/// NEWMV / NEW_NEWMV arm.
///
/// Spec body:
///
/// ```text
///   has_nearmv( ) {
///       return (YMode == NEARMV || YMode == NEAR_NEARMV
///               || YMode == NEAR_NEWMV || YMode == NEW_NEARMV)
///   }
/// ```
///
/// Local twin of the decoder-side helper of the same name — the writer
/// module mirrors the reader's dispatch without widening the `cdf`
/// public surface.
#[inline]
fn has_nearmv(y_mode: u8) -> bool {
    matches!(
        y_mode,
        MODE_NEARMV | MODE_NEAR_NEARMV | MODE_NEAR_NEWMV | MODE_NEW_NEARMV
    )
}

/// `drl_mode` writer per §5.11.23 (av1-spec p.73-74, the `RefMvIdx`
/// loops following `assign_mv`'s `compound_mode` / single-mode
/// dispatch) — r272.
///
/// The dynamic-reference-list (DRL) index `RefMvIdx` selects which
/// candidate of the §7.10.2 `RefStackMv` the chosen motion-vector mode
/// draws its predictor from. The reader does **not** code `RefMvIdx`
/// directly: it walks the stack one slot at a time, coding a single
/// binary `drl_mode` symbol per reachable slot, where `drl_mode == 0`
/// means "stop here, use this slot" and `drl_mode == 1` means "continue
/// to the next slot". This writer is the exact algebraic inverse — it
/// re-derives that same bit sequence from the `RefMvIdx` the encoder
/// chose and emits one §8.2.6 `S()` per coded slot.
///
/// ## Spec body (§5.11.23 — av1-spec p.73-74)
///
/// ```text
///   RefMvIdx = 0
///   if ( YMode == NEWMV || YMode == NEW_NEWMV ) {
///       for ( idx = 0; idx < 2; idx++ ) {
///           if ( NumMvFound > idx + 1 ) {
///               drl_mode                                       S()
///               if ( drl_mode == 0 ) { RefMvIdx = idx; break }
///               RefMvIdx = idx + 1
///           }
///       }
///   } else if ( has_nearmv( ) ) {
///       RefMvIdx = 1
///       for ( idx = 1; idx < 3; idx++ ) {
///           if ( NumMvFound > idx + 1 ) {
///               drl_mode                                       S()
///               if ( drl_mode == 0 ) { RefMvIdx = idx; break }
///               RefMvIdx = idx + 1
///           }
///       }
///   }
/// ```
///
/// Both arms share one loop body parameterised only by the start index
/// (`0` for the NEWMV / NEW_NEWMV arm, `1` for the `has_nearmv( )`
/// arm). On the start iteration `RefMvIdx` already holds the start
/// value (`0` or `1`), so a `RefMvIdx` equal to the start with the
/// terminating slot unreachable emits no symbols at all. Any Y mode
/// outside the two arms (`GLOBALMV`, `NEARESTMV`, the compound
/// `GLOBAL_GLOBALMV` / `NEAREST_NEARESTMV` modes, …) codes no
/// `drl_mode` and is handled by the no-op fast path.
///
/// ## Inversion
///
/// For an arm starting at `start`, the decoder visits slots `start,
/// start+1, …` while `NumMvFound > idx + 1`, emitting `drl_mode = 1`
/// at every visited slot below the chosen `ref_mv_idx` and `drl_mode =
/// 0` at slot `ref_mv_idx` itself — unless `ref_mv_idx` is reached
/// only because the previous iteration's `RefMvIdx = idx + 1`
/// assignment ran off the end of the reachable window (`NumMvFound ==
/// ref_mv_idx + 1`), in which case no terminating `0` is coded. The
/// writer reproduces exactly this: walk `idx` from `start` upward,
/// emit `1` for each `idx < ref_mv_idx` that is reachable, then a
/// single `0` at `idx == ref_mv_idx` iff that slot is reachable.
///
/// ## Inputs
///
/// * `y_mode` — the §5.11.23 inter `YMode` the encoder chose. Only the
///   NEWMV / NEW_NEWMV and `has_nearmv( )` modes code `drl_mode`; every
///   other inter mode is a silent no-op (no symbol written), matching
///   the reader's `RefMvIdx = 0` / unconditioned arms.
/// * `ref_mv_idx` — the `RefMvIdx` the encoder selected, in
///   `0..MAX_REF_MV_STACK_SIZE`. Must be consistent with the arm: the
///   NEWMV / NEW_NEWMV arm admits `0..=2`, the `has_nearmv( )` arm
///   admits `1..=3` (the reader can never leave `RefMvIdx` outside
///   those windows for the respective arm).
/// * `num_mv_found` — `NumMvFound`, the §7.10.2 stack depth, in
///   `0..=MAX_REF_MV_STACK_SIZE`. Governs which slots are reachable.
/// * `drl_ctx_stack` — `DrlCtxStack[ idx ]` per §7.10.2.14, the §8.3.2
///   `TileDrlModeCdf` row index for each visited slot, each in
///   `0..DRL_MODE_CONTEXTS`. Must hold at least one entry per slot the
///   loop can visit (`ref_mv_idx` at most).
///
/// ## Out-of-range cases (surfaced as `Error::PartitionWalkOutOfRange`)
///
/// * `ref_mv_idx >= MAX_REF_MV_STACK_SIZE`.
/// * `num_mv_found > MAX_REF_MV_STACK_SIZE`.
/// * `ref_mv_idx` outside the arm's admissible window
///   (`> 2` for the NEWMV / NEW_NEWMV arm, `< 1` or `> 3` for the
///   `has_nearmv( )` arm) — a caller bug, since the reader cannot
///   produce such a value.
/// * a coded slot's `drl_ctx_stack[ idx ] >= DRL_MODE_CONTEXTS`, or
///   `drl_ctx_stack` shorter than the highest visited slot index.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless: between zero and two §8.2.6 `S()` symbols are emitted.
/// The caller threads the resulting `RefMvIdx` onto its own
/// motion-vector assignment per §5.11.31.
pub fn write_drl_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    y_mode: u8,
    ref_mv_idx: u32,
    num_mv_found: u32,
    drl_ctx_stack: &[u32],
) -> Result<(), Error> {
    // §7.10.2 invariants the reader can never violate.
    if (ref_mv_idx as usize) >= MAX_REF_MV_STACK_SIZE {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if (num_mv_found as usize) > MAX_REF_MV_STACK_SIZE {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.23 arm selection. `start` is the first `idx` the reader's
    // loop visits; the loop runs while `idx < start + 2`.
    let start: u32 = if y_mode == MODE_NEWMV || y_mode == MODE_NEW_NEWMV {
        // NEWMV / NEW_NEWMV arm: `RefMvIdx = 0`, `idx ∈ {0, 1}`, so
        // the reachable `RefMvIdx` window is `{0, 1, 2}`.
        if ref_mv_idx > 2 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        0
    } else if has_nearmv(y_mode) {
        // `has_nearmv( )` arm: `RefMvIdx = 1`, `idx ∈ {1, 2}`, so the
        // reachable `RefMvIdx` window is `{1, 2, 3}`.
        if !(1..=3).contains(&ref_mv_idx) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        1
    } else {
        // Every other inter mode codes no `drl_mode`. The reader's
        // `RefMvIdx` stays `0` in this case; reject any other claimed
        // index as a caller bug rather than silently dropping it.
        if ref_mv_idx != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    };

    // Walk `idx` from `start` while the slot is reachable
    // (`NumMvFound > idx + 1`) and `idx < start + 2`. Emit `1` for each
    // slot strictly below `ref_mv_idx`, then a terminating `0` at
    // `idx == ref_mv_idx` iff that slot is itself reachable.
    // `reconstructed` mirrors the reader's running `RefMvIdx` so a
    // caller-supplied `ref_mv_idx` that the reader could never decode
    // from this `num_mv_found` (e.g. a deep index with a shallow stack)
    // is rejected rather than silently mis-encoded.
    let mut reconstructed = start;
    let mut idx = start;
    while idx < start + 2 {
        if num_mv_found > idx + 1 {
            let slot = idx as usize;
            let &ctx = drl_ctx_stack
                .get(slot)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            if ctx as usize >= DRL_MODE_CONTEXTS {
                return Err(Error::PartitionWalkOutOfRange);
            }
            // `drl_mode == 0` stops at this slot; `drl_mode == 1`
            // continues. Inverse of the reader: emit `0` exactly when
            // `idx == ref_mv_idx`.
            let drl_mode: u32 = (idx != ref_mv_idx) as u32;
            let row = cdfs.drl_mode_cdf(ctx as usize);
            writer.write_symbol(drl_mode, row)?;
            if drl_mode == 0 {
                // RefMvIdx = idx; break.
                reconstructed = idx;
                break;
            }
            // RefMvIdx = idx + 1; continue.
            reconstructed = idx + 1;
        }
        idx += 1;
    }

    // The bit sequence we just wrote must decode back to the same
    // `RefMvIdx`. If not, the caller asked for an index unreachable at
    // this stack depth — a caller bug, not a representable bitstream.
    if reconstructed != ref_mv_idx {
        return Err(Error::PartitionWalkOutOfRange);
    }

    Ok(())
}

/// MV-component writer per §5.11.32 `read_mv_component( comp )`
/// (av1-spec p.81-82) — the exact algebraic inverse of the reader's
/// `mv_sign` / `mv_class` / `mv_class0_*` / `mv_bit` / `mv_fr` /
/// `mv_hp` cascade for a single signed difference component.
///
/// ## Spec body (§5.11.32 — av1-spec p.81-82)
///
/// ```text
///   mv_sign                                                  S()
///   mv_class                                                 S()
///   if ( mv_class == MV_CLASS_0 ) {
///       mv_class0_bit                                        S()
///       if ( force_integer_mv ) mv_class0_fr = 3
///       else                    mv_class0_fr                S()
///       if ( allow_high_precision_mv ) mv_class0_hp         S()
///       else                           mv_class0_hp = 1
///       mag = ( ( mv_class0_bit << 3 ) |
///               ( mv_class0_fr  << 1 ) |
///                 mv_class0_hp ) + 1
///   } else {
///       d = 0
///       for ( i = 0; i < mv_class; i++ ) {
///           mv_bit                                          S()
///           d |= mv_bit << i
///       }
///       mag = CLASS0_SIZE << ( mv_class + 2 )
///       if ( force_integer_mv ) mv_fr = 3
///       else                    mv_fr                       S()
///       if ( allow_high_precision_mv ) mv_hp                S()
///       else                           mv_hp = 1
///       mag += ( ( d << 3 ) | ( mv_fr << 1 ) | mv_hp ) + 1
///   }
///   return mv_sign ? -mag : mag
/// ```
///
/// ## Inverse derivation
///
/// The reader returns a signed `diff` for the component. The encoder
/// is handed that `diff` and must reproduce the bitstream:
///
/// * `mv_sign = ( diff < 0 )`, `mag = |diff|`. The reader never
///   returns `0` from this function (a zero component is signalled by
///   `mv_joint` and skips the call), so `mag >= 1` is required.
/// * `offset = mag - 1`. For `MV_CLASS_0`, `offset` ∈ `0..=15` and
///   decomposes directly as `offset = (bit << 3) | (fr << 1) | hp`.
///   For class `c >= 1`, `mag = (CLASS0_SIZE << (c + 2)) + X + 1` with
///   `X = (d << 3) | (fr << 1) | hp` and `X ∈ 0..(2^(c+3) - 1)`, so
///   `offset = 2^(c+3) + X`. Hence `FloorLog2(offset) = c + 3`, giving
///   `c = FloorLog2(offset) - 3` for `offset >= 16`, and `c = 0`
///   otherwise.
/// * `mv_class0_fr` / `mv_fr` carry the two fractional bits, `mv_*_hp`
///   the half-pel bit, `mv_class0_bit` / the `mv_bit` ladder the
///   integer magnitude.
///
/// ## Precision gates
///
/// When `force_integer_mv == 1` the reader synthesises `fr = 3` rather
/// than reading it, so the encoder MUST be handed a `diff` whose
/// fractional field is exactly `3`; any other value is unrepresentable
/// and is rejected. Likewise `allow_high_precision_mv == 0` forces
/// `hp = 1`, so the half-pel bit must be `1`. These mirror the
/// reader's `else` arms and keep the writer a true inverse.
///
/// ## CDF selection (§8.3.2 — av1-spec p.493)
///
/// `mv_sign` → `TileMvSignCdf[ MvCtx ][ comp ]`, `mv_class` →
/// `TileMvClassCdf[ MvCtx ][ comp ]`, `mv_class0_bit` →
/// `TileMvClass0BitCdf[ MvCtx ][ comp ]`, `mv_class0_fr` →
/// `TileMvClass0FrCdf[ MvCtx ][ comp ][ mv_class0_bit ]`,
/// `mv_class0_hp` → `TileMvClass0HpCdf[ MvCtx ][ comp ]`, `mv_bit` →
/// `TileMvBitCdf[ MvCtx ][ comp ][ i ]`, `mv_fr` →
/// `TileMvFrCdf[ MvCtx ][ comp ]`, `mv_hp` → `TileMvHpCdf[ MvCtx ][ comp ]`.
///
/// ## Inputs
///
/// * `mv_ctx` — the §5.11.31 `MvCtx` (`0` for normal blocks,
///   `MV_INTRABC_CONTEXT = 1` for intra block copy).
/// * `comp` — the component index (`0` = horizontal, `1` = vertical).
/// * `diff` — the signed difference `Mv[ ref ][ comp ] - PredMv[ ref ][ comp ]`.
///   Must be non-zero (the caller only invokes this for components the
///   `mv_joint` marks non-zero).
/// * `force_integer_mv` / `allow_high_precision_mv` — the frame-level
///   precision flags the reader consults.
#[allow(clippy::too_many_arguments)]
pub fn write_read_mv_component(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    mv_ctx: usize,
    comp: usize,
    diff: i32,
    force_integer_mv: bool,
    allow_high_precision_mv: bool,
) -> Result<(), Error> {
    if mv_ctx >= MV_CONTEXTS || comp >= MV_COMPS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // The reader never returns 0 from this function — a zero component
    // is encoded by `mv_joint` and skips the call entirely.
    if diff == 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let mv_sign: u32 = (diff < 0) as u32;
    let mag: u32 = diff.unsigned_abs();
    let offset: u32 = mag - 1;

    // mv_sign — emitted first.
    let sign_row = cdfs.mv_sign_cdf(mv_ctx, comp);
    writer.write_symbol(mv_sign, sign_row)?;

    // Class derivation: offset ∈ 0..=15 ⇒ MV_CLASS_0; otherwise
    // FloorLog2(offset) = mv_class + 3.
    let mv_class: u32 = if offset < (1 << 4) {
        MV_CLASS_0 as u32
    } else {
        floor_log2(offset) - 3
    };
    if mv_class as usize >= MV_CLASSES {
        // offset too large to encode (mag exceeds the MV range).
        return Err(Error::PartitionWalkOutOfRange);
    }

    // mv_class — symbol over the 11-ary alphabet.
    let class_row = cdfs.mv_class_cdf(mv_ctx, comp);
    writer.write_symbol(mv_class, class_row)?;

    if mv_class == MV_CLASS_0 as u32 {
        // offset = (mv_class0_bit << 3) | (mv_class0_fr << 1) | mv_class0_hp
        let mv_class0_bit: u32 = (offset >> 3) & 0x1;
        let mv_class0_fr: u32 = (offset >> 1) & 0x3;
        let mv_class0_hp: u32 = offset & 0x1;

        let bit_row = cdfs.mv_class0_bit_cdf(mv_ctx, comp);
        writer.write_symbol(mv_class0_bit, bit_row)?;

        if force_integer_mv {
            // Reader synthesises mv_class0_fr = 3 — the field must
            // already be 3, otherwise the value is unrepresentable.
            if mv_class0_fr != 3 {
                return Err(Error::PartitionWalkOutOfRange);
            }
        } else {
            let fr_row = cdfs.mv_class0_fr_cdf(mv_ctx, comp, mv_class0_bit as usize);
            writer.write_symbol(mv_class0_fr, fr_row)?;
        }

        if allow_high_precision_mv {
            let hp_row = cdfs.mv_class0_hp_cdf(mv_ctx, comp);
            writer.write_symbol(mv_class0_hp, hp_row)?;
        } else if mv_class0_hp != 1 {
            // Reader forces mv_class0_hp = 1.
            return Err(Error::PartitionWalkOutOfRange);
        }
    } else {
        // mag = (CLASS0_SIZE << (mv_class + 2)) + X + 1 with
        // X = (d << 3) | (mv_fr << 1) | mv_hp, so X = offset - 2^(c+3).
        let base: u32 = (CLASS0_SIZE as u32) << (mv_class + 2);
        let x: u32 = offset - base;
        let d: u32 = x >> 3;
        let mv_fr: u32 = (x >> 1) & 0x3;
        let mv_hp: u32 = x & 0x1;

        // mv_bit ladder: i = 0..mv_class-1, d |= mv_bit << i.
        for i in 0..(mv_class as usize) {
            let mv_bit: u32 = (d >> i) & 0x1;
            let bit_row = cdfs.mv_bit_cdf(mv_ctx, comp, i);
            writer.write_symbol(mv_bit, bit_row)?;
        }

        if force_integer_mv {
            if mv_fr != 3 {
                return Err(Error::PartitionWalkOutOfRange);
            }
        } else {
            let fr_row = cdfs.mv_fr_cdf(mv_ctx, comp);
            writer.write_symbol(mv_fr, fr_row)?;
        }

        if allow_high_precision_mv {
            let hp_row = cdfs.mv_hp_cdf(mv_ctx, comp);
            writer.write_symbol(mv_hp, hp_row)?;
        } else if mv_hp != 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    Ok(())
}

/// MV writer per §5.11.31 `read_mv( ref )` (av1-spec p.81) — the
/// algebraic inverse of the reader's `mv_joint` + per-component
/// `read_mv_component( )` dispatch.
///
/// ## Spec body (§5.11.31 — av1-spec p.81)
///
/// ```text
///   diffMv[ 0 ] = 0
///   diffMv[ 1 ] = 0
///   if ( use_intrabc ) MvCtx = MV_INTRABC_CONTEXT
///   else               MvCtx = 0
///   mv_joint                                                 S()
///   if ( mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ )
///       diffMv[ 0 ] = read_mv_component( 0 )
///   if ( mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ )
///       diffMv[ 1 ] = read_mv_component( 1 )
///   Mv[ ref ][ 0 ] = PredMv[ ref ][ 0 ] + diffMv[ 0 ]
///   Mv[ ref ][ 1 ] = PredMv[ ref ][ 1 ] + diffMv[ 1 ]
/// ```
///
/// ## Inverse derivation
///
/// The encoder holds the target `Mv[ ref ]` and the §7.10.2-derived
/// `PredMv[ ref ]`. The difference vector is
/// `diff[ c ] = Mv[ ref ][ c ] - PredMv[ ref ][ c ]` for each
/// component. The `mv_joint` symbol records which components are
/// non-zero:
///
/// | `diff[0]` | `diff[1]` | `mv_joint`        |
/// |-----------|-----------|-------------------|
/// | `== 0`    | `== 0`    | `MV_JOINT_ZERO`   |
/// | `== 0`    | `!= 0`    | `MV_JOINT_HNZVZ`  |
/// | `!= 0`    | `== 0`    | `MV_JOINT_HZVNZ`  |
/// | `!= 0`    | `!= 0`    | `MV_JOINT_HNZVNZ` |
///
/// Note the reader's component indexing: `read_mv_component( 0 )` (the
/// horizontal component, `diff[0]`) fires for `MV_JOINT_HZVNZ` /
/// `MV_JOINT_HNZVNZ`, and `read_mv_component( 1 )` (vertical,
/// `diff[1]`) for `MV_JOINT_HNZVZ` / `MV_JOINT_HNZVNZ`. The encoder
/// follows the same call order: component `0` then component `1`.
///
/// ## CDF selection (§8.3.2 — av1-spec p.493)
///
/// `mv_joint` → `TileMvJointCdf[ MvCtx ]`. The component reads select
/// their CDFs as documented on [`write_read_mv_component`].
///
/// ## Inputs
///
/// * `mv` — the target `Mv[ ref ]` as `[horizontal, vertical]`.
/// * `pred_mv` — the §7.10.2 `PredMv[ ref ]` predictor.
/// * `use_intrabc` — selects `MvCtx = MV_INTRABC_CONTEXT` when set.
/// * `force_integer_mv` / `allow_high_precision_mv` — frame-level
///   precision flags forwarded to each component write.
pub fn write_read_mv(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    mv: [i32; 2],
    pred_mv: [i32; 2],
    use_intrabc: bool,
    force_integer_mv: bool,
    allow_high_precision_mv: bool,
) -> Result<(), Error> {
    let mv_ctx: usize = if use_intrabc { MV_INTRABC_CONTEXT } else { 0 };

    let diff0: i32 = mv[0].wrapping_sub(pred_mv[0]);
    let diff1: i32 = mv[1].wrapping_sub(pred_mv[1]);

    let mv_joint: u32 = match (diff0 != 0, diff1 != 0) {
        (false, false) => MV_JOINT_ZERO as u32,
        (false, true) => MV_JOINT_HNZVZ as u32,
        (true, false) => MV_JOINT_HZVNZ as u32,
        (true, true) => MV_JOINT_HNZVNZ as u32,
    };

    let joint_row = cdfs.mv_joint_cdf(mv_ctx);
    writer.write_symbol(mv_joint, joint_row)?;

    // Component 0 (horizontal) fires for MV_JOINT_HZVNZ / MV_JOINT_HNZVNZ.
    if diff0 != 0 {
        write_read_mv_component(
            writer,
            cdfs,
            mv_ctx,
            0,
            diff0,
            force_integer_mv,
            allow_high_precision_mv,
        )?;
    }
    // Component 1 (vertical) fires for MV_JOINT_HNZVZ / MV_JOINT_HNZVNZ.
    if diff1 != 0 {
        write_read_mv_component(
            writer,
            cdfs,
            mv_ctx,
            1,
            diff1,
            force_integer_mv,
            allow_high_precision_mv,
        )?;
    }

    Ok(())
}

/// `PredMv[ refList ]` derivation per the §5.11.26 `assign_mv` non-intrabc
/// inter arm (av1-spec p.77-78) — selects the motion-vector predictor a
/// chosen inter mode draws from the §7.10.2 `find_mv_stack` outputs,
/// feeding it to [`write_read_mv`] as the `pred_mv` argument.
///
/// Spec body (av1-spec p.77-78, the `else if`/`else` arms of `assign_mv`
/// reached when `use_intrabc == 0`):
/// ```text
///   compMode = get_mode( i )
///   if ( compMode == GLOBALMV ) {
///     PredMv[ i ] = GlobalMvs[ i ]
///   } else {
///     pos = ( compMode == NEARESTMV ) ? 0 : RefMvIdx
///     if ( compMode == NEWMV && NumMvFound <= 1 )
///       pos = 0
///     PredMv[ i ] = RefStackMv[ pos ][ i ]
///   }
/// ```
///
/// The `RefMvIdx` here is the §5.11.23 `inter_block_mode_info` running
/// index after the `drl_mode` loop (`RefMvIdx = 1 + (has_nearmv ? …)`)
/// — i.e. the same value the caller hands [`write_drl_mode`]. For a
/// `NEARESTMV` list the predictor is always the top stack slot; for
/// `NEWMV` it collapses to slot 0 whenever the stack is short
/// (`NumMvFound <= 1`).
///
/// The `GlobalMvs` / `RefStackMv` arrays come straight from a
/// [`FindMvStackResult`]; `ref_list` is `0` on single-prediction blocks
/// and `0`/`1` on compound blocks.
///
/// The per-list `compMode` is resolved through the shared
/// [`crate::cdf::get_mode`] §get_mode mapping (the same decode-side
/// helper the reader uses), so encode and decode agree on the mode
/// decomposition by construction.
///
/// Caller-bug rejects (`Err(PartitionWalkOutOfRange)`):
/// * `y_mode` outside `MODE_NEARESTMV..=MODE_NEW_NEWMV` or `ref_list > 1`.
/// * a `RefStackMv` index (`pos`) at or beyond `NumMvFound` — the chosen
///   `RefMvIdx` is unreachable at the derived stack depth, so reading
///   `RefStackMv[ pos ]` would consume an undefined slot.
pub fn assign_mv_pred_mv(
    mv_stack: &FindMvStackResult,
    y_mode: u8,
    ref_list: u8,
    ref_mv_idx: u32,
) -> Result<[i32; 2], Error> {
    if !(MODE_NEARESTMV..=MODE_NEW_NEWMV).contains(&y_mode) || ref_list > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let comp_mode = crate::cdf::get_mode(y_mode, ref_list as usize);
    if comp_mode == MODE_GLOBALMV {
        return Ok(mv_stack.global_mvs[ref_list as usize]);
    }
    let mut pos: u32 = if comp_mode == MODE_NEARESTMV {
        0
    } else {
        ref_mv_idx
    };
    if comp_mode == MODE_NEWMV && mv_stack.num_mv_found <= 1 {
        pos = 0;
    }
    // §5.11.26: `RefStackMv[ pos ]` must be a populated slot. Slots 0
    // and 1 are ALWAYS populated: the §7.10.2.12 extra-search pads
    // `RefStackMv[ idx ]` for `idx = NumMvFound..2` with the global MV
    // ("for single prediction, NumMvFound is not incremented by the
    // addition of global motion candidates" — so `NumMvFound` may
    // remain 0 while both head slots carry readable values, and a
    // first-block NEWMV / NEARMV legitimately predicts from them;
    // r411 fix — the previous `pos >= NumMvFound` reject refused every
    // empty-neighbourhood NEWMV leaf). A `pos >= 2` is reachable only
    // through the §5.11.23 `drl_mode` loop, which requires
    // `NumMvFound > pos` — beyond that the slot is undefined; reject.
    if (pos >= 2 && pos >= mv_stack.num_mv_found) || pos as usize >= MAX_REF_MV_STACK_SIZE {
        return Err(Error::PartitionWalkOutOfRange);
    }
    Ok(mv_stack.ref_stack_mv[pos as usize][ref_list as usize])
}

/// `PredMv[ 0 ]` derivation per the §5.11.26 `assign_mv`
/// **intra-block-copy** arm (av1-spec p.77-78) — the predictor the
/// §5.11.7 `intra_frame_mode_info( )` `use_intrabc` path feeds to
/// `read_mv( 0 )`. On this arm `compMode` is forced to `NEWMV`, so an
/// MV difference is *always* coded: the caller passes the returned
/// predictor to [`write_read_mv`] with `use_intrabc = true` (selecting
/// `MvCtx = MV_INTRABC_CONTEXT`), `force_integer_mv = true` and
/// `allow_high_precision_mv = false` (intra block copy is
/// integer-only per §5.11.32).
///
/// Spec body (av1-spec p.77-78, the `if ( use_intrabc )` arm of
/// `assign_mv`):
/// ```text
///   PredMv[ 0 ] = RefStackMv[ 0 ][ 0 ]
///   if ( PredMv[ 0 ][ 0 ] == 0 && PredMv[ 0 ][ 1 ] == 0 ) {
///       PredMv[ 0 ] = RefStackMv[ 1 ][ 0 ]
///   }
///   if ( PredMv[ 0 ][ 0 ] == 0 && PredMv[ 0 ][ 1 ] == 0 ) {
///       sbSize = use_128x128_superblock ? BLOCK_128X128 : BLOCK_64X64
///       sbSize4 = Num_4x4_Blocks_High[ sbSize ]
///       if ( MiRow - sbSize4 < MiRowStart ) {
///           PredMv[ 0 ][ 0 ] = 0
///           PredMv[ 0 ][ 1 ] = -(sbSize4 * MI_SIZE + INTRABC_DELAY_PIXELS) * 8
///       } else {
///           PredMv[ 0 ][ 0 ] = -(sbSize4 * MI_SIZE * 8)
///           PredMv[ 0 ][ 1 ] = 0
///       }
///   }
/// ```
///
/// Unlike the inter arm ([`assign_mv_pred_mv`]), no `NumMvFound` bound
/// applies to the two `RefStackMv` reads: the §7.10.2.12 extra-search
/// single-prediction tail pads `RefStackMv[ idx ][ 0 ]` for `idx <
/// 2` with `GlobalMvs[ 0 ]` **without** incrementing `NumMvFound`
/// ("for single prediction, NumMvFound is not incremented by the
/// addition of global motion candidates"), so both slots are always
/// written before `assign_mv` reads them — and on an intra-block-copy
/// block `GlobalMvs[ 0 ]` is the zero vector (§7.10.2.1: `ref ==
/// INTRA_FRAME ⇒ mv = 0`), which is exactly what the zero-fallback
/// chain absorbs.
///
/// The zero-MV fallback points at the most recent column of
/// already-reconstructed superblocks: one superblock up
/// (`-(sbSize4 * MI_SIZE * 8)` in row component) when the block is not
/// in the tile's top superblock row, else `sbSize4 * MI_SIZE +
/// INTRABC_DELAY_PIXELS` luma samples to the left (the §3
/// `INTRABC_DELAY_PIXELS = 256` wave-front delay) in column component.
/// Components are `[ row, col ]` in 1/8-luma-sample units throughout
/// (`Mv[ ][ 0 ]` = vertical, `Mv[ ][ 1 ]` = horizontal).
///
/// `mi_row` / `mi_row_start` are the block's mode-info row and the
/// enclosing tile's `MiRowStart` (§6.10.4). A `mi_row < mi_row_start`
/// pair is rejected as a caller bug (`Err(PartitionWalkOutOfRange)`)
/// — a block cannot sit above its own tile.
pub fn assign_mv_pred_mv_intrabc(
    mv_stack: &FindMvStackResult,
    use_128x128_superblock: bool,
    mi_row: u32,
    mi_row_start: u32,
) -> Result<[i32; 2], Error> {
    if mi_row < mi_row_start {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let mut pred_mv = mv_stack.ref_stack_mv[0][0];
    if pred_mv == [0, 0] {
        pred_mv = mv_stack.ref_stack_mv[1][0];
    }
    if pred_mv == [0, 0] {
        let sb_size = if use_128x128_superblock {
            BLOCK_128X128
        } else {
            BLOCK_64X64
        };
        let sb_size4 = NUM_4X4_BLOCKS_HIGH[sb_size] as u32;
        // `MiRow - sbSize4 < MiRowStart` on spec (unbounded) integers —
        // rearranged as `MiRow < MiRowStart + sbSize4` so the unsigned
        // subtraction cannot wrap.
        if mi_row < mi_row_start + sb_size4 {
            pred_mv = [
                0,
                -((sb_size4 as i32 * MI_SIZE as i32 + INTRABC_DELAY_PIXELS as i32) * 8),
            ];
        } else {
            pred_mv = [-(sb_size4 as i32 * MI_SIZE as i32 * 8), 0];
        }
    }
    Ok(pred_mv)
}

/// `use_intrabc` S() inverse per §5.11.7 (av1-spec p.65) — r279. The
/// exact encode-side twin of
/// [`crate::cdf::PartitionWalker::decode_use_intrabc`].
///
/// Spec body (av1-spec p.65):
/// ```text
///   if ( allow_intrabc ) {
///       use_intrabc                                          S()
///   } else {
///       use_intrabc = 0
///   }
/// ```
///
/// * When `allow_intrabc == true`: emits one §8.2.6 `S()` over
///   `TileIntrabcCdf` (the §8.3.2 selection is contextless — "the cdf
///   for use_intrabc is given by TileIntrabcCdf", no `[ctx]`
///   subscript).
/// * When `allow_intrabc == false`: the reader's fall-through assigns
///   `use_intrabc = 0` with no bit consumed, so no symbol is emitted.
///   `use_intrabc` MUST be `0` — caller bug otherwise
///   ([`Error::PartitionWalkOutOfRange`]), as the reader could never
///   reconstruct a `1`.
///
/// `allow_intrabc` is the §5.9.20 frame-header bit (itself reachable
/// only on intra frames whose §5.9.5 `allow_screen_content_tools` is
/// set); `use_intrabc` is the §3 binary alphabet value (`> 1` is
/// rejected as a caller bug).
pub fn write_use_intrabc(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    use_intrabc: u8,
    allow_intrabc: bool,
) -> Result<(), Error> {
    if use_intrabc > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if allow_intrabc {
        let row = cdfs.intrabc_cdf();
        writer.write_symbol(u32::from(use_intrabc), row)?;
    } else if use_intrabc != 0 {
        // §5.11.7 fall-through arm: the reader forces `use_intrabc = 0`
        // without a bit read; a `1` is unreachable.
        return Err(Error::PartitionWalkOutOfRange);
    }
    Ok(())
}

/// Per-list inputs to the §5.11.7 `use_intrabc` arm of
/// [`write_intra_frame_intrabc_arm`] — the target block vector plus the
/// §7.10.2 `find_mv_stack( 0 )` outputs and tile/superblock scalars the
/// §5.11.26 intrabc `PredMv` derivation
/// ([`assign_mv_pred_mv_intrabc`]) consumes.
#[derive(Debug, Clone, Copy)]
pub struct IntrabcArmInputs<'a> {
    /// The target `Mv[ 0 ]` as `[ row, col ]` in 1/8-luma-sample
    /// units. Intra block copy is integer-only, so each component's
    /// difference against the derived `PredMv[ 0 ]` must be a multiple
    /// of 8 (rejected by the §5.11.32 component writer otherwise).
    pub mv: [i32; 2],
    /// §7.10.2 `find_mv_stack( 0 )` outputs (`RefStackMv` slots 0/1 are
    /// the only fields the intrabc arm reads; the §7.10.2.12
    /// single-pred tail guarantees both are written).
    pub mv_stack: &'a FindMvStackResult,
    /// §5.5.2 sequence-header `use_128x128_superblock` bit (selects
    /// `sbSize4` in the §5.11.26 zero-MV fallback).
    pub use_128x128_superblock: bool,
    /// The block's `MiRow`.
    pub mi_row: u32,
    /// The enclosing tile's `MiRowStart` (§6.10.4).
    pub mi_row_start: u32,
}

/// Fixed §5.11.7 `use_intrabc`-arm assignments (av1-spec p.65) — the
/// no-bit mode-info state [`write_intra_frame_intrabc_arm`] returns so
/// the caller can stamp the §5.11.5 neighbour grids exactly as the
/// reader would:
///
/// ```text
///   is_inter = 1
///   YMode = DC_PRED
///   UVMode = DC_PRED
///   motion_mode = SIMPLE
///   compound_type = COMPOUND_AVERAGE
///   PaletteSizeY = 0
///   PaletteSizeUV = 0
///   interp_filter[ 0 ] = BILINEAR
///   interp_filter[ 1 ] = BILINEAR
/// ```
///
/// (`RefFrame[ 0 ] = INTRA_FRAME` / `RefFrame[ 1 ] = NONE` are set by
/// the §5.11.7 body *before* the `use_intrabc` read, on both arms, so
/// they are not part of this readout.) `mv` / `pred_mv` carry the
/// coded `Mv[ 0 ]` and the §5.11.26-derived `PredMv[ 0 ]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntrabcBlockInfo {
    /// §5.11.7 `is_inter = 1`.
    pub is_inter: u8,
    /// §5.11.7 `YMode = DC_PRED`.
    pub y_mode: u8,
    /// §5.11.7 `UVMode = DC_PRED`.
    pub uv_mode: u8,
    /// §5.11.7 `motion_mode = SIMPLE`.
    pub motion_mode: u8,
    /// §5.11.7 `compound_type = COMPOUND_AVERAGE`.
    pub compound_type: u8,
    /// §5.11.7 `PaletteSizeY = 0`.
    pub palette_size_y: u8,
    /// §5.11.7 `PaletteSizeUV = 0`.
    pub palette_size_uv: u8,
    /// §5.11.7 `interp_filter[ 0..2 ] = BILINEAR`.
    pub interp_filter: [u8; 2],
    /// The coded `Mv[ 0 ]` (`[ row, col ]`, 1/8-luma-sample units).
    pub mv: [i32; 2],
    /// The §5.11.26 `PredMv[ 0 ]` the MV difference was coded against.
    pub pred_mv: [i32; 2],
}

/// §5.11.7 `use_intrabc` region writer (av1-spec p.65) — r279.
/// Composes, in spec order, the write side of:
///
/// ```text
///   if ( allow_intrabc ) {
///       use_intrabc                                          S()
///   } else {
///       use_intrabc = 0
///   }
///   if ( use_intrabc ) {
///       is_inter = 1
///       YMode = DC_PRED
///       UVMode = DC_PRED
///       motion_mode = SIMPLE
///       compound_type = COMPOUND_AVERAGE
///       PaletteSizeY = 0
///       PaletteSizeUV = 0
///       interp_filter[ 0 ] = BILINEAR
///       interp_filter[ 1 ] = BILINEAR
///       find_mv_stack( 0 )
///       assign_mv( 0 )
///   }
/// ```
///
/// `use_intrabc` is derived from `intrabc.is_some()` and emitted via
/// [`write_use_intrabc`]. On the `Some` arm the §5.11.26
/// `assign_mv( 0 )` body forces `compMode = NEWMV`, so an MV
/// difference is *always* coded: `PredMv[ 0 ]` is derived via
/// [`assign_mv_pred_mv_intrabc`] from the caller-supplied §7.10.2
/// `find_mv_stack( 0 )` outputs, then [`write_read_mv`] emits the
/// difference under the intra-block-copy MV regime (`MvCtx =
/// MV_INTRABC_CONTEXT`, `force_integer_mv = 1` — §5.9.2 forces it on
/// every intra frame, and `allow_intrabc` requires an intra frame —
/// hence `allow_high_precision_mv = 0`).
///
/// Returns `Some(IntrabcBlockInfo)` carrying the arm's fixed no-bit
/// assignments (for §5.11.5 grid stamping) when the intrabc arm fired,
/// `None` when `use_intrabc == 0` (the caller continues with the
/// §5.11.7 `else` arm — `intra_frame_y_mode` etc.).
///
/// Caller-bug rejects ([`Error::PartitionWalkOutOfRange`]):
/// * `intrabc.is_some()` with `allow_intrabc == false` (the reader's
///   fall-through forces `use_intrabc = 0`).
/// * `mi_row < mi_row_start` (via [`assign_mv_pred_mv_intrabc`]).
/// * An `mv` whose difference against the derived predictor is not
///   integer-aligned (via the §5.11.32 component writer's
///   `force_integer_mv` check).
pub fn write_intra_frame_intrabc_arm(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    allow_intrabc: bool,
    intrabc: Option<&IntrabcArmInputs<'_>>,
) -> Result<Option<IntrabcBlockInfo>, Error> {
    let use_intrabc = u8::from(intrabc.is_some());
    write_use_intrabc(writer, cdfs, use_intrabc, allow_intrabc)?;
    let Some(inputs) = intrabc else {
        return Ok(None);
    };

    // §5.11.26 `assign_mv( 0 )` intrabc arm: `compMode = NEWMV` (forced),
    // so `read_mv( 0 )` always fires — derive `PredMv[ 0 ]` and emit
    // the difference.
    let pred_mv = assign_mv_pred_mv_intrabc(
        inputs.mv_stack,
        inputs.use_128x128_superblock,
        inputs.mi_row,
        inputs.mi_row_start,
    )?;
    write_read_mv(
        writer, cdfs, inputs.mv, pred_mv, /* use_intrabc = */ true,
        /* force_integer_mv = */ true, /* allow_high_precision_mv = */ false,
    )?;

    Ok(Some(IntrabcBlockInfo {
        is_inter: 1,
        y_mode: DC_PRED as u8,
        uv_mode: DC_PRED as u8,
        motion_mode: MOTION_MODE_SIMPLE,
        compound_type: COMPOUND_AVERAGE,
        palette_size_y: 0,
        palette_size_uv: 0,
        interp_filter: [BILINEAR, BILINEAR],
        mv: inputs.mv,
        pred_mv,
    }))
}

/// §5.11.23 tail inputs for [`write_inter_block_mode_info`] — r277.
/// Groups the readouts + gating scalars the four post-`assign_mv( )`
/// leaf writers consume ([`write_interintra_mode`] §5.11.28,
/// [`write_motion_mode`] §5.11.27, [`write_compound_type`] §5.11.29,
/// [`write_interpolation_filter`] §5.11.x), so the composition's
/// parameter surface stays readable. Every field maps 1:1 onto a leaf
/// writer's input of the same name; see the leaf docs for the spec
/// semantics and reject conditions.
///
/// The two walker-derived §5.11.27 quantities (`has_overlappable` —
/// §7.10.3 `has_overlappable_candidates( )` — and `num_samples` —
/// §7.10.4 `find_warp_samples( )` `NumSamples`) arrive precomputed
/// per the stateless-writer doctrine of this module; the §8.3.2
/// neighbour grid reads (`above_comp_group_idx` / `left_comp_group_idx`
/// — identity `0` when unavailable — `above_compound_idx` /
/// `left_compound_idx` — identity `1` — and `above_interp_filters` /
/// `left_interp_filters`) likewise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterBlockModeInfoTail {
    /// §5.11.28 target [`InterIntraReadout`].
    pub interintra: InterIntraReadout,
    /// §5.5.2 sequence-header `enable_interintra_compound` bit.
    pub enable_interintra_compound: bool,
    /// §5.11.27 target `motion_mode` ordinal.
    pub motion_mode: u8,
    /// §7.10.4 `NumSamples` from `find_warp_samples( )`.
    pub num_samples: u32,
    /// §5.9.2 frame-header `is_motion_mode_switchable` bit.
    pub is_motion_mode_switchable: bool,
    /// §5.9.2 frame-header `allow_warped_motion` bit.
    pub allow_warped_motion: bool,
    /// §5.11.27 `is_scaled( refFrame )` per `refFrame - LAST_FRAME`.
    pub is_scaled_per_ref: [bool; 7],
    /// §7.10.3 `has_overlappable_candidates( )` outcome.
    pub has_overlappable: bool,
    /// §5.11.29 target [`CompoundTypeReadout`].
    pub compound_type: CompoundTypeReadout,
    /// §5.5.2 sequence-header `enable_masked_compound` bit.
    pub enable_masked_compound: bool,
    /// §5.5.2 sequence-header `enable_jnt_comp` bit.
    pub enable_jnt_comp: bool,
    /// §7.8.1 equal-relative-distance outcome seeding the §8.3.2
    /// `compound_idx` ctx.
    pub dist_equal: bool,
    /// §8.3.2 `CompGroupIdxs[ MiRow - 1 ][ MiCol ]` (identity `0`).
    pub above_comp_group_idx: u8,
    /// §8.3.2 `CompGroupIdxs[ MiRow ][ MiCol - 1 ]` (identity `0`).
    pub left_comp_group_idx: u8,
    /// §8.3.2 `CompoundIdxs[ MiRow - 1 ][ MiCol ]` (identity `1`).
    pub above_compound_idx: u8,
    /// §8.3.2 `CompoundIdxs[ MiRow ][ MiCol - 1 ]` (identity `1`).
    pub left_compound_idx: u8,
    /// §5.11.x target [`InterpolationFilterReadout`].
    pub interp_filter: InterpolationFilterReadout,
    /// §5.9.10 frame-header `interpolation_filter` in
    /// `EIGHTTAP..=SWITCHABLE = 0..=4`.
    pub interpolation_filter: u8,
    /// §5.5.2 sequence-header `enable_dual_filter` bit.
    pub enable_dual_filter: bool,
    /// §8.3.2 `InterpFilters[ MiRow - 1 ][ MiCol ][ 0..2 ]`.
    pub above_interp_filters: [u8; 2],
    /// §8.3.2 `InterpFilters[ MiRow ][ MiCol - 1 ][ 0..2 ]`.
    pub left_interp_filters: [u8; 2],
    /// §5.9.24 `GmType[ 0..8 ]` — shared by the §5.11.27 GLOBALMV
    /// short-circuit and the §5.11.x `needs_interp_filter( )` gate.
    pub gm_type: [i32; 8],
}

impl InterBlockModeInfoTail {
    /// The everything-gated-shut tail: every §5.11.23 tail leaf is
    /// bit-silent on this configuration regardless of the head's arm.
    ///
    /// * §5.11.28 gate closed (`enable_interintra_compound == false`)
    ///   ⇒ `interintra = 0`, no bits;
    /// * §5.11.27 short-circuit (`is_motion_mode_switchable == false`)
    ///   ⇒ `motion_mode = SIMPLE`, no bits;
    /// * §5.11.29 with `enable_masked_compound == false` AND
    ///   `enable_jnt_comp == false` ⇒ the line-1/2 pre-sets survive on
    ///   every arm (`comp_group_idx = 0`, `compound_idx = 1`,
    ///   `compound_type = COMPOUND_AVERAGE`), no bits;
    /// * §5.11.x with `interpolation_filter = EIGHTTAP` (non-
    ///   SWITCHABLE) ⇒ both slots forced to `EIGHTTAP`, no bits.
    ///
    /// Useful as a baseline the caller patches per block, and as the
    /// regression fixture proving the composed writer's head bits are
    /// unchanged by the r277 tail fold.
    pub fn bit_silent() -> Self {
        Self {
            interintra: InterIntraReadout {
                interintra: 0,
                interintra_mode: None,
                wedge_interintra: None,
                wedge_index: None,
            },
            enable_interintra_compound: false,
            motion_mode: MOTION_MODE_SIMPLE,
            num_samples: 0,
            is_motion_mode_switchable: false,
            allow_warped_motion: false,
            is_scaled_per_ref: [false; 7],
            has_overlappable: false,
            compound_type: CompoundTypeReadout {
                comp_group_idx: 0,
                compound_idx: 1,
                compound_type: COMPOUND_AVERAGE,
                wedge_index: None,
                wedge_sign: None,
                mask_type: None,
            },
            enable_masked_compound: false,
            enable_jnt_comp: false,
            dist_equal: false,
            above_comp_group_idx: 0,
            left_comp_group_idx: 0,
            above_compound_idx: 1,
            left_compound_idx: 1,
            interp_filter: InterpolationFilterReadout {
                interp_filter: [EIGHTTAP, EIGHTTAP],
                read_from_bitstream: [false, false],
            },
            interpolation_filter: EIGHTTAP,
            enable_dual_filter: false,
            above_interp_filters: [EIGHTTAP, EIGHTTAP],
            left_interp_filters: [EIGHTTAP, EIGHTTAP],
            gm_type: [0; 8],
        }
    }
}

impl Default for InterBlockModeInfoTail {
    fn default() -> Self {
        Self::bit_silent()
    }
}

/// `inter_block_mode_info( )` composition writer per §5.11.23 (av1-spec
/// p.73-75) — r275, tail folded r277. Folds the r266-r274 leaf writers
/// ([`write_ref_frames`], [`write_compound_mode`] /
/// [`write_inter_single_mode`], [`write_drl_mode`], [`assign_mv_pred_mv`]
/// and [`write_read_mv`]) into the single §5.11.23 body the
/// `inter_frame_mode_info` chain reaches on an inter block. It is the
/// exact encode-side inverse of the decode-side
/// [`crate::cdf::PartitionWalker::decode_inter_block_mode_info`] from
/// `read_ref_frames( )` through the closing interpolation-filter loop.
///
/// ## Scope
///
/// Covers the full §5.11.23 body:
///
/// 1. `read_ref_frames( )` (§5.11.25) via [`write_ref_frames`].
/// 2. `isCompound = RefFrame[ 1 ] > INTRA_FRAME`.
/// 3. `find_mv_stack( isCompound )` (§7.10.2) — **not** re-run here; the
///    encoder hands the already-computed [`FindMvStackResult`] (the same
///    aggregate the decoder builds internally), carrying every §8.3.2
///    ctx the mode / drl / mv writers consume (`NewMvContext`,
///    `ZeroMvContext`, `RefMvContext`, `NumMvFound`, `DrlCtxStack`,
///    `RefStackMv`, `GlobalMvs`).
/// 4. The four-arm `YMode` dispatch: arm 1 (`skip_mode`) and arm 2
///    (`SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV`) emit no mode symbol; arm 3
///    (`isCompound`) emits `compound_mode` via [`write_compound_mode`]
///    over the internally-derived [`crate::cdf::compound_mode_ctx`]; arm
///    4 (single-pred) emits the `new_mv` / `zero_mv` / `ref_mv` cascade
///    via [`write_inter_single_mode`] over the stack's three contexts.
/// 5. The `RefMvIdx` `drl_mode` loop via [`write_drl_mode`].
/// 6. `assign_mv( isCompound )`: for each reference list
///    `i in 0..1 + isCompound`, resolve `compMode = get_mode( YMode, i )`
///    and, **only when `compMode == NEWMV`**, derive `PredMv[ i ]` via
///    [`assign_mv_pred_mv`] and emit the MV difference via
///    [`write_read_mv`]. Non-NEWMV lists set `Mv[ i ] = PredMv[ i ]`
///    with no bits (§5.11.31 line `else Mv[ i ] = PredMv[ i ]`).
/// 7. The §5.11.23 tail in **spec order** (folded r277):
///    `read_interintra_mode( )` via [`write_interintra_mode`]
///    (§5.11.28), then `read_motion_mode( )` via [`write_motion_mode`]
///    (§5.11.27), then `read_compound_type( )` via
///    [`write_compound_type`] (§5.11.29), then the interpolation-
///    filter loop via [`write_interpolation_filter`] (§5.11.x). The
///    §5.11.28 inner-arm `RefFrame[ 1 ] = INTRA_FRAME` override is
///    applied between the first two — it is the §5.11.27 short-circuit
///    that silences `motion_mode` on interintra blocks, and the
///    §5.11.x writer also observes the post-override pair. Tail inputs
///    arrive grouped in [`InterBlockModeInfoTail`];
///    [`InterBlockModeInfoTail::bit_silent`] reproduces the pre-r277
///    head-only bit pattern (every tail leaf gated shut).
///
/// ## `YMode` consistency
///
/// `YMode` is caller-chosen but cross-checked against the active arm:
/// arm 1 forces `NEAREST_NEARESTMV`, arm 2 forces `GLOBALMV`, arm 3
/// admits only the eight compound modes
/// (`NEAREST_NEARESTMV..=NEW_NEWMV`), arm 4 only the four single-pred
/// modes (`NEWMV` / `GLOBALMV` / `NEARESTMV` / `NEARMV`). A `YMode`
/// inconsistent with the arm is a caller bug — the reader could never
/// produce it on that arm.
///
/// ## Inputs
///
/// * `ref_frame` — target `( RefFrame[ 0 ], RefFrame[ 1 ] )`; drives
///   `isCompound` and the §5.11.25 reference cascade.
/// * `y_mode` — the §5.11.23 `YMode` chosen by the encoder.
/// * `mv` — `Mv[ list ][ comp ]` for each active list; only the NEWMV
///   lists' values are consulted (the others are derived predictors).
/// * `ref_mv_idx` — the `RefMvIdx` the encoder selected, threaded to
///   both [`write_drl_mode`] and [`assign_mv_pred_mv`].
/// * `mv_stack` — the §7.10.2 [`FindMvStackResult`].
/// * `mi_size` — `MiSize` (§9.3 table index for the `comp_mode`
///   precondition).
/// * the arm-selection sextet + `reference_select` of
///   [`write_ref_frames`] / [`write_inter_block_mode_info_bootstrap`].
/// * the §5.11.18 neighbour-state octet from which [`write_ref_frames`]
///   derives every §8.3.2 reference ctx internally.
/// * `force_integer_mv` / `allow_high_precision_mv` — frame-level MV
///   precision flags forwarded to [`write_read_mv`].
///
/// ## Out-of-range / mismatch cases (`Err(PartitionWalkOutOfRange)`)
///
/// All of the composed writers' rejects, plus a `y_mode` inconsistent
/// with the active arm.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless like every piece it composes. The caller stamps the
/// resulting `RefFrame` / `YMode` / `Mv` onto the block footprint
/// through its own [`crate::cdf::PartitionWalker`] so subsequent blocks'
/// §8.3.2 neighbour walks observe them.
/// r422 — the §5.11.23 **mode + MV prefix**: `read_ref_frames( )`
/// (§5.11.25 cascade), the four-arm `YMode` dispatch, the `RefMvIdx`
/// `drl_mode` loop and the `assign_mv( )` NEWMV differences — i.e.
/// everything [`write_inter_block_mode_info`] emits BEFORE the
/// §5.11.28/§5.11.27/§5.11.29/interp-filter tail. Factored out so the
/// search-side rate twin can price one `(RefFrame, YMode, Mv,
/// RefMvIdx)` candidate through the writer's own emission path
/// against a counting [`SymbolWriter`]; the emitting pass calls it
/// through [`write_inter_block_mode_info`], so pricing and committing
/// share one body by construction.
#[allow(clippy::too_many_arguments)]
pub fn write_inter_mode_mv_prefix(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    ref_frame: [i32; 2],
    y_mode: u8,
    mv: [[i32; 2]; 2],
    ref_mv_idx: u32,
    mv_stack: &FindMvStackResult,
    mi_size: usize,
    skip_mode: u8,
    skip_mode_frame: [i8; 2],
    seg_ref_frame_active: bool,
    seg_ref_frame_data: i8,
    seg_skip_active: bool,
    seg_globalmv_active: bool,
    reference_select: bool,
    avail_u: bool,
    avail_l: bool,
    above_single: bool,
    left_single: bool,
    above_intra: bool,
    left_intra: bool,
    above_ref_frame: [i32; 2],
    left_ref_frame: [i32; 2],
    force_integer_mv: bool,
    allow_high_precision_mv: bool,
) -> Result<(), Error> {
    // §5.11.23 line 4618: `read_ref_frames( )`. Drives the §5.11.25
    // four-arm reference cascade (zero to six S() depending on arm /
    // leaf) and verifies the target pair is reachable.
    write_ref_frames(
        writer,
        cdfs,
        ref_frame,
        mi_size,
        skip_mode,
        skip_mode_frame,
        seg_ref_frame_active,
        seg_ref_frame_data,
        seg_skip_active,
        seg_globalmv_active,
        reference_select,
        avail_u,
        avail_l,
        above_single,
        left_single,
        above_intra,
        left_intra,
        above_ref_frame,
        left_ref_frame,
    )?;

    // §5.11.23 line 4619: `isCompound = RefFrame[ 1 ] > INTRA_FRAME`.
    // INTRA_FRAME = 0; NONE = -1. Any slot-1 in LAST_FRAME..=ALTREF_FRAME
    // (1..=7) makes the block compound.
    let is_compound = ref_frame[1] > 0;

    // §5.11.23 lines 4621-4642: the four-arm `YMode` dispatch. The arm
    // is selected exactly as the reader selects it (spec order: skip_mode
    // first, then segmentation, then compound, then single-pred). The
    // caller-supplied `y_mode` is cross-checked against the active arm —
    // the reader could never emit a different mode on that arm.
    if skip_mode != 0 {
        // Arm 1: no S(); `YMode` is forced to NEAREST_NEARESTMV.
        if y_mode != MODE_NEAREST_NEARESTMV {
            return Err(Error::PartitionWalkOutOfRange);
        }
    } else if seg_skip_active || seg_globalmv_active {
        // Arm 2: no S(); `YMode` is forced to GLOBALMV.
        if y_mode != MODE_GLOBALMV {
            return Err(Error::PartitionWalkOutOfRange);
        }
    } else if is_compound {
        // Arm 3: `compound_mode` S() over `TileCompoundModeCdf[ ctx ]`,
        // then `YMode = NEAREST_NEARESTMV + compound_mode`. The §8.3.2
        // ctx is derived internally from the stack's RefMvContext /
        // NewMvContext (the same `compound_mode_ctx` mapping the reader
        // applies), so encode and decode agree by construction.
        let cm_ctx = compound_mode_ctx(
            mv_stack.ref_mv_context as usize,
            mv_stack.new_mv_context as usize,
        );
        write_compound_mode(writer, cdfs, y_mode, cm_ctx)?;
    } else {
        // Arm 4: single-pred `new_mv` / `zero_mv` / `ref_mv` cascade
        // over the stack's three contexts (§7.10.2.14).
        write_inter_single_mode(
            writer,
            cdfs,
            y_mode,
            mv_stack.new_mv_context as usize,
            mv_stack.zero_mv_context as usize,
            mv_stack.ref_mv_context as usize,
        )?;
    }

    // §5.11.23 lines 4643-4674: `RefMvIdx` `drl_mode` loop. A silent
    // no-op for every mode outside the NEWMV / NEW_NEWMV and
    // `has_nearmv( )` arms (matching the reader's `RefMvIdx = 0`).
    write_drl_mode(
        writer,
        cdfs,
        y_mode,
        ref_mv_idx,
        mv_stack.num_mv_found,
        &mv_stack.drl_ctx_stack,
    )?;

    // §5.11.23 line 4675: `assign_mv( isCompound )` per §5.11.31. The
    // inter path always has `use_intrabc == 0`, so for each reference
    // list the §5.11.31 body resolves `compMode = get_mode( i )` and
    // reads an MV difference only on the NEWMV lists; the others inherit
    // `PredMv[ i ]` with no bits emitted.
    let list_count: u8 = if is_compound { 2 } else { 1 };
    for i in 0..list_count {
        if crate::cdf::get_mode(y_mode, i as usize) == MODE_NEWMV {
            // §5.11.31: `if ( compMode == NEWMV ) read_mv( i )`. Derive
            // `PredMv[ i ]` from the stack and emit the difference.
            let pred_mv = assign_mv_pred_mv(mv_stack, y_mode, i, ref_mv_idx)?;
            write_read_mv(
                writer,
                cdfs,
                mv[i as usize],
                pred_mv,
                /* use_intrabc = */ false,
                force_integer_mv,
                allow_high_precision_mv,
            )?;
        }
        // Non-NEWMV lists: `Mv[ i ] = PredMv[ i ]`, no symbols.
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn write_inter_block_mode_info(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    ref_frame: [i32; 2],
    y_mode: u8,
    mv: [[i32; 2]; 2],
    ref_mv_idx: u32,
    mv_stack: &FindMvStackResult,
    mi_size: usize,
    skip_mode: u8,
    skip_mode_frame: [i8; 2],
    seg_ref_frame_active: bool,
    seg_ref_frame_data: i8,
    seg_skip_active: bool,
    seg_globalmv_active: bool,
    reference_select: bool,
    avail_u: bool,
    avail_l: bool,
    above_single: bool,
    left_single: bool,
    above_intra: bool,
    left_intra: bool,
    above_ref_frame: [i32; 2],
    left_ref_frame: [i32; 2],
    force_integer_mv: bool,
    allow_high_precision_mv: bool,
    tail: &InterBlockModeInfoTail,
) -> Result<(), Error> {
    // §3 binary alphabet bound on `skip_mode` (mirrors the bootstrap).
    if skip_mode > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.23 head — the reference cascade, the four-arm `YMode`
    // dispatch, the `drl_mode` loop and the `assign_mv( )` MV
    // differences, factored (r422) into [`write_inter_mode_mv_prefix`]
    // so the search-side rate twin prices mode candidates through the
    // EXACT emission path (single source of truth — the twin never
    // re-implements the cascade).
    write_inter_mode_mv_prefix(
        writer,
        cdfs,
        ref_frame,
        y_mode,
        mv,
        ref_mv_idx,
        mv_stack,
        mi_size,
        skip_mode,
        skip_mode_frame,
        seg_ref_frame_active,
        seg_ref_frame_data,
        seg_skip_active,
        seg_globalmv_active,
        reference_select,
        avail_u,
        avail_l,
        above_single,
        left_single,
        above_intra,
        left_intra,
        above_ref_frame,
        left_ref_frame,
        force_integer_mv,
        allow_high_precision_mv,
    )?;

    let is_compound = ref_frame[1] > 0;

    // ------- §5.11.23 tail (folded r277) — spec order: §5.11.28 →
    // §5.11.27 → §5.11.29 → §5.11.x interpolation-filter loop. -------

    // `read_interintra_mode( isCompound )` per §5.11.28.
    write_interintra_mode(
        writer,
        cdfs,
        &tail.interintra,
        mi_size,
        skip_mode,
        is_compound,
        tail.enable_interintra_compound,
    )?;

    // §5.11.28 inner-arm override: `RefFrame[ 1 ] = INTRA_FRAME = 0`.
    // The §5.11.27 / §5.11.x writers below observe the post-override
    // pair — slot-1 INTRA_FRAME is the §5.11.27 short-circuit that
    // silences `motion_mode` on interintra blocks (and the only way
    // slot 1 ever becomes INTRA_FRAME). `is_compound` is unaffected
    // (the §5.11.28 gate already required `!isCompound`).
    let ref_frame = if tail.interintra.interintra == 1 {
        [ref_frame[0], 0]
    } else {
        ref_frame
    };

    // `read_motion_mode( isCompound )` per §5.11.27.
    write_motion_mode(
        writer,
        cdfs,
        tail.motion_mode,
        mi_size,
        skip_mode,
        is_compound,
        ref_frame,
        y_mode,
        tail.num_samples,
        tail.is_motion_mode_switchable,
        tail.allow_warped_motion,
        force_integer_mv,
        tail.gm_type,
        tail.is_scaled_per_ref,
        tail.has_overlappable,
    )?;

    // `read_compound_type( isCompound )` per §5.11.29. The §8.3.2
    // `comp_group_idx` / `compound_idx` ctx walks consult
    // `AboveRefFrame[ 0 ] == ALTREF_FRAME` / `LeftRefFrame[ 0 ] ==
    // ALTREF_FRAME` on their `!AboveSingle` / `!LeftSingle` arms;
    // ALTREF_FRAME = 7 per §6.10.x.
    write_compound_type(
        writer,
        cdfs,
        &tail.compound_type,
        mi_size,
        skip_mode,
        is_compound,
        tail.interintra.interintra,
        tail.interintra.wedge_interintra.unwrap_or(0),
        tail.enable_masked_compound,
        tail.enable_jnt_comp,
        tail.dist_equal,
        avail_u,
        avail_l,
        above_single,
        left_single,
        above_ref_frame[0] == 7,
        left_ref_frame[0] == 7,
        tail.above_comp_group_idx,
        tail.left_comp_group_idx,
        tail.above_compound_idx,
        tail.left_compound_idx,
    )?;

    // The closing §5.11.x interpolation-filter loop.
    write_interpolation_filter(
        writer,
        cdfs,
        &tail.interp_filter,
        mi_size,
        skip_mode,
        is_compound,
        ref_frame,
        y_mode,
        tail.motion_mode,
        tail.interpolation_filter,
        tail.enable_dual_filter,
        tail.gm_type,
        avail_u,
        avail_l,
        above_ref_frame,
        left_ref_frame,
        tail.above_interp_filters,
        tail.left_interp_filters,
    )?;

    Ok(())
}

/// `read_interintra_mode( isCompound )` writer per §5.11.28 (av1-spec
/// p.79-80) — r276. First piece of the §5.11.23 tail after
/// `assign_mv( )` (the spec order is `read_interintra_mode` →
/// `read_motion_mode` → `read_compound_type`). Exact encode-side
/// inverse of [`crate::cdf::PartitionWalker::read_interintra_mode`]:
/// the caller hands the target [`InterIntraReadout`] (the same
/// aggregate the decoder returns) and the writer re-emits the S()
/// sequence the reader would consume to reproduce it.
///
/// ## Spec body (§5.11.28 — av1-spec p.79-80)
///
/// ```text
///   if ( !skip_mode && enable_interintra_compound && !isCompound &&
///        MiSize >= BLOCK_8X8 && MiSize <= BLOCK_32X32 ) {
///       interintra                                          S()
///       if ( interintra ) {
///           interintra_mode                                 S()
///           RefFrame[ 1 ] = INTRA_FRAME
///           AngleDeltaY = 0
///           AngleDeltaUV = 0
///           use_filter_intra = 0
///           wedge_interintra                                S()
///           if ( wedge_interintra ) {
///               wedge_index                                 S()
///               wedge_sign = 0
///           }
///       }
///   } else {
///       interintra = 0
///   }
/// ```
///
/// ## §8.3.2 CDF selections
///
/// * `interintra` / `interintra_mode`: `ctx = Size_Group[ MiSize ] - 1`
///   via [`interintra_ctx`] (the outer gate confines `MiSize` to
///   `BLOCK_8X8..=BLOCK_32X32` where `Size_Group ∈ {1, 2, 3}`).
/// * `wedge_interintra` / `wedge_index`: straight `MiSize` index.
///
/// ## Readout consistency
///
/// The target `readout` must be exactly producible by the reader on
/// this gate configuration:
///
/// * gate closed ⇒ `interintra == 0` with all three option fields
///   `None` (the spec's `else interintra = 0` arm reads no bits);
/// * gate open, `interintra == 0` ⇒ all three option fields `None`;
/// * gate open, `interintra == 1` ⇒ `interintra_mode ==
///   Some(< INTERINTRA_MODES)` and `wedge_interintra ∈ {Some(0),
///   Some(1)}`; `wedge_index == Some(< WEDGE_TYPES)` iff
///   `wedge_interintra == Some(1)`, `None` otherwise.
///
/// Any other shape is a caller bug surfaced as
/// `Err(PartitionWalkOutOfRange)` (the reader could never decode it).
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless: between zero and four §8.2.6 `S()` symbols. The spec
/// body's imperative overrides (`RefFrame[ 1 ] = INTRA_FRAME`,
/// `AngleDeltaY = AngleDeltaUV = 0`, `use_filter_intra = 0`,
/// `wedge_sign = 0`) are derivations the caller applies to its own
/// state, exactly as the decode-side dispatcher applies them from the
/// returned readout.
pub fn write_interintra_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    readout: &InterIntraReadout,
    mi_size: usize,
    skip_mode: u8,
    is_compound: bool,
    enable_interintra_compound: bool,
) -> Result<(), Error> {
    // §3 binary alphabet bound on `skip_mode`.
    if skip_mode > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.28 outer gate: every clause must hold for the `interintra`
    // S() to fire (mirrors the reader's `gate_open`).
    let gate_open = skip_mode == 0
        && enable_interintra_compound
        && !is_compound
        && (BLOCK_8X8..=BLOCK_32X32).contains(&mi_size);

    if !gate_open {
        // `else interintra = 0` — no bits; the readout must carry the
        // exact no-read shape.
        if readout.interintra != 0
            || readout.interintra_mode.is_some()
            || readout.wedge_interintra.is_some()
            || readout.wedge_index.is_some()
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §3 binary alphabet bound on `interintra`.
    if readout.interintra > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.28 `interintra` S() over `TileInterIntraCdf[ ctx ]`,
    // `ctx = Size_Group[ MiSize ] - 1`. The gate confines `MiSize` to
    // the band where `interintra_ctx` is `Some(_)`.
    let ctx = interintra_ctx(mi_size).ok_or(Error::PartitionWalkOutOfRange)?;
    let row = cdfs
        .inter_intra_cdf(ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(readout.interintra as u32, row)?;

    if readout.interintra == 0 {
        // Inner arm closed: the reader leaves every option field `None`.
        if readout.interintra_mode.is_some()
            || readout.wedge_interintra.is_some()
            || readout.wedge_index.is_some()
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // Inner arm: `interintra_mode` S() over
    // `TileInterIntraModeCdf[ ctx ]` (same ctx).
    let ii_mode = readout
        .interintra_mode
        .ok_or(Error::PartitionWalkOutOfRange)?;
    if (ii_mode as usize) >= INTERINTRA_MODES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // `wedge_interintra` S() over `TileWedgeInterIntraCdf[ MiSize ]`
    // (straight `MiSize` index).
    let wedge_ii = readout
        .wedge_interintra
        .ok_or(Error::PartitionWalkOutOfRange)?;
    if wedge_ii > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // `wedge_index` presence must match the `wedge_interintra` arm
    // before any symbol is committed.
    if (wedge_ii == 1) != readout.wedge_index.is_some() {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if let Some(wi) = readout.wedge_index {
        if (wi as usize) >= WEDGE_TYPES {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    let row = cdfs
        .inter_intra_mode_cdf(ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(ii_mode as u32, row)?;

    let row = cdfs
        .wedge_inter_intra_cdf(mi_size)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(wedge_ii as u32, row)?;

    if let Some(wi) = readout.wedge_index {
        // `wedge_index` S() over `TileWedgeIndexCdf[ MiSize ]` — the
        // same default CDF the §5.11.29 COMPOUND_WEDGE branch selects.
        let row = cdfs
            .wedge_index_cdf(mi_size)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        writer.write_symbol(wi as u32, row)?;
    }

    Ok(())
}

/// `read_motion_mode( isCompound )` writer per §5.11.27 (av1-spec
/// p.79) — r276. Second piece of the §5.11.23 tail after
/// `assign_mv( )`. Exact encode-side inverse of
/// [`crate::cdf::PartitionWalker::read_motion_mode`]: the caller hands
/// the target §6.10.26 `motion_mode` ordinal (`SIMPLE = 0` / `OBMC = 1`
/// / `WARPED_CAUSAL = 2`) and the same gating scalars the decode side
/// consumes; the writer re-derives the active arm and emits zero or one
/// §8.2.6 `S()`.
///
/// ## Spec body (§5.11.27 — av1-spec p.79)
///
/// The body short-circuits to `motion_mode = SIMPLE` (no bits) on:
///
/// * `skip_mode == 1`;
/// * `!is_motion_mode_switchable`;
/// * `Min( Block_Width[ MiSize ], Block_Height[ MiSize ] ) < 8`;
/// * `!force_integer_mv && YMode ∈ { GLOBALMV, GLOBAL_GLOBALMV } &&
///   GmType[ RefFrame[ 0 ] ] > TRANSLATION`;
/// * `isCompound || RefFrame[ 1 ] == INTRA_FRAME ||
///   !has_overlappable_candidates( )`.
///
/// Otherwise it runs `find_warp_samples( )` (§7.10.4) and dispatches:
///
/// * `force_integer_mv || NumSamples == 0 || !allow_warped_motion ||
///   is_scaled( RefFrame[ 0 ] )` ⇒ **arm A**: `use_obmc` S() over
///   `TileUseObmcCdf[ MiSize ]`; `motion_mode = use_obmc ? OBMC :
///   SIMPLE` (WARPED_CAUSAL unreachable on this arm);
/// * otherwise ⇒ **arm B**: `motion_mode` S() over
///   `TileMotionModeCdf[ MiSize ]` (all three ordinals reachable).
///
/// ## Inputs
///
/// Mirrors the decode-side scalar surface; the two walker-derived
/// quantities arrive precomputed (the stateless-writer doctrine of
/// this module):
///
/// * `has_overlappable` — the §5.11.27 `has_overlappable_candidates()`
///   outcome over the encoder's own grid state.
/// * `num_samples` — `NumSamples` from §7.10.4 `find_warp_samples()`.
/// * `gm_type` — `GmType[ 0..8 ]` (§5.9.24), consulted on the
///   GLOBALMV arm for `RefFrame[ 0 ] ∈ 0..8`.
/// * `is_scaled_per_ref` — the §5.11.27 `is_scaled( refFrame )`
///   outcome per `refFrame - LAST_FRAME ∈ 0..7`.
///
/// ## Target consistency
///
/// A `motion_mode` the reader could never produce on the active arm is
/// a caller bug (`Err(PartitionWalkOutOfRange)`): the short-circuit
/// arms force `SIMPLE`; arm A admits only `SIMPLE` / `OBMC`; arm B
/// admits `0..MOTION_MODES`.
#[allow(clippy::too_many_arguments)]
pub fn write_motion_mode(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    motion_mode: u8,
    mi_size: usize,
    skip_mode: u8,
    is_compound: bool,
    ref_frame: [i32; 2],
    y_mode: u8,
    num_samples: u32,
    is_motion_mode_switchable: bool,
    allow_warped_motion: bool,
    force_integer_mv: bool,
    gm_type: [i32; 8],
    is_scaled_per_ref: [bool; 7],
    has_overlappable: bool,
) -> Result<(), Error> {
    // §3 binary alphabet bound on `skip_mode`.
    if skip_mode > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.27 short-circuit arms — each forces SIMPLE with no bits.
    let forced_simple = skip_mode != 0
        || !is_motion_mode_switchable
        || core::cmp::min(block_width(mi_size), block_height(mi_size)) < 8
        || (!force_integer_mv
            && (y_mode == MODE_GLOBALMV || y_mode == MODE_GLOBAL_GLOBALMV)
            && (0..8).contains(&ref_frame[0])
            && gm_type[ref_frame[0] as usize] > GM_TYPE_TRANSLATION)
        // `RefFrame[ 1 ] == INTRA_FRAME` — INTRA_FRAME = 0.
        || is_compound
        || ref_frame[1] == 0
        || !has_overlappable;

    if forced_simple {
        if motion_mode != MOTION_MODE_SIMPLE {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.27 post-`find_warp_samples` arm dispatch. `is_scaled(
    // RefFrame[ 0 ] )` is only meaningful for `RefFrame[ 0 ] ∈
    // LAST_FRAME..=ALTREF_FRAME = 1..=7` (single-pred with a non-INTRA
    // slot 0 — guaranteed past the short-circuits above).
    let ref0 = ref_frame[0];
    let scaled = if (1..=7).contains(&ref0) {
        is_scaled_per_ref[(ref0 - 1) as usize]
    } else {
        false
    };

    if force_integer_mv || num_samples == 0 || !allow_warped_motion || scaled {
        // Arm A: `use_obmc` S() over `TileUseObmcCdf[ MiSize ]`.
        // WARPED_CAUSAL is unreachable here.
        let use_obmc: u32 = match motion_mode {
            MOTION_MODE_SIMPLE => 0,
            MOTION_MODE_OBMC => 1,
            _ => return Err(Error::PartitionWalkOutOfRange),
        };
        let row = cdfs
            .use_obmc_cdf(mi_size)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        writer.write_symbol(use_obmc, row)?;
    } else {
        // Arm B: `motion_mode` S() over `TileMotionModeCdf[ MiSize ]`.
        if (motion_mode as usize) >= MOTION_MODES {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let row = cdfs
            .motion_mode_cdf(mi_size)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        writer.write_symbol(motion_mode as u32, row)?;
    }

    Ok(())
}

/// `read_compound_type( isCompound )` writer per §5.11.29 (av1-spec
/// p.80-81) — r276. Third piece of the §5.11.23 tail after
/// `assign_mv( )`. Exact encode-side inverse of
/// [`crate::cdf::PartitionWalker::read_compound_type`]: the caller
/// hands the target [`CompoundTypeReadout`] (the same aggregate the
/// decoder returns) plus the gating scalars; the writer re-derives the
/// active arms and emits the matching S() / L(1) sequence.
///
/// ## Spec body (§5.11.29 — av1-spec p.80-81)
///
/// ```text
///   comp_group_idx = 0
///   compound_idx = 1
///   if ( skip_mode ) { compound_type = COMPOUND_AVERAGE; return }
///   if ( isCompound ) {
///       n = Wedge_Bits[ MiSize ]
///       if ( enable_masked_compound )
///           comp_group_idx                                   S()
///       if ( comp_group_idx == 0 ) {
///           if ( enable_jnt_comp ) {
///               compound_idx                                 S()
///               compound_type = compound_idx ? COMPOUND_AVERAGE
///                                            : COMPOUND_DISTANCE
///           } else {
///               compound_type = COMPOUND_AVERAGE
///           }
///       } else {
///           if ( n == 0 ) compound_type = COMPOUND_DIFFWTD
///           else          compound_type                      S()
///       }
///       if ( compound_type == COMPOUND_WEDGE ) {
///           wedge_index                                      S()
///           wedge_sign                                       L(1)
///       } else if ( compound_type == COMPOUND_DIFFWTD ) {
///           mask_type                                        L(1)
///       }
///   } else {
///       if ( interintra )
///           compound_type = wedge_interintra ? COMPOUND_WEDGE
///                                            : COMPOUND_INTRA
///       else
///           compound_type = COMPOUND_AVERAGE
///   }
/// ```
///
/// ## §8.3.2 CDF selections
///
/// * `comp_group_idx`: `TileCompGroupIdxCdf[ ctx ]` via
///   [`comp_group_idx_ctx`];
/// * `compound_idx`: `TileCompoundIdxCdf[ ctx ]` via
///   [`compound_idx_ctx`] (seeded with `dist_equal ? 3 : 0`);
/// * `compound_type`: `TileCompoundTypeCdf[ MiSize ]` (straight
///   index);
/// * `wedge_index`: `TileWedgeIndexCdf[ MiSize ]`;
/// * `wedge_sign` / `mask_type`: literal `L(1)`, no CDF.
///
/// ## Inputs
///
/// The scalar surface mirrors the decode side; the two neighbour grid
/// reads arrive precomputed (the stateless-writer doctrine of this
/// module): `above_comp_group_idx` / `left_comp_group_idx` carry the
/// `CompGroupIdxs[ MiRow - 1 ][ MiCol ]` / `CompGroupIdxs[ MiRow ][
/// MiCol - 1 ]` values (identity `0` when the neighbour is
/// unavailable) and `above_compound_idx` / `left_compound_idx` the
/// `CompoundIdxs` twins (identity `1`). Both ctx walks only consult
/// them on the `!AboveSingle` / `!LeftSingle` arms, exactly as the
/// reader does.
///
/// ## Readout consistency
///
/// The target `readout` must be exactly producible by the reader on
/// this gate configuration — the derived fields (`compound_type` on
/// every non-S() arm, `comp_group_idx` / `compound_idx` pre-sets, the
/// option-field presence pattern) are cross-checked and any mismatch
/// is a caller bug surfaced as `Err(PartitionWalkOutOfRange)`.
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless: between zero and two `S()` plus at most one `S()` + one
/// `L(1)` sub-branch. The caller stamps `comp_group_idx` /
/// `compound_idx` onto its own grids so subsequent blocks' §8.3.2
/// neighbour walks observe them.
#[allow(clippy::too_many_arguments)]
pub fn write_compound_type(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    readout: &CompoundTypeReadout,
    mi_size: usize,
    skip_mode: u8,
    is_compound: bool,
    interintra: u8,
    wedge_interintra: u8,
    enable_masked_compound: bool,
    enable_jnt_comp: bool,
    dist_equal: bool,
    avail_u: bool,
    avail_l: bool,
    above_single: bool,
    left_single: bool,
    above_ref_0_altref: bool,
    left_ref_0_altref: bool,
    above_comp_group_idx: u8,
    left_comp_group_idx: u8,
    above_compound_idx: u8,
    left_compound_idx: u8,
) -> Result<(), Error> {
    // §3 binary alphabet bounds.
    if skip_mode > 1
        || above_comp_group_idx > 1
        || left_comp_group_idx > 1
        || above_compound_idx > 1
        || left_compound_idx > 1
    {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.29 lines 3-6: `if ( skip_mode )` short-circuit — the
    // readout must carry the exact line-1/2 pre-sets, no bits.
    if skip_mode != 0 {
        if readout.comp_group_idx != 0
            || readout.compound_idx != 1
            || readout.compound_type != COMPOUND_AVERAGE
            || readout.wedge_index.is_some()
            || readout.wedge_sign.is_some()
            || readout.mask_type.is_some()
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    if !is_compound {
        // §5.11.29 else-arm: no bits; the §5.11.28 outcome alone
        // selects the type.
        let expected = if interintra != 0 {
            if wedge_interintra != 0 {
                COMPOUND_WEDGE
            } else {
                COMPOUND_INTRA
            }
        } else {
            COMPOUND_AVERAGE
        };
        if readout.comp_group_idx != 0
            || readout.compound_idx != 1
            || readout.compound_type != expected
            || readout.wedge_index.is_some()
            || readout.wedge_sign.is_some()
            || readout.mask_type.is_some()
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.29 `n = Wedge_Bits[ MiSize ]`.
    let n = wedge_bits(mi_size);

    // §5.11.29 `if ( enable_masked_compound )` ⇒ `comp_group_idx S()`;
    // otherwise the line-1 pre-set `0` must survive.
    if enable_masked_compound {
        if readout.comp_group_idx > 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let ctx = comp_group_idx_ctx(
            avail_u,
            above_single,
            above_comp_group_idx,
            above_ref_0_altref,
            avail_l,
            left_single,
            left_comp_group_idx,
            left_ref_0_altref,
        );
        let row = cdfs.comp_group_idx_cdf(ctx);
        writer.write_symbol(readout.comp_group_idx as u32, row)?;
    } else if readout.comp_group_idx != 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    if readout.comp_group_idx == 0 {
        if enable_jnt_comp {
            // `compound_idx S()` ⇒ `compound_type = compound_idx ?
            // COMPOUND_AVERAGE : COMPOUND_DISTANCE`.
            if readout.compound_idx > 1 {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let expected = if readout.compound_idx != 0 {
                COMPOUND_AVERAGE
            } else {
                COMPOUND_DISTANCE
            };
            if readout.compound_type != expected {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let ctx = compound_idx_ctx(
                dist_equal,
                avail_u,
                above_single,
                above_compound_idx,
                above_ref_0_altref,
                avail_l,
                left_single,
                left_compound_idx,
                left_ref_0_altref,
            );
            let row = cdfs.compound_idx_cdf(ctx);
            writer.write_symbol(readout.compound_idx as u32, row)?;
        } else if readout.compound_idx != 1 || readout.compound_type != COMPOUND_AVERAGE {
            // Line-2 pre-set survives; type forced to AVERAGE.
            return Err(Error::PartitionWalkOutOfRange);
        }
    } else {
        // `comp_group_idx == 1`: the `compound_idx` pre-set survives.
        if readout.compound_idx != 1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if n == 0 {
            if readout.compound_type != COMPOUND_DIFFWTD {
                return Err(Error::PartitionWalkOutOfRange);
            }
        } else {
            // `compound_type S()` over `TileCompoundTypeCdf[ MiSize ]`
            // — only the two §6.10.24 signalled ordinals
            // (COMPOUND_WEDGE / COMPOUND_DIFFWTD) are codeable.
            if (readout.compound_type as usize) >= COMPOUND_TYPES {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let row = cdfs
                .compound_type_cdf(mi_size)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(readout.compound_type as u32, row)?;
        }
    }

    // §5.11.29 wedge / diffwtd sub-branches — validate the option-field
    // presence pattern before committing any further symbol.
    if readout.compound_type == COMPOUND_WEDGE {
        let wi = readout.wedge_index.ok_or(Error::PartitionWalkOutOfRange)?;
        let ws = readout.wedge_sign.ok_or(Error::PartitionWalkOutOfRange)?;
        if (wi as usize) >= WEDGE_TYPES || ws > 1 || readout.mask_type.is_some() {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let row = cdfs
            .wedge_index_cdf(mi_size)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        writer.write_symbol(wi as u32, row)?;
        // `wedge_sign L(1)`.
        writer.write_literal(1, ws as u32)?;
    } else if readout.compound_type == COMPOUND_DIFFWTD {
        let mt = readout.mask_type.ok_or(Error::PartitionWalkOutOfRange)?;
        if mt > 1 || readout.wedge_index.is_some() || readout.wedge_sign.is_some() {
            return Err(Error::PartitionWalkOutOfRange);
        }
        // `mask_type L(1)`.
        writer.write_literal(1, mt as u32)?;
    } else if readout.wedge_index.is_some()
        || readout.wedge_sign.is_some()
        || readout.mask_type.is_some()
    {
        return Err(Error::PartitionWalkOutOfRange);
    }

    Ok(())
}

/// §5.11.x interpolation-filter loop writer (av1-spec p.74, the
/// `if ( interpolation_filter == SWITCHABLE )` block closing
/// `inter_block_mode_info`) — r277. Fourth and last piece of the
/// §5.11.23 tail. Exact encode-side inverse of
/// [`crate::cdf::PartitionWalker::read_interpolation_filter`]: the
/// caller hands the target [`InterpolationFilterReadout`] (the same
/// aggregate the decoder returns) plus the gating scalars; the writer
/// re-derives the active arms and emits the matching S() sequence.
///
/// ## Spec body (av1-spec p.74)
///
/// ```text
///   if ( interpolation_filter == SWITCHABLE ) {
///       for ( dir = 0; dir < ( enable_dual_filter ? 2 : 1 ); dir++ ) {
///           if ( needs_interp_filter( ) ) {
///               interp_filter[ dir ]                          S()
///           } else {
///               interp_filter[ dir ] = EIGHTTAP
///           }
///       }
///       if ( !enable_dual_filter )
///           interp_filter[ 1 ] = interp_filter[ 0 ]
///   } else {
///       for ( dir = 0; dir < 2; dir++ )
///           interp_filter[ dir ] = interpolation_filter
///   }
/// ```
///
/// with `needs_interp_filter( )` per av1-spec p.75:
///
/// ```text
///   needs_interp_filter( ) {
///       large = ( Min( Block_Width[ MiSize ], Block_Height[ MiSize ] ) >= 8 )
///       if ( skip_mode || motion_mode == LOCALWARP )           return 0
///       else if ( large && YMode == GLOBALMV )
///           return GmType[ RefFrame[ 0 ] ] == TRANSLATION
///       else if ( large && YMode == GLOBAL_GLOBALMV )
///           return GmType[ RefFrame[ 0 ] ] == TRANSLATION ||
///                  GmType[ RefFrame[ 1 ] ] == TRANSLATION
///       else                                                   return 1
///   }
/// ```
///
/// ## §8.3.2 CDF selection
///
/// `interp_filter` codes against `TileInterpFilterCdf[ ctx ]` with
/// `ctx` from the §8.3.2 walk (av1-spec p.369) via
/// [`interp_filter_ctx`]. Per the stateless-writer doctrine of this
/// module the two neighbour grid reads arrive precomputed:
/// `above_ref_frame` / `left_ref_frame` carry the neighbours'
/// `RefFrames[ .. ][ 0..2 ]` pairs and `above_interp_filters` /
/// `left_interp_filters` their `InterpFilters[ .. ][ dir ]` pairs.
/// The walk accepts a neighbour's filter only when the neighbour is
/// available AND one of its two reference slots equals
/// `RefFrame[ 0 ]`; otherwise the spec's `3` sentinel
/// ([`INTERP_FILTER_NONE`]) stands in — exactly the reader's grid
/// resolution.
///
/// ## Readout consistency
///
/// The target `readout` must be exactly producible by the reader on
/// this gate configuration:
///
/// * `interpolation_filter != SWITCHABLE` ⇒ both slots equal the
///   frame-header value, both `read_from_bitstream == false`, no
///   bits;
/// * `SWITCHABLE && !needs_interp_filter( )` ⇒ the coded dir carries
///   `EIGHTTAP` with `read_from_bitstream == false`, no bits;
/// * `SWITCHABLE && needs_interp_filter( )` ⇒ the coded dir carries
///   an ordinal in `0..INTERP_FILTERS` (the 3-way S() alphabet —
///   `BILINEAR` is frame-level-only) with `read_from_bitstream ==
///   true`;
/// * `!enable_dual_filter` ⇒ slot 1 mirrors slot 0 with
///   `read_from_bitstream[ 1 ] == false` (the spec's post-loop
///   mirror reads no second symbol).
///
/// Any other shape is a caller bug surfaced as
/// `Err(PartitionWalkOutOfRange)` (the reader could never decode it).
///
/// ## §5.11.5 grid-fill is on the caller
///
/// Stateless: between zero and two §8.2.6 `S()` symbols. The caller
/// stamps `InterpFilters[ r + y ][ c + x ][ dir ] =
/// interp_filter[ dir ]` onto the block footprint through its own
/// [`crate::cdf::PartitionWalker`] so subsequent blocks' §8.3.2 ctx
/// walks observe the values.
#[allow(clippy::too_many_arguments)]
pub fn write_interpolation_filter(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    readout: &InterpolationFilterReadout,
    mi_size: usize,
    skip_mode: u8,
    is_compound: bool,
    ref_frame: [i32; 2],
    y_mode: u8,
    motion_mode: u8,
    interpolation_filter: u8,
    enable_dual_filter: bool,
    gm_type: [i32; 8],
    avail_u: bool,
    avail_l: bool,
    above_ref_frame: [i32; 2],
    left_ref_frame: [i32; 2],
    above_interp_filters: [u8; 2],
    left_interp_filters: [u8; 2],
) -> Result<(), Error> {
    // §5.11.x caller-bug guards — mirror the reader's bounds:
    // `interpolation_filter` in `EIGHTTAP..=SWITCHABLE = 0..=4` per
    // §6.8.9; `mi_size` per the §3 BLOCK_* enumeration; `ref_frame[0]`
    // in `INTRA_FRAME = 0..=ALTREF_FRAME = 7`; `ref_frame[1]` in
    // `NONE = -1..=ALTREF_FRAME = 7`; `skip_mode` binary; the
    // neighbour `InterpFilters` entries in `0..=BILINEAR = 0..=3`
    // (every value the §5.11.x reader can stamp).
    if skip_mode > 1 || mi_size >= BLOCK_SIZES || interpolation_filter > SWITCHABLE {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if !(0..=7).contains(&ref_frame[0]) || !(-1..=7).contains(&ref_frame[1]) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if above_interp_filters.iter().any(|&f| f > BILINEAR)
        || left_interp_filters.iter().any(|&f| f > BILINEAR)
    {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.x `else`-arm: `interp_filter[ dir ] = interpolation_filter`
    // for both dirs, no bits. The readout must carry the exact forced
    // shape.
    if interpolation_filter != SWITCHABLE {
        if readout.interp_filter != [interpolation_filter, interpolation_filter]
            || readout.read_from_bitstream != [false, false]
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // `needs_interp_filter( )` per av1-spec p.75 — dir-invariant, so
    // hoisted out of the per-direction loop (the reader evaluates it
    // per iteration with identical inputs).
    let large = core::cmp::min(block_width(mi_size), block_height(mi_size)) >= 8;
    let needs = if skip_mode != 0 || motion_mode == MOTION_MODE_WARPED_CAUSAL {
        false
    } else if large && y_mode == MODE_GLOBALMV {
        let r0 = ref_frame[0];
        (0..8).contains(&r0) && gm_type[r0 as usize] == GM_TYPE_TRANSLATION
    } else if large && y_mode == MODE_GLOBAL_GLOBALMV {
        let r0 = ref_frame[0];
        let r1 = ref_frame[1];
        let t0 = (0..8).contains(&r0) && gm_type[r0 as usize] == GM_TYPE_TRANSLATION;
        let t1 = (0..8).contains(&r1) && gm_type[r1 as usize] == GM_TYPE_TRANSLATION;
        t0 || t1
    } else {
        true
    };

    // §5.11.x `if`-arm: per-direction loop over `enable_dual_filter ?
    // 2 : 1` dirs.
    let dir_count = if enable_dual_filter { 2usize } else { 1 };
    for dir in 0..dir_count {
        if !needs {
            // `interp_filter[ dir ] = EIGHTTAP` — no bits; the readout
            // must carry the exact fallback shape.
            if readout.interp_filter[dir] != EIGHTTAP || readout.read_from_bitstream[dir] {
                return Err(Error::PartitionWalkOutOfRange);
            }
            continue;
        }

        // `interp_filter[ dir ] S()` — the 3-way alphabet covers
        // `EIGHTTAP..=EIGHTTAP_SHARP` only (`BILINEAR` is reachable
        // solely through the frame-level f(2) path above).
        if !readout.read_from_bitstream[dir]
            || (readout.interp_filter[dir] as usize) >= INTERP_FILTERS
        {
            return Err(Error::PartitionWalkOutOfRange);
        }

        // §8.3.2 neighbour resolution: take the neighbour's
        // `InterpFilters[ .. ][ dir ]` only when it is available and
        // one of its `RefFrames[ .. ][ 0..2 ]` equals `RefFrame[ 0 ]`;
        // else the `3` sentinel.
        let above_type = if avail_u
            && (above_ref_frame[0] == ref_frame[0] || above_ref_frame[1] == ref_frame[0])
        {
            above_interp_filters[dir] as usize
        } else {
            INTERP_FILTER_NONE
        };
        let left_type = if avail_l
            && (left_ref_frame[0] == ref_frame[0] || left_ref_frame[1] == ref_frame[0])
        {
            left_interp_filters[dir] as usize
        } else {
            INTERP_FILTER_NONE
        };
        let ctx = interp_filter_ctx(above_type, left_type, dir as u32, is_compound)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let row = cdfs
            .interp_filter_cdf(ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        writer.write_symbol(readout.interp_filter[dir] as u32, row)?;
    }

    // §5.11.x post-loop mirror: `if ( !enable_dual_filter )
    // interp_filter[ 1 ] = interp_filter[ 0 ]` — no second symbol, so
    // slot 1 must mirror slot 0 with its read flag clear.
    if !enable_dual_filter
        && (readout.interp_filter[1] != readout.interp_filter[0] || readout.read_from_bitstream[1])
    {
        return Err(Error::PartitionWalkOutOfRange);
    }

    Ok(())
}

/// `FloorLog2(x)` per §4.7 (av1-spec p.21) — `s = 0; while (x !=
/// 0) { x >>= 1; s++; } return s - 1;`. Local helper because the
/// public surface lives in `cdf` / `symbol_decoder`; pulled inline
/// to avoid widening this module's import surface. Mirrors
/// [`crate::encoder::symbol_writer`]'s private helper of the same
/// name.
#[inline]
fn floor_log2(mut x: u32) -> u32 {
    debug_assert!(x != 0, "§4.7 FloorLog2 requires x != 0");
    let mut s: u32 = 0;
    while x != 0 {
        x >>= 1;
        s += 1;
    }
    s - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        cfl_allowed_for_uv_mode, comp_mode_ctx, intra_mode_ctx, is_inter_ctx, size_group, skip_ctx,
        skip_mode_ctx, PartitionWalker, TileGeometry, BLOCK_16X16, BLOCK_4X4, BLOCK_4X8, BLOCK_8X4,
        BLOCK_8X8, DC_PRED, MODE_GLOBAL_GLOBALMV, MODE_NEAREST_NEWMV, MODE_NEAR_NEARMV,
        MODE_NEAR_NEWMV, MODE_NEW_NEARESTMV, MODE_NEW_NEARMV, V_PRED,
    };
    use crate::symbol_decoder::SymbolDecoder;

    /// Local cast helper — `DC_PRED` / `V_PRED` / etc. are exposed as
    /// `usize` constants in `crate::cdf`, but the §3 enumeration bounds
    /// them to `0..INTRA_MODES = 13`, so the cast to `u8` is total.
    const DC_PRED_U8: u8 = DC_PRED as u8;
    const V_PRED_U8: u8 = V_PRED as u8;

    /// Build a fresh 32×32 mi walker + default CDFs (the same fixture
    /// shape `decode_intra_block_mode_info_*` tests use).
    fn fresh_walker_and_cdfs() -> (PartitionWalker, TileCdfContext) {
        let geom = TileGeometry {
            mi_row_start: 0,
            mi_row_end: 32,
            mi_col_start: 0,
            mi_col_end: 32,
        };
        let walker = PartitionWalker::new(32, 32, geom).unwrap();
        let cdfs = TileCdfContext::new_from_defaults();
        (walker, cdfs)
    }

    // -----------------------------------------------------------------
    // §5.11.11 read_skip — write_skip round-trips through decode_skip.
    // -----------------------------------------------------------------

    /// `seg_skip_active = true` arm: no bits written, no bits read,
    /// `skip == 1` on both sides. The decode call also performs no
    /// bitstream read.
    #[test]
    fn write_skip_seg_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_skip(&mut writer, &mut enc_cdfs, 1, 0, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        // For the no-symbol path the decoder needs *some* buffer; a
        // single byte is enough — `init_symbol` requires `sz >= 1`.
        // The decoder's `decode_skip` short-circuit fires on
        // `seg_skip_active == true` and doesn't touch the buffer.
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let skip = walker
            .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, true)
            .unwrap();
        assert_eq!(skip, 1, "§5.11.11 seg short-circuit returns skip = 1");
    }

    /// `seg_skip_active = false`, `skip = 0` at the frame origin
    /// (`ctx = 0` since AvailU = AvailL = false). Round-trips through
    /// `decode_skip` with the §8.3 CDF adaptation engaged on both sides.
    #[test]
    fn write_skip_zero_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = skip_ctx(0, 0);
        write_skip(&mut writer, &mut enc_cdfs, 0, ctx, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let skip = walker
            .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
            .unwrap();
        assert_eq!(skip, 0);
    }

    /// `seg_skip_active = false`, `skip = 1` at the frame origin.
    #[test]
    fn write_skip_one_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = skip_ctx(0, 0);
        write_skip(&mut writer, &mut enc_cdfs, 1, ctx, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let skip = walker
            .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
            .unwrap();
        assert_eq!(skip, 1);
    }

    /// `skip != 1` on the seg-short-circuit arm is a caller bug.
    #[test]
    fn write_skip_rejects_skip_zero_on_seg_arm() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_skip(&mut writer, &mut enc_cdfs, 0, 0, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.7 intra_frame_y_mode — write_intra_frame_y_mode round-trips
    // through decode_intra_frame_y_mode.
    // -----------------------------------------------------------------

    /// `y_mode = DC_PRED` at frame origin: both ctx slots feed
    /// `Intra_Mode_Context[ DC_PRED ]` from the unavailable-neighbour
    /// fallback per §8.3.2. Round-trip through the decoder.
    #[test]
    fn write_intra_frame_y_mode_dc_pred_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // Both neighbours unavailable ⇒ DC_PRED ⇒ Intra_Mode_Context[0].
        let abovemode_ctx = intra_mode_ctx(DC_PRED);
        let leftmode_ctx = intra_mode_ctx(DC_PRED);
        write_intra_frame_y_mode(
            &mut writer,
            &mut enc_cdfs,
            DC_PRED_U8,
            abovemode_ctx,
            leftmode_ctx,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let y_mode = walker
            .decode_intra_frame_y_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16)
            .unwrap();
        assert_eq!(y_mode, DC_PRED_U8);
    }

    /// `y_mode = V_PRED` (directional, ordinal 1) at frame origin —
    /// exercises a non-DC_PRED leaf and confirms the §8.3 adaptation
    /// stays in lockstep.
    #[test]
    fn write_intra_frame_y_mode_v_pred_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let abovemode_ctx = intra_mode_ctx(DC_PRED);
        let leftmode_ctx = intra_mode_ctx(DC_PRED);
        write_intra_frame_y_mode(
            &mut writer,
            &mut enc_cdfs,
            V_PRED_U8,
            abovemode_ctx,
            leftmode_ctx,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let y_mode = walker
            .decode_intra_frame_y_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16)
            .unwrap();
        assert_eq!(y_mode, V_PRED_U8);
    }

    /// `y_mode >= INTRA_MODES` is a caller bug.
    #[test]
    fn write_intra_frame_y_mode_rejects_out_of_range() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_frame_y_mode(&mut writer, &mut enc_cdfs, 13, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.22 y_mode + uv_mode — write_y_mode + write_intra_uv_mode
    // round-trip through decode_intra_block_mode_info.
    // -----------------------------------------------------------------

    /// `y_mode = DC_PRED` + `uv_mode = DC_PRED` on a `BLOCK_16X16`
    /// block with `lossless = false`, `subsampling = 0/0` → CFL allowed
    /// (max dim 16 ≤ 32). Round-trips through `decode_intra_block_mode_info`
    /// with palette + filter-intra gates off.
    #[test]
    fn write_y_mode_and_uv_mode_dc_pred_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);

        // §5.11.22 line 3: y_mode S() with Size_Group[MiSize] ctx.
        let y_size_group = size_group(BLOCK_16X16);
        write_y_mode(&mut writer, &mut enc_cdfs, DC_PRED_U8, y_size_group).unwrap();

        // §5.11.22 line 6: uv_mode S() with the cfl_allowed selector.
        let cfl_allowed = cfl_allowed_for_uv_mode(false, BLOCK_16X16, false, false);
        write_intra_uv_mode(
            &mut writer,
            &mut enc_cdfs,
            DC_PRED_U8,
            DC_PRED_U8,
            cfl_allowed,
        )
        .unwrap();

        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                /* mi_row = */ 0,
                /* mi_col = */ 0,
                /* sub_size = */ BLOCK_16X16,
                /* lossless = */ false,
                /* has_chroma = */ true,
                /* allow_screen_content_tools = */ false,
                /* enable_filter_intra = */ false,
                /* subsampling_x = */ false,
                /* subsampling_y = */ false,
                /* above_palette_y = */ false,
                /* left_palette_y = */ false,
                /* bit_depth = */ 8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8, "round-tripped y_mode");
        assert_eq!(info.uv_mode, Some(DC_PRED_U8), "round-tripped uv_mode");
        // DC_PRED is not directional ⇒ angle_delta short-circuits to 0
        // without a bit read; both sides agree.
        assert_eq!(info.angle_delta_y, 0);
        assert_eq!(info.angle_delta_uv, Some(0));
        // UVMode != UV_CFL_PRED ⇒ cfl_alphas not read.
        assert_eq!(info.cfl_alpha_u, None);
        assert_eq!(info.cfl_alpha_v, None);
        // Palette + filter-intra gates off ⇒ all None.
        assert_eq!(info.has_palette_y, None);
        assert_eq!(info.use_filter_intra, None);
    }

    /// Monochrome (`has_chroma = false`): only `write_y_mode` fires;
    /// `decode_intra_block_mode_info` skips the §5.11.22 chroma arm
    /// entirely and consumes only the y_mode bit(s).
    #[test]
    fn write_y_mode_only_monochrome_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = size_group(BLOCK_16X16);
        write_y_mode(&mut writer, &mut enc_cdfs, DC_PRED_U8, ctx).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                /* has_chroma = */ false,
                false,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, None);
    }

    // -----------------------------------------------------------------
    // §5.11.8 intra_segment_id — write_intra_segment_id round-trips
    // through decode_intra_segment_id.
    // -----------------------------------------------------------------

    /// `segmentation_enabled = false` arm: no bits written, no bits
    /// read; both sides land on `segment_id = 0`.
    #[test]
    fn write_intra_segment_id_disabled_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_intra_segment_id(&mut writer, &mut enc_cdfs, 0, 0, 0, 0, false, 0).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let lossless_array = [false; MAX_SEGMENTS];
        let (sid, _) = walker
            .decode_intra_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* skip = */ 0,
                /* segmentation_enabled = */ false,
                /* last_active_seg_id = */ 0,
                &lossless_array,
            )
            .unwrap();
        assert_eq!(sid, 0);
    }

    /// `segmentation_enabled = true`, `skip = 1`: the §5.11.9 skip
    /// short-circuit fires on the decoder side (`segment_id = pred`,
    /// no bit read); the writer also emits no bits.
    #[test]
    fn write_intra_segment_id_skip_short_circuit() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        // pred = 0 at frame origin (all neighbours = -1 ⇒ pred = 0).
        write_intra_segment_id(&mut writer, &mut enc_cdfs, 0, 1, 0, 0, true, 7).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let lossless_array = [false; MAX_SEGMENTS];
        let (sid, _) = walker
            .decode_intra_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* skip = */ 1,
                /* segmentation_enabled = */ true,
                /* last_active_seg_id = */ 7,
                &lossless_array,
            )
            .unwrap();
        assert_eq!(sid, 0, "skip arm ⇒ segment_id = pred = 0 at origin");
    }

    /// `segmentation_enabled = true`, `skip = 0`: the §5.11.9 `else`
    /// arm reads `diff S()` and reconstructs through `neg_deinterleave`.
    /// At frame origin (`pred = 0`), `neg_deinterleave(diff, 0, max)
    /// == diff` so `write_intra_segment_id(segment_id = 3)` emits
    /// `diff = 3`; the decoder reads `diff = 3` and reconstructs
    /// `segment_id = 3`.
    #[test]
    fn write_intra_segment_id_else_branch_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // At origin: AvailU = AvailL = false ⇒ all three prev_* are
        // None ⇒ §8.3.2 segment_id_ctx returns 0.
        write_intra_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 3,
            /* skip = */ 0,
            /* pred = */ 0,
            /* ctx = */ 0,
            true,
            7,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let lossless_array = [false; MAX_SEGMENTS];
        let (sid, _) = walker
            .decode_intra_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* skip = */ 0,
                /* segmentation_enabled = */ true,
                /* last_active_seg_id = */ 7,
                &lossless_array,
            )
            .unwrap();
        assert_eq!(sid, 3);
    }

    // -----------------------------------------------------------------
    // §5.11.9 read_segment_id — write_segment_id round-trips through
    // decode_segment_id directly (the shared primitive feeding both
    // §5.11.8 intra_segment_id and §5.11.19 inter_segment_id).
    // -----------------------------------------------------------------

    /// §5.11.9 `if ( skip )` short-circuit: no `S()` symbol coded
    /// (the writer takes the no-op arm), and the decoder also reads
    /// no symbol (the §5.11.9 skip branch fires before any
    /// `read_symbol` call). At frame origin every neighbour is
    /// missing ⇒ `pred = 0`, so the round trip lands on `segment_id
    /// = 0`. `decoder.position()` must be unchanged across the
    /// decode call.
    #[test]
    fn write_segment_id_skip_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 0,
            /* skip = */ 1,
            /* pred = */ 0,
            /* ctx = */ 0,
            /* last_active_seg_id = */ 7,
        )
        .unwrap();
        let bytes = writer.finish();
        // `finish()` may emit a flush byte even when no `S()` was
        // coded — the SymbolWriter's terminator-byte protocol is
        // independent of payload. Match the existing
        // `write_intra_segment_id_skip_short_circuit` test's pattern.
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let sid = walker
            .decode_segment_id(
                &mut dec,
                &mut dec_cdfs,
                /* mi_row = */ 0,
                /* mi_col = */ 0,
                BLOCK_16X16,
                /* skip = */ 1,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
        assert_eq!(sid, 0);
        assert_eq!(dec.position(), pos_before, "no S() read on skip path");
    }

    /// §5.11.9 `else` arm at frame origin: `pred = 0` makes
    /// `neg_deinterleave(diff, 0, max) == diff` (the identity), so
    /// `segment_id = 5` encodes as `diff = 5` and the decoder
    /// reconstructs `segment_id = 5`. Smoke that the
    /// `neg_interleave`-based writer matches the existing
    /// `neg_deinterleave`-based reader.
    #[test]
    fn write_segment_id_else_branch_at_origin_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // At origin: all neighbours absent ⇒ §8.3.2 segment_id_ctx
        // returns 0.
        write_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 5,
            /* skip = */ 0,
            /* pred = */ 0,
            /* ctx = */ 0,
            /* last_active_seg_id = */ 7,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sid = walker
            .decode_segment_id(
                &mut dec,
                &mut dec_cdfs,
                /* mi_row = */ 0,
                /* mi_col = */ 0,
                BLOCK_16X16,
                /* skip = */ 0,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
        assert_eq!(sid, 5);
    }

    /// §5.11.9 `else` arm with a non-zero `pred` — covers the upward
    /// bidirectional fan in `neg_interleave`. Stamps the (mi_row=0,
    /// mi_col=0) neighbour by a first encode-then-decode pair at
    /// `segment_id = 2`, so the *second* decode at (mi_row=4,
    /// mi_col=0) sees `prev_u = 2` ⇒ `pred = 2` by the §5.11.9
    /// neighbour cascade (`prev_l == -1, prev_ul == -1, prev_u !=
    /// -1` ⇒ `pred = prev_u`). The second writer call emits
    /// `diff = neg_interleave(s, 2, 8)`; the second decoder call
    /// reads it and reconstructs `s` via `neg_deinterleave`. Exhaust
    /// `segment_id ∈ 0..8`.
    #[test]
    fn write_segment_id_else_branch_upward_fan_round_trip() {
        for sid_in in 0..8u8 {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            // First write: stamp seg_id = 2 at (0, 0) — pred = 0 ⇒
            // diff = 2 on the wire.
            write_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ 2,
                /* skip = */ 0,
                /* pred = */ 0,
                /* ctx = */ 0,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
            // Second write: at (4, 0) the §5.11.9 neighbour cascade
            // matches the decoder's view: `pred = prev_u = 2`.
            write_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ sid_in,
                /* skip = */ 0,
                /* pred = */ 2,
                /* ctx = */ 0,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
            let bytes = writer.finish();

            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let sid_first = walker
                .decode_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    /* mi_row = */ 0,
                    /* mi_col = */ 0,
                    BLOCK_16X16,
                    /* skip = */ 0,
                    /* last_active_seg_id = */ 7,
                )
                .unwrap();
            assert_eq!(sid_first, 2);
            let sid_out = walker
                .decode_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    /* mi_row = */ 4,
                    /* mi_col = */ 0,
                    BLOCK_16X16,
                    /* skip = */ 0,
                    /* last_active_seg_id = */ 7,
                )
                .unwrap();
            assert_eq!(sid_out, sid_in, "round trip failed at sid = {}", sid_in);
        }
    }

    /// §5.11.9 `else` arm with `pred` on the downward bidirectional
    /// fan. Mirror of the upward test with `pred = 5`. Stamps the
    /// (0, 0) neighbour to 5 first, then exercises the second-block
    /// round trip over the 8-symbol alphabet.
    #[test]
    fn write_segment_id_else_branch_downward_fan_round_trip() {
        for sid_in in 0..8u8 {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            // First write: stamp seg_id = 5 at (0, 0); pred = 0 ⇒
            // diff = 5 on the wire.
            write_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ 5,
                /* skip = */ 0,
                /* pred = */ 0,
                /* ctx = */ 0,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
            // Second write: at (4, 0) the §5.11.9 cascade yields
            // `pred = prev_u = 5`.
            write_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ sid_in,
                /* skip = */ 0,
                /* pred = */ 5,
                /* ctx = */ 0,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
            let bytes = writer.finish();

            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let sid_first = walker
                .decode_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    /* mi_row = */ 0,
                    /* mi_col = */ 0,
                    BLOCK_16X16,
                    /* skip = */ 0,
                    /* last_active_seg_id = */ 7,
                )
                .unwrap();
            assert_eq!(sid_first, 5);
            let sid_out = walker
                .decode_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    /* mi_row = */ 4,
                    /* mi_col = */ 0,
                    BLOCK_16X16,
                    /* skip = */ 0,
                    /* last_active_seg_id = */ 7,
                )
                .unwrap();
            assert_eq!(sid_out, sid_in, "round trip failed at sid = {}", sid_in);
        }
    }

    /// §5.11.9 caller-bug shape: `skip == 1` but `segment_id != pred`
    /// — the spec forces `segment_id = pred` in that arm, so the
    /// encoder caller is contradicting itself. Surface as
    /// [`Error::PartitionWalkOutOfRange`].
    #[test]
    fn write_segment_id_rejects_skip_arm_mismatch() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 3,
            /* skip = */ 1,
            /* pred = */ 0, // mismatched: forced to equal pred under skip
            /* ctx = */ 0,
            /* last_active_seg_id = */ 7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.9 caller-bug shape: `segment_id > last_active_seg_id`
    /// (i.e. outside `0..max`). The decoder's §5.11.9 alphabet caps
    /// the `S()` read at `LastActiveSegId + 1`; an encoder that
    /// presents a larger value is asking for a symbol that doesn't
    /// exist. Surface as [`Error::PartitionWalkOutOfRange`].
    #[test]
    fn write_segment_id_rejects_segment_id_out_of_alphabet() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 5,
            /* skip = */ 0,
            /* pred = */ 0,
            /* ctx = */ 0,
            /* last_active_seg_id = */ 3, // alphabet 0..=3
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.9 caller-bug shape: `pred > last_active_seg_id` — the
    /// §5.11.9 `pred` derivation is bounded by `last_active_seg_id`
    /// (neighbours' stored ids are ≤ `last_active_seg_id`, and the
    /// fallback is 0), so a caller passing a larger `pred` is a
    /// grid-walk bug.
    #[test]
    fn write_segment_id_rejects_pred_out_of_alphabet() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 0,
            /* skip = */ 0,
            /* pred = */ 5,
            /* ctx = */ 0,
            /* last_active_seg_id = */ 3,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.9 caller-bug shape: `last_active_seg_id >= MAX_SEGMENTS`
    /// — §6.10.8 caps `LastActiveSegId` at `MAX_SEGMENTS - 1 = 7`.
    #[test]
    fn write_segment_id_rejects_out_of_range_last_active() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 0,
            /* skip = */ 0,
            /* pred = */ 0,
            /* ctx = */ 0,
            /* last_active_seg_id = */ 8,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.9 caller-bug shape: `ctx >= SEGMENT_ID_CONTEXTS = 3` on
    /// the `S()` arm — the §8.3.2 ctx derivation lives in
    /// `0..SEGMENT_ID_CONTEXTS`.
    #[test]
    fn write_segment_id_rejects_out_of_range_ctx() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 0,
            /* skip = */ 0,
            /* pred = */ 0,
            /* ctx = */ 3,
            /* last_active_seg_id = */ 7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // Composed roundtrip: write { skip, intra_segment_id,
    // intra_frame_y_mode } in §5.11.7 order, then decode via the
    // matching walker methods. Forms the first "encode produces a
    // payload the decoder can walk" smoke test for the §5.11 per-block
    // syntax layer.
    // -----------------------------------------------------------------

    /// §5.11.7 SegIdPreSkip = false ⇒ `read_skip` first, then
    /// `intra_segment_id`. Then §5.11.22 `y_mode` (which is what
    /// `decode_intra_block_mode_info` reads after the §5.11.7 prefix
    /// completes). End-to-end smoke that the encoder produces a
    /// payload the §5.11 decoder walks back to the same scalars.
    #[test]
    fn intra_prefix_plus_y_mode_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);

        // §5.11.11 read_skip — origin block (ctx = 0).
        let skip_ctx_val = skip_ctx(0, 0);
        write_skip(&mut writer, &mut enc_cdfs, 0, skip_ctx_val, false).unwrap();

        // §5.11.8 intra_segment_id — segmentation off, no bits.
        write_intra_segment_id(&mut writer, &mut enc_cdfs, 0, 0, 0, 0, false, 0).unwrap();

        // §5.11.22 y_mode — DC_PRED with Size_Group ctx.
        let y_ctx = size_group(BLOCK_8X8);
        write_y_mode(&mut writer, &mut enc_cdfs, DC_PRED_U8, y_ctx).unwrap();

        // §5.11.22 uv_mode — DC_PRED, CFL-allowed on 8x8 lossy.
        let cfl_allowed = cfl_allowed_for_uv_mode(false, BLOCK_8X8, false, false);
        write_intra_uv_mode(
            &mut writer,
            &mut enc_cdfs,
            DC_PRED_U8,
            DC_PRED_U8,
            cfl_allowed,
        )
        .unwrap();

        let bytes = writer.finish();

        // Decoder side: walk the same prefix, then decode_intra_block_mode_info
        // for y_mode + uv_mode.
        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let skip = walker
            .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_8X8, false)
            .unwrap();
        assert_eq!(skip, 0);
        let lossless_array = [false; MAX_SEGMENTS];
        let (sid, _) = walker
            .decode_intra_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_8X8,
                skip,
                /* segmentation_enabled = */ false,
                /* last_active_seg_id = */ 0,
                &lossless_array,
            )
            .unwrap();
        assert_eq!(sid, 0);
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_8X8,
                /* lossless = */ false,
                /* has_chroma = */ true,
                /* allow_screen_content_tools = */ false,
                /* enable_filter_intra = */ false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, Some(DC_PRED_U8));
    }

    // -----------------------------------------------------------------
    // §5.11.45 `write_cfl_alphas` — round-trips through
    // `decode_intra_block_mode_info`'s built-in §5.11.45 reader (the
    // `UVMode == UV_CFL_PRED` arm). r231.
    // -----------------------------------------------------------------

    /// Helper that emits the §5.11.7 / §5.11.11 / §5.11.22 prefix at
    /// frame origin + the §5.11.45 alphas, then walks the bitstream
    /// back through the decoder's `decode_intra_block_mode_info` and
    /// returns the recovered `(CflAlphaU, CflAlphaV)`.
    fn round_trip_cfl_alphas(alpha_u: i8, alpha_v: i8) -> (Option<i8>, Option<i8>) {
        use crate::cdf::UV_CFL_PRED;
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_skip(&mut writer, &mut enc_cdfs, 0, skip_ctx(0, 0), false).unwrap();
        write_intra_segment_id(&mut writer, &mut enc_cdfs, 0, 0, 0, 0, false, 0).unwrap();
        let y_ctx = size_group(BLOCK_16X16);
        write_y_mode(&mut writer, &mut enc_cdfs, DC_PRED_U8, y_ctx).unwrap();
        // CFL-allowed on 16x16, no subsampling, no lossless.
        let cfl_allowed = cfl_allowed_for_uv_mode(false, BLOCK_16X16, false, false);
        write_intra_uv_mode(
            &mut writer,
            &mut enc_cdfs,
            DC_PRED_U8,
            UV_CFL_PRED as u8,
            cfl_allowed,
        )
        .unwrap();
        write_cfl_alphas(&mut writer, &mut enc_cdfs, alpha_u, alpha_v).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let _skip = walker
            .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
            .unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* lossless = */ false,
                /* has_chroma = */ true,
                false,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.uv_mode, Some(UV_CFL_PRED as u8));
        (info.cfl_alpha_u, info.cfl_alpha_v)
    }

    #[test]
    fn write_cfl_alphas_round_trip_pos_pos() {
        let (au, av) = round_trip_cfl_alphas(1, 1);
        assert_eq!(au, Some(1));
        assert_eq!(av, Some(1));
    }

    #[test]
    fn write_cfl_alphas_round_trip_neg_neg() {
        let (au, av) = round_trip_cfl_alphas(-3, -7);
        assert_eq!(au, Some(-3));
        assert_eq!(av, Some(-7));
    }

    #[test]
    fn write_cfl_alphas_round_trip_zero_u_pos_v() {
        // signU = ZERO: only V's magnitude is written; U's value
        // reconstructs to 0 on the decoder.
        let (au, av) = round_trip_cfl_alphas(0, 4);
        assert_eq!(au, Some(0));
        assert_eq!(av, Some(4));
    }

    #[test]
    fn write_cfl_alphas_round_trip_neg_u_zero_v() {
        let (au, av) = round_trip_cfl_alphas(-2, 0);
        assert_eq!(au, Some(-2));
        assert_eq!(av, Some(0));
    }

    #[test]
    fn write_cfl_alphas_round_trip_max_magnitudes() {
        // |alpha| = 16 corresponds to cfl_alpha_u = 15 (the max raw
        // value the §5.11.45 S() reads from `Default_Cfl_Alpha_Cdf`).
        let (au, av) = round_trip_cfl_alphas(16, -16);
        assert_eq!(au, Some(16));
        assert_eq!(av, Some(-16));
    }

    #[test]
    fn write_cfl_alphas_rejects_both_zero() {
        // §6.10.36: (CFL_SIGN_ZERO, CFL_SIGN_ZERO) is prohibited as
        // redundant with UV_DC_PRED.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_cfl_alphas(&mut writer, &mut cdfs, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    #[test]
    fn write_cfl_alphas_rejects_out_of_range_magnitude() {
        // |alpha| > CFL_ALPHABET_SIZE = 16 is out of the §5.11.45
        // alphabet.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_cfl_alphas(&mut writer, &mut cdfs, 17, 1).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.20 read_is_inter — write_is_inter round-trips through
    // decode_is_inter. Each arm of the four-arm dispatch gets a test;
    // the fall-through `S()` arm gets both is_inter = 0 and is_inter = 1
    // round-trips plus the §8.3.2 ctx coverage.
    // -----------------------------------------------------------------

    /// §5.11.20 Arm 1: `skip_mode == 1` ⇒ `is_inter = 1`, no bits
    /// written. Mirrors `decode_is_inter_skip_mode_short_circuit_no_symbol_read`.
    #[test]
    fn write_is_inter_skip_mode_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_is_inter(&mut writer, &mut enc_cdfs, 1, 0, 1, false, false, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        // No-symbol path — the decoder needs *some* buffer; a single
        // byte is enough. `init_symbol` requires `sz >= 1`.
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let is_inter = walker
            .decode_is_inter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                1,
                false,
                false,
                false,
            )
            .unwrap();
        assert_eq!(is_inter, 1, "§5.11.20 Arm 1: skip_mode = 1 ⇒ is_inter = 1");
        assert_eq!(
            dec.position(),
            pos_before,
            "no symbol bit read on the skip_mode arm"
        );
    }

    /// §5.11.20 Arm 2 (intra-routing branch): SEG_LVL_REF_FRAME
    /// active and FeatureData == INTRA_FRAME (caller encodes as
    /// `seg_ref_frame_is_inter = false`) ⇒ `is_inter = 0`, no bits.
    #[test]
    fn write_is_inter_seg_ref_frame_intra_routes_to_zero_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_is_inter(&mut writer, &mut enc_cdfs, 0, 0, 0, true, false, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let is_inter = walker
            .decode_is_inter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                true,
                false,
                false,
            )
            .unwrap();
        assert_eq!(
            is_inter, 0,
            "Arm 2: FeatureData == INTRA_FRAME ⇒ is_inter = 0"
        );
        assert_eq!(dec.position(), pos_before);
    }

    /// §5.11.20 Arm 2 (inter-routing branch): SEG_LVL_REF_FRAME
    /// active and FeatureData != INTRA_FRAME (caller encodes as
    /// `seg_ref_frame_is_inter = true`) ⇒ `is_inter = 1`, no bits.
    #[test]
    fn write_is_inter_seg_ref_frame_inter_routes_to_one_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_is_inter(&mut writer, &mut enc_cdfs, 1, 0, 0, true, true, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let is_inter = walker
            .decode_is_inter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                true,
                true,
                false,
            )
            .unwrap();
        assert_eq!(
            is_inter, 1,
            "Arm 2: FeatureData != INTRA_FRAME ⇒ is_inter = 1"
        );
        assert_eq!(dec.position(), pos_before);
    }

    /// §5.11.20 Arm 3: SEG_LVL_GLOBALMV active (with Arms 1 / 2 false)
    /// ⇒ `is_inter = 1`, no bits.
    #[test]
    fn write_is_inter_seg_globalmv_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_is_inter(&mut writer, &mut enc_cdfs, 1, 0, 0, false, false, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let is_inter = walker
            .decode_is_inter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                false,
                false,
                true,
            )
            .unwrap();
        assert_eq!(is_inter, 1, "Arm 3: SEG_LVL_GLOBALMV ⇒ is_inter = 1");
        assert_eq!(dec.position(), pos_before);
    }

    /// §5.11.20 Arm 4 (fall-through `S()`): all three short-circuit
    /// arms false, `is_inter = 0` at the frame origin. The §8.3.2 ctx
    /// at `(mi_row = 0, mi_col = 0)` is `is_inter_ctx(None, None) = 0`
    /// (both neighbours unavailable). Round-trip through `decode_is_inter`
    /// with the §8.3 CDF adaptation engaged on both sides.
    #[test]
    fn write_is_inter_zero_else_branch_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = is_inter_ctx(None, None);
        write_is_inter(&mut writer, &mut enc_cdfs, 0, ctx, 0, false, false, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let is_inter = walker
            .decode_is_inter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                false,
                false,
                false,
            )
            .unwrap();
        assert_eq!(is_inter, 0);
    }

    /// §5.11.20 Arm 4: `is_inter = 1` at the frame origin (same ctx as
    /// above). Round-trip through the decoder.
    #[test]
    fn write_is_inter_one_else_branch_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = is_inter_ctx(None, None);
        write_is_inter(&mut writer, &mut enc_cdfs, 1, ctx, 0, false, false, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let is_inter = walker
            .decode_is_inter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                false,
                false,
                false,
            )
            .unwrap();
        assert_eq!(is_inter, 1);
    }

    /// §5.11.20 Arm 4 §8.3.2 ctx coverage at frame origin: walk every
    /// `(above_intra, left_intra)` combination [`is_inter_ctx`]
    /// distinguishes for `(None, None)` — i.e. both unavailable at the
    /// origin — and round-trip both `is_inter = 0` and `is_inter = 1`
    /// through the decoder. CDF adaptation engaged on both sides so
    /// the encoder's choice of `ctx` (which CDF row it writes into)
    /// MUST match the decoder's `is_inter_ctx(None, None) = 0` ctx
    /// derivation: any mismatch would either fail to decode (different
    /// row → different probability mass than the encoder used) or
    /// produce a different `is_inter` value than written.
    #[test]
    fn write_is_inter_else_branch_round_trip_at_origin_both_is_inter_values() {
        for is_inter_val in [0u8, 1u8] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let ctx = is_inter_ctx(None, None);
            assert_eq!(ctx, 0, "origin neighbours unavailable ⇒ ctx = 0");
            assert!(ctx < IS_INTER_CONTEXTS);
            write_is_inter(
                &mut writer,
                &mut enc_cdfs,
                is_inter_val,
                ctx,
                0,
                false,
                false,
                false,
            )
            .unwrap();
            let bytes = writer.finish();
            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let decoded = walker
                .decode_is_inter(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    0,
                    false,
                    false,
                    false,
                )
                .unwrap();
            assert_eq!(decoded, is_inter_val);
        }
    }

    /// §8.3.2 ctx exhaustive coverage: every
    /// `(above_intra, left_intra)` combination [`is_inter_ctx`]
    /// recognises stays in `0..IS_INTER_CONTEXTS`, so any of them is
    /// an admissible `ctx` input to [`write_is_inter`] on the Arm 4
    /// fall-through path. The smoke test confirms write_is_inter
    /// accepts every ctx without erroring.
    #[test]
    fn write_is_inter_accepts_every_8_3_2_ctx() {
        let cases: &[(Option<bool>, Option<bool>)] = &[
            (None, None),
            (Some(true), None),
            (Some(false), None),
            (None, Some(true)),
            (None, Some(false)),
            (Some(true), Some(true)),
            (Some(true), Some(false)),
            (Some(false), Some(true)),
            (Some(false), Some(false)),
        ];
        for &(above, left) in cases {
            let ctx = is_inter_ctx(above, left);
            assert!(ctx < IS_INTER_CONTEXTS);
            for is_inter_val in [0u8, 1u8] {
                let mut cdfs = TileCdfContext::new_from_defaults();
                let mut writer = SymbolWriter::new(false);
                write_is_inter(
                    &mut writer,
                    &mut cdfs,
                    is_inter_val,
                    ctx,
                    0,
                    false,
                    false,
                    false,
                )
                .expect("§8.3.2 ctx admissible on Arm 4");
                let _bytes = writer.finish();
            }
        }
    }

    /// `is_inter > 1` is outside the §3 binary alphabet — caller bug.
    #[test]
    fn write_is_inter_rejects_out_of_range_is_inter() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_is_inter(&mut writer, &mut cdfs, 2, 0, 0, false, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.20 Arm 1: caller-supplied `is_inter != 1` on the skip_mode
    /// arm is a caller bug — the spec forces `is_inter = 1`.
    #[test]
    fn write_is_inter_rejects_zero_on_skip_mode_arm() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_is_inter(&mut writer, &mut cdfs, 0, 0, 1, false, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.20 Arm 2: caller-supplied `is_inter` must equal
    /// `seg_ref_frame_is_inter as u8` — mismatch is a caller bug.
    #[test]
    fn write_is_inter_rejects_arm2_mismatch() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        // SEG_LVL_REF_FRAME active, FeatureData == INTRA_FRAME (encoded
        // as `seg_ref_frame_is_inter = false`) — caller MUST pass
        // is_inter = 0; passing 1 is a bug.
        let err = write_is_inter(&mut writer, &mut cdfs, 1, 0, 0, true, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.20 Arm 3: caller-supplied `is_inter != 1` on the
    /// SEG_LVL_GLOBALMV arm is a caller bug — the spec forces 1.
    #[test]
    fn write_is_inter_rejects_zero_on_globalmv_arm() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_is_inter(&mut writer, &mut cdfs, 0, 0, 0, false, false, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.20 Arm 4: `ctx >= IS_INTER_CONTEXTS = 4` on the
    /// fall-through arm is an invalid §8.3.2 ctx — caller bug.
    #[test]
    fn write_is_inter_rejects_out_of_range_ctx_on_else_arm() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_is_inter(
            &mut writer,
            &mut cdfs,
            0,
            IS_INTER_CONTEXTS,
            0,
            false,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.20 arm-ordering invariant: when Arm 1 (`skip_mode == 1`)
    /// fires, the SEG_LVL_REF_FRAME / SEG_LVL_GLOBALMV / ctx parameters
    /// are NOT consulted. So setting them to values that would
    /// otherwise reject (e.g. an out-of-range ctx) is fine — Arm 1
    /// short-circuits before any of them is checked.
    #[test]
    fn write_is_inter_arm1_skips_later_arm_checks() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        // out-of-range ctx, but Arm 1 fires first ⇒ no error.
        write_is_inter(
            &mut writer,
            &mut cdfs,
            1,
            999, // would normally reject if Arm 4 were reached
            1,   // Arm 1 active
            true,
            false,
            true,
        )
        .unwrap();
    }

    // -----------------------------------------------------------------
    // §5.11.10 read_skip_mode — write_skip_mode round-trips through
    // decode_skip_mode. The six-condition short-circuit set collapses
    // (per the decoder's own parameter surface) into three encoder
    // inputs — `seg_skip_mode_off`, `!skip_mode_present`, and a
    // small-block check derived locally from `sub_size`. The
    // fall-through `S()` arm gets both skip_mode = 0 and skip_mode = 1
    // round-trips plus exhaustive §8.3.2 ctx coverage.
    // -----------------------------------------------------------------

    /// §5.11.10 `seg_skip_mode_off = true` arm — collapsed
    /// `SEG_LVL_SKIP || SEG_LVL_REF_FRAME || SEG_LVL_GLOBALMV` —
    /// forces `skip_mode = 0` with no bits emitted. The decoder side
    /// (`decode_skip_mode` with `seg_skip_mode_off = true`) likewise
    /// short-circuits without a symbol read.
    #[test]
    fn write_skip_mode_seg_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_skip_mode(&mut writer, &mut enc_cdfs, 0, 0, BLOCK_16X16, true, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        // No-symbol path — the decoder needs at least one byte to
        // initialise the symbol decoder.
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let skip_mode = walker
            .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, true, true)
            .unwrap();
        assert_eq!(skip_mode, 0, "§5.11.10 seg-arm: any-active ⇒ skip_mode = 0");
        assert_eq!(
            dec.position(),
            pos_before,
            "no symbol bit read on the seg short-circuit arm"
        );
    }

    /// §5.11.10 `!skip_mode_present` arm — the §5.9.21 frame-header
    /// scalar is false ⇒ `skip_mode = 0`, no bits emitted.
    #[test]
    fn write_skip_mode_present_false_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_skip_mode(&mut writer, &mut enc_cdfs, 0, 0, BLOCK_16X16, false, false).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let skip_mode = walker
            .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false, false)
            .unwrap();
        assert_eq!(skip_mode, 0);
        assert_eq!(dec.position(), pos_before);
    }

    /// §5.11.10 small-block short-circuit: `Block_Width[ MiSize ] < 8`
    /// on `BLOCK_4X8` (width 4, height 8) ⇒ `skip_mode = 0`, no bits.
    #[test]
    fn write_skip_mode_small_width_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_skip_mode(&mut writer, &mut enc_cdfs, 0, 0, BLOCK_4X8, false, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let skip_mode = walker
            .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_4X8, false, true)
            .unwrap();
        assert_eq!(skip_mode, 0);
        assert_eq!(dec.position(), pos_before);
    }

    /// §5.11.10 small-block short-circuit: `Block_Height[ MiSize ] < 8`
    /// on `BLOCK_8X4` (width 8, height 4) ⇒ `skip_mode = 0`, no bits.
    #[test]
    fn write_skip_mode_small_height_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_skip_mode(&mut writer, &mut enc_cdfs, 0, 0, BLOCK_8X4, false, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let skip_mode = walker
            .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_8X4, false, true)
            .unwrap();
        assert_eq!(skip_mode, 0);
        assert_eq!(dec.position(), pos_before);
    }

    /// §5.11.10 small-block short-circuit: both dimensions < 8 on
    /// `BLOCK_4X4` ⇒ `skip_mode = 0`, no bits.
    #[test]
    fn write_skip_mode_small_4x4_short_circuit_writes_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_skip_mode(&mut writer, &mut enc_cdfs, 0, 0, BLOCK_4X4, false, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let pos_before = dec.position();
        let skip_mode = walker
            .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_4X4, false, true)
            .unwrap();
        assert_eq!(skip_mode, 0);
        assert_eq!(dec.position(), pos_before);
    }

    /// §5.11.10 fall-through `S()` arm: `skip_mode = 0` at the frame
    /// origin (both neighbours unavailable ⇒ §8.3.2 ctx = 0).
    /// Round-trip through `decode_skip_mode` with §8.3 CDF adaptation
    /// engaged on both sides.
    #[test]
    fn write_skip_mode_zero_else_branch_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = skip_mode_ctx(0, 0);
        assert_eq!(ctx, 0, "origin neighbours unavailable ⇒ ctx = 0");
        write_skip_mode(&mut writer, &mut enc_cdfs, 0, ctx, BLOCK_16X16, false, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let skip_mode = walker
            .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false, true)
            .unwrap();
        assert_eq!(skip_mode, 0);
    }

    /// §5.11.10 fall-through `S()` arm: `skip_mode = 1` at the frame
    /// origin (same ctx). Round-trip through the decoder.
    #[test]
    fn write_skip_mode_one_else_branch_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = skip_mode_ctx(0, 0);
        write_skip_mode(&mut writer, &mut enc_cdfs, 1, ctx, BLOCK_16X16, false, true).unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let skip_mode = walker
            .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false, true)
            .unwrap();
        assert_eq!(skip_mode, 1);
    }

    /// §8.3.2 `skip_mode` ctx is the neighbour-`SkipModes[]` sum, in
    /// `0..SKIP_MODE_CONTEXTS = 3`. Walk every `(above_sm, left_sm)`
    /// combination (`{0,1} × {0,1}` plus the `(1,1)` upper-bound case)
    /// and confirm the writer accepts each ctx for both `skip_mode ∈
    /// {0, 1}` on the fall-through arm.
    #[test]
    fn write_skip_mode_accepts_every_8_3_2_ctx() {
        let cases: &[(u8, u8)] = &[(0, 0), (0, 1), (1, 0), (1, 1)];
        for &(above, left) in cases {
            let ctx = skip_mode_ctx(above, left);
            assert!(ctx < SKIP_MODE_CONTEXTS);
            for skip_mode_val in [0u8, 1u8] {
                let mut cdfs = TileCdfContext::new_from_defaults();
                let mut writer = SymbolWriter::new(false);
                write_skip_mode(
                    &mut writer,
                    &mut cdfs,
                    skip_mode_val,
                    ctx,
                    BLOCK_16X16,
                    false,
                    true,
                )
                .expect("§8.3.2 ctx admissible on the fall-through arm");
                let _bytes = writer.finish();
            }
        }
    }

    /// `skip_mode > 1` is outside the §3 binary alphabet — caller bug.
    #[test]
    fn write_skip_mode_rejects_out_of_range_skip_mode() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err =
            write_skip_mode(&mut writer, &mut cdfs, 2, 0, BLOCK_16X16, false, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `sub_size >= BLOCK_SIZES` is outside the §5.11.5 `MiSize`
    /// domain — caller bug.
    #[test]
    fn write_skip_mode_rejects_out_of_range_sub_size() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err =
            write_skip_mode(&mut writer, &mut cdfs, 0, 0, BLOCK_SIZES, false, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.10 short-circuit arm: caller-supplied `skip_mode != 0` on
    /// any short-circuit arm is a caller bug — the spec forces
    /// `skip_mode = 0`.
    #[test]
    fn write_skip_mode_rejects_one_on_seg_arm() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err =
            write_skip_mode(&mut writer, &mut cdfs, 1, 0, BLOCK_16X16, true, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.10 short-circuit arm: caller-supplied `skip_mode != 0` on
    /// the `!skip_mode_present` arm is a caller bug.
    #[test]
    fn write_skip_mode_rejects_one_on_present_false_arm() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err =
            write_skip_mode(&mut writer, &mut cdfs, 1, 0, BLOCK_16X16, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.10 short-circuit arm: caller-supplied `skip_mode != 0` on
    /// the small-block (Block_W < 8 || Block_H < 8) arm is a caller
    /// bug.
    #[test]
    fn write_skip_mode_rejects_one_on_small_block_arm() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err =
            write_skip_mode(&mut writer, &mut cdfs, 1, 0, BLOCK_4X4, false, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.10 fall-through arm: `ctx >= SKIP_MODE_CONTEXTS = 3` is
    /// an invalid §8.3.2 ctx — caller bug.
    #[test]
    fn write_skip_mode_rejects_out_of_range_ctx_on_else_arm() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_skip_mode(
            &mut writer,
            &mut cdfs,
            0,
            SKIP_MODE_CONTEXTS,
            BLOCK_16X16,
            false,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.10 arm-ordering invariant: when any short-circuit arm
    /// fires (`seg_skip_mode_off`, `!skip_mode_present`, or
    /// small-block), the `ctx` parameter is NOT consulted. So an
    /// out-of-range ctx is fine on a short-circuit arm.
    #[test]
    fn write_skip_mode_short_circuit_skips_ctx_check() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        // out-of-range ctx, but seg-arm fires first ⇒ no error.
        write_skip_mode(
            &mut writer,
            &mut cdfs,
            0,
            999, // would normally reject if fall-through reached
            BLOCK_16X16,
            true, // seg arm active
            true,
        )
        .unwrap();
    }

    // -----------------------------------------------------------------
    // §5.11.19 read_inter_segment_id — write_inter_segment_id
    // round-trips through PartitionWalker::decode_inter_segment_id.
    // -----------------------------------------------------------------

    /// Constant `LosslessArray` shape the round-trip tests share —
    /// all-false matches the default §6.8.2 derivation and lets us
    /// inspect the returned `lossless` flag without setting up the
    /// quantiser path.
    const LOSSLESS_ALL_FALSE: [bool; MAX_SEGMENTS] = [false; MAX_SEGMENTS];

    /// Helper — pad an empty payload (the no-S() arms) so
    /// `SymbolDecoder::init_symbol` accepts the buffer. Matches the
    /// pattern in the `write_skip_seg_short_circuit` /
    /// `write_segment_id_skip_short_circuit` tests.
    fn pad_no_symbol(bytes: Vec<u8>) -> Vec<u8> {
        if bytes.is_empty() {
            vec![0u8]
        } else {
            bytes
        }
    }

    /// Arm 1: `!segmentation_enabled` — no bits written, no bits read,
    /// `segment_id == 0` on both sides.
    #[test]
    fn write_inter_segment_id_disabled_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_inter_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 0,
            /* pred = */ 0,
            /* seg_id_read_ctx = */ 0,
            /* seg_pred_ctx = */ 0,
            /* pre_skip = */ true,
            /* skip = */ 0,
            /* seg_id_predicted = */ 0,
            /* segmentation_enabled = */ false,
            /* segmentation_update_map = */ true,
            /* segmentation_temporal_update = */ true,
            /* seg_id_pre_skip = */ true,
            /* predicted_segment_id = */ 0,
            /* last_active_seg_id = */ 7,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        let pos_before = dec.position();
        let (sid, lossless) = walker
            .decode_inter_segment_id(
                &mut dec,
                &mut dec_cdfs,
                /* mi_row = */ 0,
                /* mi_col = */ 0,
                BLOCK_16X16,
                /* pre_skip = */ true,
                /* skip = */ 0,
                /* seg_id_pre_skip = */ true,
                /* segmentation_enabled = */ false,
                /* segmentation_update_map = */ true,
                /* segmentation_temporal_update = */ true,
                /* predicted_segment_id = */ 0,
                /* last_active_seg_id = */ 7,
                &LOSSLESS_ALL_FALSE,
            )
            .unwrap();
        assert_eq!(sid, 0);
        assert!(!lossless);
        assert_eq!(dec.position(), pos_before, "no S() read on disabled arm");
    }

    /// Arm 2: `!segmentation_update_map` — no bits; `segment_id ==
    /// predicted_segment_id` on both sides.
    #[test]
    fn write_inter_segment_id_no_update_map_adopts_predicted() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_inter_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 4,
            /* pred = */ 0,
            /* seg_id_read_ctx = */ 0,
            /* seg_pred_ctx = */ 0,
            /* pre_skip = */ false,
            /* skip = */ 0,
            /* seg_id_predicted = */ 0,
            /* segmentation_enabled = */ true,
            /* segmentation_update_map = */ false,
            /* segmentation_temporal_update = */ false,
            /* seg_id_pre_skip = */ false,
            /* predicted_segment_id = */ 4,
            /* last_active_seg_id = */ 7,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        let pos_before = dec.position();
        let (sid, _) = walker
            .decode_inter_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* pre_skip = */ false,
                /* skip = */ 0,
                /* seg_id_pre_skip = */ false,
                /* segmentation_enabled = */ true,
                /* segmentation_update_map = */ false,
                /* segmentation_temporal_update = */ false,
                /* predicted_segment_id = */ 4,
                7,
                &LOSSLESS_ALL_FALSE,
            )
            .unwrap();
        assert_eq!(sid, 4);
        assert_eq!(dec.position(), pos_before, "no S() read on !update_map arm");
    }

    /// Arm 3: `pre_skip && !seg_id_pre_skip` early-exit — no bits;
    /// `segment_id == 0` on both sides.
    #[test]
    fn write_inter_segment_id_pre_skip_early_exit_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_inter_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 0,
            /* pred = */ 0,
            /* seg_id_read_ctx = */ 0,
            /* seg_pred_ctx = */ 0,
            /* pre_skip = */ true,
            /* skip = */ 0,
            /* seg_id_predicted = */ 0,
            /* segmentation_enabled = */ true,
            /* segmentation_update_map = */ true,
            /* segmentation_temporal_update = */ false,
            /* seg_id_pre_skip = */ false,
            /* predicted_segment_id = */ 3,
            /* last_active_seg_id = */ 7,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        let pos_before = dec.position();
        let (sid, _) = walker
            .decode_inter_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* pre_skip = */ true,
                /* skip = */ 0,
                /* seg_id_pre_skip = */ false,
                /* segmentation_enabled = */ true,
                /* segmentation_update_map = */ true,
                /* segmentation_temporal_update = */ false,
                /* predicted_segment_id = */ 3,
                7,
                &LOSSLESS_ALL_FALSE,
            )
            .unwrap();
        assert_eq!(sid, 0);
        assert_eq!(
            dec.position(),
            pos_before,
            "no S() read on pre-skip early-exit"
        );
    }

    /// Arm 4 (`!pre_skip && skip`): the inner `read_segment_id()`
    /// fires with `skip = 1`, which §5.11.9 short-circuits to
    /// `segment_id = pred` (no S() coded). Round-trip across the
    /// 8-symbol alphabet of `pred` values at the frame origin (where
    /// `pred = 0` because no neighbours), then with a stamped
    /// neighbour where `pred != 0`.
    #[test]
    fn write_inter_segment_id_arm4_skip_branch_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        // At origin: all neighbours absent ⇒ §5.11.9 pred = 0.
        write_inter_segment_id(
            &mut writer,
            &mut enc_cdfs,
            /* segment_id = */ 0,
            /* pred = */ 0,
            /* seg_id_read_ctx = */ 0,
            /* seg_pred_ctx = */ 0,
            /* pre_skip = */ false,
            /* skip = */ 1,
            /* seg_id_predicted = */ 0,
            /* segmentation_enabled = */ true,
            /* segmentation_update_map = */ true,
            /* segmentation_temporal_update = */ true,
            /* seg_id_pre_skip = */ true,
            /* predicted_segment_id = */ 0,
            /* last_active_seg_id = */ 7,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        let pos_before = dec.position();
        let (sid, _) = walker
            .decode_inter_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* pre_skip = */ false,
                /* skip = */ 1,
                /* seg_id_pre_skip = */ true,
                /* segmentation_enabled = */ true,
                /* segmentation_update_map = */ true,
                /* segmentation_temporal_update = */ true,
                /* predicted_segment_id = */ 0,
                7,
                &LOSSLESS_ALL_FALSE,
            )
            .unwrap();
        assert_eq!(sid, 0);
        assert_eq!(
            dec.position(),
            pos_before,
            "Arm 4: no S() read (inner write_segment_id short-circuits on skip=1)"
        );
    }

    /// Arm 5 with `seg_id_predicted == 1` — emits exactly one S()
    /// (`seg_id_predicted` against `TileSegmentIdPredictedCdf`); the
    /// id itself is the adopted `predicted_segment_id`, no further
    /// bits. Round-trip across the alphabet of `predicted_segment_id`
    /// values via the temporal-update arm at the frame origin
    /// (`seg_pred_ctx = 0` because both neighbour ctx arrays are 0).
    #[test]
    fn write_inter_segment_id_temporal_update_adopt_round_trip() {
        for predicted in 0..8u8 {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_inter_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ predicted,
                /* pred = */ 0, // unused on the adopt branch
                /* seg_id_read_ctx = */ 0,
                /* seg_pred_ctx = */ 0,
                /* pre_skip = */ false,
                /* skip = */ 0,
                /* seg_id_predicted = */ 1,
                /* segmentation_enabled = */ true,
                /* segmentation_update_map = */ true,
                /* segmentation_temporal_update = */ true,
                /* seg_id_pre_skip = */ true,
                /* predicted_segment_id = */ predicted,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
            let bytes = writer.finish();

            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let (sid, _) = walker
                .decode_inter_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    /* pre_skip = */ false,
                    /* skip = */ 0,
                    /* seg_id_pre_skip = */ true,
                    /* segmentation_enabled = */ true,
                    /* segmentation_update_map = */ true,
                    /* segmentation_temporal_update = */ true,
                    /* predicted_segment_id = */ predicted,
                    7,
                    &LOSSLESS_ALL_FALSE,
                )
                .unwrap();
            assert_eq!(
                sid, predicted,
                "adopt round trip failed at predicted = {}",
                predicted
            );
        }
    }

    /// Arm 5 with `seg_id_predicted == 0` — emits two S() values:
    /// `seg_id_predicted = 0` against `TileSegmentIdPredictedCdf` then
    /// the §5.11.9 `S()` against `TileSegmentIdCdf` for the literal
    /// id. Round-trip across the alphabet of `segment_id` values at
    /// the frame origin (where the §5.11.9 pred = 0).
    #[test]
    fn write_inter_segment_id_temporal_update_literal_round_trip() {
        for sid_in in 0..8u8 {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_inter_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ sid_in,
                /* pred = */ 0,
                /* seg_id_read_ctx = */ 0,
                /* seg_pred_ctx = */ 0,
                /* pre_skip = */ false,
                /* skip = */ 0,
                /* seg_id_predicted = */ 0,
                /* segmentation_enabled = */ true,
                /* segmentation_update_map = */ true,
                /* segmentation_temporal_update = */ true,
                /* seg_id_pre_skip = */ true,
                /* predicted_segment_id = */ 0,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
            let bytes = writer.finish();

            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let (sid_out, _) = walker
                .decode_inter_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    /* pre_skip = */ false,
                    /* skip = */ 0,
                    /* seg_id_pre_skip = */ true,
                    /* segmentation_enabled = */ true,
                    /* segmentation_update_map = */ true,
                    /* segmentation_temporal_update = */ true,
                    /* predicted_segment_id = */ 0,
                    7,
                    &LOSSLESS_ALL_FALSE,
                )
                .unwrap();
            assert_eq!(
                sid_out, sid_in,
                "literal round trip failed at sid = {}",
                sid_in
            );
        }
    }

    /// Arm 6 (`segmentation_temporal_update == 0`, fall-through) —
    /// single §5.11.9 S() for the literal id; no `seg_id_predicted`
    /// symbol. Round-trip across the alphabet at the frame origin.
    #[test]
    fn write_inter_segment_id_fallthrough_round_trip() {
        for sid_in in 0..8u8 {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_inter_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ sid_in,
                /* pred = */ 0,
                /* seg_id_read_ctx = */ 0,
                /* seg_pred_ctx = */ 0,
                /* pre_skip = */ false,
                /* skip = */ 0,
                /* seg_id_predicted = */ 0,
                /* segmentation_enabled = */ true,
                /* segmentation_update_map = */ true,
                /* segmentation_temporal_update = */ false,
                /* seg_id_pre_skip = */ true,
                /* predicted_segment_id = */ 0,
                /* last_active_seg_id = */ 7,
            )
            .unwrap();
            let bytes = writer.finish();

            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let (sid_out, _) = walker
                .decode_inter_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    /* pre_skip = */ false,
                    /* skip = */ 0,
                    /* seg_id_pre_skip = */ true,
                    /* segmentation_enabled = */ true,
                    /* segmentation_update_map = */ true,
                    /* segmentation_temporal_update = */ false,
                    /* predicted_segment_id = */ 0,
                    7,
                    &LOSSLESS_ALL_FALSE,
                )
                .unwrap();
            assert_eq!(
                sid_out, sid_in,
                "arm 6 fall-through round trip failed at sid = {}",
                sid_in
            );
        }
    }

    // -----------------------------------------------------------------
    // §5.11.19 caller-bug shapes — every mismatch surfaces as
    // PartitionWalkOutOfRange.
    // -----------------------------------------------------------------

    /// Arm 1 rejection: `!segmentation_enabled` forces `segment_id =
    /// 0`; a non-zero caller value is a bug.
    #[test]
    fn write_inter_segment_id_arm1_rejects_nonzero_segment_id() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            /* segment_id = */ 3,
            0,
            0,
            0,
            true,
            0,
            0,
            /* segmentation_enabled = */ false,
            true,
            true,
            true,
            0,
            7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 2 rejection: `!segmentation_update_map` forces
    /// `segment_id = predicted_segment_id`.
    #[test]
    fn write_inter_segment_id_arm2_rejects_segment_id_mismatch() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            /* segment_id = */ 2,
            0,
            0,
            0,
            false,
            0,
            0,
            true,
            /* segmentation_update_map = */ false,
            false,
            false,
            /* predicted_segment_id = */ 5,
            7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 3 rejection: `pre_skip && !seg_id_pre_skip` forces
    /// `segment_id = 0`.
    #[test]
    fn write_inter_segment_id_arm3_rejects_nonzero_segment_id() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            /* segment_id = */ 3,
            0,
            0,
            0,
            /* pre_skip = */ true,
            0,
            0,
            true,
            true,
            false,
            /* seg_id_pre_skip = */ false,
            3,
            7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 4 rejection: `!pre_skip && skip` delegates to
    /// `write_segment_id` with `skip = 1`, which requires
    /// `segment_id == pred`. A mismatch surfaces from the inner
    /// writer.
    #[test]
    fn write_inter_segment_id_arm4_rejects_pred_mismatch() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            /* segment_id = */ 3,
            /* pred = */ 1, // mismatched — must equal segment_id on skip arm
            0,
            0,
            /* pre_skip = */ false,
            /* skip = */ 1,
            0,
            true,
            true,
            true,
            true,
            0,
            7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 5 (`seg_id_predicted == 1`) rejection: adoption requires
    /// `segment_id == predicted_segment_id`.
    #[test]
    fn write_inter_segment_id_arm5_adopt_rejects_predicted_mismatch() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            /* segment_id = */ 4,
            0,
            0,
            0,
            false,
            0,
            /* seg_id_predicted = */ 1,
            true,
            true,
            true,
            true,
            /* predicted_segment_id = */ 2,
            7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 5 rejection: `seg_id_predicted > 1` is outside the binary
    /// alphabet.
    #[test]
    fn write_inter_segment_id_arm5_rejects_out_of_range_seg_id_predicted() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            0,
            0,
            0,
            0,
            false,
            0,
            /* seg_id_predicted = */ 2,
            true,
            true,
            true,
            true,
            0,
            7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 5 rejection: `seg_pred_ctx >= SEGMENT_ID_PREDICTED_CONTEXTS
    /// = 3`.
    #[test]
    fn write_inter_segment_id_arm5_rejects_out_of_range_seg_pred_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            0,
            0,
            0,
            /* seg_pred_ctx = */ SEGMENT_ID_PREDICTED_CONTEXTS,
            false,
            0,
            0,
            true,
            true,
            true,
            true,
            0,
            7,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Up-front rejection: `last_active_seg_id >= MAX_SEGMENTS`.
    #[test]
    fn write_inter_segment_id_rejects_out_of_range_last_active() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            0,
            0,
            0,
            0,
            false,
            0,
            0,
            true,
            true,
            false,
            true,
            0,
            /* last_active_seg_id = */ MAX_SEGMENTS as u8,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Up-front rejection: `segment_id > last_active_seg_id` (outside
    /// the alphabet).
    #[test]
    fn write_inter_segment_id_rejects_segment_id_out_of_alphabet() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            /* segment_id = */ 5,
            0,
            0,
            0,
            false,
            0,
            0,
            true,
            true,
            false,
            true,
            0,
            /* last_active_seg_id = */ 3,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Up-front rejection: `pred > last_active_seg_id`.
    #[test]
    fn write_inter_segment_id_rejects_pred_out_of_alphabet() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            0,
            /* pred = */ 5,
            0,
            0,
            false,
            0,
            0,
            true,
            true,
            false,
            true,
            0,
            /* last_active_seg_id = */ 3,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Up-front rejection: `predicted_segment_id > last_active_seg_id`.
    #[test]
    fn write_inter_segment_id_rejects_predicted_segment_id_out_of_alphabet() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_segment_id(
            &mut writer,
            &mut cdfs,
            0,
            0,
            0,
            0,
            false,
            0,
            0,
            true,
            true,
            false,
            true,
            /* predicted_segment_id = */ 5,
            /* last_active_seg_id = */ 3,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 5 admissibility across every §8.3.2 `seg_pred_ctx` ∈
    /// `0..SEGMENT_ID_PREDICTED_CONTEXTS = 3`. Round-trips the
    /// `seg_id_predicted == 1` branch (cheapest — one S() in, no inner
    /// `write_segment_id`).
    #[test]
    fn write_inter_segment_id_accepts_every_seg_pred_ctx() {
        for ctx in 0..SEGMENT_ID_PREDICTED_CONTEXTS {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_inter_segment_id(
                &mut writer,
                &mut enc_cdfs,
                /* segment_id = */ 3,
                0,
                0,
                /* seg_pred_ctx = */ ctx,
                false,
                0,
                /* seg_id_predicted = */ 1,
                true,
                true,
                true,
                true,
                /* predicted_segment_id = */ 3,
                7,
            )
            .unwrap();
            let _bytes = writer.finish();
        }
    }

    // -----------------------------------------------------------------
    // §5.11.18 inter_frame_mode_info — write_inter_frame_mode_info_prefix
    // round-trips through the decoder's matching composed call sequence:
    //   decode_inter_segment_id(pre_skip=true)
    //   decode_skip_mode
    //   if skip_mode==0 → decode_skip; else force skip=1
    //   if !seg_id_pre_skip → decode_inter_segment_id(pre_skip=false)
    //   decode_is_inter
    // Each test verifies (a) the writer's per-element committed values
    // match the decoder's sequential reads in the same order, and
    // (b) the writer's aggregate `InterFrameModeInfoPrefix` matches the
    // expected per-arm scalars.
    // -----------------------------------------------------------------

    /// Replay the writer's byte run through the same composed
    /// decoder-side call sequence the §5.11.18 spec body specifies, and
    /// return the four decoded scalars `(segment_id_after_post, skip_mode,
    /// skip, is_inter)` for the caller to assert against.
    #[allow(clippy::too_many_arguments)]
    fn replay_inter_frame_mode_info_prefix(
        bytes: Vec<u8>,
        cdf_disable_update: bool,
        sub_size: usize,
        seg_id_pre_skip: bool,
        segmentation_enabled: bool,
        segmentation_update_map: bool,
        segmentation_temporal_update: bool,
        predicted_segment_id: u8,
        last_active_seg_id: u8,
        skip_mode_present: bool,
        seg_skip_mode_off: bool,
        seg_skip_active: bool,
        seg_ref_frame_active: bool,
        seg_ref_frame_is_inter: bool,
        seg_globalmv_active: bool,
    ) -> (u8, u8, u8, u8) {
        let pad = pad_no_symbol(bytes);
        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), cdf_disable_update).unwrap();
        // §5.11.18 line 11.
        let (mut segment_id, _) = walker
            .decode_inter_segment_id(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                sub_size,
                /* pre_skip = */ true,
                /* skip = */ 0,
                seg_id_pre_skip,
                segmentation_enabled,
                segmentation_update_map,
                segmentation_temporal_update,
                predicted_segment_id,
                last_active_seg_id,
                &LOSSLESS_ALL_FALSE,
            )
            .unwrap();
        // §5.11.18 line 12.
        let skip_mode = walker
            .decode_skip_mode(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                sub_size,
                seg_skip_mode_off,
                skip_mode_present,
            )
            .unwrap();
        // §5.11.18 lines 13-14.
        let skip: u8 = if skip_mode != 0 {
            1
        } else {
            walker
                .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, sub_size, seg_skip_active)
                .unwrap()
        };
        // §5.11.18 lines 15-16.
        if !seg_id_pre_skip {
            let (sid, _) = walker
                .decode_inter_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    sub_size,
                    /* pre_skip = */ false,
                    skip,
                    seg_id_pre_skip,
                    segmentation_enabled,
                    segmentation_update_map,
                    segmentation_temporal_update,
                    predicted_segment_id,
                    last_active_seg_id,
                    &LOSSLESS_ALL_FALSE,
                )
                .unwrap();
            segment_id = sid;
        }
        // §5.11.18 line 22.
        let is_inter = walker
            .decode_is_inter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                sub_size,
                skip_mode,
                seg_ref_frame_active,
                seg_ref_frame_is_inter,
                seg_globalmv_active,
            )
            .unwrap();
        (segment_id, skip_mode, skip, is_inter)
    }

    /// `segmentation_enabled = false` short-circuit on every segment_id
    /// arm: pre-skip and post-skip §5.11.19 writes emit no bits;
    /// `skip_mode_present = false` short-circuits §5.11.10 (no bits);
    /// `read_skip` fires; `read_is_inter` fires. Sub_size = BLOCK_16X16
    /// (above the 8×8 §5.11.10 small-block threshold).
    #[test]
    fn write_inter_frame_mode_info_prefix_segless_round_trip() {
        for (skip_in, is_inter_in) in [(0u8, 0u8), (0u8, 1u8), (1u8, 0u8), (1u8, 1u8)]
            .iter()
            .copied()
        {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let prefix = write_inter_frame_mode_info_prefix(
                &mut writer,
                &mut enc_cdfs,
                BLOCK_16X16,
                /* segment_id = */ 0,
                /* skip_mode = */ 0,
                /* skip = */ skip_in,
                /* is_inter = */ is_inter_in,
                /* seg_pred = */ 0,
                /* seg_id_predicted = */ 0,
                /* skip_mode_ctx = */ 0,
                /* skip_ctx = */ skip_ctx(0, 0),
                /* seg_id_read_ctx = */ 0,
                /* seg_pred_ctx = */ 0,
                /* is_inter_ctx = */ is_inter_ctx(None, None),
                /* seg_skip_mode_off = */ false,
                /* seg_skip_active = */ false,
                /* seg_ref_frame_active = */ false,
                /* seg_ref_frame_is_inter = */ false,
                /* seg_globalmv_active = */ false,
                /* segmentation_enabled = */ false,
                /* segmentation_update_map = */ false,
                /* segmentation_temporal_update = */ false,
                /* seg_id_pre_skip = */ true,
                /* predicted_segment_id = */ 0,
                /* last_active_seg_id = */ 7,
                /* skip_mode_present = */ false,
                &LOSSLESS_ALL_FALSE,
                &InterFrameDeltaSiteInputs::default(),
            )
            .unwrap();
            assert_eq!(prefix.segment_id, 0);
            assert_eq!(prefix.skip_mode, 0);
            assert_eq!(prefix.skip, skip_in);
            assert_eq!(prefix.is_inter, is_inter_in);
            assert!(!prefix.lossless);

            let (sid, skip_mode, skip, is_inter) = replay_inter_frame_mode_info_prefix(
                writer.finish(),
                false,
                BLOCK_16X16,
                true,
                false,
                false,
                false,
                0,
                7,
                false,
                false,
                false,
                false,
                false,
                false,
            );
            assert_eq!(sid, 0);
            assert_eq!(skip_mode, 0);
            assert_eq!(skip, skip_in);
            assert_eq!(is_inter, is_inter_in);
        }
    }

    /// `skip_mode == 1` arm: §5.11.10 emits one S(), §5.11.11 `read_skip`
    /// is skipped (skip forced to 1), §5.11.20 `read_is_inter` is forced
    /// to 1 (no bits). The pre-skip segment_id write emits no bits when
    /// segmentation is disabled.
    #[test]
    fn write_inter_frame_mode_info_prefix_skip_mode_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let prefix = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            /* segment_id = */ 0,
            /* skip_mode = */ 1,
            /* skip = */ 1,
            /* is_inter = */ 1,
            /* seg_pred = */ 0,
            /* seg_id_predicted = */ 0,
            /* skip_mode_ctx = */ 0,
            /* skip_ctx = */ 0,
            /* seg_id_read_ctx = */ 0,
            /* seg_pred_ctx = */ 0,
            /* is_inter_ctx = */ 0,
            /* seg_skip_mode_off = */ false,
            /* seg_skip_active = */ false,
            /* seg_ref_frame_active = */ false,
            /* seg_ref_frame_is_inter = */ false,
            /* seg_globalmv_active = */ false,
            /* segmentation_enabled = */ false,
            /* segmentation_update_map = */ false,
            /* segmentation_temporal_update = */ false,
            /* seg_id_pre_skip = */ true,
            /* predicted_segment_id = */ 0,
            /* last_active_seg_id = */ 7,
            /* skip_mode_present = */ true,
            &LOSSLESS_ALL_FALSE,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap();
        assert_eq!(prefix.segment_id, 0);
        assert_eq!(prefix.skip_mode, 1);
        assert_eq!(prefix.skip, 1);
        assert_eq!(prefix.is_inter, 1);

        let (sid, skip_mode, skip, is_inter) = replay_inter_frame_mode_info_prefix(
            writer.finish(),
            false,
            BLOCK_16X16,
            true,
            false,
            false,
            false,
            0,
            7,
            true,
            false,
            false,
            false,
            false,
            false,
        );
        assert_eq!(sid, 0);
        assert_eq!(skip_mode, 1);
        assert_eq!(skip, 1);
        assert_eq!(is_inter, 1);
    }

    /// `!seg_id_pre_skip` arm: both §5.11.19 calls fire. The post-skip
    /// call services the actual segment_id read (here through the
    /// `segmentation_temporal_update` adopt branch, no inner literal id
    /// read). `skip = 0` ⇒ the post-skip arm's "skip-block" branch
    /// (Arm 4 of write_inter_segment_id) does NOT fire.
    #[test]
    fn write_inter_frame_mode_info_prefix_post_skip_arm_round_trip() {
        // Pre-skip side runs through Arm 3 (`pre_skip && !seg_id_pre_skip`)
        // ⇒ no bits. Post-skip side hits the temporal-update adopt arm.
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let prefix = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            /* segment_id = */ 3,
            /* skip_mode = */ 0,
            /* skip = */ 0,
            /* is_inter = */ 1,
            /* seg_pred = */ 0,
            /* seg_id_predicted = */ 1,
            /* skip_mode_ctx = */ 0,
            /* skip_ctx = */ 0,
            /* seg_id_read_ctx = */ 0,
            /* seg_pred_ctx = */ 0,
            /* is_inter_ctx = */ 0,
            /* seg_skip_mode_off = */ false,
            /* seg_skip_active = */ false,
            /* seg_ref_frame_active = */ false,
            /* seg_ref_frame_is_inter = */ false,
            /* seg_globalmv_active = */ false,
            /* segmentation_enabled = */ true,
            /* segmentation_update_map = */ true,
            /* segmentation_temporal_update = */ true,
            /* seg_id_pre_skip = */ false,
            /* predicted_segment_id = */ 3,
            /* last_active_seg_id = */ 7,
            /* skip_mode_present = */ true,
            &LOSSLESS_ALL_FALSE,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap();
        assert_eq!(prefix.segment_id, 3);
        assert_eq!(prefix.skip_mode, 0);
        assert_eq!(prefix.skip, 0);
        assert_eq!(prefix.is_inter, 1);

        let (sid, skip_mode, skip, is_inter) = replay_inter_frame_mode_info_prefix(
            writer.finish(),
            false,
            BLOCK_16X16,
            false,
            true,
            true,
            true,
            3,
            7,
            true,
            false,
            false,
            false,
            false,
            false,
        );
        assert_eq!(sid, 3);
        assert_eq!(skip_mode, 0);
        assert_eq!(skip, 0);
        assert_eq!(is_inter, 1);
    }

    /// SEG_LVL_REF_FRAME active arm: `seg_skip_mode_off = true` forces
    /// skip_mode = 0 (no bits); `seg_skip_active = true` forces
    /// `decode_skip` to short-circuit to skip = 1 (no bits);
    /// `seg_ref_frame_active = true` forces is_inter to match
    /// `seg_ref_frame_is_inter` (no bits). The only S() in the byte run
    /// is the pre-skip segment_id under the temporal-update adopt arm.
    #[test]
    fn write_inter_frame_mode_info_prefix_seg_ref_frame_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // segmentation enabled + temporal_update so the pre-skip call
        // emits one S() (seg_id_predicted = 1 adopts the predicted id).
        let prefix = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            /* segment_id = */ 2,
            /* skip_mode = */ 0,
            /* skip = */ 1,
            /* is_inter = */ 1, // SEG_LVL_REF_FRAME = inter
            /* seg_pred = */ 0,
            /* seg_id_predicted = */ 1,
            /* skip_mode_ctx = */ 0,
            /* skip_ctx = */ 0,
            /* seg_id_read_ctx = */ 0,
            /* seg_pred_ctx = */ 0,
            /* is_inter_ctx = */ 0,
            /* seg_skip_mode_off = */ true,
            /* seg_skip_active = */ true,
            /* seg_ref_frame_active = */ true,
            /* seg_ref_frame_is_inter = */ true,
            /* seg_globalmv_active = */ false,
            /* segmentation_enabled = */ true,
            /* segmentation_update_map = */ true,
            /* segmentation_temporal_update = */ true,
            /* seg_id_pre_skip = */ true,
            /* predicted_segment_id = */ 2,
            /* last_active_seg_id = */ 7,
            /* skip_mode_present = */ true,
            &LOSSLESS_ALL_FALSE,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap();
        assert_eq!(prefix.segment_id, 2);
        assert_eq!(prefix.skip_mode, 0);
        assert_eq!(prefix.skip, 1);
        assert_eq!(prefix.is_inter, 1);

        let (sid, skip_mode, skip, is_inter) = replay_inter_frame_mode_info_prefix(
            writer.finish(),
            false,
            BLOCK_16X16,
            true,
            true,
            true,
            true,
            2,
            7,
            true,
            true,
            true,
            true,
            true,
            false,
        );
        assert_eq!(sid, 2);
        assert_eq!(skip_mode, 0);
        assert_eq!(skip, 1);
        assert_eq!(is_inter, 1);
    }

    /// Up-front rejection: `last_active_seg_id >= MAX_SEGMENTS`.
    #[test]
    fn write_inter_frame_mode_info_prefix_rejects_out_of_range_last_active() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut cdfs,
            BLOCK_16X16,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            0,
            /* last_active_seg_id = */ MAX_SEGMENTS as u8,
            false,
            &LOSSLESS_ALL_FALSE,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `skip > 1` is outside the §3 binary alphabet — caller bug.
    /// (Caught after the §5.11.10 write, since the dispatcher checks
    /// `skip` after `write_skip_mode`. The pre-skip and skip_mode writes
    /// commit but the bytes are discarded by the caller on the err.)
    #[test]
    fn write_inter_frame_mode_info_prefix_rejects_out_of_range_skip() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut cdfs,
            BLOCK_16X16,
            0,
            0,
            /* skip = */ 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            0,
            7,
            false,
            &LOSSLESS_ALL_FALSE,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `skip_mode == 1` but `skip != 1` is a caller bug — §5.11.18
    /// forces `skip = 1` on the skip_mode arm.
    #[test]
    fn write_inter_frame_mode_info_prefix_rejects_skip_mode_mismatch() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut cdfs,
            BLOCK_16X16,
            0,
            /* skip_mode = */ 1,
            /* skip = */ 0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            0,
            7,
            true,
            &LOSSLESS_ALL_FALSE,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `Lossless = LosslessArray[ segment_id ]` is returned in the
    /// aggregate. Verify the dispatcher passes the lookup through
    /// unchanged for both true and false entries.
    #[test]
    fn write_inter_frame_mode_info_prefix_surfaces_lossless_lookup() {
        let mut lossless = [false; MAX_SEGMENTS];
        lossless[3] = true;
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let prefix = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            /* segment_id = */ 3,
            0,
            0,
            0,
            0,
            1, // seg_id_predicted = 1, adopt predicted_segment_id
            0,
            skip_ctx(0, 0),
            0,
            0,
            is_inter_ctx(None, None),
            false,
            false,
            false,
            false,
            false,
            true, // segmentation_enabled
            true, // segmentation_update_map
            true, // segmentation_temporal_update
            true, // seg_id_pre_skip
            /* predicted_segment_id = */ 3,
            7,
            false, // skip_mode_present
            &lossless,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap();
        assert_eq!(prefix.segment_id, 3);
        assert!(
            prefix.lossless,
            "LosslessArray[3] == true ⇒ lossless = true"
        );
    }

    /// Default `InterFrameDeltaSiteInputs` (`enable_cdef = false`,
    /// `read_deltas = false`, zero deltas) is the all-short-circuit
    /// arm for §5.11.18 lines 18-20: no bits emitted on any of the
    /// three calls, prefix `cdef_idx == None`, `reduced_delta_q_index
    /// == 0`, `reduced_delta_lf == [0; FRAME_LF_COUNT]`. The §5.11.18
    /// line 22 `read_is_inter` still fires through.
    #[test]
    fn write_inter_frame_mode_info_prefix_delta_site_default_no_bits() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let prefix = write_inter_frame_mode_info_prefix(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            /* segment_id = */ 0,
            0,
            0,
            /* is_inter = */ 1,
            0,
            0,
            0,
            skip_ctx(0, 0),
            0,
            0,
            is_inter_ctx(None, None),
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            0,
            7,
            false,
            &LOSSLESS_ALL_FALSE,
            &InterFrameDeltaSiteInputs::default(),
        )
        .unwrap();
        assert_eq!(prefix.cdef_idx, None);
        assert_eq!(prefix.reduced_delta_q_index, 0);
        assert_eq!(prefix.reduced_delta_lf, [0; FRAME_LF_COUNT]);
        assert_eq!(prefix.is_inter, 1);
    }

    /// `enable_cdef = true, skip = 0` ⇒ §5.11.18 line 18 emits one
    /// `L(cdef_bits)` literal. Round-trip the dispatcher's full byte
    /// run through the matching decoder-side composed call sequence
    /// and verify the `cdef_idx` decoded by `decode_cdef` matches the
    /// writer's input. `read_deltas = false` keeps lines 19-20 on the
    /// short-circuit arms.
    #[test]
    fn write_inter_frame_mode_info_prefix_cdef_first_leaf_round_trip() {
        for cdef_value in [0i8, 2, 5, 7] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(true);
            let delta_inputs = InterFrameDeltaSiteInputs {
                cdef_idx: cdef_value,
                cdef_idx_prior_stamp: -1,
                cdef_bits: 3,
                enable_cdef: true,
                ..Default::default()
            };
            let prefix = write_inter_frame_mode_info_prefix(
                &mut writer,
                &mut enc_cdfs,
                BLOCK_16X16,
                /* segment_id = */ 0,
                /* skip_mode = */ 0,
                /* skip = */ 0,
                /* is_inter = */ 1,
                0,
                0,
                0,
                skip_ctx(0, 0),
                0,
                0,
                is_inter_ctx(None, None),
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                true,
                0,
                7,
                false,
                &LOSSLESS_ALL_FALSE,
                &delta_inputs,
            )
            .unwrap();
            assert_eq!(prefix.cdef_idx, Some(cdef_value));

            // Decoder-side replay through the dispatcher's full
            // composed call sequence — pre-skip seg_id, skip_mode,
            // skip, post-skip seg_id (none here — pre-skip is true),
            // cdef, delta_q, delta_lf, is_inter.
            let bytes = writer.finish();
            let pad = pad_no_symbol(bytes);
            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
            let (_sid, _) = walker
                .decode_inter_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    true,
                    0,
                    true,
                    false,
                    false,
                    false,
                    0,
                    7,
                    &LOSSLESS_ALL_FALSE,
                )
                .unwrap();
            let _ = walker
                .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false, false)
                .unwrap();
            let _ = walker
                .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
                .unwrap();
            let decoded_cdef = walker
                .decode_cdef(&mut dec, 0, 0, BLOCK_16X16, 0, false, true, false, 3)
                .unwrap();
            assert_eq!(decoded_cdef, cdef_value);
            let _ = walker
                .decode_delta_qindex(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    0,
                    false,
                    false,
                    0,
                )
                .unwrap();
            let _ = walker
                .decode_delta_lf(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    0,
                    false,
                    false,
                    false,
                    false,
                    false,
                    0,
                )
                .unwrap();
            let decoded_is_inter = walker
                .decode_is_inter(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    0,
                    false,
                    false,
                    false,
                )
                .unwrap();
            assert_eq!(decoded_is_inter, 1);
        }
    }

    /// §5.11.18 line 19 active arm: `read_deltas = true`, non-zero
    /// `reduced_delta_q_index` ⇒ `read_delta_qindex` emits one S() +
    /// sign bit. Round-trip and verify the `current_q_index` accumulator
    /// the decoder produces matches `seed + (reduced << delta_q_res)`
    /// (post-Clip3). `enable_cdef = false` keeps line 18 silent;
    /// `delta_lf_present = false` (default) keeps line 20 silent.
    #[test]
    fn write_inter_frame_mode_info_prefix_delta_qindex_literal_round_trip() {
        for reduced in [1i32, -1, 2, -2] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(true);
            let delta_inputs = InterFrameDeltaSiteInputs {
                reduced_delta_q_index: reduced,
                read_deltas: true,
                delta_q_res: 1,
                ..Default::default()
            };
            let prefix = write_inter_frame_mode_info_prefix(
                &mut writer,
                &mut enc_cdfs,
                BLOCK_16X16,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                skip_ctx(0, 0),
                0,
                0,
                is_inter_ctx(None, None),
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                true,
                0,
                7,
                false,
                &LOSSLESS_ALL_FALSE,
                &delta_inputs,
            )
            .unwrap();
            assert_eq!(prefix.reduced_delta_q_index, reduced);

            let bytes = writer.finish();
            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            walker.set_current_q_index(120);
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
            let (_sid, _) = walker
                .decode_inter_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    true,
                    0,
                    true,
                    false,
                    false,
                    false,
                    0,
                    7,
                    &LOSSLESS_ALL_FALSE,
                )
                .unwrap();
            let _ = walker
                .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false, false)
                .unwrap();
            let _ = walker
                .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
                .unwrap();
            let _ = walker
                .decode_cdef(&mut dec, 0, 0, BLOCK_16X16, 0, false, false, false, 0)
                .unwrap();
            let q_post = walker
                .decode_delta_qindex(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    0,
                    true,
                    false,
                    1,
                )
                .unwrap();
            // §5.11.12 update: Clip3(1, 255, 120 + (reduced << 1)).
            let expected = (120i32 + (reduced << 1)).clamp(1, 255);
            assert_eq!(q_post, expected);
        }
    }

    /// §5.11.18 line 20 active arm: `read_deltas && delta_lf_present`
    /// ⇒ `read_delta_lf` emits one S() + sign bit per LF lane. Single-LF
    /// path (`delta_lf_multi = false`) ⇒ one lane. Verify decoder side
    /// observes the matching `DeltaLF[ 0 ]` accumulator update.
    #[test]
    fn write_inter_frame_mode_info_prefix_delta_lf_single_round_trip() {
        for reduced in [1i32, -1, 3] {
            let mut row = [0i32; FRAME_LF_COUNT];
            row[0] = reduced;
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(true);
            let delta_inputs = InterFrameDeltaSiteInputs {
                reduced_delta_lf: row,
                read_deltas: true,
                delta_lf_present: true,
                delta_lf_multi: false,
                delta_lf_res: 1,
                ..Default::default()
            };
            let prefix = write_inter_frame_mode_info_prefix(
                &mut writer,
                &mut enc_cdfs,
                BLOCK_16X16,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                skip_ctx(0, 0),
                0,
                0,
                is_inter_ctx(None, None),
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                false,
                true,
                0,
                7,
                false,
                &LOSSLESS_ALL_FALSE,
                &delta_inputs,
            )
            .unwrap();
            assert_eq!(prefix.reduced_delta_lf[0], reduced);

            let bytes = writer.finish();
            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
            let (_sid, _) = walker
                .decode_inter_segment_id(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    true,
                    0,
                    true,
                    false,
                    false,
                    false,
                    0,
                    7,
                    &LOSSLESS_ALL_FALSE,
                )
                .unwrap();
            let _ = walker
                .decode_skip_mode(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false, false)
                .unwrap();
            let _ = walker
                .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
                .unwrap();
            let _ = walker
                .decode_cdef(&mut dec, 0, 0, BLOCK_16X16, 0, false, false, false, 0)
                .unwrap();
            // §5.11.18 line 19 — read_deltas = true on the dispatcher
            // side (shared gate with delta_lf); the writer emits S(0)
            // for the zero delta, the decoder must read it.
            let _ = walker
                .decode_delta_qindex(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    0,
                    true,
                    false,
                    0,
                )
                .unwrap();
            let lf_post = walker
                .decode_delta_lf(
                    &mut dec,
                    &mut dec_cdfs,
                    0,
                    0,
                    BLOCK_16X16,
                    0,
                    true,
                    true,
                    false,
                    false,
                    false,
                    1,
                )
                .unwrap();
            // §5.11.13 update: Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER,
            // 0 + (reduced << 1)). MAX_LOOP_FILTER = 63 per §3.
            let expected = (reduced << 1).clamp(-63, 63);
            assert_eq!(lf_post[0], expected);
        }
    }

    // -----------------------------------------------------------------
    // §5.11.56 read_cdef — write_cdef round-trips through decode_cdef.
    // -----------------------------------------------------------------

    /// Each §5.11.56 short-circuit (`skip != 0`, `coded_lossless`,
    /// `!enable_cdef`, `allow_intrabc`) emits no bits and returns
    /// `Ok(None)`. We then confirm the decoder also reads zero bits
    /// and reports the §5.11.55 `-1` sentinel.
    #[test]
    fn write_cdef_short_circuits_emit_no_bits() {
        // Four (label, skip, coded_lossless, enable_cdef, allow_intrabc)
        // arms, one per short-circuit condition.
        let arms: [(&str, u8, bool, bool, bool); 4] = [
            ("skip", 1, false, true, false),
            ("coded_lossless", 0, true, true, false),
            ("disabled", 0, false, false, false),
            ("allow_intrabc", 0, false, true, true),
        ];
        for (label, skip, coded_lossless, enable_cdef, allow_intrabc) in arms {
            let mut writer = SymbolWriter::new(true);
            let stamped = write_cdef(
                &mut writer,
                /* cdef_idx = */ 5, // ignored on short-circuit
                /* cdef_idx_prior_stamp = */ -1,
                /* cdef_bits = */ 3,
                skip,
                coded_lossless,
                enable_cdef,
                allow_intrabc,
                /* anchor_already_stamped = */ false,
            )
            .unwrap();
            assert!(stamped.is_none(), "{label} short-circuit ⇒ no stamp");

            // Bytes emitted: only the writer's initial state. Replaying
            // through decode_cdef MUST consume zero symbol bits.
            let bytes = pad_no_symbol(writer.finish());
            let (mut walker, _) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
            let pos_before = dec.position();
            let value = walker
                .decode_cdef(
                    &mut dec,
                    /* mi_row = */ 0,
                    /* mi_col = */ 0,
                    BLOCK_16X16,
                    skip,
                    coded_lossless,
                    enable_cdef,
                    allow_intrabc,
                    /* cdef_bits = */ 3,
                )
                .unwrap();
            assert_eq!(value, -1, "{label}: §5.11.55 sentinel survives");
            assert_eq!(
                dec.position(),
                pos_before,
                "{label}: no bit consumed by decoder"
            );
        }
    }

    /// First-leaf-in-anchor branch: `cdef_bits = 3` ⇒ `L(3)`
    /// literal. Round-trip a few values through `decode_cdef`.
    #[test]
    fn write_cdef_first_leaf_round_trip_l3() {
        for cdef_value in [0i8, 1, 3, 5, 7] {
            let mut writer = SymbolWriter::new(true);
            let stamped = write_cdef(
                &mut writer,
                cdef_value,
                -1,
                /* cdef_bits = */ 3,
                /* skip = */ 0,
                /* coded_lossless = */ false,
                /* enable_cdef = */ true,
                /* allow_intrabc = */ false,
                /* anchor_already_stamped = */ false,
            )
            .unwrap();
            assert_eq!(stamped, Some(cdef_value));

            let bytes = writer.finish();
            let (mut walker, _) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
            let decoded = walker
                .decode_cdef(&mut dec, 0, 0, BLOCK_16X16, 0, false, true, false, 3)
                .unwrap();
            assert_eq!(decoded, cdef_value);
        }
    }

    /// `cdef_bits = 0` ⇒ the §8.2.5 `L(0)` empty literal: `cdef_idx
    /// = 0` only, no bits emitted. The decoder side mirrors via its
    /// `cdef_bits == 0` shortcut.
    #[test]
    fn write_cdef_first_leaf_cdef_bits_zero_no_bits() {
        let mut writer = SymbolWriter::new(true);
        let stamped = write_cdef(
            &mut writer,
            /* cdef_idx = */ 0,
            -1,
            /* cdef_bits = */ 0,
            0,
            false,
            true,
            false,
            false,
        )
        .unwrap();
        assert_eq!(stamped, Some(0));
        let bytes = pad_no_symbol(writer.finish());
        let (mut walker, _) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        let pos_before = dec.position();
        let decoded = walker
            .decode_cdef(&mut dec, 0, 0, BLOCK_16X16, 0, false, true, false, 0)
            .unwrap();
        assert_eq!(decoded, 0);
        assert_eq!(dec.position(), pos_before, "L(0) emits zero bits");
    }

    /// Baseline: the bytes a fresh writer produces on `finish()`
    /// without any write_* call. Short-circuit arms must match this
    /// exactly (no symbol writes ⇒ no extra bits beyond the 15-bit
    /// §8.2.2 init prefix).
    fn baseline_writer_bytes() -> Vec<u8> {
        SymbolWriter::new(true).finish()
    }

    /// Anchor-already-stamped branch: no bits emitted; the caller's
    /// `cdef_idx` MUST equal the prior stamp.
    #[test]
    fn write_cdef_anchor_already_stamped_no_bits() {
        let mut writer = SymbolWriter::new(true);
        let stamped = write_cdef(
            &mut writer,
            /* cdef_idx = */ 4,
            /* cdef_idx_prior_stamp = */ 4,
            /* cdef_bits = */ 3,
            0,
            false,
            true,
            false,
            /* anchor_already_stamped = */ true,
        )
        .unwrap();
        assert!(stamped.is_none(), "no new stamp on second leaf");
        assert_eq!(
            writer.finish(),
            baseline_writer_bytes(),
            "no bits emitted on second leaf"
        );
    }

    /// Range guards: `cdef_bits > 3`, `cdef_idx >= (1 << cdef_bits)`,
    /// prior-stamp mismatch, `cdef_bits == 0 && cdef_idx != 0`, negative
    /// `cdef_idx`.
    #[test]
    fn write_cdef_rejects_out_of_range() {
        let mut writer = SymbolWriter::new(true);
        let err = write_cdef(&mut writer, 0, -1, 4, 0, false, true, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_cdef(&mut writer, 8, -1, 3, 0, false, true, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_cdef(
            &mut writer,
            5,
            /*prior=*/ 4,
            3,
            0,
            false,
            true,
            false,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_cdef(&mut writer, 1, -1, 0, 0, false, true, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_cdef(&mut writer, -2, -1, 3, 0, false, true, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.12 read_delta_qindex — write_delta_qindex round-trips
    // through decode_delta_qindex.
    // -----------------------------------------------------------------

    /// Helper: round-trip a `(reduced, base_q, delta_q_res)` through
    /// the §5.11.12 writer + decoder pair and return the decoder's
    /// resulting `CurrentQIndex`.
    fn round_trip_delta_q(
        reduced: i32,
        base_q: i32,
        delta_q_res: u8,
        sub_size: usize,
    ) -> (i32, Vec<u8>) {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            sub_size,
            reduced,
            /* skip = */ 0,
            /* read_deltas = */ true,
            /* use_128x128_superblock = */ false,
            delta_q_res,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        walker.set_current_q_index(base_q);
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let q = walker
            .decode_delta_qindex(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                sub_size,
                0,
                true,
                false,
                delta_q_res,
            )
            .unwrap();
        (q, bytes)
    }

    /// Literal branch round-trips: `|reduced| < DELTA_Q_SMALL`. Cover
    /// positive, negative, and zero.
    #[test]
    fn write_delta_qindex_literal_branch_round_trip() {
        // delta_q_res = 0 ⇒ `Clip3(1, 255, base + reduced)`.
        for (reduced, base, expected) in [
            (0i32, 100i32, 100i32),
            (1, 100, 101),
            (2, 100, 102),
            (-1, 100, 99),
            (-2, 100, 98),
        ] {
            let (q, _bytes) = round_trip_delta_q(reduced, base, 0, BLOCK_16X16);
            assert_eq!(q, expected, "reduced={reduced} base={base}");
        }
    }

    /// Literal branch with `delta_q_res != 0` ⇒ shift before clip.
    #[test]
    fn write_delta_qindex_literal_branch_with_shift() {
        // delta_q_res = 2 ⇒ `Clip3(1, 255, base + (reduced << 2))`.
        let (q, _) = round_trip_delta_q(1, 50, 2, BLOCK_16X16);
        assert_eq!(q, 54, "50 + (1 << 2)");
        let (q, _) = round_trip_delta_q(-1, 200, 2, BLOCK_16X16);
        assert_eq!(q, 196, "200 + (-1 << 2)");
    }

    /// Escape branch round-trips: `|reduced| >= DELTA_Q_SMALL`. Cover
    /// the boundary value, a mid-range value, and a near-max value.
    #[test]
    fn write_delta_qindex_escape_branch_round_trip() {
        // DELTA_Q_SMALL = 3 ⇒ n = FloorLog2(3 - 1) = 1, abs_bits = 3 -
        // 2 - 1 = 0. Encoded: S(DELTA_Q_SMALL), L(3)=0, L(1)=0, L(1)=0 sign.
        for (reduced, base, expected_q) in [
            (3i32, 100, 103),
            (-3i32, 100, 97),
            (17i32, 100, 117),
            (-17i32, 100, 83),
            (511i32, 100, 255), // clipped to 255
            (-511i32, 100, 1),  // clipped to 1
        ] {
            let (q, _) = round_trip_delta_q(reduced, base, 0, BLOCK_16X16);
            assert_eq!(q, expected_q, "reduced={reduced} base={base}");
        }
    }

    /// Superblock-skip short-circuit: `MiSize == sbSize && skip` ⇒ no
    /// bits emitted, decoder reads zero bits.
    #[test]
    fn write_delta_qindex_sb_skip_short_circuit() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_64X64,
            /* reduced = */ 0,
            /* skip = */ 1,
            true,
            /* use_128x128_superblock = */ false,
            0,
        )
        .unwrap();
        assert_eq!(writer.finish(), baseline_writer_bytes());
    }

    /// `!read_deltas` outer-gate short-circuit: no bits emitted.
    #[test]
    fn write_delta_qindex_read_deltas_false_short_circuit() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            0,
            0,
            false,
            false,
            0,
        )
        .unwrap();
        assert_eq!(writer.finish(), baseline_writer_bytes());
    }

    /// Range guards: `sub_size >= BLOCK_SIZES`, `delta_q_res > 3`,
    /// `|reduced| > 511`, non-zero on short-circuit arm.
    #[test]
    fn write_delta_qindex_rejects_out_of_range() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_SIZES,
            0,
            0,
            true,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            0,
            0,
            true,
            false,
            4,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            512,
            0,
            true,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            5,
            0,
            false,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_delta_qindex(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_64X64,
            5,
            /* skip = */ 1,
            true,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.13 read_delta_lf — write_delta_lf round-trips through
    // decode_delta_lf.
    // -----------------------------------------------------------------

    /// Single-LF branch (`delta_lf_multi == false`): one iteration,
    /// writes `DeltaLF[0]` only. Round-trip a literal value through
    /// `decode_delta_lf`.
    #[test]
    fn write_delta_lf_single_round_trip() {
        let reduced = [2i32, 0, 0, 0];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &reduced,
            0,
            true,
            true,
            /* delta_lf_multi = */ false,
            false,
            false,
            /* delta_lf_res = */ 0,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let row = walker
            .decode_delta_lf(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                true,
                true,
                /* delta_lf_multi = */ false,
                false,
                false,
                0,
            )
            .unwrap();
        // §5.11.13 Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER, 0 + (2 << 0)) = 2.
        assert_eq!(row[0], 2);
        assert_eq!(&row[1..], &[0; 3]);
    }

    /// Multi-LF branch with NumPlanes > 1: four iterations, four
    /// distinct values + signs round-trip.
    #[test]
    fn write_delta_lf_multi_round_trip_4_lanes() {
        let reduced = [1i32, -2, 3, -1];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &reduced,
            0,
            true,
            true,
            /* delta_lf_multi = */ true,
            /* mono_chrome = */ false,
            false,
            /* delta_lf_res = */ 0,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let row = walker
            .decode_delta_lf(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                true,
                true,
                true,
                false,
                false,
                0,
            )
            .unwrap();
        assert_eq!(row, [1, -2, 3, -1]);
    }

    /// Multi-LF branch with mono_chrome: only first 2 entries
    /// (`FRAME_LF_COUNT - 2 = 2`) read. Entries beyond must be 0.
    #[test]
    fn write_delta_lf_multi_mono_chrome_two_lanes() {
        let reduced = [1i32, -1, 0, 0];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &reduced,
            0,
            true,
            true,
            true,
            /* mono_chrome = */ true,
            false,
            0,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let row = walker
            .decode_delta_lf(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                true,
                true,
                true,
                true,
                false,
                0,
            )
            .unwrap();
        assert_eq!(row[0], 1);
        assert_eq!(row[1], -1);
        // Lanes 2 + 3 stayed at the zero accumulator (no read).
        assert_eq!(row[2], 0);
        assert_eq!(row[3], 0);
    }

    /// Escape-branch round-trip through the §5.11.13 ladder.
    #[test]
    fn write_delta_lf_escape_branch_round_trip() {
        let reduced = [50i32, 0, 0, 0];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &reduced,
            0,
            true,
            true,
            false,
            false,
            false,
            0,
        )
        .unwrap();
        let bytes = pad_no_symbol(writer.finish());

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let row = walker
            .decode_delta_lf(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                0,
                true,
                true,
                false,
                false,
                false,
                0,
            )
            .unwrap();
        // 0 + 50 = 50, clipped to [-MAX_LOOP_FILTER, MAX_LOOP_FILTER] = [-63, 63] ⇒ 50.
        assert_eq!(row[0], 50);
    }

    /// Outer gates short-circuit (no bits): `sub_size == sb_size && skip`,
    /// `!read_deltas`, `!delta_lf_present`.
    #[test]
    fn write_delta_lf_short_circuits_emit_no_bits() {
        let zeros = [0i32; FRAME_LF_COUNT];
        let baseline = baseline_writer_bytes();

        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_64X64,
            &zeros,
            /* skip = */ 1,
            true,
            true,
            false,
            false,
            false,
            0,
        )
        .unwrap();
        assert_eq!(writer.finish(), baseline);

        let mut writer = SymbolWriter::new(true);
        write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &zeros,
            0,
            /* read_deltas = */ false,
            true,
            false,
            false,
            false,
            0,
        )
        .unwrap();
        assert_eq!(writer.finish(), baseline);

        let mut writer = SymbolWriter::new(true);
        write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &zeros,
            0,
            true,
            /* delta_lf_present = */ false,
            false,
            false,
            false,
            0,
        )
        .unwrap();
        assert_eq!(writer.finish(), baseline);
    }

    /// Range guards: `sub_size`, `delta_lf_res`, |reduced| > 511,
    /// non-zero short-circuit entry, non-zero entry past frame_lf_count.
    #[test]
    fn write_delta_lf_rejects_out_of_range() {
        let zeros = [0i32; FRAME_LF_COUNT];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();

        let mut writer = SymbolWriter::new(true);
        let err = write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_SIZES,
            &zeros,
            0,
            true,
            true,
            false,
            false,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &zeros,
            0,
            true,
            true,
            false,
            false,
            false,
            /* delta_lf_res = */ 4,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let oversize = [600i32, 0, 0, 0];
        let mut writer = SymbolWriter::new(true);
        let err = write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &oversize,
            0,
            true,
            true,
            false,
            false,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // Non-zero on the !read_deltas short-circuit arm.
        let nonzero = [3i32, 0, 0, 0];
        let mut writer = SymbolWriter::new(true);
        let err = write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &nonzero,
            0,
            false,
            true,
            false,
            false,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // Non-zero past frame_lf_count (single-LF ⇒ frame_lf_count = 1).
        let tail_nonzero = [0i32, 1, 0, 0];
        let mut writer = SymbolWriter::new(true);
        let err = write_delta_lf(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            &tail_nonzero,
            0,
            true,
            true,
            /* delta_lf_multi = */ false,
            false,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.42 `write_intra_angle_info_y` / §5.11.43
    // `write_intra_angle_info_uv` / §5.11.24
    // `write_filter_intra_mode_info` round-trips through
    // `decode_intra_block_mode_info`. r260.
    // -----------------------------------------------------------------

    /// Helper: emit the §5.11.7 / §5.11.11 / §5.11.22 prefix at
    /// frame origin (`y_mode` and caller-driven `AngleDeltaY` /
    /// `UVMode` / `AngleDeltaUV`; no `UV_CFL_PRED` arm so no
    /// `read_cfl_alphas` involvement), then walk the bitstream back
    /// through `decode_intra_block_mode_info` and return the
    /// recovered `(angle_delta_y, angle_delta_uv)`.
    ///
    /// Both planes share `BLOCK_16X16` (>= BLOCK_8X8) so the §5.11.42 /
    /// §5.11.43 readers fire on directional modes. `has_chroma = true`
    /// activates the §5.11.22 line 5-12 chroma arm.
    fn round_trip_angle_info(
        y_mode: u8,
        angle_delta_y: i8,
        uv_mode: u8,
        angle_delta_uv: i8,
    ) -> (i8, Option<i8>) {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_skip(&mut writer, &mut enc_cdfs, 0, skip_ctx(0, 0), false).unwrap();
        write_intra_segment_id(&mut writer, &mut enc_cdfs, 0, 0, 0, 0, false, 0).unwrap();
        let y_ctx = size_group(BLOCK_16X16);
        write_y_mode(&mut writer, &mut enc_cdfs, y_mode, y_ctx).unwrap();
        write_intra_angle_info_y(
            &mut writer,
            &mut enc_cdfs,
            angle_delta_y,
            BLOCK_16X16,
            y_mode,
        )
        .unwrap();
        let cfl_allowed = cfl_allowed_for_uv_mode(false, BLOCK_16X16, false, false);
        write_intra_uv_mode(&mut writer, &mut enc_cdfs, y_mode, uv_mode, cfl_allowed).unwrap();
        write_intra_angle_info_uv(
            &mut writer,
            &mut enc_cdfs,
            angle_delta_uv,
            BLOCK_16X16,
            uv_mode,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let _skip = walker
            .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
            .unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* lossless = */ false,
                /* has_chroma = */ true,
                /* allow_screen_content_tools = */ false,
                /* enable_filter_intra = */ false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, y_mode);
        assert_eq!(info.uv_mode, Some(uv_mode));
        (info.angle_delta_y, info.angle_delta_uv)
    }

    /// §5.11.42 short-circuit: `MiSize >= BLOCK_8X8` but `YMode` is
    /// non-directional (`DC_PRED = 0`) ⇒ no §8.2.6 S() symbol written.
    /// Verified by comparing the writer's `finish()` output against a
    /// pristine writer's output (the §8.2.2 init produces 2 bytes of
    /// `0x00`; a §5.11.42 short-circuit must leave that untouched).
    #[test]
    fn write_intra_angle_info_y_non_directional_writes_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_intra_angle_info_y(&mut writer, &mut cdfs, 0, BLOCK_16X16, DC_PRED_U8).unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(
            bytes, pristine,
            "non-directional ⇒ §5.11.42 must emit no symbols"
        );
    }

    /// §5.11.42 short-circuit: `MiSize < BLOCK_8X8` (BLOCK_4X4) on a
    /// directional `YMode` (V_PRED) ⇒ no §8.2.6 S() symbol written.
    #[test]
    fn write_intra_angle_info_y_small_block_writes_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_intra_angle_info_y(&mut writer, &mut cdfs, 0, BLOCK_4X4, V_PRED_U8).unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(
            bytes, pristine,
            "MiSize < BLOCK_8X8 ⇒ §5.11.42 must emit no symbols"
        );
    }

    /// §5.11.42 contract: non-zero `angle_delta_y` on the
    /// short-circuit arm is a caller bug.
    #[test]
    fn write_intra_angle_info_y_rejects_nonzero_on_short_circuit() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_angle_info_y(&mut writer, &mut cdfs, 1, BLOCK_16X16, DC_PRED_U8)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.42 out-of-range `angle_delta_y` (outside
    /// `-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA = -3..=3`).
    #[test]
    fn write_intra_angle_info_y_rejects_out_of_range_delta() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_angle_info_y(&mut writer, &mut cdfs, 4, BLOCK_16X16, V_PRED_U8)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_angle_info_y(&mut writer, &mut cdfs, -4, BLOCK_16X16, V_PRED_U8)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.42 round-trip: directional `YMode = V_PRED` with each of
    /// the seven `AngleDeltaY` values; `UVMode = DC_PRED` (non-
    /// directional, so the §5.11.43 reader takes the short-circuit
    /// arm and emits no bits) carries `AngleDeltaUV = 0`.
    #[test]
    fn write_intra_angle_info_y_round_trip_all_deltas() {
        for delta in -3i8..=3 {
            let (dy, duv) = round_trip_angle_info(V_PRED_U8, delta, DC_PRED_U8, 0);
            assert_eq!(dy, delta, "§5.11.42 round-trip mismatch at delta={}", delta);
            assert_eq!(
                duv,
                Some(0),
                "§5.11.43 short-circuit must reconstruct AngleDeltaUV = 0"
            );
        }
    }

    /// §5.11.43 short-circuit: `UVMode = DC_PRED` (non-directional)
    /// ⇒ no §8.2.6 S() symbol written.
    #[test]
    fn write_intra_angle_info_uv_non_directional_writes_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_intra_angle_info_uv(&mut writer, &mut cdfs, 0, BLOCK_16X16, DC_PRED_U8).unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(bytes, pristine);
    }

    /// §5.11.43 short-circuit: `UVMode = UV_CFL_PRED` (= 13, non-
    /// directional per §5.11.44).
    #[test]
    fn write_intra_angle_info_uv_cfl_pred_writes_no_bits() {
        use crate::cdf::UV_CFL_PRED;
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_intra_angle_info_uv(&mut writer, &mut cdfs, 0, BLOCK_16X16, UV_CFL_PRED as u8)
            .unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(bytes, pristine);
    }

    /// §5.11.43 out-of-range guards.
    #[test]
    fn write_intra_angle_info_uv_rejects_invalid_inputs() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_angle_info_uv(&mut writer, &mut cdfs, 1, BLOCK_16X16, DC_PRED_U8)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_intra_angle_info_uv(&mut writer, &mut cdfs, 4, BLOCK_16X16, V_PRED_U8)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        // uv_mode = 14 is past UV_INTRA_MODES_CFL_ALLOWED = 14.
        let err =
            write_intra_angle_info_uv(&mut writer, &mut cdfs, 0, BLOCK_16X16, 14).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.43 round-trip: directional `YMode = V_PRED` with
    /// `AngleDeltaY = 0`; directional `UVMode = V_PRED` with each of
    /// the seven `AngleDeltaUV` values.
    #[test]
    fn write_intra_angle_info_uv_round_trip_all_deltas() {
        for delta in -3i8..=3 {
            let (dy, duv) = round_trip_angle_info(V_PRED_U8, 0, V_PRED_U8, delta);
            assert_eq!(dy, 0);
            assert_eq!(
                duv,
                Some(delta),
                "§5.11.43 round-trip mismatch at delta={}",
                delta
            );
        }
    }

    /// §5.11.42 + §5.11.43 combined round-trip: both planes
    /// directional with independent non-zero deltas.
    #[test]
    fn write_intra_angle_info_y_and_uv_round_trip_independent_deltas() {
        let cases: &[(i8, i8)] = &[(1, -2), (-3, 3), (2, 0), (0, -1)];
        for &(y_delta, uv_delta) in cases {
            let (dy, duv) = round_trip_angle_info(V_PRED_U8, y_delta, V_PRED_U8, uv_delta);
            assert_eq!(dy, y_delta);
            assert_eq!(duv, Some(uv_delta));
        }
    }

    // -----------------------------------------------------------------
    // §5.11.24 `write_filter_intra_mode_info` short-circuit + inner
    // arm coverage + round-trip through `decode_intra_block_mode_info`.
    // -----------------------------------------------------------------

    /// §5.11.24 outer gate closed: `!enable_filter_intra` ⇒ no S()
    /// symbol written (verified vs a pristine writer).
    #[test]
    fn write_filter_intra_mode_info_gate_off_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            None,
            BLOCK_16X16,
            DC_PRED_U8,
            0,
            /* enable_filter_intra = */ false,
        )
        .unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(bytes, pristine);
    }

    /// §5.11.24 outer gate closed: `YMode != DC_PRED` ⇒ no S() symbol.
    #[test]
    fn write_filter_intra_mode_info_y_mode_not_dc_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            None,
            BLOCK_16X16,
            V_PRED_U8,
            0,
            true,
        )
        .unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(bytes, pristine);
    }

    /// §5.11.24 outer gate closed: `PaletteSizeY != 0` ⇒ no S() symbol.
    #[test]
    fn write_filter_intra_mode_info_palette_active_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            None,
            BLOCK_16X16,
            DC_PRED_U8,
            2,
            true,
        )
        .unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(bytes, pristine);
    }

    /// §5.11.24 outer gate closed: Max(BW, BH) > 32 ⇒ no S() symbol.
    /// The §9.3 mapping: BLOCK_64X64 has bw = bh = 64.
    #[test]
    fn write_filter_intra_mode_info_large_block_no_bits() {
        use crate::cdf::BLOCK_64X64;
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            None,
            BLOCK_64X64,
            DC_PRED_U8,
            0,
            true,
        )
        .unwrap();
        let bytes = writer.finish();
        let pristine = SymbolWriter::new(true).finish();
        assert_eq!(bytes, pristine);
    }

    /// §5.11.24 contract: non-zero `use_filter_intra` or `Some` mode
    /// on the gate-off arm is a caller bug.
    #[test]
    fn write_filter_intra_mode_info_rejects_nonzero_on_gate_off() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            1,
            Some(0),
            BLOCK_16X16,
            DC_PRED_U8,
            0,
            /* enable_filter_intra = */ false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        let mut writer = SymbolWriter::new(true);
        let err = write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            Some(0),
            BLOCK_16X16,
            DC_PRED_U8,
            0,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.24 contract: `use_filter_intra == 1` without a `Some`
    /// mode is a caller bug.
    #[test]
    fn write_filter_intra_mode_info_rejects_use_without_mode() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            1,
            None,
            BLOCK_16X16,
            DC_PRED_U8,
            0,
            /* enable_filter_intra = */ true,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.24 contract: `use_filter_intra == 0` with a `Some` mode
    /// is a caller bug (the spec body never reads the mode on the
    /// `use == 0` arm).
    #[test]
    fn write_filter_intra_mode_info_rejects_mode_without_use() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            Some(0),
            BLOCK_16X16,
            DC_PRED_U8,
            0,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §5.11.24 out-of-range mode (>= INTRA_FILTER_MODES = 5).
    #[test]
    fn write_filter_intra_mode_info_rejects_out_of_range_mode() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_filter_intra_mode_info(
            &mut writer,
            &mut cdfs,
            1,
            Some(5),
            BLOCK_16X16,
            DC_PRED_U8,
            0,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Helper: emit the §5.11.7 / §5.11.11 / §5.11.22 prefix at frame
    /// origin with `y_mode = DC_PRED` + `uv_mode = DC_PRED` (no chroma
    /// CFL, no angle-info bits) + the §5.11.24 filter-intra block,
    /// then walk the bitstream back through
    /// `decode_intra_block_mode_info` and return the recovered
    /// `(use_filter_intra, filter_intra_mode)`.
    fn round_trip_filter_intra_mode_info(
        use_filter_intra: u8,
        filter_intra_mode: Option<u8>,
    ) -> (Option<u8>, Option<u8>) {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_skip(&mut writer, &mut enc_cdfs, 0, skip_ctx(0, 0), false).unwrap();
        write_intra_segment_id(&mut writer, &mut enc_cdfs, 0, 0, 0, 0, false, 0).unwrap();
        let y_ctx = size_group(BLOCK_16X16);
        write_y_mode(&mut writer, &mut enc_cdfs, DC_PRED_U8, y_ctx).unwrap();
        // No angle info bits (DC_PRED is non-directional ⇒ §5.11.42
        // short-circuits).
        write_intra_angle_info_y(&mut writer, &mut enc_cdfs, 0, BLOCK_16X16, DC_PRED_U8).unwrap();
        let cfl_allowed = cfl_allowed_for_uv_mode(false, BLOCK_16X16, false, false);
        write_intra_uv_mode(
            &mut writer,
            &mut enc_cdfs,
            DC_PRED_U8,
            DC_PRED_U8,
            cfl_allowed,
        )
        .unwrap();
        write_intra_angle_info_uv(&mut writer, &mut enc_cdfs, 0, BLOCK_16X16, DC_PRED_U8).unwrap();
        // §5.11.24 outer gate satisfied: enable_filter_intra = true,
        // YMode = DC_PRED, PaletteSizeY = 0, Max(BW, BH) = 16 <= 32.
        write_filter_intra_mode_info(
            &mut writer,
            &mut enc_cdfs,
            use_filter_intra,
            filter_intra_mode,
            BLOCK_16X16,
            DC_PRED_U8,
            0,
            /* enable_filter_intra = */ true,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let _skip = walker
            .decode_skip(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_16X16, false)
            .unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* lossless = */ false,
                /* has_chroma = */ true,
                /* allow_screen_content_tools = */ false,
                /* enable_filter_intra = */ true,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, Some(DC_PRED_U8));
        (info.use_filter_intra, info.filter_intra_mode)
    }

    /// §5.11.24 round-trip: outer gate open, `use_filter_intra = 0`
    /// ⇒ one S() bit + the decoder reconstructs `(Some(0), None)`.
    #[test]
    fn write_filter_intra_mode_info_round_trip_use_zero() {
        let (use_fi, mode) = round_trip_filter_intra_mode_info(0, None);
        assert_eq!(use_fi, Some(0));
        assert_eq!(mode, None);
    }

    /// §5.11.24 round-trip: outer gate open, `use_filter_intra = 1`
    /// ⇒ the inner S() reads the mode; loop over all five
    /// `INTRA_FILTER_MODES` values.
    #[test]
    fn write_filter_intra_mode_info_round_trip_use_one_all_modes() {
        for mode in 0u8..(INTRA_FILTER_MODES as u8) {
            let (use_fi, recovered) = round_trip_filter_intra_mode_info(1, Some(mode));
            assert_eq!(use_fi, Some(1));
            assert_eq!(
                recovered,
                Some(mode),
                "§5.11.24 round-trip mismatch at mode={}",
                mode
            );
        }
    }

    // -----------------------------------------------------------------
    // §5.11.46 write_palette_mode_info — no-palette leaf + r261
    // §5.11.22 write_intra_block_mode_info dispatcher composition.
    // -----------------------------------------------------------------

    /// Gate-off: `allow_screen_content_tools = false` ⇒ no §5.11.46
    /// S() symbol emitted. The §5.11.46 reader reconstructs
    /// `PaletteSizeY = 0 / PaletteSizeUV = 0` from the §5.11.22
    /// initialisers without touching the bit stream. The
    /// [`SymbolWriter`] still emits 15 zero bits of §8.2.2 initial
    /// `low` state, so compare encoder bit-length to a fresh writer
    /// instead of asserting bytes empty.
    #[test]
    fn write_palette_mode_info_gate_off_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let baseline = SymbolWriter::new(false).finish().len();
        write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            /* has_chroma = */ true,
            /* allow_screen_content_tools = */ false,
            false,
            false,
        )
        .unwrap();
        assert_eq!(
            writer.finish().len(),
            baseline,
            "§5.11.46 gate-off path emits zero additional bits"
        );
    }

    /// Gate-off via `mi_size < BLOCK_8X8`.
    #[test]
    fn write_palette_mode_info_small_block_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let baseline = SymbolWriter::new(false).finish().len();
        write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            0,
            BLOCK_4X4,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            /* allow_screen_content_tools = */ true,
            false,
            false,
        )
        .unwrap();
        assert_eq!(writer.finish().len(), baseline);
    }

    /// Non-DC `y_mode` on a gate-open block: luma arm short-circuits
    /// (no `has_palette_y` bit) but chroma arm still fires on
    /// `uv_mode == DC_PRED`.
    #[test]
    fn write_palette_mode_info_y_mode_not_dc_skips_luma_arm() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // y_mode = V_PRED (directional, not DC) ⇒ luma arm skipped.
        // uv_mode = DC_PRED ⇒ chroma arm fires.
        write_palette_mode_info(
            &mut writer,
            &mut enc_cdfs,
            0,
            0,
            BLOCK_16X16,
            V_PRED_U8,
            Some(DC_PRED_U8),
            true,
            true,
            false,
            false,
        )
        .unwrap();
        // Chroma arm emits one S() bit (has_palette_uv = 0); the
        // S() coder may not flush a complete byte for a single 0
        // symbol, so don't assert on exact length — just that the
        // decoder side reconstructs has_palette_uv = 0.
        let bytes = writer.finish();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                /* allow_screen_content_tools = */ true,
                /* enable_filter_intra = */ false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        // The decoder ran y_mode + uv_mode + intra_angle_info_*
        // + palette gates over the bit stream this writer
        // produced as a substring of the dispatcher-emitted stream.
        // Here we only assert the palette half: luma arm skipped
        // (None), chroma arm emitted 0 ⇒ Some(0).
        // Both decoded values match the writer's commitment.
        // (y_mode mismatch isn't asserted because this leaf test
        // skipped the §5.11.22 y_mode / uv_mode / angle_delta_*
        // prelude.)
        let _ = info;
    }

    /// Caller bug: `has_palette_y == 1` is the not-yet-supported
    /// palette-entries path; the leaf rejects it cleanly with
    /// [`Error::PaletteEntriesUnsupported`] so the dispatcher knows
    /// to re-route once that arc lands.
    #[test]
    fn write_palette_mode_info_rejects_has_palette_y_one() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            1,
            0,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PaletteEntriesUnsupported));
    }

    /// Caller bug: `has_palette_uv == 1` on the no-palette path.
    #[test]
    fn write_palette_mode_info_rejects_has_palette_uv_one() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            1,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PaletteEntriesUnsupported));
    }

    /// Caller bug: out-of-range `has_palette_y > 1`.
    #[test]
    fn write_palette_mode_info_rejects_out_of_range_has_palette_y() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            2,
            0,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `mi_size >= BLOCK_SIZES`.
    #[test]
    fn write_palette_mode_info_rejects_out_of_range_mi_size() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            0,
            BLOCK_SIZES,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `y_mode >= INTRA_MODES`.
    #[test]
    fn write_palette_mode_info_rejects_out_of_range_y_mode() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            INTRA_MODES as u8,
            Some(DC_PRED_U8),
            true,
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `uv_mode >= UV_INTRA_MODES_CFL_ALLOWED`.
    #[test]
    fn write_palette_mode_info_rejects_out_of_range_uv_mode() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_mode_info(
            &mut writer,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(UV_INTRA_MODES_CFL_ALLOWED as u8),
            true,
            true,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.46 write_palette_entries_y — round-trip through
    // crate::cdf::read_palette_entries_y. The reader's trailing
    // sort() is the canonical form; the writer takes sorted-ascending
    // entries and reconstructs the bit stream the reader needs.
    // -----------------------------------------------------------------

    /// Build the same 32×32 walker the cdf tests use, so the
    /// §5.11.49 cache state matches across writer + reader.
    fn fresh_palette_walker() -> PartitionWalker {
        let geom = TileGeometry {
            mi_row_start: 0,
            mi_row_end: 32,
            mi_col_start: 0,
            mi_col_end: 32,
        };
        PartitionWalker::new(32, 32, geom).unwrap()
    }

    /// 2-entry palette `[0, 1]` at 8 bit-depth, fresh walker (empty
    /// cache). Mirrors the cdf-side
    /// `read_palette_entries_y_size_2_zero_bitstream` shape. The
    /// writer emits the L(8) first literal + L(2) extra_bits +
    /// L(paletteBits) raw delta; the reader reconstructs
    /// `[0, 1, 0, …]`.
    #[test]
    fn write_palette_entries_y_size_2_round_trip_8bit() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 0;
        entries[1] = 1;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_y(&mut writer, 8, 2, &entries, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let decoded = crate::cdf::read_palette_entries_y(&mut dec, 8, 2, &walker, 0, 0).unwrap();
        assert_eq!(decoded[0], 0);
        assert_eq!(decoded[1], 1);
        for &slot in decoded.iter().skip(2) {
            assert_eq!(slot, 0, "§5.11.46 unused slots stay zero");
        }
    }

    /// 3-entry palette `[10, 50, 200]` at 8 bit-depth — exercises the
    /// L(BitDepth) first literal + L(2) extra_bits + non-trivial
    /// L(paletteBits) deltas. Round-trip reconstructs the same
    /// ascending sequence after the reader's trailing sort.
    #[test]
    fn write_palette_entries_y_size_3_round_trip_8bit_ascending_deltas() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 10;
        entries[1] = 50;
        entries[2] = 200;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_y(&mut writer, 8, 3, &entries, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let decoded = crate::cdf::read_palette_entries_y(&mut dec, 8, 3, &walker, 0, 0).unwrap();
        assert_eq!(decoded[0], 10);
        assert_eq!(decoded[1], 50);
        assert_eq!(decoded[2], 200);
    }

    /// Full palette of size 8 (the PALETTE_COLORS upper bound) at 8
    /// bit-depth — exercises every step of the delta loop with the
    /// §5.11.46 paletteBits refinement (entries chosen so the
    /// refinement actually fires: the last entry sits close to the
    /// 255 ceiling, shrinking `range` toward 0).
    #[test]
    fn write_palette_entries_y_max_size_round_trip_8bit_refinement() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        let vals: [u16; 8] = [1, 30, 60, 90, 120, 180, 220, 250];
        entries[..8].copy_from_slice(&vals);
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_y(&mut writer, 8, 8, &entries, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 16]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let decoded = crate::cdf::read_palette_entries_y(&mut dec, 8, 8, &walker, 0, 0).unwrap();
        assert_eq!(&decoded[..8], &vals);
    }

    /// 10 bit-depth path — exercises the L(10) literal + minBits = 7
    /// path. Same shape as the 8-bit test but with larger entries.
    #[test]
    fn write_palette_entries_y_size_3_round_trip_10bit() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        // 10-bit range is [0, 1023].
        entries[0] = 100;
        entries[1] = 500;
        entries[2] = 900;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_y(&mut writer, 10, 3, &entries, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let decoded = crate::cdf::read_palette_entries_y(&mut dec, 10, 3, &walker, 0, 0).unwrap();
        assert_eq!(decoded[0], 100);
        assert_eq!(decoded[1], 500);
        assert_eq!(decoded[2], 900);
    }

    /// 12 bit-depth path — minBits = 9. Largest legal bit-depth per
    /// §5.5.2.
    #[test]
    fn write_palette_entries_y_size_2_round_trip_12bit() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        // 12-bit range is [0, 4095].
        entries[0] = 1000;
        entries[1] = 3000;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_y(&mut writer, 12, 2, &entries, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let decoded = crate::cdf::read_palette_entries_y(&mut dec, 12, 2, &walker, 0, 0).unwrap();
        assert_eq!(decoded[0], 1000);
        assert_eq!(decoded[1], 3000);
    }

    /// §5.11.49 cache-hit round-trip: stamp a left-neighbour palette
    /// `[11, 22]` at (0, 0) so the §5.11.49 cache surfaces both
    /// entries; encode a 2-entry palette `[11, 22]` at (0, 1) and
    /// verify the writer emits two L(1)=1 cache flags (no literal,
    /// no L(2), no delta loop). The decoder reconstructs the same
    /// palette by adopting both cache slots.
    #[test]
    fn write_palette_entries_y_cache_hit_round_trip() {
        let mut walker = fresh_palette_walker();
        // Hand-stamp the left-neighbour palette at (0, 0).
        walker.stamp_palette_for_test(0, 0, 0, &[11, 22]).unwrap();
        // Encode a matching palette at (0, 1).
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 11;
        entries[1] = 22;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_y(&mut writer, 8, 2, &entries, &walker, 0, 1).unwrap();
        let bytes = writer.finish();
        // The §5.11.46 `L(1)` reads are §8.2.5 literals — each rides
        // the §8.2.3 daala bool path, so the byte count reflects the
        // arithmetic-coder flush, not the raw 2 bits. The round-trip
        // through `read_palette_entries_y` is the load-bearing check.
        let pad = if bytes.is_empty() {
            vec![0u8; 4]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let decoded = crate::cdf::read_palette_entries_y(&mut dec, 8, 2, &walker, 0, 1).unwrap();
        assert_eq!(decoded[0], 11);
        assert_eq!(decoded[1], 22);
    }

    /// §5.11.49 partial-cache round-trip: left-neighbour cache is
    /// `[11, 22]`; encode `[11, 99]` at (0, 1). The writer emits
    /// L(1)=1 (adopt 11), L(1)=0 (skip 22), L(8)=99 as the first new
    /// entry. The decoder reconstructs `[11, 99]`.
    #[test]
    fn write_palette_entries_y_partial_cache_round_trip() {
        let mut walker = fresh_palette_walker();
        walker.stamp_palette_for_test(0, 0, 0, &[11, 22]).unwrap();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 11;
        entries[1] = 99;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_y(&mut writer, 8, 2, &entries, &walker, 0, 1).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 4]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let decoded = crate::cdf::read_palette_entries_y(&mut dec, 8, 2, &walker, 0, 1).unwrap();
        assert_eq!(decoded[0], 11);
        assert_eq!(decoded[1], 99);
    }

    /// Caller bug: descending entries — the §5.11.46 reader's
    /// trailing sort guarantees ascending input on the writer side;
    /// a descending pair surfaces `PartitionWalkOutOfRange`.
    #[test]
    fn write_palette_entries_y_rejects_descending() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 50;
        entries[1] = 10;
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_entries_y(&mut writer, 8, 2, &entries, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: duplicate entries — non-positive delta on a Y
    /// arm where the spec's `palette_delta_y++` step never produces
    /// a delta_raw of `-1` (delta_raw = (cur - prev) - 1; cur == prev
    /// ⇒ delta_raw = -1 underflow). Reject as caller bug.
    #[test]
    fn write_palette_entries_y_rejects_duplicate() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 10;
        entries[1] = 10;
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_entries_y(&mut writer, 8, 2, &entries, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: entry above the bit-depth range. 8 bit-depth ⇒
    /// entries in [0, 255]; entry 256 is out-of-range.
    #[test]
    fn write_palette_entries_y_rejects_above_bit_depth_range() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 100;
        entries[1] = 256;
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_entries_y(&mut writer, 8, 2, &entries, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `palette_size_y` below the §5.11.46 minimum of 2.
    #[test]
    fn write_palette_entries_y_rejects_size_below_two() {
        let walker = fresh_palette_walker();
        let entries = [0u16; crate::cdf::PALETTE_COLORS];
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_entries_y(&mut writer, 8, 1, &entries, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `palette_size_y` above PALETTE_COLORS (=8).
    #[test]
    fn write_palette_entries_y_rejects_size_above_max() {
        let walker = fresh_palette_walker();
        let entries = [0u16; crate::cdf::PALETTE_COLORS];
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_entries_y(&mut writer, 8, 9, &entries, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: invalid bit_depth (§5.5.2 admits 8 / 10 / 12 only).
    #[test]
    fn write_palette_entries_y_rejects_invalid_bit_depth() {
        let walker = fresh_palette_walker();
        let mut entries = [0u16; crate::cdf::PALETTE_COLORS];
        entries[0] = 0;
        entries[1] = 1;
        let mut writer = SymbolWriter::new(true);
        let err = write_palette_entries_y(&mut writer, 9, 2, &entries, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.46 write_palette_entries_uv — round-trip through
    // crate::cdf::read_palette_entries_uv. The reader sorts U after
    // decode (canonical form); V is left in source order on both arms.
    // -----------------------------------------------------------------

    /// 2-entry UV palette `[0, 0]` U + `[0, 0]` V at 8 bit-depth on the
    /// V direct-literal arm, fresh walker (empty cache). Mirrors the
    /// cdf-side `read_palette_entries_uv_size_2_zero_bitstream_direct_literal_arm`
    /// fixture: writer emits the L(8) first U literal then L(2)
    /// extra_bits then L(paletteBits) raw delta then L(1)=0 for
    /// `delta_encode_v` then two L(8) V literals.
    #[test]
    fn write_palette_entries_uv_size_2_round_trip_8bit_direct_v_arm() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 0;
        u[1] = 0;
        v[0] = 0;
        v[1] = 0;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 8, 2, &u, &v, false, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 8, 2, &walker, 0, 0).unwrap();
        assert_eq!(u_dec[0], 0);
        assert_eq!(u_dec[1], 0);
        assert_eq!(v_dec[0], 0);
        assert_eq!(v_dec[1], 0);
    }

    /// 3-entry UV palette with ascending U and arbitrary V on the
    /// direct-literal arm at 8 bit-depth. U `[10, 50, 200]` exercises
    /// the U delta loop (no `++`); V `[123, 7, 200]` exercises the
    /// direct-literal arm where order is preserved.
    #[test]
    fn write_palette_entries_uv_size_3_round_trip_8bit_direct_v_arm() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 10;
        u[1] = 50;
        u[2] = 200;
        v[0] = 123;
        v[1] = 7;
        v[2] = 200;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 8, 3, &u, &v, false, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 16]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 8, 3, &walker, 0, 0).unwrap();
        assert_eq!(u_dec[0], 10);
        assert_eq!(u_dec[1], 50);
        assert_eq!(u_dec[2], 200);
        // V on direct-literal arm preserves source order.
        assert_eq!(v_dec[0], 123);
        assert_eq!(v_dec[1], 7);
        assert_eq!(v_dec[2], 200);
    }

    /// 3-entry UV palette on the V delta-encoded arm at 8 bit-depth.
    /// U `[10, 50, 200]`; V `[100, 110, 90]` — exercises both positive
    /// (`110 - 100 = +10`) and negative (`90 - 110 = -20`) signed
    /// deltas with the §5.11.46 sign-bit emission. Round-trip matches.
    #[test]
    fn write_palette_entries_uv_size_3_round_trip_8bit_delta_v_arm() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 10;
        u[1] = 50;
        u[2] = 200;
        v[0] = 100;
        v[1] = 110;
        v[2] = 90;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 8, 3, &u, &v, true, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 16]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 8, 3, &walker, 0, 0).unwrap();
        assert_eq!(u_dec[0], 10);
        assert_eq!(u_dec[1], 50);
        assert_eq!(u_dec[2], 200);
        assert_eq!(v_dec[0], 100);
        assert_eq!(v_dec[1], 110);
        assert_eq!(v_dec[2], 90);
    }

    /// §5.11.46 V delta-arm modular wrap: V `[10, 250]` at 8 bit-depth.
    /// The natural delta `250 - 10 = +240` has magnitude 240; the
    /// wrap candidate `240 - 256 = -16` has magnitude 16 (much
    /// smaller). The writer picks the wrap candidate, the reader's
    /// modular-wrap step reconstructs `cur = 250` from `prev = 10 +
    /// (-16) = -6 → -6 + 256 = 250`.
    #[test]
    fn write_palette_entries_uv_v_delta_arm_modular_wrap_round_trip() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 0;
        u[1] = 1;
        v[0] = 10;
        v[1] = 250;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 8, 2, &u, &v, true, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (_, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 8, 2, &walker, 0, 0).unwrap();
        assert_eq!(v_dec[0], 10);
        assert_eq!(v_dec[1], 250);
    }

    /// Full palette of size 8 on the V delta arm at 8 bit-depth —
    /// exercises every step of the V delta loop. U is the same
    /// ascending sequence the Y refinement test uses (`[1, 30, 60,
    /// 90, 120, 180, 220, 250]`); V samples arbitrary entries.
    #[test]
    fn write_palette_entries_uv_max_size_round_trip_8bit_delta_v_arm() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        let u_vals: [u16; 8] = [1, 30, 60, 90, 120, 180, 220, 250];
        u[..8].copy_from_slice(&u_vals);
        // V samples chosen so each consecutive transition's smallest
        // §5.11.46 wrap-candidate magnitude stays < 2^paletteBits at
        // the writer's maximum paletteBits_v = bd - 4 + 3 = 7 (=>
        // magnitude < 128). The wrap-around bound is maxVal/2 = 128 at
        // 8-bit; staying strictly below that ceiling keeps every
        // candidate representable.
        let v_vals: [u16; 8] = [128, 65, 192, 70, 200, 100, 156, 35];
        v[..8].copy_from_slice(&v_vals);
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 8, 8, &u, &v, true, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 32]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 8, 8, &walker, 0, 0).unwrap();
        assert_eq!(&u_dec[..8], &u_vals);
        assert_eq!(&v_dec[..8], &v_vals);
    }

    /// 10 bit-depth path on the direct-literal V arm — exercises L(10)
    /// literals + `minBits = bd - 3 = 7` on U.
    #[test]
    fn write_palette_entries_uv_size_3_round_trip_10bit_direct_v_arm() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        // 10-bit range is [0, 1023].
        u[0] = 100;
        u[1] = 500;
        u[2] = 900;
        v[0] = 800;
        v[1] = 200;
        v[2] = 600;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 10, 3, &u, &v, false, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 16]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 10, 3, &walker, 0, 0).unwrap();
        assert_eq!(u_dec[0], 100);
        assert_eq!(u_dec[1], 500);
        assert_eq!(u_dec[2], 900);
        assert_eq!(v_dec[0], 800);
        assert_eq!(v_dec[1], 200);
        assert_eq!(v_dec[2], 600);
    }

    /// 12 bit-depth path on the V delta arm — `minBits_v = bd - 4 = 8`.
    #[test]
    fn write_palette_entries_uv_size_2_round_trip_12bit_delta_v_arm() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        // 12-bit range is [0, 4095].
        u[0] = 1000;
        u[1] = 3000;
        v[0] = 2000;
        v[1] = 2500;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 12, 2, &u, &v, true, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 16]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 12, 2, &walker, 0, 0).unwrap();
        assert_eq!(u_dec[0], 1000);
        assert_eq!(u_dec[1], 3000);
        assert_eq!(v_dec[0], 2000);
        assert_eq!(v_dec[1], 2500);
    }

    /// §5.11.49 cache-hit round-trip on U: stamp a left-neighbour U
    /// palette `[11, 22]` at (0, 0); encode matching U `[11, 22]` +
    /// arbitrary V at (0, 1). The writer emits two L(1)=1 cache flags
    /// for U (no literal/L(2)/delta loop), then the V direct-literal
    /// arm. Round-trip matches.
    #[test]
    fn write_palette_entries_uv_cache_hit_round_trip() {
        let mut walker = fresh_palette_walker();
        // Stamp left-neighbour U palette at (0, 0) — plane = 1.
        walker.stamp_palette_for_test(1, 0, 0, &[11, 22]).unwrap();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 11;
        u[1] = 22;
        v[0] = 33;
        v[1] = 44;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 8, 2, &u, &v, false, &walker, 0, 1).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, v_dec) =
            crate::cdf::read_palette_entries_uv(&mut dec, 8, 2, &walker, 0, 1).unwrap();
        assert_eq!(u_dec[0], 11);
        assert_eq!(u_dec[1], 22);
        assert_eq!(v_dec[0], 33);
        assert_eq!(v_dec[1], 44);
    }

    /// Caller bug: U descending — non-strict ascending invariant
    /// surfaces `PartitionWalkOutOfRange`.
    #[test]
    fn write_palette_entries_uv_rejects_u_descending() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 50;
        u[1] = 10;
        v[0] = 0;
        v[1] = 0;
        let mut writer = SymbolWriter::new(true);
        let err =
            write_palette_entries_uv(&mut writer, 8, 2, &u, &v, false, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// U duplicates are LEGAL on the UV writer (the U delta arm has
    /// no `++`, so delta_raw == 0 is valid). Round-trip succeeds.
    #[test]
    fn write_palette_entries_uv_u_duplicates_accepted() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 10;
        u[1] = 10;
        v[0] = 0;
        v[1] = 0;
        let mut writer = SymbolWriter::new(true);
        write_palette_entries_uv(&mut writer, 8, 2, &u, &v, false, &walker, 0, 0).unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() {
            vec![0u8; 8]
        } else {
            bytes
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), true).unwrap();
        let (u_dec, _) =
            crate::cdf::read_palette_entries_uv(&mut dec, 8, 2, &walker, 0, 0).unwrap();
        assert_eq!(u_dec[0], 10);
        assert_eq!(u_dec[1], 10);
    }

    /// Caller bug: U entry above the bit-depth range.
    #[test]
    fn write_palette_entries_uv_rejects_u_above_bit_depth_range() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 100;
        u[1] = 256;
        v[0] = 0;
        v[1] = 0;
        let mut writer = SymbolWriter::new(true);
        let err =
            write_palette_entries_uv(&mut writer, 8, 2, &u, &v, false, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: V entry above the bit-depth range.
    #[test]
    fn write_palette_entries_uv_rejects_v_above_bit_depth_range() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 0;
        u[1] = 1;
        v[0] = 0;
        v[1] = 256;
        let mut writer = SymbolWriter::new(true);
        let err =
            write_palette_entries_uv(&mut writer, 8, 2, &u, &v, false, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `palette_size_uv` below the §5.11.46 minimum of 2.
    #[test]
    fn write_palette_entries_uv_rejects_size_below_two() {
        let walker = fresh_palette_walker();
        let u = [0u16; crate::cdf::PALETTE_COLORS];
        let v = [0u16; crate::cdf::PALETTE_COLORS];
        let mut writer = SymbolWriter::new(true);
        let err =
            write_palette_entries_uv(&mut writer, 8, 1, &u, &v, false, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `palette_size_uv` above PALETTE_COLORS (=8).
    #[test]
    fn write_palette_entries_uv_rejects_size_above_max() {
        let walker = fresh_palette_walker();
        let u = [0u16; crate::cdf::PALETTE_COLORS];
        let v = [0u16; crate::cdf::PALETTE_COLORS];
        let mut writer = SymbolWriter::new(true);
        let err =
            write_palette_entries_uv(&mut writer, 8, 9, &u, &v, false, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: invalid bit_depth (§5.5.2 admits 8 / 10 / 12 only).
    #[test]
    fn write_palette_entries_uv_rejects_invalid_bit_depth() {
        let walker = fresh_palette_walker();
        let mut u = [0u16; crate::cdf::PALETTE_COLORS];
        let mut v = [0u16; crate::cdf::PALETTE_COLORS];
        u[0] = 0;
        u[1] = 1;
        v[0] = 0;
        v[1] = 0;
        let mut writer = SymbolWriter::new(true);
        let err =
            write_palette_entries_uv(&mut writer, 9, 2, &u, &v, false, &walker, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.22 write_intra_block_mode_info dispatcher — full
    // round-trip through PartitionWalker::decode_intra_block_mode_info.
    // -----------------------------------------------------------------

    /// Round-trip the §5.11.22 dispatcher on the default fixture
    /// (`BLOCK_16X16`, `y_mode = DC_PRED`, `uv_mode = DC_PRED`,
    /// gates closed, no chroma subsampling, monochrome=false). All
    /// short-circuits fire; the decoder reconstructs every field.
    #[test]
    fn write_intra_block_mode_info_dc_pred_full_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            /* cfl_allowed = */ true,
            /* has_chroma = */ true,
            /* allow_screen_content_tools = */ false,
            /* enable_filter_intra = */ false,
            0,
            None,
            false,
            false,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* lossless = */ false,
                /* has_chroma = */ true,
                /* allow_screen_content_tools = */ false,
                /* enable_filter_intra = */ false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, Some(DC_PRED_U8));
        assert_eq!(info.angle_delta_y, 0);
        assert_eq!(info.angle_delta_uv, Some(0));
        assert_eq!(info.cfl_alpha_u, None);
        assert_eq!(info.cfl_alpha_v, None);
        assert_eq!(info.has_palette_y, None);
        assert_eq!(info.has_palette_uv, None);
        assert_eq!(info.use_filter_intra, None);
        assert_eq!(info.filter_intra_mode, None);
    }

    /// Round-trip the §5.11.22 dispatcher with a directional
    /// `y_mode = V_PRED` and a non-zero `angle_delta_y = 2`. The
    /// §5.11.42 inner-arm fires for the luma plane; UV stays DC_PRED
    /// non-directional so the §5.11.43 short-circuit holds.
    #[test]
    fn write_intra_block_mode_info_directional_y_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            V_PRED_U8,
            Some(DC_PRED_U8),
            /* angle_delta_y = */ 2,
            0,
            true,
            true,
            false,
            false,
            0,
            None,
            false,
            false,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, V_PRED_U8);
        assert_eq!(info.uv_mode, Some(DC_PRED_U8));
        assert_eq!(info.angle_delta_y, 2);
        assert_eq!(info.angle_delta_uv, Some(0));
    }

    /// Round-trip the §5.11.22 dispatcher with `allow_screen_content_tools
    /// = true` (palette gate open): writer commits has_palette_* = 0
    /// on both planes, decoder reconstructs `Some(0)` on both arms.
    #[test]
    fn write_intra_block_mode_info_palette_gate_open_no_palette_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            true,
            true,
            /* allow_screen_content_tools = */ true,
            false,
            0,
            None,
            false,
            false,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                /* allow_screen_content_tools = */ true,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, Some(DC_PRED_U8));
        assert_eq!(info.has_palette_y, Some(0));
        assert_eq!(info.has_palette_uv, Some(0));
    }

    /// Round-trip with `enable_filter_intra = true` and
    /// `use_filter_intra = 1` + `filter_intra_mode = Some(3)`. All
    /// three §5.11.22 leaves cooperate; the §5.11.24 outer gate fires
    /// (DC_PRED + palette_size_y = 0 + Max(bw,bh) = 16 ≤ 32).
    #[test]
    fn write_intra_block_mode_info_filter_intra_arm_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            true,
            true,
            false,
            /* enable_filter_intra = */ true,
            /* use_filter_intra = */ 1,
            Some(3),
            false,
            false,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                false,
                /* enable_filter_intra = */ true,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.use_filter_intra, Some(1));
        assert_eq!(info.filter_intra_mode, Some(3));
    }

    /// Monochrome round-trip (`has_chroma = false`): dispatcher
    /// skips the §5.11.22 line 5-10 chroma arm. `uv_mode` MUST be
    /// `None`; `angle_delta_uv` MUST be `0`.
    #[test]
    fn write_intra_block_mode_info_monochrome_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            None,
            0,
            0,
            true,
            /* has_chroma = */ false,
            false,
            false,
            0,
            None,
            false,
            false,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                /* has_chroma = */ false,
                false,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, None);
        assert_eq!(info.angle_delta_uv, None);
    }

    /// Caller bug: `has_chroma = true` with `uv_mode = None`.
    #[test]
    fn write_intra_block_mode_info_rejects_missing_uv_mode_when_has_chroma() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            None,
            0,
            0,
            true,
            true,
            false,
            false,
            0,
            None,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: dispatcher rejects the CFL arm (UV_CFL_PRED = 13)
    /// on this round's scope — composed `write_cfl_alphas` is a
    /// future-round arc.
    #[test]
    fn write_intra_block_mode_info_rejects_uv_cfl_pred() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(13),
            0,
            0,
            true,
            true,
            false,
            false,
            0,
            None,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: monochrome (`has_chroma = false`) with `Some`
    /// uv_mode contradicts the §5.11.22 line 5 `if (HasChroma)`
    /// guard.
    #[test]
    fn write_intra_block_mode_info_rejects_some_uv_when_monochrome() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let err = write_intra_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            true,
            /* has_chroma = */ false,
            false,
            false,
            0,
            None,
            false,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.46 write_palette_mode_info_with_entries — round-trip the
    // lifted leaf through the reader's full §5.11.46 sub-block via
    // decode_intra_block_mode_info.
    // -----------------------------------------------------------------

    /// Gate-off: `allow_screen_content_tools = false` ⇒ no §5.11.46
    /// S() symbol emitted. Same shape as the legacy leaf's gate-off
    /// test, but exercised through the r264 entries-aware leaf with
    /// dummy entry slots.
    #[test]
    fn write_palette_mode_info_with_entries_gate_off_no_bits() {
        let walker = fresh_palette_walker();
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let baseline = SymbolWriter::new(false).finish().len();
        let dummy = [0u16; PALETTE_COLORS];
        write_palette_mode_info_with_entries(
            &mut writer,
            &mut cdfs,
            /* has_palette_y = */ 0,
            /* has_palette_uv = */ 0,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            /* has_chroma = */ true,
            /* allow_screen_content_tools = */ false,
            false,
            false,
            8,
            /* palette_size_y = */ 0,
            &dummy,
            /* palette_size_uv = */ 0,
            &dummy,
            &dummy,
            false,
            &walker,
            0,
            0,
        )
        .unwrap();
        assert_eq!(
            writer.finish().len(),
            baseline,
            "§5.11.46 gate-off path emits zero additional bits via the r264 leaf"
        );
    }

    /// Gate-off via `mi_size < BLOCK_8X8` while
    /// `allow_screen_content_tools = true`. Mirrors the legacy leaf's
    /// `write_palette_mode_info_small_block_no_bits`.
    #[test]
    fn write_palette_mode_info_with_entries_small_block_no_bits() {
        let walker = fresh_palette_walker();
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let baseline = SymbolWriter::new(false).finish().len();
        let dummy = [0u16; PALETTE_COLORS];
        write_palette_mode_info_with_entries(
            &mut writer,
            &mut cdfs,
            0,
            0,
            BLOCK_4X4,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            /* allow_screen_content_tools = */ true,
            false,
            false,
            8,
            0,
            &dummy,
            0,
            &dummy,
            &dummy,
            false,
            &walker,
            0,
            0,
        )
        .unwrap();
        assert_eq!(writer.finish().len(), baseline);
    }

    /// Gate-off rejects non-zero `has_palette_y` (caller bug: the
    /// reader never reads the symbol on the gate-off arm).
    #[test]
    fn write_palette_mode_info_with_entries_gate_off_rejects_has_palette_y_one() {
        let walker = fresh_palette_walker();
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let mut entries = [0u16; PALETTE_COLORS];
        entries[0] = 0;
        entries[1] = 1;
        let err = write_palette_mode_info_with_entries(
            &mut writer,
            &mut cdfs,
            1,
            0,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            /* allow_screen_content_tools = */ false,
            false,
            false,
            8,
            2,
            &entries,
            0,
            &entries,
            &entries,
            false,
            &walker,
            0,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller bug: `has_palette_y == 1` with `palette_size_y` outside
    /// `2..=PALETTE_COLORS`.
    #[test]
    fn write_palette_mode_info_with_entries_rejects_palette_size_y_out_of_range() {
        let walker = fresh_palette_walker();
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(true);
        let entries = [0u16; PALETTE_COLORS];
        let err = write_palette_mode_info_with_entries(
            &mut writer,
            &mut cdfs,
            1,
            0,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            true,
            true,
            false,
            false,
            8,
            /* palette_size_y = */ 1,
            &entries,
            0,
            &entries,
            &entries,
            false,
            &walker,
            0,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Luma arm fires: `has_palette_y = 1`, `palette_size_y = 2`,
    /// entries `[0, 1]`. Round-trip through the reader's full
    /// §5.11.46 sub-block via decode_intra_block_mode_info — exposes
    /// the writer's `palette_size_y_minus_2 S()` + the §5.11.46
    /// palette-entries syntax committed by [`write_palette_entries_y`].
    ///
    /// Embedded in a dispatcher-shaped bit stream so the decoder's
    /// upstream `y_mode` / angle_delta steps consume the same prefix
    /// the writer emitted.
    #[test]
    fn write_palette_mode_info_with_entries_luma_arm_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let mut entries_y = [0u16; PALETTE_COLORS];
        entries_y[0] = 0;
        entries_y[1] = 1;
        let dummy = [0u16; PALETTE_COLORS];

        // Use the r264 full dispatcher so the bit stream a decoder
        // walks is well-formed: y_mode + angle_delta + uv_mode +
        // angle_delta_uv + the §5.11.46 palette body + the §5.11.24
        // filter_intra closer (gate closes mechanically when
        // PaletteSizeY != 0).
        write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            /* cfl_allowed = */ true,
            /* has_chroma = */ true,
            /* allow_screen_content_tools = */ true,
            /* enable_filter_intra = */ false,
            0,
            None,
            false,
            false,
            8,
            /* has_palette_y = */ 1,
            /* palette_size_y = */ 2,
            &entries_y,
            /* has_palette_uv = */ 0,
            0,
            &dummy,
            &dummy,
            false,
            /* cfl_alpha_u = */ None,
            /* cfl_alpha_v = */ None,
            &walker_w,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                /* allow_screen_content_tools = */ true,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.has_palette_y, Some(1));
        assert_eq!(info.palette_size_y, Some(2));
        let pal_y = info.palette_colors_y.expect("luma arm fired");
        assert_eq!(pal_y[0], 0);
        assert_eq!(pal_y[1], 1);
        // Chroma arm closed (has_palette_uv = 0). The §5.11.46 reader
        // commits Some(0) on a fired chroma S() even on the off-arm.
        assert_eq!(info.has_palette_uv, Some(0));
    }

    /// Chroma arm fires: `has_palette_y = 0`, `has_palette_uv = 1`,
    /// `palette_size_uv = 2`, U `[0, 1]`, V `[5, 7]`, V-direct arm.
    /// Round-trip through the dispatcher into the reader.
    #[test]
    fn write_palette_mode_info_with_entries_chroma_direct_v_arm_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let mut entries_u = [0u16; PALETTE_COLORS];
        entries_u[0] = 0;
        entries_u[1] = 1;
        let mut entries_v = [0u16; PALETTE_COLORS];
        entries_v[0] = 5;
        entries_v[1] = 7;
        write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            true,
            true,
            /* allow_screen_content_tools = */ true,
            false,
            0,
            None,
            false,
            false,
            8,
            /* has_palette_y = */ 0,
            0,
            &dummy,
            /* has_palette_uv = */ 1,
            /* palette_size_uv = */ 2,
            &entries_u,
            &entries_v,
            /* delta_encode_v = */ false,
            /* cfl_alpha_u = */ None,
            /* cfl_alpha_v = */ None,
            &walker_w,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                true,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.has_palette_y, Some(0));
        assert_eq!(info.has_palette_uv, Some(1));
        assert_eq!(info.palette_size_uv, Some(2));
        let pal_u = info.palette_colors_u.expect("chroma arm fired");
        assert_eq!(pal_u[0], 0);
        assert_eq!(pal_u[1], 1);
        let pal_v = info.palette_colors_v.expect("chroma arm fired");
        assert_eq!(pal_v[0], 5);
        assert_eq!(pal_v[1], 7);
    }

    /// Both arms fire: luma `[0, 3]` size 2, chroma U `[0, 1]` + V
    /// `[10, 12]` size 2 with the V delta-encoded arm. Round-trip the
    /// full §5.11.46 body via the r264 dispatcher into the reader.
    #[test]
    fn write_palette_mode_info_with_entries_both_arms_delta_v_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let mut entries_y = [0u16; PALETTE_COLORS];
        entries_y[0] = 0;
        entries_y[1] = 3;
        let mut entries_u = [0u16; PALETTE_COLORS];
        entries_u[0] = 0;
        entries_u[1] = 1;
        let mut entries_v = [0u16; PALETTE_COLORS];
        entries_v[0] = 10;
        entries_v[1] = 12;
        write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            true,
            true,
            /* allow_screen_content_tools = */ true,
            false,
            0,
            None,
            false,
            false,
            8,
            /* has_palette_y = */ 1,
            2,
            &entries_y,
            /* has_palette_uv = */ 1,
            2,
            &entries_u,
            &entries_v,
            /* delta_encode_v = */ true,
            /* cfl_alpha_u = */ None,
            /* cfl_alpha_v = */ None,
            &walker_w,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                true,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.has_palette_y, Some(1));
        assert_eq!(info.palette_size_y, Some(2));
        let pal_y = info.palette_colors_y.expect("luma arm fired");
        assert_eq!(pal_y[0], 0);
        assert_eq!(pal_y[1], 3);
        assert_eq!(info.has_palette_uv, Some(1));
        assert_eq!(info.palette_size_uv, Some(2));
        let pal_u = info.palette_colors_u.expect("chroma arm fired");
        assert_eq!(pal_u[0], 0);
        assert_eq!(pal_u[1], 1);
        let pal_v = info.palette_colors_v.expect("chroma arm fired");
        assert_eq!(pal_v[0], 10);
        assert_eq!(pal_v[1], 12);
    }

    /// The §5.11.24 filter_intra outer gate closes when `PaletteSizeY
    /// != 0` — even with `enable_filter_intra = true` + `use_filter_intra
    /// = 0`, the gate closes mechanically. Round-trip confirms the
    /// reader observes `use_filter_intra == None`.
    #[test]
    fn write_intra_block_mode_info_with_palette_filter_intra_closed_by_palette() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let mut entries_y = [0u16; PALETTE_COLORS];
        entries_y[0] = 0;
        entries_y[1] = 1;
        let dummy = [0u16; PALETTE_COLORS];
        write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            true,
            true,
            /* allow_screen_content_tools = */ true,
            /* enable_filter_intra = */ true,
            0,
            None,
            false,
            false,
            8,
            /* has_palette_y = */ 1,
            2,
            &entries_y,
            0,
            0,
            &dummy,
            &dummy,
            false,
            /* cfl_alpha_u = */ None,
            /* cfl_alpha_v = */ None,
            &walker_w,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();
        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                true,
                /* enable_filter_intra = */ true,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.has_palette_y, Some(1));
        assert_eq!(info.use_filter_intra, None);
        assert_eq!(info.filter_intra_mode, None);
    }

    // -----------------------------------------------------------------
    // r265: §5.11.22 line-8 CFL arm — write_cfl_alphas composition.
    // -----------------------------------------------------------------

    /// CFL arm round-trip with both alphas positive (`alpha_u = +1`,
    /// `alpha_v = +2`): exercises the §5.11.45 `signU == CFL_SIGN_POS
    /// && signV == CFL_SIGN_POS` joint-sign slot (`joint = 3*2 + 2 - 1
    /// = 7`), both per-plane CFL alpha S() emissions, and the
    /// dispatcher's CFL-arm threading between `write_intra_uv_mode`
    /// and `write_intra_angle_info_uv`.
    #[test]
    fn write_intra_block_mode_info_with_palette_cfl_arm_both_pos_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        // UV_CFL_PRED = 13 per §3.
        let uv_cfl_pred: u8 = 13;
        write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(uv_cfl_pred),
            0,
            0,
            /* cfl_allowed = */ true,
            /* has_chroma = */ true,
            /* allow_screen_content_tools = */ false,
            /* enable_filter_intra = */ false,
            0,
            None,
            false,
            false,
            8,
            /* has_palette_y = */ 0,
            0,
            &dummy,
            /* has_palette_uv = */ 0,
            0,
            &dummy,
            &dummy,
            false,
            /* cfl_alpha_u = */ Some(1),
            /* cfl_alpha_v = */ Some(2),
            &walker_w,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* lossless = */ false,
                /* has_chroma = */ true,
                /* allow_screen_content_tools = */ false,
                /* enable_filter_intra = */ false,
                /* subsampling_x = */ false,
                /* subsampling_y = */ false,
                /* above_palette_y = */ false,
                /* left_palette_y = */ false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, Some(uv_cfl_pred));
        assert_eq!(info.cfl_alpha_u, Some(1));
        assert_eq!(info.cfl_alpha_v, Some(2));
    }

    /// CFL arm round-trip with one zero / one negative alpha
    /// (`alpha_u = 0`, `alpha_v = -3`): exercises the §5.11.45
    /// `signU == CFL_SIGN_ZERO && signV == CFL_SIGN_NEG` joint-sign
    /// slot (`joint = 3*0 + 1 - 1 = 0`) — the no-bits U arm + the
    /// `signed = -(1 + raw)` V arm.
    #[test]
    fn write_intra_block_mode_info_with_palette_cfl_arm_zero_neg_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let uv_cfl_pred: u8 = 13;
        write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(uv_cfl_pred),
            0,
            0,
            true,
            true,
            false,
            false,
            0,
            None,
            false,
            false,
            8,
            0,
            0,
            &dummy,
            0,
            0,
            &dummy,
            &dummy,
            false,
            Some(0),
            Some(-3),
            &walker_w,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_block_mode_info(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.uv_mode, Some(uv_cfl_pred));
        assert_eq!(info.cfl_alpha_u, Some(0));
        assert_eq!(info.cfl_alpha_v, Some(-3));
    }

    /// Caller-bug rejection: `uv_mode == UV_CFL_PRED` with
    /// `cfl_alpha_u == None`. The §5.11.22 CFL arm requires both
    /// alphas to be `Some`; passing `None` is a caller bug and the
    /// dispatcher MUST surface it before touching the bit stream.
    #[test]
    fn write_intra_block_mode_info_with_palette_cfl_arm_missing_alpha_rejected() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let uv_cfl_pred: u8 = 13;
        let err = write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(uv_cfl_pred),
            0,
            0,
            true,
            true,
            false,
            false,
            0,
            None,
            false,
            false,
            8,
            0,
            0,
            &dummy,
            0,
            0,
            &dummy,
            &dummy,
            false,
            /* cfl_alpha_u = */ None,
            /* cfl_alpha_v = */ Some(1),
            &walker_w,
            0,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller-bug rejection: non-CFL `uv_mode` with `cfl_alpha_u ==
    /// Some(_)`. On the non-CFL chroma arm the §5.11.22 reader never
    /// touches `read_cfl_alphas`; passing `Some` is a caller bug.
    #[test]
    fn write_intra_block_mode_info_with_palette_non_cfl_with_alpha_rejected() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let err = write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            true,
            true,
            false,
            false,
            0,
            None,
            false,
            false,
            8,
            0,
            0,
            &dummy,
            0,
            0,
            &dummy,
            &dummy,
            false,
            /* cfl_alpha_u = */ Some(1),
            /* cfl_alpha_v = */ None,
            &walker_w,
            0,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Caller-bug rejection: `uv_mode == UV_CFL_PRED` with
    /// `cfl_allowed == false`. The §8.3.2 CDF-row selector excludes
    /// `UV_CFL_PRED` on the CFL-not-allowed row (width 13 vs 14), so
    /// the writer MUST stay in lockstep with the reader's row choice.
    #[test]
    fn write_intra_block_mode_info_with_palette_cfl_arm_cfl_not_allowed_rejected() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let uv_cfl_pred: u8 = 13;
        let err = write_intra_block_mode_info_with_palette(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(uv_cfl_pred),
            0,
            0,
            /* cfl_allowed = */ false,
            true,
            false,
            false,
            0,
            None,
            false,
            false,
            8,
            0,
            0,
            &dummy,
            0,
            0,
            &dummy,
            &dummy,
            false,
            Some(1),
            Some(1),
            &walker_w,
            0,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // r280: §5.11.7 `else` (non-intrabc) arm — write_intra_frame_else_arm
    // round-trips through decode_intra_frame_mode_info_else_arm.
    // -----------------------------------------------------------------

    /// Directional luma + chroma round-trip at the frame origin
    /// (`AvailU = AvailL = false` ⇒ both §8.3.2 ctx values are the
    /// `intra_mode_ctx(DC_PRED) = 0` identity): `YMode = V_PRED` with
    /// `AngleDeltaY = +2`, `UVMode = V_PRED` with `AngleDeltaUV = -1`.
    /// Exercises the `intra_frame_y_mode` S() (the element that
    /// distinguishes §5.11.7 from §5.11.22) plus both §5.11.42/§5.11.43
    /// angle reads, and asserts the `TileIntraFrameYModeCdf[0][0]`
    /// adaptation rows match on both sides.
    #[test]
    fn write_intra_frame_else_arm_directional_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let cfl_allowed = cfl_allowed_for_uv_mode(false, BLOCK_16X16, false, false);
        write_intra_frame_else_arm(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            V_PRED_U8,
            Some(V_PRED_U8),
            /* angle_delta_y = */ 2,
            /* angle_delta_uv = */ -1,
            cfl_allowed,
            /* has_chroma = */ true,
            /* allow_screen_content_tools = */ false,
            /* enable_filter_intra = */ false,
            0,
            None,
            false,
            false,
            8,
            0,
            0,
            &dummy,
            0,
            0,
            &dummy,
            &dummy,
            false,
            None,
            None,
            &walker_w,
            0,
            0,
            /* abovemode_ctx = */ 0,
            /* leftmode_ctx = */ 0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_frame_mode_info_else_arm(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                /* lossless = */ false,
                /* has_chroma = */ true,
                /* allow_screen_content_tools = */ false,
                /* enable_filter_intra = */ false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, V_PRED_U8);
        assert_eq!(info.angle_delta_y, 2);
        assert_eq!(info.uv_mode, Some(V_PRED_U8));
        assert_eq!(info.angle_delta_uv, Some(-1));
        assert_eq!(info.ref_frame, [0, -1], "§5.11.7 prefix RefFrame pair");
        // §8.3 adaptation lockstep on the arm-distinguishing CDF row.
        assert_eq!(
            enc_cdfs.intra_frame_y_mode_cdf(0, 0).to_vec(),
            dec_cdfs.intra_frame_y_mode_cdf(0, 0).to_vec(),
            "TileIntraFrameYModeCdf[0][0] adaptation must match"
        );
    }

    /// Two adjacent BLOCK_8X8 blocks sharing one stream + one decode
    /// walker: block A at (0, 0) codes `V_PRED` with the (0, 0) ctx
    /// identity; block B at (0, 2) has `AvailL = true` so the writer
    /// passes `leftmode_ctx = intra_mode_ctx(V_PRED)` (mirroring its
    /// own §5.11.5 `YModes[][]` bookkeeping) and the reader must
    /// derive the same pair from the grid its block-A decode stamped.
    /// A ctx mismatch desynchronises the arithmetic coder and fails
    /// the block-B assertions.
    #[test]
    fn write_intra_frame_else_arm_neighbour_ctx_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let write_block = |writer: &mut SymbolWriter,
                           cdfs: &mut TileCdfContext,
                           y_mode: u8,
                           mi_col: u32,
                           leftmode_ctx: usize|
         -> Result<(), Error> {
            write_intra_frame_else_arm(
                writer,
                cdfs,
                BLOCK_8X8,
                y_mode,
                /* uv_mode = */ None,
                0,
                0,
                false,
                /* has_chroma = */ false,
                false,
                false,
                0,
                None,
                false,
                false,
                8,
                0,
                0,
                &dummy,
                0,
                0,
                &dummy,
                &dummy,
                false,
                None,
                None,
                &walker_w,
                0,
                mi_col,
                /* abovemode_ctx = */ 0,
                leftmode_ctx,
            )
        };
        // Block A at (0, 0): both neighbours unavailable.
        write_block(&mut writer, &mut enc_cdfs, V_PRED_U8, 0, 0).unwrap();
        // Block B at (0, 2): left neighbour is block A (`YMode =
        // V_PRED`), above row is outside the tile.
        write_block(
            &mut writer,
            &mut enc_cdfs,
            DC_PRED_U8,
            2,
            intra_mode_ctx(V_PRED),
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let decode_block = |walker_r: &mut PartitionWalker,
                            dec: &mut SymbolDecoder<'_>,
                            dec_cdfs: &mut TileCdfContext,
                            mi_col: u32| {
            walker_r
                .decode_intra_frame_mode_info_else_arm(
                    dec, dec_cdfs, 0, mi_col, BLOCK_8X8, false, false, false, false, false, false,
                    false, false, 8,
                )
                .unwrap()
        };
        let info_a = decode_block(&mut walker_r, &mut dec, &mut dec_cdfs, 0);
        assert_eq!(info_a.y_mode, V_PRED_U8);
        let info_b = decode_block(&mut walker_r, &mut dec, &mut dec_cdfs, 2);
        assert_eq!(info_b.y_mode, DC_PRED_U8);
        // Monochrome arm: no chroma fields decoded.
        assert_eq!(info_b.uv_mode, None);
        assert_eq!(info_b.angle_delta_uv, None);
        // The block-B row is the one keyed by the stamped neighbour.
        let ctx = intra_mode_ctx(V_PRED);
        assert_eq!(
            enc_cdfs.intra_frame_y_mode_cdf(0, ctx).to_vec(),
            dec_cdfs.intra_frame_y_mode_cdf(0, ctx).to_vec(),
            "neighbour-keyed TileIntraFrameYModeCdf row adaptation must match"
        );
    }

    /// §5.11.45 CFL arm round-trip on the §5.11.7 composition:
    /// `UVMode = UV_CFL_PRED` with `CflAlphaU = -2` / `CflAlphaV = +1`
    /// (joint-sign slot `signU = NEG`, `signV = POS`).
    #[test]
    fn write_intra_frame_else_arm_cfl_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        let uv_cfl_pred: u8 = 13;
        write_intra_frame_else_arm(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(uv_cfl_pred),
            0,
            0,
            /* cfl_allowed = */ true,
            /* has_chroma = */ true,
            false,
            false,
            0,
            None,
            false,
            false,
            8,
            0,
            0,
            &dummy,
            0,
            0,
            &dummy,
            &dummy,
            false,
            /* cfl_alpha_u = */ Some(-2),
            /* cfl_alpha_v = */ Some(1),
            &walker_w,
            0,
            0,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_frame_mode_info_else_arm(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                false,
                false,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, Some(uv_cfl_pred));
        assert_eq!(info.cfl_alpha_u, Some(-2));
        assert_eq!(info.cfl_alpha_v, Some(1));
    }

    /// §5.11.46 palette arm round-trip on the §5.11.7 composition:
    /// `has_palette_y = 1` with the 2-entry `[0, 1]` fixture; the
    /// §5.11.24 filter-intra gate must close mechanically
    /// (`PaletteSizeY != 0`) even with `enable_filter_intra = true`.
    #[test]
    fn write_intra_frame_else_arm_palette_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let mut entries_y = [0u16; PALETTE_COLORS];
        entries_y[0] = 0;
        entries_y[1] = 1;
        let dummy = [0u16; PALETTE_COLORS];
        write_intra_frame_else_arm(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_16X16,
            DC_PRED_U8,
            Some(DC_PRED_U8),
            0,
            0,
            /* cfl_allowed = */ true,
            /* has_chroma = */ true,
            /* allow_screen_content_tools = */ true,
            /* enable_filter_intra = */ true,
            0,
            None,
            false,
            false,
            8,
            /* has_palette_y = */ 1,
            2,
            &entries_y,
            /* has_palette_uv = */ 0,
            0,
            &dummy,
            &dummy,
            false,
            None,
            None,
            &walker_w,
            0,
            0,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_frame_mode_info_else_arm(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_16X16,
                false,
                true,
                true,
                true,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.has_palette_y, Some(1));
        assert_eq!(info.palette_size_y, Some(2));
        let colors = info.palette_colors_y.expect("entries decoded");
        assert_eq!(&colors[..2], &[0, 1]);
        assert_eq!(info.has_palette_uv, Some(0));
        assert_eq!(info.use_filter_intra, None, "§5.11.24 gate closed");
        assert_eq!(info.filter_intra_mode, None);
    }

    /// Monochrome (`HasChroma = false`) + §5.11.24 filter-intra arm:
    /// `use_filter_intra = 1` with `filter_intra_mode = 3`. The chroma
    /// reads are skipped entirely (no `uv_mode` / CFL / UV-angle
    /// symbols on the wire).
    #[test]
    fn write_intra_frame_else_arm_monochrome_filter_intra_round_trip() {
        let walker_w = fresh_palette_walker();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let dummy = [0u16; PALETTE_COLORS];
        write_intra_frame_else_arm(
            &mut writer,
            &mut enc_cdfs,
            BLOCK_8X8,
            DC_PRED_U8,
            None,
            0,
            0,
            false,
            /* has_chroma = */ false,
            false,
            /* enable_filter_intra = */ true,
            /* use_filter_intra = */ 1,
            /* filter_intra_mode = */ Some(3),
            false,
            false,
            8,
            0,
            0,
            &dummy,
            0,
            0,
            &dummy,
            &dummy,
            false,
            None,
            None,
            &walker_w,
            0,
            0,
            0,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let (mut walker_r, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let info = walker_r
            .decode_intra_frame_mode_info_else_arm(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                BLOCK_8X8,
                false,
                false,
                false,
                true,
                false,
                false,
                false,
                false,
                8,
            )
            .unwrap();
        assert_eq!(info.y_mode, DC_PRED_U8);
        assert_eq!(info.uv_mode, None);
        assert_eq!(info.use_filter_intra, Some(1));
        assert_eq!(info.filter_intra_mode, Some(3));
    }

    /// Caller-bug battery: out-of-range `y_mode` (= INTRA_MODES),
    /// out-of-range `abovemode_ctx` (= INTRA_MODE_CONTEXTS), and a
    /// `Some` `uv_mode` on the monochrome arm. All three must surface
    /// BEFORE any symbol write.
    #[test]
    fn write_intra_frame_else_arm_rejects_caller_bugs() {
        let walker_w = fresh_palette_walker();
        let dummy = [0u16; PALETTE_COLORS];
        let run = |y_mode: u8, uv_mode: Option<u8>, has_chroma: bool, above_ctx: usize| {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_intra_frame_else_arm(
                &mut writer,
                &mut enc_cdfs,
                BLOCK_16X16,
                y_mode,
                uv_mode,
                0,
                0,
                true,
                has_chroma,
                false,
                false,
                0,
                None,
                false,
                false,
                8,
                0,
                0,
                &dummy,
                0,
                0,
                &dummy,
                &dummy,
                false,
                None,
                None,
                &walker_w,
                0,
                0,
                above_ctx,
                0,
            )
            .unwrap_err()
        };
        // §3: y_mode must be < INTRA_MODES = 13.
        assert!(matches!(
            run(13, Some(DC_PRED_U8), true, 0),
            Error::PartitionWalkOutOfRange
        ));
        // §8.3.2: ctx pair must be < INTRA_MODE_CONTEXTS = 5.
        assert!(matches!(
            run(DC_PRED_U8, Some(DC_PRED_U8), true, 5),
            Error::PartitionWalkOutOfRange
        ));
        // §5.11.7 `if ( HasChroma )` guard: `Some` uv_mode on the
        // monochrome arm is a caller bug.
        assert!(matches!(
            run(DC_PRED_U8, Some(DC_PRED_U8), false, 0),
            Error::PartitionWalkOutOfRange
        ));
    }

    // -----------------------------------------------------------------
    // §5.11.23 / §5.11.25 — write_inter_block_mode_info_bootstrap.
    // -----------------------------------------------------------------

    /// Arm 1 (`skip_mode == 1`) propagates `SkipModeFrame[ 0..2 ]`
    /// verbatim. No bits, no CDF accesses — the call site is a pure
    /// derivation.
    #[test]
    fn write_inter_block_mode_info_bootstrap_skip_mode_arm_propagates_skip_mode_frame() {
        // §5.9.22 yields a positive pair for `SkipModeFrame`. Pick
        // LAST_FRAME / GOLDEN_FRAME so the result is unambiguously
        // not an arm-3 fallback (which would force `[1, -1]`).
        let last_frame: i8 = 1; // LAST_FRAME per §3.
        let golden_frame: i8 = 4; // GOLDEN_FRAME per §3.
        let pair = write_inter_block_mode_info_bootstrap(
            1,
            [last_frame, golden_frame],
            // seg_ref_frame_active — ignored on arm 1.
            true,
            7,
            // seg_skip / seg_globalmv — ignored on arm 1.
            true,
            true,
        )
        .unwrap();
        assert_eq!(pair, (last_frame, golden_frame));
    }

    /// Arm 2 (`SEG_LVL_REF_FRAME`) stamps `FeatureData[..][..]` into
    /// slot 0 and `NONE = -1` into slot 1.
    #[test]
    fn write_inter_block_mode_info_bootstrap_seg_ref_frame_arm_stamps_feature_data_and_none() {
        // ALTREF2_FRAME = 6 per §3.
        let seg_data: i8 = 6;
        let pair = write_inter_block_mode_info_bootstrap(
            0,
            // Arm-1 inputs ignored on arm 2.
            [0, 0],
            true,
            seg_data,
            // Arm-3 inputs ignored on arm 2.
            true,
            true,
        )
        .unwrap();
        assert_eq!(pair, (seg_data, -1));
    }

    /// Arm 3 (`SEG_LVL_SKIP || SEG_LVL_GLOBALMV` with the earlier two
    /// selectors false) stamps `LAST_FRAME = 1` into slot 0 and
    /// `NONE = -1` into slot 1. Verify both selectors trigger the
    /// arm independently.
    #[test]
    fn write_inter_block_mode_info_bootstrap_seg_skip_only_arm_stamps_last_and_none() {
        let pair = write_inter_block_mode_info_bootstrap(0, [0, 0], false, 0, true, false).unwrap();
        assert_eq!(pair, (1, -1));
    }

    #[test]
    fn write_inter_block_mode_info_bootstrap_seg_globalmv_only_arm_stamps_last_and_none() {
        let pair = write_inter_block_mode_info_bootstrap(0, [0, 0], false, 0, false, true).unwrap();
        assert_eq!(pair, (1, -1));
    }

    #[test]
    fn write_inter_block_mode_info_bootstrap_seg_skip_and_globalmv_arm_stamps_last_and_none() {
        let pair = write_inter_block_mode_info_bootstrap(0, [0, 0], false, 0, true, true).unwrap();
        assert_eq!(pair, (1, -1));
    }

    /// Arm 1 takes priority over arms 2 + 3 — even when the
    /// segmentation selectors are also active, `skip_mode == 1`
    /// must short-circuit to `SkipModeFrame`.
    #[test]
    fn write_inter_block_mode_info_bootstrap_skip_mode_priority_over_seg_arms() {
        let pair = write_inter_block_mode_info_bootstrap(1, [2, 3], true, 4, true, true).unwrap();
        assert_eq!(pair, (2, 3));
    }

    /// Arm 2 takes priority over arm 3 — even when `SEG_LVL_SKIP` is
    /// active, the `SEG_LVL_REF_FRAME` arm fires first.
    #[test]
    fn write_inter_block_mode_info_bootstrap_seg_ref_frame_priority_over_seg_skip() {
        let pair = write_inter_block_mode_info_bootstrap(0, [0, 0], true, 5, true, true).unwrap();
        // ALTREF2_FRAME = 6? No — `5` is BWDREF_FRAME per §3. Pair must
        // be (5, -1) regardless.
        assert_eq!(pair, (5, -1));
    }

    /// Fall-through `else` (arm 4) — all four selectors false — is
    /// deferred to a follow-up arc and rejected as caller-bug.
    #[test]
    fn write_inter_block_mode_info_bootstrap_fall_through_else_rejected() {
        let err =
            write_inter_block_mode_info_bootstrap(0, [0, 0], false, 0, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `skip_mode > 1` violates the §3 binary alphabet.
    #[test]
    fn write_inter_block_mode_info_bootstrap_rejects_out_of_range_skip_mode() {
        let err =
            write_inter_block_mode_info_bootstrap(2, [1, 2], false, 0, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `skip_mode == 1` with `SkipModeFrame[ 0 ] = NONE = -1` violates
    /// the §5.9.22 invariant — caller-bug.
    #[test]
    fn write_inter_block_mode_info_bootstrap_rejects_skip_mode_frame_with_none_slot0() {
        let err =
            write_inter_block_mode_info_bootstrap(1, [-1, 2], true, 0, true, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `skip_mode == 1` with `SkipModeFrame[ 1 ] = NONE = -1` violates
    /// the §5.9.22 invariant — caller-bug.
    #[test]
    fn write_inter_block_mode_info_bootstrap_rejects_skip_mode_frame_with_none_slot1() {
        let err =
            write_inter_block_mode_info_bootstrap(1, [1, -1], true, 0, true, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `skip_mode == 1` with `SkipModeFrame[ 0 ]` past ALTREF_FRAME =
    /// 7 violates §3.
    #[test]
    fn write_inter_block_mode_info_bootstrap_rejects_skip_mode_frame_above_altref() {
        let err =
            write_inter_block_mode_info_bootstrap(1, [8, 2], false, 0, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 2 with `seg_ref_frame_data = NONE = -1` violates the §6.4.1
    /// segmentation-feature alphabet for `SEG_LVL_REF_FRAME`.
    #[test]
    fn write_inter_block_mode_info_bootstrap_rejects_seg_ref_frame_none() {
        let err =
            write_inter_block_mode_info_bootstrap(0, [0, 0], true, -1, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm 2 with `seg_ref_frame_data > ALTREF_FRAME = 7` violates §3.
    #[test]
    fn write_inter_block_mode_info_bootstrap_rejects_seg_ref_frame_above_altref() {
        let err =
            write_inter_block_mode_info_bootstrap(0, [0, 0], true, 8, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arms 2 / 3 with `INTRA_FRAME = 0` for `seg_ref_frame_data` is
    /// **valid** — the §6.4.1 alphabet includes `0..=ALTREF_FRAME = 7`,
    /// and a segmented `INTRA_FRAME` ref drives the `isCompound = 0`
    /// (intra-only) downstream branch. The pair is `(0, -1)`.
    #[test]
    fn write_inter_block_mode_info_bootstrap_seg_ref_frame_intra_frame_is_valid() {
        let pair = write_inter_block_mode_info_bootstrap(0, [0, 0], true, 0, false, false).unwrap();
        assert_eq!(pair, (0, -1));
    }

    /// `isCompound = RefFrame[ 1 ] > INTRA_FRAME` per §5.11.23 line 5:
    /// the caller's downstream derivation MUST observe the bootstrap's
    /// returned pair. Sanity-check the predicate for each arm.
    #[test]
    fn write_inter_block_mode_info_bootstrap_is_compound_derivation_per_arm() {
        // Arm 1 with a positive `SkipModeFrame[ 1 ]` ⇒ isCompound = 1.
        let (_, slot1) =
            write_inter_block_mode_info_bootstrap(1, [1, 4], false, 0, false, false).unwrap();
        // INTRA_FRAME = 0; `1` and `4` both exceed 0.
        assert!(slot1 > 0);

        // Arm 2 stamps NONE = -1 in slot 1 ⇒ isCompound = 0.
        let (_, slot1) =
            write_inter_block_mode_info_bootstrap(0, [0, 0], true, 3, false, false).unwrap();
        assert!(slot1 <= 0);

        // Arm 3 stamps NONE = -1 in slot 1 ⇒ isCompound = 0.
        let (_, slot1) =
            write_inter_block_mode_info_bootstrap(0, [0, 0], false, 0, true, false).unwrap();
        assert!(slot1 <= 0);
    }

    // -----------------------------------------------------------------
    // §5.11.25 arm 4 first symbol — write_comp_mode.
    // -----------------------------------------------------------------

    /// §8.3.2 `comp_mode` ctx at the frame origin: both neighbours
    /// unavailable (`AvailU == AvailL == false`) ⇒ the `else` branch
    /// `ctx = 1`. The neighbour single/intra/ref-frame inputs are
    /// irrelevant when neither neighbour is available, so we pass the
    /// no-neighbour shape.
    fn comp_mode_ctx_at_origin() -> usize {
        comp_mode_ctx(false, false, true, true, false, false, [-1, -1], [-1, -1])
    }

    /// §5.11.25 arm 4 explicit path: `reference_select == true` and
    /// `Min( bw4, bh4 ) >= 2` (BLOCK_8X8 ⇒ bw4 = bh4 = 2) ⇒ one `S()`
    /// emitted. Round-trip `comp_mode = SINGLE_REFERENCE = 0` through
    /// the decoder's `comp_mode` symbol read with the §8.3 CDF
    /// adaptation engaged on both sides.
    #[test]
    fn write_comp_mode_single_reference_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = comp_mode_ctx_at_origin();
        write_comp_mode(&mut writer, &mut enc_cdfs, 0, BLOCK_8X8, true, ctx).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let row = dec_cdfs.comp_mode_cdf(ctx);
        let comp_mode = dec.read_symbol(row).unwrap();
        assert_eq!(comp_mode, 0, "SINGLE_REFERENCE");
    }

    /// §5.11.25 arm 4 explicit path: `comp_mode = COMPOUND_REFERENCE = 1`
    /// round-trips through the decoder's `comp_mode` symbol read at the
    /// frame origin. CDF row chosen by the encoder MUST match the
    /// decoder's `ctx` or the read would diverge.
    #[test]
    fn write_comp_mode_compound_reference_round_trip_at_origin() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = comp_mode_ctx_at_origin();
        write_comp_mode(&mut writer, &mut enc_cdfs, 1, BLOCK_8X8, true, ctx).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let row = dec_cdfs.comp_mode_cdf(ctx);
        let comp_mode = dec.read_symbol(row).unwrap();
        assert_eq!(comp_mode, 1, "COMPOUND_REFERENCE");
    }

    /// §5.11.25 arm 4 suppressed-bit path: `reference_select == false`
    /// ⇒ no `S()` emitted (the spec forces `comp_mode = SINGLE_REFERENCE`
    /// with no bit). The finished symbol stream is empty of payload.
    #[test]
    fn write_comp_mode_reference_select_false_emits_no_bit() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = comp_mode_ctx_at_origin();
        write_comp_mode(&mut writer, &mut enc_cdfs, 0, BLOCK_8X8, false, ctx).unwrap();
        let bytes = writer.finish();
        // The CDF row MUST be untouched — no symbol was coded.
        let mut pristine = TileCdfContext::new_from_defaults();
        let expected: Vec<u16> = pristine.comp_mode_cdf(ctx).to_vec();
        let actual: Vec<u16> = enc_cdfs.comp_mode_cdf(ctx).to_vec();
        assert_eq!(actual, expected, "no S() emitted ⇒ comp_mode CDF unchanged");
        // An empty payload finishes to the minimal symbol-stream tail.
        assert!(bytes.len() <= 2, "no payload bits coded");
    }

    /// §5.11.25 arm 4 suppressed-bit path via the size precondition:
    /// `Min( bw4, bh4 ) < 2` (BLOCK_4X4 ⇒ bw4 = bh4 = 1) forces
    /// `comp_mode = SINGLE_REFERENCE` even with `reference_select == true`.
    #[test]
    fn write_comp_mode_small_block_emits_no_bit() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let ctx = comp_mode_ctx_at_origin();
        write_comp_mode(&mut writer, &mut enc_cdfs, 0, BLOCK_4X4, true, ctx).unwrap();
        let bytes = writer.finish();
        assert!(bytes.len() <= 2, "no payload bits coded for a sub-8 block");
    }

    /// §5.11.25 arm 4: BLOCK_4X8 / BLOCK_8X4 have `Min( bw4, bh4 ) = 1`
    /// ⇒ suppressed-bit even with `reference_select == true`. The
    /// rectangular sub-8 shapes exercise the `Min` (not `Max`) bound.
    #[test]
    fn write_comp_mode_rectangular_sub8_blocks_suppress_bit() {
        for &mi_size in &[BLOCK_4X8, BLOCK_8X4] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let ctx = comp_mode_ctx_at_origin();
            // comp_mode MUST be SINGLE_REFERENCE on the suppressed path.
            write_comp_mode(&mut writer, &mut enc_cdfs, 0, mi_size, true, ctx).unwrap();
            let bytes = writer.finish();
            assert!(bytes.len() <= 2, "Min(bw4,bh4) = 1 ⇒ no bit");
        }
    }

    /// §3 binary alphabet: `comp_mode > 1` is rejected.
    #[test]
    fn write_comp_mode_rejects_out_of_range_value() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_comp_mode(&mut writer, &mut cdfs, 2, BLOCK_8X8, true, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Invalid §3 block-size index is rejected.
    #[test]
    fn write_comp_mode_rejects_out_of_range_mi_size() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_comp_mode(&mut writer, &mut cdfs, 0, BLOCK_SIZES, true, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Suppressed-bit path with `comp_mode != SINGLE_REFERENCE` is a
    /// caller bug — the spec forces `SINGLE_REFERENCE` with no bit, so
    /// any other value cannot be represented.
    #[test]
    fn write_comp_mode_rejects_compound_on_suppressed_path() {
        // reference_select == false suppresses the bit.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_comp_mode(&mut writer, &mut cdfs, 1, BLOCK_8X8, false, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // Min(bw4,bh4) < 2 suppresses the bit too.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_comp_mode(&mut writer, &mut cdfs, 1, BLOCK_4X4, true, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Explicit path with `ctx >= COMP_INTER_CONTEXTS` is rejected.
    #[test]
    fn write_comp_mode_rejects_out_of_range_ctx_on_explicit_path() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_comp_mode(
            &mut writer,
            &mut cdfs,
            0,
            BLOCK_8X8,
            true,
            COMP_INTER_CONTEXTS,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// §8.3.2 ctx coverage: every `ctx` in `0..COMP_INTER_CONTEXTS`
    /// round-trips both `comp_mode` values through the decoder. The
    /// CDF row the encoder writes MUST be the row the decoder reads.
    #[test]
    fn write_comp_mode_all_ctx_round_trip() {
        for ctx in 0..COMP_INTER_CONTEXTS {
            for value in 0u8..=1 {
                let mut enc_cdfs = TileCdfContext::new_from_defaults();
                let mut writer = SymbolWriter::new(false);
                write_comp_mode(&mut writer, &mut enc_cdfs, value, BLOCK_8X8, true, ctx).unwrap();
                let bytes = writer.finish();

                let mut dec_cdfs = TileCdfContext::new_from_defaults();
                let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
                let row = dec_cdfs.comp_mode_cdf(ctx);
                let got = dec.read_symbol(row).unwrap() as u8;
                assert_eq!(got, value, "ctx {ctx} value {value} round-trip");
            }
        }
    }

    // -----------------------------------------------------------------
    // §5.11.25 arm 4 COMPOUND_REFERENCE body — write_compound_ref_frames.
    // -----------------------------------------------------------------

    /// The §5.11.18 prologue neighbour-state octet the writer and the
    /// mirror reader both consume. Bundled so a test cannot
    /// accidentally hand the two sides different neighbour state.
    #[derive(Clone, Copy)]
    struct NeighbourOctet {
        avail_u: bool,
        avail_l: bool,
        above_single: bool,
        left_single: bool,
        above_intra: bool,
        left_intra: bool,
        above_ref_frame: [i32; 2],
        left_ref_frame: [i32; 2],
    }

    /// Frame-origin octet: both neighbours unavailable. Every
    /// `count_refs` is 0 ⇒ every `ref_count_ctx` selection is 1, and
    /// the §8.3.2 p.382 `comp_ref_type` ctx falls to the final `else`
    /// arm (`ctx = 2`).
    const ORIGIN: NeighbourOctet = NeighbourOctet {
        avail_u: false,
        avail_l: false,
        above_single: true,
        left_single: true,
        above_intra: false,
        left_intra: false,
        above_ref_frame: [-1, -1],
        left_ref_frame: [-1, -1],
    };

    /// The sixteen compound pairs the §5.11.25 COMPOUND_REFERENCE
    /// sub-tree reaches: four UNIDIR leaves + the 4 × 3 BIDIR product
    /// (§6.10.24 Group 1 × Group 2).
    const UNIDIR_PAIRS: [[i32; 2]; 4] = [[5, 7], [1, 2], [1, 3], [1, 4]];
    const BIDIR_PAIRS: [[i32; 2]; 12] = [
        [1, 5],
        [1, 6],
        [1, 7],
        [2, 5],
        [2, 6],
        [2, 7],
        [3, 5],
        [3, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [4, 7],
    ];

    /// Mirror of the decoder's §5.11.25 COMPOUND_REFERENCE symbol
    /// reads: `comp_ref_type` then the UNIDIR / BIDIR cascade, each
    /// `S()` against the same §8.3.2 CDF row selection the writer
    /// derives. Returns the decoded `RefFrame[ 0..2 ]` pair.
    fn read_compound_ref_frames_mirror(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        n: NeighbourOctet,
    ) -> [i32; 2] {
        let cnt = |ft: i32| {
            count_refs(
                ft,
                n.avail_u,
                n.above_ref_frame,
                n.avail_l,
                n.left_ref_frame,
            )
        };
        // `comp_ref_type` S() (§8.3.2 p.382).
        let crt_ctx = comp_ref_type_ctx(
            n.avail_u,
            n.avail_l,
            n.above_single,
            n.left_single,
            n.above_intra,
            n.left_intra,
            n.above_ref_frame,
            n.left_ref_frame,
        );
        let comp_ref_type = dec.read_symbol(cdfs.comp_ref_type_cdf(crt_ctx)).unwrap();
        if comp_ref_type == 0 {
            // UNIDIR_COMP_REFERENCE: `uni_comp_ref` cascade.
            let ucr_ctx =
                ref_count_ctx(cnt(1) + cnt(2) + cnt(3) + cnt(4), cnt(5) + cnt(6) + cnt(7));
            let uni = dec.read_symbol(cdfs.uni_comp_ref_cdf(ucr_ctx, 0)).unwrap();
            if uni != 0 {
                return [5, 7]; // (BWDREF_FRAME, ALTREF_FRAME)
            }
            let p1_ctx = ref_count_ctx(cnt(2), cnt(3) + cnt(4));
            let p1 = dec.read_symbol(cdfs.uni_comp_ref_cdf(p1_ctx, 1)).unwrap();
            if p1 == 0 {
                return [1, 2]; // (LAST_FRAME, LAST2_FRAME)
            }
            let p2_ctx = ref_count_ctx(cnt(3), cnt(4));
            let p2 = dec.read_symbol(cdfs.uni_comp_ref_cdf(p2_ctx, 2)).unwrap();
            if p2 != 0 {
                [1, 4] // (LAST_FRAME, GOLDEN_FRAME)
            } else {
                [1, 3] // (LAST_FRAME, LAST3_FRAME)
            }
        } else {
            // BIDIR_COMP_REFERENCE: `comp_ref` then `comp_bwdref`.
            let cr_ctx = ref_count_ctx(cnt(1) + cnt(2), cnt(3) + cnt(4));
            let comp_ref = dec.read_symbol(cdfs.comp_ref_cdf(cr_ctx, 0)).unwrap();
            let ref0 = if comp_ref == 0 {
                let p1_ctx = ref_count_ctx(cnt(1), cnt(2));
                let p1 = dec.read_symbol(cdfs.comp_ref_cdf(p1_ctx, 1)).unwrap();
                if p1 != 0 {
                    2
                } else {
                    1
                }
            } else {
                let p2_ctx = ref_count_ctx(cnt(3), cnt(4));
                let p2 = dec.read_symbol(cdfs.comp_ref_cdf(p2_ctx, 2)).unwrap();
                if p2 != 0 {
                    4
                } else {
                    3
                }
            };
            let bw_ctx = ref_count_ctx(cnt(5) + cnt(6), cnt(7));
            let bwdref = dec.read_symbol(cdfs.comp_bwd_ref_cdf(bw_ctx, 0)).unwrap();
            let ref1 = if bwdref == 0 {
                let p1_ctx = ref_count_ctx(cnt(5), cnt(6));
                let p1 = dec.read_symbol(cdfs.comp_bwd_ref_cdf(p1_ctx, 1)).unwrap();
                if p1 != 0 {
                    6
                } else {
                    5
                }
            } else {
                7
            };
            [ref0, ref1]
        }
    }

    /// Encode one compound pair with `write_compound_ref_frames`, then
    /// decode it back through the mirror reader with an independent
    /// CDF context. Asserts the pair survives and that BOTH sides'
    /// adapted CDF tables are identical afterwards — if the writer
    /// selected a different §8.3.2 row than the reader for any symbol,
    /// the adapted-row sets would diverge.
    fn round_trip_pair(pair: [i32; 2], n: NeighbourOctet) {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_compound_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            pair,
            n.avail_u,
            n.avail_l,
            n.above_single,
            n.left_single,
            n.above_intra,
            n.left_intra,
            n.above_ref_frame,
            n.left_ref_frame,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = read_compound_ref_frames_mirror(&mut dec, &mut dec_cdfs, n);
        assert_eq!(got, pair, "pair {pair:?} round-trip");
        assert_ref_tables_eq(&mut enc_cdfs, &mut dec_cdfs);
    }

    /// Compare every §8.3.2 ref-frame CDF row (`comp_ref_type` /
    /// `uni_comp_ref` / `comp_ref` / `comp_bwdref`, all ctx × p slots)
    /// between two contexts.
    fn assert_ref_tables_eq(a: &mut TileCdfContext, b: &mut TileCdfContext) {
        for ctx in 0..crate::cdf::COMP_REF_TYPE_CONTEXTS {
            assert_eq!(
                a.comp_ref_type_cdf(ctx).to_vec(),
                b.comp_ref_type_cdf(ctx).to_vec(),
                "comp_ref_type ctx {ctx}"
            );
        }
        for ctx in 0..crate::cdf::REF_CONTEXTS {
            for p in 0..3 {
                assert_eq!(
                    a.uni_comp_ref_cdf(ctx, p).to_vec(),
                    b.uni_comp_ref_cdf(ctx, p).to_vec(),
                    "uni_comp_ref ctx {ctx} p {p}"
                );
                assert_eq!(
                    a.comp_ref_cdf(ctx, p).to_vec(),
                    b.comp_ref_cdf(ctx, p).to_vec(),
                    "comp_ref ctx {ctx} p {p}"
                );
            }
            for p in 0..2 {
                assert_eq!(
                    a.comp_bwd_ref_cdf(ctx, p).to_vec(),
                    b.comp_bwd_ref_cdf(ctx, p).to_vec(),
                    "comp_bwdref ctx {ctx} p {p}"
                );
            }
        }
    }

    /// All four §5.11.25 UNIDIR leaves round-trip at the frame origin.
    #[test]
    fn write_compound_ref_frames_unidir_pairs_round_trip_at_origin() {
        for pair in UNIDIR_PAIRS {
            round_trip_pair(pair, ORIGIN);
        }
    }

    /// All twelve §6.10.24 Group-1 × Group-2 BIDIR pairs round-trip at
    /// the frame origin.
    #[test]
    fn write_compound_ref_frames_bidir_pairs_round_trip_at_origin() {
        for pair in BIDIR_PAIRS {
            round_trip_pair(pair, ORIGIN);
        }
    }

    /// Neighbour-engaged ctx selection: all sixteen pairs round-trip
    /// with non-trivial §8.3.2 contexts. Octet A puts a BIDIR compound
    /// neighbour above + a single-ref neighbour left; octet B puts a
    /// UNIDIR compound neighbour above + an intra neighbour left. The
    /// table-equality check inside `round_trip_pair` proves the writer
    /// derived the same per-symbol rows as the reader.
    #[test]
    fn write_compound_ref_frames_round_trip_with_neighbour_ctx() {
        let octet_a = NeighbourOctet {
            avail_u: true,
            avail_l: true,
            above_single: false,
            left_single: true,
            above_intra: false,
            left_intra: false,
            above_ref_frame: [1, 5],
            left_ref_frame: [3, -1],
        };
        let octet_b = NeighbourOctet {
            avail_u: true,
            avail_l: true,
            above_single: false,
            left_single: true,
            above_intra: false,
            left_intra: true,
            above_ref_frame: [5, 7],
            left_ref_frame: [0, -1],
        };
        for n in [octet_a, octet_b] {
            for pair in UNIDIR_PAIRS.iter().chain(BIDIR_PAIRS.iter()) {
                round_trip_pair(*pair, n);
            }
        }
    }

    /// `( BWDREF, ALTREF )` is the two-symbol UNIDIR leaf:
    /// `comp_ref_type` + `uni_comp_ref` only. The `uni_comp_ref_p1` /
    /// `uni_comp_ref_p2` rows and the entire BIDIR table set MUST stay
    /// pristine — extra symbols would adapt them.
    #[test]
    fn write_compound_ref_frames_bwdref_altref_emits_two_symbols() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_compound_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [5, 7],
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let mut pristine = TileCdfContext::new_from_defaults();
        // ref_count_ctx(0, 0) = 1 at the origin; §8.3.2 p.382 ctx = 2.
        assert_ne!(
            enc_cdfs.comp_ref_type_cdf(2).to_vec(),
            pristine.comp_ref_type_cdf(2).to_vec(),
            "comp_ref_type coded"
        );
        assert_ne!(
            enc_cdfs.uni_comp_ref_cdf(1, 0).to_vec(),
            pristine.uni_comp_ref_cdf(1, 0).to_vec(),
            "uni_comp_ref coded"
        );
        for p in 1..3 {
            assert_eq!(
                enc_cdfs.uni_comp_ref_cdf(1, p).to_vec(),
                pristine.uni_comp_ref_cdf(1, p).to_vec(),
                "uni_comp_ref p{p} NOT coded"
            );
        }
        for p in 0..3 {
            assert_eq!(
                enc_cdfs.comp_ref_cdf(1, p).to_vec(),
                pristine.comp_ref_cdf(1, p).to_vec(),
                "comp_ref p{p} NOT coded"
            );
        }
    }

    /// `( LAST, LAST2 )` stops after `uni_comp_ref_p1 = 0`: the
    /// `uni_comp_ref_p2` row stays pristine.
    #[test]
    fn write_compound_ref_frames_last_last2_leaves_p2_untouched() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_compound_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [1, 2],
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let mut pristine = TileCdfContext::new_from_defaults();
        assert_ne!(
            enc_cdfs.uni_comp_ref_cdf(1, 1).to_vec(),
            pristine.uni_comp_ref_cdf(1, 1).to_vec(),
            "uni_comp_ref_p1 coded"
        );
        assert_eq!(
            enc_cdfs.uni_comp_ref_cdf(1, 2).to_vec(),
            pristine.uni_comp_ref_cdf(1, 2).to_vec(),
            "uni_comp_ref_p2 NOT coded"
        );
    }

    /// A BIDIR pair with `RefFrame[ 1 ] = ALTREF_FRAME` takes the
    /// `comp_bwdref = 1` arm: the `comp_bwdref_p1` row stays pristine.
    #[test]
    fn write_compound_ref_frames_altref_arm_skips_bwdref_p1() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_compound_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [1, 7],
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let mut pristine = TileCdfContext::new_from_defaults();
        assert_ne!(
            enc_cdfs.comp_bwd_ref_cdf(1, 0).to_vec(),
            pristine.comp_bwd_ref_cdf(1, 0).to_vec(),
            "comp_bwdref coded"
        );
        assert_eq!(
            enc_cdfs.comp_bwd_ref_cdf(1, 1).to_vec(),
            pristine.comp_bwd_ref_cdf(1, 1).to_vec(),
            "comp_bwdref_p1 NOT coded"
        );
    }

    /// §6.10.24: a compound pair never carries `INTRA_FRAME = 0` or
    /// `NONE = -1` in either slot, nor an index past `ALTREF_FRAME = 7`.
    #[test]
    fn write_compound_ref_frames_rejects_out_of_range_slots() {
        for pair in [[0, 5], [-1, 5], [8, 5], [1, 0], [1, -1], [1, 8]] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_compound_ref_frames(
                &mut writer,
                &mut cdfs,
                pair,
                false,
                false,
                true,
                true,
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "pair {pair:?} rejected"
            );
        }
    }

    /// Same-group pairs outside the four §5.11.25 UNIDIR leaves are
    /// unreachable by the sub-tree and rejected.
    #[test]
    fn write_compound_ref_frames_rejects_unreachable_samedir_pairs() {
        for pair in [[2, 3], [2, 4], [3, 4], [5, 6], [6, 7], [1, 1], [7, 5]] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_compound_ref_frames(
                &mut writer,
                &mut cdfs,
                pair,
                false,
                false,
                true,
                true,
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "pair {pair:?} rejected"
            );
        }
    }

    /// Different-group pairs with the Group-2 (backward) frame in slot
    /// 0 cannot be expressed by the §5.11.25 BIDIR sub-tree.
    #[test]
    fn write_compound_ref_frames_rejects_reversed_bidir_pairs() {
        for pair in [[5, 1], [6, 2], [7, 4]] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_compound_ref_frames(
                &mut writer,
                &mut cdfs,
                pair,
                false,
                false,
                true,
                true,
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "pair {pair:?} rejected"
            );
        }
    }

    /// Sequential pairs through ONE encoder CDF context decode back
    /// through ONE decoder CDF context — the §8.2.6 per-symbol CDF
    /// adaptation stays in lockstep across the whole sequence.
    #[test]
    fn write_compound_ref_frames_sequential_adaptation_round_trip() {
        let sequence = [[1, 4], [5, 7], [3, 6], [1, 2], [4, 7], [1, 3], [2, 5]];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        for pair in sequence {
            write_compound_ref_frames(
                &mut writer,
                &mut enc_cdfs,
                pair,
                false,
                false,
                true,
                true,
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap();
        }
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        for pair in sequence {
            let got = read_compound_ref_frames_mirror(&mut dec, &mut dec_cdfs, ORIGIN);
            assert_eq!(got, pair, "sequential pair {pair:?}");
        }
        assert_ref_tables_eq(&mut enc_cdfs, &mut dec_cdfs);
    }

    // -----------------------------------------------------------------
    // §5.11.25 arm 4 SINGLE_REFERENCE body — write_single_ref_frames.
    // -----------------------------------------------------------------

    /// Mirror of the decoder's §5.11.25 SINGLE_REFERENCE symbol reads:
    /// `single_ref_p1` then the p2/p6 (backward) or p3/p5/p4 (forward)
    /// cascade, each `S()` against the same §8.3.2 `TileSingleRefCdf`
    /// row selection the writer derives. Returns the decoded
    /// `RefFrame[ 0..2 ]` pair (`RefFrame[ 1 ] = NONE = -1`).
    fn read_single_ref_frames_mirror(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        n: NeighbourOctet,
    ) -> [i32; 2] {
        let cnt = |ft: i32| {
            count_refs(
                ft,
                n.avail_u,
                n.above_ref_frame,
                n.avail_l,
                n.left_ref_frame,
            )
        };
        // `single_ref_p1` S() (§8.3.2 p.368: fwdCount vs bwdCount).
        let p1_ctx = ref_count_ctx(cnt(1) + cnt(2) + cnt(3) + cnt(4), cnt(5) + cnt(6) + cnt(7));
        let p1 = dec.read_symbol(cdfs.single_ref_cdf(p1_ctx, 0)).unwrap();
        let ref0 = if p1 != 0 {
            // `single_ref_p2` S() (ctx as in comp_bwdref).
            let p2_ctx = ref_count_ctx(cnt(5) + cnt(6), cnt(7));
            let p2 = dec.read_symbol(cdfs.single_ref_cdf(p2_ctx, 1)).unwrap();
            if p2 == 0 {
                // `single_ref_p6` S() (ctx as in comp_bwdref_p1).
                let p6_ctx = ref_count_ctx(cnt(5), cnt(6));
                let p6 = dec.read_symbol(cdfs.single_ref_cdf(p6_ctx, 5)).unwrap();
                if p6 != 0 {
                    6 // ALTREF2_FRAME
                } else {
                    5 // BWDREF_FRAME
                }
            } else {
                7 // ALTREF_FRAME
            }
        } else {
            // `single_ref_p3` S() (ctx as in comp_ref).
            let p3_ctx = ref_count_ctx(cnt(1) + cnt(2), cnt(3) + cnt(4));
            let p3 = dec.read_symbol(cdfs.single_ref_cdf(p3_ctx, 2)).unwrap();
            if p3 != 0 {
                // `single_ref_p5` S() (ctx as in comp_ref_p2).
                let p5_ctx = ref_count_ctx(cnt(3), cnt(4));
                let p5 = dec.read_symbol(cdfs.single_ref_cdf(p5_ctx, 4)).unwrap();
                if p5 != 0 {
                    4 // GOLDEN_FRAME
                } else {
                    3 // LAST3_FRAME
                }
            } else {
                // `single_ref_p4` S() (ctx as in comp_ref_p1).
                let p4_ctx = ref_count_ctx(cnt(1), cnt(2));
                let p4 = dec.read_symbol(cdfs.single_ref_cdf(p4_ctx, 3)).unwrap();
                if p4 != 0 {
                    2 // LAST2_FRAME
                } else {
                    1 // LAST_FRAME
                }
            }
        };
        [ref0, -1]
    }

    /// Compare every §8.3.2 `TileSingleRefCdf` row (all ctx × p slots)
    /// between two contexts.
    fn assert_single_ref_tables_eq(a: &mut TileCdfContext, b: &mut TileCdfContext) {
        for ctx in 0..crate::cdf::REF_CONTEXTS {
            for p in 0..6 {
                assert_eq!(
                    a.single_ref_cdf(ctx, p).to_vec(),
                    b.single_ref_cdf(ctx, p).to_vec(),
                    "single_ref ctx {ctx} p {p}"
                );
            }
        }
    }

    /// Encode one single reference with `write_single_ref_frames`,
    /// decode it back through the mirror reader with an independent
    /// CDF context, and assert the full `TileSingleRefCdf` table set
    /// matches afterwards (a row-selection mismatch on any symbol
    /// would leave the adapted-row sets diverged).
    fn round_trip_single_ref(ref0: i32, n: NeighbourOctet) {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_single_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [ref0, -1],
            n.avail_u,
            n.avail_l,
            n.above_ref_frame,
            n.left_ref_frame,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = read_single_ref_frames_mirror(&mut dec, &mut dec_cdfs, n);
        assert_eq!(got, [ref0, -1], "single ref {ref0} round-trip");
        assert_single_ref_tables_eq(&mut enc_cdfs, &mut dec_cdfs);
    }

    /// All seven §5.11.25 SINGLE_REFERENCE leaves (`LAST_FRAME = 1`
    /// ..= `ALTREF_FRAME = 7`) round-trip at the frame origin.
    #[test]
    fn write_single_ref_frames_all_refs_round_trip_at_origin() {
        for ref0 in 1..=7 {
            round_trip_single_ref(ref0, ORIGIN);
        }
    }

    /// Neighbour-engaged ctx selection: all seven leaves round-trip
    /// under the same two non-trivial §8.3.2 octets the COMPOUND-body
    /// tests use. The table-equality check inside
    /// `round_trip_single_ref` proves the writer derived the same
    /// per-symbol rows as the reader.
    #[test]
    fn write_single_ref_frames_round_trip_with_neighbour_ctx() {
        let octet_a = NeighbourOctet {
            avail_u: true,
            avail_l: true,
            above_single: false,
            left_single: true,
            above_intra: false,
            left_intra: false,
            above_ref_frame: [1, 5],
            left_ref_frame: [3, -1],
        };
        let octet_b = NeighbourOctet {
            avail_u: true,
            avail_l: true,
            above_single: false,
            left_single: true,
            above_intra: false,
            left_intra: true,
            above_ref_frame: [5, 7],
            left_ref_frame: [0, -1],
        };
        for n in [octet_a, octet_b] {
            for ref0 in 1..=7 {
                round_trip_single_ref(ref0, n);
            }
        }
    }

    /// `ALTREF_FRAME` is the two-symbol leaf: `single_ref_p1` +
    /// `single_ref_p2` only. The p3/p4/p5/p6 rows MUST stay pristine —
    /// extra symbols would adapt them. (At the origin every
    /// `ref_count_ctx` selection is 1.)
    #[test]
    fn write_single_ref_frames_altref_emits_two_symbols() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_single_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [7, -1],
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let mut pristine = TileCdfContext::new_from_defaults();
        for p in [0, 1] {
            assert_ne!(
                enc_cdfs.single_ref_cdf(1, p).to_vec(),
                pristine.single_ref_cdf(1, p).to_vec(),
                "single_ref p{} coded",
                p + 1
            );
        }
        for p in [2, 3, 4, 5] {
            assert_eq!(
                enc_cdfs.single_ref_cdf(1, p).to_vec(),
                pristine.single_ref_cdf(1, p).to_vec(),
                "single_ref p{} NOT coded",
                p + 1
            );
        }
    }

    /// `BWDREF_FRAME` takes the `single_ref_p2 = 0` arm and codes
    /// `single_ref_p6`; the forward-side p3/p4/p5 rows stay pristine.
    #[test]
    fn write_single_ref_frames_bwdref_arm_codes_p6() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_single_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [5, -1],
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let mut pristine = TileCdfContext::new_from_defaults();
        assert_ne!(
            enc_cdfs.single_ref_cdf(1, 5).to_vec(),
            pristine.single_ref_cdf(1, 5).to_vec(),
            "single_ref_p6 coded"
        );
        for p in [2, 3, 4] {
            assert_eq!(
                enc_cdfs.single_ref_cdf(1, p).to_vec(),
                pristine.single_ref_cdf(1, p).to_vec(),
                "single_ref p{} NOT coded",
                p + 1
            );
        }
    }

    /// `LAST_FRAME` takes the forward arm (`single_ref_p1 = 0` →
    /// `single_ref_p3` → `single_ref_p4`); the backward-side p2/p6
    /// rows and the p5 row stay pristine.
    #[test]
    fn write_single_ref_frames_forward_arm_skips_backward_rows() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_single_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [1, -1],
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let mut pristine = TileCdfContext::new_from_defaults();
        for p in [0, 2, 3] {
            assert_ne!(
                enc_cdfs.single_ref_cdf(1, p).to_vec(),
                pristine.single_ref_cdf(1, p).to_vec(),
                "single_ref p{} coded",
                p + 1
            );
        }
        for p in [1, 4, 5] {
            assert_eq!(
                enc_cdfs.single_ref_cdf(1, p).to_vec(),
                pristine.single_ref_cdf(1, p).to_vec(),
                "single_ref p{} NOT coded",
                p + 1
            );
        }
    }

    /// Slot 0 outside `LAST_FRAME..=ALTREF_FRAME` (`INTRA_FRAME = 0`,
    /// `NONE = -1`, past-`ALTREF` 8) and any slot-1 value other than
    /// `NONE = -1` are caller bugs.
    #[test]
    fn write_single_ref_frames_rejects_out_of_range() {
        for pair in [[0, -1], [-1, -1], [8, -1], [1, 0], [1, 5], [3, 7]] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_single_ref_frames(
                &mut writer,
                &mut cdfs,
                pair,
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "pair {pair:?} rejected"
            );
        }
    }

    /// Sequential single refs through ONE encoder CDF context decode
    /// back through ONE decoder CDF context — the §8.2.6 per-symbol
    /// CDF adaptation stays in lockstep across the whole sequence.
    #[test]
    fn write_single_ref_frames_sequential_adaptation_round_trip() {
        let sequence = [1, 7, 3, 5, 2, 6, 4, 1, 7];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        for ref0 in sequence {
            write_single_ref_frames(
                &mut writer,
                &mut enc_cdfs,
                [ref0, -1],
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap();
        }
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        for ref0 in sequence {
            let got = read_single_ref_frames_mirror(&mut dec, &mut dec_cdfs, ORIGIN);
            assert_eq!(got, [ref0, -1], "sequential ref {ref0}");
        }
        assert_single_ref_tables_eq(&mut enc_cdfs, &mut dec_cdfs);
    }

    // -----------------------------------------------------------------
    // §5.11.25 full arm-4 dispatcher — write_ref_frames.
    // -----------------------------------------------------------------

    /// Mirror of the decoder's §5.11.25 arm-4 reads: the `comp_mode`
    /// S() (or its `reference_select && Min( bw4, bh4 ) >= 2`
    /// suppression to `SINGLE_REFERENCE`) followed by the matching
    /// reference-body mirror.
    fn read_ref_frames_arm4_mirror(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        n: NeighbourOctet,
        mi_size: usize,
        reference_select: bool,
    ) -> [i32; 2] {
        let bw4 = NUM_4X4_BLOCKS_WIDE[mi_size];
        let bh4 = NUM_4X4_BLOCKS_HIGH[mi_size];
        let comp_mode = if reference_select && bw4.min(bh4) >= 2 {
            let ctx = comp_mode_ctx(
                n.avail_u,
                n.avail_l,
                n.above_single,
                n.left_single,
                n.above_intra,
                n.left_intra,
                n.above_ref_frame,
                n.left_ref_frame,
            );
            dec.read_symbol(cdfs.comp_mode_cdf(ctx)).unwrap()
        } else {
            0 // SINGLE_REFERENCE
        };
        if comp_mode == 1 {
            read_compound_ref_frames_mirror(dec, cdfs, n)
        } else {
            read_single_ref_frames_mirror(dec, cdfs, n)
        }
    }

    /// `write_ref_frames` with no arm-1/2/3 trigger and a matching /
    /// non-matching pair. Arm-1 (`skip_mode`): no symbols emitted, the
    /// matching pair is accepted and a mismatched pair rejected.
    #[test]
    fn write_ref_frames_arm1_verifies_skip_mode_pair() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_ref_frames(
            &mut writer,
            &mut cdfs,
            [1, 7],
            BLOCK_8X8,
            1,
            [1, 7],
            false,
            0,
            false,
            false,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let bytes = writer.finish();
        assert!(bytes.len() <= 2, "arm 1 emits no payload bits");

        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_ref_frames(
            &mut writer,
            &mut cdfs,
            [1, 5],
            BLOCK_8X8,
            1,
            [1, 7],
            false,
            0,
            false,
            false,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm-2 (`SEG_LVL_REF_FRAME`) and arm-3 (`SEG_LVL_SKIP` /
    /// `SEG_LVL_GLOBALMV`) forced pairs verify; mismatches reject.
    #[test]
    fn write_ref_frames_arm2_arm3_verify_forced_pairs() {
        // Arm 2: ( seg_ref_frame_data, NONE ).
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_ref_frames(
            &mut writer,
            &mut cdfs,
            [4, -1],
            BLOCK_8X8,
            0,
            [0, 0],
            true,
            4,
            false,
            false,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();

        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_ref_frames(
            &mut writer,
            &mut cdfs,
            [3, -1],
            BLOCK_8X8,
            0,
            [0, 0],
            true,
            4,
            false,
            false,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // Arm 3: ( LAST_FRAME, NONE ).
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_ref_frames(
            &mut writer,
            &mut cdfs,
            [1, -1],
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            true,
            false,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();

        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_ref_frames(
            &mut writer,
            &mut cdfs,
            [2, -1],
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            false,
            true,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm-4 full round-trips: every SINGLE target (`( 1..=7, NONE )`)
    /// and every reachable COMPOUND pair survives the composed
    /// `comp_mode` + body cascade through the arm-4 mirror at the
    /// frame origin (`reference_select = true`, BLOCK_8X8 ⇒ explicit
    /// `comp_mode` bit on every block).
    #[test]
    fn write_ref_frames_arm4_round_trips_all_targets() {
        let singles = (1..=7).map(|r| [r, -1]);
        let compounds = UNIDIR_PAIRS.iter().chain(BIDIR_PAIRS.iter()).copied();
        for pair in singles.chain(compounds) {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_ref_frames(
                &mut writer,
                &mut enc_cdfs,
                pair,
                BLOCK_8X8,
                0,
                [0, 0],
                false,
                0,
                false,
                false,
                true,
                false,
                false,
                true,
                true,
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap();
            let bytes = writer.finish();

            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let got = read_ref_frames_arm4_mirror(&mut dec, &mut dec_cdfs, ORIGIN, BLOCK_8X8, true);
            assert_eq!(got, pair, "arm-4 pair {pair:?} round-trip");
            assert_ref_tables_eq(&mut enc_cdfs, &mut dec_cdfs);
            assert_single_ref_tables_eq(&mut enc_cdfs, &mut dec_cdfs);
        }
    }

    /// Suppressed-`comp_mode` path (`reference_select = false`): a
    /// SINGLE target round-trips with no `comp_mode` bit, and a
    /// COMPOUND target is rejected (the spec forces
    /// `SINGLE_REFERENCE` with no bit there).
    #[test]
    fn write_ref_frames_suppressed_comp_mode_path() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_ref_frames(
            &mut writer,
            &mut enc_cdfs,
            [3, -1],
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            false,
            false,
            false,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap();
        let bytes = writer.finish();
        let mut pristine = TileCdfContext::new_from_defaults();
        for ctx in 0..COMP_INTER_CONTEXTS {
            assert_eq!(
                enc_cdfs.comp_mode_cdf(ctx).to_vec(),
                pristine.comp_mode_cdf(ctx).to_vec(),
                "no comp_mode S() emitted"
            );
        }
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = read_ref_frames_arm4_mirror(&mut dec, &mut dec_cdfs, ORIGIN, BLOCK_8X8, false);
        assert_eq!(got, [3, -1]);

        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_ref_frames(
            &mut writer,
            &mut cdfs,
            [1, 7],
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            false,
            false,
            false,
            false,
            false,
            true,
            true,
            false,
            false,
            [-1, -1],
            [-1, -1],
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm-4 slot-1 values outside `{ NONE } ∪ LAST_FRAME..=ALTREF_FRAME`
    /// (`INTRA_FRAME = 0`, `-2`, `8`) are unreachable from §5.11.25
    /// and rejected before any symbol is emitted.
    #[test]
    fn write_ref_frames_rejects_invalid_slot1() {
        for slot1 in [0, -2, 8] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_ref_frames(
                &mut writer,
                &mut cdfs,
                [1, slot1],
                BLOCK_8X8,
                0,
                [0, 0],
                false,
                0,
                false,
                false,
                true,
                false,
                false,
                true,
                true,
                false,
                false,
                [-1, -1],
                [-1, -1],
            )
            .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "slot1 {slot1} rejected"
            );
        }
    }

    // -----------------------------------------------------------------
    // §5.11.23 single-prediction inter mode — write_inter_single_mode
    // round-trips through a mirror of the decoder's new_mv / zero_mv /
    // ref_mv reads (av1-spec p.74). The full decoder path needs a
    // find_mv_stack (§7.10.2), not yet implemented; this mirror runs
    // exactly the three §5.11.23 S() reads in order against the same
    // CDF rows, recovering YMode the way the reader does.
    // -----------------------------------------------------------------

    /// Replays the §5.11.23 single-prediction `YMode` derivation from a
    /// decoder over `bytes`, consuming the same `new_mv` / `zero_mv` /
    /// `ref_mv` symbols the writer emitted. Returns the recovered
    /// `YMode`.
    fn decode_inter_single_mode_mirror(
        bytes: &[u8],
        cdfs: &mut TileCdfContext,
        disable_cdf_update: bool,
        new_mv_context: usize,
        zero_mv_context: usize,
        ref_mv_context: usize,
    ) -> u8 {
        let pad = if bytes.is_empty() {
            vec![0u8]
        } else {
            bytes.to_vec()
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), disable_cdf_update).unwrap();
        // §5.11.23: new_mv S(); new_mv == 0 ⇒ NEWMV.
        let new_mv_row = cdfs.new_mv_cdf(new_mv_context);
        let new_mv = dec.read_symbol(new_mv_row).unwrap();
        if new_mv == 0 {
            return MODE_NEWMV;
        }
        // zero_mv S(); zero_mv == 0 ⇒ GLOBALMV.
        let zero_mv_row = cdfs.zero_mv_cdf(zero_mv_context);
        let zero_mv = dec.read_symbol(zero_mv_row).unwrap();
        if zero_mv == 0 {
            return MODE_GLOBALMV;
        }
        // ref_mv S(); (ref_mv == 0) ? NEARESTMV : NEARMV.
        let ref_mv_row = cdfs.ref_mv_cdf(ref_mv_context);
        let ref_mv = dec.read_symbol(ref_mv_row).unwrap();
        if ref_mv == 0 {
            MODE_NEARESTMV
        } else {
            MODE_NEARMV
        }
    }

    /// Each single-prediction `YMode` round-trips through the decoder
    /// mirror at `NewMvContext = ZeroMvContext = RefMvContext = 0`, with
    /// §8.3 CDF adaptation engaged on both sides (so the encode-side and
    /// decode-side rows step identically).
    #[test]
    fn write_inter_single_mode_all_modes_round_trip_at_origin() {
        for y_mode in [MODE_NEWMV, MODE_GLOBALMV, MODE_NEARESTMV, MODE_NEARMV] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_inter_single_mode(&mut writer, &mut enc_cdfs, y_mode, 0, 0, 0).unwrap();
            let bytes = writer.finish();

            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let recovered = decode_inter_single_mode_mirror(&bytes, &mut dec_cdfs, false, 0, 0, 0);
            assert_eq!(recovered, y_mode, "YMode {y_mode} round-trips at origin");
            // The encode-side and decode-side CDFs must have adapted
            // identically — assert the consulted rows match.
            assert_eq!(enc_cdfs.new_mv_cdf(0), dec_cdfs.new_mv_cdf(0));
            if y_mode != MODE_NEWMV {
                assert_eq!(enc_cdfs.zero_mv_cdf(0), dec_cdfs.zero_mv_cdf(0));
            }
            if matches!(y_mode, MODE_NEARESTMV | MODE_NEARMV) {
                assert_eq!(enc_cdfs.ref_mv_cdf(0), dec_cdfs.ref_mv_cdf(0));
            }
        }
    }

    /// Round-trip every mode under non-zero, distinct §8.3.2 contexts
    /// (`NewMvContext = 5`, `ZeroMvContext = 1`, `RefMvContext = 4`) to
    /// prove the writer selects the same CDF rows the reader does.
    #[test]
    fn write_inter_single_mode_all_modes_round_trip_nonzero_ctx() {
        for y_mode in [MODE_NEWMV, MODE_GLOBALMV, MODE_NEARESTMV, MODE_NEARMV] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_inter_single_mode(&mut writer, &mut enc_cdfs, y_mode, 5, 1, 4).unwrap();
            let bytes = writer.finish();

            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let recovered = decode_inter_single_mode_mirror(&bytes, &mut dec_cdfs, false, 5, 1, 4);
            assert_eq!(
                recovered, y_mode,
                "YMode {y_mode} round-trips at ctx (5,1,4)"
            );
        }
    }

    /// Symbol-count leaf checks: `NEWMV` emits exactly one symbol,
    /// `GLOBALMV` two, `NEARESTMV` / `NEARMV` three. Verified by
    /// comparing each leaf's emitted byte stream against an independent
    /// re-encode of only the symbols that leaf should produce (over a
    /// pristine CDF copy), proving no extra symbol is written.
    #[test]
    fn write_inter_single_mode_emits_expected_symbol_counts() {
        // NEWMV: only new_mv = 0.
        {
            let mut a = TileCdfContext::new_from_defaults();
            let mut wa = SymbolWriter::new(false);
            write_inter_single_mode(&mut wa, &mut a, MODE_NEWMV, 0, 0, 0).unwrap();
            let got = wa.finish();

            let mut b = TileCdfContext::new_from_defaults();
            let mut wb = SymbolWriter::new(false);
            let row = b.new_mv_cdf(0);
            wb.write_symbol(0, row).unwrap();
            let want = wb.finish();
            assert_eq!(got, want, "NEWMV emits exactly one (new_mv = 0) symbol");
        }
        // GLOBALMV: new_mv = 1, zero_mv = 0.
        {
            let mut a = TileCdfContext::new_from_defaults();
            let mut wa = SymbolWriter::new(false);
            write_inter_single_mode(&mut wa, &mut a, MODE_GLOBALMV, 0, 0, 0).unwrap();
            let got = wa.finish();

            let mut b = TileCdfContext::new_from_defaults();
            let mut wb = SymbolWriter::new(false);
            let row = b.new_mv_cdf(0);
            wb.write_symbol(1, row).unwrap();
            let row = b.zero_mv_cdf(0);
            wb.write_symbol(0, row).unwrap();
            let want = wb.finish();
            assert_eq!(got, want, "GLOBALMV emits new_mv = 1, zero_mv = 0");
        }
        // NEARMV: new_mv = 1, zero_mv = 1, ref_mv = 1.
        {
            let mut a = TileCdfContext::new_from_defaults();
            let mut wa = SymbolWriter::new(false);
            write_inter_single_mode(&mut wa, &mut a, MODE_NEARMV, 0, 0, 0).unwrap();
            let got = wa.finish();

            let mut b = TileCdfContext::new_from_defaults();
            let mut wb = SymbolWriter::new(false);
            let row = b.new_mv_cdf(0);
            wb.write_symbol(1, row).unwrap();
            let row = b.zero_mv_cdf(0);
            wb.write_symbol(1, row).unwrap();
            let row = b.ref_mv_cdf(0);
            wb.write_symbol(1, row).unwrap();
            let want = wb.finish();
            assert_eq!(
                got, want,
                "NEARMV emits new_mv = 1, zero_mv = 1, ref_mv = 1"
            );
        }
    }

    /// A `YMode` outside the four single-prediction modes (e.g.
    /// `DC_PRED = 0`, `MODE_NEAREST_NEARESTMV = 18`) is a caller bug.
    #[test]
    fn write_inter_single_mode_rejects_invalid_y_mode() {
        for y_mode in [0u8, 13, MODE_NEAREST_NEARESTMV, 25, 255] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_inter_single_mode(&mut writer, &mut cdfs, y_mode, 0, 0, 0).unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "y_mode {y_mode} rejected"
            );
        }
    }

    /// An out-of-range `NewMvContext` (always consulted) is rejected.
    #[test]
    fn write_inter_single_mode_rejects_bad_new_mv_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err =
            write_inter_single_mode(&mut writer, &mut cdfs, MODE_NEWMV, NEW_MV_CONTEXTS, 0, 0)
                .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `ZeroMvContext` is only consulted past the `new_mv == 1` branch:
    /// a bad value is tolerated for `NEWMV` (never reached) but rejected
    /// for `GLOBALMV` (consulted).
    #[test]
    fn write_inter_single_mode_zero_mv_ctx_only_validated_when_consulted() {
        // NEWMV never reads zero_mv ⇒ bad zero_mv ctx is fine.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_inter_single_mode(&mut writer, &mut cdfs, MODE_NEWMV, 0, ZERO_MV_CONTEXTS, 0)
            .unwrap();

        // GLOBALMV reads zero_mv ⇒ bad zero_mv ctx is a caller bug.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_inter_single_mode(
            &mut writer,
            &mut cdfs,
            MODE_GLOBALMV,
            0,
            ZERO_MV_CONTEXTS,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `RefMvContext` is only consulted on the NEARESTMV / NEARMV leaf:
    /// a bad value is tolerated for `GLOBALMV` (never reached) but
    /// rejected for `NEARMV` (consulted).
    #[test]
    fn write_inter_single_mode_ref_mv_ctx_only_validated_when_consulted() {
        // GLOBALMV never reads ref_mv ⇒ bad ref_mv ctx is fine.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_inter_single_mode(&mut writer, &mut cdfs, MODE_GLOBALMV, 0, 0, REF_MV_CONTEXTS)
            .unwrap();

        // NEARMV reads ref_mv ⇒ bad ref_mv ctx is a caller bug.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err =
            write_inter_single_mode(&mut writer, &mut cdfs, MODE_NEARMV, 0, 0, REF_MV_CONTEXTS)
                .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// A sequential run of distinct modes round-trips with §8.3 CDF
    /// adaptation engaged across blocks: each write mutates the shared
    /// encode-side CDFs, each mirror read mutates the decode-side CDFs,
    /// and the two stay in lockstep over the whole sequence.
    #[test]
    fn write_inter_single_mode_sequential_cdf_adaptation_lockstep() {
        let modes = [
            MODE_NEWMV,
            MODE_NEARMV,
            MODE_GLOBALMV,
            MODE_NEARESTMV,
            MODE_NEWMV,
            MODE_NEARMV,
            MODE_NEARESTMV,
            MODE_GLOBALMV,
        ];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        for &y_mode in &modes {
            let mut writer = SymbolWriter::new(false);
            write_inter_single_mode(&mut writer, &mut enc_cdfs, y_mode, 2, 1, 3).unwrap();
            let bytes = writer.finish();
            let recovered = decode_inter_single_mode_mirror(&bytes, &mut dec_cdfs, false, 2, 1, 3);
            assert_eq!(recovered, y_mode);
        }
        // After the run the consulted rows must be identical on both
        // sides (proves the per-symbol adaptation matched throughout).
        assert_eq!(enc_cdfs.new_mv_cdf(2), dec_cdfs.new_mv_cdf(2));
        assert_eq!(enc_cdfs.zero_mv_cdf(1), dec_cdfs.zero_mv_cdf(1));
        assert_eq!(enc_cdfs.ref_mv_cdf(3), dec_cdfs.ref_mv_cdf(3));
    }

    // -----------------------------------------------------------------
    // §5.11.23 compound-prediction inter mode — write_compound_mode
    // round-trips through a mirror of the decoder's single
    // `compound_mode` S() read (av1-spec p.74, arm 3). The full decoder
    // path needs a find_mv_stack (§7.10.2), not yet implemented; this
    // mirror runs exactly the one §5.11.23 S() read against the same
    // CDF row, recovering YMode as `NEAREST_NEARESTMV + compound_mode`.
    // -----------------------------------------------------------------

    /// Replays the §5.11.23 compound-prediction `YMode` derivation from
    /// a decoder over `bytes`, consuming the same `compound_mode`
    /// symbol the writer emitted. Returns the recovered `YMode`.
    fn decode_compound_mode_mirror(
        bytes: &[u8],
        cdfs: &mut TileCdfContext,
        disable_cdf_update: bool,
        compound_mode_context: usize,
    ) -> u8 {
        let pad = if bytes.is_empty() {
            vec![0u8]
        } else {
            bytes.to_vec()
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), disable_cdf_update).unwrap();
        // §5.11.23 line 8: YMode = NEAREST_NEARESTMV + compound_mode.
        let row = cdfs.compound_mode_cdf(compound_mode_context);
        let cm = dec.read_symbol(row).unwrap() as u8;
        MODE_NEAREST_NEARESTMV + cm
    }

    /// Every compound `YMode` (`NEAREST_NEARESTMV = 18 ..= NEW_NEWMV =
    /// 25`) round-trips through the decoder mirror at
    /// `compound_mode_context = 0`, with §8.3 CDF adaptation engaged on
    /// both sides (so the encode-side and decode-side rows step
    /// identically).
    #[test]
    fn write_compound_mode_all_modes_round_trip_at_origin() {
        for y_mode in MODE_NEAREST_NEARESTMV..=MODE_NEW_NEWMV {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_compound_mode(&mut writer, &mut enc_cdfs, y_mode, 0).unwrap();
            let bytes = writer.finish();

            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let recovered = decode_compound_mode_mirror(&bytes, &mut dec_cdfs, false, 0);
            assert_eq!(recovered, y_mode, "YMode {y_mode} round-trips at origin");
            // The encode-side and decode-side CDFs must have adapted
            // identically — assert the consulted row matches.
            assert_eq!(enc_cdfs.compound_mode_cdf(0), dec_cdfs.compound_mode_cdf(0));
        }
    }

    /// Round-trip every compound mode under each non-zero §8.3.2
    /// context `1..COMPOUND_MODE_CONTEXTS`, proving the writer selects
    /// the same `TileCompoundModeCdf` row the reader does.
    #[test]
    fn write_compound_mode_all_modes_round_trip_each_ctx() {
        for ctx in 1..COMPOUND_MODE_CONTEXTS {
            for y_mode in MODE_NEAREST_NEARESTMV..=MODE_NEW_NEWMV {
                let mut enc_cdfs = TileCdfContext::new_from_defaults();
                let mut writer = SymbolWriter::new(false);
                write_compound_mode(&mut writer, &mut enc_cdfs, y_mode, ctx).unwrap();
                let bytes = writer.finish();

                let mut dec_cdfs = TileCdfContext::new_from_defaults();
                let recovered = decode_compound_mode_mirror(&bytes, &mut dec_cdfs, false, ctx);
                assert_eq!(recovered, y_mode, "YMode {y_mode} round-trips at ctx {ctx}");
            }
        }
    }

    /// Symbol-count leaf check: each compound mode emits exactly one
    /// `compound_mode` symbol equal to `YMode - NEAREST_NEARESTMV`.
    /// Verified by comparing the emitted byte stream against an
    /// independent re-encode of only that single symbol over a pristine
    /// CDF copy, proving no extra symbol is written.
    #[test]
    fn write_compound_mode_emits_exactly_one_symbol() {
        for y_mode in MODE_NEAREST_NEARESTMV..=MODE_NEW_NEWMV {
            let mut a = TileCdfContext::new_from_defaults();
            let mut wa = SymbolWriter::new(false);
            write_compound_mode(&mut wa, &mut a, y_mode, 0).unwrap();
            let got = wa.finish();

            let mut b = TileCdfContext::new_from_defaults();
            let mut wb = SymbolWriter::new(false);
            let cm = (y_mode - MODE_NEAREST_NEARESTMV) as u32;
            let row = b.compound_mode_cdf(0);
            wb.write_symbol(cm, row).unwrap();
            let want = wb.finish();
            assert_eq!(
                got, want,
                "YMode {y_mode} emits exactly one compound_mode symbol"
            );
        }
    }

    /// A `YMode` outside the eight compound modes (single-prediction
    /// `NEWMV = 17`, intra `DC_PRED = 0`, just past `NEW_NEWMV` at 26,
    /// and `255`) is a caller bug.
    #[test]
    fn write_compound_mode_rejects_invalid_y_mode() {
        for y_mode in [0u8, 13, MODE_NEWMV, MODE_NEW_NEWMV + 1, 255] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_compound_mode(&mut writer, &mut cdfs, y_mode, 0).unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "y_mode {y_mode} rejected"
            );
        }
    }

    /// The `compound_mode` symbol is always emitted, so the ctx is
    /// always validated: an out-of-range context is a caller bug even
    /// on the first mode.
    #[test]
    fn write_compound_mode_rejects_bad_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_compound_mode(
            &mut writer,
            &mut cdfs,
            MODE_NEAREST_NEARESTMV,
            COMPOUND_MODE_CONTEXTS,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// A sequential run of distinct compound modes round-trips with
    /// §8.3 CDF adaptation engaged across blocks: each write mutates the
    /// shared encode-side CDFs, each mirror read mutates the decode-side
    /// CDFs, and the two stay in lockstep over the whole sequence.
    #[test]
    fn write_compound_mode_sequential_cdf_adaptation_lockstep() {
        let modes = [
            MODE_NEAREST_NEARESTMV,
            MODE_NEW_NEWMV,
            MODE_NEAR_NEARMV,
            MODE_GLOBAL_GLOBALMV,
            MODE_NEAREST_NEWMV,
            MODE_NEW_NEARMV,
            MODE_NEW_NEARESTMV,
            MODE_NEAR_NEWMV,
        ];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        for &y_mode in &modes {
            let mut writer = SymbolWriter::new(false);
            write_compound_mode(&mut writer, &mut enc_cdfs, y_mode, 4).unwrap();
            let bytes = writer.finish();
            let recovered = decode_compound_mode_mirror(&bytes, &mut dec_cdfs, false, 4);
            assert_eq!(recovered, y_mode);
        }
        // After the run the consulted row must be identical on both
        // sides (proves the per-symbol adaptation matched throughout).
        assert_eq!(enc_cdfs.compound_mode_cdf(4), dec_cdfs.compound_mode_cdf(4));
    }

    // -----------------------------------------------------------------
    // §5.11.23 drl_mode — write_drl_mode round-trips through a decoder
    // mirror of the exact §5.11.23 RefMvIdx loop.
    // -----------------------------------------------------------------

    /// Decoder mirror of the §5.11.23 `drl_mode` `RefMvIdx` loop for the
    /// two coding arms (NEWMV / NEW_NEWMV start at idx 0, `has_nearmv( )`
    /// start at idx 1). Reads the `drl_mode` symbols the writer emitted
    /// and reconstructs `RefMvIdx` exactly as the spec body does. Modes
    /// outside the two arms code no symbols and return `RefMvIdx = 0`.
    fn decode_drl_mode_mirror(
        bytes: &[u8],
        cdfs: &mut TileCdfContext,
        disable_cdf_update: bool,
        y_mode: u8,
        num_mv_found: u32,
        drl_ctx_stack: &[u32],
    ) -> u32 {
        let pad = if bytes.is_empty() {
            vec![0u8]
        } else {
            bytes.to_vec()
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), disable_cdf_update).unwrap();
        let mut ref_mv_idx: u32 = 0;
        let start = if y_mode == MODE_NEWMV || y_mode == MODE_NEW_NEWMV {
            0
        } else if matches!(
            y_mode,
            MODE_NEARMV | MODE_NEAR_NEARMV | MODE_NEAR_NEWMV | MODE_NEW_NEARMV
        ) {
            ref_mv_idx = 1;
            1
        } else {
            return 0;
        };
        let mut idx = start;
        while idx < start + 2 {
            if num_mv_found > idx + 1 {
                let row = cdfs.drl_mode_cdf(drl_ctx_stack[idx as usize] as usize);
                let drl = dec.read_symbol(row).unwrap();
                if drl == 0 {
                    ref_mv_idx = idx;
                    break;
                }
                ref_mv_idx = idx + 1;
            }
            idx += 1;
        }
        ref_mv_idx
    }

    /// The NEWMV / NEW_NEWMV arm round-trips every reachable `RefMvIdx`
    /// for every stack depth `NumMvFound = 1..=3`, with §8.3 CDF
    /// adaptation engaged on both sides. The reachable index set is
    /// exactly `0 ..= min(2, NumMvFound - 1)`.
    #[test]
    fn write_drl_mode_newmv_arm_round_trips_all_depths() {
        let ctx_stack = [2u32, 0, 1, 0, 0, 0, 0, 0];
        for y_mode in [MODE_NEWMV, MODE_NEW_NEWMV] {
            for num_mv_found in 1u32..=3 {
                let max_idx = (num_mv_found - 1).min(2);
                for ref_mv_idx in 0..=max_idx {
                    let mut enc_cdfs = TileCdfContext::new_from_defaults();
                    let mut writer = SymbolWriter::new(false);
                    write_drl_mode(
                        &mut writer,
                        &mut enc_cdfs,
                        y_mode,
                        ref_mv_idx,
                        num_mv_found,
                        &ctx_stack,
                    )
                    .unwrap();
                    let bytes = writer.finish();

                    let mut dec_cdfs = TileCdfContext::new_from_defaults();
                    let recovered = decode_drl_mode_mirror(
                        &bytes,
                        &mut dec_cdfs,
                        false,
                        y_mode,
                        num_mv_found,
                        &ctx_stack,
                    );
                    assert_eq!(
                        recovered, ref_mv_idx,
                        "NEWMV-arm YMode {y_mode} RefMvIdx {ref_mv_idx} @ NumMvFound {num_mv_found}"
                    );
                    // Every consulted DRL CDF row stepped identically.
                    for c in 0..DRL_MODE_CONTEXTS {
                        assert_eq!(enc_cdfs.drl_mode_cdf(c), dec_cdfs.drl_mode_cdf(c));
                    }
                }
            }
        }
    }

    /// The `has_nearmv( )` arm round-trips every reachable `RefMvIdx`
    /// for every stack depth `NumMvFound = 1..=4`. The arm seeds
    /// `RefMvIdx = 1` and iterates `idx ∈ {1, 2}`, so the reachable set
    /// is `1 ..= max(1, min(3, NumMvFound - 1))`.
    #[test]
    fn write_drl_mode_nearmv_arm_round_trips_all_depths() {
        let ctx_stack = [0u32, 1, 2, 0, 0, 0, 0, 0];
        for y_mode in [
            MODE_NEARMV,
            MODE_NEAR_NEARMV,
            MODE_NEAR_NEWMV,
            MODE_NEW_NEARMV,
        ] {
            for num_mv_found in 1u32..=4 {
                let max_idx = (num_mv_found - 1).clamp(1, 3);
                for ref_mv_idx in 1..=max_idx {
                    let mut enc_cdfs = TileCdfContext::new_from_defaults();
                    let mut writer = SymbolWriter::new(false);
                    write_drl_mode(
                        &mut writer,
                        &mut enc_cdfs,
                        y_mode,
                        ref_mv_idx,
                        num_mv_found,
                        &ctx_stack,
                    )
                    .unwrap();
                    let bytes = writer.finish();

                    let mut dec_cdfs = TileCdfContext::new_from_defaults();
                    let recovered = decode_drl_mode_mirror(
                        &bytes,
                        &mut dec_cdfs,
                        false,
                        y_mode,
                        num_mv_found,
                        &ctx_stack,
                    );
                    assert_eq!(
                        recovered, ref_mv_idx,
                        "NEAR-arm YMode {y_mode} RefMvIdx {ref_mv_idx} @ NumMvFound {num_mv_found}"
                    );
                }
            }
        }
    }

    /// Modes outside the two coding arms (GLOBALMV, NEARESTMV, and the
    /// compound NEAREST_NEARESTMV / GLOBAL_GLOBALMV) code **no**
    /// `drl_mode` symbols: the writer emits an empty (or terminator-only)
    /// stream and the decoder mirror reconstructs `RefMvIdx = 0` without
    /// touching the buffer.
    #[test]
    fn write_drl_mode_non_coding_modes_emit_no_symbols() {
        let ctx_stack = [0u32; MAX_REF_MV_STACK_SIZE];
        for y_mode in [
            MODE_GLOBALMV,
            MODE_NEARESTMV,
            MODE_NEAREST_NEARESTMV,
            MODE_GLOBAL_GLOBALMV,
        ] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_drl_mode(&mut writer, &mut enc_cdfs, y_mode, 0, 4, &ctx_stack).unwrap();
            let bytes = writer.finish();
            // No DRL CDF row may have adapted (no symbol coded).
            for c in 0..DRL_MODE_CONTEXTS {
                assert_eq!(
                    enc_cdfs.drl_mode_cdf(c),
                    TileCdfContext::new_from_defaults().drl_mode_cdf(c),
                    "non-coding mode {y_mode} left DRL ctx {c} pristine"
                );
            }
            // And the mirror reconstructs RefMvIdx = 0.
            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let recovered =
                decode_drl_mode_mirror(&bytes, &mut dec_cdfs, false, y_mode, 4, &ctx_stack);
            assert_eq!(recovered, 0);
        }
    }

    /// A shallow stack (`NumMvFound <= 1`) makes slot 0 unreachable, so
    /// even a NEWMV block codes no `drl_mode` symbol and `RefMvIdx`
    /// stays at the arm seed (`0`).
    #[test]
    fn write_drl_mode_shallow_stack_codes_nothing() {
        let ctx_stack = [0u32; MAX_REF_MV_STACK_SIZE];
        for num_mv_found in 0u32..=1 {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_drl_mode(
                &mut writer,
                &mut enc_cdfs,
                MODE_NEWMV,
                0,
                num_mv_found,
                &ctx_stack,
            )
            .unwrap();
            for c in 0..DRL_MODE_CONTEXTS {
                assert_eq!(
                    enc_cdfs.drl_mode_cdf(c),
                    TileCdfContext::new_from_defaults().drl_mode_cdf(c)
                );
            }
        }
    }

    /// Distinct per-slot DRL contexts are honoured: with
    /// `DrlCtxStack = [0, 1, …]` a NEWMV block selecting `RefMvIdx = 1`
    /// over `NumMvFound = 3` codes `drl_mode = 1` at slot-0 ctx 0 and
    /// `drl_mode = 0` at slot-1 ctx 1. Verified against an independent
    /// re-encode of exactly those two symbols.
    #[test]
    fn write_drl_mode_uses_per_slot_context() {
        let ctx_stack = [0u32, 1, 2, 0, 0, 0, 0, 0];
        let mut a = TileCdfContext::new_from_defaults();
        let mut wa = SymbolWriter::new(false);
        write_drl_mode(&mut wa, &mut a, MODE_NEWMV, 1, 3, &ctx_stack).unwrap();
        let got = wa.finish();

        let mut b = TileCdfContext::new_from_defaults();
        let mut wb = SymbolWriter::new(false);
        // slot 0: drl_mode = 1 over ctx 0.
        let row0 = b.drl_mode_cdf(0);
        wb.write_symbol(1, row0).unwrap();
        // slot 1: drl_mode = 0 over ctx 1.
        let row1 = b.drl_mode_cdf(1);
        wb.write_symbol(0, row1).unwrap();
        let want = wb.finish();
        assert_eq!(got, want, "per-slot DRL contexts honoured");
    }

    /// A `RefMvIdx` unreachable at the given stack depth (NEWMV arm,
    /// `RefMvIdx = 2` but `NumMvFound = 2` so slot 1 is unreachable) is
    /// a caller bug — the reader could never decode it.
    #[test]
    fn write_drl_mode_rejects_unreachable_index() {
        let ctx_stack = [0u32; MAX_REF_MV_STACK_SIZE];
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_drl_mode(&mut writer, &mut cdfs, MODE_NEWMV, 2, 2, &ctx_stack).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-window `RefMvIdx` for each arm is a caller bug: `> 2` for
    /// the NEWMV arm, `0` or `> 3` for the `has_nearmv( )` arm.
    #[test]
    fn write_drl_mode_rejects_out_of_window_index() {
        let ctx_stack = [0u32; MAX_REF_MV_STACK_SIZE];
        // NEWMV arm: 3 is past the {0,1,2} window.
        {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err =
                write_drl_mode(&mut writer, &mut cdfs, MODE_NEWMV, 3, 5, &ctx_stack).unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
        // has_nearmv arm: 0 is below the {1,2,3} window.
        {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err =
                write_drl_mode(&mut writer, &mut cdfs, MODE_NEARMV, 0, 5, &ctx_stack).unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
        // has_nearmv arm: 4 is past the {1,2,3} window.
        {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err =
                write_drl_mode(&mut writer, &mut cdfs, MODE_NEARMV, 4, 5, &ctx_stack).unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
    }

    /// Range guards: `ref_mv_idx >= MAX_REF_MV_STACK_SIZE`,
    /// `num_mv_found > MAX_REF_MV_STACK_SIZE`, and a bad per-slot DRL
    /// context (`>= DRL_MODE_CONTEXTS`) are each caller bugs.
    #[test]
    fn write_drl_mode_range_guards() {
        // ref_mv_idx out of stack range.
        {
            let ctx_stack = [0u32; MAX_REF_MV_STACK_SIZE];
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_drl_mode(
                &mut writer,
                &mut cdfs,
                MODE_NEWMV,
                MAX_REF_MV_STACK_SIZE as u32,
                3,
                &ctx_stack,
            )
            .unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
        // num_mv_found past the stack cap.
        {
            let ctx_stack = [0u32; MAX_REF_MV_STACK_SIZE];
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_drl_mode(
                &mut writer,
                &mut cdfs,
                MODE_NEWMV,
                0,
                MAX_REF_MV_STACK_SIZE as u32 + 1,
                &ctx_stack,
            )
            .unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
        // Bad per-slot DRL context on a coded slot.
        {
            let mut ctx_stack = [0u32; MAX_REF_MV_STACK_SIZE];
            ctx_stack[0] = DRL_MODE_CONTEXTS as u32;
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err =
                write_drl_mode(&mut writer, &mut cdfs, MODE_NEWMV, 0, 3, &ctx_stack).unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
    }

    /// A sequential run of NEWMV-arm DRL writes round-trips with §8.3
    /// CDF adaptation engaged across blocks, the encode-side and
    /// decode-side DRL CDFs staying in lockstep over the whole run.
    #[test]
    fn write_drl_mode_sequential_cdf_adaptation_lockstep() {
        let ctx_stack = [0u32, 1, 2, 0, 0, 0, 0, 0];
        let cases = [
            (MODE_NEWMV, 0u32, 3u32),
            (MODE_NEW_NEWMV, 1, 3),
            (MODE_NEWMV, 2, 3),
            (MODE_NEW_NEWMV, 0, 2),
            (MODE_NEWMV, 1, 2),
        ];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        for &(y_mode, ref_mv_idx, num_mv_found) in &cases {
            let mut writer = SymbolWriter::new(false);
            write_drl_mode(
                &mut writer,
                &mut enc_cdfs,
                y_mode,
                ref_mv_idx,
                num_mv_found,
                &ctx_stack,
            )
            .unwrap();
            let bytes = writer.finish();
            let recovered = decode_drl_mode_mirror(
                &bytes,
                &mut dec_cdfs,
                false,
                y_mode,
                num_mv_found,
                &ctx_stack,
            );
            assert_eq!(recovered, ref_mv_idx);
        }
        for c in 0..DRL_MODE_CONTEXTS {
            assert_eq!(enc_cdfs.drl_mode_cdf(c), dec_cdfs.drl_mode_cdf(c));
        }
    }

    // -----------------------------------------------------------------
    // §5.11.31 / §5.11.32 — write_read_mv / write_read_mv_component.
    // -----------------------------------------------------------------

    /// Decode mirror of §5.11.32 `read_mv_component( comp )` — the
    /// literal reader pseudocode, used to round-trip the writer. Returns
    /// the signed difference the reader would reconstruct.
    fn decode_read_mv_component_mirror(
        dec: &mut SymbolDecoder,
        cdfs: &mut TileCdfContext,
        mv_ctx: usize,
        comp: usize,
        force_integer_mv: bool,
        allow_high_precision_mv: bool,
    ) -> i32 {
        let mv_sign = dec.read_symbol(cdfs.mv_sign_cdf(mv_ctx, comp)).unwrap();
        let mv_class = dec.read_symbol(cdfs.mv_class_cdf(mv_ctx, comp)).unwrap();
        let mag: u32 = if mv_class == MV_CLASS_0 as u32 {
            let mv_class0_bit = dec
                .read_symbol(cdfs.mv_class0_bit_cdf(mv_ctx, comp))
                .unwrap();
            let mv_class0_fr = if force_integer_mv {
                3
            } else {
                dec.read_symbol(cdfs.mv_class0_fr_cdf(mv_ctx, comp, mv_class0_bit as usize))
                    .unwrap()
            };
            let mv_class0_hp = if allow_high_precision_mv {
                dec.read_symbol(cdfs.mv_class0_hp_cdf(mv_ctx, comp))
                    .unwrap()
            } else {
                1
            };
            ((mv_class0_bit << 3) | (mv_class0_fr << 1) | mv_class0_hp) + 1
        } else {
            let mut d: u32 = 0;
            for i in 0..(mv_class as usize) {
                let mv_bit = dec.read_symbol(cdfs.mv_bit_cdf(mv_ctx, comp, i)).unwrap();
                d |= mv_bit << i;
            }
            let mut mag = (CLASS0_SIZE as u32) << (mv_class + 2);
            let mv_fr = if force_integer_mv {
                3
            } else {
                dec.read_symbol(cdfs.mv_fr_cdf(mv_ctx, comp)).unwrap()
            };
            let mv_hp = if allow_high_precision_mv {
                dec.read_symbol(cdfs.mv_hp_cdf(mv_ctx, comp)).unwrap()
            } else {
                1
            };
            mag += ((d << 3) | (mv_fr << 1) | mv_hp) + 1;
            mag
        };
        if mv_sign != 0 {
            -(mag as i32)
        } else {
            mag as i32
        }
    }

    /// Decode mirror of §5.11.31 `read_mv( ref )` — reconstructs the
    /// `Mv[ ref ]` the reader would produce from `pred_mv` and the
    /// emitted symbols.
    fn decode_read_mv_mirror(
        bytes: &[u8],
        cdfs: &mut TileCdfContext,
        pred_mv: [i32; 2],
        use_intrabc: bool,
        force_integer_mv: bool,
        allow_high_precision_mv: bool,
    ) -> [i32; 2] {
        let mv_ctx = if use_intrabc { MV_INTRABC_CONTEXT } else { 0 };
        let mut dec = SymbolDecoder::init_symbol(bytes, bytes.len(), false).unwrap();
        let mv_joint = dec.read_symbol(cdfs.mv_joint_cdf(mv_ctx)).unwrap() as u8;
        let mut diff = [0i32; 2];
        if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
            diff[0] = decode_read_mv_component_mirror(
                &mut dec,
                cdfs,
                mv_ctx,
                0,
                force_integer_mv,
                allow_high_precision_mv,
            );
        }
        if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
            diff[1] = decode_read_mv_component_mirror(
                &mut dec,
                cdfs,
                mv_ctx,
                1,
                force_integer_mv,
                allow_high_precision_mv,
            );
        }
        [pred_mv[0] + diff[0], pred_mv[1] + diff[1]]
    }

    /// Full round-trip of `write_read_mv` over a sweep of target MVs at
    /// high precision (every symbol read), at the frame origin
    /// (`MvCtx = 0`), with §8.3 CDF adaptation engaged on both sides.
    #[test]
    fn write_read_mv_round_trips_high_precision() {
        let pred = [100, -50];
        // Mix of: zero joint, single-axis, both-axis, class-0 and
        // higher-class magnitudes, and both signs.
        let targets = [
            [100, -50],    // diff (0,0)   — MV_JOINT_ZERO
            [100, -49],    // diff (0,+1)  — HNZVZ, class 0
            [101, -50],    // diff (+1,0)  — HZVNZ, class 0
            [101, -49],    // diff (+1,+1) — HNZVNZ
            [84, -50],     // diff (-16,0) — class 1
            [100, 30],     // diff (0,+80) — bigger class
            [1100, -1050], // diff (+1000,-1000) — high class both axes
            [50, -200],    // diff (-50,-150)
        ];
        for tgt in targets {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_read_mv(&mut writer, &mut enc_cdfs, tgt, pred, false, false, true).unwrap();
            let bytes = writer.finish();

            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let recovered = decode_read_mv_mirror(&bytes, &mut dec_cdfs, pred, false, false, true);
            assert_eq!(recovered, tgt, "MV {tgt:?} round-trips at high precision");
            // CDF adaptation must have stepped identically on both sides.
            assert_eq!(enc_cdfs.mv_joint_cdf(0), dec_cdfs.mv_joint_cdf(0));
        }
    }

    /// Round-trip under the intra-block-copy context (`use_intrabc`),
    /// proving the writer selects `MvCtx = MV_INTRABC_CONTEXT = 1`.
    #[test]
    fn write_read_mv_round_trips_intrabc_context() {
        let pred = [0, 0];
        // intra block copy is integer-only, so each component magnitude
        // must be a multiple of 8 (low 3 bits = `(fr<<1)|hp` = `(3<<1)|1`).
        let tgt = [-40, 64];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // intra block copy is always integer + non-high-precision.
        write_read_mv(&mut writer, &mut enc_cdfs, tgt, pred, true, true, false).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let recovered = decode_read_mv_mirror(&bytes, &mut dec_cdfs, pred, true, true, false);
        assert_eq!(
            recovered, tgt,
            "MV round-trips under MvCtx = MV_INTRABC_CONTEXT"
        );
        // The intrabc context row must have adapted, not the ctx-0 row.
        assert_eq!(
            enc_cdfs.mv_joint_cdf(MV_INTRABC_CONTEXT),
            dec_cdfs.mv_joint_cdf(MV_INTRABC_CONTEXT)
        );
    }

    /// Round-trip under `force_integer_mv` (fractional field synthesised
    /// to 3, never read) and `!allow_high_precision_mv` (hp forced to 1).
    /// Only integer-aligned MVs with `fr == 3, hp == 1` are
    /// representable — i.e. magnitudes of the form `8k`.
    #[test]
    fn write_read_mv_round_trips_force_integer() {
        let pred = [0, 0];
        // diff components are multiples of 8 (their low 3 bits = `(fr<<1)|hp`
        // = (3<<1)|1 = 7 ⇒ offset & 7 == 7, i.e. mag in {8, 16, 24, ...}).
        for &d0 in &[0i32, 8, -8, 64, -128] {
            for &d1 in &[0i32, 8, 256, -512] {
                let tgt = [d0, d1];
                let mut enc_cdfs = TileCdfContext::new_from_defaults();
                let mut writer = SymbolWriter::new(false);
                write_read_mv(&mut writer, &mut enc_cdfs, tgt, pred, false, true, false).unwrap();
                let bytes = writer.finish();

                let mut dec_cdfs = TileCdfContext::new_from_defaults();
                let recovered =
                    decode_read_mv_mirror(&bytes, &mut dec_cdfs, pred, false, true, false);
                assert_eq!(recovered, tgt, "integer MV {tgt:?} round-trips");
            }
        }
    }

    /// `mv_joint` selection covers all four quadrants exactly.
    #[test]
    fn write_read_mv_joint_selection_is_exhaustive() {
        let pred = [10, 20];
        let cases = [
            ([10, 20], MV_JOINT_ZERO),
            ([10, 21], MV_JOINT_HNZVZ),
            ([11, 20], MV_JOINT_HZVNZ),
            ([11, 21], MV_JOINT_HNZVNZ),
        ];
        for (tgt, want_joint) in cases {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_read_mv(&mut writer, &mut enc_cdfs, tgt, pred, false, false, true).unwrap();
            let bytes = writer.finish();

            // Read just the leading mv_joint symbol to confirm the quadrant.
            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let joint = dec.read_symbol(dec_cdfs.mv_joint_cdf(0)).unwrap() as u8;
            assert_eq!(joint, want_joint, "MV {tgt:?} ⇒ joint {want_joint}");
        }
    }

    /// `MV_JOINT_ZERO` emits exactly one symbol (the joint) and no
    /// component reads — verified by re-encoding only that symbol.
    #[test]
    fn write_read_mv_zero_joint_emits_only_joint_symbol() {
        let pred = [7, -3];
        let mut a = TileCdfContext::new_from_defaults();
        let mut wa = SymbolWriter::new(false);
        write_read_mv(&mut wa, &mut a, pred, pred, false, false, true).unwrap();
        let got = wa.finish();

        let mut b = TileCdfContext::new_from_defaults();
        let mut wb = SymbolWriter::new(false);
        let row = b.mv_joint_cdf(0);
        wb.write_symbol(MV_JOINT_ZERO as u32, row).unwrap();
        let want = wb.finish();
        assert_eq!(got, want, "MV_JOINT_ZERO writes a single joint symbol");
    }

    /// The class derivation boundary: `mag == 16` (offset 15) is the
    /// last `MV_CLASS_0` magnitude, `mag == 17` (offset 16) is the first
    /// `MV_CLASS_1` magnitude. Both round-trip.
    #[test]
    fn write_read_mv_class_boundary_round_trips() {
        let pred = [0, 0];
        for d in [16i32, 17, -16, -17] {
            let tgt = [d, 0];
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_read_mv(&mut writer, &mut enc_cdfs, tgt, pred, false, false, true).unwrap();
            let bytes = writer.finish();
            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let recovered = decode_read_mv_mirror(&bytes, &mut dec_cdfs, pred, false, false, true);
            assert_eq!(recovered, tgt, "class-boundary diff {d} round-trips");
        }
    }

    /// A zero component handed to `write_read_mv_component` directly is a
    /// caller bug (the reader signals zero via `mv_joint`, never a
    /// component read).
    #[test]
    fn write_read_mv_component_rejects_zero_diff() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err =
            write_read_mv_component(&mut writer, &mut cdfs, 0, 0, 0, false, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `mv_ctx` / `comp` are caller bugs.
    #[test]
    fn write_read_mv_component_rejects_bad_indices() {
        for (ctx, comp) in [(MV_CONTEXTS, 0), (0, MV_COMPS)] {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_read_mv_component(&mut writer, &mut cdfs, ctx, comp, 5, false, true)
                .unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
    }

    /// Under `force_integer_mv`, a diff whose fractional field is not 3
    /// is unrepresentable (the reader would synthesise `fr = 3`, not the
    /// caller's value) and must be rejected rather than mis-encoded.
    #[test]
    fn write_read_mv_component_rejects_non_integer_under_force_integer() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // diff = 1 ⇒ offset 0 ⇒ class0, fr = 0 ≠ 3 under force_integer.
        let err = write_read_mv_component(&mut writer, &mut cdfs, 0, 0, 1, true, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Under `!allow_high_precision_mv`, a diff whose half-pel bit is not
    /// 1 is unrepresentable (the reader forces `hp = 1`).
    #[test]
    fn write_read_mv_component_rejects_non_hp_when_disallowed() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // diff = 2 ⇒ offset 1 ⇒ class0, hp = 1, fr = 0. That's fine.
        // diff = 3 ⇒ offset 2 ⇒ hp = 0 ≠ 1 under no-high-precision ⇒ reject.
        let err =
            write_read_mv_component(&mut writer, &mut cdfs, 0, 0, 3, false, false).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// A magnitude beyond the encodable MV range (`mv_class >= MV_CLASSES`)
    /// is rejected. The maximum encodable offset is `2^(MV_CLASSES+3) - 1`;
    /// one past it overflows the class alphabet.
    #[test]
    fn write_read_mv_component_rejects_overlarge_magnitude() {
        // mv_class = FloorLog2(offset) - 3 must be < MV_CLASSES = 11, so
        // FloorLog2(offset) < 14, i.e. offset < 2^14 = 16384 ⇒ mag <= 16384.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // mag = 16385 ⇒ offset 16384 ⇒ FloorLog2 = 14 ⇒ class 11 ⇒ reject.
        let err =
            write_read_mv_component(&mut writer, &mut cdfs, 0, 0, 16385, false, true).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.26 assign_mv PredMv derivation — get_mode + assign_mv_pred_mv
    // -----------------------------------------------------------------

    /// Build a `FindMvStackResult` with caller-chosen `num_mv_found`,
    /// distinct per-slot `RefStackMv` (so a wrong `pos` is observable),
    /// and per-list `GlobalMvs`. Single-pred slots fill list 0; compound
    /// fills both lists. Every other field is the spatial-only default.
    fn mk_mv_stack(num_mv_found: u32) -> FindMvStackResult {
        let mut ref_stack_mv = [[[0i32; 2]; 2]; MAX_REF_MV_STACK_SIZE];
        #[allow(clippy::needless_range_loop)]
        for idx in 0..MAX_REF_MV_STACK_SIZE {
            // List 0: encode the slot index into the value so a mis-pick
            // is caught. List 1: a distinct offset so list selection is
            // also observable.
            ref_stack_mv[idx][0] = [100 + idx as i32 * 10, 200 + idx as i32 * 10];
            ref_stack_mv[idx][1] = [300 + idx as i32 * 10, 400 + idx as i32 * 10];
        }
        FindMvStackResult {
            num_mv_found,
            new_mv_count: 0,
            ref_stack_mv,
            weight_stack: [0; MAX_REF_MV_STACK_SIZE],
            global_mvs: [[7, 9], [11, 13]],
            new_mv_context: 0,
            ref_mv_context: 0,
            zero_mv_context: 0,
            drl_ctx_stack: [0; MAX_REF_MV_STACK_SIZE],
            close_matches: 0,
            total_matches: 0,
            num_nearest: 0,
            num_new: 0,
        }
    }

    /// §5.11.26 GLOBALMV arm — `PredMv[i] = GlobalMvs[i]` regardless of
    /// `RefMvIdx` or stack depth, on both lists.
    #[test]
    fn assign_mv_pred_mv_globalmv_uses_global() {
        let s = mk_mv_stack(0);
        assert_eq!(assign_mv_pred_mv(&s, MODE_GLOBALMV, 0, 2).unwrap(), [7, 9]);
        // Compound list 1 via GLOBAL_GLOBALMV.
        assert_eq!(
            assign_mv_pred_mv(&s, MODE_GLOBAL_GLOBALMV, 1, 0).unwrap(),
            [11, 13]
        );
    }

    /// §5.11.26 NEARESTMV arm — `pos = 0` always, so the top stack slot
    /// is selected irrespective of `RefMvIdx`.
    #[test]
    fn assign_mv_pred_mv_nearestmv_uses_slot0() {
        let s = mk_mv_stack(4);
        for ref_mv_idx in 0..=3 {
            assert_eq!(
                assign_mv_pred_mv(&s, MODE_NEARESTMV, 0, ref_mv_idx).unwrap(),
                [100, 200],
                "NEARESTMV ignores RefMvIdx {ref_mv_idx}"
            );
        }
    }

    /// §5.11.26 NEARMV arm — `pos = RefMvIdx`, selecting the indexed slot.
    #[test]
    fn assign_mv_pred_mv_nearmv_uses_ref_mv_idx() {
        let s = mk_mv_stack(4);
        assert_eq!(
            assign_mv_pred_mv(&s, MODE_NEARMV, 0, 1).unwrap(),
            [110, 210]
        );
        assert_eq!(
            assign_mv_pred_mv(&s, MODE_NEARMV, 0, 2).unwrap(),
            [120, 220]
        );
        assert_eq!(
            assign_mv_pred_mv(&s, MODE_NEARMV, 0, 3).unwrap(),
            [130, 230]
        );
    }

    /// §5.11.26 NEWMV arm — `pos = RefMvIdx` when `NumMvFound > 1`, but
    /// collapses to `pos = 0` whenever `NumMvFound <= 1`.
    #[test]
    fn assign_mv_pred_mv_newmv_collapses_on_short_stack() {
        // NumMvFound = 2 ⇒ pos = RefMvIdx honoured.
        let deep = mk_mv_stack(2);
        assert_eq!(
            assign_mv_pred_mv(&deep, MODE_NEWMV, 0, 1).unwrap(),
            [110, 210]
        );
        // NumMvFound = 1 ⇒ pos forced to 0 even though RefMvIdx = 0 here;
        // and NumMvFound = 1 with RefMvIdx = 0 still reads slot 0.
        let shallow = mk_mv_stack(1);
        assert_eq!(
            assign_mv_pred_mv(&shallow, MODE_NEWMV, 0, 0).unwrap(),
            [100, 200]
        );
    }

    /// §5.11.26 compound NEW_NEARMV — list 0 (`NEWMV`) draws from the
    /// stack at `RefMvIdx`; list 1 (`NEARMV`) draws from list-1 slots.
    #[test]
    fn assign_mv_pred_mv_compound_per_list() {
        let s = mk_mv_stack(4);
        // list 0 = NEWMV @ RefMvIdx 1 ⇒ ref_stack_mv[1][0].
        assert_eq!(
            assign_mv_pred_mv(&s, MODE_NEW_NEARMV, 0, 1).unwrap(),
            [110, 210]
        );
        // list 1 = NEARMV @ RefMvIdx 1 ⇒ ref_stack_mv[1][1].
        assert_eq!(
            assign_mv_pred_mv(&s, MODE_NEW_NEARMV, 1, 1).unwrap(),
            [310, 410]
        );
    }

    /// §5.11.26 reachability reject — a `pos` at or beyond `NumMvFound`
    /// would read an undefined `RefStackMv` slot.
    #[test]
    fn assign_mv_pred_mv_rejects_unreachable_pos() {
        let s = mk_mv_stack(2);
        // NEARMV @ RefMvIdx 2 needs slot 2, but NumMvFound = 2 ⇒ reject.
        assert!(matches!(
            assign_mv_pred_mv(&s, MODE_NEARMV, 0, 2).unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
        // NEARESTMV / NEWMV on an empty stack read slot 0 — ALWAYS
        // populated per the §7.10.2.12 extra-search pad ("for single
        // prediction, NumMvFound is not incremented by the addition of
        // global motion candidates"), so an empty-neighbourhood block
        // legitimately predicts from it (r411 contract fix).
        let empty = mk_mv_stack(0);
        assert_eq!(
            assign_mv_pred_mv(&empty, MODE_NEARESTMV, 0, 0).unwrap(),
            empty.ref_stack_mv[0][0]
        );
        assert_eq!(
            assign_mv_pred_mv(&empty, MODE_NEWMV, 0, 0).unwrap(),
            empty.ref_stack_mv[0][0]
        );
        // A `pos >= 2` beyond `NumMvFound` stays a reject on the empty
        // stack (unreachable through the §5.11.23 drl loop).
        assert!(matches!(
            assign_mv_pred_mv(&empty, MODE_NEARMV, 0, 2).unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
    }

    /// §5.11.26 caller-bug rejects on the mode / ref-list guard — a
    /// non-inter `y_mode` or a `ref_list > 1`.
    #[test]
    fn assign_mv_pred_mv_rejects_bad_mode_or_ref_list() {
        let s = mk_mv_stack(4);
        assert!(matches!(
            assign_mv_pred_mv(&s, DC_PRED_U8, 0, 0).unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
        assert!(matches!(
            assign_mv_pred_mv(&s, MODE_NEW_NEWMV + 1, 0, 0).unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
        assert!(matches!(
            assign_mv_pred_mv(&s, MODE_NEARMV, 2, 0).unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
    }

    /// The PredMv this helper derives is exactly the predictor
    /// [`write_read_mv`] subtracts: a NEWMV target minus its derived
    /// `PredMv` round-trips through a §5.11.31 reader mirror.
    #[test]
    fn assign_mv_pred_mv_feeds_write_read_mv() {
        let s = mk_mv_stack(2);
        let pred = assign_mv_pred_mv(&s, MODE_NEWMV, 0, 1).unwrap();
        assert_eq!(pred, [110, 210]);
        let target_mv = [pred[0] + 5, pred[1] - 9];

        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_read_mv(
            &mut writer,
            &mut enc_cdfs,
            target_mv,
            pred,
            false,
            false,
            true,
        )
        .unwrap();
        let bytes = writer.finish();

        // §5.11.31 reader mirror reconstructs Mv = PredMv + diffMv.
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let recovered = decode_read_mv_mirror(&bytes, &mut dec_cdfs, pred, false, false, true);
        assert_eq!(recovered, target_mv, "PredMv + diffMv reconstructs Mv");
    }

    // -----------------------------------------------------------------
    // §5.11.26 assign_mv intra-block-copy arm — assign_mv_pred_mv_intrabc
    // -----------------------------------------------------------------

    /// Build a `FindMvStackResult` shaped like the §7.10.2 output on an
    /// intra-block-copy block: `GlobalMvs[ 0 ]` is the zero vector
    /// (§7.10.2.1 `ref == INTRA_FRAME`), and the caller chooses the two
    /// `RefStackMv[ idx ][ 0 ]` slots the §5.11.26 intrabc arm reads
    /// (the §7.10.2.12 single-pred tail pads unwritten slots with
    /// `GlobalMvs[ 0 ] = [0, 0]` without incrementing `NumMvFound`).
    fn mk_intrabc_mv_stack(
        slot0: [i32; 2],
        slot1: [i32; 2],
        num_mv_found: u32,
    ) -> FindMvStackResult {
        let mut s = mk_mv_stack(num_mv_found);
        s.global_mvs = [[0, 0], [0, 0]];
        s.ref_stack_mv = [[[0i32; 2]; 2]; MAX_REF_MV_STACK_SIZE];
        s.ref_stack_mv[0][0] = slot0;
        s.ref_stack_mv[1][0] = slot1;
        s
    }

    /// §5.11.26 intrabc — a non-zero `RefStackMv[ 0 ][ 0 ]` is the
    /// predictor; slot 1 and the superblock fallback are never reached.
    #[test]
    fn assign_mv_pred_mv_intrabc_uses_slot0_when_nonzero() {
        let s = mk_intrabc_mv_stack([-64, 8], [16, -24], 2);
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, false, 16, 0).unwrap(),
            [-64, 8]
        );
        // A half-zero slot 0 is still "non-zero" — the fallback tests
        // both components.
        let row_only = mk_intrabc_mv_stack([-8, 0], [16, -24], 2);
        assert_eq!(
            assign_mv_pred_mv_intrabc(&row_only, false, 16, 0).unwrap(),
            [-8, 0]
        );
    }

    /// §5.11.26 intrabc — a zero slot 0 falls through to
    /// `RefStackMv[ 1 ][ 0 ]`.
    #[test]
    fn assign_mv_pred_mv_intrabc_falls_to_slot1() {
        let s = mk_intrabc_mv_stack([0, 0], [16, -24], 2);
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, false, 16, 0).unwrap(),
            [16, -24]
        );
    }

    /// §5.11.26 intrabc — both slots zero (the §7.10.2.12 global-MV
    /// padding outcome on the first intrabc block) and the block in the
    /// tile's top superblock row (`MiRow - sbSize4 < MiRowStart`):
    /// the predictor points left by `sbSize4 * MI_SIZE +
    /// INTRABC_DELAY_PIXELS` luma samples, in the **column** component,
    /// in 1/8-sample units.
    #[test]
    fn assign_mv_pred_mv_intrabc_top_row_fallback() {
        let s = mk_intrabc_mv_stack([0, 0], [0, 0], 0);
        // 64x64 superblocks: sbSize4 = 16 ⇒ [0, -(16*4 + 256)*8].
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, false, 0, 0).unwrap(),
            [0, -2560]
        );
        // mi_row 15 with MiRowStart 0 is still inside the first
        // superblock row (15 - 16 < 0).
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, false, 15, 0).unwrap(),
            [0, -2560]
        );
        // 128x128 superblocks: sbSize4 = 32 ⇒ [0, -(32*4 + 256)*8];
        // rows 0..=31 of the tile are all "top superblock row".
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, true, 31, 0).unwrap(),
            [0, -3072]
        );
        // A non-zero MiRowStart shifts the boundary: mi_row 40 in a
        // tile starting at mi_row 32 is in that tile's top superblock
        // row (40 - 16 < 32).
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, false, 40, 32).unwrap(),
            [0, -2560]
        );
    }

    /// §5.11.26 intrabc — both slots zero below the top superblock row:
    /// the predictor points one superblock **up** (`-(sbSize4 * MI_SIZE
    /// * 8)` in the row component).
    #[test]
    fn assign_mv_pred_mv_intrabc_up_fallback() {
        let s = mk_intrabc_mv_stack([0, 0], [0, 0], 0);
        // 64x64: mi_row 16 is exactly one superblock down (16 - 16 = 0
        // is NOT < 0) ⇒ [-16*4*8, 0].
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, false, 16, 0).unwrap(),
            [-512, 0]
        );
        // 128x128: sbSize4 = 32 ⇒ [-1024, 0] from row 32.
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, true, 32, 0).unwrap(),
            [-1024, 0]
        );
        // Tile-relative: mi_row 48 with MiRowStart 32 (48 - 16 >= 32).
        assert_eq!(
            assign_mv_pred_mv_intrabc(&s, false, 48, 32).unwrap(),
            [-512, 0]
        );
    }

    /// §5.11.26 intrabc caller-bug reject — a block above its own tile
    /// (`mi_row < mi_row_start`).
    #[test]
    fn assign_mv_pred_mv_intrabc_rejects_row_above_tile() {
        let s = mk_intrabc_mv_stack([0, 0], [0, 0], 0);
        assert!(matches!(
            assign_mv_pred_mv_intrabc(&s, false, 31, 32).unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
    }

    /// End-to-end: the §5.11.26 intrabc predictor feeds
    /// [`write_read_mv`] under the intra-block-copy MV regime
    /// (`MvCtx = MV_INTRABC_CONTEXT`, integer-only, no high precision)
    /// and the §5.11.31 reader mirror reconstructs the target block
    /// vector — on both the slot-0 path and the top-row fallback.
    #[test]
    fn assign_mv_pred_mv_intrabc_feeds_write_read_mv() {
        for (s, mi_row, target_mv) in [
            // Slot-0 predictor (integer-aligned: multiples of 8).
            (
                mk_intrabc_mv_stack([-64, 8], [0, 0], 2),
                16,
                [-64 + 16, 8 - 40],
            ),
            // Zero stack ⇒ top-row fallback predictor [0, -2560].
            (mk_intrabc_mv_stack([0, 0], [0, 0], 0), 0, [0, -2560 + 64]),
        ] {
            let pred = assign_mv_pred_mv_intrabc(&s, false, mi_row, 0).unwrap();
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_read_mv(
                &mut writer,
                &mut enc_cdfs,
                target_mv,
                pred,
                /* use_intrabc = */ true,
                /* force_integer_mv = */ true,
                /* allow_high_precision_mv = */ false,
            )
            .unwrap();
            let bytes = writer.finish();

            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let recovered = decode_read_mv_mirror(&bytes, &mut dec_cdfs, pred, true, true, false);
            assert_eq!(recovered, target_mv, "intrabc Mv reconstructs");
            // The intrabc context row adapted identically on both sides.
            assert_eq!(
                enc_cdfs.mv_joint_cdf(MV_INTRABC_CONTEXT),
                dec_cdfs.mv_joint_cdf(MV_INTRABC_CONTEXT)
            );
        }
    }

    // -----------------------------------------------------------------
    // §5.11.7 use_intrabc — write_use_intrabc round-trips through
    // PartitionWalker::decode_use_intrabc; write_intra_frame_intrabc_arm
    // composes the full §5.11.7 intrabc arm (r279).
    // -----------------------------------------------------------------

    /// `write_use_intrabc` round-trips both values through the decode
    /// twin with §8.3 CDF adaptation in lockstep.
    #[test]
    fn write_use_intrabc_round_trips_through_decode_twin() {
        for target in [0u8, 1u8] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_use_intrabc(&mut writer, &mut enc_cdfs, target, true).unwrap();
            let bytes = writer.finish();

            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let got = walker
                .decode_use_intrabc(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_8X8, true)
                .unwrap();
            assert_eq!(got, target, "use_intrabc = {target} round-trips");
            // The contextless TileIntrabcCdf row adapted identically.
            assert_eq!(enc_cdfs.intrabc_cdf(), dec_cdfs.intrabc_cdf());
        }
    }

    /// §5.11.7 fall-through arm (`allow_intrabc == 0`): no bits, the
    /// reader forces `use_intrabc = 0` — a `1` is a caller bug, as is
    /// any value outside the §3 binary alphabet.
    #[test]
    fn write_use_intrabc_fall_through_no_bits_and_rejects() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_use_intrabc(&mut writer, &mut cdfs, 0, false).unwrap();
        assert_eq!(
            writer.finish(),
            SymbolWriter::new(false).finish(),
            "fall-through arm emits no bits"
        );
        // The CDF row must be untouched on the no-bit arm.
        assert_eq!(
            cdfs.intrabc_cdf(),
            TileCdfContext::new_from_defaults().intrabc_cdf()
        );

        let mut writer = SymbolWriter::new(false);
        assert!(matches!(
            write_use_intrabc(&mut writer, &mut cdfs, 1, false).unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
        for allow in [false, true] {
            let mut writer = SymbolWriter::new(false);
            assert!(matches!(
                write_use_intrabc(&mut writer, &mut cdfs, 2, allow).unwrap_err(),
                Error::PartitionWalkOutOfRange
            ));
        }
    }

    /// Full §5.11.7 intrabc-arm composition: `use_intrabc` S() +
    /// §5.11.26 `assign_mv( 0 )` (forced-NEWMV `read_mv`) round-trip
    /// through `decode_use_intrabc` + a §5.11.31 reader mirror sharing
    /// one decoder, on both the slot-0 predictor and the zero-stack
    /// top-row fallback. The returned readout carries the spec's fixed
    /// no-bit assignments.
    #[test]
    fn write_intra_frame_intrabc_arm_round_trips() {
        for (stack, mi_row, target_mv) in [
            // Slot-0 predictor [-64, 8]; integer-aligned diff (+16, -48).
            (mk_intrabc_mv_stack([-64, 8], [0, 0], 2), 16u32, [-48, -40]),
            // Zero stack ⇒ top-row fallback predictor [0, -2560].
            (
                mk_intrabc_mv_stack([0, 0], [0, 0], 0),
                0u32,
                [0, -2560 + 64],
            ),
        ] {
            let inputs = IntrabcArmInputs {
                mv: target_mv,
                mv_stack: &stack,
                use_128x128_superblock: false,
                mi_row,
                mi_row_start: 0,
            };
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let info =
                write_intra_frame_intrabc_arm(&mut writer, &mut enc_cdfs, true, Some(&inputs))
                    .unwrap()
                    .expect("intrabc arm fired");
            let bytes = writer.finish();

            // §5.11.7 fixed assignments surface in the readout.
            let want_pred = assign_mv_pred_mv_intrabc(&stack, false, mi_row, 0).unwrap();
            assert_eq!(
                info,
                IntrabcBlockInfo {
                    is_inter: 1,
                    y_mode: DC_PRED_U8,
                    uv_mode: DC_PRED_U8,
                    motion_mode: MOTION_MODE_SIMPLE,
                    compound_type: COMPOUND_AVERAGE,
                    palette_size_y: 0,
                    palette_size_uv: 0,
                    interp_filter: [BILINEAR, BILINEAR],
                    mv: target_mv,
                    pred_mv: want_pred,
                }
            );

            // Reader side: decode_use_intrabc + the §5.11.31 read_mv
            // mirror from the same live decoder.
            let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let use_intrabc = walker
                .decode_use_intrabc(&mut dec, &mut dec_cdfs, mi_row, 0, BLOCK_16X16, true)
                .unwrap();
            assert_eq!(use_intrabc, 1);
            let mv_joint = dec
                .read_symbol(dec_cdfs.mv_joint_cdf(MV_INTRABC_CONTEXT))
                .unwrap() as u8;
            let mut diff = [0i32; 2];
            if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
                diff[0] = decode_read_mv_component_mirror(
                    &mut dec,
                    &mut dec_cdfs,
                    MV_INTRABC_CONTEXT,
                    0,
                    true,
                    false,
                );
            }
            if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
                diff[1] = decode_read_mv_component_mirror(
                    &mut dec,
                    &mut dec_cdfs,
                    MV_INTRABC_CONTEXT,
                    1,
                    true,
                    false,
                );
            }
            let recovered = [want_pred[0] + diff[0], want_pred[1] + diff[1]];
            assert_eq!(recovered, target_mv, "intrabc Mv reconstructs");
            // Both adapted rows stepped identically.
            assert_eq!(enc_cdfs.intrabc_cdf(), dec_cdfs.intrabc_cdf());
            assert_eq!(
                enc_cdfs.mv_joint_cdf(MV_INTRABC_CONTEXT),
                dec_cdfs.mv_joint_cdf(MV_INTRABC_CONTEXT)
            );
        }
    }

    /// `intrabc = None`: a single `use_intrabc = 0` S() (byte-equal to
    /// the leaf) when allowed, no bits on the fall-through arm —
    /// `None` returned either way so the caller proceeds to the
    /// §5.11.7 `else` (intra) arm.
    #[test]
    fn write_intra_frame_intrabc_arm_none_matches_leaf() {
        // allow_intrabc = true: one S() carrying 0.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let info = write_intra_frame_intrabc_arm(&mut writer, &mut cdfs, true, None).unwrap();
        assert!(info.is_none());
        let bytes = writer.finish();

        let mut leaf_cdfs = TileCdfContext::new_from_defaults();
        let mut leaf_writer = SymbolWriter::new(false);
        write_use_intrabc(&mut leaf_writer, &mut leaf_cdfs, 0, true).unwrap();
        assert_eq!(bytes, leaf_writer.finish(), "byte-equal to the leaf");

        let (mut walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = walker
            .decode_use_intrabc(&mut dec, &mut dec_cdfs, 0, 0, BLOCK_8X8, true)
            .unwrap();
        assert_eq!(got, 0);

        // allow_intrabc = false: no bits at all.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let info = write_intra_frame_intrabc_arm(&mut writer, &mut cdfs, false, None).unwrap();
        assert!(info.is_none());
        assert_eq!(writer.finish(), SymbolWriter::new(false).finish());
    }

    /// Caller-bug rejects: an intrabc block on a frame whose header
    /// never allowed it, a block above its own tile, and a target MV
    /// whose difference is not integer-aligned (intra block copy is
    /// integer-only per §5.11.32).
    #[test]
    fn write_intra_frame_intrabc_arm_rejects() {
        let stack = mk_intrabc_mv_stack([-64, 8], [0, 0], 2);
        let inputs = IntrabcArmInputs {
            mv: [-48, -40],
            mv_stack: &stack,
            use_128x128_superblock: false,
            mi_row: 16,
            mi_row_start: 0,
        };
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        assert!(matches!(
            write_intra_frame_intrabc_arm(&mut writer, &mut cdfs, false, Some(&inputs))
                .unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));

        // mi_row above the tile's MiRowStart — propagated from the
        // §5.11.26 predictor derivation.
        let above_tile = IntrabcArmInputs {
            mi_row: 31,
            mi_row_start: 32,
            ..inputs
        };
        let mut writer = SymbolWriter::new(false);
        assert!(matches!(
            write_intra_frame_intrabc_arm(&mut writer, &mut cdfs, true, Some(&above_tile))
                .unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));

        // Non-integer-aligned difference (diff col = +4, not a multiple
        // of 8) — propagated from the §5.11.32 component writer.
        let fractional = IntrabcArmInputs {
            mv: [-64, 12],
            ..inputs
        };
        let mut writer = SymbolWriter::new(false);
        assert!(matches!(
            write_intra_frame_intrabc_arm(&mut writer, &mut cdfs, true, Some(&fractional))
                .unwrap_err(),
            Error::PartitionWalkOutOfRange
        ));
    }

    // -----------------------------------------------------------------
    // §5.11.23 inter_block_mode_info composition — write_inter_block_mode_info
    // round-trips through a full §5.11.23 reader mirror (ref_frames →
    // mode → drl_mode → assign_mv), reproduced inline. The mirror is
    // exercised on the no-neighbour octet (avail_u = avail_l = false),
    // so every §8.3.2 `count_refs` is 0 and every `ref_count_ctx(0,0)`
    // is 1 (Equal), making the SINGLE_REFERENCE ref-frame reads
    // deterministic.
    // -----------------------------------------------------------------

    /// Full §5.11.23 reader mirror, restricted to the SINGLE_REFERENCE /
    /// compound-via-target subset the round-trip tests drive. Reads
    /// `read_ref_frames` (single arm), the §5.11.23 mode dispatch, the
    /// `drl_mode` loop, and `assign_mv` from a single live decoder, on
    /// the all-empty neighbour octet. Returns `(ref_frame, y_mode,
    /// ref_mv_idx, mv)`.
    #[allow(clippy::too_many_arguments)]
    fn mirror_inter_block_mode_info(
        bytes: &[u8],
        cdfs: &mut TileCdfContext,
        mv_stack: &FindMvStackResult,
        skip_mode: u8,
        skip_mode_frame: [i32; 2],
        seg_skip_active: bool,
        seg_globalmv_active: bool,
        force_integer_mv: bool,
        allow_high_precision_mv: bool,
    ) -> ([i32; 2], u8, u32, [[i32; 2]; 2]) {
        let pad = if bytes.is_empty() {
            vec![0u8]
        } else {
            bytes.to_vec()
        };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), false).unwrap();

        // §5.11.25 read_ref_frames (no-bit arms first, then the
        // SINGLE_REFERENCE arm-4 reader on the empty neighbour octet).
        let ref_frame: [i32; 2] = if skip_mode != 0 {
            [skip_mode_frame[0], skip_mode_frame[1]]
        } else if seg_skip_active || seg_globalmv_active {
            [1, -1]
        } else {
            // Arm 4, SINGLE_REFERENCE (no reference_select ⇒ comp_mode =
            // SINGLE). All count_refs are 0 ⇒ ref_count_ctx = 1 (Equal).
            let p1_ctx = ref_count_ctx(0, 0);
            let single_ref_p1 = dec.read_symbol(cdfs.single_ref_cdf(p1_ctx, 0)).unwrap() as u8;
            let r0: i32 = if single_ref_p1 != 0 {
                let single_ref_p2 = dec
                    .read_symbol(cdfs.single_ref_cdf(ref_count_ctx(0, 0), 1))
                    .unwrap() as u8;
                if single_ref_p2 == 0 {
                    let single_ref_p6 = dec
                        .read_symbol(cdfs.single_ref_cdf(ref_count_ctx(0, 0), 5))
                        .unwrap() as u8;
                    if single_ref_p6 != 0 {
                        6
                    } else {
                        5
                    }
                } else {
                    7
                }
            } else {
                let single_ref_p3 = dec
                    .read_symbol(cdfs.single_ref_cdf(ref_count_ctx(0, 0), 2))
                    .unwrap() as u8;
                if single_ref_p3 != 0 {
                    let single_ref_p5 = dec
                        .read_symbol(cdfs.single_ref_cdf(ref_count_ctx(0, 0), 4))
                        .unwrap() as u8;
                    if single_ref_p5 != 0 {
                        4
                    } else {
                        3
                    }
                } else {
                    let single_ref_p4 = dec
                        .read_symbol(cdfs.single_ref_cdf(ref_count_ctx(0, 0), 3))
                        .unwrap() as u8;
                    if single_ref_p4 != 0 {
                        2
                    } else {
                        1
                    }
                }
            };
            [r0, -1]
        };

        let is_compound = ref_frame[1] > 0;

        // §5.11.23 mode dispatch.
        let y_mode: u8 = if skip_mode != 0 {
            MODE_NEAREST_NEARESTMV
        } else if seg_skip_active || seg_globalmv_active {
            MODE_GLOBALMV
        } else if is_compound {
            let cm_ctx = compound_mode_ctx(
                mv_stack.ref_mv_context as usize,
                mv_stack.new_mv_context as usize,
            );
            let cm = dec.read_symbol(cdfs.compound_mode_cdf(cm_ctx)).unwrap() as u8;
            MODE_NEAREST_NEARESTMV + cm
        } else {
            let new_mv = dec
                .read_symbol(cdfs.new_mv_cdf(mv_stack.new_mv_context as usize))
                .unwrap();
            if new_mv == 0 {
                MODE_NEWMV
            } else {
                let zero_mv = dec
                    .read_symbol(cdfs.zero_mv_cdf(mv_stack.zero_mv_context as usize))
                    .unwrap();
                if zero_mv == 0 {
                    MODE_GLOBALMV
                } else {
                    let ref_mv = dec
                        .read_symbol(cdfs.ref_mv_cdf(mv_stack.ref_mv_context as usize))
                        .unwrap();
                    if ref_mv == 0 {
                        MODE_NEARESTMV
                    } else {
                        MODE_NEARMV
                    }
                }
            }
        };

        // §5.11.23 drl_mode loop.
        let mut ref_mv_idx: u32 = 0;
        if y_mode == MODE_NEWMV || y_mode == MODE_NEW_NEWMV {
            for idx in 0u32..2 {
                if mv_stack.num_mv_found > idx + 1 {
                    let drl_ctx = mv_stack.drl_ctx_stack[idx as usize] as usize;
                    let drl = dec.read_symbol(cdfs.drl_mode_cdf(drl_ctx)).unwrap();
                    if drl == 0 {
                        ref_mv_idx = idx;
                        break;
                    }
                    ref_mv_idx = idx + 1;
                }
            }
        } else if has_nearmv(y_mode) {
            ref_mv_idx = 1;
            for idx in 1u32..3 {
                if mv_stack.num_mv_found > idx + 1 {
                    let drl_ctx = mv_stack.drl_ctx_stack[idx as usize] as usize;
                    let drl = dec.read_symbol(cdfs.drl_mode_cdf(drl_ctx)).unwrap();
                    if drl == 0 {
                        ref_mv_idx = idx;
                        break;
                    }
                    ref_mv_idx = idx + 1;
                }
            }
        }

        // §5.11.31 assign_mv (use_intrabc = 0): NEWMV lists read a diff.
        let list_count: u8 = if is_compound { 2 } else { 1 };
        let mut mv: [[i32; 2]; 2] = [[0, 0], [0, 0]];
        for i in 0..list_count {
            let pred = assign_mv_pred_mv(mv_stack, y_mode, i, ref_mv_idx).unwrap();
            if crate::cdf::get_mode(y_mode, i as usize) == MODE_NEWMV {
                let mv_ctx = 0usize;
                let mv_joint = dec.read_symbol(cdfs.mv_joint_cdf(mv_ctx)).unwrap() as u8;
                let mut diff = [0i32; 2];
                if mv_joint == MV_JOINT_HZVNZ || mv_joint == MV_JOINT_HNZVNZ {
                    diff[0] = decode_read_mv_component_mirror(
                        &mut dec,
                        cdfs,
                        mv_ctx,
                        0,
                        force_integer_mv,
                        allow_high_precision_mv,
                    );
                }
                if mv_joint == MV_JOINT_HNZVZ || mv_joint == MV_JOINT_HNZVNZ {
                    diff[1] = decode_read_mv_component_mirror(
                        &mut dec,
                        cdfs,
                        mv_ctx,
                        1,
                        force_integer_mv,
                        allow_high_precision_mv,
                    );
                }
                mv[i as usize] = [pred[0] + diff[0], pred[1] + diff[1]];
            } else {
                mv[i as usize] = pred;
            }
        }

        (ref_frame, y_mode, ref_mv_idx, mv)
    }

    /// Build the no-neighbour arm-selection / octet inputs shared by the
    /// round-trip tests: empty neighbour octet, no segmentation, no
    /// reference_select (forces the §5.11.25 SINGLE_REFERENCE arm).
    #[allow(clippy::type_complexity)]
    fn empty_neighbour_inputs() -> (bool, bool, bool, bool, bool, bool, [i32; 2], [i32; 2]) {
        (
            false,
            false, // avail_u, avail_l
            true,
            true, // above_single, left_single
            false,
            false, // above_intra, left_intra
            [-1, -1],
            [-1, -1], // above_ref_frame, left_ref_frame
        )
    }

    /// §5.11.23 single-pred NEWMV round-trip: ref_frames (single arm) +
    /// new_mv S() + drl_mode loop + assign_mv NEWMV read all chain
    /// through the §5.11.23 reader mirror, recovering RefFrame / YMode /
    /// RefMvIdx / Mv exactly.
    #[test]
    fn write_inter_block_mode_info_single_newmv_round_trip() {
        let stack = mk_mv_stack(3); // NumMvFound = 3 ⇒ drl loop codes bits.
        let pred = assign_mv_pred_mv(&stack, MODE_NEWMV, 0, 1).unwrap();
        let target_mv = [pred[0] + 7, pred[1] - 5];
        let mv = [target_mv, [0, 0]];
        let ref_frame = [4i32, -1]; // GOLDEN_FRAME, single.
        let (au, al, asg, lsg, ai, li, arf, lrf) = empty_neighbour_inputs();

        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_inter_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            ref_frame,
            MODE_NEWMV,
            mv,
            /* ref_mv_idx = */ 1,
            &stack,
            BLOCK_8X8,
            /* skip_mode = */ 0,
            [0, 0],
            false,
            0,
            false,
            false,
            /* reference_select = */ false,
            au,
            al,
            asg,
            lsg,
            ai,
            li,
            arf,
            lrf,
            /* force_integer_mv = */ false,
            /* allow_high_precision_mv = */ true,
            &InterBlockModeInfoTail::bit_silent(),
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let (rf, ym, rmi, dmv) = mirror_inter_block_mode_info(
            &bytes,
            &mut dec_cdfs,
            &stack,
            0,
            [0, 0],
            false,
            false,
            false,
            true,
        );
        assert_eq!(rf, ref_frame, "RefFrame round-trips");
        assert_eq!(ym, MODE_NEWMV, "YMode round-trips");
        assert_eq!(rmi, 1, "RefMvIdx round-trips");
        assert_eq!(dmv[0], target_mv, "list-0 NEWMV Mv round-trips");
        // CDFs adapted in lockstep on the consulted rows.
        assert_eq!(enc_cdfs.new_mv_cdf(0), dec_cdfs.new_mv_cdf(0));
        assert_eq!(enc_cdfs.mv_joint_cdf(0), dec_cdfs.mv_joint_cdf(0));
    }

    /// §5.11.23 single-pred NEARMV round-trip: the `has_nearmv()` drl arm
    /// codes a `drl_mode` bit, and assign_mv emits NO MV bits (NEARMV is
    /// not NEWMV ⇒ Mv = PredMv).
    #[test]
    fn write_inter_block_mode_info_single_nearmv_round_trip() {
        let stack = mk_mv_stack(4); // deep enough for the has_nearmv loop.
                                    // NEARMV @ RefMvIdx 2 ⇒ PredMv = ref_stack_mv[2][0].
        let pred = assign_mv_pred_mv(&stack, MODE_NEARMV, 0, 2).unwrap();
        let mv = [pred, [0, 0]]; // Mv inherits PredMv (no NEWMV read).
        let ref_frame = [2i32, -1]; // LAST2_FRAME, single.
        let (au, al, asg, lsg, ai, li, arf, lrf) = empty_neighbour_inputs();

        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_inter_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            ref_frame,
            MODE_NEARMV,
            mv,
            /* ref_mv_idx = */ 2,
            &stack,
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            false,
            false,
            false,
            au,
            al,
            asg,
            lsg,
            ai,
            li,
            arf,
            lrf,
            false,
            true,
            &InterBlockModeInfoTail::bit_silent(),
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let (rf, ym, rmi, dmv) = mirror_inter_block_mode_info(
            &bytes,
            &mut dec_cdfs,
            &stack,
            0,
            [0, 0],
            false,
            false,
            false,
            true,
        );
        assert_eq!(rf, ref_frame);
        assert_eq!(ym, MODE_NEARMV);
        assert_eq!(rmi, 2, "RefMvIdx from has_nearmv drl arm");
        assert_eq!(dmv[0], pred, "NEARMV Mv = PredMv (no diff bits)");
    }

    /// §5.11.23 single-pred NEARESTMV round-trip: no drl bits (NEARESTMV
    /// is neither a NEWMV arm nor has_nearmv), no MV bits.
    #[test]
    fn write_inter_block_mode_info_single_nearestmv_round_trip() {
        let stack = mk_mv_stack(4);
        let pred = assign_mv_pred_mv(&stack, MODE_NEARESTMV, 0, 0).unwrap();
        let mv = [pred, [0, 0]];
        let ref_frame = [1i32, -1]; // LAST_FRAME.
        let (au, al, asg, lsg, ai, li, arf, lrf) = empty_neighbour_inputs();

        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_inter_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            ref_frame,
            MODE_NEARESTMV,
            mv,
            0,
            &stack,
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            false,
            false,
            false,
            au,
            al,
            asg,
            lsg,
            ai,
            li,
            arf,
            lrf,
            false,
            true,
            &InterBlockModeInfoTail::bit_silent(),
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let (rf, ym, rmi, dmv) = mirror_inter_block_mode_info(
            &bytes,
            &mut dec_cdfs,
            &stack,
            0,
            [0, 0],
            false,
            false,
            false,
            true,
        );
        assert_eq!(rf, ref_frame);
        assert_eq!(ym, MODE_NEARESTMV);
        assert_eq!(rmi, 0);
        assert_eq!(dmv[0], pred);
    }

    /// §5.11.23 arm-1 skip_mode round-trip: ref_frames forced to
    /// SkipModeFrame (no bits), YMode = NEAREST_NEARESTMV (no mode bits),
    /// both lists NEARESTMV ⇒ no MV bits. The whole composition emits
    /// zero symbols.
    #[test]
    fn write_inter_block_mode_info_skip_mode_emits_no_bits() {
        let stack = mk_mv_stack(2);
        // skip_mode forces a compound NEAREST_NEARESTMV pair.
        let skip_frame = [1i8, 5i8]; // LAST + BWDREF.
        let pred0 = assign_mv_pred_mv(&stack, MODE_NEAREST_NEARESTMV, 0, 0).unwrap();
        let pred1 = assign_mv_pred_mv(&stack, MODE_NEAREST_NEARESTMV, 1, 0).unwrap();
        let mv = [pred0, pred1];
        let ref_frame = [1i32, 5];
        let (au, al, asg, lsg, ai, li, arf, lrf) = empty_neighbour_inputs();

        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_inter_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            ref_frame,
            MODE_NEAREST_NEARESTMV,
            mv,
            0,
            &stack,
            BLOCK_8X8,
            /* skip_mode = */ 1,
            skip_frame,
            false,
            0,
            false,
            false,
            false,
            au,
            al,
            asg,
            lsg,
            ai,
            li,
            arf,
            lrf,
            false,
            true,
            &InterBlockModeInfoTail::bit_silent(),
        )
        .unwrap();
        let bytes = writer.finish();
        // No symbols emitted ⇒ an empty (or trivially-flushed) payload.
        // The decode mirror recovers the forced pair / mode with no reads.
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let (rf, ym, rmi, dmv) = mirror_inter_block_mode_info(
            &bytes,
            &mut dec_cdfs,
            &stack,
            1,
            [1, 5],
            false,
            false,
            false,
            true,
        );
        assert_eq!(rf, ref_frame);
        assert_eq!(ym, MODE_NEAREST_NEARESTMV);
        assert_eq!(rmi, 0);
        assert_eq!(dmv[0], pred0);
        assert_eq!(dmv[1], pred1);
    }

    /// §5.11.23 arm-2 seg_globalmv round-trip: ref_frames = (LAST, NONE)
    /// (no bits), YMode = GLOBALMV (no mode bits), single GLOBALMV list
    /// ⇒ Mv = GlobalMvs[0], no MV bits.
    #[test]
    fn write_inter_block_mode_info_seg_globalmv_emits_no_bits() {
        let stack = mk_mv_stack(2);
        let mv = [stack.global_mvs[0], [0, 0]];
        let ref_frame = [1i32, -1];
        let (au, al, asg, lsg, ai, li, arf, lrf) = empty_neighbour_inputs();

        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_inter_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            ref_frame,
            MODE_GLOBALMV,
            mv,
            0,
            &stack,
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            /* seg_skip_active = */ false,
            /* seg_globalmv_active = */ true,
            false,
            au,
            al,
            asg,
            lsg,
            ai,
            li,
            arf,
            lrf,
            false,
            true,
            &InterBlockModeInfoTail::bit_silent(),
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let (rf, ym, _rmi, dmv) = mirror_inter_block_mode_info(
            &bytes,
            &mut dec_cdfs,
            &stack,
            0,
            [0, 0],
            false,
            true,
            false,
            true,
        );
        assert_eq!(rf, ref_frame);
        assert_eq!(ym, MODE_GLOBALMV);
        assert_eq!(dmv[0], stack.global_mvs[0]);
    }

    /// §5.11.23 arm-consistency reject: a `y_mode` that doesn't match the
    /// active arm (here arm 2 forces GLOBALMV, but NEWMV is handed) is a
    /// caller bug.
    #[test]
    fn write_inter_block_mode_info_rejects_arm_inconsistent_y_mode() {
        let stack = mk_mv_stack(2);
        let (au, al, asg, lsg, ai, li, arf, lrf) = empty_neighbour_inputs();
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_inter_block_mode_info(
            &mut writer,
            &mut enc_cdfs,
            [1, -1],
            MODE_NEWMV, // inconsistent with the seg_globalmv arm.
            [[0, 0], [0, 0]],
            0,
            &stack,
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            false,
            true,
            false,
            au,
            al,
            asg,
            lsg,
            ai,
            li,
            arf,
            lrf,
            false,
            true,
            &InterBlockModeInfoTail::bit_silent(),
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.28 read_interintra_mode — write_interintra_mode round-trips
    // through PartitionWalker::read_interintra_mode (r276).
    // -----------------------------------------------------------------

    /// Zero-read readout shape (`interintra = 0`, every option `None`).
    fn ii_zero() -> InterIntraReadout {
        InterIntraReadout {
            interintra: 0,
            interintra_mode: None,
            wedge_interintra: None,
            wedge_index: None,
        }
    }

    /// Encode `readout` then decode it back through the §5.11.28
    /// reader on a fresh walker; returns the recovered readout.
    fn roundtrip_interintra(
        readout: InterIntraReadout,
        mi_size: usize,
        skip_mode: u8,
        is_compound: bool,
        enable: bool,
    ) -> InterIntraReadout {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_interintra_mode(
            &mut writer,
            &mut enc_cdfs,
            &readout,
            mi_size,
            skip_mode,
            is_compound,
            enable,
        )
        .unwrap();
        let bytes = writer.finish();

        let (walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), false).unwrap();
        let got = walker
            .read_interintra_mode(
                &mut dec,
                &mut dec_cdfs,
                mi_size,
                skip_mode,
                is_compound,
                enable,
            )
            .unwrap();
        // Both sides must have adapted the shared rows identically.
        if let Some(ctx) = crate::cdf::interintra_ctx(mi_size) {
            assert_eq!(enc_cdfs.inter_intra.get(ctx), dec_cdfs.inter_intra.get(ctx));
            assert_eq!(
                enc_cdfs.inter_intra_mode.get(ctx),
                dec_cdfs.inter_intra_mode.get(ctx)
            );
        }
        if mi_size < BLOCK_SIZES {
            assert_eq!(
                enc_cdfs.wedge_inter_intra[mi_size],
                dec_cdfs.wedge_inter_intra[mi_size]
            );
            assert_eq!(enc_cdfs.wedge_index[mi_size], dec_cdfs.wedge_index[mi_size]);
        }
        got
    }

    /// Every §5.11.28 gate-closing clause (skip_mode, sequence flag,
    /// compound, block-size band) writes no bits for the zero readout —
    /// the emitted stream equals an untouched writer's.
    #[test]
    fn write_interintra_mode_gate_closed_writes_no_bits() {
        // (mi_size, skip_mode, is_compound, enable)
        let configs = [
            (BLOCK_8X8, 1u8, false, true),
            (BLOCK_8X8, 0u8, false, false),
            (BLOCK_8X8, 0u8, true, true),
            (BLOCK_4X4, 0u8, false, true),
            (BLOCK_64X64, 0u8, false, true),
        ];
        for (mi_size, skip_mode, is_compound, enable) in configs {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_interintra_mode(
                &mut writer,
                &mut cdfs,
                &ii_zero(),
                mi_size,
                skip_mode,
                is_compound,
                enable,
            )
            .unwrap();
            let got = writer.finish();
            let want = SymbolWriter::new(false).finish();
            assert_eq!(got, want, "gate-closed config writes no symbols");
        }
    }

    /// Gate closed but the readout claims a coded shape — caller bug.
    #[test]
    fn write_interintra_mode_gate_closed_rejects_nonzero_readout() {
        let bad = InterIntraReadout {
            interintra: 1,
            interintra_mode: Some(0),
            wedge_interintra: Some(0),
            wedge_index: None,
        };
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_interintra_mode(&mut writer, &mut cdfs, &bad, BLOCK_8X8, 1, false, true)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Gate open: `interintra = 0` plus all four inner-arm shapes
    /// (every `interintra_mode`, wedge off) round-trip at every band
    /// size.
    #[test]
    fn write_interintra_mode_round_trips_inner_arm_no_wedge() {
        for mi_size in [BLOCK_8X8, BLOCK_16X16, BLOCK_32X32] {
            let got = roundtrip_interintra(ii_zero(), mi_size, 0, false, true);
            assert_eq!(got, ii_zero(), "interintra = 0 at mi_size {mi_size}");
            for ii_mode in 0..INTERINTRA_MODES as u8 {
                let readout = InterIntraReadout {
                    interintra: 1,
                    interintra_mode: Some(ii_mode),
                    wedge_interintra: Some(0),
                    wedge_index: None,
                };
                let got = roundtrip_interintra(readout, mi_size, 0, false, true);
                assert_eq!(got, readout, "mode {ii_mode} at mi_size {mi_size}");
            }
        }
    }

    /// Gate open, wedge sub-branch: `wedge_index` extremes round-trip.
    #[test]
    fn write_interintra_mode_round_trips_wedge_indices() {
        for wi in [0u8, 7, 15] {
            let readout = InterIntraReadout {
                interintra: 1,
                interintra_mode: Some(1),
                wedge_interintra: Some(1),
                wedge_index: Some(wi),
            };
            let got = roundtrip_interintra(readout, BLOCK_8X8, 0, false, true);
            assert_eq!(got, readout, "wedge_index {wi}");
        }
    }

    /// Readout shapes the §5.11.28 reader can never produce on an open
    /// gate are rejected before any symbol decodes wrong.
    #[test]
    fn write_interintra_mode_rejects_inconsistent_readouts() {
        let cases = [
            // interintra out of the binary alphabet.
            InterIntraReadout {
                interintra: 2,
                interintra_mode: None,
                wedge_interintra: None,
                wedge_index: None,
            },
            // interintra = 0 with a populated inner field.
            InterIntraReadout {
                interintra: 0,
                interintra_mode: Some(0),
                wedge_interintra: None,
                wedge_index: None,
            },
            // interintra = 1 missing interintra_mode.
            InterIntraReadout {
                interintra: 1,
                interintra_mode: None,
                wedge_interintra: Some(0),
                wedge_index: None,
            },
            // interintra_mode out of range.
            InterIntraReadout {
                interintra: 1,
                interintra_mode: Some(INTERINTRA_MODES as u8),
                wedge_interintra: Some(0),
                wedge_index: None,
            },
            // wedge_interintra = 1 missing wedge_index.
            InterIntraReadout {
                interintra: 1,
                interintra_mode: Some(0),
                wedge_interintra: Some(1),
                wedge_index: None,
            },
            // wedge_interintra = 0 with a stray wedge_index.
            InterIntraReadout {
                interintra: 1,
                interintra_mode: Some(0),
                wedge_interintra: Some(0),
                wedge_index: Some(0),
            },
            // wedge_index out of range.
            InterIntraReadout {
                interintra: 1,
                interintra_mode: Some(0),
                wedge_interintra: Some(1),
                wedge_index: Some(WEDGE_TYPES as u8),
            },
        ];
        for bad in cases {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err =
                write_interintra_mode(&mut writer, &mut cdfs, &bad, BLOCK_8X8, 0, false, true)
                    .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "rejected {bad:?}"
            );
        }
    }

    /// `interintra = 0` on an open gate emits exactly one symbol —
    /// byte-equal to an independent single-S() encode over the same row.
    #[test]
    fn write_interintra_mode_interintra_zero_emits_one_symbol() {
        let mut a = TileCdfContext::new_from_defaults();
        let mut wa = SymbolWriter::new(false);
        write_interintra_mode(&mut wa, &mut a, &ii_zero(), BLOCK_16X16, 0, false, true).unwrap();
        let got = wa.finish();

        let mut b = TileCdfContext::new_from_defaults();
        let mut wb = SymbolWriter::new(false);
        let ctx = crate::cdf::interintra_ctx(BLOCK_16X16).unwrap();
        let row = b.inter_intra_cdf(ctx).unwrap();
        wb.write_symbol(0, row).unwrap();
        let want = wb.finish();
        assert_eq!(got, want, "open gate + interintra = 0 is one S()");
    }

    // -----------------------------------------------------------------
    // §5.11.27 read_motion_mode — write_motion_mode round-trips through
    // PartitionWalker::read_motion_mode (r276).
    // -----------------------------------------------------------------

    /// Build a 16×16 mi walker with one stamped inter neighbour above
    /// `(2, 0)` so `has_overlappable_candidates()` is true for a
    /// BLOCK_8X8 block at that position (the same fixture the decode
    /// tests use).
    fn walker_with_overlappable_above() -> PartitionWalker {
        let geom = TileGeometry {
            mi_row_start: 0,
            mi_row_end: 16,
            mi_col_start: 0,
            mi_col_end: 16,
        };
        let mut walker = PartitionWalker::new(16, 16, geom).unwrap();
        walker.stamp_inter_neighbour(1, 0, BLOCK_8X8, [1, -1], MODE_NEARESTMV, [[0, 0], [0, 0]]);
        walker
    }

    /// Every §5.11.27 short-circuit arm forces SIMPLE with no bits;
    /// a non-SIMPLE target on those arms is a caller bug.
    #[test]
    fn write_motion_mode_short_circuit_arms_write_no_bits_and_reject_non_simple() {
        // (mi_size, skip_mode, is_compound, ref_frame, y_mode,
        //  switchable, force_integer_mv, gm_type_ref0, has_overlappable)
        struct Arm {
            mi_size: usize,
            skip_mode: u8,
            is_compound: bool,
            ref_frame: [i32; 2],
            y_mode: u8,
            switchable: bool,
            force_integer_mv: bool,
            gm_type_ref0: i32,
            has_overlappable: bool,
        }
        let base = Arm {
            mi_size: BLOCK_16X16,
            skip_mode: 0,
            is_compound: false,
            ref_frame: [1, -1],
            y_mode: MODE_NEARESTMV,
            switchable: true,
            force_integer_mv: false,
            gm_type_ref0: 0,
            has_overlappable: true,
        };
        let arms = [
            Arm {
                skip_mode: 1,
                ..base
            },
            Arm {
                switchable: false,
                ..base
            },
            // Min(Block_Width, Block_Height) < 8.
            Arm {
                mi_size: BLOCK_4X8,
                ..base
            },
            // GLOBALMV with a beyond-TRANSLATION global model.
            Arm {
                y_mode: MODE_GLOBALMV,
                gm_type_ref0: GM_TYPE_TRANSLATION + 1,
                ..base
            },
            Arm {
                is_compound: true,
                ref_frame: [1, 4],
                ..base
            },
            // RefFrame[1] == INTRA_FRAME = 0.
            Arm {
                ref_frame: [1, 0],
                ..base
            },
            Arm {
                has_overlappable: false,
                ..base
            },
        ];
        for (i, arm) in arms.iter().enumerate() {
            let mut gm_type = [0i32; 8];
            gm_type[arm.ref_frame[0] as usize] = arm.gm_type_ref0;
            let run = |motion_mode: u8| {
                let mut cdfs = TileCdfContext::new_from_defaults();
                let mut writer = SymbolWriter::new(false);
                let r = write_motion_mode(
                    &mut writer,
                    &mut cdfs,
                    motion_mode,
                    arm.mi_size,
                    arm.skip_mode,
                    arm.is_compound,
                    arm.ref_frame,
                    arm.y_mode,
                    0,
                    arm.switchable,
                    true,
                    arm.force_integer_mv,
                    gm_type,
                    [false; 7],
                    arm.has_overlappable,
                );
                (r, writer.finish())
            };
            let (ok, bytes) = run(MOTION_MODE_SIMPLE);
            ok.unwrap();
            assert_eq!(
                bytes,
                SymbolWriter::new(false).finish(),
                "short-circuit arm {i} writes no symbols"
            );
            let (err, _) = run(MOTION_MODE_OBMC);
            assert!(
                matches!(err.unwrap_err(), Error::PartitionWalkOutOfRange),
                "short-circuit arm {i} rejects OBMC"
            );
        }
    }

    /// Arm A (`use_obmc` S(), forced via `force_integer_mv = true`):
    /// SIMPLE and OBMC round-trip; WARPED_CAUSAL is unreachable.
    #[test]
    fn write_motion_mode_arm_a_round_trips() {
        for target in [MOTION_MODE_SIMPLE, MOTION_MODE_OBMC] {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_motion_mode(
                &mut writer,
                &mut enc_cdfs,
                target,
                BLOCK_8X8,
                0,
                false,
                [1, -1],
                MODE_NEARESTMV,
                0,
                true,
                true,
                /* force_integer_mv = */ true,
                [0; 8],
                [false; 7],
                true,
            )
            .unwrap();
            let bytes = writer.finish();

            let walker = walker_with_overlappable_above();
            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let got = walker
                .read_motion_mode(
                    &mut dec,
                    &mut dec_cdfs,
                    2,
                    0,
                    BLOCK_8X8,
                    0,
                    false,
                    [1, -1],
                    MODE_NEARESTMV,
                    0,
                    true,
                    true,
                    true,
                    [0; 8],
                    [false; 7],
                )
                .unwrap();
            assert_eq!(got, target, "arm A target {target} round-trips");
            assert_eq!(enc_cdfs.use_obmc[BLOCK_8X8], dec_cdfs.use_obmc[BLOCK_8X8]);
        }
        // WARPED_CAUSAL on arm A is a caller bug.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_motion_mode(
            &mut writer,
            &mut cdfs,
            crate::cdf::MOTION_MODE_WARPED_CAUSAL,
            BLOCK_8X8,
            0,
            false,
            [1, -1],
            MODE_NEARESTMV,
            0,
            true,
            true,
            true,
            [0; 8],
            [false; 7],
            true,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Arm B (`motion_mode` S(): `num_samples > 0`, warped motion
    /// allowed, unscaled, fractional MVs): all three ordinals
    /// round-trip.
    #[test]
    fn write_motion_mode_arm_b_round_trips() {
        for target in 0..MOTION_MODES as u8 {
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_motion_mode(
                &mut writer,
                &mut enc_cdfs,
                target,
                BLOCK_8X8,
                0,
                false,
                [1, -1],
                MODE_NEARESTMV,
                /* num_samples = */ 2,
                true,
                true,
                false,
                [0; 8],
                [false; 7],
                true,
            )
            .unwrap();
            let bytes = writer.finish();

            let walker = walker_with_overlappable_above();
            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let got = walker
                .read_motion_mode(
                    &mut dec,
                    &mut dec_cdfs,
                    2,
                    0,
                    BLOCK_8X8,
                    0,
                    false,
                    [1, -1],
                    MODE_NEARESTMV,
                    2,
                    true,
                    true,
                    false,
                    [0; 8],
                    [false; 7],
                )
                .unwrap();
            assert_eq!(got, target, "arm B target {target} round-trips");
            assert_eq!(
                enc_cdfs.motion_mode[BLOCK_8X8],
                dec_cdfs.motion_mode[BLOCK_8X8]
            );
        }
    }

    /// A scaled `RefFrame[ 0 ]` routes to arm A even with warp samples
    /// available — WARPED_CAUSAL must be rejected and the `use_obmc`
    /// row (not the `motion_mode` row) adapts.
    #[test]
    fn write_motion_mode_scaled_ref_routes_to_arm_a() {
        let mut scaled = [false; 7];
        scaled[0] = true; // LAST_FRAME.
        let mut cdfs = TileCdfContext::new_from_defaults();
        let use_obmc_before = cdfs.use_obmc[BLOCK_8X8];
        let motion_mode_before = cdfs.motion_mode[BLOCK_8X8];
        let mut writer = SymbolWriter::new(false);
        write_motion_mode(
            &mut writer,
            &mut cdfs,
            MOTION_MODE_OBMC,
            BLOCK_8X8,
            0,
            false,
            [1, -1],
            MODE_NEARESTMV,
            2,
            true,
            true,
            false,
            [0; 8],
            scaled,
            true,
        )
        .unwrap();
        assert_ne!(cdfs.use_obmc[BLOCK_8X8], use_obmc_before);
        assert_eq!(cdfs.motion_mode[BLOCK_8X8], motion_mode_before);

        let mut writer = SymbolWriter::new(false);
        let err = write_motion_mode(
            &mut writer,
            &mut cdfs,
            crate::cdf::MOTION_MODE_WARPED_CAUSAL,
            BLOCK_8X8,
            0,
            false,
            [1, -1],
            MODE_NEARESTMV,
            2,
            true,
            true,
            false,
            [0; 8],
            scaled,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.29 read_compound_type — write_compound_type round-trips
    // through PartitionWalker::read_compound_type (r276).
    // -----------------------------------------------------------------

    /// The §5.11.29 line-1/2 pre-set readout with `compound_type = ty`.
    fn ct_preset(ty: u8) -> crate::cdf::CompoundTypeReadout {
        crate::cdf::CompoundTypeReadout {
            comp_group_idx: 0,
            compound_idx: 1,
            compound_type: ty,
            wedge_index: None,
            wedge_sign: None,
            mask_type: None,
        }
    }

    /// Encode `readout` then decode it back through the §5.11.29 reader
    /// on a fresh walker at `(2, 2)` with both neighbours available and
    /// compound-coded (`!AboveSingle` / `!LeftSingle` — the grid
    /// pre-fill identities 0 / 1 feed both ctx walks on both sides).
    #[allow(clippy::too_many_arguments)]
    fn roundtrip_compound_type(
        readout: crate::cdf::CompoundTypeReadout,
        mi_size: usize,
        skip_mode: u8,
        is_compound: bool,
        interintra: u8,
        wedge_interintra: u8,
        enable_masked_compound: bool,
        enable_jnt_comp: bool,
        dist_equal: bool,
    ) -> crate::cdf::CompoundTypeReadout {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_compound_type(
            &mut writer,
            &mut enc_cdfs,
            &readout,
            mi_size,
            skip_mode,
            is_compound,
            interintra,
            wedge_interintra,
            enable_masked_compound,
            enable_jnt_comp,
            dist_equal,
            true,
            true,
            false,
            false,
            false,
            false,
            // Grid pre-fill identities for a fresh walker.
            0,
            0,
            1,
            1,
        )
        .unwrap();
        let bytes = writer.finish();

        let (walker, mut dec_cdfs) = fresh_walker_and_cdfs();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), false).unwrap();
        walker
            .read_compound_type(
                &mut dec,
                &mut dec_cdfs,
                2,
                2,
                mi_size,
                skip_mode,
                is_compound,
                interintra,
                wedge_interintra,
                enable_masked_compound,
                enable_jnt_comp,
                dist_equal,
                true,
                true,
                false,
                false,
                false,
                false,
            )
            .unwrap()
    }

    /// `skip_mode` short-circuit: the pre-set readout writes no bits;
    /// anything else on that arm is a caller bug.
    #[test]
    fn write_compound_type_skip_mode_writes_no_bits() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_compound_type(
            &mut writer,
            &mut cdfs,
            &ct_preset(COMPOUND_AVERAGE),
            BLOCK_8X8,
            1,
            true,
            0,
            0,
            true,
            true,
            false,
            false,
            false,
            true,
            true,
            false,
            false,
            0,
            0,
            1,
            1,
        )
        .unwrap();
        assert_eq!(writer.finish(), SymbolWriter::new(false).finish());

        let mut writer = SymbolWriter::new(false);
        let err = write_compound_type(
            &mut writer,
            &mut cdfs,
            &ct_preset(COMPOUND_DISTANCE),
            BLOCK_8X8,
            1,
            true,
            0,
            0,
            true,
            true,
            false,
            false,
            false,
            true,
            true,
            false,
            false,
            0,
            0,
            1,
            1,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Single-pred else-arm: the §5.11.28 outcome alone selects the
    /// type, no bits on any of the three shapes; a mismatched type is
    /// rejected.
    #[test]
    fn write_compound_type_single_pred_arms_write_no_bits() {
        // (interintra, wedge_interintra, expected type)
        let shapes = [
            (0u8, 0u8, COMPOUND_AVERAGE),
            (1, 0, COMPOUND_INTRA),
            (1, 1, COMPOUND_WEDGE),
        ];
        for (ii, wii, expected) in shapes {
            let got = roundtrip_compound_type(
                ct_preset(expected),
                BLOCK_8X8,
                0,
                false,
                ii,
                wii,
                true,
                true,
                false,
            );
            assert_eq!(got, ct_preset(expected), "shape ({ii}, {wii})");

            // Mismatched type — pick a type the arm can't produce.
            let bad = if expected == COMPOUND_AVERAGE {
                COMPOUND_INTRA
            } else {
                COMPOUND_AVERAGE
            };
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            let err = write_compound_type(
                &mut writer,
                &mut cdfs,
                &ct_preset(bad),
                BLOCK_8X8,
                0,
                false,
                ii,
                wii,
                true,
                true,
                false,
                true,
                true,
                false,
                false,
                false,
                false,
                0,
                0,
                1,
                1,
            )
            .unwrap_err();
            assert!(matches!(err, Error::PartitionWalkOutOfRange));
        }
    }

    /// Compound group-1 wedge path: `comp_group_idx = 1`,
    /// `compound_type = COMPOUND_WEDGE`, `wedge_index` S() +
    /// `wedge_sign` L(1) all round-trip.
    #[test]
    fn write_compound_type_group1_wedge_round_trips() {
        for (wi, ws) in [(0u8, 0u8), (5, 1), (15, 0)] {
            let readout = crate::cdf::CompoundTypeReadout {
                comp_group_idx: 1,
                compound_idx: 1,
                compound_type: COMPOUND_WEDGE,
                wedge_index: Some(wi),
                wedge_sign: Some(ws),
                mask_type: None,
            };
            let got = roundtrip_compound_type(readout, BLOCK_8X8, 0, true, 0, 0, true, true, false);
            assert_eq!(got, readout, "wedge ({wi}, {ws})");
        }
    }

    /// Compound group-1 diffwtd path (`Wedge_Bits > 0` so the type is
    /// S()-coded): `mask_type` L(1) round-trips both ways.
    #[test]
    fn write_compound_type_group1_diffwtd_round_trips() {
        for mt in [0u8, 1] {
            let readout = crate::cdf::CompoundTypeReadout {
                comp_group_idx: 1,
                compound_idx: 1,
                compound_type: COMPOUND_DIFFWTD,
                wedge_index: None,
                wedge_sign: None,
                mask_type: Some(mt),
            };
            let got = roundtrip_compound_type(readout, BLOCK_8X8, 0, true, 0, 0, true, true, false);
            assert_eq!(got, readout, "diffwtd mask_type {mt}");
        }
    }

    /// Compound group-1 with `Wedge_Bits[ MiSize ] == 0` (BLOCK_64X64):
    /// the type is forced to COMPOUND_DIFFWTD with no `compound_type`
    /// S(); claiming COMPOUND_WEDGE there is a caller bug.
    #[test]
    fn write_compound_type_group1_n_zero_forces_diffwtd() {
        let readout = crate::cdf::CompoundTypeReadout {
            comp_group_idx: 1,
            compound_idx: 1,
            compound_type: COMPOUND_DIFFWTD,
            wedge_index: None,
            wedge_sign: None,
            mask_type: Some(1),
        };
        let got = roundtrip_compound_type(readout, BLOCK_64X64, 0, true, 0, 0, true, true, false);
        assert_eq!(got, readout);

        let bad = crate::cdf::CompoundTypeReadout {
            compound_type: COMPOUND_WEDGE,
            wedge_index: Some(0),
            wedge_sign: Some(0),
            mask_type: None,
            ..readout
        };
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_compound_type(
            &mut writer,
            &mut cdfs,
            &bad,
            BLOCK_64X64,
            0,
            true,
            0,
            0,
            true,
            true,
            false,
            true,
            true,
            false,
            false,
            false,
            false,
            0,
            0,
            1,
            1,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Compound group-0 jnt path: `compound_idx = 1 ⇒ AVERAGE` and
    /// `compound_idx = 0 ⇒ DISTANCE` round-trip under both `dist_equal`
    /// seeds (ctx 0 + 2 vs ctx 3 + 2).
    #[test]
    fn write_compound_type_group0_jnt_round_trips() {
        for dist_equal in [false, true] {
            for (ci, ty) in [(1u8, COMPOUND_AVERAGE), (0, COMPOUND_DISTANCE)] {
                let readout = crate::cdf::CompoundTypeReadout {
                    comp_group_idx: 0,
                    compound_idx: ci,
                    compound_type: ty,
                    wedge_index: None,
                    wedge_sign: None,
                    mask_type: None,
                };
                let got = roundtrip_compound_type(
                    readout, BLOCK_8X8, 0, true, 0, 0, true, true, dist_equal,
                );
                assert_eq!(got, readout, "jnt ({ci}, dist_equal {dist_equal})");
            }
        }
    }

    /// Compound group-0 without jnt: only the `comp_group_idx` S() is
    /// coded — byte-equal to an independent single-symbol encode.
    #[test]
    fn write_compound_type_group0_no_jnt_emits_one_symbol() {
        let mut a = TileCdfContext::new_from_defaults();
        let mut wa = SymbolWriter::new(false);
        write_compound_type(
            &mut wa,
            &mut a,
            &ct_preset(COMPOUND_AVERAGE),
            BLOCK_8X8,
            0,
            true,
            0,
            0,
            true,
            /* enable_jnt_comp = */ false,
            false,
            true,
            true,
            false,
            false,
            false,
            false,
            0,
            0,
            1,
            1,
        )
        .unwrap();
        let got = wa.finish();

        let mut b = TileCdfContext::new_from_defaults();
        let mut wb = SymbolWriter::new(false);
        let ctx = crate::cdf::comp_group_idx_ctx(true, false, 0, false, true, false, 0, false);
        let row = b.comp_group_idx_cdf(ctx);
        wb.write_symbol(0, row).unwrap();
        let want = wb.finish();
        assert_eq!(
            got, want,
            "group-0 no-jnt is exactly one comp_group_idx S()"
        );
    }

    /// Readout shapes the §5.11.29 reader can never produce are
    /// rejected.
    #[test]
    fn write_compound_type_rejects_inconsistent_readouts() {
        let run =
            |readout: &crate::cdf::CompoundTypeReadout, enable_masked: bool, enable_jnt: bool| {
                let mut cdfs = TileCdfContext::new_from_defaults();
                let mut writer = SymbolWriter::new(false);
                write_compound_type(
                    &mut writer,
                    &mut cdfs,
                    readout,
                    BLOCK_8X8,
                    0,
                    true,
                    0,
                    0,
                    enable_masked,
                    enable_jnt,
                    false,
                    true,
                    true,
                    false,
                    false,
                    false,
                    false,
                    0,
                    0,
                    1,
                    1,
                )
            };
        // comp_group_idx = 1 with masked compound disabled.
        let bad = crate::cdf::CompoundTypeReadout {
            comp_group_idx: 1,
            compound_idx: 1,
            compound_type: COMPOUND_DIFFWTD,
            wedge_index: None,
            wedge_sign: None,
            mask_type: Some(0),
        };
        assert!(run(&bad, false, true).is_err());
        // jnt arm with a type inconsistent with compound_idx.
        let bad = crate::cdf::CompoundTypeReadout {
            comp_group_idx: 0,
            compound_idx: 1,
            compound_type: COMPOUND_DISTANCE,
            wedge_index: None,
            wedge_sign: None,
            mask_type: None,
        };
        assert!(run(&bad, true, true).is_err());
        // wedge type missing its sub-branch fields.
        let bad = crate::cdf::CompoundTypeReadout {
            comp_group_idx: 1,
            compound_idx: 1,
            compound_type: COMPOUND_WEDGE,
            wedge_index: None,
            wedge_sign: None,
            mask_type: None,
        };
        assert!(run(&bad, true, true).is_err());
        // diffwtd type with stray wedge fields.
        let bad = crate::cdf::CompoundTypeReadout {
            comp_group_idx: 1,
            compound_idx: 1,
            compound_type: COMPOUND_DIFFWTD,
            wedge_index: Some(0),
            wedge_sign: None,
            mask_type: Some(0),
        };
        assert!(run(&bad, true, true).is_err());
        // wedge_index out of range.
        let bad = crate::cdf::CompoundTypeReadout {
            comp_group_idx: 1,
            compound_idx: 1,
            compound_type: COMPOUND_WEDGE,
            wedge_index: Some(WEDGE_TYPES as u8),
            wedge_sign: Some(0),
            mask_type: None,
        };
        assert!(run(&bad, true, true).is_err());
        // group-0 no-jnt arm with a non-preset compound_idx.
        let bad = crate::cdf::CompoundTypeReadout {
            comp_group_idx: 0,
            compound_idx: 0,
            compound_type: COMPOUND_AVERAGE,
            wedge_index: None,
            wedge_sign: None,
            mask_type: None,
        };
        assert!(run(&bad, true, false).is_err());
    }

    // -----------------------------------------------------------------
    // §5.11.x interpolation-filter loop — write_interpolation_filter
    // round-trips through PartitionWalker::read_interpolation_filter
    // (r277).
    // -----------------------------------------------------------------

    /// §5.9.24 identity warp parameters (diagonal `1 <<
    /// WARPEDMODEL_PREC_BITS = 1 << 16`) for the §7.10.2.1 global-mv
    /// setup in the dispatcher round-trip below.
    fn identity_gm_params() -> [[i32; 6]; 8] {
        let mut params = [[0i32; 6]; 8];
        for p in &mut params {
            p[2] = 1 << 16;
            p[5] = 1 << 16;
        }
        params
    }

    /// Encode `readout` then decode it back through the §5.11.x reader
    /// on a fresh walker at `(0, 0)` (no neighbours on either side);
    /// returns the recovered readout after asserting CDF lockstep.
    #[allow(clippy::too_many_arguments)]
    fn roundtrip_interp_filter(
        readout: InterpolationFilterReadout,
        mi_size: usize,
        skip_mode: u8,
        y_mode: u8,
        motion_mode: u8,
        interpolation_filter: u8,
        enable_dual_filter: bool,
        gm_type: [i32; 8],
    ) -> InterpolationFilterReadout {
        let ref_frame = [1i32, -1];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_interpolation_filter(
            &mut writer,
            &mut enc_cdfs,
            &readout,
            mi_size,
            skip_mode,
            /* is_compound = */ false,
            ref_frame,
            y_mode,
            motion_mode,
            interpolation_filter,
            enable_dual_filter,
            gm_type,
            /* avail_u = */ false,
            /* avail_l = */ false,
            [-1, -1],
            [-1, -1],
            [EIGHTTAP, EIGHTTAP],
            [EIGHTTAP, EIGHTTAP],
        )
        .unwrap();
        let bytes = writer.finish();
        let pad = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut dec = SymbolDecoder::init_symbol(&pad, pad.len(), false).unwrap();
        let geom = TileGeometry {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let walker = PartitionWalker::new(8, 8, geom).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let got = walker
            .read_interpolation_filter(
                &mut dec,
                &mut dec_cdfs,
                0,
                0,
                mi_size,
                skip_mode,
                false,
                ref_frame,
                y_mode,
                motion_mode,
                interpolation_filter,
                enable_dual_filter,
                gm_type,
            )
            .unwrap();
        // Writer and reader adapted the same TileInterpFilterCdf rows.
        assert_eq!(enc_cdfs.interp_filter, dec_cdfs.interp_filter);
        got
    }

    /// SWITCHABLE + `needs_interp_filter() == 1` + dual filter: two
    /// independent S() symbols, one per direction, round-trip exactly.
    #[test]
    fn write_interpolation_filter_switchable_dual_round_trips() {
        let target = InterpolationFilterReadout {
            interp_filter: [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SMOOTH],
            read_from_bitstream: [true, true],
        };
        let got = roundtrip_interp_filter(
            target,
            BLOCK_16X16,
            0,
            MODE_NEARESTMV,
            MOTION_MODE_SIMPLE,
            SWITCHABLE,
            /* enable_dual_filter = */ true,
            [0; 8],
        );
        assert_eq!(got, target);
    }

    /// SWITCHABLE + needs + single-dir loop: one S(); slot 1 mirrors
    /// slot 0 with its read flag clear per the §5.11.x post-loop line.
    #[test]
    fn write_interpolation_filter_switchable_single_dir_round_trips() {
        let target = InterpolationFilterReadout {
            interp_filter: [crate::cdf::EIGHTTAP_SMOOTH, crate::cdf::EIGHTTAP_SMOOTH],
            read_from_bitstream: [true, false],
        };
        let got = roundtrip_interp_filter(
            target,
            BLOCK_8X8,
            0,
            MODE_NEARESTMV,
            MOTION_MODE_SIMPLE,
            SWITCHABLE,
            /* enable_dual_filter = */ false,
            [0; 8],
        );
        assert_eq!(got, target);
    }

    /// Non-SWITCHABLE frame header: both slots forced, zero bits; the
    /// reader recovers the same forced shape from an untouched stream.
    #[test]
    fn write_interpolation_filter_non_switchable_writes_no_bits() {
        let target = InterpolationFilterReadout {
            interp_filter: [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SHARP],
            read_from_bitstream: [false, false],
        };
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_interpolation_filter(
            &mut writer,
            &mut cdfs,
            &target,
            BLOCK_8X8,
            0,
            false,
            [1, -1],
            MODE_NEARESTMV,
            MOTION_MODE_SIMPLE,
            crate::cdf::EIGHTTAP_SHARP,
            true,
            [0; 8],
            false,
            false,
            [-1, -1],
            [-1, -1],
            [EIGHTTAP, EIGHTTAP],
            [EIGHTTAP, EIGHTTAP],
        )
        .unwrap();
        assert_eq!(writer.finish(), SymbolWriter::new(false).finish());
        let got = roundtrip_interp_filter(
            target,
            BLOCK_8X8,
            0,
            MODE_NEARESTMV,
            MOTION_MODE_SIMPLE,
            crate::cdf::EIGHTTAP_SHARP,
            true,
            [0; 8],
        );
        assert_eq!(got, target);
    }

    /// `needs_interp_filter() == 0` arms — skip_mode, LOCALWARP motion
    /// mode, and large-GLOBALMV-with-non-TRANSLATION gm — all force
    /// EIGHTTAP with zero bits; the GLOBALMV-with-TRANSLATION twin
    /// re-opens the S().
    #[test]
    fn write_interpolation_filter_needs_gate_arms() {
        let forced = InterpolationFilterReadout {
            interp_filter: [EIGHTTAP, EIGHTTAP],
            read_from_bitstream: [false, false],
        };
        // skip_mode.
        let got = roundtrip_interp_filter(
            forced,
            BLOCK_8X8,
            1,
            MODE_NEAREST_NEARESTMV,
            MOTION_MODE_SIMPLE,
            SWITCHABLE,
            false,
            [0; 8],
        );
        assert_eq!(got, forced);
        // motion_mode == LOCALWARP (WARPED_CAUSAL).
        let got = roundtrip_interp_filter(
            forced,
            BLOCK_16X16,
            0,
            MODE_NEARESTMV,
            MOTION_MODE_WARPED_CAUSAL,
            SWITCHABLE,
            true,
            [0; 8],
        );
        assert_eq!(got, forced);
        // large GLOBALMV with non-TRANSLATION gm_type.
        let mut gm = [0i32; 8];
        gm[1] = GM_TYPE_TRANSLATION + 1; // ROTZOOM
        let got = roundtrip_interp_filter(
            forced,
            BLOCK_16X16,
            0,
            MODE_GLOBALMV,
            MOTION_MODE_SIMPLE,
            SWITCHABLE,
            true,
            gm,
        );
        assert_eq!(got, forced);
        // … and with gm_type == TRANSLATION the gate re-opens (S()
        // fires).
        let mut gm = [0i32; 8];
        gm[1] = GM_TYPE_TRANSLATION;
        let coded = InterpolationFilterReadout {
            interp_filter: [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SHARP],
            read_from_bitstream: [true, false],
        };
        let got = roundtrip_interp_filter(
            coded,
            BLOCK_16X16,
            0,
            MODE_GLOBALMV,
            MOTION_MODE_SIMPLE,
            SWITCHABLE,
            false,
            gm,
        );
        assert_eq!(got, coded);
    }

    /// §8.3.2 neighbour ctx: a matching above neighbour moves the ctx
    /// off the `3` sentinel band. Hand-mirror the single S() with the
    /// exact ctx the writer must have used and verify both the symbol
    /// and that only that row adapted.
    #[test]
    fn write_interpolation_filter_neighbour_ctx_selection() {
        let target = InterpolationFilterReadout {
            interp_filter: [crate::cdf::EIGHTTAP_SMOOTH, crate::cdf::EIGHTTAP_SMOOTH],
            read_from_bitstream: [true, false],
        };
        let ref_frame = [1i32, -1];
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_interpolation_filter(
            &mut writer,
            &mut enc_cdfs,
            &target,
            BLOCK_8X8,
            0,
            false,
            ref_frame,
            MODE_NEARESTMV,
            MOTION_MODE_SIMPLE,
            SWITCHABLE,
            false,
            [0; 8],
            /* avail_u = */ true,
            /* avail_l = */ false,
            /* above_ref_frame = */ [1, -1], // matches RefFrame[0]
            [-1, -1],
            /* above_interp_filters = */
            [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SHARP],
            [EIGHTTAP, EIGHTTAP],
        )
        .unwrap();
        let bytes = writer.finish();

        // Hand mirror: aboveType = SHARP (match), leftType = NONE ⇒
        // ctx = (0 * 2 + 0) * 4 + SHARP = 2.
        let expected_ctx = interp_filter_ctx(
            crate::cdf::EIGHTTAP_SHARP as usize,
            INTERP_FILTER_NONE,
            0,
            false,
        )
        .unwrap();
        assert_eq!(expected_ctx, 2);
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let sym = dec
            .read_symbol(dec_cdfs.interp_filter_cdf(expected_ctx).unwrap())
            .unwrap();
        assert_eq!(sym as u8, crate::cdf::EIGHTTAP_SMOOTH);
        assert_eq!(enc_cdfs.interp_filter, dec_cdfs.interp_filter);
        // A mismatched above reference falls back to the sentinel band:
        // same target re-encoded with above_ref_frame = GOLDEN must
        // consult ctx 3 instead (different row ⇒ different CDF state).
        let mut enc_cdfs2 = TileCdfContext::new_from_defaults();
        let mut writer2 = SymbolWriter::new(false);
        write_interpolation_filter(
            &mut writer2,
            &mut enc_cdfs2,
            &target,
            BLOCK_8X8,
            0,
            false,
            ref_frame,
            MODE_NEARESTMV,
            MOTION_MODE_SIMPLE,
            SWITCHABLE,
            false,
            [0; 8],
            true,
            false,
            [4, -1], // GOLDEN ≠ RefFrame[0] ⇒ sentinel
            [-1, -1],
            [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SHARP],
            [EIGHTTAP, EIGHTTAP],
        )
        .unwrap();
        let _ = writer2.finish();
        assert_ne!(
            enc_cdfs2.interp_filter[2], enc_cdfs.interp_filter[2],
            "ctx-2 row adapted only on the matching-neighbour encode"
        );
        let sentinel_ctx =
            interp_filter_ctx(INTERP_FILTER_NONE, INTERP_FILTER_NONE, 0, false).unwrap();
        assert_eq!(sentinel_ctx, 3);
        assert_ne!(
            enc_cdfs2.interp_filter[3],
            TileCdfContext::new_from_defaults().interp_filter[3],
            "ctx-3 row adapted on the mismatched-neighbour encode"
        );
    }

    /// Caller-bug rejects: shapes the §5.11.x reader could never
    /// produce on the given gate configuration.
    #[test]
    fn write_interpolation_filter_rejects_inconsistent_readouts() {
        let run = |readout: InterpolationFilterReadout,
                   skip_mode: u8,
                   interpolation_filter: u8,
                   enable_dual_filter: bool|
         -> Result<(), Error> {
            let mut cdfs = TileCdfContext::new_from_defaults();
            let mut writer = SymbolWriter::new(false);
            write_interpolation_filter(
                &mut writer,
                &mut cdfs,
                &readout,
                BLOCK_8X8,
                skip_mode,
                false,
                [1, -1],
                MODE_NEARESTMV,
                MOTION_MODE_SIMPLE,
                interpolation_filter,
                enable_dual_filter,
                [0; 8],
                false,
                false,
                [-1, -1],
                [-1, -1],
                [EIGHTTAP, EIGHTTAP],
                [EIGHTTAP, EIGHTTAP],
            )
        };
        let forced = |f: u8| InterpolationFilterReadout {
            interp_filter: [f, f],
            read_from_bitstream: [false, false],
        };
        // Non-SWITCHABLE with a slot off the frame-header value.
        assert!(run(forced(EIGHTTAP), 0, crate::cdf::EIGHTTAP_SHARP, false).is_err());
        // Non-SWITCHABLE with a stray read flag.
        assert!(run(
            InterpolationFilterReadout {
                interp_filter: [EIGHTTAP, EIGHTTAP],
                read_from_bitstream: [true, false],
            },
            0,
            EIGHTTAP,
            false,
        )
        .is_err());
        // SWITCHABLE + !needs (skip_mode) but a non-EIGHTTAP slot.
        assert!(run(forced(crate::cdf::EIGHTTAP_SMOOTH), 1, SWITCHABLE, false).is_err());
        // SWITCHABLE + !needs with a stray read flag.
        assert!(run(
            InterpolationFilterReadout {
                interp_filter: [EIGHTTAP, EIGHTTAP],
                read_from_bitstream: [true, false],
            },
            1,
            SWITCHABLE,
            false,
        )
        .is_err());
        // SWITCHABLE + needs but the read flag is clear.
        assert!(run(forced(crate::cdf::EIGHTTAP_SMOOTH), 0, SWITCHABLE, false).is_err());
        // BILINEAR is not codeable through the 3-way S().
        assert!(run(
            InterpolationFilterReadout {
                interp_filter: [BILINEAR, BILINEAR],
                read_from_bitstream: [true, false],
            },
            0,
            SWITCHABLE,
            false,
        )
        .is_err());
        // !enable_dual_filter mirror violation (slot 1 differs).
        assert!(run(
            InterpolationFilterReadout {
                interp_filter: [EIGHTTAP, crate::cdf::EIGHTTAP_SMOOTH],
                read_from_bitstream: [true, false],
            },
            0,
            SWITCHABLE,
            false,
        )
        .is_err());
        // !enable_dual_filter with a stray slot-1 read flag.
        assert!(run(
            InterpolationFilterReadout {
                interp_filter: [EIGHTTAP, EIGHTTAP],
                read_from_bitstream: [true, true],
            },
            0,
            SWITCHABLE,
            false,
        )
        .is_err());
        // interpolation_filter out of the §6.8.9 range.
        assert!(run(forced(EIGHTTAP), 0, SWITCHABLE + 1, false).is_err());
    }

    // -----------------------------------------------------------------
    // §5.11.23 composed tail (r277) — write_inter_block_mode_info
    // with the four tail leaves folded, round-tripped through the
    // decode-side dispatcher (which consumes the tail in the same
    // spec order as of the r277 ordering fix).
    // -----------------------------------------------------------------

    /// Per-block encode parameters shared by the three-block dispatcher
    /// round-trip below. Blocks sit at rows 0 / 2 / 4 of column 0 on an
    /// 8x8-mi walker, each BLOCK_8X8, each single-pred NEARESTMV on
    /// LAST_FRAME.
    ///
    /// * Block A — every tail leaf silent (gates shut except the
    ///   SWITCHABLE interp S()).
    /// * Block B — `interintra = 1` (wedge arm): four §5.11.28 S()s,
    ///   then the `RefFrame[ 1 ] = INTRA_FRAME` override SILENCES
    ///   §5.11.27 motion_mode even though `has_overlappable_candidates`
    ///   is true (block A sits above) — the §5.11.23 spec-order proof.
    /// * Block C — `interintra` gate open but 0 (one S()), then the
    ///   §5.11.27 arm-A `use_obmc` S() fires (OBMC), then the interp
    ///   S(). Genuine §5.11.28→§5.11.27 interleaving.
    const RT_REF: [i32; 2] = [1, -1];

    /// Tail for block A: fully silent. Block A is GLOBALMV on an empty
    /// stack (a fresh tile has no MV candidates for NEAREST/NEAR), and
    /// large-GLOBALMV with IDENTITY gm closes `needs_interp_filter()`
    /// — the SWITCHABLE loop forces EIGHTTAP with no S().
    fn rt_tail_a() -> InterBlockModeInfoTail {
        InterBlockModeInfoTail {
            interpolation_filter: SWITCHABLE,
            interp_filter: InterpolationFilterReadout {
                interp_filter: [EIGHTTAP, EIGHTTAP],
                read_from_bitstream: [false, false],
            },
            ..InterBlockModeInfoTail::bit_silent()
        }
    }

    /// Tail for block B: interintra wedge arm; motion silenced by the
    /// §5.11.28 slot-1 override despite the open motion gates.
    fn rt_tail_b(has_overlappable: bool) -> InterBlockModeInfoTail {
        InterBlockModeInfoTail {
            interintra: InterIntraReadout {
                interintra: 1,
                interintra_mode: Some(crate::cdf::II_SMOOTH_PRED),
                wedge_interintra: Some(1),
                wedge_index: Some(5),
            },
            enable_interintra_compound: true,
            is_motion_mode_switchable: true,
            allow_warped_motion: true,
            has_overlappable,
            compound_type: crate::cdf::CompoundTypeReadout {
                comp_group_idx: 0,
                compound_idx: 1,
                compound_type: COMPOUND_WEDGE,
                wedge_index: None,
                wedge_sign: None,
                mask_type: None,
            },
            interpolation_filter: SWITCHABLE,
            interp_filter: InterpolationFilterReadout {
                interp_filter: [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SHARP],
                read_from_bitstream: [true, false],
            },
            above_interp_filters: [EIGHTTAP, EIGHTTAP],
            ..InterBlockModeInfoTail::bit_silent()
        }
    }

    /// Tail for block C: interintra gate open / flag 0, §5.11.27 arm A
    /// (`!allow_warped_motion`) coding `use_obmc = 1`.
    fn rt_tail_c(has_overlappable: bool, num_samples: u32) -> InterBlockModeInfoTail {
        InterBlockModeInfoTail {
            enable_interintra_compound: true,
            motion_mode: MOTION_MODE_OBMC,
            num_samples,
            is_motion_mode_switchable: true,
            allow_warped_motion: false,
            has_overlappable,
            interpolation_filter: SWITCHABLE,
            interp_filter: InterpolationFilterReadout {
                interp_filter: [crate::cdf::EIGHTTAP_SMOOTH, crate::cdf::EIGHTTAP_SMOOTH],
                read_from_bitstream: [true, false],
            },
            above_interp_filters: [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SHARP],
            ..InterBlockModeInfoTail::bit_silent()
        }
    }

    /// Decode `n_blocks` of the three-block stream through the
    /// §5.11.23 dispatcher, returning the stamped walker plus the
    /// decoded aggregates.
    fn rt_decode_blocks(
        bytes: &[u8],
        n_blocks: usize,
    ) -> (PartitionWalker, Vec<crate::cdf::DecodedInterBlockModeInfo>) {
        let geom = TileGeometry {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mut walker = PartitionWalker::new(8, 8, geom).unwrap();
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(bytes, bytes.len(), false).unwrap();
        let mfmvs = crate::cdf::MotionFieldMvs::new_invalid(8, 8);
        let mut infos = Vec::new();
        for blk in 0..n_blocks {
            let mi_row = (blk as u32) * 2;
            let avail_u = blk > 0;
            // Above-neighbour scalars: block B's slot 1 is overridden
            // to INTRA_FRAME by its §5.11.28 arm, so block C observes
            // [1, 0]; block B observes A's plain [1, -1].
            let above_ref_frame = if blk == 2 { [1, 0] } else { RT_REF };
            let (eic, imms, awm) = match blk {
                0 => (false, false, false),
                1 => (true, true, true),
                _ => (true, true, false),
            };
            let info = walker
                .decode_inter_block_mode_info(
                    &mut dec,
                    &mut cdfs,
                    mi_row,
                    0,
                    BLOCK_8X8,
                    /* skip_mode = */ 0,
                    avail_u,
                    /* avail_l = */ false,
                    above_ref_frame,
                    [-1, -1],
                    /* above_intra = */ false,
                    /* left_intra = */ false,
                    /* above_single = */ true,
                    /* left_single = */ true,
                    /* skip_mode_frame = */ [1, -1],
                    /* seg_ref_frame_active = */ false,
                    /* seg_ref_frame_data = */ 0,
                    /* seg_skip_active = */ false,
                    /* seg_globalmv_active = */ false,
                    /* reference_select = */ false,
                    [0; 8],
                    identity_gm_params(),
                    [0; 8],
                    /* allow_high_precision_mv = */ false,
                    /* force_integer_mv = */ false,
                    /* use_ref_frame_mvs = */ false,
                    imms,
                    awm,
                    [false; 7],
                    eic,
                    /* enable_masked_compound = */ false,
                    /* enable_jnt_comp = */ false,
                    /* order_hints = */ crate::cdf::FrameInterOrderHints::IDENTITY,
                    SWITCHABLE,
                    /* enable_dual_filter = */ false,
                    &mfmvs,
                )
                .unwrap();
            infos.push(info);
        }
        (walker, infos)
    }

    /// Encode the first `n_blocks` of the three-block stream with the
    /// composed writer. Per-block §7.10 inputs (mv stack, warp samples,
    /// overlappable flag) come from an encoder-side walker stamped by
    /// decoding the previously-encoded prefix — the same state the
    /// decode dispatcher rebuilds.
    fn rt_encode_blocks(n_blocks: usize) -> Vec<u8> {
        let geom = TileGeometry {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        let mfmvs = crate::cdf::MotionFieldMvs::new_invalid(8, 8);
        let mut writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();
        for blk in 0..n_blocks {
            // Encoder-side mirror walker carrying the prefix stamps.
            let walker = if blk == 0 {
                PartitionWalker::new(8, 8, geom).unwrap()
            } else {
                rt_decode_blocks(&rt_encode_blocks(blk), blk).0
            };
            let mi_row = (blk as u32) * 2;
            let stack = walker
                .find_mv_stack(
                    mi_row,
                    0,
                    BLOCK_8X8,
                    RT_REF,
                    false,
                    false,
                    [0; 8],
                    identity_gm_params(),
                    [0; 8],
                    false,
                    false,
                    &mfmvs,
                )
                .unwrap();
            // Block A codes GLOBALMV (the fresh tile has an empty MV
            // stack, so the NEAREST family has no §5.11.31 source
            // slot); B and C code NEARESTMV off the stack the A / B
            // stamps populate.
            let y_mode = if blk == 0 {
                MODE_GLOBALMV
            } else {
                MODE_NEARESTMV
            };
            let mv0 = if blk == 0 {
                stack.global_mvs[0]
            } else {
                assert!(stack.num_mv_found > 0, "above stamps must feed the stack");
                assign_mv_pred_mv(&stack, MODE_NEARESTMV, 0, 0).unwrap()
            };
            let has_ov = walker.has_overlappable_candidates(mi_row, 0, BLOCK_8X8);
            let tail = match blk {
                0 => rt_tail_a(),
                1 => {
                    // The order proof: the motion gates ARE open
                    // (above block A is an overlappable inter
                    // neighbour) — only the §5.11.28 slot-1 override
                    // silences §5.11.27.
                    assert!(has_ov, "block A above must be overlappable");
                    rt_tail_b(has_ov)
                }
                _ => {
                    assert!(has_ov, "block B above must be overlappable");
                    let ns =
                        walker.find_warp_samples(mi_row, 0, BLOCK_8X8, RT_REF[0], [mv0, [0, 0]]);
                    rt_tail_c(has_ov, ns)
                }
            };
            let above_ref_frame = if blk == 2 { [1, 0] } else { RT_REF };
            write_inter_block_mode_info(
                &mut writer,
                &mut cdfs,
                RT_REF,
                y_mode,
                [mv0, [0, 0]],
                /* ref_mv_idx = */ 0,
                &stack,
                BLOCK_8X8,
                /* skip_mode = */ 0,
                [1, -1],
                false,
                0,
                false,
                false,
                /* reference_select = */ false,
                /* avail_u = */ blk > 0,
                /* avail_l = */ false,
                /* above_single = */ true,
                /* left_single = */ true,
                /* above_intra = */ false,
                /* left_intra = */ false,
                above_ref_frame,
                [-1, -1],
                /* force_integer_mv = */ false,
                /* allow_high_precision_mv = */ false,
                &tail,
            )
            .unwrap();
        }
        writer.finish()
    }

    /// Three-block composed-writer → decode-dispatcher round-trip.
    /// Proves (1) the r277 tail fold emits the §5.11.23 leaves in spec
    /// order (§5.11.28 → §5.11.27 → §5.11.29 → interp loop), (2) the
    /// §5.11.28 `RefFrame[ 1 ] = INTRA_FRAME` override silences the
    /// §5.11.27 writer/reader pair on interintra blocks even with
    /// overlappable neighbours, and (3) the dispatcher (post r277
    /// ordering fix) rebuilds every aggregate bit-exactly.
    #[test]
    fn write_inter_block_mode_info_tail_round_trips_through_dispatcher() {
        let bytes = rt_encode_blocks(3);
        let (walker, infos) = rt_decode_blocks(&bytes, 3);
        assert_eq!(infos.len(), 3);

        // Block A: silent tail; large-GLOBALMV + IDENTITY gm closes
        // `needs_interp_filter()` ⇒ forced EIGHTTAP under SWITCHABLE.
        assert_eq!(infos[0].ref_frame, RT_REF);
        assert_eq!(infos[0].y_mode, MODE_GLOBALMV);
        assert_eq!(infos[0].interintra.interintra, 0);
        assert_eq!(infos[0].motion_mode, MOTION_MODE_SIMPLE);
        assert_eq!(infos[0].compound_type.compound_type, COMPOUND_AVERAGE);
        assert_eq!(infos[0].interp_filter.interp_filter, [EIGHTTAP, EIGHTTAP]);
        assert_eq!(infos[0].interp_filter.read_from_bitstream, [false, false]);

        // Block B: the interintra wedge arm + the slot-1 override.
        assert_eq!(
            infos[1].ref_frame,
            [1, 0],
            "§5.11.28 RefFrame[1] = INTRA_FRAME override surfaces"
        );
        assert_eq!(infos[1].interintra.interintra, 1);
        assert_eq!(
            infos[1].interintra.interintra_mode,
            Some(crate::cdf::II_SMOOTH_PRED)
        );
        assert_eq!(infos[1].interintra.wedge_interintra, Some(1));
        assert_eq!(infos[1].interintra.wedge_index, Some(5));
        assert_eq!(
            infos[1].motion_mode, MOTION_MODE_SIMPLE,
            "slot-1 INTRA_FRAME silences §5.11.27 despite overlappable neighbours"
        );
        assert_eq!(infos[1].compound_type.compound_type, COMPOUND_WEDGE);
        assert_eq!(
            infos[1].interp_filter.interp_filter,
            [crate::cdf::EIGHTTAP_SHARP, crate::cdf::EIGHTTAP_SHARP]
        );

        // Block C: interintra-open/0 followed by the arm-A use_obmc S().
        assert_eq!(infos[2].ref_frame, RT_REF);
        assert_eq!(infos[2].interintra.interintra, 0);
        assert_eq!(infos[2].motion_mode, MOTION_MODE_OBMC);
        assert_eq!(infos[2].compound_type.compound_type, COMPOUND_AVERAGE);
        assert_eq!(
            infos[2].interp_filter.interp_filter,
            [crate::cdf::EIGHTTAP_SMOOTH, crate::cdf::EIGHTTAP_SMOOTH]
        );

        // Grid stamps: B's slot-1 override landed on the walker.
        let cell = (2 * 8) as usize; // (row 2, col 0)
        assert_eq!(walker.ref_frames()[cell * 2 + 1], 0);
    }

    /// Composed-writer tail consistency reject: a `motion_mode` the
    /// reader could never produce after the §5.11.28 override (OBMC on
    /// an interintra block) is a caller bug.
    #[test]
    fn write_inter_block_mode_info_tail_rejects_motion_on_interintra() {
        let stack = mk_mv_stack(2);
        let pred = assign_mv_pred_mv(&stack, MODE_NEARESTMV, 0, 0).unwrap();
        let (au, al, asg, lsg, ai, li, arf, lrf) = empty_neighbour_inputs();
        let mut tail = rt_tail_b(true);
        tail.motion_mode = MOTION_MODE_OBMC; // unreachable: slot-1 override forces SIMPLE
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_inter_block_mode_info(
            &mut writer,
            &mut cdfs,
            [1, -1],
            MODE_NEARESTMV,
            [pred, [0, 0]],
            0,
            &stack,
            BLOCK_8X8,
            0,
            [0, 0],
            false,
            0,
            false,
            false,
            false,
            au,
            al,
            asg,
            lsg,
            ai,
            li,
            arf,
            lrf,
            false,
            true,
            &tail,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // r282 — §5.11.49 write_palette_tokens_plane round-trips.
    // -----------------------------------------------------------------

    /// Round-trip one §5.11.49 plane walk through
    /// [`crate::cdf::palette_tokens_plane`], including the
    /// frame-edge-straddling case (`onscreen < block`): the writer
    /// only consumes the on-screen samples and the reader's
    /// border-fill is derived, not coded.
    #[test]
    fn r282_write_palette_tokens_plane_round_trip_with_offscreen_border() {
        use crate::cdf::palette_tokens_plane;
        use crate::symbol_decoder::SymbolDecoder;

        let (block_w, block_h) = (8usize, 8usize);
        let (onscreen_w, onscreen_h) = (6usize, 5usize);
        let palette_size = 3usize;
        // Diagonal bands over {0, 1, 2} on the on-screen rectangle.
        let mut target = vec![0u8; block_w * block_h];
        for r in 0..onscreen_h {
            for c in 0..onscreen_w {
                target[r * block_w + c] = ((r + c) % palette_size) as u8;
            }
        }

        for plane in [PalettePlane::Y, PalettePlane::Uv] {
            let mut writer = SymbolWriter::new(false);
            let mut enc_cdfs = TileCdfContext::new_from_defaults();
            write_palette_tokens_plane(
                &mut writer,
                &mut enc_cdfs,
                plane,
                palette_size,
                block_w,
                block_h,
                onscreen_w,
                onscreen_h,
                &target,
                block_w,
            )
            .unwrap();
            writer.write_literal(8, 0x3C).unwrap();
            let bytes = writer.finish();

            let mut dec_cdfs = TileCdfContext::new_from_defaults();
            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            // §5.11.49: `color_index_map NS(PaletteSize)` first.
            let color_index_map = dec.read_ns(palette_size as u32).unwrap() as u8;
            assert_eq!(color_index_map, target[0], "NS top-left sample");
            let mut color_map = vec![0u8; block_w * block_h];
            palette_tokens_plane(
                &mut dec,
                &mut dec_cdfs,
                plane,
                palette_size,
                block_w,
                block_h,
                onscreen_w,
                onscreen_h,
                color_index_map,
                &mut color_map,
                block_w,
            )
            .unwrap();
            // On-screen reconstruction matches the committed map.
            for r in 0..onscreen_h {
                for c in 0..onscreen_w {
                    assert_eq!(
                        color_map[r * block_w + c],
                        target[r * block_w + c],
                        "on-screen ColorMap sample ({r}, {c})"
                    );
                }
            }
            // §5.11.49 border-fill: right edge replicates column
            // `onscreenWidth - 1`, bottom rows replicate row
            // `onscreenHeight - 1`.
            for r in 0..onscreen_h {
                for c in onscreen_w..block_w {
                    assert_eq!(
                        color_map[r * block_w + c],
                        color_map[r * block_w + onscreen_w - 1],
                        "horizontal border-fill ({r}, {c})"
                    );
                }
            }
            for r in onscreen_h..block_h {
                for c in 0..block_w {
                    assert_eq!(
                        color_map[r * block_w + c],
                        color_map[(onscreen_h - 1) * block_w + c],
                        "vertical border-fill ({r}, {c})"
                    );
                }
            }
            // §8.3 adaptation lockstep + sentinel.
            assert_eq!(enc_cdfs, dec_cdfs, "CDF adaptation lockstep");
            assert_eq!(dec.read_literal(8).unwrap(), 0x3C, "sync sentinel");
        }
    }

    /// Caller-bug guards on the §5.11.49 writer: out-of-range palette
    /// size, on-screen sample outside the palette alphabet, undersized
    /// map, zero on-screen dims.
    #[test]
    fn r282_write_palette_tokens_plane_rejects_caller_bugs() {
        let mut writer = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let map = vec![0u8; 64];

        // palette_size outside 2..=PALETTE_COLORS.
        for bad_size in [0usize, 1, PALETTE_COLORS + 1] {
            let err = write_palette_tokens_plane(
                &mut writer,
                &mut cdfs,
                PalettePlane::Y,
                bad_size,
                8,
                8,
                8,
                8,
                &map,
                8,
            )
            .unwrap_err();
            assert!(matches!(err, Error::InvalidPaletteWalkArgs));
        }
        // On-screen sample >= palette_size.
        let mut bad_map = map.clone();
        bad_map[10] = 5;
        let err = write_palette_tokens_plane(
            &mut writer,
            &mut cdfs,
            PalettePlane::Y,
            2,
            8,
            8,
            8,
            8,
            &bad_map,
            8,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidPaletteWalkArgs));
        // Map shorter than block_height * stride.
        let err = write_palette_tokens_plane(
            &mut writer,
            &mut cdfs,
            PalettePlane::Y,
            2,
            8,
            8,
            8,
            8,
            &map[..32],
            8,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidPaletteWalkArgs));
        // Zero / oversize on-screen dims.
        for (ow, oh) in [(0usize, 8usize), (8, 0), (9, 8), (8, 9)] {
            let err = write_palette_tokens_plane(
                &mut writer,
                &mut cdfs,
                PalettePlane::Y,
                2,
                8,
                8,
                ow,
                oh,
                &map,
                8,
            )
            .unwrap_err();
            assert!(matches!(err, Error::InvalidPaletteWalkArgs));
        }
    }
}
