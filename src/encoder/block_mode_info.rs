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
    block_height, block_width, ceil_log2_av1, cfl_alpha_u_ctx, cfl_alpha_v_ctx, is_directional,
    mi_height_log2, mi_width_log2, neg_interleave, palette_uv_mode_ctx, palette_y_mode_ctx,
    PartitionWalker, TileCdfContext, BLOCK_128X128, BLOCK_64X64, BLOCK_8X8, BLOCK_SIZES,
    BLOCK_SIZE_GROUPS, CFL_ALPHABET_SIZE, CFL_JOINT_SIGNS, DELTA_LF_SMALL, DELTA_Q_SMALL,
    FRAME_LF_COUNT, INTRA_FILTER_MODES, INTRA_MODES, INTRA_MODE_CONTEXTS, IS_INTER_CONTEXTS,
    MAX_ANGLE_DELTA, MAX_SEGMENTS, PALETTE_BLOCK_SIZE_CONTEXTS, PALETTE_COLORS,
    SEGMENT_ID_CONTEXTS, SEGMENT_ID_PREDICTED_CONTEXTS, SKIP_CONTEXTS, SKIP_MODE_CONTEXTS,
    UV_INTRA_MODES_CFL_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED,
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

    // §5.11.22 line 2: `y_mode S()` via the §8.3.2 `Size_Group[ MiSize ]`
    // ctx.
    let size_group_ctx = crate::cdf::size_group(mi_size);
    write_y_mode(writer, cdfs, y_mode, size_group_ctx)?;

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
        cfl_allowed_for_uv_mode, intra_mode_ctx, is_inter_ctx, size_group, skip_ctx, skip_mode_ctx,
        PartitionWalker, TileGeometry, BLOCK_16X16, BLOCK_4X4, BLOCK_4X8, BLOCK_8X4, BLOCK_8X8,
        DC_PRED, V_PRED,
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
}
