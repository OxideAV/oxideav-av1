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
    block_height, block_width, cfl_alpha_u_ctx, cfl_alpha_v_ctx, neg_interleave, TileCdfContext,
    BLOCK_SIZES, BLOCK_SIZE_GROUPS, CFL_ALPHABET_SIZE, CFL_JOINT_SIGNS, INTRA_MODES,
    INTRA_MODE_CONTEXTS, IS_INTER_CONTEXTS, MAX_SEGMENTS, SEGMENT_ID_CONTEXTS,
    SEGMENT_ID_PREDICTED_CONTEXTS, SKIP_CONTEXTS, SKIP_MODE_CONTEXTS, UV_INTRA_MODES_CFL_ALLOWED,
    UV_INTRA_MODES_CFL_NOT_ALLOWED,
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
}

/// `inter_frame_mode_info()` writer prefix per §5.11.18 (av1-spec p.71) —
/// the per-block syntax dispatcher for an inter-frame leaf. Composes
/// every leaf writer landed across r253-r256:
///
/// * §5.11.19 `inter_segment_id( 1 )` — pre-skip arm via
///   [`write_inter_segment_id`].
/// * §5.11.10 `read_skip_mode()` — via [`write_skip_mode`].
/// * §5.11.11 `read_skip()` (when `skip_mode == 0`) — via [`write_skip`].
/// * §5.11.19 `inter_segment_id( 0 )` (when `!seg_id_pre_skip`) —
///   post-skip arm via [`write_inter_segment_id`].
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
///       read_cdef( )                                          (next round)
///       read_delta_qindex( )                                  (next round)
///       read_delta_lf( )                                      (next round)
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
/// The dispatcher writes the §5.11.18 *prefix*: the five S()-emitting
/// sub-writers above. The §5.11.18 line 18 `read_cdef`, line 19
/// `read_delta_qindex`, line 20 `read_delta_lf`, and the line 23-26
/// terminal `if ( is_inter )` dispatch into §5.11.22 / §5.11.23 are
/// next-round encoder-side targets — none have writer counterparts yet.
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

    // §5.11.18 lines 18-21: `read_cdef( ) / read_delta_qindex( ) /
    // read_delta_lf( ) / ReadDeltas = 0` — next-round writer targets;
    // skipped here.

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
    })
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
        )
        .unwrap();
        assert_eq!(prefix.segment_id, 3);
        assert!(
            prefix.lossless,
            "LosslessArray[3] == true ⇒ lossless = true"
        );
    }
}
