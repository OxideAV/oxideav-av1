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
    cfl_alpha_u_ctx, cfl_alpha_v_ctx, neg_deinterleave, TileCdfContext, BLOCK_SIZE_GROUPS,
    CFL_ALPHABET_SIZE, CFL_JOINT_SIGNS, INTRA_MODES, INTRA_MODE_CONTEXTS, MAX_SEGMENTS,
    SKIP_CONTEXTS, UV_INTRA_MODES_CFL_ALLOWED, UV_INTRA_MODES_CFL_NOT_ALLOWED,
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
/// `read_segment_id()` then dispatches per §5.11.9:
///   * `if ( skip )` arm — `segment_id = pred`, no bit read.
///   * `else`        arm — `diff S()` against `TileSegmentIdCdf[ ctx ]`
///     and `segment_id = neg_deinterleave( diff, pred, max )` with
///     `max = LastActiveSegId + 1`.
///
/// The inverse of `neg_deinterleave` is found by search: for a fixed
/// `pred` / `max`, `neg_deinterleave` is a bijection on `0..max`, so
/// the encoder searches `diff ∈ 0..max` for the one whose forward
/// `neg_deinterleave( diff, pred, max ) == segment_id`. Since
/// `max = LastActiveSegId + 1 <= MAX_SEGMENTS = 8` the search is O(8).
///
/// Parameters mirror [`crate::cdf::PartitionWalker::decode_intra_segment_id`]
/// plus `pred` (the §5.11.9 neighbour-cascade value the caller already
/// has on hand from the encoder's own grid walk) and `ctx` (the
/// §8.3.2 `segment_id_ctx` — see [`crate::cdf::segment_id_ctx`]).
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
    // Range guards mirroring the decoder.
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
    // §5.11.9 skip-block short-circuit: `segment_id = pred`, no bits.
    if skip != 0 {
        if segment_id != pred {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }
    if ctx >= 3 {
        // SEGMENT_ID_CONTEXTS = 3.
        return Err(Error::PartitionWalkOutOfRange);
    }
    // Solve `neg_deinterleave( diff, pred, max ) == segment_id` for diff.
    let mut diff_opt: Option<u32> = None;
    for cand in 0..max {
        if neg_deinterleave(cand, pred as u32, max) == segment_id as u32 {
            diff_opt = Some(cand);
            break;
        }
    }
    let diff = diff_opt.ok_or(Error::PartitionWalkOutOfRange)?;
    let cdf = cdfs.segment_id_cdf(ctx);
    writer.write_symbol(diff, cdf)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        cfl_allowed_for_uv_mode, intra_mode_ctx, size_group, skip_ctx, PartitionWalker,
        TileGeometry, BLOCK_16X16, BLOCK_8X8, DC_PRED, V_PRED,
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
}
