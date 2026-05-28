//! Per-coefficient **writers** for the §5.11.39 `coefficients()` body —
//! arc 6 (round 212) first slice: `txb_skip` (the `all_zero` symbol),
//! `eob_pt` (the EOB position class + refinement bits), and `dc_sign`.
//!
//! These are the encoder counterparts to the §5.11.39 reader inside
//! [`crate::cdf::PartitionWalker::coefficients`]:
//!
//!   * The decoder's first §8.2.6 `S()` is `all_zero` against
//!     `TileTxbSkipCdf[ txSzCtx ][ ctx ]` (§8.3.2). Writer:
//!     [`write_txb_skip`].
//!   * After the `all_zero == 0` arm, the decoder reads one of
//!     `eob_pt_16` … `eob_pt_1024` (`S()` against
//!     `TileEobPt{N}Cdf[ ptype ][ ctx ]` per the §8.3.2 `eobMultisize`
//!     selector), then an optional `eob_extra` `S()` followed by an
//!     `eob_extra_bit` raw `L(1)` loop to refine `eob` toward its final
//!     value. Writer: [`write_eob_pt`].
//!   * Inside the forward-scan loop, the first non-zero coefficient
//!     (`c == 0` arm) reads one `dc_sign` `S()` against
//!     `TileDcSignCdf[ ptype ][ ctx ]` (§8.3.2). Writer:
//!     [`write_dc_sign`].
//!
//! Scope of this arc is intentionally tight: the per-coefficient
//! `coeff_base` / `coeff_base_eob` / `coeff_br` chain and the
//! per-magnitude `golomb_length_bit` / `golomb_data_bit` tail are out
//! of scope for r212 — they sit on top of these primitives in a
//! subsequent arc (and on top of the `Quant[]` context plumbing the
//! §8.3.2 `get_coeff_base_ctx` / `get_coeff_base_eob_ctx` / `get_br_ctx`
//! derivations want).
//!
//! ## Stateless on purpose
//!
//! Mirroring [`block_mode_info`]'s pattern, every writer here takes its
//! §8.3.2 *context indices* and (for `eob_pt`) the per-`eobMultisize`
//! `tx_size` / `tx_class` / `plane` / `is_inter` axes as inputs rather
//! than a [`PartitionWalker`] reference. The caller derives the §8.3.2
//! ctx (today: in tests; tomorrow: in the §5.11.39 driver loop the
//! follow-on arc lands) and feeds the same ctx the decoder side would
//! derive on the corresponding [`PartitionWalker::coefficients`] read.
//!
//! ## Spec provenance
//!
//! Sourced from `docs/video/av1/av1-spec.txt`:
//!   * §5.11.39 residual / coefficients() body (p.88–93)
//!   * §6.5.10 — `all_zero`, `eob_pt_*`, `dc_sign` syntax
//!   * §8.3.2 context derivations (p.361–378)
//!   * §9.4 `Default_Txb_Skip_Cdf` / `Default_Eob_Pt_*_Cdf` /
//!     `Default_Dc_Sign_Cdf`
//!
//! [`PartitionWalker`]: crate::cdf::PartitionWalker
//! [`PartitionWalker::coefficients`]: crate::cdf::PartitionWalker::coefficients
//! [`block_mode_info`]: crate::encoder::block_mode_info

use crate::cdf::{
    TileCdfContext, DC_SIGN_CONTEXTS, EOB_COEF_CONTEXTS, TXB_SKIP_CONTEXTS, TX_SIZES, TX_SIZES_ALL,
    TX_SIZE_SQR_UP, TX_WIDTH, TX_WIDTH_LOG2,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::Error;

/// `txb_skip` (== `all_zero`) writer per §5.11.39 line 13 (av1-spec
/// p.89) and §8.3.2 (av1-spec p.371).
///
/// Spec body (extracted):
/// ```text
///   all_zero                                                   S()
/// ```
///
/// `all_zero` is a single §8.2.6 binary symbol coded against
/// `TileTxbSkipCdf[ txSzCtx ][ ctx ]`. `txSzCtx` is the §5.11.39
/// line-4 `(Tx_Size_Sqr[txSz] + Tx_Size_Sqr_Up[txSz] + 1) >> 1`
/// derivation — passed pre-computed by the caller as `tx_sz_ctx`
/// (`0..TX_SIZES = 5`); `ctx` is the §8.3.2 `all_zero` ctx in
/// `0..TXB_SKIP_CONTEXTS = 13`.
///
/// `all_zero` MUST be `0` or `1`.
///
/// The §8.3 CDF adaptation runs in lockstep with
/// [`crate::cdf::PartitionWalker::coefficients`]'s first `S()` read,
/// so the encoder + decoder CDFs stay synchronised across a
/// multi-symbol tile payload.
pub fn write_txb_skip(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    all_zero: u8,
    tx_sz_ctx: usize,
    ctx: usize,
) -> Result<(), Error> {
    if all_zero > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_sz_ctx >= TX_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ctx >= TXB_SKIP_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs
        .txb_skip_cdf(tx_sz_ctx, ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(all_zero as u32, cdf)
}

/// `eob_pt` writer per §5.11.39 lines 19–35 + the refinement
/// lines 39–55 (av1-spec p.89–90).
///
/// Spec body (extracted):
/// ```text
///   eobMultisize = Min( Tx_Width_Log2[ txSz ], 5 )
///                + Min( Tx_Height_Log2[ txSz ], 5 ) - 4
///   if ( eobMultisize == 0 ) { eob_pt_16   S(); eobPt = eob_pt_16   + 1 }
///   else if ( eobMultisize == 1 ) { eob_pt_32   S(); eobPt = eob_pt_32   + 1 }
///   else if ( eobMultisize == 2 ) { eob_pt_64   S(); eobPt = eob_pt_64   + 1 }
///   else if ( eobMultisize == 3 ) { eob_pt_128  S(); eobPt = eob_pt_128  + 1 }
///   else if ( eobMultisize == 4 ) { eob_pt_256  S(); eobPt = eob_pt_256  + 1 }
///   else if ( eobMultisize == 5 ) { eob_pt_512  S(); eobPt = eob_pt_512  + 1 }
///   else                            { eob_pt_1024 S(); eobPt = eob_pt_1024 + 1 }
///
///   eob      = ( eobPt < 2 ) ? eobPt : ( ( 1 << ( eobPt - 2 ) ) + 1 )
///   eobShift = Max( -1, eobPt - 3 )
///   if ( eobShift >= 0 ) {
///       eob_extra                                                   S()
///       if ( eob_extra ) eob += ( 1 << eobShift )
///       for ( i = 1; i < Max( 0, eobPt - 2 ); i++ ) {
///           eobShift = Max( 0, eobPt - 2 ) - 1 - i
///           eob_extra_bit                                           L(1)
///           if ( eob_extra_bit ) eob += ( 1 << eobShift )
///       }
///   }
/// ```
///
/// ## Inverse strategy
///
/// Given a target `eob` (the final EOB value after the §5.11.39 line-39
/// base + the refinement contributions), this writer first derives
/// `eobPt` (the EOB position class) — the unique value with `(eobPt < 2
/// ? eobPt : (1 << (eobPt - 2)) + 1) ≤ eob ≤ 2 * (1 << (eobPt - 2))`
/// (for `eobPt ≥ 3`), or directly `eobPt = eob` for `eob ∈ {1, 2}`. It
/// then writes the appropriate `eob_pt_{16,32,64,128,256,512,1024}`
/// `S()` encoding `eobPt - 1` against the §9.4 default CDF row.
///
/// When `eobPt ≥ 3`, the writer encodes the residue `eob - ((1 <<
/// (eobPt - 2)) + 1)` as `(eobPt - 2)` MSB-first bits. The MSB lands as
/// `eob_extra` (`S()` against `TileEobExtraCdf[ txSzCtx ][ ptype ][ ctx
/// ]` with `ctx = eobPt - 3` per §8.3.2 p.376); the remaining
/// `(eobPt - 3)` bits land as raw `eob_extra_bit` (`L(1)`) writes in
/// descending bit-position order, matching the decoder loop.
///
/// ## CDF selection
///
/// `eobMultisize` is computed inline from `tx_size` (and the §5.11.39
/// `Tx_Width_Log2` table — `TX_HEIGHT[tx_size].trailing_zeros()` for
/// the height log2 since the spec's `Tx_Height_Log2[]` table is not
/// yet a public constant in this crate, mirroring the technique
/// already used in [`crate::cdf::PartitionWalker::coefficients`]'s
/// reader). The §8.3.2 `eob_pt_16..eob_pt_256` rows take a `ctx` axis
/// `(get_tx_class == TX_CLASS_2D) ? 0 : 1`; the `eob_pt_512` /
/// `eob_pt_1024` rows have no `[ ctx ]` axis. Same caller-supplied
/// `tx_class` the reader takes, in `0..=TX_CLASS_VERT`.
///
/// ## Caller-supplied state
///
/// * `eob` — the target EOB value in `1..=segEob`. `segEob` is the
///   §5.11.39 line-6 maximum (`min(1024, Tx_Width[tx_size] *
///   Tx_Height[tx_size])`, except `512` for `TX_16X64` / `TX_64X16`).
///   The valid range is implied by `tx_size`; caller bug otherwise.
/// * `tx_size` — `0..TX_SIZES_ALL = 19`.
/// * `tx_class` — `0..=TX_CLASS_VERT = 2`.
/// * `plane` — `0` (Y) or `1`/`2` (U/V). `ptype = plane > 0` is derived
///   internally, matching the reader.
/// * `is_inter` — `0` or `1`. Only consumed when `eobMultisize ≤ 4`
///   (for the `eob_pt_512` / `eob_pt_1024` rows there is no `is_inter`
///   axis); range-checked unconditionally so callers see a consistent
///   guard.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any caller-supplied
/// index out of range or an `eob` that no `eobPt` selection can
/// represent.
#[allow(clippy::too_many_arguments)]
pub fn write_eob_pt(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    eob: u32,
    tx_size: usize,
    tx_class: usize,
    plane: u8,
    is_inter: u8,
) -> Result<(), Error> {
    // ---------------- caller-bug guards ----------------
    if tx_size >= TX_SIZES_ALL {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_class > 2 {
        // TX_CLASS_VERT == 2 (see crate::cdf::TX_CLASS_VERT).
        return Err(Error::PartitionWalkOutOfRange);
    }
    if plane > 2 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if is_inter > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.39 line 4: `txSzCtx`. Same derivation as the reader.
    let tx_w = TX_WIDTH[tx_size];
    let tx_h = crate::cdf::TX_HEIGHT[tx_size];
    let tx_sz_sqr = {
        let side = core::cmp::min(tx_w, tx_h);
        (side.trailing_zeros() as usize) - 2
    };
    let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_size];
    let tx_sz_ctx = (tx_sz_sqr + tx_sz_sqr_up + 1) >> 1;

    // §5.11.39 line 5: ptype.
    let ptype = (plane > 0) as usize;

    // §5.11.39 line 19: `eobMultisize`.
    let tx_height_log2_local = (tx_h.trailing_zeros() as usize) & 0xFF;
    let eob_multisize: i32 = (core::cmp::min(TX_WIDTH_LOG2[tx_size], 5)
        + core::cmp::min(tx_height_log2_local, 5)) as i32
        - 4;
    debug_assert!(
        (0..=6).contains(&eob_multisize),
        "eobMultisize ∈ 0..=6 over every TX_SIZES_ALL ordinal"
    );

    // ---------------- derive eobPt from eob ----------------
    if eob == 0 {
        // §5.11.39 line 39 sets the base `eob` from `eobPt`, which is
        // itself the §5.11.39 symbol `+ 1` (the §9.4 `eob_pt_*` CDF
        // alphabet starts at 0 / eobPt = 1). A target `eob = 0` is
        // impossible on the `all_zero == 0` arm — the caller already
        // committed to a non-empty block by writing `all_zero = 0`.
        return Err(Error::PartitionWalkOutOfRange);
    }
    // For eobPt = 1 → base eob = 1; eobPt = 2 → base eob = 2; eobPt ≥
    // 3 → base eob = (1 << (eobPt - 2)) + 1, range
    // [base, 2 * (1 << (eobPt - 2))]. The unique eobPt for a given eob
    // ≥ 3 is `floor_log2(eob - 1) + 2`.
    let eob_pt: u32 = if eob <= 2 {
        eob
    } else {
        // floor_log2(eob - 1) + 2.
        let v = eob - 1;
        (31 - v.leading_zeros()) + 2
    };
    // §5.11.39 line 39: re-derive base eob to compute the refinement
    // residue.
    let base_eob: u32 = if eob_pt < 2 {
        eob_pt
    } else {
        (1u32 << (eob_pt - 2)) + 1
    };
    // The refinement bit-width is `Max(0, eobPt - 2)`.
    let refine_bits: u32 = eob_pt.saturating_sub(2);
    let residue = eob - base_eob;
    debug_assert!(
        residue < (1u32 << refine_bits.max(1)) || refine_bits == 0,
        "residue {residue} exceeds {refine_bits}-bit refinement window for eobPt {eob_pt}"
    );

    // The eobPt symbol value the §8.2.6 S() expects.
    let eob_pt_sym: u32 = eob_pt - 1;
    // Each `eob_pt_{16..1024}` CDF has a fixed alphabet size — guard the
    // caller before reaching into the CDF. The eobPt → eobMultisize map
    // requires the residue + base together fit within the table's
    // (alphabet - 1) max eobPt; an out-of-bounds eob for a given
    // `tx_size` therefore surfaces as a caller bug.
    let alphabet: usize = match eob_multisize {
        0 => 6,
        1 => 7,
        2 => 8,
        3 => 9,
        4 => 10,
        5 => 11,
        _ => 12,
    };
    if (eob_pt_sym as usize) >= alphabet {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // ---------------- emit the eob_pt S() ----------------
    let eob_pt_ctx = if tx_class == crate::cdf::TX_CLASS_2D {
        0
    } else {
        1
    };
    match eob_multisize {
        0 => {
            let cdf = cdfs
                .eob_pt_16_cdf(ptype, eob_pt_ctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(eob_pt_sym, cdf)?;
        }
        1 => {
            let cdf = cdfs
                .eob_pt_32_cdf(ptype, eob_pt_ctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(eob_pt_sym, cdf)?;
        }
        2 => {
            let cdf = cdfs
                .eob_pt_64_cdf(ptype, eob_pt_ctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(eob_pt_sym, cdf)?;
        }
        3 => {
            let cdf = cdfs
                .eob_pt_128_cdf(ptype, eob_pt_ctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(eob_pt_sym, cdf)?;
        }
        4 => {
            let cdf = cdfs
                .eob_pt_256_cdf(ptype, eob_pt_ctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(eob_pt_sym, cdf)?;
        }
        5 => {
            let cdf = cdfs
                .eob_pt_512_cdf(ptype)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(eob_pt_sym, cdf)?;
        }
        _ => {
            let cdf = cdfs
                .eob_pt_1024_cdf(ptype)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            writer.write_symbol(eob_pt_sym, cdf)?;
        }
    }

    // ---------------- emit the refinement (eob_extra + L(1) tail) -----
    // The decoder treats the refinement bits MSB-first: `eob_extra` is
    // the MSB (S() against TileEobExtraCdf[txSzCtx][ptype][eobPt - 3]),
    // followed by `(eobPt - 3)` raw L(1) bits in descending position
    // order. For eobPt < 3 the spec's `eobShift = Max(-1, eobPt - 3)`
    // arm guards against any refinement emission.
    if eob_pt >= 3 {
        // `eob_extra` MSB ctx — §8.3.2 p.376 sets it to `eobPt - 3`.
        let extra_ctx = (eob_pt - 3) as usize;
        if extra_ctx >= EOB_COEF_CONTEXTS {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let msb_shift = refine_bits - 1;
        let msb_bit = (residue >> msb_shift) & 0x1;
        let cdf = cdfs
            .eob_extra_cdf(tx_sz_ctx, ptype, extra_ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        writer.write_symbol(msb_bit, cdf)?;

        // Raw L(1) loop — matches the decoder's `eob_extra_bit` loop,
        // descending shift `(refine_bits - 2)..=0`.
        for i in 1..refine_bits {
            let shift = refine_bits - 1 - i;
            let bit = (residue >> shift) & 0x1;
            writer.write_literal(1, bit)?;
        }
    }

    Ok(())
}

/// `dc_sign` writer per §5.11.39 line 564 (av1-spec p.91) and §8.3.2
/// p.377.
///
/// Spec body (extracted, inside the forward-scan loop at `c == 0`):
/// ```text
///   if ( Quant[ pos ] != 0 ) {
///       if ( c == 0 ) {
///           dc_sign                                            S()
///           sign = dc_sign
///       } else {
///           sign_bit                                           L(1)
///           sign = sign_bit
///       }
///   } else {
///       sign = 0
///   }
/// ```
///
/// `dc_sign` is a binary symbol; the §8.2.6 `S()` reads it from
/// `TileDcSignCdf[ ptype ][ ctx ]`. `ctx` is the §8.3.2 `dc_sign` ctx
/// in `0..DC_SIGN_CONTEXTS = 3` — caller-supplied (same shape as the
/// reader's `dc_sign_ctx` parameter).
///
/// `dc_sign` MUST be `0` (positive) or `1` (negative). The decoder
/// only consumes this bit when the DC coefficient is non-zero; the
/// caller is responsible for honouring the `Quant[ 0 ] != 0` gate
/// before invoking this writer (mirroring how the §5.11.39 forward-
/// scan loop only takes the `S()` branch for non-zero `Quant[ pos ]`).
///
/// `plane` is `0` (Y) or `1`/`2` (U/V); `ptype = plane > 0` is
/// derived internally.
pub fn write_dc_sign(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    dc_sign: u8,
    plane: u8,
    ctx: usize,
) -> Result<(), Error> {
    if dc_sign > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if plane > 2 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ctx >= DC_SIGN_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let ptype = (plane > 0) as usize;
    let cdf = cdfs
        .dc_sign_cdf(ptype, ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(dc_sign as u32, cdf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        TileCdfContext, TX_16X16, TX_32X32, TX_4X4, TX_8X8, TX_CLASS_2D, TX_CLASS_HORIZ,
    };
    use crate::symbol_decoder::SymbolDecoder;

    // -----------------------------------------------------------------
    // §5.11.39 write_txb_skip — round-trips through the matching S()
    // read inside `PartitionWalker::coefficients`.
    // -----------------------------------------------------------------

    /// Tiny shim that mirrors `coefficients()`'s very first `S()` —
    /// the `all_zero` read — over the caller-supplied ctx pair.
    /// Used so the txb_skip round-trip tests need only the §5.11.39
    /// line-13 surface, not the whole reader.
    fn read_all_zero(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        tx_sz_ctx: usize,
        ctx: usize,
    ) -> u32 {
        let cdf = cdfs.txb_skip_cdf(tx_sz_ctx, ctx).unwrap();
        dec.read_symbol(cdf).unwrap()
    }

    /// `all_zero = 1` on the simplest ctx (`tx_sz_ctx = 0`, `ctx = 0`)
    /// — the §5.11.39 short-circuit arm. Round-trips through a fresh
    /// CDF context.
    #[test]
    fn write_txb_skip_one_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_txb_skip(&mut writer, &mut enc_cdfs, 1, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let all_zero = read_all_zero(&mut dec, &mut dec_cdfs, 0, 0);
        assert_eq!(all_zero, 1);
    }

    /// `all_zero = 0` — the non-short-circuit arm. Round-trips at the
    /// same ctx.
    #[test]
    fn write_txb_skip_zero_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_txb_skip(&mut writer, &mut enc_cdfs, 0, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let all_zero = read_all_zero(&mut dec, &mut dec_cdfs, 0, 0);
        assert_eq!(all_zero, 0);
    }

    /// `tx_sz_ctx = 4` (TX_64X64 group), `ctx = 12` (the
    /// TXB_SKIP_CONTEXTS upper edge). Confirms the full ctx grid
    /// round-trips, not just the (0, 0) origin.
    #[test]
    fn write_txb_skip_upper_grid_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_txb_skip(&mut writer, &mut enc_cdfs, 1, 4, 12).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let all_zero = read_all_zero(&mut dec, &mut dec_cdfs, 4, 12);
        assert_eq!(all_zero, 1);
    }

    /// Out-of-range `all_zero` is a caller bug.
    #[test]
    fn write_txb_skip_rejects_out_of_range_symbol() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_txb_skip(&mut writer, &mut cdfs, 2, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `tx_sz_ctx` is a caller bug.
    #[test]
    fn write_txb_skip_rejects_out_of_range_tx_sz_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_txb_skip(&mut writer, &mut cdfs, 0, TX_SIZES, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `ctx` is a caller bug.
    #[test]
    fn write_txb_skip_rejects_out_of_range_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_txb_skip(&mut writer, &mut cdfs, 0, 0, TXB_SKIP_CONTEXTS).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.39 write_eob_pt — round-trips through the §5.11.39 line
    // 19–55 EOB decoder surface (eob_pt_* + eob_extra + eob_extra_bit
    // loop).
    // -----------------------------------------------------------------

    /// Local helper mirroring the §5.11.39 lines 19–55 reader — pulled
    /// in to drive the round-trip without standing up the full
    /// `coefficients()` reader (which also reads the coeff_base /
    /// coeff_br chain we don't emit yet).
    #[allow(clippy::too_many_arguments)]
    fn read_eob(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        tx_size: usize,
        tx_class: usize,
        plane: u8,
        is_inter: u8,
    ) -> u32 {
        let tx_w = TX_WIDTH[tx_size];
        let tx_h = crate::cdf::TX_HEIGHT[tx_size];
        let tx_sz_sqr = (core::cmp::min(tx_w, tx_h).trailing_zeros() as usize) - 2;
        let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_size];
        let tx_sz_ctx = (tx_sz_sqr + tx_sz_sqr_up + 1) >> 1;
        let ptype = (plane > 0) as usize;
        let tx_height_log2_local = (tx_h.trailing_zeros() as usize) & 0xFF;
        let eob_multisize: i32 = (core::cmp::min(TX_WIDTH_LOG2[tx_size], 5)
            + core::cmp::min(tx_height_log2_local, 5)) as i32
            - 4;
        let eob_pt_ctx = if tx_class == TX_CLASS_2D { 0 } else { 1 };
        let eob_pt: u32 = match eob_multisize {
            0 => {
                let cdf = cdfs.eob_pt_16_cdf(ptype, eob_pt_ctx).unwrap();
                dec.read_symbol(cdf).unwrap() + 1
            }
            1 => {
                let cdf = cdfs.eob_pt_32_cdf(ptype, eob_pt_ctx).unwrap();
                dec.read_symbol(cdf).unwrap() + 1
            }
            2 => {
                let cdf = cdfs.eob_pt_64_cdf(ptype, eob_pt_ctx).unwrap();
                dec.read_symbol(cdf).unwrap() + 1
            }
            3 => {
                let cdf = cdfs.eob_pt_128_cdf(ptype, eob_pt_ctx).unwrap();
                dec.read_symbol(cdf).unwrap() + 1
            }
            4 => {
                let cdf = cdfs.eob_pt_256_cdf(ptype, eob_pt_ctx).unwrap();
                dec.read_symbol(cdf).unwrap() + 1
            }
            5 => {
                let cdf = cdfs.eob_pt_512_cdf(ptype).unwrap();
                dec.read_symbol(cdf).unwrap() + 1
            }
            _ => {
                let cdf = cdfs.eob_pt_1024_cdf(ptype).unwrap();
                dec.read_symbol(cdf).unwrap() + 1
            }
        };
        let mut eob: u32 = if eob_pt < 2 {
            eob_pt
        } else {
            (1u32 << (eob_pt - 2)) + 1
        };
        let mut eob_shift: i32 = core::cmp::max(-1, eob_pt as i32 - 3);
        if eob_shift >= 0 {
            let extra_ctx = (eob_pt - 3) as usize;
            let cdf = cdfs.eob_extra_cdf(tx_sz_ctx, ptype, extra_ctx).unwrap();
            let eob_extra = dec.read_symbol(cdf).unwrap();
            if eob_extra != 0 {
                eob += 1u32 << eob_shift;
            }
            let upper = core::cmp::max(0, eob_pt as i32 - 2);
            let mut i: i32 = 1;
            while i < upper {
                eob_shift = upper - 1 - i;
                let bit = dec.read_literal(1).unwrap();
                if bit != 0 {
                    eob += 1u32 << eob_shift;
                }
                i += 1;
            }
        }
        let _ = is_inter;
        eob
    }

    /// `eob = 1` on a TX_4X4 luma block, intra. Smallest possible EOB
    /// (eobPt = 1 ⇒ no refinement bits). Round-trips through
    /// `read_eob`.
    #[test]
    fn write_eob_pt_eob_1_round_trip_tx4x4() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_eob_pt(&mut writer, &mut enc_cdfs, 1, TX_4X4, TX_CLASS_2D, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_4X4, TX_CLASS_2D, 0, 0);
        assert_eq!(eob, 1);
    }

    /// `eob = 2` on TX_4X4 (eobPt = 2, base eob = 2, eobShift = -1 ⇒
    /// no refinement). Round-trip.
    #[test]
    fn write_eob_pt_eob_2_round_trip_tx4x4() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_eob_pt(&mut writer, &mut enc_cdfs, 2, TX_4X4, TX_CLASS_2D, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_4X4, TX_CLASS_2D, 0, 0);
        assert_eq!(eob, 2);
    }

    /// `eob = 3` on TX_4X4 (eobPt = 3 ⇒ refinement window 1 bit;
    /// residue = 0, so eob_extra writes the MSB = 0). Round-trip.
    #[test]
    fn write_eob_pt_eob_3_round_trip_tx4x4() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_eob_pt(&mut writer, &mut enc_cdfs, 3, TX_4X4, TX_CLASS_2D, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_4X4, TX_CLASS_2D, 0, 0);
        assert_eq!(eob, 3);
    }

    /// `eob = 8` on TX_4X4 (eobPt = 4 ⇒ refinement 2 bits, residue
    /// = 8 - 5 = 3 ⇒ MSB+L(1) = 1,1). Round-trip.
    #[test]
    fn write_eob_pt_eob_8_round_trip_tx4x4() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_eob_pt(&mut writer, &mut enc_cdfs, 8, TX_4X4, TX_CLASS_2D, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_4X4, TX_CLASS_2D, 0, 0);
        assert_eq!(eob, 8);
    }

    /// `eob = 16` on TX_8X8 — eobMultisize = 1 ⇒ eob_pt_32 row, eobPt
    /// = 5 ⇒ refinement 3 bits, residue = 16 - 9 = 7 ⇒ MSB+L(1)+L(1) =
    /// 1,1,1. Round-trips through the eob_pt_32 selector.
    #[test]
    fn write_eob_pt_eob_16_round_trip_tx8x8() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_eob_pt(&mut writer, &mut enc_cdfs, 16, TX_8X8, TX_CLASS_2D, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_8X8, TX_CLASS_2D, 0, 0);
        assert_eq!(eob, 16);
    }

    /// `eob = 5` on TX_16X16 chroma block, intra, TX_CLASS_HORIZ —
    /// exercises ptype = 1 axis and tx_class != TX_CLASS_2D ctx-axis
    /// flip (eob_pt_ctx = 1).
    #[test]
    fn write_eob_pt_eob_5_round_trip_tx16x16_chroma_horiz() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_eob_pt(
            &mut writer,
            &mut enc_cdfs,
            5,
            TX_16X16,
            TX_CLASS_HORIZ,
            1,
            0,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_16X16, TX_CLASS_HORIZ, 1, 0);
        assert_eq!(eob, 5);
    }

    /// `eob = 17` on TX_32X32, inter — eobMultisize = 3 (eob_pt_128
    /// row). eobPt = 6 (base = 17, refinement 4 bits, residue = 0).
    /// Round-trip.
    #[test]
    fn write_eob_pt_eob_17_round_trip_tx32x32_inter() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_eob_pt(&mut writer, &mut enc_cdfs, 17, TX_32X32, TX_CLASS_2D, 0, 1).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_32X32, TX_CLASS_2D, 0, 1);
        assert_eq!(eob, 17);
    }

    /// `eob = 0` is a caller bug — the `all_zero == 0` arm
    /// post-condition requires eob ≥ 1.
    #[test]
    fn write_eob_pt_rejects_eob_zero() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_eob_pt(&mut writer, &mut cdfs, 0, TX_4X4, TX_CLASS_2D, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `tx_size` is a caller bug.
    #[test]
    fn write_eob_pt_rejects_out_of_range_tx_size() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err =
            write_eob_pt(&mut writer, &mut cdfs, 1, TX_SIZES_ALL, TX_CLASS_2D, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `eob` for the selected `tx_size` (eob > segEob,
    /// chosen so the eobPt symbol exceeds the row's alphabet) is a
    /// caller bug.
    #[test]
    fn write_eob_pt_rejects_eob_overflow_for_tx_size() {
        // TX_4X4 ⇒ eobMultisize = 0 ⇒ eob_pt_16 alphabet = 6 ⇒ max
        // representable eobPt = 6 ⇒ max representable eob = 2 * 16 =
        // 32. (The block's physical segEob is 16; the alphabet check
        // is the writer's surface guard.)
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_eob_pt(&mut writer, &mut cdfs, 64, TX_4X4, TX_CLASS_2D, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.39 write_dc_sign — round-trips through the matching S()
    // read at the c == 0 forward-scan position.
    // -----------------------------------------------------------------

    /// Tiny shim that mirrors the `dc_sign` read at `c == 0`.
    fn read_dc_sign(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        ptype: usize,
        ctx: usize,
    ) -> u32 {
        let cdf = cdfs.dc_sign_cdf(ptype, ctx).unwrap();
        dec.read_symbol(cdf).unwrap()
    }

    /// `dc_sign = 0` (positive DC) on luma, ctx 0 — round-trips.
    #[test]
    fn write_dc_sign_positive_round_trip_luma() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_dc_sign(&mut writer, &mut enc_cdfs, 0, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let s = read_dc_sign(&mut dec, &mut dec_cdfs, 0, 0);
        assert_eq!(s, 0);
    }

    /// `dc_sign = 1` (negative DC) on chroma, ctx 2 (upper edge of
    /// DC_SIGN_CONTEXTS) — round-trips.
    #[test]
    fn write_dc_sign_negative_round_trip_chroma_ctx2() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_dc_sign(&mut writer, &mut enc_cdfs, 1, 1, 2).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        // ptype = 1 for plane = 1 (U).
        let s = read_dc_sign(&mut dec, &mut dec_cdfs, 1, 2);
        assert_eq!(s, 1);
    }

    /// Out-of-range `dc_sign` (only 0/1 are valid) is a caller bug.
    #[test]
    fn write_dc_sign_rejects_out_of_range_symbol() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_dc_sign(&mut writer, &mut cdfs, 2, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `ctx` (DC_SIGN_CONTEXTS = 3) is a caller bug.
    #[test]
    fn write_dc_sign_rejects_out_of_range_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_dc_sign(&mut writer, &mut cdfs, 0, 0, DC_SIGN_CONTEXTS).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // Sequenced round-trip: txb_skip = 0 followed by an eob_pt write.
    // Confirms CDF adaptation stays in lockstep across multiple writer
    // calls into the same SymbolWriter.
    // -----------------------------------------------------------------

    /// Sequence `txb_skip = 0` followed by `eob_pt(eob = 4)` on TX_4X4.
    /// Confirms two distinct §8.2.6 S() writes round-trip through two
    /// distinct §8.2.6 S() reads with the §8.3 CDF adaptation engaged
    /// on both sides.
    #[test]
    fn sequence_txb_skip_then_eob_pt_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_txb_skip(&mut writer, &mut enc_cdfs, 0, 0, 0).unwrap();
        write_eob_pt(&mut writer, &mut enc_cdfs, 4, TX_4X4, TX_CLASS_2D, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let all_zero = read_all_zero(&mut dec, &mut dec_cdfs, 0, 0);
        assert_eq!(all_zero, 0);
        let eob = read_eob(&mut dec, &mut dec_cdfs, TX_4X4, TX_CLASS_2D, 0, 0);
        assert_eq!(eob, 4);
    }
}
