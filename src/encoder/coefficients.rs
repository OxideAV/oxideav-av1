//! Per-coefficient **writers** for the §5.11.39 `coefficients()` body.
//!
//! Arc 6 (round 212) landed the framing primitives `txb_skip` (the
//! `all_zero` symbol), `eob_pt` (the EOB position class + refinement
//! bits), and `dc_sign`. Arc 7 (round 213) extends with the
//! per-coefficient base-level chain at each reverse-scan position:
//! `coeff_base_eob`, `coeff_base`, and `coeff_br`.
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
//!   * On the reverse-scan iteration at `c == eob - 1` the decoder
//!     reads `coeff_base_eob` (`S()` against `TileCoeffBaseEobCdf[
//!     txSzCtx ][ ptype ][ ctx ]`); for `c < eob - 1` it reads
//!     `coeff_base` (`S()` against `TileCoeffBaseCdf[ txSzCtx ][ ptype
//!     ][ ctx ]`). The `ctx` axis on both is derived by the §8.3.2
//!     helpers [`crate::cdf::get_coeff_base_eob_ctx`] /
//!     [`crate::cdf::get_coeff_base_ctx`] from the running `Quant[]`
//!     array (the caller pre-computes; the writer is stateless).
//!     Writers: [`write_coeff_base_eob`] / [`write_coeff_base`].
//!   * When the level (`coeff_base_eob + 1` or `coeff_base`) exceeds
//!     `NUM_BASE_LEVELS = 2`, the decoder runs up to
//!     `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) = 4` iterations of
//!     `coeff_br` (each `S()` against `TileCoeffBrCdf[ Min(txSzCtx,
//!     TX_32X32) ][ ptype ][ ctx ]`), terminating when the symbol
//!     falls below `BR_CDF_SIZE - 1 = 3`. The `ctx` axis is the
//!     §8.3.2 [`crate::cdf::get_br_ctx`] result. Writer:
//!     [`write_coeff_br`] (one `S()` per call — the driver loop that
//!     stacks up to 4 of them is the next arc).
//!   * Inside the forward-scan loop, the first non-zero coefficient
//!     (`c == 0` arm) reads one `dc_sign` `S()` against
//!     `TileDcSignCdf[ ptype ][ ctx ]` (§8.3.2). Writer:
//!     [`write_dc_sign`].
//!
//! Arc 8 (round 214) extends with the §5.11.39 lines 84-93 golomb
//! magnitude-tail writer for coefficient magnitudes above
//! `NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`:
//!
//!   * Inside the forward-scan loop, when `Quant[pos] > 14` the decoder
//!     enters a `golomb_length_bit` do-while unary loop followed by a
//!     `golomb_data_bit` MSB-first L(1) payload that rebuilds the
//!     magnitude as `x + COEFF_BASE_RANGE + NUM_BASE_LEVELS`. Writer:
//!     [`write_golomb`]. The §6.10.34 conformance bound
//!     (`length <= 20`) is enforced as a caller-bug reject.
//!
//! Scope of this arc is the per-magnitude golomb tail only; the full
//! `coefficients()` driver loop that sequences `coeff_base_eob` /
//! `coeff_base` / `coeff_br` / sign / `golomb` across the reverse +
//! forward scans is the next arc.
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
    get_br_ctx, get_coeff_base_ctx, get_coeff_base_eob_ctx, TileCdfContext, BR_CDF_SIZE,
    COEFF_BASE_RANGE, DC_SIGN_CONTEXTS, EOB_COEF_CONTEXTS, LEVEL_CONTEXTS, NUM_BASE_LEVELS,
    PLANE_TYPES, SIG_COEF_CONTEXTS, SIG_COEF_CONTEXTS_EOB, TXB_SKIP_CONTEXTS, TX_SIZES,
    TX_SIZES_ALL, TX_SIZE_SQR_UP, TX_WIDTH, TX_WIDTH_LOG2,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::Error;

/// Maximum §5.11.39 golomb `length` permitted by the conformance
/// constraint of §6.10.34 (av1-spec p.392): "If length is equal to 20,
/// it is a requirement of bitstream conformance that golomb_length_bit
/// is equal to 1." A `length > 20` would describe a magnitude exceeding
/// the 20-bit `Quant[pos] & 0xFFFFF` clip the §5.11.39 line-97 mask
/// enforces, so the writer rejects it as a caller bug rather than
/// emitting a non-conformant stream.
pub const GOLOMB_MAX_LENGTH: u32 = 20;

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

/// `coeff_base_eob` writer per §5.11.39 line 60 (av1-spec p.91) and
/// §8.3.2 p.376.
///
/// Spec body (extracted, inside the reverse-scan loop at the
/// `c == eob - 1` iteration):
/// ```text
///   coeff_base_eob                                              S()
///   level = coeff_base_eob + 1
/// ```
///
/// `coeff_base_eob` is a 3-symbol `S()` against
/// `TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]`. The §9.4 alphabet
/// is `{ 0, 1, 2 }`, mapping to levels `{ 1, 2, 3 }` (the §5.11.39
/// line-60 `level = coeff_base_eob + 1` derivation makes 0 the smallest
/// because the EOB-position coefficient is known non-zero).
///
/// `ctx` is the `coeff_base_eob` context the caller derives via
/// [`crate::cdf::get_coeff_base_eob_ctx`] — the §8.3.2 reduction of
/// `get_coeff_base_ctx(..., is_eob = true)` onto
/// `0..SIG_COEF_CONTEXTS_EOB = 4`. Passing the already-derived `ctx`
/// (mirroring [`write_dc_sign`]'s caller-supplied `ctx`) keeps the
/// writer stateless: the §5.11.39 driver loop the next arc lands will
/// supply the `Quant[]`-aware ctx; today the round-trip tests below
/// supply a fixed ctx.
///
/// `sym` MUST be in `0..=2` (the 3-symbol alphabet). `tx_sz_ctx` is the
/// §5.11.39 line-4 derivation in `0..TX_SIZES = 5`; `ptype` is
/// `plane > 0` in `0..PLANE_TYPES = 2`.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any caller-supplied
/// index out of range.
pub fn write_coeff_base_eob(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    sym: u8,
    tx_sz_ctx: usize,
    ptype: usize,
    ctx: usize,
) -> Result<(), Error> {
    if sym > 2 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_sz_ctx >= TX_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ptype >= PLANE_TYPES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ctx >= SIG_COEF_CONTEXTS_EOB {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs
        .coeff_base_eob_cdf(tx_sz_ctx, ptype, ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(sym as u32, cdf)
}

/// `coeff_base` writer per §5.11.39 line 63 (av1-spec p.91) and §8.3.2
/// p.371.
///
/// Spec body (extracted, inside the reverse-scan loop for
/// `c < eob - 1`):
/// ```text
///   coeff_base                                                  S()
///   level = coeff_base
/// ```
///
/// `coeff_base` is a 4-symbol `S()` against
/// `TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]`. The §9.4 alphabet is
/// `{ 0, 1, 2, 3 }`, mapping directly to levels `{ 0, 1, 2, 3 }`.
/// Symbol `0` means the coefficient at this scan position is zero;
/// symbol `3` means the magnitude continues through the §5.11.39
/// lines 65-70 `coeff_br` chain (which [`write_coeff_br`] handles one
/// `S()` at a time).
///
/// `ctx` is the `coeff_base` context the caller derives via
/// [`crate::cdf::get_coeff_base_ctx`] with `is_eob = false` — the
/// §8.3.2 neighbour-magnitude accumulation reducing onto
/// `0..SIG_COEF_CONTEXTS = 42`. Same stateless caller pattern as
/// [`write_coeff_base_eob`]; the driver loop produces the ctx from
/// the running `Quant[]` array.
///
/// `sym` MUST be in `0..=3`. `tx_sz_ctx` is `0..TX_SIZES = 5`; `ptype`
/// is `0..PLANE_TYPES = 2`.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any caller-supplied
/// index out of range.
pub fn write_coeff_base(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    sym: u8,
    tx_sz_ctx: usize,
    ptype: usize,
    ctx: usize,
) -> Result<(), Error> {
    if sym > 3 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_sz_ctx >= TX_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ptype >= PLANE_TYPES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ctx >= SIG_COEF_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs
        .coeff_base_cdf(tx_sz_ctx, ptype, ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(sym as u32, cdf)
}

/// `coeff_br` writer per §5.11.39 lines 65-70 (av1-spec p.91) and
/// §8.3.2 p.378.
///
/// Spec body (extracted, the per-iteration `S()` inside the
/// `coeff_br` chain that runs while `level > NUM_BASE_LEVELS`):
/// ```text
///   for ( idx = 0; idx < COEFF_BASE_RANGE / ( BR_CDF_SIZE - 1 ); idx++ ) {
///       coeff_br                                                S()
///       level += coeff_br
///       if ( coeff_br < BR_CDF_SIZE - 1 ) break
///   }
/// ```
///
/// Each `coeff_br` is a `BR_CDF_SIZE`-symbol `S()` against
/// `TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][ ctx ]`. The
/// `txSzCtx` clamp at `TX_32X32 = 3` lives inside
/// [`TileCdfContext::coeff_br_cdf`], so this writer simply forwards the
/// caller's `tx_sz_ctx`. The §9.4 alphabet is `{ 0, 1, 2, 3 }`
/// (`BR_CDF_SIZE = 4`); symbol `BR_CDF_SIZE - 1 = 3` means the chain
/// continues into the next `coeff_br` iteration, any smaller value
/// terminates the chain (the spec's `if (coeff_br < BR_CDF_SIZE - 1)
/// break` guard).
///
/// This writer encodes **one** `coeff_br` `S()` per call — the driver
/// loop that runs up to `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) = 4`
/// iterations is a follow-on arc. The per-call shape matches the
/// per-iteration `decoder.read_symbol(cdf)` call inside
/// [`crate::cdf::PartitionWalker::coefficients`].
///
/// `ctx` is the `coeff_br` context the caller derives via
/// [`crate::cdf::get_br_ctx`] — the §8.3.2 three-neighbour magnitude
/// accumulation onto `0..LEVEL_CONTEXTS = 21`. `sym` MUST be in
/// `0..BR_CDF_SIZE = 0..=3`; `tx_sz_ctx` is `0..TX_SIZES = 5` (the
/// selector clamps at `TX_32X32` internally); `ptype` is
/// `0..PLANE_TYPES = 2`.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any caller-supplied
/// index out of range.
pub fn write_coeff_br(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    sym: u8,
    tx_sz_ctx: usize,
    ptype: usize,
    ctx: usize,
) -> Result<(), Error> {
    if (sym as usize) >= BR_CDF_SIZE {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_sz_ctx >= TX_SIZES {
        // The selector clamps to TX_32X32 internally, but a frankly
        // out-of-range tx_sz_ctx is a caller bug worth surfacing
        // explicitly — same surface contract as the other writers in
        // this module.
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ptype >= PLANE_TYPES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if ctx >= LEVEL_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cdf = cdfs
        .coeff_br_cdf(tx_sz_ctx, ptype, ctx)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    writer.write_symbol(sym as u32, cdf)
}

/// `golomb` magnitude-tail writer per §5.11.39 lines 84-93 (av1-spec
/// p.91) and §6.10.34 semantics (av1-spec p.392).
///
/// Spec body (extracted, inside the forward-scan loop when
/// `Quant[pos] > NUM_BASE_LEVELS + COEFF_BASE_RANGE`):
/// ```text
///   length = 0
///   do {
///       length++
///       golomb_length_bit                                       L(1)
///   } while ( !golomb_length_bit )
///
///   x = 1
///   for ( i = length - 2; i >= 0; i-- ) {
///       golomb_data_bit                                         L(1)
///       x = ( x << 1 ) | golomb_data_bit
///   }
///   Quant[ pos ] = x + COEFF_BASE_RANGE + NUM_BASE_LEVELS
/// ```
///
/// ## Inverse strategy
///
/// Given a target magnitude `value` (the §5.11.39 `Quant[pos]` after the
/// `coeff_base` + `coeff_br` chain has saturated at `NUM_BASE_LEVELS +
/// COEFF_BASE_RANGE = 14` and the caller has determined the magnitude
/// must continue), the writer derives `x = value - 14` (always
/// `x >= 1`) and `length = floor_log2(x) + 1` (the bit-length of `x`).
///
/// It then emits the §5.11.39 line 76-79 unary length prefix as
/// `length - 1` `L(1) = 0` bits followed by an `L(1) = 1` terminator —
/// the do-while loop on the decoder side reads bits until the first
/// `1`, so `length` is the index (1-based) of the terminator. Finally
/// it emits the `length - 1` `golomb_data_bit` payload bits MSB-first:
/// the implicit leading `1` of `x` is the §5.11.39 line 87 `x = 1`
/// seed, and the remaining `length - 1` bits of `x` are emitted in
/// descending position order to match the decoder's
/// `x = (x << 1) | golomb_data_bit` rebuild.
///
/// ## Caller-supplied state
///
/// * `value` — the target post-golomb `Quant[pos]` magnitude (unsigned;
///   the sign lives in the `dc_sign` / `sign_bit` write that already
///   landed before this call inside the forward-scan loop). MUST
///   satisfy `value > NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`; a
///   smaller value would not have entered the §5.11.39 line-73 gate and
///   is a caller bug.
///
/// ## Conformance
///
/// The §6.10.34 semantics constrain `length <= 20`: at `length == 20`
/// the `golomb_length_bit` is required to be `1` (the do-while
/// terminator), and a `length > 20` would code a magnitude past the
/// §5.11.39 line-97 `Quant[pos] & 0xFFFFF` 20-bit clip. The writer
/// rejects `value` that would derive `length > GOLOMB_MAX_LENGTH = 20`
/// as a caller bug — `value <= 0xFFFFF + 14 = 1048589` is the
/// representable range, and `x >= 0x80000 = 524288` already saturates
/// `length == 20`.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any out-of-range
/// `value` and propagates any [`SymbolWriter::write_literal`] error.
pub fn write_golomb(writer: &mut SymbolWriter, value: u32) -> Result<(), Error> {
    let threshold = (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as u32;
    if value <= threshold {
        // §5.11.39 line 73 gate: the golomb tail is only emitted when
        // the magnitude exceeds NUM_BASE_LEVELS + COEFF_BASE_RANGE.
        return Err(Error::PartitionWalkOutOfRange);
    }
    let x: u32 = value - threshold;
    // §5.11.39 line 87 derivation: `x = 1` is the do-while loop's
    // post-condition seed; `length` is the bit-length of `x` (so that
    // `length - 1` data bits, prepended with the implicit MSB = 1,
    // rebuild `x` exactly).
    debug_assert!(x >= 1, "value > threshold guarantees x >= 1");
    let length: u32 = 32 - x.leading_zeros();
    if length > GOLOMB_MAX_LENGTH {
        // §6.10.34 conformance constraint: length <= 20.
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.39 lines 75-79: unary length prefix. The decoder's do-while
    // loop increments `length` before checking the bit, so the first
    // iteration is always entered — i.e. the bit stream is
    // `0` * (length - 1) followed by `1`. Total bits emitted = length.
    for _ in 0..(length - 1) {
        writer.write_literal(1, 0)?;
    }
    writer.write_literal(1, 1)?;

    // §5.11.39 lines 87-91: data-bit payload. The decoder seeds `x = 1`
    // and shifts in `length - 1` bits MSB-first via
    // `x = (x << 1) | bit`. The inverse: emit bits `length - 2 .. 0` of
    // `x` (the high `length - 1` bits below the implicit MSB).
    if length >= 2 {
        let payload_bits = length - 1;
        for i in (0..payload_bits).rev() {
            let bit = (x >> i) & 0x1;
            writer.write_literal(1, bit)?;
        }
    }

    Ok(())
}

/// Full §5.11.39 `coefficients()` **driver loop** — composes every
/// per-coefficient primitive above into one per-transform-block encode.
///
/// This is the encoder counterpart to
/// [`crate::cdf::PartitionWalker::coefficients`] (the §5.11.39 reader).
/// Given a caller-supplied final-`Quant[]` array (signed, post-decoder
/// values — what the decoder would produce at exit), the driver walks
/// the §5.11.39 body and emits every symbol / literal the reader would
/// consume, so the output bytes round-trip back through the reader to
/// the same `Quant[]` array.
///
/// ## Spec body (av1-spec p.88–93)
///
/// ```text
///   coefficients( plane, x4, y4, txSz ) {
///       ...                                          // §5.11.39 1-9 derive sizes + zero Quant[]
///       all_zero                                     S()                // §5.11.39 line 13
///       if ( all_zero ) { ... return }                                 // §5.11.39 line 14
///       eob_pt_<N> + eob_extra + eob_extra_bit loop                    // §5.11.39 lines 19-55
///       for ( c = eob - 1; c >= 0; c-- ) {                              // §5.11.39 lines 56-71
///           if ( c == eob - 1 ) coeff_base_eob  S(); level = sym + 1
///           else                coeff_base      S(); level = sym
///           if ( level > NUM_BASE_LEVELS ) coeff_br chain (≤ 4 iters)
///           Quant[ pos ] = level
///       }
///       for ( c = 0; c < eob; c++ ) {                                   // §5.11.39 lines 73-100
///           if ( Quant[ pos ] != 0 ) {
///               sign = c == 0 ? dc_sign S() : sign_bit L(1)
///           }
///           if ( Quant[ pos ] > 14 ) golomb tail                       // §5.11.39 lines 84-93
///           ...                                                         // dcCategory + 0xFFFFF clip + culLevel + sign apply
///       }
///   }
/// ```
///
/// ## Inverse strategy
///
/// 1. **Compute `eob`** as the largest `c + 1` over `0..scan_len` for
///    which `quant_in[ scan[c] ] != 0`. `eob == 0` ⇒ `all_zero = 1`,
///    no further writes. `eob >= 1` ⇒ `all_zero = 0` and the cascade
///    fires.
/// 2. **Maintain a running magnitude buffer.** The §8.3.2 ctx
///    derivations consume the `Quant[]` array as the reverse-scan
///    populates it. The driver mirrors this on the encoder side: it
///    initialises a local `running` buffer to zero and writes each
///    iteration's `coded_mag = min(|quant_in[pos]|, 15)` into it
///    before the next iteration's ctx scan runs. The reader's
///    `Quant[]` at the same step holds the same value, so
///    `get_coeff_base_*_ctx` / `get_br_ctx` agree on both sides.
/// 3. **Reverse-scan emission.** For each `c = eob - 1 .. 0`:
///    * `coded_mag = min(target_abs, 15)` (post-`coeff_br` saturation
///      level — `NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1 = 15` is the
///      cap because a `coeff_base{,_eob}` of 3 entry + four chain
///      iterations of `+3` each maxes out at level 15).
///    * If `is_eob`: write `coeff_base_eob(min(coded_mag - 1, 2))`
///      (alphabet `0..=2` ⇒ level `1..=3`).
///    * Else: write `coeff_base(min(coded_mag, 3))` (alphabet `0..=3`
///      ⇒ level `0..=3`).
///    * If the level lands at 3 (the chain-continues sentinel),
///      enter the `coeff_br` chain: each iter emits
///      `min(residue, BR_CDF_SIZE - 1 = 3)` where `residue` is
///      `coded_mag - chain_level_so_far`. Chain caps at 4 iterations
///      total per the spec's `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) = 4`.
///    * Stamp `running[pos] = coded_mag` for downstream ctx walks.
/// 4. **Forward-scan emission.** For each `c = 0 .. eob`:
///    * If `target_abs != 0` and `c == 0`: write `dc_sign(sign)`.
///    * Else if `target_abs != 0`: write `sign_bit` as `L(1)`.
///    * If `target_abs > NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`:
///      write the §5.11.39 lines 84-93 golomb tail via
///      [`write_golomb`].
///
/// ## Caller-supplied state
///
/// * `plane` — `0` (Y), `1`/`2` (U/V).
/// * `is_inter` — `0`/`1`. (Currently consumed only as a guard; the
///   §5.11.39 reader passes it through to the same forwarded
///   [`write_eob_pt`].)
/// * `tx_size` — `0..TX_SIZES_ALL = 19`.
/// * `tx_class` — `0..=TX_CLASS_VERT = 2`.
/// * `txb_skip_ctx` — §8.3.2 `all_zero` ctx in
///   `0..TXB_SKIP_CONTEXTS = 13`. Caller derives.
/// * `dc_sign_ctx` — §8.3.2 `dc_sign` ctx in
///   `0..DC_SIGN_CONTEXTS = 3`. Caller derives.
/// * `scan` — the §7.5 scan table for `tx_size` / `tx_class` (length
///   ≥ `seg_eob`).
/// * `quant_in` — the per-position **signed final** `Quant[]` array
///   (length ≥ `tx_w * tx_h`). The driver consumes this read-only;
///   it does NOT mutate the caller's buffer. Range constraint: each
///   `|quant_in[pos]|` must be ≤ `0xFFFFF + 14 = 1_048_589` per the
///   §5.11.39 line-97 `& 0xFFFFF` clip + §6.10.34 golomb-length
///   conformance bound (both enforced via [`write_golomb`]).
///
/// ## Returns
///
/// A [`crate::cdf::CoefficientsReadout`] mirroring what the §5.11.39
/// reader would return for the just-written TU (`all_zero` / `eob` /
/// `cul_level` / `dc_category`) — the `cul_level` / `dc_category`
/// pair feeds the §5.11.39 tail stamps into the §6.10.2
/// `Above{Level,Dc}Context` / `Left{Level,Dc}Context` arrays the
/// §8.3.2 `all_zero` / `dc_sign` ctx walks consult on neighbour TUs.
///
/// [`Error::PartitionWalkOutOfRange`] for any out-of-range index or
/// quant-buffer / scan-buffer length shortfall. Propagates
/// [`SymbolWriter::write_symbol`] / `write_literal` / `write_golomb`
/// errors.
///
/// ## Spec provenance
///
/// `docs/video/av1/av1-spec.txt` §5.11.39 (p.88–93) + §6.5.10 syntax
/// table + §8.3.2 context derivations (p.371–378) + §9.4 default
/// CDFs.
#[allow(clippy::too_many_arguments)]
pub fn write_coefficients(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    plane: u8,
    is_inter: u8,
    tx_size: usize,
    tx_class: usize,
    txb_skip_ctx: usize,
    dc_sign_ctx: usize,
    scan: &[u16],
    quant_in: &[i32],
) -> Result<crate::cdf::CoefficientsReadout, Error> {
    // ---------------- caller-bug guards (mirror the reader's) ---------
    if tx_size >= TX_SIZES_ALL {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_class > crate::cdf::TX_CLASS_VERT {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if plane > 2 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if is_inter > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if txb_skip_ctx >= TXB_SKIP_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if dc_sign_ctx >= DC_SIGN_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.39 lines 1-7: derived sizes (mirror the reader).
    let tx_w = TX_WIDTH[tx_size];
    let tx_h = crate::cdf::TX_HEIGHT[tx_size];
    let tx_sz_sqr = {
        let side = core::cmp::min(tx_w, tx_h);
        (side.trailing_zeros() as usize) - 2
    };
    let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_size];
    let tx_sz_ctx = (tx_sz_sqr + tx_sz_sqr_up + 1) >> 1;
    // §5.11.39 line 6: `segEob`.
    let seg_eob = if tx_size == crate::cdf::TX_16X64 || tx_size == crate::cdf::TX_64X16 {
        512usize
    } else {
        core::cmp::min(1024, tx_w * tx_h)
    };

    if scan.len() < seg_eob {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if quant_in.len() < tx_w * tx_h {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // ---------------- compute target eob from quant_in ----------------
    // §5.11.39 lines 56-71: the reverse-scan loop reads coefficients in
    // descending scan order; `eob` is the position past the highest
    // non-zero coefficient. We pick `eob` as the largest `c + 1` in
    // `0..seg_eob` for which `quant_in[scan[c]] != 0`. `eob == 0` ⇒
    // every scan position is zero ⇒ short-circuit.
    let mut eob: u32 = 0;
    for c in (0..seg_eob).rev() {
        let pos = scan[c] as usize;
        if pos >= quant_in.len() {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if quant_in[pos] != 0 {
            eob = (c as u32) + 1;
            break;
        }
    }

    // §5.11.39 line 13: `all_zero S()`.
    let all_zero: u8 = if eob == 0 { 1 } else { 0 };
    write_txb_skip(writer, cdfs, all_zero, tx_sz_ctx, txb_skip_ctx)?;
    if eob == 0 {
        // §5.11.39 line 14: short-circuit. No further reads on the
        // reader side ⇒ no further writes here. `culLevel` /
        // `dcCategory` keep their line-11/12 zero initialisers (the
        // §5.11.39 tail stamps still fire on the gate-closed arm).
        return Ok(crate::cdf::CoefficientsReadout {
            all_zero: true,
            eob: 0,
            cul_level: 0,
            dc_category: 0,
        });
    }

    write_coefficients_gate_open(
        writer,
        cdfs,
        plane,
        is_inter,
        tx_size,
        tx_class,
        dc_sign_ctx,
        scan,
        quant_in,
    )
}

/// §5.11.39 `coeffs()` write side — the gate-open body (everything
/// **after** the `all_zero` S() write): the `eob_pt_*` / `eob_extra` /
/// `eob_extra_bit` emission, the reverse-scan `coeff_base{_eob}` +
/// `coeff_br` level loop, and the forward-scan `dc_sign` / `sign_bit` /
/// golomb pass.
///
/// The write twin of [`PartitionWalker::coefficients_gate_open`]
/// (`crate::cdf`): a caller honouring the §5.11.39 spec order writes
/// `all_zero` via [`write_txb_skip`], performs the §5.11.47
/// `transform_type()` emission on the `all_zero == 0` arm, derives
/// `tx_class` / `scan` from the resulting `PlaneTxType`, then invokes
/// this body. `quant_in` must contain at least one non-zero coefficient
/// at a scan position (an all-zero commitment here is a caller bug —
/// the matching reader would never reach this body).
///
/// [`PartitionWalker::coefficients_gate_open`]: crate::cdf::PartitionWalker::coefficients_gate_open
#[allow(clippy::too_many_arguments)]
pub fn write_coefficients_gate_open(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    plane: u8,
    is_inter: u8,
    tx_size: usize,
    tx_class: usize,
    dc_sign_ctx: usize,
    scan: &[u16],
    quant_in: &[i32],
) -> Result<crate::cdf::CoefficientsReadout, Error> {
    // ---------------- caller-bug guards (mirror the reader's) ---------
    if tx_size >= TX_SIZES_ALL {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if tx_class > crate::cdf::TX_CLASS_VERT {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if plane > 2 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if is_inter > 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if dc_sign_ctx >= DC_SIGN_CONTEXTS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §5.11.39 lines 1-7: derived sizes (mirror the reader).
    let tx_w = TX_WIDTH[tx_size];
    let tx_h = crate::cdf::TX_HEIGHT[tx_size];
    let tx_sz_sqr = {
        let side = core::cmp::min(tx_w, tx_h);
        (side.trailing_zeros() as usize) - 2
    };
    let tx_sz_sqr_up = TX_SIZE_SQR_UP[tx_size];
    let tx_sz_ctx = (tx_sz_sqr + tx_sz_sqr_up + 1) >> 1;
    let ptype = (plane > 0) as usize;
    let seg_eob = if tx_size == crate::cdf::TX_16X64 || tx_size == crate::cdf::TX_64X16 {
        512usize
    } else {
        core::cmp::min(1024, tx_w * tx_h)
    };
    if scan.len() < seg_eob {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if quant_in.len() < tx_w * tx_h {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // Target eob from quant_in (§5.11.39 lines 56-71 reverse scan).
    let mut eob: u32 = 0;
    for c in (0..seg_eob).rev() {
        let pos = scan[c] as usize;
        if pos >= quant_in.len() {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if quant_in[pos] != 0 {
            eob = (c as u32) + 1;
            break;
        }
    }
    if eob == 0 {
        // All-zero commitments belong on the [`write_coefficients`] /
        // `write_txb_skip(all_zero = 1)` short-circuit, never here.
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.39 lines 19-55: eob_pt + eob_extra + eob_extra_bit loop.
    write_eob_pt(writer, cdfs, eob, tx_size, tx_class, plane, is_inter)?;

    // ---------------- §5.11.39 reverse-scan: base levels + br chain ---
    // Running magnitude buffer — mirrors the decoder's `Quant[]` state
    // as the reverse scan populates it. The §8.3.2 ctx helpers walk
    // this same buffer on both sides.
    let mut running: Vec<i32> = vec![0i32; tx_w * tx_h];

    let threshold_post_br: u32 = (NUM_BASE_LEVELS + COEFF_BASE_RANGE + 1) as u32; // = 15
    let threshold_golomb: u32 = (NUM_BASE_LEVELS + COEFF_BASE_RANGE) as u32; // = 14
    let max_base_eob: u8 = 2; // alphabet 0..=2 ⇒ level 1..=3
    let max_base: u8 = 3; // alphabet 0..=3 ⇒ level 0..=3

    {
        let mut c: i32 = eob as i32 - 1;
        while c >= 0 {
            let pos = scan[c as usize] as usize;
            // |q| >= 1 at c = eob - 1 (the §5.11.39 line-56 highest
            // non-zero); >= 0 below. Saturate at 15 — anything larger
            // gets the rest from the §5.11.39 lines 84-93 golomb tail.
            let target_abs: u32 = quant_in[pos].unsigned_abs();
            let coded_mag: u32 = core::cmp::min(target_abs, threshold_post_br);

            let is_eob = c == (eob as i32 - 1);
            // Sanity: the EOB position must be non-zero (line-56 says
            // so; the §8.2.6 reader debug_asserts this).
            if is_eob && coded_mag == 0 {
                return Err(Error::PartitionWalkOutOfRange);
            }

            let base_sym: u8 = if is_eob {
                // coeff_base_eob alphabet: 0..=2 ⇒ level 1..=3.
                // sym = min(coded_mag - 1, 2).
                core::cmp::min((coded_mag - 1) as u8, max_base_eob)
            } else {
                // coeff_base alphabet: 0..=3 ⇒ level 0..=3.
                // sym = min(coded_mag, 3).
                core::cmp::min(coded_mag as u8, max_base)
            };

            // §8.3.2 ctx derivations against the running magnitude
            // buffer (zero-filled at entry; populated by prior reverse-
            // scan iterations).
            if is_eob {
                let ctx = get_coeff_base_eob_ctx(&running, tx_size, tx_class, pos, c as usize);
                write_coeff_base_eob(writer, cdfs, base_sym, tx_sz_ctx, ptype, ctx)?;
            } else {
                let ctx = get_coeff_base_ctx(&running, tx_size, tx_class, pos, c as usize, false);
                write_coeff_base(writer, cdfs, base_sym, tx_sz_ctx, ptype, ctx)?;
            }

            // Level reconstructed exactly as the reader does it.
            let mut level: u32 = if is_eob {
                (base_sym as u32) + 1
            } else {
                base_sym as u32
            };

            // §5.11.39 lines 65-70: coeff_br chain when `level >
            // NUM_BASE_LEVELS`. Both reader and writer cap at
            // COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) = 4 iterations.
            if level as usize > NUM_BASE_LEVELS {
                let br_iters: u32 = (COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)) as u32;
                let br_step: u32 = (BR_CDF_SIZE - 1) as u32; // = 3
                for _ in 0..br_iters {
                    // remaining magnitude to encode in the chain.
                    let residue = coded_mag - level;
                    let sym: u8 = core::cmp::min(residue, br_step) as u8;
                    let ctx = get_br_ctx(&running, tx_size, tx_class, pos);
                    write_coeff_br(writer, cdfs, sym, tx_sz_ctx, ptype, ctx)?;
                    level += sym as u32;
                    if (sym as usize) < BR_CDF_SIZE - 1 {
                        break;
                    }
                }
            }

            // §5.11.39 line 71: stamp the post-chain magnitude into the
            // running buffer so subsequent iterations' ctx scans see it.
            running[pos] = level as i32;
            c -= 1;
        }
    }

    // ---------------- §5.11.39 forward-scan: signs + golomb ----------
    // `culLevel` / `dcCategory` mirror the reader's §5.11.39 lines
    // 94-102 derivations from the same forward walk.
    let mut cul_level: u32 = 0;
    let mut dc_category: u8 = 0;
    for (c, &pos_u16) in scan.iter().enumerate().take(eob as usize) {
        let pos = pos_u16 as usize;
        let target = quant_in[pos];
        let target_abs: u32 = target.unsigned_abs();
        let sign: u8 = if target < 0 { 1 } else { 0 };
        if target_abs != 0 {
            if c == 0 {
                write_dc_sign(writer, cdfs, sign, plane, dc_sign_ctx)?;
            } else {
                // §5.11.39 line 79: raw L(1) sign_bit.
                writer.write_literal(1, sign as u32)?;
            }
        }
        // §5.11.39 lines 84-93: golomb magnitude tail for magnitudes
        // past NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14.
        if target_abs > threshold_golomb {
            write_golomb(writer, target_abs)?;
        }
        // §5.11.39 lines 94-96: `dcCategory` derive on pos == 0 (the
        // reader's pre-clip `Quant[pos] > 0` gate is `target_abs > 0`
        // here — the writer's input is the reader's output).
        if pos == 0 && target_abs > 0 {
            dc_category = if sign != 0 { 1 } else { 2 };
        }
        // §5.11.39 lines 97-98: `Quant[pos] &= 0xFFFFF; culLevel +=
        // Quant[pos]`.
        cul_level = cul_level.saturating_add(target_abs & 0xFFFFF);
    }
    // §5.11.39 line 102: `culLevel = Min( 63, culLevel )`.
    let cul_level: u8 = core::cmp::min(63, cul_level) as u8;

    Ok(crate::cdf::CoefficientsReadout {
        all_zero: false,
        eob,
        cul_level,
        dc_category,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        get_br_ctx, get_coeff_base_ctx, get_coeff_base_eob_ctx, TileCdfContext, TX_16X16, TX_32X32,
        TX_4X4, TX_8X8, TX_CLASS_2D, TX_CLASS_HORIZ,
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

    // -----------------------------------------------------------------
    // §5.11.39 write_coeff_base_eob — round-trips through a single S()
    // read against the matching `TileCoeffBaseEobCdf` row.
    // -----------------------------------------------------------------

    /// Shim mirroring the §5.11.39 line-60 reader: one S() against
    /// `coeff_base_eob_cdf(tx_sz_ctx, ptype, ctx)`. Returns the §9.4
    /// symbol (0..=2); the spec maps `level = sym + 1`.
    fn read_coeff_base_eob(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        tx_sz_ctx: usize,
        ptype: usize,
        ctx: usize,
    ) -> u32 {
        let cdf = cdfs.coeff_base_eob_cdf(tx_sz_ctx, ptype, ctx).unwrap();
        dec.read_symbol(cdf).unwrap()
    }

    /// `coeff_base_eob = 0` (level 1) on the simplest ctx — round-trips.
    #[test]
    fn write_coeff_base_eob_zero_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_coeff_base_eob(&mut writer, &mut enc_cdfs, 0, 0, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sym = read_coeff_base_eob(&mut dec, &mut dec_cdfs, 0, 0, 0);
        assert_eq!(sym, 0);
    }

    /// `coeff_base_eob = 2` (level 3 — the largest base level codable
    /// via `coeff_base_eob` per §5.11.39 p.91) on TX_16X16 chroma at
    /// ctx 3 (upper edge of SIG_COEF_CONTEXTS_EOB) — round-trips.
    #[test]
    fn write_coeff_base_eob_max_sym_round_trip_chroma() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // tx_sz_ctx = 2 (TX_16X16's §5.11.39 line-4 derivation lands in
        // the 16-class bucket); ptype = 1 (chroma); ctx = 3
        // (SIG_COEF_CONTEXTS_EOB - 1).
        write_coeff_base_eob(&mut writer, &mut enc_cdfs, 2, 2, 1, 3).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sym = read_coeff_base_eob(&mut dec, &mut dec_cdfs, 2, 1, 3);
        assert_eq!(sym, 2);
    }

    /// Out-of-range `sym` (alphabet is 3 symbols) is a caller bug.
    #[test]
    fn write_coeff_base_eob_rejects_out_of_range_symbol() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_coeff_base_eob(&mut writer, &mut cdfs, 3, 0, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `ctx` (SIG_COEF_CONTEXTS_EOB = 4) is a caller bug.
    #[test]
    fn write_coeff_base_eob_rejects_out_of_range_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_coeff_base_eob(&mut writer, &mut cdfs, 0, 0, 0, SIG_COEF_CONTEXTS_EOB)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.39 write_coeff_base — round-trips through a single S()
    // read against the matching `TileCoeffBaseCdf` row.
    // -----------------------------------------------------------------

    /// Shim mirroring the §5.11.39 line-63 reader: one S() against
    /// `coeff_base_cdf(tx_sz_ctx, ptype, ctx)`. Returns the §9.4 symbol
    /// (0..=3) which is also the spec's `level`.
    fn read_coeff_base(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        tx_sz_ctx: usize,
        ptype: usize,
        ctx: usize,
    ) -> u32 {
        let cdf = cdfs.coeff_base_cdf(tx_sz_ctx, ptype, ctx).unwrap();
        dec.read_symbol(cdf).unwrap()
    }

    /// `coeff_base = 0` (level 0 — the coefficient is zero) on the
    /// simplest ctx — round-trips.
    #[test]
    fn write_coeff_base_zero_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_coeff_base(&mut writer, &mut enc_cdfs, 0, 0, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sym = read_coeff_base(&mut dec, &mut dec_cdfs, 0, 0, 0);
        assert_eq!(sym, 0);
    }

    /// `coeff_base = 3` (level 3 — the chain-continues sentinel per
    /// §5.11.39 line 64 `level > NUM_BASE_LEVELS` gate) on TX_32X32
    /// luma at ctx 41 (upper edge of SIG_COEF_CONTEXTS = 42) — round
    /// trips.
    #[test]
    fn write_coeff_base_continues_round_trip_luma_upper_ctx() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // tx_sz_ctx = 3 (TX_32X32); ptype = 0 (Y); ctx =
        // SIG_COEF_CONTEXTS - 1 = 41.
        write_coeff_base(&mut writer, &mut enc_cdfs, 3, 3, 0, SIG_COEF_CONTEXTS - 1).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sym = read_coeff_base(&mut dec, &mut dec_cdfs, 3, 0, SIG_COEF_CONTEXTS - 1);
        assert_eq!(sym, 3);
    }

    /// Out-of-range `sym` (alphabet is 4 symbols) is a caller bug.
    #[test]
    fn write_coeff_base_rejects_out_of_range_symbol() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_coeff_base(&mut writer, &mut cdfs, 4, 0, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `ctx` (SIG_COEF_CONTEXTS = 42) is a caller bug.
    #[test]
    fn write_coeff_base_rejects_out_of_range_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_coeff_base(&mut writer, &mut cdfs, 0, 0, 0, SIG_COEF_CONTEXTS).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // §5.11.39 write_coeff_br — round-trips through a single S() read
    // against the matching `TileCoeffBrCdf` row.
    // -----------------------------------------------------------------

    /// Shim mirroring the §5.11.39 lines 65-70 per-iteration reader:
    /// one S() against `coeff_br_cdf(tx_sz_ctx, ptype, ctx)`. Returns
    /// the §9.4 symbol (0..=BR_CDF_SIZE-1 = 0..=3).
    fn read_coeff_br(
        dec: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        tx_sz_ctx: usize,
        ptype: usize,
        ctx: usize,
    ) -> u32 {
        let cdf = cdfs.coeff_br_cdf(tx_sz_ctx, ptype, ctx).unwrap();
        dec.read_symbol(cdf).unwrap()
    }

    /// `coeff_br = 0` — the chain terminates after one iteration with
    /// no magnitude extension. Round-trips on the simplest ctx.
    #[test]
    fn write_coeff_br_zero_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_coeff_br(&mut writer, &mut enc_cdfs, 0, 0, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sym = read_coeff_br(&mut dec, &mut dec_cdfs, 0, 0, 0);
        assert_eq!(sym, 0);
    }

    /// `coeff_br = BR_CDF_SIZE - 1 = 3` — the chain-continues sentinel
    /// per §5.11.39 line 68 `if (coeff_br < BR_CDF_SIZE - 1) break`
    /// guard. On TX_8X8 chroma at ctx 20 (upper edge of
    /// LEVEL_CONTEXTS = 21).
    #[test]
    fn write_coeff_br_continues_round_trip_chroma_upper_ctx() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        write_coeff_br(
            &mut writer,
            &mut enc_cdfs,
            (BR_CDF_SIZE - 1) as u8,
            1, // tx_sz_ctx — TX_8X8.
            1, // ptype — chroma.
            LEVEL_CONTEXTS - 1,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sym = read_coeff_br(&mut dec, &mut dec_cdfs, 1, 1, LEVEL_CONTEXTS - 1);
        assert_eq!(sym, (BR_CDF_SIZE - 1) as u32);
    }

    /// `coeff_br` on TX_64X64 — the §8.3.2 selector clamps `txSzCtx`
    /// at `TX_32X32`, so a caller passing `tx_sz_ctx = TX_64X64 = 4`
    /// must still round-trip (the selector mirror in both writer +
    /// reader applies the same clamp).
    #[test]
    fn write_coeff_br_tx_size_clamp_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // tx_sz_ctx = 4 ≥ TX_32X32 = 3 ⇒ selector clamps to 3 inside
        // coeff_br_cdf; sym = 1.
        write_coeff_br(&mut writer, &mut enc_cdfs, 1, 4, 0, 0).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sym = read_coeff_br(&mut dec, &mut dec_cdfs, 4, 0, 0);
        assert_eq!(sym, 1);
    }

    /// Out-of-range `sym` (BR_CDF_SIZE = 4) is a caller bug.
    #[test]
    fn write_coeff_br_rejects_out_of_range_symbol() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_coeff_br(&mut writer, &mut cdfs, BR_CDF_SIZE as u8, 0, 0, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Out-of-range `ctx` (LEVEL_CONTEXTS = 21) is a caller bug.
    #[test]
    fn write_coeff_br_rejects_out_of_range_ctx() {
        let mut cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let err = write_coeff_br(&mut writer, &mut cdfs, 0, 0, 0, LEVEL_CONTEXTS).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    // -----------------------------------------------------------------
    // Driver-shape integration: cross-check the writer + §8.3.2 ctx
    // helpers stay in lockstep with the decoder's per-coefficient
    // ctx derivations. The driver loop itself is a follow-on arc; this
    // test just confirms a hand-built short sequence — coeff_base_eob
    // at c = eob - 1 followed by coeff_base + coeff_br at c = 0 — round
    // trips when both sides feed the same Quant[] into the §8.3.2
    // helpers.
    // -----------------------------------------------------------------

    /// Tiny synthetic block at TX_4X4 luma, TX_CLASS_2D, with eob = 2
    /// (two non-zero scan positions). At c = 1 (the EOB position) we
    /// write `coeff_base_eob = 0` (level 1); at c = 0 we write
    /// `coeff_base = 3` (level 3) followed by `coeff_br = 0` (no
    /// magnitude extension, chain terminates). Round-trips through
    /// `get_coeff_base_eob_ctx` / `get_coeff_base_ctx` / `get_br_ctx`.
    #[test]
    fn driver_shape_eob2_round_trip_tx4x4() {
        // §5.11.39 line 9: Quant[] zero-initialised. Per-position writes
        // happen during the reverse-scan loop; for the round-trip we
        // pre-populate the same Quant[] the §8.3.2 ctx helpers walk on
        // each side.
        let scan_pos_eob = 1usize; // scan[c = 1]
        let scan_pos_dc = 0usize; // scan[c = 0]

        // Reverse-scan, iteration 1 (c = eob - 1 = 1): Quant[] still
        // zero. After we record level = 1 at pos = 1.
        let mut quant_after_eob = [0i32; 16];
        quant_after_eob[scan_pos_eob] = 1;
        // Reverse-scan, iteration 2 (c = 0): Quant[] now has the EOB
        // level recorded. After we record level = 3 at pos = 0.
        // (`coeff_br = 0` ⇒ no further increment.)

        // §8.3.2 ctx derivations. Both sides compute these identically
        // from the running Quant[] array; we cache them here so the
        // round-trip test stays self-documenting.
        let ctx_eob = get_coeff_base_eob_ctx(&[0i32; 16], TX_4X4, TX_CLASS_2D, scan_pos_eob, 1);
        let ctx_base =
            get_coeff_base_ctx(&quant_after_eob, TX_4X4, TX_CLASS_2D, scan_pos_dc, 0, false);
        let ctx_br = get_br_ctx(&quant_after_eob, TX_4X4, TX_CLASS_2D, scan_pos_dc);

        // tx_sz_ctx for TX_4X4 == 0 (§5.11.39 line-4 derivation).
        let tx_sz_ctx = 0usize;
        let ptype = 0usize;

        // -------- encode --------
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // iteration 1: coeff_base_eob = 0 ⇒ level 1.
        write_coeff_base_eob(&mut writer, &mut enc_cdfs, 0, tx_sz_ctx, ptype, ctx_eob).unwrap();
        // iteration 2: coeff_base = 3 ⇒ level 3 (chain enters coeff_br).
        write_coeff_base(&mut writer, &mut enc_cdfs, 3, tx_sz_ctx, ptype, ctx_base).unwrap();
        // coeff_br = 0 ⇒ no increment, chain terminates.
        write_coeff_br(&mut writer, &mut enc_cdfs, 0, tx_sz_ctx, ptype, ctx_br).unwrap();
        let bytes = writer.finish();

        // -------- decode --------
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        // iteration 1 (c = eob - 1): coeff_base_eob.
        let sym_eob = {
            let cdf = dec_cdfs
                .coeff_base_eob_cdf(tx_sz_ctx, ptype, ctx_eob)
                .unwrap();
            dec.read_symbol(cdf).unwrap()
        };
        assert_eq!(sym_eob, 0, "coeff_base_eob round-trips");
        // iteration 2 (c = 0): coeff_base.
        let sym_base = {
            let cdf = dec_cdfs.coeff_base_cdf(tx_sz_ctx, ptype, ctx_base).unwrap();
            dec.read_symbol(cdf).unwrap()
        };
        assert_eq!(sym_base, 3, "coeff_base round-trips");
        // coeff_br for level extension.
        let sym_br = {
            let cdf = dec_cdfs.coeff_br_cdf(tx_sz_ctx, ptype, ctx_br).unwrap();
            dec.read_symbol(cdf).unwrap()
        };
        assert_eq!(sym_br, 0, "coeff_br round-trips");
    }

    // -----------------------------------------------------------------
    // §5.11.39 write_golomb — round-trips through the matching L(1)
    // loops (golomb_length_bit + golomb_data_bit) at the magnitude-tail
    // entry inside the forward-scan loop.
    // -----------------------------------------------------------------

    /// Shim mirroring the §5.11.39 lines 75-93 reader: a do-while unary
    /// length prefix followed by `length - 1` raw L(1) data bits.
    /// Returns the rebuilt `Quant[pos]` magnitude (`x + COEFF_BASE_RANGE
    /// + NUM_BASE_LEVELS`). The §6.10.34 conformance constraint allows
    /// `length` up to 20.
    fn read_golomb(dec: &mut SymbolDecoder<'_>) -> u32 {
        let mut length: u32 = 0;
        loop {
            length += 1;
            let bit = dec.read_literal(1).unwrap();
            if bit != 0 {
                break;
            }
            assert!(length <= GOLOMB_MAX_LENGTH, "length must terminate by 20");
        }
        let mut x: u32 = 1;
        if length >= 2 {
            let payload_bits = length - 1;
            for _ in 0..payload_bits {
                let bit = dec.read_literal(1).unwrap();
                x = (x << 1) | bit;
            }
        }
        x + (COEFF_BASE_RANGE as u32) + (NUM_BASE_LEVELS as u32)
    }

    /// `value = 15` — the smallest §5.11.39 magnitude that enters the
    /// golomb tail (just past the `NUM_BASE_LEVELS + COEFF_BASE_RANGE =
    /// 14` gate). `x = 1`, `length = 1` ⇒ just the unary terminator,
    /// no data bits. Round-trips.
    #[test]
    fn write_golomb_min_magnitude_round_trip() {
        let mut writer = SymbolWriter::new(false);
        write_golomb(&mut writer, 15).unwrap();
        let bytes = writer.finish();

        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let value = read_golomb(&mut dec);
        assert_eq!(value, 15);
    }

    /// `value = 16` — `x = 2`, `length = 2` ⇒ one zero, one terminator,
    /// one data bit = 0. Round-trips.
    #[test]
    fn write_golomb_value_16_round_trip() {
        let mut writer = SymbolWriter::new(false);
        write_golomb(&mut writer, 16).unwrap();
        let bytes = writer.finish();

        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let value = read_golomb(&mut dec);
        assert_eq!(value, 16);
    }

    /// `value = 21` — `x = 7 = 0b111`, `length = 3` ⇒ `00 1 11` (two
    /// zeros, terminator, two data bits = 1, 1). Round-trips and
    /// confirms MSB-first data emission.
    #[test]
    fn write_golomb_value_21_round_trip() {
        let mut writer = SymbolWriter::new(false);
        write_golomb(&mut writer, 21).unwrap();
        let bytes = writer.finish();

        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let value = read_golomb(&mut dec);
        assert_eq!(value, 21);
    }

    /// `value = 270` — `x = 256 = 0b1_0000_0000`, `length = 9`. Eight
    /// zero data bits below the implicit MSB. Round-trips and exercises
    /// the middle-of-range length.
    #[test]
    fn write_golomb_value_270_round_trip() {
        let mut writer = SymbolWriter::new(false);
        write_golomb(&mut writer, 270).unwrap();
        let bytes = writer.finish();

        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let value = read_golomb(&mut dec);
        assert_eq!(value, 270);
    }

    /// `value` exhaustively swept across small magnitudes — every
    /// representable magnitude with `length <= 4` (i.e. `x ∈ 1..=15`,
    /// `Quant[pos] ∈ 15..=29`) round-trips. This exercises every byte
    /// boundary the BitWriter padding might wander past on a single
    /// magnitude.
    #[test]
    fn write_golomb_small_magnitudes_round_trip_exhaustive() {
        for value in 15u32..=29 {
            let mut writer = SymbolWriter::new(false);
            write_golomb(&mut writer, value).unwrap();
            let bytes = writer.finish();

            let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
            let got = read_golomb(&mut dec);
            assert_eq!(got, value, "value {value} did not round-trip");
        }
    }

    /// `value = 1048589` — the maximum representable magnitude
    /// (`x = 0xFFFFF = 1_048_575`, `length = 20`). Confirms the
    /// §6.10.34 upper edge round-trips.
    #[test]
    fn write_golomb_max_magnitude_round_trip() {
        let value: u32 = (0xFFFFFu32) + (COEFF_BASE_RANGE as u32) + (NUM_BASE_LEVELS as u32);
        let mut writer = SymbolWriter::new(false);
        write_golomb(&mut writer, value).unwrap();
        let bytes = writer.finish();

        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = read_golomb(&mut dec);
        assert_eq!(got, value);
    }

    /// `value = 14` is a caller bug — the §5.11.39 line-73 gate
    /// requires `Quant[pos] > NUM_BASE_LEVELS + COEFF_BASE_RANGE` for
    /// the golomb tail to be entered at all.
    #[test]
    fn write_golomb_rejects_value_at_threshold() {
        let mut writer = SymbolWriter::new(false);
        let err = write_golomb(&mut writer, 14).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `value = 0` is also a caller bug — well below the gate.
    #[test]
    fn write_golomb_rejects_zero() {
        let mut writer = SymbolWriter::new(false);
        let err = write_golomb(&mut writer, 0).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `value = 1048590` exceeds the §6.10.34 conformance limit
    /// (`length` would be 21) — caller bug.
    #[test]
    fn write_golomb_rejects_value_past_conformance_limit() {
        let value: u32 = (0xFFFFFu32) + (COEFF_BASE_RANGE as u32) + (NUM_BASE_LEVELS as u32) + 1;
        let mut writer = SymbolWriter::new(false);
        let err = write_golomb(&mut writer, value).unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Sequence `dc_sign` then `write_golomb` — the natural pairing
    /// inside the §5.11.39 forward-scan loop at `c == 0` when the DC
    /// magnitude exceeds 14. Confirms the L(1) golomb bits compose
    /// correctly after a §8.2.6 S() sign emission.
    #[test]
    fn sequence_dc_sign_then_golomb_round_trip() {
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        // dc_sign = 1 (negative DC), luma ptype, ctx 0.
        write_dc_sign(&mut writer, &mut enc_cdfs, 1, 0, 0).unwrap();
        // golomb magnitude = 18 (x = 4, length = 3).
        write_golomb(&mut writer, 18).unwrap();
        let bytes = writer.finish();

        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let sign = {
            let cdf = dec_cdfs.dc_sign_cdf(0, 0).unwrap();
            dec.read_symbol(cdf).unwrap()
        };
        assert_eq!(sign, 1);
        let magnitude = read_golomb(&mut dec);
        assert_eq!(magnitude, 18);
    }

    // =================================================================
    // §5.11.39 driver loop — end-to-end round-trips through the
    // §5.11.39 `coefficients()` reader inside
    // `PartitionWalker::coefficients`.
    // =================================================================

    use crate::cdf::{PartitionWalker, TileGeometry, TX_4X4 as TX_4X4_C, TX_8X8 as TX_8X8_C};

    /// Helper: build a minimal `PartitionWalker` to host a
    /// `coefficients()` invocation. The reader is stateless on `self`
    /// (the body only reads `decoder` / `cdfs` / the caller args), so
    /// any well-formed walker works for the round-trip.
    fn make_walker() -> PartitionWalker {
        let geom = TileGeometry {
            mi_row_start: 0,
            mi_row_end: 8,
            mi_col_start: 0,
            mi_col_end: 8,
        };
        PartitionWalker::new(8, 8, geom).expect("walker construction")
    }

    /// Helper: encode `quant_in` with the driver, then decode the bytes
    /// back through `PartitionWalker::coefficients`, and return both the
    /// reader's `CoefficientsReadout` and the populated decoder
    /// `Quant[]` array. Both sides start with fresh §9.4 default CDFs
    /// and CDF adaptation enabled.
    #[allow(clippy::too_many_arguments)]
    fn drive_round_trip(
        plane: u8,
        is_inter: u8,
        tx_size: usize,
        tx_class: usize,
        txb_skip_ctx: usize,
        dc_sign_ctx: usize,
        scan: &[u16],
        quant_in: &[i32],
    ) -> (crate::cdf::CoefficientsReadout, Vec<i32>) {
        // ----- encode -----
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        let mut writer = SymbolWriter::new(false);
        let wout = write_coefficients(
            &mut writer,
            &mut enc_cdfs,
            plane,
            is_inter,
            tx_size,
            tx_class,
            txb_skip_ctx,
            dc_sign_ctx,
            scan,
            quant_in,
        )
        .expect("driver encode succeeded");
        let bytes = writer.finish();

        // ----- decode -----
        let mut walker = make_walker();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut quant_out = vec![0i32; quant_in.len()];
        let readout = walker
            .coefficients(
                &mut dec,
                &mut dec_cdfs,
                plane,
                is_inter,
                tx_size,
                tx_class,
                txb_skip_ctx,
                dc_sign_ctx,
                scan,
                &mut quant_out,
            )
            .expect("reader decode succeeded");

        // §5.11.39 line-13 short-circuit: all_zero path returns eob = 0;
        // matches the driver's gate-closed short-circuit return. The
        // writer's mirrored readout (`all_zero` / `eob` / `cul_level` /
        // `dc_category`) must equal the reader's on every path — the
        // §5.11.39 tail stamps both sides perform are fed from it.
        if quant_in.iter().all(|&q| q == 0) {
            assert_eq!(wout.eob, 0, "encoder eob == 0 for all-zero input");
            assert!(readout.all_zero, "decoder all_zero == true");
            assert_eq!(readout.eob, 0);
        } else {
            assert!(!readout.all_zero, "non-empty block must clear all_zero");
        }
        assert_eq!(
            wout, readout,
            "writer readout mirror must equal reader readout"
        );
        (readout, quant_out)
    }

    /// End-to-end round-trip 1: all-zero TX_4X4 luma block. The
    /// driver's §5.11.39 line-13 short-circuit emits a single
    /// `txb_skip = 1` S(); the reader returns
    /// `CoefficientsReadout { all_zero: true, eob: 0, .. }` and the
    /// `Quant[]` array is fully zero.
    #[test]
    fn driver_round_trip_all_zero_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let quant_in = vec![0i32; 16];
        let (_readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert!(quant_out.iter().all(|&q| q == 0));
    }

    /// End-to-end round-trip 2: TX_4X4 luma with a single non-zero
    /// coefficient at scan[0] (the DC). Magnitude 1, positive sign.
    /// Exercises the smallest non-trivial coefficient encode:
    /// `txb_skip = 0`, `eob_pt(eob = 1)`, one `coeff_base_eob` at
    /// `c = 0`, one `dc_sign`. No `coeff_br`, no `sign_bit`, no
    /// golomb.
    #[test]
    fn driver_round_trip_single_dc_one_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = 1;
        let (_readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(quant_out[0], 1);
        assert!(quant_out[1..].iter().all(|&q| q == 0));
    }

    /// End-to-end round-trip 3: TX_4X4 luma with two non-zero
    /// coefficients — DC = -1 and scan[1] = +2. Exercises the
    /// reverse-scan loop with two iterations (`coeff_base_eob` at
    /// `c = 1`, `coeff_base` at `c = 0`), the negative-DC `dc_sign`
    /// emission at `c = 0`, and a positive `sign_bit` L(1) at `c = 1`.
    #[test]
    fn driver_round_trip_two_coeffs_signed_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = -1;
        quant_in[1] = 2;
        let (_readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(quant_out[0], -1);
        assert_eq!(quant_out[1], 2);
        assert!(quant_out[2..].iter().all(|&q| q == 0));
    }

    /// End-to-end round-trip 4: TX_4X4 luma with magnitude 3 at DC,
    /// which forces the §5.11.39 line-64 `level > NUM_BASE_LEVELS = 2`
    /// gate open ⇒ enters the `coeff_br` chain (one iteration with
    /// `coeff_br = 0`, no extension). Confirms the chain-entry +
    /// immediate-terminate path.
    #[test]
    fn driver_round_trip_level3_no_br_extension_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = 3;
        let (_readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(quant_out[0], 3);
    }

    /// End-to-end round-trip 5: TX_4X4 luma with magnitude 8 at DC.
    /// `coeff_base_eob = 2` (level 3) + `coeff_br` chain of sym = 3,
    /// then sym = 2 (level 3 + 3 + 2 = 8). Exercises a mid-chain
    /// terminate after 2 iterations.
    #[test]
    fn driver_round_trip_level8_br_chain_terminate_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = 8;
        let (_readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(quant_out[0], 8);
    }

    /// End-to-end round-trip 6: TX_4X4 luma with magnitude 14 at DC —
    /// the §5.11.39 line-73 golomb-gate boundary. coeff_base_eob = 2 +
    /// four coeff_br iters (each sym = 3 or the last sym = 2; here
    /// 3+3+3+2 = 11 ⇒ level 14, full chain except last). No golomb.
    #[test]
    fn driver_round_trip_level14_boundary_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = 14;
        let (_readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(quant_out[0], 14);
    }

    /// End-to-end round-trip 7: TX_4X4 luma with magnitude 15 at DC.
    /// coeff_base_eob = 2 + four coeff_br iters each sym = 3 ⇒ level
    /// saturates at 15; then forward-scan golomb tail fires (x = 1,
    /// length = 1). Exercises the §5.11.39 line-73 golomb entry +
    /// minimal-magnitude golomb (single terminator bit, no data bits).
    #[test]
    fn driver_round_trip_level15_min_golomb_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = 15;
        let (_readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(quant_out[0], 15);
    }

    /// End-to-end round-trip 8: TX_4X4 luma with magnitude 30 at
    /// scan[1] (mid-block AC, not DC) — the forward-scan sign at c = 1
    /// is an L(1) `sign_bit`, not `dc_sign`. Magnitude saturates the
    /// base+br chain at 15 then golomb writes x = 16, length = 5.
    #[test]
    fn driver_round_trip_ac_golomb_with_sign_bit_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        // DC = 0 ⇒ no dc_sign, no first-coeff sign; only the c=1
        // sign_bit will be emitted.
        quant_in[1] = -30;
        let (readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(quant_out[1], -30);
        assert_eq!(quant_out[0], 0);
        assert_eq!(readout.eob, 2, "eob counts past the highest non-zero");
    }

    /// End-to-end round-trip 9: TX_8X8 luma with a small dense pattern
    /// across the first 6 scan positions, mixed signs, all magnitudes
    /// ≤ 3. Exercises the reverse-scan loop over several iterations
    /// with non-trivial Quant[] state feeding the §8.3.2 ctx walks on
    /// both sides.
    #[test]
    fn driver_round_trip_dense_small_pattern_tx8x8_luma() {
        let scan: Vec<u16> = (0..64u16).collect();
        let mut quant_in = vec![0i32; 64];
        quant_in[0] = 2;
        quant_in[1] = -1;
        quant_in[2] = 3;
        quant_in[3] = -2;
        quant_in[4] = 1;
        quant_in[5] = -1;
        let (readout, quant_out) =
            drive_round_trip(0, 0, TX_8X8_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(readout.eob, 6, "eob = 6 (highest non-zero at scan[5])");
        for (i, (&expect, &got)) in quant_in.iter().zip(quant_out.iter()).enumerate() {
            assert_eq!(
                got, expect,
                "Quant[{i}] mismatch: expected {expect}, got {got}"
            );
        }
    }

    /// End-to-end round-trip 10: TX_4X4 chroma U with positive DC and
    /// a single AC. Exercises the `plane = 1` / `ptype = 1` axis —
    /// distinct CDF rows on both sides — plus `dc_sign_ctx = 1` and
    /// `txb_skip_ctx = 5` (off the (0, 0) origin).
    #[test]
    fn driver_round_trip_chroma_ptype_axis_tx4x4() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = 4;
        quant_in[2] = -1;
        let (_readout, quant_out) =
            drive_round_trip(1, 0, TX_4X4_C, TX_CLASS_2D, 5, 1, &scan, &quant_in);
        assert_eq!(quant_out[0], 4);
        assert_eq!(quant_out[2], -1);
    }

    /// End-to-end round-trip 11: TX_4X4 luma with a large golomb-tail
    /// magnitude (200) at scan[2] (AC). x = 186 = 0b1011_1010,
    /// length = 8. Exercises a wide golomb data payload composed with
    /// a normal-magnitude DC and base-only AC at c = 1.
    #[test]
    fn driver_round_trip_large_golomb_tx4x4_luma() {
        let scan: Vec<u16> = (0..16u16).collect();
        let mut quant_in = vec![0i32; 16];
        quant_in[0] = 1;
        quant_in[1] = -2;
        quant_in[2] = 200;
        let (readout, quant_out) =
            drive_round_trip(0, 0, TX_4X4_C, TX_CLASS_2D, 0, 0, &scan, &quant_in);
        assert_eq!(readout.eob, 3);
        assert_eq!(quant_out[0], 1);
        assert_eq!(quant_out[1], -2);
        assert_eq!(quant_out[2], 200);
    }

    /// Driver caller-bug guards: each out-of-range argument returns
    /// [`Error::PartitionWalkOutOfRange`] without producing partial
    /// output (the SymbolWriter may have buffered an early write
    /// before the guard fires; the test asserts on the Result, not on
    /// bytes).
    #[test]
    fn driver_rejects_out_of_range_arguments() {
        let scan: Vec<u16> = (0..16u16).collect();
        let quant_in = vec![0i32; 16];
        let mut cdfs = TileCdfContext::new_from_defaults();

        // tx_size out of range.
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(
            &mut w,
            &mut cdfs,
            0,
            0,
            TX_SIZES_ALL,
            TX_CLASS_2D,
            0,
            0,
            &scan,
            &quant_in,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // tx_class out of range (TX_CLASS_VERT = 2; 3 is illegal).
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(&mut w, &mut cdfs, 0, 0, TX_4X4_C, 3, 0, 0, &scan, &quant_in)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // plane out of range.
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(
            &mut w,
            &mut cdfs,
            3,
            0,
            TX_4X4_C,
            TX_CLASS_2D,
            0,
            0,
            &scan,
            &quant_in,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // is_inter out of range.
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(
            &mut w,
            &mut cdfs,
            0,
            2,
            TX_4X4_C,
            TX_CLASS_2D,
            0,
            0,
            &scan,
            &quant_in,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // txb_skip_ctx out of range.
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(
            &mut w,
            &mut cdfs,
            0,
            0,
            TX_4X4_C,
            TX_CLASS_2D,
            TXB_SKIP_CONTEXTS,
            0,
            &scan,
            &quant_in,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // dc_sign_ctx out of range.
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(
            &mut w,
            &mut cdfs,
            0,
            0,
            TX_4X4_C,
            TX_CLASS_2D,
            0,
            DC_SIGN_CONTEXTS,
            &scan,
            &quant_in,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // scan too short.
        let short_scan: Vec<u16> = (0..4u16).collect();
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(
            &mut w,
            &mut cdfs,
            0,
            0,
            TX_4X4_C,
            TX_CLASS_2D,
            0,
            0,
            &short_scan,
            &quant_in,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));

        // quant_in too small.
        let tiny_quant: Vec<i32> = vec![0i32; 4];
        let mut w = SymbolWriter::new(false);
        let err = write_coefficients(
            &mut w,
            &mut cdfs,
            0,
            0,
            TX_4X4_C,
            TX_CLASS_2D,
            0,
            0,
            &scan,
            &tiny_quant,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }
}
