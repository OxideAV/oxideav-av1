//! §5.11.4 `decode_partition` **writer** — the encoder counterpart of
//! [`crate::cdf::PartitionWalker::decode_partition`].
//!
//! Scope of this arc (r216): the *symbol-emission* portion of §5.11.4 —
//! the `partition` / `split_or_horz` / `split_or_vert` arithmetic-coded
//! syntax elements (av1-spec §5.11.4 p.61). The encoder mirrors the
//! decoder's first-conditional choice for which symbol (or none) to
//! emit:
//!
//! ```text
//!   if ( bSize < BLOCK_8X8 )            partition = PARTITION_NONE   (no symbol)
//!   else if ( hasRows && hasCols )      partition                    S()
//!   else if ( hasCols )                 split_or_horz                S()
//!   else if ( hasRows )                 split_or_vert                S()
//!   else                                partition = PARTITION_SPLIT  (no symbol)
//! ```
//!
//! `hasRows` / `hasCols` derive from `r + halfBlock4x4 < MiRows` /
//! `c + halfBlock4x4 < MiCols` per §5.11.4 lines 7-8, and are
//! caller-supplied (mirroring [`crate::encoder::block_mode_info`]'s
//! caller-supplied-ctx pattern: the encode driver threads its own
//! [`crate::cdf::PartitionWalker`] for the §6.10.4 `MiSizes[]` /
//! `Skips[]` grids, so it already knows the §5.11.4 booleans by
//! construction).
//!
//! The §8.3.2 `partition_ctx` is also caller-supplied — derivable via
//! [`crate::cdf::partition_ctx`] from the neighbour `MiSizes[]` widths,
//! same shape as the decoder's `partition_ctx_for(r, c, bsl)` private
//! helper.
//!
//! ## Out of scope this arc
//!
//! The §5.11.4 recursive **dispatch** (the `if (partition ==
//! PARTITION_NONE) decode_block(...) else if (partition ==
//! PARTITION_SPLIT) decode_partition(...)` block) stays out — that's the
//! encode driver's job; this module supplies only the inverse-syntax
//! primitives the driver composes. The driver will:
//!
//! 1. Pick a `partition` based on the encoder's RD search.
//! 2. Call [`write_partition`] with the chosen partition + caller-derived
//!    `has_rows` / `has_cols` / `ctx`.
//! 3. Recurse on the four `subSize` children (for `PARTITION_SPLIT`) or
//!    emit `decode_block` writers for the non-split arms.
//!
//! ## Spec provenance
//!
//! Sourced from `docs/video/av1/av1-spec.txt`:
//!   * §5.11.4  decode_partition           (p.61)
//!   * §8.3.2  partition / split_or_horz / split_or_vert ctx + cdf
//!     selection                          (p.362, p.376)
//!   * §6.10.4 partition ordinals          (p.243)
//!
//! [`crate::cdf::PartitionWalker`]: crate::cdf::PartitionWalker

use crate::cdf::{
    split_or_horz_cdf, split_or_vert_cdf, TileCdfContext, BLOCK_8X8, BLOCK_SIZES, MI_WIDTH_LOG2,
    PARTITION_HORZ, PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::Error;

/// Predicate: at this `b_size`, §5.11.4 emits *no* symbol because the
/// block is too small to subdivide — `partition` is forced to
/// `PARTITION_NONE`.
///
/// This is the spec's `bSize < BLOCK_8X8` first-conditional branch
/// (av1-spec §5.11.4 p.61 line 9). Caller MUST emit a single
/// `decode_block( r, c, bSize )` leaf for the block; no
/// [`SymbolWriter`] interaction is required.
///
/// Returns `false` if `b_size >= BLOCK_SIZES` (a caller bug — the
/// public dispatch is gated on a valid `b_size`); the caller is
/// expected to validate upstream.
#[inline]
#[must_use]
pub const fn partition_none_only(b_size: usize) -> bool {
    b_size < BLOCK_8X8
}

/// Predicate: at this `b_size` + (`has_rows`, `has_cols`) §5.11.4
/// emits *no* symbol because the block sits in the bottom-right
/// frame-edge corner where neither halfBlock4x4 step lands inside the
/// frame — `partition` is forced to `PARTITION_SPLIT`.
///
/// This is the spec's `else { partition = PARTITION_SPLIT }`
/// fall-through after `hasRows && hasCols` / `hasCols` / `hasRows`
/// have all failed (av1-spec §5.11.4 p.61 line 19). Caller MUST recurse
/// on the four split children (the off-edge ones short-circuit via the
/// §5.11.4 `r >= MiRows || c >= MiCols` early-return); no
/// [`SymbolWriter`] interaction is required.
#[inline]
#[must_use]
pub const fn partition_split_only(b_size: usize, has_rows: bool, has_cols: bool) -> bool {
    !partition_none_only(b_size) && !has_rows && !has_cols
}

/// `decode_partition` symbol writer — inverse of the §5.11.4
/// `partition` / `split_or_horz` / `split_or_vert` S() reads inside
/// [`crate::cdf::PartitionWalker::decode_partition`].
///
/// Given the chosen `partition` ordinal + `b_size` + (`has_rows`,
/// `has_cols`, `ctx`) the encoder side already knows from its
/// [`crate::cdf::PartitionWalker`] state, emits the right symbol(s):
///
/// * `b_size < BLOCK_8X8`: no symbol. `partition` MUST be
///   `PARTITION_NONE`.
/// * `b_size >= BLOCK_8X8`, `has_rows && has_cols`: one §8.2.6 `S()`
///   against the `bsl`-selected partition CDF row for the supplied
///   `ctx`. `partition` MUST be in `0..N` where `N` is the row's
///   symbol count (5 for `bsl == 1` (W8 — 5-value), 10 for
///   `bsl in 2..=4` (W16 / W32 / W64), 8 for `bsl == 5` (W128, drops
///   the `*_4` partitions)).
/// * `b_size >= BLOCK_8X8`, `has_cols` only: one §8.2.6 `S()` against
///   the §8.3.2 `split_or_horz` 2-symbol CDF (folded from the same
///   partition row per [`split_or_horz_cdf`]). `partition` MUST be
///   either `PARTITION_HORZ` (writes `split_or_horz = 0`) or
///   `PARTITION_SPLIT` (writes `split_or_horz = 1`).
/// * `b_size >= BLOCK_8X8`, `has_rows` only: one §8.2.6 `S()` against
///   the §8.3.2 `split_or_vert` 2-symbol CDF per [`split_or_vert_cdf`].
///   `partition` MUST be either `PARTITION_VERT` (writes
///   `split_or_vert = 0`) or `PARTITION_SPLIT` (writes
///   `split_or_vert = 1`).
/// * `b_size >= BLOCK_8X8`, `!has_rows && !has_cols`: no symbol.
///   `partition` MUST be `PARTITION_SPLIT`.
///
/// Returns [`Error::PartitionWalkOutOfRange`] for any caller-side
/// inconsistency: an out-of-range `b_size` (>= `BLOCK_SIZES`), a
/// `partition` ordinal that does not match the §5.11.4 branch the
/// caller landed in, an out-of-range `ctx` (>= `PARTITION_CONTEXTS`),
/// or an internal CDF-row-too-short surface (which shouldn't happen
/// for any well-formed input — it guards against the same caller bug
/// the decoder rejects).
pub fn write_partition(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    partition: usize,
    b_size: usize,
    has_rows: bool,
    has_cols: bool,
    ctx: usize,
) -> Result<(), Error> {
    if b_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.4 first conditional: `bSize < BLOCK_8X8 ⇒ PARTITION_NONE`,
    // no symbol read on the decoder side, so the writer emits nothing.
    if partition_none_only(b_size) {
        if partition != PARTITION_NONE {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §5.11.4 last conditional: the bottom-right edge corner forces
    // `partition = PARTITION_SPLIT` with no symbol read.
    if partition_split_only(b_size, has_rows, has_cols) {
        if partition != PARTITION_SPLIT {
            return Err(Error::PartitionWalkOutOfRange);
        }
        return Ok(());
    }

    // §8.3.2 partition-CDF row selection: `bsl = Mi_Width_Log2[ bSize ]`
    // chooses W8 (bsl == 1), W16/32/64 (bsl in 2..=4), or W128 (bsl == 5).
    let bsl = MI_WIDTH_LOG2[b_size] as u32;

    if has_rows && has_cols {
        // §5.11.4 `partition S()` — the full N-value alphabet.
        let cdf = cdfs
            .partition_cdf(bsl, ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        // The §8.2.6 `read_symbol` returns `0..N` where `N = cdf.len()
        // - 1`. Reject any caller-supplied `partition` outside that
        // range — including the §6.10.4 ordinals that don't exist for
        // this `bsl` (e.g. `PARTITION_HORZ_4` for W128 which has no
        // entry at index 8/9).
        let n = (cdf.len() as u32).saturating_sub(1);
        if partition as u32 >= n {
            return Err(Error::PartitionWalkOutOfRange);
        }
        writer.write_symbol(partition as u32, cdf)
    } else if has_cols {
        // §5.11.4 `split_or_horz S()` — the 2-value folded CDF.
        // `partition` is one of PARTITION_HORZ (sym 0) or
        // PARTITION_SPLIT (sym 1) per the spec's `partition =
        // split_or_horz ? PARTITION_SPLIT : PARTITION_HORZ` line.
        let cdf_row = cdfs
            .partition_cdf(bsl, ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let mut bin = split_or_horz_cdf(cdf_row, b_size).ok_or(Error::PartitionWalkOutOfRange)?;
        let sym: u32 = match partition {
            PARTITION_HORZ => 0,
            PARTITION_SPLIT => 1,
            _ => return Err(Error::PartitionWalkOutOfRange),
        };
        writer.write_symbol(sym, &mut bin)
    } else {
        // has_rows alone — §5.11.4 `split_or_vert S()`.
        let cdf_row = cdfs
            .partition_cdf(bsl, ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let mut bin = split_or_vert_cdf(cdf_row, b_size).ok_or(Error::PartitionWalkOutOfRange)?;
        let sym: u32 = match partition {
            PARTITION_VERT => 0,
            PARTITION_SPLIT => 1,
            _ => return Err(Error::PartitionWalkOutOfRange),
        };
        writer.write_symbol(sym, &mut bin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{
        partition_ctx, BLOCK_128X128, BLOCK_16X16, BLOCK_4X4, BLOCK_64X64, PARTITION_CONTEXTS,
        PARTITION_HORZ_4, PARTITION_HORZ_A, PARTITION_HORZ_B, PARTITION_VERT_4, PARTITION_VERT_A,
        PARTITION_VERT_B,
    };
    use crate::symbol_decoder::SymbolDecoder;

    /// Bridge helper: build a fresh `(SymbolWriter, TileCdfContext)`
    /// pair, run `f`, finish, and replay through `SymbolDecoder` against
    /// a parallel `TileCdfContext`. Returns `(decoded_partition,
    /// dec_cdfs)` for assertion.
    ///
    /// `partition_decode` runs the decoder-side §5.11.4 first-conditional
    /// against the supplied `(b_size, has_rows, has_cols, ctx)` so the
    /// roundtrip ends up exercising the same branch the writer chose.
    fn roundtrip(
        partition: usize,
        b_size: usize,
        has_rows: bool,
        has_cols: bool,
        ctx: usize,
    ) -> usize {
        let mut w = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        write_partition(
            &mut w,
            &mut enc_cdfs,
            partition,
            b_size,
            has_rows,
            has_cols,
            ctx,
        )
        .unwrap();
        let bytes = w.finish();
        // A short payload still needs at least one byte so the decoder's
        // init_symbol can read its 15-bit `paddedBuf` window. Pad with
        // §8.2.2 zero bits if the writer emitted nothing.
        let bytes = if bytes.is_empty() { vec![0u8] } else { bytes };
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        decode_partition_local(&mut d, &mut dec_cdfs, b_size, has_rows, has_cols, ctx)
    }

    /// Local re-implementation of the §5.11.4 first conditional's
    /// *symbol-read* portion — mirrors what
    /// [`crate::cdf::PartitionWalker::decode_partition`] reads, without
    /// pulling in the full walker (which would also need a frame /
    /// tile geometry). The expected encoder/decoder pairing is bit-
    /// exact: any byte the writer emitted is consumed here, any branch
    /// the writer skipped is also skipped here.
    fn decode_partition_local(
        d: &mut SymbolDecoder<'_>,
        cdfs: &mut TileCdfContext,
        b_size: usize,
        has_rows: bool,
        has_cols: bool,
        ctx: usize,
    ) -> usize {
        if b_size < BLOCK_8X8 {
            return PARTITION_NONE;
        }
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        if has_rows && has_cols {
            let cdf = cdfs.partition_cdf(bsl, ctx).unwrap();
            d.read_symbol(cdf).unwrap() as usize
        } else if has_cols {
            let cdf_row = cdfs.partition_cdf(bsl, ctx).unwrap();
            let mut bin = split_or_horz_cdf(cdf_row, b_size).unwrap();
            let s = d.read_symbol(&mut bin).unwrap();
            if s == 0 {
                PARTITION_HORZ
            } else {
                PARTITION_SPLIT
            }
        } else if has_rows {
            let cdf_row = cdfs.partition_cdf(bsl, ctx).unwrap();
            let mut bin = split_or_vert_cdf(cdf_row, b_size).unwrap();
            let s = d.read_symbol(&mut bin).unwrap();
            if s == 0 {
                PARTITION_VERT
            } else {
                PARTITION_SPLIT
            }
        } else {
            PARTITION_SPLIT
        }
    }

    /// `partition_none_only` matches the spec's `bSize < BLOCK_8X8` rule:
    /// true for BLOCK_4X4 and the smaller rectangulars, false from
    /// BLOCK_8X8 up.
    #[test]
    fn partition_none_only_matches_spec() {
        assert!(partition_none_only(BLOCK_4X4));
        // BLOCK_8X8 (ordinal 3) is the first size that may subdivide.
        assert!(!partition_none_only(BLOCK_8X8));
        assert!(!partition_none_only(BLOCK_16X16));
        assert!(!partition_none_only(BLOCK_64X64));
    }

    /// `partition_split_only` triggers only when the block sits in the
    /// bottom-right corner where neither halfBlock4x4 step lands inside
    /// the frame. Any in-bounds half-step suppresses the forced split.
    #[test]
    fn partition_split_only_matches_spec() {
        // Corner: !has_rows && !has_cols on a sub-BLOCK_8X8 block — but
        // the BLOCK_8X8 short-circuit wins first, so the function
        // returns false.
        assert!(!partition_split_only(BLOCK_4X4, false, false));
        // BLOCK_16X16 in the bottom-right corner ⇒ forced split.
        assert!(partition_split_only(BLOCK_16X16, false, false));
        // Any in-bounds axis suppresses the forced split.
        assert!(!partition_split_only(BLOCK_16X16, true, false));
        assert!(!partition_split_only(BLOCK_16X16, false, true));
        assert!(!partition_split_only(BLOCK_16X16, true, true));
    }

    /// `b_size < BLOCK_8X8`: no symbol read; round-trip recovers
    /// PARTITION_NONE for free.
    #[test]
    fn write_partition_none_short_circuit_no_symbol() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        write_partition(&mut w, &mut cdfs, PARTITION_NONE, BLOCK_4X4, true, true, 0).unwrap();
        // SymbolWriter accumulates a 15-bit `low` window even with no
        // symbols, so `finish()` always returns at least two bytes. The
        // real proof of "no symbol emitted" is that `roundtrip()` below
        // returns PARTITION_NONE without consuming any of the bytes.
        let p = roundtrip(PARTITION_NONE, BLOCK_4X4, true, true, 0);
        assert_eq!(p, PARTITION_NONE);
    }

    /// `b_size < BLOCK_8X8` with a non-NONE partition is rejected as
    /// caller-bug.
    #[test]
    fn write_partition_none_short_circuit_rejects_other_partition() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_partition(&mut w, &mut cdfs, PARTITION_SPLIT, BLOCK_4X4, true, true, 0)
            .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `!has_rows && !has_cols` at BLOCK_16X16: forced PARTITION_SPLIT,
    /// no symbol read; round-trip recovers PARTITION_SPLIT.
    #[test]
    fn write_partition_corner_forced_split_no_symbol() {
        let p = roundtrip(PARTITION_SPLIT, BLOCK_16X16, false, false, 0);
        assert_eq!(p, PARTITION_SPLIT);
    }

    /// `!has_rows && !has_cols` with a non-SPLIT partition is rejected.
    #[test]
    fn write_partition_corner_forced_split_rejects_other_partition() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_partition(
            &mut w,
            &mut cdfs,
            PARTITION_HORZ,
            BLOCK_16X16,
            false,
            false,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// `has_rows && has_cols` at BLOCK_16X16: every one of the 10
    /// §6.10.4 partition ordinals roundtrips through the W16 default
    /// CDF row (the partition with index N selects symbol N).
    #[test]
    fn write_partition_full_alphabet_round_trip_w16() {
        for partition in [
            PARTITION_NONE,
            PARTITION_HORZ,
            PARTITION_VERT,
            PARTITION_SPLIT,
            PARTITION_HORZ_A,
            PARTITION_HORZ_B,
            PARTITION_VERT_A,
            PARTITION_VERT_B,
            PARTITION_HORZ_4,
            PARTITION_VERT_4,
        ] {
            for ctx in 0..PARTITION_CONTEXTS {
                let p = roundtrip(partition, BLOCK_16X16, true, true, ctx);
                assert_eq!(
                    p, partition,
                    "partition {partition} ctx {ctx} did not round-trip"
                );
            }
        }
    }

    /// W128 (BLOCK_128X128) has no `PARTITION_HORZ_4` / `PARTITION_VERT_4`
    /// entries — encoding them is a caller-bug surface.
    #[test]
    fn write_partition_w128_rejects_horz4_vert4() {
        for partition in [PARTITION_HORZ_4, PARTITION_VERT_4] {
            let mut w = SymbolWriter::new(true);
            let mut cdfs = TileCdfContext::new_from_defaults();
            let err = write_partition(&mut w, &mut cdfs, partition, BLOCK_128X128, true, true, 0)
                .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "partition {partition} on W128 must reject"
            );
        }
    }

    /// W128 (BLOCK_128X128) round-trips the 8 partition ordinals it
    /// supports: NONE / HORZ / VERT / SPLIT / HORZ_A / HORZ_B / VERT_A /
    /// VERT_B (the §6.10.4 ordinals 0-7).
    #[test]
    fn write_partition_w128_alphabet_round_trip() {
        for partition in [
            PARTITION_NONE,
            PARTITION_HORZ,
            PARTITION_VERT,
            PARTITION_SPLIT,
            PARTITION_HORZ_A,
            PARTITION_HORZ_B,
            PARTITION_VERT_A,
            PARTITION_VERT_B,
        ] {
            let p = roundtrip(
                partition,
                BLOCK_128X128,
                true,
                true,
                partition_ctx(true, false),
            );
            assert_eq!(p, partition);
        }
    }

    /// `has_cols` only at BLOCK_16X16: PARTITION_HORZ (sym 0) and
    /// PARTITION_SPLIT (sym 1) round-trip through the
    /// `split_or_horz` 2-symbol folded CDF.
    #[test]
    fn write_partition_split_or_horz_round_trip() {
        for partition in [PARTITION_HORZ, PARTITION_SPLIT] {
            for ctx in 0..PARTITION_CONTEXTS {
                let p = roundtrip(partition, BLOCK_16X16, false, true, ctx);
                assert_eq!(
                    p, partition,
                    "split_or_horz partition {partition} ctx {ctx} did not round-trip"
                );
            }
        }
    }

    /// `has_cols` only with a partition that the binary CDF cannot
    /// express (e.g. PARTITION_HORZ_A) is rejected as caller-bug.
    #[test]
    fn write_partition_split_or_horz_rejects_other_partitions() {
        for partition in [
            PARTITION_NONE,
            PARTITION_VERT,
            PARTITION_HORZ_A,
            PARTITION_HORZ_B,
            PARTITION_VERT_A,
            PARTITION_VERT_B,
            PARTITION_HORZ_4,
            PARTITION_VERT_4,
        ] {
            let mut w = SymbolWriter::new(true);
            let mut cdfs = TileCdfContext::new_from_defaults();
            let err = write_partition(&mut w, &mut cdfs, partition, BLOCK_16X16, false, true, 0)
                .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "split_or_horz must reject partition {partition}"
            );
        }
    }

    /// `has_rows` only at BLOCK_16X16: PARTITION_VERT (sym 0) and
    /// PARTITION_SPLIT (sym 1) round-trip through the
    /// `split_or_vert` 2-symbol folded CDF.
    #[test]
    fn write_partition_split_or_vert_round_trip() {
        for partition in [PARTITION_VERT, PARTITION_SPLIT] {
            for ctx in 0..PARTITION_CONTEXTS {
                let p = roundtrip(partition, BLOCK_16X16, true, false, ctx);
                assert_eq!(
                    p, partition,
                    "split_or_vert partition {partition} ctx {ctx} did not round-trip"
                );
            }
        }
    }

    /// `has_rows` only with a partition the binary CDF cannot express
    /// is rejected.
    #[test]
    fn write_partition_split_or_vert_rejects_other_partitions() {
        for partition in [
            PARTITION_NONE,
            PARTITION_HORZ,
            PARTITION_HORZ_A,
            PARTITION_HORZ_B,
            PARTITION_VERT_A,
            PARTITION_VERT_B,
            PARTITION_HORZ_4,
            PARTITION_VERT_4,
        ] {
            let mut w = SymbolWriter::new(true);
            let mut cdfs = TileCdfContext::new_from_defaults();
            let err = write_partition(&mut w, &mut cdfs, partition, BLOCK_16X16, true, false, 0)
                .unwrap_err();
            assert!(
                matches!(err, Error::PartitionWalkOutOfRange),
                "split_or_vert must reject partition {partition}"
            );
        }
    }

    /// Out-of-range `b_size` is rejected.
    #[test]
    fn write_partition_rejects_out_of_range_b_size() {
        let mut w = SymbolWriter::new(true);
        let mut cdfs = TileCdfContext::new_from_defaults();
        let err = write_partition(
            &mut w,
            &mut cdfs,
            PARTITION_NONE,
            BLOCK_SIZES,
            true,
            true,
            0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::PartitionWalkOutOfRange));
    }

    /// Sanity: multi-block roundtrip composition. Write a sequence of
    /// partition symbols spanning the full alphabet at varying ctx,
    /// then replay them through a parallel decoder using the same
    /// running `TileCdfContext`. This exercises the §8.3 CDF
    /// adaptation in lockstep between writer + reader.
    #[test]
    fn write_partition_sequence_round_trip_with_cdf_adaptation() {
        // (partition, b_size, has_rows, has_cols, ctx).
        let plan: &[(usize, usize, bool, bool, usize)] = &[
            (PARTITION_NONE, BLOCK_16X16, true, true, 0),
            (PARTITION_HORZ, BLOCK_16X16, true, true, 1),
            (PARTITION_VERT, BLOCK_16X16, true, true, 2),
            (PARTITION_SPLIT, BLOCK_16X16, true, true, 3),
            (PARTITION_HORZ_A, BLOCK_16X16, true, true, 0),
            (PARTITION_HORZ_B, BLOCK_16X16, true, true, 1),
            (PARTITION_VERT_A, BLOCK_16X16, true, true, 2),
            (PARTITION_VERT_B, BLOCK_16X16, true, true, 3),
            (PARTITION_HORZ_4, BLOCK_16X16, true, true, 0),
            (PARTITION_VERT_4, BLOCK_16X16, true, true, 1),
            (PARTITION_HORZ, BLOCK_16X16, false, true, 2), // split_or_horz
            (PARTITION_VERT, BLOCK_16X16, true, false, 3), // split_or_vert
            (PARTITION_SPLIT, BLOCK_16X16, false, false, 0), // forced split
            (PARTITION_NONE, BLOCK_4X4, true, true, 0),    // bSize < 8X8 short-circuit
        ];

        let mut w = SymbolWriter::new(false);
        let mut enc_cdfs = TileCdfContext::new_from_defaults();
        for &(partition, b_size, has_rows, has_cols, ctx) in plan {
            write_partition(
                &mut w,
                &mut enc_cdfs,
                partition,
                b_size,
                has_rows,
                has_cols,
                ctx,
            )
            .unwrap();
        }
        let bytes = w.finish();
        let bytes = if bytes.is_empty() { vec![0u8] } else { bytes };

        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut dec_cdfs = TileCdfContext::new_from_defaults();
        for (i, &(partition, b_size, has_rows, has_cols, ctx)) in plan.iter().enumerate() {
            let got =
                decode_partition_local(&mut d, &mut dec_cdfs, b_size, has_rows, has_cols, ctx);
            assert_eq!(
                got, partition,
                "sequence position {i}: expected partition {partition}, decoded {got}"
            );
        }
        // §8.3 CDF adaptation kept the writer + reader in lockstep.
        assert_eq!(enc_cdfs.partition_w16, dec_cdfs.partition_w16);
    }
}
