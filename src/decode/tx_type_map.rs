//! AV1 transform-type mapping — §6.10.15 / §5.11.48.
//!
//! Two intra extended-tx sets are used depending on TX area:
//!
//! - `EXT_TX_SET_INTRA_1` (txSet=1, 7 types): used for TX blocks of
//!   area ≤ 16×16.
//! - `EXT_TX_SET_INTRA_2` (txSet=2, 5 types): used for TX blocks of
//!   area ≤ 32×32.
//! - Implicit `DCT_DCT` (txSet=0): for TX > 32×32.
//!
//! Phase 2 never reads `tx_type` from the bitstream (the tile decoder
//! exits before coefficient decode), but the mapping is ported here
//! so Phase 3 can wire it up verbatim.
//!
//! Round 24 — inter mapping groundwork. The spec also defines three
//! inter extended-tx sets (§5.11.48 `get_tx_set` + §6.10.15 inversion
//! tables `Tx_Type_Inter_Inv_Set{1,2,3}`):
//!
//! - `TX_SET_INTER_1` (16 types): TX blocks with `txSzSqr ≤ TX_8X8`
//!   when neither `reduced_tx_set` nor `txSzSqrUp == TX_32X32`.
//! - `TX_SET_INTER_2` (12 types): TX blocks with `txSzSqr == TX_16X16`
//!   under the same gates.
//! - `TX_SET_INTER_3` (2 types — `IDTX`/`DCT_DCT`): TX blocks with
//!   `reduced_tx_set` set OR `txSzSqrUp == TX_32X32`.
//! - `TX_SET_DCTONLY` (implicit `DCT_DCT`, no symbol): TX blocks with
//!   `txSzSqrUp > TX_32X32`.
//!
//! Round 24 lands the inverse tables + `get_tx_set` selector for the
//! inter side. Wiring it into `decode/superblock.rs::inter_luma_residual_tu`
//! and `reconstruct_inter_chroma_block` (which currently hard-code
//! `DCT_DCT` for the inter sites that already migrated to the spec
//! `inverse_2d_spec` path in r23) requires the `inter_tx_type` CDF
//! reads from §5.11.45 + the `TileInterTxTypeSet{1,2,3}Cdf` defaults
//! and is deferred to a future round so this round can ship the
//! tables in isolation under regression tests.

use crate::transform::TxType;

/// Map a raw `tx_type` symbol (as decoded via `read_intra_tx_type`)
/// into the spec's [`TxType`] enum, given the extended-tx set index.
///
/// `tx_set = 0` always implies `DCT_DCT` (no signaling occurs for
/// TX > 32×32). Out-of-range `raw` values fall back to `DCT_DCT` —
/// callers should validate earlier, but we pick a defined result.
pub fn intra_tx_type_for(tx_set: u32, raw: u32) -> TxType {
    match tx_set {
        1 => match raw {
            0 => TxType::DctDct,
            1 => TxType::AdstDct,
            2 => TxType::DctAdst,
            3 => TxType::AdstAdst,
            4 => TxType::IdtIdt,
            5 => TxType::VDct,
            6 => TxType::HDct,
            _ => TxType::DctDct,
        },
        2 => match raw {
            0 => TxType::DctDct,
            1 => TxType::AdstDct,
            2 => TxType::DctAdst,
            3 => TxType::AdstAdst,
            4 => TxType::IdtIdt,
            _ => TxType::DctDct,
        },
        _ => TxType::DctDct,
    }
}

/// Pick the intra-frame extended-tx set per spec §6.10.15. Returns:
///
/// - `1` for area ≤ 16×16 (7-type set)
/// - `2` for area ≤ 32×32 (5-type set)
/// - `0` for area > 32×32 (implicit `DCT_DCT`)
pub fn ext_tx_set_for_intra(tx_w: u32, tx_h: u32) -> u32 {
    let area = tx_w * tx_h;
    if area <= 16 * 16 {
        1
    } else if area <= 32 * 32 {
        2
    } else {
        0
    }
}

/// 4-way size context used to index the intra `ext_tx` CDFs:
/// `TX_4X4=0`, `TX_8X8=1`, `TX_16X16=2`, `TX_32X32=3`. Non-square
/// sizes map to the square equivalent by area.
pub fn ext_tx_size_ctx(tx_w: u32, tx_h: u32) -> u32 {
    let area = tx_w * tx_h;
    if area <= 4 * 4 {
        0
    } else if area <= 8 * 8 {
        1
    } else if area <= 16 * 16 {
        2
    } else {
        3
    }
}

/// Inter-frame extended-tx set selector — spec §5.11.48
/// `get_tx_set(txSz)` for the `is_inter == 1` branch.
///
/// The spec computes:
///
/// ```text
/// txSzSqr   = Tx_Size_Sqr   [txSz]   // = min(log2W, log2H)
/// txSzSqrUp = Tx_Size_Sqr_Up[txSz]   // = max(log2W, log2H)
/// if (txSzSqrUp > TX_32X32)              -> TX_SET_DCTONLY (0)
/// if (reduced_tx_set || txSzSqrUp == 32) -> TX_SET_INTER_3 (3)
/// if (txSzSqr == TX_16X16)               -> TX_SET_INTER_2 (2)
/// otherwise                              -> TX_SET_INTER_1 (1)
/// ```
///
/// Returns `0`, `1`, `2`, or `3` as named above. This wraps the
/// per-shape min/max-side log2 derivation so callers don't need to
/// thread `Tx_Size_Sqr*` tables themselves; passing `tx_w` and
/// `tx_h` in samples is enough.
pub fn ext_tx_set_for_inter(tx_w: u32, tx_h: u32, reduced_tx_set: bool) -> u32 {
    let log2_w = log2_dim(tx_w);
    let log2_h = log2_dim(tx_h);
    let sqr = log2_w.min(log2_h); // 2..=6 i.e. TX_4X4..=TX_64X64
    let sqr_up = log2_w.max(log2_h);
    if sqr_up > 5 {
        // log2(32) = 5 → > TX_32X32 means at least one side ≥ 64.
        0
    } else if reduced_tx_set || sqr_up == 5 {
        3
    } else if sqr == 4 {
        // log2(16) = 4 → square min-side is 16×16.
        2
    } else {
        1
    }
}

/// Map a raw `inter_tx_type` symbol (as decoded under one of the
/// `TileInterTxTypeSet{1,2,3}Cdf` CDFs in §5.11.45) into the spec's
/// [`TxType`] enum, given the inter extended-tx `set` returned by
/// [`ext_tx_set_for_inter`].
///
/// Tables from §6.10.15:
///
/// ```text
/// Tx_Type_Inter_Inv_Set1[16] = {
///   IDTX, V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST,
///   DCT_DCT, ADST_DCT, DCT_ADST, FLIPADST_DCT, DCT_FLIPADST,
///   ADST_ADST, FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST
/// }
/// Tx_Type_Inter_Inv_Set2[12] = {
///   IDTX, V_DCT, H_DCT, DCT_DCT, ADST_DCT, DCT_ADST, FLIPADST_DCT,
///   DCT_FLIPADST, ADST_ADST, FLIPADST_FLIPADST, ADST_FLIPADST,
///   FLIPADST_ADST
/// }
/// Tx_Type_Inter_Inv_Set3[2] = { IDTX, DCT_DCT }
/// ```
///
/// `set = 0` (`TX_SET_DCTONLY`) carries no symbol — callers should
/// short-circuit, but we still return `DctDct` for safety. Out-of-range
/// `raw` values fall back to `DctDct`.
pub fn inter_tx_type_for(set: u32, raw: u32) -> TxType {
    match set {
        1 => match raw {
            0 => TxType::IdtIdt,
            1 => TxType::VDct,
            2 => TxType::HDct,
            3 => TxType::VAdst,
            4 => TxType::HAdst,
            5 => TxType::VFlipAdst,
            6 => TxType::HFlipAdst,
            7 => TxType::DctDct,
            8 => TxType::AdstDct,
            9 => TxType::DctAdst,
            10 => TxType::FlipAdstDct,
            11 => TxType::DctFlipAdst,
            12 => TxType::AdstAdst,
            13 => TxType::FlipAdstFlipAdst,
            14 => TxType::AdstFlipAdst,
            15 => TxType::FlipAdstAdst,
            _ => TxType::DctDct,
        },
        2 => match raw {
            0 => TxType::IdtIdt,
            1 => TxType::VDct,
            2 => TxType::HDct,
            3 => TxType::DctDct,
            4 => TxType::AdstDct,
            5 => TxType::DctAdst,
            6 => TxType::FlipAdstDct,
            7 => TxType::DctFlipAdst,
            8 => TxType::AdstAdst,
            9 => TxType::FlipAdstFlipAdst,
            10 => TxType::AdstFlipAdst,
            11 => TxType::FlipAdstAdst,
            _ => TxType::DctDct,
        },
        3 => match raw {
            0 => TxType::IdtIdt,
            1 => TxType::DctDct,
            _ => TxType::DctDct,
        },
        _ => TxType::DctDct,
    }
}

/// Number of legal raw symbols for an inter ext-tx `set` (i.e.
/// the CDF cardinality). 0 for `TX_SET_DCTONLY`.
pub fn inter_tx_type_set_size(set: u32) -> u32 {
    match set {
        1 => 16,
        2 => 12,
        3 => 2,
        _ => 0,
    }
}

/// Spec §5.11.40 / §6.10.15 `is_tx_type_in_set` — inter side. Returns
/// `true` when the candidate `tx_type` is a member of the inter
/// extended-tx `set` (one of 1/2/3, with 0 the DCT-only fallback that
/// always reports `false` outside `DctDct`).
///
/// Encoded directly from the spec's `Tx_Type_In_Set_Inter` membership
/// matrix (06.bitstream.syntax.md): set1 admits all 16 types, set2
/// admits 12 (excludes the V_/H_ ADST/FlipADST quartet), set3 admits
/// only `IDTX` and `DCT_DCT`.
pub fn is_inter_tx_type_in_set(set: u32, tx_type: TxType) -> bool {
    match set {
        1 => true,
        2 => !matches!(
            tx_type,
            TxType::VAdst | TxType::HAdst | TxType::VFlipAdst | TxType::HFlipAdst
        ),
        3 => matches!(tx_type, TxType::IdtIdt | TxType::DctDct),
        _ => matches!(tx_type, TxType::DctDct),
    }
}

/// Spec §5.11.40 chroma branch of `compute_tx_type` for inter blocks.
/// Maps the luma `tx_type` (read from `TxTypes[y4][x4]`) into the
/// chroma TX type for a chroma TU of dimensions `(c_tx_w, c_tx_h)`,
/// honouring the `reduced_tx_set` gate and falling back to `DctDct`
/// when the luma TX type isn't valid for the chroma extended-tx set.
///
/// Returns `DctDct` for TX sizes whose `txSzSqrUp > TX_32X32` (the
/// `TX_SET_DCTONLY` short-circuit), matching the spec's first
/// `if (Lossless || txSzSqrUp > TX_32X32) return DCT_DCT` line.
pub fn compute_inter_chroma_tx_type(
    luma_tx_type: TxType,
    c_tx_w: u32,
    c_tx_h: u32,
    reduced_tx_set: bool,
) -> TxType {
    let set = ext_tx_set_for_inter(c_tx_w, c_tx_h, reduced_tx_set);
    if set == 0 {
        // TX_SET_DCTONLY — every TX > 32×32 collapses to DCT_DCT.
        return TxType::DctDct;
    }
    if is_inter_tx_type_in_set(set, luma_tx_type) {
        luma_tx_type
    } else {
        TxType::DctDct
    }
}

/// Tiny `log2(n)` for AV1 TX side lengths (one of 4/8/16/32/64).
/// Returns 0 for unsupported inputs so the inter-set selector
/// degrades to `TX_SET_DCTONLY` rather than panicking.
fn log2_dim(n: u32) -> u32 {
    match n {
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ext_tx_set_boundaries() {
        assert_eq!(ext_tx_set_for_intra(4, 4), 1);
        assert_eq!(ext_tx_set_for_intra(16, 16), 1);
        assert_eq!(ext_tx_set_for_intra(32, 32), 2);
        assert_eq!(ext_tx_set_for_intra(64, 64), 0);
    }

    #[test]
    fn ext_tx_size_ctx_table() {
        assert_eq!(ext_tx_size_ctx(4, 4), 0);
        assert_eq!(ext_tx_size_ctx(8, 8), 1);
        assert_eq!(ext_tx_size_ctx(16, 16), 2);
        assert_eq!(ext_tx_size_ctx(32, 32), 3);
        assert_eq!(ext_tx_size_ctx(64, 64), 3);
    }

    #[test]
    fn intra_tx_type_set1_exhaustive() {
        let expected = [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
            TxType::IdtIdt,
            TxType::VDct,
            TxType::HDct,
        ];
        for (raw, want) in expected.iter().enumerate() {
            assert_eq!(intra_tx_type_for(1, raw as u32), *want);
        }
    }

    #[test]
    fn intra_tx_type_set0_is_dctdct() {
        assert_eq!(intra_tx_type_for(0, 0), TxType::DctDct);
        assert_eq!(intra_tx_type_for(0, 5), TxType::DctDct);
    }

    /// Round 24 — every entry of `Tx_Type_Inter_Inv_Set1` from spec
    /// §6.10.15 transcribed verbatim. Pinning all 16 keeps a future
    /// careless edit from silently scrambling the inter dispatch.
    #[test]
    fn inter_tx_type_set1_exhaustive() {
        let expected = [
            TxType::IdtIdt,
            TxType::VDct,
            TxType::HDct,
            TxType::VAdst,
            TxType::HAdst,
            TxType::VFlipAdst,
            TxType::HFlipAdst,
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::FlipAdstDct,
            TxType::DctFlipAdst,
            TxType::AdstAdst,
            TxType::FlipAdstFlipAdst,
            TxType::AdstFlipAdst,
            TxType::FlipAdstAdst,
        ];
        for (raw, want) in expected.iter().enumerate() {
            assert_eq!(inter_tx_type_for(1, raw as u32), *want);
        }
        assert_eq!(inter_tx_type_set_size(1), 16);
    }

    /// Round 24 — every entry of `Tx_Type_Inter_Inv_Set2` from spec
    /// §6.10.15. Note the ordering differs from set1: set2 omits the
    /// `V_ADST/H_ADST/V_FLIPADST/H_FLIPADST` quartet and threads
    /// `DCT_DCT` in at index 3 (where set1 has `V_ADST`).
    #[test]
    fn inter_tx_type_set2_exhaustive() {
        let expected = [
            TxType::IdtIdt,
            TxType::VDct,
            TxType::HDct,
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::FlipAdstDct,
            TxType::DctFlipAdst,
            TxType::AdstAdst,
            TxType::FlipAdstFlipAdst,
            TxType::AdstFlipAdst,
            TxType::FlipAdstAdst,
        ];
        for (raw, want) in expected.iter().enumerate() {
            assert_eq!(inter_tx_type_for(2, raw as u32), *want);
        }
        assert_eq!(inter_tx_type_set_size(2), 12);
    }

    /// Round 24 — `Tx_Type_Inter_Inv_Set3 = { IDTX, DCT_DCT }` from
    /// spec §6.10.15. This is the reduced set used either when
    /// `reduced_tx_set` is signalled or when the TX upsized square
    /// is exactly 32×32.
    #[test]
    fn inter_tx_type_set3_exhaustive() {
        assert_eq!(inter_tx_type_for(3, 0), TxType::IdtIdt);
        assert_eq!(inter_tx_type_for(3, 1), TxType::DctDct);
        // Out-of-range raw → DctDct.
        assert_eq!(inter_tx_type_for(3, 2), TxType::DctDct);
        assert_eq!(inter_tx_type_set_size(3), 2);
    }

    /// Round 25 / task #167 — `is_inter_tx_type_in_set` membership
    /// matrix matches `Tx_Type_In_Set_Inter` from spec §6.10.15
    /// (06.bitstream.syntax.md). Set1 admits all 16 types; set2
    /// excludes the V_/H_ ADST/FlipADST quartet (4 entries); set3
    /// admits only `IDTX` and `DCT_DCT`. Set0 (DCTONLY) reports false
    /// outside `DctDct`.
    #[test]
    fn is_inter_tx_type_in_set_matches_spec_matrix() {
        let all = [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
            TxType::FlipAdstDct,
            TxType::DctFlipAdst,
            TxType::FlipAdstFlipAdst,
            TxType::AdstFlipAdst,
            TxType::FlipAdstAdst,
            TxType::IdtIdt,
            TxType::VDct,
            TxType::HDct,
            TxType::VAdst,
            TxType::HAdst,
            TxType::VFlipAdst,
            TxType::HFlipAdst,
        ];
        // Set1: all 16 in.
        for t in all {
            assert!(is_inter_tx_type_in_set(1, t), "set1 must admit {t:?}");
        }
        // Set2: 12 in (excludes V_ADST, H_ADST, V_FLIPADST, H_FLIPADST).
        let set2_excluded = [
            TxType::VAdst,
            TxType::HAdst,
            TxType::VFlipAdst,
            TxType::HFlipAdst,
        ];
        for t in all {
            let want = !set2_excluded.contains(&t);
            assert_eq!(
                is_inter_tx_type_in_set(2, t),
                want,
                "set2 membership wrong for {t:?}",
            );
        }
        // Set3: only IDTX + DCT_DCT.
        for t in all {
            let want = matches!(t, TxType::IdtIdt | TxType::DctDct);
            assert_eq!(
                is_inter_tx_type_in_set(3, t),
                want,
                "set3 membership wrong for {t:?}",
            );
        }
        // Set0 (DCTONLY) — only DCT_DCT.
        for t in all {
            let want = matches!(t, TxType::DctDct);
            assert_eq!(
                is_inter_tx_type_in_set(0, t),
                want,
                "set0 membership wrong for {t:?}",
            );
        }
    }

    /// Round 25 / task #167 — `compute_inter_chroma_tx_type` honours
    /// the spec's chroma branch of §5.11.40 `compute_tx_type`:
    /// luma TX type passes through when in the chroma TX set, else
    /// `DctDct`. TX > 32×32 short-circuits to `DctDct` regardless.
    #[test]
    fn compute_inter_chroma_tx_type_passthrough_and_fallback() {
        // 16×16 chroma TU under reduced_tx_set=false → set2 (12 types).
        // Set2 admits AdstDct → passes through.
        assert_eq!(
            compute_inter_chroma_tx_type(TxType::AdstDct, 16, 16, false),
            TxType::AdstDct,
        );
        // Set2 excludes VAdst → falls back to DctDct.
        assert_eq!(
            compute_inter_chroma_tx_type(TxType::VAdst, 16, 16, false),
            TxType::DctDct,
        );
        // 8×8 chroma TU → set1 (16 types) → VAdst passes.
        assert_eq!(
            compute_inter_chroma_tx_type(TxType::VAdst, 8, 8, false),
            TxType::VAdst,
        );
        // 32×32 chroma TU → set3 (only IDTX/DCT_DCT) → AdstDct
        // collapses to DctDct.
        assert_eq!(
            compute_inter_chroma_tx_type(TxType::AdstDct, 32, 32, false),
            TxType::DctDct,
        );
        assert_eq!(
            compute_inter_chroma_tx_type(TxType::IdtIdt, 32, 32, false),
            TxType::IdtIdt,
        );
        // reduced_tx_set forces set3 even on 8×8.
        assert_eq!(
            compute_inter_chroma_tx_type(TxType::AdstDct, 8, 8, true),
            TxType::DctDct,
        );
        // 64×64 chroma TU → DCTONLY short-circuit.
        assert_eq!(
            compute_inter_chroma_tx_type(TxType::AdstDct, 64, 64, false),
            TxType::DctDct,
        );
    }

    /// Round 24 — `set = 0` (`TX_SET_DCTONLY`) carries no symbol;
    /// the helper still returns a defined value for any raw input.
    #[test]
    fn inter_tx_type_set0_is_dctdct() {
        assert_eq!(inter_tx_type_for(0, 0), TxType::DctDct);
        assert_eq!(inter_tx_type_for(0, 7), TxType::DctDct);
        assert_eq!(inter_tx_type_set_size(0), 0);
    }

    /// Round 24 — `get_tx_set` selector for inter, exhaustive across
    /// the 19 TX_SIZES_ALL shapes (square + 2:1 + 1:4) in both
    /// `reduced_tx_set` orientations. Encodes spec §5.11.48 verbatim.
    #[test]
    fn ext_tx_set_for_inter_exhaustive() {
        // (w, h, reduced_tx_set, expected_set)
        let cases = [
            // Squares — under reduced_tx_set everything ≤32 collapses to set3.
            (4, 4, false, 1),
            (8, 8, false, 1),
            (16, 16, false, 2),
            (32, 32, false, 3),
            (64, 64, false, 0),
            (4, 4, true, 3),
            (8, 8, true, 3),
            (16, 16, true, 3),
            (32, 32, true, 3),
            (64, 64, true, 0),
            // 2:1 rectangles — txSzSqr = log2(min), txSzSqrUp = log2(max).
            (4, 8, false, 1), // sqr=2 (TX_4X4) sqr_up=3 → set1
            (8, 4, false, 1),
            (8, 16, false, 1), // sqr=3 (TX_8X8) sqr_up=4 → set1
            (16, 8, false, 1),
            (16, 32, false, 3), // sqr_up=5 → set3
            (32, 16, false, 3),
            (32, 64, false, 0), // sqr_up=6 → DCTONLY
            (64, 32, false, 0),
            // 1:4 rectangles.
            (4, 16, false, 1), // sqr=2 sqr_up=4 → set1
            (16, 4, false, 1),
            (8, 32, false, 3), // sqr_up=5 → set3
            (32, 8, false, 3),
            (16, 64, false, 0), // sqr_up=6 → DCTONLY
            (64, 16, false, 0),
            // reduced_tx_set forces set3 unless DCTONLY applies.
            (4, 8, true, 3),
            (16, 32, true, 3),
            (16, 64, true, 0),
        ];
        for (w, h, rtxs, want) in cases.iter() {
            let got = ext_tx_set_for_inter(*w, *h, *rtxs);
            assert_eq!(
                got, *want,
                "ext_tx_set_for_inter({w}, {h}, reduced={rtxs}) = {got}, want {want}"
            );
        }
    }

    /// Round 24 — sanity sweep: for every legal inter ext-tx set, every
    /// raw value in `0..set_size` must map to a `TxType` the live
    /// `inverse_2d_spec` path supports for the relevant TX_SIZE bucket.
    /// We don't assert per-shape spec coverage here (that lives in
    /// `transform::tests::inverse_2d_spec_covers_every_spec_allowed_pair`)
    /// — just that no raw symbol falls into the `DctDct`-fallback
    /// branch at the top of its set, which would silently swallow a
    /// real signalled type.
    #[test]
    fn inter_tx_type_in_set_never_falls_through() {
        for set in 1..=3u32 {
            let n = inter_tx_type_set_size(set);
            let dctdct_fallthrough_count = (0..n)
                .filter(|raw| inter_tx_type_for(set, *raw) == TxType::DctDct)
                .count();
            // Each set legitimately has at most one DCT_DCT entry;
            // anything more would mean a fallthrough in the match.
            // Set1: DCT_DCT at raw=7. Set2: DCT_DCT at raw=3.
            // Set3: DCT_DCT at raw=1.
            assert!(
                dctdct_fallthrough_count <= 1,
                "set {set} has {dctdct_fallthrough_count} DctDct entries — \
                 expected at most one (the legitimate spec slot)",
            );
        }
    }
}
