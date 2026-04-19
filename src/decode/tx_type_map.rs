//! AV1 intra transform-type mapping — §6.10.15.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/tx_type_map.go`
//! (MIT, KarpelesLab/goavif).
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
}
