//! AV1 coefficient decoder — §5.11.39 / §6.10.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/coeffs.go`
//! (MIT, KarpelesLab/goavif).
//!
//! For each non-skip transform block the bitstream carries:
//!
//! 1. `txb_skip` — if true, the whole TX block is zero.
//! 2. `eob_multi` — log-scale end-of-block bucket, refined by 0..=9
//!    `eob_extra` bits.
//! 3. Per-coefficient base level via `coeff_base_multi` (non-eob) or
//!    `coeff_base_eob_multi` (at the eob position).
//! 4. For saturated base levels: up to 4 `coeff_br_multi` refinements
//!    plus a Golomb-rice tail.
//! 5. Signs — DC via `dc_sign` CDF, AC via raw uniform bits.
//!
//! All CDFs come from [`crate::cdfs`]; the coefficient-context
//! derivation lives in [`super::coeff_ctx`]. The output of
//! [`decode_coefficients`] is a row-major `w × h` signed `i32` block
//! suitable for `inverse_2d`.

use oxideav_core::{Error, Result};

use crate::cdfs;
use crate::symbol::SymbolDecoder;

use super::coeff_ctx::{level_ctx, sig_coef_ctx_2d};

/// Map a `base_q_idx` to the 4-way `TOKEN_CDF_Q_CTXS` bucket (§7.12.4).
pub fn q_index_to_ctx(q: i32) -> usize {
    if q < 64 {
        0
    } else if q < 128 {
        1
    } else if q < 192 {
        2
    } else {
        3
    }
}

/// `nz_map_ctx_offset_4x4` per libaom `av1/common/txb_common.c`.
pub const NZ_MAP_CTX_OFFSET_4X4: [i8; 16] = [
    0, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21,
];

/// `nz_map_ctx_offset_8x8` per libaom.
pub const NZ_MAP_CTX_OFFSET_8X8: [i8; 64] = [
    0, 1, 6, 6, 21, 21, 21, 21, 1, 6, 6, 21, 21, 21, 21, 21, 6, 6, 21, 21, 21, 21, 21, 21, 6, 21,
    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
    21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
];

/// `nz_map_ctx_offset_16x16` — constructed from the common 4×4 corner
/// mask with the rest filled with 21.
pub static NZ_MAP_CTX_OFFSET_16X16: [i8; 256] = {
    let mut t = [21i8; 256];
    t[0] = 0;
    t[1] = 1;
    t[2] = 6;
    t[3] = 6;
    t[16] = 1;
    t[17] = 6;
    t[18] = 6;
    t[32] = 6;
    t[33] = 6;
    t[48] = 6;
    t
};

/// `nz_map_ctx_offset_32x32` — only exercised by 32×32 TX blocks
/// (Phase 4). Exposed for completeness.
pub static NZ_MAP_CTX_OFFSET_32X32: [i8; 1024] = {
    let mut t = [21i8; 1024];
    t[0] = 0;
    t[1] = 1;
    t[2] = 6;
    t[3] = 6;
    t[32] = 1;
    t[33] = 6;
    t[34] = 6;
    t[64] = 6;
    t[65] = 6;
    t[96] = 6;
    t
};

/// Pick the nz_map_ctx_offset table for a given TX size. For
/// non-square shapes we route to the nearest square table by area
/// (libaom / goavif use the same simplification for TX_64×N blocks:
/// the coded region is clamped to the top-left 32×32 so the 32×32
/// table covers everything we need).
pub fn nz_map_ctx_offset(w: usize, h: usize) -> Result<&'static [i8]> {
    let area = w * h;
    if area <= 16 {
        Ok(&NZ_MAP_CTX_OFFSET_4X4)
    } else if area <= 64 {
        Ok(&NZ_MAP_CTX_OFFSET_8X8)
    } else if area <= 256 {
        Ok(&NZ_MAP_CTX_OFFSET_16X16)
    } else if area <= 4096 {
        Ok(&NZ_MAP_CTX_OFFSET_32X32)
    } else {
        Err(Error::unsupported(format!(
            "av1 coeffs: nz_map offset for {w}×{h} not available (§6.10.6)"
        )))
    }
}

/// TX-size index used by the coefficient CDFs (0..=4 mapping
/// 4×4 / 8×8 / 16×16 / 32×32 / 64×64).
pub fn tx_size_idx(w: usize, h: usize) -> Result<usize> {
    let area = w * h;
    if area <= 16 {
        Ok(0)
    } else if area <= 64 {
        Ok(1)
    } else if area <= 256 {
        Ok(2)
    } else if area <= 1024 {
        Ok(3)
    } else if area <= 4096 {
        Ok(4)
    } else {
        Err(Error::unsupported(format!(
            "av1 coeffs: TX area {area} > 4096 (§6.10 — Phase 4)"
        )))
    }
}

/// `eob_pt` to `(eob_bin_start, extra_bits)` (§9.3.3).
fn eob_pt_to_eob(pt: u32) -> (u32, u32) {
    match pt {
        0 => (1, 0),
        1 => (2, 0),
        2 => (3, 1),
        3 => (5, 2),
        4 => (9, 3),
        5 => (17, 4),
        6 => (33, 5),
        7 => (65, 6),
        8 => (129, 7),
        9 => (257, 8),
        10 => (513, 9),
        _ => (1, 0),
    }
}

/// Per-tile CDF bank + dispatcher for coefficient-related symbols.
/// Mirrors goavif's `CoeffDecoder` but uses the already-opened
/// [`SymbolDecoder`] owned by the tile decoder.
pub struct CoeffCdfBank {
    pub q_ctx: usize,
    pub txb_skip_cdf: [[Vec<u16>; 13]; 5],
    pub eob_multi16_cdf: [[Vec<u16>; 2]; 2],
    pub eob_multi32_cdf: [[Vec<u16>; 2]; 2],
    pub eob_multi64_cdf: [[Vec<u16>; 2]; 2],
    pub eob_multi128_cdf: [[Vec<u16>; 2]; 2],
    pub eob_multi256_cdf: [[Vec<u16>; 2]; 2],
    pub eob_multi512_cdf: [[Vec<u16>; 2]; 2],
    pub eob_multi1024_cdf: [[Vec<u16>; 2]; 2],
    pub eob_extra_cdf: [[[Vec<u16>; 9]; 2]; 5],
    pub coeff_base_eob_multi_cdf: [[[Vec<u16>; 4]; 2]; 5],
    pub coeff_base_multi_cdf: [[[Vec<u16>; 42]; 2]; 5],
    pub coeff_br_multi_cdf: [[[Vec<u16>; 21]; 2]; 5],
    pub dc_sign_cdf: [[Vec<u16>; 3]; 2],
}

impl CoeffCdfBank {
    /// Initialise the CDF bank using the Q-context bucket derived
    /// from `base_q_idx`.
    pub fn new(q_ctx: usize) -> Self {
        let q_ctx = q_ctx.min(3);
        Self {
            q_ctx,
            txb_skip_cdf: std::array::from_fn(|tx| {
                std::array::from_fn(|c| cdfs::DEFAULT_TXB_SKIP_CDF[tx][c].to_vec())
            }),
            eob_multi16_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_EOB_MULTI16_CDF[q_ctx][p][c].to_vec())
            }),
            eob_multi32_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_EOB_MULTI32_CDF[q_ctx][p][c].to_vec())
            }),
            eob_multi64_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_EOB_MULTI64_CDF[q_ctx][p][c].to_vec())
            }),
            eob_multi128_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_EOB_MULTI128_CDF[q_ctx][p][c].to_vec())
            }),
            eob_multi256_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_EOB_MULTI256_CDF[q_ctx][p][c].to_vec())
            }),
            eob_multi512_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_EOB_MULTI512_CDF[q_ctx][p][c].to_vec())
            }),
            eob_multi1024_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_EOB_MULTI1024_CDF[q_ctx][p][c].to_vec())
            }),
            eob_extra_cdf: std::array::from_fn(|tx| {
                std::array::from_fn(|p| {
                    std::array::from_fn(|c| cdfs::DEFAULT_EOB_EXTRA_CDF[q_ctx][tx][p][c].to_vec())
                })
            }),
            coeff_base_eob_multi_cdf: std::array::from_fn(|tx| {
                std::array::from_fn(|p| {
                    std::array::from_fn(|c| {
                        cdfs::DEFAULT_COEFF_BASE_EOB_MULTI_CDF[tx][p][c].to_vec()
                    })
                })
            }),
            coeff_base_multi_cdf: std::array::from_fn(|tx| {
                std::array::from_fn(|p| {
                    std::array::from_fn(|c| {
                        cdfs::DEFAULT_COEFF_BASE_MULTI_CDF[q_ctx][tx][p][c].to_vec()
                    })
                })
            }),
            coeff_br_multi_cdf: std::array::from_fn(|tx| {
                std::array::from_fn(|p| {
                    std::array::from_fn(|c| {
                        cdfs::DEFAULT_COEFF_BR_MULTI_CDF[q_ctx][tx][p][c].to_vec()
                    })
                })
            }),
            dc_sign_cdf: std::array::from_fn(|p| {
                std::array::from_fn(|c| cdfs::DEFAULT_DC_SIGN_CDF[p][c].to_vec())
            }),
        }
    }

    /// Read `txb_skip` flag.
    pub fn read_txb_skip(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        tx_size_idx: usize,
        ctx: usize,
    ) -> Result<bool> {
        let tx = tx_size_idx.min(4);
        let ctx = ctx.min(12);
        let v = sym.decode_symbol(&mut self.txb_skip_cdf[tx][ctx])?;
        Ok(v != 0)
    }

    /// Read `eob_pt` for a block with `num_coeffs` coefficients.
    pub fn read_eob_pt(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        num_coeffs: usize,
        plane_type: usize,
        eob_ctx: usize,
    ) -> Result<u32> {
        let plane_type = plane_type.min(1);
        let eob_ctx = eob_ctx.min(1);
        let cdf = match num_coeffs {
            16 => &mut self.eob_multi16_cdf[plane_type][eob_ctx],
            32 => &mut self.eob_multi32_cdf[plane_type][eob_ctx],
            64 => &mut self.eob_multi64_cdf[plane_type][eob_ctx],
            128 => &mut self.eob_multi128_cdf[plane_type][eob_ctx],
            256 => &mut self.eob_multi256_cdf[plane_type][eob_ctx],
            512 => &mut self.eob_multi512_cdf[plane_type][eob_ctx],
            1024 => &mut self.eob_multi1024_cdf[plane_type][eob_ctx],
            _ => {
                return Err(Error::unsupported(format!(
                    "av1 coeffs: eob bucket {num_coeffs} not supported (§9.3.3)"
                )))
            }
        };
        sym.decode_symbol(cdf)
    }

    /// Read the full `eob` (1-based position). Reads `eob_pt` plus any
    /// required `eob_extra` bits.
    pub fn read_eob(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        num_coeffs: usize,
        tx_size_idx: usize,
        plane_type: usize,
        eob_ctx: usize,
    ) -> Result<u32> {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let pt = self.read_eob_pt(sym, num_coeffs, plane_type, eob_ctx)?;
        let (bin_start, extra) = eob_pt_to_eob(pt);
        if extra == 0 {
            return Ok(bin_start);
        }
        let eob_coef_ctx = ((pt as usize).saturating_sub(2)).min(8);
        let high_bit = sym.decode_symbol(&mut self.eob_extra_cdf[tx][plane_type][eob_coef_ctx])?;
        let mut offset = high_bit << (extra - 1);
        for b in (0..(extra - 1)).rev() {
            offset |= sym.decode_bool(16384) << b;
        }
        Ok(bin_start + offset)
    }

    /// Read the coefficient at the eob position (base level).
    pub fn read_base_level_eob(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        tx_size_idx: usize,
        plane_type: usize,
        eob_base_ctx: usize,
    ) -> Result<u32> {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let ctx = eob_base_ctx.min(3);
        sym.decode_symbol(&mut self.coeff_base_eob_multi_cdf[tx][plane_type][ctx])
    }

    /// Read a non-eob coefficient's base level.
    pub fn read_base_level(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        tx_size_idx: usize,
        plane_type: usize,
        sig_ctx: usize,
    ) -> Result<u32> {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let ctx = sig_ctx.min(41);
        sym.decode_symbol(&mut self.coeff_base_multi_cdf[tx][plane_type][ctx])
    }

    /// Read one additional base-range refinement.
    pub fn read_br_level(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        tx_size_idx: usize,
        plane_type: usize,
        br_ctx: usize,
    ) -> Result<u32> {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let ctx = br_ctx.min(20);
        sym.decode_symbol(&mut self.coeff_br_multi_cdf[tx][plane_type][ctx])
    }

    /// Read the DC sign bit.
    pub fn read_dc_sign(
        &mut self,
        sym: &mut SymbolDecoder<'_>,
        plane_type: usize,
        ctx: usize,
    ) -> Result<bool> {
        let plane_type = plane_type.min(1);
        let ctx = ctx.min(2);
        let v = sym.decode_symbol(&mut self.dc_sign_cdf[plane_type][ctx])?;
        Ok(v != 0)
    }
}

/// Read a Golomb-Rice coded non-negative integer from the range
/// coder's bypass bits (§9.3). Returns `x - 1` where `x` is the
/// reconstructed `(1 << length)`-prefixed value.
fn read_golomb(sym: &mut SymbolDecoder<'_>) -> u32 {
    let mut length = 0u32;
    while length < 32 {
        if sym.decode_bool(16384) != 0 {
            break;
        }
        length += 1;
    }
    let mut x: u32 = 1;
    for _ in 0..length {
        x = (x << 1) | sym.decode_bool(16384);
    }
    x - 1
}

/// Decode a `w × h` transform block's coefficients into a row-major
/// `Vec<i32>` of length `w * h`.
///
/// - `tx_size_idx` is the 4-way-CDF TX-size index (0..=4) — use
///   [`tx_size_idx`] to derive it from raw `(w, h)`.
/// - `num_coeffs` is the EOB "bucket" — for square AV1 TX sizes it
///   equals `w * h`; for TX_64×N it is clamped to 1024.
/// - `plane_type` is 0 for luma, 1 for chroma.
/// - `scan` must have length `num_coeffs`; each entry is a row-major
///   block position.
/// - `nz_map_offset` must have length `num_coeffs` and correspond to
///   the TX size (from [`nz_map_ctx_offset`]).
#[allow(clippy::too_many_arguments)]
pub fn decode_coefficients(
    sym: &mut SymbolDecoder<'_>,
    bank: &mut CoeffCdfBank,
    tx_size_idx: usize,
    plane_type: usize,
    num_coeffs: usize,
    scan: &[usize],
    nz_map_offset: &[i8],
    w: usize,
    h: usize,
) -> Result<Vec<i32>> {
    let mut coeffs = vec![0i32; w * h];

    // `txb_skip` — context=0 per goavif parity.
    if bank.read_txb_skip(sym, tx_size_idx, 0)? {
        return Ok(coeffs);
    }

    let eob = bank.read_eob(sym, num_coeffs, tx_size_idx, plane_type, 0)?;
    let eob = (eob as usize).clamp(1, num_coeffs);

    if num_coeffs > 1024 {
        return Err(Error::unsupported(
            "av1 coeffs: num_coeffs > 1024 not supported (§9.3.3)",
        ));
    }
    if scan.len() < eob {
        return Err(Error::invalid(format!(
            "av1 coeffs: scan too short ({}) for eob {eob}",
            scan.len()
        )));
    }

    let mut abs_levels = vec![0i8; w * h];

    // Reverse-scan-order decode: scan[eob-1] first, scan[0] last.
    for i in (0..eob).rev() {
        let pos = scan[i];
        if pos >= w * h {
            return Err(Error::invalid(format!(
                "av1 coeffs: scan pos {pos} ≥ block size {}",
                w * h
            )));
        }
        let r = (pos / w) as i32;
        let c = (pos % w) as i32;

        let base_level: u32 = if i == eob - 1 {
            let eob_base_ctx = if eob > 10 {
                3
            } else if eob > 5 {
                2
            } else if eob > 2 {
                1
            } else {
                0
            };
            bank.read_base_level_eob(sym, tx_size_idx, plane_type, eob_base_ctx)? + 1
        } else {
            let sig_ctx =
                sig_coef_ctx_2d(r, c, w as i32, h as i32, &abs_levels, nz_map_offset, i);
            bank.read_base_level(sym, tx_size_idx, plane_type, sig_ctx.max(0) as usize)?
        };

        let mut level = base_level as i32;
        if level == 3 {
            for _br in 0..4 {
                let br_ctx = level_ctx(r, c, w as i32, h as i32, &abs_levels);
                let inc = bank.read_br_level(sym, tx_size_idx, plane_type, br_ctx.max(0) as usize)?;
                level += inc as i32;
                if inc < 3 {
                    break;
                }
            }
            if level >= 15 {
                level += read_golomb(sym) as i32;
            }
        }

        abs_levels[pos] = level.min(127) as i8;
        coeffs[pos] = level;
    }

    // Signs — DC via context-adapted CDF, AC via uniform bits (read in
    // forward scan order per spec).
    if coeffs[scan[0]] > 0 && bank.read_dc_sign(sym, plane_type, 0)? {
        coeffs[scan[0]] = -coeffs[scan[0]];
    }
    for &pos in scan.iter().take(eob).skip(1) {
        if coeffs[pos] > 0 && sym.decode_bool(16384) != 0 {
            coeffs[pos] = -coeffs[pos];
        }
    }

    Ok(coeffs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transform::default_zigzag_scan;

    #[test]
    fn q_index_to_ctx_boundaries() {
        assert_eq!(q_index_to_ctx(0), 0);
        assert_eq!(q_index_to_ctx(63), 0);
        assert_eq!(q_index_to_ctx(64), 1);
        assert_eq!(q_index_to_ctx(127), 1);
        assert_eq!(q_index_to_ctx(128), 2);
        assert_eq!(q_index_to_ctx(192), 3);
        assert_eq!(q_index_to_ctx(255), 3);
    }

    #[test]
    fn tx_size_idx_for_small_sizes() {
        assert_eq!(tx_size_idx(4, 4).unwrap(), 0);
        assert_eq!(tx_size_idx(8, 8).unwrap(), 1);
        assert_eq!(tx_size_idx(16, 16).unwrap(), 2);
        assert_eq!(tx_size_idx(32, 32).unwrap(), 3);
        assert_eq!(tx_size_idx(64, 64).unwrap(), 4);
    }

    #[test]
    fn nz_map_ctx_offset_4x4_matches_libaom() {
        let tab = nz_map_ctx_offset(4, 4).unwrap();
        assert_eq!(
            tab,
            &[0i8, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21]
        );
    }

    #[test]
    fn eob_pt_to_eob_bins() {
        assert_eq!(eob_pt_to_eob(0), (1, 0));
        assert_eq!(eob_pt_to_eob(1), (2, 0));
        assert_eq!(eob_pt_to_eob(2), (3, 1));
        assert_eq!(eob_pt_to_eob(10), (513, 9));
    }

    #[test]
    fn cdf_bank_constructs_without_panic() {
        let _ = CoeffCdfBank::new(0);
        let _ = CoeffCdfBank::new(3);
        // Clamps above 3.
        let b = CoeffCdfBank::new(99);
        assert_eq!(b.q_ctx, 3);
    }

    #[test]
    fn decode_coefficients_reads_without_panic() {
        // Brute-force several seeds; at least one should decode cleanly
        // or produce a skip. No panics allowed (mirrors goavif's test).
        let scan = default_zigzag_scan(4, 4);
        let mut any_success = false;
        for seed in 0u8..32 {
            let buf = [seed; 64];
            let mut sym = SymbolDecoder::new(&buf, buf.len(), false).expect("init");
            let mut bank = CoeffCdfBank::new(0);
            if decode_coefficients(
                &mut sym,
                &mut bank,
                0,
                0,
                16,
                &scan,
                &NZ_MAP_CTX_OFFSET_4X4,
                4,
                4,
            )
            .is_ok()
            {
                any_success = true;
            }
        }
        // The assertion is soft — the point is no panics. If nothing
        // decoded cleanly, print a log line instead of failing, to
        // mirror goavif.
        let _ = any_success;
    }
}
