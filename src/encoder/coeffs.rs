//! AV1 coefficient encoder — round 3 scaffold (item 5 partial).
//!
//! Mirrors [`crate::decode::coeffs::CoeffCdfBank`] / `decode_coefficients`
//! on the forward side. Round 3 ships only the API surface plus the
//! txb_skip emit; the full coefficient-stream encode (eob_pt, eob_extra,
//! coeff_base_*, br_level, signs, golomb tail) is a round 4+ item.
//!
//! ## Round 3 status
//!
//! The current encoder always emits `skip = 1` at the leaf-block level
//! (see [`crate::encoder::tile::write_tile_group_skip_intra_64`]); the
//! decoder short-circuits before any coefficient read. So in practice
//! this module's symbol emitters never fire on round-3 streams. They
//! exist as the inverse-shape skeleton that round 4+ can hang the full
//! coefficient encoder on without re-deriving the CDF wiring.
//!
//! ## Round 4+ checklist
//!
//! 1. `encode_eob_pt` + `encode_eob_extra` (mirror `read_eob`).
//! 2. `encode_base_level_eob` / `encode_base_level` / `encode_br_level`
//!    + `encode_golomb` for the tail.
//! 3. `encode_dc_sign` + raw `encode_bool(., 16384)` for AC signs.
//! 4. Tying it all together in a forward `encode_coefficients(coeffs,
//!    scan, …)` that mirrors [`crate::decode::coeffs::decode_coefficients`].
//! 5. Wire into a non-skip leaf-block emit path that:
//!    - computes `residual = src - pred`,
//!    - runs `fdct4x4` (or higher TX size, eventually),
//!    - quantises via [`crate::encoder::quant::PlaneQuant`],
//!    - emits coefficients via this module.

use crate::cdfs;
use crate::encoder::symbol::SymbolEncoder;

pub use crate::decode::coeffs::{nz_map_ctx_offset, q_index_to_ctx, tx_size_idx};

/// Per-tile encoder-side CDF bank for coefficient-related symbols.
/// Mirrors [`crate::decode::coeffs::CoeffCdfBank`] field-for-field so
/// the per-symbol context indexing stays in lock-step with the decoder.
pub struct CoeffCdfBankEnc {
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

impl CoeffCdfBankEnc {
    /// Initialise the encoder-side CDF bank — same defaults as the
    /// decoder's [`crate::decode::coeffs::CoeffCdfBank::new`].
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

    /// Emit `txb_skip` flag. `true` ⇒ the entire TX block is zero —
    /// no further coefficient symbols are emitted for this block.
    pub fn write_txb_skip(
        &mut self,
        sym: &mut SymbolEncoder,
        tx_size_idx: usize,
        ctx: usize,
        skip: bool,
    ) {
        let tx = tx_size_idx.min(4);
        let ctx = ctx.min(12);
        sym.encode_symbol(&mut self.txb_skip_cdf[tx][ctx], skip as u32);
    }

    // Round 4+ stubs — mirror the decoder's `read_*` API. Documented
    // here so the round-4 author can fill them in without re-deriving
    // the CDF wiring.
    //
    //   - write_eob_pt(sym, num_coeffs, plane_type, eob_ctx, pt)
    //   - write_eob(sym, num_coeffs, tx_size_idx, plane_type, eob_ctx, eob)
    //     -> internally splits into pt + extra-bit emit per §9.3.3
    //   - write_base_level_eob(sym, tx_size_idx, plane_type, eob_base_ctx, level)
    //   - write_base_level(sym, tx_size_idx, plane_type, sig_ctx, level)
    //   - write_br_level(sym, tx_size_idx, plane_type, br_ctx, inc)
    //   - write_dc_sign(sym, plane_type, ctx, sign)
    //   - write_golomb(sym, value)  // reuse encode_bool(., 16384)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::coeffs::CoeffCdfBank;
    use crate::symbol::SymbolDecoder;

    /// Encoder-side bank initialises identically to the decoder side.
    /// Pin the txb_skip CDF for q_ctx = 1, tx = 0 to confirm the
    /// init-from-defaults wiring matches.
    #[test]
    fn encoder_bank_init_matches_decoder() {
        let enc = CoeffCdfBankEnc::new(1);
        let dec = CoeffCdfBank::new(1);
        // Spot-check: txb_skip[0][0] should be byte-identical.
        assert_eq!(enc.txb_skip_cdf[0][0], dec.txb_skip_cdf[0][0]);
        assert_eq!(enc.dc_sign_cdf[0][0], dec.dc_sign_cdf[0][0]);
    }

    /// txb_skip emit / decode roundtrip — `false` then `true` then
    /// `false` etc.
    #[test]
    fn txb_skip_roundtrip() {
        let bits = [false, true, false, false, true, true, false];
        let mut enc_bank = CoeffCdfBankEnc::new(2);
        let mut sym = SymbolEncoder::new(true);
        for &b in &bits {
            enc_bank.write_txb_skip(&mut sym, 0, 0, b);
        }
        let buf = sym.finish();

        let mut dec_bank = CoeffCdfBank::new(2);
        let mut dec = SymbolDecoder::new(&buf, buf.len(), true).expect("init");
        for (i, &want) in bits.iter().enumerate() {
            let got = dec_bank
                .read_txb_skip(&mut dec, 0, 0)
                .expect("decode txb_skip");
            assert_eq!(got, want, "txb_skip mismatch at {i}");
        }
    }
}
