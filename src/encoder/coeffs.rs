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

    /// Emit `eob_pt` — the log-scale bucket for the end-of-block
    /// position. Mirrors `CoeffCdfBank::read_eob_pt`.
    pub fn write_eob_pt(
        &mut self,
        sym: &mut SymbolEncoder,
        num_coeffs: usize,
        plane_type: usize,
        eob_ctx: usize,
        pt: u32,
    ) {
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
            _ => return,
        };
        sym.encode_symbol(cdf, pt);
    }

    /// Emit the full `eob` (1-based). Writes `eob_pt` then the required
    /// `eob_extra` bits. Mirrors `CoeffCdfBank::read_eob`.
    pub fn write_eob(
        &mut self,
        sym: &mut SymbolEncoder,
        num_coeffs: usize,
        tx_size_idx: usize,
        plane_type: usize,
        eob_ctx: usize,
        eob: u32,
    ) {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let (pt, extra_bits, offset) = eob_to_pt(eob);
        self.write_eob_pt(sym, num_coeffs, plane_type, eob_ctx, pt);
        if extra_bits == 0 {
            return;
        }
        let eob_coef_ctx = ((pt as usize).saturating_sub(2)).min(8);
        let high_bit = (offset >> (extra_bits - 1)) & 1;
        sym.encode_symbol(
            &mut self.eob_extra_cdf[tx][plane_type][eob_coef_ctx],
            high_bit,
        );
        for b in (0..(extra_bits - 1)).rev() {
            sym.encode_bool((offset >> b) & 1, 16384);
        }
    }

    /// Emit the coefficient at the eob position (base level - 1).
    pub fn write_base_level_eob(
        &mut self,
        sym: &mut SymbolEncoder,
        tx_size_idx: usize,
        plane_type: usize,
        eob_base_ctx: usize,
        level_minus1: u32,
    ) {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let ctx = eob_base_ctx.min(3);
        sym.encode_symbol(
            &mut self.coeff_base_eob_multi_cdf[tx][plane_type][ctx],
            level_minus1,
        );
    }

    /// Emit a non-eob coefficient's base level.
    pub fn write_base_level(
        &mut self,
        sym: &mut SymbolEncoder,
        tx_size_idx: usize,
        plane_type: usize,
        sig_ctx: usize,
        level: u32,
    ) {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let ctx = sig_ctx.min(41);
        sym.encode_symbol(&mut self.coeff_base_multi_cdf[tx][plane_type][ctx], level);
    }

    /// Emit one additional base-range refinement level.
    pub fn write_br_level(
        &mut self,
        sym: &mut SymbolEncoder,
        tx_size_idx: usize,
        plane_type: usize,
        br_ctx: usize,
        inc: u32,
    ) {
        let tx = tx_size_idx.min(4);
        let plane_type = plane_type.min(1);
        let ctx = br_ctx.min(20);
        sym.encode_symbol(&mut self.coeff_br_multi_cdf[tx][plane_type][ctx], inc);
    }

    /// Emit the DC sign bit.
    pub fn write_dc_sign(
        &mut self,
        sym: &mut SymbolEncoder,
        plane_type: usize,
        ctx: usize,
        sign: bool,
    ) {
        let plane_type = plane_type.min(1);
        let ctx = ctx.min(2);
        sym.encode_symbol(&mut self.dc_sign_cdf[plane_type][ctx], sign as u32);
    }
}

/// Convert an eob (1-based position) to the `(pt, extra_bits, offset)`
/// triple used by `write_eob`. Mirrors `eob_pt_to_eob` from the
/// decoder but in the forward direction.
fn eob_to_pt(eob: u32) -> (u32, u32, u32) {
    // pt table: pt → (bin_start, extra_bits)
    // 0 → (1, 0)
    // 1 → (2, 0)
    // 2 → (3, 1)
    // 3 → (5, 2)
    // 4 → (9, 3)
    // 5 → (17, 4)
    // 6 → (33, 5)
    // 7 → (65, 6)
    // 8 → (129, 7)
    // 9 → (257, 8)
    // 10 → (513, 9)
    let pt_table: [(u32, u32); 11] = [
        (1, 0),
        (2, 0),
        (3, 1),
        (5, 2),
        (9, 3),
        (17, 4),
        (33, 5),
        (65, 6),
        (129, 7),
        (257, 8),
        (513, 9),
    ];
    for (pt, &(bin_start, extra)) in pt_table.iter().enumerate().rev() {
        if eob >= bin_start {
            let offset = eob - bin_start;
            return (pt as u32, extra, offset);
        }
    }
    (0, 0, 0)
}

/// Write Golomb-Rice coded `value` as raw 50/50 bypass bits.
/// Mirrors `read_golomb` from the decoder.
fn write_golomb(sym: &mut SymbolEncoder, value: u32) {
    let x = value + 1;
    // Count length as the highest set bit position.
    let length = if x == 0 { 0 } else { 31 - x.leading_zeros() };
    // Write `length` zero bits (unary prefix).
    for _ in 0..length {
        sym.encode_bool(0, 16384);
    }
    // Write terminating 1 bit.
    sym.encode_bool(1, 16384);
    // Write `length` data bits, MSB first.
    for i in (0..length).rev() {
        sym.encode_bool((x >> i) & 1, 16384);
    }
}

/// Encode a `w × h` transform block's quantised coefficients into the
/// range-coder stream. Mirrors
/// [`crate::decode::coeffs::decode_coefficients`] exactly — every
/// symbol emitted here corresponds to a `decode_symbol` /
/// `decode_bool` call on the decoder side in the same order.
///
/// `coeffs` is a row-major `w × h` signed quantised coefficient block
/// (output of the forward transform + quantisation step). Zero blocks
/// are handled by emitting only `txb_skip = true`.
///
/// Returns `true` if any non-zero coefficient was emitted (i.e.
/// `txb_skip == false`).
/// Spec-correct `txb_skip_ctx` for an isolated TX block (no above /
/// left neighbours, full TU footprint matching block dims). Mirrors
/// [`crate::decode::coeff_ctx::txb_skip_ctx_spec`] with `top = left =
/// 0` and `bw == w && bh == h`:
///
/// * `plane_type == 0` (luma) ⇒ `0`  — `bw == w && bh == h` short-
///   circuit returns 0 in the spec.
/// * `plane_type != 0` (chroma) ⇒ `7` — chroma branch is `(top != 0) +
///   (left != 0) + 7`, both indicators 0 ⇒ 7.
///
/// Round-r-next bug: `encode_coefficients` previously hard-coded
/// `ctx = 0` for **both** luma AND chroma. The decoder side reads
/// chroma `txb_skip` from `txb_skip_cdf[tx][7]` (very different
/// distribution from `[tx][0]`), so the wire bits were interpreted
/// against the wrong probability and dav1d would refuse the frame.
/// Self-roundtrip happened to clear because the test only asserted
/// the Y plane mean; the chroma plane was reconstructed to the
/// predictor (128) by accident through saturating quant arithmetic.
#[inline]
pub fn isolated_txb_skip_ctx(plane_type: usize) -> usize {
    if plane_type == 0 {
        0
    } else {
        7
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_coefficients(
    sym: &mut SymbolEncoder,
    bank: &mut CoeffCdfBankEnc,
    tx_size_idx: usize,
    plane_type: usize,
    num_coeffs: usize,
    scan: &[usize],
    nz_map_offset: &[i8],
    w: usize,
    h: usize,
    coeffs: &[i32],
) -> bool {
    use crate::decode::coeff_ctx::{level_ctx, sig_coef_ctx_2d};

    // Spec §5.11.39 / §9.4 `all_zero` — `txb_skip` reads from the
    // `txb_skip_cdf[tx_size_idx][txb_ctx]` CDF where `txb_ctx` depends
    // on the plane and the immediate above / left neighbours. The
    // round-40 encoder only ever emits a single isolated 64×64 SB with
    // no neighbours, so we can hard-derive the spec ctx via
    // [`isolated_txb_skip_ctx`]; non-isolated encodes will need a
    // future neighbour-aware path mirroring the decoder.
    let txb_ctx = isolated_txb_skip_ctx(plane_type);

    // txb_skip: emit 1 (skip) if all zero.
    let eob_pos = scan.iter().take(num_coeffs).rposition(|&p| coeffs[p] != 0);
    if eob_pos.is_none() {
        bank.write_txb_skip(sym, tx_size_idx, txb_ctx, true);
        return false;
    }
    bank.write_txb_skip(sym, tx_size_idx, txb_ctx, false);

    let eob = eob_pos.unwrap() + 1; // 1-based

    // Write eob.
    bank.write_eob(sym, num_coeffs, tx_size_idx, plane_type, 0, eob as u32);

    // Build partial abs_levels for context — filled forward (scan[eob-1]
    // first, like the decoder's reverse pass).
    let mut abs_levels = vec![0i8; w * h];

    for i in (0..eob).rev() {
        let pos = scan[i];
        let coeff_abs = coeffs[pos].unsigned_abs() as i32;
        let r = (pos / w) as i32;
        let c = (pos % w) as i32;

        let base_level_raw: u32 = if i == eob - 1 {
            // eob coefficient: level - 1 (clamped to 0..2 for the 3-way CDF)
            let eob_base_ctx = if eob > 10 {
                3
            } else if eob > 5 {
                2
            } else if eob > 2 {
                1
            } else {
                0
            };
            let emit = (coeff_abs - 1).clamp(0, 2) as u32;
            bank.write_base_level_eob(sym, tx_size_idx, plane_type, eob_base_ctx, emit);
            emit + 1
        } else {
            let sig_ctx = sig_coef_ctx_2d(r, c, w as i32, h as i32, &abs_levels, nz_map_offset, i);
            // Clamp to 3 (not 2): the 4-way base-level CDF encodes 0/1/2/3+,
            // where 3 signals "use br_level for the remainder".
            let emit = coeff_abs.min(3) as u32;
            bank.write_base_level(sym, tx_size_idx, plane_type, sig_ctx.max(0) as usize, emit);
            emit
        };

        // If base level == 3 (saturated), emit br levels.
        let mut level = base_level_raw as i32;
        if level == 3 {
            let mut remaining = coeff_abs - 3;
            for _br in 0..4 {
                let br_ctx = level_ctx(r, c, w as i32, h as i32, &abs_levels);
                let inc = remaining.min(2) as u32; // br CDF is 3-way: 0/1/2
                bank.write_br_level(sym, tx_size_idx, plane_type, br_ctx.max(0) as usize, inc);
                level += inc as i32;
                remaining -= inc as i32;
                if inc < 3 {
                    break;
                }
            }
            // Golomb tail for very large values.
            if level >= 15 {
                let tail = (coeff_abs - level) as u32;
                write_golomb(sym, tail);
            }
        }

        abs_levels[pos] = level.min(127) as i8;
    }

    // Signs — DC first via dc_sign CDF, then AC via uniform bits.
    if coeffs[scan[0]] != 0 {
        let sign = coeffs[scan[0]] < 0;
        bank.write_dc_sign(sym, plane_type, 0, sign);
    }
    for &pos in scan.iter().take(eob).skip(1) {
        if coeffs[pos] != 0 {
            let sign = if coeffs[pos] < 0 { 1 } else { 0 };
            sym.encode_bool(sign, 16384);
        }
    }

    true
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

    /// `encode_coefficients` + `decode_coefficients` roundtrip for a 4×4
    /// block. Exercises the all-zero path (txb_skip=true) and a
    /// single-non-zero path (only DC non-zero), then verifies that the
    /// decoded coefficients match the originals.
    #[test]
    fn encode_coefficients_roundtrip_4x4() {
        use crate::decode::coeffs::decode_coefficients;
        use crate::transform::scan::default_zigzag_scan;

        let w = 4;
        let h = 4;
        let num_coeffs = w * h;
        let scan = default_zigzag_scan(w, h);
        let nz_offset = nz_map_ctx_offset(w, h).expect("nz_map_ctx_offset");
        let tx_size_idx = tx_size_idx(w, h).expect("tx_size_idx");

        // --- all-zero block ---
        let zero_coeffs = vec![0i32; num_coeffs];
        let mut enc_bank = CoeffCdfBankEnc::new(1);
        let mut enc_sym = SymbolEncoder::new(true);
        let had_nz = encode_coefficients(
            &mut enc_sym,
            &mut enc_bank,
            tx_size_idx,
            0, // luma
            num_coeffs,
            &scan,
            nz_offset,
            w,
            h,
            &zero_coeffs,
        );
        assert!(!had_nz, "all-zero block should report no non-zeros");
        let buf = enc_sym.finish();

        let mut dec_bank = CoeffCdfBank::new(1);
        let mut dec_sym = SymbolDecoder::new(&buf, buf.len(), true).expect("init");
        let decoded = decode_coefficients(
            &mut dec_sym,
            &mut dec_bank,
            tx_size_idx,
            0,
            num_coeffs,
            &scan,
            nz_offset,
            w,
            h,
        )
        .expect("decode all-zero");
        assert!(
            decoded.iter().all(|&v| v == 0),
            "all-zero must decode to all-zero"
        );

        // --- single DC non-zero ---
        let mut coeffs = vec![0i32; num_coeffs];
        coeffs[0] = 5; // only DC non-zero
        let mut enc_bank = CoeffCdfBankEnc::new(1);
        let mut enc_sym = SymbolEncoder::new(true);
        let had_nz = encode_coefficients(
            &mut enc_sym,
            &mut enc_bank,
            tx_size_idx,
            0,
            num_coeffs,
            &scan,
            nz_offset,
            w,
            h,
            &coeffs,
        );
        assert!(had_nz, "DC-only block should report non-zeros");
        let buf = enc_sym.finish();

        let mut dec_bank = CoeffCdfBank::new(1);
        let mut dec_sym = SymbolDecoder::new(&buf, buf.len(), true).expect("init");
        let decoded = decode_coefficients(
            &mut dec_sym,
            &mut dec_bank,
            tx_size_idx,
            0,
            num_coeffs,
            &scan,
            nz_offset,
            w,
            h,
        )
        .expect("decode DC-only");
        assert_eq!(decoded[0], 5, "DC coefficient must round-trip");
        for (i, &v) in decoded.iter().enumerate().skip(1) {
            assert_eq!(v, 0, "AC[{i}] should be zero in DC-only block");
        }
    }

    /// `encode_coefficients` correctly emits eob for a multi-coefficient
    /// block and the decoder recovers all non-zero values.
    #[test]
    fn encode_coefficients_roundtrip_multi_coeff() {
        use crate::decode::coeffs::decode_coefficients;
        use crate::transform::scan::default_zigzag_scan;

        let w = 4;
        let h = 4;
        let num_coeffs = w * h;
        let scan = default_zigzag_scan(w, h);
        let nz_offset = nz_map_ctx_offset(w, h).expect("nz_map_ctx_offset");
        let tx_size_idx_val = tx_size_idx(w, h).expect("tx_size_idx");

        // Coefficients at scan positions 0, 2, 5 (DC + two AC terms).
        let mut coeffs = vec![0i32; num_coeffs];
        coeffs[scan[0]] = 3;
        coeffs[scan[2]] = -1;
        coeffs[scan[5]] = 2;

        let mut enc_bank = CoeffCdfBankEnc::new(1);
        let mut enc_sym = SymbolEncoder::new(true);
        let had_nz = encode_coefficients(
            &mut enc_sym,
            &mut enc_bank,
            tx_size_idx_val,
            0,
            num_coeffs,
            &scan,
            nz_offset,
            w,
            h,
            &coeffs,
        );
        assert!(had_nz);
        let buf = enc_sym.finish();

        let mut dec_bank = CoeffCdfBank::new(1);
        let mut dec_sym = SymbolDecoder::new(&buf, buf.len(), true).expect("init");
        let decoded = decode_coefficients(
            &mut dec_sym,
            &mut dec_bank,
            tx_size_idx_val,
            0,
            num_coeffs,
            &scan,
            nz_offset,
            w,
            h,
        )
        .expect("decode multi-coeff");

        assert_eq!(decoded[scan[0]], 3, "DC must round-trip");
        assert_eq!(decoded[scan[2]], -1, "AC at scan[2] must round-trip");
        assert_eq!(decoded[scan[5]], 2, "AC at scan[5] must round-trip");
        // Everything beyond the eob must be zero.
        for i in 6..num_coeffs {
            assert_eq!(
                decoded[scan[i]], 0,
                "beyond-eob coeff at scan[{i}] must be zero"
            );
        }
    }
}
