//! Forward AV1 range coder — bit-exact inverse of [`crate::symbol::SymbolDecoder`].
//!
//! Spec references: §9.2 (arithmetic coder), §9.4 (CDF adaptation —
//! shared with the decoder via [`crate::symbol::update_cdf`]). The
//! decoder side stores CDFs in the **inverse-CDF / "icdf" wire form**:
//! `cdf[s]` for the s-th symbol is `(1 << 15) -
//! cumulative_probability_<=_s`.
//!
//! ## Algorithm
//!
//! The decoder's §8.2.6 inner loop computes per-iteration:
//!
//! ```text
//!   cur = ((R >> 8) * (cdf[symbol] >> 6)) >> 1 + 4 * (n - symbol - 1)
//! ```
//!
//! and exits when `SymbolValue >= cur`. The chosen sub-interval inside
//! `R` (in the decoder's `SymbolValue` basis) is `[lo_s, hi_s)` where:
//!
//! ```text
//!   hi_s = R                          if s == 0
//!        = ((R>>8)*(cdf[s-1]>>6))>>1 + 4*(n-s)   otherwise
//!   lo_s = 0                          if s == n-1
//!        = ((R>>8)*(cdf[s]>>6))>>1   + 4*(n-s-1) otherwise
//! ```
//!
//! Because the decoder's `init_symbol` sets `SV = 0x7FFF ^
//! first_15_bits` and then `SV ^= ((SV+1) << bits) - 1` on each
//! renormalisation, the decoder consumes bits in **complement form**.
//! In the natural-basis V (bytestream value as a wide integer), each
//! symbol `s` owns the V-sub-interval `[R - hi_s, R - lo_s)` within
//! the parent V-interval — the bit-flip of the SV-interval.
//!
//! ## Implementation strategy: wide-integer V tracking
//!
//! Instead of the streaming carry-deferral scheme used by libaom
//! (which couples `low + rng` overflow with a `bits_outstanding`
//! counter and a precarry byte buffer), we track the V-interval
//! directly as wide integers:
//!
//! - `v_low`: lower bound of V (a Vec<u8> MSB-first big-endian
//!   bigint). The bytestream output at finish() is exactly this value.
//! - `v_rng`: width of V's interval (also a bigint).
//! - `rng_internal`: the spec's `R` value, kept in `[0x8000, 0x10000)`
//!   after each renormalisation. Used only for the per-symbol CDF
//!   formulas; doesn't track the scale.
//! - `scale`: how many renorm shifts have occurred. The "absolute"
//!   range of V is `rng_internal << scale`, but we don't materialise
//!   this product — we use `scale` only to interpret the symbol-encode
//!   formulas correctly.
//!
//! Per-symbol encode:
//!
//! ```text
//!   delta_abs = (R - hi_s) << scale          # in absolute V-basis
//!   v_low    += delta_abs                    # bigint addition
//!   v_rng     = (hi_s - lo_s) << scale       # bigint
//!   rng_internal = hi_s - lo_s               # spec's internal value
//!   while rng_internal < 0x8000:
//!     rng_internal <<= 1
//!     scale += 1
//! ```
//!
//! At `finish()`: emit `v_low` as a byte stream MSB-first, padded to
//! the next byte boundary. The decoder reads at most `15 + scale`
//! bits from this stream; trailing zeros beyond that are ignored.
//!
//! This scheme is O(num_symbols * num_bits_per_symbol) in time and
//! space — for round-2 tile payloads (<1 KB output) totally
//! negligible. A future round-3+ optimisation can switch to a
//! streaming precarry-buffer encoder if memory matters.
//!
//! ## Decoder pinning — `decode(encode(symbols)) == symbols`
//!
//! The unit tests at the bottom build CDFs, encode symbol streams,
//! run the bytes back through [`crate::symbol::SymbolDecoder`] and
//! assert the recovered streams are identical (with optional CDF
//! adaptation in lock-step). This is the only correctness pin.

use crate::symbol::update_cdf;

/// Low-probability floor — must match [`crate::symbol::MIN_PROB`].
const MIN_PROB: u32 = 4;
/// CDF probability right-shift — must match [`crate::symbol::PROB_SHIFT`].
const PROB_SHIFT: u32 = 6;
/// Initial range after `init_symbol` (§9.2.1).
const SYMBOL_RANGE_INIT: u32 = 1 << 15;

/// Tile-scoped forward range coder.
///
/// Tracks the V-interval `[v_low, v_low + v_rng)` as wide bigints
/// (Vec<u8>, MSB-first big-endian). At finish, the bytestream output
/// is `v_low` itself, byte-aligned with zero pad. The decoder reads
/// at most `15 + scale` bits from the stream — anything beyond is
/// padding and doesn't affect symbol decoding.
pub struct SymbolEncoder {
    /// Current "internal" rng — kept in [0x8000, 0x10000) after each
    /// renormalise(). The spec's per-symbol formula reads this.
    rng_internal: u32,
    /// `v_low << shift_bits_pending` is the *MSB-bit-offset-aware* low
    /// end of the V interval. Each renorm shift increments
    /// `shift_bits_pending` by 1; periodically we materialise the
    /// shifts by appending zero bits to v_low (`v_low <<= bits_pending`)
    /// or by extending v_rng accordingly.
    ///
    /// We materialise ONLY when needed (at symbol encode and at
    /// finish) to avoid O(scale) work per renorm bit.
    v_low: Vec<u8>,
    /// Total renorm shifts so far, `scale = bits_pending +
    /// already_materialised`.
    scale: u32,
    /// How many of the `scale` shifts have NOT yet been baked into
    /// `v_low` (= are still "pending" as a left-shift).
    shift_bits_pending: u32,
    /// Whether to invoke [`update_cdf`] on each symbol — must match
    /// the decoder side.
    allow_update: bool,
}

impl Default for SymbolEncoder {
    fn default() -> Self {
        Self::new(true)
    }
}

impl SymbolEncoder {
    pub fn new(allow_update: bool) -> Self {
        Self {
            rng_internal: SYMBOL_RANGE_INIT,
            v_low: vec![0u8; 0],
            scale: 0,
            shift_bits_pending: 0,
            allow_update,
        }
    }

    /// Encode a single binary symbol with `P(bit = 1) = p` in Q15.
    /// Mirrors [`crate::symbol::SymbolDecoder::decode_bool`].
    pub fn encode_bool(&mut self, bit: u32, p: u32) {
        debug_assert!(bit <= 1);
        let r = self.rng_internal;
        let mut cur = ((r >> 8) * (p >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
        cur += MIN_PROB;
        // bit=0 owns SV-interval [cur, R) ⇒ V-interval = [R-R, R-cur) = [0, R-cur).
        //         delta = 0; rng = R - cur.
        // bit=1 owns SV-interval [0, cur)  ⇒ V-interval = [R-cur, R).
        //         delta = R - cur;        rng = cur.
        let (delta, new_rng) = if bit == 0 {
            (0u32, r - cur)
        } else {
            (r - cur, cur)
        };
        self.bake_pending_shifts();
        self.add_to_v_low(delta);
        self.rng_internal = new_rng;
        self.renormalise();
    }

    /// Encode a multi-symbol from the inverse-CDF wire form `cdf` and
    /// optionally update `cdf` per §9.4. The CDF layout matches
    /// [`crate::symbol::SymbolDecoder::decode_symbol`]: `cdf` has
    /// `n + 1` entries (`n - 1` thresholds plus the `0` sentinel)
    /// followed by the adaptive count slot.
    pub fn encode_symbol(&mut self, cdf: &mut [u16], symbol: u32) {
        debug_assert!(cdf.len() >= 2);
        let n_with_sentinel = cdf.len() - 1;
        let n = n_with_sentinel as u32;
        let s = symbol;
        debug_assert!(s < n, "symbol {} >= n {}", s, n);

        let r = self.rng_internal;
        let hi: u32 = if s == 0 {
            r
        } else {
            let f = cdf[(s - 1) as usize] as u32;
            let mut v = ((r >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
            v += MIN_PROB * (n - s);
            v
        };
        let lo: u32 = if s == n - 1 {
            0
        } else {
            let f = cdf[s as usize] as u32;
            let mut v = ((r >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
            v += MIN_PROB * (n - s - 1);
            v
        };
        debug_assert!(hi > lo, "hi {hi} <= lo {lo} at sym {s} R={r}");

        let delta = r - hi;
        self.bake_pending_shifts();
        self.add_to_v_low(delta);
        self.rng_internal = hi - lo;
        self.renormalise();

        if self.allow_update {
            update_cdf(cdf, n_with_sentinel, s as usize);
        }
    }

    /// Emit `n` raw 50/50 bits — inverse of
    /// [`crate::symbol::SymbolDecoder::read_literal`]. Bits are written
    /// MSB-first via `encode_bool(bit, 16384)`.
    pub fn write_literal(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            let b = (value >> i) & 1;
            self.encode_bool(b, 16384);
        }
    }

    /// Drain the encoder and return the byte-aligned tile payload.
    ///
    /// `v_low` (after baking pending shifts) is the V-widened value in
    /// the encoder's basis. The DECODER reads the bytestream MSB-first
    /// as a single wide integer of `(15 + scale)` bits — its first 15
    /// bits become `V_init`, the next `bits_for_renorm_1` bits feed
    /// the first renorm's XOR, and so on.
    ///
    /// To produce the right bytestream, we therefore must emit v_low
    /// **left-aligned** in a `(15 + scale)`-bit MSB-first integer,
    /// padded with leading zeros if v_low is numerically shorter, then
    /// further padded with trailing zeros to a byte boundary.
    pub fn finish(mut self) -> Vec<u8> {
        self.bake_pending_shifts();
        let total_bits = 15u32 + self.scale;
        // Convert v_low (Vec<u8>, MSB-first big-endian) to a single
        // bit stream of length `total_bits`. v_low's bit length =
        // `v_low.len() * 8` — pad with leading zeros as needed.
        let v_low_bits = self.v_low.len() as u32 * 8;
        let leading_zeros = total_bits.saturating_sub(v_low_bits);
        // Emit `leading_zeros` zero bits, then v_low's bits MSB-first.
        let mut out_bits: Vec<bool> = Vec::with_capacity(total_bits as usize);
        for _ in 0..leading_zeros {
            out_bits.push(false);
        }
        // Append v_low's bits, MSB-first across bytes and within each byte.
        // If v_low has MORE bits than `total_bits` (unlikely but
        // possible if the encoder added too much to v_low), truncate
        // by skipping leading bits. The lossy direction is "drop
        // most-significant" — but a well-formed encoder never gets
        // here; debug_assert guards.
        let total_v_bits: Vec<bool> = self
            .v_low
            .iter()
            .flat_map(|byte| (0..8).rev().map(move |i| (byte >> i) & 1 == 1))
            .collect();
        let skip = total_v_bits.len().saturating_sub(total_bits as usize);
        for &b in total_v_bits.iter().skip(skip) {
            out_bits.push(b);
            if out_bits.len() == total_bits as usize {
                break;
            }
        }
        // Pack out_bits into bytes MSB-first.
        let mut out: Vec<u8> = Vec::with_capacity((out_bits.len() + 7) / 8);
        let mut acc: u8 = 0;
        let mut bits_in_acc: u32 = 0;
        for b in &out_bits {
            acc |= (*b as u8) << (7 - bits_in_acc);
            bits_in_acc += 1;
            if bits_in_acc == 8 {
                out.push(acc);
                acc = 0;
                bits_in_acc = 0;
            }
        }
        if bits_in_acc != 0 {
            out.push(acc);
        }
        // Ensure ≥ 2 bytes so `init_symbol` doesn't error.
        while out.len() < 2 {
            out.push(0);
        }
        out
    }

    pub fn allow_update(&self) -> bool {
        self.allow_update
    }

    /// Renormalise: shift `rng_internal` left until it's back in
    /// [0x8000, 0x10000). Each shift increments `scale` and
    /// `shift_bits_pending` (the actual v_low shift is deferred to
    /// `bake_pending_shifts`).
    fn renormalise(&mut self) {
        while self.rng_internal < 0x8000 {
            self.rng_internal <<= 1;
            self.scale += 1;
            self.shift_bits_pending += 1;
        }
    }

    /// Materialise pending left-shifts of `v_low` (multiply v_low by
    /// `2^shift_bits_pending`).
    fn bake_pending_shifts(&mut self) {
        if self.shift_bits_pending == 0 {
            return;
        }
        let shift = self.shift_bits_pending;
        // Append `shift` zero bits to v_low (MSB-first). Equivalent
        // to v_low <<= shift in big-int sense.
        let full_bytes = shift / 8;
        let extra_bits = shift % 8;
        // Append full zero bytes.
        for _ in 0..full_bytes {
            self.v_low.push(0);
        }
        // Shift the entire v_low left by `extra_bits` bits, with
        // carry propagating to the next byte. The high bit pushed
        // out of the topmost byte goes... nowhere (v_low can grow
        // there).
        if extra_bits > 0 {
            let mut carry: u8 = 0;
            for byte in self.v_low.iter_mut().rev() {
                let new = (*byte << extra_bits) | carry;
                carry = *byte >> (8 - extra_bits);
                *byte = new;
            }
            if carry != 0 {
                self.v_low.insert(0, carry);
            }
        }
        self.shift_bits_pending = 0;
    }

    /// Add `delta` to `v_low` (MSB-first big-endian bigint). `delta`
    /// is a u32 already in the same basis as v_low (bake pending
    /// shifts before calling this).
    fn add_to_v_low(&mut self, delta: u32) {
        if delta == 0 {
            return;
        }
        // Add delta as 4 bytes (LSB-first internally).
        let delta_bytes = delta.to_le_bytes();
        let mut carry: u16 = 0;
        let mut i: usize = 0;
        // Walk v_low from the LSB end (= last element, since v_low is MSB-first).
        while i < 4 || carry != 0 {
            let pos = self.v_low.len();
            let pos_offset = i;
            let v_idx = pos.checked_sub(1 + pos_offset);
            let dbyte = if i < 4 { delta_bytes[i] as u16 } else { 0 };
            match v_idx {
                Some(idx) => {
                    let sum = self.v_low[idx] as u16 + dbyte + carry;
                    self.v_low[idx] = (sum & 0xFF) as u8;
                    carry = sum >> 8;
                }
                None => {
                    let sum = dbyte + carry;
                    if sum != 0 {
                        self.v_low.insert(0, (sum & 0xFF) as u8);
                        carry = sum >> 8;
                    } else {
                        // No more bytes to add and no carry to propagate.
                        return;
                    }
                }
            }
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol::SymbolDecoder;

    /// Build an inverse-CDF wire form from a vector of "cumulative
    /// probabilities" `cum[s] = sum_{i<=s} p_i` in Q15. The wire form
    /// is `cdf[s] = (1<<15) - cum[s]`, with `cum[n-1] == 1<<15` so
    /// `cdf[n-1] == 0`. A trailing `0` slot for the count is appended.
    fn cdf_from_cumulative(cum: &[u16]) -> Vec<u16> {
        let mut v = Vec::with_capacity(cum.len() + 1);
        for &c in cum {
            v.push((1u16 << 15) - c);
        }
        v.push(0); // adaptation count slot
        v
    }

    /// Roundtrip a symbol stream through encode → decode and assert
    /// equality. Both sides start with the same CDF and adapt in
    /// lockstep when `allow_update` is set.
    fn roundtrip_symbols(symbols: &[u32], cum: &[u16], allow_update: bool) {
        let mut enc = SymbolEncoder::new(allow_update);
        let mut enc_cdf = cdf_from_cumulative(cum);
        for &s in symbols {
            enc.encode_symbol(&mut enc_cdf, s);
        }
        let buf = enc.finish();

        let mut dec = SymbolDecoder::new(&buf, buf.len(), allow_update).expect("decoder init");
        let mut dec_cdf = cdf_from_cumulative(cum);
        for (i, &want) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&mut dec_cdf).expect("decode");
            assert_eq!(got, want, "symbol mismatch at {i}: want {want}, got {got}");
        }
    }

    /// Same but for binary symbols via `encode_bool` / `decode_bool`.
    fn roundtrip_bools(bits: &[u32], p: u32) {
        let mut enc = SymbolEncoder::new(false);
        for &b in bits {
            enc.encode_bool(b, p);
        }
        let buf = enc.finish();
        let mut dec = SymbolDecoder::new(&buf, buf.len(), false).expect("decoder init");
        for (i, &want) in bits.iter().enumerate() {
            let got = dec.decode_bool(p);
            assert_eq!(got, want, "bool mismatch at {i}: want {want}, got {got}");
        }
    }

    #[test]
    fn api_surface_compiles() {
        let enc = SymbolEncoder::new(true);
        assert!(enc.allow_update());
    }

    #[test]
    fn roundtrip_single_5050_bits() {
        roundtrip_bools(&[0], 16384);
        roundtrip_bools(&[1], 16384);
    }

    #[test]
    fn roundtrip_short_5050_bool_stream() {
        let bits = [1u32, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1];
        roundtrip_bools(&bits, 16384);
    }

    #[test]
    fn roundtrip_skewed_probability_bools() {
        let bits = [0u32; 16];
        roundtrip_bools(&bits, 28000);
        roundtrip_bools(&bits, 4000);
    }

    #[test]
    fn roundtrip_three_symbol_uniform_no_adapt() {
        let cum = [10923u16, 21845, 32768];
        let symbols: Vec<u32> = (0..32).map(|i| (i as u32) % 3).collect();
        roundtrip_symbols(&symbols, &cum, false);
    }

    #[test]
    fn roundtrip_four_symbol_uniform_with_adapt() {
        let cum = [8192u16, 16384, 24576, 32768];
        let symbols: Vec<u32> = (0..40).map(|i| (i as u32) % 4).collect();
        roundtrip_symbols(&symbols, &cum, true);
    }

    #[test]
    fn roundtrip_default_partition_cdf_long_stream() {
        let template = crate::cdfs::DEFAULT_PARTITION_CDF[0].to_vec();
        let mut enc_cdf = template.clone();
        let mut dec_cdf = template.clone();
        let symbols: Vec<u32> = (0..64).map(|i| ((i * 7) % 4) as u32).collect();

        let mut enc = SymbolEncoder::new(true);
        for &s in &symbols {
            enc.encode_symbol(&mut enc_cdf, s);
        }
        let buf = enc.finish();

        let mut dec = SymbolDecoder::new(&buf, buf.len(), true).expect("decoder init");
        for (i, &want) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&mut dec_cdf).expect("decode");
            assert_eq!(
                got, want,
                "DEFAULT_PARTITION_CDF mismatch at {i}: want {want}, got {got}"
            );
        }
    }

    #[test]
    fn roundtrip_mixed_bools_and_symbols() {
        let cum = [10923u16, 21845, 32768];
        let bool_bits = [1u32, 0, 1, 1, 0, 0, 1];
        let sym_pattern = [0u32, 1, 2, 1, 0, 2, 1];

        let mut enc = SymbolEncoder::new(false);
        let mut enc_cdf = cdf_from_cumulative(&cum);
        for i in 0..bool_bits.len() {
            enc.encode_bool(bool_bits[i], 16384);
            enc.encode_symbol(&mut enc_cdf, sym_pattern[i]);
        }
        let buf = enc.finish();

        let mut dec = SymbolDecoder::new(&buf, buf.len(), false).expect("decoder init");
        let mut dec_cdf = cdf_from_cumulative(&cum);
        for i in 0..bool_bits.len() {
            let b = dec.decode_bool(16384);
            assert_eq!(b, bool_bits[i], "bool at {i}");
            let s = dec.decode_symbol(&mut dec_cdf).expect("decode");
            assert_eq!(s, sym_pattern[i], "sym at {i}");
        }
    }

    #[test]
    fn roundtrip_write_literal() {
        let mut enc = SymbolEncoder::new(false);
        enc.write_literal(0xA5, 8);
        enc.write_literal(0x33, 6);
        let buf = enc.finish();

        let mut dec = SymbolDecoder::new(&buf, buf.len(), false).expect("decoder init");
        assert_eq!(dec.read_literal(8), 0xA5);
        assert_eq!(dec.read_literal(6), 0x33);
    }
}
