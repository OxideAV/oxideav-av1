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
//! ## Implementation strategy: streaming 16-bit live register
//!
//! The decoder reads V MSB-first as a `(15 + scale)`-bit integer:
//! the first 15 bits seed `SV`, then each renormalisation shift
//! consumes one more bit. We mirror this by building V bit-by-bit
//! into the output as encoding proceeds, using a tiny in-flight
//! register:
//!
//! - `low: u32` — only bits `0..16` carry meaning. Bits `0..15` are
//!   the **live region** where the next `delta` (a 16-bit value
//!   bounded by `R_internal < 2^16`) will land via `low += delta`.
//!   Bit `16` is the **carry slot** — set if the add overflows.
//! - `rng: u32` — spec's `R_internal`, kept in `[0x8000, 0x10000)`
//!   after each renormalisation.
//! - `out: Vec<u8>` — V's already-flushed bytes, MSB-first packed.
//!   Carries from `low` overflowing propagate back into `out`'s
//!   most-recently pushed byte (then earlier bytes if that byte
//!   rolled over from `0xFF` → `0x00`).
//! - `bit_buf: u8` + `bit_buf_n: u8` — partial byte being assembled,
//!   top-aligned (filled bit 7 first, bit 6 second, etc.).
//! - `first_renorm_done: bool` — the *first* renorm bit always lives
//!   at "V-position −1" (above V's MSB) and is dropped. Tracked here
//!   so we drop it exactly once.
//!
//! ### Per-symbol encode
//!
//! ```text
//!   delta = R_internal - hi      # in [0, R_internal), 16-bit max
//!   new_rng = hi - lo            # the chosen sub-interval width
//!   low += delta                 # narrows v_low; carry out of bit 16
//!     handled by `propagate_carry()`
//!   low &= 0xFFFF                # drop carry slot; live region = bits 0..15
//!   rng = new_rng
//!   while rng < 0x8000:          # renormalise
//!     rng <<= 1
//!     low <<= 1                  # promote bit 15 → bit 16
//!     emit_bit((low >> 16) & 1)  # the bit that just exited live region
//!     low &= 0xFFFF
//! ```
//!
//! ### V-position derivation
//!
//! The K-th renorm shift emits the bit that was at v_low's position
//! 15 just before the shift (= position 16 just after the shift in
//! the un-masked register). Mapped to V's MSB-first indexing on the
//! *final* `(15 + scale_FINAL)`-bit integer, that bit ends up at
//! V-position `K − 2`:
//!
//! - K = 1: V[−1] — above V's MSB. **Dropped.**
//! - K ≥ 2: V[K − 2]. Emitted to `bit_buf` / `out`.
//!
//! At `finish()` we emit the remaining 16 bits = V[scale − 1 ..
//! 15 + scale] from `low`'s bottom 16 bits MSB-first. Total emitted =
//! `(scale − 1) + 16 = scale + 15` ✓.
//!
//! ### Carry propagation
//!
//! When `low + delta` overflows bit 16 (so `low` reaches `[2^16,
//! 2^17)`), the carry out is `+1` at v_low's bit 16. That bit's
//! current V-position is V[K − 2] where K is the current renorm
//! count — i.e. the *most recently emitted* V bit (or the dropped
//! K=1 bit if no real emit has happened yet). Add 1 there; if it was
//! a 1, it becomes 0 and the carry propagates to the next-older bit,
//! walking back through `bit_buf` and then `out`.
//!
//! The K=1 dropped bit is provably 0 (the first encode runs with
//! `R_internal = 2^15` so `delta < 2^15` and the post-shift bit-16
//! value is `delta >> 15 = 0`). A carry chain that walks off the
//! front of `out` is therefore absorbed at the dropped position
//! (`+1` to 0 = 1, no further propagation). For a malformed encode
//! that would push v_low past `2^(15+scale)` we'd see propagation
//! beyond that — debug-asserted.
//!
//! ### Why this is streaming
//!
//! Internal state size is O(1): `low` is 4 bytes, `rng` is 4,
//! `bit_buf` is 1, etc. The only growing allocation is `out`, which
//! is the V bytestream we're producing (necessarily O(output)).
//! Per-symbol cost is O(1) amortised — one u32 add, ≤ 16 renorm
//! shifts (each O(1)), and the carry walk-back which amortises to
//! O(1) per symbol because carry chains can't refill faster than
//! they drain.
//!
//! The previous wide-bigint design (round 2) held a SECOND O(output)
//! buffer (`v_low: Vec<u8>`) and `bake_pending_shifts` /
//! `add_to_v_low` were O(len) per symbol — total O(N²). For
//! tile-payload sizes ≤ 1 KB this was fine; for full-frame coefficient
//! streams (10⁵–10⁷ symbols) the new design is dramatically smaller.
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
/// Streams V's bits MSB-first into `out` as encoding proceeds. The
/// internal state size is O(1) (a 16-bit live register, the current
/// range, and a partial-byte buffer); the only growing allocation is
/// `out`, which is the V bytestream itself.
pub struct SymbolEncoder {
    /// Spec's `R_internal` — kept in [0x8000, 0x10000) after each
    /// renormalise(). Reads the per-symbol CDF formula.
    rng: u32,
    /// 16-bit live register. Bits 0..15 hold the v_low value being
    /// built; bit 16 is a transient carry slot that's masked off
    /// after `propagate_carry()` resolves overflows.
    low: u32,
    /// V bytestream emitted so far, MSB-first packed. Carry from a
    /// `low` overflow propagates back into this buffer.
    out: Vec<u8>,
    /// Partial byte being built — top-aligned (bit 7 filled first).
    bit_buf: u8,
    /// Number of bits in `bit_buf` (0..=7).
    bit_buf_n: u8,
    /// Set after the first renorm shift fires. The first renorm bit
    /// is at V-position −1 (above V's MSB) and is dropped; this flag
    /// guarantees we drop it exactly once.
    first_renorm_done: bool,
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
            rng: SYMBOL_RANGE_INIT,
            low: 0,
            out: Vec::new(),
            bit_buf: 0,
            bit_buf_n: 0,
            first_renorm_done: false,
            allow_update,
        }
    }

    /// Encode a single binary symbol with `P(bit = 1) = p` in Q15.
    /// Mirrors [`crate::symbol::SymbolDecoder::decode_bool`].
    pub fn encode_bool(&mut self, bit: u32, p: u32) {
        debug_assert!(bit <= 1);
        let r = self.rng;
        let mut cur = ((r >> 8) * (p >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
        cur += MIN_PROB;
        // bit=0 owns SV-interval [cur, R) ⇒ V-interval [0, R-cur):
        //         delta = 0; rng = R - cur.
        // bit=1 owns SV-interval [0, cur)  ⇒ V-interval [R-cur, R):
        //         delta = R - cur;        rng = cur.
        let (delta, new_rng) = if bit == 0 {
            (0u32, r - cur)
        } else {
            (r - cur, cur)
        };
        self.add_low(delta);
        self.rng = new_rng;
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

        let r = self.rng;
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
        self.add_low(delta);
        self.rng = hi - lo;
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
    /// We've emitted `scale − 1` real V bits via the per-renorm
    /// `emit_bit` calls (the K=1 bit is dropped). The remaining 16
    /// bits = V[scale − 1 .. 15 + scale] are the bottom 16 bits of
    /// `low`, emitted MSB-first.
    pub fn finish(mut self) -> Vec<u8> {
        // Emit the 16 live bits of `low` MSB-first (= V's LSB area).
        for i in (0..16).rev() {
            let b = ((self.low >> i) & 1) as u8;
            self.emit_bit(b);
        }
        // Flush partial byte, padding the remainder with zeros.
        if self.bit_buf_n > 0 {
            self.out.push(self.bit_buf);
            self.bit_buf = 0;
            self.bit_buf_n = 0;
        }
        // Ensure ≥ 2 bytes so `SymbolDecoder::new` doesn't error on
        // its 15-bit init read.
        while self.out.len() < 2 {
            self.out.push(0);
        }
        self.out
    }

    pub fn allow_update(&self) -> bool {
        self.allow_update
    }

    /// Add `delta` to `low`. On overflow into bit 16 (so `low` reaches
    /// `[2^16, 2^17)`), the carry propagates back into the most-
    /// recently emitted V bit; `low` is then masked back to 16 bits.
    #[inline]
    fn add_low(&mut self, delta: u32) {
        self.low = self.low.wrapping_add(delta);
        if self.low >= (1 << 16) {
            self.propagate_carry();
            self.low &= 0xFFFF;
        }
    }

    /// Renormalise: shift `rng` left until it's back in [0x8000,
    /// 0x10000). Each shift emits one V bit (the bit just promoted
    /// out of `low`'s live region into bit 16).
    #[inline]
    fn renormalise(&mut self) {
        while self.rng < 0x8000 {
            self.rng <<= 1;
            self.low <<= 1;
            let b = ((self.low >> 16) & 1) as u8;
            self.low &= 0xFFFF;
            if self.first_renorm_done {
                self.emit_bit(b);
            } else {
                // K=1 emit: V[−1], above V's MSB. Discard. Provably
                // a 0 bit (initial encode runs with R = 2^15 so delta
                // < 2^15, hence post-shift bit-16 = 0).
                debug_assert_eq!(
                    b, 0,
                    "av1 SymbolEncoder: first renorm bit must be 0 \
                     (initial R = 2^15 ⇒ delta < 2^15)"
                );
                self.first_renorm_done = true;
            }
        }
    }

    /// Append a single bit to the output stream (MSB-first across
    /// bytes and within each byte).
    #[inline]
    fn emit_bit(&mut self, b: u8) {
        debug_assert!(b <= 1);
        self.bit_buf |= b << (7 - self.bit_buf_n);
        self.bit_buf_n += 1;
        if self.bit_buf_n == 8 {
            self.out.push(self.bit_buf);
            self.bit_buf = 0;
            self.bit_buf_n = 0;
        }
    }

    /// Propagate a +1 carry into the most-recently-emitted V bit. If
    /// the bit was a 1, it becomes 0 and the carry continues into the
    /// next-older bit, walking back through `bit_buf` and then `out`.
    ///
    /// A carry that walks off the front of `out` is absorbed by the
    /// dropped K=1 bit (which is provably 0 ⇒ +1 = 1 ⇒ stops there).
    fn propagate_carry(&mut self) {
        if self.bit_buf_n > 0 {
            // Most recent bit is in bit_buf at position (8 - bit_buf_n).
            // Adding `1 << position` increments that bit; overflow of
            // bit_buf (carry out of bit 7) means the carry continues
            // into out's last byte.
            let pos = 8 - self.bit_buf_n;
            let mask: u8 = 1 << pos;
            let (new_buf, carry_out) = self.bit_buf.overflowing_add(mask);
            self.bit_buf = new_buf;
            if carry_out {
                self.propagate_into_out();
            }
        } else {
            // bit_buf is empty — most recent bit is at LSB (bit 0) of
            // out's last byte.
            self.propagate_into_out();
        }
    }

    /// Propagate +1 from `out`'s last byte upward. Each 0xFF byte
    /// rolls to 0x00 with carry continuing to the previous byte.
    fn propagate_into_out(&mut self) {
        let mut i = self.out.len();
        while i > 0 {
            i -= 1;
            let (new, carry) = self.out[i].overflowing_add(1);
            self.out[i] = new;
            if !carry {
                return;
            }
        }
        // Carry walked off the front of `out`. By the encoder
        // invariant (v_low + v_rng ≤ 2^(15+scale)), this can only
        // happen when the dropped K=1 bit absorbs the carry — it's
        // always 0 so +1 = 1, no further propagation. We model this
        // by simply dropping the carry on the floor here.
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

    /// 1M-symbol roundtrip: stresses both encoder throughput and the
    /// streaming property — encoder state stays O(1) (`low` is 4
    /// bytes, `bit_buf` is 1, etc.), only `out` grows. Memory usage:
    /// `out.len()` bytes ≈ symbol-count × bits-per-symbol / 8. For a
    /// uniform 4-way CDF that is roughly 250 KB.
    #[test]
    fn roundtrip_one_million_symbols_uniform() {
        let cum = [8192u16, 16384, 24576, 32768];
        let n: usize = 1_000_000;
        let mut enc_cdf = cdf_from_cumulative(&cum);
        let mut enc = SymbolEncoder::new(false);
        // Deterministic xorshift-style symbol generator to exercise
        // every CDF branch uniformly.
        let mut state: u32 = 0x1234_5678;
        let mut symbols: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let s = state & 0x3;
            symbols.push(s);
            enc.encode_symbol(&mut enc_cdf, s);
        }
        let buf = enc.finish();

        let mut dec_cdf = cdf_from_cumulative(&cum);
        let mut dec = SymbolDecoder::new(&buf, buf.len(), false).expect("decoder init");
        for (i, &want) in symbols.iter().enumerate() {
            let got = dec.decode_symbol(&mut dec_cdf).expect("decode");
            if got != want {
                panic!("1M roundtrip mismatch at {i}: want {want}, got {got}");
            }
        }
    }

    /// 1M-bit skewed-probability bool stream — exercises the carry
    /// propagation path heavily (skewed-towards-1 streams pile up
    /// 0xFF runs which then carry-flip to 0x00).
    #[test]
    fn roundtrip_one_million_skewed_bools() {
        let n: usize = 1_000_000;
        let p: u32 = 28000; // p(bit=1) = 28000/32768 ≈ 0.85
        let mut enc = SymbolEncoder::new(false);
        let mut state: u32 = 0xABCD_EF01;
        let mut bits: Vec<u32> = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // ~85% probability of 1
            let b = if (state & 0xFFFF) < (p * 2) { 1 } else { 0 };
            bits.push(b);
            enc.encode_bool(b, p);
        }
        let buf = enc.finish();

        let mut dec = SymbolDecoder::new(&buf, buf.len(), false).expect("decoder init");
        for (i, &want) in bits.iter().enumerate() {
            let got = dec.decode_bool(p);
            if got != want {
                panic!("1M skewed bool roundtrip mismatch at {i}: want {want}, got {got}");
            }
        }
    }
}
