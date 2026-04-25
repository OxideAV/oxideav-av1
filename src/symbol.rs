//! AV1 range-coder symbol decoder.
//!
//! Mirrors libaom's `daala_bitreader`. Sections §3 (arithmetic coder) +
//! §9.2 (symbol coding) + §9.4 (CDF adaptation) of the AV1
//! specification (2019-01-08).
//!
//! CDFs are stored in **wire format**: `cdf[0..N-2]` = Q15
//! `P(X>i) * 32768` values (monotonically decreasing), `cdf[N-1]` = 0
//! sentinel, `cdf[N]` = update counter (starts at 0, saturates at 32).
//! This is the representation produced by our
//! [`crate::cdfs`] tables and by libaom's `AOM_CDFn` macros.
//!
//! The decoder consumes bits MSB-first from the tile payload and
//! maintains three pieces of state:
//!
//! - `symbol_range` — current interval size, normalised to
//!   `[2^14, 2^15)` after each `renormalise()`.
//! - `symbol_value` — the interval offset, advanced from the stream
//!   after each renorm.
//! - `max_bits` — remaining bit budget for the tile payload. Goes
//!   negative only if the encoder over-emitted — callers may flag it,
//!   but the decoder keeps returning zeros.

use oxideav_core::{Error, Result};

/// Low-probability floor per symbol (libaom `EC_MIN_PROB`).
pub const MIN_PROB: u32 = 4;

/// CDF probability right-shift (libaom `EC_PROB_SHIFT`).
pub const PROB_SHIFT: u32 = 6;

/// Initial range after `init_symbol()` (§9.2.1).
pub const SYMBOL_RANGE_INIT: u32 = 1 << 15;

/// Threshold below which `renormalise()` shifts the range left.
pub const SYMBOL_CARRY: u32 = 0x8000;

/// Tile-scoped range-coder state.
pub struct SymbolDecoder<'a> {
    data: &'a [u8],
    /// Bit position within `data` (MSB-first).
    bit_pos: usize,
    /// Maximum bit index allowed (usually `8 * min(sz, data.len())`).
    bit_limit: usize,
    symbol_value: u32,
    symbol_range: u32,
    max_bits: i64,
    allow_update: bool,
}

impl<'a> SymbolDecoder<'a> {
    /// Initialise per §9.2.1.
    ///
    /// `sz` is the number of payload bytes the decoder may consume
    /// (usually the tile size). Clamped to `data.len()`.
    ///
    /// When `allow_update` is true, each subsequent `decode_symbol()`
    /// call adapts the CDF in-place per §9.4.
    pub fn new(data: &'a [u8], sz: usize, allow_update: bool) -> Result<Self> {
        let sz = sz.min(data.len());
        if sz == 0 {
            return Err(Error::invalid(
                "av1 symbol: empty payload — §9.2.1 init_symbol",
            ));
        }
        let mut me = Self {
            data,
            bit_pos: 0,
            bit_limit: sz * 8,
            symbol_value: 0,
            symbol_range: SYMBOL_RANGE_INIT,
            max_bits: (sz as i64) * 8 - 15,
            allow_update,
        };
        let num_bits = 15usize.min(sz * 8);
        let buf_val = me.read_bits(num_bits);
        let padded_buf = buf_val << (15 - num_bits as u32);
        me.symbol_value = ((1u32 << 15) - 1) ^ padded_buf;
        Ok(me)
    }

    /// Decode a single symbol from `cdf`.
    ///
    /// `cdf` must have `N + 1 + 1` entries: `N` probability thresholds,
    /// a 0 sentinel, and the adaptive-update counter. Returns a value
    /// in `0..N-1`. Adapts the CDF in place when `allow_update` was
    /// set at construction.
    ///
    /// Implements §8.2.6 `read_symbol`. The wire format here is the
    /// **inverse CDF** (as produced by libaom's `AOM_CDFn` / our
    /// generator), so each `cdf[symbol]` already equals the spec's
    /// `f = (1 << 15) - cdf_forward[symbol]`. The §8.2.6 inner loop is:
    /// ```text
    ///   cur = SymbolRange
    ///   do {
    ///     symbol++
    ///     prev = cur
    ///     cur = ((R >> 8) * (cdf[symbol] >> 6)) >> 1 + 4*(N-symbol-1)
    ///   } while (SymbolValue < cur)
    ///   SymbolRange = prev - cur
    ///   SymbolValue -= cur
    /// ```
    /// (note: spec's "f" is our `cdf[symbol]` directly, since the wire
    /// format is the inverse CDF).
    pub fn decode_symbol(&mut self, cdf: &mut [u16]) -> Result<u32> {
        if cdf.len() < 2 {
            return Err(Error::invalid("av1 symbol: CDF too short"));
        }
        // N = number of symbols; cdf has N+1 entries (including the
        // sentinel) plus the counter tail slot.
        let n_with_sentinel = cdf.len() - 1;
        if n_with_sentinel < 1 {
            return Ok(0);
        }
        let n = n_with_sentinel;
        // §8.2.6: cur starts at SymbolRange (so prev becomes
        // SymbolRange on the first iteration when symbol == 0).
        let mut cur: u32 = self.symbol_range;
        let mut prev: u32;
        let mut symbol: usize = 0;
        loop {
            prev = cur;
            let f = cdf[symbol] as u32;
            cur = ((self.symbol_range >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
            cur += MIN_PROB * (n as u32 - symbol as u32 - 1);
            if self.symbol_value >= cur {
                // Spec's `do { ... } while (SymbolValue < cur)` — exit
                // condition is `SymbolValue >= cur`. The selected
                // symbol's interval is [cur, prev).
                break;
            }
            symbol += 1;
            if symbol >= n - 1 {
                // The last (implicit) symbol — its lower bound is 0
                // and we re-enter the body with f = 0 ⇒ cur = 0; that
                // would always satisfy `SymbolValue >= 0`, so we
                // short-circuit and treat `symbol = n - 1` as the
                // result. Update prev/cur to the spec-equivalent
                // values for the SymbolRange/Value adjustment below.
                prev = cur;
                cur = 0;
                break;
            }
        }
        self.symbol_range = prev - cur;
        self.symbol_value -= cur;
        self.renormalise();
        if self.allow_update {
            update_cdf(cdf, n, symbol);
        }
        Ok(symbol as u32)
    }

    /// Decode a single bit per §8.2.3, parameterised by `p = P(bit=1)`
    /// in Q15 (`0..=32768`). The §8.2.3 path itself only uses
    /// `p = 1<<14` (50/50); accepting an arbitrary `p` lets us also
    /// drive the rare boolean reads (cdef_idx etc.) that ride the
    /// range coder via a direct Q15 probability.
    ///
    /// Internally a 2-symbol read with cdf `{(1<<15)-p, 1<<15, 0}` —
    /// in our inverse wire format that's `cdf[0] = p`. We inline the
    /// 2-symbol path of `decode_symbol` so the per-bit cost stays
    /// minimal.
    pub fn decode_bool(&mut self, p: u32) -> u32 {
        // §8.2.6 `cur` for symbol 0, with f = (1<<15) - cdf_spec[0] = p:
        //   cur = ((R >> 8) * (p >> 6)) >> 1 + 4*(N - 0 - 1) = ... + 4
        let mut cur = ((self.symbol_range >> 8) * (p >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
        cur += MIN_PROB;
        let prev = self.symbol_range;
        let bit = if self.symbol_value >= cur {
            // Exit at symbol 0 (spec returns 0 when SV >= cur). The
            // caller uses bit=1 for the high-prob side per
            // `read_symbol`'s convention.
            self.symbol_range = prev - cur;
            self.symbol_value -= cur;
            0
        } else {
            // Advance to symbol 1 (the implicit last). cur becomes 0.
            self.symbol_range = cur;
            1
        };
        self.renormalise();
        bit
    }

    /// Read `n` raw 50/50 bools (§8.2.5 `read_literal`).
    pub fn read_literal(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.decode_bool(16384);
        }
        v
    }

    /// Remaining bit budget for this tile's payload. Negative values
    /// indicate the encoder over-emitted past the size prefix — the
    /// frame is still decodable but a well-formed encoder should not
    /// produce this.
    pub fn max_bits(&self) -> i64 {
        self.max_bits
    }

    /// §8.2.6 step 6 of the renormalisation: advance `SymbolValue` and
    /// `SymbolRange` by `bits = 15 - FloorLog2(SymbolRange)` new bits
    /// from the bitstream. The XOR with `((SymbolValue + 1) << bits) -
    /// 1` is what makes our internal `SymbolValue` match the spec's —
    /// this is the bit-COMPLEMENTING form, which is necessary because
    /// init_symbol XORs with `(1<<15)-1` at start.
    fn renormalise(&mut self) {
        if self.symbol_range >= SYMBOL_CARRY {
            return;
        }
        // FloorLog2(SymbolRange): how many bits the range occupies.
        // bits = 15 - FloorLog2(SymbolRange)
        let bits = 15 - floor_log2(self.symbol_range);
        // SymbolRange <<= bits brings range back into [2^15, 2^16).
        self.symbol_range <<= bits;
        // Read `bits` new bits, padded if past payload.
        let max_avail = self.max_bits.max(0) as u32;
        let num_bits = bits.min(max_avail);
        let new_data = self.read_bits(num_bits as usize);
        let padded_data = new_data << (bits - num_bits);
        // §8.2.6: SymbolValue = paddedData ^ (((SymbolValue + 1) << bits) - 1)
        let mask = ((self.symbol_value + 1) << bits).wrapping_sub(1);
        self.symbol_value = padded_data ^ mask;
        self.max_bits -= bits as i64;
    }

    fn read_bits(&mut self, n: usize) -> u32 {
        let mut v = 0u32;
        for _ in 0..n {
            v = (v << 1) | self.read_one_bit();
        }
        v
    }

    fn read_one_bit(&mut self) -> u32 {
        if self.bit_pos >= self.bit_limit {
            // Past the allotted payload — spec treats trailing bits as 0.
            self.bit_pos += 1;
            return 0;
        }
        let byte = self.bit_pos >> 3;
        let bit = 7 - (self.bit_pos & 7);
        let b = (self.data[byte] >> bit) & 1;
        self.bit_pos += 1;
        b as u32
    }
}

/// `FloorLog2(x)` — spec §4.7.2: largest integer `k` such that
/// `(1 << k) <= x`. For `x == 0` the result is undefined; callers
/// guarantee `x > 0` (renormalise only runs while `range > 0`).
fn floor_log2(x: u32) -> u32 {
    debug_assert!(x > 0);
    31 - x.leading_zeros()
}

/// CDF adaptation per §9.4.
///
/// `cdf` has `n + 1` probability entries (including the 0 sentinel at
/// index `n - 1`) plus a count slot at index `n`. The update rate is:
///
/// ```text
/// rate = 3 + (count >= 16) + (count >= 32) + (n > 3)
/// ```
///
/// Each non-sentinel entry is nudged by `(tmp - entry) / 2^rate` toward
/// `0` (entries before the decoded symbol) or `2^15` (entries at and
/// beyond the symbol). The count slot increments up to a saturation of
/// 32.
pub fn update_cdf(cdf: &mut [u16], n: usize, symbol: usize) {
    let count = cdf[n];
    let mut rate: u32 = 3;
    if count >= 16 {
        rate += 1;
    }
    if count >= 32 {
        rate += 1;
    }
    if n > 3 {
        rate += 1;
    }
    if count < 32 {
        cdf[n] = count + 1;
    }
    let mut tmp: u32 = 0;
    for (i, entry) in cdf.iter_mut().enumerate().take(n.saturating_sub(1)) {
        if i == symbol {
            tmp = 1 << 15;
        }
        let v = *entry as u32;
        // Arithmetic-shift semantics: `(v - tmp) >> rate` with sign
        // extension. libaom relies on C's implementation-defined
        // right-shift of a signed difference; we reproduce that by
        // casting to i32 for the difference then back.
        let diff = (v as i32) - (tmp as i32);
        let shifted = diff >> rate;
        *entry = (v as i32 - shifted) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_rejects_empty() {
        let data: [u8; 0] = [];
        assert!(SymbolDecoder::new(&data, 0, false).is_err());
    }

    #[test]
    fn init_consumes_15_bits() {
        // 4 bytes of all-ones: paddedBuf = 0x7FFF, XOR with 0x7FFF = 0.
        let buf = [0xFFu8; 4];
        let d = SymbolDecoder::new(&buf, 4, false).expect("init");
        assert_eq!(d.symbol_range, SYMBOL_RANGE_INIT);
        assert_eq!(d.symbol_value, 0);
        assert_eq!(d.max_bits(), 4 * 8 - 15);
    }

    #[test]
    fn decode_bool_extreme_prob() {
        // §8.2.6 with `p = 32767` ≈ P(bit=1) ≈ 1: the very first
        // symbol of an all-zero buffer (max SymbolValue = 0x7FFF)
        // should land in the high-prob slice. After the round-15
        // §8.2.6 fix the spec convention applies: `0` is the
        // high-prob branch (the loop exits immediately when
        // `SymbolValue >= cur`), so the result is `0`.
        let buf = [0u8; 32];
        let mut d = SymbolDecoder::new(&buf, 32, false).expect("init");
        let bit = d.decode_bool(32767);
        assert_eq!(bit, 0);
    }

    #[test]
    fn update_cdf_increments_count() {
        // 3-symbol CDF, all mass on symbol 1.
        let mut cdf: Vec<u16> = vec![1u16 << 15, 0, 0, 0];
        update_cdf(&mut cdf, 3, 0);
        assert_eq!(cdf[3], 1, "count slot must increment on update");
    }

    #[test]
    fn decode_symbol_two_way_picks_high_prob_on_zero_bits() {
        // 2-symbol CDF with 50/50 split stored as wire form
        // [32768 - 16384, 0, 0] = [16384, 0, 0]. With an all-zero
        // buffer the seeded symbol_value is 0x7FFF (max). Per §8.2.6
        // the first iteration computes cur = R/2 + 4 ≈ 16388, and
        // since SymbolValue (0x7FFF=32767) >= cur the loop exits
        // immediately with symbol = 0.
        let buf = [0u8; 8];
        let mut d = SymbolDecoder::new(&buf, 8, false).expect("init");
        let mut cdf: Vec<u16> = vec![16384, 0, 0];
        let s = d.decode_symbol(&mut cdf).expect("decode");
        assert_eq!(s, 0);
    }

    #[test]
    fn decode_partition_cdf_block_8x8() {
        // Sanity: real default CDF decode returns a value in range.
        let buf = [0xAAu8; 16];
        let mut d = SymbolDecoder::new(&buf, 16, true).expect("init");
        let mut cdf = crate::cdfs::DEFAULT_PARTITION_CDF[0].to_vec();
        let s = d.decode_symbol(&mut cdf).expect("decode");
        assert!(s < 4, "partition-8x8 must produce symbol in 0..=3, got {s}");
    }

    /// §8.2.6 worked example: with `buf = [0xAA; 16]` the first 15
    /// bits are `0x5555`, so after init `SymbolValue = 0x7FFF ^
    /// 0x5555 = 0x2AAA = 10922`. For `cdf = [13636, 7258, 2376, 0,
    /// 0]` the §8.2.6 loop is:
    ///
    ///   - symbol=0, cur = (32768>>8)*(13636>>6)>>1 + 4*3 = 13644.
    ///     SymbolValue (10922) < cur → continue.
    ///   - symbol=1, cur = (32768>>8)*(7258>>6)>>1 + 4*2 = 7240.
    ///     SymbolValue (10922) >= cur → exit.
    ///
    /// → returned symbol = 1. This locks in the spec convention so a
    /// future regression that re-introduces the round-1..14
    /// inverted-decoder bug is caught on this single test.
    #[test]
    fn decode_partition_cdf_block_8x8_spec_value() {
        let buf = [0xAAu8; 16];
        let mut d = SymbolDecoder::new(&buf, 16, false).expect("init");
        let mut cdf = crate::cdfs::DEFAULT_PARTITION_CDF[0].to_vec();
        let s = d.decode_symbol(&mut cdf).expect("decode");
        assert_eq!(s, 1, "spec §8.2.6 returns symbol 1 here, got {s}");
    }

    /// Golden vector for the §8.2.6-correct symbol decoder: for the
    /// same input buffer + CDF + adaptation, 32 consecutive decoded
    /// symbols must match byte-for-byte across builds. Captured after
    /// the round-15 spec fix.
    #[test]
    fn decode_symbols_match_reference() {
        let mut buf = [0u8; 256];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((i * 37) ^ 0x5A) as u8;
        }
        let mut d = SymbolDecoder::new(&buf, buf.len(), true).expect("init");
        let mut cdf = crate::cdfs::DEFAULT_PARTITION_CDF[0].to_vec();
        let expected = [
            0, 1, 0, 0, 3, 3, 3, 0, 2, 0, 1, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3,
            3, 0, 3,
        ];
        let got: Vec<u32> = (0..32)
            .map(|_| d.decode_symbol(&mut cdf).expect("decode"))
            .collect();
        assert_eq!(&got[..], &expected[..]);
    }
}
