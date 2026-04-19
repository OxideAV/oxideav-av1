//! AV1 range-coder symbol decoder.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/entropy/decoder.go`
//! (MIT) which in turn mirrors libaom's `daala_bitreader`. Sections §3
//! (arithmetic coder) + §9.2 (symbol coding) + §9.4 (CDF adaptation) of
//! the AV1 specification (2019-01-08).
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
    /// (equivalent to goavif's `Decoder.Init` `sz` argument — usually
    /// the tile size). Clamped to `data.len()`.
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
        let mut symbol: usize = 0;
        let mut prob: u32;
        loop {
            let f = cdf[symbol] as u32;
            let mut factor = ((self.symbol_range >> 8) * (f >> PROB_SHIFT)) >> (7 - PROB_SHIFT);
            factor += MIN_PROB * (n as u32 - symbol as u32 - 1);
            prob = self.symbol_range - factor;
            if self.symbol_value >= prob {
                self.symbol_value -= prob;
                self.symbol_range -= prob;
                symbol += 1;
                if symbol == n - 1 {
                    // Last symbol is implicit — no further range update.
                    break;
                }
            } else {
                self.symbol_range = prob;
                break;
            }
        }
        self.renormalise();
        if self.allow_update {
            update_cdf(cdf, n, symbol);
        }
        Ok(symbol as u32)
    }

    /// Decode a single bit with P(bit=1) given directly as a Q15
    /// probability (`0..=32768`). This is the §9.2.5 `boolean(p)` path
    /// used by a handful of uncompressed-looking literals (cdef_idx,
    /// etc.) that ride the range coder.
    pub fn decode_bool(&mut self, p: u32) -> u32 {
        let mut split = ((self.symbol_range - 1) * p) >> 15;
        split += MIN_PROB;
        let bit;
        if self.symbol_value < split {
            self.symbol_range = split;
            bit = 0;
        } else {
            self.symbol_range -= split;
            self.symbol_value -= split;
            bit = 1;
        }
        self.renormalise();
        bit
    }

    /// Read `n` raw 50/50 bools (spec §5.9.3 `L(n)` — not to be
    /// confused with the SPS/HDR bit reader `f(n)`). `n` must be in
    /// `0..=32`.
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

    fn renormalise(&mut self) {
        while self.symbol_range < SYMBOL_CARRY {
            self.symbol_range <<= 1;
            self.symbol_value = (self.symbol_value << 1) | self.read_bits(1);
            self.max_bits -= 1;
        }
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
        // extension — goavif relies on Go's two's-complement wrap with
        // `uint32 - uint32 >> rate` acting as a logical shift of the
        // wrapped difference. In Rust, we reproduce that by casting to
        // i32 for the difference then back.
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
        // P(bit=1) ≈ 32767 / 32768 ≈ 1.0, but with an all-zero buffer
        // the interval register is far from the top — first call should
        // still return 0.
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
    fn decode_symbol_two_way_selects_higher_on_zero_bits() {
        // 2-symbol CDF with 50/50 split stored as wire form
        // [32768 - 16384, 0, 0] = [16384, 0, 0]. With an all-zero
        // buffer the seeded symbol_value is ~0x7FFF, which is > prob
        // on the first iteration → symbol 1 selected.
        let buf = [0u8; 8];
        let mut d = SymbolDecoder::new(&buf, 8, false).expect("init");
        let mut cdf: Vec<u16> = vec![16384, 0, 0];
        let s = d.decode_symbol(&mut cdf).expect("decode");
        assert_eq!(s, 1);
    }

    #[test]
    fn decode_partition_cdf_block_8x8() {
        // Exercise a real default CDF from the generator. The exact
        // output depends on the stream but we just want to confirm
        // that decoding proceeds without panic and returns a valid
        // symbol in range.
        let buf = [0xAAu8; 16];
        let mut d = SymbolDecoder::new(&buf, 16, true).expect("init");
        let mut cdf = crate::cdfs::DEFAULT_PARTITION_CDF[0].to_vec();
        let s = d.decode_symbol(&mut cdf).expect("decode");
        assert!(s < 4, "partition-8x8 must produce symbol in 0..=3, got {s}");
    }

    /// Cross-validated against goavif's `entropy.Decoder` — for the
    /// same input buffer + CDF + adaptation, 32 consecutive decoded
    /// symbols must match byte-for-byte. The Go reference was captured
    /// with `cmd/check_goavif_symbol` in tools/ (not shipped).
    #[test]
    fn decode_symbols_match_goavif() {
        let mut buf = [0u8; 256];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = ((i * 37) ^ 0x5A) as u8;
        }
        let mut d = SymbolDecoder::new(&buf, buf.len(), true).expect("init");
        let mut cdf = crate::cdfs::DEFAULT_PARTITION_CDF[0].to_vec();
        let expected = [
            1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 2, 1, 2, 3, 0, 2, 0, 3, 3, 0, 1, 0, 0, 0, 0, 3, 0,
            0, 3, 2,
        ];
        let got: Vec<u32> = (0..32)
            .map(|_| d.decode_symbol(&mut cdf).expect("decode"))
            .collect();
        assert_eq!(&got[..], &expected[..]);
    }
}
