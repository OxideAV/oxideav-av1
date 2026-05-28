//! The AV1 entropy ("Symbol") **encoder** — the inverse of the §8.2
//! `SymbolDecoder` (`init_symbol` / `read_symbol` / `exit_symbol`).
//!
//! Tile content — coefficients, motion vectors, partition trees, etc. —
//! is arithmetically coded with the per-symbol CDFs the spec spells out
//! in §9.4 / §9.5. The encoder side has to produce the very same
//! bytestream the §8.2 decoder will read back. This module is the engine
//! the future tile-content **encode** passes will sit on top of.
//!
//! Like its decoder counterpart, this module is deliberately self
//! contained: only the §8.2 state transitions are implemented here, and
//! the CDF arrays themselves stay with the per-syntax-element wiring
//! (§5.11 / §9.4 / §9.5) that consumes them. The writer accepts any
//! caller-supplied `cdf: &mut [u16]` slice that obeys the §8.2.6
//! contract: length `N + 1`, `cdf[N-1] == 1 << 15`, `cdf[N]` the §8.3
//! adaptation counter (modified in place exactly like the decoder
//! mutates its CDF arrays).
//!
//! ## Derivation of the encoder
//!
//! The decoder's §8.2.2 / §8.2.6 state is `(SymbolValue, SymbolRange,
//! SymbolMaxBits)`. Define `u = (SymbolRange - 1) - SymbolValue`. After
//! §8.2.2 initialisation, `u == paddedBuf` — literally the top 15 bits
//! of the input stream the decoder will consume. The §8.2.6 symbol-
//! decode step `SymbolValue -= cur` rewrites to
//! `u_post = u_pre - (SymbolRange - prev)`. The §8.2.6 renormalisation
//! step expands to `u_new = (u_post << bits) | paddedData`, i.e. each
//! renorm shift makes room for the next `bits` bytestream bits at the
//! bottom of `u`.
//!
//! The encoder is the symmetric construction: maintain `(low, range)`
//! where `low` is the lower edge of the encoder's still-valid `u`
//! interval and `range` is its width. Encoding a symbol whose decoder
//! interval is `[cur, prev)` adds `range_pre - prev` to `low` (shifting
//! the interval's lower edge up to the symbol's slot in `u`-space) and
//! sets `range = prev - cur`. The §8.2.6 renormalisation shifts both
//! `low` and `range` left by `bits = 15 - FloorLog2(range)` — exactly
//! the reverse of the decoder's `(u << bits) | paddedData` step.
//!
//! `low` grows as bits accumulate; at [`Self::finish`] the encoder
//! emits the leading `15 + accumulated_renorm_bits` bits of `low`
//! MSB-first as the bytestream. The decoder's `paddedBuf` is then the
//! top 15 bits of that emission, which by construction lies in the
//! encoder's recorded `[low, low + range)` symbol interval for every
//! decoded symbol.
//!
//! Because the bit-width of `low` grows linearly with the symbol count,
//! we represent it as a `Vec<bool>` of MSB-first bits with explicit
//! big-integer addition; correct, simple, and bounded by the test
//! sizes (a few hundred symbols at <16 bits per renorm step ⇒ a few
//! KB at worst).
//!
//! ## What this module does *not* implement yet
//!
//! * `read_literal` / `NS` / `decode_subexp_bool` inverses — they
//!   compose §8.2.3's boolean encode the same way the decoder side
//!   composes `read_bool`. Future arcs add them on top of the
//!   primitives below.
//! * Tile-payload framing (`init_symbol(sz)` / `exit_symbol()`'s
//!   trailing-bit sanity check) — the §5.11 `tile_group_obu` writer
//!   will wrap this with the `tile_size_minus_1` size field and the
//!   §5.3.4 trailer.
//! * Byte-level flushing during encode — the current implementation
//!   keeps the entire accumulated `low` in memory until
//!   [`Self::finish`]. This is bounded for tile-payload sizes that
//!   fit comfortably in RAM. A future arc that needs to encode
//!   multi-megabyte tiles can add the standard carry-buffer flush.

use crate::Error;

/// `EC_PROB_SHIFT` — number of bits to reduce CDF precision during
/// arithmetic coding (§3, symbol-table constant; value 6). Same as
/// [`crate::symbol_decoder`].
const EC_PROB_SHIFT: u32 = 6;

/// `EC_MIN_PROB` — minimum probability assigned to each symbol during
/// arithmetic coding (§3, symbol-table constant; value 4). Same as
/// [`crate::symbol_decoder`].
const EC_MIN_PROB: u32 = 4;

/// `FloorLog2(x)` per §4.7.
fn floor_log2(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        31 - x.leading_zeros()
    }
}

/// The AV1 symbol (arithmetic) **encoder** — inverse of
/// [`crate::symbol_decoder::SymbolDecoder`].
///
/// Construct with [`SymbolWriter::new`]. Call
/// [`SymbolWriter::write_symbol`] (or [`SymbolWriter::write_bool`]) for
/// each syntax element. Call [`SymbolWriter::finish`] at end-of-payload
/// to flush the accumulated `low` and return the encoded bytes — the
/// resulting `Vec<u8>` can be handed straight to a §8.2 decoder via
/// `SymbolDecoder::init_symbol(&bytes, bytes.len(), …)` and round-trips
/// the same symbol sequence.
///
/// `disable_cdf_update` mirrors the §5.9.2 frame-header flag the
/// decoder side already plumbs: when `true`, [`Self::write_symbol`]
/// skips the §8.3 CDF adaptation. Boolean writes ([`Self::write_bool`])
/// follow the §8.2.3 "fresh CDF per call" pattern and are unaffected.
#[derive(Debug)]
pub struct SymbolWriter {
    /// Accumulated `low` value as MSB-first bits. The length of the
    /// vector grows by `15` at init and by `bits` per renorm step; the
    /// bit at index 0 is the most significant bit of the final encoded
    /// bytestream.
    low_bits: Vec<bool>,
    /// Width of the current sub-interval. Kept in `[1, 1 << 16]`
    /// transiently and renormalised back to `[1 << 15, 1 << 16]`
    /// before each symbol write completes.
    range: u32,
    /// Honour the §5.9.2 `disable_cdf_update` flag — same semantics as
    /// the decoder side. `read_bool` / `write_bool` are unaffected
    /// (per the §8.2.3 note) because the CDF is reconstructed per
    /// call.
    disable_cdf_update: bool,
}

impl SymbolWriter {
    /// Initialise an empty encoder, mirroring §8.2.2's `SymbolRange =
    /// 1 << 15`. The accumulated `low` is conceptually a 15-bit zero;
    /// we represent it as 15 `false` bits so subsequent shifts /
    /// adds operate on a uniform bit grid.
    pub fn new(disable_cdf_update: bool) -> Self {
        Self {
            // The initial `low = 0` in 15-bit precision (matching the
            // §8.2.2 `SymbolRange = 1 << 15` initial window).
            low_bits: vec![false; 15],
            range: 1 << 15,
            disable_cdf_update,
        }
    }

    /// `write_symbol( symbol, cdf )` — inverse of §8.2.6
    /// `read_symbol( cdf )`. Encodes `symbol` (which must be in
    /// `0..N` where `N = cdf.len() - 1`) and, if
    /// `!disable_cdf_update`, applies the §8.3 CDF update in
    /// lockstep with the decoder so the next encode sees the same
    /// adapted CDF the decoder will.
    pub fn write_symbol(&mut self, symbol: u32, cdf: &mut [u16]) -> Result<(), Error> {
        let n = cdf.len() as u32 - 1;
        debug_assert!(n >= 1, "§8.2.6: N must be greater than 1");
        debug_assert!(symbol < n, "symbol {symbol} out of range for N={n}");

        // §8.2.6: compute the boundary points of the symbol's
        // sub-interval the decoder would land in. For symbol s the
        // decoder's `cur(s)` is the lower edge of the post-decode value
        // range; the upper edge is the previous iteration's `cur(s -
        // 1)` (or `SymbolRange` itself for s == 0).
        let cur = self.cur_for_symbol(cdf, symbol, n);
        let prev = if symbol == 0 {
            self.range
        } else {
            self.cur_for_symbol(cdf, symbol - 1, n)
        };

        // Decoder semantics: `SymbolValue >= cur` lands in symbol s's
        // interval `[cur, prev)`. In `u = (SymbolRange - 1) -
        // SymbolValue` coordinates that's `[range - prev, range - cur)`.
        // Encoder: shift `low` up by `range - prev` (the bottom of the
        // symbol's u-interval) and shrink `range` to the width.
        let offset = self.range - prev;
        self.range = prev - cur;
        debug_assert!(self.range >= 1, "§8.2.6 produced a zero-width sub-interval");
        self.add_to_low(offset);

        // §8.2.6 renormalisation: shift `low` and `range` left by
        // `bits = 15 - FloorLog2(range)` so the new `range >= (1 <<
        // 15)`. The shift on `low_bits` is just appending `bits` zero
        // bits.
        let bits = 15 - floor_log2(self.range);
        self.range <<= bits;
        for _ in 0..bits {
            self.low_bits.push(false);
        }

        // §8.3 CDF update — identical to the decoder side. Skipped when
        // the frame header's `disable_cdf_update` was set.
        if !self.disable_cdf_update {
            crate::symbol_decoder::update_cdf_for_encoder(cdf, symbol, n);
        }

        Ok(())
    }

    /// `write_bool( bit )` — inverse of §8.2.3 `read_bool()` (which is
    /// itself §8.2.6 over the fixed CDF `[1<<14, 1<<15, 0]`). The CDF
    /// is reconstructed per call, exactly as on the decoder side, so
    /// the §8.3 update has no observable effect.
    pub fn write_bool(&mut self, bit: u32) -> Result<(), Error> {
        let mut cdf: [u16; 3] = [1 << 14, 1 << 15, 0];
        self.write_symbol(bit, &mut cdf)
    }

    /// `write_literal( n, value )` — inverse of §8.2.5
    /// [`crate::symbol_decoder::SymbolDecoder::read_literal`].
    ///
    /// The §8.2.5 decoder is the loop
    /// `for ( i = 0; i < n; i++ ) x = 2 * x + read_bool()`; the
    /// inverse emits the low `n` bits of `value` MSB-first via
    /// [`Self::write_bool`]. `value` must fit in `n` bits (`n <= 32`).
    pub fn write_literal(&mut self, n: u32, value: u32) -> Result<(), Error> {
        debug_assert!(n <= 32, "§8.2.5 read_literal only consumes up to 32 bits");
        debug_assert!(
            n == 32 || value < (1u32 << n),
            "value {value} does not fit in {n} bits"
        );
        // MSB-first: bit (n-1) emitted first so the decoder's `x = 2*x +
        // bit` rebuild yields the original value.
        for i in (0..n).rev() {
            let bit = (value >> i) & 0x1;
            self.write_bool(bit)?;
        }
        Ok(())
    }

    /// `write_ns( n, value )` — inverse of §4.10.10 `NS(n)` (the
    /// arithmetic-coded counterpart of §4.10.7 `ns(n)`), implemented as
    /// the inverse of
    /// [`crate::symbol_decoder::SymbolDecoder::read_ns`].
    ///
    /// The §4.10.10 decoder is
    ///
    /// ```text
    ///   w = FloorLog2(n) + 1
    ///   m = (1 << w) - n
    ///   v = L(w - 1)
    ///   if (v < m) return v
    ///   extra_bit = L(1)
    ///   return (v << 1) - m + extra_bit
    /// ```
    ///
    /// The inverse: values `0..m` are coded directly in `w - 1` literal
    /// bits; values `m..n` are coded as `(value + m)` in `w` literal
    /// bits (the high bit of `value + m` becomes the §4.10.10 `v`
    /// post-condition `v >= m`, and the low bit lands as `extra_bit`).
    /// This mirrors `BitWriter::write_ns` (§4.10.7) one level up the
    /// stack but over the arithmetic-coded literal path.
    pub fn write_ns(&mut self, n: u32, value: u32) -> Result<(), Error> {
        debug_assert!(n >= 1, "§4.10.10 NS(n) requires n >= 1");
        debug_assert!(value < n, "NS({n}) value {value} out of range");
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        if value < m {
            self.write_literal(w - 1, value)
        } else {
            // `(value + m)` written in `w` literal bits. The decoder's
            // `L(w - 1)` reads the high `w - 1` bits — call it `v` —
            // and then `L(1)` reads the low bit as `extra_bit`. The
            // reconstruction `(v << 1) - m + extra_bit` rearranges to
            // `(v << 1) | extra_bit - m == (value + m) - m == value`.
            let coded = value + m;
            self.write_literal(w, coded)
        }
    }

    /// `write_subexp_bool( num_syms, k, value )` — inverse of §5.9.28
    /// [`crate::symbol_decoder::SymbolDecoder::decode_subexp_bool`].
    ///
    /// The §5.9.28 decoder loop:
    ///
    /// ```text
    ///   i = 0; mk = 0
    ///   while (1) {
    ///     b2 = i ? k + i - 1 : k
    ///     a = 1 << b2
    ///     if ( numSyms <= mk + 3 * a ) {
    ///       subexp_unif_bools = NS(numSyms - mk)
    ///       return subexp_unif_bools + mk
    ///     } else {
    ///       subexp_more_bools = L(1)
    ///       if ( subexp_more_bools ) { i++; mk += a }
    ///       else { subexp_bools = L(b2); return subexp_bools + mk }
    ///     }
    ///   }
    /// ```
    ///
    /// The inverse walks the same `(i, mk)` ladder against the supplied
    /// `value`: while the current step's `a`-wide chunk does not cover
    /// `value`, emit `subexp_more_bools = 1`, advance to the next rung.
    /// Once the step covers `value`, emit either the uniform tail
    /// (`NS(numSyms - mk)` carrying `value - mk`) or the fixed-width
    /// tail (`subexp_more_bools = 0` followed by `L(b2)` carrying
    /// `value - mk`).
    pub fn write_subexp_bool(&mut self, num_syms: u32, k: u32, value: u32) -> Result<(), Error> {
        debug_assert!(num_syms >= 1, "§5.9.28 subexp requires numSyms >= 1");
        debug_assert!(value < num_syms, "subexp value {value} out of range");
        let mut i: u32 = 0;
        let mut mk: u32 = 0;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b2;
            if num_syms <= mk + 3 * a {
                // Uniform tail: the §5.9.28 decoder takes `NS(numSyms -
                // mk)` and adds `mk`. The encoder writes `value - mk`
                // through the same `NS(numSyms - mk)`.
                return self.write_ns(num_syms - mk, value - mk);
            }
            if value >= mk + a {
                // Not yet covered by this rung — emit `subexp_more_bools
                // = 1` and advance.
                self.write_literal(1, 1)?;
                i += 1;
                mk += a;
            } else {
                // Fixed-width tail: emit `subexp_more_bools = 0` then
                // `L(b2)` carrying `value - mk`.
                self.write_literal(1, 0)?;
                return self.write_literal(b2, value - mk);
            }
        }
    }

    /// Compute the decoder's `cur(s)` for the symbol-search step at the
    /// current `range`. Pulled into a helper so [`Self::write_symbol`]
    /// can also use it for `prev = cur(s - 1)`.
    fn cur_for_symbol(&self, cdf: &[u16], s: u32, n: u32) -> u32 {
        let f = (1u32 << 15) - u32::from(cdf[s as usize]);
        let mut cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
        cur += EC_MIN_PROB * (n - s - 1);
        cur
    }

    /// Add `offset` to `low` (treated as a big-endian MSB-first
    /// integer). The addition happens at the *least significant* bit
    /// of `low_bits` and carries propagate up through the vector. If
    /// the carry escapes past the most significant bit, we prepend a
    /// new `true` bit — but this only happens when `low + offset >= 2
    /// ^ low_bits.len()`, which the per-symbol invariant `low +
    /// range <= 2 ^ low_bits.len()` (mirroring the decoder's
    /// `SymbolValue` invariant) keeps from ever triggering. We assert
    /// instead of `insert(0, true)` so a future bug shows up as a
    /// loud test failure rather than a silently-mis-aligned
    /// bitstream.
    fn add_to_low(&mut self, offset: u32) {
        let mut carry: u64 = u64::from(offset);
        // Walk from LSB (last bit) to MSB (first bit).
        for i in (0..self.low_bits.len()).rev() {
            if carry == 0 {
                break;
            }
            let bit_val: u64 = if self.low_bits[i] { 1 } else { 0 };
            let sum = bit_val + (carry & 1);
            self.low_bits[i] = (sum & 1) != 0;
            carry = (carry >> 1) + (sum >> 1);
        }
        debug_assert!(
            carry == 0,
            "`low + offset` overflowed the §8.2.6 interval invariant"
        );
    }

    /// Flush the encoder and return the encoded byte sequence.
    ///
    /// The accumulated `low_bits` are the MSB-first bits of the
    /// bytestream. We pack them eight at a time into output bytes,
    /// padding any partial trailing byte with §8.2.2 zero bits — the
    /// decoder consumes these via its `numBits = Min(bits, Max(0,
    /// SymbolMaxBits))` clamp.
    pub fn finish(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.low_bits.len().div_ceil(8));
        let mut cur: u8 = 0;
        let mut bits_in_cur: u32 = 0;
        for b in self.low_bits {
            cur |= (b as u8) << (7 - bits_in_cur);
            bits_in_cur += 1;
            if bits_in_cur == 8 {
                out.push(cur);
                cur = 0;
                bits_in_cur = 0;
            }
        }
        if bits_in_cur != 0 {
            out.push(cur);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol_decoder::SymbolDecoder;

    /// `write_bool` followed by `read_bool` recovers the original bit.
    #[test]
    fn bool_round_trip_single_zero() {
        let mut w = SymbolWriter::new(true);
        w.write_bool(0).unwrap();
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        assert_eq!(d.read_bool().unwrap(), 0);
    }

    #[test]
    fn bool_round_trip_single_one() {
        let mut w = SymbolWriter::new(true);
        w.write_bool(1).unwrap();
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        assert_eq!(d.read_bool().unwrap(), 1);
    }

    /// Mixed pattern of bools.
    #[test]
    fn bool_round_trip_pattern() {
        let pattern = [0u32, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1];
        let mut w = SymbolWriter::new(true);
        for &b in &pattern {
            w.write_bool(b).unwrap();
        }
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        for &expected in &pattern {
            assert_eq!(d.read_bool().unwrap(), expected, "bit stream mismatch");
        }
    }

    /// Multi-symbol round trip with a uniform-ish N=4 CDF and the §8.3
    /// adaptation enabled on *both* sides.
    #[test]
    fn multi_symbol_round_trip_adapting_cdf() {
        let symbols = [0u32, 2, 1, 3, 0, 1, 2, 3, 0, 0, 1, 2, 3, 1, 2, 0, 3, 1];
        let start_cdf: [u16; 5] = [8192, 16384, 24576, 32768, 0];

        let mut w = SymbolWriter::new(false);
        let mut enc_cdf = start_cdf;
        for &s in &symbols {
            w.write_symbol(s, &mut enc_cdf).unwrap();
        }
        let bytes = w.finish();

        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut dec_cdf = start_cdf;
        for &expected in &symbols {
            let got = d.read_symbol(&mut dec_cdf).unwrap();
            assert_eq!(got, expected, "symbol mismatch");
        }
        assert_eq!(enc_cdf, dec_cdf, "encoder/decoder CDFs diverged");
    }

    /// Skewed CDF (heavy bias toward symbol 0).
    #[test]
    fn skewed_cdf_round_trip_disabled_update() {
        let cdf_template: [u16; 3] = [28672, 32768, 0];
        let symbols = [0u32; 32];
        let mut w = SymbolWriter::new(true);
        let mut wc = cdf_template;
        for &s in &symbols {
            w.write_symbol(s, &mut wc).unwrap();
        }
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        let mut dc = cdf_template;
        for &expected in &symbols {
            assert_eq!(d.read_symbol(&mut dc).unwrap(), expected);
        }
    }

    /// Opposite skew — encoding the low-probability symbol repeatedly
    /// stresses the carry path because each `range - prev` add is
    /// large.
    #[test]
    fn rare_symbol_round_trip() {
        let cdf_template: [u16; 3] = [4096, 32768, 0];
        let symbols = [1u32; 20];
        let mut w = SymbolWriter::new(true);
        let mut wc = cdf_template;
        for &s in &symbols {
            w.write_symbol(s, &mut wc).unwrap();
        }
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        let mut dc = cdf_template;
        for &expected in &symbols {
            assert_eq!(d.read_symbol(&mut dc).unwrap(), expected);
        }
    }

    /// Pseudo-random symbol sequence with a longer N=8 CDF and §8.3
    /// adaptation.
    #[test]
    fn long_sequence_round_trip_with_adaptation() {
        let start_cdf: [u16; 9] = [3000, 7000, 12000, 16000, 20000, 24000, 28000, 32768, 0];
        let mut x: u32 = 0xDEAD_BEEF;
        let mut symbols = Vec::with_capacity(200);
        for _ in 0..200 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            symbols.push(x % 8);
        }

        let mut w = SymbolWriter::new(false);
        let mut enc_cdf = start_cdf;
        for &s in &symbols {
            w.write_symbol(s, &mut enc_cdf).unwrap();
        }
        let bytes = w.finish();

        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let mut dec_cdf = start_cdf;
        for &expected in &symbols {
            assert_eq!(d.read_symbol(&mut dec_cdf).unwrap(), expected);
        }
        assert_eq!(enc_cdf, dec_cdf);
    }

    /// `write_bool` mixed with `write_symbol` on the same writer.
    #[test]
    fn mixed_bool_and_symbol_round_trip() {
        let bools = [1u32, 0, 1, 1];
        let multi: [u32; 4] = [2, 0, 3, 1];
        let start_cdf: [u16; 5] = [6000, 14000, 22000, 32768, 0];

        let mut w = SymbolWriter::new(true);
        for &b in &bools {
            w.write_bool(b).unwrap();
        }
        let mut wc = start_cdf;
        for &s in &multi {
            w.write_symbol(s, &mut wc).unwrap();
        }
        let bytes = w.finish();

        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        for &expected in &bools {
            assert_eq!(d.read_bool().unwrap(), expected);
        }
        let mut dc = start_cdf;
        for &expected in &multi {
            assert_eq!(d.read_symbol(&mut dc).unwrap(), expected);
        }
    }

    /// Empty encode — `finish()` on a fresh writer must still produce
    /// a buffer the decoder can wrap (with no symbols read).
    #[test]
    fn empty_payload_finish_does_not_panic() {
        let w = SymbolWriter::new(true);
        let bytes = w.finish();
        let _ = SymbolDecoder::init_symbol(&bytes, bytes.len(), true);
    }

    /// All-ones boolean stream.
    #[test]
    fn all_ones_bool_stream_round_trip() {
        let n = 64;
        let mut w = SymbolWriter::new(true);
        for _ in 0..n {
            w.write_bool(1).unwrap();
        }
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        for _ in 0..n {
            assert_eq!(d.read_bool().unwrap(), 1);
        }
    }

    /// All-zeros boolean stream.
    #[test]
    fn all_zeros_bool_stream_round_trip() {
        let n = 64;
        let mut w = SymbolWriter::new(true);
        for _ in 0..n {
            w.write_bool(0).unwrap();
        }
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        for _ in 0..n {
            assert_eq!(d.read_bool().unwrap(), 0);
        }
    }

    // -----------------------------------------------------------------
    // §8.2.5 read_literal / §4.10.10 NS / §5.9.28 decode_subexp_bool —
    // composition-wrapper round-trips through the decoder side.
    // -----------------------------------------------------------------

    /// `write_literal(n, v)` round-trips through §8.2.5 `read_literal(n)`
    /// for a representative set of widths covering 1 through 16 bits.
    #[test]
    fn write_literal_round_trip_widths_1_to_16() {
        let cases: &[(u32, u32)] = &[
            (1, 0),
            (1, 1),
            (3, 0),
            (3, 5),
            (3, 7),
            (4, 0xA),
            (8, 0xCD),
            (12, 0xABC),
            (16, 0xDEAD),
            (16, 0xBEEF),
            (16, 0),
        ];
        for &(n, v) in cases {
            let mut w = SymbolWriter::new(true);
            w.write_literal(n, v).unwrap();
            let bytes = w.finish();
            let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
            assert_eq!(
                d.read_literal(n).unwrap(),
                v,
                "L({n}) round-trip mismatch for {v}"
            );
        }
    }

    /// `write_literal(0, 0)` writes no bits; the decoder's `read_literal(0)`
    /// also consumes no bits, so the writer's state is unchanged.
    #[test]
    fn write_literal_n_zero_emits_no_bits() {
        let mut w = SymbolWriter::new(true);
        w.write_literal(0, 0).unwrap();
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        assert_eq!(d.read_literal(0).unwrap(), 0);
    }

    /// Several `write_literal` calls back-to-back on the same writer
    /// must read back in order as separate values.
    #[test]
    fn write_literal_concatenated_round_trip() {
        let mut w = SymbolWriter::new(true);
        w.write_literal(3, 5).unwrap();
        w.write_literal(5, 17).unwrap();
        w.write_literal(8, 0xAB).unwrap();
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        assert_eq!(d.read_literal(3).unwrap(), 5);
        assert_eq!(d.read_literal(5).unwrap(), 17);
        assert_eq!(d.read_literal(8).unwrap(), 0xAB);
    }

    /// `write_ns(n, v)` round-trips through §4.10.10 `NS(n)` for every
    /// representable value at a range of `n`s covering the low / power-
    /// of-two / non-power-of-two cases the §5.9 subexp wrappers hit.
    #[test]
    fn write_ns_round_trip_against_read_ns() {
        for n in [1u32, 2, 3, 4, 5, 7, 8, 9, 13, 16, 17] {
            for v in 0..n {
                let mut w = SymbolWriter::new(true);
                w.write_ns(n, v).unwrap();
                let bytes = w.finish();
                let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
                assert_eq!(
                    d.read_ns(n).unwrap(),
                    v,
                    "NS({n}) round-trip mismatch for {v}"
                );
            }
        }
    }

    /// `write_ns(1, 0)` is a no-op (the decoder's `L(0)` returns 0
    /// without consuming any bits).
    #[test]
    fn write_ns_n_equals_one_no_bits() {
        let mut w = SymbolWriter::new(true);
        w.write_ns(1, 0).unwrap();
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        assert_eq!(d.read_ns(1).unwrap(), 0);
    }

    /// `write_subexp_bool(numSyms, k, v)` round-trips through §5.9.28
    /// `decode_subexp_bool` for a representative numSyms × k × value
    /// grid. Values are sampled rather than exhaustive to keep the test
    /// quick while still exercising the three §5.9.28 code paths:
    /// the immediate-uniform branch (`numSyms <= 3 * a`), the
    /// `subexp_more_bools=0` fixed-width tail, and the
    /// `subexp_more_bools=1` ladder advance.
    #[test]
    fn write_subexp_bool_round_trip_grid() {
        let cases: &[(u32, u32, &[u32])] = &[
            // Small numSyms → immediate uniform branch.
            (2, 0, &[0, 1]),
            (3, 1, &[0, 1, 2]),
            // Larger numSyms with k=1 forces several ladder advances.
            (32, 1, &[0, 1, 2, 3, 6, 7, 8, 15, 16, 31]),
            // numSyms equal to a power of two with k=2.
            (64, 2, &[0, 5, 11, 12, 24, 32, 47, 63]),
            // numSyms not a power of two; exercises the tail uniform
            // branch over a non-power-of-two interval.
            (50, 0, &[0, 1, 2, 10, 25, 40, 49]),
        ];
        for &(num_syms, k, values) in cases {
            for &v in values {
                let mut w = SymbolWriter::new(true);
                w.write_subexp_bool(num_syms, k, v).unwrap();
                let bytes = w.finish();
                let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
                assert_eq!(
                    d.decode_subexp_bool(num_syms, k).unwrap(),
                    v,
                    "subexp_bool(num_syms={num_syms}, k={k}) round-trip mismatch for {v}"
                );
            }
        }
    }

    /// Several subexp values back-to-back on the same writer.
    #[test]
    fn write_subexp_bool_concatenated_round_trip() {
        let triples: &[(u32, u32, u32)] = &[(16, 1, 5), (8, 0, 3), (32, 2, 17), (64, 1, 0)];
        let mut w = SymbolWriter::new(true);
        for &(num_syms, k, v) in triples {
            w.write_subexp_bool(num_syms, k, v).unwrap();
        }
        let bytes = w.finish();
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), true).unwrap();
        for &(num_syms, k, v) in triples {
            assert_eq!(d.decode_subexp_bool(num_syms, k).unwrap(), v);
        }
    }
}
