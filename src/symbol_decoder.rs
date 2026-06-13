//! The AV1 entropy ("Symbol") decoder — the arithmetic-coding engine
//! specified in §8.2 (Parsing process for symbol decoder) and §8.3
//! (Parsing process for CDF encoded syntax elements) of the AV1
//! Bitstream & Decoding Process Specification.
//!
//! Tile content — coefficients, motion vectors, partition trees, etc. —
//! is coded with an adaptive arithmetic coder rather than the plain
//! `f(n)` / `uvlc()` reads used for the headers. This module is the
//! engine that the future tile-content decode passes will sit on top
//! of. It is deliberately self-contained: the state variables
//! (`SymbolValue` / `SymbolRange` / `SymbolMaxBits` per §8.2.4),
//! initialisation (§8.2.2), boolean decode (§8.2.3), `read_literal`
//! (§8.2.5), the CDF-adaptive multisymbol `read_symbol` (§8.2.6 plus
//! the §8.3 CDF update), and `exit_symbol` (§8.2.4) all live here.
//!
//! The CDF arrays themselves (the default tables copied with the `Tile`
//! prefix in §8.2.2, and the §8.3.2 CDF-selection process that maps a
//! syntax element name to a CDF) are *not* part of this module — those
//! land alongside the tile-content decode that consumes them. This
//! module operates on any caller-supplied `cdf: &mut [u16]` slice that
//! obeys the §8.2.6 contract (length `N + 1`, `cdf[N-1] == 1 << 15`,
//! `cdf[N]` the adaptation counter).
//!
//! Bit input is provided by the same MSB-first `BitReader` used for
//! the headers (§8.1 `f(n)`), so `init_symbol` / renormalisation /
//! `exit_symbol` advance the *same* bit-position indicator the rest of
//! the OBU walk uses.

use crate::bitreader::BitReader;
use crate::Error;

/// `EC_PROB_SHIFT` — number of bits to reduce CDF precision during
/// arithmetic coding (§3, symbol-table constant; value 6).
const EC_PROB_SHIFT: u32 = 6;

/// `EC_MIN_PROB` — minimum probability assigned to each symbol during
/// arithmetic coding (§3, symbol-table constant; value 4).
const EC_MIN_PROB: u32 = 4;

/// `FloorLog2(x)` per §4.7 — floor of the base-2 logarithm of `x`.
///
/// The spec guarantees `x >= 1` at every call site in §8.2; this helper
/// returns 0 for `x == 0` defensively (never reached in practice — the
/// renormalisation step only invokes it on a `SymbolRange` that is at
/// least `1 << 8`, and the §8.3 rate calculation on `N >= 1`).
fn floor_log2(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        31 - x.leading_zeros()
    }
}

/// `inverse_recenter(r, v)` per §5.9.29. Shared by the
/// `decode_*_subexp_with_ref_bool` helpers (the arithmetic-coded twins
/// of the §5.9.26/§5.9.27 header-bitstream recentred reads).
fn inverse_recenter(r: i64, v: i64) -> i64 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// The AV1 symbol (arithmetic) decoder, §8.2.
///
/// Wraps the crate's MSB-first `BitReader` positioned at the
/// byte-aligned start of an arithmetic-coded partition (§8.2.2 notes the
/// position is always byte aligned when `init_symbol` is invoked, because
/// the uncompressed header and data partitions are always a whole number
/// of bytes long).
#[derive(Debug)]
pub struct SymbolDecoder<'a> {
    reader: BitReader<'a>,
    /// `SymbolValue` — §8.2.2 / §8.2.4.
    value: u32,
    /// `SymbolRange` — §8.2.2 / §8.2.4. Always in `(1 << 8) ..= (1 << 15)`.
    range: u32,
    /// `SymbolMaxBits` — §8.2.2 / §8.2.4. May go negative (signed) once
    /// every available bit has been read; further reads then draw the
    /// §8.2.2 padding zero bits.
    max_bits: i64,
    /// When `true`, `read_symbol` skips the §8.3 CDF update (mirrors the
    /// frame-header `disable_cdf_update` flag). The §8.2.6 decode step
    /// itself is unaffected.
    disable_cdf_update: bool,
}

impl<'a> SymbolDecoder<'a> {
    /// `init_symbol( sz )` — §8.2.2.
    ///
    /// Initialise the arithmetic decoder over the next `sz` bytes,
    /// reading from `bytes` (which must be positioned at the byte
    /// boundary that begins the partition; the caller slices the OBU
    /// payload so byte 0 of `bytes` is the first partition byte).
    ///
    /// `disable_cdf_update` is the frame-header flag of the same name
    /// (§5.9.2); it governs whether [`Self::read_symbol`] adapts the CDF
    /// after each decode. It does not affect the boolean-coded
    /// [`Self::read_bool`] / [`Self::read_literal`] paths, whose §8.2.3
    /// CDF is reconstructed per call and therefore never adapted (per
    /// the §8.2.3 note).
    pub fn init_symbol(
        bytes: &'a [u8],
        sz: usize,
        disable_cdf_update: bool,
    ) -> Result<Self, Error> {
        let mut reader = BitReader::new(bytes);

        // numBits = Min(sz * 8, 15)
        let num_bits = core::cmp::min(sz.saturating_mul(8), 15) as u32;
        // buf = f(numBits)
        let buf = reader.f(num_bits)? as u32;
        // paddedBuf = buf << (15 - numBits)
        let padded_buf = buf << (15 - num_bits);
        // SymbolValue = ((1 << 15) - 1) ^ paddedBuf
        let value = ((1u32 << 15) - 1) ^ padded_buf;
        // SymbolRange = 1 << 15
        let range = 1u32 << 15;
        // SymbolMaxBits = 8 * sz - 15
        let max_bits = (sz as i64) * 8 - 15;

        Ok(Self {
            reader,
            value,
            range,
            max_bits,
            disable_cdf_update,
        })
    }

    /// `get_position()` — the underlying bit position indicator, exposed
    /// so [`Self::exit_symbol`] callers / tests can reason about the
    /// §8.2.4 trailing-bit accounting.
    pub fn position(&self) -> usize {
        self.reader.position()
    }

    /// `SymbolValue` accessor (§8.2.4 state) — for tests / debugging.
    #[cfg(test)]
    pub(crate) fn symbol_value(&self) -> u32 {
        self.value
    }

    /// `SymbolRange` accessor (§8.2.4 state) — for tests / debugging.
    #[cfg(test)]
    pub(crate) fn symbol_range(&self) -> u32 {
        self.range
    }

    /// `SymbolMaxBits` accessor (§8.2.4 state) — for tests / debugging.
    #[cfg(test)]
    pub(crate) fn symbol_max_bits(&self) -> i64 {
        self.max_bits
    }

    /// `read_symbol( cdf )` — §8.2.6 (symbol decode) followed by the
    /// §8.3 CDF update.
    ///
    /// `cdf` is an array of length `N + 1` giving the cumulative
    /// distribution for a symbol with `N` possible values: `cdf[0..N]`
    /// are the cumulative frequencies (with `cdf[N-1] == 1 << 15` per the
    /// §8.2.6 note), and `cdf[N]` is the adaptation counter (§8.3). The
    /// array is modified in place to adapt to the stream unless
    /// `disable_cdf_update` was set at `init_symbol`.
    ///
    /// Returns the decoded `symbol` in `0..N`.
    pub fn read_symbol(&mut self, cdf: &mut [u16]) -> Result<u32, Error> {
        // N = number of possible symbol values = cdf.len() - 1.
        let n = cdf.len() as u32 - 1;
        debug_assert!(n >= 1, "§8.2.6: N must be greater than 1");

        // §8.2.6 symbol-search loop.
        //   cur = SymbolRange
        //   symbol = -1
        //   do { symbol++; prev = cur;
        //        f = (1<<15) - cdf[symbol]
        //        cur = ((SymbolRange >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT)
        //        cur += EC_MIN_PROB * (N - symbol - 1)
        //   } while ( SymbolValue < cur )
        let mut cur = self.range;
        let mut prev;
        // symbol starts at -1 and is pre-incremented at the loop top, so
        // the first iteration evaluates symbol = 0.
        let mut symbol: i64 = -1;
        loop {
            symbol += 1;
            prev = cur;
            let s = symbol as usize;
            let f = (1u32 << 15) - u32::from(cdf[s]);
            cur = ((self.range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 - EC_PROB_SHIFT);
            cur += EC_MIN_PROB * (n - symbol as u32 - 1);
            if self.value >= cur {
                break;
            }
        }
        let symbol = symbol as u32;

        // SymbolRange = prev - cur
        self.range = prev - cur;
        // SymbolValue = SymbolValue - cur
        self.value -= cur;

        // Renormalisation (§8.2.6 ordered steps 1..7).
        self.renormalize()?;

        // §8.3 CDF update.
        if !self.disable_cdf_update {
            update_cdf(cdf, symbol, n);
        }

        Ok(symbol)
    }

    /// Renormalise `SymbolRange` / `SymbolValue` after a §8.2.6 decode,
    /// reading the new bits from the bitstream (or §8.2.2 padding zeros
    /// once `SymbolMaxBits` is exhausted). The seven ordered steps of
    /// §8.2.6.
    ///
    /// On exit the §8.2.6 invariants checked by
    /// [`Self::check_post_renorm_invariants`] hold; a CDF / partition-
    /// arithmetic bug that drives the state out of range surfaces as
    /// [`Error::SymbolStateInvariantBroken`] instead of leaving the
    /// decoder in an undefined state.
    fn renormalize(&mut self) -> Result<(), Error> {
        // 1. bits = 15 - FloorLog2(SymbolRange)
        let bits = 15 - floor_log2(self.range);
        // 2. SymbolRange = SymbolRange << bits
        self.range <<= bits;
        // 3. numBits = Min(bits, Max(0, SymbolMaxBits))
        let avail = self.max_bits.max(0);
        let num_bits = core::cmp::min(i64::from(bits), avail) as u32;
        // 4. newData = f(numBits)
        let new_data = self.reader.f(num_bits)? as u32;
        // 5. paddedData = newData << (bits - numBits)
        let padded_data = new_data << (bits - num_bits);
        // 6. SymbolValue = paddedData ^ (((SymbolValue + 1) << bits) - 1)
        self.value = padded_data ^ (((self.value + 1) << bits) - 1);
        // 7. SymbolMaxBits = SymbolMaxBits - bits
        self.max_bits -= i64::from(bits);

        self.check_post_renorm_invariants()
    }

    /// Verify the two §8.2.6 post-renormalisation invariants.
    ///
    /// 1. `32768 ≤ SymbolRange ≤ 65535` — the §8.2.6 ordered steps 1
    ///    (`bits = 15 − FloorLog2(SymbolRange)`) and 2
    ///    (`SymbolRange <<= bits`) restore the range to the top half
    ///    of the 16-bit window.
    /// 2. `SymbolValue < SymbolRange` — the §8.2.6 symbol-search loop
    ///    terminates when the accumulator first lands inside the
    ///    selected sub-interval; the subsequent `SymbolValue -= cur` /
    ///    `SymbolRange = prev − cur` rewrite plus the renormalisation
    ///    shift preserve this ordering.
    ///
    /// Both invariants are stated in §8.2.6 of the AV1 Bitstream &
    /// Decoding Process Specification and verified row-for-row across
    /// the 256-symbol cross-implementation oracle in
    /// `docs/video/av1/fixtures/issue_796/msac-trace.md`. A violation
    /// indicates either a malformed bitstream or an internal bug
    /// (CDF misordering, renormalisation off-by-one, partition-
    /// arithmetic precision error, etc.); we surface it as
    /// [`Error::SymbolStateInvariantBroken`] rather than allowing the
    /// decoder to continue from an undefined state.
    fn check_post_renorm_invariants(&self) -> Result<(), Error> {
        const SYMBOL_RANGE_MIN: u32 = 1 << 15; // 32768
        const SYMBOL_RANGE_MAX: u32 = (1 << 16) - 1; // 65535
        if self.range < SYMBOL_RANGE_MIN || self.range > SYMBOL_RANGE_MAX {
            return Err(Error::SymbolStateInvariantBroken);
        }
        if self.value >= self.range {
            return Err(Error::SymbolStateInvariantBroken);
        }
        Ok(())
    }

    /// `read_bool()` — §8.2.3 boolean decode.
    ///
    /// Decodes a pseudo-raw bit assuming equal probability for 0 and 1.
    /// Per §8.2.3 a length-3 CDF `[1 << 14, 1 << 15, 0]` is constructed
    /// and `read_symbol` is invoked on it. The §8.2.3 note observes that
    /// the CDF update can be omitted because the modified values are
    /// never reused — we honour that by feeding a fresh local array
    /// each call, so any adaptation that does happen is discarded.
    pub fn read_bool(&mut self) -> Result<u32, Error> {
        let mut cdf: [u16; 3] = [1 << 14, 1 << 15, 0];
        self.read_symbol(&mut cdf)
    }

    /// `read_literal( n )` — §8.2.5.
    ///
    /// ```text
    ///   x = 0
    ///   for ( i = 0; i < n; i++ ) x = 2 * x + read_bool()
    /// ```
    ///
    /// i.e. an `n`-bit unsigned value, MSB first, with each bit decoded
    /// via the §8.2.3 boolean process. This is the `L(n)` descriptor of
    /// §4.10.8.
    pub fn read_literal(&mut self, n: u32) -> Result<u32, Error> {
        let mut x: u32 = 0;
        for _ in 0..n {
            x = 2 * x + self.read_bool()?;
        }
        Ok(x)
    }

    /// `NS( n )` — §4.10.10. The arithmetic-coded counterpart of `ns(n)`
    /// (§4.10.7): an unsigned non-symmetric integer in `0..n`, with the
    /// underlying bits decoded via `read_literal` (`L(..)`) instead of
    /// the raw `f(..)`.
    ///
    /// ```text
    ///   w = FloorLog2(n) + 1
    ///   m = (1 << w) - n
    ///   v = L(w - 1)
    ///   if (v < m) return v
    ///   extra_bit = L(1)
    ///   return (v << 1) - m + extra_bit
    /// ```
    pub fn read_ns(&mut self, n: u32) -> Result<u32, Error> {
        debug_assert!(n >= 1);
        let w = floor_log2(n) + 1;
        let m = (1u32 << w) - n;
        let v = self.read_literal(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra_bit = self.read_literal(1)?;
        Ok((v << 1) - m + extra_bit)
    }

    /// `decode_subexp_bool( numSyms, k )` — §5.9.28 (bool variant).
    ///
    /// The arithmetic-coded subexponential code used by
    /// `decode_signed_subexp_with_ref_bool` for the warp/global-motion
    /// parameters that live *inside* a tile (as opposed to the §5.9.28
    /// `decode_subexp` read directly from the header bitstream). `L(1)`
    /// for `subexp_more_bools`, `L(b2)` for `subexp_bools`, `NS()` for
    /// the final `subexp_unif_bools`.
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
    pub fn decode_subexp_bool(&mut self, num_syms: u32, k: u32) -> Result<u32, Error> {
        let mut i: u32 = 0;
        let mut mk: u32 = 0;
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1u32 << b2;
            if num_syms <= mk + 3 * a {
                let subexp_unif_bools = self.read_ns(num_syms - mk)?;
                return Ok(subexp_unif_bools + mk);
            }
            let subexp_more_bools = self.read_literal(1)?;
            if subexp_more_bools != 0 {
                i += 1;
                mk += a;
            } else {
                let subexp_bools = self.read_literal(b2)?;
                return Ok(subexp_bools + mk);
            }
        }
    }

    /// `decode_unsigned_subexp_with_ref_bool( mx, k, r )` — av1-spec
    /// p.108. The arithmetic-coded twin of §5.9.27
    /// `decode_unsigned_subexp_with_ref` (which reads `decode_subexp`
    /// directly from the header bitstream). Returns a value in `0..mx`.
    ///
    /// ```text
    ///   v = decode_subexp_bool(mx, k)
    ///   if ( (r << 1) <= mx ) return inverse_recenter(r, v)
    ///   else return mx - 1 - inverse_recenter(mx - 1 - r, v)
    /// ```
    pub fn decode_unsigned_subexp_with_ref_bool(
        &mut self,
        mx: i64,
        k: u32,
        r: i64,
    ) -> Result<i64, Error> {
        // `mx` for the §5.11.58 callers is `Wiener_Taps_Max[j] + 1 -
        // Wiener_Taps_Min[j]` (<= 64) or `Sgrproj_Xqd_Max[i] + 1 -
        // Sgrproj_Xqd_Min[i]` (<= 128), well within u32.
        let v = i64::from(self.decode_subexp_bool(mx as u32, k)?);
        if (r << 1) <= mx {
            Ok(inverse_recenter(r, v))
        } else {
            Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
        }
    }

    /// `decode_signed_subexp_with_ref_bool( low, high, k, r )` —
    /// av1-spec p.108. The arithmetic-coded twin of §5.9.26
    /// `decode_signed_subexp_with_ref`. Returns a value in `low..high`.
    ///
    /// ```text
    ///   x = decode_unsigned_subexp_with_ref_bool(high - low, k, r - low)
    ///   return x + low
    /// ```
    pub fn decode_signed_subexp_with_ref_bool(
        &mut self,
        low: i64,
        high: i64,
        k: u32,
        r: i64,
    ) -> Result<i64, Error> {
        let x = self.decode_unsigned_subexp_with_ref_bool(high - low, k, r - low)?;
        Ok(x + low)
    }

    /// `exit_symbol()` — §8.2.4.
    ///
    /// Consumes the trailing bits of the arithmetic-coded partition and
    /// leaves the bitstream position byte aligned at the start of the
    /// next partition. Returns the `(trailingBitPosition,
    /// paddingEndPosition)` pair the §8.2.4 conformance requirements are
    /// stated in terms of, so callers / tests can assert them against the
    /// original bytes (the conformance check that the trailing bit is 1
    /// and the padding bits are 0 is the *encoder's* obligation, not the
    /// decoder's — we surface the positions rather than enforce them).
    ///
    /// It is a requirement of bitstream conformance that `SymbolMaxBits`
    /// is `>= -14` whenever this is invoked; we treat a violation as an
    /// [`Error::SymbolExitUnderflow`] rather than panicking.
    pub fn exit_symbol(&mut self) -> Result<(usize, usize), Error> {
        // Conformance: SymbolMaxBits >= -14.
        if self.max_bits < -14 {
            return Err(Error::SymbolExitUnderflow);
        }

        // trailingBitPosition = get_position() - Min(15, SymbolMaxBits + 15)
        let pos = self.reader.position() as i64;
        let trailing = pos - core::cmp::min(15, self.max_bits + 15);
        let trailing_bit_position = trailing as usize;

        // Advance the bitstream position by Max(0, SymbolMaxBits).
        let advance = self.max_bits.max(0) as usize;
        // f(advance) would over-read in u64 for advance > 64, so step a
        // bounded amount at a time; the value is discarded.
        let mut remaining = advance;
        while remaining > 0 {
            let chunk = core::cmp::min(remaining, 32) as u32;
            self.reader.f(chunk)?;
            remaining -= chunk as usize;
        }

        // paddingEndPosition = get_position()
        let padding_end_position = self.reader.position();

        Ok((trailing_bit_position, padding_end_position))
    }
}

/// §8.3 CDF update. Adapts `cdf` (length `N + 1`) toward the just-decoded
/// `symbol` using a rate that depends on how many times the CDF has been
/// used (the `cdf[N]` counter).
///
/// ```text
///   rate = 3 + (cdf[N] > 15) + (cdf[N] > 31) + Min(FloorLog2(N), 2)
///   tmp = 0
///   for ( i = 0; i < N - 1; i++ ) {
///     tmp = (i == symbol) ? (1 << 15) : tmp
///     if ( tmp < cdf[i] ) cdf[i] -= (cdf[i] - tmp) >> rate
///     else                cdf[i] += (tmp - cdf[i]) >> rate
///   }
///   cdf[N] += (cdf[N] < 32)
/// ```
pub(crate) fn update_cdf(cdf: &mut [u16], symbol: u32, n: u32) {
    let count = u32::from(cdf[n as usize]);
    let rate = 3 + u32::from(count > 15) + u32::from(count > 31) + core::cmp::min(floor_log2(n), 2);

    let mut tmp: u32 = 0;
    for i in 0..(n - 1) {
        if i == symbol {
            tmp = 1 << 15;
        }
        let c = u32::from(cdf[i as usize]);
        let updated = if tmp < c {
            c - ((c - tmp) >> rate)
        } else {
            c + ((tmp - c) >> rate)
        };
        cdf[i as usize] = updated as u16;
    }
    // cdf[N] += (cdf[N] < 32)
    if count < 32 {
        cdf[n as usize] = (count + 1) as u16;
    }
}

/// Re-export of [`update_cdf`] for the encoder side
/// ([`crate::encoder::symbol_writer::SymbolWriter`]) — the §8.3 update is
/// shared between encode and decode, so both sides apply the exact same
/// adaptation rule to the same CDF arrays.
pub(crate) fn update_cdf_for_encoder(cdf: &mut [u16], symbol: u32, n: u32) {
    update_cdf(cdf, symbol, n);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// §8.2.2 byte-exact initialisation over a full 15-bit window.
    ///
    /// sz = 2 ⇒ numBits = Min(16, 15) = 15. buf = top 15 bits of the two
    /// bytes. With bytes 0b1010_1010 0b1100_1100:
    ///   top 15 bits = 101010101100110 = 0x5566 >> ? — compute directly:
    ///   bits = 1010 1010 1100 110 (15) = 0b101010101100110 = 21862.
    ///   paddedBuf = buf << (15 - 15) = buf = 21862.
    ///   SymbolValue = 0x7FFF ^ 21862 = 32767 ^ 21862 = 10905.
    ///   SymbolRange = 1 << 15 = 32768.
    ///   SymbolMaxBits = 8 * 2 - 15 = 1.
    #[test]
    fn init_symbol_full_window() {
        let bytes = [0b1010_1010u8, 0b1100_1100u8];
        let d = SymbolDecoder::init_symbol(&bytes, 2, false).unwrap();
        let buf: u32 = 0b101010101100110;
        assert_eq!(buf, 21862);
        assert_eq!(d.symbol_value(), 0x7FFF ^ 21862);
        assert_eq!(d.symbol_range(), 1 << 15);
        assert_eq!(d.symbol_max_bits(), 1);
        // 15 bits consumed from the reader.
        assert_eq!(d.position(), 15);
    }

    /// §8.2.2 with sz = 1 ⇒ numBits = Min(8, 15) = 8, so the 8-bit buf is
    /// left-shifted by 7 to fill the 15-bit window.
    #[test]
    fn init_symbol_partial_window_left_shift() {
        let bytes = [0b1011_0001u8];
        let d = SymbolDecoder::init_symbol(&bytes, 1, false).unwrap();
        let buf: u32 = 0b1011_0001; // 177
        let padded = buf << (15 - 8); // 177 << 7 = 22656
        assert_eq!(d.symbol_value(), 0x7FFF ^ padded);
        assert_eq!(d.symbol_range(), 1 << 15);
        assert_eq!(d.symbol_max_bits(), 8 - 15); // 8 * sz - 15 = -7 (sz = 1)
        assert_eq!(d.position(), 8);
    }

    /// Hand-traced single §8.2.6 decode against a uniform binary CDF.
    ///
    /// We use the §8.2.3 boolean CDF [1<<14, 1<<15, 0] (N = 2). We feed
    /// a known 2-byte window and hand-compute the §8.2.6 arithmetic to
    /// assert the decoded bit, the post-decode SymbolValue/SymbolRange,
    /// and the consumed bit position — i.e. a fully byte-exact trace.
    #[test]
    fn read_symbol_boolean_cdf_trace() {
        // sz = 2 ⇒ numBits = 15. Choose bytes so the maths is checkable.
        // bytes = 0b0000_0000 0b0000_0010.
        //   top 15 bits = 000000000000001 = 1. paddedBuf = 1.
        //   SymbolValue = 0x7FFF ^ 1 = 0x7FFE = 32766.
        //   SymbolRange = 32768. SymbolMaxBits = 1.
        let bytes = [0x00u8, 0x02u8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 2, true).unwrap();
        assert_eq!(d.symbol_value(), 0x7FFE);

        // §8.2.6 with cdf = [1<<14, 1<<15, 0], N = 2, SymbolRange=32768.
        //  iter symbol=0: f = (1<<15) - cdf[0] = 32768 - 16384 = 16384.
        //    cur = ((32768>>8) * (16384>>6)) >> (7-6)
        //        = (128 * 256) >> 1 = 32768 >> 1 = 16384.
        //    cur += EC_MIN_PROB * (N - 0 - 1) = 4 * 1 = 4 ⇒ cur = 16388.
        //    SymbolValue (32766) >= cur (16388) ⇒ break, symbol = 0.
        //  prev was SymbolRange = 32768.
        //  SymbolRange = prev - cur = 32768 - 16388 = 16380.
        //  SymbolValue = 32766 - 16388 = 16378.
        let mut cdf: [u16; 3] = [1 << 14, 1 << 15, 0];
        let sym = d.read_symbol(&mut cdf).unwrap();
        assert_eq!(sym, 0);

        // Renormalisation:
        //  bits = 15 - FloorLog2(16380). FloorLog2(16380) = 13 (2^13=8192,
        //    2^14=16384 > 16380) ⇒ bits = 2.
        //  SymbolRange = 16380 << 2 = 65520.
        //  numBits = Min(2, Max(0, 1)) = 1.
        //  newData = f(1). After init we consumed 15 bits; bit 15 is the
        //    16th bit overall = LSB-but-one of byte 1 (0b0000_0010): bit
        //    index 15 ⇒ within byte 1, bit position 7 (the 8th bit of
        //    byte 1) — value 0. So newData = 0.
        //  paddedData = 0 << (2 - 1) = 0.
        //  SymbolValue = 0 ^ (((16378 + 1) << 2) - 1)
        //              = ((16379 << 2) - 1) = 65516 - 1 = 65515.
        //  SymbolMaxBits = 1 - 2 = -1.
        assert_eq!(d.symbol_range(), 65520);
        assert_eq!(d.symbol_value(), 65515);
        assert_eq!(d.symbol_max_bits(), -1);
        // 15 (init) + 1 (renorm) = 16 bits consumed.
        assert_eq!(d.position(), 16);
    }

    /// `read_bool` is `read_symbol` over the §8.2.3 fixed CDF; it must
    /// agree with the hand-traced decode above and never adapt across
    /// calls (a fresh CDF each call).
    #[test]
    fn read_bool_matches_boolean_trace() {
        let bytes = [0x00u8, 0x02u8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 2, false).unwrap();
        let bit = d.read_bool().unwrap();
        assert_eq!(bit, 0);
        // Even with disable_cdf_update=false the boolean path is unaffected
        // because each read_bool builds a throwaway CDF.
        assert_eq!(d.symbol_range(), 65520);
        assert_eq!(d.symbol_value(), 65515);
    }

    /// `read_literal(n)` builds an n-bit value MSB first from n bool
    /// decodes. We assert it equals the documented composition.
    #[test]
    fn read_literal_composes_bits_msb_first() {
        // Use a window that decodes to a known multi-bit literal. We
        // cross-check read_literal(3) against three sequential read_bool
        // calls on an identical decoder.
        let bytes = [0x12u8, 0x34u8, 0x56u8, 0x78u8];

        let mut d_lit = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        let lit = d_lit.read_literal(3).unwrap();

        let mut d_bits = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        let b0 = d_bits.read_bool().unwrap();
        let b1 = d_bits.read_bool().unwrap();
        let b2 = d_bits.read_bool().unwrap();
        let composed = (b0 << 2) | (b1 << 1) | b2;

        assert_eq!(lit, composed);
        // Both decoders must be in identical state afterwards.
        assert_eq!(d_lit.symbol_value(), d_bits.symbol_value());
        assert_eq!(d_lit.symbol_range(), d_bits.symbol_range());
        assert_eq!(d_lit.position(), d_bits.position());
    }

    /// §8.3 CDF update applied to a 3-symbol CDF (N = 3) with a known
    /// starting distribution and a chosen symbol, checked term by term.
    #[test]
    fn update_cdf_adapts_toward_symbol() {
        // N = 3 ⇒ cdf length 4. Start: cdf = [10000, 20000, 32768, 0].
        // count = cdf[N] = cdf[3] = 0.
        // rate = 3 + (0>15) + (0>31) + Min(FloorLog2(3), 2)
        //      = 3 + 0 + 0 + Min(1, 2) = 4.
        // symbol = 1.
        // loop i in 0..N-1 = 0..2:
        //  i=0: tmp = (0==1)?... ⇒ tmp stays 0. c = 10000.
        //       tmp(0) < c(10000) ⇒ cdf[0] = 10000 - ((10000-0)>>4)
        //       = 10000 - 625 = 9375.
        //  i=1: tmp = (1==1) ⇒ tmp = 32768. c = 20000.
        //       tmp(32768) >= c(20000) ⇒ cdf[1] = 20000 + ((32768-20000)>>4)
        //       = 20000 + (12768>>4) = 20000 + 798 = 20798.
        // cdf[N]: count<32 ⇒ cdf[3] = 1.
        let mut cdf: [u16; 4] = [10000, 20000, 32768, 0];
        update_cdf(&mut cdf, 1, 3);
        assert_eq!(cdf, [9375, 20798, 32768, 1]);
    }

    /// The §8.3 rate increases the count up to the cap of 32 and the
    /// `cdf[N] > 15` / `> 31` rate bumps fire at the documented points.
    #[test]
    fn update_cdf_counter_caps_at_32() {
        let mut cdf: [u16; 3] = [16384, 32768, 31];
        update_cdf(&mut cdf, 0, 2);
        assert_eq!(cdf[2], 32); // 31 < 32 ⇒ +1.
                                // Already at 32: stays 32.
        let mut cdf2: [u16; 3] = [16384, 32768, 32];
        update_cdf(&mut cdf2, 0, 2);
        assert_eq!(cdf2[2], 32);
    }

    /// Adaptive multisymbol decode: decode the *same* CDF twice with
    /// updates enabled and confirm the CDF mutated between calls (the
    /// §8.3 adaptation actually runs through `read_symbol`), and that with
    /// `disable_cdf_update` the CDF is left untouched.
    #[test]
    fn read_symbol_adapts_cdf_when_enabled() {
        let bytes = [0x80u8, 0x00u8, 0x00u8, 0x00u8];

        let mut d = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut cdf: [u16; 4] = [10923, 21845, 32768, 0];
        let before = cdf;
        let _ = d.read_symbol(&mut cdf).unwrap();
        assert_ne!(cdf, before, "§8.3 update should have mutated the CDF");
        assert_eq!(cdf[3], 1, "counter incremented");

        let mut d2 = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        let mut cdf2: [u16; 4] = [10923, 21845, 32768, 0];
        let before2 = cdf2;
        let _ = d2.read_symbol(&mut cdf2).unwrap();
        assert_eq!(cdf2, before2, "disable_cdf_update must leave CDF intact");
    }

    /// `read_ns` (NS(n), §4.10.10) mirrors the §4.10.7 ns(n) value table
    /// but over the arithmetic-coded literal path. For n = 1, w = 1 ⇒
    /// L(0) returns 0 ⇒ returns 0 with no bits consumed.
    #[test]
    fn read_ns_n_equals_1_consumes_no_arithmetic_bits() {
        let bytes = [0xFFu8, 0x00u8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 2, true).unwrap();
        let pos_before = d.position();
        let v = d.read_ns(1).unwrap();
        assert_eq!(v, 0);
        // L(0) is read_literal(0) which decodes zero bools ⇒ no bitstream
        // movement.
        assert_eq!(d.position(), pos_before);
    }

    /// `decode_subexp_bool` with a small `numSyms` that hits the first-
    /// iteration uniform branch immediately. With k = 0, b2 = 0, a = 1,
    /// 3*a = 3; numSyms = 2 <= mk(0) + 3 ⇒ NS(2) decoded directly.
    #[test]
    fn decode_subexp_bool_immediate_uniform_branch() {
        let bytes = [0x00u8, 0x00u8, 0x00u8, 0x00u8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        // NS(2): w = FloorLog2(2)+1 = 2, m = (1<<2) - 2 = 2, v = L(1).
        // Whatever L(1) decodes, v < m? v is 0 or 1 < 2 ⇒ return v.
        // So result equals a single L(1) read.
        let result = d.decode_subexp_bool(2, 0).unwrap();
        assert!(result < 2);

        // Cross-check against a fresh decoder doing NS(2) directly.
        let mut d2 = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        let ns2 = d2.read_ns(2).unwrap();
        assert_eq!(result, ns2);
        assert_eq!(d.position(), d2.position());
    }

    /// §8.2.4 exit accounting. After init with sz = 2 and no symbol
    /// decoding, SymbolMaxBits = 1; exit advances Max(0, 1) = 1 bit and
    /// leaves the position byte aligned at 16.
    #[test]
    fn exit_symbol_advances_to_byte_boundary() {
        let bytes = [0x80u8, 0x01u8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 2, true).unwrap();
        // Position after init = 15; SymbolMaxBits = 1.
        assert_eq!(d.position(), 15);
        let (trailing, padding_end) = d.exit_symbol().unwrap();
        // trailingBitPosition = 15 - Min(15, 1 + 15) = 15 - 15 = 0.
        assert_eq!(trailing, 0);
        // Advance by Max(0, 1) = 1 ⇒ position 16. paddingEnd = 16, a
        // multiple of 8 (byte aligned) per the §8.2.4 note.
        assert_eq!(padding_end, 16);
        assert_eq!(d.position(), 16);
        assert_eq!(padding_end % 8, 0);
    }

    /// §8.2.4 conformance gate: SymbolMaxBits < -14 is rejected. We drive
    /// SymbolMaxBits below -14 by repeated decodes on a tiny partition,
    /// then assert exit_symbol surfaces the error rather than panicking.
    #[test]
    fn exit_symbol_rejects_underflow() {
        // sz = 1 ⇒ SymbolMaxBits starts at -7. A few boolean decodes will
        // each subtract `bits` (>= 1), pushing it below -14.
        let bytes = [0x55u8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 1, true).unwrap();
        // Decode bools until SymbolMaxBits drops below -14.
        for _ in 0..16 {
            if d.symbol_max_bits() < -14 {
                break;
            }
            // Past the buffer end the renorm reads padding zeros only
            // (numBits clamps to 0), so this never errors on input.
            let _ = d.read_bool().unwrap();
        }
        assert!(d.symbol_max_bits() < -14);
        assert_eq!(d.exit_symbol().unwrap_err(), Error::SymbolExitUnderflow);
    }

    /// Decoding past the end of the byte buffer must succeed using the
    /// §8.2.2 padding zero bits (numBits clamps to 0 once SymbolMaxBits
    /// is exhausted) rather than returning UnexpectedEnd.
    #[test]
    fn decode_past_buffer_uses_padding_zeros() {
        let bytes = [0xFFu8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 1, true).unwrap();
        // sz=1 ⇒ SymbolMaxBits starts at -7 already (every renorm bit is
        // padding). Several decodes must all succeed without touching the
        // (already drained) reader past its 8 real bits.
        for _ in 0..8 {
            let mut cdf: [u16; 3] = [1 << 14, 1 << 15, 0];
            let r = d.read_symbol(&mut cdf);
            assert!(r.is_ok());
        }
    }

    /// §8.2.6 post-renormalisation invariant #1: `32768 ≤ SymbolRange ≤
    /// 65535`. We sweep a few different starting byte windows + a
    /// boolean-CDF mix and confirm the range stays in the top half of
    /// the 16-bit window after every decode.
    #[test]
    fn post_renorm_range_in_top_half_across_decodes() {
        let starts: [[u8; 4]; 4] = [
            [0x00, 0x00, 0x00, 0x00],
            [0xFF, 0xFF, 0xFF, 0xFF],
            [0x80, 0x01, 0x42, 0x37],
            [0x55, 0xAA, 0x33, 0xCC],
        ];
        for bytes in &starts {
            let mut d = SymbolDecoder::init_symbol(bytes, bytes.len(), true).unwrap();
            for _ in 0..16 {
                let mut cdf: [u16; 3] = [1 << 14, 1 << 15, 0];
                d.read_symbol(&mut cdf).unwrap();
                let r = d.symbol_range();
                assert!(
                    (1u32 << 15..=u32::from(u16::MAX)).contains(&r),
                    "§8.2.6 invariant #1 broken: SymbolRange = {r}"
                );
            }
        }
    }

    /// §8.2.6 post-renormalisation invariant #2: `SymbolValue <
    /// SymbolRange`. Same sweep as above, asserting the ordering on
    /// every step.
    #[test]
    fn post_renorm_value_below_range_across_decodes() {
        let starts: [[u8; 4]; 4] = [
            [0x00, 0x00, 0x00, 0x00],
            [0xFF, 0xFF, 0xFF, 0xFF],
            [0x80, 0x01, 0x42, 0x37],
            [0x55, 0xAA, 0x33, 0xCC],
        ];
        for bytes in &starts {
            let mut d = SymbolDecoder::init_symbol(bytes, bytes.len(), true).unwrap();
            for _ in 0..16 {
                let mut cdf: [u16; 3] = [1 << 14, 1 << 15, 0];
                d.read_symbol(&mut cdf).unwrap();
                assert!(
                    d.symbol_value() < d.symbol_range(),
                    "§8.2.6 invariant #2 broken: SymbolValue={}, SymbolRange={}",
                    d.symbol_value(),
                    d.symbol_range()
                );
            }
        }
    }

    /// Direct unit test of [`SymbolDecoder::check_post_renorm_invariants`]
    /// on a hand-constructed in-range state and on each of the three
    /// out-of-range edges (`range < 32768`, `range > 65535`,
    /// `value >= range`).
    #[test]
    fn invariant_check_returns_error_on_each_violation() {
        // In-range: 32768 <= range, value < range.
        let bytes = [0x00u8, 0x00u8];
        let mut d = SymbolDecoder::init_symbol(&bytes, 2, true).unwrap();
        d.value = 1;
        d.range = 1 << 15;
        assert!(d.check_post_renorm_invariants().is_ok());

        // Boundary OK: range = 65535, value = 65534.
        d.value = 65534;
        d.range = 65535;
        assert!(d.check_post_renorm_invariants().is_ok());

        // range below 32768.
        d.value = 0;
        d.range = (1 << 15) - 1;
        assert_eq!(
            d.check_post_renorm_invariants().unwrap_err(),
            Error::SymbolStateInvariantBroken
        );

        // range above 65535.
        d.value = 0;
        d.range = (1 << 16) + 1;
        assert_eq!(
            d.check_post_renorm_invariants().unwrap_err(),
            Error::SymbolStateInvariantBroken
        );

        // value >= range.
        d.value = 1 << 15;
        d.range = 1 << 15;
        assert_eq!(
            d.check_post_renorm_invariants().unwrap_err(),
            Error::SymbolStateInvariantBroken
        );
    }

    /// The §8.2.6 invariants must hold across the *full* sequence of a
    /// non-boolean adaptive CDF decode (4-symbol CDF with §8.3 update
    /// enabled). This is the closest reproducible analogue of the
    /// `docs/video/av1/fixtures/issue_796/msac-trace.md` "Invariants
    /// observed across all 256 rows" assertion that we can run in-tree
    /// (without the issue_796 bitstream).
    #[test]
    fn invariants_hold_across_adaptive_multisymbol_run() {
        let bytes: [u8; 16] = [
            0x9F, 0x12, 0x37, 0xAB, 0x4C, 0xDE, 0x5F, 0x68, 0x71, 0x82, 0x93, 0xA4, 0xB5, 0xC6,
            0xD7, 0xE8,
        ];
        let mut d = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        // 4-symbol CDF (N = 4); cdf[3] = 1 << 15 per §8.2.6 contract.
        let mut cdf: [u16; 5] = [8192, 16384, 24576, 32768, 0];
        for _ in 0..40 {
            d.read_symbol(&mut cdf).unwrap();
            let r = d.symbol_range();
            let v = d.symbol_value();
            assert!(
                (1u32 << 15..=u32::from(u16::MAX)).contains(&r),
                "invariant #1 broken: SymbolRange = {r}"
            );
            assert!(
                v < r,
                "invariant #2 broken: SymbolValue={v} SymbolRange={r}"
            );
        }
    }
}
