//! MSB-first bit reader used by every syntax element described with
//! the descriptors in §4.10 of the AV1 Bitstream & Decoding Process
//! Specification.
//!
//! The reader is intentionally minimal: it walks a borrowed byte slice
//! one bit at a time, reading the most-significant bit of the current
//! byte first per §8.1 (`read_bit`). Each successful read advances the
//! bit position; reads past the end of the underlying buffer fail with
//! [`Error::UnexpectedEnd`].
//!
//! The primitives needed by the rounds shipped so far:
//!
//!   * [`BitReader::f`] — `f(n)` per §4.10.2 / §8.1
//!   * [`BitReader::uvlc`] — `uvlc()` per §4.10.3
//!   * [`BitReader::su`] — `su(n)` per §4.10.6
//!
//! `leb128()` (§4.10.5) is byte-aligned and already implemented in
//! [`crate::obu::parse_leb128`] against the raw payload slice.

use crate::Error;

/// MSB-first bit reader.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    /// Total bit position from the start of `bytes`, in bits.
    pos: usize,
}

impl<'a> BitReader<'a> {
    /// Wrap a byte slice. Position starts at bit 0 (MSB of byte 0).
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Current absolute bit position from the start of the buffer
    /// (i.e. the AV1 `get_position()` value, modulo the OBU header /
    /// size-field bytes the caller stripped).
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    /// Read the next bit per §8.1 `read_bit()`.
    fn read_bit(&mut self) -> Result<u8, Error> {
        let byte_idx = self.pos >> 3;
        let bit_idx = 7 - (self.pos & 0x7); // MSB-first
        let byte = *self.bytes.get(byte_idx).ok_or(Error::UnexpectedEnd)?;
        self.pos += 1;
        Ok((byte >> bit_idx) & 0x1)
    }

    /// `f(n)` per §4.10.2 / §8.1: unsigned `n`-bit number, MSB first.
    ///
    /// `n` may be 0..=64. Returns 0 when `n == 0` per the §8.1 loop
    /// (`x = 0` and the body is skipped).
    pub(crate) fn f(&mut self, n: u32) -> Result<u64, Error> {
        debug_assert!(n <= 64);
        let mut x: u64 = 0;
        for _ in 0..n {
            x = (x << 1) | u64::from(self.read_bit()?);
        }
        Ok(x)
    }

    /// `su(n)` per §4.10.6 — signed integer reconstructed from the
    /// bottom `n` bits of a two's-complement integer. The high bit of
    /// the read `n`-bit value is the sign bit.
    ///
    /// `n` must be in `1..=32`. The result fits in `i32`.
    pub(crate) fn su(&mut self, n: u32) -> Result<i32, Error> {
        debug_assert!((1..=32).contains(&n));
        let value = self.f(n)? as u32;
        let sign_mask: u32 = 1 << (n - 1);
        if value & sign_mask != 0 {
            // value - 2 * signMask reinterpreted in i64 to avoid wrap.
            let v = i64::from(value) - (i64::from(sign_mask) << 1);
            Ok(v as i32)
        } else {
            Ok(value as i32)
        }
    }

    /// `uvlc()` per §4.10.3.
    ///
    /// The spec caps the returned value at `(1 << 32) - 1` (the loop
    /// returns this sentinel when `leadingZeros >= 32`). We propagate
    /// that sentinel as-is.
    pub(crate) fn uvlc(&mut self) -> Result<u32, Error> {
        let mut leading_zeros: u32 = 0;
        loop {
            let done = self.f(1)?;
            if done == 1 {
                break;
            }
            leading_zeros += 1;
            if leading_zeros >= 32 {
                // §4.10.3: return (1 << 32) - 1 unconditionally.
                return Ok(u32::MAX);
            }
        }
        let value = self.f(leading_zeros)?;
        // §4.10.3: return value + (1 << leadingZeros) - 1. All terms
        // fit comfortably in u64 (leading_zeros < 32, value < 2^32).
        let sum = value + ((1u64) << leading_zeros) - 1;
        // The sum cannot exceed (1 << 32) - 1 + (1 << 31) - 1 < 2^33,
        // so the only over-u32 case is one we've already short-circuited
        // above. Cast is safe.
        Ok(u32::try_from(sum.min(u64::from(u32::MAX))).expect("clamped"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f1_walks_bits_msb_first() {
        // 0b1010_1100 = 0xAC.
        let mut br = BitReader::new(&[0xAC]);
        assert_eq!(br.f(1).unwrap(), 1);
        assert_eq!(br.f(1).unwrap(), 0);
        assert_eq!(br.f(1).unwrap(), 1);
        assert_eq!(br.f(1).unwrap(), 0);
        assert_eq!(br.f(1).unwrap(), 1);
        assert_eq!(br.f(1).unwrap(), 1);
        assert_eq!(br.f(1).unwrap(), 0);
        assert_eq!(br.f(1).unwrap(), 0);
        assert_eq!(br.position(), 8);
    }

    #[test]
    fn fn_reads_multi_bit_field() {
        // 0b1010_1100 0b0101_0011 — f(12) over the whole = 0xACA.
        let mut br = BitReader::new(&[0xAC, 0x53]);
        let v = br.f(12).unwrap();
        // First 12 bits = 1010_1100 0101 = 0xAC5.
        assert_eq!(v, 0xAC5);
        assert_eq!(br.position(), 12);
    }

    #[test]
    fn f_n_equals_zero_returns_zero_no_advance() {
        let mut br = BitReader::new(&[0xFF]);
        assert_eq!(br.f(0).unwrap(), 0);
        assert_eq!(br.position(), 0);
    }

    #[test]
    fn f_returns_unexpected_end_past_buffer() {
        let mut br = BitReader::new(&[0x80]);
        // Drain the byte then try one more bit.
        let _ = br.f(8).unwrap();
        let err = br.f(1).unwrap_err();
        assert_eq!(err, Error::UnexpectedEnd);
    }

    #[test]
    fn uvlc_decodes_zero() {
        // leadingZeros=0, done=1 — single bit "1" encodes 0.
        let mut br = BitReader::new(&[0b1000_0000]);
        assert_eq!(br.uvlc().unwrap(), 0);
        assert_eq!(br.position(), 1);
    }

    #[test]
    fn uvlc_decodes_one_two_three() {
        // value=1: bits "0 1 0" = 0b010_..., 3 bits. (lz=1, value=0,
        // return 0 + 2 - 1 = 1).
        let mut br = BitReader::new(&[0b0100_0000]);
        assert_eq!(br.uvlc().unwrap(), 1);
        assert_eq!(br.position(), 3);
        // value=2: bits "0 1 1" - lz=1, value=1, return 1+1=2.
        let mut br = BitReader::new(&[0b0110_0000]);
        assert_eq!(br.uvlc().unwrap(), 2);
        // value=3: bits "0 0 1 0 0" - lz=2, value=0, return 0+3=3.
        let mut br = BitReader::new(&[0b0010_0000]);
        assert_eq!(br.uvlc().unwrap(), 3);
        assert_eq!(br.position(), 5);
    }

    #[test]
    fn su_positive_value() {
        // su(7) reading 0010101 = 21 (positive — top bit of 7 is 0).
        // Encode in one byte high nibble: 0010_1010 = 0x2A, position
        // 0..6 spans the high seven bits = 0010_101 = 21.
        let mut br = BitReader::new(&[0b0010_1010]);
        assert_eq!(br.su(7).unwrap(), 21);
        assert_eq!(br.position(), 7);
    }

    #[test]
    fn su_negative_value() {
        // su(7) reading 1111111 = -1 (signMask bit set, value=127,
        // 127 - 128 = -1).
        let mut br = BitReader::new(&[0b1111_1110]);
        assert_eq!(br.su(7).unwrap(), -1);
    }

    #[test]
    fn su_min_negative() {
        // su(7) reading 1000000 = signMask only ⇒ 64 - 128 = -64.
        let mut br = BitReader::new(&[0b1000_0000]);
        assert_eq!(br.su(7).unwrap(), -64);
    }

    #[test]
    fn uvlc_returns_sentinel_on_32_leading_zeros() {
        // 32 zero bits followed by anything must yield u32::MAX per
        // §4.10.3 short-circuit. 32 / 8 = 4 zero bytes is enough.
        let mut br = BitReader::new(&[0x00, 0x00, 0x00, 0x00, 0x80]);
        assert_eq!(br.uvlc().unwrap(), u32::MAX);
    }
}
