//! MSB-first bit writer — encoder counterpart to
//! [`crate::bitreader::BitReader`].
//!
//! Each `write_bit(b)` appends `b` as the next most-significant
//! unwritten bit in the current byte (the inverse of §8.1
//! `read_bit()`). When 8 bits have been buffered the byte is pushed
//! onto the output buffer. `finish()` flushes a trailing partial
//! byte with zero-padding in the low bits (this matches the parser's
//! contract — every reader stops at a known bit position and the
//! padding bits are unread).
//!
//! Primitives needed by the encoder rounds shipped so far:
//!
//!   * [`BitWriter::write_bit`] — single bit.
//!   * [`BitWriter::write_bits`] — `n`-bit unsigned, MSB first
//!     (inverse of §4.10.2 `f(n)` / §8.1).
//!   * [`BitWriter::write_leb128`] — byte-aligned variable-length
//!     value per §4.10.5 (inverse of
//!     [`crate::obu::parse_leb128`]).
//!   * [`BitWriter::byte_align`] — pad to the next byte boundary
//!     with `0` bits. Used by the §5.3.4 `trailing_bits()` / OBU
//!     framing.
//!
//! `write_uvlc` / `write_su` / `write_ns` will land alongside the
//! frame-header writer in the next arc.

/// MSB-first bit writer.
#[derive(Debug, Default, Clone)]
pub struct BitWriter {
    /// Bytes already flushed (every fully-buffered byte ends up
    /// here; the current partial byte lives in `cur` until either
    /// 8 bits are buffered or [`Self::finish`] runs).
    bytes: Vec<u8>,
    /// Partial byte being assembled, packed MSB-first.
    cur: u8,
    /// Number of bits in `cur` already populated (0..=7).
    bits_in_cur: u32,
}

impl BitWriter {
    /// Construct an empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total number of bits written so far.
    pub fn bit_position(&self) -> usize {
        self.bytes.len() * 8 + self.bits_in_cur as usize
    }

    /// True iff the next write would be on a byte boundary.
    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_cur == 0
    }

    /// Append `b` (the low bit of `b` is the value) as the next
    /// MSB-first bit. `b` may carry bits other than the low one;
    /// those are discarded.
    pub fn write_bit(&mut self, b: u8) {
        let bit = b & 0x1;
        // Position 7..0 from the MSB toward the LSB.
        let shift = 7 - self.bits_in_cur;
        self.cur |= bit << shift;
        self.bits_in_cur += 1;
        if self.bits_in_cur == 8 {
            self.bytes.push(self.cur);
            self.cur = 0;
            self.bits_in_cur = 0;
        }
    }

    /// Write `n` bits of `value` (the low `n` bits of `value`),
    /// MSB-first. `n` must be in `0..=64`. `n == 0` is a no-op.
    /// This is the inverse of [`crate::bitreader::BitReader::f`].
    pub fn write_bits(&mut self, n: u32, value: u64) {
        debug_assert!(n <= 64, "write_bits supports up to 64 bits");
        if n == 0 {
            return;
        }
        // Emit bit (n - 1) first (the MSB of the low n bits).
        for i in (0..n).rev() {
            let bit = ((value >> i) & 0x1) as u8;
            self.write_bit(bit);
        }
    }

    /// Pad the current byte with `0` bits until the next byte
    /// boundary. No-op when already aligned.
    pub fn byte_align(&mut self) {
        if self.bits_in_cur == 0 {
            return;
        }
        // Padding bits stay 0 — `cur` was initialised to 0 and
        // `write_bit` only ORs into populated slots.
        self.bytes.push(self.cur);
        self.cur = 0;
        self.bits_in_cur = 0;
    }

    /// Write a `leb128()` value per §4.10.5. The encoding emits
    /// seven value-bits per byte (low 7 bits) with the high bit of
    /// each non-terminal byte set to 1; the terminal byte's high
    /// bit is 0. Conformance cap is `(1 << 32) - 1` per §4.10.5;
    /// values up to that cap are accepted (debug-asserted).
    ///
    /// The output is byte-aligned: the caller must ensure the
    /// writer is on a byte boundary before invoking. (`leb128()`
    /// is a byte-aligned primitive per §4.10.5.)
    pub fn write_leb128(&mut self, value: u64) {
        debug_assert!(
            value <= u64::from(u32::MAX),
            "§4.10.5 caps leb128 values at (1 << 32) - 1"
        );
        debug_assert!(
            self.is_byte_aligned(),
            "§4.10.5 leb128() is byte-aligned; pad first"
        );
        let mut v = value;
        loop {
            let chunk = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                // Terminal byte: high bit 0.
                self.bytes.push(chunk);
                return;
            }
            // Non-terminal byte: high bit 1.
            self.bytes.push(chunk | 0x80);
        }
    }

    /// Number of bytes the encoded `leb128(value)` would consume.
    /// Useful for the §5.3.1 size-field-self-counts dance (the OBU
    /// writer reserves space for `obu_size` after computing the
    /// payload length).
    pub fn leb128_size(value: u64) -> usize {
        debug_assert!(value <= u64::from(u32::MAX));
        let mut v = value;
        let mut n = 1;
        while v >= 0x80 {
            v >>= 7;
            n += 1;
        }
        n
    }

    /// Consume the writer and return the assembled bytes. A
    /// trailing partial byte (if any) is padded with `0` bits in
    /// its low bits and emitted; the result is therefore always
    /// `ceil(bit_position / 8)` bytes long at the call site.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_cur != 0 {
            self.bytes.push(self.cur);
        }
        self.bytes
    }

    /// Borrow the bytes flushed so far without consuming the
    /// writer. Excludes any in-progress partial byte. Useful for
    /// the OBU writer when it needs the byte count of an aligned
    /// region.
    pub fn flushed_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_bit_packs_msb_first() {
        // 0b1010_1100 = 0xAC.
        let mut bw = BitWriter::new();
        for bit in [1, 0, 1, 0, 1, 1, 0, 0] {
            bw.write_bit(bit);
        }
        assert_eq!(bw.bit_position(), 8);
        assert_eq!(bw.finish(), vec![0xAC]);
    }

    #[test]
    fn write_bits_round_trips_through_reader() {
        // Pack value=0xAC5 at width 12, then read it back through
        // the reader to confirm parity with §4.10.2 / §8.1.
        let mut bw = BitWriter::new();
        bw.write_bits(12, 0xAC5);
        // Bit position is 12 (mid-byte).
        assert_eq!(bw.bit_position(), 12);
        let bytes = bw.finish();
        // First 12 bits are 1010_1100_0101; with 4 trailing zero
        // pad bits => 1010_1100_0101_0000 = 0xAC50 = [0xAC, 0x50].
        assert_eq!(bytes, vec![0xAC, 0x50]);
        // Round-trip through the reader.
        let mut br = crate::bitreader::BitReader::new(&bytes);
        assert_eq!(br.f(12).unwrap(), 0xAC5);
    }

    #[test]
    fn write_bits_n_zero_is_noop() {
        let mut bw = BitWriter::new();
        bw.write_bits(0, 0xFFFF_FFFF);
        assert_eq!(bw.bit_position(), 0);
        assert!(bw.is_byte_aligned());
        assert!(bw.finish().is_empty());
    }

    #[test]
    fn byte_align_pads_with_zeros() {
        let mut bw = BitWriter::new();
        bw.write_bits(3, 0b101);
        assert!(!bw.is_byte_aligned());
        bw.byte_align();
        assert!(bw.is_byte_aligned());
        // 3 bits of 0b101 + 5 zero-pad bits => 0b1010_0000 = 0xA0.
        assert_eq!(bw.finish(), vec![0xA0]);
    }

    #[test]
    fn byte_align_when_aligned_is_noop() {
        let mut bw = BitWriter::new();
        bw.write_bits(8, 0xCD);
        assert!(bw.is_byte_aligned());
        bw.byte_align();
        assert_eq!(bw.finish(), vec![0xCD]);
    }

    #[test]
    fn leb128_single_byte_zero() {
        let mut bw = BitWriter::new();
        bw.write_leb128(0);
        assert_eq!(bw.finish(), vec![0x00]);
        assert_eq!(BitWriter::leb128_size(0), 1);
    }

    #[test]
    fn leb128_two_byte_value_matches_parser() {
        // Parser test asserts (0x81, 0x01) decodes to 129.
        let mut bw = BitWriter::new();
        bw.write_leb128(129);
        assert_eq!(bw.finish(), vec![0x81, 0x01]);
        assert_eq!(BitWriter::leb128_size(129), 2);
        // And the parser agrees:
        let (v, n) = crate::obu::parse_leb128(&[0x81, 0x01]).unwrap();
        assert_eq!((v, n), (129, 2));
    }

    #[test]
    fn leb128_round_trip_against_parser() {
        // Cover a range of values that exercise 1..=5 byte encodings.
        // 1 byte: 0..=0x7f. 2 byte: up to 0x3fff. 3 byte: up to
        // 0x1f_ffff. 4 byte: up to 0x0fff_ffff. 5 byte: up to
        // u32::MAX (the §4.10.5 cap).
        for value in [
            0u64,
            1,
            127,
            128,
            0x3fff,
            0x4000,
            1_000_000,
            u64::from(u32::MAX),
        ] {
            let mut bw = BitWriter::new();
            bw.write_leb128(value);
            let bytes = bw.finish();
            assert_eq!(bytes.len(), BitWriter::leb128_size(value));
            let (decoded, consumed) = crate::obu::parse_leb128(&bytes).unwrap();
            assert_eq!(decoded, value, "round-trip mismatch for {value}");
            assert_eq!(consumed, bytes.len());
        }
    }

    #[test]
    fn finish_pads_partial_trailing_byte() {
        let mut bw = BitWriter::new();
        bw.write_bits(4, 0b1011);
        // Without explicit byte_align, finish() should still pad.
        assert_eq!(bw.finish(), vec![0b1011_0000]);
    }

    #[test]
    fn write_64_bits_round_trips() {
        let mut bw = BitWriter::new();
        bw.write_bits(64, 0xDEAD_BEEF_CAFE_BABE);
        let bytes = bw.finish();
        let mut br = crate::bitreader::BitReader::new(&bytes);
        assert_eq!(br.f(64).unwrap(), 0xDEAD_BEEF_CAFE_BABE);
    }
}
