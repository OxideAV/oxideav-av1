//! AV1 bitstream writer — inverse of [`crate::bitreader::BitReader`].
//!
//! AV1 (AOMedia Video 1, AOM Bitstream & Decoding Process Specification 2019)
//! writes bits MSB-first inside each byte. This module mirrors the
//! primitive writers used by the encoder side:
//!
//! * `f(n, v)`  — write an unsigned `n`-bit field, big-endian. §4.10.2 inverse.
//! * `su(n, v)` — write a signed two's-complement `n`-bit field. §4.10.5 inverse.
//! * `uvlc(v)`  — write a universal variable-length code. §4.10.3 inverse.
//! * `leb128(v)`— write a little-endian variable-length unsigned integer
//!                with continuation bit `0x80`. §4.10.5 inverse.
//!
//! A `BitWriter` accumulates bits into a `Vec<u8>` and exposes
//! `byte_align(pad_with_one_zero=…)` so callers can implement the
//! `byte_alignment()` and `trailing_bits()` syntactic anchors.

/// MSB-first bit writer over an owned `Vec<u8>`.
#[derive(Default, Clone)]
pub struct BitWriter {
    /// Completed bytes.
    out: Vec<u8>,
    /// Partial byte, MSB-first. Bits in slots `[7-(bits_in_acc-1)..=7]` are
    /// the unwritten bits — i.e. `acc` is left-aligned to its valid count.
    acc: u8,
    /// Count of valid bits in `acc` (`0..=7`).
    bits_in_acc: u32,
}

impl BitWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bits already written, including any partial-byte contribution.
    pub fn bit_position(&self) -> u64 {
        (self.out.len() as u64) * 8 + self.bits_in_acc as u64
    }

    pub fn is_byte_aligned(&self) -> bool {
        self.bits_in_acc == 0
    }

    /// Consume the writer and return the accumulated bytes. Any partial
    /// byte is flushed with zero padding in the trailing bit positions.
    pub fn finish(mut self) -> Vec<u8> {
        if self.bits_in_acc > 0 {
            self.out.push(self.acc);
            self.acc = 0;
            self.bits_in_acc = 0;
        }
        self.out
    }

    /// Read-only view of the bytes written so far. Does NOT include the
    /// partial-byte accumulator.
    pub fn whole_bytes(&self) -> &[u8] {
        &self.out
    }

    /// Number of bytes the resulting buffer would have if `finish()` were
    /// called now (i.e. completed bytes + 1 if a partial byte is buffered).
    pub fn byte_len_after_flush(&self) -> usize {
        self.out.len() + if self.bits_in_acc > 0 { 1 } else { 0 }
    }

    /// Write a single bit (`0` or `1`).
    #[inline]
    pub fn bit(&mut self, b: bool) {
        let v = b as u8;
        if self.bits_in_acc == 0 {
            self.acc = v << 7;
            self.bits_in_acc = 1;
        } else {
            self.acc |= v << (7 - self.bits_in_acc);
            self.bits_in_acc += 1;
        }
        if self.bits_in_acc == 8 {
            self.out.push(self.acc);
            self.acc = 0;
            self.bits_in_acc = 0;
        }
    }

    /// `f(n, v)` — write `n` low-order bits of `v` MSB-first. `n` ≤ 32.
    pub fn f(&mut self, n: u32, v: u32) {
        debug_assert!(n <= 32);
        if n == 0 {
            return;
        }
        for i in (0..n).rev() {
            let b = (v >> i) & 1;
            self.bit(b != 0);
        }
    }

    /// `f(n, v)` for n in 33..=64.
    pub fn f64(&mut self, n: u32, v: u64) {
        debug_assert!(n <= 64);
        if n <= 32 {
            self.f(n, v as u32);
            return;
        }
        let high_bits = n - 32;
        self.f(high_bits, (v >> 32) as u32);
        self.f(32, v as u32);
    }

    /// `su(n, v)` — signed two's-complement `n`-bit field.
    pub fn su(&mut self, n: u32, v: i32) {
        debug_assert!(n > 0 && n <= 32);
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let unsigned = (v as u32) & mask;
        self.f(n, unsigned);
    }

    /// `uvlc(v)` — universal variable-length code (§4.10.3 inverse).
    /// `value = (1 << leadingZeros) - 1 + payload` ⇒ leadingZeros equals
    /// `floor(log2(value + 1))`.
    pub fn uvlc(&mut self, value: u32) {
        if value == u32::MAX {
            // Spec caps leading_zeros at 32 — emit 32 zero bits.
            for _ in 0..32 {
                self.bit(false);
            }
            return;
        }
        let val_plus_1 = value + 1;
        let leading = 31 - val_plus_1.leading_zeros();
        for _ in 0..leading {
            self.bit(false);
        }
        self.bit(true);
        if leading > 0 {
            let payload = value - ((1u32 << leading) - 1);
            self.f(leading, payload);
        }
    }

    /// `leb128(v)` — little-endian unsigned variable-length integer.
    /// Bytes are emitted to the output stream byte-aligned. Caller must
    /// ensure the writer is byte-aligned at call time.
    pub fn leb128(&mut self, value: u64) {
        debug_assert!(self.is_byte_aligned());
        let mut v = value;
        loop {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                byte |= 0x80;
                self.f(8, byte as u32);
            } else {
                self.f(8, byte as u32);
                break;
            }
        }
    }

    /// `leb128(v)` written in exactly `fixed_len` bytes (1..=8). Useful
    /// when the size prefix must be a known byte width (e.g. so the
    /// payload can be written first and the length back-patched without
    /// shifting the buffer).
    pub fn leb128_fixed(&mut self, value: u64, fixed_len: usize) {
        debug_assert!(self.is_byte_aligned());
        debug_assert!(fixed_len >= 1 && fixed_len <= 8);
        let mut v = value;
        for i in 0..fixed_len {
            let mut byte = (v & 0x7f) as u8;
            v >>= 7;
            let last = i + 1 == fixed_len;
            if !last {
                byte |= 0x80;
            }
            self.f(8, byte as u32);
        }
    }

    /// `ns(n, v)` — non-symmetric uniform integer (§4.10.6 inverse).
    /// `n` MUST be > 0; for `n == 1` no bits are emitted.
    pub fn ns(&mut self, n: u32, v: u32) {
        debug_assert!(n > 0);
        if n == 1 {
            return;
        }
        let w = ceil_log2(n);
        let m = (1u32 << w) - n;
        if v < m {
            self.f(w - 1, v);
        } else {
            let extra_bit = (v + m) & 1;
            let high = (v + m) >> 1;
            self.f(w - 1, high);
            self.bit(extra_bit != 0);
        }
    }

    /// `byte_alignment()` — pad with zero bits up to the next byte
    /// boundary (§5.3.5). No-op if already byte-aligned.
    pub fn byte_align(&mut self) {
        while self.bits_in_acc != 0 {
            self.bit(false);
        }
    }

    /// `trailing_bits(nbBits)` — emit `1` followed by `nbBits-1` zero
    /// bits, used to terminate header bodies (§5.3.4).
    pub fn trailing_bits(&mut self, nb_bits: u32) {
        if nb_bits == 0 {
            return;
        }
        self.bit(true);
        for _ in 1..nb_bits {
            self.bit(false);
        }
    }

    /// Convenience: write a single trailing-`1` bit then byte-align with
    /// zero padding. This is the §5.3.4 OBU body terminator pattern used
    /// by the Sequence Header and Frame Header OBUs.
    pub fn write_obu_trailing_bits(&mut self) {
        self.bit(true);
        self.byte_align();
    }
}

/// Ceiling of log2(x). Mirrors the bitreader helper.
pub fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        return 0;
    }
    32 - (x - 1).leading_zeros()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitreader::BitReader;

    #[test]
    fn bit_msb_first_roundtrip() {
        let mut w = BitWriter::new();
        w.bit(true);
        w.f(2, 0b01);
        w.f(5, 0b1_0001);
        w.f(8, 0b0101_0101);
        let buf = w.finish();
        assert_eq!(buf, vec![0b1011_0001u8, 0b0101_0101]);

        let mut br = BitReader::new(&buf);
        assert_eq!(br.f(1).unwrap(), 1);
        assert_eq!(br.f(2).unwrap(), 0b01);
        assert_eq!(br.f(5).unwrap(), 0b1_0001);
        assert_eq!(br.f(8).unwrap(), 0b0101_0101);
    }

    #[test]
    fn leb128_roundtrip_via_bitreader() {
        for v in [0u64, 1, 0x7F, 0x80, 300, 0xFFFF, 0x1FFFFF] {
            let mut w = BitWriter::new();
            w.leb128(v);
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(br.leb128().unwrap(), v, "leb128 roundtrip {v}");
        }
    }

    #[test]
    fn leb128_fixed_writes_known_length() {
        let mut w = BitWriter::new();
        w.leb128_fixed(7, 4);
        let buf = w.finish();
        assert_eq!(buf.len(), 4);
        // continuation bits set on first 3 bytes, last byte 0x07.
        assert_eq!(buf[0] & 0x80, 0x80);
        assert_eq!(buf[1] & 0x80, 0x80);
        assert_eq!(buf[2] & 0x80, 0x80);
        assert_eq!(buf[3], 0x07);

        let mut br = BitReader::new(&buf);
        assert_eq!(br.leb128().unwrap(), 7);
    }

    #[test]
    fn uvlc_roundtrip_via_bitreader() {
        for v in [0u32, 1, 2, 3, 4, 5, 12, 127, 1023, 65535, 0x1FFFFF] {
            let mut w = BitWriter::new();
            w.uvlc(v);
            w.byte_align();
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(br.uvlc().unwrap(), v, "uvlc roundtrip {v}");
        }
    }

    #[test]
    fn su_roundtrip_via_bitreader() {
        for &v in &[0i32, 1, -1, 7, -7, 63, -64] {
            let mut w = BitWriter::new();
            w.su(7, v);
            w.byte_align();
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(br.su(7).unwrap(), v, "su(7) roundtrip {v}");
        }
    }

    #[test]
    fn ns_roundtrip_via_bitreader() {
        // n = 5 ⇒ w = 3, m = 8 - 5 = 3.
        for v in 0u32..5 {
            let mut w = BitWriter::new();
            w.ns(5, v);
            w.byte_align();
            let buf = w.finish();
            let mut br = BitReader::new(&buf);
            assert_eq!(br.ns(5).unwrap(), v, "ns(5) roundtrip {v}");
        }
    }

    #[test]
    fn ns_n_one_emits_no_bits() {
        let mut w = BitWriter::new();
        assert_eq!(w.bit_position(), 0);
        w.ns(1, 0);
        assert_eq!(w.bit_position(), 0);
    }

    #[test]
    fn byte_align_pads_zeros() {
        let mut w = BitWriter::new();
        w.f(3, 0b101);
        w.byte_align();
        let buf = w.finish();
        assert_eq!(buf, vec![0b1010_0000]);
    }

    #[test]
    fn write_obu_trailing_bits_terminates_with_one_then_zeros() {
        let mut w = BitWriter::new();
        w.f(3, 0b101);
        w.write_obu_trailing_bits();
        let buf = w.finish();
        // Expected: 101 1 0000 = 0b1011_0000 = 0xB0
        assert_eq!(buf, vec![0xB0]);
    }
}
