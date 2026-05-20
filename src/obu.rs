//! OBU bytestream walker.
//!
//! This module decodes the byte-aligned wrapper that the AV1 spec
//! places around every Open Bitstream Unit (OBU): the one-byte header
//! defined in §5.3.2 (`obu_header`), the optional extension header
//! defined in §5.3.3 (`obu_extension_header`), and the optional
//! `leb128` size field defined alongside the general OBU syntax in
//! §5.3.1 plus the `leb128()` parsing process in §4.10.5.
//!
//! Round 1 scope: just the wrapper. We expose a descriptor that
//! identifies the OBU type, the optional temporal_id / spatial_id,
//! and the slice of payload bytes. We do **not** descend into the
//! payload — `sequence_header_obu`, `frame_header_obu`,
//! `tile_group_obu`, etc. are out of scope for this round.
//!
//! The walker assumes the "Low overhead bitstream format" of §5.2:
//! each OBU has `obu_has_size_field == 1` and an explicit `leb128`
//! `obu_size`. Annex B's length-delimited format is not implemented
//! yet.
//!
//! References (all under `docs/video/av1/`):
//!   * §4.10.5 — `leb128()`
//!   * §5.3.1 — General OBU syntax
//!   * §5.3.2 — OBU header syntax
//!   * §5.3.3 — OBU extension header syntax
//!   * §6.2.2 — OBU header semantics (obu_type table)

use crate::Error;

/// Symbolic obu_type values from §6.2.2.
///
/// Values 0 and 9..=14 are reserved; we collapse them into
/// [`ObuType::Reserved`] with the raw 4-bit value preserved so
/// callers can still observe the on-wire byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObuType {
    /// `0` and `9..=14`: reserved for future use.
    Reserved(u8),
    /// `1` — `OBU_SEQUENCE_HEADER`.
    SequenceHeader,
    /// `2` — `OBU_TEMPORAL_DELIMITER`.
    TemporalDelimiter,
    /// `3` — `OBU_FRAME_HEADER`.
    FrameHeader,
    /// `4` — `OBU_TILE_GROUP`.
    TileGroup,
    /// `5` — `OBU_METADATA`.
    Metadata,
    /// `6` — `OBU_FRAME`.
    Frame,
    /// `7` — `OBU_REDUNDANT_FRAME_HEADER`.
    RedundantFrameHeader,
    /// `8` — `OBU_TILE_LIST`.
    TileList,
    /// `15` — `OBU_PADDING`.
    Padding,
}

impl ObuType {
    /// Decode the 4-bit `obu_type` field per §6.2.2.
    ///
    /// `raw` must be in `0..=15`; callers are responsible for the
    /// bit-field mask before calling.
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::SequenceHeader,
            2 => Self::TemporalDelimiter,
            3 => Self::FrameHeader,
            4 => Self::TileGroup,
            5 => Self::Metadata,
            6 => Self::Frame,
            7 => Self::RedundantFrameHeader,
            8 => Self::TileList,
            15 => Self::Padding,
            other => Self::Reserved(other),
        }
    }

    /// The raw 4-bit `obu_type` value that produced `self`.
    pub fn as_raw(&self) -> u8 {
        match self {
            Self::Reserved(v) => *v,
            Self::SequenceHeader => 1,
            Self::TemporalDelimiter => 2,
            Self::FrameHeader => 3,
            Self::TileGroup => 4,
            Self::Metadata => 5,
            Self::Frame => 6,
            Self::RedundantFrameHeader => 7,
            Self::TileList => 8,
            Self::Padding => 15,
        }
    }
}

/// One OBU as observed by the walker.
///
/// `header_len` is the number of header bytes consumed (1 byte for
/// `obu_header` plus 1 byte if `obu_extension_flag == 1`, plus the
/// `Leb128Bytes` count from §4.10.5 when an explicit size field is
/// present). `payload_len` is the value of `obu_size` — i.e. the
/// payload byte count not including header or size field, per §6.2.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuDescriptor<'a> {
    /// Decoded `obu_type`.
    pub obu_type: ObuType,
    /// `obu_extension_flag` (§5.3.2).
    pub extension_flag: bool,
    /// `obu_has_size_field` (§5.3.2).
    pub has_size_field: bool,
    /// `temporal_id` from §5.3.3 when `extension_flag` is set,
    /// otherwise inferred to 0 per §6.2.3.
    pub temporal_id: u8,
    /// `spatial_id` from §5.3.3 when `extension_flag` is set,
    /// otherwise inferred to 0 per §6.2.3.
    pub spatial_id: u8,
    /// `obu_size` value (§6.2.1): payload length in bytes.
    pub payload_len: usize,
    /// Number of bytes consumed by the OBU header + size field.
    pub header_len: usize,
    /// Slice covering exactly the `payload_len` payload bytes.
    pub payload: &'a [u8],
}

/// Iterator over OBUs in a bytestream that uses the §5.2 low-overhead
/// format (every OBU carries an explicit `obu_size`).
#[derive(Debug)]
pub struct ObuIter<'a> {
    rest: &'a [u8],
    /// Total bytes already consumed, for diagnostic offsets.
    offset: usize,
    /// Set once an error has been emitted so subsequent `next()`
    /// calls return `None` rather than re-emitting the same error.
    done: bool,
}

impl<'a> ObuIter<'a> {
    /// Wrap a bytestream that is expected to contain a concatenation
    /// of OBUs in low-overhead format.
    pub fn new(bytes: &'a [u8]) -> Self {
        Self {
            rest: bytes,
            offset: 0,
            done: false,
        }
    }

    /// Byte offset of the next OBU header within the original buffer.
    pub fn position(&self) -> usize {
        self.offset
    }
}

/// Parse a single OBU out of `bytes` and return the descriptor plus
/// the total number of bytes consumed (header + payload).
///
/// This is the parsing primitive [`ObuIter`] is built on; callers
/// that prefer not to use the iterator can call it directly.
pub fn parse_obu(bytes: &[u8]) -> Result<(ObuDescriptor<'_>, usize), Error> {
    // §5.3.2: at least one header byte must be present.
    let first = *bytes.first().ok_or(Error::UnexpectedEnd)?;
    let mut cursor = 1usize;

    // §5.3.2 field layout, MSB first:
    //   bit 7      obu_forbidden_bit  (must be 0, §6.2.2)
    //   bits 6..3  obu_type           f(4)
    //   bit 2      obu_extension_flag f(1)
    //   bit 1      obu_has_size_field f(1)
    //   bit 0      obu_reserved_1bit  (must be 0, §6.2.2; ignored)
    let forbidden_bit = (first >> 7) & 0x1;
    if forbidden_bit != 0 {
        return Err(Error::ForbiddenBitSet);
    }
    let obu_type_raw = (first >> 3) & 0x0f;
    let extension_flag = ((first >> 2) & 0x1) != 0;
    let has_size_field = ((first >> 1) & 0x1) != 0;
    // The reserved low bit is ignored per §6.2.2 ("The value is ignored
    // by a decoder."); we do not enforce it.

    // §5.3.3: optional extension header.
    let (temporal_id, spatial_id) = if extension_flag {
        let ext = *bytes.get(cursor).ok_or(Error::UnexpectedEnd)?;
        cursor += 1;
        let temporal = (ext >> 5) & 0x7; // f(3)
        let spatial = (ext >> 3) & 0x3; // f(2)
                                        // extension_header_reserved_3bits is ignored per §6.2.3.
        (temporal, spatial)
    } else {
        // §6.2.3: when not present, temporal_id and spatial_id are
        // inferred to be 0.
        (0u8, 0u8)
    };

    // §5.3.1 + §4.10.5: optional leb128 obu_size.
    let payload_len = if has_size_field {
        let (size, size_bytes) = parse_leb128(&bytes[cursor..])?;
        cursor += size_bytes;
        usize::try_from(size).map_err(|_| Error::SizeOverflow)?
    } else {
        // §5.2 low-overhead format requires has_size_field == 1.
        // Annex B (length-delimited) is out of scope this round.
        return Err(Error::MissingSizeField);
    };

    let payload_end = cursor.checked_add(payload_len).ok_or(Error::SizeOverflow)?;
    if payload_end > bytes.len() {
        return Err(Error::UnexpectedEnd);
    }
    let payload = &bytes[cursor..payload_end];

    let descriptor = ObuDescriptor {
        obu_type: ObuType::from_raw(obu_type_raw),
        extension_flag,
        has_size_field,
        temporal_id,
        spatial_id,
        payload_len,
        header_len: cursor,
        payload,
    };

    Ok((descriptor, payload_end))
}

/// Parse one `leb128()` value per §4.10.5 and return
/// `(value, Leb128Bytes)`.
///
/// The spec caps the value at `(1 << 32) - 1` and the encoding at
/// 8 bytes. We surface both as [`Error::Leb128Overflow`] /
/// [`Error::Leb128TooLong`] rather than `Result<u32, ...>` because
/// callers may want the byte count even on failure for diagnostics
/// — but in this round we just propagate the error.
pub fn parse_leb128(bytes: &[u8]) -> Result<(u64, usize), Error> {
    let mut value: u64 = 0;
    let mut leb128_bytes: usize = 0;
    // §4.10.5: the loop iterates i = 0..8.
    for i in 0..8 {
        let byte = *bytes.get(i).ok_or(Error::UnexpectedEnd)?;
        // value |= ( (leb128_byte & 0x7f) << (i*7) )
        let chunk = u64::from(byte & 0x7f) << (i * 7);
        value |= chunk;
        leb128_bytes += 1;
        if (byte & 0x80) == 0 {
            // §4.10.5: bit 7 == 0 terminates the encoding.
            // §4.10.5 conformance requirement: returned value
            // must be <= (1 << 32) - 1.
            if value > u64::from(u32::MAX) {
                return Err(Error::Leb128Overflow);
            }
            return Ok((value, leb128_bytes));
        }
    }
    // Reached i == 8 without seeing the terminator — §4.10.5 requires
    // the MSB of the 8th byte to be 0, so anything past this is a
    // conformance violation.
    Err(Error::Leb128TooLong)
}

impl<'a> Iterator for ObuIter<'a> {
    type Item = Result<ObuDescriptor<'a>, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done || self.rest.is_empty() {
            return None;
        }
        match parse_obu(self.rest) {
            Ok((descriptor, consumed)) => {
                self.rest = &self.rest[consumed..];
                self.offset += consumed;
                Some(Ok(descriptor))
            }
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- §4.10.5 leb128() unit tests ---------------------------------

    #[test]
    fn leb128_decodes_single_byte_zero() {
        // The simplest valid encoding: one byte, value 0.
        let (v, n) = parse_leb128(&[0x00]).expect("zero is a valid leb128");
        assert_eq!(v, 0);
        assert_eq!(n, 1);
    }

    #[test]
    fn leb128_decodes_multi_byte_value() {
        // §4.10.5: value |= ((byte & 0x7f) << (i*7)).
        // 0x81 0x01 ==> ((0x01 & 0x7f) << 0) | ((0x01 & 0x7f) << 7)
        //            == 1 | 128 = 129.
        let (v, n) = parse_leb128(&[0x81, 0x01]).expect("two-byte leb128 decodes");
        assert_eq!(v, 129);
        assert_eq!(n, 2);
    }

    #[test]
    fn leb128_allows_redundant_zero_padding() {
        // §4.10.5 Note: multiple encodings of the same value are
        // permitted by the spec. 0x80 0x00 must decode to 0.
        let (v, n) = parse_leb128(&[0x80, 0x00]).expect("padded zero is valid leb128");
        assert_eq!(v, 0);
        assert_eq!(n, 2);
    }

    #[test]
    fn leb128_rejects_value_above_u32_max() {
        // §4.10.5 bitstream-conformance requirement: value must be
        // <= (1 << 32) - 1. Encode 1 << 32 == 0x1_0000_0000.
        // Bytes 0..4 supply zero, byte 5 contributes (1 << 28),
        // and the 5th 7-bit chunk shifted by 28 gives the overflow.
        // We can produce 2^32 as 0x80 0x80 0x80 0x80 0x10.
        let err = parse_leb128(&[0x80, 0x80, 0x80, 0x80, 0x10])
            .expect_err("2^32 exceeds the §4.10.5 cap");
        assert_eq!(err, Error::Leb128Overflow);
    }

    #[test]
    fn leb128_rejects_encoding_longer_than_eight_bytes() {
        // §4.10.5: the MSB of the 8th byte must be 0. If every one
        // of the 8 bytes has bit 7 set, the encoding is malformed.
        let nine_bytes = [0x80u8; 9];
        let err = parse_leb128(&nine_bytes).expect_err("8 continuation bytes must be rejected");
        assert_eq!(err, Error::Leb128TooLong);
    }

    // --- §5.3.2 OBU header unit tests --------------------------------

    /// Build a one-byte OBU header per §5.3.2 field layout.
    fn obu_header_byte(obu_type: u8, ext_flag: bool, has_size_field: bool) -> u8 {
        debug_assert!(obu_type <= 0x0f);
        let ext = u8::from(ext_flag);
        let sz = u8::from(has_size_field);
        // obu_forbidden_bit (0) | obu_type (4) | ext (1) | sz (1) | reserved (0)
        (obu_type << 3) | (ext << 2) | (sz << 1)
    }

    #[test]
    fn parses_temporal_delimiter_no_extension_empty_payload() {
        // Temporal delimiter OBU (§7.5): obu_type == 2, payload empty.
        let header = obu_header_byte(2, false, true);
        let bytes = [header, 0x00 /* leb128 obu_size = 0 */];
        let (obu, consumed) = parse_obu(&bytes).expect("td OBU parses");
        assert_eq!(obu.obu_type, ObuType::TemporalDelimiter);
        assert!(!obu.extension_flag);
        assert!(obu.has_size_field);
        assert_eq!(obu.temporal_id, 0);
        assert_eq!(obu.spatial_id, 0);
        assert_eq!(obu.payload_len, 0);
        assert_eq!(obu.header_len, 2);
        assert!(obu.payload.is_empty());
        assert_eq!(consumed, 2);
    }

    #[test]
    fn parses_frame_obu_with_extension_header() {
        // Frame OBU (obu_type == 6) with extension_flag == 1.
        // Build §5.3.3: temporal_id=3, spatial_id=2, reserved=0.
        // ext byte = (3 << 5) | (2 << 3) = 0x70.
        let header = obu_header_byte(6, true, true);
        let ext = (3 << 5) | (2 << 3);
        // leb128 obu_size = 5, payload bytes 0xa1..0xa5.
        let bytes = [header, ext, 0x05, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5];
        let (obu, consumed) = parse_obu(&bytes).expect("frame OBU parses");
        assert_eq!(obu.obu_type, ObuType::Frame);
        assert!(obu.extension_flag);
        assert!(obu.has_size_field);
        assert_eq!(obu.temporal_id, 3);
        assert_eq!(obu.spatial_id, 2);
        assert_eq!(obu.payload_len, 5);
        assert_eq!(obu.header_len, 3); // 1 header + 1 ext + 1 leb128
        assert_eq!(obu.payload, &[0xa1, 0xa2, 0xa3, 0xa4, 0xa5]);
        assert_eq!(consumed, 8);
    }

    #[test]
    fn iterator_walks_concatenated_obus() {
        // Sequence header (1) with 2-byte payload, then frame (6) with
        // 3-byte payload, then temporal delimiter (2) empty.
        let mut stream = Vec::new();
        stream.push(obu_header_byte(1, false, true));
        stream.push(0x02);
        stream.extend_from_slice(&[0x11, 0x22]);
        stream.push(obu_header_byte(6, false, true));
        stream.push(0x03);
        stream.extend_from_slice(&[0x33, 0x44, 0x55]);
        stream.push(obu_header_byte(2, false, true));
        stream.push(0x00);

        let mut iter = ObuIter::new(&stream);
        let a = iter.next().unwrap().unwrap();
        assert_eq!(a.obu_type, ObuType::SequenceHeader);
        assert_eq!(a.payload, &[0x11, 0x22]);
        let b = iter.next().unwrap().unwrap();
        assert_eq!(b.obu_type, ObuType::Frame);
        assert_eq!(b.payload, &[0x33, 0x44, 0x55]);
        let c = iter.next().unwrap().unwrap();
        assert_eq!(c.obu_type, ObuType::TemporalDelimiter);
        assert!(c.payload.is_empty());
        assert!(iter.next().is_none());
    }

    #[test]
    fn rejects_forbidden_bit_set() {
        // §6.2.2: obu_forbidden_bit must be 0. Set the top bit and
        // confirm rejection.
        let bad = obu_header_byte(2, false, true) | 0x80;
        let err = parse_obu(&[bad, 0x00]).expect_err("obu_forbidden_bit=1 must be rejected");
        assert_eq!(err, Error::ForbiddenBitSet);
    }

    #[test]
    fn rejects_truncated_payload() {
        // obu_size claims 4 bytes of payload but we only supply 2.
        let header = obu_header_byte(4, false, true);
        let bytes = [header, 0x04, 0xaa, 0xbb];
        let err = parse_obu(&bytes).expect_err("truncated payload must error");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    #[test]
    fn rejects_missing_size_field() {
        // has_size_field == 0 is legal in the spec for stream-delimited
        // delivery but not in this walker's low-overhead-only contract.
        let header = obu_header_byte(2, false, false);
        let err = parse_obu(&[header]).expect_err("size-less OBU must be rejected");
        assert_eq!(err, Error::MissingSizeField);
    }

    #[test]
    fn reserved_obu_type_preserves_raw_value() {
        // obu_type == 9 is reserved per §6.2.2; we should see the raw
        // value preserved so callers can choose to skip it via obu_size.
        let header = obu_header_byte(9, false, true);
        let bytes = [header, 0x01, 0xde];
        let (obu, _) = parse_obu(&bytes).expect("reserved OBU parses");
        assert_eq!(obu.obu_type, ObuType::Reserved(9));
        assert_eq!(obu.obu_type.as_raw(), 9);
        assert_eq!(obu.payload, &[0xde]);
    }
}
