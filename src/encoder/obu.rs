//! OBU framing writer — inverse of [`crate::obu`].
//!
//! Emits the §5.3 wrapper:
//!
//!   * §5.3.2 `obu_header()` — one byte with `obu_forbidden_bit ==
//!     0`, the 4-bit `obu_type`, `obu_extension_flag`,
//!     `obu_has_size_field`, and the reserved low bit.
//!   * §5.3.3 `obu_extension_header()` — one byte with 3-bit
//!     `temporal_id`, 2-bit `spatial_id`, and three reserved bits
//!     (left at 0).
//!   * §5.3.1 / §4.10.5 — optional `leb128()` `obu_size` size
//!     field, when `obu_has_size_field == 1` (the §5.2 low-overhead
//!     bytestream format every OBU emitted here uses).
//!
//! Multiple OBUs in a temporal unit are written by calling
//! [`ObuWriter::write`] in sequence on the same output `Vec<u8>`
//! (or any byte sink that implements [`std::io::Write`]); the
//! concatenation is the §5.2 byte-aligned stream.
//!
//! See also [`crate::obu::ObuType`] — the same `obu_type`
//! taxonomy the parser uses.

use crate::encoder::bitwriter::BitWriter;
use crate::obu::ObuType;

/// §5.3.3 extension header fields. Both `temporal_id` (3 bits) and
/// `spatial_id` (2 bits) live here; the three reserved bits are
/// written as 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuExtensionHeader {
    pub temporal_id: u8,
    pub spatial_id: u8,
}

impl ObuExtensionHeader {
    /// Construct with field-range debug-asserts (temporal_id in
    /// `0..=7`, spatial_id in `0..=3`).
    pub fn new(temporal_id: u8, spatial_id: u8) -> Self {
        debug_assert!(temporal_id <= 7, "§5.3.3 temporal_id is 3 bits");
        debug_assert!(spatial_id <= 3, "§5.3.3 spatial_id is 2 bits");
        Self {
            temporal_id,
            spatial_id,
        }
    }
}

/// §5.3.2 OBU header descriptor.
///
/// `extension` is `Some` iff `obu_extension_flag == 1` should be
/// emitted. `has_size` is the value of `obu_has_size_field`. This
/// writer always emits `obu_size` when `has_size == true`; the
/// §5.2 low-overhead format requires it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObuHeader {
    pub obu_type: ObuType,
    pub extension: Option<ObuExtensionHeader>,
    pub has_size: bool,
}

impl ObuHeader {
    /// Construct a header with no extension and `has_size = true`
    /// (the §5.2 low-overhead default).
    pub fn new(obu_type: ObuType) -> Self {
        Self {
            obu_type,
            extension: None,
            has_size: true,
        }
    }

    /// Add an extension header (`temporal_id` / `spatial_id`).
    pub fn with_extension(mut self, ext: ObuExtensionHeader) -> Self {
        self.extension = Some(ext);
        self
    }
}

/// OBU writer. Stateless — each `write` call emits one complete
/// OBU into the supplied output buffer.
#[derive(Debug, Default)]
pub struct ObuWriter;

impl ObuWriter {
    /// Append one OBU to `out`. The byte layout is:
    ///
    ///   * `obu_header` (1 byte, §5.3.2).
    ///   * `obu_extension_header` (1 byte, §5.3.3) iff
    ///     `header.extension.is_some()`.
    ///   * `obu_size` leb128 (§4.10.5) iff `header.has_size`.
    ///   * `payload` bytes (must be byte-aligned by the caller).
    ///
    /// Returns the total number of bytes appended.
    pub fn write(&self, out: &mut Vec<u8>, header: &ObuHeader, payload: &[u8]) -> usize {
        let start_len = out.len();
        write_obu(out, header, payload);
        out.len() - start_len
    }
}

/// Append one OBU. Free function form of [`ObuWriter::write`] for
/// callers that don't want to construct the (zero-sized) writer.
pub fn write_obu(out: &mut Vec<u8>, header: &ObuHeader, payload: &[u8]) {
    // §5.3.2: build the header byte MSB-first.
    //   bit 7      obu_forbidden_bit  (0)
    //   bits 6..3  obu_type           f(4)
    //   bit 2      obu_extension_flag f(1)
    //   bit 1      obu_has_size_field f(1)
    //   bit 0      obu_reserved_1bit  (0)
    let mut bw = BitWriter::new();
    bw.write_bits(1, 0); // obu_forbidden_bit
    bw.write_bits(4, u64::from(header.obu_type.as_raw() & 0x0f));
    bw.write_bits(1, u64::from(header.extension.is_some()));
    bw.write_bits(1, u64::from(header.has_size));
    bw.write_bits(1, 0); // obu_reserved_1bit
    debug_assert!(bw.is_byte_aligned());

    if let Some(ext) = header.extension {
        // §5.3.3: temporal_id (3) | spatial_id (2) | reserved (3).
        bw.write_bits(3, u64::from(ext.temporal_id & 0x7));
        bw.write_bits(2, u64::from(ext.spatial_id & 0x3));
        bw.write_bits(3, 0); // extension_header_reserved_3bits
        debug_assert!(bw.is_byte_aligned());
    }

    let header_bytes = bw.finish();
    out.extend_from_slice(&header_bytes);

    if header.has_size {
        let mut size_bw = BitWriter::new();
        size_bw.write_leb128(payload.len() as u64);
        out.extend_from_slice(&size_bw.finish());
    }

    out.extend_from_slice(payload);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obu::{parse_obu, ObuIter};

    #[test]
    fn temporal_delimiter_byte_exact() {
        // §7.5 TD: obu_type=2, no extension, has_size=1, empty payload.
        // Header byte: obu_type=2 in bits 6..3 => 0b0001_0xxx;
        // ext=0 (bit 2), size=1 (bit 1), reserved=0 (bit 0).
        // => 0b0001_0010 = 0x12. Then leb128(0) = 0x00.
        let mut out = Vec::new();
        ObuWriter.write(&mut out, &ObuHeader::new(ObuType::TemporalDelimiter), &[]);
        assert_eq!(out, vec![0x12, 0x00]);
    }

    #[test]
    fn round_trip_through_parser_temporal_delimiter() {
        let mut out = Vec::new();
        write_obu(&mut out, &ObuHeader::new(ObuType::TemporalDelimiter), &[]);
        let (desc, consumed) = parse_obu(&out).unwrap();
        assert_eq!(desc.obu_type, ObuType::TemporalDelimiter);
        assert!(!desc.extension_flag);
        assert!(desc.has_size_field);
        assert_eq!(desc.temporal_id, 0);
        assert_eq!(desc.spatial_id, 0);
        assert_eq!(desc.payload_len, 0);
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn round_trip_through_parser_frame_with_extension() {
        // Frame (obu_type=6) + extension(temporal=3, spatial=2),
        // 5-byte payload.
        let payload = [0xa1, 0xa2, 0xa3, 0xa4, 0xa5];
        let header = ObuHeader::new(ObuType::Frame).with_extension(ObuExtensionHeader::new(3, 2));
        let mut out = Vec::new();
        write_obu(&mut out, &header, &payload);
        let (desc, consumed) = parse_obu(&out).unwrap();
        assert_eq!(desc.obu_type, ObuType::Frame);
        assert!(desc.extension_flag);
        assert!(desc.has_size_field);
        assert_eq!(desc.temporal_id, 3);
        assert_eq!(desc.spatial_id, 2);
        assert_eq!(desc.payload_len, 5);
        assert_eq!(desc.payload, &payload);
        assert_eq!(consumed, out.len());
    }

    #[test]
    fn round_trip_all_obu_types_no_extension() {
        for obu_type in [
            ObuType::SequenceHeader,
            ObuType::TemporalDelimiter,
            ObuType::FrameHeader,
            ObuType::TileGroup,
            ObuType::Metadata,
            ObuType::Frame,
            ObuType::RedundantFrameHeader,
            ObuType::TileList,
            ObuType::Padding,
        ] {
            let payload: Vec<u8> = (0..7).collect();
            let mut out = Vec::new();
            write_obu(&mut out, &ObuHeader::new(obu_type), &payload);
            let (desc, _) = parse_obu(&out).unwrap();
            assert_eq!(desc.obu_type, obu_type);
            assert_eq!(desc.payload, &payload[..]);
        }
    }

    #[test]
    fn multi_obu_concatenation_iter_walks() {
        // Three OBUs back-to-back in a buffer, then iterate via the
        // existing ObuIter.
        let mut out = Vec::new();
        write_obu(&mut out, &ObuHeader::new(ObuType::TemporalDelimiter), &[]);
        write_obu(
            &mut out,
            &ObuHeader::new(ObuType::SequenceHeader),
            &[0x11, 0x22],
        );
        write_obu(
            &mut out,
            &ObuHeader::new(ObuType::Frame),
            &[0x33, 0x44, 0x55],
        );

        let mut iter = ObuIter::new(&out);
        let a = iter.next().unwrap().unwrap();
        assert_eq!(a.obu_type, ObuType::TemporalDelimiter);
        assert!(a.payload.is_empty());
        let b = iter.next().unwrap().unwrap();
        assert_eq!(b.obu_type, ObuType::SequenceHeader);
        assert_eq!(b.payload, &[0x11, 0x22]);
        let c = iter.next().unwrap().unwrap();
        assert_eq!(c.obu_type, ObuType::Frame);
        assert_eq!(c.payload, &[0x33, 0x44, 0x55]);
        assert!(iter.next().is_none());
    }

    #[test]
    fn leb128_size_field_handles_multi_byte() {
        // Payload of 200 bytes forces a 2-byte leb128 (129..=16383
        // range straddles 0x80; 200 > 0x7f so 2 bytes).
        let payload = vec![0u8; 200];
        let mut out = Vec::new();
        write_obu(&mut out, &ObuHeader::new(ObuType::Padding), &payload);
        // Header byte (1) + leb128(200) (2) + payload (200) = 203.
        assert_eq!(out.len(), 203);
        let (desc, consumed) = parse_obu(&out).unwrap();
        assert_eq!(desc.payload_len, 200);
        assert_eq!(consumed, 203);
    }

    #[test]
    fn extension_header_byte_layout_exact() {
        // Extension byte = (temporal << 5) | (spatial << 3).
        // temporal=7, spatial=3 => 0b111_11_000 = 0xF8.
        let header =
            ObuHeader::new(ObuType::FrameHeader).with_extension(ObuExtensionHeader::new(7, 3));
        let mut out = Vec::new();
        write_obu(&mut out, &header, &[]);
        // Header byte for obu_type=3, ext=1, size=1:
        //   0b0001_1110 = 0x1E.
        assert_eq!(out[0], 0x1E);
        assert_eq!(out[1], 0xF8);
        assert_eq!(out[2], 0x00); // leb128(0) size
    }

    #[test]
    fn reserved_obu_type_round_trips() {
        // ObuType::Reserved(9) — the parser surfaces obu_type==9 as
        // Reserved(9); the writer must round-trip the same raw value.
        let header = ObuHeader::new(ObuType::Reserved(9));
        let mut out = Vec::new();
        write_obu(&mut out, &header, &[0xde]);
        let (desc, _) = parse_obu(&out).unwrap();
        assert_eq!(desc.obu_type, ObuType::Reserved(9));
    }
}
