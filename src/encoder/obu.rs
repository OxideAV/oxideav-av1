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
//!   * §5.3.4 `trailing_bits()` — appended to the body of any OBU
//!     with `obu_size > 0` whose `obu_type` is not `OBU_TILE_GROUP`
//!     / `OBU_TILE_LIST` / `OBU_FRAME` (per the §5.3.1
//!     `open_bitstream_unit()` tail). [`write_obu_with_size`] handles
//!     this for the caller — the body writer returns its bytes
//!     byte-aligned and the wrapper appends the §5.3.4 trailer
//!     before computing `obu_size`.
//!
//! Multiple OBUs in a temporal unit are written by calling
//! [`ObuWriter::write`] in sequence on the same output `Vec<u8>`
//! (or any byte sink that implements [`std::io::Write`]); the
//! concatenation is the §5.2 byte-aligned stream. The
//! [`write_temporal_unit`] convenience walks a sequence of
//! [`ObuFrame`] descriptors and emits them in §7.5 order with the
//! §5.3.4 trailing_bits applied per OBU.
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

/// `true` if the §5.3.1 wrapper appends `trailing_bits()` to a
/// non-empty body for this `obu_type`. From the §5.3.1
/// `open_bitstream_unit()` tail: every type **except**
/// `OBU_TILE_GROUP`, `OBU_TILE_LIST`, and `OBU_FRAME` gets the
/// trailing-bits treatment when `obu_size > 0`. (The three excluded
/// types own their own byte-alignment inside their body syntax —
/// `tile_group_obu()` and friends call `byte_alignment()` directly.)
pub fn obu_type_takes_trailing_bits(t: ObuType) -> bool {
    !matches!(t, ObuType::TileGroup | ObuType::TileList | ObuType::Frame)
}

/// Wrap a freshly-built OBU body into a complete OBU unit per §5.3.1,
/// applying §5.3.4 `trailing_bits` for the OBU types that require it,
/// then writing `obu_header` + `leb128(obu_size)` + the body bytes
/// into `out`.
///
/// `body_bytes` is the byte-aligned payload the relevant body writer
/// (`write_sequence_header_obu` / `write_frame_header_obu` / etc.)
/// returned via `BitWriter::finish` — it must NOT already include the
/// §5.3.4 trailer. The wrapper appends a one-byte `0x80` trailer
/// (`trailing_one_bit = 1` followed by 7 zero pad bits) per §5.3.4
/// when `obu_size > 0` and the §5.3.1 type-gate fires; the resulting
/// `obu_size` is the trailer-inclusive byte count.
///
/// Returns the number of bytes appended to `out`.
pub fn write_obu_with_size(out: &mut Vec<u8>, header: &ObuHeader, body_bytes: &[u8]) -> usize {
    debug_assert!(
        header.has_size,
        "write_obu_with_size requires obu_has_size_field == 1 (§5.2 low-overhead)"
    );
    let needs_trailer = !body_bytes.is_empty() && obu_type_takes_trailing_bits(header.obu_type);
    let start_len = out.len();
    if needs_trailer {
        // §5.3.4 trailer on a byte-aligned body collapses to one byte
        // `0x80` (`trailing_one_bit` + 7 zero pads). `obu_size` then
        // accounts for body bytes plus this trailer byte.
        let mut buf = Vec::with_capacity(body_bytes.len() + 1);
        buf.extend_from_slice(body_bytes);
        buf.push(0x80);
        write_obu(out, header, &buf);
    } else {
        write_obu(out, header, body_bytes);
    }
    out.len() - start_len
}

/// One OBU's worth of pre-built body bytes plus the framing header.
///
/// `body` is the byte-aligned payload the OBU body writer
/// (`write_sequence_header_obu` / `write_frame_header_obu` / etc.)
/// produced; `header` is the §5.3.2 header descriptor (and `has_size`
/// must be `true` because [`write_temporal_unit`] emits the §5.2
/// low-overhead format).
#[derive(Debug, Clone)]
pub struct ObuFrame {
    pub header: ObuHeader,
    pub body: Vec<u8>,
}

impl ObuFrame {
    /// Convenience constructor with `has_size = true` and no extension.
    pub fn new(obu_type: ObuType, body: Vec<u8>) -> Self {
        Self {
            header: ObuHeader::new(obu_type),
            body,
        }
    }
}

/// Aggregate a sequence of OBUs into one §7.5 temporal unit and
/// return the byte-aligned concatenation.
///
/// The §5.2 low-overhead format permits any concatenation of
/// `open_bitstream_unit()` outputs; this helper walks the supplied
/// `frames` in order, calls [`write_obu_with_size`] for each, and
/// returns the resulting buffer. Per §7.5 the caller is expected to
/// place an `OBU_TEMPORAL_DELIMITER` at the start of the unit and any
/// applicable `OBU_SEQUENCE_HEADER` before the first frame headers
/// of a new coded video sequence — see [`build_temporal_unit`] for a
/// helper that handles the TD prefix automatically.
pub fn write_temporal_unit(frames: &[ObuFrame]) -> Vec<u8> {
    let mut out = Vec::new();
    for frame in frames {
        write_obu_with_size(&mut out, &frame.header, &frame.body);
    }
    out
}

/// Build a §7.5 temporal unit with an automatic `OBU_TEMPORAL_DELIMITER`
/// prefix and an optional `OBU_SEQUENCE_HEADER`.
///
/// `seq_payload`, when supplied, is wrapped as an `OBU_SEQUENCE_HEADER`
/// OBU and emitted right after the TD (per §7.5: "the first OBU in
/// the first frame_unit of each temporal_unit must be a temporal
/// delimiter OBU"). The `frame_obus` follow in caller order.
pub fn build_temporal_unit(seq_payload: Option<&[u8]>, frame_obus: &[ObuFrame]) -> Vec<u8> {
    let mut frames: Vec<ObuFrame> = Vec::with_capacity(2 + frame_obus.len());
    // §7.5: OBU_TEMPORAL_DELIMITER carries zero payload — the §5.3.1
    // wrapper skips trailing_bits because obu_size == 0.
    frames.push(ObuFrame::new(ObuType::TemporalDelimiter, Vec::new()));
    if let Some(sh) = seq_payload {
        frames.push(ObuFrame::new(ObuType::SequenceHeader, sh.to_vec()));
    }
    frames.extend_from_slice(frame_obus);
    write_temporal_unit(&frames)
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

    // -----------------------------------------------------------------
    // §5.3.1 + §5.3.4 — write_obu_with_size / temporal_unit aggregation
    // -----------------------------------------------------------------

    #[test]
    fn write_obu_with_size_appends_trailing_bits_for_sequence_header() {
        // §5.3.1 gate fires (SH is not TG/TL/FRAME and obu_size > 0)
        // ⇒ one 0x80 trailer byte appended; obu_size counts the trailer.
        let body = vec![0x11, 0x22, 0x33];
        let mut out = Vec::new();
        write_obu_with_size(&mut out, &ObuHeader::new(ObuType::SequenceHeader), &body);
        let (desc, _consumed) = parse_obu(&out).unwrap();
        assert_eq!(desc.obu_type, ObuType::SequenceHeader);
        assert_eq!(desc.payload_len, 4); // body + 1-byte trailer
        assert_eq!(&desc.payload[..3], &body[..]);
        assert_eq!(desc.payload[3], 0x80);
    }

    #[test]
    fn write_obu_with_size_skips_trailer_for_frame_obu() {
        // §5.3.1 explicit exclusion: OBU_FRAME owns its own alignment.
        let body = vec![0xAA, 0xBB];
        let mut out = Vec::new();
        write_obu_with_size(&mut out, &ObuHeader::new(ObuType::Frame), &body);
        let (desc, _) = parse_obu(&out).unwrap();
        assert_eq!(desc.payload_len, 2);
        assert_eq!(desc.payload, &body[..]);
    }

    #[test]
    fn write_obu_with_size_skips_trailer_for_tile_group_and_tile_list() {
        for ty in [ObuType::TileGroup, ObuType::TileList] {
            let body = vec![0x77, 0x88];
            let mut out = Vec::new();
            write_obu_with_size(&mut out, &ObuHeader::new(ty), &body);
            let (desc, _) = parse_obu(&out).unwrap();
            assert_eq!(desc.payload_len, 2, "type {ty:?}");
            assert_eq!(desc.payload, &body[..], "type {ty:?}");
        }
    }

    #[test]
    fn write_obu_with_size_zero_body_emits_no_trailer() {
        // §5.3.1 conditional `obu_size > 0` short-circuits for empty
        // bodies, including OBU_TEMPORAL_DELIMITER.
        let mut out = Vec::new();
        write_obu_with_size(&mut out, &ObuHeader::new(ObuType::TemporalDelimiter), &[]);
        // 1 header byte + 1 leb128(0) byte = 2 bytes total.
        assert_eq!(out, vec![0x12, 0x00]);
    }

    #[test]
    fn write_temporal_unit_walks_in_order() {
        // TD + SH + FH, all framed. Walk the result with ObuIter.
        let frames = vec![
            ObuFrame::new(ObuType::TemporalDelimiter, Vec::new()),
            ObuFrame::new(ObuType::SequenceHeader, vec![0xAA, 0xBB]),
            ObuFrame::new(ObuType::FrameHeader, vec![0x11]),
        ];
        let bytes = write_temporal_unit(&frames);
        let descs: Vec<_> = ObuIter::new(&bytes).collect::<Result<_, _>>().unwrap();
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
        // SH body + 1-byte §5.3.4 trailer => payload_len = 3
        assert_eq!(descs[1].payload_len, 3);
        assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
        assert_eq!(descs[2].payload_len, 2);
    }

    #[test]
    fn build_temporal_unit_emits_td_then_sh_then_frames() {
        let sh_payload = [0xDE, 0xAD];
        let fh = ObuFrame::new(ObuType::FrameHeader, vec![0xBE, 0xEF]);
        let bytes = build_temporal_unit(Some(&sh_payload), &[fh]);
        let descs: Vec<_> = ObuIter::new(&bytes).collect::<Result<_, _>>().unwrap();
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[0].payload_len, 0);
        assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
        // SH body (2) + trailer (1) = 3.
        assert_eq!(descs[1].payload_len, 3);
        assert_eq!(&descs[1].payload[..2], &sh_payload);
        assert_eq!(descs[1].payload[2], 0x80);
        assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
        assert_eq!(descs[2].payload_len, 3); // FH body + trailer
    }

    #[test]
    fn build_temporal_unit_without_sh_just_emits_td_and_frames() {
        let fh = ObuFrame::new(ObuType::FrameHeader, vec![0x42]);
        let bytes = build_temporal_unit(None, &[fh]);
        let descs: Vec<_> = ObuIter::new(&bytes).collect::<Result<_, _>>().unwrap();
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[1].obu_type, ObuType::FrameHeader);
    }

    #[test]
    fn obu_type_takes_trailing_bits_matches_spec_exclusions() {
        for ty in [
            ObuType::SequenceHeader,
            ObuType::TemporalDelimiter,
            ObuType::FrameHeader,
            ObuType::Metadata,
            ObuType::RedundantFrameHeader,
            ObuType::Padding,
            ObuType::Reserved(9),
        ] {
            assert!(obu_type_takes_trailing_bits(ty), "type {ty:?}");
        }
        for ty in [ObuType::TileGroup, ObuType::TileList, ObuType::Frame] {
            assert!(!obu_type_takes_trailing_bits(ty), "type {ty:?}");
        }
    }
}
