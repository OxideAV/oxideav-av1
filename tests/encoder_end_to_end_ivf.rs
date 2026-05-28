//! End-to-end smoke fixture for the encoder arc (round 208).
//!
//! Builds a minimal `SequenceHeader` + `FrameHeader`, wraps them in
//! the §5.3 OBU framing (with §5.3.4 `trailing_bits`), assembles the
//! §7.5 temporal unit (TD + SH + FH), writes it through the IVF v0
//! container, then walks the file back with the parser-side
//! primitives and confirms the embedded headers round-trip.
//!
//! Until the §5.11 `tile_group_obu()` writer lands (next arc), the
//! per-frame OBU sequence is just `OBU_FRAME_HEADER` with zero tiles
//! — enough to validate the framing + container layer end-to-end but
//! not enough to drive a full decode-to-pixels.
//!
//! Spec references (all under `docs/video/av1/`):
//!
//!   * §5.3.1 — `open_bitstream_unit()` framing tail.
//!   * §5.3.4 — `trailing_bits(nbBits)`.
//!   * §5.5.1 — Sequence header OBU.
//!   * §5.9.1 / §5.9.2 — `frame_header_obu()` / `uncompressed_header()`.
//!   * §7.5 — Temporal unit decoding process (TD-prefix invariant).
//!
//! IVF v0 layout is documented inline in
//! `src/encoder/ivf.rs` (file-header byte-exact against
//! `docs/video/av1/fixtures/tiny-i-only-16x16-prof0/input.ivf`).

use oxideav_av1::encoder::ivf::{
    IvfWriter, FOURCC_AV01, IVF_FILE_HEADER_LEN, IVF_FRAME_HEADER_LEN,
};
use oxideav_av1::encoder::temporal_unit::{encode_temporal_unit, TemporalUnitPlan};
use oxideav_av1::frame_header::parse_frame_header;
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;
use std::io::Cursor;

// Same descriptors the per-module round-trip tests use. The two
// `_PAYLOAD` constants are the byte-exact OBU bodies from
// `docs/video/av1/fixtures/tiny-i-only-16x16-prof0/input.ivf`.
const TINY_SEQ_PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
const TINY_FRAME_PAYLOAD: &[u8] = &[
    0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67, 0x6c,
    0xc7, 0xee, 0x51, 0x80,
];

#[test]
fn end_to_end_ivf_round_trip_through_parser() {
    // 1. Parse the seed seq + frame headers (the descriptors the
    //    encoder will re-emit).
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).expect("seed SH parses");
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).expect("seed FH parses");

    // 2. Build a §7.5 temporal unit: TD + SH + FH.
    let frames = [fh.clone()];
    let plan = TemporalUnitPlan {
        seq: &seq,
        emit_sequence_header: true,
        frames: &frames,
    };
    let tu_bytes = encode_temporal_unit(&plan);
    assert!(!tu_bytes.is_empty(), "temporal unit should not be empty");

    // 3. Wrap the temporal unit in IVF v0 — one frame, pts=0.
    let mut buf: Vec<u8> = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut iw = IvfWriter::new(cursor, FOURCC_AV01, 16, 16, 25, 1).expect("IVF header writes");
        iw.write_frame(&tu_bytes, 0).expect("IVF frame writes");
        iw.patch_frame_count().expect("frame_count patch writes");
        assert_eq!(iw.frames_written(), 1);
    }

    // 4. Demux the IVF file back. The file header is a fixed 32-byte
    //    layout; per-frame is 12-byte header + payload.
    assert_eq!(&buf[0..4], b"DKIF");
    let frame_count = u32::from_le_bytes([buf[24], buf[25], buf[26], buf[27]]);
    assert_eq!(frame_count, 1, "patched frame_count");
    // Per-frame header at offset 32.
    let size_off = IVF_FILE_HEADER_LEN;
    let size = u32::from_le_bytes([
        buf[size_off],
        buf[size_off + 1],
        buf[size_off + 2],
        buf[size_off + 3],
    ]) as usize;
    assert_eq!(size, tu_bytes.len());
    let payload_off = size_off + IVF_FRAME_HEADER_LEN;
    let payload = &buf[payload_off..payload_off + size];
    assert_eq!(payload, &tu_bytes[..]);

    // 5. Walk the OBUs and reparse SH + FH from their payloads.
    let descs: Vec<_> = ObuIter::new(payload).collect::<Result<_, _>>().unwrap();
    assert_eq!(descs.len(), 3, "TD + SH + FH");
    assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
    assert_eq!(descs[0].payload_len, 0);

    assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
    let reparsed_seq = parse_sequence_header(descs[1].payload).expect("reparsed SH");
    let mut expected_seq = seq.clone();
    expected_seq.bits_consumed = reparsed_seq.bits_consumed;
    assert_eq!(
        reparsed_seq, expected_seq,
        "SH round-trips through IVF + parser"
    );

    assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
    let reparsed_fh = parse_frame_header(descs[2].payload, &seq).expect("reparsed FH");
    let mut expected_fh = fh.clone();
    expected_fh.bits_consumed = reparsed_fh.bits_consumed;
    assert_eq!(
        reparsed_fh, expected_fh,
        "FH round-trips through IVF + parser"
    );
}

#[test]
fn end_to_end_ivf_multi_frame_round_trip() {
    // Two temporal units in the same IVF: TU0 carries SH+FH, TU1
    // re-uses the SH from TU0 and ships just FH.
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).expect("seed SH parses");
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).expect("seed FH parses");

    let frames0 = [fh.clone()];
    let plan0 = TemporalUnitPlan {
        seq: &seq,
        emit_sequence_header: true,
        frames: &frames0,
    };
    let tu0 = encode_temporal_unit(&plan0);

    let frames1 = [fh.clone()];
    let plan1 = TemporalUnitPlan {
        seq: &seq,
        emit_sequence_header: false,
        frames: &frames1,
    };
    let tu1 = encode_temporal_unit(&plan1);

    let mut buf: Vec<u8> = Vec::new();
    {
        let cursor = Cursor::new(&mut buf);
        let mut iw = IvfWriter::new(cursor, FOURCC_AV01, 16, 16, 25, 1).expect("IVF header writes");
        iw.write_frame(&tu0, 0).unwrap();
        iw.write_frame(&tu1, 1).unwrap();
        iw.patch_frame_count().unwrap();
    }

    // Frame 0: TU0 (TD + SH + FH).
    let off0 = IVF_FILE_HEADER_LEN;
    let size0 =
        u32::from_le_bytes([buf[off0], buf[off0 + 1], buf[off0 + 2], buf[off0 + 3]]) as usize;
    let tu0_round = &buf[off0 + IVF_FRAME_HEADER_LEN..off0 + IVF_FRAME_HEADER_LEN + size0];
    assert_eq!(tu0_round, &tu0[..]);
    let descs0: Vec<_> = ObuIter::new(tu0_round).collect::<Result<_, _>>().unwrap();
    assert_eq!(descs0.len(), 3);
    assert_eq!(descs0[0].obu_type, ObuType::TemporalDelimiter);
    assert_eq!(descs0[1].obu_type, ObuType::SequenceHeader);
    assert_eq!(descs0[2].obu_type, ObuType::FrameHeader);

    // Frame 1: TU1 (TD + FH).
    let off1 = off0 + IVF_FRAME_HEADER_LEN + size0;
    let size1 =
        u32::from_le_bytes([buf[off1], buf[off1 + 1], buf[off1 + 2], buf[off1 + 3]]) as usize;
    let tu1_round = &buf[off1 + IVF_FRAME_HEADER_LEN..off1 + IVF_FRAME_HEADER_LEN + size1];
    assert_eq!(tu1_round, &tu1[..]);
    let descs1: Vec<_> = ObuIter::new(tu1_round).collect::<Result<_, _>>().unwrap();
    assert_eq!(descs1.len(), 2);
    assert_eq!(descs1[0].obu_type, ObuType::TemporalDelimiter);
    assert_eq!(descs1[1].obu_type, ObuType::FrameHeader);

    // The shared frame header should reparse identically against the
    // (cached) sequence header.
    let reparsed = parse_frame_header(descs1[1].payload, &seq).unwrap();
    let mut expected = fh.clone();
    expected.bits_consumed = reparsed.bits_consumed;
    assert_eq!(reparsed, expected);
}
