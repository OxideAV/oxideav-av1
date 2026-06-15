//! `oxideav-core` registry-path integration tests for the AV1 decoder.
//!
//! Proves the `RuntimeContext` surface wired in `src/registry.rs`:
//!
//! 1. `register` installs a decoder factory for codec id `av1`.
//! 2. The three container identifiers resolve to `av1` (ISOBMFF `av01` /
//!    IVF `AV01` FourCC, Matroska `V_AV1`).
//! 3. A known intra AV1 fixture — the same encoder-produced IVF buffer
//!    the crate's `decode_av1` round-trip tests use — decodes through the
//!    `Decoder` trait, recovering the input planes byte-for-byte.
//!
//! The registered decoder is intra-only; this exercises the reachable
//! subset of that surface, not the full AV1 feature set.

use oxideav_av1::encoder::{encode_intra_frame_yuv, Yuv420Frame16x16};
use oxideav_av1::registry::{make_decoder, register, register_codecs, CODEC_ID_STR};
use oxideav_av1::{parse_frame_header, parse_sequence_header};

use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, CodecTag, Error as CoreError, Frame, Packet,
    ProbeContext, RuntimeContext, TimeBase,
};

// The §5.5.1 sequence header + §5.9 frame header descriptor bytes the
// crate's existing encode/decode round-trip tests parse to drive the
// 16×16 4:2:0 intra encoder.
const TINY_SEQ_PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
const TINY_FRAME_PAYLOAD: &[u8] = &[
    0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67, 0x6c,
    0xc7, 0xee, 0x51, 0x80,
];

/// Build a known intra AV1 fixture (IVF v0 buffer) plus the input frame
/// it should decode back to. Reuses the encoder driver exactly as the
/// `decode_av1` round-trip tests do.
fn intra_fixture() -> (Vec<u8>, Yuv420Frame16x16) {
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap();
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).unwrap();
    let mut input = Yuv420Frame16x16::default();
    // A horizontal chroma gradient over a flat luma plane — a non-trivial
    // payload that still lands inside the intra-only round-trip scope.
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = 100;
        }
    }
    for row in input.u.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (16 + (j as u8) * 27) % 251;
        }
    }
    for row in input.v.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (232u8).wrapping_sub((j as u8) * 27);
        }
    }
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    (encoded.ivf_bytes, input)
}

#[test]
fn register_installs_decoder_via_runtime_context() {
    let mut ctx = RuntimeContext::new();
    register(&mut ctx);
    assert!(
        ctx.codecs.has_decoder(&CodecId::new(CODEC_ID_STR)),
        "register must install an av1 decoder factory"
    );
}

#[test]
fn container_tags_resolve_to_av1() {
    let mut reg = CodecRegistry::new();
    register_codecs(&mut reg);

    // ISOBMFF sample entry `av01` upper-cases to the IVF FourCC `AV01`.
    let fourcc = CodecTag::fourcc(b"AV01");
    assert_eq!(
        reg.resolve_tag_ref(&ProbeContext::new(&fourcc))
            .map(CodecId::as_str),
        Some(CODEC_ID_STR),
    );

    // Matroska / WebM Codec ID.
    let mkv = CodecTag::matroska("V_AV1");
    assert_eq!(
        reg.resolve_tag_ref(&ProbeContext::new(&mkv))
            .map(CodecId::as_str),
        Some(CODEC_ID_STR),
    );
}

#[test]
fn intra_fixture_decodes_through_trait_surface() {
    let (ivf_bytes, input) = intra_fixture();

    // Resolve the decoder the way a demuxer would: from CodecParameters
    // carrying the resolved codec id, through the factory.
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("factory builds a decoder");

    let pkt = Packet::new(0, TimeBase::new(1, 1), ivf_bytes);
    dec.send_packet(&pkt).unwrap();

    // Drain the single recovered frame.
    let frame = dec.receive_frame().expect("one frame out");
    let Frame::Video(vf) = frame else {
        panic!("expected a video frame");
    };
    assert_eq!(vf.planes.len(), 3, "4:2:0 emits three planes");

    // Plane 0 = luma (16×16), planes 1/2 = chroma (8×8), each tightly
    // packed row-major. Compare against the encoder's input plane-by-
    // plane — the intra path is lossless.
    let mut expected_y = Vec::with_capacity(16 * 16);
    for row in &input.y {
        expected_y.extend_from_slice(row);
    }
    assert_eq!(vf.planes[0].data, expected_y, "luma byte-exact");
    assert_eq!(vf.planes[0].stride, 16);

    let mut expected_u = Vec::with_capacity(8 * 8);
    for row in &input.u {
        expected_u.extend_from_slice(row);
    }
    assert_eq!(vf.planes[1].data, expected_u, "U byte-exact");
    assert_eq!(vf.planes[1].stride, 8);

    let mut expected_v = Vec::with_capacity(8 * 8);
    for row in &input.v {
        expected_v.extend_from_slice(row);
    }
    assert_eq!(vf.planes[2].data, expected_v, "V byte-exact");
    assert_eq!(vf.planes[2].stride, 8);
}

#[test]
fn drained_decoder_reports_need_more_then_eof() {
    let (ivf_bytes, _input) = intra_fixture();
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();

    // Before any packet: NeedMore.
    assert!(matches!(
        dec.receive_frame().unwrap_err(),
        CoreError::NeedMore
    ));

    let pkt = Packet::new(0, TimeBase::new(1, 1), ivf_bytes);
    dec.send_packet(&pkt).unwrap();
    let _ = dec.receive_frame().expect("frame");

    // Queue drained, not yet flushed: NeedMore again.
    assert!(matches!(
        dec.receive_frame().unwrap_err(),
        CoreError::NeedMore
    ));

    // After flush with an empty queue: Eof.
    dec.flush().unwrap();
    assert!(matches!(dec.receive_frame().unwrap_err(), CoreError::Eof));
}

#[test]
fn out_of_scope_packet_surfaces_invalid_data() {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();
    // Garbage that is not a valid IVF buffer must surface as InvalidData,
    // not panic.
    let pkt = Packet::new(0, TimeBase::new(1, 1), vec![0u8; 8]);
    let err = dec.send_packet(&pkt).expect_err("invalid IVF rejected");
    assert!(matches!(err, CoreError::InvalidData(_)));
}
