//! `oxideav-core` registry-path integration tests for the AV1 decoder.
//!
//! Proves the `RuntimeContext` surface wired in `src/registry.rs`:
//!
//! 1. `register` installs a decoder factory for codec id `av1`.
//! 2. The three container identifiers resolve to `av1` (ISOBMFF `av01` /
//!    IVF `AV01` FourCC, Matroska `V_AV1`).
//! 3. Real validator-produced conformance streams decode through the
//!    `Decoder` trait byte-identical to the independent third-party
//!    decoder's output — both fed as one whole IVF packet and split one
//!    temporal unit per packet (the container framing), proving the
//!    §7.20 reference / CDF session state carries ACROSS packets.
//!
//! As of r394 the registered decoder is the spec-faithful frame driver
//! (`decoder::SpecDecodeSession` / `decode_av1_spec`) — the full
//! conformance-validated surface, not the historical intra-only
//! encoder-mirror path.

use oxideav_av1::registry::{make_decoder, register, register_codecs, CODEC_ID_STR};

use oxideav_core::{
    CodecId, CodecParameters, CodecRegistry, CodecTag, Error as CoreError, Frame, Packet,
    ProbeContext, RuntimeContext, TimeBase,
};

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0);
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// `tiny-i-only-16x16-prof0` — the conformance corpus\' smallest stream
/// (one 16×16 profile-0 KEY frame) and its independent-decoder pixels.
/// Same bytes as `tests/fixture_conformance.rs`.
const TINY_I_ONLY_16X16_IVF: &str = concat!(
    "444b494600002000415630311000100019000000010000000100000000000000240000000000",
    "00000000000012000a0a000000019ffbfff3008032141000bc00000240000000789d762f676c",
    "c7ee5180",
);

const TINY_I_ONLY_16X16_EXPECTED_YUV: &str = concat!(
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "515151515151515151515151515151515151515151515151515151515a5a5a5a5a5a5a5a5a5a",
    "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
    "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5aefefefefefefefefefefefefefefefefefefefefefef",
    "efefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefef",
    "efefefef",
);

/// `i-frame-then-p-64x64` — KEY + one INTER frame (ref-frame setup,
/// primary-ref CDF forwarding). Same bytes as
/// `tests/fixture_conformance.rs` (where the output is pinned against
/// the independent decoder by SHA-256); here the two feeding shapes
/// must agree with each other.
const I_FRAME_THEN_P_64X64_IVF: &str = concat!(
    "444b494600002000415630314000400019000000010000000200000000000000c40100000000",
    "00000000000012000a0a00000002afffbfff300832b30314002be000094209002c2ae3306dea",
    "a663e815f2a46f7966185d3d3166537fff70eb8d8d007775419b7e1f5c93578ef2da07f4a1e2",
    "0a44effeb51cb778bab9463a362e16ebb0f50fac519a0cbbef8972bc6ce63448f4b21ed3c1a9",
    "a187f2d62f574dfd39d95bce8250e9f5091954b87f9bf3e10106db9a6dde506f691a7443a70b",
    "c10820836d57ea8c0b91296deb2262dad2f6a1249f2f0b2e7b83f6d00d441198e0126e2a621f",
    "82ebff8e9c0788a0ecfaba1b108672a99300f80f176dc6c5d0ca01e0ace38b251c3ef9e30192",
    "fea8ddf08711f0807d2590d252ac011fcfb76aa025e6d1dc5dcb29b5e177e52b21c7c12b8c94",
    "01fe8ed3cf16e12fc3d5bf272b8b5641612a46dcc42bc0c29e3633cde9101d95816607349124",
    "4a19e7f8fd5f09cd45dbae1c1a1ba8cd5652cbacfdbe7f9770755baf050baaf675bab833f7c5",
    "14a75d3d2471cf8d7157accffdfe810e67b0f4dc097fb705c54ad9454774e4535ae6d53aa81c",
    "393648af1cc10cd7b71f35b33cee1b554e5be3ac29d8569665a7948136221d8a5753993fa731",
    "c8a06bafe71783a032c396b4947303c544838065c23ca41e43476feb40645d38a76cf4b90ded",
    "6b141a0000000100000000000000120032163201e0400000235e000012841200000400d00a2d",
    "168b"
);

/// The registered decoder must decode a real external KEY frame
/// byte-identical to the independent decoder — fed as one whole IVF
/// buffer packet.
#[test]
fn tiny_intra_stream_decodes_through_trait_surface() {
    let ivf = unhex(TINY_I_ONLY_16X16_IVF);
    let expected = unhex(TINY_I_ONLY_16X16_EXPECTED_YUV);

    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("factory builds a decoder");

    let pkt = Packet::new(0, TimeBase::new(1, 1), ivf);
    dec.send_packet(&pkt).unwrap();

    let frame = dec.receive_frame().expect("one frame out");
    let Frame::Video(vf) = frame else {
        panic!("expected a video frame");
    };
    assert_eq!(vf.planes.len(), 3, "4:2:0 emits three planes");
    assert_eq!(vf.planes[0].stride, 16);
    assert_eq!(vf.planes[1].stride, 8);
    assert_eq!(vf.planes[2].stride, 8);
    let got: Vec<u8> = vf
        .planes
        .iter()
        .flat_map(|p| p.data.iter().copied())
        .collect();
    assert_eq!(got, expected, "trait-surface decode is byte-exact");
}

/// Decode the KEY + INTER GOP twice — once as ONE whole-IVF packet,
/// once split one temporal unit per packet — and require identical
/// output. The split feed only decodes correctly if the §7.20
/// reference store, the §8.3.1 primary-ref CDF forwarding, and the
/// cached sequence header survive across `send_packet` calls.
#[test]
fn inter_gop_decodes_across_split_packets() {
    let ivf = unhex(I_FRAME_THEN_P_64X64_IVF);
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let tb = TimeBase::new(1, 1);

    // (a) whole IVF buffer in one packet.
    let mut whole = make_decoder(&params).unwrap();
    whole.send_packet(&Packet::new(0, tb, ivf.clone())).unwrap();
    let mut whole_frames: Vec<Vec<Vec<u8>>> = Vec::new();
    while let Ok(Frame::Video(vf)) = whole.receive_frame() {
        whole_frames.push(vf.planes.into_iter().map(|p| p.data).collect());
    }
    assert_eq!(whole_frames.len(), 2, "KEY + INTER = two shown frames");

    // (b) one temporal unit per packet: split the IVF into its frame
    // records by hand (12-byte record headers after the 32-byte file
    // header) and feed each record's payload as a raw-TU packet.
    let mut split = make_decoder(&params).unwrap();
    let mut split_frames: Vec<Vec<Vec<u8>>> = Vec::new();
    let mut pos = 32usize;
    let mut n_records = 0usize;
    while pos + 12 <= ivf.len() {
        let size = u32::from_le_bytes(ivf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        let payload = ivf[pos..pos + size].to_vec();
        pos += size;
        n_records += 1;
        split.send_packet(&Packet::new(0, tb, payload)).unwrap();
        while let Ok(Frame::Video(vf)) = split.receive_frame() {
            split_frames.push(vf.planes.into_iter().map(|p| p.data).collect());
        }
    }
    assert_eq!(n_records, 2, "fixture carries two IVF records");
    assert_eq!(
        split_frames, whole_frames,
        "per-temporal-unit packets must decode identically to the whole buffer \
         (cross-packet §7.20 session state)"
    );
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
fn drained_decoder_reports_need_more_then_eof() {
    let ivf = unhex(TINY_I_ONLY_16X16_IVF);
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();

    // Before any packet: NeedMore.
    assert!(matches!(
        dec.receive_frame().unwrap_err(),
        CoreError::NeedMore
    ));

    let pkt = Packet::new(0, TimeBase::new(1, 1), ivf);
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
    // Garbage that is neither a valid IVF buffer nor a decodable OBU
    // temporal unit must surface as InvalidData, not panic. (A frame
    // header with no preceding sequence header is the failure the
    // spec driver reports here.)
    let pkt = Packet::new(0, TimeBase::new(1, 1), vec![0x32, 0x01, 0x10, 0x00]);
    let err = dec.send_packet(&pkt).expect_err("garbage TU rejected");
    assert!(matches!(err, CoreError::InvalidData(_)));
}

/// `reset` drops the reference store and any queued frames; a KEY
/// stream re-fed after the reset decodes from scratch.
#[test]
fn reset_clears_queue_and_recovers() {
    let ivf = unhex(TINY_I_ONLY_16X16_IVF);
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).unwrap();

    let pkt = Packet::new(0, TimeBase::new(1, 1), ivf);
    dec.send_packet(&pkt).unwrap();
    // Undrained frame dropped by the seek.
    dec.reset().unwrap();
    assert!(matches!(
        dec.receive_frame().unwrap_err(),
        CoreError::NeedMore
    ));
    // The landing KEY frame decodes cleanly on the reset session.
    dec.send_packet(&pkt).unwrap();
    assert!(matches!(dec.receive_frame(), Ok(Frame::Video(_))));
}
