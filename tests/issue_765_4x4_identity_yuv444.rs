//! Regression for workspace task #765 / #776 / #786: the lossless
//! 4×4 IDENTITY YUV444 KEY frame surfaced by the
//! `libavif_encode_oxideav_libavif_decode_match` fuzz harness on
//! 2026-05-11 (fuzz run `25642881651`, fixture
//! `crates/oxideav-avif/tests/fixtures/fuzz/y_plane_divergence_match.avif`).
//!
//! Round 47 (workspace task #786) audited the WHT/lossless residual
//! path while rebuilding this regression test. Two surprises:
//!
//! 1. The pre-r47 `OBU_STREAM` bytes here did **not** match the
//!    actual `mdat` payload of `y_plane_divergence_match.avif`. The
//!    first 13 bytes (sequence/frame header bytes) matched; the
//!    remaining 14 bytes were unrelated. Round 47 swaps in the real
//!    27-byte OBU stream extracted from the AVIF container.
//! 2. Cross-decoding the corrected OBU stream through `dav1d 1.5.3`
//!    (`dav1d -i divergence.ivf -o out.yuv`) and `avifdec --raw-color`
//!    both yield Y=0x85=133, U=0xC5=197, V=0xD7=215 — **not** the
//!    `(Y, U, V) = (254, 254, 230)` figure the round-46 comment
//!    quoted (which appears to have been a libavif-RGB-converted
//!    value, not the raw YUV plane). The two reference decoders
//!    agree on the bitstream-correct YUV.
//!
//! Round 47 finding on the residual path: with the corrected
//! bitstream `oxideav-av1` decodes to `(Y, U, V) = (130, 128, 128)`.
//! Hand-tracing the §7.13.2.10 WHT through the in-bounds Y TU
//! dequantised coefficient buffer
//! `[12, 4, 0, 0, 8, -4, 0, 0, -4, 8, 0, 0, 0, 0, 0, 0]` (row pass
//! `shift = 2`, column pass `shift = 0` per §7.7.4) produces
//! residual `(0, 0) = 2`, exactly matching the runtime output. The
//! WHT shifts, the `q = DC8[0] = AC8[0] = 4` lossless dequantiser,
//! and the §7.7.4 reconstruct add-without-Round2 step are all
//! spec-correct (pinned by
//! `transform::iwht4::tests::iwht4_2d_divergence_y_tu_matches_spec`).
//! The Y/U/V deltas vs `dav1d` (3 / 69 / 87 LSB) are upstream of
//! the WHT — the §5.11.39 / §9.4 coefficient entropy decoder reads
//! different `level` values than `dav1d` for the same range coder
//! state. That divergence is tracked as a follow-up against
//! `decode_coefficients_spec`'s context / CDF lookups, **not**
//! the WHT path.
//!
//! Spec refs: §7.7.4 reconstruction; §7.13.2.10 inverse WHT;
//! §5.11.34 lossless TX size override; §5.11.39 + §6.10.6
//! coefficient context derivation; §5.11.40 compute_tx_type;
//! §5.11.47 transform_type syntax (`qindex > 0` gate).

use oxideav_av1::decoder::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

/// Real 27-byte OBU stream extracted from the `mdat` box of
/// `crates/oxideav-avif/tests/fixtures/fuzz/y_plane_divergence_match.avif`.
/// `iloc` points at offset `0x11A` with length `0x1b`; the bytes
/// below match that extent verbatim. `dav1d` and `avifdec` both
/// decode this stream to raw YUV `(Y=0x85, U=0xC5, V=0xD7) =
/// (133, 197, 215)` on a 1×1 YUV444 8-bit lossless KEY frame.
const OBU_STREAM: &[u8] = &[
    0x12, 0x00, 0x0a, 0x04, 0x38, 0x00, 0x0e, 0x49, 0x32, 0x11, 0x10, 0x00, 0x00, 0x19, 0xb9, 0xca,
    0xe3, 0x37, 0x39, 0x09, 0x47, 0xd9, 0x6e, 0x65, 0x96, 0x64, 0xaf,
];

/// Round 45 — the avif-fuzz divergence input must decode to a real
/// frame (not the pre-round-45 `Error::Unsupported` refusal that
/// guarded the missing §5.11.39 coefficient context derivation).
#[test]
fn issue_776_lossless_yuv444_decodes_without_unsupported_error() {
    let mut dec = Av1Decoder::new(CodecParameters::video(CodecId::new("av1")));
    let pkt = Packet::new(0, TimeBase::new(1, 1), OBU_STREAM.to_vec())
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt)
        .expect("round 45: coded_lossless must decode without Unsupported");

    let mut frames = 0;
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            frames += 1;
            assert_eq!(
                vf.planes.len(),
                3,
                "YUV444 lossless must surface 3 planes (Y, U, V)"
            );
        }
    }
    assert!(
        frames >= 1,
        "round 45: a Video frame must be enqueued for the lossless KEY frame"
    );
}

/// Round 46 (workspace task #776) — verify the §5.11.47
/// `transform_type` symbol-gate fix. The bitstream carries
/// `base_q_idx == 0`, so per the spec the `intra_tx_type` symbol
/// must NOT be read; previously we were reading it anyway, desyncing
/// the entropy decoder for the §5.11.39 coefficient reads. After the
/// round-46 fix the Y plane decodes to a non-default value
/// (specifically Y=130 — the §7.7.4 WHT residual of `(0, 0) = 2`
/// added to the 128 default predictor for a 1×1 frame).
#[test]
fn issue_776_round46_lossless_y_plane_no_longer_collapses_to_predictor() {
    let mut dec = Av1Decoder::new(CodecParameters::video(CodecId::new("av1")));
    let pkt = Packet::new(0, TimeBase::new(1, 1), OBU_STREAM.to_vec())
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt).expect("decode");

    let mut y_sample = None;
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            y_sample = Some(vf.planes[0].data[0]);
            break;
        }
    }
    let y = y_sample.expect("a video frame must be produced");
    assert_ne!(
        y, 128,
        "round 46: Y plane must NOT collapse to the 128 predictor \
         default — phantom `intra_tx_type` symbol read in violation \
         of §5.11.47's `qindex > 0` gate previously desynced the \
         entropy decoder, leaving every Y coefficient at zero"
    );
}

#[test]
fn issue_765_obu_stream_parses_to_lossless_yuv444_key_frame() {
    use oxideav_av1::dpb::Dpb;
    use oxideav_av1::frame_header::{parse_frame_obu_with_dpb, FrameType};
    use oxideav_av1::frame_header_tail::coded_lossless_hint;
    use oxideav_av1::obu::{iter_obus, ObuType};
    use oxideav_av1::sequence_header::parse_sequence_header;

    let mut sh = None;
    let mut found_frame = false;
    for obu in iter_obus(OBU_STREAM) {
        let obu = obu.expect("OBU walks");
        match obu.header.obu_type {
            ObuType::SequenceHeader => {
                let parsed = parse_sequence_header(obu.payload).expect("SH parses");
                assert_eq!(parsed.seq_profile, 1, "profile 1 ⇒ YUV444");
                assert_eq!(parsed.color_config.bit_depth, 8);
                assert_eq!(
                    parsed.color_config.num_planes, 3,
                    "YUV444 must surface 3 planes"
                );
                sh = Some(parsed);
            }
            ObuType::Frame => {
                let sh = sh.as_ref().expect("SH precedes Frame");
                let dpb = Dpb::new();
                let (fh, _tg) =
                    parse_frame_obu_with_dpb(sh, obu.payload, &dpb).expect("Frame parses");
                assert_eq!(fh.frame_type, FrameType::Key);
                assert_eq!(fh.frame_width, 1);
                assert_eq!(fh.frame_height, 1);
                assert_eq!(fh.quant.base_q_idx, 0, "lossless ⇒ base_q_idx == 0");
                assert!(
                    coded_lossless_hint(&fh.quant),
                    "frame should be detected as coded_lossless"
                );
                found_frame = true;
            }
            _ => {}
        }
    }
    assert!(found_frame, "OBU stream must contain a Frame OBU");
}
