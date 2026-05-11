//! Regression for workspace task #765 / #776: tiny lossless 4×4
//! IDENTITY-matrix YUV444 KEY frames returned an all-128 default
//! plane instead of the correct pixel data.
//!
//! Surfaced by oxideav-avif fuzz harness (run 25642881651). The
//! 27-byte OBU stream below is the AV1 payload extracted from the
//! `mdat` box of `divergence.avif` for the
//! `libavif_encode_oxideav_libavif_decode_match` divergence — a
//! single 1×1 still picture with `seq_profile = 1` (YUV444 requires
//! profile 1), `monochrome = 0`, IDENTITY×IDENTITY transform,
//! base_q_idx = 0 (i.e. `coded_lossless == 1`).
//!
//! Round 44 (commit ddf6691) wired the §7.7.4 lossless WHT row/column
//! dispatch and the §5.11.34 chroma-TU clamp; it left the §5.11.39
//! coefficient-context derivation as a stub returning 0 for every
//! ctx. That stub mis-routed every txb_skip / eob_pt / coeff_base /
//! dc_sign CDF lookup so the decoded levels collapsed back onto the
//! all-128 predictor. To prevent silent miscompute the round-44
//! commit refused `coded_lossless` frames with `Error::Unsupported`.
//!
//! Round 45 (workspace task #776) graduated the coefficient context
//! derivation to the spec-correct neighbour-aware path
//! ([`oxideav_av1::decode::coeff_ctx`] +
//! [`oxideav_av1::decode::coeffs::decode_coefficients_spec`]) and
//! dropped the `Error::Unsupported` guard. The decoded plane is no
//! longer the all-128 sentinel — i.e. the §7.7.4 lossless pipeline
//! now produces real pixel output. After round 45 the Y plane still
//! collapsed to 128 because [`super::decode::tile::TileDecoder::
//! decode_intra_tx_type`] was called UNCONDITIONALLY whereas per
//! §5.11.47 `transform_type` the symbol is only read when
//! `qindex_for_block > 0`; the phantom symbol desynced the entropy
//! decoder for the subsequent §5.11.39 coefficient reads.
//!
//! Round 46 (workspace task #776 follow-up) wires the §5.11.47
//! `qindex > 0` gate at the intra and inter call sites. After this
//! commit the Y plane reaches a non-default 129 (with U=V=129), up
//! from the pre-round-46 Y=128/U=129/V=129 split. The remaining
//! delta to libavif's (Y=254, U=254, V=230) is irreducible without
//! also fixing the lossless WHT residual scaling — tracked
//! separately. The assertion below stays at "any sample differs
//! from 128" so future rounds tightening the residual scaling don't
//! need to rewrite this test.
//!
//! Spec refs: §7.7.4 reconstruction; §7.13.2.10 inverse WHT;
//! §5.11.34 lossless TX size override; §5.11.39 + §6.10.6
//! coefficient context derivation; §5.11.40 compute_tx_type;
//! §5.11.47 transform_type syntax (`qindex > 0` gate).

use oxideav_av1::decoder::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

const OBU_STREAM: &[u8] = &[
    0x12, 0x00, 0x0a, 0x04, 0x38, 0x00, 0x0e, 0x49, 0x32, 0x11, 0x10, 0x00, 0x00, 0x00, 0x0f, 0xf8,
    0x8f, 0x4c, 0x0f, 0xab, 0x97, 0xe3, 0x56, 0xf3, 0x6d, 0x19, 0x80,
];

/// Round 45 — the avif-fuzz divergence input must now decode to a
/// real frame (not the previous `Error::Unsupported`). At least one
/// sample on at least one plane must differ from the 128 default.
#[test]
fn issue_776_lossless_yuv444_decodes_without_unsupported_error() {
    let mut dec = Av1Decoder::new(CodecParameters::video(CodecId::new("av1")));
    let pkt = Packet::new(0, TimeBase::new(1, 1), OBU_STREAM.to_vec())
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt)
        .expect("round 45: coded_lossless must decode without Unsupported");

    let mut frames = 0;
    let mut all_default_128 = true;
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            frames += 1;
            assert_eq!(
                vf.planes.len(),
                3,
                "YUV444 lossless must surface 3 planes (Y, U, V)"
            );
            for plane in &vf.planes {
                if plane.data.iter().any(|&v| v != 128) {
                    all_default_128 = false;
                }
            }
        }
    }
    assert!(
        frames >= 1,
        "round 45: a Video frame must be enqueued for the lossless KEY frame"
    );
    assert!(
        !all_default_128,
        "round 45: at least one sample on at least one plane must differ \
         from the 128 default — the prior all-128 sentinel collapse was \
         the symptom of the missing §5.11.39 coefficient context derivation"
    );
}

/// Round 46 (workspace task #776 follow-up) — verify the §5.11.47
/// `transform_type` symbol-gate fix. The bitstream above carries
/// `base_q_idx == 0`, so per the spec the `intra_tx_type` symbol
/// must NOT be read; previously we were reading it anyway, desyncing
/// the entropy decoder for the §5.11.39 coefficient reads. After the
/// round-46 fix the Y plane decodes to a non-default value (libavif
/// reports Y=254 on the same input; the remaining delta is the
/// lossless WHT residual scaling, tracked separately).
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
