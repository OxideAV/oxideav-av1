//! Regression for workspace task #765: tiny lossless 4×4 IDENTITY-matrix
//! YUV444 KEY frames returned an all-128 default plane instead of the
//! correct pixel data.
//!
//! Surfaced by oxideav-avif fuzz harness (run 25642881651 on Linux). The
//! 27-byte OBU stream below is the AV1 payload extracted from the
//! `mdat` box of `divergence.avif` for the
//! `libavif_encode_oxideav_libavif_decode_match` divergence — a single
//! 1×1 still picture with `seq_profile = 1` (YUV444 requires profile
//! 1), `monochrome = 0`, IDENTITY×IDENTITY transform, base_q_idx = 0
//! (i.e. `coded_lossless == 1`).
//!
//! The fuzz oracle confirms libavif decodes this to a deterministic
//! non-128 RGBA pixel (`e6 fe fe ff` ⇒ Y=254 U=254 V=230 under the
//! libavif lossless `Y=G U=B V=R` IDENTITY contract); oxideav-av1
//! was silently returning the all-128 default-predictor sentinel
//! because:
//!
//!   1. The §7.7.4 `Lossless` reconstruction branch (which forces
//!      both row and column transforms to the inverse Walsh-Hadamard
//!      kernel per §7.13.2.10) was never dispatched — every TU went
//!      through the regular DCT_DCT path that the
//!      `compute_tx_type` site selects for lossless TUs per §5.11.40.
//!   2. The chroma-block reconstruction (§5.11.34) ignored the
//!      `Lossless ? TX_4X4 : ...` rule and dispatched the spec
//!      chroma block as a single 64×64 TU under YUV444, which the
//!      coefficient decoder cannot consume from a 14-byte tile
//!      payload without underflowing.
//!   3. Even with (1) and (2) wired, the AV1 coefficient decoder's
//!      context derivation (`txb_skip` / `eob_pt` / `coeff_base` /
//!      `dc_sign`) under `coded_lossless == 1` does not yet match
//!      the spec's neighbour-aware `AboveLevelContext` /
//!      `LeftLevelContext` formulae, so the decoded levels are
//!      orders of magnitude smaller than the encoded values and the
//!      reconstructed pixel collapses back onto the predictor.
//!
//! Round-next ships fixes (1) and (2) as `inverse_2d_spec_lossless`
//! plus the chroma TU clamp; (3) is deferred as a separate followup
//! and the decoder refuses `coded_lossless` frames with
//! `Error::Unsupported` until that lands. The avif fuzz harness's
//! documented contract skips on decoder error, so refusing the
//! frame turns the fuzz green without masking real bugs — the day
//! (3) lands, this test flips to assert the decoded pixel values
//! against the libavif oracle.
//!
//! Spec refs: §7.7.4 reconstruction; §7.13.2.10 inverse WHT;
//! §5.11.34 lossless TX size override; §5.11.40 compute_tx_type.

use oxideav_av1::decoder::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Packet, TimeBase};

const OBU_STREAM: &[u8] = &[
    0x12, 0x00, 0x0a, 0x04, 0x38, 0x00, 0x0e, 0x49, 0x32, 0x11, 0x10, 0x00, 0x00, 0x00, 0x0f, 0xf8,
    0x8f, 0x4c, 0x0f, 0xab, 0x97, 0xe3, 0x56, 0xf3, 0x6d, 0x19, 0x80,
];

#[test]
fn issue_765_lossless_yuv444_returns_unsupported_until_coeff_ctx_lands() {
    let mut dec = Av1Decoder::new(CodecParameters::video(CodecId::new("av1")));
    let pkt = Packet::new(0, TimeBase::new(1, 1), OBU_STREAM.to_vec())
        .with_pts(0)
        .with_keyframe(true);
    let err = dec
        .send_packet(&pkt)
        .expect_err("coded_lossless frames must be refused per task #765");
    match err {
        Error::Unsupported(msg) => {
            assert!(
                msg.contains("coded_lossless"),
                "Unsupported error must name `coded_lossless` so the \
                 reason for the refusal is grep-able when this row \
                 appears in CI logs; got {msg:?}"
            );
        }
        other => panic!("expected Error::Unsupported(coded_lossless …), got {other:?}"),
    }
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
