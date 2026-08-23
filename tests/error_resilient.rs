//! r450 — §5.9.2 CODED `error_resilient_mode` on plain INTER frames.
//!
//! Unlike a SWITCH frame (r447), where §5.9.2 INFERS the flag
//! bit-free, a regular INTER frame codes `error_resilient_mode` as
//! an f(1) — and everything else about the frame stays ordinary: the
//! frame type, the normal single-slot `refresh_frame_flags`, the
//! reference SAMPLES predicting across the boundary, and every
//! frame-level election. What changes is the cross-frame DECODE
//! state: §5.9.2 infers `primary_ref_frame = PRIMARY_REF_NONE` (the
//! f(3) is not coded — per-frame default CDFs +
//! `setup_past_independence`), `use_ref_frame_mvs = 0` and
//! `allow_warped_motion = 0` (neither f(1) coded), the
//! `ref_order_hint[ i ]` block goes on the wire (each coded hint
//! must equal the decoder's stored `RefOrderHint[ i ]`, else the
//! slot is marked invalid), and the writer bypasses
//! `frame_size_with_refs`.
//!
//! Coverage:
//! * a GOP with an `error_resilient_period` cadence decodes
//!   byte-exact through the spec driver (lossy and lossless),
//! * the emitted headers carry the coded-flag shape (parsed back off
//!   the wire): INTER type, normal refresh, `PRIMARY_REF_NONE`,
//!   `use_ref_frame_mvs = 0`, `allow_warped_motion = 0`, true
//!   `ref_order_hint[]` state,
//! * the cadence composes with the SWITCH cadence and with the
//!   tile/election surfaces,
//! * the staging dump feeds black-box reference-decoder validation
//!   and corpus pinning.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.2.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning, Yuv420Frame};
use oxideav_av1::frame_header::{
    FrameHeader, FrameType, RefInfo, ALL_FRAMES_PUB, PRIMARY_REF_NONE,
};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Deterministic textured frame with per-frame translation.
fn moving_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            let (si, sj) = (i + t, j + 2 * t);
            f.y[i * wu + j] = ((si * 5 + sj * 3 + (si / 16) * (sj / 16)) % 256) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = ((128 + i * 2 + j + t) % 256) as u8;
            f.v[i * cw + j] = ((64 + i + j * 2 + 2 * t) % 256) as u8;
        }
    }
    f
}

fn tuned(error_resilient_period: u32, s_frame_period: u32, tiles: (u32, u32)) -> GopTuning {
    GopTuning {
        error_resilient_period,
        s_frame_period,
        tiles,
        ..GopTuning::default()
    }
}

/// Encode with the cadence, decode through the spec driver, assert
/// byte-exactness against the encoder recon (and the input at q 0).
fn assert_er_round_trip(frames: &[Yuv420Frame], q: u8, tuning: GopTuning) {
    let enc = encode_gop_yuv420_with_q_seg_tuned(frames, q, &[], tuning).expect("ER GOP encodes");
    let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("spec driver decodes own ER GOP");
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.gop.recon[idx];
        assert_eq!(f.planes[0], rc.y, "frame {idx}: luma decode != recon");
        assert_eq!(f.planes[1], rc.u, "frame {idx}: U decode != recon");
        assert_eq!(f.planes[2], rc.v, "frame {idx}: V decode != recon");
        if q == 0 {
            assert_eq!(f.planes[0], frames[idx].y, "lossless frame {idx} luma");
            assert_eq!(f.planes[1], frames[idx].u, "lossless frame {idx} U");
            assert_eq!(f.planes[2], frames[idx].v, "lossless frame {idx} V");
        }
    }
}

/// Parse every coded frame header off the wire, in temporal-unit
/// order.
fn wire_headers(temporal_units: &[Vec<u8>]) -> Vec<FrameHeader> {
    let mut seq = None;
    let refinfo = RefInfo::default();
    let mut out = Vec::new();
    for tu in temporal_units {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("own stream walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    out.push(
                        oxideav_av1::frame_header::parse_frame_header_with_refs(
                            desc.payload,
                            seq.as_ref().expect("SH precedes frames"),
                            &refinfo,
                        )
                        .expect("frame header parses"),
                    );
                }
                _ => {}
            }
        }
    }
    out
}

/// A 7-frame GOP with ER frames at display positions 2, 4 and 6
/// decodes byte-exact, lossy and lossless.
#[test]
fn error_resilient_cadence_round_trips_byte_exact() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 80, t)).collect();
    assert_er_round_trip(&frames, 72, tuned(2, 0, (0, 0)));
    let small: Vec<Yuv420Frame> = (0..5).map(|t| moving_frame(64, 64, t)).collect();
    assert_er_round_trip(&small, 0, tuned(2, 0, (0, 0)));
}

/// The emitted ER headers carry the coded-flag shape: everything a
/// SWITCH frame infers, EXCEPT the frame stays a plain INTER frame
/// with the ordinary single-slot refresh — and the flag itself is a
/// real bit.
#[test]
fn error_resilient_headers_carry_the_coded_shape() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 80, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 80, &[], tuned(2, 0, (0, 0)))
        .expect("ER GOP encodes");
    let headers = wire_headers(&enc.gop.temporal_units);
    assert_eq!(headers.len(), 7);
    for (idx, fh) in headers.iter().enumerate() {
        if idx == 0 {
            assert_eq!(fh.frame_type, FrameType::Key);
            continue;
        }
        assert_eq!(fh.frame_type, FrameType::Inter, "frame {idx} stays INTER");
        assert_ne!(
            fh.refresh_frame_flags, ALL_FRAMES_PUB,
            "frame {idx} keeps the ordinary refresh"
        );
        let refs = fh.inter_refs.as_ref().expect("inter frame carries refs");
        if idx % 2 == 0 {
            assert!(fh.error_resilient_mode, "frame {idx} codes the f(1) as 1");
            assert_eq!(
                fh.primary_ref_frame, PRIMARY_REF_NONE,
                "frame {idx}: §5.9.2 infers PRIMARY_REF_NONE (f(3) not coded)"
            );
            assert!(
                !refs.use_ref_frame_mvs,
                "frame {idx}: use_ref_frame_mvs inferred 0"
            );
            assert_eq!(
                fh.allow_warped_motion,
                Some(false),
                "frame {idx}: allow_warped_motion inferred 0"
            );
            // The §5.9.2 ref_order_hint block is CODED under error
            // resilience: the KEY (hint 0) still holds every slot
            // outside the two-slot rotation; the rotation slots hold
            // the two most recent frames.
            let hints = fh.ref_order_hints.expect("coded ref_order_hint block");
            let p = idx as u32;
            let mut expect = [0u32; 8];
            expect[(p & 1) as usize] = p - 1;
            expect[((p - 1) & 1) as usize] = p.saturating_sub(2);
            assert_eq!(hints, expect, "frame {idx}: true slot hints on the wire");
        } else {
            assert!(!fh.error_resilient_mode, "frame {idx} keeps the flag off");
            assert_ne!(
                fh.primary_ref_frame, PRIMARY_REF_NONE,
                "frame {idx} keeps its primary-reference election"
            );
        }
    }
}

/// The ER cadence composes with the SWITCH cadence: S-frame
/// positions keep the inferred SWITCH shape, ER positions the coded
/// INTER shape, and the whole stream decodes byte-exact.
#[test]
fn error_resilient_composes_with_s_frames() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 64, t)).collect();
    let tuning = tuned(2, 3, (0, 0));
    assert_er_round_trip(&frames, 100, tuning);
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 100, &[], tuning).expect("encodes");
    let headers = wire_headers(&enc.gop.temporal_units);
    // Positions: 1 plain, 2 ER, 3 SWITCH, 4 ER, 5 plain, 6 SWITCH.
    let shapes: Vec<(FrameType, bool, u8)> = headers
        .iter()
        .map(|fh| {
            (
                fh.frame_type,
                fh.error_resilient_mode,
                fh.refresh_frame_flags,
            )
        })
        .collect();
    assert_eq!(shapes[3].0, FrameType::Switch);
    assert_eq!(shapes[6].0, FrameType::Switch);
    assert_eq!(shapes[3].2, ALL_FRAMES_PUB);
    for idx in [2usize, 4] {
        assert_eq!(shapes[idx].0, FrameType::Inter, "frame {idx}");
        assert!(shapes[idx].1, "frame {idx} codes the flag");
        assert_ne!(
            shapes[idx].2, ALL_FRAMES_PUB,
            "frame {idx} ordinary refresh"
        );
    }
    for idx in [1usize, 5] {
        assert_eq!(shapes[idx].0, FrameType::Inter, "frame {idx}");
        assert!(!shapes[idx].1, "frame {idx} keeps the flag off");
    }
}

/// The cadence composes with the multi-tile layout and the
/// default-on frame-level election surfaces.
#[test]
fn error_resilient_composes_with_tiles_and_elections() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| moving_frame(128, 64, t)).collect();
    assert_er_round_trip(&frames, 100, tuned(2, 0, (1, 0)));
}

/// Env-gated staging dump (`OXIDEAV_AV1_ERM_DIR`): the ER-cadence
/// GOP plus expected YUV for black-box reference-decoder validation
/// and corpus pinning. Inert otherwise.
#[test]
fn erm_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_ERM_DIR") else {
        eprintln!("OXIDEAV_AV1_ERM_DIR unset — skipping the error-resilient staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 80, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 72, &[], tuned(2, 0, (0, 0)))
        .expect("ER GOP encodes");
    std::fs::write(root.join("gop-96x80-q72-erm.ivf"), &enc.gop.ivf_bytes).expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &enc.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("gop-96x80-q72-erm.yuv"), &yuv).expect("write yuv");
}
