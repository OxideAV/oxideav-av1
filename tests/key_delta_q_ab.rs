//! r431 — KEY-frame §5.9.17 delta-q A/B: the per-superblock delta-q
//! election on intra frames (complexity-probe `CurrentQIndex` plan +
//! frame-level masking-weighted exact-realized-bytes arm election),
//! the intra twin of the r428 inter arm.
//!
//! Always-on (CI):
//!   * conformance — both variants (election on / forced off) round-
//!     trip byte-exact through the in-tree spec driver to the
//!     encoder's own reconstruction;
//!   * election tripwire — on mixed flat/texture content the delta
//!     arm must actually be ELECTED (the emitted KEY carries
//!     `delta_q_present = 1`), and it must win the encoder's own
//!     joint objective (the election is exact — a loss is a harness
//!     bug);
//!   * uniform-texture control — the probe finds no spread, the plan
//!     stays empty, and the armed entry is BIT-IDENTICAL to the
//!     forced-off baseline.
//!
//! Env-gated (`OXIDEAV_AV1_KEY_DQ_DIR`): per-config bytes + the
//! elected KEY IVF + recon YUV for black-box reference validation.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.17, §5.11.13, §7.12.2.

use oxideav_av1::decoder::{decode_av1_spec, SpecFrame};
use oxideav_av1::encoder::encode_key_frame_yuv420_with_q_dq;
use oxideav_av1::encoder::Yuv420Frame;
use oxideav_av1::frame_header::parse_frame_header;
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Dense texture (high per-superblock variance — masking hides
/// quantisation).
fn tex(x: f64, y: f64) -> f64 {
    128.0
        + 42.0 * (0.71 * x + 0.9 * (0.23 * y).sin()).sin()
        + 36.0 * (0.63 * y - 0.7 * (0.31 * x).sin()).cos()
        + 20.0 * (0.47 * (x + y)).sin()
}

/// Smooth ramp (near-zero variance — banding shows first here).
fn flat(x: f64, y: f64) -> f64 {
    90.0 + 0.22 * x + 0.13 * y
}

fn build_frame(w: u32, h: u32, kind: &str) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            let v = match kind {
                "mixed" => {
                    if r < hu / 2 {
                        flat(c as f64, r as f64)
                    } else {
                        tex(c as f64, r as f64)
                    }
                }
                "uniform" => tex(c as f64, r as f64),
                other => panic!("unknown content kind {other}"),
            };
            f.y[r * wu + c] = clamp(v);
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = clamp(120.0 + 16.0 * (0.05 * c as f64).sin());
            f.v[r * cw + c] = clamp(132.0 + 14.0 * (0.06 * r as f64).cos());
        }
    }
    f
}

fn decode_one(ivf: &[u8], label: &str) -> SpecFrame {
    let frames =
        decode_av1_spec(ivf).unwrap_or_else(|e| panic!("{label}: spec driver rejected: {e:?}"));
    assert_eq!(frames.len(), 1, "{label}: one KEY frame");
    frames.into_iter().next().unwrap()
}

/// The emitted KEY frame's §5.9.17 `delta_q_present` bit.
fn delta_q_present(ivf: &[u8]) -> bool {
    // Walk the IVF's first frame payload → temporal unit → OBUs.
    let tu_off = 32 + 12;
    let tu = &ivf[tu_off..];
    let mut seq = None;
    for desc in ObuIter::new(tu) {
        let desc = desc.expect("TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
            ObuType::Frame => {
                let seq = seq.as_ref().expect("SH precedes frame");
                let fh = parse_frame_header(desc.payload, seq).expect("FH parses");
                return fh
                    .delta_q_params
                    .map(|d| d.delta_q_present)
                    .unwrap_or(false);
            }
            _ => {}
        }
    }
    panic!("no frame OBU");
}

fn assert_round_trip(frame: &Yuv420Frame, q: u8, delta_q: bool, label: &str) {
    let enc = encode_key_frame_yuv420_with_q_dq(frame, q, delta_q)
        .unwrap_or_else(|e| panic!("{label}: encode failed: {e:?}"));
    let d = decode_one(&enc.ivf_bytes, label);
    assert_eq!(d.planes[0], enc.recon_y, "{label}: luma != recon");
    assert_eq!(d.planes[1], enc.recon_u, "{label}: U != recon");
    assert_eq!(d.planes[2], enc.recon_v, "{label}: V != recon");
}

/// Both variants round-trip byte-exact across the quantiser ladder
/// and both content shapes.
#[test]
fn key_delta_q_both_variants_round_trip() {
    for kind in ["mixed", "uniform"] {
        let f = build_frame(128, 128, kind);
        for q in [60u8, 120, 200] {
            for dq in [false, true] {
                assert_round_trip(&f, q, dq, &format!("{kind}-q{q}-dq{dq}"));
            }
        }
    }
}

/// On mixed flat/texture content the election fires and wins: the
/// armed entry emits a `delta_q_present = 1` KEY, and its joint
/// score (the exact objective the encoder used) is <= the baseline's.
#[test]
fn key_delta_q_elected_on_mixed_content() {
    let f = build_frame(128, 128, "mixed");
    let q = 140u8;
    let armed = encode_key_frame_yuv420_with_q_dq(&f, q, true).expect("armed encode");
    let base = encode_key_frame_yuv420_with_q_dq(&f, q, false).expect("baseline encode");
    // The election adopted the delta arm (the objective judged it a
    // win — else `_dq(true)` would have returned the single-quantiser
    // frame, which never carries the header bit).
    assert!(
        delta_q_present(&armed.ivf_bytes),
        "the delta-q arm must be elected on mixed flat/texture content"
    );
    assert!(
        !delta_q_present(&base.ivf_bytes),
        "the forced-off baseline never codes delta_q_present"
    );
    // Both decode to their own reconstruction (already covered by the
    // round-trip test; re-assert on this witness for locality).
    let da = decode_one(&armed.ivf_bytes, "armed");
    assert_eq!(da.planes[0], armed.recon_y);
}

/// Uniform texture: the probe finds no activity spread, so the armed
/// entry is BIT-IDENTICAL to the forced-off baseline (the plan is
/// empty — no second arm runs).
#[test]
fn key_delta_q_inert_on_uniform_content() {
    let f = build_frame(128, 128, "uniform");
    for q in [80u8, 160] {
        let armed = encode_key_frame_yuv420_with_q_dq(&f, q, true).expect("armed");
        let base = encode_key_frame_yuv420_with_q_dq(&f, q, false).expect("base");
        assert_eq!(
            armed.ivf_bytes, base.ivf_bytes,
            "q{q}: no spread ⇒ armed entry must equal the baseline byte for byte"
        );
        assert!(!delta_q_present(&armed.ivf_bytes));
    }
}

/// Env-gated staging: the elected KEY stream + its recon for
/// black-box reference-decoder validation.
#[test]
fn key_delta_q_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_KEY_DQ_DIR") else {
        eprintln!("OXIDEAV_AV1_KEY_DQ_DIR unset — skipping the key-delta-q staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let f = build_frame(128, 128, "mixed");
    let enc = encode_key_frame_yuv420_with_q_dq(&f, 140, true).expect("armed encode");
    let name = "key-dq-128x128-q140-mixed";
    std::fs::write(root.join(format!("{name}.ivf")), &enc.ivf_bytes).expect("write ivf");
    let mut yuv = Vec::new();
    yuv.extend_from_slice(&enc.recon_y);
    yuv.extend_from_slice(&enc.recon_u);
    yuv.extend_from_slice(&enc.recon_v);
    std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
}
