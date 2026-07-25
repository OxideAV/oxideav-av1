//! r429 — loop-restoration A/B harness: measures the §5.9.20 /
//! §5.11.57 / §7.17 per-unit Wiener + self-guided election against
//! the all-RESTORE_NONE baseline (`lr: false` — the pre-r429 shape).
//!
//! Always-on: a conformance A/B (both arms' streams must round-trip
//! byte-exact through the in-tree spec driver — on the armed arm the
//! decode reads the §5.11.58 unit payloads read_lr interleaves into
//! the tile and APPLIES §7.17 per unit, so equality proves the
//! write-side interleave, the recentred-subexp coefficient codes,
//! the tile re-emission and the encoder's §7.17 mirror are all
//! sample-exact), a KEY-header tripwire (detail content under a
//! coarse quantiser must elect a real restoration type), and the
//! measurement tripwire (the armed KEY must beat the baseline on the
//! encoder's own `D + λ·R` scale — guaranteed by the exact-bytes
//! settlement).
//!
//! Env-gated (`OXIDEAV_AV1_LR_DIR=<dir>`): dumps the armed GOP IVF +
//! display-order recon YUV for external black-box decoder validation
//! and corpus pinning.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, encode_key_frame_yuv420_with_q, EncodedGop,
    GopTuning, Yuv420Frame,
};

// ---------------------------------------------------------------------
// Content: fine natural-looking detail everywhere — the quantiser
// noise Wiener / self-guided restoration exists to undo.
// ---------------------------------------------------------------------

fn detail_scene(x: f64, y: f64) -> f64 {
    // Overlapping soft sinusoids at several scales + a slow ramp:
    // wide-band detail with no hard edges (CDEF finds little; LR's
    // whole-unit linear restoration finds a lot).
    120.0
        + 34.0 * (0.13 * x).sin() * (0.11 * y).cos()
        + 22.0 * (0.31 * x + 0.17 * y).sin()
        + 14.0 * (0.53 * x - 0.29 * y).cos()
        + 8.0 * (0.83 * x + 0.71 * y).sin()
        + 0.15 * x
        + 0.1 * y
}

fn build_frame(w: u32, h: u32, k: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let d = 0.6 * k as f64;
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = clamp(detail_scene(c as f64 + d, r as f64 + 0.5 * d));
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            let e = detail_scene(c as f64 * 2.0 + d, r as f64 * 2.0 + 0.5 * d);
            f.u[r * cw + c] = clamp(90.0 + 0.4 * e);
            f.v[r * cw + c] = clamp(150.0 - 0.35 * e);
        }
    }
    f
}

fn detail_content(w: u32, h: u32, n: usize) -> Vec<Yuv420Frame> {
    (0..n).map(|k| build_frame(w, h, k)).collect()
}

// ---------------------------------------------------------------------
// Metrics + encode helpers.
// ---------------------------------------------------------------------

fn psnr(inputs: &[Yuv420Frame], enc: &EncodedGop) -> f64 {
    let (mut sse, mut count) = (0u64, 0u64);
    for (f, rc) in inputs.iter().zip(&enc.recon) {
        for (a, b) in [(&f.y, &rc.y), (&f.u, &rc.u), (&f.v, &rc.v)] {
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = i64::from(x) - i64::from(y);
                sse += (d * d) as u64;
            }
            count += a.len() as u64;
        }
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    10.0 * ((255.0f64 * 255.0 * count as f64) / sse as f64).log10()
}

fn sse(inputs: &[Yuv420Frame], enc: &EncodedGop) -> u64 {
    let mut sse = 0u64;
    for (f, rc) in inputs.iter().zip(&enc.recon) {
        for (a, b) in [(&f.y, &rc.y), (&f.u, &rc.u), (&f.v, &rc.v)] {
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = i64::from(x) - i64::from(y);
                sse += (d * d) as u64;
            }
        }
    }
    sse
}

/// The encoder's own joint objective: `D·256 + λ·8·256·bytes`.
fn score256(inputs: &[Yuv420Frame], enc: &EncodedGop, q: u8) -> u64 {
    let lambda = 1 + u64::from(q) * u64::from(q) / 32;
    let bytes: usize = enc.temporal_units.iter().map(Vec::len).sum();
    sse(inputs, enc) * 256 + lambda * (bytes as u64) * 8 * 256
}

fn encode_arm(frames: &[Yuv420Frame], q: u8, lr: bool) -> EncodedGop {
    encode_gop_yuv420_with_q_seg_extras_tuned(
        frames,
        q,
        &[],
        &[],
        false,
        None,
        GopTuning {
            lr,
            ..GopTuning::default()
        },
    )
    .expect("encode")
    .gop
}

fn assert_round_trips(name: &str, frames: &[Yuv420Frame], enc: &EncodedGop) {
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{name}: spec driver rejected own stream: {e:?}"));
    assert_eq!(decoded.len(), frames.len(), "{name}: frame count");
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.recon[idx];
        assert_eq!(f.planes[0], rc.y, "{name}: frame {idx} luma");
        assert_eq!(f.planes[1], rc.u, "{name}: frame {idx} U");
        assert_eq!(f.planes[2], rc.v, "{name}: frame {idx} V");
    }
}

// ---------------------------------------------------------------------
// Conformance A/B: both arms decode byte-exact via the spec driver.
// ---------------------------------------------------------------------

/// The core correctness gate: on the armed arm the spec driver reads
/// the §5.11.58 unit payloads from the RE-EMITTED tile (selection
/// symbols, recentred-subexp Wiener taps / sgr weights with the
/// running references) and applies §7.17 — recon equality proves the
/// write-side interleave, the re-emission and the encoder's filter
/// mirror are all sample-exact, including the filtered §7.20
/// reference feedback into the P-frames.
#[test]
fn lr_and_baseline_streams_round_trip() {
    let frames = detail_content(192, 128, 4);
    for q in [100u8, 140] {
        let armed = encode_arm(&frames, q, true);
        let flat = encode_arm(&frames, q, false);
        assert_round_trips(&format!("lr-armed q={q}"), &frames, &armed);
        assert_round_trips(&format!("lr-off q={q}"), &frames, &flat);
    }
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_LR_DIR") {
        std::fs::create_dir_all(&dir).unwrap();
        let enc = encode_arm(&frames, 140, true);
        std::fs::write(format!("{dir}/lr-192x128-q140.ivf"), &enc.ivf_bytes).unwrap();
        let mut yuv = Vec::new();
        for rc in &enc.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(format!("{dir}/lr-192x128-q140.yuv"), &yuv).unwrap();
    }
}

// ---------------------------------------------------------------------
// Election tripwires.
// ---------------------------------------------------------------------

/// Detail content under a coarse quantiser must elect loop
/// restoration on the KEY frame — the header codes `UsesLr = 1` with
/// a real restoration type, and the stream still round-trips.
#[test]
fn lr_elected_on_detail_key_header() {
    let input = build_frame(192, 128, 0);
    let k = encode_key_frame_yuv420_with_q(&input, 140).expect("encode");
    let lr = k.fh.lr_params.expect("lossy header carries lr_params");
    assert!(
        lr.uses_lr,
        "detail content must elect loop restoration (got UsesLr = 0)"
    );
    assert_eq!(
        lr.loop_restoration_size[0], 64,
        "r429 codes 64-sample units"
    );
    let decoded = decode_av1_spec(&k.ivf_bytes).expect("spec driver");
    assert_eq!(decoded.len(), 1);
    assert_eq!(decoded[0].planes[0], k.recon_y, "KEY luma");
    assert_eq!(decoded[0].planes[1], k.recon_u, "KEY U");
    assert_eq!(decoded[0].planes[2], k.recon_v, "KEY V");
}

/// The measurement tripwire on the encoder's own `D + λ·R` scale:
/// the armed KEY strictly beats the baseline (the exact-bytes
/// settlement only adopts a plan that does), and the GOP is held to
/// a non-inferiority band (filtered-reference propagation across
/// P-frames is not covered by the per-frame settlement argument).
#[test]
fn lr_beats_baseline_on_detail_content() {
    let frames = detail_content(192, 128, 4);
    let q = 140u8;

    let key_in = &frames[..1];
    let key_on = encode_arm(key_in, q, true);
    let key_off = encode_arm(key_in, q, false);
    let (ks_on, ks_off) = (score256(key_in, &key_on, q), score256(key_in, &key_off, q));
    eprintln!(
        "lr-ab KEY q={q}: armed {:.4} dB / {} B score {ks_on} vs off {:.4} dB / {} B score {ks_off}",
        psnr(key_in, &key_on),
        key_on.temporal_units[0].len(),
        psnr(key_in, &key_off),
        key_off.temporal_units[0].len(),
    );
    assert!(
        ks_on < ks_off,
        "loop restoration must strictly improve the KEY frame's D + lambda*R on detail content \
         ({ks_on} vs {ks_off})"
    );

    let armed = encode_arm(&frames, q, true);
    let flat = encode_arm(&frames, q, false);
    let (s_on, s_off) = (score256(&frames, &armed, q), score256(&frames, &flat, q));
    eprintln!(
        "lr-ab GOP q={q}: armed {:.4} dB / {} B score {s_on} vs off {:.4} dB / {} B score {s_off}",
        psnr(&frames, &armed),
        armed.ivf_bytes.len(),
        psnr(&frames, &flat),
        flat.ivf_bytes.len(),
    );
    assert!(
        s_on as f64 <= s_off as f64 * 1.005,
        "loop restoration regressed the GOP joint score by more than 0.5%: {s_on} vs {s_off}"
    );
}

/// r429 depth-axis lock: the LR election is BitDepth-generic — a
/// 10-bit KEY on the same detail content must elect restoration
/// (§7.17 kernels at the 10-bit clip/rounding parameters, λ at the
/// 4^(BitDepth-8) distortion scale) and round-trip byte-exact
/// through the spec driver.
#[test]
fn lr_elects_and_round_trips_at_10bit() {
    use oxideav_av1::encoder::{encode_key_frame_yuv_with_q, ChromaFormat, YuvFrame};
    let (w, h) = (192u32, 128u32);
    let mut input = YuvFrame::filled(w, h, 10, ChromaFormat::Yuv420, 0);
    let (wu, hu) = (w as usize, h as usize);
    for r in 0..hu {
        for c in 0..wu {
            let v = detail_scene(c as f64, r as f64).clamp(0.0, 255.0);
            input.y[r * wu + c] = ((v * 4.0).round() as u16).min(1023);
        }
    }
    let (cw, ch) = (
        input.chroma_width() as usize,
        input.chroma_height() as usize,
    );
    for r in 0..ch {
        for c in 0..cw {
            let e = detail_scene(c as f64 * 2.0, r as f64 * 2.0);
            input.u[r * cw + c] = (((90.0 + 0.4 * e).clamp(0.0, 255.0) * 4.0) as u16).min(1023);
            input.v[r * cw + c] = (((150.0 - 0.35 * e).clamp(0.0, 255.0) * 4.0) as u16).min(1023);
        }
    }
    let k = encode_key_frame_yuv_with_q(&input, 140).expect("encode");
    let lr = k.fh.lr_params.expect("lossy header carries lr_params");
    assert!(
        lr.uses_lr,
        "10-bit detail content must elect loop restoration"
    );
    let decoded = decode_av1_spec(&k.ivf_bytes).expect("spec driver");
    assert_eq!(decoded.len(), 1);
    let le = |p: &[u16]| -> Vec<u8> { p.iter().flat_map(|&s| s.to_le_bytes()).collect() };
    assert_eq!(decoded[0].planes[0], le(&k.recon_y), "10-bit KEY luma");
    assert_eq!(decoded[0].planes[1], le(&k.recon_u), "10-bit KEY U");
    assert_eq!(decoded[0].planes[2], le(&k.recon_v), "10-bit KEY V");
}
