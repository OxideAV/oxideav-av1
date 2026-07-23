//! r428 — high-precision-MV A/B harness: measures the §5.9.2
//! `allow_high_precision_mv` election (eighth-pel search + wire
//! precision with the per-frame exact-bytes replay election back to
//! quarter-pel) against the quarter-pel baseline (pre-r428 shape).
//!
//! Always-on: a conformance A/B (both variants' streams must
//! round-trip byte-exact through the in-tree spec driver), a smoke
//! tripwire on true sub-quarter-pel content (the eighth-pel arm must
//! win the encoder's own joint objective), and the integer-motion
//! control (the exact-bytes election must flip every P-frame back to
//! the quarter-pel arm, leaving the stream bit-identical to the
//! baseline).
//!
//! Env-gated (`OXIDEAV_AV1_HPMV_AB_DIR=<dir>`): the full matrix —
//! per-config bytes + PSNR under both variants, a CSV (`hpmv_ab.csv`),
//! and the hp-on IVF + display-order recon YUV of every config for
//! external black-box decoder validation.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, EncodedGop, GopTuning, TunedGop, Yuv420Frame,
};

// ---------------------------------------------------------------------
// Content builders — a smooth scene with fine 2D texture, sampled
// under exact per-frame translations (fractional phases reachable
// only at eighth-pel precision).
// ---------------------------------------------------------------------

fn scene_y(x: f64, y: f64) -> f64 {
    128.0
        + 34.0 * (0.115 * x + 0.7 * (0.052 * y).sin()).sin()
        + 28.0 * (0.093 * y + 0.6 * (0.047 * x).sin()).cos()
        + 14.0 * (0.021 * (x + y)).sin()
        + 22.0 * (0.41 * x + 0.31 * (0.13 * y).sin() + 0.09 * y).sin()
        + 18.0 * (0.37 * y - 0.23 * (0.11 * x).sin() + 0.07 * x).cos()
}

fn scene_u(x: f64, y: f64) -> f64 {
    128.0 + 34.0 * (0.061 * x - 0.045 * y).sin()
}

fn scene_v(x: f64, y: f64) -> f64 {
    128.0 + 30.0 * (0.052 * x + 0.058 * y).cos()
}

fn sample_frame(w: u32, h: u32, dx: f64, dy: f64) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = clamp(scene_y(c as f64 + dx, r as f64 + dy));
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            let (sx, sy) = (c as f64 * 2.0 + 0.5 + dx, r as f64 * 2.0 + 0.5 + dy);
            f.u[r * cw + c] = clamp(scene_u(sx, sy));
            f.v[r * cw + c] = clamp(scene_v(sx, sy));
        }
    }
    f
}

/// Frames `0..n` of one content kind — per-frame camera translations.
fn hp_content(kind: &str, w: u32, h: u32, n: usize) -> Vec<Yuv420Frame> {
    (0..n)
        .map(|k| {
            let kf = k as f64;
            match kind {
                // Pure eighth-pel pan: +3/8 / +1/8 pels per frame —
                // phases the quarter-pel arm can never hit exactly.
                "eighth-pan" => sample_frame(w, h, 0.375 * kf, 0.125 * kf),
                // Mixed pan: +5/8 / -3/8 pels per frame.
                "mixed-pan" => sample_frame(w, h, 0.625 * kf, -0.375 * kf),
                // Exact integer pan — the measurement matrix keeps it
                // for the rate/PSNR ledger (eighth-pel interpolation
                // still wins slightly on quantised references — the
                // 8-tap kernel smooths coding noise).
                "integer-pan" => sample_frame(w, h, 3.0 * kf, 2.0 * kf),
                // Control: static scene — no vector ever leaves zero,
                // no `mv_hp` cascade is coded, and the exact-bytes
                // replay election must flip every P-frame back.
                "static" => sample_frame(w, h, 0.0, 0.0),
                other => panic!("unknown content kind {other}"),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------
// Metrics.
// ---------------------------------------------------------------------

fn psnr(inputs: &[Yuv420Frame], enc: &EncodedGop) -> f64 {
    let mut sse = 0u64;
    let mut count = 0u64;
    for (f, rc) in inputs.iter().zip(&enc.recon) {
        for (a, b) in [(&f.y, &rc.y), (&f.u, &rc.u), (&f.v, &rc.v)] {
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = i64::from(x) - i64::from(y);
                sse += (d * d) as u64;
            }
        }
        count += (f.y.len() + f.u.len() + f.v.len()) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / count as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Encode with / without the eighth-pel arm, assert BOTH round-trip
/// byte-exact through the spec driver, and return `(off, on)`.
fn encode_ab(frames: &[Yuv420Frame], q: u8) -> (TunedGop, TunedGop) {
    let arm = |hp: bool| -> TunedGop {
        encode_gop_yuv420_with_q_seg_extras_tuned(
            frames,
            q,
            &[],
            &[],
            false,
            None,
            GopTuning {
                high_precision_mv: hp,
                ..GopTuning::default()
            },
        )
        .expect("encode")
    };
    let (off, on) = (arm(false), arm(true));
    for (name, enc) in [("quarter-pel", &off.gop), ("hp-armed", &on.gop)] {
        let decoded = decode_av1_spec(&enc.ivf_bytes)
            .unwrap_or_else(|e| panic!("{name}: spec driver rejected own stream: {e:?}"));
        assert_eq!(decoded.len(), frames.len(), "{name}: frame count");
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], enc.recon[idx].y, "{name} frame {idx} luma");
            assert_eq!(f.planes[1], enc.recon[idx].u, "{name} frame {idx} U");
            assert_eq!(f.planes[2], enc.recon[idx].v, "{name} frame {idx} V");
        }
    }
    (off, on)
}

// ---------------------------------------------------------------------
// Always-on conformance A/B + smoke tripwire.
// ---------------------------------------------------------------------

/// On true sub-quarter-pel content the eighth-pel arm must pay for
/// itself under the encoder's own joint objective `SSE + λ·bits`
/// (the arm may trade `mv_hp` bits for prediction quality, so
/// neither bytes-only nor PSNR-only is the right judge).
#[test]
fn hp_mv_pays_for_itself_on_subpel_content() {
    let q = 60u8;
    let lambda = 1.0 + (f64::from(q) * f64::from(q)) / 32.0;
    let mut wins = 0usize;
    let configs = ["eighth-pan", "mixed-pan"];
    for kind in configs {
        let frames = hp_content(kind, 64, 64, 4);
        let (off, on) = encode_ab(&frames, q);
        let sse = |enc: &EncodedGop| -> f64 {
            let mut s = 0f64;
            for (f, rc) in frames.iter().zip(&enc.recon) {
                for (a, b) in [(&f.y, &rc.y), (&f.u, &rc.u), (&f.v, &rc.v)] {
                    for (&x, &y) in a.iter().zip(b.iter()) {
                        let d = f64::from(x) - f64::from(y);
                        s += d * d;
                    }
                }
            }
            s
        };
        let score_off = sse(&off.gop) + lambda * (off.gop.ivf_bytes.len() as f64) * 8.0;
        let score_on = sse(&on.gop) + lambda * (on.gop.ivf_bytes.len() as f64) * 8.0;
        if score_on <= score_off * 1.02 {
            wins += 1;
        }
        eprintln!(
            "hpmv-ab smoke {kind}: quarter {} B (score {score_off:.0}, {:.4} dB), \
             eighth {} B (score {score_on:.0}, {:.4} dB), elections {:?}",
            off.gop.ivf_bytes.len(),
            psnr(&frames, &off.gop),
            on.gop.ivf_bytes.len(),
            psnr(&frames, &on.gop),
            on.hp_mv_elections,
        );
    }
    assert!(
        wins == configs.len(),
        "the eighth-pel arm lost the joint objective on a smoke config ({wins}/{})",
        configs.len()
    );
}

/// On sub-quarter-pel content the header flag must actually be
/// elected (the replay election must NOT flip it away — eighth-pel
/// vectors are committed and the quarter-pel replay is uncodable).
#[test]
fn hp_flag_elected_on_subpel_content() {
    let frames = hp_content("eighth-pan", 64, 64, 4);
    let (_, on) = encode_ab(&frames, 60);
    assert!(
        on.hp_mv_elections.iter().any(|&e| e),
        "no P-frame elected the eighth-pel arm on eighth-pel content: {:?}",
        on.hp_mv_elections
    );
}

/// The static control: no committed vector ever leaves zero, no
/// `mv_hp` cascade is coded, and the per-frame exact-bytes replay
/// election flips every P-frame back — the armed stream must be
/// bit-identical to the baseline.
#[test]
fn static_control_elects_quarter_pel() {
    let frames = hp_content("static", 64, 64, 4);
    let (off, on) = encode_ab(&frames, 60);
    assert!(
        on.hp_mv_elections.iter().all(|&e| !e),
        "static content elected the eighth-pel arm: {:?}",
        on.hp_mv_elections
    );
    assert_eq!(
        off.gop.ivf_bytes, on.gop.ivf_bytes,
        "quarter-pel election must reproduce the baseline stream bit-exactly"
    );
}

/// Deterministic corpus-witness stream: an eighth-pel pan GOP at
/// 96×80 q60 with the default tuning (the exact bytes pinned as
/// `self-gop-96x80-q60-hpmv` in `fixture_conformance.rs`). Writes
/// `hpmv-96x80-q60.ivf` + the display-order recon YUV when
/// `OXIDEAV_AV1_HPMV_DIR` is set.
#[test]
fn hp_mv_witness_stream_round_trips() {
    let frames = hp_content("eighth-pan", 96, 80, 4);
    let on = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        60,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("witness encode");
    assert!(
        on.hp_mv_elections.iter().any(|&e| e),
        "the witness stream must elect the eighth-pel arm: {:?}",
        on.hp_mv_elections
    );
    let decoded = decode_av1_spec(&on.gop.ivf_bytes).expect("spec driver");
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], on.gop.recon[idx].y, "witness frame {idx} luma");
        assert_eq!(f.planes[1], on.gop.recon[idx].u, "witness frame {idx} U");
        assert_eq!(f.planes[2], on.gop.recon[idx].v, "witness frame {idx} V");
    }
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_HPMV_DIR") {
        let root = std::path::Path::new(&dir);
        std::fs::create_dir_all(root).expect("create out dir");
        std::fs::write(root.join("hpmv-96x80-q60.ivf"), &on.gop.ivf_bytes).expect("write ivf");
        let mut yuv: Vec<u8> = Vec::new();
        for rc in &on.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join("hpmv-96x80-q60.yuv"), &yuv).expect("write yuv");
    }
}

// ---------------------------------------------------------------------
// Env-gated full measurement.
// ---------------------------------------------------------------------

/// Full matrix: 3 content kinds x 2 sizes x 3 quantisers. Writes
/// `hpmv_ab.csv` + per-config hp-on streams under
/// `OXIDEAV_AV1_HPMV_AB_DIR` (skips silently when unset — the
/// always-on tests above cover CI).
#[test]
fn hp_mv_ab_full_measurement() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_HPMV_AB_DIR") else {
        eprintln!("OXIDEAV_AV1_HPMV_AB_DIR unset — skipping the full hp-mv measurement");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let mut csv = String::from("config,bytes_quarter,bytes_hp,psnr_quarter,psnr_hp,elections\n");
    let mut bytes_off_total = 0usize;
    let mut bytes_on_total = 0usize;
    let mut dpsnr_total = 0.0f64;
    let mut rows = 0usize;
    for kind in ["eighth-pan", "mixed-pan", "integer-pan"] {
        for (w, h) in [(64u32, 64u32), (96, 96)] {
            for q in [60u8, 100, 160] {
                let frames = hp_content(kind, w, h, 4);
                let (off, on) = encode_ab(&frames, q);
                let (p_off, p_on) = (psnr(&frames, &off.gop), psnr(&frames, &on.gop));
                let name = format!("hpmv-{kind}-{w}x{h}-q{q}");
                csv.push_str(&format!(
                    "{name},{},{},{p_off:.4},{p_on:.4},{:?}\n",
                    off.gop.ivf_bytes.len(),
                    on.gop.ivf_bytes.len(),
                    on.hp_mv_elections,
                ));
                bytes_off_total += off.gop.ivf_bytes.len();
                bytes_on_total += on.gop.ivf_bytes.len();
                dpsnr_total += p_on - p_off;
                rows += 1;
                std::fs::write(root.join(format!("{name}.ivf")), &on.gop.ivf_bytes)
                    .expect("write ivf");
                let mut yuv: Vec<u8> = Vec::new();
                for rc in &on.gop.recon {
                    yuv.extend_from_slice(&rc.y);
                    yuv.extend_from_slice(&rc.u);
                    yuv.extend_from_slice(&rc.v);
                }
                std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
            }
        }
    }
    std::fs::write(root.join("hpmv_ab.csv"), &csv).expect("write csv");
    eprintln!(
        "hpmv-ab totals over {rows} configs: quarter {bytes_off_total} B, \
         eighth {bytes_on_total} B, mean dPSNR {:+.4} dB",
        dpsnr_total / rows as f64
    );
}
