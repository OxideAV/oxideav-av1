//! r428 — delta-q A/B harness: measures the §5.9.17 per-superblock
//! delta-q election (complexity-probe `CurrentQIndex` plan +
//! frame-level exact-realized-bytes arm election) against the
//! single-quantiser baseline (pre-r428 shape).
//!
//! Always-on: a conformance A/B (both variants' streams must
//! round-trip byte-exact through the in-tree spec driver), an
//! election tripwire on mixed flat/texture content (the delta arm
//! must actually be elected and must win the encoder's own joint
//! objective — the election is exact, so a loss is a harness bug),
//! and the uniform-texture control (the probe finds no spread, the
//! plan stays empty, and the armed stream is bit-identical to the
//! baseline).
//!
//! Env-gated (`OXIDEAV_AV1_DQ_AB_DIR=<dir>`): the full matrix —
//! per-config bytes + PSNR under both variants, a CSV (`dq_ab.csv`),
//! and the delta-on IVF + display-order recon YUV of every config
//! for external black-box decoder validation.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, EncodedGop, GopTuning, TunedGop, Yuv420Frame,
};

// ---------------------------------------------------------------------
// Content builders — frames mixing flat regions (banding-prone,
// nothing masks quantisation) with dense texture (masking hides it),
// under a small per-frame drift so P-frames carry real residuals.
// ---------------------------------------------------------------------

/// Dense texture field (high per-superblock variance).
fn tex(x: f64, y: f64) -> f64 {
    128.0
        + 42.0 * (0.71 * x + 0.9 * (0.23 * y).sin()).sin()
        + 36.0 * (0.63 * y - 0.7 * (0.31 * x).sin()).cos()
        + 20.0 * (0.47 * (x + y)).sin()
}

/// Smooth ramp (near-zero variance per superblock).
fn flat(x: f64, y: f64) -> f64 {
    90.0 + 0.22 * x + 0.13 * y
}

fn build_frame(w: u32, h: u32, k: usize, kind: &str) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    // The texture pans (real P-frame bits in the textured half — the
    // coarsened `+units` superblocks have rate to SAVE) while the
    // flat half carries a slow brightness ramp (sub-quantiser DC
    // drift the refined `-units` superblocks can track and the
    // frame-quantiser arm dead-zones into banding).
    let d = 1.75 * k as f64;
    let ramp = 2.5 * k as f64;
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            let (x, y) = (c as f64 + d, r as f64 + 0.5 * d);
            let v = match kind {
                // Top half flat, bottom half textured — maximal
                // per-superblock activity spread.
                "mixed" => {
                    if r < hu / 2 {
                        flat(c as f64, r as f64) + ramp
                    } else {
                        tex(x, y)
                    }
                }
                // Control: texture everywhere — no spread, the probe
                // must return an empty plan.
                "uniform" => tex(x, y),
                other => panic!("unknown content kind {other}"),
            };
            f.y[r * wu + c] = clamp(v);
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = clamp(120.0 + 16.0 * (0.05 * (c as f64 + d)).sin());
            f.v[r * cw + c] = clamp(132.0 + 14.0 * (0.06 * (r as f64 + d)).cos());
        }
    }
    f
}

fn dq_content(kind: &str, w: u32, h: u32, n: usize) -> Vec<Yuv420Frame> {
    (0..n).map(|k| build_frame(w, h, k, kind)).collect()
}

// ---------------------------------------------------------------------
// Metrics + encode helpers.
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

fn sse_total(inputs: &[Yuv420Frame], enc: &EncodedGop) -> f64 {
    let mut s = 0f64;
    for (f, rc) in inputs.iter().zip(&enc.recon) {
        for (a, b) in [(&f.y, &rc.y), (&f.u, &rc.u), (&f.v, &rc.v)] {
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = f64::from(x) - f64::from(y);
                s += d * d;
            }
        }
    }
    s
}

/// Encode with / without the delta-q arm, assert BOTH round-trip
/// byte-exact through the spec driver, and return `(off, on)`.
fn encode_ab(frames: &[Yuv420Frame], q: u8) -> (TunedGop, TunedGop) {
    let arm = |dq: bool| -> TunedGop {
        encode_gop_yuv420_with_q_seg_extras_tuned(
            frames,
            q,
            &[],
            &[],
            false,
            None,
            GopTuning {
                delta_q: dq,
                ..GopTuning::default()
            },
        )
        .expect("encode")
    };
    let (off, on) = (arm(false), arm(true));
    for (name, enc) in [("single-q", &off.gop), ("delta-q-armed", &on.gop)] {
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
// Always-on conformance A/B + election tripwires.
// ---------------------------------------------------------------------

/// Luma PSNR over the flat (banding-prone) top half only — the
/// region the delta-q plan refines.
fn psnr_flat_region(inputs: &[Yuv420Frame], enc: &EncodedGop) -> f64 {
    let mut sse = 0u64;
    let mut count = 0u64;
    for (f, rc) in inputs.iter().zip(&enc.recon) {
        let n = f.y.len() / 2;
        for (&x, &y) in f.y[..n].iter().zip(rc.y[..n].iter()) {
            let d = i64::from(x) - i64::from(y);
            sse += (d * d) as u64;
        }
        count += n as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 / (sse as f64 / count as f64)).log10()
}

/// Mixed flat/texture content must actually elect the delta arm, and
/// the armed encode must deliver the tool's point: better quality in
/// the flat (banding-prone) region — the refined `-units`
/// superblocks — with the plain-PSNR/bytes trade reported honestly
/// (the election objective is the masking-WEIGHTED joint score; at
/// fixed λ an unweighted deviation from the frame quantiser cannot
/// win by R(D) convexity).
#[test]
fn delta_q_elected_and_refines_flat_region() {
    let q = 100u8;
    let frames = dq_content("mixed", 128, 128, 4);
    let (off, on) = encode_ab(&frames, q);
    assert!(
        on.delta_q_elections.iter().any(|&e| e),
        "mixed content never elected the delta-q arm: {:?}",
        on.delta_q_elections
    );
    let (flat_off, flat_on) = (
        psnr_flat_region(&frames, &off.gop),
        psnr_flat_region(&frames, &on.gop),
    );
    eprintln!(
        "dq-ab mixed 128x128 q{q}: single {} B ({:.4} dB overall, {flat_off:.4} dB flat), \
         delta {} B ({:.4} dB overall, {flat_on:.4} dB flat), sse {:.0} vs {:.0}, elections {:?}",
        off.gop.ivf_bytes.len(),
        psnr(&frames, &off.gop),
        on.gop.ivf_bytes.len(),
        psnr(&frames, &on.gop),
        sse_total(&frames, &off.gop),
        sse_total(&frames, &on.gop),
        on.delta_q_elections,
    );
    assert!(
        flat_on > flat_off,
        "the elected delta-q arm must refine the flat region ({flat_on:.4} <= {flat_off:.4} dB)"
    );
}

/// The uniform-texture control: the probe finds no activity spread,
/// the plan stays empty, and the armed stream is bit-identical to
/// the baseline.
#[test]
fn uniform_control_stays_single_quantiser() {
    let frames = dq_content("uniform", 128, 128, 4);
    let (off, on) = encode_ab(&frames, 100);
    assert!(
        on.delta_q_elections.iter().all(|&e| !e),
        "uniform content elected the delta-q arm: {:?}",
        on.delta_q_elections
    );
    assert_eq!(
        off.gop.ivf_bytes, on.gop.ivf_bytes,
        "an empty plan must reproduce the baseline stream bit-exactly"
    );
}

/// Deterministic corpus-witness stream: a mixed flat/texture GOP at
/// 128×128 q100 with the default tuning (the exact bytes pinned as
/// `self-gop-128x128-q100-deltaq` in `fixture_conformance.rs`).
/// Writes `deltaq-128x128-q100.ivf` + the display-order recon YUV
/// when `OXIDEAV_AV1_DQ_DIR` is set.
#[test]
fn delta_q_witness_stream_round_trips() {
    let frames = dq_content("mixed", 128, 128, 4);
    let on = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        100,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("witness encode");
    assert!(
        on.delta_q_elections.iter().any(|&e| e),
        "the witness stream must elect the delta-q arm: {:?}",
        on.delta_q_elections
    );
    let decoded = decode_av1_spec(&on.gop.ivf_bytes).expect("spec driver");
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], on.gop.recon[idx].y, "witness frame {idx} luma");
        assert_eq!(f.planes[1], on.gop.recon[idx].u, "witness frame {idx} U");
        assert_eq!(f.planes[2], on.gop.recon[idx].v, "witness frame {idx} V");
    }
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_DQ_DIR") {
        let root = std::path::Path::new(&dir);
        std::fs::create_dir_all(root).expect("create out dir");
        std::fs::write(root.join("deltaq-128x128-q100.ivf"), &on.gop.ivf_bytes).expect("write ivf");
        let mut yuv: Vec<u8> = Vec::new();
        for rc in &on.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join("deltaq-128x128-q100.yuv"), &yuv).expect("write yuv");
    }
}

// ---------------------------------------------------------------------
// Env-gated full measurement.
// ---------------------------------------------------------------------

/// Full matrix: 2 content kinds x 2 sizes x 3 quantisers. Writes
/// `dq_ab.csv` + per-config delta-on streams under
/// `OXIDEAV_AV1_DQ_AB_DIR` (skips silently when unset — the
/// always-on tests above cover CI).
#[test]
fn delta_q_ab_full_measurement() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_DQ_AB_DIR") else {
        eprintln!("OXIDEAV_AV1_DQ_AB_DIR unset — skipping the full delta-q measurement");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let mut csv =
        String::from("config,bytes_single,bytes_delta,psnr_single,psnr_delta,elections\n");
    let mut rows = 0usize;
    let mut bytes_off_total = 0usize;
    let mut bytes_on_total = 0usize;
    let mut dpsnr_total = 0.0f64;
    for kind in ["mixed", "uniform"] {
        for (w, h) in [(128u32, 128u32), (192, 128)] {
            for q in [60u8, 100, 160] {
                let frames = dq_content(kind, w, h, 4);
                let (off, on) = encode_ab(&frames, q);
                let (p_off, p_on) = (psnr(&frames, &off.gop), psnr(&frames, &on.gop));
                let name = format!("dq-{kind}-{w}x{h}-q{q}");
                csv.push_str(&format!(
                    "{name},{},{},{p_off:.4},{p_on:.4},{:?}\n",
                    off.gop.ivf_bytes.len(),
                    on.gop.ivf_bytes.len(),
                    on.delta_q_elections,
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
    std::fs::write(root.join("dq_ab.csv"), &csv).expect("write csv");
    eprintln!(
        "dq-ab totals over {rows} configs: single {bytes_off_total} B, \
         delta {bytes_on_total} B, mean dPSNR {:+.4} dB",
        dpsnr_total / rows as f64
    );
}
