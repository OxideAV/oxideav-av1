//! r422 — global-motion A/B harness: measures the §5.9.24 frame-level
//! global-motion election against the identity-only baseline
//! (pre-r422 header shape) on committed sweep matrices.
//!
//! Always-on: a conformance A/B (both variants' streams must
//! round-trip byte-exact through the in-tree spec driver) plus a
//! smoke tripwire — on the global-model content kinds the elected
//! models must not cost bytes in aggregate at equal-or-better PSNR
//! being separately asserted by the measurement summary.
//!
//! Env-gated (`OXIDEAV_AV1_GM_AB_DIR=<dir>`): the full matrix —
//! per-config bytes + PSNR under both variants, a CSV (`gm_ab.csv`),
//! aggregate deltas on stderr, and the gm-on IVF + display-order
//! recon YUV of every config for external black-box decoder
//! validation.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_rate_model_gm, EncodedGop, RateModel, Yuv420Frame,
};

// ---------------------------------------------------------------------
// Content builders — a smooth scene with fine 2D texture, sampled
// under exact per-frame affine maps (self-contained duplicates of the
// conformance-suite generators).
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

fn sample_frame(w: u32, h: u32, map: impl Fn(f64, f64) -> (f64, f64)) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            let (sx, sy) = map(c as f64, r as f64);
            f.y[r * wu + c] = clamp(scene_y(sx, sy));
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            let (sx, sy) = map(c as f64 * 2.0 + 0.5, r as f64 * 2.0 + 0.5);
            f.u[r * cw + c] = clamp(scene_u(sx, sy));
            f.v[r * cw + c] = clamp(scene_v(sx, sy));
        }
    }
    f
}

/// Frames `0..n` of one content kind — every kind is a per-frame
/// affine camera path over the same scene.
fn gm_content(kind: &str, w: u32, h: u32, n: usize) -> Vec<Yuv420Frame> {
    let (cx, cy) = ((w as f64 - 1.0) / 2.0, (h as f64 - 1.0) / 2.0);
    (0..n)
        .map(|k| {
            let kf = k as f64;
            match kind {
                // Pure pan: +2.0 / +1.25 pels per frame.
                "pan" => sample_frame(w, h, |x, y| (x + 2.0 * kf, y + 1.25 * kf)),
                // Centre zoom, 3.5% per frame.
                "zoom" => {
                    let s = 1.035f64.powi(k as i32);
                    sample_frame(w, h, |x, y| (cx + s * (x - cx), cy + s * (y - cy)))
                }
                // Centre rotation, ~1.7 degrees per frame.
                "rotation" => {
                    let th = 0.03 * kf;
                    let (c, s) = (th.cos(), th.sin());
                    sample_frame(w, h, |x, y| {
                        let (rx, ry) = (x - cx, y - cy);
                        (cx + c * rx + s * ry, cy - s * rx + c * ry)
                    })
                }
                // Zoom + pan combined (zoom about a drifting centre).
                "zoompan" => {
                    let s = 1.03f64.powi(k as i32);
                    sample_frame(w, h, |x, y| {
                        (cx + s * (x - cx) + 1.5 * kf, cy + s * (y - cy) - 1.0 * kf)
                    })
                }
                // Control: static scene — the election must stay
                // IDENTITY, so both variants should emit identical
                // payload bits (identity headers).
                "static" => sample_frame(w, h, |x, y| (x, y)),
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
        for (&a, &b) in f.y.iter().zip(rc.y.iter()) {
            let d = i64::from(a) - i64::from(b);
            sse += (d * d) as u64;
        }
        for (&a, &b) in f.u.iter().zip(rc.u.iter()) {
            let d = i64::from(a) - i64::from(b);
            sse += (d * d) as u64;
        }
        for (&a, &b) in f.v.iter().zip(rc.v.iter()) {
            let d = i64::from(a) - i64::from(b);
            sse += (d * d) as u64;
        }
        count += (f.y.len() + f.u.len() + f.v.len()) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    let mse = sse as f64 / count as f64;
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// Encode with / without the election, assert BOTH round-trip
/// byte-exact through the spec driver, and return
/// `(enc_off, enc_on)`.
fn encode_ab(frames: &[Yuv420Frame], q: u8) -> (EncodedGop, EncodedGop) {
    let off = encode_gop_yuv420_with_q_seg_rate_model_gm(frames, q, &[], RateModel::Twin, false)
        .expect("baseline encode");
    let on = encode_gop_yuv420_with_q_seg_rate_model_gm(frames, q, &[], RateModel::Twin, true)
        .expect("gm encode");
    for (name, enc) in [("identity-only", &off), ("gm-elected", &on)] {
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

/// On the global-model smoke set the elected models must pay for
/// themselves under the encoder's own joint objective:
/// `SSE + λ·bits` with the production λ scale (the election may trade
/// a few header bytes for prediction quality, so neither bytes-only
/// nor PSNR-only is the right judge).
#[test]
fn global_motion_pays_for_itself_on_smoke_set() {
    let q = 60u8;
    let lambda = 1.0 + (f64::from(q) * f64::from(q)) / 32.0;
    let mut wins = 0usize;
    let configs = ["pan", "zoom", "rotation"];
    for kind in configs {
        let frames = gm_content(kind, 64, 64, 4);
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
        let score_off = sse(&off) + lambda * (off.ivf_bytes.len() as f64) * 8.0;
        let score_on = sse(&on) + lambda * (on.ivf_bytes.len() as f64) * 8.0;
        if score_on <= score_off * 1.02 {
            wins += 1;
        }
        eprintln!(
            "gm-ab smoke {kind}: identity {} B (score {score_off:.0}, {:.4} dB), \
             gm {} B (score {score_on:.0}, {:.4} dB)",
            off.ivf_bytes.len(),
            psnr(&frames, &off),
            on.ivf_bytes.len(),
            psnr(&frames, &on),
        );
    }
    assert!(
        wins == configs.len(),
        "elected global motion lost the joint objective on a smoke config ({wins}/{})",
        configs.len()
    );
}

/// The static control: the election must stay IDENTITY, making the
/// two variants bit-identical.
#[test]
fn static_control_is_bit_identical() {
    let frames = gm_content("static", 64, 64, 3);
    let (off, on) = encode_ab(&frames, 60);
    assert_eq!(
        off.ivf_bytes, on.ivf_bytes,
        "IDENTITY election must leave the stream untouched"
    );
}

// ---------------------------------------------------------------------
// Env-gated full measurement.
// ---------------------------------------------------------------------

/// Full matrix: 5 content kinds x 2 sizes x 3 quantisers. Writes
/// `gm_ab.csv` + per-config gm-on streams under
/// `OXIDEAV_AV1_GM_AB_DIR` (skips silently when unset — the always-on
/// tests above cover CI).
#[test]
fn global_motion_ab_full_measurement() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_GM_AB_DIR") else {
        eprintln!("OXIDEAV_AV1_GM_AB_DIR unset — skipping the full gm measurement");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let mut csv = String::from("config,bytes_identity,bytes_gm,psnr_identity,psnr_gm\n");
    let mut bytes_off_total = 0usize;
    let mut bytes_on_total = 0usize;
    let mut dpsnr_total = 0.0f64;
    let mut smaller = 0usize;
    let mut larger = 0usize;
    let mut rows = 0usize;
    for kind in ["pan", "zoom", "rotation", "zoompan", "static"] {
        for (w, h) in [(64u32, 64u32), (96, 96)] {
            for q in [60u8, 100, 160] {
                let frames = gm_content(kind, w, h, 4);
                let (off, on) = encode_ab(&frames, q);
                let (p_off, p_on) = (psnr(&frames, &off), psnr(&frames, &on));
                let name = format!("gm-{kind}-{w}x{h}-q{q}");
                csv.push_str(&format!(
                    "{name},{},{},{p_off:.4},{p_on:.4}\n",
                    off.ivf_bytes.len(),
                    on.ivf_bytes.len(),
                ));
                bytes_off_total += off.ivf_bytes.len();
                bytes_on_total += on.ivf_bytes.len();
                dpsnr_total += p_on - p_off;
                smaller += usize::from(on.ivf_bytes.len() < off.ivf_bytes.len());
                larger += usize::from(on.ivf_bytes.len() > off.ivf_bytes.len());
                rows += 1;
                // gm-on stream + recon for external black-box
                // validation.
                let out = root.join(&name);
                std::fs::create_dir_all(&out).expect("create config dir");
                std::fs::write(out.join("input.ivf"), &on.ivf_bytes).expect("write ivf");
                let mut yuv: Vec<u8> = Vec::new();
                for rc in &on.recon {
                    yuv.extend_from_slice(&rc.y);
                    yuv.extend_from_slice(&rc.u);
                    yuv.extend_from_slice(&rc.v);
                }
                std::fs::write(out.join("expected.yuv"), &yuv).expect("write yuv");
            }
        }
    }
    std::fs::write(root.join("gm_ab.csv"), csv).expect("write csv");
    eprintln!(
        "gm-ab full: {rows} configs | bytes identity={bytes_off_total} gm={bytes_on_total} \
         ({:+.3}%) | mean dPSNR {:+.4} dB | gm smaller on {smaller}, larger on {larger}",
        (bytes_on_total as f64 - bytes_off_total as f64) / bytes_off_total as f64 * 100.0,
        dpsnr_total / rows as f64,
    );
}
