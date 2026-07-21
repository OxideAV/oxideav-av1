//! r421 — rate-twin A/B harness: measures the true-bit-accounting
//! rate model ([`RateModel::Twin`]) against the pre-r421 heuristic
//! baseline ([`RateModel::Heuristic`]) on committed sweep matrices.
//!
//! Always-on: a small conformance A/B (both models' streams must
//! round-trip byte-exact through the in-tree spec driver — the
//! heuristic path keeps working even though production entry points
//! are twin-only).
//!
//! Env-gated (`OXIDEAV_AV1_RATE_AB_DIR=<dir>`): the full three-matrix
//! measurement — per-config bytes + PSNR under both models, a CSV
//! (`rate_ab.csv`), per-matrix aggregate deltas on stderr, and the
//! twin-model IVF + display-order recon YUV of every config for
//! external black-box decoder validation.

use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_rate_model, encode_key_frame_yuv420_with_q_rate_model,
    encode_pyramid_gop_yuv420_with_q_rate_model, RateModel, Yuv420Frame,
};

// ---------------------------------------------------------------------
// Content builders (self-contained duplicates of the conformance-suite
// generators — deterministic, seeded).
// ---------------------------------------------------------------------

fn moving_gradient(w: u32, h: u32, shift_y: usize, shift_x: usize, seed: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let s = seed as usize;
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            let (si, sj) = (i + shift_y, j + shift_x);
            f.y[i * wu + j] = ((si * 5 + sj * 3 + (si / 16) * (sj / 16) + s) % 256) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            let (si, sj) = (i + shift_y / 2, j + shift_x / 2);
            f.u[i * cw + j] = ((128 + si * 2 + sj + s) % 256) as u8;
            f.v[i * cw + j] = ((64 + si + sj * 2 + s) % 256) as u8;
        }
    }
    f
}

fn xorshift(state: &mut u32) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    *state
}

fn noise_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let mut f = Yuv420Frame::filled(w, h, 0);
    let mut s = 0x8811_2233u32 ^ seed;
    for p in f.y.iter_mut() {
        *p = (xorshift(&mut s) & 0xFF) as u8;
    }
    for p in f.u.iter_mut() {
        *p = (xorshift(&mut s) & 0xFF) as u8;
    }
    for p in f.v.iter_mut() {
        *p = (xorshift(&mut s) & 0xFF) as u8;
    }
    f
}

fn dither_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    const COLORS: [u8; 4] = [16, 80, 160, 240];
    let mut f = Yuv420Frame::filled(w, h, 128);
    let mut s = 0x1234_5678u32 ^ seed;
    for p in f.y.iter_mut() {
        *p = COLORS[(xorshift(&mut s) & 3) as usize];
    }
    f
}

fn jitter_dither_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    const COLORS: [u8; 4] = [16, 80, 160, 240];
    let mut f = Yuv420Frame::filled(w, h, 128);
    let mut s = 0x0A11_CE55u32 ^ seed;
    for p in f.y.iter_mut() {
        let r = xorshift(&mut s);
        let base = COLORS[(r & 3) as usize];
        let jitter = ((r >> 2) % 3) as i32 - 1;
        *p = (i32::from(base) + jitter).clamp(0, 255) as u8;
    }
    f
}

fn ramp_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let s = seed as usize;
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            f.y[i * wu + j] = ((i * 2 + j + s) % 256) as u8;
        }
    }
    f
}

fn banded_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 90);
    let mut s = 0xBEEF_0001u32 ^ seed;
    for i in 0..hu {
        let band = (xorshift(&mut s) & 0x7F) as u8 + 64;
        for j in 0..wu {
            f.y[i * wu + j] = band.wrapping_add((j as u8) >> 3);
        }
    }
    f
}

fn tiled_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    // One 64x64 noise tile repeated — intra-bc territory.
    let mut tile = vec![0u8; 64 * 64];
    let mut s = 0x00C0_FFEEu32 ^ seed;
    for p in tile.iter_mut() {
        *p = (xorshift(&mut s) & 0xFF) as u8;
    }
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    for i in 0..hu {
        for j in 0..wu {
            f.y[i * wu + j] = tile[(i % 64) * 64 + (j % 64)];
        }
    }
    f
}

fn checker_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let s = seed as usize;
    let mut f = Yuv420Frame::filled(w, h, 100);
    for i in 0..hu {
        for j in 0..wu {
            let par = ((i >> 3) + (j >> 3) + s) & 1;
            f.y[i * wu + j] = if par == 0 { 60 } else { 190 };
        }
    }
    f
}

fn texture_frame(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let s = seed as usize;
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            f.y[i * wu + j] = ((((i % 97) * (j % 89)) / 31 + i / 2 + j / 3 + s) % 256) as u8;
        }
    }
    for i in 0..hu / 2 {
        for j in 0..wu / 2 {
            f.u[i * (wu / 2) + j] = ((i * 3 + j + s) % 200) as u8 + 20;
            f.v[i * (wu / 2) + j] = ((i + j * 3 + s) % 200) as u8 + 30;
        }
    }
    f
}

fn intra_content(kind: &str, w: u32, h: u32, seed: u32) -> Yuv420Frame {
    match kind {
        "gradient" => moving_gradient(w, h, 0, 0, seed),
        "noise" => noise_frame(w, h, seed),
        "dither" => dither_frame(w, h, seed),
        "jitter" => jitter_dither_frame(w, h, seed),
        "ramp" => ramp_frame(w, h, seed),
        "bands" => banded_frame(w, h, seed),
        "tiles" => tiled_frame(w, h, seed),
        "checker" => checker_frame(w, h, seed),
        "texture" => texture_frame(w, h, seed),
        _ => unreachable!("unknown intra content kind"),
    }
}

fn gop_content(kind: &str, w: u32, h: u32, n: usize, seed: u32) -> Vec<Yuv420Frame> {
    match kind {
        "move" => (0..n)
            .map(|k| moving_gradient(w, h, 2 * k, 3 * k, seed))
            .collect(),
        "static" => vec![moving_gradient(w, h, 0, 0, seed); n],
        "cut" => (0..n)
            .map(|k| {
                if k == n / 2 {
                    moving_gradient(w, h, 50, 90, seed.wrapping_add(700))
                } else {
                    moving_gradient(w, h, 2 * k, 5 * k, seed)
                }
            })
            .collect(),
        "noise" => (0..n).map(|k| noise_frame(w, h, seed + k as u32)).collect(),
        "shear" => (0..n)
            .map(|k| {
                // Per-row horizontal shift ramp — the §5.11.27 OBMC
                // territory the r419 witnesses drove.
                let (wu, hu) = (w as usize, h as usize);
                let mut f = Yuv420Frame::filled(w, h, 0);
                let base = moving_gradient(w, h, 0, 0, seed);
                for i in 0..hu {
                    let shift = (k * i) / hu.max(1);
                    for j in 0..wu {
                        f.y[i * wu + j] = base.y[i * wu + (j + shift) % wu];
                    }
                }
                for i in 0..hu / 2 {
                    let shift = (k * i) / (hu / 2).max(1) / 2;
                    for j in 0..wu / 2 {
                        f.u[i * (wu / 2) + j] = base.u[i * (wu / 2) + (j + shift) % (wu / 2)];
                        f.v[i * (wu / 2) + j] = base.v[i * (wu / 2) + (j + shift) % (wu / 2)];
                    }
                }
                f
            })
            .collect(),
        "zoom" => (0..n)
            .map(|k| {
                // Progressive centre zoom — WARPED_CAUSAL territory.
                let (wu, hu) = (w as usize, h as usize);
                let base = texture_frame(w, h, seed);
                let mut f = Yuv420Frame::filled(w, h, 0);
                let num = 64 + k as i64;
                for i in 0..hu {
                    for j in 0..wu {
                        let sy = ((i as i64 - hu as i64 / 2) * 64 / num + hu as i64 / 2)
                            .clamp(0, hu as i64 - 1) as usize;
                        let sx = ((j as i64 - wu as i64 / 2) * 64 / num + wu as i64 / 2)
                            .clamp(0, wu as i64 - 1) as usize;
                        f.y[i * wu + j] = base.y[sy * wu + sx];
                    }
                }
                let (cw, ch) = (wu / 2, hu / 2);
                for i in 0..ch {
                    for j in 0..cw {
                        let sy = ((i as i64 - ch as i64 / 2) * 64 / num + ch as i64 / 2)
                            .clamp(0, ch as i64 - 1) as usize;
                        let sx = ((j as i64 - cw as i64 / 2) * 64 / num + cw as i64 / 2)
                            .clamp(0, cw as i64 - 1) as usize;
                        f.u[i * cw + j] = base.u[sy * cw + sx];
                        f.v[i * cw + j] = base.v[sy * cw + sx];
                    }
                }
                f
            })
            .collect(),
        "blend" => {
            let f0 = moving_gradient(w, h, 0, 0, seed);
            let f_last = moving_gradient(w, h, 0, 0, seed.wrapping_add(100));
            (0..n)
                .map(|k| {
                    let num = k as u32;
                    let den = (n - 1).max(1) as u32;
                    let mut out = Yuv420Frame::filled(w, h, 0);
                    let mix = |x: u8, y: u8| {
                        ((u32::from(x) * (den - num) + u32::from(y) * num) / den) as u8
                    };
                    for ((o, &x), &y) in out.y.iter_mut().zip(&f0.y).zip(&f_last.y) {
                        *o = mix(x, y);
                    }
                    for ((o, &x), &y) in out.u.iter_mut().zip(&f0.u).zip(&f_last.u) {
                        *o = mix(x, y);
                    }
                    for ((o, &x), &y) in out.v.iter_mut().zip(&f0.v).zip(&f_last.v) {
                        *o = mix(x, y);
                    }
                    out
                })
                .collect()
        }
        "screen" => (0..n)
            .map(|k| {
                // Dither panel widening over a moving gradient.
                let mut f = moving_gradient(w, h, k, 2 * k, seed);
                let panel_w = ((w as usize) * (k + 1) / (n + 1)).max(8);
                let d = dither_frame(w, h, seed);
                let (wu, hu) = (w as usize, h as usize);
                for i in 0..hu {
                    for j in 0..panel_w.min(wu) {
                        f.y[i * wu + j] = d.y[i * wu + j];
                    }
                }
                f
            })
            .collect(),
        "fine" => (0..n)
            .map(|k| {
                // Per-4x4-cell checkerboard motion.
                let base = moving_gradient(w, h, 0, 0, seed);
                let (wu, hu) = (w as usize, h as usize);
                let mut f = Yuv420Frame::filled(w, h, 0);
                for i in 0..hu {
                    for j in 0..wu {
                        let cell = ((i >> 2) + (j >> 2)) & 1;
                        let s = if cell == 0 { k } else { 2 * k };
                        f.y[i * wu + j] = base.y[((i + s) % hu) * wu + (j + s) % wu];
                    }
                }
                f.u = base.u.clone();
                f.v = base.v.clone();
                f
            })
            .collect(),
        "halfpel" => (0..n)
            .map(|k| {
                let base = |y: usize, x: usize| -> u8 {
                    (((y % 97) * (x % 89)) / 31 + y / 2 + x / 3) as u8
                };
                let (wu, hu) = (w as usize, h as usize);
                let mut f = Yuv420Frame::filled(w, h, 0);
                for i in 0..hu {
                    for j in 0..wu {
                        f.y[i * wu + j] = base(2 * i + k, 2 * j + k);
                    }
                }
                for i in 0..hu / 2 {
                    for j in 0..wu / 2 {
                        f.u[i * (wu / 2) + j] = base(4 * i + k, 4 * j + k) / 2 + 64;
                        f.v[i * (wu / 2) + j] = base(4 * i + k + 7, 4 * j + k + 3) / 2 + 32;
                    }
                }
                f
            })
            .collect(),
        _ => unreachable!("unknown gop content kind"),
    }
}

// ---------------------------------------------------------------------
// Metrics.
// ---------------------------------------------------------------------

/// Global PSNR (all planes, all frames) of recon vs input.
fn psnr(inputs: &[&Yuv420Frame], recons: &[(&[u8], &[u8], &[u8])]) -> f64 {
    assert_eq!(inputs.len(), recons.len());
    let mut sse = 0u64;
    let mut count = 0u64;
    for (f, (ry, ru, rv)) in inputs.iter().zip(recons) {
        for (&a, &b) in f.y.iter().zip(ry.iter()) {
            let d = i64::from(a) - i64::from(b);
            sse += (d * d) as u64;
        }
        for (&a, &b) in f.u.iter().zip(ru.iter()) {
            let d = i64::from(a) - i64::from(b);
            sse += (d * d) as u64;
        }
        for (&a, &b) in f.v.iter().zip(rv.iter()) {
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

struct AbRow {
    matrix: &'static str,
    name: String,
    bytes_h: usize,
    bytes_t: usize,
    psnr_h: f64,
    psnr_t: f64,
}

fn summarize(rows: &[AbRow]) {
    for matrix in ["intra", "inter", "pyramid"] {
        let sel: Vec<&AbRow> = rows.iter().filter(|r| r.matrix == matrix).collect();
        if sel.is_empty() {
            continue;
        }
        let bh: usize = sel.iter().map(|r| r.bytes_h).sum();
        let bt: usize = sel.iter().map(|r| r.bytes_t).sum();
        let finite = |v: f64| if v.is_finite() { v } else { 100.0 };
        let ph: f64 = sel.iter().map(|r| finite(r.psnr_h)).sum::<f64>() / sel.len() as f64;
        let pt: f64 = sel.iter().map(|r| finite(r.psnr_t)).sum::<f64>() / sel.len() as f64;
        let smaller = sel.iter().filter(|r| r.bytes_t < r.bytes_h).count();
        let larger = sel.iter().filter(|r| r.bytes_t > r.bytes_h).count();
        eprintln!(
            "rate-ab {matrix}: {} configs | bytes heuristic={bh} twin={bt} ({:+.3}%) | \
             mean PSNR heuristic={ph:.4} dB twin={pt:.4} dB ({:+.4} dB) | \
             twin smaller on {smaller}, larger on {larger}",
            sel.len(),
            (bt as f64 - bh as f64) / bh as f64 * 100.0,
            pt - ph,
        );
        // Worst byte regression, for follow-up triage.
        if let Some(worst) = sel
            .iter()
            .filter(|r| r.bytes_t > r.bytes_h)
            .max_by_key(|r| r.bytes_t - r.bytes_h)
        {
            eprintln!(
                "rate-ab {matrix}: worst byte regression {}: {} -> {} (PSNR {:+.4} dB)",
                worst.name,
                worst.bytes_h,
                worst.bytes_t,
                finite(worst.psnr_t) - finite(worst.psnr_h),
            );
        }
    }
}

// ---------------------------------------------------------------------
// Always-on conformance A/B.
// ---------------------------------------------------------------------

/// Both rate models must produce spec-driver-conformant streams whose
/// decode equals the encoder reconstruction byte-for-byte (the
/// heuristic model stays alive as the measurement baseline even
/// though production entry points are twin-only).
#[test]
fn both_rate_models_round_trip_spec_driver() {
    let f = texture_frame(96, 80, 5);
    for model in [RateModel::Heuristic, RateModel::Twin] {
        let enc = encode_key_frame_yuv420_with_q_rate_model(&f, 60, model).unwrap();
        let frames = oxideav_av1::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(frames.len(), 1, "{model:?}");
        assert_eq!(frames[0].planes[0], enc.recon_y, "{model:?}: luma");
        assert_eq!(frames[0].planes[1], enc.recon_u, "{model:?}: U");
        assert_eq!(frames[0].planes[2], enc.recon_v, "{model:?}: V");
    }
    let gop = gop_content("shear", 64, 64, 3, 31);
    for model in [RateModel::Heuristic, RateModel::Twin] {
        let enc = encode_gop_yuv420_with_q_seg_rate_model(&gop, 60, &[], model).unwrap();
        let frames = oxideav_av1::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(frames.len(), 3, "{model:?}");
        for (k, fr) in frames.iter().enumerate() {
            assert_eq!(fr.planes[0], enc.recon[k].y, "{model:?}: frame {k} luma");
            assert_eq!(fr.planes[1], enc.recon[k].u, "{model:?}: frame {k} U");
            assert_eq!(fr.planes[2], enc.recon[k].v, "{model:?}: frame {k} V");
        }
    }
}

/// The twin model must never lose the `D + λ·R` game it plays: on a
/// representative config set, the twin's own objective (distortion +
/// λ·true-bits, evaluated post-hoc from the emitted payload sizes and
/// reconstructions) is no worse than the heuristic's beyond a small
/// tolerance. This is a weak-form regression tripwire, not the full
/// measurement (which is env-gated).
#[test]
fn twin_model_is_no_worse_on_smoke_set() {
    let configs: &[(&str, u32, u32, u8)] = &[
        ("texture", 64, 64, 60),
        ("gradient", 96, 80, 100),
        ("jitter", 64, 64, 60),
        ("checker", 64, 128, 160),
    ];
    let mut wins = 0usize;
    for &(kind, w, h, q) in configs {
        let f = intra_content(kind, w, h, 9);
        let enc_h = encode_key_frame_yuv420_with_q_rate_model(&f, q, RateModel::Heuristic).unwrap();
        let enc_t = encode_key_frame_yuv420_with_q_rate_model(&f, q, RateModel::Twin).unwrap();
        // Post-hoc joint objective at the encoder's own λ scale:
        // score = SSE + λ·bits.
        let lambda = 1.0 + (q as f64 * q as f64) / 32.0;
        let sse = |enc: &oxideav_av1::encoder::EncodedKeyFrame| -> f64 {
            let mut s = 0f64;
            for (&a, &b) in f.y.iter().zip(&enc.recon_y) {
                let d = f64::from(a) - f64::from(b);
                s += d * d;
            }
            for (&a, &b) in f.u.iter().zip(&enc.recon_u) {
                let d = f64::from(a) - f64::from(b);
                s += d * d;
            }
            for (&a, &b) in f.v.iter().zip(&enc.recon_v) {
                let d = f64::from(a) - f64::from(b);
                s += d * d;
            }
            s
        };
        let score_h = sse(&enc_h) + lambda * (enc_h.ivf_bytes.len() as f64) * 8.0;
        let score_t = sse(&enc_t) + lambda * (enc_t.ivf_bytes.len() as f64) * 8.0;
        if score_t <= score_h * 1.02 {
            wins += 1;
        }
        eprintln!(
            "rate-ab smoke {kind}-{w}x{h}-q{q}: heuristic {} B (score {score_h:.0}), \
             twin {} B (score {score_t:.0})",
            enc_h.ivf_bytes.len(),
            enc_t.ivf_bytes.len(),
        );
    }
    assert!(
        wins >= configs.len() - 1,
        "twin model lost the joint objective on more than one smoke config ({wins}/{})",
        configs.len()
    );
}

// ---------------------------------------------------------------------
// Env-gated full measurement.
// ---------------------------------------------------------------------

#[test]
fn rate_twin_ab_full_measurement() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_RATE_AB_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    let mut rows: Vec<AbRow> = Vec::new();
    let mut csv = String::from("matrix,name,bytes_heuristic,bytes_twin,psnr_heuristic,psnr_twin\n");

    // ---- Intra matrix: 5 geometries x 7 quantisers x 9 contents ----
    let geometries: [(u32, u32); 5] = [(64, 64), (96, 80), (176, 144), (64, 128), (120, 88)];
    let qs: [u8; 7] = [0, 30, 60, 100, 160, 200, 255];
    let contents = [
        "gradient", "noise", "dither", "jitter", "ramp", "bands", "tiles", "checker", "texture",
    ];
    for (gi, &(w, h)) in geometries.iter().enumerate() {
        for &q in &qs {
            for (ci, &kind) in contents.iter().enumerate() {
                let f = intra_content(kind, w, h, (gi * 31 + ci * 7) as u32);
                let name = format!("kf-{w}x{h}-q{q}-{kind}");
                let enc_h =
                    encode_key_frame_yuv420_with_q_rate_model(&f, q, RateModel::Heuristic).unwrap();
                let enc_t =
                    encode_key_frame_yuv420_with_q_rate_model(&f, q, RateModel::Twin).unwrap();
                let p_h = psnr(&[&f], &[(&enc_h.recon_y, &enc_h.recon_u, &enc_h.recon_v)]);
                let p_t = psnr(&[&f], &[(&enc_t.recon_y, &enc_t.recon_u, &enc_t.recon_v)]);
                std::fs::write(format!("{dir}/{name}.ivf"), &enc_t.ivf_bytes).unwrap();
                let mut yuv = Vec::new();
                yuv.extend_from_slice(&enc_t.recon_y);
                yuv.extend_from_slice(&enc_t.recon_u);
                yuv.extend_from_slice(&enc_t.recon_v);
                std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
                csv.push_str(&format!(
                    "intra,{name},{},{},{p_h:.4},{p_t:.4}\n",
                    enc_h.ivf_bytes.len(),
                    enc_t.ivf_bytes.len()
                ));
                rows.push(AbRow {
                    matrix: "intra",
                    name,
                    bytes_h: enc_h.ivf_bytes.len(),
                    bytes_t: enc_t.ivf_bytes.len(),
                    psnr_h: p_h,
                    psnr_t: p_t,
                });
            }
        }
    }

    // ---- Inter (P-GOP) matrix: 11 contents x 6 quantisers ----
    let gop_kinds = [
        "move",
        "static",
        "cut",
        "noise",
        "blend",
        "halfpel",
        "fine",
        "shear",
        "zoom",
        "screen",
        "bands-gop",
    ];
    let gqs: [u8; 6] = [0, 30, 60, 100, 160, 255];
    for (ci, &kind) in gop_kinds.iter().enumerate() {
        for &q in &gqs {
            let (w, h, n) = (64u32, 64u32, 4usize);
            let frames = if kind == "bands-gop" {
                (0..n)
                    .map(|k| banded_frame(w, h, (ci * 13 + k) as u32))
                    .collect()
            } else {
                gop_content(kind, w, h, n, (ci * 13) as u32)
            };
            let name = format!("gop-{w}x{h}-q{q}-{kind}-n{n}");
            let enc_h =
                encode_gop_yuv420_with_q_seg_rate_model(&frames, q, &[], RateModel::Heuristic)
                    .unwrap();
            let enc_t =
                encode_gop_yuv420_with_q_seg_rate_model(&frames, q, &[], RateModel::Twin).unwrap();
            let inputs: Vec<&Yuv420Frame> = frames.iter().collect();
            let rc_h: Vec<(&[u8], &[u8], &[u8])> = enc_h
                .recon
                .iter()
                .map(|r| (r.y.as_slice(), r.u.as_slice(), r.v.as_slice()))
                .collect();
            let rc_t: Vec<(&[u8], &[u8], &[u8])> = enc_t
                .recon
                .iter()
                .map(|r| (r.y.as_slice(), r.u.as_slice(), r.v.as_slice()))
                .collect();
            let p_h = psnr(&inputs, &rc_h);
            let p_t = psnr(&inputs, &rc_t);
            std::fs::write(format!("{dir}/{name}.ivf"), &enc_t.ivf_bytes).unwrap();
            let mut yuv = Vec::new();
            for r in &enc_t.recon {
                yuv.extend_from_slice(&r.y);
                yuv.extend_from_slice(&r.u);
                yuv.extend_from_slice(&r.v);
            }
            std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
            csv.push_str(&format!(
                "inter,{name},{},{},{p_h:.4},{p_t:.4}\n",
                enc_h.ivf_bytes.len(),
                enc_t.ivf_bytes.len()
            ));
            rows.push(AbRow {
                matrix: "inter",
                name,
                bytes_h: enc_h.ivf_bytes.len(),
                bytes_t: enc_t.ivf_bytes.len(),
                psnr_h: p_h,
                psnr_t: p_t,
            });
        }
    }

    // ---- Pyramid matrix: 5 geometries x 6 quantisers (rotating content) ----
    let pyr_contents = [
        "move", "static", "cut", "noise", "blend", "halfpel", "fine", "shear", "zoom", "screen",
    ];
    let lengths = [2usize, 3, 4, 5, 7, 9];
    for (gi, &(w, h)) in geometries.iter().enumerate() {
        for (qi, &q) in gqs.iter().enumerate() {
            let kind = pyr_contents[(gi + qi) % pyr_contents.len()];
            let n = lengths[(gi * gqs.len() + qi) % lengths.len()];
            let frames = gop_content(kind, w, h, n, (gi * 31 + qi * 7) as u32);
            let name = format!("pyr-{w}x{h}-q{q}-{kind}-n{n}");
            let enc_h =
                encode_pyramid_gop_yuv420_with_q_rate_model(&frames, q, RateModel::Heuristic)
                    .unwrap();
            let enc_t =
                encode_pyramid_gop_yuv420_with_q_rate_model(&frames, q, RateModel::Twin).unwrap();
            let inputs: Vec<&Yuv420Frame> = frames.iter().collect();
            let rc_h: Vec<(&[u8], &[u8], &[u8])> = enc_h
                .recon
                .iter()
                .map(|r| (r.y.as_slice(), r.u.as_slice(), r.v.as_slice()))
                .collect();
            let rc_t: Vec<(&[u8], &[u8], &[u8])> = enc_t
                .recon
                .iter()
                .map(|r| (r.y.as_slice(), r.u.as_slice(), r.v.as_slice()))
                .collect();
            let p_h = psnr(&inputs, &rc_h);
            let p_t = psnr(&inputs, &rc_t);
            std::fs::write(format!("{dir}/{name}.ivf"), &enc_t.ivf_bytes).unwrap();
            let mut yuv = Vec::new();
            for r in &enc_t.recon {
                yuv.extend_from_slice(&r.y);
                yuv.extend_from_slice(&r.u);
                yuv.extend_from_slice(&r.v);
            }
            std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
            csv.push_str(&format!(
                "pyramid,{name},{},{},{p_h:.4},{p_t:.4}\n",
                enc_h.ivf_bytes.len(),
                enc_t.ivf_bytes.len()
            ));
            rows.push(AbRow {
                matrix: "pyramid",
                name,
                bytes_h: enc_h.ivf_bytes.len(),
                bytes_t: enc_t.ivf_bytes.len(),
                psnr_h: p_h,
                psnr_t: p_t,
            });
        }
    }

    std::fs::write(format!("{dir}/rate_ab.csv"), csv).unwrap();
    summarize(&rows);
}
