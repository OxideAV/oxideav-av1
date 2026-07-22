//! r424 — deep-pyramid / adaptive-mini-GOP A/B harness.
//!
//! Always-on: the r424 deep tuning (recursive dyadic pyramid up to 16
//! frames per mini-GOP, per-layer q offsets, primary-reference carry
//! election) and the r423 two-level baseline tuning (`max_mini_gop =
//! 4`, flat q, `PRIMARY_REF_NONE`) both round-trip byte-exact through
//! the crate's spec-faithful frame driver on the same inputs; the
//! adaptive driver's behavioural witnesses (deep chunks on static
//! content, scene-cut isolation, live primary elections) are pinned.
//!
//! Env-gated (`OXIDEAV_AV1_PYR_AB_DIR`): the measurement matrix —
//! baseline vs deep vs adaptive on identical inputs, per-config bytes
//! and PSNR (CSV + aggregate deltas), every deep/adaptive stream and
//! its reconstruction dumped for external black-box decoder
//! validation.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_adaptive_gop_yuv420_with_q_tuned, encode_pyramid_gop_yuv420_with_q_tuned,
    AdaptiveTuning, EncodedGop, PyramidTuning, RateModel, Yuv420Frame,
};
use oxideav_av1::PRIMARY_REF_NONE;

/// Deterministic textured frame with translation `(sy, sx)`.
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

/// Content builder (the sweep-matrix subset the deep/adaptive arcs
/// care about: translation, stillness, a hard cut, shear, zoom,
/// unpredictable noise).
fn build_content(kind: &str, w: u32, h: u32, n: usize, seed: u32) -> Vec<Yuv420Frame> {
    match kind {
        "move" => (0..n)
            .map(|k| moving_gradient(w, h, 2 * k, 3 * k, seed))
            .collect(),
        "static" => {
            let f = moving_gradient(w, h, 0, 0, seed);
            vec![f; n]
        }
        "cut" => (0..n)
            .map(|k| {
                if k >= n / 2 {
                    moving_gradient(w, h, 40 + 2 * k, 70 + 3 * k, seed.wrapping_add(700))
                } else {
                    moving_gradient(w, h, 2 * k, 3 * k, seed)
                }
            })
            .collect(),
        "shear" => {
            let tex = |i: usize, j: usize, s: u32| -> u8 {
                ((i * 7 + j * 11 + (i / 4) * (j / 8) + ((i * j) / 13) + s as usize) % 256) as u8
            };
            (0..n)
                .map(|k| {
                    let (wu, hu) = (w as usize, h as usize);
                    let (ramp0, ramp1) = (hu / 3, 2 * hu / 3);
                    let mut f = Yuv420Frame::filled(w, h, 128);
                    for i in 0..hu {
                        let top = 4 * k;
                        let dx = if i < ramp0 {
                            top
                        } else if i >= ramp1 {
                            0
                        } else {
                            top * (ramp1 - i) / (ramp1 - ramp0)
                        };
                        for j in 0..wu {
                            f.y[i * wu + j] = tex(i, j + dx, seed);
                        }
                    }
                    for i in 0..hu / 2 {
                        for j in 0..wu / 2 {
                            f.u[i * (wu / 2) + j] = tex(i, j, seed) / 2 + 64;
                            f.v[i * (wu / 2) + j] = tex(i + 5, j + 3, seed) / 2 + 32;
                        }
                    }
                    f
                })
                .collect()
        }
        "zoom" => {
            let tex = |y: i64, x: i64, s: u32| -> u8 {
                let (y, x) = (y as f64 / 8.0, x as f64 / 8.0);
                let v = 96.0
                    + 60.0 * ((y * 0.11).sin() * (x * 0.13).cos())
                    + 40.0 * ((y * 0.05 + x * 0.07 + (s % 7) as f64).sin());
                v.clamp(0.0, 255.0) as u8
            };
            (0..n)
                .map(|k| {
                    let (wu, hu) = (w as usize, h as usize);
                    let (cy, cx) = ((hu / 2) as i64, (wu / 2) as i64);
                    let (num, den) = (16 + k as i64, 16i64);
                    let mut f = Yuv420Frame::filled(w, h, 128);
                    for i in 0..hu {
                        for j in 0..wu {
                            let sy = cy * 8 + ((i as i64) - cy) * 8 * num / den;
                            let sx = cx * 8 + ((j as i64) - cx) * 8 * num / den;
                            f.y[i * wu + j] = tex(sy, sx, seed);
                        }
                    }
                    for i in 0..hu / 2 {
                        for j in 0..wu / 2 {
                            f.u[i * (wu / 2) + j] = tex(4 * i as i64, 4 * j as i64, seed) / 2 + 64;
                            f.v[i * (wu / 2) + j] =
                                tex(4 * i as i64 + 9, 4 * j as i64 + 5, seed) / 2 + 32;
                        }
                    }
                    f
                })
                .collect()
        }
        "noise" => {
            let mut state = 0x8811_2233u32 ^ seed;
            let mut next = move || {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                state
            };
            (0..n)
                .map(|_| {
                    let mut f = Yuv420Frame::filled(w, h, 0);
                    for p in f.y.iter_mut() {
                        *p = (next() & 0xFF) as u8;
                    }
                    for p in f.u.iter_mut() {
                        *p = (next() & 0xFF) as u8;
                    }
                    for p in f.v.iter_mut() {
                        *p = (next() & 0xFF) as u8;
                    }
                    f
                })
                .collect()
        }
        other => panic!("unknown content kind {other}"),
    }
}

/// The r423 two-level baseline tuning.
fn baseline_tuning() -> PyramidTuning {
    PyramidTuning {
        model: RateModel::Twin,
        max_mini_gop: 4,
        layer_q_offsets: false,
        primary_ref: false,
    }
}

/// Spec-driver round trip: decoded display-order output must equal
/// the encoder reconstruction byte-for-byte.
fn assert_round_trip(enc: &EncodedGop, n: usize, label: &str) {
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{label}: spec driver rejected stream: {e:?}"));
    assert_eq!(decoded.len(), n, "{label}: display frame count");
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.recon[idx].y, "{label} display {idx}: luma");
        assert_eq!(f.planes[1], enc.recon[idx].u, "{label} display {idx}: U");
        assert_eq!(f.planes[2], enc.recon[idx].v, "{label} display {idx}: V");
    }
}

/// Global PSNR (all three planes) of a reconstruction against the
/// input frames.
fn psnr(frames: &[Yuv420Frame], enc: &EncodedGop) -> f64 {
    let mut sse = 0u64;
    let mut count = 0u64;
    for (f, rc) in frames.iter().zip(&enc.recon) {
        for (a, b) in
            rc.y.iter()
                .zip(&f.y)
                .chain(rc.u.iter().zip(&f.u))
                .chain(rc.v.iter().zip(&f.v))
        {
            let d = i64::from(*a) - i64::from(*b);
            sse += (d * d) as u64;
            count += 1;
        }
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    10.0 * ((255.0f64 * 255.0 * count as f64) / sse as f64).log10()
}

/// Deep tuning and two-level baseline both round-trip byte-exact on
/// the same inputs (three-level and four-level mini-GOPs live).
#[test]
fn deep_and_baseline_round_trip() {
    for (kind, n, q) in [("move", 17usize, 60u8), ("static", 12, 100), ("cut", 9, 60)] {
        let frames = build_content(kind, 64, 64, n, 31);
        let deep = encode_pyramid_gop_yuv420_with_q_tuned(&frames, q, PyramidTuning::default())
            .unwrap_or_else(|e| panic!("{kind} n={n} q={q}: deep encode failed: {e:?}"));
        assert_round_trip(&deep.gop, n, &format!("deep {kind} n={n} q={q}"));
        // A 17-frame GOP at max 16 = KEY + one L=16 chunk (four
        // levels); 12 frames = KEY + L=11 (non-dyadic recursion).
        assert!(
            deep.chunk_lengths.iter().all(|&l| l <= 16),
            "chunk cap: {:?}",
            deep.chunk_lengths
        );
        let base = encode_pyramid_gop_yuv420_with_q_tuned(&frames, q, baseline_tuning())
            .unwrap_or_else(|e| panic!("{kind} n={n} q={q}: baseline encode failed: {e:?}"));
        assert_round_trip(&base.gop, n, &format!("baseline {kind} n={n} q={q}"));
        assert!(
            base.chunk_lengths.iter().all(|&l| l <= 4),
            "baseline chunk cap: {:?}",
            base.chunk_lengths
        );
        // The baseline never elects a primary reference; the deep
        // tuning records one election outcome per coded inter frame.
        assert!(base
            .primary_elections
            .iter()
            .all(|&(_, p)| p == PRIMARY_REF_NONE));
        assert_eq!(deep.primary_elections.len(), n - 1);
    }
}

/// The primary-reference election is LIVE through the pyramid: on
/// coherent moving content the carry must win on some frames (the
/// election is exact-bytes, so a non-NONE outcome means the carried
/// state priced strictly smaller than per-frame defaults).
#[test]
fn deep_primary_election_wins_on_coherent_content() {
    let frames = build_content("move", 64, 64, 9, 77);
    let deep = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 60, PyramidTuning::default())
        .expect("deep encode");
    let wins = deep
        .primary_elections
        .iter()
        .filter(|&&(_, p)| p != PRIMARY_REF_NONE)
        .count();
    assert!(
        wins > 0,
        "no frame elected a carried primary reference: {:?}",
        deep.primary_elections
    );
}

/// Adaptive driver on static content: the motion probe reads ~0, so
/// the class election must commit deep mini-GOPs (length >= 8).
#[test]
fn adaptive_static_elects_deep_chunks() {
    let frames = build_content("static", 64, 64, 17, 5);
    let tuned = encode_adaptive_gop_yuv420_with_q_tuned(&frames, 60, AdaptiveTuning::default())
        .expect("adaptive encode");
    assert!(
        tuned.chunk_lengths.iter().any(|&l| l >= 8),
        "static content must go deep: {:?}",
        tuned.chunk_lengths
    );
    assert!(tuned.cuts.iter().all(|&c| !c), "no cuts in static content");
    assert_round_trip(&tuned.gop, 17, "adaptive static");
}

/// Adaptive driver on scene-cut content: the cut transition is
/// detected and NO committed mini-GOP spans it — the chunk that
/// contains the first post-cut frame starts there as a flat P step.
#[test]
fn adaptive_scene_cut_is_isolated() {
    let n = 13usize;
    let frames = build_content("cut", 64, 64, n, 3);
    let tuned = encode_adaptive_gop_yuv420_with_q_tuned(&frames, 60, AdaptiveTuning::default())
        .expect("adaptive encode");
    let cut_t = tuned
        .cuts
        .iter()
        .position(|&c| c)
        .expect("the constructed cut must trip the probe");
    // Reconstruct chunk spans: chunk i covers displays
    // [starts[i], starts[i] + len - 1].
    let mut start = 1usize;
    let mut isolated = false;
    for &l in &tuned.chunk_lengths {
        let end = start + l - 1;
        // The cut transition sits between displays cut_t and cut_t+1:
        // no chunk may contain both.
        assert!(
            !(start <= cut_t && cut_t < end),
            "chunk {start}..={end} spans the cut transition {cut_t}: {:?}",
            tuned.chunk_lengths
        );
        if start == cut_t + 1 {
            assert_eq!(l, 1, "the post-cut chunk must be a flat P step");
            isolated = true;
        }
        start = end + 1;
    }
    assert!(isolated, "no chunk starts at the post-cut frame");
    assert_round_trip(&tuned.gop, n, "adaptive cut");
}

/// Env-gated fixture-generation twin (`OXIDEAV_AV1_PYR_FIXTURE_DIR`):
/// the two r424 corpus candidates — the 17-frame four-level deep
/// pyramid and the 13-frame adaptive scene-cut stream — written as
/// IVF + encoder-reconstruction YUV for external black-box decoder
/// validation and corpus pinning (fixture bytes + notes staged under
/// `docs/video/av1/fixtures/`).
#[test]
fn pyramid_deep_fixture_dump() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_PYR_FIXTURE_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    let mut dump = |name: &str, gop: &EncodedGop| {
        std::fs::write(format!("{dir}/{name}.ivf"), &gop.ivf_bytes).unwrap();
        let mut yuv = Vec::new();
        for rc in &gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
    };
    // Four-level deep pyramid: KEY + one L=16 mini-GOP, per-layer q,
    // primary-reference election live on every coded frame.
    let frames: Vec<Yuv420Frame> = (0..17)
        .map(|k| moving_gradient(64, 64, 2 * k, 3 * k, 29))
        .collect();
    let deep = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 60, PyramidTuning::default())
        .expect("deep fixture encode");
    assert_round_trip(&deep.gop, 17, "fixture deep len17");
    assert_eq!(deep.chunk_lengths, vec![16]);
    dump("self-pyr-64x64-q60-len17-deep", &deep.gop);
    // Adaptive scene-cut stream: mixed pyramid depths + the flat P
    // step absorbing the cut.
    let cut_frames = build_content("cut", 96, 80, 13, 3);
    let adap = encode_adaptive_gop_yuv420_with_q_tuned(&cut_frames, 60, AdaptiveTuning::default())
        .expect("adaptive fixture encode");
    assert_round_trip(&adap.gop, 13, "fixture adaptive cut n13");
    assert!(adap.cuts.iter().any(|&c| c), "cut must trip the probe");
    dump("self-adaptive-96x80-q60-cut-n13", &adap.gop);
    eprintln!(
        "fixture dump: deep chunks {:?}, adaptive chunks {:?} (cuts {:?})",
        deep.chunk_lengths, adap.chunk_lengths, adap.cuts
    );
}

/// Env-gated measurement matrix: baseline (r423 two-level) vs deep
/// (r424) vs adaptive on identical inputs — per-config bytes + PSNR
/// CSV, aggregate deltas, and every deep/adaptive stream +
/// reconstruction dumped for external black-box decoder validation.
#[test]
fn pyramid_deep_ab_measurement_matrix() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_PYR_AB_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    let contents = ["move", "static", "cut", "shear", "zoom", "noise"];
    let qs: [u8; 3] = [60, 100, 160];
    let lens = [9usize, 17];
    let mut csv = String::from(
        "content,w,h,n,q,base_bytes,base_psnr,deep_bytes,deep_psnr,\
         adap_bytes,adap_psnr,deep_chunks,adap_chunks,primary_wins\n",
    );
    let (mut base_total, mut deep_total, mut adap_total) = (0u64, 0u64, 0u64);
    let (mut base_psnr_sum, mut deep_psnr_sum, mut adap_psnr_sum) = (0f64, 0f64, 0f64);
    let mut configs = 0u32;
    let mut deep_smaller = 0u32;
    for (ci, &content) in contents.iter().enumerate() {
        for (qi, &q) in qs.iter().enumerate() {
            for (ni, &n) in lens.iter().enumerate() {
                let (w, h) = if (ci + qi + ni) % 3 == 0 {
                    (96u32, 80u32)
                } else {
                    (64, 64)
                };
                let frames = build_content(content, w, h, n, (ci * 37 + qi * 11 + ni) as u32);
                let base =
                    encode_pyramid_gop_yuv420_with_q_tuned(&frames, q, baseline_tuning()).unwrap();
                let deep =
                    encode_pyramid_gop_yuv420_with_q_tuned(&frames, q, PyramidTuning::default())
                        .unwrap();
                let adap =
                    encode_adaptive_gop_yuv420_with_q_tuned(&frames, q, AdaptiveTuning::default())
                        .unwrap();
                for (label, gop) in [
                    ("base", &base.gop),
                    ("deep", &deep.gop),
                    ("adap", &adap.gop),
                ] {
                    assert_round_trip(gop, n, &format!("{label} {content} {w}x{h} n={n} q={q}"));
                }
                let (bb, db, ab) = (
                    base.gop.ivf_bytes.len() as u64,
                    deep.gop.ivf_bytes.len() as u64,
                    adap.gop.ivf_bytes.len() as u64,
                );
                let (bp, dp, ap) = (
                    psnr(&frames, &base.gop),
                    psnr(&frames, &deep.gop),
                    psnr(&frames, &adap.gop),
                );
                let wins = deep
                    .primary_elections
                    .iter()
                    .filter(|&&(_, p)| p != PRIMARY_REF_NONE)
                    .count();
                csv.push_str(&format!(
                    "{content},{w},{h},{n},{q},{bb},{bp:.3},{db},{dp:.3},{ab},{ap:.3},\
                     {:?},{:?},{wins}\n",
                    deep.chunk_lengths, adap.chunk_lengths
                ));
                base_total += bb;
                deep_total += db;
                adap_total += ab;
                base_psnr_sum += bp.min(100.0);
                deep_psnr_sum += dp.min(100.0);
                adap_psnr_sum += ap.min(100.0);
                configs += 1;
                if db < bb {
                    deep_smaller += 1;
                }
                for (tag, gop) in [("deep", &deep.gop), ("adap", &adap.gop)] {
                    let name = format!("pyrab-{tag}-{content}-{w}x{h}-q{q}-n{n}");
                    std::fs::write(format!("{dir}/{name}.ivf"), &gop.ivf_bytes).unwrap();
                    let mut yuv = Vec::new();
                    for rc in &gop.recon {
                        yuv.extend_from_slice(&rc.y);
                        yuv.extend_from_slice(&rc.u);
                        yuv.extend_from_slice(&rc.v);
                    }
                    std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
                }
            }
        }
    }
    std::fs::write(format!("{dir}/pyr_ab.csv"), &csv).unwrap();
    let f = configs as f64;
    eprintln!(
        "pyramid deep A/B over {configs} configs:\n\
         base {base_total} B, mean PSNR {:.3} dB\n\
         deep {deep_total} B ({:+.2}% vs base, smaller on {deep_smaller}/{configs}), \
         mean PSNR {:.3} dB ({:+.3})\n\
         adap {adap_total} B ({:+.2}% vs base), mean PSNR {:.3} dB ({:+.3})",
        base_psnr_sum / f,
        (deep_total as f64 / base_total as f64 - 1.0) * 100.0,
        deep_psnr_sum / f,
        (deep_psnr_sum - base_psnr_sum) / f,
        (adap_total as f64 / base_total as f64 - 1.0) * 100.0,
        adap_psnr_sum / f,
        (adap_psnr_sum - base_psnr_sum) / f,
    );
}
