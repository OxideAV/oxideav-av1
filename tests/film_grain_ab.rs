//! r441 — the §5.9.30 FILM-GRAIN election: A/B harness + conformance
//! smoke.
//!
//! What these tests pin:
//!
//!   * A grain-elected GOP decodes BIT-EXACT through the in-tree spec
//!     driver — the §7.18.3 synthesis the encoder mirrors onto its
//!     published output planes equals the decoder's grained output
//!     byte for byte on EVERY frame (KEY + P), while the reference
//!     chain inside the stream stays pre-grain (§7.20).
//!   * The wire shape: `film_grain_params_present = 1` on the
//!     sequence header, a full §5.9.30 parameter block with
//!     `apply_grain = 1` and a DIFFERENT `grain_seed` on every coded
//!     frame (`update_grain = 1` on the inter arm).
//!   * The ELECTION: noisy content (spatially white, temporally
//!     decorrelated residual) elects the arm under the documented
//!     perceptually-neutral-rate objective; clean and textured
//!     content keeps the no-grain shape BIT-IDENTICAL to the
//!     `film_grain = false` baseline.
//!
//! Env-gated measurement (`OXIDEAV_AV1_FG_AB=1`): noise level × q
//! matrix — elected rate vs the plain arm's, with the plain-PSNR
//! trade reported honestly.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.30, §7.18.3, §7.20.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_extras_tuned, GopTuning, Yuv420Frame};
use oxideav_av1::frame_header::FrameHeader;
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Smooth moving base — what survives the encoder-side denoiser.
fn base_value(w: usize, r: usize, c: usize, t: usize) -> f64 {
    let x = c as f64 + 1.1 * t as f64;
    let y = r as f64 + 0.5 * t as f64;
    let _ = w;
    120.0 + 60.0 * (0.021 * x).sin() * (0.026 * y).cos() + 18.0 * (0.047 * (x + y)).sin()
}

/// Noisy content: the smooth base plus deterministic white noise
/// re-rolled per frame (temporally decorrelated — the §5.9.30 use
/// case).
fn noisy_frame(w: u32, h: u32, t: usize, amp: i32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    let mut state = 0x2454_1013u32.wrapping_add((t as u32).wrapping_mul(0x9e37_79b9));
    let mut rnd = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (((state >> 23) & 255) as i32 - 128) * amp / 128
    };
    for r in 0..hu {
        for c in 0..wu {
            let v = base_value(wu, r, c, t) + f64::from(rnd());
            f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = (116 + ((r + c + t) % 24)) as u8;
            f.v[r * cw + c] = (132 + ((r * 2 + c) % 20)) as u8;
        }
    }
    f
}

/// Clean twin of [`noisy_frame`] — the probe must reject it.
fn clean_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
    noisy_frame(w, h, t, 0)
}

fn encode(frames: &[Yuv420Frame], q: u8, film_grain: bool) -> oxideav_av1::encoder::TunedGop {
    encode_gop_yuv420_with_q_seg_extras_tuned(
        frames,
        q,
        &[],
        &[],
        false,
        None,
        GopTuning {
            film_grain,
            ..GopTuning::default()
        },
    )
    .expect("gop encode")
}

fn wire_headers(tus: &[Vec<u8>]) -> (bool, Vec<FrameHeader>) {
    let mut seq = None;
    let mut out = Vec::new();
    let mut fg_present = false;
    // §7.20-tracked reference state: the §5.9.22 skipModeAllowed
    // derivation (which GATES the skip_mode_present bit) reads the
    // TRUE per-slot order hints, so a ref-less parse desyncs on any
    // frame whose slots hold two distinct forward hints (r450 —
    // caught by the segmented grain witness).
    let mut ref_info = oxideav_av1::frame_header::RefInfo::default();
    for tu in tus {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    let s = parse_sequence_header(desc.payload).expect("SH parses");
                    fg_present = s.film_grain_params_present;
                    seq = Some(s);
                }
                ObuType::Frame => {
                    let sq = seq.as_ref().expect("SH precedes");
                    let fh = oxideav_av1::frame_header::parse_frame_header_with_refs(
                        desc.payload,
                        sq,
                        &ref_info,
                    )
                    .expect("FH parses");
                    let fs = fh.frame_size.as_ref().expect("sized header");
                    for slot in 0..8 {
                        if fh.refresh_frame_flags & (1 << slot) != 0 {
                            ref_info.valid[slot] = true;
                            ref_info.order_hint[slot] = fh.order_hint;
                            ref_info.upscaled_width[slot] = fs.upscaled_width;
                            ref_info.frame_height[slot] = fs.frame_height;
                            ref_info.frame_type_is_key[slot] =
                                fh.frame_type == oxideav_av1::frame_header::FrameType::Key;
                        }
                    }
                    out.push(fh);
                }
                _ => {}
            }
        }
    }
    (fg_present, out)
}

fn assert_decodes_to_recons(name: &str, enc: &oxideav_av1::encoder::TunedGop) {
    let frames = oxideav_av1::decode_av1(&enc.gop.ivf_bytes)
        .unwrap_or_else(|e| panic!("{name}: decode: {e:?}"));
    assert_eq!(frames.len(), enc.gop.recon.len(), "{name}: frame count");
    for (i, f) in frames.iter().enumerate() {
        match f {
            Frame::Spec(s) => {
                assert_eq!(s.planes[0], enc.gop.recon[i].y, "{name}: frame {i} luma");
                assert_eq!(s.planes[1], enc.gop.recon[i].u, "{name}: frame {i} U");
                assert_eq!(s.planes[2], enc.gop.recon[i].v, "{name}: frame {i} V");
            }
            other => panic!("{name}: non-Spec frame {other:?}"),
        }
    }
}

/// Noisy content elects the arm; the grained output decodes
/// bit-exact through the spec driver on every frame; the wire
/// carries full §5.9.30 blocks with per-frame seeds.
#[test]
fn noisy_gop_elects_grain_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_frame(128, 96, t, 6)).collect();
    let on = encode(&frames, 60, true);
    let off = encode(&frames, 60, false);
    assert!(
        on.film_grain_elected,
        "white temporally-decorrelated noise must elect the grain arm \
         (grain {} B vs plain {} B)",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    assert!(
        on.gop.ivf_bytes.len() < off.gop.ivf_bytes.len(),
        "the grain arm's whole point is rate: {} vs {} B",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    let (fg_present, headers) = wire_headers(&on.gop.temporal_units);
    assert!(fg_present, "sequence gate must open");
    // Field-level audit on the KEY and the FIRST P-frame only: later
    // P-frames' §5.9.25 gm coefficients recenter against the CARRIED
    // PrevGmParams (r423), so a stateless header parse desyncs before
    // the §5.9.30 block. The bit-exact decode assertion below is the
    // authoritative per-frame witness — a wire seed differing from
    // the shared schedule (or a dropped block) would desync the
    // §7.18.3 output from the published planes.
    let mut seeds = Vec::new();
    for fh in headers.iter().take(2) {
        let fg = fh
            .film_grain_params
            .as_ref()
            .expect("frame headers carry §5.9.30 blocks");
        assert!(fg.apply_grain, "audited frames apply grain");
        assert!(fg.num_y_points > 0, "real scaling points on the wire");
        assert!(fg.update_grain, "full parameters on every frame");
        seeds.push(fg.grain_seed);
    }
    seeds.dedup();
    assert_eq!(seeds.len(), 2, "per-frame grain seeds must differ");
    assert_decodes_to_recons("fg-noisy", &on);
}

/// Clean content: the probe rejects the arm and the stream stays
/// bit-identical to the baseline.
#[test]
fn clean_gop_stays_bit_identical() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| clean_frame(128, 96, t)).collect();
    let on = encode(&frames, 100, true);
    let off = encode(&frames, 100, false);
    assert!(!on.film_grain_elected);
    assert_eq!(
        on.gop.ivf_bytes, off.gop.ivf_bytes,
        "probe-rejected content must stay bit-identical"
    );
    assert_decodes_to_recons("fg-clean", &on);
}

/// Static textured content (co-located residual repeats frame to
/// frame): the temporal-decorrelation gate rejects the arm.
#[test]
fn static_texture_rejected_by_temporal_gate() {
    let make = |_t: usize| -> Yuv420Frame {
        let (w, h) = (128u32, 96u32);
        let (wu, hu) = (w as usize, h as usize);
        let mut f = Yuv420Frame::filled(w, h, 128);
        for r in 0..hu {
            for c in 0..wu {
                f.y[r * wu + c] = ((r * 7 + c * 13 + (r % 5) * (c % 7) * 3) % 256) as u8;
            }
        }
        f
    };
    let frames: Vec<Yuv420Frame> = (0..3).map(make).collect();
    let on = encode(&frames, 100, true);
    assert!(
        !on.film_grain_elected,
        "static texture must not be replaced by synthetic grain"
    );
}

/// Env-gated measurement matrix (`OXIDEAV_AV1_FG_AB=1`): noise
/// amplitude × q — elected vs plain bytes with the honest plain-PSNR
/// trade.
#[test]
fn fg_ab_measurement_matrix() {
    if std::env::var_os("OXIDEAV_AV1_FG_AB").is_none() {
        eprintln!("OXIDEAV_AV1_FG_AB unset — skipping the film-grain measurement matrix");
        return;
    }
    let psnr = |a: &[u8], b: &[u8]| -> f64 {
        let sse: u64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| {
                let d = i64::from(x) - i64::from(y);
                (d * d) as u64
            })
            .sum();
        if sse == 0 {
            return f64::INFINITY;
        }
        10.0 * ((255.0f64 * 255.0 * a.len() as f64) / sse as f64).log10()
    };
    for amp in [3i32, 6, 12] {
        for q in [60u8, 100, 140, 180] {
            let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_frame(128, 96, t, amp)).collect();
            let on = encode(&frames, q, true);
            let off = encode(&frames, q, false);
            let mut p_on = 0.0;
            let mut p_off = 0.0;
            for (i, f) in frames.iter().enumerate() {
                p_on += psnr(&on.gop.recon[i].y, &f.y);
                p_off += psnr(&off.gop.recon[i].y, &f.y);
            }
            p_on /= frames.len() as f64;
            p_off /= frames.len() as f64;
            eprintln!(
                "fg-ab amp{amp} q{q}: plain {} B / {p_off:.2} dB | grain{} {} B / {p_on:.2} dB",
                off.gop.ivf_bytes.len(),
                if on.film_grain_elected { "*" } else { "-" },
                on.gop.ivf_bytes.len(),
            );
        }
    }
}

/// Env-gated staging dump (`OXIDEAV_AV1_FG_DIR`): a grain-elected
/// GOP plus expected YUV for black-box reference-decoder validation
/// and corpus pinning. Inert otherwise.
#[test]
fn fg_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_FG_DIR") else {
        eprintln!("OXIDEAV_AV1_FG_DIR unset — skipping the film-grain staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_frame(128, 96, t, 6)).collect();
    let enc = encode(&frames, 60, true);
    assert!(enc.film_grain_elected, "staged GOP must elect film grain");
    std::fs::write(
        root.join("gop-128x96-q60-film-grain.ivf"),
        &enc.gop.ivf_bytes,
    )
    .expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &enc.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("gop-128x96-q60-film-grain.yuv"), &yuv).expect("write yuv");
}

/// r444 — spatially CORRELATED grain, re-rolled per frame: the r441
/// whiteness probe rejected it; the relaxed probe admits it, the AR
/// fit recovers the correlation, and the fitted candidate lands
/// `ar_coeff_lag >= 1` with real §5.9.30 taps on the wire. Bit-exact
/// decode on every frame — the §7.18.3 synthesis mirror covers the
/// AR grain generation path.
#[test]
fn correlated_noise_elects_ar_taps_and_decodes_bit_exact() {
    let ar_frame = |w: u32, h: u32, t: usize| -> Yuv420Frame {
        let (wu, hu) = (w as usize, h as usize);
        let mut f = Yuv420Frame::filled(w, h, 128);
        let mut state = 0x51ed_2718u32.wrapping_add((t as u32).wrapping_mul(0x9e37_79b9));
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(((state >> 23) & 255) as i32 - 128) * 6.0 / 128.0
        };
        for r in 0..hu {
            // AR(1) along the row: n[x] = 0.55 n[x-1] + w — the
            // §7.18.3 lag-1 causal neighbourhood models it.
            let mut prev = 0.0f64;
            for c in 0..wu {
                let n = 0.55 * prev + rnd();
                prev = n;
                let v = base_value(wu, r, c, t) + n * 1.6;
                f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        for r in 0..ch {
            for c in 0..cw {
                f.u[r * cw + c] = (116 + ((r + c + t) % 24)) as u8;
                f.v[r * cw + c] = (132 + ((r * 2 + c) % 20)) as u8;
            }
        }
        f
    };
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| ar_frame(128, 96, t)).collect();
    let on = encode(&frames, 60, true);
    let off = encode(&frames, 60, false);
    assert!(
        on.film_grain_elected,
        "correlated temporally-decorrelated noise must elect the grain arm \
         (grain {} B vs plain {} B)",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    let (fg_present, headers) = wire_headers(&on.gop.temporal_units);
    assert!(fg_present, "sequence gate must open");
    let fg = headers[0]
        .film_grain_params
        .as_ref()
        .expect("KEY carries the §5.9.30 block");
    assert!(
        fg.ar_coeff_lag >= 1,
        "the fitted AR taps must land on the wire (lag {})",
        fg.ar_coeff_lag
    );
    let num_pos = 2 * usize::from(fg.ar_coeff_lag) * (usize::from(fg.ar_coeff_lag) + 1);
    assert!(
        fg.ar_coeffs_y_plus_128[..num_pos].iter().any(|&c| c != 128),
        "at least one non-zero luma tap"
    );
    // The immediate-left tap (last lag-1 position) models the planted
    // horizontal AR — it must be decisively positive.
    assert!(
        fg.ar_coeffs_y_plus_128[num_pos - 1] > 138,
        "left tap models the planted correlation (got {})",
        fg.ar_coeffs_y_plus_128[num_pos - 1]
    );
    assert_decodes_to_recons("fg-ar", &on);
}

/// r444 — grain on ALL THREE planes: the chroma noise profile elects
/// per-plane §5.9.30 scaling points (identity index mults), and the
/// grained output decodes bit-exact — the §7.18.3 chroma synthesis
/// (own gaussian arrays, luma-correlation tap, cb/cr blend index)
/// mirrors the decoder byte for byte.
#[test]
fn chroma_noise_elects_chroma_points_and_decodes_bit_exact() {
    let chroma_noisy = |w: u32, h: u32, t: usize| -> Yuv420Frame {
        let mut f = noisy_frame(w, h, t, 6);
        let (cw, ch) = ((w as usize) / 2, (h as usize) / 2);
        let mut state = 0x0bad_5eedu32.wrapping_add((t as u32).wrapping_mul(0x85eb_ca6b));
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 24) & 31) as i32 - 16) / 2 * 3
        };
        for r in 0..ch {
            for c in 0..cw {
                let u = i32::from(f.u[r * cw + c]) + rnd();
                let v = i32::from(f.v[r * cw + c]) + rnd();
                f.u[r * cw + c] = u.clamp(0, 255) as u8;
                f.v[r * cw + c] = v.clamp(0, 255) as u8;
            }
        }
        f
    };
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| chroma_noisy(128, 96, t)).collect();
    let on = encode(&frames, 60, true);
    assert!(
        on.film_grain_elected,
        "three-plane noise must elect the grain arm"
    );
    let (fg_present, headers) = wire_headers(&on.gop.temporal_units);
    assert!(fg_present, "sequence gate must open");
    let fg = headers[0]
        .film_grain_params
        .as_ref()
        .expect("KEY carries the §5.9.30 block");
    assert!(
        fg.num_cb_points > 0 && fg.num_cr_points > 0,
        "chroma scaling points must land on the wire (cb {} cr {})",
        fg.num_cb_points,
        fg.num_cr_points
    );
    assert_eq!(
        (fg.cb_mult, fg.cb_luma_mult, fg.cb_offset),
        (192, 128, 256),
        "identity §7.18.3 index mults"
    );
    assert!(
        fg.point_cb_scaling[..usize::from(fg.num_cb_points)]
            .iter()
            .any(|&s| s > 0),
        "real Cb scaling"
    );
    assert_decodes_to_recons("fg-chroma", &on);
}

/// r447 — luma-tracking chroma noise elects
/// `chroma_scaling_from_luma = 1`: when the chroma noise amplitude
/// matches luma's, the csfl candidate models it through the LUMA
/// scaling function alone (§7.18.3.4 reads the luma points for every
/// plane; the §7.18.3.5 blend indexes at the co-located average
/// luma), saving the whole per-plane point + mult/offset surface on
/// every header — strictly fewer bytes at a matching amplitude, so
/// the score + rate mandate settle on it. The grained output decodes
/// bit-exact (the corpus's first exercise of the csfl synthesis path
/// on a self-encoded stream).
#[test]
fn luma_tracking_chroma_noise_elects_csfl_and_decodes_bit_exact() {
    let csfl_noisy = |w: u32, h: u32, t: usize| -> Yuv420Frame {
        let mut f = noisy_frame(w, h, t, 12);
        let (cw, ch) = ((w as usize) / 2, (h as usize) / 2);
        let mut state = 0x0bad_5eedu32.wrapping_add((t as u32).wrapping_mul(0x85eb_ca6b));
        // The SAME amplitude shape as the luma noise (±12) — the csfl
        // arm's forced luma-LUT amplitude is a match, and the
        // per-plane point surface buys nothing.
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 23) & 255) as i32 - 128) * 12 / 128
        };
        // Smooth chroma base (no sawtooth wrap edges — the residual
        // against the denoised twin must be the NOISE, not pattern
        // edges the low-amplitude noise cannot drown).
        for r in 0..ch {
            for c in 0..cw {
                let base_u = 116.0 + 8.0 * (0.05 * (r as f64 + c as f64)).sin();
                let base_v = 132.0 + 8.0 * (0.04 * (r as f64 * 2.0 + c as f64)).cos();
                let u = base_u.round() as i32 + rnd();
                let v = base_v.round() as i32 + rnd();
                f.u[r * cw + c] = u.clamp(0, 255) as u8;
                f.v[r * cw + c] = v.clamp(0, 255) as u8;
            }
        }
        f
    };
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| csfl_noisy(128, 96, t)).collect();
    let on = encode(&frames, 60, true);
    assert!(
        on.film_grain_elected,
        "luma-tracking three-plane noise must elect the grain arm"
    );
    let (fg_present, headers) = wire_headers(&on.gop.temporal_units);
    assert!(fg_present, "sequence gate must open");
    let fg = headers[0]
        .film_grain_params
        .as_ref()
        .expect("KEY carries the §5.9.30 block");
    assert!(
        fg.chroma_scaling_from_luma,
        "csfl must land on the wire (cb {} cr {} lag {})",
        fg.num_cb_points, fg.num_cr_points, fg.ar_coeff_lag
    );
    assert_eq!(fg.num_cb_points, 0, "§5.9.30: csfl codes no cb points");
    assert_eq!(fg.num_cr_points, 0, "§5.9.30: csfl codes no cr points");
    // Field-level audit on the KEY + first P only (later headers'
    // §5.9.25 gm coefficients recenter against the carried
    // PrevGmParams, so a stateless parse desyncs before §5.9.30);
    // the bit-exact decode below is the per-frame witness.
    for h in headers.iter().take(2) {
        let fg = h.film_grain_params.as_ref().expect("every header armed");
        assert!(fg.chroma_scaling_from_luma, "csfl on every frame header");
    }
    assert_decodes_to_recons("fg-csfl", &on);
}

/// Env-gated staging dump (`OXIDEAV_AV1_FG444_DIR`): the r444
/// AR-taps + chroma-points film-grain stream + expected (grained)
/// YUV for black-box reference-decoder validation and corpus
/// pinning. Inert otherwise.
#[test]
fn fg_r444_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_FG444_DIR") else {
        eprintln!("OXIDEAV_AV1_FG444_DIR unset — skipping the r444 fg staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let ar_frame = |w: u32, h: u32, t: usize| -> Yuv420Frame {
        let (wu, hu) = (w as usize, h as usize);
        let mut f = Yuv420Frame::filled(w, h, 128);
        let mut state = 0x51ed_2718u32.wrapping_add((t as u32).wrapping_mul(0x9e37_79b9));
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(((state >> 23) & 255) as i32 - 128) * 6.0 / 128.0
        };
        for r in 0..hu {
            let mut prev = 0.0f64;
            for c in 0..wu {
                let n = 0.55 * prev + rnd();
                prev = n;
                let v = base_value(wu, r, c, t) + n * 1.6;
                f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        for r in 0..ch {
            for c in 0..cw {
                f.u[r * cw + c] = (116 + ((r + c + t) % 24)) as u8;
                f.v[r * cw + c] = (132 + ((r * 2 + c) % 20)) as u8;
            }
        }
        f
    };
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| ar_frame(128, 96, t)).collect();
    let on = encode(&frames, 60, true);
    assert!(on.film_grain_elected, "staged GOP must elect the grain arm");
    let (_, headers) = wire_headers(&on.gop.temporal_units);
    let fg = headers[0]
        .film_grain_params
        .as_ref()
        .expect("KEY carries the §5.9.30 block");
    assert!(fg.ar_coeff_lag >= 1, "staged stream carries AR taps");
    std::fs::write(
        root.join("gop-128x96-q60-film-grain-ar.ivf"),
        &on.gop.ivf_bytes,
    )
    .expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &on.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("gop-128x96-q60-film-grain-ar.yuv"), &yuv).expect("write yuv");
}

// ---------------------------------------------------------------------
// r450 — the §5.9.30 election on SEGMENTED (SEG_LVL_ALT_Q ladder)
// GOPs: the unsegmented gate is lifted for the plain delta ladder,
// so a P header may carry BOTH the §5.9.14 feature table and the
// full §5.9.30 grain block.
// ---------------------------------------------------------------------

/// Segmented noisy content elects the grain arm: the grain twin
/// codes the DENOISED frames under the same two-segment ALT_Q
/// ladder, every P header carries the §5.9.14 table AND the grain
/// block, and the grained output decodes bit-exact.
#[test]
fn segmented_ladder_noise_elects_grain_and_decodes_bit_exact() {
    let seg_encode = |frames: &[Yuv420Frame], fg: bool| -> oxideav_av1::encoder::TunedGop {
        encode_gop_yuv420_with_q_seg_extras_tuned(
            frames,
            60,
            &[0, -32],
            &[],
            false,
            None,
            GopTuning {
                film_grain: fg,
                ..GopTuning::default()
            },
        )
        .expect("segmented gop encode")
    };
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_frame(128, 96, t, 8)).collect();
    let on = seg_encode(&frames, true);
    let off = seg_encode(&frames, false);
    assert!(
        on.film_grain_elected,
        "segmented noisy GOP must elect the grain arm (grain {} B vs plain {} B)",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    assert_decodes_to_recons("seg-fg", &on);
    let (fg_present, headers) = wire_headers(&on.gop.temporal_units);
    assert!(fg_present, "sequence gate must open");
    for (k, fh) in headers.iter().enumerate() {
        if std::env::var_os("OXIDEAV_AV1_FG_DEBUG").is_some() {
            eprintln!(
                "hdr {k}: type {:?} fg {:?} seg_en {:?} temporal {:?} primary {}",
                fh.frame_type,
                fh.film_grain_params.as_ref().map(|g| (
                    g.apply_grain,
                    g.update_grain,
                    g.grain_seed
                )),
                fh.segmentation_params.as_ref().map(|x| x.enabled),
                fh.segmentation_params.as_ref().map(|x| x.temporal_update),
                fh.primary_ref_frame,
            );
        }
        let fg = fh
            .film_grain_params
            .as_ref()
            .unwrap_or_else(|| panic!("frame {k} carries the §5.9.30 block"));
        assert!(fg.apply_grain, "frame {k}: apply_grain");
        if k > 0 {
            let sp = fh
                .segmentation_params
                .as_ref()
                .unwrap_or_else(|| panic!("frame {k} carries the §5.9.14 table"));
            assert!(
                sp.enabled,
                "frame {k}: segmentation and film grain must ride the SAME header"
            );
        }
    }
    // Clean segmented content keeps the plain shape bit-identical.
    let clean: Vec<Yuv420Frame> = (0..4).map(|t| clean_frame(128, 96, t)).collect();
    let c_on = seg_encode(&clean, true);
    let c_off = seg_encode(&clean, false);
    assert!(
        !c_on.film_grain_elected,
        "clean segmented GOP must not elect"
    );
    assert_eq!(
        c_on.gop.ivf_bytes, c_off.gop.ivf_bytes,
        "non-elected segmented arm must be bit-identical to the baseline"
    );
}

/// Env-gated staging dump (`OXIDEAV_AV1_FG_SEG_DIR`): the
/// segmentation × film-grain GOP plus expected YUV for black-box
/// reference-decoder validation and corpus pinning. Inert
/// otherwise.
#[test]
fn fg_seg_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_FG_SEG_DIR") else {
        eprintln!("OXIDEAV_AV1_FG_SEG_DIR unset — skipping the seg-grain staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_frame(128, 96, t, 8)).collect();
    let enc = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        60,
        &[0, -32],
        &[],
        false,
        None,
        GopTuning {
            film_grain: true,
            ..GopTuning::default()
        },
    )
    .expect("segmented gop encode");
    assert!(enc.film_grain_elected, "staged GOP must elect");
    std::fs::write(
        root.join("gop-128x96-q60-seg-film-grain.ivf"),
        &enc.gop.ivf_bytes,
    )
    .expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &enc.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("gop-128x96-q60-seg-film-grain.yuv"), &yuv).expect("write yuv");
}
