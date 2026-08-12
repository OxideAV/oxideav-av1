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
use oxideav_av1::frame_header::{parse_frame_header, FrameHeader};
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
                    out.push(
                        parse_frame_header(desc.payload, seq.as_ref().expect("SH precedes"))
                            .expect("FH parses"),
                    );
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
