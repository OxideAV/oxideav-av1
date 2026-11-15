//! r489 — the decoder's film-grain opt-out
//! ([`SpecDecodeSession::set_film_grain`], the registry decoder's
//! `film_grain` option).
//!
//! The default is the conformant output (§7.18.3 synthesis on every
//! shown frame that asks for it). With the opt-out the same frames are
//! output as the §7.20 store holds them — the pre-grain reconstruction
//! — which a black-box decoder run with its own grain switch off must
//! reproduce sample for sample (checked when one is installed). Grain
//! touches the output copy only, so a stream without grain decodes the
//! same under both settings.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.30, §7.18.3, §7.20, §7.21.

use oxideav_av1::decoder::{SpecDecodeSession, decode_av1_spec};
use oxideav_av1::encoder::{GopTuning, Yuv420Frame, encode_gop_yuv420_with_q_seg_extras_tuned};
use oxideav_av1::registry::make_decoder;
use oxideav_core::{CodecParameters, ExecutionContext, Frame, Packet, TimeBase};
use std::process::Command;

/// Smooth moving base plus white noise re-rolled per frame — the
/// content the encoder's §5.9.30 election codes as denoised frames
/// with a grain parameter set.
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
            let x = c as f64 + 1.1 * t as f64;
            let y = r as f64 + 0.5 * t as f64;
            let base = 120.0
                + 60.0 * (0.021 * x).sin() * (0.026 * y).cos()
                + 18.0 * (0.047 * (x + y)).sin();
            f.y[r * wu + c] = (base + f64::from(rnd())).round().clamp(0.0, 255.0) as u8;
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

fn encode(frames: &[Yuv420Frame], q: u8) -> oxideav_av1::encoder::TunedGop {
    encode_gop_yuv420_with_q_seg_extras_tuned(
        frames,
        q,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("gop encode")
}

type Planes = Vec<Vec<u8>>;

fn session_decode(ivf: &[u8], grain: bool, threads: usize) -> Vec<Planes> {
    let mut session = SpecDecodeSession::new();
    assert!(session.film_grain(), "the default is the conformant output");
    session.set_film_grain(grain);
    assert_eq!(session.film_grain(), grain);
    session.set_threads(threads);
    session
        .decode_ivf(ivf)
        .expect("decode")
        .into_iter()
        .map(|f| f.planes)
        .collect()
}

fn registry_decode(units: &[Vec<u8>], option: Option<&str>, threads: usize) -> Vec<Planes> {
    let mut params = CodecParameters::video("av1".into());
    if let Some(v) = option {
        params.options.insert("film_grain", v);
    }
    let mut dec = make_decoder(&params).expect("make_decoder");
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut out = Vec::new();
    let drain = |dec: &mut Box<dyn oxideav_core::Decoder>, out: &mut Vec<Planes>| loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v.planes.into_iter().map(|p| p.data).collect()),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore | oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    };
    for (i, tu) in units.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 25), tu.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        drain(&mut dec, &mut out);
    }
    dec.flush().expect("flush");
    drain(&mut dec, &mut out);
    out
}

/// A black-box decoder's output with its grain switch off (`None`
/// when no such binary is installed).
fn black_box_without_grain(ivf: &[u8]) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("oxideav-av1-grain-off-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let (src, dst) = (dir.join("in.ivf"), dir.join("out.yuv"));
    std::fs::write(&src, ivf).ok()?;
    let status = Command::new("dav1d")
        .arg("-q")
        .arg("-i")
        .arg(&src)
        .arg("-o")
        .arg(&dst)
        .args(["--muxer", "yuv", "--filmgrain", "0"])
        .status();
    let out = match status {
        Ok(s) => {
            assert!(s.success(), "the black-box decoder rejected the stream");
            Some(std::fs::read(&dst).expect("black-box decoder output"))
        }
        Err(_) => {
            assert!(
                std::env::var_os("OXIDEAV_AV1_REQUIRE_BLACK_BOX").is_none(),
                "no black-box decoder on PATH"
            );
            None
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    out
}

#[test]
fn opt_out_outputs_the_pre_grain_reconstruction() {
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_frame(128, 96, t, 6)).collect();
    let enc = encode(&frames, 60);
    assert!(
        enc.film_grain_elected,
        "the noisy GOP must carry film grain"
    );
    let ivf = &enc.gop.ivf_bytes;

    // Default: the conformant, grained output (the encoder's published
    // reconstruction).
    let grained = session_decode(ivf, true, 1);
    assert_eq!(grained.len(), frames.len());
    for (f, r) in grained.iter().zip(&enc.gop.recon) {
        assert!(f[0] == r.y && f[1] == r.u && f[2] == r.v);
    }
    let via_plain_entry: Vec<Planes> = decode_av1_spec(ivf)
        .expect("decode")
        .into_iter()
        .map(|f| f.planes)
        .collect();
    assert!(via_plain_entry == grained);

    // Opt-out: every frame differs from its grained output, and the
    // result is the same at every worker budget and through the
    // registry option.
    let plain = session_decode(ivf, false, 1);
    assert_eq!(plain.len(), frames.len());
    for (i, (p, g)) in plain.iter().zip(&grained).enumerate() {
        assert!(
            p[0] != g[0],
            "frame {i}: the opt-out left the luma grain in"
        );
        assert_eq!(p[0].len(), g[0].len());
    }
    for threads in [2usize, 4] {
        assert!(session_decode(ivf, false, threads) == plain);
        assert!(session_decode(ivf, true, threads) == grained);
    }
    for threads in [1usize, 4] {
        for off in ["false", "0", "off"] {
            assert!(registry_decode(&enc.gop.temporal_units, Some(off), threads) == plain);
        }
        for on in [None, Some("true"), Some("1"), Some("on")] {
            assert!(registry_decode(&enc.gop.temporal_units, on, threads) == grained);
        }
    }

    // The pre-grain frames are what a black-box decoder outputs with
    // its own grain switch off.
    if let Some(out) = black_box_without_grain(ivf) {
        let ours: Vec<u8> = plain.iter().flatten().flatten().copied().collect();
        assert!(
            out == ours,
            "the opt-out output differs from a black-box decoder's grain-free output"
        );
    }
}

#[test]
fn a_stream_without_grain_decodes_the_same_under_both_settings() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| noisy_frame(64, 64, t, 0)).collect();
    let enc = encode(&frames, 80);
    assert!(!enc.film_grain_elected);
    let on = session_decode(&enc.gop.ivf_bytes, true, 1);
    let off = session_decode(&enc.gop.ivf_bytes, false, 1);
    assert!(on == off);
}

#[test]
fn the_registry_option_rejects_unknown_values() {
    let mut params = CodecParameters::video("av1".into());
    params.options.insert("film_grain", "maybe");
    assert!(make_decoder(&params).is_err());
}

#[test]
fn the_opt_out_survives_a_reference_reset() {
    let mut session = SpecDecodeSession::new();
    session.set_film_grain(false);
    session.reset_references();
    assert!(!session.film_grain());
}
