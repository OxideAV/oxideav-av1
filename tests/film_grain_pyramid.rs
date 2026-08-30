//! r452 — §5.9.30 film grain on the B-PYRAMID and ADAPTIVE drivers.
//!
//! The r441/r444/r447 election was GOP-scoped (KEY + P chain). The
//! pyramid drivers now ride the same driver-agnostic ladder: the
//! grain arm re-runs the WHOLE out-of-order refresh graph over the
//! DENOISED frames with full §5.9.30 parameters on every coded
//! header — KEY, ALT, MID and B alike, each seeded at its own
//! `order_hint` — so a decoded-not-shown frame's later
//! `show_existing_frame` output carries the §7.21 `load_grain_params()`
//! synthesis of THAT frame, and the perceptually-neutral-rate
//! objective settles the arm against the plain encode.
//!
//! Coverage:
//! * noisy content ELECTS the arm on both drivers, and the published
//!   (grained) reconstruction decodes bit-exact through the spec
//!   driver on every display position — shown frames and
//!   `show_existing_frame` outputs alike,
//! * every coded header off the wire carries `apply_grain = 1` (the
//!   decoded-not-shown ones included) under an opened sequence gate,
//! * clean content keeps the plain shape, bit-identical to the
//!   `film_grain = false` tuning,
//! * an env-gated staging dump feeds black-box reference-decoder
//!   validation and corpus pinning.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.30, §7.18.3, §7.21.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_adaptive_gop_yuv420_with_q_tuned, encode_pyramid_gop_yuv420_with_q_tuned,
    AdaptiveTuning, PyramidTuning, Yuv420Frame,
};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

fn base_value(r: usize, c: usize, t: usize) -> f64 {
    let x = c as f64 + 1.1 * t as f64;
    let y = r as f64 + 0.5 * t as f64;
    120.0 + 60.0 * (0.021 * x).sin() * (0.026 * y).cos() + 18.0 * (0.047 * (x + y)).sin()
}

/// Smooth drifting base plus temporally-decorrelated white noise of
/// amplitude `amp` (0 = clean).
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
            let v = base_value(r, c, t) + f64::from(rnd());
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

fn pyramid(film_grain: bool) -> PyramidTuning {
    PyramidTuning {
        film_grain,
        max_mini_gop: 4,
        ..PyramidTuning::default()
    }
}

/// Every temporal unit's coded frame headers: `(show_frame,
/// apply_grain)` per `OBU_FRAME` / `OBU_FRAME_HEADER` that codes a
/// frame (show-existing headers are skipped), plus the sequence gate.
fn wire_grain_shape(tus: &[Vec<u8>]) -> (bool, Vec<(bool, bool)>) {
    let mut seq = None;
    let mut gate = false;
    let mut out = Vec::new();
    let mut ref_info = oxideav_av1::frame_header::RefInfo::default();
    for tu in tus {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    let s = parse_sequence_header(desc.payload).expect("SH parses");
                    gate = s.film_grain_params_present;
                    seq = Some(s);
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let sq = seq.as_ref().expect("SH precedes");
                    let fh = oxideav_av1::frame_header::parse_frame_header_with_refs(
                        desc.payload,
                        sq,
                        &ref_info,
                    )
                    .expect("FH parses");
                    if fh.show_existing_frame {
                        continue;
                    }
                    let fs = fh.frame_size.as_ref().expect("sized header");
                    for slot in 0..8 {
                        if fh.refresh_frame_flags & (1 << slot) != 0 {
                            ref_info.valid[slot] = true;
                            ref_info.order_hint[slot] = fh.order_hint;
                            ref_info.upscaled_width[slot] = fs.upscaled_width;
                            ref_info.frame_height[slot] = fs.frame_height;
                            ref_info.render_width[slot] = fs.render_width;
                            ref_info.render_height[slot] = fs.render_height;
                        }
                    }
                    let apply = fh
                        .film_grain_params
                        .as_ref()
                        .map(|g| g.apply_grain)
                        .unwrap_or(false);
                    out.push((fh.show_frame, apply));
                }
                _ => {}
            }
        }
    }
    (gate, out)
}

fn assert_decodes_to_recon(ivf: &[u8], recon: &[oxideav_av1::encoder::GopFrameRecon]) {
    let decoded = decode_av1_spec(ivf).expect("spec driver decodes");
    assert_eq!(decoded.len(), recon.len());
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(
            f.planes[0], recon[idx].y,
            "frame {idx}: luma decode != published recon"
        );
        assert_eq!(
            f.planes[1], recon[idx].u,
            "frame {idx}: U decode != published recon"
        );
        assert_eq!(
            f.planes[2], recon[idx].v,
            "frame {idx}: V decode != published recon"
        );
    }
}

/// Noisy content elects the arm on the pyramid driver; every coded
/// header (decoded-not-shown ALT / MID included) carries the grain
/// block, and the grained output decodes bit-exact on every display
/// position — the `show_existing_frame` outputs synthesize the
/// stored frame's own parameters via §7.21.
#[test]
fn pyramid_driver_elects_film_grain_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..9).map(|t| noisy_frame(96, 80, t, 24)).collect();
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 40, pyramid(true))
        .expect("pyramid encodes");
    assert!(
        enc.film_grain_elected,
        "noisy content must elect the grain arm"
    );
    let (gate, shape) = wire_grain_shape(&enc.gop.temporal_units);
    assert!(gate, "sequence gate opened");
    assert_eq!(shape.len(), 9, "nine coded frames");
    assert!(
        shape.iter().any(|&(show, _)| !show),
        "a 4-deep mini-GOP codes decoded-not-shown frames"
    );
    for (k, &(show, apply)) in shape.iter().enumerate() {
        assert!(
            apply,
            "coded frame {k} (show_frame = {show}) carries apply_grain = 1"
        );
    }
    assert_decodes_to_recon(&enc.gop.ivf_bytes, &enc.gop.recon);
}

/// The adaptive driver rides the same settlement.
#[test]
fn adaptive_driver_elects_film_grain_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..8).map(|t| noisy_frame(96, 80, t, 24)).collect();
    let tuning = AdaptiveTuning {
        pyramid: pyramid(true),
        ..AdaptiveTuning::default()
    };
    let enc =
        encode_adaptive_gop_yuv420_with_q_tuned(&frames, 40, tuning).expect("adaptive encodes");
    assert!(
        enc.film_grain_elected,
        "noisy content must elect the grain arm"
    );
    let (gate, shape) = wire_grain_shape(&enc.gop.temporal_units);
    assert!(gate);
    assert!(shape.iter().all(|&(_, apply)| apply));
    assert_decodes_to_recon(&enc.gop.ivf_bytes, &enc.gop.recon);
}

/// Clean content keeps the plain shape — bit-identical to the
/// `film_grain = false` tuning (and the sequence gate stays shut).
#[test]
fn clean_content_keeps_the_plain_pyramid_shape() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| noisy_frame(64, 64, t, 0)).collect();
    let on = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 40, pyramid(true)).expect("on");
    let off = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 40, pyramid(false)).expect("off");
    assert!(!on.film_grain_elected);
    assert_eq!(on.gop.ivf_bytes, off.gop.ivf_bytes);
    let (gate, _) = wire_grain_shape(&on.gop.temporal_units);
    assert!(!gate);
}

/// Env-gated staging dump (`OXIDEAV_AV1_PYR_FG_DIR`): the 9-frame
/// 96×80 q40 grained pyramid as `input.ivf` plus the spec driver's
/// `expected.yuv` (planar I420, display order) for black-box
/// reference-decoder validation and corpus pinning.
#[test]
fn pyramid_film_grain_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_PYR_FG_DIR") else {
        eprintln!("OXIDEAV_AV1_PYR_FG_DIR unset — skipping the staging dump");
        return;
    };
    let frames: Vec<Yuv420Frame> = (0..9).map(|t| noisy_frame(96, 80, t, 24)).collect();
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 40, pyramid(true))
        .expect("pyramid encodes");
    assert!(enc.film_grain_elected);
    let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("decodes");
    let mut yuv = Vec::new();
    for f in &decoded {
        for p in &f.planes {
            yuv.extend_from_slice(p);
        }
    }
    std::fs::create_dir_all(&dir).expect("staging dir");
    std::fs::write(format!("{dir}/input.ivf"), &enc.gop.ivf_bytes).expect("write ivf");
    std::fs::write(format!("{dir}/expected.yuv"), &yuv).expect("write yuv");
}
