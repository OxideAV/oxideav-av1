//! r428 — CDEF A/B harness: measures the frame-level §5.9.19/§7.15
//! CDEF election (decoder-mirror filtering on the encoder's recon
//! path + source-scored strength search) against the
//! all-zero-strength baseline (pre-r428 shape).
//!
//! Always-on: a conformance A/B (both variants' streams must
//! round-trip byte-exact through the in-tree spec driver — on the
//! armed variant that decode APPLIES §7.15, so equality proves the
//! encoder-side mirror is sample-exact), an election tripwire on
//! ringing-prone edge content (the filter must be elected and must
//! improve PSNR — the election is pure distortion at zero tile
//! bits), and the KEY-frame header check (elected strengths on the
//! wire).
//!
//! Env-gated (`OXIDEAV_AV1_CDEF_AB_DIR=<dir>`): the full matrix —
//! per-config bytes + PSNR under both variants, a CSV
//! (`cdef_ab.csv`), and the cdef-on IVF + display-order recon YUV of
//! every config for external black-box decoder validation.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, encode_key_frame_yuv420_with_q, EncodedGop,
    GopTuning, TunedGop, Yuv420Frame,
};

// ---------------------------------------------------------------------
// Content builders — sharp directional edges over flat regions
// (classic ringing bait for a quantised DCT), with a slow pan so
// P-frames stay alive.
// ---------------------------------------------------------------------

fn edge_scene(x: f64, y: f64) -> f64 {
    // Diagonal hard bands + a vertical bar over a gentle ripple (the
    // ripple defeats the palette election — >16 distinct values — so
    // the bands ring like natural quantised edges).
    let ripple = 6.0 * (0.9 * x).sin() * (0.8 * y).sin();
    let d = (0.31 * x - 0.42 * y).sin();
    let band = if d > 0.55 { 205.0 } else { 72.0 };
    if (x as i64).rem_euclid(37) < 4 {
        232.0 + ripple
    } else {
        band + ripple
    }
}

fn build_frame(w: u32, h: u32, k: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let d = 0.75 * k as f64;
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = clamp(edge_scene(c as f64 + d, r as f64 + 0.5 * d));
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            let e = edge_scene(c as f64 * 2.0 + d, r as f64 * 2.0 + 0.5 * d);
            f.u[r * cw + c] = clamp(96.0 + 0.25 * e);
            f.v[r * cw + c] = clamp(160.0 - 0.2 * e);
        }
    }
    f
}

fn cdef_content(w: u32, h: u32, n: usize) -> Vec<Yuv420Frame> {
    (0..n).map(|k| build_frame(w, h, k)).collect()
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

/// Encode with / without the CDEF arm, assert BOTH round-trip
/// byte-exact through the spec driver (the armed decode APPLIES
/// §7.15 — equality proves the encoder mirror), return `(off, on)`.
fn encode_ab(frames: &[Yuv420Frame], q: u8) -> (TunedGop, TunedGop) {
    let arm = |cdef: bool| -> TunedGop {
        encode_gop_yuv420_with_q_seg_extras_tuned(
            frames,
            q,
            &[],
            &[],
            false,
            None,
            GopTuning {
                cdef,
                ..GopTuning::default()
            },
        )
        .expect("encode")
    };
    let (off, on) = (arm(false), arm(true));
    for (name, enc) in [("cdef-off", &off.gop), ("cdef-armed", &on.gop)] {
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

/// Ringing-prone edge content must elect CDEF and must gain PSNR —
/// the election is pure distortion at zero tile bits, so an elected
/// loss would be a mirror bug.
#[test]
fn cdef_elected_and_gains_on_edge_content() {
    let q = 140u8;
    let frames = cdef_content(96, 80, 4);
    let (off, on) = encode_ab(&frames, q);
    let (p_off, p_on) = (psnr(&frames, &off.gop), psnr(&frames, &on.gop));
    eprintln!(
        "cdef-ab edges 96x80 q{q}: off {} B ({p_off:.4} dB), on {} B ({p_on:.4} dB), \
         P-frame elections {:?}",
        off.gop.ivf_bytes.len(),
        on.gop.ivf_bytes.len(),
        on.cdef_elections,
    );
    assert!(
        p_on > p_off,
        "the CDEF arm must gain PSNR on edge content ({p_on:.4} <= {p_off:.4} dB)"
    );
}

/// The elected strengths must actually reach the KEY-frame header.
#[test]
fn cdef_strengths_land_on_the_key_header() {
    let f = build_frame(96, 80, 0);
    let k = encode_key_frame_yuv420_with_q(&f, 140).expect("key encode");
    let cdef = k.fh.cdef_params.expect("lossy header carries cdef params");
    assert!(
        cdef.cdef_y_pri_strength[0] > 0
            || cdef.cdef_y_sec_strength[0] > 0
            || cdef.cdef_uv_pri_strength[0] > 0
            || cdef.cdef_uv_sec_strength[0] > 0,
        "edge content must elect a non-zero CDEF strength set"
    );
    assert_eq!(cdef.cdef_bits, 0, "frame-level arm codes one strength set");
}

/// Lossless control: the §5.9.19 gate is closed (`CodedLossless`) —
/// both arms must produce bit-identical streams.
#[test]
fn lossless_control_is_bit_identical() {
    let frames = cdef_content(64, 64, 3);
    let (off, on) = encode_ab(&frames, 0);
    assert_eq!(
        off.gop.ivf_bytes, on.gop.ivf_bytes,
        "CodedLossless closes the CDEF gate on both arms"
    );
}

/// Deterministic corpus-witness stream: an edge-content GOP at 96×80
/// q140 with the default tuning (the exact bytes pinned as
/// `self-gop-96x80-q140-cdef` in `fixture_conformance.rs`). Writes
/// `cdef-96x80-q140.ivf` + the display-order recon YUV when
/// `OXIDEAV_AV1_CDEF_DIR` is set.
#[test]
fn cdef_witness_stream_round_trips() {
    let frames = cdef_content(96, 80, 4);
    let on = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        140,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("witness encode");
    let decoded = decode_av1_spec(&on.gop.ivf_bytes).expect("spec driver");
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], on.gop.recon[idx].y, "witness frame {idx} luma");
        assert_eq!(f.planes[1], on.gop.recon[idx].u, "witness frame {idx} U");
        assert_eq!(f.planes[2], on.gop.recon[idx].v, "witness frame {idx} V");
    }
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_CDEF_DIR") {
        let root = std::path::Path::new(&dir);
        std::fs::create_dir_all(root).expect("create out dir");
        std::fs::write(root.join("cdef-96x80-q140.ivf"), &on.gop.ivf_bytes).expect("write ivf");
        let mut yuv: Vec<u8> = Vec::new();
        for rc in &on.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join("cdef-96x80-q140.yuv"), &yuv).expect("write yuv");
    }
}

// ---------------------------------------------------------------------
// Env-gated full measurement.
// ---------------------------------------------------------------------

/// Full matrix: 2 sizes x 4 quantisers. Writes `cdef_ab.csv` +
/// per-config cdef-on streams under `OXIDEAV_AV1_CDEF_AB_DIR` (skips
/// silently when unset — the always-on tests above cover CI).
#[test]
fn cdef_ab_full_measurement() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_CDEF_AB_DIR") else {
        eprintln!("OXIDEAV_AV1_CDEF_AB_DIR unset — skipping the full cdef measurement");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let mut csv = String::from("config,bytes_off,bytes_on,psnr_off,psnr_on,elections\n");
    let mut rows = 0usize;
    let mut dpsnr_total = 0.0f64;
    for (w, h) in [(96u32, 80u32), (192, 128)] {
        for q in [60u8, 100, 140, 200] {
            let frames = cdef_content(w, h, 4);
            let (off, on) = encode_ab(&frames, q);
            let (p_off, p_on) = (psnr(&frames, &off.gop), psnr(&frames, &on.gop));
            let name = format!("cdef-edges-{w}x{h}-q{q}");
            csv.push_str(&format!(
                "{name},{},{},{p_off:.4},{p_on:.4},{:?}\n",
                off.gop.ivf_bytes.len(),
                on.gop.ivf_bytes.len(),
                on.cdef_elections,
            ));
            dpsnr_total += p_on - p_off;
            rows += 1;
            std::fs::write(root.join(format!("{name}.ivf")), &on.gop.ivf_bytes).expect("write ivf");
            let mut yuv: Vec<u8> = Vec::new();
            for rc in &on.gop.recon {
                yuv.extend_from_slice(&rc.y);
                yuv.extend_from_slice(&rc.u);
                yuv.extend_from_slice(&rc.v);
            }
            std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
        }
    }
    std::fs::write(root.join("cdef_ab.csv"), &csv).expect("write csv");
    eprintln!(
        "cdef-ab totals over {rows} configs: mean dPSNR {:+.4} dB",
        dpsnr_total / rows as f64
    );
}
