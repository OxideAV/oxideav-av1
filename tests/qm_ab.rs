//! r439 — §5.9.12 quantizer-matrix ELECTION: on unsegmented lossy
//! frames with real high-frequency luma energy, a second full search
//! runs under `using_qmatrix = 1` at the §9.5.3 level keyed off the
//! frame quantiser, and a frame-level joint-objective election over
//! exact realized bytes keeps the better arm (KEY frames and inter
//! GOPs alike; the winner feeds the §5.9.17 delta-q election).
//!
//! What these tests pin:
//!
//!   * The election FIRES on textured content — the elected stream
//!     carries `using_qmatrix = 1` + the §9.5.3 levels on the wire —
//!     and both arms decode BIT-EXACT through the spec driver (a
//!     decoder skipping the §7.12.3 QM-scaled `q2` desyncs the
//!     coefficient chain and fails the pixel match).
//!   * Smooth gradient content never trials the arm (the
//!     high-frequency probe gate) — the armed encoder is
//!     bit-identical to the `qm: false` baseline there.
//!   * Env-gated measurement (`OXIDEAV_AV1_QM_MEASURE=1`): bytes +
//!     PSNR for the armed encoder vs the flat-quantiser baseline
//!     over a natural-texture / gradient / mixed matrix.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.12, §7.12.2, §7.12.3,
//! §9.5.3.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_tuned, encode_key_frame_yuv420_with_q, GopTuning, Yuv420Frame,
};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

// ---------------------------------------------------------------------
// Content.
// ---------------------------------------------------------------------

/// Natural-ish texture: smooth large-scale luminance variation plus a
/// deterministic fine-grain component — plenty of mid/high-frequency
/// energy without hard synthetic edges. `t` pans the field.
fn natural(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    let mut lcg = 0x2545_F491u32;
    let mut noise = vec![0i32; (wu + 32) * (hu + 32)];
    for n in noise.iter_mut() {
        lcg = lcg.wrapping_mul(1664525).wrapping_add(1013904223);
        *n = ((lcg >> 24) as i32 & 15) - 8;
    }
    for r in 0..hu {
        for c in 0..wu {
            let (sr, sc) = (r as f64, (c + 2 * t) as f64);
            let base = 128.0
                + 46.0 * (0.041 * sr).sin() * (0.057 * sc).cos()
                + 22.0 * (0.013 * (sr + sc)).sin();
            let g = noise[(r + t) * (wu + 32) + c + 2 * t];
            f.y[r * wu + c] = (base as i32 + g).clamp(0, 255) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = (120 + ((r + c + t) % 17)) as u8;
            f.v[r * cw + c] = (132 + ((2 * r + c + t) % 13)) as u8;
        }
    }
    f
}

/// Pure smooth gradient — essentially zero second-difference energy;
/// the QM probe must refuse the arm here.
fn gradient(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = (((r + c + t) * 200) / (wu + hu)).min(255) as u8;
        }
    }
    f
}

// ---------------------------------------------------------------------
// Wire audit + decode helpers.
// ---------------------------------------------------------------------

/// Every coded frame header's `(using_qmatrix, qm_y)` in decode order.
fn wire_qm(tus: &[Vec<u8>], w: u32, h: u32) -> Vec<(bool, u8)> {
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    for i in 0..8 {
        refinfo.valid[i] = true;
        refinfo.upscaled_width[i] = w;
        refinfo.frame_height[i] = h;
        refinfo.render_width[i] = w;
        refinfo.render_height[i] = h;
    }
    let mut out = Vec::new();
    for tu in tus {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let fh = parse_frame_header_with_refs(
                        desc.payload,
                        seq.as_ref().expect("SH precedes frames"),
                        &refinfo,
                    )
                    .expect("frame header parses");
                    if fh.show_existing_frame {
                        continue;
                    }
                    let qp = fh
                        .quantization_params
                        .expect("coded frame carries q params");
                    out.push((qp.using_qmatrix, qp.qm_y));
                }
                _ => {}
            }
        }
    }
    out
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

fn arm(frames: &[Yuv420Frame], q: u8, qm: bool) -> oxideav_av1::encoder::TunedGop {
    encode_gop_yuv420_with_q_seg_tuned(
        frames,
        q,
        &[],
        GopTuning {
            qm,
            ..GopTuning::default()
        },
    )
    .expect("encode")
}

fn psnr(frames: &[Yuv420Frame], enc: &oxideav_av1::encoder::EncodedGop) -> f64 {
    let mut sse = 0u64;
    let mut count = 0u64;
    for (f, rc) in frames.iter().zip(&enc.recon) {
        for (a, b) in [(&f.y, &rc.y), (&f.u, &rc.u), (&f.v, &rc.v)] {
            for (&x, &y) in a.iter().zip(b.iter()) {
                let d = i64::from(x) - i64::from(y);
                sse += (d * d) as u64;
            }
            count += a.len() as u64;
        }
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    10.0 * ((255.0f64 * 255.0 * count as f64) / sse as f64).log10()
}

// ---------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------

/// Textured content at a coarse quantiser: the election fires (the
/// KEY and/or inter frames carry `using_qmatrix = 1` on the wire),
/// the election trace agrees with the wire, and the whole stream
/// decodes bit-exact through the spec driver.
#[test]
fn qm_election_fires_on_texture_and_decodes_bit_exact() {
    // q100 measured as a reliable election point on this content
    // (the harness matrix: −14.85% bytes at −0.067 dB, two of three
    // P-frames elected).
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| natural(96, 80, t)).collect();
    let on = arm(&frames, 100, true);
    let ids = wire_qm(&on.gop.temporal_units, 96, 80);
    assert_eq!(ids.len(), frames.len());
    assert!(
        ids.iter().any(|&(u, _)| u),
        "designed content must elect the QM arm at least once: {ids:?}"
    );
    // Trace/wire agreement on the P-frames (index 0 is the KEY).
    for (k, &(u, lvl)) in ids.iter().enumerate().skip(1) {
        let expect = on.qm_elections[k - 1];
        assert_eq!(u, expect, "frame {k} wire using_qmatrix");
        if u {
            assert!(lvl < 15, "frame {k}: coded level in the §9.5.3 range");
        }
    }
    assert_decodes_to_recons("qm-on", &on);
    // The baseline decodes too and never codes the flag.
    let off = arm(&frames, 100, false);
    assert!(off.qm_elections.iter().all(|&e| !e));
    assert!(wire_qm(&off.gop.temporal_units, 96, 80)
        .iter()
        .all(|&(u, _)| !u));
    assert_decodes_to_recons("qm-off", &off);
}

/// The KEY-frame arm through the public entry: over a small
/// (size, q) grid at least one textured KEY elects QM, the returned
/// header descriptor matches the wire on EVERY config, and every
/// stream decodes bit-exact.
#[test]
fn qm_key_frame_election_decodes_bit_exact() {
    let mut fired = Vec::new();
    for &(w, h) in &[(96u32, 80u32), (128, 128)] {
        for &q in &[100u8, 140, 180] {
            let f = natural(w, h, 0);
            let k = encode_key_frame_yuv420_with_q(&f, q).expect("key encode");
            let qp =
                k.fh.quantization_params
                    .as_ref()
                    .expect("descriptor carries q params");
            let ids = wire_qm(core::slice::from_ref(&k.temporal_unit_bytes), w, h);
            assert_eq!(ids.len(), 1);
            assert_eq!(
                ids[0].0, qp.using_qmatrix,
                "{w}x{h} q{q}: descriptor/wire agreement"
            );
            if qp.using_qmatrix {
                fired.push((w, h, q, ids[0].1));
            }
            let decoded = oxideav_av1::decoder::decode_av1_spec(&k.ivf_bytes).expect("decodes");
            assert_eq!(decoded.len(), 1);
            assert_eq!(decoded[0].planes[0], k.recon_y, "{w}x{h} q{q}: luma");
            assert_eq!(decoded[0].planes[1], k.recon_u, "{w}x{h} q{q}: U");
            assert_eq!(decoded[0].planes[2], k.recon_v, "{w}x{h} q{q}: V");
        }
    }
    assert!(
        !fired.is_empty(),
        "at least one textured KEY config must elect the QM arm"
    );
}

/// Smooth gradients carry no high-frequency energy: the probe gate
/// refuses the arm and the armed encoder is BIT-IDENTICAL to the
/// baseline.
#[test]
fn qm_probe_gate_keeps_gradients_bit_identical() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| gradient(96, 80, t)).collect();
    let on = arm(&frames, 100, true);
    let off = arm(&frames, 100, false);
    assert_eq!(on.gop.ivf_bytes, off.gop.ivf_bytes);
    assert!(on.qm_elections.iter().all(|&e| !e));
}

/// Env-gated measurement (`OXIDEAV_AV1_QM_MEASURE=1`): bytes + PSNR,
/// armed vs baseline, over a natural / gradient / mixed-q matrix.
/// Inert otherwise.
#[test]
fn qm_measurement_matrix() {
    if std::env::var_os("OXIDEAV_AV1_QM_MEASURE").is_none() {
        eprintln!("OXIDEAV_AV1_QM_MEASURE unset — skipping the QM measurement matrix");
        return;
    }
    let mut rows = Vec::new();
    for (kind, gen) in [
        ("natural", natural as fn(u32, u32, usize) -> Yuv420Frame),
        ("gradient", gradient as fn(u32, u32, usize) -> Yuv420Frame),
    ] {
        for &(w, h) in &[(96u32, 80u32), (128, 128)] {
            for &q in &[60u8, 100, 140, 180, 220] {
                let frames: Vec<Yuv420Frame> = (0..4).map(|t| gen(w, h, t)).collect();
                let on = arm(&frames, q, true);
                let off = arm(&frames, q, false);
                let (bon, boff) = (on.gop.ivf_bytes.len(), off.gop.ivf_bytes.len());
                let (pon, poff) = (psnr(&frames, &on.gop), psnr(&frames, &off.gop));
                rows.push(format!(
                    "{kind} {w}x{h} q{q}: qm {} B {:.3} dB | flat {} B {:.3} dB | Δbytes {:+.2}% ΔPSNR {:+.3} dB elected {:?}",
                    bon,
                    pon,
                    boff,
                    poff,
                    100.0 * (bon as f64 - boff as f64) / boff as f64,
                    pon - poff,
                    on.qm_elections,
                ));
            }
        }
    }
    for r in &rows {
        eprintln!("QM-AB {r}");
    }
}

/// Env-gated staging dump (`OXIDEAV_AV1_QM_DIR`): the elected QM
/// streams + expected YUV for black-box reference-decoder validation
/// and corpus pinning. Inert otherwise.
#[test]
fn qm_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_QM_DIR") else {
        eprintln!("OXIDEAV_AV1_QM_DIR unset — skipping the QM staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");

    // KEY: scan the textured grid and stage the first config whose
    // KEY elects `using_qmatrix = 1`.
    let mut staged_key = None;
    'outer: for &(w, h) in &[(128u32, 128u32), (96, 80)] {
        for &q in &[140u8, 100, 180] {
            let f = natural(w, h, 0);
            let k = encode_key_frame_yuv420_with_q(&f, q).expect("key encode");
            if k.fh
                .quantization_params
                .as_ref()
                .is_some_and(|qp| qp.using_qmatrix)
            {
                staged_key = Some((w, h, q, k));
                break 'outer;
            }
        }
    }
    let (kw, kh, kq, k) = staged_key.expect("a textured KEY config must elect the QM arm");
    std::fs::write(
        root.join(format!("kf-{kw}x{kh}-q{kq}-qm.ivf")),
        &k.ivf_bytes,
    )
    .expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    yuv.extend_from_slice(&k.recon_y);
    yuv.extend_from_slice(&k.recon_u);
    yuv.extend_from_slice(&k.recon_v);
    std::fs::write(root.join(format!("kf-{kw}x{kh}-q{kq}-qm.yuv")), &yuv).expect("write yuv");

    // GOP: 4 textured frames at q100 (the measured election point).
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| natural(96, 80, t)).collect();
    let on = arm(&frames, 100, true);
    let ids = wire_qm(&on.gop.temporal_units, 96, 80);
    assert!(
        ids.iter().any(|&(u, _)| u),
        "staged GOP must elect the QM arm: {ids:?}"
    );
    std::fs::write(root.join("gop-96x80-q100-qm.ivf"), &on.gop.ivf_bytes).expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &on.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("gop-96x80-q100-qm.yuv"), &yuv).expect("write yuv");
    let off = arm(&frames, 100, false);
    std::fs::write(
        root.join("qm-staging-notes.txt"),
        format!(
            "staged kf: {kw}x{kh} q{kq} wire {:?}\ngop wire qm: {:?}\ngop elections: {:?}\ngop bytes elected: {}\ngop bytes flat baseline: {}\n",
            wire_qm(core::slice::from_ref(&k.temporal_unit_bytes), kw, kh),
            ids,
            on.qm_elections,
            on.gop.ivf_bytes.len(),
            off.gop.ivf_bytes.len(),
        ),
    )
    .expect("write notes");
}
