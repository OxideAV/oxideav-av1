//! r456 — the §5.9.30 `ar_coeff_lag = 3` ring on the election ladder:
//! the correlation-match term now scores horizontal lags 1..=3 and a
//! lag-3 candidate is offered where the residual carries distance-3
//! structure, so grain correlated ONLY three columns apart elects the
//! 24-tap ring; white grain keeps the r441/r444 shape.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.30, §7.18.3.3.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, GopTuning, TunedGop, Yuv420Frame,
};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

fn base_value(r: usize, c: usize, t: usize) -> f64 {
    let x = c as f64 + 1.1 * t as f64;
    let y = r as f64 + 0.5 * t as f64;
    120.0 + 60.0 * (0.021 * x).sin() * (0.026 * y).cos() + 18.0 * (0.047 * (x + y)).sin()
}

/// Smooth drifting base plus noise re-rolled per frame. `lag3 = true`
/// sums each white sample with the one three columns to its left
/// (ρ(1) = ρ(2) = 0, ρ(3) = ½ — structure ONLY at distance 3).
fn grainy(w: u32, h: u32, t: usize, amp: i32, lag3: bool) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    let mut state = 0x2454_1013u32.wrapping_add((t as u32).wrapping_mul(0x9e37_79b9));
    let mut rnd = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (((state >> 23) & 255) as i32 - 128) * amp / 128
    };
    for r in 0..hu {
        let mut white: Vec<i32> = Vec::with_capacity(wu);
        for _ in 0..wu {
            white.push(rnd());
        }
        for c in 0..wu {
            let n = if lag3 && c >= 3 {
                (white[c] + white[c - 3]) * 7 / 10
            } else {
                white[c]
            };
            let v = base_value(r, c, t) + f64::from(n);
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

fn encode(frames: &[Yuv420Frame], film_grain: bool) -> TunedGop {
    encode_gop_yuv420_with_q_seg_extras_tuned(
        frames,
        60,
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

/// The KEY header's `ar_coeff_lag`.
fn key_ar_lag(enc: &TunedGop) -> u8 {
    let mut seq = None;
    for desc in ObuIter::new(&enc.gop.temporal_units[0]) {
        let desc = desc.expect("TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
            ObuType::Frame => {
                let fh = parse_frame_header_with_refs(
                    desc.payload,
                    seq.as_ref().expect("SH first"),
                    &RefInfo::default(),
                )
                .expect("FH parses");
                return fh
                    .film_grain_params
                    .expect("grained KEY carries the §5.9.30 block")
                    .ar_coeff_lag;
            }
            _ => {}
        }
    }
    panic!("no KEY frame");
}

fn assert_bit_exact(enc: &TunedGop, what: &str) {
    let dec: Vec<_> = oxideav_av1::decode_av1(&enc.gop.ivf_bytes)
        .expect("decode")
        .into_iter()
        .map(|f| match f {
            Frame::Spec(s) => s,
            other => panic!("non-Spec frame {other:?}"),
        })
        .collect();
    assert_eq!(dec.len(), enc.gop.recon.len(), "{what}: frame count");
    for (i, f) in dec.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "{what}: frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "{what}: frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "{what}: frame {i} V");
    }
}

/// Distance-3 grain elects the lag-3 ring and decodes bit-exact.
#[test]
fn distance_three_grain_elects_lag_three_ring() {
    let frames: Vec<Yuv420Frame> = (0..2).map(|t| grainy(192, 160, t, 10, true)).collect();
    let on = encode(&frames, true);
    let off = encode(&frames, false);
    assert!(
        on.film_grain_elected,
        "distance-3 grain must elect the arm ({} B vs plain {} B)",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    assert_eq!(
        key_ar_lag(&on),
        3,
        "the lag-3 ring must win on distance-3 structure"
    );
    assert_bit_exact(&on, "lag-3 grain");
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_FG_LAG3_DUMP") {
        std::fs::create_dir_all(&dir).expect("dump dir");
        let name = "self-gop-192x160-q60-film-grain-lag3";
        std::fs::write(format!("{dir}/{name}.ivf"), &on.gop.ivf_bytes).expect("ivf dump");
        let mut yuv = Vec::new();
        for rc in &on.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(format!("{dir}/{name}.yuv"), yuv).expect("yuv dump");
        eprintln!(
            "{name}: {} bytes (plain {})",
            on.gop.ivf_bytes.len(),
            off.gop.ivf_bytes.len()
        );
    }
}

/// The RATE mandate still governs: on the 4-frame 128×96 GOP the
/// 24-tap ring outscores every shallower candidate on the neutrality
/// terms yet realizes MORE bytes than the plain arm (the per-header
/// cost of four lag-3 blocks outruns what the denoised frames save),
/// so a shallower ring is elected — strictly fewer bytes than plain,
/// never the lag-3 header.
#[test]
fn rate_mandate_outranks_the_neutrality_score() {
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| grainy(128, 96, t, 10, true)).collect();
    let on = encode(&frames, true);
    let off = encode(&frames, false);
    assert!(
        on.film_grain_elected,
        "distance-3 grain elects some ring on the 4-frame GOP"
    );
    assert!(
        on.gop.ivf_bytes.len() < off.gop.ivf_bytes.len(),
        "elected arm codes fewer bytes"
    );
    assert!(
        key_ar_lag(&on) <= 2,
        "the lag-3 header forfeits the rate mandate here"
    );
    assert_bit_exact(&on, "rate-mandate grain");
}

/// White grain keeps a shallow ring (the lag-3 candidate is gated off
/// and the deeper correlation gaps stay ~0).
#[test]
fn white_grain_keeps_shallow_ring() {
    let frames: Vec<Yuv420Frame> = (0..2).map(|t| grainy(128, 96, t, 8, false)).collect();
    let on = encode(&frames, true);
    assert!(on.film_grain_elected, "white grain must elect the arm");
    assert!(
        key_ar_lag(&on) <= 1,
        "white grain never pays for a deep ring"
    );
    assert_bit_exact(&on, "white grain");
}
