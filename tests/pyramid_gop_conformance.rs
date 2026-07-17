//! Round-trip tests for the r415 conformance-grade B-pyramid GOP
//! encoder ([`oxideav_av1::encoder::encode_pyramid_gop_yuv420_with_q`]).
//!
//! Mirrors the `gop_inter_conformance.rs` contract: every produced
//! stream must decode through the crate's SPEC-FAITHFUL frame driver
//! (`decode_av1_spec`) to the encoder's own per-frame reconstruction
//! in DISPLAY order, byte-for-byte — the KEY frame, every shown B
//! frame, and every `show_existing_frame` output of a
//! decoded-not-shown ALT / MID frame. On the lossless arm
//! (`base_q_idx == 0`) the reconstruction additionally equals the
//! input planes. The external half (byte-exact decode in independent
//! black-box AV1 decoders) is validated during the round via the
//! env-gated sweep dump below and pinned via `fixture_conformance.rs`.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{encode_pyramid_gop_yuv420_with_q, EncodedGop, Yuv420Frame};

/// Deterministic textured frame with per-frame translation `(sy, sx)`.
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

/// Encode the GOP at `q` as a B-pyramid, decode through the spec
/// driver, and assert per-frame byte-exactness against the encoder
/// reconstruction in display order (and the inputs at `q == 0`).
fn assert_pyramid_round_trip(frames: &[Yuv420Frame], q: u8) -> EncodedGop {
    let (w, h) = (frames[0].width, frames[0].height);
    let enc = encode_pyramid_gop_yuv420_with_q(frames, q)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: pyramid encode failed: {e:?}"));
    assert_eq!(enc.recon.len(), frames.len());
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: spec driver rejected own pyramid: {e:?}"));
    assert_eq!(
        decoded.len(),
        frames.len(),
        "{w}x{h} q={q}: one shown frame per input, display order"
    );
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!((f.width, f.height), (w, h), "{w}x{h} q={q} display {idx}");
        assert_eq!(f.bit_depth, 8);
        assert_eq!(f.planes.len(), 3);
        let rc = &enc.recon[idx];
        assert_eq!(
            f.planes[0], rc.y,
            "{w}x{h} q={q} display {idx}: luma decode != encoder recon"
        );
        assert_eq!(
            f.planes[1], rc.u,
            "{w}x{h} q={q} display {idx}: U decode != encoder recon"
        );
        assert_eq!(
            f.planes[2], rc.v,
            "{w}x{h} q={q} display {idx}: V decode != encoder recon"
        );
        if q == 0 {
            assert_eq!(f.planes[0], frames[idx].y, "lossless display {idx}: luma");
            assert_eq!(f.planes[1], frames[idx].u, "lossless display {idx}: U");
            assert_eq!(f.planes[2], frames[idx].v, "lossless display {idx}: V");
        }
    }
    enc
}

/// Every GOP length 1..=9 must round-trip: exercises all four
/// mini-GOP plans (L = 1, 2, 3, 4) plus every tail combination and
/// the anchor-slot handover between consecutive mini-GOPs.
#[test]
fn pyramid_every_gop_length_1_to_9_round_trips() {
    for n in 1..=9usize {
        let frames: Vec<Yuv420Frame> = (0..n)
            .map(|k| moving_gradient(64, 64, 2 * k, 3 * k, 17))
            .collect();
        assert_pyramid_round_trip(&frames, 60);
    }
}

/// Lossless full pyramid (5 frames = KEY + one L=4 mini-GOP): the
/// decoded display-order output must equal the INPUT byte-for-byte
/// through the out-of-order coding, backward prediction and
/// show_existing_frame replays.
#[test]
fn pyramid_lossless_moving_content_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..5)
        .map(|k| moving_gradient(64, 64, 3 * k, 5 * k, 9))
        .collect();
    assert_pyramid_round_trip(&frames, 0);
}

/// Static content: identical frames — B frames collapse to §5.11.10
/// skip-mode / skip leaves against converged references; the §5.9.22
/// forward/backward SkipModeFrame pair is live on every B frame.
#[test]
fn pyramid_lossless_static_content_round_trips() {
    let f = moving_gradient(64, 64, 0, 0, 4);
    let frames = vec![f.clone(), f.clone(), f.clone(), f.clone(), f];
    assert_pyramid_round_trip(&frames, 0);
}

/// Lossy pyramids across the §8.3.1 coefficient-CDF q-context
/// boundaries, two full mini-GOPs (9 frames).
#[test]
fn pyramid_lossy_q_sweep_round_trips() {
    for q in [40u8, 100, 200] {
        let frames: Vec<Yuv420Frame> = (0..9)
            .map(|k| moving_gradient(64, 64, 2 * k, 3 * k, u32::from(q)))
            .collect();
        assert_pyramid_round_trip(&frames, q);
    }
}

/// Multi-superblock extent with motion (96x80, chunks 4 + 2).
#[test]
fn pyramid_lossy_multi_superblock_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..7)
        .map(|k| moving_gradient(96, 80, k, 4 * k, 88))
        .collect();
    assert_pyramid_round_trip(&frames, 60);
}

/// A content cut mid-GOP (frame 3 is unrelated): the ALT frame far
/// from its anchor forces intra fallback leaves, and the B frames
/// around the cut lean on whichever reference side matches.
#[test]
fn pyramid_lossy_content_cut_round_trips() {
    let mut frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| moving_gradient(64, 64, 2 * k, 7 * k, 21))
        .collect();
    frames.push(moving_gradient(64, 64, 50, 90, 173));
    frames.push(moving_gradient(64, 64, 52, 94, 173));
    assert_pyramid_round_trip(&frames, 60);
}

/// Half-pel motion through the pyramid (the §7.11.3.4 sub-pel taps on
/// both forward and backward references).
#[test]
fn pyramid_lossy_halfpel_motion_round_trips() {
    let base = |y: usize, x: usize| -> u8 { (((y % 97) * (x % 89)) / 31 + y / 2 + x / 3) as u8 };
    let mut frames = Vec::new();
    for k in 0..5usize {
        let (w, h) = (64u32, 64u32);
        let mut f = Yuv420Frame::filled(w, h, 0);
        for i in 0..64 {
            for j in 0..64 {
                f.y[i * 64 + j] = base(2 * i + k, 2 * j + k);
            }
        }
        for i in 0..32 {
            for j in 0..32 {
                f.u[i * 32 + j] = base(4 * i + k, 4 * j + k) / 2 + 64;
                f.v[i * 32 + j] = base(4 * i + k + 7, 4 * j + k + 3) / 2 + 32;
            }
        }
        frames.push(f);
    }
    assert_pyramid_round_trip(&frames, 50);
    assert_pyramid_round_trip(&frames, 0);
}

/// Noise content at a mid quantiser: dense coefficients on
/// bidirectional TUs.
#[test]
fn pyramid_lossy_noise_round_trips() {
    let mut state = 0x8811_2233u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut frames = Vec::new();
    for _ in 0..3 {
        let mut f = Yuv420Frame::filled(64, 64, 0);
        for p in f.y.iter_mut() {
            *p = (next() & 0xFF) as u8;
        }
        for p in f.u.iter_mut() {
            *p = (next() & 0xFF) as u8;
        }
        for p in f.v.iter_mut() {
            *p = (next() & 0xFF) as u8;
        }
        frames.push(f);
    }
    assert_pyramid_round_trip(&frames, 120);
}

/// Blend content: each middle frame is the average of its two
/// neighbours — bidirectional COMPOUND_AVERAGE territory.
#[test]
fn pyramid_lossy_blend_round_trips() {
    let f0 = moving_gradient(64, 64, 0, 0, 40);
    let f4 = moving_gradient(64, 64, 0, 0, 140);
    let lerp = |a: &Yuv420Frame, b: &Yuv420Frame, num: u16, den: u16| -> Yuv420Frame {
        let mut out = Yuv420Frame::filled(64, 64, 0);
        let mix = |x: u8, y: u8| -> u8 {
            ((u32::from(x) * u32::from(den - num) + u32::from(y) * u32::from(num)) / u32::from(den))
                as u8
        };
        for ((o, &x), &y) in out.y.iter_mut().zip(&a.y).zip(&b.y) {
            *o = mix(x, y);
        }
        for ((o, &x), &y) in out.u.iter_mut().zip(&a.u).zip(&b.u) {
            *o = mix(x, y);
        }
        for ((o, &x), &y) in out.v.iter_mut().zip(&a.v).zip(&b.v) {
            *o = mix(x, y);
        }
        out
    };
    let frames = vec![
        f0.clone(),
        lerp(&f0, &f4, 1, 4),
        lerp(&f0, &f4, 2, 4),
        lerp(&f0, &f4, 3, 4),
        f4,
    ];
    assert_pyramid_round_trip(&frames, 60);
}

/// Env-gated external-validation dump: when `OXIDEAV_AV1_SWEEP_DIR`
/// is set, encode the full config matrix and write each stream's IVF
/// plus the encoder reconstruction as concatenated yuv420p
/// (display-order) for byte-comparison against independent black-box
/// decoders. Without the env var this is a no-op shell (the same
/// configs' spec-driver round trips run in the tests above).
#[test]
fn pyramid_external_sweep_dump() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SWEEP_DIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    let geometries: [(u32, u32); 5] = [(64, 64), (96, 80), (176, 144), (64, 128), (120, 88)];
    let qs: [u8; 6] = [0, 30, 60, 100, 160, 255];
    let contents = ["move", "static", "cut", "noise", "blend", "halfpel"];
    let lengths = [2usize, 3, 4, 5, 7, 9];
    let mut count = 0u32;
    for (gi, &(w, h)) in geometries.iter().enumerate() {
        for (qi, &q) in qs.iter().enumerate() {
            // Rotate content / length with geometry and q so the
            // matrix stays dense without exploding runtime.
            let content = contents[(gi + qi) % contents.len()];
            let n = lengths[(gi * qs.len() + qi) % lengths.len()];
            let frames = build_content(content, w, h, n, (gi * 31 + qi * 7) as u32);
            let enc = assert_pyramid_round_trip(&frames, q);
            let name = format!("pyr-{w}x{h}-q{q}-{content}-n{n}");
            std::fs::write(format!("{dir}/{name}.ivf"), &enc.ivf_bytes).unwrap();
            let mut yuv = Vec::new();
            for rc in &enc.recon {
                yuv.extend_from_slice(&rc.y);
                yuv.extend_from_slice(&rc.u);
                yuv.extend_from_slice(&rc.v);
            }
            std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
            count += 1;
        }
    }
    eprintln!("pyramid sweep: wrote {count} streams to {dir}");
}

/// Content builder for the sweep matrix.
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
                if k == n / 2 {
                    moving_gradient(w, h, 50, 90, seed.wrapping_add(700))
                } else {
                    moving_gradient(w, h, 2 * k, 5 * k, seed)
                }
            })
            .collect(),
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
        "blend" => {
            let a = moving_gradient(w, h, 0, 0, seed);
            let b = moving_gradient(w, h, 0, 0, seed.wrapping_add(100));
            (0..n)
                .map(|k| {
                    let num = k as u32;
                    let den = (n - 1).max(1) as u32;
                    let mut out = Yuv420Frame::filled(w, h, 0);
                    let mix = |x: u8, y: u8| -> u8 {
                        ((u32::from(x) * (den - num) + u32::from(y) * num) / den) as u8
                    };
                    for ((o, &x), &y) in out.y.iter_mut().zip(&a.y).zip(&b.y) {
                        *o = mix(x, y);
                    }
                    for ((o, &x), &y) in out.u.iter_mut().zip(&a.u).zip(&b.u) {
                        *o = mix(x, y);
                    }
                    for ((o, &x), &y) in out.v.iter_mut().zip(&a.v).zip(&b.v) {
                        *o = mix(x, y);
                    }
                    out
                })
                .collect()
        }
        _ => {
            // halfpel
            let base =
                |y: usize, x: usize| -> u8 { (((y % 97) * (x % 89)) / 31 + y / 2 + x / 3) as u8 };
            (0..n)
                .map(|k| {
                    let mut f = Yuv420Frame::filled(w, h, 0);
                    let (wu, hu) = (w as usize, h as usize);
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
                .collect()
        }
    }
}
