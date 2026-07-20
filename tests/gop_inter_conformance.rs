//! Round-trip tests for the r411 conformance-grade KEY + P GOP
//! encoder ([`oxideav_av1::encoder::encode_gop_yuv420_with_q`]).
//!
//! Mirrors the `key_frame_conformance.rs` contract: every produced
//! stream must decode through the crate's SPEC-FAITHFUL frame driver
//! (`decode_av1_spec`, the driver that decodes the 44-stream
//! conformance corpus byte-exact) to the encoder's own per-frame
//! reconstruction, byte-for-byte — KEY frame and every INTER P-frame.
//! On the lossless arm (`base_q_idx == 0`) the reconstruction
//! additionally equals the input planes. The external half (byte-exact
//! decode in independent black-box AV1 decoders) is validated during
//! the round and pinned via `fixture_conformance.rs`.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q, Yuv420Frame};

/// Deterministic textured frame with per-frame translation `(sy, sx)`
/// — consecutive frames are shifted copies plus a mild luma ramp, so
/// the motion search finds real non-zero vectors and residuals stay
/// small but non-trivial.
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

/// Encode the GOP at `q`, decode through the spec driver, and assert
/// per-frame byte-exactness against the encoder reconstruction (and
/// the inputs at `q == 0`).
fn assert_gop_round_trip(frames: &[Yuv420Frame], q: u8) {
    let (w, h) = (frames[0].width, frames[0].height);
    let enc = encode_gop_yuv420_with_q(frames, q)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: GOP encode failed: {e:?}"));
    assert_eq!(enc.recon.len(), frames.len());
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: spec driver rejected own GOP: {e:?}"));
    assert_eq!(
        decoded.len(),
        frames.len(),
        "{w}x{h} q={q}: one shown frame per input"
    );
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!((f.width, f.height), (w, h), "{w}x{h} q={q} frame {idx}");
        assert_eq!(f.bit_depth, 8);
        assert_eq!(f.planes.len(), 3);
        let rc = &enc.recon[idx];
        assert_eq!(
            f.planes[0], rc.y,
            "{w}x{h} q={q} frame {idx}: luma decode != encoder recon"
        );
        assert_eq!(
            f.planes[1], rc.u,
            "{w}x{h} q={q} frame {idx}: U decode != encoder recon"
        );
        assert_eq!(
            f.planes[2], rc.v,
            "{w}x{h} q={q} frame {idx}: V decode != encoder recon"
        );
        if q == 0 {
            assert_eq!(
                f.planes[0], frames[idx].y,
                "{w}x{h} lossless frame {idx}: luma != input"
            );
            assert_eq!(
                f.planes[1], frames[idx].u,
                "{w}x{h} lossless frame {idx}: U != input"
            );
            assert_eq!(
                f.planes[2], frames[idx].v,
                "{w}x{h} lossless frame {idx}: V != input"
            );
        }
    }
}

/// Moving content: three frames, each a (3, 5)-per-frame translated
/// copy — the P-frames carry real NEWMV vectors.
#[test]
fn gop_lossless_moving_content_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| moving_gradient(64, 64, 3 * k, 5 * k, 9))
        .collect();
    assert_gop_round_trip(&frames, 0);
}

/// Static content: identical frames — the P-frames collapse to
/// GLOBALMV skip leaves (zero vector, zero residual).
#[test]
fn gop_lossless_static_content_round_trips() {
    let f = moving_gradient(64, 64, 0, 0, 4);
    let frames = vec![f.clone(), f.clone(), f];
    assert_gop_round_trip(&frames, 0);
}

/// Multi-superblock frame with motion + a content cut on the last
/// frame (forcing the intra-fallback arm inside a P-frame).
#[test]
fn gop_lossless_multi_superblock_with_content_cut() {
    let mut frames: Vec<Yuv420Frame> = (0..2)
        .map(|k| moving_gradient(96, 80, 2 * k, 7 * k, 21))
        .collect();
    frames.push(moving_gradient(96, 80, 50, 90, 173));
    assert_gop_round_trip(&frames, 0);
}

/// Lossy GOPs across the §8.3.1 coefficient-CDF q-context boundaries.
#[test]
fn gop_lossy_moving_content_round_trips() {
    for q in [40u8, 100, 200] {
        let frames: Vec<Yuv420Frame> = (0..3)
            .map(|k| moving_gradient(64, 64, 2 * k, 3 * k, u32::from(q)))
            .collect();
        assert_gop_round_trip(&frames, q);
    }
}

/// Lossy HD-tile-ish extent with motion (multi-superblock walk, both
/// P-frames predicting from the previous P's reconstruction).
#[test]
fn gop_lossy_multi_superblock_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| moving_gradient(176, 144, k, 4 * k, 88))
        .collect();
    assert_gop_round_trip(&frames, 60);
}

/// Half-pel motion: frames sample a smooth 2x-resolution base pattern
/// at a half-luma-sample shift per frame, so the §7.11.3.4 sub-pel
/// taps beat every integer vector and the refined quarter-pel MVs
/// carry real fractional phases.
#[test]
fn gop_lossy_halfpel_motion_round_trips() {
    let base = |y: usize, x: usize| -> u8 { (((y % 97) * (x % 89)) / 31 + y / 2 + x / 3) as u8 };
    let mut frames = Vec::new();
    for k in 0..3usize {
        let (w, h) = (64u32, 64u32);
        let mut f = Yuv420Frame::filled(w, h, 0);
        for i in 0..64 {
            for j in 0..64 {
                // 2x-resolution sample at a k-half-pel diagonal shift.
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
    assert_gop_round_trip(&frames, 50);
    assert_gop_round_trip(&frames, 0);
}

/// r419 — vertically-sheared motion (per-row horizontal shift ramp
/// misaligned with every partition boundary): the §5.11.27 OBMC
/// election territory. Round-trips at three quantisers including
/// lossless.
#[test]
fn gop_shear_motion_round_trips() {
    let tex = |i: usize, j: usize| -> u8 {
        ((i * 7 + j * 11 + (i / 4) * (j / 8) + ((i * j) / 13)) % 256) as u8
    };
    let dx_of = |i: usize, k: usize| -> usize {
        let (ramp0, ramp1, top) = (24usize, 48usize, 4 * k);
        if i < ramp0 {
            top
        } else if i >= ramp1 {
            0
        } else {
            top * (ramp1 - i) / (ramp1 - ramp0)
        }
    };
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| {
            let mut f = Yuv420Frame::filled(64, 64, 128);
            for i in 0..64 {
                for j in 0..64 {
                    f.y[i * 64 + j] = tex(i, j + dx_of(i, k));
                }
            }
            f
        })
        .collect();
    for q in [0u8, 100, 200] {
        assert_gop_round_trip(&frames, q);
    }
}

/// Noise content at a mid quantiser: dense coefficients on the inter
/// TUs and deep splits.
#[test]
fn gop_lossy_noise_round_trips() {
    let mut state = 0x8811_2233u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let mut frames = Vec::new();
    for _ in 0..2 {
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
    assert_gop_round_trip(&frames, 120);
}
