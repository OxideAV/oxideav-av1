//! Public-API encode → decode round trips (r428 rewrite).
//!
//! Until r428 this file pinned the HISTORICAL encoder-mirror
//! drivers: the fixed-16×16 and dyn-extent intra encoders whose
//! non-conformant output only the matching mirror decode arm could
//! read. That whole surface is retired (see the r428 CHANGELOG
//! entry) — `encode_av1` has emitted spec-conformant streams since
//! r409, and `decode_av1` now rides the spec-faithful driver
//! exclusively, surfacing every shown frame as
//! [`oxideav_av1::decoder::Frame::Spec`].
//!
//! The rewritten suite keeps this file's ROLE — the public-API
//! pixel-roundtrip gate — on the conformance-grade encoders: every
//! stream must decode through the public [`oxideav_av1::decode_av1`]
//! to the encoder's own reconstruction sample-exact (and to the
//! INPUT planes exactly on the lossless arm), across the dimension /
//! quantiser / content axes the mirror suite historically walked
//! (including the old fixed 16×16 shape), plus monochrome and the
//! non-4:2:0 pairings through the general entries.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q, encode_key_frame_yuv420_with_q, encode_key_frame_yuv_with_q,
    ChromaFormat, Yuv420Frame, YuvFrame,
};
use oxideav_av1::{decode_av1, encode_av1};

// ---------------------------------------------------------------------
// Content generators (the mirror suite's shapes: flat, gradient,
// noise-like texture, hard checker edges).
// ---------------------------------------------------------------------

fn gradient(w: u32, h: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = ((r * 3 + c * 5) % 256) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = ((128 + r * 2 + c) % 256) as u8;
            f.v[r * cw + c] = ((64 + r + c * 2) % 256) as u8;
        }
    }
    f
}

/// Deterministic LCG "noise" — the mirror suite's worst-case
/// full-range texture.
fn noise(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let mut state = seed | 1;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    };
    let mut f = Yuv420Frame::filled(w, h, 0);
    for v in f.y.iter_mut().chain(f.u.iter_mut()).chain(f.v.iter_mut()) {
        *v = next();
    }
    f
}

fn checker(w: u32, h: u32, cell: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = if ((r / cell) + (c / cell)) % 2 == 0 {
                225
            } else {
                40
            };
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = if ((r / cell) + (c / cell)) % 2 == 0 {
                90
            } else {
                170
            };
            f.v[r * cw + c] = 128;
        }
    }
    f
}

// ---------------------------------------------------------------------
// Assertion helpers.
// ---------------------------------------------------------------------

/// Every decoded frame must surface as `Frame::Spec` — the mirror
/// variants are gone, and a conformance-grade stream must never take
/// any other path.
fn spec_frames(ivf: &[u8], expected: usize, label: &str) -> Vec<oxideav_av1::decoder::SpecFrame> {
    let frames =
        decode_av1(ivf).unwrap_or_else(|e| panic!("{label}: decode_av1 rejected stream: {e:?}"));
    assert_eq!(frames.len(), expected, "{label}: shown frame count");
    frames
        .into_iter()
        .map(|f| match f {
            Frame::Spec(s) => s,
            #[allow(unreachable_patterns)]
            other => panic!("{label}: non-Spec frame variant {other:?}"),
        })
        .collect()
}

/// KEY-frame round trip at `q` on 8-bit 4:2:0 content: decode ==
/// encoder recon always; decode == input exactly at `q == 0`.
fn assert_key_round_trip(frame: &Yuv420Frame, q: u8, label: &str) {
    let enc = encode_key_frame_yuv420_with_q(frame, q)
        .unwrap_or_else(|e| panic!("{label} q{q}: encode failed: {e:?}"));
    let decoded = spec_frames(&enc.ivf_bytes, 1, label);
    let f = &decoded[0];
    assert_eq!((f.width, f.height), (frame.width, frame.height), "{label}");
    assert_eq!(f.bit_depth, 8, "{label}");
    assert_eq!(f.planes[0], enc.recon_y, "{label} q{q}: luma != recon");
    assert_eq!(f.planes[1], enc.recon_u, "{label} q{q}: U != recon");
    assert_eq!(f.planes[2], enc.recon_v, "{label} q{q}: V != recon");
    if q == 0 {
        assert_eq!(f.planes[0], frame.y, "{label} lossless: luma != input");
        assert_eq!(f.planes[1], frame.u, "{label} lossless: U != input");
        assert_eq!(f.planes[2], frame.v, "{label} lossless: V != input");
    }
}

// ---------------------------------------------------------------------
// The historical dimension × quantiser × content matrix, now on the
// conformance-grade KEY encoder through the public decode.
// ---------------------------------------------------------------------

/// The old fixed-mirror shape: 16×16 across the quantiser ladder.
#[test]
fn key_16x16_all_q_round_trip() {
    for q in [0u8, 60, 120, 255] {
        assert_key_round_trip(&gradient(16, 16), q, "16x16-gradient");
        assert_key_round_trip(&noise(16, 16, 7), q, "16x16-noise");
    }
}

/// Minimum extent.
#[test]
fn key_8x8_round_trips() {
    for q in [0u8, 100, 255] {
        assert_key_round_trip(&noise(8, 8, 3), q, "8x8-noise");
    }
}

/// Rectangular extents (the dyn mirror's axis), lossless + lossy.
#[test]
fn key_rectangular_extents_round_trip() {
    for (w, h) in [(24u32, 40u32), (64, 16), (16, 64), (56, 32)] {
        assert_key_round_trip(&gradient(w, h), 0, "rect-gradient");
        assert_key_round_trip(&checker(w, h, 5), 90, "rect-checker");
    }
}

/// Multi-superblock extents (the multi-SB mirror's axis).
#[test]
fn key_multi_superblock_round_trips() {
    for (w, h) in [(96u32, 80u32), (128, 64), (128, 128)] {
        assert_key_round_trip(&gradient(w, h), 0, "msb-gradient");
        assert_key_round_trip(&noise(w, h, 21), 120, "msb-noise");
    }
}

/// Hard-edge content across the ladder (worst case for the lossy
/// arm's ringing; CDEF may or may not elect — recon equality is the
/// contract either way).
#[test]
fn key_checker_all_q_round_trip() {
    for q in [0u8, 60, 160, 255] {
        assert_key_round_trip(&checker(64, 64, 7), q, "64x64-checker");
    }
}

// ---------------------------------------------------------------------
// Monochrome + non-4:2:0 pairings through the general entry (the
// y-only mirror suite's replacement).
// ---------------------------------------------------------------------

fn textured_wide(w: u32, h: u32, bit_depth: u8, fmt: ChromaFormat) -> YuvFrame {
    let mut f = YuvFrame::filled(w, h, bit_depth, fmt, 0);
    let (wu, hu) = (w as usize, h as usize);
    let maxv = (1u32 << bit_depth) - 1;
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] =
                (((r * 37 + c * 59 + (r / 8) * (c / 8)) as u32 * maxv) / 4096).min(maxv) as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for r in 0..ch {
        for c in 0..cw {
            let base = (maxv / 2) as usize;
            if !f.u.is_empty() {
                f.u[r * cw + c] = ((base + r * 5 + c * 3) as u32).min(maxv) as u16;
                f.v[r * cw + c] = ((base / 2 + r * 2 + c * 7) as u32).min(maxv) as u16;
            }
        }
    }
    f
}

fn plane_bytes(bit_depth: u8, p: &[u16]) -> Vec<u8> {
    if bit_depth == 8 {
        p.iter().map(|&s| s as u8).collect()
    } else {
        p.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}

/// Monochrome KEY frames across depths and quantisers — the y-only
/// mirror suite's coverage, now through the general conformance
/// encoder + public decode.
#[test]
fn key_monochrome_round_trips() {
    for bd in [8u8, 10, 12] {
        for q in [0u8, 80, 200] {
            let frame = textured_wide(48, 32, bd, ChromaFormat::Monochrome);
            let enc = encode_key_frame_yuv_with_q(&frame, q)
                .unwrap_or_else(|e| panic!("mono bd{bd} q{q}: encode failed: {e:?}"));
            let decoded = spec_frames(&enc.ivf_bytes, 1, "mono");
            let f = &decoded[0];
            assert_eq!(f.bit_depth, bd);
            assert_eq!(f.planes.len(), 1, "monochrome surfaces luma only");
            assert_eq!(
                f.planes[0],
                plane_bytes(bd, &enc.recon_y),
                "mono bd{bd} q{q}"
            );
            if q == 0 {
                assert_eq!(f.planes[0], plane_bytes(bd, &frame.y), "mono lossless");
            }
        }
    }
}

/// One 4:2:2 and one 4:4:4 KEY pairing through the public decode
/// (the format matrix suite walks all twelve pairings against the
/// spec driver; this pins the PUBLIC entry's parity).
#[test]
fn key_non_420_public_decode_round_trips() {
    for (bd, fmt) in [(10u8, ChromaFormat::Yuv422), (8u8, ChromaFormat::Yuv444)] {
        let frame = textured_wide(64, 48, bd, fmt);
        let enc = encode_key_frame_yuv_with_q(&frame, 70)
            .unwrap_or_else(|e| panic!("{fmt:?} bd{bd}: encode failed: {e:?}"));
        let decoded = spec_frames(&enc.ivf_bytes, 1, "non-420");
        let f = &decoded[0];
        assert_eq!(f.planes[0], plane_bytes(bd, &enc.recon_y));
        assert_eq!(f.planes[1], plane_bytes(bd, &enc.recon_u));
        assert_eq!(f.planes[2], plane_bytes(bd, &enc.recon_v));
    }
}

// ---------------------------------------------------------------------
// The public top-level entries.
// ---------------------------------------------------------------------

/// `encode_av1` (lossless KEY) → `decode_av1` recovers the input
/// planes byte-for-byte through `Frame::Spec`.
#[test]
fn public_encode_av1_lossless_round_trips() {
    for (w, h) in [(16u32, 16u32), (64, 64), (96, 80)] {
        let frame = gradient(w, h);
        let mut blob = frame.y.clone();
        blob.extend_from_slice(&frame.u);
        blob.extend_from_slice(&frame.v);
        let ivf =
            encode_av1(&blob, w, h).unwrap_or_else(|e| panic!("{w}x{h}: encode_av1 failed: {e:?}"));
        let decoded = spec_frames(&ivf, 1, "encode_av1");
        assert_eq!(decoded[0].planes[0], frame.y, "{w}x{h} luma");
        assert_eq!(decoded[0].planes[1], frame.u, "{w}x{h} U");
        assert_eq!(decoded[0].planes[2], frame.v, "{w}x{h} V");
    }
}

/// A public-API inter GOP: every shown frame surfaces as
/// `Frame::Spec` equal to the encoder reconstruction.
#[test]
fn public_gop_round_trips_through_decode_av1() {
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| {
            let mut f = gradient(64, 64);
            for v in f.y.iter_mut() {
                *v = v.wrapping_add(2 * k);
            }
            f
        })
        .collect();
    for q in [0u8, 80] {
        let enc = encode_gop_yuv420_with_q(&frames, q)
            .unwrap_or_else(|e| panic!("gop q{q}: encode failed: {e:?}"));
        let decoded = spec_frames(&enc.ivf_bytes, frames.len(), "gop");
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], enc.recon[idx].y, "gop q{q} frame {idx} luma");
            assert_eq!(f.planes[1], enc.recon[idx].u, "gop q{q} frame {idx} U");
            assert_eq!(f.planes[2], enc.recon[idx].v, "gop q{q} frame {idx} V");
        }
    }
}

/// Malformed inputs surface typed errors (never panics) — the mirror
/// arm's rejection duties now fall entirely to the spec driver.
#[test]
fn malformed_streams_error_cleanly() {
    assert!(decode_av1(&[]).is_err(), "empty buffer");
    assert!(decode_av1(&[0u8; 16]).is_err(), "garbage header");
    let frame = gradient(16, 16);
    let enc = encode_key_frame_yuv420_with_q(&frame, 0).unwrap();
    let mut truncated = enc.ivf_bytes.clone();
    truncated.truncate(enc.ivf_bytes.len() / 2);
    assert!(decode_av1(&truncated).is_err(), "truncated stream");
}
