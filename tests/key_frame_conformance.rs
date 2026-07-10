//! Round-trip + conformance-shape tests for the r409 conformance-grade
//! KEY-frame encoder ([`oxideav_av1::encoder::encode_key_frame_yuv420`]).
//!
//! The gold standard for these streams is external: they decode
//! byte-identically to the encoder's own reconstruction in independent
//! AV1 decoders (validated as opaque black-box binaries during the
//! round; the pinned expected-pixel fixtures live in
//! `fixture_conformance.rs`). In-tree, these tests pin the internal
//! half of the contract:
//!
//!   * the crate's SPEC-FAITHFUL frame driver (`decode_av1_spec`, the
//!     same driver that decodes the 39-stream conformance corpus
//!     byte-exact) decodes every produced stream to the encoder's
//!     reconstruction, byte-for-byte;
//!   * on the lossless arm the reconstruction equals the input;
//!   * the public `decode_av1` surfaces the same frames as
//!     `Frame::Spec` (the encoder-mirror path must never claim these
//!     conformant streams).

use oxideav_av1::decoder::{decode_av1_spec, Frame};
use oxideav_av1::encoder::{encode_key_frame_yuv420_with_q, Yuv420Frame, KEY_FRAME_MAX_DIM};

/// Deterministic textured input (distinct per-plane gradients).
fn gradient(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let s = seed as usize;
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            f.y[i * wu + j] = ((i * 3 + j * 7 + s) % 256) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = ((128 + i * 5 + j + s) % 256) as u8;
        }
    }
    for i in 0..ch {
        for j in 0..cw {
            f.v[i * cw + j] = ((64 + i + j * 4 + s) % 256) as u8;
        }
    }
    f
}

/// Encode at `q`, decode through the spec driver, and assert the
/// decoded planes equal the encoder reconstruction byte-for-byte (and
/// the input too when `q == 0`). Also asserts public-API parity.
fn assert_key_frame_round_trip(w: u32, h: u32, q: u8) {
    let input = gradient(w, h, u32::from(q));
    let enc = encode_key_frame_yuv420_with_q(&input, q)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: encode failed: {e:?}"));
    let frames = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: spec driver rejected own stream: {e:?}"));
    assert_eq!(frames.len(), 1, "{w}x{h} q={q}: one shown frame");
    let f = &frames[0];
    assert_eq!((f.width, f.height), (w, h), "{w}x{h} q={q}: extents");
    assert_eq!(f.bit_depth, 8);
    assert_eq!(f.planes.len(), 3);
    assert_eq!(
        f.planes[0], enc.recon_y,
        "{w}x{h} q={q}: luma decode != encoder recon"
    );
    assert_eq!(
        f.planes[1], enc.recon_u,
        "{w}x{h} q={q}: U decode != encoder recon"
    );
    assert_eq!(
        f.planes[2], enc.recon_v,
        "{w}x{h} q={q}: V decode != encoder recon"
    );
    if q == 0 {
        assert_eq!(f.planes[0], input.y, "{w}x{h} lossless luma != input");
        assert_eq!(f.planes[1], input.u, "{w}x{h} lossless U != input");
        assert_eq!(f.planes[2], input.v, "{w}x{h} lossless V != input");
    }
    // Public-API parity: the conformant stream must ride the spec
    // path (Frame::Spec), never the encoder-mirror path.
    let pub_frames = oxideav_av1::decode_av1(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: public decode_av1 rejected: {e:?}"));
    assert_eq!(pub_frames.len(), 1);
    match &pub_frames[0] {
        Frame::Spec(s) => assert_eq!(s, f, "{w}x{h} q={q}: public-API frame != spec frame"),
        other => panic!("{w}x{h} q={q}: conformant stream rode the mirror path: {other:?}"),
    }
}

#[test]
fn key_frame_lossless_round_trips_single_superblock() {
    assert_key_frame_round_trip(64, 64, 0);
    assert_key_frame_round_trip(16, 16, 0);
    assert_key_frame_round_trip(8, 8, 0);
}

#[test]
fn key_frame_lossless_round_trips_multi_superblock() {
    assert_key_frame_round_trip(96, 80, 0);
    assert_key_frame_round_trip(128, 64, 0);
}

#[test]
fn key_frame_lossy_decodes_to_encoder_recon() {
    assert_key_frame_round_trip(64, 64, 80);
    assert_key_frame_round_trip(96, 80, 120);
    assert_key_frame_round_trip(176, 144, 50);
    assert_key_frame_round_trip(8, 8, 200);
}

#[test]
fn key_frame_flat_grey_is_all_skip_and_tiny() {
    // Flat mid-grey: DC prediction is exact at every leaf, so every
    // TU quantises to zero and every leaf codes skip = 1 — the stream
    // should be very small and still round-trip.
    let input = Yuv420Frame::filled(64, 64, 128);
    let enc = encode_key_frame_yuv420_with_q(&input, 0).unwrap();
    assert!(
        enc.ivf_bytes.len() < 200,
        "all-skip 64x64 stream should be tiny, got {} bytes",
        enc.ivf_bytes.len()
    );
    let frames = decode_av1_spec(&enc.ivf_bytes).unwrap();
    assert_eq!(frames[0].planes[0], input.y);
    assert_eq!(frames[0].planes[1], input.u);
    assert_eq!(frames[0].planes[2], input.v);
}

#[test]
fn key_frame_encode_is_deterministic() {
    let input = gradient(64, 64, 7);
    let a = encode_key_frame_yuv420_with_q(&input, 40).unwrap();
    let b = encode_key_frame_yuv420_with_q(&input, 40).unwrap();
    assert_eq!(a.ivf_bytes, b.ivf_bytes);
    assert_eq!(a.recon_y, b.recon_y);
}

#[test]
fn key_frame_rejects_bad_dimensions() {
    for (w, h) in [
        (0u32, 64u32),
        (12, 64),
        (64, 4),
        (KEY_FRAME_MAX_DIM + 8, 64),
    ] {
        let f = Yuv420Frame::filled(w.max(8), h.max(8), 0);
        let mut f = f;
        f.width = w;
        f.height = h;
        assert!(
            encode_key_frame_yuv420_with_q(&f, 0).is_err(),
            "{w}x{h} must be rejected"
        );
    }
    // Plane-length mismatch.
    let mut f = Yuv420Frame::filled(64, 64, 0);
    f.y.pop();
    assert!(encode_key_frame_yuv420_with_q(&f, 0).is_err());
}
