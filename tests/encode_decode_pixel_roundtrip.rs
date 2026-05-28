//! Integration test for round 224 (arc 18) — the **full encode →
//! decode → pixel-equality** roundtrip via the public API.
//!
//! Composes:
//!   * Encoder side: [`oxideav_av1::encoder::encode_intra_frame_yuv`]
//!     (introduced in arc 17 / round 223) — 4:2:0 YUV 16×16 in,
//!     IVF v0 bytes out.
//!   * Decoder side: [`oxideav_av1::decode_av1`] (introduced in arc 18
//!     / round 224) — IVF v0 bytes in, `Vec<Frame>` out.
//!
//! The lossless WHT chain at `base_q_idx = 0` guarantees bit-exact
//! pixel recovery on arbitrary 4:2:0 YUV inputs.
//!
//! This is the milestone exercise that closes the
//! encoder-side → decoder-side loop on the public API surface.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{encode_intra_frame_y, encode_intra_frame_yuv, Yuv420Frame16x16};
use oxideav_av1::{decode_av1, parse_frame_header, parse_sequence_header};

const TINY_SEQ_PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
const TINY_FRAME_PAYLOAD: &[u8] = &[
    0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67, 0x6c,
    0xc7, 0xee, 0x51, 0x80,
];

fn tiny_descriptors() -> (
    oxideav_av1::sequence_header::SequenceHeader,
    oxideav_av1::frame_header::FrameHeader,
) {
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap();
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).unwrap();
    (seq, fh)
}

#[test]
fn encode_decode_flat_grey_yuv_roundtrip() {
    let (seq, fh) = tiny_descriptors();
    let input = Yuv420Frame16x16::default(); // every plane all-128
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    assert_eq!(decoded.len(), 1, "one frame in, one frame out");
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y, "luma plane bit-exact recovery");
            assert_eq!(u, &input.u, "U plane bit-exact recovery");
            assert_eq!(v, &input.v, "V plane bit-exact recovery");
        }
    }
}

#[test]
fn encode_decode_horizontal_chroma_gradient_roundtrip() {
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    // Flat luma at 100, U gradient 16->232, V gradient 232->16 across cols.
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = 100;
        }
    }
    for row in input.u.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (16 + (j as u8) * 27) % 251;
        }
    }
    for row in input.v.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (232u8).wrapping_sub((j as u8) * 27);
        }
    }
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y);
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
    }
}

#[test]
fn encode_decode_pseudo_random_yuv_roundtrip() {
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    // LCG over the 256 luma + 128 chroma sample positions.
    let mut rng: u32 = 0xDEADBEEF;
    let mut next = || -> u8 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 16) & 0xFF) as u8
    };
    for row in input.y.iter_mut() {
        for cell in row.iter_mut() {
            *cell = next();
        }
    }
    for row in input.u.iter_mut() {
        for cell in row.iter_mut() {
            *cell = next();
        }
    }
    for row in input.v.iter_mut() {
        for cell in row.iter_mut() {
            *cell = next();
        }
    }
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y, "luma plane bit-exact on pseudo-random input");
            assert_eq!(u, &input.u, "U plane bit-exact on pseudo-random input");
            assert_eq!(v, &input.v, "V plane bit-exact on pseudo-random input");
        }
    }
}

#[test]
fn encode_decode_extremes_yuv_roundtrip() {
    let (seq, fh) = tiny_descriptors();
    // 0/255 extremes — exercises the §7.13.3 lossless clamp at the
    // §7.12.3 step-1 lattice boundary.
    let mut input = Yuv420Frame16x16::default();
    for (i, row) in input.y.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if (i + j) & 1 == 0 { 0 } else { 255 };
        }
    }
    for (i, row) in input.u.iter_mut().enumerate() {
        for cell in row.iter_mut() {
            *cell = if i & 1 == 0 { 0 } else { 255 };
        }
    }
    for row in input.v.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if j & 1 == 0 { 0 } else { 255 };
        }
    }
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y);
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
    }
}

#[test]
fn y_only_encode_path_does_not_roundtrip_under_yuv_sh() {
    // The Y-only `encode_intra_frame_y` path emits no chroma syntax
    // under the tiny fixture's `monochrome = false` SH. A spec-
    // conformant decoder expects chroma reads on the HasChroma cells,
    // so the public `decode_av1` rejects the stream rather than
    // hallucinating chroma samples. The full 4:2:0 YUV roundtrip is the
    // arc-18 milestone — see the YUV tests above.
    let (seq, fh) = tiny_descriptors();
    let mut luma = [[0u8; 16]; 16];
    for (i, row) in luma.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = ((11 * i + 23 * j) & 0xFF) as u8;
        }
    }
    let encoded = encode_intra_frame_y(&luma, &seq, &fh).unwrap();
    let res = decode_av1(&encoded.ivf_bytes);
    assert!(res.is_err());
}
