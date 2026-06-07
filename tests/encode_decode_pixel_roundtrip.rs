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
        other => panic!("expected Yuv420_16x16, got {other:?}"),
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
        other => panic!("expected Yuv420_16x16, got {other:?}"),
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
        other => panic!("expected Yuv420_16x16, got {other:?}"),
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
        other => panic!("expected Yuv420_16x16, got {other:?}"),
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

// --------------------------------------------------------------------------
// Arc r228 — 13-mode intra prediction picker tests.
//
// The encoder's `pick_best_intra_mode_4x4_y` selects whichever of the 13
// §6.10.x Y intra modes yields the smallest residual SSD against the
// input. These tests confirm:
//
//   1. A non-DC_PRED mode is actually picked on inputs designed to favour
//      directional / smooth / paeth prediction.
//   2. The full roundtrip (encode → decode → pixels) remains bit-exact on
//      arbitrary inputs (the picker change must not regress the existing
//      lossless WHT guarantee).
//   3. The encoder's `committed_y_modes` array is in-range and matches the
//      `encode_intra_frame_y` per-cell selection (one mode per BLOCK_4X4
//      leaf).
// --------------------------------------------------------------------------

#[test]
fn encoder_picks_non_dc_pred_on_horizontal_gradient_y_only() {
    // A purely-horizontal gradient (`col * 16`, constant across rows) is
    // best matched by V_PRED at most cells past the leftmost column — the
    // above-row sample equals every cell below it by construction. DC_PRED
    // would average a non-zero left col + above row to a worse value.
    let (seq, fh) = tiny_descriptors();
    let mut luma = [[0u8; 16]; 16];
    for row in luma.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (j * 16) as u8;
        }
    }
    let result = oxideav_av1::encoder::encode_intra_frame_y(&luma, &seq, &fh).unwrap();
    assert_eq!(result.committed_y_modes.len(), 16);
    let non_dc_count = result
        .committed_y_modes
        .iter()
        .filter(|&&m| m as usize != 0 /* DC_PRED ordinal */)
        .count();
    assert!(
        non_dc_count > 0,
        "expected at least one non-DC_PRED y_mode on a horizontal gradient, got {:?}",
        result.committed_y_modes
    );
}

#[test]
fn encoder_picks_non_dc_pred_on_vertical_gradient_y_only() {
    // A purely-vertical gradient (`row * 16`, constant across cols) is
    // best matched by H_PRED at most cells past the top row.
    let (seq, fh) = tiny_descriptors();
    let mut luma = [[0u8; 16]; 16];
    for (i, row) in luma.iter_mut().enumerate() {
        for cell in row.iter_mut() {
            *cell = (i * 16) as u8;
        }
    }
    let result = oxideav_av1::encoder::encode_intra_frame_y(&luma, &seq, &fh).unwrap();
    let non_dc_count = result
        .committed_y_modes
        .iter()
        .filter(|&&m| m as usize != 0)
        .count();
    assert!(
        non_dc_count > 0,
        "expected at least one non-DC_PRED y_mode on a vertical gradient, got {:?}",
        result.committed_y_modes
    );
}

#[test]
fn encoder_committed_y_modes_are_all_in_intra_mode_range() {
    // Whatever the picker selects, every entry must be in 0..=12 (the §3
    // INTRA_MODES set). Run on a pseudo-random input to stress every cell.
    let (seq, fh) = tiny_descriptors();
    let mut luma = [[0u8; 16]; 16];
    let mut state: u64 = 0xCAFE_F00D_BEEF_1234;
    for row in luma.iter_mut() {
        for cell in row.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *cell = (state >> 56) as u8;
        }
    }
    let result = oxideav_av1::encoder::encode_intra_frame_y(&luma, &seq, &fh).unwrap();
    for (cell_idx, &m) in result.committed_y_modes.iter().enumerate() {
        assert!(
            (m as usize) < 13,
            "cell {cell_idx}: y_mode = {m} out of 0..13"
        );
    }
}

#[test]
fn encode_decode_roundtrip_with_13_mode_picker_on_horizontal_gradient() {
    // The full encode → decode roundtrip must still be pixel-exact even
    // though the encoder now picks from 13 modes — the lossless WHT chain
    // is bit-exact for any prediction.
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    for row in input.y.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (j * 16) as u8;
        }
    }
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y, "Y bit-exact after 13-mode picker + decode");
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
        other => panic!("expected Yuv420_16x16, got {other:?}"),
    }
    // At least one cell should have picked a non-DC mode on the gradient.
    let non_dc_count = encoded
        .committed_y_modes
        .iter()
        .filter(|&&m| m as usize != 0)
        .count();
    assert!(
        non_dc_count > 0,
        "expected non-DC_PRED on a horizontal gradient under YUV encode, got {:?}",
        encoded.committed_y_modes
    );
}

#[test]
fn encode_decode_roundtrip_with_13_mode_picker_on_pseudorandom_yuv() {
    // Pseudo-random YUV — the picker selection per cell is essentially
    // arbitrary, but the roundtrip is still bit-exact end-to-end.
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    let mut state: u64 = 0xFACE_F00D_BAAD_BEEF;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for row in input.y.iter_mut() {
        for cell in row.iter_mut() {
            *cell = step();
        }
    }
    for row in input.u.iter_mut() {
        for cell in row.iter_mut() {
            *cell = step();
        }
    }
    for row in input.v.iter_mut() {
        for cell in row.iter_mut() {
            *cell = step();
        }
    }
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y);
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
        other => panic!("expected Yuv420_16x16, got {other:?}"),
    }
}

#[test]
fn encoder_keeps_dc_pred_on_flat_grey_input_y_only() {
    // Flat-128 input ⇒ every prediction mode produces 128 ⇒ tie ⇒ the
    // picker's enumeration order gives DC_PRED. Verifies the "no
    // gratuitous mode-switching" baseline.
    let (seq, fh) = tiny_descriptors();
    let luma = [[128u8; 16]; 16];
    let result = oxideav_av1::encoder::encode_intra_frame_y(&luma, &seq, &fh).unwrap();
    for (cell_idx, &m) in result.committed_y_modes.iter().enumerate() {
        assert_eq!(
            m as usize, 0,
            "flat-128 input: cell {cell_idx} should pick DC_PRED (ordinal 0), got {m}"
        );
    }
}

// --------------------------------------------------------------------------
// Arc r229 — chroma 13-mode intra prediction picker tests.
//
// The encoder's `pick_best_intra_mode_4x4_chroma` selects whichever of the
// 13 §6.10.x intra modes yields the smallest *combined* U+V residual SSD
// against the input chroma 4×4 block. These tests confirm:
//
//   1. The picker can select a non-DC_PRED mode on chroma surfaces with
//      a clear directional / smooth structure.
//   2. The decoder dispatches the decoded `uv_mode` correctly so the full
//      roundtrip stays pixel-exact under the lossless WHT chain.
//   3. The encoder's `committed_uv_modes` field is in-range and matches
//      the number of HasChroma cells (4 under the 4:2:0 BLOCK_4X4 walk).
//   4. Flat-grey input keeps the DC_PRED tie-break.
// --------------------------------------------------------------------------

#[test]
fn encoder_picks_non_dc_pred_on_horizontal_chroma_gradient() {
    // U and V planes carry a horizontal gradient (j * 30) while luma
    // stays flat at 100 — V_PRED is the best fit on chroma cells past
    // the leftmost column (the above-row reconstructed sample equals
    // every cell below it by construction).
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = 100;
        }
    }
    for row in input.u.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = ((j as u32) * 30) as u8;
        }
    }
    for row in input.v.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (255u8).wrapping_sub(((j as u32) * 30) as u8);
        }
    }
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let non_dc_count = encoded
        .committed_uv_modes
        .iter()
        .filter(|&&m| m as usize != 0 /* DC_PRED ordinal */)
        .count();
    assert!(
        non_dc_count > 0,
        "expected at least one non-DC_PRED uv_mode on a horizontal chroma gradient, got {:?}",
        encoded.committed_uv_modes
    );
}

#[test]
fn encoder_picks_non_dc_pred_on_vertical_chroma_gradient() {
    // Vertical chroma gradient — H_PRED is the best fit on cells past
    // the top row.
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = 100;
        }
    }
    for (i, row) in input.u.iter_mut().enumerate() {
        for cell in row.iter_mut() {
            *cell = ((i as u32) * 30) as u8;
        }
    }
    for (i, row) in input.v.iter_mut().enumerate() {
        for cell in row.iter_mut() {
            *cell = (255u8).wrapping_sub(((i as u32) * 30) as u8);
        }
    }
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let non_dc_count = encoded
        .committed_uv_modes
        .iter()
        .filter(|&&m| m as usize != 0)
        .count();
    assert!(
        non_dc_count > 0,
        "expected at least one non-DC_PRED uv_mode on a vertical chroma gradient, got {:?}",
        encoded.committed_uv_modes
    );
}

#[test]
fn encoder_committed_uv_modes_are_all_in_intra_mode_range() {
    // Every committed_uv_modes entry must be in 0..=UV_CFL_PRED (= 13)
    // — the §3 INTRA_MODES set plus the §7.11.5.3 UV_CFL_PRED slot
    // landed in r231. Run against pseudo-random chroma planes.
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    let mut state: u64 = 0xDEAD_BEEF_FEED_BABE;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    for row in input.u.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    for row in input.v.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    assert_eq!(encoded.committed_uv_modes.len(), 4);
    for (idx, &m) in encoded.committed_uv_modes.iter().enumerate() {
        assert!(
            (m as usize) <= 13,
            "chroma cell {idx}: uv_mode = {m} out of 0..=13"
        );
    }
}

#[test]
fn encode_decode_roundtrip_with_chroma_picker_on_chroma_gradient() {
    // Full encode → decode roundtrip stays bit-exact on a chroma-only
    // gradient. The lossless WHT chain is bit-exact for any prediction,
    // and the decoder's r229 dispatcher routes the decoded uv_mode
    // through the matching §7.11.2.{2..6} chroma kernel.
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = 100;
        }
    }
    for row in input.u.iter_mut() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = ((j as u32) * 30) as u8;
        }
    }
    for (i, row) in input.v.iter_mut().enumerate() {
        for cell in row.iter_mut() {
            *cell = ((i as u32) * 30) as u8;
        }
    }
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y, "Y bit-exact after chroma picker + decode");
            assert_eq!(u, &input.u, "U bit-exact after chroma picker + decode");
            assert_eq!(v, &input.v, "V bit-exact after chroma picker + decode");
        }
        other => panic!("expected Yuv420_16x16, got {other:?}"),
    }
    let non_dc_count = encoded
        .committed_uv_modes
        .iter()
        .filter(|&&m| m as usize != 0)
        .count();
    assert!(
        non_dc_count > 0,
        "expected non-DC_PRED uv_mode on chroma gradient, got {:?}",
        encoded.committed_uv_modes
    );
}

#[test]
fn encode_decode_roundtrip_with_chroma_picker_on_pseudorandom_yuv() {
    // Pseudo-random YUV — chroma picker selection per cell is
    // essentially arbitrary, but the roundtrip is still bit-exact
    // end-to-end. Mirrors the luma r228 test against the chroma path.
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    let mut state: u64 = 0xBEEF_C0DE_FACE_FACE;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    for row in input.u.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    for row in input.v.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y);
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
        other => panic!("expected Yuv420_16x16, got {other:?}"),
    }
}

#[test]
fn encoder_keeps_dc_pred_on_flat_grey_input_chroma() {
    // Flat-128 input on every plane ⇒ every chroma prediction collapses
    // to 128 ⇒ ties ⇒ DC_PRED wins on every chroma cell.
    let (seq, fh) = tiny_descriptors();
    let input = Yuv420Frame16x16::default(); // all-128 mid-grey
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    for (idx, &m) in encoded.committed_uv_modes.iter().enumerate() {
        assert_eq!(
            m as usize, 0,
            "flat-128 chroma: cell {idx} should pick DC_PRED (0), got {m}"
        );
    }
}

#[test]
fn encode_decode_roundtrip_with_chroma_picker_on_extremes() {
    // 0/255 extremes on chroma planes — the picker may select any of
    // the 13 modes per cell, but the lossless arm must still recover
    // bit-exact.
    let (seq, fh) = tiny_descriptors();
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
    let encoded = oxideav_av1::encoder::encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y);
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
        other => panic!("expected Yuv420_16x16, got {other:?}"),
    }
}

// -------------------------------------------------------------------------
// Arc r230 — dynamic-extent encode -> decode_av1 roundtrip.
//
// `encode_intra_frame_yuv_dyn` is the Vec-backed sibling of
// `encode_intra_frame_yuv`. It builds its own SH+FH from the requested
// (width, height) so the public `decode_av1` entry routes to the dyn
// decoder branch and emits `Frame::Yuv420Dyn`. These tests close the
// encoder→decoder loop on the new path for both 32×32 and 64×64
// frames — the milestone exercise this arc unlocks.
// -------------------------------------------------------------------------

#[test]
fn dyn_encode_decode_flat_grey_32x32_roundtrip() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    let input = Yuv420Frame::filled(32, 32, 128);
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    assert_eq!(decoded.len(), 1);
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 32);
            assert_eq!(*height, 32);
            assert_eq!(y, &input.y, "luma bit-exact at 32×32");
            assert_eq!(u, &input.u, "U bit-exact at 32×32");
            assert_eq!(v, &input.v, "V bit-exact at 32×32");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_pseudorandom_32x32_roundtrip_bit_exact() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    let mut input = Yuv420Frame::filled(32, 32, 0);
    let mut state: u64 = 0xCAFE_BABE_F00D_BEEF;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = step();
    }
    for p in input.u.iter_mut() {
        *p = step();
    }
    for p in input.v.iter_mut() {
        *p = step();
    }
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 32);
            assert_eq!(*height, 32);
            assert_eq!(y, &input.y, "Y mismatch 32×32 pseudo-random");
            assert_eq!(u, &input.u, "U mismatch 32×32 pseudo-random");
            assert_eq!(v, &input.v, "V mismatch 32×32 pseudo-random");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_horizontal_gradient_32x32_roundtrip_bit_exact() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    let mut input = Yuv420Frame::filled(32, 32, 0);
    let width = input.width as usize;
    let cwidth = input.chroma_width() as usize;
    for i in 0..(input.height as usize) {
        for j in 0..width {
            input.y[i * width + j] = (j * 8) as u8;
        }
    }
    for i in 0..(input.chroma_height() as usize) {
        for j in 0..cwidth {
            input.u[i * cwidth + j] = (j * 16) as u8;
            input.v[i * cwidth + j] = (255u8).wrapping_sub((j * 16) as u8);
        }
    }
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn { y, u, v, .. } => {
            assert_eq!(y, &input.y);
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_flat_grey_64x64_roundtrip() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    let input = Yuv420Frame::filled(64, 64, 200);
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 64);
            assert_eq!(*height, 64);
            assert_eq!(y, &input.y);
            assert_eq!(u, &input.u);
            assert_eq!(v, &input.v);
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_pseudorandom_64x64_roundtrip_bit_exact() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    let mut input = Yuv420Frame::filled(64, 64, 0);
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = step();
    }
    for p in input.u.iter_mut() {
        *p = step();
    }
    for p in input.v.iter_mut() {
        *p = step();
    }
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 64);
            assert_eq!(*height, 64);
            assert_eq!(y, &input.y, "Y mismatch 64×64 pseudo-random");
            assert_eq!(u, &input.u, "U mismatch 64×64 pseudo-random");
            assert_eq!(v, &input.v, "V mismatch 64×64 pseudo-random");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

// ---------------------------------------------------------------------
// r231 — UV_CFL_PRED (§7.11.5.3 chroma-from-luma) integration tests.
// ---------------------------------------------------------------------

/// Construct a 4:2:0 YUV input where the chroma planes are
/// `Cu = 128 + (y - 128) / 4` / `Cv = 128 - (y - 128) / 4` — a strong
/// linear function of the luma sample, the textbook case where
/// UV_CFL_PRED beats every §6.10.x intra mode on combined U+V SSD.
fn luma_correlated_yuv_input() -> Yuv420Frame16x16 {
    let mut input = Yuv420Frame16x16::default();
    // Build a luma gradient (column-direction) so the chroma planes
    // have a useful range to track.
    for i in 0..16usize {
        for j in 0..16usize {
            input.y[i][j] = (16u32 * j as u32) as u8;
        }
    }
    // Chroma planes: 8×8, each chroma sample tracks the corresponding
    // 2×2 luma average. Compute the average and apply the linear map.
    for ci in 0..8usize {
        for cj in 0..8usize {
            let li = ci * 2;
            let lj = cj * 2;
            let avg = (input.y[li][lj] as u32
                + input.y[li][lj + 1] as u32
                + input.y[li + 1][lj] as u32
                + input.y[li + 1][lj + 1] as u32)
                / 4;
            let centred = avg as i32 - 128;
            input.u[ci][cj] = (128i32 + centred / 4).clamp(0, 255) as u8;
            input.v[ci][cj] = (128i32 - centred / 4).clamp(0, 255) as u8;
        }
    }
    input
}

#[test]
fn encode_decode_cfl_yuv_roundtrip_bit_exact() {
    // r231: full encode → decode → pixel-equality on a chroma signal
    // that's a clean linear function of luma. The encoder picks
    // UV_CFL_PRED for at least one chroma cell; the decoder mirrors
    // its CFL prediction and the lossless WHT chain stays bit-exact.
    let (seq, fh) = tiny_descriptors();
    let input = luma_correlated_yuv_input();
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            assert_eq!(y, &input.y, "luma plane bit-exact on CFL input");
            assert_eq!(u, &input.u, "U plane bit-exact on CFL input");
            assert_eq!(v, &input.v, "V plane bit-exact on CFL input");
        }
        other => panic!("expected Yuv420_16x16, got {other:?}"),
    }
}

#[test]
fn encoder_picks_cfl_on_luma_correlated_chroma() {
    // r231: the chroma picker should commit UV_CFL_PRED (= 13) for at
    // least one chroma cell on a luma-correlated input. Confirms that
    // the §5.11.45 `read_cfl_alphas` arm fires in the bitstream, not
    // just that the encoder *could* pick CFL.
    let (seq, fh) = tiny_descriptors();
    let input = luma_correlated_yuv_input();
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    let cfl_count = encoded
        .committed_uv_modes
        .iter()
        .filter(|&&m| m as usize == 13)
        .count();
    assert!(
        cfl_count > 0,
        "expected at least one UV_CFL_PRED pick on luma-correlated chroma, got modes {:?}",
        encoded.committed_uv_modes
    );
}

#[test]
fn encoder_committed_uv_modes_include_zero_or_more_cfl_picks() {
    // Sanity-check companion to `encoder_picks_cfl_on_luma_correlated_chroma`:
    // on a uniformly random YUV input where chroma is uncorrelated
    // with luma, the picker is allowed to pick CFL but typically
    // settles on a §6.10.x mode. Either way, every entry stays in
    // `0..=13`.
    let (seq, fh) = tiny_descriptors();
    let mut input = Yuv420Frame16x16::default();
    let mut state: u64 = 0xABAD_BABA_DEAD_BEEF;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for row in input.y.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    for row in input.u.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    for row in input.v.iter_mut() {
        for c in row.iter_mut() {
            *c = step();
        }
    }
    let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
    for (idx, &m) in encoded.committed_uv_modes.iter().enumerate() {
        assert!(
            (m as usize) <= 13,
            "chroma cell {idx}: uv_mode = {m} out of 0..=13"
        );
    }
}

// ---------------------------------------------------------------------
// r232 — UV_CFL_PRED on the dynamic-extent driver (§7.11.5.3 +
// `encode_intra_frame_yuv_dyn` / `Frame::Yuv420Dyn`).
// ---------------------------------------------------------------------

/// Construct a 4:2:0 YUV input at arbitrary (width, height) where the
/// chroma planes are `Cu = 128 + (lumaAvg2x2 - 128) / 4` /
/// `Cv = 128 - (lumaAvg2x2 - 128) / 4` — the same textbook CFL signal
/// the r231 fixed-size CFL tests use, scaled to a dynamic-extent
/// frame. Luma is a column-direction gradient over the full width so
/// the chroma planes have a useful range to track.
fn luma_correlated_yuv_input_dyn(width: u32, height: u32) -> oxideav_av1::encoder::Yuv420Frame {
    use oxideav_av1::encoder::Yuv420Frame;
    let cw = (width / 2) as usize;
    let ch = (height / 2) as usize;
    let w = width as usize;
    let h = height as usize;
    let mut input = Yuv420Frame::filled(width, height, 0);
    for i in 0..h {
        for j in 0..w {
            // Wrap-around gradient so larger frames still produce
            // varied luma without saturating to 255.
            let v = ((j as u32) * 8) % 256;
            input.y[i * w + j] = v as u8;
        }
    }
    for ci in 0..ch {
        for cj in 0..cw {
            let li = ci * 2;
            let lj = cj * 2;
            let avg = (input.y[li * w + lj] as u32
                + input.y[li * w + (lj + 1)] as u32
                + input.y[(li + 1) * w + lj] as u32
                + input.y[(li + 1) * w + (lj + 1)] as u32)
                / 4;
            let centred = avg as i32 - 128;
            input.u[ci * cw + cj] = (128i32 + centred / 4).clamp(0, 255) as u8;
            input.v[ci * cw + cj] = (128i32 - centred / 4).clamp(0, 255) as u8;
        }
    }
    input
}

#[test]
fn dyn_encode_decode_cfl_yuv_roundtrip_bit_exact_32x32() {
    // r232: full encode → decode → pixel-equality on the dyn driver
    // with a chroma signal that's a clean linear function of luma.
    // The dyn picker now evaluates UV_CFL_PRED for at least one
    // chroma cell; the dyn decoder mirrors its CFL prediction and the
    // lossless WHT chain stays bit-exact.
    use oxideav_av1::encoder::encode_intra_frame_yuv_dyn;
    let input = luma_correlated_yuv_input_dyn(32, 32);
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 32);
            assert_eq!(*height, 32);
            assert_eq!(y, &input.y, "Y plane bit-exact on 32×32 CFL input");
            assert_eq!(u, &input.u, "U plane bit-exact on 32×32 CFL input");
            assert_eq!(v, &input.v, "V plane bit-exact on 32×32 CFL input");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_cfl_yuv_roundtrip_bit_exact_64x64() {
    // r232: same CFL roundtrip as the 32×32 variant but at the
    // arc-r230 maximum single-super-block extent. Confirms the
    // §7.11.5.3 subsampled-luma window honours the right/bottom-edge
    // clamps when the luma plane is non-multiple-of-2 at the chroma
    // boundary (here: width = height = 64, all clamps strict-inside).
    use oxideav_av1::encoder::encode_intra_frame_yuv_dyn;
    let input = luma_correlated_yuv_input_dyn(64, 64);
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 64);
            assert_eq!(*height, 64);
            assert_eq!(y, &input.y, "Y plane bit-exact on 64×64 CFL input");
            assert_eq!(u, &input.u, "U plane bit-exact on 64×64 CFL input");
            assert_eq!(v, &input.v, "V plane bit-exact on 64×64 CFL input");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_pseudorandom_32x32_still_roundtrips_with_cfl_arm() {
    // r232 regression-guard: with the CFL arm now wired into the dyn
    // picker, pseudo-random YUV (where chroma is uncorrelated with
    // luma) must still roundtrip bit-exact — i.e. the picker only
    // commits CFL when it actually beats every §6.10.x mode, never as
    // a regression that erodes the lossless contract.
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    let mut input = Yuv420Frame::filled(32, 32, 0);
    let mut state: u64 = 0xC0FFEE_C0FFEE_u64;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = step();
    }
    for p in input.u.iter_mut() {
        *p = step();
    }
    for p in input.v.iter_mut() {
        *p = step();
    }
    let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 32);
            assert_eq!(*height, 32);
            assert_eq!(y, &input.y, "Y mismatch on pseudo-random 32×32");
            assert_eq!(u, &input.u, "U mismatch on pseudo-random 32×32");
            assert_eq!(v, &input.v, "V mismatch on pseudo-random 32×32");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

// -------------------------------------------------------------------------
// r196/r233 — `base_q_idx > 0` (lossy quant) on the dyn driver.
//
// The lossy contract is: `decode_av1(encoded.ivf_bytes)` matches
// `encoded.reconstructed_*` byte-for-byte at any caller-supplied
// `base_q_idx`. The recovered planes do NOT in general equal `input.*`
// — quantization introduces rounding error bounded by the §7.12.2
// `dc_q_lookup` / `ac_q_lookup` step at the chosen qindex. The test
// guard is therefore "decoder output == encoder reconstruction", not
// "decoder output == input".
//
// The encoder side's running reconstructed buffers are the bit-exact
// image the decoder would reconstruct from the same FH + tile bytes
// because the encoder runs the same `dequantize_step1 →
// inverse_transform_2d` pipeline on its own quantized coefficients
// before stamping `recon_*`.
// -------------------------------------------------------------------------

#[test]
fn dyn_encode_decode_lossy_q1_flat_grey_32x32_decoder_matches_encoder_recon() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_with_q, Yuv420Frame};
    let input = Yuv420Frame::filled(32, 32, 128);
    // q_idx = 1 — the smallest non-zero qindex. Still well within the
    // bit-exact tolerance for flat-grey input (every DC residual
    // collapses to a quantized zero and chroma DC_PRED brings the
    // residual to zero on a flat plane).
    let encoded =
        encode_intra_frame_yuv_dyn_with_q(&input, 1).expect("encode succeeds at base_q_idx = 1");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn { y, u, v, .. } => {
            assert_eq!(
                y, &encoded.reconstructed_y,
                "Y decoder output must match encoder reconstruction at base_q_idx = 1"
            );
            assert_eq!(
                u, &encoded.reconstructed_u,
                "U decoder output must match encoder reconstruction at base_q_idx = 1"
            );
            assert_eq!(
                v, &encoded.reconstructed_v,
                "V decoder output must match encoder reconstruction at base_q_idx = 1"
            );
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_lossy_q16_horizontal_gradient_32x32_decoder_matches_encoder_recon() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_with_q, Yuv420Frame};
    // q_idx = 16: `dc_q_lookup[0][16] = 20`, `ac_q_lookup[0][16] = 23`.
    // A horizontal gradient exercises both the V_PRED-favouring picker
    // and the §7.13.3 forward DCT_DCT path with non-trivial AC content.
    let mut input = Yuv420Frame::filled(32, 32, 0);
    let width = input.width as usize;
    let cwidth = input.chroma_width() as usize;
    for i in 0..(input.height as usize) {
        for j in 0..width {
            input.y[i * width + j] = (j * 8) as u8;
        }
    }
    for i in 0..(input.chroma_height() as usize) {
        for j in 0..cwidth {
            input.u[i * cwidth + j] = (j * 16) as u8;
            input.v[i * cwidth + j] = (255u8).wrapping_sub((j * 16) as u8);
        }
    }
    let encoded =
        encode_intra_frame_yuv_dyn_with_q(&input, 16).expect("encode succeeds at base_q_idx = 16");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn { y, u, v, .. } => {
            assert_eq!(
                y, &encoded.reconstructed_y,
                "Y decoder output must match encoder reconstruction at base_q_idx = 16"
            );
            assert_eq!(
                u, &encoded.reconstructed_u,
                "U decoder output must match encoder reconstruction at base_q_idx = 16"
            );
            assert_eq!(
                v, &encoded.reconstructed_v,
                "V decoder output must match encoder reconstruction at base_q_idx = 16"
            );
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_lossy_q64_pseudorandom_32x32_decoder_matches_encoder_recon() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_with_q, Yuv420Frame};
    // q_idx = 64: a typical "mid-quality" qindex.
    // `dc_q_lookup[0][64] = 76`, `ac_q_lookup[0][64] = 83`. The picker
    // sees a pseudo-random input across the full byte range; the lossy
    // arm must still round-trip encoder-recon → decoder-out bit-exact.
    let mut input = Yuv420Frame::filled(32, 32, 0);
    let mut state: u64 = 0xDEAD_BEEF_FEED_FACE;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = step();
    }
    for p in input.u.iter_mut() {
        *p = step();
    }
    for p in input.v.iter_mut() {
        *p = step();
    }
    let encoded =
        encode_intra_frame_yuv_dyn_with_q(&input, 64).expect("encode succeeds at base_q_idx = 64");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn { y, u, v, .. } => {
            assert_eq!(
                y, &encoded.reconstructed_y,
                "Y decoder output must match encoder reconstruction at base_q_idx = 64"
            );
            assert_eq!(
                u, &encoded.reconstructed_u,
                "U decoder output must match encoder reconstruction at base_q_idx = 64"
            );
            assert_eq!(
                v, &encoded.reconstructed_v,
                "V decoder output must match encoder reconstruction at base_q_idx = 64"
            );
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_lossy_q1_64x64_decoder_matches_encoder_recon() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_with_q, Yuv420Frame};
    // Largest dyn extent + smallest non-zero qindex. Exercises the
    // multi-cell §7.13.3 forward DCT walk on the 64×64 super-block.
    let input = Yuv420Frame::filled(64, 64, 200);
    let encoded = encode_intra_frame_yuv_dyn_with_q(&input, 1).expect("encode succeeds");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn { y, u, v, .. } => {
            assert_eq!(y, &encoded.reconstructed_y);
            assert_eq!(u, &encoded.reconstructed_u);
            assert_eq!(v, &encoded.reconstructed_v);
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn dyn_encode_decode_lossy_q_zero_via_with_q_matches_legacy_lossless_path() {
    // Regression guard: `encode_intra_frame_yuv_dyn_with_q(_, 0)` must
    // produce the exact same IVF bytes + reconstruction as the legacy
    // `encode_intra_frame_yuv_dyn`. Catches any future divergence
    // between the two entry points on the lossless arm.
    use oxideav_av1::encoder::{
        encode_intra_frame_yuv_dyn, encode_intra_frame_yuv_dyn_with_q, Yuv420Frame,
    };
    let mut input = Yuv420Frame::filled(32, 32, 0);
    let mut state: u64 = 0x4242_4242_4242_4242;
    let mut step = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = step();
    }
    for p in input.u.iter_mut() {
        *p = step();
    }
    for p in input.v.iter_mut() {
        *p = step();
    }
    let lossless_legacy = encode_intra_frame_yuv_dyn(&input).expect("legacy lossless encodes");
    let lossless_via_q0 = encode_intra_frame_yuv_dyn_with_q(&input, 0).expect("with_q(0) encodes");
    assert_eq!(
        lossless_legacy.ivf_bytes, lossless_via_q0.ivf_bytes,
        "with_q(0) must produce identical IVF bytes to the legacy lossless path"
    );
    assert_eq!(
        lossless_legacy.reconstructed_y, lossless_via_q0.reconstructed_y,
        "with_q(0) Y recon must match legacy lossless"
    );
    assert_eq!(
        lossless_legacy.reconstructed_u, lossless_via_q0.reconstructed_u,
        "with_q(0) U recon must match legacy lossless"
    );
    assert_eq!(
        lossless_legacy.reconstructed_v, lossless_via_q0.reconstructed_v,
        "with_q(0) V recon must match legacy lossless"
    );
    // Bit-exact equality against input is the lossless contract still
    // in effect at q=0.
    assert_eq!(lossless_via_q0.reconstructed_y, input.y);
    assert_eq!(lossless_via_q0.reconstructed_u, input.u);
    assert_eq!(lossless_via_q0.reconstructed_v, input.v);
}

#[test]
fn dyn_encode_decode_lossy_q255_decoder_still_matches_encoder_recon() {
    // Maximum lossy qindex (`base_q_idx = 255`) — `dc_q[0][255] = 1828`
    // / `ac_q[0][255] = 1828`. Every coefficient quantizes to zero, so
    // the encoder's reconstruction equals its prediction sequence. The
    // decoder must reproduce the same picture byte-for-byte; this is
    // the worst-case stress for the §7.12.3 dequantize + §7.13.3
    // inverse DCT pipeline at the high end of the qindex table.
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_with_q, Yuv420Frame};
    let mut input = Yuv420Frame::filled(32, 32, 0);
    // Use a horizontal gradient so the predictor's choices are
    // non-trivial across the frame.
    let width = input.width as usize;
    for i in 0..(input.height as usize) {
        for j in 0..width {
            input.y[i * width + j] = (j * 8) as u8;
        }
    }
    let encoded = encode_intra_frame_yuv_dyn_with_q(&input, 255)
        .expect("encode succeeds at base_q_idx = 255");
    let decoded = decode_av1(&encoded.ivf_bytes).expect("decode succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn { y, u, v, .. } => {
            assert_eq!(
                y, &encoded.reconstructed_y,
                "Y decoder output must match encoder reconstruction at base_q_idx = 255"
            );
            assert_eq!(
                u, &encoded.reconstructed_u,
                "U decoder output must match encoder reconstruction at base_q_idx = 255"
            );
            assert_eq!(
                v, &encoded.reconstructed_v,
                "V decoder output must match encoder reconstruction at base_q_idx = 255"
            );
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

// ----------------------------------------------------------------------
// r197 / r234 — Rectangular frame-extent coverage on the dyn driver.
//
// The §5.11.4 partition tree's per-quadrant `r >= mi_rows || c >=
// mi_cols` early return + `EncodeNode::dummy_oob` sentinel already let
// the dyn encoder + decoder handle any extent in `MIN_DIM..=MAX_DIM`
// (multiples of 8). These integration tests promote that property from
// "works incidentally" to a tested invariant, sweeping the cardinal
// rectangular shapes — short+wide, tall+narrow, partial-coverage at
// every super-block class (BLOCK_16X16 / 32X32 / 64X64 root) — across
// the lossless WHT arm and the §7.13.3 lossy DCT arm at every plumbed
// qindex.
//
// The picker's `tx_size = TX_4X4` everywhere stays unchanged this
// round; only the **frame extent** is allowed to be rectangular. The
// rectangular **TX_SIZE** family (`TX_4X8` / `TX_8X4` / `TX_8X16` /
// `TX_16X8`) remains a follow-up arc.

fn random_yuv(w: u32, h: u32, seed: u64) -> oxideav_av1::encoder::Yuv420Frame {
    use oxideav_av1::encoder::Yuv420Frame;
    let mut input = Yuv420Frame::filled(w, h, 0);
    let mut state: u64 = seed
        .wrapping_mul(2654435761)
        .wrapping_add((w as u64) << 16)
        .wrapping_add(h as u64);
    let mut next = || -> u8 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = next();
    }
    for p in input.u.iter_mut() {
        *p = next();
    }
    for p in input.v.iter_mut() {
        *p = next();
    }
    input
}

/// Helper — assert encoder → decoder bit-exact round-trip on one
/// rectangular `(w, h)` extent at one `q`. The encoder's
/// reconstruction is what the decoder must reproduce (lossless input
/// equality is asserted separately at `q == 0` only).
fn assert_rect_roundtrip_bit_exact(w: u32, h: u32, q: u8) {
    use oxideav_av1::encoder::encode_intra_frame_yuv_dyn_with_q;
    let input = random_yuv(w, h, 0xA5A5_5A5A);
    let encoded = encode_intra_frame_yuv_dyn_with_q(&input, q)
        .unwrap_or_else(|e| panic!("encode {w}x{h} q={q} failed: {e:?}"));
    let decoded = decode_av1(&encoded.ivf_bytes)
        .unwrap_or_else(|e| panic!("decode {w}x{h} q={q} failed: {e:?}"));
    assert_eq!(
        decoded.len(),
        1,
        "one frame in, one frame out ({w}x{h} q={q})"
    );
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, w, "decoded width mismatch at {w}x{h} q={q}");
            assert_eq!(*height, h, "decoded height mismatch at {w}x{h} q={q}");
            assert_eq!(
                y, &encoded.reconstructed_y,
                "Y decoder ≠ encoder.recon at {w}x{h} q={q}"
            );
            assert_eq!(
                u, &encoded.reconstructed_u,
                "U decoder ≠ encoder.recon at {w}x{h} q={q}"
            );
            assert_eq!(
                v, &encoded.reconstructed_v,
                "V decoder ≠ encoder.recon at {w}x{h} q={q}"
            );
            if q == 0 {
                assert_eq!(y, &input.y, "Y lossless WHT recovery mismatch at {w}x{h}");
                assert_eq!(u, &input.u, "U lossless WHT recovery mismatch at {w}x{h}");
                assert_eq!(v, &input.v, "V lossless WHT recovery mismatch at {w}x{h}");
            }
        }
        other => panic!("expected Yuv420Dyn at {w}x{h} q={q}, got {other:?}"),
    }
}

// --- Lossless WHT arm (`q == 0`) at every cardinal rectangular shape. ---

#[test]
fn dyn_encode_decode_rect_lossless_16x32_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(16, 32, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_32x16_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(32, 16, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_8x16_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(8, 16, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_16x8_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(16, 8, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_24x32_pseudorandom_bit_exact() {
    // Partial-coverage extent — mi 6×8, root BLOCK_32X32, the right
    // BLOCK_16X16 quadrants partially overflow the frame width.
    assert_rect_roundtrip_bit_exact(24, 32, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_32x24_pseudorandom_bit_exact() {
    // Transposed partial-coverage shape — mi 8×6, bottom quadrants
    // partially out-of-frame on height.
    assert_rect_roundtrip_bit_exact(32, 24, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_40x16_pseudorandom_bit_exact() {
    // Forces BLOCK_64X64 root super-block (max(mi) = 10).
    assert_rect_roundtrip_bit_exact(40, 16, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_16x40_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(16, 40, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_48x32_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(48, 32, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_32x48_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(32, 48, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_32x64_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(32, 64, 0);
}

#[test]
fn dyn_encode_decode_rect_lossless_64x32_pseudorandom_bit_exact() {
    assert_rect_roundtrip_bit_exact(64, 32, 0);
}

// --- §7.13.3 lossy DCT arm — every rectangular extent across qindex
//     spans the dyn driver supports. Asserts encoder.recon ↔ decoder
//     bit-equality (the lossy self-decode contract). ---

#[test]
fn dyn_encode_decode_rect_lossy_16x32_q1_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(16, 32, 1);
}

#[test]
fn dyn_encode_decode_rect_lossy_32x16_q16_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(32, 16, 16);
}

#[test]
fn dyn_encode_decode_rect_lossy_8x16_q64_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(8, 16, 64);
}

#[test]
fn dyn_encode_decode_rect_lossy_16x8_q255_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(16, 8, 255);
}

#[test]
fn dyn_encode_decode_rect_lossy_24x32_q32_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(24, 32, 32);
}

#[test]
fn dyn_encode_decode_rect_lossy_40x16_q128_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(40, 16, 128);
}

#[test]
fn dyn_encode_decode_rect_lossy_16x40_q200_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(16, 40, 200);
}

#[test]
fn dyn_encode_decode_rect_lossy_32x64_q64_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(32, 64, 64);
}

#[test]
fn dyn_encode_decode_rect_lossy_64x32_q255_decoder_matches_encoder_recon() {
    assert_rect_roundtrip_bit_exact(64, 32, 255);
}

// --- Coverage / API surface: a rectangular extent the encoder accepts
//     must surface in the produced IVF v0 header (`bytes 12..14 = width
//     LE`, `bytes 14..16 = height LE`). This rules out a "decode
//     happens to work because the encoder fell back to a square" sort
//     of latent bug. ---

#[test]
fn dyn_encode_rect_writes_correct_dimensions_in_ivf_v0_header() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    for (w, h) in [
        (16u32, 32u32),
        (32, 16),
        (8, 16),
        (16, 8),
        (40, 24),
        (24, 40),
    ] {
        let input = Yuv420Frame::filled(w, h, 64);
        let res = encode_intra_frame_yuv_dyn(&input)
            .unwrap_or_else(|e| panic!("encode {w}x{h} failed: {e:?}"));
        let ivf_w = u16::from_le_bytes([res.ivf_bytes[12], res.ivf_bytes[13]]) as u32;
        let ivf_h = u16::from_le_bytes([res.ivf_bytes[14], res.ivf_bytes[15]]) as u32;
        assert_eq!(ivf_w, w, "IVF v0 header width mismatch for {w}x{h}");
        assert_eq!(ivf_h, h, "IVF v0 header height mismatch for {w}x{h}");
        let fs = res.fh.frame_size.as_ref().expect("intra has frame_size");
        assert_eq!(fs.frame_width, w);
        assert_eq!(fs.frame_height, h);
        // The FH `mi_cols`/`mi_rows` derivation: 2 * ceil(dim/8).
        assert_eq!(fs.mi_cols, 2 * ((w + 7) >> 3));
        assert_eq!(fs.mi_rows, 2 * ((h + 7) >> 3));
    }
}

// =====================================================================
// r207 — multi-super-block monochrome (Y-only) dyn driver integration
// tests.
//
// Exercises the §5.11.1 SB-grid walk introduced in r207 by encoding
// then decoding frames whose extent exceeds the prior single-SB cap of
// 64 × 64 (up to the new 128 × 128 ceiling). The end-to-end contract
// is the same as for the ≤ 64×64 path: at `base_q_idx = 0` the
// recovered Y plane equals the input plane-for-plane; at
// `base_q_idx > 0` the decoded plane equals the encoder-internal
// reconstruction byte-for-byte.
// =====================================================================

#[test]
fn mono_y_multi_sb_dyn_encode_decode_lossless_roundtrip_96x64() {
    use oxideav_av1::encoder::{encode_intra_frame_y_dyn_multi_sb, MonoYFrameMultiSb};
    let mut input = MonoYFrameMultiSb::filled(96, 64, 0);
    let mut s: u64 = 0xCAFE_F00D_DEAD_BEEF;
    for p in input.y.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *p = (s >> 56) as u8;
    }
    let enc = encode_intra_frame_y_dyn_multi_sb(&input).expect("encode");
    assert_eq!(enc.reconstructed_y, input.y);
    let dec = decode_av1(&enc.ivf_bytes).expect("decode");
    assert_eq!(dec.len(), 1);
    match &dec[0] {
        Frame::YDyn { width, height, y } => {
            assert_eq!(*width, 96);
            assert_eq!(*height, 64);
            assert_eq!(y, &input.y, "96x64 lossless bit-exact recovery");
        }
        other => panic!("expected YDyn at 96x64, got {other:?}"),
    }
}

#[test]
fn mono_y_multi_sb_dyn_encode_decode_lossless_roundtrip_128x128() {
    use oxideav_av1::encoder::{encode_intra_frame_y_dyn_multi_sb, MonoYFrameMultiSb};
    let mut input = MonoYFrameMultiSb::filled(128, 128, 0);
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    for p in input.y.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *p = (s >> 56) as u8;
    }
    let enc = encode_intra_frame_y_dyn_multi_sb(&input).expect("encode");
    assert_eq!(enc.reconstructed_y, input.y);
    let dec = decode_av1(&enc.ivf_bytes).expect("decode");
    match &dec[0] {
        Frame::YDyn { width, height, y } => {
            assert_eq!(*width, 128);
            assert_eq!(*height, 128);
            assert_eq!(y, &input.y, "128x128 lossless bit-exact recovery");
        }
        other => panic!("expected YDyn at 128x128, got {other:?}"),
    }
}

#[test]
fn mono_y_multi_sb_dyn_encode_decode_lossless_roundtrip_rectangular_edges() {
    use oxideav_av1::encoder::{encode_intra_frame_y_dyn_multi_sb, MonoYFrameMultiSb};
    // Stress per-SB OOB short-circuit by selecting extents that
    // create partial-coverage edge SBs in the row, column, and
    // bottom-right corner. Each row exercises a different
    // (width % 64, height % 64) combination.
    let extents: &[(u32, u32)] = &[
        (72, 64),   // 1 partial col SB on the right
        (64, 72),   // 1 partial row SB on the bottom
        (104, 72),  // partial col + partial row + partial corner
        (128, 72),  // full col SB row, partial row SB
        (88, 128),  // partial col, two full row SBs
        (120, 120), // 2×2 grid all partial
    ];
    let mut s: u64 = 0xDEAD_BEEF_C0DE_BABE;
    for &(w, h) in extents {
        let mut input = MonoYFrameMultiSb::filled(w, h, 0);
        for p in input.y.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = (s >> 56) as u8;
        }
        let enc = encode_intra_frame_y_dyn_multi_sb(&input)
            .unwrap_or_else(|_| panic!("encode failed at {w}x{h}"));
        assert_eq!(
            enc.reconstructed_y, input.y,
            "encoder recon at {w}x{h} must equal input on the lossless WHT arm"
        );
        let dec = decode_av1(&enc.ivf_bytes).unwrap_or_else(|_| panic!("decode failed at {w}x{h}"));
        match &dec[0] {
            Frame::YDyn { width, height, y } => {
                assert_eq!(*width, w);
                assert_eq!(*height, h);
                assert_eq!(y, &input.y, "lossless bit-exact at {w}x{h}");
            }
            other => panic!("expected YDyn at {w}x{h}, got {other:?}"),
        }
    }
}

#[test]
fn mono_y_multi_sb_dyn_encode_decode_lossy_q32_self_consistent_96x64() {
    use oxideav_av1::encoder::{encode_intra_frame_y_dyn_multi_sb_with_q, MonoYFrameMultiSb};
    let mut input = MonoYFrameMultiSb::filled(96, 64, 0);
    let mut s: u64 = 0xFEED_FACE_C0DE_BABE;
    for p in input.y.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *p = (s >> 56) as u8;
    }
    let enc = encode_intra_frame_y_dyn_multi_sb_with_q(&input, 32).expect("encode q=32");
    let dec = decode_av1(&enc.ivf_bytes).expect("decode");
    match &dec[0] {
        Frame::YDyn { width, height, y } => {
            assert_eq!(*width, 96);
            assert_eq!(*height, 64);
            assert_eq!(
                y, &enc.reconstructed_y,
                "lossy q=32 decoder out = encoder recon byte-for-byte"
            );
            // Verify quantization changed the output: at least one
            // sample should differ from the input.
            let differs = y.iter().zip(input.y.iter()).any(|(a, b)| a != b);
            assert!(differs, "q=32 must perturb at least one sample");
        }
        other => panic!("expected YDyn, got {other:?}"),
    }
}

#[test]
fn mono_y_multi_sb_dyn_encode_writes_correct_dimensions_in_ivf_header() {
    use oxideav_av1::encoder::{encode_intra_frame_y_dyn_multi_sb, MonoYFrameMultiSb};
    // Multi-SB extents must surface verbatim in the IVF v0 header.
    for (w, h) in [(96u32, 64u32), (128, 64), (64, 128), (128, 128), (104, 72)] {
        let input = MonoYFrameMultiSb::filled(w, h, 64);
        let res = encode_intra_frame_y_dyn_multi_sb(&input)
            .unwrap_or_else(|e| panic!("encode {w}x{h} failed: {e:?}"));
        let ivf_w = u16::from_le_bytes([res.ivf_bytes[12], res.ivf_bytes[13]]) as u32;
        let ivf_h = u16::from_le_bytes([res.ivf_bytes[14], res.ivf_bytes[15]]) as u32;
        assert_eq!(ivf_w, w, "IVF v0 width mismatch at {w}x{h}");
        assert_eq!(ivf_h, h, "IVF v0 height mismatch at {w}x{h}");
        let fs = res.fh.frame_size.as_ref().expect("intra has frame_size");
        assert_eq!(fs.frame_width, w);
        assert_eq!(fs.frame_height, h);
        assert_eq!(fs.mi_cols, 2 * ((w + 7) >> 3));
        assert_eq!(fs.mi_rows, 2 * ((h + 7) >> 3));
    }
}

// ----------------------------------------------------------------------
// r214 — multi-super-block 4:2:0 YUV dyn driver integration tests
// ----------------------------------------------------------------------

#[test]
fn yuv_multi_sb_dyn_encode_decode_lossless_roundtrip_96x64() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_multi_sb, Yuv420FrameMultiSb};
    let mut input = Yuv420FrameMultiSb::filled(96, 64, 0);
    let mut s: u64 = 0xABCD_DEAD_C0DE_FACE;
    let mut next = || -> u8 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = next();
    }
    for p in input.u.iter_mut() {
        *p = next();
    }
    for p in input.v.iter_mut() {
        *p = next();
    }
    let enc = encode_intra_frame_yuv_dyn_multi_sb(&input).expect("encode");
    let dec = decode_av1(&enc.ivf_bytes).expect("decode");
    match &dec[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 96);
            assert_eq!(*height, 64);
            assert_eq!(y, &input.y, "Y bit-exact recovery at 96x64");
            assert_eq!(u, &input.u, "U bit-exact recovery at 96x64");
            assert_eq!(v, &input.v, "V bit-exact recovery at 96x64");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn yuv_multi_sb_dyn_encode_decode_lossless_roundtrip_128x128() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_multi_sb, Yuv420FrameMultiSb};
    let mut input = Yuv420FrameMultiSb::filled(128, 128, 0);
    let mut s: u64 = 0x4242_BABE_DEAD_F00D;
    let mut next = || -> u8 {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = next();
    }
    for p in input.u.iter_mut() {
        *p = next();
    }
    for p in input.v.iter_mut() {
        *p = next();
    }
    let enc = encode_intra_frame_yuv_dyn_multi_sb(&input).expect("encode");
    let dec = decode_av1(&enc.ivf_bytes).expect("decode");
    match &dec[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 128);
            assert_eq!(*height, 128);
            assert_eq!(y, &input.y, "Y bit-exact recovery at 128x128");
            assert_eq!(u, &input.u, "U bit-exact recovery at 128x128");
            assert_eq!(v, &input.v, "V bit-exact recovery at 128x128");
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn yuv_multi_sb_dyn_encode_decode_lossless_roundtrip_rectangular_edges() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_multi_sb, Yuv420FrameMultiSb};
    // Six rectangular multi-SB extents that exercise:
    //   * one extra SB along x with a partial right SB (72x64, 104x72),
    //   * one extra SB along y with a partial bottom SB (64x72, 88x128),
    //   * a 2x2 grid with partial cells (120x120),
    //   * a full 2-SB column (128x72).
    let extents: &[(u32, u32)] = &[
        (72, 64),
        (64, 72),
        (104, 72),
        (128, 72),
        (88, 128),
        (120, 120),
    ];
    let mut s: u64 = 0x1010_F00D_3434_CAFE;
    let next = |s: &mut u64| -> u8 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 56) as u8
    };
    for &(w, h) in extents {
        let mut input = Yuv420FrameMultiSb::filled(w, h, 0);
        for p in input.y.iter_mut() {
            *p = next(&mut s);
        }
        for p in input.u.iter_mut() {
            *p = next(&mut s);
        }
        for p in input.v.iter_mut() {
            *p = next(&mut s);
        }
        let enc = encode_intra_frame_yuv_dyn_multi_sb(&input)
            .unwrap_or_else(|_| panic!("encode failed at {w}x{h}"));
        let dec = decode_av1(&enc.ivf_bytes).unwrap_or_else(|_| panic!("decode failed at {w}x{h}"));
        match &dec[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, w, "decoded width at {w}x{h}");
                assert_eq!(*height, h, "decoded height at {w}x{h}");
                assert_eq!(y, &input.y, "Y bit-exact recovery at {w}x{h}");
                assert_eq!(u, &input.u, "U bit-exact recovery at {w}x{h}");
                assert_eq!(v, &input.v, "V bit-exact recovery at {w}x{h}");
            }
            other => panic!("expected Yuv420Dyn at {w}x{h}, got {other:?}"),
        }
    }
}

#[test]
fn yuv_multi_sb_dyn_encode_decode_lossy_q32_self_consistent_96x64() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_multi_sb_with_q, Yuv420FrameMultiSb};
    let mut input = Yuv420FrameMultiSb::filled(96, 64, 0);
    let mut s: u64 = 0x7777_AAAA_3333_FFFF;
    let next = |s: &mut u64| -> u8 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 56) as u8
    };
    for p in input.y.iter_mut() {
        *p = next(&mut s);
    }
    for p in input.u.iter_mut() {
        *p = next(&mut s);
    }
    for p in input.v.iter_mut() {
        *p = next(&mut s);
    }
    let enc = encode_intra_frame_yuv_dyn_multi_sb_with_q(&input, 32).expect("encode q=32");
    let dec = decode_av1(&enc.ivf_bytes).expect("decode");
    match &dec[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, 96);
            assert_eq!(*height, 64);
            assert_eq!(
                y, &enc.reconstructed_y,
                "Y lossy q=32 decoder out = encoder recon byte-for-byte"
            );
            assert_eq!(
                u, &enc.reconstructed_u,
                "U lossy q=32 decoder out = encoder recon byte-for-byte"
            );
            assert_eq!(
                v, &enc.reconstructed_v,
                "V lossy q=32 decoder out = encoder recon byte-for-byte"
            );
        }
        other => panic!("expected Yuv420Dyn, got {other:?}"),
    }
}

#[test]
fn yuv_multi_sb_dyn_encode_writes_correct_dimensions_in_ivf_header() {
    use oxideav_av1::encoder::{encode_intra_frame_yuv_dyn_multi_sb, Yuv420FrameMultiSb};
    // Multi-SB YUV extents must surface verbatim in the IVF v0 header.
    for (w, h) in [(96u32, 64u32), (128, 64), (64, 128), (128, 128), (104, 72)] {
        let input = Yuv420FrameMultiSb::filled(w, h, 128);
        let res = encode_intra_frame_yuv_dyn_multi_sb(&input)
            .unwrap_or_else(|e| panic!("encode {w}x{h} failed: {e:?}"));
        let ivf_w = u16::from_le_bytes([res.ivf_bytes[12], res.ivf_bytes[13]]) as u32;
        let ivf_h = u16::from_le_bytes([res.ivf_bytes[14], res.ivf_bytes[15]]) as u32;
        assert_eq!(ivf_w, w, "IVF v0 width mismatch at {w}x{h}");
        assert_eq!(ivf_h, h, "IVF v0 height mismatch at {w}x{h}");
        let fs = res.fh.frame_size.as_ref().expect("intra has frame_size");
        assert_eq!(fs.frame_width, w);
        assert_eq!(fs.frame_height, h);
        assert_eq!(fs.mi_cols, 2 * ((w + 7) >> 3));
        assert_eq!(fs.mi_rows, 2 * ((h + 7) >> 3));
    }
}

// --------------------------------------------------------------------------
// Round 249 — public-entry-point graduation of the dyn-driver 4:2:0 YUV
// codepath.
//
// Before r249 the public `encode_av1` was a hard-stubbed
// `Err(NotImplemented)` while `encode_intra_frame_yuv_dyn` had been a
// crate-public dyn encoder since r230 (with chroma 13-mode + UV_CFL_PRED
// arms graduated in r229 / r231 / r232 and frame-size generalisation
// 8..=64 per axis in r230). These tests exercise the new public
// `oxideav_av1::encode_av1(pixels, width, height) -> Result<Vec<u8>>`
// signature against the matching public `decode_av1` entry, end-to-end,
// confirming:
//
//   1. The byte-input convention is planar `Y || U || V` (luma plane
//      first, then half-resolution U then half-resolution V).
//   2. The `base_q_idx = 0` lossless-WHT arm round-trips bit-exactly
//      across the public surface.
//   3. The dyn dispatch in the public `decode_av1` accepts the dyn
//      encoder's output without manual driver hookup.
//   4. Out-of-scope inputs surface as `Error::PartitionWalkOutOfRange`
//      rather than the historical `Error::NotImplemented`.
// --------------------------------------------------------------------------

#[test]
fn public_encode_av1_decode_av1_roundtrip_16x16_lossless() {
    use oxideav_av1::{decode_av1, encode_av1};
    // Build a planar 4:2:0 YUV buffer at 16x16 with deterministic
    // pseudorandom samples on every plane.
    let (w, h) = (16u32, 16u32);
    let y_size = (w * h) as usize;
    let uv_size = ((w / 2) * (h / 2)) as usize;
    let mut pixels = vec![0u8; y_size + 2 * uv_size];
    let mut s: u64 = 0x2026_0607_DEAD_F00D;
    for byte in pixels.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (s >> 56) as u8;
    }
    let ivf = encode_av1(&pixels, w, h).expect("public encode_av1 succeeds at 16x16");
    let decoded = decode_av1(&ivf).expect("public decode_av1 succeeds on dyn-emitted IVF");
    assert_eq!(decoded.len(), 1, "one IVF frame in, one Frame out");
    let expected_y = &pixels[..y_size];
    let expected_u = &pixels[y_size..y_size + uv_size];
    let expected_v = &pixels[y_size + uv_size..];
    // At exactly 16x16 the decoder's `decode_frame` dispatcher routes
    // to the fixed-size [`Frame::Yuv420_16x16`] arm (the §5.11.4 single-
    // root BLOCK_16X16 walk), so the decoded variant is the fixed-shape
    // one. The pixel bytes round-trip identically to the dyn variant at
    // any other extent — the only difference is the carrier type.
    match &decoded[0] {
        Frame::Yuv420_16x16 { y, u, v } => {
            for (row, exp_chunk) in expected_y.chunks_exact(w as usize).enumerate() {
                assert_eq!(&y[row][..], exp_chunk, "Y row {row} bit-exact");
            }
            for (row, exp_chunk) in expected_u.chunks_exact((w / 2) as usize).enumerate() {
                assert_eq!(&u[row][..], exp_chunk, "U row {row} bit-exact");
            }
            for (row, exp_chunk) in expected_v.chunks_exact((w / 2) as usize).enumerate() {
                assert_eq!(&v[row][..], exp_chunk, "V row {row} bit-exact");
            }
        }
        other => panic!("expected Yuv420_16x16 at 16x16, got {other:?}"),
    }
}

#[test]
fn public_encode_av1_decode_av1_roundtrip_32x24_rectangular_lossless() {
    use oxideav_av1::{decode_av1, encode_av1};
    // A non-square dyn extent — the r230 dyn driver supports any aligned
    // (w, h) in {8, 16, 24, 32, 40, 48, 56, 64} per axis; (32, 24)
    // exercises the rectangular path through the public surface.
    let (w, h) = (32u32, 24u32);
    let y_size = (w * h) as usize;
    let uv_size = ((w / 2) * (h / 2)) as usize;
    let mut pixels = vec![0u8; y_size + 2 * uv_size];
    let mut s: u64 = 0x1234_5678_9ABC_DEF0;
    for byte in pixels.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (s >> 56) as u8;
    }
    let ivf = encode_av1(&pixels, w, h).expect("public encode_av1 succeeds at 32x24");
    let decoded = decode_av1(&ivf).expect("public decode_av1 succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, w);
            assert_eq!(*height, h);
            assert_eq!(y.as_slice(), &pixels[..y_size]);
            assert_eq!(u.as_slice(), &pixels[y_size..y_size + uv_size]);
            assert_eq!(v.as_slice(), &pixels[y_size + uv_size..]);
        }
        other => panic!("expected Yuv420Dyn at 32x24, got {other:?}"),
    }
}

#[test]
fn public_encode_av1_rejects_wrong_pixel_buffer_length() {
    use oxideav_av1::encode_av1;
    // The byte-input convention is `Y || U || V`; lengths that don't
    // match `w * h + 2 * (w/2) * (h/2)` must surface
    // `PartitionWalkOutOfRange` rather than silently truncating /
    // padding.
    let res = encode_av1(&[0u8; 100], 16, 16);
    assert!(res.is_err(), "wrong pixel-buffer length must fail");
}

#[test]
fn public_encode_av1_rejects_non_multiple_of_8_dim() {
    use oxideav_av1::encode_av1;
    // The dyn driver requires every axis to be a multiple of MIN_DIM (8)
    // — the §5.11.5 4:2:0 chroma sample-cell constraint. The public
    // surface must propagate the validation refusal.
    let pixels = vec![0u8; 12 * 12 + 2 * 6 * 6];
    let res = encode_av1(&pixels, 12, 12);
    assert!(res.is_err(), "non-multiple-of-8 dim must fail");
}

#[test]
fn public_encode_av1_rejects_above_max_dim() {
    use oxideav_av1::encode_av1;
    // Above the single-SB MAX_DIM (64), the dyn driver's
    // `Yuv420Frame::validate` surfaces refusal. Wider extents reach the
    // multi-SB driver, which is not yet plumbed through the
    // (pixels, width, height) public signature this arc.
    let (w, h) = (72u32, 72u32);
    let y_size = (w * h) as usize;
    let uv_size = ((w / 2) * (h / 2)) as usize;
    let pixels = vec![0u8; y_size + 2 * uv_size];
    let res = encode_av1(&pixels, w, h);
    assert!(res.is_err(), "above MAX_DIM dim must fail at this arc");
}

#[test]
fn public_encode_av1_decode_av1_roundtrip_all_8x8_corner_lossless() {
    use oxideav_av1::{decode_av1, encode_av1};
    // 8x8 — the minimum aligned extent the dyn driver accepts. The
    // smallest viable through-the-public-API roundtrip; pins the
    // lower bound of the encode_av1 / decode_av1 envelope.
    let (w, h) = (8u32, 8u32);
    let y_size = (w * h) as usize;
    let uv_size = ((w / 2) * (h / 2)) as usize;
    let mut pixels = vec![0u8; y_size + 2 * uv_size];
    let mut s: u64 = 0xA5A5_5A5A_C3C3_3C3C;
    for byte in pixels.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (s >> 56) as u8;
    }
    let ivf = encode_av1(&pixels, w, h).expect("public encode_av1 succeeds at 8x8");
    let decoded = decode_av1(&ivf).expect("public decode_av1 succeeds");
    match &decoded[0] {
        Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            assert_eq!(*width, w);
            assert_eq!(*height, h);
            assert_eq!(y.as_slice(), &pixels[..y_size]);
            assert_eq!(u.as_slice(), &pixels[y_size..y_size + uv_size]);
            assert_eq!(v.as_slice(), &pixels[y_size + uv_size..]);
        }
        other => panic!("expected Yuv420Dyn at 8x8, got {other:?}"),
    }
}
