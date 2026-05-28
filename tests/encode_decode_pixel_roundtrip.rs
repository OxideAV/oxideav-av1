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
