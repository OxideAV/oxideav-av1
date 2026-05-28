//! End-to-end integration test for the **pixel-space encoder driver**
//! landed in arc 15 (round 221).
//!
//! Validates the encoder's first complete pixel-in / bytes-out path:
//!
//!   1. **Bit-exact pixel roundtrip on flat-DC input.** Encoding a
//!      16×16 luma plane filled with mid-grey (128) at `base_q_idx = 0`
//!      produces a zero `Quant[]` at every BLOCK_4X4 leaf; the
//!      encoder-internal reconstruction (the same `dequantize_step1` +
//!      `inverse_transform_2d` chain the decoder runs) recovers the
//!      input bit-exact. This is the first round-tripping pixel input
//!      through the encoder.
//!   2. **Structural roundtrip through the OBU framework.** The IVF
//!      bytes the driver emits are a valid AV1 elementary stream —
//!      every OBU walks back through [`oxideav_av1::ObuIter`] and the
//!      SequenceHeader / FrameHeader OBUs reparse identically.
//!   3. **Per-leaf dispatch coverage.** The §5.11.4 dispatch order over
//!      the 16 BLOCK_4X4 cells of a 4×4 mi grid is the Z-curve over the
//!      two-level recursive split, so each cell is visited exactly once
//!      and the encoder-internal reconstruction matches the inline
//!      driver reconstruction across all 16 cells.
//!
//! ## What this test does NOT verify (next-arc work)
//!
//!   * Pixel roundtrip through the standalone `decode_av1` entry —
//!     that entry is still a stub. Full decoder pipelining is a
//!     separate arc.
//!   * Bit-exact roundtrip on arbitrary inputs — `q_index = 0` + DCT
//!     is not lossless (the §7.13.3 `Lossless` path requires the WHT
//!     row/column kernel, which has no encoder counterpart yet).
//!     Lossless input/output equivalence at non-mid-grey inputs needs
//!     a forward WHT primitive landed in a subsequent arc.
//!   * (r221 only) Chroma planes. r223 added the YUV path —
//!     [`encode_intra_frame_yuv`] is covered by the
//!     `yuv_encode_*` test cases below.

use oxideav_av1::encoder::pixel_driver::{
    encode_intra_frame_y, encode_intra_frame_yuv, internal_roundtrip, EncodedFrame,
    EncodedFrameYuv, Yuv420Frame16x16, CHROMA_HEIGHT, CHROMA_WIDTH, FRAME_HEIGHT, FRAME_WIDTH,
};
use oxideav_av1::frame_header::parse_frame_header;
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

const TINY_SEQ_PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
const TINY_FRAME_PAYLOAD: &[u8] = &[
    0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67, 0x6c,
    0xc7, 0xee, 0x51, 0x80,
];

fn encode_flat_128_frame() -> EncodedFrame {
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap();
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).unwrap();
    let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
    encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds")
}

#[test]
fn pixel_driver_flat_128_input_reconstructs_bit_exact() {
    let result = encode_flat_128_frame();
    for i in 0..FRAME_HEIGHT {
        for j in 0..FRAME_WIDTH {
            assert_eq!(
                result.reconstructed_y[i][j], 128,
                "reconstructed[{i}][{j}] = {} != 128",
                result.reconstructed_y[i][j]
            );
        }
    }
}

#[test]
fn pixel_driver_flat_128_committed_quants_are_all_zero() {
    let result = encode_flat_128_frame();
    assert_eq!(result.committed_quants.len(), 16);
    for (cell_idx, quant) in result.committed_quants.iter().enumerate() {
        assert_eq!(quant.len(), 16, "cell {cell_idx} quant length");
        for (k, &q) in quant.iter().enumerate() {
            assert_eq!(q, 0, "cell {cell_idx} quant[{k}] = {q}, expected 0");
        }
    }
}

#[test]
fn pixel_driver_ivf_bytes_walk_back_as_td_sh_fh_tg() {
    let result = encode_flat_128_frame();
    // IVF preamble: "DKIF".
    assert_eq!(&result.ivf_bytes[0..4], b"DKIF");
    // frame_count == 1.
    let frame_count = u32::from_le_bytes([
        result.ivf_bytes[24],
        result.ivf_bytes[25],
        result.ivf_bytes[26],
        result.ivf_bytes[27],
    ]);
    assert_eq!(frame_count, 1);
    // Temporal-unit OBU walk: TD + SH + FrameHeader + TileGroup.
    let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
        .collect::<Result<_, _>>()
        .expect("OBU walk succeeds");
    assert_eq!(descs.len(), 4);
    assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
    assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
    assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
    assert_eq!(descs[3].obu_type, ObuType::TileGroup);
}

#[test]
fn pixel_driver_sh_fh_obus_reparse_equal_to_input() {
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap();
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).unwrap();
    let result = encode_flat_128_frame();
    let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
        .collect::<Result<_, _>>()
        .unwrap();
    let reparsed_seq = parse_sequence_header(descs[1].payload).unwrap();
    let mut expected_seq = seq.clone();
    expected_seq.bits_consumed = reparsed_seq.bits_consumed;
    assert_eq!(reparsed_seq, expected_seq);
    let reparsed_fh = parse_frame_header(descs[2].payload, &seq).unwrap();
    let mut expected_fh = fh.clone();
    expected_fh.bits_consumed = reparsed_fh.bits_consumed;
    assert_eq!(reparsed_fh, expected_fh);
}

#[test]
fn pixel_driver_internal_helper_matches_driver_reconstruction() {
    let result = encode_flat_128_frame();
    let recon = internal_roundtrip::reconstruct_from_quants(&result.committed_quants);
    assert_eq!(recon, result.reconstructed_y);
}

// ---------------------------------------------------------------------
// Round 223 — chroma (YUV 4:2:0) integration coverage.
// ---------------------------------------------------------------------

fn encode_flat_128_yuv_frame() -> EncodedFrameYuv {
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap();
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).unwrap();
    let input = Yuv420Frame16x16::default();
    encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds")
}

#[test]
fn pixel_driver_yuv_flat_128_input_reconstructs_bit_exact_on_every_plane() {
    let result = encode_flat_128_yuv_frame();
    for i in 0..FRAME_HEIGHT {
        for j in 0..FRAME_WIDTH {
            assert_eq!(result.reconstructed_y[i][j], 128);
        }
    }
    for i in 0..CHROMA_HEIGHT {
        for j in 0..CHROMA_WIDTH {
            assert_eq!(result.reconstructed_u[i][j], 128);
            assert_eq!(result.reconstructed_v[i][j], 128);
        }
    }
}

#[test]
fn pixel_driver_yuv_flat_128_committed_quants_zero_on_every_plane() {
    let result = encode_flat_128_yuv_frame();
    assert_eq!(result.committed_quants_y.len(), 16);
    assert_eq!(result.committed_quants_u.len(), 4);
    assert_eq!(result.committed_quants_v.len(), 4);
    for q in result.committed_quants_y.iter() {
        for &v in q {
            assert_eq!(v, 0);
        }
    }
    for q in result.committed_quants_u.iter() {
        for &v in q {
            assert_eq!(v, 0);
        }
    }
    for q in result.committed_quants_v.iter() {
        for &v in q {
            assert_eq!(v, 0);
        }
    }
}

#[test]
fn pixel_driver_yuv_lossless_pseudorandom_roundtrips_bit_exact() {
    // Lossless WHT chain ⇒ arbitrary 4:2:0 input round-trips
    // pixel-for-pixel on every plane at base_q_idx = 0.
    let seq = parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap();
    let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).unwrap();
    let mut input = Yuv420Frame16x16 {
        y: [[0u8; FRAME_WIDTH]; FRAME_HEIGHT],
        u: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        v: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    };
    let mut state: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let step = |state: &mut u64| -> u8 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 56) as u8
    };
    for row in input.y.iter_mut() {
        for cell in row.iter_mut() {
            *cell = step(&mut state);
        }
    }
    for row in input.u.iter_mut() {
        for cell in row.iter_mut() {
            *cell = step(&mut state);
        }
    }
    for row in input.v.iter_mut() {
        for cell in row.iter_mut() {
            *cell = step(&mut state);
        }
    }
    let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
    assert_eq!(result.reconstructed_y, input.y);
    assert_eq!(result.reconstructed_u, input.u);
    assert_eq!(result.reconstructed_v, input.v);
}

#[test]
fn pixel_driver_yuv_ivf_bytes_walk_back_as_td_sh_fh_tg() {
    let result = encode_flat_128_yuv_frame();
    assert_eq!(&result.ivf_bytes[0..4], b"DKIF");
    let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
        .collect::<Result<_, _>>()
        .expect("OBU walk succeeds");
    assert_eq!(descs.len(), 4);
    assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
    assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
    assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
    assert_eq!(descs[3].obu_type, ObuType::TileGroup);
}

#[test]
fn pixel_driver_tile_group_obu_carries_a_nonempty_entropy_payload() {
    // The §5.11.1 tile-group body the driver emits must carry actual
    // entropy bytes (the per-block syntax). For a single-tile / 1-byte
    // tile-size frame the body layout is: optional `tg_start` /
    // `tg_end` (skipped — `start_and_end_present = false`),
    // byte-alignment pad, then the last-tile payload (no size prefix).
    // We just verify the OBU payload is nonempty.
    let result = encode_flat_128_frame();
    let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
        .collect::<Result<_, _>>()
        .unwrap();
    let tg_desc = &descs[3];
    assert_eq!(tg_desc.obu_type, ObuType::TileGroup);
    assert!(
        tg_desc.payload_len > 0,
        "tile-group OBU payload must be nonempty"
    );
}
