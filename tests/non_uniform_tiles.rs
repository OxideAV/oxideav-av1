//! r433 — §5.9.15 NON-UNIFORM tile layouts on the WRITE side
//! (`uniform_tile_spacing_flag = 0`): explicitly sized tile columns /
//! rows coded through the `width_in_sbs_minus_1` /
//! `height_in_sbs_minus_1` `ns()` walks.
//!
//! What these tests pin:
//!
//!   * Header shape — the emitted `tile_info()` codes the flag at 0,
//!     the parser recovers the EXACT requested column/row starts, and
//!     `TileColsLog2` / `TileRowsLog2` take the §5.9.15 ceil-log2 of
//!     the realized counts.
//!   * Decode round trips — KEY frames and full GOPs under uneven
//!     explicit layouts decode through the public spec driver to the
//!     encoder reconstruction sample-exact (lossless arm equals the
//!     input exactly). The uneven splits place tile boundaries where
//!     no uniform layout can, so the per-tile §8.2/§8.3.1 resets and
//!     the tile-scoped availability run at fresh offsets.
//!   * §5.9.15 legality — layouts violating the window (wrong sum,
//!     zero-size tile, over-wide tile) are rejected up front.
//!
//! Spec: docs/video/av1/av1-spec.txt §4.10.7, §5.9.15, §5.11.1,
//! §6.8.14.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_tile_layout, encode_key_frame_yuv420_with_q_tile_layout, Yuv420Frame,
};
use oxideav_av1::frame_header::parse_frame_header;
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;
use oxideav_av1::tile_info::TileInfo;

// ---------------------------------------------------------------------
// Content.
// ---------------------------------------------------------------------

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

// ---------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------

/// Parse the KEY temporal unit's frame header and surface its
/// `TileInfo`.
fn parsed_tile_info(tu: &[u8]) -> TileInfo {
    let mut seq = None;
    for desc in ObuIter::new(tu) {
        let desc = desc.expect("own TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("own SH parses"));
            }
            ObuType::Frame | ObuType::FrameHeader => {
                let fh = parse_frame_header(desc.payload, seq.as_ref().expect("SH first"))
                    .expect("own FH parses");
                return fh.tile_info.expect("header carries tile_info");
            }
            _ => {}
        }
    }
    panic!("no frame OBU in the temporal unit");
}

fn spec_frames(ivf: &[u8], expected: usize, label: &str) -> Vec<oxideav_av1::decoder::SpecFrame> {
    let frames = oxideav_av1::decode_av1(ivf)
        .unwrap_or_else(|e| panic!("{label}: decode_av1 rejected stream: {e:?}"));
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

fn assert_key_round_trip(frame: &Yuv420Frame, q: u8, widths: &[u32], heights: &[u32], label: &str) {
    let enc = encode_key_frame_yuv420_with_q_tile_layout(frame, q, widths, heights)
        .unwrap_or_else(|e| panic!("{label} q{q} {widths:?}x{heights:?}: encode failed: {e:?}"));
    let ti = parsed_tile_info(&enc.temporal_unit_bytes);
    assert!(!ti.uniform_tile_spacing_flag, "{label}: flag must be 0");
    assert_eq!(ti.tile_cols as usize, widths.len(), "{label}: TileCols");
    assert_eq!(ti.tile_rows as usize, heights.len(), "{label}: TileRows");
    // The parser must recover the EXACT requested starts (sb units
    // are 16 mi on the 64-pel superblocks every stream here codes).
    let mut start = 0u32;
    for (i, &w) in widths.iter().enumerate() {
        assert_eq!(ti.mi_col_starts[i], start * 16, "{label}: col start {i}");
        start += w;
    }
    let mut start = 0u32;
    for (i, &h) in heights.iter().enumerate() {
        assert_eq!(ti.mi_row_starts[i], start * 16, "{label}: row start {i}");
        start += h;
    }
    let decoded = spec_frames(&enc.ivf_bytes, 1, label);
    let f = &decoded[0];
    assert_eq!(f.planes[0], enc.recon_y, "{label}: luma");
    assert_eq!(f.planes[1], enc.recon_u, "{label}: U");
    assert_eq!(f.planes[2], enc.recon_v, "{label}: V");
    if q == 0 {
        assert_eq!(f.planes[0], frame.y, "{label} lossless: luma != input");
        assert_eq!(f.planes[1], frame.u, "{label} lossless: U != input");
        assert_eq!(f.planes[2], frame.v, "{label} lossless: V != input");
    }
}

// ---------------------------------------------------------------------
// KEY-frame round trips.
// ---------------------------------------------------------------------

/// Uneven explicit layouts a uniform layout cannot express: a 192-wide
/// frame (3 superblock columns) split 1+2 and 2+1, plus the
/// three-single-column split, across the quantiser ladder (q = 140
/// runs with the CDEF / LR elections armed) and both content shapes.
#[test]
fn uneven_key_layouts_round_trip_pixel_exact() {
    let shapes: [(Yuv420Frame, &str); 2] = [
        (noise(192, 128, 51), "noise-192x128"),
        (gradient(192, 128), "grad-192x128"),
    ];
    for (frame, label) in &shapes {
        for (widths, heights) in [
            (&[1u32, 2][..], &[1u32, 1][..]),
            (&[2, 1][..], &[1, 1][..]),
            (&[1, 1, 1][..], &[2][..]),
        ] {
            for q in [0u8, 60, 140] {
                assert_key_round_trip(frame, q, widths, heights, label);
            }
        }
    }
}

/// A wider frame: 320×64 (5 superblock columns, 1 row) split
/// 1+3+1 — an interior wide tile flanked by single-superblock
/// columns.
#[test]
fn wide_uneven_columns_round_trip() {
    let frame = gradient(320, 64);
    for q in [0u8, 72] {
        assert_key_round_trip(&frame, q, &[1, 3, 1], &[1], "grad-320x64-1-3-1");
    }
}

/// Uneven ROWS: 128×192 (2 superblock cols, 3 rows) split 1+2 rows.
#[test]
fn uneven_rows_round_trip() {
    let frame = noise(128, 192, 77);
    for q in [0u8, 72] {
        assert_key_round_trip(&frame, q, &[2], &[1, 2], "noise-128x192-rows-1-2");
    }
}

// ---------------------------------------------------------------------
// GOP round trip.
// ---------------------------------------------------------------------

/// A 3-frame GOP under the 2+1 / 1+1 explicit layout: every frame
/// (KEY + both P-frames with the full election set) codes
/// `uniform_tile_spacing_flag = 0` and decodes pixel-exact.
#[test]
fn uneven_gop_round_trips_pixel_exact() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| noise(192, 128, 300 + t)).collect();
    let enc = encode_gop_yuv420_with_q_tile_layout(&frames, 72, &[2, 1], &[1, 1])
        .expect("explicit-layout GOP");
    let ti = parsed_tile_info(&enc.temporal_units[0]);
    assert!(!ti.uniform_tile_spacing_flag);
    assert_eq!((ti.tile_cols, ti.tile_rows), (2, 2));
    assert_eq!(ti.mi_col_starts, vec![0, 32, 48]);
    let decoded = spec_frames(&enc.ivf_bytes, 3, "uneven-gop");
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.recon[i].v, "frame {i} V");
    }
}

// ---------------------------------------------------------------------
// §5.9.15 legality.
// ---------------------------------------------------------------------

/// Layouts outside the legal window are rejected up front: wrong
/// column sum, a zero-width tile, wrong row sum.
/// r456 — a frame whose width / height is NOT a superblock multiple:
/// the explicit writer codes the LAST column / row as the superblock
/// CEILING of its mi extent (§5.9.15 sets `MiColStarts[ TileCols ] =
/// MiCols`), so a 160-wide frame (2.5 superblocks) splits `[2, 1]` and
/// `[1, 2]` and a 96-high one (1.5 superblocks) splits `[1, 1]`.
/// (Before r456 the writer took the floor and underflowed on every
/// such frame — surfaced by the superres × explicit-layout pairing,
/// whose 160-wide coded extent is exactly this shape.)
#[test]
fn non_superblock_multiple_extents_round_trip() {
    let frame = noise(160, 96, 5);
    assert_key_round_trip(&frame, 72, &[2, 1], &[2], "160x96 [2,1]");
    assert_key_round_trip(&frame, 72, &[1, 2], &[1, 1], "160x96 [1,2] rows [1,1]");
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| noise(160, 96, 40 + t)).collect();
    let enc = encode_gop_yuv420_with_q_tile_layout(&frames, 72, &[2, 1], &[1, 1])
        .expect("160x96 GOP with the [2,1]x[1,1] layout encodes");
    let decoded = spec_frames(&enc.ivf_bytes, 3, "160x96 gop");
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.recon[i].y, "160x96 gop frame {i}: luma");
        assert_eq!(f.planes[1], enc.recon[i].u, "160x96 gop frame {i}: U");
        assert_eq!(f.planes[2], enc.recon[i].v, "160x96 gop frame {i}: V");
    }
}

#[test]
fn illegal_explicit_layouts_are_rejected() {
    let frame = noise(192, 128, 9);
    // Sum != sbCols (3).
    assert!(encode_key_frame_yuv420_with_q_tile_layout(&frame, 60, &[1, 1], &[1, 1]).is_err());
    assert!(encode_key_frame_yuv420_with_q_tile_layout(&frame, 60, &[2, 2], &[1, 1]).is_err());
    // Zero-width tile.
    assert!(encode_key_frame_yuv420_with_q_tile_layout(&frame, 60, &[0, 3], &[1, 1]).is_err());
    // Sum != sbRows (2).
    assert!(encode_key_frame_yuv420_with_q_tile_layout(&frame, 60, &[1, 2], &[1, 2]).is_err());
    // Empty axes.
    assert!(encode_key_frame_yuv420_with_q_tile_layout(&frame, 60, &[], &[2]).is_err());
}

// ---------------------------------------------------------------------
// Black-box revalidation hook.
// ---------------------------------------------------------------------

/// Env-gated dump for external validation (`AV1_NUT_DUMP_DIR`): the
/// non-uniform KEY + GOP streams for the black-box reference
/// decoders. Inert otherwise.
#[test]
fn dump_non_uniform_streams_for_blackbox_validation() {
    let Ok(dir) = std::env::var("AV1_NUT_DUMP_DIR") else {
        return;
    };
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).expect("dump dir");
    let frame = noise(192, 128, 51);
    let key = encode_key_frame_yuv420_with_q_tile_layout(&frame, 72, &[1, 2], &[1, 1])
        .expect("non-uniform KEY");
    std::fs::write(dir.join("nut-key-1p2.ivf"), &key.ivf_bytes).expect("write");
    let wide = gradient(320, 64);
    let key2 = encode_key_frame_yuv420_with_q_tile_layout(&wide, 72, &[1, 3, 1], &[1])
        .expect("non-uniform KEY 1-3-1");
    std::fs::write(dir.join("nut-key-1-3-1.ivf"), &key2.ivf_bytes).expect("write");
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| noise(192, 128, 300 + t)).collect();
    let gop = encode_gop_yuv420_with_q_tile_layout(&frames, 72, &[2, 1], &[1, 1])
        .expect("non-uniform GOP");
    std::fs::write(dir.join("nut-gop-2p1.ivf"), &gop.ivf_bytes).expect("write");
}
