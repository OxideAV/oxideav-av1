//! r431 — §5.9.15 multi-tile WRITE arm: KEY-frame streams with
//! explicitly configured uniform tile layouts.
//!
//! What these tests pin:
//!
//!   * `(0, 0)` byte-identity — the tile-parameterised entry with the
//!     single-tile layout must reproduce the historical single-tile
//!     stream BYTE FOR BYTE (the per-tile walk collapses to the
//!     whole-frame walk).
//!   * Header shape — the §5.9.15 `tile_info()` round-trips with the
//!     requested `TileColsLog2` / `TileRowsLog2` and the realized
//!     `TileCols` / `TileRows`, and the §5.11.1 tile-group body
//!     carries one complete §8.2 partition per tile with
//!     `tile_size_minus_1` fields on every non-last tile.
//!   * Decode round trip — every multi-tile stream decodes through
//!     the public spec driver ([`oxideav_av1::decode_av1`]) to the
//!     encoder's own reconstruction sample-exact (and to the input
//!     exactly on the lossless arm), across layouts × quantisers ×
//!     content, including CDEF- and LR-electing configurations.
//!
//! The per-tile CDF reset, the §5.11.2 `clear_above_context()`
//! scoping, the tile-scoped §7.11.2 availability (`haveAbove` /
//! `haveLeft` collapse at tile edges) and the §5.11.3 tile-end
//! `sbWidth4` / `sbHeight4` derivations are all load-bearing here: a
//! single wrong neighbour read desyncs the §8.2 coder mid-tile and
//! the decode comparison fails.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.15, §5.11.1, §5.11.2,
//! §5.11.3, §5.11.51, §6.8.14, §6.10.1, §7.11.2.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_key_frame_yuv420_with_q, encode_key_frame_yuv420_with_q_tiles,
    encode_key_frame_yuv_with_q_tiles, parse_tile_group_obu_body, ChromaFormat, Yuv420Frame,
    YuvFrame,
};
use oxideav_av1::frame_header::parse_frame_header;
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

// ---------------------------------------------------------------------
// Content generators.
// ---------------------------------------------------------------------

/// Deterministic LCG noise — worst-case texture (maximises coded
/// symbols per tile, so any cross-tile context leak desyncs fast).
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

/// Strong diagonal gradient — directional intra modes win everywhere,
/// so tile-edge availability (the §7.11.2 `haveAbove` / `haveLeft`
/// collapse) is exercised on every boundary superblock.
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

/// Walk the temporal unit to the `OBU_FRAME`, split it into (frame
/// header, §5.11.1 tile-group body) and return the parsed
/// `TileInfo` plus the per-tile §8.2 payloads.
fn parsed_layout(tu: &[u8]) -> (oxideav_av1::tile_info::TileInfo, Vec<Vec<u8>>) {
    let mut seq = None;
    for desc in ObuIter::new(tu) {
        let desc = desc.expect("own TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("own SH parses"));
            }
            ObuType::Frame => {
                let seq = seq.as_ref().expect("SH precedes the frame OBU");
                let fh = parse_frame_header(desc.payload, seq).expect("own FH parses");
                let ti = fh.tile_info.clone().expect("KEY header carries tile_info");
                let tg_offset = fh.bits_consumed.div_ceil(8);
                let parsed = parse_tile_group_obu_body(
                    &desc.payload[tg_offset..],
                    ti.tile_cols * ti.tile_rows,
                    ti.tile_cols_log2,
                    ti.tile_rows_log2,
                    u32::from(ti.tile_size_bytes),
                )
                .expect("own tile-group body parses");
                let payloads = parsed.tiles.into_iter().map(|t| t.bytes).collect();
                return (ti, payloads);
            }
            _ => {}
        }
    }
    panic!("no OBU_FRAME in the temporal unit");
}

// ---------------------------------------------------------------------
// (0, 0) byte identity.
// ---------------------------------------------------------------------

/// The tile-parameterised entry at `(0, 0)` must be BYTE-IDENTICAL to
/// the historical single-tile entry — across the quantiser ladder and
/// both content shapes (the whole refactor is a no-op on the
/// single-tile layout).
#[test]
fn tiles_0_0_is_byte_identical_to_the_single_tile_entry() {
    for (frame, label) in [(noise(128, 64, 7), "noise"), (gradient(128, 64), "grad")] {
        for q in [0u8, 60, 140] {
            let a = encode_key_frame_yuv420_with_q(&frame, q).expect("single-tile encode");
            let b = encode_key_frame_yuv420_with_q_tiles(&frame, q, 0, 0).expect("(0,0) encode");
            assert_eq!(
                a.ivf_bytes, b.ivf_bytes,
                "{label} q{q}: (0,0) layout must reproduce the single-tile stream byte for byte"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Header + §5.11.1 body shape.
// ---------------------------------------------------------------------

/// 256×128 at `(1, 1)`: sbCols = 4, sbRows = 2 ⇒ 2×2 tiles of 128×64.
/// The header must code the requested log2 pair, the §5.11.1 body
/// must carry 4 payloads (3 size fields + residual last), and every
/// payload must be non-empty (each tile is a complete §8.2 partition
/// with its own `exit_symbol` padding).
#[test]
fn multi_tile_header_and_body_shape() {
    let frame = noise(256, 128, 11);
    let enc = encode_key_frame_yuv420_with_q_tiles(&frame, 60, 1, 1).expect("2x2 encode");
    let (ti, payloads) = parsed_layout(&enc.temporal_unit_bytes);
    assert_eq!((ti.tile_cols, ti.tile_rows), (2, 2));
    assert_eq!((ti.tile_cols_log2, ti.tile_rows_log2), (1, 1));
    assert_eq!(ti.context_update_tile_id, 0);
    assert!(ti.uniform_tile_spacing_flag);
    assert_eq!(ti.mi_col_starts, vec![0, 32, 64]);
    assert_eq!(ti.mi_row_starts, vec![0, 16, 32]);
    assert_eq!(payloads.len(), 4);
    assert!(payloads.iter().all(|p| !p.is_empty()));
}

// ---------------------------------------------------------------------
// Decode round trips: layouts × quantisers × content.
// ---------------------------------------------------------------------

fn assert_tiled_round_trip(frame: &Yuv420Frame, q: u8, cl2: u32, rl2: u32, label: &str) {
    let enc = encode_key_frame_yuv420_with_q_tiles(frame, q, cl2, rl2)
        .unwrap_or_else(|e| panic!("{label} q{q} ({cl2},{rl2}): encode failed: {e:?}"));
    let decoded = spec_frames(&enc.ivf_bytes, 1, label);
    let f = &decoded[0];
    assert_eq!((f.width, f.height), (frame.width, frame.height), "{label}");
    assert_eq!(f.planes[0], enc.recon_y, "{label} q{q} ({cl2},{rl2}): luma");
    assert_eq!(f.planes[1], enc.recon_u, "{label} q{q} ({cl2},{rl2}): U");
    assert_eq!(f.planes[2], enc.recon_v, "{label} q{q} ({cl2},{rl2}): V");
    if q == 0 {
        assert_eq!(f.planes[0], frame.y, "{label} lossless: luma != input");
        assert_eq!(f.planes[1], frame.u, "{label} lossless: U != input");
        assert_eq!(f.planes[2], frame.v, "{label} lossless: V != input");
    }
}

/// Column-only, row-only and 2×2 layouts on 256×128, lossless +
/// mid + high quantisers, both content shapes. The q = 140 rows run
/// with the CDEF / LR elections armed (the default entry) — an
/// adopting election re-emits every tile and re-assembles the body.
#[test]
fn multi_tile_layouts_round_trip_pixel_exact() {
    let shapes: [(Yuv420Frame, &str); 2] = [
        (noise(256, 128, 23), "noise-256x128"),
        (gradient(256, 128), "grad-256x128"),
    ];
    for (frame, label) in &shapes {
        for &(cl2, rl2) in &[(1u32, 0u32), (0, 1), (1, 1)] {
            for q in [0u8, 60, 140] {
                assert_tiled_round_trip(frame, q, cl2, rl2, label);
            }
        }
    }
}

/// The widest legal column split of a 256-wide frame: `(2, 0)` ⇒ four
/// 64-wide tile columns (every tile exactly one superblock wide — the
/// left/above availability collapses on EVERY superblock).
#[test]
fn one_superblock_wide_tile_columns_round_trip() {
    let frame = gradient(256, 64);
    for q in [0u8, 72] {
        assert_tiled_round_trip(&frame, q, 2, 0, "grad-256x64-4cols");
    }
}

/// Non-power-of-two realized tile count: 192-wide (sbCols = 3) at
/// `tile_cols_log2 = 1` realizes ceil(3/2) = 2-SB tiles ⇒ TileCols =
/// 2 with UNEQUAL widths (128 + 64) — the §5.9.15 uniform walk's
/// rounding arm.
#[test]
fn uneven_uniform_tile_widths_round_trip() {
    let frame = noise(192, 128, 31);
    let enc = encode_key_frame_yuv420_with_q_tiles(&frame, 60, 1, 1).expect("192x128 encode");
    let (ti, payloads) = parsed_layout(&enc.temporal_unit_bytes);
    assert_eq!((ti.tile_cols, ti.tile_rows), (2, 2));
    assert_eq!(ti.mi_col_starts, vec![0, 32, 48]);
    assert_eq!(payloads.len(), 4);
    assert_tiled_round_trip(&frame, 60, 1, 1, "noise-192x128");
}

/// Layouts outside the §5.9.15 legal window are rejected up front
/// (e.g. more tile columns than superblock columns).
#[test]
fn illegal_layouts_are_rejected() {
    let frame = noise(64, 64, 3);
    // sbCols = sbRows = 1 ⇒ maxLog2 = 0 on both axes.
    assert!(encode_key_frame_yuv420_with_q_tiles(&frame, 60, 1, 0).is_err());
    assert!(encode_key_frame_yuv420_with_q_tiles(&frame, 60, 0, 1).is_err());
}

// ---------------------------------------------------------------------
// INTER arm: multi-tile GOPs (KEY + P frames, every frame tiled).
// ---------------------------------------------------------------------

/// Shifting-gradient content so the P-frames commit real inter tools
/// (motion compensation across the frame; the shift crosses tile
/// boundaries, so §7.11.3 prediction reads the reference across tile
/// edges — legal — while every syntax context stays tile-scoped).
fn moving(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = (((r * 3) + (c + 5 * t) * 5) % 256) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = ((128 + r + (c + 3 * t)) % 256) as u8;
            f.v[r * cw + c] = ((64 + r * 2 + (c + 2 * t)) % 256) as u8;
        }
    }
    f
}

/// A 3-frame GOP at `(1, 1)` on 192×128 (2×2 tiles, uneven 128+64
/// columns): every frame — the KEY and both P-frames, with the full
/// election set armed (gm / primary-ref / hp / delta-q / CDEF / LR;
/// each election re-emits and re-assembles ALL tiles) — must decode
/// through the public spec driver to the encoder reconstruction
/// sample-exact.
#[test]
fn multi_tile_gop_round_trips_pixel_exact() {
    use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning};
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| moving(192, 128, t)).collect();
    let tuning = GopTuning {
        tiles: (1, 1),
        ..GopTuning::default()
    };
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 72, &[], tuning).expect("tiled GOP");
    // Header shape on the KEY temporal unit.
    let (ti, payloads) = parsed_layout(&enc.gop.temporal_units[0]);
    assert_eq!((ti.tile_cols, ti.tile_rows), (2, 2));
    assert_eq!(payloads.len(), 4);
    let decoded = spec_frames(&enc.gop.ivf_bytes, 3, "tiled-gop");
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "frame {i} V");
    }
}

/// `(0, 0)` GOP byte identity — the tiles knob at the single-tile
/// layout must not move a single bit of the default GOP encode.
#[test]
fn gop_tiles_0_0_is_byte_identical() {
    use oxideav_av1::encoder::{
        encode_gop_yuv420_with_q, encode_gop_yuv420_with_q_seg_tuned, GopTuning,
    };
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| moving(128, 64, t)).collect();
    let a = encode_gop_yuv420_with_q(&frames, 72).expect("default GOP");
    let b = encode_gop_yuv420_with_q_seg_tuned(&frames, 72, &[], GopTuning::default())
        .expect("(0,0) GOP");
    assert_eq!(
        a.ivf_bytes, b.gop.ivf_bytes,
        "(0,0) tiles must reproduce the default GOP stream byte for byte"
    );
}

// ---------------------------------------------------------------------
// §7.3 camera-frame tile COLUMNS → §5.12 tile-list assembly.
// ---------------------------------------------------------------------

/// One 256×64 camera frame at `tile_cols_log2 = 2` (four 64-wide
/// §7.3.1-conformant tiles) — a §5.12 tile list references its four
/// columns through `anchor_tile_col` and assembles a 2×2 output
/// whose quadrants equal the camera reconstruction's columns
/// sample-exact. This is the r431 unlock over the r430 shape (one
/// frame = one tile): tile-list material now comes from WITHIN a
/// camera frame, not only across frames.
#[test]
fn camera_frame_tile_columns_feed_the_tile_list() {
    use oxideav_av1::decoder::SpecFrame;
    use oxideav_av1::encoder::encode_camera_frame_yuv420_tiles;
    use oxideav_av1::tile_list::{decode_tile_list, TileListEntry, TileListObu};

    // Textured anchor + a camera view the RD ladder can motion-
    // compensate from (content shifted, brightness ramped).
    let anchor = {
        let mut f = Yuv420Frame::filled(256, 64, 0);
        for r in 0..64usize {
            for c in 0..256usize {
                f.y[r * 256 + c] = ((r * 7 + c * 3 + (r / 8) * (c / 8) * 5) % 220) as u8 + 18;
            }
        }
        for v in f.u.iter_mut() {
            *v = 90;
        }
        for v in f.v.iter_mut() {
            *v = 160;
        }
        f
    };
    let cam = {
        let mut f = anchor.clone();
        for r in 0..64usize {
            for c in (2..256usize).rev() {
                f.y[r * 256 + c] = anchor.y[r * 256 + c - 2].saturating_add((r % 3) as u8);
            }
        }
        f
    };
    let e = encode_camera_frame_yuv420_tiles(&cam, &anchor, 60, 2).expect("tiled camera encode");
    assert_eq!(e.coded_tiles.len(), 4, "four tile columns");
    assert_eq!(e.coded_tile_data, e.coded_tiles[0]);
    let ti = e.fh.tile_info.clone().expect("camera header tile_info");
    assert_eq!((ti.tile_cols, ti.tile_rows), (4, 1));

    let anchor_frame = SpecFrame {
        width: 256,
        height: 64,
        planes: vec![anchor.y.clone(), anchor.u.clone(), anchor.v.clone()],
        plane_dims: vec![(256, 64), (128, 32), (128, 32)],
        bit_depth: 8,
    };
    // 2×2 output from the four columns of the ONE camera frame.
    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 1,
        output_frame_height_in_tiles_minus_1: 1,
        entries: (0..4u8)
            .map(|col| TileListEntry {
                anchor_frame_idx: 0,
                anchor_tile_row: 0,
                anchor_tile_col: col,
                coded_tile_data: e.coded_tiles[col as usize].clone(),
            })
            .collect(),
    };
    let out = decode_tile_list(&e.seq, &e.fh, &[anchor_frame], &tl).expect("tile list decodes");
    assert_eq!((out.width, out.height), (128, 128));
    // Quadrant (i%2, i/2) of the output == column i of the camera
    // reconstruction (per plane, 4:2:0 subsampled on chroma).
    for i in 0..4usize {
        let (qx, qy) = (i % 2, i / 2);
        for plane in 0..3usize {
            let sub = usize::from(plane > 0);
            let (tw, th) = (64 >> sub, 64 >> sub);
            let (pw, _) = out.plane_dims[plane];
            let src: &[u8] = match plane {
                0 => &e.recon.y,
                1 => &e.recon.u,
                _ => &e.recon.v,
            };
            let src_w = 256 >> sub;
            for y in 0..th {
                let d0 = (qy * th + y) * pw as usize + qx * tw;
                let s0 = y * src_w + i * tw;
                assert_eq!(
                    &out.planes[plane][d0..d0 + tw],
                    &src[s0..s0 + tw],
                    "plane {plane} row {y} of column {i}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// Env-gated fixture staging (black-box reference-decoder validation
// + docs corpus material — skips silently in CI).
// ---------------------------------------------------------------------

/// Writes the r431 multi-tile staging set under
/// `OXIDEAV_AV1_TILE_DIR`: per-layout IVF + the encoder
/// reconstruction as raw planar YUV (the black-box reference decode
/// must reproduce it byte for byte).
#[test]
fn multi_tile_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_TILE_DIR") else {
        eprintln!("OXIDEAV_AV1_TILE_DIR unset — skipping the multi-tile staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let configs: [(Yuv420Frame, u8, u32, u32, &str); 4] = [
        (noise(256, 128, 23), 60, 1, 1, "tiles-2x2-256x128-q60-noise"),
        (gradient(256, 128), 140, 1, 1, "tiles-2x2-256x128-q140-grad"),
        (gradient(256, 64), 72, 2, 0, "tiles-4x1-256x64-q72-grad"),
        (noise(192, 128, 31), 60, 1, 1, "tiles-2x2-192x128-q60-noise"),
    ];
    for (frame, q, cl2, rl2, name) in &configs {
        let enc = encode_key_frame_yuv420_with_q_tiles(frame, *q, *cl2, *rl2)
            .unwrap_or_else(|e| panic!("{name}: encode failed: {e:?}"));
        std::fs::write(root.join(format!("{name}.ivf")), &enc.ivf_bytes).expect("write ivf");
        let mut yuv: Vec<u8> = Vec::new();
        yuv.extend_from_slice(&enc.recon_y);
        yuv.extend_from_slice(&enc.recon_u);
        yuv.extend_from_slice(&enc.recon_v);
        std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
    }
    // The tiled GOP (KEY + 2 P-frames, full election set, 2×2 tiles
    // on every frame).
    {
        use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning};
        let frames: Vec<Yuv420Frame> = (0..3).map(|t| moving(192, 128, t)).collect();
        let tuning = GopTuning {
            tiles: (1, 1),
            ..GopTuning::default()
        };
        let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 72, &[], tuning).expect("tiled GOP");
        let name = "tiles-2x2-gop3-192x128-q72-move";
        std::fs::write(root.join(format!("{name}.ivf")), &enc.gop.ivf_bytes).expect("write ivf");
        let mut yuv: Vec<u8> = Vec::new();
        for rc in &enc.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
    }
}

// ---------------------------------------------------------------------
// General-format arm: 10-bit 4:2:2 multi-tile.
// ---------------------------------------------------------------------

/// The general entry rides the same per-tile walk: a 10-bit 4:2:2
/// 256×64 two-column encode must round-trip sample-exact.
#[test]
fn ten_bit_422_two_tile_columns_round_trip() {
    let (w, h) = (256u32, 64u32);
    let mut f = YuvFrame::filled(w, h, 10, ChromaFormat::Yuv422, 512);
    let (wu, hu) = (w as usize, h as usize);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = ((r * 13 + c * 7) % 1024) as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = ((512 + r * 5 + c * 3) % 1024) as u16;
            f.v[r * cw + c] = ((256 + r * 2 + c * 9) % 1024) as u16;
        }
    }
    let enc = encode_key_frame_yuv_with_q_tiles(&f, 60, 1, 0).expect("10-bit 4:2:2 encode");
    let decoded = spec_frames(&enc.ivf_bytes, 1, "422-10bit-2cols");
    let d = &decoded[0];
    assert_eq!(d.bit_depth, 10);
    // 10-bit planes surface as little-endian 2-byte samples.
    let pack = |p: &[u16]| -> Vec<u8> { p.iter().flat_map(|&v| v.to_le_bytes()).collect() };
    assert_eq!(d.planes[0], pack(&enc.recon_y), "422-10bit luma");
    assert_eq!(d.planes[1], pack(&enc.recon_u), "422-10bit U");
    assert_eq!(d.planes[2], pack(&enc.recon_v), "422-10bit V");
}
