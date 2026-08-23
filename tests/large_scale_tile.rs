//! r430 — §7.3 large-scale-tile decoding end to end: self-built
//! camera frames (the §7.3.1-constrained inter encode) packaged as
//! §5.12 tile-list entries and decoded against an externally supplied
//! anchor set, quadrant by quadrant.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.12, §6.11, §7.3
//! (incl. §7.3.2 decode camera tile process).

use oxideav_av1::decoder::SpecFrame;
use oxideav_av1::encoder::{
    encode_camera_frame_yuv420, write_obu_with_size, write_sequence_header_obu, ObuHeader,
    Yuv420Frame,
};
use oxideav_av1::obu::ObuType;
use oxideav_av1::tile_list::{
    decode_tile_list, decode_tile_list_stream, parse_tile_list_obu, write_tile_list_obu,
    TileListEntry, TileListObu,
};
use oxideav_av1::Error;

/// A textured 64x64 plane set, seeded so each anchor differs.
fn textured(seed: u32, w: u32, h: u32) -> Yuv420Frame {
    let mut y = vec![0u8; (w * h) as usize];
    for r in 0..h as usize {
        for c in 0..w as usize {
            let v = seed
                .wrapping_mul(31)
                .wrapping_add((r as u32) * 7)
                .wrapping_add((c as u32) * 13)
                .wrapping_add(((r / 8) as u32 * (c / 8) as u32) * 5);
            y[r * w as usize + c] = (v % 220) as u8 + 18;
        }
    }
    let cw = (w / 2) as usize;
    let ch = (h / 2) as usize;
    let u = vec![(80 + 9 * seed % 100) as u8; cw * ch];
    let v = vec![(200u8).wrapping_sub((7 * seed % 90) as u8); cw * ch];
    Yuv420Frame {
        width: w,
        height: h,
        y,
        u,
        v,
    }
}

/// A camera view of `anchor`: the anchor content shifted right by
/// two pixels with a small brightness ramp — close enough that the
/// RD ladder motion-compensates from the anchor.
fn camera_view(anchor: &Yuv420Frame) -> Yuv420Frame {
    let (w, h) = (anchor.width as usize, anchor.height as usize);
    let mut y = vec![0u8; w * h];
    for r in 0..h {
        for c in 0..w {
            let src_c = c.saturating_sub(2);
            y[r * w + c] = anchor.y[r * w + src_c].saturating_add((r % 3) as u8);
        }
    }
    Yuv420Frame {
        width: anchor.width,
        height: anchor.height,
        y,
        u: anchor.u.clone(),
        v: anchor.v.clone(),
    }
}

/// Wrap raw 4:2:0 planes as the decoder-facing anchor type.
fn anchor_spec_frame(f: &Yuv420Frame) -> SpecFrame {
    SpecFrame {
        width: f.width,
        height: f.height,
        planes: vec![f.y.clone(), f.u.clone(), f.v.clone()],
        plane_dims: vec![
            (f.width, f.height),
            (f.width / 2, f.height / 2),
            (f.width / 2, f.height / 2),
        ],
        bit_depth: 8,
    }
}

/// Flatten a SpecFrame region compare: quadrant (qx, qy) of `frame`
/// against a full 64x64 recon plane set.
fn assert_quadrant_matches(
    frame: &SpecFrame,
    qx: u32,
    qy: u32,
    recon: &oxideav_av1::encoder::CameraFrameEncode,
    tag: &str,
) {
    let tile_w = 64u32;
    let tile_h = 64u32;
    for (plane, (dst_plane, sub)) in [(0usize, 0u32), (1, 1), (2, 1)].iter().enumerate() {
        let _ = dst_plane;
        let (pw, _ph) = frame.plane_dims[plane];
        let (tw, th) = (tile_w >> sub, tile_h >> sub);
        let (dx, dy) = ((qx * tile_w) >> sub, (qy * tile_h) >> sub);
        let src: &[u8] = match plane {
            0 => &recon.recon.y,
            1 => &recon.recon.u,
            _ => &recon.recon.v,
        };
        for y in 0..th as usize {
            let d0 = (dy as usize + y) * pw as usize + dx as usize;
            let s0 = y * tw as usize;
            assert_eq!(
                &frame.planes[plane][d0..d0 + tw as usize],
                &src[s0..s0 + tw as usize],
                "{tag}: plane {plane} row {y} of quadrant ({qx},{qy})"
            );
        }
    }
}

#[test]
fn four_camera_tiles_assemble_into_a_2x2_output_frame() {
    // Four distinct anchors, four camera views, one §5.12 tile list
    // assembling a 128x128 output from four 64x64 coded tiles.
    let anchors: Vec<Yuv420Frame> = (0..4).map(|s| textured(s, 64, 64)).collect();
    let cameras: Vec<Yuv420Frame> = anchors.iter().map(camera_view).collect();
    let encodes: Vec<_> = cameras
        .iter()
        .zip(&anchors)
        .map(|(cam, anc)| encode_camera_frame_yuv420(cam, anc, 60).expect("camera encodes"))
        .collect();
    // §7.3 carries ONE frame header for the whole tile list — the
    // same-config encodes must agree on it.
    for e in &encodes[1..] {
        assert_eq!(
            e.frame_header_payload, encodes[0].frame_header_payload,
            "same-config camera frames share one header"
        );
    }

    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 1,
        output_frame_height_in_tiles_minus_1: 1,
        entries: encodes
            .iter()
            .enumerate()
            .map(|(i, e)| TileListEntry {
                anchor_frame_idx: i as u8,
                anchor_tile_row: 0,
                anchor_tile_col: 0,
                coded_tile_data: e.coded_tile_data.clone(),
            })
            .collect(),
    };
    let anchor_frames: Vec<SpecFrame> = anchors.iter().map(anchor_spec_frame).collect();
    let out = decode_tile_list(&encodes[0].seq, &encodes[0].fh, &anchor_frames, &tl)
        .expect("tile list decodes");
    assert_eq!((out.width, out.height), (128, 128));
    assert_eq!(out.bit_depth, 8);
    for (i, e) in encodes.iter().enumerate() {
        let (qx, qy) = ((i as u32) % 2, (i as u32) / 2);
        assert_quadrant_matches(&out, qx, qy, e, "2x2 assembly");
    }

    // The byte-level packaging: SH + FH + TILE_LIST OBUs.
    let mut stream = Vec::new();
    write_obu_with_size(
        &mut stream,
        &ObuHeader::new(ObuType::SequenceHeader),
        &write_sequence_header_obu(&encodes[0].seq),
    );
    write_obu_with_size(
        &mut stream,
        &ObuHeader::new(ObuType::FrameHeader),
        &encodes[0].frame_header_payload,
    );
    write_obu_with_size(
        &mut stream,
        &ObuHeader::new(ObuType::TileList),
        &write_tile_list_obu(&tl).expect("tile list writes"),
    );
    let out2 = decode_tile_list_stream(&stream, &anchor_frames).expect("stream decodes");
    assert_eq!(out, out2, "stream walker must reproduce the direct call");

    // And the tile-list OBU round trip.
    let bytes = write_tile_list_obu(&tl).expect("writes");
    assert_eq!(parse_tile_list_obu(&bytes).expect("parses"), tl);
}

#[test]
fn repeated_anchor_entries_and_partial_coverage_are_honoured() {
    // One anchor, two entries referencing it, on a 2x2 grid — the
    // remaining two output tiles stay untouched (zero), per §7.3.1
    // "the output frame may not be fully covered with decoded tiles".
    let anchor = textured(9, 64, 64);
    let cam = camera_view(&anchor);
    let e = encode_camera_frame_yuv420(&cam, &anchor, 48).expect("camera encodes");
    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 1,
        output_frame_height_in_tiles_minus_1: 1,
        entries: vec![
            TileListEntry {
                anchor_frame_idx: 0,
                anchor_tile_row: 0,
                anchor_tile_col: 0,
                coded_tile_data: e.coded_tile_data.clone(),
            },
            TileListEntry {
                anchor_frame_idx: 0,
                anchor_tile_row: 0,
                anchor_tile_col: 0,
                coded_tile_data: e.coded_tile_data.clone(),
            },
        ],
    };
    let anchor_frames = vec![anchor_spec_frame(&anchor)];
    let out = decode_tile_list(&e.seq, &e.fh, &anchor_frames, &tl).expect("decodes");
    assert_quadrant_matches(&out, 0, 0, &e, "repeat entry 0");
    assert_quadrant_matches(&out, 1, 0, &e, "repeat entry 1");
    // Bottom half untouched.
    let (pw, _) = out.plane_dims[0];
    assert!(
        out.planes[0][(64 * pw as usize)..].iter().all(|&v| v == 0),
        "uncovered output tiles must stay untouched"
    );
}

#[test]
fn lossless_camera_tile_reproduces_the_source() {
    let anchor = textured(3, 64, 64);
    let cam = camera_view(&anchor);
    let e = encode_camera_frame_yuv420(&cam, &anchor, 0).expect("lossless camera encodes");
    // base_q_idx == 0: recon == input.
    assert_eq!(e.recon.y, cam.y);
    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 0,
        entries: vec![TileListEntry {
            anchor_frame_idx: 0,
            anchor_tile_row: 0,
            anchor_tile_col: 0,
            coded_tile_data: e.coded_tile_data.clone(),
        }],
    };
    let out = decode_tile_list(&e.seq, &e.fh, &[anchor_spec_frame(&anchor)], &tl).expect("decodes");
    assert_eq!(out.planes[0], cam.y, "lossless luma");
    assert_eq!(out.planes[1], cam.u, "lossless U");
    assert_eq!(out.planes[2], cam.v, "lossless V");
}

#[test]
fn wide_camera_frame_is_its_own_multiple_of_sb_tile() {
    // 128x64: TileWidth = 128 = 2 * TileHeight — the §7.3.1
    // "integer multiple of TileHeight" arm beyond the square case.
    let anchor = textured(5, 128, 64);
    let cam = camera_view(&anchor);
    let e = encode_camera_frame_yuv420(&cam, &anchor, 60).expect("camera encodes");
    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 0,
        entries: vec![TileListEntry {
            anchor_frame_idx: 0,
            anchor_tile_row: 0,
            anchor_tile_col: 0,
            coded_tile_data: e.coded_tile_data.clone(),
        }],
    };
    let out = decode_tile_list(&e.seq, &e.fh, &[anchor_spec_frame(&anchor)], &tl).expect("decodes");
    assert_eq!((out.width, out.height), (128, 64));
    assert_eq!(out.planes[0], e.recon.y);
    assert_eq!(out.planes[1], e.recon.u);
    assert_eq!(out.planes[2], e.recon.v);
}

#[test]
fn conformance_gate_rejects_out_of_envelope_inputs() {
    let anchor = textured(7, 64, 64);
    let cam = camera_view(&anchor);
    let e = encode_camera_frame_yuv420(&cam, &anchor, 60).expect("camera encodes");
    let entry = TileListEntry {
        anchor_frame_idx: 0,
        anchor_tile_row: 0,
        anchor_tile_col: 0,
        coded_tile_data: e.coded_tile_data.clone(),
    };
    let tl1 = TileListObu {
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 0,
        entries: vec![entry.clone()],
    };
    let anchors = vec![anchor_spec_frame(&anchor)];

    // More entries than output tiles (§7.3.1).
    let tl_over = TileListObu {
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 0,
        entries: vec![entry.clone(), entry.clone()],
    };
    assert_eq!(
        decode_tile_list(&e.seq, &e.fh, &anchors, &tl_over),
        Err(Error::TileListInvalid)
    );

    // anchor_frame_idx beyond the anchor array.
    let mut tl_bad = tl1.clone();
    tl_bad.entries[0].anchor_frame_idx = 1;
    assert_eq!(
        decode_tile_list(&e.seq, &e.fh, &anchors, &tl_bad),
        Err(Error::TileListInvalid)
    );

    // anchor_tile_col outside the frame's tile grid.
    let mut tl_bad = tl1.clone();
    tl_bad.entries[0].anchor_tile_col = 1;
    assert_eq!(
        decode_tile_list(&e.seq, &e.fh, &anchors, &tl_bad),
        Err(Error::TileListInvalid)
    );

    // Anchor of the wrong geometry.
    let small = textured(1, 128, 64);
    assert_eq!(
        decode_tile_list(&e.seq, &e.fh, &[anchor_spec_frame(&small)], &tl1),
        Err(Error::TileListInvalid)
    );

    // A frame header outside the §7.3 constraint list: unfreeze the
    // CDF flags.
    let mut fh_bad = e.fh.clone();
    fh_bad.disable_cdf_update = false;
    assert_eq!(
        decode_tile_list(&e.seq, &fh_bad, &anchors, &tl1),
        Err(Error::TileListInvalid)
    );
    let mut fh_bad = e.fh.clone();
    fh_bad.refresh_frame_flags = 1;
    assert_eq!(
        decode_tile_list(&e.seq, &fh_bad, &anchors, &tl1),
        Err(Error::TileListInvalid)
    );
    // A sequence header outside the list: order hints on.
    let mut seq_bad = e.seq.clone();
    seq_bad.enable_order_hint = true;
    assert_eq!(
        decode_tile_list(&seq_bad, &e.fh, &anchors, &tl1),
        Err(Error::TileListInvalid)
    );
}

// ---------------------------------------------------------------------
// r433 — 2-D camera grids: tile ROWS via taller camera frames.
// ---------------------------------------------------------------------

/// A 128×128 camera frame at `tile_cols_log2 = 1` codes a 2×2 grid
/// of 64×64 §7.3.1-conformant tiles (each one superblock high — the
/// r433 write arm forces one tile row per superblock row). A §5.12
/// tile list addressing all four grid tiles through
/// `(anchor_tile_row, anchor_tile_col)` reassembles the full frame
/// byte-exact to the camera reconstruction.
#[test]
fn camera_grid_2x2_reassembles_through_anchor_tile_rows() {
    use oxideav_av1::encoder::encode_camera_frame_yuv420_tiles;
    let anchor = textured(11, 128, 128);
    let cam = camera_view(&anchor);
    let e = encode_camera_frame_yuv420_tiles(&cam, &anchor, 60, 1).expect("camera grid encodes");
    assert_eq!(e.coded_tiles.len(), 4, "2x2 grid");
    let fh_ti = e.fh.tile_info.as_ref().expect("camera header tile info");
    assert_eq!((fh_ti.tile_cols, fh_ti.tile_rows), (2, 2));
    let anchor_frames = vec![anchor_spec_frame(&anchor)];
    // One entry per grid tile at its own output position (tile-scan
    // order: row-major).
    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 1,
        output_frame_height_in_tiles_minus_1: 1,
        entries: (0..4)
            .map(|i| TileListEntry {
                anchor_frame_idx: 0,
                anchor_tile_row: (i / 2) as u8,
                anchor_tile_col: (i % 2) as u8,
                coded_tile_data: e.coded_tiles[i].clone(),
            })
            .collect(),
    };
    let out = decode_tile_list(&e.seq, &e.fh, &anchor_frames, &tl).expect("grid decodes");
    assert_eq!((out.width, out.height), (128, 128));
    assert_eq!(out.planes[0], e.recon.y, "grid luma");
    assert_eq!(out.planes[1], e.recon.u, "grid U");
    assert_eq!(out.planes[2], e.recon.v, "grid V");
}

/// The lossless 2-row arm: a 64×128 camera frame (single column, two
/// tile rows) at `q = 0` reproduces the source exactly through the
/// per-row tile-list assembly.
#[test]
fn lossless_camera_tile_rows_reproduce_the_source() {
    use oxideav_av1::encoder::encode_camera_frame_yuv420_tiles;
    let anchor = textured(13, 64, 128);
    let cam = camera_view(&anchor);
    let e = encode_camera_frame_yuv420_tiles(&cam, &anchor, 0, 0).expect("camera rows encode");
    assert_eq!(e.coded_tiles.len(), 2, "two tile rows");
    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 1,
        entries: (0..2)
            .map(|r| TileListEntry {
                anchor_frame_idx: 0,
                anchor_tile_row: r as u8,
                anchor_tile_col: 0,
                coded_tile_data: e.coded_tiles[r].clone(),
            })
            .collect(),
    };
    let out = decode_tile_list(&e.seq, &e.fh, &[anchor_spec_frame(&anchor)], &tl).expect("decodes");
    assert_eq!((out.width, out.height), (64, 128));
    assert_eq!(out.planes[0], cam.y, "lossless rows: luma");
    assert_eq!(out.planes[1], cam.u, "lossless rows: U");
    assert_eq!(out.planes[2], cam.v, "lossless rows: V");
}

/// r450 — the §5.9.12 QUANTIZER-MATRIX election on the §7.3 camera
/// arm: §7.3.1's constraint list never bars `using_qmatrix`, so a
/// textured camera frame may elect the §9.5.3 dequant tables; the
/// §7.3.2 camera-tile decode dequantizes through the same frame
/// header and must reproduce the encoder reconstruction
/// byte-for-byte. The `qm = false` twin stays bit-identical to the
/// historical flat-quantiser shape.
#[test]
fn qm_camera_frame_elects_and_reassembles() {
    use oxideav_av1::encoder::encode_camera_frame_yuv420_tiles_qm;
    // "Natural" texture (smooth waves + per-frame re-rolled grain,
    // the measured QM-election regime from `tests/qm_ab.rs`): the
    // anchor is the t = 0 field, the view the t = 1 field — the
    // camera pair motion-compensates the waves but the re-rolled
    // grain leaves a textured residual a §9.5.3 matrix can reshape.
    // (The plain `textured`/`camera_view` shift pair's residual is
    // near-zero — the election correctly stays off there, which the
    // ladder's non-elected assert also witnesses.)
    let natural = |t: usize| -> Yuv420Frame {
        let (w, h) = (128u32, 64u32);
        let (wu, hu) = (w as usize, h as usize);
        let mut f = Yuv420Frame::filled(w, h, 128);
        let mut lcg = 0x2545_f491u32.wrapping_add((t as u32).wrapping_mul(0x9e37_79b9));
        let mut noise = vec![0i32; (wu + 32) * (hu + 32)];
        for n in noise.iter_mut() {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *n = ((lcg >> 24) as i32 & 15) - 8;
        }
        for r in 0..hu {
            for c in 0..wu {
                let (sr, sc) = (r as f64, (c + 2 * t) as f64);
                let base = 128.0
                    + 46.0 * (0.041 * sr).sin() * (0.057 * sc).cos()
                    + 22.0 * (0.013 * (sr + sc)).sin();
                let g = noise[(r + t) * (wu + 32) + c + 2 * t];
                f.y[r * wu + c] = (base as i32 + g).clamp(0, 255) as u8;
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        for r in 0..ch {
            for c in 0..cw {
                f.u[r * cw + c] = (120 + ((r + c + t) % 17)) as u8;
                f.v[r * cw + c] = (132 + ((2 * r + c + t) % 13)) as u8;
            }
        }
        f
    };
    let anchor = natural(0);
    let view = natural(1);
    // The election is content/q-dependent; walk a small q ladder and
    // require the arm to fire at least once (measured: it does).
    let mut elected = None;
    for &q in &[100u8, 140, 180] {
        let on = encode_camera_frame_yuv420_tiles_qm(&view, &anchor, q, 0, true)
            .expect("camera encode (qm armed)");
        let off = encode_camera_frame_yuv420_tiles_qm(&view, &anchor, q, 0, false)
            .expect("camera encode (baseline)");
        let off2 = oxideav_av1::encoder::encode_camera_frame_yuv420(&view, &anchor, q)
            .expect("historical entry");
        assert_eq!(
            off.coded_tile_data, off2.coded_tile_data,
            "q{q}: qm=false twin must stay bit-identical to the historical shape"
        );
        let qp = on
            .fh
            .quantization_params
            .as_ref()
            .expect("camera header carries q params");
        if qp.using_qmatrix {
            elected = Some((q, on));
            break;
        }
        assert_eq!(
            on.coded_tile_data, off.coded_tile_data,
            "q{q}: a non-elected qm arm must keep the baseline bytes"
        );
    }
    let (q, e) = elected.expect("textured content must elect QM somewhere on the ladder");
    // §7.3.2 reassembly at the elected header: one whole-frame tile
    // (128×64 is a single 2-SB-wide, 1-SB-high §7.3.1 tile row of
    // two tiles? no — tile_cols_log2 = 0 keeps ONE 128-wide tile).
    let tl = TileListObu {
        output_frame_width_in_tiles_minus_1: 0,
        output_frame_height_in_tiles_minus_1: 0,
        entries: vec![TileListEntry {
            anchor_frame_idx: 0,
            anchor_tile_row: 0,
            anchor_tile_col: 0,
            coded_tile_data: e.coded_tile_data.clone(),
        }],
    };
    let out = decode_tile_list(&e.seq, &e.fh, &[anchor_spec_frame(&anchor)], &tl)
        .unwrap_or_else(|err| panic!("q{q}: QM camera tile decodes: {err:?}"));
    assert_eq!(out.planes[0], e.recon.y, "q{q}: luma");
    assert_eq!(out.planes[1], e.recon.u, "q{q}: U");
    assert_eq!(out.planes[2], e.recon.v, "q{q}: V");
}
