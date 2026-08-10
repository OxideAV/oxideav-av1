//! r431 — spatially scalable streams: independently-coded spatial
//! layers behind §5.3.3 `spatial_id` extension headers and a nested
//! §6.7.5 operating-point list.
//!
//! What these tests pin:
//!
//!   * Operating-point semantics — decoding at point `k` yields
//!     exactly the shown frames of spatial layers `0..=S-1-k`
//!     (§5.3.1 `drop_obu`), each byte-identical to its layer's
//!     encoder reconstruction: dropping any spatial-layer suffix
//!     leaves every surviving frame's references AND its §8.3.1
//!     primary-reference CDF chain intact (per-layer §7.20 slot
//!     pairs).
//!   * Header shape — one KEY frame (layer 0, `allFrames` refresh),
//!     §5.9.2 `INTRA_ONLY` openers on the enhancement layers
//!     (explicit non-`allFrames` refresh masks), smaller layers on
//!     the §5.9.5 `frame_size_override_flag = 1` arm under the
//!     shared top-layer sequence header.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.3.1, §5.3.3, §5.5.1, §5.9.2,
//! §5.9.5, §5.9.7, §6.7.5, §7.5.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{encode_spatial_layered_gop_yuv420_with_q, Yuv420Frame};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, FrameType, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Moving textured content at any dimension pair (`t` shifts phase).
fn moving(w: u32, h: u32, t: usize, seed: u32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for r in 0..hu {
        for c in 0..wu {
            let v = (r * 3
                + (c + 4 * t) * 5
                + (seed as usize) * 11
                + (r / 8) * (c / 8) * (1 + seed as usize % 3))
                % 256;
            f.y[r * wu + c] = v as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = ((120 + r + c + 2 * t) % 256) as u8;
            f.v[r * cw + c] = ((70 + r * 2 + c + t) % 256) as u8;
        }
    }
    f
}

fn two_layers(n: usize, q_seedless: bool) -> Vec<Vec<Yuv420Frame>> {
    let seed = u32::from(!q_seedless);
    vec![
        (0..n).map(|t| moving(64, 64, t, seed)).collect(),
        (0..n).map(|t| moving(128, 128, t, seed + 1)).collect(),
    ]
}

fn spec_planes(f: &Frame) -> (&Vec<Vec<u8>>, u32, u32) {
    match f {
        Frame::Spec(s) => (&s.planes, s.width, s.height),
        #[allow(unreachable_patterns)]
        other => panic!("non-Spec frame variant {other:?}"),
    }
}

/// Decoding operating point 0 (all layers) yields the interleaved
/// per-instant frames — layer 0 then layer 1 per temporal unit —
/// each byte-identical to its layer's reconstruction; point 1 (base
/// layer only) yields exactly layer 0's frames.
#[test]
fn spatial_two_layer_operating_points_decode_bit_exact() {
    let layers = two_layers(3, true);
    let enc = encode_spatial_layered_gop_yuv420_with_q(&layers, 72).expect("spatial encode");
    assert_eq!(enc.layer_dims, vec![(64, 64), (128, 128)]);

    // Operating point 0 — both layers, 6 shown frames interleaved.
    let full =
        oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 0).expect("full-stream decode");
    assert_eq!(full.len(), 6, "3 instants x 2 layers");
    for i in 0..3 {
        for s in 0..2 {
            let (planes, w, h) = spec_planes(&full[i * 2 + s]);
            let (ew, eh) = enc.layer_dims[s];
            assert_eq!((w, h), (ew, eh), "instant {i} layer {s} dims");
            let rc = &enc.layer_recons[s][i];
            assert_eq!(planes[0], rc.y, "instant {i} layer {s} luma");
            assert_eq!(planes[1], rc.u, "instant {i} layer {s} U");
            assert_eq!(planes[2], rc.v, "instant {i} layer {s} V");
        }
    }

    // Operating point 1 — base layer alone, bit-identical frames.
    let base =
        oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 1).expect("base-layer decode");
    assert_eq!(base.len(), 3);
    for (i, f) in base.iter().enumerate() {
        let (planes, w, h) = spec_planes(f);
        assert_eq!((w, h), (64, 64));
        let rc = &enc.layer_recons[0][i];
        assert_eq!(planes[0], rc.y, "base instant {i} luma");
        assert_eq!(planes[1], rc.u, "base instant {i} U");
        assert_eq!(planes[2], rc.v, "base instant {i} V");
    }
}

/// Three layers: the §6.7.5 masks are the nested spatial prefixes
/// (`0x701 / 0x301 / 0x101`), and every operating point decodes its
/// layer subset bit-exactly.
#[test]
fn spatial_three_layer_nested_points() {
    let layers = vec![
        (0..2).map(|t| moving(64, 64, t, 3)).collect::<Vec<_>>(),
        (0..2).map(|t| moving(128, 64, t, 4)).collect::<Vec<_>>(),
        (0..2).map(|t| moving(128, 128, t, 5)).collect::<Vec<_>>(),
    ];
    let enc = encode_spatial_layered_gop_yuv420_with_q(&layers, 100).expect("3-layer encode");
    let idcs: Vec<u16> = enc
        .seq
        .operating_points
        .iter()
        .map(|o| o.operating_point_idc)
        .collect();
    assert_eq!(idcs, vec![0x701, 0x301, 0x101]);
    for k in 0..3usize {
        let live = 3 - k;
        let out = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, k as u8)
            .unwrap_or_else(|e| panic!("op {k} decode: {e:?}"));
        assert_eq!(out.len(), 2 * live, "op {k} shown frames");
        for i in 0..2 {
            for s in 0..live {
                let (planes, w, h) = spec_planes(&out[i * live + s]);
                assert_eq!((w, h), enc.layer_dims[s], "op {k} instant {i} layer {s}");
                let rc = &enc.layer_recons[s][i];
                assert_eq!(planes[0], rc.y, "op {k} instant {i} layer {s} luma");
                assert_eq!(planes[1], rc.u, "op {k} instant {i} layer {s} U");
                assert_eq!(planes[2], rc.v, "op {k} instant {i} layer {s} V");
            }
        }
    }
}

/// Header-shape audit on the wire: unit 0 carries the shared SH +
/// one KEY (layer 0) + one INTRA_ONLY per enhancement layer (with
/// the layer's own slot-pair refresh mask); the smaller layer's
/// frames ride `frame_size_override_flag = 1`; every frame OBU
/// carries its `spatial_id`.
#[test]
fn spatial_stream_header_shapes() {
    let layers = two_layers(2, false);
    let enc = encode_spatial_layered_gop_yuv420_with_q(&layers, 72).expect("spatial encode");

    let seq = {
        let mut found = None;
        for desc in ObuIter::new(&enc.temporal_units[0]) {
            let desc = desc.expect("TU walks");
            if desc.obu_type == ObuType::SequenceHeader {
                found = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
        }
        found.expect("unit 0 carries the SH")
    };
    assert_eq!(seq.max_frame_width_minus_1 + 1, 128);
    assert_eq!(seq.max_frame_height_minus_1 + 1, 128);
    assert_eq!(seq.operating_points_cnt_minus_1, 1);
    assert_eq!(seq.operating_points[0].operating_point_idc, 0x301);
    assert_eq!(seq.operating_points[1].operating_point_idc, 0x101);

    // Unit 0: layer-0 KEY + layer-1 INTRA_ONLY.
    let mut frame_idx = 0usize;
    for desc in ObuIter::new(&enc.temporal_units[0]) {
        let desc = desc.expect("TU walks");
        if desc.obu_type != ObuType::Frame {
            continue;
        }
        assert!(desc.extension_flag, "layered frame OBUs carry extensions");
        assert_eq!(usize::from(desc.spatial_id), frame_idx, "spatial order");
        assert_eq!(desc.temporal_id, 0);
        let fh = parse_frame_header_with_refs(desc.payload, &seq, &RefInfo::default())
            .expect("frame header parses");
        match frame_idx {
            0 => {
                assert_eq!(fh.frame_type, FrameType::Key);
                // 64x64 under a 128x128 budget: explicit dimensions.
                assert!(fh.frame_size_override_flag);
                let fs = fh.frame_size.expect("sized");
                assert_eq!((fs.frame_width, fs.frame_height), (64, 64));
            }
            1 => {
                assert_eq!(fh.frame_type, FrameType::IntraOnly);
                assert_eq!(fh.refresh_frame_flags, 0b11 << 2);
                assert!(!fh.frame_size_override_flag, "top layer = budget dims");
                assert!(!fh.error_resilient_mode);
            }
            _ => panic!("unexpected third frame OBU in unit 0"),
        }
        frame_idx += 1;
    }
    assert_eq!(frame_idx, 2);
}

/// Input validation: layer counts, mismatched lengths, a layer
/// exceeding the top layer's budget.
#[test]
fn spatial_input_rejects() {
    let l64: Vec<Yuv420Frame> = (0..2).map(|t| moving(64, 64, t, 7)).collect();
    let l128: Vec<Yuv420Frame> = (0..2).map(|t| moving(128, 128, t, 8)).collect();
    // One layer is not a scalable stream.
    assert!(encode_spatial_layered_gop_yuv420_with_q(std::slice::from_ref(&l64), 72).is_err());
    // Unequal lengths.
    assert!(
        encode_spatial_layered_gop_yuv420_with_q(&[l64.clone(), l128[..1].to_vec()], 72).is_err()
    );
    // Base larger than the top layer's budget.
    assert!(encode_spatial_layered_gop_yuv420_with_q(&[l128, l64], 72).is_err());
}

/// Env-gated staging dump (`OXIDEAV_AV1_SVC_DIR`): the two-layer
/// stream + per-layer expected YUV for black-box reference-decoder
/// validation at each operating point.
#[test]
fn spatial_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SVC_DIR") else {
        eprintln!("OXIDEAV_AV1_SVC_DIR unset — skipping the spatial staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let layers = two_layers(4, true);
    let enc = encode_spatial_layered_gop_yuv420_with_q(&layers, 72).expect("spatial encode");
    let name = "svc-s2-64-128-q72";
    std::fs::write(root.join(format!("{name}.ivf")), &enc.ivf_bytes).expect("write ivf");
    for (s, lr) in enc.layer_recons.iter().enumerate() {
        let mut yuv: Vec<u8> = Vec::new();
        for rc in lr {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join(format!("{name}.layer{s}.yuv")), &yuv).expect("write yuv");
    }
    // The interleaved full-stream expectation (decode order).
    let mut full: Vec<u8> = Vec::new();
    for i in 0..4 {
        for s in 0..2 {
            let rc = &enc.layer_recons[s][i];
            full.extend_from_slice(&rc.y);
            full.extend_from_slice(&rc.u);
            full.extend_from_slice(&rc.v);
        }
    }
    std::fs::write(root.join(format!("{name}.full.yuv")), &full).expect("write yuv");
}

// ---------------------------------------------------------------------
// r436 — PER-LAYER tile layouts + tile-group packaging.
// ---------------------------------------------------------------------

use oxideav_av1::encoder::encode_spatial_layered_gop_yuv420_with_q_tiles;

/// Per-spatial-layer §5.9.15 uniform layouts: a `(1, 0)` two-column
/// base layer at 128×64 under a `(2, 1)` eight-tile enhancement
/// layer at 256×128 — every frame of each layer codes its OWN
/// layout, and both operating points still decode bit-exactly.
#[test]
fn spatial_per_layer_tiles_decode_bit_exact() {
    let layers = vec![
        (0..3).map(|t| moving(128, 64, t, 11)).collect::<Vec<_>>(),
        (0..3).map(|t| moving(256, 128, t, 12)).collect::<Vec<_>>(),
    ];
    let enc =
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 84, Some(&[(1, 0), (2, 1)]), 1)
            .expect("per-layer tiled spatial encode");
    assert_eq!(enc.layer_dims, vec![(128, 64), (256, 128)]);

    // Wire audit on unit 0: the KEY (layer 0) codes 2×1 tiles, the
    // INTRA_ONLY (layer 1) 4×2 — each layer its own layout.
    let seq = {
        let mut found = None;
        for desc in ObuIter::new(&enc.temporal_units[0]) {
            let desc = desc.expect("TU walks");
            if desc.obu_type == ObuType::SequenceHeader {
                found = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
        }
        found.expect("unit 0 carries the SH")
    };
    let mut expect = [(2u32, 1u32), (4, 2)].iter();
    for desc in ObuIter::new(&enc.temporal_units[0]) {
        let desc = desc.expect("TU walks");
        if desc.obu_type != ObuType::Frame {
            continue;
        }
        let fh = parse_frame_header_with_refs(desc.payload, &seq, &RefInfo::default())
            .expect("frame header parses");
        let ti = fh.tile_info.expect("tile info coded");
        let &(ec, er) = expect.next().expect("two frames in unit 0");
        assert_eq!(
            (ti.tile_cols, ti.tile_rows),
            (ec, er),
            "layer {} layout",
            desc.spatial_id
        );
    }
    assert!(expect.next().is_none(), "both unit-0 frames audited");

    // Operating point 0 — both layers, bit-exact interleave.
    let full =
        oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 0).expect("full-stream decode");
    assert_eq!(full.len(), 6);
    for i in 0..3 {
        for s in 0..2 {
            let (planes, w, h) = spec_planes(&full[i * 2 + s]);
            assert_eq!((w, h), enc.layer_dims[s], "instant {i} layer {s} dims");
            let rc = &enc.layer_recons[s][i];
            assert_eq!(planes[0], rc.y, "instant {i} layer {s} luma");
            assert_eq!(planes[1], rc.u, "instant {i} layer {s} U");
            assert_eq!(planes[2], rc.v, "instant {i} layer {s} V");
        }
    }
    // Operating point 1 — the tiled base layer alone.
    let base =
        oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 1).expect("base-layer decode");
    assert_eq!(base.len(), 3);
    for (i, f) in base.iter().enumerate() {
        let (planes, ..) = spec_planes(f);
        let rc = &enc.layer_recons[0][i];
        assert_eq!(planes[0], rc.y, "base instant {i} luma");
        assert_eq!(planes[1], rc.u, "base instant {i} U");
        assert_eq!(planes[2], rc.v, "base instant {i} V");
    }
}

/// The §5.9.15 legality window is PER LAYER: a layout the small base
/// layer cannot express rejects even though the enhancement layer
/// could code it — and the same layout on the enhancement layer
/// alone is accepted.
#[test]
fn spatial_per_layer_tile_legality_windows() {
    let layers = vec![
        (0..2).map(|t| moving(64, 64, t, 13)).collect::<Vec<_>>(),
        (0..2).map(|t| moving(128, 128, t, 14)).collect::<Vec<_>>(),
    ];
    // (1, 0) needs two superblock columns — the 64×64 base has one.
    assert!(
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 72, Some(&[(1, 0), (0, 0)]), 1)
            .is_err(),
        "base-layer window must reject (1, 0) at 64×64"
    );
    // The same layout is legal at the 128×128 enhancement layer.
    let enc =
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 72, Some(&[(0, 0), (1, 0)]), 1)
            .expect("enhancement-layer (1, 0) encodes");
    let full = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 0).expect("decode");
    assert_eq!(full.len(), 4);
    // Mismatched per-layer list length rejects.
    assert!(
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 72, Some(&[(0, 0)]), 1).is_err()
    );
}

/// `layer_tiles = None` / all-`(0, 0)` + `tile_groups <= 1`
/// reproduce the r431 untiled spatial stream BIT FOR BIT.
#[test]
fn spatial_tiles_default_reproduces_untiled_stream() {
    let layers = two_layers(2, false);
    let plain = encode_spatial_layered_gop_yuv420_with_q(&layers, 72).expect("plain encode");
    let zeroed =
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 72, Some(&[(0, 0), (0, 0)]), 1)
            .expect("zero-layout encode");
    assert_eq!(plain.ivf_bytes, zeroed.ivf_bytes, "all-(0,0) layouts");
    let grouped = encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 72, None, 0)
        .expect("groups=0 encode");
    assert_eq!(plain.ivf_bytes, grouped.ivf_bytes, "tile_groups clamp");
}

/// §5.11.1 tile-group packaging under §5.3.3 extension headers: the
/// four-tile enhancement layer splits into `OBU_FRAME_HEADER` + two
/// `OBU_TILE_GROUP` OBUs (every one carrying `spatial_id = 1`), the
/// single-tile base layer keeps its `OBU_FRAME` — and the stream
/// still decodes bit-exactly at both operating points.
#[test]
fn spatial_tile_groups_split_framing() {
    let layers = vec![
        (0..2).map(|t| moving(64, 64, t, 15)).collect::<Vec<_>>(),
        (0..2).map(|t| moving(128, 128, t, 16)).collect::<Vec<_>>(),
    ];
    let enc =
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 76, Some(&[(0, 0), (1, 1)]), 2)
            .expect("grouped spatial encode");
    for (u, tu) in enc.temporal_units.iter().enumerate() {
        let mut shapes: Vec<(ObuType, u8)> = Vec::new();
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::TemporalDelimiter | ObuType::SequenceHeader => {}
                t => {
                    assert!(desc.extension_flag, "unit {u}: frame OBUs carry extensions");
                    shapes.push((t, desc.spatial_id));
                }
            }
        }
        assert_eq!(
            shapes,
            vec![
                (ObuType::Frame, 0),
                (ObuType::FrameHeader, 1),
                (ObuType::TileGroup, 1),
                (ObuType::TileGroup, 1),
            ],
            "unit {u} OBU shapes"
        );
    }
    let full = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 0).expect("decode");
    assert_eq!(full.len(), 4);
    for i in 0..2 {
        for s in 0..2 {
            let (planes, ..) = spec_planes(&full[i * 2 + s]);
            let rc = &enc.layer_recons[s][i];
            assert_eq!(planes[0], rc.y, "instant {i} layer {s} luma");
            assert_eq!(planes[1], rc.u, "instant {i} layer {s} U");
            assert_eq!(planes[2], rc.v, "instant {i} layer {s} V");
        }
    }
    let base = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 1).expect("base decode");
    assert_eq!(base.len(), 2);
}

/// Env-gated staging dump (`OXIDEAV_AV1_SVC_TILES_DIR`): the
/// per-layer-tiled + tile-group-split spatial stream for black-box
/// reference-decoder validation and corpus pinning.
#[test]
fn spatial_per_layer_tiles_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SVC_TILES_DIR") else {
        eprintln!("OXIDEAV_AV1_SVC_TILES_DIR unset — skipping the tiled spatial staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let layers = vec![
        (0..4).map(|t| moving(128, 64, t, 11)).collect::<Vec<_>>(),
        (0..4).map(|t| moving(256, 128, t, 12)).collect::<Vec<_>>(),
    ];
    let enc =
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 84, Some(&[(1, 0), (2, 1)]), 2)
            .expect("per-layer tiled spatial encode");
    let name = "svc-s2-tiles-128-256-q84";
    std::fs::write(root.join(format!("{name}.ivf")), &enc.ivf_bytes).expect("write ivf");
    for (s, lr) in enc.layer_recons.iter().enumerate() {
        let mut yuv: Vec<u8> = Vec::new();
        for rc in lr {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join(format!("{name}.layer{s}.yuv")), &yuv).expect("write yuv");
    }
    let mut full: Vec<u8> = Vec::new();
    for i in 0..4 {
        for s in 0..2 {
            let rc = &enc.layer_recons[s][i];
            full.extend_from_slice(&rc.y);
            full.extend_from_slice(&rc.u);
            full.extend_from_slice(&rc.v);
        }
    }
    std::fs::write(root.join(format!("{name}.full.yuv")), &full).expect("write yuv");
}

// ---------------------------------------------------------------------
// r439 — the §6.8.14 `context_update_tile_id` election on the SVC
// driver.
// ---------------------------------------------------------------------

/// Left half flat, right half textured + moving (the tile-1 CDF
/// donation prices consumers smaller under a `(1, 0)` layout).
fn hetero(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    for r in 0..hu {
        for c in 0..wu {
            let v = if c < wu / 2 {
                96 + ((r / 16) % 2) * 4
            } else {
                (r * 7 + (c + 5 * t) * 13 + (r % 5) * (c % 7) * 3) % 256
            };
            f.y[r * wu + c] = v as u8;
        }
    }
    f
}

/// Per-layer donor elections on a two-layer tiled SVC stream: at
/// least one frame's `context_update_tile_id` is patched off 0 on
/// the wire, and BOTH §6.7.5 operating points decode bit-exact —
/// dropping the enhancement layer leaves every base-layer frame's
/// patched donation intact (per-layer §7.20 slot pairs, per-layer
/// freeze-at-first-consumption).
#[test]
fn svc_ctx_update_election_fires_and_operating_points_decode_bit_exact() {
    let layers = vec![
        (0..5).map(|t| hetero(128, 64, t)).collect::<Vec<_>>(),
        (0..5).map(|t| hetero(256, 128, t)).collect::<Vec<_>>(),
    ];
    let enc =
        encode_spatial_layered_gop_yuv420_with_q_tiles(&layers, 80, Some(&[(1, 0), (1, 0)]), 1)
            .expect("tiled spatial encode");

    // Wire audit: walk every coded frame header of every unit.
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    for i in 0..8 {
        refinfo.valid[i] = true;
        refinfo.upscaled_width[i] = 256;
        refinfo.frame_height[i] = 128;
        refinfo.render_width[i] = 256;
        refinfo.render_height[i] = 128;
    }
    let mut ids = Vec::new();
    for tu in &enc.temporal_units {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let fh = parse_frame_header_with_refs(
                        desc.payload,
                        seq.as_ref().expect("SH precedes frames"),
                        &refinfo,
                    )
                    .expect("frame header parses");
                    if fh.show_existing_frame {
                        continue;
                    }
                    ids.push(fh.tile_info.expect("tiled stream").context_update_tile_id);
                }
                _ => {}
            }
        }
    }
    assert_eq!(ids.len(), 10, "one coded frame per layer per instant");
    assert!(
        ids.iter().any(|&id| id != 0),
        "designed content must patch at least one SVC donation: {ids:?}"
    );

    // Both operating points decode their layer subsets bit-exact.
    let full = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 0).expect("full decode");
    assert_eq!(full.len(), 10);
    for i in 0..5 {
        for s in 0..2 {
            let (planes, _, _) = spec_planes(&full[i * 2 + s]);
            let rc = &enc.layer_recons[s][i];
            assert_eq!(planes[0], rc.y, "instant {i} layer {s} luma");
            assert_eq!(planes[1], rc.u, "instant {i} layer {s} U");
            assert_eq!(planes[2], rc.v, "instant {i} layer {s} V");
        }
    }
    let base = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 1).expect("base decode");
    assert_eq!(base.len(), 5);
    for (i, f) in base.iter().enumerate() {
        let (planes, _, _) = spec_planes(f);
        let rc = &enc.layer_recons[0][i];
        assert_eq!(planes[0], rc.y, "base instant {i} luma");
        assert_eq!(planes[1], rc.u, "base instant {i} U");
        assert_eq!(planes[2], rc.v, "base instant {i} V");
    }
}
