//! r456 — SPATIAL × TEMPORAL scalability: independently coded spatial
//! layers, each carrying the dyadic temporal ladder, behind §5.3.3
//! `(temporal_id, spatial_id)` extension headers and an `S × T`
//! §6.7.5 operating-point list. Decoding at every point yields
//! exactly the shown frames of the selected spatial prefix at the
//! selected temporal prefix, each byte-identical to the layer's
//! encoder reconstruction (§5.3.1 `drop_obu`).
//!
//! Spec: docs/video/av1/av1-spec.txt §5.3.1, §5.3.3, §5.5.1, §5.9.2,
//! §6.7.5, §7.5, §7.20.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_spatial_layered_gop_yuv420_with_q, encode_spatial_temporal_layered_gop_yuv420_with_q,
    temporal_layer_of, SpatialLayeredGop, Yuv420Frame,
};
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

fn two_layers(n: usize) -> Vec<Vec<Yuv420Frame>> {
    vec![
        (0..n).map(|t| moving(64, 64, t, 1)).collect(),
        (0..n).map(|t| moving(128, 128, t, 2)).collect(),
    ]
}

fn spec_planes(f: &Frame) -> (&Vec<Vec<u8>>, u32, u32) {
    match f {
        Frame::Spec(s) => (&s.planes, s.width, s.height),
        other => panic!("non-Spec frame {other:?}"),
    }
}

/// Decode at every §6.7.5 point and match the expected
/// (instant-major, layer-minor) frame list of that point.
fn assert_every_operating_point(enc: &SpatialLayeredGop, s_count: usize, t_count: u8, n: usize) {
    assert_eq!(enc.operating_points.len(), s_count * usize::from(t_count));
    assert_eq!(enc.operating_points[0], (s_count as u8, t_count));
    assert_eq!(enc.temporal_ids.len(), n);
    for (i, &tid) in enc.temporal_ids.iter().enumerate() {
        assert_eq!(
            tid,
            temporal_layer_of(i, t_count),
            "instant {i} temporal id"
        );
    }
    for (p, &(sc, tc)) in enc.operating_points.iter().enumerate() {
        let out = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, p as u8)
            .unwrap_or_else(|e| panic!("op {p} ({sc}x{tc}) decode: {e:?}"));
        let mut expected: Vec<(usize, usize)> = Vec::new();
        for i in 0..n {
            if enc.temporal_ids[i] < tc {
                for s in 0..usize::from(sc) {
                    expected.push((i, s));
                }
            }
        }
        assert_eq!(out.len(), expected.len(), "op {p} ({sc}x{tc}): frame count");
        for (f, &(i, s)) in out.iter().zip(&expected) {
            let (planes, w, h) = spec_planes(f);
            assert_eq!(
                (w, h),
                enc.layer_dims[s],
                "op {p}: instant {i} layer {s} dims"
            );
            let rc = &enc.layer_recons[s][i];
            assert_eq!(planes[0], rc.y, "op {p}: instant {i} layer {s} luma");
            assert_eq!(planes[1], rc.u, "op {p}: instant {i} layer {s} U");
            assert_eq!(planes[2], rc.v, "op {p}: instant {i} layer {s} V");
        }
    }
}

/// Every frame OBU carries `(temporal_id = tid(i), spatial_id = s)`;
/// the sequence header's `operating_point_idc` list is the
/// `(spatial mask << 8) | temporal mask` product.
fn assert_wire_shape(enc: &SpatialLayeredGop, s_count: usize, t_count: u8) {
    for (i, tu) in enc.temporal_units.iter().enumerate() {
        let mut sids: Vec<u8> = Vec::new();
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    let sh = parse_sequence_header(desc.payload).expect("SH parses");
                    assert_eq!(
                        usize::from(sh.operating_points_cnt_minus_1) + 1,
                        s_count * usize::from(t_count)
                    );
                    for (op, &(sc, tc)) in sh.operating_points.iter().zip(&enc.operating_points) {
                        assert_eq!(
                            op.operating_point_idc,
                            (((1u16 << sc) - 1) << 8) | ((1u16 << tc) - 1),
                            "operating point ({sc}x{tc}) idc"
                        );
                    }
                }
                ObuType::TemporalDelimiter => {}
                _ => {
                    assert!(desc.extension_flag, "unit {i}: frame OBUs carry extensions");
                    assert_eq!(
                        desc.temporal_id, enc.temporal_ids[i],
                        "unit {i} temporal id"
                    );
                    sids.push(desc.spatial_id);
                }
            }
        }
        assert_eq!(
            sids,
            (0..s_count as u8).collect::<Vec<_>>(),
            "unit {i} spatial order"
        );
    }
}

fn dump(enc: &SpatialLayeredGop, name: &str) {
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_SVC_ST_DUMP") {
        std::fs::create_dir_all(&dir).expect("dump dir");
        std::fs::write(format!("{dir}/{name}.ivf"), &enc.ivf_bytes).expect("ivf dump");
        for (p, &(sc, tc)) in enc.operating_points.iter().enumerate() {
            let mut yuv = Vec::new();
            for (i, &tid) in enc.temporal_ids.iter().enumerate() {
                if tid < tc {
                    for s in 0..usize::from(sc) {
                        let rc = &enc.layer_recons[s][i];
                        yuv.extend_from_slice(&rc.y);
                        yuv.extend_from_slice(&rc.u);
                        yuv.extend_from_slice(&rc.v);
                    }
                }
            }
            std::fs::write(format!("{dir}/{name}.op{p}.yuv"), yuv).expect("yuv dump");
        }
        eprintln!(
            "{name}: {} bytes, ops {:?}",
            enc.ivf_bytes.len(),
            enc.operating_points
        );
    }
}

/// Two spatial × two temporal layers: four operating points.
#[test]
fn two_by_two_ladder_decodes_at_every_operating_point() {
    let n = 5;
    let enc = encode_spatial_temporal_layered_gop_yuv420_with_q(&two_layers(n), 84, 2)
        .expect("S2T2 encode");
    assert_wire_shape(&enc, 2, 2);
    assert_every_operating_point(&enc, 2, 2, n);
    dump(&enc, "self-svc-st-64-128-q84-t2");
}

/// Two spatial × three temporal layers (the `0 2 1 2 0` ladder): six
/// operating points, the deepest per-layer slot budget in use.
#[test]
fn two_by_three_ladder_decodes_at_every_operating_point() {
    let n = 6;
    let enc = encode_spatial_temporal_layered_gop_yuv420_with_q(&two_layers(n), 72, 3)
        .expect("S2T3 encode");
    assert_wire_shape(&enc, 2, 3);
    assert_every_operating_point(&enc, 2, 3, n);
    dump(&enc, "self-svc-st-64-128-q72-t3");
}

/// Three spatial × two temporal layers (a two-slot budget per layer).
#[test]
fn three_by_two_ladder_decodes_at_every_operating_point() {
    let n = 4;
    let layers = vec![
        (0..n).map(|t| moving(64, 64, t, 1)).collect::<Vec<_>>(),
        (0..n).map(|t| moving(128, 64, t, 2)).collect(),
        (0..n).map(|t| moving(128, 128, t, 3)).collect(),
    ];
    let enc =
        encode_spatial_temporal_layered_gop_yuv420_with_q(&layers, 84, 2).expect("S3T2 encode");
    assert_wire_shape(&enc, 3, 2);
    assert_every_operating_point(&enc, 3, 2, n);
}

/// The pure spatial shape is untouched: the generalized core reports
/// all-zero temporal ids and the `(S - k, 1)` point list, and rejects
/// ladders deeper than the per-layer slot budget.
#[test]
fn spatial_only_shape_and_rejects() {
    let enc = encode_spatial_layered_gop_yuv420_with_q(&two_layers(3), 84).expect("spatial");
    assert_eq!(enc.temporal_ids, vec![0, 0, 0]);
    assert_eq!(enc.operating_points, vec![(2, 1), (1, 1)]);
    assert!(encode_spatial_temporal_layered_gop_yuv420_with_q(&two_layers(3), 84, 1).is_err());
    assert!(encode_spatial_temporal_layered_gop_yuv420_with_q(&two_layers(3), 84, 5).is_err());
    let four = vec![
        (0..3).map(|t| moving(64, 64, t, 1)).collect::<Vec<_>>(),
        (0..3).map(|t| moving(64, 64, t, 2)).collect(),
        (0..3).map(|t| moving(128, 64, t, 3)).collect(),
        (0..3).map(|t| moving(128, 128, t, 4)).collect(),
    ];
    // Four spatial layers leave two slots each: three temporal layers
    // fit, four do not.
    assert!(encode_spatial_temporal_layered_gop_yuv420_with_q(&four, 84, 4).is_err());
}
