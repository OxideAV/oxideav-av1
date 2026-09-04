//! r456 — spatial scalability with INTER-LAYER PREDICTION: every
//! enhancement-layer inter frame carries the next-lower layer's
//! reconstruction of the same instant as its GOLDEN reference,
//! predicted through the §7.11.3.3 scaled path in both axes
//! (`is_scaled( GOLDEN_FRAME ) = 1`), alongside its own layer's LAST.
//! Lower layers never reference upper ones, so every §6.7.5 spatial
//! suffix still drops cleanly; decoding at every operating point is
//! byte-identical to the per-layer reconstructions.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.3.3, §5.9.2, §5.11.27, §6.7.5,
//! §6.8.2, §7.11.3.3, §7.20.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_spatial_layered_gop_yuv420_with_q, encode_spatial_layered_gop_yuv420_with_q_inter_layer,
    encode_spatial_temporal_layered_gop_yuv420_with_q, SpatialLayeredGop, Yuv420Frame,
};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, FrameType, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Moving textured content at the enhancement extent.
fn moving(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for r in 0..hu {
        for c in 0..wu {
            let x = c as f64 + 2.0 * t as f64;
            let y = r as f64 + 1.0 * t as f64;
            let v = 128.0
                + 70.0 * (0.05 * x).sin() * (0.07 * y).cos()
                + 30.0 * (0.11 * (x - y)).sin()
                + 12.0 * (0.31 * x).cos();
            f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
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

/// A fresh texture at EVERY instant (frequencies keyed by `t`): the
/// own-layer LAST reference is useless, the same-instant lower layer
/// is the only predictor — the inter-layer arm's win case.
fn scene_cut(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    let (fa, fb, fc) = (
        0.03 + 0.021 * t as f64,
        0.05 + 0.017 * t as f64,
        0.09 + 0.033 * t as f64,
    );
    for r in 0..hu {
        for c in 0..wu {
            let (x, y) = (c as f64, r as f64);
            let v = 128.0
                + 70.0 * (fa * x + 0.7 * t as f64).sin() * (fb * y).cos()
                + 30.0 * (fc * (x - y)).sin();
            f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = ((120 + (r * (t + 1)) % 40 + c) % 256) as u8;
            f.v[r * cw + c] = ((70 + r * 2 + (c * (t + 2)) % 30) % 256) as u8;
        }
    }
    f
}

/// 2×2 box downscale — the base layer is the SAME content at half
/// the extent, so the scaled inter-layer reference is a real
/// predictor.
fn half(f: &Yuv420Frame) -> Yuv420Frame {
    let (w, h) = (f.width as usize, f.height as usize);
    let (hw, hh) = (w / 2, h / 2);
    let mut o = Yuv420Frame::filled(hw as u32, hh as u32, 0);
    for r in 0..hh {
        for c in 0..hw {
            let s = u32::from(f.y[2 * r * w + 2 * c])
                + u32::from(f.y[2 * r * w + 2 * c + 1])
                + u32::from(f.y[(2 * r + 1) * w + 2 * c])
                + u32::from(f.y[(2 * r + 1) * w + 2 * c + 1]);
            o.y[r * hw + c] = ((s + 2) / 4) as u8;
        }
    }
    let (cw, ch) = (w / 2, h / 2);
    let (hcw, hch) = (hw / 2, hh / 2);
    for r in 0..hch {
        for c in 0..hcw {
            for (src, dst) in [(&f.u, &mut o.u), (&f.v, &mut o.v)] {
                let s = u32::from(src[2 * r * cw + 2 * c])
                    + u32::from(src[2 * r * cw + 2 * c + 1])
                    + u32::from(src[(2 * r + 1) * cw + 2 * c])
                    + u32::from(src[(2 * r + 1) * cw + 2 * c + 1]);
                dst[r * hcw + c] = ((s + 2) / 4) as u8;
            }
        }
    }
    let _ = ch;
    o
}

fn pyramid_layers(n: usize, count: usize) -> Vec<Vec<Yuv420Frame>> {
    pyramid_layers_of(n, count, moving)
}

fn pyramid_layers_of(
    n: usize,
    count: usize,
    gen: fn(u32, u32, usize) -> Yuv420Frame,
) -> Vec<Vec<Yuv420Frame>> {
    let top: Vec<Yuv420Frame> = (0..n).map(|t| gen(128, 128, t)).collect();
    let mut layers: Vec<Vec<Yuv420Frame>> = vec![top];
    for _ in 1..count {
        let lower: Vec<Yuv420Frame> = layers[0].iter().map(half).collect();
        layers.insert(0, lower);
    }
    layers
}

fn spec_planes(f: &Frame) -> (&Vec<Vec<u8>>, u32, u32) {
    match f {
        Frame::Spec(s) => (&s.planes, s.width, s.height),
        other => panic!("non-Spec frame {other:?}"),
    }
}

fn assert_every_operating_point(enc: &SpatialLayeredGop, n: usize) {
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
        assert_eq!(out.len(), expected.len(), "op {p}: frame count");
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

/// Every enhancement-layer INTER frame names a slot OUTSIDE its own
/// layer's slot range as GOLDEN (`ref_frame_idx[ 3 ]`) whenever the
/// lower layer's frame of that instant is a reference frame; the
/// header parses against the true per-slot extents (the slot map
/// tracked through `refresh_frame_flags`).
fn assert_inter_layer_wire(enc: &SpatialLayeredGop, s_count: usize, budget: usize) -> usize {
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    let mut inter_layer_frames = 0usize;
    for (i, tu) in enc.temporal_units.iter().enumerate() {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let fh = parse_frame_header_with_refs(
                        desc.payload,
                        seq.as_ref().expect("SH precedes"),
                        &refinfo,
                    )
                    .expect("FH parses");
                    let s = usize::from(desc.spatial_id);
                    let fs = fh.frame_size.as_ref().expect("sized header");
                    assert_eq!((fs.upscaled_width, fs.frame_height), enc.layer_dims[s]);
                    if fh.frame_type == FrameType::Inter && s > 0 {
                        let ir = fh.inter_refs.as_ref().expect("inter refs");
                        let golden = usize::from(ir.ref_frame_idx[3]);
                        let own = budget * s..budget * s + budget;
                        let lower_stored = refinfo.valid.iter().enumerate().any(|(slot, &v)| {
                            v && slot < budget * s && refinfo.order_hint[slot] == fh.order_hint
                        });
                        if lower_stored {
                            assert!(
                                !own.contains(&golden),
                                "unit {i} layer {s}: GOLDEN must name the lower layer's slot"
                            );
                            assert_eq!(refinfo.order_hint[golden], fh.order_hint);
                            assert_eq!(
                                (refinfo.upscaled_width[golden], refinfo.frame_height[golden]),
                                enc.layer_dims[s - 1],
                                "unit {i} layer {s}: GOLDEN sits at the lower layer's extent"
                            );
                            inter_layer_frames += 1;
                        }
                    }
                    for slot in 0..8 {
                        if fh.refresh_frame_flags & (1 << slot) != 0 {
                            refinfo.valid[slot] = true;
                            refinfo.order_hint[slot] = fh.order_hint;
                            refinfo.upscaled_width[slot] = fs.upscaled_width;
                            refinfo.frame_height[slot] = fs.frame_height;
                            refinfo.render_width[slot] = fs.render_width;
                            refinfo.render_height[slot] = fs.render_height;
                            refinfo.frame_type_is_key[slot] = fh.frame_type == FrameType::Key;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let _ = s_count;
    inter_layer_frames
}

fn dump(enc: &SpatialLayeredGop, name: &str) {
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_SVC_IL_DUMP") {
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

/// Two layers (64 → 128), plain temporal shape, smoothly moving
/// content: every enhancement inter frame carries the scaled base
/// reference and both operating points decode bit-exact. (Rate is
/// content-dependent here — the own-layer LAST already predicts a
/// pan near-perfectly, so the extra reference costs its signalling;
/// the win case is below.)
#[test]
fn two_layer_inter_layer_prediction_decodes_at_both_points() {
    let n = 4;
    let layers = pyramid_layers(n, 2);
    let enc = encode_spatial_layered_gop_yuv420_with_q_inter_layer(&layers, 84, 1)
        .expect("inter-layer encode");
    assert_eq!(assert_inter_layer_wire(&enc, 2, 2), n - 1);
    assert_every_operating_point(&enc, n);
    if std::env::var_os("OXIDEAV_AV1_SVC_IL_DUMP").is_some() {
        let plain =
            encode_spatial_layered_gop_yuv420_with_q(&layers, 84).expect("independent encode");
        eprintln!(
            "moving content: inter-layer {} bytes vs independent {} bytes",
            enc.ivf_bytes.len(),
            plain.ivf_bytes.len()
        );
    }
}

/// The win case: a fresh texture at every instant. The own-layer
/// LAST predicts nothing, the same-instant base layer predicts the
/// enhancement through the scaled path — the inter-layer stream
/// must be strictly smaller than the independently coded twin.
#[test]
fn scene_cut_content_wins_through_the_scaled_base_reference() {
    let n = 4;
    let layers = pyramid_layers_of(n, 2, scene_cut);
    let enc = encode_spatial_layered_gop_yuv420_with_q_inter_layer(&layers, 84, 1)
        .expect("inter-layer encode");
    let plain = encode_spatial_layered_gop_yuv420_with_q(&layers, 84).expect("independent encode");
    assert_eq!(assert_inter_layer_wire(&enc, 2, 2), n - 1);
    assert_every_operating_point(&enc, n);
    if std::env::var_os("OXIDEAV_AV1_SVC_IL_DUMP").is_some() {
        eprintln!(
            "scene-cut content: inter-layer {} bytes vs independent {} bytes",
            enc.ivf_bytes.len(),
            plain.ivf_bytes.len()
        );
    }
    assert!(
        enc.ivf_bytes.len() < plain.ivf_bytes.len(),
        "inter-layer prediction must save bytes on scene-cut layers ({} vs {})",
        enc.ivf_bytes.len(),
        plain.ivf_bytes.len()
    );
    dump(&enc, "self-svc-il-64-128-q84-cuts");
}

/// Three layers (32 → 64 → 128): each layer references the NEXT-LOWER
/// one; three operating points decode bit-exact.
#[test]
fn three_layer_chain_references_the_next_lower_layer() {
    let n = 3;
    let layers = pyramid_layers(n, 3);
    let enc = encode_spatial_layered_gop_yuv420_with_q_inter_layer(&layers, 100, 1)
        .expect("3-layer inter-layer encode");
    assert_eq!(assert_inter_layer_wire(&enc, 3, 2), 2 * (n - 1));
    assert_every_operating_point(&enc, n);
}

/// Inter-layer × the temporal ladder (2 × 2): the enhancement frames
/// of reference temporal layers carry the scaled base reference, the
/// top temporal layer predicts within its own layer, all four
/// operating points decode bit-exact.
#[test]
fn inter_layer_with_temporal_ladder_decodes_at_every_point() {
    let n = 5;
    let layers = pyramid_layers(n, 2);
    let enc = encode_spatial_layered_gop_yuv420_with_q_inter_layer(&layers, 84, 2)
        .expect("inter-layer S2T2 encode");
    let independent = encode_spatial_temporal_layered_gop_yuv420_with_q(&layers, 84, 2)
        .expect("independent S2T2 encode");
    assert_eq!(enc.operating_points, independent.operating_points);
    // Instants 2 and 4 (temporal layer 0) carry the inter-layer ref;
    // instants 1 and 3 (top layer, base non-reference) do not.
    assert_eq!(assert_inter_layer_wire(&enc, 2, 4), 2);
    assert_every_operating_point(&enc, n);
    dump(&enc, "self-svc-il-64-128-q84-t2");
}
