//! r430 — the temporally scalable GOP encoder
//! ([`oxideav_av1::encoder::encode_temporal_layered_gop_yuv420_with_q`])
//! decoded at every §6.7.5 operating point: the full stream and each
//! reduced layer set must reproduce the encoder reconstructions of
//! exactly the surviving frames, byte for byte.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.3.1 (drop_obu),
//! §5.3.3, §5.5.1, §6.7.5, §7.5.

use oxideav_av1::decoder::{decode_av1_spec, decode_av1_spec_at_operating_point, SpecFrame};
use oxideav_av1::encoder::{
    encode_temporal_layered_gop_yuv420_with_q, temporal_layer_of, Yuv420Frame,
};
use oxideav_av1::obu::{ObuIter, ObuType};

/// Moving-diagonal source (distinct content per frame so every layer
/// carries real residual).
fn source_frames(n: usize, w: u32, h: u32) -> Vec<Yuv420Frame> {
    (0..n)
        .map(|t| {
            let mut y = vec![0u8; (w * h) as usize];
            for r in 0..h as usize {
                for c in 0..w as usize {
                    let v = (r + 2 * c + 5 * t) as u32;
                    y[r * w as usize + c] = ((v * 7) % 200) as u8 + 20;
                }
            }
            let cw = (w / 2) as usize;
            let ch = (h / 2) as usize;
            let u = vec![(90 + 6 * t) as u8; cw * ch];
            let v = vec![(170u8).wrapping_sub(5 * t as u8); cw * ch];
            Yuv420Frame {
                width: w,
                height: h,
                y,
                u,
                v,
            }
        })
        .collect()
}

/// The encoder reconstruction of display frame `i`, flattened in the
/// decoder's plane order for comparison.
fn recon_planes(gop: &oxideav_av1::encoder::EncodedGop, i: usize) -> Vec<u8> {
    let r = &gop.recon[i];
    let mut out = Vec::with_capacity(r.y.len() + r.u.len() + r.v.len());
    out.extend_from_slice(&r.y);
    out.extend_from_slice(&r.u);
    out.extend_from_slice(&r.v);
    out
}

fn frame_planes(f: &SpecFrame) -> Vec<u8> {
    let mut out = Vec::new();
    for p in &f.planes {
        out.extend_from_slice(p);
    }
    out
}

/// Full end-to-end sweep for one (n, layers, q) configuration.
fn assert_layered_gop_round_trip(n: usize, layers: u8, q: u8) {
    let frames = source_frames(n, 64, 64);
    let enc =
        encode_temporal_layered_gop_yuv420_with_q(&frames, q, layers).expect("layered GOP encodes");
    assert_eq!(enc.temporal_ids.len(), n);
    for (i, &tid) in enc.temporal_ids.iter().enumerate() {
        assert_eq!(tid, temporal_layer_of(i, layers), "ladder at {i}");
    }

    // Sequence header signalling: L operating points, §6.7.5 masks.
    let seq = &enc.gop.seq;
    assert_eq!(seq.operating_points_cnt_minus_1, layers - 1);
    assert_eq!(seq.operating_points.len(), usize::from(layers));
    for (k, op) in seq.operating_points.iter().enumerate() {
        let expect = 0x100 | ((1u16 << (usize::from(layers) - k)) - 1);
        assert_eq!(op.operating_point_idc, expect, "op {k} idc");
    }

    // Wire shape: every frame-carrying OBU carries the §5.3.3
    // extension header with the ladder's temporal_id; TD and SH OBUs
    // stay bare.
    for (i, tu) in enc.gop.temporal_units.iter().enumerate() {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("own stream walks");
            match desc.obu_type {
                ObuType::TemporalDelimiter | ObuType::SequenceHeader => {
                    assert!(!desc.extension_flag, "TU {i}: TD/SH must stay bare");
                }
                _ => {
                    assert!(
                        desc.extension_flag,
                        "TU {i}: frame OBU needs the ext header"
                    );
                    assert_eq!(desc.temporal_id, enc.temporal_ids[i], "TU {i} temporal_id");
                    assert_eq!(desc.spatial_id, 0, "TU {i} spatial_id");
                }
            }
        }
    }

    // Full decode (operating point 0) = every reconstruction.
    let full = decode_av1_spec(&enc.gop.ivf_bytes).expect("full decode");
    assert_eq!(full.len(), n, "full decode surfaces every shown frame");
    for (i, f) in full.iter().enumerate() {
        assert_eq!(
            frame_planes(f),
            recon_planes(&enc.gop, i),
            "full decode frame {i} != encoder recon"
        );
    }

    // Every reduced operating point = exactly the surviving-layer
    // frame subset, byte-identical to the same reconstructions.
    for k in 1..layers {
        let top = layers - 1 - k; // highest surviving temporal layer
        let out = decode_av1_spec_at_operating_point(&enc.gop.ivf_bytes, k)
            .unwrap_or_else(|e| panic!("op {k} decode failed: {e:?}"));
        let survivors: Vec<usize> = (0..n).filter(|&i| enc.temporal_ids[i] <= top).collect();
        assert_eq!(
            out.len(),
            survivors.len(),
            "op {k} must decode the tid <= {top} subset"
        );
        for (f, &i) in out.iter().zip(&survivors) {
            assert_eq!(
                frame_planes(f),
                recon_planes(&enc.gop, i),
                "op {k}: surviving frame {i} != encoder recon"
            );
        }
    }
}

#[test]
fn two_layer_gop_decodes_at_both_operating_points() {
    assert_layered_gop_round_trip(6, 2, 60);
}

#[test]
fn three_layer_gop_decodes_at_every_operating_point() {
    assert_layered_gop_round_trip(8, 3, 72);
}

#[test]
fn lossless_two_layer_gop_round_trips_the_input() {
    let n = 4;
    let frames = source_frames(n, 64, 64);
    let enc =
        encode_temporal_layered_gop_yuv420_with_q(&frames, 0, 2).expect("lossless layered GOP");
    let full = decode_av1_spec(&enc.gop.ivf_bytes).expect("full decode");
    assert_eq!(full.len(), n);
    // base_q_idx == 0: reconstruction == input, so the decode must
    // recover the source frames themselves.
    for (i, f) in full.iter().enumerate() {
        let mut want = frames[i].y.clone();
        want.extend_from_slice(&frames[i].u);
        want.extend_from_slice(&frames[i].v);
        assert_eq!(frame_planes(f), want, "lossless frame {i}");
    }
}

#[test]
fn layer_count_bounds_are_enforced() {
    let frames = source_frames(2, 64, 64);
    assert!(encode_temporal_layered_gop_yuv420_with_q(&frames, 60, 1).is_err());
    assert!(encode_temporal_layered_gop_yuv420_with_q(&frames, 60, 5).is_err());
}

/// Local-only black-box hook: when `AV1_R430_DUMP_DIR` is set, write
/// the three-layer fixture for external reference-decoder
/// cross-checks (independent decoders' operating-point
/// selection flags). Inert in CI (no env).
#[test]
fn dump_three_layer_fixture_for_blackbox_when_requested() {
    let Some(dir) = std::env::var_os("AV1_R430_DUMP_DIR") else {
        return;
    };
    let frames = source_frames(8, 64, 64);
    let enc = encode_temporal_layered_gop_yuv420_with_q(&frames, 72, 3).expect("encodes");
    let d = std::path::Path::new(&dir);
    std::fs::write(d.join("layers3.ivf"), &enc.gop.ivf_bytes).expect("dump");
    for k in 0..3u8 {
        let out = decode_av1_spec_at_operating_point(&enc.gop.ivf_bytes, k).expect("decodes");
        let mut buf = Vec::new();
        for f in &out {
            buf.extend_from_slice(&frame_planes(f));
        }
        std::fs::write(d.join(format!("layers3_op{k}.yuv")), buf).expect("dump");
    }
}
