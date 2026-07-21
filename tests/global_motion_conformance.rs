//! r422 — conformance witnesses for the frame-level **global-motion
//! election** (§5.9.24): on content whose motion field IS a global
//! model (pan / zoom / rotation), the encoder must (a) elect the
//! matching `GmType` on at least the LAST reference of some P frame,
//! (b) write it through the §5.9.25 signed-subexp arm, and (c) still
//! decode byte-exact through the crate's spec-faithful frame driver
//! against its own reconstruction — proving the search mirror, the
//! write pass and the decoder all agree about the model's effect on
//! `GlobalMvs`, GLOBALMV leaves and the ≥8×8 global-warp prediction
//! path.
//!
//! External black-box validation of these streams is exercised during
//! the round and pinned via `fixture_conformance.rs` corpus entries.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q, EncodedGop, Yuv420Frame};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, FrameType, RefInfo};
use oxideav_av1::uncompressed_header_tail::WarpModelType;
use oxideav_av1::{ObuIter, ObuType};

/// Smooth, non-repeating luma base "scene" — sampled at continuous
/// coordinates so the frame generators can apply exact affine maps.
fn scene_y(x: f64, y: f64) -> f64 {
    // Coarse structure + fine 2D texture: every 16×16 patch must be
    // textured in BOTH directions, or block matching degenerates to
    // the aperture problem and the pre-pass vectors go noisy.
    128.0
        + 34.0 * (0.115 * x + 0.7 * (0.052 * y).sin()).sin()
        + 28.0 * (0.093 * y + 0.6 * (0.047 * x).sin()).cos()
        + 14.0 * (0.021 * (x + y)).sin()
        + 22.0 * (0.41 * x + 0.31 * (0.13 * y).sin() + 0.09 * y).sin()
        + 18.0 * (0.37 * y - 0.23 * (0.11 * x).sin() + 0.07 * x).cos()
}

fn scene_u(x: f64, y: f64) -> f64 {
    128.0 + 34.0 * (0.061 * x - 0.045 * y).sin()
}

fn scene_v(x: f64, y: f64) -> f64 {
    128.0 + 30.0 * (0.052 * x + 0.058 * y).cos()
}

/// Sample one frame of `w`×`h` 4:2:0 video where luma pel `(px, py)`
/// shows the scene at `map(px, py)` (and chroma at the co-sited
/// luma coordinates).
fn sample_frame(w: u32, h: u32, map: impl Fn(f64, f64) -> (f64, f64)) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            let (sx, sy) = map(c as f64, r as f64);
            f.y[r * wu + c] = clamp(scene_y(sx, sy));
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            let (sx, sy) = map(c as f64 * 2.0 + 0.5, r as f64 * 2.0 + 0.5);
            f.u[r * cw + c] = clamp(scene_u(sx, sy));
            f.v[r * cw + c] = clamp(scene_v(sx, sy));
        }
    }
    f
}

/// Parse the (single) frame header of temporal unit `tu_index` (the
/// 1-based P-frame index of the two-slot GOP rotation) and return the
/// per-ref `(GmType, gm_params)` global-motion state.
///
/// The header must be parsed against the decoder session's TRUE
/// per-slot state (the §5.9.22 `skipModeAllowed` derivation reads the
/// stored `RefOrderHint[]`), so this rebuilds the GOP's slot rotation
/// for P-frame `k`: slot `k & 1` holds frame `k - 1`, slot
/// `(k - 1) & 1` holds frame `k - 2` (clamped at the KEY), every
/// other slot still the KEY.
fn frame_gm(
    enc: &EncodedGop,
    tu_index: usize,
    w: u32,
    h: u32,
) -> (Vec<WarpModelType>, Vec<[i32; 6]>) {
    let k = tu_index as u32;
    let mut ref_info = RefInfo::default();
    for slot in 0..8usize {
        ref_info.valid[slot] = true;
        ref_info.upscaled_width[slot] = w;
        ref_info.frame_height[slot] = h;
        ref_info.render_width[slot] = w;
        ref_info.render_height[slot] = h;
        ref_info.frame_type_is_key[slot] = true;
    }
    ref_info.order_hint[(k & 1) as usize] = k - 1;
    ref_info.order_hint[((k - 1) & 1) as usize] = k.saturating_sub(2);
    let tu = &enc.temporal_units[tu_index];
    for obu in ObuIter::new(tu) {
        let obu = obu.expect("own stream parses");
        if obu.obu_type == ObuType::Frame {
            let fh = parse_frame_header_with_refs(obu.payload, &enc.seq, &ref_info)
                .expect("own frame header parses");
            assert_eq!(fh.frame_type, FrameType::Inter, "P-frame TU expected");
            let gm = fh
                .global_motion_params
                .expect("inter header carries §5.9.24 state");
            return (gm.gm_type.to_vec(), gm.gm_params.to_vec());
        }
    }
    panic!("temporal unit {tu_index} has no OBU_FRAME");
}

/// Encode, then decode through the spec driver and require byte-exact
/// per-frame equality with the encoder reconstruction.
fn assert_round_trip(frames: &[Yuv420Frame], q: u8) -> EncodedGop {
    let enc = encode_gop_yuv420_with_q(frames, q).expect("GOP encode");
    let decoded = decode_av1_spec(&enc.ivf_bytes).expect("spec driver accepts own stream");
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.recon[idx].y, "frame {idx} luma");
        assert_eq!(f.planes[1], enc.recon[idx].u, "frame {idx} U");
        assert_eq!(f.planes[2], enc.recon[idx].v, "frame {idx} V");
    }
    enc
}

/// Env-gated stream dump (`OXIDEAV_AV1_GM_DIR`) for external
/// black-box decoder validation and corpus pinning: writes
/// `<dir>/<name>/input.ivf` plus the display-order yuv420p
/// reconstruction as `expected.yuv` — the fixture form used under
/// `docs/video/av1/fixtures/`.
fn maybe_dump(name: &str, enc: &EncodedGop) {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_GM_DIR") else {
        return;
    };
    let out = std::path::Path::new(&dir).join(name);
    std::fs::create_dir_all(&out).expect("create dump dir");
    std::fs::write(out.join("input.ivf"), &enc.ivf_bytes).expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &enc.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(out.join("expected.yuv"), &yuv).expect("write yuv");
}

/// LAST_FRAME reference index in the §6.8.18 per-ref arrays.
const LAST: usize = 1;

#[test]
fn pan_content_elects_translation_and_round_trips() {
    // The scene pans by (+2.0, +1.25) luma pels per frame.
    let (w, h) = (64u32, 64u32);
    let frames: Vec<Yuv420Frame> = (0..4)
        .map(|k| {
            let (dx, dy) = (2.0 * k as f64, 1.25 * k as f64);
            sample_frame(w, h, |x, y| (x + dx, y + dy))
        })
        .collect();
    let enc = assert_round_trip(&frames, 60);
    maybe_dump("self-gop-64x64-q60-gm-pan", &enc);
    let mut elected = 0usize;
    for tu in 1..enc.temporal_units.len() {
        let (types, params) = frame_gm(&enc, tu, w, h);
        if types[LAST] == WarpModelType::Translation {
            elected += 1;
            assert!(
                params[LAST][0] != 0 || params[LAST][1] != 0,
                "a TRANSLATION model must carry a non-zero vector"
            );
        }
    }
    assert!(
        elected >= 2,
        "panning content should elect TRANSLATION on most P frames (got {elected}/3)"
    );
}

#[test]
fn zoom_content_elects_rotzoom_and_round_trips() {
    // The camera zooms out 3.5% per frame about the frame centre: the
    // reference position of a current pel moves AWAY from the centre
    // frame-over-frame, i.e. the per-frame model is a uniform scale
    // > 1 — squarely ROTZOOM territory (and, on ≥8×8 GLOBALMV leaves,
    // the §7.11.3 global-warp prediction path).
    let (w, h) = (64u32, 64u32);
    let (cx, cy) = (31.5f64, 31.5f64);
    let frames: Vec<Yuv420Frame> = (0..4)
        .map(|k| {
            let s = 1.035f64.powi(k);
            sample_frame(w, h, |x, y| (cx + s * (x - cx), cy + s * (y - cy)))
        })
        .collect();
    let enc = assert_round_trip(&frames, 60);
    maybe_dump("self-gop-64x64-q60-gm-zoom-warp", &enc);
    let mut elected = 0usize;
    for tu in 1..enc.temporal_units.len() {
        let (types, params) = frame_gm(&enc, tu, w, h);
        if types[LAST] == WarpModelType::RotZoom {
            elected += 1;
            let p = params[LAST];
            // §5.9.24 derived pair + an expanding diagonal (the
            // per-frame scale is ~1.035).
            assert_eq!(p[4], -p[3]);
            assert_eq!(p[5], p[2]);
            let diag = f64::from(p[2]) / 65536.0;
            assert!(
                (1.005..1.10).contains(&diag),
                "zoom diagonal off: {diag} (p2 = {})",
                p[2]
            );
        }
    }
    assert!(
        elected >= 2,
        "zooming content should elect ROTZOOM on most P frames (got {elected}/3)"
    );
}

#[test]
fn rotation_content_elects_rotzoom_and_round_trips() {
    // The camera rotates ~1.7° per frame about the frame centre.
    let (w, h) = (64u32, 64u32);
    let (cx, cy) = (31.5f64, 31.5f64);
    let frames: Vec<Yuv420Frame> = (0..4)
        .map(|k| {
            let th = 0.03f64 * k as f64;
            let (c, s) = (th.cos(), th.sin());
            sample_frame(w, h, |x, y| {
                let (rx, ry) = (x - cx, y - cy);
                (cx + c * rx + s * ry, cy - s * rx + c * ry)
            })
        })
        .collect();
    let enc = assert_round_trip(&frames, 60);
    maybe_dump("self-gop-64x64-q60-gm-rotation", &enc);
    let mut elected = 0usize;
    for tu in 1..enc.temporal_units.len() {
        let (types, params) = frame_gm(&enc, tu, w, h);
        if types[LAST] == WarpModelType::RotZoom {
            elected += 1;
            let p = params[LAST];
            assert_eq!(p[4], -p[3]);
            assert_eq!(p[5], p[2]);
            assert!(p[3] != 0, "a rotation must carry a non-zero off-diagonal");
        }
    }
    assert!(
        elected >= 2,
        "rotating content should elect ROTZOOM on most P frames (got {elected}/3)"
    );
}

#[test]
fn static_content_keeps_identity_headers() {
    // No motion ⇒ the election must leave every ref IDENTITY (the
    // pre-r422 header byte stream shape).
    let (w, h) = (64u32, 64u32);
    let frames: Vec<Yuv420Frame> = (0..3).map(|_| sample_frame(w, h, |x, y| (x, y))).collect();
    let enc = assert_round_trip(&frames, 60);
    for tu in 1..enc.temporal_units.len() {
        let (types, _) = frame_gm(&enc, tu, w, h);
        assert!(
            types.iter().all(|t| *t == WarpModelType::Identity),
            "static content must not elect a model"
        );
    }
}
