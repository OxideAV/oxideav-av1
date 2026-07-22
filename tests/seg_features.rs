//! r426 — ladder item 8: the §5.9.14 inter-override segmentation
//! features (SEG_LVL_SKIP / SEG_LVL_GLOBALMV / SEG_LVL_REF_FRAME).
//!
//! The decoder has implemented the §5.11.10/§5.11.11/§5.11.20/
//! §5.11.23/§5.11.25 feature arms since r394, but no conformance
//! stream ever pinned them — the external encoder's CLI cannot
//! signal the features. OUR encoder now can: every test encodes a
//! GOP whose feature segment is elected per leaf by the r426
//! twin-priced trials, asserts the anti-desync invariant through the
//! crate's spec-faithful frame driver (which parses the
//! `SegIdPreSkip = 1` pre-skip segment-id arm end-to-end), and
//! asserts the committed maps actually reach the feature segment.
//! With `OXIDEAV_AV1_SEGFEAT_DIR` set the streams dump for the
//! triple-black-box-decoder pin flow.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, GopTuning, SegExtras, Yuv420Frame,
};

/// Static textured background with a small moving square: the
/// background leaves are pure-derivation SEG_LVL_SKIP material, the
/// square keeps the normal ladder busy.
fn static_with_mover(w: u32, h: u32, k: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            f.y[i * wu + j] = (60 + ((i * 5) ^ (j * 3)) % 120) as u8;
        }
    }
    // Moving 16x16 square.
    let (sx, sy) = (8 + 6 * k, 8 + 4 * k);
    for i in sy..(sy + 16).min(hu) {
        for j in sx..(sx + 16).min(wu) {
            f.y[i * wu + j] = 220;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = (100 + (i + 2 * j) % 40) as u8;
            f.v[i * cw + j] = (140 + (2 * i + j) % 40) as u8;
        }
    }
    f
}

/// Whole-frame ACCELERATING pan over a smooth sinusoid scene (the
/// r422 motion prepass tracks this content class reliably): frame
/// `k` translates by `(0, k (k + 3))` luma pels (integer — the
/// §5.9.25 model grid represents it exactly), so the per-frame
/// velocity CHANGES — the §5.9.22 skip-mode compound average of LAST
/// and GOLDEN (whose implied velocities disagree) blurs while the
/// TRANSLATION global model tracks each frame exactly, making the
/// mode/ref-silent GLOBALMV segment the cheapest arm.
fn panning(w: u32, h: u32, k: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let kf = k as f64;
    let (dx, dy) = (kf * (kf + 3.0), 0.0);
    let scene_y = |x: f64, y: f64| 128.0 + 55.0 * (0.11 * x).sin() * (0.093 * y).cos();
    let scene_u = |x: f64, y: f64| 120.0 + 40.0 * (0.061 * x + 0.043 * y).sin();
    let scene_v = |x: f64, y: f64| 132.0 + 34.0 * (0.052 * x - 0.058 * y).cos();
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    let mut f = Yuv420Frame::filled(w, h, 0);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = clamp(scene_y(c as f64 + dx, r as f64 + dy));
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            let (sx, sy) = (c as f64 * 2.0 + 0.5 + dx, r as f64 * 2.0 + 0.5 + dy);
            f.u[r * cw + c] = clamp(scene_u(sx, sy));
            f.v[r * cw + c] = clamp(scene_v(sx, sy));
        }
    }
    f
}

/// Encode with `extras`, decode through the spec driver, assert the
/// anti-desync invariant on every frame and that at least one
/// P-frame commits cells of `feature_seg`. Returns the IVF bytes.
fn assert_feature_round_trip(
    name: &str,
    frames: &[Yuv420Frame],
    q: u8,
    alt_q: &[i16],
    extras: &SegExtras,
    feature_seg: i32,
) -> Vec<u8> {
    let tuned = encode_gop_yuv420_with_q_seg_extras_tuned(
        frames,
        q,
        alt_q,
        &[],
        false,
        Some(extras),
        GopTuning::default(),
    )
    .unwrap_or_else(|e| panic!("{name}: encode failed: {e:?}"));
    let enc = &tuned.gop;
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{name}: spec driver rejected the stream: {e:?}"));
    assert_eq!(decoded.len(), frames.len(), "{name}: frame count");
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.recon[idx];
        assert_eq!(f.planes[0], rc.y, "{name} frame {idx}: luma desync");
        assert_eq!(f.planes[1], rc.u, "{name} frame {idx}: U desync");
        assert_eq!(f.planes[2], rc.v, "{name} frame {idx}: V desync");
    }
    assert!(
        tuned
            .p_segment_maps
            .iter()
            .any(|map| map.contains(&feature_seg)),
        "{name}: no P-frame committed feature-segment {feature_seg} cells"
    );
    if let Some(dir) = std::env::var_os("OXIDEAV_AV1_SEGFEAT_DIR") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).expect("dump dir");
        std::fs::write(dir.join(format!("{name}.ivf")), &enc.ivf_bytes).expect("dump ivf");
    }
    enc.ivf_bytes.clone()
}

/// SEG_LVL_SKIP: the static background elects the pure-derivation
/// skip segment (no §5.11.11 bit, silent GLOBALMV/LAST, no residual)
/// under `SegIdPreSkip = 1`.
#[test]
fn seg_lvl_skip_static_background_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|k| static_with_mover(96, 80, k)).collect();
    let mut x = SegExtras::default();
    x.skip[1] = true;
    assert_feature_round_trip("segfeat-skip-96x80-q80", &frames, 80, &[0, 0], &x, 1);
}

/// SEG_LVL_GLOBALMV: panning content with the r422 TRANSLATION model
/// elects the mode/ref-silent global segment.
#[test]
fn seg_lvl_globalmv_pan_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|k| panning(96, 80, k)).collect();
    let mut x = SegExtras::default();
    x.globalmv[1] = true;
    assert_feature_round_trip("segfeat-globalmv-96x80-q72", &frames, 72, &[0, 0], &x, 1);
}

/// SEG_LVL_REF_FRAME (data = LAST_FRAME): single-LAST winners
/// re-label onto the reference-silent segment by exact twin bits.
#[test]
fn seg_lvl_ref_frame_last_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|k| static_with_mover(64, 64, 2 * k)).collect();
    let mut x = SegExtras::default();
    x.ref_frame[1] = Some(1);
    assert_feature_round_trip("segfeat-reflast-64x64-q60", &frames, 60, &[0, 0], &x, 1);
}

/// Malformed feature plans reject: features on segment 0, empty
/// alt_q carrier, out-of-ladder reference data, SKIP on a lossless
/// segment.
#[test]
fn seg_feature_validation_rejects_malformed_plans() {
    let frames: Vec<Yuv420Frame> = (0..2).map(|k| static_with_mover(64, 64, k)).collect();
    let cases: Vec<(u8, Vec<i16>, SegExtras)> = vec![
        (60, vec![0, 0], {
            let mut x = SegExtras::default();
            x.skip[0] = true; // segment 0 must stay feature-free
            x
        }),
        (60, vec![], {
            let mut x = SegExtras::default();
            x.skip[1] = true; // needs the alt_q carrier
            x
        }),
        (60, vec![0, 0], {
            let mut x = SegExtras::default();
            x.ref_frame[1] = Some(2); // LAST2 is outside the P-GOP ladder
            x
        }),
        (60, vec![0, -60], {
            let mut x = SegExtras::default();
            x.skip[1] = true; // SKIP on a lossless segment
            x
        }),
    ];
    for (q, alt_q, x) in cases {
        assert!(
            encode_gop_yuv420_with_q_seg_extras_tuned(
                &frames,
                q,
                &alt_q,
                &[],
                false,
                Some(&x),
                GopTuning::default(),
            )
            .is_err(),
            "q={q} alt_q={alt_q:?} extras={x:?} must be rejected"
        );
    }
}
