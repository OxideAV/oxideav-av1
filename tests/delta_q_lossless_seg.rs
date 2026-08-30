//! r453 — §5.9.17 delta-q on SEG_LVL_ALT_Q tables carrying a LOSSLESS
//! segment, under the EXACT §7.12.2 guard.
//!
//! `LosslessArray[]` is derived with `ignoreDeltaQ = 1` (`base_q_idx +
//! data`), while a block dequantizes at `get_qindex( 0, segmentId ) =
//! Clip3( 0, 255, CurrentQIndex + data )`. A lossless segment stays
//! lossless under delta-q exactly when every realized `CurrentQIndex`
//! keeps `CurrentQIndex + data <= 0`; the encoder caps the plan's
//! upward units accordingly (a `-255` table keeps the full span, a
//! table whose lossless data sits at `-base_q_idx` keeps only the
//! refining swings), replacing the conservative r436 rule that kept
//! such tables on the single-quantiser arm.
//!
//! Coverage:
//! * a `-255` lossless table with an exactness-demand region ELECTS
//!   the delta arm on every P-frame of mixed content — the first
//!   `delta_q_present = 1` headers carrying a lossless segment —
//!   decodes byte-exact through the spec driver, and the region
//!   stays pixel-exact against the input on all three planes (the
//!   mechanism path's lossless-segment leaves may legitimately be
//!   skip leaves, so exactness is asserted on the region API, which
//!   codes them by construction),
//! * a table whose lossless data sits exactly at `-base_q_idx`
//!   (headroom 0: only the downward swings survive) keeps the region
//!   pixel-exact,
//! * `delta_q: false` keeps the single-quantiser shape on the same
//!   table,
//! * an env-gated staging dump feeds black-box reference-decoder
//!   validation and corpus pinning.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.17, §5.11.13, §7.12.2.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_lossless_tuned, GopTuning, LosslessRegion, TunedGop, Yuv420Frame,
};
use oxideav_av1::frame_header::{FrameHeader, FrameType, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Dense texture field (high per-superblock variance).
fn tex(x: f64, y: f64) -> f64 {
    128.0
        + 42.0 * (0.71 * x + 0.9 * (0.23 * y).sin()).sin()
        + 36.0 * (0.63 * y - 0.7 * (0.31 * x).sin()).cos()
        + 20.0 * (0.47 * (x + y)).sin()
}

/// Smooth ramp (near-zero variance per superblock).
fn flat(x: f64, y: f64) -> f64 {
    90.0 + 0.22 * x + 0.13 * y
}

/// Top half flat (slow brightness ramp the refined superblocks can
/// track), bottom half panning texture — maximal per-superblock
/// activity spread for the §5.9.17 probe; the activity policy maps
/// the textured leaves to the TOP segment of the table.
fn mixed_frame(w: u32, h: u32, k: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let d = 1.75 * k as f64;
    let ramp = 2.5 * k as f64;
    let mut f = Yuv420Frame::filled(w, h, 0);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            let (x, y) = (c as f64 + d, r as f64 + 0.5 * d);
            let v = if r < hu / 2 {
                flat(c as f64, r as f64) + ramp
            } else {
                tex(x, y)
            };
            f.y[r * wu + c] = clamp(v);
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = clamp(120.0 + 16.0 * (0.05 * (c as f64 + d)).sin());
            f.v[r * cw + c] = clamp(132.0 + 14.0 * (0.06 * (r as f64 + d)).cos());
        }
    }
    f
}

fn frames(n: usize) -> Vec<Yuv420Frame> {
    (0..n).map(|k| mixed_frame(128, 128, k)).collect()
}

fn tuning() -> GopTuning {
    GopTuning {
        delta_q: true,
        // Axis isolation: hold the §5.9.12 QM election off.
        qm: false,
        ..GopTuning::default()
    }
}

/// Every coded header's `(delta_q_present, has a lossless-segment
/// table)` pair, parsed against tracked §7.20 state.
fn wire_delta_q_shape(tus: &[Vec<u8>]) -> Vec<(bool, bool)> {
    let mut seq = None;
    let mut ref_info = RefInfo::default();
    let mut out = Vec::new();
    for tu in tus {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("own stream walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let sq = seq.as_ref().expect("SH precedes frames");
                    let fh: FrameHeader = oxideav_av1::frame_header::parse_frame_header_with_refs(
                        desc.payload,
                        sq,
                        &ref_info,
                    )
                    .expect("frame header parses");
                    if fh.show_existing_frame {
                        continue;
                    }
                    let fs = fh.frame_size.as_ref().expect("coded frame has a size");
                    for slot in 0..8 {
                        if fh.refresh_frame_flags & (1 << slot) != 0 {
                            ref_info.valid[slot] = true;
                            ref_info.order_hint[slot] = fh.order_hint;
                            ref_info.upscaled_width[slot] = fs.upscaled_width;
                            ref_info.frame_height[slot] = fs.frame_height;
                            ref_info.render_width[slot] = fs.render_width;
                            ref_info.render_height[slot] = fs.render_height;
                            ref_info.frame_type_is_key[slot] = fh.frame_type == FrameType::Key;
                        }
                    }
                    let dq = fh
                        .delta_q_params
                        .as_ref()
                        .map(|d| d.delta_q_present)
                        .unwrap_or(false);
                    // A lossless segment: base_q_idx + ALT_Q data
                    // clamps to 0 (§5.9.2 `LosslessArray[]`, with
                    // the r413 tables' zero chroma deltas).
                    let base = fh
                        .quantization_params
                        .as_ref()
                        .map(|q| i32::from(q.base_q_idx))
                        .unwrap_or(0);
                    let ll = fh.segmentation_params.as_ref().is_some_and(|sp| {
                        (0..8).any(|sid| {
                            sp.segment_feature_active[sid][0]
                                && base + i32::from(sp.segment_feature_data[sid][0]) <= 0
                        })
                    });
                    out.push((dq, ll));
                }
                _ => {}
            }
        }
    }
    out
}

/// Decode through the spec driver, byte-exact to the recon (the
/// anti-desync gate: the §5.11.13 `CurrentQIndex` walk and the
/// §5.9.14 map compose identically on both sides), and assert the
/// committed maps really reach the lossless top segment.
fn assert_round_trip(frames: &[Yuv420Frame], tuned: &TunedGop, top: i32) {
    let enc = &tuned.gop;
    let decoded = decode_av1_spec(&enc.ivf_bytes).expect("spec driver decodes");
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.recon[idx];
        assert_eq!(f.planes[0], rc.y, "frame {idx}: luma decode != recon");
        assert_eq!(f.planes[1], rc.u, "frame {idx}: U decode != recon");
        assert_eq!(f.planes[2], rc.v, "frame {idx}: V decode != recon");
    }
    assert_eq!(tuned.p_segment_maps.len(), frames.len() - 1);
    assert!(
        tuned.p_segment_maps.iter().any(|m| m.contains(&top)),
        "no P-frame committed a lossless-segment cell"
    );
}

/// The exactness-demand rectangle (luma samples) every witness codes
/// pixel-exact: its leaves ride the lossless segment by construction.
/// It sits inside the textured half.
const REGION: LosslessRegion = LosslessRegion {
    x: 48,
    y: 80,
    width: 32,
    height: 32,
};

/// Assert the region is pixel-exact against the INPUT on every frame
/// and plane (the §7.12.2 exactness the capped plan must preserve).
fn assert_region_exact(frames: &[Yuv420Frame], tuned: &TunedGop) {
    let decoded = decode_av1_spec(&tuned.gop.ivf_bytes).expect("decodes");
    for (idx, f) in decoded.iter().enumerate() {
        let src = &frames[idx];
        let w = src.width as usize;
        let (x0, y0) = (REGION.x as usize, REGION.y as usize);
        let (x1, y1) = (x0 + REGION.width as usize, y0 + REGION.height as usize);
        for y in y0..y1 {
            for x in x0..x1 {
                assert_eq!(
                    f.planes[0][y * w + x],
                    src.y[y * w + x],
                    "frame {idx}: region luma ({y},{x}) not exact"
                );
            }
        }
        let cw = w / 2;
        for y in y0 / 2..y1 / 2 {
            for x in x0 / 2..x1 / 2 {
                assert_eq!(
                    f.planes[1][y * cw + x],
                    src.u[y * cw + x],
                    "frame {idx}: U ({y},{x})"
                );
                assert_eq!(
                    f.planes[2][y * cw + x],
                    src.v[y * cw + x],
                    "frame {idx}: V ({y},{x})"
                );
            }
        }
    }
}

fn encode_region(frames: &[Yuv420Frame], q: u8, alt_q: &[i16], tuning: GopTuning) -> TunedGop {
    encode_gop_yuv420_with_q_seg_lossless_tuned(frames, q, alt_q, &[REGION], false, tuning)
        .expect("region encode")
}

/// The spec note's `-255` table: full plan headroom, the delta arm is
/// ELECTED on every P-frame, the headers carry `delta_q_present = 1`
/// alongside the lossless segment, and the region stays exact.
#[test]
fn minus_255_table_elects_delta_q_with_the_region_exact() {
    let frames = frames(3);
    let tuned = encode_region(&frames, 100, &[0, -255], tuning());
    assert_round_trip(&frames, &tuned, 1);
    assert_region_exact(&frames, &tuned);
    assert!(
        tuned.delta_q_elections.iter().all(|&e| e),
        "mixed content must elect the delta arm on every P-frame: {:?}",
        tuned.delta_q_elections
    );
    let shape = wire_delta_q_shape(&tuned.gop.temporal_units);
    assert_eq!(shape.len(), 3);
    for (idx, &(dq, ll)) in shape.iter().enumerate().skip(1) {
        assert!(ll, "frame {idx}: the table carries a lossless segment");
        assert!(
            dq,
            "frame {idx}: delta_q_present = 1 on the lossless-segment table"
        );
    }
}

/// Headroom 0: the lossless data sits exactly at `-base_q_idx`, so
/// only the downward swings survive — the region stays pixel-exact
/// on every frame while the arm is open.
#[test]
fn zero_headroom_table_keeps_the_region_exact() {
    let frames = frames(3);
    let tuned = encode_region(&frames, 60, &[0, -60], tuning());
    assert_round_trip(&frames, &tuned, 1);
    assert_region_exact(&frames, &tuned);
}

/// `delta_q: false` keeps the single-quantiser shape on the same
/// table (the A/B baseline), and the region stays exact there too.
#[test]
fn delta_q_off_keeps_the_single_quantiser_shape() {
    let frames = frames(3);
    let off = encode_region(
        &frames,
        100,
        &[0, -255],
        GopTuning {
            delta_q: false,
            ..tuning()
        },
    );
    assert_round_trip(&frames, &off, 1);
    assert_region_exact(&frames, &off);
    assert!(off.delta_q_elections.iter().all(|&e| !e));
    let shape = wire_delta_q_shape(&off.gop.temporal_units);
    assert!(shape.iter().all(|&(dq, _)| !dq));
}

/// Env-gated staging dump (`OXIDEAV_AV1_DQ_LL_DIR`): the 3-frame
/// 128×128 q100 `-255`-table GOP with the exactness region as
/// `input.ivf` plus the spec driver's `expected.yuv` (planar I420,
/// frame-major) for black-box reference-decoder validation and
/// corpus pinning.
#[test]
fn delta_q_lossless_seg_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_DQ_LL_DIR") else {
        eprintln!("OXIDEAV_AV1_DQ_LL_DIR unset — skipping the staging dump");
        return;
    };
    let frames = frames(3);
    let tuned = encode_region(&frames, 100, &[0, -255], tuning());
    assert!(tuned.delta_q_elections.iter().all(|&e| e));
    let decoded = decode_av1_spec(&tuned.gop.ivf_bytes).expect("decodes");
    let mut yuv = Vec::new();
    for f in &decoded {
        for p in &f.planes {
            yuv.extend_from_slice(p);
        }
    }
    std::fs::create_dir_all(&dir).expect("staging dir");
    std::fs::write(format!("{dir}/input.ivf"), &tuned.gop.ivf_bytes).expect("write ivf");
    std::fs::write(format!("{dir}/expected.yuv"), &yuv).expect("write yuv");
}
