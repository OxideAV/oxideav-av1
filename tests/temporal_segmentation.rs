//! r423 — §5.11.19 temporal segment-map coding + §5.9.2 primary-
//! reference cross-frame carry: conformance and A/B measurement.
//!
//! Always-on:
//!
//! * Persistent-segment GOPs (static and moving SEG_LVL_ALT_Q
//!   regions) must round-trip byte-exact through the in-tree
//!   spec-faithful frame driver under the DEFAULT tuning (primary
//!   reference + temporal election live), and the §5.9.14
//!   `temporal_update` election must actually fire on at least one
//!   P-frame of the persistent-segment content.
//! * Every A/B variant (primary reference off; temporal election
//!   off) must round-trip the same way — the switches select between
//!   conformant streams, never between "works" and "broken".
//! * Aggregate tripwires: on the persistent-segment matrix the
//!   temporal election must not cost bytes in aggregate against the
//!   spatial-only baseline, and the primary-reference carry must not
//!   cost bytes in aggregate against the `PRIMARY_REF_NONE`
//!   baseline (forward CDF inheritance is a pure win on these GOP
//!   lengths).
//!
//! Env-gated (`OXIDEAV_AV1_SEG_AB_DIR=<dir>`): the full measurement
//! matrix — per-config bytes under the three variants, a CSV
//! (`seg_ab.csv`), aggregate deltas on stderr, and the default-tuning
//! IVF of every config for external black-box decoder validation.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg, encode_gop_yuv420_with_q_seg_tuned, EncodedGop, GopTuning,
    Yuv420Frame,
};

// ---------------------------------------------------------------------
// Content builders.
// ---------------------------------------------------------------------

/// Textured scene with a FLAT rectangle whose top-left corner sits at
/// `(rx, ry)` — the encoder's activity policy maps the flat region to
/// segment 0 and the textured remainder to the higher segments, so a
/// static `(rx, ry)` yields a frame-over-frame-identical segment map
/// (the §5.11.21 prediction hits everywhere) while a moving one
/// exercises the prediction-miss fallback at the region edges.
fn seg_frame(w: u32, h: u32, rx: usize, ry: usize, rw: usize, rh: usize, k: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            // Texture pattern scrolls by one sample per frame so the
            // P-frames carry real residuals + motion.
            let (si, sj) = (i + k, j + 2 * k);
            f.y[i * wu + j] = ((si * 13 + sj * 7 + (si / 8) * (sj / 8) * 5) % 256) as u8;
        }
    }
    // Flat rectangle (clipped to the frame).
    for i in ry..(ry + rh).min(hu) {
        for j in rx..(rx + rw).min(wu) {
            f.y[i * wu + j] = 96;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = ((120 + i + j + k) % 256) as u8;
            f.v[i * cw + j] = ((150 + 2 * i + j) % 256) as u8;
        }
    }
    f
}

/// GOP with a STATIC flat region — the natural temporal-update win
/// case (segment affiliations persist frame over frame).
fn static_region_gop(n: usize, w: u32, h: u32) -> Vec<Yuv420Frame> {
    (0..n)
        .map(|k| seg_frame(w, h, 8, 8, (w as usize) / 2, (h as usize) / 2, k))
        .collect()
}

/// GOP whose flat region MOVES 8 luma samples right per frame — the
/// brief's "moving segmented region": interior blocks keep their
/// affiliation (prediction hits), leading/trailing edges flip
/// (prediction misses ride the spatial fallback).
fn moving_region_gop(n: usize, w: u32, h: u32) -> Vec<Yuv420Frame> {
    (0..n)
        .map(|k| seg_frame(w, h, 8 + 8 * k, 16, (w as usize) / 2, (h as usize) / 2, k))
        .collect()
}

/// The r413 SEG_LVL_ALT_Q ladder used across the segmentation tests.
const ALT_Q: [i16; 3] = [0, 24, 48];

// ---------------------------------------------------------------------
// Round-trip helpers.
// ---------------------------------------------------------------------

fn assert_round_trip(enc: &EncodedGop, n_frames: usize, what: &str) {
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{what}: spec driver rejected own stream: {e:?}"));
    assert_eq!(decoded.len(), n_frames, "{what}: shown frame count");
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.recon[idx];
        assert_eq!(f.planes[0], rc.y, "{what} frame {idx}: luma");
        assert_eq!(f.planes[1], rc.u, "{what} frame {idx}: U");
        assert_eq!(f.planes[2], rc.v, "{what} frame {idx}: V");
    }
}

fn total_bytes(enc: &EncodedGop) -> usize {
    enc.temporal_units.iter().map(Vec::len).sum()
}

fn tuned(frames: &[Yuv420Frame], q: u8, alt_q: &[i16], tuning: GopTuning) -> EncodedGop {
    encode_gop_yuv420_with_q_seg_tuned(frames, q, alt_q, tuning).expect("tuned encode")
}

// ---------------------------------------------------------------------
// Always-on conformance + election witnesses.
// ---------------------------------------------------------------------

/// Default tuning on the static persistent-segment GOP: byte-exact
/// spec-driver round trip AND the §5.9.14 `temporal_update` election
/// fires (segment affiliations persist, so the temporal arm's 1-bit
/// adoptions beat the spatial S() cascade).
#[test]
fn static_segments_elect_temporal_and_round_trip() {
    let frames = static_region_gop(5, 96, 64);
    let enc = encode_gop_yuv420_with_q_seg(&frames, 72, &ALT_Q).unwrap();
    assert_round_trip(&enc, frames.len(), "static-region default");
    assert_eq!(enc.seg_temporal_updates.len(), frames.len() - 1);
    assert!(
        enc.seg_temporal_updates.iter().any(|&b| b),
        "persistent segments must elect temporal_update on at least one P-frame \
         (got {:?})",
        enc.seg_temporal_updates
    );
}

/// Moving-region GOP: round trip under default tuning (the
/// prediction-miss edges exercise the §5.11.19 `seg_id_predicted = 0`
/// spatial fallback inside temporal frames).
#[test]
fn moving_segments_round_trip_under_temporal_coding() {
    let frames = moving_region_gop(5, 96, 64);
    let enc = encode_gop_yuv420_with_q_seg(&frames, 72, &ALT_Q).unwrap();
    assert_round_trip(&enc, frames.len(), "moving-region default");
    assert_eq!(enc.seg_temporal_updates.len(), frames.len() - 1);
}

/// Every tuning variant is conformant — the switches trade rate, not
/// correctness. Also the aggregate tripwires (see module docs).
#[test]
fn tuning_variants_round_trip_with_aggregate_tripwires() {
    let q = 72u8;
    let mut on_total = 0usize; // default: primary ref + temporal election
    let mut spatial_total = 0usize; // primary ref, spatial-only maps
    let mut none_total = 0usize; // PRIMARY_REF_NONE baseline
    for (name, frames) in [
        ("static", static_region_gop(4, 96, 64)),
        ("moving", moving_region_gop(4, 96, 64)),
    ] {
        let on = tuned(&frames, q, &ALT_Q, GopTuning::default());
        let spatial = tuned(
            &frames,
            q,
            &ALT_Q,
            GopTuning {
                temporal_seg: false,
                ..GopTuning::default()
            },
        );
        let none = tuned(
            &frames,
            q,
            &ALT_Q,
            GopTuning {
                primary_ref: false,
                ..GopTuning::default()
            },
        );
        assert_round_trip(&on, frames.len(), &format!("{name} temporal-elected"));
        assert_round_trip(&spatial, frames.len(), &format!("{name} spatial-only"));
        assert_round_trip(&none, frames.len(), &format!("{name} primary-ref-none"));
        assert!(
            spatial.seg_temporal_updates.iter().all(|&b| !b),
            "{name}: temporal_seg = false must force spatial maps"
        );
        assert!(
            none.seg_temporal_updates.iter().all(|&b| !b),
            "{name}: PRIMARY_REF_NONE frames cannot elect temporal maps"
        );
        on_total += total_bytes(&on);
        spatial_total += total_bytes(&spatial);
        none_total += total_bytes(&none);
        eprintln!(
            "seg-ab {name}: temporal-elected {} B, spatial-only {} B, primary-none {} B",
            total_bytes(&on),
            total_bytes(&spatial),
            total_bytes(&none)
        );
    }
    // Aggregate tripwires (per-frame the election is exact-bits over
    // identical trees; across independently-searched runs only the
    // aggregate is asserted).
    assert!(
        on_total <= spatial_total,
        "temporal election cost bytes in aggregate: {on_total} > {spatial_total}"
    );
    assert!(
        spatial_total <= none_total,
        "primary-reference carry cost bytes in aggregate: {spatial_total} > {none_total}"
    );
}

/// Unsegmented GOP under the primary-reference carry: the §6.8.21 CDF
/// inheritance alone must round-trip and not cost bytes in aggregate
/// against the per-frame-defaults baseline.
#[test]
fn unsegmented_primary_ref_carry_round_trips_and_saves() {
    let frames = static_region_gop(4, 64, 64);
    let on = tuned(&frames, 60, &[], GopTuning::default());
    let off = tuned(
        &frames,
        60,
        &[],
        GopTuning {
            primary_ref: false,
            ..GopTuning::default()
        },
    );
    assert_round_trip(&on, frames.len(), "unsegmented primary-ref");
    assert_round_trip(&off, frames.len(), "unsegmented primary-none");
    let (a, b) = (total_bytes(&on), total_bytes(&off));
    eprintln!("cdf-inheritance ab: primary-ref {a} B vs none {b} B");
    assert!(a <= b, "CDF inheritance cost bytes in aggregate: {a} > {b}");
}

/// Lossless arm sanity: the carry + temporal chain must hold at
/// `base_q_idx == 0` too (recon == input; segmentation stays off —
/// the ALT_Q ladder requires lossy per-segment q).
#[test]
fn lossless_primary_ref_carry_round_trips() {
    let frames = static_region_gop(4, 64, 64);
    let enc = tuned(&frames, 0, &[], GopTuning::default());
    assert_round_trip(&enc, frames.len(), "lossless primary-ref");
    for (idx, f) in frames.iter().enumerate() {
        assert_eq!(enc.recon[idx].y, f.y, "lossless frame {idx}: luma != input");
        assert_eq!(enc.recon[idx].u, f.u, "lossless frame {idx}: U != input");
        assert_eq!(enc.recon[idx].v, f.v, "lossless frame {idx}: V != input");
    }
}

// ---------------------------------------------------------------------
// Env-gated full measurement matrix.
// ---------------------------------------------------------------------

/// Full matrix behind `OXIDEAV_AV1_SEG_AB_DIR` (skips silently when
/// unset — the always-on tests above are the CI tier). Writes
/// `seg_ab.csv` (config, bytes under the three variants, elected
/// temporal frame count) plus the default-tuning IVF per config for
/// external black-box decoder validation.
#[test]
fn seg_ab_measurement_matrix() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SEG_AB_DIR") else {
        eprintln!("OXIDEAV_AV1_SEG_AB_DIR unset — skipping the full seg measurement");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("create out dir");
    let mut csv = String::from("config,q,temporal_bytes,spatial_bytes,none_bytes,elected\n");
    let mut agg = [0usize; 3];
    for (name, frames) in [
        ("static-128x64x8", static_region_gop(8, 128, 64)),
        ("moving-128x64x8", moving_region_gop(8, 128, 64)),
        ("static-192x128x6", static_region_gop(6, 192, 128)),
        ("moving-192x128x6", moving_region_gop(6, 192, 128)),
    ] {
        for q in [40u8, 72, 120] {
            let on = tuned(&frames, q, &ALT_Q, GopTuning::default());
            let spatial = tuned(
                &frames,
                q,
                &ALT_Q,
                GopTuning {
                    temporal_seg: false,
                    ..GopTuning::default()
                },
            );
            let none = tuned(
                &frames,
                q,
                &ALT_Q,
                GopTuning {
                    primary_ref: false,
                    ..GopTuning::default()
                },
            );
            assert_round_trip(&on, frames.len(), &format!("{name} q{q} temporal"));
            assert_round_trip(&spatial, frames.len(), &format!("{name} q{q} spatial"));
            assert_round_trip(&none, frames.len(), &format!("{name} q{q} none"));
            let (a, b, c) = (total_bytes(&on), total_bytes(&spatial), total_bytes(&none));
            let elected = on.seg_temporal_updates.iter().filter(|&&x| x).count();
            agg[0] += a;
            agg[1] += b;
            agg[2] += c;
            csv.push_str(&format!("{name},{q},{a},{b},{c},{elected}\n"));
            std::fs::write(dir.join(format!("{name}-q{q}.ivf")), &on.ivf_bytes).expect("write ivf");
        }
    }
    std::fs::write(dir.join("seg_ab.csv"), &csv).expect("write csv");
    eprintln!(
        "seg-ab aggregate: temporal {} B, spatial {} B, primary-none {} B \
         (temporal saves {:.2}% vs spatial, {:.2}% vs none)",
        agg[0],
        agg[1],
        agg[2],
        100.0 * (agg[1] as f64 - agg[0] as f64) / agg[1] as f64,
        100.0 * (agg[2] as f64 - agg[0] as f64) / agg[2] as f64,
    );
}
