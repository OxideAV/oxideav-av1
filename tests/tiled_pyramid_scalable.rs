//! r433 — §5.9.15 tile layouts folded into the OUT-OF-ORDER and
//! SCALABLE drivers: the B-pyramid / adaptive mini-GOP encoder
//! (`PyramidTuning::tiles` / `tile_groups`) and the temporally
//! scalable ladder (`encode_temporal_layered_gop_yuv420_with_q_tiles`).
//!
//! What these tests pin:
//!
//!   * Tiled pyramids — every frame of a 2×2-tiled B-pyramid (ALT /
//!     MID / B roles, `show_existing_frame` chains, the full election
//!     set) decodes through the public spec driver byte-exact to the
//!     encoder reconstructions, in display order; with
//!     `tile_groups = 2` each coded frame rides the
//!     `OBU_FRAME_HEADER + 2 tile groups` packaging inside its
//!     temporal unit and the decoded pixels are IDENTICAL to the
//!     single-group pyramid's.
//!   * Tiled temporal ladders — a 3-layer dyadic ladder with 2×1
//!     tiles decodes at EVERY §6.7.5 operating point to exactly the
//!     surviving frames' reconstructions (the §5.3.1 drop rule walks
//!     extension-carrying `OBU_FRAME_HEADER` / `OBU_TILE_GROUP` OBUs
//!     on the split-group shape).
//!   * `(0, 0)` / `tile_groups = 1` byte-identity on both drivers.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.3.1, §5.3.3, §5.9.15,
//! §5.11.1, §6.7.5, §6.10.1, §7.5.

use oxideav_av1::decoder::{decode_av1_spec, decode_av1_spec_at_operating_point, SpecFrame};
use oxideav_av1::encoder::{
    encode_pyramid_gop_yuv420_with_q, encode_pyramid_gop_yuv420_with_q_tuned,
    encode_temporal_layered_gop_yuv420_with_q, encode_temporal_layered_gop_yuv420_with_q_tiles,
    temporal_layer_of, PyramidTuning, Yuv420Frame,
};
use oxideav_av1::obu::{ObuIter, ObuType};

// ---------------------------------------------------------------------
// Content.
// ---------------------------------------------------------------------

/// Moving gradient — every frame distinct, motion crossing tile
/// boundaries.
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

fn frame_planes(f: &SpecFrame) -> Vec<u8> {
    let mut out = Vec::new();
    for p in &f.planes {
        out.extend_from_slice(p);
    }
    out
}

fn recon_planes(gop: &oxideav_av1::encoder::EncodedGop, i: usize) -> Vec<u8> {
    let r = &gop.recon[i];
    let mut out = Vec::with_capacity(r.y.len() + r.u.len() + r.v.len());
    out.extend_from_slice(&r.y);
    out.extend_from_slice(&r.u);
    out.extend_from_slice(&r.v);
    out
}

/// Count `OBU_FRAME` vs `OBU_FRAME_HEADER`+`OBU_TILE_GROUP` shapes
/// across a stream's temporal units. `show_existing_frame` headers
/// also ride `OBU_FRAME_HEADER`, so the split-shape assertion counts
/// tile-group OBUs against CODED frames.
fn obu_census(tus: &[Vec<u8>]) -> (usize, usize, usize) {
    let (mut frames, mut fhs, mut tgs) = (0usize, 0usize, 0usize);
    for tu in tus {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("own TU walks");
            match desc.obu_type {
                ObuType::Frame => frames += 1,
                ObuType::FrameHeader => fhs += 1,
                ObuType::TileGroup => tgs += 1,
                _ => {}
            }
        }
    }
    (frames, fhs, tgs)
}

// ---------------------------------------------------------------------
// Tiled B-pyramids.
// ---------------------------------------------------------------------

/// 2×2-tiled 6-frame pyramid at q 72: display-order decode equals the
/// encoder reconstructions byte for byte.
#[test]
fn tiled_pyramid_round_trips_pixel_exact() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| moving(192, 128, t)).collect();
    let tuning = PyramidTuning {
        tiles: (1, 1),
        ..PyramidTuning::default()
    };
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 72, tuning).expect("tiled pyramid");
    let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("tiled pyramid decodes");
    assert_eq!(decoded.len(), frames.len());
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(
            frame_planes(f),
            recon_planes(&enc.gop, i),
            "tiled pyramid display {i}"
        );
    }
}

/// The split-group pyramid: same layout with `tile_groups = 2` —
/// every CODED frame carries two tile-group OBUs (no `OBU_FRAME`
/// anywhere), and the decoded pixels equal the single-group
/// pyramid's exactly (framing-only change).
#[test]
fn tiled_pyramid_split_groups_decode_identical() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| moving(192, 128, t)).collect();
    let single = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        72,
        PyramidTuning {
            tiles: (1, 1),
            ..PyramidTuning::default()
        },
    )
    .expect("single-group pyramid");
    let split = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        72,
        PyramidTuning {
            tiles: (1, 1),
            tile_groups: 2,
            ..PyramidTuning::default()
        },
    )
    .expect("split-group pyramid");
    let (frames_obus, _fhs, tgs) = obu_census(&split.gop.temporal_units);
    assert_eq!(frames_obus, 0, "split pyramid must not emit OBU_FRAME");
    // 6 coded frames (KEY + 5) × 2 groups.
    assert_eq!(tgs, 12, "two tile groups per coded frame");
    let a = decode_av1_spec(&single.gop.ivf_bytes).expect("single decodes");
    let b = decode_av1_spec(&split.gop.ivf_bytes).expect("split decodes");
    assert_eq!(a.len(), b.len());
    for (i, (fa, fb)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            frame_planes(fa),
            frame_planes(fb),
            "display {i}: split-group pyramid pixels"
        );
    }
}

/// `(0, 0)` / `tile_groups = 1` byte identity on the pyramid driver.
#[test]
fn pyramid_tiles_0_0_is_byte_identical() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| moving(128, 64, t)).collect();
    let a = encode_pyramid_gop_yuv420_with_q(&frames, 72).expect("default pyramid");
    let b = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 72, PyramidTuning::default())
        .expect("(0,0) pyramid");
    assert_eq!(a.ivf_bytes, b.gop.ivf_bytes);
}

// ---------------------------------------------------------------------
// Tiled temporal ladders.
// ---------------------------------------------------------------------

/// 3-layer, 8-frame ladder with 2×1 tiles and `tile_groups = 2`:
/// every §6.7.5 operating point decodes to exactly the surviving
/// frames' reconstructions.
#[test]
fn tiled_temporal_ladder_decodes_at_every_operating_point() {
    let layers = 3u8;
    let frames: Vec<Yuv420Frame> = (0..8).map(|t| moving(192, 128, t)).collect();
    let enc = encode_temporal_layered_gop_yuv420_with_q_tiles(&frames, 72, layers, (1, 0), 2)
        .expect("tiled ladder");
    let (frames_obus, _fhs, tgs) = obu_census(&enc.gop.temporal_units);
    assert_eq!(frames_obus, 0, "split ladder must not emit OBU_FRAME");
    assert_eq!(tgs, 16, "two tile groups per coded frame");
    for k in 0..layers {
        let decoded = decode_av1_spec_at_operating_point(&enc.gop.ivf_bytes, k)
            .unwrap_or_else(|e| panic!("op {k}: decode failed: {e:?}"));
        let keep_max = layers - 1 - k;
        let surviving: Vec<usize> = (0..frames.len())
            .filter(|&i| temporal_layer_of(i, layers) <= keep_max)
            .collect();
        assert_eq!(decoded.len(), surviving.len(), "op {k}: frame count");
        for (f, &i) in decoded.iter().zip(&surviving) {
            assert_eq!(
                frame_planes(f),
                recon_planes(&enc.gop, i),
                "op {k}: display {i}"
            );
        }
    }
}

/// `(0, 0)` / `1` byte identity on the ladder driver.
#[test]
fn temporal_ladder_tiles_0_0_is_byte_identical() {
    let frames: Vec<Yuv420Frame> = (0..8).map(|t| moving(64, 64, t)).collect();
    let a = encode_temporal_layered_gop_yuv420_with_q(&frames, 72, 3).expect("default ladder");
    let b = encode_temporal_layered_gop_yuv420_with_q_tiles(&frames, 72, 3, (0, 0), 1)
        .expect("(0,0) ladder");
    assert_eq!(a.gop.ivf_bytes, b.gop.ivf_bytes);
}

// ---------------------------------------------------------------------
// Black-box revalidation hook.
// ---------------------------------------------------------------------

/// Env-gated dump (`AV1_TPS_DUMP_DIR`): tiled pyramid (single + split
/// groups) and the tiled ladder for the black-box reference decoders.
/// Inert otherwise.
#[test]
fn dump_tiled_pyramid_scalable_for_blackbox_validation() {
    let Ok(dir) = std::env::var("AV1_TPS_DUMP_DIR") else {
        return;
    };
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).expect("dump dir");
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| moving(192, 128, t)).collect();
    let single = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        72,
        PyramidTuning {
            tiles: (1, 1),
            ..PyramidTuning::default()
        },
    )
    .expect("single-group pyramid");
    std::fs::write(dir.join("tps-pyramid-tiled.ivf"), &single.gop.ivf_bytes).expect("write");
    let split = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        72,
        PyramidTuning {
            tiles: (1, 1),
            tile_groups: 2,
            ..PyramidTuning::default()
        },
    )
    .expect("split-group pyramid");
    std::fs::write(dir.join("tps-pyramid-tiled-g2.ivf"), &split.gop.ivf_bytes).expect("write");
    let ladder_frames: Vec<Yuv420Frame> = (0..8).map(|t| moving(192, 128, t)).collect();
    let ladder = encode_temporal_layered_gop_yuv420_with_q_tiles(&ladder_frames, 72, 3, (1, 0), 2)
        .expect("tiled ladder");
    std::fs::write(dir.join("tps-ladder-tiled-g2.ivf"), &ladder.gop.ivf_bytes).expect("write");
}
