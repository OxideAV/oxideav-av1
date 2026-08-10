//! r436 — §6.8.14 `context_update_tile_id` ELECTION: on multi-tile
//! GOPs each P-frame replays its committed trees from EVERY tile of
//! its primary frame's frame-end CDF states and keeps the start
//! state realizing the smallest §5.11.1 body; a win patches the
//! primary frame's already-emitted fixed-width field in place.
//!
//! What these tests pin:
//!
//!   * The election FIRES on tile-heterogeneous content (a flat
//!     tile 0 beside textured tiles adapts tile-0 CDFs away from
//!     the frame's real statistics — a later tile's state prices
//!     the next frame smaller), and the patched field is on the
//!     wire (header audit).
//!   * §6.8.21/§8.4 semantics survive the patch: the whole stream
//!     still decodes BIT-EXACT through the in-tree spec driver —
//!     the decoder loads `Saved*` state from the PATCHED tile id,
//!     so any encoder/decoder disagreement about the donation
//!     desyncs the §8.3.1 chain and breaks the pixel match.
//!   * The election never loses bytes: every P-frame's realized
//!     size is `<=` its own tile-0-donation arm by construction
//!     (the committed arm is candidate 0 of the comparison), and
//!     the knob off (`ctx_update_elect: false`) reproduces the
//!     pre-r436 stream bit for bit.
//!   * Single-tile layouts are inert (no field on the wire).
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.15, §6.8.14, §6.8.21,
//! §8.4.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning, Yuv420Frame};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Left half flat (tile 0 of a `(1, 0)` layout), right half strongly
/// textured and moving — tile 0's adapted CDFs converge on
/// skip/flat statistics while the frame's real coding work happens
/// in tile 1.
fn hetero_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
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

fn assert_decodes_to_recons(name: &str, enc: &oxideav_av1::encoder::TunedGop) {
    let frames = oxideav_av1::decode_av1(&enc.gop.ivf_bytes)
        .unwrap_or_else(|e| panic!("{name}: decode: {e:?}"));
    assert_eq!(frames.len(), enc.gop.recon.len(), "{name}: frame count");
    for (i, f) in frames.iter().enumerate() {
        match f {
            Frame::Spec(s) => {
                assert_eq!(s.planes[0], enc.gop.recon[i].y, "{name}: frame {i} luma");
                assert_eq!(s.planes[1], enc.gop.recon[i].u, "{name}: frame {i} U");
                assert_eq!(s.planes[2], enc.gop.recon[i].v, "{name}: frame {i} V");
            }
            other => panic!("{name}: non-Spec frame {other:?}"),
        }
    }
}

/// The wire-side `context_update_tile_id` of every frame in the
/// stream, in decode order (parsed from each frame's header).
fn wire_ctx_ids(enc: &oxideav_av1::encoder::TunedGop, w: u32, h: u32) -> Vec<u32> {
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    for i in 0..8 {
        refinfo.valid[i] = true;
        refinfo.upscaled_width[i] = w;
        refinfo.frame_height[i] = h;
        refinfo.render_width[i] = w;
        refinfo.render_height[i] = h;
    }
    let mut out = Vec::new();
    for tu in &enc.gop.temporal_units {
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
                    out.push(fh.tile_info.expect("tiled stream").context_update_tile_id);
                }
                _ => {}
            }
        }
    }
    out
}

/// The election fires on heterogeneous two-column tiles, the patched
/// ids are on the wire, and the stream decodes bit-exact through the
/// spec driver (which loads the §8.4 donation from the PATCHED id).
#[test]
fn ctx_update_election_fires_and_patched_stream_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| hetero_frame(128, 64, t)).collect();
    let on = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        80,
        &[],
        GopTuning {
            tiles: (1, 0),
            ..GopTuning::default()
        },
    )
    .expect("election encode");
    assert!(
        on.ctx_donor_elections
            .iter()
            .any(|e| matches!(e, Some(t) if *t != 0)),
        "designed content must elect a non-zero donor at least once: {:?}",
        on.ctx_donor_elections
    );
    assert_decodes_to_recons("ctx-elect-on", &on);

    // Wire audit: frame k's coded id equals the election its CONSUMER
    // (frame k+1) reported; the last frame's donation is unconsumed
    // and stays 0.
    let ids = wire_ctx_ids(&on, 128, 64);
    assert_eq!(ids.len(), frames.len());
    for (k, &id) in ids.iter().enumerate() {
        let expect = on
            .ctx_donor_elections
            .get(k) // consumer of frame k is P-frame index k (1-based frame k+1)
            .copied()
            .flatten()
            .unwrap_or(0);
        assert_eq!(id, expect, "frame {k} wire ctx id");
    }
}

/// The knob off keeps every donation at tile 0 on the wire, decodes
/// bit-exact, and the ON stream never spends MORE bytes on any
/// P-frame than the OFF stream spends on the same frame position
/// with the same donation chain prefix — measured here end to end:
/// the designed content's ON stream is strictly smaller.
#[test]
fn ctx_update_election_off_is_tile0_and_on_saves_bytes() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| hetero_frame(128, 64, t)).collect();
    let on = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        80,
        &[],
        GopTuning {
            tiles: (1, 0),
            ..GopTuning::default()
        },
    )
    .expect("on");
    let off = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        80,
        &[],
        GopTuning {
            tiles: (1, 0),
            ctx_update_elect: false,
            ..GopTuning::default()
        },
    )
    .expect("off");
    assert!(off.ctx_donor_elections.iter().all(|e| e.is_none()));
    assert!(wire_ctx_ids(&off, 128, 64).iter().all(|&id| id == 0));
    assert_decodes_to_recons("ctx-elect-off", &off);
    assert!(
        on.gop.ivf_bytes.len() < off.gop.ivf_bytes.len(),
        "designed content must realize a net saving: on {} vs off {}",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
}

/// Single-tile layouts carry no `context_update_tile_id` field — the
/// knob is inert and the streams are bit-identical.
#[test]
fn ctx_update_election_inert_on_single_tile() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| hetero_frame(64, 64, t)).collect();
    let on = encode_gop_yuv420_with_q_seg_tuned(&frames, 80, &[], GopTuning::default())
        .expect("single-tile on");
    let off = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        80,
        &[],
        GopTuning {
            ctx_update_elect: false,
            ..GopTuning::default()
        },
    )
    .expect("single-tile off");
    assert_eq!(on.gop.ivf_bytes, off.gop.ivf_bytes);
    assert!(on.ctx_donor_elections.iter().all(|e| e.is_none()));
}

/// Env-gated staging dump (`OXIDEAV_AV1_CTX_ELECT_DIR`): the elected
/// stream + expected YUV for black-box reference-decoder validation
/// and corpus pinning.
#[test]
fn ctx_update_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_CTX_ELECT_DIR") else {
        eprintln!("OXIDEAV_AV1_CTX_ELECT_DIR unset — skipping the ctx-update staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| hetero_frame(128, 64, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        80,
        &[],
        GopTuning {
            tiles: (1, 0),
            ..GopTuning::default()
        },
    )
    .expect("election encode");
    assert!(
        enc.ctx_donor_elections
            .iter()
            .any(|e| matches!(e, Some(t) if *t != 0)),
        "staged stream must carry a non-zero elected donor"
    );
    let name = "gop-128x64-q80-ctx-update-elect";
    std::fs::write(root.join(format!("{name}.ivf")), &enc.gop.ivf_bytes).expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &enc.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
    // The measured-bytes justification: the same GOP under the
    // tile-0 baseline.
    let off = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        80,
        &[],
        GopTuning {
            tiles: (1, 0),
            ctx_update_elect: false,
            ..GopTuning::default()
        },
    )
    .expect("baseline encode");
    let per_tu: Vec<(usize, usize)> = enc
        .gop
        .temporal_units
        .iter()
        .zip(off.gop.temporal_units.iter())
        .map(|(a, b)| (a.len(), b.len()))
        .collect();
    std::fs::write(
        root.join(format!("{name}.elections.txt")),
        format!(
            "elections: {:?}\nivf bytes elected: {}\nivf bytes tile-0 baseline: {}\nper-frame (elected, baseline): {:?}\n",
            enc.ctx_donor_elections,
            enc.gop.ivf_bytes.len(),
            off.gop.ivf_bytes.len(),
            per_tu,
        ),
    )
    .expect("write elections");
}

// ---------------------------------------------------------------------
// r436 — the §6.8.14 election on the TEMPORAL LADDER driver.
// ---------------------------------------------------------------------

use oxideav_av1::encoder::encode_temporal_layered_gop_yuv420_with_q_tiles;

// ---------------------------------------------------------------------
// r439 — the §6.8.14 election on the B-PYRAMID driver.
// ---------------------------------------------------------------------

use oxideav_av1::encoder::{encode_pyramid_gop_yuv420_with_q_tuned, PyramidTuning};

/// Coded (non-`show_existing_frame`) frame headers' wire
/// `context_update_tile_id`s across a stream, in decode order.
fn wire_ctx_ids_tus(tus: &[Vec<u8>], w: u32, h: u32) -> Vec<u32> {
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    for i in 0..8 {
        refinfo.valid[i] = true;
        refinfo.upscaled_width[i] = w;
        refinfo.frame_height[i] = h;
        refinfo.render_width[i] = w;
        refinfo.render_height[i] = h;
    }
    let mut out = Vec::new();
    for tu in tus {
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
                    // `show_existing_frame` short headers carry no
                    // tile info — skip them.
                    if fh.show_existing_frame {
                        continue;
                    }
                    out.push(fh.tile_info.expect("tiled stream").context_update_tile_id);
                }
                _ => {}
            }
        }
    }
    out
}

/// The pyramid drives the §6.8.14 election through its out-of-order
/// refresh graph: the flat left tile beside the textured moving right
/// tile makes tile-1 donations price consumers smaller, the patched
/// fields land on the wire (inside multi-frame temporal units — ALT /
/// MID frames ride their B leaf's unit), and the whole stream decodes
/// display-order bit-exact through the spec driver.
#[test]
fn pyramid_ctx_update_election_fires_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| hetero_frame(128, 64, t)).collect();
    let on = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        80,
        PyramidTuning {
            tiles: (1, 0),
            ..PyramidTuning::default()
        },
    )
    .expect("pyramid election encode");
    let elected: Vec<&(u32, Option<u32>)> = on
        .ctx_donor_elections
        .iter()
        .filter(|(_, e)| e.is_some())
        .collect();
    assert!(
        !elected.is_empty(),
        "designed content must elect a non-zero donor at least once: {:?}",
        on.ctx_donor_elections
    );
    // Wire audit: every Some(t) election patched exactly one coded
    // frame's field to a non-zero id (a slot's donor set freezes at
    // first consumption, so patch targets are distinct).
    let ids = wire_ctx_ids_tus(&on.gop.temporal_units, 128, 64);
    assert_eq!(ids.len(), frames.len(), "one coded frame per display");
    assert_eq!(
        ids.iter().filter(|&&id| id != 0).count(),
        elected.len(),
        "patched wire ids must match the election trace: ids {ids:?} trace {:?}",
        on.ctx_donor_elections
    );
    // §6.8.21/§8.4 semantics survive the patches: display-order decode
    // equals the encoder reconstructions byte for byte.
    let decoded =
        oxideav_av1::decoder::decode_av1_spec(&on.gop.ivf_bytes).expect("patched pyramid decodes");
    assert_eq!(decoded.len(), frames.len());
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], on.gop.recon[i].y, "display {i} luma");
        assert_eq!(f.planes[1], on.gop.recon[i].u, "display {i} U");
        assert_eq!(f.planes[2], on.gop.recon[i].v, "display {i} V");
    }
    // The off arm keeps every donation at tile 0 and reports no
    // elections.
    let off = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        80,
        PyramidTuning {
            tiles: (1, 0),
            ctx_update_elect: false,
            ..PyramidTuning::default()
        },
    )
    .expect("pyramid baseline encode");
    assert!(off.ctx_donor_elections.is_empty());
    assert!(wire_ctx_ids_tus(&off.gop.temporal_units, 128, 64)
        .iter()
        .all(|&id| id == 0));
}

/// Single-tile pyramids carry no `context_update_tile_id` field — the
/// knob is inert and the streams are bit-identical.
#[test]
fn pyramid_ctx_update_election_inert_on_single_tile() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| hetero_frame(64, 64, t)).collect();
    let on = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 80, PyramidTuning::default())
        .expect("single-tile on");
    let off = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        80,
        PyramidTuning {
            ctx_update_elect: false,
            ..PyramidTuning::default()
        },
    )
    .expect("single-tile off");
    assert_eq!(on.gop.ivf_bytes, off.gop.ivf_bytes);
    assert!(on.ctx_donor_elections.is_empty());
}

/// The multi-consumer discipline on the §6.7.5 ladder: a slot's
/// donor set freezes at its FIRST consumption (several frames may
/// chain their §8.3.1 primary off the same slot — the KEY seeds all
/// eight), the patched fields land on the wire, and EVERY operating
/// point still decodes bit-exact — dropping any layer suffix leaves
/// each surviving frame's patched donation intact.
#[test]
fn ladder_ctx_update_election_survives_every_operating_point() {
    let frames: Vec<Yuv420Frame> = (0..8).map(|t| hetero_frame(128, 64, t)).collect();
    let enc = encode_temporal_layered_gop_yuv420_with_q_tiles(&frames, 80, 3, (1, 0), 1)
        .expect("tiled layered encode");

    // Wire audit: at least one frame's coded id was patched off 0.
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    for i in 0..8 {
        refinfo.valid[i] = true;
        refinfo.upscaled_width[i] = 128;
        refinfo.frame_height[i] = 64;
        refinfo.render_width[i] = 128;
        refinfo.render_height[i] = 64;
    }
    let mut ids = Vec::new();
    for tu in &enc.gop.temporal_units {
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
                    ids.push(fh.tile_info.expect("tiled stream").context_update_tile_id);
                }
                _ => {}
            }
        }
    }
    assert_eq!(ids.len(), frames.len());
    assert!(
        ids.iter().any(|&id| id != 0),
        "designed content must patch at least one ladder donation: {ids:?}"
    );

    // Every §6.7.5 operating point decodes its layer subset
    // bit-exact (op k keeps temporal ids <= 2 - k).
    for k in 0..3u8 {
        let out = oxideav_av1::decode_av1_at_operating_point(&enc.gop.ivf_bytes, k)
            .unwrap_or_else(|e| panic!("op {k} decode: {e:?}"));
        let keep: Vec<usize> = enc
            .temporal_ids
            .iter()
            .enumerate()
            .filter(|(_, &tid)| tid <= 2 - k)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(out.len(), keep.len(), "op {k} shown frames");
        for (f, &i) in out.iter().zip(keep.iter()) {
            match f {
                Frame::Spec(s) => {
                    assert_eq!(s.planes[0], enc.gop.recon[i].y, "op {k} frame {i} luma");
                    assert_eq!(s.planes[1], enc.gop.recon[i].u, "op {k} frame {i} U");
                    assert_eq!(s.planes[2], enc.gop.recon[i].v, "op {k} frame {i} V");
                }
                other => panic!("op {k}: non-Spec frame {other:?}"),
            }
        }
    }
}

/// Env-gated staging dump (`OXIDEAV_AV1_CTX_ELECT_PYR_DIR`): the
/// elected tiled-pyramid and tiled-SVC streams + expected YUV for
/// black-box reference-decoder validation and corpus pinning. Inert
/// otherwise.
#[test]
fn pyramid_svc_ctx_update_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_CTX_ELECT_PYR_DIR") else {
        eprintln!(
            "OXIDEAV_AV1_CTX_ELECT_PYR_DIR unset — skipping the pyramid/SVC ctx staging dump"
        );
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");

    // Tiled B-pyramid with elected donors.
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| hetero_frame(128, 64, t)).collect();
    let pyr = encode_pyramid_gop_yuv420_with_q_tuned(
        &frames,
        80,
        PyramidTuning {
            tiles: (1, 0),
            ..PyramidTuning::default()
        },
    )
    .expect("pyramid election encode");
    assert!(
        pyr.ctx_donor_elections.iter().any(|(_, e)| e.is_some()),
        "staged pyramid must elect at least one donor: {:?}",
        pyr.ctx_donor_elections
    );
    std::fs::write(
        root.join("pyr-128x64-q80-tiles-ctx-elect.ivf"),
        &pyr.gop.ivf_bytes,
    )
    .expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &pyr.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("pyr-128x64-q80-tiles-ctx-elect.yuv"), &yuv).expect("write yuv");
    std::fs::write(
        root.join("pyr-ctx-elections.txt"),
        format!(
            "elections (order_hint, donor): {:?}\n",
            pyr.ctx_donor_elections
        ),
    )
    .expect("write notes");

    // Tiled two-layer SVC with per-layer elected donors.
    let svc_layers = vec![
        (0..5).map(|t| hetero_frame(128, 64, t)).collect::<Vec<_>>(),
        (0..5)
            .map(|t| hetero_frame(256, 128, t))
            .collect::<Vec<_>>(),
    ];
    let svc = oxideav_av1::encoder::encode_spatial_layered_gop_yuv420_with_q_tiles(
        &svc_layers,
        80,
        Some(&[(1, 0), (1, 0)]),
        1,
    )
    .expect("tiled SVC encode");
    std::fs::write(root.join("svc-128-256-q80-ctx-elect.ivf"), &svc.ivf_bytes).expect("write ivf");
    // Full-interleave expected output (per instant: layer 0 then 1).
    let mut yuv: Vec<u8> = Vec::new();
    for i in 0..5 {
        for s in 0..2 {
            let rc = &svc.layer_recons[s][i];
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
    }
    std::fs::write(root.join("svc-128-256-q80-ctx-elect.yuv"), &yuv).expect("write yuv");
    // Base-layer (operating point 1) expected output.
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &svc.layer_recons[0] {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("svc-128-256-q80-ctx-elect-op1.yuv"), &yuv).expect("write yuv");
}
