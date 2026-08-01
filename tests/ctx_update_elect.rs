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
