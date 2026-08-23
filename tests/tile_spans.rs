//! r450 — §5.11.1 EXPLICIT tile spans on single-group frames:
//! `tile_start_and_end_present_flag = 1` with the full
//! `tg_start = 0 ..= tg_end = NumTiles − 1` range coded.
//!
//! The flag-1 shape was READ since r433 (multi-group frames code it
//! on every split group) but never EMITTED on a whole-frame group;
//! `GopTuning::tile_spans` closes the write gap. Because §5.10
//! requires `tile_start_and_end_present_flag == 0` inside an
//! `OBU_FRAME`, the arm takes the split packaging — a standalone
//! `OBU_FRAME_HEADER` (§5.3.4 trailing bits) followed by ONE
//! `OBU_TILE_GROUP` whose prologue codes the span. The per-tile
//! entropy payloads are byte-identical to the flag-0 shape — only
//! the OBU framing and the group prologue change, which these
//! witnesses pin: the wire shape (flag + span bits parsed off the
//! group body), the decode (byte-exact through the spec driver on
//! KEY + P chains), and the payload identity against the flag-0
//! twin.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.11.1, §5.10, §6.10.1.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning, Yuv420Frame};
use oxideav_av1::obu::{ObuIter, ObuType};

fn moving_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            let (si, sj) = (i + t, j + 2 * t);
            f.y[i * wu + j] = ((si * 5 + sj * 3 + (si / 16) * (sj / 16)) % 256) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = ((128 + i * 2 + j + t) % 256) as u8;
            f.v[i * cw + j] = ((64 + i + j * 2 + 2 * t) % 256) as u8;
        }
    }
    f
}

fn tuned(tile_spans: bool, tiles: (u32, u32)) -> GopTuning {
    GopTuning {
        tile_spans,
        tiles,
        ..GopTuning::default()
    }
}

fn encode(frames: &[Yuv420Frame], q: u8, tuning: GopTuning) -> oxideav_av1::encoder::TunedGop {
    encode_gop_yuv420_with_q_seg_tuned(frames, q, &[], tuning).expect("GOP encodes")
}

/// Per-TU OBU walk: `(obu_type, payload)` pairs.
fn obus(tu: &[u8]) -> Vec<(ObuType, Vec<u8>)> {
    ObuIter::new(tu)
        .map(|d| {
            let d = d.expect("own stream walks");
            (d.obu_type, d.payload.to_vec())
        })
        .collect()
}

/// The spans arm re-packages every frame as `OBU_FRAME_HEADER` +
/// ONE `OBU_TILE_GROUP` whose prologue codes
/// `tile_start_and_end_present_flag = 1, tg_start = 0, tg_end =
/// NumTiles − 1`; the entropy payloads are byte-identical to the
/// flag-0 twin.
#[test]
fn tile_spans_code_the_flag_and_keep_payloads_identical() {
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| moving_frame(128, 128, t)).collect();
    // (1, 1) → a 2×2 §5.9.15 uniform layout: tileBits = 2.
    let on = encode(&frames, 100, tuned(true, (1, 1)));
    let off = encode(&frames, 100, tuned(false, (1, 1)));
    assert_eq!(on.gop.temporal_units.len(), off.gop.temporal_units.len());
    for (k, (tu_on, tu_off)) in on
        .gop
        .temporal_units
        .iter()
        .zip(&off.gop.temporal_units)
        .enumerate()
    {
        let o_on = obus(tu_on);
        let o_off = obus(tu_off);
        // The baseline packs one OBU_FRAME; the spans arm splits it.
        let (fh_on, tg_on) = {
            let fh = o_on
                .iter()
                .find(|(t, _)| *t == ObuType::FrameHeader)
                .unwrap_or_else(|| panic!("TU {k}: spans arm must emit OBU_FRAME_HEADER"));
            let tgs: Vec<_> = o_on
                .iter()
                .filter(|(t, _)| *t == ObuType::TileGroup)
                .collect();
            assert_eq!(tgs.len(), 1, "TU {k}: exactly ONE tile group");
            (fh.1.clone(), tgs[0].1.clone())
        };
        assert!(
            !o_off.iter().any(|(t, _)| *t == ObuType::FrameHeader),
            "TU {k}: baseline keeps the OBU_FRAME packing"
        );
        let frame_off = &o_off
            .iter()
            .find(|(t, _)| *t == ObuType::Frame)
            .expect("baseline OBU_FRAME")
            .1;
        // §5.11.1 prologue on the spans body: flag(1) = 1,
        // tg_start f(2) = 0, tg_end f(2) = 3 → the first byte is
        // 0b1_00_11_000 (byte_alignment pads with zeros).
        assert_eq!(
            tg_on[0], 0x98,
            "TU {k}: coded span prologue (flag=1, tg 0..=3, aligned)"
        );
        // Payload identity: past the 1-byte prologue, the spans body
        // equals the baseline's tile-group body. The baseline
        // OBU_FRAME carries the (identical) header bits first; its
        // tile-group body starts right after — locate it by suffix
        // match against the spans body's tail.
        let tail = &tg_on[1..];
        assert!(
            frame_off.ends_with(tail),
            "TU {k}: entropy payloads must be byte-identical past the prologue"
        );
        // The header bits are the same modulo the §5.3.4 trailing
        // bits on the standalone OBU_FRAME_HEADER: the OBU_FRAME's
        // header prefix must be a prefix of (or equal to) the
        // standalone payload's un-padded bits. Cheap witness: the
        // standalone header starts with the same bytes as the
        // OBU_FRAME body up to the last (alignment) byte.
        let n = fh_on.len().saturating_sub(1);
        assert!(
            n == 0 || frame_off[..n] == fh_on[..n],
            "TU {k}: identical header bits under both packagings"
        );
    }
}

/// KEY + P chains with the spans arm decode byte-exact through the
/// spec driver (lossy and lossless), across uniform layouts.
#[test]
fn tile_spans_round_trip_byte_exact() {
    for &(w, h, tiles, q) in &[
        (128u32, 128u32, (1u32, 1u32), 100u8),
        (128, 64, (1, 0), 72),
        (128, 64, (1, 0), 0),
    ] {
        let frames: Vec<Yuv420Frame> = (0..4).map(|t| moving_frame(w, h, t)).collect();
        let enc = encode(&frames, q, tuned(true, tiles));
        let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("spec driver decodes spans GOP");
        assert_eq!(decoded.len(), frames.len());
        for (idx, f) in decoded.iter().enumerate() {
            let rc = &enc.gop.recon[idx];
            assert_eq!(f.planes[0], rc.y, "{w}x{h} q{q} frame {idx}: luma");
            assert_eq!(f.planes[1], rc.u, "{w}x{h} q{q} frame {idx}: U");
            assert_eq!(f.planes[2], rc.v, "{w}x{h} q{q} frame {idx}: V");
            if q == 0 {
                assert_eq!(f.planes[0], frames[idx].y, "lossless frame {idx}");
            }
        }
    }
}

/// Inert shapes: single-tile frames never code the flag (§5.11.1
/// reads it only when `NumTiles > 1`), and `tile_groups > 1` keeps
/// the r433 multi-group split — both stay bit-identical to their
/// `tile_spans = false` twins.
#[test]
fn tile_spans_inert_on_single_tile_and_multi_group() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| moving_frame(64, 64, t)).collect();
    let on = encode(&frames, 100, tuned(true, (0, 0)));
    let off = encode(&frames, 100, tuned(false, (0, 0)));
    assert_eq!(
        on.gop.ivf_bytes, off.gop.ivf_bytes,
        "single-tile frames must ignore tile_spans"
    );
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| moving_frame(128, 128, t)).collect();
    let mk = |spans| GopTuning {
        tile_spans: spans,
        tiles: (1, 1),
        tile_groups: 2,
        ..GopTuning::default()
    };
    let on = encode(&frames, 100, mk(true));
    let off = encode(&frames, 100, mk(false));
    assert_eq!(
        on.gop.ivf_bytes, off.gop.ivf_bytes,
        "multi-group frames already code the flag per split group"
    );
}

/// Env-gated staging dump (`OXIDEAV_AV1_TILE_SPANS_DIR`): a
/// spans-coded GOP plus expected YUV for black-box
/// reference-decoder validation and corpus pinning. Inert
/// otherwise.
#[test]
fn tile_spans_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_TILE_SPANS_DIR") else {
        eprintln!("OXIDEAV_AV1_TILE_SPANS_DIR unset — skipping the tile-spans staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| moving_frame(128, 128, t)).collect();
    let enc = encode(&frames, 100, tuned(true, (1, 1)));
    std::fs::write(
        root.join("gop-128x128-q100-tilespans.ivf"),
        &enc.gop.ivf_bytes,
    )
    .expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &enc.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("gop-128x128-q100-tilespans.yuv"), &yuv).expect("write yuv");
}
