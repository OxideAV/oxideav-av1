//! r453 — §5.9.2 `frame_refs_short_signaling` on the B-PYRAMID and
//! ADAPTIVE drivers.
//!
//! The r452 write twin covered the plain-GOP P chain. The pyramid
//! drivers now ride the same §7.8 lockstep: a coded role whose
//! `set_frame_refs()` derivation over the TRUE stored `RefOrderHint[]`
//! state still NAMES every slot its RD ladder searches adopts the
//! derived map — LAST / GOLDEN stay on the role's rotation slots
//! (§7.8 seeds them from the two explicit indices), the searched
//! backward slots (the mini-GOP ALT, the nearest backward midpoint,
//! the enclosing midpoint) are re-addressed onto the ordinals the
//! derivation gave them, and the header codes only `last_frame_idx` /
//! `gold_frame_idx` (7 wire bits replace the explicit 22). The
//! primary-reference candidates and the §6.8.14 donor settlement
//! resolve through the CODED map, so the whole out-of-order refresh
//! graph runs both sides of the §7.8 algorithm in lockstep.
//!
//! Coverage:
//! * a short-signaled pyramid decodes byte-exact through the spec
//!   driver on every display position (decoded-not-shown ALT / MID
//!   frames and their `show_existing_frame` outputs included),
//! * every non-KEY coded header off the wire carries the short shape,
//!   parsed back against the TRUE tracked `RefOrderHint[]` state: the
//!   f(1) set, the explicit indices naming forward slots, and the
//!   §7.8-derived backward ordinals landing on slots that really hold
//!   future frames,
//! * the adaptive driver rides the same adoption,
//! * `short_ref_signaling: false` keeps the explicit 22-bit shape on
//!   every frame,
//! * an env-gated staging dump feeds black-box reference-decoder
//!   validation and corpus pinning.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.2, §7.8.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_adaptive_gop_yuv420_with_q_tuned, encode_pyramid_gop_yuv420_with_q_tuned,
    AdaptiveTuning, PyramidTuning, Yuv420Frame,
};
use oxideav_av1::frame_header::{FrameHeader, FrameType, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Deterministic textured frame with per-frame translation.
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

fn pyramid(short_ref_signaling: bool) -> PyramidTuning {
    PyramidTuning {
        short_ref_signaling,
        max_mini_gop: 4,
        // Clean synthetic content — keep the §5.9.30 probe out of the
        // axis under test.
        film_grain: false,
        ..PyramidTuning::default()
    }
}

/// Every coded frame header off the wire in temporal-unit order
/// (show-existing headers skipped), parsed against the tracked §7.20
/// `RefOrderHint[]` / size state so a short-signaled header's §7.8
/// derivation runs over the TRUE slot hints. Returns each header
/// paired with a snapshot of the hint state it was parsed under.
fn wire_headers(tus: &[Vec<u8>]) -> Vec<(FrameHeader, [u32; 8])> {
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
                    let fh = oxideav_av1::frame_header::parse_frame_header_with_refs(
                        desc.payload,
                        sq,
                        &ref_info,
                    )
                    .expect("frame header parses");
                    if fh.show_existing_frame {
                        continue;
                    }
                    let hints = ref_info.order_hint;
                    // §7.20 reference_frame_update() — the parts the
                    // next header's parse consults.
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
                    out.push((fh, hints));
                }
                _ => {}
            }
        }
    }
    out
}

fn assert_decodes_to_recon(ivf: &[u8], recon: &[oxideav_av1::encoder::GopFrameRecon]) {
    let decoded = decode_av1_spec(ivf).expect("spec driver decodes");
    assert_eq!(decoded.len(), recon.len());
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], recon[idx].y, "frame {idx}: luma");
        assert_eq!(f.planes[1], recon[idx].u, "frame {idx}: U");
        assert_eq!(f.planes[2], recon[idx].v, "frame {idx}: V");
    }
}

/// The short-shape wire audit shared by both drivers: every non-KEY
/// coded header rides the §5.9.2 short arm, the explicit indices name
/// slots holding FORWARD frames (§7.8 conformance), and every derived
/// backward ordinal lands on a slot whose stored hint really is a
/// future frame.
fn assert_short_wire_shape(tus: &[Vec<u8>], coded_frames: usize) {
    let headers = wire_headers(tus);
    assert_eq!(headers.len(), coded_frames);
    let mut hidden = 0usize;
    for (idx, (fh, hints)) in headers.iter().enumerate() {
        if idx == 0 {
            assert_eq!(fh.frame_type, FrameType::Key);
            continue;
        }
        if !fh.show_frame {
            hidden += 1;
        }
        let refs = fh.inter_refs.as_ref().expect("inter frame carries refs");
        assert!(
            refs.frame_refs_short_signaling,
            "frame {idx} (order_hint {}): short shape expected",
            fh.order_hint
        );
        let last = refs.last_frame_idx.expect("short => last_frame_idx");
        let gold = refs.gold_frame_idx.expect("short => gold_frame_idx");
        assert_eq!(refs.ref_frame_idx[0], last, "frame {idx}: LAST seed");
        assert_eq!(refs.ref_frame_idx[3], gold, "frame {idx}: GOLDEN seed");
        // §7.8 bitstream conformance: both explicit slots hold
        // frames coded BEFORE this one in output order.
        assert!(
            hints[last as usize] < fh.order_hint,
            "frame {idx}: lastOrderHint < curFrameHint"
        );
        assert!(
            hints[gold as usize] < fh.order_hint,
            "frame {idx}: goldOrderHint < curFrameHint"
        );
        assert!(refs.ref_frame_idx.iter().all(|&s| s < 8));
    }
    assert!(
        hidden > 0,
        "a 4-deep mini-GOP codes decoded-not-shown frames"
    );
}

/// A short-signaled 9-frame pyramid decodes byte-exact on every
/// display position and every non-KEY header carries the short shape.
#[test]
fn pyramid_short_refs_round_trip_and_wire_shape() {
    let frames: Vec<Yuv420Frame> = (0..9).map(|t| moving_frame(96, 80, t)).collect();
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 60, pyramid(true))
        .expect("pyramid encodes");
    assert_decodes_to_recon(&enc.gop.ivf_bytes, &enc.gop.recon);
    assert_short_wire_shape(&enc.gop.temporal_units, 9);
}

/// The adaptive driver rides the same adoption.
#[test]
fn adaptive_short_refs_round_trip_and_wire_shape() {
    let frames: Vec<Yuv420Frame> = (0..8).map(|t| moving_frame(96, 80, t)).collect();
    let tuning = AdaptiveTuning {
        pyramid: pyramid(true),
        ..AdaptiveTuning::default()
    };
    let enc =
        encode_adaptive_gop_yuv420_with_q_tuned(&frames, 60, tuning).expect("adaptive encodes");
    assert_decodes_to_recon(&enc.gop.ivf_bytes, &enc.gop.recon);
    let headers = wire_headers(&enc.gop.temporal_units);
    assert_eq!(headers.len(), 8);
    for (idx, (fh, _)) in headers.iter().enumerate().skip(1) {
        let refs = fh.inter_refs.as_ref().expect("inter frame carries refs");
        assert!(
            refs.frame_refs_short_signaling,
            "frame {idx} (order_hint {}): short shape expected",
            fh.order_hint
        );
    }
}

/// `short_ref_signaling: false` keeps the explicit 22-bit shape on
/// every frame (the A/B baseline), and still decodes byte-exact.
#[test]
fn pyramid_short_refs_off_keeps_the_explicit_shape() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 64, t)).collect();
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 60, pyramid(false))
        .expect("pyramid encodes");
    assert_decodes_to_recon(&enc.gop.ivf_bytes, &enc.gop.recon);
    for (idx, (fh, _)) in wire_headers(&enc.gop.temporal_units)
        .iter()
        .enumerate()
        .skip(1)
    {
        let refs = fh.inter_refs.as_ref().expect("inter frame carries refs");
        assert!(
            !refs.frame_refs_short_signaling,
            "frame {idx}: explicit shape expected on the off arm"
        );
        assert!(refs.last_frame_idx.is_none());
        assert!(refs.gold_frame_idx.is_none());
    }
}

/// The short arm composes with a tiled layout and the §6.8.14 donor
/// election (the settlement resolves the consumed slot through the
/// CODED map).
#[test]
fn pyramid_short_refs_compose_with_tiles_and_donor_election() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| moving_frame(128, 96, t)).collect();
    let tuning = PyramidTuning {
        tiles: (1, 0),
        ..pyramid(true)
    };
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 72, tuning).expect("pyramid encodes");
    assert_decodes_to_recon(&enc.gop.ivf_bytes, &enc.gop.recon);
    assert_short_wire_shape(&enc.gop.temporal_units, 6);
}

/// Env-gated staging dump (`OXIDEAV_AV1_PYR_SHORTREFS_DIR`): the
/// 9-frame 96×80 q60 short-signaled pyramid as `input.ivf` plus the
/// spec driver's `expected.yuv` (planar I420, display order) for
/// black-box reference-decoder validation and corpus pinning.
#[test]
fn pyramid_short_refs_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_PYR_SHORTREFS_DIR") else {
        eprintln!("OXIDEAV_AV1_PYR_SHORTREFS_DIR unset — skipping the staging dump");
        return;
    };
    let frames: Vec<Yuv420Frame> = (0..9).map(|t| moving_frame(96, 80, t)).collect();
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 60, pyramid(true))
        .expect("pyramid encodes");
    let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("decodes");
    let mut yuv = Vec::new();
    for f in &decoded {
        for p in &f.planes {
            yuv.extend_from_slice(p);
        }
    }
    std::fs::create_dir_all(&dir).expect("staging dir");
    std::fs::write(format!("{dir}/input.ivf"), &enc.gop.ivf_bytes).expect("write ivf");
    std::fs::write(format!("{dir}/expected.yuv"), &yuv).expect("write yuv");
}
