//! r452 — §5.9.2 `frame_refs_short_signaling` WRITE twin.
//!
//! With `enable_order_hint`, an inter frame header may code
//! `frame_refs_short_signaling = 1`: only `last_frame_idx` and
//! `gold_frame_idx` (3 bits each) go on the wire and the DECODER
//! derives the full seven-entry `ref_frame_idx[]` through §7.8
//! `set_frame_refs()` from its stored `RefOrderHint[]` — 7 bits
//! replace the explicit 22. The encoder adopts the same derivation as
//! its reference map (`GopTuning::short_ref_signaling`), so both
//! sides run the identical §7.8 algorithm in lockstep: LAST / GOLDEN
//! stay on the slots the RD ladder codes (§7.8 seeds them from the
//! explicit indices), the unsearched ordinals land on the derived
//! slots, and every downstream twin (§7.8 sign bias, §5.9.22 skip
//! mode, §7.9 projection) runs over the adopted map.
//!
//! Coverage:
//! * a short-signaled GOP decodes byte-exact through the spec driver
//!   (lossy and lossless),
//! * the emitted headers carry the short shape, parsed back off the
//!   wire against the TRUE tracked `RefOrderHint[]` state: the f(1)
//!   set, the two explicit indices naming the LAST / GOLDEN rotation
//!   slots, and the decoder-derived map seeding those ordinals,
//! * SWITCH-cadence positions keep the explicit shape (they re-anchor
//!   every slot), and the tuning composes with tiles,
//! * the `false` tuning is bit-identical to the default GOP,
//! * an env-gated staging dump feeds black-box reference-decoder
//!   validation and corpus pinning.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.2, §7.8.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning, Yuv420Frame};
use oxideav_av1::frame_header::{FrameHeader, FrameType, RefInfo, ALL_FRAMES_PUB};
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

fn tuned(short_ref_signaling: bool, s_frame_period: u32, tiles: (u32, u32)) -> GopTuning {
    GopTuning {
        short_ref_signaling,
        s_frame_period,
        tiles,
        ..GopTuning::default()
    }
}

/// Encode, decode through the spec driver, assert byte-exactness
/// against the encoder recon (and the input at q 0).
fn assert_round_trip(frames: &[Yuv420Frame], q: u8, tuning: GopTuning) -> Vec<u8> {
    let enc = encode_gop_yuv420_with_q_seg_tuned(frames, q, &[], tuning).expect("GOP encodes");
    let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("spec driver decodes own GOP");
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.gop.recon[idx];
        assert_eq!(f.planes[0], rc.y, "frame {idx}: luma decode != recon");
        assert_eq!(f.planes[1], rc.u, "frame {idx}: U decode != recon");
        assert_eq!(f.planes[2], rc.v, "frame {idx}: V decode != recon");
        if q == 0 {
            assert_eq!(f.planes[0], frames[idx].y, "lossless frame {idx} luma");
            assert_eq!(f.planes[1], frames[idx].u, "lossless frame {idx} U");
            assert_eq!(f.planes[2], frames[idx].v, "lossless frame {idx} V");
        }
    }
    enc.gop.ivf_bytes
}

/// Parse every coded frame header off the wire in temporal-unit
/// order, tracking the §7.20 `RefOrderHint[]` / size state the way
/// the decoder does so a short-signaled header's §7.8 derivation
/// runs over the TRUE slot hints.
fn wire_headers(temporal_units: &[Vec<u8>]) -> Vec<FrameHeader> {
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    let mut out = Vec::new();
    for tu in temporal_units {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("own stream walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let fh = oxideav_av1::frame_header::parse_frame_header_with_refs(
                        desc.payload,
                        seq.as_ref().expect("SH precedes frames"),
                        &refinfo,
                    )
                    .expect("frame header parses");
                    // §7.20 reference_frame_update() — the parts the
                    // next header's parse consults.
                    let fs = fh.frame_size.as_ref().expect("coded frame has a size");
                    for i in 0..8 {
                        if (fh.refresh_frame_flags >> i) & 1 != 0 {
                            refinfo.valid[i] = true;
                            refinfo.order_hint[i] = fh.order_hint;
                            refinfo.upscaled_width[i] = fs.upscaled_width;
                            refinfo.frame_height[i] = fs.frame_height;
                            refinfo.render_width[i] = fs.render_width;
                            refinfo.render_height[i] = fs.render_height;
                            refinfo.frame_type_is_key[i] = fh.frame_type == FrameType::Key;
                        }
                    }
                    out.push(fh);
                }
                _ => {}
            }
        }
    }
    out
}

/// A 7-frame short-signaled GOP decodes byte-exact, lossy and
/// lossless.
#[test]
fn short_ref_signaling_round_trips_byte_exact() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 64, t)).collect();
    assert_round_trip(&frames, 80, tuned(true, 0, (0, 0)));
    let small: Vec<Yuv420Frame> = (0..5).map(|t| moving_frame(64, 64, t)).collect();
    assert_round_trip(&small, 0, tuned(true, 0, (0, 0)));
}

/// Every inter header codes the short shape and the decoder's §7.8
/// derivation seeds LAST / GOLDEN from the two explicit indices —
/// the two-slot rotation the GOP driver runs.
#[test]
fn short_ref_headers_carry_the_derived_map() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 64, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 80, &[], tuned(true, 0, (0, 0)))
        .expect("GOP encodes");
    let headers = wire_headers(&enc.gop.temporal_units);
    assert_eq!(headers.len(), 7);
    for (idx, fh) in headers.iter().enumerate() {
        if idx == 0 {
            assert_eq!(fh.frame_type, FrameType::Key);
            continue;
        }
        let p = idx as u32;
        let refs = fh.inter_refs.as_ref().expect("inter frame carries refs");
        assert!(
            refs.frame_refs_short_signaling,
            "frame {idx}: frame_refs_short_signaling must be coded as 1"
        );
        let last = (p & 1) as u8;
        let gold = ((p - 1) & 1) as u8;
        assert_eq!(
            refs.last_frame_idx,
            Some(last),
            "frame {idx}: last_frame_idx"
        );
        assert_eq!(
            refs.gold_frame_idx,
            Some(gold),
            "frame {idx}: gold_frame_idx"
        );
        // §7.8 seeds ref_frame_idx[ LAST_FRAME - LAST_FRAME ] and
        // ref_frame_idx[ GOLDEN_FRAME - LAST_FRAME ] from the explicit
        // indices; the derived entries all name real slots.
        assert_eq!(
            refs.ref_frame_idx[0], last,
            "frame {idx}: derived LAST slot"
        );
        assert_eq!(
            refs.ref_frame_idx[3], gold,
            "frame {idx}: derived GOLDEN slot"
        );
        assert!(refs.ref_frame_idx.iter().all(|&s| s < 8));
        // With every stored hint BEHIND the current frame (a forward-
        // only GOP) §7.8 fills the remaining ordinals from the forward
        // list in anti-chronological order — LAST2 lands on a slot
        // holding the most recent remaining hint, never on LAST's own
        // slot (it is marked used).
        assert_ne!(
            refs.ref_frame_idx[1], last,
            "frame {idx}: LAST2 is not LAST's slot"
        );
        assert_ne!(
            refs.ref_frame_idx[1], gold,
            "frame {idx}: LAST2 is not GOLDEN's slot"
        );
    }
}

/// The tuning composes with a SWITCH cadence and a tiled layout:
/// S-frames keep the explicit 22-bit shape (§5.9.2 re-anchors every
/// slot there), plain P-frames ride the short arm, and the stream
/// decodes byte-exact.
#[test]
fn short_ref_signaling_composes_with_s_frames_and_tiles() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(128, 64, t)).collect();
    let tuning = tuned(true, 3, (1, 0));
    let ivf = assert_round_trip(&frames, 100, tuning);
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 100, &[], tuned(true, 3, (1, 0)))
        .expect("GOP encodes");
    assert_eq!(enc.gop.ivf_bytes, ivf, "encode is deterministic");
    let headers = wire_headers(&enc.gop.temporal_units);
    for (idx, fh) in headers.iter().enumerate().skip(1) {
        let refs = fh.inter_refs.as_ref().expect("inter frame carries refs");
        if idx % 3 == 0 {
            assert_eq!(
                fh.frame_type,
                FrameType::Switch,
                "frame {idx} is the S-frame"
            );
            assert_eq!(fh.refresh_frame_flags, ALL_FRAMES_PUB);
            assert!(
                !refs.frame_refs_short_signaling,
                "frame {idx}: the SWITCH frame keeps the explicit map"
            );
        } else {
            assert!(
                refs.frame_refs_short_signaling,
                "frame {idx}: plain P-frame rides the short arm"
            );
        }
    }
}

/// `short_ref_signaling: false` is the A/B baseline — bit-identical
/// to the default tuning.
#[test]
fn short_ref_signaling_off_is_bit_identical_to_default() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| moving_frame(64, 64, t)).collect();
    let base = encode_gop_yuv420_with_q_seg_tuned(&frames, 80, &[], GopTuning::default())
        .expect("GOP encodes");
    let off = encode_gop_yuv420_with_q_seg_tuned(&frames, 80, &[], tuned(false, 0, (0, 0)))
        .expect("GOP encodes");
    assert_eq!(base.gop.ivf_bytes, off.gop.ivf_bytes);
}

/// Env-gated staging dump (`OXIDEAV_AV1_SHORTREFS_DIR`): the 7-frame
/// 96×64 q80 short-signaled GOP as `input.ivf` plus the spec driver's
/// `expected.yuv` (planar I420, frame-major) for black-box
/// reference-decoder validation and corpus pinning.
#[test]
fn short_ref_signaling_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SHORTREFS_DIR") else {
        eprintln!("OXIDEAV_AV1_SHORTREFS_DIR unset — skipping the staging dump");
        return;
    };
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 64, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 80, &[], tuned(true, 0, (0, 0)))
        .expect("GOP encodes");
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
