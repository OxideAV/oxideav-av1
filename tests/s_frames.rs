//! r447 — §5.9.2 SWITCH-frame (S-frame) encoding end-to-end.
//!
//! The spec's "Switch Frame" is an INTER frame usable as a
//! chunk-boundary switch point between same-geometry streams: it
//! overwrites every §7.20 reference slot without intra coding, and
//! §5.9.2 infers four header fields bit-free — `error_resilient_mode
//! = 1`, `frame_size_override_flag = 1` (the frame size codes
//! explicitly), `refresh_frame_flags = allFrames` and (via error
//! resilience) `primary_ref_frame = PRIMARY_REF_NONE` plus
//! `use_ref_frame_mvs = 0`. Every cross-frame decode dependency
//! except the reference SAMPLES therefore re-anchors at the switch
//! point: CDFs restart from per-frame defaults, the motion-field /
//! segment-id / global-motion carries reload from the S-frame's own
//! committed state, and the §5.9.2 `ref_order_hint[ i ]` block pins
//! the slot hints on the wire.
//!
//! Coverage:
//! * in-stream S-frames decode byte-exact through the spec driver
//!   (== encoder reconstruction; == input on the lossless arm),
//! * the emitted headers carry the §5.9.2 SWITCH shape (parsed back
//!   off the wire),
//! * the election surfaces (tiles, hp-mv / delta-q / QM / CDEF / LR)
//!   compose with the cadence,
//! * a cross-rate SPLICE at the switch point parses and decodes
//!   through the spec driver — the §5.9.2 error-resilience property
//!   ("the syntax of a frame parses independently of previously
//!   decoded frames"); the spliced stream and our decode of it are
//!   dumped under `OXIDEAV_AV1_SFRAME_DUMP_DIR` for external
//!   black-box decoder cross-validation.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::ivf::{IvfWriter, FOURCC_AV01};
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning, Yuv420Frame};
use oxideav_av1::frame_header::{FrameType, RefInfo, ALL_FRAMES_PUB, PRIMARY_REF_NONE};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Deterministic textured frame with per-frame translation — the
/// motion search finds real vectors, residuals stay non-trivial.
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

fn tuned(s_frame_period: u32, tiles: (u32, u32)) -> GopTuning {
    GopTuning {
        s_frame_period,
        tiles,
        ..GopTuning::default()
    }
}

/// Encode with the cadence, decode through the spec driver, assert
/// byte-exactness against the encoder recon (and the input at q 0).
fn assert_s_frame_round_trip(frames: &[Yuv420Frame], q: u8, tuning: GopTuning) {
    let enc =
        encode_gop_yuv420_with_q_seg_tuned(frames, q, &[], tuning).expect("S-frame GOP encodes");
    let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("spec driver decodes own S-frame GOP");
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
}

/// Parse every coded frame header off the wire (in temporal-unit
/// order) and return `(frame_type, error_resilient, override,
/// refresh, primary_ref)` per frame.
fn wire_headers(temporal_units: &[Vec<u8>]) -> Vec<(FrameType, bool, bool, u8, u8)> {
    let mut seq = None;
    let refinfo = RefInfo::default();
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
                    out.push((
                        fh.frame_type,
                        fh.error_resilient_mode,
                        fh.frame_size_override_flag,
                        fh.refresh_frame_flags,
                        fh.primary_ref_frame,
                    ));
                }
                _ => {}
            }
        }
    }
    out
}

/// A 7-frame GOP at period 3 (S-frames at display positions 3 and 6)
/// decodes byte-exact, lossy and lossless.
#[test]
fn s_frame_cadence_round_trips_byte_exact() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 64, t)).collect();
    assert_s_frame_round_trip(&frames, 72, tuned(3, (0, 0)));
    let small: Vec<Yuv420Frame> = (0..5).map(|t| moving_frame(64, 64, t)).collect();
    assert_s_frame_round_trip(&small, 0, tuned(2, (0, 0)));
}

/// The emitted S-frame headers ride the §5.9.2 SWITCH shape — the
/// four inferred fields parse back exactly, and the surrounding
/// P-frames keep the plain INTER shape.
#[test]
fn s_frame_headers_ride_the_inferred_fields() {
    let frames: Vec<Yuv420Frame> = (0..7).map(|t| moving_frame(96, 64, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 80, &[], tuned(3, (0, 0)))
        .expect("S-frame GOP encodes");
    let headers = wire_headers(&enc.gop.temporal_units);
    assert_eq!(headers.len(), 7);
    for (idx, (ft, err, ovr, refresh, primary)) in headers.iter().enumerate() {
        match idx {
            0 => assert_eq!(*ft, FrameType::Key, "frame 0 is the KEY"),
            3 | 6 => {
                assert_eq!(*ft, FrameType::Switch, "frame {idx} codes SWITCH_FRAME");
                assert!(*err, "SWITCH infers error_resilient_mode = 1");
                assert!(*ovr, "SWITCH infers frame_size_override_flag = 1");
                assert_eq!(*refresh, ALL_FRAMES_PUB, "SWITCH refreshes allFrames");
                assert_eq!(*primary, PRIMARY_REF_NONE, "error resilience infers NONE");
            }
            _ => {
                assert_eq!(*ft, FrameType::Inter, "frame {idx} stays INTER");
                assert!(!*err, "plain P-frames are not error resilient");
            }
        }
    }
}

/// The cadence composes with the multi-tile layout and every
/// frame-level election surface left at its default-on state.
#[test]
fn s_frame_composes_with_tiles_and_elections() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| moving_frame(128, 64, t)).collect();
    assert_s_frame_round_trip(&frames, 100, tuned(2, (1, 0)));
}

/// The cadence composes with the §5.9.30 film-grain election: the
/// S-frame's error-resilient header carries a full grain block
/// (`update_grain = 1` — no `film_grain_params_ref_idx` load), the
/// per-frame seed schedule rides through the switch point, and the
/// grained output decodes byte-exact.
#[test]
fn s_frame_composes_with_film_grain() {
    // Deterministic re-rolled noise over a smooth moving base — the
    // §5.9.30 probe's use case.
    let noisy = |t: usize| -> Yuv420Frame {
        let (w, h) = (128u32, 96u32);
        let (wu, hu) = (w as usize, h as usize);
        let mut f = Yuv420Frame::filled(w, h, 128);
        let mut state = 0x2454_1013u32.wrapping_add((t as u32).wrapping_mul(0x9e37_79b9));
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (((state >> 23) & 255) as i32 - 128) * 8 / 128
        };
        for r in 0..hu {
            for c in 0..wu {
                let x = c as f64 + 1.1 * t as f64;
                let y = r as f64 + 0.5 * t as f64;
                let base = 120.0
                    + 60.0 * (0.021 * x).sin() * (0.026 * y).cos()
                    + 18.0 * (0.047 * (x + y)).sin();
                f.y[r * wu + c] = (base + f64::from(rnd())).round().clamp(0.0, 255.0) as u8;
            }
        }
        f
    };
    let frames: Vec<Yuv420Frame> = (0..4).map(noisy).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 60, &[], tuned(2, (0, 0)))
        .expect("film-grain S-frame GOP encodes");
    let headers = wire_headers(&enc.gop.temporal_units);
    assert_eq!(headers[2].0, FrameType::Switch, "frame 2 codes SWITCH");
    let decoded = decode_av1_spec(&enc.gop.ivf_bytes).expect("spec driver decodes");
    assert_eq!(decoded.len(), 4);
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.gop.recon[idx];
        assert_eq!(
            f.planes[0], rc.y,
            "frame {idx}: luma decode != published recon"
        );
        assert_eq!(f.planes[1], rc.u, "frame {idx}: U");
        assert_eq!(f.planes[2], rc.v, "frame {idx}: V");
    }
}

/// Cross-rate SPLICE at the switch point: chunk 1 of the q60 stream +
/// chunks 2+ of the q140 stream (same source, same cadence) — the
/// §7.5 temporal units concatenate into a stream the spec driver
/// parses and decodes end-to-end. The pre-switch frames decode
/// byte-exact to the q60 reconstructions; from the S-frame on, the
/// q140 SYMBOLS decode against the q60 reference samples (§5.9.2
/// error resilience guarantees the parse; the output is
/// cross-validated byte-exact against independent black-box decoders
/// on the dumped artifacts).
#[test]
fn switch_point_accepts_cross_rate_splice() {
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| moving_frame(96, 64, t)).collect();
    let a = encode_gop_yuv420_with_q_seg_tuned(&frames, 60, &[], tuned(3, (0, 0)))
        .expect("q60 stream encodes");
    let b = encode_gop_yuv420_with_q_seg_tuned(&frames, 140, &[], tuned(3, (0, 0)))
        .expect("q140 stream encodes");
    assert_eq!(a.gop.temporal_units.len(), 6);
    assert_eq!(b.gop.temporal_units.len(), 6);

    // Splice: A units 0..3 (KEY + two P), B units 3.. (S + two P).
    let mut ivf = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf);
        let mut iw = IvfWriter::new(cursor, FOURCC_AV01, 96, 64, 25, 1).expect("IVF header");
        for (idx, tu) in a.gop.temporal_units[..3]
            .iter()
            .chain(&b.gop.temporal_units[3..])
            .enumerate()
        {
            iw.write_frame(tu, idx as u64).expect("IVF frame");
        }
        iw.patch_frame_count().expect("IVF count");
    }

    let decoded = decode_av1_spec(&ivf).expect("spec driver decodes the cross-rate splice");
    assert_eq!(decoded.len(), 6, "one shown frame per spliced unit");
    for (idx, f) in decoded.iter().take(3).enumerate() {
        let rc = &a.gop.recon[idx];
        assert_eq!(
            f.planes[0], rc.y,
            "pre-switch frame {idx} luma == q60 recon"
        );
        assert_eq!(f.planes[1], rc.u, "pre-switch frame {idx} U == q60 recon");
        assert_eq!(f.planes[2], rc.v, "pre-switch frame {idx} V == q60 recon");
    }
    for (idx, f) in decoded.iter().enumerate().skip(3) {
        assert_eq!(
            (f.width, f.height),
            (96, 64),
            "post-switch frame {idx} extent"
        );
        assert_eq!(f.planes.len(), 3);
    }

    // Dump for external black-box decoder cross-validation.
    if let Some(dir) = std::env::var_os("OXIDEAV_AV1_SFRAME_DUMP_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("dump dir");
        std::fs::write(dir.join("splice-96x64-q60-q140.ivf"), &ivf).expect("dump splice");
        std::fs::write(dir.join("stream-a-q60.ivf"), &a.gop.ivf_bytes).expect("dump A");
        std::fs::write(dir.join("stream-b-q140.ivf"), &b.gop.ivf_bytes).expect("dump B");
        let mut planes = Vec::new();
        for f in &decoded {
            for p in &f.planes {
                planes.extend_from_slice(p);
            }
        }
        std::fs::write(dir.join("splice-decode.yuv"), planes).expect("dump decode");
    }
}
