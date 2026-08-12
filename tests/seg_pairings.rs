//! r436 — the §5.9.14 segmentation PAIRINGS the corpus lacked:
//! §5.9.17 per-superblock delta-q AND §5.9.19 per-unit CDEF on
//! actively segmented frames.
//!
//! What these tests pin:
//!
//!   * §7.12.2 `get_qindex` composition — on a segmented delta-q
//!     frame a SEG_LVL_ALT_Q block quantises at `CurrentQIndex +
//!     FeatureData` (the running §5.11.13 index, NOT `base_q_idx`),
//!     clipped to `[0, 255]`. The encoder's search/twin and the
//!     decoder ride the same derivation; a disagreement desyncs the
//!     residual chain and breaks the bit-exact match.
//!   * §7.15 CDEF over segmented frames — per-unit strength ids
//!     (`cdef_bits > 0`) beside a live segment map, including the
//!     r423 temporal segment-map election chain riding CDEF-filtered
//!     references.
//!   * Both pairings decode BIT-EXACT through the in-tree spec
//!     driver, and the wire audit shows `segmentation_enabled = 1`
//!     together with `delta_q_present = 1` / non-zero CDEF params on
//!     the same frame header.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.14, §5.9.17, §5.9.19,
//! §5.11.13, §7.12.2, §7.15.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning, TunedGop, Yuv420Frame};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, FrameHeader, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Mixed content: a flat plateau (delta-q refines — banding bait), a
/// strongly textured field (delta-q coarsens — masking), and sharp
/// diagonal bands over a ripple (CDEF ringing bait), slowly panning
/// so P-frames stay alive.
fn mixed_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    for r in 0..hu {
        for c in 0..wu {
            let x = c as f64 + 0.75 * t as f64;
            let y = r as f64 + 0.4 * t as f64;
            let v = if r < hu / 3 {
                // flat plateau with a whisper of gradient
                80.0 + (c as f64) * 0.05
            } else if r < 2 * hu / 3 {
                // ringing bait: hard diagonal bands over a ripple
                let ripple = 6.0 * (0.9 * x).sin() * (0.8 * y).sin();
                let d = (0.31 * x - 0.42 * y).sin();
                (if d > 0.55 { 205.0 } else { 72.0 }) + ripple
            } else {
                // heavy texture
                ((r * 7 + (c + 5 * t) * 13 + (r % 5) * (c % 7) * 3) % 256) as f64
            };
            f.y[r * wu + c] = clamp(v);
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = ((110 + r + c + 2 * t) % 256) as u8;
            f.v[r * cw + c] = ((90 + r * 2 + c + t) % 256) as u8;
        }
    }
    f
}

/// Delta-q probe bait: ENTIRE superblock columns of flat plateau
/// beside superblock columns of heavy texture (the §5.9.17 probe is
/// per-64×64 source-luma variance — a mixed superblock reads as
/// "textured" and the plan collapses to zero).
fn flat_texture_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
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
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = ((110 + r + c + 2 * t) % 256) as u8;
            f.v[r * cw + c] = ((90 + r * 2 + c + t) % 256) as u8;
        }
    }
    f
}

fn assert_decodes_to_recons(name: &str, enc: &TunedGop) {
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

/// Every frame header of the stream, in decode order.
fn wire_headers(enc: &TunedGop, w: u32, h: u32) -> Vec<FrameHeader> {
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
                    out.push(
                        parse_frame_header_with_refs(
                            desc.payload,
                            seq.as_ref().expect("SH precedes frames"),
                            &refinfo,
                        )
                        .expect("frame header parses"),
                    );
                }
                _ => {}
            }
        }
    }
    out
}

const SEG_TABLE: [i16; 3] = [0, -24, 32];

/// §5.9.17 delta-q on an actively segmented GOP: the probe finds the
/// flat/texture spread, the election adopts the per-superblock plan
/// on segmented P-frames, and at least one wire header carries
/// `segmentation_enabled = 1` AND `delta_q_present = 1` together —
/// the §7.12.2 `CurrentQIndex + FeatureData` composition decodes
/// bit-exact.
#[test]
fn segmented_delta_q_pairs_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| flat_texture_frame(128, 128, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        120,
        &SEG_TABLE,
        GopTuning {
            // Isolate the delta-q axis (CDEF/LR off — their own
            // pairing is covered below).
            cdef: false,
            cdef_units: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented delta-q encode");
    assert!(
        enc.delta_q_elections.iter().any(|&e| e),
        "designed content must elect the delta-q arm on a segmented P-frame: {:?}",
        enc.delta_q_elections
    );
    assert!(
        enc.p_segment_maps.iter().any(|m| m.iter().any(|&s| s != 0)),
        "the segment map must actually commit non-zero segments"
    );
    assert_decodes_to_recons("seg-delta-q", &enc);

    let headers = wire_headers(&enc, 128, 128);
    assert!(
        headers.iter().any(|fh| {
            let seg_on = fh.segmentation_params.as_ref().is_some_and(|sp| sp.enabled);
            let dq_on = fh
                .delta_q_params
                .as_ref()
                .is_some_and(|dq| dq.delta_q_present);
            seg_on && dq_on
        }),
        "no wire header pairs segmentation_enabled with delta_q_present"
    );
}

/// §5.9.19/§7.15 CDEF (frame-level + per-unit ids) on an actively
/// segmented GOP: the election fires beside a live segment map and
/// the CDEF-filtered reference chain decodes bit-exact.
#[test]
fn segmented_cdef_pairs_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| mixed_frame(128, 96, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        140,
        &SEG_TABLE,
        GopTuning {
            // Isolate the CDEF axis.
            delta_q: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented cdef encode");
    assert!(
        enc.cdef_elections.iter().any(|&e| e),
        "designed edge content must elect CDEF on a segmented P-frame: {:?}",
        enc.cdef_elections
    );
    assert!(
        enc.p_segment_maps.iter().any(|m| m.iter().any(|&s| s != 0)),
        "the segment map must actually commit non-zero segments"
    );
    assert_decodes_to_recons("seg-cdef", &enc);

    let headers = wire_headers(&enc, 128, 96);
    assert!(
        headers.iter().any(|fh| {
            let seg_on = fh.segmentation_params.as_ref().is_some_and(|sp| sp.enabled);
            let cdef_on = fh.cdef_params.as_ref().is_some_and(|cp| {
                cp.cdef_y_pri_strength.iter().any(|&s| s != 0)
                    || cp.cdef_uv_pri_strength.iter().any(|&s| s != 0)
                    || cp.cdef_bits > 0
            });
            seg_on && cdef_on
        }),
        "no wire header pairs segmentation_enabled with live CDEF params"
    );
}

/// r441 — §5.9.12 quantizer matrices on an actively segmented GOP:
/// the QM election fires beside a live segment map (`using_qmatrix =
/// 1` composing with SEG_LVL_ALT_Q on the same frame header), every
/// non-zero segment's residual chain rides its own §5.9.2
/// `SegQMLevel[ plane ][ segment_id ]` row through the per-segment
/// quantiser bundle, and the stream decodes bit-exact — a bundle
/// that dropped the QM state would desync the encoder recon from the
/// decoder's §7.12.3 dequantisation on every non-zero-segment block.
#[test]
fn segmented_qm_pairs_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| mixed_frame(128, 96, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        120,
        &SEG_TABLE,
        GopTuning {
            // Isolate the QM axis.
            delta_q: false,
            cdef: false,
            cdef_units: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented qm encode");
    assert!(
        enc.qm_elections.iter().any(|&e| e),
        "designed textured content must elect the QM arm on a segmented P-frame: {:?}",
        enc.qm_elections
    );
    assert!(
        enc.p_segment_maps.iter().any(|m| m.iter().any(|&s| s != 0)),
        "the segment map must actually commit non-zero segments"
    );
    assert_decodes_to_recons("seg-qm", &enc);

    let headers = wire_headers(&enc, 128, 96);
    assert!(
        headers.iter().any(|fh| {
            let seg_on = fh.segmentation_params.as_ref().is_some_and(|sp| sp.enabled);
            let qm_on = fh
                .quantization_params
                .as_ref()
                .is_some_and(|q| q.using_qmatrix);
            seg_on && qm_on
        }),
        "no wire header pairs segmentation_enabled with using_qmatrix"
    );
}

/// r441 — the §5.9.2 lossless-segment sentinel: a segmented table
/// whose segment reaches qindex 0 keeps `SegQMLevel = 15` for that
/// segment (its blocks ride the flat WHT chain) while the OTHER
/// segments still take the elected §9.5.3 level — the whole
/// composition decodes bit-exact.
#[test]
fn segmented_qm_lossless_sentinel_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| mixed_frame(128, 96, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        120,
        &[0, -120], // segment 1 reaches qindex 0 — lossless
        GopTuning {
            delta_q: false,
            cdef: false,
            cdef_units: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented qm lossless-sentinel encode");
    // The election may or may not fire per frame — what this test
    // pins is that WHEN the arm runs over a lossless-segment table,
    // the sentinel derivation keeps every block's quantisation in
    // lockstep with the decoder.
    assert_decodes_to_recons("seg-qm-lossless-sentinel", &enc);
}

/// The lossless-segment guard: a table whose segment reaches qindex 0
/// keeps the delta-q arm OFF (the conservative §7.12.2-note gate),
/// and the stream still decodes bit-exact.
#[test]
fn lossless_segment_table_keeps_delta_q_off() {
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| mixed_frame(64, 64, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        120,
        &[0, -120], // segment 1 reaches qindex 0 — lossless
        GopTuning {
            cdef: false,
            cdef_units: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("lossless-segment encode");
    assert!(
        enc.delta_q_elections.iter().all(|&e| !e),
        "a lossless-segment table must keep the delta-q arm off"
    );
    let headers = wire_headers(&enc, 64, 64);
    assert!(headers.iter().all(|fh| {
        !fh.delta_q_params
            .as_ref()
            .is_some_and(|dq| dq.delta_q_present)
    }));
    assert_decodes_to_recons("seg-lossless-no-dq", &enc);
}

/// Env-gated staging dump (`OXIDEAV_AV1_SEG_PAIR_DIR`): both paired
/// streams + expected YUV for black-box reference-decoder validation
/// and corpus pinning.
#[test]
fn seg_pairings_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SEG_PAIR_DIR") else {
        eprintln!("OXIDEAV_AV1_SEG_PAIR_DIR unset — skipping the seg-pairings staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| mixed_frame(128, 96, t)).collect();
    let dump = |name: &str, enc: &TunedGop| {
        std::fs::write(root.join(format!("{name}.ivf")), &enc.gop.ivf_bytes).expect("write ivf");
        let mut yuv: Vec<u8> = Vec::new();
        for rc in &enc.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
    };
    let dq_frames: Vec<Yuv420Frame> = (0..5).map(|t| flat_texture_frame(128, 128, t)).collect();
    let dq = encode_gop_yuv420_with_q_seg_tuned(
        &dq_frames,
        120,
        &SEG_TABLE,
        GopTuning {
            cdef: false,
            cdef_units: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented delta-q encode");
    assert!(dq.delta_q_elections.iter().any(|&e| e));
    dump("gop-128x128-q120-seg-delta-q", &dq);
    let cd = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        140,
        &SEG_TABLE,
        GopTuning {
            delta_q: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented cdef encode");
    assert!(cd.cdef_elections.iter().any(|&e| e));
    // r441 — the QM × segmentation pairing stream.
    let qm = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        120,
        &SEG_TABLE,
        GopTuning {
            delta_q: false,
            cdef: false,
            cdef_units: false,
            lr: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented qm encode");
    assert!(qm.qm_elections.iter().any(|&e| e));
    dump("gop-128x96-q120-seg-qm", &qm);
    dump("gop-128x96-q140-seg-cdef", &cd);
    let lr = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        140,
        &SEG_TABLE,
        GopTuning {
            delta_q: false,
            cdef: false,
            cdef_units: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented lr encode");
    assert!(lr.lr_elections.iter().any(|&e| e));
    dump("gop-128x96-q140-seg-lr", &lr);
}

/// §5.9.20/§7.17 loop restoration on an actively segmented GOP (the
/// last in-loop pairing): per-unit Wiener/self-guided plans elected
/// beside a live segment map, the §5.11.57 interleave re-emitted
/// around the committed segment-carrying trees — and the restored
/// reference chain decodes bit-exact.
#[test]
fn segmented_lr_pairs_and_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..5).map(|t| mixed_frame(128, 96, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        140,
        &SEG_TABLE,
        GopTuning {
            // Isolate the LR axis.
            delta_q: false,
            cdef: false,
            cdef_units: false,
            ..GopTuning::default()
        },
    )
    .expect("segmented lr encode");
    assert!(
        enc.lr_elections.iter().any(|&e| e),
        "designed detail content must elect LR on a segmented P-frame: {:?}",
        enc.lr_elections
    );
    assert!(
        enc.p_segment_maps.iter().any(|m| m.iter().any(|&s| s != 0)),
        "the segment map must actually commit non-zero segments"
    );
    assert_decodes_to_recons("seg-lr", &enc);

    let headers = wire_headers(&enc, 128, 96);
    assert!(
        headers.iter().any(|fh| {
            let seg_on = fh.segmentation_params.as_ref().is_some_and(|sp| sp.enabled);
            let lr_on = fh.lr_params.as_ref().is_some_and(|lp| lp.uses_lr);
            seg_on && lr_on
        }),
        "no wire header pairs segmentation_enabled with UsesLr"
    );
}
