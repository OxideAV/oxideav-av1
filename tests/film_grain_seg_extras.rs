//! r456 — §5.9.30 film grain × the two segmentation tables the r450
//! election still excluded: FEATURE-EXTRA segmentation
//! (SEG_LVL_SKIP / SEG_LVL_GLOBALMV / SEG_LVL_REF_FRAME, §5.9.14) and
//! SEG_LVL_ALT_Q tables carrying a LOSSLESS segment. The grain arm
//! codes the denoised frames under the SAME table, every P header
//! carries BOTH the feature table and the full grain block, and the
//! streams decode bit-exact to the encoder's §7.18.3-synthesized
//! output mirror.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.14, §5.9.30, §7.12.2,
//! §7.18.3, §7.20.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, GopTuning, SegExtras, TunedGop, Yuv420Frame,
};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, FrameHeader, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

fn base_value(r: usize, c: usize, t: usize) -> f64 {
    let x = c as f64 + 1.1 * t as f64;
    let y = r as f64 + 0.5 * t as f64;
    120.0 + 60.0 * (0.021 * x).sin() * (0.026 * y).cos() + 18.0 * (0.047 * (x + y)).sin()
}

/// Smooth drifting base plus deterministic white noise re-rolled per
/// frame (temporally decorrelated — the §5.9.30 use case); the RIGHT
/// half is a static flat panel with lower-amplitude noise, so a
/// segment-map election has two distinct regions to work with.
fn noisy_split(w: u32, h: u32, t: usize, amp: i32) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    let mut state = 0x2454_1013u32.wrapping_add((t as u32).wrapping_mul(0x9e37_79b9));
    let mut rnd = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (((state >> 23) & 255) as i32 - 128) * amp / 128
    };
    for r in 0..hu {
        for c in 0..wu {
            let base = if c < wu / 2 {
                base_value(r, c, t)
            } else {
                96.0 + 20.0 * (0.05 * r as f64).sin()
            };
            let v = base + f64::from(rnd());
            f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = (116 + ((r + c + t) % 24)) as u8;
            f.v[r * cw + c] = (132 + ((r * 2 + c) % 20)) as u8;
        }
    }
    f
}

fn encode(
    frames: &[Yuv420Frame],
    q: u8,
    alt_q: &[i16],
    extras: Option<&SegExtras>,
    film_grain: bool,
) -> TunedGop {
    encode_gop_yuv420_with_q_seg_extras_tuned(
        frames,
        q,
        alt_q,
        &[],
        false,
        extras,
        GopTuning {
            film_grain,
            ..GopTuning::default()
        },
    )
    .expect("gop encode")
}

fn headers(enc: &TunedGop, w: u32, h: u32) -> (bool, Vec<FrameHeader>) {
    let mut seq = None;
    let mut refinfo = RefInfo::default();
    for i in 0..8 {
        refinfo.valid[i] = true;
        refinfo.upscaled_width[i] = w;
        refinfo.frame_height[i] = h;
        refinfo.render_width[i] = w;
        refinfo.render_height[i] = h;
    }
    let mut fg_present = false;
    let mut out = Vec::new();
    for tu in &enc.gop.temporal_units {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    let sh = parse_sequence_header(desc.payload).expect("SH parses");
                    fg_present = sh.film_grain_params_present;
                    seq = Some(sh);
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let fh = parse_frame_header_with_refs(
                        desc.payload,
                        seq.as_ref().expect("SH precedes frames"),
                        &refinfo,
                    )
                    .expect("frame header parses");
                    // §7.20-tracked reference state: the §5.9.22
                    // skipModeAllowed derivation reads the TRUE
                    // per-slot order hints (a ref-less parse desyncs
                    // on frames whose slots hold two forward hints).
                    let fs = fh.frame_size.as_ref().expect("sized header");
                    for slot in 0..8 {
                        if fh.refresh_frame_flags & (1 << slot) != 0 {
                            refinfo.valid[slot] = true;
                            refinfo.order_hint[slot] = fh.order_hint;
                            refinfo.upscaled_width[slot] = fs.upscaled_width;
                            refinfo.frame_height[slot] = fs.frame_height;
                            refinfo.render_width[slot] = fs.render_width;
                            refinfo.render_height[slot] = fs.render_height;
                            refinfo.frame_type_is_key[slot] =
                                fh.frame_type == oxideav_av1::frame_header::FrameType::Key;
                        }
                    }
                    out.push(fh);
                }
                _ => {}
            }
        }
    }
    (fg_present, out)
}

fn assert_bit_exact(enc: &TunedGop, what: &str) {
    let dec: Vec<_> = oxideav_av1::decode_av1(&enc.gop.ivf_bytes)
        .expect("decode")
        .into_iter()
        .map(|f| match f {
            Frame::Spec(s) => s,
            other => panic!("non-Spec frame {other:?}"),
        })
        .collect();
    assert_eq!(dec.len(), enc.gop.recon.len(), "{what}: frame count");
    for (i, f) in dec.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "{what}: frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "{what}: frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "{what}: frame {i} V");
    }
}

/// Every P header carries the §5.9.14 table AND an `apply_grain = 1`
/// block; the KEY carries the block alone (the KEY of a segmented
/// GOP is unsegmented).
fn assert_wire_pairing(enc: &TunedGop, w: u32, h: u32, what: &str) {
    let (fg_present, hs) = headers(enc, w, h);
    assert!(fg_present, "{what}: sequence gate must open");
    for (k, fh) in hs.iter().enumerate() {
        let fg = fh
            .film_grain_params
            .as_ref()
            .unwrap_or_else(|| panic!("{what}: frame {k} carries the §5.9.30 block"));
        assert!(fg.apply_grain, "{what}: frame {k}: apply_grain");
        if k > 0 {
            let sp = fh
                .segmentation_params
                .as_ref()
                .unwrap_or_else(|| panic!("{what}: frame {k} carries the §5.9.14 table"));
            assert!(
                sp.enabled,
                "{what}: frame {k}: segmentation on the same header"
            );
        }
    }
}

fn dump(enc: &TunedGop, name: &str) {
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_FG_SEG_EXTRAS_DUMP") {
        std::fs::create_dir_all(&dir).expect("dump dir");
        std::fs::write(format!("{dir}/{name}.ivf"), &enc.gop.ivf_bytes).expect("ivf dump");
        let mut yuv = Vec::new();
        for rc in &enc.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(format!("{dir}/{name}.yuv"), yuv).expect("yuv dump");
        eprintln!(
            "{name}: {} bytes, elected {}",
            enc.gop.ivf_bytes.len(),
            enc.film_grain_elected
        );
    }
}

/// SEG_LVL_SKIP on the static flat panel's segment × film grain.
#[test]
fn seg_skip_table_elects_grain_and_decodes_bit_exact() {
    let (w, h) = (128u32, 96u32);
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_split(w, h, t, 8)).collect();
    let mut x = SegExtras::default();
    x.skip[1] = true;
    let on = encode(&frames, 60, &[0, 16], Some(&x), true);
    let off = encode(&frames, 60, &[0, 16], Some(&x), false);
    assert!(
        on.film_grain_elected,
        "noisy SEG_LVL_SKIP GOP must elect the grain arm ({} B vs plain {} B)",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    assert_wire_pairing(&on, w, h, "seg-skip × grain");
    assert_bit_exact(&on, "seg-skip × grain");
    assert_bit_exact(&off, "seg-skip plain");
    dump(&on, "self-gop-128x96-q60-seg-skip-film-grain");
}

/// SEG_LVL_GLOBALMV + SEG_LVL_REF_FRAME feature table × film grain.
#[test]
fn seg_globalmv_refframe_table_elects_grain_and_decodes_bit_exact() {
    let (w, h) = (128u32, 96u32);
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_split(w, h, t, 8)).collect();
    let mut x = SegExtras::default();
    x.globalmv[1] = true;
    x.ref_frame[2] = Some(1);
    let on = encode(&frames, 72, &[0, 8, -8], Some(&x), true);
    assert!(
        on.film_grain_elected,
        "noisy GLOBALMV/REF_FRAME GOP must elect the grain arm ({} B)",
        on.gop.ivf_bytes.len()
    );
    assert_wire_pairing(&on, w, h, "seg-globalmv/refframe × grain");
    assert_bit_exact(&on, "seg-globalmv/refframe × grain");
}

/// A SEG_LVL_ALT_Q table whose second segment clamps to qindex 0 (a
/// LOSSLESS segment inside a lossy frame — WHT leaves, no tx symbols)
/// × film grain: the r450 gate is lifted, the pairing rides one
/// header, and the stream decodes bit-exact.
#[test]
fn lossless_segment_table_elects_grain_and_decodes_bit_exact() {
    let (w, h) = (128u32, 96u32);
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_split(w, h, t, 8)).collect();
    let on = encode(&frames, 60, &[0, -60], None, true);
    let off = encode(&frames, 60, &[0, -60], None, false);
    assert!(
        on.film_grain_elected,
        "noisy lossless-segment GOP must elect the grain arm ({} B vs plain {} B)",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    assert_wire_pairing(&on, w, h, "lossless-seg × grain");
    assert_bit_exact(&on, "lossless-seg × grain");
    assert_bit_exact(&off, "lossless-seg plain");
    dump(&on, "self-gop-128x96-q60-lossless-seg-film-grain");
}

/// Clean content under the same tables keeps the plain shape
/// bit-identical (the probe rejects it before any arm is spent).
#[test]
fn clean_tables_stay_bit_identical() {
    let (w, h) = (128u32, 96u32);
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| noisy_split(w, h, t, 0)).collect();
    let mut x = SegExtras::default();
    x.skip[1] = true;
    for (alt_q, extras) in [(&[0i16, 16][..], Some(&x)), (&[0i16, -60][..], None)] {
        let on = encode(&frames, 60, alt_q, extras, true);
        let off = encode(&frames, 60, alt_q, extras, false);
        assert!(!on.film_grain_elected);
        assert_eq!(on.gop.ivf_bytes, off.gop.ivf_bytes);
    }
}
