//! r441 — the §5.9.8 SUPERRES election: A/B harness + conformance
//! smoke.
//!
//! What these tests pin:
//!
//!   * A forced superres KEY frame (`use_superres = 1`, coded width
//!     from the §5.9.8 derivation) decodes BIT-EXACT through the
//!     in-tree spec driver at the UPSCALED extent — the encoder's
//!     §7.16 recon mirror equals the decoder's output byte for byte.
//!   * The wire shape: `frame_size_override_flag = 0` seeds
//!     `FrameWidth` from the sequence maximum (the upscaled width),
//!     `coded_denom` rides `f(SUPERRES_DENOM_BITS)`, and the parsed
//!     header re-derives the coded width exactly.
//!   * The ELECTION: inside the arming window the winner is decided
//!     by the plain joint objective over original-extent SSE + exact
//!     realized bytes — an elected stream therefore scores no worse
//!     than the flat arm by construction; outside an elected win the
//!     stream is bit-identical to the `superres = false` baseline.
//!   * A GOP whose KEY elects superres: the P chain predicts from the
//!     UPSCALED reference (the §7.20 store), and every frame decodes
//!     bit-exact.
//!
//! Env-gated measurement (`OXIDEAV_AV1_SR_AB=1`): the content × q
//! matrix behind the committed `superres_arm_allowed` regime.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.5, §5.9.8, §7.16, §7.20.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::key_frame::{
    encode_key_frame_yuv420_with_q_sr, encode_key_frame_yuv420_with_q_sr_forced,
};
use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_extras_tuned, GopTuning, Yuv420Frame};
use oxideav_av1::frame_header::{parse_frame_header, SUPERRES_NUM};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Horizontally smooth content — wide vertical bands with soft
/// gradients plus a mild 2-D swell. What a §7.16 upscaler can
/// reproduce almost exactly, so the downscaled arm keeps the rate
/// win.
fn smooth_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    for r in 0..hu {
        for c in 0..wu {
            let x = c as f64 + 1.4 * t as f64;
            let y = r as f64 + 0.6 * t as f64;
            let v = 118.0
                + 68.0 * (0.019 * x).sin() * (0.023 * y).cos()
                + 22.0 * (0.041 * (x + y)).sin();
            f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = (112.0 + 30.0 * (0.03 * (c as f64 + t as f64)).sin()) as u8;
            f.v[r * cw + c] = (140.0 - 25.0 * (0.025 * (r as f64)).cos()) as u8;
        }
    }
    f
}

/// Fine horizontal detail — the arm's worst case (the upscaler cannot
/// recreate the lost columns).
fn detail_frame(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = ((r * 7 + (c + 3 * t) * 13 + (r % 5) * (c % 7) * 3) % 256) as u8;
        }
    }
    f
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut sse = 0u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = i64::from(x) - i64::from(y);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return f64::INFINITY;
    }
    10.0 * ((255.0f64 * 255.0 * a.len() as f64) / sse as f64).log10()
}

fn decoded_frames(ivf: &[u8]) -> Vec<oxideav_av1::decoder::SpecFrame> {
    oxideav_av1::decode_av1(ivf)
        .expect("decode")
        .into_iter()
        .map(|f| match f {
            Frame::Spec(s) => s,
            other => panic!("non-Spec frame {other:?}"),
        })
        .collect()
}

/// Forced §5.9.8 arm: wire audit + bit-exact decode at the upscaled
/// extent.
#[test]
fn forced_superres_key_decodes_bit_exact() {
    let f = smooth_frame(128, 96, 0);
    let enc = encode_key_frame_yuv420_with_q_sr_forced(&f, 140, 16).expect("forced sr encode");

    // Wire audit: SH carries enable_superres + the UPSCALED maximum;
    // FH derives the coded width through §5.9.8.
    let mut seq = None;
    let mut fh = None;
    for desc in ObuIter::new(&enc.temporal_unit_bytes) {
        let desc = desc.expect("TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
            ObuType::Frame => {
                fh = Some(
                    parse_frame_header(desc.payload, seq.as_ref().expect("SH precedes"))
                        .expect("FH parses"),
                );
            }
            _ => {}
        }
    }
    let seq = seq.expect("SH present");
    let fh = fh.expect("FH present");
    assert!(seq.enable_superres, "sequence gate must open");
    assert_eq!(seq.max_frame_width_minus_1 + 1, 128);
    let fs = fh.frame_size.expect("frame size");
    assert!(fs.use_superres);
    assert_eq!(fs.upscaled_width, 128);
    assert_eq!(fs.superres_denom, 16);
    assert_eq!(
        fs.frame_width,
        (128 * SUPERRES_NUM + 16 / 2) / 16,
        "§5.9.8 derivation"
    );

    // Bit-exact decode at the upscaled extent.
    let frames = decoded_frames(&enc.ivf_bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].width, 128, "output at UpscaledWidth");
    let recon_y: Vec<u8> = enc.recon_y.clone();
    assert_eq!(frames[0].planes[0], recon_y, "luma recon mirror");
    assert_eq!(frames[0].planes[1], enc.recon_u, "U recon mirror");
    assert_eq!(frames[0].planes[2], enc.recon_v, "V recon mirror");
}

/// Every candidate denominator decodes bit-exact (96 admits both 12
/// and 16; exercise a non-halving ratio too).
#[test]
fn forced_superres_denom_ladder_decodes_bit_exact() {
    for (w, h, denom) in [(96u32, 96u32, 12u32), (96, 96, 16), (72, 64, 9)] {
        let f = smooth_frame(w, h, 1);
        let enc = encode_key_frame_yuv420_with_q_sr_forced(&f, 160, denom)
            .unwrap_or_else(|e| panic!("{w}x{h} denom {denom}: {e:?}"));
        let frames = decoded_frames(&enc.ivf_bytes);
        assert_eq!(frames[0].width, w, "{w}x{h} denom {denom}: width");
        assert_eq!(
            frames[0].planes[0], enc.recon_y,
            "{w}x{h} denom {denom}: luma"
        );
        assert_eq!(frames[0].planes[1], enc.recon_u, "{w}x{h} d{denom}: U");
        assert_eq!(frames[0].planes[2], enc.recon_v, "{w}x{h} d{denom}: V");
    }
}

/// The election: smooth content inside the window elects the arm (and
/// the joint objective guarantees it scores no worse); detail content
/// keeps the flat shape BIT-IDENTICAL to the baseline.
#[test]
fn election_wins_on_smooth_and_stays_identical_on_detail() {
    let smooth = smooth_frame(128, 96, 0);
    let elected = encode_key_frame_yuv420_with_q_sr(&smooth, 180, true).expect("elected");
    let baseline = encode_key_frame_yuv420_with_q_sr(&smooth, 180, false).expect("baseline");
    let fs = elected.fh.frame_size.expect("frame size");
    assert!(
        fs.use_superres,
        "smooth coarse-q content must elect the superres arm \
         (elected {} B vs baseline {} B)",
        elected.ivf_bytes.len(),
        baseline.ivf_bytes.len()
    );
    let frames = decoded_frames(&elected.ivf_bytes);
    assert_eq!(frames[0].planes[0], elected.recon_y);

    let detail = detail_frame(128, 96, 0);
    let d_on = encode_key_frame_yuv420_with_q_sr(&detail, 140, true).expect("detail on");
    let d_off = encode_key_frame_yuv420_with_q_sr(&detail, 140, false).expect("detail off");
    assert_eq!(
        d_on.ivf_bytes, d_off.ivf_bytes,
        "an unelected frame must stay bit-identical to the baseline"
    );
}

/// GOP composition: a KEY that elects superres feeds the P chain its
/// UPSCALED reconstruction — every frame decodes bit-exact.
#[test]
fn gop_with_superres_key_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| smooth_frame(128, 96, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        180,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("gop encode");
    let key_fs = {
        let mut seq = None;
        let mut out = None;
        for desc in ObuIter::new(&enc.gop.temporal_units[0]) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
                }
                ObuType::Frame => {
                    out = Some(
                        parse_frame_header(desc.payload, seq.as_ref().expect("SH"))
                            .expect("FH parses")
                            .frame_size
                            .expect("frame size"),
                    );
                }
                _ => {}
            }
        }
        out.expect("KEY header present")
    };
    assert!(
        key_fs.use_superres,
        "smooth coarse-q content must elect superres on the GOP KEY"
    );
    let decoded = decoded_frames(&enc.gop.ivf_bytes);
    assert_eq!(decoded.len(), enc.gop.recon.len());
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "frame {i} V");
    }
}

/// r444 — the LR × superres pairing (§7.4 order: CDEF → §7.16 →
/// §7.17): a forced §5.9.8 arm on upscaler-blurred content ELECTS
/// loop restoration operating at the UPSCALED extent — the §5.9.20
/// header opens a non-NONE `FrameRestorationType`, the §5.11.57
/// window maps superblock columns through the superres denominator
/// ratio, and the stream decodes BIT-EXACT to the encoder's
/// (upscaled, restored) reconstruction mirror.
#[test]
fn forced_superres_arm_elects_loop_restoration() {
    // Vertically structured content with a mid-frequency horizontal
    // swell: the §7.16 upscaler low-passes the columns, and a §7.17.4
    // Wiener fit against the ORIGINAL source recovers part of the
    // loss — worth its subexp taps at coarse quantisers.
    let (w, h) = (128u32, 96u32);
    let mut f = smooth_frame(w, h, 0);
    let wu = w as usize;
    for r in 0..h as usize {
        for c in 0..wu {
            let base = f.y[r * wu + c] as f64;
            let ripple = 34.0 * (0.55 * c as f64).sin() * (0.16 * r as f64).cos();
            f.y[r * wu + c] = (base + ripple).round().clamp(0.0, 255.0) as u8;
        }
    }
    let enc = encode_key_frame_yuv420_with_q_sr_forced(&f, 140, 16).expect("forced sr encode");
    let mut seq = None;
    let mut fh = None;
    for desc in ObuIter::new(&enc.temporal_unit_bytes) {
        let desc = desc.expect("TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
            ObuType::Frame => {
                fh = Some(
                    parse_frame_header(desc.payload, seq.as_ref().expect("SH precedes"))
                        .expect("FH parses"),
                );
            }
            _ => {}
        }
    }
    let fh = fh.expect("FH present");
    let fs = fh.frame_size.expect("frame size");
    assert!(fs.use_superres, "the forced arm codes use_superres = 1");
    let lrp = fh.lr_params.expect("lossy header carries lr_params");
    assert!(
        lrp.uses_lr,
        "upscaler-blurred content must elect §7.17 restoration on the superres arm \
         (frame_restoration_type {:?})",
        lrp.frame_restoration_type
    );
    // Bit-exact decode at the upscaled extent, LR live.
    let frames = decoded_frames(&enc.ivf_bytes);
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].width, w, "output at UpscaledWidth");
    assert_eq!(frames[0].planes[0], enc.recon_y, "luma recon mirror");
    assert_eq!(frames[0].planes[1], enc.recon_u, "U recon mirror");
    assert_eq!(frames[0].planes[2], enc.recon_v, "V recon mirror");
}

/// r444 — GOP composition of the pairing: a KEY that elects BOTH
/// §5.9.8 superres and §7.17 restoration feeds the P chain its
/// upscaled RESTORED reconstruction (the §7.20 payload) — every frame
/// decodes bit-exact.
#[test]
fn gop_with_superres_lr_key_decodes_bit_exact() {
    // Mostly smooth (the frame-mean §5.9.8 probe passes) with ONE
    // narrow strip of mid-frequency horizontal ripple: the §7.16
    // downscale blurs the strip, and the §7.17.4 horizontal Wiener
    // taps recover part of it against the ORIGINAL source — the LR ×
    // superres pairing's textbook win.
    let (w, h) = (128u32, 96u32);
    let mk = |t: usize| {
        let mut f = smooth_frame(w, h, t);
        let wu = w as usize;
        for r in 0..h as usize {
            let band = 25.0 * (1.3 * (r as f64 + 0.7 * t as f64)).sin();
            for c in 0..wu {
                let base = f.y[r * wu + c] as f64;
                f.y[r * wu + c] = (base + band).round().clamp(0.0, 255.0) as u8;
            }
        }
        f
    };
    let frames: Vec<Yuv420Frame> = (0..4).map(mk).collect();
    let enc = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        140,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("gop encode");
    let mut seq = None;
    let mut key_fh = None;
    for desc in ObuIter::new(&enc.gop.temporal_units[0]) {
        let desc = desc.expect("TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
            ObuType::Frame => {
                key_fh = Some(
                    parse_frame_header(desc.payload, seq.as_ref().expect("SH")).expect("FH parses"),
                );
            }
            _ => {}
        }
    }
    let key_fh = key_fh.expect("KEY header present");
    assert!(
        key_fh.frame_size.expect("frame size").use_superres,
        "the GOP KEY must elect superres on this content"
    );
    assert!(
        key_fh.lr_params.expect("lr params").uses_lr,
        "the GOP KEY must pair loop restoration with the superres arm"
    );
    let decoded = decoded_frames(&enc.gop.ivf_bytes);
    assert_eq!(decoded.len(), enc.gop.recon.len());
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "frame {i} V");
    }
}

/// Extract the FIRST frame header (+ sequence header) of a stream's
/// first temporal unit.
fn first_frame_header(
    tu: &[u8],
) -> (
    oxideav_av1::sequence_header::SequenceHeader,
    oxideav_av1::frame_header::FrameHeader,
) {
    let mut seq = None;
    let mut fh = None;
    for desc in ObuIter::new(tu) {
        let desc = desc.expect("TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
            ObuType::Frame | ObuType::FrameHeader if fh.is_none() => {
                fh = Some(
                    parse_frame_header(desc.payload, seq.as_ref().expect("SH precedes"))
                        .expect("FH parses"),
                );
            }
            _ => {}
        }
    }
    (seq.expect("SH present"), fh.expect("FH present"))
}

/// r444 — SEGMENTED GOP × superres: the KEY of a plain segmented GOP
/// (itself unsegmented) elects the §5.9.8 arm; the segmented P chain
/// predicts from the upscaled reference, its §5.11.19 temporal
/// prediction rides the extent-checked all-zero
/// `load_previous_segment_ids()` arm against the KEY's mi-mismatched
/// map, and every frame decodes bit-exact.
#[test]
fn segmented_gop_with_superres_key_decodes_bit_exact() {
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| smooth_frame(128, 96, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        180,
        &[0, -60],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("segmented gop encode");
    let (seq, key_fh) = first_frame_header(&enc.gop.temporal_units[0]);
    assert!(
        key_fh.frame_size.expect("frame size").use_superres,
        "smooth coarse-q content must elect superres on the segmented GOP's KEY"
    );
    let p_fh = ObuIter::new(&enc.gop.temporal_units[1])
        .filter_map(|d| {
            let d = d.expect("TU walks");
            (d.obu_type == ObuType::Frame)
                .then(|| parse_frame_header(d.payload, &seq).expect("P FH parses"))
        })
        .next()
        .expect("P frame OBU present");
    assert!(
        p_fh.segmentation_params.expect("P header").enabled,
        "the P chain must stay segmented"
    );
    let decoded = decoded_frames(&enc.gop.ivf_bytes);
    assert_eq!(decoded.len(), enc.gop.recon.len());
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "frame {i} V");
    }
}

/// r444 — B-PYRAMID × superres: the pyramid KEY elects the §5.9.8
/// arm; the out-of-order refresh graph (ALT / MID / B roles, primary
/// carries, backward references) rides the KEY's upscaled §7.20
/// reconstruction and every display frame decodes bit-exact.
#[test]
fn pyramid_gop_with_superres_key_decodes_bit_exact() {
    use oxideav_av1::encoder::{encode_pyramid_gop_yuv420_with_q_tuned, PyramidTuning};
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| smooth_frame(128, 96, t)).collect();
    let enc = encode_pyramid_gop_yuv420_with_q_tuned(&frames, 180, PyramidTuning::default())
        .expect("pyramid encode");
    let (seq, key_fh) = first_frame_header(&enc.gop.temporal_units[0]);
    assert!(seq.enable_superres, "the pyramid sequence gate must open");
    assert!(
        key_fh.frame_size.expect("frame size").use_superres,
        "smooth coarse-q content must elect superres on the pyramid KEY"
    );
    let decoded = decoded_frames(&enc.gop.ivf_bytes);
    assert_eq!(decoded.len(), enc.gop.recon.len());
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "frame {i} V");
    }
}

/// r444 — TEMPORAL LADDER × superres: the §6.7.5 ladder KEY elects
/// the §5.9.8 arm (the multi-OP repack preserves the elected
/// sequence gate + upscaled maximum), every later layer frame codes
/// its `use_superres = 0` bit under the repacked header, and the
/// stream decodes bit-exact at the FULL operating point AND at a
/// reduced one (the §5.3.1 drop rule composing with the upscaled
/// KEY reference).
#[test]
fn temporal_ladder_with_superres_key_decodes_bit_exact() {
    use oxideav_av1::decoder::decode_av1_spec_at_operating_point;
    use oxideav_av1::encoder::encode_temporal_layered_gop_yuv420_with_q;
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| smooth_frame(128, 96, t)).collect();
    let enc = encode_temporal_layered_gop_yuv420_with_q(&frames, 180, 3).expect("ladder encode");
    let (seq, key_fh) = first_frame_header(&enc.gop.temporal_units[0]);
    assert!(seq.enable_superres, "the ladder sequence gate must open");
    assert!(
        key_fh.frame_size.expect("frame size").use_superres,
        "smooth coarse-q content must elect superres on the ladder KEY"
    );
    // Full-point decode: every display frame bit-exact.
    let decoded = decoded_frames(&enc.gop.ivf_bytes);
    assert_eq!(decoded.len(), enc.gop.recon.len());
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "frame {i} V");
    }
    // Reduced operating point 1: only frames whose temporal id
    // survives, each bit-exact.
    let reduced = decode_av1_spec_at_operating_point(&enc.gop.ivf_bytes, 1).expect("op-1 decode");
    let keep: Vec<usize> = enc
        .temporal_ids
        .iter()
        .enumerate()
        .filter(|(_, &tid)| tid <= 1)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(reduced.len(), keep.len(), "op-1 frame count");
    for (k, &i) in keep.iter().enumerate() {
        assert_eq!(reduced[k].planes[0], enc.gop.recon[i].y, "op-1 frame {i}");
    }
}

/// Env-gated measurement matrix (`OXIDEAV_AV1_SR_AB=1`): elected vs
/// flat baseline over content × q × geometry — the numbers behind the
/// committed `superres_arm_allowed` window.
#[test]
fn sr_ab_measurement_matrix() {
    if std::env::var_os("OXIDEAV_AV1_SR_AB").is_none() {
        eprintln!("OXIDEAV_AV1_SR_AB unset — skipping the superres measurement matrix");
        return;
    }
    type ContentGen = fn(u32, u32, usize) -> Yuv420Frame;
    let contents: [(&str, ContentGen); 2] = [("smooth", smooth_frame), ("detail", detail_frame)];
    let geoms = [(96u32, 96u32), (128, 96), (192, 128)];
    for (name, gen) in contents {
        for &(w, h) in &geoms {
            for q in [60u8, 100, 140, 180, 220] {
                let f = gen(w, h, 0);
                let off = encode_key_frame_yuv420_with_q_sr(&f, q, false).expect("off");
                let off_psnr = psnr(&off.recon_y, &f.y);
                let mut line = format!(
                    "sr-ab {name} {w}x{h} q{q}: flat {} B / {off_psnr:.2} dB",
                    off.ivf_bytes.len()
                );
                for denom in 9u32..=16 {
                    let Ok(arm) = encode_key_frame_yuv420_with_q_sr_forced(&f, q, denom) else {
                        continue;
                    };
                    let arm_psnr = psnr(&arm.recon_y, &f.y);
                    line += &format!(" | d{denom} {} B / {arm_psnr:.2} dB", arm.ivf_bytes.len());
                }
                eprintln!("{line}");
            }
        }
    }
}

/// Env-gated staging dump (`OXIDEAV_AV1_SR_DIR`): a superres-elected
/// GOP + expected YUV for black-box reference-decoder validation and
/// corpus pinning. Inert otherwise.
#[test]
fn sr_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SR_DIR") else {
        eprintln!("OXIDEAV_AV1_SR_DIR unset — skipping the superres staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| smooth_frame(128, 96, t)).collect();
    let enc = encode_gop_yuv420_with_q_seg_extras_tuned(
        &frames,
        180,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .expect("gop encode");
    // The KEY must have elected a §5.9.8 denominator.
    let ivf = &enc.gop.ivf_bytes;
    let mut seq = None;
    let mut elected = false;
    for desc in ObuIter::new(&enc.gop.temporal_units[0]) {
        let desc = desc.expect("TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("SH parses"));
            }
            ObuType::Frame => {
                let fh =
                    parse_frame_header(desc.payload, seq.as_ref().expect("SH")).expect("FH parses");
                elected = fh.frame_size.expect("frame size").use_superres;
            }
            _ => {}
        }
    }
    assert!(elected, "staged GOP must elect superres on the KEY");
    std::fs::write(root.join("gop-128x96-q180-superres.ivf"), ivf).expect("write ivf");
    let mut yuv: Vec<u8> = Vec::new();
    for rc in &enc.gop.recon {
        yuv.extend_from_slice(&rc.y);
        yuv.extend_from_slice(&rc.u);
        yuv.extend_from_slice(&rc.v);
    }
    std::fs::write(root.join("gop-128x96-q180-superres.yuv"), &yuv).expect("write yuv");
}

/// Env-gated staging dump (`OXIDEAV_AV1_SR444_DIR`): the r444
/// election streams — LR × superres GOP, segmented-GOP superres, and
/// the SVC superres openers — plus expected YUVs for black-box
/// reference-decoder validation and corpus pinning. Inert otherwise.
#[test]
fn r444_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_SR444_DIR") else {
        eprintln!("OXIDEAV_AV1_SR444_DIR unset — skipping the r444 staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let dump_yuv = |path: &std::path::Path, recon: &[oxideav_av1::encoder::GopFrameRecon]| {
        let mut yuv: Vec<u8> = Vec::new();
        for rc in recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(path, &yuv).expect("write yuv");
    };

    // (1) LR × superres — the elected pairing witness content.
    {
        let (w, h) = (128u32, 96u32);
        let mk = |t: usize| {
            let mut f = smooth_frame(w, h, t);
            let wu = w as usize;
            for r in 0..h as usize {
                let band = 25.0 * (1.3 * (r as f64 + 0.7 * t as f64)).sin();
                for c in 0..wu {
                    let base = f.y[r * wu + c] as f64;
                    f.y[r * wu + c] = (base + band).round().clamp(0.0, 255.0) as u8;
                }
            }
            f
        };
        let frames: Vec<Yuv420Frame> = (0..4).map(mk).collect();
        let enc = encode_gop_yuv420_with_q_seg_extras_tuned(
            &frames,
            140,
            &[],
            &[],
            false,
            None,
            GopTuning::default(),
        )
        .expect("gop encode");
        let (_, key_fh) = first_frame_header(&enc.gop.temporal_units[0]);
        assert!(
            key_fh.frame_size.expect("fs").use_superres,
            "KEY elects superres"
        );
        assert!(key_fh.lr_params.expect("lr").uses_lr, "KEY pairs LR");
        std::fs::write(
            root.join("gop-128x96-q140-superres-lr.ivf"),
            &enc.gop.ivf_bytes,
        )
        .expect("write ivf");
        dump_yuv(
            &root.join("gop-128x96-q140-superres-lr.yuv"),
            &enc.gop.recon,
        );
    }

    // (2) Segmented GOP whose KEY elects superres.
    {
        let frames: Vec<Yuv420Frame> = (0..4).map(|t| smooth_frame(128, 96, t)).collect();
        let enc = encode_gop_yuv420_with_q_seg_extras_tuned(
            &frames,
            180,
            &[0, -60],
            &[],
            false,
            None,
            GopTuning::default(),
        )
        .expect("segmented gop encode");
        let (seq, key_fh) = first_frame_header(&enc.gop.temporal_units[0]);
        assert!(
            key_fh.frame_size.expect("fs").use_superres,
            "KEY elects superres"
        );
        let p_fh = ObuIter::new(&enc.gop.temporal_units[1])
            .filter_map(|d| {
                let d = d.expect("TU walks");
                (d.obu_type == ObuType::Frame)
                    .then(|| parse_frame_header(d.payload, &seq).expect("P FH parses"))
            })
            .next()
            .expect("P frame OBU present");
        assert!(
            p_fh.segmentation_params.expect("P header").enabled,
            "P stays segmented"
        );
        std::fs::write(
            root.join("gop-128x96-q180-seg-superres.ivf"),
            &enc.gop.ivf_bytes,
        )
        .expect("write ivf");
        dump_yuv(
            &root.join("gop-128x96-q180-seg-superres.yuv"),
            &enc.gop.recon,
        );
    }

    // (3) SVC with superres openers (the spatial_svc witness shape).
    {
        use oxideav_av1::encoder::encode_spatial_layered_gop_yuv420_with_q;
        fn smooth(w: u32, h: u32, t: usize) -> Yuv420Frame {
            smooth_frame(w, h, t)
        }
        let layers: Vec<Vec<Yuv420Frame>> = vec![
            (0..3).map(|t| smooth(96, 80, t)).collect(),
            (0..3).map(|t| smooth(192, 160, t)).collect(),
        ];
        let enc = encode_spatial_layered_gop_yuv420_with_q(&layers, 180).expect("svc encode");
        assert!(enc.seq.enable_superres, "shared §5.9.8 gate open");
        std::fs::write(root.join("svc-96-192-q180-superres.ivf"), &enc.ivf_bytes)
            .expect("write ivf");
        // Full-point decode order: layer 0 then layer 1 per instant.
        let mut yuv: Vec<u8> = Vec::new();
        for i in 0..3 {
            for s in 0..2 {
                let rc = &enc.layer_recons[s][i];
                yuv.extend_from_slice(&rc.y);
                yuv.extend_from_slice(&rc.u);
                yuv.extend_from_slice(&rc.v);
            }
        }
        std::fs::write(root.join("svc-96-192-q180-superres.yuv"), &yuv).expect("write yuv");
        // Base-layer (operating point 1) expected output.
        let mut yuv1: Vec<u8> = Vec::new();
        for i in 0..3 {
            let rc = &enc.layer_recons[0][i];
            yuv1.extend_from_slice(&rc.y);
            yuv1.extend_from_slice(&rc.u);
            yuv1.extend_from_slice(&rc.v);
        }
        std::fs::write(root.join("svc-96-192-q180-superres-op1.yuv"), &yuv1).expect("write yuv");
    }
}
