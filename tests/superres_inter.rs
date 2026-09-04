//! r456 — §5.9.8 superres on INTER frames: the mid-GOP election codes
//! probe-passing P-frames at the downscaled width against references
//! held at the UPSCALED extent — the §7.11.3.3 scaled sampling path
//! (`is_scaled( refFrame ) == 1` for every reference), the §5.11.27
//! `use_obmc` collapse, §7.16 upscaling between the CDEF and LR
//! stages — and the stream decodes BIT-EXACT to the encoder's
//! upscaled-extent reconstruction mirror.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.2, §5.9.5, §5.9.7, §5.9.8,
//! §5.11.27, §6.8.2, §7.11.3.3, §7.16, §7.20.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_seg_extras_tuned, encode_gop_yuv_seg_extras_tuned_layout, GopTuning,
    TunedGop, TunedGopYuv, Yuv420Frame, YuvFrame,
};
use oxideav_av1::frame_header::{parse_frame_header_with_refs, FrameHeader, FrameSize, RefInfo};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Horizontally smooth, slowly drifting content — the §5.9.8 arm's
/// win regime (the probe passes on every frame); the drift keeps the
/// P-frames from collapsing to whole-frame skips so the inter
/// election has residual to trade.
fn smooth_drift(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    for r in 0..hu {
        for c in 0..wu {
            let x = c as f64 + 2.3 * t as f64;
            let y = r as f64 + 1.1 * t as f64;
            let v = 118.0
                + 68.0 * (0.019 * x).sin() * (0.023 * y).cos()
                + 22.0 * (0.041 * (x + y)).sin()
                + 9.0 * (0.11 * y + 0.4 * t as f64).sin();
            f.y[r * wu + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for r in 0..ch {
        for c in 0..cw {
            f.u[r * cw + c] = (112.0 + 30.0 * (0.03 * (c as f64 + t as f64)).sin()) as u8;
            f.v[r * cw + c] = (140.0 - 25.0 * (0.025 * (r as f64 + 0.5 * t as f64)).cos()) as u8;
        }
    }
    f
}

/// Fine horizontal detail — probe-failing on every frame.
fn detail(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 128);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = (((c + t) % 3) * 90 + (r % 5) * 12) as u8;
        }
    }
    f
}

fn decoded(ivf: &[u8]) -> Vec<oxideav_av1::decoder::SpecFrame> {
    oxideav_av1::decode_av1(ivf)
        .expect("decode")
        .into_iter()
        .map(|f| match f {
            Frame::Spec(s) => s,
            other => panic!("non-Spec frame {other:?}"),
        })
        .collect()
}

fn encode(frames: &[Yuv420Frame], q: u8, tuning: GopTuning) -> TunedGop {
    encode_gop_yuv420_with_q_seg_extras_tuned(frames, q, &[], &[], false, None, tuning)
        .expect("gop encode")
}

/// Every coded frame's §5.9.5/§5.9.8 `FrameSize` in decode order,
/// parsed with the references' true stored extents.
fn frame_sizes(enc: &TunedGop, w: u32, h: u32) -> Vec<FrameSize> {
    frame_headers(&enc.gop.temporal_units, w, h)
        .into_iter()
        .map(|fh| fh.frame_size.expect("coded frame carries a size"))
        .collect()
}

/// Every coded frame header in decode order, parsed with the
/// references' true stored extents (every §7.20 slot of a GOP holds
/// the upscaled extent).
fn frame_headers(tus: &[Vec<u8>], w: u32, h: u32) -> Vec<FrameHeader> {
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
    out
}

fn assert_bit_exact(enc: &TunedGop, what: &str) {
    let dec = decoded(&enc.gop.ivf_bytes);
    assert_eq!(dec.len(), enc.gop.recon.len(), "{what}: frame count");
    for (i, f) in dec.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "{what}: frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "{what}: frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "{what}: frame {i} V");
    }
}

/// The election fires on at least one P-frame of the designed
/// content, the wire carries `use_superres = 1` with the §5.9.8
/// derivation landing exactly on the coded width, and the whole GOP
/// decodes bit-exact at the upscaled extent.
#[test]
fn mid_gop_election_fires_and_decodes_bit_exact() {
    let (w, h) = (128u32, 96u32);
    let frames: Vec<Yuv420Frame> = (0..6).map(|t| smooth_drift(w, h, t)).collect();
    let enc = encode(&frames, 180, GopTuning::default());
    let elected: Vec<Option<u32>> = enc.superres_inter_elections.clone();
    assert!(
        elected.iter().any(|e| e.is_some()),
        "designed content must elect the §5.9.8 inter arm on some P-frame: {elected:?}"
    );
    let sizes = frame_sizes(&enc, w, h);
    assert_eq!(sizes.len(), frames.len());
    for (k, e) in elected.iter().enumerate() {
        let fs = &sizes[k + 1];
        match e {
            Some(denom) => {
                assert!(fs.use_superres, "P{} must code use_superres = 1", k + 1);
                assert_eq!(fs.superres_denom, *denom);
                assert_eq!(fs.upscaled_width, w);
                assert_eq!(fs.frame_width, (w * 8 + denom / 2) / denom);
                assert_eq!(fs.mi_cols, 2 * ((fs.frame_width + 7) >> 3));
            }
            None => {
                assert!(!fs.use_superres, "P{} keeps the flat shape", k + 1);
                assert_eq!(fs.frame_width, w);
            }
        }
    }
    assert_bit_exact(&enc, "inter superres election");
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_SR_INTER_DUMP") {
        let off = encode(
            &frames,
            180,
            GopTuning {
                superres: false,
                ..GopTuning::default()
            },
        );
        let psnr = |t: &TunedGop| -> f64 {
            let mut se = 0u64;
            let mut n = 0u64;
            for (rc, src) in t.gop.recon.iter().zip(&frames) {
                for (a, b) in rc.y.iter().zip(&src.y) {
                    let d = i64::from(*a) - i64::from(*b);
                    se += (d * d) as u64;
                    n += 1;
                }
            }
            10.0 * (255.0f64 * 255.0 * n as f64 / se.max(1) as f64).log10()
        };
        eprintln!(
            "superres-inter elections {elected:?}; bytes on {} vs off {}; luma PSNR on {:.2} off {:.2}; per-TU on {:?} off {:?}",
            enc.gop.ivf_bytes.len(),
            off.gop.ivf_bytes.len(),
            psnr(&enc),
            psnr(&off),
            enc.gop.temporal_units.iter().map(Vec::len).collect::<Vec<_>>(),
            off.gop.temporal_units.iter().map(Vec::len).collect::<Vec<_>>(),
        );
        std::fs::create_dir_all(&dir).expect("dump dir");
        std::fs::write(
            format!("{dir}/self-gop-128x96-q180-superres-inter.ivf"),
            &enc.gop.ivf_bytes,
        )
        .expect("ivf dump");
        let mut yuv = Vec::new();
        for rc in &enc.gop.recon {
            yuv.extend_from_slice(&rc.y);
            yuv.extend_from_slice(&rc.u);
            yuv.extend_from_slice(&rc.v);
        }
        std::fs::write(
            format!("{dir}/self-gop-128x96-q180-superres-inter.yuv"),
            yuv,
        )
        .expect("yuv dump");
    }
}

/// Probe-failing content never opens the sequence gate: the stream is
/// bit-identical to the `superres: false` tuning.
#[test]
fn detail_content_stays_bit_identical_to_off_arm() {
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| detail(128, 96, t)).collect();
    let on = encode(&frames, 180, GopTuning::default());
    let off = encode(
        &frames,
        180,
        GopTuning {
            superres: false,
            ..GopTuning::default()
        },
    );
    assert!(on.superres_inter_elections.iter().all(|e| e.is_none()));
    assert_eq!(on.gop.ivf_bytes, off.gop.ivf_bytes);
}

/// The off arm on the election content: still bit-exact, never codes
/// the §5.9.8 bit (the sequence gate stays shut).
#[test]
fn off_arm_keeps_flat_shape_bit_exact() {
    let (w, h) = (128u32, 96u32);
    let frames: Vec<Yuv420Frame> = (0..4).map(|t| smooth_drift(w, h, t)).collect();
    let enc = encode(
        &frames,
        180,
        GopTuning {
            superres: false,
            ..GopTuning::default()
        },
    );
    assert!(enc.superres_inter_elections.iter().all(|e| e.is_none()));
    for fs in frame_sizes(&enc, w, h) {
        assert!(!fs.use_superres);
        assert_eq!(fs.frame_width, w);
    }
    assert_bit_exact(&enc, "off arm");
}

/// r456 — an EXPLICIT (§5.9.15 non-uniform) column layout rides the
/// superres arms: the `[3, 2]` layout of a 320-wide frame is remapped
/// onto the downscaled superblock grid per candidate denominator
/// (column count preserved, `uniform_tile_spacing_flag = 0` on the
/// wire, every non-rightmost column >= 128 luma samples per the
/// Annex A superres rule) and the stream decodes bit-exact.
#[test]
fn explicit_layout_rides_the_superres_arms() {
    let (w, h) = (320u32, 96u32);
    let frames: Vec<YuvFrame> = (0..4)
        .map(|t| YuvFrame::from_yuv420_8bit(&smooth_drift(w, h, t)))
        .collect();
    let widths = [3u32, 2];
    let heights = [2u32];
    let enc: TunedGopYuv = encode_gop_yuv_seg_extras_tuned_layout(
        &frames,
        180,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
        Some((&widths, &heights)),
    )
    .expect("layout gop encode");
    let headers = frame_headers(&enc.gop.temporal_units, w, h);
    assert_eq!(headers.len(), frames.len());
    let mut elected_any = false;
    for (k, fh) in headers.iter().enumerate() {
        let fs = fh.frame_size.as_ref().expect("size");
        let ti = fh.tile_info.as_ref().expect("tile info");
        assert!(
            !ti.uniform_tile_spacing_flag,
            "frame {k}: explicit layout on the wire"
        );
        assert_eq!(ti.tile_cols, 2, "frame {k}: column count preserved");
        assert_eq!(ti.tile_rows, 1, "frame {k}: single tile row");
        let sb_cols = (fs.mi_cols + 15) >> 4;
        let starts = &ti.mi_col_starts[..=ti.tile_cols as usize];
        assert_eq!(
            starts[ti.tile_cols as usize], fs.mi_cols,
            "frame {k}: last start"
        );
        if fs.use_superres {
            elected_any = true;
            assert_eq!(fs.upscaled_width, w);
            assert!(fs.frame_width < w);
            let first_sb = starts[1].div_ceil(16);
            assert!(
                first_sb >= 2 && first_sb < sb_cols,
                "frame {k}: remapped widths {starts:?}"
            );
            assert!(
                (starts[1] - starts[0]) * 4 >= 128,
                "frame {k}: Annex A tile width"
            );
        } else {
            assert_eq!(
                starts[1],
                3 * 16,
                "frame {k}: flat frames keep the [3, 2] layout"
            );
        }
    }
    assert!(
        elected_any,
        "designed content must elect the arm on some frame"
    );
    let dec = decoded(&enc.gop.ivf_bytes);
    assert_eq!(dec.len(), enc.gop.recon.len());
    let narrow = |p: &[u16]| -> Vec<u8> { p.iter().map(|&s| s as u8).collect() };
    for (i, f) in dec.iter().enumerate() {
        assert_eq!(f.planes[0], narrow(&enc.gop.recon[i].y), "frame {i} luma");
        assert_eq!(f.planes[1], narrow(&enc.gop.recon[i].u), "frame {i} U");
        assert_eq!(f.planes[2], narrow(&enc.gop.recon[i].v), "frame {i} V");
    }
    if let Ok(dir) = std::env::var("OXIDEAV_AV1_SR_INTER_DUMP") {
        eprintln!(
            "explicit-layout superres elections {:?}",
            enc.superres_inter_elections
        );
        std::fs::create_dir_all(&dir).expect("dump dir");
        std::fs::write(
            format!("{dir}/self-gop-320x96-q180-superres-explicit-tiles.ivf"),
            &enc.gop.ivf_bytes,
        )
        .expect("ivf dump");
        let mut yuv = Vec::new();
        for rc in &enc.gop.recon {
            for p in [&rc.y, &rc.u, &rc.v] {
                yuv.extend(p.iter().map(|&s| s as u8));
            }
        }
        std::fs::write(
            format!("{dir}/self-gop-320x96-q180-superres-explicit-tiles.yuv"),
            yuv,
        )
        .expect("yuv dump");
    }
}
