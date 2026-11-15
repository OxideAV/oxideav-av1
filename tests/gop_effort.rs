//! r489 — the search-effort ladder of the KEY + P GOP encoder
//! ([`oxideav_av1::encoder::encode_gop_yuv_with_options`]).
//!
//! Every [`GopEffort`] level must produce a conformant stream: this
//! crate's decoder — and, when one is installed, a black-box decoder
//! process — reproduces the encoder's own reconstruction sample for
//! sample, on every §6.4.1 format pairing, on extents off the 8-sample
//! grid, on multi-tile layouts and on the lossless arm (where the
//! reconstruction additionally equals the input). The bounded levels
//! must also be what they are for: smaller than the all-intra coding
//! of the same frames at matched quality, and far cheaper to run than
//! the exhaustive level.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.2, §5.11.4, §5.11.23–§5.11.26,
//! §7.10.2, §7.11.3.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    ChromaFormat, EncodedGopYuv, GopEffort, GopOptions, StillOptions, YuvFrame,
    encode_gop_yuv_with_options, encode_gop_yuv_with_q, encode_key_frame_yuv_with_options,
};
use std::process::Command;

const BOUNDED: &[GopEffort] = &[
    GopEffort::Fastest,
    GopEffort::Fast,
    GopEffort::Balanced,
    GopEffort::Thorough,
];

/// A deterministic scene with real inter-frame structure: a textured
/// background panning by `(t, 2t)`, a block moving the other way, and
/// a static band — at any extent / depth / format.
fn scene(w: u32, h: u32, bit_depth: u8, fmt: ChromaFormat, t: usize) -> YuvFrame {
    let mut f = YuvFrame::filled(w, h, bit_depth, fmt, 0);
    let (wu, hu) = (w as usize, h as usize);
    let shift = u32::from(bit_depth - 8);
    let tex = |i: usize, j: usize| -> usize {
        let a = (i * 7 + j * 5) % 61;
        let b = ((i / 6) * 13 + (j / 9) * 29) % 97;
        (40 + a + b + ((i ^ j) & 15)) % 256
    };
    let (ox, oy) = (wu / 3 + 3 * t, hu / 4 + t);
    for i in 0..hu {
        for j in 0..wu {
            let v8 = if i >= hu - hu / 6 {
                // Static band.
                tex(i, j * 2)
            } else if (oy..oy + hu / 4).contains(&i) && (ox..ox + wu / 5).contains(&j) {
                // Object moving against the pan.
                200 - ((i - oy) * 3 + (j - ox) * 2) % 90
            } else {
                tex(i + t, j + 2 * t)
            };
            let extra = (i * 3 + j * 11 + t) & ((1usize << shift) - 1);
            f.y[i * wu + j] = ((v8 << shift) + extra) as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for i in 0..ch {
        for j in 0..cw {
            let (si, sj) = (i + t / 2, j + t);
            f.u[i * cw + j] = (((128 + si * 2 + sj) % 256) << shift) as u16;
            f.v[i * cw + j] = (((64 + si + sj * 2) % 256) << shift) as u16;
        }
    }
    f
}

fn gop(w: u32, h: u32, bd: u8, fmt: ChromaFormat, n: usize) -> Vec<YuvFrame> {
    (0..n).map(|t| scene(w, h, bd, fmt, t)).collect()
}

fn plane_bytes(bit_depth: u8, p: &[u16]) -> Vec<u8> {
    if bit_depth == 8 {
        p.iter().map(|&s| s as u8).collect()
    } else {
        p.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}

/// This crate's decoder reproduces the encoder's reconstruction.
fn assert_own_decode(tag: &str, enc: &EncodedGopYuv, frames: &[YuvFrame]) {
    let (bd, fmt) = (frames[0].bit_depth, frames[0].format);
    let dims = (frames[0].width, frames[0].height);
    let out =
        decode_av1_spec(&enc.ivf_bytes).unwrap_or_else(|e| panic!("{tag}: decode failed: {e:?}"));
    assert_eq!(out.len(), frames.len(), "{tag}: frame count");
    assert_eq!(enc.recon.len(), frames.len(), "{tag}: recon count");
    for (i, (f, r)) in out.iter().zip(&enc.recon).enumerate() {
        assert_eq!((f.width, f.height), dims, "{tag}: frame {i} extent");
        assert!(
            f.planes[0] == plane_bytes(bd, &r.y),
            "{tag}: frame {i} luma differs from the encoder reconstruction"
        );
        if fmt != ChromaFormat::Monochrome {
            assert!(
                f.planes[1] == plane_bytes(bd, &r.u) && f.planes[2] == plane_bytes(bd, &r.v),
                "{tag}: frame {i} chroma differs from the encoder reconstruction"
            );
        }
    }
}

/// A black-box decoder process reproduces the reconstruction too
/// (skipped at run time when no such binary is installed).
fn assert_black_box(tag: &str, enc: &EncodedGopYuv, frames: &[YuvFrame]) {
    let (bd, fmt) = (frames[0].bit_depth, frames[0].format);
    let dir = std::env::temp_dir().join(format!(
        "oxideav-av1-effort-{}-{}",
        std::process::id(),
        tag.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
    ));
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let (src, dst) = (dir.join("in.ivf"), dir.join("out.yuv"));
    std::fs::write(&src, &enc.ivf_bytes).expect("write stream");
    let status = Command::new("dav1d")
        .arg("-q")
        .arg("-i")
        .arg(&src)
        .arg("-o")
        .arg(&dst)
        .args(["--muxer", "yuv"])
        .status();
    match status {
        Ok(s) => {
            assert!(
                s.success(),
                "{tag}: the black-box decoder rejected the stream"
            );
            let out = std::fs::read(&dst).expect("black-box decoder output");
            let mut want = Vec::new();
            for r in &enc.recon {
                want.extend(plane_bytes(bd, &r.y));
                if fmt != ChromaFormat::Monochrome {
                    want.extend(plane_bytes(bd, &r.u));
                    want.extend(plane_bytes(bd, &r.v));
                }
            }
            assert!(
                out == want,
                "{tag}: black-box decoder output differs from the encoder reconstruction"
            );
        }
        Err(_) => {
            assert!(
                std::env::var_os("OXIDEAV_AV1_REQUIRE_BLACK_BOX").is_none(),
                "{tag}: no black-box decoder on PATH"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

fn luma_psnr(frames: &[YuvFrame], recon: &[Vec<u16>]) -> f64 {
    let peak = f64::from((1u32 << frames[0].bit_depth) - 1);
    let (mut sse, mut n) = (0f64, 0usize);
    for (f, r) in frames.iter().zip(recon) {
        for (a, b) in f.y.iter().zip(r) {
            let d = f64::from(*a) - f64::from(*b);
            sse += d * d;
        }
        n += f.y.len();
    }
    10.0 * (peak * peak * n as f64 / sse.max(1e-9)).log10()
}

fn encode(tag: &str, frames: &[YuvFrame], opts: &GopOptions) -> EncodedGopYuv {
    let enc = encode_gop_yuv_with_options(frames, opts)
        .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
    assert_own_decode(tag, &enc, frames);
    assert_black_box(tag, &enc, frames);
    enc
}

#[test]
fn every_bounded_effort_round_trips_on_every_format() {
    const FORMATS: &[(u8, ChromaFormat)] = &[
        (8, ChromaFormat::Yuv420),
        (10, ChromaFormat::Yuv420),
        (8, ChromaFormat::Yuv444),
        (12, ChromaFormat::Yuv444),
        (10, ChromaFormat::Yuv422),
        (8, ChromaFormat::Monochrome),
    ];
    for &(bd, fmt) in FORMATS {
        let frames = gop(96, 72, bd, fmt, 4);
        for &effort in BOUNDED {
            for q in [40u8, 140] {
                let tag = format!("{bd}-bit {fmt:?} {effort:?} q={q}");
                encode(&tag, &frames, &GopOptions::new(q).with_effort(effort));
            }
        }
    }
}

#[test]
fn bounded_efforts_take_extents_off_the_mi_grid() {
    for (w, h) in [(33u32, 17u32), (71, 49), (64, 34), (130, 70), (5, 9)] {
        let frames = gop(w, h, 8, ChromaFormat::Yuv420, 3);
        for &effort in BOUNDED {
            let tag = format!("{w}x{h} {effort:?}");
            encode(&tag, &frames, &GopOptions::new(100).with_effort(effort));
        }
    }
}

#[test]
fn bounded_efforts_are_lossless_at_q0() {
    for &(bd, fmt) in &[(8u8, ChromaFormat::Yuv420), (10, ChromaFormat::Yuv444)] {
        let frames = gop(72, 56, bd, fmt, 3);
        for &effort in BOUNDED {
            let tag = format!("lossless {bd}-bit {fmt:?} {effort:?}");
            let enc = encode(&tag, &frames, &GopOptions::new(0).with_effort(effort));
            for (i, (f, r)) in frames.iter().zip(&enc.recon).enumerate() {
                assert!(
                    f.y == r.y && f.u == r.u && f.v == r.v,
                    "{tag}: frame {i} differs from the input"
                );
            }
        }
    }
}

#[test]
fn tiles_and_threads_do_not_change_the_stream() {
    for (w, h, fmt, tiles) in [
        (192u32, 144u32, ChromaFormat::Yuv420, (1u32, 1u32)),
        (320, 128, ChromaFormat::Yuv420, (2, 1)),
        (200, 136, ChromaFormat::Yuv444, (1, 1)),
    ] {
        let frames = gop(w, h, 8, fmt, 4);
        for &effort in BOUNDED {
            let base = GopOptions::new(110)
                .with_effort(effort)
                .with_tiles_log2(tiles.0, tiles.1);
            assert_eq!(base.threads, 1, "serial is the default");
            let tag = format!("{w}x{h} {fmt:?} tiles {tiles:?} {effort:?}");
            let serial = encode(&tag, &frames, &base);
            for threads in [2usize, 3, 8] {
                let par = encode_gop_yuv_with_options(&frames, &base.with_threads(threads))
                    .unwrap_or_else(|e| panic!("{tag} threads={threads}: {e:?}"));
                assert!(
                    par.ivf_bytes == serial.ivf_bytes,
                    "{tag}: the stream differs at threads={threads}"
                );
                for (a, b) in par.recon.iter().zip(&serial.recon) {
                    assert!(
                        a.y == b.y && a.u == b.u && a.v == b.v,
                        "{tag}: the reconstruction differs at threads={threads}"
                    );
                }
            }
        }
    }
}

#[test]
fn a_thread_budget_is_inert_on_one_tile_and_at_best_effort() {
    let frames = gop(96, 64, 8, ChromaFormat::Yuv420, 3);
    let one = GopOptions::new(100);
    let serial = encode_gop_yuv_with_options(&frames, &one).expect("serial");
    let budget = encode_gop_yuv_with_options(&frames, &one.with_threads(8)).expect("budget");
    assert!(serial.ivf_bytes == budget.ivf_bytes);
    let small = gop(64, 48, 8, ChromaFormat::Yuv420, 2);
    let best = GopOptions::new(90).with_effort(GopEffort::Best);
    let a = encode_gop_yuv_with_options(&small, &best).expect("best");
    let b = encode_gop_yuv_with_options(&small, &best.with_threads(4)).expect("best threads");
    assert!(a.ivf_bytes == b.ivf_bytes);
}

#[test]
fn best_effort_is_the_exhaustive_encoder() {
    let frames = gop(64, 48, 8, ChromaFormat::Yuv420, 3);
    let best =
        encode_gop_yuv_with_options(&frames, &GopOptions::new(90).with_effort(GopEffort::Best))
            .expect("best");
    let historical = encode_gop_yuv_with_q(&frames, 90).expect("historical");
    assert!(
        best.ivf_bytes == historical.ivf_bytes,
        "GopEffort::Best must reproduce encode_gop_yuv_with_q byte for byte"
    );
}

#[test]
fn default_effort_is_balanced_and_beats_all_intra_at_matched_quality() {
    assert_eq!(GopOptions::new(100).effort, GopEffort::Balanced);
    let frames = gop(160, 120, 8, ChromaFormat::Yuv420, 6);
    let inter = encode("default effort", &frames, &GopOptions::new(100));
    let inter_psnr = luma_psnr(
        &frames,
        &inter.recon.iter().map(|r| r.y.clone()).collect::<Vec<_>>(),
    );
    // The all-intra coding of the same frames at the coarsest
    // quantiser that still reaches the inter stream's quality.
    let mut intra_bytes_at_quality: Option<usize> = None;
    for q in (20u8..=250).rev().step_by(10) {
        let mut bytes = 0usize;
        let mut ys = Vec::new();
        for f in &frames {
            let k = encode_key_frame_yuv_with_options(f, &StillOptions::new(q)).expect("intra");
            bytes += k.temporal_unit_bytes.len();
            ys.push(k.recon_y);
        }
        if luma_psnr(&frames, &ys) >= inter_psnr {
            intra_bytes_at_quality = Some(bytes);
            break;
        }
    }
    let intra = intra_bytes_at_quality.expect("some quantiser reaches the inter quality");
    let inter_bytes: usize = inter.temporal_units.iter().map(Vec::len).sum();
    assert!(
        inter_bytes * 10 < intra * 8,
        "inter {inter_bytes} B at {inter_psnr:.2} dB should be well under all-intra {intra} B at \
         matched quality"
    );
}

#[test]
fn rejects_what_the_gop_encoder_rejects() {
    assert!(encode_gop_yuv_with_options(&[], &GopOptions::new(100)).is_err());
    let mut frames = gop(64, 48, 8, ChromaFormat::Yuv420, 2);
    frames[1] = scene(48, 64, 8, ChromaFormat::Yuv420, 1);
    assert!(encode_gop_yuv_with_options(&frames, &GopOptions::new(100)).is_err());
    // A tile layout outside the §5.9.15 window for the picture.
    let frames = gop(64, 48, 8, ChromaFormat::Yuv420, 2);
    assert!(
        encode_gop_yuv_with_options(&frames, &GopOptions::new(100).with_tiles_log2(3, 3)).is_err()
    );
}
