//! r450 — the §5.9.30 FILM-GRAIN election across the §6.4.1
//! chroma/bit-depth axis: monochrome, 10/12-bit, 4:2:2 and 4:4:4.
//!
//! The 4:2:0 8-bit grain surface (white / AR / csfl) is pinned by
//! `film_grain_ab.rs`; these witnesses drive the SAME election
//! through the general-format GOP entry and pin the format-specific
//! wire shapes:
//!
//!   * MONOCHROME — §5.9.30 reads no `chroma_scaling_from_luma` bit
//!     and no chroma point/mult/offset surface at all
//!     (`mono_chrome` suppresses them); §7.18.3 synthesizes luma
//!     grain only.
//!   * 10/12-BIT — `generate_grain` seeds Gaussian samples at
//!     `12 - bit_depth + grain_scale_shift` and the §7.18.3.5 blend
//!     clips at the depth's range; the noise-estimate normalization
//!     and the scaling-LUT §7.18.3.4 interpolation (`scale_lut`
//!     indexes at `bit_depth - 8`) both carry depth terms.
//!   * 4:2:2 / 4:4:4 — the per-plane chroma gates stay UNCOUPLED
//!     (the §5.9.30 both-or-neither constraint binds only 4:2:0),
//!     so a single-chroma-plane grain header
//!     (`num_cb_points > 0, num_cr_points == 0`) is legal wire.
//!
//! Every elected stream must decode BIT-EXACT through the in-tree
//! spec driver (the §7.18.3 output mirror), and the staging dump
//! feeds the black-box reference-decoder validation + corpus pins.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.30, §6.4.1, §7.18.3.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::{
    encode_gop_yuv_seg_extras_tuned, ChromaFormat, GopTuning, TunedGopYuv, YuvFrame,
};
use oxideav_av1::frame_header::{parse_frame_header, FrameHeader};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

/// Smooth moving base at 8-bit scale — what survives the
/// encoder-side denoiser.
fn base_value(r: usize, c: usize, t: usize) -> f64 {
    let x = c as f64 + 1.1 * t as f64;
    let y = r as f64 + 0.5 * t as f64;
    120.0 + 60.0 * (0.021 * x).sin() * (0.026 * y).cos() + 18.0 * (0.047 * (x + y)).sin()
}

/// Deterministic per-(plane, frame) white-noise stream at 8-bit
/// scale.
fn noise_stream(plane: u32, t: usize, amp: i32) -> impl FnMut() -> i32 {
    let mut state = 0x2454_1013u32
        .wrapping_add((t as u32).wrapping_mul(0x9e37_79b9))
        .wrapping_add(plane.wrapping_mul(0x85eb_ca6b));
    move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (((state >> 23) & 255) as i32 - 128) * amp / 128
    }
}

/// Noisy content at any (depth, format) pairing: the smooth base
/// plus white noise re-rolled per frame, both depth-scaled.
/// `amp_y` / `amp_cb` / `amp_cr` are 8-bit-scale amplitudes (0 =
/// clean plane; chroma planes get a smooth base of their own).
#[allow(clippy::too_many_arguments)]
fn noisy_frame(
    w: u32,
    h: u32,
    bit_depth: u8,
    fmt: ChromaFormat,
    t: usize,
    amp_y: i32,
    amp_cb: i32,
    amp_cr: i32,
) -> YuvFrame {
    let mut f = YuvFrame::filled(w, h, bit_depth, fmt, 0);
    let (wu, hu) = (w as usize, h as usize);
    let shift = i32::from(bit_depth - 8);
    let max = (1i32 << bit_depth) - 1;
    let scale =
        |v8: f64| -> u16 { (((v8 * f64::from(1 << shift)).round() as i32).clamp(0, max)) as u16 };
    let mut rnd_y = noise_stream(0, t, amp_y);
    for r in 0..hu {
        for c in 0..wu {
            f.y[r * wu + c] = scale(base_value(r, c, t) + f64::from(rnd_y()));
        }
    }
    if fmt != ChromaFormat::Monochrome {
        let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
        let mut rnd_u = noise_stream(1, t, amp_cb);
        let mut rnd_v = noise_stream(2, t, amp_cr);
        for r in 0..ch {
            for c in 0..cw {
                let base_u = 116.0 + 8.0 * (0.05 * (r as f64 + c as f64)).sin();
                let base_v = 132.0 + 8.0 * (0.04 * (r as f64 * 2.0 + c as f64)).cos();
                f.u[r * cw + c] = scale(base_u + f64::from(rnd_u()));
                f.v[r * cw + c] = scale(base_v + f64::from(rnd_v()));
            }
        }
    }
    f
}

fn encode(frames: &[YuvFrame], q: u8, film_grain: bool) -> TunedGopYuv {
    encode_gop_yuv_seg_extras_tuned(
        frames,
        q,
        &[],
        &[],
        false,
        None,
        GopTuning {
            film_grain,
            ..GopTuning::default()
        },
    )
    .expect("gop encode")
}

fn wire_headers(tus: &[Vec<u8>]) -> (bool, Vec<FrameHeader>) {
    let mut seq = None;
    let mut out = Vec::new();
    let mut fg_present = false;
    for tu in tus {
        for desc in ObuIter::new(tu) {
            let desc = desc.expect("TU walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    let s = parse_sequence_header(desc.payload).expect("SH parses");
                    fg_present = s.film_grain_params_present;
                    seq = Some(s);
                }
                ObuType::Frame => {
                    out.push(
                        parse_frame_header(desc.payload, seq.as_ref().expect("SH precedes"))
                            .expect("FH parses"),
                    );
                }
                _ => {}
            }
        }
    }
    (fg_present, out)
}

/// Little-endian (10/12-bit) or single-byte (8-bit) plane view — the
/// `SpecFrame` output layout.
fn plane_bytes(bit_depth: u8, p: &[u16]) -> Vec<u8> {
    if bit_depth == 8 {
        p.iter().map(|&s| s as u8).collect()
    } else {
        p.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}

/// The published (grained) reconstruction must equal the public
/// decode output byte-for-byte on every plane of every frame.
fn assert_decodes_to_recons(name: &str, enc: &TunedGopYuv, bit_depth: u8, fmt: ChromaFormat) {
    let frames = oxideav_av1::decode_av1(&enc.gop.ivf_bytes)
        .unwrap_or_else(|e| panic!("{name}: decode: {e:?}"));
    assert_eq!(frames.len(), enc.gop.recon.len(), "{name}: frame count");
    let num_planes = if fmt == ChromaFormat::Monochrome {
        1
    } else {
        3
    };
    for (i, f) in frames.iter().enumerate() {
        let Frame::Spec(s) = f else {
            panic!("{name}: frame {i}: expected the spec-driver surface");
        };
        assert_eq!(s.bit_depth, bit_depth, "{name}: frame {i} depth");
        assert_eq!(s.planes.len(), num_planes, "{name}: frame {i} planes");
        let rc = &enc.gop.recon[i];
        assert_eq!(
            s.planes[0],
            plane_bytes(bit_depth, &rc.y),
            "{name}: frame {i} luma"
        );
        if num_planes == 3 {
            assert_eq!(
                s.planes[1],
                plane_bytes(bit_depth, &rc.u),
                "{name}: frame {i} Cb"
            );
            assert_eq!(
                s.planes[2],
                plane_bytes(bit_depth, &rc.v),
                "{name}: frame {i} Cr"
            );
        }
    }
}

fn grain_header<'a>(
    headers: &'a [FrameHeader],
    name: &str,
) -> &'a oxideav_av1::uncompressed_header_tail::FilmGrainParams {
    headers[0]
        .film_grain_params
        .as_ref()
        .unwrap_or_else(|| panic!("{name}: KEY must carry the §5.9.30 block"))
}

/// MONOCHROME grain: the election runs on the single plane, the
/// §5.9.30 header suppresses the whole chroma surface, and the
/// luma-only §7.18.3 synthesis decodes bit-exact.
#[test]
fn mono_8bit_noise_elects_grain_and_decodes_bit_exact() {
    let frames: Vec<YuvFrame> = (0..4)
        .map(|t| noisy_frame(96, 80, 8, ChromaFormat::Monochrome, t, 12, 0, 0))
        .collect();
    let on = encode(&frames, 40, true);
    let off = encode(&frames, 40, false);
    assert!(
        on.film_grain_elected,
        "mono white noise must elect the grain arm (grain {} B vs plain {} B)",
        on.gop.ivf_bytes.len(),
        off.gop.ivf_bytes.len()
    );
    let (fg_present, headers) = wire_headers(&on.gop.temporal_units);
    assert!(fg_present, "sequence gate must open");
    let fg = grain_header(&headers, "mono-8");
    assert!(fg.apply_grain && fg.num_y_points > 0, "luma grain points");
    assert!(
        !fg.chroma_scaling_from_luma && fg.num_cb_points == 0 && fg.num_cr_points == 0,
        "mono_chrome suppresses the chroma surface"
    );
    assert_decodes_to_recons("mono-8", &on, 8, ChromaFormat::Monochrome);
}

/// Clean monochrome content keeps the plain shape bit-identical.
#[test]
fn mono_clean_content_keeps_plain_shape() {
    let frames: Vec<YuvFrame> = (0..4)
        .map(|t| noisy_frame(96, 80, 8, ChromaFormat::Monochrome, t, 0, 0, 0))
        .collect();
    let on = encode(&frames, 40, true);
    let off = encode(&frames, 40, false);
    assert!(!on.film_grain_elected, "clean mono must not elect");
    assert_eq!(
        on.gop.ivf_bytes, off.gop.ivf_bytes,
        "non-elected arm must be bit-identical to the baseline"
    );
}

/// 12-bit monochrome: the depth-scaled noise estimate still lands
/// the election and the high-depth luma-only synthesis mirrors.
#[test]
fn mono_12bit_noise_elects_grain_and_decodes_bit_exact() {
    let frames: Vec<YuvFrame> = (0..4)
        .map(|t| noisy_frame(96, 80, 12, ChromaFormat::Monochrome, t, 12, 0, 0))
        .collect();
    let on = encode(&frames, 40, true);
    assert!(on.film_grain_elected, "12-bit mono noise must elect");
    let (_, headers) = wire_headers(&on.gop.temporal_units);
    let fg = grain_header(&headers, "mono-12");
    assert!(fg.num_y_points > 0 && fg.num_cb_points == 0 && fg.num_cr_points == 0);
    assert_decodes_to_recons("mono-12", &on, 12, ChromaFormat::Monochrome);
}

/// 10-bit 4:2:0 with three-plane noise: the chroma gates fire at
/// depth, the §5.9.30 both-or-neither 4:2:0 constraint holds, and
/// the §7.18.3 blend clips at the 10-bit range.
#[test]
fn yuv420_10bit_noise_elects_grain_and_decodes_bit_exact() {
    let frames: Vec<YuvFrame> = (0..4)
        .map(|t| noisy_frame(96, 80, 10, ChromaFormat::Yuv420, t, 12, 12, 12))
        .collect();
    let on = encode(&frames, 60, true);
    assert!(on.film_grain_elected, "10-bit 4:2:0 noise must elect");
    let (_, headers) = wire_headers(&on.gop.temporal_units);
    let fg = grain_header(&headers, "420-10");
    assert!(fg.num_y_points > 0, "luma points");
    assert_eq!(
        fg.num_cb_points == 0,
        fg.num_cr_points == 0,
        "4:2:0 both-or-neither chroma constraint"
    );
    assert_decodes_to_recons("420-10", &on, 10, ChromaFormat::Yuv420);
}

/// 4:2:2 with Cb-only noise: outside 4:2:0 the per-plane gates stay
/// uncoupled, so `num_cb_points > 0, num_cr_points == 0` is legal
/// §5.9.30 wire — the corpus's first single-chroma-plane grain
/// header.
#[test]
fn yuv422_cb_only_noise_lands_single_plane_chroma_grain() {
    let frames: Vec<YuvFrame> = (0..4)
        .map(|t| noisy_frame(96, 80, 8, ChromaFormat::Yuv422, t, 8, 14, 0))
        .collect();
    let on = encode(&frames, 60, true);
    assert!(on.film_grain_elected, "4:2:2 luma+Cb noise must elect");
    let (_, headers) = wire_headers(&on.gop.temporal_units);
    let fg = grain_header(&headers, "422-cb");
    assert!(fg.num_y_points > 0, "luma points");
    assert!(
        fg.num_cb_points > 0 && fg.num_cr_points == 0,
        "Cb-only chroma grain must survive uncoupled (cb {} cr {})",
        fg.num_cb_points,
        fg.num_cr_points
    );
    assert_decodes_to_recons("422-cb", &on, 8, ChromaFormat::Yuv422);
}

/// 4:4:4 12-bit with three-plane noise: full-resolution chroma grain
/// (`chroma_sub{x,y} == 0` §7.18.3 arms) at the deepest depth.
#[test]
fn yuv444_12bit_noise_elects_grain_and_decodes_bit_exact() {
    let frames: Vec<YuvFrame> = (0..4)
        .map(|t| noisy_frame(96, 80, 12, ChromaFormat::Yuv444, t, 12, 12, 12))
        .collect();
    let on = encode(&frames, 60, true);
    assert!(on.film_grain_elected, "12-bit 4:4:4 noise must elect");
    let (_, headers) = wire_headers(&on.gop.temporal_units);
    let fg = grain_header(&headers, "444-12");
    assert!(fg.num_y_points > 0, "luma points");
    assert_decodes_to_recons("444-12", &on, 12, ChromaFormat::Yuv444);
}

/// Env-gated staging dump (`OXIDEAV_AV1_FG_FMT_DIR`): the three
/// format-axis grain streams for black-box reference-decoder
/// validation and corpus pinning. Inert otherwise.
#[test]
fn fg_format_fixture_staging() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_FG_FMT_DIR") else {
        eprintln!("OXIDEAV_AV1_FG_FMT_DIR unset — skipping the format-axis grain staging dump");
        return;
    };
    let root = std::path::Path::new(&dir);
    std::fs::create_dir_all(root).expect("create out dir");
    let cases: &[(&str, u8, u8, ChromaFormat, i32, i32, i32)] = &[
        (
            "gop-96x80-q40-mono-film-grain",
            40,
            8,
            ChromaFormat::Monochrome,
            12,
            0,
            0,
        ),
        (
            "gop-96x80-q60-10bit-film-grain",
            60,
            10,
            ChromaFormat::Yuv420,
            12,
            12,
            12,
        ),
        (
            "gop-96x80-q60-422-cb-film-grain",
            60,
            8,
            ChromaFormat::Yuv422,
            8,
            14,
            0,
        ),
    ];
    for &(name, q, bd, fmt, ay, acb, acr) in cases {
        let frames: Vec<YuvFrame> = (0..4)
            .map(|t| noisy_frame(96, 80, bd, fmt, t, ay, acb, acr))
            .collect();
        let enc = encode(&frames, q, true);
        assert!(enc.film_grain_elected, "{name}: staged GOP must elect");
        std::fs::write(root.join(format!("{name}.ivf")), &enc.gop.ivf_bytes).expect("write ivf");
        let mut yuv: Vec<u8> = Vec::new();
        for rc in &enc.gop.recon {
            yuv.extend_from_slice(&plane_bytes(bd, &rc.y));
            if fmt != ChromaFormat::Monochrome {
                yuv.extend_from_slice(&plane_bytes(bd, &rc.u));
                yuv.extend_from_slice(&plane_bytes(bd, &rc.v));
            }
        }
        std::fs::write(root.join(format!("{name}.yuv")), &yuv).expect("write yuv");
    }
}
