//! r487 — picture extents OFF the 8-sample mi grid on every multi-frame
//! encoder entry point (plain GOP, B-pyramid, adaptive mini-GOPs, the
//! temporal ladder, spatial layers with and without inter-layer
//! prediction, the spatial × temporal product).
//!
//! §5.9.5 codes the true `FrameWidth` / `FrameHeight` and derives
//! `MiCols = 2 * ((FrameWidth + 7) >> 3)` (rows alike); the decoder
//! reconstructs every mi block in full while §7.11.3.3 clamps every
//! reference fetch to the reference's true last column / row and
//! §7.17 runs on the true extent. Every stream here must decode — in
//! this crate's decoder and, when one is installed, in a black-box
//! decoder process — to the encoder's reconstruction sample for
//! sample, at the true extent; lossless streams reproduce the input.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.5, §5.9.8, §6.8.2, §7.11.3.2,
//! §7.11.3.3, §7.17, §7.20.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    ChromaFormat, GopFrameReconYuv, YuvFrame, encode_adaptive_gop_yuv_with_q,
    encode_gop_yuv_with_q, encode_pyramid_gop_yuv_with_q, encode_spatial_layered_gop_yuv_with_q,
    encode_spatial_layered_gop_yuv_with_q_inter_layer,
    encode_spatial_temporal_layered_gop_yuv_with_q, encode_temporal_layered_gop_yuv_with_q,
};
use oxideav_av1::encoder::{
    Yuv420Frame, encode_gop_yuv_with_q_tile_layout, encode_gop_yuv420_with_q,
    encode_gop_yuv420_with_q_seg, encode_temporal_layered_gop_yuv420_with_q,
};
use std::process::Command;

/// Deterministic moving texture at any extent / depth / format.
fn textured(w: u32, h: u32, bit_depth: u8, fmt: ChromaFormat, t: usize) -> YuvFrame {
    let mut f = YuvFrame::filled(w, h, bit_depth, fmt, 0);
    let (wu, hu) = (w as usize, h as usize);
    let shift = u32::from(bit_depth - 8);
    for i in 0..hu {
        for j in 0..wu {
            let (si, sj) = (i + t, j + 2 * t);
            let v8 = (si * 5 + sj * 3 + (si / 16) * (sj / 16)) % 256;
            let extra = (si * 7 + sj * 13) & ((1usize << shift) - 1);
            f.y[i * wu + j] = ((v8 << shift) + extra) as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for i in 0..ch {
        for j in 0..cw {
            let (si, sj) = (i + t, j + t);
            let u8v = (128 + si * 2 + sj) % 256;
            let v8v = (64 + si + sj * 2) % 256;
            let extra = (si * 3 + sj * 5) & ((1usize << shift) - 1);
            f.u[i * cw + j] = ((u8v << shift) + extra) as u16;
            f.v[i * cw + j] = ((v8v << shift) + extra) as u16;
        }
    }
    f
}

/// One frame's planes at the stream depth.
struct Planes {
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

fn wide(r: &[GopFrameReconYuv]) -> Vec<Planes> {
    r.iter()
        .map(|g| Planes {
            y: g.y.clone(),
            u: g.u.clone(),
            v: g.v.clone(),
        })
        .collect()
}

fn narrow8(r: &[oxideav_av1::encoder::GopFrameRecon]) -> Vec<Planes> {
    let up = |p: &[u8]| p.iter().map(|&s| u16::from(s)).collect::<Vec<u16>>();
    r.iter()
        .map(|g| Planes {
            y: up(&g.y),
            u: up(&g.u),
            v: up(&g.v),
        })
        .collect()
}

fn plane_bytes(bit_depth: u8, p: &[u16]) -> Vec<u8> {
    if bit_depth == 8 {
        p.iter().map(|&s| s as u8).collect()
    } else {
        p.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}

/// The planar byte image of a reconstruction list (the layout a raw
/// planar dump of the decoded stream has).
fn recon_bytes(bit_depth: u8, fmt: ChromaFormat, recon: &[Planes]) -> Vec<u8> {
    let mut out = Vec::new();
    for r in recon {
        out.extend(plane_bytes(bit_depth, &r.y));
        if fmt != ChromaFormat::Monochrome {
            out.extend(plane_bytes(bit_depth, &r.u));
            out.extend(plane_bytes(bit_depth, &r.v));
        }
    }
    out
}

/// This crate's decoder reproduces `recon` (frame count, true extent,
/// every plane).
fn assert_own_decode(
    tag: &str,
    ivf: &[u8],
    dims: (u32, u32),
    bit_depth: u8,
    fmt: ChromaFormat,
    recon: &[Planes],
) {
    let frames = decode_av1_spec(ivf).unwrap_or_else(|e| panic!("{tag}: decode failed: {e:?}"));
    assert_eq!(frames.len(), recon.len(), "{tag}: frame count");
    for (i, (f, r)) in frames.iter().zip(recon).enumerate() {
        assert_eq!((f.width, f.height), dims, "{tag}: frame {i} extent");
        assert_eq!(
            f.planes[0],
            plane_bytes(bit_depth, &r.y),
            "{tag}: frame {i} luma"
        );
        if fmt != ChromaFormat::Monochrome {
            assert_eq!(
                f.planes[1],
                plane_bytes(bit_depth, &r.u),
                "{tag}: frame {i} U"
            );
            assert_eq!(
                f.planes[2],
                plane_bytes(bit_depth, &r.v),
                "{tag}: frame {i} V"
            );
        }
    }
}

/// Decode `ivf` with a black-box decoder process and return its raw
/// planar output; `None` when no such binary is installed (the check
/// is skipped at run time).
fn black_box_decode(tag: &str, ivf: &[u8], extra: &[&str]) -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!(
        "oxideav-av1-offgrid-{}-{}",
        std::process::id(),
        tag.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("in.ivf");
    let dst = dir.join("out.yuv");
    std::fs::write(&src, ivf).ok()?;
    let status = Command::new("dav1d")
        .arg("-q")
        .arg("-i")
        .arg(&src)
        .arg("-o")
        .arg(&dst)
        .args(["--muxer", "yuv"])
        .args(extra)
        .status();
    let out = match status {
        Ok(s) => {
            assert!(
                s.success(),
                "{tag}: the black-box decoder rejected the stream"
            );
            Some(std::fs::read(&dst).expect("black-box decoder output"))
        }
        Err(_) => {
            // `OXIDEAV_AV1_REQUIRE_BLACK_BOX=1` turns the run-time
            // skip into a failure (a host that is expected to carry
            // the binary).
            assert!(
                std::env::var_os("OXIDEAV_AV1_REQUIRE_BLACK_BOX").is_none(),
                "{tag}: no black-box decoder on PATH"
            );
            None
        }
    };
    let _ = std::fs::remove_dir_all(&dir);
    out
}

fn assert_black_box(
    tag: &str,
    ivf: &[u8],
    extra: &[&str],
    bit_depth: u8,
    fmt: ChromaFormat,
    recon: &[Planes],
) {
    if let Some(out) = black_box_decode(tag, ivf, extra) {
        assert!(
            out == recon_bytes(bit_depth, fmt, recon),
            "{tag}: black-box decoder output differs from the encoder reconstruction"
        );
    }
}

fn assert_lossless(tag: &str, frames: &[YuvFrame], recon: &[Planes]) {
    for (i, (f, r)) in frames.iter().zip(recon).enumerate() {
        assert!(
            f.y == r.y && f.u == r.u && f.v == r.v,
            "{tag}: lossless frame {i} differs from the input"
        );
    }
}

/// (width, height): odd, ×2, ×4-not-×8, one axis on the grid, and an
/// extent below one mi block on an axis.
const SIZES: &[(u32, u32)] = &[
    (33, 17),
    (34, 18),
    (36, 20),
    (44, 28),
    (71, 49),
    (64, 34),
    (37, 64),
    (5, 9),
];

const FORMATS: &[(u8, ChromaFormat)] = &[
    (8, ChromaFormat::Yuv420),
    (10, ChromaFormat::Yuv420),
    (8, ChromaFormat::Yuv444),
    (10, ChromaFormat::Yuv444),
    (8, ChromaFormat::Yuv422),
    (8, ChromaFormat::Monochrome),
];

fn gop(w: u32, h: u32, bd: u8, fmt: ChromaFormat, n: usize) -> Vec<YuvFrame> {
    (0..n).map(|t| textured(w, h, bd, fmt, t)).collect()
}

#[test]
fn plain_gop_any_extent() {
    for &(w, h) in SIZES {
        for &(bd, fmt) in FORMATS {
            for q in [0u8, 100] {
                let tag = format!("gop {w}x{h} bd{bd} {fmt:?} q{q}");
                let frames = gop(w, h, bd, fmt, 3);
                let enc = encode_gop_yuv_with_q(&frames, q)
                    .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
                let recon = wide(&enc.recon);
                assert_own_decode(&tag, &enc.ivf_bytes, (w, h), bd, fmt, &recon);
                assert_black_box(&tag, &enc.ivf_bytes, &[], bd, fmt, &recon);
                if q == 0 {
                    assert_lossless(&tag, &frames, &recon);
                }
            }
        }
    }
}

#[test]
fn pyramid_and_adaptive_gop_any_extent() {
    for &(w, h) in &SIZES[..5] {
        for &(bd, fmt) in &FORMATS[..4] {
            let frames = gop(w, h, bd, fmt, 6);
            // Lossless on the 8-bit 4:2:0 pairing only (the search is
            // the expensive part; the lossless arm is format-agnostic).
            let qs: &[u8] = if bd == 8 && fmt == ChromaFormat::Yuv420 {
                &[0, 100]
            } else {
                &[100]
            };
            for &q in qs {
                let tag = format!("pyramid {w}x{h} bd{bd} {fmt:?} q{q}");
                let enc = encode_pyramid_gop_yuv_with_q(&frames, q)
                    .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
                let recon = wide(&enc.recon);
                assert_own_decode(&tag, &enc.ivf_bytes, (w, h), bd, fmt, &recon);
                assert_black_box(&tag, &enc.ivf_bytes, &[], bd, fmt, &recon);
                if q == 0 {
                    assert_lossless(&tag, &frames, &recon);
                }
                let tag = format!("adaptive {w}x{h} bd{bd} {fmt:?} q{q}");
                let enc = encode_adaptive_gop_yuv_with_q(&frames, q)
                    .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
                let recon = wide(&enc.recon);
                assert_own_decode(&tag, &enc.ivf_bytes, (w, h), bd, fmt, &recon);
                assert_black_box(&tag, &enc.ivf_bytes, &[], bd, fmt, &recon);
            }
        }
    }
}

#[test]
fn temporal_ladder_any_extent() {
    for &(w, h) in SIZES {
        for &(bd, fmt) in &FORMATS[..4] {
            for layers in [2u8, 3] {
                let tag = format!("temporal{layers} {w}x{h} bd{bd} {fmt:?}");
                let frames = gop(w, h, bd, fmt, 5);
                let enc = encode_temporal_layered_gop_yuv_with_q(&frames, 90, layers)
                    .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
                let recon = wide(&enc.gop.recon);
                assert_own_decode(&tag, &enc.gop.ivf_bytes, (w, h), bd, fmt, &recon);
                assert_black_box(&tag, &enc.gop.ivf_bytes, &[], bd, fmt, &recon);
            }
        }
    }
}

/// (base layer, top layer) extents — both off the grid, one of each,
/// and an odd pair at the 2:1 §6.8.2 bound.
const LAYER_SIZES: &[[(u32, u32); 2]] = &[
    [(34, 18), (66, 36)],
    [(33, 25), (66, 50)],
    [(36, 20), (72, 40)],
    [(40, 24), (75, 45)],
    [(23, 19), (45, 37)],
];

fn check_spatial(
    tag: &str,
    enc: &oxideav_av1::encoder::SpatialLayeredGopYuv,
    dims: &[(u32, u32); 2],
    bd: u8,
    fmt: ChromaFormat,
) {
    assert_eq!(enc.layer_dims, dims.to_vec(), "{tag}: layer extents");
    // Operating point 0 = every layer at every instant: this crate's
    // decoder emits the frames instant-major, layer-minor.
    let out = oxideav_av1::decode_av1_at_operating_point(&enc.ivf_bytes, 0)
        .unwrap_or_else(|e| panic!("{tag}: decode failed: {e:?}"));
    let n = enc.layer_recons[0].len();
    assert_eq!(out.len(), 2 * n, "{tag}: frame count");
    for (k, f) in out.iter().enumerate() {
        let (i, s) = (k / 2, k % 2);
        let oxideav_av1::decoder::Frame::Spec(f) = f else {
            panic!("{tag}: non-Spec frame");
        };
        let r = &enc.layer_recons[s][i];
        assert_eq!((f.width, f.height), dims[s], "{tag}: instant {i} layer {s}");
        assert_eq!(
            f.planes[0],
            plane_bytes(bd, &r.y),
            "{tag}: instant {i} layer {s} luma"
        );
        if fmt != ChromaFormat::Monochrome {
            assert_eq!(
                f.planes[1],
                plane_bytes(bd, &r.u),
                "{tag}: instant {i} layer {s} U"
            );
            assert_eq!(
                f.planes[2],
                plane_bytes(bd, &r.v),
                "{tag}: instant {i} layer {s} V"
            );
        }
    }
    // The black-box decoder outputs the highest spatial layer of the
    // selected operating point.
    let top = wide(&enc.layer_recons[1]);
    assert_black_box(
        tag,
        &enc.ivf_bytes,
        &["--oppoint", "0", "--alllayers", "0"],
        bd,
        fmt,
        &top,
    );
}

#[test]
fn spatial_layers_any_extent() {
    for dims in LAYER_SIZES {
        for &(bd, fmt) in &FORMATS[..4] {
            let layers: Vec<Vec<YuvFrame>> =
                dims.iter().map(|&(w, h)| gop(w, h, bd, fmt, 4)).collect();
            let tag = format!("spatial {dims:?} bd{bd} {fmt:?}");
            let enc = encode_spatial_layered_gop_yuv_with_q(&layers, 90)
                .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
            check_spatial(&tag, &enc, dims, bd, fmt);

            let tag = format!("inter-layer {dims:?} bd{bd} {fmt:?}");
            let enc = encode_spatial_layered_gop_yuv_with_q_inter_layer(&layers, 90, 2)
                .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
            check_spatial(&tag, &enc, dims, bd, fmt);

            let tag = format!("spatial x temporal {dims:?} bd{bd} {fmt:?}");
            let enc = encode_spatial_temporal_layered_gop_yuv_with_q(&layers, 90, 2)
                .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
            check_spatial(&tag, &enc, dims, bd, fmt);
        }
    }
}

/// The 8-bit 4:2:0 sibling entries, `SEG_LVL_ALT_Q` segmentation and
/// an explicit multi-tile layout on off-grid extents.
#[test]
fn siblings_segmentation_and_tiles_any_extent() {
    let narrow = |f: &YuvFrame| Yuv420Frame {
        width: f.width,
        height: f.height,
        y: f.y.iter().map(|&s| s as u8).collect(),
        u: f.u.iter().map(|&s| s as u8).collect(),
        v: f.v.iter().map(|&s| s as u8).collect(),
    };
    for &(w, h) in &[(33u32, 17u32), (70, 46), (5, 9)] {
        let wide = gop(w, h, 8, ChromaFormat::Yuv420, 4);
        let frames: Vec<Yuv420Frame> = wide.iter().map(narrow).collect();

        let tag = format!("yuv420 gop {w}x{h}");
        let enc = encode_gop_yuv420_with_q(&frames, 80)
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
        let recon = narrow8(&enc.recon);
        assert_own_decode(
            &tag,
            &enc.ivf_bytes,
            (w, h),
            8,
            ChromaFormat::Yuv420,
            &recon,
        );
        assert_black_box(&tag, &enc.ivf_bytes, &[], 8, ChromaFormat::Yuv420, &recon);

        let tag = format!("yuv420 segmented gop {w}x{h}");
        let enc = encode_gop_yuv420_with_q_seg(&frames, 80, &[0, 30, -30])
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
        let recon = narrow8(&enc.recon);
        assert_own_decode(
            &tag,
            &enc.ivf_bytes,
            (w, h),
            8,
            ChromaFormat::Yuv420,
            &recon,
        );
        assert_black_box(&tag, &enc.ivf_bytes, &[], 8, ChromaFormat::Yuv420, &recon);

        let tag = format!("yuv420 temporal ladder {w}x{h}");
        let enc = encode_temporal_layered_gop_yuv420_with_q(&frames, 80, 2)
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
        let recon = narrow8(&enc.gop.recon);
        assert_own_decode(
            &tag,
            &enc.gop.ivf_bytes,
            (w, h),
            8,
            ChromaFormat::Yuv420,
            &recon,
        );
        assert_black_box(
            &tag,
            &enc.gop.ivf_bytes,
            &[],
            8,
            ChromaFormat::Yuv420,
            &recon,
        );
    }
    // Two tile columns + two tile rows over a 3 × 2 superblock grid
    // whose last column / row is partial.
    for &(bd, fmt) in &FORMATS[..4] {
        let (w, h) = (150u32, 100u32);
        let tag = format!("tiled gop {w}x{h} bd{bd} {fmt:?}");
        let frames = gop(w, h, bd, fmt, 3);
        let enc = encode_gop_yuv_with_q_tile_layout(&frames, 100, &[2, 1], &[1, 1])
            .unwrap_or_else(|e| panic!("{tag}: encode failed: {e:?}"));
        let recon = wide(&enc.recon);
        assert_own_decode(&tag, &enc.ivf_bytes, (w, h), bd, fmt, &recon);
        assert_black_box(&tag, &enc.ivf_bytes, &[], bd, fmt, &recon);
    }
}
