//! r427 — the chroma/bit-depth encoder axis: KEY + inter GOP
//! conformance across every §6.4.1 (bit depth, chroma format) pairing
//! the three AV1 profiles admit.
//!
//! Every stream here is validated the same way the historical 8-bit
//! 4:2:0 suites are: the crate's own spec-faithful decode driver
//! ([`oxideav_av1::decoder::decode_av1_spec`]) must reproduce the
//! encoder's running reconstruction sample-for-sample, and at
//! `base_q_idx == 0` (CodedLossless) the reconstruction must equal
//! the input exactly. External black-box decoder validation of the
//! same pairings is exercised during the round via the
//! `OXIDEAV_AV1_R427_FIXDIR` dump below and pinned in
//! `fixture_conformance.rs`.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv_with_q, encode_key_frame_yuv_with_q, ChromaFormat, YuvFrame,
};

/// All twelve §6.4.1-admissible (depth, format) pairings.
const PAIRINGS: &[(u8, ChromaFormat)] = &[
    (8, ChromaFormat::Yuv420),
    (10, ChromaFormat::Yuv420),
    (12, ChromaFormat::Yuv420),
    (8, ChromaFormat::Yuv422),
    (10, ChromaFormat::Yuv422),
    (12, ChromaFormat::Yuv422),
    (8, ChromaFormat::Yuv444),
    (10, ChromaFormat::Yuv444),
    (12, ChromaFormat::Yuv444),
    (8, ChromaFormat::Monochrome),
    (10, ChromaFormat::Monochrome),
    (12, ChromaFormat::Monochrome),
];

/// Little-endian byte view of a `u16` plane — the `SpecFrame` output
/// layout for 10/12-bit streams.
fn plane_le(p: &[u16]) -> Vec<u8> {
    p.iter().flat_map(|&s| s.to_le_bytes()).collect()
}

/// 8-bit byte view (the `SpecFrame` layout for 8-bit streams).
fn plane_8(p: &[u16]) -> Vec<u8> {
    p.iter().map(|&s| s as u8).collect()
}

fn plane_bytes(bit_depth: u8, p: &[u16]) -> Vec<u8> {
    if bit_depth == 8 {
        plane_8(p)
    } else {
        plane_le(p)
    }
}

/// Deterministic textured frame at any depth/format: gradients +
/// texture cross-terms scaled to the depth's full range, shifted by
/// `(sy, sx)` for inter content.
fn textured(w: u32, h: u32, bit_depth: u8, fmt: ChromaFormat, sy: usize, sx: usize) -> YuvFrame {
    let mut f = YuvFrame::filled(w, h, bit_depth, fmt, 0);
    let (wu, hu) = (w as usize, h as usize);
    let shift = u32::from(bit_depth - 8);
    for i in 0..hu {
        for j in 0..wu {
            let (si, sj) = (i + sy, j + sx);
            let v8 = (si * 5 + sj * 3 + (si / 16) * (sj / 16)) % 256;
            // Depth-scaled with a small sub-8-bit dither so 10/12-bit
            // samples exercise the extra precision bits.
            let extra = (si * 7 + sj * 13) & ((1usize << shift) - 1);
            f.y[i * wu + j] = ((v8 << shift) + extra) as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for i in 0..ch {
        for j in 0..cw {
            let (si, sj) = (i + sy, j + sx);
            let u8v = (128 + si * 2 + sj) % 256;
            let v8v = (64 + si + sj * 2) % 256;
            let extra = (si * 3 + sj * 5) & ((1usize << shift) - 1);
            f.u[i * cw + j] = ((u8v << shift) + extra) as u16;
            f.v[i * cw + j] = ((v8v << shift) + extra) as u16;
        }
    }
    f
}

/// KEY-frame round trip at one pairing: spec-driver output equals the
/// encoder reconstruction on every plane; lossless equals the input.
fn assert_key_round_trip(bit_depth: u8, fmt: ChromaFormat, q: u8) {
    let input = textured(96, 80, bit_depth, fmt, 0, 0);
    let enc = encode_key_frame_yuv_with_q(&input, q)
        .unwrap_or_else(|e| panic!("bd={bit_depth} {fmt:?} q={q}: encode failed: {e:?}"));
    let frames = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("bd={bit_depth} {fmt:?} q={q}: spec driver rejected: {e:?}"));
    assert_eq!(frames.len(), 1);
    let f = &frames[0];
    assert_eq!(f.bit_depth, bit_depth, "{fmt:?} q={q}");
    let num_planes = if fmt == ChromaFormat::Monochrome {
        1
    } else {
        3
    };
    assert_eq!(f.planes.len(), num_planes, "bd={bit_depth} {fmt:?} q={q}");
    assert_eq!(
        f.planes[0],
        plane_bytes(bit_depth, &enc.recon_y),
        "bd={bit_depth} {fmt:?} q={q}: luma"
    );
    if num_planes == 3 {
        assert_eq!(
            f.planes[1],
            plane_bytes(bit_depth, &enc.recon_u),
            "bd={bit_depth} {fmt:?} q={q}: U"
        );
        assert_eq!(
            f.planes[2],
            plane_bytes(bit_depth, &enc.recon_v),
            "bd={bit_depth} {fmt:?} q={q}: V"
        );
    }
    if q == 0 {
        assert_eq!(
            enc.recon_y, input.y,
            "bd={bit_depth} {fmt:?}: lossless luma"
        );
        assert_eq!(enc.recon_u, input.u, "bd={bit_depth} {fmt:?}: lossless U");
        assert_eq!(enc.recon_v, input.v, "bd={bit_depth} {fmt:?}: lossless V");
    }
}

/// Inter GOP round trip at one pairing (KEY + 2 P-frames, moving
/// content).
fn assert_gop_round_trip(bit_depth: u8, fmt: ChromaFormat, q: u8) {
    let frames: Vec<YuvFrame> = (0..3)
        .map(|k| textured(64, 64, bit_depth, fmt, 2 * k, 3 * k))
        .collect();
    let enc = encode_gop_yuv_with_q(&frames, q)
        .unwrap_or_else(|e| panic!("bd={bit_depth} {fmt:?} q={q}: GOP encode failed: {e:?}"));
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("bd={bit_depth} {fmt:?} q={q}: spec driver rejected: {e:?}"));
    assert_eq!(decoded.len(), frames.len());
    for (idx, f) in decoded.iter().enumerate() {
        assert_eq!(f.bit_depth, bit_depth);
        assert_eq!(
            f.planes[0],
            plane_bytes(bit_depth, &enc.recon[idx].y),
            "bd={bit_depth} {fmt:?} q={q} frame {idx}: luma"
        );
        if fmt != ChromaFormat::Monochrome {
            assert_eq!(
                f.planes[1],
                plane_bytes(bit_depth, &enc.recon[idx].u),
                "bd={bit_depth} {fmt:?} q={q} frame {idx}: U"
            );
            assert_eq!(
                f.planes[2],
                plane_bytes(bit_depth, &enc.recon[idx].v),
                "bd={bit_depth} {fmt:?} q={q} frame {idx}: V"
            );
        }
        if q == 0 {
            assert_eq!(enc.recon[idx].y, frames[idx].y, "lossless frame {idx} luma");
            assert_eq!(enc.recon[idx].u, frames[idx].u, "lossless frame {idx} U");
            assert_eq!(enc.recon[idx].v, frames[idx].v, "lossless frame {idx} V");
        }
    }
}

// --- KEY sweeps: one test per pairing (lossless + one lossy point) ---

#[test]
fn key_10bit_420_round_trips() {
    assert_key_round_trip(10, ChromaFormat::Yuv420, 0);
    assert_key_round_trip(10, ChromaFormat::Yuv420, 60);
}

#[test]
fn key_12bit_420_round_trips() {
    assert_key_round_trip(12, ChromaFormat::Yuv420, 0);
    assert_key_round_trip(12, ChromaFormat::Yuv420, 110);
}

#[test]
fn key_444_round_trips_all_depths() {
    assert_key_round_trip(8, ChromaFormat::Yuv444, 0);
    assert_key_round_trip(8, ChromaFormat::Yuv444, 60);
    assert_key_round_trip(10, ChromaFormat::Yuv444, 90);
    assert_key_round_trip(12, ChromaFormat::Yuv444, 0);
}

#[test]
fn key_422_round_trips_all_depths() {
    assert_key_round_trip(8, ChromaFormat::Yuv422, 0);
    assert_key_round_trip(8, ChromaFormat::Yuv422, 60);
    assert_key_round_trip(10, ChromaFormat::Yuv422, 0);
    assert_key_round_trip(12, ChromaFormat::Yuv422, 130);
}

#[test]
fn key_monochrome_round_trips_all_depths() {
    assert_key_round_trip(8, ChromaFormat::Monochrome, 0);
    assert_key_round_trip(8, ChromaFormat::Monochrome, 60);
    assert_key_round_trip(10, ChromaFormat::Monochrome, 90);
    assert_key_round_trip(12, ChromaFormat::Monochrome, 0);
}

/// The 8-bit 4:2:0 arm of the general entry must stay byte-identical
/// to the historical `encode_key_frame_yuv420_with_q` (which now
/// routes through the same core).
#[test]
fn key_8bit_420_general_entry_matches_legacy_bytes() {
    use oxideav_av1::encoder::{encode_key_frame_yuv420_with_q, Yuv420Frame};
    let (w, h) = (96u32, 80u32);
    let mut legacy = Yuv420Frame::filled(w, h, 128);
    for (i, p) in legacy.y.iter_mut().enumerate() {
        *p = ((i * 7) % 256) as u8;
    }
    for (i, p) in legacy.u.iter_mut().enumerate() {
        *p = ((i * 3 + 40) % 256) as u8;
    }
    for (i, p) in legacy.v.iter_mut().enumerate() {
        *p = ((i * 5 + 90) % 256) as u8;
    }
    let wide = YuvFrame::from_yuv420_8bit(&legacy);
    for q in [0u8, 60, 255] {
        let a = encode_key_frame_yuv420_with_q(&legacy, q).unwrap();
        let b = encode_key_frame_yuv_with_q(&wide, q).unwrap();
        assert_eq!(a.ivf_bytes, b.ivf_bytes, "q={q}: general != legacy bytes");
    }
}

// --- Inter GOP sweeps ---

#[test]
fn gop_10bit_420_round_trips() {
    assert_gop_round_trip(10, ChromaFormat::Yuv420, 0);
    assert_gop_round_trip(10, ChromaFormat::Yuv420, 60);
}

#[test]
fn gop_444_round_trips() {
    assert_gop_round_trip(8, ChromaFormat::Yuv444, 60);
    assert_gop_round_trip(10, ChromaFormat::Yuv444, 0);
}

#[test]
fn gop_422_round_trips() {
    assert_gop_round_trip(8, ChromaFormat::Yuv422, 0);
    assert_gop_round_trip(10, ChromaFormat::Yuv422, 72);
}

#[test]
fn gop_12bit_round_trips() {
    assert_gop_round_trip(12, ChromaFormat::Yuv420, 60);
    assert_gop_round_trip(12, ChromaFormat::Yuv444, 0);
}

#[test]
fn gop_monochrome_round_trips() {
    assert_gop_round_trip(8, ChromaFormat::Monochrome, 0);
    assert_gop_round_trip(10, ChromaFormat::Monochrome, 60);
}

/// Sequence-header profile election is visible on the wire: parse the
/// emitted stream's SH for every pairing and check `seq_profile` and
/// the §5.5.2 color-config fields.
#[test]
fn emitted_profiles_match_the_6_4_1_table() {
    for &(bd, fmt) in PAIRINGS {
        let input = textured(64, 64, bd, fmt, 0, 0);
        let enc = encode_key_frame_yuv_with_q(&input, 60).unwrap();
        let expect = oxideav_av1::encoder::elect_seq_profile(bd, fmt).unwrap();
        assert_eq!(
            enc.seq.seq_profile, expect,
            "bd={bd} {fmt:?}: wrong profile"
        );
        assert_eq!(enc.seq.color_config.bit_depth, bd);
        let (ssx, ssy) = fmt.subsampling();
        assert_eq!(enc.seq.color_config.subsampling_x, ssx == 1);
        assert_eq!(enc.seq.color_config.subsampling_y, ssy == 1);
        assert_eq!(
            enc.seq.color_config.mono_chrome,
            fmt == ChromaFormat::Monochrome
        );
    }
}

/// r427 fixture dump — writes one KEY IVF + one 3-frame GOP IVF per
/// pairing (plus raw recon planes in the spec-driver byte layout) for
/// external black-box decoder validation and corpus pinning. Inert
/// unless `OXIDEAV_AV1_R427_FIXDIR` is set.
#[test]
fn r427_format_matrix_fixture_dump() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_R427_FIXDIR") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    for &(bd, fmt) in PAIRINGS {
        let tag = format!(
            "{}bit-{}",
            bd,
            match fmt {
                ChromaFormat::Yuv420 => "420",
                ChromaFormat::Yuv422 => "422",
                ChromaFormat::Yuv444 => "444",
                ChromaFormat::Monochrome => "mono",
            }
        );
        // KEY pin: 96x80 textured, q=60.
        let input = textured(96, 80, bd, fmt, 0, 0);
        let enc = encode_key_frame_yuv_with_q(&input, 60).unwrap();
        std::fs::write(format!("{dir}/self-kf-96x80-q60-{tag}.ivf"), &enc.ivf_bytes).unwrap();
        let mut yuv = Vec::new();
        yuv.extend_from_slice(&plane_bytes(bd, &enc.recon_y));
        yuv.extend_from_slice(&plane_bytes(bd, &enc.recon_u));
        yuv.extend_from_slice(&plane_bytes(bd, &enc.recon_v));
        std::fs::write(format!("{dir}/self-kf-96x80-q60-{tag}.yuv"), &yuv).unwrap();
        // GOP pin: 64x64 3-frame moving texture, q=60.
        let frames: Vec<YuvFrame> = (0..3)
            .map(|k| textured(64, 64, bd, fmt, 2 * k, 3 * k))
            .collect();
        let enc = encode_gop_yuv_with_q(&frames, 60).unwrap();
        std::fs::write(
            format!("{dir}/self-gop-64x64-q60-{tag}.ivf"),
            &enc.ivf_bytes,
        )
        .unwrap();
        let mut yuv = Vec::new();
        for rc in &enc.recon {
            yuv.extend_from_slice(&plane_bytes(bd, &rc.y));
            yuv.extend_from_slice(&plane_bytes(bd, &rc.u));
            yuv.extend_from_slice(&plane_bytes(bd, &rc.v));
        }
        std::fs::write(format!("{dir}/self-gop-64x64-q60-{tag}.yuv"), &yuv).unwrap();
    }
    eprintln!("r427 format-matrix pins dumped to {dir}");
}

/// r427 — the deep-pyramid driver at the new pairings (cheap sweep:
/// one 5-frame pyramid at 10-bit 4:2:0 and one at 8-bit 4:4:4): the
/// out-of-order coding, show_existing units and §7.20 slot rotation
/// must hold at any depth/format exactly like the flat GOP.
#[test]
fn pyramid_round_trips_at_new_pairings() {
    use oxideav_av1::encoder::encode_pyramid_gop_yuv_with_q;
    for &(bd, fmt) in &[(10u8, ChromaFormat::Yuv420), (8u8, ChromaFormat::Yuv444)] {
        let frames: Vec<YuvFrame> = (0..5)
            .map(|k| textured(64, 64, bd, fmt, k, 2 * k))
            .collect();
        let enc = encode_pyramid_gop_yuv_with_q(&frames, 60)
            .unwrap_or_else(|e| panic!("bd={bd} {fmt:?}: pyramid encode failed: {e:?}"));
        let decoded = decode_av1_spec(&enc.ivf_bytes)
            .unwrap_or_else(|e| panic!("bd={bd} {fmt:?}: spec driver rejected: {e:?}"));
        assert_eq!(decoded.len(), frames.len());
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.bit_depth, bd);
            assert_eq!(
                f.planes[0],
                plane_bytes(bd, &enc.recon[idx].y),
                "bd={bd} {fmt:?} display {idx}: luma"
            );
            assert_eq!(
                f.planes[1],
                plane_bytes(bd, &enc.recon[idx].u),
                "bd={bd} {fmt:?} display {idx}: U"
            );
            assert_eq!(
                f.planes[2],
                plane_bytes(bd, &enc.recon[idx].v),
                "bd={bd} {fmt:?} display {idx}: V"
            );
        }
        // External black-box validation of these exact streams runs
        // during the round via the matrix dump flow.
        if let Ok(dir) = std::env::var("OXIDEAV_AV1_R427_FIXDIR") {
            std::fs::create_dir_all(&dir).unwrap();
            let tag = format!(
                "{}bit-{}",
                bd,
                if fmt == ChromaFormat::Yuv444 {
                    "444"
                } else {
                    "420"
                }
            );
            std::fs::write(
                format!("{dir}/self-pyr-64x64-q60-{tag}.ivf"),
                &enc.ivf_bytes,
            )
            .unwrap();
            let mut yuv = Vec::new();
            for rc in &enc.recon {
                yuv.extend_from_slice(&plane_bytes(bd, &rc.y));
                yuv.extend_from_slice(&plane_bytes(bd, &rc.u));
                yuv.extend_from_slice(&plane_bytes(bd, &rc.v));
            }
            std::fs::write(format!("{dir}/self-pyr-64x64-q60-{tag}.yuv"), &yuv).unwrap();
        }
    }
}
