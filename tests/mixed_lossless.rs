//! r426 — per-segment lossless mixing (§5.9.14 `SEG_LVL_ALT_Q` down
//! to qindex 0 + the §5.9.2 `LosslessArray[]` leaf semantics inside
//! an otherwise-lossy frame).
//!
//! Layer 1 (this file's `mechanism_*` tests): the SEG_LVL_ALT_Q
//! election is driven by the r413 activity policy — a delta table
//! whose top segment clamps to qindex 0 makes every textured leaf a
//! lossless-segment leaf. Every stream must decode through the
//! crate's spec-faithful frame driver byte-exact to the encoder's own
//! reconstruction (the anti-desync gate — mixed-segment superblocks
//! share one adaptive CDF/context state between the TX_4X4/WHT
//! lossless leaves and the lossy transform-tree leaves).
//!
//! Layer 2 (`region_*` tests): the r426 exactness-demand API — a
//! caller-supplied pixel region is coded pixel-exact (asserted
//! against the INPUT planes on every frame) while the rest of the
//! frame stays on the lossy ladder.

use oxideav_av1::decoder::decode_av1_spec;
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q_lossless_regions, encode_gop_yuv420_with_q_seg_lossless_tuned,
    encode_gop_yuv420_with_q_seg_tuned, GopTuning, LosslessRegion, Yuv420Frame,
};

/// Deterministic frame: a flat left half (luma 64, so the r413
/// activity policy maps its leaves to segment 0) and a textured right
/// half (high mean-absolute-deviation, mapping to the top segment),
/// with per-frame translation `(sy, sx)` on the textured half.
fn half_flat_half_texture(
    w: u32,
    h: u32,
    shift_y: usize,
    shift_x: usize,
    seed: u32,
) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let s = seed as usize;
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            f.y[i * wu + j] = if j < wu / 2 {
                64
            } else {
                // Two-level aperiodic synthetic texture (binary
                // alphabet, high activity): the lossy ladder distorts
                // it heavily at any quantiser while the WHT/qindex-0
                // chain codes it exactly — the RD election must send
                // these leaves through the lossless segment.
                let (si, sj) = (i + shift_y, j + shift_x);
                let hash = (si.wrapping_mul(2654435761) ^ sj.wrapping_mul(40503)).wrapping_add(s);
                if (hash >> 4) & 1 == 1 {
                    235
                } else {
                    16
                }
            };
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = if j < cw / 2 {
                120
            } else {
                ((i * 3 + j * 5 + s) % 256) as u8
            };
            f.v[i * cw + j] = if j < cw / 2 {
                130
            } else {
                ((i * 7 + j + 2 * s) % 256) as u8
            };
        }
    }
    f
}

/// Encode a segmented GOP whose TOP segment clamps to qindex 0,
/// decode through the spec driver, and assert the anti-desync
/// invariant plus real mixed-segment coverage in the committed maps.
fn assert_mixed_round_trip(
    frames: &[Yuv420Frame],
    q: u8,
    alt_q: &[i16],
    expect_lossless_cells: bool,
) {
    let (w, h) = (frames[0].width, frames[0].height);
    let tuned = encode_gop_yuv420_with_q_seg_tuned(frames, q, alt_q, GopTuning::default())
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: mixed encode failed: {e:?}"));
    let enc = &tuned.gop;
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: spec driver rejected mixed GOP: {e:?}"));
    assert_eq!(decoded.len(), frames.len(), "{w}x{h} q={q}: frame count");
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.recon[idx];
        assert_eq!(
            f.planes[0], rc.y,
            "{w}x{h} q={q} frame {idx}: luma decode != encoder recon"
        );
        assert_eq!(
            f.planes[1], rc.u,
            "{w}x{h} q={q} frame {idx}: U decode != encoder recon"
        );
        assert_eq!(
            f.planes[2], rc.v,
            "{w}x{h} q={q} frame {idx}: V decode != encoder recon"
        );
    }
    // Real mixing: the committed maps must reach BOTH the lossy
    // segment 0 (every P-frame — the flat half) and the lossless top
    // segment (at least one P-frame; skip leaves inherit the
    // §5.11.20 pred cascade bit-silently, so frames whose textured
    // half collapses to skips legitimately show fewer coded cells).
    let top = (alt_q.len() - 1) as i32;
    assert_eq!(tuned.p_segment_maps.len(), frames.len() - 1);
    for (k, map) in tuned.p_segment_maps.iter().enumerate() {
        assert!(
            map.contains(&0),
            "{w}x{h} q={q} P{k}: no segment-0 (lossy) cells"
        );
    }
    if expect_lossless_cells {
        assert!(
            tuned.p_segment_maps.iter().any(|map| map.contains(&top)),
            "{w}x{h} q={q}: no P-frame committed a lossless-segment cell"
        );
    }
}

/// Two-segment mixed TABLE at q = 60 — the anti-desync roundtrip
/// witness for a header whose LosslessArray is mixed (whether the
/// activity election commits lossless cells on this content is the
/// RD ladder's call; the guaranteed-commitment witnesses are the
/// `region_*` tests).
#[test]
fn mechanism_two_segment_mix_q60_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| half_flat_half_texture(64, 64, 2 * k, 3 * k, 5))
        .collect();
    assert_mixed_round_trip(&frames, 60, &[0, -60], false);
}

/// Three-segment ladder at q = 72 whose top segment clamps to 0 —
/// lossy / lossy-delta / lossless blocks share superblocks.
#[test]
fn mechanism_three_segment_ladder_q72_round_trips() {
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| half_flat_half_texture(64, 64, 3 * k, 5 * k, 11))
        .collect();
    assert_mixed_round_trip(&frames, 72, &[0, 24, -72], false);
}

/// Multi-superblock mixed frame (96x80) with a content cut on the
/// last frame — the §5.11.22 intra-fallback arm fires next to
/// lossless-segment inter leaves.
#[test]
fn mechanism_multi_superblock_cut_q100_round_trips() {
    let mut frames: Vec<Yuv420Frame> = (0..2)
        .map(|k| half_flat_half_texture(96, 80, 2 * k, 4 * k, 3))
        .collect();
    frames.push(half_flat_half_texture(96, 80, 40, 70, 91));
    assert_mixed_round_trip(&frames, 100, &[0, -255], true);
}

/// "Camera" frame: smooth moving gradients everywhere, with a
/// synthetic text/UI panel (tiny luma alphabet, hard edges) at
/// `(px, py)`, `pw × ph` luma samples, translated with the content.
fn camera_with_text_panel(
    w: u32,
    h: u32,
    shift: usize,
    panel: (usize, usize, usize, usize),
) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let (px, py, pw, ph) = panel;
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            f.y[i * wu + j] = if j >= px && j < px + pw && i >= py && i < py + ph {
                // Text-like glyph pattern: 2 values, hard edges,
                // aperiodic — the §5.11.22/§5.11.47 lossy arms ring
                // on it at any quantiser.
                let (gi, gj) = (i - py, j - px);
                let hash = gi.wrapping_mul(2654435761) ^ gj.wrapping_mul(40503);
                if (hash >> 3) & 3 == 0 {
                    16
                } else {
                    235
                }
            } else {
                // Smooth camera background with motion.
                let (si, sj) = (i + shift, j + 2 * shift);
                (128 + ((si * 2 + sj) % 96)) as u8
            };
        }
    }
    let (cw, ch) = (wu / 2, hu / 2);
    for i in 0..ch {
        for j in 0..cw {
            let in_panel = 2 * j >= px && 2 * j < px + pw && 2 * i >= py && 2 * i < py + ph;
            f.u[i * cw + j] = if in_panel {
                200
            } else {
                ((110 + i + j + shift) % 256) as u8
            };
            f.v[i * cw + j] = if in_panel {
                40
            } else {
                ((140 + 2 * i + j) % 256) as u8
            };
        }
    }
    f
}

/// Encode through the exactness-demand entry point, decode through
/// the spec driver, and assert: (a) the anti-desync invariant on
/// every frame; (b) the demanded region decodes PIXEL-EXACT against
/// the INPUT on every P-frame — luma over the rectangle, chroma over
/// its 4:2:0 footprint.
fn assert_region_exact(frames: &[Yuv420Frame], q: u8, regions: &[LosslessRegion], auto: bool) {
    let (w, h) = (frames[0].width, frames[0].height);
    let enc = encode_gop_yuv420_with_q_lossless_regions(frames, q, regions, auto)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: region encode failed: {e:?}"));
    let decoded = decode_av1_spec(&enc.ivf_bytes)
        .unwrap_or_else(|e| panic!("{w}x{h} q={q}: spec driver rejected region GOP: {e:?}"));
    assert_eq!(decoded.len(), frames.len(), "{w}x{h} q={q}: frame count");
    let (wu, cw) = (w as usize, (w / 2) as usize);
    for (idx, f) in decoded.iter().enumerate() {
        let rc = &enc.recon[idx];
        assert_eq!(f.planes[0], rc.y, "{w}x{h} q={q} frame {idx}: luma desync");
        assert_eq!(f.planes[1], rc.u, "{w}x{h} q={q} frame {idx}: U desync");
        assert_eq!(f.planes[2], rc.v, "{w}x{h} q={q} frame {idx}: V desync");
        // r426 — the KEY frame is segmented too (the §5.11.8 per-block
        // intra_segment_id arm): the exactness contract covers EVERY
        // frame of the GOP, frame 0 included.
        let input = &frames[idx];
        for reg in regions {
            let x0 = reg.x.min(w) as usize;
            let y0 = reg.y.min(h) as usize;
            let x1 = (reg.x + reg.width).min(w) as usize;
            let y1 = (reg.y + reg.height).min(h) as usize;
            for y in y0..y1 {
                for x in x0..x1 {
                    assert_eq!(
                        f.planes[0][y * wu + x],
                        input.y[y * wu + x],
                        "{w}x{h} q={q} frame {idx}: luma ({x},{y}) not exact"
                    );
                }
            }
            for cy in (y0 / 2)..y1.div_ceil(2).min((h as usize) / 2) {
                for cx in (x0 / 2)..x1.div_ceil(2).min(wu / 2) {
                    assert_eq!(
                        f.planes[1][cy * cw + cx],
                        input.u[cy * cw + cx],
                        "{w}x{h} q={q} frame {idx}: U ({cx},{cy}) not exact"
                    );
                    assert_eq!(
                        f.planes[2][cy * cw + cx],
                        input.v[cy * cw + cx],
                        "{w}x{h} q={q} frame {idx}: V ({cx},{cy}) not exact"
                    );
                }
            }
        }
    }
}

/// The canonical case: a pixel-exact text/UI panel over a lossy
/// moving camera background at q = 60.
#[test]
fn region_text_panel_over_camera_q60_pixel_exact() {
    let panel = (32usize, 16usize, 24usize, 16usize);
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| camera_with_text_panel(64, 64, 3 * k, panel))
        .collect();
    let regions = [LosslessRegion {
        x: 32,
        y: 16,
        width: 24,
        height: 16,
    }];
    assert_region_exact(&frames, 60, &regions, false);
}

/// Unaligned demand rectangle (odd origin/size — exercises the mi
/// clipping + 2×2-group dilation) at a harsh quantiser.
#[test]
fn region_unaligned_rect_q160_pixel_exact() {
    let panel = (21usize, 9usize, 19usize, 14usize);
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| camera_with_text_panel(64, 64, 2 * k, panel))
        .collect();
    let regions = [LosslessRegion {
        x: 21,
        y: 9,
        width: 19,
        height: 14,
    }];
    assert_region_exact(&frames, 160, &regions, false);
}

/// Two disjoint demand rectangles on a multi-superblock frame, with
/// the auto-lossless election live at the same time.
#[test]
fn region_two_rects_multi_sb_q100_pixel_exact() {
    let panel = (48usize, 24usize, 32usize, 24usize);
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| camera_with_text_panel(96, 80, 2 * k, panel))
        .collect();
    let regions = [
        LosslessRegion {
            x: 48,
            y: 24,
            width: 32,
            height: 24,
        },
        LosslessRegion {
            x: 0,
            y: 64,
            width: 20,
            height: 12,
        },
    ];
    assert_region_exact(&frames, 100, &regions, true);
}

/// The demand machinery works with an arbitrary caller table too —
/// the lossless segment sits at index 2 behind a lossy delta ladder.
#[test]
fn region_with_three_segment_table_round_trips() {
    let panel = (32usize, 32usize, 16usize, 16usize);
    let frames: Vec<Yuv420Frame> = (0..3)
        .map(|k| {
            let mut f = camera_with_text_panel(64, 64, 3 * k, panel);
            // "Typing": the panel content CHANGES every frame, so the
            // P-frames must re-code the demanded leaves (a static
            // panel legitimately collapses to bit-silent skip leaves
            // once the segmented KEY frame is exact).
            for y in 32..48usize {
                for x in 32..48usize {
                    let h = (y * 31 + x * 17 + 97 * k) % 5;
                    f.y[y * 64 + x] = if h < 2 { 16 } else { 235 };
                }
            }
            f
        })
        .collect();
    let regions = [LosslessRegion {
        x: 32,
        y: 32,
        width: 16,
        height: 16,
    }];
    let tuned = encode_gop_yuv420_with_q_seg_lossless_tuned(
        &frames,
        72,
        &[0, 24, -72],
        &regions,
        false,
        GopTuning::default(),
    )
    .expect("three-segment region encode");
    let decoded = decode_av1_spec(&tuned.gop.ivf_bytes).expect("spec driver decode");
    for (idx, f) in decoded.iter().enumerate().skip(1) {
        let input = &frames[idx];
        for y in 32..48usize {
            for x in 32..48usize {
                assert_eq!(
                    f.planes[0][y * 64 + x],
                    input.y[y * 64 + x],
                    "frame {idx}: luma ({x},{y}) not exact"
                );
            }
        }
    }
    // The demanded leaves must sit on segment 2 (the table's
    // lossless slot) wherever they are CODED — P1 must code them
    // (its reference is the lossy KEY frame); later frames may
    // legitimately collapse to skip leaves (exact prediction from an
    // already-exact region inherits the §5.11.20 pred bit-silently).
    assert!(
        tuned.p_segment_maps[0].contains(&2),
        "P1: no segment-2 (lossless) cells committed"
    );
}

/// The relaxed validation still rejects the malformed tables: a
/// base-0 segmented GOP, su(1+8)-unrepresentable deltas, and over-255
/// sums.
#[test]
fn mechanism_validation_rejects_malformed_tables() {
    let frames: Vec<Yuv420Frame> = (0..2)
        .map(|k| half_flat_half_texture(64, 64, k, k, 1))
        .collect();
    for (q, alt_q) in [
        (0u8, &[0i16, -255][..]), // base-0 segmented
        (60, &[0, -256][..]),     // below su(1+8)
        (60, &[0, 256][..]),      // above su(1+8)
        (200, &[0, 100][..]),     // qindex > 255
        (60, &[5, -255][..]),     // segment 0 must ride base_q
    ] {
        assert!(
            encode_gop_yuv420_with_q_seg_tuned(&frames, q, alt_q, GopTuning::default()).is_err(),
            "q={q} alt_q={alt_q:?} must be rejected"
        );
    }
}

/// r426 measurement + pin-generation harness — the "typing panel over
/// moving camera" matrix. For each config the SAME input GOP encodes
/// four ways:
///
///   * `lossy`  — plain `encode_gop_yuv420_with_q` at `q` (no
///     exactness anywhere; the rate floor),
///   * `mixed`  — `encode_gop_yuv420_with_q_lossless_regions` (the
///     panel PIXEL-EXACT on every frame, everything else lossy),
///   * `auto`   — the same plus the content-driven election outside
///     the demand mask,
///   * `full-lossless` — plain `q = 0` (everything exact; the rate
///     ceiling the mixed arm must undercut).
///
/// Always-on tripwire: `mixed` must undercut `full-lossless` (the
/// demanded panel is a small fraction of the frame). `mixed` vs
/// `lossy` is measured, NOT asserted — on synthetic panels the
/// qindex-0 WHT chain can beat the lossy transform path outright
/// (no ringing re-coding, exact references collapsing later frames
/// to skips), so exactness there is better than free. With
/// `OXIDEAV_AV1_MIXLL_AB_DIR` set, the harness dumps a CSV of all
/// four byte counts per config plus the `mixed` IVF streams (the
/// conformance-pin candidates).
#[test]
fn mixed_ab_measurement_matrix() {
    struct Cfg {
        name: &'static str,
        w: u32,
        h: u32,
        q: u8,
        frames: usize,
        panel: (usize, usize, usize, usize),
        /// A SECOND typing panel OUTSIDE the demand mask — the
        /// auto-lossless election's material (`None` on the pure
        /// demand configs).
        free_panel: Option<(usize, usize, usize, usize)>,
    }
    let cfgs = [
        Cfg {
            name: "typing-64x64-q60",
            w: 64,
            h: 64,
            q: 60,
            frames: 3,
            panel: (32, 16, 24, 16),
            free_panel: None,
        },
        Cfg {
            name: "typing-96x80-q100",
            w: 96,
            h: 80,
            q: 100,
            frames: 3,
            panel: (48, 24, 32, 24),
            free_panel: None,
        },
        Cfg {
            name: "typing-96x80-q160",
            w: 96,
            h: 80,
            q: 160,
            frames: 4,
            panel: (21, 9, 43, 30),
            free_panel: None,
        },
        Cfg {
            name: "two-panels-96x80-q100",
            w: 96,
            h: 80,
            q: 100,
            frames: 3,
            panel: (48, 8, 32, 16),
            free_panel: Some((8, 48, 32, 16)),
        },
        // Antialiased free panel: a 12-value luma alphabet — past the
        // §5.11.46 PALETTE_COLORS = 8 ceiling, so neither the palette
        // nor the lossy ladder can code it exactly; the auto-lossless
        // election is the only exact arm.
        Cfg {
            name: "aa-panel-96x80-q100",
            w: 96,
            h: 80,
            q: 100,
            frames: 3,
            panel: (48, 8, 32, 16),
            free_panel: Some((8, 48, 32, 16)),
        },
    ];
    let dump_dir = std::env::var_os("OXIDEAV_AV1_MIXLL_AB_DIR");
    let mut csv =
        String::from("config,lossy,mixed,auto,full_lossless,free_ssd_mixed,free_ssd_auto\n");
    for cfg in &cfgs {
        let (px, py, pw, ph) = cfg.panel;
        let frames: Vec<Yuv420Frame> = (0..cfg.frames)
            .map(|k| {
                let mut f = camera_with_text_panel(cfg.w, cfg.h, 2 * k, cfg.panel);
                // Typing: panel content changes every frame.
                let wu = cfg.w as usize;
                for y in py..py + ph {
                    for x in px..px + pw {
                        let hsh = (y * 31 + x * 17 + 97 * k) % 5;
                        f.y[y * wu + x] = if hsh < 2 { 16 } else { 235 };
                    }
                }
                if let Some((fx, fy, fw, fh)) = cfg.free_panel {
                    let antialiased = cfg.name.starts_with("aa-");
                    for y in fy..fy + fh {
                        for x in fx..fx + fw {
                            let hsh = (y * 13 + x * 29 + 41 * k) % 5;
                            f.y[y * wu + x] = if antialiased {
                                // Irregular 12-value alphabet (glyph
                                // body / background / edge shades at
                                // aperiodic positions): past the
                                // palette ceiling AND transform-
                                // hostile — the lossy arms ring on it.
                                let e = (y.wrapping_mul(2654435761)
                                    ^ x.wrapping_mul(40503)
                                    ^ k.wrapping_mul(97))
                                    % 12;
                                (20 + e * 19) as u8
                            } else if hsh < 2 {
                                20
                            } else {
                                230
                            };
                        }
                    }
                }
                f
            })
            .collect();
        let regions = [LosslessRegion {
            x: px as u32,
            y: py as u32,
            width: pw as u32,
            height: ph as u32,
        }];
        let lossy = oxideav_av1::encoder::encode_gop_yuv420_with_q(&frames, cfg.q)
            .expect("lossy encode")
            .ivf_bytes;
        let mixed = encode_gop_yuv420_with_q_lossless_regions(&frames, cfg.q, &regions, false)
            .expect("mixed encode")
            .ivf_bytes;
        let auto = encode_gop_yuv420_with_q_lossless_regions(&frames, cfg.q, &regions, true)
            .expect("auto encode")
            .ivf_bytes;
        let full_ll = oxideav_av1::encoder::encode_gop_yuv420_with_q(&frames, 0)
            .expect("lossless encode")
            .ivf_bytes;
        assert!(
            mixed.len() < full_ll.len(),
            "{}: mixed ({}) must undercut full-lossless ({})",
            cfg.name,
            mixed.len(),
            full_ll.len()
        );
        // Pin safety: the demanded panel decodes pixel-exact on
        // EVERY frame of the mixed stream (the dumped IVFs are the
        // conformance-pin candidates).
        {
            let decoded = decode_av1_spec(&mixed).expect("spec decode of mixed");
            let wu = cfg.w as usize;
            for (idx, f) in decoded.iter().enumerate() {
                for y in py..py + ph {
                    for x in px..px + pw {
                        assert_eq!(
                            f.planes[0][y * wu + x],
                            frames[idx].y[y * wu + x],
                            "{} frame {idx}: panel luma ({x},{y}) not exact",
                            cfg.name
                        );
                    }
                }
            }
        }
        // Informational: the free panel's decoded luma SSD under the
        // mixed vs auto arms (the auto election is content-driven —
        // measured, never asserted).
        let free_ssd = |ivf: &[u8]| -> u64 {
            let Some((fx, fy, fw, fh)) = cfg.free_panel else {
                return 0;
            };
            let decoded = decode_av1_spec(ivf).expect("spec decode");
            let wu = cfg.w as usize;
            let mut ssd = 0u64;
            for (idx, f) in decoded.iter().enumerate() {
                for y in fy..fy + fh {
                    for x in fx..fx + fw {
                        let d = i64::from(f.planes[0][y * wu + x])
                            - i64::from(frames[idx].y[y * wu + x]);
                        ssd += (d * d) as u64;
                    }
                }
            }
            ssd
        };
        csv.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            cfg.name,
            lossy.len(),
            mixed.len(),
            auto.len(),
            full_ll.len(),
            free_ssd(&mixed),
            free_ssd(&auto)
        ));
        if let Some(dir) = dump_dir.as_ref() {
            let dir = std::path::Path::new(dir);
            std::fs::create_dir_all(dir).expect("dump dir");
            std::fs::write(dir.join(format!("{}-mixed.ivf", cfg.name)), &mixed).expect("dump ivf");
        }
    }
    if let Some(dir) = dump_dir.as_ref() {
        let dir = std::path::Path::new(dir);
        std::fs::write(dir.join("mixll_ab.csv"), csv).expect("dump csv");
    }
}
