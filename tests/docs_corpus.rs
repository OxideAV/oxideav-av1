//! Integration tests against the docs/video/av1/ fixture corpus.
//!
//! Each fixture under `../../docs/video/av1/fixtures/<name>/` ships an
//! `input.ivf` (raw AV1 OBU stream wrapped in IVF), an
//! `expected.yuv` byte-for-byte ground truth, a `notes.md` describing
//! the bitstream feature focus, and a `trace.txt` that captures the
//! per-step OBU + frame-header events that an instrumented FFmpeg
//! `cbs_av1` frontend emitted on this clip. Pixel reconstruction in
//! the corpus was performed by libdav1d 1.5.0; we treat the
//! `expected.yuv` as the ground-truth target.
//!
//! This driver decodes every fixture through the in-tree
//! [`Av1Decoder`] and reports the per-fixture pixel-match rate against
//! the expected YUV.
//!
//! Acceptance:
//! * Tier::BitExact — must round-trip exactly. Failure = CI red.
//! * Tier::ReportOnly — divergence is logged but the test does NOT
//!   fail. Use this for fixtures that exercise codec features the
//!   in-tree decoder is still bringing up (multi-ref, super-res,
//!   show-existing-frame, IntraBC, multi-tile, 128 SBs, HBD, …).
//! * Tier::Ignored — disabled with #[ignore]; for fixtures that
//!   require infrastructure that does not yet exist.
//!
//! All fixtures start as ReportOnly. As the decoder progresses,
//! individual fixtures should be promoted to BitExact in the matrix
//! below.
//!
//! The trace.txt files are NOT consumed by this driver — they are an
//! aid for the human implementer when localising divergences. Each
//! per-fixture `evaluate()` call references the trace path in the
//! eprintln! header so that a failing run prints a clickable pointer.
//!
//! Spec references throughout follow the **AV1 Bitstream & Decoding
//! Process Specification** in `docs/video/av1/av1-spec.pdf`. Per the
//! workspace policy in `feedback_no_external_libs.md`, NO external
//! decoder source (libaom, libdav1d, libavcodec, etc.) was consulted
//! while writing this driver — fixtures are data, traces are
//! behavioural diff targets, the spec PDF is the authority.

use std::fs;
use std::path::PathBuf;

use oxideav_av1::decoder::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Error, Frame, Packet, TimeBase};

const IVF_HEADER_SIGNATURE_LEN: usize = 4;
const IVF_FRAME_HEADER_LEN: usize = 12;

/// Locate `docs/video/av1/fixtures/<name>/`. Tests run with CWD set to
/// the crate root, so we walk two levels up to reach the workspace
/// root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/av1/fixtures").join(name)
}

/// Iterate every elementary AV1 frame inside an IVF byte slice. The
/// returned `Vec<Vec<u8>>` is in stream order; each entry is the raw
/// OBU bytes for one IVF "frame" (which can contain multiple OBUs:
/// TEMPORAL_DELIMITER, SEQUENCE_HEADER, FRAME, TILE_GROUP, …).
fn ivf_frames(ivf: &[u8]) -> Vec<Vec<u8>> {
    assert!(ivf.len() >= 32, "IVF too short");
    assert_eq!(&ivf[0..IVF_HEADER_SIGNATURE_LEN], b"DKIF", "not IVF");
    let header_len = u16::from_le_bytes([ivf[6], ivf[7]]) as usize;
    let mut off = header_len;
    let mut out = Vec::new();
    while off + IVF_FRAME_HEADER_LEN <= ivf.len() {
        let size =
            u32::from_le_bytes([ivf[off], ivf[off + 1], ivf[off + 2], ivf[off + 3]]) as usize;
        off += IVF_FRAME_HEADER_LEN;
        if off + size > ivf.len() {
            break;
        }
        out.push(ivf[off..off + size].to_vec());
        off += size;
    }
    out
}

#[derive(Clone, Copy, Debug)]
enum PixFmt {
    /// 4:2:0, 8-bit. Luma full-res, chroma half-w * half-h. 8-bit per
    /// sample.
    Yuv420P8,
    /// 4:4:4, 8-bit. All planes full-res.
    Yuv444P8,
    /// 4:2:2, 10-bit little-endian. Chroma half-w * full-h, 16-bit
    /// containers carrying 10-bit values (top 6 bits = 0).
    Yuv422P10Le,
    /// 4:2:2, 12-bit little-endian.
    Yuv422P12Le,
    /// Monochrome 8-bit (luma only).
    Gray8,
}

impl PixFmt {
    /// Bit depth of the source samples.
    fn bit_depth(&self) -> u32 {
        match self {
            PixFmt::Yuv420P8 | PixFmt::Yuv444P8 | PixFmt::Gray8 => 8,
            PixFmt::Yuv422P10Le => 10,
            PixFmt::Yuv422P12Le => 12,
        }
    }

    /// Bytes per source sample in the reference `expected.yuv`. 1 for
    /// 8-bit, 2 for 10/12-bit (little-endian u16 containers).
    fn ref_bytes_per_sample(&self) -> usize {
        if self.bit_depth() == 8 {
            1
        } else {
            2
        }
    }

    /// Per-frame size in bytes of the reference YUV at the given
    /// luma dimensions.
    fn frame_bytes(&self, width: usize, height: usize) -> usize {
        let bps = self.ref_bytes_per_sample();
        let y = width * height;
        match self {
            PixFmt::Yuv420P8 => (y + 2 * (width.div_ceil(2) * height.div_ceil(2))) * bps,
            PixFmt::Yuv444P8 => (y + 2 * y) * bps,
            PixFmt::Yuv422P10Le | PixFmt::Yuv422P12Le => {
                (y + 2 * (width.div_ceil(2) * height)) * bps
            }
            PixFmt::Gray8 => y * bps,
        }
    }

    /// Number of planes in the reference layout (3 for YUV, 1 for
    /// monochrome).
    fn planes(&self) -> usize {
        match self {
            PixFmt::Gray8 => 1,
            _ => 3,
        }
    }

    /// Per-plane (width, height) for plane index `p` in samples (NOT
    /// bytes — multiply by `ref_bytes_per_sample()` to get the byte
    /// span in the reference buffer).
    fn plane_dims(&self, width: usize, height: usize, p: usize) -> (usize, usize) {
        match (self, p) {
            (PixFmt::Yuv420P8, 0) | (PixFmt::Yuv444P8, 0) | (PixFmt::Gray8, 0) => (width, height),
            (PixFmt::Yuv420P8, _) => (width.div_ceil(2), height.div_ceil(2)),
            (PixFmt::Yuv444P8, _) => (width, height),
            (PixFmt::Yuv422P10Le, 0) | (PixFmt::Yuv422P12Le, 0) => (width, height),
            (PixFmt::Yuv422P10Le, _) | (PixFmt::Yuv422P12Le, _) => (width.div_ceil(2), height),
            _ => (0, 0),
        }
    }
}

/// Per-frame decode result against the per-frame slice of
/// `expected.yuv`. All counters are in samples (post-narrowing for
/// HBD), not bytes.
#[derive(Default)]
struct FrameDiff {
    y_total: usize,
    y_exact: usize,
    y_max: i32,
    uv_total: usize,
    uv_exact: usize,
    uv_max: i32,
}

impl FrameDiff {
    fn pct(&self) -> f64 {
        let exact = self.y_exact + self.uv_exact;
        let total = self.y_total + self.uv_total;
        if total == 0 {
            0.0
        } else {
            exact as f64 / total as f64 * 100.0
        }
    }

    fn merge(&mut self, other: &FrameDiff) {
        self.y_total += other.y_total;
        self.y_exact += other.y_exact;
        self.y_max = self.y_max.max(other.y_max);
        self.uv_total += other.uv_total;
        self.uv_exact += other.uv_exact;
        self.uv_max = self.uv_max.max(other.uv_max);
    }
}

/// Compare a single plane of our (u8) output against the reference.
/// For 8-bit reference data we compare byte-for-byte. For HBD
/// reference data we read u16-LE, narrow to u8 by `>> shift`, and
/// then compare — this matches what `Av1Decoder::enqueue_video_frame`
/// does to its internal u16 buffers (see decoder.rs §enqueue, the
/// `narrow` helper). Match-pct is therefore measured in the SAME
/// 8-bit space; HBD round-tripping is always lossy by 2-4 bits and
/// would be invisible if we compared in u16 space.
fn diff_plane(our: &[u8], refp: &[u8], bit_depth: u32) -> (usize, usize, i32) {
    let mut ex = 0usize;
    let mut max = 0i32;
    if bit_depth == 8 {
        let n = our.len().min(refp.len());
        for i in 0..n {
            let d = (our[i] as i32 - refp[i] as i32).abs();
            if d == 0 {
                ex += 1;
            }
            if d > max {
                max = d;
            }
        }
        (n, ex, max)
    } else {
        let shift = bit_depth - 8;
        let n_samples = (refp.len() / 2).min(our.len());
        for i in 0..n_samples {
            let lo = refp[i * 2];
            let hi = refp[i * 2 + 1];
            let r16 = u16::from_le_bytes([lo, hi]);
            let r8 = (r16 >> shift).min(255) as i32;
            let o8 = our[i] as i32;
            let d = (o8 - r8).abs();
            if d == 0 {
                ex += 1;
            }
            if d > max {
                max = d;
            }
        }
        (n_samples, ex, max)
    }
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // BitExact unused at first commit; promote fixtures over time
enum Tier {
    /// Must decode bit-exactly. Test fails on any divergence.
    BitExact,
    /// Decode is permitted to diverge from the reference; the
    /// per-fixture stats are logged but the test does not fail.
    /// Promote to `BitExact` once the underlying gap is closed.
    ReportOnly,
}

struct CorpusCase {
    name: &'static str,
    width: usize,
    height: usize,
    n_frames: usize,
    pix_fmt: PixFmt,
    tier: Tier,
}

/// Read fixture files + decode + score against expected.yuv. Returns
/// `None` if the fixture files are missing (handy for CI-without-docs
/// setups: the av1 crate ships its own checkout, while
/// `docs/video/av1/fixtures/` lives in the workspace umbrella repo).
/// Otherwise returns the per-frame decode outcomes (one entry per
/// *requested* expected frame), plus an indicator of the total number
/// of visible frames the decoder actually produced.
struct DecodeReport {
    per_frame: Vec<Result<FrameDiff, String>>,
    visible_produced: usize,
    /// Top-level error (if `send_packet` or `receive_frame` returned
    /// something non-`NeedMore`). Recorded for the report; does NOT
    /// stop subsequent packets.
    fatal: Option<String>,
}

fn decode_fixture(case: &CorpusCase) -> Option<DecodeReport> {
    let dir = fixture_dir(case.name);
    let ivf_path = dir.join("input.ivf");
    let yuv_path = dir.join("expected.yuv");
    let trace_path = dir.join("trace.txt");
    let ivf = match fs::read(&ivf_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip {}: missing {} ({e}). docs/ corpus is in the workspace \
                 umbrella repo; the standalone crate checkout has no fixtures.",
                case.name,
                ivf_path.display()
            );
            return None;
        }
    };
    let yuv_ref = match fs::read(&yuv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, yuv_path.display());
            return None;
        }
    };
    eprintln!(
        "fixture {}: ivf={} bytes, expected.yuv={} bytes, trace={}",
        case.name,
        ivf.len(),
        yuv_ref.len(),
        trace_path.display()
    );

    let frames = ivf_frames(&ivf);
    assert!(
        !frames.is_empty(),
        "fixture {} has no IVF frames",
        case.name
    );

    let frame_size = case.pix_fmt.frame_bytes(case.width, case.height);
    assert_eq!(
        yuv_ref.len(),
        case.n_frames * frame_size,
        "fixture {} expected.yuv size mismatch (have {} bytes, expected {} = {} frames * {})",
        case.name,
        yuv_ref.len(),
        case.n_frames * frame_size,
        case.n_frames,
        frame_size
    );

    let params = CodecParameters::video(CodecId::new(oxideav_av1::CODEC_ID_STR));
    let mut dec = Av1Decoder::new(params);
    let mut visible_idx = 0usize;
    let mut per_frame: Vec<Result<FrameDiff, String>> = Vec::with_capacity(case.n_frames);
    let mut fatal: Option<String> = None;

    for (pkt_idx, frame_bytes) in frames.iter().enumerate() {
        let mut pkt = Packet::new(0, TimeBase::new(1, 1000), frame_bytes.clone());
        pkt.pts = Some(pkt_idx as i64);
        if let Err(e) = dec.send_packet(&pkt) {
            let msg = format!("packet {pkt_idx}: send_packet: {e:?}");
            // Record as a per-frame error AND keep the first one as
            // the fatal banner — but keep iterating, so that we get
            // the full picture (some fixtures may have a recoverable
            // `Unsupported` early in the stream).
            per_frame.push(Err(msg.clone()));
            if fatal.is_none() {
                fatal = Some(msg);
            }
            continue;
        }
        // Drain any visible frames produced by this packet.
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    if visible_idx >= case.n_frames {
                        // Decoder produced more visible frames than the
                        // reference. Record but do not compare.
                        visible_idx += 1;
                        continue;
                    }
                    let ref_off = visible_idx * frame_size;
                    let ref_slice = &yuv_ref[ref_off..ref_off + frame_size];
                    // Slice the reference into planes by walking
                    // plane_dims() in order.
                    let mut ref_off_within = 0usize;
                    let bps = case.pix_fmt.ref_bytes_per_sample();
                    let mut diff = FrameDiff::default();
                    let mut size_mismatch: Option<String> = None;
                    for p in 0..case.pix_fmt.planes() {
                        let (pw, ph) = case.pix_fmt.plane_dims(case.width, case.height, p);
                        let plane_bytes = pw * ph * bps;
                        let ref_plane = &ref_slice[ref_off_within..ref_off_within + plane_bytes];
                        ref_off_within += plane_bytes;
                        let our_plane = match vf.planes.get(p) {
                            Some(pl) => pl.data.as_slice(),
                            None => {
                                size_mismatch = Some(format!(
                                    "visible {visible_idx}: decoder produced {} planes, \
                                     reference expects {}",
                                    vf.planes.len(),
                                    case.pix_fmt.planes()
                                ));
                                break;
                            }
                        };
                        let expected_our_len = pw * ph;
                        if our_plane.len() != expected_our_len {
                            size_mismatch = Some(format!(
                                "visible {visible_idx} plane {p}: our len {}, \
                                 expected {} (= {pw}x{ph} u8 samples; ref bytes = {plane_bytes})",
                                our_plane.len(),
                                expected_our_len
                            ));
                            break;
                        }
                        let (n, ex, mx) =
                            diff_plane(our_plane, ref_plane, case.pix_fmt.bit_depth());
                        if p == 0 {
                            diff.y_total += n;
                            diff.y_exact += ex;
                            diff.y_max = diff.y_max.max(mx);
                        } else {
                            diff.uv_total += n;
                            diff.uv_exact += ex;
                            diff.uv_max = diff.uv_max.max(mx);
                        }
                    }
                    if let Some(msg) = size_mismatch {
                        per_frame.push(Err(msg));
                    } else {
                        per_frame.push(Ok(diff));
                    }
                    visible_idx += 1;
                }
                Ok(_) => continue,
                Err(Error::NeedMore) => break,
                Err(e) => {
                    let msg = format!("visible {visible_idx}: receive_frame: {e:?}");
                    per_frame.push(Err(msg.clone()));
                    if fatal.is_none() {
                        fatal = Some(msg);
                    }
                    break;
                }
            }
        }
    }

    Some(DecodeReport {
        per_frame,
        visible_produced: visible_idx,
        fatal,
    })
}

/// Pretty-print + tier-aware assertion.
fn evaluate(case: &CorpusCase) {
    let report = match decode_fixture(case) {
        Some(r) => r,
        None => return, // missing fixture — already logged
    };

    let mut agg = FrameDiff::default();
    let mut errors: Vec<String> = Vec::new();
    for (i, r) in report.per_frame.iter().enumerate() {
        match r {
            Ok(d) => {
                eprintln!(
                    "  frame {i}: Y {}/{} exact (max diff {}), UV {}/{} exact (max diff {}), pct={:.2}%",
                    d.y_exact, d.y_total, d.y_max,
                    d.uv_exact, d.uv_total, d.uv_max,
                    d.pct()
                );
                agg.merge(d);
            }
            Err(e) => {
                eprintln!("  frame {i}: ERROR {e}");
                errors.push(format!("frame {i}: {e}"));
            }
        }
    }

    let pct = agg.pct();
    eprintln!(
        "[{:?}] {}: aggregate {}/{} exact ({pct:.2}%), Y max diff {}, UV max diff {}, \
         visible_produced={}/{}{}",
        case.tier,
        case.name,
        agg.y_exact + agg.uv_exact,
        agg.y_total + agg.uv_total,
        agg.y_max,
        agg.uv_max,
        report.visible_produced,
        case.n_frames,
        match &report.fatal {
            Some(f) => format!(", first_fatal=\"{f}\""),
            None => String::new(),
        }
    );

    match case.tier {
        Tier::BitExact => {
            assert!(
                errors.is_empty(),
                "{}: {} frame errors prevented bit-exact comparison: {:?}",
                case.name,
                errors.len(),
                errors
            );
            assert_eq!(
                agg.y_exact + agg.uv_exact,
                agg.y_total + agg.uv_total,
                "{}: not bit-exact (Y max diff {}, UV max diff {}; {:.4}% match)",
                case.name,
                agg.y_max,
                agg.uv_max,
                pct
            );
        }
        Tier::ReportOnly => {
            // Don't fail. The eprintln output above is the report.
            // TODO(av1-corpus): tighten to BitExact once the
            // underlying decoder gap for this fixture is closed.
            let _ = pct;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// All fixtures start as ReportOnly. As the in-tree AV1 decoder closes
// the relevant gap, individual cases should be promoted to BitExact.
//
// Trace files (referenced in `evaluate()` via the eprintln! header)
// live alongside each fixture and capture the OBU + frame-header
// event sequence emitted by an instrumented FFmpeg cbs_av1 frontend
// on the bitstream — useful for diffing against our own decoder
// trace if/when divergence localisation is needed.

/// Smallest possible AV1 bitstream: one 16x16 keyframe in profile 0.
/// One TEMPORAL_DELIMITER + one SEQUENCE_HEADER + one FRAME OBU.
/// Trace: docs/video/av1/fixtures/tiny-i-only-16x16-prof0/trace.txt
#[test]
fn corpus_tiny_i_only_16x16_prof0() {
    evaluate(&CorpusCase {
        name: "tiny-i-only-16x16-prof0",
        width: 16,
        height: 16,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// 64x64 keyframe over `testsrc`. Single 64x64 superblock with a
/// non-trivial partition tree.
/// Trace: docs/video/av1/fixtures/i-only-64x64-prof0/trace.txt
#[test]
fn corpus_i_only_64x64_prof0() {
    evaluate(&CorpusCase {
        name: "i-only-64x64-prof0",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// One KEY frame followed by one INTER. Single-reference inter is
/// the Phase 7 wired path; this is the canonical smoke test.
/// Trace: docs/video/av1/fixtures/i-frame-then-p-64x64/trace.txt
#[test]
fn corpus_i_frame_then_p_64x64() {
    evaluate(&CorpusCase {
        name: "i-frame-then-p-64x64",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Profile 0 (Main) baseline: 4:2:0, 8-bit over a `gradients`
/// source. Default decoder configuration.
/// Trace: docs/video/av1/fixtures/profile-0-yuv420-8bit/trace.txt
#[test]
fn corpus_profile_0_yuv420_8bit() {
    evaluate(&CorpusCase {
        name: "profile-0-yuv420-8bit",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Profile 1 (High): 4:4:4, 8-bit. Confirms full-resolution chroma
/// path (no subsampling).
/// Trace: docs/video/av1/fixtures/profile-1-yuv444-8bit/trace.txt
#[test]
fn corpus_profile_1_yuv444_8bit() {
    evaluate(&CorpusCase {
        name: "profile-1-yuv444-8bit",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv444P8,
        tier: Tier::ReportOnly,
    });
}

/// Profile 2 (Professional): 4:2:2, 10-bit. The decoder narrows
/// HBD to u8 when emitting VideoFrames (see Av1Decoder
/// `enqueue_video_frame`'s `narrow` helper) — the comparison is
/// therefore done in 8-bit space (we right-shift the 10-bit
/// reference by 2 bits).
/// Trace: docs/video/av1/fixtures/profile-2-yuv422-10bit/trace.txt
#[test]
fn corpus_profile_2_yuv422_10bit() {
    evaluate(&CorpusCase {
        name: "profile-2-yuv422-10bit",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv422P10Le,
        tier: Tier::ReportOnly,
    });
}

/// Profile 2: 4:2:2, 12-bit (the highest bit-depth path). Same HBD
/// caveat as the 10-bit case.
/// Trace: docs/video/av1/fixtures/profile-2-yuv422-12bit/trace.txt
#[test]
fn corpus_profile_2_yuv422_12bit() {
    evaluate(&CorpusCase {
        name: "profile-2-yuv422-12bit",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv422P12Le,
        tier: Tier::ReportOnly,
    });
}

/// Monochrome (no chroma planes). Single-plane decoder path.
/// Trace: docs/video/av1/fixtures/monochrome-grey-only/trace.txt
#[test]
fn corpus_monochrome_grey_only() {
    evaluate(&CorpusCase {
        name: "monochrome-grey-only",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Gray8,
        tier: Tier::ReportOnly,
    });
}

/// `use_128x128_superblock=1` in the sequence header — partition
/// tree under a 128x128 SB.
/// Trace: docs/video/av1/fixtures/superblocks-128/trace.txt
#[test]
fn corpus_superblocks_128() {
    evaluate(&CorpusCase {
        name: "superblocks-128",
        width: 128,
        height: 128,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Multi-tile: tile_cols=2, tile_rows=1. Tile-info parsing +
/// per-tile header iteration.
/// Trace: docs/video/av1/fixtures/tile-cols-2-rows-1/trace.txt
#[test]
fn corpus_tile_cols_2_rows_1() {
    evaluate(&CorpusCase {
        name: "tile-cols-2-rows-1",
        width: 256,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// `av1_metadata=td=insert` ensures TEMPORAL_DELIMITER OBUs between
/// every TU. The "negative" extension-header case: obu_extension_flag
/// is absent throughout but the OBU header still has to be parsed
/// correctly.
/// Trace: docs/video/av1/fixtures/obu-with-extension-headers/trace.txt
#[test]
fn corpus_obu_with_extension_headers() {
    evaluate(&CorpusCase {
        name: "obu-with-extension-headers",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Film-grain synthesis: `apply_grain=1` with aomenc's canonical
/// FGS test pattern (`--film-grain-test=1`). Two frames so the AR
/// scrambler state has somewhere to evolve.
/// Trace: docs/video/av1/fixtures/film-grain-on/trace.txt
#[test]
fn corpus_film_grain_on() {
    evaluate(&CorpusCase {
        name: "film-grain-on",
        width: 64,
        height: 64,
        n_frames: 2,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Frame super-resolution (`use_superres=1`, coded denom=3 →
/// upscale ratio 8/12). The bitstream codes only the downscaled
/// version; the decoder must run the §7.16 super-res process to
/// reach the 128x64 output.
/// Trace: docs/video/av1/fixtures/super-resolution/trace.txt
#[test]
fn corpus_super_resolution() {
    evaluate(&CorpusCase {
        name: "super-resolution",
        width: 128,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// `allow_screen_content_tools=1` over a blocky drawbox-grid
/// content. IntraBC may or may not actually fire (depends on
/// libaom's heuristics) but the screen-content frame-header bit
/// is set.
/// Trace: docs/video/av1/fixtures/screen-content-tools/trace.txt
#[test]
fn corpus_screen_content_tools() {
    evaluate(&CorpusCase {
        name: "screen-content-tools",
        width: 256,
        height: 128,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// ARF group with `show_existing_frame=1` repeats. Exercises the
/// §6.8.2 "show existing frame" path: reference-buffer-only output
/// without a fresh tile decode.
/// Trace: docs/video/av1/fixtures/show-existing-frame/trace.txt.gz
#[test]
fn corpus_show_existing_frame() {
    evaluate(&CorpusCase {
        name: "show-existing-frame",
        width: 64,
        height: 64,
        n_frames: 20,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}

/// Coded lossless: base_q_idx=0, identity transform path. Per
/// §6.8.2 + §7.6 CDEF and Loop Restoration are disabled when
/// CodedLossless is true.
/// Trace: docs/video/av1/fixtures/lossless-i-only/trace.txt
#[test]
fn corpus_lossless_i_only() {
    evaluate(&CorpusCase {
        name: "lossless-i-only",
        width: 64,
        height: 64,
        n_frames: 1,
        pix_fmt: PixFmt::Yuv420P8,
        tier: Tier::ReportOnly,
    });
}
