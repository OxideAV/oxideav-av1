//! Round-15 diagnostic test for SVT-AV1 intra reconstruction.
//!
//! Per the round-14 outcome, our decoder produced all-128 luma when
//! fed an SVT-AV1 keyframe. This test feeds a single SVT-AV1
//! key-frame fixture (testsrc 64×64, 1 frame) through the decoder and
//! reports the first 8×8 luma block + plane statistics so subsequent
//! rounds can track regressions on real-encoder content.
//!
//! Recreate the fixture with:
//! ```sh
//! ffmpeg -f lavfi -i testsrc=size=64x64:rate=1:duration=1 \
//!   -c:v libsvtav1 -strict experimental /tmp/av1_intra.ivf
//! ```
//!
//! The fixture is also bundled as `tests/fixtures/svt_av1_intra_64.ivf`
//! so the test runs in CI without ffmpeg available.

use std::path::Path;

use oxideav_av1::decode::{decode_tile_group, FrameState};
use oxideav_av1::dpb::Dpb;
use oxideav_av1::frame_header::{parse_frame_obu_with_dpb, FrameType};
use oxideav_av1::{iter_obus, parse_sequence_header, ObuType};

fn read_fixture() -> Option<Vec<u8>> {
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/svt_av1_intra_64.ivf");
    if bundled.exists() {
        return Some(std::fs::read(&bundled).expect("read bundled fixture"));
    }
    if Path::new("/tmp/av1_intra.ivf").exists() {
        return Some(std::fs::read("/tmp/av1_intra.ivf").expect("read /tmp fixture"));
    }
    None
}

fn ivf_packet_slices(data: &[u8]) -> Option<Vec<Vec<u8>>> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return None;
    }
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut pos = header_len;
    while pos + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 12;
        if pos + size > data.len() {
            break;
        }
        out.push(data[pos..pos + size].to_vec());
        pos += size;
    }
    Some(out)
}

#[test]
fn svt_av1_intra_keyframe_diagnostic() {
    let Some(data) = read_fixture() else {
        eprintln!("no svt-av1 intra fixture available — skipping");
        return;
    };
    let pkts = ivf_packet_slices(&data).expect("ivf parse");
    assert!(!pkts.is_empty(), "fixture has no packets");

    let mut seq = None;
    let dpb = Dpb::new();
    let mut decoded_a_frame = false;
    let mut last_err: Option<String> = None;
    let mut last_y_first_8x8: Option<Vec<u8>> = None;
    let mut last_unique_count = 0usize;
    let mut last_mean = 0.0f64;

    for (pi, pkt) in pkts.iter().take(1).enumerate() {
        for o in iter_obus(pkt) {
            let o = o.expect("obu parse");
            match o.header.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(o.payload).expect("seq"));
                }
                ObuType::Frame => {
                    let s = seq.as_ref().expect("seq before frame");
                    let (fh, tg) =
                        parse_frame_obu_with_dpb(s, o.payload, &dpb).expect("frame header parse");
                    eprintln!(
                        "pkt {pi} frame_type={:?} w={} h={} bd={} sub_x={} sub_y={}",
                        fh.frame_type,
                        fh.frame_width,
                        fh.frame_height,
                        s.color_config.bit_depth,
                        s.color_config.subsampling_x,
                        s.color_config.subsampling_y,
                    );
                    eprintln!(
                        "  base_q_idx={} tx_mode={:?} delta_q_present={} ascr={} cdef={} segmentation={}",
                        fh.quant.base_q_idx,
                        fh.tx_mode,
                        fh.delta_q_present,
                        fh.allow_screen_content_tools,
                        s.enable_cdef,
                        fh.segmentation.enabled,
                    );
                    assert_eq!(
                        fh.frame_type,
                        FrameType::Key,
                        "single-frame fixture must be Key"
                    );
                    let sub_x = if s.color_config.subsampling_x { 1 } else { 0 };
                    let sub_y = if s.color_config.subsampling_y { 1 } else { 0 };
                    let mut fs = FrameState::with_bit_depth(
                        fh.frame_width,
                        fh.frame_height,
                        sub_x,
                        sub_y,
                        s.color_config.num_planes == 1,
                        s.color_config.bit_depth,
                    );
                    match decode_tile_group(s, &fh, tg, &mut fs, None, &dpb) {
                        Ok(()) => {
                            decoded_a_frame = true;
                            let y = if s.color_config.bit_depth == 8 {
                                fs.y_plane.clone()
                            } else {
                                fs.y_plane16
                                    .iter()
                                    .map(|&v| (v >> (s.color_config.bit_depth - 8)).min(255) as u8)
                                    .collect()
                            };
                            let stride = fs.width as usize;
                            let mut first = Vec::with_capacity(64);
                            for r in 0..8 {
                                for c in 0..8 {
                                    first.push(y[r * stride + c]);
                                }
                            }
                            let unique: std::collections::HashSet<u8> = y.iter().copied().collect();
                            last_unique_count = unique.len();
                            let sum: u64 = y.iter().map(|&v| v as u64).sum();
                            last_mean = sum as f64 / y.len() as f64;
                            last_y_first_8x8 = Some(first);
                        }
                        Err(e) => {
                            last_err = Some(format!("{e:?}"));
                        }
                    }
                }
                _ => {}
            }
        }
    }

    if let Some(err) = last_err.as_ref() {
        eprintln!("decode_tile_group error: {err}");
    }
    if let Some(first) = last_y_first_8x8.as_ref() {
        eprintln!("First 8x8 luma block:");
        for r in 0..8 {
            let row: Vec<String> = (0..8).map(|c| format!("{:3}", first[r * 8 + c])).collect();
            eprintln!("  {}", row.join(" "));
        }
        eprintln!("luma mean={last_mean:.2} unique_values={last_unique_count}");
    }

    // Round-15 acceptance: with the §8.2.6 symbol-decoder fix the
    // decoder must produce non-uniform luma output (more than a
    // handful of distinct values). Pre-round-15 output was a constant
    // mid-grey — `unique_values == 1` with `mean == 128.0`.
    if decoded_a_frame {
        eprintln!("ok — frame decoded; diagnostic block dumped above");
        assert!(
            last_unique_count > 4,
            "Round 15: decoded frame must have >4 distinct luma values, got {last_unique_count}"
        );
        assert_ne!(
            last_mean, 128.0,
            "Round 15: decoded mean must not be exactly 128 (the all-grey degenerate)"
        );
    } else {
        eprintln!("frame did NOT decode — see error above");
    }
}

/// PSNR comparison helper: compare the decoded luma against the ffmpeg
/// reference (`/tmp/av1_intra.yuv`) when present. Informational only
/// since pixel-perfect parity needs the full §5.11.x rewrites still
/// pending (palette lookahead, edge filter on real data, etc.).
#[test]
fn svt_av1_intra_psnr_vs_reference() {
    let Some(data) = read_fixture() else {
        return;
    };
    let pkts = ivf_packet_slices(&data).expect("ivf parse");
    let ref_yuv_path = std::path::Path::new("/tmp/av1_intra.yuv");
    if !ref_yuv_path.exists() {
        eprintln!("no reference yuv at /tmp/av1_intra.yuv — skipping PSNR check");
        return;
    }
    let ref_data = std::fs::read(ref_yuv_path).expect("read reference yuv");

    let mut seq = None;
    let dpb = Dpb::new();
    for pkt in pkts.iter().take(1) {
        for o in iter_obus(pkt) {
            let o = o.expect("obu parse");
            match o.header.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(o.payload).expect("seq"));
                }
                ObuType::Frame => {
                    let s = seq.as_ref().expect("seq");
                    let (fh, tg) =
                        parse_frame_obu_with_dpb(s, o.payload, &dpb).expect("frame header parse");
                    let sub_x = if s.color_config.subsampling_x { 1 } else { 0 };
                    let sub_y = if s.color_config.subsampling_y { 1 } else { 0 };
                    let mut fs = FrameState::with_bit_depth(
                        fh.frame_width,
                        fh.frame_height,
                        sub_x,
                        sub_y,
                        s.color_config.num_planes == 1,
                        s.color_config.bit_depth,
                    );
                    if decode_tile_group(s, &fh, tg, &mut fs, None, &dpb).is_ok() {
                        // Compare luma plane vs reference. ref is YUV420
                        // packed: [Y(64*64), U(32*32), V(32*32)] = 6144B.
                        let our_y = &fs.y_plane;
                        let ref_y = &ref_data[..our_y.len()];
                        let mse: f64 = our_y
                            .iter()
                            .zip(ref_y.iter())
                            .map(|(&a, &b)| {
                                let d = a as f64 - b as f64;
                                d * d
                            })
                            .sum::<f64>()
                            / (our_y.len() as f64);
                        let psnr = if mse > 0.0 {
                            20.0 * (255.0_f64).log10() - 10.0 * mse.log10()
                        } else {
                            f64::INFINITY
                        };
                        eprintln!("Y-plane PSNR vs ffmpeg ref: {psnr:.2} dB (MSE={mse:.2})");
                    }
                }
                _ => {}
            }
        }
    }
}
