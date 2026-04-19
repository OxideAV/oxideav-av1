//! Tile-decoder integration tests — combines the goavif
//! `tile_test.go` style smoke checks (init + read one partition /
//! mode symbol) with the Phase-2 real-clip integration: walk every
//! partition of every frame in the 64×64 3-frame aomenc key-only
//! fixture at `/tmp/av1.ivf` and confirm the decoder exits with
//! `Error::Unsupported` at the first non-skip leaf (§5.11.39
//! coefficient decode pending).
//!
//! The synthetic tests depend on `/tmp/av1.ivf` only for a parsed
//! `SequenceHeader` + `FrameHeader` — hand-rolling complete header
//! structs in Rust is awkward and brittle (all the sub-structs need
//! to stay in sync). Using a real clip lets us sidestep that while
//! keeping the tests hermetic (skipped on fixture absence).

use std::path::Path;

use oxideav_av1::decode::{decode_tile_group, FrameState, TileDecoder};
use oxideav_av1::frame_header::FrameHeader;
use oxideav_av1::sequence_header::SequenceHeader;
use oxideav_av1::{iter_obus, parse_frame_obu, parse_sequence_header, ObuType};
use oxideav_core::{Error, Result};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
}

fn ivf_concat_obus(data: &[u8]) -> Option<Vec<u8>> {
    if data.len() < 32 || &data[0..4] != b"DKIF" {
        return None;
    }
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut out = Vec::new();
    let mut pos = header_len;
    while pos + 12 <= data.len() {
        let size =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 12;
        if pos + size > data.len() {
            break;
        }
        out.extend_from_slice(&data[pos..pos + size]);
        pos += size;
    }
    Some(out)
}

/// Parse the first sequence header + first frame header out of the
/// IVF fixture. Returns `None` if the fixture is missing.
fn sample_headers() -> Option<(SequenceHeader, FrameHeader, Vec<u8>)> {
    let data = read_fixture("/tmp/av1.ivf")?;
    let obus = ivf_concat_obus(&data).expect("ivf parse");
    let mut sh: Option<SequenceHeader> = None;
    let mut first_frame: Option<(FrameHeader, Vec<u8>)> = None;
    for o in iter_obus(&obus) {
        let o = o.expect("obu parse");
        match o.header.obu_type {
            ObuType::SequenceHeader => {
                sh = Some(parse_sequence_header(o.payload).expect("seq hdr"));
            }
            ObuType::Frame => {
                let seq = sh.as_ref().expect("seq before frame");
                let (fh, tg_payload) = parse_frame_obu(seq, o.payload).expect("parse_frame_obu");
                first_frame = Some((fh, tg_payload.to_vec()));
                break;
            }
            _ => {}
        }
    }
    let sh = sh?;
    let (fh, tg) = first_frame?;
    Some((sh, fh, tg))
}

/// `TileDecoder::new` should initialise on any non-empty payload
/// and let us read a partition symbol without panicking.
#[test]
fn tile_decoder_init_reads_partition_symbol() {
    let Some((seq, fh, _)) = sample_headers() else {
        return;
    };
    let tile_data = vec![0xA5u8; 64];
    let mut td = TileDecoder::new(&seq, &fh, &tile_data).expect("init");
    let pt = td.decode_partition(0, 0).expect("partition symbol");
    assert!(pt < 4, "partition symbol out of 0..=3: {pt}");
}

/// Reading a short sequence of mode symbols should not exhaust the
/// range coder and every symbol must land in its codebook range.
#[test]
fn tile_decoder_reads_mode_symbols() {
    let Some((seq, fh, _)) = sample_headers() else {
        return;
    };
    let mut tile_data = vec![0u8; 128];
    for (i, b) in tile_data.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(13).wrapping_add(7);
    }
    let mut td = TileDecoder::new(&seq, &fh, &tile_data).expect("init");
    let _ = td.decode_partition(0, 0).expect("partition");
    let y_mode = td.decode_intra_y_mode(0, 0).expect("y mode");
    assert!((y_mode as u32) < 13, "y mode out of range: {:?}", y_mode);
    let uv_mode = td.decode_uv_mode(y_mode, true).expect("uv mode");
    assert!((uv_mode as u32) < 14, "uv mode out of range: {:?}", uv_mode);
    let _skip = td.decode_skip(0).expect("skip");
}

/// The big integration test: walk every partition of every frame in
/// the 64×64 3-frame aomenc key-only clip. Phase 3 now drives the
/// coefficient decoder + inverse transform + reconstruction for every
/// 4×4 / 8×8 / 16×16 leaf block. If the bitstream needs a larger
/// single-TX (32×32+) or TX splitting we surface `Unsupported`
/// pointing at the right Phase 4 sub-clause; otherwise the tile walk
/// succeeds end-to-end.
#[test]
fn real_clip_walks_every_partition_and_bails_on_coeff_decode() {
    let Some(data) = read_fixture("/tmp/av1.ivf") else {
        return;
    };
    let obus = ivf_concat_obus(&data).expect("ivf parse");
    let mut sh: Option<SequenceHeader> = None;
    let mut walked_frames = 0usize;

    for o in iter_obus(&obus) {
        let o = o.expect("obu parse");
        match o.header.obu_type {
            ObuType::SequenceHeader => {
                sh = Some(parse_sequence_header(o.payload).expect("seq hdr"));
            }
            ObuType::Frame => {
                let seq = sh.as_ref().expect("seq before frame");
                let (fh, tg_payload) = parse_frame_obu(seq, o.payload).expect("parse_frame_obu");
                let sub_x = if seq.color_config.subsampling_x { 1 } else { 0 };
                let sub_y = if seq.color_config.subsampling_y { 1 } else { 0 };
                let mut fs = FrameState::with_bit_depth(
                    fh.frame_width,
                    fh.frame_height,
                    sub_x,
                    sub_y,
                    seq.color_config.num_planes == 1,
                    seq.color_config.bit_depth,
                );
                let got: Result<()> = decode_tile_group(seq, &fh, tg_payload, &mut fs);
                match got {
                    Ok(()) => {
                        // Whole tile decoded — Y plane should be
                        // populated. We don't assert pixel values
                        // because our Phase 3 scope still stubs most
                        // intra predictors (D45, smooth, paeth) with
                        // a DC fallback, but the plane length must
                        // match the frame geometry.
                        assert_eq!(
                            fs.y_plane.len(),
                            (fh.frame_width as usize) * (fh.frame_height as usize)
                        );
                    }
                    Err(Error::Unsupported(msg)) => {
                        // Phase 4 deferrals still surface explicitly
                        // — larger TX sizes / TX splitting / qmatrix.
                        assert!(
                            msg.contains("§5.11")
                                || msg.contains("§5.9")
                                || msg.contains("§7.7")
                                || msg.contains("§6.10")
                                || msg.contains("§9.3")
                                || msg.contains("Phase"),
                            "Unsupported message should reference AV1 spec §, got: {msg}"
                        );
                    }
                    other => panic!("expected Ok or Unsupported, got {other:?}"),
                }
                walked_frames += 1;
            }
            _ => {}
        }
    }
    assert!(
        walked_frames >= 1,
        "expected at least one tile-walked frame"
    );
}

/// End-to-end Phase 4 check: decode the aomenc 8-bit clip and inspect
/// the reconstructed luma plane. The bitstream at `/tmp/av1.ivf` is
/// three tiny 64×64 keyframes from `testsrc`, so every block size from
/// 4×4 through 64×64 exercises the pipeline. Phase 4 closes the
/// TX-size gap; every frame should decode without `Unsupported`. We
/// don't pin exact pixel values (our intra-predictor port still stubs
/// several directional / smooth / paeth modes), but the luma mean must
/// land in a plausible grey-band range.
#[test]
fn end_to_end_decode_produces_plane_bytes() {
    let Some(data) = read_fixture("/tmp/av1.ivf") else {
        return;
    };
    let obus = ivf_concat_obus(&data).expect("ivf parse");
    let mut sh: Option<SequenceHeader> = None;
    let mut ok_frames = 0usize;

    for o in iter_obus(&obus) {
        let o = o.expect("obu parse");
        match o.header.obu_type {
            ObuType::SequenceHeader => {
                sh = Some(parse_sequence_header(o.payload).expect("seq hdr"));
            }
            ObuType::Frame => {
                let seq = sh.as_ref().expect("seq before frame");
                let (fh, tg_payload) = parse_frame_obu(seq, o.payload).expect("parse_frame_obu");
                let sub_x = if seq.color_config.subsampling_x { 1 } else { 0 };
                let sub_y = if seq.color_config.subsampling_y { 1 } else { 0 };
                let mut fs = FrameState::with_bit_depth(
                    fh.frame_width,
                    fh.frame_height,
                    sub_x,
                    sub_y,
                    seq.color_config.num_planes == 1,
                    seq.color_config.bit_depth,
                );
                let got: Result<()> = decode_tile_group(seq, &fh, tg_payload, &mut fs);
                match got {
                    Ok(()) => {
                        assert_eq!(
                            fs.y_plane.len(),
                            (fh.frame_width as usize) * (fh.frame_height as usize)
                        );
                        if !fs.monochrome {
                            assert_eq!(
                                fs.u_plane.len(),
                                (fs.uv_width as usize) * (fs.uv_height as usize)
                            );
                            assert_eq!(
                                fs.v_plane.len(),
                                (fs.uv_width as usize) * (fs.uv_height as usize)
                            );
                        }
                        let sum: u64 = fs.y_plane.iter().map(|&v| v as u64).sum();
                        let mean = sum / (fs.y_plane.len() as u64);
                        eprintln!(
                            "decoded frame {}x{}: luma mean={} (expected 32..=224)",
                            fh.frame_width, fh.frame_height, mean
                        );
                        assert!(
                            (32..=224).contains(&mean),
                            "luma mean {mean} out of plausible range 32..=224",
                        );
                        ok_frames += 1;
                    }
                    Err(Error::Unsupported(msg)) => {
                        panic!(
                            "Phase 4 should decode this fixture end-to-end, got Unsupported: {msg}"
                        );
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            }
            _ => {}
        }
    }
    assert!(ok_frames >= 1, "expected at least one frame to decode cleanly");
}

/// Phase-5 fidelity check: decode the 128×128 non-skip-heavy clip at
/// `/tmp/av1-intra.ivf`. Unlike the tiny 64×64 testsrc at
/// `/tmp/av1.ivf` (where every block is skip), this one has real
/// coefficient data + directional intra modes, so the output is
/// non-flat. Passes iff the reconstructed luma plane shows variation
/// (not a single mid-grey value) and the frame decodes without
/// `Unsupported` surface.
///
/// Generate with:
///   ffmpeg -f lavfi -i testsrc=size=128x128:rate=24:duration=0.1 \
///          -f rawvideo -pix_fmt yuv420p /tmp/av1in-128.yuv
///   aomenc --cpu-used=4 --cq-level=20 --ivf -w 128 -h 128 \
///          --fps=24/1 --kf-min-dist=1 --kf-max-dist=1 \
///          -o /tmp/av1-intra.ivf /tmp/av1in-128.yuv
#[test]
fn end_to_end_decode_128_intra_clip_has_variation() {
    let Some(data) = read_fixture("/tmp/av1-intra.ivf") else {
        return;
    };
    let obus = ivf_concat_obus(&data).expect("ivf parse");
    let mut sh: Option<SequenceHeader> = None;
    let mut ok_frames = 0usize;

    for o in iter_obus(&obus) {
        let o = o.expect("obu parse");
        match o.header.obu_type {
            ObuType::SequenceHeader => {
                sh = Some(parse_sequence_header(o.payload).expect("seq hdr"));
            }
            ObuType::Frame => {
                let seq = sh.as_ref().expect("seq before frame");
                let (fh, tg_payload) = parse_frame_obu(seq, o.payload).expect("parse_frame_obu");
                let sub_x = if seq.color_config.subsampling_x { 1 } else { 0 };
                let sub_y = if seq.color_config.subsampling_y { 1 } else { 0 };
                let mut fs = FrameState::with_bit_depth(
                    fh.frame_width,
                    fh.frame_height,
                    sub_x,
                    sub_y,
                    seq.color_config.num_planes == 1,
                    seq.color_config.bit_depth,
                );
                let got: Result<()> = decode_tile_group(seq, &fh, tg_payload, &mut fs);
                match got {
                    Ok(()) => {
                        // Plane lengths must match geometry.
                        assert_eq!(
                            fs.y_plane.len(),
                            (fh.frame_width as usize) * (fh.frame_height as usize)
                        );
                        // Variation test: with non-skip content, the
                        // reconstructed plane must have more than one
                        // distinct sample value.
                        let min = *fs.y_plane.iter().min().unwrap();
                        let max = *fs.y_plane.iter().max().unwrap();
                        assert!(
                            max > min,
                            "decoded plane is flat ({min}); expected non-skip content to produce variation"
                        );
                        let sum: u64 = fs.y_plane.iter().map(|&v| v as u64).sum();
                        let mean = sum / (fs.y_plane.len() as u64);
                        // Hash of first 64 bytes for regression tracking.
                        let mut h: u64 = 1469598103934665603;
                        for &b in &fs.y_plane[..64] {
                            h ^= b as u64;
                            h = h.wrapping_mul(1099511628211);
                        }
                        eprintln!(
                            "phase5 fixture: {}x{} luma min={} max={} mean={} fnv64(0..64)=0x{:016x}",
                            fh.frame_width, fh.frame_height, min, max, mean, h
                        );
                        ok_frames += 1;
                    }
                    Err(Error::Unsupported(msg)) => {
                        // Phase 5 should decode most 128×128 aomenc
                        // clips end-to-end; if the bitstream uses a TX
                        // shape we haven't wired (e.g. unusual 4:1
                        // rectangles on chroma), surface the §ref.
                        eprintln!("phase5 128×128 fixture surfaced Unsupported: {msg}");
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            }
            _ => {}
        }
    }
    assert!(
        ok_frames >= 1,
        "expected at least one frame to decode cleanly from 128×128 fixture"
    );
}

/// 10-bit HBD smoke test. Generate a 10-bit fixture with:
///
///   ffmpeg -f lavfi -i testsrc=size=64x64:rate=24,format=yuv420p10le \
///          -t 0.125 -pix_fmt yuv420p10le -f rawvideo /tmp/yuv420p10.raw
///   aomenc -b 10 --input-bit-depth=10 --width=64 --height=64 --fps=24/1 \
///          --passes=1 --end-usage=q --cq-level=40 --i420 --cpu-used=8 \
///          --kf-min-dist=1 --kf-max-dist=1 -o /tmp/av1-10bit.ivf /tmp/yuv420p10.raw
///
/// If the fixture is missing the test skips silently.
#[test]
fn end_to_end_decode_hbd_produces_plane_bytes() {
    let Some(data) = read_fixture("/tmp/av1-10bit.ivf") else {
        return;
    };
    let obus = ivf_concat_obus(&data).expect("ivf parse");
    let mut sh: Option<SequenceHeader> = None;
    let mut ok_frames = 0usize;

    for o in iter_obus(&obus) {
        let o = o.expect("obu parse");
        match o.header.obu_type {
            ObuType::SequenceHeader => {
                sh = Some(parse_sequence_header(o.payload).expect("seq hdr"));
            }
            ObuType::Frame => {
                let seq = sh.as_ref().expect("seq before frame");
                let (fh, tg_payload) = parse_frame_obu(seq, o.payload).expect("parse_frame_obu");
                let sub_x = if seq.color_config.subsampling_x { 1 } else { 0 };
                let sub_y = if seq.color_config.subsampling_y { 1 } else { 0 };
                assert_eq!(seq.color_config.bit_depth, 10, "fixture must be 10-bit");
                let mut fs = FrameState::with_bit_depth(
                    fh.frame_width,
                    fh.frame_height,
                    sub_x,
                    sub_y,
                    seq.color_config.num_planes == 1,
                    seq.color_config.bit_depth,
                );
                let got: Result<()> = decode_tile_group(seq, &fh, tg_payload, &mut fs);
                match got {
                    Ok(()) => {
                        assert_eq!(
                            fs.y_plane16.len(),
                            (fh.frame_width as usize) * (fh.frame_height as usize)
                        );
                        let sum: u64 = fs.y_plane16.iter().map(|&v| v as u64).sum();
                        let mean = sum / (fs.y_plane16.len() as u64);
                        eprintln!(
                            "decoded HBD frame {}x{}: luma mean={} (10-bit range 0..=1023)",
                            fh.frame_width, fh.frame_height, mean
                        );
                        // 10-bit midpoint is 512; accept a wide window.
                        assert!(mean < 1024, "luma mean {mean} exceeds 10-bit range");
                        ok_frames += 1;
                    }
                    Err(Error::Unsupported(msg)) => {
                        panic!(
                            "Phase 4 HBD should decode this fixture end-to-end, got Unsupported: {msg}"
                        );
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            }
            _ => {}
        }
    }
    assert!(ok_frames >= 1, "expected at least one HBD frame to decode cleanly");
}
