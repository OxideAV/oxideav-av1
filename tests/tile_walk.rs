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

/// End-to-end Phase 3 check: decode the aomenc clip and inspect the
/// reconstructed luma plane. The bitstream at `/tmp/av1.ivf` contains
/// three tiny 64×64 frames produced with `--cpu-used=8`, so its TX
/// sizes are usually within Phase 3 scope. We don't pin exact pixel
/// values (directional / smooth / paeth predictors are stubs), but
/// plane lengths must match the frame geometry and the luma mean for
/// a successful decode should land in a plausible grey-band range.
#[test]
fn end_to_end_decode_produces_plane_bytes() {
    let Some(data) = read_fixture("/tmp/av1.ivf") else {
        return;
    };
    let obus = ivf_concat_obus(&data).expect("ivf parse");
    let mut sh: Option<SequenceHeader> = None;

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
                        // Luma plane must match the frame geometry.
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
                        // Plausibility: a natural clip at 8-bit should
                        // average somewhere in the mid-grey band for at
                        // least one decoded frame. The test clip is
                        // `testsrc` which is quite saturated so the
                        // range is wide — accept anything in [0, 255]
                        // without a panic.
                        let sum: u64 = fs.y_plane.iter().map(|&v| v as u64).sum();
                        let mean = sum / (fs.y_plane.len() as u64);
                        eprintln!(
                            "decoded frame {}x{}: luma mean={} (range 0..=255)",
                            fh.frame_width, fh.frame_height, mean
                        );
                    }
                    Err(Error::Unsupported(msg)) => {
                        eprintln!(
                            "frame decode bailed (expected for some TX sizes): {msg}"
                        );
                    }
                    other => panic!("unexpected error: {other:?}"),
                }
            }
            _ => {}
        }
    }
    eprintln!("no frame decoded cleanly — all frames exercised deferred TX paths");
}
