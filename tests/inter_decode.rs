//! Integration test for Phase 7's single-reference translational
//! inter path. Decodes a 128×128 multi-frame AVIS fixture produced
//! by aomenc with `--kf-min-dist=100 --error-resilient=1` so every
//! non-key frame is a conventional inter frame using Identity global
//! motion (which keeps our bitstream parser in sync without DPB
//! state).
//!
//! Fixture path: `/tmp/av1-inter.ivf`. Tests that cannot find the
//! fixture are skipped (logged, not failed) so CI without ffmpeg /
//! aomenc still passes.
//!
//! Recreate with:
//! ```sh
//! ffmpeg -y -f lavfi -i "testsrc=size=128x128:rate=24:duration=0.5" \
//!     -f rawvideo -pix_fmt yuv420p /tmp/av1in-128.yuv
//! aomenc --ivf -w 128 -h 128 --fps=24/1 --cpu-used=8 --cq-level=50 \
//!     --tile-columns=0 --tile-rows=0 \
//!     --kf-min-dist=100 --kf-max-dist=100 \
//!     --lag-in-frames=0 --passes=1 --end-usage=q \
//!     --enable-cdef=0 --enable-restoration=0 --enable-qm=0 \
//!     --enable-fwd-kf=0 --auto-alt-ref=0 \
//!     --enable-global-motion=0 --enable-warped-motion=0 \
//!     --error-resilient=1 --deltaq-mode=0 --loopfilter-control=0 \
//!     --limit=2 \
//!     -o /tmp/av1-inter.ivf /tmp/av1in-128.yuv
//! ```

use std::path::Path;

use oxideav_av1::frame_header::FrameType;
use oxideav_av1::{iter_obus, parse_frame_header, parse_sequence_header, ObuType};
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

fn read_fixture(path: &str) -> Option<Vec<u8>> {
    if !Path::new(path).exists() {
        eprintln!("fixture {path} missing — skipping test");
        return None;
    }
    Some(std::fs::read(path).expect("read fixture"))
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

/// Sanity: the fixture encodes exactly two frames and the second is
/// flagged as inter by the frame header.
#[test]
fn fixture_has_one_key_then_one_inter() {
    let Some(data) = read_fixture("/tmp/av1-inter.ivf") else {
        return;
    };
    let pkts = ivf_packet_slices(&data).expect("ivf parse");
    assert_eq!(pkts.len(), 2, "expected exactly 2 frames in fixture");

    let mut seq = None;
    let mut types = Vec::new();
    for (i, pkt) in pkts.iter().enumerate() {
        for o in iter_obus(pkt) {
            let o = o.expect("obu parse");
            match o.header.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(o.payload).expect("seq"));
                }
                ObuType::Frame | ObuType::FrameHeader => {
                    let s = seq.as_ref().expect("seq before frame");
                    match parse_frame_header(s, o.payload) {
                        Ok(fh) => types.push(fh.frame_type),
                        Err(e) => {
                            eprintln!("pkt {i} frame header parse error: {e:?}");
                            // Second-frame parse may bail in non-fatal ways
                            // because our parser doesn't carry DPB state; we
                            // tolerate that for the sanity test and rely on
                            // the full decode test below for the real check.
                        }
                    }
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        types.first(),
        Some(&FrameType::Key),
        "first frame should be Key"
    );
    assert!(
        types.iter().skip(1).any(|t| *t == FrameType::Inter),
        "second frame should be Inter, got {types:?}"
    );
}

struct VideoPlanes {
    width: u32,
    height: u32,
    y: Vec<u8>,
    _u: Option<Vec<u8>>,
    _v: Option<Vec<u8>>,
}

fn collect_video_planes(f: &Frame) -> Option<VideoPlanes> {
    let Frame::Video(v) = f else {
        return None;
    };
    let y = v.planes.first()?.data.clone();
    let _u = v.planes.get(1).map(|p| p.data.clone());
    let _v = v.planes.get(2).map(|p| p.data.clone());
    Some(VideoPlanes {
        width: v.width,
        height: v.height,
        y,
        _u,
        _v,
    })
}

fn luma_mean(y: &[u8]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    let sum: u64 = y.iter().map(|&v| v as u64).sum();
    (sum as f64) / (y.len() as f64)
}

#[test]
fn decoder_produces_both_frames_for_inter_fixture() {
    let Some(data) = read_fixture("/tmp/av1-inter.ivf") else {
        return;
    };
    let pkts = ivf_packet_slices(&data).expect("ivf parse");
    assert_eq!(pkts.len(), 2);

    let params = CodecParameters::video(CodecId::new(oxideav_av1::CODEC_ID_STR));
    let mut dec = oxideav_av1::make_decoder(&params).expect("build decoder");
    let tb = TimeBase::new(1, 24);

    let mut frames = Vec::new();
    // Only feed the first 2 packets — additional frames exceed the
    // narrow Phase 7 scope (multi-ref DPB / complex gm_params).
    for (i, bytes) in pkts.iter().take(2).enumerate() {
        let pkt = Packet::new(0, tb, bytes.clone()).with_pts(i as i64);
        match dec.send_packet(&pkt) {
            Ok(()) => {}
            Err(e) => panic!("send_packet frame {i} failed: {e:?}"),
        }
        while let Ok(f) = dec.receive_frame() {
            frames.push(f);
        }
    }

    assert_eq!(
        frames.len(),
        2,
        "expected 2 decoded frames, got {}",
        frames.len()
    );
    let p0 = collect_video_planes(&frames[0]).expect("frame 0 video");
    let p1 = collect_video_planes(&frames[1]).expect("frame 1 video");
    assert_eq!(p0.width, 128);
    assert_eq!(p0.height, 128);
    assert_eq!(p1.width, 128);
    assert_eq!(p1.height, 128);

    let m0 = luma_mean(&p0.y);
    let m1 = luma_mean(&p1.y);
    eprintln!("frame 0 luma mean = {m0:.2}, frame 1 luma mean = {m1:.2}");
    // testsrc produces a visually rich picture — both means should sit
    // well inside the mid range, not pinned at 0 or 255.
    assert!(
        (10.0..=250.0).contains(&m0),
        "frame 0 luma mean {m0} out of plausible range"
    );
    assert!(
        (10.0..=250.0).contains(&m1),
        "frame 1 luma mean {m1} out of plausible range"
    );

    // The two frames must differ; testsrc animates between them.
    assert_ne!(
        p0.y, p1.y,
        "frame 1 should differ from frame 0 (testsrc animates)"
    );
}
