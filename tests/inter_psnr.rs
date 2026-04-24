//! PSNR-oriented integration test for the inter MC pipeline.
//!
//! Decodes `/tmp/av1-inter.ivf` with our crate and compares the first
//! P-frame (luma + chroma) against `/tmp/av1-inter-ref.yuv`, which is
//! the ground truth produced by `ffmpeg -c:v libaom-av1 -> rawvideo`.
//! The test logs per-plane PSNR; it does not fail below a threshold so
//! the test remains useful while inter-MC accuracy improves.
//!
//! Recreate inputs with:
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
//! ffmpeg -y -i /tmp/av1-inter.ivf -f rawvideo -pix_fmt yuv420p \
//!     /tmp/av1-inter-ref.yuv
//! ```

use std::path::Path;

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

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "psnr length mismatch");
    if a.is_empty() {
        return f64::INFINITY;
    }
    let mut sse: u64 = 0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x as i32) - (y as i32);
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 99.0;
    }
    let mse = (sse as f64) / (a.len() as f64);
    10.0 * (255.0 * 255.0 / mse).log10()
}

fn measure(fixture: &str, ref_yuv: &str, w: usize, h: usize) -> Option<(f64, f64, f64, f64, f64, f64)> {
    let data = read_fixture(fixture)?;
    let raw_ref = read_fixture(ref_yuv)?;
    let pkts = ivf_packet_slices(&data).expect("ivf parse");
    assert!(pkts.len() >= 2);

    let params = CodecParameters::video(CodecId::new(oxideav_av1::CODEC_ID_STR));
    let mut dec = oxideav_av1::make_decoder(&params).expect("build decoder");
    let tb = TimeBase::new(1, 24);
    let mut frames = Vec::new();
    for (i, bytes) in pkts.iter().take(2).enumerate() {
        let pkt = Packet::new(0, tb, bytes.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(f) = dec.receive_frame() {
            frames.push(f);
        }
    }
    assert_eq!(frames.len(), 2);
    let Frame::Video(v0) = &frames[0] else { panic!() };
    let Frame::Video(v1) = &frames[1] else { panic!() };
    assert_eq!(v1.width as usize, w);
    assert_eq!(v1.height as usize, h);
    let uvw = w / 2;
    let uvh = h / 2;
    let y_len = w * h;
    let uv_len = uvw * uvh;
    let frame_size = y_len + 2 * uv_len;
    let ref0_y = &raw_ref[0..y_len];
    let ref0_u = &raw_ref[y_len..y_len + uv_len];
    let ref0_v = &raw_ref[y_len + uv_len..frame_size];
    let ref1_y = &raw_ref[frame_size..frame_size + y_len];
    let ref1_u = &raw_ref[frame_size + y_len..frame_size + y_len + uv_len];
    let ref1_v = &raw_ref[frame_size + y_len + uv_len..frame_size + frame_size];

    let kf_y = psnr(&v0.planes[0].data, ref0_y);
    let kf_u = psnr(&v0.planes[1].data, ref0_u);
    let kf_v = psnr(&v0.planes[2].data, ref0_v);
    let pf_y = psnr(&v1.planes[0].data, ref1_y);
    let pf_u = psnr(&v1.planes[1].data, ref1_u);
    let pf_v = psnr(&v1.planes[2].data, ref1_v);
    Some((kf_y, kf_u, kf_v, pf_y, pf_u, pf_v))
}

#[test]
fn inter_first_pframe_psnr_vs_libaom() {
    let Some(data) = read_fixture("/tmp/av1-inter.ivf") else {
        return;
    };
    let Some(raw_ref) = read_fixture("/tmp/av1-inter-ref.yuv") else {
        return;
    };
    let pkts = ivf_packet_slices(&data).expect("ivf parse");
    assert!(pkts.len() >= 2);

    let params = CodecParameters::video(CodecId::new(oxideav_av1::CODEC_ID_STR));
    let mut dec = oxideav_av1::make_decoder(&params).expect("build decoder");
    let tb = TimeBase::new(1, 24);

    let mut frames = Vec::new();
    for (i, bytes) in pkts.iter().take(2).enumerate() {
        let pkt = Packet::new(0, tb, bytes.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        while let Ok(f) = dec.receive_frame() {
            frames.push(f);
        }
    }
    assert_eq!(frames.len(), 2, "expected 2 decoded frames");

    // Extract P-frame planes (frame index 1).
    let Frame::Video(v1) = &frames[1] else {
        panic!("frame 1 not video");
    };
    let w = v1.width as usize;
    let h = v1.height as usize;
    let uvw = w / 2;
    let uvh = h / 2;
    let y_len = w * h;
    let uv_len = uvw * uvh;

    // Ground-truth frame 1 starts at offset = y_len + 2*uv_len.
    let frame_size = y_len + 2 * uv_len;
    let ref_start = frame_size;
    let ref_y = &raw_ref[ref_start..ref_start + y_len];
    let ref_u = &raw_ref[ref_start + y_len..ref_start + y_len + uv_len];
    let ref_v = &raw_ref[ref_start + y_len + uv_len..ref_start + frame_size];

    let our_y = &v1.planes[0].data;
    let our_u = &v1.planes[1].data;
    let our_v = &v1.planes[2].data;

    let psnr_y = psnr(our_y, ref_y);
    let psnr_u = psnr(our_u, ref_u);
    let psnr_v = psnr(our_v, ref_v);
    eprintln!("inter P-frame PSNR: Y={psnr_y:.2} dB  U={psnr_u:.2} dB  V={psnr_v:.2} dB");

    // Also report key-frame PSNR for context.
    let Frame::Video(v0) = &frames[0] else {
        panic!("frame 0 not video");
    };
    let ref0_y = &raw_ref[0..y_len];
    let ref0_u = &raw_ref[y_len..y_len + uv_len];
    let ref0_v = &raw_ref[y_len + uv_len..frame_size];
    let kf_y = psnr(&v0.planes[0].data, ref0_y);
    let kf_u = psnr(&v0.planes[1].data, ref0_u);
    let kf_v = psnr(&v0.planes[2].data, ref0_v);
    eprintln!("intra K-frame PSNR: Y={kf_y:.2} dB  U={kf_u:.2} dB  V={kf_v:.2} dB");
}

#[test]
fn inter_psnr_gray_clip() {
    let Some((kfy, kfu, kfv, pfy, pfu, pfv)) =
        measure("/tmp/av1-gray.ivf", "/tmp/av1-gray-ref.yuv", 64, 64)
    else {
        return;
    };
    eprintln!("gray K-frame PSNR: Y={kfy:.2} U={kfu:.2} V={kfv:.2}");
    eprintln!("gray P-frame PSNR: Y={pfy:.2} U={pfu:.2} V={pfv:.2}");
}

#[test]
fn inter_psnr_testsrc_64() {
    let Some((kfy, kfu, kfv, pfy, pfu, pfv)) =
        measure("/tmp/av1-inter64.ivf", "/tmp/av1-inter64-ref.yuv", 64, 64)
    else {
        return;
    };
    eprintln!("testsrc64 K-frame PSNR: Y={kfy:.2} U={kfu:.2} V={kfv:.2}");
    eprintln!("testsrc64 P-frame PSNR: Y={pfy:.2} U={pfu:.2} V={pfv:.2}");
}
