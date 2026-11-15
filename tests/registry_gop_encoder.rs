//! r489 — the framework encoder's inter mode (`gop = N`, `effort`).
//!
//! With `gop > 1` the registry encoder collects `N` frames, codes them
//! as one KEY + P group through
//! [`oxideav_av1::encoder::encode_gop_yuv_with_options`] and emits one
//! packet per frame (the first of each group a sync sample). The
//! packets must be exactly the direct encoder's temporal units, the
//! stream must decode through the framework decoder to the direct
//! encoder's reconstruction, the `av1C` extradata must describe the
//! sequence header the stream carries, and an inter stream must be
//! smaller than the all-intra coding of the same frames.

use oxideav_av1::codec_config::Av1CodecConfig;
use oxideav_av1::encoder::{
    ChromaFormat, GopEffort, GopOptions, YuvFrame, encode_gop_yuv_with_options,
};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::registry::{make_decoder, make_encoder};
use oxideav_av1::sequence_header::parse_sequence_header;
use oxideav_core::{
    CodecId, CodecParameters, Encoder, ExecutionContext, Frame, Packet, PixelFormat, VideoFrame,
    VideoPlane,
};

fn scene(w: u32, h: u32, bd: u8, fmt: ChromaFormat, t: usize) -> YuvFrame {
    let mut f = YuvFrame::filled(w, h, bd, fmt, 0);
    let (wu, hu) = (w as usize, h as usize);
    let shift = u32::from(bd - 8);
    for i in 0..hu {
        for j in 0..wu {
            let (si, sj) = (i + t, j + 2 * t);
            let v = (40 + (si * 7 + sj * 5) % 61 + ((si / 6) * 13 + (sj / 9) * 29) % 97) % 256;
            f.y[i * wu + j] = (v << shift) as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = (((128 + i * 2 + j + t) % 256) << shift) as u16;
            f.v[i * cw + j] = (((64 + i + j * 2 + t) % 256) << shift) as u16;
        }
    }
    f
}

fn to_video_frame(f: &YuvFrame, pts: Option<i64>) -> VideoFrame {
    let pack = |p: &[u16], w: usize| -> VideoPlane {
        let mut data = Vec::new();
        for &s in p {
            if f.bit_depth > 8 {
                data.extend_from_slice(&s.to_le_bytes());
            } else {
                data.push(s as u8);
            }
        }
        VideoPlane {
            stride: w * if f.bit_depth > 8 { 2 } else { 1 },
            data,
        }
    };
    let mut planes = vec![pack(&f.y, f.width as usize)];
    if f.format != ChromaFormat::Monochrome {
        planes.push(pack(&f.u, f.chroma_width() as usize));
        planes.push(pack(&f.v, f.chroma_width() as usize));
    }
    VideoFrame { pts, planes }
}

fn params(w: u32, h: u32, pf: PixelFormat, opts: &[(&str, &str)]) -> CodecParameters {
    let mut p = CodecParameters::video(CodecId::new("av1"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    for (k, v) in opts {
        p.options.insert(*k, *v);
    }
    p
}

fn drain(enc: &mut Box<dyn Encoder>) -> Vec<Packet> {
    let mut out = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        out.push(p);
    }
    out
}

fn encode_all(p: &CodecParameters, frames: &[YuvFrame], threads: usize) -> Vec<Packet> {
    let mut enc = make_encoder(p).expect("encoder constructs");
    if threads > 1 {
        enc.set_execution_context(&ExecutionContext::with_threads(threads));
    }
    let mut out = Vec::new();
    for f in frames {
        enc.send_frame(&Frame::Video(to_video_frame(f, None)))
            .expect("send_frame");
        out.extend(drain(&mut enc));
    }
    enc.flush().expect("flush");
    out.extend(drain(&mut enc));
    assert!(matches!(
        enc.receive_packet(),
        Err(oxideav_core::Error::Eof)
    ));
    out
}

fn decode_all(packets: &[Packet], extradata: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let mut p = CodecParameters::video(CodecId::new("av1"));
    p.extradata = extradata.to_vec();
    let mut dec = make_decoder(&p).expect("make_decoder");
    let mut out = Vec::new();
    for pkt in packets {
        dec.send_packet(pkt).expect("send_packet");
        while let Ok(Frame::Video(v)) = dec.receive_frame() {
            out.push(v.planes.into_iter().map(|pl| pl.data).collect());
        }
    }
    dec.flush().expect("flush");
    while let Ok(Frame::Video(v)) = dec.receive_frame() {
        out.push(v.planes.into_iter().map(|pl| pl.data).collect());
    }
    out
}

fn plane_bytes(bd: u8, p: &[u16]) -> Vec<u8> {
    if bd == 8 {
        p.iter().map(|&s| s as u8).collect()
    } else {
        p.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }
}

#[test]
fn gop_mode_emits_the_direct_encoders_groups() {
    for (pf, bd, fmt) in [
        (PixelFormat::Yuv420P, 8u8, ChromaFormat::Yuv420),
        (PixelFormat::Yuv444P10Le, 10, ChromaFormat::Yuv444),
        (PixelFormat::Gray8, 8, ChromaFormat::Monochrome),
    ] {
        let (w, h) = (96u32, 72u32);
        let frames: Vec<YuvFrame> = (0..7).map(|t| scene(w, h, bd, fmt, t)).collect();
        let p = params(w, h, pf, &[("gop", "3"), ("q", "110"), ("effort", "fast")]);
        let mut enc = make_encoder(&p).expect("encoder constructs");
        let extradata = enc.output_params().extradata.clone();

        // Packets arrive a group at a time.
        let mut packets = Vec::new();
        for (i, f) in frames.iter().enumerate() {
            enc.send_frame(&Frame::Video(to_video_frame(f, None)))
                .expect("send_frame");
            let got = drain(&mut enc);
            let expect = if (i + 1) % 3 == 0 { 3 } else { 0 };
            assert_eq!(got.len(), expect, "{pf:?}: packets after frame {i}");
            packets.extend(got);
        }
        enc.flush().expect("flush");
        packets.extend(drain(&mut enc));
        assert_eq!(packets.len(), frames.len(), "{pf:?}: one packet per frame");

        // Timestamps run; the first packet of each group is the sync
        // sample.
        for (i, pkt) in packets.iter().enumerate() {
            assert_eq!(pkt.pts, Some(i as i64), "{pf:?}");
            assert_eq!(pkt.dts, Some(i as i64), "{pf:?}");
            assert_eq!(
                pkt.flags.keyframe,
                i % 3 == 0,
                "{pf:?}: packet {i} sync flag"
            );
        }

        // The packets are the direct encoder's temporal units, group
        // by group, and the stream decodes to its reconstruction.
        let opts = GopOptions::new(110).with_effort(GopEffort::Fast);
        let mut want_planes: Vec<Vec<Vec<u8>>> = Vec::new();
        for (g, chunk) in frames.chunks(3).enumerate() {
            let direct = encode_gop_yuv_with_options(chunk, &opts).expect("direct");
            for (k, tu) in direct.temporal_units.iter().enumerate() {
                assert!(
                    packets[g * 3 + k].data == *tu,
                    "{pf:?}: group {g} packet {k} differs from the direct encode"
                );
            }
            for r in &direct.recon {
                let mut planes = vec![plane_bytes(bd, &r.y)];
                if fmt != ChromaFormat::Monochrome {
                    planes.push(plane_bytes(bd, &r.u));
                    planes.push(plane_bytes(bd, &r.v));
                }
                want_planes.push(planes);
            }
        }
        assert!(
            decode_all(&packets, &extradata) == want_planes,
            "{pf:?}: framework decode differs from the encoder reconstruction"
        );

        // The av1C record describes the sequence header on the wire.
        let seq = ObuIter::new(&packets[0].data)
            .filter_map(Result::ok)
            .find(|o| o.obu_type == ObuType::SequenceHeader)
            .map(|o| parse_sequence_header(o.payload).expect("sequence header"))
            .expect("the group's first packet carries the sequence header");
        assert_eq!(
            Av1CodecConfig::from_sequence_header(&seq).to_bytes(),
            extradata,
            "{pf:?}: extradata vs the coded sequence header"
        );
    }
}

#[test]
fn gop_mode_is_smaller_than_all_intra_and_honours_threads() {
    let (w, h) = (256u32, 192u32);
    let frames: Vec<YuvFrame> = (0..8)
        .map(|t| scene(w, h, 8, ChromaFormat::Yuv420, t))
        .collect();
    let intra = encode_all(
        &params(w, h, PixelFormat::Yuv420P, &[("q", "120")]),
        &frames,
        1,
    );
    let inter_params = params(w, h, PixelFormat::Yuv420P, &[("q", "120"), ("gop", "8")]);
    let inter = encode_all(&inter_params, &frames, 1);
    let bytes = |p: &[Packet]| p.iter().map(|x| x.data.len()).sum::<usize>();
    assert!(intra.iter().all(|p| p.flags.keyframe));
    assert_eq!(inter.iter().filter(|p| p.flags.keyframe).count(), 1);
    assert!(
        bytes(&inter) * 2 < bytes(&intra),
        "inter {} B should be well under all-intra {} B",
        bytes(&inter),
        bytes(&intra)
    );
    // A thread budget derives a tile layout (as on the intra modes);
    // the packets at a given layout do not depend on the budget.
    let tiled = params(
        w,
        h,
        PixelFormat::Yuv420P,
        &[
            ("q", "120"),
            ("gop", "8"),
            ("tile_cols_log2", "1"),
            ("tile_rows_log2", "1"),
        ],
    );
    let serial = encode_all(&tiled, &frames, 1);
    let threaded = encode_all(&tiled, &frames, 4);
    assert_eq!(serial.len(), threaded.len());
    for (a, b) in serial.iter().zip(&threaded) {
        assert!(a.data == b.data, "the packets depend on the thread budget");
    }
    let auto = encode_all(&inter_params, &frames, 4);
    assert_eq!(auto.len(), frames.len());
    assert_eq!(
        decode_all(&auto, &[]).len(),
        frames.len(),
        "the auto-tiled inter stream decodes"
    );
}

#[test]
fn gop_options_are_validated() {
    let ok = |opts: &[(&str, &str)]| make_encoder(&params(64, 48, PixelFormat::Yuv420P, opts));
    assert!(ok(&[("gop", "1")]).is_ok());
    assert!(ok(&[("gop", "64"), ("effort", "best")]).is_ok());
    assert!(ok(&[("gop", "0")]).is_err());
    assert!(ok(&[("gop", "65")]).is_err());
    assert!(ok(&[("gop", "many")]).is_err());
    assert!(ok(&[("effort", "ludicrous")]).is_err());
    assert!(ok(&[("gop", "4"), ("still", "true")]).is_err());
    assert!(ok(&[("gop", "1"), ("still", "true")]).is_ok());
}
