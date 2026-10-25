//! `Decoder::output_params` / `output_params_generation` (oxideav-core
//! 0.1.42): the registry decoder states the emitted frames' size,
//! pixel layout, colour signal and picture rate at the stream level,
//! as early as it can, and bumps the generation exactly at the first
//! frame of a new shape.
//!
//! The `svt_*.ivf` fixtures come from black-box ffmpeg / libsvtav1
//! (`tests/fixtures/output_params/gen.sh`, which also prints the
//! `ffprobe -show_streams` values pinned below). The 4:2:2 / 4:4:4 /
//! monochrome / 12-bit layouts are encoded in-process with this
//! crate's own encoder; `OXIDEAV_AV1_OUTPUT_PARAMS_DUMP=<dir>` writes
//! them out as IVF for the same ffprobe check (verified: 4:2:2 8-bit
//! `yuv422p`, 4:4:4 8/10-bit `yuv444p` / `yuv444p10le`, monochrome
//! 8/10-bit `gray` / `gray10le`, 4:2:0 12-bit `yuv420p12le`, each at
//! the encoded size, `color_range=tv`).

use oxideav_av1::encoder::ivf::{IvfReader, IvfWriter};
use oxideav_av1::registry::{make_decoder, make_encoder};
use oxideav_core::{
    CodecId, CodecParameters, ColorPrimaries, ColorRange, ColorSignal, Decoder, Error, Frame,
    MatrixCoefficients, Packet, PixelFormat, TimeBase, TransferCharacteristics, VideoFrame,
    VideoPlane,
};

const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/output_params");

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIX}/{name}")).expect("fixture readable")
}

/// The temporal-unit payloads of an IVF file, in file order.
fn temporal_units(ivf: &[u8]) -> Vec<Vec<u8>> {
    IvfReader::new(ivf)
        .expect("ivf header")
        .read_all()
        .expect("ivf records")
        .into_iter()
        .map(|r| r.payload)
        .collect()
}

fn decoder(extradata: Vec<u8>, threads: usize) -> Box<dyn Decoder> {
    let mut p = CodecParameters::video(CodecId::new("av1"));
    p.extradata = extradata;
    let mut d = make_decoder(&p).expect("decoder");
    d.set_execution_context(&oxideav_core::ExecutionContext::with_threads(threads));
    d
}

/// Bytes per sample and chroma shift of a planar layout.
fn layout(pf: PixelFormat) -> (usize, Option<(u32, u32)>) {
    use PixelFormat::*;
    match pf {
        Gray8 => (1, None),
        Gray10Le | Gray12Le => (2, None),
        Yuv420P => (1, Some((1, 1))),
        Yuv422P => (1, Some((1, 0))),
        Yuv444P => (1, Some((0, 0))),
        Yuv420P10Le | Yuv420P12Le => (2, Some((1, 1))),
        Yuv422P10Le | Yuv422P12Le => (2, Some((1, 0))),
        Yuv444P10Le | Yuv444P12Le => (2, Some((0, 0))),
        other => panic!("unexpected layout {other:?}"),
    }
}

/// The emitted frame's planes are the truth: they must have exactly
/// the shape `params` states.
fn assert_frame_matches(vf: &VideoFrame, params: &CodecParameters) {
    let (w, h) = (params.width.unwrap(), params.height.unwrap());
    let (bps, chroma) = layout(params.pixel_format.unwrap());
    let mut dims = vec![(w, h)];
    if let Some((sx, sy)) = chroma {
        let c = ((w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy);
        dims.extend([c, c]);
    }
    assert_eq!(vf.planes.len(), dims.len(), "plane count");
    for (plane, &(pw, ph)) in vf.planes.iter().zip(&dims) {
        assert_eq!(plane.stride, pw as usize * bps, "plane stride");
        assert_eq!(plane.data.len(), plane.stride * ph as usize, "plane rows");
    }
}

/// Feed `units` one per packet, draining after each; returns, per
/// returned frame, `(generation, params)` read right after it.
fn decode_all(dec: &mut dyn Decoder, units: &[Vec<u8>]) -> Vec<(u64, CodecParameters)> {
    let mut seen = Vec::new();
    let drain = |dec: &mut dyn Decoder, seen: &mut Vec<_>| {
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    let p = dec.output_params().expect("described after a frame");
                    assert_frame_matches(&vf, &p);
                    assert_eq!(dec.output_pixel_format(), p.pixel_format);
                    seen.push((dec.output_params_generation(), p));
                }
                Ok(_) => panic!("video only"),
                Err(Error::NeedMore) | Err(Error::Eof) => break,
                Err(e) => panic!("decode: {e}"),
            }
        }
    };
    for (i, u) in units.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), u.clone()).with_pts(i as i64))
            .expect("packet");
        drain(dec, &mut seen);
    }
    dec.flush().unwrap();
    drain(dec, &mut seen);
    seen
}

fn bt709_full() -> ColorSignal {
    ColorSignal::new(
        ColorRange::Full,
        ColorPrimaries::BT709,
        TransferCharacteristics::BT709,
        MatrixCoefficients::BT709,
    )
}

/// ffprobe: `width=70 height=46 pix_fmt=yuv420p color_range=tv`, no
/// colour description; `width=64 height=48 pix_fmt=yuv420p10le
/// color_range=pc color_space=bt709 color_transfer=bt709
/// color_primaries=bt709`.
#[test]
fn svt_fixtures_match_ffprobe_before_and_after_the_first_frame() {
    for (name, w, h, pf, color) in [
        (
            "svt_420p_70x46.ivf",
            70,
            46,
            PixelFormat::Yuv420P,
            ColorSignal::from_code_points(2, 2, 2, false),
        ),
        (
            "svt_420p10_64x48_bt709_full.ivf",
            64,
            48,
            PixelFormat::Yuv420P10Le,
            bt709_full(),
        ),
    ] {
        let units = temporal_units(&fixture(name));
        let mut dec = decoder(Vec::new(), 1);
        assert!(dec.output_params().is_none(), "{name}: nothing known yet");
        // The first temporal unit carries the sequence header: the
        // description is known once it is parsed, before any receive.
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), units[0].clone()))
            .unwrap();
        let early = dec.output_params().expect("known from the sequence header");
        assert_eq!((early.width, early.height), (Some(w), Some(h)), "{name}");
        assert_eq!(early.pixel_format, Some(pf), "{name}");
        assert_eq!(early.color_signal, color, "{name}");
        // libsvtav1 writes no §5.5.3 timing info (ffprobe's
        // r_frame_rate comes from the IVF header), so no rate is stated.
        assert_eq!(early.frame_rate, None, "{name}");
        assert_eq!(dec.output_params_generation(), 0);

        let mut dec = decoder(Vec::new(), 1);
        let seen = decode_all(dec.as_mut(), &units);
        assert_eq!(seen.len(), 2, "{name}: two shown frames");
        for (g, p) in &seen {
            assert_eq!(*g, 0, "{name}: fixed shape, no bump");
            assert_eq!(
                (p.width, p.height, p.pixel_format),
                (Some(w), Some(h), Some(pf))
            );
            assert_eq!(p.color_signal, color);
        }
    }
}

/// The crate's own encoder covers the layouts libsvtav1 does not:
/// 4:2:2, 4:4:4, monochrome, 12-bit. Its `av1C` extradata (no
/// `configOBUs`) states the layout up front, before any packet.
#[test]
fn own_encoder_layouts_are_stated_from_extradata_then_from_the_frames() {
    let dump = std::env::var("OXIDEAV_AV1_OUTPUT_PARAMS_DUMP").ok();
    for (pf, w, h) in [
        (PixelFormat::Yuv422P, 48, 32),
        (PixelFormat::Yuv444P10Le, 32, 16),
        (PixelFormat::Gray8, 32, 32),
        (PixelFormat::Gray10Le, 16, 32),
        (PixelFormat::Yuv420P12Le, 48, 16),
        (PixelFormat::Yuv444P, 16, 16),
    ] {
        let (units, extradata) = own_encode(pf, w, h, 2, false);
        if let Some(dir) = &dump {
            let mut wr = IvfWriter::new(Vec::new(), *b"AV01", w as u16, h as u16, 30, 1).unwrap();
            for (i, u) in units.iter().enumerate() {
                wr.write_frame(u, i as u64).unwrap();
            }
            std::fs::write(format!("{dir}/own_{pf:?}_{w}x{h}.ivf"), wr.into_inner()).unwrap();
        }
        let dec = decoder(extradata.clone(), 1);
        let early = dec.output_params().expect("av1C states the layout");
        assert_eq!(early.pixel_format, Some(pf), "{pf:?} from av1C");
        assert_eq!(early.width, None, "av1C carries no size");
        let mut dec = decoder(extradata, 1);
        let seen = decode_all(dec.as_mut(), &units);
        assert_eq!(seen.len(), 2);
        for (g, p) in &seen {
            assert_eq!(*g, 0);
            assert_eq!(
                (p.width, p.height, p.pixel_format),
                (Some(w), Some(h), Some(pf))
            );
            assert_eq!(p.color_signal.range, ColorRange::Limited);
        }
    }
}

/// Encode `n` textured frames with the crate's framework encoder
/// (every frame a KEY frame under a repeated sequence header).
fn own_encode(pf: PixelFormat, w: u32, h: u32, n: usize, full: bool) -> (Vec<Vec<u8>>, Vec<u8>) {
    let mut p = CodecParameters::video(CodecId::new("av1"));
    p.width = Some(w);
    p.height = Some(h);
    p.pixel_format = Some(pf);
    p.options.insert("q", "120");
    if full {
        p.options.insert("full_range", "true");
        p.options.insert("color_primaries", "1");
        p.options.insert("transfer_characteristics", "1");
        p.options.insert("matrix_coefficients", "1");
    }
    let mut enc = make_encoder(&p).expect("encoder");
    let extradata = enc.output_params().extradata.clone();
    let (bps, chroma) = layout(pf);
    let mut units = Vec::new();
    for i in 0..n {
        let mut dims = vec![(w, h)];
        if let Some((sx, sy)) = chroma {
            let c = ((w + (1 << sx) - 1) >> sx, (h + (1 << sy) - 1) >> sy);
            dims.extend([c, c]);
        }
        let planes = dims
            .iter()
            .enumerate()
            .map(|(pi, &(pw, ph))| {
                let mut data = Vec::new();
                for k in 0..(pw * ph) as usize {
                    let v = ((k * (3 + pi) + i * 17) % 200) as u16 + 16;
                    if bps == 2 {
                        data.extend_from_slice(&(v << 2).to_le_bytes());
                    } else {
                        data.push(v as u8);
                    }
                }
                VideoPlane {
                    stride: pw as usize * bps,
                    data,
                }
            })
            .collect();
        enc.send_frame(&Frame::Video(VideoFrame {
            pts: Some(i as i64),
            planes,
        }))
        .expect("frame");
    }
    enc.flush().unwrap();
    while let Ok(pkt) = enc.receive_packet() {
        units.push(pkt.data);
    }
    assert_eq!(units.len(), n);
    (units, extradata)
}

/// A stream whose coded video sequences change shape mid-stream
/// (size, then bit depth + colour signal, then layout): the generation
/// bumps exactly at the first frame of each new shape — not when the
/// new sequence header is parsed — and the answer before that frame
/// still describes the last frame returned. Identical at 1 and 8
/// worker threads.
#[test]
fn mid_stream_changes_bump_the_generation_at_the_first_new_frame() {
    let a = temporal_units(&fixture("svt_420p_70x46.ivf"));
    let b = temporal_units(&fixture("svt_420p10_64x48_bt709_full.ivf"));
    let (c, _) = own_encode(PixelFormat::Yuv444P, 32, 32, 2, true);
    let (d, _) = own_encode(PixelFormat::Yuv444P, 32, 32, 2, true);
    let units: Vec<Vec<u8>> = a.iter().chain(&b).chain(&c).chain(&d).cloned().collect();
    let expect = [
        (0, 70, 46, PixelFormat::Yuv420P),
        (0, 70, 46, PixelFormat::Yuv420P),
        (1, 64, 48, PixelFormat::Yuv420P10Le),
        (1, 64, 48, PixelFormat::Yuv420P10Le),
        (2, 32, 32, PixelFormat::Yuv444P),
        (2, 32, 32, PixelFormat::Yuv444P),
        // A repeated identical shape is not a change.
        (2, 32, 32, PixelFormat::Yuv444P),
        (2, 32, 32, PixelFormat::Yuv444P),
    ];
    let mut runs = Vec::new();
    for threads in [1, 8] {
        let mut dec = decoder(Vec::new(), threads);
        let seen = decode_all(dec.as_mut(), &units);
        let got: Vec<_> = seen
            .iter()
            .map(|(g, p)| {
                (
                    *g,
                    p.width.unwrap(),
                    p.height.unwrap(),
                    p.pixel_format.unwrap(),
                )
            })
            .collect();
        assert_eq!(got, expect, "threads={threads}");
        assert_eq!(seen[2].1.color_signal, bt709_full());
        assert_eq!(seen[4].1.color_signal, bt709_full());
        assert_eq!(seen[0].1.color_signal.range, ColorRange::Limited);
        runs.push(
            seen.iter()
                .map(|(g, p)| (*g, format!("{p:?}")))
                .collect::<Vec<_>>(),
        );

        // Parsing the next sequence's header does not move the
        // answer: it still describes the last frame returned.
        let mut dec = decoder(Vec::new(), threads);
        let first = decode_all(dec.as_mut(), &a);
        dec.reset().unwrap();
        let before = format!("{:?}", dec.output_params().unwrap());
        assert_eq!(before, format!("{:?}", first.last().unwrap().1));
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), b[0].clone()))
            .unwrap();
        assert_eq!(dec.output_params_generation(), 0, "no bump at the header");
        assert_eq!(format!("{:?}", dec.output_params().unwrap()), before);
    }
    assert_eq!(runs[0], runs[1], "1 vs 8 threads");
}
