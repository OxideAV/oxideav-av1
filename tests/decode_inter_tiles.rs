//! r489 — tile-parallel reconstruction of INTER frames and the
//! replayed (single-parse) reconstruction walk of the frame-parallel
//! decoder: multi-tile KEY + P streams decode BYTE-IDENTICAL at every
//! worker budget, through the registry decoder (one temporal unit per
//! packet — frames on workers, each walking its tiles on workers) and
//! through a direct session (frames in line, tiles on workers).
//!
//! The streams come from this crate's GOP encoder at multi-tile
//! layouts: panning content (vectors reaching above and left of the
//! frame), a block moving against the pan, intra leaves inside inter
//! frames, CDEF and — at the `Thorough` effort — loop restoration, so
//! every stitched grid (mode info, samples, restoration units) and the
//! in-loop filters behind the stitch are exercised.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.11.1, §5.11.2 (tiles are
//! independently decodable), §7.9, §7.11.3, §7.20.

use oxideav_av1::decoder::{SpecDecodeSession, decode_av1_spec};
use oxideav_av1::encoder::{
    ChromaFormat, GopEffort, GopOptions, YuvFrame, encode_gop_yuv_with_options,
};
use oxideav_av1::registry::make_decoder;
use oxideav_core::{CodecParameters, ExecutionContext, Frame, Packet, TimeBase};

fn scene(w: u32, h: u32, fmt: ChromaFormat, t: usize) -> YuvFrame {
    let mut f = YuvFrame::filled(w, h, 8, fmt, 0);
    let (wu, hu) = (w as usize, h as usize);
    let tex = |i: usize, j: usize| -> usize {
        (40 + (i * 7 + j * 5) % 61 + ((i / 6) * 13 + (j / 9) * 29) % 97 + ((i ^ j) & 15)) % 256
    };
    let (ox, oy) = (wu / 3 + 4 * t, hu / 4 + 2 * t);
    for i in 0..hu {
        for j in 0..wu {
            let v = if (oy..oy + hu / 4).contains(&i) && (ox..ox + wu / 5).contains(&j) {
                200 - ((i - oy) * 3 + (j - ox) * 2) % 90
            } else {
                // The pan moves content DOWN and RIGHT, so the vectors
                // of the top / left blocks point outside the frame.
                tex(i + 3 * (8 - t), j + 5 * (8 - t))
            };
            f.y[i * wu + j] = v as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for i in 0..ch {
        for j in 0..cw {
            f.u[i * cw + j] = ((128 + i * 2 + j + 3 * t) % 256) as u16;
            f.v[i * cw + j] = ((64 + i + j * 2 + t) % 256) as u16;
        }
    }
    f
}

type Planes = Vec<Vec<u8>>;

/// `(temporal units, IVF bytes, serial reference)`.
fn stream(
    w: u32,
    h: u32,
    fmt: ChromaFormat,
    n: usize,
    effort: GopEffort,
    tiles: (u32, u32),
) -> (Vec<Vec<u8>>, Vec<u8>, Vec<Planes>) {
    let frames: Vec<YuvFrame> = (0..n).map(|t| scene(w, h, fmt, t)).collect();
    let opts = GopOptions::new(96)
        .with_effort(effort)
        .with_tiles_log2(tiles.0, tiles.1);
    let enc = encode_gop_yuv_with_options(&frames, &opts).expect("multi-tile GOP");
    let reference: Vec<Planes> = decode_av1_spec(&enc.ivf_bytes)
        .expect("serial decode")
        .into_iter()
        .map(|f| f.planes)
        .collect();
    assert_eq!(reference.len(), n);
    (enc.temporal_units, enc.ivf_bytes, reference)
}

fn decode_registry(units: &[Vec<u8>], threads: usize) -> Vec<Planes> {
    let mut dec = make_decoder(&CodecParameters::video("av1".into())).expect("make_decoder");
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut out = Vec::new();
    let drain = |dec: &mut Box<dyn oxideav_core::Decoder>, out: &mut Vec<Planes>| loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v.planes.into_iter().map(|p| p.data).collect()),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore | oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    };
    for (i, tu) in units.iter().enumerate() {
        let pkt = Packet::new(0, TimeBase::new(1, 25), tu.clone()).with_pts(i as i64);
        dec.send_packet(&pkt).expect("send_packet");
        drain(&mut dec, &mut out);
    }
    dec.flush().expect("flush");
    drain(&mut dec, &mut out);
    out
}

fn decode_session(ivf: &[u8], threads: usize) -> Vec<Planes> {
    let mut session = SpecDecodeSession::new();
    session.set_threads(threads);
    session
        .decode_ivf(ivf)
        .expect("session decode")
        .into_iter()
        .map(|f| f.planes)
        .collect()
}

fn assert_every_budget(tag: &str, units: &[Vec<u8>], ivf: &[u8], reference: &[Planes]) {
    for threads in [1usize, 2, 3, 4, 8] {
        assert!(
            decode_registry(units, threads) == reference,
            "{tag}: registry decode differs at {threads} worker(s)"
        );
        assert!(
            decode_session(ivf, threads) == reference,
            "{tag}: session decode differs at {threads} worker(s)"
        );
    }
}

#[test]
fn the_serial_tile_walk_opt_out_decodes_the_same_frames() {
    let (units, ivf, reference) =
        stream(256, 192, ChromaFormat::Yuv420, 5, GopEffort::Fast, (1, 1));
    for threads in [2usize, 4, 8] {
        let mut session = SpecDecodeSession::new();
        assert!(
            session.inter_tile_parallel(),
            "tile-parallel is the default"
        );
        session.set_inter_tile_parallel(false);
        assert!(!session.inter_tile_parallel());
        session.set_threads(threads);
        let out: Vec<Planes> = session
            .decode_ivf(&ivf)
            .expect("session decode")
            .into_iter()
            .map(|f| f.planes)
            .collect();
        assert!(out == reference, "serial tile walk differs at {threads}");

        let mut params = CodecParameters::video("av1".into());
        params.options.insert("inter_tile_parallel", "false");
        let mut dec = make_decoder(&params).expect("make_decoder");
        dec.set_execution_context(&ExecutionContext::with_threads(threads));
        let mut got: Vec<Planes> = Vec::new();
        for (i, tu) in units.iter().enumerate() {
            let pkt = Packet::new(0, TimeBase::new(1, 25), tu.clone()).with_pts(i as i64);
            dec.send_packet(&pkt).expect("send_packet");
            while let Ok(Frame::Video(v)) = dec.receive_frame() {
                got.push(v.planes.into_iter().map(|p| p.data).collect());
            }
        }
        dec.flush().expect("flush");
        while let Ok(Frame::Video(v)) = dec.receive_frame() {
            got.push(v.planes.into_iter().map(|p| p.data).collect());
        }
        assert!(got == reference, "registry opt-out differs at {threads}");
    }
    let mut params = CodecParameters::video("av1".into());
    params.options.insert("inter_tile_parallel", "sometimes");
    assert!(make_decoder(&params).is_err());
}

#[test]
fn four_tile_p_gop_decodes_identically_for_every_worker_budget() {
    let (units, ivf, reference) = stream(
        256,
        192,
        ChromaFormat::Yuv420,
        6,
        GopEffort::Balanced,
        (1, 1),
    );
    assert_every_budget("2x2 tiles 4:2:0", &units, &ivf, &reference);
}

#[test]
fn eight_tile_p_gop_with_restoration_decodes_identically_for_every_worker_budget() {
    let (units, ivf, reference) = stream(
        320,
        192,
        ChromaFormat::Yuv420,
        5,
        GopEffort::Thorough,
        (2, 1),
    );
    assert_every_budget("4x2 tiles 4:2:0 + LR", &units, &ivf, &reference);
}

#[test]
fn tiled_444_p_gop_decodes_identically_for_every_worker_budget() {
    let (units, ivf, reference) =
        stream(192, 128, ChromaFormat::Yuv444, 4, GopEffort::Fast, (1, 1));
    assert_every_budget("2x2 tiles 4:4:4", &units, &ivf, &reference);
}

#[test]
fn single_tile_p_gop_replays_identically_for_every_worker_budget() {
    // One tile: the frame-parallel path alone (syntax walk recorded,
    // reconstruction walk replayed).
    let (units, ivf, reference) = stream(
        160,
        120,
        ChromaFormat::Yuv420,
        8,
        GopEffort::Balanced,
        (0, 0),
    );
    assert_every_budget("single tile", &units, &ivf, &reference);
}
