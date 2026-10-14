//! r475 — still-image decode budgets: the worker-budget identity gate.
//!
//! 1. A multi-tile still (deblock + CDEF + loop restoration elected by
//!    the crate's own encoder) decodes BYTE-IDENTICAL through the
//!    registry decoder for every `ExecutionContext` budget (1 / 2 / 4
//!    / 8 workers — the tile-parallel walk and the band-parallel
//!    post-filters) and equal to the direct serial session, at 4:2:0
//!    and 4:4:4, on picture extents that are not multiples of the
//!    superblock (partial tiles, partial bands, partial stripes).
//! 2. (`tests/decode_alloc_budget.rs`) the serial decode's
//!    live-allocation high-water mark and allocation count stay within
//!    a per-sample budget.

use oxideav_av1::decoder::SpecDecodeSession;
use oxideav_av1::encoder::still::{encode_still_yuv, StillOptions};
use oxideav_av1::encoder::yuv_frame::{ChromaFormat, YuvFrame};
use oxideav_av1::registry::make_decoder;
use oxideav_core::{CodecParameters, ExecutionContext, Frame, Packet, TimeBase};

/// A textured picture: slow gradients, a diagonal ridge pattern and
/// a pseudo-random grain, so the encoder keeps real coefficients and
/// elects the in-loop filters.
fn textured(width: u32, height: u32, format: ChromaFormat) -> YuvFrame {
    let mut f = YuvFrame::filled(width, height, 8, format, 0);
    let mut seed = 0x1234_5678u32;
    let mut noise = move || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 24) as i32 & 15
    };
    for y in 0..height as usize {
        for x in 0..width as usize {
            let g = (x * 3 / 7 + y * 2 / 5) % 256;
            let ridge = if ((x + y) / 13) % 3 == 0 { 40 } else { 0 };
            let v = (g as i32 + ridge + noise() - 8).clamp(0, 255);
            f.y[y * width as usize + x] = v as u16;
        }
    }
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    for y in 0..ch {
        for x in 0..cw {
            f.u[y * cw + x] = ((x * 5 / 3 + y) % 200 + 28) as u16;
            f.v[y * cw + x] = ((x + y * 7 / 4) % 180 + 40) as u16;
        }
    }
    f
}

fn encode_tiled(frame: &YuvFrame) -> Vec<u8> {
    let opts = StillOptions {
        tile_cols_log2: 1,
        tile_rows_log2: 1,
        ..StillOptions::new(100)
    };
    encode_still_yuv(frame, &opts)
        .expect("still encode")
        .temporal_unit_bytes
}

fn decode_registry(tu: &[u8], threads: usize) -> Vec<Vec<u8>> {
    let params = CodecParameters::video("av1".into());
    let mut dec = make_decoder(&params).expect("make_decoder");
    if threads > 1 {
        dec.set_execution_context(&ExecutionContext::with_threads(threads));
    }
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), tu.to_vec()))
        .expect("send_packet");
    dec.flush().expect("flush");
    let mut planes = Vec::new();
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => planes.extend(v.planes.into_iter().map(|p| p.data)),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore) => continue,
            Err(oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    }
    assert!(!planes.is_empty(), "no frame decoded");
    planes
}

fn check_identity_for_every_budget(frame: &YuvFrame) {
    let tu = encode_tiled(frame);
    let mut session = SpecDecodeSession::new();
    let direct = session.decode_temporal_unit(&tu).expect("direct decode");
    assert_eq!(direct.len(), 1);
    let serial = decode_registry(&tu, 1);
    assert_eq!(serial, direct[0].planes, "registry vs direct session");
    for threads in [2usize, 3, 4, 8] {
        let out = decode_registry(&tu, threads);
        assert_eq!(out, serial, "{threads} workers vs serial");
        let mut s = SpecDecodeSession::new();
        s.set_threads(threads);
        let d = s.decode_temporal_unit(&tu).expect("budgeted direct decode");
        assert_eq!(d[0].planes, serial, "direct session at {threads} workers");
    }
}

#[test]
fn tiled_420_still_decodes_identically_for_every_worker_budget() {
    check_identity_for_every_budget(&textured(1000, 700, ChromaFormat::Yuv420));
}

#[test]
fn tiled_444_still_decodes_identically_for_every_worker_budget() {
    check_identity_for_every_budget(&textured(776, 520, ChromaFormat::Yuv444));
}
