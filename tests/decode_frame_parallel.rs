//! Non-reference frames decoded on worker threads: a B-pyramid GOP fed
//! one temporal unit per packet decodes BYTE-IDENTICAL through the
//! registry decoder for every `ExecutionContext` budget, and through
//! `SpecDecodeSession::decode_temporal_unit_pipelined` with the pending
//! frames collected only after the whole stream was submitted. Under a
//! budget the reference frames decode on workers too, each waiting only
//! for the slots it reads.

use oxideav_av1::decoder::{SpecDecodeSession, decode_av1_spec};
use oxideav_av1::encoder::{Yuv420Frame, encode_pyramid_gop_yuv420_with_q};
use oxideav_av1::registry::make_decoder;
use oxideav_core::{CodecParameters, ExecutionContext, Frame, Packet, TimeBase};

fn moving(w: u32, h: u32, t: usize) -> Yuv420Frame {
    let (wu, hu) = (w as usize, h as usize);
    let mut f = Yuv420Frame::filled(w, h, 0);
    for i in 0..hu {
        for j in 0..wu {
            let (si, sj) = (i + t, j + 2 * t);
            f.y[i * wu + j] = ((si * 5 + sj * 3 + (si / 16) * (sj / 16)) % 256) as u8;
        }
    }
    for i in 0..hu / 2 {
        for j in 0..wu / 2 {
            f.u[i * (wu / 2) + j] = ((128 + i * 2 + j + t) % 256) as u8;
            f.v[i * (wu / 2) + j] = ((64 + i + j * 2 + t) % 256) as u8;
        }
    }
    f
}

fn gop() -> (Vec<Vec<u8>>, Vec<Vec<Vec<u8>>>) {
    gop_of(5)
}

/// A KEY frame plus `n - 1` B-pyramid frames: hidden ALT / MID
/// references, shown leaves and `show_existing_frame` units.
fn gop_of(n: usize) -> (Vec<Vec<u8>>, Vec<Vec<Vec<u8>>>) {
    let frames: Vec<Yuv420Frame> = (0..n).map(|t| moving(72, 40, t)).collect();
    let enc = encode_pyramid_gop_yuv420_with_q(&frames, 90).expect("pyramid encode");
    let reference: Vec<Vec<Vec<u8>>> = decode_av1_spec(&enc.ivf_bytes)
        .expect("spec decode")
        .into_iter()
        .map(|f| f.planes)
        .collect();
    assert_eq!(reference.len(), frames.len());
    (enc.temporal_units, reference)
}

fn decode_registry(units: &[Vec<u8>], threads: usize) -> Vec<Vec<Vec<u8>>> {
    let mut dec = make_decoder(&CodecParameters::video("av1".into())).expect("make_decoder");
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    let mut out = Vec::new();
    let drain = |dec: &mut Box<dyn oxideav_core::Decoder>, out: &mut Vec<Vec<Vec<u8>>>| loop {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => out.push(v.planes.into_iter().map(|p| p.data).collect()),
            Ok(_) => {}
            Err(oxideav_core::Error::NeedMore | oxideav_core::Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    };
    for (i, tu) in units.iter().enumerate() {
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 30), tu.clone()).with_pts(i as i64))
            .expect("send_packet");
        drain(&mut dec, &mut out);
    }
    dec.flush().expect("flush");
    drain(&mut dec, &mut out);
    out
}

#[test]
fn pyramid_decodes_identically_for_every_worker_budget() {
    let (units, reference) = gop();
    for threads in [1usize, 2, 3, 4, 8] {
        let out = decode_registry(&units, threads);
        assert_eq!(out.len(), reference.len(), "{threads} workers: frame count");
        for (i, (a, b)) in out.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a, b, "{threads} workers: display frame {i}");
        }
    }
}

#[test]
fn pipelined_session_frames_resolve_after_the_stream() {
    let (units, reference) = gop();
    let mut s = SpecDecodeSession::new();
    s.set_threads(4);
    let mut pending = Vec::new();
    for tu in &units {
        pending.extend(s.decode_temporal_unit_pipelined(tu).expect("submit"));
    }
    let frames: Vec<Vec<Vec<u8>>> = pending
        .into_iter()
        .map(|p| p.into_frame().expect("deferred decode").planes)
        .collect();
    assert_eq!(frames, reference);
}

#[test]
fn deep_pyramid_reference_frames_decode_identically_for_every_worker_budget() {
    // A full 16-frame mini-GOP: four layers of hidden references, each
    // read by frames decoded on other workers.
    let (units, reference) = gop_of(17);
    for threads in [1usize, 2, 3, 4, 8, 16] {
        let out = decode_registry(&units, threads);
        assert_eq!(out.len(), reference.len(), "{threads} workers: frame count");
        for (i, (a, b)) in out.iter().zip(reference.iter()).enumerate() {
            assert_eq!(a, b, "{threads} workers: display frame {i}");
        }
    }
    let mut s = SpecDecodeSession::new();
    s.set_threads(8);
    let mut pending = Vec::new();
    for tu in &units {
        pending.extend(s.decode_temporal_unit_pipelined(tu).expect("submit"));
    }
    let frames: Vec<Vec<Vec<u8>>> = pending
        .into_iter()
        .map(|p| p.into_frame().expect("deferred decode").planes)
        .collect();
    assert_eq!(frames, reference);
}
