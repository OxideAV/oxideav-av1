//! Decode-budget harness (r475): decode an IVF file or a raw temporal
//! unit (`.obu` / `.av1`, the AVIF `av01` item payload) and report the
//! wall time per §7.4 decode stage, the end-to-end wall through the
//! registry decoder (`make_decoder`, the path a HEIF / AVIF consumer
//! takes) and a digest of every output plane.
//!
//! ```text
//! cargo run --release --example decode_bench -- input.ivf [more ...] [threads=N] [repeat=N]
//! ```
//!
//! `threads=N` hands the registry decoder an `ExecutionContext` budget
//! of `N` workers (serial when absent); `repeat=N` decodes each input
//! `N` times with a fresh decoder (the reported times are the best).
//! The digest is a 64-bit FNV-1a over the planes in output order, so
//! two runs are byte-identical iff their digests agree. Peak memory is
//! the caller's job (`/usr/bin/time -l` on macOS, `-v` on Linux).

use oxideav_av1::decoder::{DecodeStageTimings, SpecDecodeSession};
use oxideav_core::{CodecParameters, ExecutionContext, Frame, Packet, TimeBase};
use std::time::{Duration, Instant};

fn fnv1a(h: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0100_0000_01b3);
    }
}

fn main() {
    let mut inputs = Vec::new();
    let mut threads = 1usize;
    let mut repeat = 1usize;
    let mut mode = Mode::Both;
    let mut dump: Option<String> = None;
    for a in std::env::args().skip(1) {
        if let Some(v) = a.strip_prefix("threads=") {
            threads = v.parse().expect("threads=N");
        } else if let Some(v) = a.strip_prefix("repeat=") {
            repeat = v.parse().expect("repeat=N");
        } else if let Some(v) = a.strip_prefix("mode=") {
            mode = match v {
                "direct" => Mode::Direct,
                "registry" => Mode::Registry,
                _ => Mode::Both,
            };
        } else if let Some(v) = a.strip_prefix("dump=") {
            dump = Some(v.to_string());
        } else {
            inputs.push(a);
        }
    }
    assert!(
        !inputs.is_empty(),
        "usage: decode_bench <in.ivf|in.obu> [more ...] [threads=N] [repeat=N]"
    );
    for input in &inputs {
        bench(input, threads, repeat, mode, dump.as_deref());
    }
}

/// `mode=direct` runs only the per-stage session pass, `mode=registry`
/// only the end-to-end registry pass (for a clean peak-RSS reading of
/// one path); the default runs both.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Direct,
    Registry,
    Both,
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn bench(input: &str, threads: usize, repeat: usize, mode: Mode, dump: Option<&str>) {
    let data = std::fs::read(input).expect("read input");
    let is_ivf = data.len() >= 4 && &data[..4] == b"DKIF";

    // ---- Per-stage breakdown through the direct session. ----
    let mut best_direct = f64::MAX;
    let mut stages = DecodeStageTimings::new();
    let direct_runs = if mode == Mode::Registry { 0 } else { repeat };
    let registry_runs = if mode == Mode::Direct { 0 } else { repeat };
    for _ in 0..direct_runs {
        let t0 = Instant::now();
        let mut session = SpecDecodeSession::new();
        if threads > 1 {
            session.set_threads(threads);
        }
        let mut sum = DecodeStageTimings::new();
        if is_ivf {
            let reader = oxideav_av1::encoder::ivf::IvfReader::new(&data).expect("ivf");
            for rec in reader.read_all().expect("ivf records") {
                let _ = session.decode_temporal_unit(&rec.payload).expect("decode");
                let s = session.last_stage_timings();
                sum.parse += s.parse;
                sum.tiles += s.tiles;
                sum.deblock += s.deblock;
                sum.cdef += s.cdef;
                sum.superres += s.superres;
                sum.restoration += s.restoration;
                sum.ref_store += s.ref_store;
                sum.grain += s.grain;
                sum.output += s.output;
            }
        } else {
            let _ = session.decode_temporal_unit(&data).expect("decode");
            sum = session.last_stage_timings();
        }
        let wall = t0.elapsed().as_secs_f64();
        if wall < best_direct {
            best_direct = wall;
            stages = sum;
        }
    }

    // ---- End-to-end through the registry decoder. ----
    let params = CodecParameters::video("av1".into());
    let mut best = f64::MAX;
    let mut digest = 0u64;
    let mut frames = 0usize;
    let mut dims = (0usize, 0usize);
    let mut dumped: Vec<u8> = Vec::new();
    for _ in 0..registry_runs {
        let t0 = Instant::now();
        let mut dec = oxideav_av1::registry::make_decoder(&params).expect("make_decoder");
        if threads > 1 {
            dec.set_execution_context(&ExecutionContext::with_threads(threads));
        }
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), data.clone()))
            .expect("send_packet");
        dec.flush().expect("flush");
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        frames = 0;
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(v)) => {
                    frames += 1;
                    for p in &v.planes {
                        fnv1a(&mut h, &p.data);
                        if dump.is_some() {
                            dumped.extend_from_slice(&p.data);
                        }
                    }
                    if let Some(p) = v.planes.first() {
                        dims = (p.stride, p.data.len() / p.stride.max(1));
                    }
                }
                Ok(_) => {}
                Err(oxideav_core::Error::NeedMore) => continue,
                Err(oxideav_core::Error::Eof) => break,
                Err(e) => panic!("receive_frame: {e}"),
            }
        }
        drop(dec);
        best = best.min(t0.elapsed().as_secs_f64());
        digest = h;
    }
    if let Some(path) = dump {
        std::fs::write(path, &dumped).expect("write dump");
    }
    println!(
        "{input}: {frames} frame(s) {}x{} (plane 0 bytes), threads={threads}, best {:.3} s (direct session {:.3} s), digest {digest:016x}",
        dims.0, dims.1, best, best_direct
    );
    let total = stages.total();
    let row = |name: &str, d: Duration| {
        println!(
            "  {name:<12} {:>9.1} ms {:>5.1} %",
            ms(d),
            100.0 * d.as_secs_f64() / total.as_secs_f64().max(1e-9)
        );
    };
    row("parse", stages.parse);
    row("tiles", stages.tiles);
    row("deblock", stages.deblock);
    row("cdef", stages.cdef);
    row("superres", stages.superres);
    row("restoration", stages.restoration);
    row("ref_store", stages.ref_store);
    row("grain", stages.grain);
    row("output", stages.output);
    row("(sum)", total);
}
