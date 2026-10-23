//! Encode-budget harness (r479): encode a raw planar picture as ONE
//! still through [`oxideav_av1::encoder::encode_still_yuv`] and report
//! the wall clock, the coded size, a digest of the temporal unit (two
//! runs are byte-identical iff the digests agree), the PSNR of the
//! reconstruction against the source, and the encoder's memory by
//! stage: a counting allocator samples its live bytes and high-water
//! mark at every stage boundary the encoder's stage probe fires.
//!
//! ```text
//! cargo run --release --example encode_bench -- --raw photo.yuv --size 4032x3024 \
//!     [--fmt yuv420p|yuv420p10le|yuv444p|gray] [--q 102] [--speed fast|balanced|thorough] \
//!     [--threads N] [--repeat N] [--out still.obu] [--trace MiB]
//! ```
//!
//! The process holds the raw bytes and the `u16` frame built from them
//! (the same two copies the AVIF registry encoder's bench holds) and
//! nothing else before the encode starts, so `/usr/bin/time -l`'s peak
//! RSS is the encoder's plus those copies. `--trace N` prints every
//! allocation of at least `N` MiB with the stage it lands in.

use oxideav_av1::encoder::stage_probe::set_stage_probe;
use oxideav_av1::encoder::yuv_frame::{ChromaFormat, YuvFrame};
use oxideav_av1::encoder::{StillOptions, StillSpeed, encode_still_yuv};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static TRACE_MIN: AtomicUsize = AtomicUsize::new(usize::MAX);
/// `--sample N`: every `N`-th allocation of a thread records its
/// backtrace (0 = off).
static SAMPLE_EVERY: AtomicUsize = AtomicUsize::new(0);
static SAMPLES_BT: Mutex<Vec<(usize, std::backtrace::Backtrace)>> = Mutex::new(Vec::new());

thread_local! {
    static SAMPLE_COUNTER: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static IN_HOOK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn maybe_sample(size: usize) {
    let every = SAMPLE_EVERY.load(Ordering::Relaxed);
    if every == 0 {
        return;
    }
    let fire = SAMPLE_COUNTER.with(|c| {
        let n = c.get() + 1;
        c.set(n);
        n % every == 0
    });
    if !fire || IN_HOOK.with(|f| f.replace(true)) {
        return;
    }
    let bt = std::backtrace::Backtrace::force_capture();
    SAMPLES_BT
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((size, bt));
    IN_HOOK.with(|f| f.set(false));
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            let live = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(live, Ordering::Relaxed);
            if layout.size() >= TRACE_MIN.load(Ordering::Relaxed) {
                trace("ALLOC", layout.size(), live);
            }
            maybe_sample(layout.size());
            System.alloc(layout)
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe {
            let live = LIVE.fetch_sub(layout.size(), Ordering::Relaxed) - layout.size();
            if layout.size() >= TRACE_MIN.load(Ordering::Relaxed) {
                trace("FREE ", layout.size(), live);
            }
            System.dealloc(ptr, layout)
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
            let old = layout.size();
            if new_size > old {
                let live = LIVE.fetch_add(new_size - old, Ordering::Relaxed) + (new_size - old);
                PEAK.fetch_max(live, Ordering::Relaxed);
                if new_size >= TRACE_MIN.load(Ordering::Relaxed) {
                    trace("GROW ", new_size, live);
                }
            } else {
                LIVE.fetch_sub(old - new_size, Ordering::Relaxed);
            }
            System.realloc(ptr, layout, new_size)
        }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Stage samples `(label, live MiB at the boundary, peak MiB since
/// the previous boundary, process RSS MiB at the boundary)` the probe
/// collects. Each label names the stage that STARTS at the boundary,
/// so a row's peak column is the previous row's stage.
type Sample = (&'static str, f64, f64, f64, f64);
static SAMPLES: Mutex<Vec<Sample>> = Mutex::new(Vec::new());
/// Wall-clock origin of the current run (seconds since the harness
/// started, captured at each boundary).
static RUN_T0: Mutex<Option<Instant>> = Mutex::new(None);

/// The process's resident set in MiB (`ps`; `-1` when unavailable).
fn rss_mib() -> f64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map_or(-1.0, |kib| kib / 1024.0)
}
static STAGE: Mutex<&'static str> = Mutex::new("harness");

fn trace(what: &str, size: usize, live: usize) {
    use std::io::Write;
    let stage = *STAGE.lock().unwrap_or_else(|e| e.into_inner());
    let msg = format!(
        "{what} {:>7.1} MiB -> live {:>7.1} MiB [{stage}]\n",
        mib(size),
        mib(live)
    );
    let _ = std::io::stderr().write_all(msg.as_bytes());
}

fn mib(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn psnr(a: &[u16], b: &[u16], bit_depth: u8) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    let mse = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum::<f64>()
        / a.len() as f64;
    let max = f64::from((1u32 << bit_depth) - 1);
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (max * max / mse).log10()
    }
}

struct Args {
    raw: String,
    width: u32,
    height: u32,
    fmt: String,
    q: u8,
    speed: StillSpeed,
    threads: usize,
    repeat: usize,
    out: Option<String>,
    trace_mib: Option<usize>,
    sample: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        raw: String::new(),
        width: 0,
        height: 0,
        fmt: "yuv420p".into(),
        q: 102,
        speed: StillSpeed::Fast,
        threads: 1,
        repeat: 1,
        out: None,
        trace_mib: None,
        sample: 0,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--raw" => a.raw = it.next().expect("--raw PATH"),
            "--size" => {
                let s = it.next().expect("--size WxH");
                let (w, h) = s.split_once('x').expect("WxH");
                a.width = w.parse().expect("width");
                a.height = h.parse().expect("height");
            }
            "--fmt" => a.fmt = it.next().expect("--fmt"),
            "--q" => a.q = it.next().expect("--q N").parse().expect("N"),
            "--speed" => {
                a.speed = match it.next().expect("--speed").as_str() {
                    "fast" => StillSpeed::Fast,
                    "balanced" => StillSpeed::Balanced,
                    "thorough" => StillSpeed::Thorough,
                    other => panic!("unknown --speed {other}"),
                }
            }
            "--threads" => a.threads = it.next().expect("--threads N").parse().expect("N"),
            "--repeat" => a.repeat = it.next().expect("--repeat N").parse().expect("N"),
            "--out" => a.out = it.next(),
            "--trace" => a.trace_mib = Some(it.next().expect("--trace MiB").parse().expect("MiB")),
            "--sample" => a.sample = it.next().expect("--sample N").parse().expect("N"),
            other => panic!("unknown argument {other}"),
        }
    }
    assert!(
        !a.raw.is_empty() && a.width > 0 && a.height > 0,
        "need --raw and --size"
    );
    a
}

/// Build the `u16` frame from the raw planar bytes (8-bit samples one
/// byte each, 10-bit little-endian two bytes each).
fn frame_of(raw: &[u8], w: u32, h: u32, fmt: &str) -> YuvFrame {
    let (format, bit_depth) = match fmt {
        "yuv420p" => (ChromaFormat::Yuv420, 8u8),
        "yuv420p10le" => (ChromaFormat::Yuv420, 10),
        "yuv444p" => (ChromaFormat::Yuv444, 8),
        "yuv444p10le" => (ChromaFormat::Yuv444, 10),
        "yuv422p" => (ChromaFormat::Yuv422, 8),
        "gray" => (ChromaFormat::Monochrome, 8),
        other => panic!("unknown --fmt {other}"),
    };
    let mut f = YuvFrame::filled(w, h, bit_depth, format, 0);
    let (cw, ch) = (f.chroma_width() as usize, f.chroma_height() as usize);
    let luma = (w * h) as usize;
    let chroma = cw * ch;
    let bps = if bit_depth > 8 { 2 } else { 1 };
    assert_eq!(
        raw.len(),
        (luma + 2 * chroma) * bps,
        "raw size is not {w}x{h} {fmt}"
    );
    let widen = |dst: &mut [u16], src: &[u8]| {
        if bps == 2 {
            for (d, c) in dst.iter_mut().zip(src.as_chunks::<2>().0.iter()) {
                *d = u16::from_le_bytes([c[0], c[1]]);
            }
        } else {
            for (d, &s) in dst.iter_mut().zip(src) {
                *d = u16::from(s);
            }
        }
    };
    widen(&mut f.y, &raw[..luma * bps]);
    widen(&mut f.u, &raw[luma * bps..(luma + chroma) * bps]);
    widen(&mut f.v, &raw[(luma + chroma) * bps..]);
    f
}

/// Aggregate the sampled backtraces by their first four crate-level
/// frames and print the heaviest sites (count × stride ≈ allocations,
/// bytes likewise).
fn report_samples(every: usize) {
    if every == 0 {
        return;
    }
    let samples = std::mem::take(&mut *SAMPLES_BT.lock().unwrap_or_else(|e| e.into_inner()));
    let mut by_site: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();
    for (size, bt) in &samples {
        let text = bt.to_string();
        let frames: Vec<&str> = text
            .lines()
            .filter_map(|l| {
                let l = l.trim();
                let (_, name) = l.split_once(": ")?;
                (name.contains("oxideav") && !name.contains("encode_bench")).then_some(name)
            })
            .take(5)
            .collect();
        let key = frames.join(" <- ");
        let e = by_site.entry(key).or_insert((0, 0));
        e.0 += 1;
        e.1 += size;
    }
    let mut rows: Vec<(String, usize, usize)> =
        by_site.into_iter().map(|(k, (n, b))| (k, n, b)).collect();
    rows.sort_by_key(|a| std::cmp::Reverse(a.1));
    println!(
        "allocation sites ({} samples, 1 in {every}): est. allocations / est. MiB / site",
        samples.len()
    );
    for (k, n, b) in rows.iter().take(30) {
        println!(
            "  {:>10} {:>9.1}  {}",
            n * every,
            (b * every) as f64 / (1024.0 * 1024.0),
            k
        );
    }
}

fn main() {
    let a = parse_args();
    if let Some(m) = a.trace_mib {
        TRACE_MIN.store(m << 20, Ordering::Relaxed);
    }
    let raw = std::fs::read(&a.raw).expect("read --raw");
    let frame = frame_of(&raw, a.width, a.height, &a.fmt);
    let held = LIVE.load(Ordering::Relaxed);
    SAMPLE_EVERY.store(a.sample, Ordering::Relaxed);
    let mut opts = StillOptions::new(a.q);
    opts.speed = a.speed;
    opts = opts.threads(a.threads);

    set_stage_probe(Some(Box::new(|label: &'static str| {
        let live = LIVE.load(Ordering::Relaxed);
        let peak = PEAK.swap(live, Ordering::Relaxed);
        let rss = rss_mib();
        let t = RUN_T0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map_or(0.0, |t0| t0.elapsed().as_secs_f64());
        SAMPLES.lock().unwrap_or_else(|e| e.into_inner()).push((
            label,
            mib(live),
            mib(peak),
            rss,
            t,
        ));
        *STAGE.lock().unwrap_or_else(|e| e.into_inner()) = label;
    })));

    let mut best = f64::MAX;
    let mut report: Vec<Sample> = Vec::new();
    let mut peak = 0usize;
    let mut allocs = 0usize;
    let mut result = None;
    for _ in 0..a.repeat.max(1) {
        // The previous run's result would otherwise sit under this
        // run's peak.
        drop(result.take());
        SAMPLES.lock().unwrap_or_else(|e| e.into_inner()).clear();
        ALLOCS.store(0, Ordering::Relaxed);
        PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
        *STAGE.lock().unwrap_or_else(|e| e.into_inner()) = "pre-enter";
        let t0 = Instant::now();
        *RUN_T0.lock().unwrap_or_else(|e| e.into_inner()) = Some(t0);
        let still = encode_still_yuv(&frame, &opts).expect("still encodes");
        let wall = t0.elapsed().as_secs_f64();
        let live = LIVE.load(Ordering::Relaxed);
        let p = PEAK.swap(live, Ordering::Relaxed);
        let mut s = SAMPLES.lock().unwrap_or_else(|e| e.into_inner()).clone();
        s.push(("return", mib(live), mib(p), rss_mib(), wall));
        *STAGE.lock().unwrap_or_else(|e| e.into_inner()) = "harness";
        let run_peak = s.iter().map(|x| x.2).fold(0.0, f64::max);
        if wall < best {
            best = wall;
            report = s;
            peak = (run_peak * 1024.0 * 1024.0) as usize;
            allocs = ALLOCS.load(Ordering::Relaxed);
        }
        result = Some(still);
    }
    set_stage_probe(None);
    SAMPLE_EVERY.store(0, Ordering::Relaxed);
    report_samples(a.sample);
    let still = result.expect("at least one run");
    let bytes = still.temporal_unit_bytes.len();
    let digest = fnv1a(&still.temporal_unit_bytes);
    let py = psnr(&still.recon_y, &frame.y, frame.bit_depth);
    let pu = psnr(&still.recon_u, &frame.u, frame.bit_depth);
    let pv = psnr(&still.recon_v, &frame.v, frame.bit_depth);
    println!(
        "{}x{} {} q={} {:?} threads={}: best {:.3} s, {} bytes, digest {:016x}, PSNR {:.2} / {:.2} / {:.2} dB, {} allocations, harness holds {:.1} MiB, peak live {:.1} MiB ({:.1} MiB above the held source)",
        a.width,
        a.height,
        a.fmt,
        a.q,
        a.speed,
        a.threads,
        best,
        bytes,
        digest,
        py,
        pu,
        pv,
        allocs,
        mib(held),
        mib(peak),
        mib(peak) - mib(held)
    );
    println!(
        "  {:<12} {:>12} {:>14} {:>10} {:>9}   (peak = during the stage above)",
        "stage starts", "live MiB", "peak MiB", "RSS MiB", "at s"
    );
    for (label, live, p, rss, t) in &report {
        println!("  {label:<12} {live:>12.1} {p:>14.1} {rss:>10.1} {t:>9.3}");
    }
    if let Some(path) = a.out {
        std::fs::write(path, &still.temporal_unit_bytes).expect("write --out");
    }
}
