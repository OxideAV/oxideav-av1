//! Inter (GOP) encode harness (r489): encode a raw planar 8-bit 4:2:0
//! sequence as one KEY + P GOP through
//! [`oxideav_av1::encoder::encode_gop_yuv_with_options`] and report the
//! wall clock, the coded size, the luma PSNR of the encoder's own
//! reconstruction, and whether this crate's decoder reproduces that
//! reconstruction sample for sample.
//!
//! ```text
//! cargo run --release --example gop_bench -- --raw seq.yuv --size 320x240 \
//!     [--frames 8] [--q 120] [--effort fastest|fast|balanced|thorough|best|intra] \
//!     [--threads N] [--tiles CxR] [--repeat N] [--out seq.ivf]
//! ```
//!
//! `--effort intra` codes every frame as its own KEY frame through the
//! still-picture `fast` preset (the all-intra comparison point).
//! `--tiles CxR` sets the §5.9.15 uniform layout (`TileColsLog2` x
//! `TileRowsLog2`); the bounded efforts search its tiles on `--threads`
//! workers and produce the same stream at every budget.

use oxideav_av1::encoder::yuv_frame::{ChromaFormat, YuvFrame};
use oxideav_av1::encoder::{
    GopEffort, GopOptions, StillOptions, encode_gop_yuv_with_options,
    encode_key_frame_yuv_with_options,
};
use std::time::Instant;

fn psnr(a: &[u16], b: &[u16]) -> f64 {
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| {
            let d = f64::from(*x) - f64::from(*y);
            d * d
        })
        .sum();
    if sse == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64 * 255.0 * a.len() as f64 / sse).log10()
}

fn main() {
    let mut raw: Option<String> = None;
    let mut size = (320u32, 240u32);
    let mut frames_n = 8usize;
    let mut q = 120u8;
    let mut effort = String::from("default");
    let mut threads = 1usize;
    let mut repeat = 1usize;
    let mut out: Option<String> = None;
    let mut tiles = (0u32, 0u32);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().expect("option value");
        match a.as_str() {
            "--raw" => raw = Some(val()),
            "--size" => {
                let v = val();
                let (w, h) = v.split_once('x').expect("--size WxH");
                size = (w.parse().expect("width"), h.parse().expect("height"));
            }
            "--frames" => frames_n = val().parse().expect("--frames N"),
            "--q" => q = val().parse().expect("--q 0..=255"),
            "--effort" => effort = val(),
            "--threads" => threads = val().parse().expect("--threads N"),
            "--tiles" => {
                let v = val();
                let (c, r) = v.split_once('x').expect("--tiles COLSLOG2xROWSLOG2");
                tiles = (c.parse().expect("cols log2"), r.parse().expect("rows log2"));
            }
            "--repeat" => repeat = val().parse().expect("--repeat N"),
            "--out" => out = Some(val()),
            other => panic!("unknown argument {other:?}"),
        }
    }
    let raw = raw.expect("--raw <file.yuv> (8-bit 4:2:0 planar)");
    let data = std::fs::read(&raw).expect("read raw");
    let (w, h) = (size.0 as usize, size.1 as usize);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let frame_bytes = w * h + 2 * cw * ch;
    let avail = data.len() / frame_bytes;
    assert!(avail >= 1, "raw file holds no complete frame");
    let n = frames_n.min(avail);
    let frames: Vec<YuvFrame> = (0..n)
        .map(|i| {
            let b = &data[i * frame_bytes..(i + 1) * frame_bytes];
            let wide = |s: &[u8]| s.iter().map(|&v| u16::from(v)).collect::<Vec<u16>>();
            let mut f = YuvFrame::filled(size.0, size.1, 8, ChromaFormat::Yuv420, 0);
            f.y = wide(&b[..w * h]);
            f.u = wide(&b[w * h..w * h + cw * ch]);
            f.v = wide(&b[w * h + cw * ch..]);
            f
        })
        .collect();

    let mut times = Vec::new();
    let mut last: Option<(Vec<u8>, Vec<Vec<u16>>)> = None;
    for _ in 0..repeat.max(1) {
        let t0 = Instant::now();
        let (ivf, recon_y): (Vec<u8>, Vec<Vec<u16>>) = if effort == "intra" {
            let opts = StillOptions::new(q).with_threads(threads);
            let mut bytes = 0usize;
            let mut ys = Vec::new();
            let mut all = Vec::new();
            for f in &frames {
                let k = encode_key_frame_yuv_with_options(f, &opts).expect("key frame");
                bytes += k.temporal_unit_bytes.len();
                all.extend_from_slice(&k.temporal_unit_bytes);
                ys.push(k.recon_y);
            }
            let _ = bytes;
            (all, ys)
        } else {
            let e = match effort.as_str() {
                "fastest" => GopEffort::Fastest,
                "fast" => GopEffort::Fast,
                "balanced" | "default" => GopEffort::Balanced,
                "thorough" => GopEffort::Thorough,
                "best" => GopEffort::Best,
                other => panic!("unknown effort {other:?}"),
            };
            let opts = GopOptions::new(q)
                .with_effort(e)
                .with_threads(threads)
                .with_tiles_log2(tiles.0, tiles.1);
            let g = encode_gop_yuv_with_options(&frames, &opts).expect("gop");
            (g.ivf_bytes, g.recon.into_iter().map(|r| r.y).collect())
        };
        times.push(t0.elapsed().as_secs_f64());
        last = Some((ivf, recon_y));
    }
    let (ivf, recon_y) = last.expect("one run");
    times.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let median = times[times.len() / 2];
    let mean_psnr = frames
        .iter()
        .zip(&recon_y)
        .map(|(f, r)| psnr(&f.y, r).min(99.0))
        .sum::<f64>()
        / n as f64;

    // Decode with this crate's decoder and compare against the
    // encoder's reconstruction (the GOP path only: the intra arm's
    // concatenated temporal units are not an IVF file).
    let mut verdict = String::from("n/a");
    if effort != "intra" {
        let decoded = oxideav_av1::decoder::decode_av1_spec(&ivf).expect("decode");
        let same = decoded.len() == recon_y.len()
            && decoded.iter().zip(&recon_y).all(|(d, r)| {
                d.planes[0].len() == r.len()
                    && d.planes[0]
                        .iter()
                        .zip(r)
                        .all(|(a, b)| u32::from(*a) == u32::from(*b))
            });
        verdict = if same { "bit-exact" } else { "MISMATCH" }.to_string();
    }
    println!(
        "{raw} {}x{} frames={n} q={q} effort={effort} threads={threads}: median {:.3} s over {} run(s), {} bytes, luma PSNR {:.2} dB, decode {verdict}",
        size.0,
        size.1,
        median,
        times.len(),
        ivf.len(),
        mean_psnr
    );
    if let Some(p) = out {
        std::fs::write(p, &ivf).expect("write out");
    }
}
