//! r450 — env-gated EXTERNAL-STREAM sweep harness (inert in CI).
//!
//! Points `OXIDEAV_AV1_EXT_SWEEP_DIR` at a directory of
//! `<name>.ivf` + `<name>.yuv` pairs, where the YUV is the expected
//! decode-order output (all shown frames' planes concatenated; one
//! byte per sample at 8-bit, little-endian pairs at 10/12-bit) as
//! produced by independent black-box reference decoders. Every pair
//! is decoded through the public [`oxideav_av1::decode_av1`] entry
//! and compared byte-for-byte; mismatches report the first
//! diverging frame/plane instead of a blind byte count.
//!
//! This is the round workflow behind the corpus pins in
//! `fixture_conformance.rs`: acquire streams exercising untested
//! tool combinations from an external encoder (black-box CLI only),
//! validate the expected output across several independent
//! reference decoders, sweep them through this harness, fix any
//! divergence spec-first, then pin the stream by hex + digest.

use oxideav_av1::decoder::Frame;

#[test]
fn external_stream_sweep() {
    let Ok(dir) = std::env::var("OXIDEAV_AV1_EXT_SWEEP_DIR") else {
        eprintln!("OXIDEAV_AV1_EXT_SWEEP_DIR unset — skipping the external-stream sweep");
        return;
    };
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .expect("sweep dir readable")
        .filter_map(|e| {
            let p = e.expect("dir entry").path();
            (p.extension().is_some_and(|x| x == "ivf"))
                .then(|| p.file_stem().unwrap().to_string_lossy().into_owned())
        })
        .collect();
    names.sort();
    assert!(!names.is_empty(), "no .ivf files under {dir}");
    let mut failures = Vec::new();
    for name in &names {
        let ivf = std::fs::read(format!("{dir}/{name}.ivf")).expect("ivf readable");
        let expected = std::fs::read(format!("{dir}/{name}.yuv")).expect("yuv readable");
        let frames = match oxideav_av1::decode_av1(&ivf) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("{name}: DECODE ERROR {e:?}");
                failures.push(name.clone());
                continue;
            }
        };
        let mut got: Vec<u8> = Vec::new();
        let mut layout: Vec<(usize, usize)> = Vec::new(); // (frame, plane) per span end
        for (fi, f) in frames.iter().enumerate() {
            let Frame::Spec(s) = f else {
                panic!("{name}: non-spec frame surface");
            };
            for (pi, p) in s.planes.iter().enumerate() {
                got.extend_from_slice(p);
                layout.push((fi, pi));
            }
        }
        if got == expected {
            eprintln!(
                "{name}: MATCH ({} frames, {} bytes)",
                frames.len(),
                got.len()
            );
            continue;
        }
        if got.len() != expected.len() {
            eprintln!(
                "{name}: LENGTH MISMATCH got {} expected {} ({} frames)",
                got.len(),
                expected.len(),
                frames.len()
            );
        } else {
            let first = got.iter().zip(&expected).position(|(a, b)| a != b).unwrap();
            let n = got.iter().zip(&expected).filter(|(a, b)| a != b).count();
            // Locate the span holding the first diff.
            let mut acc = 0usize;
            let mut where_ = (0usize, 0usize);
            for (span_i, f) in frames.iter().enumerate() {
                let Frame::Spec(s) = f else { unreachable!() };
                let mut hit = false;
                for (pi, p) in s.planes.iter().enumerate() {
                    if first < acc + p.len() {
                        where_ = (span_i, pi);
                        hit = true;
                        break;
                    }
                    acc += p.len();
                }
                if hit {
                    break;
                }
            }
            eprintln!(
                "{name}: {n} of {} bytes differ; first at byte {first} (frame {}, plane {})",
                got.len(),
                where_.0,
                where_.1
            );
        }
        failures.push(name.clone());
    }
    assert!(failures.is_empty(), "external sweep failures: {failures:?}");
}
