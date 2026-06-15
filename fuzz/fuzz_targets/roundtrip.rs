#![no_main]
//! `roundtrip` fuzz target — encode then decode-of-own-output.
//!
//! Splits the attacker bytes into width / height selectors plus a YUV
//! 4:2:0 plane blob, drives them through [`oxideav_av1::encode_av1`]
//! (the §5.9.2 CodedLossless intra arm), and feeds any resulting IVF
//! bytes straight back into [`oxideav_av1::decode_av1`].
//!
//! This stresses three surfaces with attacker-chosen dimensions:
//!
//! * The encoder's dimension / plane-length validation
//!   (`Yuv420Frame::validate`) — width / height in `[8, 64]`, both
//!   multiples of 8, exact `Y || U || V` plane length.
//! * The intra partition-tree + coefficient writer.
//! * The decode-of-own-output path (`decode_av1` on the encoder's IVF).
//!
//! The contract under test: panic-freedom across the encode + re-decode
//! pair. Both the path where `encode_av1` rejects the dimensions and the
//! path where it produces a stream that `decode_av1` consumes must be
//! reached without a panic.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        // Still exercise the short-buffer rejection path.
        let _ = oxideav_av1::encode_av1(data, 8, 8);
        return;
    }

    // Derive an in-range, multiples-of-8 dimension from the first two
    // bytes so the encode path is actually reached on a meaningful
    // fraction of inputs (rather than always bailing on a length
    // mismatch). Range [8, 64] step 8 ⇒ 8 choices per axis.
    let pick = |b: u8| -> u32 { 8 + ((b as u32) % 8) * 8 };
    let width = pick(data[0]);
    let height = pick(data[1]);

    let body = &data[2..];

    let chroma_w = (width / 2) as usize;
    let chroma_h = (height / 2) as usize;
    let y_size = (width as usize) * (height as usize);
    let uv_size = chroma_w * chroma_h;
    let expected = y_size + 2 * uv_size;

    // Build a plane buffer of exactly the length `encode_av1` requires,
    // filled from the attacker body (repeated / truncated as needed).
    let mut pixels = vec![0u8; expected];
    if !body.is_empty() {
        for (i, p) in pixels.iter_mut().enumerate() {
            *p = body[i % body.len()];
        }
    }

    // Also feed the raw (possibly wrong-length) body once so the
    // length-mismatch rejection arm is covered.
    let _ = oxideav_av1::encode_av1(body, width, height);

    if let Ok(ivf) = oxideav_av1::encode_av1(&pixels, width, height) {
        // Decode-of-own-output: must not panic.
        let _ = oxideav_av1::decode_av1(&ivf);
    }
});
