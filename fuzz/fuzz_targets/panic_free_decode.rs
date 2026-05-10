#![no_main]

//! Fuzz: arbitrary bytes → `Av1Decoder::send_packet` returns a `Result`,
//! never panics.
//!
//! Pinned regression: commit `eea1a0a` (released as 0.1.7) fixed a
//! `subtract-with-overflow` panic in `decode::coeffs::read_golomb`
//! where a length-32 unary prefix would shift `1u32 << 32` and wrap
//! `x` to 0, then `x - 1` aborted. The corpus seed
//! `corpus/panic_free_decode/golomb-overflow.bin` carries the original
//! 6-byte input that surfaced the bug from oxideav-avif fuzz CI run
//! 25623885786 (`ff 0a 0a ff ff 22`); a regression would re-fail this
//! harness immediately.

use libfuzzer_sys::fuzz_target;
use oxideav_av1::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, MediaType, Packet, TimeBase};

fuzz_target!(|data: &[u8]| {
    // Cap input size to keep iterations fast — the golomb regression
    // was 6 bytes; the largest pathological payloads we've seen in
    // sibling fuzz runs are well under 64 KiB.
    if data.len() > 65_536 {
        return;
    }

    let mut params = CodecParameters::video(CodecId::new("av1"));
    params.media_type = MediaType::Video;
    let mut dec = Av1Decoder::new(params);

    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());
    // We don't care whether the decoder accepts the input — only that
    // every code path returns a `Result` instead of unwinding. The
    // assertion is the absence of a panic.
    let _ = dec.send_packet(&pkt);
    // Drain any frames that emerged (also panic-free).
    for _ in 0..16 {
        if dec.receive_frame().is_err() {
            break;
        }
    }
});
