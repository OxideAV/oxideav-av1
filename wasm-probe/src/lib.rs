//! Bare-WebAssembly run-time probe (#33): one decode and one encode
//! through the public API, folded into a digest that must equal the
//! native one ([`EXPECTED_DIGEST`]).
//!
//! * Decode: `aom_resize_superres.ivf` (a resize + §7.16 superres
//!   stream with mid-stream shape changes) through the registry
//!   decoder under a 4-worker [`ExecutionContext`] budget — the path
//!   an AVIF / HEIF consumer takes; every per-stage timing and every
//!   threaded post-filter arm is on it.
//! * Encode: a textured 4:2:0 64×64 still at `base_q_idx = 80` with a
//!   4-thread search budget, then decoded back.
//!
//! On a target without a clock or threads both must run to completion
//! (serial, zero stage timings) and produce the native bytes.

use oxideav_av1::encoder::still::{StillOptions, encode_still_yuv};
use oxideav_av1::encoder::yuv_frame::{ChromaFormat, YuvFrame};
use oxideav_av1::registry::{CODEC_ID_STR, make_decoder};
use oxideav_core::{CodecId, CodecParameters, ExecutionContext, Frame, Packet, TimeBase};

const RESIZE_SUPERRES_IVF: &[u8] =
    include_bytes!("../../tests/fixtures/stream_params/aom_resize_superres.ivf");

/// The native digest of [`run`].
pub const EXPECTED_DIGEST: u64 = 0xceab_0f8a_f1d0_c795;

/// 64-bit FNV-1a step over `bytes`.
fn fnv(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Decode every frame of `ivf` through the registry decoder under a
/// `threads`-wide budget, folding each pixel plane into `h`. Returns
/// the digest and the frame count.
fn decode_digest(mut h: u64, ivf: &[u8], threads: usize) -> (u64, u32) {
    let params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    let mut dec = make_decoder(&params).expect("decoder");
    dec.set_execution_context(&ExecutionContext::with_threads(threads));
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), ivf.to_vec()))
        .expect("send_packet");
    dec.flush().expect("flush");
    let mut frames = 0;
    while let Ok(frame) = dec.receive_frame() {
        if let Frame::Video(vf) = frame {
            for plane in vf.image_planes() {
                h = fnv(h, &(plane.stride as u64).to_le_bytes());
                h = fnv(h, &plane.data);
            }
            frames += 1;
        }
    }
    (h, frames)
}

/// The probe: decode + encode, every output byte folded into one
/// 64-bit digest.
#[must_use]
pub fn run() -> u64 {
    let (mut h, frames) = decode_digest(0xcbf2_9ce4_8422_2325, RESIZE_SUPERRES_IVF, 4);
    assert!(frames > 0, "the fixture decodes to at least one frame");

    let (w, hgt) = (64u32, 64u32);
    let mut src = YuvFrame::filled(w, hgt, 8, ChromaFormat::Yuv420, 0);
    for y in 0..hgt {
        for x in 0..w {
            src.y[(y * w + x) as usize] = ((x * 3 + y * 5 + (x * y) / 7) & 0xff) as u16;
        }
    }
    for (i, (u, v)) in src.u.iter_mut().zip(src.v.iter_mut()).enumerate() {
        *u = (64 + (i as u16 * 7) % 128) & 0xff;
        *v = (200 - (i as u16 * 3) % 96) & 0xff;
    }
    let enc = encode_still_yuv(&src, &StillOptions::new(80).with_threads(4)).expect("encode");
    h = fnv(h, &enc.ivf_bytes);
    let (h, back) = decode_digest(h, &enc.ivf_bytes, 1);
    assert_eq!(back, 1, "the still decodes to one frame");
    h
}

/// The WebAssembly export: [`run`]'s digest.
#[unsafe(no_mangle)]
pub extern "C" fn oxideav_av1_wasm_probe() -> u64 {
    run()
}

/// The WebAssembly export: [`EXPECTED_DIGEST`].
#[unsafe(no_mangle)]
pub extern "C" fn oxideav_av1_wasm_probe_expected() -> u64 {
    EXPECTED_DIGEST
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_digest_is_pinned() {
        let d = super::run();
        assert_eq!(d, super::EXPECTED_DIGEST, "native digest {d:#018x}");
    }
}
