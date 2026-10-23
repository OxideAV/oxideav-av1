#![no_main]
//! `still` fuzz target — the lossy still-picture encoder and the
//! decode-of-own-output path (r479).
//!
//! The attacker bytes pick the picture shape (`8..=72` per axis, any
//! value — the encoder pads non-multiple-of-8 extents internally), the
//! §6.4.1 (bit depth, chroma format) pairing, `base_q_idx`, the speed
//! preset, the thread budget (which also selects the automatic tile
//! layout) and the sample planes, then drive
//! [`oxideav_av1::encoder::encode_still_yuv`] and feed the temporal
//! unit straight back into [`oxideav_av1::decoder::SpecDecodeSession`].
//! The lossy path runs every election the KEY-frame driver owns — the
//! §7.14 deblocking ladder on its row windows, the §7.15 CDEF unit
//! engine and its banded write-back, the §7.17 restoration search and
//! its stripe-parked apply, the compact committed trees the CDEF /
//! restoration re-emissions expand — so the contract under test is
//! panic-freedom across the whole encode + re-decode pair, plus the
//! decoder's reconstruction equalling the encoder's for every shape.

use libfuzzer_sys::fuzz_target;
use oxideav_av1::decoder::SpecDecodeSession;
use oxideav_av1::encoder::yuv_frame::{ChromaFormat, YuvFrame};
use oxideav_av1::encoder::{StillOptions, StillSpeed, encode_still_yuv};

fuzz_target!(|data: &[u8]| {
    if data.len() < 6 {
        return;
    }
    let width = 8 + u32::from(data[0]) % 65;
    let height = 8 + u32::from(data[1]) % 65;
    let (bit_depth, format) = match data[2] % 8 {
        0 => (8u8, ChromaFormat::Yuv420),
        1 => (8, ChromaFormat::Monochrome),
        2 => (10, ChromaFormat::Yuv420),
        3 => (8, ChromaFormat::Yuv444),
        4 => (10, ChromaFormat::Yuv422),
        5 => (12, ChromaFormat::Yuv444),
        6 => (12, ChromaFormat::Yuv420),
        _ => (8, ChromaFormat::Yuv422),
    };
    let mut opts = StillOptions::new(data[3]);
    opts.speed = match data[4] % 4 {
        0 => StillSpeed::Balanced,
        1 => StillSpeed::Thorough,
        _ => StillSpeed::Fast,
    };
    opts = opts.threads(1 + usize::from(data[5] % 4));
    opts.full_range = data[4] & 4 != 0;
    opts.reduced_header = data[4] & 8 == 0;
    // `Thorough` runs minutes on a large picture; keep it to tiny ones.
    if opts.speed == StillSpeed::Thorough && width * height > 256 {
        opts.speed = StillSpeed::Balanced;
    }
    let body = &data[6..];
    let mut frame = YuvFrame::filled(width, height, bit_depth, format, 0);
    let max = (1u16 << bit_depth) - 1;
    let mut k = 0usize;
    let mut next = || {
        let v = if body.is_empty() {
            0
        } else {
            let b = body[k % body.len()];
            k += 1;
            u16::from(b) * 257
        };
        v % (max + 1)
    };
    for p in frame.y.iter_mut() {
        *p = next();
    }
    for p in frame.u.iter_mut() {
        *p = next();
    }
    for p in frame.v.iter_mut() {
        *p = next();
    }
    let Ok(still) = encode_still_yuv(&frame, &opts) else {
        return;
    };
    let mut session = SpecDecodeSession::new();
    let frames = session
        .decode_temporal_unit(&still.temporal_unit_bytes)
        .expect("the encoder's own temporal unit decodes");
    let f = frames.last().expect("one frame");
    let widen = |p: &[u8]| -> Vec<u16> {
        if bit_depth > 8 {
            p.chunks(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        } else {
            p.iter().map(|&b| u16::from(b)).collect()
        }
    };
    assert_eq!(widen(&f.planes[0]), still.recon_y, "luma reconstruction");
    if f.planes.len() > 1 {
        assert_eq!(widen(&f.planes[1]), still.recon_u, "u reconstruction");
        assert_eq!(widen(&f.planes[2]), still.recon_v, "v reconstruction");
    }
});
