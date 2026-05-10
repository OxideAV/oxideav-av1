#![no_main]

//! Fuzz: encoder→decoder self-roundtrip on a single 8-bit 4:2:0
//! keyframe sized within the round-1 envelope (≤ 64×64, multiples of
//! 8). The encoder currently emits a fixed DC_PRED skip=1 stream
//! regardless of pixel input (round-3 deliverable), so this harness
//! primarily fuzzes the OBU framing + sequence/frame header
//! permutations driven by `(width, height, base_q_idx)`.
//!
//! Asserts: every packet our encoder produces decodes back through
//! `Av1Decoder` to a `Frame::Video` with the declared plane sizes,
//! and the decoder never panics on any of those bitstreams.

use libfuzzer_sys::fuzz_target;
use oxideav_av1::encoder::{write_keyframe_stream, FrameConfig, SequenceConfig};
use oxideav_av1::{Av1Decoder, CODEC_ID_STR};
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, MediaType, Packet, TimeBase};

fuzz_target!(|data: &[u8]| {
    let Some((w, h, base_q_idx)) = parse_shape(data) else {
        return;
    };

    let seq = SequenceConfig {
        width: w,
        height: h,
    };
    let frame = FrameConfig { base_q_idx };
    let bytes = write_keyframe_stream(&seq, &frame);

    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.media_type = MediaType::Video;
    let mut dec = Av1Decoder::new(params);

    let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
    if dec.send_packet(&pkt).is_err() {
        // Encoder output should always parse — any error is a
        // self-roundtrip break.
        panic!("encoder output rejected by decoder for {w}x{h} q={base_q_idx}");
    }
    let out = match dec.receive_frame() {
        Ok(f) => f,
        Err(e) => panic!("decoder yielded no frame for {w}x{h}: {e:?}"),
    };
    let Frame::Video(vf) = out else {
        panic!("expected Frame::Video");
    };
    assert_eq!(vf.planes.len(), 3, "Yuv420P → 3 planes");
    let y_len = (w as usize) * (h as usize);
    let c_len = ((w / 2) as usize) * ((h / 2) as usize);
    assert_eq!(vf.planes[0].data.len(), y_len, "Y plane length");
    assert_eq!(vf.planes[1].data.len(), c_len, "U plane length");
    assert_eq!(vf.planes[2].data.len(), c_len, "V plane length");
});

/// Carve fuzz bytes into `(width, height, base_q_idx)`. Width and
/// height are forced to a multiple of 8 in the [8, 64] range (round-1
/// encoder envelope).
fn parse_shape(data: &[u8]) -> Option<(u32, u32, u8)> {
    if data.len() < 3 {
        return None;
    }
    // 8..=64 in steps of 8 → 8 bins; map into 1..=8.
    let w = ((data[0] % 8) as u32 + 1) * 8;
    let h = ((data[1] % 8) as u32 + 1) * 8;
    let q = data[2];
    Some((w, h, q))
}
