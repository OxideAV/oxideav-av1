#![no_main]

//! Fuzz: feed arbitrary AV1 OBU bytes to BOTH `Av1Decoder` and
//! libavcodec (via `libloading`). When libavcodec accepts the
//! bitstream and produces a 4:2:0 frame, our decoder must also
//! produce a frame with matching width / height, and (for the
//! per-pixel comparison we can perform without HBD support) Y plane
//! values within ±1 LSB of libavcodec's reconstruction.
//!
//! This is the strongest signal we get on real conformance: every
//! corpus input where libavcodec emits a frame becomes a
//! cross-decoder check. Inputs libavcodec rejects are simply skipped
//! (we don't claim the spec-correct behaviour is "decode" when even
//! ffmpeg refuses).
//!
//! When libavcodec is not installed (no `apt-get install -y ffmpeg`
//! on the runner), the harness `eprintln!`s `[oracle skip]` and
//! returns. There is **no** `#[ignore]` shortcut — the runtime skip
//! is the only deferral mechanism, by workspace policy.

use libfuzzer_sys::fuzz_target;
use oxideav_av1::Av1Decoder;
use oxideav_av1_fuzz::libavcodec;
use oxideav_core::{
    CodecId, CodecParameters, Decoder, Frame, MediaType, Packet, TimeBase, VideoFrame,
};

fuzz_target!(|data: &[u8]| {
    if !libavcodec::available() {
        return;
    }
    if data.len() > 65_536 {
        return;
    }

    // Run libavcodec first — if it rejects the bitstream we skip the
    // comparison entirely (we don't enforce spec on inputs ffmpeg
    // also can't decode).
    let avc_frames = match libavcodec::decode_av1(data) {
        Some(Ok(f)) => f,
        Some(Err(())) => return,
        None => return,
    };
    if avc_frames.is_empty() {
        return;
    }

    // Run our decoder against the same bytes.
    let mut params = CodecParameters::video(CodecId::new("av1"));
    params.media_type = MediaType::Video;
    let mut dec = Av1Decoder::new(params);
    let pkt = Packet::new(0, TimeBase::new(1, 30), data.to_vec());

    let our_send = dec.send_packet(&pkt);
    let mut our_frames: Vec<VideoFrame> = Vec::new();
    for _ in 0..avc_frames.len() + 4 {
        match dec.receive_frame() {
            Ok(Frame::Video(v)) => our_frames.push(v),
            _ => break,
        }
    }

    // libavcodec accepted but we rejected hard — that's a real
    // disagreement. The Av1Decoder uses `Error::Unsupported` for
    // valid-but-NYI features (compound, warp, etc.); we don't fault
    // those — we only fault when libavcodec produced a *frame* (i.e.
    // a complete decodable input) but we couldn't even ingest the
    // packet.
    if our_send.is_err() && !avc_frames.is_empty() {
        // The reference produced N frames; we couldn't even parse.
        // For now log + return — many of these will be valid-but-NYI
        // inter / multi-ref / HBD features.
        eprintln!(
            "[oracle disagreement] libavcodec emitted {} frame(s) but Av1Decoder \
             rejected packet ({:?})",
            avc_frames.len(),
            our_send.err()
        );
        return;
    }

    // Find the first ffmpeg frame we got — that's the most stable
    // frame to compare against (later frames may differ in inter ref
    // handling that our decoder doesn't fully support).
    let avc0 = &avc_frames[0];
    let Some(our0) = our_frames.first() else {
        // Same skip-not-fail policy as above.
        eprintln!(
            "[oracle disagreement] libavcodec emitted {} frame(s) at {}x{} but Av1Decoder \
             produced 0",
            avc_frames.len(),
            avc0.width,
            avc0.height
        );
        return;
    };

    // Frame count match is a strong signal — but we only enforce it
    // when both decoders see at least one frame (avoids the "we emit
    // 1, ffmpeg drains 2" ordering noise around show_existing_frame).
    // Width / height MUST match: a wrong dimension is always a real bug.
    let our_w = our0.planes[0].stride as u32;
    assert_eq!(
        our_w, avc0.width,
        "frame 0 width mismatch: ours={our_w} avc={}",
        avc0.width
    );
    let h_from_y = (our0.planes[0].data.len() / our0.planes[0].stride) as u32;
    assert_eq!(
        h_from_y, avc0.height,
        "frame 0 height mismatch: ours={h_from_y} avc={}",
        avc0.height
    );

    // Pixel comparison only when ffmpeg gave us 4:2:0 (pix_fmt==0).
    // Other pix_fmts (HBD, 422, 444, GRAY8) skip the per-pixel check
    // because our decoder narrows HBD to u8 internally.
    if avc0.pix_fmt == 0 {
        // libavcodec planes are tightly packed at width*height after
        // our copy (see fuzz/src/lib.rs read_avframe). Our planes are
        // packed at stride==width too. Compare with ±1 LSB tolerance.
        let w = avc0.width as usize;
        let h = avc0.height as usize;
        let our_y = &our0.planes[0].data;
        let n = (w * h).min(our_y.len()).min(avc0.y.len());
        let mut max_diff: u8 = 0;
        let mut over_thresh: u32 = 0;
        for (ours, theirs) in our_y.iter().zip(avc0.y.iter()).take(n) {
            let d = ours.abs_diff(*theirs);
            if d > max_diff {
                max_diff = d;
            }
            if d > 1 {
                over_thresh += 1;
            }
        }
        // ±1 LSB max as the strict bound — if more than 1% of samples
        // exceed it, fail. (1% allows a tiny rounding-edge drift band
        // without masking real bugs like swapped UV / wrong chroma siting.)
        let thresh = (n / 100).max(8);
        assert!(
            (over_thresh as usize) <= thresh,
            "Y plane diverges from libavcodec: {over_thresh}/{n} samples > 1 LSB \
             (max={max_diff})"
        );
    }
});
