//! Regression: the `dimension-mismatch-49w.bin` fuzz seed pinned the
//! round-42 ffmpeg-oracle disagreement where libavcodec reported
//! `49 × 3` (the spec-mandated `UpscaledWidth × FrameHeight` from
//! §7.18.2 output process) but our decoder published the coded
//! post-superres `FrameWidth = 26`.
//!
//! AV1 spec §5.9.8 superres_params:
//!   `FrameWidth = (UpscaledWidth * SUPERRES_NUM + (SuperresDenom/2)) /
//!    SuperresDenom`
//!
//! AV1 spec §7.18.2 intermediate output preparation:
//!   `w = UpscaledWidth`
//!   `h = FrameHeight`
//!
//! The decoder pipeline operates on the coded `FrameWidth × FrameHeight`
//! plane and we have not implemented the §7.16 upscaling step that
//! converts coded → upscaled. Rather than publish a surface at the
//! wrong dimension we refuse the frame as `Error::Unsupported`,
//! matching the existing valid-but-NYI policy for compound / warp /
//! OBMC / inter-intra. The fuzz oracle's `our_send.is_err()` early
//! return then handles the disagreement without polluting the
//! per-pixel comparison branch.
//!
//! This test pins the refusal contract — if a future change starts
//! silently emitting frames at coded `FrameWidth` again, this test
//! goes red before fuzz CI does.

use oxideav_av1::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, MediaType, Packet, TimeBase};

const SEED: &[u8] =
    include_bytes!("../fuzz/corpus/ffmpeg_oracle_decode/dimension-mismatch-49w.bin");

#[test]
fn superres_frame_returns_unsupported_not_wrong_dim() {
    // The seed is 196 bytes of OBU stream:
    //   OBU#0  SequenceHeader (88 B payload) — profile 2, reduced still
    //          picture, max_frame_width=49, enable_superres=1.
    //   OBU#1  Frame (50 B payload) — KeyFrame, reduced still pic so
    //          frame_size_override_flag=0 ⇒ UpscaledWidth=49, then
    //          superres_params yields SuperresDenom > 8 ⇒ coded
    //          FrameWidth=26.
    // Per §7.18.2 the OUTPUT surface must be 49 × 3, not 26 × 3.
    let mut params = CodecParameters::video(CodecId::new("av1"));
    params.media_type = MediaType::Video;
    let mut dec = Av1Decoder::new(params);
    let pkt = Packet::new(0, TimeBase::new(1, 30), SEED.to_vec());
    let res = dec.send_packet(&pkt);
    let err = res.expect_err(
        "decoder must refuse superres frames with Error::Unsupported \
         per §7.18.2; emitting at coded FrameWidth would publish the \
         wrong dimension",
    );
    let msg = format!("{err:?}");
    assert!(
        msg.contains("superres") || msg.contains("§7.18.2") || msg.contains("§7.16"),
        "expected superres-NYI Unsupported error, got: {msg}"
    );
}
