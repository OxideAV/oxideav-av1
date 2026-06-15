//! `oxideav-core` framework integration: codec registration plus the
//! [`oxideav_core::Decoder`] implementation that bridges the crate's
//! existing intra-only IVF decode driver
//! ([`crate::decode_av1`]) onto the packet-to-frame trait surface.
//!
//! The wrapper does not re-implement any decode logic: each [`Packet`]
//! payload is handed verbatim to [`crate::decode_av1`] (an IVF v0 buffer
//! per the §7.5 temporal-unit grammar the existing driver walks), the
//! recovered [`crate::decoder::Frame`]s are queued, and successive
//! `receive_frame` calls drain that queue one [`VideoFrame`] at a time —
//! returning `Error::NeedMore` while empty and `Error::Eof` once the
//! stream has been flushed and the queue is exhausted.
//!
//! Registration claims the three container identifiers an AV1
//! elementary stream is carried under:
//!
//! * the ISOBMFF / MP4 sample-entry type `av01` (AV1 ISOBMFF Binding
//!   Specification §2.2, `class AV1SampleEntry extends
//!   VisualSampleEntry('av01')`),
//! * the IVF codec FourCC `AV01` (the `DKIF`-magic elementary-stream
//!   wrapper this crate's [`crate::encoder::ivf`] reader / writer round-
//!   trips; see `FOURCC_AV01`),
//! * the Matroska / WebM Codec ID `V_AV1` (the `V_<NAME>` video Codec ID
//!   convention WebM documents for `V_VP8` / `V_VP9`).
//!
//! Because `CodecTag::fourcc` upper-cases alphabetic bytes, the ISOBMFF
//! `av01` and IVF `AV01` sample-entry / FourCC identifiers collapse to a
//! single [`CodecTag::Fourcc`] claim.
//!
//! The registered decoder is intra-only — it covers the same scope as
//! the historical [`crate::decode_av1`] free function (single-tile
//! keyframes, the 4:2:0 / monochrome extents the pixel driver accepts).
//! The surface is therefore partial but reachable through the
//! `RuntimeContext` path; out-of-scope streams surface the same
//! diagnosable [`crate::Error`] the direct API returns.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Error as CoreError, Frame as CoreFrame, Packet, Result as CoreResult, RuntimeContext,
    VideoFrame, VideoPlane,
};

use crate::decoder::Frame as Av1Frame;

/// Canonical codec id. `oxideav-meta::register_all` calls
/// `crate::__oxideav_entry`, which delegates to [`register`].
pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 codec into `reg`.
///
/// Installs the intra-only decoder factory ([`make_decoder`]) and claims
/// the three container identifiers (ISOBMFF `av01` / IVF `AV01` FourCC,
/// Matroska `V_AV1`).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("av1_sw").with_decode();
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .tags([CodecTag::fourcc(b"AV01"), CodecTag::matroska("V_AV1")]),
    );
}

/// Unified entry point invoked by the macro-generated wrapper and by
/// `oxideav-meta::register_all`.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
}

/// Decoder factory — the [`CodecInfo::decoder`] callback.
///
/// The AV1 elementary-stream framing (IVF v0) is self-describing, so no
/// per-stream setup is parsed from [`CodecParameters`] here; each packet
/// carries its own file header. `params.codec_id` is threaded through so
/// [`Decoder::codec_id`] reports the resolved id.
///
/// ## Errors
///
/// Never fails at construction time — an unconfigured / empty stream is
/// represented by an idle decoder that returns `NeedMore` until fed.
pub fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    Ok(Box::new(Av1Decoder {
        codec_id: params.codec_id.clone(),
        queue: std::collections::VecDeque::new(),
        eof: false,
    }))
}

/// Packet-to-frame wrapper driving [`crate::decode_av1`].
struct Av1Decoder {
    codec_id: CodecId,
    /// Frames recovered from already-decoded packets, awaiting drain by
    /// `receive_frame`. Held as `(VideoFrame)` in stream order.
    queue: std::collections::VecDeque<VideoFrame>,
    eof: bool,
}

impl std::fmt::Debug for Av1Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Av1Decoder")
            .field("codec_id", &self.codec_id)
            .field("queued", &self.queue.len())
            .field("eof", &self.eof)
            .finish()
    }
}

impl Decoder for Av1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        // The packet payload is a complete IVF v0 buffer (file header +
        // one or more frame records). Decode it whole and queue every
        // recovered frame for drain by `receive_frame`.
        let frames = crate::decode_av1(&packet.data)
            .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e}")))?;
        for frame in &frames {
            self.queue
                .push_back(av1_frame_to_video_frame(frame, packet.pts));
        }
        Ok(())
    }

    fn receive_frame(&mut self) -> CoreResult<CoreFrame> {
        match self.queue.pop_front() {
            Some(vf) => Ok(CoreFrame::Video(vf)),
            None => {
                if self.eof {
                    Err(CoreError::Eof)
                } else {
                    Err(CoreError::NeedMore)
                }
            }
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }

    fn reset(&mut self) -> CoreResult<()> {
        // No cross-packet carry on the intra-only path: each IVF buffer
        // is self-contained. A seek just drops any undrained frames and
        // clears EOF.
        self.queue.clear();
        self.eof = false;
        Ok(())
    }
}

/// Convert a decoded [`crate::decoder::Frame`] into an `oxideav-core`
/// [`VideoFrame`].
///
/// Each plane is packed into a tight row-major byte buffer (one byte per
/// sample — every decoded extent on the intra-only path is 8-bit). Planes
/// are emitted in plane-major order (luma, then the two chroma planes for
/// the 4:2:0 variants; luma only for the monochrome variant).
fn av1_frame_to_video_frame(frame: &Av1Frame, pts: Option<i64>) -> VideoFrame {
    match frame {
        Av1Frame::Yuv420_16x16 { y, u, v } => {
            let planes = vec![
                plane_from_rows_16x16(y),
                plane_from_rows_chroma(u),
                plane_from_rows_chroma(v),
            ];
            VideoFrame { pts, planes }
        }
        Av1Frame::Yuv420Dyn {
            width,
            height,
            y,
            u,
            v,
        } => {
            let cw = (*width as usize) / 2;
            let planes = vec![
                VideoPlane {
                    stride: *width as usize,
                    data: y.clone(),
                },
                VideoPlane {
                    stride: cw,
                    data: u.clone(),
                },
                VideoPlane {
                    stride: cw,
                    data: v.clone(),
                },
            ];
            let _ = height;
            VideoFrame { pts, planes }
        }
        Av1Frame::YDyn { width, height, y } => {
            let _ = height;
            VideoFrame {
                pts,
                planes: vec![VideoPlane {
                    stride: *width as usize,
                    data: y.clone(),
                }],
            }
        }
    }
}

/// Flatten a fixed 16×16 luma plane (`[[u8; W]; H]`) into a row-major
/// [`VideoPlane`].
fn plane_from_rows_16x16<const W: usize, const H: usize>(rows: &[[u8; W]; H]) -> VideoPlane {
    let mut data = Vec::with_capacity(W * H);
    for row in rows {
        data.extend_from_slice(row);
    }
    VideoPlane { stride: W, data }
}

/// Flatten a fixed chroma plane (`[[u8; W]; H]`) into a row-major
/// [`VideoPlane`].
fn plane_from_rows_chroma<const W: usize, const H: usize>(rows: &[[u8; W]; H]) -> VideoPlane {
    let mut data = Vec::with_capacity(W * H);
    for row in rows {
        data.extend_from_slice(row);
    }
    VideoPlane { stride: W, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::ProbeContext;

    #[test]
    fn register_via_runtime_context_installs_decoder() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        let codec_id = CodecId::new(CODEC_ID_STR);
        assert!(
            ctx.codecs.has_decoder(&codec_id),
            "codec registration should install a decoder factory"
        );
    }

    #[test]
    fn register_claims_container_tags() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);

        // ISOBMFF sample-entry `av01` / IVF FourCC `AV01` collapse to a
        // single upper-cased Fourcc tag.
        let fourcc = CodecTag::fourcc(b"AV01");
        assert_eq!(
            reg.resolve_tag_ref(&ProbeContext::new(&fourcc))
                .map(oxideav_core::CodecId::as_str),
            Some(CODEC_ID_STR),
            "FourCC AV01 / sample-entry av01 must resolve to av1"
        );

        // Matroska / WebM Codec ID.
        let mkv = CodecTag::matroska("V_AV1");
        assert_eq!(
            reg.resolve_tag_ref(&ProbeContext::new(&mkv))
                .map(oxideav_core::CodecId::as_str),
            Some(CODEC_ID_STR),
            "Matroska V_AV1 must resolve to av1"
        );
    }
}
