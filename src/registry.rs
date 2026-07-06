//! `oxideav-core` framework integration: codec registration plus the
//! [`oxideav_core::Decoder`] implementation that bridges the crate's
//! spec-faithful frame decoder
//! ([`crate::decoder::SpecDecodeSession`]) onto the packet-to-frame
//! trait surface.
//!
//! The wrapper does not re-implement any decode logic. Each [`Packet`]
//! payload is either
//!
//! * a complete IVF v0 buffer (`DKIF` magic — the elementary-stream
//!   wrapper this crate's [`crate::encoder::ivf`] reader / writer
//!   round-trips): every frame record's payload is fed to the session
//!   in file order, or
//! * one §7.5 temporal-unit body (the low-overhead OBU bytestream a
//!   container demuxer extracts — the Matroska "Block contains one
//!   Temporal Unit" / ISOBMFF sample framing): fed to the session
//!   verbatim.
//!
//! The [`crate::decoder::SpecDecodeSession`] holds the §7.20
//! reference-frame store, the cached sequence header, and the per-slot
//! CDF / motion-field / segment-id state ACROSS packets, so a GOP
//! split one-temporal-unit-per-packet decodes identically to the same
//! bytes in one buffer. Recovered [`SpecFrame`]s are queued, and
//! successive `receive_frame` calls drain that queue one [`VideoFrame`]
//! at a time — returning `Error::NeedMore` while empty and `Error::Eof`
//! once the stream has been flushed and the queue is exhausted.
//!
//! Registration claims the three container identifiers an AV1
//! elementary stream is carried under:
//!
//! * the ISOBMFF / MP4 sample-entry type `av01` (AV1 ISOBMFF Binding
//!   Specification §2.2, `class AV1SampleEntry extends
//!   VisualSampleEntry('av01')`),
//! * the IVF codec FourCC `AV01`,
//! * the Matroska / WebM Codec ID `V_AV1` (the `V_<NAME>` video Codec
//!   ID convention WebM documents for `V_VP8` / `V_VP9`).
//!
//! Because `CodecTag::fourcc` upper-cases alphabetic bytes, the ISOBMFF
//! `av01` and IVF `AV01` sample-entry / FourCC identifiers collapse to a
//! single [`CodecTag::Fourcc`] claim.
//!
//! The registered surface equals [`crate::decoder::decode_av1_spec`]'s:
//! the full conformance-validated decoder — KEY / INTER GOPs with the
//! cross-frame session state, `show_existing_frame`, segmentation,
//! quantizer matrices, compound / OBMC / warped motion, film grain,
//! superres, 8/10/12-bit output. Out-of-scope streams surface the same
//! diagnosable [`crate::Error`] the direct API returns.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, CodecTag, Decoder,
    Error as CoreError, Frame as CoreFrame, Packet, Result as CoreResult, RuntimeContext,
    VideoFrame, VideoPlane,
};

use crate::decoder::{SpecDecodeSession, SpecFrame};

/// Canonical codec id. `oxideav-meta::register_all` calls
/// `crate::__oxideav_entry`, which delegates to [`register`].
pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 codec into `reg`.
///
/// Installs the spec-driver decoder factory ([`make_decoder`]) and
/// claims the three container identifiers (ISOBMFF `av01` / IVF `AV01`
/// FourCC, Matroska `V_AV1`).
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
/// The AV1 elementary-stream framing is self-describing (the sequence
/// header OBU arrives in-band), so no per-stream setup is parsed from
/// [`CodecParameters`] here. `params.codec_id` is threaded through so
/// [`Decoder::codec_id`] reports the resolved id.
///
/// ## Errors
///
/// Never fails at construction time — an unconfigured / empty stream is
/// represented by an idle decoder that returns `NeedMore` until fed.
pub fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    Ok(Box::new(Av1Decoder {
        codec_id: params.codec_id.clone(),
        session: SpecDecodeSession::new(),
        queue: std::collections::VecDeque::new(),
        eof: false,
    }))
}

/// Packet-to-frame wrapper driving [`SpecDecodeSession`].
struct Av1Decoder {
    codec_id: CodecId,
    /// The cross-packet §7.20 session-state stack.
    session: SpecDecodeSession,
    /// Frames recovered from already-decoded packets, awaiting drain by
    /// `receive_frame`. Held as `VideoFrame` in output order.
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

/// The IVF file-header magic (`DKIF`) — distinguishes a whole-file IVF
/// packet from a raw temporal-unit packet.
const IVF_MAGIC: &[u8; 4] = b"DKIF";

impl Decoder for Av1Decoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> CoreResult<()> {
        let mut push = |frames: Vec<SpecFrame>| {
            for frame in &frames {
                self.queue
                    .push_back(spec_frame_to_video_frame(frame, packet.pts));
            }
        };
        if packet.data.len() >= 4 && &packet.data[..4] == IVF_MAGIC {
            // Whole IVF v0 buffer: walk the frame records in file
            // order through the persistent session (identical to
            // `decode_av1_spec`, but the reference state carries into
            // subsequent packets).
            let reader = crate::encoder::ivf::IvfReader::new(&packet.data)
                .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e:?}")))?;
            let records = reader
                .read_all()
                .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e:?}")))?;
            for record in records {
                let frames = self
                    .session
                    .decode_temporal_unit(&record.payload)
                    .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e}")))?;
                push(frames);
            }
        } else {
            // One §7.5 temporal-unit body per packet (the container
            // demuxer framing).
            let frames = self
                .session
                .decode_temporal_unit(&packet.data)
                .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e}")))?;
            push(frames);
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
        // A seek discontinuity: drop undrained frames and the §7.20
        // reference store (the landing point must be a KEY frame,
        // which rebuilds every slot); keep the cached sequence header
        // (containers need not repeat it mid-stream).
        self.queue.clear();
        self.session.reset_references();
        self.eof = false;
        Ok(())
    }
}

/// Convert a decoded [`SpecFrame`] into an `oxideav-core`
/// [`VideoFrame`].
///
/// Planes are emitted in the decoder's plane-major order (luma, then
/// chroma when present). `SpecFrame::planes` already holds tight
/// row-major bytes — one byte per sample at 8-bit, packed little-endian
/// `u16` at 10/12-bit — so each plane moves verbatim with its byte
/// stride (`width` samples × 1 or 2 bytes).
fn spec_frame_to_video_frame(frame: &SpecFrame, pts: Option<i64>) -> VideoFrame {
    let bytes_per_sample: usize = if frame.bit_depth > 8 { 2 } else { 1 };
    let planes = frame
        .planes
        .iter()
        .zip(frame.plane_dims.iter())
        .map(|(data, &(w, _h))| VideoPlane {
            stride: (w as usize) * bytes_per_sample,
            data: data.clone(),
        })
        .collect();
    VideoFrame { pts, planes }
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
