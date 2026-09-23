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
    Encoder, Error as CoreError, Frame as CoreFrame, Packet, PixelFormat, Result as CoreResult,
    RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};

use crate::codec_config::Av1CodecConfig;
use crate::decoder::{SpecDecodeSession, SpecFrame};
use crate::encoder::{
    encode_key_frame_yuv_with_q, encode_still_yuv, quality_to_base_q_idx, ChromaFormat,
    StillOptions, StillSpeed, YuvFrame, DEFAULT_STILL_BASE_Q_IDX,
};

/// Canonical codec id. `oxideav-meta::register_all` calls
/// `crate::__oxideav_entry`, which delegates to [`register`].
pub const CODEC_ID_STR: &str = "av1";

/// Register the AV1 codec into `reg`.
///
/// Installs the spec-driver decoder factory ([`make_decoder`]) and
/// claims the three container identifiers (ISOBMFF `av01` / IVF `AV01`
/// FourCC, Matroska `V_AV1`).
pub fn register_codecs(reg: &mut CodecRegistry) {
    let caps = CodecCapabilities::video("av1_sw")
        .with_decode()
        .with_encode();
    reg.register(
        CodecInfo::new(CodecId::new(CODEC_ID_STR))
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder)
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
/// header OBU normally arrives in-band), so a stream needs no setup
/// from [`CodecParameters`]. r460 — when `params.extradata` carries
/// the container's codec configuration it is honoured too: either an
/// `av1C` record ([`crate::codec_config::Av1CodecConfig`] — the AVIF /
/// HEIF `av01` item property or the ISOBMFF `av01` sample-entry box
/// payload, whose `configOBUs` tail may hold a Sequence Header OBU) or
/// a bare OBU sequence (a Sequence Header OBU, optionally preceded by
/// a temporal delimiter). Those OBUs are fed to the session before the
/// first packet, so an item / sample whose payload omits the sequence
/// header still decodes; a payload that repeats it (the av1-avif §2.1
/// "exactly one Sequence Header OBU" shape) simply re-caches it.
/// `params.codec_id` is threaded through so [`Decoder::codec_id`]
/// reports the resolved id.
///
/// ## Errors
///
/// Construction fails only when the extradata's configuration OBUs
/// are malformed (`Error::invalid`); an empty / unconfigured stream is
/// represented by an idle decoder that returns `NeedMore` until fed.
pub fn make_decoder(params: &CodecParameters) -> CoreResult<Box<dyn Decoder>> {
    let mut session = SpecDecodeSession::new();
    let config_obus = codec_config_obus(&params.extradata);
    if !config_obus.is_empty() {
        session
            .decode_temporal_unit(config_obus)
            .map_err(|e| CoreError::invalid(format!("oxideav-av1: extradata: {e}")))?;
    }
    Ok(Box::new(Av1Decoder {
        codec_id: params.codec_id.clone(),
        session,
        queue: std::collections::VecDeque::new(),
        eof: false,
    }))
}

/// The configuration OBUs an extradata blob carries: the `configOBUs`
/// tail of an `av1C` record, or the blob itself when it is a bare OBU
/// sequence starting with a Sequence Header / temporal delimiter OBU
/// (§5.3.1: forbidden bit clear, `obu_type` 1 or 2). Anything else
/// yields an empty slice (ignored).
fn codec_config_obus(extradata: &[u8]) -> &[u8] {
    if let Some(cfg) = crate::codec_config::Av1CodecConfig::parse(extradata) {
        let n = cfg.config_obus.len();
        return &extradata[extradata.len() - n..];
    }
    match extradata.first() {
        Some(&b) if b & 0x80 == 0 && matches!((b >> 3) & 0xF, 1 | 2) => extradata,
        _ => &[],
    }
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

// ───────────────────────── Encoder ─────────────────────────

/// The `CodecOptions` schema of the framework encoder (r460) — the
/// string-bag twin of [`StillOptions`].
///
/// | key              | kind                              | default                      |
/// |------------------|-----------------------------------|------------------------------|
/// | `still`          | `true` / `false`                  | `false`                      |
/// | `q`              | `base_q_idx` 0..=255 (0 lossless) | `DEFAULT_STILL_BASE_Q_IDX`   |
/// | `quality`        | 0..=100 (100 lossless); overrides `q` | —                        |
/// | `lossless`       | `true` forces `base_q_idx = 0`    | `false`                      |
/// | `speed`          | `fast` / `balanced` / `thorough`  | `balanced`                   |
/// | `tile_cols_log2` | §5.9.15 `TileColsLog2`            | `0`                          |
/// | `tile_rows_log2` | §5.9.15 `TileRowsLog2`            | `0`                          |
/// | `full_range`     | `true` / `false` (§5.5.2 `color_range`) | from the pixel format (`YuvJ*` full) |
///
/// `still = true` codes every frame as an independent still picture
/// (`still_picture = 1` + `reduced_still_picture_header = 1`, one
/// temporal unit per packet — the AVIF / HEIF `av01` item payload);
/// `still = false` codes every frame as a KEY frame under a full
/// sequence header repeated per temporal unit (an all-intra AV1
/// video every packet of which is a sync sample).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1EncoderOptions {
    /// See the schema table.
    pub still: bool,
    /// The resolved `base_q_idx`.
    pub base_q_idx: u8,
    /// Search effort.
    pub speed: StillSpeed,
    /// §5.9.15 `TileColsLog2`.
    pub tile_cols_log2: u32,
    /// §5.9.15 `TileRowsLog2`.
    pub tile_rows_log2: u32,
    /// §5.5.2 `color_range`.
    pub full_range: bool,
}

impl Av1EncoderOptions {
    /// Parse the option bag (see the schema table). `full_range`
    /// defaults from `pixel_format` (`YuvJ*` = full range).
    ///
    /// ## Errors
    ///
    /// `Error::invalid` on an unparsable value or unknown key.
    pub fn from_params(params: &CodecParameters) -> CoreResult<Self> {
        let mut o = Self {
            still: false,
            base_q_idx: DEFAULT_STILL_BASE_Q_IDX,
            speed: StillSpeed::Balanced,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            full_range: matches!(
                params.pixel_format,
                Some(PixelFormat::YuvJ420P | PixelFormat::YuvJ422P | PixelFormat::YuvJ444P)
            ),
        };
        let bad = |k: &str, v: &str| CoreError::invalid(format!("oxideav-av1: option {k}={v:?}"));
        let parse_bool = |k: &str, v: &str| -> CoreResult<bool> {
            match v {
                "true" | "1" | "yes" => Ok(true),
                "false" | "0" | "no" => Ok(false),
                _ => Err(bad(k, v)),
            }
        };
        let mut quality: Option<u8> = None;
        let mut lossless = false;
        for (k, v) in params.options.iter() {
            match k {
                "still" => o.still = parse_bool(k, v)?,
                "q" | "base_q_idx" => o.base_q_idx = v.parse().map_err(|_| bad(k, v))?,
                "quality" => {
                    let q: u8 = v.parse().map_err(|_| bad(k, v))?;
                    if q > 100 {
                        return Err(bad(k, v));
                    }
                    quality = Some(q);
                }
                "lossless" => lossless = parse_bool(k, v)?,
                "speed" => {
                    o.speed = match v {
                        "fast" => StillSpeed::Fast,
                        "balanced" => StillSpeed::Balanced,
                        "thorough" => StillSpeed::Thorough,
                        _ => return Err(bad(k, v)),
                    }
                }
                "tile_cols_log2" => o.tile_cols_log2 = v.parse().map_err(|_| bad(k, v))?,
                "tile_rows_log2" => o.tile_rows_log2 = v.parse().map_err(|_| bad(k, v))?,
                "full_range" => o.full_range = parse_bool(k, v)?,
                _ => {
                    return Err(CoreError::invalid(format!(
                        "oxideav-av1: unknown option {k:?}"
                    )))
                }
            }
        }
        if let Some(q) = quality {
            o.base_q_idx = quality_to_base_q_idx(q);
        }
        if lossless {
            o.base_q_idx = 0;
        }
        Ok(o)
    }

    fn still_options(&self) -> StillOptions {
        StillOptions {
            base_q_idx: self.base_q_idx,
            tile_cols_log2: self.tile_cols_log2,
            tile_rows_log2: self.tile_rows_log2,
            speed: self.speed,
            full_range: self.full_range,
            reduced_header: true,
        }
    }
}

/// Map a framework pixel format onto the encoder's `(bit depth,
/// chroma format)` pairing.
fn pixel_format_layout(pf: PixelFormat) -> Option<(u8, ChromaFormat)> {
    Some(match pf {
        PixelFormat::Yuv420P | PixelFormat::YuvJ420P => (8, ChromaFormat::Yuv420),
        PixelFormat::Yuv422P | PixelFormat::YuvJ422P => (8, ChromaFormat::Yuv422),
        PixelFormat::Yuv444P | PixelFormat::YuvJ444P => (8, ChromaFormat::Yuv444),
        PixelFormat::Gray8 => (8, ChromaFormat::Monochrome),
        PixelFormat::Yuv420P10Le => (10, ChromaFormat::Yuv420),
        PixelFormat::Yuv422P10Le => (10, ChromaFormat::Yuv422),
        PixelFormat::Yuv444P10Le => (10, ChromaFormat::Yuv444),
        PixelFormat::Gray10Le => (10, ChromaFormat::Monochrome),
        PixelFormat::Yuv420P12Le => (12, ChromaFormat::Yuv420),
        PixelFormat::Yuv422P12Le => (12, ChromaFormat::Yuv422),
        PixelFormat::Yuv444P12Le => (12, ChromaFormat::Yuv444),
        PixelFormat::Gray12Le => (12, ChromaFormat::Monochrome),
        _ => return None,
    })
}

/// Encoder factory — the [`CodecInfo::encoder`] callback (r460).
///
/// Needs `width`, `height` and a planar YUV / gray `pixel_format`
/// (8-bit 4:2:0 / 4:2:2 / 4:4:4 / gray, the `YuvJ*` full-range
/// variants, and the 10 / 12-bit little-endian siblings incl.
/// `Gray10Le` / `Gray12Le`). Options per [`Av1EncoderOptions`].
/// [`Encoder::output_params`] carries the `av1C` record bytes in
/// `extradata` (no `configOBUs` — av1-avif §2.2.1), so a container
/// writer fills its codec-configuration property directly.
///
/// ## Errors
///
/// `Error::invalid` on missing / unsupported geometry or format and on
/// bad options.
pub fn make_encoder(params: &CodecParameters) -> CoreResult<Box<dyn Encoder>> {
    let width = params
        .width
        .ok_or_else(|| CoreError::invalid("oxideav-av1: encoder needs width"))?;
    let height = params
        .height
        .ok_or_else(|| CoreError::invalid("oxideav-av1: encoder needs height"))?;
    let pf = params
        .pixel_format
        .ok_or_else(|| CoreError::invalid("oxideav-av1: encoder needs pixel_format"))?;
    let (bit_depth, format) = pixel_format_layout(pf).ok_or_else(|| {
        CoreError::invalid(format!("oxideav-av1: unsupported pixel format {pf:?}"))
    })?;
    let opts = Av1EncoderOptions::from_params(params)?;
    // Probe the geometry the same way the encoder will (the r410 shape
    // rules) so a bad size fails at construction, not at the first
    // frame.
    YuvFrame::filled(width, height, bit_depth, format, 0)
        .validate()
        .map_err(|e| CoreError::invalid(format!("oxideav-av1: unsupported geometry: {e}")))?;
    // The av1C fields follow from the pairing + size alone (the tool
    // gates the encoder opens do not appear in the record).
    let mut seq = crate::encoder::build_intra_only_seq_yuv(width, height, bit_depth, format)
        .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e}")))?;
    if opts.still {
        crate::encoder::still::apply_still_picture_shape(&mut seq);
    }
    seq.color_config.color_range = opts.full_range;
    let mut out = params.clone();
    out.codec_id = CodecId::new(CODEC_ID_STR);
    out.extradata = Av1CodecConfig::from_sequence_header(&seq).to_bytes();
    let time_base = match params.frame_rate {
        Some(r) if r.num > 0 && r.den > 0 => TimeBase::new(r.den, r.num),
        _ => TimeBase::new(1, 25),
    };
    Ok(Box::new(Av1Encoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        params: out,
        opts,
        width,
        height,
        bit_depth,
        format,
        time_base,
        next_pts: 0,
        queue: std::collections::VecDeque::new(),
        eof: false,
    }))
}

/// Frame-to-packet wrapper around [`encode_still_yuv`] /
/// [`encode_key_frame_yuv_with_q`].
struct Av1Encoder {
    codec_id: CodecId,
    params: CodecParameters,
    opts: Av1EncoderOptions,
    width: u32,
    height: u32,
    bit_depth: u8,
    format: ChromaFormat,
    time_base: TimeBase,
    next_pts: i64,
    queue: std::collections::VecDeque<Packet>,
    eof: bool,
}

impl std::fmt::Debug for Av1Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Av1Encoder")
            .field("opts", &self.opts)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bit_depth", &self.bit_depth)
            .field("format", &self.format)
            .field("queued", &self.queue.len())
            .finish()
    }
}

impl Av1Encoder {
    /// Lift a framework frame's planes into the encoder's `u16`
    /// representation (little-endian pairs at 10 / 12 bit).
    fn to_yuv(&self, frame: &VideoFrame) -> CoreResult<YuvFrame> {
        let planes = frame.image_planes();
        let need = usize::from(self.format.num_planes());
        if planes.len() < need {
            return Err(CoreError::invalid(format!(
                "oxideav-av1: frame carries {} planes, format needs {need}",
                planes.len()
            )));
        }
        let bps: usize = if self.bit_depth > 8 { 2 } else { 1 };
        let lift = |p: &VideoPlane, w: usize, h: usize| -> CoreResult<Vec<u16>> {
            let row_bytes = w * bps;
            if p.stride < row_bytes || p.data.len() < p.stride * (h - 1) + row_bytes {
                return Err(CoreError::invalid(
                    "oxideav-av1: frame plane shorter than the declared geometry",
                ));
            }
            let mut out = Vec::with_capacity(w * h);
            for y in 0..h {
                let row = &p.data[y * p.stride..y * p.stride + row_bytes];
                if bps == 1 {
                    out.extend(row.iter().map(|&b| u16::from(b)));
                } else {
                    out.extend(
                        row.chunks_exact(2)
                            .map(|c| u16::from_le_bytes([c[0], c[1]])),
                    );
                }
            }
            Ok(out)
        };
        let (w, h) = (self.width as usize, self.height as usize);
        let probe = YuvFrame::filled(self.width, self.height, self.bit_depth, self.format, 0);
        let (cw, ch) = (
            probe.chroma_width() as usize,
            probe.chroma_height() as usize,
        );
        let y = lift(&planes[0], w, h)?;
        let (u, v) = if need > 1 {
            (lift(&planes[1], cw, ch)?, lift(&planes[2], cw, ch)?)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(YuvFrame {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            format: self.format,
            y,
            u,
            v,
        })
    }
}

impl Encoder for Av1Encoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn output_params(&self) -> &CodecParameters {
        &self.params
    }

    fn send_frame(&mut self, frame: &CoreFrame) -> CoreResult<()> {
        let CoreFrame::Video(vf) = frame else {
            return Err(CoreError::invalid(
                "oxideav-av1: encoder takes video frames",
            ));
        };
        let input = self.to_yuv(vf)?;
        let tu = if self.opts.still {
            encode_still_yuv(&input, &self.opts.still_options())
                .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e}")))?
                .temporal_unit_bytes
        } else {
            // TODO(r460 followup): the all-intra arm ignores tiles /
            // speed / full_range — it rides the historical KEY entry.
            encode_key_frame_yuv_with_q(&input, self.opts.base_q_idx)
                .map_err(|e| CoreError::invalid(format!("oxideav-av1: {e}")))?
                .temporal_unit_bytes
        };
        let pts = vf.pts.unwrap_or(self.next_pts);
        self.next_pts = pts + 1;
        self.queue.push_back(
            Packet::new(0, self.time_base, tu)
                .with_pts(pts)
                .with_dts(pts)
                .with_duration(1)
                .with_keyframe(true),
        );
        Ok(())
    }

    fn receive_packet(&mut self) -> CoreResult<Packet> {
        match self.queue.pop_front() {
            Some(p) => Ok(p),
            None if self.eof => Err(CoreError::Eof),
            None => Err(CoreError::NeedMore),
        }
    }

    fn flush(&mut self) -> CoreResult<()> {
        self.eof = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxideav_core::ProbeContext;

    fn textured_frame(width: u32, height: u32, pf: PixelFormat) -> (VideoFrame, YuvFrame) {
        let (bd, fmt) = pixel_format_layout(pf).expect("supported");
        let mut yuv = YuvFrame::filled(width, height, bd, fmt, 0);
        let max = (1u32 << bd) - 1;
        for (i, s) in yuv.y.iter_mut().enumerate() {
            *s = ((i as u32 * 37 + (i as u32 / width) * 91) % max) as u16;
        }
        for (i, s) in yuv.u.iter_mut().enumerate() {
            *s = ((i as u32 * 13) % max) as u16;
        }
        for (i, s) in yuv.v.iter_mut().enumerate() {
            *s = ((i as u32 * 29 + 5) % max) as u16;
        }
        let pack = |p: &[u16], w: usize| -> VideoPlane {
            let mut data = Vec::new();
            for &s in p {
                if bd > 8 {
                    data.extend_from_slice(&s.to_le_bytes());
                } else {
                    data.push(s as u8);
                }
            }
            VideoPlane {
                stride: w * if bd > 8 { 2 } else { 1 },
                data,
            }
        };
        let mut planes = vec![pack(&yuv.y, width as usize)];
        if fmt != ChromaFormat::Monochrome {
            planes.push(pack(&yuv.u, yuv.chroma_width() as usize));
            planes.push(pack(&yuv.v, yuv.chroma_width() as usize));
        }
        (
            VideoFrame {
                pts: Some(7),
                planes,
            },
            yuv,
        )
    }

    fn video_params(width: u32, height: u32, pf: PixelFormat) -> CodecParameters {
        let mut p = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        p.width = Some(width);
        p.height = Some(height);
        p.pixel_format = Some(pf);
        p
    }

    /// r460 — the framework encoder's still arm produces a
    /// reduced-header still per frame whose `av1C` extradata matches
    /// the in-band sequence header, and the framework decoder (fed
    /// that extradata + the packet) reproduces the direct-API
    /// reconstruction.
    #[test]
    fn framework_still_encode_round_trips_through_the_framework_decoder() {
        for (pf, opts) in [
            (PixelFormat::Yuv420P, [("still", "true"), ("quality", "60")]),
            (PixelFormat::Gray10Le, [("still", "true"), ("q", "80")]),
            (
                PixelFormat::YuvJ444P,
                [("still", "true"), ("lossless", "true")],
            ),
        ] {
            let mut params = video_params(32, 24, pf);
            for (k, v) in opts {
                params.options.insert(k, v);
            }
            let mut enc = make_encoder(&params).expect("encoder constructs");
            let (frame, yuv) = textured_frame(32, 24, pf);
            enc.send_frame(&CoreFrame::Video(frame))
                .expect("frame accepted");
            let pkt = enc.receive_packet().expect("one packet");
            assert!(pkt.flags.keyframe && pkt.pts == Some(7));
            assert!(matches!(enc.receive_packet(), Err(CoreError::NeedMore)));
            enc.flush().unwrap();
            assert!(matches!(enc.receive_packet(), Err(CoreError::Eof)));

            let extradata = enc.output_params().extradata.clone();
            let cfg = Av1CodecConfig::parse(&extradata).expect("av1C extradata");
            let seq = crate::sequence_header::parse_sequence_header(
                crate::obu::ObuIter::new(&pkt.data)
                    .filter_map(Result::ok)
                    .find(|d| d.obu_type == crate::obu::ObuType::SequenceHeader)
                    .expect("in-band sequence header")
                    .payload,
            )
            .expect("parses");
            assert!(seq.still_picture && seq.reduced_still_picture_header);
            assert!(cfg.matches_sequence_header(&seq), "{pf:?}: av1C mismatch");
            assert_eq!(seq.color_config.color_range, pf == PixelFormat::YuvJ444P);

            let opts = Av1EncoderOptions::from_params(&params).unwrap();
            let direct = encode_still_yuv(&yuv, &opts.still_options()).expect("direct still");
            assert_eq!(
                direct.temporal_unit_bytes, pkt.data,
                "{pf:?}: framework != direct"
            );

            let mut dparams = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            dparams.extradata = extradata;
            let mut dec = make_decoder(&dparams).expect("decoder");
            dec.send_packet(&pkt).expect("decodes");
            let CoreFrame::Video(out) = dec.receive_frame().expect("frame") else {
                panic!("video frame expected");
            };
            let widen = |p: &[u8]| -> Vec<u16> {
                if cfg.bit_depth() > 8 {
                    p.chunks(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect()
                } else {
                    p.iter().map(|&b| u16::from(b)).collect()
                }
            };
            assert_eq!(widen(&out.planes[0].data), direct.recon_y);
            if opts.base_q_idx == 0 {
                assert_eq!(direct.recon_y, yuv.y, "{pf:?}: lossless luma");
            }
        }
    }

    /// r460 — the all-intra (video) arm emits KEY frames under a full
    /// sequence header, one packet per frame with running timestamps.
    #[test]
    fn framework_all_intra_encode_emits_one_key_packet_per_frame() {
        let mut params = video_params(16, 16, PixelFormat::Yuv420P);
        params.options.insert("q", "200");
        params.options.insert("speed", "fast");
        let mut enc = make_encoder(&params).expect("encoder constructs");
        assert_eq!(enc.output_params().extradata, vec![0x81, 0x00, 0x0c, 0x00]);
        for i in 0..2 {
            let (mut frame, _) = textured_frame(16, 16, PixelFormat::Yuv420P);
            frame.pts = None;
            enc.send_frame(&CoreFrame::Video(frame)).expect("frame");
            let pkt = enc.receive_packet().expect("packet");
            assert_eq!(pkt.pts, Some(i));
            let frames = crate::decoder::decode_av1_spec(&{
                let mut buf = Vec::new();
                let cur = std::io::Cursor::new(&mut buf);
                let mut w = crate::encoder::IvfWriter::new(
                    cur,
                    crate::encoder::ivf::FOURCC_AV01,
                    16,
                    16,
                    25,
                    1,
                )
                .unwrap();
                w.write_frame(&pkt.data, 0).unwrap();
                w.patch_frame_count().unwrap();
                buf
            })
            .expect("decodes");
            assert_eq!(frames.len(), 1);
        }
    }

    #[test]
    fn encoder_options_and_geometry_are_validated() {
        let mut params = video_params(16, 16, PixelFormat::Yuv420P);
        params.options.insert("speed", "warp");
        assert!(make_encoder(&params).is_err());
        let mut params = video_params(16, 16, PixelFormat::Yuv420P);
        params.options.insert("bogus", "1");
        assert!(make_encoder(&params).is_err());
        let params = video_params(16, 16, PixelFormat::Rgb24);
        assert!(make_encoder(&params).is_err());
        let mut params = video_params(16, 16, PixelFormat::Yuv420P);
        params.options.insert("quality", "100");
        let o = Av1EncoderOptions::from_params(&params).unwrap();
        assert_eq!(o.base_q_idx, 0);
        params.options.insert("full_range", "true");
        params.options.insert("tile_cols_log2", "1");
        let o = Av1EncoderOptions::from_params(&params).unwrap();
        assert!(o.full_range && o.tile_cols_log2 == 1);
    }

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

    /// r460 — an `av1C` extradata whose `configOBUs` carry the
    /// sequence header lets an item payload that omits it decode; the
    /// same payload without the extradata is refused (no sequence
    /// header), and a payload that repeats the header decodes to the
    /// same pixels.
    #[test]
    fn av1c_extradata_config_obus_seed_the_sequence_header() {
        use crate::encoder::{encode_still_yuv420, StillOptions, Yuv420Frame};
        use crate::obu::{ObuIter, ObuType};
        let mut frame = Yuv420Frame::filled(32, 16, 60);
        for (i, s) in frame.y.iter_mut().enumerate() {
            *s = (i * 7 % 200) as u8 + 20;
        }
        let still = encode_still_yuv420(&frame, &StillOptions::new(90)).expect("encodes");
        // Split the temporal unit into its sequence header OBU and a
        // header-less payload (temporal delimiter + frame OBU).
        let mut seq_obu = Vec::new();
        let mut frame_obus = Vec::new();
        for desc in ObuIter::new(&still.temporal_unit_bytes) {
            let desc = desc.expect("well-formed OBU");
            let obu = crate::encoder::obu::ObuFrame::new(desc.obu_type, desc.payload.to_vec());
            if desc.obu_type == ObuType::SequenceHeader {
                seq_obu.push(obu);
            } else {
                frame_obus.push(obu);
            }
        }
        let config_obus = crate::encoder::obu::write_temporal_unit(&seq_obu);
        let payload = crate::encoder::obu::write_temporal_unit(&frame_obus);
        let mut av1c = still.codec_config.clone();
        av1c.config_obus = config_obus.clone();

        let decode_with = |extradata: Vec<u8>, packet: &[u8]| -> CoreResult<Vec<Vec<u8>>> {
            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.extradata = extradata;
            let mut dec = make_decoder(&params)?;
            dec.send_packet(&Packet::new(
                0,
                oxideav_core::TimeBase::new(1, 1),
                packet.to_vec(),
            ))?;
            match dec.receive_frame()? {
                CoreFrame::Video(v) => Ok(v.planes.into_iter().map(|p| p.data).collect()),
                other => panic!("expected a video frame, got {other:?}"),
            }
        };
        let expected = decode_with(Vec::new(), &still.temporal_unit_bytes).expect("in-band");
        assert!(
            decode_with(Vec::new(), &payload).is_err(),
            "a header-less payload must be refused without extradata"
        );
        let via_av1c = decode_with(av1c.to_bytes(), &payload).expect("av1C configOBUs");
        assert_eq!(via_av1c, expected);
        let via_raw = decode_with(config_obus, &payload).expect("bare OBU extradata");
        assert_eq!(via_raw, expected);
        let repeated = decode_with(av1c.to_bytes(), &still.temporal_unit_bytes)
            .expect("repeated sequence header");
        assert_eq!(repeated, expected);
        // A record without configOBUs is accepted and ignored.
        let bare = decode_with(still.codec_config.to_bytes(), &still.temporal_unit_bytes)
            .expect("bare av1C");
        assert_eq!(bare, expected);
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
