//! r460 — the STILL-PICTURE entry point: one intra frame coded under a
//! §6.4.1 `still_picture = 1` / `reduced_still_picture_header = 1`
//! sequence header, with the same quality controls the KEY-frame
//! encoder has (`base_q_idx`, the elected §5.9.17 delta-q plan, the
//! §5.9.12 quantizer-matrix ladder, `base_q_idx = 0` lossless) — the
//! shape every AVIF / HEIF `av01` image item carries (av1-avif §2.1
//! recommends the reduced header "so that AV1 header overhead is
//! minimized"; every third-party AVIF producer measured in r460 emits
//! it).
//!
//! What the reduced header changes on the wire (§5.5.1 / §5.9.2):
//! one operating point (`seq_level_idx` only — no `operating_point_idc`
//! / tier / timing / decoder-model blocks), `frame_id_numbers_present_flag
//! = 0`, every inter tool gate (`enable_interintra_compound`,
//! `enable_masked_compound`, `enable_warped_motion`,
//! `enable_dual_filter`, `enable_order_hint`, `enable_jnt_comp`,
//! `enable_ref_frame_mvs`) inferred to 0 with `OrderHintBits = 0`,
//! `seq_force_screen_content_tools` / `seq_force_integer_mv` inferred
//! to SELECT, and a frame header that is KEY / shown / not showable /
//! error-resilient with `refresh_frame_flags = allFrames` and
//! `disable_frame_end_update_cdf = 1` — all derived, none coded. The
//! coded tile data is byte-identical to the KEY-frame encoder's for
//! the same decisions, so every §7 reconstruction guarantee carries
//! over: the decoder output equals [`EncodedStill::recon_y`] /
//! `recon_u` / `recon_v` sample for sample.
//!
//! The Annex A level (`seq_level_idx`) is elected from the picture
//! size ([`elect_seq_level_idx`]): the smallest level whose
//! `MaxPicSize` / `MaxHSize` / `MaxVSize` admit the frame (a still
//! has no display-rate / decode-rate / bit-rate constraint to meet),
//! `31` (maximum parameters) beyond level 6.3.

use crate::codec_config::Av1CodecConfig;
use crate::encoder::key_frame::{encode_key_frame_yuv_full, EncodedKeyFrameYuv, KeyExtras};
use crate::encoder::rate_twin::RateModel;
use crate::encoder::yuv_frame::{Yuv420Frame, YuvFrame};
use crate::frame_header::FrameHeader;
use crate::sequence_header::{SequenceHeader, SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};
use crate::Error;

/// Search-effort presets for [`StillOptions::speed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StillSpeed {
    /// The frame-level elections that each run a complete second
    /// search (the §5.9.12 quantizer-matrix ladder, the §5.9.17
    /// delta-q plan) stay off; the block-level RD search is
    /// unchanged. Roughly a third of `Balanced`'s wall clock on
    /// textured content.
    Fast,
    /// The KEY-frame encoder's production shape: QM + delta-q
    /// elections on, the §5.9.8 superres election off (it costs one
    /// full search per candidate denominator and a still gains
    /// nothing from a downscaled coding extent).
    #[default]
    Balanced,
    /// Everything `Balanced` runs plus the §5.9.8 superres election.
    Thorough,
}

/// Options for [`encode_still_yuv`] / [`encode_still_yuv420`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StillOptions {
    /// §5.9.12 `base_q_idx` (0..=255). `0` codes the §5.9.2
    /// `CodedLossless` arm: the decoded picture equals the input
    /// sample for sample.
    pub base_q_idx: u8,
    /// §5.9.15 uniform tile layout `(TileColsLog2, TileRowsLog2)` —
    /// `(0, 0)` codes one tile. Must lie inside the §5.9.15 legal
    /// window for the picture size (see
    /// [`crate::tile_info::TileInfo::uniform_layout`]).
    pub tile_cols_log2: u32,
    /// See [`StillOptions::tile_cols_log2`].
    pub tile_rows_log2: u32,
    /// Search effort.
    pub speed: StillSpeed,
    /// §5.5.2 `color_range` — `true` signals full-range samples (an
    /// alpha auxiliary item is conventionally full range). Signalling
    /// only; the coding is unchanged.
    pub full_range: bool,
    /// `false` codes the same still picture under a FULL sequence
    /// header (`still_picture = 1`, `reduced_still_picture_header =
    /// 0`) — the shape a third-party producer's
    /// `--full-still-picture-hdr` emits. The frame header then carries
    /// the ordinary KEY-frame fields (`show_existing_frame`,
    /// `frame_type`, `show_frame`, ...).
    pub reduced_header: bool,
}

impl StillOptions {
    /// Lossy still at `base_q_idx`, single tile, `Balanced` effort,
    /// studio range, reduced header.
    #[must_use]
    pub fn new(base_q_idx: u8) -> Self {
        Self {
            base_q_idx,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            speed: StillSpeed::Balanced,
            full_range: false,
            reduced_header: true,
        }
    }

    /// The lossless shape (`base_q_idx = 0`).
    #[must_use]
    pub fn lossless(mut self) -> Self {
        self.base_q_idx = 0;
        self
    }

    /// Map a 0..=100 quality dial (100 = lossless, 0 = coarsest) onto
    /// `base_q_idx`: `round((100 - quality) * 255 / 100)`.
    #[must_use]
    pub fn from_quality(quality: u8) -> Self {
        Self::new(quality_to_base_q_idx(quality))
    }
}

impl Default for StillOptions {
    fn default() -> Self {
        Self::new(DEFAULT_STILL_BASE_Q_IDX)
    }
}

/// The `base_q_idx` [`StillOptions::default`] codes at — the middle of
/// the useful photographic range (a third-party producer's quality 50
/// lands near it).
pub const DEFAULT_STILL_BASE_Q_IDX: u8 = 120;

/// The [`StillOptions::from_quality`] mapping.
#[must_use]
pub fn quality_to_base_q_idx(quality: u8) -> u8 {
    let q = u32::from(quality.min(100));
    (((100 - q) * 255 + 50) / 100) as u8
}

/// Result of [`encode_still_yuv`] / [`encode_still_yuv420`].
#[derive(Debug, Clone)]
pub struct EncodedStill {
    /// The bare §7.5 temporal unit (TD + SH + `OBU_FRAME`) — the AV1
    /// Image Item Data / ISOBMFF sample payload.
    pub temporal_unit_bytes: Vec<u8>,
    /// Complete IVF v0 file (header + one frame record) around the
    /// same temporal unit.
    pub ivf_bytes: Vec<u8>,
    /// Encoder reconstruction of the luma plane (row-major, `u16` at
    /// the input's bit depth). The decoded output equals these sample
    /// for sample; at `base_q_idx == 0` they additionally equal the
    /// input.
    pub recon_y: Vec<u16>,
    /// U plane reconstruction (subsampled extent; empty on
    /// monochrome).
    pub recon_u: Vec<u16>,
    /// V plane reconstruction.
    pub recon_v: Vec<u16>,
    /// The emitted sequence header descriptor.
    pub seq: SequenceHeader,
    /// The emitted frame header descriptor.
    pub fh: FrameHeader,
    /// The `av1C` record fields a container needs, derived from
    /// `seq` (`config_obus` empty, per av1-avif §2.2.1).
    pub codec_config: Av1CodecConfig,
}

impl EncodedStill {
    fn from_key(k: EncodedKeyFrameYuv) -> Self {
        let codec_config = Av1CodecConfig::from_sequence_header(&k.seq);
        Self {
            temporal_unit_bytes: k.temporal_unit_bytes,
            ivf_bytes: k.ivf_bytes,
            recon_y: k.recon_y,
            recon_u: k.recon_u,
            recon_v: k.recon_v,
            seq: k.seq,
            fh: k.fh,
            codec_config,
        }
    }
}

/// Annex A.3 level limits `(seq_level_idx, MaxPicSize, MaxHSize,
/// MaxVSize)` in ascending order — one row per level GROUP (the x.1
/// / x.2 / x.3 siblings share the picture-size limits and differ only
/// in the rate limits a still never meets).
const LEVEL_PIC_LIMITS: [(u8, u64, u32, u32); 6] = [
    (0, 147_456, 2048, 1152),    // 2.0
    (1, 278_784, 2816, 1584),    // 2.1
    (4, 665_856, 4352, 2448),    // 3.0
    (5, 1_065_024, 5504, 3096),  // 3.1
    (8, 2_359_296, 6144, 3456),  // 4.0
    (12, 8_912_896, 8192, 4352), // 5.0
];

/// Level 6.0 — `MaxPicSize 35651584`, `MaxHSize 16384`, `MaxVSize
/// 8704`.
const LEVEL_6_0: (u8, u64, u32, u32) = (16, 35_651_584, 16384, 8704);

/// `seq_level_idx = 31`: the "maximum parameters" level (§6.4.1 / A.3
/// — no limits apply).
pub const SEQ_LEVEL_IDX_MAX_PARAMETERS: u8 = 31;

/// Elect the smallest Annex A level whose `MaxPicSize` / `MaxHSize` /
/// `MaxVSize` admit a `width × height` picture (the still-picture
/// election — no rate terms). Returns
/// [`SEQ_LEVEL_IDX_MAX_PARAMETERS`] when even level 6.0's limits are
/// exceeded.
#[must_use]
pub fn elect_seq_level_idx(width: u32, height: u32) -> u8 {
    let pic = u64::from(width) * u64::from(height);
    LEVEL_PIC_LIMITS
        .iter()
        .chain(core::iter::once(&LEVEL_6_0))
        .find(|&&(_, max_pic, max_h, max_v)| pic <= max_pic && width <= max_h && height <= max_v)
        .map_or(SEQ_LEVEL_IDX_MAX_PARAMETERS, |&(idx, _, _, _)| idx)
}

/// Turn an intra-capable sequence header (as
/// [`crate::encoder::yuv_frame::build_intra_only_seq_yuv`] builds it)
/// into the reduced still-picture shape: every field §5.5.1 infers
/// under `reduced_still_picture_header = 1` is set to its inferred
/// value so the encoder's own derivations match the decoder's, and
/// operating point 0 carries the Annex A level elected from the
/// maximum frame size.
pub fn apply_still_picture_shape(seq: &mut SequenceHeader) {
    seq.still_picture = true;
    seq.reduced_still_picture_header = true;
    seq.timing_info_present_flag = false;
    seq.timing_info = None;
    seq.decoder_model_info_present_flag = false;
    seq.decoder_model_info = None;
    seq.initial_display_delay_present_flag = false;
    seq.operating_points_cnt_minus_1 = 0;
    seq.operating_points.truncate(1);
    if let Some(op) = seq.operating_points.first_mut() {
        op.operating_point_idc = 0;
        op.seq_level_idx = elect_seq_level_idx(
            seq.max_frame_width_minus_1 + 1,
            seq.max_frame_height_minus_1 + 1,
        );
        op.seq_tier = 0;
        op.decoder_model_present_for_this_op = false;
        op.operating_parameters_info = None;
        op.initial_display_delay_present_for_this_op = false;
        op.initial_display_delay_minus_1 = None;
    }
    seq.frame_id_numbers_present_flag = false;
    seq.delta_frame_id_length_minus_2 = 0;
    seq.additional_frame_id_length_minus_1 = 0;
    seq.enable_interintra_compound = false;
    seq.enable_masked_compound = false;
    seq.enable_warped_motion = false;
    seq.enable_dual_filter = false;
    seq.enable_order_hint = false;
    seq.enable_jnt_comp = false;
    seq.enable_ref_frame_mvs = false;
    seq.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
    seq.seq_force_integer_mv = SELECT_INTEGER_MV;
    seq.order_hint_bits = 0;
}

/// Encode one still picture at any §6.4.1 (bit depth, chroma format)
/// pairing — 8/10/12-bit × 4:2:0 / 4:2:2 / 4:4:4 / monochrome (an
/// alpha auxiliary item is a monochrome still).
///
/// ## Errors
///
/// * [`YuvFrame::validate`] failures (shape / depth / sample range)
///   and a tile layout outside the §5.9.15 legal window —
///   [`Error::PartitionWalkOutOfRange`].
/// * Internal writer overflow surfaces the underlying [`Error`].
pub fn encode_still_yuv(input: &YuvFrame, opts: &StillOptions) -> Result<EncodedStill, Error> {
    let elections = opts.speed != StillSpeed::Fast && opts.base_q_idx > 0;
    let extras = KeyExtras {
        tiles: (opts.tile_cols_log2, opts.tile_rows_log2),
        delta_q: elections,
        qm: elections,
        superres_elect: opts.speed == StillSpeed::Thorough && opts.base_q_idx > 0,
        still: opts.reduced_header,
        full_range: opts.full_range,
        still_full_header: !opts.reduced_header,
        ..KeyExtras::default()
    };
    let (k, _carry) = encode_key_frame_yuv_full(
        input,
        opts.base_q_idx,
        RateModel::Twin,
        &[],
        None,
        true,
        true,
        true,
        &extras,
    )?;
    Ok(EncodedStill::from_key(k))
}

/// 8-bit 4:2:0 sibling of [`encode_still_yuv`].
pub fn encode_still_yuv420(
    input: &Yuv420Frame,
    opts: &StillOptions,
) -> Result<EncodedStill, Error> {
    encode_still_yuv(&YuvFrame::from_yuv420_8bit(input), opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::decode_av1_spec;
    use crate::encoder::yuv_frame::ChromaFormat;

    fn textured(width: u32, height: u32, bit_depth: u8, format: ChromaFormat) -> YuvFrame {
        let mut f = YuvFrame::filled(width, height, bit_depth, format, 0);
        let max = (1u32 << bit_depth) - 1;
        for y in 0..height {
            for x in 0..width {
                let v = (x * 37 + y * 91 + (x * y) % 23) * max / (width * 37 + height * 91 + 23);
                f.y[(y * width + x) as usize] = v as u16;
            }
        }
        let (cw, ch) = (f.chroma_width(), f.chroma_height());
        for y in 0..ch {
            for x in 0..cw {
                f.u[(y * cw + x) as usize] = ((x * 53 + y * 7) % max) as u16;
                f.v[(y * cw + x) as usize] = ((x * 11 + y * 61) % max) as u16;
            }
        }
        f
    }

    fn assert_decodes_to_recon(still: &EncodedStill, bit_depth: u8) {
        let frames = decode_av1_spec(&still.ivf_bytes).expect("still decodes");
        assert_eq!(frames.len(), 1);
        let f = &frames[0];
        let widen = |p: &[u8]| -> Vec<u16> {
            if bit_depth > 8 {
                p.chunks(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            } else {
                p.iter().map(|&b| u16::from(b)).collect()
            }
        };
        assert_eq!(widen(&f.planes[0]), still.recon_y, "luma");
        if f.planes.len() > 1 {
            assert_eq!(widen(&f.planes[1]), still.recon_u, "u");
            assert_eq!(widen(&f.planes[2]), still.recon_v, "v");
        } else {
            assert!(still.recon_u.is_empty());
        }
    }

    #[test]
    fn level_election_follows_annex_a_picture_limits() {
        assert_eq!(elect_seq_level_idx(426, 240), 0);
        assert_eq!(elect_seq_level_idx(640, 480), 4);
        assert_eq!(elect_seq_level_idx(1280, 720), 5);
        assert_eq!(elect_seq_level_idx(1920, 1080), 8);
        assert_eq!(elect_seq_level_idx(3840, 2160), 12);
        assert_eq!(elect_seq_level_idx(4032, 3024), 16);
        assert_eq!(elect_seq_level_idx(2048, 1152), 8);
        assert_eq!(elect_seq_level_idx(512, 288), 0);
        assert_eq!(elect_seq_level_idx(2049, 8), 1);
        assert_eq!(elect_seq_level_idx(16385, 8), SEQ_LEVEL_IDX_MAX_PARAMETERS);
        assert_eq!(elect_seq_level_idx(8, 8705), SEQ_LEVEL_IDX_MAX_PARAMETERS);
    }

    #[test]
    fn quality_dial_endpoints() {
        assert_eq!(quality_to_base_q_idx(100), 0);
        assert_eq!(quality_to_base_q_idx(0), 255);
        assert_eq!(quality_to_base_q_idx(50), 128);
        assert_eq!(quality_to_base_q_idx(200), 0);
    }

    #[test]
    fn reduced_still_round_trips_every_format_pairing() {
        for &(bd, fmt) in &[
            (8u8, ChromaFormat::Yuv420),
            (8, ChromaFormat::Monochrome),
            (10, ChromaFormat::Yuv422),
            (12, ChromaFormat::Yuv444),
        ] {
            let input = textured(48, 40, bd, fmt);
            let still = encode_still_yuv(&input, &StillOptions::new(96)).expect("still encodes");
            assert!(still.seq.still_picture && still.seq.reduced_still_picture_header);
            assert_eq!(still.seq.operating_points.len(), 1);
            assert_eq!(still.seq.operating_points[0].seq_level_idx, 0);
            assert!(!still.seq.enable_order_hint && still.seq.order_hint_bits == 0);
            assert!(still.fh.disable_frame_end_update_cdf);
            assert_eq!(still.codec_config.bit_depth(), bd);
            assert_eq!(
                still.codec_config.monochrome,
                fmt == ChromaFormat::Monochrome
            );
            assert!(still.codec_config.matches_sequence_header(&still.seq));
            assert_decodes_to_recon(&still, bd);
        }
    }

    #[test]
    fn lossless_still_reproduces_the_input() {
        let input = textured(40, 24, 8, ChromaFormat::Yuv420);
        let still =
            encode_still_yuv(&input, &StillOptions::new(0)).expect("lossless still encodes");
        assert_eq!(still.recon_y, input.y);
        assert_eq!(still.recon_u, input.u);
        assert_eq!(still.recon_v, input.v);
        assert_decodes_to_recon(&still, 8);
    }

    #[test]
    fn full_header_still_and_options_round_trip() {
        let input = textured(64, 32, 8, ChromaFormat::Yuv420);
        let mut opts = StillOptions::new(140);
        opts.reduced_header = false;
        opts.full_range = true;
        opts.speed = StillSpeed::Fast;
        let still = encode_still_yuv(&input, &opts).expect("full-header still encodes");
        assert!(still.seq.still_picture && !still.seq.reduced_still_picture_header);
        assert!(still.seq.color_config.color_range);
        assert_eq!(still.seq.operating_points[0].seq_level_idx, 0);
        assert_decodes_to_recon(&still, 8);

        let yuv420 = Yuv420Frame::filled(16, 16, 77);
        let s2 = encode_still_yuv420(&yuv420, &StillOptions::default()).expect("encodes");
        assert_eq!(s2.recon_y.len(), 256);
        assert_decodes_to_recon(&s2, 8);
    }

    #[test]
    fn tiled_still_round_trips() {
        let input = textured(256, 128, 8, ChromaFormat::Yuv420);
        let mut opts = StillOptions::new(110);
        opts.tile_cols_log2 = 1;
        opts.tile_rows_log2 = 1;
        opts.speed = StillSpeed::Fast;
        let still = encode_still_yuv(&input, &opts).expect("tiled still encodes");
        let ti = still.fh.tile_info.as_ref().expect("tile info");
        assert_eq!((ti.tile_cols, ti.tile_rows), (2, 2));
        assert_decodes_to_recon(&still, 8);
    }
}
