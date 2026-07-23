//! General YUV input frame for the conformance-grade encoders (r427).
//!
//! [`YuvFrame`] carries any §6.4.1 (bit depth, chroma format) pairing
//! the three AV1 profiles admit — 8 / 10 / 12-bit samples in 4:2:0,
//! 4:2:2, 4:4:4 or monochrome layout — as `u16` planes. The module
//! also owns the §6.4.1 `seq_profile` election and the §5.5.2
//! `color_config` synthesis that map a pairing onto the wire:
//!
//! | `seq_profile` | Bit depth | Monochrome | Chroma subsampling      |
//! |---------------|-----------|------------|-------------------------|
//! | 0             | 8 or 10   | yes        | YUV 4:2:0               |
//! | 1             | 8 or 10   | no         | YUV 4:4:4               |
//! | 2             | 8 or 10   | yes        | YUV 4:2:2               |
//! | 2             | 12        | yes        | 4:2:0 / 4:2:2 / 4:4:4   |
//!
//! (§5.5.2 semantics, "seq_profile" table; monochrome is only
//! signalable on profiles 0 and 2.)
//!
//! The historical 8-bit 4:2:0 entry points keep their
//! [`Yuv420Frame`]-based signatures; [`YuvFrame::from_yuv420_8bit`]
//! widens those inputs into this representation so the whole encoder
//! pipeline runs on one sample type.

use crate::frame_header::{FrameHeader, FrameSize, FrameType, ALL_FRAMES_PUB, PRIMARY_REF_NONE};
use crate::sequence_header::{
    ColorConfig, OperatingPoint, SequenceHeader, CP_UNSPECIFIED, CSP_UNKNOWN, MC_UNSPECIFIED,
    SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS, TC_UNSPECIFIED,
};
use crate::tile_info::TileInfo;
use crate::uncompressed_header_tail::{
    CdefParams, DeltaLfParams, DeltaQParams, FilmGrainParams, FrameRestorationType,
    GlobalMotionParams, LoopFilterParams, LrParams, QuantizationParams, SegmentationParams, TxMode,
};
use crate::Error;

/// §6.4.2 chroma layout selector: the `(subsampling_x, subsampling_y,
/// mono_chrome)` triple every geometry derivation keys off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Luma only (`mono_chrome = 1`, `NumPlanes = 1`). §5.5.2 infers
    /// `subsampling_x = subsampling_y = 1` on the mono arm.
    Monochrome,
    /// `subsampling_x = subsampling_y = 1` — chroma at half extent on
    /// both axes.
    Yuv420,
    /// `subsampling_x = 1`, `subsampling_y = 0` — chroma at half
    /// horizontal extent, full vertical extent.
    Yuv422,
    /// `subsampling_x = subsampling_y = 0` — chroma at full extent.
    Yuv444,
}

impl ChromaFormat {
    /// §6.4.2 `(subsampling_x, subsampling_y)` for this layout (the
    /// §5.5.2-inferred `(1, 1)` on the monochrome arm).
    #[must_use]
    pub fn subsampling(self) -> (u8, u8) {
        match self {
            ChromaFormat::Monochrome | ChromaFormat::Yuv420 => (1, 1),
            ChromaFormat::Yuv422 => (1, 0),
            ChromaFormat::Yuv444 => (0, 0),
        }
    }

    /// §5.5.2 `NumPlanes` (`1` for monochrome, `3` otherwise).
    #[must_use]
    pub fn num_planes(self) -> u8 {
        match self {
            ChromaFormat::Monochrome => 1,
            _ => 3,
        }
    }
}

/// Dynamic-extent YUV input at any conformant (bit depth, chroma
/// format) pairing. Samples are `u16` regardless of depth; every
/// sample must lie in `[0, (1 << bit_depth) - 1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YuvFrame {
    /// Luma width in pixels (multiple of 8).
    pub width: u32,
    /// Luma height in pixels (multiple of 8).
    pub height: u32,
    /// §5.5.2 `BitDepth`: 8, 10 or 12.
    pub bit_depth: u8,
    /// Chroma layout.
    pub format: ChromaFormat,
    /// Luma plane, row-major, length `width * height`.
    pub y: Vec<u16>,
    /// Cb plane at the subsampled extent ([`Self::chroma_width`] ×
    /// [`Self::chroma_height`]); empty on the monochrome arm.
    pub u: Vec<u16>,
    /// Cr plane; same shape as `u`.
    pub v: Vec<u16>,
}

impl YuvFrame {
    /// All-`fill` constant frame (every plane set to `fill`).
    #[must_use]
    pub fn filled(width: u32, height: u32, bit_depth: u8, format: ChromaFormat, fill: u16) -> Self {
        let (ssx, ssy) = format.subsampling();
        let (cw, ch) = if format == ChromaFormat::Monochrome {
            (0, 0)
        } else {
            (width >> ssx, height >> ssy)
        };
        Self {
            width,
            height,
            bit_depth,
            format,
            y: vec![fill; (width * height) as usize],
            u: vec![fill; (cw * ch) as usize],
            v: vec![fill; (cw * ch) as usize],
        }
    }

    /// Widen an 8-bit 4:2:0 [`Yuv420Frame`] into the general
    /// representation (lossless).
    #[must_use]
    pub fn from_yuv420_8bit(input: &Yuv420Frame) -> Self {
        let widen = |p: &[u8]| p.iter().map(|&s| u16::from(s)).collect::<Vec<u16>>();
        Self {
            width: input.width,
            height: input.height,
            bit_depth: 8,
            format: ChromaFormat::Yuv420,
            y: widen(&input.y),
            u: widen(&input.u),
            v: widen(&input.v),
        }
    }

    /// Chroma plane width in samples (`width >> subsampling_x`; `0`
    /// on the monochrome arm).
    #[must_use]
    pub fn chroma_width(&self) -> u32 {
        if self.format == ChromaFormat::Monochrome {
            0
        } else {
            self.width >> self.format.subsampling().0
        }
    }

    /// Chroma plane height in samples (`height >> subsampling_y`; `0`
    /// on the monochrome arm).
    #[must_use]
    pub fn chroma_height(&self) -> u32 {
        if self.format == ChromaFormat::Monochrome {
            0
        } else {
            self.height >> self.format.subsampling().1
        }
    }

    /// Validate shape + sample range: dimensions multiples of 8 in
    /// `[8, 4096]` per axis, `bit_depth ∈ {8, 10, 12}`, plane lengths
    /// consistent with the format (empty chroma on monochrome), every
    /// sample `< (1 << bit_depth)`.
    ///
    /// ## Errors
    ///
    /// [`Error::PartitionWalkOutOfRange`] on any violation.
    pub fn validate(&self) -> Result<(), Error> {
        if !matches!(self.bit_depth, 8 | 10 | 12) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.width < 8
            || self.height < 8
            || self.width > crate::encoder::key_frame::KEY_FRAME_MAX_DIM
            || self.height > crate::encoder::key_frame::KEY_FRAME_MAX_DIM
            || self.width % 8 != 0
            || self.height % 8 != 0
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let expected_y = (self.width * self.height) as usize;
        let expected_uv = (self.chroma_width() * self.chroma_height()) as usize;
        if self.y.len() != expected_y || self.u.len() != expected_uv || self.v.len() != expected_uv
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let ceil = 1u16 << self.bit_depth;
        let in_range = |p: &[u16]| p.iter().all(|&s| s < ceil);
        if !in_range(&self.y) || !in_range(&self.u) || !in_range(&self.v) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// §6.4.1 `seq_profile` election for a (bit depth, chroma format)
/// pairing (see the module-docs table).
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] when no profile admits the
/// pairing: `bit_depth` outside `{8, 10, 12}`, or 8/10-bit 4:4:4
/// monochrome-style mismatches (profile 1 forbids monochrome — but
/// monochrome never carries the 4:4:4 layout here, so the only
/// invalid inputs are bad depths).
pub fn elect_seq_profile(bit_depth: u8, format: ChromaFormat) -> Result<u8, Error> {
    match (bit_depth, format) {
        (12, _) => Ok(2),
        (8 | 10, ChromaFormat::Yuv422) => Ok(2),
        (8 | 10, ChromaFormat::Yuv444) => Ok(1),
        (8 | 10, ChromaFormat::Yuv420 | ChromaFormat::Monochrome) => Ok(0),
        _ => Err(Error::PartitionWalkOutOfRange),
    }
}

/// §5.5.2 `color_config` synthesis for a (bit depth, chroma format)
/// pairing: unspecified colour description, studio range, `CSP
/// unknown`, `separate_uv_delta_q = 0` — only the depth / layout /
/// plane-count fields vary.
#[must_use]
pub fn color_config_for(bit_depth: u8, format: ChromaFormat) -> ColorConfig {
    let (ssx, ssy) = format.subsampling();
    ColorConfig {
        high_bitdepth: bit_depth >= 10,
        twelve_bit: bit_depth == 12,
        bit_depth,
        mono_chrome: format == ChromaFormat::Monochrome,
        num_planes: format.num_planes(),
        color_description_present_flag: false,
        color_primaries: crate::sequence_header::CP_UNSPECIFIED,
        transfer_characteristics: crate::sequence_header::TC_UNSPECIFIED,
        matrix_coefficients: crate::sequence_header::MC_UNSPECIFIED,
        color_range: false,
        subsampling_x: ssx == 1,
        subsampling_y: ssy == 1,
        chroma_sample_position: CSP_UNKNOWN,
        separate_uv_delta_q: false,
    }
}

/// General-format sibling of
/// [`build_intra_only_yuv420_8bit_seq`]: the same
/// minimal intra-capable `SequenceHeader` shape with `seq_profile` +
/// §5.5.2 `color_config` elected from `(bit_depth, format)`.
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] when [`elect_seq_profile`]
/// rejects the pairing.
pub fn build_intra_only_seq_yuv(
    max_width: u32,
    max_height: u32,
    bit_depth: u8,
    format: ChromaFormat,
) -> Result<SequenceHeader, Error> {
    let mut seq = build_intra_only_yuv420_8bit_seq(max_width, max_height);
    seq.seq_profile = elect_seq_profile(bit_depth, format)?;
    seq.color_config = color_config_for(bit_depth, format);
    Ok(seq)
}

// ----------------------------------------------------------------------
// r428 — shared frame/sequence scaffolding relocated from the retired
// encoder-mirror `pixel_driver_dyn` module: the legacy 8-bit 4:2:0
// input carrier and the minimal SH / FH synthesis every
// conformance-grade driver seeds its headers from. The mirror emit
// arms themselves are gone (see the r428 CHANGELOG entry); these
// items were always the conformance encoders' property.
// ----------------------------------------------------------------------

/// Historical lower dimension bound of the legacy dyn driver; the
/// conformance-grade encoders share the same floor (dimensions are
/// multiples of 8).
pub const MIN_DIM: u32 = 8;

/// Historical upper dimension bound of the legacy dyn driver
/// ([`Yuv420Frame::validate`] keeps enforcing it for back-compat; the
/// conformance-grade encoders accept up to
/// [`crate::encoder::key_frame::KEY_FRAME_MAX_DIM`] via
/// [`YuvFrame::validate`]).
pub const MAX_DIM: u32 = 64;

/// §5.11.1 superblock step in 4×4 units at the 64×64 superblock size.
pub const SB_SIZE4_64: u32 = 16;

/// Dynamic-extent 4:2:0 8-bit YUV input — the historical input type
/// of the 8-bit `*_yuv420` encoder entry points. Plane data is
/// Vec-backed so the same struct admits any allowed (width, height)
/// combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420Frame {
    /// Luma width in pixels.
    pub width: u32,
    /// Luma height in pixels.
    pub height: u32,
    /// Luma plane (`Y`), row-major, length `width * height`.
    pub y: Vec<u8>,
    /// First chroma plane (`U` / `Cb`) at half horizontal + vertical
    /// resolution; length `(width / 2) * (height / 2)`.
    pub u: Vec<u8>,
    /// Second chroma plane (`V` / `Cr`); same shape as `u`.
    pub v: Vec<u8>,
}

impl Yuv420Frame {
    /// All-`fill` input. Useful for tests + as the default
    /// constructor; every plane is set to the same value.
    #[must_use]
    pub fn filled(width: u32, height: u32, fill: u8) -> Self {
        let cw = width / 2;
        let ch = height / 2;
        Self {
            width,
            height,
            y: vec![fill; (width * height) as usize],
            u: vec![fill; (cw * ch) as usize],
            v: vec![fill; (cw * ch) as usize],
        }
    }

    /// Chroma plane width — `width / 2` per the 4:2:0 sampling pattern.
    #[must_use]
    pub fn chroma_width(&self) -> u32 {
        self.width / 2
    }

    /// Chroma plane height — `height / 2` per the 4:2:0 sampling
    /// pattern.
    #[must_use]
    pub fn chroma_height(&self) -> u32 {
        self.height / 2
    }

    /// Validate the input's dimensions + plane lengths against the
    /// HISTORICAL dyn-driver bounds (`[8, 64]` multiples of 8) —
    /// kept byte-for-byte for API stability. The conformance-grade
    /// entry points do NOT consult this; they validate the widened
    /// [`YuvFrame`] instead (up to
    /// [`crate::encoder::key_frame::KEY_FRAME_MAX_DIM`]).
    pub fn validate(&self) -> Result<(), Error> {
        if self.width < MIN_DIM
            || self.height < MIN_DIM
            || self.width > MAX_DIM
            || self.height > MAX_DIM
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.width % MIN_DIM != 0 || self.height % MIN_DIM != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let expected_y = (self.width * self.height) as usize;
        let expected_uv = (self.chroma_width() * self.chroma_height()) as usize;
        if self.y.len() != expected_y || self.u.len() != expected_uv || self.v.len() != expected_uv
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// §5.5.1 `OrderHintBits` every encoder-built sequence header carries
/// (r413): `order_hint_bits_minus_1 = 6`. Seven bits give a modulus of
/// 128 — [`crate::encoder::inter_frame::GOP_MAX_FRAMES`] = 64 output
/// hints per GOP never wrap, so §7.4 `get_relative_dist` orders every
/// in-GOP pair correctly.
pub const ENCODER_ORDER_HINT_BITS: u8 = 7;

/// Build the minimal 4:2:0 8-bit `SequenceHeader` for a frame whose
/// maximum extent is `(max_width, max_height)`. The resulting
/// `sequence_header_obu()` payload accepts a frame of any
/// width × height ≤ `(max_width, max_height)` — the per-frame size is
/// carried by the FrameHeader's `frame_size`.
#[must_use]
pub fn build_intra_only_yuv420_8bit_seq(max_width: u32, max_height: u32) -> SequenceHeader {
    debug_assert!((1..=0xFFFF).contains(&max_width));
    debug_assert!((1..=0xFFFF).contains(&max_height));
    let max_w_minus_1 = max_width - 1;
    let max_h_minus_1 = max_height - 1;
    // §5.5.1 frame_width_bits_minus_1 must be wide enough to fit
    // `max_frame_width_minus_1`. The minimum bit count is
    // `bits_to_represent(max_w_minus_1) - 1`. Use 0..=15.
    let bits_for = |v: u32| -> u8 {
        if v == 0 {
            0
        } else {
            (32 - v.leading_zeros() - 1) as u8
        }
    };
    let frame_width_bits_minus_1 = bits_for(max_w_minus_1);
    let frame_height_bits_minus_1 = bits_for(max_h_minus_1);

    SequenceHeader {
        seq_profile: 0,
        still_picture: false,
        reduced_still_picture_header: false,
        timing_info_present_flag: false,
        timing_info: None,
        decoder_model_info_present_flag: false,
        decoder_model_info: None,
        initial_display_delay_present_flag: false,
        operating_points_cnt_minus_1: 0,
        operating_points: vec![OperatingPoint {
            operating_point_idc: 0,
            seq_level_idx: 0,
            seq_tier: 0,
            decoder_model_present_for_this_op: false,
            operating_parameters_info: None,
            initial_display_delay_present_for_this_op: false,
            initial_display_delay_minus_1: None,
        }],
        frame_width_bits_minus_1,
        frame_height_bits_minus_1,
        max_frame_width_minus_1: max_w_minus_1,
        max_frame_height_minus_1: max_h_minus_1,
        frame_id_numbers_present_flag: false,
        delta_frame_id_length_minus_2: 0,
        additional_frame_id_length_minus_1: 0,
        // use_128x128_superblock = false ⇒ sb_size = 64 — the
        // conformance-grade drivers walk 64×64 superblocks.
        use_128x128_superblock: false,
        enable_filter_intra: false,
        enable_intra_edge_filter: false,
        // r417: inter-intra compound rides every stream this builder
        // seeds — single-reference 8x8..32x32 inter leaves code the
        // §5.11.28 cascade and may select smooth / wedge inter-intra
        // blends (intra-only frames are unaffected).
        enable_interintra_compound: true,
        // r415: masked compound rides every stream this builder seeds
        // — inter GOP compound leaves code the §5.11.29
        // `comp_group_idx` cascade and may select COMPOUND_WEDGE /
        // COMPOUND_DIFFWTD (intra-only frames are unaffected).
        enable_masked_compound: true,
        // r419: warped motion rides every stream this builder seeds —
        // inter frames code `allow_warped_motion = 1`, eligible
        // single-reference leaves code the §5.11.27 arm-B 3-way
        // `motion_mode` S(), and the RD ladder may commit
        // WARPED_CAUSAL winners (intra-only frames are unaffected).
        enable_warped_motion: true,
        enable_dual_filter: false,
        // r413: §5.5.1 order hints ride every stream this builder
        // seeds (KEY-only and GOP alike) — the §5.9.22
        // `skip_mode_params()` derivation and the §7.9 motion-field
        // groundwork both key off `OrderHintBits > 0`. Intra-only
        // frames simply carry `order_hint = 0`.
        enable_order_hint: true,
        // r416: jnt-comp rides every stream this builder seeds — inter
        // GOP compound leaves code the §5.11.29 `compound_idx` S() and
        // may select the §7.11.3.15 COMPOUND_DISTANCE blend
        // (intra-only frames are unaffected).
        enable_jnt_comp: true,
        // r413: temporal MV prediction — GOP P-frames run §7.9
        // motion-field estimation (`use_ref_frame_mvs = 1`); the
        // seq-level gate must be open.
        enable_ref_frame_mvs: true,
        seq_force_screen_content_tools: SELECT_SCREEN_CONTENT_TOOLS,
        seq_force_integer_mv: SELECT_INTEGER_MV,
        order_hint_bits: ENCODER_ORDER_HINT_BITS,
        enable_superres: false,
        // r428 — the frame-level §5.9.19/§7.15 CDEF election needs
        // the sequence gate open; frames that elect nothing code the
        // all-zero strength set (identity filter, ~2 header bytes).
        enable_cdef: true,
        enable_restoration: false,
        color_config: ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            mono_chrome: false,
            num_planes: 3,
            color_description_present_flag: false,
            color_primaries: CP_UNSPECIFIED,
            transfer_characteristics: TC_UNSPECIFIED,
            matrix_coefficients: MC_UNSPECIFIED,
            color_range: false,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: CSP_UNKNOWN,
            separate_uv_delta_q: false,
        },
        film_grain_params_present: false,
        bits_consumed: 0,
    }
}

/// Build the minimal intra-only `FrameHeader` at `base_q_idx = 0`
/// (lossless), `Only4x4` TxMode, in-loop filters disabled.
///
/// Equivalent to [`build_intra_only_yuv420_8bit_fh_with_q`] with
/// `base_q_idx = 0`; kept for back-compat with historical callers.
#[must_use]
pub fn build_intra_only_yuv420_8bit_fh(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
) -> FrameHeader {
    build_intra_only_yuv420_8bit_fh_with_q(seq, width, height, 0)
}

/// Build the minimal intra-only `FrameHeader` at the caller-supplied
/// `base_q_idx`. `base_q_idx == 0` is the §5.9.2 `CodedLossless` arm
/// (lossless WHT path on the leaf transform); `base_q_idx > 0`
/// selects the lossy path. The conformance-grade KEY / inter drivers
/// seed their headers from this shape and then override the fields
/// their configuration needs.
#[must_use]
pub fn build_intra_only_yuv420_8bit_fh_with_q(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> FrameHeader {
    let fs = FrameSize {
        frame_width: width,
        frame_height: height,
        render_width: width,
        render_height: height,
        superres_denom: 8, // SUPERRES_NUM
        upscaled_width: width,
        mi_cols: 2 * ((width + 7) >> 3),
        mi_rows: 2 * ((height + 7) >> 3),
        use_superres: false,
        coded_denom: 0,
        render_and_frame_size_different: false,
    };
    let ti = TileInfo {
        uniform_tile_spacing_flag: true,
        tile_cols: 1,
        tile_rows: 1,
        tile_cols_log2: 0,
        tile_rows_log2: 0,
        context_update_tile_id: 0,
        tile_size_bytes: 1,
        mi_col_starts: vec![0, fs.mi_cols],
        mi_row_starts: vec![0, fs.mi_rows],
    };
    let qp = QuantizationParams {
        base_q_idx,
        delta_q_y_dc: 0,
        diff_uv_delta: false,
        delta_q_u_dc: 0,
        delta_q_u_ac: 0,
        delta_q_v_dc: 0,
        delta_q_v_ac: 0,
        using_qmatrix: false,
        qm_y: 0,
        qm_u: 0,
        qm_v: 0,
    };
    FrameHeader {
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        display_frame_id: None,
        frame_type: FrameType::Key,
        frame_is_intra: true,
        show_frame: true,
        showable_frame: false,
        error_resilient_mode: true,
        disable_cdf_update: false,
        allow_screen_content_tools: seq.seq_force_screen_content_tools != 0,
        force_integer_mv: true,
        current_frame_id: 0,
        frame_size_override_flag: false,
        order_hint: 0,
        primary_ref_frame: PRIMARY_REF_NONE,
        refresh_frame_flags: ALL_FRAMES_PUB,
        ref_order_hints: None,
        frame_size: Some(fs),
        allow_intrabc: false,
        disable_frame_end_update_cdf: false,
        tile_info: Some(ti),
        quantization_params: Some(qp),
        segmentation_params: Some(SegmentationParams::disabled()),
        delta_q_params: Some(DeltaQParams::default()),
        delta_lf_params: Some(DeltaLfParams::default()),
        loop_filter_params: Some(LoopFilterParams::short_circuit()),
        // §5.9.19: the short-circuit shape ONLY where the parser
        // short-circuits (CodedLossless || allow_intrabc ||
        // !enable_cdef — this builder emits allow_intrabc = 0);
        // otherwise the block is CODED and defaults to the all-zero
        // strength set (identity filter) until the r428 election
        // lands a winner.
        cdef_params: Some(if base_q_idx == 0 || !seq.enable_cdef {
            CdefParams::short_circuit()
        } else {
            CdefParams {
                short_circuited: false,
                ..CdefParams::short_circuit()
            }
        }),
        lr_params: Some(LrParams {
            frame_restoration_type: [FrameRestorationType::None; 3],
            uses_lr: false,
            uses_chroma_lr: false,
            lr_unit_shift: 0,
            lr_uv_shift: 0,
            loop_restoration_size: [0; 3],
            short_circuited: true,
        }),
        // §5.9.21 `read_tx_mode()`: `ONLY_4X4` is only valid under the
        // §5.9.2 `CodedLossless` arm (`base_q_idx == 0` + all delta_q
        // zero). For lossy quant the drivers override to
        // `TxModeSelect`; `TxModeLargest` is the historical default.
        tx_mode: Some(if base_q_idx == 0 {
            TxMode::Only4x4
        } else {
            TxMode::TxModeLargest
        }),
        reference_select: Some(false),
        skip_mode_present: Some(false),
        skip_mode_frame: None,
        allow_warped_motion: Some(false),
        reduced_tx_set: Some(false),
        global_motion_params: Some(GlobalMotionParams::identity()),
        film_grain_params: Some(FilmGrainParams::reset()),
        inter_refs: None,
        bits_consumed: 0,
    }
}

/// §5.11.1 SB-grid traversal order: every 64×64 superblock origin in
/// 4×4-unit coordinates, row-major.
///
/// Equivalent to the spec's literal:
/// ```text
/// for (r = MiRowStart; r < MiRowEnd; r += sbSize4) {
///     for (c = MiColStart; c < MiColEnd; c += sbSize4) {
///         decode_partition(r, c, sbSize)
///     }
/// }
/// ```
/// with `MiRowStart = MiColStart = 0`, `MiRowEnd = mi_rows`,
/// `MiColEnd = mi_cols`, and `sbSize4 = SB_SIZE4_64 = 16`.
#[must_use]
pub fn sb_grid_origins(mi_rows: u32, mi_cols: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut r = 0u32;
    while r < mi_rows {
        let mut c = 0u32;
        while c < mi_cols {
            out.push((r, c));
            c += SB_SIZE4_64;
        }
        r += SB_SIZE4_64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_header::parse_sequence_header;

    #[test]
    fn profile_election_matches_the_6_4_1_table() {
        assert_eq!(elect_seq_profile(8, ChromaFormat::Yuv420).unwrap(), 0);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Yuv420).unwrap(), 0);
        assert_eq!(elect_seq_profile(8, ChromaFormat::Monochrome).unwrap(), 0);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Monochrome).unwrap(), 0);
        assert_eq!(elect_seq_profile(8, ChromaFormat::Yuv444).unwrap(), 1);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Yuv444).unwrap(), 1);
        assert_eq!(elect_seq_profile(8, ChromaFormat::Yuv422).unwrap(), 2);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Yuv422).unwrap(), 2);
        for fmt in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            assert_eq!(elect_seq_profile(12, fmt).unwrap(), 2);
        }
        assert!(elect_seq_profile(9, ChromaFormat::Yuv420).is_err());
        assert!(elect_seq_profile(16, ChromaFormat::Yuv444).is_err());
    }

    /// Every admissible pairing's synthesized sequence header must
    /// round-trip through the §5.5 parser with the exact depth /
    /// subsampling / plane-count fields.
    #[test]
    fn general_seq_builder_round_trips_every_pairing() {
        let pairings: &[(u8, ChromaFormat)] = &[
            (8, ChromaFormat::Yuv420),
            (10, ChromaFormat::Yuv420),
            (12, ChromaFormat::Yuv420),
            (8, ChromaFormat::Yuv422),
            (10, ChromaFormat::Yuv422),
            (12, ChromaFormat::Yuv422),
            (8, ChromaFormat::Yuv444),
            (10, ChromaFormat::Yuv444),
            (12, ChromaFormat::Yuv444),
            (8, ChromaFormat::Monochrome),
            (10, ChromaFormat::Monochrome),
            (12, ChromaFormat::Monochrome),
        ];
        for &(bd, fmt) in pairings {
            let seq = build_intra_only_seq_yuv(128, 96, bd, fmt).unwrap();
            let payload = crate::encoder::sequence_obu::write_sequence_header_obu(&seq);
            let parsed = parse_sequence_header(&payload).unwrap();
            let (ssx, ssy) = fmt.subsampling();
            assert_eq!(parsed.seq_profile, elect_seq_profile(bd, fmt).unwrap());
            assert_eq!(parsed.color_config.bit_depth, bd, "{bd} {fmt:?}");
            assert_eq!(
                parsed.color_config.mono_chrome,
                fmt == ChromaFormat::Monochrome
            );
            assert_eq!(parsed.color_config.num_planes, fmt.num_planes());
            assert_eq!(parsed.color_config.subsampling_x, ssx == 1, "{bd} {fmt:?}");
            assert_eq!(parsed.color_config.subsampling_y, ssy == 1, "{bd} {fmt:?}");
        }
    }

    #[test]
    fn yuv_frame_validation_rejects_bad_shapes() {
        // Good frames validate.
        for fmt in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            for bd in [8u8, 10, 12] {
                let f = YuvFrame::filled(64, 32, bd, fmt, (1 << bd) - 1);
                f.validate().unwrap_or_else(|e| {
                    panic!("filled({bd}, {fmt:?}) must validate: {e:?}");
                });
            }
        }
        // Sample over range.
        let mut f = YuvFrame::filled(8, 8, 10, ChromaFormat::Yuv444, 0);
        f.y[3] = 1 << 10;
        assert!(f.validate().is_err());
        // Chroma length mismatch (4:2:2 chroma is width/2 × FULL height).
        let mut f = YuvFrame::filled(16, 16, 8, ChromaFormat::Yuv422, 0);
        assert_eq!(f.u.len(), 8 * 16);
        f.u.truncate(8 * 8);
        assert!(f.validate().is_err());
        // Monochrome must carry empty chroma.
        let mut f = YuvFrame::filled(16, 16, 8, ChromaFormat::Monochrome, 0);
        assert!(f.u.is_empty() && f.v.is_empty());
        f.u = vec![0; 64];
        assert!(f.validate().is_err());
        // Bad depth / bad extent.
        let f = YuvFrame::filled(16, 16, 9, ChromaFormat::Yuv420, 0);
        assert!(f.validate().is_err());
        let f = YuvFrame::filled(12, 16, 8, ChromaFormat::Yuv420, 0);
        assert!(f.validate().is_err());
    }

    #[test]
    fn widening_from_yuv420_8bit_is_lossless() {
        let mut base = Yuv420Frame::filled(16, 8, 0);
        for (i, s) in base.y.iter_mut().enumerate() {
            *s = (i % 251) as u8;
        }
        for (i, s) in base.u.iter_mut().enumerate() {
            *s = (i % 249) as u8;
        }
        for (i, s) in base.v.iter_mut().enumerate() {
            *s = (i % 247) as u8;
        }
        let wide = YuvFrame::from_yuv420_8bit(&base);
        wide.validate().unwrap();
        assert_eq!(wide.bit_depth, 8);
        assert_eq!(wide.format, ChromaFormat::Yuv420);
        assert!(wide.y.iter().zip(&base.y).all(|(&w, &b)| w == u16::from(b)));
        assert!(wide.u.iter().zip(&base.u).all(|(&w, &b)| w == u16::from(b)));
        assert!(wide.v.iter().zip(&base.v).all(|(&w, &b)| w == u16::from(b)));
    }
}
