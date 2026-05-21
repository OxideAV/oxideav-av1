//! Sequence header OBU parser.
//!
//! Implements `sequence_header_obu()` per §5.5 of the AV1 Bitstream &
//! Decoding Process Specification:
//!
//!   * §5.5.1 — General sequence header OBU syntax
//!   * §5.5.2 — Color config syntax
//!   * §5.5.3 — Timing info syntax
//!   * §5.5.4 — Decoder model info syntax
//!   * §5.5.5 — Operating parameters info syntax
//!
//! Semantics references:
//!
//!   * §6.4.1 — General sequence header OBU semantics
//!   * §6.4.2 — Color config semantics
//!   * §6.4.3 — Timing info semantics
//!   * §6.4.4 — Decoder model info semantics
//!
//! Constants used (§3 Symbols and abbreviated terms):
//!
//!   * `SELECT_SCREEN_CONTENT_TOOLS = 2`
//!   * `SELECT_INTEGER_MV = 2`
//!   * `CP_UNSPECIFIED = 2`
//!   * `TC_UNSPECIFIED = 2`
//!   * `MC_UNSPECIFIED = 2`
//!   * `CSP_UNKNOWN = 0`
//!   * `CP_BT_709 = 1`, `TC_SRGB = 13`, `MC_IDENTITY = 0`
//!
//! Scope for this round: parse the sequence header into a strongly
//! typed [`SequenceHeader`] descriptor. We do **not** consume the
//! `trailing_bits` (§5.3.4): the OBU walker's caller already framed
//! the OBU; the trailing bits inside the payload are part of the
//! payload slice but the parser returns the bit count it consumed so
//! a future round (frame header, byte-aligned tile groups) can act on
//! the difference.

use crate::bitreader::BitReader;
use crate::Error;

// ---------------------------------------------------------------------
// §3 constants
// ---------------------------------------------------------------------

/// `SELECT_SCREEN_CONTENT_TOOLS` (§3): sentinel for
/// `seq_force_screen_content_tools` indicating the decision is made
/// per-frame via `allow_screen_content_tools`.
pub const SELECT_SCREEN_CONTENT_TOOLS: u8 = 2;

/// `SELECT_INTEGER_MV` (§3): sentinel for `seq_force_integer_mv`
/// indicating the decision is made per-frame via `force_integer_mv`.
pub const SELECT_INTEGER_MV: u8 = 2;

/// `CP_UNSPECIFIED` (§6.4.2 colour primaries table).
pub const CP_UNSPECIFIED: u8 = 2;
/// `TC_UNSPECIFIED` (§6.4.2 transfer characteristics table).
pub const TC_UNSPECIFIED: u8 = 2;
/// `MC_UNSPECIFIED` (§6.4.2 matrix coefficients table).
pub const MC_UNSPECIFIED: u8 = 2;
/// `CSP_UNKNOWN` (§6.4.2 chroma sample position table).
pub const CSP_UNKNOWN: u8 = 0;
const CP_BT_709: u8 = 1;
const TC_SRGB: u8 = 13;
const MC_IDENTITY: u8 = 0;

// ---------------------------------------------------------------------
// Descriptor types
// ---------------------------------------------------------------------

/// `timing_info()` per §5.5.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimingInfo {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    /// Present iff `equal_picture_interval == true`.
    pub num_ticks_per_picture_minus_1: Option<u32>,
}

/// `decoder_model_info()` per §5.5.4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderModelInfo {
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
}

/// `operating_parameters_info(op)` per §5.5.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatingParametersInfo {
    pub decoder_buffer_delay: u64,
    pub encoder_buffer_delay: u64,
    pub low_delay_mode_flag: bool,
}

/// One operating point as listed inside §5.5.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperatingPoint {
    pub operating_point_idc: u16, // 12 bits
    pub seq_level_idx: u8,        // 5 bits
    pub seq_tier: u8,             // 0 or 1
    pub decoder_model_present_for_this_op: bool,
    pub operating_parameters_info: Option<OperatingParametersInfo>,
    pub initial_display_delay_present_for_this_op: bool,
    /// `Some` iff `initial_display_delay_present_for_this_op` is set.
    pub initial_display_delay_minus_1: Option<u8>, // 4 bits
}

/// `color_config()` per §5.5.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorConfig {
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    /// Derived per §5.5.2: 8 / 10 / 12.
    pub bit_depth: u8,
    pub mono_chrome: bool,
    /// Derived: 1 if `mono_chrome`, else 3.
    pub num_planes: u8,
    pub color_description_present_flag: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

/// `sequence_header_obu()` per §5.5.1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceHeader {
    pub seq_profile: u8, // 3 bits, 0..=7 (0..=2 are the conformant values)
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info_present_flag: bool,
    pub timing_info: Option<TimingInfo>,
    pub decoder_model_info_present_flag: bool,
    pub decoder_model_info: Option<DecoderModelInfo>,
    pub initial_display_delay_present_flag: bool,
    pub operating_points_cnt_minus_1: u8, // 5 bits
    pub operating_points: Vec<OperatingPoint>,
    pub frame_width_bits_minus_1: u8,  // 4 bits
    pub frame_height_bits_minus_1: u8, // 4 bits
    pub max_frame_width_minus_1: u32,
    pub max_frame_height_minus_1: u32,
    pub frame_id_numbers_present_flag: bool,
    pub delta_frame_id_length_minus_2: u8,      // 4 bits
    pub additional_frame_id_length_minus_1: u8, // 3 bits
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    pub seq_force_screen_content_tools: u8,
    pub seq_force_integer_mv: u8,
    pub order_hint_bits: u8, // 0 or order_hint_bits_minus_1+1 (1..=8)
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color_config: ColorConfig,
    pub film_grain_params_present: bool,
    /// Number of bits consumed from the OBU payload by this parse —
    /// useful for the §5.3.1 `trailing_bits` accounting a future round
    /// will wire up.
    pub bits_consumed: usize,
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

/// Parse a `sequence_header_obu()` payload per §5.5.1.
///
/// `payload` is the slice the OBU walker returned as
/// `ObuDescriptor::payload`, i.e. everything *after* the OBU header,
/// extension header, and `obu_size`. The function consumes the
/// `sequence_header_obu()` fields and returns the descriptor; it does
/// **not** read the trailing bits (§5.3.4) — the OBU walker frames
/// the unit and the caller can compute how many bits remain.
///
/// Errors:
///
///   * [`Error::UnexpectedEnd`] — payload ran out before the header
///     finished.
///   * [`Error::ReservedProfile`] — §6.4.1 conformance: `seq_profile`
///     was 3..=7 (reserved).
///   * [`Error::ReducedStillRequiresStill`] — §6.4.1 conformance:
///     `reduced_still_picture_header == 1` with `still_picture == 0`.
pub fn parse_sequence_header(payload: &[u8]) -> Result<SequenceHeader, Error> {
    let mut br = BitReader::new(payload);

    let seq_profile = br.f(3)? as u8;
    // §6.4.1 conformance: seq_profile <= 2.
    if seq_profile > 2 {
        return Err(Error::ReservedProfile(seq_profile));
    }
    let still_picture = br.f(1)? == 1;
    let reduced_still_picture_header = br.f(1)? == 1;
    // §6.4.1 conformance: reduced_still_picture_header == 1 implies
    // still_picture == 1.
    if reduced_still_picture_header && !still_picture {
        return Err(Error::ReducedStillRequiresStill);
    }

    let timing_info_present_flag;
    let mut timing_info = None;
    let mut decoder_model_info_present_flag = false;
    let mut decoder_model_info = None;
    let initial_display_delay_present_flag;
    let operating_points_cnt_minus_1;
    let mut operating_points: Vec<OperatingPoint> = Vec::new();

    if reduced_still_picture_header {
        timing_info_present_flag = false;
        initial_display_delay_present_flag = false;
        operating_points_cnt_minus_1 = 0;
        let seq_level_idx = br.f(5)? as u8;
        operating_points.push(OperatingPoint {
            operating_point_idc: 0,
            seq_level_idx,
            seq_tier: 0,
            decoder_model_present_for_this_op: false,
            operating_parameters_info: None,
            initial_display_delay_present_for_this_op: false,
            initial_display_delay_minus_1: None,
        });
    } else {
        timing_info_present_flag = br.f(1)? == 1;
        if timing_info_present_flag {
            let ti = parse_timing_info(&mut br)?;
            timing_info = Some(ti);
            decoder_model_info_present_flag = br.f(1)? == 1;
            if decoder_model_info_present_flag {
                decoder_model_info = Some(parse_decoder_model_info(&mut br)?);
            }
        }
        initial_display_delay_present_flag = br.f(1)? == 1;
        operating_points_cnt_minus_1 = br.f(5)? as u8;
        for _ in 0..=operating_points_cnt_minus_1 {
            let operating_point_idc = br.f(12)? as u16;
            let seq_level_idx = br.f(5)? as u8;
            let seq_tier = if seq_level_idx > 7 { br.f(1)? as u8 } else { 0 };
            let mut dec_model_present_for_this_op = false;
            let mut op_params = None;
            if decoder_model_info_present_flag {
                dec_model_present_for_this_op = br.f(1)? == 1;
                if dec_model_present_for_this_op {
                    let dmi = decoder_model_info.ok_or(Error::UnexpectedEnd)?;
                    op_params = Some(parse_operating_parameters_info(&mut br, &dmi)?);
                }
            }
            let mut idd_present_for_op = false;
            let mut idd_minus_1 = None;
            if initial_display_delay_present_flag {
                idd_present_for_op = br.f(1)? == 1;
                if idd_present_for_op {
                    idd_minus_1 = Some(br.f(4)? as u8);
                }
            }
            operating_points.push(OperatingPoint {
                operating_point_idc,
                seq_level_idx,
                seq_tier,
                decoder_model_present_for_this_op: dec_model_present_for_this_op,
                operating_parameters_info: op_params,
                initial_display_delay_present_for_this_op: idd_present_for_op,
                initial_display_delay_minus_1: idd_minus_1,
            });
        }
    }

    let frame_width_bits_minus_1 = br.f(4)? as u8;
    let frame_height_bits_minus_1 = br.f(4)? as u8;
    let n_w = u32::from(frame_width_bits_minus_1) + 1;
    let n_h = u32::from(frame_height_bits_minus_1) + 1;
    let max_frame_width_minus_1 = br.f(n_w)? as u32;
    let max_frame_height_minus_1 = br.f(n_h)? as u32;

    let frame_id_numbers_present_flag = if reduced_still_picture_header {
        false
    } else {
        br.f(1)? == 1
    };
    let mut delta_frame_id_length_minus_2 = 0u8;
    let mut additional_frame_id_length_minus_1 = 0u8;
    if frame_id_numbers_present_flag {
        delta_frame_id_length_minus_2 = br.f(4)? as u8;
        additional_frame_id_length_minus_1 = br.f(3)? as u8;
    }

    let use_128x128_superblock = br.f(1)? == 1;
    let enable_filter_intra = br.f(1)? == 1;
    let enable_intra_edge_filter = br.f(1)? == 1;

    let enable_interintra_compound;
    let enable_masked_compound;
    let enable_warped_motion;
    let enable_dual_filter;
    let enable_order_hint;
    let enable_jnt_comp;
    let enable_ref_frame_mvs;
    let seq_force_screen_content_tools;
    let seq_force_integer_mv;
    let order_hint_bits;

    if reduced_still_picture_header {
        enable_interintra_compound = false;
        enable_masked_compound = false;
        enable_warped_motion = false;
        enable_dual_filter = false;
        enable_order_hint = false;
        enable_jnt_comp = false;
        enable_ref_frame_mvs = false;
        seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        seq_force_integer_mv = SELECT_INTEGER_MV;
        order_hint_bits = 0;
    } else {
        enable_interintra_compound = br.f(1)? == 1;
        enable_masked_compound = br.f(1)? == 1;
        enable_warped_motion = br.f(1)? == 1;
        enable_dual_filter = br.f(1)? == 1;
        enable_order_hint = br.f(1)? == 1;
        if enable_order_hint {
            enable_jnt_comp = br.f(1)? == 1;
            enable_ref_frame_mvs = br.f(1)? == 1;
        } else {
            enable_jnt_comp = false;
            enable_ref_frame_mvs = false;
        }
        let seq_choose_screen_content_tools = br.f(1)? == 1;
        seq_force_screen_content_tools = if seq_choose_screen_content_tools {
            SELECT_SCREEN_CONTENT_TOOLS
        } else {
            br.f(1)? as u8
        };
        seq_force_integer_mv = if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = br.f(1)? == 1;
            if seq_choose_integer_mv {
                SELECT_INTEGER_MV
            } else {
                br.f(1)? as u8
            }
        } else {
            SELECT_INTEGER_MV
        };
        if enable_order_hint {
            let order_hint_bits_minus_1 = br.f(3)? as u8;
            order_hint_bits = order_hint_bits_minus_1 + 1;
        } else {
            order_hint_bits = 0;
        }
    }

    let enable_superres = br.f(1)? == 1;
    let enable_cdef = br.f(1)? == 1;
    let enable_restoration = br.f(1)? == 1;
    let color_config = parse_color_config(&mut br, seq_profile)?;
    let film_grain_params_present = br.f(1)? == 1;

    Ok(SequenceHeader {
        seq_profile,
        still_picture,
        reduced_still_picture_header,
        timing_info_present_flag,
        timing_info,
        decoder_model_info_present_flag,
        decoder_model_info,
        initial_display_delay_present_flag,
        operating_points_cnt_minus_1,
        operating_points,
        frame_width_bits_minus_1,
        frame_height_bits_minus_1,
        max_frame_width_minus_1,
        max_frame_height_minus_1,
        frame_id_numbers_present_flag,
        delta_frame_id_length_minus_2,
        additional_frame_id_length_minus_1,
        use_128x128_superblock,
        enable_filter_intra,
        enable_intra_edge_filter,
        enable_interintra_compound,
        enable_masked_compound,
        enable_warped_motion,
        enable_dual_filter,
        enable_order_hint,
        enable_jnt_comp,
        enable_ref_frame_mvs,
        seq_force_screen_content_tools,
        seq_force_integer_mv,
        order_hint_bits,
        enable_superres,
        enable_cdef,
        enable_restoration,
        color_config,
        film_grain_params_present,
        bits_consumed: br.position(),
    })
}

fn parse_timing_info(br: &mut BitReader<'_>) -> Result<TimingInfo, Error> {
    let num_units_in_display_tick = br.f(32)? as u32;
    let time_scale = br.f(32)? as u32;
    let equal_picture_interval = br.f(1)? == 1;
    let num_ticks_per_picture_minus_1 = if equal_picture_interval {
        Some(br.uvlc()?)
    } else {
        None
    };
    Ok(TimingInfo {
        num_units_in_display_tick,
        time_scale,
        equal_picture_interval,
        num_ticks_per_picture_minus_1,
    })
}

fn parse_decoder_model_info(br: &mut BitReader<'_>) -> Result<DecoderModelInfo, Error> {
    let buffer_delay_length_minus_1 = br.f(5)? as u8;
    let num_units_in_decoding_tick = br.f(32)? as u32;
    let buffer_removal_time_length_minus_1 = br.f(5)? as u8;
    let frame_presentation_time_length_minus_1 = br.f(5)? as u8;
    Ok(DecoderModelInfo {
        buffer_delay_length_minus_1,
        num_units_in_decoding_tick,
        buffer_removal_time_length_minus_1,
        frame_presentation_time_length_minus_1,
    })
}

fn parse_operating_parameters_info(
    br: &mut BitReader<'_>,
    dmi: &DecoderModelInfo,
) -> Result<OperatingParametersInfo, Error> {
    let n = u32::from(dmi.buffer_delay_length_minus_1) + 1;
    let decoder_buffer_delay = br.f(n)?;
    let encoder_buffer_delay = br.f(n)?;
    let low_delay_mode_flag = br.f(1)? == 1;
    Ok(OperatingParametersInfo {
        decoder_buffer_delay,
        encoder_buffer_delay,
        low_delay_mode_flag,
    })
}

fn parse_color_config(br: &mut BitReader<'_>, seq_profile: u8) -> Result<ColorConfig, Error> {
    let high_bitdepth = br.f(1)? == 1;
    let mut twelve_bit = false;
    let bit_depth = if seq_profile == 2 && high_bitdepth {
        twelve_bit = br.f(1)? == 1;
        if twelve_bit {
            12
        } else {
            10
        }
    } else {
        // §5.5.2: seq_profile <= 2 — guaranteed by parse_sequence_header.
        if high_bitdepth {
            10
        } else {
            8
        }
    };
    let mono_chrome = if seq_profile == 1 {
        false
    } else {
        br.f(1)? == 1
    };
    let num_planes = if mono_chrome { 1 } else { 3 };
    let color_description_present_flag = br.f(1)? == 1;
    let (color_primaries, transfer_characteristics, matrix_coefficients) =
        if color_description_present_flag {
            (br.f(8)? as u8, br.f(8)? as u8, br.f(8)? as u8)
        } else {
            (CP_UNSPECIFIED, TC_UNSPECIFIED, MC_UNSPECIFIED)
        };

    if mono_chrome {
        // §5.5.2 mono_chrome path: read color_range, derive
        // subsampling=1, return early (separate_uv_delta_q stays 0).
        let color_range = br.f(1)? == 1;
        return Ok(ColorConfig {
            high_bitdepth,
            twelve_bit,
            bit_depth,
            mono_chrome,
            num_planes,
            color_description_present_flag,
            color_primaries,
            transfer_characteristics,
            matrix_coefficients,
            color_range,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: CSP_UNKNOWN,
            separate_uv_delta_q: false,
        });
    }

    let color_range;
    let subsampling_x;
    let subsampling_y;
    let mut chroma_sample_position = CSP_UNKNOWN;

    if color_primaries == CP_BT_709
        && transfer_characteristics == TC_SRGB
        && matrix_coefficients == MC_IDENTITY
    {
        color_range = true;
        subsampling_x = false;
        subsampling_y = false;
    } else {
        color_range = br.f(1)? == 1;
        if seq_profile == 0 {
            subsampling_x = true;
            subsampling_y = true;
        } else if seq_profile == 1 {
            // YUV 4:4:4.
            subsampling_x = false;
            subsampling_y = false;
        } else {
            // seq_profile == 2.
            if bit_depth == 12 {
                subsampling_x = br.f(1)? == 1;
                subsampling_y = if subsampling_x { br.f(1)? == 1 } else { false };
            } else {
                subsampling_x = true;
                subsampling_y = false;
            }
        }
        if subsampling_x && subsampling_y {
            chroma_sample_position = br.f(2)? as u8;
        }
    }
    let separate_uv_delta_q = br.f(1)? == 1;
    Ok(ColorConfig {
        high_bitdepth,
        twelve_bit,
        bit_depth,
        mono_chrome,
        num_planes,
        color_description_present_flag,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        color_range,
        subsampling_x,
        subsampling_y,
        chroma_sample_position,
        separate_uv_delta_q,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SEQUENCE_HEADER OBU payload from
    /// `docs/video/av1/fixtures/tiny-i-only-16x16-prof0/input.ivf`.
    ///
    /// IVF layout: 32-byte file header + 12-byte frame header before
    /// the OBU stream. First two OBUs are the TD (`12 00`) and the
    /// SEQ_HEADER (`0a 0a <10 bytes payload>`):
    ///
    ///   12 00                     -- TD: type=2 has_size=1 obu_size=0
    ///   0a 0a 00 00 00 01 9f fb   -- SEQ_HEADER header + size byte +
    ///   ff f3 00 80               -- 10-byte payload
    ///
    /// Trace expects:
    ///   profile=0 still=0 reduced=0 max_w=16 max_h=16
    ///   level0=0 tier0=0 num_ops=1 use_128sb=1
    ///   enable_filter_intra=1 enable_intra_edge_filter=1
    ///   enable_interintra=1 enable_masked=1 enable_warped=1
    ///   enable_dual_filter=1 enable_order_hint=1 order_hint_bits=7
    ///   enable_jnt_comp=1 enable_ref_mvs=1 enable_superres=0
    ///   enable_cdef=1 enable_restoration=1 monochrome=0
    ///   high_bitdepth=0 twelve_bit=0 color_range=0
    ///   subsampling_x=1 subsampling_y=1 film_grain_present=0
    const TINY_16X16_PROF0: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];

    #[test]
    fn parses_tiny_16x16_profile0_fixture() {
        let sh = parse_sequence_header(TINY_16X16_PROF0).expect("decodes");
        assert_eq!(sh.seq_profile, 0);
        assert!(!sh.still_picture);
        assert!(!sh.reduced_still_picture_header);
        assert!(!sh.timing_info_present_flag);
        assert!(!sh.decoder_model_info_present_flag);
        assert!(!sh.initial_display_delay_present_flag);
        assert_eq!(sh.operating_points_cnt_minus_1, 0);
        assert_eq!(sh.operating_points.len(), 1);
        let op = &sh.operating_points[0];
        assert_eq!(op.operating_point_idc, 0);
        assert_eq!(op.seq_level_idx, 0);
        assert_eq!(op.seq_tier, 0);
        assert!(!op.decoder_model_present_for_this_op);
        assert!(!op.initial_display_delay_present_for_this_op);
        // max_w=16 => max_frame_width_minus_1 = 15.
        assert_eq!(sh.max_frame_width_minus_1, 15);
        assert_eq!(sh.max_frame_height_minus_1, 15);
        assert!(!sh.frame_id_numbers_present_flag);
        assert!(sh.use_128x128_superblock);
        assert!(sh.enable_filter_intra);
        assert!(sh.enable_intra_edge_filter);
        assert!(sh.enable_interintra_compound);
        assert!(sh.enable_masked_compound);
        assert!(sh.enable_warped_motion);
        assert!(sh.enable_dual_filter);
        assert!(sh.enable_order_hint);
        assert!(sh.enable_jnt_comp);
        assert!(sh.enable_ref_frame_mvs);
        assert_eq!(sh.order_hint_bits, 7);
        assert!(!sh.enable_superres);
        assert!(sh.enable_cdef);
        assert!(sh.enable_restoration);
        assert_eq!(sh.color_config.bit_depth, 8);
        assert!(!sh.color_config.mono_chrome);
        assert_eq!(sh.color_config.num_planes, 3);
        assert!(!sh.color_config.color_description_present_flag);
        assert_eq!(sh.color_config.color_primaries, CP_UNSPECIFIED);
        assert_eq!(sh.color_config.transfer_characteristics, TC_UNSPECIFIED);
        assert_eq!(sh.color_config.matrix_coefficients, MC_UNSPECIFIED);
        assert!(!sh.color_config.color_range);
        assert!(sh.color_config.subsampling_x);
        assert!(sh.color_config.subsampling_y);
        assert!(!sh.film_grain_params_present);
    }

    /// SEQUENCE_HEADER OBU payload from
    /// `docs/video/av1/fixtures/monochrome-grey-only/input.ivf`.
    ///
    /// Trace expects: profile=0 still=0 reduced=0 max_w=64 max_h=64
    /// monochrome=1 color_range=1 subsampling_x=1 subsampling_y=1
    /// (everything else same shape as tiny but max=64x64).
    const MONOCHROME_GREY_ONLY: &[u8] =
        &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x35, 0x40];

    #[test]
    fn parses_monochrome_fixture() {
        let sh = parse_sequence_header(MONOCHROME_GREY_ONLY).expect("decodes");
        assert_eq!(sh.seq_profile, 0);
        assert_eq!(sh.max_frame_width_minus_1, 63);
        assert_eq!(sh.max_frame_height_minus_1, 63);
        assert!(sh.color_config.mono_chrome);
        assert_eq!(sh.color_config.num_planes, 1);
        assert!(sh.color_config.color_range);
        assert!(sh.color_config.subsampling_x);
        assert!(sh.color_config.subsampling_y);
        assert!(sh.use_128x128_superblock);
        assert_eq!(sh.order_hint_bits, 7);
    }

    /// 12-bit 4:2:2 profile-2 fixture
    /// (`profile-2-yuv422-12bit/input.ivf`).
    ///
    /// Trace: profile=2 high_bitdepth=1 twelve_bit=1
    /// subsampling_x=1 subsampling_y=0 max_w=64 max_h=64.
    const PROFILE2_422_12BIT: &[u8] = &[
        0x40, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x3c, 0x44, 0x32, 0x1b, 0x14,
    ];

    #[test]
    fn parses_profile2_422_12bit_fixture() {
        let sh = parse_sequence_header(PROFILE2_422_12BIT).expect("decodes");
        assert_eq!(sh.seq_profile, 2);
        assert_eq!(sh.color_config.bit_depth, 12);
        assert!(sh.color_config.high_bitdepth);
        assert!(sh.color_config.twelve_bit);
        assert!(sh.color_config.subsampling_x);
        assert!(!sh.color_config.subsampling_y);
        assert!(!sh.color_config.mono_chrome);
        assert_eq!(sh.max_frame_width_minus_1, 63);
        assert_eq!(sh.max_frame_height_minus_1, 63);
    }

    // --- Unit / synthetic tests --------------------------------------

    #[test]
    fn rejects_reserved_profile() {
        // seq_profile=3 (top 3 bits of first byte = 0b011 — but spec
        // requires <=2). Build a byte starting with 0b011_xxxxx.
        let payload = [0b0110_0000u8];
        let err = parse_sequence_header(&payload).expect_err("profile 3 must be rejected");
        assert!(matches!(err, Error::ReservedProfile(3)));
    }

    #[test]
    fn rejects_reduced_still_without_still_picture() {
        // bits: profile(3)=0, still_picture(1)=0, reduced(1)=1, ...
        // First byte: 0b000_01_xxx = 0x08.
        let payload = [0x08u8, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let err = parse_sequence_header(&payload).expect_err("reduced w/o still must be rejected");
        assert!(matches!(err, Error::ReducedStillRequiresStill));
    }

    #[test]
    fn unexpected_end_on_truncated_payload() {
        // Need at least 5 bits to read profile+still+reduced and a few
        // bits more for level / max_w / etc. 1 byte is far too short
        // for any conformant sequence header.
        let payload = [0x00u8];
        let err = parse_sequence_header(&payload).expect_err("truncated payload");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    #[test]
    fn reduced_still_picture_path_minimal() {
        // Build a minimal reduced-still SH:
        //   profile=0(3b)=000
        //   still_picture=1(1b)
        //   reduced=1(1b)
        //   seq_level_idx[0]=0(5b)
        //   frame_width_bits_minus_1=0(4b) => n=1
        //   frame_height_bits_minus_1=0(4b) => n=1
        //   max_frame_width_minus_1(1b)=0
        //   max_frame_height_minus_1(1b)=0
        //   use_128x128_superblock(1b)=1
        //   enable_filter_intra(1b)=0
        //   enable_intra_edge_filter(1b)=0
        //   -- reduced path bypasses the inter / order-hint block --
        //   enable_superres(1b)=0
        //   enable_cdef(1b)=0
        //   enable_restoration(1b)=0
        //   color_config: high_bitdepth(1b)=0 (=> bit_depth=8)
        //                 mono_chrome(1b)=0
        //                 color_description_present_flag(1b)=0
        //                 [no color_primaries/etc.]
        //                 not mono / not srgb fast path => color_range(1b)=0
        //                 seq_profile==0 => sx=sy=1 implicit
        //                 sx && sy => chroma_sample_position(2b)=0
        //                 separate_uv_delta_q(1b)=0
        //   film_grain_params_present(1b)=0
        //
        // Total bits = 3+1+1+5+4+4+1+1+1+1+1+1+1+1+1+1+1+1+2+1+1 = 34
        // → packs into 5 bytes with 6 padding bits.
        //
        // Build bit string MSB-first:
        //   000 1 1 00000 0000 0000 0 0 1 0 0 0 0 0 0 0 0 0 00 0 0
        //   ^seq_profile=0 ^st=1 ^rd=1 ^level=0 ^fwbits=0 ^fhbits=0
        //   ^maxw=0 ^maxh=0 ^use128=1 ^efi=0 ^eief=0 ^superres=0
        //   ^cdef=0 ^rst=0 ^hbd=0 ^mono=0 ^cdpf=0 ^color_range=0
        //   ^csp=00 ^sep_uv=0 ^film_grain=0
        let bits = "0001100000000000000010000000000000";
        let payload = pack_bits_msb(bits);
        let sh = parse_sequence_header(&payload).expect("reduced-still SH decodes");
        assert!(sh.still_picture);
        assert!(sh.reduced_still_picture_header);
        assert_eq!(sh.operating_points_cnt_minus_1, 0);
        assert_eq!(sh.max_frame_width_minus_1, 0);
        assert_eq!(sh.max_frame_height_minus_1, 0);
        assert!(sh.use_128x128_superblock);
        assert!(!sh.enable_filter_intra);
        // Reduced path forces all inter-prediction enable bits off.
        assert!(!sh.enable_interintra_compound);
        assert!(!sh.enable_masked_compound);
        assert!(!sh.enable_warped_motion);
        assert!(!sh.enable_dual_filter);
        assert!(!sh.enable_order_hint);
        assert!(!sh.enable_jnt_comp);
        assert!(!sh.enable_ref_frame_mvs);
        assert_eq!(
            sh.seq_force_screen_content_tools,
            SELECT_SCREEN_CONTENT_TOOLS
        );
        assert_eq!(sh.seq_force_integer_mv, SELECT_INTEGER_MV);
        assert_eq!(sh.order_hint_bits, 0);
        assert_eq!(sh.color_config.bit_depth, 8);
        assert!(!sh.color_config.mono_chrome);
        assert_eq!(sh.color_config.num_planes, 3);
        assert!(sh.color_config.subsampling_x);
        assert!(sh.color_config.subsampling_y);
        assert_eq!(sh.color_config.chroma_sample_position, CSP_UNKNOWN);
    }

    /// Pack a string of '0' / '1' characters into bytes, MSB first
    /// within each byte. Padding bits at the end are set to 0.
    fn pack_bits_msb(bits: &str) -> Vec<u8> {
        let mut out = Vec::with_capacity(bits.len().div_ceil(8));
        let mut cur: u8 = 0;
        let mut n: u32 = 0;
        for ch in bits.chars() {
            cur = (cur << 1) | if ch == '1' { 1 } else { 0 };
            n += 1;
            if n == 8 {
                out.push(cur);
                cur = 0;
                n = 0;
            }
        }
        if n > 0 {
            cur <<= 8 - n;
            out.push(cur);
        }
        out
    }
}
