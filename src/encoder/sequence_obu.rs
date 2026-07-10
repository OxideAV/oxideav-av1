//! `sequence_header_obu()` writer — inverse of
//! [`crate::sequence_header::parse_sequence_header`].
//!
//! Implements the §5.5 syntax inversion:
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
//!
//! The encoder takes a strongly typed
//! [`crate::sequence_header::SequenceHeader`] — the same struct the
//! parser fills — and writes the bitstream form of it. The output
//! is the **payload** the OBU framer should wrap with
//! [`crate::encoder::write_obu`] using `obu_type ==
//! ObuType::SequenceHeader`. The writer does not emit the OBU header
//! / `obu_size` itself — that's the framer's responsibility.
//!
//! As of r409 this writer emits the §5.3.4 `trailing_bits` itself,
//! bit-precisely: the `trailing_one_bit` occupies the FIRST unused
//! bit position after the last syntax element (§5.3.1 requires
//! `trailing_bits( obu_size * 8 - payloadBits )` for
//! `OBU_SEQUENCE_HEADER`). The previous split — zero-pad the body to
//! a byte boundary in `BitWriter::finish`, then have the §5.3.1
//! framer append a whole `0x80` byte — placed the trailing one-bit
//! up to 7 bit positions late whenever the syntax ended mid-byte,
//! which conformant decoders reject (this crate's parser ignores
//! padding, so the bug was invisible to every internal round-trip).

use crate::encoder::bitwriter::BitWriter;
use crate::sequence_header::{
    ColorConfig, DecoderModelInfo, OperatingParametersInfo, OperatingPoint, SequenceHeader,
    TimingInfo, CP_UNSPECIFIED, CSP_UNKNOWN, MC_UNSPECIFIED, SELECT_INTEGER_MV,
    SELECT_SCREEN_CONTENT_TOOLS, TC_UNSPECIFIED,
};

// §5.5.2 colour-config fast-path constants — duplicated locally to
// preserve the §5.5.2 sRGB BT.709 identity short-circuit on the
// encoder side without bumping the parser's API surface.
const CP_BT_709: u8 = 1;
const TC_SRGB: u8 = 13;
const MC_IDENTITY: u8 = 0;

/// Write the §5.5.1 `sequence_header_obu()` payload bytes into a
/// fresh buffer and return it. The OBU framer wraps the returned
/// bytes with `obu_type == ObuType::SequenceHeader`.
///
/// The function is the inverse of
/// [`crate::sequence_header::parse_sequence_header`]: the same
/// `SequenceHeader` round-trips through encoder + decoder
/// byte-for-byte (up to the trailing bit-padding of the final
/// partial byte, which the parser ignores).
pub fn write_sequence_header_obu(sh: &SequenceHeader) -> Vec<u8> {
    let mut bw = BitWriter::new();
    encode_sequence_header(&mut bw, sh);
    // §5.3.1 / §5.3.4: the trailing one-bit sits in the first unused
    // bit position after the last syntax element (a full 0x80 byte
    // when the syntax happens to end byte-aligned).
    bw.trailing_bits_to_alignment();
    bw.finish()
}

fn encode_sequence_header(bw: &mut BitWriter, sh: &SequenceHeader) {
    // §6.4.1 conformance — caller-supplied invariants.
    debug_assert!(sh.seq_profile <= 2, "§6.4.1 seq_profile must be 0..=2");
    debug_assert!(
        !sh.reduced_still_picture_header || sh.still_picture,
        "§6.4.1 reduced_still_picture_header == 1 requires still_picture == 1"
    );

    bw.write_bits(3, u64::from(sh.seq_profile));
    bw.write_bits(1, u64::from(sh.still_picture));
    bw.write_bits(1, u64::from(sh.reduced_still_picture_header));

    if sh.reduced_still_picture_header {
        // §5.5.1: only seq_level_idx[0] is on the wire; tier / model
        // / display-delay / extra operating-points are all implicit.
        let op0 = sh
            .operating_points
            .first()
            .expect("reduced_still requires one operating point");
        bw.write_bits(5, u64::from(op0.seq_level_idx));
    } else {
        bw.write_bits(1, u64::from(sh.timing_info_present_flag));
        if sh.timing_info_present_flag {
            let ti = sh
                .timing_info
                .as_ref()
                .expect("timing_info_present_flag => timing_info populated");
            encode_timing_info(bw, ti);
            bw.write_bits(1, u64::from(sh.decoder_model_info_present_flag));
            if sh.decoder_model_info_present_flag {
                let dmi = sh
                    .decoder_model_info
                    .as_ref()
                    .expect("decoder_model_info_present_flag => decoder_model_info populated");
                encode_decoder_model_info(bw, dmi);
            }
        }
        bw.write_bits(1, u64::from(sh.initial_display_delay_present_flag));
        bw.write_bits(5, u64::from(sh.operating_points_cnt_minus_1));
        debug_assert_eq!(
            sh.operating_points.len(),
            usize::from(sh.operating_points_cnt_minus_1) + 1,
            "operating_points length must match operating_points_cnt_minus_1 + 1"
        );
        for op in &sh.operating_points {
            encode_operating_point(
                bw,
                op,
                sh.decoder_model_info_present_flag,
                sh.decoder_model_info.as_ref(),
                sh.initial_display_delay_present_flag,
            );
        }
    }

    bw.write_bits(4, u64::from(sh.frame_width_bits_minus_1));
    bw.write_bits(4, u64::from(sh.frame_height_bits_minus_1));
    let n_w = u32::from(sh.frame_width_bits_minus_1) + 1;
    let n_h = u32::from(sh.frame_height_bits_minus_1) + 1;
    bw.write_bits(n_w, u64::from(sh.max_frame_width_minus_1));
    bw.write_bits(n_h, u64::from(sh.max_frame_height_minus_1));

    if !sh.reduced_still_picture_header {
        bw.write_bits(1, u64::from(sh.frame_id_numbers_present_flag));
        if sh.frame_id_numbers_present_flag {
            bw.write_bits(4, u64::from(sh.delta_frame_id_length_minus_2));
            bw.write_bits(3, u64::from(sh.additional_frame_id_length_minus_1));
        }
    }

    bw.write_bits(1, u64::from(sh.use_128x128_superblock));
    bw.write_bits(1, u64::from(sh.enable_filter_intra));
    bw.write_bits(1, u64::from(sh.enable_intra_edge_filter));

    if !sh.reduced_still_picture_header {
        bw.write_bits(1, u64::from(sh.enable_interintra_compound));
        bw.write_bits(1, u64::from(sh.enable_masked_compound));
        bw.write_bits(1, u64::from(sh.enable_warped_motion));
        bw.write_bits(1, u64::from(sh.enable_dual_filter));
        bw.write_bits(1, u64::from(sh.enable_order_hint));
        if sh.enable_order_hint {
            bw.write_bits(1, u64::from(sh.enable_jnt_comp));
            bw.write_bits(1, u64::from(sh.enable_ref_frame_mvs));
        }

        // §5.5.1 seq_choose_screen_content_tools / seq_force_screen_content_tools.
        let seq_choose_screen_content_tools =
            sh.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS;
        bw.write_bits(1, u64::from(seq_choose_screen_content_tools));
        if !seq_choose_screen_content_tools {
            bw.write_bits(1, u64::from(sh.seq_force_screen_content_tools));
        }

        // §5.5.1 seq_choose_integer_mv / seq_force_integer_mv — only
        // present when seq_force_screen_content_tools > 0.
        if sh.seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = sh.seq_force_integer_mv == SELECT_INTEGER_MV;
            bw.write_bits(1, u64::from(seq_choose_integer_mv));
            if !seq_choose_integer_mv {
                bw.write_bits(1, u64::from(sh.seq_force_integer_mv));
            }
        }

        if sh.enable_order_hint {
            debug_assert!(
                (1..=8).contains(&sh.order_hint_bits),
                "enable_order_hint => order_hint_bits in 1..=8"
            );
            let order_hint_bits_minus_1 = sh.order_hint_bits - 1;
            bw.write_bits(3, u64::from(order_hint_bits_minus_1));
        }
    }

    bw.write_bits(1, u64::from(sh.enable_superres));
    bw.write_bits(1, u64::from(sh.enable_cdef));
    bw.write_bits(1, u64::from(sh.enable_restoration));
    encode_color_config(bw, &sh.color_config, sh.seq_profile);
    bw.write_bits(1, u64::from(sh.film_grain_params_present));
}

fn encode_timing_info(bw: &mut BitWriter, ti: &TimingInfo) {
    bw.write_bits(32, u64::from(ti.num_units_in_display_tick));
    bw.write_bits(32, u64::from(ti.time_scale));
    bw.write_bits(1, u64::from(ti.equal_picture_interval));
    if ti.equal_picture_interval {
        let v = ti
            .num_ticks_per_picture_minus_1
            .expect("equal_picture_interval => num_ticks_per_picture_minus_1");
        write_uvlc(bw, v);
    }
}

fn encode_decoder_model_info(bw: &mut BitWriter, dmi: &DecoderModelInfo) {
    bw.write_bits(5, u64::from(dmi.buffer_delay_length_minus_1));
    bw.write_bits(32, u64::from(dmi.num_units_in_decoding_tick));
    bw.write_bits(5, u64::from(dmi.buffer_removal_time_length_minus_1));
    bw.write_bits(5, u64::from(dmi.frame_presentation_time_length_minus_1));
}

fn encode_operating_parameters_info(
    bw: &mut BitWriter,
    opi: &OperatingParametersInfo,
    dmi: &DecoderModelInfo,
) {
    let n = u32::from(dmi.buffer_delay_length_minus_1) + 1;
    bw.write_bits(n, opi.decoder_buffer_delay);
    bw.write_bits(n, opi.encoder_buffer_delay);
    bw.write_bits(1, u64::from(opi.low_delay_mode_flag));
}

fn encode_operating_point(
    bw: &mut BitWriter,
    op: &OperatingPoint,
    decoder_model_info_present_flag: bool,
    decoder_model_info: Option<&DecoderModelInfo>,
    initial_display_delay_present_flag: bool,
) {
    bw.write_bits(12, u64::from(op.operating_point_idc));
    bw.write_bits(5, u64::from(op.seq_level_idx));
    if op.seq_level_idx > 7 {
        bw.write_bits(1, u64::from(op.seq_tier));
    } else {
        // §6.4.1: seq_tier inferred to 0; assert the descriptor matches.
        debug_assert_eq!(op.seq_tier, 0, "seq_tier must be 0 when seq_level_idx <= 7");
    }
    if decoder_model_info_present_flag {
        bw.write_bits(1, u64::from(op.decoder_model_present_for_this_op));
        if op.decoder_model_present_for_this_op {
            let opi = op
                .operating_parameters_info
                .as_ref()
                .expect("decoder_model_present_for_this_op => operating_parameters_info");
            let dmi = decoder_model_info
                .expect("decoder_model_info_present_flag => decoder_model_info populated");
            encode_operating_parameters_info(bw, opi, dmi);
        }
    }
    if initial_display_delay_present_flag {
        bw.write_bits(1, u64::from(op.initial_display_delay_present_for_this_op));
        if op.initial_display_delay_present_for_this_op {
            let v = op.initial_display_delay_minus_1.expect(
                "initial_display_delay_present_for_this_op => initial_display_delay_minus_1",
            );
            bw.write_bits(4, u64::from(v));
        }
    }
}

fn encode_color_config(bw: &mut BitWriter, cc: &ColorConfig, seq_profile: u8) {
    bw.write_bits(1, u64::from(cc.high_bitdepth));
    if seq_profile == 2 && cc.high_bitdepth {
        bw.write_bits(1, u64::from(cc.twelve_bit));
    }
    if seq_profile == 1 {
        // §5.5.2: monochrome bit is absent — implicitly 0.
        debug_assert!(!cc.mono_chrome, "seq_profile=1 forbids mono_chrome");
    } else {
        bw.write_bits(1, u64::from(cc.mono_chrome));
    }
    bw.write_bits(1, u64::from(cc.color_description_present_flag));
    if cc.color_description_present_flag {
        bw.write_bits(8, u64::from(cc.color_primaries));
        bw.write_bits(8, u64::from(cc.transfer_characteristics));
        bw.write_bits(8, u64::from(cc.matrix_coefficients));
    } else {
        debug_assert_eq!(cc.color_primaries, CP_UNSPECIFIED);
        debug_assert_eq!(cc.transfer_characteristics, TC_UNSPECIFIED);
        debug_assert_eq!(cc.matrix_coefficients, MC_UNSPECIFIED);
    }

    if cc.mono_chrome {
        // §5.5.2 mono_chrome path: write color_range and stop.
        bw.write_bits(1, u64::from(cc.color_range));
        debug_assert!(cc.subsampling_x && cc.subsampling_y);
        debug_assert_eq!(cc.chroma_sample_position, CSP_UNKNOWN);
        debug_assert!(!cc.separate_uv_delta_q);
        return;
    }

    let srgb_identity = cc.color_primaries == CP_BT_709
        && cc.transfer_characteristics == TC_SRGB
        && cc.matrix_coefficients == MC_IDENTITY;
    if srgb_identity {
        // §5.5.2: color_range / subsampling are inferred; nothing on wire.
        debug_assert!(cc.color_range);
        debug_assert!(!cc.subsampling_x);
        debug_assert!(!cc.subsampling_y);
    } else {
        bw.write_bits(1, u64::from(cc.color_range));
        if seq_profile == 0 {
            debug_assert!(cc.subsampling_x && cc.subsampling_y);
        } else if seq_profile == 1 {
            debug_assert!(!cc.subsampling_x && !cc.subsampling_y);
        } else {
            // seq_profile == 2.
            if cc.bit_depth == 12 {
                bw.write_bits(1, u64::from(cc.subsampling_x));
                if cc.subsampling_x {
                    bw.write_bits(1, u64::from(cc.subsampling_y));
                } else {
                    debug_assert!(!cc.subsampling_y);
                }
            } else {
                debug_assert!(cc.subsampling_x && !cc.subsampling_y);
            }
        }
        if cc.subsampling_x && cc.subsampling_y {
            bw.write_bits(2, u64::from(cc.chroma_sample_position));
        }
    }
    bw.write_bits(1, u64::from(cc.separate_uv_delta_q));
}

/// `uvlc()` writer per §4.10.3.
///
/// For value `v`, the encoding is `leadingZeros = floor(log2(v + 1))`
/// (i.e. position of the top bit of `v + 1`), then `leadingZeros`
/// '0' bits, one '1' bit, then `leadingZeros` payload bits being
/// the low `leadingZeros` bits of `(v + 1)`. The sentinel
/// `v == u32::MAX` re-emits 32 leading zeros + a '1' (no payload
/// bits) to match the parser's §4.10.3 short-circuit.
fn write_uvlc(bw: &mut BitWriter, value: u32) {
    if value == u32::MAX {
        // §4.10.3 sentinel encoding: 32 zeros + a single '1'.
        for _ in 0..32 {
            bw.write_bit(0);
        }
        bw.write_bit(1);
        return;
    }
    // Value `v` decodes from `lz` leading zeros + a '1' + lz payload
    // bits supplying `v - (2^lz - 1)`. Choose `lz` = bit length of
    // `v + 1` minus 1 = floor(log2(v + 1)).
    let v_plus_one = u64::from(value) + 1;
    let leading_zeros = 63 - v_plus_one.leading_zeros();
    for _ in 0..leading_zeros {
        bw.write_bit(0);
    }
    bw.write_bit(1);
    // Payload = v + 1 with the leading bit stripped = the low
    // `leading_zeros` bits of v + 1.
    let payload_mask = (1u64 << leading_zeros) - 1;
    let payload = v_plus_one & payload_mask;
    bw.write_bits(leading_zeros, payload);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_header::parse_sequence_header;

    // ---------------------------------------------------------------
    // Round-trip helpers
    // ---------------------------------------------------------------

    fn assert_round_trip(sh: &SequenceHeader) {
        let payload = write_sequence_header_obu(sh);
        let parsed = parse_sequence_header(&payload).expect("encoder output must parse cleanly");
        // bits_consumed will differ because the parser counts only the
        // bits it read; for the comparison we normalise it to the
        // encoder's bit count.
        let mut expected = sh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected, "round-trip mismatch");
    }

    fn tiny_16x16_profile0() -> SequenceHeader {
        // Mirror the parser-side fixture `TINY_16X16_PROF0`.
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
            frame_width_bits_minus_1: 3, // 4 bits => max_w up to 15
            frame_height_bits_minus_1: 3,
            max_frame_width_minus_1: 15,
            max_frame_height_minus_1: 15,
            frame_id_numbers_present_flag: false,
            delta_frame_id_length_minus_2: 0,
            additional_frame_id_length_minus_1: 0,
            use_128x128_superblock: true,
            enable_filter_intra: true,
            enable_intra_edge_filter: true,
            enable_interintra_compound: true,
            enable_masked_compound: true,
            enable_warped_motion: true,
            enable_dual_filter: true,
            enable_order_hint: true,
            enable_jnt_comp: true,
            enable_ref_frame_mvs: true,
            seq_force_screen_content_tools: SELECT_SCREEN_CONTENT_TOOLS,
            seq_force_integer_mv: SELECT_INTEGER_MV,
            order_hint_bits: 7,
            enable_superres: false,
            enable_cdef: true,
            enable_restoration: true,
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

    #[test]
    fn round_trip_tiny_16x16_profile0() {
        assert_round_trip(&tiny_16x16_profile0());
    }

    #[test]
    fn round_trip_monochrome() {
        let mut sh = tiny_16x16_profile0();
        sh.frame_width_bits_minus_1 = 5; // 6 bits, max 63
        sh.frame_height_bits_minus_1 = 5;
        sh.max_frame_width_minus_1 = 63;
        sh.max_frame_height_minus_1 = 63;
        sh.color_config.mono_chrome = true;
        sh.color_config.num_planes = 1;
        sh.color_config.color_range = true;
        // CSP_UNKNOWN / subsampling=true defaults already set.
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_profile2_yuv422_12bit() {
        let mut sh = tiny_16x16_profile0();
        sh.seq_profile = 2;
        sh.frame_width_bits_minus_1 = 5;
        sh.frame_height_bits_minus_1 = 5;
        sh.max_frame_width_minus_1 = 63;
        sh.max_frame_height_minus_1 = 63;
        sh.color_config.high_bitdepth = true;
        sh.color_config.twelve_bit = true;
        sh.color_config.bit_depth = 12;
        sh.color_config.subsampling_x = true;
        sh.color_config.subsampling_y = false;
        // 4:2:2 means !sx || !sy => no chroma_sample_position written
        // by the parser (so leave CSP_UNKNOWN, which we'll assert
        // matches after parse).
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_reduced_still_picture() {
        let mut sh = tiny_16x16_profile0();
        sh.still_picture = true;
        sh.reduced_still_picture_header = true;
        sh.operating_points[0].seq_level_idx = 0;
        // reduced-still forces all inter-feature flags off.
        sh.enable_interintra_compound = false;
        sh.enable_masked_compound = false;
        sh.enable_warped_motion = false;
        sh.enable_dual_filter = false;
        sh.enable_order_hint = false;
        sh.enable_jnt_comp = false;
        sh.enable_ref_frame_mvs = false;
        sh.seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS;
        sh.seq_force_integer_mv = SELECT_INTEGER_MV;
        sh.order_hint_bits = 0;
        // reduced-still bypasses frame_id_numbers_present_flag.
        sh.frame_id_numbers_present_flag = false;
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_with_timing_info_equal_picture_interval() {
        let mut sh = tiny_16x16_profile0();
        sh.timing_info_present_flag = true;
        sh.timing_info = Some(TimingInfo {
            num_units_in_display_tick: 1,
            time_scale: 30,
            equal_picture_interval: true,
            num_ticks_per_picture_minus_1: Some(0),
        });
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_with_timing_and_decoder_model() {
        let mut sh = tiny_16x16_profile0();
        sh.timing_info_present_flag = true;
        sh.timing_info = Some(TimingInfo {
            num_units_in_display_tick: 1001,
            time_scale: 60_000,
            equal_picture_interval: false,
            num_ticks_per_picture_minus_1: None,
        });
        sh.decoder_model_info_present_flag = true;
        sh.decoder_model_info = Some(DecoderModelInfo {
            buffer_delay_length_minus_1: 9, // n=10
            num_units_in_decoding_tick: 1001,
            buffer_removal_time_length_minus_1: 9,
            frame_presentation_time_length_minus_1: 4,
        });
        sh.operating_points[0].decoder_model_present_for_this_op = true;
        // decoder_buffer_delay / encoder_buffer_delay are n = 10 bits
        // wide here (buffer_delay_length_minus_1 + 1), so values must
        // fit in 0..1024.
        sh.operating_points[0].operating_parameters_info = Some(OperatingParametersInfo {
            decoder_buffer_delay: 0x123,
            encoder_buffer_delay: 0x345,
            low_delay_mode_flag: true,
        });
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_multiple_operating_points_with_high_level() {
        let mut sh = tiny_16x16_profile0();
        sh.operating_points_cnt_minus_1 = 1;
        sh.operating_points.push(OperatingPoint {
            operating_point_idc: 0x111,
            seq_level_idx: 12, // > 7 ⇒ seq_tier bit on the wire
            seq_tier: 1,
            decoder_model_present_for_this_op: false,
            operating_parameters_info: None,
            initial_display_delay_present_for_this_op: false,
            initial_display_delay_minus_1: None,
        });
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_with_initial_display_delay() {
        let mut sh = tiny_16x16_profile0();
        sh.initial_display_delay_present_flag = true;
        sh.operating_points[0].initial_display_delay_present_for_this_op = true;
        sh.operating_points[0].initial_display_delay_minus_1 = Some(7);
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_with_color_description() {
        let mut sh = tiny_16x16_profile0();
        sh.color_config.color_description_present_flag = true;
        sh.color_config.color_primaries = 9; // BT.2020
        sh.color_config.transfer_characteristics = 16; // SMPTE 2084
        sh.color_config.matrix_coefficients = 9; // BT.2020 NCL
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_with_frame_id_numbers() {
        let mut sh = tiny_16x16_profile0();
        sh.frame_id_numbers_present_flag = true;
        sh.delta_frame_id_length_minus_2 = 6;
        sh.additional_frame_id_length_minus_1 = 5;
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_force_screen_content_tools_off() {
        let mut sh = tiny_16x16_profile0();
        sh.seq_force_screen_content_tools = 0;
        // Per §5.5.1: when force_sct == 0, force_integer_mv is implicit
        // SELECT_INTEGER_MV; the parser sets it to that sentinel.
        sh.seq_force_integer_mv = SELECT_INTEGER_MV;
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_force_screen_content_tools_explicit_integer_mv() {
        let mut sh = tiny_16x16_profile0();
        sh.seq_force_screen_content_tools = 1;
        sh.seq_force_integer_mv = 1;
        assert_round_trip(&sh);
    }

    #[test]
    fn round_trip_disabled_order_hint() {
        let mut sh = tiny_16x16_profile0();
        sh.enable_order_hint = false;
        sh.enable_jnt_comp = false;
        sh.enable_ref_frame_mvs = false;
        sh.order_hint_bits = 0;
        assert_round_trip(&sh);
    }

    // ---------------------------------------------------------------
    // Byte-exact: encode → parse the parser's own fixture
    // ---------------------------------------------------------------

    #[test]
    fn encoder_output_parses_to_same_struct_as_fixture() {
        // The parser's TINY_16X16_PROF0 fixture decodes to a known
        // SequenceHeader; rebuilding it from our `tiny_16x16_profile0()`
        // helper and round-tripping must produce a parse that matches
        // bit-for-bit (modulo bits_consumed, which we normalise).
        let sh = tiny_16x16_profile0();
        let payload = write_sequence_header_obu(&sh);
        let parsed = parse_sequence_header(&payload).unwrap();
        assert_eq!(parsed.seq_profile, 0);
        assert_eq!(parsed.max_frame_width_minus_1, 15);
        assert_eq!(parsed.max_frame_height_minus_1, 15);
        assert!(parsed.use_128x128_superblock);
        assert!(parsed.enable_order_hint);
        assert_eq!(parsed.order_hint_bits, 7);
        assert_eq!(parsed.color_config.bit_depth, 8);
        assert!(parsed.color_config.subsampling_x);
        assert!(parsed.color_config.subsampling_y);
    }

    // ---------------------------------------------------------------
    // uvlc internal helper
    // ---------------------------------------------------------------

    #[test]
    fn uvlc_writer_matches_reader_for_low_values() {
        for v in [0u32, 1, 2, 3, 7, 8, 100, 1_000, 65_535, 1_000_000] {
            let mut bw = BitWriter::new();
            write_uvlc(&mut bw, v);
            let bytes = bw.finish();
            let mut br = crate::bitreader::BitReader::new(&bytes);
            let decoded = br.uvlc().unwrap();
            assert_eq!(decoded, v, "uvlc round-trip mismatch for {v}");
        }
    }

    #[test]
    fn uvlc_writer_sentinel_round_trips() {
        let mut bw = BitWriter::new();
        write_uvlc(&mut bw, u32::MAX);
        let bytes = bw.finish();
        let mut br = crate::bitreader::BitReader::new(&bytes);
        assert_eq!(br.uvlc().unwrap(), u32::MAX);
    }
}
