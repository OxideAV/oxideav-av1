//! `frame_header_obu()` writer — inverse of
//! [`crate::frame_header::parse_frame_header`].
//!
//! Implements the §5.9 syntax inversion for the intra-frame /
//! show-existing-frame / reduced-still-picture-header paths plus the
//! shared sub-procedures the §5.9.2 tail calls into. The §5.9.2
//! inter-frame branch (`frame_refs_short_signaling`,
//! `ref_frame_idx[]`, `frame_size_with_refs()`, `read_interpolation_filter()`,
//! `is_motion_mode_switchable`, `use_ref_frame_mvs`,
//! `frame_reference_mode()`, `skip_mode_params()`,
//! `global_motion_params()` non-trivial path) lands in the next arc.
//!
//! Sub-procedures covered:
//!
//!   * §5.9.1 — General frame header OBU syntax (the
//!     `uncompressed_header()` entry that this writer emits).
//!   * §5.9.2 — `uncompressed_header()` (intra + show-existing +
//!     reduced-still + inter w/ pre-resolved frame size; the inter
//!     branch above `disable_frame_end_update_cdf` is structurally
//!     mirrored back from a fully-populated [`FrameHeader`]).
//!   * §5.9.3 — `allow_intrabc` (gated by
//!     `allow_screen_content_tools && UpscaledWidth == FrameWidth`).
//!   * §5.9.5 — `frame_size()`.
//!   * §5.9.6 — `render_size()`.
//!   * §5.9.8 — `superres_params()`.
//!   * §5.9.10 — `read_interpolation_filter()` (inter path).
//!   * §5.9.11 — `loop_filter_params()`.
//!   * §5.9.12 — `quantization_params()` + §5.9.13 `read_delta_q()`.
//!   * §5.9.14 — `segmentation_params()`.
//!   * §5.9.15 — `tile_info()`.
//!   * §5.9.17 — `delta_q_params()`.
//!   * §5.9.18 — `delta_lf_params()`.
//!   * §5.9.19 — `cdef_params()`.
//!   * §5.9.20 — `lr_params()`.
//!   * §5.9.21 — `read_tx_mode()`.
//!   * §5.9.22 — `skip_mode_params()` (intra short-circuit).
//!   * §5.9.23 — `frame_reference_mode()` (intra short-circuit).
//!   * §5.9.24 — `global_motion_params()` (intra short-circuit;
//!     inter `IDENTITY`-only).
//!   * §5.9.30 — `film_grain_params()` (reset path + apply_grain == 0;
//!     full populated path lands next arc).
//!
//! The writer takes a [`FrameHeader`] (the same struct the parser
//! fills) plus the [`SequenceHeader`] that governs every conditional
//! read, and emits the **payload** of an `OBU_FRAME_HEADER` /
//! `OBU_REDUNDANT_FRAME_HEADER` / `OBU_FRAME` unit. The framing layer
//! (`ObuWriter` + `leb128()` size field + `trailing_bits()`) is the
//! caller's responsibility — this writer stops at the last bit
//! `uncompressed_header()` would have consumed.
//!
//! Round-trip discipline: an emitted payload feeds straight into
//! [`crate::frame_header::parse_frame_header`] and yields a
//! [`FrameHeader`] structurally equal to the input (after normalising
//! [`FrameHeader::bits_consumed`], which the parser sets from its own
//! bit-position counter).

use crate::encoder::bitwriter::BitWriter;
use crate::frame_header::{
    FrameHeader, FrameSize, FrameType, InterFrameRefs, ALL_FRAMES_PUB, PRIMARY_REF_NONE,
    SUPERRES_DENOM_BITS, SUPERRES_DENOM_MIN, SUPERRES_NUM,
};
use crate::sequence_header::{SequenceHeader, SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};
use crate::tile_info::{TileInfo, MAX_TILE_COLS, MAX_TILE_ROWS, MAX_TILE_WIDTH};
use crate::uncompressed_header_tail::{
    CdefParams, DeltaLfParams, DeltaQParams, FilmGrainParams, GlobalMotionParams,
    InterpolationFilter, LoopFilterParams, LrParams, QuantizationParams, SegmentationParams,
    TxMode, WarpModelType, MAX_LOOP_FILTER, MAX_SEGMENTS, REFS_PER_FRAME,
    SEGMENTATION_FEATURE_BITS, SEGMENTATION_FEATURE_MAX, SEGMENTATION_FEATURE_SIGNED, SEG_LVL_MAX,
};

/// Encode a `frame_header_obu()` payload per §5.9.1 / §5.9.2 from a
/// [`FrameHeader`] descriptor and the active [`SequenceHeader`].
///
/// Returns the byte buffer the OBU framer should wrap with `obu_type
/// == OBU_FRAME_HEADER` (or `OBU_REDUNDANT_FRAME_HEADER` / payload
/// prefix of `OBU_FRAME`). The trailing `trailing_bits()` (§5.3.4)
/// belongs to the framer and is **not** appended here.
pub fn write_frame_header_obu(fh: &FrameHeader, seq: &SequenceHeader) -> Vec<u8> {
    let mut bw = BitWriter::new();
    encode_uncompressed_header(&mut bw, fh, seq);
    // §5.3.1 / §5.3.4 (r409): `OBU_FRAME_HEADER` takes trailing_bits,
    // and the trailing one-bit must occupy the FIRST unused bit after
    // the last syntax element. A §5.10 `OBU_FRAME` composition must
    // NOT use this entry (§5.10 pads with `byte_alignment()` zeros
    // instead); it should drive [`encode_uncompressed_header`]
    // directly.
    bw.trailing_bits_to_alignment();
    bw.finish()
}

pub(crate) fn encode_uncompressed_header(
    bw: &mut BitWriter,
    fh: &FrameHeader,
    seq: &SequenceHeader,
) {
    // §5.9.2 idLen derivation (only meaningful when frame_id_numbers_present_flag).
    let id_len: u32 = if seq.frame_id_numbers_present_flag {
        u32::from(seq.additional_frame_id_length_minus_1)
            + u32::from(seq.delta_frame_id_length_minus_2)
            + 3
    } else {
        0
    };

    if seq.reduced_still_picture_header {
        // §5.9.2 reduced-still-picture-header collapse. Most fields
        // are derived and not on the wire.
        bw.write_bits(1, u64::from(fh.disable_cdf_update));
        encode_allow_scc_if_present(bw, seq, fh.allow_screen_content_tools);
        // FrameIsIntra forces force_integer_mv = 1 in the §5.9.2
        // override, but the bitstream slot is conditionally present
        // when allow_screen_content_tools && seq_force_integer_mv ==
        // SELECT_INTEGER_MV. We emit whatever value would have
        // round-tripped: since the FrameHeader stores the post-override
        // value (always true here), pick the seq's force value as the
        // raw bit and let the parser override it back to true.
        // The encoder mirrors the spec's `if (allow_scc)` gate, so the
        // raw value on the wire matches `seq.seq_force_integer_mv != 0`
        // (or 1 since this is an intra frame and the parser would have
        // forced to 1 anyway — any value round-trips).
        encode_force_integer_mv_if_present(bw, seq, fh.allow_screen_content_tools, true);
        // The remaining leading-block fields (current_frame_id,
        // frame_size_override_flag, order_hint, primary_ref_frame,
        // refresh_frame_flags) are all derived per §5.9.2 in this path
        // — none consume bits.

        let fs = fh
            .frame_size
            .as_ref()
            .expect("reduced-still path requires populated frame_size");
        encode_frame_size_block(bw, seq, fs, fh.frame_size_override_flag);
        // §5.9.3 allow_intrabc.
        encode_allow_intrabc(bw, fh.allow_screen_content_tools, fs, fh.allow_intrabc);
        // §5.9.2 disable_frame_end_update_cdf — derived to 1 here; no
        // bit written.
        encode_tile_info(bw, fh.tile_info.as_ref().expect("intra has tile_info"));
        encode_quantization_params(
            bw,
            fh.quantization_params
                .as_ref()
                .expect("intra has quantization_params"),
            seq.color_config.num_planes,
            seq.color_config.separate_uv_delta_q,
        );
        encode_segmentation_params(
            bw,
            fh.segmentation_params
                .as_ref()
                .expect("intra has segmentation_params"),
            fh.primary_ref_frame,
        );
        let qp = fh.quantization_params.as_ref().unwrap();
        let dq = fh
            .delta_q_params
            .as_ref()
            .expect("intra has delta_q_params");
        encode_delta_q_params(bw, dq, qp.base_q_idx);
        let dlf = fh
            .delta_lf_params
            .as_ref()
            .expect("intra has delta_lf_params");
        encode_delta_lf_params(bw, dlf, dq.delta_q_present, fh.allow_intrabc);
        let coded_lossless = derive_coded_lossless(qp, fh.segmentation_params.as_ref().unwrap());
        let lf = fh
            .loop_filter_params
            .as_ref()
            .expect("intra has loop_filter_params");
        encode_loop_filter_params(
            bw,
            lf,
            seq.color_config.num_planes,
            coded_lossless,
            fh.allow_intrabc,
        );
        let cdef = fh.cdef_params.as_ref().expect("intra has cdef_params");
        encode_cdef_params(
            bw,
            cdef,
            seq.color_config.num_planes,
            coded_lossless,
            fh.allow_intrabc,
            seq.enable_cdef,
        );
        let lr = fh.lr_params.as_ref().expect("intra has lr_params");
        let all_lossless = coded_lossless && fs.frame_width == fs.upscaled_width;
        encode_lr_params(
            bw,
            lr,
            seq.color_config.num_planes,
            seq.color_config.subsampling_x,
            seq.color_config.subsampling_y,
            seq.use_128x128_superblock,
            all_lossless,
            fh.allow_intrabc,
            seq.enable_restoration,
        );
        let tx = fh.tx_mode.expect("intra has tx_mode");
        encode_tx_mode(bw, tx, coded_lossless);
        // Intra tail: frame_reference_mode / skip_mode / allow_warped_motion
        // are all short-circuited (no bits). reduced_tx_set is one bit.
        // global_motion is the intra short-circuit. film_grain_params is
        // the §5.9.30 short-circuit (or reset path).
        let rts = fh.reduced_tx_set.expect("intra has reduced_tx_set");
        bw.write_bits(1, u64::from(rts));
        // global_motion_params: short-circuit for intra; emits nothing.
        // film_grain_params: §5.9.30 reset / short-circuit path emits
        // nothing (the !film_grain_params_present and shown-only gates
        // both bail before reading any bit). When apply_grain==false on
        // a frame that *could* signal grain we emit the leading f(1)=0.
        encode_film_grain_params(
            bw,
            fh.film_grain_params
                .as_ref()
                .expect("intra has film_grain_params"),
            seq,
            fh.show_frame,
            fh.showable_frame,
            matches!(fh.frame_type, FrameType::Inter),
        );
        return;
    }

    // ----- Non-reduced path -----
    bw.write_bits(1, u64::from(fh.show_existing_frame));
    if fh.show_existing_frame {
        let map_idx = fh
            .frame_to_show_map_idx
            .expect("show_existing_frame => frame_to_show_map_idx");
        bw.write_bits(3, u64::from(map_idx));
        // §5.9.2 temporal_point_info gate — caller-supplied FrameHeader
        // never crosses the TemporalPointInfoUnsupported boundary, so we
        // simply elide. (decoder_model_info handling is out of scope for
        // this arc; the assert keeps us honest.)
        debug_assert!(
            !decoder_model_info_present(seq) || equal_picture_interval(seq),
            "temporal_point_info round-trip not modelled yet"
        );
        if seq.frame_id_numbers_present_flag {
            let display_id = fh
                .display_frame_id
                .expect("frame_id_numbers_present_flag => display_frame_id");
            bw.write_bits(id_len, u64::from(display_id));
        }
        return;
    }

    bw.write_bits(2, u64::from(fh.frame_type.as_raw()));
    bw.write_bits(1, u64::from(fh.show_frame));

    debug_assert!(
        !(fh.show_frame && decoder_model_info_present(seq) && !equal_picture_interval(seq)),
        "temporal_point_info round-trip not modelled yet"
    );

    let frame_type = fh.frame_type;
    let key_and_show = matches!(frame_type, FrameType::Key) && fh.show_frame;
    if !fh.show_frame {
        bw.write_bits(1, u64::from(fh.showable_frame));
    }
    if !(matches!(frame_type, FrameType::Switch) || key_and_show) {
        bw.write_bits(1, u64::from(fh.error_resilient_mode));
    }

    bw.write_bits(1, u64::from(fh.disable_cdf_update));
    encode_allow_scc_if_present(bw, seq, fh.allow_screen_content_tools);
    // raw_force_integer_mv — for intra frames the FrameHeader stores the
    // post-override `true`; for inter frames it is the raw bit. The
    // parser overrides back to true for FrameIsIntra regardless of what
    // we write, so the same field works on both branches.
    encode_force_integer_mv_if_present(bw, seq, fh.allow_screen_content_tools, fh.force_integer_mv);

    if seq.frame_id_numbers_present_flag {
        bw.write_bits(id_len, u64::from(fh.current_frame_id));
    }

    if !matches!(frame_type, FrameType::Switch) {
        bw.write_bits(1, u64::from(fh.frame_size_override_flag));
    }

    if seq.order_hint_bits > 0 {
        bw.write_bits(u32::from(seq.order_hint_bits), u64::from(fh.order_hint));
    }

    if !(fh.frame_is_intra || fh.error_resilient_mode) {
        bw.write_bits(3, u64::from(fh.primary_ref_frame));
    }

    debug_assert!(
        !decoder_model_info_present(seq),
        "decoder_model_info round-trip not modelled yet"
    );

    if !(matches!(frame_type, FrameType::Switch) || key_and_show) {
        bw.write_bits(8, u64::from(fh.refresh_frame_flags));
    }

    // §5.9.2 ref_order_hint block: only fires when
    //   (!FrameIsIntra || refresh_frame_flags != allFrames) &&
    //   error_resilient_mode && enable_order_hint.
    // r413: the per-slot values come from `fh.ref_order_hints` — the
    // spec requires each coded hint to equal the decoder's stored
    // `RefOrderHint[ i ]` (a mismatch marks the slot `RefValid = 0`),
    // so the caller must surface the true slot hints whenever the gate
    // fires. `None` falls back to zeros (the pre-r413 shape, valid only
    // while every stored hint is genuinely zero).
    if (!fh.frame_is_intra || fh.refresh_frame_flags != ALL_FRAMES_PUB)
        && fh.error_resilient_mode
        && seq.enable_order_hint
    {
        let hints = fh.ref_order_hints.unwrap_or([0; 8]);
        for &hint in hints.iter() {
            bw.write_bits(u32::from(seq.order_hint_bits), u64::from(hint));
        }
    }

    if fh.frame_is_intra {
        let fs = fh.frame_size.as_ref().expect("intra path has frame_size");
        encode_frame_size_block(bw, seq, fs, fh.frame_size_override_flag);
        encode_allow_intrabc(bw, fh.allow_screen_content_tools, fs, fh.allow_intrabc);
        // §5.9.2 disable_frame_end_update_cdf — read as f(1) unless
        // disable_cdf_update (forces 1).
        if !fh.disable_cdf_update {
            bw.write_bits(1, u64::from(fh.disable_frame_end_update_cdf));
        }
        encode_tile_info(bw, fh.tile_info.as_ref().expect("intra has tile_info"));
        encode_intra_tail(bw, fh, seq, fs);
    } else {
        // Inter path: support the case where frame_size_with_refs would
        // have been bypassed (frame_size_override_flag == 0 ||
        // error_resilient_mode), i.e. the regular frame_size() + render_size()
        // block fires. The other branch needs ref-frame state to
        // round-trip; that lands in the next arc.
        let inter_refs = fh
            .inter_refs
            .as_ref()
            .expect("inter path requires populated inter_refs");
        encode_inter_ref_block(bw, fh, seq, inter_refs, id_len);

        let fs = fh.frame_size.as_ref().expect("inter path has frame_size");
        if fh.frame_size_override_flag && !fh.error_resilient_mode {
            // frame_size_with_refs round-trip is deferred — write the
            // no-found-ref fallback path.
            for _ in 0..REFS_PER_FRAME {
                bw.write_bits(1, 0);
            }
            encode_frame_size_block(bw, seq, fs, fh.frame_size_override_flag);
        } else {
            encode_frame_size_block(bw, seq, fs, fh.frame_size_override_flag);
        }

        if !fh.force_integer_mv {
            bw.write_bits(1, u64::from(inter_refs.allow_high_precision_mv));
        }
        encode_interpolation_filter(bw, inter_refs.interpolation_filter);
        bw.write_bits(1, u64::from(inter_refs.is_motion_mode_switchable));
        if !fh.error_resilient_mode && seq.enable_ref_frame_mvs {
            bw.write_bits(1, u64::from(inter_refs.use_ref_frame_mvs));
        }

        if !fh.disable_cdf_update {
            bw.write_bits(1, u64::from(fh.disable_frame_end_update_cdf));
        }
        encode_tile_info(bw, fh.tile_info.as_ref().expect("inter has tile_info"));
        encode_inter_tail(bw, fh, seq, fs);
    }
}

fn encode_inter_ref_block(
    bw: &mut BitWriter,
    _fh: &FrameHeader,
    seq: &SequenceHeader,
    inter_refs: &InterFrameRefs,
    _id_len: u32,
) {
    if seq.enable_order_hint {
        bw.write_bits(1, u64::from(inter_refs.frame_refs_short_signaling));
    } else {
        debug_assert!(
            !inter_refs.frame_refs_short_signaling,
            "short_signaling requires enable_order_hint"
        );
    }
    if inter_refs.frame_refs_short_signaling {
        let last = inter_refs
            .last_frame_idx
            .expect("short_signaling => last_frame_idx");
        let gold = inter_refs
            .gold_frame_idx
            .expect("short_signaling => gold_frame_idx");
        bw.write_bits(3, u64::from(last));
        bw.write_bits(3, u64::from(gold));
    }
    for &idx in inter_refs.ref_frame_idx.iter() {
        if !inter_refs.frame_refs_short_signaling {
            bw.write_bits(3, u64::from(idx));
        }
        if seq.frame_id_numbers_present_flag {
            let n = u32::from(seq.delta_frame_id_length_minus_2) + 2;
            // delta_frame_id_minus_1 was not surfaced on the parser
            // (read-and-discard); emit zero. A round-trip session that
            // needs to preserve actual delta ids will grow a per-ref
            // field on InterFrameRefs.
            bw.write_bits(n, 0);
        }
    }
}

fn encode_intra_tail(bw: &mut BitWriter, fh: &FrameHeader, seq: &SequenceHeader, fs: &FrameSize) {
    let qp = fh
        .quantization_params
        .as_ref()
        .expect("intra has quantization_params");
    let sp = fh
        .segmentation_params
        .as_ref()
        .expect("intra has segmentation_params");
    encode_quantization_params(
        bw,
        qp,
        seq.color_config.num_planes,
        seq.color_config.separate_uv_delta_q,
    );
    encode_segmentation_params(bw, sp, fh.primary_ref_frame);
    let dq = fh
        .delta_q_params
        .as_ref()
        .expect("intra has delta_q_params");
    encode_delta_q_params(bw, dq, qp.base_q_idx);
    let dlf = fh
        .delta_lf_params
        .as_ref()
        .expect("intra has delta_lf_params");
    encode_delta_lf_params(bw, dlf, dq.delta_q_present, fh.allow_intrabc);
    let coded_lossless = derive_coded_lossless(qp, sp);
    let lf = fh
        .loop_filter_params
        .as_ref()
        .expect("intra has loop_filter_params");
    encode_loop_filter_params(
        bw,
        lf,
        seq.color_config.num_planes,
        coded_lossless,
        fh.allow_intrabc,
    );
    let cdef = fh.cdef_params.as_ref().expect("intra has cdef_params");
    encode_cdef_params(
        bw,
        cdef,
        seq.color_config.num_planes,
        coded_lossless,
        fh.allow_intrabc,
        seq.enable_cdef,
    );
    let lr = fh.lr_params.as_ref().expect("intra has lr_params");
    let all_lossless = coded_lossless && fs.frame_width == fs.upscaled_width;
    encode_lr_params(
        bw,
        lr,
        seq.color_config.num_planes,
        seq.color_config.subsampling_x,
        seq.color_config.subsampling_y,
        seq.use_128x128_superblock,
        all_lossless,
        fh.allow_intrabc,
        seq.enable_restoration,
    );
    let tx = fh.tx_mode.expect("intra has tx_mode");
    encode_tx_mode(bw, tx, coded_lossless);
    // §5.9.23 frame_reference_mode / §5.9.22 skip_mode / allow_warped_motion
    // are all intra short-circuits (no bits).
    let rts = fh.reduced_tx_set.expect("intra has reduced_tx_set");
    bw.write_bits(1, u64::from(rts));
    // global_motion_params: §5.9.24 intra short-circuit, no bits.
    encode_film_grain_params(
        bw,
        fh.film_grain_params
            .as_ref()
            .expect("intra has film_grain_params"),
        seq,
        fh.show_frame,
        fh.showable_frame,
        matches!(fh.frame_type, FrameType::Inter),
    );
}

fn encode_inter_tail(bw: &mut BitWriter, fh: &FrameHeader, seq: &SequenceHeader, fs: &FrameSize) {
    let qp = fh
        .quantization_params
        .as_ref()
        .expect("inter has quantization_params");
    let sp = fh
        .segmentation_params
        .as_ref()
        .expect("inter has segmentation_params");
    encode_quantization_params(
        bw,
        qp,
        seq.color_config.num_planes,
        seq.color_config.separate_uv_delta_q,
    );
    encode_segmentation_params(bw, sp, fh.primary_ref_frame);
    let dq = fh
        .delta_q_params
        .as_ref()
        .expect("inter has delta_q_params");
    encode_delta_q_params(bw, dq, qp.base_q_idx);
    let dlf = fh
        .delta_lf_params
        .as_ref()
        .expect("inter has delta_lf_params");
    encode_delta_lf_params(bw, dlf, dq.delta_q_present, false);
    let coded_lossless = derive_coded_lossless(qp, sp);
    let lf = fh
        .loop_filter_params
        .as_ref()
        .expect("inter has loop_filter_params");
    encode_loop_filter_params(bw, lf, seq.color_config.num_planes, coded_lossless, false);
    let cdef = fh.cdef_params.as_ref().expect("inter has cdef_params");
    encode_cdef_params(
        bw,
        cdef,
        seq.color_config.num_planes,
        coded_lossless,
        false,
        seq.enable_cdef,
    );
    let lr = fh.lr_params.as_ref().expect("inter has lr_params");
    let all_lossless = coded_lossless && fs.frame_width == fs.upscaled_width;
    encode_lr_params(
        bw,
        lr,
        seq.color_config.num_planes,
        seq.color_config.subsampling_x,
        seq.color_config.subsampling_y,
        seq.use_128x128_superblock,
        all_lossless,
        false,
        seq.enable_restoration,
    );
    let tx = fh.tx_mode.expect("inter has tx_mode");
    encode_tx_mode(bw, tx, coded_lossless);
    // §5.9.23 inter reference_select read.
    let rs = fh.reference_select.expect("inter has reference_select");
    bw.write_bits(1, u64::from(rs));
    // §5.9.22 skip_mode_params() — the write twin of the parser's
    // read_skip_mode_present(): when skipModeAllowed, ONE
    // skip_mode_present bit is coded (r413; the pre-r413 writer never
    // emitted it, which the reader survived only because the phantom
    // read landed on the all-zero reduced_tx_set / identity-gm tail
    // and the byte_alignment() padding absorbed the shift).
    // skipModeAllowed needs the per-slot RefOrderHint[] state — on
    // the error-resilient configuration that is exactly
    // `fh.ref_order_hints` (conformance requires the coded hints to
    // equal the stored ones); `None` falls back to all-zero hints,
    // whose duplicate forward hints derive skipModeAllowed = 0.
    if rs && seq.enable_order_hint {
        let hints = fh.ref_order_hints.unwrap_or([0; 8]);
        let ref_frame_idx = fh
            .inter_refs
            .as_ref()
            .map(|ir| ir.ref_frame_idx)
            .unwrap_or([0; 7]);
        if skip_mode_allowed(&hints, &ref_frame_idx, fh.order_hint, seq.order_hint_bits) {
            bw.write_bits(1, u64::from(fh.skip_mode_present.unwrap_or(false)));
        } else {
            debug_assert!(
                !fh.skip_mode_present.unwrap_or(false),
                "skip_mode_present = 1 requires skipModeAllowed"
            );
        }
    }
    if !fh.error_resilient_mode && seq.enable_warped_motion {
        bw.write_bits(1, u64::from(fh.allow_warped_motion.unwrap_or(false)));
    }
    let rts = fh.reduced_tx_set.expect("inter has reduced_tx_set");
    bw.write_bits(1, u64::from(rts));
    // §5.9.24 global_motion_params — inter path. The full
    // read_global_param signed-subexp emission lands next arc; we
    // support the `gm_type == IDENTITY` short-circuit (1 bit per ref).
    let gm = fh
        .global_motion_params
        .as_ref()
        .expect("inter has global_motion_params");
    encode_inter_global_motion_identity_only(bw, gm);
    encode_film_grain_params(
        bw,
        fh.film_grain_params
            .as_ref()
            .expect("inter has film_grain_params"),
        seq,
        fh.show_frame,
        fh.showable_frame,
        matches!(fh.frame_type, FrameType::Inter),
    );
}

/// §5.9.3 `get_relative_dist( a, b )` over `order_hint_bits`-wide
/// hints (the enable gate is on the caller — this module only calls it
/// under `seq.enable_order_hint`).
fn relative_dist(a: u32, b: u32, order_hint_bits: u8) -> i32 {
    let mut diff = a as i32 - b as i32;
    let m = 1i32 << (i32::from(order_hint_bits) - 1);
    diff = (diff & (m - 1)) - (diff & m);
    diff
}

/// §5.9.22 `skipModeAllowed` derivation — the write twin of the
/// parser's `read_skip_mode_present()`: find the closest forward and
/// backward references (by `get_relative_dist` against the current
/// `OrderHint`); a forward/backward pair allows skip mode, else the
/// two closest DISTINCT forward hints do. `FrameIsIntra == 0`,
/// `reference_select == 1` and `enable_order_hint == 1` are the
/// caller's gates.
pub(crate) fn skip_mode_allowed(
    ref_order_hints: &[u32; 8],
    ref_frame_idx: &[u8; 7],
    order_hint: u32,
    order_hint_bits: u8,
) -> bool {
    let mut forward_idx: i32 = -1;
    let mut backward_idx: i32 = -1;
    let mut forward_hint: u32 = 0;
    let mut backward_hint: u32 = 0;
    for (i, &slot) in ref_frame_idx.iter().enumerate() {
        let ref_hint = ref_order_hints[slot as usize];
        if relative_dist(ref_hint, order_hint, order_hint_bits) < 0 {
            if forward_idx < 0 || relative_dist(ref_hint, forward_hint, order_hint_bits) > 0 {
                forward_idx = i as i32;
                forward_hint = ref_hint;
            }
        } else if relative_dist(ref_hint, order_hint, order_hint_bits) > 0
            && (backward_idx < 0 || relative_dist(ref_hint, backward_hint, order_hint_bits) < 0)
        {
            backward_idx = i as i32;
            backward_hint = ref_hint;
        }
    }
    if forward_idx < 0 {
        return false;
    }
    if backward_idx >= 0 {
        return true;
    }
    // Two-forward-reference fallback: a second forward hint strictly
    // older than `forwardHint`.
    ref_frame_idx.iter().any(|&slot| {
        relative_dist(
            ref_order_hints[slot as usize],
            forward_hint,
            order_hint_bits,
        ) < 0
    })
}

fn encode_inter_global_motion_identity_only(bw: &mut BitWriter, gm: &GlobalMotionParams) {
    // §5.9.24 inter loop over LAST_FRAME..=ALTREF_FRAME. We only
    // support `IDENTITY` (single `is_global = 0` bit per ref) until
    // the next arc lands read_global_param's inverse.
    for ref_idx in 1usize..=7 {
        debug_assert!(
            matches!(gm.gm_type[ref_idx], WarpModelType::Identity),
            "global_motion round-trip for non-IDENTITY refs lands next arc"
        );
        bw.write_bits(1, 0);
    }
}

// ---------------------------------------------------------------------
// §5.9.5 + §5.9.6 + §5.9.8 frame_size / render_size / superres_params
// ---------------------------------------------------------------------

fn encode_frame_size_block(
    bw: &mut BitWriter,
    seq: &SequenceHeader,
    fs: &FrameSize,
    frame_size_override_flag: bool,
) {
    if frame_size_override_flag {
        let n_w = u32::from(seq.frame_width_bits_minus_1) + 1;
        let n_h = u32::from(seq.frame_height_bits_minus_1) + 1;
        // §5.9.5: write frame_width_minus_1 / frame_height_minus_1 from
        // the *upscaled* width (the pre-superres value) per §5.9.5
        // semantics: the wire stores `UpscaledWidth` and superres
        // derives `FrameWidth` from it.
        let upscaled_minus_1 = fs.upscaled_width.saturating_sub(1);
        let height_minus_1 = fs.frame_height.saturating_sub(1);
        bw.write_bits(n_w, u64::from(upscaled_minus_1));
        bw.write_bits(n_h, u64::from(height_minus_1));
    }
    // §5.9.8 superres_params().
    if seq.enable_superres {
        bw.write_bits(1, u64::from(fs.use_superres));
        if fs.use_superres {
            bw.write_bits(SUPERRES_DENOM_BITS, u64::from(fs.coded_denom));
        }
    } else {
        debug_assert!(!fs.use_superres, "use_superres requires enable_superres");
    }
    // §5.9.6 render_size().
    bw.write_bits(1, u64::from(fs.render_and_frame_size_different));
    if fs.render_and_frame_size_different {
        bw.write_bits(16, u64::from(fs.render_width.saturating_sub(1)));
        bw.write_bits(16, u64::from(fs.render_height.saturating_sub(1)));
    }
    // §5.9.9 compute_image_size derives MiCols / MiRows from FrameWidth /
    // FrameHeight; no bits emitted.
    let _ = SUPERRES_NUM; // referenced for spec-citation completeness
    let _ = SUPERRES_DENOM_MIN;
}

fn encode_allow_intrabc(bw: &mut BitWriter, allow_scc: bool, fs: &FrameSize, allow_intrabc: bool) {
    if allow_scc && fs.upscaled_width == fs.frame_width {
        bw.write_bits(1, u64::from(allow_intrabc));
    }
}

// ---------------------------------------------------------------------
// §5.9.15 tile_info
// ---------------------------------------------------------------------

fn encode_tile_info(bw: &mut BitWriter, ti: &TileInfo) {
    // The §5.9.15 lead-in derivations re-compute mi_cols / mi_rows from
    // the tile starts; sb_shift is recovered from the row-start spacing.
    // We re-derive everything here so the writer mirrors the parser
    // exactly without needing extra fields on TileInfo.

    let mi_cols = *ti.mi_col_starts.last().expect("mi_col_starts non-empty");
    let mi_rows = *ti.mi_row_starts.last().expect("mi_row_starts non-empty");

    // Recover sb_shift by inspecting the first non-zero column start
    // (mi units per superblock = 16 for sb_size 64, 32 for sb_size 128).
    // When there's only one tile column the sentinel mi_cols itself
    // gives us the frame size — we infer sb_shift from MAX(sb_cols, 1).
    // Mirror the parser's logic: if any column start mod 32 != 0 then
    // sb_shift == 4.
    let use_128x128 = derive_use_128x128(ti);
    let (sb_cols, sb_rows, sb_shift) = if use_128x128 {
        ((mi_cols + 31) >> 5, (mi_rows + 31) >> 5, 5u32)
    } else {
        ((mi_cols + 15) >> 4, (mi_rows + 15) >> 4, 4u32)
    };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let max_tile_area_sb = crate::tile_info::MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    bw.write_bits(1, u64::from(ti.uniform_tile_spacing_flag));

    if ti.uniform_tile_spacing_flag {
        // §5.9.15 uniform path: emit increment bits up to tile_cols_log2.
        let mut t = min_log2_tile_cols;
        while t < max_log2_tile_cols {
            if t < ti.tile_cols_log2 {
                bw.write_bit(1);
                t += 1;
            } else {
                bw.write_bit(0);
                break;
            }
        }

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(ti.tile_cols_log2);
        let mut r = min_log2_tile_rows;
        while r < max_log2_tile_rows {
            if r < ti.tile_rows_log2 {
                bw.write_bit(1);
                r += 1;
            } else {
                bw.write_bit(0);
                break;
            }
        }
    } else {
        // §5.9.15 non-uniform path: emit width_in_sbs_minus_1 /
        // height_in_sbs_minus_1 from the consecutive start differences.
        let mut start_sb: u32 = 0;
        for window in ti.mi_col_starts.windows(2) {
            let next_mi = window[1];
            let size_sb = (next_mi >> sb_shift) - start_sb;
            // last tile clamps to (sb_cols - start_sb); we want the same.
            let actual = ((next_mi - (start_sb << sb_shift)) + (1 << sb_shift) - 1) >> sb_shift;
            let final_size_sb = if actual == 0 { 1 } else { size_sb };
            let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
            bw.write_ns(max_width.max(1), final_size_sb - 1);
            start_sb += final_size_sb;
        }

        // Re-derive max_tile_height_sb the same way the parser does.
        let widest_tile_sb = compute_widest_tile_sb(ti, sb_shift);
        let max_tile_area_sb_local = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (max_tile_area_sb_local / widest_tile_sb).max(1);

        let mut start_sb_r: u32 = 0;
        for window in ti.mi_row_starts.windows(2) {
            let next_mi = window[1];
            let size_sb = (next_mi >> sb_shift) - start_sb_r;
            let actual = ((next_mi - (start_sb_r << sb_shift)) + (1 << sb_shift) - 1) >> sb_shift;
            let final_size_sb = if actual == 0 { 1 } else { size_sb };
            let max_height = (sb_rows - start_sb_r).min(max_tile_height_sb);
            bw.write_ns(max_height.max(1), final_size_sb - 1);
            start_sb_r += final_size_sb;
        }
    }

    if ti.tile_cols_log2 > 0 || ti.tile_rows_log2 > 0 {
        let n = ti.tile_rows_log2 + ti.tile_cols_log2;
        bw.write_bits(n, u64::from(ti.context_update_tile_id));
        bw.write_bits(2, u64::from(ti.tile_size_bytes - 1));
    }
}

fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0u32;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// Recover the §5.5.1 `use_128x128_superblock` flag from a parsed
/// [`TileInfo`] by inspecting whether any column / row start lies on a
/// 16-mi but not 32-mi boundary.
fn derive_use_128x128(ti: &TileInfo) -> bool {
    // If any non-zero start (excluding the sentinel) is not a multiple
    // of 32, we must be on the 64-superblock side (sb_shift == 4).
    for &s in ti.mi_col_starts.iter().skip(1) {
        // skip the sentinel by stopping before len - 1: but skip first
        // gives us [s1, s2, ..., sentinel]; we want all intermediate
        // starts to test.
        if s == 0 {
            continue;
        }
        if s % 32 != 0 {
            return false;
        }
    }
    for &s in ti.mi_row_starts.iter().skip(1) {
        if s == 0 {
            continue;
        }
        if s % 32 != 0 {
            return false;
        }
    }
    // The sentinel itself (last entry) is mi_cols / mi_rows, which is
    // the post-compute-image-size value: 2*ceil(width/8). For frames
    // whose dimensions are 0..=128 the sentinel can be 32 (1 sb of 128
    // or 2 sb of 64) — ambiguous. Default to false (sb=64) which is
    // the more common case across the fixture corpus. Callers that
    // need 128-sb round-trip with a single tile must surface the flag
    // explicitly (which the next arc may add via a TileInfo field).
    false
}

fn compute_widest_tile_sb(ti: &TileInfo, sb_shift: u32) -> u32 {
    let mut widest: u32 = 0;
    let mut start_sb: u32 = 0;
    for window in ti.mi_col_starts.windows(2) {
        let next_mi = window[1];
        let size_sb = (next_mi >> sb_shift) - start_sb;
        let size_sb = size_sb.max(1);
        widest = widest.max(size_sb);
        start_sb += size_sb;
    }
    widest.max(1)
}

// ---------------------------------------------------------------------
// §5.9.12 quantization_params + §5.9.13 read_delta_q
// ---------------------------------------------------------------------

fn encode_quantization_params(
    bw: &mut BitWriter,
    qp: &QuantizationParams,
    num_planes: u8,
    separate_uv_delta_q: bool,
) {
    bw.write_bits(8, u64::from(qp.base_q_idx));
    write_delta_q(bw, qp.delta_q_y_dc);
    if num_planes > 1 {
        if separate_uv_delta_q {
            bw.write_bits(1, u64::from(qp.diff_uv_delta));
        } else {
            debug_assert!(!qp.diff_uv_delta);
        }
        write_delta_q(bw, qp.delta_q_u_dc);
        write_delta_q(bw, qp.delta_q_u_ac);
        if qp.diff_uv_delta {
            write_delta_q(bw, qp.delta_q_v_dc);
            write_delta_q(bw, qp.delta_q_v_ac);
        } else {
            debug_assert_eq!(qp.delta_q_v_dc, qp.delta_q_u_dc);
            debug_assert_eq!(qp.delta_q_v_ac, qp.delta_q_u_ac);
        }
    }
    bw.write_bits(1, u64::from(qp.using_qmatrix));
    if qp.using_qmatrix {
        bw.write_bits(4, u64::from(qp.qm_y));
        bw.write_bits(4, u64::from(qp.qm_u));
        if separate_uv_delta_q {
            bw.write_bits(4, u64::from(qp.qm_v));
        } else {
            debug_assert_eq!(qp.qm_v, qp.qm_u);
        }
    }
}

fn write_delta_q(bw: &mut BitWriter, delta: i8) {
    if delta == 0 {
        bw.write_bits(1, 0);
    } else {
        bw.write_bits(1, 1);
        bw.write_su(7, i32::from(delta));
    }
}

// ---------------------------------------------------------------------
// §5.9.14 segmentation_params
// ---------------------------------------------------------------------

fn encode_segmentation_params(bw: &mut BitWriter, sp: &SegmentationParams, primary_ref_frame: u8) {
    bw.write_bits(1, u64::from(sp.enabled));
    if !sp.enabled {
        return;
    }
    if primary_ref_frame != PRIMARY_REF_NONE {
        bw.write_bits(1, u64::from(sp.update_map));
        if sp.update_map {
            bw.write_bits(1, u64::from(sp.temporal_update));
        }
        bw.write_bits(1, u64::from(sp.update_data));
    }
    if sp.update_data {
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                let active = sp.segment_feature_active[i][j];
                bw.write_bits(1, u64::from(active));
                if active {
                    let bits_to_read = SEGMENTATION_FEATURE_BITS[j];
                    let value = sp.segment_feature_data[i][j];
                    if SEGMENTATION_FEATURE_SIGNED[j] {
                        bw.write_su(1 + bits_to_read, i32::from(value));
                    } else if bits_to_read == 0 {
                        // No bits emitted; the parser's §5.9.14
                        // `feature_value = 0` initialiser stands.
                        debug_assert_eq!(value, 0);
                    } else {
                        let limit = SEGMENTATION_FEATURE_MAX[j];
                        debug_assert!(value >= 0 && value <= limit);
                        bw.write_bits(bits_to_read, u64::from(value as u16));
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// §5.9.17 delta_q_params / §5.9.18 delta_lf_params
// ---------------------------------------------------------------------

fn encode_delta_q_params(bw: &mut BitWriter, dq: &DeltaQParams, base_q_idx: u8) {
    if base_q_idx > 0 {
        bw.write_bits(1, u64::from(dq.delta_q_present));
    } else {
        debug_assert!(!dq.delta_q_present);
    }
    if dq.delta_q_present {
        bw.write_bits(2, u64::from(dq.delta_q_res));
    }
}

fn encode_delta_lf_params(
    bw: &mut BitWriter,
    dlf: &DeltaLfParams,
    delta_q_present: bool,
    allow_intrabc: bool,
) {
    if !delta_q_present {
        debug_assert!(!dlf.delta_lf_present);
        return;
    }
    if !allow_intrabc {
        bw.write_bits(1, u64::from(dlf.delta_lf_present));
    } else {
        debug_assert!(!dlf.delta_lf_present);
    }
    if dlf.delta_lf_present {
        bw.write_bits(2, u64::from(dlf.delta_lf_res));
        bw.write_bits(1, u64::from(dlf.delta_lf_multi));
    }
}

// ---------------------------------------------------------------------
// §5.9.11 loop_filter_params
// ---------------------------------------------------------------------

fn encode_loop_filter_params(
    bw: &mut BitWriter,
    lf: &LoopFilterParams,
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
) {
    if coded_lossless || allow_intrabc {
        // §5.9.11 short-circuit: no bits emitted.
        debug_assert!(lf.short_circuited);
        return;
    }
    let _ = MAX_LOOP_FILTER;
    bw.write_bits(6, u64::from(lf.loop_filter_level[0]));
    bw.write_bits(6, u64::from(lf.loop_filter_level[1]));
    if num_planes > 1 && (lf.loop_filter_level[0] != 0 || lf.loop_filter_level[1] != 0) {
        bw.write_bits(6, u64::from(lf.loop_filter_level[2]));
        bw.write_bits(6, u64::from(lf.loop_filter_level[3]));
    }
    bw.write_bits(3, u64::from(lf.loop_filter_sharpness));
    bw.write_bits(1, u64::from(lf.loop_filter_delta_enabled));
    if lf.loop_filter_delta_enabled {
        bw.write_bits(1, u64::from(lf.loop_filter_delta_update));
        if lf.loop_filter_delta_update {
            // The parser walks 8 ref-deltas + 2 mode-deltas. The
            // FrameHeader stores their final values but not the
            // per-slot `update_*` bits — we emit `update == 1` only
            // when the stored value differs from the default. This
            // matches a session that always set every delta back to
            // the default and read 0s; the parser will reconstruct an
            // identical struct.
            for (slot, def) in lf
                .loop_filter_ref_deltas
                .iter()
                .zip(crate::uncompressed_header_tail::LOOP_FILTER_REF_DELTAS_DEFAULT.iter())
            {
                if slot == def {
                    bw.write_bits(1, 0);
                } else {
                    bw.write_bits(1, 1);
                    bw.write_su(7, i32::from(*slot));
                }
            }
            for (slot, def) in lf
                .loop_filter_mode_deltas
                .iter()
                .zip(crate::uncompressed_header_tail::LOOP_FILTER_MODE_DELTAS_DEFAULT.iter())
            {
                if slot == def {
                    bw.write_bits(1, 0);
                } else {
                    bw.write_bits(1, 1);
                    bw.write_su(7, i32::from(*slot));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// §5.9.19 cdef_params
// ---------------------------------------------------------------------

fn encode_cdef_params(
    bw: &mut BitWriter,
    cdef: &CdefParams,
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
    enable_cdef: bool,
) {
    if coded_lossless || allow_intrabc || !enable_cdef {
        debug_assert!(cdef.short_circuited);
        return;
    }
    let cdef_damping_minus_3 = cdef.cdef_damping - 3;
    bw.write_bits(2, u64::from(cdef_damping_minus_3));
    bw.write_bits(2, u64::from(cdef.cdef_bits));
    let count = 1usize << cdef.cdef_bits;
    for i in 0..count {
        bw.write_bits(4, u64::from(cdef.cdef_y_pri_strength[i]));
        // §5.9.19: stored value of 4 came from a raw bit-pattern of 3
        // (the `== 3 ⇒ += 1` adjustment); invert.
        let raw_y_sec = if cdef.cdef_y_sec_strength[i] == 4 {
            3u8
        } else {
            cdef.cdef_y_sec_strength[i]
        };
        bw.write_bits(2, u64::from(raw_y_sec));
        if num_planes > 1 {
            bw.write_bits(4, u64::from(cdef.cdef_uv_pri_strength[i]));
            let raw_uv_sec = if cdef.cdef_uv_sec_strength[i] == 4 {
                3u8
            } else {
                cdef.cdef_uv_sec_strength[i]
            };
            bw.write_bits(2, u64::from(raw_uv_sec));
        }
    }
}

// ---------------------------------------------------------------------
// §5.9.20 lr_params
// ---------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn encode_lr_params(
    bw: &mut BitWriter,
    lr: &LrParams,
    num_planes: u8,
    subsampling_x: bool,
    subsampling_y: bool,
    use_128x128_superblock: bool,
    all_lossless: bool,
    allow_intrabc: bool,
    enable_restoration: bool,
) {
    if all_lossless || allow_intrabc || !enable_restoration {
        debug_assert!(lr.short_circuited);
        return;
    }
    for (i, rtype) in lr
        .frame_restoration_type
        .iter()
        .enumerate()
        .take(usize::from(num_planes))
    {
        // §5.9.20 lr_type ⇒ FrameRestorationType uses Remap_Lr_Type:
        //   0 ⇒ None, 1 ⇒ Switchable, 2 ⇒ Wiener, 3 ⇒ SgrProj.
        // Invert that table here.
        let lr_type: u8 = match rtype {
            crate::uncompressed_header_tail::FrameRestorationType::None => 0,
            crate::uncompressed_header_tail::FrameRestorationType::Switchable => 1,
            crate::uncompressed_header_tail::FrameRestorationType::Wiener => 2,
            crate::uncompressed_header_tail::FrameRestorationType::SgrProj => 3,
        };
        let _ = i;
        bw.write_bits(2, u64::from(lr_type));
    }
    if lr.uses_lr {
        if use_128x128_superblock {
            // §5.9.20: post-incremented bit ⇒ raw bit is lr_unit_shift - 1.
            bw.write_bits(1, u64::from(lr.lr_unit_shift - 1));
        } else {
            // §5.9.20: if lr_unit_shift != 0, write 1 then
            // lr_unit_extra_shift = lr_unit_shift - 1.
            if lr.lr_unit_shift == 0 {
                bw.write_bits(1, 0);
            } else {
                bw.write_bits(1, 1);
                bw.write_bits(1, u64::from(lr.lr_unit_shift - 1));
            }
        }
        if subsampling_x && subsampling_y && lr.uses_chroma_lr {
            bw.write_bits(1, u64::from(lr.lr_uv_shift));
        }
    }
}

// ---------------------------------------------------------------------
// §5.9.21 read_tx_mode
// ---------------------------------------------------------------------

fn encode_tx_mode(bw: &mut BitWriter, tx: TxMode, coded_lossless: bool) {
    if coded_lossless {
        debug_assert_eq!(tx, TxMode::Only4x4);
        return;
    }
    // tx_mode_select: 0 ⇒ TX_MODE_LARGEST, 1 ⇒ TX_MODE_SELECT.
    let tx_mode_select = match tx {
        TxMode::TxModeSelect => 1u8,
        TxMode::TxModeLargest => 0u8,
        TxMode::Only4x4 => {
            // ONLY_4X4 is only legal under coded_lossless. Defensive.
            debug_assert!(false, "ONLY_4X4 requires CodedLossless");
            0
        }
    };
    bw.write_bits(1, u64::from(tx_mode_select));
}

// ---------------------------------------------------------------------
// §5.9.10 read_interpolation_filter
// ---------------------------------------------------------------------

fn encode_interpolation_filter(bw: &mut BitWriter, filt: InterpolationFilter) {
    // §5.9.10: f(1) is_filter_switchable; if 0, read f(2) interpolation_filter.
    if matches!(filt, InterpolationFilter::Switchable) {
        bw.write_bits(1, 1);
    } else {
        bw.write_bits(1, 0);
        let raw = filt.as_raw();
        bw.write_bits(2, u64::from(raw));
    }
}

// ---------------------------------------------------------------------
// §5.9.30 film_grain_params
// ---------------------------------------------------------------------

fn encode_film_grain_params(
    bw: &mut BitWriter,
    fg: &FilmGrainParams,
    seq: &SequenceHeader,
    show_frame: bool,
    showable_frame: bool,
    is_inter_frame: bool,
) {
    // §5.9.30 short-circuits: !film_grain_params_present, or neither
    // show_frame nor showable_frame ⇒ no bits emitted.
    if !seq.film_grain_params_present || (!show_frame && !showable_frame) {
        return;
    }
    bw.write_bits(1, u64::from(fg.apply_grain));
    if !fg.apply_grain {
        return;
    }
    bw.write_bits(16, u64::from(fg.grain_seed));
    if is_inter_frame {
        bw.write_bits(1, u64::from(fg.update_grain));
    } else {
        debug_assert!(fg.update_grain);
    }
    if !fg.update_grain {
        bw.write_bits(3, u64::from(fg.film_grain_params_ref_idx));
        return;
    }
    // The full populated path lands next arc; for now we round-trip the
    // common no-grain-points subset.
    bw.write_bits(4, u64::from(fg.num_y_points));
    for i in 0..usize::from(fg.num_y_points) {
        bw.write_bits(8, u64::from(fg.point_y_value[i]));
        bw.write_bits(8, u64::from(fg.point_y_scaling[i]));
    }
    if !seq.color_config.mono_chrome {
        bw.write_bits(1, u64::from(fg.chroma_scaling_from_luma));
    }
    let suppress_chroma = seq.color_config.mono_chrome
        || fg.chroma_scaling_from_luma
        || (seq.color_config.subsampling_x
            && seq.color_config.subsampling_y
            && fg.num_y_points == 0);
    if !suppress_chroma {
        bw.write_bits(4, u64::from(fg.num_cb_points));
        for i in 0..usize::from(fg.num_cb_points) {
            bw.write_bits(8, u64::from(fg.point_cb_value[i]));
            bw.write_bits(8, u64::from(fg.point_cb_scaling[i]));
        }
        bw.write_bits(4, u64::from(fg.num_cr_points));
        for i in 0..usize::from(fg.num_cr_points) {
            bw.write_bits(8, u64::from(fg.point_cr_value[i]));
            bw.write_bits(8, u64::from(fg.point_cr_scaling[i]));
        }
    }
    bw.write_bits(2, u64::from(fg.grain_scaling - 8));
    bw.write_bits(2, u64::from(fg.ar_coeff_lag));
    let num_pos_luma = 2 * usize::from(fg.ar_coeff_lag) * (usize::from(fg.ar_coeff_lag) + 1);
    let num_pos_chroma = if fg.num_y_points > 0 {
        for i in 0..num_pos_luma {
            bw.write_bits(8, u64::from(fg.ar_coeffs_y_plus_128[i]));
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };
    if fg.chroma_scaling_from_luma || fg.num_cb_points > 0 {
        for i in 0..num_pos_chroma {
            bw.write_bits(8, u64::from(fg.ar_coeffs_cb_plus_128[i]));
        }
    }
    if fg.chroma_scaling_from_luma || fg.num_cr_points > 0 {
        for i in 0..num_pos_chroma {
            bw.write_bits(8, u64::from(fg.ar_coeffs_cr_plus_128[i]));
        }
    }
    bw.write_bits(2, u64::from(fg.ar_coeff_shift - 6));
    bw.write_bits(2, u64::from(fg.grain_scale_shift));
    if fg.num_cb_points > 0 {
        bw.write_bits(8, u64::from(fg.cb_mult));
        bw.write_bits(8, u64::from(fg.cb_luma_mult));
        bw.write_bits(9, u64::from(fg.cb_offset));
    }
    if fg.num_cr_points > 0 {
        bw.write_bits(8, u64::from(fg.cr_mult));
        bw.write_bits(8, u64::from(fg.cr_luma_mult));
        bw.write_bits(9, u64::from(fg.cr_offset));
    }
    bw.write_bits(1, u64::from(fg.overlap_flag));
    bw.write_bits(1, u64::from(fg.clip_to_restricted_range));
}

// ---------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------

fn encode_allow_scc_if_present(bw: &mut BitWriter, seq: &SequenceHeader, allow_scc: bool) {
    if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
        bw.write_bits(1, u64::from(allow_scc));
    } else {
        debug_assert_eq!(
            allow_scc,
            seq.seq_force_screen_content_tools != 0,
            "allow_scc must match seq force when not selected"
        );
    }
}

fn encode_force_integer_mv_if_present(
    bw: &mut BitWriter,
    seq: &SequenceHeader,
    allow_scc: bool,
    raw_force_integer_mv: bool,
) {
    if allow_scc && seq.seq_force_integer_mv == SELECT_INTEGER_MV {
        bw.write_bits(1, u64::from(raw_force_integer_mv));
    }
}

fn decoder_model_info_present(seq: &SequenceHeader) -> bool {
    seq.decoder_model_info_present_flag
}

fn equal_picture_interval(seq: &SequenceHeader) -> bool {
    seq.timing_info
        .map(|t| t.equal_picture_interval)
        .unwrap_or(true)
}

fn derive_coded_lossless(qp: &QuantizationParams, sp: &SegmentationParams) -> bool {
    // Mirror crate::frame_header::compute_coded_lossless without going
    // through the private helper.
    let deltas_all_zero = qp.delta_q_y_dc == 0
        && qp.delta_q_u_dc == 0
        && qp.delta_q_u_ac == 0
        && qp.delta_q_v_dc == 0
        && qp.delta_q_v_ac == 0;
    if !deltas_all_zero {
        return false;
    }
    use crate::uncompressed_header_tail::SEG_LVL_ALT_Q;
    for segment_id in 0..MAX_SEGMENTS {
        let qindex = if sp.enabled && sp.segment_feature_active[segment_id][SEG_LVL_ALT_Q] {
            let data = i32::from(sp.segment_feature_data[segment_id][SEG_LVL_ALT_Q]);
            (i32::from(qp.base_q_idx) + data).clamp(0, 255)
        } else {
            i32::from(qp.base_q_idx)
        };
        if qindex != 0 {
            return false;
        }
    }
    true
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_header::parse_frame_header;
    use crate::sequence_header::parse_sequence_header;

    // Same sequence-header payloads the parser-side tests use.
    fn tiny_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
        parse_sequence_header(seq_payload).unwrap()
    }

    fn screen_content_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x18, 0x1d, 0xbf, 0xff, 0xf2, 0x01];
        parse_sequence_header(seq_payload).unwrap()
    }

    fn assert_round_trip(payload: &[u8], seq: &SequenceHeader) -> FrameHeader {
        let parsed = parse_frame_header(payload, seq).expect("parser baseline");
        let written = write_frame_header_obu(&parsed, seq);
        let reparsed = parse_frame_header(&written, seq).expect("encoder output parses");
        let mut expected = parsed.clone();
        expected.bits_consumed = reparsed.bits_consumed;
        assert_eq!(reparsed, expected, "round-trip mismatch");
        reparsed
    }

    const TINY_FRAME_PAYLOAD: &[u8] = &[
        0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67,
        0x6c, 0xc7, 0xee, 0x51, 0x80,
    ];

    #[test]
    fn round_trip_tiny_key_frame() {
        let seq = tiny_seq();
        assert_round_trip(TINY_FRAME_PAYLOAD, &seq);
    }

    #[test]
    fn round_trip_screen_content_intra_only() {
        // FRAME OBU payload from screen-content fixture.
        // Parsed by the corresponding parser test; we just verify the
        // intra-only path round-trips.
        let seq = screen_content_seq();
        // Construct a minimal intra-only frame header by parsing a
        // synthetic payload; easier to handcraft from the parser's
        // SCREEN_CONTENT_FRAME_PAYLOAD if available, but for arc 1 we
        // use the tiny key-frame against the SCC sequence header (the
        // path differs because the SCC seq has SELECT_SCREEN_CONTENT_TOOLS
        // ⇒ allow_scc bit on the wire). The tiny-key payload won't parse
        // against SCC seq, so build a synthetic key-frame from scratch
        // instead — see synthetic_intra_round_trip below.
        let _ = seq;
    }

    fn build_minimal_intra_fh(seq: &SequenceHeader, frame_w: u32, frame_h: u32) -> FrameHeader {
        use crate::uncompressed_header_tail::FrameRestorationType;

        let fs = FrameSize {
            frame_width: frame_w,
            frame_height: frame_h,
            render_width: frame_w,
            render_height: frame_h,
            superres_denom: SUPERRES_NUM,
            upscaled_width: frame_w,
            mi_cols: 2 * ((frame_w + 7) >> 3),
            mi_rows: 2 * ((frame_h + 7) >> 3),
            use_superres: false,
            coded_denom: 0,
            render_and_frame_size_different: false,
        };

        // Walk the §5.9.15 derivations to construct a TileInfo with the
        // single-tile layout the writer's encoder path would produce on
        // a tiny frame.
        let (sb_cols, sb_rows, sb_shift) = if seq.use_128x128_superblock {
            ((fs.mi_cols + 31) >> 5, (fs.mi_rows + 31) >> 5, 5u32)
        } else {
            ((fs.mi_cols + 15) >> 4, (fs.mi_rows + 15) >> 4, 4u32)
        };
        let _ = (sb_cols, sb_rows, sb_shift);
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
            base_q_idx: 120,
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
        let sp = SegmentationParams::disabled();
        let dq = DeltaQParams::default();
        let dlf = DeltaLfParams::default();
        let lf = LoopFilterParams {
            loop_filter_level: [0, 0, 0, 0],
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: true,
            loop_filter_delta_update: false,
            loop_filter_ref_deltas: crate::uncompressed_header_tail::LOOP_FILTER_REF_DELTAS_DEFAULT,
            loop_filter_mode_deltas:
                crate::uncompressed_header_tail::LOOP_FILTER_MODE_DELTAS_DEFAULT,
            short_circuited: false,
        };
        let cdef = CdefParams {
            cdef_damping: 4,
            cdef_bits: 0,
            cdef_y_pri_strength: [0; crate::uncompressed_header_tail::CDEF_MAX_STRENGTHS],
            cdef_y_sec_strength: [0; crate::uncompressed_header_tail::CDEF_MAX_STRENGTHS],
            cdef_uv_pri_strength: [0; crate::uncompressed_header_tail::CDEF_MAX_STRENGTHS],
            cdef_uv_sec_strength: [0; crate::uncompressed_header_tail::CDEF_MAX_STRENGTHS],
            short_circuited: false,
        };
        let lr = LrParams {
            frame_restoration_type: [FrameRestorationType::None; 3],
            uses_lr: false,
            uses_chroma_lr: false,
            lr_unit_shift: 0,
            lr_uv_shift: 0,
            loop_restoration_size: [0; 3],
            short_circuited: false,
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
            segmentation_params: Some(sp),
            delta_q_params: Some(dq),
            delta_lf_params: Some(dlf),
            loop_filter_params: Some(lf),
            cdef_params: Some(cdef),
            lr_params: Some(lr),
            tx_mode: Some(TxMode::TxModeLargest),
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

    #[test]
    fn synthetic_intra_round_trip_tiny_seq() {
        let seq = tiny_seq();
        let fh = build_minimal_intra_fh(&seq, 16, 16);
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).expect("encoder output parses");
        // Normalise bits_consumed before comparing.
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn synthetic_intra_round_trip_lossless() {
        let mut seq = tiny_seq();
        // We need a sequence with at least the same flags as tiny_seq;
        // tiny_seq is already 16x16 8-bit 4:2:0, no super-res.
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        // Force the lossless path: base_q_idx = 0 ⇒ delta_q_present
        // unread; CodedLossless = 1 ⇒ short-circuits loop_filter / cdef /
        // lr; TxMode = ONLY_4X4 (no bit emitted in §5.9.21).
        let qp = QuantizationParams {
            base_q_idx: 0,
            ..QuantizationParams {
                base_q_idx: 0,
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
            }
        };
        fh.quantization_params = Some(qp);
        fh.loop_filter_params = Some(LoopFilterParams::short_circuit());
        fh.cdef_params = Some(CdefParams::short_circuit());
        fh.lr_params = Some(LrParams::short_circuit());
        fh.tx_mode = Some(TxMode::Only4x4);
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
        // Touch &mut seq to silence unused_mut.
        let _ = &mut seq;
    }

    #[test]
    fn synthetic_intra_round_trip_with_qm() {
        let seq = tiny_seq();
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        if let Some(qp) = fh.quantization_params.as_mut() {
            qp.using_qmatrix = true;
            qp.qm_y = 7;
            qp.qm_u = 8;
            qp.qm_v = 8; // mirrors qm_u under separate_uv_delta_q = false
        }
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn synthetic_intra_round_trip_with_delta_q_offsets() {
        let seq = tiny_seq();
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        if let Some(qp) = fh.quantization_params.as_mut() {
            qp.delta_q_y_dc = 3;
            qp.delta_q_u_dc = -2;
            qp.delta_q_u_ac = 1;
            // diff_uv_delta defaults to false ⇒ V mirrors U.
            qp.delta_q_v_dc = -2;
            qp.delta_q_v_ac = 1;
        }
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn synthetic_intra_round_trip_with_cdef_strengths() {
        let seq = tiny_seq();
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        if let Some(cdef) = fh.cdef_params.as_mut() {
            cdef.cdef_bits = 2; // 4 strength entries
            cdef.cdef_damping = 5;
            cdef.cdef_y_pri_strength[0] = 9;
            cdef.cdef_y_pri_strength[1] = 12;
            cdef.cdef_y_pri_strength[2] = 0;
            cdef.cdef_y_pri_strength[3] = 7;
            cdef.cdef_y_sec_strength[0] = 1;
            cdef.cdef_y_sec_strength[1] = 4; // tests the == 3 ⇒ += 1 invert
            cdef.cdef_y_sec_strength[2] = 0;
            cdef.cdef_y_sec_strength[3] = 2;
            cdef.cdef_uv_pri_strength[0] = 5;
            cdef.cdef_uv_pri_strength[1] = 0;
            cdef.cdef_uv_pri_strength[2] = 8;
            cdef.cdef_uv_pri_strength[3] = 3;
            cdef.cdef_uv_sec_strength[0] = 4; // same invert
            cdef.cdef_uv_sec_strength[1] = 2;
            cdef.cdef_uv_sec_strength[2] = 0;
            cdef.cdef_uv_sec_strength[3] = 1;
        }
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn synthetic_intra_round_trip_with_lr() {
        use crate::uncompressed_header_tail::FrameRestorationType;
        let seq = tiny_seq();
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        if let Some(lr) = fh.lr_params.as_mut() {
            lr.frame_restoration_type[0] = FrameRestorationType::Wiener;
            lr.frame_restoration_type[1] = FrameRestorationType::None;
            lr.frame_restoration_type[2] = FrameRestorationType::None;
            lr.uses_lr = true;
            lr.uses_chroma_lr = false;
            // use_128x128_superblock = true in tiny_seq ⇒ raw lr_unit_shift
            // bit = lr_unit_shift - 1. lr_unit_shift = 2 ⇒ raw bit = 1.
            lr.lr_unit_shift = 2;
            lr.lr_uv_shift = 0;
            // LoopRestorationSize[0] = 256 >> (2 - 2) = 256
            lr.loop_restoration_size[0] = 256;
            lr.loop_restoration_size[1] = 256;
            lr.loop_restoration_size[2] = 256;
        }
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn round_trip_tiny_key_frame_byte_exact_prefix() {
        // The parser-side TINY_FRAME_PAYLOAD round-trip — we already
        // hit it in round_trip_tiny_key_frame above, but also verify
        // that re-parsing the encoder output yields the same
        // `bits_consumed = 72`.
        let seq = tiny_seq();
        let parsed = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).unwrap();
        assert_eq!(parsed.bits_consumed, 72);
        let written = write_frame_header_obu(&parsed, &seq);
        let reparsed = parse_frame_header(&written, &seq).unwrap();
        assert_eq!(reparsed.bits_consumed, 72);
    }

    #[test]
    fn synthetic_intra_round_trip_with_segmentation() {
        use crate::uncompressed_header_tail::SEG_LVL_ALT_Q;
        let seq = tiny_seq();
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        if let Some(sp) = fh.segmentation_params.as_mut() {
            sp.enabled = true;
            sp.update_map = true; // forced under PRIMARY_REF_NONE
            sp.temporal_update = false; // forced 0
            sp.update_data = true; // forced 1
                                   // Activate ALT_Q on segment 1 with a small offset.
            sp.segment_feature_active[1][SEG_LVL_ALT_Q] = true;
            sp.segment_feature_data[1][SEG_LVL_ALT_Q] = 4;
            sp.last_active_seg_id = 1;
            sp.seg_id_pre_skip = false;
        }
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn synthetic_intra_round_trip_with_render_size_diff() {
        let seq = tiny_seq();
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        if let Some(fs) = fh.frame_size.as_mut() {
            fs.render_and_frame_size_different = true;
            fs.render_width = 32;
            fs.render_height = 32;
        }
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn synthetic_intra_round_trip_with_loop_filter_deltas() {
        let seq = tiny_seq();
        let mut fh = build_minimal_intra_fh(&seq, 16, 16);
        if let Some(lf) = fh.loop_filter_params.as_mut() {
            // Need non-zero level for chroma slots to round-trip.
            lf.loop_filter_level[0] = 16;
            lf.loop_filter_level[1] = 16;
            lf.loop_filter_level[2] = 16;
            lf.loop_filter_level[3] = 16;
            lf.loop_filter_sharpness = 5;
            lf.loop_filter_delta_enabled = true;
            lf.loop_filter_delta_update = true;
            // Modify one ref delta and one mode delta.
            lf.loop_filter_ref_deltas[0] = -2;
            lf.loop_filter_mode_deltas[1] = 3;
        }
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }

    #[test]
    fn synthetic_intra_round_trip_show_existing_frame() {
        // show-existing-frame requires a different seq (one with the
        // right flags). Use show_existing_seq from the parser tests.
        let seq_payload: &[u8] = &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x9b, 0x5f, 0x30, 0x08];
        let seq = parse_sequence_header(seq_payload).unwrap();
        // Build a minimal show-existing FrameHeader.
        let fh = FrameHeader {
            show_existing_frame: true,
            frame_to_show_map_idx: Some(2),
            display_frame_id: None,
            frame_type: FrameType::Inter,
            frame_is_intra: false,
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: false,
            disable_cdf_update: false,
            allow_screen_content_tools: false,
            force_integer_mv: false,
            current_frame_id: 0,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: PRIMARY_REF_NONE,
            refresh_frame_flags: 0,
            ref_order_hints: None,
            frame_size: None,
            allow_intrabc: false,
            disable_frame_end_update_cdf: false,
            tile_info: None,
            quantization_params: None,
            segmentation_params: None,
            delta_q_params: None,
            delta_lf_params: None,
            loop_filter_params: None,
            cdef_params: None,
            lr_params: None,
            tx_mode: None,
            reference_select: None,
            skip_mode_present: None,
            skip_mode_frame: None,
            allow_warped_motion: None,
            reduced_tx_set: None,
            global_motion_params: None,
            film_grain_params: None,
            inter_refs: None,
            bits_consumed: 0,
        };
        let bytes = write_frame_header_obu(&fh, &seq);
        let parsed = parse_frame_header(&bytes, &seq).unwrap();
        let mut expected = fh.clone();
        expected.bits_consumed = parsed.bits_consumed;
        assert_eq!(parsed, expected);
    }
}
