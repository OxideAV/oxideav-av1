//! AV1 frame header OBU parser — §5.9.
//!
//! This is the most syntactically heavy header in AV1. We parse every
//! field up to and including `tile_info()` (§5.9.15) — frame type,
//! dimensions (with superres), render size, intrabc, interpolation filter,
//! ref frame indices, tile column / row boundaries, and `TileSizeBytes`.
//! The post-tile_info sub-sections (quantization, segmentation, deblock,
//! CDEF, loop restoration, tx_mode, frame_reference_mode, skip_mode,
//! global motion, film grain) are not parsed here; they are only needed
//! for pixel reconstruction which lives outside this crate.
//!
//! The `frame_size_with_refs()` path (inter frame using reference-frame
//! state) requires cross-frame state we don't yet carry — that case
//! surfaces as `Error::Unsupported` with the exact §ref.

use oxideav_core::{Error, Result};

use crate::bitreader::{ceil_log2, BitReader};
use crate::dpb::{get_relative_dist, Dpb};
use crate::frame_header_tail::{
    coded_lossless_hint, parse_cdef_params, parse_delta_lf_params, parse_delta_q_params,
    parse_film_grain_params, parse_global_motion_params, parse_loop_filter_params, parse_lr_params,
    parse_quantization_params, parse_segmentation_params, parse_tx_mode, CdefParams,
    FilmGrainParams, GmType, LoopFilterParams, LoopRestorationParams, QuantizationParams,
    SegmentationParams, TxMode, PRIMARY_REF_NONE,
};
use crate::sequence_header::{SequenceHeader, SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};
use crate::tile_info::{mi_cols_rows, parse_tile_info, TileInfo};

pub const NUM_REF_FRAMES: usize = 8;
pub const REFS_PER_FRAME: usize = 7;
pub const SUPERRES_DENOM_BITS: u32 = 3;
pub const SUPERRES_DENOM_MIN: u32 = 9;
pub const SUPERRES_NUM: u32 = 8;

/// §3 reference-frame numbering — `INTRA_FRAME = 0`, `LAST_FRAME = 1`,
/// `LAST2 = 2`, `LAST3 = 3`, `GOLDEN = 4`, `BWDREF = 5`, `ALTREF2 = 6`,
/// `ALTREF = 7`. The skip-mode derivation in §5.9.21 stores
/// `SkipModeFrame[0..=1]` as `LAST_FRAME + i` where `i` is a
/// `ref_frame_idx` index in `0..REFS_PER_FRAME`.
pub const LAST_FRAME: u8 = 1;

/// `frame_type` values §5.9.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    Key,
    Inter,
    IntraOnly,
    Switch,
}

impl FrameType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            _ => Self::Switch,
        }
    }
}

/// Top-level uncompressed header. We expose the fields the caller is most
/// likely to need (frame type / dimensions / show flag) plus a `parse_depth`
/// marker indicating how far the parser successfully advanced.
#[derive(Clone, Debug)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub display_frame_id: u32,

    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,

    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: u32,
    pub force_integer_mv: u32,

    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,

    pub order_hint: u32,
    pub primary_ref_frame: u32,

    pub refresh_frame_flags: u8,
    pub ref_order_hint: [u32; NUM_REF_FRAMES],
    pub ref_frame_idx: [u32; REFS_PER_FRAME],

    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub use_superres: bool,
    pub superres_denom: u32,

    pub render_and_frame_size_different: bool,
    pub render_width: u32,
    pub render_height: u32,

    pub allow_intrabc: bool,
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    pub interpolation_filter: u32,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    pub disable_frame_end_update_cdf: bool,
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,

    /// Parsed `tile_info()` (§5.9.15), when the parser advanced that far.
    /// `None` for the `ShowExistingFrame` path and for frames where the
    /// parser stopped earlier due to an unresolved dependency.
    pub tile_info: Option<TileInfo>,

    /// §5.9.12 — quantisation params.
    pub quant: QuantizationParams,
    /// §5.9.14 — segmentation params.
    pub segmentation: SegmentationParams,
    /// §5.9.16 — `delta_q_params` present flag + resolution.
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    /// §5.9.16 — `delta_lf_params`.
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    /// §5.9.11 — loop-filter params.
    pub loop_filter: LoopFilterParams,
    /// §5.9.19 — CDEF params.
    pub cdef: CdefParams,
    /// §5.9.20 — loop-restoration params.
    pub lr: LoopRestorationParams,
    /// §5.9.21 — `read_tx_mode`.
    pub tx_mode: TxMode,
    /// §5.9.22 — `frame_reference_mode.reference_select`.
    pub reference_select: bool,
    /// §5.9.23 — `skip_mode_params.skip_mode_present`. Intra-only
    /// frames never enable skip mode; kept as a flag for inter
    /// expansion. Only ever `true` when `skip_mode_allowed` is `true`
    /// and the frame header's `skip_mode_present` bit was 1.
    pub skip_mode_present: bool,
    /// §5.9.21 derived `skipModeAllowed` — `true` iff the OrderHint
    /// trail of the chosen references brackets the current frame's
    /// `OrderHint` (forward + backward, or two distinct forward refs).
    /// Drives whether the bitstream emits the `skip_mode_present`
    /// bit at all (§5.9.22 last clause).
    pub skip_mode_allowed: bool,
    /// §5.9.21 / §7.9.2 `SkipModeFrame[0..=1]` — the two reference-
    /// frame slots a SKIP_MODE block uses for compound prediction.
    /// Stored as `LAST_FRAME + i` (i.e. logical reference indices
    /// `1..=7`); both entries are `0` when `skip_mode_allowed` is
    /// `false`. The convention `SkipModeFrame[0] < SkipModeFrame[1]`
    /// is preserved per spec (`Min`/`Max` of the bracketing pair).
    pub skip_mode_frame: [u8; 2],
    /// §5.9.24 — `global_motion_params.gm_type[ref]`. Only populated
    /// for inter frames; `Identity` for intra-only.
    pub gm_type: [GmType; NUM_REF_FRAMES],
    /// §5.9.24 — `global_motion_params.gm_params[ref][0..=5]`. Stored
    /// as signed integers at their spec precision (§5.9.27): `[0]/[1]`
    /// are the translation components (1/8-pel for TRANSLATION, 1/16-
    /// pel for higher types); `[2..=5]` are the alpha / affine warp
    /// coefficients. Only populated for inter frames.
    pub gm_params: [[i32; 6]; NUM_REF_FRAMES],
    /// §5.9.30 — film-grain params.
    pub film_grain: FilmGrainParams,

    /// Last successfully-parsed milestone — see `ParseDepth`.
    pub parse_depth: ParseDepth,
}

/// How far the frame_header parser advanced before yielding back to the
/// caller. Useful so a high-level decoder knows whether to treat the
/// remaining bytes as opaque.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseDepth {
    /// Stopped right after `show_existing_frame` short path.
    ShowExistingFrame,
    /// Fully parsed up to (but not including) `tile_info()`.
    UpToTileInfo,
    /// Parsed through `tile_info()` (§5.9.15). All tile byte-boundary
    /// information is available in `FrameHeader::tile_info`. Downstream
    /// parsing (quantization, segmentation, loop filter, CDEF, LR, …)
    /// is still out of scope for this crate.
    ThroughTileInfo,
    /// Parsed the full uncompressed header through `film_grain_params`
    /// (§5.9.30). Every sub-section is populated.
    Complete,
}

/// Parse a frame_header_obu / frame_obu payload. For OBU_FRAME the caller
/// must subsequently parse `tile_group_obu` from the remaining bytes.
///
/// Convenience entry point: passes an empty DPB so `skipModeAllowed`
/// is always derived as `false`. Use [`parse_frame_header_with_dpb`]
/// when reference OrderHints are tracked across frames.
pub fn parse_frame_header(seq: &SequenceHeader, payload: &[u8]) -> Result<FrameHeader> {
    parse_frame_header_with_dpb(seq, payload, &Dpb::new())
}

/// DPB-aware frame_header parser. The supplied [`Dpb`] supplies the
/// per-slot `RefOrderHint[]` values consulted by the §5.9.21 skip-mode
/// allowed derivation; an empty DPB falls through to
/// `skipModeAllowed = 0` exactly like the legacy entry point.
pub fn parse_frame_header_with_dpb(
    seq: &SequenceHeader,
    payload: &[u8],
    dpb: &Dpb,
) -> Result<FrameHeader> {
    let mut br = BitReader::new(payload);
    parse_uncompressed_header(seq, &mut br, dpb)
}

/// Parse a frame_obu() payload (§5.10): `frame_header_obu()` followed by
/// `byte_alignment()` then `tile_group_obu()`. Returns the parsed
/// `FrameHeader` plus the byte offset into `payload` at which the tile
/// group sub-OBU starts.
pub fn parse_frame_obu<'a>(
    seq: &SequenceHeader,
    payload: &'a [u8],
) -> Result<(FrameHeader, &'a [u8])> {
    parse_frame_obu_with_dpb(seq, payload, &Dpb::new())
}

/// DPB-aware [`parse_frame_obu`] variant — see
/// [`parse_frame_header_with_dpb`] for the DPB rationale.
pub fn parse_frame_obu_with_dpb<'a>(
    seq: &SequenceHeader,
    payload: &'a [u8],
    dpb: &Dpb,
) -> Result<(FrameHeader, &'a [u8])> {
    let mut br = BitReader::new(payload);
    let fh = parse_uncompressed_header(seq, &mut br, dpb)?;
    // `frame_obu()` applies byte_alignment() between the frame header and
    // the tile_group sub-OBU.
    br.byte_alignment()?;
    let header_bytes = (br.bit_position() / 8) as usize;
    if header_bytes > payload.len() {
        return Err(Error::invalid("av1 frame_obu: header longer than payload"));
    }
    Ok((fh, &payload[header_bytes..]))
}

/// `AV1_TRACE_BITS=1` env-gated bit-position trace used to bisect
/// uncompressed_header bit-account regressions. Returns `true` when
/// the env var is set to a non-empty, non-"0" value. The check is
/// performed lazily on each call (cheap; the eprintln branch is what
/// matters). Off by default — zero overhead in production builds.
fn av1_trace_bits() -> bool {
    match std::env::var("AV1_TRACE_BITS") {
        Ok(v) => !v.is_empty() && v != "0",
        Err(_) => false,
    }
}

/// §5.9.1 uncompressed_header().
fn parse_uncompressed_header(
    seq: &SequenceHeader,
    br: &mut BitReader<'_>,
    dpb: &Dpb,
) -> Result<FrameHeader> {
    let trace = av1_trace_bits();
    macro_rules! tp {
        ($br:expr, $tag:literal) => {
            if trace {
                eprintln!("AV1_TRACE_BITS: {} bit_pos={}", $tag, $br.bit_position());
            }
        };
    }
    tp!(br, "enter_uncompressed_header");
    let id_len = if seq.frame_id_numbers_present {
        seq.additional_frame_id_length_minus_1 as u32 + seq.delta_frame_id_length_minus_2 as u32 + 3
    } else {
        0
    };
    let all_frames = (1u32 << NUM_REF_FRAMES) - 1;

    // §5.9.1 — when `reduced_still_picture_header` is set (the common
    // AVIF still case) the show_existing_frame / frame_type / show_frame
    // / showable_frame bits are NOT coded. The spec dictates fixed
    // defaults and parsing continues on to the normal tail (q / seg /
    // tile_info / …). Previous versions returned ParseDepth::ShowExisting-
    // Frame with tile_info: None here, which caused AVIF stills to
    // surface a "parse-only build" error from decoder.rs.
    let (frame_type, show_frame, showable_frame) = if seq.reduced_still_picture_header {
        (FrameType::Key, true, false)
    } else {
        let show_existing_frame = br.bit()?;
        if show_existing_frame {
            let frame_to_show_map_idx = br.f(3)? as u8;
            if let Some(info) = seq.decoder_model_info {
                if seq.decoder_model_info_present
                    && seq
                        .timing_info
                        .map(|t| !t.equal_picture_interval)
                        .unwrap_or(true)
                {
                    let _ = br.f(info.frame_presentation_time_length_minus_1 as u32 + 1)?;
                }
            }
            let display_frame_id = if seq.frame_id_numbers_present {
                br.f(id_len)?
            } else {
                0
            };
            // show_existing_frame takes subsequent state from the DPB's
            // RefFrame[frame_to_show_map_idx] which we don't track yet.
            return finish_minimal(
                true,
                frame_to_show_map_idx,
                display_frame_id,
                FrameType::Key,
                true,
                true,
                false,
                ParseDepth::ShowExistingFrame,
            );
        }
        let frame_type = FrameType::from_u32(br.f(2)?);
        let show_frame = br.bit()?;
        if show_frame
            && seq.decoder_model_info_present
            && seq
                .timing_info
                .map(|t| !t.equal_picture_interval)
                .unwrap_or(true)
        {
            if let Some(info) = seq.decoder_model_info {
                let _ = br.f(info.frame_presentation_time_length_minus_1 as u32 + 1)?;
            }
        }
        let showable_frame = if show_frame {
            frame_type != FrameType::Key
        } else {
            br.bit()?
        };
        (frame_type, show_frame, showable_frame)
    };
    let error_resilient_mode =
        if frame_type == FrameType::Switch || (frame_type == FrameType::Key && show_frame) {
            true
        } else {
            br.bit()?
        };

    let disable_cdf_update = br.bit()?;
    tp!(br, "after_disable_cdf_update");
    let allow_screen_content_tools =
        if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
            br.f(1)?
        } else {
            seq.seq_force_screen_content_tools
        };
    let force_integer_mv = if allow_screen_content_tools != 0 {
        if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            br.f(1)?
        } else {
            seq.seq_force_integer_mv
        }
    } else {
        // Per §5.9.1: if frame_type intra-only/key force_integer_mv is 1
        match frame_type {
            FrameType::Key | FrameType::IntraOnly => 1,
            _ => 0,
        }
    };
    let mut current_frame_id = 0u32;
    if seq.frame_id_numbers_present {
        current_frame_id = br.f(id_len)?;
    }
    let frame_size_override_flag = if frame_type == FrameType::Switch {
        true
    } else if seq.reduced_still_picture_header {
        false
    } else {
        br.bit()?
    };
    let order_hint = if seq.enable_order_hint {
        br.f(seq.order_hint_bits)?
    } else {
        0
    };
    tp!(br, "after_order_hint");
    let primary_ref_frame = if frame_type == FrameType::Key
        || frame_type == FrameType::IntraOnly
        || error_resilient_mode
    {
        7 // PRIMARY_REF_NONE
    } else {
        br.f(3)?
    };
    tp!(br, "after_primary_ref_frame");

    // Decoder model buffer-removal-time (§5.9.4): we skim the structure but
    // do not retain values.
    if seq.decoder_model_info_present {
        let buffer_removal_time_present_flag = br.bit()?;
        if buffer_removal_time_present_flag {
            // For each operating point: if op idc and op_pt_idc test passes...
            for op in &seq.operating_points {
                if op.decoder_model_present {
                    if let Some(info) = seq.decoder_model_info {
                        let _t = br.f(info.buffer_removal_time_length_minus_1 as u32 + 1)?;
                    }
                }
            }
        }
    }

    let refresh_frame_flags = if frame_type == FrameType::Key && show_frame {
        all_frames as u8
    } else {
        br.f(8)? as u8
    };
    tp!(br, "after_refresh_frame_flags");

    // §5.9.2 — ref_order_hint is gated by `error_resilient_mode &&
    // enable_order_hint` (under the outer `!FrameIsIntra ||
    // refresh_frame_flags != allFrames` umbrella). Round 17 fixed an
    // over-read here that consumed 8 * OrderHintBits extra bits on every
    // non-error-resilient inter frame — the over-read happened to land
    // inside the long key/inter payloads from aomenc but tripped the
    // bitreader on SVT-AV1's small "non-shown overlay" frame OBUs (pkt
    // 1 frames 3-5 of `/tmp/av1_inter.ivf`).
    let frame_is_intra_for_check = matches!(frame_type, FrameType::Key | FrameType::IntraOnly);
    let mut ref_order_hint = [0u32; NUM_REF_FRAMES];
    if (!frame_is_intra_for_check || refresh_frame_flags != all_frames as u8)
        && error_resilient_mode
        && seq.enable_order_hint
    {
        for v in ref_order_hint.iter_mut() {
            *v = br.f(seq.order_hint_bits)?;
        }
    }

    // ---- §5.9.5/.6 Frame size + render size + intrabc / ref_frame_idx
    // loop. The spec ORDER differs sharply between intra and inter:
    //
    //   Intra (Key / IntraOnly):
    //     frame_size()
    //     render_size()
    //     allow_intrabc (gated)
    //
    //   Inter (Inter / Switch):
    //     frame_refs_short_signaling
    //     [last/gold idx + set_frame_refs]
    //     ref_frame_idx[] loop
    //     if (frame_size_override_flag && !error_resilient_mode)
    //         frame_size_with_refs()
    //     else { frame_size(); render_size(); }
    //     allow_high_precision_mv
    //     read_interpolation_filter()
    //     is_motion_mode_switchable
    //     use_ref_frame_mvs
    //
    // Round 21 fix: previously we called frame_size + render_size BEFORE
    // the ref_frame_idx loop for inter frames. The bit count for an
    // override-off / superres-off frame happens to match spec (1 bit for
    // render_diff), but the bit POSITION shifts by ~13 bits because
    // spec's `frame_refs_short_signaling=0` path consumes 21 ref_frame_idx
    // bits. The misalignment caused 10/48 SVT-AV1 chain frames to mis-
    // interpret tile_group bits as gm_params type bits.
    let mut ref_frame_idx = [0u32; REFS_PER_FRAME];
    let mut allow_high_precision_mv = false;
    let mut is_filter_switchable = false;
    let mut interpolation_filter = 0u32;
    let mut is_motion_mode_switchable = false;
    let mut use_ref_frame_mvs = false;
    let (frame_width, frame_height, upscaled_width, use_superres, superres_denom);
    let render_and_frame_size_different;
    let render_width;
    let render_height;
    let allow_intrabc;
    if frame_type == FrameType::Key || frame_type == FrameType::IntraOnly {
        let (fw, fh) = parse_frame_size(seq, br, frame_size_override_flag)?;
        let (sup, denom, upscaled) = parse_superres(seq, br, fw)?;
        frame_width = sup;
        frame_height = fh;
        upscaled_width = upscaled;
        use_superres = denom != SUPERRES_NUM;
        superres_denom = denom;
        let (rdiff, rw, rh) = parse_render_size(br, frame_width, frame_height)?;
        render_and_frame_size_different = rdiff;
        render_width = rw;
        render_height = rh;
        tp!(br, "after_render_size");
        // §5.9.1 — allow_intrabc only when intra_only/key + screen
        // content tools active + no superres scaling.
        allow_intrabc = if allow_screen_content_tools != 0 && frame_width == upscaled_width {
            br.bit()?
        } else {
            false
        };
    } else {
        // Inter / Switch: ref_frame_idx loop, THEN frame_size+render_size.
        tp!(br, "before_frame_refs_block");
        let frame_refs_short_signaling = if seq.enable_order_hint {
            br.bit()?
        } else {
            false
        };
        if frame_refs_short_signaling {
            let last_frame_idx = br.f(3)? as u8;
            let gold_frame_idx = br.f(3)? as u8;
            // §7.8 set_frame_refs(): compute the remaining
            // ref_frame_idx[] entries from the DPB's RefOrderHint[]
            // trail. Bit-neutral (no bitstream side effects) but lets
            // the §7.20 PrevGmParams chain consult the correct slot
            // for chained inter frames.
            ref_frame_idx = set_frame_refs(
                last_frame_idx,
                gold_frame_idx,
                order_hint,
                seq.order_hint_bits,
                dpb,
            );
        }
        for v in ref_frame_idx.iter_mut() {
            if !frame_refs_short_signaling {
                *v = br.f(3)?;
            }
            if seq.frame_id_numbers_present {
                let n = seq.delta_frame_id_length_minus_2 as u32 + 2;
                let _delta_frame_id_minus_1 = br.f(n)?;
            }
        }
        // §5.9.7 frame_size_with_refs OR §5.9.5 frame_size + §5.9.6
        // render_size, exactly per spec.
        if frame_size_override_flag && !error_resilient_mode {
            // §5.9.7: 7 found_ref bits, then if any set, inherit from
            // that ref slot's saved size; otherwise fall through to
            // frame_size + render_size + superres + compute_image_size.
            // We don't track per-ref RefFrameWidth/Height yet, so we
            // model the all-zero (no found_ref matches) case which is
            // valid per spec — the chain test fixture happens to never
            // hit this branch.
            let mut any_found = false;
            for _ in 0..REFS_PER_FRAME {
                if br.bit()? {
                    any_found = true;
                }
            }
            if any_found {
                return Err(Error::unsupported(
                    "av1 frame_size_with_refs (§5.9.7) found_ref=1 path requires \
                     RefFrameWidth/Height tracking — parse-only crate stops here",
                ));
            }
            // found_ref==0 fall-through: frame_size() + render_size()
            // + superres_params() + compute_image_size().
            let (fw, fh) = parse_frame_size(seq, br, frame_size_override_flag)?;
            let (sup, denom, upscaled) = parse_superres(seq, br, fw)?;
            frame_width = sup;
            frame_height = fh;
            upscaled_width = upscaled;
            use_superres = denom != SUPERRES_NUM;
            superres_denom = denom;
            let (rdiff, rw, rh) = parse_render_size(br, frame_width, frame_height)?;
            render_and_frame_size_different = rdiff;
            render_width = rw;
            render_height = rh;
        } else {
            let (fw, fh) = parse_frame_size(seq, br, frame_size_override_flag)?;
            let (sup, denom, upscaled) = parse_superres(seq, br, fw)?;
            frame_width = sup;
            frame_height = fh;
            upscaled_width = upscaled;
            use_superres = denom != SUPERRES_NUM;
            superres_denom = denom;
            let (rdiff, rw, rh) = parse_render_size(br, frame_width, frame_height)?;
            render_and_frame_size_different = rdiff;
            render_width = rw;
            render_height = rh;
        }
        tp!(br, "after_render_size");
        // skip allow_high_precision_mv-related branch fields
        allow_high_precision_mv = if force_integer_mv != 0 {
            false
        } else {
            br.bit()?
        };
        is_filter_switchable = br.bit()?;
        interpolation_filter = if is_filter_switchable { 4 } else { br.f(2)? };
        is_motion_mode_switchable = br.bit()?;
        use_ref_frame_mvs = if error_resilient_mode || !seq.enable_ref_frame_mvs {
            false
        } else {
            br.bit()?
        };
        tp!(br, "after_frame_refs_block");
        // Inter never enables intrabc.
        allow_intrabc = false;
    }

    let disable_frame_end_update_cdf = if seq.reduced_still_picture_header || disable_cdf_update {
        true
    } else {
        br.bit()?
    };
    tp!(br, "after_disable_frame_end_update_cdf");

    // §5.9.15 tile_info() — comes immediately after
    // `disable_frame_end_update_cdf`. The decoding-process calls in between
    // (`init_non_coeff_cdfs`, `motion_field_estimation`, …) do not read
    // bits from the stream.
    let (mi_cols, mi_rows) = mi_cols_rows(frame_width, frame_height);
    let tile_info = parse_tile_info(br, seq.use_128x128_superblock, mi_cols, mi_rows)?;
    tp!(br, "after_tile_info");

    // §5.9.12 – §5.9.30 — the entire uncompressed-header tail.
    let quant = parse_quantization_params(br, seq)?;
    tp!(br, "after_quant");
    let segmentation = parse_segmentation_params(br, primary_ref_frame)?;
    tp!(br, "after_segmentation");
    let (delta_q_present, delta_q_res) = parse_delta_q_params(br, quant.base_q_idx)?;
    tp!(br, "after_delta_q");
    let (delta_lf_present, delta_lf_res, delta_lf_multi) =
        parse_delta_lf_params(br, delta_q_present, allow_intrabc)?;
    tp!(br, "after_delta_lf");

    let coded_lossless = coded_lossless_hint(&quant);
    let loop_filter = parse_loop_filter_params(br, seq, &quant, allow_intrabc)?;
    tp!(br, "after_loop_filter");
    let cdef = parse_cdef_params(br, seq, &quant, allow_intrabc)?;
    tp!(br, "after_cdef");
    let lr = parse_lr_params(br, seq, &quant, allow_intrabc)?;
    tp!(br, "after_lr");
    let tx_mode = parse_tx_mode(br, coded_lossless)?;
    tp!(br, "after_tx_mode");

    let frame_is_intra = matches!(frame_type, FrameType::Key | FrameType::IntraOnly);
    let reference_select = if frame_is_intra { false } else { br.bit()? };
    tp!(br, "after_reference_select");

    // §5.9.21 / §5.9.22 — skip_mode_params. The derivation walks the
    // chosen reference frames' `OrderHint`s (sourced from the DPB's
    // `RefOrderHint[]` array per §5.9.4 / §5.9.16) to find the two
    // refs bracketing the current frame's `OrderHint`. Forward+
    // backward OR a pair of distinct forward refs both unlock
    // `skipModeAllowed`; intra-only / single-reference / order-hint-
    // disabled frames yield `0` and the `skip_mode_present` bit is
    // omitted from the bitstream.
    let (skip_mode_allowed, skip_mode_frame) = derive_skip_mode_allowed(
        frame_is_intra,
        reference_select,
        seq.enable_order_hint,
        seq.order_hint_bits,
        order_hint,
        &ref_frame_idx,
        dpb,
    );
    let skip_mode_present = if skip_mode_allowed { br.bit()? } else { false };
    tp!(br, "after_skip_mode_present");

    // §5.9.25 — allow_warped_motion. Present only when the sequence
    // header set `enable_warped_motion`, the frame is non-intra, and
    // not error-resilient. Comes BEFORE `reduced_tx_set` per spec.
    let allow_warped_motion =
        if !frame_is_intra && !error_resilient_mode && seq.enable_warped_motion {
            br.bit()?
        } else {
            false
        };
    tp!(br, "after_allow_warped_motion");

    let reduced_tx_set = br.bit()?;
    tp!(br, "after_reduced_tx_set");

    let mut gm_type = [GmType::Identity; NUM_REF_FRAMES];
    // §5.9.24: gm_params[ref][i] = (i % 3 == 2) ? 1<<WARPEDMODEL_PREC_BITS : 0
    // for ref = LAST_FRAME..ALTREF_FRAME, before any per-frame coding.
    // The previous all-zero default broke the saved-state chain — when
    // an intra frame's gm_params got copied into SavedGmParams[], the
    // next inter frame loaded `[0,0,0,0,0,0]` and computed a negative
    // `r = -sub` for the alpha components, derailing decode_subexp.
    let identity_alpha: i32 = 1 << 16;
    let mut gm_params = [[0, 0, identity_alpha, 0, 0, identity_alpha]; NUM_REF_FRAMES];
    if !frame_is_intra {
        // §5.9.24 / §5.9.25 — `read_global_param`'s reference value
        // `r = (PrevGmParams[ref][idx] >> precDiff) - sub` requires the
        // `PrevGmParams[]` slot loaded by `load_previous()` whenever
        // `primary_ref_frame != PRIMARY_REF_NONE` (§7.4 / §7.20). Round
        // 18 wires the DPB-saved-GmParams pointer through so non-error-
        // resilient inter frames can re-anchor the subexp decoder on the
        // ref's actual saved warp matrix instead of the identity default.
        // Falls back to identity when the slot has never been refreshed.
        let prev_gm_params = if primary_ref_frame == PRIMARY_REF_NONE {
            None
        } else {
            let slot_idx = ref_frame_idx[primary_ref_frame as usize] as usize;
            dpb.saved_gm_params_for(slot_idx)
        };
        parse_global_motion_params(
            br,
            &mut gm_type,
            &mut gm_params,
            allow_high_precision_mv,
            prev_gm_params.as_ref(),
        )?;
    }

    let film_grain = parse_film_grain_params(br, seq, frame_type, show_frame, showable_frame)?;

    let parse_depth = ParseDepth::Complete;

    Ok(FrameHeader {
        show_existing_frame: false,
        frame_to_show_map_idx: 0,
        display_frame_id: 0,
        frame_type,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update,
        allow_screen_content_tools,
        force_integer_mv,
        current_frame_id,
        frame_size_override_flag,
        order_hint,
        primary_ref_frame,
        refresh_frame_flags,
        ref_order_hint,
        ref_frame_idx,
        frame_width,
        frame_height,
        upscaled_width,
        use_superres,
        superres_denom,
        render_and_frame_size_different,
        render_width,
        render_height,
        allow_intrabc,
        allow_high_precision_mv,
        is_filter_switchable,
        interpolation_filter,
        is_motion_mode_switchable,
        use_ref_frame_mvs,
        disable_frame_end_update_cdf,
        allow_warped_motion,
        reduced_tx_set,
        tile_info: Some(tile_info),
        quant,
        segmentation,
        delta_q_present,
        delta_q_res,
        delta_lf_present,
        delta_lf_res,
        delta_lf_multi,
        loop_filter,
        cdef,
        lr,
        tx_mode,
        reference_select,
        skip_mode_present,
        skip_mode_allowed,
        skip_mode_frame,
        gm_type,
        gm_params,
        film_grain,
        parse_depth,
    })
}

#[allow(clippy::too_many_arguments)]
fn finish_minimal(
    show_existing_frame: bool,
    frame_to_show_map_idx: u8,
    display_frame_id: u32,
    frame_type: FrameType,
    show_frame: bool,
    showable_frame: bool,
    error_resilient_mode: bool,
    parse_depth: ParseDepth,
) -> Result<FrameHeader> {
    Ok(FrameHeader {
        show_existing_frame,
        frame_to_show_map_idx,
        display_frame_id,
        frame_type,
        show_frame,
        showable_frame,
        error_resilient_mode,
        disable_cdf_update: false,
        allow_screen_content_tools: 0,
        force_integer_mv: 0,
        current_frame_id: 0,
        frame_size_override_flag: false,
        order_hint: 0,
        primary_ref_frame: 7,
        refresh_frame_flags: 0,
        ref_order_hint: [0; NUM_REF_FRAMES],
        ref_frame_idx: [0; REFS_PER_FRAME],
        frame_width: 0,
        frame_height: 0,
        upscaled_width: 0,
        use_superres: false,
        superres_denom: SUPERRES_NUM,
        render_and_frame_size_different: false,
        render_width: 0,
        render_height: 0,
        allow_intrabc: false,
        allow_high_precision_mv: false,
        is_filter_switchable: false,
        interpolation_filter: 0,
        is_motion_mode_switchable: false,
        use_ref_frame_mvs: false,
        disable_frame_end_update_cdf: true,
        allow_warped_motion: false,
        reduced_tx_set: false,
        tile_info: None,
        quant: QuantizationParams::default(),
        segmentation: SegmentationParams::default(),
        delta_q_present: false,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_res: 0,
        delta_lf_multi: false,
        loop_filter: LoopFilterParams::default(),
        cdef: CdefParams::default(),
        lr: LoopRestorationParams::default(),
        tx_mode: TxMode::Only4x4,
        reference_select: false,
        skip_mode_present: false,
        skip_mode_allowed: false,
        skip_mode_frame: [0; 2],
        gm_type: [GmType::Identity; NUM_REF_FRAMES],
        // Identity warp matrix per §5.9.24 — alpha diagonals = 1<<16.
        gm_params: [[0, 0, 1 << 16, 0, 0, 1 << 16]; NUM_REF_FRAMES],
        film_grain: FilmGrainParams::default(),
        parse_depth,
    })
}

fn parse_frame_size(
    seq: &SequenceHeader,
    br: &mut BitReader<'_>,
    frame_size_override_flag: bool,
) -> Result<(u32, u32)> {
    if frame_size_override_flag {
        let w = br.f(seq.frame_width_bits as u32)? + 1;
        let h = br.f(seq.frame_height_bits as u32)? + 1;
        Ok((w, h))
    } else {
        Ok((seq.max_frame_width, seq.max_frame_height))
    }
}

/// Returns (frame_width, superres_denom, upscaled_width). When superres is
/// disabled the frame width is unchanged and the denom is 8 (== SUPERRES_NUM).
fn parse_superres(
    seq: &SequenceHeader,
    br: &mut BitReader<'_>,
    upscaled_width: u32,
) -> Result<(u32, u32, u32)> {
    let use_superres = if seq.enable_superres {
        br.bit()?
    } else {
        false
    };
    let denom = if use_superres {
        let coded = br.f(SUPERRES_DENOM_BITS)?;
        coded + SUPERRES_DENOM_MIN
    } else {
        SUPERRES_NUM
    };
    // FrameWidth = (UpscaledWidth * SUPERRES_NUM + (SuperresDenom/2)) / SuperresDenom
    let frame_width = (upscaled_width * SUPERRES_NUM + denom / 2) / denom;
    Ok((frame_width, denom, upscaled_width))
}

fn parse_render_size(
    br: &mut BitReader<'_>,
    frame_width: u32,
    frame_height: u32,
) -> Result<(bool, u32, u32)> {
    let different = br.bit()?;
    if different {
        let rw = br.f(16)? + 1;
        let rh = br.f(16)? + 1;
        Ok((true, rw, rh))
    } else {
        Ok((false, frame_width, frame_height))
    }
}

#[allow(dead_code)]
pub(crate) fn _bit_field_width_for(seq: &SequenceHeader) -> u32 {
    ceil_log2(seq.max_frame_width)
}

/// §7.8 set_frame_refs process — compute `ref_frame_idx[0..REFS_PER_FRAME]`
/// from the explicitly-signalled `last_frame_idx` / `gold_frame_idx` plus
/// the DPB's `RefOrderHint[]` trail. Used when `frame_refs_short_signaling
/// = 1` in the uncompressed header (§5.9.2).
///
/// Reference name → `ref_frame_idx[]` index mapping (spec §3, indices
/// shifted by `LAST_FRAME = 1`):
///   - `LAST_FRAME` → 0
///   - `LAST2_FRAME` → 1
///   - `LAST3_FRAME` → 2
///   - `GOLDEN_FRAME` → 3
///   - `BWDREF_FRAME` → 4
///   - `ALTREF2_FRAME` → 5
///   - `ALTREF_FRAME` → 6
///
/// Algorithm (verbatim §7.8):
///   1. Initialise every entry of `ref_frame_idx[]` to -1, then set
///      LAST=last_frame_idx, GOLDEN=gold_frame_idx.
///   2. Build `shiftedOrderHints[i] = curFrameHint + get_relative_dist(
///      RefOrderHint[i], OrderHint)` with `curFrameHint = 1 <<
///      (OrderHintBits - 1)`.
///   3. ALTREF picks the latest backward (≥ curFrameHint) candidate.
///   4. BWDREF picks the earliest backward.
///   5. ALTREF2 picks the earliest remaining backward (post-BWDREF).
///   6. The forward chain `Ref_Frame_List = {LAST2, LAST3, BWDREF,
///      ALTREF2, ALTREF}` fills its still-unset entries with
///      successively-later forward references (latest first).
///   7. Any remaining unset slots take the global earliest output-
///      order reference (§7.8 last paragraph).
///
/// Slots whose DPB entry is invalid (`!valid`) cannot be used for
/// candidate selection (steps 3-7); the explicit LAST / GOLDEN slots
/// are kept unconditionally per the literal spec text. The function
/// is bit-neutral — it consumes no bitstream bits — and can be called
/// safely for fixtures with empty / partial DPBs (returns all zeros
/// for unfilled entries which matches the prior fallback behaviour).
pub(crate) fn set_frame_refs(
    last_frame_idx: u8,
    gold_frame_idx: u8,
    order_hint: u32,
    order_hint_bits: u32,
    dpb: &Dpb,
) -> [u32; REFS_PER_FRAME] {
    // §3 named indices (shifted by `- LAST_FRAME` to fit ref_frame_idx[]).
    const LAST: usize = 0; // LAST_FRAME - LAST_FRAME
    const GOLDEN: usize = 3; // GOLDEN_FRAME - LAST_FRAME
    const BWDREF: usize = 4; // BWDREF_FRAME - LAST_FRAME
    const ALTREF2: usize = 5; // ALTREF2_FRAME - LAST_FRAME
    const ALTREF: usize = 6; // ALTREF_FRAME - LAST_FRAME

    let mut ref_frame_idx: [i32; REFS_PER_FRAME] = [-1; REFS_PER_FRAME];
    ref_frame_idx[LAST] = last_frame_idx as i32;
    ref_frame_idx[GOLDEN] = gold_frame_idx as i32;

    let mut used_frame = [false; NUM_REF_FRAMES];
    if (last_frame_idx as usize) < NUM_REF_FRAMES {
        used_frame[last_frame_idx as usize] = true;
    }
    if (gold_frame_idx as usize) < NUM_REF_FRAMES {
        used_frame[gold_frame_idx as usize] = true;
    }

    // §7.8 falls back to all-zero ref_frame_idx[] (then explicit LAST /
    // GOLDEN above) when OrderHint signalling is disabled. Without
    // OrderHints we can't run the chain — short-circuit to leave the
    // rest as `0` (the implicit "shared earliest" fallback).
    if order_hint_bits == 0 {
        let mut out = [0u32; REFS_PER_FRAME];
        for (i, v) in out.iter_mut().enumerate() {
            *v = ref_frame_idx[i].max(0) as u32;
        }
        return out;
    }

    let cur_frame_hint: i32 = 1 << (order_hint_bits - 1);
    let mut shifted_order_hints = [0i32; NUM_REF_FRAMES];
    for (i, slot) in dpb.slots.iter().enumerate() {
        if slot.valid {
            shifted_order_hints[i] =
                cur_frame_hint + get_relative_dist(slot.order_hint, order_hint, order_hint_bits);
        } else {
            // Mark as "earlier than any valid frame" so that step 7's
            // global earliest scan won't pick an invalid slot if any
            // valid slot exists. We sentinel using i32::MIN.
            shifted_order_hints[i] = i32::MIN;
        }
    }

    // ALTREF: find_latest_backward — latest hint ≥ curFrameHint.
    let mut latest_order_hint = i32::MIN;
    let mut altref_ref: i32 = -1;
    for (i, &hint) in shifted_order_hints.iter().enumerate() {
        if used_frame[i] || !dpb.slots[i].valid {
            continue;
        }
        if hint >= cur_frame_hint && (altref_ref < 0 || hint >= latest_order_hint) {
            altref_ref = i as i32;
            latest_order_hint = hint;
        }
    }
    if altref_ref >= 0 {
        ref_frame_idx[ALTREF] = altref_ref;
        used_frame[altref_ref as usize] = true;
    }

    // BWDREF: find_earliest_backward — earliest hint ≥ curFrameHint.
    let mut earliest_order_hint = i32::MAX;
    let mut bwdref_ref: i32 = -1;
    for (i, &hint) in shifted_order_hints.iter().enumerate() {
        if used_frame[i] || !dpb.slots[i].valid {
            continue;
        }
        if hint >= cur_frame_hint && (bwdref_ref < 0 || hint < earliest_order_hint) {
            bwdref_ref = i as i32;
            earliest_order_hint = hint;
        }
    }
    if bwdref_ref >= 0 {
        ref_frame_idx[BWDREF] = bwdref_ref;
        used_frame[bwdref_ref as usize] = true;
    }

    // ALTREF2: find_earliest_backward (after BWDREF claim).
    let mut earliest_order_hint = i32::MAX;
    let mut altref2_ref: i32 = -1;
    for (i, &hint) in shifted_order_hints.iter().enumerate() {
        if used_frame[i] || !dpb.slots[i].valid {
            continue;
        }
        if hint >= cur_frame_hint && (altref2_ref < 0 || hint < earliest_order_hint) {
            altref2_ref = i as i32;
            earliest_order_hint = hint;
        }
    }
    if altref2_ref >= 0 {
        ref_frame_idx[ALTREF2] = altref2_ref;
        used_frame[altref2_ref as usize] = true;
    }

    // Forward fill in Ref_Frame_List order: LAST2, LAST3, BWDREF,
    // ALTREF2, ALTREF (indices 1, 2, 4, 5, 6).
    const REF_FRAME_LIST: [usize; 5] = [1, 2, BWDREF, ALTREF2, ALTREF];
    for &slot in REF_FRAME_LIST.iter() {
        if ref_frame_idx[slot] < 0 {
            // find_latest_forward — latest hint < curFrameHint.
            let mut latest_order_hint = i32::MIN;
            let mut chosen: i32 = -1;
            for (i, &hint) in shifted_order_hints.iter().enumerate() {
                if used_frame[i] || !dpb.slots[i].valid {
                    continue;
                }
                if hint < cur_frame_hint && (chosen < 0 || hint >= latest_order_hint) {
                    chosen = i as i32;
                    latest_order_hint = hint;
                }
            }
            if chosen >= 0 {
                ref_frame_idx[slot] = chosen;
                used_frame[chosen as usize] = true;
            }
        }
    }

    // Final fallback — use the global earliest valid output-order slot
    // for any still-unset entry (§7.8 last paragraph). When NO slots
    // are valid we leave the unset entries at 0 (matches the prior
    // fallback behaviour for empty-DPB / first-inter-frame edge cases).
    let mut earliest_order_hint = i32::MAX;
    let mut earliest: i32 = -1;
    for (i, &hint) in shifted_order_hints.iter().enumerate() {
        if !dpb.slots[i].valid {
            continue;
        }
        if earliest < 0 || hint < earliest_order_hint {
            earliest = i as i32;
            earliest_order_hint = hint;
        }
    }
    let mut out = [0u32; REFS_PER_FRAME];
    for (i, v) in out.iter_mut().enumerate() {
        if ref_frame_idx[i] >= 0 {
            *v = ref_frame_idx[i] as u32;
        } else if earliest >= 0 {
            *v = earliest as u32;
        } else {
            *v = 0;
        }
    }
    out
}

/// §5.9.21 / §7.9.2 — derive `(skipModeAllowed, SkipModeFrame[0..=1])`.
///
/// Walks `ref_frame_idx[0..REFS_PER_FRAME]` looking up the per-slot
/// `RefOrderHint[]` from the supplied [`Dpb`]. The two outputs match
/// the spec: a forward+backward bracketing pair (smallest gap on
/// each side of the current `OrderHint`) sets
/// `SkipModeFrame[0..=1] = (Min, Max)` of the bracketing indices;
/// when no backward reference exists, the spec falls back to a pair
/// of forward references — `(forwardIdx, secondForwardIdx)` — chosen
/// by largest gap between them. Slots whose DPB entry is invalid are
/// treated as if `RefOrderHint == 0` AND skipped from the candidate
/// pool, since an invalid slot cannot be used for compound prediction.
///
/// Returned `SkipModeFrame[0..=1]` values are stored as `LAST_FRAME +
/// i` per spec (logical reference indices `1..=7`); both entries
/// default to `0` when the derivation rejects (`skipModeAllowed = 0`).
pub(crate) fn derive_skip_mode_allowed(
    frame_is_intra: bool,
    reference_select: bool,
    enable_order_hint: bool,
    order_hint_bits: u32,
    order_hint: u32,
    ref_frame_idx: &[u32; REFS_PER_FRAME],
    dpb: &Dpb,
) -> (bool, [u8; 2]) {
    // Spec gating: the entire derivation collapses to 0 for
    // intra-only frames, single-reference inter frames, or sequences
    // without OrderHint signalling.
    if frame_is_intra || !reference_select || !enable_order_hint {
        return (false, [0, 0]);
    }

    let mut forward_idx: i32 = -1;
    let mut forward_hint: u32 = 0;
    let mut backward_idx: i32 = -1;
    let mut backward_hint: u32 = 0;

    for (i, &slot_idx) in ref_frame_idx.iter().enumerate().take(REFS_PER_FRAME) {
        let slot = slot_idx as usize;
        let dpb_slot = dpb.slots.get(slot).cloned().unwrap_or_default();
        if !dpb_slot.valid {
            // Without a valid DPB entry we don't know this slot's
            // OrderHint — exclude it from skip-mode bracketing.
            continue;
        }
        let ref_hint = dpb_slot.order_hint;
        let cmp_to_cur = get_relative_dist(ref_hint, order_hint, order_hint_bits);
        if cmp_to_cur < 0 {
            // Forward (past) reference. Track the largest forward
            // hint (= closest to the current frame from the past).
            if forward_idx < 0 || get_relative_dist(ref_hint, forward_hint, order_hint_bits) > 0 {
                forward_idx = i as i32;
                forward_hint = ref_hint;
            }
        } else if cmp_to_cur > 0 {
            // Backward (future) reference. Track the smallest backward
            // hint (= closest to the current frame from the future).
            if backward_idx < 0 || get_relative_dist(ref_hint, backward_hint, order_hint_bits) < 0 {
                backward_idx = i as i32;
                backward_hint = ref_hint;
            }
        }
    }

    if forward_idx < 0 {
        // No forward reference — skip mode requires at least one.
        return (false, [0, 0]);
    }

    if backward_idx >= 0 {
        // Forward + backward case: SkipModeFrame[0..=1] are the
        // (Min, Max) of the two bracketing indices, stored as
        // `LAST_FRAME + i`.
        let lo = forward_idx.min(backward_idx) as u8;
        let hi = forward_idx.max(backward_idx) as u8;
        return (true, [LAST_FRAME + lo, LAST_FRAME + hi]);
    }

    // Forward-only case: search for a second forward reference whose
    // OrderHint sits earlier than `forwardHint` (chosen by largest
    // backward gap from `forwardHint`).
    let mut second_forward_idx: i32 = -1;
    let mut second_forward_hint: u32 = 0;
    for (i, &slot_idx) in ref_frame_idx.iter().enumerate().take(REFS_PER_FRAME) {
        let slot = slot_idx as usize;
        let dpb_slot = dpb.slots.get(slot).cloned().unwrap_or_default();
        if !dpb_slot.valid {
            continue;
        }
        let ref_hint = dpb_slot.order_hint;
        if get_relative_dist(ref_hint, forward_hint, order_hint_bits) < 0
            && (second_forward_idx < 0
                || get_relative_dist(ref_hint, second_forward_hint, order_hint_bits) > 0)
        {
            second_forward_idx = i as i32;
            second_forward_hint = ref_hint;
        }
    }
    if second_forward_idx < 0 {
        return (false, [0, 0]);
    }
    let lo = forward_idx.min(second_forward_idx) as u8;
    let hi = forward_idx.max(second_forward_idx) as u8;
    (true, [LAST_FRAME + lo, LAST_FRAME + hi])
}

#[cfg(test)]
mod skip_mode_tests {
    use super::*;

    #[test]
    fn intra_or_no_reference_select_returns_zero() {
        let dpb = Dpb::new();
        let refs = [0u32; REFS_PER_FRAME];
        // Intra frame -> always disallowed.
        let (allowed, frames) = derive_skip_mode_allowed(true, true, true, 8, 4, &refs, &dpb);
        assert!(!allowed);
        assert_eq!(frames, [0, 0]);
        // No reference_select (single-ref frame) -> disallowed.
        let (allowed, _) = derive_skip_mode_allowed(false, false, true, 8, 4, &refs, &dpb);
        assert!(!allowed);
        // OrderHint disabled at the sequence level -> disallowed.
        let (allowed, _) = derive_skip_mode_allowed(false, true, false, 8, 4, &refs, &dpb);
        assert!(!allowed);
    }

    #[test]
    fn empty_dpb_is_disallowed() {
        let dpb = Dpb::new();
        let refs = [0u32; REFS_PER_FRAME];
        let (allowed, _) = derive_skip_mode_allowed(false, true, true, 8, 4, &refs, &dpb);
        assert!(!allowed);
    }

    #[test]
    fn forward_plus_backward_picks_min_max_indices() {
        // OrderHint window = 8 bits. Current frame OrderHint = 4.
        // Slots: 0 -> hint=2 (past), 1 -> hint=6 (future).
        let mut dpb = Dpb::new();
        dpb.slots[0] = crate::dpb::RefSlot {
            order_hint: 2,
            valid: true,
            frame: None,
            ..Default::default()
        };
        dpb.slots[1] = crate::dpb::RefSlot {
            order_hint: 6,
            valid: true,
            frame: None,
            ..Default::default()
        };
        // ref_frame_idx[0] -> slot 0 (past), ref_frame_idx[1] -> slot 1
        // (future). Other refs reuse slot 0.
        let mut refs = [0u32; REFS_PER_FRAME];
        refs[1] = 1;
        let (allowed, frames) = derive_skip_mode_allowed(false, true, true, 8, 4, &refs, &dpb);
        assert!(allowed);
        // Forward index = 0 (past), Backward index = 1 (future).
        // SkipModeFrame[0] = LAST_FRAME + 0 = 1, SkipModeFrame[1] =
        // LAST_FRAME + 1 = 2.
        assert_eq!(frames, [LAST_FRAME, LAST_FRAME + 1]);
    }

    #[test]
    fn forward_only_two_distinct_picks_widest_pair() {
        // Both refs in the past — forward-only branch fires.
        let mut dpb = Dpb::new();
        dpb.slots[0] = crate::dpb::RefSlot {
            order_hint: 2,
            valid: true,
            frame: None,
            ..Default::default()
        };
        dpb.slots[1] = crate::dpb::RefSlot {
            order_hint: 1,
            valid: true,
            frame: None,
            ..Default::default()
        };
        let mut refs = [0u32; REFS_PER_FRAME];
        refs[0] = 0;
        refs[1] = 1;
        // Other ref slots invalid (default).
        let (allowed, frames) = derive_skip_mode_allowed(false, true, true, 8, 4, &refs, &dpb);
        assert!(allowed);
        // forwardIdx tracks closest-to-current = i=0 (hint=2);
        // secondForwardIdx tracks earlier = i=1 (hint=1). Result is
        // (Min, Max) = (0, 1).
        assert_eq!(frames, [LAST_FRAME, LAST_FRAME + 1]);
    }

    #[test]
    fn forward_only_single_ref_is_disallowed() {
        // One past reference repeated across all slots — no second
        // distinct forward ref -> disallowed.
        let mut dpb = Dpb::new();
        dpb.slots[0] = crate::dpb::RefSlot {
            order_hint: 2,
            valid: true,
            frame: None,
            ..Default::default()
        };
        let refs = [0u32; REFS_PER_FRAME]; // all -> slot 0
        let (allowed, frames) = derive_skip_mode_allowed(false, true, true, 8, 4, &refs, &dpb);
        assert!(!allowed);
        assert_eq!(frames, [0, 0]);
    }
}

#[cfg(test)]
mod set_frame_refs_tests {
    use super::*;
    use crate::dpb::RefSlot;

    /// §7.8 — Empty DPB: no slot is valid, so both LAST and GOLDEN are
    /// the explicitly-signalled indices and the rest fall back to 0
    /// (the implicit "earliest valid" path is skipped). This matches
    /// the prior all-zero ref_frame_idx[] fallback for first-inter-
    /// frame edge cases.
    #[test]
    fn empty_dpb_keeps_explicit_last_and_golden_only() {
        let dpb = Dpb::new();
        let out = set_frame_refs(2, 5, 4, 7, &dpb);
        assert_eq!(out[0], 2); // LAST_FRAME = last_frame_idx
        assert_eq!(out[3], 5); // GOLDEN_FRAME = gold_frame_idx
                               // ALTREF / BWDREF / ALTREF2 / LAST2 / LAST3 left at the
                               // earliest fallback (= 0 when no valid slots).
        for &v in out.iter() {
            assert!(v < NUM_REF_FRAMES as u32);
        }
    }

    /// §7.8 — Forward+backward DPB: cur=4 (OrderHintBits=3, range
    /// 0..7). Slot 0 = past (hint=2), slot 1 = future (hint=6), slot 2
    /// = future (hint=5), slot 3 = past (hint=1). last_frame_idx=0,
    /// gold_frame_idx=3. ALTREF picks the latest backward (slot 1,
    /// shifted hint = 4 + (6-4) = 6); BWDREF picks the earliest
    /// backward of remaining (slot 2 with shifted hint=5); ALTREF2 has
    /// no remaining backward.
    #[test]
    fn picks_latest_backward_for_altref_and_earliest_for_bwdref() {
        let mut dpb = Dpb::new();
        dpb.slots[0] = RefSlot {
            order_hint: 2,
            valid: true,
            frame: None,
            ..Default::default()
        };
        dpb.slots[1] = RefSlot {
            order_hint: 6,
            valid: true,
            frame: None,
            ..Default::default()
        };
        dpb.slots[2] = RefSlot {
            order_hint: 5,
            valid: true,
            frame: None,
            ..Default::default()
        };
        dpb.slots[3] = RefSlot {
            order_hint: 1,
            valid: true,
            frame: None,
            ..Default::default()
        };
        // last=0 (past), gold=3 (past). cur=4, OrderHintBits=3 (range 0..7).
        let out = set_frame_refs(0, 3, 4, 3, &dpb);
        assert_eq!(out[0], 0); // LAST_FRAME
        assert_eq!(out[3], 3); // GOLDEN_FRAME
                               // ALTREF (idx 6 in ref_frame_idx[]) = slot 1 (hint=6, latest
                               // backward).
        assert_eq!(out[6], 1);
        // BWDREF (idx 4) = slot 2 (hint=5, earliest backward after
        // ALTREF claimed slot 1).
        assert_eq!(out[4], 2);
    }

    /// §7.8 — All-forward DPB (no backward refs): ALTREF / BWDREF /
    /// ALTREF2 are NOT set in steps 3-5 (their "≥ curFrameHint" gate
    /// fails). Step 6 fills them via Ref_Frame_List using
    /// find_latest_forward.
    #[test]
    fn all_forward_dpb_falls_through_to_forward_chain() {
        let mut dpb = Dpb::new();
        // All slots are past references (hint < cur=4).
        for (i, oh) in [0, 1, 2, 3].iter().enumerate() {
            dpb.slots[i] = RefSlot {
                order_hint: *oh,
                valid: true,
                frame: None,
                ..Default::default()
            };
        }
        let out = set_frame_refs(0, 3, 4, 3, &dpb);
        assert_eq!(out[0], 0); // LAST
        assert_eq!(out[3], 3); // GOLDEN
                               // ALTREF (idx 6), BWDREF (idx 4), ALTREF2 (idx 5) and LAST2
                               // (idx 1) / LAST3 (idx 2) all filled by find_latest_forward.
                               // Each should point to a valid slot in 0..=3 (no out-of-range).
        for &v in out.iter() {
            assert!((v as usize) < 4);
        }
        // Ref_Frame_List order is [LAST2, LAST3, BWDREF, ALTREF2,
        // ALTREF]. The first entry (LAST2) gets the LATEST remaining
        // forward (slot 2 with hint=2, since slots 0 and 3 were claimed
        // for LAST/GOLDEN).
        assert_eq!(out[1], 2);
        assert_eq!(out[2], 1);
    }

    /// §7.8 — `order_hint_bits == 0` short-circuit path. Without
    /// OrderHints the chain search is meaningless; we keep the
    /// explicit LAST/GOLDEN and leave the rest at 0.
    #[test]
    fn order_hint_disabled_short_circuits() {
        let mut dpb = Dpb::new();
        dpb.slots[3] = RefSlot {
            order_hint: 0,
            valid: true,
            frame: None,
            ..Default::default()
        };
        let out = set_frame_refs(2, 5, 0, 0, &dpb);
        assert_eq!(out[0], 2);
        assert_eq!(out[3], 5);
        // Other slots remain 0 (the spec leaves ref_frame_idx[] at -1
        // initially, and our impl resolves -1 → max(0) = 0).
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert_eq!(out[4], 0);
        assert_eq!(out[5], 0);
        assert_eq!(out[6], 0);
    }

    /// Bit-neutrality: set_frame_refs is a pure function on (idx_bits,
    /// OrderHints). It must NOT touch the bit reader. We can't test
    /// that directly without a reader, but we verify that calling it
    /// twice with the same inputs returns identical outputs.
    #[test]
    fn deterministic_for_identical_inputs() {
        let mut dpb = Dpb::new();
        dpb.slots[0] = RefSlot {
            order_hint: 1,
            valid: true,
            frame: None,
            ..Default::default()
        };
        dpb.slots[5] = RefSlot {
            order_hint: 7,
            valid: true,
            frame: None,
            ..Default::default()
        };
        let a = set_frame_refs(0, 5, 4, 3, &dpb);
        let b = set_frame_refs(0, 5, 4, 3, &dpb);
        assert_eq!(a, b);
    }

    /// `dpb.saved_gm_params_for(ref_frame_idx[primary_ref_frame])`
    /// regression: with set_frame_refs the LAST_FRAME slot resolves
    /// to `last_frame_idx` (not 0), so a non-zero `last_frame_idx`
    /// supplied via short-signaling now points to the correct DPB slot
    /// and `saved_gm_params_for` returns the warp matrix saved by that
    /// reference frame.
    #[test]
    fn primary_ref_zero_picks_last_frame_idx_slot() {
        let mut dpb = Dpb::new();
        // Slot 5 is the "real" LAST_FRAME for this inter frame. Save
        // a non-identity gm_params there so the test can verify the
        // lookup is plumbed correctly.
        let mut gm = [[0i32; 6]; NUM_REF_FRAMES];
        gm[1] = [123, 456, 1 << 16, 0, 0, 1 << 16];
        dpb.refresh_with_gm(1 << 5, 4, None, &gm);
        let out = set_frame_refs(5, 0, 4, 3, &dpb);
        assert_eq!(out[0], 5); // LAST_FRAME -> slot 5
        let saved = dpb.saved_gm_params_for(out[0] as usize).expect("slot 5");
        assert_eq!(saved[1], [123, 456, 1 << 16, 0, 0, 1 << 16]);
    }

    /// Round 21 regression — pin §5.9.2 inter-branch order: frame_size
    /// and render_size MUST be parsed AFTER the `ref_frame_idx[]` loop,
    /// not before. Previously we read render_diff at the position of
    /// `frame_refs_short_signaling`, which silently mis-aligned the
    /// bitstream by ~13 bits on every non-short-signaling inter frame.
    /// The test crafts a minimal SVT-AV1-style inter Frame OBU payload
    /// with `frame_refs_short_signaling=0` (forcing the spec's 21-bit
    /// per-slot ref_frame_idx loop to be exercised) and checks that the
    /// parser arrives at the disable_frame_end_update_cdf checkpoint at
    /// the bit position the spec dictates. Built with a minimal
    /// quantization, segmentation, loop_filter, cdef, lr, tx_mode, and
    /// global_motion tail so the parser can complete without hitting
    /// `out of bits`.
    #[test]
    fn inter_branch_reads_frame_size_after_ref_frame_idx_loop() {
        use crate::sequence_header::{ColorConfig, SequenceHeader};
        // Synthesise a minimal sequence header with the SVT-AV1
        // chain-walk fixture's settings: 128x128, 4:2:0, no superres,
        // order-hint enabled, frame_id_numbers absent, no warped /
        // dual filter / film_grain.
        let seq = SequenceHeader {
            seq_profile: 0,
            still_picture: false,
            reduced_still_picture_header: false,
            timing_info_present: false,
            timing_info: None,
            decoder_model_info_present: false,
            decoder_model_info: None,
            initial_display_delay_present: false,
            operating_points_cnt: 1,
            operating_points: Vec::new(),
            frame_width_bits: 7,
            frame_height_bits: 7,
            max_frame_width_minus_1: 127,
            max_frame_height_minus_1: 127,
            max_frame_width: 128,
            max_frame_height: 128,
            frame_id_numbers_present: false,
            delta_frame_id_length_minus_2: 0,
            additional_frame_id_length_minus_1: 0,
            use_128x128_superblock: false,
            enable_filter_intra: false,
            enable_intra_edge_filter: false,
            enable_interintra_compound: false,
            enable_masked_compound: false,
            enable_warped_motion: false,
            enable_dual_filter: false,
            enable_order_hint: true,
            enable_jnt_comp: false,
            enable_ref_frame_mvs: false,
            seq_force_screen_content_tools: 0,
            seq_force_integer_mv: 0,
            order_hint_bits: 7,
            enable_superres: false,
            enable_cdef: true,
            enable_restoration: true,
            color_config: ColorConfig {
                high_bitdepth: false,
                twelve_bit: false,
                bit_depth: 8,
                num_planes: 3,
                mono_chrome: false,
                subsampling_x: true,
                subsampling_y: true,
                color_range: false,
                color_primaries: 0,
                transfer_characteristics: 0,
                matrix_coefficients: 0,
                separate_uv_deltas: false,
                chroma_sample_position: 0,
            },
            film_grain_params_present: false,
        };

        // Hand-build the bit stream using a small writer. We only
        // care about reaching `before_frame_refs_block` and observing
        // the bit position at which `frame_refs_short_signaling` is
        // consumed.
        let mut bits: Vec<u8> = Vec::new();
        let mut acc: u32 = 0;
        let mut nbits: u32 = 0;
        let put = |bv: u32, n: u32, bits: &mut Vec<u8>, acc: &mut u32, nbits: &mut u32| {
            *acc = (*acc << n) | bv;
            *nbits += n;
            while *nbits >= 8 {
                let shift = *nbits - 8;
                bits.push((*acc >> shift) as u8);
                *acc &= (1u32 << shift) - 1;
                *nbits = shift;
            }
        };
        // show_existing=0, frame_type=INTER(1), show_frame=1,
        // error_resilient=0, disable_cdf_update=0,
        // (allow_screen_content_tools skipped: seq_force=0, takes that
        // value), frame_size_override=0, order_hint=7'0,
        // primary_ref_frame=0, refresh_frame_flags=0,
        // frame_refs_short_signaling=0, 7×ref_frame_idx=0,
        // (frame_size+render_size: render_diff=0 = 1 bit),
        // allow_high_precision_mv=0, is_filter_switchable=1,
        // is_motion_mode_switchable=0,
        // (use_ref_frame_mvs: enable_ref_frame_mvs=false → 0 bits),
        // disable_frame_end_update_cdf=1.
        put(0, 1, &mut bits, &mut acc, &mut nbits); // show_existing
        put(1, 2, &mut bits, &mut acc, &mut nbits); // INTER
        put(1, 1, &mut bits, &mut acc, &mut nbits); // show_frame
        put(0, 1, &mut bits, &mut acc, &mut nbits); // error_resilient
        put(0, 1, &mut bits, &mut acc, &mut nbits); // disable_cdf_update
        put(0, 1, &mut bits, &mut acc, &mut nbits); // frame_size_override
        put(0, 7, &mut bits, &mut acc, &mut nbits); // order_hint
        put(0, 3, &mut bits, &mut acc, &mut nbits); // primary_ref_frame
        put(0, 8, &mut bits, &mut acc, &mut nbits); // refresh_frame_flags
                                                    // ---- inter branch (per spec ORDER) ----
        put(0, 1, &mut bits, &mut acc, &mut nbits); // frame_refs_short_signaling=0
        for _ in 0..REFS_PER_FRAME {
            put(0, 3, &mut bits, &mut acc, &mut nbits); // ref_frame_idx[i]=0
        }
        put(0, 1, &mut bits, &mut acc, &mut nbits); // render_diff=0
        put(0, 1, &mut bits, &mut acc, &mut nbits); // high_precision_mv=0
        put(1, 1, &mut bits, &mut acc, &mut nbits); // is_filter_switchable=1
        put(0, 1, &mut bits, &mut acc, &mut nbits); // is_motion_mode_switchable=0
                                                    // use_ref_frame_mvs gated off (enable_ref_frame_mvs=false)
        put(1, 1, &mut bits, &mut acc, &mut nbits); // disable_frame_end_update_cdf=1
                                                    // tile_info: uniform=1, no col/row increments (max==min),
                                                    // no context_update_tile_id (cols_log2==rows_log2==0).
        put(1, 1, &mut bits, &mut acc, &mut nbits); // uniform=1
                                                    // sb_cols=2, max_log2_tile_cols=1, min=0 → loop reads 1 bit (0=stop).
        put(0, 1, &mut bits, &mut acc, &mut nbits); // cols_increment=0
                                                    // min_log2_tile_rows = max(0-0, 0)=0; max=1 → loop reads 1 bit.
        put(0, 1, &mut bits, &mut acc, &mut nbits); // rows_increment=0
                                                    // ---- post-tile_info tail, all minimal ----
                                                    // quant: base_q_idx=128 (=non-zero so coded_lossless_hint=false),
                                                    // Y_DC=absent, U_DC=absent, U_AC=absent, using_qmatrix=0.
        put(128, 8, &mut bits, &mut acc, &mut nbits); // base_q_idx
        put(0, 1, &mut bits, &mut acc, &mut nbits); // Y_DC absent
        put(0, 1, &mut bits, &mut acc, &mut nbits); // U_DC absent
        put(0, 1, &mut bits, &mut acc, &mut nbits); // U_AC absent
        put(0, 1, &mut bits, &mut acc, &mut nbits); // using_qmatrix=0
                                                    // segmentation: enabled=0
        put(0, 1, &mut bits, &mut acc, &mut nbits);
        // delta_q_params: present=0
        put(0, 1, &mut bits, &mut acc, &mut nbits);
        // delta_lf_params: gated off (delta_q_present=0)
        // loop_filter: levels=0/0 → no UV; sharpness=0; delta_enabled=0
        put(0, 6, &mut bits, &mut acc, &mut nbits); // level_y0
        put(0, 6, &mut bits, &mut acc, &mut nbits); // level_y1
        put(0, 3, &mut bits, &mut acc, &mut nbits); // sharpness
        put(0, 1, &mut bits, &mut acc, &mut nbits); // delta_enabled
                                                    // cdef: damping=0, bits=0 → 1 iter (4 + 2 + 4 + 2 = 12)
        put(0, 2, &mut bits, &mut acc, &mut nbits); // damping
        put(0, 2, &mut bits, &mut acc, &mut nbits); // cdef_bits
        put(0, 4, &mut bits, &mut acc, &mut nbits); // y_pri
        put(0, 2, &mut bits, &mut acc, &mut nbits); // y_sec
        put(0, 4, &mut bits, &mut acc, &mut nbits); // uv_pri
        put(0, 2, &mut bits, &mut acc, &mut nbits); // uv_sec
                                                    // lr: 3 lr_types all 0 (RESTORE_NONE) → uses_lr=false, no extra
        put(0, 2, &mut bits, &mut acc, &mut nbits);
        put(0, 2, &mut bits, &mut acc, &mut nbits);
        put(0, 2, &mut bits, &mut acc, &mut nbits);
        // tx_mode_select=0 → LARGEST
        put(0, 1, &mut bits, &mut acc, &mut nbits);
        // reference_select=0
        put(0, 1, &mut bits, &mut acc, &mut nbits);
        // skip_mode_present gated off (reference_select=0)
        // allow_warped_motion gated off (enable_warped_motion=false)
        // reduced_tx_set
        put(0, 1, &mut bits, &mut acc, &mut nbits);
        // global_motion: 7 ref slots × 1 bit (is_global=0 → IDENTITY)
        for _ in 0..7 {
            put(0, 1, &mut bits, &mut acc, &mut nbits);
        }
        // film_grain gated off (film_grain_params_present=false)
        // pad to byte
        if nbits > 0 {
            put(0, 8 - nbits, &mut bits, &mut acc, &mut nbits);
        }

        let dpb = Dpb::new();
        let fh = parse_frame_header_with_dpb(&seq, &bits, &dpb).expect("parse");
        assert_eq!(fh.frame_type, FrameType::Inter);
        assert!(!fh.error_resilient_mode);
        // tile_info reached implies inter-branch + frame_size + render
        // bit accounting all matched spec — any mis-ordering would have
        // mis-interpreted some bit downstream and surfaced as either a
        // wrong tile_cols or "out of bits".
        let ti = fh.tile_info.as_ref().expect("tile_info parsed");
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        // Sanity: no high_precision_mv was set (we encoded 0).
        assert!(!fh.allow_high_precision_mv);
        assert!(fh.is_filter_switchable);
        // Frame_size came from sequence-level max dims (no override).
        assert_eq!(fh.frame_width, 128);
        assert_eq!(fh.frame_height, 128);
    }
}
