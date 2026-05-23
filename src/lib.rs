//! # oxideav-av1
//!
//! **Status:** orphan-rebuild scaffold (post 2026-05-20 audit), clean
//! room rebuild in progress.
//!
//! The decoder/encoder pipeline is not wired up yet. Bitstream
//! parsing has reached:
//!
//!   * **Round 1.** OBU bytestream walker described in §5.3 of the
//!     AV1 Bitstream & Decoding Process Specification — boundaries
//!     in a low-overhead bitstream plus `obu_type` /
//!     `obu_extension_flag` / `obu_has_size_field` / `temporal_id` /
//!     `spatial_id` / `obu_size` fields and a payload slice for each
//!     unit. See [`obu`].
//!
//!   * **Round 2.** Sequence header OBU parse per §5.5
//!     (`sequence_header_obu`, `color_config`, `timing_info`,
//!     `decoder_model_info`, `operating_parameters_info`). Returns a
//!     strongly typed [`sequence_header::SequenceHeader`] descriptor
//!     plus a bit-position so the trailing-bits accounting from
//!     §5.3.1 can plug in cleanly next round. See [`sequence_header`].
//!
//!   * **Round 3.** Leading slice of `uncompressed_header()` per
//!     §5.9.2 — `show_existing_frame` / `frame_to_show_map_idx` /
//!     `display_frame_id` / `frame_type` / `show_frame` /
//!     `showable_frame` / `error_resilient_mode` /
//!     `disable_cdf_update` / `allow_screen_content_tools` /
//!     `force_integer_mv` / `current_frame_id` /
//!     `frame_size_override_flag` / `order_hint` /
//!     `primary_ref_frame` / `refresh_frame_flags`. Composes with
//!     round-2's `SequenceHeader` to drive every conditional read.
//!     See [`frame_header`].
//!
//!   * **Round 4.** Frame-size sub-syntax block per §5.9.5
//!     (`frame_size`) + §5.9.6 (`render_size`) + §5.9.8
//!     (`superres_params`) + §5.9.9 (`compute_image_size`). For
//!     every intra (`KEY_FRAME` / `INTRA_ONLY_FRAME`) frame in the
//!     §5.9.2 syntax tree, [`parse_frame_header`] now drops into
//!     `frame_size()` + `render_size()` after `refresh_frame_flags`
//!     and surfaces the eight-field [`FrameSize`] descriptor
//!     (`frame_width` / `frame_height` / `render_width` /
//!     `render_height` / `superres_denom` / `upscaled_width` /
//!     `mi_cols` / `mi_rows`). The §5.9.7 `frame_size_with_refs()`
//!     `found_ref` shortcut is **not** implemented yet — it reads
//!     `RefUpscaledWidth[]` / `RefFrameHeight[]` /
//!     `RefRenderWidth[]` / `RefRenderHeight[]` from a
//!     reference-frame state table this round does not track —
//!     so inter-frame parsing still stops at `refresh_frame_flags`
//!     with `frame_size = None`. See [`frame_header`].
//!
//!   * **Round 6.** `allow_intrabc` (§5.9.3 path of §5.9.2) +
//!     `tile_info()` (§5.9.15) wired into the streaming
//!     [`parse_frame_header`] walk. For intra frames whose
//!     `allow_screen_content_tools && UpscaledWidth == FrameWidth`
//!     conjunction holds, the parser now consumes the `allow_intrabc`
//!     `f(1)` slot — otherwise the spec's `allow_intrabc = 0`
//!     initialiser stands. After the `frame_size()` / `render_size()`
//!     block (intra path), the parser then walks `tile_info()` per
//!     §5.9.15 and surfaces a typed [`tile_info::TileInfo`]
//!     (`uniform_tile_spacing_flag`, `tile_cols`, `tile_rows`,
//!     `tile_cols_log2`, `tile_rows_log2`, `context_update_tile_id`,
//!     `tile_size_bytes`, `mi_col_starts`, `mi_row_starts`). The
//!     non-uniform-spacing path consumes the `ns(maxWidth)` /
//!     `ns(maxHeight)` `width_in_sbs_minus_1` / `height_in_sbs_minus_1`
//!     fields via the new [`bitreader::BitReader::ns`] primitive
//!     (§4.10.7). Tile-content decode (motion vectors, transform /
//!     quantisation, in-loop filters, film grain) is still out of
//!     scope. See [`tile_info`].
//!
//!   * **Round 5.** Uncompressed-header tail sub-syntaxes — §5.9.10
//!     `read_interpolation_filter()` (returns
//!     [`InterpolationFilter`]), §5.9.11 `loop_filter_params()`
//!     (returns [`LoopFilterParams`] with the `CodedLossless ||
//!     allow_intrabc` short-circuit, the four `loop_filter_level[]`
//!     fields with the `NumPlanes > 1 && (level[0] || level[1])`
//!     gate on the chroma slots, the `f(3)` `loop_filter_sharpness`,
//!     and the `loop_filter_delta_enabled / delta_update /
//!     update_ref_delta[i] / update_mode_delta[i]` per-slot
//!     update walk over `TOTAL_REFS_PER_FRAME = 8` ref-deltas + 2
//!     mode-deltas with `su(7)` signed offsets), and §5.9.12
//!     `quantization_params()` + §5.9.13 `read_delta_q()` (returns
//!     [`QuantizationParams`] with `base_q_idx`, the four
//!     `delta_q_y_dc / delta_q_u_dc / delta_q_u_ac / delta_q_v_dc /
//!     delta_q_v_ac` per-plane offsets, `diff_uv_delta` /
//!     `using_qmatrix` / `qm_y` / `qm_u` / `qm_v`). These remain
//!     available as standalone parser entry points
//!     ([`parse_interpolation_filter`], [`parse_loop_filter_params`],
//!     [`parse_quantization_params`]) for callers that want to
//!     exercise the parsers on a raw byte slice. See
//!     [`uncompressed_header_tail`].
//!
//!   * **Round 7.** §5.9.12 `quantization_params()` and §5.9.14
//!     `segmentation_params()` wired into the streaming
//!     [`parse_frame_header`] walk (intra path). After `tile_info()`
//!     the parser now consumes `quantization_params()` and surfaces a
//!     typed [`QuantizationParams`] on [`FrameHeader::quantization_params`],
//!     then `segmentation_params()` and surfaces a typed
//!     [`SegmentationParams`] on [`FrameHeader::segmentation_params`]
//!     covering `segmentation_enabled`, `segmentation_update_map`,
//!     `segmentation_temporal_update`, `segmentation_update_data`,
//!     the full §5.9.14 `FeatureEnabled[i][j]` /
//!     `FeatureData[i][j]` 8×8 table (`segment_feature_active` /
//!     `segment_feature_data`), and the §5.9.14 trailing
//!     `SegIdPreSkip` / `LastActiveSegId` derivations. The §5.9.14
//!     `primary_ref_frame == PRIMARY_REF_NONE` collapse is honoured
//!     (`update_map = 1`, `temporal_update = 0`, `update_data = 1`,
//!     no bitstream reads for the three update flags). Per-feature
//!     `Segmentation_Feature_Bits` / `Segmentation_Feature_Signed` /
//!     `Segmentation_Feature_Max` Table 5.9.14 tables are exposed as
//!     [`SEGMENTATION_FEATURE_BITS`] / [`SEGMENTATION_FEATURE_SIGNED`]
//!     / [`SEGMENTATION_FEATURE_MAX`]. See
//!     [`uncompressed_header_tail::parse_segmentation_params`].
//!
//!   * **Round 8.** §5.9.17 `delta_q_params()` and §5.9.18
//!     `delta_lf_params()` wired into the streaming
//!     [`parse_frame_header`] walk (intra path). After
//!     `segmentation_params()` the parser consumes `delta_q_params()`
//!     and surfaces a typed [`DeltaQParams`] on
//!     [`FrameHeader::delta_q_params`] (`delta_q_present` — read as
//!     `f(1)` only when `base_q_idx > 0`; `delta_q_res` — `f(2)`,
//!     read only when `delta_q_present == 1`), then `delta_lf_params()`
//!     and surfaces a typed [`DeltaLfParams`] on
//!     [`FrameHeader::delta_lf_params`] (`delta_lf_present` — gated on
//!     `delta_q_present` and suppressed when `allow_intrabc == 1`;
//!     `delta_lf_res` / `delta_lf_multi` — read only when
//!     `delta_lf_present == 1`). Both remain available as standalone
//!     parser entry points
//!     ([`uncompressed_header_tail::parse_delta_q_params`] /
//!     [`uncompressed_header_tail::parse_delta_lf_params`]).
//!
//! Frame decoding past `delta_lf_params()` (the remaining tail —
//! `loop_filter_params()` streaming wire-in / `cdef_params()` /
//! `lr_params()` / `read_tx_mode()` / `frame_reference_mode()` /
//! tile-content decode) is still out of scope. [`decode_av1`] /
//! [`encode_av1`] continue to return [`Error::NotImplemented`].

#![warn(missing_debug_implementations)]

use oxideav_core::RuntimeContext;

mod bitreader;
pub mod frame_header;
pub mod obu;
pub mod sequence_header;
pub mod tile_info;
pub mod uncompressed_header_tail;

pub use frame_header::{
    parse_frame_header, FrameHeader, FrameSize, FrameType, NUM_REF_FRAMES, PRIMARY_REF_NONE,
    SUPERRES_DENOM_BITS, SUPERRES_DENOM_MIN, SUPERRES_NUM,
};
pub use obu::{parse_leb128, parse_obu, ObuDescriptor, ObuIter, ObuType};
pub use sequence_header::{
    parse_sequence_header, ColorConfig, DecoderModelInfo, OperatingParametersInfo, OperatingPoint,
    SequenceHeader, TimingInfo,
};
pub use tile_info::{
    parse_tile_info, TileInfo, MAX_TILE_AREA, MAX_TILE_COLS, MAX_TILE_ROWS, MAX_TILE_WIDTH,
};
pub use uncompressed_header_tail::{
    parse_delta_lf_params, parse_delta_q_params, parse_interpolation_filter,
    parse_loop_filter_params, parse_quantization_params, parse_segmentation_params, DeltaLfParams,
    DeltaQParams, InterpolationFilter, LoopFilterParams, QuantizationParams, SegmentationParams,
    LOOP_FILTER_MODE_DELTAS_DEFAULT, LOOP_FILTER_REF_DELTAS_DEFAULT, MAX_LOOP_FILTER, MAX_SEGMENTS,
    SEGMENTATION_FEATURE_BITS, SEGMENTATION_FEATURE_MAX, SEGMENTATION_FEATURE_SIGNED,
    SEG_LVL_ALT_LF_U, SEG_LVL_ALT_LF_V, SEG_LVL_ALT_LF_Y_H, SEG_LVL_ALT_LF_Y_V, SEG_LVL_ALT_Q,
    SEG_LVL_GLOBALMV, SEG_LVL_MAX, SEG_LVL_REF_FRAME, SEG_LVL_SKIP, TOTAL_REFS_PER_FRAME,
};

/// Crate-local error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A high-level API path is still a scaffold pending the
    /// clean-room rebuild.
    NotImplemented,
    /// The input ended in the middle of an OBU header, extension
    /// header, `leb128()` value, or declared payload.
    UnexpectedEnd,
    /// `obu_forbidden_bit` was set, in violation of §6.2.2.
    ForbiddenBitSet,
    /// The OBU header had `obu_has_size_field == 0`; the walker only
    /// accepts the §5.2 low-overhead format with explicit sizes.
    MissingSizeField,
    /// A `leb128()` value exceeded `(1 << 32) - 1`, the §4.10.5
    /// bitstream-conformance cap.
    Leb128Overflow,
    /// A `leb128()` encoding consumed more than 8 bytes — §4.10.5
    /// requires the MSB of the 8th byte to be 0.
    Leb128TooLong,
    /// An `obu_size` value did not fit in `usize` on this target.
    SizeOverflow,
    /// `seq_profile` was greater than 2 — values 3..=7 are reserved
    /// per §6.4.1.
    ReservedProfile(u8),
    /// `reduced_still_picture_header == 1` but `still_picture == 0`,
    /// in violation of the §6.4.1 conformance requirement.
    ReducedStillRequiresStill,
    /// `idLen` (= `additional_frame_id_length_minus_1 +
    /// `delta_frame_id_length_minus_2 + 3`) exceeded the §6.8.2
    /// requirement that the bit width of `display_frame_id` /
    /// `current_frame_id` must not exceed 16.
    InvalidIdLen,
    /// The frame-header parser hit a `temporal_point_info()` call
    /// site (§5.9.31) — i.e. `decoder_model_info_present_flag &&
    /// !equal_picture_interval`. Decoder-model frame timing isn't
    /// implemented yet; every fixture in this round's corpus parses
    /// without ever triggering this path.
    TemporalPointInfoUnsupported,
    /// The frame-header parser hit the §5.9.2 `if (!FrameIsIntra ||
    /// refresh_frame_flags != allFrames) { if (error_resilient_mode
    /// && enable_order_hint) { ... } }` ref_order_hint walk. The
    /// reads themselves are simple (`NUM_REF_FRAMES *
    /// order_hint_bits` bits of `f(...)`), but the spec then
    /// requires per-slot `RefValid[i] = 0` updates against the
    /// session's `RefOrderHint[]` array. We don't yet track that
    /// state across calls, so we refuse to descend rather than
    /// silently discard the updates. No fixture in the current
    /// corpus exercises this path.
    RefOrderHintWalkUnsupported,
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => write!(
                f,
                "oxideav-av1: orphan-rebuild scaffold — no decoder/encoder wired up"
            ),
            Self::UnexpectedEnd => write!(f, "oxideav-av1: unexpected end of OBU bytestream"),
            Self::ForbiddenBitSet => {
                write!(f, "oxideav-av1: obu_forbidden_bit was set (§6.2.2)")
            }
            Self::MissingSizeField => write!(
                f,
                "oxideav-av1: obu_has_size_field == 0; only the §5.2 low-overhead format is supported"
            ),
            Self::Leb128Overflow => {
                write!(f, "oxideav-av1: leb128 value exceeded the §4.10.5 cap")
            }
            Self::Leb128TooLong => write!(
                f,
                "oxideav-av1: leb128 encoding used more than 8 bytes (§4.10.5)"
            ),
            Self::SizeOverflow => {
                write!(f, "oxideav-av1: obu_size did not fit in usize on this target")
            }
            Self::ReservedProfile(p) => write!(
                f,
                "oxideav-av1: seq_profile {p} is reserved (only 0..=2 are conformant, §6.4.1)"
            ),
            Self::ReducedStillRequiresStill => write!(
                f,
                "oxideav-av1: reduced_still_picture_header == 1 requires still_picture == 1 (§6.4.1)"
            ),
            Self::InvalidIdLen => write!(
                f,
                "oxideav-av1: idLen (delta_frame_id_length_minus_2 + additional_frame_id_length_minus_1 + 3) exceeded 16 (§6.8.2)"
            ),
            Self::TemporalPointInfoUnsupported => write!(
                f,
                "oxideav-av1: temporal_point_info() / decoder-model framing not implemented yet (§5.9.31)"
            ),
            Self::RefOrderHintWalkUnsupported => write!(
                f,
                "oxideav-av1: ref_order_hint walk in §5.9.2 needs RefOrderHint[] state (not yet tracked)"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Decode an AV1 elementary stream.
///
/// Still a stub: this round only added the OBU bytestream walker.
pub fn decode_av1(_bytes: &[u8]) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// Encode YUV data into an AV1 elementary stream.
pub fn encode_av1(_pixels: &[u8], _width: u32, _height: u32) -> Result<Vec<u8>, Error> {
    Err(Error::NotImplemented)
}

/// No-op codec registration — the clean-room scaffold does not yet
/// register a working decoder or encoder.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("av1", register);
