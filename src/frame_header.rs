//! Frame header OBU parser — `uncompressed_header()` through
//! `read_tx_mode()` (§5.9.2 leading slice, §5.9.3 `allow_intrabc`,
//! §5.9.5–§5.9.9, §5.9.11, §5.9.12, §5.9.14, §5.9.15, §5.9.17, §5.9.18,
//! §5.9.19, §5.9.20, §5.9.21).
//!
//! Round 3 covered everything in `uncompressed_header()` through
//! `refresh_frame_flags`. Round 4 extended the parser past
//! `refresh_frame_flags` with the four frame-size sub-syntaxes from
//! §5.9.5 / §5.9.6 / §5.9.7 / §5.9.8 plus the §5.9.9
//! `compute_image_size()` derivation. Round 6 wired the §5.9.3
//! `allow_intrabc` `f(1)` slot (gated by `allow_screen_content_tools &&
//! UpscaledWidth == FrameWidth`) and the §5.9.15 `tile_info()` walk
//! into the streaming parser for intra frames. Round 7 wired §5.9.12
//! `quantization_params()` and §5.9.14 `segmentation_params()` after
//! `tile_info()`. Round 8 wired §5.9.17 `delta_q_params()` and §5.9.18
//! `delta_lf_params()` after `segmentation_params()`. Round 9 derived
//! `CodedLossless` from the §5.9.2 `LosslessArray[]` scan and wired
//! §5.9.11 `loop_filter_params()` after `delta_lf_params()`. Round 10
//! wired §5.9.19 `cdef_params()`; round 11 wired §5.9.20 `lr_params()`
//! (deriving `AllLossless`). Round 12 (this round) wires §5.9.21
//! `read_tx_mode()` after `lr_params()`. The downstream blocks
//! (`frame_reference_mode()`, `skip_mode_params()`, …) remain to be
//! wired in subsequent rounds.
//!
//! ## Syntax / semantics references (all in `docs/video/av1/`)
//!
//!   * §5.9.1 — General frame header OBU syntax
//!   * §5.9.2 — Uncompressed header syntax
//!   * §5.9.5 — `frame_size()`
//!   * §5.9.6 — `render_size()`
//!   * §5.9.7 — `frame_size_with_refs()`
//!   * §5.9.8 — `superres_params()`
//!   * §5.9.9 — `compute_image_size()`
//!   * §5.9.11 — `loop_filter_params()` (via
//!     [`crate::uncompressed_header_tail::parse_loop_filter_params`];
//!     `CodedLossless` derived in-module via `compute_coded_lossless`)
//!   * §5.9.17 — `delta_q_params()` (via
//!     [`crate::uncompressed_header_tail::parse_delta_q_params`])
//!   * §5.9.18 — `delta_lf_params()` (via
//!     [`crate::uncompressed_header_tail::parse_delta_lf_params`])
//!   * §5.9.19 — `cdef_params()` (via
//!     [`crate::uncompressed_header_tail::parse_cdef_params`])
//!   * §5.9.20 — `lr_params()` (via
//!     [`crate::uncompressed_header_tail::parse_lr_params`])
//!   * §5.9.21 — `read_tx_mode()` (via
//!     [`crate::uncompressed_header_tail::parse_tx_mode`])
//!   * §6.8.21 — TX mode semantics
//!   * §6.8.1 — General frame header OBU semantics
//!   * §6.8.2 — Uncompressed header semantics
//!   * §6.8.4 — Frame size semantics
//!   * §6.8.5 — Render size semantics
//!   * §6.8.6 — Frame size with refs semantics
//!   * §6.8.7 — Superres params semantics
//!   * §6.8.8 — Compute image size semantics
//!   * §6.8.10 — Loop filter params semantics
//!   * §6.8.15 — Quantizer index delta params semantics
//!   * §6.8.16 — Loop filter delta params semantics
//!   * §8.7 — `get_qindex()` (the `ignoreDeltaQ` branch that feeds
//!     the `CodedLossless` derivation)
//!
//! §3 constants used here:
//!
//!   * `NUM_REF_FRAMES = 8` — number of reference frame slots; the
//!     spec derives `allFrames = (1 << NUM_REF_FRAMES) - 1 = 0xff`.
//!   * `PRIMARY_REF_NONE = 7` — sentinel for `primary_ref_frame`
//!     indicating no primary reference (loaded as default state).
//!   * `SUPERRES_NUM = 8` — numerator for the superres ratio.
//!   * `SUPERRES_DENOM_MIN = 9` — smallest denominator (with
//!     `coded_denom` defaulting to 0 ⇒ denom = 9).
//!   * `SUPERRES_DENOM_BITS = 3` — bit width of the `coded_denom`
//!     field when `use_superres == 1`.
//!
//! Composition with §5.5: the parser takes a borrowed
//! [`SequenceHeader`] from round 2; sequence-header state controls
//! several conditional reads inside `uncompressed_header()`:
//! `frame_id_numbers_present_flag` (gates `display_frame_id` /
//! `current_frame_id`), `reduced_still_picture_header` (collapses the
//! whole leading block to fixed values), `decoder_model_info_present`
//! (governs `temporal_point_info()` — deferred, see below),
//! `seq_force_screen_content_tools` (decides whether
//! `allow_screen_content_tools` is read or inferred), the matching
//! `seq_force_integer_mv` for `force_integer_mv`, and
//! `order_hint_bits` (the width of the `order_hint` field).
//!
//! ## What the parser does NOT do this round
//!
//!   * `temporal_point_info()` (§5.9.31) — gated by
//!     `decoder_model_info_present_flag && !equal_picture_interval`.
//!     The §5.9.2 leading block uses it twice but every fixture under
//!     `docs/video/av1/fixtures/` is encoded without a decoder model,
//!     so the call site is silently a no-op for the corpus. The
//!     parser returns [`Error::TemporalPointInfoUnsupported`] if it
//!     would actually need to descend into it — i.e. when both gate
//!     conditions are true. The next round can land §5.9.31 alongside
//!     the rest of `tile_info()`.
//!   * `mark_ref_frames()` (§7.20) — a derivation that updates
//!     `RefValid` / `RefOrderHint`; deferred to the inter-frame round
//!     that introduces ref-frame state.
//!   * `frame_size_with_refs()` for INTER frames (§5.9.7) — the
//!     `found_ref` branch reads `UpscaledWidth` / `FrameWidth` /
//!     `FrameHeight` / `RenderWidth` / `RenderHeight` from the
//!     reference-frame state arrays (`RefUpscaledWidth[]` /
//!     `RefFrameHeight[]` / `RefRenderWidth[]` / `RefRenderHeight[]`)
//!     that are not yet tracked across calls in this round.
//!     [`FrameHeader::frame_size`] is `None` for inter frames whose
//!     `frame_size_with_refs()` path would have been taken; the
//!     parser stops at `refresh_frame_flags`. Intra (`KEY_FRAME` /
//!     `INTRA_ONLY_FRAME`) frames go through the no-ref-state
//!     `frame_size()` + `render_size()` path so they parse all the
//!     way to `compute_image_size()`. Inter frames that would have
//!     taken the `frame_size()`+`render_size()` branch (because
//!     `frame_size_override_flag == 0` or `error_resilient_mode ==
//!     1`) still bail out early this round because the parser also
//!     hasn't implemented the inter-frame `ref_frame_idx[]` /
//!     `delta_frame_id_minus_1` walk between `refresh_frame_flags`
//!     and the size block.
//!   * `cdef_params()` (§5.9.19), `lr_params()` (§5.9.20),
//!     `read_tx_mode()` (§5.9.21), `frame_reference_mode()` (§5.9.23),
//!     and everything past `loop_filter_params()` — next round's work.
//!     The intra path now parses through `loop_filter_params()`
//!     (§5.9.11).
//!
//! The bit count consumed is returned in [`FrameHeader::bits_consumed`]
//! so the next round can continue at exactly the right bit.

use crate::bitreader::BitReader;
use crate::sequence_header::{SequenceHeader, SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS};
use crate::tile_info::{read_tile_info, TileInfo};
use crate::uncompressed_header_tail::{
    prev_gm_params_default, read_cdef_params, read_delta_lf_params, read_delta_q_params,
    read_film_grain_params, read_global_motion_params, read_interpolation_filter,
    read_loop_filter_params, read_lr_params, read_quantization_params, read_segmentation_params,
    read_tx_mode, CdefParams, DeltaLfParams, DeltaQParams, FilmGrainContext, FilmGrainParams,
    GlobalMotionParams, InterpolationFilter, LoopFilterParams, LrParams, QuantizationParams,
    SegmentationParams, TxMode, ALTREF_FRAME, LAST_FRAME, MAX_SEGMENTS, REFS_PER_FRAME,
    SEG_LVL_ALT_Q,
};
use crate::Error;

// ---------------------------------------------------------------------
// §3 constants
// ---------------------------------------------------------------------

/// `NUM_REF_FRAMES` (§3): number of reference-frame slots.
pub const NUM_REF_FRAMES: u8 = 8;

/// `PRIMARY_REF_NONE` (§3): sentinel for `primary_ref_frame` meaning
/// no primary reference frame is used (CDF / loop-filter / segment
/// state are reset to defaults).
pub const PRIMARY_REF_NONE: u8 = 7;

/// `allFrames = (1 << NUM_REF_FRAMES) - 1` from §5.9.2.
const ALL_FRAMES: u8 = 0xff;

/// `SUPERRES_NUM` (§3): numerator for the super-resolution upscaling
/// ratio.
pub const SUPERRES_NUM: u32 = 8;

/// `SUPERRES_DENOM_MIN` (§3): smallest denominator for the
/// super-resolution upscaling ratio.
pub const SUPERRES_DENOM_MIN: u32 = 9;

/// `SUPERRES_DENOM_BITS` (§3): bit width of the `coded_denom` field
/// when `use_superres == 1` (§5.9.8).
pub const SUPERRES_DENOM_BITS: u32 = 3;

// §3 reference-frame index symbols (the values RefFrame[0] takes per the
// §6.10.X table: LAST_FRAME=1 .. ALTREF_FRAME=7). `ref_frame_idx[]` is
// indexed by `refFrame - LAST_FRAME`, i.e. 0..=6, so each symbol below
// maps to slot `<SYMBOL> - LAST_FRAME` in the array. Used by §7.8
// `set_frame_refs()`.
const LAST2_FRAME: usize = 2;
const LAST3_FRAME: usize = 3;
const GOLDEN_FRAME: usize = 4;
const BWDREF_FRAME: usize = 5;
const ALTREF2_FRAME: usize = 6;

// ---------------------------------------------------------------------
// FrameType (§6.8.2)
// ---------------------------------------------------------------------

/// `frame_type` per the §6.8.2 enumeration:
///
/// | code | name              |
/// |-----:|:------------------|
/// |   0  | `KEY_FRAME`       |
/// |   1  | `INTER_FRAME`     |
/// |   2  | `INTRA_ONLY_FRAME`|
/// |   3  | `SWITCH_FRAME`    |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    Key,
    Inter,
    IntraOnly,
    Switch,
}

impl FrameType {
    /// Decode the raw 2-bit `frame_type` field.
    pub fn from_raw(raw: u8) -> Self {
        match raw & 0x3 {
            0 => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            _ => Self::Switch,
        }
    }

    /// Raw 2-bit encoding that produced `self`.
    pub fn as_raw(&self) -> u8 {
        match self {
            Self::Key => 0,
            Self::Inter => 1,
            Self::IntraOnly => 2,
            Self::Switch => 3,
        }
    }

    /// `FrameIsIntra = (frame_type == INTRA_ONLY_FRAME || frame_type ==
    /// KEY_FRAME)` per §5.9.2.
    pub fn is_intra(&self) -> bool {
        matches!(self, Self::Key | Self::IntraOnly)
    }
}

// ---------------------------------------------------------------------
// FrameSize (§5.9.5–§5.9.9)
// ---------------------------------------------------------------------

/// Result of the §5.9.5–§5.9.9 frame-size sub-syntax block.
///
/// All four sub-syntaxes (`frame_size()`, `render_size()`,
/// `superres_params()`, `compute_image_size()`) produce a single
/// coherent description of the frame's dimensions; this struct
/// surfaces them together rather than scattering them across
/// optional [`FrameHeader`] fields.
///
/// The §5.9.7 `frame_size_with_refs()` `found_ref == 1` branch
/// short-circuits `frame_size()` + `render_size()` and instead reads
/// `UpscaledWidth` / `FrameWidth` / `FrameHeight` / `RenderWidth` /
/// `RenderHeight` directly from the reference-frame state arrays
/// before calling `superres_params()` + `compute_image_size()`. This
/// round does not track that state, so [`FrameHeader::frame_size`]
/// is `None` whenever the parser would have entered that branch (see
/// the module-level note).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameSize {
    /// `FrameWidth` per §5.9.5 — the **post-superres** coded width,
    /// in pixels. For super-resolved frames this is the downscaled
    /// width that the rest of the decoder works on; the original
    /// (pre-downscale) width is [`Self::upscaled_width`].
    pub frame_width: u32,
    /// `FrameHeight` per §5.9.5. Super-resolution is horizontal only,
    /// so this equals the height as written in the bitstream / the
    /// sequence header's `max_frame_height_minus_1 + 1`.
    pub frame_height: u32,
    /// `RenderWidth` per §5.9.6 — the intended display width. When
    /// `render_and_frame_size_different == 0` this equals
    /// [`Self::upscaled_width`].
    pub render_width: u32,
    /// `RenderHeight` per §5.9.6 — the intended display height. When
    /// `render_and_frame_size_different == 0` this equals
    /// [`Self::frame_height`].
    pub render_height: u32,
    /// `SuperresDenom` per §5.9.8 — equals [`SUPERRES_NUM`] when
    /// `use_superres == 0`, otherwise `coded_denom +
    /// SUPERRES_DENOM_MIN`. The valid range is
    /// `SUPERRES_DENOM_MIN..=SUPERRES_DENOM_MIN + (1 <<
    /// SUPERRES_DENOM_BITS) - 1` (i.e. 9..=16).
    pub superres_denom: u32,
    /// `UpscaledWidth` per §5.9.8 — the width *before* superres
    /// downscaling. Equal to [`Self::frame_width`] when super-res is
    /// not used.
    pub upscaled_width: u32,
    /// `MiCols` per §5.9.9 — the §3 `MI_SIZE = 4` block grid column
    /// count: `2 * ((FrameWidth + 7) >> 3)`.
    pub mi_cols: u32,
    /// `MiRows` per §5.9.9 — the §3 `MI_SIZE = 4` block grid row
    /// count: `2 * ((FrameHeight + 7) >> 3)`.
    pub mi_rows: u32,
    /// `use_superres` per §5.9.8 — `true` when `coded_denom` was read
    /// from the bitstream.
    pub use_superres: bool,
    /// `coded_denom` per §5.9.8 — the raw `f(SUPERRES_DENOM_BITS)`
    /// value, or `0` when `use_superres == 0` (the implicit default
    /// the spec uses to derive `SuperresDenom = SUPERRES_NUM`).
    pub coded_denom: u8,
    /// `render_and_frame_size_different` per §5.9.6.
    pub render_and_frame_size_different: bool,
}

impl FrameSize {
    /// Convenience: did `use_superres == 1` and did the superres
    /// downscale actually change `FrameWidth` away from
    /// `UpscaledWidth`? `super_resolution` fixture: yes;
    /// `tiny-i-only-16x16-prof0`: no (`use_superres == 0`).
    pub fn is_super_resolved(&self) -> bool {
        self.use_superres && self.frame_width != self.upscaled_width
    }
}

// ---------------------------------------------------------------------
// RefInfo (cross-frame reference state)
// ---------------------------------------------------------------------

/// Minimal cross-frame reference-buffer state the inter-frame path of
/// `uncompressed_header()` (§5.9.2) reads from.
///
/// Each array is indexed by a reference-frame **slot** number
/// (`0..NUM_REF_FRAMES`), exactly as the spec's `RefValid[]` /
/// `RefOrderHint[]` / `RefFrameId[]` and the per-frame dimension arrays
/// (`RefUpscaledWidth[]` / `RefFrameHeight[]` / `RefRenderWidth[]` /
/// `RefRenderHeight[]`) are. A decoder session maintains these across
/// frames via the §7.20 reference frame update process — that process
/// is out of scope here; this round only *consumes* the state so an
/// inter `uncompressed_header()` can be parsed end-to-end.
///
/// The fields the parser actually touches this round:
///
///   * `order_hint[]` — used by §7.8 `set_frame_refs()`
///     (`shiftedOrderHints[]`) and the `OrderHints[refFrame]` /
///     `RefFrameSignBias[]` derivations that follow the size block.
///   * `upscaled_width[]` / `frame_height[]` / `render_width[]` /
///     `render_height[]` — used by §5.9.7 `frame_size_with_refs()` on
///     the `found_ref == 1` branch.
///   * `valid[]` / `frame_id[]` — used by the conformance constraints
///     (`RefValid[ref_frame_idx[i]] == 1`, the `expectedFrameId[]`
///     match) and the §5.9.2 `ref_order_hint` walk; surfaced so a
///     session-aware caller can seed them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefInfo {
    /// `RefValid[i]` — `true` when slot `i` holds a decoded frame
    /// available for reference.
    pub valid: [bool; NUM_REF_FRAMES as usize],
    /// `RefOrderHint[i]` — the stored `order_hint` of the frame in
    /// slot `i` (the least-significant `OrderHintBits` of its expected
    /// output order).
    pub order_hint: [u32; NUM_REF_FRAMES as usize],
    /// `RefFrameId[i]` — the stored `current_frame_id` of the frame in
    /// slot `i` (only meaningful when `frame_id_numbers_present_flag`).
    pub frame_id: [u32; NUM_REF_FRAMES as usize],
    /// `RefUpscaledWidth[i]` — the stored `UpscaledWidth` of slot `i`.
    pub upscaled_width: [u32; NUM_REF_FRAMES as usize],
    /// `RefFrameHeight[i]` — the stored `FrameHeight` of slot `i`.
    pub frame_height: [u32; NUM_REF_FRAMES as usize],
    /// `RefRenderWidth[i]` — the stored `RenderWidth` of slot `i`.
    pub render_width: [u32; NUM_REF_FRAMES as usize],
    /// `RefRenderHeight[i]` — the stored `RenderHeight` of slot `i`.
    pub render_height: [u32; NUM_REF_FRAMES as usize],
}

impl Default for RefInfo {
    /// All slots invalid with zeroed hints / ids / dimensions — the
    /// state immediately after a `(KEY_FRAME && show_frame)` frame
    /// resets every slot per the §5.9.2 `RefValid[i] = 0;
    /// RefOrderHint[i] = 0` loop, *before* the §7.20 update stores the
    /// just-decoded frame.
    fn default() -> Self {
        Self {
            valid: [false; NUM_REF_FRAMES as usize],
            order_hint: [0; NUM_REF_FRAMES as usize],
            frame_id: [0; NUM_REF_FRAMES as usize],
            upscaled_width: [0; NUM_REF_FRAMES as usize],
            frame_height: [0; NUM_REF_FRAMES as usize],
            render_width: [0; NUM_REF_FRAMES as usize],
            render_height: [0; NUM_REF_FRAMES as usize],
        }
    }
}

// ---------------------------------------------------------------------
// InterFrameRefs (§5.9.2 inter-frame reference signaling)
// ---------------------------------------------------------------------

/// Reference-frame signaling read on the **inter** branch of §5.9.2,
/// surfaced on [`FrameHeader::inter_refs`] for inter frames (and `None`
/// for intra / show-existing-frame headers).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterFrameRefs {
    /// `frame_refs_short_signaling` (§5.9.2). `false` when
    /// `enable_order_hint == 0` (no bit read). When `true`,
    /// `ref_frame_idx[]` is *computed* by §7.8 `set_frame_refs()` from
    /// `last_frame_idx` / `gold_frame_idx` rather than signaled.
    pub frame_refs_short_signaling: bool,
    /// `last_frame_idx` (§5.9.2) — present only when
    /// `frame_refs_short_signaling == 1`.
    pub last_frame_idx: Option<u8>,
    /// `gold_frame_idx` (§5.9.2) — present only when
    /// `frame_refs_short_signaling == 1`.
    pub gold_frame_idx: Option<u8>,
    /// `ref_frame_idx[i]` for `i = 0..REFS_PER_FRAME` — the slot index
    /// each of the seven inter references (LAST..ALTREF) reads from.
    /// Either signaled explicitly (`f(3)` each) or computed by §7.8.
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    /// `allow_high_precision_mv` (§5.9.2). `false` (no bit) when
    /// `force_integer_mv == 1`; otherwise the read `f(1)`.
    pub allow_high_precision_mv: bool,
    /// `interpolation_filter` (§5.9.10 `read_interpolation_filter()`).
    pub interpolation_filter: InterpolationFilter,
    /// `is_motion_mode_switchable` (§5.9.2 `f(1)`).
    pub is_motion_mode_switchable: bool,
    /// `use_ref_frame_mvs` (§5.9.2). `false` (no bit) when
    /// `error_resilient_mode || !enable_ref_frame_mvs`; otherwise the
    /// read `f(1)`.
    pub use_ref_frame_mvs: bool,
}

// ---------------------------------------------------------------------
// FrameHeader
// ---------------------------------------------------------------------

/// Parsed leading slice of `uncompressed_header()` per §5.9.2.
///
/// Fields are populated according to the §5.9.2 syntax. When a field
/// is conditionally absent in the bitstream, the value here is the
/// §5.9.2 / §6.8.2 inferred default (e.g. `error_resilient_mode = true`
/// when `frame_type == SWITCH_FRAME` or `frame_type == KEY_FRAME &&
/// show_frame`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameHeader {
    /// `show_existing_frame` (§5.9.2 / §6.8.2). Inferred to `false`
    /// when `reduced_still_picture_header == 1`.
    pub show_existing_frame: bool,
    /// `frame_to_show_map_idx` — present only when
    /// `show_existing_frame == 1`. 3-bit index into the reference
    /// frame map.
    pub frame_to_show_map_idx: Option<u8>,
    /// `display_frame_id` — present only when
    /// `show_existing_frame == 1` *and*
    /// `frame_id_numbers_present_flag`. `idLen` bits.
    pub display_frame_id: Option<u32>,
    /// `frame_type` (§6.8.2). Inferred to `KEY_FRAME` when
    /// `reduced_still_picture_header == 1`.
    pub frame_type: FrameType,
    /// `FrameIsIntra` per §5.9.2 (derived from `frame_type`).
    pub frame_is_intra: bool,
    /// `show_frame` — inferred to `true` when
    /// `reduced_still_picture_header == 1`; inferred when this is a
    /// show-existing-frame replay.
    pub show_frame: bool,
    /// `showable_frame` — inferred per §5.9.2:
    /// `(frame_type != KEY_FRAME)` when `show_frame == 1` is read
    /// from above, and `0` when reduced-still-picture-header.
    pub showable_frame: bool,
    /// `error_resilient_mode`. Inferred to `true` for `SWITCH_FRAME`
    /// or `(KEY_FRAME && show_frame)`; otherwise read as `f(1)`.
    pub error_resilient_mode: bool,
    /// `disable_cdf_update`. Read as `f(1)` unconditionally in the
    /// non-show-existing path.
    pub disable_cdf_update: bool,
    /// `allow_screen_content_tools`. Read as `f(1)` when
    /// `seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS`;
    /// otherwise inferred from the sequence header.
    pub allow_screen_content_tools: bool,
    /// `force_integer_mv` after the §5.9.2 `if (FrameIsIntra)` override.
    /// `1` when `FrameIsIntra` regardless of the bitstream bit; `0`
    /// when `!allow_screen_content_tools`; else read or inferred from
    /// the sequence header's `seq_force_integer_mv`.
    pub force_integer_mv: bool,
    /// `current_frame_id` (only when
    /// `frame_id_numbers_present_flag == 1`). Otherwise `0` per §5.9.2.
    pub current_frame_id: u32,
    /// `frame_size_override_flag`. `1` for `SWITCH_FRAME`; `0` for
    /// `reduced_still_picture_header`; otherwise read.
    pub frame_size_override_flag: bool,
    /// `order_hint` — `order_hint_bits` wide. `0` when
    /// `order_hint_bits == 0` (§5.5.1 disabled-order-hint mode).
    pub order_hint: u32,
    /// `primary_ref_frame`. `PRIMARY_REF_NONE` when `FrameIsIntra ||
    /// error_resilient_mode`; otherwise the read 3-bit value.
    pub primary_ref_frame: u8,
    /// `refresh_frame_flags`. `0xff` for `SWITCH_FRAME` or
    /// `(KEY_FRAME && show_frame)`; `0` when show-existing-frame
    /// replays a non-KEY frame; otherwise read.
    pub refresh_frame_flags: u8,
    /// `frame_size()` (§5.9.5) + `render_size()` (§5.9.6) +
    /// `superres_params()` (§5.9.8) + `compute_image_size()`
    /// (§5.9.9) result.
    ///
    /// `None` when:
    ///
    ///   * `show_existing_frame == 1` — §5.9.2 returns immediately
    ///     after the replay block; there is no frame-size syntax.
    ///   * the frame is INTER (would have entered
    ///     `frame_size_with_refs()` or the inter-frame ref-walk
    ///     before `frame_size()`) — the parser doesn't yet track
    ///     `RefUpscaledWidth[]` / `RefFrameHeight[]` /
    ///     `RefRenderWidth[]` / `RefRenderHeight[]` across calls.
    ///
    /// `Some(FrameSize)` for every intra (`KEY_FRAME` /
    /// `INTRA_ONLY_FRAME`) frame.
    pub frame_size: Option<FrameSize>,
    /// `allow_intrabc` per §5.9.3 — the `f(1)` slot inside
    /// `uncompressed_header()` for intra frames, gated by
    /// `allow_screen_content_tools && UpscaledWidth == FrameWidth`.
    /// `false` when the gate is not satisfied (the spec's
    /// `allow_intrabc = 0` initialiser at the top of §5.9.2 stands) or
    /// when the parser stopped before this point (inter frames whose
    /// `frame_size_with_refs()` walk isn't modelled yet, or
    /// show-existing-frame replays).
    pub allow_intrabc: bool,
    /// `disable_frame_end_update_cdf` per §5.9.2 — read as `f(1)` when
    /// `!reduced_still_picture_header && !disable_cdf_update`,
    /// otherwise derived to `true`. Sits between the
    /// `if (FrameIsIntra) { … allow_intrabc … }` block and the
    /// `tile_info()` call, so it must be consumed before the tile
    /// walk for the bit-position accounting to align with the spec.
    pub disable_frame_end_update_cdf: bool,
    /// `tile_info()` (§5.9.15) result. `None` for inter frames /
    /// show-existing-frame replays — those branches stop at
    /// `refresh_frame_flags` because the inter path needs ref-frame
    /// state to fully parse `frame_size_with_refs()`. `Some(TileInfo)`
    /// for every intra (`KEY_FRAME` / `INTRA_ONLY_FRAME`) frame.
    pub tile_info: Option<TileInfo>,
    /// `quantization_params()` (§5.9.12) result. `Some` whenever
    /// `tile_info` is `Some` (intra frames); `None` for inter / show-
    /// existing-frame paths that stop before this point.
    pub quantization_params: Option<QuantizationParams>,
    /// `segmentation_params()` (§5.9.14) result. `Some` whenever
    /// `quantization_params` is `Some` (intra frames); `None` for inter
    /// / show-existing-frame paths that stop before this point.
    pub segmentation_params: Option<SegmentationParams>,
    /// `delta_q_params()` (§5.9.17) result. `Some` whenever
    /// `segmentation_params` is `Some` (intra frames); `None` for inter
    /// / show-existing-frame paths that stop before this point. The
    /// `delta_q_present` `f(1)` slot is only consumed when
    /// `base_q_idx > 0`.
    pub delta_q_params: Option<DeltaQParams>,
    /// `delta_lf_params()` (§5.9.18) result. `Some` whenever
    /// `delta_q_params` is `Some` (intra frames); `None` for inter /
    /// show-existing-frame paths. The block is a no-op when
    /// `delta_q_present == 0`, and the `delta_lf_present` slot is
    /// suppressed when `allow_intrabc == 1`.
    pub delta_lf_params: Option<DeltaLfParams>,
    /// `loop_filter_params()` (§5.9.11) result. `Some` whenever
    /// `delta_lf_params` is `Some` (intra frames); `None` for inter /
    /// show-existing-frame paths. The §5.9.11 `CodedLossless ||
    /// allow_intrabc` short-circuit consumes no bits and resets the
    /// ref-deltas to their §5.9.11 defaults; `CodedLossless` is derived
    /// from the §5.9.2 lines that scan `LosslessArray[]` over the
    /// segment qindexes (via §8.7's `get_qindex(1, segmentId)`).
    pub loop_filter_params: Option<LoopFilterParams>,
    /// `cdef_params()` (§5.9.19) result. `Some` whenever
    /// `loop_filter_params` is `Some` (intra frames); `None` for inter /
    /// show-existing-frame paths. The §5.9.19 `CodedLossless ||
    /// allow_intrabc || !enable_cdef` short-circuit consumes no bits and
    /// leaves `cdef_bits = 0` / `CdefDamping = 3` with zero strengths.
    pub cdef_params: Option<CdefParams>,
    /// `lr_params()` (§5.9.20) result. `Some` whenever `cdef_params` is
    /// `Some` (intra frames); `None` for inter / show-existing-frame
    /// paths. The §5.9.20 `AllLossless || allow_intrabc ||
    /// !enable_restoration` short-circuit consumes no bits and leaves
    /// every plane `RESTORE_NONE` with `UsesLr = 0`. `AllLossless` is
    /// derived as `CodedLossless && (FrameWidth == UpscaledWidth)` —
    /// i.e. `CodedLossless` with no active super-resolution downscale.
    pub lr_params: Option<LrParams>,
    /// `read_tx_mode()` (§5.9.21) result. `Some` whenever `lr_params` is
    /// `Some` (intra frames); `None` for inter / show-existing-frame
    /// paths. When `CodedLossless == 1` the §5.9.21 first branch forces
    /// [`TxMode::Only4x4`] with no bits read; otherwise the `f(1)`
    /// `tx_mode_select` slot selects [`TxMode::TxModeSelect`] (`1`) or
    /// [`TxMode::TxModeLargest`] (`0`).
    pub tx_mode: Option<TxMode>,
    /// `reference_select` (§5.9.23 `frame_reference_mode()`). `Some`
    /// whenever `tx_mode` is `Some` (intra frames); `None` for inter /
    /// show-existing-frame paths. For an intra frame the §5.9.23
    /// `FrameIsIntra` branch forces `reference_select = 0` with no bits
    /// read.
    pub reference_select: Option<bool>,
    /// `skip_mode_present` (§5.9.22 `skip_mode_params()`). `Some` for
    /// intra frames; `None` otherwise. For an intra frame the §5.9.22
    /// `FrameIsIntra` branch sets `skipModeAllowed = 0` so
    /// `skip_mode_present = 0` with no bits read.
    pub skip_mode_present: Option<bool>,
    /// `allow_warped_motion` (§5.9.2). `Some` for intra frames; `None`
    /// otherwise. The §5.9.2 `FrameIsIntra || error_resilient_mode ||
    /// !enable_warped_motion` guard forces it to `0` (no bits read) for
    /// every intra frame.
    pub allow_warped_motion: Option<bool>,
    /// `reduced_tx_set` (§5.9.2 `f(1)`). `Some` for intra frames;
    /// `None` otherwise. Always read from the bitstream (one bit) on
    /// the intra path.
    pub reduced_tx_set: Option<bool>,
    /// `global_motion_params()` (§5.9.24) result. `Some` for intra
    /// frames; `None` otherwise. On the intra path the §5.9.24
    /// `FrameIsIntra` short-circuit leaves every ref `IDENTITY` with no
    /// bits read.
    pub global_motion_params: Option<GlobalMotionParams>,
    /// `film_grain_params()` (§5.9.30) result. `Some` for intra frames;
    /// `None` otherwise. The §5.9.30 short-circuits
    /// (`!film_grain_params_present`, hidden frame, or `apply_grain ==
    /// 0`) all yield the `reset_grain_params()` defaults.
    pub film_grain_params: Option<FilmGrainParams>,
    /// Inter-frame reference signaling (§5.9.2 inter branch + §5.9.7 +
    /// §5.9.10). `Some` for inter / switch frames whose
    /// `uncompressed_header()` parses end-to-end; `None` for intra /
    /// show-existing-frame headers (which have no inter reference
    /// block).
    pub inter_refs: Option<InterFrameRefs>,
    /// Total bits consumed from `payload` by this parse. For intra
    /// frames this now reaches the end of `uncompressed_header()`
    /// (`film_grain_params()`); inter / show-existing paths stop early
    /// as documented on the individual fields.
    pub bits_consumed: usize,
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

/// Parse the leading structural slice of `uncompressed_header()`
/// (§5.9.2) from a frame-header OBU payload, given the active
/// [`SequenceHeader`].
///
/// `payload` is the slice the OBU walker returned for an
/// `OBU_FRAME_HEADER` / `OBU_REDUNDANT_FRAME_HEADER` / `OBU_FRAME`
/// payload (the frame OBU's per-tile data sits after the
/// uncompressed-header bits but is out of scope this round).
///
/// The parser stops after `refresh_frame_flags` is determined; the
/// remaining §5.9.2 work (`frame_size()` / `render_size()` /
/// `frame_size_with_refs()` / interpolation filter / loop-filter
/// params / tile info / quant / segment / CDEF / LR / TX mode / frame
/// reference mode / skip mode / global motion / film grain) is
/// deferred to the next round. [`FrameHeader::bits_consumed`] reports
/// the bit position the next round should resume from.
///
/// ## Errors
///
///   * [`Error::UnexpectedEnd`] — payload ran out mid-header.
///   * [`Error::TemporalPointInfoUnsupported`] — the call sites
///     guarded by `decoder_model_info_present_flag &&
///     !equal_picture_interval` are not implemented yet. Every
///     fixture under `docs/video/av1/fixtures/` parses without
///     hitting this path; bitstreams that enable a decoder model
///     plus variable picture interval will be picked up in a future
///     round alongside §5.9.31.
///   * [`Error::InvalidIdLen`] — §6.8.2 conformance:
///     `idLen <= 16` (the constraint stated alongside
///     `display_frame_id`); we surface the violation here rather than
///     silently reading >16 bits.
///
/// This entry point seeds the cross-frame reference state with
/// [`RefInfo::default`] (every slot invalid, hints / dimensions
/// zeroed). For an inter frame that uses `set_frame_refs()` or
/// `frame_size_with_refs()` against real reference dimensions, use
/// [`parse_frame_header_with_refs`] and pass the session's [`RefInfo`].
pub fn parse_frame_header(payload: &[u8], seq: &SequenceHeader) -> Result<FrameHeader, Error> {
    let mut br = BitReader::new(payload);
    parse_with(&mut br, seq, &RefInfo::default())
}

/// Parse `uncompressed_header()` (§5.9.2) against an explicit
/// cross-frame reference state.
///
/// Identical to [`parse_frame_header`] except the inter-frame path
/// resolves `set_frame_refs()` (§7.8), `frame_size_with_refs()`
/// (§5.9.7), and the `OrderHints[]` / `RefFrameSignBias[]`
/// derivations against the supplied `ref_info` (the decoder session's
/// `RefValid[]` / `RefOrderHint[]` / `RefFrameId[]` plus per-slot
/// `RefUpscaledWidth[]` / `RefFrameHeight[]` / `RefRenderWidth[]` /
/// `RefRenderHeight[]` arrays).
pub fn parse_frame_header_with_refs(
    payload: &[u8],
    seq: &SequenceHeader,
    ref_info: &RefInfo,
) -> Result<FrameHeader, Error> {
    let mut br = BitReader::new(payload);
    parse_with(&mut br, seq, ref_info)
}

fn parse_with(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    ref_info: &RefInfo,
) -> Result<FrameHeader, Error> {
    // §5.9.2 idLen derivation (only valid when frame_id_numbers_present_flag).
    let id_len: u32 = if seq.frame_id_numbers_present_flag {
        u32::from(seq.additional_frame_id_length_minus_1)
            + u32::from(seq.delta_frame_id_length_minus_2)
            + 3
    } else {
        0
    };
    if seq.frame_id_numbers_present_flag && id_len > 16 {
        // §6.8.2 conformance: idLen <= 16.
        return Err(Error::InvalidIdLen);
    }

    // The big reduced-still-picture-header collapse from §5.9.2.
    if seq.reduced_still_picture_header {
        // The reduced path skips every show_existing branch + the
        // frame_type read; everything in the leading block is fixed.
        // We must still read disable_cdf_update, allow_screen_content_tools
        // (conditionally), force_integer_mv (conditionally),
        // frame_size_override_flag (none — derived 0),
        // order_hint (none — order_hint_bits is forced to 0 in §5.5.1
        // when reduced_still_picture_header == 1), primary_ref_frame
        // (derived PRIMARY_REF_NONE), refresh_frame_flags (derived
        // allFrames since frame_type==KEY && show_frame).
        let disable_cdf_update = br.f(1)? == 1;
        let allow_screen_content_tools = read_allow_scc(br, seq)?;
        // FrameIsIntra is true for a derived KEY frame, so
        // force_integer_mv is forced to 1 by the §5.9.2 override
        // regardless of what the seq forces / the bit reads.
        // We still need to consume the bitstream slot if the spec
        // would have asked us to. The spec gates the read on
        // allow_screen_content_tools then overrides to 1 for
        // FrameIsIntra, so we follow it literally.
        consume_force_integer_mv_if_present(br, seq, allow_screen_content_tools)?;
        let force_integer_mv = true; // FrameIsIntra branch.

        // frame_id_numbers_present_flag is gated upstream — §5.5.1
        // forces it off when reduced_still_picture_header is set, so
        // current_frame_id is implicitly 0 and there's no read.
        // frame_size_override_flag is derived 0.
        // order_hint: order_hint_bits is 0 in this path (§5.5.1
        // wrote it out), so we don't read.
        // primary_ref_frame = PRIMARY_REF_NONE (derived).
        // decoder_model_info_present_flag is forced off too in §5.5.1
        // for the reduced path, so no temporal-point-info read.
        // refresh_frame_flags: (KEY && show_frame) so allFrames.
        // Reduced-still is always an intra frame, so the §5.9.2
        // `if (!FrameIsIntra || refresh_frame_flags != allFrames)`
        // ref_order_hint block is skipped (`!true || (allFrames !=
        // allFrames)` = `false || false`), and we drop directly into
        // the `if (FrameIsIntra) { frame_size(); render_size(); }`
        // branch.
        let frame_size = parse_frame_size_block(br, seq, false)?;
        let allow_intrabc = read_allow_intrabc(br, allow_screen_content_tools, &frame_size)?;
        // §5.9.2 disable_frame_end_update_cdf gate. In the reduced-still
        // path `reduced_still_picture_header` is true ⇒ derived to 1, no
        // read. We still surface the value on `FrameHeader`.
        let disable_frame_end_update_cdf = true;
        let tile_info = Some(read_tile_info(
            br,
            frame_size.mi_cols,
            frame_size.mi_rows,
            seq.use_128x128_superblock,
        )?);
        // §5.9.2 spec order after tile_info(): quantization_params()
        // then segmentation_params(). The reduced-still path is an
        // intra (KEY) frame with primary_ref_frame = PRIMARY_REF_NONE,
        // so segmentation collapses to the no-prev-ref branch.
        let quantization_params = read_quantization_params(
            br,
            seq.color_config.num_planes,
            seq.color_config.separate_uv_delta_q,
        )?;
        let segmentation_params = read_segmentation_params(br, PRIMARY_REF_NONE)?;
        // §5.9.2 spec order after segmentation_params(): delta_q_params()
        // (§5.9.17) then delta_lf_params() (§5.9.18). delta_q_params reads
        // the `delta_q_present` `f(1)` slot only when `base_q_idx > 0`;
        // delta_lf_params is a no-op when `delta_q_present == 0` and its
        // `delta_lf_present` slot is suppressed when `allow_intrabc == 1`.
        let delta_q_params = read_delta_q_params(br, quantization_params.base_q_idx)?;
        let delta_lf_params =
            read_delta_lf_params(br, delta_q_params.delta_q_present, allow_intrabc)?;
        // §5.9.2 lines after delta_lf_params(): derive CodedLossless by
        // scanning LosslessArray[] over the segment qindexes, then
        // loop_filter_params() (§5.9.11). The §5.9.11 short-circuit
        // fires when CodedLossless || allow_intrabc.
        let coded_lossless = compute_coded_lossless(&quantization_params, &segmentation_params);
        let loop_filter_params = read_loop_filter_params(
            br,
            seq.color_config.num_planes,
            coded_lossless,
            allow_intrabc,
        )?;
        // §5.9.2 spec order after loop_filter_params(): cdef_params()
        // (§5.9.19). The §5.9.19 short-circuit fires on `CodedLossless
        // || allow_intrabc || !enable_cdef`.
        let cdef_params = read_cdef_params(
            br,
            seq.color_config.num_planes,
            coded_lossless,
            allow_intrabc,
            seq.enable_cdef,
        )?;
        // §5.9.2 spec order: AllLossless is derived as
        // `CodedLossless && (FrameWidth == UpscaledWidth)` before the
        // loop_filter / cdef / lr block; then lr_params() (§5.9.20). The
        // §5.9.20 short-circuit fires on `AllLossless || allow_intrabc
        // || !enable_restoration`.
        let all_lossless = coded_lossless && frame_size.frame_width == frame_size.upscaled_width;
        let lr_params = read_lr_params(
            br,
            seq.color_config.num_planes,
            seq.color_config.subsampling_x,
            seq.color_config.subsampling_y,
            seq.use_128x128_superblock,
            all_lossless,
            allow_intrabc,
            seq.enable_restoration,
        )?;
        // §5.9.2 spec order after lr_params(): read_tx_mode() (§5.9.21).
        // No bits when CodedLossless == 1 (TxMode = ONLY_4X4); otherwise
        // one `tx_mode_select` `f(1)` bit.
        let tx_mode = read_tx_mode(br, coded_lossless)?;
        // §5.9.2 spec order after read_tx_mode(): frame_reference_mode()
        // / skip_mode_params() / allow_warped_motion / reduced_tx_set /
        // global_motion_params() / film_grain_params(). The reduced-still
        // frame is a shown, non-showable KEY frame.
        let (
            reference_select,
            skip_mode_present,
            allow_warped_motion,
            reduced_tx_set,
            global_motion_params,
            film_grain_params,
        ) = read_intra_uncompressed_header_tail(br, seq, true, false, FrameType::Key)?;
        return Ok(FrameHeader {
            show_existing_frame: false,
            frame_to_show_map_idx: None,
            display_frame_id: None,
            frame_type: FrameType::Key,
            frame_is_intra: true,
            show_frame: true,
            showable_frame: false,
            error_resilient_mode: true, // KEY_FRAME && show_frame => 1.
            disable_cdf_update,
            allow_screen_content_tools,
            force_integer_mv,
            current_frame_id: 0,
            frame_size_override_flag: false,
            order_hint: 0,
            primary_ref_frame: PRIMARY_REF_NONE,
            refresh_frame_flags: ALL_FRAMES,
            frame_size: Some(frame_size),
            allow_intrabc,
            disable_frame_end_update_cdf,
            tile_info,
            quantization_params: Some(quantization_params),
            segmentation_params: Some(segmentation_params),
            delta_q_params: Some(delta_q_params),
            delta_lf_params: Some(delta_lf_params),
            loop_filter_params: Some(loop_filter_params),
            cdef_params: Some(cdef_params),
            lr_params: Some(lr_params),
            tx_mode: Some(tx_mode),
            reference_select: Some(reference_select),
            skip_mode_present: Some(skip_mode_present),
            allow_warped_motion: Some(allow_warped_motion),
            reduced_tx_set: Some(reduced_tx_set),
            global_motion_params: Some(global_motion_params),
            film_grain_params: Some(film_grain_params),
            inter_refs: None,
            bits_consumed: br.position(),
        });
    }

    // -----------------------------------------------------------------
    // Non-reduced path.
    // -----------------------------------------------------------------
    let show_existing_frame = br.f(1)? == 1;
    if show_existing_frame {
        let map_idx = br.f(3)? as u8;
        // temporal_point_info gate.
        if decoder_model_info_present(seq) && !equal_picture_interval(seq) {
            return Err(Error::TemporalPointInfoUnsupported);
        }
        let display_id = if seq.frame_id_numbers_present_flag {
            Some(br.f(id_len)? as u32)
        } else {
            None
        };
        // We don't carry the RefFrameType state across calls in this
        // round, so we can't actually look up RefFrameType[map_idx]
        // to know whether it was KEY_FRAME — but the trace tells the
        // decoder this from session state, not the bitstream. For
        // the structural parser we return frame_type = INTER (the
        // common case for replays); a downstream session-aware layer
        // can correct it from its own RefFrameType array. We pick
        // INTER here because §5.9.2 only forces refresh_frame_flags
        // to allFrames if the replayed frame was KEY, and we leave
        // refresh_frame_flags at 0 to match the trace's common
        // "ref-frame-state-replay" reading.
        return Ok(FrameHeader {
            show_existing_frame: true,
            frame_to_show_map_idx: Some(map_idx),
            display_frame_id: display_id,
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
            allow_warped_motion: None,
            reduced_tx_set: None,
            global_motion_params: None,
            film_grain_params: None,
            inter_refs: None,
            bits_consumed: br.position(),
        });
    }

    let frame_type = FrameType::from_raw(br.f(2)? as u8);
    let frame_is_intra = frame_type.is_intra();
    let show_frame = br.f(1)? == 1;

    if show_frame && decoder_model_info_present(seq) && !equal_picture_interval(seq) {
        return Err(Error::TemporalPointInfoUnsupported);
    }

    let showable_frame = if show_frame {
        !matches!(frame_type, FrameType::Key)
    } else {
        br.f(1)? == 1
    };

    let error_resilient_mode = if matches!(frame_type, FrameType::Switch)
        || (matches!(frame_type, FrameType::Key) && show_frame)
    {
        true
    } else {
        br.f(1)? == 1
    };

    // Section that initialises RefValid / RefOrderHint / OrderHints
    // only matters for the inter-frame round — it doesn't touch the
    // bitstream.

    let disable_cdf_update = br.f(1)? == 1;
    let allow_screen_content_tools = read_allow_scc(br, seq)?;
    // Read force_integer_mv from the bitstream if conditions ask for
    // it; the §5.9.2 FrameIsIntra override then forces it to true.
    let raw_force_integer_mv =
        consume_force_integer_mv_if_present(br, seq, allow_screen_content_tools)?;
    let force_integer_mv = if frame_is_intra {
        true
    } else {
        raw_force_integer_mv
    };

    let current_frame_id = if seq.frame_id_numbers_present_flag {
        br.f(id_len)? as u32
    } else {
        0
    };

    let frame_size_override_flag = if matches!(frame_type, FrameType::Switch) {
        true
    } else {
        // reduced-still path already returned; this branch only.
        br.f(1)? == 1
    };

    let order_hint = if seq.order_hint_bits == 0 {
        0
    } else {
        br.f(u32::from(seq.order_hint_bits))? as u32
    };

    let primary_ref_frame = if frame_is_intra || error_resilient_mode {
        PRIMARY_REF_NONE
    } else {
        br.f(3)? as u8
    };

    // temporal_point_info() under decoder_model_info_present_flag &&
    // buffer_removal_time_present_flag is the next syntax block; the
    // spec immediately follows with the operating-point loop reading
    // buffer_removal_time[opNum]. Since we don't claim support for
    // decoder-model frames in this round, we refuse to descend if
    // the gate is active.
    if decoder_model_info_present(seq) {
        return Err(Error::TemporalPointInfoUnsupported);
    }

    let refresh_frame_flags = if matches!(frame_type, FrameType::Switch)
        || (matches!(frame_type, FrameType::Key) && show_frame)
    {
        ALL_FRAMES
    } else {
        br.f(8)? as u8
    };

    // §5.9.2: `if ( !FrameIsIntra || refresh_frame_flags != allFrames )`
    // — the ref_order_hint block only fires when the frame is inter
    // OR when an intra frame leaves some ref slots intact. For an
    // intra frame the inner gate further requires
    // `error_resilient_mode && enable_order_hint`. When taken it reads
    // `NUM_REF_FRAMES` `ref_order_hint[i]` values of `OrderHintBits`
    // each and invalidates any slot whose stored hint no longer
    // matches. Every fixture's intra frame has
    // `refresh_frame_flags == 0xff` (= `allFrames`) and the lone inter
    // fixture has `error_resilient_mode == 0`, so this block reads zero
    // bits in the current corpus; it is modelled so error-resilient
    // inter streams parse correctly. The spec's `RefValid[i] = 0`
    // invalidation on a hint mismatch is a conformance / session-state
    // update with no effect on the bit position or on the parses below
    // (§7.8 `set_frame_refs()` keys off `RefOrderHint[]`, not
    // `RefValid[]`), so we read the bits and leave the session's
    // `ref_info` untouched here.
    if (!frame_is_intra || refresh_frame_flags != ALL_FRAMES)
        && error_resilient_mode
        && seq.enable_order_hint
    {
        for _ in 0..NUM_REF_FRAMES as usize {
            let _ref_order_hint = br.f(u32::from(seq.order_hint_bits))?;
        }
    }

    // §5.9.2: the size block. For intra frames we always go through
    // frame_size() + render_size() (the no-superres / non-ref path is
    // exactly what `parse_frame_size_block` handles). For inter frames
    // the spec walks `frame_refs_short_signaling`, `ref_frame_idx[]`,
    // and then conditionally calls `frame_size_with_refs()` or
    // `frame_size()`+`render_size()`, followed by the inter-only
    // `allow_high_precision_mv` / `read_interpolation_filter()` /
    // `is_motion_mode_switchable` / `use_ref_frame_mvs` block. Both
    // paths converge on `disable_frame_end_update_cdf` + the shared
    // tile/quant/segment/.../film-grain tail.
    let mut inter_refs: Option<InterFrameRefs> = None;
    let (
        frame_size,
        allow_intrabc,
        disable_frame_end_update_cdf,
        tile_info,
        quantization_params,
        segmentation_params,
        delta_q_params,
        delta_lf_params,
        loop_filter_params,
        cdef_params,
        lr_params,
        tx_mode,
        reference_select,
        skip_mode_present,
        allow_warped_motion,
        reduced_tx_set,
        global_motion_params,
        film_grain_params,
    ) = if frame_is_intra {
        let fs = parse_frame_size_block(br, seq, frame_size_override_flag)?;
        let aib = read_allow_intrabc(br, allow_screen_content_tools, &fs)?;
        // §5.9.2 disable_frame_end_update_cdf — read as `f(1)` unless
        // `reduced_still_picture_header || disable_cdf_update`. The
        // reduced-still path is the early return above; here we know
        // `reduced_still_picture_header == false`.
        let dfeuc = if disable_cdf_update {
            true
        } else {
            br.f(1)? == 1
        };
        let ti = read_tile_info(br, fs.mi_cols, fs.mi_rows, seq.use_128x128_superblock)?;
        // §5.9.2 spec order after tile_info(): quantization_params()
        // (§5.9.12) then segmentation_params() (§5.9.14). Both depend
        // on session state already settled by this point:
        // num_planes / separate_uv_delta_q from the sequence header,
        // primary_ref_frame from the per-frame slot above.
        let qp = read_quantization_params(
            br,
            seq.color_config.num_planes,
            seq.color_config.separate_uv_delta_q,
        )?;
        let sp = read_segmentation_params(br, primary_ref_frame)?;
        // §5.9.2 spec order after segmentation_params(): delta_q_params()
        // (§5.9.17) then delta_lf_params() (§5.9.18). delta_q reads its
        // `delta_q_present` slot only when `base_q_idx > 0`; delta_lf is
        // a no-op when `delta_q_present == 0` and its `delta_lf_present`
        // slot is suppressed when `allow_intrabc == 1`.
        let dq = read_delta_q_params(br, qp.base_q_idx)?;
        let dlf = read_delta_lf_params(br, dq.delta_q_present, aib)?;
        // §5.9.2 lines after delta_lf_params(): derive CodedLossless by
        // scanning LosslessArray[] over the segment qindexes (§8.7
        // get_qindex(1, segmentId)), then loop_filter_params()
        // (§5.9.11). The §5.9.11 short-circuit fires on
        // CodedLossless || allow_intrabc.
        let coded_lossless = compute_coded_lossless(&qp, &sp);
        let lf = read_loop_filter_params(br, seq.color_config.num_planes, coded_lossless, aib)?;
        // §5.9.2 spec order after loop_filter_params(): cdef_params()
        // (§5.9.19). The §5.9.19 short-circuit fires on `CodedLossless
        // || allow_intrabc || !enable_cdef`.
        let cdef = read_cdef_params(
            br,
            seq.color_config.num_planes,
            coded_lossless,
            aib,
            seq.enable_cdef,
        )?;
        // §5.9.2 spec order: AllLossless is derived as
        // `CodedLossless && (FrameWidth == UpscaledWidth)` before the
        // loop_filter / cdef / lr block; then lr_params() (§5.9.20). The
        // §5.9.20 short-circuit fires on `AllLossless || allow_intrabc
        // || !enable_restoration`. Super-resolution downscaling
        // (`FrameWidth != UpscaledWidth`) keeps AllLossless 0 even when
        // CodedLossless is 1, so a downscaled lossless frame still walks
        // the full LR path.
        let all_lossless = coded_lossless && fs.frame_width == fs.upscaled_width;
        let lr = read_lr_params(
            br,
            seq.color_config.num_planes,
            seq.color_config.subsampling_x,
            seq.color_config.subsampling_y,
            seq.use_128x128_superblock,
            all_lossless,
            aib,
            seq.enable_restoration,
        )?;
        // §5.9.2 spec order after lr_params(): read_tx_mode() (§5.9.21).
        // No bits when CodedLossless == 1 (TxMode = ONLY_4X4); otherwise
        // one `tx_mode_select` `f(1)` bit.
        let tx = read_tx_mode(br, coded_lossless)?;
        // §5.9.2 spec order after read_tx_mode(): frame_reference_mode()
        // / skip_mode_params() / allow_warped_motion / reduced_tx_set /
        // global_motion_params() / film_grain_params(). For an intra
        // frame everything but `reduced_tx_set` (one `f(1)` bit) and the
        // `film_grain_params()` reads collapses without consuming bits.
        let (rs, smp, awm, rts, gm, fg) =
            read_intra_uncompressed_header_tail(br, seq, show_frame, showable_frame, frame_type)?;
        (
            Some(fs),
            aib,
            dfeuc,
            Some(ti),
            Some(qp),
            Some(sp),
            Some(dq),
            Some(dlf),
            Some(lf),
            Some(cdef),
            Some(lr),
            Some(tx),
            Some(rs),
            Some(smp),
            Some(awm),
            Some(rts),
            Some(gm),
            Some(fg),
        )
    } else {
        // -------------------------------------------------------------
        // Inter / switch frame path (§5.9.2 `else` of `if (FrameIsIntra)`).
        // -------------------------------------------------------------

        // §5.9.2 frame_refs_short_signaling + ref_frame_idx[].
        let frame_refs_short_signaling = if seq.enable_order_hint {
            br.f(1)? == 1
        } else {
            false
        };
        let (short_last, short_gold, mut ref_frame_idx) = if frame_refs_short_signaling {
            let last_frame_idx = br.f(3)? as u8;
            let gold_frame_idx = br.f(3)? as u8;
            // §7.8 set_frame_refs() computes ref_frame_idx[] from
            // last_frame_idx / gold_frame_idx + RefOrderHint[].
            let computed = set_frame_refs(
                last_frame_idx,
                gold_frame_idx,
                order_hint,
                seq.order_hint_bits,
                seq.enable_order_hint,
                &ref_info.order_hint,
            );
            (Some(last_frame_idx), Some(gold_frame_idx), computed)
        } else {
            (None, None, [0u8; REFS_PER_FRAME])
        };

        for entry in ref_frame_idx.iter_mut().take(REFS_PER_FRAME) {
            if !frame_refs_short_signaling {
                *entry = br.f(3)? as u8;
            }
            // §5.9.2: when frame_id_numbers_present_flag the
            // delta_frame_id_minus_1 / expectedFrameId[] block reads
            // `delta_frame_id_length_minus_2 + 2` bits per ref. No
            // fixture in the corpus enables frame ids, but model it so
            // such streams parse.
            if seq.frame_id_numbers_present_flag {
                let n = u32::from(seq.delta_frame_id_length_minus_2) + 2;
                let _delta_frame_id_minus_1 = br.f(n)?;
            }
        }

        // §5.9.2: frame_size_with_refs() when the size is overridden and
        // the frame isn't error-resilient; otherwise frame_size() +
        // render_size().
        let fs = if frame_size_override_flag && !error_resilient_mode {
            read_frame_size_with_refs(br, seq, frame_size_override_flag, &ref_frame_idx, ref_info)?
        } else {
            parse_frame_size_block(br, seq, frame_size_override_flag)?
        };

        // §5.9.2: allow_high_precision_mv.
        let allow_high_precision_mv = if force_integer_mv {
            false
        } else {
            br.f(1)? == 1
        };

        // §5.9.10 read_interpolation_filter().
        let interpolation_filter = read_interpolation_filter(br)?;

        // §5.9.2: is_motion_mode_switchable.
        let is_motion_mode_switchable = br.f(1)? == 1;

        // §5.9.2: use_ref_frame_mvs.
        let use_ref_frame_mvs = if error_resilient_mode || !seq.enable_ref_frame_mvs {
            false
        } else {
            br.f(1)? == 1
        };

        // §5.9.2 OrderHints[] / RefFrameSignBias[] derivation reads no
        // bits (it consults RefOrderHint[ ref_frame_idx[i] ]). We don't
        // surface the per-ref sign-bias array this round; the size-block
        // and motion-precision fields above are what the inter path
        // needs to keep parsing.

        // §5.9.2: disable_frame_end_update_cdf — read as `f(1)` unless
        // `reduced_still_picture_header || disable_cdf_update`. The
        // reduced-still path returned earlier, so only disable_cdf_update
        // matters here.
        let dfeuc = if disable_cdf_update {
            true
        } else {
            br.f(1)? == 1
        };

        // §5.9.2: tile_info() + quant/segment/delta tail (shared with
        // the intra path, but `allow_intrabc` is always 0 for inter).
        let aib = false;
        let ti = read_tile_info(br, fs.mi_cols, fs.mi_rows, seq.use_128x128_superblock)?;
        let qp = read_quantization_params(
            br,
            seq.color_config.num_planes,
            seq.color_config.separate_uv_delta_q,
        )?;
        let sp = read_segmentation_params(br, primary_ref_frame)?;
        let dq = read_delta_q_params(br, qp.base_q_idx)?;
        let dlf = read_delta_lf_params(br, dq.delta_q_present, aib)?;
        let coded_lossless = compute_coded_lossless(&qp, &sp);
        let lf = read_loop_filter_params(br, seq.color_config.num_planes, coded_lossless, aib)?;
        let cdef = read_cdef_params(
            br,
            seq.color_config.num_planes,
            coded_lossless,
            aib,
            seq.enable_cdef,
        )?;
        let all_lossless = coded_lossless && fs.frame_width == fs.upscaled_width;
        let lr = read_lr_params(
            br,
            seq.color_config.num_planes,
            seq.color_config.subsampling_x,
            seq.color_config.subsampling_y,
            seq.use_128x128_superblock,
            all_lossless,
            aib,
            seq.enable_restoration,
        )?;
        let tx = read_tx_mode(br, coded_lossless)?;

        // §5.9.23 frame_reference_mode() — inter frame reads
        // reference_select `f(1)`.
        let reference_select = br.f(1)? == 1;
        // §5.9.22 skip_mode_params(). skipModeAllowed requires
        // reference_select && enable_order_hint && a valid
        // forward+backward (or two-forward) reference pair derived from
        // RefOrderHint[ ref_frame_idx[i] ]. The lone inter fixture has
        // reference_select == 0 so skipModeAllowed == 0 with no bit
        // read; the full derivation against arbitrary RefOrderHint[]
        // values is a followup.
        let skip_mode_present = read_skip_mode_present(
            br,
            reference_select,
            seq.enable_order_hint,
            order_hint,
            seq.order_hint_bits,
            &ref_frame_idx,
            ref_info,
        )?;
        // §5.9.2 allow_warped_motion guard: read `f(1)` only when
        // `!error_resilient_mode && enable_warped_motion` (FrameIsIntra
        // is false on this branch).
        let allow_warped_motion = if error_resilient_mode || !seq.enable_warped_motion {
            false
        } else {
            br.f(1)? == 1
        };
        // §5.9.2 reduced_tx_set f(1).
        let reduced_tx_set = br.f(1)? == 1;
        // §5.9.24 global_motion_params() — inter path reads.
        let gm = read_global_motion_params(
            br,
            false,
            allow_high_precision_mv,
            &prev_gm_params_default(),
        )?;
        // §5.9.30 film_grain_params().
        let fg = read_film_grain_params(
            br,
            FilmGrainContext {
                film_grain_params_present: seq.film_grain_params_present,
                show_frame,
                showable_frame,
                is_inter_frame: matches!(frame_type, FrameType::Inter),
                mono_chrome: seq.color_config.mono_chrome,
                subsampling_x: seq.color_config.subsampling_x,
                subsampling_y: seq.color_config.subsampling_y,
            },
        )?;

        inter_refs = Some(InterFrameRefs {
            frame_refs_short_signaling,
            last_frame_idx: short_last,
            gold_frame_idx: short_gold,
            ref_frame_idx,
            allow_high_precision_mv,
            interpolation_filter,
            is_motion_mode_switchable,
            use_ref_frame_mvs,
        });

        (
            Some(fs),
            aib,
            dfeuc,
            Some(ti),
            Some(qp),
            Some(sp),
            Some(dq),
            Some(dlf),
            Some(lf),
            Some(cdef),
            Some(lr),
            Some(tx),
            Some(reference_select),
            Some(skip_mode_present),
            Some(allow_warped_motion),
            Some(reduced_tx_set),
            Some(gm),
            Some(fg),
        )
    };

    Ok(FrameHeader {
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        display_frame_id: None,
        frame_type,
        frame_is_intra,
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
        frame_size,
        allow_intrabc,
        disable_frame_end_update_cdf,
        tile_info,
        quantization_params,
        segmentation_params,
        delta_q_params,
        delta_lf_params,
        loop_filter_params,
        cdef_params,
        lr_params,
        tx_mode,
        reference_select,
        skip_mode_present,
        allow_warped_motion,
        reduced_tx_set,
        global_motion_params,
        film_grain_params,
        inter_refs,
        bits_consumed: br.position(),
    })
}

/// `get_relative_dist( a, b )` per §5.9.3 — the sign-extended distance
/// between two order hints. Returns `0` when order hints are disabled.
fn get_relative_dist(a: i64, b: i64, order_hint_bits: u8, enable_order_hint: bool) -> i64 {
    if !enable_order_hint {
        return 0;
    }
    let mut diff = a - b;
    let m = 1i64 << (i64::from(order_hint_bits) - 1);
    diff = (diff & (m - 1)) - (diff & m);
    diff
}

/// `set_frame_refs()` per §7.8 — compute the seven `ref_frame_idx[]`
/// entries from `last_frame_idx` / `gold_frame_idx` and the session's
/// `RefOrderHint[]` array when `frame_refs_short_signaling == 1`.
///
/// `ref_frame_idx[]` is indexed by `refFrame - LAST_FRAME`
/// (`0..REFS_PER_FRAME`); each entry holds a slot index in
/// `0..NUM_REF_FRAMES`. The algorithm:
///
///   1. Seed LAST_FRAME / GOLDEN_FRAME from the two explicit indices,
///      mark them used, and prepare `shiftedOrderHints[]`.
///   2. ALTREF_FRAME = latest backward ref, BWDREF_FRAME / ALTREF2_FRAME
///      = the two earliest backward refs.
///   3. The remaining `Ref_Frame_List` slots = forward refs in
///      anti-chronological order.
///   4. Any still-unset entry = the ref with the smallest output order.
fn set_frame_refs(
    last_frame_idx: u8,
    gold_frame_idx: u8,
    order_hint: u32,
    order_hint_bits: u8,
    enable_order_hint: bool,
    ref_order_hint: &[u32; NUM_REF_FRAMES as usize],
) -> [u8; REFS_PER_FRAME] {
    const N: usize = NUM_REF_FRAMES as usize;
    let last = last_frame_idx as usize;
    let gold = gold_frame_idx as usize;

    // -1 sentinel for "not yet assigned"; resolved to u8 at the end.
    // ref_frame_idx[] is indexed by `refFrame - LAST_FRAME`, so
    // LAST_FRAME maps to slot 0 and GOLDEN_FRAME to `GOLDEN_FRAME -
    // LAST_FRAME`.
    let mut ref_frame_idx: [i32; REFS_PER_FRAME] = [-1; REFS_PER_FRAME];
    ref_frame_idx[0] = last as i32; // §7.8: ref_frame_idx[LAST_FRAME - LAST_FRAME].
    ref_frame_idx[GOLDEN_FRAME - LAST_FRAME] = gold as i32;

    let mut used_frame = [false; N];
    used_frame[last] = true;
    used_frame[gold] = true;

    let cur_frame_hint: i64 = 1 << (i64::from(order_hint_bits) - 1);
    let mut shifted = [0i64; N];
    for (i, slot) in shifted.iter_mut().enumerate() {
        *slot = cur_frame_hint
            + get_relative_dist(
                i64::from(ref_order_hint[i]),
                i64::from(order_hint),
                order_hint_bits,
                enable_order_hint,
            );
    }

    // find_latest_backward(): unused slot with the highest hint at or
    // beyond curFrameHint.
    let find_latest_backward = |used: &[bool; N]| -> i32 {
        let mut r = -1i32;
        let mut latest = 0i64;
        for i in 0..N {
            let h = shifted[i];
            if !used[i] && h >= cur_frame_hint && (r < 0 || h >= latest) {
                r = i as i32;
                latest = h;
            }
        }
        r
    };
    // find_earliest_backward(): unused slot with the lowest hint at or
    // beyond curFrameHint.
    let find_earliest_backward = |used: &[bool; N]| -> i32 {
        let mut r = -1i32;
        let mut earliest = 0i64;
        for i in 0..N {
            let h = shifted[i];
            if !used[i] && h >= cur_frame_hint && (r < 0 || h < earliest) {
                r = i as i32;
                earliest = h;
            }
        }
        r
    };
    // find_latest_forward(): unused slot with the highest hint below
    // curFrameHint.
    let find_latest_forward = |used: &[bool; N]| -> i32 {
        let mut r = -1i32;
        let mut latest = 0i64;
        for i in 0..N {
            let h = shifted[i];
            if !used[i] && h < cur_frame_hint && (r < 0 || h >= latest) {
                r = i as i32;
                latest = h;
            }
        }
        r
    };

    // ALTREF_FRAME — latest backward.
    let r = find_latest_backward(&used_frame);
    if r >= 0 {
        ref_frame_idx[ALTREF_FRAME - LAST_FRAME] = r;
        used_frame[r as usize] = true;
    }
    // BWDREF_FRAME — earliest backward.
    let r = find_earliest_backward(&used_frame);
    if r >= 0 {
        ref_frame_idx[BWDREF_FRAME - LAST_FRAME] = r;
        used_frame[r as usize] = true;
    }
    // ALTREF2_FRAME — next earliest backward.
    let r = find_earliest_backward(&used_frame);
    if r >= 0 {
        ref_frame_idx[ALTREF2_FRAME - LAST_FRAME] = r;
        used_frame[r as usize] = true;
    }

    // Remaining references = forward refs in anti-chronological order.
    const REF_FRAME_LIST: [usize; REFS_PER_FRAME - 2] = [
        LAST2_FRAME,
        LAST3_FRAME,
        BWDREF_FRAME,
        ALTREF2_FRAME,
        ALTREF_FRAME,
    ];
    for &ref_frame in REF_FRAME_LIST.iter() {
        if ref_frame_idx[ref_frame - LAST_FRAME] < 0 {
            let r = find_latest_forward(&used_frame);
            if r >= 0 {
                ref_frame_idx[ref_frame - LAST_FRAME] = r;
                used_frame[r as usize] = true;
            }
        }
    }

    // Finally, any remaining entries = the ref with smallest output
    // order.
    let mut r = -1i32;
    let mut earliest = 0i64;
    for (i, &h) in shifted.iter().enumerate().take(N) {
        if r < 0 || h < earliest {
            r = i as i32;
            earliest = h;
        }
    }
    for entry in ref_frame_idx.iter_mut() {
        if *entry < 0 {
            *entry = r;
        }
    }

    let mut out = [0u8; REFS_PER_FRAME];
    for (o, v) in out.iter_mut().zip(ref_frame_idx.iter()) {
        // Every entry is assigned a non-negative slot by the final
        // fallback (which always finds a ref when NUM_REF_FRAMES > 0).
        *o = (*v).max(0) as u8;
    }
    out
}

/// `frame_size_with_refs()` per §5.9.7. Scans the seven references for
/// the first one whose `found_ref` bit is set, inheriting that ref's
/// dimensions; if none is found, falls back to `frame_size()` +
/// `render_size()`. On a found ref it runs `superres_params()` +
/// `compute_image_size()` against the inherited width.
fn read_frame_size_with_refs(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    frame_size_override_flag: bool,
    ref_frame_idx: &[u8; REFS_PER_FRAME],
    ref_info: &RefInfo,
) -> Result<FrameSize, Error> {
    let mut found = false;
    let mut upscaled_width = 0u32;
    let mut frame_height = 0u32;
    let mut render_width = 0u32;
    let mut render_height = 0u32;
    for &slot in ref_frame_idx.iter() {
        let found_ref = br.f(1)? == 1;
        if found_ref {
            let s = slot as usize;
            upscaled_width = ref_info.upscaled_width[s];
            frame_height = ref_info.frame_height[s];
            render_width = ref_info.render_width[s];
            render_height = ref_info.render_height[s];
            found = true;
            break;
        }
    }

    if !found {
        // §5.9.7: frame_size() + render_size().
        return parse_frame_size_block(br, seq, frame_size_override_flag);
    }

    // §5.9.8 superres_params() — runs against the inherited
    // UpscaledWidth (which §5.9.7 assigned to FrameWidth before this
    // call).
    let (use_superres, coded_denom, superres_denom) = if seq.enable_superres {
        let use_superres = br.f(1)? == 1;
        if use_superres {
            let coded_denom = br.f(SUPERRES_DENOM_BITS)? as u8;
            (
                true,
                coded_denom,
                u32::from(coded_denom) + SUPERRES_DENOM_MIN,
            )
        } else {
            (false, 0u8, SUPERRES_NUM)
        }
    } else {
        (false, 0u8, SUPERRES_NUM)
    };
    // The inherited width is the pre-superres UpscaledWidth; §5.9.8 then
    // downscales FrameWidth.
    let frame_width = (upscaled_width * SUPERRES_NUM + (superres_denom / 2)) / superres_denom;

    // §5.9.9 compute_image_size().
    let mi_cols = 2 * ((frame_width + 7) >> 3);
    let mi_rows = 2 * ((frame_height + 7) >> 3);

    Ok(FrameSize {
        frame_width,
        frame_height,
        render_width,
        render_height,
        superres_denom,
        upscaled_width,
        mi_cols,
        mi_rows,
        use_superres,
        coded_denom,
        // §5.9.7's found-ref branch does not call render_size(), so no
        // `render_and_frame_size_different` bit is read; the inherited
        // render dimensions stand.
        render_and_frame_size_different: false,
    })
}

/// `skip_mode_params()` per §5.9.22. Returns `skip_mode_present`. The
/// `skipModeAllowed` derivation scans `RefOrderHint[ ref_frame_idx[i] ]`
/// for a forward + backward reference pair (or two forward refs); when
/// allowed it reads `skip_mode_present` `f(1)`, otherwise it is `0` with
/// no bit read.
#[allow(clippy::too_many_arguments)]
fn read_skip_mode_present(
    br: &mut BitReader<'_>,
    reference_select: bool,
    enable_order_hint: bool,
    order_hint: u32,
    order_hint_bits: u8,
    ref_frame_idx: &[u8; REFS_PER_FRAME],
    ref_info: &RefInfo,
) -> Result<bool, Error> {
    // §5.9.22: FrameIsIntra is false on this branch (inter path).
    if !reference_select || !enable_order_hint {
        return Ok(false);
    }

    let rel = |a: u32| -> i64 {
        get_relative_dist(
            i64::from(a),
            i64::from(order_hint),
            order_hint_bits,
            enable_order_hint,
        )
    };

    let mut forward_idx: i32 = -1;
    let mut backward_idx: i32 = -1;
    let mut forward_hint: u32 = 0;
    let mut backward_hint: u32 = 0;
    for (i, &slot) in ref_frame_idx.iter().enumerate() {
        let ref_hint = ref_info.order_hint[slot as usize];
        if rel(ref_hint) < 0 {
            if forward_idx < 0
                || rel_pair(ref_hint, forward_hint, order_hint_bits, enable_order_hint) > 0
            {
                forward_idx = i as i32;
                forward_hint = ref_hint;
            }
        } else if rel(ref_hint) > 0
            && (backward_idx < 0
                || rel_pair(ref_hint, backward_hint, order_hint_bits, enable_order_hint) < 0)
        {
            backward_idx = i as i32;
            backward_hint = ref_hint;
        }
    }

    let skip_mode_allowed = if forward_idx < 0 {
        false
    } else if backward_idx >= 0 {
        true
    } else {
        // Two-forward-reference fallback.
        let mut second_forward_idx: i32 = -1;
        let mut second_forward_hint: u32 = 0;
        for (i, &slot) in ref_frame_idx.iter().enumerate() {
            let ref_hint = ref_info.order_hint[slot as usize];
            if rel_pair(ref_hint, forward_hint, order_hint_bits, enable_order_hint) < 0
                && (second_forward_idx < 0
                    || rel_pair(
                        ref_hint,
                        second_forward_hint,
                        order_hint_bits,
                        enable_order_hint,
                    ) > 0)
            {
                second_forward_idx = i as i32;
                second_forward_hint = ref_hint;
            }
        }
        second_forward_idx >= 0
    };

    if skip_mode_allowed {
        Ok(br.f(1)? == 1)
    } else {
        Ok(false)
    }
}

/// `get_relative_dist(a, b)` for two stored order hints (helper for
/// `read_skip_mode_present`).
fn rel_pair(a: u32, b: u32, order_hint_bits: u8, enable_order_hint: bool) -> i64 {
    get_relative_dist(
        i64::from(a),
        i64::from(b),
        order_hint_bits,
        enable_order_hint,
    )
}

/// Derive `CodedLossless` per the §5.9.2 lines that follow
/// `delta_lf_params()`:
///
/// ```text
/// CodedLossless = 1
/// for ( segmentId = 0; segmentId < MAX_SEGMENTS; segmentId++ ) {
///     qindex = get_qindex( 1, segmentId )
///     LosslessArray[ segmentId ] = qindex == 0 && DeltaQYDc == 0 &&
///                                  DeltaQUAc == 0 && DeltaQUDc == 0 &&
///                                  DeltaQVAc == 0 && DeltaQVDc == 0
///     if ( !LosslessArray[ segmentId ] )
///         CodedLossless = 0
/// }
/// ```
///
/// `get_qindex( 1, segmentId )` (the §8.7 quantiser-index function with
/// `ignoreDeltaQ == 1`) is, for the uncompressed-header derivation:
///   * `Clip3( 0, 255, base_q_idx + FeatureData[ segmentId ][
///     SEG_LVL_ALT_Q ] )` when `seg_feature_active_idx( segmentId,
///     SEG_LVL_ALT_Q ) == segmentation_enabled &&
///     FeatureEnabled[ segmentId ][ SEG_LVL_ALT_Q ]` is 1;
///   * `base_q_idx` otherwise.
///
/// All four `DeltaQ?Ac` / `DeltaQ?Dc` checks reduce to the four
/// non-Y-AC `delta_q_*` offsets the §5.9.12 parse already surfaced on
/// [`QuantizationParams`] (`DeltaQYDc` plus the U/V DC/AC pair).
fn compute_coded_lossless(qp: &QuantizationParams, sp: &SegmentationParams) -> bool {
    // The DeltaQ?* == 0 conjunction is segment-independent: when any of
    // the five §5.9.12 offsets is non-zero, no segment can be lossless.
    let deltas_all_zero = qp.delta_q_y_dc == 0
        && qp.delta_q_u_dc == 0
        && qp.delta_q_u_ac == 0
        && qp.delta_q_v_dc == 0
        && qp.delta_q_v_ac == 0;
    if !deltas_all_zero {
        return false;
    }
    for segment_id in 0..MAX_SEGMENTS {
        // get_qindex( 1, segmentId ) with ignoreDeltaQ = 1.
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

/// Read the §5.9.2 tail that follows `read_tx_mode()` on the **intra**
/// path: `frame_reference_mode()` (§5.9.23), `skip_mode_params()`
/// (§5.9.22), the `allow_warped_motion` slot, `reduced_tx_set` (`f(1)`),
/// `global_motion_params()` (§5.9.24), and `film_grain_params()`
/// (§5.9.30).
///
/// For an intra frame every gate but `reduced_tx_set` collapses without
/// reading bits:
///   * `frame_reference_mode()`: `FrameIsIntra` ⇒ `reference_select = 0`.
///   * `skip_mode_params()`: `FrameIsIntra` ⇒ `skipModeAllowed = 0` ⇒
///     `skip_mode_present = 0`.
///   * `allow_warped_motion`: the §5.9.2 `FrameIsIntra || ...` guard ⇒
///     `0`.
///   * `global_motion_params()`: the §5.9.24 `FrameIsIntra` early
///     return ⇒ identity defaults, no bits.
///
/// `reduced_tx_set` is one `f(1)` bit, and `film_grain_params()` reads
/// per §5.9.30 (which short-circuits unless the sequence enables grain
/// *and* the frame is shown/showable *and* `apply_grain == 1`).
#[allow(clippy::type_complexity)]
fn read_intra_uncompressed_header_tail(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    show_frame: bool,
    showable_frame: bool,
    frame_type: FrameType,
) -> Result<(bool, bool, bool, bool, GlobalMotionParams, FilmGrainParams), Error> {
    // §5.9.23 frame_reference_mode() — FrameIsIntra ⇒ reference_select = 0.
    let reference_select = false;
    // §5.9.22 skip_mode_params() — FrameIsIntra ⇒ skip_mode_present = 0.
    let skip_mode_present = false;
    // §5.9.2 allow_warped_motion guard fires for any intra frame.
    let allow_warped_motion = false;
    // §5.9.2 reduced_tx_set f(1).
    let reduced_tx_set = br.f(1)? == 1;
    // §5.9.24 global_motion_params() — FrameIsIntra short-circuit.
    // allow_high_precision_mv is 0 for intra frames (no read happens on
    // this path), and PrevGmParams is irrelevant under the early return.
    let global_motion = read_global_motion_params(br, true, false, &prev_gm_params_default())?;
    // §5.9.30 film_grain_params().
    let film_grain = read_film_grain_params(
        br,
        FilmGrainContext {
            film_grain_params_present: seq.film_grain_params_present,
            show_frame,
            showable_frame,
            is_inter_frame: matches!(frame_type, FrameType::Inter),
            mono_chrome: seq.color_config.mono_chrome,
            subsampling_x: seq.color_config.subsampling_x,
            subsampling_y: seq.color_config.subsampling_y,
        },
    )?;
    Ok((
        reference_select,
        skip_mode_present,
        allow_warped_motion,
        reduced_tx_set,
        global_motion,
        film_grain,
    ))
}

/// `allow_intrabc` per §5.9.3 path of §5.9.2: read `f(1)` only when
/// `allow_screen_content_tools && UpscaledWidth == FrameWidth`.
/// Otherwise the §5.9.2 `allow_intrabc = 0` initialiser stands.
fn read_allow_intrabc(
    br: &mut BitReader<'_>,
    allow_screen_content_tools: bool,
    fs: &FrameSize,
) -> Result<bool, Error> {
    if allow_screen_content_tools && fs.upscaled_width == fs.frame_width {
        Ok(br.f(1)? == 1)
    } else {
        Ok(false)
    }
}

/// Read `frame_size()` + `render_size()` + `superres_params()` +
/// `compute_image_size()` in spec order (§5.9.5, §5.9.6, §5.9.8,
/// §5.9.9), returning the combined [`FrameSize`].
///
/// `frame_size_override_flag` is passed in because the §5.9.5
/// `if (frame_size_override_flag)` gate selects between reading
/// `frame_width_minus_1` / `frame_height_minus_1` (with bit widths
/// from §5.5.1's `frame_width_bits_minus_1` /
/// `frame_height_bits_minus_1`) and using the sequence header's
/// `max_frame_width_minus_1 + 1` / `max_frame_height_minus_1 + 1`.
///
/// The §5.9.7 `frame_size_with_refs()` `found_ref == 1` shortcut is
/// **not** handled here — callers that would have descended into
/// `frame_size_with_refs()` should return `FrameHeader::frame_size =
/// None` and stop bit accounting at `refresh_frame_flags`.
fn parse_frame_size_block(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    frame_size_override_flag: bool,
) -> Result<FrameSize, Error> {
    // §5.9.5 frame_size().
    let (frame_width_initial, frame_height) = if frame_size_override_flag {
        let n_w = u32::from(seq.frame_width_bits_minus_1) + 1;
        let frame_width_minus_1 = br.f(n_w)? as u32;
        let n_h = u32::from(seq.frame_height_bits_minus_1) + 1;
        let frame_height_minus_1 = br.f(n_h)? as u32;
        (frame_width_minus_1 + 1, frame_height_minus_1 + 1)
    } else {
        (
            seq.max_frame_width_minus_1 + 1,
            seq.max_frame_height_minus_1 + 1,
        )
    };

    // §5.9.5 calls superres_params() (§5.9.8) **before**
    // compute_image_size(), but in §5.9.2's `if (FrameIsIntra)` path
    // it's `frame_size(); render_size();` — i.e. frame_size() itself
    // includes both superres_params() and compute_image_size(), and
    // render_size() runs after compute_image_size(). The order is
    // therefore:
    //
    //   1. §5.9.5 frame_size: optionally read frame_width/height
    //   2. §5.9.8 superres_params: read use_superres + coded_denom,
    //      shuffle UpscaledWidth / FrameWidth.
    //   3. §5.9.9 compute_image_size: derive MiCols / MiRows from
    //      the post-superres FrameWidth.
    //   4. §5.9.6 render_size: read render_and_frame_size_different
    //      + optionally render_width_minus_1 / render_height_minus_1.
    //
    // The §5.9.6 default-RenderWidth uses UpscaledWidth (the
    // pre-downscale width), which is why superres_params has to
    // settle UpscaledWidth before render_size.

    // §5.9.8 superres_params().
    let (use_superres, coded_denom, superres_denom) = if seq.enable_superres {
        let use_superres = br.f(1)? == 1;
        if use_superres {
            let coded_denom = br.f(SUPERRES_DENOM_BITS)? as u8;
            (
                true,
                coded_denom,
                u32::from(coded_denom) + SUPERRES_DENOM_MIN,
            )
        } else {
            (false, 0u8, SUPERRES_NUM)
        }
    } else {
        // `use_superres = 0` is inferred when enable_superres is off.
        (false, 0u8, SUPERRES_NUM)
    };
    let upscaled_width = frame_width_initial;
    // FrameWidth = (UpscaledWidth * SUPERRES_NUM + (SuperresDenom /
    // 2)) / SuperresDenom — rounded-half-up integer divide.
    let frame_width = (upscaled_width * SUPERRES_NUM + (superres_denom / 2)) / superres_denom;

    // §5.9.9 compute_image_size().
    let mi_cols = 2 * ((frame_width + 7) >> 3);
    let mi_rows = 2 * ((frame_height + 7) >> 3);

    // §5.9.6 render_size().
    let render_and_frame_size_different = br.f(1)? == 1;
    let (render_width, render_height) = if render_and_frame_size_different {
        let rw = (br.f(16)? as u32) + 1;
        let rh = (br.f(16)? as u32) + 1;
        (rw, rh)
    } else {
        (upscaled_width, frame_height)
    };

    Ok(FrameSize {
        frame_width,
        frame_height,
        render_width,
        render_height,
        superres_denom,
        upscaled_width,
        mi_cols,
        mi_rows,
        use_superres,
        coded_denom,
        render_and_frame_size_different,
    })
}

/// `allow_screen_content_tools` per §5.9.2: read iff
/// `seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS`;
/// otherwise the value is the sequence header's force value.
fn read_allow_scc(br: &mut BitReader<'_>, seq: &SequenceHeader) -> Result<bool, Error> {
    if seq.seq_force_screen_content_tools == SELECT_SCREEN_CONTENT_TOOLS {
        Ok(br.f(1)? == 1)
    } else {
        Ok(seq.seq_force_screen_content_tools != 0)
    }
}

/// Read `force_integer_mv` from the bitstream when the §5.9.2 syntax
/// table asks for it, returning the bitstream-derived value (the
/// caller is responsible for the `FrameIsIntra` override).
fn consume_force_integer_mv_if_present(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    allow_screen_content_tools: bool,
) -> Result<bool, Error> {
    if allow_screen_content_tools {
        if seq.seq_force_integer_mv == SELECT_INTEGER_MV {
            Ok(br.f(1)? == 1)
        } else {
            Ok(seq.seq_force_integer_mv != 0)
        }
    } else {
        Ok(false)
    }
}

fn decoder_model_info_present(seq: &SequenceHeader) -> bool {
    seq.decoder_model_info_present_flag
}

/// `equal_picture_interval` lives inside the `timing_info` struct;
/// the §5.9.2 gate "`!equal_picture_interval`" only fires when timing
/// info is present (because `decoder_model_info_present_flag == 1`
/// requires `timing_info_present_flag == 1` per §5.5.1). For the
/// docs gap above, we conservatively treat "no timing info" as
/// "equal picture interval", i.e. the temporal-point-info branch is
/// not taken.
fn equal_picture_interval(seq: &SequenceHeader) -> bool {
    seq.timing_info
        .map(|t| t.equal_picture_interval)
        .unwrap_or(true)
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_header::parse_sequence_header;

    /// Build a sequence-header-like context by parsing the
    /// SEQ_HEADER payload from the tiny-i-only fixture and returning
    /// the parsed [`SequenceHeader`].
    fn tiny_seq() -> SequenceHeader {
        // Payload extracted from
        // docs/video/av1/fixtures/tiny-i-only-16x16-prof0/input.ivf
        // (SEQ_HEADER OBU payload).
        let seq_payload: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
        parse_sequence_header(seq_payload).expect("seq header parses")
    }

    fn show_existing_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x9b, 0x5f, 0x30, 0x08];
        parse_sequence_header(seq_payload).expect("show-existing seq header parses")
    }

    fn screen_content_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x18, 0x1d, 0xbf, 0xff, 0xf2, 0x01];
        parse_sequence_header(seq_payload).expect("screen-content seq header parses")
    }

    fn super_resolution_seq() -> SequenceHeader {
        let seq_payload: &[u8] = &[0x18, 0x19, 0x7f, 0xff, 0xf8, 0x04];
        parse_sequence_header(seq_payload).expect("super-res seq header parses")
    }

    // -------------------------------------------------------------
    // Fixture 1: tiny-i-only KEY frame (show_frame=1)
    // -------------------------------------------------------------

    /// FRAME OBU payload from `tiny-i-only-16x16-prof0`. Trace says:
    /// show_existing=0 frame_type=0(KEY) show_frame=1 showable=0
    /// error_resilient=1 disable_cdf_update=0 allow_screen_content=0
    /// force_integer_mv=0 (raw; FrameIsIntra forces to 1) order_hint=0
    /// primary_ref_frame=7 refresh_flags=0xff.
    const TINY_FRAME_PAYLOAD: &[u8] = &[
        0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67,
        0x6c, 0xc7, 0xee, 0x51, 0x80,
    ];

    #[test]
    fn parses_tiny_key_frame_prefix() {
        let seq = tiny_seq();
        let fh = parse_frame_header(TINY_FRAME_PAYLOAD, &seq).expect("frame header parses");
        assert!(!fh.show_existing_frame);
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(fh.frame_is_intra);
        assert!(fh.show_frame);
        assert!(!fh.showable_frame, "KEY + show_frame => showable=0");
        assert!(
            fh.error_resilient_mode,
            "KEY + show_frame => error_resilient forced to 1"
        );
        assert!(!fh.disable_cdf_update);
        assert!(!fh.allow_screen_content_tools);
        // FrameIsIntra path forces force_integer_mv = true.
        assert!(fh.force_integer_mv);
        assert_eq!(fh.current_frame_id, 0);
        assert!(!fh.frame_size_override_flag);
        assert_eq!(fh.order_hint, 0);
        assert_eq!(fh.primary_ref_frame, PRIMARY_REF_NONE);
        assert_eq!(fh.refresh_frame_flags, 0xff);
        // The §5.9.2 leading block reads (in this fixture):
        //   show_existing(1) frame_type(2) show_frame(1)
        //   [showable+error_resilient derived for KEY+show_frame]
        //   disable_cdf_update(1) allow_scc(1)
        //   [force_integer_mv not read: allow_scc=0]
        //   [current_frame_id not read: frame_id_numbers_present_flag=0]
        //   frame_size_override_flag(1) order_hint(7)
        //   [primary_ref_frame derived: FrameIsIntra]
        //   [refresh_frame_flags derived: KEY && show_frame]
        // = 14 bits. Round 4 then continues into §5.9.5–§5.9.9:
        //   [frame_size_override=0 ⇒ no n_w / n_h reads]
        //   [enable_superres=0 ⇒ no use_superres / coded_denom reads]
        //   render_and_frame_size_different(1)=0
        //   [render_width / render_height not read]
        // = +1 bit ⇒ 15 bits. Round 6 adds:
        //   [allow_intrabc not read: allow_scc=0]
        //   disable_frame_end_update_cdf(1)=0
        //   uniform_tile_spacing_flag(1)=1
        //   [16x16 + use_128sb=1 ⇒ sbCols=sbRows=1 ⇒ TileColsLog2 /
        //    TileRowsLog2 both saturate at 0 ⇒ no increment reads]
        //   [TileColsLog2==0 && TileRowsLog2==0 ⇒ no
        //    context_update_tile_id / tile_size_bytes_minus_1 reads]
        // = +2 bits ⇒ 17 bits. Round 7 adds:
        //   quantization_params: base_q_idx(8)=120,
        //     DeltaQYDc.delta_coded(1)=0, DeltaQUDc.delta_coded(1)=0,
        //     DeltaQUAc.delta_coded(1)=0,
        //     [V mirrors U: no read], using_qmatrix(1)=0
        //   = +12 bits.
        //   segmentation_params: seg_enabled(1)=0 ⇒ +1 bit, no further reads.
        // = 30 bits. Round 8 adds:
        //   delta_q_params: base_q_idx(120) > 0 ⇒ delta_q_present(1)=0
        //     read ⇒ +1 bit; delta_q_present=0 ⇒ no delta_q_res read.
        //   delta_lf_params: delta_q_present=0 ⇒ whole block skipped,
        //     no bits read.
        // = 31 bits. Round 9 adds loop_filter_params (§5.9.11):
        //   [base_q_idx=120, no delta_q offsets, seg disabled ⇒
        //    CodedLossless=0 ⇒ full path, NOT short-circuit]
        //   loop_filter_level[0](6)=0  loop_filter_level[1](6)=0
        //   [level[0]==0 && level[1]==0 ⇒ no chroma level[2]/[3] reads]
        //   loop_filter_sharpness(3)=0  loop_filter_delta_enabled(1)=1
        //   loop_filter_delta_update(1)=0
        //   [delta_update=0 ⇒ no ref/mode-delta update walk]
        //   = +17 bits.
        // = 48 bits. Round 10 adds cdef_params (§5.9.19):
        //   [enable_cdef=1, CodedLossless=0, allow_intrabc=0 ⇒ full path]
        //   cdef_damping_minus_3(2)=1  cdef_bits(2)=0
        //   entry 0 (NumPlanes=3): cdef_y_pri_strength(4)=0
        //     cdef_y_sec_strength(2)=0  cdef_uv_pri_strength(4)=0
        //     cdef_uv_sec_strength(2)=0
        //   = +16 bits.
        // = 64 bits. Round 11 adds lr_params (§5.9.20):
        //   [AllLossless=0 (CodedLossless=0), allow_intrabc=0,
        //    enable_restoration=1 ⇒ full path]
        //   NumPlanes=3 ⇒ lr_type[0..3] each f(2): 6 bits. Trace shows
        //   y=0/u=0/v=0 ⇒ UsesLr=0 ⇒ no unit_shift / uv_shift bits.
        //   = +6 bits.
        // = 70 bits. Round 12 adds read_tx_mode (§5.9.21):
        //   [CodedLossless=0 ⇒ tx_mode_select(1) read; trace tx_mode=1
        //    ⇒ TX_MODE_LARGEST ⇒ tx_mode_select=0]
        //   = +1 bit.
        // = 71 bits. Round 13 adds the §5.9.2 tail after read_tx_mode():
        //   frame_reference_mode() [intra ⇒ reference_select=0, no bits]
        //   skip_mode_params()     [intra ⇒ skip_mode_present=0, no bits]
        //   allow_warped_motion    [intra guard ⇒ 0, no bits]
        //   reduced_tx_set(1)=0    [trace reduced_tx_set=0] ⇒ +1 bit
        //   global_motion_params() [intra ⇒ identity, no bits]
        //   film_grain_params()    [film_grain_params_present=0 ⇒
        //                           reset_grain_params, no bits]
        //   = +1 bit.
        // = 72 bits total.
        assert_eq!(fh.bits_consumed, 72);
        let qp = fh
            .quantization_params
            .as_ref()
            .expect("intra frame produces quantization_params");
        assert_eq!(qp.base_q_idx, 120, "trace base_q_idx=120");
        assert_eq!(qp.delta_q_y_dc, 0);
        assert_eq!(qp.delta_q_u_dc, 0);
        assert_eq!(qp.delta_q_u_ac, 0);
        assert!(!qp.using_qmatrix);
        let sp = fh
            .segmentation_params
            .as_ref()
            .expect("intra frame produces segmentation_params");
        assert!(!sp.enabled);
        assert!(!sp.update_map);
        assert!(!sp.temporal_update);
        assert!(!sp.update_data);
        assert!(!sp.seg_id_pre_skip);
        assert_eq!(sp.last_active_seg_id, 0);
        let dq = fh
            .delta_q_params
            .as_ref()
            .expect("intra frame produces delta_q_params");
        // base_q_idx=120 > 0 ⇒ delta_q_present read, trace says 0.
        assert!(!dq.delta_q_present);
        assert_eq!(dq.delta_q_res, 0);
        let dlf = fh
            .delta_lf_params
            .as_ref()
            .expect("intra frame produces delta_lf_params");
        // delta_q_present=0 ⇒ whole delta_lf_params block is a no-op.
        assert!(!dlf.delta_lf_present);
        assert_eq!(dlf.delta_lf_res, 0);
        assert!(!dlf.delta_lf_multi);
        let lf = fh
            .loop_filter_params
            .as_ref()
            .expect("intra frame produces loop_filter_params");
        // base_q_idx=120, no delta offsets, seg disabled ⇒
        // CodedLossless=0 ⇒ §5.9.11 full path (NOT short-circuit).
        assert!(!lf.short_circuited);
        assert_eq!(lf.loop_filter_level, [0, 0, 0, 0]);
        assert_eq!(lf.loop_filter_sharpness, 0);
        assert!(lf.loop_filter_delta_enabled);
        assert!(!lf.loop_filter_delta_update);
        let cdef = fh
            .cdef_params
            .as_ref()
            .expect("intra frame produces cdef_params");
        // enable_cdef=1, CodedLossless=0 ⇒ §5.9.19 full path. Trace
        // CDEF idx=0: damping=4, bits=0, all strengths 0.
        assert!(!cdef.short_circuited);
        assert_eq!(cdef.cdef_damping, 4);
        assert_eq!(cdef.cdef_bits, 0);
        assert_eq!(cdef.cdef_y_pri_strength[0], 0);
        assert_eq!(cdef.cdef_y_sec_strength[0], 0);
        assert_eq!(cdef.cdef_uv_pri_strength[0], 0);
        assert_eq!(cdef.cdef_uv_sec_strength[0], 0);
        let tx = fh.tx_mode.expect("intra frame produces tx_mode");
        // CodedLossless=0 ⇒ §5.9.21 reads tx_mode_select; trace tx_mode=1
        // ⇒ TX_MODE_LARGEST.
        assert_eq!(tx, TxMode::TxModeLargest);
        // Round 13 §5.9.2 tail: intra ⇒ reference_select / skip_mode /
        // allow_warped_motion all 0 (no bits); reduced_tx_set is one
        // bit (trace=0); global_motion identity; film grain reset.
        assert_eq!(fh.reference_select, Some(false));
        assert_eq!(fh.skip_mode_present, Some(false));
        assert_eq!(fh.allow_warped_motion, Some(false));
        assert_eq!(fh.reduced_tx_set, Some(false));
        let gm = fh
            .global_motion_params
            .as_ref()
            .expect("intra frame produces global_motion_params");
        assert!(gm.short_circuited);
        assert_eq!(gm, &GlobalMotionParams::identity());
        let fg = fh
            .film_grain_params
            .as_ref()
            .expect("intra frame produces film_grain_params");
        // tiny-i-only seq has film_grain_params_present=0 ⇒ reset.
        assert!(!fg.apply_grain);
        assert_eq!(fg, &FilmGrainParams::reset());
        let ti = fh.tile_info.as_ref().expect("intra frame has tile_info");
        assert!(ti.uniform_tile_spacing_flag);
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(ti.context_update_tile_id, 0);
        assert!(!fh.allow_intrabc);
        assert!(!fh.disable_frame_end_update_cdf);
        let fs = fh.frame_size.expect("intra frame produces frame_size");
        assert_eq!(fs.frame_width, 16);
        assert_eq!(fs.frame_height, 16);
        assert_eq!(fs.upscaled_width, 16);
        assert_eq!(fs.render_width, 16);
        assert_eq!(fs.render_height, 16);
        assert_eq!(fs.superres_denom, SUPERRES_NUM);
        // MiCols = 2 * ((16 + 7) >> 3) = 2 * (23 >> 3) = 2 * 2 = 4.
        assert_eq!(fs.mi_cols, 4);
        assert_eq!(fs.mi_rows, 4);
        assert!(!fs.use_superres);
        assert_eq!(fs.coded_denom, 0);
        assert!(!fs.render_and_frame_size_different);
        assert!(!fs.is_super_resolved());
    }

    // -------------------------------------------------------------
    // Fixture 2: show-existing-frame, first frame (KEY, show_frame=0,
    // allow_scc=1)
    // -------------------------------------------------------------

    /// First FRAME OBU payload from `show-existing-frame/input.ivf`,
    /// the actual KEY frame that is later replayed. Trace:
    /// show_existing=0 frame_type=0 show_frame=0 showable=0
    /// error_resilient=0 disable_cdf=0 allow_scc=1 force_integer_mv=0
    /// (raw; FrameIsIntra forces 1) order_hint=0 primary_ref=7
    /// refresh_flags=0xff.
    const SHOW_EXISTING_KEY_PAYLOAD: &[u8] = &[
        0x01, 0x00, 0x7f, 0x89, 0x10, 0x02, 0x08, 0x10, 0x82, 0x00, 0x0a, 0x02, 0xdc, 0x85, 0x28,
        0xf5,
    ];

    #[test]
    fn parses_show_existing_underlying_key_frame() {
        let seq = show_existing_seq();
        let fh = parse_frame_header(SHOW_EXISTING_KEY_PAYLOAD, &seq).expect("frame header parses");
        assert!(!fh.show_existing_frame);
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(!fh.show_frame);
        assert!(!fh.showable_frame);
        assert!(!fh.error_resilient_mode, "KEY + show_frame=0 reads er_mode");
        assert!(fh.allow_screen_content_tools);
        // FrameIsIntra still forces force_integer_mv to 1 even though
        // the raw bitstream bit was 0.
        assert!(fh.force_integer_mv);
        assert_eq!(fh.order_hint, 0);
        assert_eq!(fh.primary_ref_frame, PRIMARY_REF_NONE);
        assert_eq!(fh.refresh_frame_flags, 0xff);
        // 64x64 still picture, enable_superres=0, no override (trace
        // `w=64`), no render-size difference.
        let fs = fh.frame_size.expect("intra frame produces frame_size");
        assert_eq!(fs.frame_width, 64);
        assert_eq!(fs.frame_height, 64);
        assert_eq!(fs.upscaled_width, 64);
        assert_eq!(fs.render_width, 64);
        assert_eq!(fs.render_height, 64);
        assert_eq!(fs.superres_denom, SUPERRES_NUM);
        // MiCols = 2 * ((64 + 7) >> 3) = 2 * 8 = 16.
        assert_eq!(fs.mi_cols, 16);
        assert_eq!(fs.mi_rows, 16);
        assert!(!fs.use_superres);
        // allow_scc=1, UpscaledWidth=FrameWidth=64 ⇒ allow_intrabc IS read.
        // Trace says allow_intrabc=0.
        assert!(!fh.allow_intrabc);
        // 64x64 use_128sb=0: sbCols=sbRows=1 ⇒ single tile (uniform-
        // spacing-flag bit value doesn't change the semantic result).
        let ti = fh.tile_info.as_ref().expect("intra frame has tile_info");
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
    }

    // -------------------------------------------------------------
    // Synthetic: minimal show-existing-frame replay
    // -------------------------------------------------------------

    #[test]
    fn synthetic_show_existing_frame_replay() {
        // Construct an uncompressed_header() against the tiny seq
        // (frame_id_numbers_present_flag = 0). Bits:
        //   show_existing(1)=1
        //   frame_to_show_map_idx(3)=5
        // = 4 bits = 0b1101 = 1101_0000 = 0xD0 (high nibble).
        let payload = [0b1101_0000u8];
        let seq = tiny_seq();
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert!(fh.show_existing_frame);
        assert_eq!(fh.frame_to_show_map_idx, Some(5));
        assert!(fh.display_frame_id.is_none());
        assert_eq!(fh.refresh_frame_flags, 0);
        // Show-existing-frame returns early after reading 4 bits.
        assert_eq!(fh.bits_consumed, 4);
    }

    // -------------------------------------------------------------
    // Synthetic: reduced-still-picture-header path
    // -------------------------------------------------------------

    #[test]
    fn reduced_still_picture_path() {
        // Use the screen-content-tools SEQ (which is reduced-still,
        // 256x128, enable_superres=0).
        let seq = screen_content_seq();
        assert!(seq.reduced_still_picture_header);
        assert_eq!(
            seq.seq_force_screen_content_tools, SELECT_SCREEN_CONTENT_TOOLS,
            "reduced still forces SELECT"
        );
        assert!(!seq.enable_superres);
        // Bits we need to consume:
        //   disable_cdf_update(1)=0
        //   allow_screen_content_tools(1)=1     [seq force is SELECT]
        //   force_integer_mv(1)=0               [seq force is SELECT]
        //   [frame_size_override=0 derived; no n_w/n_h reads]
        //   [enable_superres=0; no use_superres/coded_denom reads]
        //   render_and_frame_size_different(1)=0
        //   allow_intrabc(1)=0       [allow_scc=1, UpscaledWidth==FrameWidth]
        //   uniform_tile_spacing_flag(1)=1
        //   increment_tile_cols_log2(1)=0   [stop: 256x128 use_128sb=1 ⇒
        //                                    sbCols=2, maxLog2TileCols=1;
        //                                    TileColsLog2 stays at 0]
        //   [no row-increment reads: maxLog2TileRows=0]
        //   [no context_update_tile_id / tile_size_bytes reads:
        //    TileColsLog2==0 && TileRowsLog2==0]
        // = 7 bits = 0b0100_0100 = 0x44. Round 7 appends:
        //   base_q_idx(8) = 0, three delta_q delta_coded(1) = 0 each,
        //   using_qmatrix(1) = 0, seg_enabled(1) = 0 ⇒ +13 zero bits.
        // = 20 bits. Round 13 appends the §5.9.2 tail: reduced_tx_set(1)
        //   = 0 (the only bit read on the intra tail; film_grain is reset
        //   because this seq has film_grain_params_present=0).
        // Total = 21 bits = 0b0100_0100 0000_0000 0000_0000 = three
        // zero-padded bytes. 0x44, 0x00, 0x00.
        let payload = [0x44u8, 0x00, 0x00];
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(fh.frame_is_intra);
        assert!(fh.show_frame);
        assert!(!fh.showable_frame);
        assert!(fh.error_resilient_mode);
        assert!(!fh.disable_cdf_update);
        assert!(fh.allow_screen_content_tools);
        // FrameIsIntra forces force_integer_mv=1 even though the bit
        // we wrote was 0.
        assert!(fh.force_integer_mv);
        assert_eq!(fh.refresh_frame_flags, 0xff);
        assert_eq!(fh.primary_ref_frame, PRIMARY_REF_NONE);
        assert!(!fh.allow_intrabc);
        assert!(!fh.reduced_tx_set.expect("intra has reduced_tx_set"));
        assert!(
            fh.global_motion_params
                .as_ref()
                .expect("intra has global_motion")
                .short_circuited
        );
        assert!(
            !fh.film_grain_params
                .as_ref()
                .expect("intra has film_grain")
                .apply_grain
        );
        assert_eq!(fh.bits_consumed, 21);
        let qp = fh.quantization_params.as_ref().expect("intra has qp");
        assert_eq!(qp.base_q_idx, 0);
        assert!(!qp.using_qmatrix);
        let sp = fh.segmentation_params.as_ref().expect("intra has sp");
        assert!(!sp.enabled);
        let fs = fh.frame_size.expect("reduced-still produces frame_size");
        assert_eq!(fs.frame_width, 256);
        assert_eq!(fs.frame_height, 128);
        assert_eq!(fs.upscaled_width, 256);
        // MiCols = 2 * ((256+7) >> 3) = 2 * 32 = 64.
        assert_eq!(fs.mi_cols, 64);
        // MiRows = 2 * ((128+7) >> 3) = 2 * 16 = 32.
        assert_eq!(fs.mi_rows, 32);
        assert!(!fs.use_superres);
        assert!(!fs.render_and_frame_size_different);
        assert_eq!(fs.render_width, 256);
        assert_eq!(fs.render_height, 128);
        let ti = fh.tile_info.as_ref().expect("intra frame has tile_info");
        assert!(ti.uniform_tile_spacing_flag);
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
    }

    // -------------------------------------------------------------
    // Synthetic: reduced-still without SELECT for force_integer_mv
    // -------------------------------------------------------------

    #[test]
    fn reduced_still_without_select_force_int_mv() {
        // super_resolution fixture is reduced-still with allow_scc
        // path through SELECT. Same shape as screen-content above
        // but with enable_superres=1 (so superres_params reads one
        // more bit).
        let seq = super_resolution_seq();
        assert!(seq.reduced_still_picture_header);
        assert!(seq.enable_superres);
        // disable_cdf=0, allow_scc=1, force_int_mv=0, use_superres=0,
        // render_and_frame_size_different=0 = 5 bits.
        // Round 6 adds:
        //   allow_intrabc(1)=0   [allow_scc=1, UpscaledWidth=128==FrameWidth=128]
        //   uniform_tile_spacing_flag(1)=1
        //   [128x64 use_128sb=1 ⇒ sbCols=sbRows=1 ⇒ no
        //    increment / context_update_tile_id reads]
        // = 7 bits. Round 7 appends 13 zero bits for quant +
        // segmentation (same as the screen-content reduced-still test
        // above). = 20 bits. Round 13 appends reduced_tx_set(1)=0
        // (film_grain reset: film_grain_params_present=0). Total = 21
        // bits. 0x42 0x00 0x00.
        let payload = [0x42u8, 0x00, 0x00];
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert!(fh.allow_screen_content_tools);
        assert!(fh.force_integer_mv);
        let fs = fh.frame_size.expect("reduced-still produces frame_size");
        // super_resolution_seq has max_w=128 max_h=64. With
        // use_superres=0 in this synthetic, SuperresDenom = SUPERRES_NUM
        // and FrameWidth = UpscaledWidth = 128.
        assert_eq!(fs.frame_width, 128);
        assert_eq!(fs.upscaled_width, 128);
        assert_eq!(fs.frame_height, 64);
        assert!(!fs.use_superres);
        assert_eq!(fs.superres_denom, SUPERRES_NUM);
        assert!(!fh.allow_intrabc);
        assert!(!fh.reduced_tx_set.expect("intra has reduced_tx_set"));
        assert_eq!(fh.bits_consumed, 21);
        let sp = fh.segmentation_params.as_ref().expect("intra has sp");
        assert!(!sp.enabled);
        let ti = fh.tile_info.as_ref().expect("intra frame has tile_info");
        assert!(ti.uniform_tile_spacing_flag);
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
    }

    // -------------------------------------------------------------
    // Synthetic: reduced-still with use_superres = 1 and a non-zero
    // coded_denom. Exercises the §5.9.8 read + the downscale math.
    // -------------------------------------------------------------

    #[test]
    fn reduced_still_with_use_superres() {
        // super_resolution_seq: max_w=128, max_h=64, enable_superres=1.
        // Bits (reduced-still + KEY+show_frame path):
        //   bit 0 = 0 (disable_cdf_update)
        //   bit 1 = 1 (allow_screen_content_tools; seq force = SELECT)
        //   bit 2 = 0 (force_integer_mv raw; FrameIsIntra ⇒ 1)
        //   [frame_size_override = 0 derived]
        //   bit 3 = 1 (use_superres)
        //   bits 4..6 = coded_denom(3) MSB-first = 011 (= 3 ⇒
        //                                       SuperresDenom = 12)
        //   bit 7 = 0 (render_and_frame_size_different)
        //   Round 6 adds:
        //   [allow_intrabc not read: UpscaledWidth=128 != FrameWidth=85]
        //   bit 8 = 1 (uniform_tile_spacing_flag)
        //   [85x64 use_128sb=1: sbCols=sbRows=1 ⇒ no increment reads;
        //    TileColsLog2==0 && TileRowsLog2==0 ⇒ no trailing reads]
        // = 9 bits. Round 7 appends 13 zero bits for quant + seg.
        // = 22 bits. Round 10 adds cdef_params (§5.9.19): base_q_idx=0
        // and no delta_q offsets ⇒ CodedLossless=1 ⇒ §5.9.19
        // short-circuit (0 bits). Round 11 adds lr_params (§5.9.20):
        // AllLossless = CodedLossless && (FrameWidth == UpscaledWidth)
        // = true && (85 == 128) = false ⇒ NOT short-circuit; full path
        // reads NumPlanes=3 lr_type f(2) ⇒ +6 zero bits ⇒ all NONE ⇒
        // UsesLr=0 ⇒ no shift bits. = 28 bits. read_tx_mode (§5.9.21):
        // CodedLossless=1 ⇒ ONLY_4X4, no bits. Round 13 appends
        // reduced_tx_set(1)=0 (film_grain reset: film_grain_params_present
        // =0). Total = 29 bits. The trailing 3 bits are padding inside
        // the 4-byte buffer.
        let payload = [0x56u8, 0x80u8, 0x00u8, 0x00u8];
        let seq = super_resolution_seq();
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert_eq!(fh.bits_consumed, 29);
        let fs = fh.frame_size.expect("reduced-still produces frame_size");
        assert!(fs.use_superres);
        assert_eq!(fs.coded_denom, 3);
        assert_eq!(fs.superres_denom, 12);
        assert_eq!(fs.upscaled_width, 128);
        // FrameWidth = (128 * 8 + 12/2) / 12 = (1024 + 6) / 12 = 1030/12 = 85.
        assert_eq!(fs.frame_width, 85);
        assert_eq!(fs.frame_height, 64);
        // MiCols = 2 * ((85+7) >> 3) = 2 * (92 >> 3) = 2 * 11 = 22.
        assert_eq!(fs.mi_cols, 22);
        // MiRows = 2 * ((64+7) >> 3) = 2 * 8 = 16.
        assert_eq!(fs.mi_rows, 16);
        // §5.9.6 default: RenderWidth = UpscaledWidth (the pre-downscale
        // width), RenderHeight = FrameHeight.
        assert_eq!(fs.render_width, 128);
        assert_eq!(fs.render_height, 64);
        assert!(fs.is_super_resolved());
        // UpscaledWidth=128 != FrameWidth=85 ⇒ allow_intrabc gate fails,
        // §5.9.2 initialiser stands.
        assert!(!fh.allow_intrabc);
        let ti = fh.tile_info.as_ref().expect("intra frame has tile_info");
        assert!(ti.uniform_tile_spacing_flag);
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        // base_q_idx=0 ⇒ CodedLossless=1 ⇒ §5.9.21 ONLY_4X4, no bits read.
        assert_eq!(fh.tx_mode, Some(TxMode::Only4x4));
    }

    // -------------------------------------------------------------
    // Synthetic: render_size with explicit different render dims.
    // -------------------------------------------------------------

    #[test]
    fn explicit_render_and_frame_size_different() {
        // tiny_seq: 16x16, no superres. Bits to write (KEY +
        // show_frame=1 path):
        //   show_existing(1)=0
        //   frame_type(2)=00 (KEY)
        //   show_frame(1)=1
        //   [showable derived]
        //   [er_mode derived]
        //   disable_cdf_update(1)=0
        //   allow_screen_content_tools(1)=0  [seq is SELECT, but we
        //                                     write 0 to keep test
        //                                     consistent with tiny
        //                                     fixture]
        //   [force_integer_mv not read; allow_scc=0]
        //   [current_frame_id not read; frame_id_numbers_present=0]
        //   frame_size_override_flag(1)=0
        //   order_hint(7)=0
        //   [primary_ref derived]
        //   [refresh_frame_flags derived = allFrames]
        //   render_and_frame_size_different(1)=1
        //   render_width_minus_1(16)=39  ⇒ render_width = 40
        //   render_height_minus_1(16)=29 ⇒ render_height = 30
        //   [allow_intrabc not read: allow_scc=0]
        //   disable_frame_end_update_cdf(1)=0
        //   uniform_tile_spacing_flag(1)=1   [tile_info on 16x16]
        // Total bits = 1+2+1+1+1+1+7+1+16+16+1+1 = 49.
        let seq = tiny_seq();
        // Build bit-stream by writing bit-by-bit then packing.
        let bits: &[u8] = &[
            0, // show_existing
            0, 0, // frame_type=00 (KEY)
            1, // show_frame
            0, // disable_cdf_update
            0, // allow_screen_content_tools
            0, // frame_size_override_flag
            // order_hint(7) = 0 ⇒ seven zeros
            0, 0, 0, 0, 0, 0, 0, // render_and_frame_size_different
            1, // render_width_minus_1(16) = 39 = 0000_0000_0010_0111
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1,
            // render_height_minus_1(16) = 29 = 0000_0000_0001_1101
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
            // disable_frame_end_update_cdf = 0
            0, // uniform_tile_spacing_flag = 1
            1, // Round 7 trailing zero bits:
            //   quantization_params: base_q_idx(8)=0, 3× delta_coded(1)=0,
            //     using_qmatrix(1)=0 = 12 bits.
            //   segmentation_params: seg_enabled(1)=0 = 1 bit.
            // = +13 zero bits.
            0, 0, 0, 0, 0, 0, 0, 0, // base_q_idx = 0
            0, 0, 0, // three delta_coded gates
            0, // using_qmatrix
            0, // seg_enabled
            // base_q_idx=0, no deltas, seg disabled ⇒ CodedLossless=1 ⇒
            // delta_q (base_q_idx==0, no read), loop_filter / cdef / lr /
            // read_tx_mode all short-circuit (0 bits). Round 13 reads
            // reduced_tx_set(1)=0 (film_grain reset:
            // film_grain_params_present=0 in tiny_seq).
            0, // reduced_tx_set
        ];
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        let fs = fh.frame_size.expect("intra frame produces frame_size");
        assert!(fs.render_and_frame_size_different);
        assert_eq!(fs.render_width, 40);
        assert_eq!(fs.render_height, 30);
        assert_eq!(fs.frame_width, 16);
        assert_eq!(fs.frame_height, 16);
        assert!(!fh.reduced_tx_set.expect("intra has reduced_tx_set"));
        assert_eq!(fh.bits_consumed, bits.len());
    }

    // -------------------------------------------------------------
    // Synthetic: frame_size_override_flag = 1 reads explicit
    // frame_width_minus_1 / frame_height_minus_1.
    // -------------------------------------------------------------

    #[test]
    fn frame_size_override_reads_explicit_dims() {
        // tiny_seq: max_w=16 max_h=16. With override=1 we read
        // frame_width_bits_minus_1+1 = frame_width_bits_minus_1+1
        // bits. For tiny_seq the bit widths are tied to the
        // 16x16 sequence — frame_width_bits_minus_1 is the bits to
        // encode "16-1=15" → 4 bits. So the parser will read 4 bits
        // for width and 4 bits for height.
        let seq = tiny_seq();
        let n_w = u32::from(seq.frame_width_bits_minus_1) + 1;
        let n_h = u32::from(seq.frame_height_bits_minus_1) + 1;
        assert!((1..=16).contains(&n_w));
        assert!((1..=16).contains(&n_h));
        // We'll target frame_width=12, frame_height=10.
        let target_w = 12u32;
        let target_h = 10u32;
        let w_minus_1 = target_w - 1; // 11
        let h_minus_1 = target_h - 1; // 9
        let mut bits = vec![
            0u8, // show_existing
            0, 0, // frame_type=KEY
            1, // show_frame
            0, // disable_cdf_update
            0, // allow_scc (seq forces SELECT but we write 0)
            1, // frame_size_override_flag = 1
            // order_hint(7) = 0
            0, 0, 0, 0, 0, 0, 0,
        ];
        // frame_width_minus_1(n_w) MSB first.
        for i in (0..n_w).rev() {
            bits.push(((w_minus_1 >> i) & 1) as u8);
        }
        // frame_height_minus_1(n_h) MSB first.
        for i in (0..n_h).rev() {
            bits.push(((h_minus_1 >> i) & 1) as u8);
        }
        // render_and_frame_size_different = 0
        bits.push(0);
        // [allow_intrabc not read: allow_scc=0]
        // disable_frame_end_update_cdf = 0
        bits.push(0);
        // uniform_tile_spacing_flag = 1; 12x10 with use_128sb=1 ⇒
        // MiCols=MiRows=4 ⇒ sbCols=sbRows=1 ⇒ no further reads.
        bits.push(1);
        // Round 7: quant + segmentation trailing block — all zero bits.
        //   base_q_idx(8)=0, 3× delta_coded(1)=0, using_qmatrix(1)=0
        //   = 12 zero bits for quantization_params.
        //   seg_enabled(1)=0 = 1 zero bit for segmentation_params.
        bits.resize(bits.len() + 13, 0);
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert!(fh.frame_size_override_flag);
        let fs = fh.frame_size.expect("intra frame produces frame_size");
        assert_eq!(fs.frame_width, target_w);
        assert_eq!(fs.frame_height, target_h);
        // RenderWidth = UpscaledWidth = pre-superres FrameWidth = 12.
        assert_eq!(fs.render_width, 12);
        assert_eq!(fs.render_height, 10);
        let ti = fh.tile_info.as_ref().expect("intra has tile_info");
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
    }

    // -------------------------------------------------------------
    // FrameType enum unit
    // -------------------------------------------------------------

    #[test]
    fn frame_type_roundtrip() {
        for raw in 0u8..4u8 {
            let ft = FrameType::from_raw(raw);
            assert_eq!(ft.as_raw(), raw);
        }
        assert!(FrameType::Key.is_intra());
        assert!(!FrameType::Inter.is_intra());
        assert!(FrameType::IntraOnly.is_intra());
        assert!(!FrameType::Switch.is_intra());
    }

    // -------------------------------------------------------------
    // Error paths
    // -------------------------------------------------------------

    #[test]
    fn unexpected_end_on_truncated_payload() {
        let seq = tiny_seq();
        let payload: [u8; 0] = [];
        let err = parse_frame_header(&payload, &seq).expect_err("must error");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -------------------------------------------------------------
    // Round 6: synthetic allow_intrabc = 1 path
    // -------------------------------------------------------------

    /// `allow_intrabc` is gated by
    /// `allow_screen_content_tools && UpscaledWidth == FrameWidth`.
    /// Use the screen-content-tools (reduced-still) sequence which
    /// has `seq_force_screen_content_tools = SELECT` and
    /// `enable_superres = 0`, then write `allow_intrabc = 1` into the
    /// bitstream.
    #[test]
    fn allow_intrabc_read_when_gated() {
        let seq = screen_content_seq();
        // Bits (reduced-still + KEY+show_frame path):
        //   disable_cdf_update(1)=0
        //   allow_screen_content_tools(1)=1
        //   force_integer_mv(1)=0
        //   render_and_frame_size_different(1)=0
        //   allow_intrabc(1)=1
        //   [disable_frame_end_update_cdf derived 1 in reduced-still]
        //   uniform_tile_spacing_flag(1)=1
        //   [256x128 use_128sb=1 ⇒ sbCols=2, maxLog2TileCols=1.
        //    Need increment_tile_cols_log2(1)=0 to stop at TileColsLog2=0
        //    (single tile column).]
        // = 7 bits. 0b0100_1010 = 0x4A. Round 7: +13 zero bits for quant +
        // segmentation ⇒ 0x4A, 0x00, 0x00.
        let payload = [0x4Au8, 0x00, 0x00];
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        assert!(fh.allow_intrabc);
        assert!(fh.disable_frame_end_update_cdf, "reduced-still derives 1");
        let ti = fh.tile_info.as_ref().expect("intra has tile_info");
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
    }

    // -------------------------------------------------------------
    // Round 6: synthetic with TileColsLog2 > 0 ⇒
    // context_update_tile_id read.
    // -------------------------------------------------------------

    /// Build a reduced-still bitstream that lands on TileColsLog2=1
    /// with a non-zero `context_update_tile_id`.
    #[test]
    fn context_update_tile_id_read_when_log2_nonzero() {
        let seq = screen_content_seq();
        // Bits:
        //   disable_cdf_update(1)=0
        //   allow_screen_content_tools(1)=1
        //   force_integer_mv(1)=0
        //   render_and_frame_size_different(1)=0
        //   allow_intrabc(1)=0
        //   uniform_tile_spacing_flag(1)=1
        //   increment_tile_cols_log2(1)=1  ⇒ TileColsLog2 = 1
        //   [maxLog2TileCols=1, loop exits]
        //   [TileRowsLog2 stays at 0]
        //   context_update_tile_id(1)=1  [TileRowsLog2 + TileColsLog2 = 1]
        //   tile_size_bytes_minus_1(2)=10 ⇒ TileSizeBytes = 3.
        // = 10 bits. Round 7: +13 zero bits for quant + segmentation
        // ⇒ 23 total bits, padded with zeros. Layout:
        //   0 1 0 0 0 1 1 1 | 1 0 0 0 0 0 0 0 | 0 0 0 0 0 0 0 0 0
        // = 0x47 0x80 0x00.
        let payload = [0x47u8, 0x80u8, 0x00u8];
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        let ti = fh.tile_info.as_ref().expect("intra has tile_info");
        assert_eq!(ti.tile_cols, 2);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(ti.tile_cols_log2, 1);
        assert_eq!(ti.context_update_tile_id, 1);
        assert_eq!(ti.tile_size_bytes, 3);
    }

    // -------------------------------------------------------------
    // Round 7: streaming-parser synthetic with segmentation_enabled = 1
    // and a single active per-segment feature.
    // -------------------------------------------------------------

    /// Build a reduced-still bitstream (screen-content seq) that walks
    /// past `tile_info()` with `tile_cols = tile_rows = 1` and then
    /// activates `SEG_LVL_ALT_Q` on segment 0 with a non-zero signed
    /// `feature_value`.
    ///
    /// `primary_ref_frame == PRIMARY_REF_NONE` (reduced-still KEY) so
    /// segmentation_params collapses the three update flags to fixed
    /// values (`update_map=1`, `temporal_update=0`, `update_data=1`),
    /// and the per-segment per-feature inner loop walks 8×8 = 64
    /// feature_enabled bits plus the `su(9)` payload for the one
    /// active feature.
    #[test]
    fn segmentation_streaming_synthetic_alt_q_active() {
        use crate::uncompressed_header_tail::{
            MAX_SEGMENTS, SEG_LVL_ALT_Q, SEG_LVL_MAX, SEG_LVL_REF_FRAME,
        };
        let seq = screen_content_seq();
        // Pre-segmentation bits: same as the reduced_still_picture_path
        // test up through the tile_info() block + the trailing
        // quantization_params (12 zero bits). Then the §5.9.14 walk
        // reads its own bits: 1 (seg_enabled) + 64 (feature_enabled)
        // + 9 (su for the one active feature).
        let mut bits: Vec<u8> = vec![
            0, 1, 0, // disable_cdf_update / allow_scc / force_int_mv
            0, // render_and_frame_size_different
            0, // allow_intrabc
            // disable_frame_end_update_cdf is derived (reduced-still ⇒ 1, no bit)
            1, // uniform_tile_spacing_flag
            0, // increment_tile_cols_log2 (stops at TileColsLog2 = 0)
            // increment_tile_rows_log2 not read (maxLog2TileRows = 0)
            // no context_update_tile_id / tile_size_bytes reads.
            // base_q_idx(8) = 0
            0, 0, 0, 0, 0, 0, 0, 0, // delta_q_y_dc.delta_coded
            0, // delta_q_u_dc.delta_coded
            0, // delta_q_u_ac.delta_coded
            0, // using_qmatrix
            0,
        ];
        // Segmentation:
        bits.push(1); // segmentation_enabled
                      // primary_ref_frame == PRIMARY_REF_NONE ⇒ no update-flag reads.
                      // Segment 0 features:
                      // feature 0 (SEG_LVL_ALT_Q): enabled=1, su(9) value = -123.
                      // -123 in 9-bit two's complement: 2^9 - 123 = 389 = 0b1_1000_0101.
        bits.push(1); // feature_enabled[0][0]
        for i in (0..9).rev() {
            bits.push(((389u32 >> i) & 1) as u8);
        }
        // Segment 0 features 1..=7 disabled (7 zero bits) plus
        // segments 1..=7 × 8 features all disabled (56 zero bits).
        let tail_zeros = (SEG_LVL_MAX - 1) + (MAX_SEGMENTS - 1) * SEG_LVL_MAX;
        bits.resize(bits.len() + tail_zeros, 0);
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        let sp = fh.segmentation_params.as_ref().expect("intra has sp");
        assert!(sp.enabled);
        // primary_ref_frame == PRIMARY_REF_NONE collapses the three
        // update flags without bitstream reads.
        assert!(sp.update_map);
        assert!(!sp.temporal_update);
        assert!(sp.update_data);
        assert!(sp.segment_feature_active[0][SEG_LVL_ALT_Q]);
        assert_eq!(sp.segment_feature_data[0][SEG_LVL_ALT_Q], -123);
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if (i, j) != (0, SEG_LVL_ALT_Q) {
                    assert!(
                        !sp.segment_feature_active[i][j],
                        "feature ({i},{j}) must be inactive"
                    );
                    assert_eq!(sp.segment_feature_data[i][j], 0);
                }
            }
        }
        // SEG_LVL_ALT_Q is index 0 < SEG_LVL_REF_FRAME ⇒ SegIdPreSkip stays 0.
        assert!(!sp.seg_id_pre_skip);
        assert_eq!(SEG_LVL_ALT_Q, 0);
        assert_eq!(SEG_LVL_REF_FRAME, 5);
        assert_eq!(sp.last_active_seg_id, 0);
        // Round 9: with base_q_idx=0, no delta offsets, and segment 0's
        // SEG_LVL_ALT_Q = -123 clamping qindex to Clip3(0,255,-123)=0,
        // every segment qindex is 0 ⇒ CodedLossless=1 ⇒ §5.9.11
        // short-circuit (no bits read, delta_enabled=0).
        let lf = fh
            .loop_filter_params
            .as_ref()
            .expect("intra has loop_filter_params");
        assert!(lf.short_circuited);
        assert!(!lf.loop_filter_delta_enabled);
        assert_eq!(lf.loop_filter_level, [0, 0, 0, 0]);
        // CodedLossless=1 ⇒ §5.9.21 ONLY_4X4 with no bits read.
        assert_eq!(fh.tx_mode, Some(TxMode::Only4x4));
    }

    // -------------------------------------------------------------
    // Round 9: compute_coded_lossless unit coverage
    // -------------------------------------------------------------

    #[test]
    fn coded_lossless_true_when_base_q_zero_no_deltas_seg_off() {
        let qp = QuantizationParams {
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
        };
        let sp = SegmentationParams::disabled();
        assert!(compute_coded_lossless(&qp, &sp));
    }

    #[test]
    fn coded_lossless_false_when_base_q_nonzero() {
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
        assert!(!compute_coded_lossless(&qp, &sp));
    }

    #[test]
    fn coded_lossless_false_when_any_delta_q_nonzero() {
        // Even with base_q_idx==0, a non-zero DeltaQ?* breaks lossless
        // for all segments (the conjunction is segment-independent).
        let qp = QuantizationParams {
            base_q_idx: 0,
            delta_q_y_dc: 0,
            diff_uv_delta: false,
            delta_q_u_dc: 0,
            delta_q_u_ac: -3,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            using_qmatrix: false,
            qm_y: 0,
            qm_u: 0,
            qm_v: 0,
        };
        let sp = SegmentationParams::disabled();
        assert!(!compute_coded_lossless(&qp, &sp));
    }

    #[test]
    fn coded_lossless_alt_q_clamps_to_zero() {
        // base_q_idx=120 but segment 0's SEG_LVL_ALT_Q = -200 clamps its
        // qindex to Clip3(0,255,120-200)=0. The other 7 segments keep
        // qindex=120 (≠0), so CodedLossless is still 0.
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
        let mut sp = SegmentationParams::disabled();
        sp.enabled = true;
        sp.segment_feature_active[0][SEG_LVL_ALT_Q] = true;
        sp.segment_feature_data[0][SEG_LVL_ALT_Q] = -200;
        assert!(!compute_coded_lossless(&qp, &sp));

        // Now make EVERY segment's ALT_Q clamp its qindex to 0.
        for seg in sp.segment_feature_active.iter_mut() {
            seg[SEG_LVL_ALT_Q] = true;
        }
        for seg in sp.segment_feature_data.iter_mut() {
            seg[SEG_LVL_ALT_Q] = -200;
        }
        assert!(compute_coded_lossless(&qp, &sp));
    }

    #[test]
    fn coded_lossless_alt_q_ignored_when_segmentation_disabled() {
        // FeatureEnabled is meaningless unless segmentation_enabled is
        // set; seg_feature_active_idx ANDs with segmentation_enabled.
        // With seg disabled and base_q_idx=0, qindex stays base_q_idx=0.
        let qp = QuantizationParams {
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
        };
        let mut sp = SegmentationParams::disabled();
        // active flag set but enabled=false ⇒ seg_feature_active_idx=0.
        sp.segment_feature_active[0][SEG_LVL_ALT_Q] = true;
        sp.segment_feature_data[0][SEG_LVL_ALT_Q] = 50;
        assert!(compute_coded_lossless(&qp, &sp));
    }

    // -------------------------------------------------------------
    // Round 9: streaming loop_filter_params full path (non-lossless)
    // -------------------------------------------------------------

    #[test]
    fn loop_filter_streaming_full_path_with_levels() {
        // screen_content_seq is reduced-still 256x128 profile-0
        // (NumPlanes=3). Build a payload that reaches loop_filter_params
        // with a non-zero base_q_idx (⇒ CodedLossless=0 ⇒ full path) and
        // non-zero loop_filter_level[0], exercising the chroma level[2]/
        // [3] reads.
        let seq = screen_content_seq();
        let mut bits: Vec<u8> = vec![
            0, 1, 0, // disable_cdf_update / allow_scc / force_int_mv
            0, // render_and_frame_size_different
            0, // allow_intrabc
            // disable_frame_end_update_cdf derived (reduced-still ⇒ 1).
            1, // uniform_tile_spacing_flag
            0, // increment_tile_cols_log2 (stops at 0)
        ];
        // quantization_params: base_q_idx(8) = 40 (non-lossless),
        // three delta_coded(1)=0, using_qmatrix(1)=0.
        for i in (0..8).rev() {
            bits.push(((40u32 >> i) & 1) as u8);
        }
        bits.extend_from_slice(&[0, 0, 0, 0]); // 3 delta_coded + using_qmatrix
                                               // segmentation_enabled(1)=0.
        bits.push(0);
        // delta_q_params: base_q_idx>0 ⇒ delta_q_present(1)=0.
        bits.push(0);
        // delta_lf_params: delta_q_present=0 ⇒ no bits.
        // loop_filter_params (§5.9.11), full path (CodedLossless=0):
        //   loop_filter_level[0](6) = 9
        for i in (0..6).rev() {
            bits.push(((9u32 >> i) & 1) as u8);
        }
        //   loop_filter_level[1](6) = 0
        bits.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        //   level[0]!=0 ⇒ chroma level[2](6)=16, level[3](6)=7.
        for i in (0..6).rev() {
            bits.push(((16u32 >> i) & 1) as u8);
        }
        for i in (0..6).rev() {
            bits.push(((7u32 >> i) & 1) as u8);
        }
        //   loop_filter_sharpness(3) = 2
        for i in (0..3).rev() {
            bits.push(((2u32 >> i) & 1) as u8);
        }
        //   loop_filter_delta_enabled(1) = 0 (stop; no update walk).
        bits.push(0);
        // After loop_filter_params() the parser walks cdef_params()
        // (enable_cdef=0 ⇒ short-circuit, 0 bits), lr_params()
        // (enable_restoration=0 ⇒ short-circuit, 0 bits), read_tx_mode()
        // (CodedLossless=0 ⇒ tx_mode_select(1)=0 ⇒ TX_MODE_LARGEST), then
        // the round-13 tail: reduced_tx_set(1)=0 plus film_grain reset
        // (film_grain_params_present=0). Provide two extra zero bytes of
        // tail so those reads land on padding rather than the buffer end.
        let mut payload = vec![0u8; bits.len().div_ceil(8) + 2];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let fh = parse_frame_header(&payload, &seq).expect("decodes");
        let qp = fh.quantization_params.as_ref().expect("intra has qp");
        assert_eq!(qp.base_q_idx, 40);
        let lf = fh
            .loop_filter_params
            .as_ref()
            .expect("intra has loop_filter_params");
        assert!(!lf.short_circuited);
        assert_eq!(lf.loop_filter_level[0], 9);
        assert_eq!(lf.loop_filter_level[1], 0);
        assert_eq!(lf.loop_filter_level[2], 16);
        assert_eq!(lf.loop_filter_level[3], 7);
        assert_eq!(lf.loop_filter_sharpness, 2);
        assert!(!lf.loop_filter_delta_enabled);
        // base_q_idx=40 ⇒ CodedLossless=0 ⇒ §5.9.21 reads the
        // `tx_mode_select` bit from the zero-padded tail (= 0) ⇒
        // TX_MODE_LARGEST.
        assert_eq!(fh.tx_mode, Some(TxMode::TxModeLargest));
        // Round-13 tail off the zero padding: reduced_tx_set=0,
        // global motion identity, film grain reset.
        assert_eq!(fh.reduced_tx_set, Some(false));
        assert!(
            fh.global_motion_params
                .as_ref()
                .expect("intra has global_motion")
                .short_circuited
        );
        assert!(
            !fh.film_grain_params
                .as_ref()
                .expect("intra has film_grain")
                .apply_grain
        );
    }

    // -------------------------------------------------------------
    // §7.8 set_frame_refs()
    // -------------------------------------------------------------

    /// `get_relative_dist()` (§5.9.3) sign-extends within the
    /// `OrderHintBits`-wide window: with 7 bits the window is
    /// `[-64, 63]`, so a hint of 120 vs OrderHint 1 wraps negative.
    #[test]
    fn get_relative_dist_sign_extends() {
        // 7-bit order hints. dist(2, 1) = +1, dist(1, 2) = -1.
        assert_eq!(get_relative_dist(2, 1, 7, true), 1);
        assert_eq!(get_relative_dist(1, 2, 7, true), -1);
        // Wrap: dist(0, 127) within a 7-bit window = +1 (0 is "after"
        // 127 modulo 128).
        assert_eq!(get_relative_dist(0, 127, 7, true), 1);
        // Disabled order hints => 0.
        assert_eq!(get_relative_dist(50, 1, 7, false), 0);
    }

    /// All references in the past (forward refs only). With
    /// `OrderHint = 8` and four valid backward... er, forward slots at
    /// hints {7,6,5,4}, every backward search fails and the forward
    /// loop + final fallback assign slots. LAST/GOLDEN keep their
    /// explicit indices.
    #[test]
    fn set_frame_refs_all_forward() {
        // Slots 0..3 hold past frames (hints 4,5,6,7); 4..7 unused
        // (hint 0). OrderHint = 8, 7-bit hints.
        let ref_order_hint = [4u32, 5, 6, 7, 0, 0, 0, 0];
        // last_frame_idx = 3 (hint 7, the closest past frame),
        // gold_frame_idx = 0 (hint 4, the oldest).
        let idx = set_frame_refs(3, 0, 8, 7, true, &ref_order_hint);
        // ref_frame_idx[LAST-LAST]=last, [GOLDEN-LAST]=gold per §7.8.
        assert_eq!(idx[LAST_FRAME - LAST_FRAME], 3, "LAST = last_frame_idx");
        assert_eq!(idx[GOLDEN_FRAME - LAST_FRAME], 0, "GOLDEN = gold_frame_idx");
        // No backward refs exist (all hints < curFrameHint), so
        // ALTREF/BWDREF/ALTREF2 fall to the forward / smallest-order
        // fallbacks; every entry resolves to a valid slot (0..=7).
        for (i, &v) in idx.iter().enumerate() {
            assert!(
                (v as usize) < NUM_REF_FRAMES as usize,
                "idx[{i}]={v} in range"
            );
        }
    }

    /// Mixed past + future refs: a forward ref (past) and a backward
    /// ref (future) are present, so ALTREF picks the latest backward
    /// and the LAST/GOLDEN explicit indices are preserved.
    #[test]
    fn set_frame_refs_backward_ref_picked_for_altref() {
        // OrderHint = 4 (7-bit). Slot 0 hint 2 (past), slot 1 hint 3
        // (past), slot 2 hint 6 (future), slot 3 hint 8 (further
        // future). 4..7 unused.
        let ref_order_hint = [2u32, 3, 6, 8, 0, 0, 0, 0];
        let idx = set_frame_refs(1, 0, 4, 7, true, &ref_order_hint);
        assert_eq!(idx[LAST_FRAME - LAST_FRAME], 1, "LAST = last_frame_idx=1");
        assert_eq!(
            idx[GOLDEN_FRAME - LAST_FRAME],
            0,
            "GOLDEN = gold_frame_idx=0"
        );
        // ALTREF = find_latest_backward(): the unused future slot with
        // the highest shifted hint = slot 3 (hint 8).
        assert_eq!(
            idx[ALTREF_FRAME - LAST_FRAME],
            3,
            "ALTREF = latest backward (slot 3)"
        );
        // BWDREF = find_earliest_backward(): next earliest future =
        // slot 2 (hint 6).
        assert_eq!(
            idx[BWDREF_FRAME - LAST_FRAME],
            2,
            "BWDREF = earliest backward (slot 2)"
        );
    }
}
