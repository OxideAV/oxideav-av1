//! Sub-syntax functions called from the tail of
//! `uncompressed_header()` (§5.9.2). Rounds 5–8 of the clean-room
//! rebuild landed `read_interpolation_filter()` (§5.9.10),
//! `loop_filter_params()` (§5.9.11), `quantization_params()` (§5.9.12),
//! `segmentation_params()` (§5.9.14), `delta_q_params()` (§5.9.17),
//! `delta_lf_params()` (§5.9.18), `cdef_params()` (§5.9.19),
//! `lr_params()` (§5.9.20), and — added this round —
//! `read_tx_mode()` (§5.9.21).
//!
//! `quantization_params()`, `segmentation_params()`,
//! `delta_q_params()`, `delta_lf_params()`, `cdef_params()`,
//! `lr_params()`, and `read_tx_mode()` are also wired into the streaming
//! [`crate::parse_frame_header`] walk (intra path) since they sit
//! before the remaining `frame_reference_mode()` block. The standalone
//! entry points
//! remain available so callers that want to exercise the parsers on a
//! raw byte slice can do so without rebuilding a `SequenceHeader` and a
//! streaming context.
//!
//! ## Syntax / semantics references (all in `docs/video/av1/`)
//!
//!   * §5.9.10 — `read_interpolation_filter()`
//!   * §5.9.11 — `loop_filter_params()`
//!   * §5.9.12 — `quantization_params()`
//!   * §5.9.13 — `read_delta_q()` (helper for §5.9.12)
//!   * §5.9.14 — `segmentation_params()`
//!   * §5.9.17 — `delta_q_params()`
//!   * §5.9.18 — `delta_lf_params()`
//!   * §5.9.19 — `cdef_params()`
//!   * §5.9.20 — `lr_params()`
//!   * §5.9.21 — `read_tx_mode()`
//!   * §6.8.21 — TX mode semantics
//!   * §6.10.14 — CDEF params semantics
//!   * §6.10.15 — Loop restoration params semantics
//!   * §6.8.9  — Interpolation filter semantics
//!   * §6.8.10 — Loop filter semantics
//!   * §6.8.11 — Quantization params semantics
//!   * §6.8.12 — Delta quantizer semantics
//!   * §6.8.13 — Segmentation params semantics
//!   * §6.8.15 — Quantizer index delta params semantics
//!   * §6.8.16 — Loop filter delta params semantics
//!   * §4.10.6 — `su(n)` signed-integer descriptor (used for
//!     `loop_filter_ref_deltas` / `loop_filter_mode_deltas` / the
//!     `delta_q` field of `read_delta_q()` / signed `feature_value`
//!     reads inside `segmentation_params()`).
//!
//! ## §3 constants referenced here
//!
//!   * `TOTAL_REFS_PER_FRAME = 8` — total number of reference-frame
//!     types including the implicit `INTRA_FRAME`. Used as the loop
//!     bound for the `loop_filter_ref_deltas[i]` update walk inside
//!     `loop_filter_params()`.
//!   * `MAX_SEGMENTS = 8` — number of segments per §5.9.14.
//!   * `SEG_LVL_MAX = 8` — number of per-segment features per §5.9.14.
//!   * `SEG_LVL_ALT_Q = 0` / `SEG_LVL_ALT_LF_Y_V = 1` /
//!     `SEG_LVL_ALT_LF_Y_H = 2` / `SEG_LVL_ALT_LF_U = 3` /
//!     `SEG_LVL_ALT_LF_V = 4` / `SEG_LVL_REF_FRAME = 5` /
//!     `SEG_LVL_SKIP = 6` / `SEG_LVL_GLOBALMV = 7` — feature indices.
//!   * `MAX_LOOP_FILTER = 63` — clip bound for the four loop-filter
//!     features (per §5.9.14 `Segmentation_Feature_Max`).

use crate::bitreader::BitReader;
use crate::Error;

// ---------------------------------------------------------------------
// §3 constants
// ---------------------------------------------------------------------

/// `TOTAL_REFS_PER_FRAME` per §3 — `INTRA_FRAME` plus the seven
/// inter-prediction reference frame types.
pub const TOTAL_REFS_PER_FRAME: usize = 8;

/// `MAX_SEGMENTS` per §3 — number of segments allowed in the
/// segmentation map (§5.9.14 outer loop bound).
pub const MAX_SEGMENTS: usize = 8;

/// `SEG_LVL_MAX` per §3 — number of segment features per
/// §5.9.14 inner loop bound.
pub const SEG_LVL_MAX: usize = 8;

/// `SEG_LVL_ALT_Q` per §3 — feature index 0 (quantiser delta).
pub const SEG_LVL_ALT_Q: usize = 0;

/// `SEG_LVL_ALT_LF_Y_V` per §3 — feature index 1 (vertical-luma
/// loop-filter delta).
pub const SEG_LVL_ALT_LF_Y_V: usize = 1;

/// `SEG_LVL_ALT_LF_Y_H` per §3 — feature index 2 (horizontal-luma
/// loop-filter delta).
pub const SEG_LVL_ALT_LF_Y_H: usize = 2;

/// `SEG_LVL_ALT_LF_U` per §3 — feature index 3 (U-plane loop-filter
/// delta).
pub const SEG_LVL_ALT_LF_U: usize = 3;

/// `SEG_LVL_ALT_LF_V` per §3 — feature index 4 (V-plane loop-filter
/// delta).
pub const SEG_LVL_ALT_LF_V: usize = 4;

/// `SEG_LVL_REF_FRAME` per §3 — feature index 5 (reference frame).
/// `SegIdPreSkip` is set when any active feature has `j >=
/// SEG_LVL_REF_FRAME` (§5.9.14 trailing derivation).
pub const SEG_LVL_REF_FRAME: usize = 5;

/// `SEG_LVL_SKIP` per §3 — feature index 6 (skip).
pub const SEG_LVL_SKIP: usize = 6;

/// `SEG_LVL_GLOBALMV` per §3 — feature index 7 (global MV).
pub const SEG_LVL_GLOBALMV: usize = 7;

/// `MAX_LOOP_FILTER` per §3 — clipping limit for the four
/// loop-filter features in `Segmentation_Feature_Max` (§5.9.14).
pub const MAX_LOOP_FILTER: i16 = 63;

/// `Segmentation_Feature_Bits[ SEG_LVL_MAX ]` table from §5.9.14:
/// the bit width of `feature_value` for each feature index (or 0
/// when the feature has no associated value).
pub const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];

/// `Segmentation_Feature_Signed[ SEG_LVL_MAX ]` table from §5.9.14:
/// 1 for features whose value is read as `su(1+bitsToRead)`, 0 for
/// features whose value is read as `f(bitsToRead)`.
pub const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];

/// `Segmentation_Feature_Max[ SEG_LVL_MAX ]` table from §5.9.14:
/// the absolute-value clip bound for each feature. Indices 6 and 7
/// have a max of 0 (skip / globalmv are boolean — they have no
/// associated value, so `feature_value = 0` is forced through
/// `Clip3(0, 0, ...)`).
pub const SEGMENTATION_FEATURE_MAX: [i16; SEG_LVL_MAX] = [
    255,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    MAX_LOOP_FILTER,
    7,
    0,
    0,
];

// ---------------------------------------------------------------------
// §5.9.10 read_interpolation_filter
// ---------------------------------------------------------------------

/// `interpolation_filter` per §6.8.9.
///
/// | code | name                |
/// |-----:|:--------------------|
/// |   0  | `EIGHTTAP`          |
/// |   1  | `EIGHTTAP_SMOOTH`   |
/// |   2  | `EIGHTTAP_SHARP`    |
/// |   3  | `BILINEAR`          |
/// |   4  | `SWITCHABLE`        |
///
/// The `SWITCHABLE` code is **not** encoded as `f(2) == 4` in the
/// bitstream — it's instead signalled by `is_filter_switchable == 1`
/// per §5.9.10, in which case the §5.9.10 syntax skips the `f(2)`
/// `interpolation_filter` field entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationFilter {
    Eighttap,
    EighttapSmooth,
    EighttapSharp,
    Bilinear,
    Switchable,
}

impl InterpolationFilter {
    /// Decode the raw `f(2)` `interpolation_filter` value (only valid
    /// for codes 0..=3 — `SWITCHABLE` is never written as `f(2)` in
    /// the bitstream).
    pub fn from_raw(raw: u8) -> Self {
        match raw & 0x3 {
            0 => Self::Eighttap,
            1 => Self::EighttapSmooth,
            2 => Self::EighttapSharp,
            _ => Self::Bilinear,
        }
    }

    /// Numeric code from §6.8.9.
    pub fn as_raw(&self) -> u8 {
        match self {
            Self::Eighttap => 0,
            Self::EighttapSmooth => 1,
            Self::EighttapSharp => 2,
            Self::Bilinear => 3,
            Self::Switchable => 4,
        }
    }

    /// Whether this filter is the §5.9.10 `is_filter_switchable == 1`
    /// catch-all.
    pub fn is_switchable(&self) -> bool {
        matches!(self, Self::Switchable)
    }
}

/// Read `read_interpolation_filter()` (§5.9.10) from `payload`.
///
/// Returns the chosen [`InterpolationFilter`] plus the number of bits
/// consumed. The caller is responsible for positioning `payload` at
/// the right bit before invoking — the standalone entry points in
/// this round always start at bit 0.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`] — payload exhausted mid-read.
pub fn parse_interpolation_filter(payload: &[u8]) -> Result<(InterpolationFilter, usize), Error> {
    let mut br = BitReader::new(payload);
    let filt = read_interpolation_filter(&mut br)?;
    Ok((filt, br.position()))
}

pub(crate) fn read_interpolation_filter(
    br: &mut BitReader<'_>,
) -> Result<InterpolationFilter, Error> {
    let is_filter_switchable = br.f(1)? == 1;
    if is_filter_switchable {
        Ok(InterpolationFilter::Switchable)
    } else {
        let raw = br.f(2)? as u8;
        Ok(InterpolationFilter::from_raw(raw))
    }
}

// ---------------------------------------------------------------------
// §5.9.11 loop_filter_params
// ---------------------------------------------------------------------

/// Defaults applied when the §5.9.11 `(CodedLossless || allow_intrabc)`
/// short-circuit fires. Mirrors the spec's literal initialisers per
/// `loop_filter_ref_deltas[ INTRA_FRAME ] = 1; loop_filter_ref_deltas[
/// LAST_FRAME ] = 0; ...` block. INTRA_FRAME is index 0,
/// LAST_FRAME..ALTREF2_FRAME are indices 1..=7 per the §3 enumeration
/// (the spec lists LAST=1, LAST2=2, LAST3=3, GOLDEN=4, BWDREF=5,
/// ALTREF2=6, ALTREF=7 ordering — but only the values matter here,
/// indexed by the §3 numbering).
pub const LOOP_FILTER_REF_DELTAS_DEFAULT: [i8; TOTAL_REFS_PER_FRAME] = {
    // Spec defaults per §5.9.11 short-circuit path:
    //   INTRA_FRAME    = 1
    //   LAST_FRAME     = 0
    //   LAST2_FRAME    = 0
    //   LAST3_FRAME    = 0
    //   GOLDEN_FRAME   = -1
    //   BWDREF_FRAME   = 0
    //   ALTREF2_FRAME  = -1
    //   ALTREF_FRAME   = -1
    //
    // Per §3's numbering of the inter-prediction enums:
    //   INTRA=0, LAST=1, LAST2=2, LAST3=3, GOLDEN=4, BWDREF=5,
    //   ALTREF2=6, ALTREF=7.
    [1, 0, 0, 0, -1, 0, -1, -1]
};

/// Defaults for `loop_filter_mode_deltas[]` per §5.9.11 short-circuit.
pub const LOOP_FILTER_MODE_DELTAS_DEFAULT: [i8; 2] = [0, 0];

/// Parsed `loop_filter_params()` per §5.9.11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopFilterParams {
    /// `loop_filter_level[0..=3]`. Indices 2 and 3 are populated only
    /// when `NumPlanes > 1` and at least one of `loop_filter_level[0]`
    /// / `loop_filter_level[1]` is non-zero; otherwise the spec leaves
    /// them at the default 0.
    pub loop_filter_level: [u8; 4],
    /// `loop_filter_sharpness` (`f(3)`). 0 when the §5.9.11
    /// short-circuit fires.
    pub loop_filter_sharpness: u8,
    /// `loop_filter_delta_enabled` (`f(1)`). 0 when the §5.9.11
    /// short-circuit fires.
    pub loop_filter_delta_enabled: bool,
    /// `loop_filter_delta_update` (`f(1)`). Only meaningful when
    /// `loop_filter_delta_enabled == 1`. 0 otherwise.
    pub loop_filter_delta_update: bool,
    /// `loop_filter_ref_deltas[0..=7]` after the §5.9.11 update walk.
    /// In the streaming parser these values "maintain previous value"
    /// (§6.8.10) when `update_ref_delta == 0`; the standalone parser
    /// returns the spec's "no previous value" defaults
    /// ([`LOOP_FILTER_REF_DELTAS_DEFAULT`]) for slots that weren't
    /// updated this frame.
    pub loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    /// `loop_filter_mode_deltas[0..=1]` after the §5.9.11 update walk.
    /// Same "no previous value" caveat as
    /// [`Self::loop_filter_ref_deltas`].
    pub loop_filter_mode_deltas: [i8; 2],
    /// Whether the §5.9.11 short-circuit (`CodedLossless ||
    /// allow_intrabc`) fired and the parser returned without reading
    /// any bits.
    pub short_circuited: bool,
}

impl LoopFilterParams {
    /// The §5.9.11 short-circuit-path output (no bits consumed).
    pub const fn short_circuit() -> Self {
        Self {
            loop_filter_level: [0; 4],
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: false,
            loop_filter_delta_update: false,
            loop_filter_ref_deltas: LOOP_FILTER_REF_DELTAS_DEFAULT,
            loop_filter_mode_deltas: LOOP_FILTER_MODE_DELTAS_DEFAULT,
            short_circuited: true,
        }
    }
}

/// Parse `loop_filter_params()` per §5.9.11.
///
/// `num_planes` is §5.5.2's `NumPlanes` derived value (1 for
/// monochrome, otherwise 3). `coded_lossless` and `allow_intrabc` are
/// the runtime-derived flags from §5.9.2 that select the
/// short-circuit path; standalone callers that haven't run the
/// preceding `quantization_params()` + intrabc reads can pass
/// `coded_lossless = false, allow_intrabc = false` to exercise the
/// full bitstream path.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_loop_filter_params(
    payload: &[u8],
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<(LoopFilterParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let lf = read_loop_filter_params(&mut br, num_planes, coded_lossless, allow_intrabc)?;
    Ok((lf, br.position()))
}

pub(crate) fn read_loop_filter_params(
    br: &mut BitReader<'_>,
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<LoopFilterParams, Error> {
    if coded_lossless || allow_intrabc {
        return Ok(LoopFilterParams::short_circuit());
    }

    let mut loop_filter_level = [0u8; 4];
    loop_filter_level[0] = br.f(6)? as u8;
    loop_filter_level[1] = br.f(6)? as u8;
    if num_planes > 1 && (loop_filter_level[0] != 0 || loop_filter_level[1] != 0) {
        loop_filter_level[2] = br.f(6)? as u8;
        loop_filter_level[3] = br.f(6)? as u8;
    }

    let loop_filter_sharpness = br.f(3)? as u8;
    let loop_filter_delta_enabled = br.f(1)? == 1;

    let mut loop_filter_ref_deltas = LOOP_FILTER_REF_DELTAS_DEFAULT;
    let mut loop_filter_mode_deltas = LOOP_FILTER_MODE_DELTAS_DEFAULT;
    let mut loop_filter_delta_update = false;

    if loop_filter_delta_enabled {
        loop_filter_delta_update = br.f(1)? == 1;
        if loop_filter_delta_update {
            for slot in loop_filter_ref_deltas.iter_mut() {
                let update_ref_delta = br.f(1)? == 1;
                if update_ref_delta {
                    *slot = br.su(7)? as i8;
                }
            }
            for slot in loop_filter_mode_deltas.iter_mut() {
                let update_mode_delta = br.f(1)? == 1;
                if update_mode_delta {
                    *slot = br.su(7)? as i8;
                }
            }
        }
    }

    Ok(LoopFilterParams {
        loop_filter_level,
        loop_filter_sharpness,
        loop_filter_delta_enabled,
        loop_filter_delta_update,
        loop_filter_ref_deltas,
        loop_filter_mode_deltas,
        short_circuited: false,
    })
}

// ---------------------------------------------------------------------
// §5.9.12 quantization_params + §5.9.13 read_delta_q
// ---------------------------------------------------------------------

/// `quantization_params()` per §5.9.12 + §6.8.11.
///
/// `base_q_idx` is the unsigned 8-bit base Y AC qindex. The four
/// `delta_q_*` fields hold the per-plane DC / AC offsets relative to
/// `base_q_idx`, decoded through §5.9.13 `read_delta_q()`. For
/// `num_planes == 1` (monochrome) the U/V deltas remain at 0 per
/// §5.9.12 line `DeltaQUDc = 0; DeltaQUAc = 0; DeltaQVDc = 0;
/// DeltaQVAc = 0`. For `num_planes > 1` with
/// `separate_uv_delta_q == 0`, `delta_q_v_*` mirror their
/// `delta_q_u_*` counterparts per §5.9.12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationParams {
    /// `base_q_idx` (`f(8)`).
    pub base_q_idx: u8,
    /// `DeltaQYDc` — Y DC offset relative to `base_q_idx`.
    pub delta_q_y_dc: i8,
    /// `diff_uv_delta` — read only when `NumPlanes > 1 &&
    /// separate_uv_delta_q == 1`, otherwise 0.
    pub diff_uv_delta: bool,
    /// `DeltaQUDc` — U DC offset.
    pub delta_q_u_dc: i8,
    /// `DeltaQUAc` — U AC offset.
    pub delta_q_u_ac: i8,
    /// `DeltaQVDc` — V DC offset (mirrors `delta_q_u_dc` when
    /// `diff_uv_delta == 0`).
    pub delta_q_v_dc: i8,
    /// `DeltaQVAc` — V AC offset.
    pub delta_q_v_ac: i8,
    /// `using_qmatrix` (`f(1)`).
    pub using_qmatrix: bool,
    /// `qm_y` (`f(4)`) — only meaningful when `using_qmatrix == 1`.
    pub qm_y: u8,
    /// `qm_u` (`f(4)`).
    pub qm_u: u8,
    /// `qm_v` (`f(4)`) — mirrors `qm_u` when `separate_uv_delta_q == 0`.
    pub qm_v: u8,
}

/// Parse `quantization_params()` per §5.9.12.
///
/// `num_planes` is §5.5.2's `NumPlanes`. `separate_uv_delta_q` is
/// §5.5.2's `separate_uv_delta_q` — both are surfaced on
/// [`crate::SequenceHeader::color_config`].
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_quantization_params(
    payload: &[u8],
    num_planes: u8,
    separate_uv_delta_q: bool,
) -> Result<(QuantizationParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let q = read_quantization_params(&mut br, num_planes, separate_uv_delta_q)?;
    Ok((q, br.position()))
}

pub(crate) fn read_quantization_params(
    br: &mut BitReader<'_>,
    num_planes: u8,
    separate_uv_delta_q: bool,
) -> Result<QuantizationParams, Error> {
    let base_q_idx = br.f(8)? as u8;
    let delta_q_y_dc = read_delta_q(br)?;

    let mut diff_uv_delta = false;
    let mut delta_q_u_dc = 0i8;
    let mut delta_q_u_ac = 0i8;
    let mut delta_q_v_dc = 0i8;
    let mut delta_q_v_ac = 0i8;

    if num_planes > 1 {
        if separate_uv_delta_q {
            diff_uv_delta = br.f(1)? == 1;
        }
        delta_q_u_dc = read_delta_q(br)?;
        delta_q_u_ac = read_delta_q(br)?;
        if diff_uv_delta {
            delta_q_v_dc = read_delta_q(br)?;
            delta_q_v_ac = read_delta_q(br)?;
        } else {
            delta_q_v_dc = delta_q_u_dc;
            delta_q_v_ac = delta_q_u_ac;
        }
    }

    let using_qmatrix = br.f(1)? == 1;
    let (qm_y, qm_u, qm_v) = if using_qmatrix {
        let qm_y = br.f(4)? as u8;
        let qm_u = br.f(4)? as u8;
        let qm_v = if separate_uv_delta_q {
            br.f(4)? as u8
        } else {
            qm_u
        };
        (qm_y, qm_u, qm_v)
    } else {
        (0, 0, 0)
    };

    Ok(QuantizationParams {
        base_q_idx,
        delta_q_y_dc,
        diff_uv_delta,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
        using_qmatrix,
        qm_y,
        qm_u,
        qm_v,
    })
}

/// `read_delta_q()` per §5.9.13: a `delta_coded == 1` prefix bit gates
/// a `su(1+6) = su(7)` signed offset; absent ⇒ `delta_q = 0`.
fn read_delta_q(br: &mut BitReader<'_>) -> Result<i8, Error> {
    let delta_coded = br.f(1)? == 1;
    if delta_coded {
        Ok(br.su(7)? as i8)
    } else {
        Ok(0)
    }
}

// ---------------------------------------------------------------------
// §5.9.14 segmentation_params
// ---------------------------------------------------------------------

/// Parsed `segmentation_params()` per §5.9.14.
///
/// The decoded `FeatureEnabled[i][j]` / `FeatureData[i][j]` arrays from
/// the §5.9.14 inner loop are surfaced as `segment_feature_active` /
/// `segment_feature_data` (outer index `i` ∈ `0..MAX_SEGMENTS`, inner
/// index `j` ∈ `0..SEG_LVL_MAX`).
///
/// When `segmentation_enabled == 0` the spec leaves
/// `FeatureEnabled[i][j]` / `FeatureData[i][j]` at their previous-frame
/// (or `setup_past_independence`-reset) values; this structural parser
/// reports a fully-zero array in that case because the standalone walk
/// does not carry per-session history. The streaming-parser caller in
/// `parse_frame_header` follows the same convention — the persistence
/// model is the next round's `load_segmentation_params()` work.
///
/// When `segmentation_enabled == 1 && segmentation_update_data == 0`
/// the spec keeps the existing per-segment feature data; this
/// structural parser also reports zeros in that case, with
/// [`Self::update_data`] left `false` so a session-aware
/// layer can detect the "use saved state" branch and substitute the
/// loaded data.
///
/// The §5.9.14 trailing derivations `SegIdPreSkip` and `LastActiveSegId`
/// are also surfaced as [`Self::seg_id_pre_skip`] and
/// [`Self::last_active_seg_id`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentationParams {
    /// `segmentation_enabled` (`f(1)`).
    pub enabled: bool,
    /// `segmentation_update_map`. Per §5.9.14: forced to `1` when
    /// `primary_ref_frame == PRIMARY_REF_NONE`; otherwise read as
    /// `f(1)`. `false` when `enabled == 0`.
    pub update_map: bool,
    /// `segmentation_temporal_update`. Per §5.9.14: forced to `0` when
    /// `primary_ref_frame == PRIMARY_REF_NONE`; otherwise read as
    /// `f(1)` only when `update_map == 1`. `false` when `enabled == 0`.
    pub temporal_update: bool,
    /// `segmentation_update_data`. Per §5.9.14: forced to `1` when
    /// `primary_ref_frame == PRIMARY_REF_NONE`; otherwise read as
    /// `f(1)`. `false` when `enabled == 0`.
    pub update_data: bool,
    /// `FeatureEnabled[i][j]` from the §5.9.14 inner loop.
    pub segment_feature_active: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `FeatureData[i][j]` from the §5.9.14 inner loop, after the
    /// `Clip3(-limit, limit, feature_value)` / `Clip3(0, limit,
    /// feature_value)` clamp. Stored as `i16` to fit the worst-case
    /// signed range `[-255, 255]` of `SEG_LVL_ALT_Q`.
    pub segment_feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    /// `SegIdPreSkip` trailing derivation: `1` if any active feature has
    /// `j >= SEG_LVL_REF_FRAME`, otherwise `0`.
    pub seg_id_pre_skip: bool,
    /// `LastActiveSegId` trailing derivation: highest segment index `i`
    /// with at least one active feature. `0` when no feature is active.
    pub last_active_seg_id: u8,
}

impl SegmentationParams {
    /// The §5.9.14 disabled-path output: every slot zeroed.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal_update: false,
            update_data: false,
            segment_feature_active: [[false; SEG_LVL_MAX]; MAX_SEGMENTS],
            segment_feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
            seg_id_pre_skip: false,
            last_active_seg_id: 0,
        }
    }
}

/// Parse `segmentation_params()` per §5.9.14.
///
/// `primary_ref_frame` is the §5.9.2 `primary_ref_frame` value — the
/// `PRIMARY_REF_NONE = 7` sentinel collapses the three update flags
/// to fixed values (`update_map = 1`, `temporal_update = 0`,
/// `update_data = 1`).
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_segmentation_params(
    payload: &[u8],
    primary_ref_frame: u8,
) -> Result<(SegmentationParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let s = read_segmentation_params(&mut br, primary_ref_frame)?;
    Ok((s, br.position()))
}

pub(crate) fn read_segmentation_params(
    br: &mut BitReader<'_>,
    primary_ref_frame: u8,
) -> Result<SegmentationParams, Error> {
    let enabled = br.f(1)? == 1;
    if !enabled {
        return Ok(SegmentationParams::disabled());
    }

    let (update_map, temporal_update, update_data) =
        if primary_ref_frame == crate::frame_header::PRIMARY_REF_NONE {
            // §5.9.14: `primary_ref_frame == PRIMARY_REF_NONE` collapses
            // the three update flags to fixed values without reading.
            (true, false, true)
        } else {
            let um = br.f(1)? == 1;
            let tu = if um { br.f(1)? == 1 } else { false };
            let ud = br.f(1)? == 1;
            (um, tu, ud)
        };

    let mut feature_active = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
    let mut feature_data = [[0i16; SEG_LVL_MAX]; MAX_SEGMENTS];

    if update_data {
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                let feature_enabled = br.f(1)? == 1;
                feature_active[i][j] = feature_enabled;
                let clipped = if feature_enabled {
                    let bits_to_read = SEGMENTATION_FEATURE_BITS[j];
                    let limit = SEGMENTATION_FEATURE_MAX[j];
                    if SEGMENTATION_FEATURE_SIGNED[j] {
                        // §5.9.14 reads `su(1+bitsToRead)` even when
                        // `bitsToRead == 0`. For indices 6/7 the bits
                        // are 0 and signed flag is 0, so this branch
                        // doesn't fire; defensive nonetheless.
                        let feature_value = br.su(1 + bits_to_read)?;
                        clip3_i32(-i32::from(limit), i32::from(limit), feature_value) as i16
                    } else if bits_to_read == 0 {
                        // §5.9.14: `feature_value = 0` initialiser
                        // stands when `bits_to_read == 0`. Clip3 with
                        // limit == 0 forces 0.
                        0
                    } else {
                        let feature_value = br.f(bits_to_read)? as i32;
                        clip3_i32(0, i32::from(limit), feature_value) as i16
                    }
                } else {
                    0
                };
                feature_data[i][j] = clipped;
            }
        }
    }

    // §5.9.14 trailing derivation of SegIdPreSkip / LastActiveSegId.
    let mut seg_id_pre_skip = false;
    let mut last_active_seg_id: u8 = 0;
    for (i, row) in feature_active.iter().enumerate() {
        for (j, &active) in row.iter().enumerate() {
            if active {
                last_active_seg_id = i as u8;
                if j >= SEG_LVL_REF_FRAME {
                    seg_id_pre_skip = true;
                }
            }
        }
    }

    Ok(SegmentationParams {
        enabled,
        update_map,
        temporal_update,
        update_data,
        segment_feature_active: feature_active,
        segment_feature_data: feature_data,
        seg_id_pre_skip,
        last_active_seg_id,
    })
}

// ---------------------------------------------------------------------
// §5.9.17 delta_q_params
// ---------------------------------------------------------------------

/// Parsed `delta_q_params()` per §5.9.17.
///
/// Per the syntax the `delta_q_present` `f(1)` slot is read only when
/// `base_q_idx > 0`; otherwise it stays at its `delta_q_present = 0`
/// initialiser and no bit is consumed. `delta_q_res` (the `f(2)` left
/// shift applied to decoded quantiser-index deltas — §6.8.15) is read
/// only when `delta_q_present == 1`, otherwise it stays 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeltaQParams {
    /// `delta_q_present` — whether per-block quantiser-index deltas are
    /// signalled at superblock granularity (§6.8.15).
    pub delta_q_present: bool,
    /// `delta_q_res` (`f(2)`) — left shift applied to decoded quantiser
    /// index delta values. `0` when `delta_q_present == 0`.
    pub delta_q_res: u8,
}

/// Parse `delta_q_params()` per §5.9.17 from a raw byte slice.
///
/// `base_q_idx` is §5.9.12's `base_q_idx`; the `delta_q_present` slot is
/// only read when it is greater than 0.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_delta_q_params(
    payload: &[u8],
    base_q_idx: u8,
) -> Result<(DeltaQParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let d = read_delta_q_params(&mut br, base_q_idx)?;
    Ok((d, br.position()))
}

pub(crate) fn read_delta_q_params(
    br: &mut BitReader<'_>,
    base_q_idx: u8,
) -> Result<DeltaQParams, Error> {
    // §5.9.17: delta_q_res / delta_q_present default to 0.
    let mut delta_q_present = false;
    let mut delta_q_res = 0u8;
    if base_q_idx > 0 {
        delta_q_present = br.f(1)? == 1;
    }
    if delta_q_present {
        delta_q_res = br.f(2)? as u8;
    }
    Ok(DeltaQParams {
        delta_q_present,
        delta_q_res,
    })
}

// ---------------------------------------------------------------------
// §5.9.18 delta_lf_params
// ---------------------------------------------------------------------

/// Parsed `delta_lf_params()` per §5.9.18.
///
/// The whole block is gated on `delta_q_present` (from §5.9.17). When
/// it is set, `delta_lf_present` is read as `f(1)` only when
/// `!allow_intrabc` (§5.9.18); when `delta_lf_present == 1` the
/// `delta_lf_res` (`f(2)`) and `delta_lf_multi` (`f(1)`) fields follow.
/// All three default to 0 / false otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeltaLfParams {
    /// `delta_lf_present` — whether per-block loop-filter deltas are
    /// signalled (§6.8.16).
    pub delta_lf_present: bool,
    /// `delta_lf_res` (`f(2)`) — left shift applied to decoded
    /// loop-filter delta values. `0` when `delta_lf_present == 0`.
    pub delta_lf_res: u8,
    /// `delta_lf_multi` — when set, separate loop-filter deltas are
    /// sent for horizontal-luma / vertical-luma / U / V edges; when
    /// clear, a single delta applies to all edges (§6.8.16). `false`
    /// when `delta_lf_present == 0`.
    pub delta_lf_multi: bool,
}

/// Parse `delta_lf_params()` per §5.9.18 from a raw byte slice.
///
/// `delta_q_present` is §5.9.17's `delta_q_present` (the whole block is
/// a no-op when it is `false`). `allow_intrabc` is §5.9.3's
/// `allow_intrabc` (the `delta_lf_present` slot is suppressed when it
/// is `true`).
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_delta_lf_params(
    payload: &[u8],
    delta_q_present: bool,
    allow_intrabc: bool,
) -> Result<(DeltaLfParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let d = read_delta_lf_params(&mut br, delta_q_present, allow_intrabc)?;
    Ok((d, br.position()))
}

pub(crate) fn read_delta_lf_params(
    br: &mut BitReader<'_>,
    delta_q_present: bool,
    allow_intrabc: bool,
) -> Result<DeltaLfParams, Error> {
    // §5.9.18: all three fields default to 0 / false.
    let mut delta_lf_present = false;
    let mut delta_lf_res = 0u8;
    let mut delta_lf_multi = false;
    if delta_q_present {
        if !allow_intrabc {
            delta_lf_present = br.f(1)? == 1;
        }
        if delta_lf_present {
            delta_lf_res = br.f(2)? as u8;
            delta_lf_multi = br.f(1)? == 1;
        }
    }
    Ok(DeltaLfParams {
        delta_lf_present,
        delta_lf_res,
        delta_lf_multi,
    })
}

// ---------------------------------------------------------------------
// §5.9.19 cdef_params
// ---------------------------------------------------------------------

/// Maximum number of CDEF strength entries. `cdef_bits` is read as
/// `f(2)`, so its value is in `0..=3` and the §5.9.19 loop runs
/// `1 << cdef_bits` ≤ `1 << 3 = 8` times. The four strength arrays are
/// therefore sized at 8.
pub const CDEF_MAX_STRENGTHS: usize = 8;

/// Parsed `cdef_params()` per §5.9.19 + §6.10.14.
///
/// CDEF (constrained directional enhancement filter) deringing
/// parameters. When the §5.9.19 short-circuit fires (`CodedLossless ||
/// allow_intrabc || !enable_cdef`) the spec leaves `cdef_bits = 0`,
/// `CdefDamping = 3`, and the four strength arrays at their index-0
/// zero defaults; [`Self::short_circuited`] records that no bits were
/// read.
///
/// For the full path the parser reads `cdef_damping_minus_3` (`f(2)`,
/// `CdefDamping = cdef_damping_minus_3 + 3`), `cdef_bits` (`f(2)`), and
/// then for each of the `1 << cdef_bits` entries reads
/// `cdef_y_pri_strength[i]` (`f(4)`) / `cdef_y_sec_strength[i]`
/// (`f(2)`) and, when `NumPlanes > 1`, `cdef_uv_pri_strength[i]`
/// (`f(4)`) / `cdef_uv_sec_strength[i]` (`f(2)`). The §5.9.19 secondary
/// `== 3 ⇒ += 1` adjustment (so a raw `3` becomes `4`) is applied
/// literally to both the Y and UV secondary strengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdefParams {
    /// `CdefDamping` = `cdef_damping_minus_3 + 3`. `3` on the
    /// short-circuit path.
    pub cdef_damping: u8,
    /// `cdef_bits` (`f(2)`). `0` on the short-circuit path. The number
    /// of valid strength entries is `1 << cdef_bits`.
    pub cdef_bits: u8,
    /// `cdef_y_pri_strength[0..(1 << cdef_bits)]`. Each `f(4)`.
    pub cdef_y_pri_strength: [u8; CDEF_MAX_STRENGTHS],
    /// `cdef_y_sec_strength[0..(1 << cdef_bits)]`. Each `f(2)`, with the
    /// §5.9.19 `== 3 ⇒ += 1` adjustment so a stored value is in
    /// `{0, 1, 2, 4}`.
    pub cdef_y_sec_strength: [u8; CDEF_MAX_STRENGTHS],
    /// `cdef_uv_pri_strength[0..(1 << cdef_bits)]`. Each `f(4)`. All 0
    /// when `NumPlanes == 1` (monochrome) — the §5.9.19 chroma reads
    /// are gated on `NumPlanes > 1`.
    pub cdef_uv_pri_strength: [u8; CDEF_MAX_STRENGTHS],
    /// `cdef_uv_sec_strength[0..(1 << cdef_bits)]`. Each `f(2)` with the
    /// same `== 3 ⇒ += 1` adjustment. All 0 when `NumPlanes == 1`.
    pub cdef_uv_sec_strength: [u8; CDEF_MAX_STRENGTHS],
    /// Whether the §5.9.19 short-circuit (`CodedLossless ||
    /// allow_intrabc || !enable_cdef`) fired and the parser returned
    /// without reading any bits.
    pub short_circuited: bool,
}

impl CdefParams {
    /// The §5.9.19 short-circuit-path output (no bits consumed):
    /// `cdef_bits = 0`, `CdefDamping = 3`, all strengths 0.
    pub const fn short_circuit() -> Self {
        Self {
            cdef_damping: 3,
            cdef_bits: 0,
            cdef_y_pri_strength: [0; CDEF_MAX_STRENGTHS],
            cdef_y_sec_strength: [0; CDEF_MAX_STRENGTHS],
            cdef_uv_pri_strength: [0; CDEF_MAX_STRENGTHS],
            cdef_uv_sec_strength: [0; CDEF_MAX_STRENGTHS],
            short_circuited: true,
        }
    }
}

/// Parse `cdef_params()` per §5.9.19 from a raw byte slice.
///
/// `num_planes` is §5.5.2's `NumPlanes` derived value (1 for
/// monochrome, otherwise 3). `coded_lossless`, `allow_intrabc`, and
/// `enable_cdef` are the runtime / sequence-header flags from §5.9.2 /
/// §5.5.1 that select the short-circuit path; standalone callers that
/// want the full bitstream path can pass `coded_lossless = false,
/// allow_intrabc = false, enable_cdef = true`.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_cdef_params(
    payload: &[u8],
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
    enable_cdef: bool,
) -> Result<(CdefParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let cdef = read_cdef_params(
        &mut br,
        num_planes,
        coded_lossless,
        allow_intrabc,
        enable_cdef,
    )?;
    Ok((cdef, br.position()))
}

pub(crate) fn read_cdef_params(
    br: &mut BitReader<'_>,
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
    enable_cdef: bool,
) -> Result<CdefParams, Error> {
    // §5.9.19 short-circuit: no bits read, all strengths zero.
    if coded_lossless || allow_intrabc || !enable_cdef {
        return Ok(CdefParams::short_circuit());
    }

    let cdef_damping_minus_3 = br.f(2)? as u8;
    let cdef_damping = cdef_damping_minus_3 + 3;
    let cdef_bits = br.f(2)? as u8;

    let mut cdef_y_pri_strength = [0u8; CDEF_MAX_STRENGTHS];
    let mut cdef_y_sec_strength = [0u8; CDEF_MAX_STRENGTHS];
    let mut cdef_uv_pri_strength = [0u8; CDEF_MAX_STRENGTHS];
    let mut cdef_uv_sec_strength = [0u8; CDEF_MAX_STRENGTHS];

    // `1 << cdef_bits` ≤ 8 (cdef_bits is f(2) ⇒ 0..=3).
    let count = 1usize << cdef_bits;
    for i in 0..count {
        cdef_y_pri_strength[i] = br.f(4)? as u8;
        let mut y_sec = br.f(2)? as u8;
        // §5.9.19: `if (cdef_y_sec_strength[i] == 3) += 1`.
        if y_sec == 3 {
            y_sec += 1;
        }
        cdef_y_sec_strength[i] = y_sec;
        if num_planes > 1 {
            cdef_uv_pri_strength[i] = br.f(4)? as u8;
            let mut uv_sec = br.f(2)? as u8;
            if uv_sec == 3 {
                uv_sec += 1;
            }
            cdef_uv_sec_strength[i] = uv_sec;
        }
    }

    Ok(CdefParams {
        cdef_damping,
        cdef_bits,
        cdef_y_pri_strength,
        cdef_y_sec_strength,
        cdef_uv_pri_strength,
        cdef_uv_sec_strength,
        short_circuited: false,
    })
}

// ---------------------------------------------------------------------
// §5.9.20 lr_params
// ---------------------------------------------------------------------

/// `RESTORATION_TILESIZE_MAX` per §3 — the maximum size (in luma
/// samples) of a loop restoration unit. `LoopRestorationSize[0]` is
/// derived as `RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift)`.
pub const RESTORATION_TILESIZE_MAX: u32 = 256;

/// The per-plane restoration type (§6.10.15 `FrameRestorationType`).
///
/// The §5.9.20 `lr_type` (`f(2)`) bitstream value is mapped through the
/// `Remap_Lr_Type[4]` lookup table into one of these. The numeric
/// discriminants are the §6.10.15 `FrameRestorationType` symbol values
/// (`RESTORE_NONE = 0`, `RESTORE_WIENER = 1`, `RESTORE_SGRPROJ = 2`,
/// `RESTORE_SWITCHABLE = 3`), which is also the encoding the
/// `LOOP_RESTORATION` trace lines log for `y_type` / `u_type` /
/// `v_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameRestorationType {
    /// `RESTORE_NONE` (0) — no loop restoration on this plane.
    None = 0,
    /// `RESTORE_WIENER` (1) — Wiener filter restoration.
    Wiener = 1,
    /// `RESTORE_SGRPROJ` (2) — self-guided projection restoration.
    SgrProj = 2,
    /// `RESTORE_SWITCHABLE` (3) — per-unit switchable restoration.
    Switchable = 3,
}

impl FrameRestorationType {
    /// `Remap_Lr_Type[lr_type]` per §5.9.20:
    ///
    /// ```text
    /// Remap_Lr_Type[4] = {
    ///   RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ
    /// }
    /// ```
    ///
    /// `lr_type` is read as `f(2)` so it is always in `0..=3`; the
    /// `_ => None` arm is unreachable for a conforming bitstream and
    /// exists only to keep the match total.
    #[must_use]
    pub const fn remap(lr_type: u8) -> Self {
        match lr_type {
            0 => FrameRestorationType::None,
            1 => FrameRestorationType::Switchable,
            2 => FrameRestorationType::Wiener,
            3 => FrameRestorationType::SgrProj,
            _ => FrameRestorationType::None,
        }
    }

    /// The §6.10.15 `FrameRestorationType` symbol value (0..=3).
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Parsed `lr_params()` per §5.9.20 + §6.10.15.
///
/// Loop-restoration parameters. When the §5.9.20 short-circuit fires
/// (`AllLossless || allow_intrabc || !enable_restoration`) the spec sets
/// `FrameRestorationType[0..3] = RESTORE_NONE`, `UsesLr = 0`, and
/// returns without reading any bits; [`Self::short_circuited`] records
/// that no bits were read.
///
/// Otherwise the parser reads one `lr_type` (`f(2)`) per plane
/// (`NumPlanes` of them), mapping each through `Remap_Lr_Type` into
/// [`FrameRestorationType`]. `UsesLr` is set if any plane uses
/// restoration; `usesChromaLr` is set if any chroma plane (`i > 0`)
/// does. When `UsesLr`, the loop-restoration unit size is read:
/// `lr_unit_shift` (`f(1)`, post-incremented for 128×128 superblocks;
/// otherwise extended by `lr_unit_extra_shift` `f(1)` when the first bit
/// is set), then `lr_uv_shift` (`f(1)`, only for 4:2:0 with chroma LR;
/// 0 otherwise). The three `LoopRestorationSize[]` entries are derived
/// from `RESTORATION_TILESIZE_MAX` and the two shifts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LrParams {
    /// `FrameRestorationType[0..NumPlanes]` (Y, U, V). Planes beyond
    /// `NumPlanes` (chroma on monochrome) stay at `RESTORE_NONE` — the
    /// §5.9.20 short-circuit and the monochrome `NumPlanes == 1` loop
    /// bound both leave them untouched.
    pub frame_restoration_type: [FrameRestorationType; 3],
    /// `UsesLr` — `true` when any plane uses loop restoration.
    pub uses_lr: bool,
    /// `usesChromaLr` — `true` when any chroma plane (`i > 0`) uses loop
    /// restoration. Gates the §5.9.20 `lr_uv_shift` read together with
    /// `subsampling_x && subsampling_y`.
    pub uses_chroma_lr: bool,
    /// `lr_unit_shift`. `0` when `!UsesLr`. For 128×128 superblocks the
    /// read `f(1)` is post-incremented (so `1..=2`); otherwise it is
    /// `0` or `1 + lr_unit_extra_shift` (`1..=2`). Feeds
    /// `LoopRestorationSize[0]`.
    pub lr_unit_shift: u8,
    /// `lr_uv_shift`. `0` unless `subsampling_x && subsampling_y &&
    /// usesChromaLr`, in which case it is the read `f(1)`.
    pub lr_uv_shift: u8,
    /// `LoopRestorationSize[0..3]` (Y, U, V) in plane samples.
    /// `LoopRestorationSize[0] = RESTORATION_TILESIZE_MAX >>
    /// (2 - lr_unit_shift)`; the two chroma entries are that value
    /// `>> lr_uv_shift`. All three are `0` on the short-circuit /
    /// `!UsesLr` path (no size is signalled).
    pub loop_restoration_size: [u32; 3],
    /// Whether the §5.9.20 short-circuit (`AllLossless || allow_intrabc
    /// || !enable_restoration`) fired and the parser returned without
    /// reading any bits.
    pub short_circuited: bool,
}

impl LrParams {
    /// The §5.9.20 short-circuit-path output (no bits consumed): all
    /// planes `RESTORE_NONE`, `UsesLr = 0`, shifts and sizes `0`.
    #[must_use]
    pub const fn short_circuit() -> Self {
        Self {
            frame_restoration_type: [FrameRestorationType::None; 3],
            uses_lr: false,
            uses_chroma_lr: false,
            lr_unit_shift: 0,
            lr_uv_shift: 0,
            loop_restoration_size: [0; 3],
            short_circuited: true,
        }
    }
}

/// Parse `lr_params()` per §5.9.20 from a raw byte slice.
///
/// `num_planes` is §5.5.2's `NumPlanes` derived value (1 for
/// monochrome, otherwise 3). `subsampling_x` / `subsampling_y` are the
/// §5.5.2 chroma-subsampling flags (both `true` for 4:2:0).
/// `use_128x128_superblock` is the §5.5.1 sequence-header flag.
/// `all_lossless`, `allow_intrabc`, and `enable_restoration` are the
/// §5.9.2 / §5.5.1 flags that select the short-circuit path; standalone
/// callers that want the full bitstream path can pass
/// `all_lossless = false, allow_intrabc = false,
/// enable_restoration = true`.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
#[allow(clippy::too_many_arguments)]
pub fn parse_lr_params(
    payload: &[u8],
    num_planes: u8,
    subsampling_x: bool,
    subsampling_y: bool,
    use_128x128_superblock: bool,
    all_lossless: bool,
    allow_intrabc: bool,
    enable_restoration: bool,
) -> Result<(LrParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let lr = read_lr_params(
        &mut br,
        num_planes,
        subsampling_x,
        subsampling_y,
        use_128x128_superblock,
        all_lossless,
        allow_intrabc,
        enable_restoration,
    )?;
    Ok((lr, br.position()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_lr_params(
    br: &mut BitReader<'_>,
    num_planes: u8,
    subsampling_x: bool,
    subsampling_y: bool,
    use_128x128_superblock: bool,
    all_lossless: bool,
    allow_intrabc: bool,
    enable_restoration: bool,
) -> Result<LrParams, Error> {
    // §5.9.20 short-circuit: no bits read, all planes RESTORE_NONE.
    if all_lossless || allow_intrabc || !enable_restoration {
        return Ok(LrParams::short_circuit());
    }

    let mut frame_restoration_type = [FrameRestorationType::None; 3];
    let mut uses_lr = false;
    let mut uses_chroma_lr = false;

    // §5.9.20: for ( i = 0; i < NumPlanes; i++ ).
    for (i, slot) in frame_restoration_type
        .iter_mut()
        .enumerate()
        .take(usize::from(num_planes))
    {
        let lr_type = br.f(2)? as u8;
        let rtype = FrameRestorationType::remap(lr_type);
        *slot = rtype;
        if rtype != FrameRestorationType::None {
            uses_lr = true;
            if i > 0 {
                uses_chroma_lr = true;
            }
        }
    }

    let mut lr_unit_shift = 0u8;
    let mut lr_uv_shift = 0u8;
    let mut loop_restoration_size = [0u32; 3];

    if uses_lr {
        // §5.9.20 lr_unit_shift derivation.
        if use_128x128_superblock {
            lr_unit_shift = br.f(1)? as u8;
            lr_unit_shift += 1;
        } else {
            lr_unit_shift = br.f(1)? as u8;
            if lr_unit_shift != 0 {
                let lr_unit_extra_shift = br.f(1)? as u8;
                lr_unit_shift += lr_unit_extra_shift;
            }
        }

        // §5.9.20: LoopRestorationSize[0] =
        //   RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift).
        // lr_unit_shift is in 0..=2 here, so the shift amount is 0..=2.
        loop_restoration_size[0] = RESTORATION_TILESIZE_MAX >> (2 - u32::from(lr_unit_shift));

        // §5.9.20: lr_uv_shift only signalled for 4:2:0 chroma LR.
        if subsampling_x && subsampling_y && uses_chroma_lr {
            lr_uv_shift = br.f(1)? as u8;
        } else {
            lr_uv_shift = 0;
        }

        loop_restoration_size[1] = loop_restoration_size[0] >> u32::from(lr_uv_shift);
        loop_restoration_size[2] = loop_restoration_size[0] >> u32::from(lr_uv_shift);
    }

    Ok(LrParams {
        frame_restoration_type,
        uses_lr,
        uses_chroma_lr,
        lr_unit_shift,
        lr_uv_shift,
        loop_restoration_size,
        short_circuited: false,
    })
}

// ---------------------------------------------------------------------
// §5.9.21 read_tx_mode
// ---------------------------------------------------------------------

/// `TX_MODES` per §3 — the number of distinct `TxMode` values.
pub const TX_MODES: u8 = 3;

/// `TxMode` per §5.9.21 + §6.8.21.
///
/// Specifies how the per-block transform size is determined. The numeric
/// discriminants are the §6.8.21 `TxMode` symbol values, which is also
/// the encoding the `FRAME_HEADER` trace lines log for the `tx_mode`
/// column:
///
/// ```text
///   0  ONLY_4X4
///   1  TX_MODE_LARGEST
///   2  TX_MODE_SELECT
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TxMode {
    /// `ONLY_4X4` (0) — the inverse transform uses only 4×4 transforms.
    /// Selected unconditionally when `CodedLossless == 1`.
    Only4x4 = 0,
    /// `TX_MODE_LARGEST` (1) — the inverse transform uses the largest
    /// transform size that fits inside the block.
    TxModeLargest = 1,
    /// `TX_MODE_SELECT` (2) — the transform size is signalled explicitly
    /// per block.
    TxModeSelect = 2,
}

impl TxMode {
    /// The §6.8.21 `TxMode` symbol value (0..=2).
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Parse `read_tx_mode()` per §5.9.21 from a raw byte slice.
///
/// When `coded_lossless` is `true` the §5.9.21 first branch fires: no
/// bits are read and `TxMode = ONLY_4X4`. Otherwise the parser reads
/// `tx_mode_select` (`f(1)`): `1` ⇒ `TX_MODE_SELECT`, `0` ⇒
/// `TX_MODE_LARGEST`.
///
/// `coded_lossless` is the §5.9.2 `CodedLossless` derived value
/// (see [`crate::frame_header`]'s `compute_coded_lossless`).
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_tx_mode(payload: &[u8], coded_lossless: bool) -> Result<(TxMode, usize), Error> {
    let mut br = BitReader::new(payload);
    let tx_mode = read_tx_mode(&mut br, coded_lossless)?;
    Ok((tx_mode, br.position()))
}

pub(crate) fn read_tx_mode(br: &mut BitReader<'_>, coded_lossless: bool) -> Result<TxMode, Error> {
    // §5.9.21:
    //   if ( CodedLossless == 1 ) {
    //       TxMode = ONLY_4X4
    //   } else {
    //       tx_mode_select  f(1)
    //       TxMode = tx_mode_select ? TX_MODE_SELECT : TX_MODE_LARGEST
    //   }
    if coded_lossless {
        return Ok(TxMode::Only4x4);
    }
    let tx_mode_select = br.f(1)? == 1;
    Ok(if tx_mode_select {
        TxMode::TxModeSelect
    } else {
        TxMode::TxModeLargest
    })
}

// ---------------------------------------------------------------------
// §5.9.24 global_motion_params + §5.9.25 read_global_param +
// §5.9.26–§5.9.29 decode_signed_subexp_with_ref helpers
// ---------------------------------------------------------------------

/// `REFS_PER_FRAME` per §3 — the number of inter-prediction reference
/// frames a frame may use (`LAST_FRAME`..`ALTREF_FRAME`, i.e. 7). This
/// is the length of the per-ref global-motion arrays.
pub const REFS_PER_FRAME: usize = 7;

/// `INTRA_FRAME` per §3 — the implicit reference-frame index 0. The
/// inter references `LAST_FRAME`..`ALTREF_FRAME` are 1..=7.
pub const INTRA_FRAME: usize = 0;

/// `LAST_FRAME` per §3 — the first inter-prediction reference index.
pub const LAST_FRAME: usize = 1;

/// `LAST2_FRAME` per §3 — second forward-direction inter reference.
pub const LAST2_FRAME: usize = 2;

/// `LAST3_FRAME` per §3 — third forward-direction inter reference.
pub const LAST3_FRAME: usize = 3;

/// `GOLDEN_FRAME` per §3 — fourth forward-direction inter reference.
pub const GOLDEN_FRAME: usize = 4;

/// `BWDREF_FRAME` per §3 — first backward-direction inter reference.
/// `ref >= BWDREF_FRAME && ref <= ALTREF_FRAME` is the §8.3.2
/// `check_backward()` predicate used by the `comp_mode` ctx walk.
pub const BWDREF_FRAME: usize = 5;

/// `ALTREF2_FRAME` per §3 — second backward-direction inter reference.
pub const ALTREF2_FRAME: usize = 6;

/// `ALTREF_FRAME` per §3 — the last inter-prediction reference index.
pub const ALTREF_FRAME: usize = 7;

/// `SINGLE_REFERENCE` per §3 — `comp_mode = 0` ⇒ block uses one ref.
pub const SINGLE_REFERENCE: u8 = 0;

/// `COMPOUND_REFERENCE` per §3 — `comp_mode = 1` ⇒ block uses two refs.
pub const COMPOUND_REFERENCE: u8 = 1;

/// `UNIDIR_COMP_REFERENCE` per §3 — `comp_ref_type = 0` ⇒ both
/// reference frames come from the same direction group (forward or
/// backward).
pub const UNIDIR_COMP_REFERENCE: u8 = 0;

/// `BIDIR_COMP_REFERENCE` per §3 — `comp_ref_type = 1` ⇒ one
/// reference frame from each direction group.
pub const BIDIR_COMP_REFERENCE: u8 = 1;

/// `WARPEDMODEL_PREC_BITS` per §3 — internal precision of warped-motion
/// models. The §5.9.24 identity default sets `gm_params[ref][i]` to
/// `1 << WARPEDMODEL_PREC_BITS` for the two diagonal slots (`i % 3 == 2`).
pub const WARPEDMODEL_PREC_BITS: u32 = 16;

/// `GM_ABS_TRANS_BITS` per §3 — number of bits for the translational
/// components of a ROTZOOM / AFFINE model.
pub const GM_ABS_TRANS_BITS: u32 = 12;

/// `GM_ABS_TRANS_ONLY_BITS` per §3 — number of bits for the
/// translational components of a TRANSLATION model.
pub const GM_ABS_TRANS_ONLY_BITS: u32 = 9;

/// `GM_ABS_ALPHA_BITS` per §3 — number of bits for the
/// non-translational components of a global-motion model.
pub const GM_ABS_ALPHA_BITS: u32 = 12;

/// `GM_ALPHA_PREC_BITS` per §3 — fractional bits for non-translational
/// warp-model coefficients.
pub const GM_ALPHA_PREC_BITS: u32 = 15;

/// `GM_TRANS_PREC_BITS` per §3 — fractional bits for translational
/// warp-model coefficients (ROTZOOM / AFFINE).
pub const GM_TRANS_PREC_BITS: u32 = 6;

/// `GM_TRANS_ONLY_PREC_BITS` per §3 — fractional bits for the
/// translational components of a pure-TRANSLATION warp.
pub const GM_TRANS_ONLY_PREC_BITS: u32 = 3;

/// Warp-model type per §6.8.18 (`GmType[ref]`). The numeric
/// discriminants are the §3 `IDENTITY` / `TRANSLATION` / `ROTZOOM` /
/// `AFFINE` symbol values; the §5.9.24 `type >= ROTZOOM` /
/// `type >= TRANSLATION` comparisons depend on this exact ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum WarpModelType {
    /// `IDENTITY` (0) — no warp; the per-ref default and the value used
    /// for every ref on the §5.9.24 `FrameIsIntra` short-circuit.
    Identity = 0,
    /// `TRANSLATION` (1) — a pure translation (only `gm_params[ref][0]`
    /// / `[1]` are read).
    Translation = 1,
    /// `ROTZOOM` (2) — rotation + symmetric zoom + translation
    /// (`gm_params[ref][2..=3]` plus the derived `[4]` / `[5]`).
    RotZoom = 2,
    /// `AFFINE` (3) — a general affine transform (`gm_params[ref][2..=5]`).
    Affine = 3,
}

impl WarpModelType {
    /// The §6.8.18 `GmType` symbol value (0..=3).
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }
}

/// Parsed `global_motion_params()` per §5.9.24 + §6.8.18.
///
/// `gm_type[i]` / `gm_params[i][..]` are indexed by **reference-frame
/// index** (0..=7), matching the spec's `GmType[ref]` /
/// `gm_params[ref][i]` arrays where `ref` ranges over
/// `LAST_FRAME`..`ALTREF_FRAME`. Index `INTRA_FRAME = 0` is never
/// written by §5.9.24 and stays at the identity default.
///
/// On the §5.9.24 `FrameIsIntra` short-circuit every ref keeps the
/// identity initialiser (`GmType = IDENTITY`, the diagonal
/// `gm_params[ref][2] = gm_params[ref][5] = 1 << WARPEDMODEL_PREC_BITS`,
/// every other slot `0`) and no bits are read;
/// [`Self::short_circuited`] records this.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalMotionParams {
    /// `GmType[ref]` for `ref` in `0..=ALTREF_FRAME`. Index 0
    /// (`INTRA_FRAME`) is always `IDENTITY`.
    pub gm_type: [WarpModelType; TOTAL_REFS_PER_FRAME],
    /// `gm_params[ref][0..6]` for `ref` in `0..=ALTREF_FRAME`. The six
    /// warp-model coefficients are stored at `WARPEDMODEL_PREC_BITS`
    /// internal precision (i.e. the value `read_global_param` writes,
    /// before any §7.13.1 setup). The identity model is
    /// `[0, 0, 1<<16, 0, 0, 1<<16]`.
    pub gm_params: [[i32; 6]; TOTAL_REFS_PER_FRAME],
    /// Whether the §5.9.24 `FrameIsIntra` short-circuit fired and the
    /// parser returned the identity defaults without reading any bits.
    pub short_circuited: bool,
}

impl GlobalMotionParams {
    /// The §5.9.24 identity initialiser: every ref `IDENTITY`, with
    /// `gm_params[ref][i] = (i % 3 == 2) ? 1 << WARPEDMODEL_PREC_BITS :
    /// 0`. This is also the output on the `FrameIsIntra` short-circuit.
    #[must_use]
    pub const fn identity() -> Self {
        let one = 1i32 << WARPEDMODEL_PREC_BITS;
        Self {
            gm_type: [WarpModelType::Identity; TOTAL_REFS_PER_FRAME],
            gm_params: [[0, 0, one, 0, 0, one]; TOTAL_REFS_PER_FRAME],
            short_circuited: true,
        }
    }
}

/// The §7.13.1 default `PrevGmParams[ref][idx]` (i.e. the
/// `setup_past_independence` initialiser): `(idx % 3 == 2) ? 1 <<
/// WARPEDMODEL_PREC_BITS : 0`. Used as the prediction reference for
/// `read_global_param` when the frame has no previous-frame state
/// (the only path the streaming parser exercises — intra frames with
/// `primary_ref_frame == PRIMARY_REF_NONE`, which short-circuit before
/// any `read_global_param` call anyway).
#[must_use]
pub const fn prev_gm_params_default() -> [[i32; 6]; TOTAL_REFS_PER_FRAME] {
    let one = 1i32 << WARPEDMODEL_PREC_BITS;
    [[0, 0, one, 0, 0, one]; TOTAL_REFS_PER_FRAME]
}

/// `inverse_recenter(r, v)` per §5.9.29.
fn inverse_recenter(r: i64, v: i64) -> i64 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `decode_subexp(numSyms)` per §5.9.28.
///
/// `k = 3` is the §5.9.28 fixed sub-exponential parameter. The loop
/// reads `subexp_more_bits` (`f(1)`) until the remaining range fits,
/// then either `subexp_final_bits` (`ns(numSyms - mk)`) or
/// `subexp_bits` (`f(b2)`).
fn decode_subexp(br: &mut BitReader<'_>, num_syms: u32) -> Result<u32, Error> {
    let mut i: u32 = 0;
    let mut mk: u32 = 0;
    let k: u32 = 3;
    loop {
        let b2 = if i != 0 { k + i - 1 } else { k };
        let a = 1u32 << b2;
        if num_syms <= mk + 3 * a {
            let subexp_final_bits = br.ns(num_syms - mk)?;
            return Ok(subexp_final_bits + mk);
        }
        let subexp_more_bits = br.f(1)? == 1;
        if subexp_more_bits {
            i += 1;
            mk += a;
        } else {
            let subexp_bits = br.f(b2)? as u32;
            return Ok(subexp_bits + mk);
        }
    }
}

/// `decode_unsigned_subexp_with_ref(mx, r)` per §5.9.27. Returns a
/// value in `0..mx`.
fn decode_unsigned_subexp_with_ref(br: &mut BitReader<'_>, mx: i64, r: i64) -> Result<i64, Error> {
    // `mx` is bounded by `2 * (1 << absBits) + 1 <= 2 * 4096 + 1`, so it
    // fits in u32 for the `decode_subexp` call.
    let v = i64::from(decode_subexp(br, mx as u32)?);
    if (r << 1) <= mx {
        Ok(inverse_recenter(r, v))
    } else {
        Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
    }
}

/// `decode_signed_subexp_with_ref(low, high, r)` per §5.9.26. Returns a
/// value in `low..high`.
fn decode_signed_subexp_with_ref(
    br: &mut BitReader<'_>,
    low: i64,
    high: i64,
    r: i64,
) -> Result<i64, Error> {
    let x = decode_unsigned_subexp_with_ref(br, high - low, r - low)?;
    Ok(x + low)
}

/// `read_global_param(type, ref, idx)` per §5.9.25.
///
/// Reads one warp-model coefficient into `gm_params[ref][idx]`, using
/// `prev_gm_params[ref][idx]` (the spec's `PrevGmParams`) as the
/// sub-exponential prediction reference. `allow_high_precision_mv`
/// adjusts the translational `idx < 2` precision per §5.9.25.
#[allow(clippy::too_many_arguments)]
fn read_global_param(
    br: &mut BitReader<'_>,
    gm_type: WarpModelType,
    gm_params: &mut [[i32; 6]; TOTAL_REFS_PER_FRAME],
    prev_gm_params: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
    allow_high_precision_mv: bool,
    ref_idx: usize,
    idx: usize,
) -> Result<(), Error> {
    let not_high_prec: u32 = u32::from(!allow_high_precision_mv);
    let (abs_bits, prec_bits) = if idx < 2 {
        if gm_type == WarpModelType::Translation {
            (
                GM_ABS_TRANS_ONLY_BITS - not_high_prec,
                GM_TRANS_ONLY_PREC_BITS - not_high_prec,
            )
        } else {
            (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
        }
    } else {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    };
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    let round: i64 = if idx % 3 == 2 {
        1i64 << WARPEDMODEL_PREC_BITS
    } else {
        0
    };
    let sub: i64 = if idx % 3 == 2 { 1i64 << prec_bits } else { 0 };
    let mx: i64 = 1i64 << abs_bits;
    let r: i64 = (i64::from(prev_gm_params[ref_idx][idx]) >> prec_diff) - sub;
    let decoded = decode_signed_subexp_with_ref(br, -mx, mx + 1, r)?;
    let value = (decoded << prec_diff) + round;
    gm_params[ref_idx][idx] = value as i32;
    Ok(())
}

/// Parse `global_motion_params()` per §5.9.24 from a raw byte slice.
///
/// `frame_is_intra` selects the §5.9.24 `FrameIsIntra` short-circuit
/// (identity defaults, no bits). `allow_high_precision_mv` and
/// `prev_gm_params` feed `read_global_param` on the full (inter) path;
/// standalone callers exercising the full syntax should pass the
/// `PrevGmParams` defaults via [`prev_gm_params_default`].
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_global_motion_params(
    payload: &[u8],
    frame_is_intra: bool,
    allow_high_precision_mv: bool,
    prev_gm_params: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
) -> Result<(GlobalMotionParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let gm = read_global_motion_params(
        &mut br,
        frame_is_intra,
        allow_high_precision_mv,
        prev_gm_params,
    )?;
    Ok((gm, br.position()))
}

pub(crate) fn read_global_motion_params(
    br: &mut BitReader<'_>,
    frame_is_intra: bool,
    allow_high_precision_mv: bool,
    prev_gm_params: &[[i32; 6]; TOTAL_REFS_PER_FRAME],
) -> Result<GlobalMotionParams, Error> {
    // §5.9.24 initialiser: every ref IDENTITY with the diagonal defaults.
    let mut gm = GlobalMotionParams::identity();
    if frame_is_intra {
        // `gm.short_circuited` is already `true` from `identity()`.
        return Ok(gm);
    }
    gm.short_circuited = false;

    for ref_idx in LAST_FRAME..=ALTREF_FRAME {
        let is_global = br.f(1)? == 1;
        let gm_type = if is_global {
            let is_rot_zoom = br.f(1)? == 1;
            if is_rot_zoom {
                WarpModelType::RotZoom
            } else {
                let is_translation = br.f(1)? == 1;
                if is_translation {
                    WarpModelType::Translation
                } else {
                    WarpModelType::Affine
                }
            }
        } else {
            WarpModelType::Identity
        };
        gm.gm_type[ref_idx] = gm_type;

        if gm_type as u8 >= WarpModelType::RotZoom as u8 {
            read_global_param(
                br,
                gm_type,
                &mut gm.gm_params,
                prev_gm_params,
                allow_high_precision_mv,
                ref_idx,
                2,
            )?;
            read_global_param(
                br,
                gm_type,
                &mut gm.gm_params,
                prev_gm_params,
                allow_high_precision_mv,
                ref_idx,
                3,
            )?;
            if gm_type == WarpModelType::Affine {
                read_global_param(
                    br,
                    gm_type,
                    &mut gm.gm_params,
                    prev_gm_params,
                    allow_high_precision_mv,
                    ref_idx,
                    4,
                )?;
                read_global_param(
                    br,
                    gm_type,
                    &mut gm.gm_params,
                    prev_gm_params,
                    allow_high_precision_mv,
                    ref_idx,
                    5,
                )?;
            } else {
                // §5.9.24: ROTZOOM derives [4] / [5] from [3] / [2].
                gm.gm_params[ref_idx][4] = -gm.gm_params[ref_idx][3];
                gm.gm_params[ref_idx][5] = gm.gm_params[ref_idx][2];
            }
        }
        if gm_type as u8 >= WarpModelType::Translation as u8 {
            read_global_param(
                br,
                gm_type,
                &mut gm.gm_params,
                prev_gm_params,
                allow_high_precision_mv,
                ref_idx,
                0,
            )?;
            read_global_param(
                br,
                gm_type,
                &mut gm.gm_params,
                prev_gm_params,
                allow_high_precision_mv,
                ref_idx,
                1,
            )?;
        }
    }
    Ok(gm)
}

// ---------------------------------------------------------------------
// §5.9.30 film_grain_params
// ---------------------------------------------------------------------

/// Maximum number of luma scaling-function points. `num_y_points` is
/// `f(4)` (0..=15) but §6.8.20 conformance caps it at 14; the array is
/// sized at 14.
pub const MAX_NUM_Y_POINTS: usize = 14;

/// Maximum number of chroma scaling-function points per plane.
/// `num_cb_points` / `num_cr_points` are `f(4)` and §6.8.20-capped at
/// 10.
pub const MAX_NUM_CHROMA_POINTS: usize = 10;

/// Maximum number of luma auto-regressive coefficients:
/// `numPosLuma = 2 * ar_coeff_lag * (ar_coeff_lag + 1)` with
/// `ar_coeff_lag` in `0..=3` ⇒ `2 * 3 * 4 = 24`.
pub const MAX_AR_COEFFS_Y: usize = 24;

/// Maximum number of chroma auto-regressive coefficients:
/// `numPosChroma = numPosLuma + 1 = 25` when `num_y_points > 0`.
pub const MAX_AR_COEFFS_UV: usize = 25;

/// Parsed `film_grain_params()` per §5.9.30 + §6.8.20.
///
/// The `apply_grain == 0` / `!film_grain_params_present` /
/// hidden-frame short-circuits all reduce to the
/// `reset_grain_params()` output, which §6.8.20 defines as every field
/// set to `0`; [`Self::apply_grain`] then reads `false`. On the
/// `update_grain == 0` predicted path only `apply_grain`, `grain_seed`,
/// `update_grain`, and `film_grain_params_ref_idx` are populated from
/// the bitstream (the remaining fields would be loaded from the
/// referenced frame via `load_grain_params` — out of scope for a
/// stateless header parser, so they stay at their defaults and
/// [`Self::predicted`] records the predicted path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilmGrainParams {
    /// `apply_grain` (`f(1)`). `false` on every short-circuit path.
    pub apply_grain: bool,
    /// `grain_seed` (`f(16)`).
    pub grain_seed: u16,
    /// `update_grain` (`f(1)` for `INTER_FRAME`, else derived `1`).
    pub update_grain: bool,
    /// `film_grain_params_ref_idx` (`f(3)`) — only meaningful when
    /// [`Self::predicted`] is `true`.
    pub film_grain_params_ref_idx: u8,
    /// `num_y_points` (`f(4)`).
    pub num_y_points: u8,
    /// `point_y_value[i]` / `point_y_scaling[i]` (`f(8)` each), the
    /// first `num_y_points` entries valid.
    pub point_y_value: [u8; MAX_NUM_Y_POINTS],
    /// See [`Self::point_y_value`].
    pub point_y_scaling: [u8; MAX_NUM_Y_POINTS],
    /// `chroma_scaling_from_luma` (`f(1)`, `0` when `mono_chrome`).
    pub chroma_scaling_from_luma: bool,
    /// `num_cb_points` (`f(4)`), `0` on the §5.9.30 chroma-suppression
    /// branch (`mono_chrome || chroma_scaling_from_luma || (4:2:0 &&
    /// num_y_points == 0)`).
    pub num_cb_points: u8,
    /// `point_cb_value[i]` / `point_cb_scaling[i]` (`f(8)` each).
    pub point_cb_value: [u8; MAX_NUM_CHROMA_POINTS],
    /// See [`Self::point_cb_value`].
    pub point_cb_scaling: [u8; MAX_NUM_CHROMA_POINTS],
    /// `num_cr_points` (`f(4)`).
    pub num_cr_points: u8,
    /// `point_cr_value[i]` / `point_cr_scaling[i]` (`f(8)` each).
    pub point_cr_value: [u8; MAX_NUM_CHROMA_POINTS],
    /// See [`Self::point_cr_value`].
    pub point_cr_scaling: [u8; MAX_NUM_CHROMA_POINTS],
    /// `GrainScaling = grain_scaling_minus_8 + 8` (`f(2)` ⇒ 8..=11).
    pub grain_scaling: u8,
    /// `ar_coeff_lag` (`f(2)` ⇒ 0..=3).
    pub ar_coeff_lag: u8,
    /// `ar_coeffs_y_plus_128[i]` (`f(8)`), the first `numPosLuma`
    /// entries valid (only when `num_y_points > 0`).
    pub ar_coeffs_y_plus_128: [u8; MAX_AR_COEFFS_Y],
    /// `ar_coeffs_cb_plus_128[i]` (`f(8)`), the first `numPosChroma`
    /// entries valid (when `chroma_scaling_from_luma || num_cb_points`).
    pub ar_coeffs_cb_plus_128: [u8; MAX_AR_COEFFS_UV],
    /// `ar_coeffs_cr_plus_128[i]` (`f(8)`), the first `numPosChroma`
    /// entries valid (when `chroma_scaling_from_luma || num_cr_points`).
    pub ar_coeffs_cr_plus_128: [u8; MAX_AR_COEFFS_UV],
    /// `ArCoeffShift = ar_coeff_shift_minus_6 + 6` (`f(2)` ⇒ 6..=9).
    pub ar_coeff_shift: u8,
    /// `grain_scale_shift` (`f(2)` ⇒ 0..=3).
    pub grain_scale_shift: u8,
    /// `cb_mult` (`f(8)`), valid when `num_cb_points > 0`.
    pub cb_mult: u8,
    /// `cb_luma_mult` (`f(8)`).
    pub cb_luma_mult: u8,
    /// `cb_offset` (`f(9)` ⇒ 0..=511).
    pub cb_offset: u16,
    /// `cr_mult` (`f(8)`), valid when `num_cr_points > 0`.
    pub cr_mult: u8,
    /// `cr_luma_mult` (`f(8)`).
    pub cr_luma_mult: u8,
    /// `cr_offset` (`f(9)` ⇒ 0..=511).
    pub cr_offset: u16,
    /// `overlap_flag` (`f(1)`).
    pub overlap_flag: bool,
    /// `clip_to_restricted_range` (`f(1)`).
    pub clip_to_restricted_range: bool,
    /// Whether the §5.9.30 `update_grain == 0` predicted path fired
    /// (`film_grain_params_ref_idx` valid; the AR / scaling fields are
    /// inherited from the referenced frame, not present in this
    /// header's bitstream).
    pub predicted: bool,
}

impl FilmGrainParams {
    /// The §6.8.20 `reset_grain_params()` output: every field `0` /
    /// `false`.
    #[must_use]
    pub const fn reset() -> Self {
        Self {
            apply_grain: false,
            grain_seed: 0,
            update_grain: false,
            film_grain_params_ref_idx: 0,
            num_y_points: 0,
            point_y_value: [0; MAX_NUM_Y_POINTS],
            point_y_scaling: [0; MAX_NUM_Y_POINTS],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            point_cb_value: [0; MAX_NUM_CHROMA_POINTS],
            point_cb_scaling: [0; MAX_NUM_CHROMA_POINTS],
            num_cr_points: 0,
            point_cr_value: [0; MAX_NUM_CHROMA_POINTS],
            point_cr_scaling: [0; MAX_NUM_CHROMA_POINTS],
            grain_scaling: 0,
            ar_coeff_lag: 0,
            ar_coeffs_y_plus_128: [0; MAX_AR_COEFFS_Y],
            ar_coeffs_cb_plus_128: [0; MAX_AR_COEFFS_UV],
            ar_coeffs_cr_plus_128: [0; MAX_AR_COEFFS_UV],
            ar_coeff_shift: 0,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap_flag: false,
            clip_to_restricted_range: false,
            predicted: false,
        }
    }
}

/// Inputs to `film_grain_params()` that come from the §5.9.2 per-frame
/// state and the §5.5.x sequence header. Bundled so the standalone
/// parser and the streaming wiring share one shape.
#[derive(Debug, Clone, Copy)]
pub struct FilmGrainContext {
    /// `film_grain_params_present` (§5.5.2 sequence-header flag).
    pub film_grain_params_present: bool,
    /// `show_frame` (§5.9.2).
    pub show_frame: bool,
    /// `showable_frame` (§5.9.2).
    pub showable_frame: bool,
    /// `frame_type == INTER_FRAME` — gates the `update_grain` read.
    pub is_inter_frame: bool,
    /// `mono_chrome` (§5.5.2).
    pub mono_chrome: bool,
    /// `subsampling_x` (§5.5.2).
    pub subsampling_x: bool,
    /// `subsampling_y` (§5.5.2).
    pub subsampling_y: bool,
}

/// Parse `film_grain_params()` per §5.9.30 from a raw byte slice.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_film_grain_params(
    payload: &[u8],
    ctx: FilmGrainContext,
) -> Result<(FilmGrainParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let fg = read_film_grain_params(&mut br, ctx)?;
    Ok((fg, br.position()))
}

pub(crate) fn read_film_grain_params(
    br: &mut BitReader<'_>,
    ctx: FilmGrainContext,
) -> Result<FilmGrainParams, Error> {
    // §5.9.30: short-circuit when film grain is disabled at the
    // sequence level or the frame is neither shown nor showable.
    if !ctx.film_grain_params_present || (!ctx.show_frame && !ctx.showable_frame) {
        return Ok(FilmGrainParams::reset());
    }

    let apply_grain = br.f(1)? == 1;
    if !apply_grain {
        return Ok(FilmGrainParams::reset());
    }

    let mut fg = FilmGrainParams::reset();
    fg.apply_grain = true;

    fg.grain_seed = br.f(16)? as u16;

    // §5.9.30: update_grain is `f(1)` only for INTER_FRAME; derived 1
    // otherwise.
    fg.update_grain = if ctx.is_inter_frame {
        br.f(1)? == 1
    } else {
        true
    };

    if !fg.update_grain {
        // Predicted path: read the ref idx, then inherit the rest from
        // the referenced frame (load_grain_params) — which a stateless
        // header parser cannot resolve. grain_seed is preserved
        // (tempGrainSeed dance) — and we already stored it above.
        fg.film_grain_params_ref_idx = br.f(3)? as u8;
        fg.predicted = true;
        return Ok(fg);
    }

    fg.num_y_points = br.f(4)? as u8;
    // §5.9.30 conformance: `num_y_points <= 14`. The `f(4)` literal can
    // code 15 on a corrupt / adversarial stream, which would index past
    // the `[u8; MAX_NUM_Y_POINTS]` arrays — reject instead.
    if usize::from(fg.num_y_points) > MAX_NUM_Y_POINTS {
        return Err(Error::FilmGrainPointCountOverflow);
    }
    for i in 0..usize::from(fg.num_y_points) {
        fg.point_y_value[i] = br.f(8)? as u8;
        fg.point_y_scaling[i] = br.f(8)? as u8;
    }

    fg.chroma_scaling_from_luma = if ctx.mono_chrome {
        false
    } else {
        br.f(1)? == 1
    };

    // §5.9.30: the chroma-points suppression branch.
    let suppress_chroma = ctx.mono_chrome
        || fg.chroma_scaling_from_luma
        || (ctx.subsampling_x && ctx.subsampling_y && fg.num_y_points == 0);
    if suppress_chroma {
        fg.num_cb_points = 0;
        fg.num_cr_points = 0;
    } else {
        fg.num_cb_points = br.f(4)? as u8;
        // §5.9.30 conformance: `num_cb_points <= 10`.
        if usize::from(fg.num_cb_points) > MAX_NUM_CHROMA_POINTS {
            return Err(Error::FilmGrainPointCountOverflow);
        }
        for i in 0..usize::from(fg.num_cb_points) {
            fg.point_cb_value[i] = br.f(8)? as u8;
            fg.point_cb_scaling[i] = br.f(8)? as u8;
        }
        fg.num_cr_points = br.f(4)? as u8;
        // §5.9.30 conformance: `num_cr_points <= 10`.
        if usize::from(fg.num_cr_points) > MAX_NUM_CHROMA_POINTS {
            return Err(Error::FilmGrainPointCountOverflow);
        }
        for i in 0..usize::from(fg.num_cr_points) {
            fg.point_cr_value[i] = br.f(8)? as u8;
            fg.point_cr_scaling[i] = br.f(8)? as u8;
        }
    }

    let grain_scaling_minus_8 = br.f(2)? as u8;
    fg.grain_scaling = grain_scaling_minus_8 + 8;
    fg.ar_coeff_lag = br.f(2)? as u8;

    let num_pos_luma = 2 * usize::from(fg.ar_coeff_lag) * (usize::from(fg.ar_coeff_lag) + 1);
    let num_pos_chroma = if fg.num_y_points > 0 {
        for i in 0..num_pos_luma {
            fg.ar_coeffs_y_plus_128[i] = br.f(8)? as u8;
        }
        num_pos_luma + 1
    } else {
        num_pos_luma
    };

    if fg.chroma_scaling_from_luma || fg.num_cb_points > 0 {
        for i in 0..num_pos_chroma {
            fg.ar_coeffs_cb_plus_128[i] = br.f(8)? as u8;
        }
    }
    if fg.chroma_scaling_from_luma || fg.num_cr_points > 0 {
        for i in 0..num_pos_chroma {
            fg.ar_coeffs_cr_plus_128[i] = br.f(8)? as u8;
        }
    }

    let ar_coeff_shift_minus_6 = br.f(2)? as u8;
    fg.ar_coeff_shift = ar_coeff_shift_minus_6 + 6;
    fg.grain_scale_shift = br.f(2)? as u8;

    if fg.num_cb_points > 0 {
        fg.cb_mult = br.f(8)? as u8;
        fg.cb_luma_mult = br.f(8)? as u8;
        fg.cb_offset = br.f(9)? as u16;
    }
    if fg.num_cr_points > 0 {
        fg.cr_mult = br.f(8)? as u8;
        fg.cr_luma_mult = br.f(8)? as u8;
        fg.cr_offset = br.f(9)? as u16;
    }

    fg.overlap_flag = br.f(1)? == 1;
    fg.clip_to_restricted_range = br.f(1)? == 1;

    Ok(fg)
}

/// `Clip3(a, b, x)` per §"Conventions": clamp `x` to `[a, b]`.
fn clip3_i32(a: i32, b: i32, x: i32) -> i32 {
    if x < a {
        a
    } else if x > b {
        b
    } else {
        x
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // §5.9.10 read_interpolation_filter
    // -----------------------------------------------------------------

    #[test]
    fn interpolation_filter_switchable_one_bit() {
        // is_filter_switchable = 1 ⇒ Switchable, 1 bit consumed.
        let payload = [0b1000_0000u8];
        let (filt, n) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::Switchable);
        assert!(filt.is_switchable());
        assert_eq!(n, 1);
    }

    #[test]
    fn interpolation_filter_eighttap_three_bits() {
        // is_filter_switchable = 0, interpolation_filter f(2) = 00 ⇒
        // EIGHTTAP, 3 bits consumed.
        let payload = [0b0000_0000u8];
        let (filt, n) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::Eighttap);
        assert_eq!(n, 3);
    }

    #[test]
    fn interpolation_filter_eighttap_smooth() {
        // is_filter_switchable = 0, f(2) = 01 ⇒ EIGHTTAP_SMOOTH.
        // bits = 0 01 = 001 = 0010_0000 = 0x20.
        let payload = [0b0010_0000u8];
        let (filt, n) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::EighttapSmooth);
        assert_eq!(n, 3);
    }

    #[test]
    fn interpolation_filter_eighttap_sharp() {
        // is_filter_switchable = 0, f(2) = 10 ⇒ EIGHTTAP_SHARP.
        // bits = 0 10 = 010 = 0100_0000 = 0x40.
        let payload = [0b0100_0000u8];
        let (filt, _) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::EighttapSharp);
    }

    #[test]
    fn interpolation_filter_bilinear() {
        // is_filter_switchable = 0, f(2) = 11 ⇒ BILINEAR.
        // bits = 0 11 = 011 = 0110_0000 = 0x60.
        let payload = [0b0110_0000u8];
        let (filt, _) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::Bilinear);
    }

    #[test]
    fn interpolation_filter_raw_round_trip() {
        for raw in 0u8..4u8 {
            let f = InterpolationFilter::from_raw(raw);
            assert_eq!(f.as_raw(), raw);
        }
        assert_eq!(InterpolationFilter::Switchable.as_raw(), 4);
    }

    #[test]
    fn interpolation_filter_unexpected_end() {
        let payload: [u8; 0] = [];
        let err = parse_interpolation_filter(&payload).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.11 loop_filter_params
    // -----------------------------------------------------------------

    #[test]
    fn loop_filter_short_circuit_coded_lossless() {
        // CodedLossless = true ⇒ short-circuit, no bits read.
        let payload: [u8; 0] = [];
        let (lf, n) = parse_loop_filter_params(&payload, 3, true, false).expect("decodes");
        assert!(lf.short_circuited);
        assert_eq!(n, 0);
        assert_eq!(lf.loop_filter_level, [0; 4]);
        assert_eq!(lf.loop_filter_sharpness, 0);
        assert!(!lf.loop_filter_delta_enabled);
        assert_eq!(lf.loop_filter_ref_deltas, LOOP_FILTER_REF_DELTAS_DEFAULT);
        assert_eq!(lf.loop_filter_mode_deltas, LOOP_FILTER_MODE_DELTAS_DEFAULT);
        // INTRA_FRAME default per §5.9.11 short-circuit is 1.
        assert_eq!(lf.loop_filter_ref_deltas[0], 1);
        // GOLDEN_FRAME / ALTREF2_FRAME / ALTREF_FRAME defaults = -1.
        assert_eq!(lf.loop_filter_ref_deltas[4], -1);
        assert_eq!(lf.loop_filter_ref_deltas[6], -1);
        assert_eq!(lf.loop_filter_ref_deltas[7], -1);
    }

    #[test]
    fn loop_filter_short_circuit_allow_intrabc() {
        let payload: [u8; 0] = [];
        let (lf, n) = parse_loop_filter_params(&payload, 3, false, true).expect("decodes");
        assert!(lf.short_circuited);
        assert_eq!(n, 0);
    }

    #[test]
    fn loop_filter_full_path_levels_only() {
        // num_planes=3, neither short-circuit. Bits:
        //   loop_filter_level[0] f(6) = 0
        //   loop_filter_level[1] f(6) = 0   (so 2/3 are NOT read)
        //   loop_filter_sharpness f(3) = 0
        //   loop_filter_delta_enabled f(1) = 0
        // Total = 6 + 6 + 3 + 1 = 16 bits, all zero.
        let payload = [0u8; 2];
        let (lf, n) = parse_loop_filter_params(&payload, 3, false, false).expect("decodes");
        assert!(!lf.short_circuited);
        assert_eq!(lf.loop_filter_level, [0; 4]);
        assert_eq!(lf.loop_filter_sharpness, 0);
        assert!(!lf.loop_filter_delta_enabled);
        assert!(!lf.loop_filter_delta_update);
        assert_eq!(n, 16);
    }

    #[test]
    fn loop_filter_levels_with_planes() {
        // Construct loop_filter_level[0]=42, level[1]=17 — both
        // non-zero so num_planes=3 path reads two more 6-bit values.
        // We pack:
        //   level[0]=42 (6 bits) = 101010
        //   level[1]=17 (6 bits) = 010001
        //   level[2]=3  (6 bits) = 000011
        //   level[3]=4  (6 bits) = 000100
        //   sharpness=5 (3 bits) = 101
        //   delta_enabled=0 (1 bit)
        // Total = 6+6+6+6+3+1 = 28 bits.
        let bits: &[u8] = &[
            1, 0, 1, 0, 1, 0, // level[0] = 42
            0, 1, 0, 0, 0, 1, // level[1] = 17
            0, 0, 0, 0, 1, 1, // level[2] = 3
            0, 0, 0, 1, 0, 0, // level[3] = 4
            1, 0, 1, // sharpness = 5
            0, // delta_enabled = 0
        ];
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (lf, n) = parse_loop_filter_params(&payload, 3, false, false).expect("decodes");
        assert_eq!(lf.loop_filter_level, [42, 17, 3, 4]);
        assert_eq!(lf.loop_filter_sharpness, 5);
        assert!(!lf.loop_filter_delta_enabled);
        assert_eq!(n, bits.len());
    }

    #[test]
    fn loop_filter_mono_skips_plane2_plane3() {
        // num_planes=1, levels non-zero, but the `NumPlanes > 1` gate
        // suppresses level[2]/level[3] reads.
        //   level[0]=10 (6) = 001010
        //   level[1]=20 (6) = 010100
        //   sharpness=0 (3) = 000
        //   delta_enabled=0 (1) = 0
        // Total = 6+6+3+1 = 16 bits.
        let bits: &[u8] = &[
            0, 0, 1, 0, 1, 0, // 10
            0, 1, 0, 1, 0, 0, // 20
            0, 0, 0, // sharpness
            0, // delta_enabled
        ];
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (lf, n) = parse_loop_filter_params(&payload, 1, false, false).expect("decodes");
        assert_eq!(lf.loop_filter_level, [10, 20, 0, 0]);
        assert_eq!(n, 16);
    }

    #[test]
    fn loop_filter_delta_update_walks_refs_and_modes() {
        // num_planes=1, level[0]=level[1]=0 to avoid the plane-2/3
        // reads. Then delta_enabled=1 / delta_update=1 with a sparse
        // pattern: only loop_filter_ref_deltas[0] gets updated to -3,
        // and only loop_filter_mode_deltas[1] gets updated to 5.
        let mut bits: Vec<u8> = vec![
            0, 0, 0, 0, 0, 0, // level[0] = 0
            0, 0, 0, 0, 0, 0, // level[1] = 0
            0, 0, 0, // sharpness = 0
            1, // delta_enabled
            1, // delta_update
        ];
        // 8 ref slots — first one updated to -3 (su(7) = -3 in 7-bit
        // two's complement: -3 = 0b1111101 = 0x7D).
        bits.push(1); // update_ref_delta[0]
        for i in (0..7).rev() {
            bits.push(((0x7Du32 >> i) & 1) as u8); // -3
        }
        bits.resize(bits.len() + (TOTAL_REFS_PER_FRAME - 1), 0);
        // 2 mode slots — second one updated to 5.
        bits.push(0); // update_mode_delta[0]
        bits.push(1); // update_mode_delta[1]
        for i in (0..7).rev() {
            bits.push(((5u32 >> i) & 1) as u8); // 5
        }
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (lf, n) = parse_loop_filter_params(&payload, 1, false, false).expect("decodes");
        assert!(lf.loop_filter_delta_enabled);
        assert!(lf.loop_filter_delta_update);
        assert_eq!(lf.loop_filter_ref_deltas[0], -3);
        // Slots 1..=7 were not updated, but the standalone parser
        // surfaces the §5.9.11 short-circuit defaults for any slot it
        // didn't read. The spec says "maintain previous value" — which
        // is "loaded from a previous frame's state" in the streaming
        // decoder. For a standalone trace we report the defaults, but
        // we do NOT assert on slots[1..] here because they aren't
        // bitstream-visible in this synthetic trace.
        assert_eq!(lf.loop_filter_mode_deltas[1], 5);
        assert_eq!(n, bits.len());
    }

    // -----------------------------------------------------------------
    // §5.9.12 quantization_params
    // -----------------------------------------------------------------

    #[test]
    fn quantization_mono_no_qm() {
        // num_planes=1, separate_uv_delta_q=false.
        //   base_q_idx f(8) = 64 (0b0100_0000)
        //   read_delta_q for DeltaQYDc:
        //     delta_coded f(1) = 0 ⇒ delta_q = 0
        //   (mono: U/V deltas skipped)
        //   using_qmatrix f(1) = 0
        // Total = 8 + 1 + 1 = 10 bits = `0100_0000 0 0` packed:
        // bytes = 0100_0000 0000_0000 → 0x40 0x00.
        let payload = [0x40u8, 0x00];
        let (q, n) = parse_quantization_params(&payload, 1, false).expect("decodes");
        assert_eq!(q.base_q_idx, 64);
        assert_eq!(q.delta_q_y_dc, 0);
        assert_eq!(q.delta_q_u_dc, 0);
        assert_eq!(q.delta_q_u_ac, 0);
        assert_eq!(q.delta_q_v_dc, 0);
        assert_eq!(q.delta_q_v_ac, 0);
        assert!(!q.using_qmatrix);
        assert!(!q.diff_uv_delta);
        assert_eq!(n, 10);
    }

    #[test]
    fn quantization_3plane_no_separate_uv() {
        // num_planes=3, separate_uv_delta_q=false:
        //   base_q_idx f(8) = 128 = 1000_0000
        //   DeltaQYDc: delta_coded=1, su(7)= -4 (0b1111100 = 0x7C)
        //   (separate_uv_delta_q=false ⇒ no diff_uv_delta bit)
        //   DeltaQUDc: delta_coded=0
        //   DeltaQUAc: delta_coded=1, su(7)= +3 (0b000_0011)
        //   diff_uv_delta=false ⇒ V mirrors U
        //   using_qmatrix=0
        let bits: Vec<u8> = {
            let mut b = vec![];
            // base_q_idx = 128 = 1000_0000
            for i in (0..8).rev() {
                b.push(((128u32 >> i) & 1) as u8);
            }
            // DeltaQYDc: delta_coded=1, value=-4 (0b1111100)
            b.push(1);
            for i in (0..7).rev() {
                b.push(((0x7Cu32 >> i) & 1) as u8);
            }
            // DeltaQUDc: delta_coded=0
            b.push(0);
            // DeltaQUAc: delta_coded=1, value=+3 (0b0000011)
            b.push(1);
            for i in (0..7).rev() {
                b.push(((3u32 >> i) & 1) as u8);
            }
            // using_qmatrix = 0
            b.push(0);
            b
        };
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (q, n) = parse_quantization_params(&payload, 3, false).expect("decodes");
        assert_eq!(q.base_q_idx, 128);
        assert_eq!(q.delta_q_y_dc, -4);
        assert!(!q.diff_uv_delta);
        assert_eq!(q.delta_q_u_dc, 0);
        assert_eq!(q.delta_q_u_ac, 3);
        // V mirrors U.
        assert_eq!(q.delta_q_v_dc, q.delta_q_u_dc);
        assert_eq!(q.delta_q_v_ac, q.delta_q_u_ac);
        assert!(!q.using_qmatrix);
        assert_eq!(n, bits.len());
    }

    #[test]
    fn quantization_separate_uv_with_diff() {
        // num_planes=3, separate_uv_delta_q=true:
        //   base_q_idx = 100 = 0b0110_0100
        //   DeltaQYDc: delta_coded=0
        //   diff_uv_delta=1
        //   DeltaQUDc: delta_coded=1 value=+1 (0b0000_001)
        //   DeltaQUAc: delta_coded=0
        //   DeltaQVDc: delta_coded=1 value=-1 (0b1111_111)
        //   DeltaQVAc: delta_coded=0
        //   using_qmatrix=1
        //   qm_y=3 (0b0011)
        //   qm_u=4 (0b0100)
        //   (separate_uv_delta_q=true ⇒ qm_v read)
        //   qm_v=5 (0b0101)
        let mut bits = vec![];
        for i in (0..8).rev() {
            bits.push(((100u32 >> i) & 1) as u8);
        }
        bits.push(0); // delta_coded for DeltaQYDc
        bits.push(1); // diff_uv_delta
        bits.push(1); // delta_coded
        for i in (0..7).rev() {
            bits.push(((1u32 >> i) & 1) as u8);
        }
        bits.push(0); // delta_coded for DeltaQUAc
        bits.push(1); // delta_coded
        for i in (0..7).rev() {
            bits.push(((0x7Fu32 >> i) & 1) as u8); // -1
        }
        bits.push(0); // delta_coded for DeltaQVAc
        bits.push(1); // using_qmatrix
        for i in (0..4).rev() {
            bits.push(((3u32 >> i) & 1) as u8); // qm_y
        }
        for i in (0..4).rev() {
            bits.push(((4u32 >> i) & 1) as u8); // qm_u
        }
        for i in (0..4).rev() {
            bits.push(((5u32 >> i) & 1) as u8); // qm_v
        }
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (q, n) = parse_quantization_params(&payload, 3, true).expect("decodes");
        assert_eq!(q.base_q_idx, 100);
        assert_eq!(q.delta_q_y_dc, 0);
        assert!(q.diff_uv_delta);
        assert_eq!(q.delta_q_u_dc, 1);
        assert_eq!(q.delta_q_u_ac, 0);
        assert_eq!(q.delta_q_v_dc, -1);
        assert_eq!(q.delta_q_v_ac, 0);
        assert!(q.using_qmatrix);
        assert_eq!(q.qm_y, 3);
        assert_eq!(q.qm_u, 4);
        assert_eq!(q.qm_v, 5);
        assert_eq!(n, bits.len());
    }

    #[test]
    fn quantization_using_qmatrix_without_separate_uv() {
        // num_planes=3, separate_uv_delta_q=false, using_qmatrix=1
        // ⇒ qm_v = qm_u (no separate read).
        //   base_q_idx = 0
        //   DeltaQYDc: delta_coded=0
        //   (no diff_uv_delta — gated by separate_uv_delta_q)
        //   DeltaQUDc: delta_coded=0
        //   DeltaQUAc: delta_coded=0
        //   (V mirrors U)
        //   using_qmatrix=1
        //   qm_y=7 (0b0111)
        //   qm_u=9 (0b1001)
        // Bits = 8 + 1 + 1 + 1 + 1 + 4 + 4 = 20.
        let mut bits = vec![0u8; 8]; // base_q_idx = 0
        bits.push(0); // DeltaQYDc delta_coded
        bits.push(0); // DeltaQUDc delta_coded
        bits.push(0); // DeltaQUAc delta_coded
        bits.push(1); // using_qmatrix
        for i in (0..4).rev() {
            bits.push(((7u32 >> i) & 1) as u8); // qm_y
        }
        for i in (0..4).rev() {
            bits.push(((9u32 >> i) & 1) as u8); // qm_u
        }
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (q, _) = parse_quantization_params(&payload, 3, false).expect("decodes");
        assert!(q.using_qmatrix);
        assert_eq!(q.qm_y, 7);
        assert_eq!(q.qm_u, 9);
        assert_eq!(q.qm_v, 9, "qm_v mirrors qm_u when !separate_uv_delta_q");
    }

    #[test]
    fn quantization_unexpected_end() {
        let payload: [u8; 0] = [];
        let err = parse_quantization_params(&payload, 1, false).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.14 segmentation_params
    // -----------------------------------------------------------------

    use crate::frame_header::PRIMARY_REF_NONE;

    #[test]
    fn segmentation_disabled_one_bit() {
        // segmentation_enabled = 0 ⇒ 1 bit consumed, every field 0.
        let payload = [0b0000_0000u8];
        let (s, n) = parse_segmentation_params(&payload, PRIMARY_REF_NONE).expect("decodes");
        assert!(!s.enabled);
        assert!(!s.update_map);
        assert!(!s.temporal_update);
        assert!(!s.update_data);
        assert_eq!(
            s.segment_feature_active,
            [[false; SEG_LVL_MAX]; MAX_SEGMENTS]
        );
        assert_eq!(s.segment_feature_data, [[0i16; SEG_LVL_MAX]; MAX_SEGMENTS]);
        assert!(!s.seg_id_pre_skip);
        assert_eq!(s.last_active_seg_id, 0);
        assert_eq!(n, 1);
    }

    #[test]
    fn segmentation_primary_ref_none_no_active_features() {
        // segmentation_enabled = 1, primary_ref_frame == PRIMARY_REF_NONE
        // ⇒ update_map=1, temporal_update=0, update_data=1 (no
        // bitstream reads for the update flags). Then update_data=1
        // walks 8×8 = 64 feature_enabled bits, all zero ⇒ no further
        // reads. Total = 1 + 64 = 65 bits.
        let mut bits = vec![1u8]; // segmentation_enabled
                                  // 64 zero `feature_enabled` bits — one per (segment, feature)
                                  // slot, all disabled.
        bits.resize(1 + MAX_SEGMENTS * SEG_LVL_MAX, 0);
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (s, n) = parse_segmentation_params(&payload, PRIMARY_REF_NONE).expect("decodes");
        assert!(s.enabled);
        assert!(s.update_map);
        assert!(!s.temporal_update);
        assert!(s.update_data);
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                assert!(!s.segment_feature_active[i][j]);
                assert_eq!(s.segment_feature_data[i][j], 0);
            }
        }
        assert!(!s.seg_id_pre_skip);
        assert_eq!(s.last_active_seg_id, 0);
        assert_eq!(n, bits.len());
    }

    #[test]
    fn segmentation_with_primary_ref_reads_three_update_bits() {
        // primary_ref_frame = 0 (i.e. != PRIMARY_REF_NONE):
        //   segmentation_enabled = 1
        //   segmentation_update_map = 1
        //   segmentation_temporal_update = 1
        //   segmentation_update_data = 0
        //   ⇒ no inner-loop reads.
        //   = 4 bits.
        let payload = [0b1110_0000u8];
        let (s, n) = parse_segmentation_params(&payload, 0).expect("decodes");
        assert!(s.enabled);
        assert!(s.update_map);
        assert!(s.temporal_update);
        assert!(!s.update_data);
        assert_eq!(n, 4);
    }

    #[test]
    fn segmentation_update_map_zero_skips_temporal_bit() {
        // primary_ref_frame=0, segmentation_enabled=1, update_map=0
        // (so temporal_update is NOT read), update_data=0. = 3 bits.
        let payload = [0b1000_0000u8];
        let (s, n) = parse_segmentation_params(&payload, 0).expect("decodes");
        assert!(s.enabled);
        assert!(!s.update_map);
        assert!(!s.temporal_update);
        assert!(!s.update_data);
        assert_eq!(n, 3);
    }

    #[test]
    fn segmentation_feature_alt_q_signed() {
        // primary_ref_frame == PRIMARY_REF_NONE ⇒ no update-flag reads.
        // segmentation_enabled = 1, update_data = 1.
        // Segment 0:
        //   feature 0 (ALT_Q): enabled=1, value = su(9) = -50.
        //     Encode -50 in 9-bit two's complement: 2^9 - 50 = 462 = 0b1_1100_1110.
        //   features 1..7: enabled=0.
        // Segments 1..7: feature_enabled[*][*] = 0 ⇒ 7 × 8 = 56 bits.
        let mut bits = vec![1u8]; // segmentation_enabled
                                  // Segment 0, feature 0: enabled=1, su(9) = -50 ⇒ 462 = 0b111001110.
        bits.push(1); // feature_enabled
        for i in (0..9).rev() {
            bits.push(((462u32 >> i) & 1) as u8);
        }
        // Segment 0, features 1..=7: enabled=0 — 7 zero bits.
        // Then segments 1..=7, all 8 features: enabled=0 — 7×8 = 56 zero
        // bits.
        let tail_zeros = (SEG_LVL_MAX - 1) + (MAX_SEGMENTS - 1) * SEG_LVL_MAX;
        bits.resize(bits.len() + tail_zeros, 0);
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (s, n) = parse_segmentation_params(&payload, PRIMARY_REF_NONE).expect("decodes");
        assert!(s.enabled);
        assert!(s.segment_feature_active[0][SEG_LVL_ALT_Q]);
        assert_eq!(s.segment_feature_data[0][SEG_LVL_ALT_Q], -50);
        // All other features inactive.
        for i in 0..MAX_SEGMENTS {
            for j in 0..SEG_LVL_MAX {
                if (i, j) != (0, SEG_LVL_ALT_Q) {
                    assert!(!s.segment_feature_active[i][j]);
                }
            }
        }
        // Active features at j=0 ⇒ SegIdPreSkip stays 0.
        assert!(!s.seg_id_pre_skip);
        // Active in segment 0 only.
        assert_eq!(s.last_active_seg_id, 0);
        // Total bits: 1 (enabled) + 1 (fe) + 9 (su) + 7 (zero fe for rest of seg 0)
        //           + 7 segments × 8 features (zero fe) = 1 + 1 + 9 + 7 + 56 = 74.
        assert_eq!(n, 74);
    }

    #[test]
    fn segmentation_feature_alt_q_clipped_high() {
        // SEG_LVL_ALT_Q has bits=8, signed=1, max=255.
        // su(1+8) = su(9) range is [-256, 255]. We pick +256 (=0x100,
        // 9 bits = 0b1_0000_0000 = sign bit set ⇒ decodes to -256),
        // which is clipped to -255 by Clip3(-255, 255, -256).
        let mut bits = vec![1u8]; // segmentation_enabled
        bits.push(1); // feature_enabled[0][0]
        for i in (0..9).rev() {
            bits.push(((0x100u32 >> i) & 1) as u8);
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
        let (s, _) = parse_segmentation_params(&payload, PRIMARY_REF_NONE).expect("decodes");
        // Clip3(-255, 255, -256) = -255.
        assert_eq!(s.segment_feature_data[0][SEG_LVL_ALT_Q], -255);
    }

    #[test]
    fn segmentation_feature_ref_frame_unsigned() {
        // SEG_LVL_REF_FRAME has bits=3, signed=0, max=7.
        // Encode feature_value = 6 = 0b110 as f(3).
        let mut bits = vec![1u8]; // segmentation_enabled
                                  // Segment 0, features 0..=4: enabled=0 (5 zero bits).
        bits.resize(bits.len() + SEG_LVL_REF_FRAME, 0);
        // Segment 0, feature 5 (SEG_LVL_REF_FRAME): enabled=1, value=6.
        bits.push(1);
        for i in (0..3).rev() {
            bits.push(((6u32 >> i) & 1) as u8);
        }
        // Segment 0, features 6/7 disabled (2 zero bits) plus
        // segments 1..=7 × 8 features all disabled (56 zero bits).
        let tail_zeros = 2 + (MAX_SEGMENTS - 1) * SEG_LVL_MAX;
        bits.resize(bits.len() + tail_zeros, 0);
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (s, _) = parse_segmentation_params(&payload, PRIMARY_REF_NONE).expect("decodes");
        assert!(s.segment_feature_active[0][SEG_LVL_REF_FRAME]);
        assert_eq!(s.segment_feature_data[0][SEG_LVL_REF_FRAME], 6);
        // Feature j=5 = SEG_LVL_REF_FRAME ⇒ SegIdPreSkip becomes 1.
        assert!(s.seg_id_pre_skip);
        assert_eq!(s.last_active_seg_id, 0);
    }

    #[test]
    fn segmentation_feature_skip_no_value_bits() {
        // SEG_LVL_SKIP (j=6) has bits=0 ⇒ no feature_value read,
        // feature_data forced to 0. Activate it for segment 3.
        let mut bits = vec![1u8]; // segmentation_enabled
                                  // Segments 0..=2: all 8 features enabled=0 (24 zero bits)
                                  // plus segment 3 features 0..=5 disabled (6 zero bits).
        bits.resize(bits.len() + 3 * SEG_LVL_MAX + SEG_LVL_SKIP, 0);
        bits.push(1); // feature_enabled[3][6]
        bits.push(0); // feature_enabled[3][7]
                      // Segments 4..=7: all 8 features enabled=0 — 4×8 = 32 zero bits.
        bits.resize(bits.len() + (MAX_SEGMENTS - 4) * SEG_LVL_MAX, 0);
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (s, n) = parse_segmentation_params(&payload, PRIMARY_REF_NONE).expect("decodes");
        assert!(s.segment_feature_active[3][SEG_LVL_SKIP]);
        assert_eq!(s.segment_feature_data[3][SEG_LVL_SKIP], 0);
        // Feature j=6 >= SEG_LVL_REF_FRAME ⇒ SegIdPreSkip=1.
        assert!(s.seg_id_pre_skip);
        // Active in segment 3 ⇒ last_active_seg_id = 3.
        assert_eq!(s.last_active_seg_id, 3);
        // Total bits = 1 (enabled) + 64 (8 feature_enabled bits × 8 segs)
        //   + 0 (no value bits for skip). = 65.
        assert_eq!(n, 65);
    }

    #[test]
    fn segmentation_unexpected_end() {
        let payload: [u8; 0] = [];
        let err = parse_segmentation_params(&payload, PRIMARY_REF_NONE).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.17 delta_q_params
    // -----------------------------------------------------------------

    #[test]
    fn delta_q_base_q_idx_zero_reads_nothing() {
        // base_q_idx == 0 ⇒ the `delta_q_present` `f(1)` slot is not
        // read; delta_q_present stays 0 and 0 bits are consumed even on
        // an empty payload.
        let payload: [u8; 0] = [];
        let (d, n) = parse_delta_q_params(&payload, 0).expect("decodes");
        assert!(!d.delta_q_present);
        assert_eq!(d.delta_q_res, 0);
        assert_eq!(n, 0);
    }

    #[test]
    fn delta_q_present_zero_reads_one_bit() {
        // base_q_idx > 0, delta_q_present = 0 (single 0 bit), no
        // delta_q_res read. 1 bit consumed.
        let payload = [0b0000_0000u8];
        let (d, n) = parse_delta_q_params(&payload, 120).expect("decodes");
        assert!(!d.delta_q_present);
        assert_eq!(d.delta_q_res, 0);
        assert_eq!(n, 1);
    }

    #[test]
    fn delta_q_present_one_reads_res() {
        // base_q_idx > 0, delta_q_present = 1, delta_q_res = f(2) = 10
        // (= 2). bits: 1 10 = 110 = 0b1100_0000 = 0xC0. 3 bits.
        let payload = [0b1100_0000u8];
        let (d, n) = parse_delta_q_params(&payload, 120).expect("decodes");
        assert!(d.delta_q_present);
        assert_eq!(d.delta_q_res, 2);
        assert_eq!(n, 3);
    }

    #[test]
    fn delta_q_present_bit_unexpected_end() {
        // base_q_idx > 0 but no bytes ⇒ reading delta_q_present errors.
        let payload: [u8; 0] = [];
        let err = parse_delta_q_params(&payload, 120).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.18 delta_lf_params
    // -----------------------------------------------------------------

    #[test]
    fn delta_lf_gated_off_when_delta_q_absent() {
        // delta_q_present == false ⇒ the whole block is skipped; all
        // fields stay default and 0 bits are consumed.
        let payload: [u8; 0] = [];
        let (d, n) = parse_delta_lf_params(&payload, false, false).expect("decodes");
        assert!(!d.delta_lf_present);
        assert_eq!(d.delta_lf_res, 0);
        assert!(!d.delta_lf_multi);
        assert_eq!(n, 0);
    }

    #[test]
    fn delta_lf_present_zero_reads_one_bit() {
        // delta_q_present, !allow_intrabc ⇒ read delta_lf_present = 0
        // (single 0 bit), no further reads. 1 bit consumed.
        let payload = [0b0000_0000u8];
        let (d, n) = parse_delta_lf_params(&payload, true, false).expect("decodes");
        assert!(!d.delta_lf_present);
        assert_eq!(d.delta_lf_res, 0);
        assert!(!d.delta_lf_multi);
        assert_eq!(n, 1);
    }

    #[test]
    fn delta_lf_present_full_path() {
        // delta_q_present, !allow_intrabc ⇒ delta_lf_present = 1,
        // delta_lf_res = f(2) = 11 (= 3), delta_lf_multi = f(1) = 1.
        // bits: 1 11 1 = 1111 = 0b1111_0000 = 0xF0. 4 bits consumed.
        let payload = [0b1111_0000u8];
        let (d, n) = parse_delta_lf_params(&payload, true, false).expect("decodes");
        assert!(d.delta_lf_present);
        assert_eq!(d.delta_lf_res, 3);
        assert!(d.delta_lf_multi);
        assert_eq!(n, 4);
    }

    #[test]
    fn delta_lf_suppressed_by_allow_intrabc() {
        // delta_q_present but allow_intrabc ⇒ delta_lf_present slot is
        // not read; stays 0. No bits consumed even on empty payload.
        let payload: [u8; 0] = [];
        let (d, n) = parse_delta_lf_params(&payload, true, true).expect("decodes");
        assert!(!d.delta_lf_present);
        assert_eq!(d.delta_lf_res, 0);
        assert!(!d.delta_lf_multi);
        assert_eq!(n, 0);
    }

    #[test]
    fn delta_lf_present_bit_unexpected_end() {
        // delta_q_present, !allow_intrabc, empty payload ⇒ reading
        // delta_lf_present errors.
        let payload: [u8; 0] = [];
        let err = parse_delta_lf_params(&payload, true, false).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.19 cdef_params
    // -----------------------------------------------------------------

    /// Pack an MSB-first bit list into a byte buffer.
    fn pack_bits(bits: &[u8]) -> Vec<u8> {
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        payload
    }

    #[test]
    fn cdef_short_circuit_coded_lossless() {
        // CodedLossless ⇒ short-circuit, no bits read.
        let payload: [u8; 0] = [];
        let (c, n) = parse_cdef_params(&payload, 3, true, false, true).expect("decodes");
        assert!(c.short_circuited);
        assert_eq!(c.cdef_damping, 3);
        assert_eq!(c.cdef_bits, 0);
        assert_eq!(c.cdef_y_pri_strength, [0; CDEF_MAX_STRENGTHS]);
        assert_eq!(c.cdef_y_sec_strength, [0; CDEF_MAX_STRENGTHS]);
        assert_eq!(n, 0);
    }

    #[test]
    fn cdef_short_circuit_allow_intrabc() {
        let payload: [u8; 0] = [];
        let (c, n) = parse_cdef_params(&payload, 3, false, true, true).expect("decodes");
        assert!(c.short_circuited);
        assert_eq!(n, 0);
    }

    #[test]
    fn cdef_short_circuit_enable_cdef_off() {
        // !enable_cdef ⇒ short-circuit even when lossless/intrabc clear.
        let payload: [u8; 0] = [];
        let (c, n) = parse_cdef_params(&payload, 3, false, false, false).expect("decodes");
        assert!(c.short_circuited);
        assert_eq!(c.cdef_damping, 3);
        assert_eq!(n, 0);
    }

    #[test]
    fn cdef_full_path_single_entry_3plane() {
        // num_planes=3, full path, cdef_bits=0 ⇒ 1 entry.
        //   cdef_damping_minus_3 = 2 (f2) = 10  ⇒ CdefDamping = 5
        //   cdef_bits            = 0 (f2) = 00
        //   entry 0:
        //     cdef_y_pri_strength  = 9  (f4) = 1001
        //     cdef_y_sec_strength  = 1  (f2) = 01
        //     cdef_uv_pri_strength = 6  (f4) = 0110
        //     cdef_uv_sec_strength = 2  (f2) = 10
        // Total = 2+2 + 4+2+4+2 = 16 bits.
        let bits: &[u8] = &[
            1, 0, // damping_minus_3 = 2
            0, 0, // cdef_bits = 0
            1, 0, 0, 1, // y_pri = 9
            0, 1, // y_sec = 1
            0, 1, 1, 0, // uv_pri = 6
            1, 0, // uv_sec = 2
        ];
        let payload = pack_bits(bits);
        let (c, n) = parse_cdef_params(&payload, 3, false, false, true).expect("decodes");
        assert!(!c.short_circuited);
        assert_eq!(c.cdef_damping, 5);
        assert_eq!(c.cdef_bits, 0);
        assert_eq!(c.cdef_y_pri_strength[0], 9);
        assert_eq!(c.cdef_y_sec_strength[0], 1);
        assert_eq!(c.cdef_uv_pri_strength[0], 6);
        assert_eq!(c.cdef_uv_sec_strength[0], 2);
        // Unused entries stay 0.
        assert_eq!(c.cdef_y_pri_strength[1], 0);
        assert_eq!(n, 16);
    }

    #[test]
    fn cdef_sec_strength_three_becomes_four() {
        // §5.9.19: cdef_*_sec_strength == 3 is bumped to 4.
        //   damping_minus_3 = 0, cdef_bits = 0
        //   entry 0:
        //     y_pri = 0  (f4) = 0000
        //     y_sec = 3  (f2) = 11  ⇒ stored 4
        //     uv_pri= 0  (f4) = 0000
        //     uv_sec= 3  (f2) = 11  ⇒ stored 4
        let bits: &[u8] = &[
            0, 0, // damping_minus_3 = 0
            0, 0, // cdef_bits = 0
            0, 0, 0, 0, // y_pri = 0
            1, 1, // y_sec = 3 -> 4
            0, 0, 0, 0, // uv_pri = 0
            1, 1, // uv_sec = 3 -> 4
        ];
        let payload = pack_bits(bits);
        let (c, n) = parse_cdef_params(&payload, 3, false, false, true).expect("decodes");
        assert_eq!(c.cdef_y_sec_strength[0], 4);
        assert_eq!(c.cdef_uv_sec_strength[0], 4);
        assert_eq!(c.cdef_damping, 3);
        assert_eq!(n, 16);
    }

    #[test]
    fn cdef_full_path_mono_skips_chroma() {
        // num_planes=1 ⇒ the `NumPlanes > 1` chroma reads are skipped.
        //   damping_minus_3 = 1, cdef_bits = 1 ⇒ 2 entries
        //   entry 0: y_pri = 15 (1111), y_sec = 2 (10)
        //   entry 1: y_pri = 8  (1000), y_sec = 0 (00)
        // Total = 2 + 2 + (4+2)*2 = 16 bits.
        let bits: &[u8] = &[
            0, 1, // damping_minus_3 = 1 ⇒ CdefDamping 4
            0, 1, // cdef_bits = 1 ⇒ 2 entries
            1, 1, 1, 1, // entry0 y_pri = 15
            1, 0, // entry0 y_sec = 2
            1, 0, 0, 0, // entry1 y_pri = 8
            0, 0, // entry1 y_sec = 0
        ];
        let payload = pack_bits(bits);
        let (c, n) = parse_cdef_params(&payload, 1, false, false, true).expect("decodes");
        assert!(!c.short_circuited);
        assert_eq!(c.cdef_damping, 4);
        assert_eq!(c.cdef_bits, 1);
        assert_eq!(c.cdef_y_pri_strength[0], 15);
        assert_eq!(c.cdef_y_sec_strength[0], 2);
        assert_eq!(c.cdef_y_pri_strength[1], 8);
        assert_eq!(c.cdef_y_sec_strength[1], 0);
        // No chroma reads happened.
        assert_eq!(c.cdef_uv_pri_strength, [0; CDEF_MAX_STRENGTHS]);
        assert_eq!(c.cdef_uv_sec_strength, [0; CDEF_MAX_STRENGTHS]);
        assert_eq!(n, 16);
    }

    #[test]
    fn cdef_full_path_eight_entries() {
        // cdef_bits = 3 ⇒ 1 << 3 = 8 entries; num_planes=1 to keep it
        // compact. Each entry: y_pri (f4) + y_sec (f2) = 6 bits.
        // Header = 2 (damping) + 2 (cdef_bits) = 4 bits.
        // Total = 4 + 8*6 = 52 bits.
        let mut bits: Vec<u8> = Vec::new();
        bits.extend_from_slice(&[0, 0]); // damping_minus_3 = 0
        bits.extend_from_slice(&[1, 1]); // cdef_bits = 3
        for i in 0..8u8 {
            // y_pri = i (f4)
            for b in (0..4).rev() {
                bits.push((i >> b) & 1);
            }
            // y_sec = 1 (f2) = 01
            bits.push(0);
            bits.push(1);
        }
        let payload = pack_bits(&bits);
        let (c, n) = parse_cdef_params(&payload, 1, false, false, true).expect("decodes");
        assert_eq!(c.cdef_bits, 3);
        for i in 0..8usize {
            assert_eq!(c.cdef_y_pri_strength[i], i as u8);
            assert_eq!(c.cdef_y_sec_strength[i], 1);
        }
        assert_eq!(n, 52);
    }

    #[test]
    fn cdef_full_path_unexpected_end() {
        // Full path but empty payload ⇒ reading cdef_damping_minus_3
        // errors.
        let payload: [u8; 0] = [];
        let err = parse_cdef_params(&payload, 3, false, false, true).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.20 lr_params
    // -----------------------------------------------------------------

    #[test]
    fn lr_remap_table_matches_spec() {
        // Remap_Lr_Type[4] = { NONE, SWITCHABLE, WIENER, SGRPROJ }.
        assert_eq!(FrameRestorationType::remap(0), FrameRestorationType::None);
        assert_eq!(
            FrameRestorationType::remap(1),
            FrameRestorationType::Switchable
        );
        assert_eq!(FrameRestorationType::remap(2), FrameRestorationType::Wiener);
        assert_eq!(
            FrameRestorationType::remap(3),
            FrameRestorationType::SgrProj
        );
        // §6.10.15 FrameRestorationType symbol values.
        assert_eq!(FrameRestorationType::None.as_u8(), 0);
        assert_eq!(FrameRestorationType::Wiener.as_u8(), 1);
        assert_eq!(FrameRestorationType::SgrProj.as_u8(), 2);
        assert_eq!(FrameRestorationType::Switchable.as_u8(), 3);
    }

    #[test]
    fn lr_short_circuit_all_lossless() {
        // AllLossless ⇒ short-circuit, no bits read.
        let payload: [u8; 0] = [];
        let (lr, n) =
            parse_lr_params(&payload, 3, true, true, false, true, false, true).expect("decodes");
        assert!(lr.short_circuited);
        assert!(!lr.uses_lr);
        assert!(!lr.uses_chroma_lr);
        assert_eq!(lr.frame_restoration_type, [FrameRestorationType::None; 3]);
        assert_eq!(lr.lr_unit_shift, 0);
        assert_eq!(lr.lr_uv_shift, 0);
        assert_eq!(lr.loop_restoration_size, [0; 3]);
        assert_eq!(n, 0);
    }

    #[test]
    fn lr_short_circuit_allow_intrabc() {
        // allow_intrabc ⇒ short-circuit even when lossless clear.
        let payload: [u8; 0] = [];
        let (lr, n) =
            parse_lr_params(&payload, 3, true, true, false, false, true, true).expect("decodes");
        assert!(lr.short_circuited);
        assert_eq!(n, 0);
    }

    #[test]
    fn lr_short_circuit_enable_restoration_off() {
        // !enable_restoration ⇒ short-circuit even when lossless/intrabc
        // clear. Matches the `screen-content-tools` / lossless fixtures
        // whose LR trace is all-zero.
        let payload: [u8; 0] = [];
        let (lr, n) =
            parse_lr_params(&payload, 3, true, true, false, false, false, false).expect("decodes");
        assert!(lr.short_circuited);
        assert!(!lr.uses_lr);
        assert_eq!(n, 0);
    }

    #[test]
    fn lr_uses_lr_zero_reads_only_three_types() {
        // num_planes=3, all lr_type = 0 (NONE) ⇒ UsesLr = 0, no shift
        // bits read. Six bits consumed (3 * f(2)).
        let payload = pack_bits(&[
            0, 0, // plane 0 lr_type = 0
            0, 0, // plane 1 lr_type = 0
            0, 0, // plane 2 lr_type = 0
        ]);
        let (lr, n) =
            parse_lr_params(&payload, 3, true, true, false, false, false, true).expect("decodes");
        assert!(!lr.short_circuited);
        assert!(!lr.uses_lr);
        assert!(!lr.uses_chroma_lr);
        assert_eq!(lr.frame_restoration_type, [FrameRestorationType::None; 3]);
        assert_eq!(lr.lr_unit_shift, 0);
        assert_eq!(lr.loop_restoration_size, [0; 3]);
        // 3 planes * f(2) = 6 bits read (position is in bits).
        assert_eq!(n, 6);
    }

    #[test]
    fn lr_luma_only_non128_shift_zero() {
        // num_planes=3, plane0 = WIENER (lr_type=2), planes 1/2 = NONE.
        // UsesLr=1, usesChromaLr=0 (only luma). Non-128 SB:
        //   lr_unit_shift bit = 0 ⇒ shift stays 0, no extra bit.
        // usesChromaLr=0 ⇒ no lr_uv_shift bit.
        let payload = pack_bits(&[
            1, 0, // plane 0 lr_type = 2 (WIENER)
            0, 0, // plane 1 lr_type = 0 (NONE)
            0, 0, // plane 2 lr_type = 0 (NONE)
            0, // lr_unit_shift = 0
        ]);
        let (lr, n) =
            parse_lr_params(&payload, 3, true, true, false, false, false, true).expect("decodes");
        assert!(lr.uses_lr);
        assert!(!lr.uses_chroma_lr);
        assert_eq!(lr.frame_restoration_type[0], FrameRestorationType::Wiener);
        assert_eq!(lr.frame_restoration_type[1], FrameRestorationType::None);
        assert_eq!(lr.frame_restoration_type[2], FrameRestorationType::None);
        assert_eq!(lr.lr_unit_shift, 0);
        assert_eq!(lr.lr_uv_shift, 0);
        // LoopRestorationSize[0] = 256 >> (2 - 0) = 64.
        assert_eq!(lr.loop_restoration_size[0], 64);
        assert_eq!(lr.loop_restoration_size[1], 64);
        assert_eq!(lr.loop_restoration_size[2], 64);
        // 6 (types) + 1 (unit_shift) = 7 bits read (position in bits).
        assert_eq!(n, 7);
    }

    #[test]
    fn lr_non128_unit_shift_one_plus_extra() {
        // Non-128 SB, lr_unit_shift bit = 1 ⇒ read lr_unit_extra_shift.
        // extra = 1 ⇒ lr_unit_shift = 1 + 1 = 2.
        // plane0 = SGRPROJ (lr_type=3), rest NONE. usesChromaLr=0.
        let payload = pack_bits(&[
            1, 1, // plane 0 lr_type = 3 (SGRPROJ)
            0, 0, // plane 1 NONE
            0, 0, // plane 2 NONE
            1, // lr_unit_shift bit = 1
            1, // lr_unit_extra_shift = 1
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, true, false, false, false, true).expect("decodes");
        assert!(lr.uses_lr);
        assert_eq!(lr.frame_restoration_type[0], FrameRestorationType::SgrProj);
        assert_eq!(lr.lr_unit_shift, 2);
        // LoopRestorationSize[0] = 256 >> (2 - 2) = 256.
        assert_eq!(lr.loop_restoration_size[0], 256);
        assert_eq!(lr.loop_restoration_size[1], 256);
    }

    #[test]
    fn lr_non128_unit_shift_one_extra_zero() {
        // Non-128 SB, lr_unit_shift bit = 1, extra = 0 ⇒ shift = 1.
        let payload = pack_bits(&[
            1, 0, // plane 0 WIENER
            0, 0, 0, 0, // planes 1/2 NONE
            1, // lr_unit_shift bit = 1
            0, // lr_unit_extra_shift = 0
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, true, false, false, false, true).expect("decodes");
        assert_eq!(lr.lr_unit_shift, 1);
        // LoopRestorationSize[0] = 256 >> (2 - 1) = 128.
        assert_eq!(lr.loop_restoration_size[0], 128);
    }

    #[test]
    fn lr_128_superblock_post_increment() {
        // use_128x128_superblock ⇒ lr_unit_shift = read_bit + 1, no
        // extra bit. bit = 1 ⇒ shift = 2. Mirrors the superblocks-128
        // fixture (unit_shift=2). plane0 = SWITCHABLE (lr_type=1).
        let payload = pack_bits(&[
            0, 1, // plane 0 lr_type = 1 (SWITCHABLE)
            0, 0, 0, 0, // planes 1/2 NONE
            1, // lr_unit_shift bit (post-incremented to 2)
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, true, true, false, false, true).expect("decodes");
        assert_eq!(
            lr.frame_restoration_type[0],
            FrameRestorationType::Switchable
        );
        assert!(lr.uses_lr);
        assert!(!lr.uses_chroma_lr);
        assert_eq!(lr.lr_unit_shift, 2);
        assert_eq!(lr.loop_restoration_size[0], 256);
    }

    #[test]
    fn lr_128_superblock_bit_zero_gives_shift_one() {
        // use_128x128_superblock, bit = 0 ⇒ shift = 0 + 1 = 1.
        let payload = pack_bits(&[
            1, 0, // plane 0 WIENER
            0, 0, 0, 0, // planes 1/2 NONE
            0, // lr_unit_shift bit ⇒ +1 = 1
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, true, true, false, false, true).expect("decodes");
        assert_eq!(lr.lr_unit_shift, 1);
        assert_eq!(lr.loop_restoration_size[0], 128);
    }

    #[test]
    fn lr_chroma_lr_420_reads_uv_shift() {
        // plane2 (V) = SGRPROJ ⇒ usesChromaLr=1. 4:2:0
        // (subsampling_x && subsampling_y) ⇒ read lr_uv_shift.
        // Mirrors i-only-64x64-prof0 (v_type=2, unit_shift=2,
        // uv_shift=0), 128 SB.
        let payload = pack_bits(&[
            0, 0, // plane 0 NONE
            0, 0, // plane 1 NONE
            1, 0, // plane 2 lr_type = 2 (WIENER) ⇒ chroma LR
            1, // 128 SB lr_unit_shift bit ⇒ shift = 2
            0, // lr_uv_shift = 0
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, true, true, false, false, true).expect("decodes");
        assert!(lr.uses_lr);
        assert!(lr.uses_chroma_lr);
        assert_eq!(lr.frame_restoration_type[2], FrameRestorationType::Wiener);
        assert_eq!(lr.lr_unit_shift, 2);
        assert_eq!(lr.lr_uv_shift, 0);
        assert_eq!(lr.loop_restoration_size[0], 256);
        assert_eq!(lr.loop_restoration_size[1], 256);
        assert_eq!(lr.loop_restoration_size[2], 256);
    }

    #[test]
    fn lr_chroma_lr_420_uv_shift_one_halves_chroma() {
        // 4:2:0 chroma LR, lr_uv_shift = 1 ⇒ chroma sizes halved.
        let payload = pack_bits(&[
            0, 0, // plane 0 NONE
            1, 0, // plane 1 lr_type = 2 (WIENER) ⇒ chroma LR
            0, 0, // plane 2 NONE
            1, // 128 SB lr_unit_shift bit ⇒ shift = 2
            1, // lr_uv_shift = 1
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, true, true, false, false, true).expect("decodes");
        assert!(lr.uses_chroma_lr);
        assert_eq!(lr.lr_uv_shift, 1);
        // LoopRestorationSize[0] = 256; chroma = 256 >> 1 = 128.
        assert_eq!(lr.loop_restoration_size[0], 256);
        assert_eq!(lr.loop_restoration_size[1], 128);
        assert_eq!(lr.loop_restoration_size[2], 128);
    }

    #[test]
    fn lr_chroma_lr_non420_skips_uv_shift() {
        // 4:4:4 (subsampling_x=0, subsampling_y=0) with chroma LR ⇒ the
        // lr_uv_shift read is gated off; lr_uv_shift = 0 and no bit
        // consumed beyond lr_unit_shift.
        let payload = pack_bits(&[
            0, 0, // plane 0 NONE
            1, 0, // plane 1 WIENER ⇒ chroma LR
            0, 0, // plane 2 NONE
            1, // 128 SB lr_unit_shift bit ⇒ shift = 2
        ]);
        let (lr, n) =
            parse_lr_params(&payload, 3, false, false, true, false, false, true).expect("decodes");
        assert!(lr.uses_chroma_lr);
        assert_eq!(lr.lr_uv_shift, 0);
        // 6 (types) + 1 (unit_shift) = 7 bits, no uv_shift bit
        // (position in bits).
        assert_eq!(n, 7);
        assert_eq!(lr.loop_restoration_size[1], lr.loop_restoration_size[0]);
    }

    #[test]
    fn lr_chroma_lr_422_skips_uv_shift() {
        // 4:2:2 (subsampling_x=1, subsampling_y=0): the spec gates
        // lr_uv_shift on subsampling_x && subsampling_y, so 4:2:2 skips
        // it even with chroma LR.
        let payload = pack_bits(&[
            0, 0, // plane 0 NONE
            0, 0, // plane 1 NONE
            1, 0, // plane 2 WIENER ⇒ chroma LR
            1, // 128 SB lr_unit_shift bit ⇒ shift = 2
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, false, true, false, false, true).expect("decodes");
        assert!(lr.uses_chroma_lr);
        assert_eq!(lr.lr_uv_shift, 0);
    }

    #[test]
    fn lr_monochrome_reads_one_type_only() {
        // num_planes=1 (monochrome): only plane 0 lr_type read.
        // plane0 = WIENER ⇒ UsesLr=1, usesChromaLr=0 (no i>0 planes).
        // Non-128 SB, lr_unit_shift bit = 0.
        let payload = pack_bits(&[
            1, 0, // plane 0 lr_type = 2 (WIENER)
            0, // lr_unit_shift = 0
        ]);
        let (lr, n) =
            parse_lr_params(&payload, 1, true, true, false, false, false, true).expect("decodes");
        assert!(lr.uses_lr);
        assert!(!lr.uses_chroma_lr);
        assert_eq!(lr.frame_restoration_type[0], FrameRestorationType::Wiener);
        // Planes 1/2 left at default NONE (never read).
        assert_eq!(lr.frame_restoration_type[1], FrameRestorationType::None);
        assert_eq!(lr.frame_restoration_type[2], FrameRestorationType::None);
        assert_eq!(lr.lr_uv_shift, 0);
        // 1 (type) * f(2) + 1 (unit_shift) = 3 bits (position in bits).
        assert_eq!(n, 3);
    }

    #[test]
    fn lr_all_planes_distinct_types() {
        // y=SWITCHABLE(1), u=WIENER(2), v=SGRPROJ(3). UsesLr=1,
        // usesChromaLr=1. Non-128 SB, shift bit = 0, 4:2:0 ⇒ uv_shift.
        let payload = pack_bits(&[
            0, 1, // plane 0 lr_type = 1 (SWITCHABLE)
            1, 0, // plane 1 lr_type = 2 (WIENER)
            1, 1, // plane 2 lr_type = 3 (SGRPROJ)
            0, // lr_unit_shift = 0
            0, // lr_uv_shift = 0
        ]);
        let (lr, _n) =
            parse_lr_params(&payload, 3, true, true, false, false, false, true).expect("decodes");
        assert_eq!(
            lr.frame_restoration_type[0],
            FrameRestorationType::Switchable
        );
        assert_eq!(lr.frame_restoration_type[1], FrameRestorationType::Wiener);
        assert_eq!(lr.frame_restoration_type[2], FrameRestorationType::SgrProj);
        assert!(lr.uses_lr);
        assert!(lr.uses_chroma_lr);
        assert_eq!(lr.lr_unit_shift, 0);
        // 256 >> (2 - 0) = 64.
        assert_eq!(lr.loop_restoration_size[0], 64);
    }

    #[test]
    fn lr_full_path_unexpected_end() {
        // Full path but empty payload ⇒ reading the first lr_type errors.
        let payload: [u8; 0] = [];
        let err = parse_lr_params(&payload, 3, true, true, false, false, false, true)
            .expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    #[test]
    fn lr_unexpected_end_at_unit_shift() {
        // num_planes=8 worth of demand isn't valid, but a 12-bit demand
        // against an 8-bit buffer forces the shift read off the end:
        // three planes (6 type bits) where plane0 = WIENER (UsesLr=1)
        // fill bits 0..5, then a non-128-SB lr_unit_shift bit = 1 at
        // bit 6 triggers lr_unit_extra_shift at bit 7, and finally the
        // lr_uv_shift read for 4:2:0 chroma LR needs bit 8 — which lies
        // in a second byte the 1-byte buffer doesn't provide.
        //
        // plane0 = WIENER(2), plane1 = NONE, plane2 = WIENER(2)
        // ⇒ usesChromaLr=1. unit_shift bit=1, extra=1 ⇒ shift fully read
        // by bit 7; lr_uv_shift then demands bit 8 ⇒ UnexpectedEnd.
        let payload = pack_bits(&[
            1, 0, // plane 0 lr_type = 2 (WIENER)
            0, 0, // plane 1 NONE
            1, 0, // plane 2 lr_type = 2 (WIENER) ⇒ chroma LR
            1, // bit 6 lr_unit_shift = 1
            1, // bit 7 lr_unit_extra_shift = 1
               // lr_uv_shift would be bit 8 (second byte) — absent.
        ]);
        // payload is exactly 8 meaningful bits = 1 byte. The lr_uv_shift
        // read needs a 9th bit.
        let err = parse_lr_params(&payload, 3, true, true, false, false, false, true)
            .expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    #[test]
    fn lr_unexpected_end_at_first_type() {
        // Empty payload, full path ⇒ reading the first lr_type errors.
        let empty: [u8; 0] = [];
        let err = parse_lr_params(&empty, 1, false, false, false, false, false, true)
            .expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.21 read_tx_mode
    // -----------------------------------------------------------------

    #[test]
    fn tx_mode_symbol_values() {
        // §6.8.21 TxMode symbol values + §3 TX_MODES count.
        assert_eq!(TxMode::Only4x4.as_u8(), 0);
        assert_eq!(TxMode::TxModeLargest.as_u8(), 1);
        assert_eq!(TxMode::TxModeSelect.as_u8(), 2);
        assert_eq!(TX_MODES, 3);
    }

    #[test]
    fn tx_mode_coded_lossless_only_4x4_no_bits() {
        // §5.9.21 CodedLossless == 1 ⇒ TxMode = ONLY_4X4, no bits read.
        let payload: [u8; 0] = [];
        let (tx, n) = parse_tx_mode(&payload, true).expect("decodes");
        assert_eq!(tx, TxMode::Only4x4);
        assert_eq!(n, 0);
    }

    #[test]
    fn tx_mode_select_set_reads_one_bit() {
        // §5.9.21 CodedLossless == 0, tx_mode_select = 1 ⇒ TX_MODE_SELECT.
        let payload = pack_bits(&[1]);
        let (tx, n) = parse_tx_mode(&payload, false).expect("decodes");
        assert_eq!(tx, TxMode::TxModeSelect);
        assert_eq!(n, 1);
    }

    #[test]
    fn tx_mode_select_clear_is_largest() {
        // §5.9.21 CodedLossless == 0, tx_mode_select = 0 ⇒ TX_MODE_LARGEST.
        let payload = pack_bits(&[0]);
        let (tx, n) = parse_tx_mode(&payload, false).expect("decodes");
        assert_eq!(tx, TxMode::TxModeLargest);
        assert_eq!(n, 1);
    }

    #[test]
    fn tx_mode_coded_lossless_ignores_bitstream() {
        // Even with a set `tx_mode_select` bit available, the §5.9.21
        // CodedLossless first branch consumes no bits and returns
        // ONLY_4X4.
        let payload = pack_bits(&[1]);
        let (tx, n) = parse_tx_mode(&payload, true).expect("decodes");
        assert_eq!(tx, TxMode::Only4x4);
        assert_eq!(n, 0);
    }

    #[test]
    fn tx_mode_unexpected_end() {
        // CodedLossless == 0 with no bytes ⇒ the tx_mode_select read
        // errors.
        let empty: [u8; 0] = [];
        let err = parse_tx_mode(&empty, false).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.24 global_motion_params
    // -----------------------------------------------------------------

    /// Pack `(value, width)` MSB-first fields into a byte buffer.
    fn pack_fields(fields: &[(u32, u32)]) -> Vec<u8> {
        let mut bits: Vec<u8> = Vec::new();
        for &(value, width) in fields {
            for i in (0..width).rev() {
                bits.push(((value >> i) & 1) as u8);
            }
        }
        pack_bits(&bits)
    }

    #[test]
    fn warp_model_type_symbol_values() {
        // §3 / §6.8.18 symbol values; the §5.9.24 `>= ROTZOOM` /
        // `>= TRANSLATION` comparisons rely on this ordering.
        assert_eq!(WarpModelType::Identity.as_u8(), 0);
        assert_eq!(WarpModelType::Translation.as_u8(), 1);
        assert_eq!(WarpModelType::RotZoom.as_u8(), 2);
        assert_eq!(WarpModelType::Affine.as_u8(), 3);
    }

    #[test]
    fn global_motion_identity_default_values() {
        // §5.9.24 initialiser: IDENTITY with diagonal slots
        // 1 << WARPEDMODEL_PREC_BITS.
        let one = 1i32 << WARPEDMODEL_PREC_BITS;
        let gm = GlobalMotionParams::identity();
        assert!(gm.short_circuited);
        for r in 0..TOTAL_REFS_PER_FRAME {
            assert_eq!(gm.gm_type[r], WarpModelType::Identity);
            assert_eq!(gm.gm_params[r], [0, 0, one, 0, 0, one]);
        }
        assert_eq!(REFS_PER_FRAME, 7);
        assert_eq!(LAST_FRAME, 1);
        assert_eq!(ALTREF_FRAME, 7);
        assert_eq!(INTRA_FRAME, 0);
    }

    #[test]
    fn global_motion_intra_short_circuit_no_bits() {
        // §5.9.24 FrameIsIntra ⇒ identity defaults, no bits read.
        let payload: [u8; 0] = [];
        let (gm, n) = parse_global_motion_params(&payload, true, false, &prev_gm_params_default())
            .expect("decodes");
        assert_eq!(n, 0);
        assert_eq!(gm, GlobalMotionParams::identity());
    }

    #[test]
    fn global_motion_inter_all_identity_seven_bits() {
        // Inter frame, every ref `is_global = 0` ⇒ IDENTITY, no further
        // reads. The §5.9.24 loop runs over LAST_FRAME..=ALTREF_FRAME
        // (7 refs) ⇒ exactly 7 bits.
        let payload = pack_bits(&[0, 0, 0, 0, 0, 0, 0]);
        let (gm, n) = parse_global_motion_params(&payload, false, false, &prev_gm_params_default())
            .expect("decodes");
        assert_eq!(n, 7, "7 `is_global` bits");
        assert!(!gm.short_circuited);
        let identity_params = prev_gm_params_default();
        for (r, want) in identity_params
            .iter()
            .enumerate()
            .take(ALTREF_FRAME + 1)
            .skip(LAST_FRAME)
        {
            assert_eq!(gm.gm_type[r], WarpModelType::Identity);
            assert_eq!(gm.gm_params[r], *want);
        }
    }

    #[test]
    fn global_motion_inter_single_translation_zero_value() {
        // First ref (LAST_FRAME) carries a TRANSLATION model whose two
        // translational coefficients decode to 0; the other six refs are
        // IDENTITY.
        //
        // LAST_FRAME bits:
        //   is_global=1 is_rot_zoom=0 is_translation=1  ⇒ TRANSLATION
        //   read_global_param(idx=0): with allow_high_precision_mv=0,
        //     absBits=8, precBits=2, precDiff=14, round=0, sub=0,
        //     mx=256, r=(PrevGmParams[1][0]=0 >> 14)=0.
        //     decode_signed_subexp_with_ref(-256, 257, 0):
        //       decode_unsigned_subexp_with_ref(513, 256):
        //         v = decode_subexp(513): numSyms=513 > 0 + 3*8=24 ⇒
        //           subexp_more_bits=0, subexp_bits=f(3)=000 ⇒ v=0.
        //         (256<<1=512) <= 513 ⇒ inverse_recenter(256, 0)=256.
        //       x=256 ⇒ signed result = 256 + (-256) = 0.
        //     value = (0 << 14) + 0 = 0.
        //     bits consumed: subexp_more_bits(1)=0 + subexp_bits(3)=000.
        //   read_global_param(idx=1): identical structure ⇒ 4 bits, 0.
        // Then refs LAST2_FRAME..ALTREF_FRAME: is_global=0 (6 bits).
        let payload = pack_fields(&[
            (1, 1), // is_global
            (0, 1), // is_rot_zoom
            (1, 1), // is_translation ⇒ TRANSLATION
            (0, 1), // idx0 subexp_more_bits = 0
            (0, 3), // idx0 subexp_bits = 0
            (0, 1), // idx1 subexp_more_bits = 0
            (0, 3), // idx1 subexp_bits = 0
            (0, 1), // LAST2 is_global = 0
            (0, 1), // LAST3 is_global = 0
            (0, 1), // GOLDEN is_global = 0
            (0, 1), // BWDREF is_global = 0
            (0, 1), // ALTREF2 is_global = 0
            (0, 1), // ALTREF is_global = 0
        ]);
        let (gm, n) = parse_global_motion_params(&payload, false, false, &prev_gm_params_default())
            .expect("decodes");
        // 3 (type) + 4 (idx0) + 4 (idx1) + 6 (remaining is_global) = 17.
        assert_eq!(n, 17);
        assert!(!gm.short_circuited);
        assert_eq!(gm.gm_type[LAST_FRAME], WarpModelType::Translation);
        // Translation only writes [0] / [1]; the diagonal defaults from
        // the identity initialiser persist for [2] / [5].
        let one = 1i32 << WARPEDMODEL_PREC_BITS;
        assert_eq!(gm.gm_params[LAST_FRAME], [0, 0, one, 0, 0, one]);
        for r in (LAST_FRAME + 1)..=ALTREF_FRAME {
            assert_eq!(gm.gm_type[r], WarpModelType::Identity);
        }
    }

    #[test]
    fn global_motion_inter_unexpected_end() {
        // Inter frame with no bytes ⇒ the first `is_global` read errors.
        let empty: [u8; 0] = [];
        let err = parse_global_motion_params(&empty, false, false, &prev_gm_params_default())
            .expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.30 film_grain_params
    // -----------------------------------------------------------------

    fn fg_ctx_present_shown() -> FilmGrainContext {
        FilmGrainContext {
            film_grain_params_present: true,
            show_frame: true,
            showable_frame: false,
            is_inter_frame: false,
            mono_chrome: false,
            subsampling_x: true,
            subsampling_y: true,
        }
    }

    #[test]
    fn film_grain_not_present_resets_no_bits() {
        // §5.9.30 !film_grain_params_present ⇒ reset, no bits read.
        let ctx = FilmGrainContext {
            film_grain_params_present: false,
            ..fg_ctx_present_shown()
        };
        let payload = pack_bits(&[1, 1, 1, 1, 1, 1, 1, 1]);
        let (fg, n) = parse_film_grain_params(&payload, ctx).expect("decodes");
        assert_eq!(n, 0);
        assert_eq!(fg, FilmGrainParams::reset());
    }

    #[test]
    fn film_grain_hidden_frame_resets_no_bits() {
        // §5.9.30 (!show_frame && !showable_frame) ⇒ reset, no bits.
        let ctx = FilmGrainContext {
            show_frame: false,
            showable_frame: false,
            ..fg_ctx_present_shown()
        };
        let payload = pack_bits(&[1, 1, 1, 1]);
        let (fg, n) = parse_film_grain_params(&payload, ctx).expect("decodes");
        assert_eq!(n, 0);
        assert_eq!(fg, FilmGrainParams::reset());
    }

    #[test]
    fn film_grain_apply_grain_zero_one_bit() {
        // §5.9.30 apply_grain = 0 ⇒ reset, one bit read.
        let ctx = fg_ctx_present_shown();
        let payload = pack_bits(&[0]);
        let (fg, n) = parse_film_grain_params(&payload, ctx).expect("decodes");
        assert_eq!(n, 1);
        assert!(!fg.apply_grain);
        assert_eq!(fg, FilmGrainParams::reset());
    }

    #[test]
    fn film_grain_predicted_path_inter_frame() {
        // §5.9.30 INTER_FRAME, apply_grain=1, update_grain=0 ⇒ read
        // film_grain_params_ref_idx then return (predicted path).
        let ctx = FilmGrainContext {
            is_inter_frame: true,
            ..fg_ctx_present_shown()
        };
        let payload = pack_fields(&[
            (1, 1),       // apply_grain
            (0xBEEF, 16), // grain_seed
            (0, 1),       // update_grain (INTER ⇒ read) = 0
            (5, 3),       // film_grain_params_ref_idx = 5
        ]);
        let (fg, n) = parse_film_grain_params(&payload, ctx).expect("decodes");
        // 1 + 16 + 1 + 3 = 21 bits.
        assert_eq!(n, 21);
        assert!(fg.apply_grain);
        assert!(!fg.update_grain);
        assert!(fg.predicted);
        assert_eq!(fg.grain_seed, 0xBEEF);
        assert_eq!(fg.film_grain_params_ref_idx, 5);
        // num_y_points etc. stay at reset defaults on the predicted path.
        assert_eq!(fg.num_y_points, 0);
    }

    #[test]
    fn film_grain_chroma_suppressed_420_no_y_points() {
        // §5.9.30 4:2:0 + num_y_points == 0 ⇒ chroma points suppressed.
        // KEY frame (update_grain derived 1), chroma_scaling_from_luma=0.
        let ctx = fg_ctx_present_shown(); // 4:2:0, not mono, not inter.
        let payload = pack_fields(&[
            (1, 1),  // apply_grain
            (0, 16), // grain_seed
            // update_grain derived 1 (not inter) ⇒ no bit
            (0, 4), // num_y_points = 0
            (0, 1), // chroma_scaling_from_luma = 0
            // suppress_chroma (4:2:0 && num_y_points==0) ⇒ no num_cb/cr
            (1, 2), // grain_scaling_minus_8 = 1 ⇒ GrainScaling = 9
            (0, 2), // ar_coeff_lag = 0 ⇒ numPosLuma = 0
            // num_y_points==0 ⇒ no ar_coeffs_y; cfl=0 && num_cb=0 ⇒ none
            (2, 2), // ar_coeff_shift_minus_6 = 2 ⇒ ArCoeffShift = 8
            (0, 2), // grain_scale_shift = 0
            // num_cb_points==0 && num_cr_points==0 ⇒ no mult/offset
            (1, 1), // overlap_flag = 1
            (0, 1), // clip_to_restricted_range = 0
        ]);
        let (fg, n) = parse_film_grain_params(&payload, ctx).expect("decodes");
        // 1 + 16 + 4 + 1 + 2 + 2 + 2 + 2 + 1 + 1 = 32 bits.
        assert_eq!(n, 32);
        assert!(fg.apply_grain);
        assert!(fg.update_grain);
        assert!(!fg.predicted);
        assert_eq!(fg.num_y_points, 0);
        assert!(!fg.chroma_scaling_from_luma);
        assert_eq!(fg.num_cb_points, 0);
        assert_eq!(fg.num_cr_points, 0);
        assert_eq!(fg.grain_scaling, 9);
        assert_eq!(fg.ar_coeff_lag, 0);
        assert_eq!(fg.ar_coeff_shift, 8);
        assert_eq!(fg.grain_scale_shift, 0);
        assert!(fg.overlap_flag);
        assert!(!fg.clip_to_restricted_range);
    }

    #[test]
    fn film_grain_full_luma_and_chroma_points() {
        // §5.9.30 full path: 1 Y point + 1 Cb point + 1 Cr point,
        // ar_coeff_lag = 1 ⇒ numPosLuma = 2*1*2 = 4, numPosChroma = 5.
        let ctx = fg_ctx_present_shown();
        let mut fields: Vec<(u32, u32)> = vec![
            (1, 1),       // apply_grain
            (0xABCD, 16), // grain_seed
            // update_grain derived 1
            (1, 4),  // num_y_points = 1
            (10, 8), // point_y_value[0]
            (20, 8), // point_y_scaling[0]
            (0, 1),  // chroma_scaling_from_luma = 0
            // not suppressed: num_y_points>0 so 4:2:0 branch doesn't fire
            (1, 4),  // num_cb_points = 1
            (30, 8), // point_cb_value[0]
            (40, 8), // point_cb_scaling[0]
            (1, 4),  // num_cr_points = 1
            (50, 8), // point_cr_value[0]
            (60, 8), // point_cr_scaling[0]
            (3, 2),  // grain_scaling_minus_8 = 3 ⇒ 11
            (1, 2),  // ar_coeff_lag = 1 ⇒ numPosLuma = 4
        ];
        // ar_coeffs_y_plus_128[0..4] (num_y_points>0).
        for v in [100u32, 101, 102, 103] {
            fields.push((v, 8));
        }
        // numPosChroma = 5; num_cb_points>0 ⇒ cb coeffs.
        for v in [110u32, 111, 112, 113, 114] {
            fields.push((v, 8));
        }
        // num_cr_points>0 ⇒ cr coeffs.
        for v in [120u32, 121, 122, 123, 124] {
            fields.push((v, 8));
        }
        fields.push((1, 2)); // ar_coeff_shift_minus_6 = 1 ⇒ 7
        fields.push((2, 2)); // grain_scale_shift = 2
                             // num_cb_points>0 ⇒ cb mult/offset.
        fields.push((128, 8)); // cb_mult
        fields.push((192, 8)); // cb_luma_mult
        fields.push((256, 9)); // cb_offset
                               // num_cr_points>0 ⇒ cr mult/offset.
        fields.push((130, 8)); // cr_mult
        fields.push((190, 8)); // cr_luma_mult
        fields.push((300, 9)); // cr_offset
        fields.push((0, 1)); // overlap_flag
        fields.push((1, 1)); // clip_to_restricted_range
        let payload = pack_fields(&fields);
        let (fg, _n) = parse_film_grain_params(&payload, ctx).expect("decodes");
        assert!(fg.apply_grain);
        assert!(fg.update_grain);
        assert_eq!(fg.grain_seed, 0xABCD);
        assert_eq!(fg.num_y_points, 1);
        assert_eq!(fg.point_y_value[0], 10);
        assert_eq!(fg.point_y_scaling[0], 20);
        assert!(!fg.chroma_scaling_from_luma);
        assert_eq!(fg.num_cb_points, 1);
        assert_eq!(fg.point_cb_value[0], 30);
        assert_eq!(fg.point_cb_scaling[0], 40);
        assert_eq!(fg.num_cr_points, 1);
        assert_eq!(fg.point_cr_value[0], 50);
        assert_eq!(fg.point_cr_scaling[0], 60);
        assert_eq!(fg.grain_scaling, 11);
        assert_eq!(fg.ar_coeff_lag, 1);
        assert_eq!(&fg.ar_coeffs_y_plus_128[0..4], &[100, 101, 102, 103]);
        assert_eq!(&fg.ar_coeffs_cb_plus_128[0..5], &[110, 111, 112, 113, 114]);
        assert_eq!(&fg.ar_coeffs_cr_plus_128[0..5], &[120, 121, 122, 123, 124]);
        assert_eq!(fg.ar_coeff_shift, 7);
        assert_eq!(fg.grain_scale_shift, 2);
        assert_eq!(fg.cb_mult, 128);
        assert_eq!(fg.cb_luma_mult, 192);
        assert_eq!(fg.cb_offset, 256);
        assert_eq!(fg.cr_mult, 130);
        assert_eq!(fg.cr_luma_mult, 190);
        assert_eq!(fg.cr_offset, 300);
        assert!(!fg.overlap_flag);
        assert!(fg.clip_to_restricted_range);
    }

    #[test]
    fn film_grain_unexpected_end() {
        // apply_grain=1 then EOF before grain_seed ⇒ error.
        let ctx = fg_ctx_present_shown();
        let payload = pack_bits(&[1]); // only apply_grain available
        let err = parse_film_grain_params(&payload, ctx).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }
}
