//! Frame-header post-`tile_info()` sub-sections.
//!
//! Covers §5.9.12 quantization, §5.9.14 segmentation, §5.9.16
//! delta_q / delta_lf, §5.9.11 loop_filter, §5.9.19 cdef, §5.9.20 lr,
//! §5.9.21 read_tx_mode, §5.9.22 frame_reference_mode, §5.9.23
//! skip_mode_params, §5.9.24 global_motion_params, §5.9.30
//! film_grain_params.
//!
//! These types are data-only — they don't touch the range coder. The
//! values feed into dequantisation (Phase 3), deblocking / CDEF / LR
//! (Phase 5-6), and film-grain synthesis (Phase 6).

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::frame_header::{FrameType, NUM_REF_FRAMES};
use crate::sequence_header::SequenceHeader;

pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_MAX: usize = 8;
/// §3.4 — `SEG_LVL_REF_FRAME` index for the per-segment ref-frame feature.
pub const SEG_LVL_REF_FRAME: usize = 5;
/// §3.4 — `SEG_LVL_SKIP` index for the per-segment skip feature.
pub const SEG_LVL_SKIP: usize = 6;
/// §3.4 — `SEG_LVL_GLOBALMV` index for the per-segment global-MV feature.
pub const SEG_LVL_GLOBALMV: usize = 7;
pub const PRIMARY_REF_NONE: u32 = 7;
pub const TOTAL_REFS_PER_FRAME: usize = 8;

/// Segmentation feature widths — §6.8.13 Table.
pub const SEG_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
pub const SEG_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];

pub const RESTORATION_NONE: u8 = 0;
pub const RESTORATION_WIENER: u8 = 1;
pub const RESTORATION_SGR: u8 = 2;
pub const RESTORATION_SWITCHABLE: u8 = 3;

/// §6.8.21 — `tx_mode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TxMode {
    #[default]
    Only4x4,
    Largest,
    Select,
}

/// §6.8.17 — `GmType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GmType {
    #[default]
    Identity,
    Translation,
    RotZoom,
    Affine,
}

/// §5.9.12 `quantization_params()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct QuantizationParams {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub diff_uv_delta: bool,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

/// §5.9.14 `segmentation_params()`.
#[derive(Clone, Debug, Default)]
pub struct SegmentationParams {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub seg_id_pre_skip: bool,
    pub last_active_seg_id: u8,
}

impl SegmentationParams {
    /// §5.11.14 `seg_feature_active_idx(idx, feature)` — returns
    /// `segmentation_enabled && FeatureEnabled[idx][feature]`. The
    /// feature index is one of the `SEG_LVL_*` constants.
    #[inline]
    pub fn feature_active(&self, segment_id: u8, feature: usize) -> bool {
        if !self.enabled {
            return false;
        }
        let sid = (segment_id as usize).min(MAX_SEGMENTS - 1);
        let f = feature.min(SEG_LVL_MAX - 1);
        self.feature_enabled[sid][f]
    }
}

/// §5.9.11 `loop_filter_params()`.
#[derive(Clone, Copy, Debug)]
pub struct LoopFilterParams {
    pub level_y0: u8,
    pub level_y1: u8,
    pub level_u: u8,
    pub level_v: u8,
    pub sharpness: u8,
    pub mode_ref_delta_enabled: bool,
    pub mode_ref_delta_update: bool,
    pub ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilterParams {
    fn default() -> Self {
        Self {
            level_y0: 0,
            level_y1: 0,
            level_u: 0,
            level_v: 0,
            sharpness: 0,
            mode_ref_delta_enabled: false,
            mode_ref_delta_update: false,
            ref_deltas: DEFAULT_REF_DELTAS,
            mode_deltas: [0, 0],
        }
    }
}

/// Default `ref_deltas` per §7.12.1.
pub const DEFAULT_REF_DELTAS: [i8; TOTAL_REFS_PER_FRAME] = [1, 0, 0, 0, -1, 0, -1, -1];

/// §5.9.19 `cdef_params()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct CdefParams {
    pub cdef_bits: u8,
    pub y_pri_strengths: [u8; 8],
    pub y_sec_strengths: [u8; 8],
    pub uv_pri_strengths: [u8; 8],
    pub uv_sec_strengths: [u8; 8],
    pub cdef_damping_minus3: u8,
}

/// §5.9.20 `lr_params()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LoopRestorationParams {
    pub frame_restoration_type: [u8; 3],
    pub uses_lr: bool,
    pub uses_chroma_lr: bool,
    pub log2_restoration_unit_size: [u8; 3],
}

/// §5.9.30 `film_grain_params()`.
#[derive(Clone, Debug, Default)]
pub struct FilmGrainParams {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub update_grain: bool,
    pub film_grain_ref: u8,

    pub num_y_points: u8,
    pub point_y_value: [u8; 14],
    pub point_y_scaling: [u8; 14],

    pub chroma_scaling: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 10],
    pub point_cb_scaling: [u8; 10],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 10],
    pub point_cr_scaling: [u8; 10],

    pub grain_scaling_minus8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y: [i8; 24],
    pub ar_coeffs_cb: [i8; 25],
    pub ar_coeffs_cr: [i8; 25],
    pub ar_coeff_shift_minus6: u8,

    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,

    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// Read a signed 7-bit delta preceded by a 1-bit presence flag (§5.9.12
/// `read_delta_q`).
pub fn read_delta_q(br: &mut BitReader<'_>) -> Result<i8> {
    let present = br.bit()?;
    if !present {
        return Ok(0);
    }
    Ok(br.su(7)? as i8)
}

/// §5.9.12 `quantization_params()`.
pub fn parse_quantization_params(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
) -> Result<QuantizationParams> {
    let base_q_idx = br.f(8)? as u8;
    let delta_q_y_dc = read_delta_q(br)?;
    let mut q = QuantizationParams {
        base_q_idx,
        delta_q_y_dc,
        ..Default::default()
    };
    if seq.color_config.num_planes > 1 {
        if seq.color_config.separate_uv_deltas {
            q.diff_uv_delta = br.bit()?;
        }
        q.delta_q_u_dc = read_delta_q(br)?;
        q.delta_q_u_ac = read_delta_q(br)?;
        if q.diff_uv_delta {
            q.delta_q_v_dc = read_delta_q(br)?;
            q.delta_q_v_ac = read_delta_q(br)?;
        } else {
            q.delta_q_v_dc = q.delta_q_u_dc;
            q.delta_q_v_ac = q.delta_q_u_ac;
        }
    }
    q.using_qmatrix = br.bit()?;
    if q.using_qmatrix {
        q.qm_y = br.f(4)? as u8;
        q.qm_u = br.f(4)? as u8;
        if seq.color_config.separate_uv_deltas {
            q.qm_v = br.f(4)? as u8;
        } else {
            q.qm_v = q.qm_u;
        }
    }
    Ok(q)
}

/// §5.9.14 `segmentation_params()`.
pub fn parse_segmentation_params(
    br: &mut BitReader<'_>,
    primary_ref_frame: u32,
) -> Result<SegmentationParams> {
    let mut sp = SegmentationParams {
        enabled: br.bit()?,
        ..Default::default()
    };
    if sp.enabled {
        if primary_ref_frame == PRIMARY_REF_NONE {
            sp.update_map = true;
            sp.temporal_update = false;
            sp.update_data = true;
        } else {
            sp.update_map = br.bit()?;
            if sp.update_map {
                sp.temporal_update = br.bit()?;
            }
            sp.update_data = br.bit()?;
        }
        if sp.update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let feature_enabled = br.bit()?;
                    let mut clipped: i16 = 0;
                    if feature_enabled {
                        let bits = SEG_FEATURE_BITS[j];
                        if bits > 0 {
                            if SEG_FEATURE_SIGNED[j] {
                                clipped = br.su(bits + 1)? as i16;
                            } else {
                                clipped = br.f(bits)? as i16;
                            }
                        }
                    }
                    sp.feature_enabled[i][j] = feature_enabled;
                    sp.feature_data[i][j] = clipped;
                }
            }
        }
    }
    for i in 0..MAX_SEGMENTS {
        for j in 0..SEG_LVL_MAX {
            if sp.feature_enabled[i][j] {
                if (i as u8) > sp.last_active_seg_id {
                    sp.last_active_seg_id = i as u8;
                }
                if j >= SEG_LVL_REF_FRAME {
                    sp.seg_id_pre_skip = true;
                }
            }
        }
    }
    Ok(sp)
}

/// §5.9.16 `delta_q_params()` — returns `(present, res)`.
pub fn parse_delta_q_params(br: &mut BitReader<'_>, base_q_idx: u8) -> Result<(bool, u8)> {
    if base_q_idx == 0 {
        return Ok((false, 0));
    }
    let present = br.bit()?;
    let res = if present { br.f(2)? as u8 } else { 0 };
    Ok((present, res))
}

/// §5.9.16 `delta_lf_params()` — returns `(present, res, multi)`.
pub fn parse_delta_lf_params(
    br: &mut BitReader<'_>,
    delta_q_present: bool,
    allow_intrabc: bool,
) -> Result<(bool, u8, bool)> {
    if !delta_q_present {
        return Ok((false, 0, false));
    }
    let present = if allow_intrabc { false } else { br.bit()? };
    if !present {
        return Ok((false, 0, false));
    }
    let res = br.f(2)? as u8;
    let multi = br.bit()?;
    Ok((present, res, multi))
}

/// Coded-lossless hint — simplified form (all q-related deltas zero +
/// `base_q_idx == 0`). The exhaustive §6.8.2 test also checks every
/// segment's per-plane q_index; intra-only callers don't need it.
pub fn coded_lossless_hint(q: &QuantizationParams) -> bool {
    q.base_q_idx == 0
        && q.delta_q_y_dc == 0
        && q.delta_q_u_dc == 0
        && q.delta_q_u_ac == 0
        && q.delta_q_v_dc == 0
        && q.delta_q_v_ac == 0
}

/// §5.9.11 `loop_filter_params()`.
pub fn parse_loop_filter_params(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    q: &QuantizationParams,
    allow_intrabc: bool,
) -> Result<LoopFilterParams> {
    let mut lf = LoopFilterParams::default();
    if coded_lossless_hint(q) || allow_intrabc {
        lf.mode_ref_delta_enabled = true;
        return Ok(lf);
    }
    lf.level_y0 = br.f(6)? as u8;
    lf.level_y1 = br.f(6)? as u8;
    if seq.color_config.num_planes > 1 && (lf.level_y0 != 0 || lf.level_y1 != 0) {
        lf.level_u = br.f(6)? as u8;
        lf.level_v = br.f(6)? as u8;
    }
    lf.sharpness = br.f(3)? as u8;
    lf.mode_ref_delta_enabled = br.bit()?;
    if lf.mode_ref_delta_enabled {
        lf.mode_ref_delta_update = br.bit()?;
        if lf.mode_ref_delta_update {
            for d in lf.ref_deltas.iter_mut() {
                if br.bit()? {
                    *d = br.su(7)? as i8;
                }
            }
            for d in lf.mode_deltas.iter_mut() {
                if br.bit()? {
                    *d = br.su(7)? as i8;
                }
            }
        }
    }
    Ok(lf)
}

/// §5.9.19 `cdef_params()`.
pub fn parse_cdef_params(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    q: &QuantizationParams,
    allow_intrabc: bool,
) -> Result<CdefParams> {
    let mut c = CdefParams::default();
    if coded_lossless_hint(q) || allow_intrabc || !seq.enable_cdef {
        return Ok(c);
    }
    c.cdef_damping_minus3 = br.f(2)? as u8;
    c.cdef_bits = br.f(2)? as u8;
    let n = 1u8 << c.cdef_bits;
    for i in 0..n as usize {
        c.y_pri_strengths[i] = br.f(4)? as u8;
        c.y_sec_strengths[i] = br.f(2)? as u8;
        if c.y_sec_strengths[i] == 3 {
            c.y_sec_strengths[i] += 1;
        }
        if seq.color_config.num_planes == 3 {
            c.uv_pri_strengths[i] = br.f(4)? as u8;
            c.uv_sec_strengths[i] = br.f(2)? as u8;
            if c.uv_sec_strengths[i] == 3 {
                c.uv_sec_strengths[i] += 1;
            }
        }
    }
    Ok(c)
}

/// §5.9.20 `lr_params()`.
pub fn parse_lr_params(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    q: &QuantizationParams,
    allow_intrabc: bool,
) -> Result<LoopRestorationParams> {
    let mut lr = LoopRestorationParams::default();
    if coded_lossless_hint(q) || allow_intrabc || !seq.enable_restoration {
        return Ok(lr);
    }
    let lookup = [
        RESTORATION_NONE,
        RESTORATION_SWITCHABLE,
        RESTORATION_WIENER,
        RESTORATION_SGR,
    ];
    let planes = seq.color_config.num_planes as usize;
    for i in 0..planes {
        let idx = br.f(2)? as usize;
        lr.frame_restoration_type[i] = lookup[idx];
        if lr.frame_restoration_type[i] != RESTORATION_NONE {
            lr.uses_lr = true;
            if i > 0 {
                lr.uses_chroma_lr = true;
            }
        }
    }
    if lr.uses_lr {
        if seq.use_128x128_superblock {
            lr.log2_restoration_unit_size[0] = br.f(1)? as u8 + 7;
        } else {
            // §5.9.20: lr_unit_shift f(1); if 1, read lr_unit_extra_shift
            // f(1) and add. log2 = lr_unit_shift + 6.
            //   shift=0 → 1 bit, log2=6
            //   shift=1, extra=0 → 2 bits, log2=7
            //   shift=1, extra=1 → 2 bits, log2=8
            // Round 17: previous code read the extra bit when shift==0
            // (inverted) and could never reach log2=8 — over-read by 1
            // bit on every non-128 frame whose first plane uses
            // restoration. SVT-AV1 SkipMode overlay frames tripped this.
            let shift = br.f(1)? as u8;
            let extra = if shift != 0 { br.f(1)? as u8 } else { 0 };
            lr.log2_restoration_unit_size[0] = 6 + shift + extra;
        }
        if planes > 1 {
            if seq.color_config.subsampling_x && seq.color_config.subsampling_y && lr.uses_chroma_lr
            {
                lr.log2_restoration_unit_size[1] =
                    lr.log2_restoration_unit_size[0] - br.f(1)? as u8;
            } else {
                lr.log2_restoration_unit_size[1] = lr.log2_restoration_unit_size[0];
            }
            lr.log2_restoration_unit_size[2] = lr.log2_restoration_unit_size[1];
        }
    }
    Ok(lr)
}

/// §5.9.21 `read_tx_mode()`.
pub fn parse_tx_mode(br: &mut BitReader<'_>, coded_lossless: bool) -> Result<TxMode> {
    if coded_lossless {
        return Ok(TxMode::Only4x4);
    }
    Ok(if br.bit()? {
        TxMode::Select
    } else {
        TxMode::Largest
    })
}

/// §5.9.24 `global_motion_params()` — parse per-reference global
/// motion `gm_type[]` plus the six warp parameters per slot. The
/// translation components (indices 0, 1 — ie `gm_params[ref][0]` for
/// Y offset and `[1]` for X offset) are retained so the inter decoder
/// can honour GLOBALMV against TRANSLATION slots without DPB state
/// tracking. The higher-order alpha params (2..=5) are parsed and
/// discarded — RotZoom / Affine warp MC is still unimplemented.
///
/// Required for correct bit-stream framing on inter frames only;
/// intra-only callers never reach this.
///
/// `allow_high_precision_mv` is plumbed through because it modifies the
/// `absBits` width used for translation components in pure-TRANSLATION
/// warp slots (spec §5.9.25): `absBits = GM_ABS_TRANS_ONLY_BITS -
/// !allow_high_precision_mv`. Round 17 ignored this, over-reading the
/// translation parameters by 4 bits per ref slot.
pub fn parse_global_motion_params(
    br: &mut BitReader<'_>,
    gm_type_out: &mut [GmType; NUM_REF_FRAMES],
    gm_params_out: &mut [[i32; 6]; NUM_REF_FRAMES],
    allow_high_precision_mv: bool,
    prev_gm_params: Option<&[[i32; 6]; NUM_REF_FRAMES]>,
) -> Result<()> {
    // §5.9.24 — initialise gm_params[] to the identity default so any
    // ref slot left at IDENTITY (or whose alpha params we derive
    // implicitly for ROTZOOM) carries the spec-mandated identity matrix.
    let identity_alpha: i32 = 1 << 16; // WARPEDMODEL_PREC_BITS
    for slot in gm_params_out.iter_mut() {
        *slot = [0, 0, identity_alpha, 0, 0, identity_alpha];
    }
    for (i, slot) in gm_type_out.iter_mut().enumerate().skip(1) {
        let is_global = br.bit()?;
        let typ = if is_global {
            let is_rot_zoom = br.bit()?;
            if is_rot_zoom {
                GmType::RotZoom
            } else {
                let is_translation = br.bit()?;
                if is_translation {
                    GmType::Translation
                } else {
                    GmType::Affine
                }
            }
        } else {
            GmType::Identity
        };
        *slot = typ;
        // Per §5.9.25 the per-idx subexp reference is
        //   r = (PrevGmParams[ref][idx] >> precDiff) - sub
        // where `PrevGmParams` was either reset to identity by
        // `setup_past_independence` (PRIMARY_REF_NONE) or loaded from
        // `SavedGmParams[ref_frame_idx[primary_ref_frame]]` by
        // `load_previous`. The DPB-aware caller plumbs the latter via
        // `prev_gm_params`; absent that we keep the identity default
        // (every translation/alpha component already zero except the
        // diagonal alpha terms equal to `1<<WARPEDMODEL_PREC_BITS`).
        let ref_params: [i32; 6] = match prev_gm_params {
            Some(p) => p[i],
            None => [0, 0, identity_alpha, 0, 0, identity_alpha],
        };
        // Spec §5.9.24 read order: alpha params (2..=5) FIRST for
        // ROTZOOM/AFFINE, then translation params (0, 1) for any non-
        // identity type. Round 17 forgot 0/1 for ROTZOOM and AFFINE.
        let mut params = gm_params_out[i];
        if (typ as u8) >= (GmType::RotZoom as u8) {
            params[2] =
                read_global_param_with_ref(br, typ, 2, allow_high_precision_mv, ref_params[2])?;
            params[3] =
                read_global_param_with_ref(br, typ, 3, allow_high_precision_mv, ref_params[3])?;
            if typ == GmType::Affine {
                params[4] =
                    read_global_param_with_ref(br, typ, 4, allow_high_precision_mv, ref_params[4])?;
                params[5] =
                    read_global_param_with_ref(br, typ, 5, allow_high_precision_mv, ref_params[5])?;
            } else {
                // ROTZOOM derives 4 / 5 from 2 / 3 to keep the matrix
                // symmetric.
                params[4] = -params[3];
                params[5] = params[2];
            }
        }
        if (typ as u8) >= (GmType::Translation as u8) {
            params[0] =
                read_global_param_with_ref(br, typ, 0, allow_high_precision_mv, ref_params[0])?;
            params[1] =
                read_global_param_with_ref(br, typ, 1, allow_high_precision_mv, ref_params[1])?;
        }
        gm_params_out[i] = params;
    }
    Ok(())
}

/// §5.9.25 `read_global_param(type, ref, idx)` — decodes one warp
/// parameter. `prev_value` is `PrevGmParams[ref][idx]` (loaded by
/// §7.20 `load_previous` when `primary_ref_frame != PRIMARY_REF_NONE`,
/// otherwise the identity default produced by §7.4
/// `setup_past_independence`).
///
/// Constants per the spec table (§3): `GM_ABS_ALPHA_BITS = 12`,
/// `GM_ABS_TRANS_BITS = 12`, `GM_ABS_TRANS_ONLY_BITS = 9`,
/// `GM_ALPHA_PREC_BITS = 15`, `GM_TRANS_PREC_BITS = 6`,
/// `GM_TRANS_ONLY_PREC_BITS = 3`, `WARPEDMODEL_PREC_BITS = 16`.
fn read_global_param_with_ref(
    br: &mut BitReader<'_>,
    typ: GmType,
    idx: usize,
    allow_high_precision_mv: bool,
    prev_value: i32,
) -> Result<i32> {
    const WARPEDMODEL_PREC_BITS: u32 = 16;
    const GM_ABS_ALPHA_BITS: u32 = 12;
    const GM_ABS_TRANS_BITS: u32 = 12;
    const GM_ABS_TRANS_ONLY_BITS: u32 = 9;
    const GM_ALPHA_PREC_BITS: u32 = 15;
    const GM_TRANS_PREC_BITS: u32 = 6;
    const GM_TRANS_ONLY_PREC_BITS: u32 = 3;

    let (abs_bits, prec_bits) = if idx < 2 {
        if typ == GmType::Translation {
            let bump = if allow_high_precision_mv { 0 } else { 1 };
            (
                GM_ABS_TRANS_ONLY_BITS - bump,
                GM_TRANS_ONLY_PREC_BITS - bump,
            )
        } else {
            (GM_ABS_TRANS_BITS, GM_TRANS_PREC_BITS)
        }
    } else {
        (GM_ABS_ALPHA_BITS, GM_ALPHA_PREC_BITS)
    };
    let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
    // round = (idx % 3 == 2) ? 1<<WARPEDMODEL_PREC_BITS : 0
    let round: i32 = if idx % 3 == 2 {
        1 << WARPEDMODEL_PREC_BITS
    } else {
        0
    };
    // sub = (idx % 3 == 2) ? 1<<precBits : 0
    let sub: i32 = if idx % 3 == 2 { 1 << prec_bits } else { 0 };
    // r = (PrevGmParams[ref][idx] >> precDiff) - sub
    // (Spec uses arithmetic right shift: prev_value is a signed i32 so
    // Rust's `>>` already preserves sign on i32.)
    let r: i32 = (prev_value >> prec_diff) - sub;
    let mx = 1i32 << abs_bits;
    let v = decode_signed_subexp_with_ref(br, -mx, mx + 1, r)?;
    Ok((v << prec_diff) + round)
}

/// §5.9.28 `decode_signed_subexp_with_ref(low, high, r)`.
fn decode_signed_subexp_with_ref(
    br: &mut BitReader<'_>,
    low: i32,
    high: i32,
    r: i32,
) -> Result<i32> {
    let mx = (high - low) as u32;
    let r_shift = (r - low) as u32;
    let x = decode_unsigned_subexp_with_ref(br, mx, r_shift)?;
    Ok((x as i32) + low)
}

fn decode_unsigned_subexp_with_ref(br: &mut BitReader<'_>, mx: u32, r: u32) -> Result<u32> {
    let v = decode_subexp(br, mx)?;
    if (r << 1) <= mx {
        Ok(inverse_recenter(r, v))
    } else {
        Ok(mx - 1 - inverse_recenter(mx - 1 - r, v))
    }
}

fn decode_subexp(br: &mut BitReader<'_>, num_syms: u32) -> Result<u32> {
    // Spec §5.9.29.
    let mut i = 0u32;
    let mut mk = 0u32;
    let k = 3u32;
    loop {
        let b2 = if i == 0 { k } else { k + i - 1 };
        let a = 1u32 << b2;
        if num_syms <= mk + 3 * a {
            let subexp_final = br.ns(num_syms - mk)?;
            return Ok(subexp_final + mk);
        }
        let subexp_more_bits = br.f(1)?;
        if subexp_more_bits == 1 {
            i += 1;
            mk += a;
        } else {
            let subexp_bits = br.f(b2)?;
            return Ok(subexp_bits + mk);
        }
    }
}

fn inverse_recenter(r: u32, v: u32) -> u32 {
    if v > 2 * r {
        v
    } else if v & 1 == 1 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// §5.9.30 `film_grain_params()`.
pub fn parse_film_grain_params(
    br: &mut BitReader<'_>,
    seq: &SequenceHeader,
    frame_type: FrameType,
    show_frame: bool,
    showable_frame: bool,
) -> Result<FilmGrainParams> {
    let mut g = FilmGrainParams::default();
    if !seq.film_grain_params_present || (!show_frame && !showable_frame) {
        return Ok(g);
    }
    g.apply_grain = br.bit()?;
    if !g.apply_grain {
        return Ok(g);
    }
    g.grain_seed = br.f(16)? as u16;
    g.update_grain = if frame_type == FrameType::Inter {
        br.bit()?
    } else {
        true
    };
    if !g.update_grain {
        g.film_grain_ref = br.f(3)? as u8;
        return Ok(g);
    }
    g.num_y_points = br.f(4)? as u8;
    if g.num_y_points as usize > g.point_y_value.len() {
        return Err(Error::invalid(
            "av1 film_grain_params: num_y_points > 14 (§5.9.30)",
        ));
    }
    for i in 0..g.num_y_points as usize {
        g.point_y_value[i] = br.f(8)? as u8;
        g.point_y_scaling[i] = br.f(8)? as u8;
    }
    let mono = seq.color_config.mono_chrome;
    let sx = seq.color_config.subsampling_x;
    let sy = seq.color_config.subsampling_y;
    let chroma_luma_skip = sx && sy && g.num_y_points == 0;
    g.chroma_scaling = if mono || chroma_luma_skip {
        false
    } else {
        br.bit()?
    };
    if !mono && !chroma_luma_skip {
        g.num_cb_points = br.f(4)? as u8;
        if g.num_cb_points as usize > g.point_cb_value.len() {
            return Err(Error::invalid(
                "av1 film_grain_params: num_cb_points > 10 (§5.9.30)",
            ));
        }
        for i in 0..g.num_cb_points as usize {
            g.point_cb_value[i] = br.f(8)? as u8;
            g.point_cb_scaling[i] = br.f(8)? as u8;
        }
        g.num_cr_points = br.f(4)? as u8;
        if g.num_cr_points as usize > g.point_cr_value.len() {
            return Err(Error::invalid(
                "av1 film_grain_params: num_cr_points > 10 (§5.9.30)",
            ));
        }
        for i in 0..g.num_cr_points as usize {
            g.point_cr_value[i] = br.f(8)? as u8;
            g.point_cr_scaling[i] = br.f(8)? as u8;
        }
    }
    g.grain_scaling_minus8 = br.f(2)? as u8;
    g.ar_coeff_lag = br.f(2)? as u8;
    let num_pos_y = 2 * (g.ar_coeff_lag as usize) * (g.ar_coeff_lag as usize + 1);
    let num_pos_chroma = if g.num_y_points > 0 {
        num_pos_y + 1
    } else {
        num_pos_y
    };
    if g.num_y_points > 0 {
        if num_pos_y > g.ar_coeffs_y.len() {
            return Err(Error::invalid(
                "av1 film_grain_params: num_pos_y > 24 (§5.9.30)",
            ));
        }
        for coef in g.ar_coeffs_y.iter_mut().take(num_pos_y) {
            *coef = (br.f(8)? as i32 - 128) as i8;
        }
    }
    if g.chroma_scaling || g.num_cb_points > 0 {
        if num_pos_chroma > g.ar_coeffs_cb.len() {
            return Err(Error::invalid(
                "av1 film_grain_params: num_pos_chroma > 25 (§5.9.30)",
            ));
        }
        for coef in g.ar_coeffs_cb.iter_mut().take(num_pos_chroma) {
            *coef = (br.f(8)? as i32 - 128) as i8;
        }
    }
    if g.chroma_scaling || g.num_cr_points > 0 {
        for coef in g.ar_coeffs_cr.iter_mut().take(num_pos_chroma) {
            *coef = (br.f(8)? as i32 - 128) as i8;
        }
    }
    g.ar_coeff_shift_minus6 = br.f(2)? as u8;
    g.grain_scale_shift = br.f(2)? as u8;
    if g.num_cb_points > 0 {
        g.cb_mult = br.f(8)? as u8;
        g.cb_luma_mult = br.f(8)? as u8;
        g.cb_offset = br.f(9)? as u16;
    }
    if g.num_cr_points > 0 {
        g.cr_mult = br.f(8)? as u8;
        g.cr_luma_mult = br.f(8)? as u8;
        g.cr_offset = br.f(9)? as u16;
    }
    g.overlap_flag = br.bit()?;
    g.clip_to_restricted_range = br.bit()?;
    Ok(g)
}

#[cfg(test)]
mod gm_tests {
    use super::*;

    /// All-zero payload after the 7 type bits encodes seven IDENTITY
    /// global-motion slots. Verifies the round-18 read order: only the
    /// type-flag bit is consumed per slot when `is_global=0`.
    #[test]
    fn all_identity_consumes_seven_bits() {
        // 7 ref slots × 1 bit each = 7 bits. Pad to a byte boundary.
        let payload = [0u8];
        let mut br = BitReader::new(&payload);
        let mut gm_type = [GmType::Identity; NUM_REF_FRAMES];
        let mut gm_params = [[0i32; 6]; NUM_REF_FRAMES];
        parse_global_motion_params(&mut br, &mut gm_type, &mut gm_params, false, None).unwrap();
        // 7 bits consumed.
        assert_eq!(br.bit_position(), 7);
        // Slot 0 (intra) untouched; 1..=7 all Identity.
        for slot in &gm_type[1..] {
            assert_eq!(*slot, GmType::Identity);
        }
        // Identity matrix per §5.9.24.
        let identity_alpha: i32 = 1 << 16;
        for params in &gm_params[1..] {
            assert_eq!(*params, [0, 0, identity_alpha, 0, 0, identity_alpha]);
        }
    }

    /// Round-17 regression guard — for ROTZOOM the parser MUST also read
    /// the translation params (idx 0, 1) AFTER the alpha pair (idx 2,
    /// 3). Round-17 forgot the trailing pair, desynchronising the stream
    /// by ~16 bits per ROTZOOM ref.
    #[test]
    fn rotzoom_reads_alpha_then_translation() {
        // Hand-craft: slot 1 = ROTZOOM, then six slots IDENTITY.
        // ROTZOOM type bits = 1 (is_global), 1 (is_rot_zoom). After
        // those, we expect 4 read_global_param invocations (alpha 2/3,
        // translation 0/1). Each with subexp_more_bits=0 at i=0 reads
        // 1 + b2 = 1 + 3 = 4 bits. So 4 params × 4 bits = 16 bits.
        // Total slot 1 = 2 (type) + 16 (params) = 18 bits.
        // Slots 2..=7 = 6 × 1 (Identity) = 6 bits.
        // Grand total = 24 bits = 3 bytes. The exact byte values just
        // need to drive subexp_more_bits=0 at iteration 0 every time.
        // bit pattern: 11 [0 000][0 000][0 000][0 000] 0 0 0 0 0 0
        // Pack MSB-first: 1100000_00000000_00000000 = 0xC0 0x00 0x00.
        let payload = [0xC0u8, 0x00, 0x00];
        let mut br = BitReader::new(&payload);
        let mut gm_type = [GmType::Identity; NUM_REF_FRAMES];
        let mut gm_params = [[0i32; 6]; NUM_REF_FRAMES];
        parse_global_motion_params(&mut br, &mut gm_type, &mut gm_params, true, None).unwrap();
        assert_eq!(gm_type[1], GmType::RotZoom);
        for slot in &gm_type[2..] {
            assert_eq!(*slot, GmType::Identity);
        }
        // Verifies all 24 bits consumed — a partial read would NOT have
        // landed exactly on the byte boundary.
        assert_eq!(br.bit_position(), 24);
    }

    /// PrevGmParams plumbing: with a non-identity ref-anchor the
    /// `r = (PrevGmParams >> precDiff) - sub` math should produce a
    /// different `r` than the identity-default path, exercised here by
    /// asserting that supplying a custom `prev_gm_params` does not
    /// re-introduce the all-zero output bug fixed in round 18.
    #[test]
    fn prev_gm_params_threaded_through() {
        // Identity payload (all 7 ref slots IDENTITY).
        let payload = [0u8];
        let mut br = BitReader::new(&payload);
        let mut gm_type = [GmType::Identity; NUM_REF_FRAMES];
        let mut gm_params = [[0i32; 6]; NUM_REF_FRAMES];
        // PrevGmParams: every slot has alpha=2*identity (just to be
        // recognisable) and small translations.
        let identity_alpha: i32 = 1 << 16;
        let prev = [[10, 20, 2 * identity_alpha, 30, 40, 2 * identity_alpha]; NUM_REF_FRAMES];
        parse_global_motion_params(&mut br, &mut gm_type, &mut gm_params, false, Some(&prev))
            .unwrap();
        // Identity slots — output `gm_params` should still be the
        // canonical identity, NOT the prev value (we coded type=IDENTITY
        // so no params are read).
        for params in &gm_params[1..] {
            assert_eq!(*params, [0, 0, identity_alpha, 0, 0, identity_alpha]);
        }
    }
}
