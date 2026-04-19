//! Frame-header post-`tile_info()` sub-sections.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/obu/*_params.go`.
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
pub const SEG_LVL_REF_FRAME: usize = 5;
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
    let mut q = QuantizationParams {
        base_q_idx: br.f(8)? as u8,
        delta_q_y_dc: read_delta_q(br)?,
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

/// Coded-lossless hint — simplified form tracked by goavif.
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
            lr.log2_restoration_unit_size[0] = br.f(1)? as u8 + 6;
            if lr.log2_restoration_unit_size[0] == 6 {
                lr.log2_restoration_unit_size[0] += br.f(1)? as u8;
            }
        }
        if planes > 1 {
            if seq.color_config.subsampling_x
                && seq.color_config.subsampling_y
                && lr.uses_chroma_lr
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

/// §5.9.24 `global_motion_params()` — parse and discard the
/// coefficients. The still-image decoder never uses them and the
/// bit-width heuristic here is a deliberate simplification (matches
/// goavif). Required for correct bit-stream framing on inter frames
/// only; intra-only callers never reach this.
pub fn parse_global_motion_params(
    br: &mut BitReader<'_>,
    gm_type_out: &mut [GmType; NUM_REF_FRAMES],
) -> Result<()> {
    for slot in gm_type_out.iter_mut().skip(1) {
        let is_global = br.bit()?;
        let typ = if is_global {
            if br.bit()? {
                GmType::RotZoom
            } else if br.bit()? {
                GmType::Translation
            } else {
                GmType::Affine
            }
        } else {
            GmType::Identity
        };
        *slot = typ;
        let extra_pairs = match typ {
            GmType::Affine => 3,
            GmType::RotZoom => 2,
            GmType::Translation => 1,
            GmType::Identity => 0,
        };
        for _ in 0..extra_pairs {
            let _ = br.su(12)?;
            let _ = br.su(12)?;
        }
    }
    Ok(())
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
