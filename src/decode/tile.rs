//! AV1 tile decoder — §5.11 `decode_tile`.
//!
//! Phase 2 keeps only the symbol decoder + the CDF bank required by
//! the partition / mode / segment / CFL / skip readers. Coefficient
//! decode and inter-frame syntax live in sibling modules.
//!
//! The top-level entry point is [`decode_tile_group`]. It iterates
//! over every tile in the `OBU_FRAME` / `OBU_TILE_GROUP` payload and
//! walks each tile's superblock grid (§5.11.2 `decode_tile`).
//! Whenever a non-skip leaf is reached the decoder returns
//! `Error::Unsupported("av1 coefficient decode pending (§5.11.39)")`
//! because Phase 3 will wire up the coefficient reader; until then
//! the bitstream is valid but no pixels are produced.

use oxideav_core::{Error, Result};

use crate::cdfs;
use crate::dpb::Dpb;
use crate::frame_header::FrameHeader;
use crate::frame_header_tail::RESTORATION_NONE;
use crate::sequence_header::SequenceHeader;
use crate::symbol::SymbolDecoder;
use crate::tile_group::{parse_tile_group_header, split_tile_payloads, TilePayload};

use super::coeffs::{q_index_to_ctx, CoeffCdfBank};
use super::frame_state::FrameState;
use super::inter::InterDecoder;
use super::lr_unit::{
    decode_lr_unit as decode_lr_unit_impl, default_sgrproj_cdf, default_switchable_cdf,
    default_wiener_cdf, LrRef,
};
use super::modes::{IntraMode, INTRA_MODES, UV_MODES};
use super::superblock;
use super::tx_type_map::{ext_tx_set_for_inter, inter_tx_type_for};
use crate::frame_header::FrameType;
use crate::lr::UnitParams as LrUnitParams;
use crate::transform::TxType;

use std::sync::Arc;

/// §9.4.1 `FRAME_LF_COUNT` — the number of per-frame loop-filter
/// strength values. Used to size the delta-lf-multi CDF / state
/// arrays. Matches the spec's constants table.
pub const FRAME_LF_COUNT: usize = 4;

/// §9.4.1 `DELTA_Q_SMALL` — symbol value that indicates the
/// alternative literal-bits encoding of `delta_q_abs`. Shared with
/// `DELTA_LF_SMALL` per the spec.
pub const DELTA_Q_SMALL: u32 = 3;

/// §9.4.1 `DELTA_LF_SMALL` — mirrors [`DELTA_Q_SMALL`] for the LF
/// delta symbol.
pub const DELTA_LF_SMALL: u32 = 3;

/// §9.4.1 `MAX_LOOP_FILTER` — saturation bound for the clipped LF
/// delta accumulator. Value is 63 per the spec constants table.
pub const MAX_LOOP_FILTER: i32 = 63;

/// Top-level entry point: walk every tile in a tile-group payload,
/// running [`TileDecoder::decode`] on each. The underlying frame
/// state is mutated in place.
///
/// `tile_payload` is the entire tile-group OBU payload (header +
/// per-tile size prefixes + concatenated tile data, as surfaced by
/// [`crate::tile_group::split_tile_payloads`]).
///
/// `prev_frame`, if present, supplies the LAST reference for single-
/// reference translational inter blocks. Intra frames ignore it; non-
/// intra frames without a reference surface `Error::Unsupported`.
///
/// `dpb` carries the §7.20 reconstructed picture buffer so SKIP_MODE
/// compound MC can fetch from `SkipModeFrame[0..=1]` independently
/// (Round 14 wiring).
pub fn decode_tile_group(
    seq: &SequenceHeader,
    frame: &FrameHeader,
    tile_payload: &[u8],
    frame_state: &mut FrameState,
    prev_frame: Option<&Arc<FrameState>>,
    dpb: &Dpb,
) -> Result<()> {
    let tile_info = frame.tile_info.as_ref().ok_or_else(|| {
        Error::invalid("av1 decode_tile_group: frame header missing tile_info (§5.9.15)")
    })?;
    // Allocate per-plane LR unit storage so the superblock walker can
    // index into it. Must happen before any tile decodes.
    alloc_lr_units(seq, frame, frame_state);
    let tgh = parse_tile_group_header(tile_payload, tile_info)?;
    let tiles = split_tile_payloads(tile_payload, tile_info, &tgh)?;
    for tp in &tiles {
        let bytes = &tile_payload[tp.offset..tp.offset + tp.len];
        let mut td = TileDecoder::new(seq, frame, bytes, prev_frame.cloned(), dpb.clone())?;
        td.decode(frame_state, tp)?;
    }
    finish_frame(frame, frame_state);
    Ok(())
}

/// Size per-plane `lr_unit_info` storage from `log2_restoration_unit_size`
/// (§5.9.20) so the superblock walker can index into it. Planes whose
/// `FrameRestorationType == RESTORE_NONE` still get a 1×1 slot so
/// out-of-range indexing clamps cleanly.
fn alloc_lr_units(_seq: &SequenceHeader, frame: &FrameHeader, fs: &mut FrameState) {
    let num_planes = if fs.monochrome { 1 } else { 3 };
    for plane in 0..num_planes {
        let log2 = frame.lr.log2_restoration_unit_size[plane];
        let (pw, ph) = if plane == 0 {
            (fs.width, fs.height)
        } else {
            (fs.uv_width, fs.uv_height)
        };
        let unit_size = if log2 == 0 { 64 } else { 1u32 << log2 };
        // libaom `av1_lr_count_units`: round to nearest, minimum 1.
        let horz = ((pw + unit_size / 2) / unit_size).max(1);
        let vert = ((ph + unit_size / 2) / unit_size).max(1);
        fs.alloc_lr_units(plane, unit_size, horz, vert);
    }
}

/// Run the post-reconstruction passes — deblocking, CDEF, loop
/// restoration, then film grain — over the fully reconstructed
/// planes. Called automatically by [`decode_tile_group`] after the
/// last tile completes.
///
/// Loop restoration consumes the per-unit parameters the superblock
/// walker decoded into `fs.lr_unit_info[]` (§5.11.40-.44). Film grain
/// synthesises the §7.20.2 32×32 tiled noise pattern when the frame
/// header signals `apply_grain`.
pub fn finish_frame(frame: &FrameHeader, fs: &mut FrameState) {
    apply_deblocking(frame, fs);
    apply_cdef(frame, fs);
    apply_lr(frame, fs);
    apply_film_grain(frame, fs);
}

fn apply_deblocking(frame: &FrameHeader, fs: &mut FrameState) {
    use crate::loopfilter::{
        apply_plane_edges, apply_plane_edges16, EdgePlane, EdgePlane16, LfModeType, MiGrid, MiInfo,
        INTRA_FRAME, LAST_FRAME,
    };
    let lf = &frame.loop_filter;
    let seg = &frame.segmentation;
    if lf.level_y0 == 0 && lf.level_y1 == 0 && lf.level_u == 0 && lf.level_v == 0 {
        return;
    }
    // Translate the FrameState's MI grid to a per-cell `MiInfo`
    // that the edge driver can consume. Block / TX dimensions come
    // from the per-MI `mi_size_idx` and `tx_size` slots stamped by
    // the leaf walker; missing entries default to 4×4.
    let mi_cols = fs.mi_cols as usize;
    let mi_rows = fs.mi_rows as usize;
    let mut cells = Vec::with_capacity(mi_cols * mi_rows);
    for r in 0..mi_rows {
        for c in 0..mi_cols {
            let m = &fs.mi[r * mi_cols + c];
            let (block_w, block_h) = block_dims_from_idx(m.mi_size_idx);
            let (tx_w, tx_h) = match m.tx_size {
                Some(ts) => (ts.width(), ts.height()),
                None => (4, 4),
            };
            let (ref_frame, mode_type) = if m.is_inter {
                // Narrow inter path always rides on LAST_FRAME (idx 0).
                // GLOBALMV / GLOBAL_GLOBALMV would keep modeType==0
                // per §7.14.4; without an explicit mode slot we infer
                // from a zero MV (good enough for the zero-MV cases
                // SVT-AV1's narrow path produces).
                let mt = if m.mv_row == 0 && m.mv_col == 0 {
                    LfModeType::Zero
                } else {
                    LfModeType::One
                };
                (LAST_FRAME, mt)
            } else {
                (INTRA_FRAME, LfModeType::Zero)
            };
            cells.push(MiInfo {
                ref_frame,
                mode_type,
                skip: m.skip,
                segment_id: m.segment_id,
                tx_w: tx_w.min(255) as u8,
                tx_h: tx_h.min(255) as u8,
                block_w: block_w.min(255) as u8,
                block_h: block_h.min(255) as u8,
                delta_lf: 0,
            });
        }
    }
    let grid = MiGrid {
        cells: &cells,
        mi_cols,
        mi_rows,
        sub_x: fs.sub_x as usize,
        sub_y: fs.sub_y as usize,
    };

    if fs.bit_depth == 8 {
        let width = fs.width as usize;
        let height = fs.height as usize;
        let uvw = fs.uv_width as usize;
        let uvh = fs.uv_height as usize;
        apply_plane_edges(
            EdgePlane {
                pix: &mut fs.y_plane,
                stride: width,
                width,
                height,
            },
            0,
            &grid,
            lf,
            seg,
        );
        if !fs.monochrome {
            apply_plane_edges(
                EdgePlane {
                    pix: &mut fs.u_plane,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                },
                1,
                &grid,
                lf,
                seg,
            );
            apply_plane_edges(
                EdgePlane {
                    pix: &mut fs.v_plane,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                },
                2,
                &grid,
                lf,
                seg,
            );
        }
    } else {
        let width = fs.width as usize;
        let height = fs.height as usize;
        let uvw = fs.uv_width as usize;
        let uvh = fs.uv_height as usize;
        let bd = fs.bit_depth;
        apply_plane_edges16(
            EdgePlane16 {
                pix: &mut fs.y_plane16,
                stride: width,
                width,
                height,
                bit_depth: bd,
            },
            0,
            &grid,
            lf,
            seg,
        );
        if !fs.monochrome {
            apply_plane_edges16(
                EdgePlane16 {
                    pix: &mut fs.u_plane16,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                    bit_depth: bd,
                },
                1,
                &grid,
                lf,
                seg,
            );
            apply_plane_edges16(
                EdgePlane16 {
                    pix: &mut fs.v_plane16,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                    bit_depth: bd,
                },
                2,
                &grid,
                lf,
                seg,
            );
        }
    }
}

fn apply_cdef(frame: &FrameHeader, fs: &mut FrameState) {
    use crate::cdef::{
        apply_frame_spec, apply_frame_spec16, Plane as CdefPlane, Plane16 as CdefPlane16,
        SbStrengths,
    };
    let params = &frame.cdef;
    // The spec (§7.15) dispatches CDEF per 64×64 luma SB using the
    // per-SB `cdef_idx` stamped in by the leaf-block walker. Any SB
    // whose `cdef_idx` is still `-1` (skip-block SBs, lossless, intrabc
    // path, or disabled) is passed through unchanged.
    let damping_base = (params.cdef_damping_minus3 as i32) + 3;
    let cdef_idx = fs.cdef_idx.clone();
    let cdef_cols = fs.cdef_sb_cols as usize;
    let strengths = move |sb_col: usize, sb_row: usize| -> Option<SbStrengths> {
        let row = sb_row.min((fs_cdef_rows(cdef_idx.len(), cdef_cols)).saturating_sub(1));
        let col = sb_col.min(cdef_cols.saturating_sub(1));
        let idx = cdef_idx.get(row * cdef_cols + col).copied().unwrap_or(-1);
        if idx < 0 {
            return None;
        }
        let i = (idx as usize) & 0x7;
        Some(SbStrengths {
            pri_y: params.y_pri_strengths[i] as i32,
            sec_y: params.y_sec_strengths[i] as i32,
            pri_uv: params.uv_pri_strengths[i] as i32,
            sec_uv: params.uv_sec_strengths[i] as i32,
            damping: damping_base,
        })
    };

    let sb_size_luma = 64usize; // §7.15 uses 64×64 CDEF blocks regardless of SB_128 mode.
    let width = fs.width as usize;
    let height = fs.height as usize;
    let uvw = fs.uv_width as usize;
    let uvh = fs.uv_height as usize;
    let sub_x = fs.sub_x;
    let sub_y = fs.sub_y;
    let bit_depth = fs.bit_depth;
    let mut dir_table = Vec::new();
    let sb_cols = (width + 63) >> 6;
    let sb_rows = (height + 63) >> 6;

    if bit_depth == 8 {
        apply_frame_spec(
            CdefPlane {
                pix: &mut fs.y_plane,
                stride: width,
                width,
                height,
            },
            0,
            0,
            0,
            sb_size_luma,
            &mut dir_table,
            sb_cols,
            sb_rows,
            0,
            &strengths,
        );
        if !fs.monochrome {
            apply_frame_spec(
                CdefPlane {
                    pix: &mut fs.u_plane,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                },
                1,
                sub_x,
                sub_y,
                sb_size_luma,
                &mut dir_table,
                sb_cols,
                sb_rows,
                0,
                &strengths,
            );
            apply_frame_spec(
                CdefPlane {
                    pix: &mut fs.v_plane,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                },
                2,
                sub_x,
                sub_y,
                sb_size_luma,
                &mut dir_table,
                sb_cols,
                sb_rows,
                0,
                &strengths,
            );
        }
    } else {
        apply_frame_spec16(
            CdefPlane16 {
                pix: &mut fs.y_plane16,
                stride: width,
                width,
                height,
            },
            0,
            0,
            0,
            sb_size_luma,
            &mut dir_table,
            sb_cols,
            sb_rows,
            bit_depth,
            &strengths,
        );
        if !fs.monochrome {
            apply_frame_spec16(
                CdefPlane16 {
                    pix: &mut fs.u_plane16,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                },
                1,
                sub_x,
                sub_y,
                sb_size_luma,
                &mut dir_table,
                sb_cols,
                sb_rows,
                bit_depth,
                &strengths,
            );
            apply_frame_spec16(
                CdefPlane16 {
                    pix: &mut fs.v_plane16,
                    stride: uvw,
                    width: uvw,
                    height: uvh,
                },
                2,
                sub_x,
                sub_y,
                sb_size_luma,
                &mut dir_table,
                sb_cols,
                sb_rows,
                bit_depth,
                &strengths,
            );
        }
    }
}

fn fs_cdef_rows(total: usize, cols: usize) -> usize {
    total.checked_div(cols).unwrap_or(0)
}

/// Translate a stored `mi_size_idx` (matches [`crate::decode::block::BlockSize`]
/// discriminant) to a `(width, height)` pair in luma samples. Falls
/// back to 4×4 for unknown / `Invalid` values so the loop filter
/// stays conservative.
fn block_dims_from_idx(idx: u8) -> (usize, usize) {
    use crate::decode::block::BlockSize;
    let bs = match idx {
        0 => BlockSize::Block4x4,
        1 => BlockSize::Block4x8,
        2 => BlockSize::Block8x4,
        3 => BlockSize::Block8x8,
        4 => BlockSize::Block8x16,
        5 => BlockSize::Block16x8,
        6 => BlockSize::Block16x16,
        7 => BlockSize::Block16x32,
        8 => BlockSize::Block32x16,
        9 => BlockSize::Block32x32,
        10 => BlockSize::Block32x64,
        11 => BlockSize::Block64x32,
        12 => BlockSize::Block64x64,
        13 => BlockSize::Block64x128,
        14 => BlockSize::Block128x64,
        15 => BlockSize::Block128x128,
        16 => BlockSize::Block4x16,
        17 => BlockSize::Block16x4,
        18 => BlockSize::Block8x32,
        19 => BlockSize::Block32x8,
        20 => BlockSize::Block16x64,
        21 => BlockSize::Block64x16,
        _ => return (4, 4),
    };
    (bs.width() as usize, bs.height() as usize)
}

/// Apply §5.11.40-.44 loop restoration across every plane whose
/// `FrameRestorationType != RESTORE_NONE`, consulting the per-unit
/// parameters that the superblock walker stored in `fs.lr_unit_info`.
fn apply_lr(frame: &FrameHeader, fs: &mut FrameState) {
    use crate::lr::{apply_frame, apply_frame16, Plane, Plane16};

    let lr = &frame.lr;
    if !lr.uses_lr {
        return;
    }
    let num_planes = if fs.monochrome { 1 } else { 3 };
    for plane in 0..num_planes {
        if lr.frame_restoration_type[plane] == crate::frame_header_tail::RESTORATION_NONE {
            continue;
        }
        let unit_size = fs.lr_unit_size[plane] as usize;
        if unit_size == 0 {
            continue;
        }
        let cols = fs.lr_cols[plane] as usize;
        let units: Vec<crate::lr::UnitParams> = fs.lr_unit_info[plane].clone();
        let params_for = |c: usize, r: usize| -> crate::lr::UnitParams {
            let idx = r * cols + c;
            units.get(idx).copied().unwrap_or_default()
        };
        if fs.bit_depth == 8 {
            let (pix, width, height) = match plane {
                0 => (&mut fs.y_plane, fs.width as usize, fs.height as usize),
                1 => (&mut fs.u_plane, fs.uv_width as usize, fs.uv_height as usize),
                _ => (&mut fs.v_plane, fs.uv_width as usize, fs.uv_height as usize),
            };
            let stride = width;
            apply_frame(
                Plane {
                    pix,
                    stride,
                    width,
                    height,
                },
                unit_size,
                params_for,
            );
        } else {
            let (pix, width, height) = match plane {
                0 => (&mut fs.y_plane16, fs.width as usize, fs.height as usize),
                1 => (
                    &mut fs.u_plane16,
                    fs.uv_width as usize,
                    fs.uv_height as usize,
                ),
                _ => (
                    &mut fs.v_plane16,
                    fs.uv_width as usize,
                    fs.uv_height as usize,
                ),
            };
            let stride = width;
            apply_frame16(
                Plane16 {
                    pix,
                    stride,
                    width,
                    height,
                },
                unit_size,
                params_for,
                fs.bit_depth,
            );
        }
    }
}

/// Run §7.20.2 film-grain synthesis on the reconstructed frame when
/// the header carries a non-zero `grain_seed`.
fn apply_film_grain(frame: &FrameHeader, fs: &mut FrameState) {
    use crate::filmgrain::{
        apply_with_template, apply_with_template16, new_chroma_template, new_luma_template,
        scaling::{build_lut, Point},
        Params as FgParams,
    };

    let g = &frame.film_grain;
    if !g.apply_grain || g.grain_seed == 0 {
        return;
    }
    let ar_lag = g.ar_coeff_lag as usize;
    let ar_shift = g.ar_coeff_shift_minus6 as u32 + 6;
    let scaling_shift = g.grain_scale_shift + 8;

    // Luma.
    if g.num_y_points > 0 {
        let points: Vec<Point> = (0..g.num_y_points as usize)
            .map(|i| Point {
                value: g.point_y_value[i],
                scale: g.point_y_scaling[i],
            })
            .collect();
        let lut = build_lut(&points);
        let y_coeffs_count = 2 * ar_lag * (ar_lag + 1);
        let y_coeffs = &g.ar_coeffs_y[..y_coeffs_count.min(g.ar_coeffs_y.len())];
        let tpl = new_luma_template(g.grain_seed, ar_lag, y_coeffs, ar_shift);
        let p = FgParams {
            grain_seed: g.grain_seed,
            scaling_y: lut,
            scaling_shift,
            clip_to_restricted_range: g.clip_to_restricted_range,
            overlap_flag: g.overlap_flag,
            ..FgParams::default()
        };
        if fs.bit_depth == 8 {
            let w = fs.width as usize;
            let h = fs.height as usize;
            apply_with_template(&mut fs.y_plane, w, h, w, &lut, &tpl, &p);
        } else {
            let w = fs.width as usize;
            let h = fs.height as usize;
            apply_with_template16(&mut fs.y_plane16, w, h, w, &lut, &tpl, &p, fs.bit_depth);
        }
    }

    if fs.monochrome {
        return;
    }

    let apply_chroma = |scaling_values: &[u8],
                        scaling_scales: &[u8],
                        num_points: u8,
                        ar_coeffs: &[i8],
                        plane8: &mut Vec<u8>,
                        plane16: &mut Vec<u16>,
                        w: usize,
                        h: usize,
                        bd: u32| {
        if num_points == 0 {
            return;
        }
        let points: Vec<Point> = (0..num_points as usize)
            .map(|i| Point {
                value: scaling_values[i],
                scale: scaling_scales[i],
            })
            .collect();
        let lut = build_lut(&points);
        let n_coeffs = (2 * ar_lag * (ar_lag + 1) + 1).min(ar_coeffs.len());
        let c_coeffs = &ar_coeffs[..n_coeffs];
        let seed = g.grain_seed ^ 0xA5A5;
        let tpl = new_chroma_template(seed, ar_lag, c_coeffs, ar_shift);
        let p = FgParams {
            grain_seed: seed,
            scaling_shift,
            clip_to_restricted_range: g.clip_to_restricted_range,
            overlap_flag: g.overlap_flag,
            ..FgParams::default()
        };
        if bd == 8 {
            apply_with_template(plane8, w, h, w, &lut, &tpl, &p);
        } else {
            apply_with_template16(plane16, w, h, w, &lut, &tpl, &p, bd);
        }
    };

    let cw = fs.uv_width as usize;
    let ch = fs.uv_height as usize;
    let bd = fs.bit_depth;
    apply_chroma(
        &g.point_cb_value,
        &g.point_cb_scaling,
        g.num_cb_points,
        &g.ar_coeffs_cb,
        &mut fs.u_plane,
        &mut fs.u_plane16,
        cw,
        ch,
        bd,
    );
    apply_chroma(
        &g.point_cr_value,
        &g.point_cr_scaling,
        g.num_cr_points,
        &g.ar_coeffs_cr,
        &mut fs.v_plane,
        &mut fs.v_plane16,
        cw,
        ch,
        bd,
    );
}

/// Per-tile decoder state. Owns a mutable copy of every CDF used by
/// Phase 2's partition / mode / segment / CFL / skip readers. The
/// coefficient decoder (§5.11.39) is **not** instantiated here — the
/// tile decoder bails with `Error::Unsupported` at the first non-skip
/// leaf. Phase 3 will extend this struct with the coefficient bank.
pub struct TileDecoder<'a> {
    /// Per-frame sequence header (needed for `num_planes`,
    /// `subsampling_x/y`, `use_128x128_superblock`).
    pub seq: &'a SequenceHeader,
    /// Per-frame uncompressed header.
    pub frame: &'a FrameHeader,
    /// Range-coder state.
    pub symbol: SymbolDecoder<'a>,
    /// Superblock side in luma samples — 64 or 128.
    pub sb_size: u32,

    // ---- CDF bank — mutable, per-tile. Copied from the default
    // tables at construction; `decode_symbol` adapts them in place
    // when `allow_update` was set at construction of the symbol
    // decoder (== `!frame.disable_cdf_update`).
    pub partition_cdf: Vec<Vec<u16>>,
    pub kf_y_mode_cdf: Vec<Vec<Vec<u16>>>,
    pub uv_mode_cdf: Vec<Vec<Vec<u16>>>,
    pub angle_delta_cdf: Vec<Vec<u16>>,
    pub skip_cdf: Vec<Vec<u16>>,
    /// `tx_depth` CDF — §5.11.16 / §9.4.8. Indexed as
    /// `tx_size_cdf[maxTxDepthClass][ctx]` where `maxTxDepthClass ∈
    /// 0..=3` maps to Max_Tx_Depth ∈ {1, 2, 3, 4}. `ctx` is the 3-way
    /// above/left neighbour context.
    pub tx_size_cdf: Vec<Vec<Vec<u16>>>,
    pub cfl_sign_cdf: Vec<u16>,
    pub cfl_alpha_cdf: Vec<Vec<u16>>,
    pub seg_cdf: Vec<Vec<u16>>,
    /// `seg_id_predicted` CDF — §5.11.19 / §9.4. Indexed by the
    /// `AboveSegPredContext + LeftSegPredContext` 3-way ctx. 2-symbol
    /// flag per entry.
    pub seg_id_predicted_cdf: Vec<Vec<u16>>,
    /// `skip_mode` CDF — §5.11.10 / §9.4. Indexed by the
    /// `AboveSkipModes + LeftSkipModes` 3-way ctx. 2-symbol flag per
    /// entry. Only consulted when `frame.skip_mode_present` is true.
    pub skip_mode_cdf: Vec<Vec<u16>>,
    /// Intra-frame `tx_type` CDF — set 1 (7 types, `tx_size_ctx ∈ 0..=3`,
    /// intra_mode ∈ 0..=12). Used by blocks whose area ≤ 16×16.
    pub intra_ext_tx_cdf_set1: Vec<Vec<Vec<u16>>>,
    /// Intra-frame `tx_type` CDF — set 2 (5 types, same shape as set 1).
    /// Used by blocks whose area ≤ 32×32. Blocks larger than 32×32 use
    /// implicit DCT_DCT.
    pub intra_ext_tx_cdf_set2: Vec<Vec<Vec<u16>>>,
    /// Inter-frame `tx_type` CDF — set 1 (16 types, indexed by
    /// `Tx_Size_Sqr[txSz] ∈ {TX_4X4=0, TX_8X8=1}`). §5.11.45 / §9.4
    /// `Default_Inter_Tx_Type_Set1_Cdf[2][17]`.
    pub inter_ext_tx_cdf_set1: Vec<Vec<u16>>,
    /// Inter-frame `tx_type` CDF — set 2 (12 types, single context;
    /// only consulted when `Tx_Size_Sqr[txSz] == TX_16X16`). §5.11.45 /
    /// §9.4 `Default_Inter_Tx_Type_Set2_Cdf[13]`.
    pub inter_ext_tx_cdf_set2: Vec<u16>,
    /// Inter-frame `tx_type` CDF — set 3 (2 types, indexed by
    /// `Tx_Size_Sqr[txSz] ∈ {TX_4X4=0..TX_32X32=3}`). §5.11.45 / §9.4
    /// `Default_Inter_Tx_Type_Set3_Cdf[4][3]`.
    pub inter_ext_tx_cdf_set3: Vec<Vec<u16>>,
    /// `delta_q_abs` CDF — §5.11.12. Single 4-symbol CDF, no context.
    pub delta_q_cdf: Vec<u16>,
    /// `delta_lf_abs` CDF — §5.11.13. In multi mode the decoder needs
    /// one CDF per LF frame index, so we keep `FRAME_LF_COUNT` copies.
    pub delta_lf_cdf: Vec<Vec<u16>>,
    /// `use_intrabc` CDF — §5.11.7. 2-symbol.
    pub intrabc_cdf: Vec<u16>,
    /// `use_filter_intra` CDF — §5.11.24, indexed by `MiSize` (22
    /// block sizes). 2-symbol per size.
    pub use_filter_intra_cdf: Vec<Vec<u16>>,
    /// `filter_intra_mode` CDF — §5.11.24, 5-symbol.
    pub filter_intra_mode_cdf: Vec<u16>,
    /// `has_palette_y` CDF — §5.11.46. Indexed as
    /// `[bsizeCtx = Mi_Width_Log2 + Mi_Height_Log2 - 2 ∈ 0..=6][ctx ∈
    /// 0..=2]`, 2-symbol each.
    pub palette_y_mode_cdf: Vec<Vec<Vec<u16>>>,
    /// `has_palette_uv` CDF — §5.11.46. Indexed as
    /// `[(PaletteSizeY > 0) ? 1 : 0]`, 2-symbol each.
    pub palette_uv_mode_cdf: Vec<Vec<u16>>,
    /// `palette_size_y_minus_2` CDF — §5.11.46 / §9.4. Indexed as
    /// `[bsizeCtx ∈ 0..=6]`. 7-symbol (palette sizes 2..=8 → values
    /// 0..=6).
    pub palette_y_size_cdf: Vec<Vec<u16>>,
    /// `palette_size_uv_minus_2` CDF — §5.11.46 / §9.4. Same shape
    /// as the Y variant.
    pub palette_uv_size_cdf: Vec<Vec<u16>>,
    /// `palette_color_idx_y` CDFs — one bank per palette size
    /// (2..=8 → indices 0..=6), each `[ctx ∈ 0..=4]`-indexed. The
    /// inner CDF has `palette_size`-many real symbols.
    pub palette_y_color_cdfs: [Vec<Vec<u16>>; 7],
    /// `palette_color_idx_uv` CDFs — chroma counterpart.
    pub palette_uv_color_cdfs: [Vec<Vec<u16>>; 7],
    /// `txfm_split` CDF — §5.11.17 var-tx partition flag.
    /// `TXFM_PARTITION_CONTEXTS = 21`; 2-symbol each.
    pub txfm_split_cdf: Vec<Vec<u16>>,
    /// Coefficient-CDF bank — all of §5.11.39's CDFs in one place.
    pub coeff_bank: CoeffCdfBank,
    /// §5.11.40-.44 Loop-Restoration per-unit CDFs.
    pub switchable_restore_cdf: Vec<u16>,
    pub wiener_restore_cdf: Vec<u16>,
    pub sgrproj_restore_cdf: Vec<u16>,
    /// Per-plane Wiener / SGR reference values (§5.11.42 / §5.11.44).
    pub lr_ref: [LrRef; 3],
    /// §5.11.19 / §9.4 — `AboveSegPredContext[MiCol]`. One entry per
    /// MI column, stamped by `inter_segment_id` whenever a block reads
    /// or implicitly fixes its `seg_id_predicted` flag (per the
    /// `for(i)` loops in §5.11.19). Cleared at tile entry per the
    /// general "Above context arrays start at 0" rule (§5.11.1 /
    /// §5.11.49). Used to form the 3-way context for the next block's
    /// `seg_id_predicted` symbol.
    pub above_seg_pred: Vec<u8>,
    /// §5.11.19 / §9.4 — `LeftSegPredContext[MiRow]`. One entry per
    /// MI row, stamped by `inter_segment_id`. Cleared at tile entry.
    pub left_seg_pred: Vec<u8>,
    /// Inter-frame syntax reader. `None` for key / intra-only frames.
    pub inter: Option<InterDecoder>,
    /// Reference frame carried from the previous decoded frame —
    /// supplies the LAST ref for single-reference translational inter
    /// blocks. `None` on key frames.
    pub prev_frame: Option<Arc<FrameState>>,
    /// §7.20 reference picture buffer — every slot's reconstructed
    /// planes (when present) plus per-slot OrderHint. Round 14 wires
    /// this so SKIP_MODE compound MC (§7.11.3.9) can dispatch to two
    /// independent references named by `SkipModeFrame[0..=1]`.
    pub dpb: Dpb,
    /// Spec §5.11.5 `ReadDeltas`. Set to `delta_q_present` at the start
    /// of each superblock (§5.11.4 top-of-SB hook) and cleared by the
    /// first block's `intra_frame_mode_info()` / `inter_frame_mode_info()`
    /// call once it has consumed the delta symbols. Held on the tile
    /// decoder so individual leaf calls can inspect / clear it.
    pub read_deltas: bool,
    /// Spec §5.11.12 `CurrentQIndex` — per-tile running quantizer index
    /// that rides on top of `base_q_idx`. Initialised to `base_q_idx` at
    /// tile entry (§5.11.1 `decode_tiles`) and updated by every
    /// `read_delta_qindex()` call to
    /// `Clip3(1, 255, CurrentQIndex + (reducedDeltaQIndex << delta_q_res))`.
    /// Consumed by the luma/chroma dequantizers so per-SB Q deltas land
    /// in reconstruction.
    pub current_q_index: i32,
    /// Spec §5.11.13 `DeltaLF[i]` — per-tile running loop-filter level
    /// deltas, one entry per LF frame-index (`FRAME_LF_COUNT == 4`).
    /// Zeroed at tile entry per §5.11.2 `decode_tile`, updated by
    /// `read_delta_lf()` (§5.11.13). Tracked here so a future loop-filter
    /// pass can consume the deltas; `apply_deblocking` currently uses
    /// frame-level levels only.
    pub delta_lf: [i32; FRAME_LF_COUNT],

    /// §5.11.39 / §6.10.6 `AboveLevelContext[plane][x4]` — per-MI-cell
    /// `culLevel` accumulator per plane, polled by `txb_skip_ctx` and
    /// `dc_sign_ctx`. Indexed `[plane][x4]` where `x4` is in 4×4-cell
    /// units on the plane (chroma is subsampled).
    pub above_level_ctx: [Vec<u8>; 3],
    /// §5.11.39 / §6.10.6 `LeftLevelContext[plane][y4]`.
    pub left_level_ctx: [Vec<u8>; 3],
    /// §5.11.39 / §6.10.6 `AboveDcContext[plane][x4]` — 2-bit DC sign
    /// category (0=zero/absent, 1=positive, 2=negative).
    pub above_dc_ctx: [Vec<u8>; 3],
    /// §5.11.39 / §6.10.6 `LeftDcContext[plane][y4]`.
    pub left_dc_ctx: [Vec<u8>; 3],
}

impl<'a> TileDecoder<'a> {
    /// Initialise a tile decoder for `tile_data`. `tile_data` must be
    /// exactly the per-tile payload extracted via
    /// [`crate::tile_group::split_tile_payloads`] — this constructor
    /// will call `SymbolDecoder::new` on it.
    pub fn new(
        seq: &'a SequenceHeader,
        frame: &'a FrameHeader,
        tile_data: &'a [u8],
        prev_frame: Option<Arc<FrameState>>,
        dpb: Dpb,
    ) -> Result<Self> {
        let sz = tile_data.len();
        let allow_update = !frame.disable_cdf_update;
        let symbol = SymbolDecoder::new(tile_data, sz, allow_update)?;
        let sb_size = if seq.use_128x128_superblock { 128 } else { 64 };
        let q_ctx = q_index_to_ctx(frame.quant.base_q_idx as i32);
        let coeff_bank = CoeffCdfBank::new(q_ctx);
        let inter = if matches!(frame.frame_type, FrameType::Inter | FrameType::Switch) {
            Some(InterDecoder::new(frame.allow_high_precision_mv))
        } else {
            None
        };
        // §5.11.19 / §11.5 — `Above*Context[]` and `Left*Context[]`
        // arrays are zero-initialised for each tile (the spec phrasing
        // is "set equal to 0"). The seg-pred arrays span the full MI
        // grid in each direction so they index by absolute MiCol /
        // MiRow without per-tile origin bookkeeping.
        let mi_cols = (frame.frame_width + 3) >> 2;
        let mi_rows = (frame.frame_height + 3) >> 2;
        let above_seg_pred = vec![0u8; mi_cols.max(1) as usize];
        let left_seg_pred = vec![0u8; mi_rows.max(1) as usize];
        // §5.11.39 / §6.10.6: per-plane Above/Left context arrays at
        // 4×4-cell granularity. Plane-0 spans the full MI grid; chroma
        // planes shrink by `subsampling_x/y`.
        let sub_x = if seq.color_config.subsampling_x { 1 } else { 0 };
        let sub_y = if seq.color_config.subsampling_y { 1 } else { 0 };
        let chroma_x4 = ((mi_cols + (1 << sub_x) - 1) >> sub_x).max(1) as usize;
        let chroma_y4 = ((mi_rows + (1 << sub_y) - 1) >> sub_y).max(1) as usize;
        let above_level_ctx = [
            vec![0u8; mi_cols.max(1) as usize],
            vec![0u8; chroma_x4],
            vec![0u8; chroma_x4],
        ];
        let left_level_ctx = [
            vec![0u8; mi_rows.max(1) as usize],
            vec![0u8; chroma_y4],
            vec![0u8; chroma_y4],
        ];
        let above_dc_ctx = [
            vec![0u8; mi_cols.max(1) as usize],
            vec![0u8; chroma_x4],
            vec![0u8; chroma_x4],
        ];
        let left_dc_ctx = [
            vec![0u8; mi_rows.max(1) as usize],
            vec![0u8; chroma_y4],
            vec![0u8; chroma_y4],
        ];
        let mut td = Self {
            seq,
            frame,
            symbol,
            sb_size,
            partition_cdf: Vec::new(),
            kf_y_mode_cdf: Vec::new(),
            uv_mode_cdf: Vec::new(),
            angle_delta_cdf: Vec::new(),
            skip_cdf: Vec::new(),
            tx_size_cdf: Vec::new(),
            cfl_sign_cdf: Vec::new(),
            cfl_alpha_cdf: Vec::new(),
            seg_cdf: Vec::new(),
            seg_id_predicted_cdf: Vec::new(),
            skip_mode_cdf: Vec::new(),
            intra_ext_tx_cdf_set1: Vec::new(),
            intra_ext_tx_cdf_set2: Vec::new(),
            inter_ext_tx_cdf_set1: Vec::new(),
            inter_ext_tx_cdf_set2: Vec::new(),
            inter_ext_tx_cdf_set3: Vec::new(),
            delta_q_cdf: Vec::new(),
            delta_lf_cdf: Vec::new(),
            intrabc_cdf: Vec::new(),
            use_filter_intra_cdf: Vec::new(),
            filter_intra_mode_cdf: Vec::new(),
            palette_y_mode_cdf: Vec::new(),
            palette_uv_mode_cdf: Vec::new(),
            palette_y_size_cdf: Vec::new(),
            palette_uv_size_cdf: Vec::new(),
            palette_y_color_cdfs: Default::default(),
            palette_uv_color_cdfs: Default::default(),
            txfm_split_cdf: Vec::new(),
            coeff_bank,
            switchable_restore_cdf: default_switchable_cdf(),
            wiener_restore_cdf: default_wiener_cdf(),
            sgrproj_restore_cdf: default_sgrproj_cdf(),
            lr_ref: [LrRef::default(); 3],
            above_seg_pred,
            left_seg_pred,
            inter,
            prev_frame,
            dpb,
            read_deltas: false,
            // §5.11.1: `CurrentQIndex = base_q_idx` at tile entry.
            current_q_index: frame.quant.base_q_idx as i32,
            // §5.11.2: `for (i=0; i < FRAME_LF_COUNT; i++) DeltaLF[i] = 0`.
            delta_lf: [0; FRAME_LF_COUNT],
            above_level_ctx,
            left_level_ctx,
            above_dc_ctx,
            left_dc_ctx,
        };
        td.init_cdfs();
        Ok(td)
    }

    // (Helper `level_ctx_for_plane` is a free fn — see [`level_ctx_for_plane`].)

    /// Decode the next restoration unit's syntax for `plane` (spec
    /// §5.11.40 `lr_unit_info()`). Mutates the tile-local LR
    /// CDFs + per-plane reference values in place; returns the
    /// decoded `UnitParams`. Callers should short-circuit when the
    /// plane's `FrameRestorationType == RESTORE_NONE`.
    pub fn decode_lr_unit(&mut self, plane: usize) -> Result<LrUnitParams> {
        let rt = self.frame.lr.frame_restoration_type[plane];
        if rt == RESTORATION_NONE {
            return Ok(LrUnitParams::default());
        }
        decode_lr_unit_impl(
            &mut self.symbol,
            rt,
            plane,
            &mut self.lr_ref[plane],
            &mut self.switchable_restore_cdf,
            &mut self.wiener_restore_cdf,
            &mut self.sgrproj_restore_cdf,
        )
    }

    /// Initialise the per-tile CDFs from [`crate::cdfs`] defaults.
    /// Each entry is copied into an owned `Vec<u16>` so the range
    /// coder can adapt it in place.
    fn init_cdfs(&mut self) {
        self.partition_cdf = cdfs::DEFAULT_PARTITION_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.kf_y_mode_cdf = cdfs::DEFAULT_KF_Y_MODE_CDF
            .iter()
            .map(|row| row.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
            .collect();
        self.uv_mode_cdf = cdfs::DEFAULT_UV_MODE_CDF
            .iter()
            .map(|row| row.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
            .collect();
        self.angle_delta_cdf = cdfs::DEFAULT_ANGLE_DELTA_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.skip_cdf = cdfs::DEFAULT_SKIP_CDF.iter().map(|c| c.to_vec()).collect();
        self.tx_size_cdf = cdfs::DEFAULT_TX_SIZE_CDF
            .iter()
            .map(|row| row.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
            .collect();
        self.cfl_sign_cdf = cdfs::DEFAULT_CFL_SIGN_CDF.to_vec();
        self.cfl_alpha_cdf = cdfs::DEFAULT_CFL_ALPHA_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.seg_cdf = cdfs::DEFAULT_SPATIAL_PRED_SEG_TREE_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        // §5.11.19 `seg_id_predicted` — 3-context 2-symbol CDF.
        self.seg_id_predicted_cdf = cdfs::DEFAULT_SEGMENT_ID_PREDICTED_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        // §5.11.10 `skip_mode` — 3-context 2-symbol CDF.
        self.skip_mode_cdf = cdfs::DEFAULT_SKIP_MODE_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.intra_ext_tx_cdf_set1 = cdfs::DEFAULT_INTRA_EXT_TX_CDF_SET1
            .iter()
            .map(|row| row.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
            .collect();
        self.intra_ext_tx_cdf_set2 = cdfs::DEFAULT_INTRA_EXT_TX_CDF_SET2
            .iter()
            .map(|row| row.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
            .collect();
        // §5.11.45 / §9.4 — three default inter ext-tx CDFs. Set 1 is
        // 2 contexts × 16-symbol; set 2 is single 12-symbol; set 3 is
        // 4 contexts × 2-symbol. Stored owned so the range coder can
        // adapt them in place.
        self.inter_ext_tx_cdf_set1 = cdfs::DEFAULT_INTER_EXT_TX_CDF_SET1
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.inter_ext_tx_cdf_set2 = cdfs::DEFAULT_INTER_EXT_TX_CDF_SET2.to_vec();
        self.inter_ext_tx_cdf_set3 = cdfs::DEFAULT_INTER_EXT_TX_CDF_SET3
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.delta_q_cdf = cdfs::DEFAULT_DELTA_Q_CDF.to_vec();
        // Spec §5.11.13 / §9.4.1: `FRAME_LF_COUNT == 4` copies in the
        // multi path. We allocate them unconditionally; the decoder
        // skips reads when `delta_lf_multi` is cleared.
        self.delta_lf_cdf = (0..FRAME_LF_COUNT)
            .map(|_| cdfs::DEFAULT_DELTA_LF_CDF.to_vec())
            .collect();
        self.intrabc_cdf = cdfs::DEFAULT_INTRABC_CDF.to_vec();
        self.use_filter_intra_cdf = cdfs::DEFAULT_USE_FILTER_INTRA_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.filter_intra_mode_cdf = cdfs::DEFAULT_FILTER_INTRA_MODE_CDF.to_vec();
        // §5.11.46 palette CDFs — 2-symbol flags. Shape: `[7][3][…]`
        // for Y (bsizeCtx × neighbour-palette ctx), `[2][…]` for UV.
        self.palette_y_mode_cdf = cdfs::DEFAULT_PALETTE_Y_MODE_CDF
            .iter()
            .map(|row| row.iter().map(|c| c.to_vec()).collect::<Vec<_>>())
            .collect();
        self.palette_uv_mode_cdf = cdfs::DEFAULT_PALETTE_UV_MODE_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        // §5.11.46 palette size CDFs — 7-symbol (sizes 2..=8 ⇒
        // signal value 0..=6). Indexed by bsizeCtx ∈ 0..=6.
        self.palette_y_size_cdf = cdfs::DEFAULT_PALETTE_Y_SIZE_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_size_cdf = cdfs::DEFAULT_PALETTE_UV_SIZE_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        // §5.11.49 palette colour-index CDFs — one bank per palette
        // size, each `[ctx ∈ 0..=4]`. Slot `[k]` holds the CDF for
        // PaletteSize == k+2.
        self.palette_y_color_cdfs[0] = cdfs::DEFAULT_PALETTE_Y_COLOR_SIZE_2_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_y_color_cdfs[1] = cdfs::DEFAULT_PALETTE_Y_COLOR_SIZE_3_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_y_color_cdfs[2] = cdfs::DEFAULT_PALETTE_Y_COLOR_SIZE_4_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_y_color_cdfs[3] = cdfs::DEFAULT_PALETTE_Y_COLOR_SIZE_5_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_y_color_cdfs[4] = cdfs::DEFAULT_PALETTE_Y_COLOR_SIZE_6_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_y_color_cdfs[5] = cdfs::DEFAULT_PALETTE_Y_COLOR_SIZE_7_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_y_color_cdfs[6] = cdfs::DEFAULT_PALETTE_Y_COLOR_SIZE_8_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_color_cdfs[0] = cdfs::DEFAULT_PALETTE_UV_COLOR_SIZE_2_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_color_cdfs[1] = cdfs::DEFAULT_PALETTE_UV_COLOR_SIZE_3_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_color_cdfs[2] = cdfs::DEFAULT_PALETTE_UV_COLOR_SIZE_4_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_color_cdfs[3] = cdfs::DEFAULT_PALETTE_UV_COLOR_SIZE_5_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_color_cdfs[4] = cdfs::DEFAULT_PALETTE_UV_COLOR_SIZE_6_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_color_cdfs[5] = cdfs::DEFAULT_PALETTE_UV_COLOR_SIZE_7_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.palette_uv_color_cdfs[6] = cdfs::DEFAULT_PALETTE_UV_COLOR_SIZE_8_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.txfm_split_cdf = cdfs::DEFAULT_TXFM_SPLIT_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
    }

    /// Walk the tile's superblock grid and decode every MI unit's
    /// mode info. Returns `Ok(())` only if the tile decodes end to
    /// end with every leaf marked skip; otherwise returns
    /// `Error::Unsupported("av1 coefficient decode pending (§5.11.39)")`.
    pub fn decode(&mut self, fs: &mut FrameState, tp: &TilePayload) -> Result<()> {
        let tile_info = self
            .frame
            .tile_info
            .as_ref()
            .ok_or_else(|| Error::invalid("av1 decode_tile: tile_info missing"))?;
        // Tile extents in MI units (§5.11.1).
        let mi_col_start = tile_info.mi_col_starts[tp.tile_col as usize];
        let mi_col_end = tile_info.mi_col_starts[(tp.tile_col + 1) as usize];
        let mi_row_start = tile_info.mi_row_starts[tp.tile_row as usize];
        let mi_row_end = tile_info.mi_row_starts[(tp.tile_row + 1) as usize];
        // MI units are 4-sample blocks; superblock step in MI is 16
        // (64-sample SB) or 32 (128-sample SB).
        let sb_mi = self.sb_size >> 2;
        let mut mi_row = mi_row_start;
        while mi_row < mi_row_end {
            let mut mi_col = mi_col_start;
            while mi_col < mi_col_end {
                let sb_x = mi_col << 2;
                let sb_y = mi_row << 2;
                superblock::decode_superblock(self, fs, sb_x, sb_y)?;
                mi_col += sb_mi;
            }
            mi_row += sb_mi;
        }
        Ok(())
    }

    // ----- Symbol-level primitives. -----
    //
    // Each wraps one of the tile decoder's CDFs; out-of-range contexts
    // are clamped silently so invalid inputs don't panic.

    /// Decode a partition symbol for a square block of BSL class
    /// `bsl_ctx` (0..=4) and left/above context `ctx` (0..=3).
    pub fn decode_partition(&mut self, bsl_ctx: u32, ctx: u32) -> Result<u32> {
        let cdf_idx = (bsl_ctx * 4 + ctx) as usize;
        if cdf_idx >= self.partition_cdf.len() {
            return Err(Error::invalid(format!(
                "av1 decode_partition: ctx index {cdf_idx} out of range (§5.11.4)"
            )));
        }
        self.symbol.decode_symbol(&mut self.partition_cdf[cdf_idx])
    }

    /// Decode the Y-plane intra mode for a KEY_FRAME block, given
    /// 5-bucket above/left contexts.
    pub fn decode_intra_y_mode(&mut self, above_ctx: u32, left_ctx: u32) -> Result<IntraMode> {
        let a = (above_ctx as usize).min(4);
        let l = (left_ctx as usize).min(4);
        let raw = self.symbol.decode_symbol(&mut self.kf_y_mode_cdf[a][l])?;
        if (raw as usize) >= INTRA_MODES {
            return Err(Error::invalid(format!(
                "av1 kf_y_mode: symbol {raw} out of range (§5.11.18)"
            )));
        }
        IntraMode::from_u32(raw).ok_or_else(|| {
            Error::invalid(format!(
                "av1 kf_y_mode: invalid intra mode {raw} (§5.11.18)"
            ))
        })
    }

    /// Decode the UV-plane intra mode. `cfl_allowed` selects between
    /// the 13-symbol (CFL not allowed) and 14-symbol (CFL allowed)
    /// CDF sets.
    pub fn decode_uv_mode(&mut self, y_mode: IntraMode, cfl_allowed: bool) -> Result<IntraMode> {
        let cfl_idx = if cfl_allowed { 1 } else { 0 };
        let y_idx = y_mode as usize;
        let raw = self
            .symbol
            .decode_symbol(&mut self.uv_mode_cdf[cfl_idx][y_idx])?;
        if (raw as usize) >= UV_MODES {
            return Err(Error::invalid(format!(
                "av1 uv_mode: symbol {raw} out of range (§5.11.18)"
            )));
        }
        IntraMode::from_u32(raw)
            .ok_or_else(|| Error::invalid(format!("av1 uv_mode: invalid mode {raw} (§5.11.18)")))
    }

    /// Decode `angle_delta` for a directional mode. `dir_idx` is the
    /// CDF index per spec §9.4: `mode - V_PRED` (so the 6 directional
    /// modes hit slots 2..=7 of the 8-entry `angle_delta_cdf` table).
    /// Returns the signed delta in `-3..=3`.
    pub fn decode_angle_delta(&mut self, dir_idx: u32) -> Result<i32> {
        if dir_idx >= 8 {
            return Err(Error::invalid(format!(
                "av1 angle_delta: dir_idx {dir_idx} out of range (§5.11.18)"
            )));
        }
        let raw = self
            .symbol
            .decode_symbol(&mut self.angle_delta_cdf[dir_idx as usize])?;
        Ok(raw as i32 - 3)
    }

    /// Decode the `skip` flag given a context `0..=2`.
    pub fn decode_skip(&mut self, ctx: u32) -> Result<bool> {
        let ctx = (ctx as usize).min(2);
        let raw = self.symbol.decode_symbol(&mut self.skip_cdf[ctx])?;
        Ok(raw != 0)
    }

    /// Decode `tx_depth` per §5.11.16 / §9.4.8.
    ///
    /// `max_tx_depth` is the spec's `Max_Tx_Depth[MiSize]` (must be
    /// ≥ 1 — the caller short-circuits `Block4x4`). `ctx` is the 3-way
    /// neighbour context from `(aboveW >= maxTxWidth) + (leftH >=
    /// maxTxHeight)` (clamped to `0..=2`). Returns the decoded
    /// `tx_depth ∈ 0..=2`.
    pub fn decode_tx_depth(&mut self, max_tx_depth: u32, ctx: u32) -> Result<u32> {
        // `DEFAULT_TX_SIZE_CDF` shape is `[4][3][…]`:
        //   class 0 → Max_Tx_Depth = 1 (8×8 CDF, 2 symbols)
        //   class 1 → Max_Tx_Depth = 2 (16×16 CDF, 3 symbols)
        //   class 2 → Max_Tx_Depth = 3 (32×32 CDF, 3 symbols)
        //   class 3 → Max_Tx_Depth = 4 (64×64 CDF, 3 symbols)
        let class = max_tx_depth.saturating_sub(1).min(3) as usize;
        let ctx = (ctx as usize).min(2);
        self.symbol.decode_symbol(&mut self.tx_size_cdf[class][ctx])
    }

    /// Decode the segment_id for a block (spatial prediction,
    /// §5.11.9). `ctx` is the 3-way max-based neighbor context.
    pub fn decode_segment_id(&mut self, ctx: u32) -> Result<u8> {
        let ctx = (ctx as usize).min(2);
        let raw = self.symbol.decode_symbol(&mut self.seg_cdf[ctx])?;
        Ok(raw as u8)
    }

    /// §5.11.19 `seg_id_predicted` — 2-symbol flag indicating whether
    /// the temporally-predicted segment_id (spatial-min over the
    /// previous frame's `PrevSegmentIds[][]`) should be reused for
    /// this block. `ctx` is `AboveSegPredContext + LeftSegPredContext`,
    /// clamped to `0..=2`. Only consulted when `segmentation_enabled
    /// && segmentation_update_map && segmentation_temporal_update`.
    pub fn decode_seg_id_predicted(&mut self, ctx: u32) -> Result<bool> {
        let ctx = (ctx as usize).min(2);
        let raw = self
            .symbol
            .decode_symbol(&mut self.seg_id_predicted_cdf[ctx])?;
        Ok(raw != 0)
    }

    /// §5.11.19 / §9.4 — current `seg_id_predicted` ctx for the block
    /// rooted at `(mi_col, mi_row)`. Sums the most-recent
    /// `AboveSegPredContext[mi_col]` and `LeftSegPredContext[mi_row]`
    /// stamps; both arrays default to 0 at tile entry so the first
    /// block sees ctx == 0.
    pub fn seg_id_predicted_ctx(&self, mi_col: u32, mi_row: u32) -> u32 {
        let above = self
            .above_seg_pred
            .get(mi_col as usize)
            .copied()
            .unwrap_or(0) as u32;
        let left = self
            .left_seg_pred
            .get(mi_row as usize)
            .copied()
            .unwrap_or(0) as u32;
        above + left
    }

    /// §5.11.19 — stamp the `seg_id_predicted` value across a block's
    /// MI extent so subsequent blocks see the updated context. `bw4 /
    /// bh4` are the block dimensions in 4×4 MI units.
    pub fn stamp_seg_id_predicted(
        &mut self,
        mi_col: u32,
        mi_row: u32,
        bw4: u32,
        bh4: u32,
        v: bool,
    ) {
        let v = v as u8;
        let cols = self.above_seg_pred.len() as u32;
        let rows = self.left_seg_pred.len() as u32;
        for i in 0..bw4 {
            let c = mi_col + i;
            if c < cols {
                self.above_seg_pred[c as usize] = v;
            }
        }
        for i in 0..bh4 {
            let r = mi_row + i;
            if r < rows {
                self.left_seg_pred[r as usize] = v;
            }
        }
    }

    /// §5.11.10 `read_skip_mode` — 2-symbol flag enabling the AV1
    /// SKIP_MODE compound coding. `ctx` is the §9.4 `AboveSkipModes +
    /// LeftSkipModes` 3-way ctx (clamped). Caller must already have
    /// short-circuited on the spec gating predicates
    /// (`!skip_mode_present`, `seg_feature_active(SEG_LVL_SKIP / REF /
    /// GLOBALMV)`, `Block_Width < 8`, `Block_Height < 8`).
    pub fn decode_skip_mode(&mut self, ctx: u32) -> Result<bool> {
        let ctx = (ctx as usize).min(2);
        let raw = self.symbol.decode_symbol(&mut self.skip_mode_cdf[ctx])?;
        Ok(raw != 0)
    }

    /// Decode the CFL joint-sign symbol (0..=7). Spec §6.10.14.
    pub fn decode_cfl_sign(&mut self) -> Result<u32> {
        self.symbol.decode_symbol(&mut self.cfl_sign_cdf)
    }

    /// Decode a single plane's CFL alpha magnitude (0..=15); caller
    /// adds 1 to get the actual magnitude (1..=16 in Q3). `ctx` is
    /// the 6-way context derived from the joint sign.
    pub fn decode_cfl_alpha(&mut self, ctx: u32) -> Result<u32> {
        let ctx = (ctx as usize).min(5);
        self.symbol.decode_symbol(&mut self.cfl_alpha_cdf[ctx])
    }

    /// Decode the intra-frame `tx_type` for a transform of dimensions
    /// `w × h` under Y intra mode `y_mode` (§6.10.15).
    ///
    /// The spec partitions TX sizes into three sets:
    /// - `ExtTxSet = 0` (area > 32×32): implicit `DCT_DCT`, no symbol.
    /// - `ExtTxSet = 1` (area ≤ 16×16): 7-type CDF (CDF set 1).
    /// - `ExtTxSet = 2` (area ≤ 32×32): 5-type CDF (CDF set 2).
    pub fn decode_intra_tx_type(
        &mut self,
        w: usize,
        h: usize,
        y_mode: IntraMode,
    ) -> Result<TxType> {
        let tx_set = ext_tx_set_for_intra(w, h);
        if tx_set == 0 {
            return Ok(TxType::DctDct);
        }
        let size_ctx = ext_tx_size_ctx(w, h);
        let mode_idx = (y_mode as usize).min(12);
        let raw = match tx_set {
            1 => self
                .symbol
                .decode_symbol(&mut self.intra_ext_tx_cdf_set1[size_ctx][mode_idx])?,
            2 => self
                .symbol
                .decode_symbol(&mut self.intra_ext_tx_cdf_set2[size_ctx][mode_idx])?,
            _ => 0,
        };
        Ok(intra_tx_type_for(tx_set, raw))
    }

    /// Decode the inter-frame `tx_type` for a TU of dimensions `w × h`
    /// (§5.11.45 / §5.11.47 / §6.10.15). Mirrors the intra path:
    ///
    /// - `set = 0` (`TX_SET_DCTONLY`, `txSzSqrUp > TX_32X32`): no
    ///   symbol; implicit `DCT_DCT`.
    /// - `set = 1` (`TX_SET_INTER_1`, `txSzSqr ≤ TX_8X8`): 16-symbol
    ///   CDF indexed by `Tx_Size_Sqr[txSz] ∈ {0, 1}`.
    /// - `set = 2` (`TX_SET_INTER_2`, `txSzSqr == TX_16X16`):
    ///   single-context 12-symbol CDF.
    /// - `set = 3` (`TX_SET_INTER_3`, `reduced_tx_set` or
    ///   `txSzSqrUp == TX_32X32`): 2-symbol CDF indexed by
    ///   `Tx_Size_Sqr[txSz] ∈ {0..=3}`.
    ///
    /// Caller passes `reduced_tx_set` from the frame header. The
    /// returned [`TxType`] is the spec's `Tx_Type_Inter_Inv_Set*[raw]`
    /// lookup; out-of-range raw symbols safely fall back to `DCT_DCT`.
    pub fn decode_inter_tx_type(
        &mut self,
        w: usize,
        h: usize,
        reduced_tx_set: bool,
    ) -> Result<TxType> {
        let set = ext_tx_set_for_inter(w as u32, h as u32, reduced_tx_set);
        if set == 0 {
            return Ok(TxType::DctDct);
        }
        // Tx_Size_Sqr index: min(log2W, log2H) - 2 ⇒ TX_4X4=0..TX_32X32=3.
        let size_sqr = tx_size_sqr_index(w, h);
        let raw = match set {
            1 => {
                let ctx = size_sqr.min(self.inter_ext_tx_cdf_set1.len().saturating_sub(1));
                self.symbol
                    .decode_symbol(&mut self.inter_ext_tx_cdf_set1[ctx])?
            }
            2 => self.symbol.decode_symbol(&mut self.inter_ext_tx_cdf_set2)?,
            3 => {
                let ctx = size_sqr.min(self.inter_ext_tx_cdf_set3.len().saturating_sub(1));
                self.symbol
                    .decode_symbol(&mut self.inter_ext_tx_cdf_set3[ctx])?
            }
            _ => 0,
        };
        Ok(inter_tx_type_for(set, raw))
    }

    /// §5.11.12 `read_delta_qindex()` — reads the `delta_q_abs`
    /// symbol, then optionally a literal-bits tail + sign bit. Returns
    /// the decoded `reducedDeltaQIndex` (before multiplying by
    /// `1 << delta_q_res`); callers apply the shift when updating
    /// `CurrentQIndex`.
    ///
    /// Caller MUST gate this on the spec's `ReadDeltas` predicate
    /// (i.e. `delta_q_present` for the frame, and the block is not the
    /// whole-SB skip degenerate case — see §5.11.12).
    pub fn decode_delta_qindex(&mut self) -> Result<i32> {
        let mut delta_q_abs = self.symbol.decode_symbol(&mut self.delta_q_cdf)?;
        if delta_q_abs == DELTA_Q_SMALL {
            let delta_q_rem_bits = self.symbol.read_literal(3) + 1;
            let delta_q_abs_bits = self.symbol.read_literal(delta_q_rem_bits);
            delta_q_abs = delta_q_abs_bits + (1 << delta_q_rem_bits) + 1;
        }
        if delta_q_abs != 0 {
            let sign = self.symbol.read_literal(1);
            let v = delta_q_abs as i32;
            Ok(if sign != 0 { -v } else { v })
        } else {
            Ok(0)
        }
    }

    /// §5.11.13 `read_delta_lf()` — decodes a single `delta_lf_abs`
    /// (+ optional literal tail + sign) for LF frame-index `lf_idx`.
    /// The caller is responsible for honouring `delta_lf_multi` and
    /// the `MiSize == sbSize && skip` short-circuit.
    pub fn decode_delta_lf_abs(&mut self, lf_idx: usize) -> Result<i32> {
        let lf_idx = lf_idx.min(FRAME_LF_COUNT - 1);
        let mut delta_lf_abs = self.symbol.decode_symbol(&mut self.delta_lf_cdf[lf_idx])?;
        if delta_lf_abs == DELTA_LF_SMALL {
            let n = self.symbol.read_literal(3) + 1;
            let bits = self.symbol.read_literal(n);
            delta_lf_abs = bits + (1 << n) + 1;
        }
        if delta_lf_abs != 0 {
            let sign = self.symbol.read_literal(1);
            let v = delta_lf_abs as i32;
            Ok(if sign != 0 { -v } else { v })
        } else {
            Ok(0)
        }
    }

    /// §5.11.7 `use_intrabc` — 2-symbol flag read only when
    /// `allow_intrabc` is true for the frame. Returns `true` when the
    /// block uses intra block-copy.
    pub fn decode_use_intrabc(&mut self) -> Result<bool> {
        let raw = self.symbol.decode_symbol(&mut self.intrabc_cdf)?;
        Ok(raw != 0)
    }

    /// §5.11.24 `use_filter_intra` — 2-symbol flag, indexed by
    /// `MiSize`. Caller must have already checked the eligibility
    /// predicate (enable_filter_intra && YMode==DC_PRED && PaletteSizeY
    /// == 0 && max(bw,bh) <= 32).
    pub fn decode_use_filter_intra(&mut self, bs_idx: usize) -> Result<bool> {
        let bs_idx = bs_idx.min(self.use_filter_intra_cdf.len() - 1);
        let raw = self
            .symbol
            .decode_symbol(&mut self.use_filter_intra_cdf[bs_idx])?;
        Ok(raw != 0)
    }

    /// §5.11.24 `filter_intra_mode` — 5-symbol CDF. Only read when
    /// [`Self::decode_use_filter_intra`] returned true.
    pub fn decode_filter_intra_mode(&mut self) -> Result<u32> {
        self.symbol.decode_symbol(&mut self.filter_intra_mode_cdf)
    }

    /// §5.11.46 `has_palette_y` — 2-symbol flag. `bsize_ctx` is the
    /// spec's `Mi_Width_Log2[MiSize] + Mi_Height_Log2[MiSize] - 2`
    /// value (0..=6 for the eligible BLOCK_8X8..BLOCK_64X64 range).
    /// `neighbor_ctx` is `(AvailU && PaletteSizeAbove > 0) + (AvailL
    /// && PaletteSizeLeft > 0)`, clamped to 0..=2.
    pub fn decode_has_palette_y(&mut self, bsize_ctx: usize, neighbor_ctx: usize) -> Result<bool> {
        let bs = bsize_ctx.min(self.palette_y_mode_cdf.len() - 1);
        let nc = neighbor_ctx.min(self.palette_y_mode_cdf[bs].len() - 1);
        let raw = self
            .symbol
            .decode_symbol(&mut self.palette_y_mode_cdf[bs][nc])?;
        Ok(raw != 0)
    }

    /// §5.11.46 `has_palette_uv` — 2-symbol flag. `ctx` is the spec's
    /// `(PaletteSizeY > 0) ? 1 : 0`. Only called when `HasChroma &&
    /// UVMode == DC_PRED`.
    pub fn decode_has_palette_uv(&mut self, ctx: usize) -> Result<bool> {
        let c = ctx.min(self.palette_uv_mode_cdf.len() - 1);
        let raw = self
            .symbol
            .decode_symbol(&mut self.palette_uv_mode_cdf[c])?;
        Ok(raw != 0)
    }

    /// §5.11.46 `palette_size_y_minus_2` — 7-symbol size value
    /// (encoded palette sizes 2..=8 → raw 0..=6). `bsize_ctx` is the
    /// `Mi_Width_Log2 + Mi_Height_Log2 - 2` value (0..=6).
    pub fn decode_palette_size_y(&mut self, bsize_ctx: usize) -> Result<u32> {
        let bs = bsize_ctx.min(self.palette_y_size_cdf.len() - 1);
        let raw = self
            .symbol
            .decode_symbol(&mut self.palette_y_size_cdf[bs])?;
        Ok(raw)
    }

    /// §5.11.46 `palette_size_uv_minus_2` — chroma counterpart.
    pub fn decode_palette_size_uv(&mut self, bsize_ctx: usize) -> Result<u32> {
        let bs = bsize_ctx.min(self.palette_uv_size_cdf.len() - 1);
        let raw = self
            .symbol
            .decode_symbol(&mut self.palette_uv_size_cdf[bs])?;
        Ok(raw)
    }

    /// §5.11.49 `palette_color_idx_y` / `palette_color_idx_uv` — read
    /// one palette colour index. The CDF is selected by
    /// `(palette_size, ctx)` and the inner alphabet has `palette_size`
    /// symbols. `is_y` flips between the Y and UV colour CDF banks.
    pub fn decode_palette_color_idx(
        &mut self,
        palette_size: u8,
        ctx: usize,
        is_y: bool,
    ) -> Result<u32> {
        if !(2..=8).contains(&palette_size) {
            return Err(Error::invalid(format!(
                "av1 palette_color_idx: palette_size {palette_size} out of range (§5.11.49)"
            )));
        }
        let bank_idx = (palette_size - 2) as usize;
        let bank = if is_y {
            &mut self.palette_y_color_cdfs[bank_idx]
        } else {
            &mut self.palette_uv_color_cdfs[bank_idx]
        };
        let c = ctx.min(bank.len() - 1);
        self.symbol.decode_symbol(&mut bank[c])
    }

    /// §5.11.17 `txfm_split` — 2-symbol flag signalling whether a
    /// var-tx inter transform unit splits further. `ctx` is the
    /// §9.4.8 formula `(txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 -
    /// maxTxSz) * 6 + above + left`, clamped to `0..=20`.
    pub fn decode_txfm_split(&mut self, ctx: usize) -> Result<bool> {
        let c = ctx.min(self.txfm_split_cdf.len() - 1);
        let raw = self.symbol.decode_symbol(&mut self.txfm_split_cdf[c])?;
        Ok(raw != 0)
    }
}

/// Intra-frame ext-tx set per §6.10.15. Returns 0 for implicit
/// `DCT_DCT`, 1 for the 7-type set (area ≤ 16×16), or 2 for the 5-type
/// set (area ≤ 32×32).
fn ext_tx_set_for_intra(w: usize, h: usize) -> u32 {
    let area = w * h;
    if area <= 16 * 16 {
        1
    } else if area <= 32 * 32 {
        2
    } else {
        0
    }
}

/// 4-way size context used to index the intra ext-tx CDFs: TX_4X4=0,
/// TX_8X8=1, TX_16X16=2, TX_32X32=3. Non-square sizes map to the
/// square equivalent by area.
fn ext_tx_size_ctx(w: usize, h: usize) -> usize {
    let area = w * h;
    if area <= 4 * 4 {
        0
    } else if area <= 8 * 8 {
        1
    } else if area <= 16 * 16 {
        2
    } else {
        3
    }
}

/// `Tx_Size_Sqr[txSz]` as a TX_SIZES index (0=TX_4X4 .. 3=TX_32X32).
/// Computed from the TU dimensions per the spec table — `min(log2W,
/// log2H) - 2`. Used to index the inter ext-tx set 1 / set 3 CDFs.
fn tx_size_sqr_index(w: usize, h: usize) -> usize {
    let m = w.min(h);
    match m {
        4 => 0,
        8 => 1,
        16 => 2,
        32 => 3,
        _ => 3, // TX_64X64 / oversize collapses to the largest CDF row.
    }
}

/// Map a raw symbol decoded via [`TileDecoder::decode_intra_tx_type`]
/// into the spec's `TxType`. See spec §6.10.15 Table 13-4. `tx_set = 0`
/// always implies `DCT_DCT`.
fn intra_tx_type_for(tx_set: u32, raw: u32) -> TxType {
    match tx_set {
        1 => match raw {
            0 => TxType::DctDct,
            1 => TxType::AdstDct,
            2 => TxType::DctAdst,
            3 => TxType::AdstAdst,
            4 => TxType::IdtIdt,
            5 => TxType::VDct,
            6 => TxType::HDct,
            _ => TxType::DctDct,
        },
        2 => match raw {
            0 => TxType::DctDct,
            1 => TxType::AdstDct,
            2 => TxType::DctAdst,
            3 => TxType::AdstAdst,
            4 => TxType::IdtIdt,
            _ => TxType::DctDct,
        },
        _ => TxType::DctDct,
    }
}

/// Neighbor context for spatial segment prediction (§5.11.9). Counts
/// the number of distinct non-zero `segment_id` values among the
/// available above/left neighbors, clamped to 0..=2.
pub fn segment_id_ctx(above_id: u8, left_id: u8, have_above: bool, have_left: bool) -> u32 {
    let mut count = 0u32;
    if have_above && above_id != 0 {
        count += 1;
    }
    if have_left && left_id != 0 {
        count += 1;
    }
    count
}

/// Split a CFL joint-sign symbol into `(sign_u, sign_v)` where each
/// element is -1, 0, or +1. Spec §5.11.45 / §6.10.14:
///
///   signU = (cfl_alpha_signs + 1) / 3
///   signV = (cfl_alpha_signs + 1) % 3
///
/// then map { ZERO=0 → 0, NEG=1 → -1, POS=2 → +1 }.
///
/// Round-5 fix: the previous mapping used a hand-rolled table that
/// didn't match the spec arithmetic — joint=3 in particular returned
/// `(0,-1)` instead of the spec's `(-1,-1)`, which silently zeroed
/// CflAlphaU on a sizeable share of CFL-coded blocks.
pub fn cfl_signs(joint: u32) -> (i32, i32) {
    let v = (joint + 1) as i32;
    let su_code = v / 3;
    let sv_code = v % 3;
    let to_sign = |c: i32| -> i32 {
        match c {
            0 => 0,
            1 => -1,
            2 => 1,
            _ => 0,
        }
    };
    (to_sign(su_code), to_sign(sv_code))
}

/// CFL alpha CDF context for a given joint sign + plane. Per AV1
/// spec §9.4 (table at lines 21164-21227 of the spec text):
///
/// - cfl_alpha_u CDF ctx = `(signU - 1) * 3 + signV`, defined when
///   signU != 0 (joint ∈ {2..=7}).
/// - cfl_alpha_v CDF ctx = `(signV - 1) * 3 + signU`, defined when
///   signV != 0 (joint ∈ {0,1,3,4,6,7}).
///
/// where the spec sign codes are { ZERO=0, NEG=1, POS=2 } — i.e.
/// `signU = (joint + 1) / 3`, `signV = (joint + 1) % 3`.
///
/// Returns 0 when the requested plane's sign is zero (the caller
/// will not actually issue a CDF read in that case; we keep the
/// branch so callers can treat the helper as total).
///
/// Round-5 fix: the previous table-based mapping followed the
/// pre-round-5 broken `cfl_signs` enumeration; both helpers shifted
/// in lock-step.
pub fn cfl_alpha_ctx(joint: u32, plane: u32) -> u32 {
    let v = (joint + 1) as i32;
    let su_code = v / 3; // 0=ZERO, 1=NEG, 2=POS
    let sv_code = v % 3;
    if plane == 0 {
        if su_code == 0 {
            return 0;
        }
        ((su_code - 1) * 3 + sv_code) as u32
    } else {
        if sv_code == 0 {
            return 0;
        }
        ((sv_code - 1) * 3 + su_code) as u32
    }
}

/// Build a [`super::coeffs::LevelCtxArrays`] view over a tile's
/// per-plane Above/Left context arrays for `plane ∈ 0..=2`.
///
/// Free function rather than a method so callers can pass
/// `&mut td.symbol` and `&mut td.coeff_bank` to
/// [`super::coeffs::decode_coefficients_spec`] alongside the returned
/// view without tripping the borrow checker (the view borrows only the
/// 4 per-plane Vec fields, not the rest of `TileDecoder`).
pub fn level_ctx_for_plane<'r>(
    above_level_ctx: &'r mut [Vec<u8>; 3],
    left_level_ctx: &'r mut [Vec<u8>; 3],
    above_dc_ctx: &'r mut [Vec<u8>; 3],
    left_dc_ctx: &'r mut [Vec<u8>; 3],
    plane: usize,
) -> super::coeffs::LevelCtxArrays<'r> {
    let p = plane.min(2);
    let max_x4 = above_level_ctx[p].len();
    let max_y4 = left_level_ctx[p].len();
    // index_mut panics on OOB but plane is clamped to 2.
    super::coeffs::LevelCtxArrays {
        above_level: above_level_ctx[p].as_mut_slice(),
        left_level: left_level_ctx[p].as_mut_slice(),
        above_dc: above_dc_ctx[p].as_mut_slice(),
        left_dc: left_dc_ctx[p].as_mut_slice(),
        max_x4,
        max_y4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_id_ctx_counts_nonzero_neighbors() {
        assert_eq!(segment_id_ctx(0, 0, true, true), 0);
        assert_eq!(segment_id_ctx(3, 0, true, true), 1);
        assert_eq!(segment_id_ctx(0, 5, true, true), 1);
        assert_eq!(segment_id_ctx(2, 4, true, true), 2);
        // Unavailable neighbors never contribute.
        assert_eq!(segment_id_ctx(7, 7, false, false), 0);
    }

    #[test]
    fn cfl_signs_table() {
        // Spec §6.10.14:
        //   joint | (signU, signV)
        //   0     | (ZERO, NEG)  → ( 0, -1)
        //   1     | (ZERO, POS)  → ( 0, +1)
        //   2     | (NEG,  ZERO) → (-1,  0)
        //   3     | (NEG,  NEG)  → (-1, -1)
        //   4     | (NEG,  POS)  → (-1, +1)
        //   5     | (POS,  ZERO) → (+1,  0)
        //   6     | (POS,  NEG)  → (+1, -1)
        //   7     | (POS,  POS)  → (+1, +1)
        assert_eq!(cfl_signs(0), (0, -1));
        assert_eq!(cfl_signs(1), (0, 1));
        assert_eq!(cfl_signs(2), (-1, 0));
        assert_eq!(cfl_signs(3), (-1, -1));
        assert_eq!(cfl_signs(4), (-1, 1));
        assert_eq!(cfl_signs(5), (1, 0));
        assert_eq!(cfl_signs(6), (1, -1));
        assert_eq!(cfl_signs(7), (1, 1));
    }

    #[test]
    fn cfl_alpha_ctx_plane_zero_when_sign_zero() {
        // For plane=0 (U), sign_u == 0 when joint in {0, 1}.
        assert_eq!(cfl_alpha_ctx(0, 0), 0);
        assert_eq!(cfl_alpha_ctx(1, 0), 0);
        // For plane=1 (V), sign_v == 0 when joint in {2, 5}.
        assert_eq!(cfl_alpha_ctx(2, 1), 0);
        assert_eq!(cfl_alpha_ctx(5, 1), 0);
    }

    /// Round 25 — `Tx_Size_Sqr` indexing for the inter ext-tx CDFs.
    /// The min-side mapping must produce the spec's TX_SIZES enum
    /// values (TX_4X4=0, TX_8X8=1, TX_16X16=2, TX_32X32=3) for every
    /// shape the inter sets actually consult.
    #[test]
    fn tx_size_sqr_index_table() {
        assert_eq!(tx_size_sqr_index(4, 4), 0);
        assert_eq!(tx_size_sqr_index(8, 8), 1);
        assert_eq!(tx_size_sqr_index(16, 16), 2);
        assert_eq!(tx_size_sqr_index(32, 32), 3);
        // 2:1 / 1:4 rectangles fold to the min side.
        assert_eq!(tx_size_sqr_index(8, 4), 0);
        assert_eq!(tx_size_sqr_index(4, 8), 0);
        assert_eq!(tx_size_sqr_index(16, 4), 0);
        assert_eq!(tx_size_sqr_index(4, 16), 0);
        assert_eq!(tx_size_sqr_index(16, 8), 1);
        assert_eq!(tx_size_sqr_index(32, 8), 1);
        assert_eq!(tx_size_sqr_index(32, 16), 2);
        assert_eq!(tx_size_sqr_index(16, 64), 2);
    }

    /// Round 25 — pin the wire shape of the inter ext-tx CDFs the
    /// inter Y site reads. Each CDF must have `N+1` entries (N inverted
    /// probabilities + 1 update counter), with the (N-1)th entry the
    /// 0 sentinel and the Nth entry the 0 update counter.
    #[test]
    fn inter_ext_tx_cdf_shapes_match_spec() {
        // Set 1: 16-symbol, 2 contexts.
        assert_eq!(cdfs::DEFAULT_INTER_EXT_TX_CDF_SET1.len(), 2);
        for row in cdfs::DEFAULT_INTER_EXT_TX_CDF_SET1 {
            assert_eq!(row.len(), 17, "set1 row must hold 17 entries");
            assert_eq!(row[15], 0, "set1 sentinel must be 0");
            assert_eq!(row[16], 0, "set1 counter must start at 0");
        }
        // Set 2: 12-symbol, single context.
        assert_eq!(cdfs::DEFAULT_INTER_EXT_TX_CDF_SET2.len(), 13);
        assert_eq!(cdfs::DEFAULT_INTER_EXT_TX_CDF_SET2[11], 0);
        assert_eq!(cdfs::DEFAULT_INTER_EXT_TX_CDF_SET2[12], 0);
        // Set 3: 2-symbol, 4 contexts.
        assert_eq!(cdfs::DEFAULT_INTER_EXT_TX_CDF_SET3.len(), 4);
        for row in cdfs::DEFAULT_INTER_EXT_TX_CDF_SET3 {
            assert_eq!(row.len(), 3, "set3 row must hold 3 entries");
            assert_eq!(row[1], 0);
            assert_eq!(row[2], 0);
        }
        // Set 1 must be monotonically decreasing in each row (wire
        // form is `Q15 P(X>i)*32768`, decreasing).
        for row in cdfs::DEFAULT_INTER_EXT_TX_CDF_SET1 {
            for win in row[..15].windows(2) {
                assert!(win[0] >= win[1], "wire CDF must be decreasing");
            }
        }
        for win in cdfs::DEFAULT_INTER_EXT_TX_CDF_SET2[..11].windows(2) {
            assert!(win[0] >= win[1], "wire CDF set2 must be decreasing");
        }
    }
}
