//! §5.11.57 `read_lr` / §5.11.58 `read_lr_unit` **write side** — the
//! per-superblock loop-restoration unit emitter that mirrors the
//! decode-walker [`crate::cdf::PartitionWalker::read_lr`] /
//! [`crate::cdf::PartitionWalker::decode_lr_unit`].
//!
//! Loop restoration (§7.17) applies a per-unit Wiener or self-guided
//! projection filter to the post-CDEF reconstruction. The bitstream
//! carries, per restoration unit, a filter selection symbol
//! (`use_wiener` / `use_sgrproj` / `restoration_type`) and — when the
//! unit is `RESTORE_WIENER` or `RESTORE_SGRPROJ` — its filter
//! coefficients coded with the §5.11.58 `decode_signed_subexp_with_ref_bool`
//! recentred-subexponential bool code.
//!
//! The §5.11.58 reader advances a running `RefLrWiener` / `RefSgrXqd`
//! reference across units in a tile; the writer maintains the identical
//! state ([`LrWriteState`]) so the emitted subexp codes reference the
//! same value the decoder will, byte-for-byte.

use crate::cdf::{
    count_units_in_frame_pub, round2_pub, LrParams, LrUnit, TileCdfContext, RESTORE_NONE,
    RESTORE_SGRPROJ, RESTORE_SWITCHABLE, RESTORE_WIENER,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::loop_restoration::{
    SGRPROJ_PARAMS_BITS, SGRPROJ_PRJ_BITS, SGRPROJ_PRJ_SUBEXP_K, SGRPROJ_XQD_MAX, SGRPROJ_XQD_MID,
    SGRPROJ_XQD_MIN, SGR_PARAMS, WIENER_COEFFS, WIENER_TAPS_K, WIENER_TAPS_MAX, WIENER_TAPS_MID,
    WIENER_TAPS_MIN,
};
use crate::Error;

const MI_SIZE: u32 = 4;
const SUPERRES_NUM: u32 = crate::frame_header::SUPERRES_NUM;
const BLOCK_SIZES: usize = 22;

/// Running §5.11.2 / §5.11.58 reference state the encoder advances as it
/// writes successive loop-restoration units in a tile — the write-side
/// twin of the decode-walker's `RefLrWiener` / `RefSgrXqd` fields.
///
/// §5.11.2 `decode_tile()` resets every slot to its mid value at tile
/// entry; [`Self::new`] / [`Self::reset`] perform the same fill, and each
/// [`write_lr_unit`] call that emits a `RESTORE_WIENER` tap or a
/// `RESTORE_SGRPROJ` `xqd` overwrites the matching slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LrWriteState {
    ref_lr_wiener: [[[i32; WIENER_COEFFS]; 2]; 3],
    ref_sgr_xqd: [[i32; 2]; 3],
}

impl Default for LrWriteState {
    fn default() -> Self {
        Self::new()
    }
}

impl LrWriteState {
    /// Construct a fresh tile-entry state (every `RefLrWiener` slot at
    /// `Wiener_Taps_Mid`, every `RefSgrXqd` slot at `Sgrproj_Xqd_Mid`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            ref_lr_wiener: [[WIENER_TAPS_MID; 2]; 3],
            ref_sgr_xqd: [SGRPROJ_XQD_MID; 3],
        }
    }

    /// §5.11.2 tile-entry reset — re-fill every reference slot to mid.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// `read_lr_unit(plane, unitRow, unitCol)` **write side** per §5.11.58
/// (av1-spec p.106) — emit one loop-restoration unit's filter selection
/// plus (for `RESTORE_WIENER` / `RESTORE_SGRPROJ`) its coefficients.
///
/// `frame_restoration_type` is the §5.9.20 `FrameRestorationType[ plane ]`
/// ordinal that decides which selection symbol is coded:
///
/// * `RESTORE_WIENER` ⇒ binary `use_wiener` (`unit.restoration_type ∈
///   {RESTORE_NONE, RESTORE_WIENER}` — anything else is a caller bug).
/// * `RESTORE_SGRPROJ` ⇒ binary `use_sgrproj` (`unit.restoration_type ∈
///   {RESTORE_NONE, RESTORE_SGRPROJ}`).
/// * `RESTORE_SWITCHABLE` ⇒ 3-symbol `restoration_type`
///   (`RESTORE_NONE` / `RESTORE_WIENER` / `RESTORE_SGRPROJ`).
///
/// `state` is advanced exactly as the §5.11.58 reader advances its
/// `RefLrWiener` / `RefSgrXqd`, so successive units reference the prior
/// unit's taps. The emitted bytes round-trip through
/// [`crate::cdf::PartitionWalker::decode_lr_unit`].
///
/// Returns [`Error::PartitionWalkOutOfRange`] for a `plane >= 3` /
/// `frame_restoration_type > 3` caller bug, or a `unit.restoration_type`
/// that contradicts `frame_restoration_type` (e.g. a `RESTORE_SGRPROJ`
/// unit under a `RESTORE_WIENER` plane).
pub fn write_lr_unit(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut LrWriteState,
    plane: usize,
    frame_restoration_type: u8,
    unit: &LrUnit,
) -> Result<(), Error> {
    if plane >= 3 || frame_restoration_type > RESTORE_SWITCHABLE {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let rt = unit.restoration_type;

    // §5.11.58 filter-selection S(): the symbol coded depends on the
    // plane's FrameRestorationType, and the unit's restoration_type must
    // be consistent with it.
    match frame_restoration_type {
        RESTORE_WIENER => {
            if rt != RESTORE_NONE && rt != RESTORE_WIENER {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let sym = u32::from(rt == RESTORE_WIENER);
            writer.write_symbol(sym, cdfs.use_wiener_cdf())?;
        }
        RESTORE_SGRPROJ => {
            if rt != RESTORE_NONE && rt != RESTORE_SGRPROJ {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let sym = u32::from(rt == RESTORE_SGRPROJ);
            writer.write_symbol(sym, cdfs.use_sgrproj_cdf())?;
        }
        RESTORE_SWITCHABLE => {
            if rt > RESTORE_SGRPROJ {
                return Err(Error::PartitionWalkOutOfRange);
            }
            writer.write_symbol(u32::from(rt), cdfs.restoration_type_cdf())?;
        }
        // RESTORE_NONE plane: read_lr never reaches read_lr_unit, so the
        // writer must not be called for it either.
        _ => return Err(Error::PartitionWalkOutOfRange),
    }

    match rt {
        RESTORE_WIENER => {
            let ref_wiener = &mut state.ref_lr_wiener[plane];
            for (pass_taps, ref_pass) in unit.wiener.iter().zip(ref_wiener.iter_mut()) {
                let first_coeff = if plane != 0 { 1 } else { 0 };
                // Chroma `[pass][0]` is forced to 0 (no symbol).
                if plane != 0 && pass_taps[0] != 0 {
                    return Err(Error::PartitionWalkOutOfRange);
                }
                for j in first_coeff..WIENER_COEFFS {
                    let min = i64::from(WIENER_TAPS_MIN[j]);
                    let max = i64::from(WIENER_TAPS_MAX[j]);
                    let k = WIENER_TAPS_K[j];
                    let r = i64::from(ref_pass[j]);
                    let v = i64::from(pass_taps[j]);
                    if v < min || v > max {
                        return Err(Error::PartitionWalkOutOfRange);
                    }
                    writer.write_signed_subexp_with_ref_bool(min, max + 1, k, r, v)?;
                    ref_pass[j] = v as i32;
                }
            }
        }
        RESTORE_SGRPROJ => {
            if unit.sgr_set >= SGR_PARAMS.len() {
                return Err(Error::PartitionWalkOutOfRange);
            }
            writer.write_literal(SGRPROJ_PARAMS_BITS, unit.sgr_set as u32)?;
            for i in 0..2 {
                let radius = SGR_PARAMS[unit.sgr_set][i * 2];
                let min = i64::from(SGRPROJ_XQD_MIN[i]);
                let max = i64::from(SGRPROJ_XQD_MAX[i]);
                let v = i64::from(unit.sgr_xqd[i]);
                if radius != 0 {
                    if v < min || v > max {
                        return Err(Error::PartitionWalkOutOfRange);
                    }
                    let r = i64::from(state.ref_sgr_xqd[plane][i]);
                    writer.write_signed_subexp_with_ref_bool(
                        min,
                        max + 1,
                        SGRPROJ_PRJ_SUBEXP_K,
                        r,
                        v,
                    )?;
                } else {
                    // radius == 0: no symbol. The §5.11.58 derived value
                    // must match (i == 1 ⇒ Clip3(min, max, (1 <<
                    // SGRPROJ_PRJ_BITS) - RefSgrXqd[plane][0]); else 0).
                    let derived = if i == 1 {
                        let d = (1i32 << SGRPROJ_PRJ_BITS) - state.ref_sgr_xqd[plane][0];
                        d.clamp(min as i32, max as i32)
                    } else {
                        0
                    };
                    if unit.sgr_xqd[i] != derived {
                        return Err(Error::PartitionWalkOutOfRange);
                    }
                }
                state.ref_sgr_xqd[plane][i] = v as i32;
            }
        }
        // RESTORE_NONE: no coefficients follow.
        _ => {}
    }
    Ok(())
}

/// `read_lr(r, c, bSize)` **write side** per §5.11.57 (av1-spec p.105) —
/// emit every loop-restoration unit the superblock `(r, c)` of size
/// `bSize` covers, in §5.11.57 raster order.
///
/// `units` supplies the decoded payloads keyed by `(plane, unitRow,
/// unitCol)`; the driver walks the same per-plane unit-window arithmetic
/// the reader does and, for each `(unitRow, unitCol)` it visits, looks up
/// the matching unit and emits it via [`write_lr_unit`]. A missing unit
/// for a coordinate the window visits is a caller bug
/// ([`Error::PartitionWalkOutOfRange`]).
///
/// Emits nothing when `allow_intrabc` is set (§5.11.57 returns
/// immediately) or every plane is `RESTORE_NONE`.
#[allow(clippy::too_many_arguments)]
pub fn write_lr(
    writer: &mut SymbolWriter,
    cdfs: &mut TileCdfContext,
    state: &mut LrWriteState,
    r: u32,
    c: u32,
    b_size: usize,
    params: &LrParams,
    units: &[((usize, u32, u32), LrUnit)],
) -> Result<(), Error> {
    if b_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if params.allow_intrabc {
        return Ok(());
    }
    let w = num_4x4_blocks_wide(b_size);
    let h = num_4x4_blocks_high(b_size);
    for plane in 0..params.num_planes {
        let frt = params.frame_restoration_type[plane];
        if frt == RESTORE_NONE {
            continue;
        }
        let sub_x = if plane == 0 { 0 } else { params.subsampling_x };
        let sub_y = if plane == 0 { 0 } else { params.subsampling_y };
        let unit_size = params.loop_restoration_size[plane];
        if unit_size == 0 {
            continue;
        }
        let unit_rows = count_units_in_frame_pub(unit_size, round2_pub(params.frame_height, sub_y));
        let unit_cols =
            count_units_in_frame_pub(unit_size, round2_pub(params.upscaled_width, sub_x));
        let mi_size_shift_y = MI_SIZE >> sub_y;
        let unit_row_start = (r * mi_size_shift_y).div_ceil(unit_size);
        let unit_row_end =
            core::cmp::min(unit_rows, ((r + h) * mi_size_shift_y).div_ceil(unit_size));
        let mi_size_shift_x = MI_SIZE >> sub_x;
        let (numerator, denominator) = if params.use_superres {
            (
                mi_size_shift_x * params.superres_denom,
                unit_size * SUPERRES_NUM,
            )
        } else {
            (mi_size_shift_x, unit_size)
        };
        let unit_col_start = (c * numerator).div_ceil(denominator);
        let unit_col_end = core::cmp::min(unit_cols, ((c + w) * numerator).div_ceil(denominator));
        let mut unit_row = unit_row_start;
        while unit_row < unit_row_end {
            let mut unit_col = unit_col_start;
            while unit_col < unit_col_end {
                let key = (plane, unit_row, unit_col);
                let unit = units
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, u)| u)
                    .ok_or(Error::PartitionWalkOutOfRange)?;
                write_lr_unit(writer, cdfs, state, plane, frt, unit)?;
                unit_col += 1;
            }
            unit_row += 1;
        }
    }
    Ok(())
}

// §3 `Num_4x4_Blocks_Wide[ bSize ]` / `Num_4x4_Blocks_High` — re-derived
// from the canonical [`crate::cdf`] tables so this module stays
// self-contained without re-exporting the (large) const arrays.
fn num_4x4_blocks_wide(b_size: usize) -> u32 {
    crate::cdf::num_4x4_blocks_wide_pub(b_size)
}
fn num_4x4_blocks_high(b_size: usize) -> u32 {
    crate::cdf::num_4x4_blocks_high_pub(b_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{PartitionWalker, TileGeometry, BLOCK_64X64};
    use crate::symbol_decoder::SymbolDecoder;

    fn rt_geometry(mi_rows: u32, mi_cols: u32) -> TileGeometry {
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        }
    }

    fn single_plane_params(unit_size: u32, frt: u8) -> LrParams {
        LrParams {
            num_planes: 1,
            frame_restoration_type: [frt, RESTORE_NONE, RESTORE_NONE],
            loop_restoration_size: [unit_size, unit_size, unit_size],
            subsampling_x: 0,
            subsampling_y: 0,
            frame_height: 64,
            upscaled_width: 64,
            use_superres: false,
            superres_denom: SUPERRES_NUM,
            allow_intrabc: false,
        }
    }

    #[test]
    fn wiener_unit_round_trips() {
        // A single luma 64x64 superblock, one Wiener unit covering it.
        let unit = LrUnit {
            restoration_type: RESTORE_WIENER,
            wiener: [[-3, 4, 12], [2, -7, 20]],
            sgr_set: 0,
            sgr_xqd: [0; 2],
        };
        let mut writer = SymbolWriter::new(false);
        let mut wcdfs = TileCdfContext::new_from_defaults();
        let mut wstate = LrWriteState::new();
        write_lr_unit(
            &mut writer,
            &mut wcdfs,
            &mut wstate,
            0,
            RESTORE_WIENER,
            &unit,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut walker = PartitionWalker::new(16, 16, rt_geometry(16, 16)).unwrap();
        let mut dcdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = walker
            .decode_lr_unit(&mut dec, &mut dcdfs, 0, RESTORE_WIENER)
            .unwrap();
        assert_eq!(got, unit);
        // Adapted CDFs must match too.
        assert_eq!(wcdfs.use_wiener, dcdfs.use_wiener);
    }

    #[test]
    fn sgrproj_unit_round_trips() {
        // Pick a set whose Sgr_Params has both radii non-zero so both
        // xqd values are coded.
        let set = SGR_PARAMS
            .iter()
            .position(|p| p[0] != 0 && p[2] != 0)
            .unwrap_or(0);
        let unit = LrUnit {
            restoration_type: RESTORE_SGRPROJ,
            wiener: [[0; WIENER_COEFFS]; 2],
            sgr_set: set,
            sgr_xqd: [-10, 40],
        };
        let mut writer = SymbolWriter::new(false);
        let mut wcdfs = TileCdfContext::new_from_defaults();
        let mut wstate = LrWriteState::new();
        write_lr_unit(
            &mut writer,
            &mut wcdfs,
            &mut wstate,
            0,
            RESTORE_SGRPROJ,
            &unit,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut walker = PartitionWalker::new(16, 16, rt_geometry(16, 16)).unwrap();
        let mut dcdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = walker
            .decode_lr_unit(&mut dec, &mut dcdfs, 0, RESTORE_SGRPROJ)
            .unwrap();
        assert_eq!(got, unit);
    }

    #[test]
    fn switchable_none_round_trips() {
        // A RESTORE_SWITCHABLE plane whose unit selects RESTORE_NONE —
        // one 3-symbol restoration_type and no coefficients.
        let unit = LrUnit {
            restoration_type: RESTORE_NONE,
            wiener: [[0; WIENER_COEFFS]; 2],
            sgr_set: 0,
            sgr_xqd: [0; 2],
        };
        let mut writer = SymbolWriter::new(false);
        let mut wcdfs = TileCdfContext::new_from_defaults();
        let mut wstate = LrWriteState::new();
        write_lr_unit(
            &mut writer,
            &mut wcdfs,
            &mut wstate,
            0,
            RESTORE_SWITCHABLE,
            &unit,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut walker = PartitionWalker::new(16, 16, rt_geometry(16, 16)).unwrap();
        let mut dcdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = walker
            .decode_lr_unit(&mut dec, &mut dcdfs, 0, RESTORE_SWITCHABLE)
            .unwrap();
        assert_eq!(got, unit);
    }

    #[test]
    fn read_lr_superblock_round_trips() {
        // One 64x64 superblock, unit_size 64 => exactly one luma unit.
        let params = single_plane_params(64, RESTORE_WIENER);
        let unit = LrUnit {
            restoration_type: RESTORE_WIENER,
            wiener: [[1, -2, 8], [-1, 3, 10]],
            sgr_set: 0,
            sgr_xqd: [0; 2],
        };
        let units = vec![((0usize, 0u32, 0u32), unit)];

        let mut writer = SymbolWriter::new(false);
        let mut wcdfs = TileCdfContext::new_from_defaults();
        let mut wstate = LrWriteState::new();
        write_lr(
            &mut writer,
            &mut wcdfs,
            &mut wstate,
            0,
            0,
            BLOCK_64X64,
            &params,
            &units,
        )
        .unwrap();
        let bytes = writer.finish();

        let mut walker = PartitionWalker::new(16, 16, rt_geometry(16, 16)).unwrap();
        let mut dcdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = walker
            .read_lr(&mut dec, &mut dcdfs, 0, 0, BLOCK_64X64, &params)
            .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].plane, 0);
        assert_eq!(got[0].unit_row, 0);
        assert_eq!(got[0].unit_col, 0);
        assert_eq!(got[0].unit, units[0].1);
    }

    #[test]
    fn allow_intrabc_emits_nothing() {
        let mut params = single_plane_params(64, RESTORE_WIENER);
        params.allow_intrabc = true;
        let mut writer = SymbolWriter::new(false);
        let mut wcdfs = TileCdfContext::new_from_defaults();
        let mut wstate = LrWriteState::new();
        write_lr(
            &mut writer,
            &mut wcdfs,
            &mut wstate,
            0,
            0,
            BLOCK_64X64,
            &params,
            &[],
        )
        .unwrap();
        let bytes = writer.finish();
        // exit_symbol pads to a minimal byte-aligned tail; the body must
        // carry no real symbols. The decode walker returns an empty set.
        let mut walker = PartitionWalker::new(16, 16, rt_geometry(16, 16)).unwrap();
        let mut dcdfs = TileCdfContext::new_from_defaults();
        let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
        let got = walker
            .read_lr(&mut dec, &mut dcdfs, 0, 0, BLOCK_64X64, &params)
            .unwrap();
        assert!(got.is_empty());
    }
}
