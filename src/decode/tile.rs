//! AV1 tile decoder — §5.11 `decode_tile`.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/tile.go`
//! (MIT, KarpelesLab/goavif). The goavif port also carries the
//! coefficient decoder (`CoeffDecoder`) and optional inter-frame
//! syntax reader (`InterDecoder`); Phase 2 keeps only the symbol
//! decoder + the CDF bank required by the partition / mode /
//! segment / CFL / skip readers.
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
use crate::frame_header::FrameHeader;
use crate::sequence_header::SequenceHeader;
use crate::symbol::SymbolDecoder;
use crate::tile_group::{parse_tile_group_header, split_tile_payloads, TilePayload};

use super::coeffs::{q_index_to_ctx, CoeffCdfBank};
use super::frame_state::FrameState;
use super::modes::{IntraMode, INTRA_MODES, UV_MODES};
use super::superblock;
use crate::transform::TxType;

/// Top-level entry point: walk every tile in a tile-group payload,
/// running [`TileDecoder::decode`] on each. The underlying frame
/// state is mutated in place.
///
/// `tile_payload` is the entire tile-group OBU payload (header +
/// per-tile size prefixes + concatenated tile data, as surfaced by
/// [`crate::tile_group::split_tile_payloads`]).
pub fn decode_tile_group(
    seq: &SequenceHeader,
    frame: &FrameHeader,
    tile_payload: &[u8],
    frame_state: &mut FrameState,
) -> Result<()> {
    let tile_info = frame.tile_info.as_ref().ok_or_else(|| {
        Error::invalid(
            "av1 decode_tile_group: frame header missing tile_info (§5.9.15)",
        )
    })?;
    let tgh = parse_tile_group_header(tile_payload, tile_info)?;
    let tiles = split_tile_payloads(tile_payload, tile_info, &tgh)?;
    for tp in &tiles {
        let bytes = &tile_payload[tp.offset..tp.offset + tp.len];
        let mut td = TileDecoder::new(seq, frame, bytes)?;
        td.decode(frame_state, tp)?;
    }
    Ok(())
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
    pub cfl_sign_cdf: Vec<u16>,
    pub cfl_alpha_cdf: Vec<Vec<u16>>,
    pub seg_cdf: Vec<Vec<u16>>,
    /// Intra-frame `tx_type` CDF — set 1 (7 types, `tx_size_ctx ∈ 0..=3`,
    /// intra_mode ∈ 0..=12). Used by blocks whose area ≤ 16×16.
    pub intra_ext_tx_cdf_set1: Vec<Vec<Vec<u16>>>,
    /// Intra-frame `tx_type` CDF — set 2 (5 types, same shape as set 1).
    /// Used by blocks whose area ≤ 32×32. Blocks larger than 32×32 use
    /// implicit DCT_DCT.
    pub intra_ext_tx_cdf_set2: Vec<Vec<Vec<u16>>>,
    /// Coefficient-CDF bank — all of §5.11.39's CDFs in one place.
    pub coeff_bank: CoeffCdfBank,
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
    ) -> Result<Self> {
        let sz = tile_data.len();
        let allow_update = !frame.disable_cdf_update;
        let symbol = SymbolDecoder::new(tile_data, sz, allow_update)?;
        let sb_size = if seq.use_128x128_superblock { 128 } else { 64 };
        let q_ctx = q_index_to_ctx(frame.quant.base_q_idx as i32);
        let coeff_bank = CoeffCdfBank::new(q_ctx);
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
            cfl_sign_cdf: Vec::new(),
            cfl_alpha_cdf: Vec::new(),
            seg_cdf: Vec::new(),
            intra_ext_tx_cdf_set1: Vec::new(),
            intra_ext_tx_cdf_set2: Vec::new(),
            coeff_bank,
        };
        td.init_cdfs();
        Ok(td)
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
        self.cfl_sign_cdf = cdfs::DEFAULT_CFL_SIGN_CDF.to_vec();
        self.cfl_alpha_cdf = cdfs::DEFAULT_CFL_ALPHA_CDF
            .iter()
            .map(|c| c.to_vec())
            .collect();
        self.seg_cdf = cdfs::DEFAULT_SPATIAL_PRED_SEG_TREE_CDF
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
    }

    /// Walk the tile's superblock grid and decode every MI unit's
    /// mode info. Returns `Ok(())` only if the tile decodes end to
    /// end with every leaf marked skip; otherwise returns
    /// `Error::Unsupported("av1 coefficient decode pending (§5.11.39)")`.
    pub fn decode(&mut self, fs: &mut FrameState, tp: &TilePayload) -> Result<()> {
        let tile_info = self.frame.tile_info.as_ref().ok_or_else(|| {
            Error::invalid("av1 decode_tile: tile_info missing")
        })?;
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
    // Each mirrors a goavif TileDecoder method, minus the bounds
    // check style (goavif silently clamps out-of-range contexts; we
    // clamp too so invalid inputs don't panic).

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
    pub fn decode_intra_y_mode(
        &mut self,
        above_ctx: u32,
        left_ctx: u32,
    ) -> Result<IntraMode> {
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
    pub fn decode_uv_mode(
        &mut self,
        y_mode: IntraMode,
        cfl_allowed: bool,
    ) -> Result<IntraMode> {
        let cfl_idx = if cfl_allowed { 1 } else { 0 };
        let y_idx = y_mode as usize;
        let raw = self.symbol.decode_symbol(&mut self.uv_mode_cdf[cfl_idx][y_idx])?;
        if (raw as usize) >= UV_MODES {
            return Err(Error::invalid(format!(
                "av1 uv_mode: symbol {raw} out of range (§5.11.18)"
            )));
        }
        IntraMode::from_u32(raw).ok_or_else(|| {
            Error::invalid(format!("av1 uv_mode: invalid mode {raw} (§5.11.18)"))
        })
    }

    /// Decode `angle_delta` for a directional mode. `dir_idx` is the
    /// directional mode index (`y_mode - D45_PRED`, in 0..=7).
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

    /// Decode the segment_id for a block (spatial prediction,
    /// §5.11.9). `ctx` is the 3-way max-based neighbor context.
    pub fn decode_segment_id(&mut self, ctx: u32) -> Result<u8> {
        let ctx = (ctx as usize).min(2);
        let raw = self.symbol.decode_symbol(&mut self.seg_cdf[ctx])?;
        Ok(raw as u8)
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
pub fn segment_id_ctx(
    above_id: u8,
    left_id: u8,
    have_above: bool,
    have_left: bool,
) -> u32 {
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
/// element is -1, 0, or +1. Spec §6.10.14.
pub fn cfl_signs(joint: u32) -> (i32, i32) {
    match joint {
        0 => (-1, -1),
        1 => (-1, 0),
        2 => (-1, 1),
        3 => (0, -1),
        4 => (0, 1),
        5 => (1, -1),
        6 => (1, 0),
        7 => (1, 1),
        _ => (0, 0),
    }
}

/// CFL alpha CDF context for a given joint sign + plane. Matches
/// libaom's CFL_ALPHA_CTX mapping (goavif parity).
pub fn cfl_alpha_ctx(joint: u32, plane: u32) -> u32 {
    let (su, sv) = cfl_signs(joint);
    if plane == 0 {
        if su == 0 {
            return 0;
        }
        match joint {
            0 => 0,
            1 => 1,
            2 => 2,
            5 => 3,
            6 => 4,
            7 => 5,
            _ => 0,
        }
    } else {
        if sv == 0 {
            return 0;
        }
        match joint {
            0 => 0,
            3 => 1,
            5 => 2,
            2 => 3,
            4 => 4,
            7 => 5,
            _ => 0,
        }
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
        assert_eq!(cfl_signs(0), (-1, -1));
        assert_eq!(cfl_signs(7), (1, 1));
        assert_eq!(cfl_signs(3), (0, -1));
    }

    #[test]
    fn cfl_alpha_ctx_plane_zero_when_sign_zero() {
        // For plane=0 (U), sign_u == 0 when joint in {3, 4}.
        assert_eq!(cfl_alpha_ctx(3, 0), 0);
        assert_eq!(cfl_alpha_ctx(4, 0), 0);
        // For plane=1 (V), sign_v == 0 when joint in {1, 6}.
        assert_eq!(cfl_alpha_ctx(1, 1), 0);
        assert_eq!(cfl_alpha_ctx(6, 1), 0);
    }
}
