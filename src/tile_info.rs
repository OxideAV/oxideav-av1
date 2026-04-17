//! AV1 `tile_info()` parser — §5.9.15.
//!
//! The tile-info syntax follows the frame-header `disable_frame_end_update_cdf`
//! bit and is the last piece of the uncompressed header relevant to tile
//! boundary calculation. Full decoder implementations use the parsed values
//! to split each `OBU_TILE_GROUP` payload into per-tile byte ranges, and to
//! know which `CurrentQIndex` and `MiRow/MiCol` extents apply to each tile.
//!
//! AV1 defines two spacings:
//!
//! * **Uniform** — `tile_cols_log2` / `tile_rows_log2` are read via a
//!   monotonic sequence of `increment_*_log2` bits. Column / row widths are
//!   then derived by dividing the frame superblock grid evenly.
//! * **Non-uniform** — each tile declares its width / height in superblocks
//!   using the `ns(maxWidth|maxHeight)` encoding.
//!
//! Both paths populate `MiColStarts[]` / `MiRowStarts[]` terminated by
//! `MiCols` / `MiRows`, so `tile_cols = MiColStarts.len() - 1`.
//!
//! This module is pure-Rust, no external dependencies. The parser asks
//! the caller for the already-computed `sb_size_log2` / `MiCols` / `MiRows`,
//! which the frame-header code has available at this point.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;

/// §3 constants (Table 3-1).
pub const MAX_TILE_WIDTH_PX: u32 = 4096;
pub const MAX_TILE_AREA_PX: u32 = 4096 * 2304;
pub const MAX_TILE_COLS: u32 = 64;
pub const MAX_TILE_ROWS: u32 = 64;
pub const MI_SIZE_LOG2: u32 = 2;

/// Smallest `k` such that `blk_size << k >= target` — §6.2.14.
pub fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0u32;
    loop {
        let shifted = blk_size.checked_shl(k).unwrap_or(u32::MAX);
        if shifted >= target {
            return k;
        }
        k += 1;
        if k >= 31 {
            return k;
        }
    }
}

/// Parsed `tile_info()`.
#[derive(Clone, Debug)]
pub struct TileInfo {
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub uniform_tile_spacing_flag: bool,
    /// `MiColStarts[0..=tile_cols]`; last entry == MiCols.
    pub mi_col_starts: Vec<u32>,
    /// `MiRowStarts[0..=tile_rows]`; last entry == MiRows.
    pub mi_row_starts: Vec<u32>,
    /// `TileSizeBytes` — 1..=4 or 0 for single-tile frames.
    pub tile_size_bytes: u32,
    pub context_update_tile_id: u32,
}

impl TileInfo {
    pub fn num_tiles(&self) -> u32 {
        self.tile_cols * self.tile_rows
    }
}

/// Parse `tile_info()` starting from the current bit position of `br`.
///
/// Parameters:
/// * `use_128x128_superblock` — from the sequence header.
/// * `mi_cols` / `mi_rows` — derived from `FrameWidth` / `FrameHeight` via
///   §6.8.4: `MiCols = 2 * ((FrameWidth + 7) >> 3)` and symmetric.
pub fn parse_tile_info(
    br: &mut BitReader<'_>,
    use_128x128_superblock: bool,
    mi_cols: u32,
    mi_rows: u32,
) -> Result<TileInfo> {
    let sb_shift = if use_128x128_superblock { 5 } else { 4 };
    let sb_size = sb_shift + 2; // log2 of superblock side in luma samples
    let sb_cols = if use_128x128_superblock {
        (mi_cols + 31) >> 5
    } else {
        (mi_cols + 15) >> 4
    };
    let sb_rows = if use_128x128_superblock {
        (mi_rows + 31) >> 5
    } else {
        (mi_rows + 15) >> 4
    };
    if sb_cols == 0 || sb_rows == 0 {
        return Err(Error::invalid(
            "av1 tile_info: zero-sized superblock grid (§5.9.15)",
        ));
    }
    let max_tile_width_sb = MAX_TILE_WIDTH_PX >> sb_size;
    let max_tile_area_sb = MAX_TILE_AREA_PX >> (2 * sb_size);

    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let uniform_tile_spacing_flag = br.bit()?;
    let mut mi_col_starts: Vec<u32> = Vec::new();
    let mut mi_row_starts: Vec<u32> = Vec::new();
    let tile_cols_log2;
    let tile_rows_log2;
    let tile_cols;
    let tile_rows;

    if uniform_tile_spacing_flag {
        let mut t_cols_log2 = min_log2_tile_cols;
        while t_cols_log2 < max_log2_tile_cols {
            let increment = br.bit()?;
            if increment {
                t_cols_log2 += 1;
            } else {
                break;
            }
        }
        let tile_width_sb = (sb_cols + (1u32 << t_cols_log2) - 1) >> t_cols_log2;
        let mut i = 0u32;
        let mut start_sb = 0u32;
        while start_sb < sb_cols {
            mi_col_starts.push(start_sb << sb_shift);
            start_sb += tile_width_sb;
            i += 1;
        }
        mi_col_starts.push(mi_cols);
        tile_cols = i;
        tile_cols_log2 = t_cols_log2;

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(t_cols_log2);
        let mut t_rows_log2 = min_log2_tile_rows;
        while t_rows_log2 < max_log2_tile_rows {
            let increment = br.bit()?;
            if increment {
                t_rows_log2 += 1;
            } else {
                break;
            }
        }
        let tile_height_sb = (sb_rows + (1u32 << t_rows_log2) - 1) >> t_rows_log2;
        let mut j = 0u32;
        let mut start_sb = 0u32;
        while start_sb < sb_rows {
            mi_row_starts.push(start_sb << sb_shift);
            start_sb += tile_height_sb;
            j += 1;
        }
        mi_row_starts.push(mi_rows);
        tile_rows = j;
        tile_rows_log2 = t_rows_log2;
    } else {
        // Non-uniform spacing: each tile declares its width / height.
        let mut widest_tile_sb = 0u32;
        let mut start_sb = 0u32;
        let mut i = 0u32;
        while start_sb < sb_cols {
            mi_col_starts.push(start_sb << sb_shift);
            let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
            let width_in_sbs_minus_1 = br.ns(max_width)?;
            let size_sb = width_in_sbs_minus_1 + 1;
            widest_tile_sb = widest_tile_sb.max(size_sb);
            start_sb += size_sb;
            i += 1;
        }
        mi_col_starts.push(mi_cols);
        tile_cols = i;
        tile_cols_log2 = tile_log2(1, tile_cols);

        let area_sb = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (area_sb / widest_tile_sb.max(1)).max(1);

        let mut start_sb = 0u32;
        let mut j = 0u32;
        while start_sb < sb_rows {
            mi_row_starts.push(start_sb << sb_shift);
            let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
            let height_in_sbs_minus_1 = br.ns(max_height)?;
            let size_sb = height_in_sbs_minus_1 + 1;
            start_sb += size_sb;
            j += 1;
        }
        mi_row_starts.push(mi_rows);
        tile_rows = j;
        tile_rows_log2 = tile_log2(1, tile_rows);
    }

    if tile_cols > MAX_TILE_COLS || tile_rows > MAX_TILE_ROWS {
        return Err(Error::invalid(format!(
            "av1 tile_info: tile_cols={tile_cols} tile_rows={tile_rows} \
             exceed MAX_TILE_{{COLS,ROWS}}=64 (§5.9.15)",
        )));
    }

    let (context_update_tile_id, tile_size_bytes) = if tile_cols_log2 > 0 || tile_rows_log2 > 0 {
        let id_bits = tile_cols_log2 + tile_rows_log2;
        let id = br.f(id_bits)?;
        let tile_size_bytes_minus_1 = br.f(2)?;
        (id, tile_size_bytes_minus_1 + 1)
    } else {
        (0, 0)
    };

    Ok(TileInfo {
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        uniform_tile_spacing_flag,
        mi_col_starts,
        mi_row_starts,
        tile_size_bytes,
        context_update_tile_id,
    })
}

/// Compute `MiCols` / `MiRows` from the frame dimensions (§6.8.4).
pub fn mi_cols_rows(frame_width: u32, frame_height: u32) -> (u32, u32) {
    let mi_cols = 2 * ((frame_width + 7) >> 3);
    let mi_rows = 2 * ((frame_height + 7) >> 3);
    (mi_cols, mi_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_log2_basics() {
        assert_eq!(tile_log2(1, 1), 0);
        assert_eq!(tile_log2(1, 2), 1);
        assert_eq!(tile_log2(1, 3), 2);
        assert_eq!(tile_log2(1, 4), 2);
        assert_eq!(tile_log2(1, 5), 3);
        assert_eq!(tile_log2(8, 8), 0);
        assert_eq!(tile_log2(8, 9), 1);
    }

    #[test]
    fn mi_cols_rows_64x64() {
        let (mc, mr) = mi_cols_rows(64, 64);
        // (64+7)>>3 = 8; *2 = 16.
        assert_eq!(mc, 16);
        assert_eq!(mr, 16);
    }

    #[test]
    fn single_tile_uniform_64x64_64sb() {
        // `uniform_tile_spacing_flag=1`, then min_log2 == max_log2 so no
        // increments are read. For a 64x64 frame with 64-sample superblocks,
        // sb_cols = sb_rows = 1, so min=max=0 and the uniform branch exits
        // immediately with a single tile. tile_cols_log2 == tile_rows_log2 ==
        // 0, so context_update_tile_id + tile_size_bytes are not read.
        //
        // Bitstream: 0b1_0000000 (msb-first) — one bit `1` is consumed for
        // `uniform_tile_spacing_flag`; remaining bits are padding.
        let data = [0b1000_0000u8];
        let mut br = BitReader::new(&data);
        let ti = parse_tile_info(&mut br, false, 16, 16).unwrap();
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(ti.mi_col_starts, vec![0, 16]);
        assert_eq!(ti.mi_row_starts, vec![0, 16]);
        assert_eq!(ti.tile_size_bytes, 0);
        assert!(ti.uniform_tile_spacing_flag);
    }
}
