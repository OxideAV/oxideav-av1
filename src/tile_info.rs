//! `tile_info()` sub-syntax (§5.9.15) — round 6 of the clean-room
//! rebuild.
//!
//! The function takes the post-`compute_image_size()` `MiCols` /
//! `MiRows` derived in §5.9.9, plus the sequence-header
//! `use_128x128_superblock` bit (§5.5.1), and walks the per-frame tile
//! layout: either a uniform spacing path that reads
//! `increment_tile_cols_log2` / `increment_tile_rows_log2` until the
//! `TileColsLog2` / `TileRowsLog2` counts saturate against
//! `maxLog2TileCols` / `maxLog2TileRows`, or a non-uniform path that
//! reads `width_in_sbs_minus_1` / `height_in_sbs_minus_1` via the §4.10.7
//! `ns(n)` descriptor.
//!
//! ## Syntax / semantics references (all in `docs/video/av1/`)
//!
//!   * §3 — `MAX_TILE_WIDTH = 4096`, `MAX_TILE_AREA = 4096 * 2304`,
//!     `MAX_TILE_ROWS = 64`, `MAX_TILE_COLS = 64`, `MI_SIZE = 4`.
//!   * §4.7 — `FloorLog2(x)` helper.
//!   * §4.10.7 — `ns(n)` descriptor used for `width_in_sbs_minus_1`
//!     and `height_in_sbs_minus_1`.
//!   * §5.9.15 — `tile_info()` syntax (this file).
//!   * §5.9.16 — `tile_log2(blkSize, target)` helper function.
//!   * §6.8.14 — Tile info semantics + conformance constraints
//!     (`TileCols <= MAX_TILE_COLS`, `TileRows <= MAX_TILE_ROWS`,
//!     `tileWidthSb < maxTileWidthSb`,
//!     `tileWidthSb * tileHeightSb < maxTileAreaSb`,
//!     `context_update_tile_id < TileCols * TileRows`).

use crate::bitreader::BitReader;
use crate::Error;

// ---------------------------------------------------------------------
// §3 constants
// ---------------------------------------------------------------------

/// `MAX_TILE_WIDTH` per §3 — maximum tile width in luma samples.
pub const MAX_TILE_WIDTH: u32 = 4096;

/// `MAX_TILE_AREA` per §3 — maximum tile area in luma samples.
pub const MAX_TILE_AREA: u32 = 4096 * 2304;

/// `MAX_TILE_ROWS` per §3 — maximum number of tile rows in a frame.
pub const MAX_TILE_ROWS: u32 = 64;

/// `MAX_TILE_COLS` per §3 — maximum number of tile columns in a frame.
pub const MAX_TILE_COLS: u32 = 64;

// ---------------------------------------------------------------------
// §5.9.16 tile_log2 helper
// ---------------------------------------------------------------------

/// `tile_log2(blkSize, target)` per §5.9.16 — smallest `k` such that
/// `blkSize << k >= target`.
fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0u32;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

// ---------------------------------------------------------------------
// TileInfo (§6.8.14)
// ---------------------------------------------------------------------

/// Parsed `tile_info()` per §5.9.15.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileInfo {
    /// `uniform_tile_spacing_flag` per §6.8.14.
    pub uniform_tile_spacing_flag: bool,
    /// `TileCols` — number of tiles across the frame (`<= MAX_TILE_COLS`
    /// per §6.8.14 conformance).
    pub tile_cols: u32,
    /// `TileRows` — number of tiles down the frame (`<= MAX_TILE_ROWS`).
    pub tile_rows: u32,
    /// `TileColsLog2` — base-2 logarithm of the desired number of
    /// tiles across the frame. May exceed `log2(tile_cols)` when small
    /// frame sizes force the actual tile count to be smaller than the
    /// desired count (§6.8.14 note: "the tile size is rounded up to a
    /// multiple of the maximum superblock size").
    pub tile_cols_log2: u32,
    /// `TileRowsLog2` — base-2 logarithm of the desired number of
    /// tiles down the frame.
    pub tile_rows_log2: u32,
    /// `context_update_tile_id` per §6.8.14 — which tile's CDFs are
    /// snapshotted at the `disable_frame_end_update_cdf == 0` end-of-
    /// frame update. `0` when both `TileColsLog2 == 0 &&
    /// TileRowsLog2 == 0` (the `f(TileRowsLog2 + TileColsLog2)` read is
    /// skipped because the field has zero bit width).
    pub context_update_tile_id: u32,
    /// `TileSizeBytes` per §6.8.14 — the number of bytes used to code
    /// each tile size in the §5.11 tile group OBU. `1` when both
    /// `TileColsLog2 == 0 && TileRowsLog2 == 0` (the
    /// `tile_size_bytes_minus_1` field is skipped and the spec leaves
    /// `TileSizeBytes` undefined; we surface the no-read default of `1`
    /// — the value never matters because a single-tile frame has no
    /// inter-tile size fields).
    pub tile_size_bytes: u8,
    /// `MiColStarts[0..=TileCols]` per §6.8.14 — start column (in units
    /// of 4×4 luma samples) for each tile across the image plus the
    /// sentinel `MiColStarts[TileCols] == MiCols`.
    pub mi_col_starts: Vec<u32>,
    /// `MiRowStarts[0..=TileRows]` per §6.8.14 — start row for each
    /// tile down the image plus the sentinel `MiRowStarts[TileRows] ==
    /// MiRows`.
    pub mi_row_starts: Vec<u32>,
}

impl TileInfo {
    /// Convenience for the degenerate `tile_cols == 1 && tile_rows == 1`
    /// case — every fixture in `docs/video/av1/fixtures/` except
    /// `tile-cols-2-rows-1` lands here.
    pub fn is_single_tile(&self) -> bool {
        self.tile_cols == 1 && self.tile_rows == 1
    }
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

/// Parse `tile_info()` per §5.9.15 from a bit-aligned byte slice.
///
/// `mi_cols` / `mi_rows` are the §5.9.9 `compute_image_size()`-derived
/// values (carried on [`crate::FrameSize::mi_cols`] /
/// [`crate::FrameSize::mi_rows`]). `use_128x128_superblock` is the
/// sequence-header bit (§5.5.1).
///
/// Returns `(TileInfo, bits_consumed)`. The caller is responsible for
/// positioning `payload` at the right bit — the standalone entry point
/// reads from bit 0.
///
/// ## Errors
///
///   * [`Error::UnexpectedEnd`] — payload exhausted mid-read.
pub fn parse_tile_info(
    payload: &[u8],
    mi_cols: u32,
    mi_rows: u32,
    use_128x128_superblock: bool,
) -> Result<(TileInfo, usize), Error> {
    let mut br = BitReader::new(payload);
    let ti = read_tile_info(&mut br, mi_cols, mi_rows, use_128x128_superblock)?;
    Ok((ti, br.position()))
}

pub(crate) fn read_tile_info(
    br: &mut BitReader<'_>,
    mi_cols: u32,
    mi_rows: u32,
    use_128x128_superblock: bool,
) -> Result<TileInfo, Error> {
    // §5.9.15 lead-in derivations.
    let (sb_cols, sb_rows, sb_shift) = if use_128x128_superblock {
        ((mi_cols + 31) >> 5, (mi_rows + 31) >> 5, 5u32)
    } else {
        ((mi_cols + 15) >> 4, (mi_rows + 15) >> 4, 4u32)
    };
    let sb_size = sb_shift + 2;
    let max_tile_width_sb = MAX_TILE_WIDTH >> sb_size;
    let max_tile_area_sb = MAX_TILE_AREA >> (2 * sb_size);
    let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));

    let uniform_tile_spacing_flag = br.f(1)? == 1;

    let mut mi_col_starts: Vec<u32> = Vec::new();
    let mut mi_row_starts: Vec<u32> = Vec::new();
    let tile_cols;
    let tile_rows;
    let tile_cols_log2;
    let tile_rows_log2;

    if uniform_tile_spacing_flag {
        // §5.9.15 uniform path.
        let mut tile_cols_log2_local = min_log2_tile_cols;
        while tile_cols_log2_local < max_log2_tile_cols {
            let increment = br.f(1)? == 1;
            if increment {
                tile_cols_log2_local += 1;
            } else {
                break;
            }
        }
        let tile_width_sb = (sb_cols + (1u32 << tile_cols_log2_local) - 1) >> tile_cols_log2_local;
        let mut i: u32 = 0;
        let mut start_sb: u32 = 0;
        while start_sb < sb_cols {
            mi_col_starts.push(start_sb << sb_shift);
            i += 1;
            start_sb += tile_width_sb;
        }
        mi_col_starts.push(mi_cols);
        tile_cols = i;

        let min_log2_tile_rows = min_log2_tiles.saturating_sub(tile_cols_log2_local);
        let mut tile_rows_log2_local = min_log2_tile_rows;
        while tile_rows_log2_local < max_log2_tile_rows {
            let increment = br.f(1)? == 1;
            if increment {
                tile_rows_log2_local += 1;
            } else {
                break;
            }
        }
        let tile_height_sb = (sb_rows + (1u32 << tile_rows_log2_local) - 1) >> tile_rows_log2_local;
        let mut i: u32 = 0;
        let mut start_sb: u32 = 0;
        while start_sb < sb_rows {
            mi_row_starts.push(start_sb << sb_shift);
            i += 1;
            start_sb += tile_height_sb;
        }
        mi_row_starts.push(mi_rows);
        tile_rows = i;

        tile_cols_log2 = tile_cols_log2_local;
        tile_rows_log2 = tile_rows_log2_local;
    } else {
        // §5.9.15 non-uniform path.
        let mut widest_tile_sb: u32 = 0;
        let mut start_sb: u32 = 0;
        let mut i: u32 = 0;
        while start_sb < sb_cols {
            mi_col_starts.push(start_sb << sb_shift);
            let max_width = (sb_cols - start_sb).min(max_tile_width_sb);
            // §4.10.7 ns(maxWidth).
            let width_in_sbs_minus_1 = br.ns(max_width.max(1))?;
            let size_sb = width_in_sbs_minus_1 + 1;
            widest_tile_sb = widest_tile_sb.max(size_sb);
            start_sb += size_sb;
            i += 1;
        }
        mi_col_starts.push(mi_cols);
        tile_cols = i;
        let tile_cols_log2_local = tile_log2(1, tile_cols);

        // Recompute maxTileAreaSb per §5.9.15 (it's overwritten in
        // the non-uniform branch).
        let max_tile_area_sb_local = if min_log2_tiles > 0 {
            (sb_rows * sb_cols) >> (min_log2_tiles + 1)
        } else {
            sb_rows * sb_cols
        };
        let max_tile_height_sb = (max_tile_area_sb_local / widest_tile_sb).max(1);

        let mut start_sb: u32 = 0;
        let mut i: u32 = 0;
        while start_sb < sb_rows {
            mi_row_starts.push(start_sb << sb_shift);
            let max_height = (sb_rows - start_sb).min(max_tile_height_sb);
            let height_in_sbs_minus_1 = br.ns(max_height.max(1))?;
            let size_sb = height_in_sbs_minus_1 + 1;
            start_sb += size_sb;
            i += 1;
        }
        mi_row_starts.push(mi_rows);
        tile_rows = i;
        let tile_rows_log2_local = tile_log2(1, tile_rows);

        tile_cols_log2 = tile_cols_log2_local;
        tile_rows_log2 = tile_rows_log2_local;
    }

    // §5.9.15 trailing fields: context_update_tile_id +
    // tile_size_bytes_minus_1 only when at least one of the log2
    // counts is > 0.
    let (context_update_tile_id, tile_size_bytes) = if tile_cols_log2 > 0 || tile_rows_log2 > 0 {
        let n = tile_rows_log2 + tile_cols_log2;
        let id = br.f(n)? as u32;
        let tsb_minus_1 = br.f(2)? as u8;
        (id, tsb_minus_1 + 1)
    } else {
        (0, 1)
    };

    Ok(TileInfo {
        uniform_tile_spacing_flag,
        tile_cols,
        tile_rows,
        tile_cols_log2,
        tile_rows_log2,
        context_update_tile_id,
        tile_size_bytes,
        mi_col_starts,
        mi_row_starts,
    })
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_log2_smallest_k() {
        // tile_log2(1, 1) = 0  ( (1 << 0) >= 1 )
        assert_eq!(tile_log2(1, 1), 0);
        // tile_log2(1, 2) = 1
        assert_eq!(tile_log2(1, 2), 1);
        // tile_log2(1, 3) = 2  ( (1 << 1) = 2 < 3, (1 << 2) = 4 >= 3 )
        assert_eq!(tile_log2(1, 3), 2);
        // tile_log2(4, 8) = 1 ( (4 << 1) = 8 >= 8 )
        assert_eq!(tile_log2(4, 8), 1);
        // tile_log2(64, 4) = 0 ( (64 << 0) = 64 >= 4 )
        assert_eq!(tile_log2(64, 4), 0);
    }

    /// A 16×16 frame with `use_128x128_superblock = 0`: sbCols = 1,
    /// sbRows = 1 ⇒ `minLog2TileCols = 0`, `maxLog2TileCols = 0`,
    /// `maxLog2TileRows = 0`. The uniform-spacing path reads
    /// `uniform_tile_spacing_flag` and immediately falls through with
    /// `TileColsLog2 = 0`, `TileRowsLog2 = 0`. No
    /// `context_update_tile_id` / `tile_size_bytes_minus_1` is read.
    /// Total bits consumed: 1.
    #[test]
    fn tiny_16x16_single_tile() {
        // MiCols = MiRows = 4 (per round-4 derivation for 16×16).
        // bits: uniform_tile_spacing_flag = 1.  ⇒ first bit set.
        let payload = [0b1000_0000u8];
        let (ti, bits) = parse_tile_info(&payload, 4, 4, false).expect("parses");
        assert!(ti.uniform_tile_spacing_flag);
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(ti.tile_cols_log2, 0);
        assert_eq!(ti.tile_rows_log2, 0);
        assert_eq!(ti.context_update_tile_id, 0);
        assert_eq!(ti.tile_size_bytes, 1);
        assert_eq!(bits, 1);
        assert!(ti.is_single_tile());
        // MiColStarts = [0, MiCols], MiRowStarts = [0, MiRows].
        assert_eq!(ti.mi_col_starts, vec![0, 4]);
        assert_eq!(ti.mi_row_starts, vec![0, 4]);
    }

    /// A 256×64 frame with `use_128x128_superblock = 0` and
    /// `uniform_tile_spacing_flag = 1` matching the `tile-cols-2-rows-1`
    /// fixture's trace (`tile_cols = 2, tile_rows = 1`). Verifies the
    /// `increment_tile_cols_log2` bit walks the desired column count up
    /// to log2(2) = 1.
    #[test]
    fn synthetic_256x64_two_tile_columns() {
        // MiCols = 2 * ((256+7) >> 3) = 64, MiRows = 2 * ((64+7) >> 3) = 16.
        // sbCols = (64 + 15) >> 4 = 79 >> 4 = 4 (since 64 in MI units = 256
        //   luma samples; 79/16 = 4 with remainder, so 4 superblocks).
        // Actually (64+15)>>4 = 79>>4 = 4. sb_shift=4. sb_size=6.
        // maxTileWidthSb = 4096 >> 6 = 64. maxTileAreaSb = (4096*2304) >> 12 = 2304.
        // sbRows = (16+15) >> 4 = 1.
        // minLog2TileCols = tile_log2(64, 4) = 0.
        // maxLog2TileCols = tile_log2(1, min(4, 64)) = tile_log2(1, 4) = 2.
        // maxLog2TileRows = tile_log2(1, min(1, 64)) = 0.
        // minLog2Tiles = max(0, tile_log2(2304, 4)) = max(0, 0) = 0.
        //
        // Uniform path: TileColsLog2 starts at 0. While 0 < 2, read
        // increment. To reach TileColsLog2 = 1 we read 1 (increment),
        // then 0 (stop). 2 bits.
        // tileWidthSb = (4 + (1<<1) - 1) >> 1 = 5 >> 1 = 2. So:
        //   startSb=0 ⇒ MiColStarts[0] = 0; i=1; startSb=2.
        //   startSb=2 ⇒ MiColStarts[1] = 2<<4 = 32; i=2; startSb=4.
        //   startSb=4 stops. MiColStarts[2] = 64. TileCols = 2.
        // minLog2TileRows = max(0 - 1, 0) = 0. TileRowsLog2 = 0;
        // 0 < 0 is false. No row increments. tileHeightSb = 1.
        //   startSb=0 ⇒ MiRowStarts[0] = 0; i=1; startSb=1.
        //   startSb=1 stops. MiRowStarts[1] = 16. TileRows = 1.
        // TileColsLog2 = 1, TileRowsLog2 = 0 ⇒ read context_update_tile_id
        // as f(1) and tile_size_bytes_minus_1 as f(2). For tile_cols=2,
        // context_update_tile_id < 2 ⇒ legal values 0 or 1.
        //
        // Bits: uniform_tile_spacing=1; increment_cols=1; increment_cols_stop=0;
        //       context_update_tile_id(1)=0; tile_size_bytes_minus_1(2)=00
        //       (⇒ TileSizeBytes = 1).
        // Total: 1+1+1+1+2 = 6 bits. = 0b110_0_00_00 = 0xC0.
        let payload = [0b1100_0000u8];
        let (ti, bits) = parse_tile_info(&payload, 64, 16, false).expect("parses");
        assert!(ti.uniform_tile_spacing_flag);
        assert_eq!(ti.tile_cols, 2);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(ti.tile_cols_log2, 1);
        assert_eq!(ti.tile_rows_log2, 0);
        assert_eq!(ti.context_update_tile_id, 0);
        assert_eq!(ti.tile_size_bytes, 1);
        assert_eq!(bits, 6);
        // MiColStarts = [0, 32, 64], MiRowStarts = [0, 16].
        assert_eq!(ti.mi_col_starts, vec![0, 32, 64]);
        assert_eq!(ti.mi_row_starts, vec![0, 16]);
    }

    /// 64×64 with `use_128x128_superblock = 0`, uniform spacing.
    /// sbCols = sbRows = 1 ⇒ everything collapses to a single tile,
    /// just like 16×16. 1 bit consumed.
    #[test]
    fn frame_64x64_single_superblock() {
        // MiCols = MiRows = 16. sbCols = (16+15)>>4 = 1; sbRows similarly.
        // Only 1 bit consumed (uniform_tile_spacing).
        let payload = [0b1000_0000u8];
        let (ti, bits) = parse_tile_info(&payload, 16, 16, false).expect("parses");
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(bits, 1);
        assert_eq!(ti.mi_col_starts, vec![0, 16]);
        assert_eq!(ti.mi_row_starts, vec![0, 16]);
    }

    /// 128×128 superblock mode: ensure the `sb_shift = 5` path is
    /// exercised. With a 128×128 frame and use_128x128_superblock = 1:
    /// MiCols = MiRows = 32. sbCols = (32+31)>>5 = 1.
    #[test]
    fn frame_128x128_with_128_superblock() {
        let payload = [0b1000_0000u8];
        let (ti, bits) = parse_tile_info(&payload, 32, 32, true).expect("parses");
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(bits, 1);
        // sb_shift = 5 ⇒ mi_col_starts[0] = 0 (single tile).
        assert_eq!(ti.mi_col_starts, vec![0, 32]);
        assert_eq!(ti.mi_row_starts, vec![0, 32]);
    }

    /// Non-uniform tile spacing. Construct a 2-column layout on a
    /// 256-luma-wide frame (MiCols = 64, sbCols = 4), with
    /// `width_in_sbs_minus_1` values 1 (⇒ tile #0 = 2 SBs) then 1
    /// (⇒ tile #1 = 2 SBs), then a single-row layout
    /// `height_in_sbs_minus_1 = 0` (⇒ tile = 1 SB).
    ///
    /// `max_width` on the first iteration = min(4, 64) = 4 ⇒ ns(4) ⇒
    /// `value = 1` encoded as `01` (2 bits).
    /// Second iteration `max_width = min(4-2, 64) = 2` ⇒ ns(2):
    /// w = 2, m = (1 << 2) - 2 = 2. v = f(1). Possible v in {0, 1};
    /// v < m for both ⇒ no extra bit. v = 1 encodes value 1, 1 bit.
    /// Loop stops at start_sb=4=sb_cols.
    /// TileCols = 2. TileColsLog2 = tile_log2(1, 2) = 1.
    /// max_tile_area_sb_local: min_log2_tiles is recomputed via local
    /// = 0 here (sbCols=4, sbRows=1 ⇒ sbRows*sbCols=4; min_log2_tiles is
    /// computed early as max(minLog2TileCols, tile_log2(maxTileAreaSb,
    /// sbRows*sbCols)) = max(0, tile_log2(2304, 4)) = 0; since
    /// min_log2_tiles == 0, max_tile_area_sb_local = 4).
    /// widest_tile_sb = 2. max_tile_height_sb = max(4/2, 1) = 2.
    /// Row loop: max_height = min(1-0, 2) = 1 ⇒ ns(1) reads 0 bits and
    /// returns 0. size_sb = 1. start_sb = 1. Loop exits.
    /// TileRows = 1. TileRowsLog2 = tile_log2(1, 1) = 0.
    /// Trailing: TileColsLog2=1, TileRowsLog2=0 ⇒ read context(1) +
    /// tile_size_bytes_minus_1(2).
    /// Bit string: uniform=0, col0 ns=01, col1 ns=1, [row: 0 bits],
    /// context(1)=0, tsb(2)=00.
    /// = 0 01 1 0 00 = 7 bits = 0b0011_0000 = 0x30.
    #[test]
    fn non_uniform_two_tile_columns() {
        let payload = [0b0011_0000u8];
        let (ti, bits) = parse_tile_info(&payload, 64, 16, false).expect("parses");
        assert!(!ti.uniform_tile_spacing_flag);
        assert_eq!(ti.tile_cols, 2);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(ti.tile_cols_log2, 1);
        assert_eq!(ti.tile_rows_log2, 0);
        assert_eq!(ti.context_update_tile_id, 0);
        assert_eq!(ti.tile_size_bytes, 1);
        assert_eq!(bits, 7);
        // sb_shift = 4. First tile starts at 0 ⇒ mi 0; second starts at
        // 2 SB = 2 << 4 = 32; sentinel = 64.
        assert_eq!(ti.mi_col_starts, vec![0, 32, 64]);
        assert_eq!(ti.mi_row_starts, vec![0, 16]);
    }

    #[test]
    fn truncated_payload_returns_unexpected_end() {
        let err = parse_tile_info(&[], 4, 4, false).expect_err("empty payload");
        assert_eq!(err, Error::UnexpectedEnd);
    }
}
