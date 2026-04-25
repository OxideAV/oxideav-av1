//! Mutable per-frame state for the AV1 mode decoder.
//!
//! For Phase 2 (mode decode only) this struct omits reconstructed
//! pixel planes and CDEF signalling and keeps only the MI grid plus
//! the subsampling / bit-depth context needed by the chroma code paths.

use crate::lr::UnitParams as LrUnitParams;
use crate::transform::TxSize;

use super::modes::IntraMode;

/// Per-MI-unit (4×4 block) decoded mode information. Values are
/// populated by the mode decoder in [`super::superblock`] and left
/// untouched by Phase 2 for blocks the bitstream marks as skip — for
/// non-skip blocks the decoder returns `Error::Unsupported` pointing
/// at §5.11.39 coefficient decode before writing anything further.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModeInfo {
    /// Y-plane intra mode (spec `y_mode`).
    pub mode: Option<IntraMode>,
    /// UV-plane mode (spec `uv_mode`). `None` for monochrome frames.
    pub uv_mode: Option<IntraMode>,
    /// Skip flag — `true` means the residual for this block is all
    /// zero (prediction only output). §5.11.13.
    pub skip: bool,
    /// `segment_id` — 0..=7 per §5.11.9. Left at 0 when segmentation
    /// is disabled.
    pub segment_id: u8,
    /// `angle_delta` for the directional modes (`D45..D67`). Range
    /// `-3..=3`; 0 otherwise.
    pub angle_delta: i8,
    /// `angle_delta_uv` for chroma directional modes.
    pub angle_delta_uv: i8,
    /// `cfl_alpha_u` — signed CFL alpha magnitude for U. 0 when CFL
    /// is not active.
    pub cfl_alpha_u: i32,
    /// `cfl_alpha_v` — signed CFL alpha magnitude for V.
    pub cfl_alpha_v: i32,
    /// `dc_sign_u` / `dc_sign_v` — retained for Phase 3 coefficient
    /// decode; left at 0 in Phase 2.
    pub dc_sign_u: i8,
    pub dc_sign_v: i8,
    /// `txb_skip` — retained for Phase 3. All zero in Phase 2.
    pub txb_skip: bool,
    /// `is_inter` — marks whether the block used inter prediction.
    /// Used by the inter decoder to form the above/left context for
    /// the `is_inter` CDF.
    pub is_inter: bool,
    /// Per-block MV (eighth-pel units). Only meaningful when
    /// `is_inter` is true; `(0, 0)` otherwise.
    pub mv_row: i32,
    pub mv_col: i32,
    /// Per-MI transform size chosen for this block — spec's
    /// `InterTxSizes[row][col]`. Stored so neighbouring blocks can
    /// derive the `tx_depth` context (§5.11.16 / §9.4.8). `None` until
    /// the block's `read_block_tx_size()` has run.
    pub tx_size: Option<TxSize>,
    /// Per-MI block size (spec `MiSizes[row][col]`), also needed for
    /// `tx_depth`'s inter-block context derivation.
    pub mi_size_idx: u8,
    /// Spec §5.11.24 `use_filter_intra` — `true` when the recursive
    /// filter-intra path (§7.11.2.3) is in play. Only meaningful on
    /// intra blocks predicted with `DC_PRED` on sizes ≤ 32×32.
    pub use_filter_intra: bool,
    /// Spec §5.11.24 `filter_intra_mode`. One of 0..=4
    /// (`FILTER_DC_PRED..FILTER_PAETH_PRED`). Unused when
    /// `use_filter_intra` is `false`.
    pub filter_intra_mode: u8,
    /// Spec §5.11.46 `PaletteSizeY` — number of palette colors on the
    /// luma plane (0 when the block isn't palette-coded). Held on the
    /// MI grid so neighbour blocks can form the `has_palette_y`
    /// context (§9.4.6) and pull the colours into their cache
    /// (§5.11.46 `get_palette_cache`).
    pub palette_size_y: u8,
    /// Spec §5.11.46 `PaletteSizeUV`. Same treatment as
    /// `palette_size_y`.
    pub palette_size_uv: u8,
    /// Spec §5.11.46 `PaletteColors[0][...]` — Y palette colours,
    /// sorted ascending. First `palette_size_y` entries valid; the
    /// rest are zero-padded.
    pub palette_colors_y: [u16; 8],
    /// Spec §5.11.46 `PaletteColors[1][...]` — U palette colours,
    /// sorted ascending. First `palette_size_uv` entries valid.
    pub palette_colors_u: [u16; 8],
    /// V palette colours — emitted alongside U by `palette_mode_info`
    /// but kept in the encoder's order (the spec's
    /// `delta_encode_palette_colors_v` path may emit them in a
    /// non-sorted sequence). First `palette_size_uv` entries valid.
    pub palette_colors_v: [u16; 8],
}

/// Mutable per-frame decoder state.
///
/// Carries the MI grid dimensions + per-MI-unit mode info plus
/// reconstructed Y/U/V plane buffers. Plane strides equal the plane
/// widths (no left/right padding).
pub struct FrameState {
    pub width: u32,
    pub height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    /// MI grid indexed as `mi[mi_row * mi_cols + mi_col]`.
    pub mi: Vec<ModeInfo>,
    /// Chroma subsampling factors (0 or 1 each). `sub_x = 1, sub_y =
    /// 1` is 4:2:0.
    pub sub_x: u32,
    pub sub_y: u32,
    /// `true` iff the sequence header set `monochrome = 1`.
    pub monochrome: bool,
    /// Sample bit depth — 8, 10, or 12.
    pub bit_depth: u32,
    /// Luma plane — row-major `width * height` bytes (8-bit depth
    /// only).
    pub y_plane: Vec<u8>,
    /// U plane — row-major `uv_width * uv_height` bytes.
    pub u_plane: Vec<u8>,
    /// V plane — row-major `uv_width * uv_height` bytes.
    pub v_plane: Vec<u8>,
    /// 10/12-bit luma plane — row-major `width * height` samples.
    /// Zero-length when `bit_depth == 8`.
    pub y_plane16: Vec<u16>,
    /// 10/12-bit U plane.
    pub u_plane16: Vec<u16>,
    /// 10/12-bit V plane.
    pub v_plane16: Vec<u16>,
    /// UV plane width after subsampling.
    pub uv_width: u32,
    /// UV plane height after subsampling.
    pub uv_height: u32,

    /// Per-restoration-unit LR parameters, indexed as
    /// `lr_unit_info[plane][row * lr_cols[plane] + col]` (§5.11.40-.44).
    /// Each inner `Vec` has `lr_cols[plane] * lr_rows[plane]` entries.
    pub lr_unit_info: [Vec<LrUnitParams>; 3],
    /// Number of restoration-unit columns per plane.
    pub lr_cols: [u32; 3],
    /// Number of restoration-unit rows per plane.
    pub lr_rows: [u32; 3],
    /// Per-plane restoration unit size in luma-plane samples (or
    /// chroma-plane samples on planes 1/2 if subsampled).
    pub lr_unit_size: [u32; 3],

    /// Per-64×64-luma-SB `cdef_idx`, indexed as
    /// `cdef_idx[sb_row * cdef_sb_cols + sb_col]`. `-1` means no CDEF
    /// filtering applied to this SB (spec §5.11.55 / §5.11.56).
    /// Initialised to `-1` on frame setup; stamped in by the leaf-block
    /// walker on the first non-skip block in each 64×64 region.
    pub cdef_idx: Vec<i8>,
    /// Number of 64×64-SB columns — `(width + 63) >> 6`.
    pub cdef_sb_cols: u32,
    /// Number of 64×64-SB rows — `(height + 63) >> 6`.
    pub cdef_sb_rows: u32,
}

impl FrameState {
    /// Allocate a blank per-frame state sized for the given frame
    /// dimensions + chroma subsampling.
    pub fn new(width: u32, height: u32, sub_x: u32, sub_y: u32, monochrome: bool) -> Self {
        Self::with_bit_depth(width, height, sub_x, sub_y, monochrome, 8)
    }

    /// Like [`FrameState::new`] but sets the sample bit depth
    /// explicitly. Out-of-range values are clamped to 8.
    pub fn with_bit_depth(
        width: u32,
        height: u32,
        sub_x: u32,
        sub_y: u32,
        monochrome: bool,
        bit_depth: u32,
    ) -> Self {
        let bit_depth = match bit_depth {
            8 | 10 | 12 => bit_depth,
            _ => 8,
        };
        let mi_cols = (width + 3) >> 2;
        let mi_rows = (height + 3) >> 2;
        let mi_len = (mi_cols as usize) * (mi_rows as usize);
        let uv_width = if monochrome {
            0
        } else {
            (width + sub_x) >> sub_x
        };
        let uv_height = if monochrome {
            0
        } else {
            (height + sub_y) >> sub_y
        };
        let use_hbd = bit_depth > 8;
        let y_len = (width as usize) * (height as usize);
        let uv_len = (uv_width as usize) * (uv_height as usize);
        // Start planes at mid-grey for the given bit depth so fallback
        // intra prediction has a sensible floor. 8-bit uses 128; the
        // HBD midpoint is 128 << (bd-8).
        let hbd_mid: u16 = 1u16 << (bit_depth - 1);
        Self {
            width,
            height,
            mi_cols,
            mi_rows,
            mi: vec![ModeInfo::default(); mi_len],
            sub_x,
            sub_y,
            monochrome,
            bit_depth,
            y_plane: if use_hbd {
                Vec::new()
            } else {
                vec![0u8; y_len]
            },
            u_plane: if use_hbd || monochrome {
                Vec::new()
            } else {
                vec![0u8; uv_len]
            },
            v_plane: if use_hbd || monochrome {
                Vec::new()
            } else {
                vec![0u8; uv_len]
            },
            y_plane16: if use_hbd {
                vec![hbd_mid; y_len]
            } else {
                Vec::new()
            },
            u_plane16: if use_hbd && !monochrome {
                vec![hbd_mid; uv_len]
            } else {
                Vec::new()
            },
            v_plane16: if use_hbd && !monochrome {
                vec![hbd_mid; uv_len]
            } else {
                Vec::new()
            },
            uv_width,
            uv_height,
            lr_unit_info: [Vec::new(), Vec::new(), Vec::new()],
            lr_cols: [0, 0, 0],
            lr_rows: [0, 0, 0],
            lr_unit_size: [0, 0, 0],
            cdef_idx: {
                let cols = (width + 63) >> 6;
                let rows = (height + 63) >> 6;
                vec![-1i8; (cols as usize) * (rows as usize)]
            },
            cdef_sb_cols: (width + 63) >> 6,
            cdef_sb_rows: (height + 63) >> 6,
        }
    }

    /// Mutable access to the per-64×64 SB `cdef_idx` entry covering
    /// the luma-sample `(x, y)`. Out-of-range coords clip to the
    /// nearest SB.
    pub fn cdef_idx_mut(&mut self, x: u32, y: u32) -> &mut i8 {
        let cols = self.cdef_sb_cols.max(1) as usize;
        let rows = self.cdef_sb_rows.max(1) as usize;
        let sb_col = ((x >> 6) as usize).min(cols - 1);
        let sb_row = ((y >> 6) as usize).min(rows - 1);
        &mut self.cdef_idx[sb_row * cols + sb_col]
    }

    /// Immutable counterpart of [`Self::cdef_idx_mut`].
    pub fn cdef_idx_at(&self, sb_col: u32, sb_row: u32) -> i8 {
        let cols = self.cdef_sb_cols.max(1) as usize;
        let rows = self.cdef_sb_rows.max(1) as usize;
        let sb_col = (sb_col as usize).min(cols - 1);
        let sb_row = (sb_row as usize).min(rows - 1);
        self.cdef_idx[sb_row * cols + sb_col]
    }

    /// Allocate per-plane `lr_unit_info` storage sized for `cols × rows`
    /// restoration units. Entries are initialised to `FilterType::None`
    /// so unsignalled units pass through unchanged. Matches libaom's
    /// `av1_alloc_restoration_struct`.
    pub fn alloc_lr_units(&mut self, plane: usize, unit_size_samples: u32, cols: u32, rows: u32) {
        self.lr_cols[plane] = cols;
        self.lr_rows[plane] = rows;
        self.lr_unit_size[plane] = unit_size_samples;
        self.lr_unit_info[plane] = vec![LrUnitParams::default(); (cols as usize) * (rows as usize)];
    }

    /// Index into the per-plane `lr_unit_info` table. Out-of-range
    /// coordinates clip to the last entry in that plane.
    pub fn lr_unit_mut(&mut self, plane: usize, col: u32, row: u32) -> &mut LrUnitParams {
        let cols = self.lr_cols[plane].max(1) as usize;
        let rows = self.lr_rows[plane].max(1) as usize;
        let col = (col as usize).min(cols - 1);
        let row = (row as usize).min(rows - 1);
        &mut self.lr_unit_info[plane][row * cols + col]
    }

    /// Immutable counterpart of [`FrameState::lr_unit_mut`].
    pub fn lr_unit_at(&self, plane: usize, col: u32, row: u32) -> &LrUnitParams {
        let cols = self.lr_cols[plane].max(1) as usize;
        let rows = self.lr_rows[plane].max(1) as usize;
        let col = (col as usize).min(cols - 1);
        let row = (row as usize).min(rows - 1);
        &self.lr_unit_info[plane][row * cols + col]
    }

    /// Mutable access to the ModeInfo at `(mi_col, mi_row)`. Panics on
    /// out-of-bounds indices — the decoder must clip before calling.
    pub fn mi_mut(&mut self, mi_col: u32, mi_row: u32) -> &mut ModeInfo {
        let idx = (mi_row as usize) * (self.mi_cols as usize) + (mi_col as usize);
        &mut self.mi[idx]
    }

    /// Immutable access to the ModeInfo at `(mi_col, mi_row)`.
    pub fn mi_at(&self, mi_col: u32, mi_row: u32) -> &ModeInfo {
        let idx = (mi_row as usize) * (self.mi_cols as usize) + (mi_col as usize);
        &self.mi[idx]
    }

    /// Spec §5.11.46 helper — return the palette colour list stored
    /// at MI cell `(mi_col, mi_row)` for `plane` (0 = Y, 1 = UV).
    /// Length matches the MI's recorded palette size; an empty slice
    /// means the cell isn't palette-coded.
    pub fn palette_colors_at(&self, plane: usize, mi_col: u32, mi_row: u32) -> &[u16] {
        let mi = self.mi_at(mi_col, mi_row);
        match plane {
            0 => {
                let n = mi.palette_size_y as usize;
                &mi.palette_colors_y[..n.min(8)]
            }
            _ => {
                let n = mi.palette_size_uv as usize;
                &mi.palette_colors_u[..n.min(8)]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_allocates_mi_grid_for_64x64_420() {
        let fs = FrameState::new(64, 64, 1, 1, false);
        assert_eq!(fs.mi_cols, 16);
        assert_eq!(fs.mi_rows, 16);
        assert_eq!(fs.mi.len(), 256);
        assert_eq!(fs.bit_depth, 8);
    }

    #[test]
    fn with_bit_depth_clamps_invalid() {
        let fs = FrameState::with_bit_depth(64, 64, 1, 1, false, 9);
        assert_eq!(fs.bit_depth, 8);
        let fs = FrameState::with_bit_depth(64, 64, 1, 1, false, 10);
        assert_eq!(fs.bit_depth, 10);
    }

    #[test]
    fn mi_mut_round_trips() {
        let mut fs = FrameState::new(32, 32, 1, 1, false);
        fs.mi_mut(3, 5).skip = true;
        fs.mi_mut(3, 5).segment_id = 2;
        assert!(fs.mi_at(3, 5).skip);
        assert_eq!(fs.mi_at(3, 5).segment_id, 2);
    }
}
