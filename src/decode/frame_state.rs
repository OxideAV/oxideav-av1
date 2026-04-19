//! Mutable per-frame state for the AV1 mode decoder.
//!
//! Adapted from the `FrameState` struct in
//! `github.com/KarpelesLab/goavif/av1/decoder/superblock.go` (MIT,
//! KarpelesLab/goavif). The goavif version also carries reconstructed
//! pixel planes + CDEF signalling; for Phase 2 (mode decode only) we
//! drop the pixel buffers and keep only the MI grid plus the
//! subsampling / bit-depth context needed by the chroma code paths.

use crate::lr::UnitParams as LrUnitParams;

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
        }
    }

    /// Allocate per-plane `lr_unit_info` storage sized for `cols × rows`
    /// restoration units. Entries are initialised to `FilterType::None`
    /// so unsignalled units pass through unchanged. Matches libaom's
    /// `av1_alloc_restoration_struct`.
    pub fn alloc_lr_units(
        &mut self,
        plane: usize,
        unit_size_samples: u32,
        cols: u32,
        rows: u32,
    ) {
        self.lr_cols[plane] = cols;
        self.lr_rows[plane] = rows;
        self.lr_unit_size[plane] = unit_size_samples;
        self.lr_unit_info[plane] =
            vec![LrUnitParams::default(); (cols as usize) * (rows as usize)];
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
