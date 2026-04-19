//! Mutable per-frame state for the AV1 mode decoder.
//!
//! Adapted from the `FrameState` struct in
//! `github.com/KarpelesLab/goavif/av1/decoder/superblock.go` (MIT,
//! KarpelesLab/goavif). The goavif version also carries reconstructed
//! pixel planes + CDEF signalling; for Phase 2 (mode decode only) we
//! drop the pixel buffers and keep only the MI grid plus the
//! subsampling / bit-depth context needed by the chroma code paths.

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
/// Carries the MI grid dimensions + per-MI-unit mode info. The shape
/// is intentionally smaller than goavif's `FrameState`: there are no
/// Y/U/V pixel planes and no CDEF index grid because Phase 2 stops
/// before any pixel is reconstructed.
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
        }
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
