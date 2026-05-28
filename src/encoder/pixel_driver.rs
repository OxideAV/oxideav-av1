//! Pixel-space encoder driver — arc 15 (round 221).
//!
//! The first end-to-end **pixel-in / bytes-out** entry point. Composes
//! every encoder primitive landed between r206–r220 into a single call
//! that takes a frame's luma samples and returns the IVF bytes the
//! decoder pipeline can walk back:
//!
//!   * §7.11.2.5 DC_PRED neighbour-derived prediction
//!     ([`crate::cdf::predict_intra_dc_pred`]).
//!   * Pixel residual = input − prediction.
//!   * Forward 4×4 DCT primitive ([`crate::encoder::forward_dct_4x4`],
//!     r219) — pixel residual → DCT-II coefficients.
//!   * Forward quantization ([`crate::encoder::forward_quantize`], r220)
//!     — coefficients → `Quant[]` levels for the §5.11.39 writers.
//!   * §5.11.4 recursive partition writer
//!     ([`crate::encoder::write_partition_tree`], r217) — emits
//!     PARTITION_NONE leaves at BLOCK_4X4 per 4×4 cell of the frame.
//!   * §5.11.39 coefficient writer
//!     ([`crate::encoder::write_coefficients`], r212–r215) — one Y-plane
//!     transform block per leaf.
//!   * §5.11.1 tile-group OBU framing
//!     ([`crate::encoder::write_tile_group_obu`], r210) — wraps the
//!     entropy bytes plus the byte-aligned tile-group header.
//!   * §7.5 temporal unit aggregation
//!     ([`crate::encoder::encode_temporal_unit`], r208) — TD + optional
//!     SH + per-frame OBU sequence (FrameHeader OBU + TileGroup OBU).
//!   * IVF v0 container ([`crate::encoder::IvfWriter`], r206).
//!
//! ## Scope (arc 15)
//!
//! Hard scope per the implementer round: a single 16×16 **monochrome**
//! intra-only frame at `base_q_idx = 0`, BLOCK_4X4 leaves, TX_4X4
//! DCT_DCT, default scan, no segmentation, no QM, no chroma. The
//! 16-cell luma is fed in row-major as `[[u8; 16]; 16]`.
//!
//! The encoder builds its own DC_PRED prediction by walking the
//! 16 BLOCK_4X4 leaves in §5.11.4 dispatch order (NW, NE, SW, SE
//! recursion ⇒ the leaves at `(r, c)` are visited in the order
//! `(0,0), (0,1), (1,0), (1,1), (0,2), (0,3), …`). At each leaf the
//! prediction is computed from the **already-reconstructed** neighbour
//! pixels — exactly what the decoder would observe on its parallel
//! walk. The reconstruction the encoder uses is the same
//! `inverse_transform_2d(dequantize_step1(Quant))` chain the decoder
//! runs at this leaf; this is the encoder's correctness contract.
//!
//! ## Verification
//!
//! Internal pixel-roundtrip helpers in [`internal_roundtrip`] exercise
//! the encoder-side inverse chain (dequantize + inverse transform + add
//! prediction) and surface the recovered pixels alongside the encoded
//! bytes. Integration tests assert:
//!
//!   1. The IVF bytes parse back as TD + SH + FrameHeader OBU + TileGroup
//!      OBU through [`crate::obu::ObuIter`] (structural roundtrip).
//!   2. The encoder-internal reconstructed pixels equal the input when
//!      the input pixels match the running DC_PRED prediction at every
//!      block — i.e. for the uniform mid-grey input (every sample =
//!      128). At `base_q_idx = 0`, DCT_DCT, DC_PRED on a flat-128
//!      input ⇒ zero residual ⇒ zero coefficients ⇒ zero `Quant[]` ⇒
//!      zero recovered residual ⇒ recovered pixels = 128 (bit-exact).
//!   3. For non-flat inputs the encoder-internal recovered pixels are
//!      within one quantization step of the input — the §7.12.3 step-1
//!      truncating divide bounds the per-coefficient error by one
//!      lattice step.
//!
//! ## Arc 17 (round 223) — chroma path
//!
//! [`encode_intra_frame_yuv`] extends the driver to a full 4:2:0
//! `Y[16x16] + U[8x8] + V[8x8]` input. The tiny fixture's sequence
//! header is already `monochrome = false`, `subsampling_x = 1`,
//! `subsampling_y = 1`, so the same `SequenceHeader` / `FrameHeader`
//! pair feeds both entry points — only the per-leaf coefficient stream
//! gains a chroma surface. Per §5.11.5 `HasChroma` at 4:2:0 with
//! `bw4 == bh4 == 1`, chroma fires only on leaves whose `MiRow` and
//! `MiCol` are both odd ⇒ the four leaves at cells `(1,1)`, `(1,3)`,
//! `(3,1)`, `(3,3)` carry the chroma coefficients (one chroma 4×4 per
//! 8×8 luma quadrant). Chroma uses the same DC_PRED + forward WHT +
//! forward quantize + dequantize + inverse WHT chain Y uses, and the
//! reconstructed chroma planes are returned alongside the luma plane
//! in [`EncodedFrameYuv`].
//!
//! ## What this arc does NOT do
//!
//!   * Drive the existing decoder back through `parse_obu` →
//!     PartitionWalker → coefficients → inverse transform → pixels. The
//!     workspace decoder entry [`crate::decode_av1`] is still a stub;
//!     reaching pixel roundtrip via the decoder is a separate arc.
//!   * Anything beyond TX_4X4 DCT_DCT. Forward transforms for larger
//!     sizes / non-DCT kernels are subsequent arcs.
//!   * Non-`DC_PRED` chroma modes. r223 always picks `uv_mode = DC_PRED`
//!     (matches the luma side). CFL / V_PRED / etc. are future arcs.
//!
//! ## Spec provenance
//!
//! `docs/video/av1/av1-spec.txt`:
//!   * §5.11.39 — `coefficients()` (p.88–93).
//!   * §5.11.4  — `decode_partition` (p.61–62).
//!   * §5.11.5  — `decode_block` (p.62).
//!   * §7.11.2.5 — DC_PRED sample generation (p.249).
//!   * §7.12.3   — Reconstruction (`dequantize_step1`, p.293–295).
//!   * §7.13.3   — Inverse transform 2D (p.305–307).

use crate::cdf::{
    dequantize_step1, predict_intra_d_mode, predict_intra_dc_pred, predict_intra_h_pred,
    predict_intra_paeth_pred, predict_intra_smooth_h_pred, predict_intra_smooth_pred,
    predict_intra_smooth_v_pred, predict_intra_v_pred, QuantizerParams, TileCdfContext,
    TileGeometry, BLOCK_16X16, D113_PRED, D135_PRED, D157_PRED, D203_PRED, D45_PRED, D67_PRED,
    DCT_DCT, DC_PRED, H_PRED, PAETH_PRED, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED, TX_4X4,
    TX_CLASS_2D, V_PRED,
};
use crate::encoder::forward_quantize::forward_quantize;
use crate::encoder::forward_transform::forward_dct_4x4;
use crate::encoder::forward_wht::forward_wht_4x4;
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::obu::ObuFrame;
use crate::encoder::partition_tree::{
    write_partition_tree, EncodeBlock, EncodeNode, PartitionTreeWriter, PlaneCoefficients,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::temporal_unit::{encode_temporal_unit, TemporalUnitPlan};
use crate::encoder::tile_group_obu::{write_tile_group_obu, TileGroupObu, TilePayload};
use crate::frame_header::FrameHeader;
use crate::obu::ObuType;
use crate::scan::get_default_scan;
use crate::sequence_header::SequenceHeader;
use crate::transform::inverse_transform_2d;
use crate::Error;

/// Frame extent the driver supports this arc: a 16×16 monochrome
/// luma surface (matches the
/// `docs/video/av1/fixtures/tiny-i-only-16x16-prof0` sequence header
/// already used by the IVF round-trip test).
pub const FRAME_WIDTH: usize = 16;
/// Frame height — see [`FRAME_WIDTH`].
pub const FRAME_HEIGHT: usize = 16;
/// `MiCols` derived per §5.9.9: `2 * ((16 + 7) >> 3) = 4`.
pub const MI_COLS: u32 = 4;
/// `MiRows` derived per §5.9.9: `2 * ((16 + 7) >> 3) = 4`.
pub const MI_ROWS: u32 = 4;
/// Number of 4×4 cells across the luma plane: `FRAME_WIDTH / 4 = 4`.
pub const CELLS_WIDE: usize = FRAME_WIDTH / 4;
/// Number of 4×4 cells down the luma plane.
pub const CELLS_HIGH: usize = FRAME_HEIGHT / 4;
/// Chroma-plane width at 4:2:0 (`FRAME_WIDTH >> subsampling_x = 16 >> 1`).
pub const CHROMA_WIDTH: usize = FRAME_WIDTH / 2;
/// Chroma-plane height at 4:2:0 (`FRAME_HEIGHT >> subsampling_y = 16 >> 1`).
pub const CHROMA_HEIGHT: usize = FRAME_HEIGHT / 2;
/// Number of 4×4 chroma cells across one chroma plane: `CHROMA_WIDTH / 4`.
pub const CHROMA_CELLS_WIDE: usize = CHROMA_WIDTH / 4;
/// Number of 4×4 chroma cells down one chroma plane.
pub const CHROMA_CELLS_HIGH: usize = CHROMA_HEIGHT / 4;

/// Result of [`encode_intra_frame_y`] — bundles the IVF bytes with the
/// encoder's internal reconstructed pixels (for the verification path).
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// IVF bytes — file header + one IVF frame with the §7.5 temporal
    /// unit body (TD + optional SH + FrameHeader OBU + TileGroup OBU).
    pub ivf_bytes: Vec<u8>,
    /// §7.5 temporal unit bytes (the IVF frame payload).
    pub temporal_unit_bytes: Vec<u8>,
    /// Encoder-internal **reconstructed pixels** the decoder would
    /// produce on a parallel walk. Bit-exact when the input matches the
    /// running DC_PRED prediction at every block; within one quantizer
    /// step otherwise.
    pub reconstructed_y: [[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    /// Per-block-position `Quant[]` arrays the encoder committed to,
    /// indexed in §5.11.4 dispatch order (NW, NE, SW, SE recursion ⇒
    /// `[(0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3),
    ///  (2,0), (2,1), (3,0), (3,1), (2,2), (2,3), (3,2), (3,3)]` in
    ///  (cell_row, cell_col) coordinates).
    pub committed_quants: Vec<Vec<i32>>,
    /// Per-block-position §6.10.x Y intra mode the encoder selected for
    /// each leaf, indexed in the same §5.11.4 dispatch order as
    /// `committed_quants`. Each value is in `0..NUM_INTRA_MODES_Y`
    /// (`DC_PRED = 0`, `V_PRED = 1`, …, `PAETH_PRED = 12`). Added in
    /// arc r228 alongside the 13-mode intra picker.
    pub committed_y_modes: Vec<u8>,
}

/// 4:2:0 YUV input the chroma driver consumes: 16×16 luma + two 8×8
/// chroma planes (`subsampling_x = 1`, `subsampling_y = 1`). Matches the
/// `tiny-i-only-16x16-prof0` fixture's color config.
#[derive(Debug, Clone, Copy)]
pub struct Yuv420Frame16x16 {
    /// Luma plane (`Y`). Row-major `[row][col]`.
    pub y: [[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    /// First chroma plane (`U` / `Cb`) at half horizontal+vertical
    /// resolution.
    pub u: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    /// Second chroma plane (`V` / `Cr`).
    pub v: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
}

impl Default for Yuv420Frame16x16 {
    /// All-128 mid-grey input — the "no neighbour" DC_PRED equilibrium
    /// on every plane.
    fn default() -> Self {
        Self {
            y: [[128u8; FRAME_WIDTH]; FRAME_HEIGHT],
            u: [[128u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
            v: [[128u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        }
    }
}

/// Result of [`encode_intra_frame_yuv`] — the chroma-aware sibling of
/// [`EncodedFrame`]. Bundles the IVF bytes with all three reconstructed
/// planes plus per-block `Quant[]` arrays for both the luma walk and
/// (separately) the chroma walk.
#[derive(Debug, Clone)]
pub struct EncodedFrameYuv {
    /// IVF bytes — file header + one IVF frame.
    pub ivf_bytes: Vec<u8>,
    /// §7.5 temporal unit bytes.
    pub temporal_unit_bytes: Vec<u8>,
    /// Reconstructed luma plane (encoder-internal walk).
    pub reconstructed_y: [[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    /// Reconstructed U plane.
    pub reconstructed_u: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    /// Reconstructed V plane.
    pub reconstructed_v: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    /// Per-luma-cell committed `Quant[]` in §5.11.4 dispatch order; same
    /// shape + meaning as [`EncodedFrame::committed_quants`].
    pub committed_quants_y: Vec<Vec<i32>>,
    /// Per-luma-cell §6.10.x Y intra mode the encoder selected; same
    /// shape + meaning as [`EncodedFrame::committed_y_modes`]. Added in
    /// arc r228 alongside the 13-mode intra picker.
    pub committed_y_modes: Vec<u8>,
    /// Per-chroma-cell U-plane `Quant[]` in chroma-dispatch order:
    /// `[(0,0), (0,1), (1,0), (1,1)]` covering the four chroma 4×4
    /// blocks (one per 8×8 luma quadrant, surfaced in the §5.11.4 cell
    /// order that visits the carrying mi `(1,1), (1,3), (3,1), (3,3)`).
    pub committed_quants_u: Vec<Vec<i32>>,
    /// Per-chroma-cell V-plane `Quant[]` — same indexing as
    /// `committed_quants_u`.
    pub committed_quants_v: Vec<Vec<i32>>,
}

/// Single 4×4 cell of the luma plane the encoder visits per leaf. Used
/// inside the per-block driver; exported so callers can predict the
/// dispatch order through [`dispatch_order_cells`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellCoord {
    /// Cell row index in `0..CELLS_HIGH`. Multiply by `4` to get the
    /// pixel-row index in the luma plane.
    pub cell_row: usize,
    /// Cell column index in `0..CELLS_WIDE`. Multiply by `4` to get
    /// the pixel-column index.
    pub cell_col: usize,
}

impl CellCoord {
    /// Top-left pixel position of this cell in the luma plane.
    pub fn pixel_origin(self) -> (usize, usize) {
        (self.cell_row * 4, self.cell_col * 4)
    }
}

/// §5.11.4 dispatch-order cell sequence for a `CELLS_WIDE × CELLS_HIGH`
/// grid rooted at a single BLOCK_16X16 PARTITION_SPLIT. The §5.11.4
/// recursion is NW → NE → SW → SE, so for a 4×4 cell grid (16 cells)
/// the order is the Z-curve over the two-level split tree.
///
/// Returns the 16 cell coordinates in the order
/// `(0,0), (0,1), (1,0), (1,1), (0,2), (0,3), (1,2), (1,3),
///  (2,0), (2,1), (3,0), (3,1), (2,2), (2,3), (3,2), (3,3)`.
#[must_use]
pub fn dispatch_order_cells() -> Vec<CellCoord> {
    let mut out = Vec::with_capacity(CELLS_WIDE * CELLS_HIGH);
    // Outer BLOCK_16X16 SPLIT ⇒ 4 BLOCK_8X8 quadrants at (0,0), (0,2),
    // (2,0), (2,2) (in mi units). Each BLOCK_8X8 SPLIT ⇒ 4 BLOCK_4X4
    // leaves at (mi+0,mi+0), (mi+0,mi+1), (mi+1,mi+0), (mi+1,mi+1).
    // Because the cell index equals the mi index for BLOCK_4X4 leaves
    // (NUM_4X4_BLOCKS_WIDE[BLOCK_4X4] = 1), the cell-coord walk lines
    // up directly with the §5.11.4 recursion.
    for &(quad_r, quad_c) in &[(0u32, 0u32), (0, 2), (2, 0), (2, 2)] {
        for dr in 0..2u32 {
            for dc in 0..2u32 {
                out.push(CellCoord {
                    cell_row: (quad_r + dr) as usize,
                    cell_col: (quad_c + dc) as usize,
                });
            }
        }
    }
    out
}

/// Compute the DC_PRED prediction for the 4×4 cell at `(cell_row,
/// cell_col)` against the running `reconstructed` luma buffer. Mirrors
/// the §7.11.2.5 four-arm dispatch on `(haveLeft, haveAbove)` via
/// [`predict_intra_dc_pred`] in a contained-extent form.
///
/// `bit_depth` must be `8`; this driver runs at `BitDepth = 8` per
/// §5.5.2 (the tiny fixture's sequence header).
fn dc_pred_for_cell_y(
    reconstructed: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    cell_row: usize,
    cell_col: usize,
) -> [u8; 16] {
    let row0 = cell_row * 4;
    let col0 = cell_col * 4;
    let have_above = (cell_row > 0) as u8;
    let have_left = (cell_col > 0) as u8;
    let mut above_row: [u16; 4] = [0; 4];
    let mut left_col: [u16; 4] = [0; 4];
    if have_above != 0 {
        for j in 0..4 {
            above_row[j] = reconstructed[row0 - 1][col0 + j] as u16;
        }
    }
    if have_left != 0 {
        for i in 0..4 {
            left_col[i] = reconstructed[row0 + i][col0 - 1] as u16;
        }
    }
    let mut pred16 = [0u16; 16];
    predict_intra_dc_pred(
        have_left,
        have_above,
        2,
        2,
        4,
        4,
        8,
        &above_row,
        &left_col,
        &mut pred16,
    )
    .expect("oxideav-av1 pixel_driver: DC_PRED arguments in range");
    let mut pred8 = [0u8; 16];
    for (slot, v) in pred8.iter_mut().zip(pred16.iter().copied()) {
        *slot = v as u8;
    }
    pred8
}

/// §7.11.2.5 DC_PRED prediction for one 4×4 chroma cell against the
/// running reconstructed chroma plane. Same shape as the luma helper
/// but indexes a smaller chroma surface (`CHROMA_HEIGHT × CHROMA_WIDTH`).
///
/// `(c_row, c_col)` are 4×4 chroma-cell coordinates in
/// `0..CHROMA_CELLS_HIGH` × `0..CHROMA_CELLS_WIDE`.
fn dc_pred_for_cell_chroma(
    reconstructed: &[[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    c_row: usize,
    c_col: usize,
) -> [u8; 16] {
    let row0 = c_row * 4;
    let col0 = c_col * 4;
    let have_above = (c_row > 0) as u8;
    let have_left = (c_col > 0) as u8;
    let mut above_row: [u16; 4] = [0; 4];
    let mut left_col: [u16; 4] = [0; 4];
    if have_above != 0 {
        for j in 0..4 {
            above_row[j] = reconstructed[row0 - 1][col0 + j] as u16;
        }
    }
    if have_left != 0 {
        for i in 0..4 {
            left_col[i] = reconstructed[row0 + i][col0 - 1] as u16;
        }
    }
    let mut pred16 = [0u16; 16];
    predict_intra_dc_pred(
        have_left,
        have_above,
        2,
        2,
        4,
        4,
        8,
        &above_row,
        &left_col,
        &mut pred16,
    )
    .expect("oxideav-av1 pixel_driver: chroma DC_PRED arguments in range");
    let mut pred8 = [0u8; 16];
    for (slot, v) in pred8.iter_mut().zip(pred16.iter().copied()) {
        *slot = v as u8;
    }
    pred8
}

/// Map a luma cell coordinate to its `(mi_row, mi_col)` mi-grid
/// coordinates. At BLOCK_4X4 the cell index equals the mi index per
/// [`dispatch_order_cells`].
#[inline]
fn cell_to_mi(cc: CellCoord) -> (u32, u32) {
    (cc.cell_row as u32, cc.cell_col as u32)
}

/// §5.11.5 `HasChroma` predicate for the 4:2:0 BLOCK_4X4 leaf walk used
/// by [`encode_intra_frame_yuv`]: at `bw4 == 1`, `bh4 == 1`,
/// `subsampling_x == 1`, `subsampling_y == 1`, `HasChroma` is `true`
/// iff `(MiRow & 1) != 0 && (MiCol & 1) != 0`. The other three quadrants
/// of each 8×8 luma block defer the chroma coefficients to the SE
/// corner (which covers their shared chroma 4×4).
#[inline]
fn cell_has_chroma_420(cc: CellCoord) -> bool {
    let (mi_row, mi_col) = cell_to_mi(cc);
    (mi_row & 1) != 0 && (mi_col & 1) != 0
}

/// Map the SE-corner luma cell that carries the chroma coefficients to
/// its chroma 4×4 block coordinates. Inverse of "chroma 4×4 (cr,cc)
/// hangs off luma cell (2*cr+1, 2*cc+1)".
#[inline]
fn cell_to_chroma_block(cc: CellCoord) -> (usize, usize) {
    debug_assert!(cell_has_chroma_420(cc));
    ((cc.cell_row - 1) / 2, (cc.cell_col - 1) / 2)
}

// ---------------------------------------------------------------------
// Arc r228 — full 13-mode intra-prediction picker.
//
// For each 4×4 luma cell the encoder now computes a prediction with each
// of the 13 §6.10.x intra Y modes (`DC_PRED`, `V_PRED`, `H_PRED`,
// `D45_PRED`, `D135_PRED`, `D113_PRED`, `D157_PRED`, `D203_PRED`,
// `D67_PRED`, `SMOOTH_PRED`, `SMOOTH_V_PRED`, `SMOOTH_H_PRED`,
// `PAETH_PRED`) and selects whichever yields the smallest residual SSD
// against the actual input. The chosen mode is then committed via
// `write_y_mode` (r211) and the residual is encoded with
// `forward_transform_2d` (r227) + `forward_quantize` (r220) +
// `write_coefficients` (r215).
//
// The picker is residual-variance-only — it does NOT model the §5.11.22
// `y_mode` rate cost (the §8.3.2 `TileYModeCdf[ Size_Group ]` row
// probabilities) — which means non-DC_PRED choices can occasionally cost
// more bits than they save in residual energy. Full RD with bit-cost
// weighting is a follow-up.
//
// The lossless `base_q_idx = 0` arm is preserved: every mode's
// prediction is integer, the residual is forward-WHT'd, and the
// quantize / dequantize chain is bit-exact ⇒ the existing end-to-end
// roundtrip tests still pass regardless of which mode the picker
// selects.
//
// All neighbour derivations (`AboveRow[]`, `LeftCol[]`, `AboveRow[-1]`)
// come from the running reconstructed luma plane via plain index reads,
// mirroring the §7.11.2.1 prologue restricted to the BLOCK_4X4 cell
// extent.
// ---------------------------------------------------------------------

/// Number of §6.10.x Y intra modes — `DC_PRED` through `PAETH_PRED`
/// inclusive (`13` per §3 `INTRA_MODES`).
pub const NUM_INTRA_MODES_Y: usize = 13;

/// The 13 §6.10.x Y intra mode ordinals (`0..=12`), in the order the
/// picker enumerates them. Constant; exposed so test code can assert
/// completeness.
pub const INTRA_MODE_CANDIDATES_Y: [usize; NUM_INTRA_MODES_Y] = [
    DC_PRED,
    V_PRED,
    H_PRED,
    D45_PRED,
    D135_PRED,
    D113_PRED,
    D157_PRED,
    D203_PRED,
    D67_PRED,
    SMOOTH_PRED,
    SMOOTH_V_PRED,
    SMOOTH_H_PRED,
    PAETH_PRED,
];

/// §7.11.2.1 prologue — derive the §7.11.2.{2..6} neighbour arrays for
/// one BLOCK_4X4 cell at `(row0, col0)` against the running
/// `reconstructed` luma plane. Returns the seven values the per-mode
/// kernels consume:
///
///   * `have_above` / `have_left` — `1` when the cell has a neighbour
///     above / to the left (`row0 > 0` / `col0 > 0`).
///   * `above_ext` — the head-extended `AboveRow[]` buffer
///     (`above_ext[0]` = index `-2`, `above_ext[1]` = index `-1`,
///     `above_ext[2 + k]` = index `k` for `k = 0..w+h-1`). Used both by
///     the V_PRED / SMOOTH leaves (which read offsets `2 .. 2 + w`) and
///     by `predict_intra_directional` (which reads index `-2..w+h-1`).
///     Length `2 + w + h = 10` for 4×4.
///   * `left_ext` — head-extended `LeftCol[]`, same convention. Length
///     `10`.
///   * `above_left` — the §7.11.2.1 corner sample (`AboveRow[-1]` =
///     `LeftCol[-1]`), with the §7.11.2.1 four-arm dispatch.
///
/// All values are derived from the running reconstructed plane, exactly
/// as the spec's §7.11.2.1 prologue would after the previous block's
/// §7.12.3 step-3 merge. The `enable_intra_edge_filter` /
/// `enable_intra_edge_upsample` pre-passes are no-ops on the
/// `upsample = 0` / `filter_strength = 0` arm this picker takes.
fn derive_intra_neighbours_4x4_y(
    reconstructed: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    row0: usize,
    col0: usize,
) -> (u8, u8, [u16; 10], [u16; 10], u16) {
    let w = 4usize;
    let h = 4usize;
    let bit_depth = 8u8;
    let have_above = (row0 > 0) as u8;
    let have_left = (col0 > 0) as u8;

    // §7.11.2.1 AboveRow[-1] / LeftCol[-1] corner (av1-spec p.242).
    let above_left: u16 = if have_above != 0 && have_left != 0 {
        reconstructed[row0 - 1][col0 - 1] as u16
    } else if have_above != 0 {
        reconstructed[row0 - 1][col0] as u16
    } else if have_left != 0 {
        reconstructed[row0][col0 - 1] as u16
    } else {
        1u16 << (bit_depth - 1)
    };

    // §7.11.2.1 AboveRow[ 0..w+h-1 ]. Index `k` lives at offset `k + 2`
    // in the head-extended buffer.
    let mut above_ext = [0u16; 10];
    above_ext[0] = above_left;
    above_ext[1] = above_left;
    if have_above != 0 {
        // Real samples from CurrFrame above the cell.
        let above_row = &reconstructed[row0 - 1];
        // §7.11.2.1 arm 3: take samples `[col0 .. col0 + w + h - 1]`,
        // clamped to the right edge of the frame (`FRAME_WIDTH - 1`).
        for k in 0..(w + h) {
            let col = (col0 + k).min(FRAME_WIDTH - 1);
            above_ext[2 + k] = above_row[col] as u16;
        }
    } else if have_left != 0 {
        // §7.11.2.1 arm 1 (have_left only): broadcast `CurrFrame[y][x-1]`.
        let sample = reconstructed[row0][col0 - 1] as u16;
        for slot in above_ext.iter_mut().skip(2).take(w + h) {
            *slot = sample;
        }
    } else {
        // §7.11.2.1 arm 2: `(1 << (BitDepth - 1)) - 1`.
        let mid_minus = ((1u32 << (bit_depth - 1)) - 1) as u16;
        for slot in above_ext.iter_mut().skip(2).take(w + h) {
            *slot = mid_minus;
        }
    }

    // §7.11.2.1 LeftCol[ 0..w+h-1 ].
    let mut left_ext = [0u16; 10];
    left_ext[0] = above_left;
    left_ext[1] = above_left;
    if have_left != 0 {
        for k in 0..(w + h) {
            let row = (row0 + k).min(FRAME_HEIGHT - 1);
            left_ext[2 + k] = reconstructed[row][col0 - 1] as u16;
        }
    } else if have_above != 0 {
        let sample = reconstructed[row0 - 1][col0] as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = sample;
        }
    } else {
        let mid_plus = ((1u32 << (bit_depth - 1)) + 1) as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = mid_plus;
        }
    }

    (have_above, have_left, above_ext, left_ext, above_left)
}

/// §7.11.2.1 prologue for one chroma 4×4 cell — mirror of
/// [`derive_intra_neighbours_4x4_y`] against the smaller
/// `CHROMA_WIDTH × CHROMA_HEIGHT` plane. Currently unused (chroma still
/// picks `DC_PRED` in r228); kept as the seed for the next-arc chroma
/// 13-mode picker.
#[allow(dead_code)]
fn derive_intra_neighbours_4x4_chroma(
    reconstructed: &[[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    row0: usize,
    col0: usize,
) -> (u8, u8, [u16; 10], [u16; 10], u16) {
    let w = 4usize;
    let h = 4usize;
    let bit_depth = 8u8;
    let have_above = (row0 > 0) as u8;
    let have_left = (col0 > 0) as u8;

    let above_left: u16 = if have_above != 0 && have_left != 0 {
        reconstructed[row0 - 1][col0 - 1] as u16
    } else if have_above != 0 {
        reconstructed[row0 - 1][col0] as u16
    } else if have_left != 0 {
        reconstructed[row0][col0 - 1] as u16
    } else {
        1u16 << (bit_depth - 1)
    };

    let mut above_ext = [0u16; 10];
    above_ext[0] = above_left;
    above_ext[1] = above_left;
    if have_above != 0 {
        let above_row = &reconstructed[row0 - 1];
        for k in 0..(w + h) {
            let col = (col0 + k).min(CHROMA_WIDTH - 1);
            above_ext[2 + k] = above_row[col] as u16;
        }
    } else if have_left != 0 {
        let sample = reconstructed[row0][col0 - 1] as u16;
        for slot in above_ext.iter_mut().skip(2).take(w + h) {
            *slot = sample;
        }
    } else {
        let mid_minus = ((1u32 << (bit_depth - 1)) - 1) as u16;
        for slot in above_ext.iter_mut().skip(2).take(w + h) {
            *slot = mid_minus;
        }
    }

    let mut left_ext = [0u16; 10];
    left_ext[0] = above_left;
    left_ext[1] = above_left;
    if have_left != 0 {
        for k in 0..(w + h) {
            let row = (row0 + k).min(CHROMA_HEIGHT - 1);
            left_ext[2 + k] = reconstructed[row][col0 - 1] as u16;
        }
    } else if have_above != 0 {
        let sample = reconstructed[row0 - 1][col0] as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = sample;
        }
    } else {
        let mid_plus = ((1u32 << (bit_depth - 1)) + 1) as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = mid_plus;
        }
    }

    (have_above, have_left, above_ext, left_ext, above_left)
}

/// Compute the prediction for a single §6.10.x Y intra mode against
/// the supplied pre-derived neighbour arrays. Returns the 16 sample
/// `pred[0..16]` as `u8` (clamped to `[0, 255]` for `bit_depth = 8`).
///
/// Routes to the matching `cdf::predict_intra_*` leaf:
///
///   * `DC_PRED`        → [`predict_intra_dc_pred`]
///   * `V_PRED`         → [`predict_intra_v_pred`]
///   * `H_PRED`         → [`predict_intra_h_pred`]
///   * `D{45,113,135,157,203,67}_PRED` → [`predict_intra_d_mode`]
///   * `SMOOTH_PRED`    → [`predict_intra_smooth_pred`]
///   * `SMOOTH_V_PRED`  → [`predict_intra_smooth_v_pred`]
///   * `SMOOTH_H_PRED`  → [`predict_intra_smooth_h_pred`]
///   * `PAETH_PRED`     → [`predict_intra_paeth_pred`]
///
/// Returns `None` for an unrecognised mode (caller bug).
fn predict_intra_mode_4x4(
    mode: usize,
    have_above: u8,
    have_left: u8,
    above_ext: &[u16; 10],
    left_ext: &[u16; 10],
    above_left: u16,
) -> Option<[u8; 16]> {
    let w = 4usize;
    let h = 4usize;
    let log2_w = 2u32;
    let log2_h = 2u32;
    let bit_depth = 8u8;
    // The V_PRED / H_PRED / DC_PRED / SMOOTH / PAETH leaves consume the
    // §7.11.2.1 `AboveRow[ 0..w-1 ]` / `LeftCol[ 0..h-1 ]` (indices `0
    // .. w+h-1` minus the head extension). Slice them out.
    let above_row = &above_ext[2..2 + w + h];
    let left_col = &left_ext[2..2 + w + h];

    let mut pred16 = [0u16; 16];
    match mode {
        m if m == DC_PRED => {
            predict_intra_dc_pred(
                have_left,
                have_above,
                log2_w,
                log2_h,
                w,
                h,
                bit_depth,
                above_row,
                left_col,
                &mut pred16,
            )
            .ok()?;
        }
        m if m == V_PRED => {
            predict_intra_v_pred(w, h, above_row, &mut pred16).ok()?;
        }
        m if m == H_PRED => {
            predict_intra_h_pred(w, h, left_col, &mut pred16).ok()?;
        }
        m if (D45_PRED..=D67_PRED).contains(&m) => {
            // `predict_intra_d_mode` reads index `-2..w+h-1` from the
            // head-extended buffers we built; pass the full `above_ext`
            // / `left_ext` (length 10) so the negative-index reads land
            // on the corner.
            predict_intra_d_mode(
                m,
                /* angle_delta = */ 0,
                w,
                h,
                /* upsample_above = */ 0,
                /* upsample_left = */ 0,
                above_ext,
                left_ext,
                &mut pred16,
            )
            .ok()?;
        }
        m if m == SMOOTH_PRED => {
            predict_intra_smooth_pred(log2_w, log2_h, w, h, above_row, left_col, &mut pred16)
                .ok()?;
        }
        m if m == SMOOTH_V_PRED => {
            predict_intra_smooth_v_pred(log2_h, w, h, above_row, left_col, &mut pred16).ok()?;
        }
        m if m == SMOOTH_H_PRED => {
            predict_intra_smooth_h_pred(log2_w, w, h, above_row, left_col, &mut pred16).ok()?;
        }
        m if m == PAETH_PRED => {
            predict_intra_paeth_pred(w, h, above_row, left_col, above_left, &mut pred16).ok()?;
        }
        _ => return None,
    }

    let mut pred8 = [0u8; 16];
    for (slot, v) in pred8.iter_mut().zip(pred16.iter().copied()) {
        *slot = v as u8;
    }
    Some(pred8)
}

/// Pick the §6.10.x Y intra mode with the smallest residual SSD against
/// the actual block samples. Returns `(best_mode, best_prediction)`.
///
/// SSD is computed over the 16 cell samples as `Σ (input - pred)²`,
/// which is a coarse proxy for "smaller residual" but ignores the
/// §5.11.22 `y_mode` rate cost. Full RD with bit-cost weighting is a
/// follow-up arc.
///
/// `DC_PRED` is the tie-breaker: when multiple modes yield the same
/// SSD, the lowest mode ordinal wins (mirrors the enumeration order in
/// [`INTRA_MODE_CANDIDATES_Y`]).
fn pick_best_intra_mode_4x4_y(
    reconstructed: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    input: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    row0: usize,
    col0: usize,
) -> (u8, [u8; 16]) {
    let (have_above, have_left, above_ext, left_ext, above_left) =
        derive_intra_neighbours_4x4_y(reconstructed, row0, col0);
    let mut best_mode = DC_PRED as u8;
    let mut best_pred = [0u8; 16];
    let mut best_ssd: u64 = u64::MAX;
    for &mode in &INTRA_MODE_CANDIDATES_Y {
        let Some(pred) = predict_intra_mode_4x4(
            mode, have_above, have_left, &above_ext, &left_ext, above_left,
        ) else {
            continue;
        };
        let mut ssd: u64 = 0;
        for i in 0..4 {
            for j in 0..4 {
                let p = input[row0 + i][col0 + j] as i32;
                let q = pred[i * 4 + j] as i32;
                let d = (p - q) as i64;
                ssd += (d * d) as u64;
            }
        }
        if ssd < best_ssd {
            best_ssd = ssd;
            best_mode = mode as u8;
            best_pred = pred;
        }
    }
    (best_mode, best_pred)
}

/// Encode a 16×16 monochrome Y-only intra-only frame at `base_q_idx = 0`
/// against the tiny-fixture-derived sequence + frame headers.
///
/// Returns the IVF bytes (one frame), the temporal-unit body bytes, and
/// the encoder-internal reconstructed pixel plane. See [`EncodedFrame`].
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] for any internal partition-tree
/// or coefficient-writer overflow (none expected for the in-range
/// 16×16 monochrome frame, but propagated for diagnostic visibility).
pub fn encode_intra_frame_y(
    luma_in: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    seq: &SequenceHeader,
    fh: &FrameHeader,
) -> Result<EncodedFrame, Error> {
    // Quantizer state matching the tiny fixture's `base_q_idx = 0`
    // header: dc_q = ac_q = 4, dqDenom = 1, no QM.
    let qp = QuantizerParams::neutral(0, 8);
    let scan = get_default_scan(TX_4X4).to_vec();
    // §5.9.2 `CodedLossless` predicate, simplified for the
    // neutral-quantizer / no-segmentation path this driver runs on:
    // `base_q_idx == 0 && all DeltaQ?? == 0` (the `QuantizerParams::neutral`
    // constructor zeros every delta). When true, the per-block §6.10.1
    // `Lossless` flag is `1` for every block in the frame ⇒ §5.11.5
    // forces `tx_size = TX_4X4` and §7.13.3 routes through the
    // §7.13.2.10 inverse WHT path. The encoder mirrors this with the
    // forward WHT instead of forward DCT, which makes the
    // `(forward_wht, forward_quantize) → (dequantize_step1, inverse_WHT)`
    // chain bit-exact on arbitrary inputs.
    let lossless = qp.base_q_idx == 0
        && qp.delta_q_y_dc == 0
        && qp.delta_q_u_dc == 0
        && qp.delta_q_u_ac == 0
        && qp.delta_q_v_dc == 0
        && qp.delta_q_v_ac == 0;

    // Encoder-side state: a running reconstructed-pixel buffer and a
    // per-cell list of committed `Quant[]` arrays in dispatch order.
    let mut reconstructed: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
    let mut committed_quants: Vec<Vec<i32>> = Vec::with_capacity(CELLS_WIDE * CELLS_HIGH);
    let mut committed_y_modes: Vec<u8> = Vec::with_capacity(CELLS_WIDE * CELLS_HIGH);

    // Build the §5.11.4 dispatch-ordered list of (cell, EncodeBlock)
    // pairs. For each cell, the leaf carries the per-plane
    // `coefficients` from `forward_quantize(forward_dct_4x4(residual))`.
    // The intra prediction is computed from the running
    // `reconstructed` buffer — this keeps the encoder's predictor in
    // lockstep with what the decoder would compute on its parallel walk.
    let cells = dispatch_order_cells();
    let mut leaves: Vec<EncodeBlock> = Vec::with_capacity(cells.len());
    for cc in &cells {
        let (row0, col0) = cc.pixel_origin();
        // r228: pick the §6.10.x Y intra mode with the smallest residual
        // SSD against the actual block samples (variance-based picker,
        // no bit-cost weighting yet). Replaces the hardcoded DC_PRED.
        let (y_mode_pick, pred) = pick_best_intra_mode_4x4_y(&reconstructed, luma_in, row0, col0);
        committed_y_modes.push(y_mode_pick);
        let _ = dc_pred_for_cell_y; // keep symbol referenced for now
                                    // Residual = input - prediction.
        let mut residual = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let p = luma_in[row0 + i][col0 + j] as i64;
                let q = pred[i * 4 + j] as i64;
                residual[i * 4 + j] = p - q;
            }
        }
        // Forward transform. On the lossless arm route through the
        // forward 4×4 WHT (the bit-exact integer inverse of the
        // §7.13.2.10 inverse WHT used by §7.13.3's `Lossless` branch);
        // otherwise the forward DCT (lossy at q_index > 0).
        let coeffs = if lossless {
            forward_wht_4x4(&residual).to_vec()
        } else {
            forward_dct_4x4(&residual).to_vec()
        };
        // Forward quantize at q_index = 0, DCT_DCT, no QM (seg_qm_level
        // = 15 takes the identity branch). On the lossless arm, every
        // `forward_wht_4x4` output is divisible by `4 = q2 / dq_denom`
        // (the row pass multiplies by `1 << 2`), so the forward
        // quantizer round-trip is bit-exact.
        let quant = forward_quantize(&coeffs, TX_4X4, /* plane = */ 0, 0, DCT_DCT, 15, &qp);
        committed_quants.push(quant.clone());

        // Encoder-internal pixel reconstruction: decoder walk of these
        // same `quant` values. Pass the `lossless` flag through to the
        // §7.13.3 dispatcher so the inverse WHT path matches the
        // encoder's forward WHT choice.
        let dequant = dequantize_step1(&quant, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let residual_back = inverse_transform_2d(
            &dequant, TX_4X4, DCT_DCT, /* bit_depth = */ 8, lossless,
        );
        for i in 0..4 {
            for j in 0..4 {
                let p = pred[i * 4 + j] as i64 + residual_back[i * 4 + j];
                let clamped = p.clamp(0, 255) as u8;
                reconstructed[row0 + i][col0 + j] = clamped;
            }
        }

        // Build the §5.11.5 leaf for the §5.11.4 driver. Y-only ⇒
        // `uv_mode = None`.
        let plane_y = PlaneCoefficients {
            plane: 0,
            is_inter: 0,
            tx_size: TX_4X4,
            tx_class: TX_CLASS_2D,
            txb_skip_ctx: 0,
            dc_sign_ctx: 0,
            scan: scan.clone(),
            quant: quant.clone(),
        };
        leaves.push(EncodeBlock {
            skip: 0,
            segment_id: 0,
            segment_pred: 0,
            y_mode: y_mode_pick,
            uv_mode: None,
            coefficients: vec![plane_y],
        });
    }

    // Assemble the §5.11.4 partition tree: BLOCK_16X16 root SPLIT into
    // 4 BLOCK_8X8 SPLITs, each into 4 BLOCK_4X4 leaves.
    let leaf_iter = leaves.into_iter().map(EncodeNode::Leaf);
    let leaves_arr: Vec<EncodeNode> = leaf_iter.collect();
    debug_assert_eq!(leaves_arr.len(), 16);
    // Build innermost BLOCK_8X8 SPLIT nodes (4 of them, one per quad).
    let mut leaf_iter = leaves_arr.into_iter();
    let mut quads: Vec<EncodeNode> = Vec::with_capacity(4);
    for _ in 0..4 {
        let a = Box::new(leaf_iter.next().unwrap());
        let b = Box::new(leaf_iter.next().unwrap());
        let c = Box::new(leaf_iter.next().unwrap());
        let d = Box::new(leaf_iter.next().unwrap());
        quads.push(EncodeNode::Split([a, b, c, d]));
    }
    let nw = Box::new(quads.remove(0));
    let ne = Box::new(quads.remove(0));
    let sw = Box::new(quads.remove(0));
    let se = Box::new(quads.remove(0));
    let root = EncodeNode::Split([nw, ne, sw, se]);

    // §5.11 entropy-coder run for the single tile covering the frame.
    let mut writer = SymbolWriter::new(false);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut state = PartitionTreeWriter::new(
        MI_ROWS,
        MI_COLS,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: MI_ROWS,
            mi_col_start: 0,
            mi_col_end: MI_COLS,
        },
        /* segmentation_enabled = */ false,
        /* last_active_seg_id = */ 0,
        /* lossless = */ false,
        /* subsampling_x = */ false,
        /* subsampling_y = */ false,
    )
    .expect("oxideav-av1 pixel_driver: partition state construction");
    write_partition_tree(&mut writer, &mut cdfs, &mut state, &root, 0, 0, BLOCK_16X16)?;
    let tile_bytes = writer.finish();

    // §5.11.1 tile-group OBU body.
    let tile_group = TileGroupObu {
        num_tiles: 1,
        tile_cols_log2: 0,
        tile_rows_log2: 0,
        tile_size_bytes: 1,
        tg_start: 0,
        tg_end: 0,
        start_and_end_present: false,
        tiles: vec![TilePayload::new(tile_bytes)],
    };
    let tile_group_body = write_tile_group_obu(&tile_group)?;

    // Build the per-frame OBU sequence: FrameHeader OBU + TileGroup OBU.
    // `encode_temporal_unit` handles the SH-emission gate.
    // We hand-build the frame_obus list because the temporal-unit
    // helper only knows about FrameHeaders; tile-group OBUs are
    // appended after each frame's FH.
    let fh_body = crate::encoder::frame_obu::write_frame_header_obu(fh, seq);
    let frame_obus: Vec<ObuFrame> = vec![
        ObuFrame::new(ObuType::FrameHeader, fh_body),
        ObuFrame::new(ObuType::TileGroup, tile_group_body),
    ];

    // §7.5 temporal unit: TD + SH + (FH + TileGroup).
    let temporal_unit_bytes = {
        use crate::encoder::obu::build_temporal_unit;
        use crate::encoder::sequence_obu::write_sequence_header_obu;
        let sh_body = write_sequence_header_obu(seq);
        build_temporal_unit(Some(&sh_body), &frame_obus)
    };
    // `TemporalUnitPlan` is the public entry; we shell out to
    // `build_temporal_unit` directly because `TemporalUnitPlan` only
    // emits one OBU per frame. Keep `TemporalUnitPlan` in scope for
    // the next arc that lands a `frames: &[FrameAndTiles]` extension.
    let _ = TemporalUnitPlan {
        seq,
        emit_sequence_header: true,
        frames: &[],
    };
    let _ = encode_temporal_unit;

    // IVF v0 container — one frame at pts = 0.
    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(
            cursor,
            FOURCC_AV01,
            FRAME_WIDTH as u16,
            FRAME_HEIGHT as u16,
            25,
            1,
        )
        .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.write_frame(&temporal_unit_bytes, 0)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedFrame {
        ivf_bytes,
        temporal_unit_bytes,
        reconstructed_y: reconstructed,
        committed_quants,
        committed_y_modes,
    })
}

/// Encode a 16×16 4:2:0 YUV intra-only frame at `base_q_idx = 0` —
/// the chroma sibling of [`encode_intra_frame_y`].
///
/// Composes the per-plane DC_PRED + forward WHT + forward quantize +
/// `write_coefficients` chain for the four 4×4 chroma blocks at the
/// §5.11.5 `HasChroma` cells `(1,1), (1,3), (3,1), (3,3)`. The
/// non-chroma luma cells leave their `uv_mode` as `None` and emit no
/// chroma coefficient pass, matching the decoder's parallel walk
/// (`HasChroma == 0` ⇒ no chroma syntax). Lossless WHT path mirrors the
/// luma side, so chroma planes also round-trip pixel-for-pixel at
/// `base_q_idx = 0` on arbitrary inputs.
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] for any internal partition-tree
/// or coefficient-writer overflow.
pub fn encode_intra_frame_yuv(
    input: &Yuv420Frame16x16,
    seq: &SequenceHeader,
    fh: &FrameHeader,
) -> Result<EncodedFrameYuv, Error> {
    // Same neutral-quantiser / no-segmentation / no-QM setup as the
    // luma-only path. `QuantizerParams::neutral` zeros every chroma
    // delta too, so `get_dc_quant(plane = 1) = get_ac_quant(plane = 1)
    // = dc_q(8, 0) = ac_q(8, 0) = 4`, same as luma.
    let qp = QuantizerParams::neutral(0, 8);
    let scan = get_default_scan(TX_4X4).to_vec();
    let lossless = qp.base_q_idx == 0
        && qp.delta_q_y_dc == 0
        && qp.delta_q_u_dc == 0
        && qp.delta_q_u_ac == 0
        && qp.delta_q_v_dc == 0
        && qp.delta_q_v_ac == 0;

    // Running reconstructed-pixel buffers per plane.
    let mut recon_y: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
    let mut recon_u: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT] = [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT];
    let mut recon_v: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT] = [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT];
    let mut committed_quants_y: Vec<Vec<i32>> = Vec::with_capacity(CELLS_WIDE * CELLS_HIGH);
    let mut committed_quants_u: Vec<Vec<i32>> =
        Vec::with_capacity(CHROMA_CELLS_WIDE * CHROMA_CELLS_HIGH);
    let mut committed_quants_v: Vec<Vec<i32>> =
        Vec::with_capacity(CHROMA_CELLS_WIDE * CHROMA_CELLS_HIGH);
    let mut committed_y_modes: Vec<u8> = Vec::with_capacity(CELLS_WIDE * CELLS_HIGH);

    let cells = dispatch_order_cells();
    let mut leaves: Vec<EncodeBlock> = Vec::with_capacity(cells.len());

    for cc in &cells {
        // --- Luma walk (identical to encode_intra_frame_y) ---
        let (row0, col0) = cc.pixel_origin();
        // r228: pick from all 13 §6.10.x Y intra modes by residual SSD
        // (same scheme as `encode_intra_frame_y`).
        let (y_mode_pick, pred_y) = pick_best_intra_mode_4x4_y(&recon_y, &input.y, row0, col0);
        committed_y_modes.push(y_mode_pick);
        let mut residual_y = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let p = input.y[row0 + i][col0 + j] as i64;
                let q = pred_y[i * 4 + j] as i64;
                residual_y[i * 4 + j] = p - q;
            }
        }
        let coeffs_y = if lossless {
            forward_wht_4x4(&residual_y).to_vec()
        } else {
            forward_dct_4x4(&residual_y).to_vec()
        };
        let quant_y = forward_quantize(&coeffs_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        committed_quants_y.push(quant_y.clone());

        let dequant_y = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let resid_back_y = inverse_transform_2d(&dequant_y, TX_4X4, DCT_DCT, 8, lossless);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + resid_back_y[i * 4 + j];
                recon_y[row0 + i][col0 + j] = p.clamp(0, 255) as u8;
            }
        }

        // --- Chroma walk (only on the §5.11.5 HasChroma cells) ---
        let mut coefficients: Vec<PlaneCoefficients> = Vec::with_capacity(3);
        coefficients.push(PlaneCoefficients {
            plane: 0,
            is_inter: 0,
            tx_size: TX_4X4,
            tx_class: TX_CLASS_2D,
            txb_skip_ctx: 0,
            dc_sign_ctx: 0,
            scan: scan.clone(),
            quant: quant_y.clone(),
        });

        let uv_mode = if cell_has_chroma_420(*cc) {
            let (cr, cc_idx) = cell_to_chroma_block(*cc);
            let crow0 = cr * 4;
            let ccol0 = cc_idx * 4;

            // Walk U then V — same chain as luma, plane = 1 / 2.
            for (plane, recon_chroma, input_chroma, committed) in [
                (1u8, &mut recon_u, &input.u, &mut committed_quants_u),
                (2u8, &mut recon_v, &input.v, &mut committed_quants_v),
            ] {
                let pred_c = dc_pred_for_cell_chroma(recon_chroma, cr, cc_idx);
                let mut residual_c = [0i64; 16];
                for i in 0..4 {
                    for j in 0..4 {
                        let p = input_chroma[crow0 + i][ccol0 + j] as i64;
                        let q = pred_c[i * 4 + j] as i64;
                        residual_c[i * 4 + j] = p - q;
                    }
                }
                let coeffs_c = if lossless {
                    forward_wht_4x4(&residual_c).to_vec()
                } else {
                    forward_dct_4x4(&residual_c).to_vec()
                };
                let quant_c = forward_quantize(&coeffs_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);
                committed.push(quant_c.clone());

                let dequant_c = dequantize_step1(&quant_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);
                let resid_back_c = inverse_transform_2d(&dequant_c, TX_4X4, DCT_DCT, 8, lossless);
                for i in 0..4 {
                    for j in 0..4 {
                        let p = pred_c[i * 4 + j] as i64 + resid_back_c[i * 4 + j];
                        recon_chroma[crow0 + i][ccol0 + j] = p.clamp(0, 255) as u8;
                    }
                }

                coefficients.push(PlaneCoefficients {
                    plane,
                    is_inter: 0,
                    tx_size: TX_4X4,
                    tx_class: TX_CLASS_2D,
                    txb_skip_ctx: 0,
                    dc_sign_ctx: 0,
                    scan: scan.clone(),
                    quant: quant_c,
                });
            }
            Some(DC_PRED as u8)
        } else {
            None
        };

        leaves.push(EncodeBlock {
            skip: 0,
            segment_id: 0,
            segment_pred: 0,
            y_mode: y_mode_pick,
            uv_mode,
            coefficients,
        });
    }

    // --- Assemble + emit (same as luma-only path) ---
    let leaf_iter = leaves.into_iter().map(EncodeNode::Leaf);
    let leaves_arr: Vec<EncodeNode> = leaf_iter.collect();
    debug_assert_eq!(leaves_arr.len(), 16);
    let mut leaf_iter = leaves_arr.into_iter();
    let mut quads: Vec<EncodeNode> = Vec::with_capacity(4);
    for _ in 0..4 {
        let a = Box::new(leaf_iter.next().unwrap());
        let b = Box::new(leaf_iter.next().unwrap());
        let c = Box::new(leaf_iter.next().unwrap());
        let d = Box::new(leaf_iter.next().unwrap());
        quads.push(EncodeNode::Split([a, b, c, d]));
    }
    let nw = Box::new(quads.remove(0));
    let ne = Box::new(quads.remove(0));
    let sw = Box::new(quads.remove(0));
    let se = Box::new(quads.remove(0));
    let root = EncodeNode::Split([nw, ne, sw, se]);

    let mut writer = SymbolWriter::new(false);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // 4:2:0 subsampling matches the tiny fixture's color config so the
    // CFL-allowed / chroma-arm derivations in the per-leaf writer use
    // the right subsampling booleans.
    let mut state = PartitionTreeWriter::new(
        MI_ROWS,
        MI_COLS,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: MI_ROWS,
            mi_col_start: 0,
            mi_col_end: MI_COLS,
        },
        /* segmentation_enabled = */ false,
        /* last_active_seg_id = */ 0,
        /* lossless = */ false,
        /* subsampling_x = */ true,
        /* subsampling_y = */ true,
    )
    .expect("oxideav-av1 pixel_driver: partition state construction (yuv)");
    write_partition_tree(&mut writer, &mut cdfs, &mut state, &root, 0, 0, BLOCK_16X16)?;
    let tile_bytes = writer.finish();

    let tile_group = TileGroupObu {
        num_tiles: 1,
        tile_cols_log2: 0,
        tile_rows_log2: 0,
        tile_size_bytes: 1,
        tg_start: 0,
        tg_end: 0,
        start_and_end_present: false,
        tiles: vec![TilePayload::new(tile_bytes)],
    };
    let tile_group_body = write_tile_group_obu(&tile_group)?;

    let fh_body = crate::encoder::frame_obu::write_frame_header_obu(fh, seq);
    let frame_obus: Vec<ObuFrame> = vec![
        ObuFrame::new(ObuType::FrameHeader, fh_body),
        ObuFrame::new(ObuType::TileGroup, tile_group_body),
    ];

    let temporal_unit_bytes = {
        use crate::encoder::obu::build_temporal_unit;
        use crate::encoder::sequence_obu::write_sequence_header_obu;
        let sh_body = write_sequence_header_obu(seq);
        build_temporal_unit(Some(&sh_body), &frame_obus)
    };

    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(
            cursor,
            FOURCC_AV01,
            FRAME_WIDTH as u16,
            FRAME_HEIGHT as u16,
            25,
            1,
        )
        .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.write_frame(&temporal_unit_bytes, 0)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedFrameYuv {
        ivf_bytes,
        temporal_unit_bytes,
        reconstructed_y: recon_y,
        reconstructed_u: recon_u,
        reconstructed_v: recon_v,
        committed_quants_y,
        committed_quants_u,
        committed_quants_v,
        committed_y_modes,
    })
}

/// Internal-only helpers re-exposed for the integration tests; not part
/// of the public encoder API.
pub mod internal_roundtrip {
    use super::*;

    /// Re-derive the encoder's reconstructed plane from a list of
    /// committed `Quant[]` arrays + per-cell §6.10.x Y intra modes.
    /// Mirrors the per-leaf reconstruction loop inside
    /// [`super::encode_intra_frame_y`]; provided so test code can
    /// independently verify the per-leaf pixel reconstruction without
    /// re-running the encoder driver.
    ///
    /// `y_modes[i]` must be the §6.10.x ordinal the encoder selected
    /// for `cells[i]` (i.e. the `committed_y_modes[i]` value the encoder
    /// surfaced in [`super::EncodedFrame`]). On the r228 picker every
    /// value is in `0..NUM_INTRA_MODES_Y`.
    pub fn reconstruct_from_quants(
        quants: &[Vec<i32>],
        y_modes: &[u8],
    ) -> [[u8; FRAME_WIDTH]; FRAME_HEIGHT] {
        let qp = QuantizerParams::neutral(0, 8);
        let lossless = qp.base_q_idx == 0
            && qp.delta_q_y_dc == 0
            && qp.delta_q_u_dc == 0
            && qp.delta_q_u_ac == 0
            && qp.delta_q_v_dc == 0
            && qp.delta_q_v_ac == 0;
        let mut reconstructed: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] =
            [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let cells = dispatch_order_cells();
        assert_eq!(quants.len(), cells.len());
        assert_eq!(y_modes.len(), cells.len());
        for (i, cc) in cells.iter().enumerate() {
            let (row0, col0) = cc.pixel_origin();
            let (have_above, have_left, above_ext, left_ext, above_left) =
                derive_intra_neighbours_4x4_y(&reconstructed, row0, col0);
            let pred = predict_intra_mode_4x4(
                y_modes[i] as usize,
                have_above,
                have_left,
                &above_ext,
                &left_ext,
                above_left,
            )
            .expect("oxideav-av1 reconstruct_from_quants: y_modes[i] in range");
            let dequant = dequantize_step1(&quants[i], TX_4X4, 0, 0, DCT_DCT, 15, &qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, lossless);
            for di in 0..4 {
                for dj in 0..4 {
                    let p = pred[di * 4 + dj] as i64 + residual[di * 4 + dj];
                    reconstructed[row0 + di][col0 + dj] = p.clamp(0, 255) as u8;
                }
            }
        }
        reconstructed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_header::parse_frame_header;
    use crate::obu::{ObuIter, ObuType};
    use crate::sequence_header::parse_sequence_header;

    // Tiny-i-only-16x16-prof0 fixture payloads (also used by the IVF
    // end-to-end test).
    const TINY_SEQ_PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
    const TINY_FRAME_PAYLOAD: &[u8] = &[
        0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67,
        0x6c, 0xc7, 0xee, 0x51, 0x80,
    ];

    fn tiny_seq() -> SequenceHeader {
        parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap()
    }

    fn tiny_fh(seq: &SequenceHeader) -> FrameHeader {
        parse_frame_header(TINY_FRAME_PAYLOAD, seq).unwrap()
    }

    #[test]
    fn dispatch_order_cells_visits_each_cell_exactly_once() {
        let cells = dispatch_order_cells();
        assert_eq!(cells.len(), CELLS_WIDE * CELLS_HIGH);
        let mut seen = [[false; CELLS_WIDE]; CELLS_HIGH];
        for cc in &cells {
            assert!(cc.cell_row < CELLS_HIGH);
            assert!(cc.cell_col < CELLS_WIDE);
            assert!(
                !seen[cc.cell_row][cc.cell_col],
                "duplicate cell ({}, {})",
                cc.cell_row, cc.cell_col
            );
            seen[cc.cell_row][cc.cell_col] = true;
        }
        for row in &seen {
            for &s in row {
                assert!(s);
            }
        }
    }

    #[test]
    fn dispatch_order_cells_first_quad_is_z_curve() {
        // First four cells = NW quadrant of NW outer split = the four
        // BLOCK_4X4 leaves of the BLOCK_8X8 at mi (0, 0):
        // (0,0), (0,1), (1,0), (1,1) in cell coords.
        let cells = dispatch_order_cells();
        assert_eq!(
            cells[..4],
            [
                CellCoord {
                    cell_row: 0,
                    cell_col: 0
                },
                CellCoord {
                    cell_row: 0,
                    cell_col: 1
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 0
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 1
                },
            ]
        );
        // Cells 4..8 = NE outer quadrant = mi (0, 2) BLOCK_8X8: (0,2),
        // (0,3), (1,2), (1,3).
        assert_eq!(
            cells[4..8],
            [
                CellCoord {
                    cell_row: 0,
                    cell_col: 2
                },
                CellCoord {
                    cell_row: 0,
                    cell_col: 3
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 2
                },
                CellCoord {
                    cell_row: 1,
                    cell_col: 3
                },
            ]
        );
    }

    #[test]
    fn encode_uniform_128_input_produces_zero_quant_levels() {
        // Flat mid-grey input ⇒ DC_PRED at (0,0) is the no-neighbour
        // mid-grey 128 ⇒ residual = 0 ⇒ coefficients = 0 ⇒ quants = 0.
        // Subsequent leaves see reconstructed-128 neighbours, so DC_PRED
        // again yields 128 ⇒ residual = 0 chain repeats.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(result.committed_quants.len(), CELLS_WIDE * CELLS_HIGH);
        for (idx, quant) in result.committed_quants.iter().enumerate() {
            assert_eq!(quant.len(), 16);
            for (k, &q) in quant.iter().enumerate() {
                assert_eq!(q, 0, "cell {idx} quant[{k}] = {q}, expected 0");
            }
        }
    }

    #[test]
    fn encode_uniform_128_input_reconstructs_bit_exact() {
        // The encoder's internal reconstruction must match the input
        // exactly for the flat-DC case: zero residual ⇒ recovered pixels
        // = DC_PRED prediction = 128 everywhere.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        for i in 0..FRAME_HEIGHT {
            for j in 0..FRAME_WIDTH {
                assert_eq!(
                    result.reconstructed_y[i][j], 128,
                    "reconstructed_y[{i}][{j}] = {} != 128",
                    result.reconstructed_y[i][j]
                );
            }
        }
    }

    #[test]
    fn encode_uniform_128_internal_roundtrip_matches_driver() {
        // The `internal_roundtrip::reconstruct_from_quants` helper must
        // produce the same pixels as the driver's inline reconstruction
        // when fed the same committed `Quant[]` arrays.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let recon = internal_roundtrip::reconstruct_from_quants(
            &result.committed_quants,
            &result.committed_y_modes,
        );
        assert_eq!(recon, result.reconstructed_y);
    }

    #[test]
    fn encode_uniform_64_input_internal_inverse_chain_is_bit_exact() {
        // Flat input ≠ mid-grey ⇒ first-block residual is `-64` (every
        // pixel minus the no-neighbour DC_PRED = 128 default). The
        // r222 forward WHT routes the q_index = 0 path through the
        // §7.13.3 `Lossless` arm, so the encoder's reconstruction is
        // pixel-perfect bit-exact — no per-pixel error envelope this
        // time (subsequent blocks predict from the recovered 64 pixels
        // ⇒ zero residual chain).
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[64u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let recon = internal_roundtrip::reconstruct_from_quants(
            &result.committed_quants,
            &result.committed_y_modes,
        );
        assert_eq!(recon, result.reconstructed_y);
        for i in 0..FRAME_HEIGHT {
            for j in 0..FRAME_WIDTH {
                assert_eq!(
                    result.reconstructed_y[i][j], 64,
                    "[{i}][{j}]: recon = {} != 64 (lossless WHT path)",
                    result.reconstructed_y[i][j]
                );
            }
        }
    }

    /// Bit-exact pixel roundtrip on a non-uniform input — the r222
    /// milestone unlock. At `base_q_idx = 0` the §7.13.3 `Lossless`
    /// arm routes through the §7.13.2.10 WHT, which is a pure
    /// integer-reversible butterfly. The encoder mirrors with
    /// `forward_wht_4x4`, and every coefficient ends up divisible by
    /// the lossless q2 = 4 (the WHT row pass multiplies by `1 << 2`),
    /// so the quantize → dequantize chain is exact. End-to-end the
    /// encoder's reconstruction must equal the input pixel-for-pixel.
    #[test]
    fn encode_non_uniform_input_roundtrips_bit_exact_lossless() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        // Deterministic non-uniform pattern: pixel = (16 * row + col)
        // mod 256, which exercises a wide range of inter-cell deltas
        // and forces non-zero residuals at every block.
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for (i, row) in luma.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = ((16 * i + j) & 0xFF) as u8;
            }
        }
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(
            result.reconstructed_y, luma,
            "lossless WHT roundtrip failed on non-uniform input"
        );
    }

    /// Bit-exact pixel roundtrip on a horizontal-gradient input —
    /// stresses the row-direction WHT path. The gradient covers the
    /// full `[0, 255]` u8 range across the frame width.
    #[test]
    fn encode_horizontal_gradient_roundtrips_bit_exact_lossless() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for row in luma.iter_mut() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (j * 16) as u8;
            }
        }
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(result.reconstructed_y, luma);
    }

    /// Bit-exact pixel roundtrip on a pseudo-random input — the
    /// definitive "any pixels round-trip exactly" test. Deterministic
    /// LCG so failures replay identically.
    #[test]
    fn encode_pseudorandom_input_roundtrips_bit_exact_lossless() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        for row in luma.iter_mut() {
            for cell in row.iter_mut() {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *cell = (state >> 56) as u8;
            }
        }
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        assert_eq!(
            result.reconstructed_y, luma,
            "lossless WHT roundtrip failed on pseudo-random input"
        );
    }

    #[test]
    fn encoded_ivf_bytes_parse_back_as_td_sh_fh_tg() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");

        // IVF preamble.
        assert_eq!(&result.ivf_bytes[0..4], b"DKIF");
        // Frame count == 1.
        let frame_count = u32::from_le_bytes([
            result.ivf_bytes[24],
            result.ivf_bytes[25],
            result.ivf_bytes[26],
            result.ivf_bytes[27],
        ]);
        assert_eq!(frame_count, 1);

        // Walk the temporal-unit OBUs: TD + SH + FH + TG.
        let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(descs.len(), 4, "TD + SH + FH + TG");
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
        assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
        assert_eq!(descs[3].obu_type, ObuType::TileGroup);
    }

    #[test]
    fn encoded_ivf_sh_fh_round_trip_through_parser() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        // SH parses back equal (up to bits_consumed).
        let reparsed_seq = parse_sequence_header(descs[1].payload).unwrap();
        let mut expected_seq = seq.clone();
        expected_seq.bits_consumed = reparsed_seq.bits_consumed;
        assert_eq!(reparsed_seq, expected_seq);
        // FH parses back equal (up to bits_consumed).
        let reparsed_fh = parse_frame_header(descs[2].payload, &seq).unwrap();
        let mut expected_fh = fh.clone();
        expected_fh.bits_consumed = reparsed_fh.bits_consumed;
        assert_eq!(reparsed_fh, expected_fh);
    }

    // -----------------------------------------------------------------
    // Arc 17 (round 223) — chroma-aware `encode_intra_frame_yuv` tests
    // -----------------------------------------------------------------

    #[test]
    fn has_chroma_cells_are_se_corners_of_each_8x8_quadrant() {
        // For the 4×4 mi grid at 4:2:0, has_chroma fires at the four
        // cells whose mi coords are both odd.
        let cells = dispatch_order_cells();
        let chroma_cells: Vec<CellCoord> = cells
            .iter()
            .copied()
            .filter(|cc| cell_has_chroma_420(*cc))
            .collect();
        assert_eq!(chroma_cells.len(), 4);
        // (1,1), (1,3), (3,1), (3,3) — but in dispatch order they appear
        // at positions 3 (NW quadrant SE), 7 (NE quadrant SE), 11 (SW
        // quadrant SE), 15 (SE quadrant SE).
        let positions: Vec<usize> = cells
            .iter()
            .enumerate()
            .filter(|(_, cc)| cell_has_chroma_420(**cc))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(positions, vec![3, 7, 11, 15]);
    }

    #[test]
    fn cell_to_chroma_block_maps_se_corners_to_chroma_grid() {
        assert_eq!(
            cell_to_chroma_block(CellCoord {
                cell_row: 1,
                cell_col: 1,
            }),
            (0, 0)
        );
        assert_eq!(
            cell_to_chroma_block(CellCoord {
                cell_row: 1,
                cell_col: 3,
            }),
            (0, 1)
        );
        assert_eq!(
            cell_to_chroma_block(CellCoord {
                cell_row: 3,
                cell_col: 1,
            }),
            (1, 0)
        );
        assert_eq!(
            cell_to_chroma_block(CellCoord {
                cell_row: 3,
                cell_col: 3,
            }),
            (1, 1)
        );
    }

    #[test]
    fn yuv_encode_flat_128_input_zero_quants_everywhere() {
        // Flat 128 on every plane ⇒ DC_PRED prediction = 128 at the
        // origin ⇒ zero residual ⇒ zero coefficients on Y, U, V.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16::default();
        let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        assert_eq!(result.committed_quants_y.len(), CELLS_WIDE * CELLS_HIGH);
        assert_eq!(
            result.committed_quants_u.len(),
            CHROMA_CELLS_WIDE * CHROMA_CELLS_HIGH
        );
        assert_eq!(
            result.committed_quants_v.len(),
            CHROMA_CELLS_WIDE * CHROMA_CELLS_HIGH
        );
        for q in result.committed_quants_y.iter() {
            for &v in q {
                assert_eq!(v, 0);
            }
        }
        for q in result.committed_quants_u.iter() {
            for &v in q {
                assert_eq!(v, 0);
            }
        }
        for q in result.committed_quants_v.iter() {
            for &v in q {
                assert_eq!(v, 0);
            }
        }
    }

    #[test]
    fn yuv_encode_flat_128_reconstructs_bit_exact() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16::default();
        let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        for i in 0..FRAME_HEIGHT {
            for j in 0..FRAME_WIDTH {
                assert_eq!(result.reconstructed_y[i][j], 128);
            }
        }
        for i in 0..CHROMA_HEIGHT {
            for j in 0..CHROMA_WIDTH {
                assert_eq!(result.reconstructed_u[i][j], 128);
                assert_eq!(result.reconstructed_v[i][j], 128);
            }
        }
    }

    #[test]
    fn yuv_encode_flat_64_chroma_roundtrips_bit_exact_lossless() {
        // Non-mid-grey flat chroma stresses the lossless WHT path on the
        // chroma side. Y stays at 128 (no-residual baseline); U / V at
        // 64 produce a non-zero first-block residual that must
        // round-trip exactly through the lossless arm.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16 {
            y: [[128u8; FRAME_WIDTH]; FRAME_HEIGHT],
            u: [[64u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
            v: [[192u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        };
        let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        for i in 0..CHROMA_HEIGHT {
            for j in 0..CHROMA_WIDTH {
                assert_eq!(
                    result.reconstructed_u[i][j], 64,
                    "U[{i}][{j}] = {} != 64",
                    result.reconstructed_u[i][j]
                );
                assert_eq!(
                    result.reconstructed_v[i][j], 192,
                    "V[{i}][{j}] = {} != 192",
                    result.reconstructed_v[i][j]
                );
            }
        }
    }

    #[test]
    fn yuv_encode_pseudorandom_input_roundtrips_bit_exact_lossless() {
        // The chroma equivalent of the luma pseudo-random roundtrip
        // test. Every plane is independently driven by an LCG so the
        // residuals are non-zero across the full input range.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut input = Yuv420Frame16x16 {
            y: [[0u8; FRAME_WIDTH]; FRAME_HEIGHT],
            u: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
            v: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        };
        let mut state: u64 = 0x0123_4567_89AB_CDEF;
        let step = |state: &mut u64| -> u8 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*state >> 56) as u8
        };
        for row in input.y.iter_mut() {
            for cell in row.iter_mut() {
                *cell = step(&mut state);
            }
        }
        for row in input.u.iter_mut() {
            for cell in row.iter_mut() {
                *cell = step(&mut state);
            }
        }
        for row in input.v.iter_mut() {
            for cell in row.iter_mut() {
                *cell = step(&mut state);
            }
        }
        let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        assert_eq!(result.reconstructed_y, input.y, "Y roundtrip mismatch");
        assert_eq!(result.reconstructed_u, input.u, "U roundtrip mismatch");
        assert_eq!(result.reconstructed_v, input.v, "V roundtrip mismatch");
    }

    #[test]
    fn yuv_encode_chroma_gradient_roundtrips_bit_exact_lossless() {
        // Independent horizontal gradient on Y / U / V — stresses the
        // row-direction WHT path on every plane.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut input = Yuv420Frame16x16 {
            y: [[0u8; FRAME_WIDTH]; FRAME_HEIGHT],
            u: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
            v: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        };
        for row in input.y.iter_mut() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (j * 16) as u8;
            }
        }
        for row in input.u.iter_mut() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (j * 32) as u8;
            }
        }
        for row in input.v.iter_mut() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (255 - (j * 32)) as u8;
            }
        }
        let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        assert_eq!(result.reconstructed_y, input.y);
        assert_eq!(result.reconstructed_u, input.u);
        assert_eq!(result.reconstructed_v, input.v);
    }

    #[test]
    fn yuv_encoded_ivf_bytes_parse_back_as_td_sh_fh_tg() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16::default();
        let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        assert_eq!(&result.ivf_bytes[0..4], b"DKIF");
        let frame_count = u32::from_le_bytes([
            result.ivf_bytes[24],
            result.ivf_bytes[25],
            result.ivf_bytes[26],
            result.ivf_bytes[27],
        ]);
        assert_eq!(frame_count, 1);
        let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(descs.len(), 4, "TD + SH + FH + TG");
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
        assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
        assert_eq!(descs[3].obu_type, ObuType::TileGroup);
    }

    #[test]
    fn yuv_encoded_ivf_sh_fh_round_trip_through_parser() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16::default();
        let result = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        let descs: Vec<_> = ObuIter::new(&result.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        let reparsed_seq = parse_sequence_header(descs[1].payload).unwrap();
        let mut expected_seq = seq.clone();
        expected_seq.bits_consumed = reparsed_seq.bits_consumed;
        assert_eq!(reparsed_seq, expected_seq);
        let reparsed_fh = parse_frame_header(descs[2].payload, &seq).unwrap();
        let mut expected_fh = fh.clone();
        expected_fh.bits_consumed = reparsed_fh.bits_consumed;
        assert_eq!(reparsed_fh, expected_fh);
    }

    #[test]
    fn yuv_chroma_tile_group_payload_is_larger_than_luma_only() {
        // The chroma walk adds four extra coefficient-block writes per
        // frame (one U + one V block at each of the four chroma cells).
        // A non-flat input forces those writes to produce non-zero
        // entropy ⇒ the YUV tile-group payload must exceed the Y-only
        // tile-group payload for the same luma input. Confirms the
        // chroma syntax is actually reaching the bitstream.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for (i, row) in luma.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = ((16 * i + j) & 0xFF) as u8;
            }
        }
        let y_only = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let input = Yuv420Frame16x16 {
            y: luma,
            u: [[64u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
            v: [[192u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        };
        let yuv = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        let descs_y: Vec<_> = ObuIter::new(&y_only.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        let descs_yuv: Vec<_> = ObuIter::new(&yuv.temporal_unit_bytes)
            .collect::<Result<_, _>>()
            .unwrap();
        let tg_y = descs_y.last().unwrap();
        let tg_yuv = descs_yuv.last().unwrap();
        assert!(
            tg_yuv.payload_len > tg_y.payload_len,
            "YUV tile-group payload ({}) not larger than Y-only ({})",
            tg_yuv.payload_len,
            tg_y.payload_len
        );
    }

    #[test]
    fn yuv_encode_committed_quants_y_matches_y_only_driver() {
        // Feeding the YUV encoder with input.u = input.v = flat-128 and
        // input.y = X should produce the same committed luma `Quant[]`
        // arrays as feeding the luma-only encoder with X — the chroma
        // walk does not perturb the luma walk.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut luma: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for (i, row) in luma.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = ((16 * i + j) & 0xFF) as u8;
            }
        }
        let y_only = encode_intra_frame_y(&luma, &seq, &fh).expect("encode succeeds");
        let input = Yuv420Frame16x16 {
            y: luma,
            u: [[128u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
            v: [[128u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        };
        let yuv = encode_intra_frame_yuv(&input, &seq, &fh).expect("encode succeeds");
        assert_eq!(yuv.committed_quants_y, y_only.committed_quants);
        assert_eq!(yuv.reconstructed_y, y_only.reconstructed_y);
    }

    // -----------------------------------------------------------------
    // Arc r228 — 13-mode intra prediction picker unit tests.
    // -----------------------------------------------------------------

    #[test]
    fn intra_mode_candidates_y_covers_all_13_intra_modes() {
        // The picker enumerates each §6.10.x Y intra mode exactly once.
        assert_eq!(INTRA_MODE_CANDIDATES_Y.len(), NUM_INTRA_MODES_Y);
        let mut seen = [false; NUM_INTRA_MODES_Y];
        for &m in &INTRA_MODE_CANDIDATES_Y {
            assert!(
                m < NUM_INTRA_MODES_Y,
                "mode {m} out of 0..{NUM_INTRA_MODES_Y}"
            );
            assert!(!seen[m], "mode {m} listed twice in INTRA_MODE_CANDIDATES_Y");
            seen[m] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "mode ordinal {i} missing from INTRA_MODE_CANDIDATES_Y");
        }
    }

    #[test]
    fn pick_best_intra_mode_4x4_y_flat_reconstructed_flat_input_picks_dc() {
        // When `reconstructed` is flat-128 and `input` is also flat-128
        // every mode produces a perfect match ⇒ SSD ties at zero ⇒ the
        // enumeration order tie-break gives DC_PRED.
        let recon: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let input = recon;
        // Pick at cell (1, 1) so both have_above and have_left fire (cell
        // (0, 0) is the no-neighbour corner — modes give different
        // fallbacks there).
        let (mode, _) = pick_best_intra_mode_4x4_y(&recon, &input, 4, 4);
        assert_eq!(mode as usize, DC_PRED);
    }

    #[test]
    fn pick_best_intra_mode_4x4_y_horizontal_gradient_input_prefers_v_pred() {
        // Vertical-uniform input (each col is constant across rows) is
        // perfectly predicted by V_PRED when the above row matches: set
        // the reconstructed above row to the same column gradient as the
        // input cell so V_PRED gives zero residual.
        let mut recon: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        // Row 3 (just above the cell at row0 = 4) carries the gradient.
        for (j, cell) in recon[3].iter_mut().enumerate() {
            *cell = (j * 16) as u8;
        }
        let mut input: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for row in input.iter_mut() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (j * 16) as u8;
            }
        }
        let (mode, pred) = pick_best_intra_mode_4x4_y(&recon, &input, 4, 4);
        // V_PRED gives a perfect match (every input row equals the above
        // row); SSD is 0 so the tie-break order picks V_PRED only if it's
        // strictly the first zero in enumeration order. DC_PRED comes
        // first; it would also produce a non-zero residual here (it
        // averages above + left rather than copying above), so V_PRED's
        // zero SSD beats DC_PRED's positive SSD.
        assert_eq!(mode as usize, V_PRED, "expected V_PRED, got mode {mode}");
        // Verify the picker returned the matching prediction (each row = above row).
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(pred[i * 4 + j], (((4 + j) * 16) as u8));
            }
        }
    }

    #[test]
    fn pick_best_intra_mode_4x4_y_vertical_gradient_input_prefers_h_pred() {
        // Mirror of the V_PRED test: horizontal-uniform input (each row
        // constant across cols), reconstructed left column carries the
        // gradient, so H_PRED gives zero residual.
        let mut recon: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        // Col 3 (just left of the cell at col0 = 4) carries the gradient.
        for (i, row) in recon.iter_mut().enumerate() {
            row[3] = (i * 16) as u8;
        }
        let mut input: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for (i, row) in input.iter_mut().enumerate() {
            for cell in row.iter_mut() {
                *cell = (i * 16) as u8;
            }
        }
        let (mode, _) = pick_best_intra_mode_4x4_y(&recon, &input, 4, 4);
        assert_eq!(mode as usize, H_PRED, "expected H_PRED, got mode {mode}");
    }

    #[test]
    fn predict_intra_mode_4x4_returns_none_for_invalid_mode() {
        let recon: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let (have_above, have_left, above_ext, left_ext, above_left) =
            derive_intra_neighbours_4x4_y(&recon, 4, 4);
        // Mode 13 is `UV_CFL_PRED` — not in the Y intra mode set.
        let out =
            predict_intra_mode_4x4(13, have_above, have_left, &above_ext, &left_ext, above_left);
        assert!(out.is_none());
    }

    #[test]
    fn encode_intra_frame_y_committed_y_modes_has_one_entry_per_cell() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let luma = [[128u8; FRAME_WIDTH]; FRAME_HEIGHT];
        let result = encode_intra_frame_y(&luma, &seq, &fh).unwrap();
        assert_eq!(result.committed_y_modes.len(), CELLS_WIDE * CELLS_HIGH);
    }

    #[test]
    fn encode_intra_frame_yuv_committed_y_modes_has_one_entry_per_luma_cell() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16::default();
        let result = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
        assert_eq!(result.committed_y_modes.len(), CELLS_WIDE * CELLS_HIGH);
    }
}
