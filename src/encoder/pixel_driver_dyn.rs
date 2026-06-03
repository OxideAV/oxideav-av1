//! Dynamic-extent pixel-space encoder driver — arc r230.
//!
//! Generalises the round-221..r229 [`crate::encoder::pixel_driver`]
//! (which was hard-coded to a 16×16 monochrome / 4:2:0 surface) to
//! accept frames of **arbitrary** width × height — both multiples of
//! `MIN_DIM = 8` (the §5.11.5 4:2:0 chroma cell width / height
//! constraint) and both ≤ `MAX_SB = 64` (one super-block per frame for
//! this arc — multi-super-block tiling is a follow-up).
//!
//! The composition of primitives is unchanged from the r223+r229 path
//! (§7.11.2.{2..6} 13-mode intra picker + forward WHT + forward
//! quantize + §5.11.39 coefficient writer + §5.11.4 partition tree +
//! §5.11.1 tile-group + §7.5 temporal-unit + IVF v0). The only deltas
//! are:
//!
//!   1. Vec-backed [`Yuv420Frame`] in place of the fixed-size
//!      `[[u8; 16]; 16]` [`crate::encoder::Yuv420Frame16x16`].
//!   2. SH+FH built dynamically from the requested width/height through
//!      [`build_intra_only_yuv420_8bit_seq`] +
//!      [`build_intra_only_yuv420_8bit_fh`].
//!   3. A recursive [`build_partition_tree`] helper that walks the
//!      smallest covering power-of-two super-block (one of 16/32/64),
//!      emitting `EncodeNode::dummy_oob()` for fully-out-of-frame
//!      quadrants. The existing §5.11.4 driver short-circuits these
//!      via its line-1 `r >= MiRows || c >= MiCols` early return.
//!
//! ## Scope (arc r230 + r233 + r234)
//!
//! * `width`, `height` ∈ {8, 16, 24, 32, 40, 48, 56, 64} (multiples
//!   of 8, ≤ 64). Width and height are independent — rectangular
//!   frame extents (`8×16`, `16×32`, `24×40`, `32×64`, `64×32`, ...)
//!   are supported. The §5.11.4 partition tree's per-quadrant
//!   `r >= mi_rows || c >= mi_cols` early-return swallows
//!   out-of-frame quadrants of the smallest power-of-two super-block
//!   that covers `max(mi_cols, mi_rows)` — see
//!   [`build_partition_tree`] for the recursion shape.
//! * `subsampling_x = subsampling_y = 1` (4:2:0), `bit_depth = 8`,
//!   `monochrome = false`.
//! * Intra-only, single tile, `base_q_idx ∈ 0..=255` (`= 0` selects
//!   the §5.9.2 `CodedLossless` arm + forward WHT on the leaf
//!   transform; `> 0` selects §7.13.3 forward DCT_DCT + §7.12.3
//!   forward quantize). `tx_size = TX_4X4` everywhere, no
//!   segmentation, no QM, no in-loop filters.
//! * 13-mode intra picker on luma + chroma (the r228/r229 picker)
//!   plus the r194/r232 §7.11.5.3 `UV_CFL_PRED` arm on chroma.
//!
//! ## What this arc does NOT do
//!
//! * Frames > 64×64 (multi-super-block tiling).
//! * Non-8 bit-depth, non-4:2:0 sampling, monochrome.
//! * Rectangular **transform sizes** (the §3 `TX_4X8` / `TX_8X4` /
//!   `TX_8X16` / `TX_16X8` family). The leaf transform is `TX_4X4`
//!   everywhere; only the **frame extent** is allowed to be
//!   rectangular.
//! * §5.11.18 inter mode_info, per-segment / per-block delta_q, RD
//!   picker.
//!
//! ## Spec provenance
//!
//! Same set as the 16×16 driver — `docs/video/av1/av1-spec.txt`
//! §5.9.5 (`frame_size`), §5.9.9 (`MiCols` / `MiRows` derivation),
//! §5.11.1 (tile-group), §5.11.4 (`decode_partition`), §5.11.5
//! (`decode_block`), §5.11.39 (`coefficients`), §7.5 (temporal unit),
//! §7.11.2.{2..6} (intra prediction), §7.12.3 (`dequantize_step1`),
//! §7.13 (inverse transform).

use crate::cdf::{
    dequantize_step1, partition_subsize, QuantizerParams, TileCdfContext, TileGeometry,
    BLOCK_16X16, BLOCK_32X32, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8, DCT_DCT, DC_PRED, MI_WIDTH_LOG2,
    NUM_4X4_BLOCKS_WIDE, PARTITION_NONE, PARTITION_SPLIT, TX_4X4, TX_CLASS_2D, UV_CFL_PRED,
};
use crate::encoder::forward_quantize::forward_quantize;
use crate::encoder::forward_transform_2d::forward_transform_2d;
use crate::encoder::forward_wht::forward_wht_4x4;
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::obu::ObuFrame;
use crate::encoder::partition_tree::{
    write_partition_tree, EncodeBlock, EncodeNode, PartitionTreeWriter, PlaneCoefficients,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::tile_group_obu::{write_tile_group_obu, TileGroupObu, TilePayload};
use crate::frame_header::{FrameHeader, FrameSize, FrameType, ALL_FRAMES_PUB, PRIMARY_REF_NONE};
use crate::obu::ObuType;
use crate::scan::get_default_scan;
use crate::sequence_header::{
    ColorConfig, OperatingPoint, SequenceHeader, CP_UNSPECIFIED, CSP_UNKNOWN, MC_UNSPECIFIED,
    SELECT_INTEGER_MV, SELECT_SCREEN_CONTENT_TOOLS, TC_UNSPECIFIED,
};
use crate::tile_info::TileInfo;
use crate::transform::inverse_transform_2d;
use crate::uncompressed_header_tail::{
    CdefParams, DeltaLfParams, DeltaQParams, FilmGrainParams, FrameRestorationType,
    GlobalMotionParams, LoopFilterParams, LrParams, QuantizationParams, SegmentationParams, TxMode,
};
use crate::Error;

// Re-use the per-cell prediction helpers from the fixed-size driver via
// transparent slice adapters. The §7.11.2.{2..6} kernels themselves
// don't read the underlying plane (they only see the head-extended
// neighbour buffers the caller passes), so the 16×16-bound helpers are
// adapted by re-implementing the §7.11.2.1 prologue against the
// dynamic-extent plane below — keeping every kernel hit identical to
// the fixed-size driver's hit.
use crate::cdf::{
    predict_intra_d_mode, predict_intra_dc_pred, predict_intra_h_pred, predict_intra_paeth_pred,
    predict_intra_smooth_h_pred, predict_intra_smooth_pred, predict_intra_smooth_v_pred,
    predict_intra_v_pred, D113_PRED, D135_PRED, D157_PRED, D203_PRED, D45_PRED, D67_PRED, H_PRED,
    PAETH_PRED, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED, V_PRED,
};

/// §5.11.5 4:2:0 chroma constraint: width / height must be multiples of
/// 8 so the chroma plane (half-resolution) is itself a multiple of
/// 4 (the BLOCK_4X4 leaf walk). The dyn driver enforces this at
/// construction time.
pub const MIN_DIM: u32 = 8;
/// Largest super-block this arc handles in one root partition tree
/// (`BLOCK_64X64`). Multi-super-block tiling is the r207 follow-up
/// arc (`MAX_DIM_Y_MULTI_SB = 128`) on the Y-only path; the
/// `MAX_DIM = 64` ceiling is preserved for the single-SB YUV + Y
/// drivers so existing IVF outputs remain byte-for-byte identical
/// across r207. Width and height are independently bounded by this
/// constant — the per-quadrant out-of-frame sentinel in
/// [`build_partition_tree`] lets a rectangular extent
/// (e.g. `16 × 64` ⇒ `mi 4 × 16`) ride the same `BLOCK_64X64` root as
/// a square 64×64 frame would.
pub const MAX_DIM: u32 = 64;
/// r207 — multi-super-block ceiling for the §5.11.1 SB-grid walk on
/// the Y-only (mono) dyn driver. Two SBs in each axis (16-mi each at
/// `use_128x128_superblock = false`, sbSize4 = 16). Width and height
/// remain independent. Each SB is a single `BLOCK_64X64` root
/// partition tree per §5.11.1 line 13 (`sbSize = BLOCK_64X64`).
pub const MAX_DIM_Y_MULTI_SB: u32 = 128;
/// r207 — §5.11.1 sbSize4 (the number of 4×4 mi-units per
/// super-block) for `use_128x128_superblock = false`. The dyn
/// multi-SB walks iterate `(sb_r, sb_c) ∈ (0..mi_rows).step_by(16)
/// × (0..mi_cols).step_by(16)` in row-major order.
pub const SB_SIZE4_64: u32 = 16;

/// Dynamic-extent 4:2:0 YUV input the dyn driver consumes. Plane data
/// is Vec-backed so the same struct admits any allowed (width,
/// height) combination.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420Frame {
    /// Luma width in pixels. Must be a multiple of [`MIN_DIM`] and ≤
    /// [`MAX_DIM`].
    pub width: u32,
    /// Luma height in pixels. Same constraints as [`Self::width`].
    pub height: u32,
    /// Luma plane (`Y`), row-major, length `width * height`.
    pub y: Vec<u8>,
    /// First chroma plane (`U` / `Cb`) at half horizontal + vertical
    /// resolution; length `(width / 2) * (height / 2)`.
    pub u: Vec<u8>,
    /// Second chroma plane (`V` / `Cr`); same shape as `u`.
    pub v: Vec<u8>,
}

impl Yuv420Frame {
    /// All-`fill` mid-grey input. Useful for tests + as the default
    /// constructor; every plane is set to the same value.
    #[must_use]
    pub fn filled(width: u32, height: u32, fill: u8) -> Self {
        let cw = width / 2;
        let ch = height / 2;
        Self {
            width,
            height,
            y: vec![fill; (width * height) as usize],
            u: vec![fill; (cw * ch) as usize],
            v: vec![fill; (cw * ch) as usize],
        }
    }

    /// Chroma plane width — `width / 2` per the 4:2:0 sampling pattern.
    #[must_use]
    pub fn chroma_width(&self) -> u32 {
        self.width / 2
    }

    /// Chroma plane height — `height / 2` per the 4:2:0 sampling
    /// pattern.
    #[must_use]
    pub fn chroma_height(&self) -> u32 {
        self.height / 2
    }

    /// Validate the input's dimensions + plane lengths. Returns
    /// [`Error::PartitionWalkOutOfRange`] on any mismatch (a wrong
    /// shape would corrupt the encoder's per-leaf walk).
    pub fn validate(&self) -> Result<(), Error> {
        if self.width < MIN_DIM
            || self.height < MIN_DIM
            || self.width > MAX_DIM
            || self.height > MAX_DIM
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.width % MIN_DIM != 0 || self.height % MIN_DIM != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let expected_y = (self.width * self.height) as usize;
        let expected_uv = (self.chroma_width() * self.chroma_height()) as usize;
        if self.y.len() != expected_y || self.u.len() != expected_uv || self.v.len() != expected_uv
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// Result of [`encode_intra_frame_yuv_dyn`]. Bundles the IVF bytes
/// with all three reconstructed planes (encoder-internal walk; for the
/// lossless WHT arm these equal the input plane-for-plane).
#[derive(Debug, Clone)]
pub struct EncodedFrameDyn {
    /// IVF bytes — file header + one IVF frame.
    pub ivf_bytes: Vec<u8>,
    /// §7.5 temporal unit bytes (TD + SH + FH + TileGroup OBU).
    pub temporal_unit_bytes: Vec<u8>,
    /// Reconstructed Y plane (row-major, length `width * height`).
    pub reconstructed_y: Vec<u8>,
    /// Reconstructed U plane (row-major, length `(width/2) * (height/2)`).
    pub reconstructed_u: Vec<u8>,
    /// Reconstructed V plane.
    pub reconstructed_v: Vec<u8>,
    /// SequenceHeader the driver synthesised + emitted.
    pub seq: SequenceHeader,
    /// FrameHeader the driver synthesised + emitted.
    pub fh: FrameHeader,
}

// ----------------------------------------------------------------------
// SH / FH builders — synthesise the minimal valid descriptors for a
// dynamic-extent intra-only frame at base_q_idx = 0.
// ----------------------------------------------------------------------

/// Build the minimal 4:2:0 8-bit `SequenceHeader` for an intra-only
/// frame whose maximum extent is `(max_width, max_height)`. The
/// resulting `sequence_header_obu()` payload accepts a frame of any
/// width × height ≤ `(max_width, max_height)` — the per-frame size is
/// carried by the FrameHeader's `frame_size`.
///
/// The dyn driver always invokes this with `(max_width, max_height) ==
/// (width, height)` since arc r230 supports only one frame per IVF.
///
/// Mirrors the `tiny_16x16_profile0()` helper in
/// [`crate::encoder::sequence_obu`] tests (the same conformant minimal
/// shape the parser round-trips byte-for-byte) but with the requested
/// `max_frame_*` derived from the caller's dimensions.
#[must_use]
pub fn build_intra_only_yuv420_8bit_seq(max_width: u32, max_height: u32) -> SequenceHeader {
    debug_assert!((1..=0xFFFF).contains(&max_width));
    debug_assert!((1..=0xFFFF).contains(&max_height));
    let max_w_minus_1 = max_width - 1;
    let max_h_minus_1 = max_height - 1;
    // §5.5.1 frame_width_bits_minus_1 must be wide enough to fit
    // `max_frame_width_minus_1`. The minimum bit count is
    // `bits_to_represent(max_w_minus_1) - 1`. Use 0..=15.
    let bits_for = |v: u32| -> u8 {
        if v == 0 {
            0
        } else {
            (32 - v.leading_zeros() - 1) as u8
        }
    };
    let frame_width_bits_minus_1 = bits_for(max_w_minus_1);
    let frame_height_bits_minus_1 = bits_for(max_h_minus_1);

    SequenceHeader {
        seq_profile: 0,
        still_picture: false,
        reduced_still_picture_header: false,
        timing_info_present_flag: false,
        timing_info: None,
        decoder_model_info_present_flag: false,
        decoder_model_info: None,
        initial_display_delay_present_flag: false,
        operating_points_cnt_minus_1: 0,
        operating_points: vec![OperatingPoint {
            operating_point_idc: 0,
            seq_level_idx: 0,
            seq_tier: 0,
            decoder_model_present_for_this_op: false,
            operating_parameters_info: None,
            initial_display_delay_present_for_this_op: false,
            initial_display_delay_minus_1: None,
        }],
        frame_width_bits_minus_1,
        frame_height_bits_minus_1,
        max_frame_width_minus_1: max_w_minus_1,
        max_frame_height_minus_1: max_h_minus_1,
        frame_id_numbers_present_flag: false,
        delta_frame_id_length_minus_2: 0,
        additional_frame_id_length_minus_1: 0,
        // use_128x128_superblock = false ⇒ sb_size = 64 ⇒ matches our
        // single-super-block-per-frame partition tree.
        use_128x128_superblock: false,
        enable_filter_intra: false,
        enable_intra_edge_filter: false,
        enable_interintra_compound: false,
        enable_masked_compound: false,
        enable_warped_motion: false,
        enable_dual_filter: false,
        enable_order_hint: false,
        enable_jnt_comp: false,
        enable_ref_frame_mvs: false,
        seq_force_screen_content_tools: SELECT_SCREEN_CONTENT_TOOLS,
        seq_force_integer_mv: SELECT_INTEGER_MV,
        order_hint_bits: 0,
        enable_superres: false,
        enable_cdef: false,
        enable_restoration: false,
        color_config: ColorConfig {
            high_bitdepth: false,
            twelve_bit: false,
            bit_depth: 8,
            mono_chrome: false,
            num_planes: 3,
            color_description_present_flag: false,
            color_primaries: CP_UNSPECIFIED,
            transfer_characteristics: TC_UNSPECIFIED,
            matrix_coefficients: MC_UNSPECIFIED,
            color_range: false,
            subsampling_x: true,
            subsampling_y: true,
            chroma_sample_position: CSP_UNKNOWN,
            separate_uv_delta_q: false,
        },
        film_grain_params_present: false,
        bits_consumed: 0,
    }
}

/// Build the minimal intra-only `FrameHeader` for the dyn driver at
/// `base_q_idx = 0` (lossless), `Only4x4` TxMode, in-loop filters
/// disabled. Mirrors the `synthetic_intra_round_trip_lossless` test
/// shape in [`crate::encoder::frame_obu`].
///
/// Equivalent to [`build_intra_only_yuv420_8bit_fh_with_q`] with
/// `base_q_idx = 0`; kept for back-compat with arc-r230 callers.
#[must_use]
pub fn build_intra_only_yuv420_8bit_fh(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
) -> FrameHeader {
    build_intra_only_yuv420_8bit_fh_with_q(seq, width, height, 0)
}

/// Build the minimal intra-only `FrameHeader` for the dyn driver at
/// the caller-supplied `base_q_idx`, `Only4x4` TxMode, in-loop filters
/// disabled. `base_q_idx == 0` is the §5.9.2 `CodedLossless` arm
/// (lossless WHT path on the leaf transform); `base_q_idx > 0` selects
/// the lossy DCT_DCT path on the leaf transform.
///
/// `delta_q_present` stays `false` regardless — the dyn driver's
/// scope is "one global qindex for the whole frame". §5.9.13's
/// per-segment deltas, per-plane DC offsets, and per-block §5.11.34
/// `read_delta_qindex` are all out-of-scope this arc.
#[must_use]
pub fn build_intra_only_yuv420_8bit_fh_with_q(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> FrameHeader {
    let fs = FrameSize {
        frame_width: width,
        frame_height: height,
        render_width: width,
        render_height: height,
        superres_denom: 8, // SUPERRES_NUM
        upscaled_width: width,
        mi_cols: 2 * ((width + 7) >> 3),
        mi_rows: 2 * ((height + 7) >> 3),
        use_superres: false,
        coded_denom: 0,
        render_and_frame_size_different: false,
    };
    let ti = TileInfo {
        uniform_tile_spacing_flag: true,
        tile_cols: 1,
        tile_rows: 1,
        tile_cols_log2: 0,
        tile_rows_log2: 0,
        context_update_tile_id: 0,
        tile_size_bytes: 1,
        mi_col_starts: vec![0, fs.mi_cols],
        mi_row_starts: vec![0, fs.mi_rows],
    };
    let qp = QuantizationParams {
        base_q_idx,
        delta_q_y_dc: 0,
        diff_uv_delta: false,
        delta_q_u_dc: 0,
        delta_q_u_ac: 0,
        delta_q_v_dc: 0,
        delta_q_v_ac: 0,
        using_qmatrix: false,
        qm_y: 0,
        qm_u: 0,
        qm_v: 0,
    };
    FrameHeader {
        show_existing_frame: false,
        frame_to_show_map_idx: None,
        display_frame_id: None,
        frame_type: FrameType::Key,
        frame_is_intra: true,
        show_frame: true,
        showable_frame: false,
        error_resilient_mode: true,
        disable_cdf_update: false,
        allow_screen_content_tools: seq.seq_force_screen_content_tools != 0,
        force_integer_mv: true,
        current_frame_id: 0,
        frame_size_override_flag: false,
        order_hint: 0,
        primary_ref_frame: PRIMARY_REF_NONE,
        refresh_frame_flags: ALL_FRAMES_PUB,
        frame_size: Some(fs),
        allow_intrabc: false,
        disable_frame_end_update_cdf: false,
        tile_info: Some(ti),
        quantization_params: Some(qp),
        segmentation_params: Some(SegmentationParams::disabled()),
        delta_q_params: Some(DeltaQParams::default()),
        delta_lf_params: Some(DeltaLfParams::default()),
        loop_filter_params: Some(LoopFilterParams::short_circuit()),
        cdef_params: Some(CdefParams::short_circuit()),
        lr_params: Some(LrParams {
            frame_restoration_type: [FrameRestorationType::None; 3],
            uses_lr: false,
            uses_chroma_lr: false,
            lr_unit_shift: 0,
            lr_uv_shift: 0,
            loop_restoration_size: [0; 3],
            short_circuited: true,
        }),
        // §5.9.21 `read_tx_mode()`: `ONLY_4X4` is only valid under the
        // §5.9.2 `CodedLossless` arm (`base_q_idx == 0` + all delta_q
        // zero). For lossy quant (`base_q_idx > 0`) the FH writer
        // enforces `TxModeLargest` / `TxModeSelect`. The dyn driver
        // produces only TX_4X4 leaves at this arc; `TxModeLargest`
        // surfaces "the largest TX size each block can take" which the
        // decoder honours block-by-block — since every BLOCK_4X4 leaf's
        // largest TX is TX_4X4, the decoder ends up reading the same
        // coefficient blocks the encoder wrote.
        tx_mode: Some(if base_q_idx == 0 {
            TxMode::Only4x4
        } else {
            TxMode::TxModeLargest
        }),
        reference_select: Some(false),
        skip_mode_present: Some(false),
        allow_warped_motion: Some(false),
        reduced_tx_set: Some(false),
        global_motion_params: Some(GlobalMotionParams::identity()),
        film_grain_params: Some(FilmGrainParams::reset()),
        inter_refs: None,
        bits_consumed: 0,
    }
}

// ----------------------------------------------------------------------
// §7.11.2.1 neighbour-array prologue for one 4×4 cell against a
// dynamic-extent plane. Mirror of the fixed-size driver's helpers in
// `crate::encoder::pixel_driver`. The kernel calls themselves are
// identical — only the read of the running reconstructed plane changes
// shape.
// ----------------------------------------------------------------------

#[inline]
fn pix(plane: &[u8], stride: usize, r: usize, c: usize) -> u16 {
    plane[r * stride + c] as u16
}

fn derive_intra_neighbours_4x4(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    row0: usize,
    col0: usize,
) -> (u8, u8, [u16; 10], [u16; 10], u16) {
    let w = 4usize;
    let h = 4usize;
    let bit_depth = 8u8;
    let have_above = (row0 > 0) as u8;
    let have_left = (col0 > 0) as u8;
    let stride = plane_width;

    let above_left: u16 = if have_above != 0 && have_left != 0 {
        pix(plane, stride, row0 - 1, col0 - 1)
    } else if have_above != 0 {
        pix(plane, stride, row0 - 1, col0)
    } else if have_left != 0 {
        pix(plane, stride, row0, col0 - 1)
    } else {
        1u16 << (bit_depth - 1)
    };

    let mut above_ext = [0u16; 10];
    above_ext[0] = above_left;
    above_ext[1] = above_left;
    if have_above != 0 {
        for k in 0..(w + h) {
            let col = (col0 + k).min(plane_width - 1);
            above_ext[2 + k] = pix(plane, stride, row0 - 1, col);
        }
    } else if have_left != 0 {
        let sample = pix(plane, stride, row0, col0 - 1);
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
            let row = (row0 + k).min(plane_height - 1);
            left_ext[2 + k] = pix(plane, stride, row, col0 - 1);
        }
    } else if have_above != 0 {
        let sample = pix(plane, stride, row0 - 1, col0);
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

/// 13 §6.10.x intra mode candidates the picker enumerates, in tie-break
/// order (DC_PRED first).
const INTRA_MODE_CANDIDATES: [usize; 13] = [
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
        m if m == V_PRED => predict_intra_v_pred(w, h, above_row, &mut pred16).ok()?,
        m if m == H_PRED => predict_intra_h_pred(w, h, left_col, &mut pred16).ok()?,
        m if (D45_PRED..=D67_PRED).contains(&m) => {
            predict_intra_d_mode(m, 0, w, h, 0, 0, above_ext, left_ext, &mut pred16).ok()?;
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

fn pick_best_intra_mode_4x4(
    reconstructed: &[u8],
    input: &[u8],
    plane_width: usize,
    plane_height: usize,
    row0: usize,
    col0: usize,
) -> (u8, [u8; 16]) {
    let (have_above, have_left, above_ext, left_ext, above_left) =
        derive_intra_neighbours_4x4(reconstructed, plane_width, plane_height, row0, col0);
    let mut best_mode = DC_PRED as u8;
    let mut best_pred = [0u8; 16];
    let mut best_ssd: u64 = u64::MAX;
    for &mode in &INTRA_MODE_CANDIDATES {
        let Some(pred) = predict_intra_mode_4x4(
            mode, have_above, have_left, &above_ext, &left_ext, above_left,
        ) else {
            continue;
        };
        let mut ssd: u64 = 0;
        for i in 0..4 {
            for j in 0..4 {
                let p = input[(row0 + i) * plane_width + (col0 + j)] as i32;
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

/// §7.11.5.3 subsampled-luma + lumaAvg derivation for one 4:2:0 TX_4X4
/// chroma block on a Vec-backed luma plane.
///
/// Mirror of [`crate::encoder::pixel_driver::cfl_subsampled_luma_4x4_420`]
/// but reading from a dynamic-extent plane (slice + stride + height
/// clamp) rather than the fixed-size `[[u8; FRAME_WIDTH]; FRAME_HEIGHT]`
/// array. The result is `(L[16], lumaAvg)` with `L[i][j] = t << 1`
/// (3-fractional-bit fixed-point for 4:2:0) and
/// `lumaAvg = Round2(sum, 4)` for TX_4X4.
///
/// `(crow0_y, ccol0_y)` is the luma-pixel origin of the §7.11.5.3 luma
/// window (= chroma origin × 2). `(luma_w, luma_h)` are the luma plane
/// extents (used for the §7.11.5.3 `MaxLumaW - (1 << subX)` /
/// `MaxLumaH - (1 << subY)` right/bottom clamps).
fn cfl_subsampled_luma_4x4_420_dyn(
    recon_y: &[u8],
    luma_w: usize,
    luma_h: usize,
    crow0_chroma: usize,
    ccol0_chroma: usize,
) -> ([i32; 16], i32) {
    let mut l_arr = [0i32; 16];
    let mut sum: i32 = 0;
    for i in 0..4 {
        let mut luma_y0 = (crow0_chroma + i) << 1;
        if luma_y0 > luma_h - 2 {
            luma_y0 = luma_h - 2;
        }
        for j in 0..4 {
            let mut luma_x0 = (ccol0_chroma + j) << 1;
            if luma_x0 > luma_w - 2 {
                luma_x0 = luma_w - 2;
            }
            let mut t: i32 = 0;
            for dy in 0..=1usize {
                for dx in 0..=1usize {
                    t += recon_y[(luma_y0 + dy) * luma_w + (luma_x0 + dx)] as i32;
                }
            }
            let v = t << 1;
            l_arr[i * 4 + j] = v;
            sum += v;
        }
    }
    let luma_avg = (sum + 8) >> 4;
    (l_arr, luma_avg)
}

/// §3 `Round2Signed(x, n)` — symmetric rounding for signed inputs
/// (`x < 0` arm round-half-away-from-zero, matching the spec's
/// `if x < 0 then -Round2(-x, n) else Round2(x, n)` definition).
#[inline]
fn round2_signed(x: i64, n: u32) -> i64 {
    let half: i64 = 1i64 << (n - 1);
    if x < 0 {
        -(((-x) + half) >> n)
    } else {
        (x + half) >> n
    }
}

/// §7.11.5.3 CFL chroma prediction — `dc_pred[k] + Round2Signed(α *
/// (L[k] - lumaAvg), 6)` clipped to byte. `dc_pred` is the §7.11.2.5
/// DC_PRED chroma prediction this block would have produced on the
/// chroma-only arm; `l_arr` / `luma_avg` come from
/// [`cfl_subsampled_luma_4x4_420_dyn`]. Identical in shape to the
/// fixed-size driver's helper of the same name.
fn cfl_predict_4x4_for_plane_dyn(
    dc_pred: &[u8; 16],
    l_arr: &[i32; 16],
    luma_avg: i32,
    alpha: i8,
) -> [u8; 16] {
    let mut out = [0u8; 16];
    for k in 0..16 {
        let scaled = round2_signed((alpha as i64) * ((l_arr[k] - luma_avg) as i64), 6);
        let p = dc_pred[k] as i64 + scaled;
        out[k] = p.clamp(0, 255) as u8;
    }
    out
}

/// §5.11.45 compact (αU, αV) search grid the dyn driver enumerates for
/// `UV_CFL_PRED`. Same shape as the fixed-size driver — small
/// magnitudes `{±1, ±2, ±4}` per channel with single-channel and
/// equal-channel combinations (20 candidates total). The all-zero pair
/// is excluded per §6.10.36 (redundant with `UV_DC_PRED`).
const CFL_ALPHA_CANDIDATES: &[(i8, i8)] = &[
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (1, -1),
    (-1, 1),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
    (2, 2),
    (-2, -2),
    (2, -2),
    (-2, 2),
    (4, 0),
    (-4, 0),
    (0, 4),
    (0, -4),
];

/// Joint U+V chroma intra-mode picker, dyn-extent edition.
///
/// Enumerates the 13 §6.10.x intra modes (DC_PRED..PAETH_PRED) against
/// the two chroma planes and returns the lowest combined-SSD pick. The
/// per-§5.11.22 line-3 contract is one shared `intra_uv_mode()` symbol
/// per leaf (both planes use the same mode pick) — the picker
/// minimises the joint residual energy rather than choosing planes
/// independently.
///
/// When `recon_y` / `luma_*` extents are supplied (always — the dyn
/// arc keeps luma context live for §7.11.5.3 CFL evaluation), the
/// picker also evaluates `UV_CFL_PRED` over [`CFL_ALPHA_CANDIDATES`]
/// against the §7.11.5.3 subsampled-luma window built on top of the
/// `DC_PRED` chroma base. When a CFL (αU, αV) beats every §6.10.x
/// intra mode on combined SSD, the picker returns `UV_CFL_PRED = 13`
/// as the mode with `Some((αU, αV))` in the fourth tuple slot;
/// otherwise that slot is `None`. `DC_PRED` is the tie-breaker
/// (enumeration-order win).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn pick_best_intra_mode_4x4_chroma_joint(
    recon_y: &[u8],
    recon_u: &[u8],
    recon_v: &[u8],
    input_u: &[u8],
    input_v: &[u8],
    luma_w: usize,
    luma_h: usize,
    plane_width: usize,
    plane_height: usize,
    row0: usize,
    col0: usize,
) -> (u8, [u8; 16], [u8; 16], Option<(i8, i8)>) {
    let (ha_u, hl_u, above_u, left_u, al_u) =
        derive_intra_neighbours_4x4(recon_u, plane_width, plane_height, row0, col0);
    let (ha_v, hl_v, above_v, left_v, al_v) =
        derive_intra_neighbours_4x4(recon_v, plane_width, plane_height, row0, col0);
    let mut best_mode = DC_PRED as u8;
    let mut best_pred_u = [0u8; 16];
    let mut best_pred_v = [0u8; 16];
    let mut best_alpha: Option<(i8, i8)> = None;
    let mut best_ssd: u64 = u64::MAX;
    // Capture the DC_PRED chroma prediction explicitly so the CFL arm
    // can reuse it as the §7.11.5.3 `dc` base.
    let mut dc_pred_u = [0u8; 16];
    let mut dc_pred_v = [0u8; 16];
    for &mode in &INTRA_MODE_CANDIDATES {
        let Some(pred_u) = predict_intra_mode_4x4(mode, ha_u, hl_u, &above_u, &left_u, al_u) else {
            continue;
        };
        let Some(pred_v) = predict_intra_mode_4x4(mode, ha_v, hl_v, &above_v, &left_v, al_v) else {
            continue;
        };
        if mode == DC_PRED {
            dc_pred_u = pred_u;
            dc_pred_v = pred_v;
        }
        let mut ssd: u64 = 0;
        for i in 0..4 {
            for j in 0..4 {
                let pu = input_u[(row0 + i) * plane_width + (col0 + j)] as i32;
                let qu = pred_u[i * 4 + j] as i32;
                let du = (pu - qu) as i64;
                ssd += (du * du) as u64;
                let pv = input_v[(row0 + i) * plane_width + (col0 + j)] as i32;
                let qv = pred_v[i * 4 + j] as i32;
                let dv = (pv - qv) as i64;
                ssd += (dv * dv) as u64;
            }
        }
        if ssd < best_ssd {
            best_ssd = ssd;
            best_mode = mode as u8;
            best_pred_u = pred_u;
            best_pred_v = pred_v;
            best_alpha = None;
        }
    }
    // r232: UV_CFL_PRED arm. §7.11.5.3 dc base = the DC_PRED chroma
    // prediction captured above. Compute the §7.11.5.3 subsampled-luma
    // window + lumaAvg once, then enumerate the αU/αV candidate grid.
    let (l_arr, luma_avg) = cfl_subsampled_luma_4x4_420_dyn(recon_y, luma_w, luma_h, row0, col0);
    for &(au, av) in CFL_ALPHA_CANDIDATES {
        let pred_u = cfl_predict_4x4_for_plane_dyn(&dc_pred_u, &l_arr, luma_avg, au);
        let pred_v = cfl_predict_4x4_for_plane_dyn(&dc_pred_v, &l_arr, luma_avg, av);
        let mut ssd: u64 = 0;
        for i in 0..4 {
            for j in 0..4 {
                let pu = input_u[(row0 + i) * plane_width + (col0 + j)] as i32;
                let qu = pred_u[i * 4 + j] as i32;
                let du = (pu - qu) as i64;
                ssd += (du * du) as u64;
                let pv = input_v[(row0 + i) * plane_width + (col0 + j)] as i32;
                let qv = pred_v[i * 4 + j] as i32;
                let dv = (pv - qv) as i64;
                ssd += (dv * dv) as u64;
            }
        }
        if ssd < best_ssd {
            best_ssd = ssd;
            best_mode = UV_CFL_PRED as u8;
            best_pred_u = pred_u;
            best_pred_v = pred_v;
            best_alpha = Some((au, av));
        }
    }
    (best_mode, best_pred_u, best_pred_v, best_alpha)
}

// ----------------------------------------------------------------------
// Partition-tree construction.
// ----------------------------------------------------------------------

/// Smallest power-of-two super-block size (one of 16/32/64) that
/// contains the (mi_cols × mi_rows) frame extent. Used as the root
/// `b_size` for the §5.11.4 partition tree.
#[must_use]
pub fn root_super_block(mi_cols: u32, mi_rows: u32) -> usize {
    let max_mi = mi_cols.max(mi_rows);
    if max_mi <= 4 {
        BLOCK_16X16
    } else if max_mi <= 8 {
        BLOCK_32X32
    } else {
        // Up to BLOCK_64X64 (mi = 16).
        BLOCK_64X64
    }
}

/// `EncodeBlock` shape per BLOCK_4X4 leaf. Bundles the §6.10.x Y intra
/// mode pick, the optional shared `uv_mode` (Some on §5.11.5 HasChroma
/// leaves), and the per-plane `PlaneCoefficients` the §5.11.39 writer
/// consumes. Filled in by the per-leaf encode walk and consumed by
/// `build_partition_tree` to produce the actual `EncodeNode` graph.
struct LeafEnc {
    y_mode: u8,
    uv_mode: Option<u8>,
    /// r232: §5.11.45 `CflAlphaU` for the leaf when `uv_mode ==
    /// UV_CFL_PRED`; `None` for the §6.10.x intra modes.
    cfl_alpha_u: Option<i8>,
    /// r232: §5.11.45 `CflAlphaV` for the leaf when `uv_mode ==
    /// UV_CFL_PRED`; `None` for the §6.10.x intra modes.
    cfl_alpha_v: Option<i8>,
    coefficients: Vec<PlaneCoefficients>,
}

/// Recursively build the `EncodeNode` graph for one frame extent. The
/// root is `b_size` (one of BLOCK_16X16 / BLOCK_32X32 / BLOCK_64X64);
/// each level SPLIT-recurses on its four quadrants until reaching
/// BLOCK_4X4 leaves. Quadrants that fall entirely outside the frame
/// (`r >= mi_rows || c >= mi_cols`) get an `EncodeNode::dummy_oob()`
/// sentinel; the §5.11.4 driver's line-1 early return swallows them
/// without inspecting their contents.
///
/// `leaf_for(r, c)` returns the per-leaf encode bundle for the
/// BLOCK_4X4 cell at mi `(r, c)`. The function is invoked only for
/// in-frame leaves.
fn build_partition_tree(
    b_size: usize,
    r: u32,
    c: u32,
    mi_rows: u32,
    mi_cols: u32,
    leaf_for: &mut dyn FnMut(u32, u32) -> EncodeBlock,
) -> EncodeNode {
    // Fully out-of-frame ⇒ sentinel.
    if r >= mi_rows || c >= mi_cols {
        return EncodeNode::dummy_oob();
    }
    if b_size == BLOCK_4X4 {
        return EncodeNode::Leaf(leaf_for(r, c));
    }
    let sub_size = partition_subsize(PARTITION_SPLIT, b_size).expect("valid SPLIT sub-size");
    let half = (NUM_4X4_BLOCKS_WIDE[b_size] as u32) >> 1;
    let nw = Box::new(build_partition_tree(
        sub_size, r, c, mi_rows, mi_cols, leaf_for,
    ));
    let ne = Box::new(build_partition_tree(
        sub_size,
        r,
        c + half,
        mi_rows,
        mi_cols,
        leaf_for,
    ));
    let sw = Box::new(build_partition_tree(
        sub_size,
        r + half,
        c,
        mi_rows,
        mi_cols,
        leaf_for,
    ));
    let se = Box::new(build_partition_tree(
        sub_size,
        r + half,
        c + half,
        mi_rows,
        mi_cols,
        leaf_for,
    ));
    EncodeNode::Split([nw, ne, sw, se])
}

/// §5.11.4 dispatch-order BLOCK_4X4 cell walk for a (mi_rows ×
/// mi_cols) frame rooted at a single `b_size` super-block at the
/// origin. Used both by the encoder driver (to choose the order to
/// build `LeafEnc[]`s) and by tests asserting in-frame coverage.
///
/// The order is the depth-first NW/NE/SW/SE recursion of
/// [`build_partition_tree`]. Out-of-frame leaves are skipped.
#[must_use]
pub fn dispatch_order_leaves(b_size: usize, mi_rows: u32, mi_cols: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    fn walk(b_size: usize, r: u32, c: u32, mi_rows: u32, mi_cols: u32, out: &mut Vec<(u32, u32)>) {
        if r >= mi_rows || c >= mi_cols {
            return;
        }
        if b_size == BLOCK_4X4 {
            out.push((r, c));
            return;
        }
        let sub_size = partition_subsize(PARTITION_SPLIT, b_size).expect("valid SPLIT sub-size");
        let half = (NUM_4X4_BLOCKS_WIDE[b_size] as u32) >> 1;
        walk(sub_size, r, c, mi_rows, mi_cols, out);
        walk(sub_size, r, c + half, mi_rows, mi_cols, out);
        walk(sub_size, r + half, c, mi_rows, mi_cols, out);
        walk(sub_size, r + half, c + half, mi_rows, mi_cols, out);
    }
    walk(b_size, 0, 0, mi_rows, mi_cols, &mut out);
    out
}

// ----------------------------------------------------------------------
// Encoder driver.
// ----------------------------------------------------------------------

/// Encode one 4:2:0 YUV intra-only frame at `base_q_idx = 0` for any
/// allowed (width, height). Returns the IVF bytes plus the
/// encoder-internal reconstruction (bit-exact equal to the input on
/// the lossless WHT arm).
///
/// The function synthesises its own SequenceHeader + FrameHeader via
/// [`build_intra_only_yuv420_8bit_seq`] +
/// [`build_intra_only_yuv420_8bit_fh`] — the caller does not need to
/// supply a fixture descriptor.
///
/// Thin wrapper over [`encode_intra_frame_yuv_dyn_with_q`] with
/// `base_q_idx = 0`; preserved for back-compat with all arc-r230..r232
/// callers (the bit-exact lossless WHT roundtrip property is unchanged).
///
/// ## Errors
///
/// * `input.validate()` fails (dimensions out of range / wrong plane
///   length) ⇒ [`Error::PartitionWalkOutOfRange`].
/// * Any internal partition-tree / coefficient writer overflow.
pub fn encode_intra_frame_yuv_dyn(input: &Yuv420Frame) -> Result<EncodedFrameDyn, Error> {
    encode_intra_frame_yuv_dyn_with_q(input, 0)
}

/// Encode one 4:2:0 YUV intra-only frame at the caller-supplied
/// `base_q_idx` (round 233 — first lossy-quant arc on the dyn
/// driver). `base_q_idx == 0` selects the §5.9.2 `CodedLossless` arm
/// (forward WHT + bit-exact lossless WHT roundtrip on the leaf
/// transform); `base_q_idx > 0` selects the §7.13.3 forward DCT path
/// with §7.12.3 forward quantize → decoder dequantize → §7.13.3
/// inverse DCT.
///
/// Lossy contract: at `base_q_idx > 0` the encoder's running
/// reconstructed plane is the bit-exact image the decoder would
/// reconstruct from the same FH + tile bytes (the encoder runs the
/// decoder's inverse pipeline on its own quantized coefficients before
/// stamping `recon_*`). `decode_av1(encoded.ivf_bytes)` matches
/// `encoded.reconstructed_*` byte-for-byte at any `base_q_idx`. The
/// recovered planes do NOT in general equal `input.*` — quantization
/// introduces rounding error bounded by the per-axis lookup
/// (`dc_q_lookup[0][base_q_idx]` / `ac_q_lookup[0][base_q_idx]` per
/// the §7.12.2 tables).
///
/// Scope this arc:
///   * 4:2:0, 8-bit, intra-only, single tile, BLOCK_4X4 + TX_4X4 +
///     DCT_DCT (no rectangular **TX_SIZE** family, no §5.11.18 inter
///     mode_info, no per-segment / per-block delta_q, no QM, no
///     in-loop filters). Frame **extent** may be rectangular within
///     {8, 16, 24, 32, 40, 48, 56, 64} per axis.
///   * `base_q_idx ∈ 0..=255`; the picker uses `DCT_DCT` exclusively
///     at >0 (forward WHT only at the lossless arm).
///
/// ## Errors
///
/// * `input.validate()` fails ⇒ [`Error::PartitionWalkOutOfRange`].
/// * Any internal partition-tree / coefficient writer overflow.
pub fn encode_intra_frame_yuv_dyn_with_q(
    input: &Yuv420Frame,
    base_q_idx: u8,
) -> Result<EncodedFrameDyn, Error> {
    input.validate()?;
    let width = input.width as usize;
    let height = input.height as usize;
    let chroma_width = input.chroma_width() as usize;
    let chroma_height = input.chroma_height() as usize;
    let mi_cols = 2 * (((width + 7) >> 3) as u32);
    let mi_rows = 2 * (((height + 7) >> 3) as u32);
    let cells_wide = width / 4;
    let cells_high = height / 4;
    let chroma_cells_wide = chroma_width / 4;
    let chroma_cells_high = chroma_height / 4;

    // Build the SH+FH for this frame.
    let seq = build_intra_only_yuv420_8bit_seq(input.width, input.height);
    let fh = build_intra_only_yuv420_8bit_fh_with_q(&seq, input.width, input.height, base_q_idx);

    // §5.9.2 `CodedLossless` predicate: `base_q_idx == 0 && every
    // delta_q == 0`. The dyn FH keeps all delta_q* fields zero so the
    // predicate reduces to `base_q_idx == 0`. The leaf transform routes
    // through forward WHT on the lossless arm and forward DCT
    // (`forward_transform_2d(..., DCT_DCT, false)`) on the lossy arm;
    // the decoder dispatches symmetrically via `inverse_transform_2d`'s
    // `lossless` flag.
    let lossless = base_q_idx == 0;
    let qp = QuantizerParams::neutral(base_q_idx, 8);
    let scan = get_default_scan(TX_4X4).to_vec();

    // Running reconstructed-pixel buffers (Vec-backed; row-major).
    let mut recon_y = vec![0u8; width * height];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];

    // Walk every in-frame BLOCK_4X4 leaf in §5.11.4 dispatch order.
    // For each in-frame leaf, run the r228+r229 13-mode picker on luma
    // (and on chroma when §5.11.5 HasChroma fires), commit residual
    // through forward-WHT + forward-quantize, and stamp the encoder's
    // own reconstruction of the leaf back into the running plane.
    // Capture the per-leaf `LeafEnc` so the second pass can build the
    // `EncodeNode` tree without re-running the picker.
    let leaves_order = dispatch_order_leaves(root_super_block(mi_cols, mi_rows), mi_rows, mi_cols);
    debug_assert_eq!(leaves_order.len(), cells_wide * cells_high);
    let _ = (chroma_cells_wide, chroma_cells_high);

    // Per-(mi_r, mi_c) leaf bundle (only in-frame entries are populated).
    let mut leaf_table: std::collections::HashMap<(u32, u32), LeafEnc> =
        std::collections::HashMap::with_capacity(leaves_order.len());

    for &(mi_r, mi_c) in &leaves_order {
        let row0 = (mi_r as usize) * 4;
        let col0 = (mi_c as usize) * 4;
        // --- Luma walk ---
        let (y_mode, pred_y) =
            pick_best_intra_mode_4x4(&recon_y, &input.y, width, height, row0, col0);
        let mut residual_y = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let p = input.y[(row0 + i) * width + (col0 + j)] as i64;
                let q = pred_y[i * 4 + j] as i64;
                residual_y[i * 4 + j] = p - q;
            }
        }
        // §5.9.2 lossless arm: forward WHT (bit-exact lattice). Lossy
        // arm (`base_q_idx > 0`): §7.13.3 forward DCT_DCT. The
        // `lossless` flag passes through to `inverse_transform_2d` to
        // match the decoder.
        let coeffs_y = if lossless {
            forward_wht_4x4(&residual_y).to_vec()
        } else {
            forward_transform_2d(&residual_y, TX_4X4, DCT_DCT, false)
        };
        let quant_y = forward_quantize(&coeffs_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);

        let dequant_y = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let resid_back_y = inverse_transform_2d(&dequant_y, TX_4X4, DCT_DCT, 8, lossless);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + resid_back_y[i * 4 + j];
                recon_y[(row0 + i) * width + (col0 + j)] = p.clamp(0, 255) as u8;
            }
        }

        // --- Chroma walk on §5.11.5 HasChroma cells (4:2:0 BLOCK_4X4:
        // both mi coords odd). ---
        let mut coefficients: Vec<PlaneCoefficients> = Vec::with_capacity(3);
        coefficients.push(PlaneCoefficients {
            plane: 0,
            is_inter: 0,
            tx_size: TX_4X4,
            tx_class: TX_CLASS_2D,
            txb_skip_ctx: 0,
            dc_sign_ctx: 0,
            scan: scan.clone(),
            quant: quant_y,
        });

        let has_chroma = (mi_r & 1) != 0 && (mi_c & 1) != 0;
        let mut cfl_alpha_u: Option<i8> = None;
        let mut cfl_alpha_v: Option<i8> = None;
        let uv_mode_picked = if has_chroma {
            let cr = ((mi_r as usize) - 1) / 2;
            let cc = ((mi_c as usize) - 1) / 2;
            let crow0 = cr * 4;
            let ccol0 = cc * 4;
            // r232: chroma picker now evaluates UV_CFL_PRED (§7.11.5.3)
            // in addition to the 13 §6.10.x intra modes, mirroring the
            // fixed-size driver. Needs read access to the running
            // reconstructed luma plane (for the §7.11.5.3 subsampled-
            // luma window). When CFL wins on combined U+V SSD, the
            // picker surfaces the chosen αU / αV pair as signed
            // §5.11.45 `CflAlpha{U,V}` values; otherwise both are
            // `None` and §5.11.45 is not emitted (matching the
            // §5.11.22 line-8 gate).
            let (uv_pick, pred_u, pred_v, alpha_uv) = pick_best_intra_mode_4x4_chroma_joint(
                &recon_y,
                &recon_u,
                &recon_v,
                &input.u,
                &input.v,
                width,
                height,
                chroma_width,
                chroma_height,
                crow0,
                ccol0,
            );
            if let Some((au, av)) = alpha_uv {
                cfl_alpha_u = Some(au);
                cfl_alpha_v = Some(av);
            }
            for (plane, recon_chroma, input_chroma, pred_c) in [
                (1u8, &mut recon_u, &input.u, &pred_u),
                (2u8, &mut recon_v, &input.v, &pred_v),
            ] {
                let mut residual_c = [0i64; 16];
                for i in 0..4 {
                    for j in 0..4 {
                        let p = input_chroma[(crow0 + i) * chroma_width + (ccol0 + j)] as i64;
                        let q = pred_c[i * 4 + j] as i64;
                        residual_c[i * 4 + j] = p - q;
                    }
                }
                // §5.9.2 lossless arm vs lossy DCT — same dispatch as
                // the luma walk.
                let coeffs_c = if lossless {
                    forward_wht_4x4(&residual_c).to_vec()
                } else {
                    forward_transform_2d(&residual_c, TX_4X4, DCT_DCT, false)
                };
                let quant_c = forward_quantize(&coeffs_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);

                let dequant_c = dequantize_step1(&quant_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);
                let resid_back_c = inverse_transform_2d(&dequant_c, TX_4X4, DCT_DCT, 8, lossless);
                for i in 0..4 {
                    for j in 0..4 {
                        let p = pred_c[i * 4 + j] as i64 + resid_back_c[i * 4 + j];
                        recon_chroma[(crow0 + i) * chroma_width + (ccol0 + j)] =
                            p.clamp(0, 255) as u8;
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
            Some(uv_pick)
        } else {
            None
        };

        leaf_table.insert(
            (mi_r, mi_c),
            LeafEnc {
                y_mode,
                uv_mode: uv_mode_picked,
                cfl_alpha_u,
                cfl_alpha_v,
                coefficients,
            },
        );
    }

    // Build the `EncodeNode` tree from the captured leaves. The
    // callback hands `build_partition_tree` an `EncodeBlock` per
    // in-frame leaf coordinate.
    let root_b = root_super_block(mi_cols, mi_rows);
    let root = build_partition_tree(root_b, 0, 0, mi_rows, mi_cols, &mut |r, c| {
        let leaf = leaf_table
            .remove(&(r, c))
            .expect("dispatch-order leaf must have been populated");
        EncodeBlock {
            skip: 0,
            segment_id: 0,
            segment_pred: 0,
            y_mode: leaf.y_mode,
            uv_mode: leaf.uv_mode,
            // r232: dyn driver now plumbs the per-leaf §5.11.45
            // `CflAlpha{U,V}` chosen by `pick_best_intra_mode_4x4_
            // chroma_joint` through to the `EncodeBlock`. Both are
            // `Some(_)` on UV_CFL_PRED leaves and `None` otherwise
            // (mirrors the §5.11.22 line-8 gate).
            cfl_alpha_u: leaf.cfl_alpha_u,
            cfl_alpha_v: leaf.cfl_alpha_v,
            coefficients: leaf.coefficients,
        }
    });

    // §5.11 entropy-coder run for the single tile.
    let mut writer = SymbolWriter::new(false);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut state = PartitionTreeWriter::new(
        mi_rows,
        mi_cols,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        },
        /* segmentation_enabled = */ false,
        /* last_active_seg_id = */ 0,
        /* lossless = */ false,
        /* subsampling_x = */ true,
        /* subsampling_y = */ true,
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;
    write_partition_tree(&mut writer, &mut cdfs, &mut state, &root, 0, 0, root_b)?;
    let tile_bytes = writer.finish();

    // §5.11.1 tile-group body.
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

    // Per-frame OBU sequence: FrameHeader OBU + TileGroup OBU.
    let fh_body = crate::encoder::frame_obu::write_frame_header_obu(&fh, &seq);
    let frame_obus: Vec<ObuFrame> = vec![
        ObuFrame::new(ObuType::FrameHeader, fh_body),
        ObuFrame::new(ObuType::TileGroup, tile_group_body),
    ];

    // §7.5 temporal unit.
    let temporal_unit_bytes = {
        use crate::encoder::obu::build_temporal_unit;
        use crate::encoder::sequence_obu::write_sequence_header_obu;
        let sh_body = write_sequence_header_obu(&seq);
        build_temporal_unit(Some(&sh_body), &frame_obus)
    };

    // IVF v0.
    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(
            cursor,
            FOURCC_AV01,
            input.width as u16,
            input.height as u16,
            25,
            1,
        )
        .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.write_frame(&temporal_unit_bytes, 0)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedFrameDyn {
        ivf_bytes,
        temporal_unit_bytes,
        reconstructed_y: recon_y,
        reconstructed_u: recon_u,
        reconstructed_v: recon_v,
        seq,
        fh,
    })
}

// Silence the unused-import warning on builds that don't reach the
// BLOCK_8X8/BLOCK_16X16/BLOCK_32X32 reference (the constants are
// surfaced via `partition_subsize` rather than direct names in
// `build_partition_tree`).
#[allow(dead_code)]
const _BSIZE_KEEP_ALIVE: (usize, usize, usize, usize) =
    (BLOCK_8X8, BLOCK_16X16, BLOCK_32X32, MI_WIDTH_LOG2[0]);
// Also keep the PARTITION_NONE constant referenced (used by tests).
#[allow(dead_code)]
const _P_KEEP_ALIVE: usize = PARTITION_NONE;

// ----------------------------------------------------------------------
// r235 — Y-only (monochrome) dynamic-extent driver.
// ----------------------------------------------------------------------
//
// Layered on top of the r230..r234 helpers. The only deltas vs the
// 4:2:0 YUV path are:
//
//   1. The SH carries `mono_chrome = true, num_planes = 1` (instead of
//      `mono_chrome = false, num_planes = 3, subsampling_x = subsampling_y
//      = true`). §5.5.2 still enforces `subsampling_x = subsampling_y =
//      true` on the mono arm to keep the §5.9.9 `compute_image_size`
//      derivation (`MiCols = 2 * ((width + 7) >> 3)`) unchanged.
//   2. Each §5.11.5 leaf encodes only the luma plane. The §5.11.5
//      `HasChroma` predicate is gated on `NumPlanes > 1` per the
//      spec, so on `NumPlanes == 1` no chroma syntax (`uv_mode`, U/V
//      `coefficients()`, `CflAlpha{U,V}`) is emitted regardless of
//      mi-position.
//   3. The encoder reconstructs only the luma plane; the U/V Vecs are
//      not populated.
//
// Every other primitive (partition tree, 13-mode luma intra picker,
// forward WHT / forward DCT_DCT, forward quantize, §5.11.39
// coefficient writer, §5.11.1 tile-group OBU, §7.5 temporal unit, IVF
// v0) is reused unchanged from the YUV path.

/// Dynamic-extent monochrome input the r235 Y-only dyn driver consumes.
/// Mirrors [`Yuv420Frame`] but drops the chroma planes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonoYFrame {
    /// Luma width in pixels. Same constraints as [`Yuv420Frame::width`]
    /// (multiple of [`MIN_DIM`], ≤ [`MAX_DIM`]).
    pub width: u32,
    /// Luma height in pixels. Same constraints as
    /// [`Yuv420Frame::height`].
    pub height: u32,
    /// Luma plane (`Y`), row-major, length `width * height`.
    pub y: Vec<u8>,
}

impl MonoYFrame {
    /// All-`fill` mid-grey input.
    #[must_use]
    pub fn filled(width: u32, height: u32, fill: u8) -> Self {
        Self {
            width,
            height,
            y: vec![fill; (width * height) as usize],
        }
    }

    /// Validate the input's dimensions + plane length. Same rules as
    /// [`Yuv420Frame::validate`] sans the chroma checks.
    pub fn validate(&self) -> Result<(), Error> {
        if self.width < MIN_DIM
            || self.height < MIN_DIM
            || self.width > MAX_DIM
            || self.height > MAX_DIM
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.width % MIN_DIM != 0 || self.height % MIN_DIM != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.y.len() != (self.width * self.height) as usize {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// Result of [`encode_intra_frame_y_dyn`]. Bundles the IVF bytes with
/// the reconstructed luma plane (encoder-internal walk; bit-exact equal
/// to the input on the lossless WHT arm).
#[derive(Debug, Clone)]
pub struct EncodedFrameDynY {
    /// IVF bytes — file header + one IVF frame.
    pub ivf_bytes: Vec<u8>,
    /// §7.5 temporal unit bytes (TD + SH + FH + TileGroup OBU).
    pub temporal_unit_bytes: Vec<u8>,
    /// Reconstructed Y plane (row-major, length `width * height`).
    pub reconstructed_y: Vec<u8>,
    /// SequenceHeader the driver synthesised + emitted
    /// (`mono_chrome = true`, `num_planes = 1`).
    pub seq: SequenceHeader,
    /// FrameHeader the driver synthesised + emitted.
    pub fh: FrameHeader,
}

/// Build the minimal 8-bit monochrome `SequenceHeader` for an
/// intra-only frame. Same shape as
/// [`build_intra_only_yuv420_8bit_seq`] but flips the color-config to
/// `mono_chrome = true, num_planes = 1`.
///
/// Per §5.5.2 the mono arm still carries `subsampling_x =
/// subsampling_y = true` (the §5.5.2 mono path explicitly sets both
/// flags). `color_range` is written on wire so we surface the
/// implementation-defined `false` ⇒ studio swing default.
#[must_use]
pub fn build_intra_only_y_8bit_seq(max_width: u32, max_height: u32) -> SequenceHeader {
    let mut seq = build_intra_only_yuv420_8bit_seq(max_width, max_height);
    seq.color_config.mono_chrome = true;
    seq.color_config.num_planes = 1;
    // §5.5.2 mono path: subsampling stays true (per spec), CSP_UNKNOWN,
    // separate_uv_delta_q = false. Already set by the YUV builder.
    seq
}

/// Build the minimal intra-only `FrameHeader` for the r235 Y-only dyn
/// driver. Same field set as
/// [`build_intra_only_yuv420_8bit_fh_with_q`]; the `num_planes`-gated
/// FH writer sub-syntax (e.g. `loop_filter_level[2..4]`,
/// `delta_q_u_*` / `delta_q_v_*`) is suppressed automatically by the
/// FH encoder when `seq.color_config.num_planes == 1`.
#[must_use]
pub fn build_intra_only_y_8bit_fh_with_q(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> FrameHeader {
    build_intra_only_yuv420_8bit_fh_with_q(seq, width, height, base_q_idx)
}

/// Build the minimal intra-only `FrameHeader` for the r235 Y-only dyn
/// driver at `base_q_idx = 0`. Thin wrapper.
#[must_use]
pub fn build_intra_only_y_8bit_fh(seq: &SequenceHeader, width: u32, height: u32) -> FrameHeader {
    build_intra_only_y_8bit_fh_with_q(seq, width, height, 0)
}

/// Encode one monochrome intra-only frame at `base_q_idx = 0`
/// (lossless WHT arm). The reconstructed luma plane on the
/// `EncodedFrameDynY` is bit-exact equal to the input.
///
/// Thin wrapper over [`encode_intra_frame_y_dyn_with_q`].
///
/// ## Errors
///
/// * `input.validate()` fails ⇒ [`Error::PartitionWalkOutOfRange`].
/// * Any internal partition-tree / coefficient-writer overflow.
pub fn encode_intra_frame_y_dyn(input: &MonoYFrame) -> Result<EncodedFrameDynY, Error> {
    encode_intra_frame_y_dyn_with_q(input, 0)
}

/// Encode one monochrome intra-only frame at the caller-supplied
/// `base_q_idx`. `base_q_idx == 0` selects the §5.9.2 `CodedLossless`
/// arm; `base_q_idx > 0` selects the §7.13.3 forward DCT path. The
/// per-leaf code path mirrors [`encode_intra_frame_yuv_dyn_with_q`]'s
/// luma walk exactly; the chroma walk is skipped.
///
/// ## Errors
///
/// Same as [`encode_intra_frame_yuv_dyn_with_q`] minus the chroma-shape
/// checks (no chroma planes ⇒ none can mismatch).
pub fn encode_intra_frame_y_dyn_with_q(
    input: &MonoYFrame,
    base_q_idx: u8,
) -> Result<EncodedFrameDynY, Error> {
    input.validate()?;
    let width = input.width as usize;
    let height = input.height as usize;
    let mi_cols = 2 * (((width + 7) >> 3) as u32);
    let mi_rows = 2 * (((height + 7) >> 3) as u32);
    let cells_wide = width / 4;
    let cells_high = height / 4;

    // Build the mono SH+FH for this frame.
    let seq = build_intra_only_y_8bit_seq(input.width, input.height);
    let fh = build_intra_only_y_8bit_fh_with_q(&seq, input.width, input.height, base_q_idx);

    let lossless = base_q_idx == 0;
    let qp = QuantizerParams::neutral(base_q_idx, 8);
    let scan = get_default_scan(TX_4X4).to_vec();

    let mut recon_y = vec![0u8; width * height];

    // Walk every in-frame BLOCK_4X4 leaf in §5.11.4 dispatch order.
    let leaves_order = dispatch_order_leaves(root_super_block(mi_cols, mi_rows), mi_rows, mi_cols);
    debug_assert_eq!(leaves_order.len(), cells_wide * cells_high);

    let mut leaf_table: std::collections::HashMap<(u32, u32), LeafEnc> =
        std::collections::HashMap::with_capacity(leaves_order.len());

    for &(mi_r, mi_c) in &leaves_order {
        let row0 = (mi_r as usize) * 4;
        let col0 = (mi_c as usize) * 4;
        // Luma walk only.
        let (y_mode, pred_y) =
            pick_best_intra_mode_4x4(&recon_y, &input.y, width, height, row0, col0);
        let mut residual_y = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let p = input.y[(row0 + i) * width + (col0 + j)] as i64;
                let q = pred_y[i * 4 + j] as i64;
                residual_y[i * 4 + j] = p - q;
            }
        }
        let coeffs_y = if lossless {
            forward_wht_4x4(&residual_y).to_vec()
        } else {
            forward_transform_2d(&residual_y, TX_4X4, DCT_DCT, false)
        };
        let quant_y = forward_quantize(&coeffs_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);

        let dequant_y = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let resid_back_y = inverse_transform_2d(&dequant_y, TX_4X4, DCT_DCT, 8, lossless);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + resid_back_y[i * 4 + j];
                recon_y[(row0 + i) * width + (col0 + j)] = p.clamp(0, 255) as u8;
            }
        }

        // §5.11.22 line-8: `uv_mode` is gated on `NumPlanes > 1`.
        // mono ⇒ uv_mode = None on every leaf, no chroma coefficients.
        let coefficients: Vec<PlaneCoefficients> = vec![PlaneCoefficients {
            plane: 0,
            is_inter: 0,
            tx_size: TX_4X4,
            tx_class: TX_CLASS_2D,
            txb_skip_ctx: 0,
            dc_sign_ctx: 0,
            scan: scan.clone(),
            quant: quant_y,
        }];

        leaf_table.insert(
            (mi_r, mi_c),
            LeafEnc {
                y_mode,
                uv_mode: None,
                cfl_alpha_u: None,
                cfl_alpha_v: None,
                coefficients,
            },
        );
    }

    // Build the `EncodeNode` tree from the captured leaves.
    let root_b = root_super_block(mi_cols, mi_rows);
    let root = build_partition_tree(root_b, 0, 0, mi_rows, mi_cols, &mut |r, c| {
        let leaf = leaf_table
            .remove(&(r, c))
            .expect("dispatch-order leaf must have been populated");
        EncodeBlock {
            skip: 0,
            segment_id: 0,
            segment_pred: 0,
            y_mode: leaf.y_mode,
            uv_mode: leaf.uv_mode,
            cfl_alpha_u: leaf.cfl_alpha_u,
            cfl_alpha_v: leaf.cfl_alpha_v,
            coefficients: leaf.coefficients,
        }
    });

    // §5.11 entropy-coder run for the single tile. `subsampling_x` /
    // `subsampling_y` still flagged true (matches the SH).
    let mut writer = SymbolWriter::new(false);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut state = PartitionTreeWriter::new(
        mi_rows,
        mi_cols,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        },
        /* segmentation_enabled = */ false,
        /* last_active_seg_id = */ 0,
        /* lossless = */ false,
        /* subsampling_x = */ true,
        /* subsampling_y = */ true,
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;
    write_partition_tree(&mut writer, &mut cdfs, &mut state, &root, 0, 0, root_b)?;
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

    let fh_body = crate::encoder::frame_obu::write_frame_header_obu(&fh, &seq);
    let frame_obus: Vec<ObuFrame> = vec![
        ObuFrame::new(ObuType::FrameHeader, fh_body),
        ObuFrame::new(ObuType::TileGroup, tile_group_body),
    ];

    let temporal_unit_bytes = {
        use crate::encoder::obu::build_temporal_unit;
        use crate::encoder::sequence_obu::write_sequence_header_obu;
        let sh_body = write_sequence_header_obu(&seq);
        build_temporal_unit(Some(&sh_body), &frame_obus)
    };

    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(
            cursor,
            FOURCC_AV01,
            input.width as u16,
            input.height as u16,
            25,
            1,
        )
        .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.write_frame(&temporal_unit_bytes, 0)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedFrameDynY {
        ivf_bytes,
        temporal_unit_bytes,
        reconstructed_y: recon_y,
        seq,
        fh,
    })
}

// ----------------------------------------------------------------------
// r207 — multi-super-block monochrome (Y-only) dyn driver.
// ----------------------------------------------------------------------
//
// Generalises the r235 single-SB Y-only path to extents up to
// 128×128 by following the §5.11.1 `decode_tile` SB-grid walk
// literally — for `r += sbSize4` then `c += sbSize4` (with
// `sbSize4 = SB_SIZE4_64 = 16` since
// `use_128x128_superblock = false`), the driver builds a separate
// `EncodeNode` tree per SB rooted at `BLOCK_64X64` and feeds them
// to `write_partition_tree` one at a time. Per-SB trees compose
// correctly because:
//
//   1. `PartitionTreeWriter` tracks the full `mi_sizes[]` grid; each
//      `write_partition_tree` call stamps its own SB's cells while
//      reading prior SBs' neighbour stamps for the `partition_ctx`
//      derivation.
//   2. `EncodeNode::dummy_oob` + the §5.11.4 line-1 early return
//      already short-circuit OOB quadrants of edge SBs at
//      rectangular extents (e.g. `80 × 64`, with `mi_cols = 20`),
//      so we don't have to special-case "edge SB" geometry.
//   3. The dyn driver hard-codes `txb_skip_ctx = 0` and
//      `dc_sign_ctx = 0` at every leaf, so the §5.11.1
//      `clear_left_context()` / `clear_above_context()` resets that
//      §6.10.2 specifies between SB rows / tile starts are
//      vacuously satisfied — the dyn encoder + decoder agree on
//      `0` context throughout.
//
// This arc lifts the Y-only ceiling from `MAX_DIM = 64` to
// `MAX_DIM_Y_MULTI_SB = 128`. Existing extents ≤ 64 are NOT
// affected — they still go through the r235
// `encode_intra_frame_y_dyn{,_with_q}` entries (single-SB root_b
// from `root_super_block`); the multi-SB entries are a new
// parallel API. This preserves byte-for-byte IVF output for every
// existing test + caller.
//
// Spec provenance: docs/video/av1/av1-spec.txt §5.11.1 lines 13-25
// (the explicit `for r/c += sbSize4` loop + `decode_partition(r,
// c, sbSize)` body), §6.10.2 `clear_left_context` /
// `clear_above_context` notes (level / dc / seg arrays — irrelevant
// at ctx=0 throughout), §5.5.2 mono color-config (unchanged).

/// r207 — dynamic-extent monochrome input the multi-SB Y-only dyn
/// driver consumes. Same shape as [`MonoYFrame`] but with the
/// extent ceiling raised to [`MAX_DIM_Y_MULTI_SB`] (128).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonoYFrameMultiSb {
    /// Luma width in pixels. Multiple of [`MIN_DIM`],
    /// ≤ [`MAX_DIM_Y_MULTI_SB`].
    pub width: u32,
    /// Luma height in pixels. Same constraints as
    /// [`Self::width`].
    pub height: u32,
    /// Luma plane (`Y`), row-major, length `width * height`.
    pub y: Vec<u8>,
}

impl MonoYFrameMultiSb {
    /// All-`fill` luma input. Useful for tests + as the default
    /// constructor.
    #[must_use]
    pub fn filled(width: u32, height: u32, fill: u8) -> Self {
        Self {
            width,
            height,
            y: vec![fill; (width * height) as usize],
        }
    }

    /// Validate the input's dimensions + Y-plane length. Returns
    /// [`Error::PartitionWalkOutOfRange`] on any mismatch.
    pub fn validate(&self) -> Result<(), Error> {
        if self.width < MIN_DIM
            || self.height < MIN_DIM
            || self.width > MAX_DIM_Y_MULTI_SB
            || self.height > MAX_DIM_Y_MULTI_SB
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.width % MIN_DIM != 0 || self.height % MIN_DIM != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.y.len() != (self.width * self.height) as usize {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// Result of [`encode_intra_frame_y_dyn_multi_sb`]. Same shape as
/// [`EncodedFrameDynY`].
#[derive(Debug, Clone)]
pub struct EncodedFrameDynYMultiSb {
    /// IVF bytes — file header + one IVF frame.
    pub ivf_bytes: Vec<u8>,
    /// §7.5 temporal unit bytes (TD + SH + FH + TileGroup OBU).
    pub temporal_unit_bytes: Vec<u8>,
    /// Reconstructed Y plane (row-major, length `width * height`).
    pub reconstructed_y: Vec<u8>,
    /// SequenceHeader the driver synthesised + emitted
    /// (`mono_chrome = true`, `num_planes = 1`).
    pub seq: SequenceHeader,
    /// FrameHeader the driver synthesised + emitted.
    pub fh: FrameHeader,
}

/// r207 — §5.11.1-conformant SB-grid walk over a (mi_rows ×
/// mi_cols) frame. Returns `(sb_r, sb_c)` origins in row-major
/// order; each tuple addresses one BLOCK_64X64 SB (16 × 16 mi
/// units, with the trailing SBs clipped to the frame extent via
/// the §5.11.4 line-1 OOB short-circuit on edge quadrants). Used
/// by both the multi-SB encoder and decoder.
///
/// Equivalent to the spec's literal:
/// ```text
/// for (r = MiRowStart; r < MiRowEnd; r += sbSize4) {
///     for (c = MiColStart; c < MiColEnd; c += sbSize4) {
///         decode_partition(r, c, sbSize)
///     }
/// }
/// ```
/// with `MiRowStart = MiColStart = 0`, `MiRowEnd = mi_rows`,
/// `MiColEnd = mi_cols`, and `sbSize4 = SB_SIZE4_64 = 16`.
#[must_use]
pub fn sb_grid_origins(mi_rows: u32, mi_cols: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut r = 0u32;
    while r < mi_rows {
        let mut c = 0u32;
        while c < mi_cols {
            out.push((r, c));
            c += SB_SIZE4_64;
        }
        r += SB_SIZE4_64;
    }
    out
}

/// r207 — §5.11.4 dispatch-order BLOCK_4X4 cell walk over the
/// `(mi_rows × mi_cols)` frame via the §5.11.1 SB-grid traversal
/// (i.e. outer SB-row-major × inner per-SB NW/NE/SW/SE recursion).
/// Used by the multi-SB encoder driver to populate its `leaf_table`
/// in the §5.11.1 / §5.11.4 visiting order.
///
/// Concretely: for every `(sb_r, sb_c)` from
/// [`sb_grid_origins`], invokes the per-SB NW/NE/SW/SE recursion of
/// [`build_partition_tree`] rooted at `BLOCK_64X64` and emits each
/// in-frame BLOCK_4X4 leaf in the order it would be written.
#[must_use]
pub fn sb_grid_dispatch_order_leaves(mi_rows: u32, mi_cols: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
        fn walk(
            b_size: usize,
            r: u32,
            c: u32,
            mi_rows: u32,
            mi_cols: u32,
            out: &mut Vec<(u32, u32)>,
        ) {
            if r >= mi_rows || c >= mi_cols {
                return;
            }
            if b_size == BLOCK_4X4 {
                out.push((r, c));
                return;
            }
            let sub_size =
                partition_subsize(PARTITION_SPLIT, b_size).expect("valid SPLIT sub-size");
            let half = (NUM_4X4_BLOCKS_WIDE[b_size] as u32) >> 1;
            walk(sub_size, r, c, mi_rows, mi_cols, out);
            walk(sub_size, r, c + half, mi_rows, mi_cols, out);
            walk(sub_size, r + half, c, mi_rows, mi_cols, out);
            walk(sub_size, r + half, c + half, mi_rows, mi_cols, out);
        }
        walk(BLOCK_64X64, sb_r, sb_c, mi_rows, mi_cols, &mut out);
    }
    out
}

/// r207 — encode one monochrome intra-only frame at the
/// caller-supplied `base_q_idx` for any allowed extent ∈
/// [MIN_DIM, MAX_DIM_Y_MULTI_SB]². The §5.11.1 SB-grid is walked
/// row-major; each SB is a single `BLOCK_64X64` root partition tree.
///
/// `base_q_idx == 0` selects the §5.9.2 `CodedLossless` arm (forward
/// WHT on the leaf transform — encoder-internal reconstruction
/// equals input plane-for-plane). `> 0` selects the §7.13.3 forward
/// DCT_DCT path with §7.12.3 quantize.
///
/// Multi-SB contract: at `base_q_idx > 0`, `decode_av1(encoded.ivf_bytes)`
/// matches `encoded.reconstructed_y` byte-for-byte for any allowed
/// extent. At `base_q_idx == 0` the recovered Y plane additionally
/// equals the input plane-for-plane.
///
/// ## Errors
///
/// * `input.validate()` fails ⇒ [`Error::PartitionWalkOutOfRange`].
/// * Any internal partition-tree / coefficient-writer overflow.
pub fn encode_intra_frame_y_dyn_multi_sb_with_q(
    input: &MonoYFrameMultiSb,
    base_q_idx: u8,
) -> Result<EncodedFrameDynYMultiSb, Error> {
    input.validate()?;
    let width = input.width as usize;
    let height = input.height as usize;
    let mi_cols = 2 * (((width + 7) >> 3) as u32);
    let mi_rows = 2 * (((height + 7) >> 3) as u32);

    let seq = build_intra_only_y_8bit_seq(input.width, input.height);
    let fh = build_intra_only_y_8bit_fh_with_q(&seq, input.width, input.height, base_q_idx);

    let lossless = base_q_idx == 0;
    let qp = QuantizerParams::neutral(base_q_idx, 8);
    let scan = get_default_scan(TX_4X4).to_vec();

    let mut recon_y = vec![0u8; width * height];

    // Walk every in-frame BLOCK_4X4 leaf in §5.11.1 SB-grid order:
    // outer SB-row-major over `sb_grid_origins`, inner per-SB
    // §5.11.4 NW/NE/SW/SE recursion. The dispatch order matches
    // the order `decode_tile` reads symbols in.
    let leaves_order = sb_grid_dispatch_order_leaves(mi_rows, mi_cols);
    debug_assert_eq!(
        leaves_order.len(),
        (width / 4) * (height / 4),
        "every in-frame BLOCK_4X4 cell must be visited exactly once"
    );

    let mut leaf_table: std::collections::HashMap<(u32, u32), LeafEnc> =
        std::collections::HashMap::with_capacity(leaves_order.len());

    for &(mi_r, mi_c) in &leaves_order {
        let row0 = (mi_r as usize) * 4;
        let col0 = (mi_c as usize) * 4;
        let (y_mode, pred_y) =
            pick_best_intra_mode_4x4(&recon_y, &input.y, width, height, row0, col0);
        let mut residual_y = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let p = input.y[(row0 + i) * width + (col0 + j)] as i64;
                let q = pred_y[i * 4 + j] as i64;
                residual_y[i * 4 + j] = p - q;
            }
        }
        let coeffs_y = if lossless {
            forward_wht_4x4(&residual_y).to_vec()
        } else {
            forward_transform_2d(&residual_y, TX_4X4, DCT_DCT, false)
        };
        let quant_y = forward_quantize(&coeffs_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);

        let dequant_y = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let resid_back_y = inverse_transform_2d(&dequant_y, TX_4X4, DCT_DCT, 8, lossless);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + resid_back_y[i * 4 + j];
                recon_y[(row0 + i) * width + (col0 + j)] = p.clamp(0, 255) as u8;
            }
        }

        let coefficients: Vec<PlaneCoefficients> = vec![PlaneCoefficients {
            plane: 0,
            is_inter: 0,
            tx_size: TX_4X4,
            tx_class: TX_CLASS_2D,
            txb_skip_ctx: 0,
            dc_sign_ctx: 0,
            scan: scan.clone(),
            quant: quant_y,
        }];

        leaf_table.insert(
            (mi_r, mi_c),
            LeafEnc {
                y_mode,
                uv_mode: None,
                cfl_alpha_u: None,
                cfl_alpha_v: None,
                coefficients,
            },
        );
    }

    // §5.11 entropy-coder run for the single tile, multi-SB version.
    let mut writer = SymbolWriter::new(false);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut state = PartitionTreeWriter::new(
        mi_rows,
        mi_cols,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        },
        /* segmentation_enabled = */ false,
        /* last_active_seg_id = */ 0,
        /* lossless = */ false,
        /* subsampling_x = */ true,
        /* subsampling_y = */ true,
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;

    // For each SB in §5.11.1 row-major order: build a
    // BLOCK_64X64-rooted EncodeNode tree using the captured
    // leaves, then write_partition_tree starting at that SB origin.
    for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
        let sb_root =
            build_partition_tree(BLOCK_64X64, sb_r, sb_c, mi_rows, mi_cols, &mut |r, c| {
                let leaf = leaf_table
                    .remove(&(r, c))
                    .expect("dispatch-order leaf must have been populated");
                EncodeBlock {
                    skip: 0,
                    segment_id: 0,
                    segment_pred: 0,
                    y_mode: leaf.y_mode,
                    uv_mode: leaf.uv_mode,
                    cfl_alpha_u: leaf.cfl_alpha_u,
                    cfl_alpha_v: leaf.cfl_alpha_v,
                    coefficients: leaf.coefficients,
                }
            });
        write_partition_tree(
            &mut writer,
            &mut cdfs,
            &mut state,
            &sb_root,
            sb_r,
            sb_c,
            BLOCK_64X64,
        )?;
    }
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

    let fh_body = crate::encoder::frame_obu::write_frame_header_obu(&fh, &seq);
    let frame_obus: Vec<ObuFrame> = vec![
        ObuFrame::new(ObuType::FrameHeader, fh_body),
        ObuFrame::new(ObuType::TileGroup, tile_group_body),
    ];

    let temporal_unit_bytes = {
        use crate::encoder::obu::build_temporal_unit;
        use crate::encoder::sequence_obu::write_sequence_header_obu;
        let sh_body = write_sequence_header_obu(&seq);
        build_temporal_unit(Some(&sh_body), &frame_obus)
    };

    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(
            cursor,
            FOURCC_AV01,
            input.width as u16,
            input.height as u16,
            25,
            1,
        )
        .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.write_frame(&temporal_unit_bytes, 0)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedFrameDynYMultiSb {
        ivf_bytes,
        temporal_unit_bytes,
        reconstructed_y: recon_y,
        seq,
        fh,
    })
}

/// r207 — encode one monochrome intra-only frame at `base_q_idx = 0`
/// (lossless WHT arm) for any allowed extent up to
/// [`MAX_DIM_Y_MULTI_SB`]. Thin wrapper over
/// [`encode_intra_frame_y_dyn_multi_sb_with_q`].
pub fn encode_intra_frame_y_dyn_multi_sb(
    input: &MonoYFrameMultiSb,
) -> Result<EncodedFrameDynYMultiSb, Error> {
    encode_intra_frame_y_dyn_multi_sb_with_q(input, 0)
}

// ----------------------------------------------------------------------
// r214 — multi-super-block 4:2:0 YUV dyn driver.
// ----------------------------------------------------------------------
//
// Generalises the single-SB YUV path (`encode_intra_frame_yuv_dyn{,
// _with_q}` — single `BLOCK_64X64`-root partition tree, `MAX_DIM = 64`
// cap) to extents up to 128×128 by following the same §5.11.1
// `decode_tile` SB-grid walk the r207 mono-Y multi-SB path uses (see
// `sb_grid_origins` for the `for r/c += sbSize4` body), but with the
// luma + 4:2:0 chroma walks active at every BLOCK_4X4 leaf where
// §5.11.5 HasChroma fires.
//
// HasChroma composition across SBs (4:2:0):
//   * The 4:2:0 BLOCK_4X4 HasChroma predicate is `(mi_r & 1) != 0 &&
//     (mi_c & 1) != 0` — see §5.11.5. Both `mi_r` and `mi_c` are
//     absolute (frame-global), so any in-frame chroma cell is hit
//     exactly once across the SB-grid walk regardless of which SB
//     fired the leaf. The chroma cell index `((mi_r-1)/2, (mi_c-1)/2)`
//     is likewise frame-global.
//   * The §7.11.5.3 CFL subsampled-luma window (`cfl_subsampled_luma_
//     4x4_420_dyn`) reads `recon_y` over the corresponding 8×8 luma
//     footprint — the running `recon_y` buffer already covers the
//     full frame extent and is updated leaf-by-leaf in the same
//     dispatch order the decoder reads, so the chroma walk on the
//     last leaf of an SB still sees correctly-reconstructed luma
//     stamped by every prior leaf of that SB and every prior SB. The
//     window correctly clips at the chroma plane edge — already
//     covered for the single-SB cap.
//   * Per-SB context resets (`clear_above_context` / `clear_left_
//     context` per §5.11.1 / §6.10.2) are vacuously satisfied
//     because the dyn driver hard-codes `txb_skip_ctx = 0` /
//     `dc_sign_ctx = 0` at every leaf (both luma and chroma planes)
//     — the encoder + decoder agree on `0` context throughout.
//
// This arc lifts the YUV ceiling from `MAX_DIM = 64` to a new
// `MAX_DIM_YUV_MULTI_SB = 128`. Existing extents ≤ 64 are NOT
// affected — they still go through the single-SB
// `encode_intra_frame_yuv_dyn{,_with_q}` entries; the multi-SB
// entries are a new parallel API. This preserves byte-for-byte IVF
// output for every existing test + caller.
//
// Spec provenance: docs/video/av1/av1-spec.txt §5.11.1 lines 13-25
// (the explicit `for r/c += sbSize4` loop + `decode_partition(r, c,
// sbSize)` body), §5.11.5 (`HasChroma` predicate +
// `NumPlanes > 1` gate), §7.11.5.3 (CFL subsampled-luma window —
// already verified to clip at the chroma plane edge on the single-SB
// path), §5.5.2 (4:2:0 color-config — unchanged across SBs),
// §6.10.2 (`clear_left_context` / `clear_above_context` semantics —
// vacuous at ctx=0).

/// r214 — multi-super-block YUV ceiling for the §5.11.1 SB-grid walk
/// on the 4:2:0 YUV dyn driver. Same ceiling as the mono-Y path
/// ([`MAX_DIM_Y_MULTI_SB`]) — two SBs per axis at
/// `use_128x128_superblock = false`.
pub const MAX_DIM_YUV_MULTI_SB: u32 = 128;

/// r214 — dynamic-extent 4:2:0 YUV input the multi-SB dyn driver
/// consumes. Same shape as [`Yuv420Frame`] but with the extent
/// ceiling raised to [`MAX_DIM_YUV_MULTI_SB`] (128).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Yuv420FrameMultiSb {
    /// Luma width in pixels. Multiple of [`MIN_DIM`],
    /// ≤ [`MAX_DIM_YUV_MULTI_SB`].
    pub width: u32,
    /// Luma height in pixels. Same constraints as [`Self::width`].
    pub height: u32,
    /// Luma plane (`Y`), row-major, length `width * height`.
    pub y: Vec<u8>,
    /// First chroma plane (`U` / `Cb`) at half horizontal + vertical
    /// resolution; length `(width / 2) * (height / 2)`.
    pub u: Vec<u8>,
    /// Second chroma plane (`V` / `Cr`); same shape as `u`.
    pub v: Vec<u8>,
}

impl Yuv420FrameMultiSb {
    /// All-`fill` mid-grey input. Useful for tests + as the default
    /// constructor; every plane is set to the same value.
    #[must_use]
    pub fn filled(width: u32, height: u32, fill: u8) -> Self {
        let cw = width / 2;
        let ch = height / 2;
        Self {
            width,
            height,
            y: vec![fill; (width * height) as usize],
            u: vec![fill; (cw * ch) as usize],
            v: vec![fill; (cw * ch) as usize],
        }
    }

    /// Chroma plane width — `width / 2` per the 4:2:0 sampling
    /// pattern.
    #[must_use]
    pub fn chroma_width(&self) -> u32 {
        self.width / 2
    }

    /// Chroma plane height — `height / 2` per the 4:2:0 sampling
    /// pattern.
    #[must_use]
    pub fn chroma_height(&self) -> u32 {
        self.height / 2
    }

    /// Validate the input's dimensions + plane lengths. Returns
    /// [`Error::PartitionWalkOutOfRange`] on any mismatch.
    pub fn validate(&self) -> Result<(), Error> {
        if self.width < MIN_DIM
            || self.height < MIN_DIM
            || self.width > MAX_DIM_YUV_MULTI_SB
            || self.height > MAX_DIM_YUV_MULTI_SB
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.width % MIN_DIM != 0 || self.height % MIN_DIM != 0 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let expected_y = (self.width * self.height) as usize;
        let expected_uv = (self.chroma_width() * self.chroma_height()) as usize;
        if self.y.len() != expected_y || self.u.len() != expected_uv || self.v.len() != expected_uv
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// Result of [`encode_intra_frame_yuv_dyn_multi_sb`]. Same shape as
/// [`EncodedFrameDyn`].
#[derive(Debug, Clone)]
pub struct EncodedFrameDynYuvMultiSb {
    /// IVF bytes — file header + one IVF frame.
    pub ivf_bytes: Vec<u8>,
    /// §7.5 temporal unit bytes (TD + SH + FH + TileGroup OBU).
    pub temporal_unit_bytes: Vec<u8>,
    /// Reconstructed Y plane (row-major, length `width * height`).
    pub reconstructed_y: Vec<u8>,
    /// Reconstructed U plane (row-major, length
    /// `(width/2) * (height/2)`).
    pub reconstructed_u: Vec<u8>,
    /// Reconstructed V plane.
    pub reconstructed_v: Vec<u8>,
    /// SequenceHeader the driver synthesised + emitted (4:2:0).
    pub seq: SequenceHeader,
    /// FrameHeader the driver synthesised + emitted.
    pub fh: FrameHeader,
}

/// r214 — encode one 4:2:0 YUV intra-only frame at the
/// caller-supplied `base_q_idx` for any allowed extent ∈
/// [MIN_DIM, MAX_DIM_YUV_MULTI_SB]². The §5.11.1 SB-grid is walked
/// row-major; each SB is a single `BLOCK_64X64` root partition tree.
///
/// `base_q_idx == 0` selects the §5.9.2 `CodedLossless` arm (forward
/// WHT on the leaf transform — encoder-internal reconstruction
/// equals input plane-for-plane on all three planes). `> 0` selects
/// the §7.13.3 forward DCT_DCT path with §7.12.3 quantize on all
/// three planes.
///
/// Multi-SB contract: at `base_q_idx > 0`,
/// `decode_av1(encoded.ivf_bytes)` matches
/// `encoded.reconstructed_{y,u,v}` byte-for-byte for any allowed
/// extent. At `base_q_idx == 0` the recovered planes additionally
/// equal the input planes plane-for-plane.
///
/// ## Errors
///
/// * `input.validate()` fails ⇒ [`Error::PartitionWalkOutOfRange`].
/// * Any internal partition-tree / coefficient-writer overflow.
pub fn encode_intra_frame_yuv_dyn_multi_sb_with_q(
    input: &Yuv420FrameMultiSb,
    base_q_idx: u8,
) -> Result<EncodedFrameDynYuvMultiSb, Error> {
    input.validate()?;
    let width = input.width as usize;
    let height = input.height as usize;
    let chroma_width = input.chroma_width() as usize;
    let chroma_height = input.chroma_height() as usize;
    let mi_cols = 2 * (((width + 7) >> 3) as u32);
    let mi_rows = 2 * (((height + 7) >> 3) as u32);

    let seq = build_intra_only_yuv420_8bit_seq(input.width, input.height);
    let fh = build_intra_only_yuv420_8bit_fh_with_q(&seq, input.width, input.height, base_q_idx);

    let lossless = base_q_idx == 0;
    let qp = QuantizerParams::neutral(base_q_idx, 8);
    let scan = get_default_scan(TX_4X4).to_vec();

    let mut recon_y = vec![0u8; width * height];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];

    // Walk every in-frame BLOCK_4X4 leaf in §5.11.1 SB-grid order:
    // outer SB-row-major over `sb_grid_origins`, inner per-SB
    // §5.11.4 NW/NE/SW/SE recursion. The chroma cells at any
    // (mi_r, mi_c) where both indices are odd fire HasChroma per
    // §5.11.5.
    let leaves_order = sb_grid_dispatch_order_leaves(mi_rows, mi_cols);
    debug_assert_eq!(
        leaves_order.len(),
        (width / 4) * (height / 4),
        "every in-frame BLOCK_4X4 cell must be visited exactly once"
    );

    let mut leaf_table: std::collections::HashMap<(u32, u32), LeafEnc> =
        std::collections::HashMap::with_capacity(leaves_order.len());

    for &(mi_r, mi_c) in &leaves_order {
        let row0 = (mi_r as usize) * 4;
        let col0 = (mi_c as usize) * 4;
        // --- Luma walk ---
        let (y_mode, pred_y) =
            pick_best_intra_mode_4x4(&recon_y, &input.y, width, height, row0, col0);
        let mut residual_y = [0i64; 16];
        for i in 0..4 {
            for j in 0..4 {
                let p = input.y[(row0 + i) * width + (col0 + j)] as i64;
                let q = pred_y[i * 4 + j] as i64;
                residual_y[i * 4 + j] = p - q;
            }
        }
        let coeffs_y = if lossless {
            forward_wht_4x4(&residual_y).to_vec()
        } else {
            forward_transform_2d(&residual_y, TX_4X4, DCT_DCT, false)
        };
        let quant_y = forward_quantize(&coeffs_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);

        let dequant_y = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let resid_back_y = inverse_transform_2d(&dequant_y, TX_4X4, DCT_DCT, 8, lossless);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + resid_back_y[i * 4 + j];
                recon_y[(row0 + i) * width + (col0 + j)] = p.clamp(0, 255) as u8;
            }
        }

        let mut coefficients: Vec<PlaneCoefficients> = Vec::with_capacity(3);
        coefficients.push(PlaneCoefficients {
            plane: 0,
            is_inter: 0,
            tx_size: TX_4X4,
            tx_class: TX_CLASS_2D,
            txb_skip_ctx: 0,
            dc_sign_ctx: 0,
            scan: scan.clone(),
            quant: quant_y,
        });

        // --- §5.11.5 HasChroma at 4:2:0 BLOCK_4X4: both mi coords
        // odd. The chroma cell index `((mi_r-1)/2, (mi_c-1)/2)` is
        // frame-global, so no per-SB rewrite is needed.
        let has_chroma = (mi_r & 1) != 0 && (mi_c & 1) != 0;
        let mut cfl_alpha_u: Option<i8> = None;
        let mut cfl_alpha_v: Option<i8> = None;
        let uv_mode_picked = if has_chroma {
            let cr = ((mi_r as usize) - 1) / 2;
            let cc = ((mi_c as usize) - 1) / 2;
            let crow0 = cr * 4;
            let ccol0 = cc * 4;
            let (uv_pick, pred_u, pred_v, alpha_uv) = pick_best_intra_mode_4x4_chroma_joint(
                &recon_y,
                &recon_u,
                &recon_v,
                &input.u,
                &input.v,
                width,
                height,
                chroma_width,
                chroma_height,
                crow0,
                ccol0,
            );
            if let Some((au, av)) = alpha_uv {
                cfl_alpha_u = Some(au);
                cfl_alpha_v = Some(av);
            }
            for (plane, recon_chroma, input_chroma, pred_c) in [
                (1u8, &mut recon_u, &input.u, &pred_u),
                (2u8, &mut recon_v, &input.v, &pred_v),
            ] {
                let mut residual_c = [0i64; 16];
                for i in 0..4 {
                    for j in 0..4 {
                        let p = input_chroma[(crow0 + i) * chroma_width + (ccol0 + j)] as i64;
                        let q = pred_c[i * 4 + j] as i64;
                        residual_c[i * 4 + j] = p - q;
                    }
                }
                let coeffs_c = if lossless {
                    forward_wht_4x4(&residual_c).to_vec()
                } else {
                    forward_transform_2d(&residual_c, TX_4X4, DCT_DCT, false)
                };
                let quant_c = forward_quantize(&coeffs_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);

                let dequant_c = dequantize_step1(&quant_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);
                let resid_back_c = inverse_transform_2d(&dequant_c, TX_4X4, DCT_DCT, 8, lossless);
                for i in 0..4 {
                    for j in 0..4 {
                        let p = pred_c[i * 4 + j] as i64 + resid_back_c[i * 4 + j];
                        recon_chroma[(crow0 + i) * chroma_width + (ccol0 + j)] =
                            p.clamp(0, 255) as u8;
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
            Some(uv_pick)
        } else {
            None
        };

        leaf_table.insert(
            (mi_r, mi_c),
            LeafEnc {
                y_mode,
                uv_mode: uv_mode_picked,
                cfl_alpha_u,
                cfl_alpha_v,
                coefficients,
            },
        );
    }

    // §5.11 entropy-coder run for the single tile, multi-SB version.
    let mut writer = SymbolWriter::new(false);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut state = PartitionTreeWriter::new(
        mi_rows,
        mi_cols,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        },
        /* segmentation_enabled = */ false,
        /* last_active_seg_id = */ 0,
        /* lossless = */ false,
        /* subsampling_x = */ true,
        /* subsampling_y = */ true,
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;

    // For each SB in §5.11.1 row-major order: build a
    // BLOCK_64X64-rooted EncodeNode tree using the captured leaves,
    // then write_partition_tree starting at that SB origin.
    for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
        let sb_root =
            build_partition_tree(BLOCK_64X64, sb_r, sb_c, mi_rows, mi_cols, &mut |r, c| {
                let leaf = leaf_table
                    .remove(&(r, c))
                    .expect("dispatch-order leaf must have been populated");
                EncodeBlock {
                    skip: 0,
                    segment_id: 0,
                    segment_pred: 0,
                    y_mode: leaf.y_mode,
                    uv_mode: leaf.uv_mode,
                    cfl_alpha_u: leaf.cfl_alpha_u,
                    cfl_alpha_v: leaf.cfl_alpha_v,
                    coefficients: leaf.coefficients,
                }
            });
        write_partition_tree(
            &mut writer,
            &mut cdfs,
            &mut state,
            &sb_root,
            sb_r,
            sb_c,
            BLOCK_64X64,
        )?;
    }
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

    let fh_body = crate::encoder::frame_obu::write_frame_header_obu(&fh, &seq);
    let frame_obus: Vec<ObuFrame> = vec![
        ObuFrame::new(ObuType::FrameHeader, fh_body),
        ObuFrame::new(ObuType::TileGroup, tile_group_body),
    ];

    let temporal_unit_bytes = {
        use crate::encoder::obu::build_temporal_unit;
        use crate::encoder::sequence_obu::write_sequence_header_obu;
        let sh_body = write_sequence_header_obu(&seq);
        build_temporal_unit(Some(&sh_body), &frame_obus)
    };

    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(
            cursor,
            FOURCC_AV01,
            input.width as u16,
            input.height as u16,
            25,
            1,
        )
        .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.write_frame(&temporal_unit_bytes, 0)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedFrameDynYuvMultiSb {
        ivf_bytes,
        temporal_unit_bytes,
        reconstructed_y: recon_y,
        reconstructed_u: recon_u,
        reconstructed_v: recon_v,
        seq,
        fh,
    })
}

/// r214 — encode one 4:2:0 YUV intra-only frame at `base_q_idx = 0`
/// (lossless WHT arm) for any allowed extent up to
/// [`MAX_DIM_YUV_MULTI_SB`]. Thin wrapper over
/// [`encode_intra_frame_yuv_dyn_multi_sb_with_q`].
pub fn encode_intra_frame_yuv_dyn_multi_sb(
    input: &Yuv420FrameMultiSb,
) -> Result<EncodedFrameDynYuvMultiSb, Error> {
    encode_intra_frame_yuv_dyn_multi_sb_with_q(input, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Quick coverage check: dispatch_order_leaves visits every in-frame
    // BLOCK_4X4 cell exactly once for square frame sizes.
    #[test]
    fn dispatch_order_leaves_covers_every_in_frame_cell_16x16() {
        let leaves = dispatch_order_leaves(BLOCK_16X16, 4, 4);
        assert_eq!(leaves.len(), 16);
        let mut seen = [false; 16];
        for &(r, c) in &leaves {
            let idx = (r * 4 + c) as usize;
            assert!(!seen[idx], "cell ({r}, {c}) visited twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn dispatch_order_leaves_covers_every_in_frame_cell_32x32() {
        let leaves = dispatch_order_leaves(BLOCK_32X32, 8, 8);
        assert_eq!(leaves.len(), 64);
        let mut seen = [false; 64];
        for &(r, c) in &leaves {
            let idx = (r * 8 + c) as usize;
            assert!(!seen[idx], "cell ({r}, {c}) visited twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn dispatch_order_leaves_covers_every_in_frame_cell_64x64() {
        let leaves = dispatch_order_leaves(BLOCK_64X64, 16, 16);
        assert_eq!(leaves.len(), 256);
        let mut seen = vec![false; 256];
        for &(r, c) in &leaves {
            let idx = (r * 16 + c) as usize;
            assert!(!seen[idx], "cell ({r}, {c}) visited twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn root_super_block_picks_smallest_covering_size() {
        // 16×16 ⇒ mi 4×4 ⇒ BLOCK_16X16
        assert_eq!(root_super_block(4, 4), BLOCK_16X16);
        // 32×32 ⇒ mi 8×8 ⇒ BLOCK_32X32
        assert_eq!(root_super_block(8, 8), BLOCK_32X32);
        // 64×64 ⇒ mi 16×16 ⇒ BLOCK_64X64
        assert_eq!(root_super_block(16, 16), BLOCK_64X64);
        // 24×24 ⇒ mi 6×6 ⇒ still fits in BLOCK_32X32
        assert_eq!(root_super_block(6, 6), BLOCK_32X32);
        // 40×40 ⇒ mi 10×10 ⇒ falls into BLOCK_64X64
        assert_eq!(root_super_block(10, 10), BLOCK_64X64);
    }

    #[test]
    fn yuv420_frame_validate_accepts_aligned_dims() {
        for (w, h) in [
            (8, 8),
            (16, 16),
            (24, 24),
            (32, 32),
            (64, 64),
            (32, 16),
            (16, 32),
        ] {
            let f = Yuv420Frame::filled(w, h, 128);
            assert!(f.validate().is_ok(), "{w}x{h} should validate");
        }
    }

    #[test]
    fn yuv420_frame_validate_rejects_unaligned() {
        for (w, h) in [
            (0u32, 16),
            (16, 0),
            (4, 16),
            (12, 16),
            (16, 4),
            (72, 16),
            (16, 72),
        ] {
            // Provide best-effort plane Vec lengths — validate() rejects
            // on the dim check before reading lengths anyway, but having
            // matched-length Vecs keeps the test focused on the dim
            // rejection.
            let cw = (w / 2).max(1);
            let ch = (h / 2).max(1);
            let f = Yuv420Frame {
                width: w,
                height: h,
                y: vec![0u8; (w * h) as usize],
                u: vec![0u8; (cw * ch) as usize],
                v: vec![0u8; (cw * ch) as usize],
            };
            assert!(f.validate().is_err(), "{w}x{h} should NOT validate");
        }
    }

    #[test]
    fn encode_flat_grey_16x16_internal_recon_matches_input() {
        // Cross-check the dyn driver against the fixed-size driver: a
        // 16×16 flat-128 YUV input should reconstruct to itself bit-
        // exactly (the lossless WHT path).
        let input = Yuv420Frame::filled(16, 16, 128);
        let res = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
        assert_eq!(res.reconstructed_y, input.y);
        assert_eq!(res.reconstructed_u, input.u);
        assert_eq!(res.reconstructed_v, input.v);
    }

    #[test]
    fn encode_flat_grey_32x32_internal_recon_matches_input() {
        let input = Yuv420Frame::filled(32, 32, 128);
        let res = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
        assert_eq!(res.reconstructed_y, input.y);
        assert_eq!(res.reconstructed_u, input.u);
        assert_eq!(res.reconstructed_v, input.v);
    }

    #[test]
    fn encode_flat_grey_64x64_internal_recon_matches_input() {
        let input = Yuv420Frame::filled(64, 64, 128);
        let res = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
        assert_eq!(res.reconstructed_y, input.y);
        assert_eq!(res.reconstructed_u, input.u);
        assert_eq!(res.reconstructed_v, input.v);
    }

    #[test]
    fn encode_pseudorandom_32x32_internal_recon_matches_input_bit_exact() {
        let mut input = Yuv420Frame::filled(32, 32, 0);
        let mut state: u64 = 0xABCD_1234_5678_9F0E;
        let mut next = || -> u8 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 56) as u8
        };
        for p in input.y.iter_mut() {
            *p = next();
        }
        for p in input.u.iter_mut() {
            *p = next();
        }
        for p in input.v.iter_mut() {
            *p = next();
        }
        let res = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
        assert_eq!(
            res.reconstructed_y, input.y,
            "Y lossless WHT mismatch at 32×32"
        );
        assert_eq!(
            res.reconstructed_u, input.u,
            "U lossless WHT mismatch at 32×32"
        );
        assert_eq!(
            res.reconstructed_v, input.v,
            "V lossless WHT mismatch at 32×32"
        );
    }

    #[test]
    fn encoded_ivf_carries_dynamic_dimensions_in_header() {
        let input = Yuv420Frame::filled(32, 32, 200);
        let res = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
        // IVF v0: bytes 12..14 = width LE, bytes 14..16 = height LE.
        assert_eq!(
            u16::from_le_bytes([res.ivf_bytes[12], res.ivf_bytes[13]]),
            32
        );
        assert_eq!(
            u16::from_le_bytes([res.ivf_bytes[14], res.ivf_bytes[15]]),
            32
        );
    }

    #[test]
    fn fh_carries_dynamic_dimensions() {
        let input = Yuv420Frame::filled(64, 64, 128);
        let res = encode_intra_frame_yuv_dyn(&input).expect("encode succeeds");
        let fs = res
            .fh
            .frame_size
            .as_ref()
            .expect("intra frame has frame_size");
        assert_eq!(fs.frame_width, 64);
        assert_eq!(fs.frame_height, 64);
        assert_eq!(fs.mi_cols, 16);
        assert_eq!(fs.mi_rows, 16);
    }

    #[test]
    fn encode_rejects_invalid_dimensions() {
        // Caller passing a struct whose Vec lengths don't match width *
        // height ⇒ early reject.
        let bad = Yuv420Frame {
            width: 16,
            height: 16,
            y: vec![0u8; 8],
            u: vec![0u8; 8],
            v: vec![0u8; 8],
        };
        assert!(encode_intra_frame_yuv_dyn(&bad).is_err());
    }

    // ---- r196/r233: lossy-quant path on the dyn driver. ----

    #[test]
    fn fh_with_q_zero_matches_legacy_lossless_fh_byte_for_byte() {
        // Sanity: `build_intra_only_yuv420_8bit_fh_with_q(_, 0)` and
        // the legacy `build_intra_only_yuv420_8bit_fh` must produce
        // the same FrameHeader bytes (FH is the contract surface, so a
        // field-by-field equality is the right guard).
        let seq = build_intra_only_yuv420_8bit_seq(32, 32);
        let fh_legacy = build_intra_only_yuv420_8bit_fh(&seq, 32, 32);
        let fh_q0 = build_intra_only_yuv420_8bit_fh_with_q(&seq, 32, 32, 0);
        assert_eq!(fh_legacy, fh_q0, "with_q(0) must equal the legacy FH");
    }

    #[test]
    fn fh_with_q_one_carries_base_q_idx_and_switches_to_tx_mode_largest() {
        // At base_q_idx > 0 the FH must declare `TxModeLargest`
        // (§5.9.21 disallows `Only4x4` outside `CodedLossless`).
        let seq = build_intra_only_yuv420_8bit_seq(32, 32);
        let fh = build_intra_only_yuv420_8bit_fh_with_q(&seq, 32, 32, 1);
        let qp = fh.quantization_params.as_ref().expect("intra has qparams");
        assert_eq!(qp.base_q_idx, 1);
        assert_eq!(qp.delta_q_y_dc, 0);
        assert_eq!(qp.delta_q_u_dc, 0);
        assert_eq!(qp.delta_q_u_ac, 0);
        assert_eq!(qp.delta_q_v_dc, 0);
        assert_eq!(qp.delta_q_v_ac, 0);
        assert!(!qp.using_qmatrix);
        assert_eq!(fh.tx_mode, Some(TxMode::TxModeLargest));
    }

    #[test]
    fn fh_with_q_zero_keeps_tx_mode_only_4x4() {
        // The legacy lossless arm uses `Only4x4` per §5.9.21 line 1.
        let seq = build_intra_only_yuv420_8bit_seq(32, 32);
        let fh = build_intra_only_yuv420_8bit_fh_with_q(&seq, 32, 32, 0);
        assert_eq!(fh.tx_mode, Some(TxMode::Only4x4));
    }

    #[test]
    fn fh_with_q_max_still_writes_a_valid_fh() {
        // base_q_idx = 255 ⇒ TxModeLargest. Build + spot-check.
        let seq = build_intra_only_yuv420_8bit_seq(32, 32);
        let fh = build_intra_only_yuv420_8bit_fh_with_q(&seq, 32, 32, 255);
        assert_eq!(
            fh.quantization_params.as_ref().unwrap().base_q_idx,
            255,
            "base_q_idx clamps to u8::MAX"
        );
        assert_eq!(fh.tx_mode, Some(TxMode::TxModeLargest));
    }

    #[test]
    fn encode_with_q_zero_matches_legacy_lossless_byte_for_byte() {
        // Regression guard mirror of the integration test: the lib
        // exposes both `encode_intra_frame_yuv_dyn` and
        // `encode_intra_frame_yuv_dyn_with_q(_, 0)`; they must produce
        // identical IVF bytes + reconstruction.
        let input = Yuv420Frame::filled(32, 32, 128);
        let legacy = encode_intra_frame_yuv_dyn(&input).expect("legacy ok");
        let with_q0 = encode_intra_frame_yuv_dyn_with_q(&input, 0).expect("with_q(0) ok");
        assert_eq!(legacy.ivf_bytes, with_q0.ivf_bytes);
        assert_eq!(legacy.reconstructed_y, with_q0.reconstructed_y);
        assert_eq!(legacy.reconstructed_u, with_q0.reconstructed_u);
        assert_eq!(legacy.reconstructed_v, with_q0.reconstructed_v);
    }

    #[test]
    fn encode_with_q_one_produces_different_bytes_than_q_zero() {
        // q_idx = 1 selects the §7.13.3 DCT path + TxModeLargest; bytes
        // MUST diverge from the lossless WHT/Only4x4 path. (At minimum
        // the §5.9.12 `base_q_idx` slot in the FH is `1` vs `0`.)
        let input = Yuv420Frame::filled(32, 32, 128);
        let q0 = encode_intra_frame_yuv_dyn_with_q(&input, 0).expect("q=0 ok");
        let q1 = encode_intra_frame_yuv_dyn_with_q(&input, 1).expect("q=1 ok");
        assert_ne!(
            q0.ivf_bytes, q1.ivf_bytes,
            "lossy IVF bytes must differ from lossless at the same input"
        );
        assert_eq!(q0.fh.quantization_params.as_ref().unwrap().base_q_idx, 0);
        assert_eq!(q1.fh.quantization_params.as_ref().unwrap().base_q_idx, 1);
    }

    #[test]
    fn encode_with_q_succeeds_across_a_range_of_qindexes() {
        // Smoke test: any qindex 0..=255 produces a parseable IVF byte
        // stream. The full encode → decode loop is exercised in the
        // integration tests; this lib test just makes sure no qindex
        // is rejected at the encoder.
        let input = Yuv420Frame::filled(32, 32, 64);
        for &q in &[0u8, 1, 8, 16, 32, 64, 128, 200, 255] {
            let res = encode_intra_frame_yuv_dyn_with_q(&input, q);
            assert!(res.is_ok(), "encoder must accept base_q_idx = {q}");
            let enc = res.unwrap();
            assert_eq!(enc.fh.quantization_params.as_ref().unwrap().base_q_idx, q);
        }
    }

    #[test]
    fn encode_with_q_validation_still_rejects_invalid_dims() {
        // The validation guard fires before the base_q_idx is consulted
        // — invalid dims at any qindex must be rejected.
        let bad = Yuv420Frame {
            width: 16,
            height: 16,
            y: vec![0u8; 8],
            u: vec![0u8; 8],
            v: vec![0u8; 8],
        };
        assert!(encode_intra_frame_yuv_dyn_with_q(&bad, 0).is_err());
        assert!(encode_intra_frame_yuv_dyn_with_q(&bad, 64).is_err());
    }

    #[test]
    fn encode_with_q_one_recon_equals_recon_via_decoder_pipeline() {
        // The encoder's running reconstructed plane is built by
        // running the decoder's `dequantize_step1 → inverse_transform_2d`
        // pipeline on the encoder's own `forward_quantize` output. The
        // self-decode contract is therefore: feed the produced IVF
        // bytes back through `crate::decoder::decode_av1` and observe
        // the same byte stream as `encoded.reconstructed_*`. This is
        // the lib-level mirror of the integration test.
        use crate::decoder::{decode_av1, Frame};
        let input = Yuv420Frame::filled(32, 32, 200);
        let enc = encode_intra_frame_yuv_dyn_with_q(&input, 1).expect("encode succeeds");
        let dec = decode_av1(&enc.ivf_bytes).expect("decode succeeds");
        match &dec[0] {
            Frame::Yuv420Dyn { y, u, v, .. } => {
                assert_eq!(y, &enc.reconstructed_y);
                assert_eq!(u, &enc.reconstructed_u);
                assert_eq!(v, &enc.reconstructed_v);
            }
            other => panic!("expected Yuv420Dyn, got {other:?}"),
        }
    }

    // ---- r197/r234: rectangular frame-extent coverage. ----
    //
    // The §5.11.4 partition tree's per-quadrant `r >= mi_rows || c >=
    // mi_cols` early return + `EncodeNode::dummy_oob` sentinel already
    // give the dyn driver "rectangular extent" support for free —
    // every quadrant outside the in-frame rectangle gets a
    // SPLIT-recursion that bottoms out into sentinels the
    // §5.11.4 driver swallows on line 1. These tests promote that
    // property to a tested invariant + add fixture coverage at every
    // rectangular extent the §5.11.5 4:2:0 chroma constraint admits
    // within `MIN_DIM..=MAX_DIM`.

    /// `dispatch_order_leaves` must enumerate every in-frame BLOCK_4X4
    /// cell exactly once at a rectangular `(mi_rows, mi_cols)` extent
    /// (where the root super-block is the smallest power-of-two
    /// covering `max(mi_cols, mi_rows)`). Out-of-frame cells must not
    /// be enumerated.
    #[test]
    fn dispatch_order_leaves_covers_every_in_frame_cell_rectangular() {
        // (mi_rows, mi_cols, expected_root_size) — covers every shape
        // class: short+wide, tall+narrow, partial-coverage extents.
        for (mi_rows, mi_cols) in [
            (2u32, 4u32), // 8 × 16
            (4, 2),       // 16 × 8
            (4, 8),       // 16 × 32
            (8, 4),       // 32 × 16
            (6, 8),       // 24 × 32
            (8, 6),       // 32 × 24
            (10, 4),      // 40 × 16
            (4, 10),      // 16 × 40
            (12, 8),      // 48 × 32
            (8, 12),      // 32 × 48
            (16, 8),      // 64 × 32
            (8, 16),      // 32 × 64
            (10, 16),     // 40 × 64
            (16, 10),     // 64 × 40
        ] {
            let root = root_super_block(mi_cols, mi_rows);
            let leaves = dispatch_order_leaves(root, mi_rows, mi_cols);
            let expected = (mi_rows * mi_cols) as usize;
            assert_eq!(
                leaves.len(),
                expected,
                "mi {mi_rows}x{mi_cols} (root b={root}) expected {expected} leaves, got {}",
                leaves.len()
            );
            // Exactly-once coverage.
            let mut seen = vec![false; expected];
            for &(r, c) in &leaves {
                assert!(
                    r < mi_rows && c < mi_cols,
                    "leaf ({r},{c}) outside frame mi {mi_rows}x{mi_cols}"
                );
                let idx = (r * mi_cols + c) as usize;
                assert!(!seen[idx], "cell ({r},{c}) visited twice");
                seen[idx] = true;
            }
            assert!(
                seen.iter().all(|&b| b),
                "mi {mi_rows}x{mi_cols} (root b={root}) missed at least one cell"
            );
        }
    }

    #[test]
    fn root_super_block_handles_rectangular_extents() {
        // The root picks the smallest power-of-two covering
        // max(mi_cols, mi_rows). Every rectangular extent ≤ 64×64
        // therefore lands on BLOCK_16X16 / 32X32 / 64X64.
        // mi 2×4 (8×16) ⇒ max 4 ⇒ BLOCK_16X16
        assert_eq!(root_super_block(2, 4), BLOCK_16X16);
        // mi 4×2 (16×8) ⇒ max 4 ⇒ BLOCK_16X16
        assert_eq!(root_super_block(4, 2), BLOCK_16X16);
        // mi 4×8 (16×32) ⇒ max 8 ⇒ BLOCK_32X32
        assert_eq!(root_super_block(4, 8), BLOCK_32X32);
        // mi 8×4 (32×16) ⇒ max 8 ⇒ BLOCK_32X32
        assert_eq!(root_super_block(8, 4), BLOCK_32X32);
        // mi 10×4 (40×16) ⇒ max 10 ⇒ BLOCK_64X64
        assert_eq!(root_super_block(10, 4), BLOCK_64X64);
        // mi 4×16 (16×64) ⇒ max 16 ⇒ BLOCK_64X64
        assert_eq!(root_super_block(4, 16), BLOCK_64X64);
    }

    #[test]
    fn yuv420_frame_validate_accepts_every_rectangular_aligned_dim() {
        // Sweep every (w, h) pair in {8,16,24,32,40,48,56,64}². validate()
        // accepts the lot, including the strictly-rectangular pairs.
        let dims = [8u32, 16, 24, 32, 40, 48, 56, 64];
        for &w in &dims {
            for &h in &dims {
                let f = Yuv420Frame::filled(w, h, 128);
                assert!(
                    f.validate().is_ok(),
                    "{w}x{h} should validate (every (multiples-of-8) extent ≤ MAX_DIM is in scope)"
                );
            }
        }
    }

    #[test]
    fn encode_flat_grey_rectangular_extents_recon_matches_input() {
        // Lossless WHT arm: every rectangular extent's reconstruction
        // must equal the input plane-for-plane. Spot-checks the
        // four cardinal rectangular shapes against the existing
        // BLOCK_64X64 / 32X32 / 16X16 root super-blocks.
        for (w, h) in [
            (8u32, 16u32),
            (16, 8),
            (16, 32),
            (32, 16),
            (24, 40),
            (40, 24),
            (32, 64),
            (64, 32),
        ] {
            let input = Yuv420Frame::filled(w, h, 128);
            let res = encode_intra_frame_yuv_dyn(&input)
                .unwrap_or_else(|e| panic!("encode {w}x{h} failed: {e:?}"));
            assert_eq!(res.reconstructed_y, input.y, "Y mismatch at {w}x{h}");
            assert_eq!(res.reconstructed_u, input.u, "U mismatch at {w}x{h}");
            assert_eq!(res.reconstructed_v, input.v, "V mismatch at {w}x{h}");
            // FH carries the rectangular extent verbatim.
            let fs = res.fh.frame_size.as_ref().expect("intra has frame_size");
            assert_eq!(fs.frame_width, w);
            assert_eq!(fs.frame_height, h);
        }
    }

    #[test]
    fn encode_with_q_succeeds_on_every_rectangular_extent() {
        // Smoke: every rectangular extent must encode at every qindex
        // we plumb in the lossy arm. Per-leaf reconstruction is asserted
        // by the integration tests; this lib test just confirms the
        // encoder never refuses a rectangular shape at q > 0.
        for (w, h) in [
            (8u32, 16u32),
            (16, 8),
            (16, 32),
            (32, 16),
            (24, 40),
            (40, 24),
            (48, 32),
            (32, 48),
            (32, 64),
            (64, 32),
        ] {
            for &q in &[1u8, 16, 64, 255] {
                let input = Yuv420Frame::filled(w, h, 64);
                let res = encode_intra_frame_yuv_dyn_with_q(&input, q);
                assert!(
                    res.is_ok(),
                    "encoder rejected {w}x{h} at q={q}: {:?}",
                    res.err()
                );
                let enc = res.unwrap();
                assert_eq!(enc.fh.quantization_params.as_ref().unwrap().base_q_idx, q);
                let fs = enc.fh.frame_size.as_ref().unwrap();
                assert_eq!(fs.frame_width, w);
                assert_eq!(fs.frame_height, h);
            }
        }
    }

    #[test]
    fn encode_rectangular_lossy_self_decodes_byte_for_byte() {
        // Lib-level mirror of the integration test: lossy encode of a
        // rectangular extent + immediate decode_av1 of the produced IVF
        // bytes ⇒ recovered planes equal `encoded.reconstructed_*`.
        // Exercises the §7.13.3 inverse-DCT decoder path against
        // partial-coverage out-of-frame quadrants.
        use crate::decoder::{decode_av1, Frame};
        for (w, h) in [(16u32, 32u32), (32, 16), (8, 16), (16, 8)] {
            let mut input = Yuv420Frame::filled(w, h, 0);
            let mut state: u64 = (w as u64).wrapping_mul(2654435761).wrapping_add(h as u64);
            let mut next = || -> u8 {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 56) as u8
            };
            for p in input.y.iter_mut() {
                *p = next();
            }
            for p in input.u.iter_mut() {
                *p = next();
            }
            for p in input.v.iter_mut() {
                *p = next();
            }
            let enc = encode_intra_frame_yuv_dyn_with_q(&input, 16)
                .unwrap_or_else(|e| panic!("encode {w}x{h} q=16 failed: {e:?}"));
            let dec = decode_av1(&enc.ivf_bytes)
                .unwrap_or_else(|e| panic!("decode {w}x{h} q=16 failed: {e:?}"));
            match &dec[0] {
                Frame::Yuv420Dyn {
                    width,
                    height,
                    y,
                    u,
                    v,
                } => {
                    assert_eq!(*width, w, "decoded width mismatch at {w}x{h}");
                    assert_eq!(*height, h, "decoded height mismatch at {w}x{h}");
                    assert_eq!(y, &enc.reconstructed_y, "Y mismatch at {w}x{h}");
                    assert_eq!(u, &enc.reconstructed_u, "U mismatch at {w}x{h}");
                    assert_eq!(v, &enc.reconstructed_v, "V mismatch at {w}x{h}");
                }
                other => panic!("expected Yuv420Dyn at {w}x{h}, got {other:?}"),
            }
        }
    }

    // ---- r235 — Y-only dyn driver lib tests ----

    #[test]
    fn y_dyn_seq_carries_monochrome_flag() {
        let seq = build_intra_only_y_8bit_seq(32, 32);
        assert!(seq.color_config.mono_chrome, "Y-only SH ⇒ mono_chrome");
        assert_eq!(seq.color_config.num_planes, 1, "Y-only SH ⇒ num_planes 1");
        // The mono arm still requires subsampling = true per §5.5.2.
        assert!(seq.color_config.subsampling_x);
        assert!(seq.color_config.subsampling_y);
    }

    #[test]
    fn y_dyn_validate_rejects_mismatched_y_len() {
        let bad = MonoYFrame {
            width: 16,
            height: 16,
            y: vec![0u8; 8],
        };
        assert!(bad.validate().is_err());
        let bad_dim = MonoYFrame {
            width: 12,
            height: 16,
            y: vec![0u8; 12 * 16],
        };
        assert!(bad_dim.validate().is_err());
    }

    #[test]
    fn y_dyn_encode_flat_grey_recon_matches_input() {
        let input = MonoYFrame::filled(32, 32, 128);
        let res = encode_intra_frame_y_dyn(&input).expect("encode ok");
        assert_eq!(res.reconstructed_y, input.y);
        // The SH+FH the encoder emitted carry the mono flag.
        assert!(res.seq.color_config.mono_chrome);
        assert_eq!(res.seq.color_config.num_planes, 1);
    }

    #[test]
    fn y_dyn_encode_pseudorandom_internal_recon_bit_exact_on_lossless_arm() {
        let mut input = MonoYFrame::filled(40, 32, 0);
        let mut s: u64 = 0xFACE_C0DE_BABE_1234;
        for p in input.y.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = (s >> 56) as u8;
        }
        let res = encode_intra_frame_y_dyn(&input).expect("encode ok");
        assert_eq!(
            res.reconstructed_y, input.y,
            "lossless WHT arm ⇒ encoder-internal recon equals input plane"
        );
    }

    #[test]
    fn y_dyn_encode_with_q_zero_matches_legacy_byte_for_byte() {
        let input = MonoYFrame::filled(32, 32, 128);
        let legacy = encode_intra_frame_y_dyn(&input).expect("legacy ok");
        let with_q0 = encode_intra_frame_y_dyn_with_q(&input, 0).expect("with_q(0) ok");
        assert_eq!(legacy.ivf_bytes, with_q0.ivf_bytes);
        assert_eq!(legacy.reconstructed_y, with_q0.reconstructed_y);
    }

    #[test]
    fn y_dyn_encode_with_q_one_diverges_from_lossless_arm() {
        let input = MonoYFrame::filled(32, 32, 128);
        let q0 = encode_intra_frame_y_dyn_with_q(&input, 0).expect("q=0 ok");
        let q1 = encode_intra_frame_y_dyn_with_q(&input, 1).expect("q=1 ok");
        assert_ne!(q0.ivf_bytes, q1.ivf_bytes);
        assert_eq!(q0.fh.quantization_params.as_ref().unwrap().base_q_idx, 0);
        assert_eq!(q1.fh.quantization_params.as_ref().unwrap().base_q_idx, 1);
    }

    // ---- r207 — multi-SB Y-only dyn driver lib tests ----

    #[test]
    fn sb_grid_origins_single_sb_for_le_64() {
        // 64×64 ⇒ mi 16×16 ⇒ 1 SB at (0, 0).
        let v = sb_grid_origins(16, 16);
        assert_eq!(v, vec![(0, 0)]);
        // 64×32 ⇒ mi 8×16 ⇒ 1 SB at (0, 0) (row exhausted in <16).
        let v = sb_grid_origins(8, 16);
        assert_eq!(v, vec![(0, 0)]);
        // 8×8 ⇒ mi 2×2 ⇒ still 1 SB at (0, 0).
        let v = sb_grid_origins(2, 2);
        assert_eq!(v, vec![(0, 0)]);
    }

    #[test]
    fn sb_grid_origins_multi_sb_row_major() {
        // 96×64 ⇒ mi 16×24 (rows × cols) ⇒ 2 SBs in the row at
        // c = {0, 16}, 1 SB row only ⇒ [(0,0), (0,16)].
        let v = sb_grid_origins(16, 24);
        assert_eq!(v, vec![(0, 0), (0, 16)]);
        // 128×64 ⇒ mi 16×32 ⇒ 2 SB cols ⇒ [(0,0), (0,16)].
        let v = sb_grid_origins(16, 32);
        assert_eq!(v, vec![(0, 0), (0, 16)]);
        // 64×128 ⇒ mi 32×16 ⇒ 2 SB rows ⇒ [(0,0), (16,0)].
        let v = sb_grid_origins(32, 16);
        assert_eq!(v, vec![(0, 0), (16, 0)]);
        // 128×128 ⇒ mi 32×32 ⇒ 2×2 grid ⇒ row-major.
        let v = sb_grid_origins(32, 32);
        assert_eq!(v, vec![(0, 0), (0, 16), (16, 0), (16, 16)]);
        // 96×96 ⇒ mi 24×24 ⇒ 2×2 grid still (the bottom / right SBs
        // are partial; each spans mi 8 × 8 in-frame).
        let v = sb_grid_origins(24, 24);
        assert_eq!(v, vec![(0, 0), (0, 16), (16, 0), (16, 16)]);
    }

    #[test]
    fn sb_grid_dispatch_order_leaves_covers_every_in_frame_cell_96x64() {
        // 96×64 ⇒ mi 16×24 ⇒ expect 16 × 24 / 1 = 24×16 = 384 cells.
        let leaves = sb_grid_dispatch_order_leaves(16, 24);
        assert_eq!(leaves.len(), 16 * 24);
        let mut seen = vec![false; 16 * 24];
        for &(r, c) in &leaves {
            assert!(r < 16 && c < 24);
            let idx = (r * 24 + c) as usize;
            assert!(!seen[idx], "({r},{c}) visited twice");
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn sb_grid_dispatch_order_leaves_covers_every_in_frame_cell_128x128() {
        // 128×128 ⇒ mi 32×32 ⇒ 1024 cells; 2×2 SB grid.
        let leaves = sb_grid_dispatch_order_leaves(32, 32);
        assert_eq!(leaves.len(), 1024);
        let mut seen = vec![false; 1024];
        for &(r, c) in &leaves {
            let idx = (r * 32 + c) as usize;
            assert!(!seen[idx]);
            seen[idx] = true;
        }
        assert!(seen.iter().all(|&b| b));
    }

    #[test]
    fn mono_y_multi_sb_validate_accepts_extents_up_to_128() {
        // Every (w, h) ∈ {8..=128} aligned to 8 validates.
        for w in (MIN_DIM..=MAX_DIM_Y_MULTI_SB).step_by(MIN_DIM as usize) {
            for h in (MIN_DIM..=MAX_DIM_Y_MULTI_SB).step_by(MIN_DIM as usize) {
                let f = MonoYFrameMultiSb::filled(w, h, 0);
                assert!(f.validate().is_ok(), "{w}x{h} should validate");
            }
        }
    }

    #[test]
    fn mono_y_multi_sb_validate_rejects_oversized_dims() {
        // 136 > 128 ⇒ reject.
        let bad = MonoYFrameMultiSb {
            width: 136,
            height: 64,
            y: vec![0u8; 136 * 64],
        };
        assert!(bad.validate().is_err());
        // Wrong y length.
        let bad = MonoYFrameMultiSb {
            width: 96,
            height: 64,
            y: vec![0u8; 100],
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn mono_y_multi_sb_encode_96x64_flat_grey_internal_recon_matches_input() {
        let input = MonoYFrameMultiSb::filled(96, 64, 128);
        let res = encode_intra_frame_y_dyn_multi_sb(&input).expect("encode ok");
        assert_eq!(res.reconstructed_y, input.y);
        assert!(res.seq.color_config.mono_chrome);
        assert_eq!(res.seq.color_config.num_planes, 1);
        // SH advertises max_frame_width / max_frame_height matching the
        // input extent.
        assert_eq!(res.seq.max_frame_width_minus_1, 95);
        assert_eq!(res.seq.max_frame_height_minus_1, 63);
    }

    #[test]
    fn mono_y_multi_sb_encode_128x128_pseudorandom_lossless_recon_bit_exact() {
        let mut input = MonoYFrameMultiSb::filled(128, 128, 0);
        let mut s: u64 = 0xC001_F00D_DEAD_BEEF;
        for p in input.y.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = (s >> 56) as u8;
        }
        let res = encode_intra_frame_y_dyn_multi_sb(&input).expect("encode ok");
        assert_eq!(
            res.reconstructed_y, input.y,
            "128×128 lossless WHT arm ⇒ encoder-internal recon equals input"
        );
    }

    #[test]
    fn mono_y_multi_sb_encode_with_q_zero_matches_legacy_byte_for_byte_at_64x64() {
        // The multi-SB path at extent ≤ 64×64 still walks one SB.
        // The single root selected by `root_super_block(16, 16) =
        // BLOCK_64X64` happens to match the multi-SB `BLOCK_64X64`
        // root, so at exactly 64×64 the byte stream from the legacy
        // entry and the multi-SB entry must coincide.
        let input_y = MonoYFrame::filled(64, 64, 200);
        let legacy = encode_intra_frame_y_dyn(&input_y).expect("legacy ok");
        let multi = encode_intra_frame_y_dyn_multi_sb(&MonoYFrameMultiSb {
            width: 64,
            height: 64,
            y: input_y.y.clone(),
        })
        .expect("multi ok");
        assert_eq!(
            legacy.ivf_bytes, multi.ivf_bytes,
            "at 64×64 the multi-SB and single-SB Y-only paths must produce identical IVF bytes"
        );
        assert_eq!(legacy.reconstructed_y, multi.reconstructed_y);
    }

    #[test]
    fn mono_y_multi_sb_encode_q_diverges_from_lossless() {
        let input = MonoYFrameMultiSb::filled(96, 64, 64);
        let q0 = encode_intra_frame_y_dyn_multi_sb_with_q(&input, 0).expect("q=0");
        let q32 = encode_intra_frame_y_dyn_multi_sb_with_q(&input, 32).expect("q=32");
        assert_ne!(q0.ivf_bytes, q32.ivf_bytes);
        assert_eq!(q0.fh.quantization_params.as_ref().unwrap().base_q_idx, 0);
        assert_eq!(q32.fh.quantization_params.as_ref().unwrap().base_q_idx, 32);
    }

    /// End-to-end encode → `decode_av1` → pixel-equality contract for
    /// the multi-SB Y-only path at representative extents that
    /// exercise:
    ///   * the legacy ≤ 64×64 single-SB regime (64×64),
    ///   * one extra SB along x (96×64),
    ///   * one extra SB along y (64×96),
    ///   * the full 2×2 SB grid at 128×128,
    ///   * a partial-coverage edge SB (104×72).
    #[test]
    fn mono_y_multi_sb_encode_decode_lossless_roundtrip_representative_extents() {
        use crate::decoder::{decode_av1, Frame};
        let extents: &[(u32, u32)] = &[
            (64, 64),
            (72, 64),
            (96, 64),
            (64, 72),
            (64, 96),
            (104, 72),
            (128, 64),
            (64, 128),
            (96, 96),
            (128, 128),
        ];
        let mut s: u64 = 0xBEEF_C0DE_F00D_FACE;
        for &(w, h) in extents {
            let mut input = MonoYFrameMultiSb::filled(w, h, 0);
            for p in input.y.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *p = (s >> 56) as u8;
            }
            let enc = encode_intra_frame_y_dyn_multi_sb(&input)
                .unwrap_or_else(|_| panic!("encode failed at {w}×{h}"));
            assert_eq!(
                enc.reconstructed_y, input.y,
                "encoder-internal recon must equal input at {w}×{h} on the lossless WHT arm"
            );
            let dec =
                decode_av1(&enc.ivf_bytes).unwrap_or_else(|_| panic!("decode failed at {w}×{h}"));
            assert_eq!(dec.len(), 1);
            match &dec[0] {
                Frame::YDyn { width, height, y } => {
                    assert_eq!(*width, w, "decoded width mismatch at {w}×{h}");
                    assert_eq!(*height, h, "decoded height mismatch at {w}×{h}");
                    assert_eq!(y, &enc.reconstructed_y, "Y mismatch at {w}×{h}");
                    assert_eq!(y, &input.y, "bit-exact recovery at {w}×{h}");
                }
                other => panic!("expected YDyn at {w}×{h}, got {other:?}"),
            }
        }
    }

    /// Lossy contract on the multi-SB path: encoder-internal recon =
    /// `decode_av1(enc.ivf_bytes)` byte-for-byte at q > 0 (the
    /// recovered Y plane does NOT in general equal the input —
    /// quantization rounds). Covers a 96×64 extent at q ∈ {1, 32, 200}.
    #[test]
    fn mono_y_multi_sb_lossy_encode_decode_self_consistency() {
        use crate::decoder::{decode_av1, Frame};
        let mut input = MonoYFrameMultiSb::filled(96, 64, 0);
        let mut s: u64 = 0xABCD_1234_5678_9ABC;
        for p in input.y.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *p = (s >> 56) as u8;
        }
        for &q in &[1u8, 32, 200] {
            let enc = encode_intra_frame_y_dyn_multi_sb_with_q(&input, q)
                .unwrap_or_else(|_| panic!("encode failed at q={q}"));
            let dec =
                decode_av1(&enc.ivf_bytes).unwrap_or_else(|_| panic!("decode failed at q={q}"));
            match &dec[0] {
                Frame::YDyn { width, height, y } => {
                    assert_eq!(*width, 96);
                    assert_eq!(*height, 64);
                    assert_eq!(
                        y, &enc.reconstructed_y,
                        "encoder recon = decoder out at q={q}"
                    );
                }
                other => panic!("expected YDyn, got {other:?}"),
            }
        }
    }

    // ---- r214 — multi-SB 4:2:0 YUV dyn driver lib tests ----

    #[test]
    fn yuv_multi_sb_validate_accepts_extents_up_to_128() {
        // Every (w, h) ∈ {8..=128} aligned to 8 validates.
        for w in (MIN_DIM..=MAX_DIM_YUV_MULTI_SB).step_by(MIN_DIM as usize) {
            for h in (MIN_DIM..=MAX_DIM_YUV_MULTI_SB).step_by(MIN_DIM as usize) {
                let f = Yuv420FrameMultiSb::filled(w, h, 128);
                assert!(f.validate().is_ok(), "{w}x{h} should validate");
            }
        }
    }

    #[test]
    fn yuv_multi_sb_validate_rejects_oversized_dims() {
        // 136 > 128 ⇒ reject.
        let bad = Yuv420FrameMultiSb {
            width: 136,
            height: 64,
            y: vec![0u8; 136 * 64],
            u: vec![0u8; (136 / 2) * (64 / 2)],
            v: vec![0u8; (136 / 2) * (64 / 2)],
        };
        assert!(bad.validate().is_err());
        // Wrong y length.
        let bad = Yuv420FrameMultiSb {
            width: 96,
            height: 64,
            y: vec![0u8; 100],
            u: vec![0u8; 48 * 32],
            v: vec![0u8; 48 * 32],
        };
        assert!(bad.validate().is_err());
        // Wrong U length.
        let bad = Yuv420FrameMultiSb {
            width: 96,
            height: 64,
            y: vec![0u8; 96 * 64],
            u: vec![0u8; 8],
            v: vec![0u8; 48 * 32],
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn yuv_multi_sb_encode_96x64_flat_grey_internal_recon_matches_input() {
        let input = Yuv420FrameMultiSb::filled(96, 64, 128);
        let res = encode_intra_frame_yuv_dyn_multi_sb(&input).expect("encode ok");
        assert_eq!(res.reconstructed_y, input.y);
        assert_eq!(res.reconstructed_u, input.u);
        assert_eq!(res.reconstructed_v, input.v);
        // SH advertises 4:2:0 layout (not mono).
        assert!(!res.seq.color_config.mono_chrome);
        assert_eq!(res.seq.color_config.num_planes, 3);
        assert_eq!(res.seq.max_frame_width_minus_1, 95);
        assert_eq!(res.seq.max_frame_height_minus_1, 63);
    }

    #[test]
    fn yuv_multi_sb_encode_128x128_pseudorandom_lossless_recon_bit_exact() {
        let mut input = Yuv420FrameMultiSb::filled(128, 128, 0);
        let mut s: u64 = 0x5A5A_A5A5_5A5A_A5A5;
        let mut next = || -> u8 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for p in input.y.iter_mut() {
            *p = next();
        }
        for p in input.u.iter_mut() {
            *p = next();
        }
        for p in input.v.iter_mut() {
            *p = next();
        }
        let res = encode_intra_frame_yuv_dyn_multi_sb(&input).expect("encode ok");
        assert_eq!(
            res.reconstructed_y, input.y,
            "128×128 Y lossless WHT mismatch"
        );
        assert_eq!(
            res.reconstructed_u, input.u,
            "128×128 U lossless WHT mismatch"
        );
        assert_eq!(
            res.reconstructed_v, input.v,
            "128×128 V lossless WHT mismatch"
        );
    }

    #[test]
    fn yuv_multi_sb_encode_with_q_zero_matches_legacy_byte_for_byte_at_64x64() {
        // At extent ≤ 64×64 the multi-SB path walks a single SB rooted
        // at BLOCK_64X64, which is exactly what `root_super_block(16,
        // 16)` returns for the single-SB path. The IVF bytes must
        // therefore coincide.
        let mut s: u64 = 0x1357_9BDF_2468_ACE0;
        let mut next = || -> u8 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        let mut yv = vec![0u8; 64 * 64];
        for p in yv.iter_mut() {
            *p = next();
        }
        let mut uv = vec![0u8; 32 * 32];
        for p in uv.iter_mut() {
            *p = next();
        }
        let mut vv = vec![0u8; 32 * 32];
        for p in vv.iter_mut() {
            *p = next();
        }
        let legacy_input = Yuv420Frame {
            width: 64,
            height: 64,
            y: yv.clone(),
            u: uv.clone(),
            v: vv.clone(),
        };
        let multi_input = Yuv420FrameMultiSb {
            width: 64,
            height: 64,
            y: yv,
            u: uv,
            v: vv,
        };
        let legacy = encode_intra_frame_yuv_dyn(&legacy_input).expect("legacy ok");
        let multi = encode_intra_frame_yuv_dyn_multi_sb(&multi_input).expect("multi ok");
        assert_eq!(
            legacy.ivf_bytes, multi.ivf_bytes,
            "at 64×64 the multi-SB and single-SB YUV paths must produce identical IVF bytes"
        );
        assert_eq!(legacy.reconstructed_y, multi.reconstructed_y);
        assert_eq!(legacy.reconstructed_u, multi.reconstructed_u);
        assert_eq!(legacy.reconstructed_v, multi.reconstructed_v);
    }

    #[test]
    fn yuv_multi_sb_encode_q_diverges_from_lossless() {
        let input = Yuv420FrameMultiSb::filled(96, 64, 64);
        let q0 = encode_intra_frame_yuv_dyn_multi_sb_with_q(&input, 0).expect("q=0");
        let q32 = encode_intra_frame_yuv_dyn_multi_sb_with_q(&input, 32).expect("q=32");
        assert_ne!(q0.ivf_bytes, q32.ivf_bytes);
        assert_eq!(q0.fh.quantization_params.as_ref().unwrap().base_q_idx, 0);
        assert_eq!(q32.fh.quantization_params.as_ref().unwrap().base_q_idx, 32);
    }

    /// End-to-end encode → `decode_av1` → pixel-equality contract for
    /// the multi-SB YUV path at representative extents that exercise:
    ///   * the legacy ≤ 64×64 single-SB regime (64×64),
    ///   * one extra SB along x (96×64) — partial-coverage edge SB,
    ///   * one extra SB along y (64×96),
    ///   * the full 2×2 SB grid at 128×128,
    ///   * a partial-coverage corner SB (104×72).
    #[test]
    fn yuv_multi_sb_encode_decode_lossless_roundtrip_representative_extents() {
        use crate::decoder::{decode_av1, Frame};
        let extents: &[(u32, u32)] = &[
            (64, 64),
            (72, 64),
            (96, 64),
            (64, 72),
            (64, 96),
            (104, 72),
            (128, 64),
            (64, 128),
            (96, 96),
            (128, 128),
        ];
        let mut s: u64 = 0xDEAD_BEEF_C0DE_FACE;
        let next = |s: &mut u64| -> u8 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 56) as u8
        };
        for &(w, h) in extents {
            let mut input = Yuv420FrameMultiSb::filled(w, h, 0);
            for p in input.y.iter_mut() {
                *p = next(&mut s);
            }
            for p in input.u.iter_mut() {
                *p = next(&mut s);
            }
            for p in input.v.iter_mut() {
                *p = next(&mut s);
            }
            let enc = encode_intra_frame_yuv_dyn_multi_sb(&input)
                .unwrap_or_else(|_| panic!("encode failed at {w}×{h}"));
            assert_eq!(
                enc.reconstructed_y, input.y,
                "Y recon must equal input at {w}×{h} on the lossless WHT arm"
            );
            assert_eq!(
                enc.reconstructed_u, input.u,
                "U recon must equal input at {w}×{h} on the lossless WHT arm"
            );
            assert_eq!(
                enc.reconstructed_v, input.v,
                "V recon must equal input at {w}×{h} on the lossless WHT arm"
            );
            let dec =
                decode_av1(&enc.ivf_bytes).unwrap_or_else(|_| panic!("decode failed at {w}×{h}"));
            assert_eq!(dec.len(), 1);
            match &dec[0] {
                Frame::Yuv420Dyn {
                    width,
                    height,
                    y,
                    u,
                    v,
                } => {
                    assert_eq!(*width, w, "decoded width mismatch at {w}×{h}");
                    assert_eq!(*height, h, "decoded height mismatch at {w}×{h}");
                    assert_eq!(y, &enc.reconstructed_y, "Y mismatch at {w}×{h}");
                    assert_eq!(u, &enc.reconstructed_u, "U mismatch at {w}×{h}");
                    assert_eq!(v, &enc.reconstructed_v, "V mismatch at {w}×{h}");
                    assert_eq!(y, &input.y, "Y bit-exact recovery at {w}×{h}");
                    assert_eq!(u, &input.u, "U bit-exact recovery at {w}×{h}");
                    assert_eq!(v, &input.v, "V bit-exact recovery at {w}×{h}");
                }
                other => panic!("expected Yuv420Dyn at {w}×{h}, got {other:?}"),
            }
        }
    }

    /// Lossy contract on the multi-SB YUV path: encoder-internal recon
    /// = `decode_av1(enc.ivf_bytes)` byte-for-byte on all three planes
    /// at q > 0 (the recovered planes do NOT in general equal the
    /// input — quantization rounds).
    #[test]
    fn yuv_multi_sb_lossy_encode_decode_self_consistency() {
        use crate::decoder::{decode_av1, Frame};
        let mut input = Yuv420FrameMultiSb::filled(96, 64, 0);
        let mut s: u64 = 0xBAD_F00D_FEED_BEEF;
        let next = |s: &mut u64| -> u8 {
            *s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*s >> 56) as u8
        };
        for p in input.y.iter_mut() {
            *p = next(&mut s);
        }
        for p in input.u.iter_mut() {
            *p = next(&mut s);
        }
        for p in input.v.iter_mut() {
            *p = next(&mut s);
        }
        for &q in &[1u8, 32, 200] {
            let enc = encode_intra_frame_yuv_dyn_multi_sb_with_q(&input, q)
                .unwrap_or_else(|_| panic!("encode failed at q={q}"));
            let dec =
                decode_av1(&enc.ivf_bytes).unwrap_or_else(|_| panic!("decode failed at q={q}"));
            match &dec[0] {
                Frame::Yuv420Dyn {
                    width,
                    height,
                    y,
                    u,
                    v,
                } => {
                    assert_eq!(*width, 96);
                    assert_eq!(*height, 64);
                    assert_eq!(
                        y, &enc.reconstructed_y,
                        "Y encoder recon = decoder out at q={q}"
                    );
                    assert_eq!(
                        u, &enc.reconstructed_u,
                        "U encoder recon = decoder out at q={q}"
                    );
                    assert_eq!(
                        v, &enc.reconstructed_v,
                        "V encoder recon = decoder out at q={q}"
                    );
                }
                other => panic!("expected Yuv420Dyn, got {other:?}"),
            }
        }
    }

    /// IVF v0 width/height encode-side carriage on multi-SB YUV
    /// extents — bytes 12..14 (width LE) and 14..16 (height LE) carry
    /// the input dimensions verbatim across the SB threshold.
    #[test]
    fn yuv_multi_sb_encoded_ivf_carries_dynamic_dimensions_in_header() {
        for &(w, h) in &[(96u32, 64u32), (128, 64), (64, 128), (128, 128), (104, 72)] {
            let input = Yuv420FrameMultiSb::filled(w, h, 128);
            let res = encode_intra_frame_yuv_dyn_multi_sb(&input).expect("encode ok");
            let ivf_w = u16::from_le_bytes([res.ivf_bytes[12], res.ivf_bytes[13]]);
            let ivf_h = u16::from_le_bytes([res.ivf_bytes[14], res.ivf_bytes[15]]);
            assert_eq!(ivf_w as u32, w, "IVF width at {w}×{h}");
            assert_eq!(ivf_h as u32, h, "IVF height at {w}×{h}");
        }
    }

    /// At every allowed (w, h) ∈ {8..=64} × {8..=64} aligned to 8, the
    /// Y-only dyn driver must produce a parseable IVF + a Y plane the
    /// dyn-Y decoder reproduces bit-exactly on the lossless WHT arm.
    #[test]
    fn y_dyn_encode_decode_lossless_roundtrip_every_extent() {
        use crate::decoder::{decode_av1, Frame};
        // Sample a representative set of extents (square + rectangular).
        let extents: &[(u32, u32)] = &[
            (8, 8),
            (16, 16),
            (24, 24),
            (32, 32),
            (40, 40),
            (48, 48),
            (56, 56),
            (64, 64),
            (8, 16),
            (16, 8),
            (24, 32),
            (32, 24),
            (40, 16),
            (16, 40),
        ];
        let mut s: u64 = 0xC001_BABE_F00D_FACE;
        for &(w, h) in extents {
            let mut input = MonoYFrame::filled(w, h, 0);
            for p in input.y.iter_mut() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                *p = (s >> 56) as u8;
            }
            let enc = encode_intra_frame_y_dyn(&input)
                .unwrap_or_else(|_| panic!("encode failed at {w}x{h}"));
            let dec =
                decode_av1(&enc.ivf_bytes).unwrap_or_else(|_| panic!("decode failed at {w}x{h}"));
            match &dec[0] {
                Frame::YDyn { width, height, y } => {
                    assert_eq!(*width, w, "decoded width mismatch at {w}x{h}");
                    assert_eq!(*height, h, "decoded height mismatch at {w}x{h}");
                    assert_eq!(y, &enc.reconstructed_y, "Y plane mismatch at {w}x{h}");
                    assert_eq!(y, &input.y, "Y bit-exact recovery at {w}x{h}");
                }
                other => panic!("expected YDyn at {w}x{h}, got {other:?}"),
            }
        }
    }
}
