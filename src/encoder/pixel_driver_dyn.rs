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
//! ## Scope (arc r230)
//!
//! * `width`, `height` ∈ {8, 16, 24, 32, 40, 48, 56, 64} (multiples
//!   of 8, ≤ 64).
//! * `subsampling_x = subsampling_y = 1` (4:2:0), `bit_depth = 8`,
//!   `monochrome = false`.
//! * Intra-only, single tile, `base_q_idx = 0` (lossless WHT arm),
//!   `tx_size = TX_4X4` everywhere, no segmentation, no QM, no
//!   in-loop filters.
//! * 13-mode intra picker on luma + chroma (the r228/r229 picker).
//!
//! ## What this arc does NOT do
//!
//! * Frames > 64×64 (multi-super-block tiling).
//! * Non-8 bit-depth, non-4:2:0 sampling, monochrome.
//! * `base_q_idx > 0` (only the lossless WHT arm is wired).
//! * Larger transform blocks; rectangular partitions; inter mode_info.
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
    NUM_4X4_BLOCKS_WIDE, PARTITION_NONE, PARTITION_SPLIT, TX_4X4, TX_CLASS_2D,
};
use crate::encoder::forward_quantize::forward_quantize;
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
/// (`BLOCK_64X64`). Multi-super-block tiling is a follow-up.
pub const MAX_DIM: u32 = 64;

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
#[must_use]
pub fn build_intra_only_yuv420_8bit_fh(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
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
        base_q_idx: 0,
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
        tx_mode: Some(TxMode::Only4x4),
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

#[allow(clippy::too_many_arguments)]
fn pick_best_intra_mode_4x4_chroma_joint(
    recon_u: &[u8],
    recon_v: &[u8],
    input_u: &[u8],
    input_v: &[u8],
    plane_width: usize,
    plane_height: usize,
    row0: usize,
    col0: usize,
) -> (u8, [u8; 16], [u8; 16]) {
    let (ha_u, hl_u, above_u, left_u, al_u) =
        derive_intra_neighbours_4x4(recon_u, plane_width, plane_height, row0, col0);
    let (ha_v, hl_v, above_v, left_v, al_v) =
        derive_intra_neighbours_4x4(recon_v, plane_width, plane_height, row0, col0);
    let mut best_mode = DC_PRED as u8;
    let mut best_pred_u = [0u8; 16];
    let mut best_pred_v = [0u8; 16];
    let mut best_ssd: u64 = u64::MAX;
    for &mode in &INTRA_MODE_CANDIDATES {
        let Some(pred_u) = predict_intra_mode_4x4(mode, ha_u, hl_u, &above_u, &left_u, al_u) else {
            continue;
        };
        let Some(pred_v) = predict_intra_mode_4x4(mode, ha_v, hl_v, &above_v, &left_v, al_v) else {
            continue;
        };
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
        }
    }
    (best_mode, best_pred_u, best_pred_v)
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
/// ## Errors
///
/// * `input.validate()` fails (dimensions out of range / wrong plane
///   length) ⇒ [`Error::PartitionWalkOutOfRange`].
/// * Any internal partition-tree / coefficient writer overflow.
pub fn encode_intra_frame_yuv_dyn(input: &Yuv420Frame) -> Result<EncodedFrameDyn, Error> {
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
    let fh = build_intra_only_yuv420_8bit_fh(&seq, input.width, input.height);

    // Lossless arm — encoder mirrors the §5.9.2 CodedLossless predicate
    // (base_q_idx == 0 && every delta_q == 0).
    let qp = QuantizerParams::neutral(0, 8);
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
        let coeffs_y = forward_wht_4x4(&residual_y).to_vec();
        let quant_y = forward_quantize(&coeffs_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);

        let dequant_y = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, &qp);
        let resid_back_y = inverse_transform_2d(&dequant_y, TX_4X4, DCT_DCT, 8, true);
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
        let uv_mode_picked = if has_chroma {
            let cr = ((mi_r as usize) - 1) / 2;
            let cc = ((mi_c as usize) - 1) / 2;
            let crow0 = cr * 4;
            let ccol0 = cc * 4;
            let (uv_pick, pred_u, pred_v) = pick_best_intra_mode_4x4_chroma_joint(
                &recon_u,
                &recon_v,
                &input.u,
                &input.v,
                chroma_width,
                chroma_height,
                crow0,
                ccol0,
            );
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
                let coeffs_c = forward_wht_4x4(&residual_c).to_vec();
                let quant_c = forward_quantize(&coeffs_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);

                let dequant_c = dequantize_step1(&quant_c, TX_4X4, plane, 0, DCT_DCT, 15, &qp);
                let resid_back_c = inverse_transform_2d(&dequant_c, TX_4X4, DCT_DCT, 8, true);
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
}
