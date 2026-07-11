//! Conformance-grade single-reference inter P-frame encoder (r411).
//!
//! Extends the r409/r410 KEY-frame driver ([`super::key_frame`]) into
//! a GOP encoder: [`encode_gop_yuv420_with_q`] emits one KEY frame
//! followed by INTER P-frames, each predicting from the previous
//! frame's reconstruction (`LAST_FRAME`, every §7.20 slot refreshed
//! per frame), through the REAL §5.11.18 `inter_frame_mode_info()`
//! syntax (the r411 [`super::partition_tree`] write arm, whose output
//! the spec decode walker replays bit-for-bit).
//!
//! ## Scope (r411)
//!
//! * 8-bit 4:2:0 YUV input, dimensions per the KEY-frame rules
//!   (multiples of 8 in `[8, KEY_FRAME_MAX_DIM]`).
//! * P-frame header: `error_resilient_mode = 1` (forcing
//!   `primary_ref_frame = PRIMARY_REF_NONE` — per-frame default CDFs),
//!   `refresh_frame_flags = allFrames`, all seven `ref_frame_idx`
//!   slots at 0, identity §5.9.24 global motion, `EIGHTTAP`
//!   non-switchable filter, `force_integer_mv = 0` /
//!   `allow_high_precision_mv = 0`, no order hints, no skip-mode, no
//!   segmentation; `TxMode = ONLY_4X4` on the `CodedLossless` arm
//!   (`base_q_idx == 0`) or `TX_MODE_SELECT` otherwise.
//! * Per-node RD search (leaf-vs-split, `BLOCK_64X64` down to
//!   `BLOCK_8X8` — inter frames stop above the sub-8×8 chroma
//!   `someUseIntra` stitching): every square node trials one INTER
//!   leaf (integer motion search + half/quarter-pel refinement
//!   through the real §7.11.3.4 kernel, coding `NEWMV`, or `GLOBALMV`
//!   on the zero vector — the identity-warp derivation) against one
//!   INTRA leaf (the §5.11.22 arm with the KEY driver's 13-mode +
//!   §5.11.15 tx_depth pickers), then both against the recursive
//!   split. Inter leaves additionally RD-select their §5.11.17
//!   uniform `txfm_split` depth (`Max_Tx_Size_Rect` down to two
//!   `Split_Tx_Size` steps), coding the recursion's TUs in §5.11.36
//!   transform-tree order.
//! * `skip = 1` on leaves whose every TU quantises to zero.
//!
//! ## Why the reconstruction loop is exact
//!
//! Inter prediction runs through the DECODER'S OWN §7.11.3 driver
//! ([`crate::inter_pred::reconstruct_inter_leaf_at`]) over grid state
//! stamped exactly like the §5.11.5 walker's (`MiSizes` / `IsInters` /
//! `RefFrames` / `Mvs` / `InterpFilters` / `MotionModes` / `YModes`),
//! reading the previous frame's reconstruction as the §7.20
//! `FrameStore`. The residual chain is the KEY driver's
//! ([`super::key_frame::residual_tx`] — the decoder's dequant +
//! inverse + `Clip1(pred + residual)` merge), so the encoder's recon
//! tracks the decoder's `CurrFrame` sample-for-sample by induction
//! along the dispatch order, frame over frame.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.9.2 (inter frame
//! header), §5.9.24 (global motion), §5.11.18-§5.11.31 (inter block
//! syntax), §7.10.2 (MV prediction), §7.11.3 (inter prediction),
//! §7.20 (reference frame update).

use crate::cdf::{
    get_tx_size, inter_tx_type_set, tx_size_sqr_index, QuantizerParams, TileCdfContext,
    TileGeometry, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8, DCT_DCT, EIGHTTAP, GM_TYPE_IDENTITY,
    MAX_TX_SIZE_RECT, MAX_VARTX_DEPTH, MODE_GLOBALMV, MODE_NEWMV, MOTION_MODE_SIMPLE,
    NUM_4X4_BLOCKS_WIDE, SPLIT_TX_SIZE, TX_4X4, TX_HEIGHT, TX_SIZE_SQR_UP, TX_WIDTH,
};
use crate::encoder::frame_obu::encode_uncompressed_header;
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::key_frame::{
    encode_key_frame_yuv420_with_q, encode_leaf_sq, lambda_for, leaf_rate, region_distortion,
    residual_tx, restore_region, save_region, tu_bd_stamp, BlockDecodedMirror, ReconState,
    RegionSnapshot, KEY_FRAME_MAX_DIM,
};
use crate::encoder::obu::{build_temporal_unit, ObuFrame};
use crate::encoder::partition_tree::{
    write_partition_tree_syntax, PartitionSyntaxWriter, SyntaxBlock, SyntaxFrameParams,
    SyntaxInterBlock, SyntaxInterFrameParams, SyntaxNode, VarTxSyntaxTree,
};
use crate::encoder::pixel_driver_dyn::{
    build_intra_only_yuv420_8bit_fh_with_q, sb_grid_origins, Yuv420Frame,
};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::tile_group_obu::{write_tile_group_obu, TileGroupObu, TilePayload};
use crate::frame_header::{
    FrameHeader, FrameType, InterFrameRefs, ALL_FRAMES_PUB, PRIMARY_REF_NONE,
};
use crate::inter_pred::{
    reconstruct_inter_leaf_at, InterModeInfoGrid, PlaneReconContext, RefFrameStoreEntry,
};
use crate::obu::ObuType;
use crate::sequence_header::SequenceHeader;
use crate::uncompressed_header_tail::{InterpolationFilter, TxMode, REFS_PER_FRAME};
use crate::Error;

/// One frame's reconstructed planes (row-major; U/V at
/// `(width/2) × (height/2)`).
#[derive(Debug, Clone)]
pub struct GopFrameRecon {
    /// Luma plane.
    pub y: Vec<u8>,
    /// Cb plane.
    pub u: Vec<u8>,
    /// Cr plane.
    pub v: Vec<u8>,
}

/// Result of [`encode_gop_yuv420_with_q`].
#[derive(Debug, Clone)]
pub struct EncodedGop {
    /// Complete IVF v0 file (header + one record per frame).
    pub ivf_bytes: Vec<u8>,
    /// The bare §7.5 temporal units, one per frame (the KEY frame's
    /// carries TD + SH + `OBU_FRAME`; P-frames carry TD +
    /// `OBU_FRAME`).
    pub temporal_units: Vec<Vec<u8>>,
    /// Per-frame encoder reconstruction — the decoded output equals
    /// these byte-for-byte (and the inputs too at `base_q_idx == 0`).
    pub recon: Vec<GopFrameRecon>,
    /// The emitted sequence header descriptor.
    pub seq: SequenceHeader,
}

/// GOP length bound (KEY + P-frames).
pub const GOP_MAX_FRAMES: usize = 64;

/// Integer-pel motion-search radius (luma samples per axis).
const SEARCH_RANGE: i32 = 16;

/// Lossless (`base_q_idx = 0`) GOP encode — see
/// [`encode_gop_yuv420_with_q`].
pub fn encode_gop_yuv420(frames: &[Yuv420Frame]) -> Result<EncodedGop, Error> {
    encode_gop_yuv420_with_q(frames, 0)
}

/// Encode a KEY + P GOP of 8-bit 4:2:0 frames at `base_q_idx` into a
/// spec-conformant IVF stream (see the module docs for the exact
/// scope and the reconstruction-exactness argument).
///
/// ## Errors
///
/// * Empty input, more than [`GOP_MAX_FRAMES`] frames, mismatched
///   dimensions across frames, or dimensions outside the KEY-frame
///   rules — [`Error::PartitionWalkOutOfRange`].
pub fn encode_gop_yuv420_with_q(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
) -> Result<EncodedGop, Error> {
    if frames.is_empty() || frames.len() > GOP_MAX_FRAMES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let (width, height) = (frames[0].width, frames[0].height);
    if frames
        .iter()
        .any(|f| f.width != width || f.height != height)
    {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // Frame 0: the r410 conformance-grade KEY-frame encoder (which
    // also validates the dimension rules).
    let key = encode_key_frame_yuv420_with_q(&frames[0], base_q_idx)?;
    let seq = key.seq.clone();
    let mut temporal_units = vec![key.temporal_unit_bytes.clone()];
    let mut recon = vec![GopFrameRecon {
        y: key.recon_y,
        u: key.recon_u,
        v: key.recon_v,
    }];

    for input in &frames[1..] {
        let (tu, rc) = encode_p_frame_yuv420(
            input,
            recon.last().expect("at least the KEY recon"),
            &seq,
            base_q_idx,
        )?;
        temporal_units.push(tu);
        recon.push(rc);
    }

    // IVF v0 wrap.
    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(cursor, FOURCC_AV01, width as u16, height as u16, 25, 1)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        for (idx, tu) in temporal_units.iter().enumerate() {
            iw.write_frame(tu, idx as u64)
                .map_err(|_| Error::PartitionWalkOutOfRange)?;
        }
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedGop {
        ivf_bytes,
        temporal_units,
        recon,
        seq,
    })
}

/// §5.9.2 INTER P-frame header for the r411 GOP configuration —
/// derived from the KEY builder with the inter-path fields set (see
/// the module docs for the exact configuration).
fn build_p_frame_yuv420_8bit_fh_with_q(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
    base_q_idx: u8,
) -> FrameHeader {
    let mut fh = build_intra_only_yuv420_8bit_fh_with_q(seq, width, height, base_q_idx);
    fh.frame_type = FrameType::Inter;
    fh.frame_is_intra = false;
    // §5.9.2: `showable_frame = frame_type != KEY_FRAME` on the
    // `show_frame == 1` arm (derived, no bit).
    fh.showable_frame = true;
    fh.error_resilient_mode = true;
    fh.force_integer_mv = false;
    fh.primary_ref_frame = PRIMARY_REF_NONE;
    fh.refresh_frame_flags = ALL_FRAMES_PUB;
    // §5.9.21: ONLY_4X4 rides the CodedLossless arm; the lossy arm
    // codes TX_MODE_SELECT — intra leaves carry the §5.11.15
    // `tx_depth` choice and inter leaves the §5.11.17 `txfm_split`
    // recursion.
    fh.tx_mode = Some(if base_q_idx == 0 {
        TxMode::Only4x4
    } else {
        TxMode::TxModeSelect
    });
    fh.reference_select = Some(false);
    fh.skip_mode_present = Some(false);
    fh.allow_warped_motion = Some(false);
    fh.inter_refs = Some(InterFrameRefs {
        frame_refs_short_signaling: false,
        last_frame_idx: None,
        gold_frame_idx: None,
        ref_frame_idx: [0; REFS_PER_FRAME],
        allow_high_precision_mv: false,
        interpolation_filter: InterpolationFilter::Eighttap,
        is_motion_mode_switchable: false,
        use_ref_frame_mvs: false,
    });
    fh
}

/// Encode one INTER P-frame against `reference` (the previous frame's
/// reconstruction). Returns the §7.5 temporal unit (TD + `OBU_FRAME`)
/// and this frame's reconstruction.
fn encode_p_frame_yuv420(
    input: &Yuv420Frame,
    reference: &GopFrameRecon,
    seq: &SequenceHeader,
    base_q_idx: u8,
) -> Result<(Vec<u8>, GopFrameRecon), Error> {
    if input.width < 8
        || input.height < 8
        || input.width > KEY_FRAME_MAX_DIM
        || input.height > KEY_FRAME_MAX_DIM
        || input.width % 8 != 0
        || input.height % 8 != 0
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let width = input.width as usize;
    let height = input.height as usize;
    let chroma_w = width / 2;
    let chroma_h = height / 2;
    if reference.y.len() != width * height
        || reference.u.len() != chroma_w * chroma_h
        || reference.v.len() != chroma_w * chroma_h
    {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let fh = build_p_frame_yuv420_8bit_fh_with_q(seq, input.width, input.height, base_q_idx);
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
    let lossless = base_q_idx == 0;
    let qp = QuantizerParams::neutral(base_q_idx, 8);

    let params = SyntaxFrameParams {
        subsampling_x: 1,
        subsampling_y: 1,
        num_planes: 3,
        seg_id_pre_skip: false,
        segmentation_enabled: false,
        seg_skip_active: false,
        last_active_seg_id: 0,
        lossless_array: [lossless; crate::uncompressed_header_tail::MAX_SEGMENTS],
        coded_lossless: lossless,
        enable_cdef: seq.enable_cdef,
        allow_intrabc: false,
        cdef_bits: 0,
        read_deltas: false,
        use_128x128_superblock: seq.use_128x128_superblock,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_multi: false,
        mono_chrome: false,
        delta_lf_res: 0,
        allow_screen_content_tools: fh.allow_screen_content_tools,
        enable_filter_intra: seq.enable_filter_intra,
        bit_depth: 8,
        tx_mode_select: !lossless,
        quant: qp,
        reduced_tx_set: fh.reduced_tx_set.unwrap_or(false),
        inter: Some(SyntaxInterFrameParams::single_ref_baseline(
            mi_rows, mi_cols, /* force_integer_mv = */ false,
        )),
    };

    let mut recon = ReconState {
        y: vec![0u8; width * height],
        u: vec![0u8; chroma_w * chroma_h],
        v: vec![0u8; chroma_w * chroma_h],
        width,
        height,
        chroma_w,
        chroma_h,
        mi_rows,
        mi_cols,
        lossless,
        qp,
        bd: BlockDecodedMirror::new(),
    };
    let mut ictx = PSearchCtx::new(reference, mi_rows, mi_cols, width, height);

    let mut writer = SymbolWriter::new(fh.disable_cdf_update);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.init_coeff_cdfs(base_q_idx);
    let mut state = PartitionSyntaxWriter::new(
        mi_rows,
        mi_cols,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        },
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;

    for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
        recon.bd.clear_for_sb(sb_r, sb_c, mi_rows, mi_cols);
        let tree = build_p_search_tree(sb_r, sb_c, BLOCK_64X64, input, &mut recon, &mut ictx)?;
        state.arm_read_deltas();
        write_partition_tree_syntax(
            &mut writer,
            &mut cdfs,
            &mut state,
            &tree,
            sb_r,
            sb_c,
            BLOCK_64X64,
            &params,
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

    // §5.10 `frame_obu()`: header + `byte_alignment()` + tile group.
    let frame_body = {
        let mut bw = crate::encoder::bitwriter::BitWriter::new();
        encode_uncompressed_header(&mut bw, &fh, seq);
        bw.byte_align();
        let mut body = bw.finish();
        body.extend_from_slice(&tile_group_body);
        body
    };
    // §7.5 temporal unit: TD + OBU_FRAME (the SH rode the KEY frame's
    // unit; §7.5 requires it once per coded video sequence).
    let temporal_unit = build_temporal_unit(None, &[ObuFrame::new(ObuType::Frame, frame_body)]);

    Ok((
        temporal_unit,
        GopFrameRecon {
            y: recon.y,
            u: recon.u,
            v: recon.v,
        },
    ))
}

// ---------------------------------------------------------------------
// §7.11.3 encoder-side prediction context.
// ---------------------------------------------------------------------

/// Driver-side §5.11.5 grid state + reference pixels: the exact inputs
/// [`reconstruct_inter_leaf_at`] (the decode walker's own §5.11.33
/// leaf driver) consumes, so the encoder's prediction is
/// bit-identical to the decoder's by construction.
///
/// Only the current leaf's own footprint is ever read back on the
/// r411 configuration (no OBMC / warp / compound neighbour reads and
/// no sub-8×8 inter leaves), so trial rollbacks need not restore the
/// grids — the winning candidate re-stamps its footprint before
/// predicting.
struct PSearchCtx {
    mi_rows: u32,
    mi_cols: u32,
    luma_w: u32,
    luma_h: u32,
    /// Previous frame's reconstruction, widened to the §7.11.3.4
    /// sample type.
    ref_planes: [Vec<u16>; 3],
    // §5.11.5 grids (per mi cell).
    mi_sizes: Vec<usize>,
    is_inters: Vec<u8>,
    ref_frames: Vec<i8>,
    mvs: Vec<i16>,
    interp_filters: Vec<u8>,
    motion_modes: Vec<u8>,
    y_modes: Vec<u8>,
    /// Shared all-zero grid for the compound / inter-intra side-data
    /// slices (never read at a single-reference SIMPLE leaf's origin).
    zeros: Vec<u8>,
    /// All-zero §7.11.3.8 per-cell warp-fit slice (never read on the
    /// all-SIMPLE configuration; sized per the [`crate::GridWarpContext`]
    /// contract).
    local_warp: Vec<i32>,
    /// Full-plane prediction scratch (`reconstruct_inter_leaf_at`
    /// writes at absolute plane coordinates).
    scratch: [Vec<u16>; 3],
    gm_types: [u8; 8],
    gm_flat: [i32; 48],
    zero_hints: [i32; 8],
    no_scaled: [bool; 8],
}

impl PSearchCtx {
    fn new(
        reference: &GopFrameRecon,
        mi_rows: u32,
        mi_cols: u32,
        width: usize,
        height: usize,
    ) -> Self {
        let cells = (mi_rows as usize) * (mi_cols as usize);
        let widen = |p: &[u8]| p.iter().map(|&v| u16::from(v)).collect::<Vec<u16>>();
        let mut gm_flat = [0i32; 48];
        for r in 0..8 {
            gm_flat[r * 6 + 2] = 1 << crate::cdf::WARPEDMODEL_PREC_BITS;
            gm_flat[r * 6 + 5] = 1 << crate::cdf::WARPEDMODEL_PREC_BITS;
        }
        let mut ref_frames = vec![0i8; cells * 2];
        for cell in 0..cells {
            ref_frames[cell * 2] = 0; // INTRA_FRAME pre-fill
            ref_frames[cell * 2 + 1] = -1; // NONE
        }
        PSearchCtx {
            mi_rows,
            mi_cols,
            luma_w: width as u32,
            luma_h: height as u32,
            ref_planes: [
                widen(&reference.y),
                widen(&reference.u),
                widen(&reference.v),
            ],
            mi_sizes: vec![BLOCK_4X4; cells],
            is_inters: vec![0u8; cells],
            ref_frames,
            mvs: vec![0i16; cells * 4],
            interp_filters: vec![EIGHTTAP; cells * 2],
            motion_modes: vec![MOTION_MODE_SIMPLE; cells],
            y_modes: vec![0u8; cells],
            zeros: vec![0u8; cells],
            local_warp: vec![0i32; cells * 6],
            scratch: [
                vec![0u16; width * height],
                vec![0u16; (width / 2) * (height / 2)],
                vec![0u16; (width / 2) * (height / 2)],
            ],
            gm_types: [GM_TYPE_IDENTITY as u8; 8],
            gm_flat,
            zero_hints: [0i32; 8],
            no_scaled: [false; 8],
        }
    }

    /// Stamp the leaf's §5.11.5 grid footprint (the same values the
    /// write mirror / decode walker stamp), then run the decoder's
    /// §5.11.33 leaf driver into the scratch planes.
    fn predict_leaf(
        &mut self,
        mi_row: u32,
        mi_col: u32,
        b_size: usize,
        y_mode: u8,
        mv: [i32; 2],
    ) -> Result<(), Error> {
        let n4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
        for dr in 0..n4 {
            let rr = mi_row + dr;
            if rr >= self.mi_rows {
                break;
            }
            for dc in 0..n4 {
                let cc = mi_col + dc;
                if cc >= self.mi_cols {
                    break;
                }
                let cell = (rr * self.mi_cols + cc) as usize;
                self.mi_sizes[cell] = b_size;
                self.is_inters[cell] = 1;
                self.ref_frames[cell * 2] = 1; // LAST_FRAME
                self.ref_frames[cell * 2 + 1] = -1;
                self.mvs[cell * 4] = mv[0] as i16;
                self.mvs[cell * 4 + 1] = mv[1] as i16;
                self.mvs[cell * 4 + 2] = 0;
                self.mvs[cell * 4 + 3] = 0;
                self.interp_filters[cell * 2] = EIGHTTAP;
                self.interp_filters[cell * 2 + 1] = EIGHTTAP;
                self.motion_modes[cell] = MOTION_MODE_SIMPLE;
                self.y_modes[cell] = y_mode;
            }
        }

        // Field-split borrows: the grid views borrow the mode-info
        // vectors immutably while the plane contexts borrow the
        // scratch planes mutably.
        let (luma_w, luma_h) = (self.luma_w, self.luma_h);
        let (mi_rows, mi_cols) = (self.mi_rows, self.mi_cols);
        let PSearchCtx {
            ref_planes,
            mi_sizes,
            is_inters,
            ref_frames,
            mvs,
            interp_filters,
            motion_modes,
            y_modes,
            zeros,
            local_warp,
            scratch,
            gm_types,
            gm_flat,
            zero_hints,
            no_scaled,
            ..
        } = &mut *self;

        // §7.20 `FrameStore` views — every slot holds the previous
        // frame (refresh_frame_flags = allFrames), and `ref_frame_idx`
        // resolves each reference to slot 0. Dimensions are LUMA
        // extents per the r405 contract; strides are plane samples.
        let store_y = make_store(&ref_planes[0], luma_w as usize, luma_w, luma_h);
        let store_u = make_store(&ref_planes[1], (luma_w as usize) / 2, luma_w, luma_h);
        let store_v = make_store(&ref_planes[2], (luma_w as usize) / 2, luma_w, luma_h);

        let grid = InterModeInfoGrid {
            mi_sizes,
            is_inters,
            ref_frames,
            mvs,
            interp_filters,
            compound_types: zeros,
            wedge_indices: zeros,
            wedge_signs: zeros,
            mask_types: zeros,
            interintra_modes: zeros,
            wedge_interintras: zeros,
            interintra_wedge_indices: zeros,
            order_hint_bits: 0,
            current_order_hint: 0,
            order_hints_by_ref: zero_hints,
            mi_rows,
            mi_cols,
            bit_depth: 8,
            // §7.11.3.1 step-7 warp context — identity global motion +
            // SIMPLE motion modes keep `useWarp = 0` on every leaf
            // (the decode walker threads the same tables).
            warp: Some(crate::GridWarpContext {
                motion_modes,
                local_warp_params: local_warp,
                local_warp_valid: zeros,
                y_modes,
                gm_types,
                gm_params: gm_flat,
                force_integer_mv: false,
                is_scaled: no_scaled,
            }),
            obmc: None,
        };
        let [s0, s1, s2] = scratch;
        let mut planes = [
            PlaneReconContext {
                plane: 0,
                subsampling_x: 0,
                subsampling_y: 0,
                frame_store: &store_y,
                frame_width: luma_w,
                frame_height: luma_h,
                curr: s0,
                curr_stride: luma_w as usize,
            },
            PlaneReconContext {
                plane: 1,
                subsampling_x: 1,
                subsampling_y: 1,
                frame_store: &store_u,
                frame_width: luma_w,
                frame_height: luma_h,
                curr: s1,
                curr_stride: (luma_w as usize) / 2,
            },
            PlaneReconContext {
                plane: 2,
                subsampling_x: 1,
                subsampling_y: 1,
                frame_store: &store_v,
                frame_width: luma_w,
                frame_height: luma_h,
                curr: s2,
                curr_stride: (luma_w as usize) / 2,
            },
        ];
        reconstruct_inter_leaf_at(
            &grid,
            &[0u8; 7],
            &mut planes,
            mi_row as usize,
            mi_col as usize,
        )
    }
}

/// One plane's §7.20 `FrameStore` view (all eight slots at the same
/// previous-frame plane; extents in LUMA samples per the r405
/// contract, stride in this plane's own samples).
fn make_store(
    plane: &[u16],
    stride: usize,
    luma_w: u32,
    luma_h: u32,
) -> [RefFrameStoreEntry<'_>; 8] {
    core::array::from_fn(|_| RefFrameStoreEntry {
        plane,
        stride,
        upscaled_width: luma_w,
        width: luma_w,
        height: luma_h,
    })
}

// ---------------------------------------------------------------------
// Motion search + inter leaf encoder.
// ---------------------------------------------------------------------

/// Integer-pel luma motion search: coarse step-4 grid over
/// `±SEARCH_RANGE` then a full `±3` refinement, SSD against the
/// edge-clamped reference (exactly the §7.11.3.4 phase-0 sample
/// fetch), with a small magnitude bias so near-zero vectors win ties.
/// Returns the best vector in 1/8-luma units (a multiple of 8).
fn motion_search_luma(
    input: &Yuv420Frame,
    ref_y: &[u16],
    width: usize,
    height: usize,
    row0: usize,
    col0: usize,
    n: usize,
) -> [i32; 2] {
    let cost_at = |dy: i32, dx: i32| -> u64 {
        let mut ssd = 0u64;
        for i in 0..n {
            let sy = ((row0 + i) as i32 + dy).clamp(0, height as i32 - 1) as usize;
            for j in 0..n {
                let sx = ((col0 + j) as i32 + dx).clamp(0, width as i32 - 1) as usize;
                let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                    - i64::from(ref_y[sy * width + sx]);
                ssd += (d * d) as u64;
            }
        }
        ssd + (dy.unsigned_abs() as u64 + dx.unsigned_abs() as u64) * (n as u64)
    };
    let mut best = (cost_at(0, 0), [0i32, 0]);
    let mut dy = -SEARCH_RANGE;
    while dy <= SEARCH_RANGE {
        let mut dx = -SEARCH_RANGE;
        while dx <= SEARCH_RANGE {
            if !(dy == 0 && dx == 0) {
                let c = cost_at(dy, dx);
                if c < best.0 {
                    best = (c, [dy, dx]);
                }
            }
            dx += 4;
        }
        dy += 4;
    }
    let center = best.1;
    for dy in center[0] - 3..=center[0] + 3 {
        for dx in center[1] - 3..=center[1] + 3 {
            if dy == center[0] && dx == center[1] {
                continue;
            }
            let c = cost_at(dy, dx);
            if c < best.0 {
                best = (c, [dy, dx]);
            }
        }
    }
    [best.1[0] * 8, best.1[1] * 8]
}

/// Sub-pel MV refinement through the decoder's §7.11.3 leaf driver:
/// a half-pel pass (±4 in 1/8 units) then a quarter-pel pass (±2)
/// around the running best, scoring each candidate by luma SSD over
/// the kernel's actual prediction plus a small magnitude bias.
#[allow(clippy::too_many_arguments)]
fn refine_mv_subpel(
    input: &Yuv420Frame,
    ictx: &mut PSearchCtx,
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    row0: usize,
    col0: usize,
    n: usize,
    width: usize,
    mv_int: [i32; 2],
) -> Result<[i32; 2], Error> {
    let score = |ictx: &mut PSearchCtx, mv: [i32; 2]| -> Result<u64, Error> {
        ictx.predict_leaf(mi_r, mi_c, b_size, MODE_NEWMV, mv)?;
        let mut ssd = 0u64;
        for i in 0..n {
            for j in 0..n {
                let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                    - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                ssd += (d * d) as u64;
            }
        }
        Ok(ssd + ((mv[0].unsigned_abs() + mv[1].unsigned_abs()) as u64) * (n as u64) / 8)
    };
    let mut best = (score(ictx, mv_int)?, mv_int);
    for step in [4i32, 2] {
        let center = best.1;
        for dy in [-step, 0, step] {
            for dx in [-step, 0, step] {
                if dy == 0 && dx == 0 {
                    continue;
                }
                let cand = [center[0] + dy, center[1] + dx];
                let c = score(ictx, cand)?;
                if c < best.0 {
                    best = (c, cand);
                }
            }
        }
    }
    Ok(best.1)
}

/// Encode one in-frame square INTER leaf at `b_size` (`BLOCK_8X8` …
/// `BLOCK_64X64`): motion search, §7.11.3 prediction through the
/// decoder's leaf driver, residual per TU (one
/// `Max_Tx_Size_Rect` luma TU on the lossy arm / the `TX_4X4` grid on
/// the lossless arm; chroma at the §5.11.38 size), `skip = 1` when
/// every TU quantises to zero.
fn encode_inter_leaf(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &Yuv420Frame,
    recon: &mut ReconState,
    ictx: &mut PSearchCtx,
) -> Result<SyntaxBlock, Error> {
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let n = n4 * 4;
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let width = recon.width;
    let lossless = recon.lossless;

    let mv_int = motion_search_luma(
        input,
        &ictx.ref_planes[0],
        width,
        recon.height,
        row0,
        col0,
        n,
    );
    // r411 sub-pel refinement: half-pel then quarter-pel deltas around
    // the integer winner, each candidate evaluated through the REAL
    // §7.11.3.4 kernel (so the search cost IS the coding cost).
    // `allow_high_precision_mv = 0` restricts components to quarter-pel
    // (multiples of 2 in 1/8-luma units).
    let mv = refine_mv_subpel(
        input, ictx, mi_r, mi_c, b_size, row0, col0, n, width, mv_int,
    )?;
    // Zero vector ⇒ GLOBALMV (the identity-warp §7.10.2.1 derivation
    // is exactly [0, 0], so no MV bits and no PredMv dependence).
    let y_mode = if mv == [0, 0] {
        MODE_GLOBALMV
    } else {
        MODE_NEWMV
    };
    ictx.predict_leaf(mi_r, mi_c, b_size, y_mode, mv)?;

    if lossless {
        // §5.9.2 CodedLossless: TX_4X4 everywhere, no §5.11.17 trees.
        return encode_inter_leaf_residual(mi_r, mi_c, b_size, input, recon, ictx, y_mode, mv, 0);
    }

    // §5.11.17 uniform-depth RD trial (TX_MODE_SELECT): code the leaf
    // at `Split_Tx_Size^depth[ Max_Tx_Size_Rect ]` for each reachable
    // depth against the same starting state, keep the lower `D + λ·R`.
    let max_tx = MAX_TX_SIZE_RECT[b_size];
    let mut max_depth = 0u32;
    let mut t = max_tx;
    while max_depth < MAX_VARTX_DEPTH && t != TX_4X4 {
        t = SPLIT_TX_SIZE[t];
        max_depth += 1;
    }
    let lambda = lambda_for(&recon.qp);
    let before = save_region(recon, mi_r, mi_c, n4);
    let mut best: Option<(SyntaxBlock, RegionSnapshot, u64)> = None;
    for depth in 0..=max_depth {
        let leaf =
            encode_inter_leaf_residual(mi_r, mi_c, b_size, input, recon, ictx, y_mode, mv, depth)?;
        let d = region_distortion(recon, input, mi_r, mi_c, n4);
        let score = d + lambda * (p_leaf_rate(&leaf) + 2 * u64::from(depth));
        let improves = match best.as_ref() {
            Some((_, _, s)) => score < *s,
            None => true,
        };
        if improves {
            best = Some((leaf, save_region(recon, mi_r, mi_c, n4), score));
        }
        restore_region(recon, mi_r, mi_c, &before);
    }
    let (leaf, after, _) = best.expect("at least depth 0");
    restore_region(recon, mi_r, mi_c, &after);
    Ok(leaf)
}

/// §5.11.36 luma TU visit order for a uniform `tw × tw` TU grid over
/// an `n × n` block — the quadtree halving recursion (NW / NE / SW /
/// SE at every level), NOT row-major beyond a 2×2 grid.
fn transform_tree_tu_order(
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    tw: usize,
    out: &mut Vec<(usize, usize)>,
) {
    if w <= tw && h <= tw {
        out.push((x, y));
        return;
    }
    transform_tree_tu_order(x, y, w / 2, h / 2, tw, out);
    transform_tree_tu_order(x + w / 2, y, w / 2, h / 2, tw, out);
    transform_tree_tu_order(x, y + h / 2, w / 2, h / 2, tw, out);
    transform_tree_tu_order(x + w / 2, y + h / 2, w / 2, h / 2, tw, out);
}

/// r411 — §5.11.47 per-TU LUMA transform-type RD search on the INTER
/// arm: trials every `TxType` admissible in the §5.11.48 inter set for
/// `tx_sz` (all 16 at 4×4/8×8, 12 at 16×16, IDTX+DCT at 32×32, DCT
/// alone above), scoring each full quantise→dequantise→inverse chain
/// by `D + λ·R` over the TU, then stitches the winner into the running
/// plane. The returned label is forced to `DCT_DCT` when the winning
/// TU quantises to all-zero (the §5.11.39 `all_zero` arm reads no
/// `inter_tx_type` symbol and the walker stamps `DCT_DCT`).
#[allow(clippy::too_many_arguments)]
fn residual_tx_search_luma_inter(
    input_plane: &[u8],
    recon_plane: &mut [u8],
    pw: usize,
    row0: usize,
    col0: usize,
    tx_sz: usize,
    pred: &[u8],
    qp: &QuantizerParams,
) -> (Vec<i32>, u8) {
    use crate::cdf::{dequantize_step1, is_tx_type_in_set, tx_size_sqr_index, TX_SIZE_SQR_UP};
    use crate::encoder::forward_quantize::forward_quantize;
    use crate::encoder::forward_transform_2d::forward_transform_2d;
    use crate::encoder::key_frame::repack_compact;
    use crate::transform::inverse_transform_2d;

    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    let mut residual = vec![0i64; w * h];
    for i in 0..h {
        for j in 0..w {
            residual[i * w + j] =
                i64::from(input_plane[(row0 + i) * pw + (col0 + j)]) - i64::from(pred[i * w + j]);
        }
    }
    let set = inter_tx_type_set(
        tx_size_sqr_index(tx_sz) as u32,
        TX_SIZE_SQR_UP[tx_sz] as u32,
        false,
    );
    let lambda = lambda_for(qp);
    let mut best: Option<(Vec<i32>, Vec<i64>, u8, u64)> = None;
    for t in 0..crate::cdf::TX_TYPES {
        let admissible = t == DCT_DCT || (set > 0 && is_tx_type_in_set(true, set, t));
        if !admissible {
            continue;
        }
        let coeffs = forward_transform_2d(&residual, tx_sz, t, false);
        let quant = repack_compact(forward_quantize(&coeffs, tx_sz, 0, 0, t, 15, qp), w, h);
        let all_zero = quant.iter().all(|&q| q == 0);
        let dequant = dequantize_step1(&quant, tx_sz, 0, 0, t, 15, qp);
        let res_back = inverse_transform_2d(&dequant, tx_sz, t, 8, false);
        // §7.12.3 step-3 destination remap (FLIPADST family).
        let (flip_ud, flip_lr) = crate::encoder::key_frame::step3_flips(t);
        let mut d = 0u64;
        for i in 0..h {
            let yy = if flip_ud { h - 1 - i } else { i };
            for j in 0..w {
                let xx = if flip_lr { w - 1 - j } else { j };
                let rec = (i64::from(pred[yy * w + xx]) + res_back[i * w + j]).clamp(0, 255);
                let diff = i64::from(input_plane[(row0 + yy) * pw + (col0 + xx)]) - rec;
                d += (diff * diff) as u64;
            }
        }
        let mut rate = 0u64;
        for &qv in &quant {
            if qv != 0 {
                rate += 3 + u64::from(32 - qv.unsigned_abs().leading_zeros());
            }
        }
        let score = d + lambda * rate;
        let label = if all_zero { DCT_DCT as u8 } else { t as u8 };
        let improves = match best.as_ref() {
            Some((_, _, _, s)) => score < *s,
            None => true,
        };
        if improves {
            best = Some((quant, res_back, label, score));
        }
        // A below-floor residual at DCT_DCT is pred-exact — skip the
        // remaining trials (they commit the same all-zero shape).
        if t == DCT_DCT && all_zero {
            break;
        }
    }
    let (quant, res_back, label, _) = best.expect("DCT_DCT is always admissible");
    // NOTE: the winning trial's `label` may be DCT_DCT (all-zero) even
    // though `res_back` came from a FLIPADST trial — but an all-zero
    // TU's residual is identically zero, so the remap is inert there;
    // for coded TUs `label` IS the trial type.
    let (flip_ud, flip_lr) = crate::encoder::key_frame::step3_flips(label as usize);
    for i in 0..h {
        let yy = if flip_ud { h - 1 - i } else { i };
        for j in 0..w {
            let xx = if flip_lr { w - 1 - j } else { j };
            let p = i64::from(pred[yy * w + xx]) + res_back[i * w + j];
            recon_plane[(row0 + yy) * pw + (col0 + xx)] = p.clamp(0, 255) as u8;
        }
    }
    (quant, label)
}

/// §5.11.17 uniform split-decision tree of the given depth (square
/// transform ordinals ⇒ four children per split).
fn uniform_var_tx_tree(depth: u32) -> VarTxSyntaxTree {
    if depth == 0 {
        VarTxSyntaxTree::Leaf
    } else {
        VarTxSyntaxTree::Split(vec![
            uniform_var_tx_tree(depth - 1),
            uniform_var_tx_tree(depth - 1),
            uniform_var_tx_tree(depth - 1),
            uniform_var_tx_tree(depth - 1),
        ])
    }
}

/// Residual-code one inter leaf at uniform §5.11.17 depth `depth`
/// (`0` = one `Max_Tx_Size_Rect` TU; the lossless arm ignores `depth`
/// and codes the `TX_4X4` grid). Consumes the prediction already in
/// `ictx.scratch`.
#[allow(clippy::too_many_arguments)]
fn encode_inter_leaf_residual(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &Yuv420Frame,
    recon: &mut ReconState,
    ictx: &PSearchCtx,
    y_mode: u8,
    mv: [i32; 2],
    depth: u32,
) -> Result<SyntaxBlock, Error> {
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let n = n4 * 4;
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let width = recon.width;
    let lossless = recon.lossless;
    let qp = recon.qp;

    // --- Luma residual over the §5.11.36 TU walk. ---
    let luma_tx = if lossless {
        TX_4X4
    } else {
        let mut t = MAX_TX_SIZE_RECT[b_size];
        for _ in 0..depth {
            t = SPLIT_TX_SIZE[t];
        }
        t
    };
    let (ltw, lth) = (TX_WIDTH[luma_tx], TX_HEIGHT[luma_tx]);
    let mut tu_order: Vec<(usize, usize)> = Vec::new();
    if lossless {
        // §5.11.34 direct row-major iteration (the `!is_inter ||
        // Lossless` arm).
        let mut ty = 0usize;
        while ty < n {
            let mut tx = 0usize;
            while tx < n {
                tu_order.push((tx, ty));
                tx += ltw;
            }
            ty += lth;
        }
    } else {
        // §5.11.36 transform-tree quadtree order.
        transform_tree_tu_order(0, 0, n, n, ltw, &mut tu_order);
    }
    let mut residual_quant: Vec<Vec<i32>> = Vec::new();
    let mut luma_tx_types: Vec<u8> = Vec::new();
    // Block-relative luma-TU-grid map of committed §5.11.47 types —
    // the §5.11.40 chroma-inheritance source (`TxTypes[ y4 ][ x4 ]`).
    let tu_cols = n / ltw;
    let mut tu_type_grid = vec![DCT_DCT as u8; tu_cols * (n / lth)];
    for &(tx, ty) in &tu_order {
        let (tr, tc) = (row0 + ty, col0 + tx);
        let mut pred = vec![0u8; ltw * lth];
        for i in 0..lth {
            for j in 0..ltw {
                pred[i * ltw + j] = ictx.scratch[0][(tr + i) * width + (tc + j)] as u8;
            }
        }
        let (q, tt) = if lossless {
            (
                residual_tx(
                    &input.y,
                    &mut recon.y,
                    width,
                    tr,
                    tc,
                    luma_tx,
                    &pred,
                    0,
                    lossless,
                    DCT_DCT,
                    &qp,
                ),
                DCT_DCT as u8,
            )
        } else {
            // r411: §5.11.47 per-TU luma transform-type RD search over
            // the §5.11.48 INTER set for this TX size.
            residual_tx_search_luma_inter(
                &input.y,
                &mut recon.y,
                width,
                tr,
                tc,
                luma_tx,
                &pred,
                &qp,
            )
        };
        tu_bd_stamp(&mut recon.bd, 0, tc, tr, ltw, lth);
        residual_quant.push(q);
        luma_tx_types.push(tt);
        tu_type_grid[(ty / lth) * tu_cols + tx / ltw] = tt;
    }

    // --- Chroma (every BLOCK_8X8+ leaf has chroma at 4:2:0). ---
    // §5.11.34: the chroma TU size derives from the block's committed
    // `TxSize` — on the §5.11.16 var-tx arm that is the recursion's
    // last terminal-else `txSz` (the uniform TU size here).
    let (crow0, ccol0) = ((mi_r as usize >> 1) * 4, (mi_c as usize >> 1) * 4);
    let cn = n / 2;
    let chroma_tx = if lossless {
        TX_4X4
    } else {
        get_tx_size(1, luma_tx, b_size, 1, 1).unwrap_or(TX_4X4)
    };
    let (ctw, cth) = (TX_WIDTH[chroma_tx], TX_HEIGHT[chroma_tx]);
    // §5.11.48 inter set at the CHROMA TX size — the §5.11.40
    // inheritance filter.
    let chroma_set = inter_tx_type_set(
        tx_size_sqr_index(chroma_tx) as u32,
        TX_SIZE_SQR_UP[chroma_tx] as u32,
        false,
    );
    let cw = recon.chroma_w;
    for plane in 1..=2usize {
        let mut ty = 0usize;
        while ty < cn {
            let mut tx = 0usize;
            while tx < cn {
                let (tr, tc) = (crow0 + ty, ccol0 + tx);
                // §5.11.40 inter-chroma `TxType`: the luma cell at the
                // subsampling-lifted position (block-internal here —
                // the `Max( MiCol, .. )` clip only binds at the grid
                // origin), filtered by the chroma-size inter set.
                let chroma_tt = if lossless || TX_SIZE_SQR_UP[chroma_tx] > crate::cdf::TX_32X32 {
                    DCT_DCT
                } else {
                    let luma_tt =
                        tu_type_grid[((ty * 2) / lth) * tu_cols + (tx * 2) / ltw] as usize;
                    if crate::cdf::is_tx_type_in_set(true, chroma_set, luma_tt) {
                        luma_tt
                    } else {
                        DCT_DCT
                    }
                };
                let mut pred = vec![0u8; ctw * cth];
                for i in 0..cth {
                    for j in 0..ctw {
                        pred[i * ctw + j] = ictx.scratch[plane][(tr + i) * cw + (tc + j)] as u8;
                    }
                }
                let plane_buf = if plane == 1 {
                    &mut recon.u
                } else {
                    &mut recon.v
                };
                let q = residual_tx(
                    if plane == 1 { &input.u } else { &input.v },
                    plane_buf,
                    cw,
                    tr,
                    tc,
                    chroma_tx,
                    &pred,
                    plane as u8,
                    lossless,
                    chroma_tt,
                    &qp,
                );
                tu_bd_stamp(&mut recon.bd, plane, tc, tr, ctw, cth);
                residual_quant.push(q);
                tx += ctw;
            }
            ty += cth;
        }
    }

    // §5.11.11 skip: 1 iff every TU quantised to zero (the recon then
    // equals the bare prediction on every plane). A skip leaf takes
    // the §5.11.16 else arm — no §5.11.17 trees.
    let all_zero = residual_quant.iter().all(|tu| tu.iter().all(|&q| q == 0));
    let skip = u8::from(all_zero);
    if all_zero {
        residual_quant.clear();
        luma_tx_types.clear();
    }
    // An all-DCT_DCT vector is the writer's back-compat default —
    // commit the compact empty form.
    if luma_tx_types.iter().all(|&t| t == DCT_DCT as u8) {
        luma_tx_types.clear();
    }

    let mut block = SyntaxBlock::skip_leaf(0, None);
    block.skip = skip;
    block.residual_quant = residual_quant;
    block.residual_tx_type = luma_tx_types;
    if !lossless && skip == 0 {
        // §5.11.16 var-tx arm: one uniform tree per max-TU position
        // (square blocks ⇒ exactly one).
        block.var_tx_trees = vec![uniform_var_tx_tree(depth)];
    }
    block.inter = Some(SyntaxInterBlock {
        ref_frame: [1, -1],
        y_mode,
        mv: [mv, [0, 0]],
        ref_mv_idx: 0,
    });
    Ok(block)
}

/// Crude per-leaf rate proxy for the RD decisions — [`leaf_rate`]'s
/// coefficient model plus an MV-magnitude term for NEWMV leaves and a
/// small intra surcharge (intra blocks in inter frames also code
/// `is_inter = 0` against a heavily-inter-biased context).
fn p_leaf_rate(block: &SyntaxBlock) -> u64 {
    let mut rate = leaf_rate(block);
    match &block.inter {
        Some(ib) => {
            if ib.y_mode == MODE_NEWMV {
                let bits = |v: i32| u64::from(34 - v.unsigned_abs().leading_zeros());
                rate += 6 + bits(ib.mv[0][0]) + bits(ib.mv[0][1]);
            }
        }
        None => rate += 16,
    }
    rate
}

/// Recursive rate proxy over a candidate subtree.
fn p_tree_rate(node: &SyntaxNode) -> u64 {
    match node {
        SyntaxNode::Leaf(b) => p_leaf_rate(b),
        SyntaxNode::Split(children) => 4 + children.iter().map(|c| p_tree_rate(c)).sum::<u64>(),
    }
}

/// Recursive §5.11.4 rate-distortion partition search for a P-frame:
/// at every fully-in-frame square node from `BLOCK_64X64` down to
/// `BLOCK_8X8`, trial-encode an INTER leaf and an INTRA leaf against
/// the same starting state, and (above `BLOCK_8X8`) both against a
/// `PARTITION_SPLIT` of four recursively-searched quadrants, keeping
/// the lowest `D + λ·R`. `BLOCK_8X8` is the P-frame leaf floor (see
/// the module docs). Frame-edge nodes take the §5.11.4 forced split.
fn build_p_search_tree(
    r: u32,
    c: u32,
    b_size: usize,
    input: &Yuv420Frame,
    recon: &mut ReconState,
    ictx: &mut PSearchCtx,
) -> Result<SyntaxNode, Error> {
    if r >= recon.mi_rows || c >= recon.mi_cols {
        return Ok(SyntaxNode::dummy_oob());
    }
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
    let half = n4 >> 1;
    let sub = crate::cdf::partition_subsize(crate::cdf::PARTITION_SPLIT, b_size)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let fully_inside = r + n4 <= recon.mi_rows && c + n4 <= recon.mi_cols;

    if !fully_inside {
        // §5.11.4 forced arms (dims are multiples of 8, so BLOCK_8X8
        // nodes are always fully inside and the recursion terminates).
        let children = [
            Box::new(build_p_search_tree(r, c, sub, input, recon, ictx)?),
            Box::new(build_p_search_tree(r, c + half, sub, input, recon, ictx)?),
            Box::new(build_p_search_tree(r + half, c, sub, input, recon, ictx)?),
            Box::new(build_p_search_tree(
                r + half,
                c + half,
                sub,
                input,
                recon,
                ictx,
            )?),
        ];
        return Ok(SyntaxNode::Split(children));
    }

    let lambda = lambda_for(&recon.qp);
    let before = save_region(recon, r, c, n4 as usize);

    // Candidate A: one INTER leaf.
    let inter_leaf = encode_inter_leaf(r, c, b_size, input, recon, ictx)?;
    let d_inter = region_distortion(recon, input, r, c, n4 as usize);
    let r_inter = p_leaf_rate(&inter_leaf);
    let score_inter = d_inter + lambda * r_inter;
    let after_inter = save_region(recon, r, c, n4 as usize);
    restore_region(recon, r, c, &before);

    // Candidate B: one INTRA leaf (§5.11.22 arm) with the KEY
    // driver's §5.11.15 tx_depth RD search (TX_MODE_SELECT on the
    // lossy arm; the lossless arm stays on the TX_4X4 grid).
    let intra_leaf = encode_leaf_sq(r, c, b_size, input, recon);
    let d_intra = region_distortion(recon, input, r, c, n4 as usize);
    let r_intra = p_leaf_rate(&intra_leaf);
    let score_intra = d_intra + lambda * r_intra;
    let after_intra = save_region(recon, r, c, n4 as usize);
    restore_region(recon, r, c, &before);

    let (leaf, after_leaf, score_leaf) = if score_inter <= score_intra {
        (inter_leaf, after_inter, score_inter)
    } else {
        (intra_leaf, after_intra, score_intra)
    };

    if b_size <= BLOCK_8X8 {
        // P-frame leaf floor — no split candidate.
        restore_region(recon, r, c, &after_leaf);
        return Ok(SyntaxNode::Leaf(Box::new(leaf)));
    }

    // Candidate C: PARTITION_SPLIT into four recursively-searched
    // quadrants (NW/NE/SW/SE dispatch order).
    let children = [
        Box::new(build_p_search_tree(r, c, sub, input, recon, ictx)?),
        Box::new(build_p_search_tree(r, c + half, sub, input, recon, ictx)?),
        Box::new(build_p_search_tree(r + half, c, sub, input, recon, ictx)?),
        Box::new(build_p_search_tree(
            r + half,
            c + half,
            sub,
            input,
            recon,
            ictx,
        )?),
    ];
    let d_split = region_distortion(recon, input, r, c, n4 as usize);
    let r_split = children.iter().map(|ch| p_tree_rate(ch)).sum::<u64>() + 4;
    let score_split = d_split + lambda * r_split;

    if score_leaf <= score_split {
        restore_region(recon, r, c, &after_leaf);
        Ok(SyntaxNode::Leaf(Box::new(leaf)))
    } else {
        Ok(SyntaxNode::Split(children))
    }
}
