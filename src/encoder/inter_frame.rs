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
//! ## Scope (r411, extended r412)
//!
//! * 8-bit 4:2:0 YUV input, dimensions per the KEY-frame rules
//!   (multiples of 8 in `[8, KEY_FRAME_MAX_DIM]`).
//! * P-frame header: `error_resilient_mode = 1` (forcing
//!   `primary_ref_frame = PRIMARY_REF_NONE` — per-frame default CDFs),
//!   the r412 two-slot reference rotation (frame `k` refreshes slot
//!   `(k - 1) & 1`; LAST reads the previous frame's slot, GOLDEN the
//!   frame before it), identity §5.9.24 global motion, SWITCHABLE
//!   frame filter (per-leaf §5.11.x `interp_filter` selection),
//!   `force_integer_mv = 0` / `allow_high_precision_mv = 0`, no order
//!   hints, no skip-mode, no segmentation; `TxMode = ONLY_4X4` on the
//!   `CodedLossless` arm (`base_q_idx == 0`) or `TX_MODE_SELECT`
//!   otherwise.
//! * Per-node RD search (leaf vs HORZ vs VERT vs split, `BLOCK_64X64`
//!   down to `BLOCK_8X8` — inter frames stop above the sub-8×8 chroma
//!   `someUseIntra` stitching): every square node trials one INTER
//!   leaf (per-reference LAST/GOLDEN integer motion search +
//!   half/quarter-pel refinement through the real §7.11.3.4 kernel,
//!   §5.11.24 mode selection over NEWMV / NEARESTMV / NEARMV /
//!   GLOBALMV via the r412 driver-side §7.10.2 mirror) against one
//!   INTRA leaf (the §5.11.22 arm with the KEY driver's 13-mode +
//!   §5.11.15 tx_depth pickers), the §5.11.4 PARTITION_HORZ /
//!   PARTITION_VERT pairs of rectangular inter halves, and the
//!   recursive split. Inter leaves additionally RD-select their
//!   §5.11.17 uniform `txfm_split` depth (`Max_Tx_Size_Rect` down to
//!   two `Split_Tx_Size` steps), coding the recursion's TUs in
//!   §5.11.36 transform-tree order.
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
    get_tx_size, inter_tx_type_set, tx_size_sqr_index, FindMvStackResult, PartitionWalker,
    QuantizerParams, TileCdfContext, TileGeometry, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8, DCT_DCT,
    EIGHTTAP, GM_TYPE_IDENTITY, MAX_TX_SIZE_RECT, MAX_VARTX_DEPTH, MODE_GLOBALMV,
    MODE_GLOBAL_GLOBALMV, MODE_NEARESTMV, MODE_NEAREST_NEARESTMV, MODE_NEARMV, MODE_NEAR_NEARMV,
    MODE_NEWMV, MODE_NEW_NEWMV, MOTION_MODE_SIMPLE, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE,
    PARTITION_HORZ, PARTITION_VERT, SPLIT_TX_SIZE, SWITCHABLE, TX_4X4, TX_HEIGHT, TX_SIZE_SQR_UP,
    TX_WIDTH,
};
use crate::encoder::block_mode_info::assign_mv_pred_mv;
use crate::encoder::frame_obu::encode_uncompressed_header;
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::key_frame::{
    encode_key_frame_yuv420_with_q, encode_leaf_sq, lambda_for, leaf_rate, region_distortion,
    region_distortion_wh, residual_tx, restore_region, save_region, save_region_wh, tu_bd_stamp,
    BlockDecodedMirror, ReconState, RegionSnapshot, KEY_FRAME_MAX_DIM,
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
use crate::frame_header::{FrameHeader, FrameType, InterFrameRefs, PRIMARY_REF_NONE};
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

    for (k, input) in frames[1..].iter().enumerate() {
        let p_index = (k + 1) as u32;
        let prev = recon.last().expect("at least the KEY recon");
        let prevprev = &recon[recon.len().saturating_sub(2)];
        let (tu, rc) = encode_p_frame_yuv420(input, prev, prevprev, &seq, base_q_idx, p_index)?;
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

/// §5.9.2 INTER P-frame header for the r412 GOP configuration —
/// derived from the KEY builder with the inter-path fields set (see
/// the module docs for the exact configuration).
///
/// r412 two-slot reference rotation (`p_index` is the 1-based
/// P-frame index): frame `k` refreshes slot `(k - 1) & 1`, reads
/// LAST (and every other reference except GOLDEN) from slot `k & 1`
/// — the slot holding frame `k - 1` — and GOLDEN from slot
/// `(k - 1) & 1`, which still holds frame `k - 2` when the header is
/// read (§7.20 updates the store AFTER decoding; for `k <= 2` both
/// slots reach back to the KEY frame, which every slot holds after
/// the KEY's `allFrames` refresh).
fn build_p_frame_yuv420_8bit_fh_with_q(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
    base_q_idx: u8,
    p_index: u32,
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
    fh.refresh_frame_flags = 1 << ((p_index - 1) & 1);
    // §5.9.21: ONLY_4X4 rides the CodedLossless arm; the lossy arm
    // codes TX_MODE_SELECT — intra leaves carry the §5.11.15
    // `tx_depth` choice and inter leaves the §5.11.17 `txfm_split`
    // recursion.
    fh.tx_mode = Some(if base_q_idx == 0 {
        TxMode::Only4x4
    } else {
        TxMode::TxModeSelect
    });
    // r412: per-block single/compound choice (§5.9.23) — compound
    // COMPOUND_AVERAGE leaves need the comp_mode dispatch open.
    fh.reference_select = Some(true);
    fh.skip_mode_present = Some(false);
    fh.allow_warped_motion = Some(false);
    let last_slot = (p_index & 1) as u8;
    let golden_slot = ((p_index - 1) & 1) as u8;
    let mut ref_frame_idx = [last_slot; REFS_PER_FRAME];
    // GOLDEN_FRAME = 4 → ref_frame_idx[ GOLDEN_FRAME - LAST_FRAME = 3 ].
    ref_frame_idx[3] = golden_slot;
    fh.inter_refs = Some(InterFrameRefs {
        frame_refs_short_signaling: false,
        last_frame_idx: None,
        gold_frame_idx: None,
        ref_frame_idx,
        allow_high_precision_mv: false,
        interpolation_filter: InterpolationFilter::Switchable,
        is_motion_mode_switchable: false,
        use_ref_frame_mvs: false,
    });
    fh
}

/// Encode one INTER P-frame against `prev` (the previous frame's
/// reconstruction, LAST_FRAME) and `prevprev` (the frame before it,
/// GOLDEN_FRAME — the KEY recon again for the first two P-frames).
/// Returns the §7.5 temporal unit (TD + `OBU_FRAME`) and this frame's
/// reconstruction.
fn encode_p_frame_yuv420(
    input: &Yuv420Frame,
    prev: &GopFrameRecon,
    prevprev: &GopFrameRecon,
    seq: &SequenceHeader,
    base_q_idx: u8,
    p_index: u32,
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
    for reference in [prev, prevprev] {
        if reference.y.len() != width * height
            || reference.u.len() != chroma_w * chroma_h
            || reference.v.len() != chroma_w * chroma_h
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    let fh =
        build_p_frame_yuv420_8bit_fh_with_q(seq, input.width, input.height, base_q_idx, p_index);
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
    let lossless = base_q_idx == 0;
    let qp = QuantizerParams::neutral(base_q_idx, 8);

    // ONE §5.9 inter-frame parameter bundle shared verbatim by the
    // driver-side §7.10.2 mirror and the write pass — the two
    // `find_mv_stack` runs at each leaf must see identical inputs.
    let mut ip = SyntaxInterFrameParams::single_ref_baseline(
        mi_rows, mi_cols, /* force_integer_mv = */ false,
    );
    // r412: SWITCHABLE frame filter — each inter leaf RD-selects its
    // own §5.11.x interp_filter (the header writer emits
    // `is_filter_switchable = 1`).
    ip.interpolation_filter = SWITCHABLE;
    // r412: per-block single/compound reference choice (§5.9.23).
    ip.reference_select = true;

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
        inter: Some(ip.clone()),
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
    let mut ictx = PSearchCtx::new(prev, prevprev, mi_rows, mi_cols, width, height, ip, p_index)?;

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
///
/// r412 adds [`PSearchCtx::mirror`]: a full [`PartitionWalker`] the
/// RD search stamps with each COMMITTED leaf (via
/// [`PSearchCtx::stamp_leaf`], the same
/// `stamp_encoder_block_syntax` values the write pass stamps) so the
/// §7.10.2 `find_mv_stack` scan can run at SEARCH time with exactly
/// the state the write-pass mirror will hold at that leaf — the
/// NEARESTMV / NEARMV / drl mode selection depends on it. Trials are
/// rolled back with the rect snapshot pair
/// ([`PartitionWalker::snapshot_encoder_stamp_rect`] /
/// [`PartitionWalker::restore_encoder_stamp_rect`]), mirroring the
/// pixel-plane `save_region` / `restore_region` discipline.
struct PSearchCtx {
    mi_rows: u32,
    mi_cols: u32,
    luma_w: u32,
    luma_h: u32,
    /// The two reference reconstructions, widened to the §7.11.3.4
    /// sample type: `ref_planes[ 0 ]` = the previous frame
    /// (LAST_FRAME), `ref_planes[ 1 ]` = the frame before it
    /// (GOLDEN_FRAME).
    ref_planes: [[Vec<u16>; 3]; 2],
    /// §7.20 slot each of the two references occupies this frame
    /// (`[ last_slot, golden_slot ]` — the r412 two-slot rotation).
    ref_slots: [usize; 2],
    /// §5.9.2 `ref_frame_idx[ 0..7 ]` — the header's slot map, fed
    /// verbatim to the decoder's leaf driver.
    ref_frame_idx: [u8; 7],
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
    /// r412 — `CompoundTypes[ .. ]` grid pinned at
    /// [`crate::cdf::COMPOUND_AVERAGE`]: the §5.11.29 derivation on
    /// this configuration (`enable_masked_compound ==
    /// enable_jnt_comp == false` ⇒ `comp_group_idx = 0`,
    /// `compound_idx = 1` ⇒ COMPOUND_AVERAGE, no bits); read only at
    /// compound leaf origins.
    comp_avg: Vec<u8>,
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
    /// r412 — driver-side write-mirror twin for §7.10.2 MV prediction
    /// (see the struct docs).
    mirror: PartitionWalker,
    /// The §5.9 inter bundle shared with the write pass — the
    /// `find_mv_stack` argument set.
    ip: SyntaxInterFrameParams,
}

impl PSearchCtx {
    #[allow(clippy::too_many_arguments)]
    fn new(
        prev: &GopFrameRecon,
        prevprev: &GopFrameRecon,
        mi_rows: u32,
        mi_cols: u32,
        width: usize,
        height: usize,
        ip: SyntaxInterFrameParams,
        p_index: u32,
    ) -> Result<Self, Error> {
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
        let mirror = PartitionWalker::new(
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
        Ok(PSearchCtx {
            mi_rows,
            mi_cols,
            luma_w: width as u32,
            luma_h: height as u32,
            ref_planes: [
                [widen(&prev.y), widen(&prev.u), widen(&prev.v)],
                [widen(&prevprev.y), widen(&prevprev.u), widen(&prevprev.v)],
            ],
            ref_slots: [(p_index & 1) as usize, ((p_index - 1) & 1) as usize],
            ref_frame_idx: {
                let mut idx = [(p_index & 1) as u8; 7];
                idx[3] = ((p_index - 1) & 1) as u8;
                idx
            },
            mi_sizes: vec![BLOCK_4X4; cells],
            is_inters: vec![0u8; cells],
            ref_frames,
            mvs: vec![0i16; cells * 4],
            interp_filters: vec![EIGHTTAP; cells * 2],
            motion_modes: vec![MOTION_MODE_SIMPLE; cells],
            y_modes: vec![0u8; cells],
            zeros: vec![0u8; cells],
            comp_avg: vec![crate::cdf::COMPOUND_AVERAGE; cells],
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
            mirror,
            ip,
        })
    }

    /// §7.10.2 `find_mv_stack( 0 )` against the driver mirror for the
    /// single-reference LAST_FRAME leaf at `(mi_row, mi_col)` — the
    /// argument set is character-for-character the write arm's
    /// (`write_block_syntax_inter_frame` threads the same
    /// [`SyntaxInterFrameParams`] fields), so both scans agree by
    /// construction whenever the two mirrors hold the same grids.
    fn find_stack(
        &self,
        mi_row: u32,
        mi_col: u32,
        b_size: usize,
        ref_frame: [i8; 2],
    ) -> Result<FindMvStackResult, Error> {
        self.mirror.find_mv_stack(
            mi_row,
            mi_col,
            b_size,
            [i32::from(ref_frame[0]), i32::from(ref_frame[1])],
            /* is_compound = */ ref_frame[1] > 0,
            self.ip.use_ref_frame_mvs,
            self.ip.gm_type,
            self.ip.gm_params,
            self.ip.ref_frame_sign_bias,
            self.ip.allow_high_precision_mv,
            self.ip.force_integer_mv,
            &self.ip.motion_field_mvs,
        )
    }

    /// Stamp a COMMITTED leaf into the driver mirror with the same
    /// `stamp_encoder_block_syntax` values the write pass stamps for
    /// this leaf (`cdef: None` per the
    /// [`PartitionWalker::snapshot_encoder_stamp_rect`] contract —
    /// the §5.11.56 anchor can land outside the trial rect, and no
    /// §7.10.2 / §8.3.2 read the search performs consults it).
    fn stamp_leaf(
        &mut self,
        block: &SyntaxBlock,
        mi_row: u32,
        mi_col: u32,
        b_size: usize,
        lossless: bool,
    ) {
        use crate::cdf::EncoderBlockSyntaxStamp;
        let tx_pre = if lossless {
            TX_4X4 as u8
        } else {
            MAX_TX_SIZE_RECT[b_size] as u8
        };
        let stamp = match block.inter.as_ref() {
            Some(ib) => EncoderBlockSyntaxStamp {
                mi_row,
                mi_col,
                sub_size: b_size,
                skip: block.skip,
                segment_id: block.segment_id,
                is_inter: 1,
                y_mode: ib.y_mode,
                ref_frame: ib.ref_frame,
                mv: ib.mv[0],
                mv2: ib.mv[1],
                interp_filter: ib.interp_filter,
                motion_mode: MOTION_MODE_SIMPLE,
                palette_size_y: 0,
                palette_colors_y: &[],
                palette_size_uv: 0,
                palette_colors_u: &[],
                palette_colors_v: &[],
                cdef: None,
                tx_size: tx_pre,
            },
            None => EncoderBlockSyntaxStamp {
                mi_row,
                mi_col,
                sub_size: b_size,
                skip: block.skip,
                segment_id: block.segment_id,
                is_inter: 0,
                y_mode: block.y_mode,
                ref_frame: [0, -1],
                mv: [0, 0],
                mv2: [0, 0],
                interp_filter: [0, 0],
                motion_mode: MOTION_MODE_SIMPLE,
                palette_size_y: block.palette.size_y,
                palette_colors_y: &block.palette.colors_y,
                palette_size_uv: block.palette.size_uv,
                palette_colors_u: &block.palette.colors_u,
                palette_colors_v: &block.palette.colors_v,
                cdef: None,
                tx_size: tx_pre,
            },
        };
        self.mirror.stamp_encoder_block_syntax(&stamp);
    }

    /// Stamp the leaf's §5.11.5 grid footprint (the same values the
    /// write mirror / decode walker stamp), then run the decoder's
    /// §5.11.33 leaf driver into the scratch planes.
    #[allow(clippy::too_many_arguments)]
    fn predict_leaf(
        &mut self,
        mi_row: u32,
        mi_col: u32,
        b_size: usize,
        ref_frame: [i8; 2],
        y_mode: u8,
        mv: [[i32; 2]; 2],
        filter: u8,
    ) -> Result<(), Error> {
        let bw4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[b_size] as u32;
        for dr in 0..bh4 {
            let rr = mi_row + dr;
            if rr >= self.mi_rows {
                break;
            }
            for dc in 0..bw4 {
                let cc = mi_col + dc;
                if cc >= self.mi_cols {
                    break;
                }
                let cell = (rr * self.mi_cols + cc) as usize;
                self.mi_sizes[cell] = b_size;
                self.is_inters[cell] = 1;
                self.ref_frames[cell * 2] = ref_frame[0];
                self.ref_frames[cell * 2 + 1] = ref_frame[1];
                self.mvs[cell * 4] = mv[0][0] as i16;
                self.mvs[cell * 4 + 1] = mv[0][1] as i16;
                self.mvs[cell * 4 + 2] = mv[1][0] as i16;
                self.mvs[cell * 4 + 3] = mv[1][1] as i16;
                self.interp_filters[cell * 2] = filter;
                self.interp_filters[cell * 2 + 1] = filter;
                self.motion_modes[cell] = MOTION_MODE_SIMPLE;
                self.y_modes[cell] = y_mode;
            }
        }

        // Field-split borrows: the grid views borrow the mode-info
        // vectors immutably while the plane contexts borrow the
        // scratch planes mutably.
        let (luma_w, luma_h) = (self.luma_w, self.luma_h);
        let (mi_rows, mi_cols) = (self.mi_rows, self.mi_cols);
        let ref_slots = self.ref_slots;
        let ref_frame_idx = self.ref_frame_idx;
        let PSearchCtx {
            ref_planes,
            comp_avg,
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

        // §7.20 `FrameStore` views — the r412 two-slot rotation:
        // `golden_slot` holds the frame-before-previous, every other
        // slot the previous frame (only the two rotated slots are
        // ever referenced through `ref_frame_idx`). Dimensions are
        // LUMA extents per the r405 contract; strides are plane
        // samples.
        let store_y = make_store(
            &ref_planes[0][0],
            &ref_planes[1][0],
            ref_slots[1],
            luma_w as usize,
            luma_w,
            luma_h,
        );
        let store_u = make_store(
            &ref_planes[0][1],
            &ref_planes[1][1],
            ref_slots[1],
            (luma_w as usize) / 2,
            luma_w,
            luma_h,
        );
        let store_v = make_store(
            &ref_planes[0][2],
            &ref_planes[1][2],
            ref_slots[1],
            (luma_w as usize) / 2,
            luma_w,
            luma_h,
        );

        let grid = InterModeInfoGrid {
            mi_sizes,
            is_inters,
            ref_frames,
            mvs,
            interp_filters,
            compound_types: comp_avg,
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
            &ref_frame_idx,
            &mut planes,
            mi_row as usize,
            mi_col as usize,
        )
    }
}

/// One plane's §7.20 `FrameStore` view for the r412 two-slot
/// rotation: `golden_slot` holds the frame-before-previous plane,
/// every other slot the previous frame (extents in LUMA samples per
/// the r405 contract, stride in this plane's own samples). Slots
/// outside the two rotated ones are never resolved through
/// `ref_frame_idx`, so their content is immaterial.
fn make_store<'a>(
    prev: &'a [u16],
    prevprev: &'a [u16],
    golden_slot: usize,
    stride: usize,
    luma_w: u32,
    luma_h: u32,
) -> [RefFrameStoreEntry<'a>; 8] {
    core::array::from_fn(|slot| RefFrameStoreEntry {
        plane: if slot == golden_slot { prevprev } else { prev },
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
/// `bw × bh` are the (possibly rectangular) luma block dimensions.
#[allow(clippy::too_many_arguments)]
fn motion_search_luma(
    input: &Yuv420Frame,
    ref_y: &[u16],
    width: usize,
    height: usize,
    row0: usize,
    col0: usize,
    bw: usize,
    bh: usize,
) -> [i32; 2] {
    let n = bw.max(bh) as u64;
    let cost_at = |dy: i32, dx: i32| -> u64 {
        let mut ssd = 0u64;
        for i in 0..bh {
            let sy = ((row0 + i) as i32 + dy).clamp(0, height as i32 - 1) as usize;
            for j in 0..bw {
                let sx = ((col0 + j) as i32 + dx).clamp(0, width as i32 - 1) as usize;
                let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                    - i64::from(ref_y[sy * width + sx]);
                ssd += (d * d) as u64;
            }
        }
        ssd + (dy.unsigned_abs() as u64 + dx.unsigned_abs() as u64) * n
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
    ref_frame: i8,
    row0: usize,
    col0: usize,
    bw: usize,
    bh: usize,
    width: usize,
    mv_int: [i32; 2],
) -> Result<[i32; 2], Error> {
    let score = |ictx: &mut PSearchCtx, mv: [i32; 2]| -> Result<u64, Error> {
        ictx.predict_leaf(
            mi_r,
            mi_c,
            b_size,
            [ref_frame, -1],
            MODE_NEWMV,
            [mv, [0, 0]],
            EIGHTTAP,
        )?;
        let mut ssd = 0u64;
        for i in 0..bh {
            for j in 0..bw {
                let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                    - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                ssd += (d * d) as u64;
            }
        }
        Ok(ssd + ((mv[0].unsigned_abs() + mv[1].unsigned_abs()) as u64) * (bw.max(bh) as u64) / 8)
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

/// Encode one in-frame INTER leaf at `b_size` (`BLOCK_8X8` …
/// `BLOCK_64X64`, plus the r412 rectangular HORZ/VERT halves —
/// `BLOCK_16X8` and larger): motion search, §7.11.3 prediction
/// through the decoder's leaf driver, residual per TU (one
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
    let bw4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let bh4 = NUM_4X4_BLOCKS_HIGH[b_size];
    let (bw, bh) = (bw4 * 4, bh4 * 4);
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let width = recon.width;
    let lossless = recon.lossless;

    // r412 — §5.11.24 single-pred mode + reference selection over the
    // full candidate set the syntax can express at this leaf. For
    // each codable reference (LAST_FRAME = the previous frame,
    // GOLDEN_FRAME = the frame before it — the r412 two-slot
    // rotation) the §7.10.2 `find_mv_stack` scan runs against the
    // driver mirror (the same scan the write pass will re-derive),
    // then:
    //
    // * `NEWMV` at the searched vector, with the §5.11.23 `drl_mode`
    //   index chosen to minimise the §5.11.32 difference bits against
    //   the reachable `PredMv` slots (`{0}` for `NumMvFound <= 1`,
    //   `{0, 1}` at 2, `{0, 1, 2}` beyond — the reader's loop bounds);
    // * `NEARESTMV` at `RefStackMv[ 0 ][ 0 ]` (no MV bits);
    // * `NEARMV` at every `RefStackMv[ idx ][ 0 ]` the drl loop can
    //   reach (`idx = 1` always — the arm's silent start value — plus
    //   `2` at `NumMvFound >= 3` and `3` at `NumMvFound >= 4`);
    // * `GLOBALMV` at the §7.10.2.1 derivation (identity warp ⇒ the
    //   zero vector on this frame configuration).
    //
    // Each candidate's prediction runs through the decoder's own
    // §7.11.3 leaf driver; the winner minimises luma SSD + λ · a
    // small per-mode rate proxy (mode cascade + drl + MV-difference
    // bits + a one-bit §5.11.25 cascade surcharge on the non-LAST
    // reference), mirroring [`p_leaf_rate`]'s constants.
    struct ModeCand {
        ref_frame: [i8; 2],
        y_mode: u8,
        mv: [[i32; 2]; 2],
        ref_mv_idx: u32,
        rate: u64,
    }
    let diff_bits = |mv: [i32; 2], pred: [i32; 2]| -> u64 {
        let b = |d: i32| u64::from(34 - d.unsigned_abs().leading_zeros());
        b(mv[0] - pred[0]) + b(mv[1] - pred[1])
    };
    let mut cands: Vec<ModeCand> = Vec::with_capacity(18);
    let mut searched_mv: [[i32; 2]; 2] = [[0, 0]; 2];
    for (ref_ord, rf) in [(0usize, 1i8), (1, 4)] {
        let ref_bias = ref_ord as u64;
        let stack = ictx.find_stack(mi_r, mi_c, b_size, [rf, -1])?;
        let mv_int = motion_search_luma(
            input,
            &ictx.ref_planes[ref_ord][0],
            width,
            recon.height,
            row0,
            col0,
            bw,
            bh,
        );
        // r411 sub-pel refinement: half-pel then quarter-pel deltas
        // around the integer winner, each candidate evaluated through
        // the REAL §7.11.3.4 kernel (so the search cost IS the coding
        // cost). `allow_high_precision_mv = 0` restricts components
        // to quarter-pel (multiples of 2 in 1/8-luma units).
        let mv_new = refine_mv_subpel(
            input, ictx, mi_r, mi_c, b_size, rf, row0, col0, bw, bh, width, mv_int,
        )?;
        searched_mv[ref_ord] = mv_new;
        let nfound = stack.num_mv_found;
        {
            // NEWMV: pick the reachable drl slot with the cheapest
            // §5.11.32 difference (each extra slot costs one
            // drl_mode bit).
            let window: u32 = match nfound {
                0 | 1 => 0,
                2 => 1,
                _ => 2,
            };
            let mut best: Option<(u64, u32)> = None;
            for idx in 0..=window {
                let pred = assign_mv_pred_mv(&stack, MODE_NEWMV, 0, idx)?;
                let rate = 5 + ref_bias + u64::from(idx) + diff_bits(mv_new, pred);
                if best.map_or(true, |(r, _)| rate < r) {
                    best = Some((rate, idx));
                }
            }
            let (rate, idx) = best.expect("slot 0 is always reachable");
            cands.push(ModeCand {
                ref_frame: [rf, -1],
                y_mode: MODE_NEWMV,
                mv: [mv_new, [0, 0]],
                ref_mv_idx: idx,
                rate,
            });
        }
        cands.push(ModeCand {
            ref_frame: [rf, -1],
            y_mode: MODE_NEARESTMV,
            mv: [stack.ref_stack_mv[0][0], [0, 0]],
            ref_mv_idx: 0,
            rate: 3 + ref_bias,
        });
        let near_top: u32 = match nfound {
            0..=2 => 1,
            3 => 2,
            _ => 3,
        };
        for idx in 1..=near_top {
            cands.push(ModeCand {
                ref_frame: [rf, -1],
                y_mode: MODE_NEARMV,
                mv: [stack.ref_stack_mv[idx as usize][0], [0, 0]],
                ref_mv_idx: idx,
                rate: 4 + ref_bias + u64::from(idx - 1),
            });
        }
        cands.push(ModeCand {
            ref_frame: [rf, -1],
            y_mode: MODE_GLOBALMV,
            mv: [stack.global_mvs[0], [0, 0]],
            ref_mv_idx: 0,
            rate: 4 + ref_bias,
        });
    }

    // r412 — COMPOUND_AVERAGE candidates over the { LAST, GOLDEN }
    // pair (§5.11.25 unidirectional compound; the §5.11.29 tail is
    // bit-silent on this configuration and derives COMPOUND_AVERAGE):
    // NEAREST_NEARESTMV / NEAR_NEARMV from the compound §7.10.2
    // stack, GLOBAL_GLOBALMV at the identity derivation, and
    // NEW_NEWMV re-using the two per-reference searched vectors.
    {
        let crf: [i8; 2] = [1, 4];
        let stack = ictx.find_stack(mi_r, mi_c, b_size, crf)?;
        let nfound = stack.num_mv_found;
        cands.push(ModeCand {
            ref_frame: crf,
            y_mode: MODE_NEAREST_NEARESTMV,
            mv: [stack.ref_stack_mv[0][0], stack.ref_stack_mv[0][1]],
            ref_mv_idx: 0,
            rate: 6,
        });
        let near_top: u32 = match nfound {
            0..=2 => 1,
            3 => 2,
            _ => 3,
        };
        for idx in 1..=near_top {
            cands.push(ModeCand {
                ref_frame: crf,
                y_mode: MODE_NEAR_NEARMV,
                mv: [
                    stack.ref_stack_mv[idx as usize][0],
                    stack.ref_stack_mv[idx as usize][1],
                ],
                ref_mv_idx: idx,
                rate: 7 + u64::from(idx - 1),
            });
        }
        cands.push(ModeCand {
            ref_frame: crf,
            y_mode: MODE_GLOBAL_GLOBALMV,
            mv: [stack.global_mvs[0], stack.global_mvs[1]],
            ref_mv_idx: 0,
            rate: 7,
        });
        {
            let window: u32 = match nfound {
                0 | 1 => 0,
                2 => 1,
                _ => 2,
            };
            let mut best: Option<(u64, u32)> = None;
            for idx in 0..=window {
                let pred0 = assign_mv_pred_mv(&stack, MODE_NEW_NEWMV, 0, idx)?;
                let pred1 = assign_mv_pred_mv(&stack, MODE_NEW_NEWMV, 1, idx)?;
                let rate = 8
                    + u64::from(idx)
                    + diff_bits(searched_mv[0], pred0)
                    + diff_bits(searched_mv[1], pred1);
                if best.map_or(true, |(r, _)| rate < r) {
                    best = Some((rate, idx));
                }
            }
            let (rate, idx) = best.expect("slot 0 is always reachable");
            cands.push(ModeCand {
                ref_frame: crf,
                y_mode: MODE_NEW_NEWMV,
                mv: searched_mv,
                ref_mv_idx: idx,
                rate,
            });
        }
    }
    // §6.10.27 conformance bound — the writers reject any leaf beyond
    // it, so a (pathological) out-of-bound stack candidate is simply
    // not offered.
    cands.retain(|c| {
        c.mv.iter()
            .all(|m| m[0].unsigned_abs() < (1 << 14) && m[1].unsigned_abs() < (1 << 14))
    });

    let lambda = lambda_for(&recon.qp);
    let mut best: Option<(u64, usize)> = None;
    for (ci, cand) in cands.iter().enumerate() {
        ictx.predict_leaf(
            mi_r,
            mi_c,
            b_size,
            cand.ref_frame,
            cand.y_mode,
            cand.mv,
            EIGHTTAP,
        )?;
        let mut ssd = 0u64;
        for i in 0..bh {
            for j in 0..bw {
                let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                    - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                ssd += (d * d) as u64;
            }
        }
        let score = ssd + lambda * cand.rate;
        if best.map_or(true, |(s, _)| score < s) {
            best = Some((score, ci));
        }
    }
    let (_, best_ci) = best.ok_or(Error::PartitionWalkOutOfRange)?;
    let ModeCand {
        ref_frame,
        y_mode,
        mv,
        ref_mv_idx,
        ..
    } = cands[best_ci];

    // r412 — §5.11.x SWITCHABLE interpolation-filter RD selection on
    // the winning (mode, MV): trial the three codable filters
    // (`EIGHTTAP` / `EIGHTTAP_SMOOTH` / `EIGHTTAP_SHARP`) through the
    // decoder's own §7.11.3.4 kernel and keep the lowest luma SSD
    // (the S() costs one ~equal-rate symbol whichever value is coded,
    // so distortion decides; ties keep EIGHTTAP). GLOBALMV /
    // GLOBAL_GLOBALMV leaves are `needs_interp_filter( ) == 0` on the
    // identity-warp frame configuration — the reader derives EIGHTTAP
    // with no bits, so no search happens and the committed pair must
    // be EIGHTTAP.
    let filter = if y_mode == MODE_GLOBALMV || y_mode == MODE_GLOBAL_GLOBALMV {
        EIGHTTAP
    } else {
        use crate::inter_pred::{EIGHTTAP_SHARP, EIGHTTAP_SMOOTH};
        let mut best_f = (u64::MAX, EIGHTTAP);
        for f in [EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP] {
            ictx.predict_leaf(mi_r, mi_c, b_size, ref_frame, y_mode, mv, f)?;
            let mut ssd = 0u64;
            for i in 0..bh {
                for j in 0..bw {
                    let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                        - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                    ssd += (d * d) as u64;
                }
            }
            if ssd < best_f.0 {
                best_f = (ssd, f);
            }
        }
        best_f.1
    };
    ictx.predict_leaf(mi_r, mi_c, b_size, ref_frame, y_mode, mv, filter)?;

    if lossless {
        // §5.9.2 CodedLossless: TX_4X4 everywhere, no §5.11.17 trees.
        return encode_inter_leaf_residual(
            mi_r, mi_c, b_size, input, recon, ictx, ref_frame, y_mode, mv, ref_mv_idx, filter, 0,
        );
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
    let before = save_region_wh(recon, mi_r, mi_c, bw4, bh4);
    let mut best: Option<(SyntaxBlock, RegionSnapshot, u64)> = None;
    for depth in 0..=max_depth {
        let leaf = encode_inter_leaf_residual(
            mi_r, mi_c, b_size, input, recon, ictx, ref_frame, y_mode, mv, ref_mv_idx, filter,
            depth,
        )?;
        let d = region_distortion_wh(recon, input, mi_r, mi_c, bw4, bh4);
        let score = d + lambda * (p_leaf_rate(&leaf) + 2 * u64::from(depth));
        let improves = match best.as_ref() {
            Some((_, _, s)) => score < *s,
            None => true,
        };
        if improves {
            best = Some((leaf, save_region_wh(recon, mi_r, mi_c, bw4, bh4), score));
        }
        restore_region(recon, mi_r, mi_c, &before);
    }
    let (leaf, after, _) = best.expect("at least depth 0");
    restore_region(recon, mi_r, mi_c, &after);
    Ok(leaf)
}

/// §5.11.36 / §5.11.17 luma TU visit order for a uniform-depth
/// `read_var_tx_size` recursion rooted at `tx` (r412: rect-aware —
/// each split level visits `(h4 / stepH) * (w4 / stepW)` children of
/// `Split_Tx_Size[ tx ]` in row-major `(i, j)` order, which is 4 for
/// square ordinals and 2 for rectangular ones; the pre-r412 square
/// quadtree order falls out as the square special case).
fn transform_tree_tu_order(
    x: usize,
    y: usize,
    tx: usize,
    depth: u32,
    out: &mut Vec<(usize, usize)>,
) {
    if depth == 0 {
        out.push((x, y));
        return;
    }
    let sub = SPLIT_TX_SIZE[tx];
    let (sw, sh) = (TX_WIDTH[sub], TX_HEIGHT[sub]);
    for i in 0..TX_HEIGHT[tx] / sh {
        for j in 0..TX_WIDTH[tx] / sw {
            transform_tree_tu_order(x + j * sw, y + i * sh, sub, depth - 1, out);
        }
    }
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

/// §5.11.17 uniform split-decision tree of the given depth rooted at
/// `tx` (r412: rect-aware — the per-split child count is
/// `(h4 / stepH) * (w4 / stepW)`, 4 for square ordinals and 2 for
/// rectangular ones, matching [`VarTxSyntaxTree::Split`]'s contract).
fn uniform_var_tx_tree(tx: usize, depth: u32) -> VarTxSyntaxTree {
    if depth == 0 {
        VarTxSyntaxTree::Leaf
    } else {
        let sub = SPLIT_TX_SIZE[tx];
        let count = (TX_HEIGHT[tx] / TX_HEIGHT[sub]) * (TX_WIDTH[tx] / TX_WIDTH[sub]);
        VarTxSyntaxTree::Split(
            (0..count)
                .map(|_| uniform_var_tx_tree(sub, depth - 1))
                .collect(),
        )
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
    ref_frame: [i8; 2],
    y_mode: u8,
    mv: [[i32; 2]; 2],
    ref_mv_idx: u32,
    filter: u8,
    depth: u32,
) -> Result<SyntaxBlock, Error> {
    let bw4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let bh4 = NUM_4X4_BLOCKS_HIGH[b_size];
    let (bw, bh) = (bw4 * 4, bh4 * 4);
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
        while ty < bh {
            let mut tx = 0usize;
            while tx < bw {
                tu_order.push((tx, ty));
                tx += ltw;
            }
            ty += lth;
        }
    } else {
        // §5.11.36 transform-tree recursion order (`Max_Tx_Size_Rect`
        // covers the whole block, so exactly one tree — its
        // uniform-depth leaves in the §5.11.17 row-major visit).
        transform_tree_tu_order(0, 0, MAX_TX_SIZE_RECT[b_size], depth, &mut tu_order);
    }
    let mut residual_quant: Vec<Vec<i32>> = Vec::new();
    let mut luma_tx_types: Vec<u8> = Vec::new();
    // Block-relative luma-TU-grid map of committed §5.11.47 types —
    // the §5.11.40 chroma-inheritance source (`TxTypes[ y4 ][ x4 ]`).
    let tu_cols = bw / ltw;
    let mut tu_type_grid = vec![DCT_DCT as u8; tu_cols * (bh / lth)];
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
    let (cbw, cbh) = (bw / 2, bh / 2);
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
        while ty < cbh {
            let mut tx = 0usize;
            while tx < cbw {
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
        block.var_tx_trees = vec![uniform_var_tx_tree(MAX_TX_SIZE_RECT[b_size], depth)];
    }
    block.inter = Some(SyntaxInterBlock {
        ref_frame,
        y_mode,
        mv,
        ref_mv_idx,
        interp_filter: [filter; 2],
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
            if ib.ref_frame[0] != 1 {
                // §5.11.25 single-ref cascade surcharge on the
                // non-LAST reference.
                rate += 1;
            }
            match ib.y_mode {
                MODE_NEWMV => {
                    let bits = |v: i32| u64::from(34 - v.unsigned_abs().leading_zeros());
                    rate += 6 + bits(ib.mv[0][0]) + bits(ib.mv[0][1]);
                }
                MODE_NEARESTMV => rate += 3,
                MODE_NEARMV => rate += 4 + u64::from(ib.ref_mv_idx.saturating_sub(1)),
                MODE_NEAREST_NEARESTMV => rate += 6,
                MODE_NEAR_NEARMV => rate += 7 + u64::from(ib.ref_mv_idx.saturating_sub(1)),
                MODE_NEW_NEWMV => {
                    let bits = |v: i32| u64::from(34 - v.unsigned_abs().leading_zeros());
                    rate += 8
                        + bits(ib.mv[0][0])
                        + bits(ib.mv[0][1])
                        + bits(ib.mv[1][0])
                        + bits(ib.mv[1][1]);
                }
                // GLOBALMV / GLOBAL_GLOBALMV / the remaining compound
                // modes — cascade context bits plus headroom.
                _ => rate += 4 + 3 * u64::from(ib.ref_frame[1] > 0),
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
        SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
            4 + blocks.iter().map(|b| p_leaf_rate(b)).sum::<u64>()
        }
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
    // r412: the driver mirror follows the same trial / rollback
    // discipline as the pixel planes — every candidate leaves the
    // mirror holding ITS committed stamps, and the loser's stamps are
    // rolled back so subsequent §7.10.2 scans see only the winner.
    let m_before = ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4);

    // Candidate A: one INTER leaf.
    let inter_leaf = encode_inter_leaf(r, c, b_size, input, recon, ictx)?;
    ictx.stamp_leaf(&inter_leaf, r, c, b_size, recon.lossless);
    let d_inter = region_distortion(recon, input, r, c, n4 as usize);
    let r_inter = p_leaf_rate(&inter_leaf);
    let score_inter = d_inter + lambda * r_inter;
    let after_inter = save_region(recon, r, c, n4 as usize);
    let m_after_inter = ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4);
    restore_region(recon, r, c, &before);
    ictx.mirror.restore_encoder_stamp_rect(&m_before);

    // Candidate B: one INTRA leaf (§5.11.22 arm) with the KEY
    // driver's §5.11.15 tx_depth RD search (TX_MODE_SELECT on the
    // lossy arm; the lossless arm stays on the TX_4X4 grid).
    let intra_leaf = encode_leaf_sq(r, c, b_size, input, recon);
    ictx.stamp_leaf(&intra_leaf, r, c, b_size, recon.lossless);
    let d_intra = region_distortion(recon, input, r, c, n4 as usize);
    let r_intra = p_leaf_rate(&intra_leaf);
    let score_intra = d_intra + lambda * r_intra;
    let after_intra = save_region(recon, r, c, n4 as usize);
    let m_after_intra = ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4);
    restore_region(recon, r, c, &before);
    ictx.mirror.restore_encoder_stamp_rect(&m_before);

    let (leaf, after_leaf, m_after_leaf, score_leaf) = if score_inter <= score_intra {
        (inter_leaf, after_inter, m_after_inter, score_inter)
    } else {
        (intra_leaf, after_intra, m_after_intra, score_intra)
    };

    if b_size <= BLOCK_8X8 {
        // P-frame leaf floor — no split candidate.
        restore_region(recon, r, c, &after_leaf);
        ictx.mirror.restore_encoder_stamp_rect(&m_after_leaf);
        return Ok(SyntaxNode::Leaf(Box::new(leaf)));
    }

    // Running best over the non-split candidates: NONE-leaf first,
    // then the r412 HORZ / VERT rectangular trials.
    let mut best_node: (SyntaxNode, RegionSnapshot, _, u64) = (
        SyntaxNode::Leaf(Box::new(leaf)),
        after_leaf,
        m_after_leaf,
        score_leaf,
    );

    // Candidates D/E: §5.11.4 PARTITION_HORZ / PARTITION_VERT — two
    // rectangular INTER halves (`Partition_Subsize[ p ][ bSize ]`,
    // BLOCK_16X8 and larger; intra halves stay the NONE-leaf's and
    // SPLIT's territory). The halves are encoded in §5.11.4 dispatch
    // order, so the second half's §7.10.2 scan and prediction see the
    // first half's stamps exactly as the decoder will.
    for part in [PARTITION_HORZ, PARTITION_VERT] {
        let sub = match crate::cdf::partition_subsize(part, b_size) {
            Some(sz) => sz,
            None => continue,
        };
        let (r1, c1) = if part == PARTITION_HORZ {
            (r + half, c)
        } else {
            (r, c + half)
        };
        let first = encode_inter_leaf(r, c, sub, input, recon, ictx)?;
        ictx.stamp_leaf(&first, r, c, sub, recon.lossless);
        let second = encode_inter_leaf(r1, c1, sub, input, recon, ictx)?;
        ictx.stamp_leaf(&second, r1, c1, sub, recon.lossless);
        let d = region_distortion(recon, input, r, c, n4 as usize);
        let rate = p_leaf_rate(&first) + p_leaf_rate(&second) + 4;
        let score = d + lambda * rate;
        if score < best_node.3 {
            let blocks = [Box::new(first), Box::new(second)];
            let node = if part == PARTITION_HORZ {
                SyntaxNode::Horz(blocks)
            } else {
                SyntaxNode::Vert(blocks)
            };
            best_node = (
                node,
                save_region(recon, r, c, n4 as usize),
                ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4),
                score,
            );
        }
        restore_region(recon, r, c, &before);
        ictx.mirror.restore_encoder_stamp_rect(&m_before);
    }
    let (node_leafish, after_leafish, m_after_leafish, score_leafish) = best_node;

    // Candidate C: PARTITION_SPLIT into four recursively-searched
    // quadrants (NW/NE/SW/SE dispatch order — the mirror currently
    // holds the pre-node state, so each child's §7.10.2 scan sees its
    // earlier siblings' committed stamps exactly as the write pass
    // will).
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

    if score_leafish <= score_split {
        restore_region(recon, r, c, &after_leafish);
        ictx.mirror.restore_encoder_stamp_rect(&m_after_leafish);
        Ok(node_leafish)
    } else {
        Ok(SyntaxNode::Split(children))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic textured frame with translation `(sy, sx)` — the
    /// `gop_inter_conformance.rs` generator, duplicated here so the
    /// module test can drive the search internals directly.
    fn moving_gradient(w: u32, h: u32, shift_y: usize, shift_x: usize, seed: u32) -> Yuv420Frame {
        let (wu, hu) = (w as usize, h as usize);
        let s = seed as usize;
        let mut f = Yuv420Frame::filled(w, h, 0);
        for i in 0..hu {
            for j in 0..wu {
                let (si, sj) = (i + shift_y, j + shift_x);
                f.y[i * wu + j] = ((si * 5 + sj * 3 + (si / 16) * (sj / 16) + s) % 256) as u8;
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        for i in 0..ch {
            for j in 0..cw {
                let (si, sj) = (i + shift_y / 2, j + shift_x / 2);
                f.u[i * cw + j] = ((128 + si * 2 + sj + s) % 256) as u8;
                f.v[i * cw + j] = ((64 + si + sj * 2 + s) % 256) as u8;
            }
        }
        f
    }

    fn count_modes(node: &SyntaxNode, counts: &mut [u32; 4]) {
        match node {
            SyntaxNode::Leaf(b) => {
                if let Some(ib) = b.inter.as_ref() {
                    match ib.y_mode {
                        MODE_NEARESTMV => counts[0] += 1,
                        MODE_NEARMV => counts[1] += 1,
                        MODE_GLOBALMV => counts[2] += 1,
                        MODE_NEWMV => counts[3] += 1,
                        _ => {}
                    }
                }
            }
            SyntaxNode::Split(children) => {
                for ch in children.iter() {
                    count_modes(ch, counts);
                }
            }
            SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
                for b in blocks.iter() {
                    if let Some(ib) = b.inter.as_ref() {
                        match ib.y_mode {
                            MODE_NEARESTMV => counts[0] += 1,
                            MODE_NEARMV => counts[1] += 1,
                            MODE_GLOBALMV => counts[2] += 1,
                            MODE_NEWMV => counts[3] += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// r412 — on uniformly translating content the §5.11.24 mode
    /// selection must actually REACH the stack-predicted modes: after
    /// the first leaves code NEWMV, every later leaf's §7.10.2 stack
    /// carries the shared vector, so NEARESTMV / NEARMV (no MV bits)
    /// dominate. The search tree is inspected directly; the emitted
    /// stream's conformance is covered by `gop_inter_conformance.rs`
    /// (the write pass re-derives every stack and rejects any
    /// driver / write mirror divergence).
    #[test]
    fn r412_p_search_selects_stack_predicted_modes_on_uniform_motion() {
        let f0 = moving_gradient(64, 64, 0, 0, 7);
        let f1 = moving_gradient(64, 64, 3, 5, 7);
        let base_q_idx = 60u8;
        let key = encode_key_frame_yuv420_with_q(&f0, base_q_idx).unwrap();
        let reference = GopFrameRecon {
            y: key.recon_y,
            u: key.recon_u,
            v: key.recon_v,
        };
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 64, 64, base_q_idx, 1);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let qp = QuantizerParams::neutral(base_q_idx, 8);
        let mut recon = ReconState {
            y: vec![0u8; 64 * 64],
            u: vec![0u8; 32 * 32],
            v: vec![0u8; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            qp,
            bd: BlockDecodedMirror::new(),
        };
        let ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        let mut ictx =
            PSearchCtx::new(&reference, &reference, mi_rows, mi_cols, 64, 64, ip, 1).unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree(0, 0, BLOCK_64X64, &f1, &mut recon, &mut ictx).unwrap();
        let mut counts = [0u32; 4];
        count_modes(&tree, &mut counts);
        let total: u32 = counts.iter().sum();
        assert!(total > 0, "uniform motion must produce inter leaves");
        assert!(
            counts[0] + counts[1] > 0,
            "at least one NEARESTMV / NEARMV leaf must be selected on \
             uniformly translating content (got NEAREST={} NEAR={} \
             GLOBAL={} NEW={})",
            counts[0],
            counts[1],
            counts[2],
            counts[3]
        );
        assert!(
            counts[3] > 0,
            "the first block of the shared-motion region still codes NEWMV \
             (its stack is empty); got NEW={}",
            counts[3]
        );
    }

    /// r412 — the SWITCHABLE filter search must actually reach a
    /// non-EIGHTTAP choice when it is the distortion winner: the
    /// target P-frame is constructed as the reference's OWN
    /// EIGHTTAP_SHARP half-pel interpolation (through
    /// [`PSearchCtx::predict_leaf`], the same kernel the search
    /// scores with), so a SHARP-filtered leaf predicts it exactly and
    /// every committed inter leaf must carry EIGHTTAP_SHARP.
    #[test]
    fn r412_p_search_selects_sharp_filter_on_kernel_matched_content() {
        use crate::inter_pred::EIGHTTAP_SHARP;
        let f0 = moving_gradient(64, 64, 0, 0, 77);
        let base_q_idx = 90u8;
        let key = encode_key_frame_yuv420_with_q(&f0, base_q_idx).unwrap();
        let reference = GopFrameRecon {
            y: key.recon_y,
            u: key.recon_u,
            v: key.recon_v,
        };
        // Target frame = SHARP half-pel interpolation of the reference.
        let mut f1 = Yuv420Frame::filled(64, 64, 0);
        {
            let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
            let mut probe = PSearchCtx::new(&reference, &reference, 16, 16, 64, 64, ip, 1).unwrap();
            probe
                .predict_leaf(
                    0,
                    0,
                    BLOCK_64X64,
                    [1, -1],
                    MODE_NEWMV,
                    [[0, 4], [0, 0]],
                    EIGHTTAP_SHARP,
                )
                .unwrap();
            for (dst, &src) in f1.y.iter_mut().zip(probe.scratch[0].iter()) {
                *dst = src as u8;
            }
            for (dst, &src) in f1.u.iter_mut().zip(probe.scratch[1].iter()) {
                *dst = src as u8;
            }
            for (dst, &src) in f1.v.iter_mut().zip(probe.scratch[2].iter()) {
                *dst = src as u8;
            }
        }
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 64, 64, base_q_idx, 1);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u8; 64 * 64],
            u: vec![0u8; 32 * 32],
            v: vec![0u8; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bd: BlockDecodedMirror::new(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        let mut ictx =
            PSearchCtx::new(&reference, &reference, mi_rows, mi_cols, 64, 64, ip, 1).unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree(0, 0, BLOCK_64X64, &f1, &mut recon, &mut ictx).unwrap();
        fn count_filters(node: &SyntaxNode, counts: &mut [u32; 3]) {
            match node {
                SyntaxNode::Leaf(b) => {
                    if let Some(ib) = b.inter.as_ref() {
                        counts[usize::from(ib.interp_filter[0].min(2))] += 1;
                    }
                }
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count_filters(ch, counts);
                    }
                }
                SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
                    for b in blocks.iter() {
                        if let Some(ib) = b.inter.as_ref() {
                            counts[usize::from(ib.interp_filter[0].min(2))] += 1;
                        }
                    }
                }
            }
        }
        let mut counts = [0u32; 3];
        count_filters(&tree, &mut counts);
        let sharp = counts[usize::from(EIGHTTAP_SHARP)];
        assert!(
            sharp > 0,
            "kernel-matched half-pel content must commit EIGHTTAP_SHARP \
             leaves (got EIGHTTAP={} SMOOTH={} SHARP={sharp})",
            counts[0],
            counts[1]
        );
    }

    /// r412 — the HORZ/VERT rectangular trials must actually be
    /// selected where they beat NONE and SPLIT: content whose motion
    /// boundary bisects a square node horizontally makes the two
    /// 32x16 halves each uniform-motion (two MVs — NONE cannot code
    /// that; SPLIT costs four leaves), so the tree must carry at
    /// least one HORZ node — and the emitted GOP must round-trip
    /// byte-exact through the spec driver.
    #[test]
    fn r412_p_search_selects_horz_partition_on_band_motion() {
        // Frame k: rows [0, 16) shift right by 4k, rows [16, ..)
        // shift left by 4k — the boundary bisects the top 32x32
        // quadrants at their halfway row.
        let band = |k: usize| -> Yuv420Frame {
            let (w, h) = (64usize, 64usize);
            let mut f = Yuv420Frame::filled(64, 64, 0);
            let tex = |x: i64, y: i64| -> u8 {
                let (xu, yu) = ((x + 512) as usize, (y + 512) as usize);
                ((xu * 7 + yu * 3 + (xu / 8) * (yu / 8) * 5) % 256) as u8
            };
            for i in 0..h {
                // The two bands slide in opposite directions over the
                // same infinite texture (no wraparound seam).
                let s: i64 = if i < 16 {
                    4 * k as i64
                } else {
                    -4 * (k as i64)
                };
                for j in 0..w {
                    f.y[i * w + j] = tex(j as i64 - s, i as i64);
                }
            }
            let (cw, ch) = (w / 2, h / 2);
            for i in 0..ch {
                let s: i64 = if i < 8 { 2 * k as i64 } else { -2 * (k as i64) };
                for j in 0..cw {
                    let x = ((j as i64 - s) + 512) as usize;
                    f.u[i * cw + j] = ((128 + x * 2 + i) % 256) as u8;
                    f.v[i * cw + j] = ((64 + x + i * 2) % 256) as u8;
                }
            }
            f
        };
        let frames: Vec<Yuv420Frame> = (0..3).map(band).collect();
        let base_q_idx = 60u8;

        // Tree inspection on the P-frame search.
        let key = encode_key_frame_yuv420_with_q(&frames[0], base_q_idx).unwrap();
        let reference = GopFrameRecon {
            y: key.recon_y.clone(),
            u: key.recon_u.clone(),
            v: key.recon_v.clone(),
        };
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 64, 64, base_q_idx, 1);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u8; 64 * 64],
            u: vec![0u8; 32 * 32],
            v: vec![0u8; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bd: BlockDecodedMirror::new(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        let mut ictx =
            PSearchCtx::new(&reference, &reference, mi_rows, mi_cols, 64, 64, ip, 1).unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree(0, 0, BLOCK_64X64, &frames[1], &mut recon, &mut ictx).unwrap();
        fn count_rect(node: &SyntaxNode, horz: &mut u32, vert: &mut u32) {
            match node {
                SyntaxNode::Leaf(_) => {}
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count_rect(ch, horz, vert);
                    }
                }
                SyntaxNode::Horz(_) => *horz += 1,
                SyntaxNode::Vert(_) => *vert += 1,
            }
        }
        let (mut horz, mut vert) = (0u32, 0u32);
        count_rect(&tree, &mut horz, &mut vert);
        assert!(
            horz > 0,
            "band-split motion must select at least one PARTITION_HORZ node \
             (got HORZ={horz} VERT={vert})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv420_with_q(&frames, base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), frames.len());
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], enc.recon[idx].y, "frame {idx} luma");
            assert_eq!(f.planes[1], enc.recon[idx].u, "frame {idx} U");
            assert_eq!(f.planes[2], enc.recon[idx].v, "frame {idx} V");
        }
    }

    /// r412 — the per-block reference selection must actually reach
    /// GOLDEN_FRAME where it wins: a flash GOP (frame 2 is unrelated
    /// noise, frame 3 repeats frame 1's content) makes frame 3
    /// predict far better from the frame-before-previous, so its
    /// search tree must carry GOLDEN leaves — and the emitted GOP
    /// must round-trip byte-exact through the spec driver (proving
    /// the §5.9.2 two-slot refresh rotation + `ref_frame_idx` map
    /// against the §7.20 store).
    #[test]
    fn r412_p_search_selects_golden_reference_across_flash() {
        let calm = moving_gradient(64, 64, 0, 0, 5);
        let mut flash = Yuv420Frame::filled(64, 64, 0);
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for v in flash.y.iter_mut() {
            *v = (next() & 0xFF) as u8;
        }
        // frames: KEY(calm), P1(flash), P2(calm again).
        let frames = vec![calm.clone(), flash, calm];
        let base_q_idx = 90u8;

        // Drive the P2 search directly: prev = P1 recon (flash),
        // prevprev = P1's prev = KEY recon (calm).
        let enc = encode_gop_yuv420_with_q(&frames, base_q_idx).unwrap();
        let prev = enc.recon[1].clone();
        let prevprev = enc.recon[0].clone();
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&enc.seq, 64, 64, base_q_idx, 2);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u8; 64 * 64],
            u: vec![0u8; 32 * 32],
            v: vec![0u8; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bd: BlockDecodedMirror::new(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        let mut ictx = PSearchCtx::new(&prev, &prevprev, mi_rows, mi_cols, 64, 64, ip, 2).unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree(0, 0, BLOCK_64X64, &frames[2], &mut recon, &mut ictx).unwrap();
        fn count_refs(node: &SyntaxNode, golden: &mut u32, last: &mut u32) {
            let leafs = |b: &SyntaxBlock, golden: &mut u32, last: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.ref_frame[0] == 4 {
                        *golden += 1;
                    } else {
                        *last += 1;
                    }
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafs(b, golden, last),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count_refs(ch, golden, last);
                    }
                }
                SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
                    for b in blocks.iter() {
                        leafs(b, golden, last);
                    }
                }
            }
        }
        let (mut golden, mut last) = (0u32, 0u32);
        count_refs(&tree, &mut golden, &mut last);
        assert!(
            golden > 0,
            "frame after a flash must select GOLDEN_FRAME leaves \
             (got GOLDEN={golden} LAST={last})"
        );

        // Full-stream conformance through the spec driver.
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), frames.len());
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], enc.recon[idx].y, "frame {idx} luma");
            assert_eq!(f.planes[1], enc.recon[idx].u, "frame {idx} U");
            assert_eq!(f.planes[2], enc.recon[idx].v, "frame {idx} V");
        }
    }

    /// r412 — the COMPOUND_AVERAGE candidates must actually be
    /// selected where the two-reference mean is the distortion
    /// winner: the third frame is constructed as the PIXELWISE
    /// AVERAGE of the first two frames' reconstructions (LAST and
    /// GOLDEN under the two-slot rotation), so a
    /// { LAST, GOLDEN } GLOBAL_GLOBALMV / NEAREST_NEARESTMV leaf
    /// predicts it almost exactly while either single reference
    /// misses by half the inter-frame delta — the search tree must
    /// carry compound leaves, and the emitted GOP must round-trip
    /// byte-exact through the spec driver.
    #[test]
    fn r412_p_search_selects_compound_average_on_blended_content() {
        let f0 = moving_gradient(64, 64, 0, 0, 40);
        let f1 = moving_gradient(64, 64, 0, 0, 140);
        let base_q_idx = 60u8;
        // Two-frame pre-pass to obtain the deterministic recons.
        let pre = encode_gop_yuv420_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();
        let mut f2 = Yuv420Frame::filled(64, 64, 0);
        let avg = |a: &[u8], b: &[u8], out: &mut [u8]| {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = (u16::from(x) + u16::from(y)).div_ceil(2) as u8;
            }
        };
        avg(&pre.recon[0].y, &pre.recon[1].y, &mut f2.y);
        avg(&pre.recon[0].u, &pre.recon[1].u, &mut f2.u);
        avg(&pre.recon[0].v, &pre.recon[1].v, &mut f2.v);

        // Drive the P2 search directly (prev = f1 recon in LAST,
        // prevprev = KEY recon in GOLDEN).
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 2);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u8; 64 * 64],
            u: vec![0u8; 32 * 32],
            v: vec![0u8; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bd: BlockDecodedMirror::new(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        let mut ictx = PSearchCtx::new(
            &pre.recon[1],
            &pre.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            2,
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree(0, 0, BLOCK_64X64, &f2, &mut recon, &mut ictx).unwrap();
        fn count_compound(node: &SyntaxNode, compound: &mut u32, single: &mut u32) {
            let leafc = |b: &SyntaxBlock, compound: &mut u32, single: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.ref_frame[1] > 0 {
                        *compound += 1;
                    } else {
                        *single += 1;
                    }
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, compound, single),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count_compound(ch, compound, single);
                    }
                }
                SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
                    for b in blocks.iter() {
                        leafc(b, compound, single);
                    }
                }
            }
        }
        let (mut compound, mut single) = (0u32, 0u32);
        count_compound(&tree, &mut compound, &mut single);
        assert!(
            compound > 0,
            "average-blend content must select COMPOUND_AVERAGE leaves \
             (got COMPOUND={compound} SINGLE={single})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv420_with_q(&[f0, f1, f2], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], enc.recon[idx].y, "frame {idx} luma");
            assert_eq!(f.planes[1], enc.recon[idx].u, "frame {idx} U");
            assert_eq!(f.planes[2], enc.recon[idx].v, "frame {idx} V");
        }
    }

    /// r412 — [`PSearchCtx::predict_leaf`] must thread the trial
    /// filter into the §7.11.3.4 kernel: an isolated impulse column
    /// interpolated at a half-pel phase produces distinct samples per
    /// filter family.
    #[test]
    fn r412_predict_leaf_threads_interp_filter() {
        use crate::inter_pred::{EIGHTTAP_SHARP, EIGHTTAP_SMOOTH};
        let mut refr = GopFrameRecon {
            y: vec![0u8; 64 * 64],
            u: vec![128u8; 32 * 32],
            v: vec![128u8; 32 * 32],
        };
        for i in 0..64 {
            for j in 0..64 {
                refr.y[i * 64 + j] = if j == 10 { 255 } else { 0 };
            }
        }
        let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
        let mut ictx = PSearchCtx::new(&refr, &refr, 16, 16, 64, 64, ip, 1).unwrap();
        let mut outs = Vec::new();
        for f in [EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP] {
            ictx.predict_leaf(2, 2, BLOCK_8X8, [1, -1], MODE_NEWMV, [[0, 4], [0, 0]], f)
                .unwrap();
            let mut v = Vec::new();
            for i in 8..16 {
                for j in 8..16 {
                    v.push(ictx.scratch[0][i * 64 + j]);
                }
            }
            outs.push(v);
        }
        assert_ne!(
            outs[0], outs[1],
            "EIGHTTAP vs SMOOTH must differ at half-pel"
        );
        assert_ne!(
            outs[0], outs[2],
            "EIGHTTAP vs SHARP must differ at half-pel"
        );
    }

    /// r412 — the driver mirror must agree with the WRITE mirror leaf
    /// for leaf: a full multi-superblock lossy GOP encode (which
    /// re-runs §7.10.2 per leaf inside `write_partition_tree_syntax`
    /// and hard-errors on any committed non-NEWMV MV that differs
    /// from the write-side derivation) must succeed and self-decode
    /// byte-exact through the spec driver.
    #[test]
    fn r412_gop_with_stack_modes_round_trips_through_spec_driver() {
        let frames: Vec<Yuv420Frame> = (0..4)
            .map(|k| moving_gradient(96, 80, 2 * k, 4 * k, 31))
            .collect();
        let enc = encode_gop_yuv420_with_q(&frames, 80).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), frames.len());
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], enc.recon[idx].y, "frame {idx} luma");
            assert_eq!(f.planes[1], enc.recon[idx].u, "frame {idx} U");
            assert_eq!(f.planes[2], enc.recon[idx].v, "frame {idx} V");
        }
    }
}
