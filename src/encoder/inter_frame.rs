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
//! ## Scope (r411, extended r412; every §6.4.1 format pairing r427)
//!
//! * Any §6.4.1 (bit depth, chroma format) pairing ([`YuvFrame`] —
//!   8 / 10 / 12-bit × 4:2:0 / 4:2:2 / 4:4:4 / monochrome; the
//!   historical [`Yuv420Frame`] entries widen into the same core),
//!   dimensions per the KEY-frame rules (multiples of 8 in
//!   `[8, KEY_FRAME_MAX_DIM]`). Under 4:2:2 the §5.11.38
//!   admissibility rule bars the tall partition shapes from the RD
//!   ladder.
//! * P-frame header: `error_resilient_mode = 1` (forcing
//!   `primary_ref_frame = PRIMARY_REF_NONE` — per-frame default CDFs),
//!   the r412 two-slot reference rotation (frame `k` refreshes slot
//!   `(k - 1) & 1`; LAST reads the previous frame's slot, GOLDEN the
//!   frame before it), identity §5.9.24 global motion, SWITCHABLE
//!   frame filter (per-leaf §5.11.x `interp_filter` selection),
//!   `force_integer_mv = 0` / `allow_high_precision_mv = 0`; r413
//!   adds 7-bit order hints (true §5.9.2 `ref_order_hint[]`
//!   surfacing), §5.9.22 skip-mode from `p_index = 2` on, and
//!   optional §5.9.14 `SEG_LVL_ALT_Q` spatial segmentation
//!   ([`encode_gop_yuv420_with_q_seg`]); `TxMode = ONLY_4X4` on the
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
    MODE_NEWMV, MODE_NEW_NEWMV, MOTION_MODE_OBMC, MOTION_MODE_SIMPLE, MOTION_MODE_WARPED_CAUSAL,
    NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PARTITION_HORZ, PARTITION_VERT, SPLIT_TX_SIZE,
    SWITCHABLE, TX_4X4, TX_HEIGHT, TX_SIZE_SQR_UP, TX_WIDTH,
};
use crate::encoder::block_mode_info::assign_mv_pred_mv;
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::key_frame::{
    encode_leaf_sq, lambda_for, leaf_rate, region_distortion, region_distortion_wh, residual_tx,
    restore_region, save_region, save_region_wh, tu_bd_stamp, BlockDecodedMirror, ReconState,
    RegionSnapshot,
};
use crate::encoder::obu::{build_temporal_unit, ObuFrame};
use crate::encoder::partition_tree::{
    write_partition_tree_syntax, PartitionSyntaxWriter, SyntaxBlock, SyntaxFrameParams,
    SyntaxInterBlock, SyntaxInterFrameParams, SyntaxNode, VarTxSyntaxTree,
};
use crate::encoder::pixel_driver_dyn::{
    build_intra_only_yuv420_8bit_fh_with_q, sb_grid_origins, Yuv420Frame,
};
use crate::encoder::rate_twin::{score256, RateModel, RateTwin, TuCtx, TuFork};
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::tile_group_obu::{write_tile_group_obu, TileGroupObu, TilePayload};
use crate::encoder::yuv_frame::YuvFrame;
use crate::frame_header::{FrameHeader, FrameType, InterFrameRefs, PRIMARY_REF_NONE};
use crate::inter_pred::{
    reconstruct_inter_leaf_at, InterModeInfoGrid, PlaneReconContext, RefFrameStoreEntry,
};
use crate::obu::ObuType;
use crate::sequence_header::SequenceHeader;
use crate::uncompressed_header_tail::{
    InterpolationFilter, TxMode, REFS_PER_FRAME, TOTAL_REFS_PER_FRAME,
};
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

/// r427 — general-format sibling of [`GopFrameRecon`]: `u16` planes
/// at the stream's bit depth, chroma at the subsampled extent (empty
/// on the monochrome arm).
#[derive(Debug, Clone)]
pub struct GopFrameReconYuv {
    /// Luma plane.
    pub y: Vec<u16>,
    /// Cb plane.
    pub u: Vec<u16>,
    /// Cr plane.
    pub v: Vec<u16>,
}

/// r427 — result of [`encode_gop_yuv_with_q`]: the general-format
/// sibling of [`EncodedGop`].
#[derive(Debug, Clone)]
pub struct EncodedGopYuv {
    /// Complete IVF v0 file (header + one record per frame).
    pub ivf_bytes: Vec<u8>,
    /// The bare §7.5 temporal units, one per frame.
    pub temporal_units: Vec<Vec<u8>>,
    /// Per-frame encoder reconstruction — the decoded output equals
    /// these sample-for-sample (and the inputs too at `base_q_idx ==
    /// 0`).
    pub recon: Vec<GopFrameReconYuv>,
    /// The emitted sequence header descriptor.
    pub seq: SequenceHeader,
}

/// r427 (hidden) — [`EncodedGopYuv`] plus the per-P-frame election
/// outcomes: the general-format sibling of [`TunedGop`].
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TunedGopYuv {
    pub gop: EncodedGopYuv,
    pub seg_temporal_updates: Vec<bool>,
    pub p_segment_maps: Vec<Vec<i32>>,
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

/// r423 (hidden) — [`EncodedGop`] plus the per-P-frame §5.9.14
/// `segmentation_temporal_update` election outcomes (index 0 is the
/// first P-frame; empty on unsegmented GOPs). Returned by the
/// tuned measurement entry point only — [`EncodedGop`] itself stays
/// field-stable for the public constructors.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TunedGop {
    pub gop: EncodedGop,
    pub seg_temporal_updates: Vec<bool>,
    /// r426 — each P-frame's committed §5.11.20 `SegmentIds[]` grid
    /// (row-major `mi_rows × mi_cols`; index 0 is the first P-frame;
    /// empty on unsegmented GOPs). Lets the measurement harnesses
    /// locate the per-segment-lossless blocks without re-deriving the
    /// election.
    pub p_segment_maps: Vec<Vec<i32>>,
}

/// GOP length bound (KEY + P-frames).
pub const GOP_MAX_FRAMES: usize = 64;

/// r427 — narrow a general-format GOP result back to the 8-bit
/// [`EncodedGop`] shape (caller guarantees an 8-bit stream — the
/// legacy 4:2:0 entry points).
pub(crate) fn narrow_gop_8bit(g: EncodedGopYuv) -> EncodedGop {
    let narrow = |p: Vec<u16>| p.into_iter().map(|s| s as u8).collect::<Vec<u8>>();
    EncodedGop {
        ivf_bytes: g.ivf_bytes,
        temporal_units: g.temporal_units,
        recon: g
            .recon
            .into_iter()
            .map(|r| GopFrameRecon {
                y: narrow(r.y),
                u: narrow(r.u),
                v: narrow(r.v),
            })
            .collect(),
        seq: g.seq,
    }
}

/// r413 — one §7.20 slot's stored motion-field payload on the encode
/// side (the §7.19 `SavedMvs` / `SavedRefFrames` filter output plus
/// `SavedOrderHints`), mirroring the decode driver's per-slot store so
/// the §7.9 projection runs on identical inputs.
#[derive(Debug, Clone)]
pub(crate) struct SavedMotionField {
    mf_mvs: Vec<i16>,
    mf_ref_frames: Vec<i8>,
    saved_order_hints: [i32; 8],
    mi_rows: u32,
    mi_cols: u32,
    frame_is_intra: bool,
}

impl SavedMotionField {
    /// The KEY frame's slot payload (`FrameIsIntra = 1` — §7.9.2
    /// skips intra sources, so the grids stay empty).
    pub(crate) fn intra(mi_rows: u32, mi_cols: u32) -> Self {
        SavedMotionField {
            mf_mvs: Vec::new(),
            mf_ref_frames: Vec::new(),
            saved_order_hints: [0; 8],
            mi_rows,
            mi_cols,
            frame_is_intra: true,
        }
    }
}

/// r423 — one §7.20 slot's stored cross-frame decode state on the
/// encode side, mirroring the §5.9.2 primary-reference loads field by
/// field:
///
/// * `cdfs` — the §8.4 frame-end CDF table `save_cdfs( i )` deposits
///   (the `context_update_tile_id` tile's adapted state;
///   `disable_frame_end_update_cdf = 0` on every header this encoder
///   emits). A frame electing this slot as its primary reference
///   starts from a clone with the per-row symbol counts zeroed
///   (§6.8.21 `load_cdfs`).
/// * `segment_ids` — `SavedSegmentIds[ i ][ row ][ col ]` (row-major
///   over `mi_rows × mi_cols`), the §5.9.2
///   `load_previous_segment_ids()` source feeding the §5.11.21
///   `get_segment_id()` prediction.
/// * `gm_params` — `SavedGmParams[ i ]`, the §7.21 `load_previous()`
///   `PrevGmParams` source the §5.9.24 subexp recentering codes
///   against.
#[derive(Debug, Clone)]
pub(crate) struct RefSlotCarry {
    pub(crate) cdfs: Box<TileCdfContext>,
    pub(crate) segment_ids: Vec<i32>,
    pub(crate) mi_rows: u32,
    pub(crate) mi_cols: u32,
    pub(crate) gm_params: [[i32; 6]; crate::uncompressed_header_tail::TOTAL_REFS_PER_FRAME],
}

/// r424 — one INTER frame's auxiliary election outcomes, returned by
/// [`encode_inter_frame_generic`] alongside the OBU: the §5.9.14
/// `segmentation_temporal_update` arm the exact-bits election kept
/// (r423) and the §5.9.2 `primary_ref_frame` ordinal the exact-bytes
/// election committed (r424; equals the config's ordinal when no
/// [`InterFrameConfig::alt_primaries`] were offered).
#[derive(Debug, Clone, Copy)]
pub(crate) struct InterFrameAux {
    pub(crate) seg_temporal: bool,
    pub(crate) primary_ref: u8,
}

/// Integer-pel motion-search radius (luma samples per axis).
const SEARCH_RANGE: i32 = 16;

/// r415 — one leaf's committed §5.11.29 compound selection for the
/// prediction driver ([`PSearchCtx::predict_leaf`] stamps it into the
/// §5.11.5 side-data grids the decoder's leaf driver reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CompoundSel {
    ctype: u8,
    wedge_index: u8,
    wedge_sign: u8,
    mask_type: u8,
}

impl CompoundSel {
    /// The §5.11.29 line-1/2 pre-set shape (every single-reference
    /// leaf and every compound leaf whose masked arms lost or are
    /// gated shut).
    const AVERAGE: CompoundSel = CompoundSel {
        ctype: crate::inter_pred::COMPOUND_AVERAGE,
        wedge_index: 0,
        wedge_sign: 0,
        mask_type: 0,
    };
}

/// r419 — one leaf's committed §5.11.27 motion-mode selection for the
/// prediction driver ([`PSearchCtx::predict_leaf`] stamps the ordinal
/// into the `MotionModes[]` grid and — on the warp arm — the
/// §7.11.3.8 fit into the per-cell `LocalWarpParams` slice the
/// decoder's leaf driver reads).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MotionSel {
    /// §5.11.27 `SIMPLE` — the translational §7.11.3.1 path.
    Simple,
    /// §5.11.27 `OBMC` — the leaf's own prediction plus the
    /// §7.11.3.9-10 above/left overlap blend, resolved by the leaf
    /// driver from the committed neighbour grids.
    Obmc,
    /// §5.11.27 `WARPED_CAUSAL` — the §7.11.3.5 warp with the carried
    /// §7.11.3.8 least-squares fit (already `setup_shear`-validated
    /// by the caller; an invalid fit is never committed).
    Warp([i32; 6]),
}

impl MotionSel {
    /// The committed §5.11.27 ordinal.
    fn ordinal(self) -> u8 {
        match self {
            MotionSel::Simple => MOTION_MODE_SIMPLE,
            MotionSel::Obmc => MOTION_MODE_OBMC,
            MotionSel::Warp(_) => MOTION_MODE_WARPED_CAUSAL,
        }
    }
}

/// r419 — one driver-grid cell's saved state (see
/// [`PSearchCtx::snapshot_grids_rect`]).
#[derive(Debug, Clone, Copy)]
struct DriverGridCell {
    mi_size: usize,
    is_inter: u8,
    ref_frames: [i8; 2],
    mvs: [i16; 4],
    interp_filters: [u8; 2],
    motion_mode: u8,
    y_mode: u8,
    compound_type: u8,
    wedge_index: u8,
    wedge_sign: u8,
    mask_type: u8,
    interintra_mode: u8,
    wedge_interintra: u8,
    interintra_wedge_index: u8,
    local_warp: [i32; 6],
    local_warp_valid: u8,
}

/// r419 — a rect snapshot of the §5.11.5 driver grids (row-major
/// cells over `[mi_row, r1) × [mi_col, c1)`).
struct DriverGridSnapshot {
    mi_row: u32,
    mi_col: u32,
    r1: u32,
    c1: u32,
    cells: Vec<DriverGridCell>,
}

/// r417 — one §5.11.28 inter-intra selection for
/// [`PSearchCtx::predict_leaf`]: the §7.11.5-translated intra half is
/// predicted into the search scratch (reading reconstructed
/// neighbours from `neigh` through the decode walker's own §7.11.2
/// core), then the decoder's §7.11.3.14 mask blend combines it with
/// the single-reference inter prediction.
#[derive(Clone, Copy)]
struct InterIntraTrial<'a> {
    /// §6.10.27 `interintra_mode` ordinal
    /// (`II_DC_PRED..=II_SMOOTH_PRED`).
    mode: u8,
    /// `Some(wedge_index)` selects the `wedge_interintra == 1` sub-arm
    /// (§7.11.3.11 luma-grid wedge mask, sign fixed `0` per §5.11.28);
    /// `None` the §7.11.3.13 smooth intra-variant mask.
    wedge: Option<u8>,
    /// The committed running reconstruction — the §7.11.2 neighbour
    /// source (the decode walker's intra half reads its own
    /// `CurrFrame[ plane ]`, whose encoder twin is this recon state,
    /// sample-exact by the module's induction argument).
    neigh: &'a ReconState,
}

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
    encode_gop_yuv420_with_q_seg(frames, base_q_idx, &[])
}

/// r413 — GOP encode with §5.9.14 `SEG_LVL_ALT_Q` segmentation on the
/// P-frames: `alt_q[ s ]` is segment `s`'s signed quantiser delta
/// (`get_qindex( s ) = Clip3( 0, 255, base_q_idx + alt_q[ s ] )`).
/// The encoder assigns segments per leaf by luma activity (flat
/// blocks take the higher-index segments), codes the spatial
/// §5.11.19/§5.11.20 segment map (skip leaves inherit the §5.11.20
/// `pred` cascade with no bits, per spec), and quantises each leaf's
/// residual at its segment's q-index. An empty `alt_q` disables
/// segmentation (the pre-r413 configuration). The KEY frame stays
/// unsegmented (its `segmentation_enabled = 0` header is unchanged;
/// every P-frame header re-codes the full feature table under the
/// `PRIMARY_REF_NONE` forced `update_data = 1`).
///
/// r426 — **per-segment lossless mixing**: a segment whose §5.9.14
/// `get_qindex( s ) = Clip3( 0, 255, base_q_idx + alt_q[ s ] )`
/// reaches 0 takes the full §5.9.2 `LosslessArray[ s ]` leaf
/// semantics inside the otherwise-lossy frame — its blocks code the
/// `TX_4X4`-only §5.11.34 row-major TU grid through the §7.13.2.10
/// WHT chain (bit-exact residual round-trip), read no §5.11.15 /
/// §5.11.16 tx-size symbols and no §5.11.47 tx-type symbol, exactly
/// as the decode side derives per block from
/// `Lossless = LosslessArray[ segment_id ]`. Blocks in the other
/// segments stay on the lossy ladder. `CodedLossless` stays 0 (the
/// frame still codes `TX_MODE_SELECT`, and segment 0 rides
/// `base_q_idx > 0`), so the mix is purely per-block.
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] on the
/// [`encode_gop_yuv420_with_q`] conditions, plus:
///
/// * `alt_q.len() > MAX_SEGMENTS = 8` or `alt_q.len() == 1`;
/// * `alt_q[ 0 ] != 0` (intra leaves code segment 0 at the frame
///   quantiser — the residual chain requires the identity delta);
/// * `base_q_idx == 0` with a non-empty `alt_q` (a `base_q_idx = 0`
///   GOP is the `CodedLossless` configuration — segment deltas would
///   make segment 0 the only lossless segment while the header
///   builder still derives the ONLY_4X4 frame shape);
/// * any `alt_q[ s ]` outside the §5.9.14 `su(1+8)` range
///   `-255..=255`, or any segment's `base_q_idx + alt_q[ s ] > 255`
///   (the §7.12.2 clamp makes over-255 deltas unreachable dead
///   weight; deltas that clamp BELOW zero are the per-segment
///   lossless election and are accepted).
pub fn encode_gop_yuv420_with_q_seg(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    alt_q: &[i16],
) -> Result<EncodedGop, Error> {
    encode_gop_yuv420_with_q_seg_rate_model(frames, base_q_idx, alt_q, RateModel::Twin)
}

/// r421 — [`encode_gop_yuv420_with_q_seg`] with an explicit
/// [`RateModel`], kept public (hidden) so the sweep harnesses can A/B
/// the twin-priced elections against the pre-r421 heuristic baseline.
#[doc(hidden)]
pub fn encode_gop_yuv420_with_q_seg_rate_model(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    alt_q: &[i16],
    model: RateModel,
) -> Result<EncodedGop, Error> {
    encode_gop_yuv420_with_q_seg_rate_model_gm(frames, base_q_idx, alt_q, model, true)
}

/// r422 — [`encode_gop_yuv420_with_q_seg_rate_model`] with an explicit
/// global-motion switch, kept public (hidden) so the measurement
/// harness can A/B the §5.9.24 election against the identity-only
/// baseline on identical inputs. Production entry points always
/// enable it.
#[doc(hidden)]
pub fn encode_gop_yuv420_with_q_seg_rate_model_gm(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    alt_q: &[i16],
    model: RateModel,
    global_motion: bool,
) -> Result<EncodedGop, Error> {
    encode_gop_yuv420_with_q_seg_tuned(
        frames,
        base_q_idx,
        alt_q,
        GopTuning {
            model,
            global_motion,
            ..GopTuning::default()
        },
    )
    .map(|t| t.gop)
}

/// r423 — the P-GOP encoder's tuning switches, kept public (hidden)
/// so the measurement harnesses can A/B each cross-frame feature
/// against its baseline on identical inputs. Production entry points
/// always use [`GopTuning::default`] (everything on).
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct GopTuning {
    /// RD-election rate model (r421).
    pub model: RateModel,
    /// §5.9.24 global-motion election (r422).
    pub global_motion: bool,
    /// r423 — §5.9.2 `primary_ref_frame` election: P-frames inherit
    /// the previous frame's frame-end CDFs (§6.8.21 `load_cdfs`),
    /// segment map (`load_previous_segment_ids`) and gm reference
    /// state (`load_previous`) instead of per-frame defaults.
    pub primary_ref: bool,
    /// r423 — §5.9.14 / §5.11.19 temporal segment-map coding: allow
    /// the per-frame exact-bits election between the temporal
    /// (`seg_id_predicted`) and spatial segment-id arms. `false`
    /// forces spatial coding on every frame (the A/B baseline).
    /// Inert while `primary_ref` is off (the temporal arm requires
    /// real `PrevSegmentIds` state).
    pub temporal_seg: bool,
}

impl Default for GopTuning {
    fn default() -> Self {
        GopTuning {
            model: RateModel::Twin,
            global_motion: true,
            primary_ref: true,
            temporal_seg: true,
        }
    }
}

/// r426 — one caller-supplied exactness-demand rectangle in LUMA
/// pixel coordinates (top-left `(x, y)`, `width × height`; clipped to
/// the frame). Every block overlapping the rectangle is coded on the
/// lossless segment (§5.9.14 `SEG_LVL_ALT_Q` to qindex 0 — the
/// §5.9.2 `LosslessArray[]` TX_4X4/WHT leaf semantics), so the
/// decoded region equals the input pixel-for-pixel on every frame
/// the encoder segments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LosslessRegion {
    /// Left edge (luma samples).
    pub x: u32,
    /// Top edge (luma samples).
    pub y: u32,
    /// Width (luma samples; > 0).
    pub width: u32,
    /// Height (luma samples; > 0).
    pub height: u32,
}

/// r426 — build the row-major mi-cell exactness mask for the demand
/// rectangles: a 4×4 luma cell is demanded when it intersects any
/// (frame-clipped) rectangle, then the demand dilates to the full
/// 2×2-mi group so a sub-8×8 group's single §5.11.34 `HasChroma`
/// coder (the SE cell's leaf) and the 4:2:0 chroma cositing are
/// covered by the same lossless segment.
fn build_exact_mask(
    regions: &[LosslessRegion],
    width: u32,
    height: u32,
    mi_rows: u32,
    mi_cols: u32,
) -> Vec<bool> {
    let mut mask = vec![false; (mi_rows as usize) * (mi_cols as usize)];
    for reg in regions {
        let x0 = reg.x.min(width);
        let y0 = reg.y.min(height);
        let x1 = reg.x.saturating_add(reg.width).min(width);
        let y1 = reg.y.saturating_add(reg.height).min(height);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        let (c0, r0) = (x0 / 4, y0 / 4);
        let (c1, r1) = (x1.div_ceil(4).min(mi_cols), y1.div_ceil(4).min(mi_rows));
        for r in r0..r1 {
            for c in c0..c1 {
                mask[(r * mi_cols + c) as usize] = true;
            }
        }
    }
    // 2×2-mi group dilation.
    let mut out = mask.clone();
    for r in 0..mi_rows {
        for c in 0..mi_cols {
            if mask[(r * mi_cols + c) as usize] {
                let (gr, gc) = (r & !1, c & !1);
                for rr in gr..(gr + 2).min(mi_rows) {
                    for cc in gc..(gc + 2).min(mi_cols) {
                        out[(rr * mi_cols + cc) as usize] = true;
                    }
                }
            }
        }
    }
    out
}

/// r426 — the pixel-exact-region GOP entry point: encode at
/// `base_q_idx` with the demand `regions` coded PIXEL-EXACT through
/// the per-segment lossless machinery (`alt_q = [ 0, -base_q_idx ]`
/// — segment 1's §7.12.2 qindex is exactly 0), optionally electing
/// additional synthetic leaves onto the lossless segment by twin
/// bits + distortion (`auto_detect`). The KEY frame is unsegmented
/// (r413 shape) — the exactness contract covers the P-frames; a
/// segmented-KEY arm is the follow-up.
pub fn encode_gop_yuv420_with_q_lossless_regions(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    regions: &[LosslessRegion],
    auto_detect: bool,
) -> Result<EncodedGop, Error> {
    if base_q_idx == 0 {
        // Already CodedLossless everywhere — the demand is vacuous.
        return encode_gop_yuv420_with_q(frames, 0);
    }
    let alt_q = [0i16, -i16::from(base_q_idx)];
    encode_gop_yuv420_with_q_seg_lossless_tuned(
        frames,
        base_q_idx,
        &alt_q,
        regions,
        auto_detect,
        GopTuning::default(),
    )
    .map(|t| t.gop)
}

/// r423 — [`encode_gop_yuv420_with_q_seg`] with explicit
/// [`GopTuning`] switches (the measurement-harness entry point).
#[doc(hidden)]
pub fn encode_gop_yuv420_with_q_seg_tuned(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    alt_q: &[i16],
    tuning: GopTuning,
) -> Result<TunedGop, Error> {
    encode_gop_yuv420_with_q_seg_lossless_tuned(frames, base_q_idx, alt_q, &[], false, tuning)
}

/// r426 — the full-surface GOP entry: segmentation deltas, exactness
/// demand rectangles, the auto-lossless election switch and the
/// [`GopTuning`] measurement switches. `regions`/`auto_detect`
/// require an `alt_q` table carrying a lossless segment (a delta
/// clamping the §7.12.2 qindex to 0).
#[doc(hidden)]
pub fn encode_gop_yuv420_with_q_seg_lossless_tuned(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    alt_q: &[i16],
    regions: &[LosslessRegion],
    auto_detect: bool,
    tuning: GopTuning,
) -> Result<TunedGop, Error> {
    encode_gop_yuv420_with_q_seg_extras_tuned(
        frames,
        base_q_idx,
        alt_q,
        regions,
        auto_detect,
        None,
        tuning,
    )
}

/// r426 — the full-surface GOP entry (ladder items 6 + 8):
/// segmentation deltas, exactness-demand rectangles, the
/// auto-lossless election, the three §5.9.14 inter-override features
/// and the [`GopTuning`] measurement switches. `extras` requires a
/// non-empty `alt_q` (the segmentation carrier), a feature-free
/// segment 0 (the encoder's intra/skip-mode fallback segment), no
/// SEG_LVL_SKIP on a lossless segment (a skip block codes no
/// residual — it cannot honour a qindex-0 exactness contract), and
/// SEG_LVL_REF_FRAME data within the P-GOP reference ladder.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn encode_gop_yuv420_with_q_seg_extras_tuned(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    alt_q: &[i16],
    regions: &[LosslessRegion],
    auto_detect: bool,
    extras: Option<&SegExtras>,
    tuning: GopTuning,
) -> Result<TunedGop, Error> {
    let wide: Vec<YuvFrame> = frames.iter().map(YuvFrame::from_yuv420_8bit).collect();
    let t = encode_gop_yuv_seg_extras_tuned(
        &wide,
        base_q_idx,
        alt_q,
        regions,
        auto_detect,
        extras,
        tuning,
    )?;
    Ok(TunedGop {
        gop: narrow_gop_8bit(t.gop),
        seg_temporal_updates: t.seg_temporal_updates,
        p_segment_maps: t.p_segment_maps,
    })
}

/// r427 — general-format lossless GOP encode: see
/// [`encode_gop_yuv_with_q`].
pub fn encode_gop_yuv(frames: &[YuvFrame]) -> Result<EncodedGopYuv, Error> {
    encode_gop_yuv_with_q(frames, 0)
}

/// r427 — encode a KEY + P GOP at any conformant (bit depth, chroma
/// format) pairing: the general-format sibling of
/// [`encode_gop_yuv420_with_q`] (whose 8-bit 4:2:0 arm now routes
/// through this driver). Every frame must carry the same dimensions,
/// bit depth and chroma format.
///
/// ## Errors
///
/// Same conditions as [`encode_gop_yuv420_with_q`] plus per-frame
/// [`YuvFrame::validate`] failures and cross-frame format mismatches
/// — [`Error::PartitionWalkOutOfRange`].
pub fn encode_gop_yuv_with_q(frames: &[YuvFrame], base_q_idx: u8) -> Result<EncodedGopYuv, Error> {
    encode_gop_yuv_seg_extras_tuned(
        frames,
        base_q_idx,
        &[],
        &[],
        false,
        None,
        GopTuning::default(),
    )
    .map(|t| t.gop)
}

/// r427 — the general-format GOP core: every entry point above
/// funnels here.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn encode_gop_yuv_seg_extras_tuned(
    frames: &[YuvFrame],
    base_q_idx: u8,
    alt_q: &[i16],
    regions: &[LosslessRegion],
    auto_detect: bool,
    extras: Option<&SegExtras>,
    tuning: GopTuning,
) -> Result<TunedGopYuv, Error> {
    if let Some(x) = extras {
        let ll = seg_lossless_array(base_q_idx, alt_q);
        if alt_q.is_empty()
            || x.ref_frame[0].is_some()
            || x.skip[0]
            || x.globalmv[0]
            || x.ref_frame.iter().flatten().any(|rf| !(1..=7).contains(rf))
            || (0..crate::uncompressed_header_tail::MAX_SEGMENTS).any(|s| x.skip[s] && ll[s])
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }
    let model = tuning.model;
    let global_motion = tuning.global_motion;
    if frames.is_empty() || frames.len() > GOP_MAX_FRAMES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if !alt_q.is_empty() {
        if alt_q.len() == 1
            || alt_q.len() > crate::uncompressed_header_tail::MAX_SEGMENTS
            || alt_q[0] != 0
            || base_q_idx == 0
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        for &d in alt_q {
            // §5.9.14 su(1+8) representability; §7.12.2 clamps the
            // derived qindex at 0 (the r426 per-segment lossless
            // election) but over-255 sums are rejected as dead weight.
            if !(-255..=255).contains(&d) || i32::from(base_q_idx) + i32::from(d) > 255 {
                return Err(Error::PartitionWalkOutOfRange);
            }
        }
    }
    let (width, height) = (frames[0].width, frames[0].height);
    let (bit_depth, format) = (frames[0].bit_depth, frames[0].format);
    if frames.iter().any(|f| {
        f.width != width || f.height != height || f.bit_depth != bit_depth || f.format != format
    }) {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // Frame 0: the r410 conformance-grade KEY-frame encoder (which
    // also validates the dimension rules). r423: also take the KEY
    // frame's §7.20 slot payload — its `allFrames` refresh seeds
    // every slot of the carry store below. r426: with demand
    // rectangles the KEY frame is SEGMENTED too (same delta table,
    // §5.11.8 per-block segment ids) so the demanded region is
    // pixel-exact from frame 0, not just on the P-frames.
    let key_mask: Option<Vec<bool>> = (!regions.is_empty()).then(|| {
        // §5.9.5: dimensions are validated multiples of 8, so
        // `MiCols = width / 4` / `MiRows = height / 4` exactly (the
        // same grid the fh builder below derives).
        build_exact_mask(regions, width, height, height / 4, width / 4)
    });
    let (key, key_carry) = crate::encoder::key_frame::encode_key_frame_yuv_seg_carry(
        &frames[0],
        base_q_idx,
        model,
        if key_mask.is_some() { alt_q } else { &[] },
        key_mask.as_deref(),
    )?;
    let seq = key.seq.clone();
    let mut temporal_units = vec![key.temporal_unit_bytes.clone()];
    let mut recon = vec![GopFrameReconYuv {
        y: key.recon_y,
        u: key.recon_u,
        v: key.recon_v,
    }];

    // r413 — encoder-side §7.20 motion-field store: the KEY frame's
    // `allFrames` refresh fills every slot with an intra payload;
    // each P-frame then refreshes its rotation slot with the §7.19
    // filter of its committed Mvs[] / RefFrames[] grids.
    let key_mi = {
        let fh0 = build_intra_only_yuv420_8bit_fh_with_q(&seq, width, height, base_q_idx);
        let fs0 = fh0.frame_size.expect("KEY builder always sizes");
        (fs0.mi_rows, fs0.mi_cols)
    };
    // r426 — exactness-demand mask + auto-lossless election arming:
    // both require a lossless segment in the table.
    if (!regions.is_empty() || auto_detect)
        && !seg_lossless_array(base_q_idx, alt_q)[..alt_q.len()]
            .iter()
            .any(|&l| l)
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let exact_mask: Option<Vec<bool>> =
        (!regions.is_empty()).then(|| build_exact_mask(regions, width, height, key_mi.0, key_mi.1));
    let mut mf_store: [SavedMotionField; 8] =
        core::array::from_fn(|_| SavedMotionField::intra(key_mi.0, key_mi.1));
    // r423 — encoder-side §7.20 cross-frame carry store (CDFs /
    // segment map / gm params per slot). The KEY frame's `allFrames`
    // refresh fills every slot with its payload; each P-frame then
    // refreshes its rotation slot, exactly like `mf_store`.
    let key_carry = std::rc::Rc::new(key_carry);
    let mut carry_store: [std::rc::Rc<RefSlotCarry>; 8] =
        core::array::from_fn(|_| key_carry.clone());
    let mut seg_temporal_updates: Vec<bool> = Vec::new();
    let mut p_segment_maps: Vec<Vec<i32>> = Vec::new();

    for (k, input) in frames[1..].iter().enumerate() {
        let p_index = (k + 1) as u32;
        let prev = recon.last().expect("at least the KEY recon");
        let prevprev = &recon[recon.len().saturating_sub(2)];
        // r423 — the primary reference is LAST (ordinal 0): slot
        // `p_index & 1`, holding frame `p_index - 1`.
        let primary = if tuning.primary_ref {
            Some(carry_store[(p_index & 1) as usize].clone())
        } else {
            None
        };
        let (tu, rc, saved_mf, carry, aux) = encode_p_frame_yuv(
            input,
            prev,
            prevprev,
            &seq,
            base_q_idx,
            p_index,
            alt_q,
            &mf_store,
            model,
            global_motion,
            primary.as_deref(),
            tuning.temporal_seg,
            exact_mask.as_deref(),
            auto_detect,
            extras,
        )?;
        temporal_units.push(tu);
        recon.push(rc);
        seg_temporal_updates.push(aux.seg_temporal);
        if !alt_q.is_empty() {
            p_segment_maps.push(carry.segment_ids.clone());
        }
        // §7.20: this frame refreshed slot `(p_index - 1) & 1`.
        mf_store[((p_index - 1) & 1) as usize] = saved_mf;
        carry_store[((p_index - 1) & 1) as usize] = std::rc::Rc::new(carry);
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

    Ok(TunedGopYuv {
        gop: EncodedGopYuv {
            ivf_bytes,
            temporal_units,
            recon,
            seq,
        },
        seg_temporal_updates,
        p_segment_maps,
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
/// Output order hints of the two rotated references at P-frame
/// `p_index = k` (1-based): LAST holds frame `k - 1`, GOLDEN frame
/// `k - 2`, clamped at the KEY frame (hint 0).
fn gop_ref_hints(p_index: u32) -> (u32, u32) {
    (p_index - 1, p_index.saturating_sub(2))
}

/// §5.9.2-derived [`FrameInterOrderHints`] for P-frame `p_index` —
/// `OrderHints[ ref ]` per raw `RefFrame` value (`LAST_FRAME..=
/// ALTREF_FRAME` read the LAST slot except `GOLDEN_FRAME`).
#[cfg(test)]
fn gop_order_hints(p_index: u32, order_hint_bits: u8) -> crate::cdf::FrameInterOrderHints {
    let (last_hint, golden_hint) = gop_ref_hints(p_index);
    let mut by_ref = [0i32; crate::uncompressed_header_tail::ALTREF_FRAME + 1];
    for (r, hint) in by_ref.iter_mut().enumerate().skip(1) {
        // GOLDEN_FRAME = 4; every other reference maps to the LAST slot.
        *hint = if r == 4 {
            golden_hint as i32
        } else {
            last_hint as i32
        };
    }
    crate::cdf::FrameInterOrderHints {
        order_hint_bits: u32::from(order_hint_bits),
        current_order_hint: p_index as i32,
        order_hints_by_ref: by_ref,
    }
}

/// r415 — one INTER frame's reference/role configuration for the
/// generic frame encoder ([`encode_inter_frame_generic`]): the §5.9.2
/// header fields that vary per pyramid role, the §7.20 slot state,
/// and the RD ladder.
pub(crate) struct InterFrameConfig<'a> {
    /// §5.9.2 `order_hint` — the frame's DISPLAY position.
    pub order_hint: u32,
    /// §5.9.2 `show_frame` (`false` = decoded-not-shown; a later
    /// `show_existing_frame` header displays it).
    pub show_frame: bool,
    /// §5.9.2 `refresh_frame_flags`.
    pub refresh_frame_flags: u8,
    /// §5.9.2 `ref_frame_idx[ 0..7 ]` — per-reference §7.20 slot.
    pub ref_frame_idx: [u8; 7],
    /// The TRUE per-slot stored `RefOrderHint[ i ]` state at this
    /// frame's header (drives the §5.9.22 twin + §7.8 sign bias).
    pub slot_hints: [u32; 8],
    /// RD ladder — see [`PSearchCtx::single_refs`] /
    /// [`PSearchCtx::compound_pairs`].
    pub single_refs: Vec<i8>,
    pub compound_pairs: Vec<[i8; 2]>,
    /// The distinct reference reconstructions + the §7.20 slot map
    /// onto them — see [`PSearchCtx::slot_to_plane`].
    pub refs: Vec<&'a GopFrameReconYuv>,
    pub slot_to_plane: [usize; 8],
    /// r423 — §5.9.2 `primary_ref_frame`: the ordinal into
    /// `ref_frame_idx[]` whose slot's saved state (CDFs, segment map,
    /// gm params) this frame inherits, or [`PRIMARY_REF_NONE`] for
    /// per-frame default state.
    pub primary_ref_frame: u8,
    /// r423 — the §7.20 stored state of
    /// `ref_frame_idx[ primary_ref_frame ]`. MUST be `Some` whenever
    /// `primary_ref_frame != PRIMARY_REF_NONE`.
    pub primary_carry: Option<&'a RefSlotCarry>,
    /// r423 — allow the §5.9.14 `temporal_update` election on this
    /// frame (see [`GopTuning::temporal_seg`]). Only consulted on
    /// segmented frames with a primary reference.
    pub allow_temporal_seg: bool,
    /// r424 — alternative §5.9.2 `primary_ref_frame` candidates for
    /// the exact-bytes election: after the main pass (searched and
    /// written under `primary_ref_frame` / `primary_carry` above),
    /// the committed trees are replayed bit-exactly from each
    /// candidate's §8.3.1 frame-start CDF state and the smallest
    /// total frame (header + tile) wins. `(PRIMARY_REF_NONE, None)`
    /// is the per-frame-defaults candidate. All candidates decode to
    /// the identical reconstruction (the trees carry every mode /
    /// coefficient value; only the adaptive-coding bits and the
    /// §5.9.24 gm recentering differ), so the election is purely
    /// rate. Incompatible with the segmented temporal-update election
    /// (caller bug — the pyramid drivers that use this never
    /// segment).
    pub alt_primaries: Vec<(u8, Option<&'a RefSlotCarry>)>,
    /// r426 — exactness-demand mask (row-major `mi_rows × mi_cols`
    /// cells; `None` = no demand). Requires a lossless segment in the
    /// frame's `alt_q` table. Leaves overlapping demanded cells are
    /// forced onto the lossless segment on every arm, making the
    /// demanded region reconstruct pixel-exact.
    pub exact_mask: Option<&'a [bool]>,
    /// r426 — content-driven per-segment-lossless election for
    /// leaves OUTSIDE the demand mask (see
    /// [`GopTuning`]-level docs on the twin-priced trial).
    pub auto_lossless: bool,
    /// r426 — the three §5.9.14 inter-override features (ladder item
    /// 8): SEG_LVL_REF_FRAME / SEG_LVL_SKIP / SEG_LVL_GLOBALMV per
    /// segment. `Some` requires a non-empty `alt_q` table (the
    /// segmentation-enabled carrier) and flips `SegIdPreSkip` to 1.
    pub seg_extras: Option<&'a SegExtras>,
}

/// r415 generic §5.9.2 INTER frame header — every pyramid role
/// (P / ALT / MID / B) shares this shape: non-error-resilient,
/// `PRIMARY_REF_NONE` (per-frame default CDFs), identity global
/// motion, SWITCHABLE frame filter, quarter-pel MVs,
/// `use_ref_frame_mvs = 1`, order hints on, and the §5.9.22
/// `skip_mode_present` derived from the true slot state via the
/// write twin.
fn build_inter_frame_fh(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
    base_q_idx: u8,
    cfg: &InterFrameConfig<'_>,
    alt_q: &[i16],
) -> FrameHeader {
    let mut fh = build_intra_only_yuv420_8bit_fh_with_q(seq, width, height, base_q_idx);
    fh.frame_type = FrameType::Inter;
    fh.frame_is_intra = false;
    // §5.9.2: `showable_frame = frame_type != KEY_FRAME` on the
    // `show_frame == 1` arm (derived, no bit); on the `show_frame == 0`
    // arm it is CODED — a not-shown pyramid frame must be showable for
    // its later `show_existing_frame` display.
    fh.show_frame = cfg.show_frame;
    fh.showable_frame = true;
    // r413: error resilience OFF — §5.9.2 codes primary_ref_frame and
    // opens the use_ref_frame_mvs f(1). r423: the config elects the
    // primary reference (forward CDF inheritance via §6.8.21
    // `load_cdfs` + the §7.21 `load_previous()` /
    // `load_previous_segment_ids()` state) — PRIMARY_REF_NONE keeps
    // the per-frame default state.
    fh.error_resilient_mode = false;
    fh.force_integer_mv = false;
    fh.primary_ref_frame = cfg.primary_ref_frame;
    fh.refresh_frame_flags = cfg.refresh_frame_flags;
    // §5.9.2 `order_hint` — the DISPLAY order (KEY = 0), `<
    // GOP_MAX_FRAMES = 64` so the 7-bit modulus never wraps in-GOP.
    fh.order_hint = cfg.order_hint & ((1 << u32::from(seq.order_hint_bits)) - 1);
    // The true stored per-slot hints — with error resilience off the
    // §5.9.2 ref_order_hint block is not coded, but the writer needs
    // them as session state for the §5.9.22 skipModeAllowed twin.
    fh.ref_order_hints = Some(cfg.slot_hints);
    // §5.9.21: ONLY_4X4 rides the CodedLossless arm; the lossy arm
    // codes TX_MODE_SELECT — intra leaves carry the §5.11.15
    // `tx_depth` choice and inter leaves the §5.11.17 `txfm_split`
    // recursion.
    fh.tx_mode = Some(if base_q_idx == 0 {
        TxMode::Only4x4
    } else {
        TxMode::TxModeSelect
    });
    // §5.9.23: per-block single/compound choice whenever the ladder
    // carries a compound pair.
    let reference_select = !cfg.compound_pairs.is_empty();
    fh.reference_select = Some(reference_select);
    // §5.9.22 `skip_mode_present = 1` whenever skipModeAllowed (the
    // full forward/backward derivation twin over the true slot state;
    // `reference_select` and `enable_order_hint` gate the read).
    let (skip_allowed, _) = crate::encoder::frame_obu::skip_mode_params_twin(
        &cfg.slot_hints,
        &cfg.ref_frame_idx,
        fh.order_hint,
        seq.order_hint_bits,
    );
    fh.skip_mode_present = Some(skip_allowed && reference_select && seq.enable_order_hint);
    // r419: `allow_warped_motion = 1` whenever the sequence gate is
    // open (the f(1) is coded under `!error_resilient_mode &&
    // enable_warped_motion`) — §5.11.27 arm B becomes reachable and
    // the RD ladder may commit WARPED_CAUSAL leaves.
    fh.allow_warped_motion = Some(seq.enable_warped_motion);
    fh.inter_refs = Some(InterFrameRefs {
        frame_refs_short_signaling: false,
        last_frame_idx: None,
        gold_frame_idx: None,
        ref_frame_idx: cfg.ref_frame_idx,
        allow_high_precision_mv: false,
        interpolation_filter: InterpolationFilter::Switchable,
        // r419: §5.9.2 `is_motion_mode_switchable = 1` — every
        // §5.11.27-eligible leaf codes the `use_obmc` /
        // `motion_mode` S(), and the RD ladder may commit OBMC /
        // WARPED_CAUSAL winners.
        is_motion_mode_switchable: true,
        // r413: §7.9 temporal MV prediction on every inter frame (the
        // first P-frame's projections are empty — every slot still
        // holds the intra KEY — which both sides derive identically).
        use_ref_frame_mvs: true,
    });
    // r413: §5.9.14 SEG_LVL_ALT_Q segmentation — under
    // `PRIMARY_REF_NONE` the parser forces `update_map = 1`,
    // `temporal_update = 0`, `update_data = 1` (no bits for the three
    // flags); under a primary reference (r423) the three flags are
    // CODED and this builder emits the same configuration explicitly
    // (`update_map = 1`, `update_data = 1` — the feature table is
    // re-coded every frame; `temporal_update` starts at 0, the
    // per-frame election may flip it after the tile is priced). The
    // feature table codes one active ALT_Q slot per segment
    // (`su(1+8)` each). `SegIdPreSkip = 0` (no feature at `j >=
    // SEG_LVL_REF_FRAME`); `LastActiveSegId = alt_q.len() - 1`.
    if !alt_q.is_empty() {
        fh.segmentation_params = Some(match cfg.seg_extras {
            Some(x) => segmentation_params_for_plan(alt_q, x),
            None => segmentation_params_for(alt_q),
        });
    }
    fh
}

/// The §5.9.14 feature table this encoder codes for a SEG_LVL_ALT_Q
/// delta ladder: one active ALT_Q slot per segment (`su(1+8)` each),
/// `update_map = update_data = 1`, `temporal_update` starting at 0
/// (the r423 per-frame election may flip it after the tile is
/// priced), `SegIdPreSkip = 0`, `LastActiveSegId = alt_q.len() - 1`.
pub(crate) fn segmentation_params_for(
    alt_q: &[i16],
) -> crate::uncompressed_header_tail::SegmentationParams {
    segmentation_params_for_plan(alt_q, &SegExtras::default())
}

/// r426 — the three §5.9.14 inter-override features per segment
/// (ladder item 8): `ref_frame[ s ] = Some(data)` activates
/// SEG_LVL_REF_FRAME with `FeatureData = data` (a raw `RefFrame`
/// ordinal, `LAST_FRAME = 1 ..= ALTREF_FRAME = 7`); `skip[ s ]` /
/// `globalmv[ s ]` activate SEG_LVL_SKIP / SEG_LVL_GLOBALMV
/// (bits = 0 features — no data). Any active slot flips the §5.9.14
/// `SegIdPreSkip` derivation to 1 (segment ids code BEFORE skip on
/// every block).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SegExtras {
    pub ref_frame: [Option<i8>; crate::uncompressed_header_tail::MAX_SEGMENTS],
    pub skip: [bool; crate::uncompressed_header_tail::MAX_SEGMENTS],
    pub globalmv: [bool; crate::uncompressed_header_tail::MAX_SEGMENTS],
}

impl SegExtras {
    /// Any feature slot active (⇒ §5.9.14 `SegIdPreSkip = 1`).
    pub(crate) fn any(&self) -> bool {
        self.ref_frame.iter().any(|r| r.is_some())
            || self.skip.iter().any(|&b| b)
            || self.globalmv.iter().any(|&b| b)
    }

    /// Highest segment index carrying any feature slot.
    pub(crate) fn top_segment(&self) -> usize {
        (0..crate::uncompressed_header_tail::MAX_SEGMENTS)
            .rev()
            .find(|&s| self.ref_frame[s].is_some() || self.skip[s] || self.globalmv[s])
            .unwrap_or(0)
    }
}

/// [`segmentation_params_for`] plus the r426 inter-override features:
/// SEG_LVL_REF_FRAME data is the §5.9.14 3-bit unsigned field,
/// SEG_LVL_SKIP / SEG_LVL_GLOBALMV are `bits = 0` active flags. The
/// §5.9.14 `SegIdPreSkip` / `LastActiveSegId` derivations follow the
/// combined table.
pub(crate) fn segmentation_params_for_plan(
    alt_q: &[i16],
    extras: &SegExtras,
) -> crate::uncompressed_header_tail::SegmentationParams {
    use crate::uncompressed_header_tail::{
        SegmentationParams, SEG_LVL_ALT_Q, SEG_LVL_GLOBALMV, SEG_LVL_REF_FRAME, SEG_LVL_SKIP,
    };
    let mut sp = SegmentationParams::disabled();
    sp.enabled = true;
    sp.update_map = true;
    sp.temporal_update = false;
    sp.update_data = true;
    for (seg, &d) in alt_q.iter().enumerate() {
        sp.segment_feature_active[seg][SEG_LVL_ALT_Q] = true;
        sp.segment_feature_data[seg][SEG_LVL_ALT_Q] = d;
    }
    for (seg, rf) in extras.ref_frame.iter().enumerate() {
        if let Some(data) = rf {
            sp.segment_feature_active[seg][SEG_LVL_REF_FRAME] = true;
            sp.segment_feature_data[seg][SEG_LVL_REF_FRAME] = i16::from(*data);
        }
    }
    for (seg, &on) in extras.skip.iter().enumerate() {
        sp.segment_feature_active[seg][SEG_LVL_SKIP] = on;
    }
    for (seg, &on) in extras.globalmv.iter().enumerate() {
        sp.segment_feature_active[seg][SEG_LVL_GLOBALMV] = on;
    }
    // §5.9.14 derivations over the combined table.
    sp.seg_id_pre_skip = extras.any();
    sp.last_active_seg_id = alt_q.len().saturating_sub(1).max(if extras.any() {
        extras.top_segment()
    } else {
        0
    }) as u8;
    sp
}

/// r426 — the §5.9.2 `LosslessArray[]` this encoder's headers derive:
/// `LosslessArray[ s ] = get_qindex( 1, s ) == 0` (every `DeltaQ??`
/// side value is zero on every header this encoder emits, so the
/// §5.9.2 conjunction collapses to the qindex test). Segments at or
/// past `alt_q.len()` carry no active `SEG_LVL_ALT_Q` feature —
/// §7.12.2 returns `base_q_idx` for them.
pub(crate) fn seg_lossless_array(
    base_q_idx: u8,
    alt_q: &[i16],
) -> [bool; crate::uncompressed_header_tail::MAX_SEGMENTS] {
    core::array::from_fn(|s| {
        let q = match alt_q.get(s) {
            Some(&d) => (i32::from(base_q_idx) + i32::from(d)).clamp(0, 255),
            None => i32::from(base_q_idx),
        };
        q == 0
    })
}

/// The r412/r413 two-slot P-frame [`InterFrameConfig`] for P-frame
/// `p_index` (see [`gop_ref_hints`] for the rotation).
#[cfg(test)]
fn p_frame_config<'a>(
    prev: &'a GopFrameReconYuv,
    prevprev: &'a GopFrameReconYuv,
    p_index: u32,
) -> InterFrameConfig<'a> {
    p_frame_config_primary(prev, prevprev, p_index, None)
}

/// r423 — [`p_frame_config`] with the primary-reference election:
/// `primary_carry = Some` elects `primary_ref_frame = 0` (LAST — the
/// ordinal reading slot `p_index & 1`, which holds frame
/// `p_index - 1`, the §7.20 slot the caller's carry store tracks);
/// `None` keeps `PRIMARY_REF_NONE` (the pre-r423 shape, and the A/B
/// baseline).
fn p_frame_config_primary<'a>(
    prev: &'a GopFrameReconYuv,
    prevprev: &'a GopFrameReconYuv,
    p_index: u32,
    primary_carry: Option<&'a RefSlotCarry>,
) -> InterFrameConfig<'a> {
    let last_slot = (p_index & 1) as usize;
    let golden_slot = ((p_index - 1) & 1) as usize;
    let (last_hint, golden_hint) = gop_ref_hints(p_index);
    let mut slot_hints = [0u32; 8];
    slot_hints[last_slot] = last_hint;
    slot_hints[golden_slot] = golden_hint;
    let mut ref_frame_idx = [last_slot as u8; REFS_PER_FRAME];
    // GOLDEN_FRAME = 4 → ref_frame_idx[ GOLDEN_FRAME - LAST_FRAME = 3 ].
    ref_frame_idx[3] = golden_slot as u8;
    let mut slot_to_plane = [0usize; 8];
    slot_to_plane[golden_slot] = 1;
    InterFrameConfig {
        order_hint: p_index,
        show_frame: true,
        refresh_frame_flags: 1 << ((p_index - 1) & 1),
        ref_frame_idx,
        slot_hints,
        single_refs: vec![1, 4],
        compound_pairs: vec![[1, 4]],
        refs: vec![prev, prevprev],
        slot_to_plane,
        primary_ref_frame: if primary_carry.is_some() {
            0 // LAST_FRAME ordinal — ref_frame_idx[ 0 ] = last_slot.
        } else {
            PRIMARY_REF_NONE
        },
        primary_carry,
        allow_temporal_seg: true,
        alt_primaries: Vec::new(),
        exact_mask: None,
        auto_lossless: false,
        seg_extras: None,
    }
}

#[cfg(test)]
fn build_p_frame_yuv420_8bit_fh_with_q(
    seq: &SequenceHeader,
    width: u32,
    height: u32,
    base_q_idx: u8,
    p_index: u32,
    alt_q: &[i16],
) -> FrameHeader {
    // The refs are irrelevant to the header — reuse a dummy pair.
    let dummy = GopFrameReconYuv {
        y: Vec::new(),
        u: Vec::new(),
        v: Vec::new(),
    };
    let cfg = p_frame_config(&dummy, &dummy, p_index);
    build_inter_frame_fh(seq, width, height, base_q_idx, &cfg, alt_q)
}

/// Encode one INTER P-frame against `prev` (the previous frame's
/// reconstruction, LAST_FRAME) and `prevprev` (the frame before it,
/// GOLDEN_FRAME — the KEY recon again for the first two P-frames).
/// Returns the §7.5 temporal unit (TD + `OBU_FRAME`) and this frame's
/// reconstruction.
#[allow(clippy::too_many_arguments)]
fn encode_p_frame_yuv(
    input: &YuvFrame,
    prev: &GopFrameReconYuv,
    prevprev: &GopFrameReconYuv,
    seq: &SequenceHeader,
    base_q_idx: u8,
    p_index: u32,
    alt_q: &[i16],
    mf_store: &[SavedMotionField; 8],
    model: RateModel,
    global_motion: bool,
    primary_carry: Option<&RefSlotCarry>,
    allow_temporal_seg: bool,
    exact_mask: Option<&[bool]>,
    auto_lossless: bool,
    seg_extras: Option<&SegExtras>,
) -> Result<
    (
        Vec<u8>,
        GopFrameReconYuv,
        SavedMotionField,
        RefSlotCarry,
        InterFrameAux,
    ),
    Error,
> {
    let mut cfg = p_frame_config_primary(prev, prevprev, p_index, primary_carry);
    cfg.allow_temporal_seg = allow_temporal_seg;
    cfg.exact_mask = exact_mask;
    cfg.auto_lossless = auto_lossless;
    cfg.seg_extras = seg_extras;
    let (obu, recon, saved, carry, aux) = encode_inter_frame_generic_gm(
        input,
        seq,
        base_q_idx,
        &cfg,
        alt_q,
        mf_store,
        model,
        global_motion,
    )?;
    // §7.5 temporal unit: TD + OBU_FRAME (the SH rode the KEY frame's
    // unit; §7.5 requires it once per coded video sequence). Every
    // P-frame is shown, so the one-OBU unit satisfies the "exactly
    // one shown frame per temporal unit" bitstream conformance rule.
    Ok((build_temporal_unit(None, &[obu]), recon, saved, carry, aux))
}

/// r415 — the shared INTER frame encoder every pyramid role (P / ALT /
/// MID / B) runs: builds the §5.9.2 header from an
/// [`InterFrameConfig`], derives order hints / §7.8 sign bias /
/// §5.9.22 skip mode / §7.9 temporal MVs from the config's true slot
/// state, runs the r411-r413 RD search over the config's reference
/// ladder, and writes the §5.10 `OBU_FRAME`. Returns
/// `(frame OBU, reconstruction, §7.19 motion-field payload)` — the
/// caller packs OBUs into §7.5 temporal units (each unit must carry
/// exactly ONE shown frame, so pyramid drivers bundle
/// decoded-not-shown frames with the next shown one).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_inter_frame_generic(
    input: &YuvFrame,
    seq: &SequenceHeader,
    base_q_idx: u8,
    cfg: &InterFrameConfig<'_>,
    alt_q: &[i16],
    mf_store: &[SavedMotionField; 8],
    model: RateModel,
) -> Result<
    (
        ObuFrame,
        GopFrameReconYuv,
        SavedMotionField,
        RefSlotCarry,
        InterFrameAux,
    ),
    Error,
> {
    encode_inter_frame_generic_gm(input, seq, base_q_idx, cfg, alt_q, mf_store, model, true)
}

/// r422 — [`encode_inter_frame_generic`] with the global-motion
/// election switch (`false` = identity-only headers, the pre-r422
/// baseline the measurement harness A/Bs against).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_inter_frame_generic_gm(
    input: &YuvFrame,
    seq: &SequenceHeader,
    base_q_idx: u8,
    cfg: &InterFrameConfig<'_>,
    alt_q: &[i16],
    mf_store: &[SavedMotionField; 8],
    model: RateModel,
    global_motion: bool,
) -> Result<
    (
        ObuFrame,
        GopFrameReconYuv,
        SavedMotionField,
        RefSlotCarry,
        InterFrameAux,
    ),
    Error,
> {
    input.validate()?;
    // r423 — a primary-reference election requires the §7.20 carry of
    // the elected slot and a valid ordinal (`0..7`).
    if cfg.primary_ref_frame != PRIMARY_REF_NONE
        && (usize::from(cfg.primary_ref_frame) >= REFS_PER_FRAME || cfg.primary_carry.is_none())
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let bit_depth = input.bit_depth;
    let (ssx, ssy) = input.format.subsampling();
    let mono = input.format == crate::encoder::yuv_frame::ChromaFormat::Monochrome;
    let num_planes = input.format.num_planes();
    let width = input.width as usize;
    let height = input.height as usize;
    let chroma_w = input.chroma_width() as usize;
    let chroma_h = input.chroma_height() as usize;
    for reference in &cfg.refs {
        if reference.y.len() != width * height
            || reference.u.len() != chroma_w * chroma_h
            || reference.v.len() != chroma_w * chroma_h
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    let mut fh = build_inter_frame_fh(seq, input.width, input.height, base_q_idx, cfg, alt_q);
    // r427 — the §5.11.46 palette election inside inter frames stays
    // 8-bit-4:2:0-scoped, like the KEY driver.
    fh.allow_screen_content_tools = fh.allow_screen_content_tools
        && bit_depth == 8
        && input.format == crate::encoder::yuv_frame::ChromaFormat::Yuv420;
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
    let lossless = base_q_idx == 0;
    let mut qp = QuantizerParams::neutral(base_q_idx, bit_depth);
    if !alt_q.is_empty() {
        // §7.12.2 get_qindex inputs — the write-side §5.11.47 guard
        // and the §5.11.39 quantiser chain both key off these.
        qp.segmentation_enabled = true;
        for (seg, &d) in alt_q.iter().enumerate() {
            qp.seg_alt_q_active[seg] = true;
            qp.seg_alt_q_data[seg] = d;
        }
    }

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
    // r412: per-block single/compound reference choice (§5.9.23) —
    // open exactly when the ladder carries a compound pair (matches
    // the header).
    ip.reference_select = !cfg.compound_pairs.is_empty();
    // r415: §5.5.2 masked compound — compound leaves code the
    // §5.11.29 comp_group_idx cascade and may commit COMPOUND_WEDGE /
    // COMPOUND_DIFFWTD (the RD ladder trials both against the
    // AVERAGE baseline).
    ip.enable_masked_compound = seq.enable_masked_compound;
    // r417: §5.5.2 inter-intra compound — single-reference 8x8..32x32
    // leaves code the §5.11.28 cascade and may commit smooth / wedge
    // inter-intra blends (the RD ladder trials all four II modes and
    // the 16 wedge masks against the plain leaf).
    ip.enable_interintra_compound = seq.enable_interintra_compound;
    // r416: §5.5.2 jnt-comp — compound leaves code the §5.11.29
    // `compound_idx` S() and may commit COMPOUND_DISTANCE (the RD
    // ladder trials the §7.11.3.15 distance-weighted blend against
    // the AVERAGE baseline).
    ip.enable_jnt_comp = seq.enable_jnt_comp;
    // r419: §5.9.2 motion-mode switchability + warped motion — the
    // write pass codes §5.11.27 `motion_mode` on eligible leaves and
    // the RD ladder trials OBMC (and, with the warp gates open,
    // WARPED_CAUSAL) against SIMPLE. `allow_warped_motion` mirrors
    // the header derivation: coded only when the sequence gate is
    // open (the header writer emits the f(1) under the same gate).
    ip.is_motion_mode_switchable = true;
    ip.allow_warped_motion = seq.enable_warped_motion;
    // §5.9.2 order hints — `OrderHints[ ref ]` =
    // `RefOrderHint[ ref_frame_idx[ ref - LAST_FRAME ] ]` over the
    // config's slot state.
    ip.order_hints = crate::cdf::FrameInterOrderHints {
        order_hint_bits: u32::from(seq.order_hint_bits),
        current_order_hint: fh.order_hint as i32,
        order_hints_by_ref: {
            let mut by_ref = [0i32; crate::uncompressed_header_tail::ALTREF_FRAME + 1];
            for (i, hint) in by_ref.iter_mut().enumerate().skip(1) {
                *hint = cfg.slot_hints[cfg.ref_frame_idx[i - 1] as usize] as i32;
            }
            by_ref
        },
    };
    // §7.8 `RefFrameSignBias[ ref ]` = `get_relative_dist(
    // OrderHints[ ref ], OrderHint ) > 0` — backward references (the
    // pyramid's BWDREF/ALTREF roles) flip the bias.
    for r in 1..=7usize {
        ip.ref_frame_sign_bias[r] = i32::from(
            crate::inter_pred::get_relative_dist(
                ip.order_hints.order_hints_by_ref[r],
                fh.order_hint as i32,
                u32::from(seq.order_hint_bits),
            ) > 0,
        );
    }
    // §5.9.22 skip mode — the SAME write-twin derivation the header
    // builder ran (presence AND the `SkipModeFrame[]` pair).
    {
        let (allowed, pair) = crate::encoder::frame_obu::skip_mode_params_twin(
            &cfg.slot_hints,
            &cfg.ref_frame_idx,
            fh.order_hint,
            seq.order_hint_bits,
        );
        ip.skip_mode_present = fh.skip_mode_present == Some(true) && allowed;
        ip.skip_mode_frame = pair;
    }
    // r413: segmentation map update (spatial coding; the r423
    // frame-level election may flip the temporal arm on after the
    // tile is priced). r423: `load_previous_segment_ids()` — with a
    // primary reference whose stored extent matches, `PrevSegmentIds`
    // is that slot's §7.20 `SavedSegmentIds`; otherwise all-zero
    // (also the setup_past_independence state for PRIMARY_REF_NONE),
    // exactly mirroring the decode driver's derivation.
    if !alt_q.is_empty() {
        ip.segmentation_update_map = true;
        // r423 — the main pass always searches and emits under the
        // SPATIAL arm: the search (and therefore the committed trees)
        // is bit-identical to the temporal-disabled baseline, and the
        // exact-bits election below replays those trees under the
        // §5.11.19 temporal arm — so the elected stream can only be
        // smaller-or-equal, PER FRAME, than the spatial-only encode
        // of the same input. (An early variant searched under the
        // temporal arm and elected the spatial replay as fallback;
        // measurement showed the temporal-biased trees sometimes lose
        // to a spatial-native search outright, which the flipped
        // election excludes by construction.)
        ip.segmentation_temporal_update = false;
        let cells = (mi_rows as usize) * (mi_cols as usize);
        let prev_ids = match cfg.primary_carry {
            Some(c)
                if cfg.primary_ref_frame != PRIMARY_REF_NONE
                    && c.mi_rows == mi_rows
                    && c.mi_cols == mi_cols
                    && c.segment_ids.len() == cells =>
            {
                c.segment_ids.clone()
            }
            _ => vec![0i32; cells],
        };
        ip.prev_segment_ids = Some(std::sync::Arc::new(prev_ids));
    }
    // r413: §7.9 motion-field estimation over the encoder-side §7.20
    // store — the SAME shared core the decode driver runs, so the
    // §7.10.2.5 temporal scan sees identical `MotionFieldMvs` at
    // search time, at write time and at decode time.
    ip.use_ref_frame_mvs = true;
    {
        use crate::inter_pred::{motion_field_estimation_core, MotionFieldSlot};
        let ref_frame_idx = cfg.ref_frame_idx;
        let mut order_hints = [0i32; 8];
        order_hints.copy_from_slice(&ip.order_hints.order_hints_by_ref);
        let mut slots: [Option<MotionFieldSlot<'_>>; 8] = [None; 8];
        for (i, slot) in mf_store.iter().enumerate() {
            slots[i] = Some(MotionFieldSlot {
                mf_mvs: &slot.mf_mvs,
                mf_ref_frames: &slot.mf_ref_frames,
                saved_order_hints: slot.saved_order_hints,
                mi_rows: slot.mi_rows,
                mi_cols: slot.mi_cols,
                frame_is_intra: slot.frame_is_intra,
            });
        }
        ip.motion_field_mvs = motion_field_estimation_core(
            &slots,
            &ref_frame_idx,
            &order_hints,
            fh.order_hint as i32,
            u32::from(seq.order_hint_bits),
            mi_rows,
            mi_cols,
        );
    }

    // r422 — frame-level global-motion election (§5.9.24): one coarse
    // motion pre-pass per DISTINCT reference reconstruction, then the
    // least-squares model fits + §5.9.25 grid quantization +
    // §7.11.3.6 validation of [`crate::encoder::global_motion`]. The
    // elected `(GmType, gm_params)` pairs are stamped into the shared
    // inter bundle (feeding the §7.10.2.1 `GlobalMvs` derivation and
    // the §7.11.3 global-warp arm of BOTH the search mirror and the
    // write pass) and into the frame header (feeding the §5.9.24
    // write arm) — one source of truth on every side, so the stream
    // and the search can never disagree about the model.
    if global_motion {
        let hp = fh
            .inter_refs
            .as_ref()
            .map(|ir| ir.allow_high_precision_mv)
            .unwrap_or(false);
        // Estimate once per distinct reference plane, then fan the
        // result out to every RefFrame ordinal mapped onto it.
        let mut per_plane: Vec<Option<(crate::uncompressed_header_tail::WarpModelType, [i32; 6])>> =
            vec![None; cfg.refs.len()];
        for &rf in &cfg.single_refs {
            let plane = cfg.slot_to_plane[cfg.ref_frame_idx[(rf - 1) as usize] as usize];
            let (t, p) = match per_plane[plane] {
                Some(tp) => tp,
                None => {
                    let samples = crate::encoder::global_motion::collect_motion_samples(
                        &input.y,
                        &cfg.refs[plane].y,
                        width,
                        height,
                    );
                    let tp = crate::encoder::global_motion::estimate_global_motion(&samples, hp);
                    per_plane[plane] = Some(tp);
                    tp
                }
            };
            ip.gm_type[rf as usize] = i32::from(t.as_u8());
            ip.gm_params[rf as usize] = p;
        }
        // Header twin of the elected models.
        let mut gmp = crate::uncompressed_header_tail::GlobalMotionParams::identity();
        gmp.short_circuited = false;
        for r in 1..=7usize {
            gmp.gm_type[r] = match ip.gm_type[r] {
                1 => crate::uncompressed_header_tail::WarpModelType::Translation,
                2 => crate::uncompressed_header_tail::WarpModelType::RotZoom,
                3 => crate::uncompressed_header_tail::WarpModelType::Affine,
                _ => crate::uncompressed_header_tail::WarpModelType::Identity,
            };
            gmp.gm_params[r] = ip.gm_params[r];
        }
        fh.global_motion_params = Some(gmp);
    }

    // r426 — the §5.9.14 inter-override feature tables + derivations
    // (SegIdPreSkip / LastActiveSegId over the combined table).
    let x = cfg.seg_extras.copied().unwrap_or_default();
    let seg_id_pre_skip = x.any();
    let last_active_seg_id =
        alt_q
            .len()
            .saturating_sub(1)
            .max(if x.any() { x.top_segment() } else { 0 }) as u8;
    let params = SyntaxFrameParams {
        subsampling_x: ssx,
        subsampling_y: ssy,
        num_planes,
        seg_id_pre_skip,
        segmentation_enabled: !alt_q.is_empty(),
        seg_ref_frame: x.ref_frame,
        seg_skip: x.skip,
        seg_globalmv: x.globalmv,
        last_active_seg_id,
        // r426 — per-segment §5.9.2 `LosslessArray[]`: a segment whose
        // §7.12.2 qindex clamps to 0 flips the full lossless leaf
        // semantics for its blocks while the frame stays lossy.
        lossless_array: seg_lossless_array(base_q_idx, alt_q),
        coded_lossless: lossless,
        enable_cdef: seq.enable_cdef,
        allow_intrabc: false,
        cdef_bits: 0,
        read_deltas: false,
        use_128x128_superblock: seq.use_128x128_superblock,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_multi: false,
        mono_chrome: mono,
        delta_lf_res: 0,
        allow_screen_content_tools: fh.allow_screen_content_tools,
        enable_filter_intra: seq.enable_filter_intra,
        bit_depth,
        tx_mode_select: !lossless,
        quant: qp,
        reduced_tx_set: fh.reduced_tx_set.unwrap_or(false),
        inter: Some(ip.clone()),
    };

    let mut recon = ReconState {
        y: vec![0u16; width * height],
        u: vec![0u16; chroma_w * chroma_h],
        v: vec![0u16; chroma_w * chroma_h],
        width,
        height,
        chroma_w,
        chroma_h,
        bit_depth,
        subsampling_x: ssx,
        subsampling_y: ssy,
        num_planes,
        mi_rows,
        mi_cols,
        lossless,
        allow_screen_content_tools: fh.allow_screen_content_tools,
        // §5.9.20: intra-block-copy is intra-frame-only.
        allow_intrabc: false,
        qp,
        bd: BlockDecodedMirror::new(ssx, ssy, num_planes),
        dv_hash: Default::default(),
    };
    let mut ictx = PSearchCtx::with_refs(
        &cfg.refs,
        cfg.slot_to_plane,
        cfg.ref_frame_idx,
        cfg.single_refs.clone(),
        cfg.compound_pairs.clone(),
        mi_rows,
        mi_cols,
        width,
        height,
        ip,
        base_q_idx,
        alt_q,
        bit_depth,
        ssx,
        ssy,
        num_planes,
    )?;
    // r426 — arm the §5.9.14 inter-override features (ladder item 8):
    // requires the segmentation carrier (non-empty alt_q), raw
    // RefFrame data within the ladder, and flips SegIdPreSkip.
    if let Some(x) = cfg.seg_extras {
        if alt_q.is_empty()
            || x.ref_frame
                .iter()
                .flatten()
                .any(|rf| !cfg.single_refs.contains(rf))
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        ictx.extras = Some(*x);
        ictx.seg_id_pre_skip = x.any();
    }
    // r426 — arm the exactness-demand / auto-lossless election. Both
    // require a lossless segment in the table, and the mask must
    // cover the mi grid exactly.
    if cfg.exact_mask.is_some() || cfg.auto_lossless {
        if ictx.lossless_seg.is_none() {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if let Some(mask) = cfg.exact_mask {
            if mask.len() != (mi_rows as usize) * (mi_cols as usize) {
                return Err(Error::PartitionWalkOutOfRange);
            }
            ictx.exact_mask = Some(mask.to_vec());
        }
        ictx.auto_lossless = cfg.auto_lossless;
    }

    let mut writer = SymbolWriter::new(fh.disable_cdf_update);
    // §8.3.1 frame-start CDF state (mirroring the decode driver):
    // with a primary reference, §6.8.21 `load_cdfs( ref_frame_idx[
    // primary_ref_frame ] )` — the slot's §7.20-saved frame-end state
    // (coefficient CDFs included) with the per-row symbol counts
    // zeroed; otherwise `init_non_coeff_cdfs()` (the §9.4 defaults)
    // plus the q-context-selected `init_coeff_cdfs( base_q_idx )`.
    let frame_start_cdfs: Box<TileCdfContext> = match cfg.primary_carry {
        Some(carry) if cfg.primary_ref_frame != PRIMARY_REF_NONE => {
            let mut loaded = carry.cdfs.clone();
            loaded.zero_counts();
            loaded
        }
        _ => {
            let mut c = TileCdfContext::new_from_defaults();
            c.init_coeff_cdfs(base_q_idx);
            Box::new(c)
        }
    };
    let mut cdfs = frame_start_cdfs.as_ref().clone();
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

    // r423 — the committed per-superblock syntax trees are retained:
    // the §5.9.14 `temporal_update` election below replays them
    // bit-exactly under the opposite segment-id coding arm.
    let mut trees: Vec<SyntaxNode> = Vec::new();
    for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
        recon.bd.clear_for_sb(sb_r, sb_c, mi_rows, mi_cols);
        // r421 — arm the §5.11.2 delta lifecycle on the live state AND
        // the rate twin's fork, so both enter the superblock identically.
        state.arm_read_deltas();
        let mut twin = RateTwin::snapshot(&cdfs, &state, &writer);
        twin.arm_read_deltas();
        let (tree, _cost) = build_p_search_tree(
            sb_r,
            sb_c,
            BLOCK_64X64,
            input,
            &mut recon,
            &mut ictx,
            &mut twin,
            &params,
            model,
        )?;
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
        // r421 anti-desync invariant: the search committed EXACTLY the
        // symbols the writer just emitted.
        debug_assert!(
            twin.matches(&cdfs, &writer),
            "rate twin desynced from the writer after superblock ({sb_r},{sb_c})"
        );
        trees.push(tree);
    }
    let mut tile_bytes = writer.finish();

    // r423 — §5.9.14 `segmentation_temporal_update` election by EXACT
    // realized bits: the main pass searched and wrote under the
    // spatial arm; when the frame can code the temporal arm (a
    // primary reference exists and the config allows it), replay the
    // identical committed trees under the §5.11.19 temporal arm from
    // the same §8.3.1 frame-start CDF state and keep whichever tile
    // is smaller. Both arms decode to the same reconstruction (the
    // segment ids, modes and coefficients are the trees'; only the
    // §5.11.19 id-coding bits differ), so the election is purely
    // rate. Header cost is arm-invariant (each §5.9.14 flag is one
    // f(1) either way).
    let main_ip = params
        .inter
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let mut seg_temporal_elected = false;
    let temporal_codable =
        !alt_q.is_empty() && cfg.primary_ref_frame != PRIMARY_REF_NONE && cfg.allow_temporal_seg;
    if temporal_codable {
        let mut alt_ip = main_ip.clone();
        alt_ip.segmentation_temporal_update = true;
        let mut alt_params = params.clone();
        alt_params.inter = Some(alt_ip);
        let mut alt_writer = SymbolWriter::new(fh.disable_cdf_update);
        let mut alt_cdfs = frame_start_cdfs.as_ref().clone();
        let mut alt_state = PartitionSyntaxWriter::new(
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
        for ((sb_r, sb_c), tree) in sb_grid_origins(mi_rows, mi_cols).into_iter().zip(&trees) {
            alt_state.arm_read_deltas();
            write_partition_tree_syntax(
                &mut alt_writer,
                &mut alt_cdfs,
                &mut alt_state,
                tree,
                sb_r,
                sb_c,
                BLOCK_64X64,
                &alt_params,
            )?;
        }
        let alt_bytes = alt_writer.finish();
        if alt_bytes.len() < tile_bytes.len() {
            seg_temporal_elected = true;
            tile_bytes = alt_bytes;
            cdfs = alt_cdfs;
            state = alt_state;
        }
    }
    // The header codes the elected arm (`update_map`/`update_data`
    // are already 1 on every segmented frame).
    if let Some(sp) = fh.segmentation_params.as_mut() {
        sp.temporal_update = seg_temporal_elected;
    }

    // r424 — §5.9.2 `primary_ref_frame` election by EXACT realized
    // bytes: the main pass searched and wrote under the config's
    // primary reference; each alternative candidate replays the
    // identical committed trees from its own §8.3.1 frame-start CDF
    // state (§6.8.21 `load_cdfs` of the candidate's carried slot, or
    // the per-frame defaults for the `PRIMARY_REF_NONE` candidate)
    // and re-prices the header (the 3-bit ordinal is constant; the
    // §5.9.24 subexp recentering against the candidate's
    // `PrevGmParams` is not). Every candidate decodes to the SAME
    // reconstruction — the trees carry all modes and coefficients —
    // so the election is purely rate, and the smallest header + tile
    // total wins.
    let mut prev_gm_for_header: Option<[[i32; 6]; TOTAL_REFS_PER_FRAME]> =
        if fh.primary_ref_frame != PRIMARY_REF_NONE {
            cfg.primary_carry.map(|c| c.gm_params)
        } else {
            None
        };
    if !cfg.alt_primaries.is_empty() {
        // The segmented temporal-update election above and this
        // election both swap the tile — combining them would need a
        // per-candidate seg replay matrix nobody codes yet (the
        // pyramid drivers that offer alternatives never segment).
        if temporal_codable {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let header_len = |ord: u8, prev: Option<&[[i32; 6]; TOTAL_REFS_PER_FRAME]>| -> usize {
            let mut fh_c = fh.clone();
            fh_c.primary_ref_frame = ord;
            let mut bw = crate::encoder::bitwriter::BitWriter::new();
            crate::encoder::frame_obu::encode_uncompressed_header_with_prev_gm(
                &mut bw, &fh_c, seq, prev,
            );
            bw.byte_align();
            bw.finish().len()
        };
        let mut best_total =
            header_len(fh.primary_ref_frame, prev_gm_for_header.as_ref()) + tile_bytes.len();
        #[allow(clippy::type_complexity)]
        let mut best: Option<(
            u8,
            Vec<u8>,
            TileCdfContext,
            PartitionSyntaxWriter,
            Option<[[i32; 6]; TOTAL_REFS_PER_FRAME]>,
        )> = None;
        for (ord, carry_opt) in &cfg.alt_primaries {
            if *ord != PRIMARY_REF_NONE
                && (usize::from(*ord) >= REFS_PER_FRAME || carry_opt.is_none())
            {
                return Err(Error::PartitionWalkOutOfRange);
            }
            let mut cand_cdfs: TileCdfContext = match carry_opt {
                Some(carry) if *ord != PRIMARY_REF_NONE => {
                    let mut loaded = carry.cdfs.as_ref().clone();
                    loaded.zero_counts();
                    loaded
                }
                _ => {
                    let mut c = TileCdfContext::new_from_defaults();
                    c.init_coeff_cdfs(base_q_idx);
                    c
                }
            };
            let mut cand_writer = SymbolWriter::new(fh.disable_cdf_update);
            let mut cand_state = PartitionSyntaxWriter::new(
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
            for ((sb_r, sb_c), tree) in sb_grid_origins(mi_rows, mi_cols).into_iter().zip(&trees) {
                cand_state.arm_read_deltas();
                write_partition_tree_syntax(
                    &mut cand_writer,
                    &mut cand_cdfs,
                    &mut cand_state,
                    tree,
                    sb_r,
                    sb_c,
                    BLOCK_64X64,
                    &params,
                )?;
            }
            let cand_bytes = cand_writer.finish();
            let cand_prev = if *ord != PRIMARY_REF_NONE {
                carry_opt.map(|c| c.gm_params)
            } else {
                None
            };
            let total = header_len(*ord, cand_prev.as_ref()) + cand_bytes.len();
            if total < best_total {
                best_total = total;
                best = Some((*ord, cand_bytes, cand_cdfs, cand_state, cand_prev));
            }
        }
        if let Some((ord, bytes, e_cdfs, e_state, e_prev)) = best {
            fh.primary_ref_frame = ord;
            tile_bytes = bytes;
            cdfs = e_cdfs;
            state = e_state;
            prev_gm_for_header = e_prev;
        }
    }

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
    // r423: with a primary reference the §5.9.24 subexp recentering
    // codes against the carried `SavedGmParams` (§7.21
    // `load_previous()`), not the setup_past_independence defaults.
    let frame_body = {
        let mut bw = crate::encoder::bitwriter::BitWriter::new();
        crate::encoder::frame_obu::encode_uncompressed_header_with_prev_gm(
            &mut bw,
            &fh,
            seq,
            prev_gm_for_header.as_ref(),
        );
        bw.byte_align();
        let mut body = bw.finish();
        body.extend_from_slice(&tile_group_body);
        body
    };
    let frame_obu = ObuFrame::new(ObuType::Frame, frame_body);

    // r413 — §7.19 motion field motion vector storage: filter the
    // committed Mvs[] / RefFrames[] mirror grids down to the §7.20
    // per-slot payload (per cell, the LAST candidate list whose
    // reference lies in the past and whose components sit within
    // REFMVS_LIMIT), exactly like the decode driver does after
    // decoding this frame.
    const REFMVS_LIMIT: i16 = (1 << 12) - 1;
    let cells = (mi_rows as usize) * (mi_cols as usize);
    let mut mf_ref_frames: Vec<i8> = vec![-1; cells];
    let mut mf_mvs: Vec<i16> = vec![0; cells * 2];
    {
        let raw_refs = ictx.mirror.ref_frames();
        let raw_mvs = ictx.mirror.mvs();
        let hint_bits = u32::from(seq.order_hint_bits);
        let by_ref = ictx.ip.order_hints.order_hints_by_ref;
        for cell in 0..cells {
            for list in 0..2usize {
                let r = raw_refs[cell * 2 + list];
                if r > 0 {
                    let dist = crate::inter_pred::get_relative_dist(
                        by_ref[r as usize],
                        fh.order_hint as i32,
                        hint_bits,
                    );
                    if dist < 0 {
                        let mv_row = raw_mvs[(cell * 2 + list) * 2];
                        let mv_col = raw_mvs[(cell * 2 + list) * 2 + 1];
                        if mv_row.abs() <= REFMVS_LIMIT && mv_col.abs() <= REFMVS_LIMIT {
                            mf_ref_frames[cell] = r;
                            mf_mvs[cell * 2] = mv_row;
                            mf_mvs[cell * 2 + 1] = mv_col;
                        }
                    }
                }
            }
        }
    }
    let saved_mf = SavedMotionField {
        mf_mvs,
        mf_ref_frames,
        saved_order_hints: ictx.ip.order_hints.order_hints_by_ref,
        mi_rows,
        mi_cols,
        frame_is_intra: false,
    };

    // r423 — this frame's own §7.20 slot payload for the caller's
    // carry store: the tile's adapted frame-end CDFs (§8.4
    // `save_cdfs` — single tile, `context_update_tile_id = 0`,
    // `disable_frame_end_update_cdf = 0`), the committed
    // `SavedSegmentIds` grid (the write mirror's stamps — identical
    // to the decode walker's by the lockstep argument), and
    // `SavedGmParams` (this frame's decoded GmParams table; identity
    // rows where no model was coded).
    let carry = RefSlotCarry {
        cdfs: Box::new(cdfs),
        segment_ids: state.mirror().segment_ids().to_vec(),
        mi_rows,
        mi_cols,
        gm_params: fh
            .global_motion_params
            .as_ref()
            .map(|g| g.gm_params)
            .unwrap_or_else(crate::uncompressed_header_tail::prev_gm_params_default),
    };

    Ok((
        frame_obu,
        GopFrameReconYuv {
            y: recon.y,
            u: recon.u,
            v: recon.v,
        },
        saved_mf,
        carry,
        InterFrameAux {
            seg_temporal: seg_temporal_elected,
            primary_ref: fh.primary_ref_frame,
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
    /// r427 — stream §5.5.2 `BitDepth` (threads the §7.11.2 /
    /// §7.11.3 clip + rounding parameters of the decoder kernels the
    /// search predicts through).
    bit_depth: u8,
    /// §5.5.2 `subsampling_x` (0 or 1).
    subsampling_x: u8,
    /// §5.5.2 `subsampling_y`.
    subsampling_y: u8,
    /// §5.5.2 `NumPlanes` (1 or 3).
    num_planes: u8,
    /// r415 — the DISTINCT reference reconstructions this frame reads,
    /// widened to the §7.11.3.4 sample type (one entry per distinct
    /// coded frame; [`Self::slot_to_plane`] maps §7.20 slots onto
    /// them).
    ref_planes: Vec<[Vec<u16>; 3]>,
    /// r415 — §7.20 slot → [`Self::ref_planes`] index. Slots never
    /// resolved through `ref_frame_idx` may point anywhere (their
    /// content is immaterial but must be sized like a real plane).
    slot_to_plane: [usize; 8],
    /// §5.9.2 `ref_frame_idx[ 0..7 ]` — the header's slot map, fed
    /// verbatim to the decoder's leaf driver.
    ref_frame_idx: [u8; 7],
    /// r415 — the single-reference ladder: each entry is a raw
    /// `RefFrame` ordinal (`LAST_FRAME = 1 ..= ALTREF_FRAME = 7`) the
    /// per-leaf §5.11.24 search trials with its own motion search.
    single_refs: Vec<i8>,
    /// r415 — the compound ladder: `[ fwd, bwd-or-second ]` pairs the
    /// §5.11.25 cascade can code (slot 0 must be the §6.10.24 lower
    /// ordinal); both members must appear in [`Self::single_refs`]
    /// (NEW_NEWMV reuses their searched vectors).
    compound_pairs: Vec<[i8; 2]>,
    // §5.11.5 grids (per mi cell).
    mi_sizes: Vec<usize>,
    is_inters: Vec<u8>,
    ref_frames: Vec<i8>,
    mvs: Vec<i16>,
    interp_filters: Vec<u8>,
    motion_modes: Vec<u8>,
    y_modes: Vec<u8>,
    /// r419 — §5.11.18 per-cell `AvailU` / `AvailL` grids for the
    /// §7.11.3.9 OBMC pass gates (single full-frame tile: `AvailU =
    /// MiRow > 0`, `AvailL = MiCol > 0`), precomputed once.
    avail_us: Vec<u8>,
    avail_ls: Vec<u8>,
    /// r415 — the §5.11.5 `CompoundTypes[ .. ]` grid (COMPOUND_AVERAGE
    /// initial fill; [`Self::predict_leaf`] stamps each trial's
    /// committed §5.11.29 selection over its footprint); read at
    /// compound leaf origins by the decoder's leaf driver.
    compound_types: Vec<u8>,
    /// r415 — the §5.11.5 wedge / diffwtd side-data grids
    /// (`wedge_index` / `wedge_sign` / `mask_type`), stamped alongside
    /// [`Self::compound_types`].
    wedge_indices: Vec<u8>,
    wedge_signs: Vec<u8>,
    mask_types: Vec<u8>,
    /// r417 — the §5.11.28 inter-intra side-data grids the decoder's
    /// leaf driver reads at an inter-intra origin (`RefFrames[.. ][1]
    /// == INTRA_FRAME`): decoded-twin `InterIntraModes[]` /
    /// `WedgeInterIntras[]` / interintra `WedgeIndices[]` (all-zero
    /// pre-fill; [`Self::predict_leaf`] stamps each trial's committed
    /// selection over its footprint).
    interintra_modes: Vec<u8>,
    wedge_interintras: Vec<u8>,
    interintra_wedge_indices: Vec<u8>,
    /// §7.11.3.8 per-cell `LocalWarpParams[ 0..6 ]` slice
    /// ([`crate::GridWarpContext`] contract; r419 —
    /// [`Self::predict_leaf`] stamps a `Warp` trial's fit at the leaf
    /// origin, the only cell the leaf driver reads).
    local_warp: Vec<i32>,
    /// r419 — §7.11.3.8 `LocalValid` bit per cell (twin of the decode
    /// walker's grid; stamped alongside [`Self::local_warp`]).
    local_warp_valid: Vec<u8>,
    /// Full-plane prediction scratch (`reconstruct_inter_leaf_at`
    /// writes at absolute plane coordinates).
    scratch: [Vec<u16>; 3],
    gm_types: [u8; 8],
    gm_flat: [i32; 48],
    /// r413 — `OrderHints[ ref ]` per raw `RefFrame` value (feeds the
    /// §7.11.3.15 order-hint context of the decoder's leaf driver;
    /// only the DISTANCE compound arm reads it, but the values track
    /// the real frame hints so the mirror stays derivation-exact).
    ref_hints: [i32; 8],
    /// r413 — `OrderHintBits` / current-frame `OrderHint` twins of
    /// [`Self::ref_hints`] for the leaf driver's order-hint context.
    hint_bits: u32,
    current_hint: i32,
    no_scaled: [bool; 8],
    /// r413 — §5.9.14 SEG_LVL_ALT_Q deltas per segment (empty =
    /// segmentation disabled). Drives the per-leaf segment policy,
    /// the per-segment residual quantiser, and the §5.11.20 `pred`
    /// inheritance on skip leaves.
    seg_alt_q: Vec<i16>,
    /// §5.9.12 `base_q_idx` (the segment-0 quantiser).
    base_q_idx: u8,
    /// r426 — the exactness-demand mask (row-major mi-cell grid;
    /// `None` = no demand). A leaf overlapping a demanded cell is
    /// FORCED onto the lossless segment ([`Self::lossless_seg`]) on
    /// every arm — inter residual, intra fallback, and the skip-mode
    /// admissibility guard — so the demanded region reconstructs
    /// pixel-exact by the WHT chain's bit-exactness.
    exact_mask: Option<Vec<bool>>,
    /// r426 — content-driven election: when `true`, synthetic leaves
    /// OUTSIDE the demand mask additionally trial the lossless
    /// segment against the lossy depth ladder on twin bits +
    /// distortion.
    auto_lossless: bool,
    /// r426 — the lowest segment whose §7.12.2 qindex clamps to 0
    /// (`None` when the table has no lossless segment; the mask and
    /// auto arms are inert then).
    lossless_seg: Option<u8>,
    /// r426 — the three §5.9.14 inter-override features (ladder item
    /// 8); `None` = no feature segments (the pre-r426 shape).
    extras: Option<SegExtras>,
    /// r426 — §5.9.14 `SegIdPreSkip` (1 whenever any override feature
    /// is active): segment ids code BEFORE skip, so skip leaves keep
    /// their own committed segment instead of the §5.11.9 pred.
    seg_id_pre_skip: bool,
    /// r412 — driver-side write-mirror twin for §7.10.2 MV prediction
    /// (see the struct docs).
    mirror: PartitionWalker,
    /// The §5.9 inter bundle shared with the write pass — the
    /// `find_mv_stack` argument set.
    ip: SyntaxInterFrameParams,
}

impl PSearchCtx {
    /// r412 two-slot P-frame configuration: LAST = `prev`, GOLDEN =
    /// `prevprev`, ladder `{ LAST, GOLDEN }` singles + the one
    /// unidirectional compound pair.
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn new(
        prev: &GopFrameReconYuv,
        prevprev: &GopFrameReconYuv,
        mi_rows: u32,
        mi_cols: u32,
        width: usize,
        height: usize,
        ip: SyntaxInterFrameParams,
        p_index: u32,
        base_q_idx: u8,
        seg_alt_q: &[i16],
    ) -> Result<Self, Error> {
        let last_slot = (p_index & 1) as usize;
        let golden_slot = ((p_index - 1) & 1) as usize;
        let mut slot_to_plane = [0usize; 8];
        slot_to_plane[golden_slot] = 1;
        let mut ref_frame_idx = [last_slot as u8; 7];
        ref_frame_idx[3] = golden_slot as u8;
        Self::with_refs(
            &[prev, prevprev],
            slot_to_plane,
            ref_frame_idx,
            vec![1, 4],
            vec![[1, 4]],
            mi_rows,
            mi_cols,
            width,
            height,
            ip,
            base_q_idx,
            seg_alt_q,
            8,
            1,
            1,
            3,
        )
    }

    /// r415 — the general reference configuration: `refs` are the
    /// distinct reference reconstructions, `slot_to_plane` the §7.20
    /// slot map onto them, `ref_frame_idx` the §5.9.2 header slot map,
    /// and `single_refs` / `compound_pairs` the RD ladder (see the
    /// field docs).
    #[allow(clippy::too_many_arguments)]
    fn with_refs(
        refs: &[&GopFrameReconYuv],
        slot_to_plane: [usize; 8],
        ref_frame_idx: [u8; 7],
        single_refs: Vec<i8>,
        compound_pairs: Vec<[i8; 2]>,
        mi_rows: u32,
        mi_cols: u32,
        width: usize,
        height: usize,
        ip: SyntaxInterFrameParams,
        base_q_idx: u8,
        seg_alt_q: &[i16],
        bit_depth: u8,
        subsampling_x: u8,
        subsampling_y: u8,
        num_planes: u8,
    ) -> Result<Self, Error> {
        if refs.is_empty()
            || slot_to_plane.iter().any(|&p| p >= refs.len())
            || single_refs.iter().any(|&r| !(1..=7).contains(&r))
            || compound_pairs.iter().any(|pr| {
                !single_refs.contains(&pr[0]) || !single_refs.contains(&pr[1]) || pr[0] >= pr[1]
            })
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let cells = (mi_rows as usize) * (mi_cols as usize);
        // r422 — the §5.9.24 model state comes from the shared inter
        // bundle (identity unless the frame-level election committed
        // a model), so the §7.11.3 warp context of `predict_leaf`
        // sees EXACTLY what the write pass and the decoder see.
        let mut gm_flat = [0i32; 48];
        let mut gm_types_u8 = [GM_TYPE_IDENTITY as u8; 8];
        for r in 0..8 {
            gm_flat[r * 6..r * 6 + 6].copy_from_slice(&ip.gm_params[r]);
            gm_types_u8[r] = ip.gm_type[r] as u8;
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
            bit_depth,
            subsampling_x,
            subsampling_y,
            num_planes,
            ref_planes: refs
                .iter()
                .map(|r| [r.y.clone(), r.u.clone(), r.v.clone()])
                .collect(),
            slot_to_plane,
            ref_frame_idx,
            single_refs,
            compound_pairs,
            mi_sizes: vec![BLOCK_4X4; cells],
            is_inters: vec![0u8; cells],
            ref_frames,
            mvs: vec![0i16; cells * 4],
            interp_filters: vec![EIGHTTAP; cells * 2],
            motion_modes: vec![MOTION_MODE_SIMPLE; cells],
            y_modes: vec![0u8; cells],
            avail_us: {
                let mut v = vec![1u8; cells];
                v[..mi_cols as usize].fill(0);
                v
            },
            avail_ls: {
                let mut v = vec![1u8; cells];
                for r in 0..mi_rows as usize {
                    v[r * mi_cols as usize] = 0;
                }
                v
            },
            compound_types: vec![crate::cdf::COMPOUND_AVERAGE; cells],
            wedge_indices: vec![0u8; cells],
            wedge_signs: vec![0u8; cells],
            mask_types: vec![0u8; cells],
            interintra_modes: vec![0u8; cells],
            wedge_interintras: vec![0u8; cells],
            interintra_wedge_indices: vec![0u8; cells],
            local_warp: vec![0i32; cells * 6],
            local_warp_valid: vec![0u8; cells],
            scratch: {
                let (cw, ch) = if num_planes > 1 {
                    (width >> subsampling_x, height >> subsampling_y)
                } else {
                    (0, 0)
                };
                [
                    vec![0u16; width * height],
                    vec![0u16; cw * ch],
                    vec![0u16; cw * ch],
                ]
            },
            gm_types: gm_types_u8,
            gm_flat,
            ref_hints: {
                let mut h = [0i32; 8];
                h.copy_from_slice(&ip.order_hints.order_hints_by_ref);
                h
            },
            hint_bits: ip.order_hints.order_hint_bits,
            current_hint: ip.order_hints.current_order_hint,
            seg_alt_q: seg_alt_q.to_vec(),
            base_q_idx,
            exact_mask: None,
            auto_lossless: false,
            extras: None,
            seg_id_pre_skip: false,
            lossless_seg: {
                let ll = seg_lossless_array(base_q_idx, seg_alt_q);
                (0..seg_alt_q.len()).find(|&s| ll[s]).map(|s| s as u8)
            },
            no_scaled: [false; 8],
            mirror,
            ip,
        })
    }

    /// r413 — per-leaf segment policy: map the input block's luma
    /// mean-absolute-deviation to a segment index (flat blocks take
    /// segment 0, increasingly textured blocks the higher segments).
    /// Any deterministic rule is conformant — the map is simply what
    /// the encoder codes; this one makes every segment reachable on
    /// textured content.
    fn segment_for_block(&self, input: &YuvFrame, mi_row: u32, mi_col: u32, b_size: usize) -> u8 {
        if self.seg_alt_q.is_empty() {
            return 0;
        }
        // r426 — exactness-demand override: a leaf overlapping the
        // demand mask is FORCED onto the lossless segment (the
        // caller validated the table carries one).
        if let Some(ll) = self.lossless_seg {
            if self.demands_exact(mi_row, mi_col, b_size) {
                return ll;
            }
            if self.exact_mask.is_some() || self.auto_lossless {
                // Region/auto configuration: the activity policy maps
                // non-demanded leaves onto the LOSSY prefix only —
                // lossless coding outside the demand is the r426
                // twin-priced election's call, never the policy's.
                let cap = u64::from(ll.saturating_sub(1));
                let act = self.block_activity(input, mi_row, mi_col, b_size);
                return (act / 6).min(cap) as u8;
            }
        }
        let act = self.block_activity(input, mi_row, mi_col, b_size);
        let mut top = (self.seg_alt_q.len() - 1) as u64;
        // r426 — inter-override feature segments (ladder item 8) are
        // TRIAL-elected, never activity-assigned: their blocks carry
        // forced §5.11.10/§5.11.11/§5.11.23/§5.11.25 derivations the
        // plain residual ladder does not honour. The activity policy
        // maps onto the feature-free prefix only.
        if let Some(x) = self.extras.as_ref() {
            let first_feature = (0..self.seg_alt_q.len())
                .find(|&s| x.ref_frame[s].is_some() || x.skip[s] || x.globalmv[s])
                .unwrap_or(self.seg_alt_q.len());
            top = top.min(first_feature.saturating_sub(1) as u64);
        }
        (act / 6).min(top) as u8
    }

    /// r413 — the activity metric behind [`Self::segment_for_block`]:
    /// the input block's luma mean-absolute-deviation.
    fn block_activity(&self, input: &YuvFrame, mi_row: u32, mi_col: u32, b_size: usize) -> u64 {
        let (bw, bh) = (
            NUM_4X4_BLOCKS_WIDE[b_size] * 4,
            NUM_4X4_BLOCKS_HIGH[b_size] * 4,
        );
        let (row0, col0) = ((mi_row as usize) * 4, (mi_col as usize) * 4);
        let w = input.width as usize;
        let mut sum = 0u64;
        for i in 0..bh {
            for j in 0..bw {
                sum += u64::from(input.y[(row0 + i) * w + (col0 + j)]);
            }
        }
        let n = (bw * bh) as u64;
        let mean = sum / n;
        let mut mad = 0u64;
        for i in 0..bh {
            for j in 0..bw {
                let v = u64::from(input.y[(row0 + i) * w + (col0 + j)]);
                mad += v.abs_diff(mean);
            }
        }
        mad / n
    }

    /// r413 — the quantiser bundle for one segment: §7.12.2
    /// `get_qindex( seg ) = Clip3( 0, 255, base_q_idx + alt_q[ seg ] )`
    /// with the frame's zero per-plane deltas (the encode-side
    /// residual chain quantises at exactly the q-index the §5.11.39
    /// reader will dequantise with).
    fn seg_qp(&self, frame_qp: &QuantizerParams, segment_id: u8) -> QuantizerParams {
        if self.seg_alt_q.is_empty() || segment_id == 0 {
            return *frame_qp;
        }
        let q = (i32::from(self.base_q_idx) + i32::from(self.seg_alt_q[segment_id as usize]))
            .clamp(0, 255) as u8;
        QuantizerParams::neutral(q, 8)
    }

    /// r426 — `LosslessArray[ segment_id ]` for this GOP's frames
    /// (see [`seg_lossless_array`]): `true` exactly when the block's
    /// §7.12.2 qindex clamps to 0, flipping the §5.11.15 `TX_4X4` /
    /// §7.13.2.10 WHT leaf semantics for that block.
    fn seg_lossless(&self, segment_id: u8) -> bool {
        let s = segment_id as usize;
        let q = if self.seg_alt_q.is_empty() || s >= self.seg_alt_q.len() {
            i32::from(self.base_q_idx)
        } else {
            (i32::from(self.base_q_idx) + i32::from(self.seg_alt_q[s])).clamp(0, 255)
        };
        q == 0
    }

    /// r426 — synthetic-content detector for the auto-lossless
    /// election: `true` when the input block's luma alphabet is small
    /// (`<= 16` distinct values — flat fills, text glyphs, UI edges,
    /// lightly antialiased text). The bound deliberately exceeds the
    /// §5.11.46 `PALETTE_COLORS = 8` ceiling: at `<= 8` the
    /// screen-content palette already codes the block exactly, so the
    /// lossless segment's unique territory starts where the palette
    /// alphabet ends. Pure gate: it only bounds which leaves PAY for
    /// the extra lossless trial; the election itself is twin bits +
    /// distortion.
    fn synthetic_leaf(&self, input: &YuvFrame, mi_row: u32, mi_col: u32, b_size: usize) -> bool {
        let (bw, bh) = (
            NUM_4X4_BLOCKS_WIDE[b_size] * 4,
            NUM_4X4_BLOCKS_HIGH[b_size] * 4,
        );
        let (row0, col0) = ((mi_row as usize) * 4, (mi_col as usize) * 4);
        let w = input.width as usize;
        let mut seen = [false; 256];
        let mut distinct = 0usize;
        for i in 0..bh {
            for j in 0..bw {
                let v = input.y[(row0 + i) * w + (col0 + j)] as usize;
                if !seen[v] {
                    seen[v] = true;
                    distinct += 1;
                    if distinct > 16 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// r426 — `true` iff the leaf at `(mi_row, mi_col, b_size)`
    /// overlaps a demanded cell of the exactness mask. Demanded
    /// leaves are forced onto the lossless segment on every arm.
    fn demands_exact(&self, mi_row: u32, mi_col: u32, b_size: usize) -> bool {
        let Some(mask) = self.exact_mask.as_ref() else {
            return false;
        };
        let bh4 = NUM_4X4_BLOCKS_HIGH[b_size] as u32;
        let bw4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
        let r1 = (mi_row + bh4).min(self.mi_rows);
        let c1 = (mi_col + bw4).min(self.mi_cols);
        (mi_row..r1).any(|r| (mi_col..c1).any(|c| mask[(r * self.mi_cols + c) as usize]))
    }

    /// r413 — the §5.11.20 `pred` cascade over the mirror's
    /// `SegmentIds[]` (the value a `skip == 1` leaf inherits with no
    /// bits; must be derived at SEARCH time from exactly the state
    /// the write pass will hold).
    fn segment_pred(&self, mi_row: u32, mi_col: u32) -> u8 {
        let at = |r: i64, c: i64| -> i32 {
            if r < 0 || c < 0 || r >= i64::from(self.mi_rows) || c >= i64::from(self.mi_cols) {
                return -1;
            }
            self.mirror.segment_ids()[(r as u32 * self.mi_cols + c as u32) as usize]
        };
        let (r, c) = (i64::from(mi_row), i64::from(mi_col));
        let prev_ul = if r > 0 && c > 0 { at(r - 1, c - 1) } else { -1 };
        let prev_u = if r > 0 { at(r - 1, c) } else { -1 };
        let prev_l = if c > 0 { at(r, c - 1) } else { -1 };
        #[allow(clippy::if_same_then_else)]
        let pred: i32 = if prev_u == -1 {
            if prev_l == -1 {
                0
            } else {
                prev_l
            }
        } else if prev_l == -1 {
            prev_u
        } else if prev_ul == prev_u {
            prev_u
        } else {
            prev_l
        };
        pred as u8
    }

    /// r415 — resolve a raw `RefFrame` ordinal to its
    /// [`Self::ref_planes`] index through the header slot map.
    fn plane_of_ref(&self, rf: i8) -> usize {
        self.slot_to_plane[self.ref_frame_idx[(rf - 1) as usize] as usize]
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

    /// r419 — rect snapshot of every §5.11.5 driver grid (the slices
    /// [`Self::predict_leaf`] / [`Self::stamp_intra_leaf_grids`]
    /// stamp and the decoder's leaf driver reads). The r419 OBMC
    /// neighbour scan reads committed ABOVE/LEFT cells through these
    /// grids, so — like the mirror — losing trials must roll back:
    /// [`build_p_search_tree`] pairs every mirror snapshot/restore
    /// with a driver-grid one.
    fn snapshot_grids_rect(&self, mi_row: u32, mi_col: u32, n4: u32) -> DriverGridSnapshot {
        let r1 = (mi_row + n4).min(self.mi_rows);
        let c1 = (mi_col + n4).min(self.mi_cols);
        let mut cells = Vec::with_capacity(((r1 - mi_row) * (c1 - mi_col)) as usize);
        for rr in mi_row..r1 {
            for cc in mi_col..c1 {
                let cell = (rr * self.mi_cols + cc) as usize;
                cells.push(DriverGridCell {
                    mi_size: self.mi_sizes[cell],
                    is_inter: self.is_inters[cell],
                    ref_frames: [self.ref_frames[cell * 2], self.ref_frames[cell * 2 + 1]],
                    mvs: [
                        self.mvs[cell * 4],
                        self.mvs[cell * 4 + 1],
                        self.mvs[cell * 4 + 2],
                        self.mvs[cell * 4 + 3],
                    ],
                    interp_filters: [
                        self.interp_filters[cell * 2],
                        self.interp_filters[cell * 2 + 1],
                    ],
                    motion_mode: self.motion_modes[cell],
                    y_mode: self.y_modes[cell],
                    compound_type: self.compound_types[cell],
                    wedge_index: self.wedge_indices[cell],
                    wedge_sign: self.wedge_signs[cell],
                    mask_type: self.mask_types[cell],
                    interintra_mode: self.interintra_modes[cell],
                    wedge_interintra: self.wedge_interintras[cell],
                    interintra_wedge_index: self.interintra_wedge_indices[cell],
                    local_warp: {
                        let mut w = [0i32; 6];
                        w.copy_from_slice(&self.local_warp[cell * 6..cell * 6 + 6]);
                        w
                    },
                    local_warp_valid: self.local_warp_valid[cell],
                });
            }
        }
        DriverGridSnapshot {
            mi_row,
            mi_col,
            r1,
            c1,
            cells,
        }
    }

    /// Restore a [`Self::snapshot_grids_rect`] snapshot.
    fn restore_grids_rect(&mut self, snap: &DriverGridSnapshot) {
        let mut it = snap.cells.iter();
        for rr in snap.mi_row..snap.r1 {
            for cc in snap.mi_col..snap.c1 {
                let cell = (rr * self.mi_cols + cc) as usize;
                let s = it.next().expect("snapshot covers the rect");
                self.mi_sizes[cell] = s.mi_size;
                self.is_inters[cell] = s.is_inter;
                self.ref_frames[cell * 2] = s.ref_frames[0];
                self.ref_frames[cell * 2 + 1] = s.ref_frames[1];
                self.mvs[cell * 4..cell * 4 + 4].copy_from_slice(&s.mvs);
                self.interp_filters[cell * 2] = s.interp_filters[0];
                self.interp_filters[cell * 2 + 1] = s.interp_filters[1];
                self.motion_modes[cell] = s.motion_mode;
                self.y_modes[cell] = s.y_mode;
                self.compound_types[cell] = s.compound_type;
                self.wedge_indices[cell] = s.wedge_index;
                self.wedge_signs[cell] = s.wedge_sign;
                self.mask_types[cell] = s.mask_type;
                self.interintra_modes[cell] = s.interintra_mode;
                self.wedge_interintras[cell] = s.wedge_interintra;
                self.interintra_wedge_indices[cell] = s.interintra_wedge_index;
                self.local_warp[cell * 6..cell * 6 + 6].copy_from_slice(&s.local_warp);
                self.local_warp_valid[cell] = s.local_warp_valid;
            }
        }
    }

    /// r419 — the §5.11.27 WARPED_CAUSAL fit for one leaf: the
    /// §7.10.4 `find_warp_samples` scan against the driver mirror
    /// (the identical scan the write pass validates `NumSamples`
    /// with and the decode walker re-runs at its in-walk
    /// `compute_prediction( )` position), the §7.11.3.8
    /// `warp_estimation` least squares, and the §7.11.3.6
    /// `setup_shear` validity check. `Some(params)` iff the scan
    /// found samples AND the fit is shear-valid — the only
    /// configuration where committing `WARPED_CAUSAL` changes the
    /// §7.11.3.1 step-7 prediction.
    fn warp_fit(
        &self,
        mi_row: u32,
        mi_col: u32,
        b_size: usize,
        ref0: i8,
        mv: [[i32; 2]; 2],
    ) -> Option<[i32; 6]> {
        let mut cands: Vec<crate::inter_pred::WarpSampleCand> = Vec::new();
        let ns = self.mirror.find_warp_samples_list(
            mi_row,
            mi_col,
            b_size,
            i32::from(ref0),
            mv,
            &mut cands,
        );
        if ns == 0 {
            return None;
        }
        let lw = crate::inter_pred::warp_estimation(
            &cands,
            mi_row as i32,
            mi_col as i32,
            NUM_4X4_BLOCKS_WIDE[b_size] as i32,
            NUM_4X4_BLOCKS_HIGH[b_size] as i32,
            [mv[0][0] as i16, mv[0][1] as i16],
        );
        if !lw.local_valid {
            return None;
        }
        match crate::inter_pred::setup_shear(lw.local_warp_params) {
            Some(s) if s.warp_valid => Some(lw.local_warp_params),
            _ => None,
        }
    }

    /// r421 — `NumSamples` from the same §7.10.4 `find_warp_samples`
    /// scan [`Self::warp_fit`] runs, without the fit: the §5.11.27
    /// arm-A/arm-B dispatch operand (`num_samples == 0` selects the
    /// `use_obmc` arm even when warp is enabled), needed to price the
    /// motion-mode symbol through the writer's own arm derivation.
    fn warp_num_samples(
        &self,
        mi_row: u32,
        mi_col: u32,
        b_size: usize,
        ref0: i8,
        mv: [[i32; 2]; 2],
    ) -> u32 {
        let mut cands: Vec<crate::inter_pred::WarpSampleCand> = Vec::new();
        self.mirror
            .find_warp_samples_list(mi_row, mi_col, b_size, i32::from(ref0), mv, &mut cands)
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
                skip_mode: ib.skip_mode,
                segment_id: block.segment_id,
                is_inter: 1,
                y_mode: ib.y_mode,
                // r417 — the §5.11.28 imperative override the decode
                // walker (and the write pass) stamp: `RefFrame[ 1 ] =
                // INTRA_FRAME` on the `interintra == 1` arm.
                ref_frame: if ib.interintra_mode.is_some() {
                    [
                        ib.ref_frame[0],
                        crate::uncompressed_header_tail::INTRA_FRAME as i8,
                    ]
                } else {
                    ib.ref_frame
                },
                mv: ib.mv[0],
                mv2: ib.mv[1],
                interp_filter: ib.interp_filter,
                // r419 — the committed §5.11.27 ordinal (the write
                // pass stamps the identical value).
                motion_mode: ib.motion_mode,
                palette_size_y: 0,
                palette_colors_y: &[],
                palette_size_uv: 0,
                palette_colors_u: &[],
                palette_colors_v: &[],
                cdef: None,
                tx_size: tx_pre,
                // r415 — the SAME §5.11.29 stamp values the write
                // pass commits for this leaf (search mirror and write
                // mirror must observe identical neighbour grids).
                // r416: only the MASKED ordinals set
                // `comp_group_idx = 1`; COMPOUND_DISTANCE stays on
                // the `comp_group_idx = 0` arm with `compound_idx = 0`.
                comp_group_idx: u8::from(
                    ib.ref_frame[1] > 0
                        && ib.skip_mode == 0
                        && matches!(
                            ib.compound_type,
                            crate::inter_pred::COMPOUND_WEDGE | crate::inter_pred::COMPOUND_DIFFWTD
                        ),
                ),
                compound_idx: u8::from(
                    !(ib.ref_frame[1] > 0
                        && ib.skip_mode == 0
                        && ib.compound_type == crate::inter_pred::COMPOUND_DISTANCE),
                ),
                compound_type: if ib.interintra_mode.is_some()
                    || (ib.ref_frame[1] > 0 && ib.skip_mode == 0)
                {
                    // r417 — inter-intra leaves carry the §5.11.29
                    // bit-silent derivation in `ib.compound_type`
                    // (the write pass validates the exact value).
                    ib.compound_type
                } else {
                    crate::inter_pred::COMPOUND_AVERAGE
                },
                wedge_index: ib.wedge_index,
                wedge_sign: ib.wedge_sign,
                mask_type: ib.mask_type,
                // r417 — §5.11.28 side-data stamps (decode-walker
                // grid-fill twins).
                interintra_mode: ib.interintra_mode.unwrap_or(0),
                wedge_interintra: u8::from(
                    ib.interintra_mode.is_some() && ib.wedge_interintra == 1,
                ),
                interintra_wedge_index: ib.interintra_wedge_index,
            },
            None => EncoderBlockSyntaxStamp {
                mi_row,
                mi_col,
                sub_size: b_size,
                skip: block.skip,
                skip_mode: 0,
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
                // r415 §5.11.29 compound defaults (AVERAGE, pre-set
                // comp_group_idx/compound_idx, no side data).
                comp_group_idx: 0,
                compound_idx: 1,
                compound_type: crate::inter_pred::COMPOUND_AVERAGE,
                wedge_index: 0,
                wedge_sign: 0,
                mask_type: 0,
                interintra_mode: 0,
                wedge_interintra: 0,
                interintra_wedge_index: 0,
            },
        };
        self.mirror.stamp_encoder_block_syntax(&stamp);
    }

    /// r417 — driver-grid stamps for a committed §5.11.22 INTRA leaf
    /// (the decode walker's intra grid-fill twins). `predict_leaf`
    /// stamps only inter trials, so an intra winner must overwrite
    /// the losing trial's stamps: the §5.11.33 `someUseIntra` scan
    /// reads `RefFrames[ .. ][ 0 ] == INTRA_FRAME` at the
    /// neighbouring 2×2-group cells through THESE grids when a later
    /// sub-8 inter leaf predicts its group chroma.
    fn stamp_intra_leaf_grids(&mut self, mi_row: u32, mi_col: u32, b_size: usize, y_mode: u8) {
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
                self.is_inters[cell] = 0;
                // §5.11.22: `RefFrame[ 0 ] = INTRA_FRAME`,
                // `RefFrame[ 1 ] = NONE`.
                self.ref_frames[cell * 2] = 0;
                self.ref_frames[cell * 2 + 1] = -1;
                self.mvs[cell * 4] = 0;
                self.mvs[cell * 4 + 1] = 0;
                self.mvs[cell * 4 + 2] = 0;
                self.mvs[cell * 4 + 3] = 0;
                self.interp_filters[cell * 2] = EIGHTTAP;
                self.interp_filters[cell * 2 + 1] = EIGHTTAP;
                self.motion_modes[cell] = MOTION_MODE_SIMPLE;
                self.y_modes[cell] = y_mode;
                self.compound_types[cell] = crate::inter_pred::COMPOUND_AVERAGE;
                self.wedge_indices[cell] = 0;
                self.wedge_signs[cell] = 0;
                self.mask_types[cell] = 0;
                self.interintra_modes[cell] = 0;
                self.wedge_interintras[cell] = 0;
                self.interintra_wedge_indices[cell] = 0;
            }
        }
    }

    /// Stamp the leaf's §5.11.5 grid footprint (the same values the
    /// write mirror / decode walker stamp), then run the decoder's
    /// §5.11.33 leaf driver into the scratch planes.
    ///
    /// r417 — `ii: Some(_)` selects the §5.11.28 inter-intra arm: the
    /// stamps mirror the decode walker's §5.11.28 imperative override
    /// (`RefFrame[ 1 ] = INTRA_FRAME`, the §5.11.29 bit-silent
    /// `compound_type` derivation, the inter-intra side-data grids),
    /// the §7.11.5-translated intra half is predicted into the
    /// scratch planes FIRST (the §5.11.33 order: `predict_intra`
    /// before `predict_inter`), and the leaf driver's §7.11.3.14
    /// blend consumes it in place. `ref_frame[ 1 ]` must be `-1`
    /// (the §5.11.28 gate requires a single-reference leaf).
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
        comp: CompoundSel,
        ii: Option<InterIntraTrial<'_>>,
        motion: MotionSel,
    ) -> Result<(), Error> {
        if ii.is_some() && ref_frame[1] != -1 {
            return Err(Error::PartitionWalkOutOfRange);
        }
        // r419 — §5.11.27: a non-SIMPLE motion mode exists only on a
        // single-reference non-inter-intra leaf.
        if motion != MotionSel::Simple && (ref_frame[1] != -1 || ii.is_some()) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let bw4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[b_size] as u32;
        // §5.11.28 imperative overrides the decode walker stamps on
        // the `interintra == 1` arm: `RefFrame[ 1 ] = INTRA_FRAME`
        // plus the §5.11.29 bit-silent `compound_type` derivation
        // (`wedge_interintra ? COMPOUND_WEDGE : COMPOUND_INTRA`).
        let stamp_rf1: i8 = if ii.is_some() {
            crate::uncompressed_header_tail::INTRA_FRAME as i8
        } else {
            ref_frame[1]
        };
        let stamp_ctype: u8 = match &ii {
            Some(t) if t.wedge.is_some() => crate::inter_pred::COMPOUND_WEDGE,
            Some(_) => crate::inter_pred::COMPOUND_INTRA,
            None => comp.ctype,
        };
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
                self.ref_frames[cell * 2 + 1] = stamp_rf1;
                self.mvs[cell * 4] = mv[0][0] as i16;
                self.mvs[cell * 4 + 1] = mv[0][1] as i16;
                self.mvs[cell * 4 + 2] = mv[1][0] as i16;
                self.mvs[cell * 4 + 3] = mv[1][1] as i16;
                self.interp_filters[cell * 2] = filter;
                self.interp_filters[cell * 2 + 1] = filter;
                // r419 — the committed §5.11.27 ordinal over the
                // footprint (the decode walker's grid-fill twin).
                self.motion_modes[cell] = motion.ordinal();
                self.y_modes[cell] = y_mode;
                // r415 — §5.11.29 side-data stamps (the decoder's leaf
                // driver dispatches WEDGE / DIFFWTD masks from these).
                self.compound_types[cell] = stamp_ctype;
                self.wedge_indices[cell] = comp.wedge_index;
                self.wedge_signs[cell] = comp.wedge_sign;
                self.mask_types[cell] = comp.mask_type;
                // r417 — §5.11.28 inter-intra side-data stamps (the
                // decode-walker grid-fill twins; all-zero outside the
                // inter-intra arm).
                self.interintra_modes[cell] = ii.as_ref().map_or(0, |t| t.mode);
                self.wedge_interintras[cell] =
                    u8::from(ii.as_ref().is_some_and(|t| t.wedge.is_some()));
                self.interintra_wedge_indices[cell] =
                    ii.as_ref().and_then(|t| t.wedge).unwrap_or(0);
            }
        }

        // r419 — §7.11.3.8 per-cell warp-fit stamps at the leaf
        // origin (the only cell the leaf driver's warp context
        // reads), mirroring the decode walker's in-walk
        // `compute_prediction( )` stamp.
        {
            let origin = (mi_row * self.mi_cols + mi_col) as usize;
            if let MotionSel::Warp(p) = motion {
                self.local_warp[origin * 6..origin * 6 + 6].copy_from_slice(&p);
                self.local_warp_valid[origin] = 1;
            } else {
                self.local_warp[origin * 6..origin * 6 + 6].fill(0);
                self.local_warp_valid[origin] = 0;
            }
        }

        // r417 — §5.11.33 `IsInterIntra` arm, intra half FIRST: run
        // the §7.11.5-translated `predict_intra` into the scratch
        // planes over the whole per-plane block region (av1-spec
        // §5.11.33 line 5146: one call at `(baseX, baseY, log2W,
        // log2H)`), reading reconstructed neighbours from the
        // committed recon — the decode walker's `CurrFrame[ plane ]`
        // twin. The §7.11.3.14 blend below reads it back as `pred1`.
        if let Some(t) = &ii {
            // §5.11.33 `interintra_mode → mode` translation
            // (av1-spec p.82 lines 5142-5145).
            let mode = match t.mode {
                crate::cdf::II_V_PRED => crate::cdf::V_PRED,
                crate::cdf::II_H_PRED => crate::cdf::H_PRED,
                crate::cdf::II_DC_PRED => crate::cdf::DC_PRED,
                _ => crate::cdf::SMOOTH_PRED,
            };
            // Single-tile §5.11.5 availability: `AvailU = MiRow >
            // MiRowStart`, `AvailL = MiCol > MiColStart`; the chroma
            // pair equals the luma pair on every §5.11.28-eligible
            // block (`bw4 >= 2 && bh4 >= 2` skips the §5.11.5 sub-8
            // chroma fix-ups).
            let avail_u = mi_row > 0;
            let avail_l = mi_col > 0;
            for plane in 0..usize::from(self.num_planes) {
                let (sub_x, sub_y): (u8, u8) = if plane > 0 {
                    (self.subsampling_x, self.subsampling_y)
                } else {
                    (0, 0)
                };
                let plane_sz =
                    crate::cdf::get_plane_residual_size(b_size, plane as u8, sub_x, sub_y)
                        .ok_or(Error::PartitionWalkOutOfRange)?;
                let w = NUM_4X4_BLOCKS_WIDE[plane_sz] * 4;
                let h = NUM_4X4_BLOCKS_HIGH[plane_sz] * 4;
                let tx_sz = crate::cdf::find_tx_size(w, h).ok_or(Error::PartitionWalkOutOfRange)?;
                let base_x = ((mi_col >> sub_x) * 4) as usize;
                let base_y = ((mi_row >> sub_y) * 4) as usize;
                // §5.11.33 `BlockDecoded[ ]` above-right / below-left
                // reads against the encoder's own §5.11.3 mirror.
                // Provably inert for the four §6.10.27 II modes (DC /
                // V@0 / H@0 / SMOOTH never read the extended
                // neighbour segments), threaded for §7.11.2 fidelity.
                let (have_ar, have_bl) =
                    super::key_frame::tu_corner_avail(&t.neigh.bd, plane, base_x, base_y, w, h);
                let (buf, pw, ph) = t.neigh.plane(plane);
                let read = |yy: u32, xx: u32| -> u16 { buf[(yy as usize) * pw + (xx as usize)] };
                let out = &mut self.scratch[plane];
                // §7.11.2.4 step-4 pre-pass inputs: inert on the II
                // mode set (no directional D-mode reachable — V/H ride
                // the exact-90°/180° copies at `angle_delta == 0`), so
                // the neutral pair matches the decode walker's output
                // sample-for-sample regardless of the sequence's
                // `enable_intra_edge_filter`.
                PartitionWalker::predict_intra_into_u16_plane(
                    &read,
                    out,
                    pw,
                    (pw - 1) as u32,
                    (ph - 1) as u32,
                    base_x as u32,
                    base_y as u32,
                    tx_sz,
                    mode,
                    /* angle_delta = */ 0,
                    self.bit_depth,
                    avail_u,
                    avail_l,
                    have_ar,
                    have_bl,
                    /* enable_intra_edge_filter = */ false,
                    /* filter_type = */ 0,
                    /* filter_intra_mode = */ None,
                );
            }
        }

        // Field-split borrows: the grid views borrow the mode-info
        // vectors immutably while the plane contexts borrow the
        // scratch planes mutably.
        let (luma_w, luma_h) = (self.luma_w, self.luma_h);
        let (mi_rows, mi_cols) = (self.mi_rows, self.mi_cols);
        let slot_to_plane = self.slot_to_plane;
        let ref_frame_idx = self.ref_frame_idx;
        // r427 — format scalars (copied out before the field split).
        let bit_depth = self.bit_depth;
        let (ssx, ssy) = (self.subsampling_x, self.subsampling_y);
        let num_planes = usize::from(self.num_planes);
        let PSearchCtx {
            ref_planes,
            compound_types,
            wedge_indices,
            wedge_signs,
            mask_types,
            interintra_modes,
            wedge_interintras,
            interintra_wedge_indices,
            mi_sizes,
            is_inters,
            ref_frames,
            mvs,
            interp_filters,
            motion_modes,
            y_modes,
            avail_us,
            avail_ls,
            local_warp,
            local_warp_valid,
            scratch,
            gm_types,
            gm_flat,
            ref_hints,
            hint_bits,
            current_hint,
            no_scaled,
            ..
        } = &mut *self;

        // §7.20 `FrameStore` views — r415: each slot resolves through
        // `slot_to_plane` to its distinct reference reconstruction
        // (slots outside the mapped roles point at plane 0, never
        // resolved through `ref_frame_idx`). Dimensions are LUMA
        // extents per the r405 contract; strides are plane samples.
        let store_y = make_store(
            ref_planes,
            &slot_to_plane,
            0,
            luma_w as usize,
            luma_w,
            luma_h,
        );
        let store_u = make_store(
            ref_planes,
            &slot_to_plane,
            1,
            (luma_w as usize) >> ssx,
            luma_w,
            luma_h,
        );
        let store_v = make_store(
            ref_planes,
            &slot_to_plane,
            2,
            (luma_w as usize) >> ssx,
            luma_w,
            luma_h,
        );

        let grid = InterModeInfoGrid {
            mi_sizes,
            is_inters,
            ref_frames,
            mvs,
            interp_filters,
            compound_types,
            wedge_indices,
            wedge_signs,
            mask_types,
            interintra_modes,
            wedge_interintras,
            interintra_wedge_indices,
            order_hint_bits: *hint_bits,
            current_order_hint: *current_hint,
            order_hints_by_ref: ref_hints,
            mi_rows,
            mi_cols,
            bit_depth,
            // §7.11.3.1 step-7 warp context — identity global motion;
            // r419: a `WARPED_CAUSAL` trial carries its §7.11.3.8 fit
            // in the per-cell slices stamped above (SIMPLE / OBMC
            // leaves keep `useWarp = 0`).
            warp: Some(crate::GridWarpContext {
                motion_modes,
                local_warp_params: local_warp,
                local_warp_valid,
                y_modes,
                gm_types,
                gm_params: gm_flat,
                force_integer_mv: false,
                is_scaled: no_scaled,
            }),
            // r419 — §7.11.3.9 OBMC context: an `OBMC` trial routes
            // through the decoder's own neighbour-scan dispatch over
            // the committed grids (the decode frame walk threads the
            // identical trio).
            obmc: Some(crate::GridObmcContext {
                motion_modes,
                avail_u: avail_us,
                avail_l: avail_ls,
            }),
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
                subsampling_x: ssx,
                subsampling_y: ssy,
                frame_store: &store_u,
                frame_width: luma_w,
                frame_height: luma_h,
                curr: s1,
                curr_stride: (luma_w as usize) >> ssx,
            },
            PlaneReconContext {
                plane: 2,
                subsampling_x: ssx,
                subsampling_y: ssy,
                frame_store: &store_v,
                frame_width: luma_w,
                frame_height: luma_h,
                curr: s2,
                curr_stride: (luma_w as usize) >> ssx,
            },
        ];
        reconstruct_inter_leaf_at(
            &grid,
            &ref_frame_idx,
            &mut planes[..num_planes],
            mi_row as usize,
            mi_col as usize,
        )
    }
}

/// One plane's §7.20 `FrameStore` view (r415 general form): slot `s`
/// resolves to `ref_planes[ slot_to_plane[ s ] ][ plane ]` (extents in
/// LUMA samples per the r405 contract, stride in this plane's own
/// samples). Slots outside the frame's mapped roles are never resolved
/// through `ref_frame_idx`, so their content is immaterial.
fn make_store<'a>(
    ref_planes: &'a [[Vec<u16>; 3]],
    slot_to_plane: &[usize; 8],
    plane: usize,
    stride: usize,
    luma_w: u32,
    luma_h: u32,
) -> [RefFrameStoreEntry<'a>; 8] {
    core::array::from_fn(|slot| RefFrameStoreEntry {
        plane: &ref_planes[slot_to_plane[slot]][plane],
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
    input: &YuvFrame,
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
    input: &YuvFrame,
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
            CompoundSel::AVERAGE,
            None,
            MotionSel::Simple,
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
#[allow(clippy::too_many_arguments)]
fn encode_inter_leaf(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    ictx: &mut PSearchCtx,
    // r421 — `Some((twin, params))` prices the leaf-level elections
    // (tx-depth ladder, §5.11.10 skip-mode trial) with exact twin
    // bits; `None` keeps the pre-r421 heuristics.
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    let leaf = encode_inter_leaf_base(mi_r, mi_c, b_size, input, recon, ictx, pricing)?;
    // r426 — ladder item 8: the §5.9.14 inter-override feature-segment
    // trials (SEG_LVL_SKIP / SEG_LVL_GLOBALMV / SEG_LVL_REF_FRAME)
    // against the fully-searched winner, on >= 8x8 leaves.
    if ictx.extras.is_some()
        && crate::cdf::block_width(b_size) >= 8
        && crate::cdf::block_height(b_size) >= 8
    {
        return seg_extras_trials(mi_r, mi_c, b_size, input, recon, ictx, pricing, leaf);
    }
    Ok(leaf)
}

/// The pre-r426 [`encode_inter_leaf`] body: the fully-searched leaf
/// plus the §5.11.10 skip-mode trial.
#[allow(clippy::too_many_arguments)]
fn encode_inter_leaf_base(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    ictx: &mut PSearchCtx,
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    let leaf = encode_inter_leaf_modes(mi_r, mi_c, b_size, input, recon, ictx, pricing)?;

    // r413 — §5.11.10 skip-mode trial: on a skip-mode frame, every
    // >= 8×8 leaf additionally trials the pure-derivation skip-mode
    // block (COMPOUND_AVERAGE over SkipModeFrame[] = { LAST, GOLDEN }
    // at the compound-stack NEARESTMV pair, EIGHTTAP, `skip = 1`, no
    // residual — ONE §5.11.10 S() total) against the fully-searched
    // leaf. The comparison is exact on both sides: the normal leaf's
    // residual-coded reconstruction vs the skip-mode leaf's bare
    // prediction, each scored `D + λ·R` over all three planes.
    if !ictx.ip.skip_mode_present
        || crate::cdf::block_width(b_size) < 8
        || crate::cdf::block_height(b_size) < 8
    {
        return Ok(leaf);
    }
    let sm_ref: [i8; 2] = ictx.ip.skip_mode_frame;
    let stack = ictx.find_stack(mi_r, mi_c, b_size, sm_ref)?;
    let sm_mv = [stack.ref_stack_mv[0][0], stack.ref_stack_mv[0][1]];
    if sm_mv
        .iter()
        .any(|m| m[0].unsigned_abs() >= (1 << 14) || m[1].unsigned_abs() >= (1 << 14))
    {
        return Ok(leaf);
    }
    let bw4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let bh4 = NUM_4X4_BLOCKS_HIGH[b_size];
    let lambda = lambda_for(&recon.qp);
    // r421 — twin-priced or heuristic, one scale per call (heuristic
    // rates scaled by 256 so both run the same comparison with
    // decisions identical to the pre-r421 integer scores).
    let rate_of = |blk: &SyntaxBlock| -> Result<u64, Error> {
        match pricing {
            Some((twin, params)) => twin.price_block(blk, mi_r, mi_c, b_size, params),
            None => Ok(p_leaf_rate(blk) * 256),
        }
    };
    let d_normal = region_distortion_wh(recon, input, mi_r, mi_c, bw4, bh4);
    let score_normal = score256(d_normal, lambda, rate_of(&leaf)?);
    let after_normal = save_region_wh(recon, mi_r, mi_c, bw4, bh4);

    // Predict the skip-mode leaf (re-stamps the grid footprint) and
    // stitch the bare prediction into the recon.
    ictx.predict_leaf(
        mi_r,
        mi_c,
        b_size,
        sm_ref,
        MODE_NEAREST_NEARESTMV,
        sm_mv,
        EIGHTTAP,
        CompoundSel::AVERAGE,
        None,
        MotionSel::Simple,
    )?;
    let (bw, bh) = (bw4 * 4, bh4 * 4);
    let (row0, col0) = ((mi_r as usize) * 4, (mi_c as usize) * 4);
    // r427 — subsampling-derived chroma stitch footprint (empty on
    // the monochrome arm).
    let (crow0, ccol0) = recon.chroma_origin(mi_r, mi_c);
    let (cbw, cbh) = if recon.num_planes > 1 {
        (bw >> recon.subsampling_x, bh >> recon.subsampling_y)
    } else {
        (0, 0)
    };
    let width = recon.width;
    let cw = recon.chroma_w;
    for i in 0..bh {
        for j in 0..bw {
            recon.y[(row0 + i) * width + (col0 + j)] =
                ictx.scratch[0][(row0 + i) * width + (col0 + j)];
        }
    }
    for i in 0..cbh {
        for j in 0..cbw {
            recon.u[(crow0 + i) * cw + (ccol0 + j)] =
                ictx.scratch[1][(crow0 + i) * cw + (ccol0 + j)];
            recon.v[(crow0 + i) * cw + (ccol0 + j)] =
                ictx.scratch[2][(crow0 + i) * cw + (ccol0 + j)];
        }
    }
    tu_bd_stamp(&mut recon.bd, 0, col0, row0, bw, bh);
    if recon.num_planes > 1 {
        tu_bd_stamp(&mut recon.bd, 1, ccol0, crow0, cbw, cbh);
        tu_bd_stamp(&mut recon.bd, 2, ccol0, crow0, cbw, cbh);
    }

    let mut sm_leaf = SyntaxBlock::skip_leaf(0, None);
    // §5.11.19/§5.11.20: skip leaves inherit the segment pred cascade
    // (r426: except under SegIdPreSkip, where the id codes pre-skip —
    // skip-mode blocks then commit segment 0, whose feature slots are
    // empty by the entry validation, keeping §5.11.10 codable).
    sm_leaf.segment_id = if ictx.seg_id_pre_skip {
        0
    } else {
        ictx.segment_pred(mi_r, mi_c)
    };
    sm_leaf.inter = Some(SyntaxInterBlock {
        ref_frame: sm_ref,
        y_mode: MODE_NEAREST_NEARESTMV,
        mv: sm_mv,
        ref_mv_idx: 0,
        interp_filter: [EIGHTTAP; 2],
        skip_mode: 1,
        compound_type: crate::inter_pred::COMPOUND_AVERAGE,
        wedge_index: 0,
        wedge_sign: 0,
        mask_type: 0,
        interintra_mode: None,
        wedge_interintra: 0,
        interintra_wedge_index: 0,
        // §5.11.27 line 1: skip-mode forces SIMPLE with no bits.
        motion_mode: MOTION_MODE_SIMPLE,
    });
    let d_sm = region_distortion_wh(recon, input, mi_r, mi_c, bw4, bh4);
    let score_sm = score256(d_sm, lambda, rate_of(&sm_leaf)?);
    // On a lossless-segment leaf a skip-mode block (which can never
    // code a residual) is only admissible when its bare prediction is
    // already exact — the q = 0 contract is reconstruction == input.
    // r426: the skip-mode leaf's segment is the §5.11.20 `pred`
    // cascade value, so ITS `LosslessArray[]` slot gates the check
    // (on CodedLossless frames every slot is lossless — the pre-r426
    // frame-wide guard falls out as the uniform case); a leaf
    // overlapping the exactness-demand mask takes the same exactness
    // gate regardless of the pred segment.
    let must_be_exact =
        ictx.seg_lossless(sm_leaf.segment_id) || ictx.demands_exact(mi_r, mi_c, b_size);
    if score_sm < score_normal && (!must_be_exact || d_sm == 0) {
        return Ok(sm_leaf);
    }
    // Skip-mode loses: restore the residual-coded reconstruction and
    // re-stamp the winner's grid footprint (predict_leaf writes only
    // the scratch planes, never the recon).
    restore_region(recon, mi_r, mi_c, &after_normal);
    if let Some(ib) = leaf.inter.as_ref() {
        // r417 — an inter-intra winner re-predicts through the same
        // arm (its committed compound_type is the §5.11.29 bit-silent
        // derivation, not a §5.11.29 side-data selection).
        let ii = ib.interintra_mode.map(|m| InterIntraTrial {
            mode: m,
            wedge: (ib.wedge_interintra == 1).then_some(ib.interintra_wedge_index),
            neigh: &*recon,
        });
        let sel = if ib.interintra_mode.is_some() {
            CompoundSel::AVERAGE
        } else {
            CompoundSel {
                ctype: ib.compound_type,
                wedge_index: ib.wedge_index,
                wedge_sign: ib.wedge_sign,
                mask_type: ib.mask_type,
            }
        };
        // r419 — a WARPED_CAUSAL winner re-predicts through the same
        // arm: the §7.11.3.8 fit is a pure function of the mirror
        // state (unchanged since the trial — `stamp_leaf` runs at the
        // caller), so the recomputation is bit-identical.
        let motion = match ib.motion_mode {
            MOTION_MODE_OBMC => MotionSel::Obmc,
            MOTION_MODE_WARPED_CAUSAL => MotionSel::Warp(
                ictx.warp_fit(mi_r, mi_c, b_size, ib.ref_frame[0], ib.mv)
                    .ok_or(Error::PartitionWalkOutOfRange)?,
            ),
            _ => MotionSel::Simple,
        };
        ictx.predict_leaf(
            mi_r,
            mi_c,
            b_size,
            ib.ref_frame,
            ib.y_mode,
            ib.mv,
            ib.interp_filter[0],
            sel,
            ii,
            motion,
        )?;
    }
    Ok(leaf)
}

/// r426 — re-run the committed leaf's prediction arm so the driver
/// grids (and the prediction scratch) hold the WINNER's stamps — the
/// same re-predict discipline as the skip-mode loser path.
fn repredict_leaf_arm(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    recon: &ReconState,
    ictx: &mut PSearchCtx,
    leaf: &SyntaxBlock,
) -> Result<(), Error> {
    let Some(ib) = leaf.inter.as_ref() else {
        return Ok(());
    };
    let ii = ib.interintra_mode.map(|m| InterIntraTrial {
        mode: m,
        wedge: (ib.wedge_interintra == 1).then_some(ib.interintra_wedge_index),
        neigh: recon,
    });
    let sel = if ib.interintra_mode.is_some() {
        CompoundSel::AVERAGE
    } else {
        CompoundSel {
            ctype: ib.compound_type,
            wedge_index: ib.wedge_index,
            wedge_sign: ib.wedge_sign,
            mask_type: ib.mask_type,
        }
    };
    let motion = match ib.motion_mode {
        MOTION_MODE_OBMC => MotionSel::Obmc,
        MOTION_MODE_WARPED_CAUSAL => MotionSel::Warp(
            ictx.warp_fit(mi_r, mi_c, b_size, ib.ref_frame[0], ib.mv)
                .ok_or(Error::PartitionWalkOutOfRange)?,
        ),
        _ => MotionSel::Simple,
    };
    ictx.predict_leaf(
        mi_r,
        mi_c,
        b_size,
        ib.ref_frame,
        ib.y_mode,
        ib.mv,
        ib.interp_filter[0],
        sel,
        ii,
        motion,
    )
}

/// r426 — ladder item 8: the §5.9.14 inter-override feature-segment
/// trials against the fully-searched winner, in three arms:
///
/// * **SEG_LVL_REF_FRAME** re-labelling: a single-reference winner
///   whose `RefFrame[ 0 ]` equals the feature data moves onto the
///   feature segment — identical pixels, the §5.11.25 reference bits
///   and the §5.11.20 `is_inter` S() fall bit-silent (elected by
///   exact twin bits; requires the identical §7.12.2 quantiser).
/// * **SEG_LVL_SKIP**: a pure-derivation block (skip = 1 with NO
///   §5.11.11 bit, mode GLOBALMV, `RefFrame = [ LAST, NONE ]`, no
///   residual) — the §5.9.14 static-region form; scored as bare
///   prediction against the winner like the §5.11.10 skip-mode
///   trial.
/// * **SEG_LVL_GLOBALMV**: mode/ref fall bit-silent, the residual
///   still codes at the segment's quantiser through the normal
///   §5.11.34 chain.
///
/// All trials run on `>= 8x8` leaves only and finish by re-running
/// the winner's prediction arm (grid-invariant discipline).
#[allow(clippy::too_many_arguments)]
fn seg_extras_trials(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    ictx: &mut PSearchCtx,
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
    mut best: SyntaxBlock,
) -> Result<SyntaxBlock, Error> {
    let x = ictx.extras.expect("caller gated on extras");
    let bw4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let bh4 = NUM_4X4_BLOCKS_HIGH[b_size];
    let lambda = lambda_for(&recon.qp);
    let rate_of = |blk: &SyntaxBlock| -> Result<u64, Error> {
        match pricing {
            Some((twin, params)) => twin.price_block(blk, mi_r, mi_c, b_size, params),
            None => Ok(p_leaf_rate(blk) * 256),
        }
    };
    let d_now = region_distortion_wh(recon, input, mi_r, mi_c, bw4, bh4);
    let mut score_best = score256(d_now, lambda, rate_of(&best)?);

    // Arm 1 — SEG_LVL_REF_FRAME re-labelling (no pixel change).
    for (s, rf) in x.ref_frame.iter().enumerate() {
        let Some(rf) = rf else { continue };
        if x.skip[s] || x.globalmv[s] {
            continue;
        }
        let Some(ib) = best.inter.as_ref() else {
            break;
        };
        if ib.skip_mode != 0
            || ib.ref_frame[0] != *rf
            || ib.ref_frame[1] > 0
            || best.segment_id as usize == s
        {
            continue;
        }
        // The residual was quantised at the winner's segment — the
        // re-label is only valid on an identical §7.12.2 qindex.
        if ictx.seg_qp(&recon.qp, s as u8).base_q_idx
            != ictx.seg_qp(&recon.qp, best.segment_id).base_q_idx
        {
            continue;
        }
        let mut alt = best.clone();
        alt.segment_id = s as u8;
        let sc = score256(d_now, lambda, rate_of(&alt)?);
        if sc < score_best {
            score_best = sc;
            best = alt;
        }
    }

    // Arms 2/3 — SEG_LVL_SKIP / SEG_LVL_GLOBALMV prediction trials.
    let mut after_best = save_region_wh(recon, mi_r, mi_c, bw4, bh4);
    for s in 0..crate::uncompressed_header_tail::MAX_SEGMENTS {
        let (is_skip_seg, is_gmv_seg) = (x.skip[s], x.globalmv[s]);
        if !is_skip_seg && !is_gmv_seg {
            continue;
        }
        // §5.11.25 arm 3: RefFrame = [ LAST_FRAME, NONE ]; §5.11.23
        // arm 2: YMode = GLOBALMV; Mv = GlobalMvs[ 0 ] (§7.10.2.1).
        let stack = ictx.find_stack(mi_r, mi_c, b_size, [1, -1])?;
        let gmv = [stack.global_mvs[0], [0, 0]];
        if gmv[0][0].unsigned_abs() >= (1 << 14) || gmv[0][1].unsigned_abs() >= (1 << 14) {
            continue;
        }
        ictx.predict_leaf(
            mi_r,
            mi_c,
            b_size,
            [1, -1],
            MODE_GLOBALMV,
            gmv,
            EIGHTTAP,
            CompoundSel::AVERAGE,
            None,
            MotionSel::Simple,
        )?;
        let trial: SyntaxBlock = if is_skip_seg {
            // Bare-prediction block (skip = 1, no residual) — stitch
            // the scratch prediction into the recon planes.
            let (bw, bh) = (bw4 * 4, bh4 * 4);
            let (row0, col0) = ((mi_r as usize) * 4, (mi_c as usize) * 4);
            // r427 — subsampling-derived chroma stitch footprint.
            let (crow0, ccol0) = recon.chroma_origin(mi_r, mi_c);
            let (cbw, cbh) = if recon.num_planes > 1 {
                (bw >> recon.subsampling_x, bh >> recon.subsampling_y)
            } else {
                (0, 0)
            };
            let width = recon.width;
            let cw = recon.chroma_w;
            for i in 0..bh {
                for j in 0..bw {
                    recon.y[(row0 + i) * width + (col0 + j)] =
                        ictx.scratch[0][(row0 + i) * width + (col0 + j)];
                }
            }
            for i in 0..cbh {
                for j in 0..cbw {
                    recon.u[(crow0 + i) * cw + (ccol0 + j)] =
                        ictx.scratch[1][(crow0 + i) * cw + (ccol0 + j)];
                    recon.v[(crow0 + i) * cw + (ccol0 + j)] =
                        ictx.scratch[2][(crow0 + i) * cw + (ccol0 + j)];
                }
            }
            tu_bd_stamp(&mut recon.bd, 0, col0, row0, bw, bh);
            if recon.num_planes > 1 {
                tu_bd_stamp(&mut recon.bd, 1, ccol0, crow0, cbw, cbh);
                tu_bd_stamp(&mut recon.bd, 2, ccol0, crow0, cbw, cbh);
            }
            let mut l = SyntaxBlock::skip_leaf(0, None);
            l.segment_id = s as u8;
            l.inter = Some(SyntaxInterBlock {
                ref_frame: [1, -1],
                y_mode: MODE_GLOBALMV,
                mv: gmv,
                ref_mv_idx: 0,
                interp_filter: [EIGHTTAP; 2],
                skip_mode: 0,
                compound_type: crate::inter_pred::COMPOUND_AVERAGE,
                wedge_index: 0,
                wedge_sign: 0,
                mask_type: 0,
                interintra_mode: None,
                wedge_interintra: 0,
                interintra_wedge_index: 0,
                motion_mode: MOTION_MODE_SIMPLE,
            });
            l
        } else {
            // SEG_LVL_GLOBALMV — residual-coded at the segment's
            // quantiser, over the SAME §5.11.17 uniform-depth ladder
            // the normal leaf searched (a depth-0-only trial would
            // concede distortion the mode/ref bit savings cannot
            // recoup).
            let seg_qp = ictx.seg_qp(&recon.qp, s as u8);
            let mut max_depth = 0u32;
            let mut t = MAX_TX_SIZE_RECT[b_size];
            while max_depth < MAX_VARTX_DEPTH && t != TX_4X4 {
                t = SPLIT_TX_SIZE[t];
                max_depth += 1;
            }
            let before_trial = save_region_wh(recon, mi_r, mi_c, bw4, bh4);
            let mut inner: Option<(SyntaxBlock, RegionSnapshot, u64)> = None;
            for depth in 0..=max_depth {
                let cand = encode_inter_leaf_residual(
                    mi_r,
                    mi_c,
                    b_size,
                    input,
                    recon,
                    ictx,
                    [1, -1],
                    MODE_GLOBALMV,
                    gmv,
                    0,
                    EIGHTTAP,
                    CompoundSel::AVERAGE,
                    None,
                    MOTION_MODE_SIMPLE,
                    depth,
                    s as u8,
                    &seg_qp,
                    pricing,
                )?;
                let d = region_distortion_wh(recon, input, mi_r, mi_c, bw4, bh4);
                let sc = score256(d, lambda, rate_of(&cand)?);
                if inner.as_ref().map(|(_, _, s0)| sc < *s0).unwrap_or(true) {
                    inner = Some((cand, save_region_wh(recon, mi_r, mi_c, bw4, bh4), sc));
                }
                restore_region(recon, mi_r, mi_c, &before_trial);
            }
            let (cand, after_cand, _) = inner.expect("at least depth 0");
            restore_region(recon, mi_r, mi_c, &after_cand);
            cand
        };
        let d = region_distortion_wh(recon, input, mi_r, mi_c, bw4, bh4);
        let sc = score256(d, lambda, rate_of(&trial)?);
        if sc < score_best {
            score_best = sc;
            best = trial;
            after_best = save_region_wh(recon, mi_r, mi_c, bw4, bh4);
        } else {
            restore_region(recon, mi_r, mi_c, &after_best);
        }
    }
    // Grid-invariant finish: recon holds the winner; re-run its
    // prediction arm so the §5.11.5 driver grids match.
    restore_region(recon, mi_r, mi_c, &after_best);
    repredict_leaf_arm(mi_r, mi_c, b_size, recon, ictx, &best)?;
    Ok(best)
}

/// r411/r412 fully-searched inter leaf (mode + filter + residual RD);
/// the r413 [`encode_inter_leaf`] wrapper adds the skip-mode trial on
/// top.
#[allow(clippy::too_many_arguments)]
fn encode_inter_leaf_modes(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    ictx: &mut PSearchCtx,
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    let bw4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let bh4 = NUM_4X4_BLOCKS_HIGH[b_size];
    let (bw, bh) = (bw4 * 4, bh4 * 4);
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let width = recon.width;

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
        /// r422 — the EXACT §5.11.23 mode + MV prefix price
        /// (1/256-bit units) through the writer's own emission path
        /// against the twin's current adaptive CDFs; `None` under the
        /// heuristic rate model.
        bits256: Option<u64>,
    }
    let diff_bits = |mv: [i32; 2], pred: [i32; 2]| -> u64 {
        let b = |d: i32| u64::from(34 - d.unsigned_abs().leading_zeros());
        b(mv[0] - pred[0]) + b(mv[1] - pred[1])
    };
    // r422 — exact candidate pricing under the twin model: the
    // §5.11.18 neighbour scalars the §5.11.25 cascade ctx walks read,
    // derived from the search mirror exactly as the write pass
    // derives them from ITS mirror at this leaf (both hold only prior
    // committed blocks here).
    let (nbr_avail_u, nbr_avail_l) = (mi_r > 0, mi_c > 0);
    let nbr_ref = |r: u32, c: u32| -> [i32; 2] {
        let cell = ((r * ictx.mi_cols + c) as usize) * 2;
        let rfg = ictx.mirror.ref_frames();
        [i32::from(rfg[cell]), i32::from(rfg[cell + 1])]
    };
    let above_ref_frame = if nbr_avail_u {
        nbr_ref(mi_r - 1, mi_c)
    } else {
        [0, -1]
    };
    let left_ref_frame = if nbr_avail_l {
        nbr_ref(mi_r, mi_c - 1)
    } else {
        [0, -1]
    };
    let (p_reference_select, p_fimv, p_hp) = (
        ictx.ip.reference_select,
        ictx.ip.force_integer_mv,
        ictx.ip.allow_high_precision_mv,
    );
    let price_mode = |cand_ref: [i8; 2],
                      y_mode: u8,
                      mv: [[i32; 2]; 2],
                      ref_mv_idx: u32,
                      stack: &crate::cdf::FindMvStackResult|
     -> Result<Option<u64>, Error> {
        match pricing {
            Some((twin, _)) => Ok(Some(twin.price_inter_mode(
                [i32::from(cand_ref[0]), i32::from(cand_ref[1])],
                y_mode,
                mv,
                ref_mv_idx,
                stack,
                b_size,
                p_reference_select,
                nbr_avail_u,
                nbr_avail_l,
                above_ref_frame,
                left_ref_frame,
                p_fimv,
                p_hp,
            )?)),
            None => Ok(None),
        }
    };
    let mut cands: Vec<ModeCand> = Vec::with_capacity(18);
    // r415 — per-reference searched vectors, keyed by the raw
    // `RefFrame` ordinal (NEW_NEWMV compound candidates reuse them).
    let mut searched_mv: [[i32; 2]; 8] = [[0, 0]; 8];
    let single_refs = ictx.single_refs.clone();
    // r416 — §5.11.25 `Min( bw4, bh4 ) >= 2`: sub-8×8 blocks cannot
    // code `comp_mode` (SINGLE_REFERENCE forced with no bit), so the
    // compound ladder is empty there. r421 — the §5.11.25
    // `reference_select` frame gate joins: with `reference_select ==
    // 0` the reader never codes `comp_mode` (SINGLE_REFERENCE forced),
    // so a compound candidate would be uncodable — the rate twin's
    // write-path validation surfaced configurations offering pairs the
    // header cannot express.
    let compound_pairs = if bw4.min(bh4) >= 2 && ictx.ip.reference_select {
        ictx.compound_pairs.clone()
    } else {
        Vec::new()
    };
    for (ref_ord, &rf) in single_refs.iter().enumerate() {
        let ref_bias = ref_ord as u64;
        let stack = ictx.find_stack(mi_r, mi_c, b_size, [rf, -1])?;
        let ref_plane = ictx.plane_of_ref(rf);
        let mv_int = motion_search_luma(
            input,
            &ictx.ref_planes[ref_plane][0],
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
        searched_mv[rf as usize] = mv_new;
        let nfound = stack.num_mv_found;
        {
            // NEWMV: pick the reachable drl slot with the cheapest
            // §5.11.32 difference — exact twin bits when priced
            // (drl_mode + read_mv against the current adaptive CDFs),
            // else each extra slot costs one heuristic drl bit.
            let window: u32 = match nfound {
                0 | 1 => 0,
                2 => 1,
                _ => 2,
            };
            let mut best: Option<(u64, u32, u64, Option<u64>)> = None;
            for idx in 0..=window {
                let pred = assign_mv_pred_mv(&stack, MODE_NEWMV, 0, idx)?;
                let rate = 5 + ref_bias + u64::from(idx) + diff_bits(mv_new, pred);
                let exact = price_mode([rf, -1], MODE_NEWMV, [mv_new, [0, 0]], idx, &stack)?;
                let key = exact.unwrap_or(rate);
                if best.map_or(true, |(k, _, _, _)| key < k) {
                    best = Some((key, idx, rate, exact));
                }
            }
            let (_, idx, rate, bits256) = best.expect("slot 0 is always reachable");
            cands.push(ModeCand {
                ref_frame: [rf, -1],
                y_mode: MODE_NEWMV,
                mv: [mv_new, [0, 0]],
                ref_mv_idx: idx,
                rate,
                bits256,
            });
        }
        let nearest_mv = [stack.ref_stack_mv[0][0], [0, 0]];
        cands.push(ModeCand {
            ref_frame: [rf, -1],
            y_mode: MODE_NEARESTMV,
            mv: nearest_mv,
            ref_mv_idx: 0,
            rate: 3 + ref_bias,
            bits256: price_mode([rf, -1], MODE_NEARESTMV, nearest_mv, 0, &stack)?,
        });
        let near_top: u32 = match nfound {
            0..=2 => 1,
            3 => 2,
            _ => 3,
        };
        for idx in 1..=near_top {
            let near_mv = [stack.ref_stack_mv[idx as usize][0], [0, 0]];
            cands.push(ModeCand {
                ref_frame: [rf, -1],
                y_mode: MODE_NEARMV,
                mv: near_mv,
                ref_mv_idx: idx,
                rate: 4 + ref_bias + u64::from(idx - 1),
                bits256: price_mode([rf, -1], MODE_NEARMV, near_mv, idx, &stack)?,
            });
        }
        let global_mv = [stack.global_mvs[0], [0, 0]];
        cands.push(ModeCand {
            ref_frame: [rf, -1],
            y_mode: MODE_GLOBALMV,
            mv: global_mv,
            ref_mv_idx: 0,
            rate: 4 + ref_bias,
            bits256: price_mode([rf, -1], MODE_GLOBALMV, global_mv, 0, &stack)?,
        });
    }

    // r412 — COMPOUND_AVERAGE candidates over each ladder pair
    // (§5.11.25 unidirectional or bidirectional compound; the
    // §5.11.29 tail is bit-silent on this configuration and derives
    // COMPOUND_AVERAGE): NEAREST_NEARESTMV / NEAR_NEARMV from the
    // compound §7.10.2 stack, GLOBAL_GLOBALMV at the identity
    // derivation, and NEW_NEWMV re-using the two per-reference
    // searched vectors.
    for &crf in &compound_pairs {
        let stack = ictx.find_stack(mi_r, mi_c, b_size, crf)?;
        let nfound = stack.num_mv_found;
        let nn_mv = [stack.ref_stack_mv[0][0], stack.ref_stack_mv[0][1]];
        cands.push(ModeCand {
            ref_frame: crf,
            y_mode: MODE_NEAREST_NEARESTMV,
            mv: nn_mv,
            ref_mv_idx: 0,
            rate: 6,
            bits256: price_mode(crf, MODE_NEAREST_NEARESTMV, nn_mv, 0, &stack)?,
        });
        let near_top: u32 = match nfound {
            0..=2 => 1,
            3 => 2,
            _ => 3,
        };
        for idx in 1..=near_top {
            let nn_mv = [
                stack.ref_stack_mv[idx as usize][0],
                stack.ref_stack_mv[idx as usize][1],
            ];
            cands.push(ModeCand {
                ref_frame: crf,
                y_mode: MODE_NEAR_NEARMV,
                mv: nn_mv,
                ref_mv_idx: idx,
                rate: 7 + u64::from(idx - 1),
                bits256: price_mode(crf, MODE_NEAR_NEARMV, nn_mv, idx, &stack)?,
            });
        }
        let gg_mv = [stack.global_mvs[0], stack.global_mvs[1]];
        cands.push(ModeCand {
            ref_frame: crf,
            y_mode: MODE_GLOBAL_GLOBALMV,
            mv: gg_mv,
            ref_mv_idx: 0,
            rate: 7,
            bits256: price_mode(crf, MODE_GLOBAL_GLOBALMV, gg_mv, 0, &stack)?,
        });
        {
            let window: u32 = match nfound {
                0 | 1 => 0,
                2 => 1,
                _ => 2,
            };
            let pair_mv = [searched_mv[crf[0] as usize], searched_mv[crf[1] as usize]];
            let mut best: Option<(u64, u32, u64, Option<u64>)> = None;
            for idx in 0..=window {
                let pred0 = assign_mv_pred_mv(&stack, MODE_NEW_NEWMV, 0, idx)?;
                let pred1 = assign_mv_pred_mv(&stack, MODE_NEW_NEWMV, 1, idx)?;
                let rate = 8
                    + u64::from(idx)
                    + diff_bits(pair_mv[0], pred0)
                    + diff_bits(pair_mv[1], pred1);
                let exact = price_mode(crf, MODE_NEW_NEWMV, pair_mv, idx, &stack)?;
                let key = exact.unwrap_or(rate);
                if best.map_or(true, |(k, _, _, _)| key < k) {
                    best = Some((key, idx, rate, exact));
                }
            }
            let (_, idx, rate, bits256) = best.expect("slot 0 is always reachable");
            cands.push(ModeCand {
                ref_frame: crf,
                y_mode: MODE_NEW_NEWMV,
                mv: pair_mv,
                ref_mv_idx: idx,
                rate,
                bits256,
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
    let mut best_compound: Option<(u64, usize)> = None;
    // r417 — the best SINGLE-reference candidate (the §5.11.28
    // inter-intra trials blend on top of it).
    let mut best_single: Option<(u64, usize)> = None;
    for (ci, cand) in cands.iter().enumerate() {
        ictx.predict_leaf(
            mi_r,
            mi_c,
            b_size,
            cand.ref_frame,
            cand.y_mode,
            cand.mv,
            EIGHTTAP,
            CompoundSel::AVERAGE,
            None,
            MotionSel::Simple,
        )?;
        let mut ssd = 0u64;
        for i in 0..bh {
            for j in 0..bw {
                let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                    - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                ssd += (d * d) as u64;
            }
        }
        // r422 — twin model: `D·256 + λ·bits256` with the exact
        // §5.11.23 mode-prefix price; heuristic model keeps the
        // pre-r422 proxy scale. All candidates of one election carry
        // the same scale, so the comparison is well-formed.
        let score = match cand.bits256 {
            Some(b) => score256(ssd, lambda, b),
            None => ssd + lambda * cand.rate,
        };
        if best.map_or(true, |(s, _)| score < s) {
            best = Some((score, ci));
        }
        if cand.ref_frame[1] > 0 && best_compound.map_or(true, |(s, _)| score < s) {
            best_compound = Some((score, ci));
        }
        if cand.ref_frame[1] < 0 && best_single.map_or(true, |(s, _)| score < s) {
            best_single = Some((score, ci));
        }
    }
    let (mut best_score, mut best_ci) = best.ok_or(Error::PartitionWalkOutOfRange)?;

    // r415 — §5.11.29 MASKED-compound trials on the best compound
    // candidate: every WEDGE (index, sign) pair where
    // `Wedge_Bits[ MiSize ] > 0` plus both DIFFWTD mask types, each
    // predicted through the decoder's own §7.11.3.11/§7.11.3.12 mask
    // blend and scored `D + λ·(rate + side-data bits)` against the
    // running best (which keeps COMPOUND_AVERAGE when the masks lose).
    let mut comp = CompoundSel::AVERAGE;
    if ictx.ip.enable_masked_compound {
        if let Some((_, cci)) = best_compound {
            let (c_ref, c_mode, c_mv, c_rate, c_bits256) = {
                let c = &cands[cci];
                (c.ref_frame, c.y_mode, c.mv, c.rate, c.bits256)
            };
            let mut trials: Vec<(CompoundSel, u64)> = Vec::new();
            if crate::cdf::wedge_bits(b_size) > 0 {
                for wi in 0..16u8 {
                    for ws in 0..=1u8 {
                        trials.push((
                            CompoundSel {
                                ctype: crate::inter_pred::COMPOUND_WEDGE,
                                wedge_index: wi,
                                wedge_sign: ws,
                                mask_type: 0,
                            },
                            /* comp_group_idx + wedge_index S() + sign L(1) */ 6,
                        ));
                    }
                }
            }
            for mt in 0..=1u8 {
                trials.push((
                    CompoundSel {
                        ctype: crate::inter_pred::COMPOUND_DIFFWTD,
                        wedge_index: 0,
                        wedge_sign: 0,
                        mask_type: mt,
                    },
                    /* comp_group_idx (+ compound_type S()) + mask L(1) */ 3,
                ));
            }
            for (sel, extra) in trials {
                ictx.predict_leaf(
                    mi_r,
                    mi_c,
                    b_size,
                    c_ref,
                    c_mode,
                    c_mv,
                    EIGHTTAP,
                    sel,
                    None,
                    MotionSel::Simple,
                )?;
                let mut ssd = 0u64;
                for i in 0..bh {
                    for j in 0..bw {
                        let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                            - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                        ssd += (d * d) as u64;
                    }
                }
                // r422 — twin scale: the exact mode-prefix bits plus
                // the side-symbol proxy promoted to 1/256 units.
                let score = match c_bits256 {
                    Some(b) => score256(ssd, lambda, b + extra * 256),
                    None => ssd + lambda * (c_rate + extra),
                };
                if score < best_score {
                    best_score = score;
                    best_ci = cci;
                    comp = sel;
                }
            }
        }
    }

    // r416 — §5.11.29 jnt-comp trial on the best compound candidate:
    // the COMPOUND_DISTANCE leaf (`comp_group_idx = 0`,
    // `compound_idx = 0`) predicted through the decoder's own
    // §7.11.3.15 distance-weighted blend. Rate-neutral against the
    // coded AVERAGE arm (both code the same two §5.11.29 symbols), so
    // luma distortion alone decides; ties keep the running best.
    if ictx.ip.enable_jnt_comp {
        if let Some((_, cci)) = best_compound {
            let (c_ref, c_mode, c_mv, c_rate, c_bits256) = {
                let c = &cands[cci];
                (c.ref_frame, c.y_mode, c.mv, c.rate, c.bits256)
            };
            let sel = CompoundSel {
                ctype: crate::inter_pred::COMPOUND_DISTANCE,
                wedge_index: 0,
                wedge_sign: 0,
                mask_type: 0,
            };
            ictx.predict_leaf(
                mi_r,
                mi_c,
                b_size,
                c_ref,
                c_mode,
                c_mv,
                EIGHTTAP,
                sel,
                None,
                MotionSel::Simple,
            )?;
            let mut ssd = 0u64;
            for i in 0..bh {
                for j in 0..bw {
                    let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                        - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                    ssd += (d * d) as u64;
                }
            }
            let score = match c_bits256 {
                Some(b) => score256(ssd, lambda, b),
                None => ssd + lambda * c_rate,
            };
            if score < best_score {
                best_score = score;
                best_ci = cci;
                comp = sel;
            }
        }
    }

    // r417 — §5.11.28 inter-intra trials on the best single-reference
    // candidate: each of the four §6.10.27 II modes through the
    // §7.11.3.13 smooth intra-variant mask (the intra half predicted
    // from the committed recon's neighbours), then — where
    // `Wedge_Bits[ MiSize ] > 0` — the 16 §7.11.3.11 wedge masks at
    // the best smooth mode, every trial through the decoder's own
    // §7.11.3.14 blend driver and scored `D + λ·(rate + §5.11.28
    // bits)` against the running best (which keeps the plain leaf
    // when the blends lose).
    let mut ii_sel: Option<(u8, Option<u8>)> = None;
    if ictx.ip.enable_interintra_compound
        && (crate::cdf::BLOCK_8X8..=crate::cdf::BLOCK_32X32).contains(&b_size)
    {
        if let Some((_, sci)) = best_single {
            let (s_ref, s_mode, s_mv, s_rate, s_bits256) = {
                let c = &cands[sci];
                (c.ref_frame, c.y_mode, c.mv, c.rate, c.bits256)
            };
            let leaf_ssd = |ictx: &PSearchCtx| -> u64 {
                let mut ssd = 0u64;
                for i in 0..bh {
                    for j in 0..bw {
                        let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                            - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                        ssd += (d * d) as u64;
                    }
                }
                ssd
            };
            // Stage 1 — the four smooth-mask II modes
            // (`interintra` S() + `interintra_mode` S() +
            // `wedge_interintra` S() ≈ 4 rate-proxy bits).
            let mut stage: Option<(u64, u8)> = None;
            for m in 0..crate::cdf::INTERINTRA_MODES as u8 {
                ictx.predict_leaf(
                    mi_r,
                    mi_c,
                    b_size,
                    s_ref,
                    s_mode,
                    s_mv,
                    EIGHTTAP,
                    CompoundSel::AVERAGE,
                    Some(InterIntraTrial {
                        mode: m,
                        wedge: None,
                        neigh: &*recon,
                    }),
                    MotionSel::Simple,
                )?;
                let ssd = leaf_ssd(ictx);
                if stage.map_or(true, |(s, _)| ssd < s) {
                    stage = Some((ssd, m));
                }
                let score = match s_bits256 {
                    Some(b) => score256(ssd, lambda, b + 4 * 256),
                    None => ssd + lambda * (s_rate + 4),
                };
                if score < best_score {
                    best_score = score;
                    best_ci = sci;
                    comp = CompoundSel::AVERAGE;
                    ii_sel = Some((m, None));
                }
            }
            // Stage 2 — the 16 wedge masks at the stage-1 distortion
            // winner (+ `wedge_index` S() ≈ 4 more bits).
            if crate::cdf::wedge_bits(b_size) > 0 {
                let (_, base_mode) = stage.expect("stage 1 ran");
                for wi in 0..16u8 {
                    ictx.predict_leaf(
                        mi_r,
                        mi_c,
                        b_size,
                        s_ref,
                        s_mode,
                        s_mv,
                        EIGHTTAP,
                        CompoundSel::AVERAGE,
                        Some(InterIntraTrial {
                            mode: base_mode,
                            wedge: Some(wi),
                            neigh: &*recon,
                        }),
                        MotionSel::Simple,
                    )?;
                    let score = match s_bits256 {
                        Some(b) => score256(leaf_ssd(ictx), lambda, b + 8 * 256),
                        None => leaf_ssd(ictx) + lambda * (s_rate + 8),
                    };
                    if score < best_score {
                        best_score = score;
                        best_ci = sci;
                        comp = CompoundSel::AVERAGE;
                        ii_sel = Some((base_mode, Some(wi)));
                    }
                }
            }
        }
    }
    let _ = best_score;
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
    let leaf_luma_ssd = |ictx: &PSearchCtx| -> u64 {
        let mut ssd = 0u64;
        for i in 0..bh {
            for j in 0..bw {
                let d = i64::from(input.y[(row0 + i) * width + col0 + j])
                    - i64::from(ictx.scratch[0][(row0 + i) * width + col0 + j]);
                ssd += (d * d) as u64;
            }
        }
        ssd
    };
    let (mut filter, simple_ssd) = if ictx.ip.interpolation_filter != SWITCHABLE {
        // §5.11.26 else-arm: a non-SWITCHABLE frame filter is read
        // with no bits — every leaf must carry exactly it (r421: the
        // rate twin's write-path validation surfaced searches trialling
        // filters the header cannot express).
        let f = ictx.ip.interpolation_filter;
        ictx.predict_leaf(
            mi_r,
            mi_c,
            b_size,
            ref_frame,
            y_mode,
            mv,
            f,
            comp,
            None,
            MotionSel::Simple,
        )?;
        (f, leaf_luma_ssd(ictx))
    } else if y_mode == MODE_GLOBALMV || y_mode == MODE_GLOBAL_GLOBALMV {
        ictx.predict_leaf(
            mi_r,
            mi_c,
            b_size,
            ref_frame,
            y_mode,
            mv,
            EIGHTTAP,
            comp,
            None,
            MotionSel::Simple,
        )?;
        (EIGHTTAP, leaf_luma_ssd(ictx))
    } else {
        use crate::inter_pred::{EIGHTTAP_SHARP, EIGHTTAP_SMOOTH};
        let mut best_f = (u64::MAX, EIGHTTAP);
        for f in [EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP] {
            ictx.predict_leaf(
                mi_r,
                mi_c,
                b_size,
                ref_frame,
                y_mode,
                mv,
                f,
                comp,
                None,
                MotionSel::Simple,
            )?;
            let ssd = leaf_luma_ssd(ictx);
            if ssd < best_f.0 {
                best_f = (ssd, f);
            }
        }
        (best_f.1, best_f.0)
    };

    // r419 — §5.11.27 motion-mode trials on the winning
    // single-reference (mode, MV): OBMC (the §7.11.3.9-10 overlap
    // blend, trialled per codable filter) and — where the §5.11.27
    // arm-B gates open — WARPED_CAUSAL (the §7.10.4 sample scan +
    // §7.11.3.8 least-squares fit + §7.11.3.5 warp, filter-silent
    // `EIGHTTAP` per `needs_interp_filter( )`). Every arm codes one
    // roughly-equal-rate S() whichever value it selects, so luma
    // distortion decides; ties keep the earlier arm (SIMPLE, then
    // OBMC). The eligibility derivation tracks the §5.11.27 reader's
    // short-circuit cascade exactly — an ineligible leaf never trials
    // (and the write pass re-derives the same cascade and would
    // reject an uncodable commitment).
    let mut motion = MotionSel::Simple;
    if ictx.ip.is_motion_mode_switchable
        && ii_sel.is_none()
        && ref_frame[1] == -1
        && bw.min(bh) >= 8
        && !(!ictx.ip.force_integer_mv
            && y_mode == MODE_GLOBALMV
            && ictx.ip.gm_type[ref_frame[0] as usize] > crate::cdf::GM_TYPE_TRANSLATION)
        && ictx.mirror.has_overlappable_candidates(mi_r, mi_c, b_size)
    {
        // r421 — the §5.11.27 arm is no longer priced as "roughly
        // equal-rate": under the twin model each candidate ordinal is
        // priced through the writer's own arm derivation against the
        // CURRENT adaptive CDFs (`use_obmc` and `motion_mode` rows
        // adapt away from uniform quickly, so SIMPLE vs OBMC vs
        // WARPED_CAUSAL can differ by whole bits), and the election
        // scores `D·256 + λ·bits256`. The heuristic model keeps the
        // pre-r421 pure-distortion pick.
        let num_samples = ictx.warp_num_samples(mi_r, mi_c, b_size, ref_frame[0], mv);
        // Copies of the gating scalars, so the pricing closure holds
        // no borrow of `ictx` across the `predict_leaf` trials.
        let (mm_switchable, mm_warp, mm_fimv, mm_gm_type, mm_scaled) = (
            ictx.ip.is_motion_mode_switchable,
            ictx.ip.allow_warped_motion,
            ictx.ip.force_integer_mv,
            ictx.ip.gm_type,
            ictx.ip.is_scaled_per_ref,
        );
        let mm_bits = move |mm: u8| -> Result<u64, Error> {
            match pricing {
                Some((twin, _)) => twin.price_motion_mode(
                    mm,
                    b_size,
                    false,
                    [i32::from(ref_frame[0]), i32::from(ref_frame[1])],
                    y_mode,
                    num_samples,
                    mm_switchable,
                    mm_warp,
                    mm_fimv,
                    mm_gm_type,
                    mm_scaled,
                    true,
                ),
                None => Ok(0),
            }
        };
        let mut best_score = score256(simple_ssd, lambda, mm_bits(MOTION_MODE_SIMPLE)?);
        // OBMC trial — the leaf's own prediction rides its own
        // §7.11.3.4 filter (GLOBALMV leaves stay filter-silent on
        // EIGHTTAP), the overlap bands the committed neighbours'.
        {
            use crate::inter_pred::{EIGHTTAP_SHARP, EIGHTTAP_SMOOTH};
            let obmc_bits = mm_bits(MOTION_MODE_OBMC)?;
            let fixed = [ictx.ip.interpolation_filter];
            let obmc_filters: &[u8] = if ictx.ip.interpolation_filter != SWITCHABLE {
                // §5.11.26 else-arm: the frame filter is the only
                // codable pair.
                &fixed
            } else if y_mode == MODE_GLOBALMV {
                &[EIGHTTAP]
            } else {
                &[EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP]
            };
            for &f in obmc_filters {
                ictx.predict_leaf(
                    mi_r,
                    mi_c,
                    b_size,
                    ref_frame,
                    y_mode,
                    mv,
                    f,
                    comp,
                    None,
                    MotionSel::Obmc,
                )?;
                let score = score256(leaf_luma_ssd(ictx), lambda, obmc_bits);
                if score < best_score {
                    best_score = score;
                    motion = MotionSel::Obmc;
                    filter = f;
                }
            }
        }
        // WARPED_CAUSAL trial — §5.11.27 arm B only (`NumSamples > 0`
        // over the same §7.10.4 scan the write pass and the decoder
        // re-run at this leaf, `allow_warped_motion`, unscaled
        // reference), and only when the §7.11.3.8 fit is
        // `setup_shear`-valid (an invalid fit decodes translationally
        // — never worth the commitment).
        if ictx.ip.allow_warped_motion
            && !ictx.ip.force_integer_mv
            && !ictx.ip.is_scaled_per_ref[(ref_frame[0] - 1) as usize]
        {
            if let Some(params) = ictx.warp_fit(mi_r, mi_c, b_size, ref_frame[0], mv) {
                ictx.predict_leaf(
                    mi_r,
                    mi_c,
                    b_size,
                    ref_frame,
                    y_mode,
                    mv,
                    EIGHTTAP,
                    comp,
                    None,
                    MotionSel::Warp(params),
                )?;
                let score = score256(
                    leaf_luma_ssd(ictx),
                    lambda,
                    mm_bits(MOTION_MODE_WARPED_CAUSAL)?,
                );
                if score < best_score {
                    motion = MotionSel::Warp(params);
                    // `needs_interp_filter( ) == 0` on LOCALWARP —
                    // the committed pair is the reader's bit-silent
                    // derivation: EIGHTTAP on the SWITCHABLE arm, the
                    // frame filter otherwise.
                    filter = if ictx.ip.interpolation_filter != SWITCHABLE {
                        ictx.ip.interpolation_filter
                    } else {
                        EIGHTTAP
                    };
                }
            }
        }
    }
    ictx.predict_leaf(
        mi_r,
        mi_c,
        b_size,
        ref_frame,
        y_mode,
        mv,
        filter,
        comp,
        ii_sel.map(|(m, w)| InterIntraTrial {
            mode: m,
            wedge: w,
            neigh: &*recon,
        }),
        motion,
    )?;

    // r413 — per-leaf segment (deterministic activity policy) and its
    // quantiser; a leaf that quantises to all-zero commits `skip = 1`
    // and inherits the §5.11.20 `pred` instead (inside
    // `encode_inter_leaf_residual`).
    let segment_id = ictx.segment_for_block(input, mi_r, mi_c, b_size);
    let seg_qp = ictx.seg_qp(&recon.qp, segment_id);

    // r426 — per-segment §5.9.2 `LosslessArray[]`: a leaf whose
    // segment derives `Lossless = 1` codes the §5.11.34 row-major
    // `TX_4X4` grid through the WHT chain with no §5.11.16 trees and
    // no §5.11.15 depth choice, whether the FRAME is CodedLossless
    // (the pre-r426 arm — every segment lossless) or lossy (the r426
    // mixed configuration — this segment only).
    if ictx.seg_lossless(segment_id) {
        // §5.11.15: Lossless forces TX_4X4, no §5.11.17 trees.
        return encode_inter_leaf_residual(
            mi_r,
            mi_c,
            b_size,
            input,
            recon,
            ictx,
            ref_frame,
            y_mode,
            mv,
            ref_mv_idx,
            filter,
            comp,
            ii_sel,
            motion.ordinal(),
            0,
            segment_id,
            &seg_qp,
            pricing,
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
    // r416 — sub-8×8 leaves write the GROUP's chroma (the §5.11.34
    // `HasChroma` block covers the full 2×2-cell chroma area), which
    // extends past the leaf's own mi rect: snapshot / restore / score
    // over the group-aligned rect so no trial state leaks between
    // depth candidates.
    let (sr, sc, sw4, sh4) = {
        let (mut sr, mut sc, mut sw4, mut sh4) = (mi_r, mi_c, bw4, bh4);
        if bh4 == 1 {
            sr = mi_r & !1;
            sh4 = 2;
        }
        if bw4 == 1 {
            sc = mi_c & !1;
            sw4 = 2;
        }
        (sr, sc, sw4, sh4)
    };
    let before = save_region_wh(recon, sr, sc, sw4, sh4);
    let mut best: Option<(SyntaxBlock, RegionSnapshot, u64)> = None;
    for depth in 0..=max_depth {
        let leaf = encode_inter_leaf_residual(
            mi_r,
            mi_c,
            b_size,
            input,
            recon,
            ictx,
            ref_frame,
            y_mode,
            mv,
            ref_mv_idx,
            filter,
            comp,
            ii_sel,
            motion.ordinal(),
            depth,
            segment_id,
            &seg_qp,
            pricing,
        )?;
        let d = region_distortion_wh(recon, input, sr, sc, sw4, sh4);
        // r421 — exact twin bits (the §5.11.17 depth symbols are part
        // of the block syntax, so no depth surcharge) or the pre-r421
        // proxy, one scale per call.
        let rate256 = match pricing {
            Some((twin, params)) => twin.price_block(&leaf, mi_r, mi_c, b_size, params)?,
            None => (p_leaf_rate(&leaf) + 2 * u64::from(depth)) * 256,
        };
        let score = score256(d, lambda, rate256);
        let improves = match best.as_ref() {
            Some((_, _, s)) => score < *s,
            None => true,
        };
        if improves {
            best = Some((leaf, save_region_wh(recon, sr, sc, sw4, sh4), score));
        }
        restore_region(recon, sr, sc, &before);
    }
    // r426 — content-driven per-segment-lossless election: synthetic
    // leaves (tiny luma alphabet) additionally trial the lossless
    // segment — same prediction, qindex-0 WHT residual, `D = 0` by
    // bit-exactness — against the lossy depth ladder on the identical
    // `D + λ·R` scale (twin bits when priced). The lossless arm wins
    // exactly when the ladder's distortion outprices the WHT chain's
    // rate premium.
    if ictx.auto_lossless && ictx.synthetic_leaf(input, mi_r, mi_c, b_size) {
        if let Some(ll_seg) = ictx.lossless_seg {
            let ll_qp = ictx.seg_qp(&recon.qp, ll_seg);
            let leaf = encode_inter_leaf_residual(
                mi_r,
                mi_c,
                b_size,
                input,
                recon,
                ictx,
                ref_frame,
                y_mode,
                mv,
                ref_mv_idx,
                filter,
                comp,
                ii_sel,
                motion.ordinal(),
                0,
                ll_seg,
                &ll_qp,
                pricing,
            )?;
            let d = region_distortion_wh(recon, input, sr, sc, sw4, sh4);
            let rate256 = match pricing {
                Some((twin, params)) => twin.price_block(&leaf, mi_r, mi_c, b_size, params)?,
                None => p_leaf_rate(&leaf) * 256,
            };
            let score = score256(d, lambda, rate256);
            let improves = match best.as_ref() {
                Some((_, _, s)) => score < *s,
                None => true,
            };
            if improves {
                best = Some((leaf, save_region_wh(recon, sr, sc, sw4, sh4), score));
            }
            restore_region(recon, sr, sc, &before);
        }
    }
    let (leaf, after, _) = best.expect("at least depth 0");
    restore_region(recon, sr, sc, &after);
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
///
/// r424 — with `fork = Some((tu_fork, tu_ctx))` each candidate's rate
/// is the EXACT §5.11.39 coefficient chain (priced through the
/// writer's own one-TU body against the fork's running CDF /
/// level-context state — see [`TuFork`]) and the winner is COMMITTED
/// into the fork so the leaf's next TU prices against the true
/// post-TU state; `None` keeps the magnitude proxy (the pre-r424
/// heuristic and the `RateModel::Heuristic` baseline).
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_tx_search_luma_inter(
    input_plane: &[u16],
    recon_plane: &mut [u16],
    pw: usize,
    row0: usize,
    col0: usize,
    tx_sz: usize,
    pred: &[u16],
    qp: &QuantizerParams,
    fork: Option<(&mut TuFork, &TuCtx<'_>)>,
) -> Result<(Vec<i32>, u8), Error> {
    use crate::cdf::{dequantize_step1, is_tx_type_in_set, tx_size_sqr_index, TX_SIZE_SQR_UP};
    use crate::encoder::forward_quantize::forward_quantize;
    use crate::encoder::forward_transform_2d::forward_transform_2d;
    use crate::encoder::key_frame::repack_compact;
    use crate::transform::inverse_transform_2d;

    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    let max_val = (1i64 << qp.bit_depth) - 1;
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
    // r424 — the fork prices the TU at its block-relative 4-sample
    // position; the winner is committed after the election.
    let fork_xy = fork.as_ref().map(|(_, ctx)| {
        (
            (col0 as u32 - ctx.base_x) / 4,
            (row0 as u32 - ctx.base_y) / 4,
        )
    });
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
        let res_back = inverse_transform_2d(&dequant, tx_sz, t, u32::from(qp.bit_depth), false);
        // §7.12.3 step-3 destination remap (FLIPADST family).
        let (flip_ud, flip_lr) = crate::encoder::key_frame::step3_flips(t);
        let mut d = 0u64;
        for i in 0..h {
            let yy = if flip_ud { h - 1 - i } else { i };
            for j in 0..w {
                let xx = if flip_lr { w - 1 - j } else { j };
                let rec = (i64::from(pred[yy * w + xx]) + res_back[i * w + j]).clamp(0, max_val);
                let diff = i64::from(input_plane[(row0 + yy) * pw + (col0 + xx)]) - rec;
                d += (diff * diff) as u64;
            }
        }
        let label = if all_zero { DCT_DCT as u8 } else { t as u8 };
        // r424 — exact §5.11.39 chain bits through the running fork,
        // or the pre-r424 magnitude proxy. One scale per call.
        let score = match (&fork, fork_xy) {
            (Some((tu_fork, ctx)), Some((fx, fy))) => {
                let bits256 = tu_fork.price_luma_tu(ctx, tx_sz, fx, fy, &quant, label)?;
                crate::encoder::rate_twin::score256(d, lambda, bits256)
            }
            _ => {
                let mut rate = 0u64;
                for &qv in &quant {
                    if qv != 0 {
                        rate += 3 + u64::from(32 - qv.unsigned_abs().leading_zeros());
                    }
                }
                d + lambda * rate
            }
        };
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
    // r424 — commit the elected TU into the fork: the next TU's
    // candidates price against the true post-TU CDF / context state.
    if let (Some((tu_fork, ctx)), Some((fx, fy))) = (fork, fork_xy) {
        tu_fork.commit_luma_tu(ctx, tx_sz, fx, fy, &quant, label)?;
    }
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
            recon_plane[(row0 + yy) * pw + (col0 + xx)] = p.clamp(0, max_val) as u16;
        }
    }
    Ok((quant, label))
}

/// §5.11.17 uniform split-decision tree of the given depth rooted at
/// `tx` (r412: rect-aware — the per-split child count is
/// `(h4 / stepH) * (w4 / stepW)`, 4 for square ordinals and 2 for
/// rectangular ones, matching [`VarTxSyntaxTree::Split`]'s contract).
pub(crate) fn uniform_var_tx_tree(tx: usize, depth: u32) -> VarTxSyntaxTree {
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
    input: &YuvFrame,
    recon: &mut ReconState,
    ictx: &PSearchCtx,
    ref_frame: [i8; 2],
    y_mode: u8,
    mv: [[i32; 2]; 2],
    ref_mv_idx: u32,
    filter: u8,
    comp: CompoundSel,
    // r417 — the committed §5.11.28 selection: `Some((mode, wedge))`
    // rides a single-reference leaf whose scratch prediction the
    // caller already ran through the inter-intra arm.
    ii: Option<(u8, Option<u8>)>,
    // r419 — the committed §5.11.27 ordinal (the caller's final
    // `predict_leaf` already ran the matching prediction arm).
    motion_mode: u8,
    depth: u32,
    segment_id: u8,
    seg_qp: &QuantizerParams,
    // r424 — `Some` threads a running per-TU twin fork through the
    // luma residual chain: each §5.11.47 tx-type candidate prices its
    // ACTUAL §5.11.39 coefficient chain (see [`TuFork`]).
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    let bw4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let bh4 = NUM_4X4_BLOCKS_HIGH[b_size];
    let (bw, bh) = (bw4 * 4, bh4 * 4);
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let width = recon.width;
    // r426 — the leaf's OWN §5.9.2 `Lossless = LosslessArray[
    // segment_id ]` (per-segment on mixed frames; uniform on the
    // CodedLossless and plain-lossy configurations).
    let lossless = ictx.seg_lossless(segment_id);
    let qp = *seg_qp;
    // r424 — the running per-TU fork (lossy arm only: the lossless
    // §5.11.47 guard never codes a tx-type symbol and the TU chain is
    // not searched).
    let mut tu_fork = match (&pricing, lossless) {
        (Some((twin, _)), false) => Some(twin.tu_fork()),
        _ => None,
    };
    let tu_ctx = pricing.map(|(_, params)| TuCtx {
        params,
        mi_row: mi_r,
        mi_col: mi_c,
        mi_size: b_size,
        base_x: col0 as u32,
        base_y: row0 as u32,
        is_inter: true,
        segment_id,
        y_mode: 0,
        use_filter_intra: false,
        filter_intra_mode: None,
    });

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
        let mut pred = vec![0u16; ltw * lth];
        for i in 0..lth {
            for j in 0..ltw {
                pred[i * ltw + j] = ictx.scratch[0][(tr + i) * width + (tc + j)];
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
            // the §5.11.48 INTER set for this TX size (r424: exact
            // chain pricing through the running fork when priced).
            residual_tx_search_luma_inter(
                &input.y,
                &mut recon.y,
                width,
                tr,
                tc,
                luma_tx,
                &pred,
                &qp,
                match (&mut tu_fork, &tu_ctx) {
                    (Some(f), Some(c)) => Some((f, c)),
                    _ => None,
                },
            )?
        };
        tu_bd_stamp(&mut recon.bd, 0, tc, tr, ltw, lth);
        residual_quant.push(q);
        luma_tx_types.push(tt);
        tu_type_grid[(ty / lth) * tu_cols + tx / ltw] = tt;
    }

    // --- Chroma (§5.11.34 `HasChroma` gate — r416). ---
    // A sub-8×8 leaf has chroma only on the bottom / right cell of
    // its 2×2 group (`bh4 == 1 && even MiRow` or `bw4 == 1 && even
    // MiCol` ⇒ no chroma planes); the `HasChroma` block's chroma
    // region is the §5.11.38 plane residual size of the block —
    // covering the WHOLE group at the `(MiCol >> subX) * MI_SIZE`
    // lifted origin. §5.11.34: the chroma TU size derives from the
    // block's committed `TxSize` — on the §5.11.16 var-tx arm that is
    // the recursion's last terminal-else `txSz` (the uniform TU size
    // here).
    let has_chroma = recon.num_planes > 1
        && (bh4 != 1 || recon.subsampling_y == 0 || (mi_r & 1) == 1)
        && (bw4 != 1 || recon.subsampling_x == 0 || (mi_c & 1) == 1);
    let (crow0, ccol0) = recon.chroma_origin(mi_r, mi_c);
    let (cbw, cbh) = match crate::cdf::get_plane_residual_size(
        b_size,
        1,
        recon.subsampling_x,
        recon.subsampling_y,
    ) {
        Some(psz) => (NUM_4X4_BLOCKS_WIDE[psz] * 4, NUM_4X4_BLOCKS_HIGH[psz] * 4),
        None => (0, 0),
    };
    let chroma_tx = if lossless {
        TX_4X4
    } else {
        get_tx_size(1, luma_tx, b_size, recon.subsampling_x, recon.subsampling_y).unwrap_or(TX_4X4)
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
    let (lift_x_sh, lift_y_sh) = (
        2 + u32::from(recon.subsampling_x),
        2 + u32::from(recon.subsampling_y),
    );
    for plane in 1..usize::from(recon.num_planes) {
        if !has_chroma {
            break;
        }
        let mut ty = 0usize;
        while ty < cbh {
            let mut tx = 0usize;
            while tx < cbw {
                let (tr, tc) = (crow0 + ty, ccol0 + tx);
                // §5.11.40 inter-chroma `TxType`: the luma cell at the
                // subsampling-lifted position `Max( MiRow/MiCol,
                // blockY/blockX << sub )`, filtered by the chroma-size
                // inter set. For >= 8×8 blocks the lift walks the
                // block's own luma TUs; on a sub-8×8 `HasChroma` leaf
                // the `Max` clip binds to the leaf origin (its own
                // first TU).
                let chroma_tt = if lossless || TX_SIZE_SQR_UP[chroma_tx] > crate::cdf::TX_32X32 {
                    DCT_DCT
                } else {
                    let lifted_x = ((tc >> 2) << lift_x_sh).saturating_sub(col0);
                    let lifted_y = ((tr >> 2) << lift_y_sh).saturating_sub(row0);
                    let luma_tt =
                        tu_type_grid[(lifted_y / lth) * tu_cols + lifted_x / ltw] as usize;
                    if crate::cdf::is_tx_type_in_set(true, chroma_set, luma_tt) {
                        luma_tt
                    } else {
                        DCT_DCT
                    }
                };
                let mut pred = vec![0u16; ctw * cth];
                for i in 0..cth {
                    for j in 0..ctw {
                        pred[i * ctw + j] = ictx.scratch[plane][(tr + i) * cw + (tc + j)];
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
    // r413 — §5.11.19/§5.11.20: a `skip == 1` leaf's segment_id is the
    // bit-silent `pred` cascade; a coded leaf commits the segment its
    // residual was quantised at.
    // r426 — under §5.9.14 `SegIdPreSkip = 1` (any inter-override
    // feature active) the segment id codes BEFORE skip, so a skip
    // leaf keeps its own committed segment.
    block.segment_id = if skip == 1 && !ictx.seg_id_pre_skip {
        ictx.segment_pred(mi_r, mi_c)
    } else {
        segment_id
    };
    block.residual_quant = residual_quant;
    block.residual_tx_type = luma_tx_types;
    if !lossless && skip == 0 && b_size > crate::cdf::BLOCK_4X4 {
        // §5.11.16 var-tx arm: one uniform tree per max-TU position
        // (`Max_Tx_Size_Rect` covers every block this driver codes).
        // BLOCK_4X4 takes the §5.11.16 else arm (`read_tx_size`) with
        // no trees.
        block.var_tx_trees = vec![uniform_var_tx_tree(MAX_TX_SIZE_RECT[b_size], depth)];
    }
    block.inter = Some(SyntaxInterBlock {
        ref_frame,
        y_mode,
        mv,
        ref_mv_idx,
        interp_filter: [filter; 2],
        skip_mode: 0,
        // r417 — inter-intra leaves commit the §5.11.29 bit-silent
        // derivation (the write pass validates the exact value).
        compound_type: match ii {
            Some((_, Some(_))) => crate::inter_pred::COMPOUND_WEDGE,
            Some(_) => crate::inter_pred::COMPOUND_INTRA,
            None => comp.ctype,
        },
        wedge_index: comp.wedge_index,
        wedge_sign: comp.wedge_sign,
        mask_type: comp.mask_type,
        interintra_mode: ii.map(|(m, _)| m),
        wedge_interintra: u8::from(matches!(ii, Some((_, Some(_))))),
        interintra_wedge_index: ii.and_then(|(_, w)| w).unwrap_or(0),
        motion_mode,
    });
    Ok(block)
}

/// Crude per-leaf rate proxy for the RD decisions — [`leaf_rate`]'s
/// coefficient model plus an MV-magnitude term for NEWMV leaves and a
/// small intra surcharge (intra blocks in inter frames also code
/// `is_inter = 0` against a heavily-inter-biased context).
fn p_leaf_rate(block: &SyntaxBlock) -> u64 {
    // r413 — a skip-mode leaf codes ONE §5.11.10 S() and nothing else
    // (every other value is derived); keep a small constant so the
    // trial can actually win against coded leaves.
    if block.inter.as_ref().is_some_and(|ib| ib.skip_mode != 0) {
        return 2;
    }
    let mut rate = leaf_rate(block);
    match &block.inter {
        Some(ib) => {
            if ib.ref_frame[0] != 1 {
                // §5.11.25 single-ref cascade surcharge on the
                // non-LAST reference.
                rate += 1;
            }
            // r415 — §5.11.29 masked side-data surcharge.
            match ib.compound_type {
                crate::inter_pred::COMPOUND_WEDGE => rate += 6,
                crate::inter_pred::COMPOUND_DIFFWTD => rate += 3,
                _ => {}
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
        SyntaxNode::Horz(_) | SyntaxNode::Vert(_) => {
            4 + node
                .asymmetric_blocks()
                .iter()
                .map(|b| p_leaf_rate(b))
                .sum::<u64>()
        }
        // r413 — the EXT-alphabet shapes carry a slightly larger
        // partition-symbol weight.
        _ => {
            5 + node
                .asymmetric_blocks()
                .iter()
                .map(|b| p_leaf_rate(b))
                .sum::<u64>()
        }
    }
}

/// r426 — the §5.11.22 INTRA fallback candidate with per-segment
/// lossless awareness: a leaf overlapping the exactness-demand mask
/// codes its intra arm ON THE LOSSLESS SEGMENT — the KEY driver runs
/// with the segment's quantiser (`qindex = 0`) and the `Lossless`
/// leaf shape (TX_4X4 / WHT / row-major §5.11.34 walk), and every
/// candidate is priced carrying the committed segment id — while
/// every other leaf keeps the r413 segment-0 shape. Skip leaves take
/// the §5.11.20 bit-silent pred cascade on both arms (a skip leaf in
/// a lossless coding pass has an all-zero qindex-0 residual, i.e. an
/// EXACT bare prediction, so the demand contract holds whatever
/// segment the cascade yields).
#[allow(clippy::too_many_arguments)]
fn encode_intra_candidate(
    r: u32,
    c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    ictx: &PSearchCtx,
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    if !ictx.demands_exact(r, c, b_size) {
        let mut leaf = encode_leaf_sq(r, c, b_size, input, recon, pricing)?;
        // r413 — a skip intra leaf inherits the §5.11.20 `pred`
        // (bit-silent); a coded intra leaf keeps segment 0 (its
        // residual ran at the frame quantiser — `alt_q[ 0 ] == 0`).
        if !ictx.seg_alt_q.is_empty() && leaf.skip == 1 && !ictx.seg_id_pre_skip {
            leaf.segment_id = ictx.segment_pred(r, c);
            // r426 — a lossless pred segment flips §5.11.15 to the
            // bit-silent TX_4X4 default; the skip leaf's lossy tx
            // commitment reverts to `None`.
            if ictx.seg_lossless(leaf.segment_id) {
                leaf.tx_size = None;
            }
        }
        return Ok(leaf);
    }
    let ll_seg = ictx
        .lossless_seg
        .expect("demand mask requires a lossless segment (validated at arm time)");
    // Scoped swap: the KEY driver reads its frame arm from
    // `recon.qp` / `recon.lossless`; the demanded leaf runs on the
    // lossless segment's qindex-0 configuration.
    let saved_qp = recon.qp;
    let saved_lossless = recon.lossless;
    recon.qp = QuantizerParams::neutral(0, 8);
    recon.lossless = true;
    let out = crate::encoder::key_frame::encode_leaf_sq_seg(
        r,
        c,
        b_size,
        input,
        recon,
        pricing,
        Some(ll_seg),
    );
    recon.qp = saved_qp;
    recon.lossless = saved_lossless;
    let mut leaf = out?;
    // The §5.11.19 skip arm is bit-silent — the override never
    // survives on a skip leaf (already applied inside the ladder for
    // priced candidates; re-derive here for the heuristic path).
    // r426: under SegIdPreSkip the id codes pre-skip and stands.
    if leaf.skip == 1 && !ictx.seg_id_pre_skip {
        leaf.segment_id = ictx.segment_pred(r, c);
    }
    Ok(leaf)
}

/// Recursive §5.11.4 rate-distortion partition search for a P-frame:
/// at every fully-in-frame square node from `BLOCK_64X64` down to
/// `BLOCK_8X8`, trial-encode an INTER leaf and an INTRA leaf against
/// the same starting state, and (above `BLOCK_8X8`) both against a
/// `PARTITION_SPLIT` of four recursively-searched quadrants, keeping
/// the lowest `D + λ·R`. `BLOCK_8X8` is the P-frame leaf floor (see
/// the module docs). Frame-edge nodes take the §5.11.4 forced split.
#[allow(clippy::too_many_arguments)]
fn build_p_search_tree(
    r: u32,
    c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    ictx: &mut PSearchCtx,
    twin: &mut RateTwin,
    params: &SyntaxFrameParams,
    model: RateModel,
) -> Result<(SyntaxNode, u64), Error> {
    if r >= recon.mi_rows || c >= recon.mi_cols {
        return Ok((SyntaxNode::dummy_oob(), 0));
    }
    // r416 — sub-8×8 floor: a BLOCK_4X4 node (reachable only through
    // PARTITION_SPLIT at BLOCK_8X8, and always fully inside — dims
    // are multiples of 8) is a single leaf. r417 — the r416
    // inter-only policy is lifted: the node RD-trials the §5.11.22
    // INTRA arm against the fully-searched inter leaf (the same
    // A-vs-B comparison the >= 8×8 nodes run). A mixed 2×2 group
    // (`RefFrames[ .. ][ 0 ] == INTRA_FRAME` on any cell) drives the
    // §5.11.33 `someUseIntra` chroma arm on the group's inter
    // `HasChroma` leaf — the search predicts through the decoder's
    // own driver over the live grids, so the split is exact on both
    // sides.
    if b_size == BLOCK_4X4 {
        let lambda = lambda_for(&recon.qp);
        // Group-aligned snapshot rect: a `HasChroma` 4×4 leaf codes
        // the WHOLE 2×2 group's chroma (§5.11.38 plane residual size
        // at the lifted origin) — trials must not leak between arms.
        let (sr, sc) = (r & !1, c & !1);
        let before = save_region_wh(recon, sr, sc, 2, 2);
        let m_before = ictx.mirror.snapshot_encoder_stamp_rect(r, c, 1);

        // Candidate A: the fully-searched INTER leaf.
        let inter_leaf = encode_inter_leaf(
            r,
            c,
            b_size,
            input,
            recon,
            ictx,
            (model == RateModel::Twin).then_some((&*twin, params)),
        )?;
        ictx.stamp_leaf(
            &inter_leaf,
            r,
            c,
            b_size,
            ictx.seg_lossless(inter_leaf.segment_id),
        );
        let d_inter = region_distortion_wh(recon, input, sr, sc, 2, 2);
        let h_rate_inter = p_leaf_rate(&inter_leaf);
        let node_inter = SyntaxNode::Leaf(Box::new(inter_leaf));
        let mut twin_inter = twin.clone();
        let cost_inter = twin_inter.commit_subtree(&node_inter, r, c, b_size, params)?;
        let score_inter = match model {
            RateModel::Twin => score256(d_inter, lambda, cost_inter),
            RateModel::Heuristic => score256(d_inter, lambda, h_rate_inter * 256),
        };
        let after_inter = save_region_wh(recon, sr, sc, 2, 2);
        let m_after_inter = ictx.mirror.snapshot_encoder_stamp_rect(r, c, 1);
        restore_region(recon, sr, sc, &before);
        ictx.mirror.restore_encoder_stamp_rect(&m_before);

        // Candidate B: the §5.11.22 INTRA leaf (the KEY driver's 4×4
        // arm — group-chroma coding on the `HasChroma` SE cell).
        let intra_leaf = encode_intra_candidate(
            r,
            c,
            b_size,
            input,
            recon,
            ictx,
            (model == RateModel::Twin).then_some((&*twin, params)),
        )?;
        ictx.stamp_leaf(
            &intra_leaf,
            r,
            c,
            b_size,
            ictx.seg_lossless(intra_leaf.segment_id),
        );
        let d_intra = region_distortion_wh(recon, input, sr, sc, 2, 2);
        let h_rate_intra = p_leaf_rate(&intra_leaf);
        let node_intra = SyntaxNode::Leaf(Box::new(intra_leaf));
        let mut twin_intra = twin.clone();
        let cost_intra = twin_intra.commit_subtree(&node_intra, r, c, b_size, params)?;
        let score_intra = match model {
            RateModel::Twin => score256(d_intra, lambda, cost_intra),
            RateModel::Heuristic => score256(d_intra, lambda, h_rate_intra * 256),
        };

        if score_inter <= score_intra {
            restore_region(recon, sr, sc, &after_inter);
            ictx.mirror.restore_encoder_stamp_rect(&m_after_inter);
            let SyntaxNode::Leaf(inter_leaf) = &node_inter else {
                unreachable!("candidate A is a leaf");
            };
            // Re-stamp the inter winner's mirror footprint (the intra
            // trial overwrote the mi cell); the driver grids already
            // hold the winner's stamps ([`encode_inter_leaf`]'s
            // final-predict guarantee — candidate B never touches
            // them).
            ictx.stamp_leaf(
                inter_leaf,
                r,
                c,
                b_size,
                ictx.seg_lossless(inter_leaf.segment_id),
            );
            *twin = twin_inter;
            return Ok((node_inter, cost_inter));
        }
        // Driver-grid intra stamps for the committed winner: a later
        // group cell's §5.11.33 `someUseIntra` scan must see
        // `RefFrames[ .. ][ 0 ] == INTRA_FRAME` here.
        let SyntaxNode::Leaf(intra_leaf) = &node_intra else {
            unreachable!("candidate B is a leaf");
        };
        ictx.stamp_intra_leaf_grids(r, c, b_size, intra_leaf.y_mode);
        *twin = twin_intra;
        return Ok((node_intra, cost_intra));
    }
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
    let half = n4 >> 1;
    let sub = crate::cdf::partition_subsize(crate::cdf::PARTITION_SPLIT, b_size)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let fully_inside = r + n4 <= recon.mi_rows && c + n4 <= recon.mi_cols;

    if !fully_inside {
        // §5.11.4 forced arms (dims are multiples of 8, so BLOCK_8X8
        // nodes are always fully inside and the recursion terminates;
        // the partition arm may still cost bits — the `split_or_*`
        // half-straddle bools).
        let mut cost = twin.commit_partition_symbol(crate::cdf::PARTITION_SPLIT, r, c, b_size)?;
        let (nw, c0) = build_p_search_tree(r, c, sub, input, recon, ictx, twin, params, model)?;
        let (ne, c1) =
            build_p_search_tree(r, c + half, sub, input, recon, ictx, twin, params, model)?;
        let (sw, c2) =
            build_p_search_tree(r + half, c, sub, input, recon, ictx, twin, params, model)?;
        let (se, c3) = build_p_search_tree(
            r + half,
            c + half,
            sub,
            input,
            recon,
            ictx,
            twin,
            params,
            model,
        )?;
        cost += c0 + c1 + c2 + c3;
        let children = [Box::new(nw), Box::new(ne), Box::new(sw), Box::new(se)];
        return Ok((SyntaxNode::Split(children), cost));
    }

    let lambda = lambda_for(&recon.qp);
    let before = save_region(recon, r, c, n4 as usize);
    // r412: the driver mirror follows the same trial / rollback
    // discipline as the pixel planes — every candidate leaves the
    // mirror holding ITS committed stamps, and the loser's stamps are
    // rolled back so subsequent §7.10.2 scans see only the winner.
    // r419: the §5.11.5 driver grids join the discipline — the OBMC
    // neighbour scan reads committed above/left cells through them,
    // so a losing trial's stamps must not survive the node.
    let m_before = ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4);
    let g_before = ictx.snapshot_grids_rect(r, c, n4);

    // Per-candidate scoring: exact twin bits (the candidate node's
    // full symbol sequence committed to a twin fork) under the twin
    // model, the pre-r421 proxies (scaled by 256 — identical
    // decisions) under the heuristic model.
    let score_of = |d: u64, cost256: u64, h_rate: u64| -> u64 {
        match model {
            RateModel::Twin => score256(d, lambda, cost256),
            RateModel::Heuristic => score256(d, lambda, h_rate * 256),
        }
    };

    // Candidate A: one INTER leaf.
    let inter_leaf = encode_inter_leaf(
        r,
        c,
        b_size,
        input,
        recon,
        ictx,
        (model == RateModel::Twin).then_some((&*twin, params)),
    )?;
    ictx.stamp_leaf(
        &inter_leaf,
        r,
        c,
        b_size,
        ictx.seg_lossless(inter_leaf.segment_id),
    );
    let d_inter = region_distortion(recon, input, r, c, n4 as usize);
    let h_rate_inter = p_leaf_rate(&inter_leaf);
    let node_inter = SyntaxNode::Leaf(Box::new(inter_leaf));
    let mut twin_inter = twin.clone();
    let cost_inter = twin_inter.commit_subtree(&node_inter, r, c, b_size, params)?;
    let score_inter = score_of(d_inter, cost_inter, h_rate_inter);
    let after_inter = save_region(recon, r, c, n4 as usize);
    let m_after_inter = ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4);
    let g_after_inter = ictx.snapshot_grids_rect(r, c, n4);
    restore_region(recon, r, c, &before);
    ictx.mirror.restore_encoder_stamp_rect(&m_before);
    ictx.restore_grids_rect(&g_before);

    // Candidate B: one INTRA leaf (§5.11.22 arm) with the KEY
    // driver's §5.11.15 tx_depth RD search (TX_MODE_SELECT on the
    // lossy arm; the lossless arm stays on the TX_4X4 grid).
    let intra_leaf = encode_intra_candidate(
        r,
        c,
        b_size,
        input,
        recon,
        ictx,
        (model == RateModel::Twin).then_some((&*twin, params)),
    )?;
    ictx.stamp_leaf(
        &intra_leaf,
        r,
        c,
        b_size,
        ictx.seg_lossless(intra_leaf.segment_id),
    );
    // r419 — driver-grid intra stamps (the decode walker's grid-fill
    // twins): a later leaf's OBMC neighbour scan must see
    // `RefFrames[ .. ][ 0 ] == INTRA_FRAME` on a committed intra
    // region, exactly as the decoder will.
    ictx.stamp_intra_leaf_grids(r, c, b_size, intra_leaf.y_mode);
    let d_intra = region_distortion(recon, input, r, c, n4 as usize);
    let h_rate_intra = p_leaf_rate(&intra_leaf);
    let node_intra = SyntaxNode::Leaf(Box::new(intra_leaf));
    let mut twin_intra = twin.clone();
    let cost_intra = twin_intra.commit_subtree(&node_intra, r, c, b_size, params)?;
    let score_intra = score_of(d_intra, cost_intra, h_rate_intra);
    let after_intra = save_region(recon, r, c, n4 as usize);
    let m_after_intra = ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4);
    let g_after_intra = ictx.snapshot_grids_rect(r, c, n4);
    restore_region(recon, r, c, &before);
    ictx.mirror.restore_encoder_stamp_rect(&m_before);
    ictx.restore_grids_rect(&g_before);

    // Running best over the non-split candidates: NONE-leaf first,
    // then the r412 HORZ / VERT rectangular trials. Each entry
    // carries its committed twin fork + exact cost.
    let mut best_node: (
        SyntaxNode,
        RegionSnapshot,
        _,
        DriverGridSnapshot,
        u64,
        RateTwin,
        u64,
    ) = if score_inter <= score_intra {
        (
            node_inter,
            after_inter,
            m_after_inter,
            g_after_inter,
            score_inter,
            twin_inter,
            cost_inter,
        )
    } else {
        (
            node_intra,
            after_intra,
            m_after_intra,
            g_after_intra,
            score_intra,
            twin_intra,
            cost_intra,
        )
    };

    // Candidates D..K: every asymmetric §5.11.4 partition — r412
    // PARTITION_HORZ / PARTITION_VERT (two rectangular INTER halves)
    // plus the r413 EXT-alphabet shapes (HORZ_A / HORZ_B / VERT_A /
    // VERT_B T-shapes with mixed splitSize/subSize leaves, and the
    // HORZ_4 / VERT_4 four-strip shapes at BLOCK_16X16..=BLOCK_64X64).
    // Sub-blocks are encoded in §5.11.4 dispatch order, so each later
    // block's §7.10.2 scan and prediction see the earlier blocks'
    // stamps exactly as the decoder will. Intra sub-blocks stay the
    // NONE-leaf's and SPLIT's territory.
    let quarter = half >> 1;
    let split_sz = sub;
    for part in [
        PARTITION_HORZ,
        PARTITION_VERT,
        crate::cdf::PARTITION_HORZ_A,
        crate::cdf::PARTITION_HORZ_B,
        crate::cdf::PARTITION_VERT_A,
        crate::cdf::PARTITION_VERT_B,
        crate::cdf::PARTITION_HORZ_4,
        crate::cdf::PARTITION_VERT_4,
    ] {
        let psub = match crate::cdf::partition_subsize(part, b_size) {
            Some(sz) => sz,
            None => continue,
        };
        // §5.11.4: the EXT alphabet (T-shapes + HORZ_4 / VERT_4) only
        // exists above BLOCK_8X8 — the 8×8 partition symbol is the
        // 4-value NONE / HORZ / VERT / SPLIT alphabet. (r416: the
        // former sub-8 strip skip is lifted — HORZ / VERT at 8×8 code
        // 8×4 / 4×8 leaves and HORZ_4 / VERT_4 at 16×16 code 16×4 /
        // 4×16 strips.)
        if b_size == BLOCK_8X8 && !matches!(part, PARTITION_HORZ | PARTITION_VERT) {
            continue;
        }
        // r427 — §5.11.38 admissibility: under 4:2:2 the tall shapes
        // (VERT 32×64, VERT_4 4×16 strips, ...) subsample to
        // BLOCK_INVALID chroma and must never be elected (see
        // `key_frame::chroma_partition_ok`).
        if !crate::encoder::key_frame::chroma_partition_ok(recon, psub) {
            continue;
        }
        // §5.11.4 sub-block geometry per partition, in dispatch order.
        let cells: Vec<(u32, u32, usize)> = match part {
            PARTITION_HORZ => vec![(r, c, psub), (r + half, c, psub)],
            PARTITION_VERT => vec![(r, c, psub), (r, c + half, psub)],
            crate::cdf::PARTITION_HORZ_A => vec![
                (r, c, split_sz),
                (r, c + half, split_sz),
                (r + half, c, psub),
            ],
            crate::cdf::PARTITION_HORZ_B => vec![
                (r, c, psub),
                (r + half, c, split_sz),
                (r + half, c + half, split_sz),
            ],
            crate::cdf::PARTITION_VERT_A => vec![
                (r, c, split_sz),
                (r + half, c, split_sz),
                (r, c + half, psub),
            ],
            crate::cdf::PARTITION_VERT_B => vec![
                (r, c, psub),
                (r, c + half, split_sz),
                (r + half, c + half, split_sz),
            ],
            crate::cdf::PARTITION_HORZ_4 => (0..4u32).map(|i| (r + quarter * i, c, psub)).collect(),
            _ => (0..4u32).map(|i| (r, c + quarter * i, psub)).collect(),
        };
        // r421 — a RUNNING twin fork threads the shape's blocks in
        // §5.11.4 dispatch order: each later block's leaf-internal
        // pricing AND the writer's own commitment validation (the
        // §5.11.27 cascade in particular) see the earlier siblings'
        // committed mirror stamps, exactly like the emitting pass.
        let mut twin_shape = twin.clone();
        let mut cost_shape = twin_shape.commit_partition_symbol(part, r, c, b_size)?;
        let mut blocks: Vec<SyntaxBlock> = Vec::with_capacity(cells.len());
        let mut h_rate = if cells.len() == 2 { 4 } else { 5 };
        for &(rr, cc, sz) in &cells {
            let blk = encode_inter_leaf(
                rr,
                cc,
                sz,
                input,
                recon,
                ictx,
                (model == RateModel::Twin).then_some((&twin_shape, params)),
            )?;
            ictx.stamp_leaf(&blk, rr, cc, sz, ictx.seg_lossless(blk.segment_id));
            h_rate += p_leaf_rate(&blk);
            cost_shape += twin_shape.commit_block(&blk, rr, cc, sz, params)?;
            blocks.push(blk);
        }
        let d = region_distortion(recon, input, r, c, n4 as usize);
        let bx = |b: SyntaxBlock| Box::new(b);
        let node = match part {
            PARTITION_HORZ | PARTITION_VERT => {
                let mut it = blocks.into_iter();
                let pair = [bx(it.next().unwrap()), bx(it.next().unwrap())];
                if part == PARTITION_HORZ {
                    SyntaxNode::Horz(pair)
                } else {
                    SyntaxNode::Vert(pair)
                }
            }
            crate::cdf::PARTITION_HORZ_A
            | crate::cdf::PARTITION_HORZ_B
            | crate::cdf::PARTITION_VERT_A
            | crate::cdf::PARTITION_VERT_B => {
                let mut it = blocks.into_iter();
                let trio = [
                    bx(it.next().unwrap()),
                    bx(it.next().unwrap()),
                    bx(it.next().unwrap()),
                ];
                match part {
                    crate::cdf::PARTITION_HORZ_A => SyntaxNode::HorzA(trio),
                    crate::cdf::PARTITION_HORZ_B => SyntaxNode::HorzB(trio),
                    crate::cdf::PARTITION_VERT_A => SyntaxNode::VertA(trio),
                    _ => SyntaxNode::VertB(trio),
                }
            }
            _ => {
                let mut it = blocks.into_iter();
                let four = [
                    bx(it.next().unwrap()),
                    bx(it.next().unwrap()),
                    bx(it.next().unwrap()),
                    bx(it.next().unwrap()),
                ];
                if part == crate::cdf::PARTITION_HORZ_4 {
                    SyntaxNode::Horz4(four)
                } else {
                    SyntaxNode::Vert4(four)
                }
            }
        };
        let score = score_of(d, cost_shape, h_rate);
        if score < best_node.4 {
            best_node = (
                node,
                save_region(recon, r, c, n4 as usize),
                ictx.mirror.snapshot_encoder_stamp_rect(r, c, n4),
                ictx.snapshot_grids_rect(r, c, n4),
                score,
                twin_shape,
                cost_shape,
            );
        }
        restore_region(recon, r, c, &before);
        ictx.mirror.restore_encoder_stamp_rect(&m_before);
        ictx.restore_grids_rect(&g_before);
    }
    let (
        node_leafish,
        after_leafish,
        m_after_leafish,
        g_after_leafish,
        score_leafish,
        twin_leafish,
        cost_leafish,
    ) = best_node;

    // Candidate C: PARTITION_SPLIT into four recursively-searched
    // quadrants (NW/NE/SW/SE dispatch order — the mirror currently
    // holds the pre-node state, so each child's §7.10.2 scan sees its
    // earlier siblings' committed stamps exactly as the write pass
    // will; the twin fork enters each child holding the SPLIT arm +
    // earlier siblings' symbols, exactly like the writer).
    let mut twin_split = twin.clone();
    let mut cost_split =
        twin_split.commit_partition_symbol(crate::cdf::PARTITION_SPLIT, r, c, b_size)?;
    let (nw, c0) = build_p_search_tree(
        r,
        c,
        sub,
        input,
        recon,
        ictx,
        &mut twin_split,
        params,
        model,
    )?;
    let (ne, c1) = build_p_search_tree(
        r,
        c + half,
        sub,
        input,
        recon,
        ictx,
        &mut twin_split,
        params,
        model,
    )?;
    let (sw, c2) = build_p_search_tree(
        r + half,
        c,
        sub,
        input,
        recon,
        ictx,
        &mut twin_split,
        params,
        model,
    )?;
    let (se, c3) = build_p_search_tree(
        r + half,
        c + half,
        sub,
        input,
        recon,
        ictx,
        &mut twin_split,
        params,
        model,
    )?;
    cost_split += c0 + c1 + c2 + c3;
    let children = [Box::new(nw), Box::new(ne), Box::new(sw), Box::new(se)];
    let d_split = region_distortion(recon, input, r, c, n4 as usize);
    let h_rate_split = children.iter().map(|ch| p_tree_rate(ch)).sum::<u64>() + 4;
    let score_split = score_of(d_split, cost_split, h_rate_split);

    if score_leafish <= score_split {
        restore_region(recon, r, c, &after_leafish);
        ictx.mirror.restore_encoder_stamp_rect(&m_after_leafish);
        ictx.restore_grids_rect(&g_after_leafish);
        *twin = twin_leafish;
        Ok((node_leafish, cost_leafish))
    } else {
        *twin = twin_split;
        Ok((SyntaxNode::Split(children), cost_split))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::yuv_frame::ChromaFormat;

    /// 8-bit view of a `u16` recon plane for byte-comparisons against
    /// the spec driver's 8-bit output surface.
    fn plane8(p: &[u16]) -> Vec<u8> {
        p.iter().map(|&s| s as u8).collect()
    }

    /// Test shims over the r427 general GOP core (the historical
    /// per-surface entry shapes, YuvFrame inputs).
    fn gop_seg_yuv(frames: &[YuvFrame], q: u8, alt_q: &[i16]) -> Result<EncodedGopYuv, Error> {
        encode_gop_yuv_seg_extras_tuned(frames, q, alt_q, &[], false, None, GopTuning::default())
            .map(|t| t.gop)
    }

    /// r421 — persistent per-tile search state for the module tests: a
    /// fresh rate-twin source (§8.3.1 default CDFs at the recon's
    /// quantiser, whole-frame single-tile write state, live emitting
    /// writer) plus the frame-scope parameter bundle from the search
    /// context's own inter params. Multi-superblock tests MUST reuse
    /// one of these across superblocks — the twin's mirror carries the
    /// earlier superblocks' committed stamps exactly like the real
    /// driver's.
    struct TestTileState {
        cdfs: TileCdfContext,
        state: PartitionSyntaxWriter,
        writer: SymbolWriter,
        params: SyntaxFrameParams,
    }

    fn test_tile_state(recon: &ReconState, ictx: &PSearchCtx) -> TestTileState {
        let writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.init_coeff_cdfs(recon.qp.base_q_idx);
        let state = PartitionSyntaxWriter::new(
            recon.mi_rows,
            recon.mi_cols,
            TileGeometry {
                mi_row_start: 0,
                mi_row_end: recon.mi_rows,
                mi_col_start: 0,
                mi_col_end: recon.mi_cols,
            },
        )
        .expect("test geometry is valid");
        let params = SyntaxFrameParams {
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            seg_id_pre_skip: false,
            segmentation_enabled: !ictx.seg_alt_q.is_empty(),
            seg_ref_frame: [None; crate::uncompressed_header_tail::MAX_SEGMENTS],
            seg_skip: [false; crate::uncompressed_header_tail::MAX_SEGMENTS],
            seg_globalmv: [false; crate::uncompressed_header_tail::MAX_SEGMENTS],
            last_active_seg_id: ictx.seg_alt_q.len().saturating_sub(1) as u8,
            lossless_array: seg_lossless_array(recon.qp.base_q_idx, &ictx.seg_alt_q),
            coded_lossless: recon.lossless,
            enable_cdef: false,
            allow_intrabc: false,
            cdef_bits: 0,
            read_deltas: false,
            use_128x128_superblock: false,
            delta_q_res: 0,
            delta_lf_present: false,
            delta_lf_multi: false,
            mono_chrome: false,
            delta_lf_res: 0,
            allow_screen_content_tools: recon.allow_screen_content_tools,
            enable_filter_intra: true,
            bit_depth: 8,
            tx_mode_select: !recon.lossless,
            quant: recon.qp,
            reduced_tx_set: false,
            inter: Some(ictx.ip.clone()),
        };
        TestTileState {
            cdfs,
            state,
            writer,
            params,
        }
    }

    /// Search one superblock under the twin model, then COMMIT it to
    /// the persistent tile state through the real write pass and
    /// assert the anti-desync invariant — the driver loop's exact
    /// shape.
    fn search_and_commit_sb(
        ts: &mut TestTileState,
        r: u32,
        c: u32,
        input: &YuvFrame,
        recon: &mut ReconState,
        ictx: &mut PSearchCtx,
    ) -> Result<SyntaxNode, Error> {
        ts.state.arm_read_deltas();
        let mut twin = RateTwin::snapshot(&ts.cdfs, &ts.state, &ts.writer);
        twin.arm_read_deltas();
        let (node, _) = build_p_search_tree(
            r,
            c,
            BLOCK_64X64,
            input,
            recon,
            ictx,
            &mut twin,
            &ts.params,
            RateModel::Twin,
        )?;
        write_partition_tree_syntax(
            &mut ts.writer,
            &mut ts.cdfs,
            &mut ts.state,
            &node,
            r,
            c,
            BLOCK_64X64,
            &ts.params,
        )?;
        assert!(
            twin.matches(&ts.cdfs, &ts.writer),
            "rate twin desynced from the test tile writer after superblock ({r},{c})"
        );
        Ok(node)
    }

    /// Single-superblock convenience wrapper (the pre-r421 call shape).
    fn build_p_search_tree_for_tests(
        r: u32,
        c: u32,
        b_size: usize,
        input: &YuvFrame,
        recon: &mut ReconState,
        ictx: &mut PSearchCtx,
    ) -> Result<SyntaxNode, Error> {
        assert_eq!(b_size, BLOCK_64X64, "tests search whole superblocks");
        let mut ts = test_tile_state(recon, ictx);
        search_and_commit_sb(&mut ts, r, c, input, recon, ictx)
    }

    /// Deterministic textured frame with translation `(sy, sx)` — the
    /// `gop_inter_conformance.rs` generator, duplicated here so the
    /// module test can drive the search internals directly.
    fn moving_gradient(w: u32, h: u32, shift_y: usize, shift_x: usize, seed: u32) -> YuvFrame {
        let (wu, hu) = (w as usize, h as usize);
        let s = seed as usize;
        let mut f = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 0);
        for i in 0..hu {
            for j in 0..wu {
                let (si, sj) = (i + shift_y, j + shift_x);
                f.y[i * wu + j] = ((si * 5 + sj * 3 + (si / 16) * (sj / 16) + s) % 256) as u16;
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        for i in 0..ch {
            for j in 0..cw {
                let (si, sj) = (i + shift_y / 2, j + shift_x / 2);
                f.u[i * cw + j] = ((128 + si * 2 + sj + s) % 256) as u16;
                f.v[i * cw + j] = ((64 + si + sj * 2 + s) % 256) as u16;
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
            other => {
                let blocks = other.asymmetric_blocks();
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
        let key = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f0, base_q_idx).unwrap();
        let reference = GopFrameReconYuv {
            y: key.recon_y,
            u: key.recon_u,
            v: key.recon_v,
        };
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let qp = QuantizerParams::neutral(base_q_idx, 8);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp,
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        let mut ictx = PSearchCtx::new(
            &reference,
            &reference,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f1, &mut recon, &mut ictx).unwrap();
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
        let key = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f0, base_q_idx).unwrap();
        let reference = GopFrameReconYuv {
            y: key.recon_y,
            u: key.recon_u,
            v: key.recon_v,
        };
        // Target frame = SHARP half-pel interpolation of the reference.
        let mut f1 = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        {
            let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
            let mut probe = PSearchCtx::new(
                &reference,
                &reference,
                16,
                16,
                64,
                64,
                ip,
                1,
                base_q_idx,
                &[],
            )
            .unwrap();
            probe
                .predict_leaf(
                    0,
                    0,
                    BLOCK_64X64,
                    [1, -1],
                    MODE_NEWMV,
                    [[0, 4], [0, 0]],
                    EIGHTTAP_SHARP,
                    CompoundSel::AVERAGE,
                    None,
                    MotionSel::Simple,
                )
                .unwrap();
            for (dst, &src) in f1.y.iter_mut().zip(probe.scratch[0].iter()) {
                *dst = src;
            }
            for (dst, &src) in f1.u.iter_mut().zip(probe.scratch[1].iter()) {
                *dst = src;
            }
            for (dst, &src) in f1.v.iter_mut().zip(probe.scratch[2].iter()) {
                *dst = src;
            }
        }
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        let mut ictx = PSearchCtx::new(
            &reference,
            &reference,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f1, &mut recon, &mut ictx).unwrap();
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
                other => {
                    let blocks = other.asymmetric_blocks();
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
        let band = |k: usize| -> YuvFrame {
            let (w, h) = (64usize, 64usize);
            let mut f = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
            let tex = |x: i64, y: i64| -> u16 {
                let (xu, yu) = ((x + 512) as usize, (y + 512) as usize);
                ((xu * 7 + yu * 3 + (xu / 8) * (yu / 8) * 5) % 256) as u16
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
                    f.u[i * cw + j] = ((128 + x * 2 + i) % 256) as u16;
                    f.v[i * cw + j] = ((64 + x + i * 2) % 256) as u16;
                }
            }
            f
        };
        let frames: Vec<YuvFrame> = (0..3).map(band).collect();
        let base_q_idx = 60u8;

        // Tree inspection on the P-frame search.
        let key =
            crate::encoder::key_frame::encode_key_frame_yuv_with_q(&frames[0], base_q_idx).unwrap();
        let reference = GopFrameReconYuv {
            y: key.recon_y.clone(),
            u: key.recon_u.clone(),
            v: key.recon_v.clone(),
        };
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        let mut ictx = PSearchCtx::new(
            &reference,
            &reference,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &frames[1], &mut recon, &mut ictx)
                .unwrap();
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
                // r413 — the EXT shapes count as neither HORZ nor VERT
                // for this witness.
                _ => {}
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
        let enc = encode_gop_yuv_with_q(&frames, base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), frames.len());
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(f.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(f.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
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
        let mut flash = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for v in flash.y.iter_mut() {
            *v = (next() & 0xFF) as u16;
        }
        // frames: KEY(calm), P1(flash), P2(calm again).
        let frames = vec![calm.clone(), flash, calm];
        let base_q_idx = 90u8;

        // Drive the P2 search directly: prev = P1 recon (flash),
        // prevprev = P1's prev = KEY recon (calm).
        let enc = encode_gop_yuv_with_q(&frames, base_q_idx).unwrap();
        let prev = enc.recon[1].clone();
        let prevprev = enc.recon[0].clone();
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&enc.seq, 64, 64, base_q_idx, 2, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        let mut ictx = PSearchCtx::new(
            &prev,
            &prevprev,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            2,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &frames[2], &mut recon, &mut ictx)
                .unwrap();
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
                other => {
                    let blocks = other.asymmetric_blocks();
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
            assert_eq!(f.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(f.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(f.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
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
        let pre = encode_gop_yuv_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();
        let mut f2 = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        let avg = |a: &[u16], b: &[u16], out: &mut [u16]| {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = (u32::from(x) + u32::from(y)).div_ceil(2) as u16;
            }
        };
        avg(&pre.recon[0].y, &pre.recon[1].y, &mut f2.y);
        avg(&pre.recon[0].u, &pre.recon[1].u, &mut f2.u);
        avg(&pre.recon[0].v, &pre.recon[1].v, &mut f2.v);

        // Drive the P2 search directly (prev = f1 recon in LAST,
        // prevprev = KEY recon in GOLDEN).
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 2, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
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
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f2, &mut recon, &mut ictx).unwrap();
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
                other => {
                    let blocks = other.asymmetric_blocks();
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
        let enc = encode_gop_yuv_with_q(&[f0, f2], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(f.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(f.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
    }

    /// r413 — static content on a skip-mode frame must select
    /// §5.11.10 `skip_mode = 1` leaves (the one-symbol pure-derivation
    /// block beats every coded candidate once the two reference
    /// reconstructions have converged), and the resulting stream must
    /// stay byte-exact through the spec driver.
    #[test]
    fn r413_p_search_selects_skip_mode_on_static_content() {
        let f = moving_gradient(64, 64, 2, 3, 21);
        let base_q_idx = 60u8;
        let frames = vec![f.clone(), f.clone(), f.clone(), f.clone()];
        let pre = encode_gop_yuv_with_q(&frames[..3], base_q_idx).unwrap();

        // Drive the P3 search directly (LAST = P2 recon, GOLDEN = P1
        // recon; `p_index = 3` ⇒ skip_mode_present per §5.9.22).
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 3, &[]);
        assert_eq!(fh.skip_mode_present, Some(true), "P3 header presence");
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.order_hints = gop_order_hints(3, pre.seq.order_hint_bits);
        ip.skip_mode_present = true;
        ip.skip_mode_frame = [1, 4];
        let mut ictx = PSearchCtx::new(
            &pre.recon[2],
            &pre.recon[1],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            3,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f, &mut recon, &mut ictx).unwrap();
        fn count_skip_mode(node: &SyntaxNode, sm: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, sm: &mut u32, other: &mut u32| {
                if b.inter.as_ref().is_some_and(|ib| ib.skip_mode != 0) {
                    *sm += 1;
                } else {
                    *other += 1;
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, sm, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count_skip_mode(ch, sm, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, sm, other);
                    }
                }
            }
        }
        let (mut sm, mut other) = (0u32, 0u32);
        count_skip_mode(&tree, &mut sm, &mut other);
        assert!(
            sm > 0,
            "static content on a skip-mode frame must select skip_mode leaves \
             (got SKIP_MODE={sm} OTHER={other})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&frames, base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 4);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
    }

    /// r413 — SEG_LVL_ALT_Q P-frames: the activity policy must code
    /// MULTIPLE segments on mixed flat/textured content, coded leaves
    /// must quantise at their segment's q-index (byte-exact spec-driver
    /// decode proves the whole chain), and invalid configurations are
    /// rejected.
    #[test]
    fn r413_segmentation_alt_q_gop_round_trips() {
        // Mixed content: flat left half, textured right half — the
        // MAD policy assigns different segments per leaf.
        let mut frames: Vec<YuvFrame> = Vec::new();
        for k in 0..3u32 {
            let mut f = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 100);
            for i in 0..64usize {
                for j in 32..64usize {
                    f.y[i * 64 + j] = ((i * 17 + j * 31 + (k as usize) * 5) % 256) as u16;
                }
            }
            for v in f.u.iter_mut() {
                *v = 90;
            }
            for v in f.v.iter_mut() {
                *v = 160;
            }
            frames.push(f);
        }
        let base_q_idx = 60u8;
        let alt_q: [i16; 3] = [0, 24, 48];

        let enc = gop_seg_yuv(&frames, base_q_idx, &alt_q).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }

        // Segment-diversity witness: drive the P1 search directly and
        // count distinct committed segment ids.
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&enc.seq, 64, 64, base_q_idx, 1, &alt_q);
        let sp = fh.segmentation_params.as_ref().unwrap();
        assert!(sp.enabled && sp.update_map && !sp.temporal_update && sp.update_data);
        assert_eq!(sp.last_active_seg_id, 2);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: {
                let mut q = QuantizerParams::neutral(base_q_idx, 8);
                q.segmentation_enabled = true;
                for (seg, &d) in alt_q.iter().enumerate() {
                    q.seg_alt_q_active[seg] = true;
                    q.seg_alt_q_data[seg] = d;
                }
                q
            },
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.order_hints = gop_order_hints(1, enc.seq.order_hint_bits);
        ip.segmentation_update_map = true;
        let mut ictx = PSearchCtx::new(
            &enc.recon[0],
            &enc.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &alt_q,
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &frames[1], &mut recon, &mut ictx)
                .unwrap();
        fn collect_segments(node: &SyntaxNode, seen: &mut [u32; 8]) {
            match node {
                SyntaxNode::Leaf(b) => seen[b.segment_id as usize] += 1,
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        collect_segments(ch, seen);
                    }
                }
                other => {
                    let blocks = other.asymmetric_blocks();
                    for b in blocks.iter() {
                        seen[b.segment_id as usize] += 1;
                    }
                }
            }
        }
        let mut seen = [0u32; 8];
        collect_segments(&tree, &mut seen);
        let distinct = seen.iter().filter(|&&n| n > 0).count();
        assert!(
            distinct >= 2,
            "mixed content must code multiple segments (histogram: {seen:?})"
        );

        // Invalid configurations reject. (r426: a delta clamping the
        // §7.12.2 qindex to 0 is no longer an error — it is the
        // per-segment lossless election, covered by
        // `tests/mixed_lossless.rs` — but unrepresentable su(1+8)
        // deltas, over-255 sums and base-0 segmentation still do.)
        assert!(gop_seg_yuv(&frames, 60, &[1, 2]).is_err());
        assert!(gop_seg_yuv(&frames, 60, &[0]).is_err());
        assert!(gop_seg_yuv(&frames, 60, &[0, -256]).is_err());
        assert!(gop_seg_yuv(&frames, 200, &[0, 100]).is_err());
        assert!(gop_seg_yuv(&frames, 0, &[0, 10]).is_err());
    }

    /// r413 — tri-motion content shaped exactly like a §5.11.4
    /// T-shape (one clean half + two clean quarters) must select an
    /// EXT-alphabet partition over HORZ/VERT/SPLIT (3 sub-blocks beat
    /// 4 at equal distortion), and the emitted stream must stay
    /// byte-exact through the spec driver.
    #[test]
    fn r413_p_search_selects_t_shape_partitions() {
        // Frame k: top 64x32 band shifts by (0, 2k); bottom-left
        // 32x32 by (2k, 0); bottom-right 32x32 by (2k, 2k) — the
        // HORZ_B geometry at BLOCK_64X64.
        let gen = |k: usize| -> YuvFrame {
            let mut f = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
            let tex = |i: usize, j: usize| ((i * 7 + j * 13 + (i / 8) * (j / 8)) % 256) as u16;
            for i in 0..64usize {
                for j in 0..64usize {
                    let (si, sj) = if i < 32 {
                        (i, j + 2 * k)
                    } else if j < 32 {
                        (i + 2 * k, j)
                    } else {
                        (i + 2 * k, j + 2 * k)
                    };
                    f.y[i * 64 + j] = tex(si, sj);
                }
            }
            for v in f.u.iter_mut() {
                *v = 110;
            }
            for v in f.v.iter_mut() {
                *v = 150;
            }
            f
        };
        let frames: Vec<YuvFrame> = (0..3).map(gen).collect();
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(&frames[..1], base_q_idx).unwrap();
        let key_recon = GopFrameReconYuv {
            y: pre.recon[0].y.clone(),
            u: pre.recon[0].u.clone(),
            v: pre.recon[0].v.clone(),
        };

        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.order_hints = gop_order_hints(1, pre.seq.order_hint_bits);
        let mut ictx = PSearchCtx::new(
            &key_recon,
            &key_recon,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &frames[1], &mut recon, &mut ictx)
                .unwrap();
        fn count_ext(node: &SyntaxNode, ext: &mut u32) {
            match node {
                SyntaxNode::HorzA(_)
                | SyntaxNode::HorzB(_)
                | SyntaxNode::VertA(_)
                | SyntaxNode::VertB(_)
                | SyntaxNode::Horz4(_)
                | SyntaxNode::Vert4(_) => *ext += 1,
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count_ext(ch, ext);
                    }
                }
                _ => {}
            }
        }
        let mut ext = 0u32;
        count_ext(&tree, &mut ext);
        assert!(
            ext > 0,
            "tri-motion content must select an EXT-alphabet partition (tree: {tree:?})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&frames, base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
    }

    /// r413 — use_ref_frame_mvs groundwork: on a moving GOP the
    /// encoder-side §7.9 projection must produce VALID temporal
    /// candidates from P2 on (P1's store holds only the intra KEY),
    /// and the emitted streams (non-error-resilient headers,
    /// `use_ref_frame_mvs = 1`) must stay byte-exact through the spec
    /// driver — proving search mirror, write mirror and decoder all
    /// derive the same motion field.
    #[test]
    fn r413_motion_field_estimation_projects_from_p2() {
        let frames: Vec<YuvFrame> = (0..4)
            .map(|k| moving_gradient(64, 64, k * 3, k * 2, 9))
            .collect();
        let enc = encode_gop_yuv_with_q(&frames, 60).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 4);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }

        // Recompute P2's projection from a rebuilt store (KEY intra in
        // every slot, then P1's §7.19 payload in slot 0) and assert at
        // least one valid cell — moving content stores real MVs.
        let fh1 = build_p_frame_yuv420_8bit_fh_with_q(&enc.seq, 64, 64, 60, 1, &[]);
        let fs = fh1.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let key_recon = GopFrameReconYuv {
            y: enc.recon[0].y.clone(),
            u: enc.recon[0].u.clone(),
            v: enc.recon[0].v.clone(),
        };
        let mf_store: [SavedMotionField; 8] =
            core::array::from_fn(|_| SavedMotionField::intra(mi_rows, mi_cols));
        let (_, _, saved1, _, _) = encode_p_frame_yuv(
            &frames[1],
            &key_recon,
            &key_recon,
            &enc.seq,
            60,
            1,
            &[],
            &mf_store,
            RateModel::Twin,
            true,
            None,
            true,
            None,
            false,
            None,
        )
        .unwrap();
        assert!(!saved1.frame_is_intra);
        assert!(
            saved1.mf_ref_frames.iter().any(|&r| r > 0),
            "P1 must store §7.19 motion-field candidates on moving content"
        );

        use crate::inter_pred::{motion_field_estimation_core, MotionFieldSlot};
        let mut slots: [Option<MotionFieldSlot<'_>>; 8] = [None; 8];
        let intra = SavedMotionField::intra(mi_rows, mi_cols);
        for (i, s) in slots.iter_mut().enumerate() {
            let src = if i == 0 { &saved1 } else { &intra };
            *s = Some(MotionFieldSlot {
                mf_mvs: &src.mf_mvs,
                mf_ref_frames: &src.mf_ref_frames,
                saved_order_hints: src.saved_order_hints,
                mi_rows: src.mi_rows,
                mi_cols: src.mi_cols,
                frame_is_intra: src.frame_is_intra,
            });
        }
        // P2 reads LAST from slot 0 (p_index = 2), GOLDEN from slot 1.
        let mut ref_frame_idx = [0u8; 7];
        ref_frame_idx[3] = 1;
        let hints2 = gop_order_hints(2, enc.seq.order_hint_bits);
        let mut order_hints = [0i32; 8];
        order_hints.copy_from_slice(&hints2.order_hints_by_ref);
        let mfmvs = motion_field_estimation_core(
            &slots,
            &ref_frame_idx,
            &order_hints,
            2,
            u32::from(enc.seq.order_hint_bits),
            mi_rows,
            mi_cols,
        );
        let mut valid = 0u32;
        for r in 1..=7usize {
            for y8 in 0..mfmvs.h8() {
                for x8 in 0..mfmvs.w8() {
                    if mfmvs.get(r, y8, x8)[0] != crate::cdf::MFMV_INVALID {
                        valid += 1;
                    }
                }
            }
        }
        assert!(
            valid > 0,
            "P2 §7.9 projection must land valid temporal candidates"
        );
    }

    /// r412 — [`PSearchCtx::predict_leaf`] must thread the trial
    /// filter into the §7.11.3.4 kernel: an isolated impulse column
    /// interpolated at a half-pel phase produces distinct samples per
    /// filter family.
    #[test]
    fn r412_predict_leaf_threads_interp_filter() {
        use crate::inter_pred::{EIGHTTAP_SHARP, EIGHTTAP_SMOOTH};
        let mut refr = GopFrameReconYuv {
            y: vec![0u16; 64 * 64],
            u: vec![128u16; 32 * 32],
            v: vec![128u16; 32 * 32],
        };
        for i in 0..64 {
            for j in 0..64 {
                refr.y[i * 64 + j] = if j == 10 { 255 } else { 0 };
            }
        }
        let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
        let mut ictx = PSearchCtx::new(&refr, &refr, 16, 16, 64, 64, ip, 1, 0, &[]).unwrap();
        let mut outs = Vec::new();
        for f in [EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP] {
            ictx.predict_leaf(
                2,
                2,
                BLOCK_8X8,
                [1, -1],
                MODE_NEWMV,
                [[0, 4], [0, 0]],
                f,
                CompoundSel::AVERAGE,
                None,
                MotionSel::Simple,
            )
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

    /// r417 — `predict_leaf` with an [`InterIntraTrial`] realises the
    /// §5.11.33 `IsInterIntra` arm in the search scratch: the
    /// §7.11.5-translated intra half (predicted from the committed
    /// recon's neighbours through the decode walker's §7.11.2 core)
    /// blends with the inter prediction through the decoder's own
    /// §7.11.3.14 driver, on luma AND chroma, and the §5.11.5 grid
    /// stamps mirror the decode walker's §5.11.28 imperative
    /// overrides (`RefFrame[ 1 ] = INTRA_FRAME`, the bit-silent
    /// §5.11.29 `compound_type` derivation, the inter-intra side-data
    /// grids).
    #[test]
    fn r417_predict_leaf_inter_intra_blends_intra_half() {
        // Reference: flat luma 100, flat chroma 128 ⇒ the zero-MV
        // inter half is flat.
        let refr = GopFrameReconYuv {
            y: vec![100u16; 64 * 64],
            u: vec![128u16; 32 * 32],
            v: vec![128u16; 32 * 32],
        };
        // Committed recon (the §7.11.2 neighbour source): flat luma
        // 200, flat chroma 60 ⇒ the II_V_PRED intra half is flat 200
        // (luma) / 60 (chroma) over the block.
        let mut recon = fresh_recon(64, 64, 60);
        recon.y.fill(200);
        recon.u.fill(60);
        recon.v.fill(60);
        let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
        let mut ictx = PSearchCtx::new(&refr, &refr, 16, 16, 64, 64, ip, 1, 60, &[]).unwrap();

        // Smooth (non-wedge) inter-intra trial at mi (2, 2), 8x8.
        ictx.predict_leaf(
            2,
            2,
            BLOCK_8X8,
            [1, -1],
            MODE_NEWMV,
            [[0, 0], [0, 0]],
            EIGHTTAP,
            CompoundSel::AVERAGE,
            Some(InterIntraTrial {
                mode: crate::cdf::II_V_PRED,
                wedge: None,
                neigh: &recon,
            }),
            MotionSel::Simple,
        )
        .unwrap();
        let mut smooth_luma = Vec::new();
        for i in 8..16usize {
            for j in 8..16usize {
                let v = ictx.scratch[0][i * 64 + j];
                assert!(
                    (100..=200).contains(&v),
                    "blend out of the [inter, intra] envelope at ({i}, {j}): {v}"
                );
                smooth_luma.push(v);
            }
        }
        assert!(
            smooth_luma.iter().any(|&v| v > 100),
            "intra half must contribute to the luma blend"
        );
        assert_ne!(
            &smooth_luma[0..8],
            &smooth_luma[56..64],
            "the §7.11.3.14 smooth II mask varies down the block"
        );
        // Chroma blends too (inter 128 vs intra 60).
        let cu = ictx.scratch[1][4 * 32 + 4];
        assert!(
            (60..128).contains(&cu),
            "chroma blend must pull toward the intra half: {cu}"
        );
        // §5.11.5 grid stamps at the origin cell.
        let cell = 2 * 16 + 2;
        assert_eq!(
            ictx.ref_frames[cell * 2 + 1],
            0,
            "RefFrame[1] = INTRA_FRAME"
        );
        assert_eq!(ictx.interintra_modes[cell], crate::cdf::II_V_PRED);
        assert_eq!(ictx.wedge_interintras[cell], 0);
        assert_eq!(
            ictx.compound_types[cell],
            crate::inter_pred::COMPOUND_INTRA,
            "§5.11.29 bit-silent derivation on the non-wedge arm"
        );

        // Wedge sub-arm: a different §7.11.3.11 mask ⇒ different blend.
        ictx.predict_leaf(
            2,
            2,
            BLOCK_8X8,
            [1, -1],
            MODE_NEWMV,
            [[0, 0], [0, 0]],
            EIGHTTAP,
            CompoundSel::AVERAGE,
            Some(InterIntraTrial {
                mode: crate::cdf::II_V_PRED,
                wedge: Some(7),
                neigh: &recon,
            }),
            MotionSel::Simple,
        )
        .unwrap();
        let mut wedge_luma = Vec::new();
        for i in 8..16usize {
            for j in 8..16usize {
                wedge_luma.push(ictx.scratch[0][i * 64 + j]);
            }
        }
        assert_ne!(smooth_luma, wedge_luma, "wedge mask differs from smooth");
        assert_eq!(ictx.wedge_interintras[cell], 1);
        assert_eq!(ictx.interintra_wedge_indices[cell], 7);
        assert_eq!(ictx.compound_types[cell], crate::inter_pred::COMPOUND_WEDGE);

        // A subsequent plain trial restores the non-inter-intra stamps.
        ictx.predict_leaf(
            2,
            2,
            BLOCK_8X8,
            [1, -1],
            MODE_NEWMV,
            [[0, 0], [0, 0]],
            EIGHTTAP,
            CompoundSel::AVERAGE,
            None,
            MotionSel::Simple,
        )
        .unwrap();
        assert_eq!(ictx.ref_frames[cell * 2 + 1], -1);
        assert_eq!(ictx.wedge_interintras[cell], 0);
        assert_eq!(ictx.interintra_modes[cell], 0);
        for i in 8..16usize {
            for j in 8..16usize {
                assert_eq!(
                    ictx.scratch[0][i * 64 + j],
                    100,
                    "plain zero-MV inter pred is the flat reference"
                );
            }
        }
    }

    /// r412 — the driver mirror must agree with the WRITE mirror leaf
    /// for leaf: a full multi-superblock lossy GOP encode (which
    /// re-runs §7.10.2 per leaf inside `write_partition_tree_syntax`
    /// and hard-errors on any committed non-NEWMV MV that differs
    /// from the write-side derivation) must succeed and self-decode
    /// byte-exact through the spec driver.
    #[test]
    fn r412_gop_with_stack_modes_round_trips_through_spec_driver() {
        let frames: Vec<YuvFrame> = (0..4)
            .map(|k| moving_gradient(96, 80, 2 * k, 4 * k, 31))
            .collect();
        let enc = encode_gop_yuv_with_q(&frames, 80).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), frames.len());
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(f.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(f.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
    }

    /// r415 — a B-frame search context: `refs[ 0 ]` = the forward
    /// anchor (LAST, hint 0), `refs[ 1 ]` = the backward reference
    /// (BWDREF/ALTREF2/ALTREF, hint 2), current hint 1 — §7.8 sign
    /// bias 1 on the backward ordinals.
    fn b_frame_ctx<'a>(
        anchor: &'a GopFrameReconYuv,
        future: &'a GopFrameReconYuv,
        mi_rows: u32,
        mi_cols: u32,
        base_q_idx: u8,
        skip_mode: bool,
    ) -> PSearchCtx {
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.order_hints = crate::cdf::FrameInterOrderHints {
            order_hint_bits: 7,
            current_order_hint: 1,
            order_hints_by_ref: [0, 0, 0, 0, 0, 2, 2, 2],
        };
        for r in 5..=7usize {
            ip.ref_frame_sign_bias[r] = 1;
        }
        ip.skip_mode_present = skip_mode;
        ip.skip_mode_frame = [1, 5];
        let mut slot_to_plane = [0usize; 8];
        slot_to_plane[1] = 1;
        let mut ref_frame_idx = [0u8; 7];
        ref_frame_idx[4] = 1;
        ref_frame_idx[5] = 1;
        ref_frame_idx[6] = 1;
        PSearchCtx::with_refs(
            &[anchor, future],
            slot_to_plane,
            ref_frame_idx,
            vec![1, 5],
            vec![[1, 5]],
            mi_rows,
            mi_cols,
            (mi_cols * 4) as usize,
            (mi_rows * 4) as usize,
            ip,
            base_q_idx,
            &[],
            8,
            1,
            1,
            3,
        )
        .unwrap()
    }

    fn fresh_recon(w: usize, h: usize, q: u8) -> ReconState {
        ReconState {
            y: vec![0u16; w * h],
            u: vec![0u16; (w / 2) * (h / 2)],
            v: vec![0u16; (w / 2) * (h / 2)],
            width: w,
            height: h,
            chroma_w: w / 2,
            chroma_h: h / 2,
            mi_rows: (h / 4) as u32,
            mi_cols: (w / 4) as u32,
            lossless: q == 0,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(q, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        }
    }

    /// r415 — bidirectional COMPOUND_AVERAGE must actually be selected
    /// where the forward/backward mean is the distortion winner: the
    /// target frame is the PIXELWISE AVERAGE of the anchor and the
    /// future reference, so a { LAST, BWDREF } compound leaf predicts
    /// it almost exactly while either single side misses by half the
    /// delta.
    #[test]
    fn r415_b_search_selects_bidir_compound_on_blended_content() {
        let f0 = moving_gradient(64, 64, 0, 0, 40);
        let f2 = moving_gradient(64, 64, 0, 0, 140);
        let base_q_idx = 60u8;
        let k0 = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f0, base_q_idx).unwrap();
        let k2 = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f2, base_q_idx).unwrap();
        let anchor = GopFrameReconYuv {
            y: k0.recon_y,
            u: k0.recon_u,
            v: k0.recon_v,
        };
        let future = GopFrameReconYuv {
            y: k2.recon_y,
            u: k2.recon_u,
            v: k2.recon_v,
        };
        let mut target = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        let avg = |a: &[u16], b: &[u16], out: &mut [u16]| {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = (u32::from(x) + u32::from(y)).div_ceil(2) as u16;
            }
        };
        avg(&anchor.y, &future.y, &mut target.y);
        avg(&anchor.u, &future.u, &mut target.u);
        avg(&anchor.v, &future.v, &mut target.v);

        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ictx = b_frame_ctx(&anchor, &future, 16, 16, base_q_idx, false);
        recon.bd.clear_for_sb(0, 0, 16, 16);
        let tree = build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &target, &mut recon, &mut ictx)
            .unwrap();
        fn count(node: &SyntaxNode, compound: &mut u32, single: &mut u32) {
            let leafc = |b: &SyntaxBlock, compound: &mut u32, single: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.ref_frame == [1, 5] {
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
                        count(ch, compound, single);
                    }
                }
                other => {
                    for b in other.asymmetric_blocks().iter() {
                        leafc(b, compound, single);
                    }
                }
            }
        }
        let (mut compound, mut single) = (0u32, 0u32);
        count(&tree, &mut compound, &mut single);
        assert!(
            compound > 0,
            "average-blend content must select {{ LAST, BWDREF }} bidirectional \
             compound leaves (got COMPOUND={compound} SINGLE={single})"
        );
    }

    /// r415 — the backward single reference must actually be selected
    /// where the future frame matches: the target IS the future
    /// reference's content, so BWDREF single-ref leaves dominate.
    #[test]
    fn r415_b_search_selects_backward_reference_on_future_matching_content() {
        let f0 = moving_gradient(64, 64, 0, 0, 5);
        let f2 = moving_gradient(64, 64, 9, 13, 505);
        let base_q_idx = 60u8;
        let k0 = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f0, base_q_idx).unwrap();
        let k2 = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f2, base_q_idx).unwrap();
        let anchor = GopFrameReconYuv {
            y: k0.recon_y,
            u: k0.recon_u,
            v: k0.recon_v,
        };
        let future = GopFrameReconYuv {
            y: k2.recon_y,
            u: k2.recon_u,
            v: k2.recon_v,
        };
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ictx = b_frame_ctx(&anchor, &future, 16, 16, base_q_idx, false);
        recon.bd.clear_for_sb(0, 0, 16, 16);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f2, &mut recon, &mut ictx).unwrap();
        fn count(node: &SyntaxNode, bwd: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, bwd: &mut u32, other: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.ref_frame == [5, -1] {
                        *bwd += 1;
                    } else {
                        *other += 1;
                    }
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, bwd, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count(ch, bwd, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, bwd, other);
                    }
                }
            }
        }
        let (mut bwd, mut other) = (0u32, 0u32);
        count(&tree, &mut bwd, &mut other);
        assert!(
            bwd > 0,
            "future-matching content must select BWDREF single-reference \
             leaves (got BWD={bwd} OTHER={other})"
        );
    }

    /// r415 — §5.11.29 masked compound must actually be selected
    /// where a wedge blend is the distortion winner: each 32x32
    /// quadrant of the target frame is constructed as the decoder's
    /// OWN §7.11.3.11 COMPOUND_WEDGE blend of the two references
    /// (through [`PSearchCtx::predict_leaf`], the same kernel the
    /// search scores with), so a WEDGE leaf predicts it exactly while
    /// COMPOUND_AVERAGE misses along the wedge boundary — the search
    /// tree must commit COMPOUND_WEDGE leaves, and a full GOP over
    /// the same construction must round-trip byte-exact through the
    /// spec driver (proving the coded §5.11.29 cascade end to end).
    #[test]
    fn r415_search_selects_wedge_compound_on_wedge_blend_content() {
        use crate::cdf::BLOCK_32X32;
        use crate::inter_pred::COMPOUND_WEDGE;
        let f0 = moving_gradient(64, 64, 0, 0, 40);
        let f1 = moving_gradient(64, 64, 0, 0, 140);
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();

        // Target: per-quadrant WEDGE blend (index 7, sign 0) of the
        // two reference reconstructions, through the real kernel.
        let mut target = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        {
            let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
            let mut probe =
                PSearchCtx::new(&pre.recon[1], &pre.recon[0], 16, 16, 64, 64, ip, 2, 0, &[])
                    .unwrap();
            let sel = CompoundSel {
                ctype: COMPOUND_WEDGE,
                wedge_index: 7,
                wedge_sign: 0,
                mask_type: 0,
            };
            for (r, c) in [(0u32, 0u32), (0, 8), (8, 0), (8, 8)] {
                probe
                    .predict_leaf(
                        r,
                        c,
                        BLOCK_32X32,
                        [1, 4],
                        MODE_NEW_NEWMV,
                        [[0, 0], [0, 0]],
                        EIGHTTAP,
                        sel,
                        None,
                        MotionSel::Simple,
                    )
                    .unwrap();
            }
            for (dst, &src) in target.y.iter_mut().zip(probe.scratch[0].iter()) {
                *dst = src;
            }
            for (dst, &src) in target.u.iter_mut().zip(probe.scratch[1].iter()) {
                *dst = src;
            }
            for (dst, &src) in target.v.iter_mut().zip(probe.scratch[2].iter()) {
                *dst = src;
            }
        }

        // Direct search drive (LAST = f1 recon, GOLDEN = f0 recon at
        // p_index = 2), masked compound enabled.
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 2, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.enable_masked_compound = true;
        ip.order_hints = gop_order_hints(2, pre.seq.order_hint_bits);
        let mut ictx = PSearchCtx::new(
            &pre.recon[1],
            &pre.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            2,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &target, &mut recon, &mut ictx)
            .unwrap();
        fn count(node: &SyntaxNode, wedge: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, wedge: &mut u32, other: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.compound_type == crate::inter_pred::COMPOUND_WEDGE {
                        *wedge += 1;
                    } else {
                        *other += 1;
                    }
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, wedge, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count(ch, wedge, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, wedge, other);
                    }
                }
            }
        }
        let (mut wedge, mut other) = (0u32, 0u32);
        count(&tree, &mut wedge, &mut other);
        assert!(
            wedge > 0,
            "wedge-blend content must commit COMPOUND_WEDGE leaves \
             (got WEDGE={wedge} OTHER={other})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, f1, target], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
        // Env-gated fixture dump for external black-box validation /
        // corpus pinning (no-op in normal runs).
        if let Ok(dir) = std::env::var("OXIDEAV_AV1_WEDGE_FIXDIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                format!("{dir}/self-gop-64x64-q60-wedge.ivf"),
                &enc.ivf_bytes,
            )
            .unwrap();
        }
    }

    /// r418 — the §5.11.46 palette arm must reach INTRA leaves inside
    /// INTER frames through the shared leaf encoder: a P-frame whose
    /// centre region is freshly-pasted 4-colour dither (inter
    /// prediction has nothing to track, every §7.11.2 mode misses the
    /// per-pixel pattern) commits `PaletteSizeY > 0` intra leaves in
    /// the P search tree, and the GOP round-trips byte-exact through
    /// the spec driver.
    #[test]
    fn r418_p_search_selects_palette_on_pasted_screen_content() {
        const COLORS: [u16; 4] = [16, 80, 160, 240];
        let f0 = moving_gradient(64, 64, 0, 0, 40);
        let mut f1 = moving_gradient(64, 64, 0, 0, 40);
        // Paste a 32x32 dither block (chroma pair checker) at (16, 16).
        let mut state = 0x0DDB_A115u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for i in 16..48usize {
            for j in 16..48usize {
                f1.y[i * 64 + j] = COLORS[(next() & 3) as usize];
            }
        }
        for i in 8..24usize {
            for j in 8..24usize {
                let par = ((i / 2) + (j / 2)) & 1;
                f1.u[i * 32 + j] = if par == 0 { 96 } else { 168 };
                f1.v[i * 32 + j] = if ((i / 4) & 1) == par { 108 } else { 152 };
            }
        }
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();

        // Direct search drive over the same driver state (LAST = f0
        // recon at p_index = 1).
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.order_hints = gop_order_hints(1, pre.seq.order_hint_bits);
        let mut ictx = PSearchCtx::new(
            &pre.recon[0],
            &pre.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f1, &mut recon, &mut ictx).unwrap();
        fn count(node: &SyntaxNode, pal: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, pal: &mut u32, other: &mut u32| {
                if b.inter.is_none() && b.palette.size_y > 0 {
                    *pal += 1;
                } else {
                    *other += 1;
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, pal, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count(ch, pal, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, pal, other);
                    }
                }
            }
        }
        let (mut pal, mut other) = (0u32, 0u32);
        count(&tree, &mut pal, &mut other);
        assert!(
            pal > 0,
            "pasted dither content must commit PaletteSizeY > 0 intra leaves in the \
             P-frame search (got PALETTE={pal} OTHER={other})"
        );

        // Full-stream conformance through the spec driver.
        let decoded = crate::decoder::decode_av1_spec(&pre.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&pre.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&pre.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&pre.recon[idx].v), "frame {idx} V");
        }
    }

    /// r417 — the §5.11.28 WEDGE sub-arm must actually be selected
    /// where a §7.11.3.11 wedge-masked inter-intra blend is the
    /// distortion winner: the target quadrants are constructed
    /// through the real driver with `wedge_interintra = 1` (wedge
    /// index 7) — the same incremental neighbour-state technique as
    /// the smooth witness — so a wedge inter-intra leaf predicts
    /// each quadrant near-exactly while the smooth mask and every
    /// plain arm miss the wedge boundary; the committed tree must
    /// carry `wedge_interintra = 1` leaves and the GOP must
    /// round-trip byte-exact through the spec driver.
    #[test]
    fn r417_search_selects_wedge_inter_intra_on_wedge_blend_content() {
        use crate::cdf::{BLOCK_32X32, II_V_PRED};
        let f0 = moving_gradient(64, 64, 0, 0, 40);
        // r425 — brightness 220 (was 140): the intra half of the
        // first quadrant's blend derives from border-constant
        // neighbours (~128), so the wedge contrast |inter − intra|
        // must dwarf the coding noise of the anchors or the election
        // turns rate-marginal (the r425 KEY rect arms shifted the
        // anchor reconstruction by a hair and flipped it at 140).
        let f1 = moving_gradient(64, 64, 0, 0, 220);
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();

        let mut target = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        {
            let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
            let mut probe =
                PSearchCtx::new(&pre.recon[1], &pre.recon[0], 16, 16, 64, 64, ip, 2, 0, &[])
                    .unwrap();
            let mut running = fresh_recon(64, 64, base_q_idx);
            for (r, c) in [(0u32, 0u32), (0, 8), (8, 0), (8, 8)] {
                probe
                    .predict_leaf(
                        r,
                        c,
                        BLOCK_32X32,
                        [1, -1],
                        MODE_NEARESTMV,
                        [[0, 0], [0, 0]],
                        EIGHTTAP,
                        CompoundSel::AVERAGE,
                        Some(InterIntraTrial {
                            mode: II_V_PRED,
                            wedge: Some(7),
                            neigh: &running,
                        }),
                        MotionSel::Simple,
                    )
                    .unwrap();
                let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
                for i in 0..32usize {
                    for j in 0..32usize {
                        let v = probe.scratch[0][(row0 + i) * 64 + col0 + j];
                        target.y[(row0 + i) * 64 + col0 + j] = v;
                        running.y[(row0 + i) * 64 + col0 + j] = v;
                    }
                }
                let (cr0, cc0) = (row0 / 2, col0 / 2);
                for i in 0..16usize {
                    for j in 0..16usize {
                        let u = probe.scratch[1][(cr0 + i) * 32 + cc0 + j];
                        let v = probe.scratch[2][(cr0 + i) * 32 + cc0 + j];
                        target.u[(cr0 + i) * 32 + cc0 + j] = u;
                        target.v[(cr0 + i) * 32 + cc0 + j] = v;
                        running.u[(cr0 + i) * 32 + cc0 + j] = u;
                        running.v[(cr0 + i) * 32 + cc0 + j] = v;
                    }
                }
            }
        }

        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 2, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.enable_interintra_compound = true;
        ip.order_hints = gop_order_hints(2, pre.seq.order_hint_bits);
        let mut ictx = PSearchCtx::new(
            &pre.recon[1],
            &pre.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            2,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &target, &mut recon, &mut ictx)
            .unwrap();
        fn count(node: &SyntaxNode, wedge_ii: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, wedge_ii: &mut u32, other: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.interintra_mode.is_some() && ib.wedge_interintra == 1 {
                        *wedge_ii += 1;
                    } else {
                        *other += 1;
                    }
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, wedge_ii, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count(ch, wedge_ii, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, wedge_ii, other);
                    }
                }
            }
        }
        let (mut wedge_ii, mut other) = (0u32, 0u32);
        count(&tree, &mut wedge_ii, &mut other);
        assert!(
            wedge_ii > 0,
            "wedge inter-intra blend content must commit wedge_interintra = 1 leaves \
             (got WEDGE_II={wedge_ii} OTHER={other})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, f1, target], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
    }

    /// r417 — sub-8×8 INTRA leaves in inter frames: content whose
    /// 8×8 groups mix a V_PRED-perfect 4×4 cell (its rows replicate
    /// the row above — the §7.11.2 intra copy is near-exact while no
    /// reference contains the vertically-constant patch) with three
    /// trackable moving-texture cells must commit a PARTITION_SPLIT
    /// at BLOCK_8X8 whose leaves MIX `is_inter = 0` and
    /// `is_inter = 1` — the §5.11.33 `someUseIntra` chroma arm fires
    /// on the group's inter `HasChroma` leaf — and the full GOP must
    /// round-trip byte-exact through the spec driver.
    #[test]
    fn r417_search_selects_sub8_intra_in_mixed_groups() {
        let f0 = moving_gradient(64, 64, 0, 0, 71);
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(std::slice::from_ref(&f0), base_q_idx).unwrap();
        let kr = &pre.recon[0];

        // Target: the r416 fine-motion checkerboard (per-4×4-cell
        // (0,0) / (4,4) sampling shifts of the KEY recon — proven to
        // drive PARTITION_SPLIT down to BLOCK_4X4), with the NW cell
        // of four 8×8 groups overwritten by V-replication (rows copy
        // the row directly above — vertically constant, absent from
        // the reference, near-exact under §7.11.2 V_PRED).
        let mut f2 = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        let at = |p: &[u16], w: usize, h: usize, i: i64, j: i64| -> u16 {
            let ii = i.clamp(0, h as i64 - 1) as usize;
            let jj = j.clamp(0, w as i64 - 1) as usize;
            p[ii * w + jj]
        };
        for i in 0..64i64 {
            for j in 0..64i64 {
                let par = ((i / 4) + (j / 4)) & 1;
                let (dy, dx) = if par == 0 { (0, 0) } else { (4, 4) };
                f2.y[(i * 64 + j) as usize] = at(&kr.y, 64, 64, i + dy, j + dx);
            }
        }
        f2.u.copy_from_slice(&kr.u);
        f2.v.copy_from_slice(&kr.v);
        for (r0, c0) in [(8usize, 8usize), (8, 40), (40, 8), (40, 40)] {
            for i in 0..4 {
                for j in 0..4 {
                    f2.y[(r0 + i) * 64 + c0 + j] = f2.y[(r0 - 1) * 64 + c0 + j];
                }
            }
        }

        // Direct search drive (LAST = GOLDEN = the KEY recon at
        // p_index = 1) over the target.
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.enable_interintra_compound = true;
        ip.order_hints = gop_order_hints(1, pre.seq.order_hint_bits);
        let mut ictx = PSearchCtx::new(
            &pre.recon[0],
            &pre.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f2, &mut recon, &mut ictx).unwrap();

        // Walk the committed tree: collect BLOCK_4X4 leaves with
        // §5.11.4 SPLIT geometry.
        fn walk(
            node: &SyntaxNode,
            r: u32,
            c: u32,
            b_size: usize,
            found: &mut Vec<(u32, u32, bool)>,
        ) {
            match node {
                SyntaxNode::Leaf(b) if b_size == BLOCK_4X4 => {
                    found.push((r, c, b.inter.is_some()));
                }
                SyntaxNode::Split(children) => {
                    let sub =
                        crate::cdf::partition_subsize(crate::cdf::PARTITION_SPLIT, b_size).unwrap();
                    let half = (NUM_4X4_BLOCKS_WIDE[b_size] as u32) >> 1;
                    walk(&children[0], r, c, sub, found);
                    walk(&children[1], r, c + half, sub, found);
                    walk(&children[2], r + half, c, sub, found);
                    walk(&children[3], r + half, c + half, sub, found);
                }
                _ => {}
            }
        }
        let mut sub4: Vec<(u32, u32, bool)> = Vec::new();
        walk(&tree, 0, 0, BLOCK_64X64, &mut sub4);
        let n_intra4 = sub4.iter().filter(|&&(_, _, inter)| !inter).count();
        assert!(
            n_intra4 > 0,
            "V-replicated cells must commit BLOCK_4X4 intra leaves (got {sub4:?})"
        );
        // At least one 2×2 group mixes intra and inter — the
        // §5.11.33 someUseIntra chroma arm is live in the stream.
        let mixed = sub4.iter().any(|&(r, c, inter)| {
            !inter
                && sub4
                    .iter()
                    .any(|&(r2, c2, inter2)| inter2 && (r2 & !1, c2 & !1) == (r & !1, c & !1))
        });
        assert!(mixed, "some 2x2 group must mix intra and inter ({sub4:?})");

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, f2], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
        if let Ok(dir) = std::env::var("OXIDEAV_AV1_SUB8INTRA_FIXDIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                format!("{dir}/self-gop-64x64-q60-sub8-intra.ivf"),
                &enc.ivf_bytes,
            )
            .unwrap();
        }
    }

    /// r417 — §5.11.28 inter-intra must actually be selected where
    /// the §7.11.3.14 blend is the distortion winner: the target
    /// frame is constructed QUADRANT BY QUADRANT as the encoder's
    /// own smooth II_V_PRED inter-intra blend (zero-MV LAST inter
    /// half, the intra half predicted from the running construction
    /// recon — exactly the §5.11.33 ordering the search and the
    /// decoder both realise), so an inter-intra leaf predicts each
    /// quadrant near-exactly while any plain inter leaf misses the
    /// intra-weighted region — the search tree must commit
    /// inter-intra leaves, and a full GOP over the same construction
    /// must round-trip byte-exact through the spec driver (proving
    /// the coded §5.11.28 cascade + §7.11.3.14 blend end to end).
    #[test]
    fn r417_search_selects_inter_intra_on_blended_content() {
        use crate::cdf::{BLOCK_32X32, II_V_PRED};
        let f0 = moving_gradient(64, 64, 0, 0, 40);
        let f1 = moving_gradient(64, 64, 0, 0, 140);
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();

        // Target: per-32x32-quadrant smooth inter-intra blend of the
        // LAST reconstruction and the §7.11.2 intra half, built
        // through the real driver with the SAME incremental
        // neighbour state the sequential search walk will hold.
        let mut target = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        {
            let ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
            let mut probe =
                PSearchCtx::new(&pre.recon[1], &pre.recon[0], 16, 16, 64, 64, ip, 2, 0, &[])
                    .unwrap();
            let mut running = fresh_recon(64, 64, base_q_idx);
            for (r, c) in [(0u32, 0u32), (0, 8), (8, 0), (8, 8)] {
                probe
                    .predict_leaf(
                        r,
                        c,
                        BLOCK_32X32,
                        [1, -1],
                        MODE_NEARESTMV,
                        [[0, 0], [0, 0]],
                        EIGHTTAP,
                        CompoundSel::AVERAGE,
                        Some(InterIntraTrial {
                            mode: II_V_PRED,
                            wedge: None,
                            neigh: &running,
                        }),
                        MotionSel::Simple,
                    )
                    .unwrap();
                // Adopt the blended quadrant into the target AND the
                // running neighbour recon (the next quadrant's intra
                // half reads it, like the decode walk will).
                let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
                for i in 0..32usize {
                    for j in 0..32usize {
                        let v = probe.scratch[0][(row0 + i) * 64 + col0 + j];
                        target.y[(row0 + i) * 64 + col0 + j] = v;
                        running.y[(row0 + i) * 64 + col0 + j] = v;
                    }
                }
                let (cr0, cc0) = (row0 / 2, col0 / 2);
                for i in 0..16usize {
                    for j in 0..16usize {
                        let u = probe.scratch[1][(cr0 + i) * 32 + cc0 + j];
                        let v = probe.scratch[2][(cr0 + i) * 32 + cc0 + j];
                        target.u[(cr0 + i) * 32 + cc0 + j] = u;
                        target.v[(cr0 + i) * 32 + cc0 + j] = v;
                        running.u[(cr0 + i) * 32 + cc0 + j] = u;
                        running.v[(cr0 + i) * 32 + cc0 + j] = v;
                    }
                }
            }
        }

        // Direct search drive (LAST = f1 recon, GOLDEN = f0 recon at
        // p_index = 2), inter-intra compound enabled.
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 2, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.enable_interintra_compound = true;
        ip.order_hints = gop_order_hints(2, pre.seq.order_hint_bits);
        let mut ictx = PSearchCtx::new(
            &pre.recon[1],
            &pre.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            2,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &target, &mut recon, &mut ictx)
            .unwrap();
        fn count(node: &SyntaxNode, ii: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, ii: &mut u32, other: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.interintra_mode.is_some() {
                        *ii += 1;
                    } else {
                        *other += 1;
                    }
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, ii, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count(ch, ii, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, ii, other);
                    }
                }
            }
        }
        let (mut ii, mut other) = (0u32, 0u32);
        count(&tree, &mut ii, &mut other);
        assert!(
            ii > 0,
            "inter-intra blend content must commit §5.11.28 leaves \
             (got II={ii} OTHER={other})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, f1, target], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
        // Env-gated fixture dump for external black-box validation /
        // corpus pinning (no-op in normal runs).
        if let Ok(dir) = std::env::var("OXIDEAV_AV1_INTERINTRA_FIXDIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                format!("{dir}/self-gop-64x64-q60-interintra.ivf"),
                &enc.ivf_bytes,
            )
            .unwrap();
        }
    }

    /// r416 — §5.11.29 jnt-comp must actually be selected where the
    /// §7.11.3.15 distance-weighted blend is the distortion winner:
    /// the target frame is constructed as the decoder's OWN
    /// COMPOUND_DISTANCE blend of the two references (through
    /// [`PSearchCtx::predict_leaf`] under the real P-frame order
    /// hints — LAST at distance 1, GOLDEN at distance 2 ⇒ the
    /// asymmetric `(11, 5)/16` weight pair), so a DISTANCE leaf
    /// predicts it exactly while COMPOUND_AVERAGE misses by the
    /// weight asymmetry — the search tree must commit
    /// COMPOUND_DISTANCE leaves, and a full GOP over the same
    /// construction must round-trip byte-exact through the spec
    /// driver (proving the coded `compound_idx` cascade end to end).
    #[test]
    fn r416_search_selects_distance_compound_on_weighted_blend_content() {
        use crate::cdf::BLOCK_32X32;
        use crate::inter_pred::COMPOUND_DISTANCE;
        let f0 = moving_gradient(64, 64, 0, 0, 40);
        let f1 = moving_gradient(64, 64, 0, 0, 140);
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();

        // Target: the §7.11.3.15 DISTANCE blend of the two reference
        // reconstructions through the real kernel under the frame-2
        // order hints (the same hints the GOP encode derives).
        let mut target = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        {
            let mut ip = SyntaxInterFrameParams::single_ref_baseline(16, 16, false);
            ip.order_hints = gop_order_hints(2, pre.seq.order_hint_bits);
            let mut probe =
                PSearchCtx::new(&pre.recon[1], &pre.recon[0], 16, 16, 64, 64, ip, 2, 0, &[])
                    .unwrap();
            let sel = CompoundSel {
                ctype: COMPOUND_DISTANCE,
                wedge_index: 0,
                wedge_sign: 0,
                mask_type: 0,
            };
            for (r, c) in [(0u32, 0u32), (0, 8), (8, 0), (8, 8)] {
                probe
                    .predict_leaf(
                        r,
                        c,
                        BLOCK_32X32,
                        [1, 4],
                        MODE_NEW_NEWMV,
                        [[0, 0], [0, 0]],
                        EIGHTTAP,
                        sel,
                        None,
                        MotionSel::Simple,
                    )
                    .unwrap();
            }
            for (dst, &src) in target.y.iter_mut().zip(probe.scratch[0].iter()) {
                *dst = src;
            }
            for (dst, &src) in target.u.iter_mut().zip(probe.scratch[1].iter()) {
                *dst = src;
            }
            for (dst, &src) in target.v.iter_mut().zip(probe.scratch[2].iter()) {
                *dst = src;
            }
        }

        // Direct search drive (LAST = f1 recon, GOLDEN = f0 recon at
        // p_index = 2), jnt-comp enabled.
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 2, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.enable_masked_compound = true;
        ip.enable_jnt_comp = true;
        ip.order_hints = gop_order_hints(2, pre.seq.order_hint_bits);
        let mut ictx = PSearchCtx::new(
            &pre.recon[1],
            &pre.recon[0],
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            2,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &target, &mut recon, &mut ictx)
            .unwrap();
        fn count(node: &SyntaxNode, dist: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, dist: &mut u32, other: &mut u32| {
                if let Some(ib) = b.inter.as_ref() {
                    if ib.compound_type == crate::inter_pred::COMPOUND_DISTANCE {
                        *dist += 1;
                    } else {
                        *other += 1;
                    }
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, dist, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count(ch, dist, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, dist, other);
                    }
                }
            }
        }
        let (mut dist, mut other) = (0u32, 0u32);
        count(&tree, &mut dist, &mut other);
        assert!(
            dist > 0,
            "distance-blend content must commit COMPOUND_DISTANCE leaves \
             (got DISTANCE={dist} OTHER={other})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, f1, target], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
        // Env-gated fixture dump for external black-box validation /
        // corpus pinning (no-op in normal runs).
        if let Ok(dir) = std::env::var("OXIDEAV_AV1_JNT_FIXDIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(format!("{dir}/self-gop-64x64-q60-jnt.ivf"), &enc.ivf_bytes).unwrap();
        }
    }

    /// r416 — walk a committed search tree tracking each leaf's block
    /// size: count sub-8×8 leaves (either axis < 8 samples), bare
    /// BLOCK_4X4 leaves, and four-strip (HORZ_4 / VERT_4) nodes.
    fn count_sub8(node: &SyntaxNode, b_size: usize, n_sub8: &mut u32, n_4x4: &mut u32) {
        use crate::cdf::{
            block_height, block_width, partition_subsize, BLOCK_4X4, PARTITION_HORZ,
            PARTITION_HORZ_4, PARTITION_HORZ_A, PARTITION_HORZ_B, PARTITION_SPLIT, PARTITION_VERT,
            PARTITION_VERT_4, PARTITION_VERT_A, PARTITION_VERT_B,
        };
        let leaf_hit = |sz: usize, n_sub8: &mut u32, n_4x4: &mut u32| {
            if block_width(sz) < 8 || block_height(sz) < 8 {
                *n_sub8 += 1;
            }
            if sz == BLOCK_4X4 {
                *n_4x4 += 1;
            }
        };
        match node {
            SyntaxNode::Leaf(_) => leaf_hit(b_size, n_sub8, n_4x4),
            SyntaxNode::Split(children) => {
                let s = partition_subsize(PARTITION_SPLIT, b_size).unwrap();
                for ch in children.iter() {
                    count_sub8(ch, s, n_sub8, n_4x4);
                }
            }
            SyntaxNode::Horz(_) | SyntaxNode::Vert(_) => {
                let part = if matches!(node, SyntaxNode::Horz(_)) {
                    PARTITION_HORZ
                } else {
                    PARTITION_VERT
                };
                let s = partition_subsize(part, b_size).unwrap();
                for _b in node.asymmetric_blocks() {
                    leaf_hit(s, n_sub8, n_4x4);
                }
            }
            SyntaxNode::HorzA(_)
            | SyntaxNode::HorzB(_)
            | SyntaxNode::VertA(_)
            | SyntaxNode::VertB(_) => {
                // T-shapes: two splitSize quarters + one subSize half.
                let (part, half_first) = match node {
                    SyntaxNode::HorzA(_) => (PARTITION_HORZ_A, false),
                    SyntaxNode::HorzB(_) => (PARTITION_HORZ_B, true),
                    SyntaxNode::VertA(_) => (PARTITION_VERT_A, false),
                    _ => (PARTITION_VERT_B, true),
                };
                let half = partition_subsize(part, b_size).unwrap();
                let quarter = partition_subsize(PARTITION_SPLIT, b_size).unwrap();
                let sizes: [usize; 3] = if half_first {
                    [half, quarter, quarter]
                } else {
                    [quarter, quarter, half]
                };
                for (i, _b) in node.asymmetric_blocks().iter().enumerate() {
                    leaf_hit(sizes[i], n_sub8, n_4x4);
                }
            }
            SyntaxNode::Horz4(_) | SyntaxNode::Vert4(_) => {
                let part = if matches!(node, SyntaxNode::Horz4(_)) {
                    PARTITION_HORZ_4
                } else {
                    PARTITION_VERT_4
                };
                let s = partition_subsize(part, b_size).unwrap();
                for _b in node.asymmetric_blocks() {
                    leaf_hit(s, n_sub8, n_4x4);
                }
            }
        }
    }

    /// r416 — sub-8×8 inter leaves must actually be selected where
    /// motion has 4×4 granularity: the target frame samples the KEY
    /// reconstruction with a per-4×4-cell alternating whole-sample
    /// shift, so a BLOCK_8X8 leaf's single MV misses half its cells
    /// while a PARTITION_SPLIT into four BLOCK_4X4 NEWMV leaves
    /// predicts every cell exactly — the search tree must commit
    /// BLOCK_4X4 leaves, and a full GOP over the same construction
    /// must round-trip byte-exact through the spec driver (proving
    /// the §5.11.34 `HasChroma` group-chroma coding end to end).
    #[test]
    fn r416_search_selects_sub8_split_on_fine_motion() {
        let f0 = moving_gradient(64, 64, 0, 0, 71);
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(std::slice::from_ref(&f0), base_q_idx).unwrap();
        let kr = &pre.recon[0];

        // Target: cell (ci, cj) reads the KEY recon shifted by
        // (0, 0) / (4, 4) on the cell-parity checkerboard.
        let mut target = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        let at = |p: &[u16], w: usize, h: usize, i: i64, j: i64| -> u16 {
            let ii = i.clamp(0, h as i64 - 1) as usize;
            let jj = j.clamp(0, w as i64 - 1) as usize;
            p[ii * w + jj]
        };
        for i in 0..64i64 {
            for j in 0..64i64 {
                let par = ((i / 4) + (j / 4)) & 1;
                let (dy, dx) = if par == 0 { (0, 0) } else { (4, 4) };
                target.y[(i * 64 + j) as usize] = at(&kr.y, 64, 64, i + dy, j + dx);
            }
        }
        // Chroma: zero-motion copy (luma alone drives the witness;
        // the group-chroma residual absorbs the tiled prediction).
        target.u.copy_from_slice(&kr.u);
        target.v.copy_from_slice(&kr.v);

        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.enable_masked_compound = true;
        ip.enable_jnt_comp = true;
        ip.order_hints = gop_order_hints(1, pre.seq.order_hint_bits);
        let key_recon = GopFrameReconYuv {
            y: kr.y.clone(),
            u: kr.u.clone(),
            v: kr.v.clone(),
        };
        let mut ictx = PSearchCtx::new(
            &key_recon,
            &key_recon,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &target, &mut recon, &mut ictx)
            .unwrap();
        let (mut n_sub8, mut n_4x4) = (0u32, 0u32);
        count_sub8(&tree, BLOCK_64X64, &mut n_sub8, &mut n_4x4);
        assert!(
            n_4x4 > 0,
            "4×4-granular motion must commit BLOCK_4X4 leaves (sub8={n_sub8} 4x4={n_4x4})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, target], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
        // Env-gated fixture dump for external black-box validation /
        // corpus pinning (no-op in normal runs).
        if let Ok(dir) = std::env::var("OXIDEAV_AV1_SUB8_FIXDIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                format!("{dir}/self-gop-64x64-q60-sub8-split.ivf"),
                &enc.ivf_bytes,
            )
            .unwrap();
        }
    }

    /// r416 — 4-sample-high band motion must commit sub-8 STRIPS: the
    /// target samples the KEY recon with a per-4-row-band alternating
    /// horizontal shift, so 16×4 (HORZ_4 at BLOCK_16X16) or 8×4
    /// (HORZ at BLOCK_8X8) leaves predict each band exactly while any
    /// 8-sample-high leaf straddles two bands — the tree must commit
    /// height-4 leaves, and the full GOP round-trips byte-exact.
    #[test]
    fn r416_search_selects_sub8_strips_on_band_motion() {
        let f0 = moving_gradient(64, 64, 0, 0, 93);
        let base_q_idx = 60u8;
        let pre = encode_gop_yuv_with_q(std::slice::from_ref(&f0), base_q_idx).unwrap();
        let kr = &pre.recon[0];

        let mut target = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        let at = |p: &[u16], w: usize, h: usize, i: i64, j: i64| -> u16 {
            let ii = i.clamp(0, h as i64 - 1) as usize;
            let jj = j.clamp(0, w as i64 - 1) as usize;
            p[ii * w + jj]
        };
        for i in 0..64i64 {
            for j in 0..64i64 {
                let dx = if (i / 4) & 1 == 0 { 0 } else { 5 };
                target.y[(i * 64 + j) as usize] = at(&kr.y, 64, 64, i, j + dx);
            }
        }
        target.u.copy_from_slice(&kr.u);
        target.v.copy_from_slice(&kr.v);

        let fh = build_p_frame_yuv420_8bit_fh_with_q(&pre.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.reference_select = true;
        ip.enable_masked_compound = true;
        ip.enable_jnt_comp = true;
        ip.order_hints = gop_order_hints(1, pre.seq.order_hint_bits);
        let key_recon = GopFrameReconYuv {
            y: kr.y.clone(),
            u: kr.u.clone(),
            v: kr.v.clone(),
        };
        let mut ictx = PSearchCtx::new(
            &key_recon,
            &key_recon,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree = build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &target, &mut recon, &mut ictx)
            .unwrap();
        let (mut n_sub8, mut n_4x4) = (0u32, 0u32);
        count_sub8(&tree, BLOCK_64X64, &mut n_sub8, &mut n_4x4);
        assert!(
            n_sub8 > 0,
            "band motion must commit sub-8 strip leaves (sub8={n_sub8} 4x4={n_4x4})"
        );
        fn shape_census(node: &SyntaxNode, h4: &mut u32, v4: &mut u32, hz: &mut u32, vt: &mut u32) {
            match node {
                SyntaxNode::Horz4(_) => *h4 += 1,
                SyntaxNode::Vert4(_) => *v4 += 1,
                SyntaxNode::Horz(_) => *hz += 1,
                SyntaxNode::Vert(_) => *vt += 1,
                SyntaxNode::Split(ch) => {
                    for c in ch.iter() {
                        shape_census(c, h4, v4, hz, vt);
                    }
                }
                _ => {}
            }
        }
        let (mut h4, mut v4, mut hz, mut vt) = (0u32, 0u32, 0u32, 0u32);
        shape_census(&tree, &mut h4, &mut v4, &mut hz, &mut vt);
        let _ = (v4, vt);
        assert!(
            h4 > 0 && hz > 0,
            "band motion must commit both HORZ_4 strip nodes and HORZ-at-8x8 halves \
             (horz4={h4} horz={hz} sub8={n_sub8} 4x4={n_4x4})"
        );

        let enc = encode_gop_yuv_with_q(&[f0, target], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, fr) in decoded.iter().enumerate() {
            assert_eq!(fr.planes[0], plane8(&enc.recon[idx].y), "frame {idx} luma");
            assert_eq!(fr.planes[1], plane8(&enc.recon[idx].u), "frame {idx} U");
            assert_eq!(fr.planes[2], plane8(&enc.recon[idx].v), "frame {idx} V");
        }
        if let Ok(dir) = std::env::var("OXIDEAV_AV1_SUB8_FIXDIR") {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                format!("{dir}/self-gop-64x64-q60-sub8-strips.ivf"),
                &enc.ivf_bytes,
            )
            .unwrap();
        }
    }

    /// r415 — §5.9.22 forward/backward skip mode on a B frame:
    /// identical converged references make the one-symbol skip-mode
    /// block (compound NEAREST_NEARESTMV over
    /// SkipModeFrame[] = {{ LAST, BWDREF }}) the RD winner.
    #[test]
    fn r415_b_search_selects_skip_mode_between_converged_references() {
        let f = moving_gradient(64, 64, 2, 3, 21);
        let base_q_idx = 60u8;
        let k = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f, base_q_idx).unwrap();
        let anchor = GopFrameReconYuv {
            y: k.recon_y,
            u: k.recon_u,
            v: k.recon_v,
        };
        let future = anchor.clone();
        let mut recon = fresh_recon(64, 64, base_q_idx);
        let mut ictx = b_frame_ctx(&anchor, &future, 16, 16, base_q_idx, true);
        recon.bd.clear_for_sb(0, 0, 16, 16);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f, &mut recon, &mut ictx).unwrap();
        fn count(node: &SyntaxNode, sm: &mut u32, other: &mut u32) {
            let leafc = |b: &SyntaxBlock, sm: &mut u32, other: &mut u32| {
                if b.inter.as_ref().is_some_and(|ib| ib.skip_mode != 0) {
                    *sm += 1;
                } else {
                    *other += 1;
                }
            };
            match node {
                SyntaxNode::Leaf(b) => leafc(b, sm, other),
                SyntaxNode::Split(children) => {
                    for ch in children.iter() {
                        count(ch, sm, other);
                    }
                }
                rest => {
                    for b in rest.asymmetric_blocks().iter() {
                        leafc(b, sm, other);
                    }
                }
            }
        }
        let (mut sm, mut other) = (0u32, 0u32);
        count(&tree, &mut sm, &mut other);
        assert!(
            sm > 0,
            "converged references on a skip-mode B frame must select \
             skip_mode leaves (got SKIP_MODE={sm} OTHER={other})"
        );
    }

    /// Per-leaf §5.11.27 motion-mode census over a search tree.
    fn count_motion_modes(node: &SyntaxNode, counts: &mut [u32; 3]) {
        let leafc = |b: &SyntaxBlock, counts: &mut [u32; 3]| {
            if let Some(ib) = b.inter.as_ref() {
                counts[(ib.motion_mode as usize).min(2)] += 1;
            }
        };
        match node {
            SyntaxNode::Leaf(b) => leafc(b, counts),
            SyntaxNode::Split(children) => {
                for ch in children.iter() {
                    count_motion_modes(ch, counts);
                }
            }
            rest => {
                for b in rest.asymmetric_blocks().iter() {
                    leafc(b, counts);
                }
            }
        }
    }

    /// Vertically-sheared motion content: `f0` a dense texture, `f1`
    /// the same texture with a per-ROW horizontal shift that ramps
    /// from `dx_top` (rows above `ramp0`) down to 0 (rows below
    /// `ramp1`). No block-constant MV fits a ramp row band, and no
    /// §5.11.4 partition boundary aligns with it — the §7.11.3.9-10
    /// OBMC blend of the above neighbour's larger shift over a
    /// block's top band is the best model the syntax offers.
    fn shear_pair(w: u32, h: u32, dx_top: usize, ramp0: usize, ramp1: usize) -> [YuvFrame; 2] {
        let (wu, hu) = (w as usize, h as usize);
        let tex = |i: usize, j: usize| -> u16 {
            ((i * 7 + j * 11 + (i / 4) * (j / 8) + ((i * j) / 13)) % 256) as u16
        };
        let mut f0 = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        for i in 0..hu {
            for j in 0..wu {
                f0.y[i * wu + j] = tex(i, j);
            }
        }
        let dx_of = |i: usize| -> usize {
            if i < ramp0 {
                dx_top
            } else if i >= ramp1 {
                0
            } else {
                dx_top * (ramp1 - i) / (ramp1 - ramp0)
            }
        };
        let mut f1 = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        for i in 0..hu {
            let dx = dx_of(i);
            for j in 0..wu {
                f1.y[i * wu + j] = tex(i, j + dx);
            }
        }
        [f0, f1]
    }

    /// r419 — the §5.11.27 OBMC selection must actually be REACHED:
    /// on vertically-sheared motion (per-row shift ramp misaligned
    /// with every partition boundary) at least one committed leaf
    /// carries `motion_mode = OBMC`, and the emitted GOP stream
    /// round-trips byte-exact through the spec decoder.
    #[test]
    fn r419_p_search_selects_obmc_on_sheared_motion() {
        let [f0, f1] = shear_pair(64, 64, 8, 24, 48);
        let base_q_idx = 100u8;
        let key = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f0, base_q_idx).unwrap();
        let reference = GopFrameReconYuv {
            y: key.recon_y,
            u: key.recon_u,
            v: key.recon_v,
        };
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 64, 64, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        // r419 — the frame configuration the GOP driver codes.
        ip.is_motion_mode_switchable = true;
        let mut ictx = PSearchCtx::new(
            &reference,
            &reference,
            mi_rows,
            mi_cols,
            64,
            64,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, &f1, &mut recon, &mut ictx).unwrap();
        let mut counts = [0u32; 3];
        count_motion_modes(&tree, &mut counts);
        assert!(
            counts[MOTION_MODE_OBMC as usize] > 0,
            "sheared motion must commit at least one §5.11.27 OBMC leaf \
             (got SIMPLE={} OBMC={} WARP={})",
            counts[0],
            counts[1],
            counts[2]
        );

        // Stream-level: the same content through the public GOP
        // driver decodes byte-exact in the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, f1], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(
                f.planes[0],
                plane8(&enc.recon[idx].y),
                "display {idx}: luma"
            );
            assert_eq!(f.planes[1], plane8(&enc.recon[idx].u), "display {idx}: U");
            assert_eq!(f.planes[2], plane8(&enc.recon[idx].v), "display {idx}: V");
        }
    }

    /// Smoothly zooming texture pair: `f1` samples `f0`'s texture
    /// field scaled about the frame centre by `num/den` (a true
    /// affine motion field — per-block translational MVs vary
    /// linearly across the frame, the §7.10.4 neighbour samples fit
    /// the §7.11.3.8 least squares exactly).
    fn zoom_pair(w: u32, h: u32, num: i64, den: i64) -> [YuvFrame; 2] {
        let (wu, hu) = (w as usize, h as usize);
        // Smooth wide-scale texture (sub-pel interpolable).
        let tex = |y: i64, x: i64| -> u16 {
            let (y, x) = (y as f64 / 8.0, x as f64 / 8.0);
            let v = 96.0
                + 60.0 * ((y * 0.11).sin() * (x * 0.13).cos())
                + 40.0 * ((y * 0.05 + x * 0.07).sin());
            v.clamp(0.0, 255.0) as u16
        };
        let mut f0 = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        let mut f1 = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        let (cy, cx) = ((hu / 2) as i64, (wu / 2) as i64);
        for i in 0..hu {
            for j in 0..wu {
                // f0 at identity, f1 zoomed about the centre in
                // 1/8-pel units.
                f0.y[i * wu + j] = tex((i as i64) * 8, (j as i64) * 8);
                let sy = cy * 8 + ((i as i64) - cy) * 8 * num / den;
                let sx = cx * 8 + ((j as i64) - cx) * 8 * num / den;
                f1.y[i * wu + j] = tex(sy, sx);
            }
        }
        [f0, f1]
    }

    /// r419 — the §5.11.27 WARPED_CAUSAL selection must actually be
    /// REACHED: on zooming content (a true affine motion field, so
    /// the §7.10.4 neighbour MVs fit the §7.11.3.8 least squares)
    /// at least one committed leaf carries
    /// `motion_mode = WARPED_CAUSAL`, and the emitted GOP stream
    /// round-trips byte-exact through the spec decoder.
    #[test]
    fn r419_p_search_selects_warped_causal_on_zooming_motion() {
        let [f0, f1] = zoom_pair(96, 80, 17, 16);
        let base_q_idx = 100u8;
        let key = crate::encoder::key_frame::encode_key_frame_yuv_with_q(&f0, base_q_idx).unwrap();
        assert!(
            key.seq.enable_warped_motion,
            "r419: the sequence gate must be open for §5.11.27 arm B"
        );
        let reference = GopFrameReconYuv {
            y: key.recon_y,
            u: key.recon_u,
            v: key.recon_v,
        };
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, 96, 80, base_q_idx, 1, &[]);
        assert_eq!(fh.allow_warped_motion, Some(true));
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; 96 * 80],
            u: vec![0u16; 48 * 40],
            v: vec![0u16; 48 * 40],
            width: 96,
            height: 80,
            chroma_w: 48,
            chroma_h: 40,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.is_motion_mode_switchable = true;
        ip.allow_warped_motion = true;
        let mut ictx = PSearchCtx::new(
            &reference,
            &reference,
            mi_rows,
            mi_cols,
            96,
            80,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        let mut counts = [0u32; 3];
        // r421 — one persistent tile state across the superblocks (the
        // twin's mirror must carry the earlier superblocks' stamps).
        let mut ts = test_tile_state(&recon, &ictx);
        for (sb_r, sb_c) in [(0u32, 0u32), (0, 16), (16, 0), (16, 16)] {
            if sb_r >= mi_rows || sb_c >= mi_cols {
                continue;
            }
            recon.bd.clear_for_sb(sb_r, sb_c, mi_rows, mi_cols);
            let tree =
                search_and_commit_sb(&mut ts, sb_r, sb_c, &f1, &mut recon, &mut ictx).unwrap();
            count_motion_modes(&tree, &mut counts);
        }
        assert!(
            counts[MOTION_MODE_WARPED_CAUSAL as usize] > 0,
            "zooming motion must commit at least one §5.11.27 \
             WARPED_CAUSAL leaf (got SIMPLE={} OBMC={} WARP={})",
            counts[0],
            counts[1],
            counts[2]
        );

        // Stream-level: the same content through the public GOP
        // driver decodes byte-exact in the spec driver.
        let enc = encode_gop_yuv_with_q(&[f0, f1], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(
                f.planes[0],
                plane8(&enc.recon[idx].y),
                "display {idx}: luma"
            );
            assert_eq!(f.planes[1], plane8(&enc.recon[idx].u), "display {idx}: U");
            assert_eq!(f.planes[2], plane8(&enc.recon[idx].v), "display {idx}: V");
        }
    }

    /// Deterministic noise texture (xorshift over the seed).
    fn noise_frame(w: u32, h: u32, seed: u32) -> YuvFrame {
        let mut state = 0x1357_9BDFu32 ^ seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut f = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 0);
        for p in f.y.iter_mut() {
            *p = (next() & 0xFF) as u16;
        }
        for p in f.u.iter_mut() {
            *p = 96 + (next() & 0x3F) as u16;
        }
        for p in f.v.iter_mut() {
            *p = 64 + (next() & 0x3F) as u16;
        }
        f
    }

    /// GOP census over committed INTRA leaves inside the P-frame
    /// search tree: `(filter_intra, cfl)` counts.
    fn count_intra_tools(node: &SyntaxNode, fi: &mut u32, cfl: &mut u32) {
        let leafc = |b: &SyntaxBlock, fi: &mut u32, cfl: &mut u32| {
            if b.inter.is_none() {
                if b.use_filter_intra == 1 {
                    *fi += 1;
                }
                if b.uv_mode == Some(crate::cdf::UV_CFL_PRED as u8) {
                    *cfl += 1;
                }
            }
        };
        match node {
            SyntaxNode::Leaf(b) => leafc(b, fi, cfl),
            SyntaxNode::Split(children) => {
                for ch in children.iter() {
                    count_intra_tools(ch, fi, cfl);
                }
            }
            rest => {
                for b in rest.asymmetric_blocks().iter() {
                    leafc(b, fi, cfl);
                }
            }
        }
    }

    /// Shared driver for the two intra-tool witnesses: encode
    /// `[f0, f1]` through the public GOP driver, re-run the P-frame
    /// search tree with the same reference state, and return the
    /// intra-tool census plus the stream round-trip check.
    fn p_frame_intra_tool_census(f0: &YuvFrame, f1: &YuvFrame, base_q_idx: u8) -> (u32, u32) {
        let enc = encode_gop_yuv_with_q(&[f0.clone(), f1.clone()], base_q_idx).unwrap();
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(
                f.planes[0],
                plane8(&enc.recon[idx].y),
                "display {idx}: luma"
            );
            assert_eq!(f.planes[1], plane8(&enc.recon[idx].u), "display {idx}: U");
            assert_eq!(f.planes[2], plane8(&enc.recon[idx].v), "display {idx}: V");
        }
        // Search-tree census with the identical reference state.
        let reference = enc.recon[0].clone();
        let (w, h) = (f0.width, f0.height);
        let key = crate::encoder::key_frame::encode_key_frame_yuv_with_q(f0, base_q_idx).unwrap();
        let fh = build_p_frame_yuv420_8bit_fh_with_q(&key.seq, w, h, base_q_idx, 1, &[]);
        let fs = fh.frame_size.as_ref().unwrap();
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let mut recon = ReconState {
            y: vec![0u16; (w * h) as usize],
            u: vec![0u16; ((w / 2) * (h / 2)) as usize],
            v: vec![0u16; ((w / 2) * (h / 2)) as usize],
            width: w as usize,
            height: h as usize,
            chroma_w: (w / 2) as usize,
            chroma_h: (h / 2) as usize,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(base_q_idx, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut ip = SyntaxInterFrameParams::single_ref_baseline(mi_rows, mi_cols, false);
        ip.interpolation_filter = SWITCHABLE;
        ip.is_motion_mode_switchable = true;
        ip.allow_warped_motion = true;
        let mut ictx = PSearchCtx::new(
            &reference,
            &reference,
            mi_rows,
            mi_cols,
            w as usize,
            h as usize,
            ip,
            1,
            base_q_idx,
            &[],
        )
        .unwrap();
        recon.bd.clear_for_sb(0, 0, mi_rows, mi_cols);
        let tree =
            build_p_search_tree_for_tests(0, 0, BLOCK_64X64, f1, &mut recon, &mut ictx).unwrap();
        if std::env::var("OXIDEAV_AV1_DEBUG_CENSUS").is_ok() {
            fn dump(node: &SyntaxNode, depth: usize) {
                let leafd = |b: &SyntaxBlock, depth: usize| {
                    eprintln!(
                        "{:indent$}leaf inter={} y={} uv={:?} fi={} cfl=({:?},{:?})",
                        "",
                        b.inter.is_some(),
                        b.y_mode,
                        b.uv_mode,
                        b.use_filter_intra,
                        b.cfl_alpha_u,
                        b.cfl_alpha_v,
                        indent = depth * 2
                    );
                };
                match node {
                    SyntaxNode::Leaf(b) => leafd(b, depth),
                    SyntaxNode::Split(children) => {
                        for ch in children.iter() {
                            dump(ch, depth + 1);
                        }
                    }
                    rest => {
                        for b in rest.asymmetric_blocks().iter() {
                            leafd(b, depth);
                        }
                    }
                }
            }
            dump(&tree, 0);
        }
        let (mut fi, mut cfl) = (0u32, 0u32);
        count_intra_tools(&tree, &mut fi, &mut cfl);
        (fi, cfl)
    }

    /// r419 — §5.11.24 FILTER-INTRA must be reachable on intra leaves
    /// INSIDE inter frames: the SE 32×32 quadrant of the P-frame is
    /// constructed as the §7.11.2.3 recursive prediction of its own
    /// decode-time neighbours (derived from a first static-GOP pass —
    /// blocks above/left of the region commit identically in both
    /// passes, so the region predicts exactly and the FI pick beats
    /// every §7.11.2 plain mode), fresh content the inter search has
    /// no source for.
    #[test]
    fn r419_p_search_selects_filter_intra_inside_inter_frame() {
        let base_q_idx = 60u8;
        let f0 = noise_frame(64, 64, 40);
        // Pass 1 — static GOP: the SE-quadrant neighbours' committed
        // reconstruction.
        let enc1 = encode_gop_yuv_with_q(&[f0.clone(), f0.clone()], base_q_idx).unwrap();
        let rec1 = &enc1.recon[1];
        // §7.11.2 neighbour arrays for the 32×32 TU at (32, 32) —
        // the same derivation the leaf picker runs (frame corners
        // off-screen: no above-right / below-left).
        let (above_ext, left_ext, _, _) = super::super::key_frame::build_tu_neighbours(
            &rec1.y, 64, 64, 32, 32, 32, 32, false, false, 8,
        );
        let target = super::super::key_frame::predict_filter_intra_from_neighbours(
            /* FILTER_D157_PRED-class ordinal = */ 2, 32, 32, &above_ext, &left_ext, 8,
        )
        .expect("32x32 filter-intra prediction");
        // Pass 2 — the region carries exactly that prediction.
        let mut f1 = f0.clone();
        for i in 0..32usize {
            for j in 0..32usize {
                f1.y[(32 + i) * 64 + 32 + j] = target[i * 32 + j];
            }
        }
        let (fi, _) = p_frame_intra_tool_census(&f0, &f1, base_q_idx);
        assert!(
            fi > 0,
            "the FI-constructed fresh region must commit at least one \
             §5.11.24 use_filter_intra = 1 intra leaf inside the \
             P-frame (got {fi})"
        );
    }

    /// r419 — §5.11.45 CfL must be reachable on intra leaves INSIDE
    /// inter frames: the SE 32×32 quadrant of the P-frame carries
    /// fresh textured luma whose chroma tracks the subsampled luma AC
    /// (`c = mid ± ac/2`) — the §7.11.5 luma-feedthrough model no
    /// plain §7.11.2 chroma mode can express.
    #[test]
    fn r419_p_search_selects_cfl_inside_inter_frame() {
        let base_q_idx = 60u8;
        // Noise luma, FLAT mid chroma: every plain §7.11.2 chroma
        // mode collapses to the flat field, so the §7.11.5
        // luma-feedthrough is the only model that can track the
        // region's chroma AC.
        let mut f0 = noise_frame(64, 64, 41);
        f0.u.fill(128);
        f0.v.fill(128);
        let mut f1 = f0.clone();
        // Fresh smooth-textured luma in the SE quadrant.
        let tex = |i: usize, j: usize| -> u16 {
            let v = 128.0
                + 70.0 * ((i as f64) * 0.35).sin() * ((j as f64) * 0.29).cos()
                + 30.0 * (((i + j) as f64) * 0.11).sin();
            v.clamp(0.0, 255.0) as u16
        };
        for i in 0..32usize {
            for j in 0..32usize {
                f1.y[(32 + i) * 64 + 32 + j] = tex(i, j);
            }
        }
        // Chroma = mid ± (subsampled luma AC) / 4 over the region —
        // the §7.11.5 model at the (αU, αV) = (2, −2) grid point
        // (`chroma_ac = α · luma_ac / 8`).
        let mut acc = 0i64;
        for i in 0..16usize {
            for j in 0..16usize {
                let s = i64::from(tex(2 * i, 2 * j))
                    + i64::from(tex(2 * i + 1, 2 * j))
                    + i64::from(tex(2 * i, 2 * j + 1))
                    + i64::from(tex(2 * i + 1, 2 * j + 1));
                acc += s;
            }
        }
        let mean4 = acc / 256; // 4x-scaled subsampled mean
        for i in 0..16usize {
            for j in 0..16usize {
                let s = i64::from(tex(2 * i, 2 * j))
                    + i64::from(tex(2 * i + 1, 2 * j))
                    + i64::from(tex(2 * i, 2 * j + 1))
                    + i64::from(tex(2 * i + 1, 2 * j + 1));
                let ac = (s - mean4) / 16; // ± a quarter of the subsampled AC
                let cell = (16 + i) * 32 + 16 + j;
                f1.u[cell] = (128 + ac).clamp(0, 255) as u16;
                f1.v[cell] = (128 - ac).clamp(0, 255) as u16;
            }
        }
        let (_, cfl) = p_frame_intra_tool_census(&f0, &f1, base_q_idx);
        assert!(
            cfl > 0,
            "luma-tracking chroma on a fresh region must commit at \
             least one §5.11.22 UV_CFL_PRED intra leaf inside the \
             P-frame (got {cfl})"
        );
    }

    /// r419 conformance-fixture twins: the three self-encoded GOP
    /// streams pinned in `tests/fixture_conformance.rs` regenerate
    /// deterministically from the witness content families —
    /// (1) sheared motion (§5.11.27 OBMC leaves, per the OBMC
    /// selection witness), (2) zooming motion (§5.11.27 WARPED_CAUSAL
    /// leaves, per the warp selection witness), (3) noise + FI-exact +
    /// CfL-tracking regions (§5.11.24 filter-intra and §5.11.22
    /// UV_CFL_PRED intra leaves INSIDE inter frames, per the two
    /// intra-tool witnesses). Each round-trips byte-exact through the
    /// spec driver; when `OXIDEAV_AV1_R419_FIXDIR` is set the exact
    /// streams + display-order reconstructions are written for
    /// external validation / fixture staging.
    #[test]
    fn r419_motion_mode_and_intra_tool_fixture_streams() {
        let dump = |name: &str, enc: &EncodedGopYuv| {
            let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
            assert_eq!(decoded.len(), enc.recon.len(), "{name}");
            for (idx, f) in decoded.iter().enumerate() {
                assert_eq!(
                    f.planes[0],
                    plane8(&enc.recon[idx].y),
                    "{name} display {idx}: luma"
                );
                assert_eq!(
                    f.planes[1],
                    plane8(&enc.recon[idx].u),
                    "{name} display {idx}: U"
                );
                assert_eq!(
                    f.planes[2],
                    plane8(&enc.recon[idx].v),
                    "{name} display {idx}: V"
                );
            }
            if let Ok(dir) = std::env::var("OXIDEAV_AV1_R419_FIXDIR") {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(format!("{dir}/{name}.ivf"), &enc.ivf_bytes).unwrap();
                let mut yuv = Vec::new();
                for rc in &enc.recon {
                    yuv.extend_from_slice(&plane8(&rc.y));
                    yuv.extend_from_slice(&plane8(&rc.u));
                    yuv.extend_from_slice(&plane8(&rc.v));
                }
                std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
            }
        };

        // --- self-gop-64x64-q100-shear-obmc: 3-frame shear GOP (the
        // OBMC witness content family — per-row shift ramp 24..48,
        // dx_top = 4k). ---
        {
            let tex = |i: usize, j: usize| -> u16 {
                ((i * 7 + j * 11 + (i / 4) * (j / 8) + ((i * j) / 13)) % 256) as u16
            };
            let frames: Vec<YuvFrame> = (0..3)
                .map(|k| {
                    let mut f = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 128);
                    for i in 0..64usize {
                        let (ramp0, ramp1, top) = (24usize, 48usize, 4 * k);
                        let dx = if i < ramp0 {
                            top
                        } else if i >= ramp1 {
                            0
                        } else {
                            top * (ramp1 - i) / (ramp1 - ramp0)
                        };
                        for j in 0..64usize {
                            f.y[i * 64 + j] = tex(i, j + dx);
                        }
                    }
                    f
                })
                .collect();
            let enc = encode_gop_yuv_with_q(&frames, 100).unwrap();
            dump("self-gop-64x64-q100-shear-obmc", &enc);
        }

        // --- self-gop-96x80-q100-zoom-warp: 2-frame zoom GOP (the
        // WARPED_CAUSAL witness content itself). ---
        {
            let [f0, f1] = zoom_pair(96, 80, 17, 16);
            let enc = encode_gop_yuv_with_q(&[f0, f1], 100).unwrap();
            dump("self-gop-96x80-q100-zoom-warp", &enc);
        }

        // --- self-gop-64x64-q60-fi-cfl: KEY + FI P-frame + CfL
        // P-frame (the two intra-tool witness constructions composed
        // into one GOP over a noise-luma / flat-chroma anchor). ---
        {
            let base_q_idx = 60u8;
            let mut f0 = noise_frame(64, 64, 41);
            f0.u.fill(128);
            f0.v.fill(128);
            // FI frame: SE quadrant = the §7.11.2.3 prediction of its
            // own decode-time neighbours (static pass 1).
            let enc1 = encode_gop_yuv_with_q(&[f0.clone(), f0.clone()], base_q_idx).unwrap();
            let rec1 = &enc1.recon[1];
            let (above_ext, left_ext, _, _) = super::super::key_frame::build_tu_neighbours(
                &rec1.y, 64, 64, 32, 32, 32, 32, false, false, 8,
            );
            let target = super::super::key_frame::predict_filter_intra_from_neighbours(
                2, 32, 32, &above_ext, &left_ext, 8,
            )
            .expect("32x32 filter-intra prediction");
            let mut f1 = f0.clone();
            for i in 0..32usize {
                for j in 0..32usize {
                    f1.y[(32 + i) * 64 + 32 + j] = target[i * 32 + j];
                }
            }
            // CfL frame: SE quadrant = smooth texture with
            // luma-tracking chroma at the (2, −2) grid point.
            let tex = |i: usize, j: usize| -> u16 {
                let v = 128.0
                    + 70.0 * ((i as f64) * 0.35).sin() * ((j as f64) * 0.29).cos()
                    + 30.0 * (((i + j) as f64) * 0.11).sin();
                v.clamp(0.0, 255.0) as u16
            };
            let mut f2 = f1.clone();
            for i in 0..32usize {
                for j in 0..32usize {
                    f2.y[(32 + i) * 64 + 32 + j] = tex(i, j);
                }
            }
            let mut acc = 0i64;
            for i in 0..16usize {
                for j in 0..16usize {
                    acc += i64::from(tex(2 * i, 2 * j))
                        + i64::from(tex(2 * i + 1, 2 * j))
                        + i64::from(tex(2 * i, 2 * j + 1))
                        + i64::from(tex(2 * i + 1, 2 * j + 1));
                }
            }
            let mean4 = acc / 256;
            for i in 0..16usize {
                for j in 0..16usize {
                    let s = i64::from(tex(2 * i, 2 * j))
                        + i64::from(tex(2 * i + 1, 2 * j))
                        + i64::from(tex(2 * i, 2 * j + 1))
                        + i64::from(tex(2 * i + 1, 2 * j + 1));
                    let ac = (s - mean4) / 16;
                    let cell = (16 + i) * 32 + 16 + j;
                    f2.u[cell] = (128 + ac).clamp(0, 255) as u16;
                    f2.v[cell] = (128 - ac).clamp(0, 255) as u16;
                }
            }
            let enc = encode_gop_yuv_with_q(&[f0, f1, f2], base_q_idx).unwrap();
            dump("self-gop-64x64-q60-fi-cfl", &enc);
        }
    }
}
