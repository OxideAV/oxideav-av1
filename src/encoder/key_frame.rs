//! Conformance-grade intra KEY-frame encoder (r409, generalised r410).
//!
//! Unlike the historical encoder-mirror drivers in
//! [`super::pixel_driver`] / [`super::pixel_driver_dyn`] (whose leaf
//! syntax codes `y_mode` with the §5.11.22 non-keyframe CDFs and is
//! therefore decodable only by this crate's matching mirror decoder),
//! this driver emits the REAL §5.11 keyframe syntax through the
//! spec-faithful write side ([`super::write_partition_tree_syntax`],
//! whose output the decode walker
//! [`crate::cdf::PartitionWalker::decode_partition_syntax`] replays
//! bit-for-bit): §5.11.7 `intra_frame_mode_info()` with the
//! neighbour-CDF `intra_frame_y_mode`, the §5.11.22 `uv_mode` +
//! §5.11.45 CFL alphas, and the §5.11.34 `residual()` per-TU
//! coefficient emission with live §8.3.2 contexts.
//!
//! The produced bitstream is a complete §5.2 low-overhead stream:
//! IVF v0 container, `OBU_TEMPORAL_DELIMITER` then
//! `OBU_SEQUENCE_HEADER` then the combined §5.10 `OBU_FRAME` (frame
//! header, `byte_alignment()`, tile group in one OBU) — decodable by
//! the crate's own spec-faithful frame driver
//! ([`crate::decoder::decode_av1_spec`]) and by independent AV1
//! decoders.
//!
//! ## Scope (r410, format-generalised r427)
//!
//! * Any §6.4.1 (bit depth, chroma format) pairing ([`YuvFrame`]):
//!   8 / 10 / 12-bit samples in 4:2:0, 4:2:2, 4:4:4 or monochrome
//!   layout, with `seq_profile` elected per pairing (the historical
//!   [`Yuv420Frame`] entries widen into the same core). `(width,
//!   height)` multiples of 8 in `[8, KEY_FRAME_MAX_DIM]` per axis
//!   (any rectangle; frames wider/taller than 64 ride the
//!   multi-superblock walk). The §5.11.46 palette and §5.9.20
//!   intra-block-copy elections stay 8-bit-4:2:0-scoped (an encoder
//!   choice); under 4:2:2 the §5.11.38 admissibility rule bars the
//!   tall partition shapes from the search.
//! * One KEY frame per stream (`show_frame = 1`,
//!   `error_resilient_mode = 1`, `refresh_frame_flags = allFrames`),
//!   single tile, 64×64 superblocks.
//! * `base_q_idx == 0` selects the §5.9.2 `CodedLossless` arm
//!   (forward/inverse WHT, `TxMode = ONLY_4X4`): the decoded planes
//!   equal the input byte-for-byte. `base_q_idx > 0` selects the
//!   lossy DCT_DCT arm (`TxMode = TX_MODE_LARGEST`): the decoded
//!   planes equal the encoder's own reconstruction byte-for-byte.
//! * Full square §5.11.4 partition search (r410): every in-frame
//!   square node from `BLOCK_64X64` down to `BLOCK_8X8` is
//!   trial-encoded both as one `PARTITION_NONE` leaf and as a
//!   `PARTITION_SPLIT` of four recursively-searched quadrants,
//!   keeping the lower rate-distortion score (SSD over the node's
//!   pixels + a q-scaled coefficient/mode rate proxy). Nodes that
//!   straddle the frame edge take the §5.11.4 forced-split arms.
//!   A leaf at `BLOCK_N×N` codes its luma as one
//!   `Max_Tx_Size_Rect[MiSize]` TU on the lossy `TX_MODE_LARGEST`
//!   arm (`TX_8X8` … `TX_64X64`, the latter with the §7.12.3
//!   compact-`tw` coefficient layout) or as the §5.11.35 grid of
//!   `TX_4X4` TUs on the lossless arm; chroma rides the §5.11.38
//!   `get_tx_size` derivation (`TX_4X4` … `TX_32X32`).
//! * Per-leaf luma mode decision by residual SSD over ALL 13 §6.10.x
//!   intra modes (r410): the directional D-modes run the §7.11.2.4
//!   projection kernel against §7.11.2.1 neighbour arrays derived
//!   with the real `haveAboveRight` / `haveBelowLeft` availability —
//!   the encoder mirrors the §6.10.3 `BlockDecoded[]` superblock
//!   state (§5.11.3 per-SB clear + §5.11.35 per-TU stamps) so its
//!   neighbour extension is bit-identical to the decode walker's.
//!   `angle_delta` stays 0. The chroma picker evaluates the same 13
//!   modes jointly over U+V plus `UV_CFL_PRED` (§7.11.5) over a
//!   compact (αU, αV) grid when the §5.11.22 `cfl_allowed` gate
//!   (`Block_Width <= 32 && Block_Height <= 32`) is open.
//! * `skip = 1` on leaves whose every TU quantises to zero (residual
//!   == prediction), `skip = 0` with per-TU `Quant[]` commitments
//!   otherwise.
//! * In-loop filters all disabled by the header set (`CodedLossless`
//!   forces them off at q=0; the lossy FH keeps deblock levels 0,
//!   `enable_cdef = 0`, `enable_restoration = 0`, `enable_superres =
//!   0`), so the §7.4 post chain is a no-op and the walker
//!   reconstruction IS the output frame.
//!
//! ## Why the reconstruction loop is exact
//!
//! Every per-TU prediction is computed from the encoder's running
//! reconstruction with the same §7.11.2 kernels the decode walker
//! runs — including the §7.11.2.1 `AboveRow[]` / `LeftCol[]`
//! derivation with the `aboveLimit` / `leftLimit` extension clamps
//! and the §6.10.3 `BlockDecoded[]`-driven `haveAboveRight` /
//! `haveBelowLeft` reads — and the coefficient inverse
//! ([`crate::cdf::dequantize_step1`] +
//! [`crate::transform::inverse_transform_2d`]) is the decoder's own
//! primitive. So the encoder's `recon` tracks the decoder's
//! `CurrFrame` sample-for-sample by induction along the §5.11.4
//! dispatch order. On the lossless arm the residual chain is
//! bit-exact (`recon == input` at every leaf), which the round-trip
//! suite pins.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.2/§5.3 (framing),
//! §5.5/§5.9 (headers), §5.10 (`frame_obu`), §5.11 (tile syntax; the
//! §5.11.3 `clear_block_decoded_flags` reset and §5.11.35
//! `transform_block` stamps for `BlockDecoded[]`), §7.11.2 (intra
//! prediction), §7.11.5 (CFL), §7.12/§7.13 (quant + transforms).

use crate::cdf::{
    dequantize_step1, get_tx_size, intra_tx_type_set, is_tx_type_in_set, tx_size_sqr_index,
    QuantizerParams, TileCdfContext, TileGeometry, ANGLE_STEP, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8,
    D67_PRED, DCT_DCT, DC_PRED, H_PRED, INTRA_MODES, MAX_TX_DEPTH, MAX_TX_SIZE_RECT, MODE_TO_ANGLE,
    MODE_TO_TXFM, NUM_4X4_BLOCKS_WIDE, PAETH_PRED, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED,
    SPLIT_TX_SIZE, TX_4X4, TX_HEIGHT, TX_SIZE_SQR_UP, TX_WIDTH, UV_CFL_PRED, V_PRED,
};
use crate::cdf::{
    predict_intra_dc_pred, predict_intra_directional, predict_intra_h_pred,
    predict_intra_paeth_pred, predict_intra_smooth_h_pred, predict_intra_smooth_pred,
    predict_intra_smooth_v_pred, predict_intra_v_pred,
};
use crate::encoder::forward_quantize::forward_quantize;
use crate::encoder::forward_transform_2d::forward_transform_2d;
use crate::encoder::forward_wht::forward_wht_4x4;
use crate::encoder::frame_obu::encode_uncompressed_header;
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::obu::{build_temporal_unit, ObuFrame};
use crate::encoder::partition_tree::{
    write_partition_tree_syntax, PartitionSyntaxWriter, SyntaxBlock, SyntaxFrameParams, SyntaxNode,
};
use crate::encoder::pixel_driver_dyn::{
    build_intra_only_yuv420_8bit_fh_with_q, sb_grid_origins, Yuv420Frame,
};
use crate::encoder::rate_twin::{score256, RateModel, RateTwin, TuCtx, TuFork};
use crate::encoder::sequence_obu::write_sequence_header_obu;
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::tile_group_obu::{write_tile_group_obu, TileGroupObu, TilePayload};
use crate::encoder::yuv_frame::{build_intra_only_seq_yuv, ChromaFormat, YuvFrame};
use crate::frame_header::FrameHeader;
use crate::obu::ObuType;
use crate::sequence_header::SequenceHeader;
use crate::transform::inverse_transform_2d;
use crate::Error;

/// Result of [`encode_key_frame_yuv`] / [`encode_key_frame_yuv_with_q`]
/// — the general-format sibling of [`EncodedKeyFrame`] (r427): recon
/// planes are `u16` at the input's bit depth, chroma at the input's
/// subsampled extent (empty on the monochrome arm).
#[derive(Debug, Clone)]
pub struct EncodedKeyFrameYuv {
    /// Complete IVF v0 file (header + one frame record).
    pub ivf_bytes: Vec<u8>,
    /// The bare §7.5 temporal unit (TD + SH + §5.10 `OBU_FRAME`).
    pub temporal_unit_bytes: Vec<u8>,
    /// Encoder reconstruction of the luma plane (row-major). The
    /// decoded output equals these sample-for-sample; at `base_q_idx
    /// == 0` they additionally equal the input.
    pub recon_y: Vec<u16>,
    /// U plane reconstruction (subsampled extent; empty on
    /// monochrome).
    pub recon_u: Vec<u16>,
    /// V plane reconstruction.
    pub recon_v: Vec<u16>,
    /// The emitted sequence header descriptor.
    pub seq: SequenceHeader,
    /// The emitted frame header descriptor.
    pub fh: FrameHeader,
}

/// Result of [`encode_key_frame_yuv420`] /
/// [`encode_key_frame_yuv420_with_q`].
#[derive(Debug, Clone)]
pub struct EncodedKeyFrame {
    /// Complete IVF v0 file (header + one frame record).
    pub ivf_bytes: Vec<u8>,
    /// The bare §7.5 temporal unit (TD + SH + §5.10 `OBU_FRAME`) — the
    /// per-packet payload a container muxer would carry.
    pub temporal_unit_bytes: Vec<u8>,
    /// Encoder reconstruction of the three planes (row-major). The
    /// decoded output equals these byte-for-byte; at `base_q_idx == 0`
    /// they additionally equal the input.
    pub recon_y: Vec<u8>,
    /// U plane reconstruction (`(width/2) × (height/2)`).
    pub recon_u: Vec<u8>,
    /// V plane reconstruction.
    pub recon_v: Vec<u8>,
    /// The emitted sequence header descriptor.
    pub seq: SequenceHeader,
    /// The emitted frame header descriptor.
    pub fh: FrameHeader,
}

/// Per-axis extent bound (inclusive) for [`encode_key_frame_yuv420`]
/// — r410 raises the r409 `512` cap to `4096` (the RD search works
/// superblock-by-superblock, so state stays flat; HD/UHD extents were
/// validated against independent black-box decoders during the round).
pub const KEY_FRAME_MAX_DIM: u32 = 4096;

/// §5.11.45 (αU, αV) candidate grid for the chroma `UV_CFL_PRED` arm —
/// the same compact set the dyn mirror driver enumerates.
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

/// Lossless (`base_q_idx = 0`) conformance-grade KEY-frame encode —
/// see [`encode_key_frame_yuv420_with_q`].
pub fn encode_key_frame_yuv420(input: &Yuv420Frame) -> Result<EncodedKeyFrame, Error> {
    encode_key_frame_yuv420_with_q(input, 0)
}

/// Encode one 8-bit 4:2:0 KEY frame at `base_q_idx` into a
/// spec-conformant IVF stream (see the module docs for the exact
/// scope and the reconstruction-exactness argument).
///
/// ## Errors
///
/// * Dimensions not multiples of 8, or outside
///   `[8, KEY_FRAME_MAX_DIM]` per axis, or plane lengths inconsistent
///   with the dimensions — [`Error::PartitionWalkOutOfRange`].
/// * Internal writer overflow surfaces the underlying [`Error`].
pub fn encode_key_frame_yuv420_with_q(
    input: &Yuv420Frame,
    base_q_idx: u8,
) -> Result<EncodedKeyFrame, Error> {
    encode_key_frame_yuv420_with_q_rate_model(input, base_q_idx, RateModel::Twin)
}

/// r421 — [`encode_key_frame_yuv420_with_q`] with an explicit
/// [`RateModel`], kept public (hidden) so the sweep harnesses can A/B
/// the twin-priced elections against the pre-r421 heuristic baseline
/// on identical inputs.
#[doc(hidden)]
pub fn encode_key_frame_yuv420_with_q_rate_model(
    input: &Yuv420Frame,
    base_q_idx: u8,
    model: RateModel,
) -> Result<EncodedKeyFrame, Error> {
    encode_key_frame_yuv420_with_q_carry(input, base_q_idx, model).map(|(k, _)| k)
}

/// r423 — [`encode_key_frame_yuv420_with_q_rate_model`] plus the §7.20
/// per-slot reference state the KEY frame's `allFrames` refresh
/// deposits: the §8.4 frame-end CDF table (`save_cdfs`, the adapted
/// state of the single tile — `context_update_tile_id = 0`,
/// `disable_frame_end_update_cdf = 0`), the `SavedSegmentIds` grid
/// (all-zero — `segmentation_enabled = 0` stamps the literal `0` per
/// block), and the `SavedGmParams` identity table. A following INTER
/// frame electing `primary_ref_frame != PRIMARY_REF_NONE` against a
/// KEY-refreshed slot loads exactly this state (§5.9.2 `load_cdfs` +
/// `load_previous` + `load_previous_segment_ids`).
pub(crate) fn encode_key_frame_yuv420_with_q_carry(
    input: &Yuv420Frame,
    base_q_idx: u8,
    model: RateModel,
) -> Result<(EncodedKeyFrame, crate::encoder::inter_frame::RefSlotCarry), Error> {
    encode_key_frame_yuv420_with_q_seg_carry(input, base_q_idx, model, &[], None)
}

/// r426 — [`encode_key_frame_yuv420_with_q_carry`] with §5.9.14
/// SEG_LVL_ALT_Q segmentation and the exactness-demand mask: the KEY
/// frame codes the same delta table as the GOP's P-frames
/// (`segmentation_enabled = 1`, the §5.11.8 `intra_segment_id` per
/// block), and every leaf overlapping `exact_mask` commits the
/// table's lossless segment — its blocks run the §5.9.2
/// `LosslessArray[]` TX_4X4/WHT semantics, so the demanded region
/// reconstructs pixel-exact from frame 0. `alt_q` empty keeps the
/// unsegmented r413 shape (and `exact_mask` must then be `None`).
pub(crate) fn encode_key_frame_yuv420_with_q_seg_carry(
    input: &Yuv420Frame,
    base_q_idx: u8,
    model: RateModel,
    alt_q: &[i16],
    exact_mask: Option<&[bool]>,
) -> Result<(EncodedKeyFrame, crate::encoder::inter_frame::RefSlotCarry), Error> {
    let wide = YuvFrame::from_yuv420_8bit(input);
    let (k, carry) = encode_key_frame_yuv_seg_carry(&wide, base_q_idx, model, alt_q, exact_mask)?;
    let narrow = |p: Vec<u16>| p.into_iter().map(|s| s as u8).collect::<Vec<u8>>();
    Ok((
        EncodedKeyFrame {
            ivf_bytes: k.ivf_bytes,
            temporal_unit_bytes: k.temporal_unit_bytes,
            recon_y: narrow(k.recon_y),
            recon_u: narrow(k.recon_u),
            recon_v: narrow(k.recon_v),
            seq: k.seq,
            fh: k.fh,
        },
        carry,
    ))
}

/// r427 — general-format lossless (`base_q_idx = 0`) KEY-frame encode:
/// see [`encode_key_frame_yuv_with_q`].
pub fn encode_key_frame_yuv(input: &YuvFrame) -> Result<EncodedKeyFrameYuv, Error> {
    encode_key_frame_yuv_with_q(input, 0)
}

/// r427 — encode one KEY frame at any conformant (bit depth, chroma
/// format) pairing into a spec-conformant IVF stream: 8 / 10 / 12-bit
/// samples in 4:2:0, 4:2:2, 4:4:4 or monochrome layout, with the
/// §6.4.1 `seq_profile` elected per pairing. Same scope and
/// reconstruction-exactness argument as
/// [`encode_key_frame_yuv420_with_q`] (whose 8-bit 4:2:0 arm now
/// routes through this driver); the §5.11.46 palette and §5.9.20
/// intra-block-copy elections stay 8-bit-4:2:0-scoped (an encoder
/// choice — the header gates close on the other pairings).
///
/// ## Errors
///
/// * [`YuvFrame::validate`] failures (shape / depth / sample range) —
///   [`Error::PartitionWalkOutOfRange`].
/// * Internal writer overflow surfaces the underlying [`Error`].
pub fn encode_key_frame_yuv_with_q(
    input: &YuvFrame,
    base_q_idx: u8,
) -> Result<EncodedKeyFrameYuv, Error> {
    encode_key_frame_yuv_seg_carry(input, base_q_idx, RateModel::Twin, &[], None).map(|(k, _)| k)
}

/// r427 — the general-format KEY-frame core: every entry point above
/// funnels here. `alt_q` / `exact_mask` carry the r426 §5.9.14
/// SEG_LVL_ALT_Q + exactness-demand configuration.
pub(crate) fn encode_key_frame_yuv_seg_carry(
    input: &YuvFrame,
    base_q_idx: u8,
    model: RateModel,
    alt_q: &[i16],
    exact_mask: Option<&[bool]>,
) -> Result<
    (
        EncodedKeyFrameYuv,
        crate::encoder::inter_frame::RefSlotCarry,
    ),
    Error,
> {
    // r426 — same table validation as the P-frame entry (`alt_q[0] ==
    // 0`, su(1+8) range, no over-255 sums, no base-0 segmentation).
    if !alt_q.is_empty() {
        if alt_q.len() == 1
            || alt_q.len() > crate::uncompressed_header_tail::MAX_SEGMENTS
            || alt_q[0] != 0
            || base_q_idx == 0
            || model != RateModel::Twin
        {
            // (The §5.11.9 skip-leaf pred inheritance rides the rate
            // twin's writer mirror — the segmented-KEY arm is
            // twin-model-only, like every production entry point.)
            return Err(Error::PartitionWalkOutOfRange);
        }
        for &d in alt_q {
            if !(-255..=255).contains(&d) || i32::from(base_q_idx) + i32::from(d) > 255 {
                return Err(Error::PartitionWalkOutOfRange);
            }
        }
    } else if exact_mask.is_some() {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // Shape / depth / sample-range gate (multiples of 8, [8, MAX] per
    // axis, planes consistent with the format).
    input.validate()?;
    let bit_depth = input.bit_depth;
    let (ssx, ssy) = input.format.subsampling();
    let mono = input.format == ChromaFormat::Monochrome;
    let num_planes = input.format.num_planes();
    let width = input.width as usize;
    let height = input.height as usize;
    let chroma_w = input.chroma_width() as usize;
    let chroma_h = input.chroma_height() as usize;
    // §5.11.46 palette + §5.9.20 intra-block-copy scope (r427): the
    // screen-content searches stay 8-bit 4:2:0 (their content scans
    // and the even-DV chroma alignment are built for that pairing);
    // other pairings close the §5.9.5 header gate — an encoder
    // election, not a conformance constraint.
    let sc_eligible = bit_depth == 8 && input.format == ChromaFormat::Yuv420;

    let mut seq = build_intra_only_seq_yuv(input.width, input.height, bit_depth, input.format)?;
    // r410: open the §5.11.24 filter-intra gate — the mode picker now
    // evaluates the five §7.11.2.3 recursive modes on eligible luma
    // blocks (the historical mirror drivers build their own sequence
    // headers and stay unaffected).
    seq.enable_filter_intra = true;
    let mut fh =
        build_intra_only_yuv420_8bit_fh_with_q(&seq, input.width, input.height, base_q_idx);
    // r427 — screen-content election is 8-bit-4:2:0-scoped (see
    // `sc_eligible` above); the §5.9.5 SELECT arm codes the header
    // bit either way.
    fh.allow_screen_content_tools = fh.allow_screen_content_tools && sc_eligible;
    // r418: open the §5.9.20 intra-block-copy gate only when the
    // content-level scan finds at least one §6.10.24-reachable exact
    // duplicate superblock — the per-leaf `use_intrabc` S() overhead
    // is only worth coding when a copy source provably exists.
    fh.allow_intrabc = fh.allow_screen_content_tools && intrabc_beneficial(input);
    // r410: the lossy arm codes §5.9.21 `TxMode = TX_MODE_SELECT` so
    // every leaf carries the §5.11.15 `tx_depth` choice the RD search
    // makes (lossless stays on the §5.9.2 CodedLossless ONLY_4X4 arm).
    if base_q_idx > 0 {
        fh.tx_mode = Some(crate::uncompressed_header_tail::TxMode::TxModeSelect);
    }
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);

    let lossless = base_q_idx == 0;
    let mut qp = QuantizerParams::neutral(base_q_idx, bit_depth);
    if !alt_q.is_empty() {
        // r426 — §7.12.2 get_qindex inputs: the write-side §5.11.47
        // guard and the §5.11.39 quantiser chain key off these.
        qp.segmentation_enabled = true;
        for (seg, &d) in alt_q.iter().enumerate() {
            qp.seg_alt_q_active[seg] = true;
            qp.seg_alt_q_data[seg] = d;
        }
        fh.segmentation_params = Some(crate::encoder::inter_frame::segmentation_params_for(alt_q));
    }
    // r426 — the mask requires a lossless segment; resolve it once.
    let seg_ll = crate::encoder::inter_frame::seg_lossless_array(base_q_idx, alt_q);
    let ll_seg = (0..alt_q.len())
        .find(|&sid| seg_ll[sid])
        .map(|sid| sid as u8);
    if exact_mask.is_some() && ll_seg.is_none() {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if let Some(mask) = exact_mask {
        if mask.len() != (mi_rows as usize) * (mi_cols as usize) {
            return Err(Error::PartitionWalkOutOfRange);
        }
    }

    // §5.11 frame-scope parameter bundle — mirrors the decode driver's
    // `TileDecodeParams` derivation for this header set field by field.
    let params = SyntaxFrameParams {
        subsampling_x: ssx,
        subsampling_y: ssy,
        num_planes,
        seg_id_pre_skip: false,
        segmentation_enabled: !alt_q.is_empty(),
        seg_ref_frame: [None; crate::uncompressed_header_tail::MAX_SEGMENTS],
        seg_skip: [false; crate::uncompressed_header_tail::MAX_SEGMENTS],
        seg_globalmv: [false; crate::uncompressed_header_tail::MAX_SEGMENTS],
        last_active_seg_id: alt_q.len().saturating_sub(1) as u8,
        lossless_array: seg_ll,
        coded_lossless: lossless,
        enable_cdef: seq.enable_cdef,
        allow_intrabc: fh.allow_intrabc,
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
        inter: None,
    };

    // Running reconstruction — tracks the decoder's `CurrFrame`
    // sample-for-sample (see module docs).
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
        allow_intrabc: fh.allow_intrabc,
        qp,
        bd: BlockDecodedMirror::new(ssx, ssy, num_planes),
        // r425 — arm the exact-match DV index whenever the §5.9.20
        // gate opened (input-space hash seeds for the §5.11.7
        // search; validity + SSD/RD stay reconstruction-space).
        dv_hash: if fh.allow_intrabc {
            crate::encoder::dv_hash::DvHashIndex::build(&input.y, width, height)
        } else {
            crate::encoder::dv_hash::DvHashIndex::default()
        },
    };

    // §5.11.2 tile walk: one tile, 64×64 superblocks in raster order.
    // Per superblock: §5.11.3 `clear_block_decoded_flags`, then the
    // recursive §5.11.4 rate-distortion partition search (running the
    // mode picker + residual pipeline at every leaf in NW/NE/SW/SE
    // dispatch order), then emit its syntax.
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
        // r421 — arm the §5.11.2 delta lifecycle on the live state AND
        // the rate twin's fork, so both enter the superblock identically.
        state.arm_read_deltas();
        let mut twin = RateTwin::snapshot(&cdfs, &state, &writer);
        twin.arm_read_deltas();
        let seg_demand = exact_mask.map(|mask| KeySegDemand {
            mask,
            ll_seg: ll_seg.expect("mask requires a lossless segment (validated above)"),
        });
        let (tree, _cost) = build_search_tree(
            sb_r,
            sb_c,
            BLOCK_64X64,
            input,
            &mut recon,
            &mut twin,
            &params,
            model,
            seg_demand.as_ref(),
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
        // symbols the writer just emitted, so the twin's CDFs and
        // coder range must equal the live state — always, in both
        // rate models (commits are model-independent).
        debug_assert!(
            twin.matches(&cdfs, &writer),
            "rate twin desynced from the writer after superblock ({sb_r},{sb_c})"
        );
    }
    let tile_bytes = writer.finish();

    // §5.11.1 tile-group body (single tile).
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

    // §5.10 `frame_obu()`: `frame_header_obu()` + `byte_alignment()`
    // (zero pad — NOT §5.3.4 trailing_bits; OBU_FRAME is one of the
    // §5.3.1 trailer-exempt types) + `tile_group_obu()`.
    let frame_body = {
        let mut bw = crate::encoder::bitwriter::BitWriter::new();
        encode_uncompressed_header(&mut bw, &fh, &seq);
        bw.byte_align();
        let mut body = bw.finish();
        body.extend_from_slice(&tile_group_body);
        body
    };
    // §7.5 temporal unit: TD + SH + OBU_FRAME.
    let sh_body = write_sequence_header_obu(&seq);
    let temporal_unit_bytes =
        build_temporal_unit(Some(&sh_body), &[ObuFrame::new(ObuType::Frame, frame_body)]);

    // IVF v0 wrap.
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

    // r423 — §7.20 slot payload for the KEY frame's `allFrames`
    // refresh: `save_cdfs` takes the tile's adapted frame-end CDFs
    // (single tile ⇒ `context_update_tile_id = 0` donated them;
    // `disable_frame_end_update_cdf = 0` on this header), and
    // `SavedSegmentIds` is the walker mirror's fully-stamped grid
    // (all-zero — §5.11.8 disabled-branch stamps).
    let carry = crate::encoder::inter_frame::RefSlotCarry {
        cdfs: Box::new(cdfs),
        segment_ids: state.mirror().segment_ids().to_vec(),
        mi_rows,
        mi_cols,
        gm_params: crate::uncompressed_header_tail::prev_gm_params_default(),
    };

    Ok((
        EncodedKeyFrameYuv {
            ivf_bytes,
            temporal_unit_bytes,
            recon_y: recon.y,
            recon_u: recon.u,
            recon_v: recon.v,
            seq,
            fh,
        },
        carry,
    ))
}

// ---------------------------------------------------------------------
// §6.10.3 `BlockDecoded[]` encoder-side mirror (r410).
// ---------------------------------------------------------------------

/// SB-local index span: `-1..=16` for the 64×64 superblock
/// (`sbSize4 = 16`), folded by `+1` into `0..18`.
const BD_STRIDE: usize = 18;

/// Encoder-side mirror of the decode walker's §6.10.3
/// `BlockDecoded[ plane ][ y ][ x ]` superblock-local state — the
/// §5.11.3 per-SB clear plus the §5.11.35 per-TU stamps. Drives the
/// §7.11.2.1 `haveAboveRight` / `haveBelowLeft` reads so the
/// encoder's directional-mode neighbour extension is bit-identical
/// to the decoder's.
#[derive(Clone)]
pub(crate) struct BlockDecodedMirror {
    bd: Vec<u8>,
    /// §5.5.2 `subsampling_x` for the chroma planes (r427 — the
    /// §5.11.3 clear and the §5.11.35 anchor derivations key off it).
    sub_x: u8,
    /// §5.5.2 `subsampling_y`.
    sub_y: u8,
    /// §5.5.2 `NumPlanes` (1 or 3).
    num_planes: u8,
}

impl BlockDecodedMirror {
    pub(crate) fn new(sub_x: u8, sub_y: u8, num_planes: u8) -> Self {
        Self {
            bd: vec![0u8; 3 * BD_STRIDE * BD_STRIDE],
            sub_x,
            sub_y,
            num_planes,
        }
    }

    /// Per-plane §6.4.2 `(subX, subY)` (`(0, 0)` for luma, the
    /// sequence pair for chroma).
    #[inline]
    fn plane_sub(&self, plane: usize) -> (u32, u32) {
        if plane > 0 {
            (u32::from(self.sub_x), u32::from(self.sub_y))
        } else {
            (0, 0)
        }
    }

    #[inline]
    fn slot(plane: usize, y: i32, x: i32) -> Option<usize> {
        if plane >= 3 {
            return None;
        }
        let yi = y + 1;
        let xi = x + 1;
        if yi < 0 || xi < 0 {
            return None;
        }
        let (yi, xi) = (yi as usize, xi as usize);
        if yi >= BD_STRIDE || xi >= BD_STRIDE {
            return None;
        }
        Some(plane * BD_STRIDE * BD_STRIDE + yi * BD_STRIDE + xi)
    }

    /// §5.11.3 `clear_block_decoded_flags( r, c, sbSize4 = 16 )` for a
    /// single-tile frame (`MiRowEnd = MiRows`, `MiColEnd = MiCols`) at
    /// this mirror's subsampling / plane count.
    pub(crate) fn clear_for_sb(&mut self, sb_r: u32, sb_c: u32, mi_rows: u32, mi_cols: u32) {
        const SB_SIZE4: i32 = 16;
        for plane in 0..usize::from(self.num_planes) {
            let (sx, sy) = self.plane_sub(plane);
            let (sub_x, sub_y) = (sx as i32, sy as i32);
            let sb_width4 = (mi_cols as i32 - sb_c as i32) >> sub_x;
            let sb_height4 = (mi_rows as i32 - sb_r as i32) >> sub_y;
            let y_max = SB_SIZE4 >> sub_y;
            let x_max = SB_SIZE4 >> sub_x;
            for y in -1..=y_max {
                for x in -1..=x_max {
                    let val = u8::from((y < 0 && x < sb_width4) || (x < 0 && y < sb_height4));
                    if let Some(slot) = Self::slot(plane, y, x) {
                        self.bd[slot] = val;
                    }
                }
            }
            // §5.11.3 final line: the below-left corner of the SB is
            // never available.
            if let Some(slot) = Self::slot(plane, y_max, -1) {
                self.bd[slot] = 0;
            }
        }
    }

    #[inline]
    fn get(&self, plane: usize, y: i32, x: i32) -> bool {
        Self::slot(plane, y, x).is_some_and(|s| self.bd[s] != 0)
    }

    #[inline]
    fn set(&mut self, plane: usize, y: i32, x: i32) {
        if let Some(slot) = Self::slot(plane, y, x) {
            self.bd[slot] = 1;
        }
    }
}

/// SB-local §5.11.35 anchor for a TU at plane-space `(start_x,
/// start_y)`: `(base_row, base_col)` per the spec's
/// `subBlockMiRow/Col` derivation (`sbMask = 15` for 64×64 SBs).
#[inline]
fn tu_bd_anchor(
    bd: &BlockDecodedMirror,
    plane: usize,
    start_x: usize,
    start_y: usize,
) -> (i32, i32) {
    let (sub_x, sub_y) = bd.plane_sub(plane);
    let row = ((start_y as u32) << sub_y) >> 2;
    let col = ((start_x as u32) << sub_x) >> 2;
    let base_row = ((row & 15) >> sub_y) as i32;
    let base_col = ((col & 15) >> sub_x) as i32;
    (base_row, base_col)
}

/// §7.11.2.1 `haveAboveRight` / `haveBelowLeft` for a `tx_w × tx_h`
/// TU — the §5.11.35 `BlockDecoded[]` border reads.
#[inline]
pub(crate) fn tu_corner_avail(
    bd: &BlockDecodedMirror,
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx_w: usize,
    tx_h: usize,
) -> (bool, bool) {
    let (base_row, base_col) = tu_bd_anchor(bd, plane, start_x, start_y);
    let step_x4 = (tx_w >> 2) as i32;
    let step_y4 = (tx_h >> 2) as i32;
    let above_right = bd.get(plane, base_row - 1, base_col + step_x4);
    let below_left = bd.get(plane, base_row + step_y4, base_col - 1);
    (above_right, below_left)
}

/// §5.11.35 per-TU `BlockDecoded[]` stamp over the TU footprint.
#[inline]
pub(crate) fn tu_bd_stamp(
    bd: &mut BlockDecodedMirror,
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx_w: usize,
    tx_h: usize,
) {
    let (base_row, base_col) = tu_bd_anchor(bd, plane, start_x, start_y);
    let step_x4 = (tx_w >> 2) as i32;
    let step_y4 = (tx_h >> 2) as i32;
    for i in 0..step_y4 {
        for j in 0..step_x4 {
            bd.set(plane, base_row + i, base_col + j);
        }
    }
}

/// Encoder-side running reconstruction + quantiser bundle. r427:
/// planes are `u16` at [`Self::bit_depth`], chroma at the
/// [`Self::subsampling_x`] / [`Self::subsampling_y`] extents (empty
/// when [`Self::num_planes`] is 1).
pub(crate) struct ReconState {
    pub(crate) y: Vec<u16>,
    pub(crate) u: Vec<u16>,
    pub(crate) v: Vec<u16>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) chroma_w: usize,
    pub(crate) chroma_h: usize,
    /// §5.5.2 `BitDepth` (8 / 10 / 12).
    pub(crate) bit_depth: u8,
    /// §5.5.2 `subsampling_x` (0 or 1).
    pub(crate) subsampling_x: u8,
    /// §5.5.2 `subsampling_y`.
    pub(crate) subsampling_y: u8,
    /// §5.5.2 `NumPlanes` (1 or 3).
    pub(crate) num_planes: u8,
    pub(crate) mi_rows: u32,
    pub(crate) mi_cols: u32,
    pub(crate) lossless: bool,
    /// §5.9.2 `allow_screen_content_tools` for the frame this state
    /// reconstructs — gates the §5.11.46 palette election in
    /// [`encode_leaf_sq`] (the write side rejects palette commitments
    /// when the frame-header gate is closed, so the search must not
    /// build them).
    pub(crate) allow_screen_content_tools: bool,
    /// §5.9.20 `allow_intrabc` for the frame — gates the §5.11.7
    /// intra-block-copy election in [`encode_leaf_sq`]. Only ever
    /// `true` on intra frames (the §5.9.20 read is intra-only).
    pub(crate) allow_intrabc: bool,
    pub(crate) qp: QuantizerParams,
    pub(crate) bd: BlockDecodedMirror,
    /// r425 — per-frame block-hash index over the §6.10.24-reachable
    /// intra-block-copy source region (see
    /// [`crate::encoder::dv_hash`]). Inert (`Default`) unless the KEY
    /// driver armed it alongside `allow_intrabc`; maintained once per
    /// superblock via [`Self::advance_dv_hash`].
    pub(crate) dv_hash: crate::encoder::dv_hash::DvHashIndex,
}

impl ReconState {
    pub(crate) fn plane(&self, plane: usize) -> (&[u16], usize, usize) {
        match plane {
            0 => (&self.y, self.width, self.height),
            1 => (&self.u, self.chroma_w, self.chroma_h),
            _ => (&self.v, self.chroma_w, self.chroma_h),
        }
    }

    /// r427 — chroma-space origin of the mi cell `(r, c)` per the
    /// §5.11.5 `(MiRow >> subsampling_y, MiCol >> subsampling_x)`
    /// derivation, in samples.
    #[inline]
    pub(crate) fn chroma_origin(&self, r: u32, c: u32) -> (usize, usize) {
        (
            ((r >> self.subsampling_y) as usize) * 4,
            ((c >> self.subsampling_x) as usize) * 4,
        )
    }
}

// ---------------------------------------------------------------------
// §7.11.2 exact-mirror intra prediction (r410).
// ---------------------------------------------------------------------

/// §7.11.2.1 neighbour-array derivation against a `u16` running plane
/// — the encoder twin of the decode walker's `AboveRow[]` / `LeftCol[]`
/// build (head-extended representation: spec index `k` at offset
/// `k + 2`, the `-1` corner at offset 1). `have_above` / `have_left`
/// are the §5.11.35 `(AvailU || y > 0)` / `(AvailL || x > 0)` values,
/// which for a single-tile whole-frame walk collapse to `start_y > 0`
/// / `start_x > 0`. r427: the no-neighbour fills are the §7.11.2.2
/// depth-scaled `(1 << (BitDepth - 1)) ± 1` constants.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn build_tu_neighbours(
    plane_buf: &[u16],
    pw: usize,
    ph: usize,
    start_x: usize,
    start_y: usize,
    w: usize,
    h: usize,
    have_above_right: bool,
    have_below_left: bool,
    bit_depth: u8,
) -> (Vec<u16>, Vec<u16>, bool, bool) {
    let have_above = start_y > 0;
    let have_left = start_x > 0;
    let max_x = pw - 1;
    let max_y = ph - 1;
    let read = |yy: usize, xx: usize| -> u16 { plane_buf[yy.min(max_y) * pw + xx.min(max_x)] };
    let half: u16 = 1 << (bit_depth - 1);
    let corner: u16 = if have_above && have_left {
        read(start_y - 1, start_x - 1)
    } else if have_above {
        read(start_y - 1, start_x)
    } else if have_left {
        read(start_y, start_x - 1)
    } else {
        half
    };
    let span = w + h;
    let ext_cap = 2 * span + 2;
    let mut above_ext = vec![0u16; ext_cap];
    above_ext[0] = corner;
    above_ext[1] = corner;
    if !have_above && have_left {
        let v = read(start_y, start_x - 1);
        for slot in above_ext.iter_mut().skip(2).take(span) {
            *slot = v;
        }
    } else if !have_above && !have_left {
        let v = half - 1;
        for slot in above_ext.iter_mut().skip(2).take(span) {
            *slot = v;
        }
    } else {
        let above_limit = max_x.min(start_x + (if have_above_right { 2 * w } else { w }) - 1);
        for (i, slot) in above_ext.iter_mut().skip(2).take(span).enumerate() {
            *slot = read(start_y - 1, above_limit.min(start_x + i));
        }
    }
    let mut left_ext = vec![0u16; ext_cap];
    left_ext[0] = corner;
    left_ext[1] = corner;
    if !have_left && have_above {
        let v = read(start_y - 1, start_x);
        for slot in left_ext.iter_mut().skip(2).take(span) {
            *slot = v;
        }
    } else if !have_left && !have_above {
        let v = half + 1;
        for slot in left_ext.iter_mut().skip(2).take(span) {
            *slot = v;
        }
    } else {
        let left_limit = max_y.min(start_y + (if have_below_left { 2 * h } else { h }) - 1);
        for (i, slot) in left_ext.iter_mut().skip(2).take(span).enumerate() {
            *slot = read(left_limit.min(start_y + i), start_x - 1);
        }
    }
    (above_ext, left_ext, have_above, have_left)
}

/// §7.11.2 mode dispatch over pre-built head-extended neighbour
/// arrays — the same kernel routing the decode walker performs for
/// this driver's header set (`enable_intra_edge_filter = 0`,
/// `use_filter_intra = 0` ⇒ no §7.11.2.4 step-4 pre-pass,
/// `upsampleAbove = upsampleLeft = 0`). `angle_delta` (r410, in
/// `-3..=3`) shifts the §7.11.2.4 projection angle by
/// `ANGLE_STEP * delta` for the directional modes; a V_PRED / H_PRED
/// with a non-zero delta becomes fully directional per §7.11.2.1
/// (only the exact 90°/180° cases take the plain copies).
#[allow(clippy::too_many_arguments)]
fn predict_mode_from_neighbours(
    mode: usize,
    angle_delta: i32,
    w: usize,
    h: usize,
    above_ext: &[u16],
    left_ext: &[u16],
    have_above: bool,
    have_left: bool,
    bit_depth: u8,
) -> Option<Vec<u16>> {
    let span = w + h;
    let above_row = &above_ext[2..2 + span];
    let left_col = &left_ext[2..2 + span];
    let corner = above_ext[1];
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    let mut pred = vec![0u16; w * h];
    let ok = match mode {
        m if m == DC_PRED => predict_intra_dc_pred(
            u8::from(have_left),
            u8::from(have_above),
            log2_w,
            log2_h,
            w,
            h,
            bit_depth,
            above_row,
            left_col,
            &mut pred,
        )
        .is_ok(),
        m if m == V_PRED && angle_delta == 0 => {
            predict_intra_v_pred(w, h, above_row, &mut pred).is_ok()
        }
        m if m == H_PRED && angle_delta == 0 => {
            predict_intra_h_pred(w, h, left_col, &mut pred).is_ok()
        }
        m if (V_PRED..=D67_PRED).contains(&m) => predict_intra_directional(
            w,
            h,
            MODE_TO_ANGLE[m] + angle_delta * ANGLE_STEP,
            0,
            0,
            above_ext,
            left_ext,
            &mut pred,
        )
        .is_ok(),
        m if m == SMOOTH_PRED => {
            predict_intra_smooth_pred(log2_w, log2_h, w, h, above_row, left_col, &mut pred).is_ok()
        }
        m if m == SMOOTH_V_PRED => {
            predict_intra_smooth_v_pred(log2_h, w, h, above_row, left_col, &mut pred).is_ok()
        }
        m if m == SMOOTH_H_PRED => {
            predict_intra_smooth_h_pred(log2_w, w, h, above_row, left_col, &mut pred).is_ok()
        }
        m if m == PAETH_PRED => {
            predict_intra_paeth_pred(w, h, above_row, left_col, corner, &mut pred).is_ok()
        }
        _ => false,
    };
    if !ok {
        return None;
    }
    Some(pred)
}

/// §7.11.2.3 recursive (filter-intra) prediction over pre-built
/// head-extended neighbour arrays — the `use_filter_intra == 1` luma
/// arm of the §7.11.2.1 dispatch (replaces the mode-driven kernels).
pub(crate) fn predict_filter_intra_from_neighbours(
    fim: usize,
    w: usize,
    h: usize,
    above_ext: &[u16],
    left_ext: &[u16],
    bit_depth: u8,
) -> Option<Vec<u16>> {
    let mut pred = vec![0u16; w * h];
    crate::cdf::predict_intra_recursive(w, h, fim, bit_depth, above_ext, left_ext, &mut pred)
        .ok()?;
    Some(pred)
}

/// One-TU §7.11.2 prediction from the running plane: neighbour build
/// (with real `BlockDecoded[]` corner availability) + kernel dispatch.
/// `fim = Some(mode)` (luma only) routes through the §7.11.2.3
/// recursive process instead of the mode-driven kernels.
#[allow(clippy::too_many_arguments)]
fn predict_tu(
    recon: &ReconState,
    plane: usize,
    start_x: usize,
    start_y: usize,
    w: usize,
    h: usize,
    mode: usize,
    angle_delta: i32,
    fim: Option<usize>,
) -> Vec<u16> {
    let (buf, pw, ph) = recon.plane(plane);
    let (avail_ar, avail_bl) = tu_corner_avail(&recon.bd, plane, start_x, start_y, w, h);
    let (above_ext, left_ext, have_above, have_left) = build_tu_neighbours(
        buf,
        pw,
        ph,
        start_x,
        start_y,
        w,
        h,
        avail_ar,
        avail_bl,
        recon.bit_depth,
    );
    if let Some(f) = fim.filter(|_| plane == 0) {
        return predict_filter_intra_from_neighbours(
            f,
            w,
            h,
            &above_ext,
            &left_ext,
            recon.bit_depth,
        )
        .expect("filter-intra kernel domain holds for coded TU sizes");
    }
    predict_mode_from_neighbours(
        mode,
        angle_delta,
        w,
        h,
        &above_ext,
        &left_ext,
        have_above,
        have_left,
        recon.bit_depth,
    )
    .expect("supported intra mode always predicts")
}

/// §3 `Round2Signed(x, n)`.
#[inline]
fn round2_signed(x: i64, n: u32) -> i64 {
    let half: i64 = 1i64 << (n - 1);
    if x < 0 {
        -(((-x) + half) >> n)
    } else {
        (x + half) >> n
    }
}

/// §7.11.5 predict-chroma-from-luma — layer the CFL AC contribution
/// onto an already-computed `DC_PRED` base for one `w × h` chroma TU
/// at chroma-space `(start_x, start_y)`. `max_luma_w` / `max_luma_h`
/// are the §5.11.35 `MaxLumaW` / `MaxLumaH` extents (the current
/// block's luma right/bottom edge — the last luma TU coded before the
/// chroma TUs). r427: general `(subX, subY)` — the §7.11.5 `L[]`
/// build sums the `(1 + subY) × (1 + subX)` collocated luma cell and
/// scales by `t << (3 - subX - subY)`; the output clamp is
/// `Clip1(BitDepth)`.
#[allow(clippy::too_many_arguments)]
fn cfl_layer(
    dc_pred: &[u16],
    recon_y: &[u16],
    luma_w: usize,
    start_x: usize,
    start_y: usize,
    w: usize,
    h: usize,
    alpha: i8,
    max_luma_w: usize,
    max_luma_h: usize,
    sub_x: u8,
    sub_y: u8,
    bit_depth: u8,
) -> Vec<u16> {
    let (sub_x, sub_y) = (u32::from(sub_x), u32::from(sub_y));
    // §7.11.5: `lumaX = Min( lumaX, MaxLumaW - (1 << subX) )` (and the
    // Y twin).
    let clamp_x = max_luma_w.saturating_sub(1 << sub_x);
    let clamp_y = max_luma_h.saturating_sub(1 << sub_y);
    let mut l = vec![0i64; w * h];
    let mut luma_sum: i64 = 0;
    for i in 0..h {
        let luma_y = ((start_y + i) << sub_y).min(clamp_y);
        for j in 0..w {
            let luma_x = ((start_x + j) << sub_x).min(clamp_x);
            let mut t = 0i64;
            for dy in 0..=sub_y as usize {
                for dx in 0..=sub_x as usize {
                    t += i64::from(recon_y[(luma_y + dy) * luma_w + luma_x + dx]);
                }
            }
            let v = t << (3 - sub_x - sub_y);
            l[i * w + j] = v;
            luma_sum += v;
        }
    }
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    let luma_avg = round2_signed(luma_sum, log2_w + log2_h);
    let max_val = (1i64 << bit_depth) - 1;
    let mut out = vec![0u16; w * h];
    for k in 0..w * h {
        let scaled = round2_signed((alpha as i64) * (l[k] - luma_avg), 6);
        out[k] = (i64::from(dc_pred[k]) + scaled).clamp(0, max_val) as u16;
    }
    out
}

// ---------------------------------------------------------------------
// Residual pipeline (any square TX size).
// ---------------------------------------------------------------------

/// §7.12.3 / §9.2: 64-dim transforms code only the top-left
/// `Min(32, w) × Min(32, h)` coefficients, addressed with the COMPACT
/// `tw`-stride layout. Repack the dense forward-quantize output; the
/// padding tail stays zero (never scanned). Sizes ≤ 32 pass through.
pub(crate) fn repack_compact(dense: Vec<i32>, w: usize, h: usize) -> Vec<i32> {
    let tw = w.min(32);
    let th = h.min(32);
    if tw == w && th == h {
        return dense;
    }
    let mut q = vec![0i32; w * h];
    for i in 0..th {
        for j in 0..tw {
            q[i * tw + j] = dense[i * w + j];
        }
    }
    q
}

/// §7.12.3 step-3 `flipUD` / `flipLR` destination-remap pair for a
/// `TxType` — the FLIPADST-family selectors (av1-spec lines
/// 16292-16296). Both false for every non-FLIPADST type (the whole
/// intra set — the remap below is then the identity).
#[inline]
pub(crate) fn step3_flips(tx_type: usize) -> (bool, bool) {
    use crate::cdf::{
        ADST_FLIPADST, DCT_FLIPADST, FLIPADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST, H_FLIPADST,
        V_FLIPADST,
    };
    let flip_ud = matches!(
        tx_type,
        FLIPADST_DCT | FLIPADST_ADST | V_FLIPADST | FLIPADST_FLIPADST
    );
    let flip_lr = matches!(
        tx_type,
        DCT_FLIPADST | ADST_FLIPADST | H_FLIPADST | FLIPADST_FLIPADST
    );
    (flip_ud, flip_lr)
}

/// One residual leg at `tx_sz`: forward (WHT for the lossless TX_4X4 /
/// DCT-family otherwise) + quantize, then the decoder's dequant +
/// inverse, stitching `Clip1(pred + res)` into the running plane with
/// the §7.12.3 step-3 `flipUD` / `flipLR` destination remap (the
/// FLIPADST family reaches this through the r411 inter chroma
/// inheritance; every intra-set type remaps as the identity).
/// Returns the committed `Quant[]` in the §5.11.39 coefficient layout
/// (the §7.12.3 compact-`tw` stride for 64-wide transforms, dense
/// row-major otherwise), zero-padded to `Tx_Width * Tx_Height`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_tx(
    input_plane: &[u16],
    recon_plane: &mut [u16],
    pw: usize,
    row0: usize,
    col0: usize,
    tx_sz: usize,
    pred: &[u16],
    plane: u8,
    lossless: bool,
    tx_type: usize,
    qp: &QuantizerParams,
) -> Vec<i32> {
    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    residual_tx_avail(
        input_plane,
        recon_plane,
        pw,
        row0,
        col0,
        tx_sz,
        pred,
        plane,
        lossless,
        tx_type,
        qp,
        w,
        h,
    )
}

/// r425 — clip-aware twin of [`residual_tx`] for TUs of a
/// frame-edge-straddling block that extend past the plane: only the
/// `avail_w × avail_h` on-screen sub-rectangle reads input samples
/// (the off-screen residual is coded as zero — the decoder discards
/// those samples, so the choice is free) and only on-screen samples
/// are stitched back. Fully-on-screen TUs (`avail >= (w, h)`) take
/// the exact pre-r425 path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_tx_avail(
    input_plane: &[u16],
    recon_plane: &mut [u16],
    pw: usize,
    row0: usize,
    col0: usize,
    tx_sz: usize,
    pred: &[u16],
    plane: u8,
    lossless: bool,
    tx_type: usize,
    qp: &QuantizerParams,
    avail_w: usize,
    avail_h: usize,
) -> Vec<i32> {
    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    let (aw, ah) = (avail_w.min(w), avail_h.min(h));
    let mut residual = vec![0i64; w * h];
    for i in 0..ah {
        for j in 0..aw {
            residual[i * w + j] =
                i64::from(input_plane[(row0 + i) * pw + (col0 + j)]) - i64::from(pred[i * w + j]);
        }
    }
    let coeffs = if lossless {
        debug_assert_eq!(tx_sz, TX_4X4);
        let mut r16 = [0i64; 16];
        r16.copy_from_slice(&residual);
        forward_wht_4x4(&r16).to_vec()
    } else {
        forward_transform_2d(&residual, tx_sz, tx_type, false)
    };
    let dense = forward_quantize(&coeffs, tx_sz, plane, 0, tx_type, 15, qp);
    let quant = repack_compact(dense, w, h);
    let dequant = dequantize_step1(&quant, tx_sz, plane, 0, tx_type, 15, qp);
    let res_back =
        inverse_transform_2d(&dequant, tx_sz, tx_type, u32::from(qp.bit_depth), lossless);
    let (flip_ud, flip_lr) = step3_flips(tx_type);
    let max_val = (1i64 << qp.bit_depth) - 1;
    for i in 0..h {
        let yy = if flip_ud { h - 1 - i } else { i };
        for j in 0..w {
            let xx = if flip_lr { w - 1 - j } else { j };
            if yy >= ah || xx >= aw {
                continue;
            }
            let p = i64::from(pred[yy * w + xx]) + res_back[i * w + j];
            recon_plane[(row0 + yy) * pw + (col0 + xx)] = p.clamp(0, max_val) as u16;
        }
    }
    quant
}

/// r410 — §5.11.47 per-TU LUMA transform-type RD search (lossy arm).
/// Trials every `TxType` admissible in the §5.11.48 intra set for
/// `tx_sz` (`TX_SET_INTRA_1`'s 7 types at 4×4/8×8, `TX_SET_INTRA_2`'s
/// 5 at 16×16, `DCT_DCT` alone at 32×32+), scoring each full
/// quantise→dequantise→inverse chain by `D + λ·R` over the TU, then
/// stitches the winner into the running plane. Returns the committed
/// `Quant[]` plus the `TxType` label — forced to `DCT_DCT` when the
/// winning TU quantises to all-zero (the §5.11.39 `all_zero` arm reads
/// no `transform_type` symbol and the walker stamps `DCT_DCT`).
///
/// r424 — with `fork = Some((tu_fork, tu_ctx))` each candidate's rate
/// is the EXACT §5.11.39 coefficient chain (the `all_zero` symbol at
/// its true neighbour context, the `intra_tx_type` S() against the
/// current adaptive CDFs at the leaf's §8.3.2 `intra_dir` axis, and
/// the full coefficient tail) priced through the writer's own one-TU
/// body, and the winner is committed into the running fork; `None`
/// keeps the magnitude proxy.
/// r425 — the search is clip-aware (see [`residual_tx_avail`]):
/// input reads, distortion and stitching stay inside the
/// `avail_w × avail_h` on-screen sub-rectangle; the coded off-screen
/// residual is zero. Fully-on-screen TUs pass the full TU extent.
#[allow(clippy::too_many_arguments)]
fn residual_tx_search_luma_avail(
    input_plane: &[u16],
    recon_plane: &mut [u16],
    pw: usize,
    row0: usize,
    col0: usize,
    tx_sz: usize,
    pred: &[u16],
    qp: &QuantizerParams,
    fork: Option<(&mut TuFork, &TuCtx<'_>)>,
    avail_w: usize,
    avail_h: usize,
) -> Result<(Vec<i32>, u8), Error> {
    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    let (aw, ah) = (avail_w.min(w), avail_h.min(h));
    let max_val = (1i64 << qp.bit_depth) - 1;
    let mut residual = vec![0i64; w * h];
    for i in 0..ah {
        for j in 0..aw {
            residual[i * w + j] =
                i64::from(input_plane[(row0 + i) * pw + (col0 + j)]) - i64::from(pred[i * w + j]);
        }
    }
    let set = intra_tx_type_set(
        tx_size_sqr_index(tx_sz) as u32,
        TX_SIZE_SQR_UP[tx_sz] as u32,
        false,
    );
    let lambda = lambda_for(qp);
    let fork_xy = fork.as_ref().map(|(_, ctx)| {
        (
            (col0 as u32 - ctx.base_x) / 4,
            (row0 as u32 - ctx.base_y) / 4,
        )
    });
    let mut best: Option<(Vec<i32>, Vec<i64>, u8, u64)> = None;
    for t in 0..crate::cdf::TX_TYPES {
        let admissible = t == DCT_DCT || (set > 0 && is_tx_type_in_set(false, set, t));
        if !admissible {
            continue;
        }
        let coeffs = forward_transform_2d(&residual, tx_sz, t, false);
        let quant = repack_compact(forward_quantize(&coeffs, tx_sz, 0, 0, t, 15, qp), w, h);
        let all_zero = quant.iter().all(|&q| q == 0);
        let dequant = dequantize_step1(&quant, tx_sz, 0, 0, t, 15, qp);
        let res_back = inverse_transform_2d(&dequant, tx_sz, t, u32::from(qp.bit_depth), false);
        let mut d = 0u64;
        for i in 0..ah {
            for j in 0..aw {
                let rec = (i64::from(pred[i * w + j]) + res_back[i * w + j]).clamp(0, max_val);
                let diff = i64::from(input_plane[(row0 + i) * pw + (col0 + j)]) - rec;
                d += (diff * diff) as u64;
            }
        }
        // §5.11.39: an all-zero TU codes no transform_type symbol and
        // the walker stamps DCT_DCT — the label must follow.
        let label = if all_zero { DCT_DCT as u8 } else { t as u8 };
        // r424 — exact §5.11.39 chain bits through the running fork,
        // or the pre-r424 magnitude proxy. One scale per call.
        let score = match (&fork, fork_xy) {
            (Some((tu_fork, ctx)), Some((fx, fy))) => {
                let bits256 = tu_fork.price_luma_tu(ctx, tx_sz, fx, fy, &quant, label)?;
                score256(d, lambda, bits256)
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
        // The DCT_DCT trial quantising to all-zero means the residual
        // is below the quantisation floor — pred-exact reconstruction;
        // skip the remaining types (they would commit the same
        // all-zero DCT_DCT shape at best-marginal gains).
        if t == DCT_DCT && all_zero {
            break;
        }
    }
    let (quant, res_back, label, _) = best.expect("DCT_DCT is always admissible");
    if let (Some((tu_fork, ctx)), Some((fx, fy))) = (fork, fork_xy) {
        tu_fork.commit_luma_tu(ctx, tx_sz, fx, fy, &quant, label)?;
    }
    for i in 0..ah {
        for j in 0..aw {
            let p = i64::from(pred[i * w + j]) + res_back[i * w + j];
            recon_plane[(row0 + i) * pw + (col0 + j)] = p.clamp(0, max_val) as u16;
        }
    }
    Ok((quant, label))
}

// ---------------------------------------------------------------------
// Mode pickers.
// ---------------------------------------------------------------------

/// §5.11.42/§5.11.43 angle-delta candidate range for a mode: the
/// directional modes (`V_PRED..=D67_PRED`) search the full `-3..=3`
/// span when the block is `>= BLOCK_8X8` (below that no angle symbol
/// is coded and the delta is spec-forced to 0); everything else is 0.
fn angle_delta_candidates(mode: usize, n: usize) -> core::ops::RangeInclusive<i32> {
    if (V_PRED..=D67_PRED).contains(&mode) && n >= 8 {
        -3..=3
    } else {
        0..=0
    }
}

/// SSD-minimising luma (mode, angle_delta, filter_intra) pick — ALL
/// 13 §6.10.x intra modes, the §5.11.42 `-3..=3` angle-delta span for
/// the directional ones, and (r410) the five §7.11.2.3 filter-intra
/// modes on §5.11.24-eligible blocks (`Max(w, h) <= 32`; a
/// filter-intra win codes `y_mode = DC_PRED` + `use_filter_intra`) —
/// for one `bw × bh` block (recon-neighbour prediction at whole-block
/// extent, input-target SSD; r425: rectangular shapes ride the same
/// picker). The neighbour build uses the same `BlockDecoded[]` state
/// the block's first luma TU will observe.
fn pick_y_mode(
    recon: &ReconState,
    input: &YuvFrame,
    row0: usize,
    col0: usize,
    bw: usize,
    bh: usize,
) -> (u8, i8, Option<u8>) {
    // r425 — frame-edge-straddling blocks score on the on-screen
    // sub-rectangle only (the off-screen samples are discarded by the
    // decoder and carry no distortion).
    let (aw, ah) = (bw.min(recon.width - col0), bh.min(recon.height - row0));
    let (avail_ar, avail_bl) = tu_corner_avail(&recon.bd, 0, col0, row0, bw, bh);
    let (above_ext, left_ext, have_above, have_left) = build_tu_neighbours(
        &recon.y,
        recon.width,
        recon.height,
        col0,
        row0,
        bw,
        bh,
        avail_ar,
        avail_bl,
        recon.bit_depth,
    );
    let ssd_of = |pred: &[u16]| -> u64 {
        let mut ssd = 0u64;
        for i in 0..ah {
            for j in 0..aw {
                let d = i64::from(input.y[(row0 + i) * recon.width + (col0 + j)])
                    - i64::from(pred[i * bw + j]);
                ssd += (d * d) as u64;
            }
        }
        ssd
    };
    let mut best = (DC_PRED as u8, 0i8, None);
    let mut best_ssd = u64::MAX;
    for mode in 0..INTRA_MODES {
        for delta in angle_delta_candidates(mode, bw.min(bh)) {
            let Some(pred) = predict_mode_from_neighbours(
                mode,
                delta,
                bw,
                bh,
                &above_ext,
                &left_ext,
                have_above,
                have_left,
                recon.bit_depth,
            ) else {
                continue;
            };
            let ssd = ssd_of(&pred);
            if ssd < best_ssd {
                best_ssd = ssd;
                best = (mode as u8, delta as i8, None);
            }
        }
    }
    // §5.11.24 gate: Max(Block_Width, Block_Height) <= 32 (the coded
    // y_mode is DC_PRED and this driver never codes palette).
    if bw.max(bh) <= 32 {
        for fim in 0..crate::cdf::INTRA_FILTER_MODES {
            let Some(pred) = predict_filter_intra_from_neighbours(
                fim,
                bw,
                bh,
                &above_ext,
                &left_ext,
                recon.bit_depth,
            ) else {
                continue;
            };
            let ssd = ssd_of(&pred);
            if ssd < best_ssd {
                best_ssd = ssd;
                best = (DC_PRED as u8, 0, Some(fim as u8));
            }
        }
    }
    best
}

/// Joint U+V picker over the 13 §6.10.x modes plus the §7.11.5.3
/// `UV_CFL_PRED` (αU, αV) grid (when `cfl_allowed`) for one `cn × cn`
/// chroma block. One shared `uv_mode` per §5.11.22.
#[allow(clippy::too_many_arguments)]
fn pick_uv_mode(
    recon: &ReconState,
    input: &YuvFrame,
    crow0: usize,
    ccol0: usize,
    cbw: usize,
    cbh: usize,
    cfl_allowed: bool,
    // §5.11.43 gate operand: the block's LUMA extent (`Block_Width[
    // MiSize ]`) — angle deltas are coded for `MiSize >= BLOCK_8X8`
    // regardless of the subsampled chroma extent.
    luma_n: usize,
    max_luma_w: usize,
    max_luma_h: usize,
) -> (u8, i8, Option<(i8, i8)>) {
    let pw = recon.chroma_w;
    let (ar_u, bl_u) = tu_corner_avail(&recon.bd, 1, ccol0, crow0, cbw, cbh);
    let (above_u, left_u, ha_u, hl_u) = build_tu_neighbours(
        &recon.u,
        recon.chroma_w,
        recon.chroma_h,
        ccol0,
        crow0,
        cbw,
        cbh,
        ar_u,
        bl_u,
        recon.bit_depth,
    );
    let (ar_v, bl_v) = tu_corner_avail(&recon.bd, 2, ccol0, crow0, cbw, cbh);
    let (above_v, left_v, ha_v, hl_v) = build_tu_neighbours(
        &recon.v,
        recon.chroma_w,
        recon.chroma_h,
        ccol0,
        crow0,
        cbw,
        cbh,
        ar_v,
        bl_v,
        recon.bit_depth,
    );
    // r425 — clipped blocks score on-screen chroma only.
    let (caw, cah) = (
        cbw.min(recon.chroma_w - ccol0),
        cbh.min(recon.chroma_h - crow0),
    );
    let ssd_uv = |pred_u: &[u16], pred_v: &[u16]| -> u64 {
        let mut ssd = 0u64;
        for i in 0..cah {
            for j in 0..caw {
                let idx = (crow0 + i) * pw + (ccol0 + j);
                let du = i64::from(input.u[idx]) - i64::from(pred_u[i * cbw + j]);
                let dv = i64::from(input.v[idx]) - i64::from(pred_v[i * cbw + j]);
                ssd += (du * du + dv * dv) as u64;
            }
        }
        ssd
    };
    let mut best_mode = DC_PRED as u8;
    let mut best_delta = 0i8;
    let mut best_alpha: Option<(i8, i8)> = None;
    let mut best_ssd = u64::MAX;
    let mut dc_pred_u: Vec<u16> = Vec::new();
    let mut dc_pred_v: Vec<u16> = Vec::new();
    for mode in 0..INTRA_MODES {
        for delta in angle_delta_candidates(mode, luma_n) {
            let Some(pred_u) = predict_mode_from_neighbours(
                mode,
                delta,
                cbw,
                cbh,
                &above_u,
                &left_u,
                ha_u,
                hl_u,
                recon.bit_depth,
            ) else {
                continue;
            };
            let Some(pred_v) = predict_mode_from_neighbours(
                mode,
                delta,
                cbw,
                cbh,
                &above_v,
                &left_v,
                ha_v,
                hl_v,
                recon.bit_depth,
            ) else {
                continue;
            };
            let ssd = ssd_uv(&pred_u, &pred_v);
            if mode == DC_PRED {
                dc_pred_u = pred_u;
                dc_pred_v = pred_v;
            }
            if ssd < best_ssd {
                best_ssd = ssd;
                best_mode = mode as u8;
                best_delta = delta as i8;
                best_alpha = None;
            }
        }
    }
    if cfl_allowed {
        for &(au, av) in CFL_ALPHA_CANDIDATES {
            let pred_u = cfl_layer(
                &dc_pred_u,
                &recon.y,
                recon.width,
                ccol0,
                crow0,
                cbw,
                cbh,
                au,
                max_luma_w,
                max_luma_h,
                recon.subsampling_x,
                recon.subsampling_y,
                recon.bit_depth,
            );
            let pred_v = cfl_layer(
                &dc_pred_v,
                &recon.y,
                recon.width,
                ccol0,
                crow0,
                cbw,
                cbh,
                av,
                max_luma_w,
                max_luma_h,
                recon.subsampling_x,
                recon.subsampling_y,
                recon.bit_depth,
            );
            let ssd = ssd_uv(&pred_u, &pred_v);
            if ssd < best_ssd {
                best_ssd = ssd;
                best_mode = UV_CFL_PRED as u8;
                best_delta = 0;
                best_alpha = Some((au, av));
            }
        }
    }
    (best_mode, best_delta, best_alpha)
}

// ---------------------------------------------------------------------
// Leaf encoder (any square block size).
// ---------------------------------------------------------------------

/// §5.11.40 `compute_tx_type` chroma-intra arm: `Mode_To_Txfm[
/// UVMode ]` filtered by the §5.11.48 intra set at the chroma TU size
/// (DCT_DCT fallback when out of set; lossless short-circuits before
/// the table).
fn chroma_tx_type_for(uv_mode: u8, chroma_tx: usize, lossless: bool) -> usize {
    if lossless {
        return DCT_DCT;
    }
    if TX_SIZE_SQR_UP[chroma_tx] > crate::cdf::TX_32X32 {
        return DCT_DCT;
    }
    let t = MODE_TO_TXFM
        .get(uv_mode as usize)
        .copied()
        .unwrap_or(DCT_DCT);
    let set = intra_tx_type_set(
        tx_size_sqr_index(chroma_tx) as u32,
        TX_SIZE_SQR_UP[chroma_tx] as u32,
        false,
    );
    if is_tx_type_in_set(false, set, t) {
        t
    } else {
        DCT_DCT
    }
}

/// Encode one in-frame square leaf at `b_size` (`BLOCK_4X4` …
/// `BLOCK_64X64`). Luma: mode picked on a whole-block prediction,
/// then coded as the §5.11.34 TU grid (`TX_4X4` fan-out on the
/// lossless arm — each TU re-predicted from the running
/// reconstruction with the block mode, exactly like the decode walk —
/// or one `Max_Tx_Size_Rect` TU on the lossy `TX_MODE_LARGEST` arm).
/// Chroma on §5.11.5 `HasChroma` leaves: same shape at the §5.11.38
/// `get_tx_size` chroma TU size, coded after the luma TUs per
/// §5.11.34 plane order (all U TUs, then all V TUs).
pub(crate) fn encode_leaf_sq(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    // r421 — `Some((twin, params))` prices the leaf-level elections
    // (tx-depth ladder, §5.11.46 palette combos, §5.11.7 intra-bc)
    // with exact twin bits; `None` keeps the pre-r421 heuristics.
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    encode_leaf_sq_seg(mi_r, mi_c, b_size, input, recon, pricing, None)
}

/// r426 — [`encode_leaf_sq`] with a committed segment id: every
/// candidate leaf carries `segment_override` BEFORE it is priced
/// (skip leaves on segmented INTER frames still take the §5.11.19
/// bit-silent pred cascade afterwards), so the twin prices the
/// §5.11.9 segment symbols and the per-segment
/// `Lossless`/quantiser-derived syntax shape exactly as the write
/// pass will emit them. The caller owns the matching `recon.qp` /
/// `recon.lossless` configuration for the override's segment.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_leaf_sq_seg(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
    segment_override: Option<u8>,
) -> Result<SyntaxBlock, Error> {
    // §5.11.46 palette candidates (r418) — only built when the
    // frame-header gate is open and the block is an eligible square.
    // r424: with twin pricing available the k-means proxy ladder
    // surfaces its top TWO per-`k` palettes per plane and the exact
    // full-leaf price (§5.11.46 entries + §5.11.49 tokens + the
    // palette-predicted §5.11.39 residual chain) settles `k`; the
    // heuristic path keeps the single proxy pick.
    let max_pal = if pricing.is_some() { 2 } else { 1 };
    let (pal_y_list, pal_uv_list) = if recon.allow_screen_content_tools {
        (
            palette_candidates_y(input, recon, mi_r, mi_c, b_size, max_pal),
            palette_candidates_uv(input, recon, mi_r, mi_c, b_size, max_pal),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    // §5.11.15: lossless forces TX_4X4; a BLOCK_4X4 block has no
    // tx_depth choice. Otherwise step the luma TX down from
    // `Max_Tx_Size_Rect[MiSize]` via `Split_Tx_Size` (one step for
    // BLOCK_8X8 — its tx_depth alphabet is 2-valued — two for larger
    // squares) and trial-encode the leaf at each size against the
    // same starting state, keeping the lower `D + λ·R`.
    let single_shape = recon.lossless || b_size == BLOCK_4X4;
    let cands: Vec<usize> = if single_shape {
        vec![if recon.lossless {
            TX_4X4
        } else {
            MAX_TX_SIZE_RECT[b_size]
        }]
    } else {
        let max_steps = if b_size == BLOCK_8X8 { 1 } else { MAX_TX_DEPTH };
        let mut v = vec![MAX_TX_SIZE_RECT[b_size]];
        for _ in 0..max_steps {
            let next = SPLIT_TX_SIZE[*v.last().expect("non-empty")];
            v.push(next);
        }
        v
    };
    // r418: per-TX-shape §5.11.46 palette combinations — every
    // available candidate arm (luma / chroma / both) trials against
    // the plain intra leaf over the same starting state (r424: per-`k`
    // candidates fan out the arms under twin pricing).
    let mut combos: Vec<(Option<&PaletteCandY>, Option<&PaletteCandUv>)> = vec![(None, None)];
    for p in &pal_y_list {
        combos.push((Some(p), None));
    }
    for p in &pal_uv_list {
        combos.push((None, Some(p)));
    }
    for a in &pal_y_list {
        for b in &pal_uv_list {
            combos.push((Some(a), Some(b)));
        }
    }
    // r418 §5.11.7 intra-block-copy candidate — ranked on the pristine
    // pre-trial reconstruction (the ladder below restores it between
    // trials).
    let intrabc_dv = if recon.allow_intrabc
        && matches!(
            b_size,
            BLOCK_8X8 | crate::cdf::BLOCK_16X16 | crate::cdf::BLOCK_32X32 | BLOCK_64X64
        ) {
        intrabc_best_dv(input, recon, mi_r, mi_c, b_size)
    } else {
        None
    };
    if single_shape && combos.len() == 1 && intrabc_dv.is_none() {
        let mut leaf = encode_leaf_with_tx(
            mi_r, mi_c, b_size, cands[0], input, recon, None, None, pricing,
        )?;
        // r426 — the committed segment rides every candidate before
        // any pricing (see `encode_leaf_sq_seg`).
        if let Some(seg) = segment_override {
            leaf.segment_id = seg;
        }
        // r423 — same §5.11.19 skip-leaf invariant as the ladder
        // below (see `fix_skip_segment`): a segmented caller prices
        // this leaf through the write path next. r426: the §5.11.9
        // `if ( skip ) segment_id = pred` short-circuit is
        // frame-type-agnostic — read_segment_id runs it on the intra
        // §5.11.8 arm too — so segmented KEY frames take the same
        // bit-silent inheritance.
        if let Some((twin, params)) = pricing {
            if params.segmentation_enabled && !params.seg_id_pre_skip && leaf.skip == 1 {
                leaf.segment_id = twin.spatial_segment_pred(mi_r, mi_c);
                // r426 — a lossless pred segment flips the §5.11.15
                // derivation to the bit-silent TX_4X4 default: the
                // skip leaf's tx commitment reverts to `None` (its
                // residual is empty either way).
                if params.lossless_array[leaf.segment_id as usize] {
                    leaf.tx_size = None;
                }
            }
        }
        return Ok(leaf);
    }
    let w4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let h4 = crate::cdf::NUM_4X4_BLOCKS_HIGH[b_size];
    let lambda = lambda_for(&recon.qp);
    // r421 — one election, one rate scale: exact twin bits (1/256-bit
    // units, block syntax only — the partition symbol is constant
    // across every candidate here and cancels) or the pre-r421
    // heuristic. The two scales never mix within a call.
    let rate_of = |leaf: &SyntaxBlock, depth: u64| -> Result<u64, Error> {
        match pricing {
            Some((twin, params)) => twin.price_block(leaf, mi_r, mi_c, b_size, params),
            None => Ok(leaf_rate(leaf) + 2 * depth),
        }
    };
    let score_of = |d: u64, rate: u64| -> u64 {
        match pricing {
            Some(_) => score256(d, lambda, rate),
            None => d + lambda * rate,
        }
    };
    let before = save_region_wh(recon, mi_r, mi_c, w4, h4);
    // r423 — §5.11.19: on a segmented INTER frame a `skip == 1` leaf's
    // segment id is the bit-silent spatial `pred` cascade, and the
    // write path (which prices these candidates through its own
    // emission body) validates the invariant. Trial candidates carry
    // the twin-derived pred BEFORE pricing; the plain-KEY and
    // heuristic paths (no twin, or segmentation off) are untouched —
    // their callers apply the same rule before any pricing runs.
    let fix_skip_segment = |leaf: &mut SyntaxBlock| {
        // r426 — committed segment first (see `encode_leaf_sq_seg`),
        // then the §5.11.19 bit-silent pred on skip leaves.
        if let Some(seg) = segment_override {
            leaf.segment_id = seg;
        }
        if let Some((twin, params)) = pricing {
            if params.segmentation_enabled && !params.seg_id_pre_skip && leaf.skip == 1 {
                leaf.segment_id = twin.spatial_segment_pred(mi_r, mi_c);
                // r426 — lossless pred segment ⇒ bit-silent TX_4X4
                // default (see the single-shape arm above).
                if params.lossless_array[leaf.segment_id as usize] {
                    leaf.tx_size = None;
                }
            }
        }
    };
    // r424 — §5.11.46 signed-delta V-plane arm election (ladder item
    // 5): a UV-palette leaf's V entries code either as direct
    // literals or through the `delta_encode_palette_colors_v` chain;
    // both arms decode to the identical entry values, so under twin
    // pricing the exact-bits smaller arm wins outright. A V list
    // whose deltas the signed-subexp chain cannot represent simply
    // keeps the literal arm (the writer rejects the commitment and
    // the trial is dropped).
    let elect_v_arm = |leaf: &mut SyntaxBlock| -> Result<(), Error> {
        if leaf.palette.size_uv == 0 {
            return Ok(());
        }
        if let Some((twin, params)) = pricing {
            let literal = twin.price_block(leaf, mi_r, mi_c, b_size, params)?;
            let mut alt = leaf.clone();
            alt.palette.delta_encode_v = true;
            if let Ok(delta) = twin.price_block(&alt, mi_r, mi_c, b_size, params) {
                if delta < literal {
                    *leaf = alt;
                }
            }
        }
        Ok(())
    };
    let mut best: Option<(SyntaxBlock, RegionSnapshot, u64)> = None;
    for (depth, &cand) in cands.iter().enumerate() {
        for &(py, puv) in combos.iter() {
            let mut leaf =
                encode_leaf_with_tx(mi_r, mi_c, b_size, cand, input, recon, py, puv, pricing)?;
            fix_skip_segment(&mut leaf);
            elect_v_arm(&mut leaf)?;
            let d = region_distortion_wh(recon, input, mi_r, mi_c, w4, h4);
            let score = score_of(d, rate_of(&leaf, depth as u64)?);
            let improves = match best.as_ref() {
                Some((_, _, s)) => score < *s,
                None => true,
            };
            if improves {
                best = Some((leaf, save_region_wh(recon, mi_r, mi_c, w4, h4), score));
            }
            restore_region(recon, mi_r, mi_c, &before);
        }
    }
    // r418 §5.11.7 intra-block-copy trial — one fixed-shape candidate
    // against the same starting state.
    if let Some(dv) = intrabc_dv {
        let mut leaf = encode_intrabc_leaf(mi_r, mi_c, b_size, dv, input, recon, pricing)?;
        fix_skip_segment(&mut leaf);
        let d = region_distortion_wh(recon, input, mi_r, mi_c, w4, h4);
        let score = score_of(d, rate_of(&leaf, 0)?);
        let improves = match best.as_ref() {
            Some((_, _, s)) => score < *s,
            None => true,
        };
        if improves {
            best = Some((leaf, save_region_wh(recon, mi_r, mi_c, w4, h4), score));
        }
        restore_region(recon, mi_r, mi_c, &before);
    }
    let (leaf, after, _) = best.expect("at least one tx candidate");
    restore_region(recon, mi_r, mi_c, &after);
    Ok(leaf)
}

/// §5.11.46 luma palette commitment candidate for one square leaf —
/// the colour list (sorted ascending, the §5.11.46 canonical form)
/// plus the §5.11.49 `ColorMapY` (row-major, stride = block width).
#[derive(Debug, Clone)]
pub(crate) struct PaletteCandY {
    /// `2..=PALETTE_COLORS` strictly-ascending colours.
    pub(crate) colors: Vec<u16>,
    /// `n × n` row-major indices into `colors`.
    pub(crate) map: Vec<u8>,
}

/// Build the §5.11.46 luma palette candidate for one square leaf, or
/// `None` when the block is ineligible or not exactly representable.
///
/// Eligibility mirrors the §5.11.46 outer gate on the write side
/// (square `BLOCK_8X8..=BLOCK_64X64`) plus this driver's own
/// restrictions: the block must be fully on-screen (clipped leaves
/// would need the §5.11.49 on-screen bump handling in the map) and
/// carry `2..=PALETTE_COLORS` distinct sample values (r418 scope —
/// exact-representable blocks; quantised palettes for busier blocks
/// are a follow-up arc).
#[cfg(test)]
fn palette_candidate_y(
    input: &YuvFrame,
    recon: &ReconState,
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
) -> Option<PaletteCandY> {
    palette_candidates_y(input, recon, mi_r, mi_c, b_size, 1)
        .into_iter()
        .next()
}

/// r424 — up to `max_cands` §5.11.46 luma palette candidates for one
/// leaf, best proxy score first: the exact-representable arm yields
/// its single zero-distortion palette; the k-means arm yields the
/// proxy ladder's top `max_cands` DISTINCT per-`k` palettes so the
/// caller's exact-bits election (`RateTwin::price_block` over the
/// fully-coded leaf — §5.11.46 entries, §5.11.49 tokens AND the
/// palette-predicted §5.11.39 residual chain) settles the `k` choice
/// the `SSE + λ·entry-cost` proxy used to make alone.
fn palette_candidates_y(
    input: &YuvFrame,
    recon: &ReconState,
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    max_cands: usize,
) -> Vec<PaletteCandY> {
    use crate::cdf::PALETTE_COLORS;
    // r427 — the palette search stays 8-bit 4:2:0 (the dense 256-bin
    // histogram below indexes by sample value); the general drivers
    // never open the §5.9.5 gate outside that pairing.
    if recon.bit_depth != 8 || recon.subsampling_x != 1 || recon.subsampling_y != 1 {
        return Vec::new();
    }
    // §5.11.46 outer gate — r425: rectangular shapes join the squares
    // (`MiSize >= BLOCK_8X8 && Block_Width <= 64 && Block_Height <=
    // 64`; this driver keeps both dims >= 8 — its rect leaves come
    // from the >= 16 HORZ / VERT arms).
    let bw = NUM_4X4_BLOCKS_WIDE[b_size] * 4;
    let bh = crate::cdf::NUM_4X4_BLOCKS_HIGH[b_size] * 4;
    if !(8..=64).contains(&bw) || !(8..=64).contains(&bh) {
        return Vec::new();
    }
    let (row0, col0) = ((mi_r as usize) * 4, (mi_c as usize) * 4);
    if row0 >= recon.height || col0 >= recon.width {
        return Vec::new();
    }
    // r425 — frame-edge-straddling blocks build over the ACTUAL
    // on-screen sub-rectangle (§5.11.49 codes only its
    // anti-diagonals; the off-screen map is spec-replicated below).
    let os_w = bw.min(recon.width - col0);
    let os_h = bh.min(recon.height - row0);
    let mut hist = [0u32; 256];
    let mut distinct = 0usize;
    for i in 0..os_h {
        let row = &input.y[(row0 + i) * recon.width + col0..][..os_w];
        for &v in row {
            if hist[v as usize] == 0 {
                distinct += 1;
            }
            hist[v as usize] += 1;
        }
    }
    if distinct < 2 {
        return Vec::new();
    }
    let color_lists: Vec<Vec<u16>> = if distinct <= PALETTE_COLORS {
        // Exact-representable block: the distinct values ARE the
        // palette (zero-distortion prediction).
        vec![(0..256u16).filter(|&c| hist[c as usize] > 0).collect()]
    } else if !recon.lossless && distinct <= kmeans_distinct_bound(os_w * os_h) {
        // r418 colour clustering: weighted 1-D k-means over the value
        // histogram with a size-RD pick of `k` (§5.11.46
        // `palette_size` election — each extra colour costs entry
        // bits, each dropped colour costs clustering SSE). r424: the
        // proxy ladder now surfaces its top candidates for the exact
        // election.
        kmeans_palette_1d_candidates(&hist, lambda_for(&recon.qp), max_cands)
    } else {
        return Vec::new();
    };
    color_lists
        .into_iter()
        .filter(|colors| colors.len() >= 2)
        .map(|colors| {
            // Nearest-colour LUT over the present values (identity on
            // the exact arm).
            let mut lut = [0u8; 256];
            for (c, &cnt) in hist.iter().enumerate() {
                if cnt > 0 {
                    let mut best = 0usize;
                    let mut best_d = u32::MAX;
                    for (idx, &pc) in colors.iter().enumerate() {
                        let d = (c as i32 - i32::from(pc)).unsigned_abs();
                        if d < best_d {
                            best_d = d;
                            best = idx;
                        }
                    }
                    lut[c] = best as u8;
                }
            }
            let mut map = vec![0u8; bw * bh];
            for i in 0..os_h {
                let row = &input.y[(row0 + i) * recon.width + col0..][..os_w];
                for (j, &v) in row.iter().enumerate() {
                    map[i * bw + j] = lut[v as usize];
                }
            }
            // §5.11.49 off-screen replication (right columns from the
            // last on-screen column, bottom rows from the last
            // on-screen row) — the decoder's ColorMap fill, so the
            // §7.11.4 prediction of overhanging TUs matches exactly.
            for i in 0..os_h {
                for j in os_w..bw {
                    map[i * bw + j] = map[i * bw + os_w - 1];
                }
            }
            for i in os_h..bh {
                let (head, tail) = map.split_at_mut(i * bw);
                tail[..bw].copy_from_slice(&head[(os_h - 1) * bw..][..bw]);
            }
            PaletteCandY { colors, map }
        })
        .collect()
}

/// Distinct-value bound above which the k-means arm is not attempted
/// (dense texture / noise blocks: a palette cannot win there and the
/// trial would be wasted work).
const KMEANS_MAX_DISTINCT: usize = 64;

/// Density-aware clustering gate: a block only clusters when its
/// distinct-value count is at most one eighth of its sample count
/// (capped at [`KMEANS_MAX_DISTINCT`]) — 8×8 blocks stay on the exact
/// arm, 16×16 admit up to 32 values, 32×32+ up to 64. Denser blocks
/// are texture, not palette territory, and the trial would only slow
/// the search / destabilise elections.
fn kmeans_distinct_bound(samples: usize) -> usize {
    (samples / 8).min(KMEANS_MAX_DISTINCT)
}

/// Weighted 1-D k-means (Lloyd) over a value histogram with a
/// size-RD proxy score of `k ∈ 2..=PALETTE_COLORS`: for each `k`,
/// quantile seeding + 8 Lloyd rounds, scored as `SSE + λ · (10 + 8k)`
/// (the [`leaf_rate`] palette entry cost); every `k`'s converged
/// (rounded, deduped, strictly-ascending) palette is returned
/// best-first, truncated to `max_cands` DISTINCT lists — r424: the
/// caller's exact-bits election picks among them.
fn kmeans_palette_1d_candidates(hist: &[u32; 256], lambda: u64, max_cands: usize) -> Vec<Vec<u16>> {
    use crate::cdf::PALETTE_COLORS;
    let values: Vec<(u32, u32)> = (0..256u32)
        .filter(|&c| hist[c as usize] > 0)
        .map(|c| (c, hist[c as usize]))
        .collect();
    let total: u64 = values.iter().map(|&(_, w)| u64::from(w)).sum();
    let mut scored: Vec<(Vec<u16>, u64)> = Vec::new();
    for k in 2..=PALETTE_COLORS {
        // Quantile seeding over the cumulative distribution.
        let mut centroids: Vec<f64> = Vec::with_capacity(k);
        let mut acc = 0u64;
        let mut vi = 0usize;
        for c in 0..k {
            let target = (total * (2 * c as u64 + 1)) / (2 * k as u64);
            while vi + 1 < values.len() && acc + u64::from(values[vi].1) <= target {
                acc += u64::from(values[vi].1);
                vi += 1;
            }
            centroids.push(f64::from(values[vi].0));
        }
        for _ in 0..8 {
            // Assign + recompute (1-D: nearest centroid).
            let mut sum = vec![0f64; k];
            let mut cnt = vec![0f64; k];
            for &(v, w) in &values {
                let mut bi = 0usize;
                let mut bd = f64::MAX;
                for (i, &c) in centroids.iter().enumerate() {
                    let d = (f64::from(v) - c).abs();
                    if d < bd {
                        bd = d;
                        bi = i;
                    }
                }
                sum[bi] += f64::from(v) * f64::from(w);
                cnt[bi] += f64::from(w);
            }
            for i in 0..k {
                if cnt[i] > 0.0 {
                    centroids[i] = sum[i] / cnt[i];
                }
            }
        }
        // Round, sort, dedupe.
        let mut cols: Vec<u16> = centroids
            .iter()
            .map(|&c| c.round().clamp(0.0, 255.0) as u16)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        if cols.len() < 2 {
            continue;
        }
        // Weighted SSE at the rounded palette + entry-cost RD.
        let mut sse = 0u64;
        for &(v, w) in &values {
            let d = cols
                .iter()
                .map(|&c| (i64::from(v) - i64::from(c)).unsigned_abs())
                .min()
                .unwrap_or(0);
            sse += d * d * u64::from(w);
        }
        let score = sse + lambda * (10 + 8 * cols.len() as u64);
        if !scored.iter().any(|(c, _)| *c == cols) {
            scored.push((cols, score));
        }
    }
    scored.sort_by_key(|&(_, s)| s);
    scored.truncate(max_cands);
    scored.into_iter().map(|(cols, _)| cols).collect()
}

/// §5.11.46 chroma palette commitment candidate for one square leaf —
/// the joint (U, V) colour pairs (U non-strictly ascending — the
/// §5.11.46 post-sort canonical order; V by pair index) plus the
/// §5.11.49 `ColorMapUV` (row-major over the subsampled block).
#[derive(Debug, Clone)]
pub(crate) struct PaletteCandUv {
    /// `2..=PALETTE_COLORS` U colours, non-strictly ascending.
    pub(crate) colors_u: Vec<u16>,
    /// Per-index V partner of each U colour (arbitrary order).
    pub(crate) colors_v: Vec<u16>,
    /// `(n/2) × (n/2)` row-major indices into the pair list.
    pub(crate) map: Vec<u8>,
}

/// r424 — up to `max_cands` §5.11.46 chroma palette candidates, best
/// proxy score first (the chroma twin of [`palette_candidates_y`]):
/// the distinct count is taken over joint (U, V) sample PAIRS of the
/// subsampled block — §5.11.49 codes ONE shared `ColorMapUV` for both
/// chroma planes.
fn palette_candidates_uv(
    input: &YuvFrame,
    recon: &ReconState,
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    max_cands: usize,
) -> Vec<PaletteCandUv> {
    use crate::cdf::PALETTE_COLORS;
    // r427 — same 8-bit 4:2:0 scope as the luma twin.
    if recon.bit_depth != 8 || recon.subsampling_x != 1 || recon.subsampling_y != 1 {
        return Vec::new();
    }
    // §5.11.46 outer gate — r425: rectangular shapes join the squares
    // (both dims `8..=64` for this driver). Every such leaf has
    // chroma under 4:2:0.
    let bw = NUM_4X4_BLOCKS_WIDE[b_size] * 4;
    let bh = crate::cdf::NUM_4X4_BLOCKS_HIGH[b_size] * 4;
    if !(8..=64).contains(&bw) || !(8..=64).contains(&bh) {
        return Vec::new();
    }
    let (row0, col0) = ((mi_r as usize) * 4, (mi_c as usize) * 4);
    if row0 >= recon.height || col0 >= recon.width {
        return Vec::new();
    }
    let (cbw, cbh) = (bw / 2, bh / 2);
    let (crow0, ccol0) = (row0 / 2, col0 / 2);
    let cw = recon.chroma_w;
    // r425 — clipped blocks build over the on-screen chroma
    // sub-rectangle (see the luma twin).
    let os_cw = cbw.min(cw - ccol0);
    let os_ch = cbh.min(recon.chroma_h - crow0);
    // Distinct joint (U, V) pairs with weights.
    let mut weights: std::collections::BTreeMap<(u16, u16), u32> =
        std::collections::BTreeMap::new();
    for i in 0..os_ch {
        for j in 0..os_cw {
            let off = (crow0 + i) * cw + (ccol0 + j);
            let p = (input.u[off], input.v[off]);
            *weights.entry(p).or_insert(0) += 1;
            if weights.len() > kmeans_distinct_bound(os_cw * os_ch).max(PALETTE_COLORS) {
                return Vec::new();
            }
        }
    }
    if weights.len() < 2 {
        return Vec::new();
    }
    let pair_lists: Vec<Vec<(u16, u16)>> = if weights.len() <= PALETTE_COLORS {
        // Exact-representable chroma block.
        vec![weights.keys().copied().collect()]
    } else if !recon.lossless {
        // r418 colour clustering: weighted 2-D k-means over the joint
        // (U, V) pairs with the size-RD pick of `k` (each pair codes
        // BOTH a U and a V entry — double the entry cost of the luma
        // arm). r424: the proxy ladder surfaces its top candidates
        // for the exact election.
        kmeans_palette_2d_candidates(&weights, lambda_for(&recon.qp), max_cands)
    } else {
        return Vec::new();
    };
    pair_lists
        .into_iter()
        .filter(|pairs| pairs.len() >= 2)
        .map(|mut pairs| {
            // §5.11.46 canonical order: U non-strictly ascending
            // (ties broken by V for determinism — pair order among
            // equal U values is unconstrained by the entry coding).
            pairs.sort_unstable();
            let mut map = vec![0u8; cbw * cbh];
            for i in 0..os_ch {
                for j in 0..os_cw {
                    let off = (crow0 + i) * cw + (ccol0 + j);
                    let p = (i64::from(input.u[off]), i64::from(input.v[off]));
                    // Nearest pair (exact arm: the pair itself).
                    let mut bi = 0usize;
                    let mut bd = i64::MAX;
                    for (idx, &(pu, pv)) in pairs.iter().enumerate() {
                        let d = (p.0 - i64::from(pu)).pow(2) + (p.1 - i64::from(pv)).pow(2);
                        if d < bd {
                            bd = d;
                            bi = idx;
                        }
                    }
                    map[i * cbw + j] = bi as u8;
                }
            }
            // §5.11.49 off-screen replication on the shared UV map.
            for i in 0..os_ch {
                for j in os_cw..cbw {
                    map[i * cbw + j] = map[i * cbw + os_cw - 1];
                }
            }
            for i in os_ch..cbh {
                let (head, tail) = map.split_at_mut(i * cbw);
                tail[..cbw].copy_from_slice(&head[(os_ch - 1) * cbw..][..cbw]);
            }
            PaletteCandUv {
                colors_u: pairs.iter().map(|&(u, _)| u).collect(),
                colors_v: pairs.iter().map(|&(_, v)| v).collect(),
                map,
            }
        })
        .collect()
}

/// Weighted 2-D k-means (Lloyd) over joint (U, V) pair weights with a
/// size-RD pick of `k ∈ 2..=PALETTE_COLORS` — the chroma twin of the
/// 1-D ladder (entry cost doubled: each §5.11.46 UV pair codes a U
/// and a V entry). r424: returns the proxy ladder's top `max_cands`
/// DISTINCT per-`k` centroid-pair lists, best first — the caller's
/// exact-bits election settles the final `k`.
fn kmeans_palette_2d_candidates(
    weights: &std::collections::BTreeMap<(u16, u16), u32>,
    lambda: u64,
    max_cands: usize,
) -> Vec<Vec<(u16, u16)>> {
    use crate::cdf::PALETTE_COLORS;
    let values: Vec<((f64, f64), u32)> = weights
        .iter()
        .map(|(&(u, v), &w)| ((f64::from(u), f64::from(v)), w))
        .collect();
    let total: u64 = values.iter().map(|&(_, w)| u64::from(w)).sum();
    let mut scored: Vec<(Vec<(u16, u16)>, u64)> = Vec::new();
    for k in 2..=PALETTE_COLORS {
        // Seed along the weighted BTreeMap (U-major) order.
        let mut centroids: Vec<(f64, f64)> = Vec::with_capacity(k);
        let mut acc = 0u64;
        let mut vi = 0usize;
        for c in 0..k {
            let target = (total * (2 * c as u64 + 1)) / (2 * k as u64);
            while vi + 1 < values.len() && acc + u64::from(values[vi].1) <= target {
                acc += u64::from(values[vi].1);
                vi += 1;
            }
            centroids.push(values[vi].0);
        }
        for _ in 0..8 {
            let mut sum = vec![(0f64, 0f64); k];
            let mut cnt = vec![0f64; k];
            for &((u, v), w) in &values {
                let mut bi = 0usize;
                let mut bd = f64::MAX;
                for (i, &(cu, cv)) in centroids.iter().enumerate() {
                    let d = (u - cu) * (u - cu) + (v - cv) * (v - cv);
                    if d < bd {
                        bd = d;
                        bi = i;
                    }
                }
                sum[bi].0 += u * f64::from(w);
                sum[bi].1 += v * f64::from(w);
                cnt[bi] += f64::from(w);
            }
            for i in 0..k {
                if cnt[i] > 0.0 {
                    centroids[i] = (sum[i].0 / cnt[i], sum[i].1 / cnt[i]);
                }
            }
        }
        let mut cols: Vec<(u16, u16)> = centroids
            .iter()
            .map(|&(u, v)| {
                (
                    u.round().clamp(0.0, 255.0) as u16,
                    v.round().clamp(0.0, 255.0) as u16,
                )
            })
            .collect();
        cols.sort_unstable();
        cols.dedup();
        if cols.len() < 2 {
            continue;
        }
        let mut sse = 0u64;
        for &((u, v), w) in &values {
            let d = cols
                .iter()
                .map(|&(cu, cv)| {
                    let du = (u - f64::from(cu)).abs() as u64;
                    let dv = (v - f64::from(cv)).abs() as u64;
                    du * du + dv * dv
                })
                .min()
                .unwrap_or(0);
            sse += d * u64::from(w);
        }
        let score = sse + lambda * (10 + 16 * cols.len() as u64);
        if !scored.iter().any(|(c, _)| *c == cols) {
            scored.push((cols, score));
        }
    }
    scored.sort_by_key(|&(_, s)| s);
    scored.truncate(max_cands);
    scored.into_iter().map(|(cols, _)| cols).collect()
}

// ---------------------------------------------------------------------
// r418 §5.11.7 intra-block-copy search.
// ---------------------------------------------------------------------

/// §6.10.24 `is_mv_valid` with `use_intrabc = 1` — the DV validity
/// predicate for this single-tile driver (`MiRowStart = MiColStart =
/// 0`, `MiRowEnd = mi_rows`, `MiColEnd = mi_cols`, 64×64 superblocks).
/// `dv_r` / `dv_c` are whole-pel; this driver additionally restricts
/// them to EVEN values so the 4:2:0 chroma copy stays integer-aligned
/// (candidate generation only emits even offsets).
///
/// Spec walk: integer alignment (`force_integer_mv = 1` on intra
/// frames), source rectangle inside the tile, the
/// `INTRABC_DELAY_SB64 = 4` raster-delay guard, and the wavefront
/// guard with `gradient = 1 + INTRABC_DELAY_SB64 +
/// use_128x128_superblock = 5`.
pub(crate) fn intrabc_dv_valid(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    dv_r: i32,
    dv_c: i32,
    mi_rows: u32,
    mi_cols: u32,
) -> bool {
    const INTRABC_DELAY_SB64: i64 = 4;
    let bw = (NUM_4X4_BLOCKS_WIDE[b_size] * 4) as i64;
    let bh = bw; // square driver shapes only
                 // §6.10.24 magnitude bound: |Mv| < 1 << 14 in 1/8-pel units.
    if (i64::from(dv_r) * 8).abs() >= (1 << 14) || (i64::from(dv_c) * 8).abs() >= (1 << 14) {
        return false;
    }
    let src_top = i64::from(mi_r) * 4 + i64::from(dv_r);
    let src_left = i64::from(mi_c) * 4 + i64::from(dv_c);
    let src_bottom = src_top + bh;
    let src_right = src_left + bw;
    // HasChroma sub-8 adjustments never fire: bw, bh >= 8 here.
    if src_top < 0
        || src_left < 0
        || src_bottom > i64::from(mi_rows) * 4
        || src_right > i64::from(mi_cols) * 4
    {
        return false;
    }
    let active_sb_row = (i64::from(mi_r) * 4) / 64;
    let active_sb64_col = (i64::from(mi_c) * 4) >> 6;
    let src_sb_row = (src_bottom - 1) / 64;
    let src_sb64_col = (src_right - 1) >> 6;
    let total_sb64_per_row = ((i64::from(mi_cols) - 1) >> 4) + 1;
    let active_sb64 = active_sb_row * total_sb64_per_row + active_sb64_col;
    let src_sb64 = src_sb_row * total_sb64_per_row + src_sb64_col;
    if src_sb64 >= active_sb64 - INTRABC_DELAY_SB64 {
        return false;
    }
    let gradient = 1 + INTRABC_DELAY_SB64; // + use_128x128_superblock = 0
    let wf_offset = gradient * (active_sb_row - src_sb_row);
    if src_sb_row > active_sb_row
        || src_sb64_col >= active_sb64_col - INTRABC_DELAY_SB64 + wf_offset
    {
        return false;
    }
    true
}

/// Frame-level §5.9.20 gate heuristic — `true` iff a provable
/// §6.10.24-reachable exact copy source exists, on either tier:
///
/// * **r418 superblock tier**: some pair of exact duplicate 64×64
///   tiles (all three planes) where the later can copy the earlier.
/// * **r425 glyph tier**: at least [`INTRABC_GATE_MIN_CELLS`] 16×16
///   grid cells (all three planes, non-flat luma) with an exact
///   earlier duplicate at a §6.10.24-valid displacement, covering at
///   least 1/[`INTRABC_GATE_CELL_FRACTION`] of the grid — repeated
///   glyphs / UI patterns that never align to whole superblocks.
///
/// Content without a provable copy source keeps the gate closed —
/// the per-leaf `use_intrabc` S() is pure overhead there.
fn intrabc_beneficial(input: &YuvFrame) -> bool {
    let (w, h) = (input.width as usize, input.height as usize);
    let cw = w / 2;
    let mi_rows = 2 * ((h as u32 + 7) >> 3);
    let mi_cols = 2 * ((w as u32 + 7) >> 3);
    let fnv = |hash: &mut u64, v: u16| {
        *hash ^= u64::from(v);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    };

    // --- r418 superblock tier. ---
    let (sb_rows, sb_cols) = (h / 64, w / 64);
    if sb_rows * sb_cols >= 2 {
        let mut tiles: Vec<(usize, usize, u64)> = Vec::new();
        for sbr in 0..sb_rows {
            for sbc in 0..sb_cols {
                let mut hash = 0xcbf2_9ce4_8422_2325u64;
                for i in 0..64 {
                    for j in 0..64 {
                        fnv(&mut hash, input.y[(sbr * 64 + i) * w + sbc * 64 + j]);
                    }
                }
                for i in 0..32 {
                    for j in 0..32 {
                        let off = (sbr * 32 + i) * cw + sbc * 32 + j;
                        fnv(&mut hash, input.u[off]);
                        fnv(&mut hash, input.v[off]);
                    }
                }
                tiles.push((sbr, sbc, hash));
            }
        }
        for (i, &(r1, c1, h1)) in tiles.iter().enumerate() {
            for &(r0, c0, h0) in &tiles[..i] {
                if h0 == h1
                    && intrabc_dv_valid(
                        (r1 * 16) as u32,
                        (c1 * 16) as u32,
                        BLOCK_64X64,
                        (r0 as i32 - r1 as i32) * 64,
                        (c0 as i32 - c1 as i32) * 64,
                        mi_rows,
                        mi_cols,
                    )
                {
                    return true;
                }
            }
        }
    }

    // --- r425 glyph tier: 16×16 grid cells. ---
    let (cell_rows, cell_cols) = (h / 16, w / 16);
    let total_cells = cell_rows * cell_cols;
    if total_cells < 2 * INTRABC_GATE_MIN_CELLS {
        return false;
    }
    let mut buckets: std::collections::HashMap<u64, Vec<(usize, usize)>> =
        std::collections::HashMap::new();
    let mut matched = 0usize;
    for cr in 0..cell_rows {
        for cc in 0..cell_cols {
            let (py, px) = (cr * 16, cc * 16);
            let first = input.y[py * w + px];
            let mut flat = true;
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for i in 0..16 {
                for j in 0..16 {
                    let v = input.y[(py + i) * w + px + j];
                    flat &= v == first;
                    fnv(&mut hash, v);
                }
            }
            if flat {
                continue;
            }
            for i in 0..8 {
                for j in 0..8 {
                    let off = (py / 2 + i) * cw + px / 2 + j;
                    fnv(&mut hash, input.u[off]);
                    fnv(&mut hash, input.v[off]);
                }
            }
            let bucket = buckets.entry(hash).or_default();
            // Scan a bounded prefix of earlier duplicates for one the
            // §6.10.24 lag admits.
            if bucket.iter().take(16).any(|&(sy, sx)| {
                intrabc_dv_valid(
                    (py / 4) as u32,
                    (px / 4) as u32,
                    crate::cdf::BLOCK_16X16,
                    sy as i32 - py as i32,
                    sx as i32 - px as i32,
                    mi_rows,
                    mi_cols,
                )
            }) {
                matched += 1;
            }
            bucket.push((py, px));
        }
    }
    matched >= INTRABC_GATE_MIN_CELLS && matched * INTRABC_GATE_CELL_FRACTION >= total_cells
}

/// r425 glyph-tier gate floor: fewer provable copies than this can't
/// amortise the frame-wide `use_intrabc` flag overhead.
const INTRABC_GATE_MIN_CELLS: usize = 4;

/// r425 glyph-tier gate density: matched cells must cover at least
/// `1 / INTRABC_GATE_CELL_FRACTION` of the 16×16 grid.
const INTRABC_GATE_CELL_FRACTION: usize = 64;

/// Pick the best §5.11.7 DV candidate for one square leaf by luma SSD
/// of the reconstruction copy against the input block, over a bounded
/// even-offset candidate set (multiples of the superblock stride and
/// of the block extent, leftward / upward / diagonal). Returns `None`
/// when no candidate passes §6.10.24 validity.
fn intrabc_best_dv(
    input: &YuvFrame,
    recon: &ReconState,
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
) -> Option<(i32, i32)> {
    let n = NUM_4X4_BLOCKS_WIDE[b_size] * 4;
    let (row0, col0) = ((mi_r as usize) * 4, (mi_c as usize) * 4);
    if row0 + n > recon.height || col0 + n > recon.width {
        return None;
    }
    let mut cands: Vec<(i32, i32)> = Vec::new();
    // r425 — exact-match seeds first: probe the per-frame block-hash
    // index with this block's INPUT samples (nearest sources first —
    // strictly-better ties in the SSD ranking below keep the first,
    // i.e. cheapest-DV, winner; §6.10.24-unreachable seeds fall to
    // the validity check like any other candidate). Uniform blocks
    // skip the probe: DC / palette arms already code them at
    // near-zero cost.
    if let Some(tier) = crate::encoder::dv_hash::dv_hash_size_idx(n) {
        let (hash, uniform) =
            crate::encoder::dv_hash::hash_block_direct(&input.y, recon.width, row0, col0, n);
        if !uniform {
            cands.extend(recon.dv_hash.candidates(hash, tier, row0, col0));
        }
    }
    for k in 1..=3i32 {
        for &(dr, dc) in &[(0, -64 * k), (-64 * k, 0), (-64 * k, -64 * k)] {
            cands.push((dr, dc));
        }
        if n < 64 {
            let s = (n as i32) * k;
            for &(dr, dc) in &[(0, -s), (-s, 0), (-s, -s)] {
                cands.push((dr, dc));
            }
        }
    }
    let mut best: Option<((i32, i32), u64)> = None;
    for (dr, dc) in cands {
        if !intrabc_dv_valid(mi_r, mi_c, b_size, dr, dc, recon.mi_rows, recon.mi_cols) {
            continue;
        }
        let (sr, sc) = ((row0 as i32 + dr) as usize, (col0 as i32 + dc) as usize);
        let mut ssd = 0u64;
        for i in 0..n {
            for j in 0..n {
                let d = i64::from(input.y[(row0 + i) * recon.width + col0 + j])
                    - i64::from(recon.y[(sr + i) * recon.width + sc + j]);
                ssd += (d * d) as u64;
            }
        }
        if best.as_ref().map_or(true, |&(_, s)| ssd < s) {
            best = Some(((dr, dc), ssd));
        }
    }
    best.map(|(dv, _)| dv)
}

/// Encode one square leaf on the §5.11.7 intra-block-copy arm at DV
/// `(dv_r, dv_c)` (whole-pel, even): `is_inter = 1` residual layout —
/// one `Max_Tx_Size_Rect` luma TU on the lossy arm (uniform §5.11.17
/// depth 0, committed as one no-split [`VarTxSyntaxTree`]) or the
/// row-major `TX_4X4` grid on the lossless arm, the §5.11.48 INTER
/// tx-type sets, and the §5.11.40 chroma inheritance. Prediction is
/// the §7.11.3 whole-pel copy from the running reconstruction (the
/// §6.10.24 wavefront guarantee keeps the source strictly before the
/// current superblock neighbourhood, so it cannot alias this leaf).
fn encode_intrabc_leaf(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    dv: (i32, i32),
    input: &YuvFrame,
    recon: &mut ReconState,
    // r424 — running per-TU twin fork threading (the intrabc arm
    // prices the §5.11.48 INTER-set candidates exactly like an inter
    // leaf: `is_inter = 1` on the residual layout).
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    use crate::cdf::inter_tx_type_set;
    let n = NUM_4X4_BLOCKS_WIDE[b_size] * 4;
    let (row0, col0) = ((mi_r as usize) * 4, (mi_c as usize) * 4);
    let width = recon.width;
    let lossless = recon.lossless;
    let qp = recon.qp;
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
        segment_id: 0,
        y_mode: 0,
        use_filter_intra: false,
        filter_intra_mode: None,
    });
    let (dv_r, dv_c) = dv;
    let (src_row0, src_col0) = ((row0 as i32 + dv_r) as usize, (col0 as i32 + dv_c) as usize);

    let luma_tx = if lossless {
        TX_4X4
    } else {
        MAX_TX_SIZE_RECT[b_size]
    };
    let (ltw, lth) = (TX_WIDTH[luma_tx], TX_HEIGHT[luma_tx]);
    let mut residual_quant: Vec<Vec<i32>> = Vec::new();
    let mut luma_tx_types: Vec<u8> = Vec::new();
    let tu_cols = n / ltw;
    let mut tu_type_grid = vec![DCT_DCT as u8; tu_cols * (n / lth)];
    // §5.11.34 luma walk: row-major on the lossless arm; the lossy arm
    // is a single `Max_Tx_Size_Rect` TU (depth 0) — row-major order is
    // the §5.11.36 order for both.
    let mut ty = 0usize;
    while ty < n {
        let mut tx = 0usize;
        while tx < n {
            let (tr, tc) = (row0 + ty, col0 + tx);
            let mut pred = vec![0u16; ltw * lth];
            for i in 0..lth {
                for j in 0..ltw {
                    pred[i * ltw + j] = recon.y[(src_row0 + ty + i) * width + src_col0 + tx + j];
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
                crate::encoder::inter_frame::residual_tx_search_luma_inter(
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
            tx += ltw;
        }
        ty += lth;
    }

    // --- Chroma (always `HasChroma` at the square >= 8x8 shapes;
    // r427: subsampling-derived extents — the intrabc election is
    // 8-bit-4:2:0-gated today, but the walk is format-general). ---
    let (ssx, ssy) = (
        u32::from(recon.subsampling_x),
        u32::from(recon.subsampling_y),
    );
    let (cn_w, cn_h) = (n >> ssx, n >> ssy);
    let (crow0, ccol0) = (row0 >> ssy, col0 >> ssx);
    let (csrc_row0, csrc_col0) = (
        ((row0 as i32 + dv_r) >> ssy) as usize,
        ((col0 as i32 + dv_c) >> ssx) as usize,
    );
    let chroma_tx = if lossless {
        TX_4X4
    } else {
        get_tx_size(1, luma_tx, b_size, recon.subsampling_x, recon.subsampling_y).unwrap_or(TX_4X4)
    };
    let (ctw, cth) = (TX_WIDTH[chroma_tx], TX_HEIGHT[chroma_tx]);
    let chroma_set = inter_tx_type_set(
        tx_size_sqr_index(chroma_tx) as u32,
        TX_SIZE_SQR_UP[chroma_tx] as u32,
        false,
    );
    let cw = recon.chroma_w;
    for plane in 1..usize::from(recon.num_planes) {
        let mut ty = 0usize;
        while ty < cn_h {
            let mut tx = 0usize;
            while tx < cn_w {
                let (tr, tc) = (crow0 + ty, ccol0 + tx);
                // §5.11.40 inter-chroma TxType inheritance from the
                // subsampling-lifted luma cell, filtered by the
                // chroma-size inter set.
                let chroma_tt = if lossless || TX_SIZE_SQR_UP[chroma_tx] > crate::cdf::TX_32X32 {
                    DCT_DCT
                } else {
                    let lifted_x = ((tc >> 2) << (2 + ssx)).saturating_sub(col0);
                    let lifted_y = ((tr >> 2) << (2 + ssy)).saturating_sub(row0);
                    let luma_tt =
                        tu_type_grid[(lifted_y / lth) * tu_cols + lifted_x / ltw] as usize;
                    if is_tx_type_in_set(true, chroma_set, luma_tt) {
                        luma_tt
                    } else {
                        DCT_DCT
                    }
                };
                let src_plane: &[u16] = if plane == 1 { &recon.u } else { &recon.v };
                let mut pred = vec![0u16; ctw * cth];
                for i in 0..cth {
                    for j in 0..ctw {
                        pred[i * ctw + j] =
                            src_plane[(csrc_row0 + ty + i) * cw + csrc_col0 + tx + j];
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

    let all_zero = residual_quant.iter().all(|tu| tu.iter().all(|&q| q == 0));
    let skip = u8::from(all_zero);
    if all_zero {
        residual_quant.clear();
        luma_tx_types.clear();
    }
    if luma_tx_types.iter().all(|&t| t == DCT_DCT as u8) {
        luma_tx_types.clear();
    }

    let mut block = SyntaxBlock::skip_leaf(0, None);
    block.skip = skip;
    block.intrabc_mv = Some([dv_r * 8, dv_c * 8]);
    block.residual_quant = residual_quant;
    block.residual_tx_type = luma_tx_types;
    if !lossless && skip == 0 {
        block.var_tx_trees = vec![crate::encoder::inter_frame::uniform_var_tx_tree(
            MAX_TX_SIZE_RECT[b_size],
            0,
        )];
    }
    Ok(block)
}

/// One-shape leaf encode at a fixed luma TX size — see
/// [`encode_leaf_sq`] for the §5.11.15 TX search wrapper. When
/// `palette_y` is `Some`, the luma plane rides the §5.11.46 palette
/// arm: `y_mode = DC_PRED`, no filter-intra (the §5.11.24 gate closes
/// on `PaletteSizeY > 0`), and every luma TU predicts from the
/// §7.11.4 palette-mapped samples before the coded residual. When
/// `palette_uv` is `Some`, the chroma planes ride the same arm:
/// `uv_mode = DC_PRED` (the §5.11.46 UV gate), no CFL, and every
/// chroma TU predicts from its plane's palette colours through the
/// shared §5.11.49 `ColorMapUV`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_leaf_with_tx(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    luma_tx: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    palette_y: Option<&PaletteCandY>,
    palette_uv: Option<&PaletteCandUv>,
    // r424 — `Some` threads a running per-TU twin fork through the
    // luma residual chain (exact §5.11.47/§5.11.39 candidate pricing
    // — see [`TuFork`]); `None` keeps the magnitude proxy.
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
) -> Result<SyntaxBlock, Error> {
    // r425 — rectangular leaves ride the same encoder: all extents
    // split into (bw, bh) / (cbw, cbh).
    let bw = NUM_4X4_BLOCKS_WIDE[b_size] * 4;
    let bh = crate::cdf::NUM_4X4_BLOCKS_HIGH[b_size] * 4;
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let width = recon.width;
    let lossless = recon.lossless;
    let qp = recon.qp;

    // --- Luma ---
    // §5.11.46 palette arm (r418): the coded `y_mode` is DC_PRED and
    // the §5.11.24 filter-intra gate is closed; the luma prediction is
    // the §7.11.4 palette map instead of a §7.11.2 neighbour mode.
    let (y_mode, angle_delta_y, filter_intra_mode) = match palette_y {
        Some(_) => (DC_PRED as u8, 0i8, None),
        None => pick_y_mode(recon, input, row0, col0, bw, bh),
    };
    // r424 — the running per-TU fork (lossy arm only), armed with the
    // leaf's §8.3.2 `intra_dir` inputs.
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
        is_inter: false,
        segment_id: 0,
        y_mode,
        use_filter_intra: filter_intra_mode.is_some(),
        filter_intra_mode,
    });
    let (ltw, lth) = (TX_WIDTH[luma_tx], TX_HEIGHT[luma_tx]);
    // r425 — on-screen extent for frame-edge-straddling leaves: the
    // §5.11.35 walk skips TUs whose origin is off-screen and the
    // clip-aware residual legs zero the off-screen samples.
    let os_w = bw.min(width - col0);
    let os_h = bh.min(recon.height - row0);
    let mut residual_quant: Vec<Vec<i32>> = Vec::new();
    let mut luma_tx_types: Vec<u8> = Vec::new();
    let mut ty = 0usize;
    while ty < bh {
        let mut tx = 0usize;
        while tx < bw {
            // §5.11.35 line-13 `startX >= maxX || startY >= maxY`
            // early return — the TU is never coded.
            if ty >= os_h || tx >= os_w {
                tx += ltw;
                continue;
            }
            let (tr, tc) = (row0 + ty, col0 + tx);
            let pred = match palette_y {
                Some(p) => {
                    // §7.11.4 `predict_palette` over this TU's
                    // footprint: map indices → palette colours.
                    let mut buf = vec![0u16; ltw * lth];
                    for i in 0..lth {
                        for j in 0..ltw {
                            let idx = p.map[(ty + i) * bw + (tx + j)] as usize;
                            buf[i * ltw + j] = p.colors[idx];
                        }
                    }
                    buf
                }
                None => predict_tu(
                    recon,
                    0,
                    tc,
                    tr,
                    ltw,
                    lth,
                    y_mode as usize,
                    angle_delta_y as i32,
                    filter_intra_mode.map(|f| f as usize),
                ),
            };
            let (q, tt) = if lossless {
                (
                    residual_tx_avail(
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
                        os_w - tx,
                        os_h - ty,
                    ),
                    DCT_DCT as u8,
                )
            } else {
                // r410: §5.11.47 per-TU luma transform-type RD search
                // over the §5.11.48 intra set for this TX size (r424:
                // exact chain pricing through the running fork).
                residual_tx_search_luma_avail(
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
                    os_w - tx,
                    os_h - ty,
                )?
            };
            tu_bd_stamp(&mut recon.bd, 0, tc, tr, ltw, lth);
            residual_quant.push(q);
            luma_tx_types.push(tt);
            tx += ltw;
        }
        ty += lth;
    }

    // --- Chroma on §5.11.5 HasChroma leaves ---
    // r427 general derivation (the §5.11.5 prologue): a `bw4 == 1` /
    // `bh4 == 1` block on a subsampled axis only carries chroma from
    // its odd mi coordinate (the last cell covering the chroma unit);
    // monochrome never has chroma. Under 4:2:0 this reduces to the
    // historical "BLOCK_4X4 needs both coords odd" rule.
    let bw4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
    let bh4 = crate::cdf::NUM_4X4_BLOCKS_HIGH[b_size] as u32;
    let chroma_y_edge = bh4 == 1 && recon.subsampling_y != 0 && (mi_r & 1) == 0;
    let chroma_x_edge = bw4 == 1 && recon.subsampling_x != 0 && (mi_c & 1) == 0;
    let has_chroma = recon.num_planes > 1 && !chroma_y_edge && !chroma_x_edge;
    let mut uv_mode: Option<u8> = None;
    let mut angle_delta_uv = 0i8;
    let mut cfl_alpha: Option<(i8, i8)> = None;
    if has_chroma {
        let (crow0, ccol0) = recon.chroma_origin(mi_r, mi_c);
        // §5.11.35 MaxLumaW / MaxLumaH: the block's own luma extent
        // (the last luma TU coded above ends the block).
        let max_luma_w = col0 + bw;
        let max_luma_h = row0 + bh;
        // §5.11.38 chroma residual block for this leaf.
        let plane_sz = crate::cdf::get_plane_residual_size(
            b_size,
            1,
            recon.subsampling_x,
            recon.subsampling_y,
        )
        .unwrap_or(BLOCK_4X4);
        // §8.3.2 cfl_allowed: on the lossless arm CFL is only allowed
        // when the subsampled chroma residual block is 4×4
        // (`get_plane_residual_size(MiSize, 1) == BLOCK_4X4`); on the
        // lossy arm `Max(Block_Width, Block_Height) <= 32`. r425:
        // this driver additionally keeps CFL off frame-edge-clipped
        // leaves (an encoder choice — the §7.11.5 luma-average region
        // interacts with the visited-TU extent there, and the palette
        // / directional arms carry the clipped shapes).
        let cfl_allowed = if os_w < bw || os_h < bh {
            false
        } else if lossless {
            plane_sz == BLOCK_4X4
        } else {
            bw.max(bh) <= 32
        };
        let chroma_tx = if lossless {
            TX_4X4
        } else {
            get_tx_size(1, luma_tx, b_size, recon.subsampling_x, recon.subsampling_y)
                .unwrap_or(TX_4X4)
        };
        let (ctw, cth) = (TX_WIDTH[chroma_tx], TX_HEIGHT[chroma_tx]);
        // Chroma extent of this leaf's residual grid (§5.11.38
        // subsampled plane size — sub-subsampled shapes round up to
        // the 4×4 floor).
        let cbw = NUM_4X4_BLOCKS_WIDE[plane_sz] * 4;
        let cbh = crate::cdf::NUM_4X4_BLOCKS_HIGH[plane_sz] * 4;
        // §5.11.46 UV palette arm (r418): the coded `uv_mode` is
        // DC_PRED (the UV palette gate) and CFL is off; each chroma
        // TU predicts from the §7.11.4 palette map instead.
        let (m, delta_uv, alpha) = match palette_uv {
            Some(_) => (DC_PRED as u8, 0i8, None),
            None => pick_uv_mode(
                recon,
                input,
                crow0,
                ccol0,
                cbw,
                cbh,
                cfl_allowed,
                bw.min(bh),
                max_luma_w,
                max_luma_h,
            ),
        };
        uv_mode = Some(m);
        angle_delta_uv = delta_uv;
        cfl_alpha = alpha;
        let is_cfl = m as usize == UV_CFL_PRED;
        let chroma_tx_type = chroma_tx_type_for(m, chroma_tx, lossless);
        let cw = recon.chroma_w;
        for plane in 1..usize::from(recon.num_planes) {
            let alpha_p = match (is_cfl, plane, alpha) {
                (true, 1, Some((au, _))) => Some(au),
                (true, 2, Some((_, av))) => Some(av),
                _ => None,
            };
            // r425 — on-screen chroma extent (clipped leaves).
            let os_cw = cbw.min(cw - ccol0);
            let os_ch = cbh.min(recon.chroma_h - crow0);
            let mut ty = 0usize;
            while ty < cbh {
                let mut tx = 0usize;
                while tx < cbw {
                    // §5.11.35 off-screen-origin skip (chroma maxX /
                    // maxY are the subsampled plane bounds).
                    if ty >= os_ch || tx >= os_cw {
                        tx += ctw;
                        continue;
                    }
                    let (tr, tc) = (crow0 + ty, ccol0 + tx);
                    // §5.11.35: a CFL chroma TU writes the DC_PRED
                    // base, then §7.11.5 layers the luma AC on top.
                    let pred = if let Some(p) = palette_uv {
                        // §7.11.4 `predict_palette` over this TU's
                        // footprint on the shared `ColorMapUV`.
                        let colors = if plane == 1 { &p.colors_u } else { &p.colors_v };
                        let mut buf = vec![0u16; ctw * cth];
                        for i in 0..cth {
                            for j in 0..ctw {
                                let idx = p.map[(ty + i) * cbw + (tx + j)] as usize;
                                buf[i * ctw + j] = colors[idx];
                            }
                        }
                        buf
                    } else if is_cfl {
                        let dc = predict_tu(recon, plane, tc, tr, ctw, cth, DC_PRED, 0, None);
                        cfl_layer(
                            &dc,
                            &recon.y,
                            recon.width,
                            tc,
                            tr,
                            ctw,
                            cth,
                            alpha_p.unwrap_or(0),
                            max_luma_w,
                            max_luma_h,
                            recon.subsampling_x,
                            recon.subsampling_y,
                            recon.bit_depth,
                        )
                    } else {
                        predict_tu(
                            recon,
                            plane,
                            tc,
                            tr,
                            ctw,
                            cth,
                            m as usize,
                            delta_uv as i32,
                            None,
                        )
                    };
                    let plane_buf = if plane == 1 {
                        &mut recon.u
                    } else {
                        &mut recon.v
                    };
                    let q = residual_tx_avail(
                        if plane == 1 { &input.u } else { &input.v },
                        plane_buf,
                        cw,
                        tr,
                        tc,
                        chroma_tx,
                        &pred,
                        plane as u8,
                        lossless,
                        chroma_tx_type,
                        &qp,
                        os_cw - tx,
                        os_ch - ty,
                    );
                    tu_bd_stamp(&mut recon.bd, plane, tc, tr, ctw, cth);
                    residual_quant.push(q);
                    tx += ctw;
                }
                ty += cth;
            }
        }
    }

    // §5.11.11 skip: 1 iff every visited TU quantised to zero — the
    // reconstruction is then the bare prediction on every plane (the
    // dequant + inverse of an all-zero Quant[] is exactly zero, so the
    // stitches above already equal pred).
    let all_zero = residual_quant.iter().all(|tu| tu.iter().all(|&q| q == 0));
    let skip = u8::from(all_zero);
    if all_zero {
        residual_quant.clear();
        // §5.11.47 commitments are per-visited-luma-TU on the `!skip`
        // arm only — a skip leaf must commit none.
        luma_tx_types.clear();
    }
    // An all-DCT_DCT vector is the writer's back-compat default —
    // commit the compact empty form then (keeps the lossless arm's
    // emitted bytes identical to r409).
    if luma_tx_types.iter().all(|&t| t == DCT_DCT as u8) {
        luma_tx_types.clear();
    }
    let (cfl_alpha_u, cfl_alpha_v) = match cfl_alpha {
        Some((au, av)) => (Some(au), Some(av)),
        None => (None, None),
    };
    // §5.11.46 / §5.11.49 palette commitments (r418).
    let mut palette = crate::encoder::partition_tree::SyntaxPalette::default();
    if let Some(p) = palette_y {
        palette.size_y = p.colors.len() as u8;
        palette.colors_y[..p.colors.len()].copy_from_slice(&p.colors);
        palette.color_map_y = p.map.clone();
    }
    if let Some(p) = palette_uv {
        palette.size_uv = p.colors_u.len() as u8;
        palette.colors_u[..p.colors_u.len()].copy_from_slice(&p.colors_u);
        palette.colors_v[..p.colors_v.len()].copy_from_slice(&p.colors_v);
        // §5.11.46 V-plane arm: direct literals (r418 scope; the
        // signed-delta arm is an encoder-choice follow-up).
        palette.delta_encode_v = false;
        palette.color_map_uv = p.map.clone();
    }

    Ok(SyntaxBlock {
        skip,
        segment_id: 0,
        cdef_idx: 0,
        reduced_delta_q_index: 0,
        reduced_delta_lf: [0; crate::cdf::FRAME_LF_COUNT],
        intrabc_mv: None,
        y_mode,
        uv_mode,
        angle_delta_y,
        angle_delta_uv,
        cfl_alpha_u,
        cfl_alpha_v,
        use_filter_intra: u8::from(filter_intra_mode.is_some()),
        filter_intra_mode,
        palette,
        residual_quant,
        // §5.11.15 TxSize commitment: on the lossy TX_MODE_SELECT arm
        // the tx_depth S() fires for every MiSize > BLOCK_4X4 block
        // (intra ⇒ allowSelect); lossless / BLOCK_4X4 stay on the
        // spec-forced default (`None`).
        tx_size: if !lossless && b_size > BLOCK_4X4 {
            Some(luma_tx as u8)
        } else {
            None
        },
        residual_tx_type: luma_tx_types,
        var_tx_trees: Vec::new(),
        inter: None,
    })
}

// ---------------------------------------------------------------------
// Rate-distortion partition search.
// ---------------------------------------------------------------------

/// Snapshot of the pixel region one square node covers (its luma
/// square + the collocated chroma squares) plus the §6.10.3
/// `BlockDecoded[]` mirror — the working set a §5.11.4 partition
/// trial saves/restores.
pub(crate) struct RegionSnapshot {
    w: usize,
    h: usize,
    cw: usize,
    ch: usize,
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
    bd: BlockDecodedMirror,
}

pub(crate) fn save_region(recon: &ReconState, r: u32, c: u32, n4: usize) -> RegionSnapshot {
    save_region_wh(recon, r, c, n4, n4)
}

/// r412 — rectangular twin of [`save_region`] (`w4 × h4` mi units;
/// the square helper delegates here). Chroma rides the same
/// `(r >> 1, c >> 1)` origin derivation, so rect callers keep their
/// mi origins even on both axes (the P-frame HORZ/VERT halves are
/// BLOCK_16X8 and larger, whose half offsets are even).
pub(crate) fn save_region_wh(
    recon: &ReconState,
    r: u32,
    c: u32,
    w4: usize,
    h4: usize,
) -> RegionSnapshot {
    // r425 — frame-edge-straddling nodes snapshot the on-screen
    // intersection only (there is nothing else to save or restore).
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = recon.chroma_origin(r, c);
    let w = (w4 * 4).min(recon.width - col0);
    let h = (h4 * 4).min(recon.height - row0);
    // r427 — chroma extents at the subsampled shape; monochrome
    // snapshots no chroma.
    let (cw, ch) = if recon.num_planes > 1 {
        (
            ((w4 * 4) >> recon.subsampling_x).min(recon.chroma_w - ccol0),
            ((h4 * 4) >> recon.subsampling_y).min(recon.chroma_h - crow0),
        )
    } else {
        (0, 0)
    };
    let mut y = vec![0u16; w * h];
    let mut u = vec![0u16; cw * ch];
    let mut v = vec![0u16; cw * ch];
    for i in 0..h {
        y[i * w..(i + 1) * w].copy_from_slice(&recon.y[(row0 + i) * recon.width + col0..][..w]);
    }
    for i in 0..ch {
        u[i * cw..(i + 1) * cw]
            .copy_from_slice(&recon.u[(crow0 + i) * recon.chroma_w + ccol0..][..cw]);
        v[i * cw..(i + 1) * cw]
            .copy_from_slice(&recon.v[(crow0 + i) * recon.chroma_w + ccol0..][..cw]);
    }
    RegionSnapshot {
        w,
        h,
        cw,
        ch,
        y,
        u,
        v,
        bd: recon.bd.clone(),
    }
}

pub(crate) fn restore_region(recon: &mut ReconState, r: u32, c: u32, snap: &RegionSnapshot) {
    let (w, h) = (snap.w, snap.h);
    let (cw, ch) = (snap.cw, snap.ch);
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = recon.chroma_origin(r, c);
    for i in 0..h {
        recon.y[(row0 + i) * recon.width + col0..][..w].copy_from_slice(&snap.y[i * w..][..w]);
    }
    for i in 0..ch {
        recon.u[(crow0 + i) * recon.chroma_w + ccol0..][..cw]
            .copy_from_slice(&snap.u[i * cw..][..cw]);
        recon.v[(crow0 + i) * recon.chroma_w + ccol0..][..cw]
            .copy_from_slice(&snap.v[i * cw..][..cw]);
    }
    recon.bd = snap.bd.clone();
}

/// Distortion (SSD, luma + both chroma cells) of the current
/// reconstruction against the input over one square node's region.
pub(crate) fn region_distortion(
    recon: &ReconState,
    input: &YuvFrame,
    r: u32,
    c: u32,
    n4: usize,
) -> u64 {
    region_distortion_wh(recon, input, r, c, n4, n4)
}

/// r412 — rectangular twin of [`region_distortion`].
pub(crate) fn region_distortion_wh(
    recon: &ReconState,
    input: &YuvFrame,
    r: u32,
    c: u32,
    w4: usize,
    h4: usize,
) -> u64 {
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = recon.chroma_origin(r, c);
    // r425 — straddling nodes score their on-screen intersection.
    let w = (w4 * 4).min(recon.width - col0);
    let h = (h4 * 4).min(recon.height - row0);
    let (cw, ch) = if recon.num_planes > 1 {
        (
            ((w4 * 4) >> recon.subsampling_x).min(recon.chroma_w - ccol0),
            ((h4 * 4) >> recon.subsampling_y).min(recon.chroma_h - crow0),
        )
    } else {
        (0, 0)
    };
    let mut ssd = 0u64;
    for i in 0..h {
        for j in 0..w {
            let idx = (row0 + i) * recon.width + (col0 + j);
            let d = i64::from(recon.y[idx]) - i64::from(input.y[idx]);
            ssd += (d * d) as u64;
        }
    }
    for i in 0..ch {
        for j in 0..cw {
            let idx = (crow0 + i) * recon.chroma_w + (ccol0 + j);
            let du = i64::from(recon.u[idx]) - i64::from(input.u[idx]);
            let dv = i64::from(recon.v[idx]) - i64::from(input.v[idx]);
            ssd += (du * du + dv * dv) as u64;
        }
    }
    ssd
}

/// r427 — §5.11.38 partition-admissibility gate: "It is a
/// requirement of bitstream conformance that
/// `get_plane_residual_size( subSize, 1 )` is not equal to
/// BLOCK_INVALID every time subSize is computed" (the note: UV blocks
/// must keep aspect ratios within 1:4..4:1 — under 4:2:2 every tall
/// luma shape like 32×64 or 4×16 subsamples to an out-of-range chroma
/// shape). The search must never elect such a subSize; monochrome
/// streams compute no chroma size and are unconstrained.
pub(crate) fn chroma_partition_ok(recon: &ReconState, sub_size: usize) -> bool {
    recon.num_planes == 1
        || crate::cdf::get_plane_residual_size(
            sub_size,
            1,
            recon.subsampling_x,
            recon.subsampling_y,
        )
        .is_some()
}

/// q-scaled Lagrange multiplier for the `D + λ·R` decisions. r427:
/// the SSD distortions scale with `2^(2·(BitDepth - 8))` at matched
/// relative error (and so do the squared §7.12.2 step sizes for a
/// given `q_index`), so λ carries the same factor — 8-bit behaviour
/// is bit-identical to the historical constant.
pub(crate) fn lambda_for(qp: &QuantizerParams) -> u64 {
    let l8 = 1 + (qp.base_q_idx as u64 * qp.base_q_idx as u64) / 32;
    l8 << (2 * u32::from(qp.bit_depth.saturating_sub(8)))
}

/// Crude rate proxy for one leaf: a fixed per-leaf mode/skip cost
/// plus a magnitude-aware per-nonzero-coefficient cost
/// (`3 + bitlength(|q|)` roughly tracks the §5.11.39 base + BR +
/// golomb growth). Deliberately simple — it only has to ORDER the
/// §5.11.4 candidates consistently.
pub(crate) fn leaf_rate(block: &SyntaxBlock) -> u64 {
    let mut rate = 24u64;
    for tu in &block.residual_quant {
        for &q in tu {
            if q != 0 {
                rate += 3 + u64::from(32 - q.unsigned_abs().leading_zeros());
            }
        }
    }
    // §5.11.7 intra-block-copy proxy (r418): the `use_intrabc` S()
    // plus an MV-magnitude term for the §5.11.31 `read_mv` cost.
    if let Some(mv) = block.intrabc_mv {
        let mag = |v: i32| u64::from(32 - (v.unsigned_abs() / 8 + 1).leading_zeros());
        rate += 6 + mag(mv[0]) + mag(mv[1]);
    }
    // §5.11.46 / §5.11.49 palette proxy (r418): colour entries at
    // roughly a literal each, plus a transition-weighted map term —
    // the §5.11.49 context coding makes runs nearly free while index
    // changes cost real bits.
    for (size, map) in [
        (block.palette.size_y, &block.palette.color_map_y),
        (block.palette.size_uv, &block.palette.color_map_uv),
    ] {
        if size > 0 {
            rate += 10 + 8 * u64::from(size);
            let mut transitions = 0u64;
            for w in map.windows(2) {
                if w[0] != w[1] {
                    transitions += 1;
                }
            }
            rate += map.len() as u64 / 16 + transitions * 2;
        }
    }
    rate
}

/// Recursive rate proxy over a candidate subtree (each split node adds
/// a small partition-symbol weight).
pub(crate) fn tree_rate(node: &SyntaxNode) -> u64 {
    match node {
        SyntaxNode::Leaf(b) => leaf_rate(b),
        SyntaxNode::Split(children) => 4 + children.iter().map(|c| tree_rate(c)).sum::<u64>(),
        // r412/r413: asymmetric partitions are P-frame-search
        // territory — the KEY driver never builds them.
        SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
            4 + blocks.iter().map(|b| leaf_rate(b)).sum::<u64>()
        }
        SyntaxNode::HorzA(blocks)
        | SyntaxNode::HorzB(blocks)
        | SyntaxNode::VertA(blocks)
        | SyntaxNode::VertB(blocks) => 5 + blocks.iter().map(|b| leaf_rate(b)).sum::<u64>(),
        SyntaxNode::Horz4(blocks) | SyntaxNode::Vert4(blocks) => {
            5 + blocks.iter().map(|b| leaf_rate(b)).sum::<u64>()
        }
    }
}

/// r426 — the KEY frame's exactness-demand configuration: the
/// row-major mi-cell demand mask plus the (validated) lossless
/// segment every demanded leaf commits. On intra frames the §5.11.8
/// `intra_segment_id` is coded for EVERY block — no skip-pred
/// cascade — so a demanded leaf always carries the lossless segment
/// explicitly.
pub(crate) struct KeySegDemand<'a> {
    pub(crate) mask: &'a [bool],
    pub(crate) ll_seg: u8,
}

/// r426 — one KEY-frame leaf with per-segment lossless awareness:
/// a leaf overlapping the demand mask codes on the lossless segment
/// (segment id committed on every candidate BEFORE pricing; the KEY
/// driver's `recon.qp` / `recon.lossless` swapped to the qindex-0
/// configuration for the call), everything else keeps the segment-0
/// frame-quantiser shape.
fn encode_key_leaf(
    r: u32,
    c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    pricing: Option<(&RateTwin, &SyntaxFrameParams)>,
    seg: Option<&KeySegDemand<'_>>,
) -> Result<SyntaxBlock, Error> {
    let demanded = seg.is_some_and(|d| {
        let bh4 = crate::cdf::NUM_4X4_BLOCKS_HIGH[b_size] as u32;
        let bw4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
        let r1 = (r + bh4).min(recon.mi_rows);
        let c1 = (c + bw4).min(recon.mi_cols);
        (r..r1).any(|rr| (c..c1).any(|cc| d.mask[(rr * recon.mi_cols + cc) as usize]))
    });
    if !demanded {
        return encode_leaf_sq(r, c, b_size, input, recon, pricing);
    }
    let ll_seg = seg.expect("demanded implies seg").ll_seg;
    let saved_qp = recon.qp;
    let saved_lossless = recon.lossless;
    recon.qp = QuantizerParams::neutral(0, recon.bit_depth);
    recon.lossless = true;
    let out = encode_leaf_sq_seg(r, c, b_size, input, recon, pricing, Some(ll_seg));
    recon.qp = saved_qp;
    recon.lossless = saved_lossless;
    out
}

/// Recursive §5.11.4 rate-distortion partition search. At every
/// fully-in-frame square node from `BLOCK_64X64` down to `BLOCK_8X8`,
/// trial-encode the node both as one `PARTITION_NONE` leaf and as a
/// `PARTITION_SPLIT` of four recursively-searched quadrants against
/// the same starting reconstruction (+ `BlockDecoded[]`) state, and
/// keep the lower `D + λ·R` score (ties prefer the leaf — fewer
/// symbols). Nodes straddling the frame edge take the §5.11.4 forced
/// split; out-of-frame quadrants get the short-circuited dummy node.
/// The running reconstruction ends in the winning shape's state.
///
/// r421 — rate is priced by the search-side rate twin: `twin` enters
/// holding the live write state at this node's stream position, every
/// candidate is priced by count-writing its exact symbol sequence
/// through the real write path, and the twin leaves committed to the
/// WINNING shape's symbols (so later nodes price against the adapted
/// CDFs the writer will actually hold). Under
/// [`RateModel::Heuristic`] the twin is still threaded (commits are
/// how the state stays consistent) but scores use the pre-r421
/// proxies — the sweep harnesses A/B the two models. Returns the
/// chosen node plus its exact cost in 1/256-bit units.
#[allow(clippy::too_many_arguments)]
fn build_search_tree(
    r: u32,
    c: u32,
    b_size: usize,
    input: &YuvFrame,
    recon: &mut ReconState,
    twin: &mut RateTwin,
    params: &SyntaxFrameParams,
    model: RateModel,
    seg: Option<&KeySegDemand<'_>>,
) -> Result<(SyntaxNode, u64), Error> {
    if r >= recon.mi_rows || c >= recon.mi_cols {
        // §5.11.4 line 1 — never inspected by the writer (and never
        // costed: the write pass short-circuits before any symbol).
        return Ok((SyntaxNode::dummy_oob(), 0));
    }
    if b_size == BLOCK_4X4 {
        let pricing = (model == RateModel::Twin).then_some((&*twin, params));
        let leaf = encode_key_leaf(r, c, b_size, input, recon, pricing, seg)?;
        let node = SyntaxNode::Leaf(Box::new(leaf));
        let cost = twin.commit_subtree(&node, r, c, b_size, params)?;
        return Ok((node, cost));
    }
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
    let half = n4 >> 1;
    let sub = crate::cdf::partition_subsize(crate::cdf::PARTITION_SPLIT, b_size)
        .expect("PARTITION_SPLIT subsize exists for every b_size >= BLOCK_8X8");
    let fully_inside = r + n4 <= recon.mi_rows && c + n4 <= recon.mi_cols;

    if !fully_inside {
        // §5.11.4 edge arms: a node straddling the frame edge cannot
        // code PARTITION_NONE. The half-straddle cases carry a real
        // choice — `split_or_horz` (hasCols && !hasRows) admits a
        // single clipped HORZ top block, `split_or_vert` (hasRows &&
        // !hasCols) a single clipped VERT left block — which r425
        // elects against the recursive SPLIT under the same scoring
        // as the interior ladder. The bottom-right corner stays the
        // forced no-symbol SPLIT. BLOCK_8X8 keeps the pre-r425 forced
        // split (its rect subs are the sub-8-chroma shapes).
        let has_rows = r + half < recon.mi_rows;
        let has_cols = c + half < recon.mi_cols;
        let lambda = lambda_for(&recon.qp);
        // Twin-model only: the heuristic arm stays the pre-r425
        // forced-split baseline the A/B harnesses measure against.
        let rect_part = if model != RateModel::Twin {
            None
        } else if b_size > BLOCK_8X8 && has_cols && !has_rows {
            Some(crate::cdf::PARTITION_HORZ)
        } else if b_size > BLOCK_8X8 && has_rows && !has_cols {
            Some(crate::cdf::PARTITION_VERT)
        } else {
            None
        };
        let before = save_region_wh(recon, r, c, n4 as usize, n4 as usize);
        let mut rect_best: Option<(SyntaxNode, RegionSnapshot, u64, RateTwin, u64)> = None;
        if let Some(part) = rect_part {
            if let Some(psub) = crate::cdf::partition_subsize(part, b_size)
                .filter(|&p| chroma_partition_ok(recon, p))
            {
                let mut twin_s = twin.clone();
                let mut cost_s = twin_s.commit_partition_symbol(part, r, c, b_size)?;
                let pricing_s = (model == RateModel::Twin).then_some((&twin_s, params));
                let blk = encode_key_leaf(r, c, psub, input, recon, pricing_s, seg)?;
                let h_rate = 4 + leaf_rate(&blk);
                cost_s += twin_s.commit_block(&blk, r, c, psub, params)?;
                let d = region_distortion_wh(recon, input, r, c, n4 as usize, n4 as usize);
                // §5.11.4: the second block of the pair is never
                // coded on the edge arm — the writer ignores the
                // placeholder entry.
                let placeholder = Box::new(SyntaxBlock::skip_leaf(0, None));
                let node = if part == crate::cdf::PARTITION_HORZ {
                    SyntaxNode::Horz([Box::new(blk), placeholder])
                } else {
                    SyntaxNode::Vert([Box::new(blk), placeholder])
                };
                let score = match model {
                    RateModel::Twin => score256(d, lambda, cost_s),
                    RateModel::Heuristic => score256(d, lambda, h_rate * 256),
                };
                rect_best = Some((
                    node,
                    save_region_wh(recon, r, c, n4 as usize, n4 as usize),
                    score,
                    twin_s,
                    cost_s,
                ));
                restore_region(recon, r, c, &before);
            }
        }
        // SPLIT arm (forced on the corner case; elected otherwise).
        let mut twin_b = twin.clone();
        let mut cost = twin_b.commit_partition_symbol(crate::cdf::PARTITION_SPLIT, r, c, b_size)?;
        let (nw, c0) = build_search_tree(r, c, sub, input, recon, &mut twin_b, params, model, seg)?;
        let (ne, c1) = build_search_tree(
            r,
            c + half,
            sub,
            input,
            recon,
            &mut twin_b,
            params,
            model,
            seg,
        )?;
        let (sw, c2) = build_search_tree(
            r + half,
            c,
            sub,
            input,
            recon,
            &mut twin_b,
            params,
            model,
            seg,
        )?;
        let (se, c3) = build_search_tree(
            r + half,
            c + half,
            sub,
            input,
            recon,
            &mut twin_b,
            params,
            model,
            seg,
        )?;
        cost += c0 + c1 + c2 + c3;
        let children = [Box::new(nw), Box::new(ne), Box::new(sw), Box::new(se)];
        if let Some((node, after, rect_score, twin_s, rect_cost)) = rect_best {
            let d_b = region_distortion_wh(recon, input, r, c, n4 as usize, n4 as usize);
            let r_b = match model {
                RateModel::Twin => cost,
                RateModel::Heuristic => {
                    (children.iter().map(|ch| tree_rate(ch)).sum::<u64>() + 4) * 256
                }
            };
            if rect_score <= score256(d_b, lambda, r_b) {
                restore_region(recon, r, c, &after);
                *twin = twin_s;
                return Ok((node, rect_cost));
            }
        }
        *twin = twin_b;
        return Ok((SyntaxNode::Split(children), cost));
    }

    let lambda = lambda_for(&recon.qp);
    let before = save_region(recon, r, c, n4 as usize);

    // Candidate A: one PARTITION_NONE leaf (partition symbol + block
    // syntax committed to a twin fork).
    let pricing = (model == RateModel::Twin).then_some((&*twin, params));
    let leaf = encode_key_leaf(r, c, b_size, input, recon, pricing, seg)?;
    let node_a = SyntaxNode::Leaf(Box::new(leaf));
    let d_a = region_distortion(recon, input, r, c, n4 as usize);
    let mut twin_a = twin.clone();
    let cost_a = twin_a.commit_subtree(&node_a, r, c, b_size, params)?;
    let after_a = save_region(recon, r, c, n4 as usize);
    restore_region(recon, r, c, &before);
    let score_a = {
        let r_a = match model {
            RateModel::Twin => cost_a,
            RateModel::Heuristic => {
                let lr = match &node_a {
                    SyntaxNode::Leaf(b) => leaf_rate(b),
                    _ => unreachable!("candidate A is always a leaf"),
                };
                lr * 256
            }
        };
        score256(d_a, lambda, r_a)
    };
    // Running best over the non-split candidates (ties prefer the
    // earlier candidate — fewer coded blocks).
    let mut best: (SyntaxNode, RegionSnapshot, u64, RateTwin, u64) =
        (node_a, after_a, score_a, twin_a, cost_a);

    // Candidates A2/A3 (r425): §5.11.4 PARTITION_HORZ / PARTITION_VERT
    // with two INTRA leaves — the rectangular §5.11.46 palette /
    // §7.11.2 shapes. `BLOCK_16X16` and larger (the 8×8 HORZ / VERT
    // arms code 8×4 / 4×8 leaves, whose sub-8 chroma pairing stays
    // out of this driver's scope); Twin model only (the heuristic arm
    // stays the pre-r425 NONE/SPLIT baseline the A/B harnesses
    // measure against). Each shape's second leaf searches against the
    // first leaf's committed twin state, exactly like the emitting
    // pass.
    if model == RateModel::Twin && b_size >= crate::cdf::BLOCK_16X16 {
        for part in [crate::cdf::PARTITION_HORZ, crate::cdf::PARTITION_VERT] {
            let Some(psub) = crate::cdf::partition_subsize(part, b_size) else {
                continue;
            };
            // r427 — §5.11.38 admissibility (4:2:2 forbids the tall
            // rect shapes; see `chroma_partition_ok`).
            if !chroma_partition_ok(recon, psub) {
                continue;
            }
            let cells: [(u32, u32); 2] = if part == crate::cdf::PARTITION_HORZ {
                [(r, c), (r + half, c)]
            } else {
                [(r, c), (r, c + half)]
            };
            let mut twin_s = twin.clone();
            let mut cost_s = twin_s.commit_partition_symbol(part, r, c, b_size)?;
            let mut h_rate = 4u64;
            let mut blocks: Vec<SyntaxBlock> = Vec::with_capacity(2);
            for &(rr, cc) in &cells {
                let pricing_s = (model == RateModel::Twin).then_some((&twin_s, params));
                let blk = encode_key_leaf(rr, cc, psub, input, recon, pricing_s, seg)?;
                h_rate += leaf_rate(&blk);
                cost_s += twin_s.commit_block(&blk, rr, cc, psub, params)?;
                blocks.push(blk);
            }
            let d = region_distortion(recon, input, r, c, n4 as usize);
            let mut it = blocks.into_iter();
            let pair = [
                Box::new(it.next().expect("two leaves")),
                Box::new(it.next().expect("two leaves")),
            ];
            let node = if part == crate::cdf::PARTITION_HORZ {
                SyntaxNode::Horz(pair)
            } else {
                SyntaxNode::Vert(pair)
            };
            let score = match model {
                RateModel::Twin => score256(d, lambda, cost_s),
                RateModel::Heuristic => score256(d, lambda, h_rate * 256),
            };
            if score < best.2 {
                best = (
                    node,
                    save_region(recon, r, c, n4 as usize),
                    score,
                    twin_s,
                    cost_s,
                );
            }
            restore_region(recon, r, c, &before);
        }
    }

    // Candidate B: PARTITION_SPLIT into four recursively-searched
    // quadrants (NW/NE/SW/SE dispatch order — the writer's order),
    // each child searched against the twin state its symbols will
    // actually be written under (SPLIT arm + earlier siblings).
    let mut twin_b = twin.clone();
    let mut cost_b = twin_b.commit_partition_symbol(crate::cdf::PARTITION_SPLIT, r, c, b_size)?;
    let (nw, c0) = build_search_tree(r, c, sub, input, recon, &mut twin_b, params, model, seg)?;
    let (ne, c1) = build_search_tree(
        r,
        c + half,
        sub,
        input,
        recon,
        &mut twin_b,
        params,
        model,
        seg,
    )?;
    let (sw, c2) = build_search_tree(
        r + half,
        c,
        sub,
        input,
        recon,
        &mut twin_b,
        params,
        model,
        seg,
    )?;
    let (se, c3) = build_search_tree(
        r + half,
        c + half,
        sub,
        input,
        recon,
        &mut twin_b,
        params,
        model,
        seg,
    )?;
    cost_b += c0 + c1 + c2 + c3;
    let children = [Box::new(nw), Box::new(ne), Box::new(sw), Box::new(se)];
    let d_b = region_distortion(recon, input, r, c, n4 as usize);

    let r_b = match model {
        RateModel::Twin => cost_b,
        // Heuristic scores scaled by 256 so both models run through
        // the same `score256` comparison with identical decisions to
        // the pre-r421 integer scores.
        RateModel::Heuristic => (children.iter().map(|ch| tree_rate(ch)).sum::<u64>() + 4) * 256,
    };
    let score_b = score256(d_b, lambda, r_b);
    if best.2 <= score_b {
        let (node, after, _, twin_best, cost) = best;
        restore_region(recon, r, c, &after);
        *twin = twin_best;
        Ok((node, cost))
    } else {
        *twin = twin_b;
        Ok((SyntaxNode::Split(children), cost_b))
    }
}

// ---------------------------------------------------------------------
// r418 §5.11.46 palette-election witnesses.
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::PALETTE_COLORS;

    /// 8-bit view of a `u16` recon plane for byte-comparisons against
    /// the spec driver's 8-bit output surface.
    fn plane8(p: &[u16]) -> Vec<u8> {
        p.iter().map(|&s| s as u8).collect()
    }

    /// r421 — fresh-frame search context (rate twin + frame params)
    /// for driving [`build_search_tree`] directly: §8.3.1 default
    /// CDFs at the recon's quantiser, a whole-frame single-tile write
    /// state, and the KEY driver's frame-scope parameter bundle.
    fn search_ctx_for_tests(recon: &ReconState) -> (RateTwin, SyntaxFrameParams) {
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
            segmentation_enabled: false,
            seg_ref_frame: [None; crate::uncompressed_header_tail::MAX_SEGMENTS],
            seg_skip: [false; crate::uncompressed_header_tail::MAX_SEGMENTS],
            seg_globalmv: [false; crate::uncompressed_header_tail::MAX_SEGMENTS],
            last_active_seg_id: 0,
            lossless_array: [recon.lossless; crate::uncompressed_header_tail::MAX_SEGMENTS],
            coded_lossless: recon.lossless,
            enable_cdef: false,
            allow_intrabc: recon.allow_intrabc,
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
            inter: None,
        };
        (RateTwin::snapshot(&cdfs, &state, &writer), params)
    }

    /// Deterministic 4-colour dither over the luma plane (xorshift cell
    /// noise) with flat chroma — exactly representable by a §5.11.46
    /// palette on every square leaf, while every §7.11.2 neighbour
    /// mode leaves a large residual.
    fn dither4(w: u32, h: u32, seed: u32) -> YuvFrame {
        const COLORS: [u16; 4] = [16, 80, 160, 240];
        let (wu, hu) = (w as usize, h as usize);
        let mut f = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        let mut state = 0x1234_5678u32 ^ seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for i in 0..hu {
            for j in 0..wu {
                f.y[i * wu + j] = COLORS[(next() & 3) as usize];
            }
        }
        // Chroma: per-2x2-cell checker over two (U, V) pairs — at most
        // four distinct joint pairs per chroma block, sharp cell edges
        // no §7.11.2 mode predicts.
        let (cw, ch) = (wu / 2, hu / 2);
        for i in 0..ch {
            for j in 0..cw {
                let par = ((i / 2) + (j / 2)) & 1;
                f.u[i * cw + j] = if par == 0 { 96 } else { 168 };
                f.v[i * cw + j] = if ((i / 4) & 1) == par { 108 } else { 152 };
            }
        }
        f
    }

    /// r421 — end-to-end rate-twin witness: replicate the driver loop
    /// over a multi-superblock textured frame, sum the per-superblock
    /// committed twin costs, and (a) assert the twin state matches the
    /// live writer state after every superblock (the anti-desync
    /// invariant, here as a hard assert), (b) assert the summed exact
    /// cost predicts the emitted tile payload to within the §8.2.4
    /// termination slack (15 seed bits − 14 dropped trailing zeros ±
    /// byte padding ± 1 fractional bit).
    #[test]
    fn r421_twin_costs_sum_to_writer_emission() {
        let input = dither4(128, 64, 77);
        let (mi_rows, mi_cols) = (16u32, 32u32);
        let mut recon = ReconState {
            y: vec![0u16; 128 * 64],
            u: vec![0u16; 64 * 32],
            v: vec![0u16; 64 * 32],
            width: 128,
            height: 64,
            chroma_w: 64,
            chroma_h: 32,
            mi_rows,
            mi_cols,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(60, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        let mut writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.init_coeff_cdfs(60);
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
        .unwrap();
        // Params must mirror the twin's exactly (same object).
        let (_, params) = search_ctx_for_tests(&recon);
        let mut total_cost256 = 0u64;
        for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
            recon.bd.clear_for_sb(sb_r, sb_c, mi_rows, mi_cols);
            state.arm_read_deltas();
            let mut twin = RateTwin::snapshot(&cdfs, &state, &writer);
            twin.arm_read_deltas();
            let (tree, cost) = build_search_tree(
                sb_r,
                sb_c,
                BLOCK_64X64,
                &input,
                &mut recon,
                &mut twin,
                &params,
                RateModel::Twin,
                None,
            )
            .unwrap();
            total_cost256 += cost;
            write_partition_tree_syntax(
                &mut writer,
                &mut cdfs,
                &mut state,
                &tree,
                sb_r,
                sb_c,
                BLOCK_64X64,
                &params,
            )
            .unwrap();
            assert!(
                twin.matches(&cdfs, &writer),
                "rate twin desynced after superblock ({sb_r},{sb_c})"
            );
        }
        let tile_bits = writer.finish().len() as u64 * 8;
        let predicted_bits = total_cost256 / 256;
        // Emission = 15 seed bits + renorm bits − 14 dropped trailing
        // zeros, padded up to a byte: |emitted − predicted| ≤ 1 + 7 +
        // 1 (fractional accumulation) bits — use 16 for slack.
        assert!(
            tile_bits.abs_diff(predicted_bits) <= 16,
            "twin predicted {predicted_bits} bits, tile payload is {tile_bits} bits"
        );
        // The prediction is meaningful (a real multi-superblock lossy
        // payload, thousands of bits).
        assert!(predicted_bits > 2000, "payload unexpectedly trivial");
    }

    fn count_palette_leaves(node: &SyntaxNode, pal: &mut u32, other: &mut u32) {
        count_palette_leaves_by(node, pal, other, |b| b.palette.size_y > 0)
    }

    fn count_palette_uv_leaves(node: &SyntaxNode, pal: &mut u32, other: &mut u32) {
        count_palette_leaves_by(node, pal, other, |b| b.palette.size_uv > 0)
    }

    fn count_palette_leaves_by(
        node: &SyntaxNode,
        pal: &mut u32,
        other: &mut u32,
        pred: fn(&SyntaxBlock) -> bool,
    ) {
        let leafc = |b: &SyntaxBlock, pal: &mut u32, other: &mut u32| {
            if pred(b) {
                *pal += 1;
            } else {
                *other += 1;
            }
        };
        match node {
            SyntaxNode::Leaf(b) => leafc(b, pal, other),
            SyntaxNode::Split(children) => {
                for ch in children.iter() {
                    count_palette_leaves_by(ch, pal, other, pred);
                }
            }
            rest => {
                for b in rest.asymmetric_blocks().iter() {
                    leafc(b, pal, other);
                }
            }
        }
    }

    /// The §5.11.46 luma palette arm must actually be SELECTED where a
    /// palette predicts exactly and every §7.11.2 mode misses: the
    /// search tree over 4-colour dither content commits
    /// `PaletteSizeY > 0` leaves, the committed colour lists are the
    /// §5.11.46 canonical (strictly ascending) form, and the emitted
    /// stream round-trips byte-exact through the spec driver — on the
    /// lossy AND the lossless arm.
    #[test]
    fn r418_search_selects_luma_palette_on_dither_content() {
        for q in [60u8, 0] {
            let input = dither4(64, 64, 40 + u32::from(q));
            let mut recon = ReconState {
                y: vec![0u16; 64 * 64],
                u: vec![0u16; 32 * 32],
                v: vec![0u16; 32 * 32],
                width: 64,
                height: 64,
                chroma_w: 32,
                chroma_h: 32,
                mi_rows: 16,
                mi_cols: 16,
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
            };
            recon.bd.clear_for_sb(0, 0, 16, 16);
            let (mut twin, params) = search_ctx_for_tests(&recon);
            let (tree, _) = build_search_tree(
                0,
                0,
                BLOCK_64X64,
                &input,
                &mut recon,
                &mut twin,
                &params,
                RateModel::Twin,
                None,
            )
            .unwrap();
            let (mut pal, mut other) = (0u32, 0u32);
            count_palette_leaves(&tree, &mut pal, &mut other);
            assert!(
                pal > 0,
                "q={q}: 4-colour dither content must commit PaletteSizeY > 0 leaves \
                 (got PALETTE={pal} OTHER={other})"
            );
            let (mut pal_uv, mut other_uv) = (0u32, 0u32);
            count_palette_uv_leaves(&tree, &mut pal_uv, &mut other_uv);
            assert!(
                pal_uv > 0,
                "q={q}: 2x2-cell chroma checker content must commit PaletteSizeUV > 0 \
                 leaves (got PALETTE_UV={pal_uv} OTHER={other_uv})"
            );
            fn check_sorted(node: &SyntaxNode) {
                let leafc = |b: &SyntaxBlock| {
                    let k = b.palette.size_y as usize;
                    if k > 0 {
                        assert!((2..=PALETTE_COLORS).contains(&k));
                        for i in 1..k {
                            assert!(
                                b.palette.colors_y[i] > b.palette.colors_y[i - 1],
                                "§5.11.46 canonical form: colours strictly ascending"
                            );
                        }
                        assert!(b.palette.color_map_y.iter().all(|&m| (m as usize) < k));
                    }
                    let kuv = b.palette.size_uv as usize;
                    if kuv > 0 {
                        assert!((2..=PALETTE_COLORS).contains(&kuv));
                        for i in 1..kuv {
                            assert!(
                                b.palette.colors_u[i] >= b.palette.colors_u[i - 1],
                                "§5.11.46 canonical form: U colours non-strictly ascending"
                            );
                        }
                        assert!(b.palette.color_map_uv.iter().all(|&m| (m as usize) < kuv));
                    }
                };
                match node {
                    SyntaxNode::Leaf(b) => leafc(b),
                    SyntaxNode::Split(children) => children.iter().for_each(|ch| check_sorted(ch)),
                    rest => rest.asymmetric_blocks().iter().for_each(|b| leafc(b)),
                }
            }
            check_sorted(&tree);

            // Full-stream conformance through the spec driver.
            let enc = encode_key_frame_yuv_with_q(&input, q).unwrap();
            let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "q={q}: luma");
            assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "q={q}: U");
            assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "q={q}: V");
            if q == 0 {
                assert_eq!(enc.recon_y, input.y, "lossless palette arm must be exact");
            }
        }
    }

    /// r424 — the §5.11.46 signed-delta V-plane arm must actually be
    /// ELECTED where it is cheaper: chroma content whose UV-palette V
    /// entries cluster tightly (delta 6 fits the 4-bit minimum-width
    /// delta chain; the direct literal costs the full 8 bits per
    /// entry) must commit `delta_encode_palette_colors_v = 1` on the
    /// UV-palette leaves under exact twin pricing, and the emitted
    /// stream must still round-trip byte-exact through the spec
    /// driver's §5.11.46 delta-arm reader.
    #[test]
    fn r424_search_elects_signed_delta_v_palette_arm() {
        // Luma: 4-colour dither (palette-friendly). Chroma: 2x2-cell
        // checker with a WIDE U spread and a TIGHT V spread.
        let mut input = dither4(64, 64, 91);
        for i in 0..32usize {
            for j in 0..32usize {
                let cell = ((i / 2) + (j / 2)) & 1;
                input.u[i * 32 + j] = if cell == 0 { 64 } else { 192 };
                input.v[i * 32 + j] = if cell == 0 { 100 } else { 106 };
            }
        }
        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows: 16,
            mi_cols: 16,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(60, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        recon.bd.clear_for_sb(0, 0, 16, 16);
        let (mut twin, params) = search_ctx_for_tests(&recon);
        let (tree, _) = build_search_tree(
            0,
            0,
            BLOCK_64X64,
            &input,
            &mut recon,
            &mut twin,
            &params,
            RateModel::Twin,
            None,
        )
        .unwrap();
        let (mut pal_uv, mut other_uv) = (0u32, 0u32);
        count_palette_uv_leaves(&tree, &mut pal_uv, &mut other_uv);
        assert!(pal_uv > 0, "the checker chroma must commit UV palettes");
        let (mut delta_v, mut literal_v) = (0u32, 0u32);
        count_palette_leaves_by(&tree, &mut delta_v, &mut literal_v, |b| {
            b.palette.size_uv > 0 && b.palette.delta_encode_v
        });
        assert!(
            delta_v > 0,
            "tight V clustering must elect the §5.11.46 signed-delta arm \
             (delta={delta_v} literal-or-other={literal_v})"
        );

        // Full-stream conformance through the spec driver.
        let enc = encode_key_frame_yuv_with_q(&input, 60).unwrap();
        let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "luma");
        assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "U");
        assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "V");
    }

    /// r418 — the §5.11.7 intra-block-copy arm must actually be
    /// SELECTED where a whole-pel copy of already-reconstructed
    /// content is the winner: a 192×192 frame tiled from one 64×64
    /// noise tile (every superblock identical) opens the §5.9.20 gate
    /// through the duplicate-tile scan, the search tree commits
    /// `use_intrabc = 1` leaves at §6.10.24-valid DVs, and the
    /// emitted stream round-trips byte-exact through the spec driver
    /// — on the lossy AND the lossless arm.
    #[test]
    fn r418_search_selects_intrabc_on_tiled_content() {
        // One 64x64 xorshift noise tile, repeated over 3x3 superblocks.
        let mut tile_y = [0u16; 64 * 64];
        let mut tile_u = [0u16; 32 * 32];
        let mut tile_v = [0u16; 32 * 32];
        let mut state = 0x00C0_FFEEu32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for p in tile_y.iter_mut() {
            *p = (next() & 0xFF) as u16;
        }
        for p in tile_u.iter_mut() {
            *p = (next() & 0xFF) as u16;
        }
        for p in tile_v.iter_mut() {
            *p = (next() & 0xFF) as u16;
        }
        let (w, h) = (192u32, 192u32);
        let mut input = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 0);
        for i in 0..192usize {
            for j in 0..192usize {
                input.y[i * 192 + j] = tile_y[(i % 64) * 64 + (j % 64)];
            }
        }
        for i in 0..96usize {
            for j in 0..96usize {
                input.u[i * 96 + j] = tile_u[(i % 32) * 32 + (j % 32)];
                input.v[i * 96 + j] = tile_v[(i % 32) * 32 + (j % 32)];
            }
        }

        for q in [60u8, 0] {
            let enc = encode_key_frame_yuv_with_q(&input, q).unwrap();
            assert!(
                enc.fh.allow_intrabc,
                "q={q}: duplicate-tile scan must open the §5.9.20 gate"
            );
            // Tree-level witness over the same driver state.
            let mut recon = ReconState {
                y: vec![0u16; 192 * 192],
                u: vec![0u16; 96 * 96],
                v: vec![0u16; 96 * 96],
                width: 192,
                height: 192,
                chroma_w: 96,
                chroma_h: 96,
                mi_rows: 48,
                mi_cols: 48,
                lossless: q == 0,
                allow_screen_content_tools: true,
                allow_intrabc: true,
                qp: QuantizerParams::neutral(q, 8),
                bit_depth: 8,
                subsampling_x: 1,
                subsampling_y: 1,
                num_planes: 3,
                bd: BlockDecodedMirror::new(1, 1, 3),
                dv_hash: Default::default(),
            };
            let (mut ibc, mut other) = (0u32, 0u32);
            let (mut twin, params) = search_ctx_for_tests(&recon);
            for (sb_r, sb_c) in sb_grid_origins(48, 48) {
                recon.bd.clear_for_sb(sb_r, sb_c, 48, 48);
                twin.arm_read_deltas();
                let (tree, _) = build_search_tree(
                    sb_r,
                    sb_c,
                    BLOCK_64X64,
                    &input,
                    &mut recon,
                    &mut twin,
                    &params,
                    RateModel::Twin,
                    None,
                )
                .unwrap();
                count_palette_leaves_by(&tree, &mut ibc, &mut other, |b| b.intrabc_mv.is_some());
            }
            assert!(
                ibc > 0,
                "q={q}: tiled content must commit use_intrabc = 1 leaves \
                 (got INTRABC={ibc} OTHER={other})"
            );

            // Full-stream conformance through the spec driver.
            let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "q={q}: luma");
            assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "q={q}: U");
            assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "q={q}: V");
            if q == 0 {
                assert_eq!(enc.recon_y, input.y, "lossless intrabc arm must be exact");
                assert_eq!(enc.recon_u, input.u);
                assert_eq!(enc.recon_v, input.v);
            }
        }
    }

    /// r425 — the hash-match DV search must find exact copy sources
    /// the geometric candidate set cannot reach, and the glyph-tier
    /// §5.9.20 gate must open on repeated sub-superblock patterns:
    /// a 256×192 frame with a per-superblock DISTINCT flat background
    /// (no duplicate 64×64 tiles — the r418 tier stays closed) and
    /// one 16×16 noise glyph stamped at four §6.10.24-reachable
    /// destinations whose displacements sit on none of the geometric
    /// strides. The search must commit `use_intrabc = 1` leaves at
    /// exactly those hash-seeded DVs, and the stream must round-trip
    /// byte-exact through the spec driver (whose own §6.10.24
    /// validation re-checks every committed DV).
    #[test]
    fn r425_hash_dv_search_finds_off_stride_glyph_copies() {
        let (w, h) = (256usize, 192usize);
        // Distinct flat background per superblock: kills the r418
        // duplicate-superblock tier AND flat-cell false matches.
        let mut input = YuvFrame::filled(w as u32, h as u32, 8, ChromaFormat::Yuv420, 128);
        for i in 0..h {
            for j in 0..w {
                let sb = (i / 64) * 4 + j / 64;
                input.y[i * w + j] = 60 + 3 * sb as u16;
            }
        }
        // One 16×16 xorshift noise glyph...
        let mut glyph = [0u16; 16 * 16];
        let mut state = 0x0BAD_CAFEu32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for p in glyph.iter_mut() {
            *p = (next() & 0xFF) as u16;
        }
        // ... stamped at one early source and four late destinations.
        // Sources sit in superblocks 0..=1; every destination's
        // superblock raster index is >= 5, so §6.10.24 admits the
        // copy. None of the displacements is a multiple of the
        // 16-/64-stride geometric candidate set.
        // Destination superblocks (raster 5, 6, 7, 8) all watermark
        // BELOW every other glyph superblock, so (0, 0) is each
        // destination's only reachable source — the committed DVs
        // are fully determined.
        let stamps: [(usize, usize); 5] = [(0, 0), (80, 80), (80, 160), (80, 224), (144, 32)];
        for &(sy, sx) in &stamps {
            for i in 0..16 {
                for j in 0..16 {
                    input.y[(sy + i) * w + sx + j] = glyph[i * 16 + j];
                }
            }
        }
        let expected_dvs: Vec<[i32; 2]> = stamps[1..]
            .iter()
            .map(|&(dy, dx)| [-8 * (dy as i32), -8 * (dx as i32)])
            .collect();

        // Background-only twin: the glyph tier must be what opens the
        // gate (no duplicate superblocks, no duplicate glyph cells).
        let mut bg_only = input.clone();
        for &(sy, sx) in &stamps {
            let sb = (sy / 64) * 4 + sx / 64;
            for i in 0..16 {
                for j in 0..16 {
                    bg_only.y[(sy + i) * w + sx + j] = 60 + 3 * sb as u16;
                }
            }
        }
        assert!(
            !intrabc_beneficial(&bg_only),
            "background alone opens the gate"
        );
        assert!(
            intrabc_beneficial(&input),
            "four reachable glyph duplicates must open the glyph tier"
        );

        for q in [60u8, 0] {
            let enc = encode_key_frame_yuv_with_q(&input, q).unwrap();
            assert!(enc.fh.allow_intrabc, "q={q}: glyph tier gate");

            // Tree-level witness with an armed index, mirroring the
            // driver's staircase maintenance.
            let mut recon = ReconState {
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
                allow_intrabc: true,
                qp: QuantizerParams::neutral(q, 8),
                bit_depth: 8,
                subsampling_x: 1,
                subsampling_y: 1,
                num_planes: 3,
                bd: BlockDecodedMirror::new(1, 1, 3),
                dv_hash: crate::encoder::dv_hash::DvHashIndex::build(&input.y, w, h),
            };
            let (mut twin, params) = search_ctx_for_tests(&recon);
            let mut committed: Vec<[i32; 2]> = Vec::new();
            fn collect_dvs(node: &SyntaxNode, out: &mut Vec<[i32; 2]>) {
                match node {
                    SyntaxNode::Leaf(b) => {
                        if let Some(mv) = b.intrabc_mv {
                            out.push(mv);
                        }
                    }
                    SyntaxNode::Split(ch) => {
                        for c in ch.iter() {
                            collect_dvs(c, out);
                        }
                    }
                    _ => {}
                }
            }
            for (sb_r, sb_c) in sb_grid_origins(recon.mi_rows, recon.mi_cols) {
                recon
                    .bd
                    .clear_for_sb(sb_r, sb_c, recon.mi_rows, recon.mi_cols);
                twin.arm_read_deltas();
                let (tree, _) = build_search_tree(
                    sb_r,
                    sb_c,
                    BLOCK_64X64,
                    &input,
                    &mut recon,
                    &mut twin,
                    &params,
                    RateModel::Twin,
                    None,
                )
                .unwrap();
                collect_dvs(&tree, &mut committed);
            }
            for dv in &expected_dvs {
                assert!(
                    committed.contains(dv),
                    "q={q}: hash-seeded DV {dv:?} missing from committed set {committed:?}"
                );
            }

            // Full-stream conformance through the spec driver.
            let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "q={q}: luma");
            assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "q={q}: U");
            assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "q={q}: V");
            if q == 0 {
                assert_eq!(enc.recon_y, input.y, "lossless glyph copies must be exact");
            }
        }
    }

    /// r425 — rectangular §5.11.46 palette leaves: a 64×64 superblock
    /// made of two DIFFERENT 8-colour dither strips (left / right
    /// 32×64, disjoint colour sets) must elect PARTITION_VERT with
    /// palette-coded 32×64 leaves — each strip is exactly
    /// representable at the §5.11.46 cap while the whole block's 16
    /// colours are not, and the VERT pair costs two palette lists /
    /// token walks against the SPLIT arm's four — and the stream must
    /// round-trip byte-exact through the spec driver on the lossy AND
    /// lossless arms.
    #[test]
    fn r425_rect_palette_leaf_elected_on_split_content() {
        let (w, h) = (64usize, 64usize);
        const LEFT: [u16; 8] = [16, 48, 80, 112, 144, 176, 208, 240];
        const RIGHT: [u16; 8] = [20, 52, 84, 116, 148, 180, 212, 244];
        let mut input = YuvFrame::filled(w as u32, h as u32, 8, ChromaFormat::Yuv420, 128);
        let mut state = 0x5EED_0425u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for i in 0..h {
            for j in 0..w {
                let pal = if j < 32 { &LEFT } else { &RIGHT };
                input.y[i * w + j] = pal[(next() & 7) as usize];
            }
        }

        for q in [60u8, 0] {
            let mut recon = ReconState {
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
            };
            let (mut twin, params) = search_ctx_for_tests(&recon);
            recon.bd.clear_for_sb(0, 0, recon.mi_rows, recon.mi_cols);
            twin.arm_read_deltas();
            let (tree, _) = build_search_tree(
                0,
                0,
                BLOCK_64X64,
                &input,
                &mut recon,
                &mut twin,
                &params,
                RateModel::Twin,
                None,
            )
            .unwrap();
            // The winning shape must be a rect node carrying a
            // palette leaf on a non-square block.
            fn find_rect_palette(node: &SyntaxNode) -> bool {
                match node {
                    SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
                        blocks.iter().any(|b| b.palette.size_y > 0)
                    }
                    SyntaxNode::Split(ch) => ch.iter().any(|c| find_rect_palette(c)),
                    _ => false,
                }
            }
            assert!(
                find_rect_palette(&tree),
                "q={q}: split-content superblock must elect a rect palette leaf: {tree:?}"
            );

            // Full-stream conformance through the public API + spec
            // driver (the emitting pass walks the same elected rect
            // shapes).
            let enc = encode_key_frame_yuv_with_q(&input, q).unwrap();
            let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "q={q}: luma");
            assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "q={q}: U");
            assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "q={q}: V");
            if q == 0 {
                assert_eq!(enc.recon_y, input.y, "lossless rect palette must be exact");
            }
        }
    }

    /// r425 — clipped palette leaves on the §5.11.4 `split_or_horz`
    /// edge arm: a 64×80 frame (bottom superblock row straddles the
    /// edge by 48 rows) filled with a 2-colour dither must elect the
    /// single clipped HORZ top block (64×32 coded, 16 rows on-screen)
    /// over the forced-split cascade — one palette list + one
    /// §5.11.49 on-screen-sub-rectangle token walk instead of the
    /// split arm's duplicated pairs — and the stream must round-trip
    /// byte-exact through the spec driver on the lossy AND lossless
    /// arms (the driver's own §5.11.49 reader walks the same
    /// on-screen anti-diagonals).
    #[test]
    fn r425_clipped_palette_leaf_on_bottom_edge() {
        let (w, h) = (64usize, 80usize);
        let mut input = YuvFrame::filled(w as u32, h as u32, 8, ChromaFormat::Yuv420, 128);
        let mut state = 0xC11B_0425u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for p in input.y.iter_mut() {
            *p = if next() & 1 == 0 { 40 } else { 200 };
        }

        for q in [60u8, 0] {
            let mut recon = ReconState {
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
            };
            let (mut twin, params) = search_ctx_for_tests(&recon);
            let mut found_clipped_palette = false;
            for (sb_r, sb_c) in sb_grid_origins(recon.mi_rows, recon.mi_cols) {
                recon
                    .bd
                    .clear_for_sb(sb_r, sb_c, recon.mi_rows, recon.mi_cols);
                twin.arm_read_deltas();
                let (tree, _) = build_search_tree(
                    sb_r,
                    sb_c,
                    BLOCK_64X64,
                    &input,
                    &mut recon,
                    &mut twin,
                    &params,
                    RateModel::Twin,
                    None,
                )
                .unwrap();
                if sb_r == 16 {
                    // The straddling superblock: the elected shape
                    // must be the HORZ edge arm with a palette-coded
                    // clipped top block.
                    if let SyntaxNode::Horz(blocks) = &tree {
                        found_clipped_palette = blocks[0].palette.size_y > 0;
                    }
                }
            }
            assert!(
                found_clipped_palette,
                "q={q}: the straddling superblock must elect a clipped HORZ palette leaf"
            );

            // Full-stream conformance through the public API + spec
            // driver.
            let enc = encode_key_frame_yuv_with_q(&input, q).unwrap();
            let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
            assert_eq!(frames.len(), 1);
            assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "q={q}: luma");
            assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "q={q}: U");
            assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "q={q}: V");
            if q == 0 {
                assert_eq!(
                    enc.recon_y, input.y,
                    "lossless clipped palette must be exact"
                );
            }
        }
    }

    // -----------------------------------------------------------------
    // r425 — screen-content measurement harness (ladder item 5).
    // -----------------------------------------------------------------

    /// One tile-payload encode of `input` through the search + write
    /// path: `screen` opens the §5.9.2/§5.9.20 tool gates exactly like
    /// the public driver (content-adaptive intrabc gate + armed hash
    /// index), `hash = false` leaves the r425 DV index inert (the
    /// r418-r424 geometric search). Returns (tile bytes, luma SSD).
    fn scc_tile_encode(input: &YuvFrame, q: u8, screen: bool, hash: bool) -> (usize, u64) {
        let (w, h) = (input.width as usize, input.height as usize);
        let allow_intrabc = screen && intrabc_beneficial(input);
        let mut recon = ReconState {
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
            allow_screen_content_tools: screen,
            allow_intrabc,
            qp: QuantizerParams::neutral(q, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: if hash && allow_intrabc {
                crate::encoder::dv_hash::DvHashIndex::build(&input.y, w, h)
            } else {
                Default::default()
            },
        };
        let (_, params) = search_ctx_for_tests(&recon);
        let mut writer = SymbolWriter::new(false);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.init_coeff_cdfs(q);
        let mut state = PartitionSyntaxWriter::new(
            recon.mi_rows,
            recon.mi_cols,
            TileGeometry {
                mi_row_start: 0,
                mi_row_end: recon.mi_rows,
                mi_col_start: 0,
                mi_col_end: recon.mi_cols,
            },
        )
        .expect("geometry");
        for (sb_r, sb_c) in sb_grid_origins(recon.mi_rows, recon.mi_cols) {
            recon
                .bd
                .clear_for_sb(sb_r, sb_c, recon.mi_rows, recon.mi_cols);
            state.arm_read_deltas();
            let mut twin = RateTwin::snapshot(&cdfs, &state, &writer);
            twin.arm_read_deltas();
            let (tree, _) = build_search_tree(
                sb_r,
                sb_c,
                BLOCK_64X64,
                input,
                &mut recon,
                &mut twin,
                &params,
                RateModel::Twin,
                None,
            )
            .expect("search");
            crate::encoder::partition_tree::write_partition_tree_syntax(
                &mut writer,
                &mut cdfs,
                &mut state,
                &tree,
                sb_r,
                sb_c,
                BLOCK_64X64,
                &params,
            )
            .expect("write");
        }
        let bytes = writer.finish().len();
        let mut ssd = 0u64;
        for (a, b) in recon.y.iter().zip(input.y.iter()) {
            let d = *a as i64 - *b as i64;
            ssd += (d * d) as u64;
        }
        (bytes, ssd)
    }

    /// Deterministic 2-colour glyph-text page: flat paper, six random
    /// 8×8 bit-pattern glyphs laid out in text lines — the canonical
    /// repeated-glyph screen content (palette-exact blocks, intrabc
    /// sources at arbitrary offsets).
    fn scc_glyph_text(w: usize, h: usize, seed: u32) -> YuvFrame {
        let mut f = YuvFrame::filled(w as u32, h as u32, 8, ChromaFormat::Yuv420, 128);
        for p in f.y.iter_mut() {
            *p = 235;
        }
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let glyphs: Vec<u64> = (0..6)
            .map(|_| (u64::from(next()) << 32) | u64::from(next()))
            .collect();
        let mut row = 8usize;
        while row + 8 <= h.saturating_sub(4) {
            let mut col = 8usize;
            while col + 8 <= w.saturating_sub(4) {
                if next() % 5 == 0 {
                    col += 8; // word gap
                    continue;
                }
                let g = glyphs[(next() % 6) as usize];
                for i in 0..8 {
                    for j in 0..8 {
                        if (g >> (i * 8 + j)) & 1 == 1 {
                            f.y[(row + i) * w + col + j] = 32;
                        }
                    }
                }
                col += 8;
            }
            row += 12;
        }
        f
    }

    /// UI-panel content: flat regions, a title bar, hairline borders,
    /// a 2-colour dither texture pane and a repeated 16×16 icon.
    fn scc_ui_panel(w: usize, h: usize, seed: u32) -> YuvFrame {
        let mut f = YuvFrame::filled(w as u32, h as u32, 8, ChromaFormat::Yuv420, 128);
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for i in 0..h {
            for j in 0..w {
                f.y[i * w + j] = if i < 16 {
                    64
                } else if i == 16 || j == 0 || j + 1 == w || i + 1 == h {
                    16
                } else {
                    200
                };
            }
        }
        // Dither pane in the lower-left quadrant.
        for i in (h / 2)..(h - 8) {
            for j in 8..(w / 2) {
                f.y[i * w + j] = if next() & 1 == 0 { 40 } else { 200 };
            }
        }
        // One 16×16 2-colour icon repeated on the top row of panes.
        let icon: Vec<bool> = (0..256).map(|_| next() & 1 == 1).collect();
        let mut anchors = vec![(24usize, 8usize)];
        let mut x = 8 + 40;
        while x + 16 < w - 8 {
            anchors.push((24, x));
            x += 40;
        }
        for &(ay, ax) in &anchors {
            if ay + 16 <= h && ax + 16 <= w {
                for i in 0..16 {
                    for j in 0..16 {
                        f.y[(ay + i) * w + ax + j] = if icon[i * 16 + j] { 96 } else { 224 };
                    }
                }
            }
        }
        f
    }

    /// Text lines over a smooth diagonal gradient — the mixed case
    /// (screen tools must pay only where they win).
    fn scc_text_over_gradient(w: usize, h: usize, seed: u32) -> YuvFrame {
        let mut f = scc_glyph_text(w, h, seed);
        for i in 0..h {
            for j in 0..w {
                if f.y[i * w + j] == 235 {
                    f.y[i * w + j] = (48 + (i + j) * 160 / (w + h)) as u16;
                }
            }
        }
        f
    }

    /// Always-on tripwire: on canonical glyph-text screen content the
    /// full r425 screen path must code pixel-exact luma at a fraction
    /// of the natural-coding bytes, and the hash-armed DV search must
    /// not lose to the geometric-only search.
    #[test]
    fn r425_scc_screen_tools_multiple_tripwire() {
        let input = scc_glyph_text(192, 192, 0x7357_0425);
        let (natural, _) = scc_tile_encode(&input, 60, false, false);
        let (screen, ssd) = scc_tile_encode(&input, 60, true, true);
        let (nohash, _) = scc_tile_encode(&input, 60, true, false);
        assert_eq!(ssd, 0, "glyph text must code pixel-exact luma at q60");
        assert!(
            screen * 2 < natural,
            "screen tools must at least halve glyph-text bytes (screen={screen} natural={natural})"
        );
        assert!(
            screen <= nohash,
            "hash-armed DV search must not lose to geometric-only (hash={screen} nohash={nohash})"
        );
    }

    /// Env-gated (`OXIDEAV_AV1_SCC_AB_DIR`) screen-content matrix:
    /// 3 content kinds × 2 sizes (one with a clipped bottom edge) ×
    /// 3 quantisers, natural vs screen vs screen-without-hash tile
    /// bytes + luma SSD, CSV per config + aggregate multiples.
    #[test]
    fn r425_scc_measurement_matrix() {
        let Ok(dir) = std::env::var("OXIDEAV_AV1_SCC_AB_DIR") else {
            return;
        };
        std::fs::create_dir_all(&dir).unwrap();
        let mut csv = String::from("content,w,h,q,natural,screen,nohash,ssd_screen\n");
        let (mut nat_total, mut scr_total, mut nohash_total) = (0u64, 0u64, 0u64);
        let mut exact_nat = 0u64;
        let mut exact_scr = 0u64;
        let mut exact_n = 0u32;
        for &(name, which) in &[("glyph", 0u8), ("ui", 1), ("textgrad", 2)] {
            for &(w, h) in &[(192usize, 192usize), (256, 144)] {
                for &q in &[20u8, 60, 100] {
                    let input = match which {
                        0 => scc_glyph_text(w, h, 0x7357_0425),
                        1 => scc_ui_panel(w, h, 0x0425_0426),
                        _ => scc_text_over_gradient(w, h, 0x0425_0427),
                    };
                    let (nat, _) = scc_tile_encode(&input, q, false, false);
                    let (scr, ssd) = scc_tile_encode(&input, q, true, true);
                    let (noh, _) = scc_tile_encode(&input, q, true, false);
                    csv.push_str(&format!("{name},{w},{h},{q},{nat},{scr},{noh},{ssd}\n"));
                    nat_total += nat as u64;
                    scr_total += scr as u64;
                    nohash_total += noh as u64;
                    if ssd == 0 {
                        exact_nat += nat as u64;
                        exact_scr += scr as u64;
                        exact_n += 1;
                    }
                }
            }
        }
        csv.push_str(&format!(
            "# totals natural={nat_total} screen={scr_total} nohash={nohash_total} \
             multiple={:.2} nohash_multiple={:.2} exact_multiple={:.2} exact_configs={exact_n}\n",
            nat_total as f64 / scr_total as f64,
            nat_total as f64 / nohash_total as f64,
            if exact_scr > 0 {
                exact_nat as f64 / exact_scr as f64
            } else {
                0.0
            },
        ));
        std::fs::write(format!("{dir}/scc_ab_r425.csv"), &csv).unwrap();
        eprintln!("{csv}");
    }

    /// r425 pin content — one deterministic 256×144 "screen page"
    /// exercising every ladder-item-5 arm at once: 2-colour glyph
    /// text lines (repeated glyphs at §6.10.24-reachable off-stride
    /// lags — the hash-DV win case), a two-strip 8+8-colour dither
    /// band (rect VERT palette territory), and a full-width 2-colour
    /// dither footer over the clipped bottom superblock row (144 =
    /// 2×64 + 16 — the `split_or_horz` clipped-palette arm).
    fn scc_pin_page(w: usize, h: usize) -> YuvFrame {
        let mut f = YuvFrame::filled(w as u32, h as u32, 8, ChromaFormat::Yuv420, 128);
        for p in f.y.iter_mut() {
            *p = 235;
        }
        let mut state = 0x0425_C0DEu32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        // Six "anti-aliased" glyphs: dense multi-value 8×8 patterns
        // (deliberately NOT palette-representable — the exact-copy
        // arms are the only cheap coding for a repeat).
        let glyphs: Vec<[u16; 64]> = (0..6)
            .map(|_| {
                let mut g = [0u16; 64];
                for p in g.iter_mut() {
                    *p = (next() & 0xFF) as u16;
                }
                g
            })
            .collect();
        // Text lines at a 16-row pitch (every line lands at the same
        // offset inside its 16×16 gate cell) whose content repeats
        // every third line — later repeats sit at §6.10.24-reachable
        // lags that only the hash index seeds (e.g. Δrow = −48).
        let footer_top = (h / 64) * 64;
        let mut line = 0usize;
        let mut row = 8usize;
        while row + 8 + 4 <= footer_top {
            for slot in 0..(w / 8).saturating_sub(2) {
                // Deterministic per-(repeated-line, slot) glyph pick
                // with sparse word gaps.
                let key = (line % 3) * 131 + slot;
                if key % 5 == 0 {
                    continue;
                }
                let g = &glyphs[key % 6];
                let col = 8 + slot * 8;
                for i in 0..8 {
                    for j in 0..8 {
                        f.y[(row + i) * w + col + j] = g[i * 8 + j];
                    }
                }
            }
            line += 1;
            row += 16;
        }
        // Two-strip band: cols w-64..w of the second superblock row
        // (rect VERT palette territory — two disjoint 8-colour
        // dithers).
        const LEFT: [u16; 8] = [16, 48, 80, 112, 144, 176, 208, 240];
        const RIGHT: [u16; 8] = [20, 52, 84, 116, 148, 180, 212, 244];
        let band_c = w - 64;
        for i in 64..footer_top.min(128) {
            for j in band_c..w {
                let pal = if j < band_c + 32 { &LEFT } else { &RIGHT };
                f.y[i * w + j] = pal[(next() & 7) as usize];
            }
        }
        // Clipped footer: the last (partial) superblock row.
        for i in footer_top..h {
            for j in 0..w {
                f.y[i * w + j] = if next() & 1 == 0 { 40 } else { 200 };
            }
        }
        f
    }

    /// The pin page must actually carry all three r425 arms in its
    /// elected tree: hash-seeded intrabc leaves, an interior rect
    /// palette node, and a clipped-edge HORZ palette top block — and
    /// the emitted stream must round-trip byte-exact through the spec
    /// driver (whose §6.10.24 / §5.11.49 readers re-validate them).
    #[test]
    fn r425_scc_pin_page_features() {
        let (w, h) = (256usize, 144usize);
        let input = scc_pin_page(w, h);
        let q = 60u8;
        let allow_intrabc = intrabc_beneficial(&input);
        assert!(allow_intrabc, "the glyph page must open the §5.9.20 gate");
        let mut recon = ReconState {
            y: vec![0u16; w * h],
            u: vec![0u16; (w / 2) * (h / 2)],
            v: vec![0u16; (w / 2) * (h / 2)],
            width: w,
            height: h,
            chroma_w: w / 2,
            chroma_h: h / 2,
            mi_rows: (h / 4) as u32,
            mi_cols: (w / 4) as u32,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc,
            qp: QuantizerParams::neutral(q, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: crate::encoder::dv_hash::DvHashIndex::build(&input.y, w, h),
        };
        let (mut twin, params) = search_ctx_for_tests(&recon);
        let (mut ibc, mut rect_pal, mut clipped_pal) = (0u32, 0u32, 0u32);
        fn scan(node: &SyntaxNode, ibc: &mut u32, rect_pal: &mut u32) {
            match node {
                SyntaxNode::Leaf(b) if b.intrabc_mv.is_some() => *ibc += 1,
                SyntaxNode::Leaf(_) => {}
                SyntaxNode::Split(ch) => {
                    for c in ch.iter() {
                        scan(c, ibc, rect_pal);
                    }
                }
                SyntaxNode::Horz(blocks) | SyntaxNode::Vert(blocks) => {
                    for b in blocks.iter() {
                        if b.intrabc_mv.is_some() {
                            *ibc += 1;
                        }
                        if b.palette.size_y > 0 {
                            *rect_pal += 1;
                        }
                    }
                }
                _ => {}
            }
        }
        for (sb_r, sb_c) in sb_grid_origins(recon.mi_rows, recon.mi_cols) {
            recon
                .bd
                .clear_for_sb(sb_r, sb_c, recon.mi_rows, recon.mi_cols);
            twin.arm_read_deltas();
            let (tree, _) = build_search_tree(
                sb_r,
                sb_c,
                BLOCK_64X64,
                &input,
                &mut recon,
                &mut twin,
                &params,
                RateModel::Twin,
                None,
            )
            .unwrap();
            if sb_r == 32 {
                if let SyntaxNode::Horz(blocks) = &tree {
                    if blocks[0].palette.size_y > 0 {
                        clipped_pal += 1;
                    }
                }
            }
            scan(&tree, &mut ibc, &mut rect_pal);
        }
        assert!(ibc > 0, "pin page must commit intrabc leaves");
        assert!(rect_pal > 0, "pin page must commit rect palette leaves");
        assert!(
            clipped_pal > 0,
            "pin page must elect the clipped HORZ palette arm on the footer row"
        );

        // Public-API stream conformance through the spec driver.
        let enc = encode_key_frame_yuv_with_q(&input, q).unwrap();
        let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "luma");
        assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "U");
        assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "V");
    }

    /// Env-gated pin generation twin (`OXIDEAV_AV1_SCC_FIXTURE_DIR`):
    /// the r425 KEY screen page and a 3-frame scrolled-page GOP,
    /// written as IVF + encoder-reconstruction YUV for external
    /// black-box decoder validation and corpus pinning.
    #[test]
    fn r425_scc_pin_dump() {
        let Ok(dir) = std::env::var("OXIDEAV_AV1_SCC_FIXTURE_DIR") else {
            return;
        };
        std::fs::create_dir_all(&dir).unwrap();
        // Pin A — KEY screen page.
        let input = scc_pin_page(256, 144);
        let enc = encode_key_frame_yuv_with_q(&input, 60).unwrap();
        std::fs::write(
            format!("{dir}/self-kf-256x144-q60-screen-rect.ivf"),
            &enc.ivf_bytes,
        )
        .unwrap();
        let mut yuv = Vec::new();
        yuv.extend_from_slice(&plane8(&enc.recon_y));
        yuv.extend_from_slice(&plane8(&enc.recon_u));
        yuv.extend_from_slice(&plane8(&enc.recon_v));
        std::fs::write(format!("{dir}/self-kf-256x144-q60-screen-rect.yuv"), &yuv).unwrap();

        // Pin B — 3-frame GOP: the page scrolls up 8 px per frame
        // (the text region moves; the footer dither refreshes), so
        // the P-frames mix inter text motion with palette intra
        // re-coding of the changed regions.
        let (w, h) = (192usize, 112usize);
        let page = scc_pin_page(w, h + 16);
        let frame_at = |scroll: usize| -> YuvFrame {
            let mut f = YuvFrame::filled(w as u32, h as u32, 8, ChromaFormat::Yuv420, 128);
            for i in 0..h {
                for j in 0..w {
                    f.y[i * w + j] = page.y[(i + scroll) * w + j];
                }
            }
            f
        };
        let gop: Vec<YuvFrame> = vec![frame_at(0), frame_at(8), frame_at(16)];
        let enc = crate::encoder::inter_frame::encode_gop_yuv_with_q(&gop, 60).expect("gop encode");
        std::fs::write(
            format!("{dir}/self-gop-192x112-q60-screen-scroll.ivf"),
            &enc.ivf_bytes,
        )
        .unwrap();
        let mut yuv = Vec::new();
        for fr in &enc.recon {
            yuv.extend_from_slice(&plane8(&fr.y));
            yuv.extend_from_slice(&plane8(&fr.u));
            yuv.extend_from_slice(&plane8(&fr.v));
        }
        std::fs::write(
            format!("{dir}/self-gop-192x112-q60-screen-scroll.yuv"),
            &yuv,
        )
        .unwrap();
        eprintln!("r425 scc pins dumped to {dir}");
    }

    /// r418 — the k-means clustering arm must win where the block is
    /// NOT exactly representable: 4 colour groups with ±1 jitter
    /// (~12 distinct values per block) cluster to a `<= 8`-colour
    /// §5.11.46 palette whose prediction is within ±1 everywhere,
    /// beating every §7.11.2 mode at q=60; committed leaves stay in
    /// canonical form and the stream round-trips byte-exact.
    #[test]
    fn r418_search_selects_kmeans_palette_on_jittered_dither() {
        const COLORS: [u16; 4] = [16, 80, 160, 240];
        let mut input = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 128);
        let mut state = 0x0A11_CE55u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for p in input.y.iter_mut() {
            let r = next();
            let base = COLORS[(r & 3) as usize];
            let jitter = ((r >> 2) % 3) as i32 - 1;
            *p = (i32::from(base) + jitter).clamp(0, 255) as u16;
        }
        // Distinct count per 64x64 exceeds PALETTE_COLORS (4 groups x
        // 3 jitter levels = 12).
        let mut seen = [false; 256];
        for &v in &input.y {
            seen[v as usize] = true;
        }
        assert!(seen.iter().filter(|&&b| b).count() > PALETTE_COLORS);

        let mut recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows: 16,
            mi_cols: 16,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(60, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        recon.bd.clear_for_sb(0, 0, 16, 16);
        let (mut twin, params) = search_ctx_for_tests(&recon);
        let (tree, _) = build_search_tree(
            0,
            0,
            BLOCK_64X64,
            &input,
            &mut recon,
            &mut twin,
            &params,
            RateModel::Twin,
            None,
        )
        .unwrap();
        let (mut pal, mut other) = (0u32, 0u32);
        count_palette_leaves(&tree, &mut pal, &mut other);
        assert!(
            pal > 0,
            "jittered dither must commit clustered PaletteSizeY > 0 leaves \
             (got PALETTE={pal} OTHER={other})"
        );

        let enc = encode_key_frame_yuv_with_q(&input, 60).unwrap();
        let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].planes[0], plane8(&enc.recon_y));
        assert_eq!(frames[0].planes[1], plane8(&enc.recon_u));
        assert_eq!(frames[0].planes[2], plane8(&enc.recon_v));
    }

    /// r418 conformance-fixture twins: deterministic screen-content
    /// (4-colour glyph tile repeated with period 64 — §5.11.46 palette
    /// AND §5.11.7 intra-block-copy live in one stream) and a k=8
    /// luma+UV palette frame. Each round-trips byte-exact through the
    /// spec driver; when `OXIDEAV_AV1_SCREEN_FIXDIR` is set the exact
    /// streams + reconstructions are written for external validation /
    /// fixture staging.
    #[test]
    fn r418_screen_and_palette_fixture_streams() {
        // --- self-kf-192x192-q60-screen ---
        const COLORS: [u16; 4] = [12, 92, 172, 244];
        let mut tile = vec![0u16; 64 * 64];
        let mut state = 0x5EED_0001u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for p in tile.iter_mut() {
            *p = COLORS[(next() & 3) as usize];
        }
        let mut screen = YuvFrame::filled(192, 192, 8, ChromaFormat::Yuv420, 128);
        for i in 0..192usize {
            for j in 0..192usize {
                screen.y[i * 192 + j] = tile[(i % 64) * 64 + (j % 64)];
            }
        }
        for i in 0..96usize {
            for j in 0..96usize {
                let par = ((i / 2) + (j / 2)) & 1;
                screen.u[i * 96 + j] = if par == 0 { 96 } else { 168 };
                screen.v[i * 96 + j] = if ((i / 4) & 1) == par { 108 } else { 152 };
            }
        }
        let enc_screen = encode_key_frame_yuv_with_q(&screen, 60).unwrap();
        assert!(enc_screen.fh.allow_intrabc);
        assert!(enc_screen.fh.allow_screen_content_tools);

        // --- self-kf-96x80-q100-palette (k = 8 luma dither + chroma
        // checker pairs) ---
        const COLORS8: [u16; 8] = [8, 40, 80, 120, 160, 200, 230, 250];
        let mut state8 = 0x1234_5678u32 ^ 0x2C9;
        let mut next8 = move || {
            state8 ^= state8 << 13;
            state8 ^= state8 >> 17;
            state8 ^= state8 << 5;
            state8
        };
        let mut pal = YuvFrame::filled(96, 80, 8, ChromaFormat::Yuv420, 128);
        for i in 0..80usize {
            for j in 0..96usize {
                pal.y[i * 96 + j] = COLORS8[(next8() as usize) % 8];
            }
        }
        for i in 0..40usize {
            for j in 0..48usize {
                let par = ((i / 2) + (j / 2)) & 1;
                pal.u[i * 48 + j] = if par == 0 { 96 } else { 168 };
                pal.v[i * 48 + j] = if ((i / 4) & 1) == par { 108 } else { 152 };
            }
        }
        let enc_pal = encode_key_frame_yuv_with_q(&pal, 100).unwrap();
        assert!(!enc_pal.fh.allow_intrabc, "no duplicate 64x64 tile pair");

        for (name, enc) in [
            ("self-kf-192x192-q60-screen", &enc_screen),
            ("self-kf-96x80-q100-palette", &enc_pal),
        ] {
            let frames = crate::decoder::decode_av1_spec(&enc.ivf_bytes).unwrap();
            assert_eq!(frames.len(), 1, "{name}");
            assert_eq!(frames[0].planes[0], plane8(&enc.recon_y), "{name}: luma");
            assert_eq!(frames[0].planes[1], plane8(&enc.recon_u), "{name}: U");
            assert_eq!(frames[0].planes[2], plane8(&enc.recon_v), "{name}: V");
            if let Ok(dir) = std::env::var("OXIDEAV_AV1_SCREEN_FIXDIR") {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(format!("{dir}/{name}.ivf"), &enc.ivf_bytes).unwrap();
                let mut yuv = Vec::new();
                yuv.extend_from_slice(&plane8(&enc.recon_y));
                yuv.extend_from_slice(&plane8(&enc.recon_u));
                yuv.extend_from_slice(&plane8(&enc.recon_v));
                std::fs::write(format!("{dir}/{name}.yuv"), &yuv).unwrap();
            }
        }
    }

    /// §6.10.24 validity transcription checks: raster delay, wavefront
    /// gradient, tile bounds, magnitude.
    #[test]
    fn r418_intrabc_dv_validity() {
        // 192x192 frame: mi_rows = mi_cols = 48, totalSb64PerRow = 3.
        let (mr, mc) = (48u32, 48u32);
        // SB (1,1) copying SB (0,0): srcSb64 = 0, activeSb64 = 4 —
        // fails the 4-SB64 raster delay.
        assert!(!intrabc_dv_valid(16, 16, BLOCK_64X64, -64, -64, mr, mc));
        // SB (2,2) copying SB (0,0): srcSb64 = 0 < 8 - 4, wavefront
        // offset 10 admits column 0.
        assert!(intrabc_dv_valid(32, 32, BLOCK_64X64, -128, -128, mr, mc));
        // SB (1,2) copying SB (0,0): srcSb64 = 0 < 5 - 4 = 1 and
        // 0 < 2 - 4 + 5 = 3.
        assert!(intrabc_dv_valid(16, 32, BLOCK_64X64, -64, -128, mr, mc));
        // Out of frame.
        assert!(!intrabc_dv_valid(32, 32, BLOCK_64X64, -256, 0, mr, mc));
        // Sources below or right of the wavefront are invalid.
        assert!(!intrabc_dv_valid(32, 32, BLOCK_64X64, 64, -128, mr, mc));
        // An 8x8 leaf inside SB (2,2) reaching into SB (1,1) content:
        // src bottom edge lands in SB row 1 => srcSb64 = 1*3+1 = 4 <
        // active 8 - 4... equals 4 => invalid; two rows up is fine.
        assert!(!intrabc_dv_valid(32, 32, BLOCK_8X8, -64, -64, mr, mc));
        assert!(intrabc_dv_valid(32, 32, BLOCK_8X8, -128, -64, mr, mc));
    }

    /// Ineligible shapes stay palette-free: 1-colour (flat) blocks, >8
    /// distinct colours, and frames with the §5.9.2 gate closed all
    /// yield no candidate.
    #[test]
    fn r418_palette_candidate_gates() {
        let flat = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 77);
        let recon = ReconState {
            y: vec![0u16; 64 * 64],
            u: vec![0u16; 32 * 32],
            v: vec![0u16; 32 * 32],
            width: 64,
            height: 64,
            chroma_w: 32,
            chroma_h: 32,
            mi_rows: 16,
            mi_cols: 16,
            lossless: false,
            allow_screen_content_tools: true,
            allow_intrabc: false,
            qp: QuantizerParams::neutral(60, 8),
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            num_planes: 3,
            bd: BlockDecodedMirror::new(1, 1, 3),
            dv_hash: Default::default(),
        };
        assert!(
            palette_candidate_y(&flat, &recon, 0, 0, BLOCK_64X64).is_none(),
            "1 distinct colour: no palette candidate"
        );
        let mut many = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
        for (i, p) in many.y.iter_mut().enumerate() {
            *p = (i % 251) as u16;
        }
        assert!(
            palette_candidate_y(&many, &recon, 0, 0, BLOCK_64X64).is_none(),
            "> PALETTE_COLORS distinct colours: no candidate (r418 scope)"
        );
        let two = {
            let mut f = YuvFrame::filled(64, 64, 8, ChromaFormat::Yuv420, 0);
            for (i, p) in f.y.iter_mut().enumerate() {
                *p = if (i / 3) % 2 == 0 { 10 } else { 200 };
            }
            f
        };
        let cand = palette_candidate_y(&two, &recon, 0, 0, BLOCK_64X64).expect("2-colour block");
        assert_eq!(cand.colors, vec![10, 200]);
        assert_eq!(cand.map.len(), 64 * 64);
        // BLOCK_4X4 is under the §5.11.46 outer gate.
        assert!(palette_candidate_y(&two, &recon, 0, 0, BLOCK_4X4).is_none());
    }
}
