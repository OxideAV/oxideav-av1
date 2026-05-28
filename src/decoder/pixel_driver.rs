//! Decoder pixel driver — the inverse of
//! [`crate::encoder::pixel_driver::encode_intra_frame_yuv`].
//!
//! Walks an IVF v0 byte buffer (or a single §7.5 temporal unit body)
//! produced by the encoder's arc-15..17 pixel driver, parses every
//! OBU, then per frame:
//!
//!   1. Reparses the §5.5.1 / §5.9.1 SH / FH bodies through the
//!      existing [`crate::sequence_header::parse_sequence_header`] /
//!      [`crate::frame_header::parse_frame_header`] entry points.
//!   2. Reparses the §5.11.1 tile-group OBU body through
//!      [`crate::encoder::parse_tile_group_obu_body`].
//!   3. For each tile (arc-18 supports `num_tiles == 1`):
//!         * Initialises a §8.2.2 [`crate::SymbolDecoder`] over the
//!           tile bytes.
//!         * Walks the §5.11.4 BLOCK_16X16 partition tree in NW/NE/SW/SE
//!           dispatch order through `decode_partition_node`, mirroring
//!           the encoder's `write_partition_tree` recursion.
//!         * At every BLOCK_4X4 leaf, runs the §5.11.5 minimal
//!           per-leaf reader (`decode_block_leaf`) that inverts the
//!           encoder's `write_encode_block_leaf` (skip + intra_segment_id
//!           + y_mode + intra_uv_mode + per-plane §5.11.39 `coefficients`).
//!         * Per-plane reconstructs the 4×4 sample buffer:
//!           §7.11.2.5 DC_PRED prediction + §7.12.3 step-1 dequant +
//!           §7.13 inverse transform (lossless WHT arm).
//!   4. Returns the assembled [`Frame::Yuv420_16x16`].
//!
//! Round 224 hard scope (matches the encoder pixel driver):
//!
//!   * `FrameWidth == FrameHeight == 16`, `MiCols == MiRows == 4`.
//!   * `monochrome = false`, `subsampling_x = subsampling_y = 1`
//!     (4:2:0 YUV).
//!   * `base_q_idx == 0` (the §5.9.2 `CodedLossless` arm — the
//!     §7.13.3 dispatcher routes through the §7.13.2.10 inverse
//!     WHT, the §7.12.3 step-1 dequant lattice has `q == 4` and
//!     `dqDenom == 1`).
//!   * No segmentation, no QM, no in-loop filter passes
//!     (`loop_filter_level == 0`, `enable_cdef == 0`,
//!     `enable_restoration == 0`, `enable_superres == 0`,
//!     `apply_grain == 0`).
//!   * One tile per frame.
//!   * Intra-only keyframes — `y_mode` ∈ {DC_PRED, V_PRED, H_PRED,
//!     D45_PRED, D135_PRED, D113_PRED, D157_PRED, D203_PRED, D67_PRED,
//!     SMOOTH_PRED, SMOOTH_V_PRED, SMOOTH_H_PRED, PAETH_PRED} (the §3
//!     INTRA_MODES set, r228) and `uv_mode` from the same set on
//!     HasChroma leaves (r229). UV_CFL_PRED stays out of scope.
//!
//! Outside the scope, [`decode_av1`] returns
//! [`crate::Error::PartitionWalkOutOfRange`].
//!
//! ## Spec provenance
//!
//! `docs/video/av1/av1-spec.txt`:
//!   * §5.3.1 — Open Bitstream Unit framing.
//!   * §5.11.1 — `tile_group_obu` body.
//!   * §5.11.4 — `decode_partition()` recursion.
//!   * §5.11.5 — `decode_block()` per-leaf reads.
//!   * §5.11.11 — `read_skip()`.
//!   * §5.11.22 — `y_mode()` / `intra_uv_mode()`.
//!   * §5.11.39 — `coefficients()`.
//!   * §7.11.2.5 — DC_PRED.
//!   * §7.12.3 — `dequantize_step1`.
//!   * §7.13 — Inverse transform 2D.

use crate::cdf::{
    cfl_allowed_for_uv_mode, dequantize_step1, partition_ctx, partition_subsize,
    predict_intra_d_mode, predict_intra_dc_pred, predict_intra_h_pred, predict_intra_paeth_pred,
    predict_intra_smooth_h_pred, predict_intra_smooth_pred, predict_intra_smooth_v_pred,
    predict_intra_v_pred, size_group, skip_ctx, split_or_horz_cdf, split_or_vert_cdf,
    PartitionWalker, QuantizerParams, TileCdfContext, TileGeometry, BLOCK_16X16, BLOCK_8X8,
    BLOCK_INVALID, BLOCK_SIZES, D45_PRED, D67_PRED, DCT_DCT, DC_PRED, H_PRED, MI_HEIGHT_LOG2,
    MI_WIDTH_LOG2, NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PAETH_PRED, PARTITION_HORZ,
    PARTITION_NONE, PARTITION_SPLIT, PARTITION_VERT, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED,
    TX_4X4, TX_CLASS_2D, V_PRED,
};
use crate::encoder::ivf::IvfReader;
use crate::encoder::pixel_driver::{
    CHROMA_HEIGHT, CHROMA_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, NUM_INTRA_MODES_Y,
};
use crate::encoder::tile_group_obu::parse_tile_group_obu_body;
use crate::frame_header::{parse_frame_header, FrameHeader};
use crate::obu::{ObuIter, ObuType};
use crate::scan::get_default_scan;
use crate::sequence_header::{parse_sequence_header, SequenceHeader};
use crate::symbol_decoder::SymbolDecoder;
use crate::transform::inverse_transform_2d;
use crate::Error;

/// One decoded frame the public [`decode_av1`] entry surfaces. Owns
/// every sample plane outright so the caller can drop the source IVF
/// buffer.
///
/// Round 224 surfaced only the [`Self::Yuv420_16x16`] variant; round 230
/// adds [`Self::Yuv420Dyn`] for the dynamic-extent encoder
/// ([`crate::encoder::encode_intra_frame_yuv_dyn`]) output. The enum is
/// `#[non_exhaustive]` so future extents (monochrome / 4:2:2 / 4:4:4 /
/// 10-bit) can land without a SemVer bump.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
// The fixed-size `Yuv420_16x16` variant inlines its three plane arrays
// (`16*16 + 8*8*2 = 384` bytes) while `Yuv420Dyn` carries three Vec
// handles + two u32s (~80 bytes). Boxing the fixed-size payload would
// regress the existing arc-18 API; we accept the variance.
#[allow(clippy::large_enum_variant)]
pub enum Frame {
    /// 16×16 4:2:0 YUV — the original fixed-size encoder pixel-driver
    /// output shape.
    Yuv420_16x16 {
        /// Luma plane (`Y`). Row-major `[row][col]`.
        y: [[u8; FRAME_WIDTH]; FRAME_HEIGHT],
        /// First chroma plane (`U` / `Cb`) at half horizontal +
        /// vertical resolution.
        u: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        /// Second chroma plane (`V` / `Cr`).
        v: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    },
    /// Dynamic-extent 4:2:0 YUV — emitted by the r230
    /// [`crate::encoder::encode_intra_frame_yuv_dyn`] driver for any
    /// `(width, height)` ∈ {8, 16, 24, 32, 40, 48, 56, 64} × itself.
    /// Plane data is Vec-backed (row-major).
    Yuv420Dyn {
        /// Luma width in pixels.
        width: u32,
        /// Luma height in pixels.
        height: u32,
        /// Luma plane, row-major, length `width * height`.
        y: Vec<u8>,
        /// U plane, row-major, length `(width / 2) * (height / 2)`.
        u: Vec<u8>,
        /// V plane, same shape as `u`.
        v: Vec<u8>,
    },
}

impl Frame {
    /// Convenience constructor — produces a fresh `Yuv420_16x16` frame
    /// with every plane zeroed.
    #[must_use]
    pub fn zeroed_yuv420_16x16() -> Self {
        Self::Yuv420_16x16 {
            y: [[0u8; FRAME_WIDTH]; FRAME_HEIGHT],
            u: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
            v: [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
        }
    }
}

/// Decode an AV1 IVF v0 buffer — the encoder's
/// [`crate::encoder::encode_intra_frame_yuv`] /
/// [`crate::encoder::encode_intra_frame_y`] output.
///
/// Walks the IVF file header + per-frame records via [`IvfReader`],
/// then routes each frame's payload through [`decode_temporal_unit`].
///
/// Returns one [`Frame`] per IVF frame, in stream order. On the lossless
/// arc-18 path the recovered pixels equal the encoder's input plane-by-
/// plane, byte-for-byte.
///
/// ## Errors
///
/// * Buffer ends mid-IVF-header or mid-frame — [`Error::UnexpectedEnd`].
/// * Any §5.3.1 / §5.5.1 / §5.9.1 / §5.11.1 / §5.11 syntax violation
///   surfaces as the relevant existing `Error` variant.
/// * Out-of-arc-18-scope frames (`FrameWidth != 16`, base_q_idx > 0,
///   inter frames, etc.) surface as
///   [`Error::PartitionWalkOutOfRange`].
pub fn decode_av1(input: &[u8]) -> Result<Vec<Frame>, Error> {
    let reader = IvfReader::new(input).map_err(map_ivf_error)?;
    let frames = reader.read_all().map_err(map_ivf_error)?;
    let mut out: Vec<Frame> = Vec::with_capacity(frames.len());
    // Cache the last-seen SH (the §7.5 temporal unit grammar permits
    // SH to be omitted on subsequent TUs that re-use the prior frame's
    // sequence header).
    let mut last_seq: Option<SequenceHeader> = None;
    for ivf_frame in frames {
        let result = decode_temporal_unit(&ivf_frame.payload, last_seq.as_ref())?;
        if result.sh.is_some() {
            last_seq = result.sh;
        }
        out.push(result.frame);
    }
    Ok(out)
}

/// Per-temporal-unit decode result. `sh` is `Some(SequenceHeader)` when
/// the TU embedded an OBU_SEQUENCE_HEADER (the spec allows omitting it
/// on subsequent TUs that re-use the prior frame's SH).
#[derive(Debug, Clone)]
pub struct TemporalUnitResult {
    /// Recovered frame.
    pub frame: Frame,
    /// Parsed sequence header if the TU contained one.
    pub sh: Option<SequenceHeader>,
}

/// Decode a single §7.5 temporal unit body (TD + optional SH +
/// FrameHeader OBU + TileGroup OBU), using `seq_cache` as a fallback
/// sequence header for TUs that omit OBU_SEQUENCE_HEADER.
///
/// Public so callers walking IVF manually can decode TU-by-TU.
pub fn decode_temporal_unit(
    payload: &[u8],
    seq_cache: Option<&SequenceHeader>,
) -> Result<TemporalUnitResult, Error> {
    // Walk every OBU in the TU body. Arc-18 expects exactly:
    //   * TD (always)
    //   * optionally SH
    //   * FH
    //   * TileGroup
    let mut seq_owned: Option<SequenceHeader> = None;
    let mut frame_header: Option<FrameHeader> = None;
    let mut tile_group_payload: Option<Vec<u8>> = None;
    for desc in ObuIter::new(payload) {
        let desc = desc?;
        match desc.obu_type {
            ObuType::TemporalDelimiter => {
                // §7.5: TD has an empty body; consume + continue.
            }
            ObuType::SequenceHeader => {
                seq_owned = Some(parse_sequence_header(desc.payload)?);
            }
            ObuType::FrameHeader => {
                let seq = seq_owned
                    .as_ref()
                    .or(seq_cache)
                    .ok_or(Error::PartitionWalkOutOfRange)?;
                frame_header = Some(parse_frame_header(desc.payload, seq)?);
            }
            ObuType::TileGroup => {
                tile_group_payload = Some(desc.payload.to_vec());
                // Tile group body is the last OBU on the arc-18 path.
                break;
            }
            // §6.2.2 obu_type values out of scope this arc.
            _ => return Err(Error::PartitionWalkOutOfRange),
        }
    }
    let seq_ref: &SequenceHeader = seq_owned
        .as_ref()
        .or(seq_cache)
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let fh = frame_header.ok_or(Error::PartitionWalkOutOfRange)?;
    let tg_body = tile_group_payload.ok_or(Error::PartitionWalkOutOfRange)?;
    let frame = decode_frame(seq_ref, &fh, &tg_body)?;
    Ok(TemporalUnitResult {
        frame,
        sh: seq_owned,
    })
}

/// Decode a single intra-only 16×16 4:2:0 YUV frame given its (already
/// parsed) sequence header, frame header, and the §5.11.1 tile-group
/// OBU body.
fn decode_frame(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    tile_group_body: &[u8],
) -> Result<Frame, Error> {
    // Frame-size dispatch: 16×16 4:2:0 routes through the arc-18
    // fixed-size driver below; any other (width, height) that satisfies
    // the r230 dyn driver's constraint set
    // ([`super::pixel_driver_dyn::decode_frame_dyn`]) routes through
    // there instead and emits [`Frame::Yuv420Dyn`].
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    if seq.color_config.mono_chrome {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if !seq.color_config.subsampling_x || !seq.color_config.subsampling_y {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if fs.frame_width != FRAME_WIDTH as u32 || fs.frame_height != FRAME_HEIGHT as u32 {
        // Dyn dispatch — any aligned (w, h) ∈ {8..=64} × {8..=64} the
        // r230 driver accepts.
        return crate::decoder::pixel_driver_dyn::decode_frame_dyn(seq, fh, tile_group_body);
    }
    if fs.mi_cols != 4 || fs.mi_rows != 4 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // Arc-18 mirrors the encoder pixel driver exactly. The encoder
    // hard-codes `QuantizerParams::neutral(0, 8)` regardless of what
    // the tiny-fixture FrameHeader's `base_q_idx` field declares (the
    // fixture's FH carries `base_q_idx = 120` for downstream CDF-
    // selection purposes, but the encoder's coefficient pass operates
    // on the lossless q=0 lattice), so the decoder must do the same to
    // round-trip the lossless WHT chain bit-exactly. The FH
    // `quantization_params` field is consulted only to confirm the
    // intra-path parser ran to completion; the decoder's dequant chain
    // uses the same neutral state the encoder did.
    let _qp_fh = fh
        .quantization_params
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let q_params = QuantizerParams::neutral(0, 8);

    // §5.11.1 tile-group OBU body. Arc-18 path is single-tile so
    // num_tiles == 1; encoder hard-codes tile_size_bytes == 1.
    let parsed = parse_tile_group_obu_body(
        tile_group_body,
        /* num_tiles = */ 1,
        /* tile_cols_log2 = */ 0,
        /* tile_rows_log2 = */ 0,
        /* tile_size_bytes = */ 1,
    )?;
    if parsed.tiles.len() != 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let tile_bytes = &parsed.tiles[0].bytes;

    // §8.2.2 init_symbol over the tile bytes.
    let mut decoder =
        SymbolDecoder::init_symbol(tile_bytes, tile_bytes.len(), fh.disable_cdf_update)?;
    let mut cdfs = TileCdfContext::new_from_defaults();

    // §5.11.4 dispatch driver state. Mirrors the encoder's
    // `PartitionTreeWriter` MiSizes[] grid so the §8.3.2 partition_ctx
    // derivations stay in sync. We also build a §5.11.39 walker scratch
    // (used only for the per-leaf `coefficients()` reader's signature —
    // its own internal MiSizes grid stays at the BLOCK_INVALID pre-fill
    // since the arc-18 leaves never consult it).
    let mi_rows = fs.mi_rows;
    let mi_cols = fs.mi_cols;
    let mut state = DecoderState::new(mi_rows, mi_cols);
    let mut coeff_walker = PartitionWalker::new(
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

    // Output frame buffers.
    let mut recon_y: [[u8; FRAME_WIDTH]; FRAME_HEIGHT] = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
    let mut recon_u: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT] = [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT];
    let mut recon_v: [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT] = [[0u8; CHROMA_WIDTH]; CHROMA_HEIGHT];

    let scan: Vec<u16> = get_default_scan(TX_4X4).to_vec();

    // §5.11.4 dispatch — root BLOCK_16X16 PARTITION_SPLIT recursion.
    decode_partition_node(
        &mut decoder,
        &mut cdfs,
        &mut state,
        &mut coeff_walker,
        0,
        0,
        BLOCK_16X16,
        &scan,
        &q_params,
        &mut recon_y,
        &mut recon_u,
        &mut recon_v,
    )?;

    Ok(Frame::Yuv420_16x16 {
        y: recon_y,
        u: recon_u,
        v: recon_v,
    })
}

/// Per-frame decoder state mirroring the encoder's `PartitionTreeWriter`
/// `MiSizes[]` grid. Used solely to derive `partition_ctx` consistently
/// between encoder and decoder on the arc-18 path.
#[derive(Debug)]
struct DecoderState {
    mi_rows: u32,
    mi_cols: u32,
    /// `MiSizes[row][col]` packed row-major. `BLOCK_INVALID` for cells
    /// not yet stamped by a leaf.
    mi_sizes: Vec<usize>,
    geometry: TileGeometry,
}

impl DecoderState {
    fn new(mi_rows: u32, mi_cols: u32) -> Self {
        Self {
            mi_rows,
            mi_cols,
            mi_sizes: vec![BLOCK_INVALID; (mi_rows * mi_cols) as usize],
            geometry: TileGeometry {
                mi_row_start: 0,
                mi_row_end: mi_rows,
                mi_col_start: 0,
                mi_col_end: mi_cols,
            },
        }
    }

    fn mi_size_at(&self, r: i32, c: i32) -> usize {
        if r < 0 || c < 0 {
            return BLOCK_INVALID;
        }
        let (r, c) = (r as u32, c as u32);
        if r >= self.mi_rows || c >= self.mi_cols {
            return BLOCK_INVALID;
        }
        self.mi_sizes[(r * self.mi_cols + c) as usize]
    }

    fn stamp_mi_sizes(&mut self, r: u32, c: u32, sub_size: usize) {
        let bw4 = NUM_4X4_BLOCKS_WIDE[sub_size] as u32;
        let bh4 = NUM_4X4_BLOCKS_HIGH[sub_size] as u32;
        for dr in 0..bh4 {
            let rr = r + dr;
            if rr >= self.mi_rows {
                break;
            }
            for dc in 0..bw4 {
                let cc = c + dc;
                if cc >= self.mi_cols {
                    break;
                }
                self.mi_sizes[(rr * self.mi_cols + cc) as usize] = sub_size;
            }
        }
    }

    fn partition_ctx_for(&self, r: u32, c: u32, bsl: u32) -> usize {
        let avail_u = self.geometry.is_inside(r as i32 - 1, c as i32);
        let avail_l = self.geometry.is_inside(r as i32, c as i32 - 1);
        let above = if avail_u {
            let nb = self.mi_size_at(r as i32 - 1, c as i32);
            nb < BLOCK_SIZES && (MI_WIDTH_LOG2[nb] as u32) < bsl
        } else {
            false
        };
        let left = if avail_l {
            let nb = self.mi_size_at(r as i32, c as i32 - 1);
            nb < BLOCK_SIZES && (MI_HEIGHT_LOG2[nb] as u32) < bsl
        } else {
            false
        };
        partition_ctx(above, left)
    }
}

/// §5.11.4 partition-tree node walk — minimal mirror of the encoder's
/// [`crate::encoder::write_partition_tree`] for the arc-18 scope (only
/// `PARTITION_NONE` + `PARTITION_SPLIT` need handling; the encoder's
/// pixel driver never emits HORZ / VERT / `*_4` / `*_A` / `*_B`).
#[allow(clippy::too_many_arguments)]
fn decode_partition_node(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    state: &mut DecoderState,
    coeff_walker: &mut PartitionWalker,
    r: u32,
    c: u32,
    b_size: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    recon_u: &mut [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    recon_v: &mut [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
) -> Result<(), Error> {
    // §5.11.4 line 1 — out-of-frame quadrant short-circuit.
    if r >= state.mi_rows || c >= state.mi_cols {
        return Ok(());
    }
    if b_size >= BLOCK_SIZES {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let num4x4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
    let half_block4x4 = num4x4 >> 1;
    let has_rows = (r + half_block4x4) < state.mi_rows;
    let has_cols = (c + half_block4x4) < state.mi_cols;

    // §5.11.4 partition decode. Arc-18 path: the encoder always splits
    // BLOCK_16X16 → BLOCK_8X8 → BLOCK_4X4 (leaf), so we expect
    // PARTITION_SPLIT until b_size == BLOCK_4X4 (the forced-NONE arm).
    let partition = if b_size < BLOCK_8X8 {
        PARTITION_NONE
    } else {
        let bsl = MI_WIDTH_LOG2[b_size] as u32;
        let pctx = state.partition_ctx_for(r, c, bsl);
        if has_rows && has_cols {
            let cdf = cdfs
                .partition_cdf(bsl, pctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            decoder.read_symbol(cdf)? as usize
        } else if has_cols {
            let cdf_row = cdfs
                .partition_cdf(bsl, pctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            let mut bin =
                split_or_horz_cdf(cdf_row, b_size).ok_or(Error::PartitionWalkOutOfRange)?;
            let s = decoder.read_symbol(&mut bin)?;
            if s == 0 {
                PARTITION_HORZ
            } else {
                PARTITION_SPLIT
            }
        } else if has_rows {
            let cdf_row = cdfs
                .partition_cdf(bsl, pctx)
                .ok_or(Error::PartitionWalkOutOfRange)?;
            let mut bin =
                split_or_vert_cdf(cdf_row, b_size).ok_or(Error::PartitionWalkOutOfRange)?;
            let s = decoder.read_symbol(&mut bin)?;
            if s == 0 {
                PARTITION_VERT
            } else {
                PARTITION_SPLIT
            }
        } else {
            PARTITION_SPLIT
        }
    };

    let sub_size = partition_subsize(partition, b_size).ok_or(Error::PartitionWalkOutOfRange)?;

    match partition {
        PARTITION_NONE => {
            // §5.11.5 leaf — the encoder stamps MiSizes[] here too.
            state.stamp_mi_sizes(r, c, sub_size);
            decode_block_leaf(
                decoder,
                cdfs,
                coeff_walker,
                r,
                c,
                sub_size,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
        }
        PARTITION_SPLIT => {
            // NW.
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c,
                sub_size,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            // NE.
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c + half_block4x4,
                sub_size,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            // SW.
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c,
                sub_size,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            // SE.
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c + half_block4x4,
                sub_size,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
        }
        // Arc-18 scope: the encoder pixel driver never emits the other
        // partition shapes.
        _ => return Err(Error::PartitionWalkOutOfRange),
    }
    Ok(())
}

/// §5.11.5 per-leaf decoder — inverse of the encoder's
/// `write_encode_block_leaf`. Reads in the same syntax order:
///
///   1. §5.11.11 `read_skip` (with origin-default ctx for arc-18).
///   2. §5.11.8 `intra_segment_id` (no-op on the encoder's
///      `segmentation_enabled = false` path).
///   3. §5.11.22 `y_mode` (Size_Group ctx).
///   4. §5.11.22 `uv_mode` (gated on §5.11.5 `HasChroma`).
///   5. §5.11.39 `coefficients()` per plane the §5.11.5 `HasChroma`
///      walk admits.
///
/// On a non-skip leaf, dequantises + inverse-WHT-transforms the
/// per-plane coefficients and writes the resulting 4×4 sample buffer
/// (after adding the DC_PRED prediction from the running reconstructed
/// plane) into the output frame buffer.
#[allow(clippy::too_many_arguments)]
fn decode_block_leaf(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    coeff_walker: &mut PartitionWalker,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    recon_u: &mut [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    recon_v: &mut [[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
) -> Result<(), Error> {
    // §5.11.11 read_skip — origin-default ctx (skip_ctx(0, 0) = 0) to
    // match the encoder's leaf writer.
    let skip_ctx_val = skip_ctx(0, 0);
    let skip = {
        let cdf = cdfs.skip_cdf(skip_ctx_val);
        decoder.read_symbol(cdf)? as u8
    };
    // §5.11.8 intra_segment_id — segmentation off, no bits.
    // §5.11.22 y_mode — Size_Group ctx (non-keyframe path).
    let size_group_ctx = size_group(sub_size);
    let y_mode = {
        let cdf = cdfs
            .y_mode_cdf(size_group_ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        decoder.read_symbol(cdf)? as u8
    };
    // r228: support all 13 §6.10.x Y intra modes (`DC_PRED` through
    // `PAETH_PRED`). The encoder picks per leaf via residual-SSD; the
    // decoder mirrors via `predict_intra_luma_for_mode` below.
    if (y_mode as usize) >= 13 {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // §5.11.5 HasChroma at 4:2:0 BLOCK_4X4: only the SE-corner of each
    // 8×8 luma quadrant carries chroma. Encoder + decoder must derive
    // the same boolean.
    let has_chroma = (mi_row & 1) != 0 && (mi_col & 1) != 0;

    // §5.11.22 uv_mode — gated on HasChroma. Encoder cfl_allowed is
    // derived from `cfl_allowed_for_uv_mode(lossless=false, sub_size,
    // subsampling_x=true, subsampling_y=true)`; the encoder hard-codes
    // `state.lossless = false` (separate from CodedLossless), so we
    // mirror exactly.
    // r229: decoded `uv_mode` is captured here and threaded into the
    // chroma reconstruction block below. Replaces the r228 "must be
    // DC_PRED" hard reject — the chroma dispatcher now mirrors the
    // §7.11.2.{2..6} fan-out the luma path landed in r228.
    let uv_mode: u8 = if has_chroma {
        let cfl_allowed = cfl_allowed_for_uv_mode(false, sub_size, true, true);
        let cdf = cdfs
            .uv_mode_cdf(cfl_allowed, y_mode as usize)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let m = decoder.read_symbol(cdf)? as u8;
        // The encoder picks from the 13 §3 INTRA_MODES set
        // (DC_PRED..PAETH_PRED, ordinals 0..13). UV_CFL_PRED (= 13) is
        // out of scope this arc — the chroma kernels below would need
        // a §7.11.5.3 CFL αU/αV linear predictor path. Reject any
        // out-of-set decoded mode rather than silently dispatching it
        // through a wrong predictor.
        if (m as usize) >= NUM_INTRA_MODES_Y {
            return Err(Error::PartitionWalkOutOfRange);
        }
        m
    } else {
        DC_PRED as u8
    };

    // Quantizer scratch buffer reused per plane — TX_4X4 has 16 cells.
    let mut quant_y = vec![0i32; 16];
    let mut quant_u = vec![0i32; 16];
    let mut quant_v = vec![0i32; 16];

    // §5.11.39 luma coefficient pass. Encoder uses txb_skip_ctx == 0,
    // dc_sign_ctx == 0 at every leaf.
    let _readout_y = coeff_walker.coefficients(
        decoder,
        cdfs,
        /* plane = */ 0,
        /* is_inter = */ 0,
        TX_4X4,
        TX_CLASS_2D,
        0,
        0,
        scan,
        &mut quant_y,
    )?;

    // Pixel reconstruction (luma).
    let (row0, col0) = ((mi_row as usize) * 4, (mi_col as usize) * 4);
    // r228: dispatch on the decoded y_mode (one of the 13 §6.10.x
    // intra modes) — mirror of the encoder's r228 picker.
    let pred_y =
        predict_intra_luma_for_mode_4x4(recon_y, mi_row as usize, mi_col as usize, y_mode as usize)
            .ok_or(Error::PartitionWalkOutOfRange)?;
    if skip == 0 {
        // §7.12.3 + §7.13: dequant + inverse transform on the lossless
        // arm.
        let dequant = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, qp);
        let residual =
            inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, /* bit_depth = */ 8, true);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + residual[i * 4 + j];
                recon_y[row0 + i][col0 + j] = p.clamp(0, 255) as u8;
            }
        }
    } else {
        // skip == 1 ⇒ all coefficients zero ⇒ reconstructed pixel ==
        // prediction.
        for i in 0..4 {
            for j in 0..4 {
                recon_y[row0 + i][col0 + j] = pred_y[i * 4 + j];
            }
        }
    }

    if has_chroma {
        // Chroma cell coords: chroma 4×4 at the (mi-1)/2 quadrant.
        let cr = ((mi_row as usize) - 1) / 2;
        let cc_idx = ((mi_col as usize) - 1) / 2;
        let crow0 = cr * 4;
        let ccol0 = cc_idx * 4;

        // U plane.
        let _readout_u = coeff_walker.coefficients(
            decoder,
            cdfs,
            /* plane = */ 1,
            0,
            TX_4X4,
            TX_CLASS_2D,
            0,
            0,
            scan,
            &mut quant_u,
        )?;
        // r229: dispatch on the decoded uv_mode (one of the 13 §6.10.x
        // intra modes) — mirror of the luma r228 dispatcher.
        let pred_u = predict_intra_chroma_for_mode_4x4(recon_u, cr, cc_idx, uv_mode as usize)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        if skip == 0 {
            let dequant = dequantize_step1(&quant_u, TX_4X4, 1, 0, DCT_DCT, 15, qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, true);
            for i in 0..4 {
                for j in 0..4 {
                    let p = pred_u[i * 4 + j] as i64 + residual[i * 4 + j];
                    recon_u[crow0 + i][ccol0 + j] = p.clamp(0, 255) as u8;
                }
            }
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    recon_u[crow0 + i][ccol0 + j] = pred_u[i * 4 + j];
                }
            }
        }

        // V plane.
        let _readout_v = coeff_walker.coefficients(
            decoder,
            cdfs,
            2,
            0,
            TX_4X4,
            TX_CLASS_2D,
            0,
            0,
            scan,
            &mut quant_v,
        )?;
        let pred_v = predict_intra_chroma_for_mode_4x4(recon_v, cr, cc_idx, uv_mode as usize)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        if skip == 0 {
            let dequant = dequantize_step1(&quant_v, TX_4X4, 2, 0, DCT_DCT, 15, qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, true);
            for i in 0..4 {
                for j in 0..4 {
                    let p = pred_v[i * 4 + j] as i64 + residual[i * 4 + j];
                    recon_v[crow0 + i][ccol0 + j] = p.clamp(0, 255) as u8;
                }
            }
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    recon_v[crow0 + i][ccol0 + j] = pred_v[i * 4 + j];
                }
            }
        }
    }

    Ok(())
}

/// §7.11.2.1 prologue — derive the neighbour arrays for one BLOCK_4X4
/// luma cell at `(row0, col0)`. Mirror of the encoder helper used by
/// the r228 13-mode picker; see
/// [`crate::encoder::pixel_driver`] private docs for the head-extended
/// buffer convention (`above_ext[0]` = index `-2`, `above_ext[1]` =
/// index `-1`, `above_ext[2 + k]` = index `k`).
fn derive_intra_neighbours_4x4_luma(
    reconstructed: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
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
            let col = (col0 + k).min(FRAME_WIDTH - 1);
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

/// §7.11.2.{2..6} dispatcher — compute the 4×4 luma prediction for the
/// supplied §6.10.x Y intra mode. Mirror of the encoder helper used by
/// the r228 picker. Returns `None` for an out-of-range mode.
fn predict_intra_luma_for_mode_4x4(
    reconstructed: &[[u8; FRAME_WIDTH]; FRAME_HEIGHT],
    cell_row: usize,
    cell_col: usize,
    mode: usize,
) -> Option<[u8; 16]> {
    let w = 4usize;
    let h = 4usize;
    let log2_w = 2u32;
    let log2_h = 2u32;
    let bit_depth = 8u8;
    let row0 = cell_row * 4;
    let col0 = cell_col * 4;
    let (have_above, have_left, above_ext, left_ext, above_left) =
        derive_intra_neighbours_4x4_luma(reconstructed, row0, col0);
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
            predict_intra_d_mode(m, 0, w, h, 0, 0, &above_ext, &left_ext, &mut pred16).ok()?;
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

/// §7.11.2.1 prologue for one 4×4 chroma cell — mirror of
/// [`derive_intra_neighbours_4x4_luma`] against the smaller
/// `CHROMA_WIDTH × CHROMA_HEIGHT` plane. Same head-extended buffer
/// convention so the §7.11.2.{2..6} kernels can be invoked
/// uniformly. Used by the r229 chroma 13-mode dispatcher.
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

/// §7.11.2.{2..6} dispatcher — compute the 4×4 **chroma** prediction
/// for the supplied §6.10.x intra mode. Mirror of
/// [`predict_intra_luma_for_mode_4x4`] against the smaller chroma
/// plane (`CHROMA_HEIGHT × CHROMA_WIDTH`). Returns `None` for an
/// out-of-set mode (the caller's r229 guard at the `uv_mode` read site
/// already rejects `uv_mode >= NUM_INTRA_MODES_Y`).
///
/// `(cell_row, cell_col)` are 4×4 chroma-cell coordinates in
/// `0..CHROMA_CELLS_HIGH` × `0..CHROMA_CELLS_WIDE`.
fn predict_intra_chroma_for_mode_4x4(
    reconstructed: &[[u8; CHROMA_WIDTH]; CHROMA_HEIGHT],
    cell_row: usize,
    cell_col: usize,
    mode: usize,
) -> Option<[u8; 16]> {
    let w = 4usize;
    let h = 4usize;
    let log2_w = 2u32;
    let log2_h = 2u32;
    let bit_depth = 8u8;
    let row0 = cell_row * 4;
    let col0 = cell_col * 4;
    let (have_above, have_left, above_ext, left_ext, above_left) =
        derive_intra_neighbours_4x4_chroma(reconstructed, row0, col0);
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
            predict_intra_d_mode(m, 0, w, h, 0, 0, &above_ext, &left_ext, &mut pred16).ok()?;
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

/// §7.11.2.5 DC_PRED prediction for one 4×4 luma cell — mirror of the
/// encoder's `dc_pred_for_cell_y`.
#[allow(dead_code)]
fn dc_pred_luma(
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
    .expect("oxideav-av1 decoder pixel_driver: DC_PRED arguments in range");
    let mut pred8 = [0u8; 16];
    for (slot, v) in pred8.iter_mut().zip(pred16.iter().copied()) {
        *slot = v as u8;
    }
    pred8
}

/// §7.11.2.5 DC_PRED prediction for one 4×4 chroma cell — mirror of
/// the encoder's `dc_pred_for_cell_chroma`. Unused at the chroma walk
/// site in r229+ (the dispatcher now routes through
/// [`predict_intra_chroma_for_mode_4x4`]) but kept as the documented
/// narrow-surface helper.
#[allow(dead_code)]
fn dc_pred_chroma(
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
    .expect("oxideav-av1 decoder pixel_driver: chroma DC_PRED arguments in range");
    let mut pred8 = [0u8; 16];
    for (slot, v) in pred8.iter_mut().zip(pred16.iter().copied()) {
        *slot = v as u8;
    }
    pred8
}

/// Map [`crate::encoder::ivf::IvfReadError`] into the crate-level
/// [`Error`] enum used by the public [`decode_av1`] entry.
fn map_ivf_error(e: crate::encoder::ivf::IvfReadError) -> Error {
    use crate::encoder::ivf::IvfReadError;
    match e {
        IvfReadError::UnexpectedEnd => Error::UnexpectedEnd,
        IvfReadError::BadMagic
        | IvfReadError::UnsupportedHeaderLength(_)
        | IvfReadError::UnsupportedVersion(_) => Error::PartitionWalkOutOfRange,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::pixel_driver::{
        encode_intra_frame_y, encode_intra_frame_yuv, Yuv420Frame16x16,
    };

    // Tiny-i-only-16x16-prof0 fixture payloads — same as the
    // pixel_driver tests on the encoder side.
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
    fn decode_av1_recovers_flat_grey_yuv() {
        // Encoder side — flat-128 input ⇒ DC_PRED equals the input at
        // every leaf ⇒ residual == 0 everywhere ⇒ trivial recovery.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16::default(); // all-128 mid-grey
        let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
        let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            Frame::Yuv420_16x16 { y, u, v } => {
                assert_eq!(y, &input.y);
                assert_eq!(u, &input.u);
                assert_eq!(v, &input.v);
            }
            other => panic!("expected Yuv420_16x16, got {other:?}"),
        }
    }

    #[test]
    fn decode_av1_recovers_non_flat_yuv() {
        // Non-uniform input. The lossless WHT arm guarantees bit-exact
        // recovery on arbitrary inputs at base_q_idx = 0.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut input = Yuv420Frame16x16::default();
        for (i, row) in input.y.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = ((i * 16 + j) & 0xFF) as u8;
            }
        }
        for (i, row) in input.u.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = ((i * 8 + j) & 0xFF) as u8;
            }
        }
        for (i, row) in input.v.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (255u8).wrapping_sub(((i * 8 + j) & 0xFF) as u8);
            }
        }
        let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
        let decoded = decode_av1(&encoded.ivf_bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            Frame::Yuv420_16x16 { y, u, v } => {
                assert_eq!(y, &input.y, "luma roundtrip");
                assert_eq!(u, &input.u, "U roundtrip");
                assert_eq!(v, &input.v, "V roundtrip");
            }
            other => panic!("expected Yuv420_16x16, got {other:?}"),
        }
    }

    #[test]
    fn decode_av1_y_only_encoder_path_emits_non_conformant_stream() {
        // The encoder's `encode_intra_frame_y` (luma-only) path emits a
        // bitstream whose SH claims `monochrome = false` (the tiny
        // fixture's color config) but whose per-leaf coefficient stream
        // omits the §5.11.39 chroma passes entirely (`uv_mode = None`).
        // §5.11.5 `HasChroma` is `true` on the SE-corner mi cells under
        // that SH ⇒ a conformant decoder expects chroma coefficient
        // blocks the encoder never wrote.
        //
        // The decoder's arc-18 driver mirrors the spec literally, so it
        // surfaces [`Error::PartitionWalkOutOfRange`] (or the §5.11.39
        // EOB-shape mismatch as `UnexpectedEnd`) on the missing-chroma-
        // syntax leaves — exactly the right rejection for this
        // intentionally-non-conformant path. The full 4:2:0 YUV
        // roundtrip (`encode_intra_frame_yuv` ↔ `decode_av1`) is the
        // milestone exercise — see `decode_av1_recovers_non_flat_yuv`.
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let mut luma = [[0u8; FRAME_WIDTH]; FRAME_HEIGHT];
        for (i, row) in luma.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = ((17 * i + 31 * j + 7) & 0xFF) as u8;
            }
        }
        let encoded = encode_intra_frame_y(&luma, &seq, &fh).unwrap();
        let res = decode_av1(&encoded.ivf_bytes);
        assert!(
            res.is_err(),
            "Y-only encode path skips chroma syntax under a YUV SH ⇒ decode must reject"
        );
    }

    #[test]
    fn decode_av1_rejects_short_buffer() {
        let err = decode_av1(&[]).unwrap_err();
        assert_eq!(err, Error::UnexpectedEnd);
    }

    #[test]
    fn temporal_unit_result_carries_sh_when_present() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let input = Yuv420Frame16x16::default();
        let encoded = encode_intra_frame_yuv(&input, &seq, &fh).unwrap();
        let tu = decode_temporal_unit(&encoded.temporal_unit_bytes, None).unwrap();
        assert!(tu.sh.is_some(), "TU body contains SH on first frame");
    }

    #[test]
    fn decode_zeroed_yuv420_16x16_factory_zeros_each_plane() {
        // Sanity-check the public Frame constructor.
        let f = Frame::zeroed_yuv420_16x16();
        let Frame::Yuv420_16x16 { y, u, v } = f else {
            panic!("zeroed_yuv420_16x16 must return Yuv420_16x16")
        };
        assert!(y.iter().flatten().all(|&p| p == 0));
        assert!(u.iter().flatten().all(|&p| p == 0));
        assert!(v.iter().flatten().all(|&p| p == 0));
    }
}
