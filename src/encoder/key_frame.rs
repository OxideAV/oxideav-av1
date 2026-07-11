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
//! ## Scope (r410)
//!
//! * 8-bit 4:2:0 YUV input ([`Yuv420Frame`]): `(width, height)`
//!   multiples of 8 in `[8, KEY_FRAME_MAX_DIM]` per axis (any
//!   rectangle; frames wider/taller than 64 ride the multi-superblock
//!   walk).
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
    build_intra_only_yuv420_8bit_fh_with_q, build_intra_only_yuv420_8bit_seq, sb_grid_origins,
    Yuv420Frame,
};
use crate::encoder::sequence_obu::write_sequence_header_obu;
use crate::encoder::symbol_writer::SymbolWriter;
use crate::encoder::tile_group_obu::{write_tile_group_obu, TileGroupObu, TilePayload};
use crate::frame_header::FrameHeader;
use crate::obu::ObuType;
use crate::sequence_header::SequenceHeader;
use crate::transform::inverse_transform_2d;
use crate::Error;

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
    // Own dimension gate (wider than `Yuv420Frame::validate`'s
    // single-superblock bound): multiples of 8, [8, MAX] per axis.
    if input.width < 8
        || input.height < 8
        || input.width > KEY_FRAME_MAX_DIM
        || input.height > KEY_FRAME_MAX_DIM
        || input.width % 8 != 0
        || input.height % 8 != 0
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let expected_y = (input.width * input.height) as usize;
    let expected_uv = ((input.width / 2) * (input.height / 2)) as usize;
    if input.y.len() != expected_y || input.u.len() != expected_uv || input.v.len() != expected_uv {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let width = input.width as usize;
    let height = input.height as usize;
    let chroma_w = width / 2;
    let chroma_h = height / 2;

    let mut seq = build_intra_only_yuv420_8bit_seq(input.width, input.height);
    // r410: open the §5.11.24 filter-intra gate — the mode picker now
    // evaluates the five §7.11.2.3 recursive modes on eligible luma
    // blocks (the historical mirror drivers build their own sequence
    // headers and stay unaffected).
    seq.enable_filter_intra = true;
    let mut fh =
        build_intra_only_yuv420_8bit_fh_with_q(&seq, input.width, input.height, base_q_idx);
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
    let qp = QuantizerParams::neutral(base_q_idx, 8);

    // §5.11 frame-scope parameter bundle — mirrors the decode driver's
    // `TileDecodeParams` derivation for this header set field by field.
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
        allow_intrabc: fh.allow_intrabc,
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
        inter: None,
    };

    // Running reconstruction — tracks the decoder's `CurrFrame`
    // sample-for-sample (see module docs).
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
        let tree = build_search_tree(sb_r, sb_c, BLOCK_64X64, input, &mut recon);
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

    Ok(EncodedKeyFrame {
        ivf_bytes,
        temporal_unit_bytes,
        recon_y: recon.y,
        recon_u: recon.u,
        recon_v: recon.v,
        seq,
        fh,
    })
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
struct BlockDecodedMirror {
    bd: Vec<u8>,
}

impl BlockDecodedMirror {
    fn new() -> Self {
        Self {
            bd: vec![0u8; 3 * BD_STRIDE * BD_STRIDE],
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
    /// single-tile frame (`MiRowEnd = MiRows`, `MiColEnd = MiCols`),
    /// 4:2:0, 3 planes.
    fn clear_for_sb(&mut self, sb_r: u32, sb_c: u32, mi_rows: u32, mi_cols: u32) {
        const SB_SIZE4: i32 = 16;
        for plane in 0..3usize {
            let (sub_x, sub_y): (i32, i32) = if plane > 0 { (1, 1) } else { (0, 0) };
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
fn tu_bd_anchor(plane: usize, start_x: usize, start_y: usize) -> (i32, i32) {
    let (sub_x, sub_y): (u32, u32) = if plane > 0 { (1, 1) } else { (0, 0) };
    let row = ((start_y as u32) << sub_y) >> 2;
    let col = ((start_x as u32) << sub_x) >> 2;
    let base_row = ((row & 15) >> sub_y) as i32;
    let base_col = ((col & 15) >> sub_x) as i32;
    (base_row, base_col)
}

/// §7.11.2.1 `haveAboveRight` / `haveBelowLeft` for a `tx_w × tx_h`
/// TU — the §5.11.35 `BlockDecoded[]` border reads.
#[inline]
fn tu_corner_avail(
    bd: &BlockDecodedMirror,
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx_w: usize,
    tx_h: usize,
) -> (bool, bool) {
    let (base_row, base_col) = tu_bd_anchor(plane, start_x, start_y);
    let step_x4 = (tx_w >> 2) as i32;
    let step_y4 = (tx_h >> 2) as i32;
    let above_right = bd.get(plane, base_row - 1, base_col + step_x4);
    let below_left = bd.get(plane, base_row + step_y4, base_col - 1);
    (above_right, below_left)
}

/// §5.11.35 per-TU `BlockDecoded[]` stamp over the TU footprint.
#[inline]
fn tu_bd_stamp(
    bd: &mut BlockDecodedMirror,
    plane: usize,
    start_x: usize,
    start_y: usize,
    tx_w: usize,
    tx_h: usize,
) {
    let (base_row, base_col) = tu_bd_anchor(plane, start_x, start_y);
    let step_x4 = (tx_w >> 2) as i32;
    let step_y4 = (tx_h >> 2) as i32;
    for i in 0..step_y4 {
        for j in 0..step_x4 {
            bd.set(plane, base_row + i, base_col + j);
        }
    }
}

/// Encoder-side running reconstruction + quantiser bundle.
struct ReconState {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    width: usize,
    height: usize,
    chroma_w: usize,
    chroma_h: usize,
    mi_rows: u32,
    mi_cols: u32,
    lossless: bool,
    qp: QuantizerParams,
    bd: BlockDecodedMirror,
}

impl ReconState {
    fn plane(&self, plane: usize) -> (&[u8], usize, usize) {
        match plane {
            0 => (&self.y, self.width, self.height),
            1 => (&self.u, self.chroma_w, self.chroma_h),
            _ => (&self.v, self.chroma_w, self.chroma_h),
        }
    }
}

// ---------------------------------------------------------------------
// §7.11.2 exact-mirror intra prediction (r410).
// ---------------------------------------------------------------------

/// §7.11.2.1 neighbour-array derivation against a `u8` running plane —
/// the encoder twin of the decode walker's `AboveRow[]` / `LeftCol[]`
/// build (head-extended representation: spec index `k` at offset
/// `k + 2`, the `-1` corner at offset 1). `have_above` / `have_left`
/// are the §5.11.35 `(AvailU || y > 0)` / `(AvailL || x > 0)` values,
/// which for a single-tile whole-frame walk collapse to `start_y > 0`
/// / `start_x > 0`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn build_tu_neighbours(
    plane_buf: &[u8],
    pw: usize,
    ph: usize,
    start_x: usize,
    start_y: usize,
    w: usize,
    h: usize,
    have_above_right: bool,
    have_below_left: bool,
) -> (Vec<u16>, Vec<u16>, bool, bool) {
    let have_above = start_y > 0;
    let have_left = start_x > 0;
    let max_x = pw - 1;
    let max_y = ph - 1;
    let read =
        |yy: usize, xx: usize| -> u16 { plane_buf[yy.min(max_y) * pw + xx.min(max_x)] as u16 };
    let half: u16 = 1 << 7;
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
) -> Option<Vec<u8>> {
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
            8,
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
    Some(pred.into_iter().map(|v| v as u8).collect())
}

/// §7.11.2.3 recursive (filter-intra) prediction over pre-built
/// head-extended neighbour arrays — the `use_filter_intra == 1` luma
/// arm of the §7.11.2.1 dispatch (replaces the mode-driven kernels).
fn predict_filter_intra_from_neighbours(
    fim: usize,
    w: usize,
    h: usize,
    above_ext: &[u16],
    left_ext: &[u16],
) -> Option<Vec<u8>> {
    let mut pred = vec![0u16; w * h];
    crate::cdf::predict_intra_recursive(w, h, fim, 8, above_ext, left_ext, &mut pred).ok()?;
    Some(pred.into_iter().map(|v| v as u8).collect())
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
) -> Vec<u8> {
    let (buf, pw, ph) = recon.plane(plane);
    let (avail_ar, avail_bl) = tu_corner_avail(&recon.bd, plane, start_x, start_y, w, h);
    let (above_ext, left_ext, have_above, have_left) =
        build_tu_neighbours(buf, pw, ph, start_x, start_y, w, h, avail_ar, avail_bl);
    if let Some(f) = fim.filter(|_| plane == 0) {
        return predict_filter_intra_from_neighbours(f, w, h, &above_ext, &left_ext)
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
/// chroma TUs).
#[allow(clippy::too_many_arguments)]
fn cfl_layer(
    dc_pred: &[u8],
    recon_y: &[u8],
    luma_w: usize,
    start_x: usize,
    start_y: usize,
    w: usize,
    h: usize,
    alpha: i8,
    max_luma_w: usize,
    max_luma_h: usize,
) -> Vec<u8> {
    let clamp_x = max_luma_w.saturating_sub(2);
    let clamp_y = max_luma_h.saturating_sub(2);
    let mut l = vec![0i64; w * h];
    let mut luma_sum: i64 = 0;
    for i in 0..h {
        let luma_y = ((start_y + i) << 1).min(clamp_y);
        for j in 0..w {
            let luma_x = ((start_x + j) << 1).min(clamp_x);
            let t = recon_y[luma_y * luma_w + luma_x] as i64
                + recon_y[luma_y * luma_w + luma_x + 1] as i64
                + recon_y[(luma_y + 1) * luma_w + luma_x] as i64
                + recon_y[(luma_y + 1) * luma_w + luma_x + 1] as i64;
            let v = t << 1;
            l[i * w + j] = v;
            luma_sum += v;
        }
    }
    let log2_w = w.trailing_zeros();
    let log2_h = h.trailing_zeros();
    let luma_avg = round2_signed(luma_sum, log2_w + log2_h);
    let mut out = vec![0u8; w * h];
    for k in 0..w * h {
        let scaled = round2_signed((alpha as i64) * (l[k] - luma_avg), 6);
        out[k] = ((dc_pred[k] as i64) + scaled).clamp(0, 255) as u8;
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
fn repack_compact(dense: Vec<i32>, w: usize, h: usize) -> Vec<i32> {
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

/// One residual leg at `tx_sz`: forward (WHT for the lossless TX_4X4 /
/// DCT-family otherwise) + quantize, then the decoder's dequant +
/// inverse, stitching `Clip1(pred + res)` into the running plane.
/// Returns the committed `Quant[]` in the §5.11.39 coefficient layout
/// (the §7.12.3 compact-`tw` stride for 64-wide transforms, dense
/// row-major otherwise), zero-padded to `Tx_Width * Tx_Height`.
#[allow(clippy::too_many_arguments)]
fn residual_tx(
    input_plane: &[u8],
    recon_plane: &mut [u8],
    pw: usize,
    row0: usize,
    col0: usize,
    tx_sz: usize,
    pred: &[u8],
    plane: u8,
    lossless: bool,
    tx_type: usize,
    qp: &QuantizerParams,
) -> Vec<i32> {
    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    let mut residual = vec![0i64; w * h];
    for i in 0..h {
        for j in 0..w {
            residual[i * w + j] =
                input_plane[(row0 + i) * pw + (col0 + j)] as i64 - pred[i * w + j] as i64;
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
    let res_back = inverse_transform_2d(&dequant, tx_sz, tx_type, 8, lossless);
    for i in 0..h {
        for j in 0..w {
            let p = pred[i * w + j] as i64 + res_back[i * w + j];
            recon_plane[(row0 + i) * pw + (col0 + j)] = p.clamp(0, 255) as u8;
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
#[allow(clippy::too_many_arguments)]
fn residual_tx_search_luma(
    input_plane: &[u8],
    recon_plane: &mut [u8],
    pw: usize,
    row0: usize,
    col0: usize,
    tx_sz: usize,
    pred: &[u8],
    qp: &QuantizerParams,
) -> (Vec<i32>, u8) {
    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    let mut residual = vec![0i64; w * h];
    for i in 0..h {
        for j in 0..w {
            residual[i * w + j] =
                input_plane[(row0 + i) * pw + (col0 + j)] as i64 - pred[i * w + j] as i64;
        }
    }
    let set = intra_tx_type_set(
        tx_size_sqr_index(tx_sz) as u32,
        TX_SIZE_SQR_UP[tx_sz] as u32,
        false,
    );
    let lambda = lambda_for(qp);
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
        let res_back = inverse_transform_2d(&dequant, tx_sz, t, 8, false);
        let mut d = 0u64;
        for i in 0..h {
            for j in 0..w {
                let rec = (pred[i * w + j] as i64 + res_back[i * w + j]).clamp(0, 255);
                let diff = input_plane[(row0 + i) * pw + (col0 + j)] as i64 - rec;
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
        // §5.11.39: an all-zero TU codes no transform_type symbol and
        // the walker stamps DCT_DCT — the label must follow.
        let label = if all_zero { DCT_DCT as u8 } else { t as u8 };
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
    for i in 0..h {
        for j in 0..w {
            let p = pred[i * w + j] as i64 + res_back[i * w + j];
            recon_plane[(row0 + i) * pw + (col0 + j)] = p.clamp(0, 255) as u8;
        }
    }
    (quant, label)
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
/// for one `n × n` block (recon-neighbour prediction at whole-block
/// extent, input-target SSD). The neighbour build uses the same
/// `BlockDecoded[]` state the block's first luma TU will observe.
fn pick_y_mode(
    recon: &ReconState,
    input: &Yuv420Frame,
    row0: usize,
    col0: usize,
    n: usize,
) -> (u8, i8, Option<u8>) {
    let (avail_ar, avail_bl) = tu_corner_avail(&recon.bd, 0, col0, row0, n, n);
    let (above_ext, left_ext, have_above, have_left) = build_tu_neighbours(
        &recon.y,
        recon.width,
        recon.height,
        col0,
        row0,
        n,
        n,
        avail_ar,
        avail_bl,
    );
    let ssd_of = |pred: &[u8]| -> u64 {
        let mut ssd = 0u64;
        for i in 0..n {
            for j in 0..n {
                let d =
                    input.y[(row0 + i) * recon.width + (col0 + j)] as i64 - pred[i * n + j] as i64;
                ssd += (d * d) as u64;
            }
        }
        ssd
    };
    let mut best = (DC_PRED as u8, 0i8, None);
    let mut best_ssd = u64::MAX;
    for mode in 0..INTRA_MODES {
        for delta in angle_delta_candidates(mode, n) {
            let Some(pred) = predict_mode_from_neighbours(
                mode, delta, n, n, &above_ext, &left_ext, have_above, have_left,
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
    if n <= 32 {
        for fim in 0..crate::cdf::INTRA_FILTER_MODES {
            let Some(pred) = predict_filter_intra_from_neighbours(fim, n, n, &above_ext, &left_ext)
            else {
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
    input: &Yuv420Frame,
    crow0: usize,
    ccol0: usize,
    cn: usize,
    cfl_allowed: bool,
    // §5.11.43 gate operand: the block's LUMA extent (`Block_Width[
    // MiSize ]`) — angle deltas are coded for `MiSize >= BLOCK_8X8`
    // regardless of the subsampled chroma extent.
    luma_n: usize,
    max_luma_w: usize,
    max_luma_h: usize,
) -> (u8, i8, Option<(i8, i8)>) {
    let pw = recon.chroma_w;
    let (ar_u, bl_u) = tu_corner_avail(&recon.bd, 1, ccol0, crow0, cn, cn);
    let (above_u, left_u, ha_u, hl_u) = build_tu_neighbours(
        &recon.u,
        recon.chroma_w,
        recon.chroma_h,
        ccol0,
        crow0,
        cn,
        cn,
        ar_u,
        bl_u,
    );
    let (ar_v, bl_v) = tu_corner_avail(&recon.bd, 2, ccol0, crow0, cn, cn);
    let (above_v, left_v, ha_v, hl_v) = build_tu_neighbours(
        &recon.v,
        recon.chroma_w,
        recon.chroma_h,
        ccol0,
        crow0,
        cn,
        cn,
        ar_v,
        bl_v,
    );
    let ssd_uv = |pred_u: &[u8], pred_v: &[u8]| -> u64 {
        let mut ssd = 0u64;
        for i in 0..cn {
            for j in 0..cn {
                let idx = (crow0 + i) * pw + (ccol0 + j);
                let du = input.u[idx] as i64 - pred_u[i * cn + j] as i64;
                let dv = input.v[idx] as i64 - pred_v[i * cn + j] as i64;
                ssd += (du * du + dv * dv) as u64;
            }
        }
        ssd
    };
    let mut best_mode = DC_PRED as u8;
    let mut best_delta = 0i8;
    let mut best_alpha: Option<(i8, i8)> = None;
    let mut best_ssd = u64::MAX;
    let mut dc_pred_u: Vec<u8> = Vec::new();
    let mut dc_pred_v: Vec<u8> = Vec::new();
    for mode in 0..INTRA_MODES {
        for delta in angle_delta_candidates(mode, luma_n) {
            let Some(pred_u) =
                predict_mode_from_neighbours(mode, delta, cn, cn, &above_u, &left_u, ha_u, hl_u)
            else {
                continue;
            };
            let Some(pred_v) =
                predict_mode_from_neighbours(mode, delta, cn, cn, &above_v, &left_v, ha_v, hl_v)
            else {
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
                cn,
                cn,
                au,
                max_luma_w,
                max_luma_h,
            );
            let pred_v = cfl_layer(
                &dc_pred_v,
                &recon.y,
                recon.width,
                ccol0,
                crow0,
                cn,
                cn,
                av,
                max_luma_w,
                max_luma_h,
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
fn encode_leaf_sq(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    input: &Yuv420Frame,
    recon: &mut ReconState,
) -> SyntaxBlock {
    if recon.lossless || b_size == BLOCK_4X4 {
        // §5.11.15: lossless forces TX_4X4; a BLOCK_4X4 block has no
        // tx_depth choice. One shape — no TX trial.
        let luma_tx = if recon.lossless {
            TX_4X4
        } else {
            MAX_TX_SIZE_RECT[b_size]
        };
        return encode_leaf_with_tx(mi_r, mi_c, b_size, luma_tx, input, recon);
    }
    // §5.11.15 tx_depth RD search (r410, TX_MODE_SELECT): step the
    // luma TX down from `Max_Tx_Size_Rect[MiSize]` via `Split_Tx_Size`
    // (one step for BLOCK_8X8 — its tx_depth alphabet is 2-valued —
    // two for larger squares), trial-encode the leaf at each size
    // against the same starting state, keep the lower `D + λ·R`.
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let lambda = lambda_for(&recon.qp);
    let max_steps = if b_size == BLOCK_8X8 { 1 } else { MAX_TX_DEPTH };
    let mut cands = vec![MAX_TX_SIZE_RECT[b_size]];
    for _ in 0..max_steps {
        let next = SPLIT_TX_SIZE[*cands.last().expect("non-empty")];
        cands.push(next);
    }
    let before = save_region(recon, mi_r, mi_c, n4);
    let mut best: Option<(SyntaxBlock, RegionSnapshot, u64)> = None;
    for (depth, &cand) in cands.iter().enumerate() {
        let leaf = encode_leaf_with_tx(mi_r, mi_c, b_size, cand, input, recon);
        let d = region_distortion(recon, input, mi_r, mi_c, n4);
        let r = leaf_rate(&leaf) + 2 * depth as u64;
        let score = d + lambda * r;
        let improves = match best.as_ref() {
            Some((_, _, s)) => score < *s,
            None => true,
        };
        if improves {
            best = Some((leaf, save_region(recon, mi_r, mi_c, n4), score));
        }
        restore_region(recon, mi_r, mi_c, &before);
    }
    let (leaf, after, _) = best.expect("at least one tx candidate");
    restore_region(recon, mi_r, mi_c, &after);
    leaf
}

/// One-shape leaf encode at a fixed luma TX size — see
/// [`encode_leaf_sq`] for the §5.11.15 TX search wrapper.
fn encode_leaf_with_tx(
    mi_r: u32,
    mi_c: u32,
    b_size: usize,
    luma_tx: usize,
    input: &Yuv420Frame,
    recon: &mut ReconState,
) -> SyntaxBlock {
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size];
    let n = n4 * 4;
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let width = recon.width;
    let lossless = recon.lossless;
    let qp = recon.qp;

    // --- Luma ---
    let (y_mode, angle_delta_y, filter_intra_mode) = pick_y_mode(recon, input, row0, col0, n);
    let (ltw, lth) = (TX_WIDTH[luma_tx], TX_HEIGHT[luma_tx]);
    let mut residual_quant: Vec<Vec<i32>> = Vec::new();
    let mut luma_tx_types: Vec<u8> = Vec::new();
    let mut ty = 0usize;
    while ty < n {
        let mut tx = 0usize;
        while tx < n {
            let (tr, tc) = (row0 + ty, col0 + tx);
            let pred = predict_tu(
                recon,
                0,
                tc,
                tr,
                ltw,
                lth,
                y_mode as usize,
                angle_delta_y as i32,
                filter_intra_mode.map(|f| f as usize),
            );
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
                // r410: §5.11.47 per-TU luma transform-type RD search
                // over the §5.11.48 intra set for this TX size.
                residual_tx_search_luma(&input.y, &mut recon.y, width, tr, tc, luma_tx, &pred, &qp)
            };
            tu_bd_stamp(&mut recon.bd, 0, tc, tr, ltw, lth);
            residual_quant.push(q);
            luma_tx_types.push(tt);
            tx += ltw;
        }
        ty += lth;
    }

    // --- Chroma on §5.11.5 HasChroma leaves ---
    // 4:2:0: every leaf at BLOCK_8X8+ has chroma; a BLOCK_4X4 leaf
    // only when both mi coords are odd (the SE cell of its 8×8 area).
    let has_chroma = if b_size >= BLOCK_8X8 {
        true
    } else {
        (mi_r & 1) != 0 && (mi_c & 1) != 0
    };
    let mut uv_mode: Option<u8> = None;
    let mut angle_delta_uv = 0i8;
    let mut cfl_alpha: Option<(i8, i8)> = None;
    if has_chroma {
        let (crow0, ccol0) = ((mi_r as usize >> 1) * 4, (mi_c as usize >> 1) * 4);
        // §5.11.35 MaxLumaW / MaxLumaH: the block's own luma extent
        // (the last luma TU coded above ends the block).
        let max_luma_w = col0 + n;
        let max_luma_h = row0 + n;
        // §8.3.2 cfl_allowed: on the lossless arm CFL is only allowed
        // when the subsampled chroma residual block is 4×4
        // (`get_plane_residual_size(MiSize, 1) == BLOCK_4X4`); on the
        // lossy arm `Max(Block_Width, Block_Height) <= 32`.
        let cfl_allowed = if lossless { n <= 8 } else { n <= 32 };
        let chroma_tx = if lossless {
            TX_4X4
        } else {
            get_tx_size(1, luma_tx, b_size, 1, 1).unwrap_or(TX_4X4)
        };
        let (ctw, cth) = (TX_WIDTH[chroma_tx], TX_HEIGHT[chroma_tx]);
        // Chroma extent of this leaf's residual grid (§5.11.38
        // subsampled plane size; BLOCK_4X4 leaves keep the full 4×4).
        let cn = if b_size >= BLOCK_8X8 { n / 2 } else { 4 };
        let (m, delta_uv, alpha) = pick_uv_mode(
            recon,
            input,
            crow0,
            ccol0,
            cn,
            cfl_allowed,
            n,
            max_luma_w,
            max_luma_h,
        );
        uv_mode = Some(m);
        angle_delta_uv = delta_uv;
        cfl_alpha = alpha;
        let is_cfl = m as usize == UV_CFL_PRED;
        let chroma_tx_type = chroma_tx_type_for(m, chroma_tx, lossless);
        let cw = recon.chroma_w;
        for plane in 1..=2usize {
            let alpha_p = match (is_cfl, plane, alpha) {
                (true, 1, Some((au, _))) => Some(au),
                (true, 2, Some((_, av))) => Some(av),
                _ => None,
            };
            let mut ty = 0usize;
            while ty < cn {
                let mut tx = 0usize;
                while tx < cn {
                    let (tr, tc) = (crow0 + ty, ccol0 + tx);
                    // §5.11.35: a CFL chroma TU writes the DC_PRED
                    // base, then §7.11.5 layers the luma AC on top.
                    let pred = if is_cfl {
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
                        chroma_tx_type,
                        &qp,
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

    SyntaxBlock {
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
        palette: Default::default(),
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
    }
}

// ---------------------------------------------------------------------
// Rate-distortion partition search.
// ---------------------------------------------------------------------

/// Snapshot of the pixel region one square node covers (its luma
/// square + the collocated chroma squares) plus the §6.10.3
/// `BlockDecoded[]` mirror — the working set a §5.11.4 partition
/// trial saves/restores.
struct RegionSnapshot {
    n: usize,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    bd: BlockDecodedMirror,
}

fn save_region(recon: &ReconState, r: u32, c: u32, n4: usize) -> RegionSnapshot {
    let n = n4 * 4;
    let cn = n / 2;
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = ((r as usize >> 1) * 4, (c as usize >> 1) * 4);
    let mut y = vec![0u8; n * n];
    let mut u = vec![0u8; cn * cn];
    let mut v = vec![0u8; cn * cn];
    for i in 0..n {
        y[i * n..(i + 1) * n].copy_from_slice(&recon.y[(row0 + i) * recon.width + col0..][..n]);
    }
    for i in 0..cn {
        u[i * cn..(i + 1) * cn]
            .copy_from_slice(&recon.u[(crow0 + i) * recon.chroma_w + ccol0..][..cn]);
        v[i * cn..(i + 1) * cn]
            .copy_from_slice(&recon.v[(crow0 + i) * recon.chroma_w + ccol0..][..cn]);
    }
    RegionSnapshot {
        n,
        y,
        u,
        v,
        bd: recon.bd.clone(),
    }
}

fn restore_region(recon: &mut ReconState, r: u32, c: u32, snap: &RegionSnapshot) {
    let n = snap.n;
    let cn = n / 2;
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = ((r as usize >> 1) * 4, (c as usize >> 1) * 4);
    for i in 0..n {
        recon.y[(row0 + i) * recon.width + col0..][..n].copy_from_slice(&snap.y[i * n..][..n]);
    }
    for i in 0..cn {
        recon.u[(crow0 + i) * recon.chroma_w + ccol0..][..cn]
            .copy_from_slice(&snap.u[i * cn..][..cn]);
        recon.v[(crow0 + i) * recon.chroma_w + ccol0..][..cn]
            .copy_from_slice(&snap.v[i * cn..][..cn]);
    }
    recon.bd = snap.bd.clone();
}

/// Distortion (SSD, luma + both chroma cells) of the current
/// reconstruction against the input over one square node's region.
fn region_distortion(recon: &ReconState, input: &Yuv420Frame, r: u32, c: u32, n4: usize) -> u64 {
    let n = n4 * 4;
    let cn = n / 2;
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = ((r as usize >> 1) * 4, (c as usize >> 1) * 4);
    let mut ssd = 0u64;
    for i in 0..n {
        for j in 0..n {
            let idx = (row0 + i) * recon.width + (col0 + j);
            let d = recon.y[idx] as i64 - input.y[idx] as i64;
            ssd += (d * d) as u64;
        }
    }
    for i in 0..cn {
        for j in 0..cn {
            let idx = (crow0 + i) * recon.chroma_w + (ccol0 + j);
            let du = recon.u[idx] as i64 - input.u[idx] as i64;
            let dv = recon.v[idx] as i64 - input.v[idx] as i64;
            ssd += (du * du + dv * dv) as u64;
        }
    }
    ssd
}

/// q-scaled Lagrange multiplier for the `D + λ·R` decisions.
fn lambda_for(qp: &QuantizerParams) -> u64 {
    1 + (qp.base_q_idx as u64 * qp.base_q_idx as u64) / 32
}

/// Crude rate proxy for one leaf: a fixed per-leaf mode/skip cost
/// plus a magnitude-aware per-nonzero-coefficient cost
/// (`3 + bitlength(|q|)` roughly tracks the §5.11.39 base + BR +
/// golomb growth). Deliberately simple — it only has to ORDER the
/// §5.11.4 candidates consistently.
fn leaf_rate(block: &SyntaxBlock) -> u64 {
    let mut rate = 24u64;
    for tu in &block.residual_quant {
        for &q in tu {
            if q != 0 {
                rate += 3 + u64::from(32 - q.unsigned_abs().leading_zeros());
            }
        }
    }
    rate
}

/// Recursive rate proxy over a candidate subtree (each split node adds
/// a small partition-symbol weight).
fn tree_rate(node: &SyntaxNode) -> u64 {
    match node {
        SyntaxNode::Leaf(b) => leaf_rate(b),
        SyntaxNode::Split(children) => 4 + children.iter().map(|c| tree_rate(c)).sum::<u64>(),
    }
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
fn build_search_tree(
    r: u32,
    c: u32,
    b_size: usize,
    input: &Yuv420Frame,
    recon: &mut ReconState,
) -> SyntaxNode {
    if r >= recon.mi_rows || c >= recon.mi_cols {
        // §5.11.4 line 1 — never inspected by the writer.
        return SyntaxNode::dummy_oob();
    }
    if b_size == BLOCK_4X4 {
        return SyntaxNode::Leaf(Box::new(encode_leaf_sq(r, c, b_size, input, recon)));
    }
    let n4 = NUM_4X4_BLOCKS_WIDE[b_size] as u32;
    let half = n4 >> 1;
    let sub = crate::cdf::partition_subsize(crate::cdf::PARTITION_SPLIT, b_size)
        .expect("PARTITION_SPLIT subsize exists for every b_size >= BLOCK_8X8");
    let fully_inside = r + n4 <= recon.mi_rows && c + n4 <= recon.mi_cols;

    let split_children = |input: &Yuv420Frame, recon: &mut ReconState| -> [Box<SyntaxNode>; 4] {
        [
            Box::new(build_search_tree(r, c, sub, input, recon)),
            Box::new(build_search_tree(r, c + half, sub, input, recon)),
            Box::new(build_search_tree(r + half, c, sub, input, recon)),
            Box::new(build_search_tree(r + half, c + half, sub, input, recon)),
        ]
    };

    if !fully_inside {
        // §5.11.4 forced arms: a node straddling the frame edge cannot
        // code PARTITION_NONE at this driver's leaf shapes — split.
        return SyntaxNode::Split(split_children(input, recon));
    }

    let lambda = lambda_for(&recon.qp);
    let before = save_region(recon, r, c, n4 as usize);

    // Candidate A: one PARTITION_NONE leaf.
    let leaf = encode_leaf_sq(r, c, b_size, input, recon);
    let d_a = region_distortion(recon, input, r, c, n4 as usize);
    let r_a = leaf_rate(&leaf);
    let after_a = save_region(recon, r, c, n4 as usize);
    restore_region(recon, r, c, &before);

    // Candidate B: PARTITION_SPLIT into four recursively-searched
    // quadrants (NW/NE/SW/SE dispatch order — the writer's order).
    let children = split_children(input, recon);
    let d_b = region_distortion(recon, input, r, c, n4 as usize);
    let r_b = children.iter().map(|ch| tree_rate(ch)).sum::<u64>() + 4;

    let score_a = d_a + lambda * r_a;
    let score_b = d_b + lambda * r_b;
    if score_a <= score_b {
        restore_region(recon, r, c, &after_a);
        SyntaxNode::Leaf(Box::new(leaf))
    } else {
        SyntaxNode::Split(children)
    }
}
