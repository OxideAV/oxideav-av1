//! Conformance-grade intra KEY-frame encoder (r409).
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
//! ## Scope (r409)
//!
//! * 8-bit 4:2:0 YUV input ([`Yuv420Frame`]): `(width, height)`
//!   multiples of 8 in `[8, 512]` per axis (any rectangle; frames
//!   wider/taller than 64 ride the multi-superblock walk).
//! * One KEY frame per stream (`show_frame = 1`,
//!   `error_resilient_mode = 1`, `refresh_frame_flags = allFrames`),
//!   single tile, 64×64 superblocks.
//! * `base_q_idx == 0` selects the §5.9.2 `CodedLossless` arm
//!   (forward/inverse WHT, `TxMode = ONLY_4X4`): the decoded planes
//!   equal the input byte-for-byte. `base_q_idx > 0` selects the
//!   lossy DCT_DCT arm (`TxMode = TX_MODE_LARGEST`, every leaf
//!   `TX_4X4`): the decoded planes equal the encoder's own
//!   reconstruction byte-for-byte.
//! * Partition search at the `BLOCK_8X8` level: each in-frame 8×8
//!   node is trial-encoded both as one `BLOCK_8X8` leaf (four §5.11.35
//!   TX_4X4 luma TUs on the lossless arm, one TX_8X8 luma TU on the
//!   lossy `TX_MODE_LARGEST` arm) and as a PARTITION_SPLIT of four
//!   `BLOCK_4X4` leaves, keeping the lower rate-distortion score
//!   (SSD over the node's pixels + a q-scaled coefficient/mode rate
//!   proxy). Per-leaf mode decision by residual SSD over the §6.10.x
//!   modes
//!   that consume only the `w`-above / `h`-left neighbour samples
//!   (`DC_PRED`, `V_PRED`, `H_PRED`, `SMOOTH_PRED`, `SMOOTH_V_PRED`,
//!   `SMOOTH_H_PRED`, `PAETH_PRED`) — the directional D-modes read
//!   above-right / below-left extensions whose §7.11.2 availability
//!   rules this driver does not yet mirror, so they stay out of the
//!   candidate set. The chroma picker additionally evaluates
//!   `UV_CFL_PRED` (§7.11.5.3) over a compact (αU, αV) grid.
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
//! Every per-leaf prediction is computed from the encoder's running
//! reconstruction with the same §7.11.2 kernels the decode walker
//! runs, and the coefficient inverse
//! ([`crate::cdf::dequantize_step1`] +
//! [`crate::transform::inverse_transform_2d`]) is the decoder's own
//! primitive — so the encoder's `recon` tracks the decoder's
//! `CurrFrame` sample-for-sample by induction along the §5.11.4
//! dispatch order. On the lossless arm the residual chain is
//! bit-exact (`recon == input` at every leaf), which the round-trip
//! suite pins.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.2/§5.3 (framing),
//! §5.5/§5.9 (headers), §5.10 (`frame_obu`), §5.11 (tile syntax),
//! §7.11.2 (intra prediction), §7.11.5 (CFL), §7.12/§7.13 (quant +
//! transforms).

use crate::cdf::{
    dequantize_step1, intra_tx_type_set, is_tx_type_in_set, QuantizerParams, TileCdfContext,
    TileGeometry, BLOCK_4X4, BLOCK_64X64, BLOCK_8X8, DCT_DCT, DC_PRED, H_PRED, MODE_TO_TXFM,
    PAETH_PRED, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED, TX_4X4, TX_8X8, UV_CFL_PRED, V_PRED,
};
use crate::cdf::{
    predict_intra_dc_pred, predict_intra_h_pred, predict_intra_paeth_pred,
    predict_intra_smooth_h_pred, predict_intra_smooth_pred, predict_intra_smooth_v_pred,
    predict_intra_v_pred,
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
    build_intra_only_yuv420_8bit_fh_with_q, build_intra_only_yuv420_8bit_seq,
    cfl_predict_4x4_for_plane_dyn, cfl_subsampled_luma_4x4_420_dyn, derive_intra_neighbours_4x4,
    predict_intra_mode_4x4, sb_grid_origins, Yuv420Frame,
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

/// Per-axis extent bound (inclusive) for [`encode_key_frame_yuv420`].
pub const KEY_FRAME_MAX_DIM: u32 = 512;

/// The §6.10.x intra-mode candidate set restricted to kernels that
/// consume only `AboveRow[0..w-1]` / `LeftCol[0..h-1]` / `AboveLeft`
/// (see the module docs for why the D-modes stay out).
const SAFE_INTRA_MODES: [usize; 7] = [
    DC_PRED,
    V_PRED,
    H_PRED,
    SMOOTH_PRED,
    SMOOTH_V_PRED,
    SMOOTH_H_PRED,
    PAETH_PRED,
];

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
/// * Dimensions not multiples of 8, or outside `[8, 512]` per axis,
///   or plane lengths inconsistent with the dimensions —
///   [`Error::PartitionWalkOutOfRange`].
/// * Internal writer overflow surfaces the underlying [`Error`].
pub fn encode_key_frame_yuv420_with_q(
    input: &Yuv420Frame,
    base_q_idx: u8,
) -> Result<EncodedKeyFrame, Error> {
    // Own dimension gate (wider than `Yuv420Frame::validate`'s
    // single-superblock bound): multiples of 8, [8, 512] per axis.
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

    let seq = build_intra_only_yuv420_8bit_seq(input.width, input.height);
    let fh = build_intra_only_yuv420_8bit_fh_with_q(&seq, input.width, input.height, base_q_idx);
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
        tx_mode_select: false,
        quant: qp,
        reduced_tx_set: fh.reduced_tx_set.unwrap_or(false),
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
        lossless,
        qp,
    };

    // §5.11.2 tile walk: one tile, 64×64 superblocks in raster order.
    // Per superblock: build the full-split §5.11.4 tree (running the
    // mode picker + residual pipeline at every in-frame BLOCK_4X4
    // leaf, in NW/NE/SW/SE dispatch order), then emit its syntax.
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
        let tree = build_syntax_tree(sb_r, sb_c, BLOCK_64X64, mi_rows, mi_cols, input, &mut recon);
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

/// Encoder-side running reconstruction + quantiser bundle.
struct ReconState {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    width: usize,
    height: usize,
    chroma_w: usize,
    chroma_h: usize,
    lossless: bool,
    qp: QuantizerParams,
}

/// Recursive §5.11.4 full-split tree builder. Runs the leaf encoder at
/// every in-frame BLOCK_4X4 cell in NW/NE/SW/SE dispatch order (so the
/// running reconstruction visits leaves exactly like the decode walk);
/// out-of-frame quadrants get the short-circuited dummy node.
fn build_syntax_tree(
    r: u32,
    c: u32,
    b_size: usize,
    mi_rows: u32,
    mi_cols: u32,
    input: &Yuv420Frame,
    recon: &mut ReconState,
) -> SyntaxNode {
    if r >= mi_rows || c >= mi_cols {
        // §5.11.4 line 1 — never inspected by the writer.
        return SyntaxNode::dummy_oob();
    }
    if b_size == BLOCK_4X4 {
        return SyntaxNode::Leaf(Box::new(encode_leaf(r, c, input, recon)));
    }
    if b_size == BLOCK_8X8 {
        // Frame dims are multiples of 8, so every reached 8×8 node is
        // fully in-frame — both the leaf and the split shape are legal
        // §5.11.4 choices here; pick by rate-distortion trial.
        debug_assert!(r + 1 < mi_rows && c + 1 < mi_cols);
        return choose_8x8(r, c, input, recon);
    }
    let half = (crate::cdf::NUM_4X4_BLOCKS_WIDE[b_size] as u32) >> 1;
    let sub = crate::cdf::partition_subsize(crate::cdf::PARTITION_SPLIT, b_size)
        .expect("PARTITION_SPLIT subsize exists for every b_size >= BLOCK_8X8");
    SyntaxNode::Split([
        Box::new(build_syntax_tree(r, c, sub, mi_rows, mi_cols, input, recon)),
        Box::new(build_syntax_tree(
            r,
            c + half,
            sub,
            mi_rows,
            mi_cols,
            input,
            recon,
        )),
        Box::new(build_syntax_tree(
            r + half,
            c,
            sub,
            mi_rows,
            mi_cols,
            input,
            recon,
        )),
        Box::new(build_syntax_tree(
            r + half,
            c + half,
            sub,
            mi_rows,
            mi_cols,
            input,
            recon,
        )),
    ])
}

/// Snapshot of the pixel region one 8×8 luma node covers (8×8 luma +
/// the collocated 4×4 chroma cells) — the working set the §5.11.4
/// 8×8-level partition trial saves/restores.
struct RegionSnapshot {
    y: [u8; 64],
    u: [u8; 16],
    v: [u8; 16],
}

fn save_region(recon: &ReconState, r: u32, c: u32) -> RegionSnapshot {
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = ((r as usize / 2) * 4, (c as usize / 2) * 4);
    let mut snap = RegionSnapshot {
        y: [0; 64],
        u: [0; 16],
        v: [0; 16],
    };
    for i in 0..8 {
        for j in 0..8 {
            snap.y[i * 8 + j] = recon.y[(row0 + i) * recon.width + (col0 + j)];
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            snap.u[i * 4 + j] = recon.u[(crow0 + i) * recon.chroma_w + (ccol0 + j)];
            snap.v[i * 4 + j] = recon.v[(crow0 + i) * recon.chroma_w + (ccol0 + j)];
        }
    }
    snap
}

fn restore_region(recon: &mut ReconState, r: u32, c: u32, snap: &RegionSnapshot) {
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = ((r as usize / 2) * 4, (c as usize / 2) * 4);
    for i in 0..8 {
        for j in 0..8 {
            recon.y[(row0 + i) * recon.width + (col0 + j)] = snap.y[i * 8 + j];
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            recon.u[(crow0 + i) * recon.chroma_w + (ccol0 + j)] = snap.u[i * 4 + j];
            recon.v[(crow0 + i) * recon.chroma_w + (ccol0 + j)] = snap.v[i * 4 + j];
        }
    }
}

/// Distortion (SSD, luma + both chroma cells) of the current
/// reconstruction against the input over one 8×8 node's region.
fn region_distortion(recon: &ReconState, input: &Yuv420Frame, r: u32, c: u32) -> u64 {
    let (row0, col0) = ((r as usize) * 4, (c as usize) * 4);
    let (crow0, ccol0) = ((r as usize / 2) * 4, (c as usize / 2) * 4);
    let mut ssd = 0u64;
    for i in 0..8 {
        for j in 0..8 {
            let idx = (row0 + i) * recon.width + (col0 + j);
            let d = recon.y[idx] as i64 - input.y[idx] as i64;
            ssd += (d * d) as u64;
        }
    }
    for i in 0..4 {
        for j in 0..4 {
            let idx = (crow0 + i) * recon.chroma_w + (ccol0 + j);
            let du = recon.u[idx] as i64 - input.u[idx] as i64;
            let dv = recon.v[idx] as i64 - input.v[idx] as i64;
            ssd += (du * du + dv * dv) as u64;
        }
    }
    ssd
}

/// Crude rate proxy for a candidate shape: a fixed per-leaf mode/skip
/// cost plus a magnitude-aware per-nonzero-coefficient cost
/// (`3 + bitlength(|q|)` roughly tracks the §5.11.39 base + BR + golomb
/// growth). Deliberately simple — it only has to ORDER the two §5.11.4
/// candidates consistently.
fn rate_proxy(blocks: &[&SyntaxBlock]) -> u64 {
    let mut rate = 0u64;
    for b in blocks {
        rate += 24;
        for tu in &b.residual_quant {
            for &q in tu {
                if q != 0 {
                    rate += 3 + u64::from(32 - q.unsigned_abs().leading_zeros());
                }
            }
        }
    }
    rate
}

/// §5.11.4 8×8-level partition decision: trial-encode the node both as
/// one `BLOCK_8X8` leaf and as four `BLOCK_4X4` leaves against the
/// same starting reconstruction, and keep the lower `D + λ·R` score
/// (ties prefer the leaf — fewer symbols). The running reconstruction
/// ends in the winning shape's state.
fn choose_8x8(r: u32, c: u32, input: &Yuv420Frame, recon: &mut ReconState) -> SyntaxNode {
    let lambda: u64 = 1 + (recon.qp.base_q_idx as u64 * recon.qp.base_q_idx as u64) / 32;
    let before = save_region(recon, r, c);

    // Candidate A: one BLOCK_8X8 leaf.
    let leaf8 = encode_leaf_8x8(r, c, input, recon);
    let d_a = region_distortion(recon, input, r, c);
    let r_a = rate_proxy(&[&leaf8]);
    let after_a = save_region(recon, r, c);
    restore_region(recon, r, c, &before);

    // Candidate B: PARTITION_SPLIT into four BLOCK_4X4 leaves
    // (NW/NE/SW/SE dispatch order — the same order the writer emits).
    let nw = encode_leaf(r, c, input, recon);
    let ne = encode_leaf(r, c + 1, input, recon);
    let sw = encode_leaf(r + 1, c, input, recon);
    let se = encode_leaf(r + 1, c + 1, input, recon);
    let d_b = region_distortion(recon, input, r, c);
    let r_b = rate_proxy(&[&nw, &ne, &sw, &se]) + 4; // + the extra partition symbol weight
    let score_a = d_a + lambda * r_a;
    let score_b = d_b + lambda * r_b;
    if score_a <= score_b {
        restore_region(recon, r, c, &after_a);
        SyntaxNode::Leaf(Box::new(leaf8))
    } else {
        SyntaxNode::Split([
            Box::new(SyntaxNode::Leaf(Box::new(nw))),
            Box::new(SyntaxNode::Leaf(Box::new(ne))),
            Box::new(SyntaxNode::Leaf(Box::new(sw))),
            Box::new(SyntaxNode::Leaf(Box::new(se))),
        ])
    }
}

/// §7.11.2.1 neighbour arrays for one `size`×`size` cell (safe-mode
/// subset: only `size` above + `size` left samples + the corner are
/// consumed by [`SAFE_INTRA_MODES`] kernels; the arrays are still
/// extended to `2·size` with the same clamp-replicate rule as the 4×4
/// helper so the kernel signatures stay uniform).
#[allow(clippy::type_complexity)]
fn derive_intra_neighbours_sq(
    plane: &[u8],
    pw: usize,
    ph: usize,
    row0: usize,
    col0: usize,
    size: usize,
) -> (u8, u8, Vec<u16>, Vec<u16>, u16) {
    let have_above = (row0 > 0) as u8;
    let have_left = (col0 > 0) as u8;
    let px = |r: usize, c: usize| plane[r * pw + c] as u16;
    let above_left: u16 = if have_above != 0 && have_left != 0 {
        px(row0 - 1, col0 - 1)
    } else if have_above != 0 {
        px(row0 - 1, col0)
    } else if have_left != 0 {
        px(row0, col0 - 1)
    } else {
        1u16 << 7
    };
    let n = 2 * size;
    let mut above = vec![0u16; n];
    let mut left = vec![0u16; n];
    if have_above != 0 {
        for (k, slot) in above.iter_mut().enumerate() {
            *slot = px(row0 - 1, (col0 + k).min(pw - 1));
        }
    } else if have_left != 0 {
        above.fill(px(row0, col0 - 1));
    } else {
        above.fill((1u16 << 7) - 1);
    }
    if have_left != 0 {
        for (k, slot) in left.iter_mut().enumerate() {
            *slot = px((row0 + k).min(ph - 1), col0 - 1);
        }
    } else if have_above != 0 {
        left.fill(px(row0 - 1, col0));
    } else {
        left.fill((1u16 << 7) + 1);
    }
    (have_above, have_left, above, left, above_left)
}

/// [`SAFE_INTRA_MODES`] prediction for one `size`×`size` cell (size ∈
/// {4, 8}) — generic-square twin of the 4×4 mirror helper.
fn predict_safe_mode_sq(
    mode: usize,
    have_above: u8,
    have_left: u8,
    above: &[u16],
    left: &[u16],
    above_left: u16,
    size: usize,
) -> Option<Vec<u8>> {
    let log2 = size.trailing_zeros();
    let mut pred16 = vec![0u16; size * size];
    match mode {
        m if m == DC_PRED => predict_intra_dc_pred(
            have_left,
            have_above,
            log2,
            log2,
            size,
            size,
            8,
            above,
            left,
            &mut pred16,
        )
        .ok()?,
        m if m == V_PRED => predict_intra_v_pred(size, size, above, &mut pred16).ok()?,
        m if m == H_PRED => predict_intra_h_pred(size, size, left, &mut pred16).ok()?,
        m if m == SMOOTH_PRED => {
            predict_intra_smooth_pred(log2, log2, size, size, above, left, &mut pred16).ok()?
        }
        m if m == SMOOTH_V_PRED => {
            predict_intra_smooth_v_pred(log2, size, size, above, left, &mut pred16).ok()?
        }
        m if m == SMOOTH_H_PRED => {
            predict_intra_smooth_h_pred(log2, size, size, above, left, &mut pred16).ok()?
        }
        m if m == PAETH_PRED => {
            predict_intra_paeth_pred(size, size, above, left, above_left, &mut pred16).ok()?
        }
        _ => return None,
    }
    Some(pred16.into_iter().map(|v| v as u8).collect())
}

/// Fixed-mode `size`×`size` prediction from the running plane.
fn predict_mode_sq(
    plane: &[u8],
    pw: usize,
    ph: usize,
    row0: usize,
    col0: usize,
    size: usize,
    mode: usize,
) -> Vec<u8> {
    let (ha, hl, above, left, al) = derive_intra_neighbours_sq(plane, pw, ph, row0, col0, size);
    predict_safe_mode_sq(mode, ha, hl, &above, &left, al, size)
        .expect("SAFE_INTRA_MODES members always predict")
}

/// SSD-minimising [`SAFE_INTRA_MODES`] picker for one `size`×`size`
/// luma cell.
fn pick_safe_mode_sq(
    recon_plane: &[u8],
    input_plane: &[u8],
    pw: usize,
    ph: usize,
    row0: usize,
    col0: usize,
    size: usize,
) -> (u8, Vec<u8>) {
    let (ha, hl, above, left, al) =
        derive_intra_neighbours_sq(recon_plane, pw, ph, row0, col0, size);
    let mut best_mode = DC_PRED as u8;
    let mut best_pred: Vec<u8> = Vec::new();
    let mut best_ssd = u64::MAX;
    for &mode in &SAFE_INTRA_MODES {
        let Some(pred) = predict_safe_mode_sq(mode, ha, hl, &above, &left, al, size) else {
            continue;
        };
        let mut ssd = 0u64;
        for i in 0..size {
            for j in 0..size {
                let d =
                    input_plane[(row0 + i) * pw + (col0 + j)] as i64 - pred[i * size + j] as i64;
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

/// One TX_8X8 residual leg (lossy arm only — the lossless 8×8 leaf
/// rides four §5.11.35 TX_4X4 TUs instead): forward DCT_DCT +
/// quantize, decoder dequant + inverse, `Clip1(pred + res)` stitch.
fn residual_8x8_lossy(
    input_plane: &[u8],
    recon_plane: &mut [u8],
    pw: usize,
    row0: usize,
    col0: usize,
    pred: &[u8],
    qp: &QuantizerParams,
) -> Vec<i32> {
    let mut residual = [0i64; 64];
    for i in 0..8 {
        for j in 0..8 {
            residual[i * 8 + j] =
                input_plane[(row0 + i) * pw + (col0 + j)] as i64 - pred[i * 8 + j] as i64;
        }
    }
    let coeffs = forward_transform_2d(&residual, TX_8X8, DCT_DCT, false);
    let quant = forward_quantize(&coeffs, TX_8X8, 0, 0, DCT_DCT, 15, qp);
    let dequant = dequantize_step1(&quant, TX_8X8, 0, 0, DCT_DCT, 15, qp);
    let res_back = inverse_transform_2d(&dequant, TX_8X8, DCT_DCT, 8, false);
    for i in 0..8 {
        for j in 0..8 {
            let p = pred[i * 8 + j] as i64 + res_back[i * 8 + j];
            recon_plane[(row0 + i) * pw + (col0 + j)] = p.clamp(0, 255) as u8;
        }
    }
    quant
}

/// Encode one in-frame BLOCK_8X8 leaf. Luma: the block mode is picked
/// on an 8×8 whole-block prediction; the lossless arm then codes the
/// four §5.11.35 TX_4X4 TUs sequentially (each TU re-predicted from
/// the running reconstruction with the block mode, exactly like the
/// decode walk), while the lossy `TX_MODE_LARGEST` arm codes one
/// TX_8X8 TU against the whole-block prediction. Chroma: identical to
/// the 4×4 HasChroma arm (one TX_4X4 TU per plane at the collocated
/// chroma cell), coded after the luma TUs per §5.11.34 plane order.
fn encode_leaf_8x8(
    mi_r: u32,
    mi_c: u32,
    input: &Yuv420Frame,
    recon: &mut ReconState,
) -> SyntaxBlock {
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let (width, height) = (recon.width, recon.height);
    let lossless = recon.lossless;
    let qp = recon.qp;

    // --- Luma ---
    let (y_mode, pred8) = pick_safe_mode_sq(&recon.y, &input.y, width, height, row0, col0, 8);
    let mut luma_quant: Vec<Vec<i32>> = Vec::with_capacity(4);
    if lossless {
        // Four TX_4X4 TUs in §5.11.34 row-major order, re-predicted
        // per TU from the running reconstruction.
        for ty in 0..2 {
            for tx in 0..2 {
                let (tr, tc) = (row0 + ty * 4, col0 + tx * 4);
                let pred_v = predict_mode_sq(&recon.y, width, height, tr, tc, 4, y_mode as usize);
                let mut pred = [0u8; 16];
                pred.copy_from_slice(&pred_v);
                luma_quant.push(residual_4x4(
                    &input.y,
                    &mut recon.y,
                    width,
                    tr,
                    tc,
                    &pred,
                    0,
                    true,
                    DCT_DCT,
                    &qp,
                ));
            }
        }
    } else {
        luma_quant.push(residual_8x8_lossy(
            &input.y,
            &mut recon.y,
            width,
            row0,
            col0,
            &pred8,
            &qp,
        ));
    }

    // --- Chroma (HasChroma always true at BLOCK_8X8 / 4:2:0) ---
    let cr = (mi_r as usize) / 2;
    let cc = (mi_c as usize) / 2;
    let (crow0, ccol0) = (cr * 4, cc * 4);
    let (m, pred_u, pred_v, alpha) = pick_safe_mode_4x4_chroma_joint(recon, input, crow0, ccol0);
    let chroma_tx_type = if lossless {
        DCT_DCT
    } else {
        let t = MODE_TO_TXFM.get(m as usize).copied().unwrap_or(DCT_DCT);
        let set = intra_tx_type_set(0, 0, false);
        if is_tx_type_in_set(false, set, t) {
            t
        } else {
            DCT_DCT
        }
    };
    let cw = recon.chroma_w;
    let quant_u = residual_4x4(
        &input.u,
        &mut recon.u,
        cw,
        crow0,
        ccol0,
        &pred_u,
        1,
        lossless,
        chroma_tx_type,
        &qp,
    );
    let quant_v = residual_4x4(
        &input.v,
        &mut recon.v,
        cw,
        crow0,
        ccol0,
        &pred_v,
        2,
        lossless,
        chroma_tx_type,
        &qp,
    );

    // §5.11.11 skip.
    let all_zero = luma_quant.iter().all(|tu| tu.iter().all(|&q| q == 0))
        && quant_u.iter().all(|&q| q == 0)
        && quant_v.iter().all(|&q| q == 0);
    let skip = u8::from(all_zero);
    let residual_quant: Vec<Vec<i32>> = if all_zero {
        Vec::new()
    } else {
        let mut rq = luma_quant;
        rq.push(quant_u);
        rq.push(quant_v);
        rq
    };
    let (cfl_alpha_u, cfl_alpha_v) = match alpha {
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
        uv_mode: Some(m),
        angle_delta_y: 0,
        angle_delta_uv: 0,
        cfl_alpha_u,
        cfl_alpha_v,
        use_filter_intra: 0,
        filter_intra_mode: None,
        palette: Default::default(),
        residual_quant,
        tx_size: None,
        residual_tx_type: Vec::new(),
        var_tx_trees: Vec::new(),
    }
}

/// SSD-minimising picker over [`SAFE_INTRA_MODES`] for one 4×4 cell of
/// `plane` (recon-neighbour prediction, input-target SSD).
fn pick_safe_mode_4x4(
    recon_plane: &[u8],
    input_plane: &[u8],
    pw: usize,
    ph: usize,
    row0: usize,
    col0: usize,
) -> (u8, [u8; 16]) {
    let (ha, hl, above_ext, left_ext, al) =
        derive_intra_neighbours_4x4(recon_plane, pw, ph, row0, col0);
    let mut best = (DC_PRED as u8, [0u8; 16]);
    let mut best_ssd = u64::MAX;
    for &mode in &SAFE_INTRA_MODES {
        let Some(pred) = predict_intra_mode_4x4(mode, ha, hl, &above_ext, &left_ext, al) else {
            continue;
        };
        let mut ssd = 0u64;
        for i in 0..4 {
            for j in 0..4 {
                let d = input_plane[(row0 + i) * pw + (col0 + j)] as i64 - pred[i * 4 + j] as i64;
                ssd += (d * d) as u64;
            }
        }
        if ssd < best_ssd {
            best_ssd = ssd;
            best = (mode as u8, pred);
        }
    }
    best
}

/// Joint U+V picker: [`SAFE_INTRA_MODES`] plus the §7.11.5.3
/// `UV_CFL_PRED` (αU, αV) grid. One shared `uv_mode` per §5.11.22.
#[allow(clippy::type_complexity)]
fn pick_safe_mode_4x4_chroma_joint(
    recon: &ReconState,
    input: &Yuv420Frame,
    row0: usize,
    col0: usize,
) -> (u8, [u8; 16], [u8; 16], Option<(i8, i8)>) {
    let (pw, ph) = (recon.chroma_w, recon.chroma_h);
    let (ha_u, hl_u, above_u, left_u, al_u) =
        derive_intra_neighbours_4x4(&recon.u, pw, ph, row0, col0);
    let (ha_v, hl_v, above_v, left_v, al_v) =
        derive_intra_neighbours_4x4(&recon.v, pw, ph, row0, col0);
    let mut best_mode = DC_PRED as u8;
    let mut best_pred_u = [0u8; 16];
    let mut best_pred_v = [0u8; 16];
    let mut best_alpha: Option<(i8, i8)> = None;
    let mut best_ssd = u64::MAX;
    let mut dc_pred_u = [0u8; 16];
    let mut dc_pred_v = [0u8; 16];
    let ssd_uv = |pred_u: &[u8; 16], pred_v: &[u8; 16]| -> u64 {
        let mut ssd = 0u64;
        for i in 0..4 {
            for j in 0..4 {
                let du = input.u[(row0 + i) * pw + (col0 + j)] as i64 - pred_u[i * 4 + j] as i64;
                let dv = input.v[(row0 + i) * pw + (col0 + j)] as i64 - pred_v[i * 4 + j] as i64;
                ssd += (du * du + dv * dv) as u64;
            }
        }
        ssd
    };
    for &mode in &SAFE_INTRA_MODES {
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
        let ssd = ssd_uv(&pred_u, &pred_v);
        if ssd < best_ssd {
            best_ssd = ssd;
            best_mode = mode as u8;
            best_pred_u = pred_u;
            best_pred_v = pred_v;
            best_alpha = None;
        }
    }
    // §7.11.5.3 CFL arm over the DC base — the luma window of this
    // chroma TU (the whole 8×8 luma quadrant) is fully reconstructed
    // by the time the HasChroma leaf is encoded.
    let (l_arr, luma_avg) =
        cfl_subsampled_luma_4x4_420_dyn(&recon.y, recon.width, recon.height, row0, col0);
    for &(au, av) in CFL_ALPHA_CANDIDATES {
        let pred_u = cfl_predict_4x4_for_plane_dyn(&dc_pred_u, &l_arr, luma_avg, au);
        let pred_v = cfl_predict_4x4_for_plane_dyn(&dc_pred_v, &l_arr, luma_avg, av);
        let ssd = ssd_uv(&pred_u, &pred_v);
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

/// One TX_4X4 residual leg: forward (WHT / DCT_DCT) + quantize, then
/// the decoder's dequant + inverse, stitching `Clip1(pred + res)` into
/// the running plane. Returns the committed `Quant[]`.
#[allow(clippy::too_many_arguments)]
fn residual_4x4(
    input_plane: &[u8],
    recon_plane: &mut [u8],
    pw: usize,
    row0: usize,
    col0: usize,
    pred: &[u8; 16],
    plane: u8,
    lossless: bool,
    tx_type: usize,
    qp: &QuantizerParams,
) -> Vec<i32> {
    let mut residual = [0i64; 16];
    for i in 0..4 {
        for j in 0..4 {
            residual[i * 4 + j] =
                input_plane[(row0 + i) * pw + (col0 + j)] as i64 - pred[i * 4 + j] as i64;
        }
    }
    let coeffs = if lossless {
        forward_wht_4x4(&residual).to_vec()
    } else {
        forward_transform_2d(&residual, TX_4X4, tx_type, false)
    };
    let quant = forward_quantize(&coeffs, TX_4X4, plane, 0, tx_type, 15, qp);
    let dequant = dequantize_step1(&quant, TX_4X4, plane, 0, tx_type, 15, qp);
    let res_back = inverse_transform_2d(&dequant, TX_4X4, tx_type, 8, lossless);
    for i in 0..4 {
        for j in 0..4 {
            let p = pred[i * 4 + j] as i64 + res_back[i * 4 + j];
            recon_plane[(row0 + i) * pw + (col0 + j)] = p.clamp(0, 255) as u8;
        }
    }
    quant
}

/// Stitch a prediction verbatim (the `skip == 1` reconstruction arm).
fn stitch_pred(recon_plane: &mut [u8], pw: usize, row0: usize, col0: usize, pred: &[u8; 16]) {
    for i in 0..4 {
        for j in 0..4 {
            recon_plane[(row0 + i) * pw + (col0 + j)] = pred[i * 4 + j];
        }
    }
}

/// Encode one in-frame BLOCK_4X4 leaf: mode decision, residual
/// pipeline, running-reconstruction update, and the [`SyntaxBlock`]
/// commitment bundle.
fn encode_leaf(mi_r: u32, mi_c: u32, input: &Yuv420Frame, recon: &mut ReconState) -> SyntaxBlock {
    let row0 = (mi_r as usize) * 4;
    let col0 = (mi_c as usize) * 4;
    let (width, height) = (recon.width, recon.height);
    let lossless = recon.lossless;
    let qp = recon.qp;

    // --- Luma ---
    let (y_mode, pred_y) = pick_safe_mode_4x4(&recon.y, &input.y, width, height, row0, col0);
    // Luma TxType: the §5.11.47 guard is open on the lossy arm and
    // the (empty) commitment vector defaults every luma TU to DCT_DCT
    // — the emitted symbol and this residual leg agree by
    // construction. Lossless short-circuits to DCT_DCT (§5.11.40).
    let quant_y = residual_4x4(
        &input.y,
        &mut recon.y,
        width,
        row0,
        col0,
        &pred_y,
        0,
        lossless,
        DCT_DCT,
        &qp,
    );

    // --- Chroma on §5.11.5 HasChroma cells (4:2:0 BLOCK_4X4: both mi
    // coords odd). ---
    let has_chroma = (mi_r & 1) != 0 && (mi_c & 1) != 0;
    let mut uv_mode: Option<u8> = None;
    let mut cfl_alpha: Option<(i8, i8)> = None;
    let mut quant_u: Vec<i32> = Vec::new();
    let mut quant_v: Vec<i32> = Vec::new();
    let mut preds_uv: Option<([u8; 16], [u8; 16], usize, usize)> = None;
    if has_chroma {
        let cr = ((mi_r as usize) - 1) / 2;
        let cc = ((mi_c as usize) - 1) / 2;
        let (crow0, ccol0) = (cr * 4, cc * 4);
        let (m, pred_u, pred_v, alpha) =
            pick_safe_mode_4x4_chroma_joint(recon, input, crow0, ccol0);
        uv_mode = Some(m);
        cfl_alpha = alpha;
        let cw = recon.chroma_w;
        // §5.11.40 `compute_tx_type` chroma-intra arm: `Mode_To_Txfm[
        // UVMode ]` filtered by the §5.11.48 intra set for TX_4X4
        // (DCT_DCT fallback when the derived type is out of set;
        // lossless short-circuits to DCT_DCT before the table).
        let chroma_tx_type = if lossless {
            DCT_DCT
        } else {
            let t = MODE_TO_TXFM.get(m as usize).copied().unwrap_or(DCT_DCT);
            // TX_4X4: txSzSqr == txSzSqrUp == TX_4X4 (= 0).
            let set = intra_tx_type_set(0, 0, false);
            if is_tx_type_in_set(false, set, t) {
                t
            } else {
                DCT_DCT
            }
        };
        quant_u = residual_4x4(
            &input.u,
            &mut recon.u,
            cw,
            crow0,
            ccol0,
            &pred_u,
            1,
            lossless,
            chroma_tx_type,
            &qp,
        );
        quant_v = residual_4x4(
            &input.v,
            &mut recon.v,
            cw,
            crow0,
            ccol0,
            &pred_v,
            2,
            lossless,
            chroma_tx_type,
            &qp,
        );
        preds_uv = Some((pred_u, pred_v, crow0, ccol0));
    }

    // §5.11.11 skip: 1 iff every visited TU quantised to zero — then
    // the reconstruction is the bare prediction on every plane (undo
    // the residual stitches, which are identity for all-zero Quant[]
    // anyway; the stitch above already equals pred in that case).
    let all_zero = quant_y.iter().all(|&q| q == 0)
        && quant_u.iter().all(|&q| q == 0)
        && quant_v.iter().all(|&q| q == 0);
    let skip = u8::from(all_zero);
    if all_zero {
        // The dequant+inverse of an all-zero Quant[] is exactly zero,
        // so recon already equals pred — restitch defensively to keep
        // the invariant obvious.
        stitch_pred(&mut recon.y, width, row0, col0, &pred_y);
        if let Some((pred_u, pred_v, crow0, ccol0)) = preds_uv {
            let cw = recon.chroma_w;
            stitch_pred(&mut recon.u, cw, crow0, ccol0, &pred_u);
            stitch_pred(&mut recon.v, cw, crow0, ccol0, &pred_v);
        }
    }

    let residual_quant: Vec<Vec<i32>> = if all_zero {
        Vec::new()
    } else if has_chroma {
        vec![quant_y, quant_u, quant_v]
    } else {
        vec![quant_y]
    };

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
        angle_delta_y: 0,
        angle_delta_uv: 0,
        cfl_alpha_u,
        cfl_alpha_v,
        use_filter_intra: 0,
        filter_intra_mode: None,
        palette: Default::default(),
        residual_quant,
        tx_size: None,
        residual_tx_type: Vec::new(),
        var_tx_trees: Vec::new(),
    }
}
