//! Dynamic-extent decoder pixel driver — arc r230 inverse of
//! [`crate::encoder::pixel_driver_dyn::encode_intra_frame_yuv_dyn`].
//!
//! Walks the §5.11.1 tile-group body of an IVF frame produced by the
//! r230 dyn encoder and surfaces the recovered samples as
//! [`super::pixel_driver::Frame::Yuv420Dyn`]. Reuses every existing
//! symbol-decoder / dequantizer / inverse-transform / intra-prediction
//! kernel; the only delta vs the fixed-size [`super::pixel_driver`] is
//! that the running reconstructed plane is a Vec-backed buffer sized
//! to the per-frame extent rather than `[[u8; 16]; 16]`.
//!
//! Scope mirrors the r230 + r233 + r234 encoder exactly:
//!
//!   * `subsampling_x = subsampling_y = 1` (4:2:0), `bit_depth = 8`,
//!     not monochrome.
//!   * `frame_width`, `frame_height` ∈ {8, 16, 24, 32, 40, 48, 56,
//!     64}; both must be multiples of 8 (the 4:2:0 chroma
//!     constraint). Width and height are independent — rectangular
//!     frame extents (`8×16`, `16×32`, `24×40`, `32×64`, ...) ride
//!     the smallest power-of-two super-block covering
//!     `max(mi_cols, mi_rows)` with out-of-frame quadrants swallowed
//!     by the §5.11.4 per-quadrant early return.
//!   * `base_q_idx ∈ 0..=255`. `== 0` is the §5.9.2 `CodedLossless`
//!     arm (encoder uses forward WHT, decoder uses the §7.13.2.10
//!     inverse WHT); `> 0` is the §7.13.3 inverse DCT_DCT lossy
//!     arm (decoder threads `lossless` through every leaf's
//!     `inverse_transform_2d` based on the parsed FH `base_q_idx`).
//!   * Intra-only, single tile, BLOCK_4X4 leaves, TX_4X4 (no
//!     rectangular **TX_SIZE** family), default scan, no
//!     segmentation, no QM, no in-loop filters.
//!   * 13-mode intra picker on luma + chroma (the r228/r229 picker)
//!     plus the r232 §7.11.5.3 `UV_CFL_PRED` arm on chroma.
//!
//! Outside that scope, returns [`Error::PartitionWalkOutOfRange`].

use crate::cdf::{
    cfl_allowed_for_uv_mode, cfl_alpha_u_ctx, cfl_alpha_v_ctx, dequantize_step1, partition_ctx,
    partition_subsize, predict_intra_d_mode, predict_intra_dc_pred, predict_intra_h_pred,
    predict_intra_paeth_pred, predict_intra_smooth_h_pred, predict_intra_smooth_pred,
    predict_intra_smooth_v_pred, predict_intra_v_pred, size_group, skip_ctx, split_or_horz_cdf,
    split_or_vert_cdf, PartitionWalker, QuantizerParams, TileCdfContext, TileGeometry, BLOCK_4X4,
    BLOCK_64X64, BLOCK_8X8, BLOCK_INVALID, BLOCK_SIZES, CFL_ALPHABET_SIZE, CFL_JOINT_SIGNS,
    D45_PRED, D67_PRED, DCT_DCT, DC_PRED, H_PRED, MI_HEIGHT_LOG2, MI_WIDTH_LOG2,
    NUM_4X4_BLOCKS_HIGH, NUM_4X4_BLOCKS_WIDE, PAETH_PRED, PARTITION_HORZ, PARTITION_NONE,
    PARTITION_SPLIT, PARTITION_VERT, SMOOTH_H_PRED, SMOOTH_PRED, SMOOTH_V_PRED, TX_4X4,
    TX_CLASS_2D, UV_CFL_PRED, V_PRED,
};
use crate::decoder::pixel_driver::Frame;
use crate::encoder::pixel_driver::NUM_INTRA_MODES_Y;
use crate::encoder::pixel_driver_dyn::{
    root_super_block, sb_grid_origins, MAX_DIM, MAX_DIM_YUV_MULTI_SB, MAX_DIM_Y_MULTI_SB, MIN_DIM,
};
use crate::encoder::tile_group_obu::parse_tile_group_obu_body;
use crate::film_grain::film_grain_synthesis;
use crate::frame_header::FrameHeader;
use crate::loop_filter::PlaneBuffer;
use crate::scan::get_default_scan;
use crate::sequence_header::SequenceHeader;
use crate::superres::{upscale_frame, SuperresFrameContext};
use crate::symbol_decoder::SymbolDecoder;
use crate::transform::inverse_transform_2d;
use crate::Error;

/// §7.18.3 film-grain synthesis post-processing pass for the dyn 4:2:0
/// driver.
///
/// Runs after §5.11.33 reconstruction has filled the three `u8` plane
/// buffers, mirroring the spec's "decode_frame() → film_grain_synthesis"
/// ordering (av1-spec §7.4). The pass is gated on the parsed frame
/// header's `film_grain_params.apply_grain`: when the flag is clear (the
/// common case, and every prior-arc IVF fixture) the call is a verbatim
/// no-op and the reconstructed bytes flow through unchanged, preserving
/// byte-for-byte parity with the pre-r373 output.
///
/// When `apply_grain == 1`, the `u8` planes are promoted to the `i32`
/// [`PlaneBuffer`] shape the §7.18.3 driver consumes, blended in place by
/// [`film_grain_synthesis`], then `Clip1`-narrowed back to `u8`. The
/// driver itself owns the §7.18.3.2 LFSR, §7.18.3.3 grain build,
/// §7.18.3.4 scaling LUT, and §7.18.3.5 noise blend; this bridge only
/// marshals plane shapes and reads the frame/sequence-header fields the
/// driver needs (`bit_depth`, `num_planes`, `subsampling_*`,
/// `matrix_coefficients` per §5.5.2).
#[allow(clippy::too_many_arguments)]
fn post_process_film_grain_420(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    width: u32,
    height: u32,
    chroma_width: usize,
    chroma_height: usize,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) {
    let fg = match fh.film_grain_params.as_ref() {
        Some(fg) if fg.apply_grain => fg,
        // No grain params, or apply_grain == 0 ⇒ §7.18.3 is a no-op.
        _ => return,
    };

    let cc = &seq.color_config;
    // Promote the u8 planes to the i32 PlaneBuffer shape the §7.18.3
    // driver consumes.
    let mut y_i32: Vec<i32> = recon_y.iter().map(|&v| i32::from(v)).collect();
    let mut u_i32: Vec<i32> = recon_u.iter().map(|&v| i32::from(v)).collect();
    let mut v_i32: Vec<i32> = recon_v.iter().map(|&v| i32::from(v)).collect();

    {
        let mut planes = [
            PlaneBuffer {
                rows: height,
                cols: width,
                samples: &mut y_i32,
            },
            PlaneBuffer {
                rows: chroma_height as u32,
                cols: chroma_width as u32,
                samples: &mut u_i32,
            },
            PlaneBuffer {
                rows: chroma_height as u32,
                cols: chroma_width as u32,
                samples: &mut v_i32,
            },
        ];
        film_grain_synthesis(
            fg,
            cc.bit_depth,
            cc.num_planes,
            u8::from(cc.subsampling_x),
            u8::from(cc.subsampling_y),
            cc.matrix_coefficients,
            &mut planes,
        );
    }

    // Clip1-narrow back to u8 (bit_depth == 8 on this arc).
    for (dst, src) in recon_y.iter_mut().zip(y_i32.iter()) {
        *dst = (*src).clamp(0, 255) as u8;
    }
    for (dst, src) in recon_u.iter_mut().zip(u_i32.iter()) {
        *dst = (*src).clamp(0, 255) as u8;
    }
    for (dst, src) in recon_v.iter_mut().zip(v_i32.iter()) {
        *dst = (*src).clamp(0, 255) as u8;
    }
}

/// §7.18.3 film-grain synthesis post-processing for the monochrome dyn-y
/// driver.
///
/// Monochrome (`num_planes == 1`) blends grain into the single luma plane
/// only; §7.18.3 suppresses chroma when `mono_chrome`. Gated identically
/// to [`post_process_film_grain_420`] on `apply_grain`, so the mono path
/// keeps byte-for-byte parity with prior arcs whenever grain is off.
fn post_process_film_grain_mono(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    width: u32,
    height: u32,
    recon_y: &mut [u8],
) {
    let fg = match fh.film_grain_params.as_ref() {
        Some(fg) if fg.apply_grain => fg,
        _ => return,
    };
    let cc = &seq.color_config;
    let mut y_i32: Vec<i32> = recon_y.iter().map(|&v| i32::from(v)).collect();
    {
        let mut planes = [PlaneBuffer {
            rows: height,
            cols: width,
            samples: &mut y_i32,
        }];
        film_grain_synthesis(
            fg,
            cc.bit_depth,
            cc.num_planes,
            u8::from(cc.subsampling_x),
            u8::from(cc.subsampling_y),
            cc.matrix_coefficients,
            &mut planes,
        );
    }
    for (dst, src) in recon_y.iter_mut().zip(y_i32.iter()) {
        *dst = (*src).clamp(0, 255) as u8;
    }
}

/// Result of [`post_process_superres_420`]: the (possibly widened) output
/// luma width and the three upscaled 4:2:0 plane buffers.
struct SuperresOutput {
    width: u32,
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
}

/// §7.16 horizontal superres upscaling post-processing for the dyn 4:2:0
/// driver.
///
/// In §7.4 decode order superres runs **before** film-grain synthesis
/// (deblock → CDEF → superres → loop-restoration → film-grain). On the
/// lossless intra dyn arc deblock / CDEF / LR are no-ops, so this pass
/// sits directly before [`post_process_film_grain_420`].
///
/// When `use_superres == 0` (or `frame_width == upscaled_width`) the §7.16
/// driver short-circuits to a verbatim copy, so the function returns the
/// reconstructed planes unchanged and the output width stays `frame_width`
/// — byte-for-byte parity with every prior-arc fixture.
///
/// When `use_superres == 1`, each plane is upscaled horizontally from its
/// `frame_width`-derived width to its `upscaled_width`-derived width via
/// [`upscale_frame`]; heights are unchanged (superres is horizontal only).
/// The returned tuple is `(upscaled_luma_width, [y, u, v])` with the
/// chroma planes at `(upscaled_width + 1) >> 1` columns.
fn post_process_superres_420(
    seq: &SequenceHeader,
    fs: &crate::frame_header::FrameSize,
    recon_y: Vec<u8>,
    recon_u: Vec<u8>,
    recon_v: Vec<u8>,
) -> Result<SuperresOutput, Error> {
    let frame_width = fs.frame_width;
    let upscaled_width = fs.upscaled_width;
    let height = fs.frame_height;

    // §7.16 short-circuit — no upscale required.
    if !fs.use_superres || frame_width == upscaled_width {
        return Ok(SuperresOutput {
            width: frame_width,
            y: recon_y,
            u: recon_u,
            v: recon_v,
        });
    }
    if upscaled_width <= frame_width {
        return Err(Error::PartitionWalkOutOfRange);
    }

    let cc = &seq.color_config;
    // 4:2:0 chroma widths: Round2(width, subX) with subX == 1.
    let in_cw = ((frame_width + 1) >> 1) as usize;
    let out_cw = ((upscaled_width + 1) >> 1) as usize;
    let ch = ((height + 1) >> 1) as usize;

    // Promote inputs to the i32 PlaneBuffer shape the §7.16 driver
    // consumes.
    let mut y_in: Vec<i32> = recon_y.iter().map(|&v| i32::from(v)).collect();
    let mut u_in: Vec<i32> = recon_u.iter().map(|&v| i32::from(v)).collect();
    let mut v_in: Vec<i32> = recon_v.iter().map(|&v| i32::from(v)).collect();

    let mut y_out = vec![0i32; (upscaled_width as usize) * (height as usize)];
    let mut u_out = vec![0i32; out_cw * ch];
    let mut v_out = vec![0i32; out_cw * ch];

    let ctx = SuperresFrameContext {
        use_superres: true,
        frame_width,
        upscaled_width,
        frame_height: height,
        mi_cols: fs.mi_cols,
        num_planes: cc.num_planes,
        bit_depth: cc.bit_depth,
        subsampling_x: u8::from(cc.subsampling_x),
        subsampling_y: u8::from(cc.subsampling_y),
    };

    {
        let inputs = [
            PlaneBuffer {
                rows: height,
                cols: frame_width,
                samples: &mut y_in,
            },
            PlaneBuffer {
                rows: ch as u32,
                cols: in_cw as u32,
                samples: &mut u_in,
            },
            PlaneBuffer {
                rows: ch as u32,
                cols: in_cw as u32,
                samples: &mut v_in,
            },
        ];
        let mut outputs = [
            PlaneBuffer {
                rows: height,
                cols: upscaled_width,
                samples: &mut y_out,
            },
            PlaneBuffer {
                rows: ch as u32,
                cols: out_cw as u32,
                samples: &mut u_out,
            },
            PlaneBuffer {
                rows: ch as u32,
                cols: out_cw as u32,
                samples: &mut v_out,
            },
        ];
        upscale_frame(&ctx, &inputs, &mut outputs).map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    let y = y_out.iter().map(|&v| v.clamp(0, 255) as u8).collect();
    let u = u_out.iter().map(|&v| v.clamp(0, 255) as u8).collect();
    let v = v_out.iter().map(|&v| v.clamp(0, 255) as u8).collect();
    Ok(SuperresOutput {
        width: upscaled_width,
        y,
        u,
        v,
    })
}

/// Decode one dynamic-extent intra-only frame and surface it as
/// [`Frame::Yuv420Dyn`]. Called from [`super::pixel_driver::decode_frame`]
/// for any `(frame_width, frame_height) != (16, 16)`.
///
/// Pre-conditions: the caller has already verified
/// `mono_chrome == false`, `subsampling_x == subsampling_y == 1`.
pub(crate) fn decode_frame_dyn(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    tile_group_body: &[u8],
) -> Result<Frame, Error> {
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let width = fs.frame_width;
    let height = fs.frame_height;
    // r214: ceiling lifted from `MAX_DIM` (64) to
    // `MAX_DIM_YUV_MULTI_SB` (128) on the 4:2:0 YUV dyn path. Extents
    // in `(MAX_DIM, MAX_DIM_YUV_MULTI_SB]` (i.e. > 64×64) dispatch to
    // the §5.11.1-conformant SB-grid walk via the per-SB
    // `decode_partition_node(... BLOCK_64X64)` loop below; extents
    // ≤ 64×64 keep the single-root behaviour from prior arcs for IVF
    // byte-for-byte parity with prior outputs.
    if width < MIN_DIM
        || height < MIN_DIM
        || width > MAX_DIM_YUV_MULTI_SB
        || height > MAX_DIM_YUV_MULTI_SB
        || width % MIN_DIM != 0
        || height % MIN_DIM != 0
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let chroma_width = (width / 2) as usize;
    let chroma_height = (height / 2) as usize;
    let mi_cols = fs.mi_cols;
    let mi_rows = fs.mi_rows;
    // r233: lossy-quant support on the dyn driver. The decoder now
    // reads §5.9.12 `base_q_idx` from the parsed FH and dispatches the
    // leaf inverse transform via `inverse_transform_2d`'s `lossless`
    // flag (`true` ⇔ `base_q_idx == 0` per the dyn-driver scope; no
    // per-segment / per-block delta_q this arc). At `base_q_idx > 0`
    // the encoder uses §7.13.3 forward DCT_DCT + §7.12.3 forward
    // quantize; the decoder mirrors via §7.12.3 dequantize_step1 + the
    // §7.13.3 inverse DCT_DCT (lossy arm). Only `base_q_idx` is read —
    // the per-plane / per-segment `delta_q_*` slots stay at zero per
    // this arc's scope.
    let qp_fh = fh
        .quantization_params
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let q_params = QuantizerParams::neutral(qp_fh.base_q_idx, 8);

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

    let mut decoder =
        SymbolDecoder::init_symbol(tile_bytes, tile_bytes.len(), fh.disable_cdf_update)?;
    let mut cdfs = TileCdfContext::new_from_defaults();

    let mut state = DecoderStateDyn::new(mi_rows, mi_cols);
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

    let mut recon_y = vec![0u8; (width as usize) * (height as usize)];
    let mut recon_u = vec![0u8; chroma_width * chroma_height];
    let mut recon_v = vec![0u8; chroma_width * chroma_height];
    let scan: Vec<u16> = get_default_scan(TX_4X4).to_vec();

    // r214 dispatch — extents > MAX_DIM (64) take the §5.11.1
    // multi-SB row-major walk; extents ≤ MAX_DIM keep the single-
    // root behaviour from prior arcs for IVF byte-for-byte parity.
    if width > MAX_DIM || height > MAX_DIM {
        for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
            decode_partition_node(
                &mut decoder,
                &mut cdfs,
                &mut state,
                &mut coeff_walker,
                sb_r,
                sb_c,
                BLOCK_64X64,
                width as usize,
                height as usize,
                chroma_width,
                chroma_height,
                &scan,
                &q_params,
                &mut recon_y,
                &mut recon_u,
                &mut recon_v,
            )?;
        }
    } else {
        let root_b = root_super_block(mi_cols, mi_rows);
        decode_partition_node(
            &mut decoder,
            &mut cdfs,
            &mut state,
            &mut coeff_walker,
            0,
            0,
            root_b,
            width as usize,
            height as usize,
            chroma_width,
            chroma_height,
            &scan,
            &q_params,
            &mut recon_y,
            &mut recon_u,
            &mut recon_v,
        )?;
    }

    // §7.16 superres upscaling (no-op when use_superres == 0). Runs
    // before film grain per §7.4 decode order. May widen the planes from
    // `frame_width` to `upscaled_width`.
    let sr = post_process_superres_420(seq, fs, recon_y, recon_u, recon_v)?;
    let out_width = sr.width;
    let (mut recon_y, mut recon_u, mut recon_v) = (sr.y, sr.u, sr.v);
    let out_chroma_width = ((out_width + 1) >> 1) as usize;

    // §7.18.3 film-grain synthesis post-processing (no-op when
    // apply_grain == 0, which is every prior-arc fixture).
    post_process_film_grain_420(
        seq,
        fh,
        out_width,
        height,
        out_chroma_width,
        chroma_height,
        &mut recon_y,
        &mut recon_u,
        &mut recon_v,
    );

    Ok(Frame::Yuv420Dyn {
        width: out_width,
        height,
        y: recon_y,
        u: recon_u,
        v: recon_v,
    })
}

/// Per-frame dyn decoder state — mirror of the encoder's
/// `PartitionTreeWriter` `MiSizes[]` grid.
#[derive(Debug)]
struct DecoderStateDyn {
    mi_rows: u32,
    mi_cols: u32,
    mi_sizes: Vec<usize>,
    geometry: TileGeometry,
}

impl DecoderStateDyn {
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

#[allow(clippy::too_many_arguments)]
fn decode_partition_node(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    state: &mut DecoderStateDyn,
    coeff_walker: &mut PartitionWalker,
    r: u32,
    c: u32,
    b_size: usize,
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) -> Result<(), Error> {
    // §5.11.4 line 1.
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
            state.stamp_mi_sizes(r, c, sub_size);
            decode_block_leaf(
                decoder,
                cdfs,
                coeff_walker,
                r,
                c,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
        }
        PARTITION_SPLIT => {
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c + half_block4x4,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
            decode_partition_node(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c + half_block4x4,
                sub_size,
                width,
                height,
                chroma_width,
                chroma_height,
                scan,
                qp,
                recon_y,
                recon_u,
                recon_v,
            )?;
        }
        _ => return Err(Error::PartitionWalkOutOfRange),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_block_leaf(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    coeff_walker: &mut PartitionWalker,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [u8],
    recon_u: &mut [u8],
    recon_v: &mut [u8],
) -> Result<(), Error> {
    // r233: §5.9.2 `CodedLossless` predicate per the dyn driver scope
    // (no per-segment / per-block delta_q this arc ⇒ the FH-level
    // `base_q_idx == 0` is the whole predicate). Drives the
    // `inverse_transform_2d` dispatch — WHT path at lossless,
    // §7.13.3 inverse DCT_DCT otherwise.
    let lossless = qp.base_q_idx == 0;
    let skip_ctx_val = skip_ctx(0, 0);
    let skip = {
        let cdf = cdfs.skip_cdf(skip_ctx_val);
        decoder.read_symbol(cdf)? as u8
    };
    let size_group_ctx = size_group(sub_size);
    let y_mode = {
        let cdf = cdfs
            .y_mode_cdf(size_group_ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        decoder.read_symbol(cdf)? as u8
    };
    if (y_mode as usize) >= NUM_INTRA_MODES_Y {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let has_chroma = (mi_row & 1) != 0 && (mi_col & 1) != 0;
    // r232: dyn driver now accepts `UV_CFL_PRED` (ordinal 13, the
    // §7.11.5.3 chroma-from-luma αU/αV predictor) in addition to the
    // 13 §6.10.x intra modes. When the §8.3.2 CFL-allowed CDF row
    // surfaces ordinal 13, the §5.11.22 line-8 gate fires
    // `read_cfl_alphas()` (§5.11.45) for the per-block CflAlphaU /
    // CflAlphaV; the chroma dispatcher below then routes through
    // `cfl_predict_4x4_for_plane_dyn` instead of one of the §6.10.x
    // kernels.
    let (uv_mode, cfl_alpha_u, cfl_alpha_v): (u8, i8, i8) = if has_chroma {
        let cfl_allowed = cfl_allowed_for_uv_mode(false, sub_size, true, true);
        let cdf = cdfs
            .uv_mode_cdf(cfl_allowed, y_mode as usize)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let m = decoder.read_symbol(cdf)? as u8;
        // Encoder picks from `0..=UV_CFL_PRED` (the §3 INTRA_MODES set
        // plus UV_CFL_PRED = 13). Any out-of-set decoded mode is a
        // contract violation — reject.
        if (m as usize) > UV_CFL_PRED {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if (m as usize) == UV_CFL_PRED {
            let (au, av) = read_cfl_alphas(decoder, cdfs)?;
            (m, au, av)
        } else {
            (m, 0, 0)
        }
    } else {
        (DC_PRED as u8, 0, 0)
    };

    let mut quant_y = vec![0i32; 16];
    let mut quant_u = vec![0i32; 16];
    let mut quant_v = vec![0i32; 16];

    let _readout_y = coeff_walker.coefficients(
        decoder,
        cdfs,
        /* plane = */ 0,
        0,
        TX_4X4,
        TX_CLASS_2D,
        0,
        0,
        scan,
        &mut quant_y,
    )?;

    let (row0, col0) = ((mi_row as usize) * 4, (mi_col as usize) * 4);
    let pred_y = predict_intra_for_mode_4x4(
        recon_y,
        width,
        height,
        mi_row as usize,
        mi_col as usize,
        y_mode as usize,
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;
    if skip == 0 {
        let dequant = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, qp);
        let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, lossless);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + residual[i * 4 + j];
                recon_y[(row0 + i) * width + (col0 + j)] = p.clamp(0, 255) as u8;
            }
        }
    } else {
        for i in 0..4 {
            for j in 0..4 {
                recon_y[(row0 + i) * width + (col0 + j)] = pred_y[i * 4 + j];
            }
        }
    }

    if has_chroma {
        let cr = ((mi_row as usize) - 1) / 2;
        let cc_idx = ((mi_col as usize) - 1) / 2;
        let crow0 = cr * 4;
        let ccol0 = cc_idx * 4;

        // r232: §7.11.5.3 subsampled-luma window + lumaAvg are shared
        // across U and V. Compute once (only when UV_CFL_PRED won at
        // the encoder side); otherwise the helpers stay unused
        // (`cfl_alpha_* == 0`).
        let (l_arr, luma_avg) = if uv_mode as usize == UV_CFL_PRED {
            cfl_subsampled_luma_4x4_420_dyn(recon_y, width, height, cr, cc_idx)
        } else {
            ([0i32; 16], 0)
        };

        let _readout_u = coeff_walker.coefficients(
            decoder,
            cdfs,
            1,
            0,
            TX_4X4,
            TX_CLASS_2D,
            0,
            0,
            scan,
            &mut quant_u,
        )?;
        // r232: when `uv_mode == UV_CFL_PRED` the §7.11.5.3 path takes
        // DC_PRED as the `dc` base and adds the alpha-scaled luma
        // residual.
        let pred_u: [u8; 16] = if uv_mode as usize == UV_CFL_PRED {
            let dc_pred = predict_intra_for_mode_4x4(
                recon_u,
                chroma_width,
                chroma_height,
                cr,
                cc_idx,
                DC_PRED,
            )
            .ok_or(Error::PartitionWalkOutOfRange)?;
            cfl_predict_4x4_for_plane_dyn(&dc_pred, &l_arr, luma_avg, cfl_alpha_u)
        } else {
            predict_intra_for_mode_4x4(
                recon_u,
                chroma_width,
                chroma_height,
                cr,
                cc_idx,
                uv_mode as usize,
            )
            .ok_or(Error::PartitionWalkOutOfRange)?
        };
        if skip == 0 {
            let dequant = dequantize_step1(&quant_u, TX_4X4, 1, 0, DCT_DCT, 15, qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, lossless);
            for i in 0..4 {
                for j in 0..4 {
                    let p = pred_u[i * 4 + j] as i64 + residual[i * 4 + j];
                    recon_u[(crow0 + i) * chroma_width + (ccol0 + j)] = p.clamp(0, 255) as u8;
                }
            }
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    recon_u[(crow0 + i) * chroma_width + (ccol0 + j)] = pred_u[i * 4 + j];
                }
            }
        }

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
        let pred_v: [u8; 16] = if uv_mode as usize == UV_CFL_PRED {
            let dc_pred = predict_intra_for_mode_4x4(
                recon_v,
                chroma_width,
                chroma_height,
                cr,
                cc_idx,
                DC_PRED,
            )
            .ok_or(Error::PartitionWalkOutOfRange)?;
            cfl_predict_4x4_for_plane_dyn(&dc_pred, &l_arr, luma_avg, cfl_alpha_v)
        } else {
            predict_intra_for_mode_4x4(
                recon_v,
                chroma_width,
                chroma_height,
                cr,
                cc_idx,
                uv_mode as usize,
            )
            .ok_or(Error::PartitionWalkOutOfRange)?
        };
        if skip == 0 {
            let dequant = dequantize_step1(&quant_v, TX_4X4, 2, 0, DCT_DCT, 15, qp);
            let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, lossless);
            for i in 0..4 {
                for j in 0..4 {
                    let p = pred_v[i * 4 + j] as i64 + residual[i * 4 + j];
                    recon_v[(crow0 + i) * chroma_width + (ccol0 + j)] = p.clamp(0, 255) as u8;
                }
            }
        } else {
            for i in 0..4 {
                for j in 0..4 {
                    recon_v[(crow0 + i) * chroma_width + (ccol0 + j)] = pred_v[i * 4 + j];
                }
            }
        }
    }

    // Silence the unused-import warning on BLOCK_4X4 (this driver
    // operates at the leaf level but never names the constant directly).
    let _ = BLOCK_4X4;
    Ok(())
}

/// §7.11.2.1 prologue + §7.11.2.{2..6} kernel dispatcher for one 4×4
/// cell against a dynamic-extent plane. Mirror of the encoder-side
/// helpers in [`crate::encoder::pixel_driver_dyn`].
fn predict_intra_for_mode_4x4(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
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
    let stride = plane_width;

    let have_above = (row0 > 0) as u8;
    let have_left = (col0 > 0) as u8;
    let above_left: u16 = if have_above != 0 && have_left != 0 {
        plane[(row0 - 1) * stride + (col0 - 1)] as u16
    } else if have_above != 0 {
        plane[(row0 - 1) * stride + col0] as u16
    } else if have_left != 0 {
        plane[row0 * stride + (col0 - 1)] as u16
    } else {
        1u16 << (bit_depth - 1)
    };

    let mut above_ext = [0u16; 10];
    above_ext[0] = above_left;
    above_ext[1] = above_left;
    if have_above != 0 {
        for k in 0..(w + h) {
            let col = (col0 + k).min(plane_width - 1);
            above_ext[2 + k] = plane[(row0 - 1) * stride + col] as u16;
        }
    } else if have_left != 0 {
        let sample = plane[row0 * stride + (col0 - 1)] as u16;
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
            left_ext[2 + k] = plane[row * stride + (col0 - 1)] as u16;
        }
    } else if have_above != 0 {
        let sample = plane[(row0 - 1) * stride + col0] as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = sample;
        }
    } else {
        let mid_plus = ((1u32 << (bit_depth - 1)) + 1) as u16;
        for slot in left_ext.iter_mut().skip(2).take(w + h) {
            *slot = mid_plus;
        }
    }

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

// ----------------------------------------------------------------------
// r232 — §7.11.5.3 CFL chroma-from-luma prediction (dyn driver edition).
//
// Mirrors the fixed-size driver's helpers (`read_cfl_alphas`,
// `cfl_subsampled_luma_4x4_420`, `round2_signed`,
// `cfl_predict_4x4_for_plane`) but reads from / produces results
// against the Vec-backed dyn plane buffers.
// ----------------------------------------------------------------------

/// §5.11.45 `read_cfl_alphas()` — read the joint sign + per-channel
/// magnitude and reconstruct the signed `(CflAlphaU, CflAlphaV)`.
///
/// Mirror of the fixed-size driver's helper of the same name (see
/// [`crate::decoder::pixel_driver`]). The §5.11.45 / §8.3.2 CDF
/// selection rules are independent of frame extent so the logic
/// matches byte-for-byte.
fn read_cfl_alphas(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
) -> Result<(i8, i8), Error> {
    let signs_cdf = cdfs.cfl_sign_cdf();
    let signs = decoder.read_symbol(signs_cdf)? as usize;
    if signs >= CFL_JOINT_SIGNS {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §5.11.45: signU = (signs + 1) / 3, signV = (signs + 1) % 3 with
    // values in {CFL_SIGN_ZERO=0, CFL_SIGN_NEG=1, CFL_SIGN_POS=2}.
    let sign_u: usize = (signs + 1) / 3;
    let sign_v: usize = (signs + 1) % 3;
    let cfl_alpha_u: i8 = if sign_u != 0 {
        let ctx_u = cfl_alpha_u_ctx(sign_u, sign_v);
        let row_u = cdfs.cfl_alpha_cdf(ctx_u);
        let raw_u = decoder.read_symbol(row_u)? as i32;
        if raw_u as usize >= CFL_ALPHABET_SIZE {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let mag = (1 + raw_u) as i8;
        if sign_u == 1 {
            -mag
        } else {
            mag
        }
    } else {
        0
    };
    let cfl_alpha_v: i8 = if sign_v != 0 {
        let ctx_v = cfl_alpha_v_ctx(sign_u, sign_v);
        let row_v = cdfs.cfl_alpha_cdf(ctx_v);
        let raw_v = decoder.read_symbol(row_v)? as i32;
        if raw_v as usize >= CFL_ALPHABET_SIZE {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let mag = (1 + raw_v) as i8;
        if sign_v == 1 {
            -mag
        } else {
            mag
        }
    } else {
        0
    };
    Ok((cfl_alpha_u, cfl_alpha_v))
}

/// §7.11.5.3 subsampled-luma + lumaAvg derivation for one 4:2:0 TX_4X4
/// chroma block on a Vec-backed luma plane.
///
/// Mirror of the encoder-side helper of the same name in
/// [`crate::encoder::pixel_driver_dyn`].
fn cfl_subsampled_luma_4x4_420_dyn(
    recon_y: &[u8],
    luma_w: usize,
    luma_h: usize,
    cell_row: usize,
    cell_col: usize,
) -> ([i32; 16], i32) {
    let crow0_chroma = cell_row * 4;
    let ccol0_chroma = cell_col * 4;
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
/// (`x < 0` arm round-half-away-from-zero).
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
/// (L[k] - lumaAvg), 6)` clipped to byte. Mirror of the encoder-side
/// helper of the same name.
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

// ----------------------------------------------------------------------
// r235 — Y-only (monochrome) dyn decoder.
// ----------------------------------------------------------------------
//
// Inverse of `crate::encoder::encode_intra_frame_y_dyn`. Mirrors
// [`decode_frame_dyn`] but skips every chroma branch:
//
//   * `decode_block_leaf_y` reads only luma `y_mode` + luma
//     `coefficients()` per leaf — the `uv_mode` / U-V `coefficients()`
//     / `read_cfl_alphas()` reads are gated on `NumPlanes > 1` per the
//     spec, so on `NumPlanes == 1` the bitstream contains no chroma
//     syntax.
//   * The reconstructed frame surfaces as
//     [`super::pixel_driver::Frame::YDyn`] (no U/V planes).
//
// Called from [`super::pixel_driver::decode_frame`] when
// `seq.color_config.mono_chrome == true`.

/// Decode one dynamic-extent monochrome intra-only frame and surface it
/// as [`Frame::YDyn`].
///
/// Pre-conditions (enforced by the caller in
/// [`super::pixel_driver::decode_frame`]):
/// `seq.color_config.mono_chrome == true`, `seq.color_config.num_planes
/// == 1`.
pub(crate) fn decode_frame_dyn_y(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    tile_group_body: &[u8],
) -> Result<Frame, Error> {
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let width = fs.frame_width;
    let height = fs.frame_height;
    // r207: ceiling lifted from `MAX_DIM` (64) to
    // `MAX_DIM_Y_MULTI_SB` (128) on the mono dyn path. Extents in
    // `(MAX_DIM, MAX_DIM_Y_MULTI_SB]` (i.e. > 64×64) dispatch to the
    // §5.11.1-conformant SB-grid walk via the per-SB
    // `decode_partition_node_y(... BLOCK_64X64)` loop below; extents
    // ≤ 64×64 keep the single-root behaviour from r235 for IVF
    // byte-for-byte parity with prior outputs.
    if width < MIN_DIM
        || height < MIN_DIM
        || width > MAX_DIM_Y_MULTI_SB
        || height > MAX_DIM_Y_MULTI_SB
        || width % MIN_DIM != 0
        || height % MIN_DIM != 0
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let mi_cols = fs.mi_cols;
    let mi_rows = fs.mi_rows;
    let qp_fh = fh
        .quantization_params
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let q_params = QuantizerParams::neutral(qp_fh.base_q_idx, 8);

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

    let mut decoder =
        SymbolDecoder::init_symbol(tile_bytes, tile_bytes.len(), fh.disable_cdf_update)?;
    let mut cdfs = TileCdfContext::new_from_defaults();

    let mut state = DecoderStateDyn::new(mi_rows, mi_cols);
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

    let mut recon_y = vec![0u8; (width as usize) * (height as usize)];
    let scan: Vec<u16> = get_default_scan(TX_4X4).to_vec();

    // r207 dispatch — extents > MAX_DIM (64) take the §5.11.1
    // multi-SB row-major walk; extents ≤ MAX_DIM keep the single-
    // root behaviour from r235.
    if width > MAX_DIM || height > MAX_DIM {
        for (sb_r, sb_c) in sb_grid_origins(mi_rows, mi_cols) {
            decode_partition_node_y(
                &mut decoder,
                &mut cdfs,
                &mut state,
                &mut coeff_walker,
                sb_r,
                sb_c,
                BLOCK_64X64,
                width as usize,
                height as usize,
                &scan,
                &q_params,
                &mut recon_y,
            )?;
        }
    } else {
        let root_b = root_super_block(mi_cols, mi_rows);
        decode_partition_node_y(
            &mut decoder,
            &mut cdfs,
            &mut state,
            &mut coeff_walker,
            0,
            0,
            root_b,
            width as usize,
            height as usize,
            &scan,
            &q_params,
            &mut recon_y,
        )?;
    }

    // §7.18.3 film-grain synthesis post-processing (mono / luma-only;
    // no-op when apply_grain == 0).
    post_process_film_grain_mono(seq, fh, width, height, &mut recon_y);

    Ok(Frame::YDyn {
        width,
        height,
        y: recon_y,
    })
}

#[allow(clippy::too_many_arguments)]
fn decode_partition_node_y(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    state: &mut DecoderStateDyn,
    coeff_walker: &mut PartitionWalker,
    r: u32,
    c: u32,
    b_size: usize,
    width: usize,
    height: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [u8],
) -> Result<(), Error> {
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
            state.stamp_mi_sizes(r, c, sub_size);
            decode_block_leaf_y(
                decoder,
                cdfs,
                coeff_walker,
                r,
                c,
                sub_size,
                width,
                height,
                scan,
                qp,
                recon_y,
            )?;
        }
        PARTITION_SPLIT => {
            decode_partition_node_y(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c,
                sub_size,
                width,
                height,
                scan,
                qp,
                recon_y,
            )?;
            decode_partition_node_y(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r,
                c + half_block4x4,
                sub_size,
                width,
                height,
                scan,
                qp,
                recon_y,
            )?;
            decode_partition_node_y(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c,
                sub_size,
                width,
                height,
                scan,
                qp,
                recon_y,
            )?;
            decode_partition_node_y(
                decoder,
                cdfs,
                state,
                coeff_walker,
                r + half_block4x4,
                c + half_block4x4,
                sub_size,
                width,
                height,
                scan,
                qp,
                recon_y,
            )?;
        }
        _ => return Err(Error::PartitionWalkOutOfRange),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_block_leaf_y(
    decoder: &mut SymbolDecoder<'_>,
    cdfs: &mut TileCdfContext,
    coeff_walker: &mut PartitionWalker,
    mi_row: u32,
    mi_col: u32,
    sub_size: usize,
    width: usize,
    height: usize,
    scan: &[u16],
    qp: &QuantizerParams,
    recon_y: &mut [u8],
) -> Result<(), Error> {
    let lossless = qp.base_q_idx == 0;
    let skip_ctx_val = skip_ctx(0, 0);
    let skip = {
        let cdf = cdfs.skip_cdf(skip_ctx_val);
        decoder.read_symbol(cdf)? as u8
    };
    let size_group_ctx = size_group(sub_size);
    let y_mode = {
        let cdf = cdfs
            .y_mode_cdf(size_group_ctx)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        decoder.read_symbol(cdf)? as u8
    };
    if (y_mode as usize) >= NUM_INTRA_MODES_Y {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // §5.11.5 mono arm — no uv_mode / CflAlpha / chroma coefficients
    // read because §5.11.22 line-8 and §5.11.39 walk are gated on
    // `NumPlanes > 1`.

    let mut quant_y = vec![0i32; 16];
    let _readout_y = coeff_walker.coefficients(
        decoder,
        cdfs,
        /* plane = */ 0,
        0,
        TX_4X4,
        TX_CLASS_2D,
        0,
        0,
        scan,
        &mut quant_y,
    )?;

    let (row0, col0) = ((mi_row as usize) * 4, (mi_col as usize) * 4);
    let pred_y = predict_intra_for_mode_4x4(
        recon_y,
        width,
        height,
        mi_row as usize,
        mi_col as usize,
        y_mode as usize,
    )
    .ok_or(Error::PartitionWalkOutOfRange)?;
    if skip == 0 {
        let dequant = dequantize_step1(&quant_y, TX_4X4, 0, 0, DCT_DCT, 15, qp);
        let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, lossless);
        for i in 0..4 {
            for j in 0..4 {
                let p = pred_y[i * 4 + j] as i64 + residual[i * 4 + j];
                recon_y[(row0 + i) * width + (col0 + j)] = p.clamp(0, 255) as u8;
            }
        }
    } else {
        for i in 0..4 {
            for j in 0..4 {
                recon_y[(row0 + i) * width + (col0 + j)] = pred_y[i * 4 + j];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode_av1;
    use crate::encoder::ivf::IvfReader;
    use crate::encoder::pixel_driver_dyn::{encode_intra_frame_yuv_dyn, Yuv420Frame};
    use crate::frame_header::parse_frame_header;
    use crate::obu::{ObuIter, ObuType};
    use crate::sequence_header::parse_sequence_header;
    use crate::uncompressed_header_tail::FilmGrainParams;

    /// Parse the first temporal unit of an encoded IVF stream into its
    /// (SequenceHeader, FrameHeader, tile-group body) triple — the exact
    /// inputs `decode_frame_dyn` consumes. Mirrors the §7.5 TU walk in
    /// `decode_temporal_unit` so the film-grain wiring can be exercised
    /// against a real parsed header pair.
    fn parse_first_tu(ivf_bytes: &[u8]) -> (SequenceHeader, FrameHeader, Vec<u8>) {
        let reader = IvfReader::new(ivf_bytes).expect("ivf header");
        let frames = reader.read_all().expect("ivf frames");
        let payload = &frames[0].payload;
        let mut seq: Option<SequenceHeader> = None;
        let mut fh: Option<FrameHeader> = None;
        let mut tg: Option<Vec<u8>> = None;
        for desc in ObuIter::new(payload) {
            let desc = desc.expect("obu");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("sh"));
                }
                ObuType::FrameHeader => {
                    let s = seq.as_ref().expect("sh before fh");
                    fh = Some(parse_frame_header(desc.payload, s).expect("fh"));
                }
                ObuType::TileGroup => {
                    tg = Some(desc.payload.to_vec());
                    break;
                }
                _ => {}
            }
        }
        (seq.unwrap(), fh.unwrap(), tg.unwrap())
    }

    /// A §5.9.30 film-grain block with `apply_grain == 1` and a single
    /// non-trivial luma scaling point, enough that §7.18.3.5 blends a
    /// non-zero noise field into the reconstructed luma plane.
    fn active_grain_params() -> FilmGrainParams {
        let mut fg = FilmGrainParams::reset();
        fg.apply_grain = true;
        fg.grain_seed = 0x1234;
        fg.num_y_points = 2;
        fg.point_y_value[0] = 0;
        fg.point_y_scaling[0] = 255;
        fg.point_y_value[1] = 255;
        fg.point_y_scaling[1] = 255;
        fg.grain_scaling = 8;
        fg.ar_coeff_lag = 0;
        fg.ar_coeff_shift = 6;
        fg.grain_scale_shift = 0;
        fg
    }

    #[test]
    fn dyn_film_grain_apply_grain_off_is_noop() {
        // grain off (every encoder-produced fixture) ⇒ post-process is a
        // verbatim no-op; the §7.18.3 pass must not perturb the bytes.
        let input = Yuv420Frame::filled(32, 32, 100);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let (seq, fh, tg) = parse_first_tu(&encoded.ivf_bytes);
        // The encoder's FH carries apply_grain == 0 (or None grain).
        if let Some(fg) = fh.film_grain_params.as_ref() {
            assert!(!fg.apply_grain, "encoder fixture must have grain off");
        }
        let frame = decode_frame_dyn(&seq, &fh, &tg).expect("decode");
        match frame {
            Frame::Yuv420Dyn { y, u, v, .. } => {
                assert_eq!(y, input.y);
                assert_eq!(u, input.u);
                assert_eq!(v, input.v);
            }
            _ => panic!("expected Yuv420Dyn"),
        }
    }

    #[test]
    fn dyn_film_grain_apply_grain_on_perturbs_luma() {
        // Same stream, but flip the FH's grain block to apply_grain == 1
        // with a non-trivial scaling point. The §7.18.3 pass must now
        // change the reconstructed luma vs the grain-off decode.
        let input = Yuv420Frame::filled(32, 32, 100);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let (seq, fh_off, tg) = parse_first_tu(&encoded.ivf_bytes);

        let baseline = decode_frame_dyn(&seq, &fh_off, &tg).expect("decode off");

        let mut fh_on = fh_off.clone();
        fh_on.film_grain_params = Some(active_grain_params());
        let grained = decode_frame_dyn(&seq, &fh_on, &tg).expect("decode on");

        let (by, gy) = match (&baseline, &grained) {
            (Frame::Yuv420Dyn { y: by, .. }, Frame::Yuv420Dyn { y: gy, .. }) => (by, gy),
            _ => panic!("expected Yuv420Dyn"),
        };
        assert_eq!(by.len(), gy.len());
        // Grain must have changed at least one luma sample.
        assert!(
            by.iter().zip(gy.iter()).any(|(a, b)| a != b),
            "apply_grain == 1 must perturb the luma plane",
        );
    }

    #[test]
    fn dyn_superres_off_is_noop() {
        // use_superres == 0 ⇒ §7.16 short-circuit: planes + width
        // unchanged, byte-for-byte.
        let input = Yuv420Frame::filled(32, 32, 100);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let (seq, fh, _tg) = parse_first_tu(&encoded.ivf_bytes);
        let fs = fh.frame_size.expect("frame size");
        assert!(!fs.use_superres);
        let y = input.y.clone();
        let u = input.u.clone();
        let v = input.v.clone();
        let sr = post_process_superres_420(&seq, &fs, y.clone(), u.clone(), v.clone())
            .expect("superres");
        assert_eq!(sr.width, fs.frame_width);
        assert_eq!(sr.y, y);
        assert_eq!(sr.u, u);
        assert_eq!(sr.v, v);
    }

    #[test]
    fn dyn_superres_on_widens_and_preserves_flat_plane() {
        // use_superres == 1 with upscaled_width > frame_width: a flat
        // constant plane upscales (8-tap polyphase, taps sum to a
        // normalised unit) to a flat constant plane at the new width,
        // and the output luma plane carries `upscaled_width` columns.
        let input = Yuv420Frame::filled(32, 32, 137);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let (seq, fh, _tg) = parse_first_tu(&encoded.ivf_bytes);
        let mut fs = fh.frame_size.expect("frame size");
        // Synthesise a superres frame: keep frame_width = 32 (the coded /
        // downscaled width the recon walk used) and widen the upscaled
        // target to 48.
        fs.use_superres = true;
        fs.upscaled_width = 48;

        let sr =
            post_process_superres_420(&seq, &fs, input.y.clone(), input.u.clone(), input.v.clone())
                .expect("superres");

        assert_eq!(sr.width, 48);
        // Luma: 48 × 32, all 137 (flat in ⇒ flat out).
        assert_eq!(sr.y.len(), 48 * 32);
        assert!(sr.y.iter().all(|&v| v == 137), "flat luma must stay flat");
        // Chroma: 24 × 16, all 137.
        assert_eq!(sr.u.len(), 24 * 16);
        assert_eq!(sr.v.len(), 24 * 16);
        assert!(sr.u.iter().all(|&v| v == 137));
        assert!(sr.v.iter().all(|&v| v == 137));
    }

    #[test]
    fn dyn_superres_end_to_end_through_decode_frame_dyn_widens_output() {
        // Full decode path: a flat 32×32 frame with the FH's FrameSize
        // mutated to use_superres == 1 / upscaled_width == 48 reconstructs
        // through decode_frame_dyn and surfaces a Yuv420Dyn whose width is
        // the upscaled 48 (proving the §7.16 pass is wired into the public
        // dyn driver, not just callable in isolation).
        let input = Yuv420Frame::filled(32, 32, 80);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let (seq, mut fh, tg) = parse_first_tu(&encoded.ivf_bytes);
        let mut fs = fh.frame_size.expect("frame size");
        fs.use_superres = true;
        fs.upscaled_width = 48;
        fh.frame_size = Some(fs);

        let frame = decode_frame_dyn(&seq, &fh, &tg).expect("decode");
        match frame {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(width, 48);
                assert_eq!(height, 32);
                assert_eq!(y.len(), 48 * 32);
                assert_eq!(u.len(), 24 * 16);
                assert_eq!(v.len(), 24 * 16);
                // Flat reconstruction upscales to a flat plane.
                assert!(y.iter().all(|&p| p == 80));
            }
            _ => panic!("expected Yuv420Dyn"),
        }
    }

    #[test]
    fn dyn_superres_rejects_upscaled_not_larger() {
        // use_superres == 1 but upscaled_width <= frame_width is a
        // §7.16 conformance violation; the bridge rejects it rather than
        // silently producing a narrower / equal-width frame.
        let input = Yuv420Frame::filled(32, 32, 100);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let (seq, fh, _tg) = parse_first_tu(&encoded.ivf_bytes);
        let mut fs = fh.frame_size.expect("frame size");
        fs.use_superres = true;
        fs.upscaled_width = 24; // < frame_width (32) — invalid.
        let r =
            post_process_superres_420(&seq, &fs, input.y.clone(), input.u.clone(), input.v.clone());
        assert!(matches!(r, Err(Error::PartitionWalkOutOfRange)));
    }

    #[test]
    fn dyn_superres_then_film_grain_compose_in_decode_order() {
        // §7.4 decode order: superres precedes film grain. With both
        // active, the output must be (a) at the upscaled width AND (b)
        // perturbed by grain — proving grain runs on the upscaled planes,
        // not the pre-superres ones.
        let input = Yuv420Frame::filled(32, 32, 100);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let (seq, mut fh, tg) = parse_first_tu(&encoded.ivf_bytes);
        let mut fs = fh.frame_size.expect("frame size");
        fs.use_superres = true;
        fs.upscaled_width = 48;
        fh.frame_size = Some(fs);

        // Baseline: superres only (grain off).
        let baseline = decode_frame_dyn(&seq, &fh, &tg).expect("decode superres-only");

        // Now flip grain on as well.
        let mut fh_grain = fh.clone();
        fh_grain.film_grain_params = Some(active_grain_params());
        let composed = decode_frame_dyn(&seq, &fh_grain, &tg).expect("decode superres+grain");

        let (bw, by) = match &baseline {
            Frame::Yuv420Dyn { width, y, .. } => (*width, y),
            _ => panic!("expected Yuv420Dyn"),
        };
        let (cw, cy) = match &composed {
            Frame::Yuv420Dyn { width, y, .. } => (*width, y),
            _ => panic!("expected Yuv420Dyn"),
        };
        // Both at the upscaled width.
        assert_eq!(bw, 48);
        assert_eq!(cw, 48);
        assert_eq!(by.len(), 48 * 32);
        assert_eq!(cy.len(), 48 * 32);
        // Grain perturbed the upscaled luma.
        assert!(
            by.iter().zip(cy.iter()).any(|(a, b)| a != b),
            "film grain must perturb the post-superres luma plane",
        );
    }

    #[test]
    fn dyn_decode_flat_grey_16x16_via_dyn_driver_roundtrip() {
        // 16×16 still goes through the fixed-size driver, but the
        // r230 dyn encoder synthesises its own (different) SH/FH that
        // forces the dyn dispatcher branch in `decode_frame` — under
        // r230's SH, mi_cols/rows are 4 so it falls back through the
        // fixed driver. Confirm it still roundtrips.
        let input = Yuv420Frame::filled(16, 16, 128);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        assert_eq!(decoded.len(), 1);
        match &decoded[0] {
            Frame::Yuv420_16x16 { y, .. } => {
                // The dyn encoder pinned mi=4 at 16×16 so the fixed
                // path took over; check the recovered luma plane
                // matches the input.
                let mut expected = [[0u8; 16]; 16];
                for (i, row) in expected.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        *cell = input.y[i * 16 + j];
                    }
                }
                assert_eq!(y, &expected);
            }
            Frame::Yuv420Dyn { .. } => panic!("16×16 should route through fixed driver"),
            _ => panic!("unexpected Frame variant"),
        }
    }

    #[test]
    fn dyn_decode_flat_grey_32x32_roundtrip() {
        let input = Yuv420Frame::filled(32, 32, 128);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
                assert_eq!(y, &input.y);
                assert_eq!(u, &input.u);
                assert_eq!(v, &input.v);
            }
            _ => panic!("expected Yuv420Dyn for 32×32"),
        }
    }

    #[test]
    fn dyn_decode_pseudorandom_32x32_roundtrip_bit_exact() {
        let mut input = Yuv420Frame::filled(32, 32, 0);
        let mut state: u64 = 0xDEAD_BEEF_C0FE_BABE;
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
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
                assert_eq!(y, &input.y, "Y mismatch at 32×32");
                assert_eq!(u, &input.u, "U mismatch at 32×32");
                assert_eq!(v, &input.v, "V mismatch at 32×32");
            }
            _ => panic!("expected Yuv420Dyn for 32×32"),
        }
    }

    #[test]
    fn dyn_decode_flat_grey_64x64_roundtrip() {
        let input = Yuv420Frame::filled(64, 64, 200);
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(y, &input.y);
                assert_eq!(u, &input.u);
                assert_eq!(v, &input.v);
            }
            _ => panic!("expected Yuv420Dyn for 64×64"),
        }
    }

    #[test]
    fn dyn_decode_pseudorandom_64x64_roundtrip_bit_exact() {
        let mut input = Yuv420Frame::filled(64, 64, 0);
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
        let encoded = encode_intra_frame_yuv_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::Yuv420Dyn {
                width,
                height,
                y,
                u,
                v,
            } => {
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(y, &input.y, "Y mismatch at 64×64");
                assert_eq!(u, &input.u, "U mismatch at 64×64");
                assert_eq!(v, &input.v, "V mismatch at 64×64");
            }
            _ => panic!("expected Yuv420Dyn for 64×64"),
        }
    }

    // ---- r235 Y-only dyn roundtrips ----

    fn pseudo_random_plane(seed: u64, len: usize) -> Vec<u8> {
        let mut s = seed;
        let mut out = vec![0u8; len];
        for slot in out.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = (s >> 56) as u8;
        }
        out
    }

    #[test]
    fn dyn_y_decode_flat_grey_32x32_roundtrip() {
        use crate::encoder::pixel_driver_dyn::{encode_intra_frame_y_dyn, MonoYFrame};
        let input = MonoYFrame::filled(32, 32, 128);
        let encoded = encode_intra_frame_y_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::YDyn { width, height, y } => {
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
                assert_eq!(y, &input.y);
            }
            other => panic!("expected YDyn, got {other:?}"),
        }
    }

    #[test]
    fn dyn_y_decode_pseudorandom_32x32_roundtrip_bit_exact() {
        use crate::encoder::pixel_driver_dyn::{encode_intra_frame_y_dyn, MonoYFrame};
        let input = MonoYFrame {
            width: 32,
            height: 32,
            y: pseudo_random_plane(0xCAFE_F00D_BAAD_F00D, 32 * 32),
        };
        let encoded = encode_intra_frame_y_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::YDyn { width, height, y } => {
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
                assert_eq!(y, &input.y, "Y mismatch at 32×32 mono");
            }
            other => panic!("expected YDyn, got {other:?}"),
        }
    }

    #[test]
    fn dyn_y_decode_pseudorandom_64x64_roundtrip_bit_exact() {
        use crate::encoder::pixel_driver_dyn::{encode_intra_frame_y_dyn, MonoYFrame};
        let input = MonoYFrame {
            width: 64,
            height: 64,
            y: pseudo_random_plane(0xABCD_1234_5678_9F0E, 64 * 64),
        };
        let encoded = encode_intra_frame_y_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::YDyn { width, height, y } => {
                assert_eq!(*width, 64);
                assert_eq!(*height, 64);
                assert_eq!(y, &input.y, "Y mismatch at 64×64 mono");
            }
            other => panic!("expected YDyn, got {other:?}"),
        }
    }

    #[test]
    fn dyn_y_decode_rectangular_24x40_roundtrip_bit_exact() {
        use crate::encoder::pixel_driver_dyn::{encode_intra_frame_y_dyn, MonoYFrame};
        let input = MonoYFrame {
            width: 24,
            height: 40,
            y: pseudo_random_plane(0xFEED_F00D_DEAD_BEEF, 24 * 40),
        };
        let encoded = encode_intra_frame_y_dyn(&input).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::YDyn { width, height, y } => {
                assert_eq!(*width, 24);
                assert_eq!(*height, 40);
                assert_eq!(y, &input.y, "Y mismatch at 24×40 mono");
            }
            other => panic!("expected YDyn, got {other:?}"),
        }
    }

    #[test]
    fn dyn_y_decode_lossy_q32_smoke() {
        use crate::encoder::pixel_driver_dyn::{encode_intra_frame_y_dyn_with_q, MonoYFrame};
        // At base_q_idx > 0 the lossless WHT bit-exact property goes
        // away (the §7.13.3 forward DCT + §7.12.3 quantize/dequant
        // chain is lossy by design); just verify the roundtrip
        // completes and surfaces a YDyn frame.
        let input = MonoYFrame {
            width: 32,
            height: 32,
            y: pseudo_random_plane(0x1111_2222_3333_4444, 32 * 32),
        };
        let encoded = encode_intra_frame_y_dyn_with_q(&input, 32).expect("encode");
        let decoded = decode_av1(&encoded.ivf_bytes).expect("decode");
        match &decoded[0] {
            Frame::YDyn { width, height, y } => {
                assert_eq!(*width, 32);
                assert_eq!(*height, 32);
                assert_eq!(y.len(), 32 * 32);
            }
            other => panic!("expected YDyn, got {other:?}"),
        }
    }
}
