//! Encoder-side single-reference inter (P-frame) pixel pipeline.
//!
//! The retired intra dyn mirror driver (removed in r428) built a
//! leaf's reconstruction as `recon = pred + Q^-1( Q( T( input - pred ) ) )`
//! where `pred` is the §7.11.2 intra prediction. The inter (§5.11.23
//! `is_inter == 1`) arm differs in exactly one place: `pred` comes from
//! the §7.11.3.1 motion-compensated reference samples instead of the
//! intra predictor. Every downstream stage — residual = `input - pred`,
//! §7.13.3 forward transform, §7.12.3 forward quantize, the decoder's
//! `dequantize_step1` → `inverse_transform_2d` inverse, and the
//! `recon = pred + inv_residual` stitch — is shared verbatim with the
//! intra arm.
//!
//! This module supplies the encode-side primitives for that single
//! difference:
//!
//! * [`predict_inter_block_single`] — runs the *decoder's*
//!   [`crate::inter_pred::reconstruct_inter_block`] (the §7.11.3.1
//!   single-reference translational SIMPLE arm) into a scratch buffer,
//!   so the prediction the encoder computes its residual against is, by
//!   construction, bit-identical to the prediction the decoder will
//!   reproduce from the same `(RefFrame[0], Mv)` mode-info. There is no
//!   second prediction implementation to keep in sync.
//! * [`encode_inter_block_residual_4x4`] — the §5.11.39 residual leaf
//!   for one TX_4X4 inter block: residual against the MC prediction,
//!   forward transform + quantize on the chosen arm (lossless WHT or
//!   lossy DCT_DCT), then the matching dequantize + inverse transform,
//!   and the `recon = Clip1(pred + inv_residual)` stitch. Returns the
//!   quantized coefficients (for the coefficient writer) alongside the
//!   reconstructed `4 × 4` block.
//! * [`estimate_motion_4x4_full_search`] — a §A.3-agnostic full-search
//!   integer-pel motion estimator over a square window: it scores each
//!   candidate integer MV by the sum of absolute differences (SAD)
//!   between the input block and the MC prediction at that MV, and
//!   returns the lowest-SAD MV. The estimator only *selects* an MV; the
//!   bitstream-correct prediction is always taken from the decoder
//!   primitive so encode and decode never diverge.
//!
//! The motion vector convention follows §7.11.3.1: `Mv` is
//! `[mv_row, mv_col]` in 1/8-luma-sample units. An integer-pel MV is a
//! multiple of 8.

use crate::cdf::{QuantizerParams, DCT_DCT, TX_4X4};
use crate::encoder::forward_quantize::forward_quantize;
use crate::encoder::forward_transform_2d::forward_transform_2d;
use crate::encoder::forward_wht::forward_wht_4x4;
use crate::inter_pred::{reconstruct_inter_block, InterModeInfo, RefFrameStoreEntry, EIGHTTAP};
use crate::Error;

/// One reference-plane view the encoder predicts against. Mirrors the
/// per-plane resolved [`RefFrameStoreEntry`] the decoder builds from its
/// `FrameStore`: the samples are already at the `(plane, subsampling_*)`
/// resolution the prediction is run for.
#[derive(Debug, Clone, Copy)]
pub struct EncRefPlane<'a> {
    /// Reference plane samples (row-major, `stride` columns), `u16` to
    /// share the decoder's high-bit-depth-capable buffer type. For
    /// 8-bit content every sample is `0..=255`.
    pub plane: &'a [u16],
    /// Column stride of `plane` (`>= width`).
    pub stride: usize,
    /// Plane width in samples (`= upscaled_width` with superres off).
    pub width: u32,
    /// Plane height in samples.
    pub height: u32,
}

/// §7.11.3.1 single-reference translational prediction for one block,
/// taken from the decoder primitive so it is bit-identical to what the
/// decoder reconstructs from the same `(ref_frame, mv)`.
///
/// `out` receives the `w × h` predicted block in row-major order.
/// `(x, y)` are the block's top-left plane coordinates; `mv` is
/// `[mv_row, mv_col]` in 1/8-luma-sample units. The single-reference
/// arm uses `interp_filter` for both axes (the §6.8.9 default scope this
/// encoder targets does not split the filter per axis).
#[allow(clippy::too_many_arguments)]
pub fn predict_inter_block_single(
    reference: &EncRefPlane<'_>,
    mv: [i16; 2],
    plane: u8,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    bit_depth: u8,
    subsampling_x: u8,
    subsampling_y: u8,
    interp_filter: u8,
    out: &mut [u16],
) -> Result<(), Error> {
    if out.len() < w * h {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // A single FrameStore slot named directly by RefFrame == LAST_FRAME
    // through ref_frame_idx[0] == 0.
    let store = [RefFrameStoreEntry {
        plane: reference.plane,
        stride: reference.stride,
        upscaled_width: reference.width,
        width: reference.width,
        height: reference.height,
    }];
    let ref_frame_idx = [0u8];
    let mode_info = InterModeInfo {
        ref_frame: crate::uncompressed_header_tail::LAST_FRAME as u8,
        mv,
    };
    // The §7.11.3.1 reference fetch is anchored at the block's true
    // plane coords `(x, y)` (the integer + sub-pel sample addresses are
    // `(x, y)` offset by the MV), so `reconstruct_inter_block` must be
    // called with the real `(x, y)`. It stitches the predicted block
    // into its destination at those coords; we allocate a destination
    // covering rows `[0, y + h)` × cols `[0, x + w)`, let it write at
    // `(x, y)`, then copy the `w × h` block back into the tight `out`
    // buffer the caller passed.
    if x < 0 || y < 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let dst_w = (x as usize) + w;
    let dst_h = (y as usize) + h;
    let mut dst = vec![0u16; dst_w * dst_h];
    reconstruct_inter_block(
        mode_info,
        &ref_frame_idx,
        &store,
        plane,
        x,
        y,
        w,
        h,
        bit_depth,
        subsampling_x,
        subsampling_y,
        reference.width,
        reference.height,
        interp_filter,
        interp_filter,
        &mut dst,
        dst_w,
    )?;
    for i in 0..h {
        let src = ((y as usize) + i) * dst_w + (x as usize);
        let d = i * w;
        out[d..d + w].copy_from_slice(&dst[src..src + w]);
    }
    Ok(())
}

/// The quantized-coefficient + reconstructed-pixel result of one
/// §5.11.39 inter residual leaf.
#[derive(Debug, Clone)]
pub struct InterResidualLeaf {
    /// §7.12.3 forward-quantized coefficients, scan-order length for
    /// TX_4X4 (16 entries), ready for the §5.11.39 coefficient writer.
    pub quant: Vec<i32>,
    /// The reconstructed `4 × 4` block (row-major), `recon =
    /// Clip1(pred + Q^-1(Q(T(input - pred))))` — bit-exact to what the
    /// decoder reconstructs from `quant` + the same MC prediction.
    pub recon: [u8; 16],
    /// The MC prediction the residual was taken against (row-major),
    /// surfaced for callers that stitch a running reconstructed plane.
    pub pred: [u8; 16],
}

/// §5.11.39 residual leaf for one TX_4X4 inter block.
///
/// `input` / `pred` are the `4 × 4` input and motion-compensated
/// prediction blocks (row-major). `lossless` selects the §5.9.2 arm:
/// the lossless WHT lattice (`base_q_idx == 0`) or the lossy DCT_DCT
/// (`base_q_idx > 0`). `plane` and `qp` drive §7.12.3 quantization.
///
/// Returns the quantized coefficients and the encoder's own
/// reconstruction (it runs the decoder's `dequantize_step1` →
/// `inverse_transform_2d` inverse on its quantized coefficients, so the
/// reconstruction is the exact image the decoder recovers).
pub fn encode_inter_block_residual_4x4(
    input: &[u8; 16],
    pred: &[u8; 16],
    plane: u8,
    lossless: bool,
    qp: &QuantizerParams,
) -> Result<InterResidualLeaf, Error> {
    let mut residual = [0i64; 16];
    for k in 0..16 {
        residual[k] = input[k] as i64 - pred[k] as i64;
    }
    let coeffs = if lossless {
        forward_wht_4x4(&residual).to_vec()
    } else {
        forward_transform_2d(&residual, TX_4X4, DCT_DCT, false)
    };
    let quant = forward_quantize(&coeffs, TX_4X4, plane, 0, DCT_DCT, 15, qp);
    let dequant = crate::cdf::dequantize_step1(&quant, TX_4X4, plane, 0, DCT_DCT, 15, qp);
    let resid_back = crate::transform::inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, lossless);
    let mut recon = [0u8; 16];
    for k in 0..16 {
        let v = pred[k] as i64 + resid_back[k];
        recon[k] = v.clamp(0, 255) as u8;
    }
    Ok(InterResidualLeaf {
        quant,
        recon,
        pred: {
            let mut p = [0u8; 16];
            p.copy_from_slice(pred);
            p
        },
    })
}

/// Integer-pel full-search motion estimate for one `4 × 4` luma block.
///
/// Scores every integer MV whose components lie in
/// `[-search, search]` (luma samples) by the SAD between `input` and
/// the §7.11.3.1 prediction at that MV (taken from the decoder
/// primitive), and returns the `[mv_row, mv_col]` (1/8-pel units, a
/// multiple of 8) with the lowest SAD. Ties resolve to the
/// smallest-magnitude MV (zero first), keeping the estimator
/// deterministic.
///
/// `(x, y)` are the block's plane coordinates. The search is integer-pel
/// only; sub-pel refinement is a later milestone. The returned MV is
/// always one whose prediction stays inside the reference frame's
/// §7.11.3.2 clamped fetch (the decoder primitive clamps the reference
/// access, so every candidate is valid).
#[allow(clippy::too_many_arguments)]
pub fn estimate_motion_4x4_full_search(
    reference: &EncRefPlane<'_>,
    input: &[u8; 16],
    x: i32,
    y: i32,
    bit_depth: u8,
    search: i32,
) -> Result<[i16; 2], Error> {
    let mut best_mv = [0i16; 2];
    let mut best_sad = u32::MAX;
    let mut best_cost = i32::MAX; // magnitude tie-breaker
    for dr in -search..=search {
        for dc in -search..=search {
            let mv = [(dr * 8) as i16, (dc * 8) as i16];
            let sad = block_sad_at_mv(reference, input, x, y, mv, bit_depth)?;
            let cost = dr.abs() + dc.abs();
            if sad < best_sad || (sad == best_sad && cost < best_cost) {
                best_sad = sad;
                best_cost = cost;
                best_mv = mv;
            }
        }
    }
    Ok(best_mv)
}

/// SAD between `input` and the §7.11.3.1 MC prediction at `mv` (1/8-pel)
/// for one `4 × 4` luma block at plane coords `(x, y)`.
fn block_sad_at_mv(
    reference: &EncRefPlane<'_>,
    input: &[u8; 16],
    x: i32,
    y: i32,
    mv: [i16; 2],
    bit_depth: u8,
) -> Result<u32, Error> {
    let mut pred = [0u16; 16];
    predict_inter_block_single(
        reference, mv, 0, x, y, 4, 4, bit_depth, 0, 0, EIGHTTAP, &mut pred,
    )?;
    let mut sad = 0u32;
    for k in 0..16 {
        sad += (input[k] as i32 - pred[k] as i32).unsigned_abs();
    }
    Ok(sad)
}

/// Integer-pel full search followed by §7.11.3.1 sub-pel refinement.
///
/// First runs [`estimate_motion_4x4_full_search`] for the best
/// integer-pel MV, then refines it through the 1/8-luma-sample MV grid
/// the interpolation filter supports: a half-pel (`±4`) pass over the 8
/// neighbours of the integer optimum, then a quarter-pel (`±2`) pass over
/// the 8 neighbours of the half-pel optimum, then an eighth-pel (`±1`)
/// pass. Each candidate's prediction is taken from the decoder primitive,
/// so the returned MV is one the decoder reconstructs identically.
///
/// The sub-pel passes are a local steepest-descent diamond: at every
/// stage the centre is the running best, and a candidate is accepted only
/// on a strict SAD improvement (ties keep the lower-magnitude MV), so the
/// search is deterministic and never worse than the integer optimum.
#[allow(clippy::too_many_arguments)]
pub fn estimate_motion_4x4_subpel(
    reference: &EncRefPlane<'_>,
    input: &[u8; 16],
    x: i32,
    y: i32,
    bit_depth: u8,
    search: i32,
) -> Result<[i16; 2], Error> {
    let mut best = estimate_motion_4x4_full_search(reference, input, x, y, bit_depth, search)?;
    let mut best_sad = block_sad_at_mv(reference, input, x, y, best, bit_depth)?;
    // Refinement step sizes in 1/8-pel units: half, quarter, eighth.
    for &step in &[4i16, 2, 1] {
        loop {
            let mut improved = false;
            let centre = best;
            for &(dr, dc) in &[
                (-step, 0i16),
                (step, 0),
                (0, -step),
                (0, step),
                (-step, -step),
                (-step, step),
                (step, -step),
                (step, step),
            ] {
                let cand = [centre[0] + dr, centre[1] + dc];
                let sad = block_sad_at_mv(reference, input, x, y, cand, bit_depth)?;
                let cand_cost = (cand[0].abs() as i32) + (cand[1].abs() as i32);
                let best_cost = (best[0].abs() as i32) + (best[1].abs() as i32);
                if sad < best_sad || (sad == best_sad && cand_cost < best_cost) {
                    best_sad = sad;
                    best = cand;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
    }
    Ok(best)
}

/// The encoded result of one single-reference luma P-frame: the
/// per-block motion field, the running reconstruction, and the quantized
/// coefficient blocks, in §5.11.5 raster (mi-grid) leaf order.
#[derive(Debug, Clone)]
pub struct EncodedInterFrameY {
    /// Frame width / height in samples.
    pub width: u32,
    pub height: u32,
    /// Per-`4 × 4`-cell motion vectors (row-major over the
    /// `mi_rows × mi_cols` grid), `[mv_row, mv_col]` in 1/8-pel units.
    pub mvs: Vec<[i16; 2]>,
    /// The encoder's own reconstruction of the luma plane (row-major) —
    /// `recon = Clip1(pred + Q^-1(Q(T(input - pred))))` per leaf. This is
    /// the exact plane the decoder reconstructs from `mvs` + the per-leaf
    /// `quant` against the same reference.
    pub recon: Vec<u8>,
    /// Per-cell §7.12.3 quantized coefficients (TX_4X4, 16 entries each),
    /// in the same row-major leaf order as `mvs`.
    pub quant: Vec<Vec<i32>>,
}

/// Encode a luma plane as a single-reference (P-frame) grid of TX_4X4
/// inter leaves against `reference`.
///
/// For each `4 × 4` luma block in §5.11.5 raster order the encoder runs
/// [`estimate_motion_4x4_full_search`] to pick an integer-pel MV, takes
/// the §7.11.3.1 MC prediction at that MV from the decoder primitive,
/// and codes the residual through [`encode_inter_block_residual_4x4`].
/// The reconstruction is accumulated into `recon` so a caller can verify
/// it against the decoder's independent [`crate::inter_pred::reconstruct_inter_frame`]
/// frame walk driven by the returned `mvs`.
///
/// `width` / `height` must be multiples of 4. `lossless` selects the
/// §5.9.2 arm (WHT vs DCT_DCT). The reference plane must be at least
/// `width × height`.
pub fn encode_inter_frame_y(
    input: &[u8],
    width: u32,
    height: u32,
    reference: &EncRefPlane<'_>,
    lossless: bool,
    qp: &QuantizerParams,
    search: i32,
) -> Result<EncodedInterFrameY, Error> {
    encode_inter_frame_y_opt(input, width, height, reference, lossless, qp, search, false)
}

/// As [`encode_inter_frame_y`], with an explicit `subpel` flag selecting
/// the §7.11.3.1 sub-pel-refined estimator ([`estimate_motion_4x4_subpel`])
/// over the integer-pel full search.
#[allow(clippy::too_many_arguments)]
pub fn encode_inter_frame_y_opt(
    input: &[u8],
    width: u32,
    height: u32,
    reference: &EncRefPlane<'_>,
    lossless: bool,
    qp: &QuantizerParams,
    search: i32,
    subpel: bool,
) -> Result<EncodedInterFrameY, Error> {
    if width % 4 != 0 || height % 4 != 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let w = width as usize;
    let h = height as usize;
    if input.len() < w * h {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let mi_cols = (width / 4) as usize;
    let mi_rows = (height / 4) as usize;
    let mut mvs = vec![[0i16; 2]; mi_rows * mi_cols];
    let mut quant = Vec::with_capacity(mi_rows * mi_cols);
    let mut recon = vec![0u8; w * h];

    for mi_r in 0..mi_rows {
        for mi_c in 0..mi_cols {
            let row0 = mi_r * 4;
            let col0 = mi_c * 4;
            // Gather the input 4×4 block.
            let mut blk = [0u8; 16];
            for i in 0..4 {
                for j in 0..4 {
                    blk[i * 4 + j] = input[(row0 + i) * w + (col0 + j)];
                }
            }
            // MV search against the reference (integer-pel, optionally
            // refined to the §7.11.3.1 sub-pel grid).
            let mv = if subpel {
                estimate_motion_4x4_subpel(
                    reference,
                    &blk,
                    col0 as i32,
                    row0 as i32,
                    qp.bit_depth,
                    search,
                )?
            } else {
                estimate_motion_4x4_full_search(
                    reference,
                    &blk,
                    col0 as i32,
                    row0 as i32,
                    qp.bit_depth,
                    search,
                )?
            };
            // MC prediction at the chosen MV (decoder primitive).
            let mut pred16 = [0u16; 16];
            predict_inter_block_single(
                reference,
                mv,
                0,
                col0 as i32,
                row0 as i32,
                4,
                4,
                qp.bit_depth,
                0,
                0,
                EIGHTTAP,
                &mut pred16,
            )?;
            let pred: [u8; 16] = std::array::from_fn(|k| pred16[k] as u8);
            // Residual leaf.
            let leaf = encode_inter_block_residual_4x4(&blk, &pred, 0, lossless, qp)?;
            // Stitch reconstruction.
            for i in 0..4 {
                for j in 0..4 {
                    recon[(row0 + i) * w + (col0 + j)] = leaf.recon[i * 4 + j];
                }
            }
            mvs[mi_r * mi_cols + mi_c] = mv;
            quant.push(leaf.quant);
        }
    }

    Ok(EncodedInterFrameY {
        width,
        height,
        mvs,
        recon,
        quant,
    })
}

/// The encoded result of one single-reference 4:2:0 YUV P-frame.
#[derive(Debug, Clone)]
pub struct EncodedInterFrameYuv {
    /// Luma frame width / height in samples (multiples of 8).
    pub width: u32,
    pub height: u32,
    /// Per-`4 × 4`-luma-cell motion field (row-major over the luma
    /// mi-grid), `[mv_row, mv_col]` in 1/8-luma-pel units. Chroma
    /// prediction reuses the collocated luma MV (the §7.11.3.2 chroma
    /// scaling is applied inside the prediction primitive).
    pub mvs: Vec<[i16; 2]>,
    /// Encoder reconstruction of the three planes (row-major).
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
}

/// Encode a 4:2:0 8-bit YUV plane triple as a single-reference P-frame
/// against `ref_y` / `ref_u` / `ref_v`.
///
/// The luma plane is encoded exactly like [`encode_inter_frame_y_opt`],
/// producing a per-`4 × 4`-luma-cell motion field. Each chroma `4 × 4`
/// block (covering an `8 × 8` luma region under 4:2:0) reuses the
/// **luma** MV of its collocated top-left even luma cell — the same
/// candidate the decoder's §5.11.33 frame walk reads
/// (`cand = (mi >> sub) << sub`) — and predicts through the chroma arm of
/// the decoder primitive (`subsampling_x == subsampling_y == 1`), so the
/// §7.11.3.2 chroma MV scaling is identical on both sides. Residuals are
/// coded per plane, and the reconstruction is accumulated so a caller can
/// round-trip the full motion field through
/// [`crate::inter_pred::reconstruct_inter_frame`] with three
/// `PlaneReconContext`s.
///
/// `width` / `height` must be multiples of 8. The reference planes must
/// cover the luma / chroma extents.
#[allow(clippy::too_many_arguments)]
pub fn encode_inter_frame_yuv(
    input_y: &[u8],
    input_u: &[u8],
    input_v: &[u8],
    width: u32,
    height: u32,
    ref_y: &EncRefPlane<'_>,
    ref_u: &EncRefPlane<'_>,
    ref_v: &EncRefPlane<'_>,
    lossless: bool,
    qp: &QuantizerParams,
    search: i32,
    subpel: bool,
) -> Result<EncodedInterFrameYuv, Error> {
    if width % 8 != 0 || height % 8 != 0 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    // Luma pass — reuse the Y encoder for the motion field + recon.
    let enc_y =
        encode_inter_frame_y_opt(input_y, width, height, ref_y, lossless, qp, search, subpel)?;
    let mi_cols = (width / 4) as usize;

    let cw = (width / 2) as usize;
    let ch = (height / 2) as usize;
    if input_u.len() < cw * ch || input_v.len() < cw * ch {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let c_mi_cols = cw / 4;
    let c_mi_rows = ch / 4;

    let mut recon_u = vec![0u8; cw * ch];
    let mut recon_v = vec![0u8; cw * ch];

    for (plane, (input_c, recon_c, ref_c)) in [
        (1u8, (input_u, &mut recon_u, ref_u)),
        (2u8, (input_v, &mut recon_v, ref_v)),
    ] {
        for c_mi_r in 0..c_mi_rows {
            for c_mi_c in 0..c_mi_cols {
                let crow0 = c_mi_r * 4;
                let ccol0 = c_mi_c * 4;
                // §5.11.33 collocated candidate: chroma cell (cr, cc)
                // reads the luma MV at luma cell (cr << 1, cc << 1).
                let luma_cell = (c_mi_r * 2) * mi_cols + (c_mi_c * 2);
                let mv = enc_y.mvs[luma_cell];
                // Input chroma 4×4.
                let mut blk = [0u8; 16];
                for i in 0..4 {
                    for j in 0..4 {
                        blk[i * 4 + j] = input_c[(crow0 + i) * cw + (ccol0 + j)];
                    }
                }
                // Chroma MC prediction (subsampling 1,1 ⇒ §7.11.3.2
                // chroma MV scaling inside the primitive).
                let mut pred16 = [0u16; 16];
                predict_inter_block_single(
                    ref_c,
                    mv,
                    plane,
                    ccol0 as i32,
                    crow0 as i32,
                    4,
                    4,
                    qp.bit_depth,
                    1,
                    1,
                    EIGHTTAP,
                    &mut pred16,
                )?;
                let pred: [u8; 16] = std::array::from_fn(|k| pred16[k] as u8);
                let leaf = encode_inter_block_residual_4x4(&blk, &pred, plane, lossless, qp)?;
                for i in 0..4 {
                    for j in 0..4 {
                        recon_c[(crow0 + i) * cw + (ccol0 + j)] = leaf.recon[i * 4 + j];
                    }
                }
            }
        }
    }

    Ok(EncodedInterFrameYuv {
        width,
        height,
        mvs: enc_y.mvs,
        recon_y: enc_y.recon,
        recon_u,
        recon_v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn qp_lossless() -> QuantizerParams {
        QuantizerParams::neutral(0, 8)
    }

    fn qp_lossy(q: u8) -> QuantizerParams {
        QuantizerParams::neutral(q, 8)
    }

    // Build a small reference plane (row-major) of given dims.
    fn ref_plane(width: u32, height: u32, f: impl Fn(u32, u32) -> u16) -> Vec<u16> {
        let mut v = vec![0u16; (width * height) as usize];
        for r in 0..height {
            for c in 0..width {
                v[(r * width + c) as usize] = f(r, c);
            }
        }
        v
    }

    #[test]
    fn zero_mv_prediction_copies_reference_block() {
        let w = 16u32;
        let h = 16u32;
        let plane = ref_plane(w, h, |r, c| ((r * 16 + c) % 256) as u16);
        let reference = EncRefPlane {
            plane: &plane,
            stride: w as usize,
            width: w,
            height: h,
        };
        let mut out = [0u16; 16];
        predict_inter_block_single(
            &reference,
            [0, 0],
            0,
            4,
            4,
            4,
            4,
            8,
            0,
            0,
            EIGHTTAP,
            &mut out,
        )
        .unwrap();
        // Zero-MV integer prediction equals the reference samples at the
        // block's own coordinates (the §7.11.3.1 fetch is identity).
        for i in 0..4 {
            for j in 0..4 {
                let want = plane[((4 + i) * w as usize) + (4 + j)];
                assert_eq!(out[i * 4 + j], want, "({i},{j})");
            }
        }
    }

    #[test]
    fn integer_mv_prediction_shifts_reference_block() {
        let w = 24u32;
        let h = 24u32;
        let plane = ref_plane(w, h, |r, c| ((r * 7 + c * 3) % 256) as u16);
        let reference = EncRefPlane {
            plane: &plane,
            stride: w as usize,
            width: w,
            height: h,
        };
        // MV (8, 16) in 1/8-pel == (+1 row, +2 col) integer shift.
        let mut out = [0u16; 16];
        predict_inter_block_single(
            &reference,
            [8, 16],
            0,
            8,
            8,
            4,
            4,
            8,
            0,
            0,
            EIGHTTAP,
            &mut out,
        )
        .unwrap();
        for i in 0..4 {
            for j in 0..4 {
                let want = plane[((8 + 1 + i) * w as usize) + (8 + 2 + j)];
                assert_eq!(out[i * 4 + j], want, "shifted ({i},{j})");
            }
        }
    }

    #[test]
    fn lossless_inter_residual_reconstructs_input_bit_exact() {
        // With the lossless WHT arm, recon == input regardless of the
        // prediction (the residual is coded losslessly).
        let pred: [u8; 16] = std::array::from_fn(|k| (k * 11 % 256) as u8);
        let input: [u8; 16] = std::array::from_fn(|k| ((k * 37 + 5) % 256) as u8);
        let leaf = encode_inter_block_residual_4x4(&input, &pred, 0, true, &qp_lossless()).unwrap();
        assert_eq!(leaf.recon, input, "lossless inter residual is bit-exact");
    }

    #[test]
    fn lossy_inter_residual_zero_when_pred_equals_input() {
        // When prediction == input, the residual is zero, every quantized
        // coefficient is zero, and recon == input on every arm.
        let input: [u8; 16] = std::array::from_fn(|k| ((k * 13) % 200 + 10) as u8);
        let leaf =
            encode_inter_block_residual_4x4(&input, &input, 0, false, &qp_lossy(64)).unwrap();
        assert!(
            leaf.quant.iter().all(|&c| c == 0),
            "zero residual ⇒ zero coeffs"
        );
        assert_eq!(leaf.recon, input, "zero residual ⇒ recon == input");
    }

    #[test]
    fn full_search_recovers_known_translation() {
        // Build a reference with a recognisable gradient, place a 4×4
        // block in the input that is exactly the reference shifted by
        // (+1 row, +2 col), and verify the estimator finds MV (8, 16).
        let w = 32u32;
        let h = 32u32;
        let plane = ref_plane(w, h, |r, c| ((r * 9 + c * 5) % 256) as u16);
        let reference = EncRefPlane {
            plane: &plane,
            stride: w as usize,
            width: w,
            height: h,
        };
        let (bx, by) = (12usize, 12usize);
        // Input block = reference at (by+1, bx+2).
        let input: [u8; 16] = std::array::from_fn(|k| {
            let i = k / 4;
            let j = k % 4;
            plane[((by + 1 + i) * w as usize) + (bx + 2 + j)] as u8
        });
        let mv = estimate_motion_4x4_full_search(&reference, &input, bx as i32, by as i32, 8, 3)
            .unwrap();
        assert_eq!(mv, [8, 16], "full search recovers the (+1,+2) translation");
    }

    #[test]
    fn full_search_zero_mv_for_static_match() {
        // Input equals the reference block at its own coords ⇒ best MV 0.
        let w = 20u32;
        let h = 20u32;
        let plane = ref_plane(w, h, |r, c| ((r + c) % 64) as u16 + 30);
        let reference = EncRefPlane {
            plane: &plane,
            stride: w as usize,
            width: w,
            height: h,
        };
        let (bx, by) = (8usize, 8usize);
        let input: [u8; 16] = std::array::from_fn(|k| {
            let i = k / 4;
            let j = k % 4;
            plane[((by + i) * w as usize) + (bx + j)] as u8
        });
        let mv = estimate_motion_4x4_full_search(&reference, &input, bx as i32, by as i32, 8, 2)
            .unwrap();
        assert_eq!(mv, [0, 0], "static block ⇒ zero MV");
    }

    #[test]
    fn predict_rejects_short_out_buffer() {
        let plane = vec![0u16; 16];
        let reference = EncRefPlane {
            plane: &plane,
            stride: 4,
            width: 4,
            height: 4,
        };
        let mut out = [0u16; 4];
        let r = predict_inter_block_single(
            &reference,
            [0, 0],
            0,
            0,
            0,
            4,
            4,
            8,
            0,
            0,
            EIGHTTAP,
            &mut out,
        );
        assert!(r.is_err(), "short out buffer rejected");
    }

    use crate::inter_pred::{
        reconstruct_inter_frame, InterModeInfoGrid, PlaneReconContext, RefFrameStoreEntry,
    };

    // Build the decoder's frame-walk prediction from an encoder motion
    // field + the same reference, returning CurrFrame[0] (prediction
    // only — no residual add-back, which the coefficient path supplies).
    fn decoder_frame_prediction(enc: &EncodedInterFrameY, reference: &EncRefPlane<'_>) -> Vec<u16> {
        let mi_cols = (enc.width / 4) as usize;
        let mi_rows = (enc.height / 4) as usize;
        let cells = mi_rows * mi_cols;
        let last = crate::uncompressed_header_tail::LAST_FRAME;
        let mi_sizes = vec![crate::cdf::BLOCK_4X4; cells];
        let is_inters = vec![1u8; cells];
        let mut ref_frames = vec![0i8; cells * 2];
        let mut mvs = vec![0i16; cells * 4];
        let interp_filters = vec![EIGHTTAP; cells * 2];
        let zeros = vec![0u8; cells];
        for cell in 0..cells {
            ref_frames[cell * 2] = last as i8;
            ref_frames[cell * 2 + 1] = -1; // NONE ⇒ single forward ref.
            mvs[cell * 4] = enc.mvs[cell][0]; // list 0, row.
            mvs[cell * 4 + 1] = enc.mvs[cell][1]; // list 0, col.
        }
        let order_hints_by_ref = [0i32; 8];
        let grid = InterModeInfoGrid {
            mi_sizes: &mi_sizes,
            is_inters: &is_inters,
            ref_frames: &ref_frames,
            mvs: &mvs,
            interp_filters: &interp_filters,
            compound_types: &zeros,
            wedge_indices: &zeros,
            wedge_signs: &zeros,
            mask_types: &zeros,
            interintra_modes: &zeros,
            wedge_interintras: &zeros,
            interintra_wedge_indices: &zeros,
            order_hint_bits: 7,
            current_order_hint: 0,
            order_hints_by_ref: &order_hints_by_ref,
            mi_rows: mi_rows as u32,
            mi_cols: mi_cols as u32,
            bit_depth: 8,
            warp: None,
            obmc: None,
        };
        let store = [RefFrameStoreEntry {
            plane: reference.plane,
            stride: reference.stride,
            upscaled_width: reference.width,
            width: reference.width,
            height: reference.height,
        }];
        let ref_frame_idx = [0u8];
        let mut curr = vec![0u16; (enc.width * enc.height) as usize];
        let mut planes = [PlaneReconContext {
            plane: 0,
            subsampling_x: 0,
            subsampling_y: 0,
            frame_store: &store,
            frame_width: reference.width,
            frame_height: reference.height,
            curr: &mut curr,
            curr_stride: enc.width as usize,
        }];
        reconstruct_inter_frame(&grid, &ref_frame_idx, &mut planes).unwrap();
        curr
    }

    #[test]
    fn encoder_motion_field_round_trips_through_decoder_frame_walk() {
        // The encoder picks a per-4×4 motion field; the decoder's
        // independent frame walk on that field must reproduce the exact
        // MC prediction the encoder coded its residual against.
        let w = 32u32;
        let h = 24u32;
        // Reference with structure (so the SAD search has a real minimum).
        let plane = ref_plane(w + 16, h + 16, |r, c| {
            (((r * 13) ^ (c * 7)).wrapping_add(r * c) % 256) as u16
        });
        let rw = w + 16;
        let reference = EncRefPlane {
            plane: &plane,
            stride: rw as usize,
            width: rw,
            height: h + 16,
        };
        // Input: a globally shifted view of the reference top-left region.
        let mut input = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            for c in 0..w as usize {
                // shift by (+2, +3) from reference origin.
                input[r * w as usize + c] = plane[(r + 2) * rw as usize + (c + 3)] as u8;
            }
        }
        let qp = qp_lossy(48);
        let enc = encode_inter_frame_y(&input, w, h, &reference, false, &qp, 4).unwrap();

        // Recompute the per-leaf MC prediction the encoder used and stitch
        // it into a plane, then compare with the decoder frame walk.
        let mut enc_pred = vec![0u16; (w * h) as usize];
        let mi_cols = (w / 4) as usize;
        for (cell, mv) in enc.mvs.iter().enumerate() {
            let mi_r = cell / mi_cols;
            let mi_c = cell % mi_cols;
            let mut p = [0u16; 16];
            predict_inter_block_single(
                &reference,
                *mv,
                0,
                (mi_c * 4) as i32,
                (mi_r * 4) as i32,
                4,
                4,
                8,
                0,
                0,
                EIGHTTAP,
                &mut p,
            )
            .unwrap();
            for i in 0..4 {
                for j in 0..4 {
                    enc_pred[(mi_r * 4 + i) * w as usize + (mi_c * 4 + j)] = p[i * 4 + j];
                }
            }
        }
        let dec_pred = decoder_frame_prediction(&enc, &reference);
        assert_eq!(
            enc_pred, dec_pred,
            "encoder MC prediction == decoder frame-walk prediction"
        );
    }

    #[test]
    fn lossless_inter_frame_reconstruction_is_bit_exact() {
        // On the lossless WHT arm the residual is coded losslessly, so the
        // encoder's reconstruction equals the input regardless of the
        // motion field the estimator selects.
        let w = 16u32;
        let h = 16u32;
        let plane = ref_plane(w + 8, h + 8, |r, c| ((r * 5 + c * 11) % 256) as u16);
        let reference = EncRefPlane {
            plane: &plane,
            stride: (w + 8) as usize,
            width: w + 8,
            height: h + 8,
        };
        let input: Vec<u8> = (0..(w * h)).map(|k| ((k * 17 + 3) % 256) as u8).collect();
        let enc = encode_inter_frame_y(&input, w, h, &reference, true, &qp_lossless(), 2).unwrap();
        assert_eq!(enc.recon, input, "lossless inter frame recon is bit-exact");
    }

    #[test]
    fn lossy_inter_frame_static_match_is_zero_residual() {
        // Input identical to the reference's top-left region ⇒ zero-MV
        // best match ⇒ zero residual ⇒ recon == input on the lossy arm.
        let w = 16u32;
        let h = 16u32;
        let plane = ref_plane(w + 8, h + 8, |r, c| ((r * 3 + c) % 100 + 20) as u16);
        let reference = EncRefPlane {
            plane: &plane,
            stride: (w + 8) as usize,
            width: w + 8,
            height: h + 8,
        };
        let mut input = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            for c in 0..w as usize {
                input[r * w as usize + c] = plane[r * (w + 8) as usize + c] as u8;
            }
        }
        let enc = encode_inter_frame_y(&input, w, h, &reference, false, &qp_lossy(80), 2).unwrap();
        assert!(
            enc.mvs.iter().all(|mv| *mv == [0, 0]),
            "static match ⇒ all zero MVs"
        );
        assert_eq!(enc.recon, input, "zero residual ⇒ recon == input");
    }

    #[test]
    fn subpel_estimate_never_worse_than_integer() {
        // The sub-pel refinement starts from the integer optimum and only
        // accepts strict improvements, so its SAD is <= the integer SAD.
        let w = 40u32;
        let h = 40u32;
        let plane = ref_plane(w, h, |r, c| {
            (((r * 3 + c * 2) as f32 * 1.7).sin().abs() * 200.0) as u16 + 10
        });
        let reference = EncRefPlane {
            plane: &plane,
            stride: w as usize,
            width: w,
            height: h,
        };
        let (bx, by) = (16usize, 16usize);
        // Input = reference at a half-pel-ish offset (use a blurred mix).
        let input: [u8; 16] = std::array::from_fn(|k| {
            let i = k / 4;
            let j = k % 4;
            let a = plane[(by + i) * w as usize + (bx + j)] as u32;
            let b = plane[(by + i) * w as usize + (bx + j + 1)] as u32;
            ((a + b) / 2) as u8
        });
        let int_mv =
            estimate_motion_4x4_full_search(&reference, &input, bx as i32, by as i32, 8, 3)
                .unwrap();
        let sub_mv =
            estimate_motion_4x4_subpel(&reference, &input, bx as i32, by as i32, 8, 3).unwrap();
        let int_sad = block_sad_at_mv(&reference, &input, bx as i32, by as i32, int_mv, 8).unwrap();
        let sub_sad = block_sad_at_mv(&reference, &input, bx as i32, by as i32, sub_mv, 8).unwrap();
        assert!(
            sub_sad <= int_sad,
            "sub-pel SAD {sub_sad} <= integer {int_sad}"
        );
    }

    #[test]
    fn subpel_frame_round_trips_through_decoder_frame_walk() {
        // Sub-pel motion field must reconstruct identically through the
        // decoder's independent frame walk (the sub-pel MVs drive the
        // interpolation filter the same way on both sides).
        let w = 24u32;
        let h = 20u32;
        let rw = w + 12;
        let plane = ref_plane(rw, h + 12, |r, c| {
            (((r * 11) ^ (c * 5)).wrapping_add(r + c) % 256) as u16
        });
        let reference = EncRefPlane {
            plane: &plane,
            stride: rw as usize,
            width: rw,
            height: h + 12,
        };
        let mut input = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            for c in 0..w as usize {
                let a = plane[(r + 1) * rw as usize + (c + 1)] as u32;
                let b = plane[(r + 1) * rw as usize + (c + 2)] as u32;
                input[r * w as usize + c] = ((a + b) / 2) as u8;
            }
        }
        let qp = qp_lossy(40);
        let enc = encode_inter_frame_y_opt(&input, w, h, &reference, false, &qp, 3, true).unwrap();
        // Encoder MC prediction per leaf.
        let mut enc_pred = vec![0u16; (w * h) as usize];
        let mi_cols = (w / 4) as usize;
        for (cell, mv) in enc.mvs.iter().enumerate() {
            let mi_r = cell / mi_cols;
            let mi_c = cell % mi_cols;
            let mut p = [0u16; 16];
            predict_inter_block_single(
                &reference,
                *mv,
                0,
                (mi_c * 4) as i32,
                (mi_r * 4) as i32,
                4,
                4,
                8,
                0,
                0,
                EIGHTTAP,
                &mut p,
            )
            .unwrap();
            for i in 0..4 {
                for j in 0..4 {
                    enc_pred[(mi_r * 4 + i) * w as usize + (mi_c * 4 + j)] = p[i * 4 + j];
                }
            }
        }
        let dec_pred = decoder_frame_prediction(&enc, &reference);
        assert_eq!(enc_pred, dec_pred, "sub-pel field round-trips");
        // And at least one MV is genuinely sub-pel (not a multiple of 8),
        // proving the sub-pel grid was exercised.
        assert!(
            enc.mvs.iter().any(|mv| mv[0] % 8 != 0 || mv[1] % 8 != 0),
            "sub-pel refinement produced a fractional MV"
        );
    }

    // Build the decoder's 3-plane frame-walk prediction from a luma
    // motion field, returning (CurrFrame[0], CurrFrame[1], CurrFrame[2]).
    #[allow(clippy::type_complexity)]
    fn decoder_frame_prediction_yuv(
        width: u32,
        height: u32,
        mvs: &[[i16; 2]],
        ref_y: &EncRefPlane<'_>,
        ref_u: &EncRefPlane<'_>,
        ref_v: &EncRefPlane<'_>,
    ) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
        let mi_cols = (width / 4) as usize;
        let mi_rows = (height / 4) as usize;
        let cells = mi_rows * mi_cols;
        let last = crate::uncompressed_header_tail::LAST_FRAME;
        let mi_sizes = vec![crate::cdf::BLOCK_4X4; cells];
        let is_inters = vec![1u8; cells];
        let mut ref_frames = vec![0i8; cells * 2];
        let mut grid_mvs = vec![0i16; cells * 4];
        let interp_filters = vec![EIGHTTAP; cells * 2];
        let zeros = vec![0u8; cells];
        for cell in 0..cells {
            ref_frames[cell * 2] = last as i8;
            ref_frames[cell * 2 + 1] = -1;
            grid_mvs[cell * 4] = mvs[cell][0];
            grid_mvs[cell * 4 + 1] = mvs[cell][1];
        }
        let order_hints_by_ref = [0i32; 8];
        let grid = InterModeInfoGrid {
            mi_sizes: &mi_sizes,
            is_inters: &is_inters,
            ref_frames: &ref_frames,
            mvs: &grid_mvs,
            interp_filters: &interp_filters,
            compound_types: &zeros,
            wedge_indices: &zeros,
            wedge_signs: &zeros,
            mask_types: &zeros,
            interintra_modes: &zeros,
            wedge_interintras: &zeros,
            interintra_wedge_indices: &zeros,
            order_hint_bits: 7,
            current_order_hint: 0,
            order_hints_by_ref: &order_hints_by_ref,
            mi_rows: mi_rows as u32,
            mi_cols: mi_cols as u32,
            bit_depth: 8,
            warp: None,
            obmc: None,
        };
        let ref_frame_idx = [0u8];
        let store_y = [RefFrameStoreEntry {
            plane: ref_y.plane,
            stride: ref_y.stride,
            upscaled_width: ref_y.width,
            width: ref_y.width,
            height: ref_y.height,
        }];
        let store_u = [RefFrameStoreEntry {
            plane: ref_u.plane,
            stride: ref_u.stride,
            upscaled_width: ref_u.width,
            width: ref_u.width,
            height: ref_u.height,
        }];
        let store_v = [RefFrameStoreEntry {
            plane: ref_v.plane,
            stride: ref_v.stride,
            upscaled_width: ref_v.width,
            width: ref_v.width,
            height: ref_v.height,
        }];
        let cw = (width / 2) as usize;
        let ch = (height / 2) as usize;
        let mut curr_y = vec![0u16; (width * height) as usize];
        let mut curr_u = vec![0u16; cw * ch];
        let mut curr_v = vec![0u16; cw * ch];
        {
            let mut planes = [
                PlaneReconContext {
                    plane: 0,
                    subsampling_x: 0,
                    subsampling_y: 0,
                    frame_store: &store_y,
                    frame_width: ref_y.width,
                    frame_height: ref_y.height,
                    curr: &mut curr_y,
                    curr_stride: width as usize,
                },
                PlaneReconContext {
                    plane: 1,
                    subsampling_x: 1,
                    subsampling_y: 1,
                    frame_store: &store_u,
                    frame_width: ref_u.width,
                    frame_height: ref_u.height,
                    curr: &mut curr_u,
                    curr_stride: cw,
                },
                PlaneReconContext {
                    plane: 2,
                    subsampling_x: 1,
                    subsampling_y: 1,
                    frame_store: &store_v,
                    frame_width: ref_v.width,
                    frame_height: ref_v.height,
                    curr: &mut curr_v,
                    curr_stride: cw,
                },
            ];
            reconstruct_inter_frame(&grid, &ref_frame_idx, &mut planes).unwrap();
        }
        (curr_y, curr_u, curr_v)
    }

    #[test]
    fn yuv_inter_frame_round_trips_chroma_through_decoder_frame_walk() {
        // The encoder's chroma prediction (collocated luma MV + §7.11.3.2
        // chroma scaling) must equal the decoder's 3-plane frame walk on
        // the same luma motion field.
        let w = 32u32;
        let h = 24u32;
        let rw = w + 16;
        let rh = h + 16;
        let cw = w / 2;
        let chh = h / 2;
        let crw = rw / 2;
        let crh = rh / 2;
        let plane_y = ref_plane(rw, rh, |r, c| (((r * 7) ^ (c * 3)) % 256) as u16);
        let plane_u = ref_plane(crw, crh, |r, c| ((r * 5 + c * 9) % 256) as u16);
        let plane_v = ref_plane(crw, crh, |r, c| ((r * 11 + c) % 256) as u16);
        let ref_y = EncRefPlane {
            plane: &plane_y,
            stride: rw as usize,
            width: rw,
            height: rh,
        };
        let ref_u = EncRefPlane {
            plane: &plane_u,
            stride: crw as usize,
            width: crw,
            height: crh,
        };
        let ref_v = EncRefPlane {
            plane: &plane_v,
            stride: crw as usize,
            width: crw,
            height: crh,
        };
        // Inputs = shifted views (luma +2,+3; chroma a smooth view).
        let mut in_y = vec![0u8; (w * h) as usize];
        for r in 0..h as usize {
            for c in 0..w as usize {
                in_y[r * w as usize + c] = plane_y[(r + 2) * rw as usize + (c + 3)] as u8;
            }
        }
        let mut in_u = vec![0u8; (cw * chh) as usize];
        let mut in_v = vec![0u8; (cw * chh) as usize];
        for r in 0..chh as usize {
            for c in 0..cw as usize {
                in_u[r * cw as usize + c] = plane_u[(r + 1) * crw as usize + (c + 1)] as u8;
                in_v[r * cw as usize + c] = plane_v[(r + 1) * crw as usize + (c + 1)] as u8;
            }
        }
        let qp = qp_lossy(48);
        let enc = encode_inter_frame_yuv(
            &in_y, &in_u, &in_v, w, h, &ref_y, &ref_u, &ref_v, false, &qp, 3, false,
        )
        .unwrap();

        // Encoder chroma prediction per chroma 4×4.
        let mi_cols = (w / 4) as usize;
        let c_mi_cols = (cw / 4) as usize;
        let c_mi_rows = (chh / 4) as usize;
        let mut enc_pred_u = vec![0u16; (cw * chh) as usize];
        let mut enc_pred_v = vec![0u16; (cw * chh) as usize];
        for (plane, (ref_c, pred_c)) in [
            (1u8, (&ref_u, &mut enc_pred_u)),
            (2u8, (&ref_v, &mut enc_pred_v)),
        ] {
            for c_mi_r in 0..c_mi_rows {
                for c_mi_c in 0..c_mi_cols {
                    let mv = enc.mvs[(c_mi_r * 2) * mi_cols + (c_mi_c * 2)];
                    let mut p = [0u16; 16];
                    predict_inter_block_single(
                        ref_c,
                        mv,
                        plane,
                        (c_mi_c * 4) as i32,
                        (c_mi_r * 4) as i32,
                        4,
                        4,
                        8,
                        1,
                        1,
                        EIGHTTAP,
                        &mut p,
                    )
                    .unwrap();
                    for i in 0..4 {
                        for j in 0..4 {
                            pred_c[(c_mi_r * 4 + i) * cw as usize + (c_mi_c * 4 + j)] =
                                p[i * 4 + j];
                        }
                    }
                }
            }
        }
        let (_dy, du, dv) = decoder_frame_prediction_yuv(w, h, &enc.mvs, &ref_y, &ref_u, &ref_v);
        assert_eq!(enc_pred_u, du, "U chroma prediction round-trips");
        assert_eq!(enc_pred_v, dv, "V chroma prediction round-trips");
    }

    #[test]
    fn lossless_yuv_inter_frame_reconstruction_is_bit_exact() {
        let w = 16u32;
        let h = 16u32;
        let cw = w / 2;
        let chh = h / 2;
        let plane_y = ref_plane(w + 8, h + 8, |r, c| ((r * 5 + c * 3) % 256) as u16);
        let plane_u = ref_plane(cw + 4, chh + 4, |r, c| ((r + c * 2) % 256) as u16);
        let plane_v = ref_plane(cw + 4, chh + 4, |r, c| ((r * 2 + c) % 256) as u16);
        let ref_y = EncRefPlane {
            plane: &plane_y,
            stride: (w + 8) as usize,
            width: w + 8,
            height: h + 8,
        };
        let ref_u = EncRefPlane {
            plane: &plane_u,
            stride: (cw + 4) as usize,
            width: cw + 4,
            height: chh + 4,
        };
        let ref_v = EncRefPlane {
            plane: &plane_v,
            stride: (cw + 4) as usize,
            width: cw + 4,
            height: chh + 4,
        };
        let in_y: Vec<u8> = (0..(w * h)).map(|k| ((k * 13 + 7) % 256) as u8).collect();
        let in_u: Vec<u8> = (0..(cw * chh))
            .map(|k| ((k * 19 + 1) % 256) as u8)
            .collect();
        let in_v: Vec<u8> = (0..(cw * chh))
            .map(|k| ((k * 23 + 9) % 256) as u8)
            .collect();
        let enc = encode_inter_frame_yuv(
            &in_y,
            &in_u,
            &in_v,
            w,
            h,
            &ref_y,
            &ref_u,
            &ref_v,
            true,
            &qp_lossless(),
            2,
            false,
        )
        .unwrap();
        assert_eq!(enc.recon_y, in_y, "lossless Y bit-exact");
        assert_eq!(enc.recon_u, in_u, "lossless U bit-exact");
        assert_eq!(enc.recon_v, in_v, "lossless V bit-exact");
    }

    #[test]
    fn encode_inter_frame_rejects_misaligned_dims() {
        let plane = vec![0u16; 64];
        let reference = EncRefPlane {
            plane: &plane,
            stride: 8,
            width: 8,
            height: 8,
        };
        let input = vec![0u8; 30];
        let r = encode_inter_frame_y(&input, 6, 5, &reference, true, &qp_lossless(), 1);
        assert!(r.is_err(), "non-multiple-of-4 dims rejected");
    }
}
