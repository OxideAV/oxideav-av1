//! Encoder-side single-reference inter (P-frame) pixel pipeline.
//!
//! The intra dyn driver ([`crate::encoder::pixel_driver_dyn`]) builds a
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
    let mut pred = [0u16; 16];
    for dr in -search..=search {
        for dc in -search..=search {
            let mv = [(dr * 8) as i16, (dc * 8) as i16];
            predict_inter_block_single(
                reference, mv, 0, x, y, 4, 4, bit_depth, 0, 0, EIGHTTAP, &mut pred,
            )?;
            let mut sad = 0u32;
            for k in 0..16 {
                sad += (input[k] as i32 - pred[k] as i32).unsigned_abs();
            }
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
}
