//! AV1 intra prediction primitives — §7.11.2.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/predict` (MIT,
//! KarpelesLab/goavif). This module implements every AV1 intra
//! prediction mode — the 3 "plain" modes (DC, V, H), the 6
//! directional modes (D45, D67, D113, D135, D157, D203), the 3 smooth
//! variants, Paeth, plus the chroma-from-luma (CFL) and filter-intra
//! helpers. Each predictor has both an 8-bit (`u8`) and a high-bit-depth
//! (`u16`) path for the 10/12-bit decoder; the HBD variants pass the
//! bit depth through so out-of-range samples can be clipped.
//!
//! The public `IntraMode` re-exports the decoder's mode taxonomy
//! ([`crate::decode::modes::IntraMode`]) so callers don't need a second
//! enum.
//!
//! All predictors operate on row-major tight buffers (`stride == w`).
//! Callers copy the result into the plane with whatever stride applies.
//! The (`predict`, `predict16`) wrappers allow passing a stride and
//! matches the old module shape so existing tests still compile.

pub mod cfl;
pub mod dc;
pub mod directional;
pub mod filter;
pub mod paeth;
pub mod smooth;
pub mod vh;

pub use cfl::{cfl_pred, cfl_pred16, cfl_subsample, cfl_subsample16};
pub use dc::{dc_pred, dc_pred16};
pub use directional::{directional_pred, directional_pred16, mode_to_angle_map};
pub use filter::{filter_intra_pred, filter_intra_pred16, FILTER_INTRA_TAPS};
pub use paeth::{paeth_pred, paeth_pred16};
pub use smooth::{smooth_h_pred, smooth_h_pred16, smooth_pred, smooth_pred16, smooth_v_pred, smooth_v_pred16};
pub use vh::{h_pred, h_pred16, v_pred, v_pred16};

use oxideav_core::{Error, Result};

/// AV1 intra prediction modes — shared across the whole crate. Re-export
/// of [`crate::decode::modes::IntraMode`] so callers can say
/// `predict::intra::IntraMode` without pulling in the decoder module.
pub use crate::decode::modes::IntraMode;

/// Neighbour pixel availability for the block being predicted. Used by
/// the compat API `predict` wrapper; internal predictors take the
/// neighbour slices directly.
#[derive(Clone, Copy, Debug)]
pub struct Neighbours<'a> {
    /// `above[0..w]` — row of samples immediately above the block.
    pub above: Option<&'a [u8]>,
    /// `left[0..h]` — column of samples immediately left of the block,
    /// ordered top-to-bottom.
    pub left: Option<&'a [u8]>,
}

/// Run `mode` over a `w × h` block — compat wrapper matching the
/// pre-Phase-5 shape. Returns `Ok(())` on success. The predictor writes
/// row-major into `dst` with stride `dst_stride`.
///
/// For directional / smooth / Paeth modes, missing neighbours are
/// substituted with 128 mid-grey (matching goavif's edge-replication
/// behavior for the intra-only still-image path).
pub fn predict(
    mode: IntraMode,
    n: Neighbours<'_>,
    w: usize,
    h: usize,
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(Error::invalid("av1 intra: zero-sized block"));
    }
    if dst.len() < (h - 1) * dst_stride + w {
        return Err(Error::invalid("av1 intra: dst buffer too small"));
    }
    let mut tight = vec![0u8; w * h];
    dispatch_u8(mode, n, w, h, &mut tight)?;
    for r in 0..h {
        let src_off = r * w;
        let dst_off = r * dst_stride;
        dst[dst_off..dst_off + w].copy_from_slice(&tight[src_off..src_off + w]);
    }
    Ok(())
}

/// Dispatch for the `predict` wrapper. Fills a tight `w*h` buffer.
fn dispatch_u8(
    mode: IntraMode,
    n: Neighbours<'_>,
    w: usize,
    h: usize,
    dst: &mut [u8],
) -> Result<()> {
    let above_stored: Vec<u8>;
    let left_stored: Vec<u8>;
    let above: &[u8] = match n.above {
        Some(a) if a.len() >= w => &a[..w],
        Some(a) => {
            above_stored = pad_slice(a, w, 128);
            &above_stored
        }
        None => {
            above_stored = vec![128; w];
            &above_stored
        }
    };
    let left: &[u8] = match n.left {
        Some(l) if l.len() >= h => &l[..h],
        Some(l) => {
            left_stored = pad_slice(l, h, 128);
            &left_stored
        }
        None => {
            left_stored = vec![128; h];
            &left_stored
        }
    };
    let have_above = n.above.is_some();
    let have_left = n.left.is_some();
    match mode {
        IntraMode::DcPred => {
            dc_pred(dst, w, h, above, left, have_above, have_left, 8);
            Ok(())
        }
        IntraMode::VPred => {
            if !have_above {
                return Err(Error::invalid(
                    "av1 V_PRED: above-row unavailable (§7.11.2.4)",
                ));
            }
            v_pred(dst, w, h, above);
            Ok(())
        }
        IntraMode::HPred => {
            if !have_left {
                return Err(Error::invalid(
                    "av1 H_PRED: left-column unavailable (§7.11.2.3)",
                ));
            }
            h_pred(dst, w, h, left);
            Ok(())
        }
        IntraMode::D45Pred
        | IntraMode::D67Pred
        | IntraMode::D113Pred
        | IntraMode::D135Pred
        | IntraMode::D157Pred
        | IntraMode::D203Pred => {
            let angle = mode_to_angle_map(mode);
            // Extend neighbours by replicating the last sample so the
            // projection reads safely to (w+h) samples.
            let ext_a = replicate_to_len(above, w + h + 1);
            let ext_l = replicate_to_len(left, w + h + 1);
            directional_pred(dst, w, h, &ext_a, &ext_l, angle);
            Ok(())
        }
        IntraMode::SmoothPred => {
            smooth_pred(dst, w, h, above, left);
            Ok(())
        }
        IntraMode::SmoothVPred => {
            smooth_v_pred(dst, w, h, above, left);
            Ok(())
        }
        IntraMode::SmoothHPred => {
            smooth_h_pred(dst, w, h, above, left);
            Ok(())
        }
        IntraMode::PaethPred => {
            let al = if have_above && have_left {
                // above-left isn't in the Neighbours struct; approximate
                // with a midway value so the compat API stays usable.
                128
            } else {
                128
            };
            paeth_pred(dst, w, h, above, left, al);
            Ok(())
        }
        IntraMode::CflPred => Err(Error::invalid(
            "av1 CFL_PRED: use cfl_pred directly (§7.11.5)",
        )),
    }
}

fn pad_slice(src: &[u8], target_len: usize, pad: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(target_len);
    out.extend_from_slice(src);
    while out.len() < target_len {
        let v = *out.last().unwrap_or(&pad);
        out.push(v);
    }
    out
}

fn replicate_to_len(src: &[u8], target_len: usize) -> Vec<u8> {
    if src.is_empty() {
        return vec![128; target_len];
    }
    let mut out = Vec::with_capacity(target_len);
    out.extend_from_slice(src);
    let last = *out.last().unwrap_or(&128);
    while out.len() < target_len {
        out.push(last);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_pred_average_of_neighbours() {
        let above = [100u8; 4];
        let left = [120u8; 4];
        let n = Neighbours {
            above: Some(&above),
            left: Some(&left),
        };
        let mut dst = [0u8; 16];
        predict(IntraMode::DcPred, n, 4, 4, &mut dst, 4).unwrap();
        for &v in &dst {
            assert_eq!(v, 110);
        }
    }

    #[test]
    fn v_pred_copies_above_row() {
        let above = [10u8, 20, 30, 40];
        let n = Neighbours {
            above: Some(&above),
            left: None,
        };
        let mut dst = [0u8; 16];
        predict(IntraMode::VPred, n, 4, 4, &mut dst, 4).unwrap();
        for row in 0..4 {
            assert_eq!(&dst[row * 4..row * 4 + 4], &above[..]);
        }
    }

    #[test]
    fn h_pred_copies_left_column() {
        let left = [10u8, 20, 30, 40];
        let n = Neighbours {
            above: None,
            left: Some(&left),
        };
        let mut dst = [0u8; 16];
        predict(IntraMode::HPred, n, 4, 4, &mut dst, 4).unwrap();
        for row in 0..4 {
            assert_eq!(dst[row * 4], left[row]);
            assert_eq!(dst[row * 4 + 3], left[row]);
        }
    }
}
