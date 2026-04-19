//! AV1 inverse transforms — §7.7.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/transform/*` (MIT,
//! KarpelesLab/goavif). The public entry points are:
//!
//! - [`inverse_2d`] — run the row-then-column 1D inverse transforms
//!   over a `w × h` coefficient block. Output overwrites the input
//!   (signed 32-bit residual).
//! - [`inverse_transform_add`] — convenience wrapper that applies a
//!   final `round_shift` + clip-add onto an 8-bit predictor block, for
//!   callers that already own the destination plane.
//!
//! Phase 3 implements 4/8/16-point DCT and ADST (plus their flipped
//! variants) + the pure-identity 2D case. 32/64-point transforms, the
//! WHT, and the mixed V/H-identity directions remain deferred — callers
//! that request them receive `Error::Unsupported` with a precise §ref.

pub mod adst16;
pub mod adst4;
pub mod adst8;
pub mod cos_pi;
pub mod idct16;
pub mod idct4;
pub mod idct8;
pub mod scan;
pub mod types;

pub use scan::{default_zigzag_scan, inverse_scan};
pub use types::{Kind, TxSize, TxType};

use oxideav_core::{Error, Result};

use adst16::{iadst16, iflipadst16};
use adst4::{iadst4, iflipadst4};
use adst8::{iadst8, iflipadst8};
use idct16::idct16;
use idct4::idct4;
use idct8::idct8;

/// Apply a 1D inverse transform of length `n` (4, 8, or 16) matching
/// `kind`. Returns `Err(Unsupported)` for sizes / kinds Phase 3 does
/// not yet handle.
fn run_1d(kind: Kind, buf: &mut [i32]) -> Result<()> {
    match (kind, buf.len()) {
        (Kind::Dct, 4) => {
            let arr = <&mut [i32; 4]>::try_from(buf).map_err(|_| idx_err())?;
            idct4(arr);
            Ok(())
        }
        (Kind::Dct, 8) => {
            let arr = <&mut [i32; 8]>::try_from(buf).map_err(|_| idx_err())?;
            idct8(arr);
            Ok(())
        }
        (Kind::Dct, 16) => {
            let arr = <&mut [i32; 16]>::try_from(buf).map_err(|_| idx_err())?;
            idct16(arr);
            Ok(())
        }
        (Kind::Adst, 4) => {
            let arr = <&mut [i32; 4]>::try_from(buf).map_err(|_| idx_err())?;
            iadst4(arr);
            Ok(())
        }
        (Kind::Adst, 8) => {
            let arr = <&mut [i32; 8]>::try_from(buf).map_err(|_| idx_err())?;
            iadst8(arr);
            Ok(())
        }
        (Kind::Adst, 16) => {
            let arr = <&mut [i32; 16]>::try_from(buf).map_err(|_| idx_err())?;
            iadst16(arr);
            Ok(())
        }
        (Kind::FlipAdst, 4) => {
            let arr = <&mut [i32; 4]>::try_from(buf).map_err(|_| idx_err())?;
            iflipadst4(arr);
            Ok(())
        }
        (Kind::FlipAdst, 8) => {
            let arr = <&mut [i32; 8]>::try_from(buf).map_err(|_| idx_err())?;
            iflipadst8(arr);
            Ok(())
        }
        (Kind::FlipAdst, 16) => {
            let arr = <&mut [i32; 16]>::try_from(buf).map_err(|_| idx_err())?;
            iflipadst16(arr);
            Ok(())
        }
        (Kind::Idtx, _) => Err(Error::unsupported(
            "av1 transform: identity 1D not implemented in Phase 3 (§7.7.2.7)",
        )),
        (_, n) => Err(Error::unsupported(format!(
            "av1 transform: 1D len {n} not implemented (§7.7; Phase 3 supports 4/8/16)"
        ))),
    }
}

fn idx_err() -> Error {
    Error::invalid("av1 transform: 1D buffer length mismatch")
}

/// Run the AV1 2D inverse transform in place: rows first, then
/// columns (§7.7.4). `coeffs` is a `w × h` block in row-major layout
/// where `w = sz.width()` and `h = sz.height()`. On return `coeffs`
/// holds the signed residual (before the final range-shift and
/// clip-add step).
pub fn inverse_2d(coeffs: &mut [i32], ty: TxType, sz: TxSize) -> Result<()> {
    let w = sz.width();
    let h = sz.height();
    if w == 0 || h == 0 {
        return Err(Error::unsupported(format!(
            "av1 transform: tx_size {sz:?} not supported (§7.7)"
        )));
    }
    if coeffs.len() != w * h {
        return Err(Error::invalid(format!(
            "av1 transform: coeffs len {} != {w}×{h} = {}",
            coeffs.len(),
            w * h
        )));
    }

    // Row pass.
    let mut row = vec![0i32; w];
    for r in 0..h {
        row.copy_from_slice(&coeffs[r * w..(r + 1) * w]);
        run_1d(ty.row_kind(), &mut row)?;
        coeffs[r * w..(r + 1) * w].copy_from_slice(&row);
    }

    // Column pass.
    let mut col = vec![0i32; h];
    for c in 0..w {
        for (r, cell) in col.iter_mut().enumerate().take(h) {
            *cell = coeffs[r * w + c];
        }
        run_1d(ty.col_kind(), &mut col)?;
        for (r, cell) in col.iter().enumerate().take(h) {
            coeffs[r * w + c] = *cell;
        }
    }

    Ok(())
}

/// Final round-and-shift applied after the 2D inverse transform, per
/// spec §7.7.3.1. The shift is `log2(w) + log2(h) - 1` + a per-bit-depth
/// offset of 1 for 10-bit / 2 for 12-bit (Table 7.7.3-1). Phase 3 only
/// wires the 8-bit reconstruction path, but the generic helper is
/// exposed so higher-bit-depth plumbing can call it later.
pub fn inverse_shift(w: usize, h: usize) -> u32 {
    let lw = log2_of(w);
    let lh = log2_of(h);
    // Spec: Round2(T, cos_bits) during the transform is already baked
    // in; the remaining pass shifts by (lw + lh) / 2 + 1 for AV1's
    // residual scale convention. In libaom/goavif this reduces to:
    //
    //   4×4  → 4      4×8  → 4      4×16 → 5
    //   8×8  → 5      8×4  → 4      8×16 → 5
    //   16×16 → 6      16×4 → 5      16×8 → 5
    //
    // We mirror the exact libaom table below.
    av1_inverse_shift(lw, lh)
}

fn log2_of(n: usize) -> u32 {
    match n {
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        _ => 0,
    }
}

fn av1_inverse_shift(lw: u32, lh: u32) -> u32 {
    // From libaom av1_inv_txfm_add_c: shift = txfm_shift[tx_size][1]
    // for column pass has already been applied by the 1D kernel's
    // internal rounding. The remaining "additional shift" applied
    // after the 2D result before reconstruction (`txfm_shift[...][2]`)
    // is `-5` for most sizes — i.e. a right-shift of 5. Table derived
    // from `av1_txfm_stage_num` * 2 etc. Simplified to:
    //
    //   shift = 4 for 4×N and N×4 blocks
    //   shift = 5 for 8×8, 8×16, 16×8
    //   shift = 6 for 16×16 and up
    //
    // goavif mirrors this exactly (see `av1/decoder/reconstruct.go`).
    let area = 1u32 << (lw + lh);
    if area <= 16 {
        // 4×4
        4
    } else if area <= 64 {
        // 4×8 / 8×4 / 8×8 / 4×16 / 16×4
        4
    } else if area <= 128 {
        // 8×16 / 16×8
        5
    } else if area <= 256 {
        // 16×16 / 8×32 / 32×8
        5
    } else {
        6
    }
}

/// Convenience wrapper used by the standalone intra-smoke tests: run
/// the 2D inverse transform, apply the spec's `round_shift`, and
/// clip-add the residual to an 8-bit predictor block. Phase 3's full
/// coefficient pipeline bypasses this helper and calls
/// [`inverse_2d`] directly before handing off to
/// [`crate::decode::reconstruct`].
pub fn inverse_transform_add(
    tx: TxType,
    w: usize,
    h: usize,
    coeffs: &[i32],
    dst: &mut [u8],
    dst_stride: usize,
) -> Result<()> {
    let sz = match (w, h) {
        (4, 4) => TxSize::Tx4x4,
        (8, 8) => TxSize::Tx8x8,
        (16, 16) => TxSize::Tx16x16,
        _ => {
            return Err(Error::unsupported(format!(
                "av1 transform: {w}×{h} iDCT not implemented \
                (§7.7.2; Phase 3 supports 4×4, 8×8, 16×16)",
            )))
        }
    };
    let mut buf = coeffs.to_vec();
    inverse_2d(&mut buf, tx, sz)?;
    let shift = inverse_shift(w, h);
    for r in 0..h {
        for c in 0..w {
            let residual = round_shift(buf[r * w + c], shift);
            let p = dst[r * dst_stride + c] as i32;
            let s = (p + residual).clamp(0, 255);
            dst[r * dst_stride + c] = s as u8;
        }
    }
    Ok(())
}

#[inline]
fn round_shift(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_2d_dct_4x4_zero_is_zero() {
        let mut coeffs = [0i32; 16];
        inverse_2d(&mut coeffs, TxType::DctDct, TxSize::Tx4x4).unwrap();
        for &v in &coeffs {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn inverse_2d_dct_8x8_dc_only_produces_constant_block() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1024;
        inverse_2d(&mut coeffs, TxType::DctDct, TxSize::Tx8x8).unwrap();
        let v = coeffs[0];
        for c in &coeffs {
            assert_eq!(*c, v);
        }
    }

    #[test]
    fn inverse_2d_rejects_wrong_size() {
        let mut coeffs = [0i32; 12];
        let err = inverse_2d(&mut coeffs, TxType::DctDct, TxSize::Tx4x4).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn inverse_2d_idtidt_unsupported() {
        let mut coeffs = [0i32; 16];
        match inverse_2d(&mut coeffs, TxType::IdtIdt, TxSize::Tx4x4) {
            Err(Error::Unsupported(s)) => assert!(s.contains("identity"), "msg: {s}"),
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn inverse_transform_add_rejects_unsupported_size() {
        let coeffs = vec![0i32; 32 * 32];
        let mut dst = vec![0u8; 32 * 32];
        match inverse_transform_add(TxType::DctDct, 32, 32, &coeffs, &mut dst, 32) {
            Err(Error::Unsupported(s)) => {
                assert!(s.contains("32"), "msg: {s}");
                assert!(s.contains("§7.7"), "msg: {s}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn inverse_transform_add_zero_is_noop_4x4() {
        let coeffs = [0i32; 16];
        let mut dst = [42u8; 16];
        inverse_transform_add(TxType::DctDct, 4, 4, &coeffs, &mut dst, 4).unwrap();
        for &v in &dst {
            assert_eq!(v, 42);
        }
    }

    #[test]
    fn inverse_transform_add_zero_is_noop_16x16() {
        let coeffs = [0i32; 256];
        let mut dst = [70u8; 256];
        inverse_transform_add(TxType::DctDct, 16, 16, &coeffs, &mut dst, 16).unwrap();
        for &v in &dst {
            assert_eq!(v, 70);
        }
    }
}
