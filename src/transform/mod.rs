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
//! Phase 4 closes out the kernel set: 32/64-point DCT, IDTX,
//! flipped-ADST via the same buffer-reversal helper used by libaom,
//! and the full mixed V/H (V_DCT, V_ADST, V_FLIPADST, H_*) dispatch.
//! The only kernel still not called from decoder code is IWHT4
//! (lossless mode — AVIF stills never set `Lossless=1`); it's exposed
//! for completeness.

pub mod adst16;
pub mod adst4;
pub mod adst8;
pub mod cos_pi;
pub mod flipadst;
pub mod idct16;
pub mod idct32;
pub mod idct4;
pub mod idct64;
pub mod idct8;
pub mod idtx;
pub mod iwht4;
pub mod scan;
pub mod types;

pub use scan::{default_zigzag_scan, inverse_scan};
pub use types::{Kind, TxSize, TxType};

use oxideav_core::{Error, Result};

use adst16::{iadst16, iflipadst16};
use adst4::{iadst4, iflipadst4};
use adst8::{iadst8, iflipadst8};
use flipadst::flip_1d;
use idct16::idct16;
use idct32::idct32;
use idct4::idct4;
use idct64::idct64;
use idct8::idct8;
use idtx::{idtx16, idtx32, idtx4, idtx8};
use iwht4::iwht4;

/// Apply a 1D inverse transform of length `n` matching `kind`.
/// Returns `Err(Unsupported)` only for length/kind pairs the spec
/// never instantiates (e.g. 32-point ADST, 64-point ADST).
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
        (Kind::Dct, 32) => {
            let arr = <&mut [i32; 32]>::try_from(buf).map_err(|_| idx_err())?;
            idct32(arr);
            Ok(())
        }
        (Kind::Dct, 64) => {
            let arr = <&mut [i32; 64]>::try_from(buf).map_err(|_| idx_err())?;
            idct64(arr);
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
        (Kind::Idtx, 4) => {
            let arr = <&mut [i32; 4]>::try_from(buf).map_err(|_| idx_err())?;
            idtx4(arr);
            Ok(())
        }
        (Kind::Idtx, 8) => {
            let arr = <&mut [i32; 8]>::try_from(buf).map_err(|_| idx_err())?;
            idtx8(arr);
            Ok(())
        }
        (Kind::Idtx, 16) => {
            let arr = <&mut [i32; 16]>::try_from(buf).map_err(|_| idx_err())?;
            idtx16(arr);
            Ok(())
        }
        (Kind::Idtx, 32) => {
            let arr = <&mut [i32; 32]>::try_from(buf).map_err(|_| idx_err())?;
            idtx32(arr);
            Ok(())
        }
        (Kind::Wht, 4) => {
            let arr = <&mut [i32; 4]>::try_from(buf).map_err(|_| idx_err())?;
            iwht4(arr);
            Ok(())
        }
        (Kind::Adst, 32)
        | (Kind::Adst, 64)
        | (Kind::FlipAdst, 32)
        | (Kind::FlipAdst, 64)
        | (Kind::Idtx, 64)
        | (Kind::Wht, _) => Err(Error::unsupported(format!(
            "av1 transform: 1D {kind:?} at len {} not instantiated by the spec",
            buf.len()
        ))),
        (_, n) => Err(Error::unsupported(format!(
            "av1 transform: unsupported 1D length {n} (§7.7 — must be 4/8/16/32/64)"
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

    let row_kind = ty.row_kind();
    let col_kind = ty.col_kind();
    let row_flip = ty.row_flip();
    let col_flip = ty.col_flip();

    // Row pass.
    let mut row = vec![0i32; w];
    for r in 0..h {
        row.copy_from_slice(&coeffs[r * w..(r + 1) * w]);
        if row_flip {
            flip_1d(&mut row);
        }
        run_1d(row_kind, &mut row)?;
        if row_flip {
            flip_1d(&mut row);
        }
        coeffs[r * w..(r + 1) * w].copy_from_slice(&row);
    }

    // Column pass.
    let mut col = vec![0i32; h];
    for c in 0..w {
        for (r, cell) in col.iter_mut().enumerate().take(h) {
            *cell = coeffs[r * w + c];
        }
        if col_flip {
            flip_1d(&mut col);
        }
        run_1d(col_kind, &mut col)?;
        if col_flip {
            flip_1d(&mut col);
        }
        for (r, cell) in col.iter().enumerate().take(h) {
            coeffs[r * w + c] = *cell;
        }
    }

    Ok(())
}

/// Final round-and-shift applied after the 2D inverse transform, per
/// spec §7.7.3.1. goavif mirrors libaom's per-size `txfm_shift` table
/// verbatim; the Rust port is the same table collapsed into a small
/// match by `(log2(w) + log2(h))`.
pub fn inverse_shift(w: usize, h: usize) -> u32 {
    let lw = log2_of(w);
    let lh = log2_of(h);
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
    // libaom `av1_inv_txfm_add_c` final shift = txfm_shift[tx_size][2].
    // Collapsed to area buckets: 4 for ≤64 samples, 5 for 128/256,
    // 6 for anything larger.
    let area = 1u32 << (lw + lh);
    if area <= 64 {
        4
    } else if area <= 256 {
        5
    } else {
        6
    }
}

/// Convenience wrapper used by the standalone intra-smoke tests: run
/// the 2D inverse transform, apply the spec's `round_shift`, and
/// clip-add the residual to an 8-bit predictor block. The main
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
        (32, 32) => TxSize::Tx32x32,
        (64, 64) => TxSize::Tx64x64,
        _ => {
            return Err(Error::unsupported(format!(
                "av1 transform: {w}×{h} iDCT not supported via helper \
                (§7.7.2; supported: 4×4/8×8/16×16/32×32/64×64)",
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
    fn inverse_2d_dct_32x32_dc_only_produces_constant_block() {
        let mut coeffs = vec![0i32; 32 * 32];
        coeffs[0] = 32768;
        inverse_2d(&mut coeffs, TxType::DctDct, TxSize::Tx32x32).unwrap();
        let v = coeffs[0];
        for c in &coeffs {
            assert_eq!(*c, v);
        }
    }

    #[test]
    fn inverse_2d_dct_64x64_dc_only_produces_constant_block() {
        let mut coeffs = vec![0i32; 64 * 64];
        coeffs[0] = 65536;
        inverse_2d(&mut coeffs, TxType::DctDct, TxSize::Tx64x64).unwrap();
        let v = coeffs[0];
        for c in &coeffs {
            assert_eq!(*c, v);
        }
    }

    #[test]
    fn inverse_2d_idtx_is_bitshift() {
        // For IdtIdt, 4×4 at DC only: sample = DC << 1 (row) << 1 (col) = DC << 2.
        let mut coeffs = [0i32; 16];
        coeffs[0] = 10;
        inverse_2d(&mut coeffs, TxType::IdtIdt, TxSize::Tx4x4).unwrap();
        assert_eq!(coeffs[0], 40);
        for c in coeffs.iter().skip(1) {
            assert_eq!(*c, 0);
        }
    }

    #[test]
    fn inverse_2d_rejects_wrong_size() {
        let mut coeffs = [0i32; 12];
        let err = inverse_2d(&mut coeffs, TxType::DctDct, TxSize::Tx4x4).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
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

    #[test]
    fn inverse_transform_add_zero_is_noop_32x32() {
        let coeffs = vec![0i32; 32 * 32];
        let mut dst = vec![77u8; 32 * 32];
        inverse_transform_add(TxType::DctDct, 32, 32, &coeffs, &mut dst, 32).unwrap();
        for &v in &dst {
            assert_eq!(v, 77);
        }
    }
}
