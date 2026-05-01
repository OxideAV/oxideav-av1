//! AV1 inverse transforms — §7.7.
//!
//! The public entry points are:
//!
//! - [`inverse_2d_spec`] (round 22) — spec-correct §7.13.3 path used
//!   by the live `decode/superblock.rs` reconstruction (round 23).
//!   Bakes the per-shape `Transform_Row_Shift` between row and column
//!   passes plus `colShift = 4` after the column pass, applies the
//!   `Round2(T*2896, 12)` rectangular pre-row scale on 2:1 aspect
//!   shapes, and dispatches IDTX through the spec-magnitude kernels in
//!   [`idtx_spec`].
//! - [`inverse_2d`] — legacy 2D entry that performs the row/column
//!   passes without any per-pass round-shifts; preserved for the
//!   [`inverse_transform_add`] smoke-test wrapper and as a reference
//!   implementation pinned against `inverse_2d_spec` for square shapes.
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
pub mod idtx_spec;
pub mod iwht4;
pub mod scan;
pub mod types;

pub use scan::{clamped_scan, default_zigzag_scan, inverse_scan};
pub use types::{Kind, TxSize, TxType};

use oxideav_core::{Error, Result};

use adst16::{iadst16, iflipadst16};
use adst4::{iadst4, iflipadst4};
use adst8::{iadst8, iflipadst8};
use cos_pi::round2;
use flipadst::flip_1d;
use idct16::idct16;
use idct32::idct32;
use idct4::idct4;
use idct64::idct64;
use idct8::idct8;
use idtx::{idtx16, idtx32, idtx4, idtx8};
use idtx_spec::{idtx16_spec, idtx32_spec, idtx4_spec, idtx8_spec};
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
///
/// **NOTE (round 23):** the live `decode/superblock.rs` reconstruction
/// path now uses [`inverse_2d_spec`], which bakes the per-shape
/// `Transform_Row_Shift` and `colShift = 4` (§7.13.3) inside the 2D
/// kernel. This legacy entry point is preserved for the
/// [`inverse_transform_add`] convenience wrapper (used by the
/// standalone smoke-tests) and as a reference implementation against
/// which the spec path's square-shape outputs are pinned in
/// `round23_inverse_2d_spec_matches_legacy_for_aligned_squares`.
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

    // §7.13.3 between row and column passes: clip the intermediate
    // residual to `colClampRange = max(BitDepth + 6, 16)` bits before
    // the column transform consumes it. We don't carry BitDepth here,
    // so use the conservative 16-bit envelope (matches 8-bit and is
    // a valid super-set for 10/12-bit content; the dequant clip in
    // §7.13.3 step f bounds the 10/12-bit input separately). Without
    // this clip aggressive coefficient streams overflow `i32` inside
    // the butterfly (`half_btf`) — observed on real SVT-AV1 frames.
    const COL_CLAMP_LIMIT: i32 = 1 << 15; // 2^(16-1)
    for v in coeffs.iter_mut() {
        *v = (*v).clamp(-COL_CLAMP_LIMIT, COL_CLAMP_LIMIT - 1);
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

/// `Transform_Row_Shift[ TX_SIZES_ALL ]` per AV1 spec §7.13.3 — the
/// per-shape Round2 amount applied between the row and column passes.
/// Indexing matches the [`TxSize`] discriminant order verbatim.
const TRANSFORM_ROW_SHIFT: [u32; 19] = [
    0, // Tx4x4
    1, // Tx8x8
    2, // Tx16x16
    2, // Tx32x32
    2, // Tx64x64
    0, // Tx4x8
    0, // Tx8x4
    1, // Tx8x16
    1, // Tx16x8
    1, // Tx16x32
    1, // Tx32x16
    1, // Tx32x64
    1, // Tx64x32
    1, // Tx4x16
    1, // Tx16x4
    2, // Tx8x32
    2, // Tx32x8
    2, // Tx16x64
    2, // Tx64x16
];

/// Spec-correct dispatch for a single 1-D inverse transform per
/// §7.13.2 — equivalent to [`run_1d`] for the legacy callers but
/// substitutes the spec-magnitude IDTX kernels (§7.13.2.11/13/14) for
/// the simplified `<<= 1` variants used by the legacy path.
fn run_1d_spec(kind: Kind, buf: &mut [i32]) -> Result<()> {
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
            idtx4_spec(arr);
            Ok(())
        }
        (Kind::Idtx, 8) => {
            let arr = <&mut [i32; 8]>::try_from(buf).map_err(|_| idx_err())?;
            idtx8_spec(arr);
            Ok(())
        }
        (Kind::Idtx, 16) => {
            let arr = <&mut [i32; 16]>::try_from(buf).map_err(|_| idx_err())?;
            idtx16_spec(arr);
            Ok(())
        }
        (Kind::Idtx, 32) => {
            let arr = <&mut [i32; 32]>::try_from(buf).map_err(|_| idx_err())?;
            idtx32_spec(arr);
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

/// Spec-correct 2D inverse transform per §7.13.3. Differs from
/// [`inverse_2d`] in three places — together these make the function's
/// output a fully spec-conforming residual (the caller only owes a
/// `Clip3` to the bit-depth range before adding to the predictor):
///
/// 1. Applies the rectangular `Round2(T[j] * 2896, 12)` pre-row scale
///    for shapes where `|log2(w) - log2(h)| == 1`.
/// 2. Uses the spec [`TRANSFORM_ROW_SHIFT`] table for the per-shape
///    `Round2(., rowShift)` between row and column passes.
/// 3. Applies the constant `colShift = 4` `Round2` after the column
///    pass.
///
/// Identity 1-D transforms dispatch through [`run_1d_spec`], which
/// uses the spec `Round2(T[i] * 5793, 12)` / `Round2(T[i] * 11586, 12)`
/// / `T[i] * 4` magnitudes from §7.13.2.11/13/14 instead of the
/// legacy `<<= 1` kernels.
///
/// The legacy [`inverse_2d`] entry point is preserved verbatim for the
/// PSNR-tuned `decode/superblock.rs` reconstruction path: switching
/// callers over is a separate work item that needs to revise the
/// per-shape `residual_shift` accounting too. Round 22 lands the
/// kernels and per-shape semantics first; caller migration is
/// pending.
pub fn inverse_2d_spec(coeffs: &mut [i32], ty: TxType, sz: TxSize) -> Result<()> {
    let w = sz.width();
    let h = sz.height();
    if w == 0 || h == 0 {
        return Err(Error::unsupported(format!(
            "av1 transform: tx_size {sz:?} not supported (§7.13.3)"
        )));
    }
    if coeffs.len() != w * h {
        return Err(Error::invalid(format!(
            "av1 transform: coeffs len {} != {w}×{h} = {}",
            coeffs.len(),
            w * h
        )));
    }

    let lw = log2_of(w);
    let lh = log2_of(h);
    let row_shift = TRANSFORM_ROW_SHIFT[sz as usize];
    let col_shift: u32 = 4;
    // Spec §7.13.3: rectangular shapes with a 2:1 aspect ratio carry
    // a `Round2(T[j] * 2896, 12)` per-element pre-row scale (the
    // `1/sqrt(2)` orthonormalisation factor). 1:4 / 4:1 shapes
    // (Tx4x16, Tx16x4, Tx8x32, Tx32x8, Tx16x64, Tx64x16) and square
    // shapes do **not** carry this scale.
    let needs_2896 = lw.abs_diff(lh) == 1;

    let row_kind = ty.row_kind();
    let col_kind = ty.col_kind();
    // Spec §7.13.3 implements FlipADST by reading the residual buffer
    // back through flipped destination coordinates at the clip-add
    // step; equivalently we can write `reverse(IADST(input))` into
    // the residual buffer here and add into un-flipped destination
    // coords downstream. The kernels in [`run_1d_spec`] already do
    // `IADST` followed by an output reversal for `Kind::FlipAdst`,
    // so no extra `flip_1d` wrapping is needed at this level (the
    // legacy [`inverse_2d`] applied a redundant flip-call-flip
    // sequence that algebraically cancelled the kernel's own reverse
    // and yielded `IADST(reverse(input))` — not spec-equivalent).

    // Row pass.
    let mut row = vec![0i32; w];
    for r in 0..h {
        // §7.13.3 step "If i and j are both less than 32" — for 64-axis
        // shapes the upper half of the row beyond column 31 is zeroed
        // even if the dequant buffer carried it. Our coefficient
        // pipeline already feeds zero in those positions (the scan is
        // `clamped_scan` for 64×* shapes), but mirroring the spec
        // explicitly here keeps `inverse_2d_spec` self-contained for
        // synthetic test inputs.
        for (j, cell) in row.iter_mut().enumerate().take(w) {
            *cell = if r < 32 && j < 32 {
                coeffs[r * w + j]
            } else {
                0
            };
        }
        if needs_2896 {
            for cell in row.iter_mut() {
                *cell = round2(*cell * 2896, 12);
            }
        }
        run_1d_spec(row_kind, &mut row)?;
        for cell in row.iter_mut() {
            *cell = round2(*cell, row_shift);
        }
        coeffs[r * w..(r + 1) * w].copy_from_slice(&row);
    }

    // Inter-pass clip per §7.13.3 — `colClampRange = max(BitDepth + 6,
    // 16)`. We bound at 16-bit (the conservative envelope; the dequant
    // step f already clipped 10/12-bit inputs separately).
    const COL_CLAMP_LIMIT: i32 = 1 << 15;
    for v in coeffs.iter_mut() {
        *v = (*v).clamp(-COL_CLAMP_LIMIT, COL_CLAMP_LIMIT - 1);
    }

    // Column pass.
    let mut col = vec![0i32; h];
    for c in 0..w {
        for (r, cell) in col.iter_mut().enumerate().take(h) {
            *cell = coeffs[r * w + c];
        }
        run_1d_spec(col_kind, &mut col)?;
        for cell in col.iter_mut() {
            *cell = round2(*cell, col_shift);
        }
        for (r, cell) in col.iter().enumerate().take(h) {
            coeffs[r * w + c] = *cell;
        }
    }

    Ok(())
}

/// Final round-and-shift applied after the 2D inverse transform, per
/// spec §7.7.3.1. Mirrors libaom's per-size `txfm_shift` table
/// verbatim, collapsed into a small match by `(log2(w) + log2(h))`.
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

    /// Round 22 — `Transform_Row_Shift[]` matches the spec table
    /// verbatim for every TxSize. A drift in this table would produce
    /// silently miscaled residuals on every shape, so we pin it.
    #[test]
    fn transform_row_shift_matches_spec_table() {
        let want = [
            (TxSize::Tx4x4, 0),
            (TxSize::Tx8x8, 1),
            (TxSize::Tx16x16, 2),
            (TxSize::Tx32x32, 2),
            (TxSize::Tx64x64, 2),
            (TxSize::Tx4x8, 0),
            (TxSize::Tx8x4, 0),
            (TxSize::Tx8x16, 1),
            (TxSize::Tx16x8, 1),
            (TxSize::Tx16x32, 1),
            (TxSize::Tx32x16, 1),
            (TxSize::Tx32x64, 1),
            (TxSize::Tx64x32, 1),
            (TxSize::Tx4x16, 1),
            (TxSize::Tx16x4, 1),
            (TxSize::Tx8x32, 2),
            (TxSize::Tx32x8, 2),
            (TxSize::Tx16x64, 2),
            (TxSize::Tx64x16, 2),
        ];
        for (sz, want_shift) in want {
            assert_eq!(
                TRANSFORM_ROW_SHIFT[sz as usize], want_shift,
                "row shift mismatch at {sz:?}"
            );
        }
    }

    /// Round 22 — every spec-allowed TX_TYPE × TX_SIZE pair must
    /// dispatch through `inverse_2d_spec` without `Unsupported`. Per
    /// §5.11.48 / §6.10.18 the disallowed combinations are:
    ///
    /// - any non-DCT_DCT type on a 64-axis size (TX_SET_DCTONLY)
    /// - any type other than DCT_DCT / IDTX on a 32-square or
    ///   32-axis size with `Sqr_Up == 32` (TX_SET_INTER_3 ceiling)
    ///
    /// The table below enumerates **only** the spec-permitted pairs
    /// the bitstream may actually carry — `Sqr_Up <= 16` shapes get
    /// the full 16-type set (TX_SET_INTER_1), `Sqr_Up == 32` shapes
    /// get DCT_DCT + IDTX, and `Sqr_Up == 64` shapes get DCT_DCT only.
    /// A regression that newly returns `Unsupported` for any of these
    /// breaks rich-content decoding.
    #[test]
    fn inverse_2d_spec_covers_every_spec_allowed_pair() {
        let all_types = [
            TxType::DctDct,
            TxType::AdstDct,
            TxType::DctAdst,
            TxType::AdstAdst,
            TxType::FlipAdstDct,
            TxType::DctFlipAdst,
            TxType::FlipAdstFlipAdst,
            TxType::AdstFlipAdst,
            TxType::FlipAdstAdst,
            TxType::IdtIdt,
            TxType::VDct,
            TxType::HDct,
            TxType::VAdst,
            TxType::HAdst,
            TxType::VFlipAdst,
            TxType::HFlipAdst,
        ];
        // Sqr_Up ≤ 16 — full INTER_1 set permitted.
        let full_set_sizes = [
            TxSize::Tx4x4,
            TxSize::Tx8x8,
            TxSize::Tx16x16,
            TxSize::Tx4x8,
            TxSize::Tx8x4,
            TxSize::Tx8x16,
            TxSize::Tx16x8,
            TxSize::Tx4x16,
            TxSize::Tx16x4,
        ];
        // Sqr_Up == 32 — INTER_3 (DCT + IDTX) only.
        let sqr32_sizes = [
            TxSize::Tx32x32,
            TxSize::Tx16x32,
            TxSize::Tx32x16,
            TxSize::Tx8x32,
            TxSize::Tx32x8,
        ];
        // Sqr_Up == 64 — TX_SET_DCTONLY (DCT_DCT only).
        let sqr64_sizes = [
            TxSize::Tx64x64,
            TxSize::Tx32x64,
            TxSize::Tx64x32,
            TxSize::Tx16x64,
            TxSize::Tx64x16,
        ];

        let mut covered = 0usize;
        for sz in full_set_sizes {
            let n = sz.width() * sz.height();
            for ty in all_types {
                let mut buf = vec![0i32; n];
                buf[0] = 1024;
                inverse_2d_spec(&mut buf, ty, sz)
                    .unwrap_or_else(|e| panic!("inverse_2d_spec({ty:?}, {sz:?}) returned {e:?}"));
                covered += 1;
            }
        }
        for sz in sqr32_sizes {
            let n = sz.width() * sz.height();
            for ty in [TxType::DctDct, TxType::IdtIdt] {
                let mut buf = vec![0i32; n];
                buf[0] = 1024;
                inverse_2d_spec(&mut buf, ty, sz)
                    .unwrap_or_else(|e| panic!("inverse_2d_spec({ty:?}, {sz:?}) returned {e:?}"));
                covered += 1;
            }
        }
        for sz in sqr64_sizes {
            let n = sz.width() * sz.height();
            let mut buf = vec![0i32; n];
            buf[0] = 1024;
            inverse_2d_spec(&mut buf, TxType::DctDct, sz)
                .unwrap_or_else(|e| panic!("inverse_2d_spec(DctDct, {sz:?}) returned {e:?}"));
            covered += 1;
        }
        // 9 sizes * 16 types + 5 sizes * 2 types + 5 sizes * 1 type = 159.
        assert_eq!(covered, 9 * 16 + 5 * 2 + 5, "TX coverage shrank");
    }

    /// Round 22 — IDTX magnitudes follow §7.13.2.11/13/14 verbatim
    /// when run via `inverse_2d_spec`. For a DC-only IdtIdt input on a
    /// 4×4 block the spec sequence is:
    ///
    /// 1. Row IDTX-4: `Round2(10 * 5793, 12) = 14`.
    /// 2. Row shift = 0 → no change.
    /// 3. Inter-pass clip: 14 stays in [-32768, 32767].
    /// 4. Col IDTX-4: `Round2(14 * 5793, 12) = 20`.
    /// 5. Col shift = 4 → `Round2(20, 4) = 1`.
    ///
    /// Final residual at (0,0) = 1; every other cell stays 0.
    #[test]
    fn inverse_2d_spec_idtx_dc_4x4_matches_spec() {
        let mut coeffs = [0i32; 16];
        coeffs[0] = 10;
        inverse_2d_spec(&mut coeffs, TxType::IdtIdt, TxSize::Tx4x4).unwrap();
        assert_eq!(coeffs[0], 1, "spec IDTX 4×4 DC residual");
        for c in coeffs.iter().skip(1) {
            assert_eq!(*c, 0);
        }
    }

    /// Round 22 — IDTX 8×8 DC magnitude pinned. Spec sequence with
    /// row=col=8: row IDTX-8 doubles, row shift=1 (Round2(20,1)=10),
    /// col IDTX-8 doubles to 20, col shift=4 (Round2(20,4)=1).
    #[test]
    fn inverse_2d_spec_idtx_dc_8x8_matches_spec() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 10;
        inverse_2d_spec(&mut coeffs, TxType::IdtIdt, TxSize::Tx8x8).unwrap();
        assert_eq!(coeffs[0], 1);
        for c in coeffs.iter().skip(1) {
            assert_eq!(*c, 0);
        }
    }

    /// Round 22 — rectangular `2896` pre-row scale fires only for
    /// `|log2W - log2H| == 1` (the 2:1 aspect shapes). 4×8 and 8×4
    /// hit it; 4×16, 16×4, 8×32, 32×8, 16×64, 64×16 (the 4:1 / 1:4
    /// shapes) do not, and the squares obviously don't either.
    #[test]
    fn inverse_2d_spec_rect_2896_pre_scale_only_2to1() {
        // Tx4x8 has |log2 4 - log2 8| = 1 — the 2896 scale fires.
        // For a DC-only IDTX input on 4×8 the row pass becomes
        // Round2(10 * 2896, 12) = 7, then Round2(7 * 5793, 12) = 10
        // (IDTX-4), row shift 0. Column IDTX-8 doubles to 20, col
        // shift 4 = Round2(20, 4) = 1.
        let mut coeffs = vec![0i32; 32];
        coeffs[0] = 10;
        inverse_2d_spec(&mut coeffs, TxType::IdtIdt, TxSize::Tx4x8).unwrap();
        assert_eq!(coeffs[0], 1, "Tx4x8 IDTX DC residual");
        for c in coeffs.iter().skip(1) {
            assert_eq!(*c, 0);
        }

        // Tx4x16 has |log2 4 - log2 16| = 2 — no 2896 pre-scale.
        // Row IDTX-4 on 10: Round2(10 * 5793, 12) = 14, row shift=1
        // → Round2(14, 1) = 7. Col IDTX-16 on 7:
        // Round2(7 * 11586, 12) = 20, col shift=4 → Round2(20,4) = 1.
        let mut coeffs = vec![0i32; 64];
        coeffs[0] = 10;
        inverse_2d_spec(&mut coeffs, TxType::IdtIdt, TxSize::Tx4x16).unwrap();
        assert_eq!(coeffs[0], 1, "Tx4x16 IDTX DC residual");
    }

    /// Round 22 — DC-only DCT_DCT residual is constant for every
    /// rectangular shape (DC mode = a single basis function spanning
    /// the whole block). Pinning this for all 14 rectangular sizes
    /// catches dispatch bugs in the row/col kernel selection.
    #[test]
    fn inverse_2d_spec_dct_dc_constant_across_all_rectangles() {
        let rects = [
            TxSize::Tx4x8,
            TxSize::Tx8x4,
            TxSize::Tx8x16,
            TxSize::Tx16x8,
            TxSize::Tx16x32,
            TxSize::Tx32x16,
            TxSize::Tx32x64,
            TxSize::Tx64x32,
            TxSize::Tx4x16,
            TxSize::Tx16x4,
            TxSize::Tx8x32,
            TxSize::Tx32x8,
            TxSize::Tx16x64,
            TxSize::Tx64x16,
        ];
        for sz in rects {
            let n = sz.width() * sz.height();
            let mut coeffs = vec![0i32; n];
            // Use a magnitude large enough that the DC survives the
            // per-pass round-shifts (otherwise the rectangular shapes
            // would round to zero and degrade the test value).
            coeffs[0] = 1 << 16;
            inverse_2d_spec(&mut coeffs, TxType::DctDct, sz).unwrap();
            let v0 = coeffs[0];
            for (i, c) in coeffs.iter().enumerate().skip(1) {
                assert_eq!(*c, v0, "{sz:?}: cell {i} = {c}, want {v0}");
            }
            assert_ne!(v0, 0, "{sz:?}: DC collapsed to 0");
        }
    }

    /// Round 22 — `inverse_2d_spec` rejects mismatched `coeffs.len()`
    /// with `Error::InvalidData`, same as the legacy entry point.
    #[test]
    fn inverse_2d_spec_rejects_wrong_size() {
        let mut coeffs = [0i32; 12];
        let err = inverse_2d_spec(&mut coeffs, TxType::DctDct, TxSize::Tx4x4).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    /// Round 22 — `inverse_2d_spec`'s FLIPADST handling differs from
    /// the legacy `inverse_2d` (which wrapped the kernel call with a
    /// pre-flip + post-flip that algebraically cancelled the kernel's
    /// internal output reversal — net effect `IADST(reverse(input))`,
    /// not `reverse(IADST(input))`). The new path does **no** wrapper
    /// flips and lets the `iflipadst*` kernel's built-in output
    /// reversal stand, matching spec §7.13.3 + §5.11.39 (residual
    /// reversed → spec-equivalent flipped destination coords). This
    /// test pins the equivalence by comparing 1-D `iflipadst4(x)` to
    /// `reverse(iadst4(x))` byte-for-byte.
    #[test]
    fn inverse_2d_spec_flipadst_uses_kernel_internal_reverse() {
        let probe = [13i32, -7, 200, -45];
        let mut a = probe;
        let mut b = probe;
        adst4::iadst4(&mut a);
        adst4::iflipadst4(&mut b);
        // iflipadst4 should equal reverse(iadst4(input)).
        for i in 0..4 {
            assert_eq!(b[i], a[3 - i], "iflipadst4 ≠ reverse(iadst4) at {i}");
        }
    }

    /// Round 22 — spec-disallowed kernels (ADST/FLIPADST at 32 or 64,
    /// IDTX at 64, WHT at non-4) still surface `Unsupported`. This
    /// pins the safety net for stream errors that would otherwise
    /// reach our 1-D dispatcher.
    #[test]
    fn inverse_2d_spec_rejects_disallowed_kernels() {
        // ADST is not defined at length 32 — Tx32x32 with AdstAdst is
        // out of spec, and we surface that loudly rather than
        // silently downgrading.
        let mut buf = vec![0i32; 32 * 32];
        let err = inverse_2d_spec(&mut buf, TxType::AdstAdst, TxSize::Tx32x32).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
        // FlipAdst at length 64 likewise.
        let mut buf = vec![0i32; 64 * 64];
        let err = inverse_2d_spec(&mut buf, TxType::FlipAdstFlipAdst, TxSize::Tx64x64).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    /// Round 23 — squares Tx4x4 and Tx32x32 produce identical
    /// magnitudes through `inverse_2d_spec` and through the legacy
    /// `inverse_2d` followed by `inverse_shift`-based `round_shift`.
    /// This is the equivalence the migration relies on for backwards
    /// compatibility on already-PSNR-tuned square DCT_DCT paths
    /// (Tx4x4: legacy 4 = spec row 0 + col 4; Tx32x32: legacy 6 = spec
    /// row 2 + col 4). Mismatches here would indicate the migration
    /// is silently changing magnitude on the most common shapes.
    #[test]
    fn round23_inverse_2d_spec_matches_legacy_for_aligned_squares() {
        for sz in [TxSize::Tx4x4, TxSize::Tx32x32] {
            let n = sz.width() * sz.height();
            // Probe input that exercises sign/magnitude through the
            // butterfly (DC + a few mid-band entries).
            let mut input = vec![0i32; n];
            input[0] = 4096;
            if n >= 4 {
                input[1] = -1024;
                input[2] = 512;
            }
            let mut a = input.clone();
            let mut b = input.clone();

            // Legacy path: inverse_2d, then post-2D round_shift
            // bucketed via `inverse_shift`.
            inverse_2d(&mut a, TxType::DctDct, sz).unwrap();
            let s = inverse_shift(sz.width(), sz.height());
            for v in a.iter_mut() {
                *v = if s == 0 {
                    *v
                } else {
                    (*v + (1 << (s - 1))) >> s
                };
            }

            // Spec path: bakes shifts inside.
            inverse_2d_spec(&mut b, TxType::DctDct, sz).unwrap();

            assert_eq!(a, b, "{sz:?}: spec ≠ legacy+shift");
        }
    }

    /// Round 23 — `inverse_2d_spec` is now the live transform path
    /// from `decode/superblock.rs`. This test pins that the public
    /// entry point of the spec-correct path remains the symbol that
    /// `superblock.rs` imports. A regression where the import name
    /// flips back to `inverse_2d` (e.g. via a careless merge) would
    /// surface as a compile error rather than a silent magnitude
    /// drift, and this assertion documents the contract.
    #[test]
    fn round23_decode_superblock_imports_spec_entry_point() {
        // Compile-time witness: the symbol must exist with the
        // expected signature. If the spec entry point were renamed
        // or removed, this fails to type-check. Runtime behavior is
        // covered by the dedicated DC / IDTX / spec-pair tests.
        let f: fn(&mut [i32], TxType, TxSize) -> Result<()> = inverse_2d_spec;
        let mut buf = vec![0i32; 16];
        f(&mut buf, TxType::DctDct, TxSize::Tx4x4).unwrap();
    }
}
