//! r441 — the §5.9.8 SUPERRES election helpers.
//!
//! The frame codes at the §5.9.8 downscaled `FrameWidth`; the header
//! carries `UpscaledWidth` (via the sequence maximum on KEY frames)
//! plus `coded_denom`, and the decoder horizontally upscales the
//! in-loop result through §7.16 before output / reference storage.
//! The encoder mirrors that exactly: the reconstruction is upscaled
//! through the decoder's OWN §7.16 driver (`crate::superres`), so the
//! stored reference equals the decoder's byte for byte.
//!
//! The DOWNSCALER is an encoder-side choice the spec deliberately
//! leaves open (§7.16 only defines upscaling): a `[1, 2, 1]/4`
//! horizontal low-pass followed by a centre-aligned linear resample
//! at 1/64-sample precision. Only election quality depends on it —
//! conformance never does.

use crate::encoder::yuv_frame::YuvFrame;
use crate::frame_header::{SUPERRES_DENOM_MIN, SUPERRES_NUM};
use crate::loop_filter::PlaneBuffer;
use crate::superres::{upscale_frame, SuperresFrameContext};
use crate::Error;

/// §5.9.8 `FrameWidth` derivation:
/// `(UpscaledWidth * SUPERRES_NUM + (SuperresDenom / 2)) /
/// SuperresDenom`.
#[must_use]
pub(crate) fn superres_coded_width(upscaled_width: u32, denom: u32) -> u32 {
    (upscaled_width * SUPERRES_NUM + denom / 2) / denom
}

/// The candidate §5.9.8 denominators for a frame of `width`: every
/// `SuperresDenom ∈ [SUPERRES_DENOM_MIN, 16]` whose derived coded
/// width stays inside the encoder core's geometry contract (a
/// multiple of 8, at least 8, strictly smaller than `width`). An
/// election-scoping choice — the decode side handles every §5.9.8
/// configuration.
#[must_use]
pub(crate) fn candidate_denoms(width: u32) -> Vec<u32> {
    (SUPERRES_DENOM_MIN..=16)
        .filter(|&d| {
            let wd = superres_coded_width(width, d);
            wd >= 8 && wd % 8 == 0 && wd < width
        })
        .collect()
}

/// §5.11.27 `is_scaled( refFrame )` over luma extents: the §7.11.3.3
/// scale factors `xScale = ((RefUpscaledWidth << REF_SCALE_SHIFT) +
/// (FrameWidth / 2)) / FrameWidth` (and the height twin) differ from
/// `1 << REF_SCALE_SHIFT`. r456 — the inter superres arm codes every
/// frame against references held at the UPSCALED extent, so this is
/// the gate that collapses the §5.11.27 `motion_mode` read to the
/// `use_obmc` arm and bars the §7.11.3.1 step-7 global warp.
#[must_use]
pub(crate) fn is_scaled(ref_upscaled_width: u32, ref_height: u32, width: u32, height: u32) -> bool {
    const SHIFT: u32 = crate::inter_pred::REF_SCALE_SHIFT;
    let x_scale =
        ((u64::from(ref_upscaled_width) << SHIFT) + u64::from(width / 2)) / u64::from(width);
    let y_scale = ((u64::from(ref_height) << SHIFT) + u64::from(height / 2)) / u64::from(height);
    x_scale != (1u64 << SHIFT) || y_scale != (1u64 << SHIFT)
}

/// The §5.9.8 tile-width conformance gate on a candidate denominator
/// (Annex A: "if use_superres is equal to 1 and RightMostTile is
/// equal to 0, then TileWidth is greater than or equal to 128"): a
/// uniform `(TileColsLog2, TileRowsLog2)` layout at the DOWNSCALED
/// width must keep every non-rightmost tile column at least 128 luma
/// samples wide. `tiles == (0, 0)` always passes.
#[must_use]
pub(crate) fn denom_tile_ok(width: u32, height: u32, denom: u32, tiles: (u32, u32)) -> bool {
    if tiles == (0, 0) {
        return true;
    }
    let wd = superres_coded_width(width, denom);
    let Some(ti) =
        crate::tile_info::TileInfo::uniform_layout(wd / 4, height / 4, false, tiles.0, tiles.1)
    else {
        return false;
    };
    ti.mi_col_starts
        .windows(2)
        .take(ti.tile_cols.saturating_sub(1) as usize)
        .all(|w| (w[1] - w[0]) * 4 >= 128)
}

/// r456 — bilinear resample of a luma plane to an arbitrary extent
/// (centre-aligned, edge-clamped): the integer-pel SEARCH aid for a
/// reference whose extent differs from the coded frame's in either
/// direction (a spatial-SVC lower layer is SMALLER than the frame;
/// a superres reference is wider). A selection aid only — every
/// candidate that matters is scored through the decoder's own
/// §7.11.3.3 / §7.11.3.4 kernels.
#[must_use]
pub(crate) fn resample_plane_bilinear(
    src: &[u16],
    w_src: usize,
    h_src: usize,
    w_dst: usize,
    h_dst: usize,
) -> Vec<u16> {
    debug_assert_eq!(src.len(), w_src * h_src);
    if w_src == w_dst && h_src == h_dst {
        return src.to_vec();
    }
    // Source coordinate (in 1/64 samples) of destination sample `x`:
    // (x + 1/2) · src / dst − 1/2, clamped at the plane edges.
    let axis = |dst: usize, srcn: usize| -> Vec<(usize, usize, u32)> {
        (0..dst)
            .map(|x| {
                let s64 = ((2 * x as i64 + 1) * srcn as i64 * 32) / (dst as i64) - 32;
                let s64 = s64.max(0) as usize;
                let i0 = (s64 >> 6).min(srcn - 1);
                let i1 = (i0 + 1).min(srcn - 1);
                (i0, i1, (s64 & 63) as u32)
            })
            .collect()
    };
    let xs = axis(w_dst, w_src);
    let ys = axis(h_dst, h_src);
    let mut out = vec![0u16; w_dst * h_dst];
    for (y, &(r0, r1, fy)) in ys.iter().enumerate() {
        let row0 = &src[r0 * w_src..(r0 + 1) * w_src];
        let row1 = &src[r1 * w_src..(r1 + 1) * w_src];
        for (x, &(c0, c1, fx)) in xs.iter().enumerate() {
            let top = (64 - fx) * u32::from(row0[c0]) + fx * u32::from(row0[c1]);
            let bot = (64 - fx) * u32::from(row1[c0]) + fx * u32::from(row1[c1]);
            let v = ((64 - fy) * top + fy * bot + 2048) >> 12;
            out[y * w_dst + x] = v as u16;
        }
    }
    out
}

/// r456 — remap an explicit (§5.9.15 non-uniform) column layout onto
/// the DOWNSCALED superblock grid: the widths (superblock units,
/// summing to `sb_cols_full`) scale proportionally onto
/// `sb_cols_coded` columns with the column COUNT preserved (so a
/// primary frame's per-tile donor CDFs still line up), every column
/// at least one superblock, the rounding residue settled on the
/// widest columns, and — the Annex A superres rule — every
/// non-rightmost column at least `min_sb` superblocks (128 luma
/// samples on 64-sample superblocks). `None` when no such layout
/// exists.
#[must_use]
pub(crate) fn remap_explicit_widths(
    widths_sb: &[u32],
    sb_cols_full: u32,
    sb_cols_coded: u32,
    min_sb: u32,
) -> Option<Vec<u32>> {
    let n = widths_sb.len();
    if n == 0 || widths_sb.contains(&0) || widths_sb.iter().sum::<u32>() != sb_cols_full {
        return None;
    }
    let floor_total: u32 = (n as u32 - 1) * min_sb + 1;
    if sb_cols_coded < floor_total {
        return None;
    }
    let mut out: Vec<u32> = widths_sb
        .iter()
        .map(|&w| ((w * sb_cols_coded + sb_cols_full / 2) / sb_cols_full).max(1))
        .collect();
    for (i, w) in out.iter_mut().enumerate() {
        if i + 1 < n && *w < min_sb {
            *w = min_sb;
        }
    }
    // Settle the residue: shrink the widest column that can still give
    // (respecting its floor), grow the widest column.
    loop {
        let sum: u32 = out.iter().sum();
        if sum == sb_cols_coded {
            break;
        }
        if sum > sb_cols_coded {
            let floor_of = |i: usize| if i + 1 < n { min_sb } else { 1 };
            let idx = (0..n)
                .filter(|&i| out[i] > floor_of(i))
                .max_by_key(|&i| (out[i], std::cmp::Reverse(i)))?;
            out[idx] -= 1;
        } else {
            let idx = (0..n)
                .max_by_key(|&i| (out[i], std::cmp::Reverse(i)))
                .expect("non-empty");
            out[idx] += 1;
        }
    }
    Some(out)
}

/// r456 — the explicit-layout twin of [`denom_tile_ok`]: the remapped
/// column widths for a candidate denominator when the layout stays
/// inside the §5.9.15 legal window at the downscaled width AND the
/// Annex A superres tile-width rule; `None` filters the candidate.
/// (64-sample superblocks — every conformance-grade driver's shape.)
#[must_use]
pub(crate) fn denom_explicit_ok(
    width: u32,
    height: u32,
    denom: u32,
    widths_sb: &[u32],
    heights_sb: &[u32],
) -> Option<Vec<u32>> {
    let wd = superres_coded_width(width, denom);
    let sb_full = (2 * ((width + 7) >> 3)).div_ceil(16);
    let sb_coded = (2 * ((wd + 7) >> 3)).div_ceil(16);
    let remapped = remap_explicit_widths(widths_sb, sb_full, sb_coded, 2)?;
    let ti = crate::tile_info::TileInfo::explicit_layout(
        2 * ((wd + 7) >> 3),
        2 * ((height + 7) >> 3),
        false,
        &remapped,
        heights_sb,
    )?;
    ti.mi_col_starts
        .windows(2)
        .take(ti.tile_cols.saturating_sub(1) as usize)
        .all(|w| (w[1] - w[0]) * 4 >= 128)
        .then_some(remapped)
}

/// r456 — the INTER-frame candidate set (bounded: each candidate is a
/// full P-frame search): the KEY-elected denominator when the KEY
/// took the arm (the GOP's proven ratio), else the legal denominator
/// nearest the ladder's midpoint (12), plus the strongest legal
/// downscale (16 — the §6.8.2 `2 * FrameWidth >= RefUpscaledWidth`
/// bound is met at every legal ratio). At most two entries.
#[must_use]
pub(crate) fn inter_candidate_denoms(
    width: u32,
    height: u32,
    tiles: (u32, u32),
    explicit: Option<(&[u32], &[u32])>,
    key_denom: Option<u32>,
) -> Vec<u32> {
    let legal: Vec<u32> = candidate_denoms(width)
        .into_iter()
        .filter(|&d| match explicit {
            Some((ws, hs)) => denom_explicit_ok(width, height, d, ws, hs).is_some(),
            None => denom_tile_ok(width, height, d, tiles),
        })
        .filter(|&d| 2 * superres_coded_width(width, d) >= width)
        .collect();
    if legal.is_empty() {
        return Vec::new();
    }
    let first = match key_denom {
        Some(d) if legal.contains(&d) => d,
        _ => *legal
            .iter()
            .min_by_key(|&&d| (d as i64 - 12).unsigned_abs())
            .expect("non-empty"),
    };
    let mut out = vec![first];
    let strongest = *legal.iter().max().expect("non-empty");
    if strongest != first {
        out.push(strongest);
    }
    out
}

/// r441 — the superres arm's ARMING WINDOW (the measured regime; see
/// `tests/superres_ab.rs`). On probe-passing content the arm wins
/// across the whole measured quantiser band (q60..q220: −5 % to −20 %
/// bytes at comparable-or-better PSNR), so the window is broad; tiny
/// frames' fixed header cost dominates any reshape. An encoder
/// election-scoping choice, not a conformance constraint (it also
/// bounds the extra full searches the tiered CI pays for).
#[must_use]
pub(crate) fn superres_arm_allowed(base_q_idx: u8, width: usize, height: usize) -> bool {
    base_q_idx >= 60 && width * height >= 96 * 80
}

/// r441 — the superres arm's CONTENT gate: mean absolute HORIZONTAL
/// luma second difference, in 1/16ths of an 8-bit-normalized sample
/// step (the horizontal sibling of the §5.9.12 election's probe).
/// §7.16 resamples columns only, so vertical detail survives intact —
/// what decides the arm is whether the HORIZONTAL spectrum fits
/// through the downscale: on fine-horizontal-detail content the
/// upscaler cannot recreate the lost columns (measured: an 18 dB →
/// 14 dB collapse) while on horizontally smooth content the loss is
/// under a dB against a double-digit rate win. Skipping the arm on
/// probe-failing content keeps those streams bit-identical to the
/// baseline.
#[must_use]
pub(crate) fn superres_probe(input: &YuvFrame) -> bool {
    let (w, h) = (input.width as usize, input.height as usize);
    if w < 3 {
        return false;
    }
    let mut sum = 0u64;
    let mut count = 0u64;
    for r in 0..h {
        for c in 1..w - 1 {
            let m = i64::from(input.y[r * w + c]);
            sum += (2 * m - i64::from(input.y[r * w + c - 1]) - i64::from(input.y[r * w + c + 1]))
                .unsigned_abs();
            count += 1;
        }
    }
    if count == 0 {
        return false;
    }
    let mean16 = (sum * 16) / count / (1u64 << (input.bit_depth - 8));
    mean16 < 10
}

/// Downscale `input` horizontally to `coded_width` luma columns
/// (chroma planes follow their §6.4.1 subsampled extents). The
/// per-plane resample is `[1, 2, 1]/4` low-pass + centre-aligned
/// linear interpolation at 1/64-sample precision — see the module
/// notes.
#[must_use]
pub(crate) fn downscale_width(input: &YuvFrame, coded_width: u32) -> YuvFrame {
    let (ssx, _ssy) = input.format.subsampling();
    let h = input.height as usize;
    let ch = input.chroma_height() as usize;
    let y = downscale_plane_width(&input.y, input.width as usize, h, coded_width as usize);
    let (u, v) = if input.u.is_empty() {
        (Vec::new(), Vec::new())
    } else {
        let cw_src = input.chroma_width() as usize;
        let cw_dst = (coded_width >> ssx) as usize;
        (
            downscale_plane_width(&input.u, cw_src, ch, cw_dst),
            downscale_plane_width(&input.v, cw_src, ch, cw_dst),
        )
    };
    YuvFrame {
        width: coded_width,
        height: input.height,
        bit_depth: input.bit_depth,
        format: input.format,
        y,
        u,
        v,
    }
}

pub(crate) fn downscale_plane_width(src: &[u16], w_src: usize, h: usize, w_dst: usize) -> Vec<u16> {
    debug_assert!(w_dst < w_src && w_dst > 0);
    let mut out = vec![0u16; w_dst * h];
    let mut lp = vec![0u32; w_src];
    for r in 0..h {
        let row = &src[r * w_src..(r + 1) * w_src];
        // [1, 2, 1] / 4 horizontal low-pass with edge replication.
        for x in 0..w_src {
            let l = row[x.saturating_sub(1)] as u32;
            let m = row[x] as u32;
            let rr = row[(x + 1).min(w_src - 1)] as u32;
            lp[x] = (l + 2 * m + rr + 2) >> 2;
        }
        for (x, slot) in out[r * w_dst..(r + 1) * w_dst].iter_mut().enumerate() {
            // Centre-aligned source position in 1/64 samples:
            // (x + 1/2) * w_src / w_dst - 1/2.
            let s64 = ((2 * x as u64 + 1) * w_src as u64 * 32) / (w_dst as u64) - 32;
            let i0 = (s64 >> 6) as usize;
            let frac = (s64 & 63) as u32;
            let i1 = (i0 + 1).min(w_src - 1);
            *slot = (((64 - frac) * lp[i0] + frac * lp[i1] + 32) >> 6) as u16;
        }
    }
    out
}

/// Upscale a coded-extent reconstruction to `upscaled_width` through
/// the decoder's own §7.16 driver (identical filter taps, subpel walk
/// and edge clamps — the stored reference must equal the decoder's
/// byte for byte). Planes are exact-extent (`width` is a multiple of
/// 8, so mi-padded and cropped extents coincide).
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn upscale_recon(
    planes: [&[u16]; 3],
    width: u32,
    height: u32,
    upscaled_width: u32,
    mi_cols: u32,
    bit_depth: u8,
    ssx: u8,
    ssy: u8,
    num_planes: usize,
) -> Result<(Vec<u16>, Vec<u16>, Vec<u16>), Error> {
    let ctx = SuperresFrameContext {
        use_superres: true,
        frame_width: width,
        upscaled_width,
        frame_height: height,
        mi_cols,
        num_planes: num_planes as u8,
        bit_depth,
        subsampling_x: ssx,
        subsampling_y: ssy,
    };
    let dims = |plane: usize| -> (u32, u32, u32) {
        if plane == 0 {
            (width, upscaled_width, height)
        } else {
            (
                (width + u32::from(ssx)) >> ssx,
                (upscaled_width + u32::from(ssx)) >> ssx,
                (height + u32::from(ssy)) >> ssy,
            )
        }
    };
    let mut in_owned: Vec<Vec<i32>> = Vec::with_capacity(num_planes);
    let mut out_owned: Vec<Vec<i32>> = Vec::with_capacity(num_planes);
    for (plane, src) in planes.iter().enumerate().take(num_planes) {
        let (pw, up_w, ph) = dims(plane);
        debug_assert_eq!(src.len(), (pw * ph) as usize);
        in_owned.push(src.iter().map(|&s| i32::from(s)).collect());
        out_owned.push(vec![0i32; (up_w * ph) as usize]);
    }
    {
        let mut inputs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
        for (plane, buf) in in_owned.iter_mut().enumerate() {
            let (pw, _, ph) = dims(plane);
            inputs.push(PlaneBuffer {
                rows: ph,
                cols: pw,
                samples: buf,
            });
        }
        let mut outputs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
        for (plane, buf) in out_owned.iter_mut().enumerate() {
            let (_, up_w, ph) = dims(plane);
            outputs.push(PlaneBuffer {
                rows: ph,
                cols: up_w,
                samples: buf,
            });
        }
        upscale_frame(&ctx, &inputs, &mut outputs).map_err(|_| Error::PartitionWalkOutOfRange)?;
    }
    let narrow = |v: Vec<i32>| -> Vec<u16> {
        v.into_iter()
            .map(|s| s.clamp(0, u16::MAX as i32) as u16)
            .collect()
    };
    let mut it = out_owned.into_iter();
    let y = narrow(it.next().unwrap_or_default());
    let u = narrow(it.next().unwrap_or_default());
    let v = narrow(it.next().unwrap_or_default());
    Ok((y, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// r456 — explicit column remap: count preserved, sums land on the
    /// coded grid, the Annex A floor holds on every non-rightmost
    /// column, infeasible grids return `None`.
    #[test]
    fn explicit_remap_preserves_count_and_floors() {
        assert_eq!(remap_explicit_widths(&[3, 2], 5, 3, 2), Some(vec![2, 1]));
        assert_eq!(remap_explicit_widths(&[3, 2], 5, 4, 2), Some(vec![2, 2]));
        // Three columns need 2 + 2 + 1 = 5 superblocks: the 5-wide
        // grid keeps the identity, a 6-wide grid grows the widest.
        assert_eq!(
            remap_explicit_widths(&[1, 3, 1], 5, 5, 2),
            Some(vec![2, 2, 1])
        );
        assert_eq!(
            remap_explicit_widths(&[1, 3, 1], 5, 6, 2),
            Some(vec![2, 3, 1])
        );
        // 5 columns of a 2-superblock floor cannot fit in 4.
        assert_eq!(remap_explicit_widths(&[1, 1, 1], 3, 4, 2), None);
        assert_eq!(remap_explicit_widths(&[2, 2], 4, 2, 2), None);
        // Zero-width or mis-summed inputs reject.
        assert_eq!(remap_explicit_widths(&[0, 4], 4, 3, 2), None);
        assert_eq!(remap_explicit_widths(&[2, 1], 4, 3, 2), None);
        // Identity when the grids coincide.
        assert_eq!(remap_explicit_widths(&[2, 3], 5, 5, 2), Some(vec![2, 3]));
    }

    /// r456 — the 320-wide `[3, 2]` layout survives the 16 and 10
    /// denominators (`[2, 1]` at 160, `[2, 2]` at 256) and fails the
    /// ratios whose coded width leaves fewer than 3 superblocks.
    #[test]
    fn explicit_denom_filter_on_the_320_layout() {
        assert_eq!(
            denom_explicit_ok(320, 96, 16, &[3, 2], &[2]),
            Some(vec![2, 1])
        );
        assert_eq!(
            denom_explicit_ok(320, 96, 10, &[3, 2], &[2]),
            Some(vec![2, 2])
        );
        let cands = inter_candidate_denoms(320, 96, (0, 0), Some((&[3, 2], &[2])), None);
        assert!(cands
            .iter()
            .all(|&d| denom_explicit_ok(320, 96, d, &[3, 2], &[2]).is_some()));
        assert!(cands.contains(&16));
    }

    /// §5.9.8 width derivation on the denominator ladder.
    #[test]
    fn coded_width_follows_spec_derivation() {
        // (128 * 8 + 8) / 16 = 64; (96 * 8 + 6) / 12 = 64.
        assert_eq!(superres_coded_width(128, 16), 64);
        assert_eq!(superres_coded_width(96, 12), 64);
        assert_eq!(superres_coded_width(96, 16), 48);
    }

    /// Candidates keep the coded width inside the core's geometry
    /// contract.
    #[test]
    fn candidates_are_geometry_legal() {
        for w in (8..=512).step_by(8) {
            for d in candidate_denoms(w) {
                let wd = superres_coded_width(w, d);
                assert!(wd >= 8 && wd % 8 == 0 && wd < w, "w={w} d={d} wd={wd}");
            }
        }
        // 128 admits exactly denominator 16 (129..1023/d never lands
        // on a multiple of 8 for 9..=15).
        assert_eq!(candidate_denoms(128), vec![16]);
        assert!(candidate_denoms(96).contains(&12));
    }

    /// The downscaler preserves flat fields exactly and keeps every
    /// sample inside the source range.
    #[test]
    fn downscaler_flat_and_range() {
        use crate::encoder::yuv_frame::ChromaFormat;
        let f = YuvFrame::filled(128, 16, 8, ChromaFormat::Yuv420, 173);
        let d = downscale_width(&f, 64);
        assert_eq!(d.width, 64);
        assert!(d.y.iter().all(|&s| s == 173));
        assert!(d.u.iter().all(|&s| s == 173));
        let mut g = YuvFrame::filled(128, 16, 8, ChromaFormat::Yuv420, 0);
        for (i, s) in g.y.iter_mut().enumerate() {
            *s = ((i * 7) % 256) as u16;
        }
        let dg = downscale_width(&g, 64);
        assert!(dg.y.iter().all(|&s| s <= 255));
    }
}
