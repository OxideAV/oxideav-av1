//! r441 — the §5.9.30 FILM-GRAIN election helpers.
//!
//! The tool's contract is unusual among the encoder's elections: the
//! §7.18.3 synthesis is OUTPUT-ONLY (the §7.20 reference store keeps
//! the un-grained reconstruction), and the synthesized noise field is
//! BY DESIGN not sample-matched to the source's — its value is
//! statistical. The election therefore runs a documented
//! "perceptually-neutral rate" objective (see
//! [`crate::encoder::inter_frame`]'s GOP wrapper): the grain arm's
//! distortion is measured as STRUCTURE fidelity (its pre-grain
//! reconstruction against the DENOISED source) plus a noise-AMPLITUDE
//! mismatch penalty (per frame, `luma_samples ×
//! (σ_source − σ_synthesized)²`), while the plain arm keeps the
//! ordinary source-matched SSE. Rate is exact realized bytes on both
//! arms.
//!
//! Everything estimation-side here (the denoiser, the noise probe,
//! the parameter fit) is an encoder-side choice the spec leaves open;
//! the SYNTHESIS mirror is the decoder's own §7.18.3 driver
//! (`crate::film_grain`), so the grained output equals the decoder's
//! byte for byte.

use crate::encoder::yuv_frame::YuvFrame;
use crate::film_grain::film_grain_synthesis;
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::FilmGrainParams;

/// Per-intensity noise profile of one frame's luma plane: `σ × 16`
/// (in 8-bit-normalized sample units) globally and per intensity bin.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NoiseEstimate {
    /// Global luma noise sigma × 16 (8-bit scale).
    pub sigma16: u64,
    /// Per-bin sigma × 16 over eight equal intensity bins (8-bit
    /// scale; bins with too few samples carry the global sigma).
    pub bin_sigma16: [u64; 8],
}

/// 3×3 binomial (`[1,2,1]⊗[1,2,1]/16`) denoiser with edge
/// replication, all planes — the encoder-side noise/structure split
/// behind the probe, the estimate and the grain arm's coded source.
#[must_use]
pub(crate) fn denoise_frame(input: &YuvFrame) -> YuvFrame {
    let mut out = input.clone();
    denoise_plane(
        &input.y,
        &mut out.y,
        input.width as usize,
        input.height as usize,
    );
    if !input.u.is_empty() {
        let (cw, ch) = (
            input.chroma_width() as usize,
            input.chroma_height() as usize,
        );
        denoise_plane(&input.u, &mut out.u, cw, ch);
        denoise_plane(&input.v, &mut out.v, cw, ch);
    }
    out
}

fn denoise_plane(src: &[u16], dst: &mut [u16], w: usize, h: usize) {
    if w == 0 || h == 0 {
        return;
    }
    let at = |r: usize, c: usize| -> u32 { u32::from(src[r * w + c]) };
    for r in 0..h {
        let rm = r.saturating_sub(1);
        let rp = (r + 1).min(h - 1);
        for c in 0..w {
            let cm = c.saturating_sub(1);
            let cp = (c + 1).min(w - 1);
            let sum = at(rm, cm)
                + 2 * at(rm, c)
                + at(rm, cp)
                + 2 * at(r, cm)
                + 4 * at(r, c)
                + 2 * at(r, cp)
                + at(rp, cm)
                + 2 * at(rp, c)
                + at(rp, cp);
            dst[r * w + c] = ((sum + 8) >> 4) as u16;
        }
    }
}

/// Luma noise profile of `input` against its denoised twin, in
/// 8-bit-normalized units.
#[must_use]
pub(crate) fn noise_estimate(input: &YuvFrame, denoised: &YuvFrame) -> NoiseEstimate {
    let n = input.y.len().max(1) as u64;
    let shift = u32::from(input.bit_depth - 8);
    let mut sq_sum = 0u64;
    let mut bin_sq = [0u64; 8];
    let mut bin_n = [0u64; 8];
    for (&a, &b) in input.y.iter().zip(&denoised.y) {
        let d = (i64::from(a) - i64::from(b)) >> shift;
        let sq = (d * d) as u64;
        sq_sum += sq;
        let bin = ((usize::from(a) >> shift) >> 5).min(7);
        bin_sq[bin] += sq;
        bin_n[bin] += 1;
    }
    let sigma16 = |sq: u64, count: u64| -> u64 {
        if count == 0 {
            return 0;
        }
        ((sq as f64 / count as f64).sqrt() * 16.0).round() as u64
    };
    let global = sigma16(sq_sum, n);
    let min_count = (n / 100).max(64);
    let bin_sigma16 = core::array::from_fn(|i| {
        if bin_n[i] >= min_count {
            sigma16(bin_sq[i], bin_n[i])
        } else {
            global
        }
    });
    NoiseEstimate {
        sigma16: global,
        bin_sigma16,
    }
}

/// The film-grain arm's CONTENT gate. Fires only when the luma
/// residual against the denoised twin (a) carries real energy
/// (σ ≥ 1 in 8-bit units), (b) is spatially WHITE (lag-1 horizontal
/// autocorrelation below 0.4 — texture and edges survive a high-pass
/// correlated, noise does not), and (c) DECORRELATES between the
/// first two frames at co-located samples (static texture repeats,
/// noise re-rolls). All three are encoder-side heuristics; the
/// election behind them still demands a joint-objective win.
#[must_use]
pub(crate) fn film_grain_probe(frames: &[YuvFrame], denoised: &[YuvFrame]) -> bool {
    if frames.len() < 2 {
        return false;
    }
    let est = noise_estimate(&frames[0], &denoised[0]);
    if est.sigma16 < 16 {
        return false;
    }
    let w = frames[0].width as usize;
    let h = frames[0].height as usize;
    let noise = |k: usize| -> Vec<f64> {
        frames[k]
            .y
            .iter()
            .zip(&denoised[k].y)
            .map(|(&a, &b)| f64::from(a) - f64::from(b))
            .collect()
    };
    let n0 = noise(0);
    let n1 = noise(1);
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let e0 = dot(&n0, &n0);
    if e0 <= 0.0 {
        return false;
    }
    // Spatial lag-1 horizontal autocorrelation of N_0.
    let mut lag = 0.0f64;
    for r in 0..h {
        for c in 0..w - 1 {
            lag += n0[r * w + c] * n0[r * w + c + 1];
        }
    }
    if (lag / e0).abs() >= 0.4 {
        return false;
    }
    // Temporal co-located correlation between N_0 and N_1.
    let e1 = dot(&n1, &n1);
    if e1 <= 0.0 {
        return false;
    }
    let cross = dot(&n0, &n1);
    (cross / (e0 * e1).sqrt()).abs() < 0.4
}

/// Deterministic per-frame §5.9.30 `grain_seed` schedule — a fixed
/// odd-multiplier hash of the display position, shared by the header
/// writer and the output-synthesis mirror.
#[must_use]
pub(crate) fn grain_seed_for(order_hint: u32) -> u16 {
    ((order_hint.wrapping_mul(0x9e37).wrapping_add(0x51ed)) & 0xffff) as u16
}

/// Build the §5.9.30 parameter set for a measured noise profile:
/// white grain (`ar_coeff_lag = 0`), luma-only scaling (`num_cb/cr =
/// 0`), eight piecewise-linear scaling points at the intensity-bin
/// centres, each calibrated through the decoder's OWN synthesis (one
/// flat-patch run maps `point_scaling` to realized σ, so the
/// requested amplitudes land in §7.18.3 output units).
#[must_use]
pub(crate) fn build_grain_params(est: &NoiseEstimate, bit_depth: u8) -> FilmGrainParams {
    let mut fg = FilmGrainParams::reset();
    fg.apply_grain = true;
    fg.update_grain = true;
    fg.grain_scaling = 8;
    fg.ar_coeff_lag = 0;
    fg.ar_coeff_shift = 6;
    fg.grain_scale_shift = 0;
    fg.overlap_flag = true;
    let k16 = calibration_sigma16(bit_depth);
    fg.num_y_points = 8;
    for bin in 0..8usize {
        fg.point_y_value[bin] = (bin as u8) * 32 + 16;
        // point_scaling = σ_bin / σ_per_scaling_unit, i.e.
        // σ_bin16 * 100 / k16 for the scaling-100 calibration probe.
        let scaled = (est.bin_sigma16[bin] * 100).div_ceil(k16.max(1));
        fg.point_y_scaling[bin] = scaled.min(255) as u8;
    }
    fg
}

/// One-off calibration: σ × 16 realized by `point_scaling = 100` on a
/// flat mid-gray patch through the §7.18.3 synthesis at this bit
/// depth (white grain, `GrainScaling = 8`). Deterministic — the
/// grain generator is seeded LFSR state.
fn calibration_sigma16(bit_depth: u8) -> u64 {
    let mut fg = FilmGrainParams::reset();
    fg.apply_grain = true;
    fg.update_grain = true;
    fg.grain_seed = 7391;
    fg.grain_scaling = 8;
    fg.ar_coeff_lag = 0;
    fg.ar_coeff_shift = 6;
    fg.grain_scale_shift = 0;
    fg.overlap_flag = true;
    fg.num_y_points = 2;
    fg.point_y_value = {
        let mut v = [0u8; crate::uncompressed_header_tail::MAX_NUM_Y_POINTS];
        v[0] = 0;
        v[1] = 255;
        v
    };
    fg.point_y_scaling = {
        let mut v = [0u8; crate::uncompressed_header_tail::MAX_NUM_Y_POINTS];
        v[0] = 100;
        v[1] = 100;
        v
    };
    let (w, h) = (64usize, 64usize);
    let mid = 128u16 << (bit_depth - 8);
    let flat = vec![mid; w * h];
    let mut grained = flat.iter().map(|&s| i32::from(s)).collect::<Vec<i32>>();
    {
        let mut planes = [PlaneBuffer {
            rows: h as u32,
            cols: w as u32,
            samples: &mut grained,
        }];
        film_grain_synthesis(&fg, bit_depth, 1, 1, 1, 2, &mut planes);
    }
    let shift = u32::from(bit_depth - 8);
    let mut sq = 0u64;
    for (&g, &s) in grained.iter().zip(&flat) {
        let d = (i64::from(g) - i64::from(s)) >> shift;
        sq += (d * d) as u64;
    }
    (((sq as f64 / (w * h) as f64).sqrt()) * 16.0)
        .round()
        .max(1.0) as u64
}

/// Apply the §7.18.3 synthesis to one frame's reconstruction through
/// the decoder's own driver — the returned planes equal the decoder's
/// output byte for byte.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_grain_to_recon(
    fg: &FilmGrainParams,
    y: &[u16],
    u: &[u16],
    v: &[u16],
    width: u32,
    height: u32,
    bit_depth: u8,
    ssx: u8,
    ssy: u8,
    num_planes: u8,
    matrix_coefficients: u8,
) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let dims = |plane: usize| -> (u32, u32) {
        if plane == 0 {
            (width, height)
        } else {
            (
                (width + u32::from(ssx)) >> ssx,
                (height + u32::from(ssy)) >> ssy,
            )
        }
    };
    let srcs: [&[u16]; 3] = [y, u, v];
    let mut owned: Vec<Vec<i32>> = Vec::with_capacity(usize::from(num_planes));
    for (plane, src) in srcs.iter().enumerate().take(usize::from(num_planes)) {
        let (pw, ph) = dims(plane);
        debug_assert_eq!(src.len(), (pw * ph) as usize);
        owned.push(src.iter().map(|&s| i32::from(s)).collect());
    }
    {
        let mut bufs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(usize::from(num_planes));
        for (plane, buf) in owned.iter_mut().enumerate() {
            let (pw, ph) = dims(plane);
            bufs.push(PlaneBuffer {
                rows: ph,
                cols: pw,
                samples: buf,
            });
        }
        film_grain_synthesis(
            fg,
            bit_depth,
            num_planes,
            ssx,
            ssy,
            matrix_coefficients,
            &mut bufs,
        );
    }
    let narrow = |v: Vec<i32>| -> Vec<u16> { v.into_iter().map(|s| s.max(0) as u16).collect() };
    let mut it = owned.into_iter();
    let gy = narrow(it.next().unwrap_or_default());
    let gu = it.next().map(narrow).unwrap_or_default();
    let gv = it.next().map(narrow).unwrap_or_default();
    (gy, gu, gv)
}

/// σ × 16 (8-bit-normalized) of the luma difference between two
/// equal-length planes — the neutrality term's amplitude measure.
#[must_use]
pub(crate) fn luma_sigma16(a: &[u16], b: &[u16], bit_depth: u8) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0;
    }
    let shift = u32::from(bit_depth - 8);
    let mut sq = 0u64;
    for (&x, &y) in a.iter().zip(b) {
        let d = (i64::from(x) - i64::from(y)) >> shift;
        sq += (d * d) as u64;
    }
    (((sq as f64 / a.len() as f64).sqrt()) * 16.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::yuv_frame::ChromaFormat;

    /// The denoiser preserves flat fields exactly.
    #[test]
    fn denoiser_flat_identity() {
        let f = YuvFrame::filled(32, 16, 8, ChromaFormat::Yuv420, 173);
        let d = denoise_frame(&f);
        assert_eq!(d.y, f.y);
        assert_eq!(d.u, f.u);
    }

    /// The calibration mapping is monotone: doubling the requested
    /// point scaling roughly doubles the realized sigma.
    #[test]
    fn calibration_is_positive_and_scaling_monotone() {
        let k = calibration_sigma16(8);
        assert!(k > 0, "flat-patch synthesis must realize noise");
        let est_lo = NoiseEstimate {
            sigma16: 16,
            bin_sigma16: [16; 8],
        };
        let est_hi = NoiseEstimate {
            sigma16: 32,
            bin_sigma16: [32; 8],
        };
        let lo = build_grain_params(&est_lo, 8);
        let hi = build_grain_params(&est_hi, 8);
        assert!(lo.point_y_scaling[0] > 0);
        assert!(hi.point_y_scaling[0] > lo.point_y_scaling[0]);
    }

    /// The probe rejects clean content and accepts white noise.
    #[test]
    fn probe_separates_noise_from_clean() {
        let w = 96u32;
        let h = 80u32;
        let clean: Vec<YuvFrame> = (0..2)
            .map(|_| YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 120))
            .collect();
        let clean_dn: Vec<YuvFrame> = clean.iter().map(denoise_frame).collect();
        assert!(!film_grain_probe(&clean, &clean_dn));

        // Deterministic white-ish noise via a simple LCG.
        let mut state = 0x1234_5678u32;
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 24) & 15) as i32 - 8
        };
        let noisy: Vec<YuvFrame> = (0..2)
            .map(|_| {
                let mut f = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 120);
                for s in f.y.iter_mut() {
                    *s = (i32::from(*s) + rnd()).clamp(0, 255) as u16;
                }
                f
            })
            .collect();
        let noisy_dn: Vec<YuvFrame> = noisy.iter().map(denoise_frame).collect();
        assert!(film_grain_probe(&noisy, &noisy_dn));
    }
}
