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

/// Per-intensity noise profile of one frame: `σ × 16` (in
/// 8-bit-normalized sample units) globally and per intensity bin,
/// for luma and (r444) both chroma planes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NoiseEstimate {
    /// Global luma noise sigma × 16 (8-bit scale).
    pub sigma16: u64,
    /// Per-bin sigma × 16 over eight equal intensity bins (8-bit
    /// scale; bins with too few samples carry the global sigma).
    pub bin_sigma16: [u64; 8],
    /// r444 — global Cb noise sigma × 16 (0 without chroma).
    pub cb_sigma16: u64,
    /// r444 — per-bin Cb sigma × 16 over FOUR chroma-intensity bins.
    pub cb_bin_sigma16: [u64; 4],
    /// r444 — global Cr noise sigma × 16.
    pub cr_sigma16: u64,
    /// r444 — per-bin Cr sigma × 16.
    pub cr_bin_sigma16: [u64; 4],
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

/// Noise profile of `input` against its denoised twin, in
/// 8-bit-normalized units (luma: eight intensity bins; r444 — each
/// chroma plane: a global sigma plus four intensity bins).
#[must_use]
pub(crate) fn noise_estimate(input: &YuvFrame, denoised: &YuvFrame) -> NoiseEstimate {
    let shift = u32::from(input.bit_depth - 8);
    let sigma16 = |sq: u64, count: u64| -> u64 {
        if count == 0 {
            return 0;
        }
        ((sq as f64 / count as f64).sqrt() * 16.0).round() as u64
    };
    let n = input.y.len().max(1) as u64;
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
    let global = sigma16(sq_sum, n);
    let min_count = (n / 100).max(64);
    let bin_sigma16 = core::array::from_fn(|i| {
        if bin_n[i] >= min_count {
            sigma16(bin_sq[i], bin_n[i])
        } else {
            global
        }
    });
    // r444 — chroma profiles over FOUR 64-wide intensity bins of the
    // chroma sample itself (the §7.18.3 blend's neutral `merged`
    // index — see [`build_grain_params`]'s identity mults).
    let chroma_profile = |src: &[u16], den: &[u16]| -> (u64, [u64; 4]) {
        if src.is_empty() {
            return (0, [0; 4]);
        }
        let cn = src.len() as u64;
        let mut sq_sum = 0u64;
        let mut bsq = [0u64; 4];
        let mut bn = [0u64; 4];
        for (&a, &b) in src.iter().zip(den) {
            let d = (i64::from(a) - i64::from(b)) >> shift;
            let sq = (d * d) as u64;
            sq_sum += sq;
            let bin = ((usize::from(a) >> shift) >> 6).min(3);
            bsq[bin] += sq;
            bn[bin] += 1;
        }
        let g = sigma16(sq_sum, cn);
        let minc = (cn / 100).max(32);
        (
            g,
            core::array::from_fn(|i| {
                if bn[i] >= minc {
                    sigma16(bsq[i], bn[i])
                } else {
                    g
                }
            }),
        )
    };
    let (cb_sigma16, cb_bin_sigma16) = chroma_profile(&input.u, &denoised.u);
    let (cr_sigma16, cr_bin_sigma16) = chroma_profile(&input.v, &denoised.v);
    NoiseEstimate {
        sigma16: global,
        bin_sigma16,
        cb_sigma16,
        cb_bin_sigma16,
        cr_sigma16,
        cr_bin_sigma16,
    }
}

// ---------------------------------------------------------------------
// r444 — §5.9.30 auto-regressive coefficient fitting.
// ---------------------------------------------------------------------

/// The fitted §5.9.30 auto-regressive shape: `lag` (0 = white) plus
/// the quantised coefficient lists in the §7.18.3 position order
/// (`ar_coeff_shift = 6` fixed-point, value range `[-128, 127]`;
/// stored on the wire as `+128`). `cb` / `cr` carry `numPosLuma + 1`
/// entries — the same-plane neighbourhood taps followed by the
/// LUMA-CORRELATION tap the §7.18.3 chroma filter applies at the
/// centre position.
#[derive(Debug, Clone)]
pub(crate) struct ArFit {
    pub lag: u8,
    pub y: Vec<i32>,
    pub cb: Vec<i32>,
    pub cr: Vec<i32>,
}

/// The §7.18.3 causal neighbourhood, in coefficient order: `deltaRow
/// ∈ [-lag, 0]`, `deltaCol ∈ [-lag, lag]`, stopping at the centre.
fn ar_positions(lag: usize) -> Vec<(i32, i32)> {
    let l = lag as i32;
    let mut out = Vec::with_capacity(2 * lag * (lag + 1));
    for dr in -l..=0 {
        for dc in -l..=l {
            if dr == 0 && dc == 0 {
                break;
            }
            out.push((dr, dc));
        }
    }
    out
}

/// Solve the `p × p` normal equations by Gaussian elimination with
/// partial pivoting; `None` on a (near-)singular system.
#[allow(clippy::needless_range_loop)]
fn solve_normal(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let p = b.len();
    for i in 0..p {
        let mut piv = i;
        for r in i + 1..p {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        if a[i][i].abs() < 1e-9 {
            return None;
        }
        for r in i + 1..p {
            let f = a[r][i] / a[i][i];
            for c in i..p {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut v = b[i];
        for j in i + 1..p {
            v -= a[i][j] * x[j];
        }
        x[i] = v / a[i][i];
    }
    Some(x)
}

/// Least-squares fit of the lag-`lag` causal AR model over `noise`
/// (row-major `w × h`). Returns `(coefficients, prediction gain)` —
/// the gain is the fraction of the target variance the model
/// explains — or `None` on a degenerate system.
#[allow(clippy::needless_range_loop)]
fn ls_fit_ar(noise: &[f64], w: usize, h: usize, lag: usize) -> Option<(Vec<f64>, f64)> {
    let pos = ar_positions(lag);
    let p = pos.len();
    if w <= 2 * lag || h <= lag {
        return None;
    }
    let mut ata = vec![vec![0.0f64; p]; p];
    let mut atb = vec![0.0f64; p];
    let mut var = 0.0f64;
    let mut count = 0u64;
    for y in lag..h {
        for x in lag..w - lag {
            let t = noise[y * w + x];
            var += t * t;
            count += 1;
            let row: Vec<f64> = pos
                .iter()
                .map(|&(dr, dc)| noise[(y as i32 + dr) as usize * w + (x as i32 + dc) as usize])
                .collect();
            for i in 0..p {
                for j in i..p {
                    ata[i][j] += row[i] * row[j];
                }
                atb[i] += row[i] * t;
            }
        }
    }
    if count == 0 || var <= 0.0 {
        return None;
    }
    for i in 0..p {
        for j in 0..i {
            ata[i][j] = ata[j][i];
        }
    }
    let sol = solve_normal(ata, atb.clone())?;
    // Explained energy: coeffsᵀ·atb (LS identity), as a fraction.
    let explained: f64 = sol.iter().zip(&atb).map(|(c, b)| c * b).sum();
    Some((sol, (explained / var).clamp(-1.0, 1.0)))
}

/// Quantise an AR coefficient to the wire's `ar_coeff_shift = 6`
/// fixed point.
fn quant_ar(c: f64) -> i32 {
    ((c * 64.0).round() as i32).clamp(-128, 127)
}

/// r444 — fit the §5.9.30 AR shape at ONE requested lag from the
/// first frame's noise residual (encoder-side free choice; the wire
/// carries only the quantised taps). Luma: least squares over the
/// §7.18.3 causal neighbourhood; chroma: the same lag's same-plane
/// taps plus the luma-correlation tap, fitted jointly at the chroma
/// extent (at lag 0 only the correlation tap remains — the §7.18.3
/// chroma filter still applies it). Returns `None` when the luma fit
/// degenerates or every luma tap quantises to zero (the white shape
/// already covers it); the deeper-vs-shallower choice belongs to the
/// CALLER's score settlement — each extra lag ring costs real
/// parameter bytes on every frame header.
#[must_use]
#[allow(clippy::needless_range_loop)]
pub(crate) fn fit_ar_lag(input: &YuvFrame, denoised: &YuvFrame, lag: usize) -> Option<ArFit> {
    let (w, h) = (input.width as usize, input.height as usize);
    let ny: Vec<f64> = input
        .y
        .iter()
        .zip(&denoised.y)
        .map(|(&a, &b)| f64::from(a) - f64::from(b))
        .collect();
    if lag == 0 {
        return None;
    }
    let (coeffs, _gain) = ls_fit_ar(&ny, w, h, lag)?;
    let y: Vec<i32> = coeffs.iter().map(|&c| quant_ar(c)).collect();
    if y.iter().all(|&c| c == 0) {
        return None;
    }

    // Chroma: joint LS over the same-plane neighbourhood + the
    // co-located (subsample-averaged) luma noise, mirroring the
    // §7.18.3 chroma filter's centre tap.
    let (ssx, ssy) = input.format.subsampling();
    let fit_chroma = |src: &[u16], den: &[u16]| -> Vec<i32> {
        let num_pos = 2 * lag * (lag + 1);
        if src.is_empty() {
            return vec![0; num_pos + 1];
        }
        let cw = input.chroma_width() as usize;
        let ch = input.chroma_height() as usize;
        let nc: Vec<f64> = src
            .iter()
            .zip(den)
            .map(|(&a, &b)| f64::from(a) - f64::from(b))
            .collect();
        let luma_at = |cy: usize, cx: usize| -> f64 {
            let (ly, lx) = (cy << ssy, cx << ssx);
            let mut acc = 0.0;
            let mut cnt = 0.0;
            for i in 0..=usize::from(ssy) {
                for j in 0..=usize::from(ssx) {
                    let (yy, xx) = ((ly + i).min(h - 1), (lx + j).min(w - 1));
                    acc += ny[yy * w + xx];
                    cnt += 1.0;
                }
            }
            acc / cnt
        };
        let pos = ar_positions(lag);
        let p = pos.len() + 1; // + the luma-correlation tap
        if cw <= 2 * lag || ch <= lag {
            return vec![0; num_pos + 1];
        }
        let mut ata = vec![vec![0.0f64; p]; p];
        let mut atb = vec![0.0f64; p];
        for y0 in lag..ch {
            for x0 in lag..cw - lag {
                let t = nc[y0 * cw + x0];
                let mut row: Vec<f64> = pos
                    .iter()
                    .map(|&(dr, dc)| nc[(y0 as i32 + dr) as usize * cw + (x0 as i32 + dc) as usize])
                    .collect();
                row.push(luma_at(y0, x0));
                for i in 0..p {
                    for j in i..p {
                        ata[i][j] += row[i] * row[j];
                    }
                    atb[i] += row[i] * t;
                }
            }
        }
        for i in 0..p {
            for j in 0..i {
                ata[i][j] = ata[j][i];
            }
        }
        match solve_normal(ata, atb) {
            Some(sol) => sol.iter().map(|&c| quant_ar(c)).collect(),
            None => vec![0; num_pos + 1],
        }
    };
    let cb = fit_chroma(&input.u, &denoised.u);
    let cr = fit_chroma(&input.v, &denoised.v);
    Some(ArFit {
        lag: lag as u8,
        y,
        cb,
        cr,
    })
}

/// r444 — the lag-0 chroma shape: no luma taps, only the fitted
/// LUMA-CORRELATION tap per chroma plane (the §7.18.3 chroma filter
/// applies it even at `ar_coeff_lag = 0`). For the chroma-points
/// candidate on white luma grain.
#[must_use]
pub(crate) fn fit_chroma_corr(input: &YuvFrame, denoised: &YuvFrame) -> ArFit {
    let (w, h) = (input.width as usize, input.height as usize);
    let ny: Vec<f64> = input
        .y
        .iter()
        .zip(&denoised.y)
        .map(|(&a, &b)| f64::from(a) - f64::from(b))
        .collect();
    let (ssx, ssy) = input.format.subsampling();
    let corr = |src: &[u16], den: &[u16]| -> i32 {
        if src.is_empty() {
            return 0;
        }
        let cw = input.chroma_width() as usize;
        let ch = input.chroma_height() as usize;
        let mut s_ll = 0.0f64;
        let mut s_lc = 0.0f64;
        for cy in 0..ch {
            for cx in 0..cw {
                let (ly, lx) = (cy << ssy, cx << ssx);
                let mut acc = 0.0;
                let mut cnt = 0.0;
                for i in 0..=usize::from(ssy) {
                    for j in 0..=usize::from(ssx) {
                        let (yy, xx) = ((ly + i).min(h - 1), (lx + j).min(w - 1));
                        acc += ny[yy * w + xx];
                        cnt += 1.0;
                    }
                }
                let l = acc / cnt;
                let c = f64::from(src[cy * cw + cx]) - f64::from(den[cy * cw + cx]);
                s_ll += l * l;
                s_lc += l * c;
            }
        }
        if s_ll <= 1e-9 {
            0
        } else {
            quant_ar(s_lc / s_ll)
        }
    };
    ArFit {
        lag: 0,
        y: Vec::new(),
        cb: vec![corr(&input.u, &denoised.u)],
        cr: vec![corr(&input.v, &denoised.v)],
    }
}

/// r444 — the per-plane CHROMA noise gate: real residual energy
/// (σ16 ≥ 8), spatially modelable (lag-1 |ρ| < 0.75) and temporally
/// decorrelated at co-located samples between the first two frames
/// (|ρ| < 0.4 — moving chroma texture repeats, chroma noise
/// re-rolls). Mirrors the luma probe's three gates at the chroma
/// extent.
#[must_use]
pub(crate) fn chroma_noise_gate(
    frames: &[YuvFrame],
    denoised: &[YuvFrame],
    plane: usize,
    sigma16: u64,
) -> bool {
    if sigma16 < 8 || frames.len() < 2 {
        return false;
    }
    let cw = frames[0].chroma_width() as usize;
    let ch = frames[0].chroma_height() as usize;
    fn pl(f: &YuvFrame, plane: usize) -> &[u16] {
        if plane == 1 {
            &f.u
        } else {
            &f.v
        }
    }
    if pl(&frames[0], plane).is_empty() {
        return false;
    }
    if lag1_h_rho(pl(&frames[0], plane), pl(&denoised[0], plane), cw, ch).abs() >= 0.75 {
        return false;
    }
    let n = |k: usize| -> Vec<f64> {
        pl(&frames[k], plane)
            .iter()
            .zip(pl(&denoised[k], plane))
            .map(|(&a, &b)| f64::from(a) - f64::from(b))
            .collect()
    };
    let (n0, n1) = (n(0), n(1));
    let dot = |a: &[f64], b: &[f64]| -> f64 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
    let (e0, e1) = (dot(&n0, &n0), dot(&n1, &n1));
    if e0 <= 0.0 || e1 <= 0.0 {
        return false;
    }
    (dot(&n0, &n1) / (e0 * e1).sqrt()).abs() < 0.4
}

/// Lag-1 HORIZONTAL autocorrelation of the residual between two
/// equal-shape planes — the spatial-structure measure the r444
/// correlation-match term scores (and the probe gates on).
#[must_use]
pub(crate) fn lag1_h_rho(a: &[u16], b: &[u16], w: usize, h: usize) -> f64 {
    if w < 2 || h == 0 {
        return 0.0;
    }
    let n: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| f64::from(x) - f64::from(y))
        .collect();
    let mut e = 0.0f64;
    let mut lagsum = 0.0f64;
    for r in 0..h {
        for c in 0..w {
            e += n[r * w + c] * n[r * w + c];
            if c + 1 < w {
                lagsum += n[r * w + c] * n[r * w + c + 1];
            }
        }
    }
    if e <= 0.0 {
        0.0
    } else {
        lagsum / e
    }
}

/// The film-grain arm's CONTENT gate. Fires only when the luma
/// residual against the denoised twin (a) carries real energy
/// (σ ≥ 1 in 8-bit units), (b) is spatially MODELABLE — lag-1
/// horizontal autocorrelation below 0.75 (r444: the §5.9.30 AR taps
/// model moderately correlated grain, so the r441 whiteness bound of
/// 0.4 relaxes; heavily correlated residual is structure, not noise)
/// — and (c) DECORRELATES between the first two frames at co-located
/// samples (static texture repeats, noise re-rolls — the texture
/// killer stays strict). All three are encoder-side heuristics; the
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
    // Spatial lag-1 horizontal autocorrelation of N_0 (r444: the
    // AR-modelable bound).
    let mut lag = 0.0f64;
    for r in 0..h {
        for c in 0..w - 1 {
            lag += n0[r * w + c] * n0[r * w + c + 1];
        }
    }
    if (lag / e0).abs() >= 0.75 {
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

/// Build the §5.9.30 parameter set for a measured noise profile.
/// The r441 shape — white grain (`ar_coeff_lag = 0`), luma-only
/// scaling — comes from `ar = None, chroma_cb = chroma_cr = false`
/// and is bit-identical to the r441 builder. r444 grows the two open
/// axes:
///
/// * **AR taps** (`ar: Some`) — the fitted `ar_coeff_lag` +
///   `ar_coeffs_{y,cb,cr}` land on the wire, and the scaling-point
///   calibration probe runs WITH those taps (the AR filter changes
///   the grain template's energy, so the σ-per-scaling-unit mapping
///   must include it).
/// * **Chroma points** (`chroma_cb` / `chroma_cr`) — four
///   piecewise-linear points per elected plane at the chroma
///   intensity-bin centres, each calibrated through a chroma probe
///   run; the blend index rides the IDENTITY mults (`cb_mult = 192`,
///   `cb_luma_mult = 128`, `cb_offset = 256` — §7.18.3 then derives
///   `merged = orig`), so the bins index the chroma sample directly.
///
/// Every probe is the decoder's OWN §7.18.3 synthesis on a flat
/// mid-gray patch — requested amplitudes land in output units.
#[must_use]
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)]
pub(crate) fn build_grain_params(
    est: &NoiseEstimate,
    bit_depth: u8,
    ssx: u8,
    ssy: u8,
    matrix_coefficients: u8,
    ar: Option<&ArFit>,
    chroma_cb: bool,
    chroma_cr: bool,
) -> FilmGrainParams {
    let mut fg = FilmGrainParams::reset();
    fg.apply_grain = true;
    fg.update_grain = true;
    fg.grain_scaling = 8;
    fg.ar_coeff_lag = 0;
    fg.ar_coeff_shift = 6;
    fg.grain_scale_shift = 0;
    fg.overlap_flag = true;
    if let Some(fit) = ar {
        fg.ar_coeff_lag = fit.lag;
        for (i, &c) in fit.y.iter().enumerate() {
            fg.ar_coeffs_y_plus_128[i] = (c + 128) as u8;
        }
        if chroma_cb || chroma_cr {
            for (i, &c) in fit.cb.iter().enumerate() {
                fg.ar_coeffs_cb_plus_128[i] = (c + 128) as u8;
            }
            for (i, &c) in fit.cr.iter().enumerate() {
                fg.ar_coeffs_cr_plus_128[i] = (c + 128) as u8;
            }
        }
    }
    let k16 = calibration_sigma16(&fg, bit_depth, ssx, ssy, matrix_coefficients, 0);
    fg.num_y_points = 8;
    for bin in 0..8usize {
        fg.point_y_value[bin] = (bin as u8) * 32 + 16;
        // point_scaling = σ_bin / σ_per_scaling_unit, i.e.
        // σ_bin16 * 100 / k16 for the scaling-100 calibration probe.
        let scaled = (est.bin_sigma16[bin] * 100).div_ceil(k16.max(1));
        fg.point_y_scaling[bin] = scaled.min(255) as u8;
    }
    // r444 — chroma scaling points (identity index mults; see the
    // doc comment). The per-plane calibration probes run AFTER the
    // luma points land — the chroma grain's luma-correlation tap
    // needs `num_y_points > 0`, exactly as on the final header.
    for (armed, plane) in [(chroma_cb, 1usize), (chroma_cr, 2usize)] {
        if !armed {
            continue;
        }
        let kc16 = calibration_sigma16(&fg, bit_depth, ssx, ssy, matrix_coefficients, plane);
        let bins = if plane == 1 {
            &est.cb_bin_sigma16
        } else {
            &est.cr_bin_sigma16
        };
        let mut values = [0u8; crate::uncompressed_header_tail::MAX_NUM_CHROMA_POINTS];
        let mut scalings = [0u8; crate::uncompressed_header_tail::MAX_NUM_CHROMA_POINTS];
        for bin in 0..4usize {
            values[bin] = (bin as u8) * 64 + 32;
            scalings[bin] = ((bins[bin] * 100).div_ceil(kc16.max(1))).min(255) as u8;
        }
        if plane == 1 {
            fg.num_cb_points = 4;
            fg.point_cb_value = values;
            fg.point_cb_scaling = scalings;
            fg.cb_mult = 192;
            fg.cb_luma_mult = 128;
            fg.cb_offset = 256;
        } else {
            fg.num_cr_points = 4;
            fg.point_cr_value = values;
            fg.point_cr_scaling = scalings;
            fg.cr_mult = 192;
            fg.cr_luma_mult = 128;
            fg.cr_offset = 256;
        }
    }
    fg
}

/// One-off calibration: σ × 16 realized on `plane` by
/// `point_scaling = 100` on a flat mid-gray patch through the
/// §7.18.3 synthesis at this bit depth, with `template`'s AR shape
/// live. Deterministic — the grain generator is seeded LFSR state.
/// `plane = 0` probes luma alone (single-plane synthesis, exactly
/// the r441 probe shape when the template is white); chroma probes
/// synthesize all three planes so the luma-correlation tap
/// contributes as it will on the real frames.
fn calibration_sigma16(
    template: &FilmGrainParams,
    bit_depth: u8,
    ssx: u8,
    ssy: u8,
    matrix_coefficients: u8,
    plane: usize,
) -> u64 {
    let mut fg = template.clone();
    fg.apply_grain = true;
    fg.update_grain = true;
    fg.grain_seed = 7391;
    let flat100 = |n: usize| -> ([u8; 16], [u8; 16]) {
        let mut v = [0u8; 16];
        let mut s = [0u8; 16];
        v[0] = 0;
        v[n - 1] = 255;
        s[0] = 100;
        s[n - 1] = 100;
        (v, s)
    };
    if plane == 0 {
        fg.num_y_points = 2;
        let (v, s) = flat100(2);
        fg.point_y_value[..2].copy_from_slice(&v[..2]);
        fg.point_y_scaling[..2].copy_from_slice(&s[..2]);
        fg.num_cb_points = 0;
        fg.num_cr_points = 0;
    } else {
        // Keep the template's (already-calibrated) luma points; probe
        // the chroma plane at a flat 100 LUT.
        let mut v = [0u8; crate::uncompressed_header_tail::MAX_NUM_CHROMA_POINTS];
        let mut s = [0u8; crate::uncompressed_header_tail::MAX_NUM_CHROMA_POINTS];
        v[1] = 255;
        s[0] = 100;
        s[1] = 100;
        if plane == 1 {
            fg.num_cb_points = 2;
            fg.point_cb_value = v;
            fg.point_cb_scaling = s;
            fg.cb_mult = 192;
            fg.cb_luma_mult = 128;
            fg.cb_offset = 256;
            fg.num_cr_points = 0;
        } else {
            fg.num_cr_points = 2;
            fg.point_cr_value = v;
            fg.point_cr_scaling = s;
            fg.cr_mult = 192;
            fg.cr_luma_mult = 128;
            fg.cr_offset = 256;
            fg.num_cb_points = 0;
        }
    }
    let (w, h) = (64usize, 64usize);
    let mid = 128u16 << (bit_depth - 8);
    if plane == 0 {
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
        return probe_sigma16(&grained, &flat, bit_depth);
    }
    let (cw, ch) = ((w + usize::from(ssx)) >> ssx, (h + usize::from(ssy)) >> ssy);
    let flat_y = vec![mid; w * h];
    let flat_c = vec![mid; cw * ch];
    let mut owned: Vec<Vec<i32>> = vec![
        flat_y.iter().map(|&s| i32::from(s)).collect(),
        flat_c.iter().map(|&s| i32::from(s)).collect(),
        flat_c.iter().map(|&s| i32::from(s)).collect(),
    ];
    {
        let mut bufs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(3);
        for (p, buf) in owned.iter_mut().enumerate() {
            let (pw, ph) = if p == 0 { (w, h) } else { (cw, ch) };
            bufs.push(PlaneBuffer {
                rows: ph as u32,
                cols: pw as u32,
                samples: buf,
            });
        }
        film_grain_synthesis(&fg, bit_depth, 3, ssx, ssy, matrix_coefficients, &mut bufs);
    }
    probe_sigma16(&owned[plane], &flat_c, bit_depth)
}

/// σ × 16 (8-bit-normalized) between a synthesized i32 plane and its
/// flat source.
fn probe_sigma16(grained: &[i32], flat: &[u16], bit_depth: u8) -> u64 {
    let shift = u32::from(bit_depth - 8);
    let mut sq = 0u64;
    for (&g, &s) in grained.iter().zip(flat) {
        let d = (i64::from(g) - i64::from(s)) >> shift;
        sq += (d * d) as u64;
    }
    (((sq as f64 / flat.len().max(1) as f64).sqrt()) * 16.0)
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

/// σ × 16 (8-bit-normalized) of the difference between two
/// equal-length planes — the neutrality term's amplitude measure
/// (r444: luma and chroma planes alike).
#[must_use]
pub(crate) fn plane_sigma16(a: &[u16], b: &[u16], bit_depth: u8) -> u64 {
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

    fn est_flat(luma: u64, chroma: u64) -> NoiseEstimate {
        NoiseEstimate {
            sigma16: luma,
            bin_sigma16: [luma; 8],
            cb_sigma16: chroma,
            cb_bin_sigma16: [chroma; 4],
            cr_sigma16: chroma,
            cr_bin_sigma16: [chroma; 4],
        }
    }

    /// The calibration mapping is monotone: doubling the requested
    /// point scaling roughly doubles the realized sigma.
    #[test]
    fn calibration_is_positive_and_scaling_monotone() {
        let lo = build_grain_params(&est_flat(16, 0), 8, 1, 1, 2, None, false, false);
        let hi = build_grain_params(&est_flat(32, 0), 8, 1, 1, 2, None, false, false);
        assert!(lo.point_y_scaling[0] > 0);
        assert!(hi.point_y_scaling[0] > lo.point_y_scaling[0]);
    }

    /// r444 — the chroma arm emits four calibrated points per elected
    /// plane with the identity §7.18.3 index mults, and the AR arm
    /// lands the fitted taps on the parameter block.
    #[test]
    fn chroma_points_and_ar_taps_land() {
        let fg = build_grain_params(&est_flat(24, 12), 8, 1, 1, 2, None, true, true);
        assert_eq!(fg.num_cb_points, 4);
        assert_eq!(fg.num_cr_points, 4);
        assert_eq!(fg.point_cb_value[..4], [32, 96, 160, 224]);
        assert!(fg.point_cb_scaling[..4].iter().all(|&s| s > 0));
        assert_eq!((fg.cb_mult, fg.cb_luma_mult, fg.cb_offset), (192, 128, 256));
        let ar = ArFit {
            lag: 1,
            y: vec![10, -20, 30, 40],
            cb: vec![5, -6, 7, 8, 32],
            cr: vec![0, 0, 0, 0, -32],
        };
        let fg = build_grain_params(&est_flat(24, 12), 8, 1, 1, 2, Some(&ar), true, true);
        assert_eq!(fg.ar_coeff_lag, 1);
        assert_eq!(fg.ar_coeffs_y_plus_128[..4], [138, 108, 158, 168]);
        assert_eq!(fg.ar_coeffs_cb_plus_128[4], 160, "luma-corr tap last");
        assert_eq!(fg.ar_coeffs_cr_plus_128[4], 96);
    }

    /// r444 — the AR fit recovers a planted horizontal AR(1) field
    /// and stays white on true white noise.
    #[test]
    fn ar_fit_recovers_planted_correlation() {
        use crate::encoder::yuv_frame::ChromaFormat;
        let (w, h) = (96u32, 80u32);
        let mut state = 0x1357_9bdfu32;
        let mut rnd = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(((state >> 24) & 31) as i32 - 16)
        };
        // AR(1) along rows: n[x] = 0.55 n[x-1] + w.
        let mut noisy = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        for r in 0..h as usize {
            let mut prev = 0.0f64;
            for c in 0..w as usize {
                let n = 0.55 * prev + rnd();
                prev = n;
                noisy.y[r * (w as usize) + c] = (128.0 + n).round().clamp(0.0, 255.0) as u16;
            }
        }
        let flat = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        let fit = fit_ar_lag(&noisy, &flat, 1).expect("planted AR(1) must fit lag 1");
        // The immediate-left tap is the last lag-1 position.
        let left = fit.y[fit.y.len() - 1];
        assert!(
            (20..=50).contains(&left),
            "left tap ~0.55*64: got {left} (taps {:?})",
            fit.y
        );

        let mut white = YuvFrame::filled(w, h, 8, ChromaFormat::Yuv420, 128);
        for s in white.y.iter_mut() {
            *s = (128.0 + rnd()).round().clamp(0.0, 255.0) as u16;
        }
        if let Some(fit_w) = fit_ar_lag(&white, &flat, 1) {
            // White noise: every tap must quantise to (near) zero.
            assert!(
                fit_w.y.iter().all(|&c| c.abs() <= 3),
                "white noise fits near-zero taps ({:?})",
                fit_w.y
            );
        }
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
