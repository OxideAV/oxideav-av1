//! r422 — frame-level **global-motion estimation** for the inter
//! encoder: per-reference TRANSLATION / ROTZOOM / AFFINE candidate
//! models fitted to a coarse motion-estimation pre-pass, snapped onto
//! the §5.9.25 codable grid, §7.11.3.6-validated, and offered to the
//! frame header when they explain the measured motion field.
//!
//! How a model is chosen is pure encoder freedom — nothing here reads
//! the bitstream syntax. What the election CONTROLS is spec-visible
//! state: the elected `(GmType[ref], gm_params[ref][..])` pair is
//! written through the §5.9.24 write arm
//! ([`super::frame_obu`]), feeds the §7.10.2.1 `setup_global_mv`
//! derivation at every leaf (GLOBALMV / GLOBAL_GLOBALMV candidates),
//! and — for `GmType > TRANSLATION` on `>= 8x8` SIMPLE leaves — the
//! §7.11.3 global-warp prediction path. All of those consumers run
//! the decoder's own derivations on both sides of the encoder (search
//! mirror and write pass), so the elected model can never desync the
//! stream: it only changes WHICH streams the RD ladder prefers.
//!
//! ## Pipeline
//!
//! 1. [`collect_motion_samples`] — 16×16-block motion pre-pass over
//!    the input against one reference plane: an exhaustive coarse
//!    scan at HALF resolution (±[`PREPASS_RANGE`]`/2` over 2×2-mean
//!    planes — fine-texture aliasing and reference coding blur both
//!    wash out of the coarse SAD landscape), full-pel refinement,
//!    then 1/8-pel bilinear refinement. Each block yields one
//!    `(position, motion)` sample at the block centre.
//! 2. [`estimate_global_motion`] — least-squares fits of the three
//!    §6.8.18 model classes on the sample field (with one
//!    outlier-trim + refit round), quantization through
//!    [`super::frame_obu::quantize_global_param`], §7.11.3.6
//!    `setup_shear` validation for the warp classes, and a
//!    residual-energy election: a model is only offered when it
//!    explains the bulk of the measured motion energy (see
//!    [`ELECT_ENERGY_RATIO`]) and the field is actually moving.
//!
//! The returned parameters are EXACTLY the values the decoder will
//! reconstruct from the written stream (quantization runs before
//! validation and before the residual scoring, so the election judges
//! the codable model, not the ideal one).

use crate::encoder::frame_obu::quantize_global_param;
use crate::inter_pred::setup_shear;
use crate::uncompressed_header_tail::{WarpModelType, WARPEDMODEL_PREC_BITS};

/// Pre-pass block edge (luma samples). 16×16 keeps the sample count
/// meaningful on the 64×64 conformance frames (16 samples) while
/// staying cheap on larger inputs.
const PREPASS_BLOCK: usize = 16;

/// Pre-pass integer search radius (luma samples).
const PREPASS_RANGE: i32 = 24;

/// A model must explain at least this share of the measured motion
/// energy: `err_model <= (1 - ELECT_ENERGY_RATIO) * err_identity`.
const ELECT_ENERGY_RATIO: f64 = 0.75;

/// Minimum identity residual energy (mean squared motion, pel²) for
/// any election — a static field gains nothing from global motion
/// (GLOBALMV under IDENTITY already predicts the zero vector).
const MIN_MOTION_ENERGY: f64 = 0.25;

/// A higher model class must beat the next-simpler electable class by
/// this factor to justify its extra header coefficients.
const CLASS_UPGRADE_RATIO: f64 = 0.8;

/// ... and by at least this absolute residual-energy margin (pel²) —
/// residuals at the 1/8-pel measurement noise floor must not trigger
/// upgrades.
const CLASS_UPGRADE_MIN_GAIN: f64 = 0.02;

/// One pre-pass sample: block-centre position (luma pels) and the
/// measured motion (pels, 1/8-pel granularity).
#[derive(Debug, Clone, Copy)]
pub(crate) struct MotionSample {
    x: f64,
    y: f64,
    u: f64,
    v: f64,
}

/// 16×16-block motion pre-pass of `input_y` against `ref_y` (both
/// `width * height`, row-major, any coded depth). Returns one sample per fully
/// inside block. Estimation-only — the committed prediction never
/// uses these vectors, so the search here is free to be cheap.
pub(crate) fn collect_motion_samples(
    input_y: &[u16],
    ref_y: &[u16],
    width: usize,
    height: usize,
) -> Vec<MotionSample> {
    let mut samples = Vec::new();
    if width < PREPASS_BLOCK || height < PREPASS_BLOCK {
        return samples;
    }
    let sad_int = |row0: usize, col0: usize, dy: i32, dx: i32| -> u64 {
        let mut sad = 0u64;
        for i in 0..PREPASS_BLOCK {
            let sy = ((row0 + i) as i32 + dy).clamp(0, height as i32 - 1) as usize;
            for j in 0..PREPASS_BLOCK {
                let sx = ((col0 + j) as i32 + dx).clamp(0, width as i32 - 1) as usize;
                let d = i32::from(input_y[(row0 + i) * width + col0 + j])
                    - i32::from(ref_y[sy * width + sx]);
                sad += d.unsigned_abs() as u64;
            }
        }
        sad
    };
    // 1/8-pel bilinear SAD at motion `(my8, mx8)` (1/8-pel units).
    let sad_subpel = |row0: usize, col0: usize, my8: i32, mx8: i32| -> u64 {
        let fy = (my8.rem_euclid(8)) as u32;
        let fx = (mx8.rem_euclid(8)) as u32;
        let iy = (my8 - fy as i32) / 8;
        let ix = (mx8 - fx as i32) / 8;
        let at = |r: i32, c: i32| -> u32 {
            let r = r.clamp(0, height as i32 - 1) as usize;
            let c = c.clamp(0, width as i32 - 1) as usize;
            u32::from(ref_y[r * width + c])
        };
        let mut sad = 0u64;
        for i in 0..PREPASS_BLOCK {
            let r = (row0 + i) as i32 + iy;
            for j in 0..PREPASS_BLOCK {
                let c = (col0 + j) as i32 + ix;
                // Bilinear blend of the four neighbours (rounded).
                let p00 = at(r, c);
                let p01 = at(r, c + 1);
                let p10 = at(r + 1, c);
                let p11 = at(r + 1, c + 1);
                let top = p00 * (8 - fx) + p01 * fx;
                let bot = p10 * (8 - fx) + p11 * fx;
                let pred = (top * (8 - fy) + bot * fy + 32) >> 6;
                let d = i32::from(input_y[(row0 + i) * width + col0 + j]) - pred as i32;
                sad += d.unsigned_abs() as u64;
            }
        }
        sad
    };
    // Half-resolution planes (2×2 means) for the coarse stage: the
    // downsampling attenuates fine texture (whose periodicity aliases
    // a full-resolution coarse scan) and the reference's coding blur
    // alike, so the coarse SAD landscape keeps one broad basin at the
    // true motion.
    let (hw, hh) = (width / 2, height / 2);
    let downsample = |p: &[u16]| -> Vec<u16> {
        let mut d = vec![0u16; hw * hh];
        for r in 0..hh {
            for c in 0..hw {
                let s = u32::from(p[(2 * r) * width + 2 * c])
                    + u32::from(p[(2 * r) * width + 2 * c + 1])
                    + u32::from(p[(2 * r + 1) * width + 2 * c])
                    + u32::from(p[(2 * r + 1) * width + 2 * c + 1]);
                d[r * hw + c] = ((s + 2) >> 2) as u16;
            }
        }
        d
    };
    let input_half = downsample(input_y);
    let ref_half = downsample(ref_y);
    let half_block = PREPASS_BLOCK / 2;
    // Half-resolution SAD of the block's 8×8 shadow at offset
    // `(dy, dx)` (half-resolution samples).
    let sad_half = |row0: usize, col0: usize, dy: i32, dx: i32| -> u64 {
        let (r0, c0) = (row0 / 2, col0 / 2);
        let mut sad = 0u64;
        for i in 0..half_block {
            let sy = ((r0 + i) as i32 + dy).clamp(0, hh as i32 - 1) as usize;
            for j in 0..half_block {
                let sx = ((c0 + j) as i32 + dx).clamp(0, hw as i32 - 1) as usize;
                let d = i32::from(input_half[(r0 + i) * hw + c0 + j])
                    - i32::from(ref_half[sy * hw + sx]);
                sad += d.unsigned_abs() as u64;
            }
        }
        sad
    };
    let half_range = PREPASS_RANGE / 2;
    let mut row0 = 0usize;
    while row0 + PREPASS_BLOCK <= height {
        let mut col0 = 0usize;
        while col0 + PREPASS_BLOCK <= width {
            // Coarse stage: exhaustive step-1 scan at half resolution,
            // zero-biased.
            let mut half_best = (sad_half(row0, col0, 0, 0), 0i32, 0i32);
            for dy in -half_range..=half_range {
                for dx in -half_range..=half_range {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let c = sad_half(row0, col0, dy, dx)
                        + (dy.unsigned_abs() + dx.unsigned_abs()) as u64;
                    if c < half_best.0 {
                        half_best = (c, dy, dx);
                    }
                }
            }
            // Full-pel refinement around the doubled coarse winner.
            let (cy, cx) = (half_best.1 * 2, half_best.2 * 2);
            let mut best = (sad_int(row0, col0, cy, cx), cy, cx);
            for dy in cy - 3..=cy + 3 {
                for dx in cx - 3..=cx + 3 {
                    if dy == cy && dx == cx {
                        continue;
                    }
                    let c = sad_int(row0, col0, dy, dx)
                        + (dy.unsigned_abs() + dx.unsigned_abs()) as u64;
                    if c < best.0 {
                        best = (c, dy, dx);
                    }
                }
            }
            // 1/8-pel bilinear refinement: half-pel pass then
            // eighth-pel pass around the running best.
            let mut best8 = (
                sad_subpel(row0, col0, best.1 * 8, best.2 * 8),
                best.1 * 8,
                best.2 * 8,
            );
            for step in [4i32, 2, 1] {
                let (by, bx) = (best8.1, best8.2);
                for dy in [-step, 0, step] {
                    for dx in [-step, 0, step] {
                        if dy == 0 && dx == 0 {
                            continue;
                        }
                        let c = sad_subpel(row0, col0, by + dy, bx + dx);
                        if c < best8.0 {
                            best8 = (c, by + dy, bx + dx);
                        }
                    }
                }
            }
            samples.push(MotionSample {
                x: col0 as f64 + (PREPASS_BLOCK as f64) / 2.0 - 0.5,
                y: row0 as f64 + (PREPASS_BLOCK as f64) / 2.0 - 0.5,
                u: f64::from(best8.2) / 8.0,
                v: f64::from(best8.1) / 8.0,
            });
            col0 += PREPASS_BLOCK;
        }
        row0 += PREPASS_BLOCK;
    }
    samples
}

/// Solve `A · x = b` by Gaussian elimination with partial pivoting.
/// Returns `None` on a (near-)singular system.
fn solve_linear<const N: usize>(mut a: [[f64; N]; N], mut b: [f64; N]) -> Option<[f64; N]> {
    for col in 0..N {
        // Pivot.
        let mut piv = col;
        for row in col + 1..N {
            if a[row][col].abs() > a[piv][col].abs() {
                piv = row;
            }
        }
        if a[piv][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        // Eliminate below (the pivot row is copied out — `[[f64; N]]`
        // rows are `Copy` — so the target rows can be walked with
        // iterators).
        let pivot_row = a[col];
        let pivot_b = b[col];
        for row in col + 1..N {
            let f = a[row][col] / pivot_row[col];
            for (dst, src) in a[row][col..].iter_mut().zip(&pivot_row[col..]) {
                *dst -= f * *src;
            }
            b[row] -= f * pivot_b;
        }
    }
    // Back-substitute.
    let mut x = [0.0f64; N];
    for col in (0..N).rev() {
        let mut s = b[col];
        for k in col + 1..N {
            s -= a[col][k] * x[k];
        }
        x[col] = s / a[col][col];
    }
    Some(x)
}

/// The continuous model `dst = M · src + t` as motion residuals:
/// `u = (m2 - 1)·x + m3·y + m0`, `v = m4·x + (m5 - 1)·y + m1`.
#[derive(Debug, Clone, Copy)]
struct ContinuousModel {
    /// `[m0, m1, m2, m3, m4, m5]` in the §6.8.18 slot order
    /// (translation x / y, then the x-row and y-row coefficients).
    m: [f64; 6],
}

impl ContinuousModel {
    fn identity() -> Self {
        Self {
            m: [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        }
    }

    /// Model-predicted motion at `(x, y)`.
    fn motion_at(&self, x: f64, y: f64) -> (f64, f64) {
        let u = (self.m[2] - 1.0) * x + self.m[3] * y + self.m[0];
        let v = self.m[4] * x + (self.m[5] - 1.0) * y + self.m[1];
        (u, v)
    }
}

/// Mean squared residual of `model` over `samples` (pel²).
fn mean_sq_residual(model: &ContinuousModel, samples: &[MotionSample]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0;
    for s in samples {
        let (mu, mv) = model.motion_at(s.x, s.y);
        acc += (s.u - mu).powi(2) + (s.v - mv).powi(2);
    }
    acc / samples.len() as f64
}

/// TRANSLATION fit: the mean motion vector.
fn fit_translation(samples: &[MotionSample]) -> ContinuousModel {
    let n = samples.len() as f64;
    let (mut su, mut sv) = (0.0, 0.0);
    for s in samples {
        su += s.u;
        sv += s.v;
    }
    ContinuousModel {
        m: [su / n, sv / n, 1.0, 0.0, 0.0, 1.0],
    }
}

/// ROTZOOM fit (`m2 = m5 = 1 + α`, `m3 = b`, `m4 = -b`): the 4-unknown
/// normal equations of the joint x/y least squares. The `α`/`b` cross
/// terms cancel, leaving
///
/// ```text
///   α·Q  + tx·Sx + ty·Sy = Σ(x·u + y·v)      Q = Σ(x² + y²)
///   b·Q  + tx·Sy - ty·Sx = Σ(y·u - x·v)
///   α·Sx + b·Sy  + n·tx  = Σu
///   α·Sy - b·Sx  + n·ty  = Σv
/// ```
fn fit_rotzoom(samples: &[MotionSample]) -> Option<ContinuousModel> {
    let n = samples.len() as f64;
    let (mut sx, mut sy, mut q) = (0.0, 0.0, 0.0);
    let (mut p1, mut p2, mut su, mut sv) = (0.0, 0.0, 0.0, 0.0);
    for s in samples {
        sx += s.x;
        sy += s.y;
        q += s.x * s.x + s.y * s.y;
        p1 += s.x * s.u + s.y * s.v;
        p2 += s.y * s.u - s.x * s.v;
        su += s.u;
        sv += s.v;
    }
    // Unknown order: [α, b, tx, ty].
    let a = [
        [q, 0.0, sx, sy],
        [0.0, q, sy, -sx],
        [sx, sy, n, 0.0],
        [sy, -sx, 0.0, n],
    ];
    let x = solve_linear(a, [p1, p2, su, sv])?;
    Some(ContinuousModel {
        m: [x[2], x[3], 1.0 + x[0], x[1], -x[1], 1.0 + x[0]],
    })
}

/// AFFINE fit: two independent 3-unknown least squares (the u-row and
/// v-row share the same normal matrix).
fn fit_affine(samples: &[MotionSample]) -> Option<ContinuousModel> {
    let n = samples.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut sxu, mut syu, mut su) = (0.0, 0.0, 0.0);
    let (mut sxv, mut syv, mut sv) = (0.0, 0.0, 0.0);
    for s in samples {
        sx += s.x;
        sy += s.y;
        sxx += s.x * s.x;
        sxy += s.x * s.y;
        syy += s.y * s.y;
        sxu += s.x * s.u;
        syu += s.y * s.u;
        su += s.u;
        sxv += s.x * s.v;
        syv += s.y * s.v;
        sv += s.v;
    }
    let a = [[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]];
    let xu = solve_linear(a, [sxu, syu, su])?;
    let xv = solve_linear(a, [sxv, syv, sv])?;
    Some(ContinuousModel {
        m: [xu[2], xv[2], 1.0 + xu[0], xu[1], xv[0], 1.0 + xv[1]],
    })
}

/// Quantize a continuous model of class `gm_type` onto the §5.9.25
/// codable grid, re-deriving the ROTZOOM `[4]/[5]` pair and
/// re-fitting the translation slots against the QUANTIZED linear
/// part (the residual translation is what the coarser translation
/// grid should carry). Returns the `gm_params` array the decoder will
/// reconstruct.
fn quantize_model(
    model: &ContinuousModel,
    gm_type: WarpModelType,
    allow_high_precision_mv: bool,
    samples: &[MotionSample],
) -> [i32; 6] {
    let one = 1i32 << WARPEDMODEL_PREC_BITS;
    let scale = f64::from(one);
    let mut p = [0i32, 0, one, 0, 0, one];
    if gm_type as u8 >= WarpModelType::RotZoom as u8 {
        for (idx, slot) in p.iter_mut().enumerate().take(6).skip(2) {
            let raw = (model.m[idx] * scale).round() as i64;
            let raw = raw.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
            *slot = quantize_global_param(gm_type, allow_high_precision_mv, idx, raw);
        }
        if gm_type == WarpModelType::RotZoom {
            p[4] = -p[3];
            p[5] = p[2];
        }
    }
    // Residual translation against the quantized linear part: the
    // remaining mean motion once the (quantized) linear field is
    // subtracted.
    let (mut tu, mut tv) = (0.0f64, 0.0f64);
    if !samples.is_empty() {
        let m2 = f64::from(p[2]) / scale;
        let m3 = f64::from(p[3]) / scale;
        let m4 = f64::from(p[4]) / scale;
        let m5 = f64::from(p[5]) / scale;
        for s in samples {
            tu += s.u - ((m2 - 1.0) * s.x + m3 * s.y);
            tv += s.v - (m4 * s.x + (m5 - 1.0) * s.y);
        }
        tu /= samples.len() as f64;
        tv /= samples.len() as f64;
    }
    for (idx, t) in [(0usize, tu), (1usize, tv)] {
        let raw = (t * scale).round() as i64;
        let raw = raw.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        p[idx] = quantize_global_param(gm_type, allow_high_precision_mv, idx, raw);
    }
    p
}

/// The quantized `gm_params` as a continuous model (for residual
/// scoring — the election judges the CODABLE model).
fn continuous_of_params(p: &[i32; 6]) -> ContinuousModel {
    let scale = f64::from(1i32 << WARPEDMODEL_PREC_BITS);
    ContinuousModel {
        m: [
            f64::from(p[0]) / scale,
            f64::from(p[1]) / scale,
            f64::from(p[2]) / scale,
            f64::from(p[3]) / scale,
            f64::from(p[4]) / scale,
            f64::from(p[5]) / scale,
        ],
    }
}

/// One electable candidate after quantization + validation.
struct Candidate {
    gm_type: WarpModelType,
    params: [i32; 6],
    err: f64,
}

/// Fit, quantize, validate and score one model class over `samples`.
fn candidate_for(
    gm_type: WarpModelType,
    samples: &[MotionSample],
    allow_high_precision_mv: bool,
) -> Option<Candidate> {
    let min_samples = match gm_type {
        WarpModelType::Translation => 4,
        WarpModelType::RotZoom => 6,
        _ => 8,
    };
    if samples.len() < min_samples {
        return None;
    }
    let fit = |subset: &[MotionSample]| -> Option<ContinuousModel> {
        match gm_type {
            WarpModelType::Translation => Some(fit_translation(subset)),
            WarpModelType::RotZoom => fit_rotzoom(subset),
            WarpModelType::Affine => fit_affine(subset),
            WarpModelType::Identity => None,
        }
    };
    let first = fit(samples)?;
    // One outlier-trim + refit round: drop samples whose residual
    // exceeds max(1 pel, 3× the median residual).
    let mut residuals: Vec<f64> = samples
        .iter()
        .map(|s| {
            let (mu, mv) = first.motion_at(s.x, s.y);
            ((s.u - mu).powi(2) + (s.v - mv).powi(2)).sqrt()
        })
        .collect();
    let mut sorted = residuals.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("residuals are finite"));
    let median = sorted[sorted.len() / 2];
    let cutoff = (3.0 * median).max(1.0);
    let trimmed: Vec<MotionSample> = samples
        .iter()
        .zip(residuals.drain(..))
        .filter_map(|(s, r)| (r <= cutoff).then_some(*s))
        .collect();
    let (model, scored_samples) = if trimmed.len() >= min_samples {
        (fit(&trimmed)?, trimmed)
    } else {
        (first, samples.to_vec())
    };
    let params = quantize_model(&model, gm_type, allow_high_precision_mv, &scored_samples);
    // Identity after quantization ⇒ nothing to elect.
    if params
        == [
            0,
            0,
            1 << WARPEDMODEL_PREC_BITS,
            0,
            0,
            1 << WARPEDMODEL_PREC_BITS,
        ]
    {
        return None;
    }
    // §7.11.3.6 validity for the warp classes: an invalid shear
    // decodes translationally at best — never worth the coefficients.
    if gm_type as u8 >= WarpModelType::RotZoom as u8 {
        match setup_shear(params) {
            Some(s) if s.warp_valid => {}
            _ => return None,
        }
    }
    // Score the CODABLE model over the untrimmed field (the stream
    // affects every block, not just the inliers).
    let err = mean_sq_residual(&continuous_of_params(&params), samples);
    // r426 — §7.10.2.1 stores the TRANSLATION model in (row, col)
    // order: `mv[ 0 ] = gm_params[ ref ][ 0 ] >> ..` with `mv[ 0 ]`
    // the ROW component, while the ROTZOOM/AFFINE projection reads
    // `gm_params[ 0 ]` as the X (column) offset (`xc = .. +
    // gm_params[ 0 ]`, `mv[ 1 ] = xc`). The fit works in the affine
    // x-first convention throughout (including the codable-model
    // scoring above) and swaps at the emission boundary — the
    // pre-r426 packing shipped x-first TRANSLATION params, so
    // GLOBALMV predictions ran on a 90°-swapped vector: conformant
    // both sides (the header and §7.10.2.1 agreed), but the arm
    // never won an election.
    let params = if gm_type == WarpModelType::Translation {
        [
            params[1], params[0], params[2], params[3], params[4], params[5],
        ]
    } else {
        params
    };
    Some(Candidate {
        gm_type,
        params,
        err,
    })
}

/// Estimate one reference's global-motion model from its pre-pass
/// sample field. Returns the elected `(GmType, gm_params)` — possibly
/// `(Identity, identity params)` when no model both survives
/// validation and explains the motion field.
pub(crate) fn estimate_global_motion(
    samples: &[MotionSample],
    allow_high_precision_mv: bool,
) -> (WarpModelType, [i32; 6]) {
    let one = 1i32 << WARPEDMODEL_PREC_BITS;
    let identity = (WarpModelType::Identity, [0, 0, one, 0, 0, one]);
    if samples.is_empty() {
        return identity;
    }
    let base_err = mean_sq_residual(&ContinuousModel::identity(), samples);
    if base_err < MIN_MOTION_ENERGY {
        // Static field — IDENTITY already predicts it.
        return identity;
    }
    let gate = (1.0 - ELECT_ENERGY_RATIO) * base_err;
    let mut elected: Option<Candidate> = None;
    for gm_type in [
        WarpModelType::Translation,
        WarpModelType::RotZoom,
        WarpModelType::Affine,
    ] {
        let Some(cand) = candidate_for(gm_type, samples, allow_high_precision_mv) else {
            continue;
        };
        if cand.err > gate {
            continue;
        }
        elected = match elected {
            None => Some(cand),
            // A higher class must clearly beat the simpler winner to
            // justify its extra §5.9.25 coefficients — both relatively
            // AND by an absolute margin, so a fit that is already at
            // the measurement noise floor is never "upgraded" over
            // meaningless fractions of nothing.
            Some(prev)
                if cand.err <= CLASS_UPGRADE_RATIO * prev.err
                    && prev.err - cand.err > CLASS_UPGRADE_MIN_GAIN =>
            {
                Some(cand)
            }
            Some(prev) => Some(prev),
        };
    }
    match elected {
        Some(c) => (c.gm_type, c.params),
        None => identity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise a sample field from an exact continuous model plus
    /// bounded noise.
    fn field(model: &ContinuousModel, noise: f64) -> Vec<MotionSample> {
        let mut out = Vec::new();
        let mut rng: u32 = 0x1234_5678;
        let mut next = || {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (f64::from(rng >> 8) / f64::from(1u32 << 24) - 0.5) * 2.0 * noise
        };
        for by in 0..6 {
            for bx in 0..8 {
                let x = bx as f64 * 16.0 + 7.5;
                let y = by as f64 * 16.0 + 7.5;
                let (u, v) = model.motion_at(x, y);
                out.push(MotionSample {
                    x,
                    y,
                    u: u + next(),
                    v: v + next(),
                });
            }
        }
        out
    }

    #[test]
    fn translation_field_elects_translation() {
        let model = ContinuousModel {
            m: [3.25, -1.5, 1.0, 0.0, 0.0, 1.0],
        };
        let (t, p) = estimate_global_motion(&field(&model, 0.05), false);
        assert_eq!(t, WarpModelType::Translation);
        // hp = false ⇒ translation grid step is 1/4 pel (2 << 13).
        // r426 — §7.10.2.1 TRANSLATION order: `gm_params[ 0 ]` is the
        // ROW (v) component (`mv[ 0 ] = gm_params[ 0 ] >> ..`),
        // `gm_params[ 1 ]` the COLUMN (u) — the emission boundary
        // swaps out of the fit's x-first convention.
        assert_eq!(p[0], (-1.5f64 * 65536.0) as i32);
        assert_eq!(p[1], (3.25f64 * 65536.0) as i32);
    }

    #[test]
    fn zoom_field_elects_rotzoom() {
        // 2% zoom about the origin plus a small pan.
        let model = ContinuousModel {
            m: [1.0, 0.5, 1.02, 0.0, 0.0, 1.02],
        };
        let (t, p) = estimate_global_motion(&field(&model, 0.05), false);
        assert_eq!(t, WarpModelType::RotZoom);
        // The diagonal must be near 1.02 on the alpha grid and the
        // §5.9.24 derived pair must hold.
        assert_eq!(p[4], -p[3]);
        assert_eq!(p[5], p[2]);
        assert!((f64::from(p[2]) / 65536.0 - 1.02).abs() < 0.001, "{}", p[2]);
        // Shear-valid by construction.
        assert!(matches!(setup_shear(p), Some(s) if s.warp_valid));
    }

    #[test]
    fn rotation_field_elects_rotzoom() {
        // ~1° rotation: cos ≈ 0.9998, sin ≈ 0.0175.
        let (c, s) = (0.9998f64, 0.0175f64);
        let model = ContinuousModel {
            m: [0.0, 0.0, c, s, -s, c],
        };
        let (t, p) = estimate_global_motion(&field(&model, 0.05), false);
        assert_eq!(t, WarpModelType::RotZoom);
        assert!((f64::from(p[3]) / 65536.0 - s).abs() < 0.001);
    }

    #[test]
    fn shear_field_elects_affine() {
        // A pure x-shear cannot be expressed by ROTZOOM.
        let model = ContinuousModel {
            m: [0.0, 0.0, 1.0, 0.02, 0.0, 1.0],
        };
        let (t, p) = estimate_global_motion(&field(&model, 0.02), false);
        assert_eq!(t, WarpModelType::Affine);
        assert!((f64::from(p[3]) / 65536.0 - 0.02).abs() < 0.001);
        // The y-row shear stays (near-)zero — the noise floor may
        // quantize to a couple of 1/65536 steps.
        assert!(f64::from(p[4].abs()) / 65536.0 < 0.001);
    }

    #[test]
    fn static_field_stays_identity() {
        let (t, _) = estimate_global_motion(&field(&ContinuousModel::identity(), 0.1), false);
        assert_eq!(t, WarpModelType::Identity);
    }

    #[test]
    fn incoherent_field_stays_identity() {
        // Large uncorrelated noise ⇒ no model explains 75% of it.
        let (t, _) = estimate_global_motion(&field(&ContinuousModel::identity(), 4.0), false);
        assert_eq!(t, WarpModelType::Identity);
    }

    #[test]
    fn prepass_recovers_a_pure_translation() {
        // A textured plane translated by (+5, -3) full pels. The
        // texture is 5×5-smoothed hash noise: alias-free (no repeats)
        // but correlated over ~5 pels, so the SAD landscape has a
        // single minimum with a basin wide enough for the coarse
        // step-4 scan to land in.
        let (w, h) = (64usize, 64usize);
        let hash = |r: i64, c: i64| -> u32 {
            let mut x = (r.wrapping_mul(0x9E37_79B1) ^ c.wrapping_mul(0x85EB_CA77)) as u64;
            x ^= x >> 15;
            x = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((x >> 32) & 0xFF) as u32
        };
        let tex = |r: i64, c: i64| -> u16 {
            let mut acc = 0u32;
            for dr in -2..=2i64 {
                for dc in -2..=2i64 {
                    acc += hash(r + dr, c + dc);
                }
            }
            (acc / 25) as u16
        };
        let mut refp = vec![0u16; w * h];
        let mut cur = vec![0u16; w * h];
        for r in 0..h as i64 {
            for c in 0..w as i64 {
                refp[(r * w as i64 + c) as usize] = tex(r, c);
                // The block at current (r, c) matches the reference at
                // (r - 5, c + 3), i.e. measured motion (u, v) = (+3, -5)
                // under the ref_pos = cur_pos + mv convention.
                cur[(r * w as i64 + c) as usize] = tex(r - 5, c + 3);
            }
        }
        let samples = collect_motion_samples(&cur, &refp, w, h);
        assert_eq!(samples.len(), 16);
        // Interior blocks (the border blocks see the clamp shadow)
        // must all measure (u, v) = (+3, -5).
        let interior: Vec<&MotionSample> = samples
            .iter()
            .filter(|s| s.x > 16.0 && s.x < 48.0 && s.y > 16.0 && s.y < 48.0)
            .collect();
        assert!(!interior.is_empty());
        for s in interior {
            assert!(
                (s.u - 3.0).abs() < 0.51 && (s.v + 5.0).abs() < 0.51,
                "block at ({}, {}) measured ({}, {})",
                s.x,
                s.y,
                s.u,
                s.v
            );
        }
    }
}
