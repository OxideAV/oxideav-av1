//! §7.18.3 Film grain synthesis — the final post-processing layer per
//! av1-spec p.338-347. Runs after the §7.17 loop restoration pass and
//! modifies the `OutY` / `OutU` / `OutV` output samples by adding a
//! noise pattern derived from the §5.9.30 [`FilmGrainParams`] block.
//!
//! ## Coverage (round 199 — close-out push)
//!
//! This module covers the §7.18.3 driver end-to-end at the
//! sample-modification level:
//!
//! * §7.18.3.1 — [`film_grain_synthesis`]: top-level driver that
//!   wires `RandomRegister`, `GrainCenter`, `GrainMin`, `GrainMax`,
//!   then invokes §7.18.3.3 / §7.18.3.4 / §7.18.3.5 in order.
//! * §7.18.3.2 — [`RandomRegister::next`]: 16-bit LFSR per av1-spec
//!   p.339 lines 18704-18711 (`bit = ((r >> 0) ^ (r >> 1) ^ (r >> 3) ^
//!   (r >> 12)) & 1`; tap polynomial returns the upper `bits` bits).
//! * §7.18.3.3 — [`generate_grain`]: populates the `LumaGrain` /
//!   `CbGrain` / `CrGrain` arrays via the §7.18.3.2 LFSR seeded with
//!   `grain_seed` (luma) / `grain_seed ^ 0xb524` (Cb) / `grain_seed ^
//!   0x49d8` (Cr), then runs the §7.18.3.3 AR filter with
//!   `ar_coeffs_y_plus_128` / `ar_coeffs_cb_plus_128` /
//!   `ar_coeffs_cr_plus_128` (including the chroma-from-luma cross
//!   term when `num_y_points > 0`).
//! * §7.18.3.4 — [`scaling_lookup_init`]: builds 3 × 256-entry
//!   `ScalingLut[plane][x]` from `(num_y_points, point_y_value,
//!   point_y_scaling)` etc. via piecewise linear interpolation, with
//!   the `numPoints == 0` zero-fill, the `point_y_value[0]` left
//!   plateau, the `(65536 + deltaX/2) / deltaX` per-segment
//!   reciprocal, and the `point_y_value[numPoints-1]` right plateau.
//! * §7.18.3.5 — [`add_noise_synthesis`]: builds `noiseStripe[lumaNum]
//!   [plane]` 32-luma-sample-high stripes (with the `overlap_flag`
//!   17/27 / 22/23 grain-block blending in both X / Y), assembles them
//!   into the full `noiseImage[plane][y][x]`, then blends
//!   `noiseImage[plane]` with `OutY` / `OutU` / `OutV` using
//!   `scale_lut` and the `cb_mult` / `cb_luma_mult` / `cb_offset` /
//!   `cr_mult` / `cr_luma_mult` / `cr_offset` chroma-from-luma merge.
//! * §7.18.3.5 — [`scale_lut`]: piecewise linear interpolation into
//!   `ScalingLut[plane]` for `BitDepth > 8` (the `8`-bit fast path
//!   indexes directly).
//!
//! ## Standalone-friendly surface
//!
//! Like the §7.14 / §7.15 / §7.17 drivers, the top-level entry
//! [`film_grain_synthesis`] takes:
//!
//! * The parsed [`FilmGrainParams`] (§5.9.30, populated by the
//!   bit-reader).
//! * Frame-level `(bit_depth, num_planes, sub_x, sub_y,
//!   matrix_coefficients)`.
//! * Mutable [`crate::loop_filter::PlaneBuffer`]s for the three planes
//!   (the §7.18.2 output samples — same shape produced by the §7.17
//!   restoration step).
//!
//! Internal §7.18.3.3 / §7.18.3.4 / §7.18.3.5 helpers are exposed so
//! external test harnesses can pin individual sub-stages.
//!
//! ## Out-of-scope for this arc
//!
//! * Bit-depth-12 / monochrome fast paths beyond the spec's
//!   `BitDepth == 8 || x == 255` branch in [`scale_lut`].
//! * SIMD batched filtering — the reference loop here mirrors the
//!   spec's per-sample formulation.
//! * Stripe-buffer line-buffer compression (the spec p.346 Note
//!   describes a 2-line-buffer luma / 1-line-buffer chroma variant
//!   that is purely implementation-side).

use crate::film_grain_tables::GAUSSIAN_SEQUENCE;
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::FilmGrainParams;

// =====================================================================
// §7.18.3.3 generate-grain extents — av1-spec p.339-340.
// =====================================================================

/// Width of the §7.18.3.3 `LumaGrain[][]` array — av1-spec p.339 line
/// 18720.
pub const LUMA_GRAIN_W: usize = 82;

/// Height of the §7.18.3.3 `LumaGrain[][]` array — av1-spec p.339 line
/// 18720.
pub const LUMA_GRAIN_H: usize = 73;

/// `MC_IDENTITY` per av1-spec §6.4.2 (`matrix_coefficients == 0`) —
/// referenced by the §7.18.3.5 `maxChroma` derivation when
/// `clip_to_restricted_range == 1`.
pub const MC_IDENTITY: u8 = 0;

// =====================================================================
// §7.18.3.2 random-number process — av1-spec p.338-339 lines
// 18704-18711.
// =====================================================================

/// 16-bit LFSR backing the §7.18.3.2 `get_random_number(bits)` call.
///
/// The spec stores `RandomRegister` as a frame-scope variable;
/// wrapping it in a struct makes the §7.18.3.5 `RandomRegister =
/// grain_seed ^ ((lumaNum * 37 + 178) & 255) << 8 ^ ((lumaNum * 173 +
/// 105) & 255)` per-row reset explicit.
#[derive(Debug, Clone, Copy)]
pub struct RandomRegister {
    /// `RandomRegister` per av1-spec p.339 line 18705.
    pub state: u16,
}

impl RandomRegister {
    /// Construct a fresh LFSR seeded with `seed` per §7.18.3.1 step 1.
    #[must_use]
    pub const fn new(seed: u16) -> Self {
        Self { state: seed }
    }

    /// `get_random_number(bits)` per av1-spec p.339 lines 18704-18711.
    ///
    /// `bits` is the §7.18.3.3 / §7.18.3.5 caller-supplied bit count
    /// (`11` for the §7.18.3.3 Gaussian lookup; `8` for the §7.18.3.5
    /// per-block `(offsetX, offsetY)` derivation). Returns the upper
    /// `bits` bits of the post-shift register.
    #[must_use]
    pub fn next(&mut self, bits: u32) -> u32 {
        let r = u32::from(self.state);
        // av1-spec p.339 line 18706: bit = ((r >> 0) ^ (r >> 1) ^
        //                                   (r >> 3) ^ (r >> 12)) & 1.
        let bit = (r ^ (r >> 1) ^ (r >> 3) ^ (r >> 12)) & 1;
        // av1-spec p.339 line 18707: r = (r >> 1) | (bit << 15).
        let r = (r >> 1) | (bit << 15);
        self.state = r as u16;
        // av1-spec p.339 line 18708: result = (r >> (16 - bits)) &
        //                                     ((1 << bits) - 1).
        (r >> (16 - bits)) & ((1u32 << bits) - 1)
    }
}

// =====================================================================
// Helpers — `Round2`, `Clip3`, `Clip1`.
// =====================================================================

/// `Round2(x, n)` per av1-spec "Conventions" p.27 — `(x + (1 << (n -
/// 1))) >> n`, with the `n == 0` short-circuit to `x`.
#[inline]
fn round2(x: i32, n: u32) -> i32 {
    if n == 0 {
        x
    } else {
        (x + (1 << (n - 1))) >> n
    }
}

/// `Clip3(a, b, x)` per av1-spec "Conventions" p.26 — clamp `x` to
/// `[a, b]`.
#[inline]
fn clip3(a: i32, b: i32, x: i32) -> i32 {
    x.clamp(a, b)
}

/// `Clip1(x)` per av1-spec "Conventions" p.27 — `Clip3(0, (1 <<
/// BitDepth) - 1, x)`.
#[inline]
fn clip1(bit_depth: u8, x: i32) -> i32 {
    clip3(0, (1 << bit_depth) - 1, x)
}

// =====================================================================
// §7.18.3.3 generate-grain process — av1-spec p.339-342.
// =====================================================================

/// Result of the §7.18.3.3 generate-grain process.
///
/// `luma` is the `LumaGrain[][]` array; `cb` / `cr` are the
/// `CbGrain[][]` / `CrGrain[][]` arrays. The chroma arrays are
/// `chromaW × chromaH` per av1-spec p.340 lines 18770-18772 (44/82
/// wide × 38/73 tall depending on subsampling).
#[derive(Debug, Clone)]
pub struct GrainArrays {
    /// `LumaGrain[y][x]` per av1-spec p.339 — row-major `73 × 82`.
    pub luma: Vec<i32>,
    /// `CbGrain[y][x]` per av1-spec p.340 — row-major `chromaH ×
    /// chromaW`.
    pub cb: Vec<i32>,
    /// `CrGrain[y][x]` per av1-spec p.340 — row-major `chromaH ×
    /// chromaW`.
    pub cr: Vec<i32>,
    /// `chromaW` per av1-spec p.340 line 18770.
    pub chroma_w: usize,
    /// `chromaH` per av1-spec p.340 line 18772.
    pub chroma_h: usize,
}

impl GrainArrays {
    /// `LumaGrain[y][x]` accessor — row-major into [`Self::luma`].
    #[inline]
    #[must_use]
    pub fn luma(&self, y: usize, x: usize) -> i32 {
        self.luma[y * LUMA_GRAIN_W + x]
    }

    /// `CbGrain[y][x]` accessor — row-major into [`Self::cb`].
    #[inline]
    #[must_use]
    pub fn cb(&self, y: usize, x: usize) -> i32 {
        self.cb[y * self.chroma_w + x]
    }

    /// `CrGrain[y][x]` accessor — row-major into [`Self::cr`].
    #[inline]
    #[must_use]
    pub fn cr(&self, y: usize, x: usize) -> i32 {
        self.cr[y * self.chroma_w + x]
    }
}

/// §7.18.3.3 generate-grain process — av1-spec p.339-342.
///
/// Populates the `LumaGrain` / `CbGrain` / `CrGrain` arrays per the
/// spec's two-phase build:
///
///   1. White-noise pass: index the Gaussian sequence with an 11-bit
///      LFSR draw (per `num_y_points` / `num_cb_points` /
///      `num_cr_points` / `chroma_scaling_from_luma` gates), `Round2`
///      to the `12 - BitDepth + grain_scale_shift` precision.
///   2. AR filter pass: convolve the white noise with the `(2 *
///      ar_coeff_lag + 1) × (ar_coeff_lag + 1)` causal footprint per
///      `ar_coeffs_y_plus_128 - 128`, `Round2` by `ar_coeff_shift`,
///      `Clip3(GrainMin, GrainMax, …)`.
///
/// The chroma pass folds in an extra cross-component term per
/// av1-spec p.341 lines 18830-18840: when `num_y_points > 0` and
/// `(deltaRow, deltaCol) == (0, 0)`, the `luma` average is added
/// (subsampling-folded) before the central tap.
#[must_use]
pub fn generate_grain(fg: &FilmGrainParams, bit_depth: u8, sub_x: u8, sub_y: u8) -> GrainArrays {
    let grain_center = 128i32 << (bit_depth - 8);
    let grain_min = -grain_center;
    let grain_max = (256i32 << (bit_depth - 8)) - 1 - grain_center;

    // av1-spec p.339 line 18723: shift = 12 - BitDepth +
    //                                    grain_scale_shift.
    let init_shift = 12u32.wrapping_sub(u32::from(bit_depth)) + u32::from(fg.grain_scale_shift);

    // ----- LumaGrain: white noise + AR filter -----
    let mut luma = vec![0i32; LUMA_GRAIN_H * LUMA_GRAIN_W];
    {
        let mut rr = RandomRegister::new(fg.grain_seed);
        for y in 0..LUMA_GRAIN_H {
            for x in 0..LUMA_GRAIN_W {
                let g = if fg.num_y_points > 0 {
                    let idx = rr.next(11) as usize;
                    i32::from(GAUSSIAN_SEQUENCE[idx])
                } else {
                    0
                };
                luma[y * LUMA_GRAIN_W + x] = round2(g, init_shift);
            }
        }
    }

    if fg.num_y_points > 0 {
        // av1-spec p.339 line 18748: shift = ar_coeff_shift_minus_6 +
        //                                    6 (encoded as ar_coeff_shift).
        let ar_shift = u32::from(fg.ar_coeff_shift);
        let lag = i32::from(fg.ar_coeff_lag);
        for y in 3..LUMA_GRAIN_H as i32 {
            for x in 3..(LUMA_GRAIN_W as i32 - 3) {
                let mut s: i32 = 0;
                let mut pos: usize = 0;
                'outer: for d_row in -lag..=0 {
                    for d_col in -lag..=lag {
                        if d_row == 0 && d_col == 0 {
                            break 'outer;
                        }
                        let c = i32::from(fg.ar_coeffs_y_plus_128[pos]) - 128;
                        let yi = (y + d_row) as usize;
                        let xi = (x + d_col) as usize;
                        s += luma[yi * LUMA_GRAIN_W + xi] * c;
                        pos += 1;
                    }
                }
                let yi = y as usize;
                let xi = x as usize;
                let v = luma[yi * LUMA_GRAIN_W + xi] + round2(s, ar_shift);
                luma[yi * LUMA_GRAIN_W + xi] = clip3(grain_min, grain_max, v);
            }
        }
    }

    // av1-spec p.340 lines 18770-18772: chromaW / chromaH.
    let chroma_w = if sub_x != 0 { 44 } else { 82 };
    let chroma_h = if sub_y != 0 { 38 } else { 73 };

    // ----- CbGrain: white noise -----
    let mut cb = vec![0i32; chroma_h * chroma_w];
    {
        let mut rr = RandomRegister::new(fg.grain_seed ^ 0xb524);
        for y in 0..chroma_h {
            for x in 0..chroma_w {
                let g = if fg.num_cb_points > 0 || fg.chroma_scaling_from_luma {
                    let idx = rr.next(11) as usize;
                    i32::from(GAUSSIAN_SEQUENCE[idx])
                } else {
                    0
                };
                cb[y * chroma_w + x] = round2(g, init_shift);
            }
        }
    }

    // ----- CrGrain: white noise -----
    let mut cr = vec![0i32; chroma_h * chroma_w];
    {
        let mut rr = RandomRegister::new(fg.grain_seed ^ 0x49d8);
        for y in 0..chroma_h {
            for x in 0..chroma_w {
                let g = if fg.num_cr_points > 0 || fg.chroma_scaling_from_luma {
                    let idx = rr.next(11) as usize;
                    i32::from(GAUSSIAN_SEQUENCE[idx])
                } else {
                    0
                };
                cr[y * chroma_w + x] = round2(g, init_shift);
            }
        }
    }

    // ----- Chroma AR filter (jointly Cb + Cr) -----
    if fg.chroma_scaling_from_luma || fg.num_cb_points > 0 || fg.num_cr_points > 0 {
        let ar_shift = u32::from(fg.ar_coeff_shift);
        let lag = i32::from(fg.ar_coeff_lag);
        for y in 3..chroma_h as i32 {
            for x in 3..(chroma_w as i32 - 3) {
                let mut s0: i32 = 0;
                let mut s1: i32 = 0;
                let mut pos: usize = 0;
                'outer: for d_row in -lag..=0 {
                    for d_col in -lag..=lag {
                        let c0 = i32::from(fg.ar_coeffs_cb_plus_128[pos]) - 128;
                        let c1 = i32::from(fg.ar_coeffs_cr_plus_128[pos]) - 128;
                        if d_row == 0 && d_col == 0 {
                            // av1-spec p.341 lines 18830-18840: when
                            // num_y_points > 0, the centre tap pulls
                            // the (subsampling-folded) luma average.
                            if fg.num_y_points > 0 {
                                let luma_x = ((x - 3) << sub_x) + 3;
                                let luma_y = ((y - 3) << sub_y) + 3;
                                let mut l: i32 = 0;
                                for i in 0..=i32::from(sub_y) {
                                    for j in 0..=i32::from(sub_x) {
                                        let yi = (luma_y + i) as usize;
                                        let xi = (luma_x + j) as usize;
                                        l += luma[yi * LUMA_GRAIN_W + xi];
                                    }
                                }
                                let n = u32::from(sub_x) + u32::from(sub_y);
                                let l = round2(l, n);
                                s0 += l * c0;
                                s1 += l * c1;
                            }
                            break 'outer;
                        }
                        let yi = (y + d_row) as usize;
                        let xi = (x + d_col) as usize;
                        s0 += cb[yi * chroma_w + xi] * c0;
                        s1 += cr[yi * chroma_w + xi] * c1;
                        pos += 1;
                    }
                }
                let yi = y as usize;
                let xi = x as usize;
                let v0 = cb[yi * chroma_w + xi] + round2(s0, ar_shift);
                let v1 = cr[yi * chroma_w + xi] + round2(s1, ar_shift);
                cb[yi * chroma_w + xi] = clip3(grain_min, grain_max, v0);
                cr[yi * chroma_w + xi] = clip3(grain_min, grain_max, v1);
            }
        }
    }

    GrainArrays {
        luma,
        cb,
        cr,
        chroma_w,
        chroma_h,
    }
}

// =====================================================================
// §7.18.3.4 scaling-lookup initialization — av1-spec p.342-343.
// =====================================================================

/// Three `[u8; 256]` scaling lookup tables, one per plane. Indexed by
/// the §7.18.3.5 `scale_lut(plane, index)` helper.
#[derive(Debug, Clone)]
pub struct ScalingLut {
    /// `ScalingLut[plane][x]` per av1-spec p.343 line 18875. Index 0
    /// is luma; 1 is Cb; 2 is Cr.
    pub tables: [[u8; 256]; 3],
}

impl Default for ScalingLut {
    fn default() -> Self {
        Self {
            tables: [[0u8; 256]; 3],
        }
    }
}

/// §7.18.3.4 scaling-lookup initialization process — av1-spec p.343
/// lines 18875-18903.
///
/// For each `plane ∈ 0..NumPlanes`, builds the 256-entry
/// `ScalingLut[plane]` from the §5.9.30 `(point_*_value,
/// point_*_scaling)` pairs by piecewise linear interpolation. The
/// plane-to-points mapping per `get_x` / `get_y` (p.343 lines
/// 18916-18932): luma always reads luma points; chroma reads luma
/// points when `chroma_scaling_from_luma == 1`, otherwise its own.
///
/// When `numPoints == 0` the table is zero-filled (the §7.18.3.5
/// blend then multiplies by zero noise scale).
#[must_use]
pub fn scaling_lookup_init(fg: &FilmGrainParams, num_planes: u8) -> ScalingLut {
    let mut lut = ScalingLut::default();
    for plane in 0..num_planes.min(3) {
        let (num_points, get_x, get_y): (
            u8,
            &[u8; crate::uncompressed_header_tail::MAX_NUM_Y_POINTS],
            &[u8; crate::uncompressed_header_tail::MAX_NUM_Y_POINTS],
        ) = if plane == 0 || fg.chroma_scaling_from_luma {
            (fg.num_y_points, &fg.point_y_value, &fg.point_y_scaling)
        } else if plane == 1 {
            // av1-spec p.343 get_x/get_y: chroma branches index into
            // the chroma point arrays (sized at MAX_NUM_CHROMA_POINTS
            // = 10). Adapt the read paths so the shared interpolation
            // loop can ignore the dimensional split.
            let np = fg.num_cb_points;
            scaling_lookup_fill_chroma(
                &mut lut.tables[plane as usize],
                np,
                &fg.point_cb_value,
                &fg.point_cb_scaling,
            );
            continue;
        } else {
            let np = fg.num_cr_points;
            scaling_lookup_fill_chroma(
                &mut lut.tables[plane as usize],
                np,
                &fg.point_cr_value,
                &fg.point_cr_scaling,
            );
            continue;
        };
        scaling_lookup_fill_luma(&mut lut.tables[plane as usize], num_points, get_x, get_y);
    }
    lut
}

/// `scaling_lookup_init`'s luma-shaped fill — accepts the
/// `MAX_NUM_Y_POINTS`-sized point arrays.
fn scaling_lookup_fill_luma(
    table: &mut [u8; 256],
    num_points: u8,
    point_value: &[u8; crate::uncompressed_header_tail::MAX_NUM_Y_POINTS],
    point_scaling: &[u8; crate::uncompressed_header_tail::MAX_NUM_Y_POINTS],
) {
    let np = num_points as usize;
    scaling_lookup_fill(table, np, |i| point_value[i], |i| point_scaling[i]);
}

/// `scaling_lookup_init`'s chroma-shaped fill — accepts the
/// `MAX_NUM_CHROMA_POINTS`-sized point arrays.
fn scaling_lookup_fill_chroma(
    table: &mut [u8; 256],
    num_points: u8,
    point_value: &[u8; crate::uncompressed_header_tail::MAX_NUM_CHROMA_POINTS],
    point_scaling: &[u8; crate::uncompressed_header_tail::MAX_NUM_CHROMA_POINTS],
) {
    let np = num_points as usize;
    scaling_lookup_fill(table, np, |i| point_value[i], |i| point_scaling[i]);
}

/// Shared §7.18.3.4 piecewise linear interpolation body — av1-spec
/// p.343 lines 18882-18902.
fn scaling_lookup_fill<F, G>(table: &mut [u8; 256], num_points: usize, get_x: F, get_y: G)
where
    F: Fn(usize) -> u8,
    G: Fn(usize) -> u8,
{
    if num_points == 0 {
        // av1-spec p.343 line 18883: numPoints == 0 ⇒ zero-fill.
        *table = [0u8; 256];
        return;
    }
    // av1-spec p.343 lines 18887-18889: left plateau.
    let x0 = get_x(0) as usize;
    let y0 = get_y(0);
    for v in &mut table[..x0.min(256)] {
        *v = y0;
    }
    // av1-spec p.343 lines 18890-18898: per-segment linear ramp.
    for i in 0..num_points.saturating_sub(1) {
        let xa = get_x(i) as i32;
        let xb = get_x(i + 1) as i32;
        let ya = get_y(i) as i32;
        let yb = get_y(i + 1) as i32;
        let delta_x = xb - xa;
        if delta_x <= 0 {
            // §6.8.20 conformance forbids non-increasing point_*_value
            // sequences; defensive guard keeps a misconfigured caller
            // from panicking on division by zero or backwards writes.
            continue;
        }
        let delta_y = yb - ya;
        // av1-spec p.343 line 18893: delta = deltaY * ((65536 +
        //                                              (deltaX >> 1)) /
        //                                              deltaX).
        let delta = delta_y * ((65536 + (delta_x >> 1)) / delta_x);
        for x in 0..delta_x {
            let v = ya + ((x * delta + 32768) >> 16);
            let idx = (xa + x) as usize;
            if idx < 256 {
                table[idx] = v.clamp(0, 255) as u8;
            }
        }
    }
    // av1-spec p.343 lines 18899-18901: right plateau.
    let xn = get_x(num_points - 1) as usize;
    let yn = get_y(num_points - 1);
    for v in &mut table[xn.min(256)..] {
        *v = yn;
    }
}

// =====================================================================
// §7.18.3.5 add-noise synthesis — av1-spec p.344-347.
// =====================================================================

/// §7.18.3.5 `scale_lut(plane, index)` — av1-spec p.347 lines
/// 18144-18155.
///
/// For `BitDepth == 8` (the only currently exercised path in the
/// per-sample blend) this collapses to a direct table lookup; the
/// higher-bit-depth path interpolates between two adjacent entries.
#[inline]
#[must_use]
pub fn scale_lut(lut: &ScalingLut, plane: u8, index: i32, bit_depth: u8) -> i32 {
    let shift = u32::from(bit_depth) - 8;
    let x = (index >> shift) as usize;
    let table = &lut.tables[plane as usize];
    if bit_depth == 8 || x >= 255 {
        i32::from(table[x.min(255)])
    } else {
        let rem = index - ((x as i32) << shift);
        let start = i32::from(table[x]);
        let end = i32::from(table[x + 1]);
        start + round2((end - start) * rem, shift)
    }
}

/// §7.18.3.5 add-noise synthesis process — av1-spec p.344-347.
///
/// Builds the `noiseStripe[lumaNum][plane]` 32-luma-sample-high
/// stripes (with the `overlap_flag` 17/27 left-edge and 22/23
/// chroma-edge blending), assembles them into the full
/// `noiseImage[plane]`, then blends `noiseImage[plane]` with the
/// `OutY` / `OutU` / `OutV` planes using
/// [`scale_lut`] and the §7.18.3.5 chroma-from-luma merge.
///
/// `out_planes` is mutated in place. `out_planes[0]` is luma;
/// `out_planes[1]` / `out_planes[2]` are Cb / Cr. For monochrome
/// (`num_planes == 1`) only the luma blend runs.
#[allow(clippy::too_many_arguments)]
pub fn add_noise_synthesis(
    fg: &FilmGrainParams,
    grain: &GrainArrays,
    lut: &ScalingLut,
    bit_depth: u8,
    num_planes: u8,
    sub_x: u8,
    sub_y: u8,
    matrix_coefficients: u8,
    out_planes: &mut [PlaneBuffer<'_>],
) {
    if out_planes.is_empty() {
        return;
    }
    let w = out_planes[0].cols;
    let h = out_planes[0].rows;
    if w == 0 || h == 0 {
        return;
    }

    let grain_center = 128i32 << (bit_depth - 8);
    let grain_min = -grain_center;
    let grain_max = (256i32 << (bit_depth - 8)) - 1 - grain_center;

    // ----- Build noiseStripe arrays per plane -----
    // For each plane: per-row stripes of height (34 >> planeSubY) and
    // width derived from frame width / planeSubX. Stacked into a
    // single Vec<i32> indexed [lumaNum][plane][i][x].
    let np = num_planes.min(3) as usize;
    let num_luma_stripes = h.div_ceil(2).div_ceil(16);

    // Per-plane stripe dimensions.
    let mut stripe_h = [0usize; 3];
    let mut stripe_w = [0usize; 3];
    for plane in 0..np {
        let psx = if plane > 0 { sub_x } else { 0 };
        let psy = if plane > 0 { sub_y } else { 0 };
        stripe_h[plane] = 34usize >> psy;
        // Width must hold at least all writes at (x * 2 + j) /
        // (x + j); use the per-plane scan extent.
        let plane_w = ((w + u32::from(psx)) >> psx) as usize;
        // The spec writes up to `34` extra cols on the right edge;
        // pad the buffer to cover.
        stripe_w[plane] = plane_w + 34;
    }

    // noise_stripe[lumaNum][plane] = Vec of size stripe_h*stripe_w.
    let mut noise_stripe: Vec<Vec<Vec<i32>>> = (0..num_luma_stripes)
        .map(|_| {
            (0..np)
                .map(|plane| vec![0i32; stripe_h[plane] * stripe_w[plane]])
                .collect()
        })
        .collect();

    // av1-spec p.345 lines 18964-19010: per-stripe construction.
    let mut luma_num = 0usize;
    let mut y_stripe = 0u32;
    while y_stripe < h.div_ceil(2) {
        let mut rr = RandomRegister::new(fg.grain_seed);
        let xor_hi = ((luma_num as u32).wrapping_mul(37).wrapping_add(178) & 255) << 8;
        let xor_lo = (luma_num as u32).wrapping_mul(173).wrapping_add(105) & 255;
        rr.state ^= xor_hi as u16;
        rr.state ^= xor_lo as u16;
        let mut x_block = 0u32;
        while x_block < w.div_ceil(2) {
            let rand = rr.next(8);
            let offset_x = (rand >> 4) as i32;
            let offset_y = (rand & 15) as i32;
            for plane in 0..np {
                let psx = if plane > 0 { sub_x } else { 0 };
                let psy = if plane > 0 { sub_y } else { 0 };
                let plane_off_x = if psx != 0 {
                    6 + offset_x
                } else {
                    9 + offset_x * 2
                };
                let plane_off_y = if psy != 0 {
                    6 + offset_y
                } else {
                    9 + offset_y * 2
                };
                let h_steps = 34i32 >> psy;
                let w_steps = 34i32 >> psx;
                for i in 0..h_steps {
                    for j in 0..w_steps {
                        let yi = (plane_off_y + i) as usize;
                        let xi = (plane_off_x + j) as usize;
                        let g0 = match plane {
                            0 => grain.luma[yi * LUMA_GRAIN_W + xi],
                            1 => grain.cb[yi * grain.chroma_w + xi],
                            _ => grain.cr[yi * grain.chroma_w + xi],
                        };
                        let mut g = g0;
                        if psx == 0 {
                            let col = (x_block as i32) * 2 + j;
                            if j < 2 && fg.overlap_flag && x_block > 0 {
                                let old = noise_stripe[luma_num][plane]
                                    [(i as usize) * stripe_w[plane] + col as usize];
                                g = if j == 0 {
                                    old * 27 + g * 17
                                } else {
                                    old * 17 + g * 27
                                };
                                g = clip3(grain_min, grain_max, round2(g, 5));
                            }
                            if (col as usize) < stripe_w[plane] {
                                noise_stripe[luma_num][plane]
                                    [(i as usize) * stripe_w[plane] + col as usize] = g;
                            }
                        } else {
                            let col = (x_block as i32) + j;
                            if j == 0 && fg.overlap_flag && x_block > 0 {
                                let old = noise_stripe[luma_num][plane]
                                    [(i as usize) * stripe_w[plane] + col as usize];
                                g = old * 23 + g * 22;
                                g = clip3(grain_min, grain_max, round2(g, 5));
                            }
                            if (col as usize) < stripe_w[plane] {
                                noise_stripe[luma_num][plane]
                                    [(i as usize) * stripe_w[plane] + col as usize] = g;
                            }
                        }
                    }
                }
            }
            x_block += 16;
        }
        luma_num += 1;
        y_stripe += 16;
    }

    // ----- Assemble noiseImage per plane -----
    // av1-spec p.345-346 lines 19023-19051.
    let mut noise_image: Vec<Vec<i32>> = Vec::with_capacity(np);
    for plane in 0..np {
        let psx = if plane > 0 { sub_x } else { 0 };
        let psy = if plane > 0 { sub_y } else { 0 };
        let plane_h = ((h + u32::from(psy)) >> psy) as usize;
        let plane_w = ((w + u32::from(psx)) >> psx) as usize;
        let mut img = vec![0i32; plane_h * plane_w];
        for yy in 0..plane_h {
            let stripe_idx = yy >> (5 - psy);
            let i = yy - (stripe_idx << (5 - psy));
            if stripe_idx >= noise_stripe.len() {
                continue;
            }
            for xx in 0..plane_w {
                let mut g = noise_stripe[stripe_idx][plane][i * stripe_w[plane] + xx];
                if psy == 0 {
                    if i < 2 && stripe_idx > 0 && fg.overlap_flag {
                        let old =
                            noise_stripe[stripe_idx - 1][plane][(i + 32) * stripe_w[plane] + xx];
                        g = if i == 0 {
                            old * 27 + g * 17
                        } else {
                            old * 17 + g * 27
                        };
                        g = clip3(grain_min, grain_max, round2(g, 5));
                    }
                } else if i < 1 && stripe_idx > 0 && fg.overlap_flag {
                    let old = noise_stripe[stripe_idx - 1][plane][(i + 16) * stripe_w[plane] + xx];
                    g = old * 23 + g * 22;
                    g = clip3(grain_min, grain_max, round2(g, 5));
                }
                img[yy * plane_w + xx] = g;
            }
        }
        noise_image.push(img);
    }

    // ----- Blend with output planes -----
    // av1-spec p.346-347 lines 19070-19137.
    let (min_value, max_luma, max_chroma) = if fg.clip_to_restricted_range {
        let mv = 16i32 << (bit_depth - 8);
        let ml = 235i32 << (bit_depth - 8);
        let mc = if matrix_coefficients == MC_IDENTITY {
            ml
        } else {
            240i32 << (bit_depth - 8)
        };
        (mv, ml, mc)
    } else {
        let mx = (256i32 << (bit_depth - 8)) - 1;
        (0, mx, mx)
    };
    let scaling_shift = u32::from(fg.grain_scaling);

    // Chroma blend first (uses averageLuma from OutY).
    if np >= 3 && (fg.num_cb_points > 0 || fg.num_cr_points > 0 || fg.chroma_scaling_from_luma) {
        let plane_w_c = ((w + u32::from(sub_x)) >> sub_x) as usize;
        let plane_h_c = ((h + u32::from(sub_y)) >> sub_y) as usize;
        for yy in 0..plane_h_c {
            for xx in 0..plane_w_c {
                let luma_x = xx << sub_x;
                let luma_y = yy << sub_y;
                let luma_next_x = (luma_x + 1).min((w as usize) - 1);
                let out_y = &out_planes[0];
                let y_at =
                    |x: usize, y: usize| -> i32 { out_y.samples[y * (out_y.cols as usize) + x] };
                let avg_luma = if sub_x != 0 {
                    round2(y_at(luma_x, luma_y) + y_at(luma_next_x, luma_y), 1)
                } else {
                    y_at(luma_x, luma_y)
                };

                // Cb path.
                if fg.num_cb_points > 0 || fg.chroma_scaling_from_luma {
                    let orig = out_planes[1].samples[yy * (out_planes[1].cols as usize) + xx];
                    let merged = if fg.chroma_scaling_from_luma {
                        avg_luma
                    } else {
                        let combined = avg_luma * (i32::from(fg.cb_luma_mult) - 128)
                            + orig * (i32::from(fg.cb_mult) - 128);
                        clip1(
                            bit_depth,
                            (combined >> 6) + ((i32::from(fg.cb_offset) - 256) << (bit_depth - 8)),
                        )
                    };
                    let noise = noise_image[1][yy * plane_w_c + xx];
                    let noise = round2(scale_lut(lut, 1, merged, bit_depth) * noise, scaling_shift);
                    out_planes[1].samples[yy * (out_planes[1].cols as usize) + xx] =
                        clip3(min_value, max_chroma, orig + noise);
                }

                // Cr path.
                if fg.num_cr_points > 0 || fg.chroma_scaling_from_luma {
                    let orig = out_planes[2].samples[yy * (out_planes[2].cols as usize) + xx];
                    let merged = if fg.chroma_scaling_from_luma {
                        avg_luma
                    } else {
                        let combined = avg_luma * (i32::from(fg.cr_luma_mult) - 128)
                            + orig * (i32::from(fg.cr_mult) - 128);
                        clip1(
                            bit_depth,
                            (combined >> 6) + ((i32::from(fg.cr_offset) - 256) << (bit_depth - 8)),
                        )
                    };
                    let noise = noise_image[2][yy * plane_w_c + xx];
                    let noise = round2(scale_lut(lut, 2, merged, bit_depth) * noise, scaling_shift);
                    out_planes[2].samples[yy * (out_planes[2].cols as usize) + xx] =
                        clip3(min_value, max_chroma, orig + noise);
                }
            }
        }
    }

    // Luma blend.
    if fg.num_y_points > 0 {
        for yy in 0..(h as usize) {
            for xx in 0..(w as usize) {
                let orig = out_planes[0].samples[yy * (out_planes[0].cols as usize) + xx];
                let noise = noise_image[0][yy * (w as usize) + xx];
                let noise = round2(scale_lut(lut, 0, orig, bit_depth) * noise, scaling_shift);
                out_planes[0].samples[yy * (out_planes[0].cols as usize) + xx] =
                    clip3(min_value, max_luma, orig + noise);
            }
        }
    }
}

// =====================================================================
// §7.18.3.1 top-level driver — av1-spec p.338.
// =====================================================================

/// §7.18.3.1 film grain synthesis driver — av1-spec p.338 lines
/// 18666-18689.
///
/// Wires the §7.18.3.3 / §7.18.3.4 / §7.18.3.5 sub-processes:
///
///   1. Generate `LumaGrain` / `CbGrain` / `CrGrain` via
///      [`generate_grain`].
///   2. Build `ScalingLut[]` via [`scaling_lookup_init`].
///   3. Blend the noise into `out_planes` via
///      [`add_noise_synthesis`].
///
/// On `!fg.apply_grain` the call is a no-op — the spec gates the
/// entire §7.18.3 process on `apply_grain == 1`.
pub fn film_grain_synthesis(
    fg: &FilmGrainParams,
    bit_depth: u8,
    num_planes: u8,
    sub_x: u8,
    sub_y: u8,
    matrix_coefficients: u8,
    out_planes: &mut [PlaneBuffer<'_>],
) {
    if !fg.apply_grain {
        return;
    }
    let grain = generate_grain(fg, bit_depth, sub_x, sub_y);
    let lut = scaling_lookup_init(fg, num_planes);
    add_noise_synthesis(
        fg,
        &grain,
        &lut,
        bit_depth,
        num_planes,
        sub_x,
        sub_y,
        matrix_coefficients,
        out_planes,
    );
}

// =====================================================================
// Tests — pin the LFSR, scaling lookup, grain build, and end-to-end
// no-op / identity paths.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncompressed_header_tail::FilmGrainParams;

    #[test]
    fn lfsr_first_three_draws_for_seed_one() {
        // av1-spec p.339 LFSR with seed = 1, draw 11 bits:
        //   r = 1
        //   bit = (1 ^ 0 ^ 0 ^ 0) & 1 = 1
        //   r' = (1 >> 1) | (1 << 15) = 0x8000
        //   result = (0x8000 >> 5) & ((1 << 11) - 1) = 0x400
        let mut rr = RandomRegister::new(1);
        let v0 = rr.next(11);
        assert_eq!(v0, 0x400);
        // Now state = 0x8000.
        //   bit = ((0x8000) ^ (0x4000) ^ (0x1000) ^ (0x8)) & 1 = 0
        //   r' = 0x4000 | 0 = 0x4000
        //   result = (0x4000 >> 5) & 0x7ff = 0x200
        let v1 = rr.next(11);
        assert_eq!(v1, 0x200);
    }

    #[test]
    fn lfsr_eight_bit_draw_matches_top_byte() {
        // Seed = 0xff00. r = 0xff00.
        //   bit = (0xff00 ^ 0x7f80 ^ 0x1fe0 ^ 0xf) & 1 = 1
        //   r' = 0x7f80 | 0x8000 = 0xff80
        //   8-bit result = (0xff80 >> 8) & 0xff = 0xff
        let mut rr = RandomRegister::new(0xff00);
        assert_eq!(rr.next(8), 0xff);
        assert_eq!(rr.state, 0xff80);
    }

    #[test]
    fn scaling_lookup_zero_num_points_is_zero_table() {
        let fg = FilmGrainParams::reset();
        let lut = scaling_lookup_init(&fg, 3);
        for plane in 0..3 {
            for x in 0..256 {
                assert_eq!(lut.tables[plane][x], 0);
            }
        }
    }

    #[test]
    fn scaling_lookup_two_point_ramp_interpolates() {
        // Two luma points at (0, 0) and (255, 255): the table is a
        // monotone ramp through every value.
        let mut fg = FilmGrainParams::reset();
        fg.apply_grain = true;
        fg.num_y_points = 2;
        fg.point_y_value[0] = 0;
        fg.point_y_scaling[0] = 0;
        fg.point_y_value[1] = 255;
        fg.point_y_scaling[1] = 255;
        let lut = scaling_lookup_init(&fg, 1);
        assert_eq!(lut.tables[0][0], 0);
        // The §7.18.3.4 reciprocal rounds: at x = 128, expect ~128.
        assert!((lut.tables[0][128] as i32 - 128).abs() <= 1);
        // Right plateau:
        assert_eq!(lut.tables[0][255], 255);
    }

    #[test]
    fn scaling_lookup_left_plateau_holds() {
        // First point at value=32, scaling=64 — all entries [0..32)
        // hold scaling=64.
        let mut fg = FilmGrainParams::reset();
        fg.apply_grain = true;
        fg.num_y_points = 2;
        fg.point_y_value[0] = 32;
        fg.point_y_scaling[0] = 64;
        fg.point_y_value[1] = 255;
        fg.point_y_scaling[1] = 200;
        let lut = scaling_lookup_init(&fg, 1);
        for x in 0..32 {
            assert_eq!(lut.tables[0][x], 64, "left plateau at {}", x);
        }
        assert_eq!(lut.tables[0][255], 200);
    }

    #[test]
    fn generate_grain_zero_when_num_y_points_zero() {
        // §7.18.3.3 white noise pass: num_y_points == 0 ⇒ g = 0;
        // and Round2(0, shift) is 0. The AR pass is gated on
        // num_y_points > 0, so LumaGrain stays all-zero.
        let mut fg = FilmGrainParams::reset();
        fg.apply_grain = true;
        fg.grain_seed = 12345;
        fg.grain_scale_shift = 0;
        let g = generate_grain(&fg, 8, 1, 1);
        assert!(g.luma.iter().all(|&v| v == 0));
        // Cb / Cr also gated on num_cb_points / num_cr_points and
        // chroma_scaling_from_luma — all false ⇒ zero.
        assert!(g.cb.iter().all(|&v| v == 0));
        assert!(g.cr.iter().all(|&v| v == 0));
        // Chroma dims for 4:2:0.
        assert_eq!(g.chroma_w, 44);
        assert_eq!(g.chroma_h, 38);
    }

    #[test]
    fn generate_grain_luma_first_sample_matches_round2_of_first_gauss_draw() {
        // For num_y_points > 0 the first written sample is
        //   LumaGrain[0][0] = Round2(Gaussian_Sequence[draw0],
        //                            12 - BitDepth + grain_scale_shift)
        // with draw0 from RandomRegister(grain_seed).next(11).
        let mut fg = FilmGrainParams::reset();
        fg.apply_grain = true;
        fg.grain_seed = 1;
        fg.num_y_points = 2;
        fg.point_y_value[0] = 0;
        fg.point_y_scaling[0] = 0;
        fg.point_y_value[1] = 255;
        fg.point_y_scaling[1] = 255;
        fg.grain_scale_shift = 0;
        // ar_coeff_lag = 0 ⇒ AR pass is a no-op (no neighbours).
        fg.ar_coeff_lag = 0;
        fg.ar_coeff_shift = 6;
        let g = generate_grain(&fg, 8, 1, 1);
        // Recompute the expected value.
        let mut rr = RandomRegister::new(1);
        let draw0 = rr.next(11) as usize;
        let gauss = i32::from(GAUSSIAN_SEQUENCE[draw0]);
        // shift = 12 - 8 + 0 = 4.
        let expected = round2(gauss, 4);
        assert_eq!(g.luma(0, 0), expected);
    }

    #[test]
    fn film_grain_synthesis_no_apply_is_noop() {
        let fg = FilmGrainParams::reset();
        // apply_grain = false (from reset()).
        let cols = 64u32;
        let rows = 32u32;
        let n = (cols * rows) as usize;
        let mut y = vec![100i32; n];
        let mut u = vec![100i32; n];
        let mut v = vec![100i32; n];
        let mut planes = [
            PlaneBuffer {
                rows,
                cols,
                samples: &mut y,
            },
            PlaneBuffer {
                rows,
                cols,
                samples: &mut u,
            },
            PlaneBuffer {
                rows,
                cols,
                samples: &mut v,
            },
        ];
        film_grain_synthesis(&fg, 8, 3, 0, 0, MC_IDENTITY, &mut planes);
        // No mutation.
        for &s in planes[0].samples.iter() {
            assert_eq!(s, 100);
        }
    }

    #[test]
    fn film_grain_synthesis_zero_y_points_leaves_luma_unchanged() {
        // apply_grain = 1, num_y_points = 0 ⇒ ScalingLut[0] = 0
        // and the §7.18.3.5 luma blend is gated on num_y_points > 0.
        // Luma must come out unchanged; chroma must also be unchanged
        // (num_cb_points = num_cr_points = chroma_scaling_from_luma =
        // 0).
        let mut fg = FilmGrainParams::reset();
        fg.apply_grain = true;
        fg.grain_seed = 9999;
        fg.grain_scaling = 8;
        fg.ar_coeff_shift = 6;
        let cols = 32u32;
        let rows = 32u32;
        let n = (cols * rows) as usize;
        let mut y = vec![128i32; n];
        let mut u = vec![64i32; n];
        let mut v = vec![64i32; n];
        let mut planes = [
            PlaneBuffer {
                rows,
                cols,
                samples: &mut y,
            },
            PlaneBuffer {
                rows,
                cols,
                samples: &mut u,
            },
            PlaneBuffer {
                rows,
                cols,
                samples: &mut v,
            },
        ];
        film_grain_synthesis(&fg, 8, 3, 0, 0, MC_IDENTITY, &mut planes);
        for &s in planes[0].samples.iter() {
            assert_eq!(s, 128);
        }
        for &s in planes[1].samples.iter() {
            assert_eq!(s, 64);
        }
        for &s in planes[2].samples.iter() {
            assert_eq!(s, 64);
        }
    }

    #[test]
    fn scale_lut_8bit_is_direct_indexing() {
        let mut lut = ScalingLut::default();
        lut.tables[0][7] = 42;
        lut.tables[0][255] = 99;
        assert_eq!(scale_lut(&lut, 0, 7, 8), 42);
        assert_eq!(scale_lut(&lut, 0, 255, 8), 99);
    }

    #[test]
    fn scale_lut_10bit_interpolates() {
        let mut lut = ScalingLut::default();
        // For BitDepth=10, shift = 2; index 9 ⇒ x = 9 >> 2 = 2, rem = 1.
        // table[2] = 100, table[3] = 200 ⇒ start + Round2((end -
        // start) * rem, shift) = 100 + Round2(100, 2) = 100 + 25 = 125.
        lut.tables[0][2] = 100;
        lut.tables[0][3] = 200;
        assert_eq!(scale_lut(&lut, 0, 9, 10), 125);
    }

    #[test]
    fn film_grain_synthesis_luma_ramp_mutates_samples() {
        // Full pipeline smoke: apply_grain = 1, luma scaling ramp
        // (255, 255), grain_seed = 7, ar_coeff_lag = 0 ⇒ purely
        // Gaussian noise blended with constant input. Output should
        // differ from the input on at least some samples (the
        // §7.18.3.3 Gaussian source mixes positive + negative
        // entries; the chance every blended sample collapses to the
        // input is vanishing).
        let mut fg = FilmGrainParams::reset();
        fg.apply_grain = true;
        fg.grain_seed = 7;
        fg.num_y_points = 2;
        fg.point_y_value[0] = 0;
        fg.point_y_scaling[0] = 64;
        fg.point_y_value[1] = 255;
        fg.point_y_scaling[1] = 64;
        fg.grain_scaling = 8;
        fg.ar_coeff_lag = 0;
        fg.ar_coeff_shift = 6;
        let cols = 64u32;
        let rows = 64u32;
        let n = (cols * rows) as usize;
        let mut y = vec![128i32; n];
        // Monochrome: just luma.
        let mut planes = [PlaneBuffer {
            rows,
            cols,
            samples: &mut y,
        }];
        film_grain_synthesis(&fg, 8, 1, 0, 0, MC_IDENTITY, &mut planes);
        let mutated = planes[0].samples.iter().any(|&s| s != 128);
        assert!(mutated, "film grain blend must touch at least some samples");
        // All samples remain in legal range.
        for &s in planes[0].samples.iter() {
            assert!((0..256).contains(&s));
        }
    }
}
