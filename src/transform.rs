//! §7.13 Inverse transform process (av1-spec p.295-307).
//!
//! This module implements the AV1 §7.13 inverse-transform stack: the
//! §7.13.2 1D butterflies (`B`, `H`, `cos128`, `sin128`, `brev`), the
//! §7.13.2.3 inverse DCT recursion (sizes 4..64 in powers of two), the
//! §7.13.2.6 / §7.13.2.7 / §7.13.2.8 / §7.13.2.9 inverse ADST stack
//! (sizes 4, 8, 16), the §7.13.2.10 inverse Walsh-Hadamard transform
//! (4-point only), the §7.13.2.11..§7.13.2.15 inverse identity stack
//! (sizes 4, 8, 16, 32), and the §7.13.3 2D inverse transform
//! dispatcher that runs the row-then-column composition over the 16
//! `(rowKind, colKind)` `TxType` combinations.
//!
//! ## Provenance
//!
//! Every value, table, butterfly schedule, and step in this file is
//! reproduced from the AV1 specification (`docs/video/av1/av1-spec.txt`
//! / `av1-spec.pdf`) §7.13.x. The `Cos128_Lookup[65]` table is the
//! spec's §7.13.2.1 quarter-period cosine table; the
//! butterfly-step schedules in [`inverse_dct`], [`inverse_adst8`],
//! [`inverse_adst16`] reproduce the spec's `B(...)` / `H(...)` step
//! lists verbatim from §7.13.2.3 / §7.13.2.7 / §7.13.2.8. The
//! §7.13.2.4 / §7.13.2.5 input / output permutations and the
//! §7.13.2.6 inverse ADST4 closed-form sum-of-`SINPI_*` formula are
//! transcribed unchanged. The §7.13.3 2D dispatcher reproduces the
//! `Transform_Row_Shift[]` table and the row-clamp / col-clamp
//! envelope from the spec text directly.
//!
//! ## Scope landed in round 182
//!
//! * §7.13.2.1 `B` (`flip = 0` and `flip = 1`), `H` (`flip = 0` and
//!   `flip = 1`), `cos128`, `sin128`, `brev`.
//! * §7.13.2.2 inverse DCT permutation (`brev`-based in-place
//!   reorder).
//! * §7.13.2.3 inverse DCT for `n in 2..=6` (sizes 4, 8, 16, 32, 64).
//! * §7.13.2.4 inverse ADST input permutation.
//! * §7.13.2.5 inverse ADST output permutation.
//! * §7.13.2.6 inverse ADST4 (closed-form, `SINPI_*`).
//! * §7.13.2.7 inverse ADST8 (butterfly schedule).
//! * §7.13.2.8 inverse ADST16 (butterfly schedule).
//! * §7.13.2.9 inverse ADST dispatcher.
//! * §7.13.2.10 inverse WHT4.
//! * §7.13.2.11..§7.13.2.14 inverse identity 4 / 8 / 16 / 32.
//! * §7.13.2.15 inverse identity dispatcher.
//! * §7.13.3 2D inverse transform dispatcher, including:
//!   * `Transform_Row_Shift[TX_SIZES_ALL]` table.
//!   * `Abs(log2W - log2H) == 1` rectangular post-scale by `2896`.
//!   * `Lossless` short-circuit through the WHT path (shift 2 row /
//!     shift 0 col).
//!   * Per-`PlaneTxType` row / column selector arms (DCT_DCT,
//!     ADST_DCT, DCT_ADST, ADST_ADST, FLIPADST_DCT, DCT_FLIPADST,
//!     FLIPADST_FLIPADST, ADST_FLIPADST, FLIPADST_ADST, IDTX,
//!     V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST).
//!   * Between-row-and-column clamp at `colClampRange =
//!     Max(BitDepth + 6, 16)`.
//!   * Per-row Round2 by `rowShift` and per-column Round2 by
//!     `colShift = Lossless ? 0 : 4`.
//!
//! The §7.13.3 "FLIP" reordering is performed in §7.12.3
//! step 3 (the frame-buffer write with `xx = flipLR ? w - j - 1 : j`,
//! `yy = flipUD ? h - i - 1 : i`) — NOT in this module. §7.13.3
//! itself runs the FLIPADST butterfly the same way as ADST; the flip
//! is purely a destination-coordinate transform on the
//! `CurrFrame[plane][y + yy][x + xx]` write. The §7.12.3 write is
//! split off to the next arc (the §7.12.3 frame-buffer wiring).

use crate::cdf::{
    ADST_ADST, ADST_DCT, ADST_FLIPADST, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_ADST,
    FLIPADST_DCT, FLIPADST_FLIPADST, H_ADST, H_DCT, H_FLIPADST, IDTX, TX_HEIGHT, TX_SIZES_ALL,
    TX_WIDTH, TX_WIDTH_LOG2, V_ADST, V_DCT, V_FLIPADST,
};

/// `Cos128_Lookup[65]` (§7.13.2.1 av1-spec p.296). Quarter-period
/// table: `Cos128_Lookup[k] = round(4096 * cos(k * pi / 128))` for
/// `k in 0..=64`. Reproduced verbatim from the spec text.
pub const COS128_LOOKUP: [i32; 65] = [
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036, 4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461, 3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359, 2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897, 799, 700, 601, 501, 401, 301, 201, 101, 0,
];

/// `SINPI_1_9` per §7.13.2.6 (av1-spec p.302). The §7.13.2.6 inverse
/// ADST4 closed-form uses these four constants directly.
pub const SINPI_1_9: i32 = 1321;
/// `SINPI_2_9` per §7.13.2.6 (av1-spec p.302).
pub const SINPI_2_9: i32 = 2482;
/// `SINPI_3_9` per §7.13.2.6 (av1-spec p.302).
pub const SINPI_3_9: i32 = 3344;
/// `SINPI_4_9` per §7.13.2.6 (av1-spec p.302).
pub const SINPI_4_9: i32 = 3803;

/// `Transform_Row_Shift[TX_SIZES_ALL]` (§7.13.3 av1-spec p.307). Per
/// `txSz` ordinal, the right-shift amount applied to the row-transform
/// output before the between-stage clamp. Indexed by the
/// [`crate::cdf::TX_SIZES_ALL`] enumeration.
pub const TRANSFORM_ROW_SHIFT: [u32; TX_SIZES_ALL] = [
    0, // TX_4X4
    1, // TX_8X8
    2, // TX_16X16
    2, // TX_32X32
    2, // TX_64X64
    0, // TX_4X8
    0, // TX_8X4
    1, // TX_8X16
    1, // TX_16X8
    1, // TX_16X32
    1, // TX_32X16
    1, // TX_32X64
    1, // TX_64X32
    1, // TX_4X16
    1, // TX_16X4
    2, // TX_8X32
    2, // TX_32X8
    2, // TX_16X64
    2, // TX_64X16
];

/// §4.7.2 `Round2( x, n )` (av1-spec p.18). When `n == 0` returns `x`
/// unchanged; otherwise returns `(x + (1 << (n - 1))) >> n` with
/// signed arithmetic shift. Implemented over `i64` to keep the
/// intermediate `x + half` from overflowing `i32` in the larger
/// butterfly schedules.
#[inline]
pub fn round2(x: i64, n: u32) -> i64 {
    if n == 0 {
        return x;
    }
    let half: i64 = 1i64 << (n - 1);
    (x + half) >> n
}

/// §4.7.2 `Clip3( low, high, x )` (av1-spec p.18). Standard clamp.
#[inline]
pub fn clip3(low: i64, high: i64, x: i64) -> i64 {
    if x < low {
        low
    } else if x > high {
        high
    } else {
        x
    }
}

/// §7.13.2.1 `brev( numBits, x )` (av1-spec p.296). Bit-reversal of
/// the low `num_bits` of `x`.
#[inline]
pub fn brev(num_bits: u32, x: u32) -> u32 {
    let mut t = 0u32;
    for i in 0..num_bits {
        let bit = (x >> i) & 1;
        t += bit << (num_bits - 1 - i);
    }
    t
}

/// §7.13.2.1 `cos128( angle )` (av1-spec p.296). Implements
/// `4096 * cos(angle * pi / 128)` rounded to the nearest integer via
/// the four-arm dispatch over `angle & 255`.
#[inline]
pub fn cos128(angle: i32) -> i32 {
    let angle2 = (angle as i64 & 255) as i32;
    if (0..=64).contains(&angle2) {
        COS128_LOOKUP[angle2 as usize]
    } else if (65..=128).contains(&angle2) {
        -COS128_LOOKUP[(128 - angle2) as usize]
    } else if (129..=192).contains(&angle2) {
        -COS128_LOOKUP[(angle2 - 128) as usize]
    } else {
        COS128_LOOKUP[(256 - angle2) as usize]
    }
}

/// §7.13.2.1 `sin128( angle ) = cos128( angle - 64 )` (av1-spec p.297).
#[inline]
pub fn sin128(angle: i32) -> i32 {
    cos128(angle - 64)
}

/// §7.13.2.1 butterfly `B( a, b, angle, flip, r )` (av1-spec p.296-297).
/// Performs the in-place rotation `(T[a], T[b]) <- (Round2(x, 12),
/// Round2(y, 12))` with `x = T[a] * cos128(angle) - T[b] *
/// sin128(angle)`, `y = T[a] * sin128(angle) + T[b] * cos128(angle)`.
/// When `flip == 1`, swaps `T[a]` and `T[b]` after the rotation.
///
/// The `r` argument is the §7.13.2.1 bitstream-conformance precision
/// requirement (the result is required to fit in a signed integer
/// using `r` bits of precision); this implementation uses i64
/// intermediates and the spec's §4.7.2 `Round2` to produce the same
/// integer value regardless of `r`. The `_r` parameter is documented
/// for spec faithfulness but the runtime does not enforce the bound.
#[inline]
pub fn butterfly_b(t: &mut [i64], a: usize, b: usize, angle: i32, flip: u8, _r: u32) {
    let cos_a = cos128(angle) as i64;
    let sin_a = sin128(angle) as i64;
    let ta = t[a];
    let tb = t[b];
    let x = ta * cos_a - tb * sin_a;
    let y = ta * sin_a + tb * cos_a;
    t[a] = round2(x, 12);
    t[b] = round2(y, 12);
    if flip == 1 {
        t.swap(a, b);
    }
}

/// §7.13.2.1 Hadamard `H( a, b, flip, r )` (av1-spec p.297). For
/// `flip == 0`, `(T[a], T[b]) <- (Clip3(x+y), Clip3(x-y))` where the
/// clamp is `[-2^(r-1), 2^(r-1) - 1]`. For `flip == 1`, performs
/// `H(b, a, 0, r)` (i.e. flips the input argument order).
#[inline]
pub fn butterfly_h(t: &mut [i64], a: usize, b: usize, flip: u8, r: u32) {
    let (ai, bi) = if flip == 1 { (b, a) } else { (a, b) };
    let x = t[ai];
    let y = t[bi];
    let lo: i64 = -(1i64 << (r - 1));
    let hi: i64 = (1i64 << (r - 1)) - 1;
    t[ai] = clip3(lo, hi, x + y);
    t[bi] = clip3(lo, hi, x - y);
}

/// §7.13.2.2 inverse DCT permutation (av1-spec p.297). Permutes the
/// length-`2^n` array `T` in place via `T[i] = copyT[brev(n, i)]`.
pub fn inverse_dct_permute(t: &mut [i64], n: u32) {
    let len = 1usize << n;
    // r475 — a stack copy + the precomputed bit-reversal table (the
    // per-call heap copy was 7 % of a 12 MP still decode).
    let mut copy = [0i64; 64];
    copy[..len].copy_from_slice(&t[..len]);
    let table = &BREV_TABLE[n.min(6) as usize];
    for (i, slot) in t.iter_mut().take(len).enumerate() {
        *slot = copy[table[i] as usize];
    }
}

/// `brev( n, i )` for `n ∈ 0..=6`, `i < 1 << n` — the §7.13.2.2 bit
/// reversal the DCT permutation step indexes by.
const BREV_TABLE: [[u8; 64]; 7] = {
    let mut table = [[0u8; 64]; 7];
    let mut n = 0;
    while n < 7 {
        let len = 1usize << n;
        let mut i = 0;
        while i < len {
            let mut t = 0u32;
            let mut b = 0;
            while b < n {
                t += ((i as u32 >> b) & 1) << (n - 1 - b);
                b += 1;
            }
            table[n][i] = t as u8;
            i += 1;
        }
        n += 1;
    }
    table
};

/// §7.13.2.3 inverse DCT process for arrays of length `2^n` with
/// `2 <= n <= 6` (av1-spec p.297-300). Performs the §7.13.2.2
/// permutation then the 31-step butterfly schedule per the spec
/// text. `r` is the per-stage clamping range (§7.13.3 dispatcher
/// passes `rowClampRange` for the row pass and `colClampRange` for
/// the column pass).
#[allow(clippy::too_many_arguments)]
pub fn inverse_dct(t: &mut [i64], n: u32, r: u32) {
    debug_assert!((2..=6).contains(&n));
    inverse_dct_permute(t, n);
    inverse_dct_flow(&mut DirectFlow { t, r }, n);
}

/// The two butterfly kinds of the §7.13.2.3 schedule, as the schedule
/// invokes them: `rotate` is the §7.13.2.1 `B( a, b, angle, flip )`
/// and `add_sub` the `H( a, b, flip )`.
trait DctFlow {
    fn rotate(&mut self, a: usize, b: usize, angle: i32, flip: u8);
    fn add_sub(&mut self, a: usize, b: usize, flip: u8);
}

/// The schedule applied straight to one array (the spec's reading).
struct DirectFlow<'a> {
    t: &'a mut [i64],
    r: u32,
}

impl DctFlow for DirectFlow<'_> {
    #[inline(always)]
    fn rotate(&mut self, a: usize, b: usize, angle: i32, flip: u8) {
        butterfly_b(self.t, a, b, angle, flip, self.r);
    }
    #[inline(always)]
    fn add_sub(&mut self, a: usize, b: usize, flip: u8) {
        butterfly_h(self.t, a, b, flip, self.r);
    }
}

/// The §7.13.2.3 butterfly schedule (everything after the §7.13.2.2
/// input permutation) over any [`DctFlow`].
fn inverse_dct_flow<F: DctFlow>(f: &mut F, n: u32) {
    // Step 1: permutation.
    // Step 2: n == 6 only.
    if n == 6 {
        for i in 0..16 {
            let a = 32 + i;
            let b = 63 - i;
            let angle = 63 - 4 * (brev(4, i as u32) as i32);
            f.rotate(a, b, angle, 0);
        }
    }
    // Step 3: n >= 5.
    if n >= 5 {
        for i in 0..8 {
            let a = 16 + i;
            let b = 31 - i;
            let angle = 6 + ((brev(3, (7 - i) as u32) as i32) << 3);
            f.rotate(a, b, angle, 0);
        }
    }
    // Step 4: n == 6.
    if n == 6 {
        for i in 0..16 {
            f.add_sub(32 + i * 2, 33 + i * 2, (i & 1) as u8);
        }
    }
    // Step 5: n >= 4.
    if n >= 4 {
        for i in 0..4 {
            let a = 8 + i;
            let b = 15 - i;
            let angle = 12 + ((brev(2, (3 - i) as u32) as i32) << 4);
            f.rotate(a, b, angle, 0);
        }
    }
    // Step 6: n >= 5.
    if n >= 5 {
        for i in 0..8 {
            f.add_sub(16 + 2 * i, 17 + 2 * i, (i & 1) as u8);
        }
    }
    // Step 7: n == 6.
    if n == 6 {
        for i in 0..4 {
            for j in 0..2 {
                let a = 62 - i * 4 - j;
                let b = 33 + i * 4 + j;
                let angle = 60 - 16 * (brev(2, i as u32) as i32) + 64 * (j as i32);
                f.rotate(a, b, angle, 1);
            }
        }
    }
    // Step 8: n >= 3.
    if n >= 3 {
        for i in 0..2 {
            f.rotate(4 + i, 7 - i, 56 - 32 * (i as i32), 0);
        }
    }
    // Step 9: n >= 4.
    if n >= 4 {
        for i in 0..4 {
            f.add_sub(8 + 2 * i, 9 + 2 * i, (i & 1) as u8);
        }
    }
    // Step 10: n >= 5.
    if n >= 5 {
        for i in 0..2 {
            for j in 0..2 {
                let a = 30 - 4 * i - j;
                let b = 17 + 4 * i + j;
                let angle = 24 + ((j as i32) << 6) + (((1 - i) as i32) << 5);
                f.rotate(a, b, angle, 1);
            }
        }
    }
    // Step 11: n == 6.
    if n == 6 {
        for i in 0..8 {
            for j in 0..2 {
                let a = 32 + i * 4 + j;
                let b = 35 + i * 4 - j;
                f.add_sub(a, b, (i & 1) as u8);
            }
        }
    }
    // Step 12: always.
    for i in 0..2 {
        f.rotate(2 * i, 2 * i + 1, 32 + 16 * (i as i32), (1 - i) as u8);
    }
    // Step 13: n >= 3.
    if n >= 3 {
        for i in 0..2 {
            f.add_sub(4 + 2 * i, 5 + 2 * i, i as u8);
        }
    }
    // Step 14: n >= 4.
    if n >= 4 {
        for i in 0..2 {
            f.rotate(14 - i, 9 + i, 48 + 64 * (i as i32), 1);
        }
    }
    // Step 15: n >= 5.
    if n >= 5 {
        for i in 0..4 {
            for j in 0..2 {
                let a = 16 + 4 * i + j;
                let b = 19 + 4 * i - j;
                f.add_sub(a, b, (i & 1) as u8);
            }
        }
    }
    // Step 16: n == 6.
    if n == 6 {
        for i in 0..2usize {
            for j in 0..4usize {
                let a = 61 - i * 8 - j;
                let b = 34 + i * 8 + j;
                let angle: i32 = 56 - (i as i32) * 32 + ((j >> 1) as i32) * 64;
                f.rotate(a, b, angle, 1);
            }
        }
    }
    // Step 17: always.
    for i in 0..2 {
        f.add_sub(i, 3 - i, 0);
    }
    // Step 18: n >= 3.
    if n >= 3 {
        f.rotate(6, 5, 32, 1);
    }
    // Step 19: n >= 4.
    if n >= 4 {
        for i in 0..2 {
            for j in 0..2 {
                f.add_sub(8 + 4 * i + j, 11 + 4 * i - j, i as u8);
            }
        }
    }
    // Step 20: n >= 5.
    if n >= 5 {
        for i in 0..4 {
            f.rotate(29 - i, 18 + i, 48 + ((i >> 1) as i32) * 64, 1);
        }
    }
    // Step 21: n == 6.
    if n == 6 {
        for i in 0..4 {
            for j in 0..4 {
                let a = 32 + 8 * i + j;
                let b = 39 + 8 * i - j;
                f.add_sub(a, b, (i & 1) as u8);
            }
        }
    }
    // Step 22: n >= 3.
    if n >= 3 {
        for i in 0..4 {
            f.add_sub(i, 7 - i, 0);
        }
    }
    // Step 23: n >= 4.
    if n >= 4 {
        for i in 0..2 {
            f.rotate(13 - i, 10 + i, 32, 1);
        }
    }
    // Step 24: n >= 5.
    if n >= 5 {
        for i in 0..2 {
            for j in 0..4 {
                let a = 16 + i * 8 + j;
                let b = 23 + i * 8 - j;
                f.add_sub(a, b, i as u8);
            }
        }
    }
    // Step 25: n == 6.
    if n == 6 {
        for i in 0..8 {
            let angle = if i < 4 { 48 } else { 112 };
            f.rotate(59 - i, 36 + i, angle, 1);
        }
    }
    // Step 26: n >= 4.
    if n >= 4 {
        for i in 0..8 {
            f.add_sub(i, 15 - i, 0);
        }
    }
    // Step 27: n >= 5.
    if n >= 5 {
        for i in 0..4 {
            f.rotate(27 - i, 20 + i, 32, 1);
        }
    }
    // Step 28: n == 6.
    if n == 6 {
        for i in 0..8 {
            f.add_sub(32 + i, 47 - i, 0);
            f.add_sub(48 + i, 63 - i, 1);
        }
    }
    // Step 29: n >= 5.
    if n >= 5 {
        for i in 0..16 {
            f.add_sub(i, 31 - i, 0);
        }
    }
    // Step 30: n == 6.
    if n == 6 {
        for i in 0..8 {
            f.rotate(55 - i, 40 + i, 32, 1);
        }
    }
    // Step 31: n == 6.
    if n == 6 {
        for i in 0..32 {
            f.add_sub(i, 63 - i, 0);
        }
    }
}

// ---------------------------------------------------------------------
// r487 — the §7.13.2.3 schedule over several arrays at once.
//
// The 2-D inverse transform runs the same 1-D kernel on every row and
// then on every column. Recording the schedule once per size (each
// butterfly with its indices, flip and `cos128` / `sin128` factors
// resolved) and replaying it over `L` arrays held side by side — slot
// `i` of all `L` arrays in one `[_; L]` lane group — turns every
// butterfly into a fixed-width loop with no per-sample branch, and
// pays the schedule's own bookkeeping once per `L` rows / columns.
//
// Lanes are 64-bit by default (the reading of §7.13.2.1 the scalar
// kernel uses). A group whose inputs are small enough runs in 32-bit
// lanes instead: `DctProgram::narrow_limit` is a bound on the input
// magnitude under which NO value the schedule forms — neither a
// product, nor a sum of two products before its `Round2`, nor a sum
// before its clamp — can leave the signed 32-bit range, derived by
// propagating a magnitude bound through the recorded schedule. Under
// it both lane widths compute the same integers, so the choice never
// changes a sample.
// ---------------------------------------------------------------------

/// One recorded butterfly.
#[derive(Clone, Copy, Debug)]
enum FlowStep {
    /// §7.13.2.1 `B()`: `x = T[ a ] * cos - T[ b ] * sin`, `y = T[ a ]
    /// * sin + T[ b ] * cos`, each `Round2( _, 12 )`; `x` lands in
    /// slot `x_to` and `y` in `y_to` (the flip resolved).
    Rotate {
        a: u8,
        b: u8,
        cos: i32,
        sin: i32,
        x_to: u8,
        y_to: u8,
    },
    /// §7.13.2.1 `H()`: slot `sum` takes the clamped sum, slot `diff`
    /// the clamped difference `T[ sum ] - T[ diff ]` (the flip
    /// resolved).
    AddSub { sum: u8, diff: u8 },
}

/// The recorded §7.13.2.3 schedule of one size.
struct DctProgram {
    steps: Vec<FlowStep>,
    /// Largest input magnitude for which the 32-bit lanes are exact
    /// (see the section comment).
    narrow_limit: i64,
}

struct FlowRecorder(Vec<FlowStep>);

impl DctFlow for FlowRecorder {
    fn rotate(&mut self, a: usize, b: usize, angle: i32, flip: u8) {
        let (x_to, y_to) = if flip == 1 { (b, a) } else { (a, b) };
        self.0.push(FlowStep::Rotate {
            a: a as u8,
            b: b as u8,
            cos: cos128(angle),
            sin: sin128(angle),
            x_to: x_to as u8,
            y_to: y_to as u8,
        });
    }
    fn add_sub(&mut self, a: usize, b: usize, flip: u8) {
        let (sum, diff) = if flip == 1 { (b, a) } else { (a, b) };
        self.0.push(FlowStep::AddSub {
            sum: sum as u8,
            diff: diff as u8,
        });
    }
}

/// The recorded schedule for arrays of length `1 << n` (`2 <= n <= 6`).
fn dct_program(n: u32) -> &'static DctProgram {
    static PROGRAMS: [std::sync::OnceLock<DctProgram>; 5] = [
        std::sync::OnceLock::new(),
        std::sync::OnceLock::new(),
        std::sync::OnceLock::new(),
        std::sync::OnceLock::new(),
        std::sync::OnceLock::new(),
    ];
    debug_assert!((2..=6).contains(&n));
    PROGRAMS[(n - 2) as usize].get_or_init(|| {
        let mut rec = FlowRecorder(Vec::new());
        inverse_dct_flow(&mut rec, n);
        let steps = rec.0;
        // Magnitude bound per slot as `gain * m + slack` for inputs
        // of magnitude at most `m`. A rotation's pre-rounding sum is
        // at most `|cos| * B( a ) + |sin| * B( b )` (each product at
        // most its own term), its rounded output that over 4096 plus
        // one; an add / subtract is at most `B( sum ) + B( diff )`
        // before the clamp, which only shrinks it. Every such
        // quantity (plus the rounding offset) must stay below `2^31`.
        const CEILING: f64 = 2_147_483_647.0 - 2048.0 - 1.0;
        let mut gain = [1f64; 64];
        let mut slack = [0f64; 64];
        let mut limit = f64::INFINITY;
        for step in &steps {
            match *step {
                FlowStep::Rotate {
                    a,
                    b,
                    cos,
                    sin,
                    x_to,
                    y_to,
                } => {
                    let (c, s) = (f64::from(cos.abs()), f64::from(sin.abs()));
                    let g = c * gain[a as usize] + s * gain[b as usize];
                    let e = c * slack[a as usize] + s * slack[b as usize];
                    limit = limit.min((CEILING - e) / g);
                    for slot in [x_to, y_to] {
                        gain[slot as usize] = g / 4096.0;
                        slack[slot as usize] = e / 4096.0 + 1.0;
                    }
                }
                FlowStep::AddSub { sum, diff } => {
                    let g = gain[sum as usize] + gain[diff as usize];
                    let e = slack[sum as usize] + slack[diff as usize];
                    limit = limit.min((CEILING - e) / g);
                    for slot in [sum, diff] {
                        gain[slot as usize] = g;
                        slack[slot as usize] = e;
                    }
                }
            }
        }
        // A safety margin far above the floating-point error of the
        // propagation.
        let narrow_limit = ((limit * 0.999).floor() as i64 - 1).max(0);
        DctProgram {
            steps,
            narrow_limit,
        }
    })
}

macro_rules! dct_lane_runner {
    ($name:ident, $lane:ty) => {
        /// Replay a recorded §7.13.2.3 schedule over `L` arrays of
        /// length `N` held side by side (`t[ i ][ l ]` is slot `i` of
        /// array `l`; `N` is the schedule's own power-of-two length),
        /// clamping the add / subtract butterflies to `r` bits.
        #[inline(always)]
        fn $name<const L: usize, const N: usize>(
            steps: &[FlowStep],
            t: &mut [[$lane; L]; N],
            r: u32,
        ) {
            let lo = (-(1i64 << (r - 1))) as $lane;
            let hi = ((1i64 << (r - 1)) - 1) as $lane;
            for step in steps {
                match *step {
                    FlowStep::Rotate {
                        a,
                        b,
                        cos,
                        sin,
                        x_to,
                        y_to,
                    } => {
                        let (ta, tb) = (t[usize::from(a) & (N - 1)], t[usize::from(b) & (N - 1)]);
                        let (cos, sin) = (cos as $lane, sin as $lane);
                        let mut x = [0 as $lane; L];
                        let mut y = [0 as $lane; L];
                        for l in 0..L {
                            x[l] = (ta[l].wrapping_mul(cos))
                                .wrapping_sub(tb[l].wrapping_mul(sin))
                                .wrapping_add(2048)
                                >> 12;
                            y[l] = (ta[l].wrapping_mul(sin))
                                .wrapping_add(tb[l].wrapping_mul(cos))
                                .wrapping_add(2048)
                                >> 12;
                        }
                        t[usize::from(x_to) & (N - 1)] = x;
                        t[usize::from(y_to) & (N - 1)] = y;
                    }
                    FlowStep::AddSub { sum, diff } => {
                        let (x, y) = (
                            t[usize::from(sum) & (N - 1)],
                            t[usize::from(diff) & (N - 1)],
                        );
                        let mut s = [0 as $lane; L];
                        let mut d = [0 as $lane; L];
                        for l in 0..L {
                            s[l] = x[l].wrapping_add(y[l]).max(lo).min(hi);
                            d[l] = x[l].wrapping_sub(y[l]).max(lo).min(hi);
                        }
                        t[usize::from(sum) & (N - 1)] = s;
                        t[usize::from(diff) & (N - 1)] = d;
                    }
                }
            }
        }
    };
}
dct_lane_runner!(run_dct_lanes_wide, i64);
dct_lane_runner!(run_dct_lanes_narrow, i32);

/// The §7.13.2.3 inverse DCT of `L` arrays at once: `input[ i ][ l ]`
/// is element `i` of array `l` in NATURAL order (the §7.13.2.2
/// permutation happens here); the outputs replace it. `len = 1 << n`
/// slots are used.
#[inline(always)]
fn inverse_dct_lanes<const L: usize, const N: usize>(io: &mut [[i64; L]; N], r: u32) {
    let n = N.trailing_zeros();
    let program = dct_program(n);
    let order = &BREV_TABLE[n as usize];
    let mut max_abs: u64 = 0;
    for row in io.iter() {
        for &v in row {
            max_abs = max_abs.max(v.unsigned_abs());
        }
    }
    if max_abs <= program.narrow_limit as u64 {
        let mut t = [[0i32; L]; N];
        for (i, slot) in t.iter_mut().enumerate() {
            let src = &io[usize::from(order[i]) & (N - 1)];
            for l in 0..L {
                slot[l] = src[l] as i32;
            }
        }
        run_dct_lanes_narrow(&program.steps, &mut t, r);
        for (dst, src) in io.iter_mut().zip(&t) {
            for l in 0..L {
                dst[l] = i64::from(src[l]);
            }
        }
    } else {
        let mut t = [[0i64; L]; N];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = io[usize::from(order[i]) & (N - 1)];
        }
        run_dct_lanes_wide(&program.steps, &mut t, r);
        *io = t;
    }
}

/// The §7.13.3 row pass for the DCT row kernels, `L` rows at a time:
/// `Residual[ i ][ j ] = Round2( T[ j ], rowShift )` for every row `i`
/// below 32 (rows past it, and columns past 32 of the input, read as
/// zero); `rect_scale` applies the §7.13.3 `Round2( T[ j ] * 2896, 12
/// )` pre-scale of the 2:1 rectangles. `residual` arrives zeroed.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn dct_row_pass<const L: usize, const N: usize>(
    dequant: &[i64],
    residual: &mut [i64],
    h: usize,
    rect_scale: bool,
    r: u32,
    row_shift: u32,
) {
    let w = N;
    let in_w = w.min(32);
    let mut i0 = 0;
    while i0 < h.min(32) {
        let group = &dequant[i0 * w..(i0 + L) * w];
        if group.iter().all(|&v| v == 0) {
            // Zero rows transform to zero rows.
            i0 += L;
            continue;
        }
        let mut io = [[0i64; L]; N];
        for (j, slot) in io[..in_w].iter_mut().enumerate() {
            for l in 0..L {
                let v = group[l * w + j];
                slot[l] = if rect_scale { round2(v * 2896, 12) } else { v };
            }
        }
        inverse_dct_lanes(&mut io, r);
        for l in 0..L {
            let out = &mut residual[(i0 + l) * w..(i0 + l + 1) * w];
            for (j, cell) in out.iter_mut().enumerate() {
                *cell = round2(io[j][l], row_shift);
            }
        }
        i0 += L;
    }
}

/// The §7.13.3 column pass for the DCT column kernels, `L` columns at
/// a time, in place over `residual` (`Round2( T[ i ], colShift )`).
#[inline(always)]
fn dct_column_pass<const L: usize, const N: usize>(
    residual: &mut [i64],
    w: usize,
    r: u32,
    col_shift: u32,
) {
    let h = N;
    let log2_h = N.trailing_zeros();
    // Every column equal (a block whose rows are constant — the
    // DC-only block above all): one column, replicated.
    if residual
        .chunks_exact(w)
        .all(|row| row.iter().all(|&v| v == row[0]))
    {
        let mut col = [0i64; 64];
        for (i, row) in residual.chunks_exact(w).enumerate() {
            col[i] = row[0];
        }
        inverse_dct(&mut col[..h], log2_h, r);
        for (i, row) in residual.chunks_exact_mut(w).enumerate() {
            row.fill(round2(col[i], col_shift));
        }
        return;
    }
    let mut j0 = 0;
    while j0 < w {
        let mut io = [[0i64; L]; N];
        for (i, slot) in io.iter_mut().enumerate() {
            slot.copy_from_slice(&residual[i * w + j0..i * w + j0 + L]);
        }
        inverse_dct_lanes(&mut io, r);
        for (i, slot) in io.iter().enumerate() {
            let out = &mut residual[i * w + j0..i * w + j0 + L];
            for l in 0..L {
                out[l] = round2(slot[l], col_shift);
            }
        }
        j0 += L;
    }
}

/// §7.13.2.4 inverse ADST input permutation (av1-spec p.300).
/// `T[i] = copyT[idx]` with `idx = (i & 1) ? (i - 1) : (n0 - i - 1)`
/// for `i in 0..n0` where `n0 = 1 << n`. `3 <= n <= 4`.
pub fn inverse_adst_input_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let copy: Vec<i64> = t[..n0].to_vec();
    for (i, slot) in t[..n0].iter_mut().enumerate() {
        let idx = if i & 1 == 1 { i - 1 } else { n0 - i - 1 };
        *slot = copy[idx];
    }
}

/// §7.13.2.5 inverse ADST output permutation (av1-spec p.300-301).
/// Computes the 4-bit Gray-code-like index from `i`'s low bits and
/// reads `copyT[idx]`; odd `i` values negate the result.
pub fn inverse_adst_output_permute(t: &mut [i64], n: u32) {
    let n0 = 1usize << n;
    let copy: Vec<i64> = t[..n0].to_vec();
    for (i, slot) in t[..n0].iter_mut().enumerate() {
        let a = ((i >> 3) & 1) as u32;
        let b = (((i >> 2) & 1) ^ ((i >> 3) & 1)) as u32;
        let c = (((i >> 1) & 1) ^ ((i >> 2) & 1)) as u32;
        let d = ((i & 1) ^ ((i >> 1) & 1)) as u32;
        let idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
        let v = copy[idx as usize];
        *slot = if i & 1 == 1 { -v } else { v };
    }
}

/// §7.13.2.6 inverse ADST4 process (av1-spec p.301-302). In-place
/// transform of the length-4 array `T`. `r` is the precision argument
/// the spec documents for conformance — not enforced here.
pub fn inverse_adst4(t: &mut [i64], _r: u32) {
    let t0 = t[0];
    let t1 = t[1];
    let t2 = t[2];
    let t3 = t[3];
    let mut s = [0i64; 7];
    s[0] = SINPI_1_9 as i64 * t0;
    s[1] = SINPI_2_9 as i64 * t0;
    s[2] = SINPI_3_9 as i64 * t1;
    s[3] = SINPI_4_9 as i64 * t2;
    s[4] = SINPI_1_9 as i64 * t2;
    s[5] = SINPI_2_9 as i64 * t3;
    s[6] = SINPI_4_9 as i64 * t3;
    let a7 = t0 - t2;
    let b7 = a7 + t3;
    s[0] += s[3];
    s[1] -= s[4];
    s[3] = s[2];
    s[2] = SINPI_3_9 as i64 * b7;
    s[0] += s[5];
    s[1] -= s[6];
    let x0 = s[0] + s[3];
    let x1 = s[1] + s[3];
    let x2 = s[2];
    let x3 = s[0] + s[1] - s[3];
    t[0] = round2(x0, 12);
    t[1] = round2(x1, 12);
    t[2] = round2(x2, 12);
    t[3] = round2(x3, 12);
}

/// §7.13.2.7 inverse ADST8 process (av1-spec p.302-303). 7-step
/// butterfly schedule wrapped in §7.13.2.4 / §7.13.2.5 permutations.
pub fn inverse_adst8(t: &mut [i64], r: u32) {
    // Step 1: input permutation with n = 3.
    inverse_adst_input_permute(t, 3);
    // Step 2: B(2i, 2i+1, 60 - 16i, 1, r) for i = 0..3.
    for i in 0..4 {
        butterfly_b(t, 2 * i, 2 * i + 1, 60 - 16 * (i as i32), 1, r);
    }
    // Step 3: H(i, 4 + i, 0, r) for i = 0..3.
    for i in 0..4 {
        butterfly_h(t, i, 4 + i, 0, r);
    }
    // Step 4: B(4 + 3i, 5 + i, 48 - 32i, 1, r) for i = 0..1.
    for i in 0..2 {
        butterfly_b(t, 4 + 3 * i, 5 + i, 48 - 32 * (i as i32), 1, r);
    }
    // Step 5: H(4j + i, 2 + 4j + i, 0, r) for i = 0..1, j = 0..1.
    for j in 0..2 {
        for i in 0..2 {
            butterfly_h(t, 4 * j + i, 2 + 4 * j + i, 0, r);
        }
    }
    // Step 6: B(2 + 4i, 3 + 4i, 32, 1, r) for i = 0..1.
    for i in 0..2 {
        butterfly_b(t, 2 + 4 * i, 3 + 4 * i, 32, 1, r);
    }
    // Step 7: output permutation with n = 3.
    inverse_adst_output_permute(t, 3);
}

/// §7.13.2.8 inverse ADST16 process (av1-spec p.303). 9-step
/// butterfly schedule wrapped in §7.13.2.4 / §7.13.2.5 permutations.
pub fn inverse_adst16(t: &mut [i64], r: u32) {
    // Step 1: input permutation with n = 4.
    inverse_adst_input_permute(t, 4);
    // Step 2: B(2i, 2i+1, 62 - 8i, 1, r) for i = 0..7.
    for i in 0..8 {
        butterfly_b(t, 2 * i, 2 * i + 1, 62 - 8 * (i as i32), 1, r);
    }
    // Step 3: H(i, 8 + i, 0, r) for i = 0..7.
    for i in 0..8 {
        butterfly_h(t, i, 8 + i, 0, r);
    }
    // Step 4: paired B's for i = 0..1.
    for i in 0..2 {
        butterfly_b(t, 8 + 2 * i, 9 + 2 * i, 56 - 32 * (i as i32), 1, r);
        butterfly_b(t, 13 + 2 * i, 12 + 2 * i, 8 + 32 * (i as i32), 1, r);
    }
    // Step 5: H(8j + i, 4 + 8j + i, 0, r) for i = 0..3, j = 0..1.
    for j in 0..2 {
        for i in 0..4 {
            butterfly_h(t, 8 * j + i, 4 + 8 * j + i, 0, r);
        }
    }
    // Step 6: B(4 + 8j + 3i, 5 + 8j + i, 48 - 32i, 1, r) for i,j in 0..1.
    for j in 0..2 {
        for i in 0..2 {
            butterfly_b(
                t,
                4 + 8 * j + 3 * i,
                5 + 8 * j + i,
                48 - 32 * (i as i32),
                1,
                r,
            );
        }
    }
    // Step 7: H(4j + i, 2 + 4j + i, 0, r) for i = 0..1, j = 0..3.
    for j in 0..4 {
        for i in 0..2 {
            butterfly_h(t, 4 * j + i, 2 + 4 * j + i, 0, r);
        }
    }
    // Step 8: B(2 + 4i, 3 + 4i, 32, 1, r) for i = 0..3.
    for i in 0..4 {
        butterfly_b(t, 2 + 4 * i, 3 + 4 * i, 32, 1, r);
    }
    // Step 9: output permutation with n = 4.
    inverse_adst_output_permute(t, 4);
}

/// §7.13.2.9 inverse ADST dispatcher (av1-spec p.303-304). Routes
/// `n in 2..=4` to `inverse_adst4` / `inverse_adst8` / `inverse_adst16`.
pub fn inverse_adst(t: &mut [i64], n: u32, r: u32) {
    match n {
        2 => inverse_adst4(t, r),
        3 => inverse_adst8(t, r),
        4 => inverse_adst16(t, r),
        _ => panic!("oxideav-av1 §7.13.2.9: inverse_adst expects n in 2..=4, got {n}"),
    }
}

/// §7.13.2.10 inverse Walsh-Hadamard transform process (av1-spec
/// p.304). Operates on the length-4 array `T`. `shift` is the
/// per-axis pre-scaling: §7.13.3 row pass uses `shift = 2`, column
/// pass uses `shift = 0` (Lossless path only).
pub fn inverse_wht4(t: &mut [i64], shift: u32) {
    let a0 = t[0] >> shift;
    let c0 = t[1] >> shift;
    let d0 = t[2] >> shift;
    let b0 = t[3] >> shift;
    let mut a = a0;
    let c = c0;
    let mut d = d0;
    let mut b = b0;
    a += c;
    d -= b;
    let e = (a - d) >> 1;
    b = e - b;
    let c2 = e - c;
    a -= b;
    d += c2;
    t[0] = a;
    t[1] = b;
    t[2] = c2;
    t[3] = d;
}

/// §7.13.2.11 inverse identity 4 (av1-spec p.304).
pub fn inverse_identity4(t: &mut [i64]) {
    for slot in t.iter_mut().take(4) {
        *slot = round2(*slot * 5793, 12);
    }
}

/// §7.13.2.12 inverse identity 8 (av1-spec p.304-305).
pub fn inverse_identity8(t: &mut [i64]) {
    for slot in t.iter_mut().take(8) {
        *slot *= 2;
    }
}

/// §7.13.2.13 inverse identity 16 (av1-spec p.305).
pub fn inverse_identity16(t: &mut [i64]) {
    for slot in t.iter_mut().take(16) {
        *slot = round2(*slot * 11586, 12);
    }
}

/// §7.13.2.14 inverse identity 32 (av1-spec p.305).
pub fn inverse_identity32(t: &mut [i64]) {
    for slot in t.iter_mut().take(32) {
        *slot *= 4;
    }
}

/// §7.13.2.15 inverse identity dispatcher (av1-spec p.305). `n in
/// 2..=5` selects the corresponding identity routine.
pub fn inverse_identity(t: &mut [i64], n: u32) {
    match n {
        2 => inverse_identity4(t),
        3 => inverse_identity8(t),
        4 => inverse_identity16(t),
        5 => inverse_identity32(t),
        _ => panic!("oxideav-av1 §7.13.2.15: inverse_identity expects n in 2..=5, got {n}"),
    }
}

/// §7.13.3 2D inverse transform dispatcher (av1-spec p.305-307).
///
/// Consumes `dequant` (the post-§7.12.3-step-1 `Dequant[i][j]` array
/// in row-major order, of length `w * h` where `w = Tx_Width[tx_sz]`,
/// `h = Tx_Height[tx_sz]`) and returns the §7.13.3 `Residual` array
/// (length `w * h`, row-major).
///
/// The row pass populates `T[j] = Dequant[i][j]` for `i, j < 32`
/// (other entries `0`) per the §7.13.3 row body, applies the
/// `Abs(log2W - log2H) == 1` rectangular scaling, runs the
/// row-selector kernel per `PlaneTxType`, and stores `Residual[i][j]
/// = Round2(T[j], rowShift)`. The between-stage clamp at
/// `Max(BitDepth + 6, 16) - 1` precision is applied. The column
/// pass reads `T[i] = Residual[i][j]`, runs the column-selector
/// kernel, and stores `Residual[i][j] = Round2(T[i], colShift)`.
///
/// `tx_type` must be one of [`crate::cdf::DCT_DCT`] ..
/// [`crate::cdf::H_FLIPADST`] (`0..16`). `bit_depth` is the §3
/// per-frame `BitDepth` (`8`, `10`, or `12`). `lossless` is the per-
/// block §6.8.11 `Lossless` flag (when `true`, both passes route
/// through the §7.13.2.10 WHT path; `tx_sz` must be `TX_4X4`).
///
/// Returns the `w * h` `Residual` buffer.
pub fn inverse_transform_2d(
    dequant: &[i64],
    tx_sz: usize,
    tx_type: usize,
    bit_depth: u32,
    lossless: bool,
) -> Vec<i64> {
    let mut residual = Vec::new();
    inverse_transform_2d_into(dequant, tx_sz, tx_type, bit_depth, lossless, &mut residual);
    residual
}

/// r481 — [`inverse_transform_2d`] into a reused buffer.
pub(crate) fn inverse_transform_2d_into(
    dequant: &[i64],
    tx_sz: usize,
    tx_type: usize,
    bit_depth: u32,
    lossless: bool,
    residual: &mut Vec<i64>,
) {
    assert!(
        tx_sz < TX_SIZES_ALL,
        "oxideav-av1 §7.13.3: tx_sz {tx_sz} out of range (TX_SIZES_ALL = {TX_SIZES_ALL})"
    );
    let log2_w = TX_WIDTH_LOG2[tx_sz] as u32;
    let log2_h = (TX_HEIGHT[tx_sz] as u32).trailing_zeros();
    let w = TX_WIDTH[tx_sz];
    let h = TX_HEIGHT[tx_sz];
    assert_eq!(
        dequant.len(),
        w * h,
        "oxideav-av1 §7.13.3: dequant length mismatch"
    );

    let row_shift = if lossless {
        0
    } else {
        TRANSFORM_ROW_SHIFT[tx_sz]
    };
    let col_shift = if lossless { 0 } else { 4 };
    let row_clamp_range = bit_depth + 8;
    let col_clamp_range = core::cmp::max(bit_depth + 6, 16);

    residual.clear();
    residual.resize(w * h, 0);
    // r487 — the DCT row kernels run several rows per pass (see
    // `dct_row_pass`); the other kernels keep the row-at-a-time loop.
    let row_is_dct = !lossless
        && matches!(tx_type, x if x == DCT_DCT || x == ADST_DCT || x == FLIPADST_DCT || x == H_DCT);
    if row_is_dct {
        let rect_scale = log2_w.abs_diff(log2_h) == 1;
        // Lane count by block height, array length by block width.
        macro_rules! rows {
            ($lanes:literal) => {
                match w {
                    4 => dct_row_pass::<$lanes, 4>(
                        dequant,
                        residual,
                        h,
                        rect_scale,
                        row_clamp_range,
                        row_shift,
                    ),
                    8 => dct_row_pass::<$lanes, 8>(
                        dequant,
                        residual,
                        h,
                        rect_scale,
                        row_clamp_range,
                        row_shift,
                    ),
                    16 => dct_row_pass::<$lanes, 16>(
                        dequant,
                        residual,
                        h,
                        rect_scale,
                        row_clamp_range,
                        row_shift,
                    ),
                    32 => dct_row_pass::<$lanes, 32>(
                        dequant,
                        residual,
                        h,
                        rect_scale,
                        row_clamp_range,
                        row_shift,
                    ),
                    _ => dct_row_pass::<$lanes, 64>(
                        dequant,
                        residual,
                        h,
                        rect_scale,
                        row_clamp_range,
                        row_shift,
                    ),
                }
            };
        }
        if h >= 8 { rows!(8) } else { rows!(4) }
    } else {
        // Row transforms. r475 — stack scratch (the per-call heap buffers
        // were measurable at 12 MP), and an all-zero input row is left as
        // the all-zero output row it transforms to (every §7.13.2 kernel
        // maps zeros to zeros and `Round2( 0, n ) == 0`), which skips the
        // kernel for the rows past the block's non-zero extent.
        let mut tbuf = [0i64; 64];
        let tbuf = &mut tbuf[..w];
        for i in 0..h {
            // Populate T[j] from Dequant[i][j], zero for j or i >= 32.
            let mut any_nonzero = false;
            for j in 0..w {
                tbuf[j] = if i < 32 && j < 32 {
                    dequant[i * w + j]
                } else {
                    0
                };
                any_nonzero |= tbuf[j] != 0;
            }
            if !any_nonzero {
                continue;
            }
            // Abs(log2W - log2H) == 1 scaling.
            if log2_w.abs_diff(log2_h) == 1 {
                for slot in tbuf.iter_mut() {
                    *slot = round2(*slot * 2896, 12);
                }
            }
            if lossless {
                // §7.13.3 Lossless row pass: inverse WHT, shift = 2.
                // Only reachable for TX_4X4 (the spec's Lossless
                // constraint).
                inverse_wht4(tbuf, 2);
            } else {
                apply_row_kernel(tbuf, tx_type, log2_w, row_clamp_range);
            }
            for j in 0..w {
                residual[i * w + j] = round2(tbuf[j], row_shift);
            }
        }
    }
    // Between-stage clamp at col_clamp_range bits.
    let lo: i64 = -(1i64 << (col_clamp_range - 1));
    let hi: i64 = (1i64 << (col_clamp_range - 1)) - 1;
    for cell in residual.iter_mut() {
        *cell = clip3(lo, hi, *cell);
    }
    // r487 — likewise for the DCT column kernels (`dct_column_pass`).
    let col_is_dct = !lossless
        && matches!(tx_type, x if x == DCT_DCT || x == DCT_ADST || x == DCT_FLIPADST || x == V_DCT);
    if col_is_dct {
        macro_rules! columns {
            ($lanes:literal) => {
                match h {
                    4 => dct_column_pass::<$lanes, 4>(residual, w, col_clamp_range, col_shift),
                    8 => dct_column_pass::<$lanes, 8>(residual, w, col_clamp_range, col_shift),
                    16 => dct_column_pass::<$lanes, 16>(residual, w, col_clamp_range, col_shift),
                    32 => dct_column_pass::<$lanes, 32>(residual, w, col_clamp_range, col_shift),
                    _ => dct_column_pass::<$lanes, 64>(residual, w, col_clamp_range, col_shift),
                }
            };
        }
        if w >= 8 {
            columns!(8)
        } else {
            columns!(4)
        }
        return;
    }
    // Column transforms. r481 — the column process is a function of
    // the column's input alone, so a column whose input equals the
    // previous column's input takes that column's output (a DC-only
    // block's row pass leaves every column the same input: one column
    // transform instead of `w`).
    let mut tbuf = [0i64; 64];
    let tbuf = &mut tbuf[..h];
    let mut prev_in = [0i64; 64];
    let prev_in = &mut prev_in[..h];
    for j in 0..w {
        let mut same = j > 0;
        for i in 0..h {
            let v = residual[i * w + j];
            same &= v == prev_in[i];
            tbuf[i] = v;
        }
        if same {
            for i in 0..h {
                residual[i * w + j] = residual[i * w + j - 1];
            }
            continue;
        }
        prev_in.copy_from_slice(tbuf);
        if lossless {
            inverse_wht4(tbuf, 0);
        } else {
            apply_col_kernel(tbuf, tx_type, log2_h, col_clamp_range);
        }
        for i in 0..h {
            residual[i * w + j] = round2(tbuf[i], col_shift);
        }
    }
}

/// §7.13.3 row-pass kernel selector (av1-spec p.306). DCT path for
/// `{ DCT_DCT, ADST_DCT, FLIPADST_DCT, H_DCT }`; ADST path for
/// `{ DCT_ADST, ADST_ADST, DCT_FLIPADST, FLIPADST_FLIPADST,
/// ADST_FLIPADST, FLIPADST_ADST, H_ADST, H_FLIPADST }`; identity
/// path for the rest.
fn apply_row_kernel(t: &mut [i64], tx_type: usize, log2_w: u32, row_clamp_range: u32) {
    if matches!(tx_type, x if x == DCT_DCT || x == ADST_DCT || x == FLIPADST_DCT || x == H_DCT) {
        inverse_dct(t, log2_w, row_clamp_range);
    } else if matches!(
        tx_type,
        x if x == DCT_ADST
            || x == ADST_ADST
            || x == DCT_FLIPADST
            || x == FLIPADST_FLIPADST
            || x == ADST_FLIPADST
            || x == FLIPADST_ADST
            || x == H_ADST
            || x == H_FLIPADST
    ) {
        inverse_adst(t, log2_w, row_clamp_range);
    } else {
        // IDTX, V_DCT, V_ADST, V_FLIPADST -> identity on rows.
        debug_assert!(
            matches!(tx_type, x if x == IDTX || x == V_DCT || x == V_ADST || x == V_FLIPADST)
        );
        inverse_identity(t, log2_w);
    }
}

/// §7.13.3 column-pass kernel selector (av1-spec p.306-307). DCT
/// path for `{ DCT_DCT, DCT_ADST, DCT_FLIPADST, V_DCT }`; ADST path
/// for `{ ADST_DCT, ADST_ADST, FLIPADST_DCT, FLIPADST_FLIPADST,
/// ADST_FLIPADST, FLIPADST_ADST, V_ADST, V_FLIPADST }`; identity
/// path for the rest.
fn apply_col_kernel(t: &mut [i64], tx_type: usize, log2_h: u32, col_clamp_range: u32) {
    if matches!(tx_type, x if x == DCT_DCT || x == DCT_ADST || x == DCT_FLIPADST || x == V_DCT) {
        inverse_dct(t, log2_h, col_clamp_range);
    } else if matches!(
        tx_type,
        x if x == ADST_DCT
            || x == ADST_ADST
            || x == FLIPADST_DCT
            || x == FLIPADST_FLIPADST
            || x == ADST_FLIPADST
            || x == FLIPADST_ADST
            || x == V_ADST
            || x == V_FLIPADST
    ) {
        inverse_adst(t, log2_h, col_clamp_range);
    } else {
        // IDTX, H_DCT, H_ADST, H_FLIPADST -> identity on columns.
        debug_assert!(
            matches!(tx_type, x if x == IDTX || x == H_DCT || x == H_ADST || x == H_FLIPADST)
        );
        inverse_identity(t, log2_h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdf::{TX_4X4, TX_4X8, TX_8X8, TX_16X16, TX_32X32, TX_64X64};

    /// The §7.13.3 2-D inverse transform one row, then one column, at
    /// a time through the scalar kernels — the model the lane passes
    /// are checked against.
    fn inverse_transform_2d_scalar(
        dequant: &[i64],
        tx_sz: usize,
        tx_type: usize,
        bit_depth: u32,
    ) -> Vec<i64> {
        let log2_w = TX_WIDTH_LOG2[tx_sz] as u32;
        let log2_h = (TX_HEIGHT[tx_sz] as u32).trailing_zeros();
        let (w, h) = (TX_WIDTH[tx_sz], TX_HEIGHT[tx_sz]);
        let row_shift = TRANSFORM_ROW_SHIFT[tx_sz];
        let row_clamp_range = bit_depth + 8;
        let col_clamp_range = core::cmp::max(bit_depth + 6, 16);
        let mut residual = vec![0i64; w * h];
        for i in 0..h {
            let mut t: Vec<i64> = (0..w)
                .map(|j| {
                    if i < 32 && j < 32 {
                        dequant[i * w + j]
                    } else {
                        0
                    }
                })
                .collect();
            if log2_w.abs_diff(log2_h) == 1 {
                for v in t.iter_mut() {
                    *v = round2(*v * 2896, 12);
                }
            }
            apply_row_kernel(&mut t, tx_type, log2_w, row_clamp_range);
            for j in 0..w {
                residual[i * w + j] = round2(t[j], row_shift);
            }
        }
        let lo = -(1i64 << (col_clamp_range - 1));
        let hi = (1i64 << (col_clamp_range - 1)) - 1;
        for v in residual.iter_mut() {
            *v = clip3(lo, hi, *v);
        }
        for j in 0..w {
            let mut t: Vec<i64> = (0..h).map(|i| residual[i * w + j]).collect();
            apply_col_kernel(&mut t, tx_type, log2_h, col_clamp_range);
            for i in 0..h {
                residual[i * w + j] = round2(t[i], 4);
            }
        }
        residual
    }

    /// r487 — the recorded-schedule lane passes equal the scalar
    /// kernels on every transform size and on every transform type
    /// with a DCT row or column kernel, across coefficient magnitudes
    /// from a few units to the dequantiser's clamp (so both the
    /// 32-bit and the 64-bit lanes run), sparse and dense blocks,
    /// DC-only blocks included.
    #[test]
    fn lane_dct_passes_match_the_scalar_kernels() {
        use crate::cdf::{
            ADST_DCT, DCT_ADST, DCT_DCT, DCT_FLIPADST, FLIPADST_DCT, H_DCT, IDTX, TX_SIZES_ALL,
            V_DCT,
        };
        let mut state = 0xD1CE_F00D_0BAD_5EEDu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut narrow_groups = 0usize;
        let mut wide_groups = 0usize;
        for tx_sz in 0..TX_SIZES_ALL {
            let (w, h) = (TX_WIDTH[tx_sz], TX_HEIGHT[tx_sz]);
            let types: &[usize] = if w.max(h) > 32 {
                &[DCT_DCT]
            } else if w.max(h) == 32 {
                &[DCT_DCT, IDTX]
            } else {
                &[
                    DCT_DCT,
                    ADST_DCT,
                    DCT_ADST,
                    FLIPADST_DCT,
                    DCT_FLIPADST,
                    V_DCT,
                    H_DCT,
                    IDTX,
                ]
            };
            for &tx_type in types {
                for bit_depth in [8u32, 10, 12] {
                    for trial in 0..24u32 {
                        let cap = 1i64 << (7 + bit_depth);
                        let bits = 2 + (trial % 5) * ((5 + bit_depth) / 4);
                        let mut dequant = vec![0i64; w * h];
                        let fill = |v: u64| -> i64 {
                            let mag = (v >> 8) as i64 & ((1i64 << bits) - 1);
                            (if v & 1 == 0 { mag } else { -mag }).clamp(-cap, cap - 1)
                        };
                        match trial % 4 {
                            // DC only.
                            0 => dequant[0] = fill(next()) | 1,
                            // A few low-frequency coefficients.
                            1 => {
                                for _ in 0..4 {
                                    let (i, j) =
                                        ((next() % 4) as usize % h, (next() % 4) as usize % w);
                                    dequant[i * w + j] = fill(next());
                                }
                            }
                            // Dense inside the coded 32 × 32 corner.
                            2 => {
                                for i in 0..h.min(32) {
                                    for j in 0..w.min(32) {
                                        dequant[i * w + j] = fill(next());
                                    }
                                }
                            }
                            // Every coefficient at the clamp.
                            _ => {
                                for i in 0..h.min(32) {
                                    for j in 0..w.min(32) {
                                        dequant[i * w + j] =
                                            if next() & 1 == 0 { cap - 1 } else { -cap };
                                    }
                                }
                            }
                        }
                        let max_abs = dequant.iter().map(|v| v.unsigned_abs()).max().unwrap_or(0);
                        if max_abs <= dct_program(TX_WIDTH_LOG2[tx_sz] as u32).narrow_limit as u64 {
                            narrow_groups += 1;
                        } else {
                            wide_groups += 1;
                        }
                        let mut got = Vec::new();
                        inverse_transform_2d_into(
                            &dequant, tx_sz, tx_type, bit_depth, false, &mut got,
                        );
                        let want = inverse_transform_2d_scalar(&dequant, tx_sz, tx_type, bit_depth);
                        assert_eq!(
                            got, want,
                            "tx_sz={tx_sz} tx_type={tx_type} bd={bit_depth} trial={trial}"
                        );
                    }
                }
            }
        }
        assert!(narrow_groups > 0 && wide_groups > 0, "both lane widths ran");
    }

    /// r487 — the 32-bit lane bound is sound: at inputs of exactly
    /// `narrow_limit` magnitude (every sign pattern class) the 32-bit
    /// and 64-bit lane replays agree, and no 64-bit intermediate
    /// leaves the signed 32-bit range.
    #[test]
    fn narrow_lane_limit_keeps_every_intermediate_in_32_bits() {
        let mut state = 0xA5A5_5A5A_1234_4321u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        fn check<const N: usize>(next: &mut dyn FnMut() -> u64) {
            let n = N.trailing_zeros();
            let program = dct_program(n);
            let m = program.narrow_limit;
            assert!(m >= 64, "n={n}: limit {m} too small to be useful");
            for trial in 0..400u32 {
                let mut wide = [[0i64; 8]; N];
                for slot in wide.iter_mut() {
                    for v in slot.iter_mut() {
                        let r = next();
                        *v = match trial % 4 {
                            0 => m,
                            1 => -m,
                            2 => {
                                if r & 1 == 0 {
                                    m
                                } else {
                                    -m
                                }
                            }
                            _ => (r % (2 * m as u64 + 1)) as i64 - m,
                        };
                    }
                }
                let mut narrow = [[0i32; 8]; N];
                for (dst, src) in narrow.iter_mut().zip(&wide) {
                    for (d, &v) in dst.iter_mut().zip(src) {
                        *d = v as i32;
                    }
                }
                // Replay step by step on the wide lanes, checking the
                // unrounded sums of every step.
                for step in &program.steps {
                    match *step {
                        FlowStep::Rotate { a, b, cos, sin, .. } => {
                            for (&ta, &tb) in wide[a as usize].iter().zip(&wide[b as usize]) {
                                for v in [
                                    ta * i64::from(cos),
                                    tb * i64::from(sin),
                                    ta * i64::from(cos) - tb * i64::from(sin) + 2048,
                                    ta * i64::from(sin) + tb * i64::from(cos) + 2048,
                                ] {
                                    assert!(
                                        i32::try_from(v).is_ok(),
                                        "n={n} trial={trial}: rotation term {v} overflows"
                                    );
                                }
                            }
                        }
                        FlowStep::AddSub { sum, diff } => {
                            for (&x, &y) in wide[sum as usize].iter().zip(&wide[diff as usize]) {
                                assert!(
                                    i32::try_from(x + y).is_ok() && i32::try_from(x - y).is_ok()
                                );
                            }
                        }
                    }
                    run_dct_lanes_wide(core::slice::from_ref(step), &mut wide, 20);
                }
                run_dct_lanes_narrow(&program.steps, &mut narrow, 20);
                for (ns, ws) in narrow.iter().zip(&wide) {
                    for (&a, &b) in ns.iter().zip(ws) {
                        assert_eq!(i64::from(a), b, "n={n} trial={trial}");
                    }
                }
            }
        }
        check::<4>(&mut next);
        check::<8>(&mut next);
        check::<16>(&mut next);
        check::<32>(&mut next);
        check::<64>(&mut next);
    }

    #[test]
    fn cos128_spec_anchor_points() {
        // §7.13.2.1 cos128 anchors per the Cos128_Lookup table.
        assert_eq!(cos128(0), 4096); // cos(0) * 4096.
        assert_eq!(cos128(64), 0); // cos(pi/2) * 4096.
        assert_eq!(cos128(128), -4096); // cos(pi) * 4096.
        assert_eq!(cos128(192), 0); // cos(3pi/2) * 4096.
        // angle2 = 32 -> Cos128_Lookup[32] = 2896.
        assert_eq!(cos128(32), 2896);
        // angle = 32 + 64 = 96, falls in (64..=128): -Cos128_Lookup[32] = -2896.
        assert_eq!(cos128(96), -2896);
        // angle = 32 + 128 = 160, falls in (128..=192): -Cos128_Lookup[32] = -2896.
        assert_eq!(cos128(160), -2896);
        // angle = 32 + 192 = 224, falls in (192..=256): Cos128_Lookup[32] = 2896.
        assert_eq!(cos128(224), 2896);
    }

    #[test]
    fn sin128_is_cos128_shifted() {
        // §7.13.2.1 sin128(a) == cos128(a - 64).
        for a in -64..256 {
            assert_eq!(sin128(a), cos128(a - 64));
        }
    }

    #[test]
    fn brev_basic() {
        // §7.13.2.1 brev: bit-reverse the low num_bits bits.
        assert_eq!(brev(2, 0b00), 0b00);
        assert_eq!(brev(2, 0b01), 0b10);
        assert_eq!(brev(2, 0b10), 0b01);
        assert_eq!(brev(2, 0b11), 0b11);
        assert_eq!(brev(3, 0b100), 0b001);
        assert_eq!(brev(4, 0b0001), 0b1000);
    }

    #[test]
    fn round2_signed_arithmetic() {
        // §4.7.2: positive case (x + half) >> n.
        assert_eq!(round2(0, 12), 0);
        assert_eq!(round2(4096, 12), 1);
        assert_eq!(round2(2048, 12), 1); // (2048 + 2048) >> 12 = 1.
        assert_eq!(round2(2047, 12), 0); // (2047 + 2048) >> 12 = 0.
        // Negative: arithmetic shift; (-2048 + 2048) >> 12 = 0 >> 12 = 0.
        assert_eq!(round2(-2048, 12), 0);
        // (-4096 + 2048) = -2048; -2048 >> 12 (arith) = -1.
        assert_eq!(round2(-4096, 12), -1);
        // (-6144 + 2048) = -4096; -4096 >> 12 (arith) = -1.
        assert_eq!(round2(-6144, 12), -1);
        // (-6145 + 2048) = -4097; -4097 >> 12 (arith) = -2.
        assert_eq!(round2(-6145, 12), -2);
        // n = 0 short-circuit.
        assert_eq!(round2(12345, 0), 12345);
        assert_eq!(round2(-1, 0), -1);
    }

    #[test]
    fn wht4_round_trip_zero() {
        let mut t = [0i64; 4];
        inverse_wht4(&mut t, 2);
        assert_eq!(t, [0; 4]);
    }

    #[test]
    fn identity8_doubles_values() {
        // §7.13.2.12: T[i] = T[i] * 2.
        let mut t: Vec<i64> = (1..=8).collect();
        inverse_identity8(&mut t);
        assert_eq!(t, vec![2, 4, 6, 8, 10, 12, 14, 16]);
    }

    #[test]
    fn identity32_quadruples_values() {
        let mut t: Vec<i64> = (0..32).collect();
        inverse_identity32(&mut t);
        for (i, v) in t.iter().enumerate() {
            assert_eq!(*v, 4 * i as i64);
        }
    }

    #[test]
    fn identity4_scales_by_5793_over_4096() {
        // §7.13.2.11: T[i] = Round2(T[i] * 5793, 12).
        let mut t = [4096i64, 0, 0, 0];
        inverse_identity4(&mut t);
        // Round2(4096 * 5793, 12) = (23728128 + 2048) >> 12 = 5793.
        assert_eq!(t, [5793, 0, 0, 0]);
    }

    #[test]
    fn identity16_scales_by_11586_over_4096() {
        let mut t = [0i64; 16];
        t[0] = 4096;
        inverse_identity16(&mut t);
        // Round2(4096 * 11586, 12) = 11586.
        assert_eq!(t[0], 11586);
    }

    #[test]
    fn dct_permute_2_swaps_middle_pair() {
        // brev(2, 0..4) = [0, 2, 1, 3].
        let mut t = [10i64, 20, 30, 40];
        inverse_dct_permute(&mut t, 2);
        assert_eq!(t, [10, 30, 20, 40]);
    }

    #[test]
    fn adst_input_permute_n3() {
        // §7.13.2.4 with n = 3, n0 = 8.
        // idx = (i & 1) ? (i - 1) : (n0 - i - 1) for i = 0..8:
        //  i = 0: idx = 7
        //  i = 1: idx = 0
        //  i = 2: idx = 5
        //  i = 3: idx = 2
        //  i = 4: idx = 3
        //  i = 5: idx = 4
        //  i = 6: idx = 1
        //  i = 7: idx = 6
        let mut t: Vec<i64> = (0..8).collect();
        inverse_adst_input_permute(&mut t, 3);
        assert_eq!(t, vec![7, 0, 5, 2, 3, 4, 1, 6]);
    }

    #[test]
    fn idct4_dc_only_flat_output() {
        // §7.13.2.3 IDCT4 on [DC, 0, 0, 0] -> [v, v, v, v] where
        // v = Round2(DC * 2896, 12). For DC = 4096: v = 2896.
        let mut t = [4096i64, 0, 0, 0];
        inverse_dct(&mut t, 2, 16);
        assert_eq!(t, [2896, 2896, 2896, 2896]);
    }

    #[test]
    fn idct8_zero_input_yields_zero() {
        let mut t = [0i64; 8];
        inverse_dct(&mut t, 3, 16);
        assert_eq!(t, [0; 8]);
    }

    #[test]
    fn idct16_zero_input_yields_zero() {
        let mut t = [0i64; 16];
        inverse_dct(&mut t, 4, 16);
        assert_eq!(t, [0; 16]);
    }

    #[test]
    fn idct32_zero_input_yields_zero() {
        let mut t = [0i64; 32];
        inverse_dct(&mut t, 5, 16);
        assert_eq!(t, [0; 32]);
    }

    #[test]
    fn idct64_zero_input_yields_zero() {
        let mut t = [0i64; 64];
        inverse_dct(&mut t, 6, 16);
        assert_eq!(t, [0; 64]);
    }

    #[test]
    fn iadst4_zero_input_yields_zero() {
        let mut t = [0i64; 4];
        inverse_adst4(&mut t, 16);
        assert_eq!(t, [0; 4]);
    }

    #[test]
    fn iadst8_zero_input_yields_zero() {
        let mut t = [0i64; 8];
        inverse_adst8(&mut t, 16);
        assert_eq!(t, [0; 8]);
    }

    #[test]
    fn iadst16_zero_input_yields_zero() {
        let mut t = [0i64; 16];
        inverse_adst16(&mut t, 16);
        assert_eq!(t, [0; 16]);
    }

    #[test]
    fn idtx_dispatcher_4_8_16_32() {
        let mut t = [4096i64, 0, 0, 0];
        inverse_identity(&mut t, 2);
        assert_eq!(t[0], 5793);
        let mut t = [1i64; 8];
        inverse_identity(&mut t, 3);
        assert_eq!(t, [2; 8]);
        let mut t = [0i64; 16];
        t[0] = 4096;
        inverse_identity(&mut t, 4);
        assert_eq!(t[0], 11586);
        let mut t = [1i64; 32];
        inverse_identity(&mut t, 5);
        for v in &t {
            assert_eq!(*v, 4);
        }
    }

    #[test]
    fn transform_2d_dct_dct_4x4_dc_only() {
        // Dequant = [4096, 0, ...]; row pass yields [2896; 4] row 0,
        // [0; 4] rows 1..; row Round2 by rowShift = 0 keeps values.
        // Column pass: each column has [2896, 0, 0, 0] -> IDCT4 ->
        // v' = Round2(2896 * 2896, 12) = (8386816 + 2048) >> 12 = 2048.
        // [v',v',v',v']. colShift = 4 -> Round2(2048, 4) = (2048 + 8) >> 4 = 128.
        let mut dequant = vec![0i64; 16];
        dequant[0] = 4096;
        let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, false);
        assert_eq!(residual, vec![128; 16]);
    }

    #[test]
    fn transform_2d_dct_dct_zero_yields_zero() {
        let dequant = vec![0i64; 16];
        let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, false);
        assert_eq!(residual, vec![0; 16]);
    }

    #[test]
    fn transform_2d_idtx_4x4_with_dc_only() {
        // Pure identity row + column for TX_4X4 (n = 2):
        // Row: [4096,0,0,0] -> Round2(x * 5793, 12) = [5793,0,0,0]; >> rowShift(0).
        // Between clamp: ok.
        // Col j=0: [5793,0,0,0] -> identity4 -> Round2(5793*5793, 12)
        // = Round2(33558849, 12) = (33558849 + 2048) >> 12 = 8194; >> colShift(4)
        // = Round2(8194, 4) = (8194 + 8) >> 4 = 8202 >> 4 = 512.
        // Other cols j>0: [0,0,0,0] -> 0.
        let mut dequant = vec![0i64; 16];
        dequant[0] = 4096;
        let residual = inverse_transform_2d(&dequant, TX_4X4, IDTX, 8, false);
        let mut expected = vec![0i64; 16];
        expected[0] = 512;
        assert_eq!(residual, expected);
    }

    #[test]
    fn transform_2d_lossless_zero_yields_zero() {
        let dequant = vec![0i64; 16];
        let residual = inverse_transform_2d(&dequant, TX_4X4, DCT_DCT, 8, true);
        assert_eq!(residual, vec![0; 16]);
    }

    #[test]
    fn transform_2d_rectangular_4x8_zero_yields_zero() {
        // TX_4X8: w = 4, h = 8, |log2W - log2H| = 1 ⇒ rectangular
        // scaling kicks in but is identity on zeros.
        let dequant = vec![0i64; 32];
        let residual = inverse_transform_2d(&dequant, TX_4X8, DCT_DCT, 8, false);
        assert_eq!(residual, vec![0; 32]);
    }

    #[test]
    fn transform_2d_8x8_zero_yields_zero() {
        let dequant = vec![0i64; 64];
        let residual = inverse_transform_2d(&dequant, TX_8X8, DCT_DCT, 8, false);
        assert_eq!(residual, vec![0; 64]);
    }

    #[test]
    fn transform_2d_16x16_zero_yields_zero() {
        let dequant = vec![0i64; 256];
        let residual = inverse_transform_2d(&dequant, TX_16X16, ADST_ADST, 8, false);
        assert_eq!(residual, vec![0; 256]);
    }

    #[test]
    fn transform_2d_32x32_zero_yields_zero() {
        let dequant = vec![0i64; 1024];
        let residual = inverse_transform_2d(&dequant, TX_32X32, DCT_DCT, 8, false);
        assert_eq!(residual, vec![0; 1024]);
    }

    #[test]
    fn transform_2d_64x64_zero_yields_zero() {
        let dequant = vec![0i64; 4096];
        let residual = inverse_transform_2d(&dequant, TX_64X64, DCT_DCT, 8, false);
        assert_eq!(residual, vec![0; 4096]);
    }

    #[test]
    fn transform_2d_flipadst_zero_yields_zero() {
        // FLIPADST_FLIPADST should also produce zero on zero input —
        // the flip is in the §7.12.3 frame-write, not in §7.13.3.
        let dequant = vec![0i64; 64];
        let residual = inverse_transform_2d(&dequant, TX_8X8, FLIPADST_FLIPADST, 8, false);
        assert_eq!(residual, vec![0; 64]);
    }

    #[test]
    fn transform_2d_h_v_variants_zero_yields_zero() {
        // V_DCT / H_DCT / V_ADST / H_ADST / V_FLIPADST / H_FLIPADST
        // must all produce zero on zero input.
        let variants = [V_DCT, H_DCT, V_ADST, H_ADST, V_FLIPADST, H_FLIPADST];
        for tx_type in variants {
            let dequant = vec![0i64; 64];
            let residual = inverse_transform_2d(&dequant, TX_8X8, tx_type, 8, false);
            assert_eq!(
                residual,
                vec![0; 64],
                "tx_type {tx_type} should zero-preserve"
            );
        }
    }

    #[test]
    fn idct4_linearity() {
        // §7.13.2.3 is integer-rounded so true linearity is approximate.
        // Check that scaling DC by 2 scales output by 2 (no rounding
        // error at DC = 4096 -> 2896, DC = 8192 -> 5792).
        let mut a = [4096i64, 0, 0, 0];
        let mut b = [8192i64, 0, 0, 0];
        inverse_dct(&mut a, 2, 16);
        inverse_dct(&mut b, 2, 16);
        for i in 0..4 {
            assert_eq!(b[i], 2 * a[i]);
        }
    }
}
