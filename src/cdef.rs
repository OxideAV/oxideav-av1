//! §7.15 CDEF (Constrained Directional Enhancement Filter) — the
//! de-ringing pass that sits **after** the §7.14 deblocking filter and
//! before §7.16 upscaling per av1-spec p.318-324.
//!
//! ## Coverage (round 196 — close-out push)
//!
//! This module covers the §7.15 CDEF process end-to-end at the
//! sample-filtering level:
//!
//! * §7.15.1 — [`cdef_frame`]: top-level per-8×8 walk; [`cdef_block`]:
//!   per-block driver that copies `CurrFrame[plane]` into
//!   `CdefFrame[plane]`, then (for non-`-1` `idx` blocks with at least
//!   one non-skip 4×4) invokes the §7.15.2 direction search and the
//!   §7.15.3 primary + secondary filter per plane.
//! * §7.15.2 — [`cdef_direction`]: 8-direction match against the
//!   `partial[8][15]` projection sums, scored via the `Div_Table[9]`
//!   constant; returns `(yDir, var)`.
//! * §7.15.3 — [`cdef_filter_block`]: per-plane filter that walks the
//!   `h × w` plane region, applies the primary tap pair along `dir`
//!   and the secondary tap pair along `(dir ± 2) & 7`, scaled by
//!   [`CDEF_PRI_TAPS`] / [`CDEF_SEC_TAPS`], with the [`constrain`]
//!   damping primitive and a final `Clip3(min, max, ...)` over the
//!   neighbour samples actually consulted.
//!
//! ## Standalone-friendly surface
//!
//! Like the §7.14 driver, the top-level driver takes a small
//! [`CdefFrameContext`] bundling the frame-level inputs plus closures
//! for the §5.11.x decode state the driver reads:
//!
//! * Per-block CDEF index (`cdef_idx[r][c]` per §5.11.56) — `-1` means
//!   "no CDEF for this anchor"; otherwise indexes into the §5.9.19
//!   [`crate::CdefParams::cdef_y_pri_strength`] etc. arrays.
//! * Per-block `Skips[r][c]` (per §5.11.x) — the §7.15.1 `skip`
//!   variable is the conjunction over the 4×4 cells in the 8×8 block.
//! * Per-plane subsampling, bit depth, MI dimensions, [`crate::CdefParams`]
//!   strength schedule.
//!
//! ## Bitstream-conformance gates not enforced here
//!
//! * The caller is responsible for the §5.9.19 `cdef_bits ≤ 3` cap
//!   (the `f(2)` reader guarantees this).
//! * The caller has already populated `CurrFrame[plane]` per §7.12.3
//!   and (optionally) §7.14 deblock pass.
//! * Plane buffers are sized at `(mi_rows * MI_SIZE) >> subY` ×
//!   `(mi_cols * MI_SIZE) >> subX` per av1-spec §5.11.34.
//!
//! ## Out-of-scope for this arc
//!
//! * §7.16 superres upscaling and §7.17 loop restoration — separate
//!   passes that run after CDEF.
//! * Cross-plane SIMD / cache-friendly batched filtering — the
//!   reference loop here mirrors the spec's per-sample formulation.

use crate::cdf::{MI_SIZE, MI_SIZE_LOG2};
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::CdefParams;

// =====================================================================
// §7.15 constant lookup tables — av1-spec p.320-324.
// =====================================================================

/// `Cdef_Pri_Taps[2][2]` per av1-spec p.324 lines 17900-17902. Indexed
/// as `[(priStr >> coeffShift) & 1][k]` where `k ∈ {0, 1}` selects the
/// primary tap radius.
pub const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];

/// `Cdef_Sec_Taps[2][2]` per av1-spec p.324 lines 17904-17906. Indexed
/// as `[(priStr >> coeffShift) & 1][k]` where `k ∈ {0, 1}` selects the
/// secondary tap radius.
pub const CDEF_SEC_TAPS: [[i32; 2]; 2] = [[2, 1], [2, 1]];

/// `Cdef_Directions[8][2][2]` per av1-spec p.324 lines 17950-17959.
/// `[dir][k]` returns `(dy, dx)` of the `k`-th tap offset along
/// direction `dir`. `k = 0` is the radius-1 neighbour, `k = 1` is the
/// radius-2 neighbour.
pub const CDEF_DIRECTIONS: [[[i32; 2]; 2]; 8] = [
    [[-1, 1], [-2, 2]],
    [[0, 1], [-1, 2]],
    [[0, 1], [0, 2]],
    [[0, 1], [1, 2]],
    [[1, 1], [2, 2]],
    [[1, 0], [2, 1]],
    [[1, 0], [2, 0]],
    [[1, 0], [2, -1]],
];

/// `Div_Table[9]` per av1-spec p.322 lines 17817-17819. Drives the
/// per-line-length normalisation in the §7.15.2 direction-search cost
/// accumulator.
pub const DIV_TABLE: [i32; 9] = [0, 840, 420, 280, 210, 168, 140, 120, 105];

/// `Cdef_Uv_Dir[2][2][8]` per av1-spec p.321 lines 17716-17721. Indexed
/// as `[subsampling_x][subsampling_y][yDir]`; supplies the chroma
/// `dir` translated from the luma `yDir`. For 4:4:4 (`subX = subY = 0`)
/// the chroma uses `yDir` unchanged; for 4:2:0 / 4:2:2 / 4:4:0 the
/// spec's lookup snaps adjacent directions onto a single chroma
/// direction.
pub const CDEF_UV_DIR: [[[u8; 8]; 2]; 2] = [
    [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 2, 2, 3, 4, 6, 0]],
    [[7, 0, 2, 4, 5, 6, 6, 6], [0, 1, 2, 3, 4, 5, 6, 7]],
];

// =====================================================================
// §7.15 standalone-driver surface — av1-spec p.319.
// =====================================================================

/// Caller-supplied frame-level inputs the §7.15 driver consults.
///
/// All §5.5 / §5.9.5 / §5.9.19 fields come straight from the parsed
/// frame / sequence header. Per-block predicates are closures so the
/// driver can be exercised in isolation without wiring the full
/// §5.11 decode state.
pub struct CdefFrameContext<'a> {
    /// `MiRows` per §5.9.5.
    pub mi_rows: u32,
    /// `MiCols` per §5.9.5.
    pub mi_cols: u32,
    /// `NumPlanes` per §5.5.2 — `1` for monochrome, `3` otherwise.
    pub num_planes: u8,
    /// `BitDepth` per §5.5.2 (`8`, `10`, or `12`).
    pub bit_depth: u8,
    /// `subsampling_x` per §5.5.2 (`0` or `1`).
    pub subsampling_x: u8,
    /// `subsampling_y` per §5.5.2 (`0` or `1`).
    pub subsampling_y: u8,
    /// `CdefParams` per §5.9.19 — the `(cdef_damping, cdef_bits,
    /// cdef_y_pri_strength[], cdef_y_sec_strength[], cdef_uv_*[])`
    /// schedule. The §7.15.1 driver indexes the strength arrays with
    /// the per-anchor `idx` supplied by [`Self::cdef_idx`].
    pub cdef_params: &'a CdefParams,
    /// `cdef_idx[r][c]` per §5.11.56 — `-1` means "skip CDEF for this
    /// anchor" (the driver still copies `CurrFrame` ↦ `CdefFrame`);
    /// otherwise the value is a CDEF-params index in
    /// `0..(1 << cdef_bits)`.
    pub cdef_idx: &'a dyn Fn(u32, u32) -> i8,
    /// `Skips[r][c]` per §5.11.x — the §7.15.1 `skip` conjunction
    /// over the 8×8 block's four 4×4 cells gates whether the
    /// per-plane filter actually runs.
    pub skip: &'a dyn Fn(u32, u32) -> bool,
}

impl std::fmt::Debug for CdefFrameContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CdefFrameContext")
            .field("mi_rows", &self.mi_rows)
            .field("mi_cols", &self.mi_cols)
            .field("num_planes", &self.num_planes)
            .field("bit_depth", &self.bit_depth)
            .field("subsampling_x", &self.subsampling_x)
            .field("subsampling_y", &self.subsampling_y)
            .field("cdef_params", &self.cdef_params)
            .finish_non_exhaustive()
    }
}

// =====================================================================
// §7.15.1 top-level driver. av1-spec p.319 lines 17601-17614.
// =====================================================================

/// `Num_4x4_Blocks_Wide[BLOCK_8X8] = 2` per §3 (av1-spec p.20). The
/// §7.15.1 walk steps the `(r, c)` cursor by this stride.
const STEP_4: u32 = 2;
/// `Num_4x4_Blocks_Wide[BLOCK_64X64] = 16` per §3. The §7.15.1
/// `baseR/baseC = r & cdefMask4` anchors are snapped to this stride.
const CDEF_SIZE_4: u32 = 16;
/// `~(cdefSize4 - 1)` per §7.15.1.
const CDEF_MASK_4: u32 = !(CDEF_SIZE_4 - 1);

/// §7.15 top-level driver — av1-spec p.319 lines 17601-17614.
///
/// Walks the frame in 8×8 luma blocks (2 × 2 of `MI_SIZE = 4` cells),
/// reads the per-superblock `cdef_idx[baseR][baseC]`, and dispatches to
/// [`cdef_block`].
///
/// `src_planes` is the post-deblock `CurrFrame[plane]` (read-only);
/// `dst_planes` is `CdefFrame[plane]` (written in place). Both have
/// the same `(rows, cols)` shape per the spec's `CdefFrame =
/// CurrFrame.clone(); ...` initialisation.
pub fn cdef_frame(
    ctx: &CdefFrameContext<'_>,
    src_planes: &[PlaneBuffer<'_>],
    dst_planes: &mut [PlaneBuffer<'_>],
) {
    let num_planes = ctx
        .num_planes
        .min(src_planes.len() as u8)
        .min(dst_planes.len() as u8);
    let mut r = 0u32;
    while r < ctx.mi_rows {
        let mut c = 0u32;
        while c < ctx.mi_cols {
            let base_r = r & CDEF_MASK_4;
            let base_c = c & CDEF_MASK_4;
            let idx = (ctx.cdef_idx)(base_r, base_c);
            cdef_block(ctx, src_planes, dst_planes, num_planes, r, c, idx);
            c += STEP_4;
        }
        r += STEP_4;
    }
}

// =====================================================================
// §7.15.1 CDEF block process. av1-spec p.319-321 lines 17621-17704.
// =====================================================================

/// §7.15.1 per-8×8 CDEF block driver — av1-spec p.319-321.
///
/// First copies the 8×8 luma block (and its sub-sampled chroma 8×8
/// region) from `src_planes` into `dst_planes`. If `idx == -1` the
/// driver returns immediately after the copy. Otherwise it consults
/// the §5.11.x `skip` predicate over the four 4×4 cells; when **any**
/// of them is non-skip the §7.15.2 direction search runs and the
/// §7.15.3 primary + secondary filter is applied per plane with the
/// schedule selected by `idx`.
///
/// `(r, c)` is in 4×4-block luma coordinates; the block spans
/// `(r..r+2, c..c+2)` (4×4 cells) and `(r*MI_SIZE..r*MI_SIZE+8,
/// c*MI_SIZE..c*MI_SIZE+8)` (luma samples).
pub fn cdef_block(
    ctx: &CdefFrameContext<'_>,
    src_planes: &[PlaneBuffer<'_>],
    dst_planes: &mut [PlaneBuffer<'_>],
    num_planes: u8,
    r: u32,
    c: u32,
    idx: i8,
) {
    // av1-spec p.319 lines 17631-17651: copy CurrFrame ↦ CdefFrame
    // for the luma 8×8 block, then the chroma 8×8 sub-sampled block.
    cdef_copy_plane(src_planes, dst_planes, 0, r, c, 0, 0);
    if num_planes > 1 {
        cdef_copy_plane(
            src_planes,
            dst_planes,
            1,
            r,
            c,
            ctx.subsampling_x,
            ctx.subsampling_y,
        );
        cdef_copy_plane(
            src_planes,
            dst_planes,
            2,
            r,
            c,
            ctx.subsampling_x,
            ctx.subsampling_y,
        );
    }
    // av1-spec p.319 line 17664.
    if idx < 0 {
        return;
    }
    // av1-spec p.319 lines 17666-17668.
    let coeff_shift = ctx.bit_depth as i32 - 8;
    let skip = (ctx.skip)(r, c)
        && (r + 1 >= ctx.mi_rows || (ctx.skip)(r + 1, c))
        && (c + 1 >= ctx.mi_cols || (ctx.skip)(r, c + 1))
        && (r + 1 >= ctx.mi_rows || c + 1 >= ctx.mi_cols || (ctx.skip)(r + 1, c + 1));
    if skip {
        return;
    }
    // av1-spec p.319 line 17670: §7.15.2 direction search.
    let (y_dir, var) = cdef_direction(ctx, src_planes, r, c);
    // av1-spec p.319 lines 17675-17688: luma filter.
    let idx = idx as usize;
    let pri_str_y = (ctx.cdef_params.cdef_y_pri_strength[idx] as i32) << coeff_shift;
    let sec_str_y = (ctx.cdef_params.cdef_y_sec_strength[idx] as i32) << coeff_shift;
    let dir_y = if pri_str_y == 0 { 0 } else { y_dir };
    let var_str = if (var >> 6) > 0 {
        (floor_log2((var >> 6) as u32) as i32).min(12)
    } else {
        0
    };
    let pri_str_y_scaled = if var != 0 {
        (pri_str_y * (4 + var_str) + 8) >> 4
    } else {
        0
    };
    let damping_y = ctx.cdef_params.cdef_damping as i32 + coeff_shift;
    cdef_filter_block(
        ctx,
        src_planes,
        dst_planes,
        0,
        r,
        c,
        pri_str_y_scaled,
        sec_str_y,
        damping_y,
        dir_y,
    );
    // av1-spec p.319 line 17690: terminate when monochrome.
    if num_planes == 1 {
        return;
    }
    // av1-spec p.319 lines 17692-17704: chroma U / V filter.
    let pri_str_uv = (ctx.cdef_params.cdef_uv_pri_strength[idx] as i32) << coeff_shift;
    let sec_str_uv = (ctx.cdef_params.cdef_uv_sec_strength[idx] as i32) << coeff_shift;
    let dir_uv = if pri_str_uv == 0 {
        0
    } else {
        CDEF_UV_DIR[ctx.subsampling_x.min(1) as usize][ctx.subsampling_y.min(1) as usize]
            [y_dir as usize] as i32
    };
    let damping_uv = ctx.cdef_params.cdef_damping as i32 + coeff_shift - 1;
    cdef_filter_block(
        ctx, src_planes, dst_planes, 1, r, c, pri_str_uv, sec_str_uv, damping_uv, dir_uv,
    );
    cdef_filter_block(
        ctx, src_planes, dst_planes, 2, r, c, pri_str_uv, sec_str_uv, damping_uv, dir_uv,
    );
}

/// §7.15.1 copy-loop helper — av1-spec p.319 lines 17631-17651.
/// Copies the 8×8 (post-subsampling) block from `src_planes[plane]`
/// into `dst_planes[plane]`.
fn cdef_copy_plane(
    src_planes: &[PlaneBuffer<'_>],
    dst_planes: &mut [PlaneBuffer<'_>],
    plane: u8,
    r: u32,
    c: u32,
    sub_x: u8,
    sub_y: u8,
) {
    let p = plane as usize;
    let Some(src) = src_planes.get(p) else { return };
    let Some(dst) = dst_planes.get_mut(p) else {
        return;
    };
    let start_y = ((r * MI_SIZE as u32) >> sub_y) as i32;
    let end_y = (((r + STEP_4) * MI_SIZE as u32) >> sub_y) as i32;
    let start_x = ((c * MI_SIZE as u32) >> sub_x) as i32;
    let end_x = (((c + STEP_4) * MI_SIZE as u32) >> sub_x) as i32;
    let cols = src.cols.min(dst.cols) as i32;
    let rows = src.rows.min(dst.rows) as i32;
    for y in start_y..end_y {
        if y < 0 || y >= rows {
            continue;
        }
        for x in start_x..end_x {
            if x < 0 || x >= cols {
                continue;
            }
            let v = src.samples[(y as usize) * (src.cols as usize) + (x as usize)];
            dst.samples[(y as usize) * (dst.cols as usize) + (x as usize)] = v;
        }
    }
}

// =====================================================================
// §7.15.2 CDEF direction process. av1-spec p.321-322 lines 17725-17819.
// =====================================================================

/// §7.15.2 — finds the dominant edge direction for an 8×8 luma block.
///
/// Returns `(yDir, var)`:
///
/// * `yDir ∈ 0..8` — the spec's direction ordinal (see
///   [`CDEF_DIRECTIONS`]).
/// * `var` — `(bestCost - cost[(yDir + 4) & 7]) >> 10`, a measure of
///   direction confidence consumed by §7.15.1 line 17683.
pub fn cdef_direction(
    ctx: &CdefFrameContext<'_>,
    src_planes: &[PlaneBuffer<'_>],
    r: u32,
    c: u32,
) -> (i32, i32) {
    let Some(plane0) = src_planes.first() else {
        return (0, 0);
    };
    // av1-spec p.321 lines 17747-17751: zero-init `partial[8][15]` and
    // `cost[8]`.
    let mut partial = [[0i32; 15]; 8];
    let mut cost = [0i32; 8];
    let x0 = (c as i32) << MI_SIZE_LOG2;
    let y0 = (r as i32) << MI_SIZE_LOG2;
    let shift = ctx.bit_depth as i32 - 8;
    // av1-spec p.321 lines 17756-17768: per-pixel projection-sum
    // accumulator across the 8×8 luma block.
    for i in 0..8i32 {
        for j in 0..8i32 {
            let sx = (x0 + j).clamp(0, plane0.cols as i32 - 1) as usize;
            let sy = (y0 + i).clamp(0, plane0.rows as i32 - 1) as usize;
            let raw = plane0.samples[sy * (plane0.cols as usize) + sx];
            let x = (raw >> shift) - 128;
            partial[0][(i + j) as usize] += x;
            partial[1][(i + j / 2) as usize] += x;
            partial[2][i as usize] += x;
            partial[3][(3 + i - j / 2) as usize] += x;
            partial[4][(7 + i - j) as usize] += x;
            partial[5][(3 - i / 2 + j) as usize] += x;
            partial[6][j as usize] += x;
            partial[7][(i / 2 + j) as usize] += x;
        }
    }
    // av1-spec p.322 lines 17769-17774: directions 2 (vertical) and 6
    // (horizontal) — sum of squares, normalised by Div_Table[8].
    for &v in &partial[2][..8] {
        cost[2] += v * v;
    }
    for &v in &partial[6][..8] {
        cost[6] += v * v;
    }
    cost[2] *= DIV_TABLE[8];
    cost[6] *= DIV_TABLE[8];
    // av1-spec p.322 lines 17775-17784: directions 0 and 4 — paired
    // partial sums normalised by Div_Table[i + 1].
    for i in 0..7usize {
        cost[0] += (partial[0][i] * partial[0][i] + partial[0][14 - i] * partial[0][14 - i])
            * DIV_TABLE[i + 1];
        cost[4] += (partial[4][i] * partial[4][i] + partial[4][14 - i] * partial[4][14 - i])
            * DIV_TABLE[i + 1];
    }
    cost[0] += partial[0][7] * partial[0][7] * DIV_TABLE[8];
    cost[4] += partial[4][7] * partial[4][7] * DIV_TABLE[8];
    // av1-spec p.322 lines 17785-17795: the four oblique directions
    // 1, 3, 5, 7 — middle slab (indices 3..=7) sums of squares scaled
    // by Div_Table[8], plus paired-tail term scaled by Div_Table[2*j +
    // 2].
    let mut i = 1usize;
    while i < 8 {
        for j in 0..5usize {
            cost[i] += partial[i][3 + j] * partial[i][3 + j];
        }
        cost[i] *= DIV_TABLE[8];
        for j in 0..3usize {
            cost[i] += (partial[i][j] * partial[i][j] + partial[i][10 - j] * partial[i][10 - j])
                * DIV_TABLE[2 * j + 2];
        }
        i += 2;
    }
    // av1-spec p.322 lines 17796-17811: pick `yDir = argmax_i cost[i]`
    // (strict > so ties go to the lowest index, matching the spec's
    // `if (cost[i] > bestCost)` ordering); compute the variance.
    let mut best_cost = 0i32;
    let mut y_dir = 0i32;
    for (k, &c_k) in cost.iter().enumerate() {
        if c_k > best_cost {
            best_cost = c_k;
            y_dir = k as i32;
        }
    }
    let var = (best_cost - cost[((y_dir + 4) & 7) as usize]) >> 10;
    (y_dir, var)
}

// =====================================================================
// §7.15.3 CDEF filter process. av1-spec p.323-324 lines 17823-17894.
// =====================================================================

/// §7.15.3 — per-plane CDEF filter.
///
/// Walks the `h × w` plane region of the 8×8 block (`h = 8 >> subY`,
/// `w = 8 >> subX`), and for each output sample computes
/// `Clip3(min, max, x + ((8 + sum - (sum < 0)) >> 4))` where `sum`
/// accumulates contributions from the [`CDEF_PRI_TAPS`] / [`CDEF_SEC_TAPS`]
/// taps along `dir` (primary) and `(dir ± 2) & 7` (secondary), and
/// `min` / `max` track the min/max of the neighbour samples actually
/// consulted.
#[allow(clippy::too_many_arguments)]
pub fn cdef_filter_block(
    ctx: &CdefFrameContext<'_>,
    src_planes: &[PlaneBuffer<'_>],
    dst_planes: &mut [PlaneBuffer<'_>],
    plane: u8,
    r: u32,
    c: u32,
    pri_str: i32,
    sec_str: i32,
    damping: i32,
    dir: i32,
) {
    let p = plane as usize;
    let Some(src) = src_planes.get(p) else { return };
    let Some(dst) = dst_planes.get_mut(p) else {
        return;
    };
    // av1-spec p.323 lines 17860-17866.
    let (sub_x, sub_y) = if plane == 0 {
        (0u8, 0u8)
    } else {
        (ctx.subsampling_x, ctx.subsampling_y)
    };
    let coeff_shift = ctx.bit_depth as i32 - 8;
    let x0 = ((c as i32) * MI_SIZE as i32) >> sub_x;
    let y0 = ((r as i32) * MI_SIZE as i32) >> sub_y;
    let w = 8i32 >> sub_x;
    let h = 8i32 >> sub_y;
    // §7.15.3 line 17876: `(priStr >> coeffShift) & 1` selects the tap
    // row.
    let pri_row = ((pri_str >> coeff_shift) & 1) as usize;
    // The spec also uses `(priStr >> coeffShift) & 1` for the secondary
    // tap row (av1-spec p.323 line 17884), not `secStr`.
    let sec_row = pri_row;
    let plane_w = src.cols as i32;
    let plane_h = src.rows as i32;
    // §7.15.1 is_inside_filter_region (av1-spec p.103 §5.11.52) checks
    // the candidate position against `(0, 0)..(MiRows, MiCols)` in
    // 4×4 units. For the per-plane filter that's equivalent to
    // `(0, 0)..(plane_h, plane_w)` in samples after subsampling.
    for i in 0..h {
        for j in 0..w {
            let py = y0 + i;
            let px = x0 + j;
            if py < 0 || py >= plane_h || px < 0 || px >= plane_w {
                continue;
            }
            let x = src.samples[(py as usize) * (plane_w as usize) + (px as usize)];
            let mut sum = 0i32;
            let mut max_v = x;
            let mut min_v = x;
            // av1-spec p.323-324 lines 17872-17891: primary +
            // secondary tap accumulation over `k ∈ {0, 1}` × sign ∈
            // {-1, +1}.
            for k in 0..2usize {
                for sign in [-1i32, 1i32] {
                    // Primary tap along `dir`.
                    if let Some(p_val) =
                        cdef_sample_at(src, plane_w, plane_h, x0, y0, i, j, dir, k, sign)
                    {
                        sum += CDEF_PRI_TAPS[pri_row][k] * constrain(p_val - x, pri_str, damping);
                        if p_val > max_v {
                            max_v = p_val;
                        }
                        if p_val < min_v {
                            min_v = p_val;
                        }
                    }
                    // Secondary taps along `(dir ± 2) & 7`.
                    for dir_off in [-2i32, 2i32] {
                        let dir2 = (dir + dir_off) & 7;
                        if let Some(s_val) =
                            cdef_sample_at(src, plane_w, plane_h, x0, y0, i, j, dir2, k, sign)
                        {
                            sum +=
                                CDEF_SEC_TAPS[sec_row][k] * constrain(s_val - x, sec_str, damping);
                            if s_val > max_v {
                                max_v = s_val;
                            }
                            if s_val < min_v {
                                min_v = s_val;
                            }
                        }
                    }
                }
            }
            // av1-spec p.324 line 17892.
            let delta = (8 + sum - i32::from(sum < 0)) >> 4;
            let out = clip3(min_v, max_v, x + delta);
            dst.samples[(py as usize) * (dst.cols as usize) + (px as usize)] = out;
        }
    }
}

/// §7.15.3 — `cdef_get_at` per av1-spec p.324 lines 17932-17944.
/// Returns `Some(sample)` when the candidate position is inside the
/// `is_inside_filter_region`; otherwise `None` (the spec's
/// `CdefAvailable == 0` arm).
#[allow(clippy::too_many_arguments)]
fn cdef_sample_at(
    src: &PlaneBuffer<'_>,
    plane_w: i32,
    plane_h: i32,
    x0: i32,
    y0: i32,
    i: i32,
    j: i32,
    dir: i32,
    k: usize,
    sign: i32,
) -> Option<i32> {
    let dy = CDEF_DIRECTIONS[dir as usize & 7][k][0];
    let dx = CDEF_DIRECTIONS[dir as usize & 7][k][1];
    let y = y0 + i + sign * dy;
    let x = x0 + j + sign * dx;
    if y < 0 || y >= plane_h || x < 0 || x >= plane_w {
        return None;
    }
    Some(src.samples[(y as usize) * (src.cols as usize) + (x as usize)])
}

/// §7.15.3 `constrain` primitive — av1-spec p.324 lines 17919-17925.
///
/// `dampingAdj = Max(0, damping - FloorLog2(threshold))`. Returns
/// `sign(diff) * Clip3(0, |diff|, threshold - (|diff| >> dampingAdj))`.
/// Returns `0` when `threshold == 0`.
#[must_use]
pub fn constrain(diff: i32, threshold: i32, damping: i32) -> i32 {
    if threshold == 0 {
        return 0;
    }
    let damping_adj = (damping - floor_log2(threshold as u32) as i32).max(0);
    let sign = if diff < 0 { -1i32 } else { 1i32 };
    let abs_diff = diff.unsigned_abs() as i32;
    let shifted = if damping_adj >= 31 {
        0
    } else {
        abs_diff >> damping_adj
    };
    let inner = (threshold - shifted).max(0);
    sign * clip3(0, abs_diff, inner)
}

/// §3 `FloorLog2(x)` — floor of `log2(x)`. `floor_log2(0) = 0` per
/// the spec's `BitCount(x) - 1` definition collapsing to 0 for the
/// degenerate input (callers never invoke at 0 — `constrain` gates on
/// `threshold == 0` first; the direction-search `var >> 6` gate also
/// excludes 0).
#[inline]
fn floor_log2(x: u32) -> u32 {
    if x == 0 {
        0
    } else {
        31 - x.leading_zeros()
    }
}

/// §3 `Clip3(a, b, x)` — clamp `x` to `[a, b]`.
#[inline]
fn clip3(a: i32, b: i32, x: i32) -> i32 {
    if x < a {
        a
    } else if x > b {
        b
    } else {
        x
    }
}

// =====================================================================
// Tests.
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uncompressed_header_tail::CDEF_MAX_STRENGTHS;

    fn make_params(damping: u8) -> CdefParams {
        CdefParams {
            cdef_damping: damping,
            cdef_bits: 0,
            cdef_y_pri_strength: [0; CDEF_MAX_STRENGTHS],
            cdef_y_sec_strength: [0; CDEF_MAX_STRENGTHS],
            cdef_uv_pri_strength: [0; CDEF_MAX_STRENGTHS],
            cdef_uv_sec_strength: [0; CDEF_MAX_STRENGTHS],
            short_circuited: false,
        }
    }

    #[test]
    fn constrain_returns_zero_for_zero_threshold() {
        // av1-spec p.324 line 17920: `!threshold ⇒ 0`.
        assert_eq!(constrain(5, 0, 3), 0);
        assert_eq!(constrain(-7, 0, 3), 0);
    }

    #[test]
    fn constrain_signs_match_diff_sign() {
        // For modest |diff| ≤ threshold and dampingAdj that doesn't
        // shift |diff| past threshold, constrain returns +/- diff
        // verbatim.
        // diff=5, threshold=16, damping=8. floor_log2(16)=4 ⇒ dampingAdj=4.
        // |diff|>>4 = 0. clip3(0, 5, 16 - 0) = 5. → sign(+) * 5 = 5.
        assert_eq!(constrain(5, 16, 8), 5);
        assert_eq!(constrain(-5, 16, 8), -5);
    }

    #[test]
    fn constrain_large_diff_clamped_by_threshold_minus_shifted() {
        // diff=20, threshold=8, damping=4 ⇒ dampingAdj = max(0, 4 -
        // floor_log2(8)) = max(0, 4-3) = 1. |diff|>>1 = 10.
        // threshold - 10 = -2 ⇒ inner clamps to 0. → 0.
        assert_eq!(constrain(20, 8, 4), 0);
    }

    #[test]
    fn cdef_directions_table_matches_spec() {
        // av1-spec p.324 lines 17950-17959 — sanity-check three rows.
        assert_eq!(CDEF_DIRECTIONS[0], [[-1, 1], [-2, 2]]);
        assert_eq!(CDEF_DIRECTIONS[2], [[0, 1], [0, 2]]);
        assert_eq!(CDEF_DIRECTIONS[7], [[1, 0], [2, -1]]);
    }

    #[test]
    fn div_table_matches_spec() {
        // av1-spec p.322 lines 17817-17819.
        assert_eq!(DIV_TABLE, [0, 840, 420, 280, 210, 168, 140, 120, 105]);
    }

    #[test]
    fn cdef_uv_dir_lookup_identity_for_444() {
        // av1-spec p.321 — `subX = subY = 0` row passes yDir unchanged.
        for (idx, &mapped) in CDEF_UV_DIR[0][0].iter().enumerate() {
            assert_eq!(mapped, idx as u8);
        }
    }

    #[test]
    fn direction_search_uniform_block_returns_zero_var() {
        // A uniform 100-luma block has zero high-pass projection
        // energy along every direction ⇒ bestCost = 0, var = 0.
        let samples = vec![100i32; 16 * 16];
        let mut s = samples.clone();
        let plane = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut s,
        };
        let params = make_params(3);
        let ctx = CdefFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            cdef_params: &params,
            cdef_idx: &|_, _| -1,
            skip: &|_, _| true,
        };
        let (y_dir, var) = cdef_direction(&ctx, std::slice::from_ref(&plane), 0, 0);
        assert_eq!(y_dir, 0);
        assert_eq!(var, 0);
    }

    #[test]
    fn direction_search_horizontal_stripes_pick_direction_2() {
        // A block whose samples vary along y only (horizontal stripes
        // — rows of equal value, varying between rows) has its edge
        // direction *along* the stripes, i.e. horizontal. Direction 2
        // has tap offsets `(0, ±1) / (0, ±2)` (av1-spec p.324 row
        // [[0,1],[0,2]]) — taps move purely along x, i.e. along a
        // horizontal edge. `partial[2][i] = sum over j` collects each
        // row's running total, which is the largest-variance
        // projection for row-varying input, hence the largest cost
        // and the winning direction.
        let mut samples = vec![0i32; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                samples[y * 16 + x] = 128 + (y as i32) * 10;
            }
        }
        let mut s = samples.clone();
        let plane = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut s,
        };
        let params = make_params(3);
        let ctx = CdefFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            cdef_params: &params,
            cdef_idx: &|_, _| -1,
            skip: &|_, _| true,
        };
        let (y_dir, _) = cdef_direction(&ctx, std::slice::from_ref(&plane), 0, 0);
        assert_eq!(
            y_dir, 2,
            "horizontal stripes ⇒ edge runs horizontally ⇒ direction 2"
        );
    }

    #[test]
    fn direction_search_vertical_stripes_pick_direction_6() {
        // A block whose samples vary along x only (vertical stripes)
        // has its edge direction *along* the stripes, i.e. vertical.
        // Direction 6 has tap offsets `(1, 0) / (2, 0)` (av1-spec
        // p.324 row [[1,0],[2,0]]) — taps move purely along y.
        let mut samples = vec![0i32; 16 * 16];
        for y in 0..16 {
            for x in 0..16 {
                samples[y * 16 + x] = 128 + (x as i32) * 10;
            }
        }
        let mut s = samples.clone();
        let plane = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut s,
        };
        let params = make_params(3);
        let ctx = CdefFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            cdef_params: &params,
            cdef_idx: &|_, _| -1,
            skip: &|_, _| true,
        };
        let (y_dir, _) = cdef_direction(&ctx, std::slice::from_ref(&plane), 0, 0);
        assert_eq!(
            y_dir, 6,
            "vertical stripes ⇒ edge runs vertically ⇒ direction 6"
        );
    }

    #[test]
    fn cdef_block_with_idx_neg1_only_copies() {
        // `idx == -1` → driver copies CurrFrame ↦ CdefFrame and returns.
        // Verify the destination matches the source after the call.
        let src_samples: Vec<i32> = (0i32..(16 * 16)).map(|v| v % 200).collect();
        let mut s = src_samples.clone();
        let src = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut s,
        };
        let mut d = vec![0i32; 16 * 16];
        let mut dst = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut d,
        };
        let params = make_params(3);
        let ctx = CdefFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            cdef_params: &params,
            cdef_idx: &|_, _| -1,
            skip: &|_, _| false,
        };
        cdef_block(
            &ctx,
            std::slice::from_ref(&src),
            std::slice::from_mut(&mut dst),
            1,
            0,
            0,
            -1,
        );
        // 8×8 block at (0,0) — bytes copied; rest of destination is 0.
        for y in 0..8 {
            for x in 0..8 {
                let i = y * 16 + x;
                assert_eq!(dst.samples[i], src_samples[i], "copy mismatch at ({y},{x})");
            }
        }
        for y in 8..16 {
            for x in 0..16 {
                let i = y * 16 + x;
                assert_eq!(dst.samples[i], 0, "out-of-block region must be untouched");
            }
        }
    }

    #[test]
    fn cdef_block_with_skip_only_copies() {
        // All four 4×4 cells skip ⇒ driver copies and returns (no
        // §7.15.2/§7.15.3 invocation).
        let src_samples: Vec<i32> = (0i32..(16 * 16)).map(|v| v * 3 % 200).collect();
        let mut s = src_samples.clone();
        let src = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut s,
        };
        let mut d = vec![0i32; 16 * 16];
        let mut dst = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut d,
        };
        let mut params = make_params(3);
        params.cdef_y_pri_strength[0] = 15;
        params.cdef_y_sec_strength[0] = 4;
        let ctx = CdefFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            cdef_params: &params,
            cdef_idx: &|_, _| 0,
            skip: &|_, _| true, // every 4×4 cell is skip
        };
        cdef_block(
            &ctx,
            std::slice::from_ref(&src),
            std::slice::from_mut(&mut dst),
            1,
            0,
            0,
            0,
        );
        for y in 0..8 {
            for x in 0..8 {
                let i = y * 16 + x;
                assert_eq!(
                    dst.samples[i], src_samples[i],
                    "skip path must leave samples == source"
                );
            }
        }
    }

    #[test]
    fn cdef_filter_uniform_block_is_idempotent() {
        // Uniform 100-luma 8×8 block: every neighbour `p == x` ⇒
        // `p - x = 0` ⇒ `constrain(0, ...) = 0` ⇒ sum = 0 ⇒ delta = 0.
        // Result equals input.
        let samples = vec![100i32; 16 * 16];
        let mut s = samples.clone();
        let src = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut s,
        };
        let mut d = samples.clone();
        let mut dst = PlaneBuffer {
            rows: 16,
            cols: 16,
            samples: &mut d,
        };
        let mut params = make_params(3);
        params.cdef_y_pri_strength[0] = 8;
        params.cdef_y_sec_strength[0] = 2;
        let ctx = CdefFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            cdef_params: &params,
            cdef_idx: &|_, _| 0,
            skip: &|_, _| false,
        };
        cdef_filter_block(
            &ctx,
            std::slice::from_ref(&src),
            std::slice::from_mut(&mut dst),
            0,
            0,
            0,
            8, // priStr
            2, // secStr
            6, // damping
            2, // dir = vertical
        );
        for y in 0..8 {
            for x in 0..8 {
                let i = y * 16 + x;
                assert_eq!(dst.samples[i], 100, "uniform-input filter must be no-op");
            }
        }
    }

    #[test]
    fn cdef_frame_driver_walks_every_8x8_anchor() {
        // 32×32 frame ⇒ 8×8 grid = 4×4 anchors. With `cdef_idx == -1`
        // everywhere, every visit copies the source into the dest.
        let src_samples: Vec<i32> = (0i32..(32 * 32)).map(|v| (v * 7) % 200).collect();
        let mut s = src_samples.clone();
        let src = PlaneBuffer {
            rows: 32,
            cols: 32,
            samples: &mut s,
        };
        let mut d = vec![0i32; 32 * 32];
        let mut dst = PlaneBuffer {
            rows: 32,
            cols: 32,
            samples: &mut d,
        };
        let params = make_params(3);
        let ctx = CdefFrameContext {
            mi_rows: 8,
            mi_cols: 8,
            num_planes: 1,
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            cdef_params: &params,
            cdef_idx: &|_, _| -1,
            skip: &|_, _| true,
        };
        cdef_frame(
            &ctx,
            std::slice::from_ref(&src),
            std::slice::from_mut(&mut dst),
        );
        assert_eq!(dst.samples, src_samples, "driver must copy every 8x8 cell");
    }

    #[test]
    fn cdef_frame_chroma_copy_respects_subsampling() {
        // 16×16 luma ⇒ 8×8 chroma at 4:2:0. Driver must copy chroma
        // planes at the sub-sampled stride.
        let luma_samples = vec![50i32; 16 * 16];
        let chroma_u: Vec<i32> = (0i32..(8 * 8)).map(|v| v + 80).collect();
        let chroma_v: Vec<i32> = (0i32..(8 * 8)).map(|v| v + 130).collect();
        let mut y_s = luma_samples.clone();
        let mut u_s = chroma_u.clone();
        let mut v_s = chroma_v.clone();
        let src = vec![
            PlaneBuffer {
                rows: 16,
                cols: 16,
                samples: &mut y_s,
            },
            PlaneBuffer {
                rows: 8,
                cols: 8,
                samples: &mut u_s,
            },
            PlaneBuffer {
                rows: 8,
                cols: 8,
                samples: &mut v_s,
            },
        ];
        let mut yd = vec![0i32; 16 * 16];
        let mut ud = vec![0i32; 8 * 8];
        let mut vd = vec![0i32; 8 * 8];
        let mut dst = vec![
            PlaneBuffer {
                rows: 16,
                cols: 16,
                samples: &mut yd,
            },
            PlaneBuffer {
                rows: 8,
                cols: 8,
                samples: &mut ud,
            },
            PlaneBuffer {
                rows: 8,
                cols: 8,
                samples: &mut vd,
            },
        ];
        let params = make_params(3);
        let ctx = CdefFrameContext {
            mi_rows: 4,
            mi_cols: 4,
            num_planes: 3,
            bit_depth: 8,
            subsampling_x: 1,
            subsampling_y: 1,
            cdef_params: &params,
            cdef_idx: &|_, _| -1,
            skip: &|_, _| true,
        };
        cdef_frame(&ctx, &src, &mut dst);
        assert_eq!(dst[0].samples, luma_samples);
        assert_eq!(dst[1].samples, chroma_u);
        assert_eq!(dst[2].samples, chroma_v);
    }
}
