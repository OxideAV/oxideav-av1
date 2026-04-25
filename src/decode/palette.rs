//! Palette mode decode — §5.11.46 + §5.11.49 + §5.11.50 + §7.11.4.
//!
//! AV1 lets blocks predict samples from a small palette (2..=8 colours
//! per plane). The encoder transmits two pieces of state:
//!
//! 1. The palette colours themselves, signalled by `palette_mode_info()`
//!    (§5.11.46) — a mix of cache references and literal `BitDepth`
//!    samples plus delta encoding.
//! 2. A per-pixel index into the palette, signalled by
//!    `palette_tokens()` (§5.11.49). Tokens are decoded along the
//!    block's anti-diagonal scan with a context-aware CDF (§5.11.50).
//!
//! At reconstruction time the prediction step is replaced by
//! §7.11.4 `predict_palette`: each output sample becomes
//! `palette[map[y][x]]`. There is no residual on palette blocks.
//!
//! Round 10 wires every part of this end to end. The `PaletteBlock`
//! struct returned from [`decode_palette_mode_info`] carries both
//! palette plane vectors plus the decoded colour map; the
//! reconstruction path consumes it via [`apply_palette_luma`] and
//! [`apply_palette_chroma`].

use oxideav_core::{Error, Result};

use crate::cdfs;
use crate::symbol::SymbolDecoder;

use super::frame_state::FrameState;
use super::tile::TileDecoder;

/// Maximum palette size per plane — spec `PALETTE_COLORS = 8`.
pub const PALETTE_COLORS: usize = 8;
/// Number of palette colour-context buckets — spec
/// `PALETTE_COLOR_CONTEXTS = 5`.
pub const PALETTE_COLOR_CONTEXTS: usize = 5;
/// Number of neighbours folded into the context hash — spec
/// `PALETTE_NUM_NEIGHBORS = 3`.
pub const PALETTE_NUM_NEIGHBORS: usize = 3;
/// Largest `ColorContextHash` value the spec table covers.
pub const PALETTE_MAX_COLOR_CONTEXT_HASH: usize = 8;

/// Decoded palette state for one block. Empty plane vectors mean that
/// plane was not palette-coded; callers must check `size_y` /
/// `size_uv` before consulting `colors_*` / `color_map_*`.
#[derive(Clone, Debug, Default)]
pub struct PaletteBlock {
    /// Number of luma palette colours (0 if disabled). 2..=8 when set.
    pub size_y: u8,
    /// Number of chroma palette colours (0 if disabled). 2..=8 when
    /// set; same value for U and V (the palette pair is signalled
    /// jointly by `palette_mode_info()`).
    pub size_uv: u8,
    /// Sample bit depth for the colours below — 8/10/12.
    pub bit_depth: u8,
    /// Luma palette colours, sorted ascending. `size_y` entries valid.
    pub colors_y: Vec<u16>,
    /// U-plane palette colours, sorted ascending. `size_uv` entries
    /// valid.
    pub colors_u: Vec<u16>,
    /// V-plane palette colours; *not* sorted (spec emits them
    /// independently or via signed delta encoding when
    /// `delta_encode_palette_colors_v` is set). `size_uv` entries
    /// valid.
    pub colors_v: Vec<u16>,
    /// Y-plane colour-index map. Row-major `block_h × block_w` with
    /// `size_y` valid indices per cell. Empty when `size_y == 0`.
    pub color_map_y: Vec<u8>,
    /// Width (luma samples) covered by `color_map_y`.
    pub map_w_y: u32,
    /// Height (luma samples) covered by `color_map_y`.
    pub map_h_y: u32,
    /// UV-plane colour-index map. Stored at the chroma resolution
    /// (after subsampling). Empty when `size_uv == 0`.
    pub color_map_uv: Vec<u8>,
    /// Width (chroma samples) covered by `color_map_uv`.
    pub map_w_uv: u32,
    /// Height (chroma samples) covered by `color_map_uv`.
    pub map_h_uv: u32,
}

impl PaletteBlock {
    /// `true` when the block uses palette coding on either plane.
    #[inline]
    pub fn active(&self) -> bool {
        self.size_y > 0 || self.size_uv > 0
    }

    /// `true` when the luma plane is palette-coded.
    #[inline]
    pub fn has_y(&self) -> bool {
        self.size_y > 0
    }

    /// `true` when the chroma planes are palette-coded.
    #[inline]
    pub fn has_uv(&self) -> bool {
        self.size_uv > 0
    }

    /// Sample at the (x, y) luma coordinate, after palette lookup.
    /// Out-of-range coords clip to the last in-range sample (matches
    /// the spec's onscreen-extension fixup in §5.11.49).
    #[inline]
    pub fn luma_sample(&self, x: u32, y: u32) -> u16 {
        debug_assert!(self.has_y());
        let xc = x.min(self.map_w_y.saturating_sub(1));
        let yc = y.min(self.map_h_y.saturating_sub(1));
        let idx = (yc as usize) * (self.map_w_y as usize) + (xc as usize);
        let pi = self.color_map_y[idx] as usize;
        self.colors_y[pi]
    }

    /// Sample at the (x, y) chroma coordinate, after palette lookup.
    /// `plane` is 1 for U and 2 for V.
    #[inline]
    pub fn chroma_sample(&self, plane: usize, x: u32, y: u32) -> u16 {
        debug_assert!(self.has_uv());
        let xc = x.min(self.map_w_uv.saturating_sub(1));
        let yc = y.min(self.map_h_uv.saturating_sub(1));
        let idx = (yc as usize) * (self.map_w_uv as usize) + (xc as usize);
        let pi = self.color_map_uv[idx] as usize;
        if plane == 1 {
            self.colors_u[pi]
        } else {
            self.colors_v[pi]
        }
    }
}

/// Spec §5.11.46 + §5.11.49 — decode the full palette state for the
/// block at luma-sample `(x, y)` with footprint `(bw, bh)`. Consumes
/// the `has_palette_y` / `has_palette_uv` bits, the colour list, and
/// the `palette_tokens()` index map.
///
/// `y_mode` / `uv_mode` follow the spec gating:
///   - Y palette read fires only when `YMode == DC_PRED`.
///   - UV palette read fires only when `HasChroma && UVMode ==
///     DC_PRED`.
///
/// Returns an empty `PaletteBlock` for blocks the encoder declined to
/// palette-code (the bits are still consumed so the symbol stream
/// stays aligned).
#[allow(clippy::too_many_arguments)]
pub fn decode_palette_mode_info(
    td: &mut TileDecoder<'_>,
    fs: &FrameState,
    bsize_ctx: usize,
    bw: u32,
    bh: u32,
    y_mode_dc: bool,
    uv_mode_dc: bool,
    has_chroma: bool,
    x: u32,
    y: u32,
) -> Result<PaletteBlock> {
    let mut blk = PaletteBlock {
        bit_depth: fs.bit_depth as u8,
        ..PaletteBlock::default()
    };

    let mi_col = x >> 2;
    let mi_row = y >> 2;
    let have_above = mi_row > 0 && mi_col < fs.mi_cols;
    let have_left = mi_col > 0 && mi_row < fs.mi_rows;

    // Luma palette path (§5.11.46 + §5.11.49 inner block).
    if y_mode_dc {
        let mut y_ctx = 0usize;
        if have_above && fs.mi_at(mi_col, mi_row - 1).palette_size_y > 0 {
            y_ctx += 1;
        }
        if have_left && fs.mi_at(mi_col - 1, mi_row).palette_size_y > 0 {
            y_ctx += 1;
        }
        let has_palette_y = td.decode_has_palette_y(bsize_ctx, y_ctx)?;
        if has_palette_y {
            let cache_y = build_palette_cache(fs, 0, mi_col, mi_row);
            let (size_y, colors_y) =
                decode_palette_colors_y(td, bsize_ctx, fs.bit_depth, &cache_y)?;
            blk.size_y = size_y as u8;
            blk.colors_y = colors_y;
        }
    }

    // Chroma palette path (§5.11.46 + §5.11.49 outer chroma block).
    if has_chroma && uv_mode_dc {
        let uv_ctx = if blk.size_y > 0 { 1 } else { 0 };
        let has_palette_uv = td.decode_has_palette_uv(uv_ctx)?;
        if has_palette_uv {
            let cache_uv = build_palette_cache(fs, 1, mi_col, mi_row);
            let (size_uv, colors_u, colors_v) =
                decode_palette_colors_uv(td, bsize_ctx, fs.bit_depth, &cache_uv)?;
            blk.size_uv = size_uv as u8;
            blk.colors_u = colors_u;
            blk.colors_v = colors_v;
        }
    }

    // §5.11.49 palette_tokens() — index-map decode for whichever
    // planes activated above.
    if blk.has_y() {
        decode_palette_tokens_y(td, &mut blk, fs, bw, bh, x, y)?;
    }
    if blk.has_uv() {
        decode_palette_tokens_uv(td, &mut blk, fs, bw, bh, x, y)?;
    }

    Ok(blk)
}

/// Spec §5.11.46 `get_palette_cache` — merge the above and left
/// palette colour lists into a sorted unique-merged cache. Plane is 0
/// for luma, 1 for chroma. Returns the cache vector (length 0..=16).
fn build_palette_cache(fs: &FrameState, plane: usize, mi_col: u32, mi_row: u32) -> Vec<u16> {
    let above_n = if mi_row > 0 && (mi_row * 4) % 64 != 0 && mi_col < fs.mi_cols {
        let mi = fs.mi_at(mi_col, mi_row - 1);
        if plane == 0 {
            mi.palette_size_y as usize
        } else {
            mi.palette_size_uv as usize
        }
    } else {
        0
    };
    let left_n = if mi_col > 0 && mi_row < fs.mi_rows {
        let mi = fs.mi_at(mi_col - 1, mi_row);
        if plane == 0 {
            mi.palette_size_y as usize
        } else {
            mi.palette_size_uv as usize
        }
    } else {
        0
    };
    if above_n == 0 && left_n == 0 {
        return Vec::new();
    }
    let above = if above_n > 0 {
        fs.palette_colors_at(plane, mi_col, mi_row - 1)
    } else {
        &[]
    };
    let left = if left_n > 0 {
        fs.palette_colors_at(plane, mi_col - 1, mi_row)
    } else {
        &[]
    };
    let mut cache: Vec<u16> = Vec::with_capacity(above_n + left_n);
    let (mut ai, mut li) = (0usize, 0usize);
    while ai < above_n && li < left_n {
        let a = above[ai];
        let l = left[li];
        if l < a {
            if cache.last().copied() != Some(l) {
                cache.push(l);
            }
            li += 1;
        } else {
            if cache.last().copied() != Some(a) {
                cache.push(a);
            }
            ai += 1;
            if l == a {
                li += 1;
            }
        }
    }
    while ai < above_n {
        let v = above[ai];
        ai += 1;
        if cache.last().copied() != Some(v) {
            cache.push(v);
        }
    }
    while li < left_n {
        let v = left[li];
        li += 1;
        if cache.last().copied() != Some(v) {
            cache.push(v);
        }
    }
    cache
}

/// Spec §5.11.46 luma palette colour decode. Returns the
/// `(palette_size, sorted_colors)` pair.
fn decode_palette_colors_y(
    td: &mut TileDecoder<'_>,
    bsize_ctx: usize,
    bit_depth: u32,
    cache: &[u16],
) -> Result<(usize, Vec<u16>)> {
    let raw = td.decode_palette_size_y(bsize_ctx)?;
    let palette_size = (raw as usize) + 2;
    if palette_size > PALETTE_COLORS {
        return Err(Error::invalid(format!(
            "av1 palette_size_y_minus_2={raw} (§5.11.46)"
        )));
    }
    let mut colors = vec![0u16; palette_size];
    let mut idx = 0usize;
    // Cache references — each `use_palette_color_cache_y` is L(1).
    for &cval in cache {
        if idx >= palette_size {
            break;
        }
        let bit = td.symbol.read_literal(1);
        if bit != 0 {
            colors[idx] = cval;
            idx += 1;
        }
    }
    // First non-cached colour: literal `BitDepth` bits.
    if idx < palette_size {
        colors[idx] = td.symbol.read_literal(bit_depth) as u16;
        idx += 1;
    }
    // Remaining colours: delta encoded with adaptive paletteBits.
    if idx < palette_size {
        let min_bits = bit_depth as i32 - 3;
        let extra = td.symbol.read_literal(2) as i32;
        let mut palette_bits = (min_bits + extra).max(0) as u32;
        while idx < palette_size {
            let mut delta = td.symbol.read_literal(palette_bits) as u16;
            // Spec: "palette_delta_y++"  (luma only — UV doesn't add 1).
            delta = delta.saturating_add(1);
            let prev = colors[idx - 1] as u32;
            let val = prev + delta as u32;
            let max_val = (1u32 << bit_depth) - 1;
            colors[idx] = val.min(max_val) as u16;
            // Update palette_bits per spec.
            let range = (1i32 << bit_depth) - colors[idx] as i32 - 1;
            if range > 1 {
                palette_bits = palette_bits.min(ceil_log2(range as u32));
            } else {
                palette_bits = 0;
            }
            idx += 1;
        }
    }
    // Ascending-order sort (spec note suggests merge — sort_unstable
    // is plenty for ≤8 elements).
    colors.sort_unstable();
    Ok((palette_size, colors))
}

/// Spec §5.11.46 chroma palette colour decode (joint U+V signalling).
/// Returns `(palette_size, colors_u, colors_v)`. `colors_u` is sorted
/// ascending; `colors_v` is left in the order the spec emits it.
fn decode_palette_colors_uv(
    td: &mut TileDecoder<'_>,
    bsize_ctx: usize,
    bit_depth: u32,
    cache: &[u16],
) -> Result<(usize, Vec<u16>, Vec<u16>)> {
    let raw = td.decode_palette_size_uv(bsize_ctx)?;
    let palette_size = (raw as usize) + 2;
    if palette_size > PALETTE_COLORS {
        return Err(Error::invalid(format!(
            "av1 palette_size_uv_minus_2={raw} (§5.11.46)"
        )));
    }

    // U palette — same shape as Y, except the delta decode does NOT
    // pre-increment.
    let mut u = vec![0u16; palette_size];
    let mut idx = 0usize;
    for &cval in cache {
        if idx >= palette_size {
            break;
        }
        let bit = td.symbol.read_literal(1);
        if bit != 0 {
            u[idx] = cval;
            idx += 1;
        }
    }
    if idx < palette_size {
        u[idx] = td.symbol.read_literal(bit_depth) as u16;
        idx += 1;
    }
    if idx < palette_size {
        let min_bits = bit_depth as i32 - 3;
        let extra = td.symbol.read_literal(2) as i32;
        let mut palette_bits = (min_bits + extra).max(0) as u32;
        while idx < palette_size {
            let delta = td.symbol.read_literal(palette_bits);
            let prev = u[idx - 1] as u32;
            let val = prev + delta;
            let max_val = (1u32 << bit_depth) - 1;
            u[idx] = val.min(max_val) as u16;
            let range = (1i32 << bit_depth) - u[idx] as i32;
            if range > 1 {
                palette_bits = palette_bits.min(ceil_log2(range as u32));
            } else {
                palette_bits = 0;
            }
            idx += 1;
        }
    }
    u.sort_unstable();

    // V palette — either delta-encoded with sign, or plain literals.
    let delta_encode_v = td.symbol.read_literal(1) != 0;
    let mut v = vec![0u16; palette_size];
    if delta_encode_v {
        let min_bits = bit_depth as i32 - 4;
        let extra = td.symbol.read_literal(2) as i32;
        let palette_bits = (min_bits + extra).max(0) as u32;
        let max_val = 1u32 << bit_depth;
        v[0] = td.symbol.read_literal(bit_depth) as u16;
        for i in 1..palette_size {
            let mut delta = td.symbol.read_literal(palette_bits) as i32;
            if delta != 0 {
                let sign = td.symbol.read_literal(1);
                if sign != 0 {
                    delta = -delta;
                }
            }
            let mut val = v[i - 1] as i32 + delta;
            if val < 0 {
                val += max_val as i32;
            } else if val >= max_val as i32 {
                val -= max_val as i32;
            }
            // Clip1: clamp to [0, max-1].
            let clip_max = (max_val - 1) as i32;
            v[i] = val.clamp(0, clip_max) as u16;
        }
    } else {
        for slot in v.iter_mut() {
            *slot = td.symbol.read_literal(bit_depth) as u16;
        }
    }

    Ok((palette_size, u, v))
}

#[inline]
fn ceil_log2(x: u32) -> u32 {
    if x <= 1 {
        0
    } else {
        32 - (x - 1).leading_zeros()
    }
}

/// §5.11.49 `palette_tokens()` — luma index map. Walks the block in
/// anti-diagonal (wavefront) order, decoding one
/// `palette_color_idx_y` per sample. The first sample is signalled
/// with an `NS(PaletteSizeY)` literal (`color_index_map_y`).
fn decode_palette_tokens_y(
    td: &mut TileDecoder<'_>,
    blk: &mut PaletteBlock,
    fs: &FrameState,
    bw: u32,
    bh: u32,
    x: u32,
    y: u32,
) -> Result<()> {
    let mi_col = x >> 2;
    let mi_row = y >> 2;
    let block_w = bw;
    let block_h = bh;
    let onscreen_w = block_w.min((fs.mi_cols - mi_col) * 4);
    let onscreen_h = block_h.min((fs.mi_rows - mi_row) * 4);
    blk.map_w_y = block_w;
    blk.map_h_y = block_h;
    let mut map = vec![0u8; (block_w as usize) * (block_h as usize)];

    // First sample — non-symmetric `color_index_map_y`.
    let first = decode_uniform_ns(&mut td.symbol, blk.size_y as u16) as u8;
    map[0] = first;

    // Anti-diagonal scan — i = r + c, walked from (i=1) upward.
    // j ∈ [max(0, i + 1 - onscreen_h), min(i, onscreen_w - 1)],
    // walked descending. Per §5.11.49 the inner loop runs for every
    // i-line that intersects the onscreen rectangle.
    if onscreen_w > 0 && onscreen_h > 0 {
        let n = blk.size_y as usize;
        let i_max = onscreen_w + onscreen_h - 1;
        for i in 1..i_max {
            let j_start = i.min(onscreen_w - 1);
            // j_low_inclusive = max(0, i + 1 - onscreen_h)
            let j_low = (i + 1).saturating_sub(onscreen_h);
            let mut j = j_start;
            loop {
                let r = i - j;
                let c = j;
                let (color_order, hash) =
                    palette_color_context(&map, block_w as usize, r as usize, c as usize, n);
                let ctx = palette_color_ctx_from_hash(hash);
                let raw = td.decode_palette_color_idx(blk.size_y, ctx, true)?;
                if (raw as usize) >= n {
                    return Err(Error::invalid(format!(
                        "av1 palette_color_idx_y: symbol {raw} >= palette_size {n} (§5.11.49)"
                    )));
                }
                let pal_idx = color_order[raw as usize];
                map[(r as usize) * (block_w as usize) + (c as usize)] = pal_idx;
                if j == j_low {
                    break;
                }
                j -= 1;
            }
        }
    }

    // Onscreen → fullblock fixup (§5.11.49 tail loops).
    if onscreen_w < block_w {
        for row in 0..(onscreen_h.min(block_h)) {
            let last = map[(row as usize) * (block_w as usize) + (onscreen_w - 1) as usize];
            for col in onscreen_w..block_w {
                map[(row as usize) * (block_w as usize) + (col as usize)] = last;
            }
        }
    }
    if onscreen_h < block_h {
        for row in onscreen_h..block_h {
            for col in 0..block_w {
                let src = if onscreen_h > 0 {
                    map[((onscreen_h - 1) as usize) * (block_w as usize) + (col as usize)]
                } else {
                    first
                };
                map[(row as usize) * (block_w as usize) + (col as usize)] = src;
            }
        }
    }

    blk.color_map_y = map;
    Ok(())
}

/// §5.11.49 `palette_tokens()` — chroma index map. Identical to the
/// luma path except for the `< 4` block-dim fixup that pads tiny
/// blocks out to 4×4 (so the cdf indexing stays within range).
fn decode_palette_tokens_uv(
    td: &mut TileDecoder<'_>,
    blk: &mut PaletteBlock,
    fs: &FrameState,
    bw: u32,
    bh: u32,
    x: u32,
    y: u32,
) -> Result<()> {
    let mi_col = x >> 2;
    let mi_row = y >> 2;
    let mut block_w = bw >> fs.sub_x;
    let mut block_h = bh >> fs.sub_y;
    let mut onscreen_w = (bw.min((fs.mi_cols - mi_col) * 4)) >> fs.sub_x;
    let mut onscreen_h = (bh.min((fs.mi_rows - mi_row) * 4)) >> fs.sub_y;
    if block_w < 4 {
        block_w += 2;
        onscreen_w += 2;
    }
    if block_h < 4 {
        block_h += 2;
        onscreen_h += 2;
    }
    blk.map_w_uv = block_w;
    blk.map_h_uv = block_h;
    let mut map = vec![0u8; (block_w as usize) * (block_h as usize)];

    let first = decode_uniform_ns(&mut td.symbol, blk.size_uv as u16) as u8;
    map[0] = first;

    if onscreen_w > 0 && onscreen_h > 0 {
        let n = blk.size_uv as usize;
        let i_max = onscreen_w + onscreen_h - 1;
        for i in 1..i_max {
            let j_start = i.min(onscreen_w - 1);
            let j_low = (i + 1).saturating_sub(onscreen_h);
            let mut j = j_start;
            loop {
                let r = i - j;
                let c = j;
                let (color_order, hash) =
                    palette_color_context(&map, block_w as usize, r as usize, c as usize, n);
                let ctx = palette_color_ctx_from_hash(hash);
                let raw = td.decode_palette_color_idx(blk.size_uv, ctx, false)?;
                if (raw as usize) >= n {
                    return Err(Error::invalid(format!(
                        "av1 palette_color_idx_uv: symbol {raw} >= palette_size {n} (§5.11.49)"
                    )));
                }
                let pal_idx = color_order[raw as usize];
                map[(r as usize) * (block_w as usize) + (c as usize)] = pal_idx;
                if j == j_low {
                    break;
                }
                j -= 1;
            }
        }
    }

    if onscreen_w < block_w {
        for row in 0..(onscreen_h.min(block_h)) {
            let last = map[(row as usize) * (block_w as usize) + (onscreen_w - 1) as usize];
            for col in onscreen_w..block_w {
                map[(row as usize) * (block_w as usize) + (col as usize)] = last;
            }
        }
    }
    if onscreen_h < block_h {
        for row in onscreen_h..block_h {
            for col in 0..block_w {
                let src = if onscreen_h > 0 {
                    map[((onscreen_h - 1) as usize) * (block_w as usize) + (col as usize)]
                } else {
                    first
                };
                map[(row as usize) * (block_w as usize) + (col as usize)] = src;
            }
        }
    }

    blk.color_map_uv = map;
    Ok(())
}

/// Spec §5.11.50 `get_palette_color_context` — score the three
/// neighbours (left, top-left, top), bubble-sort the palette indices
/// by score, and fold the top scores into a single hash. Returns
/// `(color_order, color_context_hash)`. `color_order[0..n]` is the
/// re-ranked palette index list (most-likely first).
pub fn palette_color_context(
    color_map: &[u8],
    stride: usize,
    r: usize,
    c: usize,
    n: usize,
) -> ([u8; PALETTE_COLORS], u32) {
    let mut scores = [0u32; PALETTE_COLORS];
    let mut color_order = [0u8; PALETTE_COLORS];
    for (i, slot) in color_order.iter_mut().enumerate() {
        *slot = i as u8;
    }
    if c > 0 {
        let neighbour = color_map[r * stride + (c - 1)] as usize;
        if neighbour < n {
            scores[neighbour] += 2;
        }
    }
    if r > 0 && c > 0 {
        let neighbour = color_map[(r - 1) * stride + (c - 1)] as usize;
        if neighbour < n {
            scores[neighbour] += 1;
        }
    }
    if r > 0 {
        let neighbour = color_map[(r - 1) * stride + c] as usize;
        if neighbour < n {
            scores[neighbour] += 2;
        }
    }
    // Insertion sort: bring the top-3 by score to the front of
    // `color_order` (with `scores` updated in lock-step). Spec
    // §5.11.50 walks `i` over `PALETTE_NUM_NEIGHBORS`; we cap at `n`
    // to avoid pulling unused palette indices forward.
    let n_top = PALETTE_NUM_NEIGHBORS.min(n);
    #[allow(clippy::needless_range_loop)]
    for i in 0..n_top {
        let mut max_score = scores[i];
        let mut max_idx = i;
        #[allow(clippy::needless_range_loop)]
        for j in (i + 1)..n {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
        }
        if max_idx != i {
            let saved_score = scores[max_idx];
            let saved_color = color_order[max_idx];
            // Shift everything between `i` and `max_idx` down by one.
            let mut k = max_idx;
            while k > i {
                scores[k] = scores[k - 1];
                color_order[k] = color_order[k - 1];
                k -= 1;
            }
            scores[i] = saved_score;
            color_order[i] = saved_color;
        }
    }
    let mut hash = 0u32;
    for (i, mult) in cdfs::PALETTE_COLOR_HASH_MULTIPLIERS
        .iter()
        .take(PALETTE_NUM_NEIGHBORS)
        .enumerate()
    {
        hash += scores[i] * mult;
    }
    (color_order, hash)
}

/// Map a `ColorContextHash` (output of [`palette_color_context`]) into
/// a 0..=4 CDF context via `Palette_Color_Context[]` (§9.3).
#[inline]
pub fn palette_color_ctx_from_hash(hash: u32) -> usize {
    let h = (hash as usize).min(PALETTE_MAX_COLOR_CONTEXT_HASH);
    cdfs::PALETTE_COLOR_CONTEXT[h] as usize
}

/// §4.10.10 `NS(n)` — uniform integer in `0..n` over the range coder.
/// Mirrors [`super::lr_unit::decode_uniform`] but lives here so the
/// palette path can stay self-contained.
fn decode_uniform_ns(sd: &mut SymbolDecoder<'_>, n: u16) -> u16 {
    if n <= 1 {
        return 0;
    }
    let l = (n as u32).ilog2() + 1; // FloorLog2(n) + 1
    let m = (1i32 << l) - n as i32;
    let v = sd.read_literal(l - 1) as i32;
    if v < m {
        v as u16
    } else {
        ((v << 1) - m + sd.read_literal(1) as i32) as u16
    }
}

/// Apply §7.11.4 luma palette prediction over a TX unit at
/// `(start_x, start_y)` of size `tx_w × tx_h`. The `block_origin`
/// argument is the block's top-left luma sample (where the colour
/// map's `[0][0]` lives); we offset into `color_map_y` accordingly.
pub fn apply_palette_luma(
    fs: &mut FrameState,
    blk: &PaletteBlock,
    block_origin: (u32, u32),
    start_x: u32,
    start_y: u32,
    tx_w: u32,
    tx_h: u32,
) {
    debug_assert!(blk.has_y());
    let (bx, by) = block_origin;
    let stride = fs.width as usize;
    if fs.bit_depth == 8 {
        for j in 0..tx_h {
            for i in 0..tx_w {
                let sx = start_x + i;
                let sy = start_y + j;
                if sx >= fs.width || sy >= fs.height {
                    continue;
                }
                let map_x = sx - bx;
                let map_y = sy - by;
                let v = blk.luma_sample(map_x, map_y);
                fs.y_plane[(sy as usize) * stride + (sx as usize)] = v as u8;
            }
        }
    } else {
        for j in 0..tx_h {
            for i in 0..tx_w {
                let sx = start_x + i;
                let sy = start_y + j;
                if sx >= fs.width || sy >= fs.height {
                    continue;
                }
                let map_x = sx - bx;
                let map_y = sy - by;
                let v = blk.luma_sample(map_x, map_y);
                fs.y_plane16[(sy as usize) * stride + (sx as usize)] = v;
            }
        }
    }
}

/// Apply §7.11.4 chroma palette prediction over a TX unit at the
/// chroma-resolution `(start_x, start_y)` of size `tx_w × tx_h`.
/// `plane` is 1 for U and 2 for V. `block_origin` is the block's
/// top-left **chroma** sample.
#[allow(clippy::too_many_arguments)]
pub fn apply_palette_chroma(
    fs: &mut FrameState,
    blk: &PaletteBlock,
    plane: usize,
    block_origin: (u32, u32),
    start_x: u32,
    start_y: u32,
    tx_w: u32,
    tx_h: u32,
) {
    debug_assert!(blk.has_uv());
    let (bx, by) = block_origin;
    let stride = fs.uv_width as usize;
    let plane_h = fs.uv_height;
    let plane_w = fs.uv_width;
    if fs.bit_depth == 8 {
        let buf = if plane == 1 {
            &mut fs.u_plane
        } else {
            &mut fs.v_plane
        };
        for j in 0..tx_h {
            for i in 0..tx_w {
                let sx = start_x + i;
                let sy = start_y + j;
                if sx >= plane_w || sy >= plane_h {
                    continue;
                }
                let map_x = sx - bx;
                let map_y = sy - by;
                let v = blk.chroma_sample(plane, map_x, map_y);
                buf[(sy as usize) * stride + (sx as usize)] = v as u8;
            }
        }
    } else {
        let buf = if plane == 1 {
            &mut fs.u_plane16
        } else {
            &mut fs.v_plane16
        };
        for j in 0..tx_h {
            for i in 0..tx_w {
                let sx = start_x + i;
                let sy = start_y + j;
                if sx >= plane_w || sy >= plane_h {
                    continue;
                }
                let map_x = sx - bx;
                let map_y = sy - by;
                let v = blk.chroma_sample(plane, map_x, map_y);
                buf[(sy as usize) * stride + (sx as usize)] = v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_log2_basic() {
        assert_eq!(ceil_log2(0), 0);
        assert_eq!(ceil_log2(1), 0);
        assert_eq!(ceil_log2(2), 1);
        assert_eq!(ceil_log2(3), 2);
        assert_eq!(ceil_log2(8), 3);
    }

    #[test]
    fn palette_color_context_no_neighbours_returns_identity() {
        let map = vec![0u8; 16];
        let (order, hash) = palette_color_context(&map, 4, 0, 0, 4);
        assert_eq!(hash, 0);
        for (i, v) in order[..4].iter().enumerate() {
            assert_eq!(*v as usize, i);
        }
    }

    #[test]
    fn palette_color_context_left_dominates() {
        // map[r=1][c=1] reads left (1,0)=2, top-left (0,0)=0, top (0,1)=2.
        // scores: 2->2 (left) + 2 (top) = 4, 0->1 (TL).
        // Top-3 by score: idx 2 (4), idx 0 (1), idx 1 (0).
        let mut map = vec![0u8; 4 * 2];
        map[4] = 2; // left at (r=1, c=0) → row*stride+col = 1*4+0
        map[0] = 0; // top-left at (0,0)
        map[1] = 2; // top at (0,1)
        let (order, hash) = palette_color_context(&map, 4, 1, 1, 4);
        assert_eq!(order[0], 2);
        assert_eq!(order[1], 0);
        assert!(hash > 0);
    }

    #[test]
    fn ns_uniform_handles_n_le_1() {
        // n == 0 / 1 should return 0 without touching the bitstream.
        let bits = vec![0xFFu8; 16];
        let mut sd = SymbolDecoder::new(&bits, bits.len(), false).expect("init");
        assert_eq!(decode_uniform_ns(&mut sd, 0), 0);
        assert_eq!(decode_uniform_ns(&mut sd, 1), 0);
    }
}
