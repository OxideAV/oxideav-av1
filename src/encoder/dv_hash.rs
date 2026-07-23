//! r425 — per-frame block-hash index for the §5.11.7 intra-block-copy
//! DV search (ladder item 5, screen-content completion).
//!
//! The r418 DV search ranked a small geometric candidate set
//! (superblock-stride / block-extent multiples) by luma SSD. That
//! finds aligned tile copies but misses the screen-content win case:
//! repeated glyphs and patterns at ARBITRARY (even) offsets. This
//! module adds an exact-match index so the search can jump straight
//! to identical source content anywhere in the frame.
//!
//! Everything here is free encoder engineering — the hash function,
//! bucket layout and seeding strategy are this crate's own design.
//! The index deliberately hashes the INPUT plane, not the running
//! reconstruction: a hit means "the source location carries
//! byte-identical input content", which stays true at every
//! quantiser (a reconstruction-space index would only match where
//! the source coded losslessly). The copy itself still runs against
//! the reconstruction — the caller re-validates every candidate
//! through the §6.10.24 predicate
//! ([`crate::encoder::key_frame::intrabc_dv_valid`]) and ranks it by
//! reconstruction-space SSD before the exact-twin-bits RD election,
//! so a stale seed can lose but never corrupt.
//!
//! Design:
//!
//! * **Base tier**: an 8×8 FNV-1a hash over the 64 raster bytes at
//!   EVEN positions (the driver restricts DVs to even whole-pel
//!   offsets so the 4:2:0 chroma copy stays integer-aligned; block
//!   origins are multiples of 8, so even source positions are
//!   exactly the reachable ones).
//! * **Composed tiers**: 16 / 32 / 64 hashes are built from the four
//!   quadrant hashes one tier down (FNV-1a over the quadrants'
//!   little-endian words, NW → NE → SW → SE). Composition makes the
//!   larger tiers O(1) per position on top of the base grid; the
//!   probe side ([`hash_block_direct`]) runs the same recursion so
//!   index and probe agree bit-for-bit.
//! * **Flat-block suppression**: uniform blocks (every sample equal)
//!   are neither indexed nor probed — DC / palette arms already code
//!   flat content at near-zero cost and flat runs would otherwise
//!   flood single buckets. Buckets are additionally capped at
//!   [`BUCKET_CAP`] positions (raster order, earliest first — the
//!   §6.10.24 lag reaches early sources soonest).
//!
//! The index is only armed on intra frames whose §5.9.20
//! `allow_intrabc` gate opened, and only up to [`MAX_INDEXED_AREA`]
//! luma samples (beyond that the geometric candidate set stands
//! alone — the grids would dominate the encoder's working set).

use std::collections::HashMap;

/// Block edge lengths with a hash tier, ascending.
pub(crate) const DV_HASH_SIZES: [usize; 4] = [8, 16, 32, 64];

/// Per-tier position step in samples. The base tiers walk every even
/// position; the 32 / 64 tiers thin to 4- / 8-sample steps to bound
/// the map population (they seed large-pattern copies, which in
/// screen content overwhelmingly recur on coarse strides — and every
/// fine-grained repeat is still caught by the 8 / 16 tiers after the
/// partition search splits).
const DV_HASH_STEPS: [usize; 4] = [2, 2, 4, 8];

/// Hard cap on positions remembered per (tier, hash) bucket. Once a
/// pattern has this many registered sources, further duplicates add
/// no search power (any §6.10.24-reachable hit already yields an
/// exact match).
const BUCKET_CAP: usize = 32;

/// Luma-sample area bound for arming the index (2048×2048).
const MAX_INDEXED_AREA: usize = 1 << 22;

const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;

#[inline]
fn fnv_mix(mut hash: u64, v: u16) -> u64 {
    hash ^= u64::from(v);
    hash = hash.wrapping_mul(FNV_PRIME);
    hash
}

/// FNV-1a over an 8×8 raster block. Returns `(hash, uniform)`.
#[inline]
fn hash8(plane: &[u16], stride: usize, y: usize, x: usize) -> (u64, bool) {
    let first = plane[y * stride + x];
    let mut hash = FNV_BASIS;
    let mut uniform = true;
    for i in 0..8 {
        let row = &plane[(y + i) * stride + x..][..8];
        for &v in row {
            uniform &= v == first;
            hash = fnv_mix(hash, v);
        }
    }
    (hash, uniform)
}

/// Compose a parent hash from its four quadrant hashes (NW, NE, SW,
/// SE). A parent is uniform iff all quadrants are uniform AND hash
/// identically (uniform quadrants hash by value, so equal hashes ⇔
/// equal fill values).
#[inline]
fn compose4(q: [(u64, bool); 4]) -> (u64, bool) {
    let mut hash = FNV_BASIS;
    let mut uniform = true;
    for &(qh, qu) in &q {
        uniform &= qu && qh == q[0].0;
        for b in qh.to_le_bytes() {
            hash = fnv_mix(hash, u16::from(b));
        }
    }
    (hash, uniform)
}

/// Hash one `size`×`size` block of `plane` through the index's
/// composition recursion — the probe-side twin of the index build.
/// `size` MUST be in [`DV_HASH_SIZES`] and the block fully inside the
/// plane. Returns `(hash, uniform)`.
pub(crate) fn hash_block_direct(
    plane: &[u16],
    stride: usize,
    y: usize,
    x: usize,
    size: usize,
) -> (u64, bool) {
    if size == 8 {
        return hash8(plane, stride, y, x);
    }
    let h = size / 2;
    compose4([
        hash_block_direct(plane, stride, y, x, h),
        hash_block_direct(plane, stride, y, x + h, h),
        hash_block_direct(plane, stride, y + h, x, h),
        hash_block_direct(plane, stride, y + h, x + h, h),
    ])
}

/// Tier index for a square block edge, when one exists.
#[inline]
pub(crate) fn dv_hash_size_idx(size: usize) -> Option<usize> {
    DV_HASH_SIZES.iter().position(|&s| s == size)
}

/// The per-frame index. `Default` is the inert empty state (never
/// matches) — the KEY driver arms it with [`DvHashIndex::build`]
/// only when the frame-level §5.9.20 gate opens.
#[derive(Default)]
pub(crate) struct DvHashIndex {
    /// Per-tier `hash → source top-left (y, x)` buckets, insertion in
    /// raster order (earliest first).
    maps: [HashMap<u64, Vec<(u16, u16)>>; 4],
}

impl DvHashIndex {
    /// Build the index over a `width`×`height` input luma plane.
    /// Falls back to the inert state above [`MAX_INDEXED_AREA`].
    pub(crate) fn build(plane: &[u16], width: usize, height: usize) -> Self {
        let mut out = DvHashIndex::default();
        if width < 8 || height < 8 || width * height > MAX_INDEXED_AREA {
            return out;
        }
        // Base grid: 8×8 hashes at every even position, then composed
        // tiers from the quadrants one tier down.
        let gw = (width - 8) / 2 + 1;
        let gh = (height - 8) / 2 + 1;
        let mut h8_grid = vec![(0u64, false); gw * gh];
        for gy in 0..gh {
            for gx in 0..gw {
                h8_grid[gy * gw + gx] = hash8(plane, width, gy * 2, gx * 2);
            }
        }
        let composed = |y: usize, x: usize, size: usize| -> (u64, bool) {
            fn go(grid: &[(u64, bool)], gw: usize, y: usize, x: usize, size: usize) -> (u64, bool) {
                if size == 8 {
                    return grid[(y / 2) * gw + x / 2];
                }
                let h = size / 2;
                compose4([
                    go(grid, gw, y, x, h),
                    go(grid, gw, y, x + h, h),
                    go(grid, gw, y + h, x, h),
                    go(grid, gw, y + h, x + h, h),
                ])
            }
            go(&h8_grid, gw, y, x, size)
        };
        for (tier, (&size, &step)) in DV_HASH_SIZES.iter().zip(DV_HASH_STEPS.iter()).enumerate() {
            if width < size || height < size {
                continue;
            }
            let mut y = 0usize;
            while y + size <= height {
                let mut x = 0usize;
                while x + size <= width {
                    let (hash, uniform) = composed(y, x, size);
                    if !uniform {
                        let bucket = out.maps[tier].entry(hash).or_default();
                        if bucket.len() < BUCKET_CAP {
                            bucket.push((y as u16, x as u16));
                        }
                    }
                    x += step;
                }
                y += step;
            }
        }
        out
    }

    /// Whether any tier holds at least one source (test hook).
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.maps.iter().all(HashMap::is_empty)
    }

    /// Exact-match DV seeds for a block of edge `DV_HASH_SIZES[tier]`
    /// at `(row0, col0)` whose probe hash is `hash`: whole-pel
    /// `(dv_r, dv_c)` offsets, nearest (smallest `|dv_r| + |dv_c|`)
    /// first. Seeds only — the caller still runs the §6.10.24
    /// validity predicate and the SSD/RD ranking, which is also what
    /// discards the sources the raster lag cannot reach yet.
    pub(crate) fn candidates(
        &self,
        hash: u64,
        tier: usize,
        row0: usize,
        col0: usize,
    ) -> Vec<(i32, i32)> {
        let Some(bucket) = self.maps[tier].get(&hash) else {
            return Vec::new();
        };
        let mut out: Vec<(i32, i32)> = bucket
            .iter()
            .map(|&(sy, sx)| (sy as i32 - row0 as i32, sx as i32 - col0 as i32))
            .filter(|&(dr, dc)| (dr, dc) != (0, 0))
            .collect();
        out.sort_by_key(|&(dr, dc)| dr.unsigned_abs() + dc.unsigned_abs());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern_plane(w: usize, h: usize, seed: u32) -> Vec<u16> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        (0..w * h).map(|_| (next() & 0xff) as u16).collect()
    }

    /// The grid-composed build and the direct probe recursion must
    /// agree on every tier.
    #[test]
    fn composed_equals_direct() {
        let (w, h) = (192, 128);
        let plane = pattern_plane(w, h, 0x2468);
        let idx = DvHashIndex::build(&plane, w, h);
        for &(y, x, s) in &[
            (0usize, 0usize, 8usize),
            (2, 6, 8),
            (16, 40, 16),
            (10, 22, 16),
            (32, 64, 32),
            (4, 8, 32),
            (0, 128, 64),
            (64, 0, 64),
        ] {
            // Any position on the tier's step grid must be findable
            // by its direct probe hash at zero displacement offset.
            let (hash, uniform) = hash_block_direct(&plane, w, y, x, s);
            assert!(!uniform);
            let tier = dv_hash_size_idx(s).unwrap();
            if y % DV_HASH_STEPS[tier] == 0 && x % DV_HASH_STEPS[tier] == 0 {
                let probe_from = (y + 16, x + 16);
                let cands = idx.candidates(hash, tier, probe_from.0, probe_from.1);
                assert!(
                    cands.contains(&(-16, -16)),
                    "index missed its own content at ({y},{x})x{s}"
                );
            }
        }
    }

    /// Uniform blocks are suppressed on both the index and probe
    /// sides.
    #[test]
    fn uniform_suppression() {
        let (w, h) = (128, 128);
        let mut plane = vec![77u16; w * h];
        let (_, uniform) = hash_block_direct(&plane, w, 0, 0, 64);
        assert!(uniform);
        plane[63 * w + 63] = 78;
        let (_, uniform) = hash_block_direct(&plane, w, 0, 0, 64);
        assert!(!uniform);
        // A fully-flat frame indexes nothing.
        let flat = vec![13u16; w * h];
        let idx = DvHashIndex::build(&flat, w, h);
        assert!(idx.is_empty());
    }

    /// Off-grid even positions are indexed on the fine tiers: a glyph
    /// at (6, 26) must be findable from any probe position.
    #[test]
    fn off_stride_positions_indexed() {
        let (w, h) = (256, 128);
        let mut plane = vec![0u16; w * h];
        let glyph = pattern_plane(8, 8, 0x1357);
        for i in 0..8 {
            for j in 0..8 {
                plane[(6 + i) * w + 26 + j] = glyph[i * 8 + j];
            }
        }
        let idx = DvHashIndex::build(&plane, w, h);
        let (hash, uniform) = hash_block_direct(&plane, w, 6, 26, 8);
        assert!(!uniform);
        let cands = idx.candidates(hash, 0, 96, 200);
        assert!(cands.contains(&(6 - 96, 26 - 200)));
    }

    /// Bucket population respects the cap and candidates come back
    /// nearest-first.
    #[test]
    fn bucket_cap_and_nearest_order() {
        let (w, h) = (448, 192);
        // Tile a non-uniform 8x8 glyph everywhere: one giant bucket.
        let mut plane = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                plane[y * w + x] = ((x % 8) * 8 + (y % 8)) as u16;
            }
        }
        let idx = DvHashIndex::build(&plane, w, h);
        let (hash, uniform) = hash_block_direct(&plane, w, 0, 0, 8);
        assert!(!uniform);
        let bucket = idx.maps[0].get(&hash).expect("glyph bucket");
        assert_eq!(bucket.len(), BUCKET_CAP);
        let cands = idx.candidates(hash, 0, 64, 64);
        for pair in cands.windows(2) {
            let d0 = pair[0].0.unsigned_abs() + pair[0].1.unsigned_abs();
            let d1 = pair[1].0.unsigned_abs() + pair[1].1.unsigned_abs();
            assert!(d0 <= d1, "candidates not nearest-first: {cands:?}");
        }
    }

    /// The area guard leaves oversize frames inert.
    #[test]
    fn area_guard() {
        let idx = DvHashIndex::build(&[], 4096, 4096);
        assert!(idx.is_empty());
    }
}
