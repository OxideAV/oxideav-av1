//! AV1 block-size + partition-type tables — §3 `BLOCK_*` / `PARTITION_*`.
//!
//! For Phase 2 (mode decode only, no coefficient decode, no pixel
//! reconstruction) we keep the spec's block-size / partition-type
//! taxonomy and drop the reconstruction-specific types.

/// `BlockSize` identifies one of AV1's block shapes (spec §3 `BLOCK_*`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockSize {
    Block4x4 = 0,
    Block4x8 = 1,
    Block8x4 = 2,
    Block8x8 = 3,
    Block8x16 = 4,
    Block16x8 = 5,
    Block16x16 = 6,
    Block16x32 = 7,
    Block32x16 = 8,
    Block32x32 = 9,
    Block32x64 = 10,
    Block64x32 = 11,
    Block64x64 = 12,
    Block64x128 = 13,
    Block128x64 = 14,
    Block128x128 = 15,
    Block4x16 = 16,
    Block16x4 = 17,
    Block8x32 = 18,
    Block32x8 = 19,
    Block16x64 = 20,
    Block64x16 = 21,
    Invalid = 22,
}

const BLOCK_WIDTHS: [u32; 23] = [
    4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 64, 128, 128, 4, 16, 8, 32, 16, 64, 0,
];

const BLOCK_HEIGHTS: [u32; 23] = [
    4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 128, 64, 128, 16, 4, 32, 8, 64, 16, 0,
];

impl BlockSize {
    /// Block width in luma samples.
    pub fn width(self) -> u32 {
        BLOCK_WIDTHS[self as usize]
    }

    /// Block height in luma samples.
    pub fn height(self) -> u32 {
        BLOCK_HEIGHTS[self as usize]
    }

    /// `true` when the block is square — needed to choose between
    /// certain partition types (§5.11.4 only splits square nodes).
    pub fn is_square(self) -> bool {
        self.width() == self.height()
    }

    /// MI-grid width: AV1's smallest unit is 4×4 luma samples, so block
    /// dimensions divide by 4 to give MI rows/cols (spec §3).
    pub fn mi_width(self) -> u32 {
        self.width() >> 2
    }

    /// MI-grid height — see `mi_width`.
    pub fn mi_height(self) -> u32 {
        self.height() >> 2
    }

    /// MI dimensions for a chroma plane at the given subsampling
    /// factors. Always clamps to at least 1×1 so the tiniest blocks
    /// still have a chroma MI footprint.
    pub fn subsampled_mi_dims(self, sub_x: u32, sub_y: u32) -> (u32, u32) {
        let mw = (self.mi_width() >> sub_x).max(1);
        let mh = (self.mi_height() >> sub_y).max(1);
        (mw, mh)
    }

    /// `Subsampled_Size[ subsize ][ subx ][ suby ]` per AV1 spec
    /// §5.11.38 (table on p.88). Maps a luma block size + chroma
    /// subsampling factors to the chroma residual block size — used by
    /// `get_plane_residual_size( subsize, plane )`. Returns
    /// `BlockSize::Invalid` for spec entries marked `BLOCK_INVALID`
    /// (combinations the encoder must not emit; bitstream conformance
    /// requires `get_plane_residual_size( ., 1 ) != BLOCK_INVALID`).
    ///
    /// Without this table a naive `(w >> sub_x) × (h >> sub_y)` halving
    /// produces TX dimensions like 8×2 or 2×8 for narrow inter blocks
    /// (e.g. `Block16x4` + 4:2:0), which are not in the AV1 TX set —
    /// the spec collapses them to 8×4 (resp. 4×8) instead. See spec
    /// table excerpt:
    ///   { {BLOCK_16X4,  BLOCK_INVALID},  {BLOCK_8X4,  BLOCK_8X4} },
    ///   { {BLOCK_4X16,  BLOCK_4X8},      {BLOCK_INVALID, BLOCK_4X8} },
    pub fn subsampled_size(self, sub_x: u32, sub_y: u32) -> BlockSize {
        // Indexing: `[subx as usize][suby as usize]`. Spec lays out
        // entries as `[subx][suby]` in `Subsampled_Size[][2][2]`.
        let sx = (sub_x & 1) as usize;
        let sy = (sub_y & 1) as usize;
        // Each row matches one luma block size in the spec table order.
        // Inner literals are flattened to `[(0,0),(0,1),(1,0),(1,1)]`.
        const I: BlockSize = BlockSize::Invalid;
        let row: [BlockSize; 4] = match self {
            BlockSize::Block4x4 => [
                BlockSize::Block4x4, // (0,0)
                BlockSize::Block4x4, // (0,1)
                BlockSize::Block4x4, // (1,0)
                BlockSize::Block4x4, // (1,1)
            ],
            BlockSize::Block4x8 => [
                BlockSize::Block4x8, // (0,0)
                BlockSize::Block4x4, // (0,1)
                I,                   // (1,0)
                BlockSize::Block4x4, // (1,1)
            ],
            BlockSize::Block8x4 => [
                BlockSize::Block8x4, // (0,0)
                I,                   // (0,1)
                BlockSize::Block4x4, // (1,0)
                BlockSize::Block4x4, // (1,1)
            ],
            BlockSize::Block8x8 => [
                BlockSize::Block8x8,
                BlockSize::Block8x4,
                BlockSize::Block4x8,
                BlockSize::Block4x4,
            ],
            BlockSize::Block8x16 => [
                BlockSize::Block8x16,
                BlockSize::Block8x8,
                I,
                BlockSize::Block4x8,
            ],
            BlockSize::Block16x8 => [
                BlockSize::Block16x8,
                I,
                BlockSize::Block8x8,
                BlockSize::Block8x4,
            ],
            BlockSize::Block16x16 => [
                BlockSize::Block16x16,
                BlockSize::Block16x8,
                BlockSize::Block8x16,
                BlockSize::Block8x8,
            ],
            BlockSize::Block16x32 => [
                BlockSize::Block16x32,
                BlockSize::Block16x16,
                I,
                BlockSize::Block8x16,
            ],
            BlockSize::Block32x16 => [
                BlockSize::Block32x16,
                I,
                BlockSize::Block16x16,
                BlockSize::Block16x8,
            ],
            BlockSize::Block32x32 => [
                BlockSize::Block32x32,
                BlockSize::Block32x16,
                BlockSize::Block16x32,
                BlockSize::Block16x16,
            ],
            BlockSize::Block32x64 => [
                BlockSize::Block32x64,
                BlockSize::Block32x32,
                I,
                BlockSize::Block16x32,
            ],
            BlockSize::Block64x32 => [
                BlockSize::Block64x32,
                I,
                BlockSize::Block32x32,
                BlockSize::Block32x16,
            ],
            BlockSize::Block64x64 => [
                BlockSize::Block64x64,
                BlockSize::Block64x32,
                BlockSize::Block32x64,
                BlockSize::Block32x32,
            ],
            BlockSize::Block64x128 => [
                BlockSize::Block64x128,
                BlockSize::Block64x64,
                I,
                BlockSize::Block32x64,
            ],
            BlockSize::Block128x64 => [
                BlockSize::Block128x64,
                I,
                BlockSize::Block64x64,
                BlockSize::Block64x32,
            ],
            BlockSize::Block128x128 => [
                BlockSize::Block128x128,
                BlockSize::Block128x64,
                BlockSize::Block64x128,
                BlockSize::Block64x64,
            ],
            BlockSize::Block4x16 => [
                BlockSize::Block4x16,
                BlockSize::Block4x8,
                I,
                BlockSize::Block4x8,
            ],
            BlockSize::Block16x4 => [
                BlockSize::Block16x4,
                I,
                BlockSize::Block8x4,
                BlockSize::Block8x4,
            ],
            BlockSize::Block8x32 => [
                BlockSize::Block8x32,
                BlockSize::Block8x16,
                I,
                BlockSize::Block4x16,
            ],
            BlockSize::Block32x8 => [
                BlockSize::Block32x8,
                I,
                BlockSize::Block16x8,
                BlockSize::Block16x4,
            ],
            BlockSize::Block16x64 => [
                BlockSize::Block16x64,
                BlockSize::Block16x32,
                I,
                BlockSize::Block8x32,
            ],
            BlockSize::Block64x16 => [
                BlockSize::Block64x16,
                I,
                BlockSize::Block32x16,
                BlockSize::Block32x8,
            ],
            BlockSize::Invalid => [I, I, I, I],
        };
        // Ordering inside `row` is `(subx,suby) ∈ {(0,0),(0,1),(1,0),(1,1)}`.
        let idx = (sx << 1) | sy;
        row[idx]
    }

    /// Largest transform that fits in this block, capped at 64×64 (AV1
    /// hard ceiling). Used by the implicit-TX path.
    pub fn max_tx_size(self) -> (u32, u32) {
        (self.width().min(64), self.height().min(64))
    }

    /// `Max_Tx_Size_Rect[MiSize]` per AV1 spec §Additional tables.
    /// Returns the largest transform (square or rectangular) that can
    /// be used for blocks of this size — used by §5.11.15
    /// `read_tx_size()` to initialise `TxSize`. Returns `None` when
    /// `self == Invalid`.
    pub fn max_tx_size_rect(self) -> Option<crate::transform::TxSize> {
        use crate::transform::TxSize;
        Some(match self {
            Self::Block4x4 => TxSize::Tx4x4,
            Self::Block4x8 => TxSize::Tx4x8,
            Self::Block8x4 => TxSize::Tx8x4,
            Self::Block8x8 => TxSize::Tx8x8,
            Self::Block8x16 => TxSize::Tx8x16,
            Self::Block16x8 => TxSize::Tx16x8,
            Self::Block16x16 => TxSize::Tx16x16,
            Self::Block16x32 => TxSize::Tx16x32,
            Self::Block32x16 => TxSize::Tx32x16,
            Self::Block32x32 => TxSize::Tx32x32,
            Self::Block32x64 => TxSize::Tx32x64,
            Self::Block64x32 => TxSize::Tx64x32,
            Self::Block64x64 => TxSize::Tx64x64,
            Self::Block64x128 => TxSize::Tx64x64,
            Self::Block128x64 => TxSize::Tx64x64,
            Self::Block128x128 => TxSize::Tx64x64,
            Self::Block4x16 => TxSize::Tx4x16,
            Self::Block16x4 => TxSize::Tx16x4,
            Self::Block8x32 => TxSize::Tx8x32,
            Self::Block32x8 => TxSize::Tx32x8,
            Self::Block16x64 => TxSize::Tx16x64,
            Self::Block64x16 => TxSize::Tx64x16,
            Self::Invalid => return None,
        })
    }

    /// `Max_Tx_Depth[MiSize]` per AV1 spec §5.11.15. Picks which
    /// `tx_depth` CDF category (8×8 / 16×16 / 32×32 / 64×64) applies
    /// when reading `tx_depth`. Returns 0 for `Block4x4` (no tx_depth
    /// symbol needed) and `Invalid`.
    pub fn max_tx_depth(self) -> u32 {
        match self {
            Self::Block4x4 => 0,
            Self::Block4x8 | Self::Block8x4 | Self::Block8x8 => 1,
            Self::Block8x16
            | Self::Block16x8
            | Self::Block16x16
            | Self::Block4x16
            | Self::Block16x4 => 2,
            Self::Block16x32
            | Self::Block32x16
            | Self::Block32x32
            | Self::Block8x32
            | Self::Block32x8 => 3,
            Self::Block32x64
            | Self::Block64x32
            | Self::Block64x64
            | Self::Block64x128
            | Self::Block128x64
            | Self::Block128x128
            | Self::Block16x64
            | Self::Block64x16 => 4,
            Self::Invalid => 0,
        }
    }
}

/// `PartitionType` enumerates the partition decisions at each node of
/// the partition tree (spec §3 `PARTITION_*`). Numeric values match the
/// CDF symbol indices in `DEFAULT_PARTITION_CDF`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionType {
    None = 0,
    Horz = 1,
    Vert = 2,
    Split = 3,
    HorzA = 4,
    HorzB = 5,
    VertA = 6,
    VertB = 7,
    Horz4 = 8,
    Vert4 = 9,
}

impl PartitionType {
    /// Decode a partition symbol value into the enum. Only `0..=9` are
    /// defined by the spec; anything else is a bitstream violation.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::None,
            1 => Self::Horz,
            2 => Self::Vert,
            3 => Self::Split,
            4 => Self::HorzA,
            5 => Self::HorzB,
            6 => Self::VertA,
            7 => Self::VertB,
            8 => Self::Horz4,
            9 => Self::Vert4,
            _ => return None,
        })
    }
}

/// A single leaf block emitted by the partition tree traversal. Used by
/// the oracle-driven `WalkPartition` helper in [`super::partition`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubBlock {
    pub x: u32,
    pub y: u32,
    pub size: BlockSize,
}

/// Square block obtained by splitting `bs` into four quadrants —
/// spec §5.11.4 `PARTITION_SPLIT`. Returns `Invalid` for already-min
/// blocks; callers gate the recursion on `bs == Block4x4` before
/// calling.
pub fn quarter_size(bs: BlockSize) -> BlockSize {
    match bs {
        BlockSize::Block128x128 => BlockSize::Block64x64,
        BlockSize::Block64x64 => BlockSize::Block32x32,
        BlockSize::Block32x32 => BlockSize::Block16x16,
        BlockSize::Block16x16 => BlockSize::Block8x8,
        BlockSize::Block8x8 => BlockSize::Block4x4,
        _ => BlockSize::Invalid,
    }
}

/// Block size obtained by splitting `bs` along the given axis. When
/// `axis_is_horz` is true the split stacks halves vertically (two
/// horizontally-long halves); otherwise it is a side-by-side vertical
/// split.
pub fn half_below_size(bs: BlockSize, axis_is_horz: bool) -> BlockSize {
    if axis_is_horz {
        match bs {
            BlockSize::Block128x128 => BlockSize::Block128x64,
            BlockSize::Block64x64 => BlockSize::Block64x32,
            BlockSize::Block32x32 => BlockSize::Block32x16,
            BlockSize::Block16x16 => BlockSize::Block16x8,
            BlockSize::Block8x8 => BlockSize::Block8x4,
            _ => BlockSize::Invalid,
        }
    } else {
        match bs {
            BlockSize::Block128x128 => BlockSize::Block64x128,
            BlockSize::Block64x64 => BlockSize::Block32x64,
            BlockSize::Block32x32 => BlockSize::Block16x32,
            BlockSize::Block16x16 => BlockSize::Block8x16,
            BlockSize::Block8x8 => BlockSize::Block4x8,
            _ => BlockSize::Invalid,
        }
    }
}

/// Block size of each of the 4 horizontal stripes produced by
/// `PARTITION_HORZ_4` on `bs`.
pub fn horz4_size(bs: BlockSize) -> BlockSize {
    match bs {
        BlockSize::Block16x16 => BlockSize::Block16x4,
        BlockSize::Block32x32 => BlockSize::Block32x8,
        BlockSize::Block64x64 => BlockSize::Block64x16,
        _ => BlockSize::Block4x4,
    }
}

/// Block size of each of the 4 vertical stripes produced by
/// `PARTITION_VERT_4` on `bs`.
pub fn vert4_size(bs: BlockSize) -> BlockSize {
    match bs {
        BlockSize::Block16x16 => BlockSize::Block4x16,
        BlockSize::Block32x32 => BlockSize::Block8x32,
        BlockSize::Block64x64 => BlockSize::Block16x64,
        _ => BlockSize::Block4x4,
    }
}

/// Reverse lookup: turn a `(width, height)` luma pair into the matching
/// `BlockSize` enumerator, or `BlockSize::Invalid` if no entry exists.
/// Useful on the inter-decode path where the caller has `bw / bh` in
/// samples but needs the `MiSize` index for CDF / context lookups.
pub fn block_size_from_wh(w: u32, h: u32) -> BlockSize {
    for (i, (bw, bh)) in BLOCK_WIDTHS.iter().zip(BLOCK_HEIGHTS.iter()).enumerate() {
        if *bw == w && *bh == h {
            return match i {
                0 => BlockSize::Block4x4,
                1 => BlockSize::Block4x8,
                2 => BlockSize::Block8x4,
                3 => BlockSize::Block8x8,
                4 => BlockSize::Block8x16,
                5 => BlockSize::Block16x8,
                6 => BlockSize::Block16x16,
                7 => BlockSize::Block16x32,
                8 => BlockSize::Block32x16,
                9 => BlockSize::Block32x32,
                10 => BlockSize::Block32x64,
                11 => BlockSize::Block64x32,
                12 => BlockSize::Block64x64,
                13 => BlockSize::Block64x128,
                14 => BlockSize::Block128x64,
                15 => BlockSize::Block128x128,
                16 => BlockSize::Block4x16,
                17 => BlockSize::Block16x4,
                18 => BlockSize::Block8x32,
                19 => BlockSize::Block32x8,
                20 => BlockSize::Block16x64,
                21 => BlockSize::Block64x16,
                _ => BlockSize::Invalid,
            };
        }
    }
    BlockSize::Invalid
}

/// BSL (block-size-log) category used for the partition CDF index:
/// 0 = 8×8, 1 = 16×16, 2 = 32×32, 3 = 64×64, 4 = 128×128.
pub fn block_size_log(bs: BlockSize) -> u32 {
    match bs {
        BlockSize::Block8x8 => 0,
        BlockSize::Block16x16 => 1,
        BlockSize::Block32x32 => 2,
        BlockSize::Block64x64 => 3,
        BlockSize::Block128x128 => 4,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_match_heights_for_square_blocks() {
        for bs in [
            BlockSize::Block4x4,
            BlockSize::Block8x8,
            BlockSize::Block16x16,
            BlockSize::Block32x32,
            BlockSize::Block64x64,
            BlockSize::Block128x128,
        ] {
            assert!(bs.is_square(), "{bs:?}");
            assert_eq!(bs.width(), bs.height());
        }
    }

    #[test]
    fn quarter_of_64x64_is_32x32() {
        assert_eq!(quarter_size(BlockSize::Block64x64), BlockSize::Block32x32);
        assert_eq!(quarter_size(BlockSize::Block4x4), BlockSize::Invalid);
    }

    // §Additional tables `Max_Tx_Size_Rect` — each block size picks the
    // largest TX that fits. Square blocks map to the square TX of the
    // same side; rectangular blocks pick the matching rectangle; 128+
    // blocks cap at 64×64.
    #[test]
    fn max_tx_size_rect_matches_spec_table() {
        use crate::transform::TxSize;
        assert_eq!(BlockSize::Block4x4.max_tx_size_rect(), Some(TxSize::Tx4x4));
        assert_eq!(BlockSize::Block8x8.max_tx_size_rect(), Some(TxSize::Tx8x8));
        assert_eq!(
            BlockSize::Block32x32.max_tx_size_rect(),
            Some(TxSize::Tx32x32)
        );
        assert_eq!(
            BlockSize::Block64x64.max_tx_size_rect(),
            Some(TxSize::Tx64x64)
        );
        // §5.11.27 — 128-wide/tall blocks cap the TX at 64×64.
        assert_eq!(
            BlockSize::Block128x128.max_tx_size_rect(),
            Some(TxSize::Tx64x64)
        );
        assert_eq!(
            BlockSize::Block128x64.max_tx_size_rect(),
            Some(TxSize::Tx64x64)
        );
        // Rectangular entries.
        assert_eq!(BlockSize::Block4x8.max_tx_size_rect(), Some(TxSize::Tx4x8));
        assert_eq!(
            BlockSize::Block16x32.max_tx_size_rect(),
            Some(TxSize::Tx16x32)
        );
        assert_eq!(
            BlockSize::Block64x16.max_tx_size_rect(),
            Some(TxSize::Tx64x16)
        );
    }

    // §5.11.15 `Max_Tx_Depth` table — Block4x4 is depth 0 (no symbol),
    // 8×8 family is 1, 16×16 family is 2, 32×32 family is 3, 64×64 and
    // bigger are 4.
    #[test]
    fn max_tx_depth_matches_spec_table() {
        assert_eq!(BlockSize::Block4x4.max_tx_depth(), 0);
        assert_eq!(BlockSize::Block8x8.max_tx_depth(), 1);
        assert_eq!(BlockSize::Block4x8.max_tx_depth(), 1);
        assert_eq!(BlockSize::Block16x16.max_tx_depth(), 2);
        assert_eq!(BlockSize::Block16x4.max_tx_depth(), 2);
        assert_eq!(BlockSize::Block32x32.max_tx_depth(), 3);
        assert_eq!(BlockSize::Block64x64.max_tx_depth(), 4);
        assert_eq!(BlockSize::Block128x128.max_tx_depth(), 4);
    }

    // §5.11.38 `Subsampled_Size` — narrow blocks must collapse to a
    // valid AV1 chroma TX size under 4:2:0, not the naive 8×2 / 2×8
    // halving. Exercise the spec's BLOCK_INVALID entries too so future
    // refactors can't silently turn them into garbage TX dispatches.
    #[test]
    fn subsampled_size_matches_spec_table() {
        // 4:0:0 (sub_x=0, sub_y=0) is always identity.
        assert_eq!(
            BlockSize::Block16x4.subsampled_size(0, 0),
            BlockSize::Block16x4
        );
        // 4:2:0 (sub_x=1, sub_y=1) — narrow forms collapse, not halve.
        assert_eq!(
            BlockSize::Block16x4.subsampled_size(1, 1),
            BlockSize::Block8x4
        );
        assert_eq!(
            BlockSize::Block4x16.subsampled_size(1, 1),
            BlockSize::Block4x8
        );
        // 4:2:2 (sub_x=1, sub_y=0) — pure horizontal halve.
        assert_eq!(
            BlockSize::Block16x16.subsampled_size(1, 0),
            BlockSize::Block8x16
        );
        // Several BLOCK_INVALID entries; encoder must avoid them.
        assert_eq!(
            BlockSize::Block16x4.subsampled_size(0, 1),
            BlockSize::Invalid
        );
        assert_eq!(
            BlockSize::Block4x16.subsampled_size(1, 0),
            BlockSize::Invalid
        );
        assert_eq!(
            BlockSize::Block8x4.subsampled_size(0, 1),
            BlockSize::Invalid
        );
        // Square block under 4:2:0 simply halves both dims.
        assert_eq!(
            BlockSize::Block64x64.subsampled_size(1, 1),
            BlockSize::Block32x32
        );
        // 128 chroma still exists per spec (no cap until TX dispatch).
        assert_eq!(
            BlockSize::Block128x128.subsampled_size(0, 0),
            BlockSize::Block128x128
        );
        // Invalid in → Invalid out.
        assert_eq!(BlockSize::Invalid.subsampled_size(1, 1), BlockSize::Invalid);
    }

    #[test]
    fn partition_from_u32_exhaustive() {
        for v in 0u32..=9 {
            assert!(PartitionType::from_u32(v).is_some());
        }
        assert!(PartitionType::from_u32(10).is_none());
    }
}
