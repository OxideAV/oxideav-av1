//! AV1 block-size + partition-type tables — §3 `BLOCK_*` / `PARTITION_*`.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/block.go` +
//! `partition.go` (MIT, KarpelesLab/goavif). The goavif port is shaped
//! around Go's pixel-reconstruction helper; for oxideav's Phase 2 (mode
//! decode only, no coefficient decode, no pixel reconstruction) we keep
//! the same block-size / partition-type taxonomy and drop the
//! reconstruction-specific types.

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

    /// Largest transform that fits in this block, capped at 64×64 (AV1
    /// hard ceiling). Used by the implicit-TX path.
    pub fn max_tx_size(self) -> (u32, u32) {
        (self.width().min(64), self.height().min(64))
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

    #[test]
    fn partition_from_u32_exhaustive() {
        for v in 0u32..=9 {
            assert!(PartitionType::from_u32(v).is_some());
        }
        assert!(PartitionType::from_u32(10).is_none());
    }
}
