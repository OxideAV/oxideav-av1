//! AV1 partition-tree walker — §5.11.4 `decode_partition`.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/partition.go`
//! (MIT, KarpelesLab/goavif).
//!
//! [`walk_partition`] is a pure geometry helper: it implements the
//! recursion shape of §5.11.4 but does not read any bits from the
//! bitstream. The caller supplies an `oracle` that picks the partition
//! type at each square node. Real bitstream decoding happens in
//! [`super::superblock`] which reads partition symbols from the range
//! coder and then dispatches over the same shapes.

use super::block::{half_below_size, quarter_size, BlockSize, PartitionType, SubBlock};

/// Callback that returns the partition decision for a square block at
/// `(x, y)` of size `bs`. Must return `PartitionType::None` once a
/// further split would drop below 4×4 (smallest allowed block).
pub type PartitionOracle<'a> = &'a mut dyn FnMut(u32, u32, BlockSize) -> PartitionType;

/// Recursively walk the partition tree rooted at `(x, y)` with initial
/// block size `bs`, calling `oracle` at each square node and `emit` on
/// each leaf block. Implements the geometry of spec §5.11.4 but does
/// NOT consume any bits.
pub fn walk_partition(
    x: u32,
    y: u32,
    bs: BlockSize,
    oracle: PartitionOracle<'_>,
    emit: &mut dyn FnMut(SubBlock),
) {
    walk_inner(x, y, bs, oracle, emit);
}

fn walk_inner(
    x: u32,
    y: u32,
    bs: BlockSize,
    oracle: PartitionOracle<'_>,
    emit: &mut dyn FnMut(SubBlock),
) {
    if bs == BlockSize::Block4x4 {
        // Smallest possible block — no further splitting.
        emit(SubBlock { x, y, size: bs });
        return;
    }

    let pt = if bs.is_square() {
        oracle(x, y, bs)
    } else {
        PartitionType::None
    };
    let w = bs.width();
    let h = bs.height();
    let hw = w / 2;
    let hh = h / 2;

    match pt {
        PartitionType::None => emit(SubBlock { x, y, size: bs }),
        PartitionType::Horz => {
            let top = half_below_size(bs, true);
            let bot = half_below_size(bs, true);
            emit(SubBlock { x, y, size: top });
            emit(SubBlock {
                x,
                y: y + hh,
                size: bot,
            });
        }
        PartitionType::Vert => {
            let left = half_below_size(bs, false);
            let right = half_below_size(bs, false);
            emit(SubBlock { x, y, size: left });
            emit(SubBlock {
                x: x + hw,
                y,
                size: right,
            });
        }
        PartitionType::Split => {
            let sub = quarter_size(bs);
            walk_inner(x, y, sub, oracle, emit);
            walk_inner(x + hw, y, sub, oracle, emit);
            walk_inner(x, y + hh, sub, oracle, emit);
            walk_inner(x + hw, y + hh, sub, oracle, emit);
        }
        // PARTITION_HORZ_{A,B} / VERT_{A,B} / HORZ_4 / VERT_4 — not
        // expanded by this geometry helper. The real bitstream decoder
        // in [`super::superblock`] handles every shape; this walker is
        // mainly for tests + diagnostics, so we treat all non-basic
        // partitions as a single leaf block.
        _ => emit(SubBlock { x, y, size: bs }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_partition_none_emits_single_leaf() {
        let mut oracle = |_x: u32, _y: u32, _bs: BlockSize| PartitionType::None;
        let mut got: Vec<SubBlock> = Vec::new();
        walk_partition(0, 0, BlockSize::Block64x64, &mut oracle, &mut |b| {
            got.push(b)
        });
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].size, BlockSize::Block64x64);
    }

    #[test]
    fn walk_partition_split_once() {
        let mut oracle = |_x: u32, _y: u32, bs: BlockSize| {
            if bs == BlockSize::Block64x64 {
                PartitionType::Split
            } else {
                PartitionType::None
            }
        };
        let mut got: Vec<SubBlock> = Vec::new();
        walk_partition(0, 0, BlockSize::Block64x64, &mut oracle, &mut |b| {
            got.push(b)
        });
        assert_eq!(got.len(), 4);
        let want = [(0u32, 0u32), (32, 0), (0, 32), (32, 32)];
        for (i, (wx, wy)) in want.iter().enumerate() {
            assert_eq!(got[i].x, *wx);
            assert_eq!(got[i].y, *wy);
            assert_eq!(got[i].size, BlockSize::Block32x32);
        }
    }

    #[test]
    fn walk_partition_full_split_to_min() {
        let mut oracle = |_x: u32, _y: u32, _bs: BlockSize| PartitionType::Split;
        let mut got: Vec<SubBlock> = Vec::new();
        walk_partition(0, 0, BlockSize::Block16x16, &mut oracle, &mut |b| {
            got.push(b)
        });
        assert_eq!(got.len(), 16);
        for b in &got {
            assert_eq!(b.size, BlockSize::Block4x4);
        }
    }

    #[test]
    fn walk_partition_horz() {
        let mut oracle = |_x: u32, _y: u32, bs: BlockSize| {
            if bs == BlockSize::Block32x32 {
                PartitionType::Horz
            } else {
                PartitionType::None
            }
        };
        let mut got: Vec<SubBlock> = Vec::new();
        walk_partition(0, 0, BlockSize::Block32x32, &mut oracle, &mut |b| {
            got.push(b)
        });
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].size, BlockSize::Block32x16);
        assert_eq!(got[1].size, BlockSize::Block32x16);
        assert_eq!(got[0].y, 0);
        assert_eq!(got[1].y, 16);
    }
}
