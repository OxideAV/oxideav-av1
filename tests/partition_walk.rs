//! Oracle-driven partition-tree walker tests — ported from
//! `github.com/KarpelesLab/goavif/av1/decoder/partition_test.go`.

use oxideav_av1::decode::{
    partition::walk_partition,
    {BlockSize, PartitionType, SubBlock},
};

#[test]
fn walk_partition_none() {
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
    // Split once at 64×64, NONE at 32×32.
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
    assert_eq!(got.len(), 4, "want 4 leaves, got {}", got.len());
    let want = [(0u32, 0u32), (32, 0), (0, 32), (32, 32)];
    for (i, (wx, wy)) in want.iter().enumerate() {
        assert_eq!(got[i].x, *wx);
        assert_eq!(got[i].y, *wy);
        assert_eq!(got[i].size, BlockSize::Block32x32);
    }
}

#[test]
fn walk_partition_full_split_to_min() {
    // Always split — walks down to 4×4.
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
