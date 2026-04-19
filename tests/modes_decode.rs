//! Intra-mode / block-size / TX-size helpers — ported from
//! `github.com/KarpelesLab/goavif/av1/decoder/modes_test.go`.

use oxideav_av1::decode::{BlockSize, IntraMode};

#[test]
fn intra_mode_is_directional() {
    let cases: &[(IntraMode, bool)] = &[
        (IntraMode::DcPred, false),
        (IntraMode::VPred, false),
        (IntraMode::HPred, false),
        (IntraMode::D45Pred, true),
        (IntraMode::D135Pred, true),
        (IntraMode::D113Pred, true),
        (IntraMode::D157Pred, true),
        (IntraMode::D203Pred, true),
        (IntraMode::D67Pred, true),
        (IntraMode::SmoothPred, false),
        (IntraMode::SmoothVPred, false),
        (IntraMode::SmoothHPred, false),
        (IntraMode::PaethPred, false),
    ];
    for (m, want) in cases {
        assert_eq!(m.is_directional(), *want, "{}", m.name());
    }
}

#[test]
fn mi_width_height_table() {
    let cases: &[(BlockSize, u32, u32)] = &[
        (BlockSize::Block4x4, 1, 1),
        (BlockSize::Block8x8, 2, 2),
        (BlockSize::Block16x16, 4, 4),
        (BlockSize::Block64x64, 16, 16),
        (BlockSize::Block16x32, 4, 8),
        (BlockSize::Block128x128, 32, 32),
    ];
    for (bs, mw, mh) in cases {
        assert_eq!(
            bs.mi_width(),
            *mw,
            "{:?} mi_width expected {}, got {}",
            bs,
            mw,
            bs.mi_width()
        );
        assert_eq!(
            bs.mi_height(),
            *mh,
            "{:?} mi_height expected {}, got {}",
            bs,
            mh,
            bs.mi_height()
        );
    }
}

#[test]
fn subsampled_mi_dims_clamp() {
    // 4×4 block in 4:2:0 chroma yields 1×1 MI (clamped from 0×0).
    let (mw, mh) = BlockSize::Block4x4.subsampled_mi_dims(1, 1);
    assert_eq!((mw, mh), (1, 1));
}

#[test]
fn max_tx_size_caps_at_64() {
    let (w, h) = BlockSize::Block128x128.max_tx_size();
    assert_eq!((w, h), (64, 64));
    let (w, h) = BlockSize::Block16x32.max_tx_size();
    assert_eq!((w, h), (16, 32));
}
