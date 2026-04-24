//! Hand-copied CDFs for syntax elements the `gen_cdfs` generator does
//! not yet emit — mirrors the `lr.rs` pattern. Every table below is
//! transcribed from the AV1 spec (version 1.0.0 errata 1, "Additional
//! tables" section, §9.4):
//!
//! - §5.11.7 / §5.11.12 / §5.11.13 delta-q / delta-lf flags
//! - §5.11.7 `use_intrabc` flag
//! - §5.11.24 `use_filter_intra` + `filter_intra_mode`
//!
//! Wire format matches the rest of [`crate::cdfs`]: for an `N`-symbol
//! CDF the array is `[p_gt_0, p_gt_1, …, p_gt_{N-2}, 0_sentinel,
//! 0_counter]` where each `p_gt_i` is the Q15 probability that the
//! symbol exceeds `i` (scaled by 32768, monotonically decreasing).

/// `delta_q_abs` CDF — 4 symbols (`0..=DELTA_Q_SMALL`, with
/// `DELTA_Q_SMALL == 3` in the spec constants table). Spec
/// §9.4 `Default_Delta_Q_Cdf[DELTA_Q_SMALL + 2]`.
pub const DEFAULT_DELTA_Q_CDF: [u16; 5] = [28160, 32120, 32677, 0, 0];

/// `delta_lf_abs` CDF — 4 symbols, same shape / same default values
/// as the delta-q CDF. Spec §9.4 `Default_Delta_Lf_Cdf[DELTA_LF_SMALL
/// + 2]`.
pub const DEFAULT_DELTA_LF_CDF: [u16; 5] = [28160, 32120, 32677, 0, 0];

/// `use_intrabc` CDF — 2-symbol flag. Spec §9.4
/// `Default_Intrabc_Cdf[2 + 1]`.
pub const DEFAULT_INTRABC_CDF: [u16; 3] = [30531, 0, 0];

/// `use_filter_intra` CDF — 2-symbol flag, indexed by `MiSize` (22
/// block-size entries). Spec §9.4
/// `Default_Filter_Intra_Cdf[BLOCK_SIZES][3]`. Only the 12 eligible
/// block sizes in §5.11.24 are ever consulted (`DC_PRED` on blocks
/// ≤ 32×32); the remaining entries are the spec's "never used" pad
/// rows (index 10..=15 in the outer dim, plus 20..=21 — see
/// spec note).
pub static DEFAULT_USE_FILTER_INTRA_CDF: [&[u16]; 22] = [
    &[4621, 0, 0],  // BLOCK_4X4
    &[6743, 0, 0],  // BLOCK_4X8
    &[5893, 0, 0],  // BLOCK_8X4
    &[7866, 0, 0],  // BLOCK_8X8
    &[12551, 0, 0], // BLOCK_8X16
    &[9394, 0, 0],  // BLOCK_16X8
    &[12408, 0, 0], // BLOCK_16X16
    &[14301, 0, 0], // BLOCK_16X32
    &[12756, 0, 0], // BLOCK_32X16
    &[22343, 0, 0], // BLOCK_32X32
    &[16384, 0, 0], // BLOCK_32X64  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X32  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X64  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X128 (never used — pad)
    &[16384, 0, 0], // BLOCK_128X64 (never used — pad)
    &[16384, 0, 0], // BLOCK_128X128(never used — pad)
    &[12770, 0, 0], // BLOCK_4X16
    &[10368, 0, 0], // BLOCK_16X4
    &[20229, 0, 0], // BLOCK_8X32
    &[18101, 0, 0], // BLOCK_32X8
    &[16384, 0, 0], // BLOCK_16X64  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X16  (never used — pad)
];

/// `filter_intra_mode` CDF — 5 symbols (`FILTER_DC_PRED`,
/// `FILTER_V_PRED`, `FILTER_H_PRED`, `FILTER_D157_PRED`,
/// `FILTER_PAETH_PRED`; spec §6.10.23). Spec §9.4
/// `Default_Filter_Intra_Mode_Cdf[6]` — the 6-element literal is
/// `{p_gt_0, p_gt_1, p_gt_2, p_gt_3, 32768, 0}`; we store the
/// same values but replace the 32768 with our 0 sentinel.
pub const DEFAULT_FILTER_INTRA_MODE_CDF: [u16; 6] = [8949, 12776, 17211, 29558, 0, 0];
