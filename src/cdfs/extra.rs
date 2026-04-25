//! Hand-copied CDFs for syntax elements the `gen_cdfs` generator does
//! not yet emit — mirrors the `lr.rs` pattern. Every table below is
//! transcribed from the AV1 spec (version 1.0.0 errata 1, "Additional
//! tables" section, §9.4):
//!
//! - §5.11.7 / §5.11.12 / §5.11.13 delta-q / delta-lf flags
//! - §5.11.7 `use_intrabc` flag
//! - §5.11.24 `use_filter_intra` + `filter_intra_mode`
//! - §5.11.46 `has_palette_y` + `has_palette_uv`
//! - §5.11.17 `txfm_split` (var-tx partition flag)
//!
//! Wire format matches the rest of [`crate::cdfs`]: for an `N`-symbol
//! CDF the array is `[p_gt_0, p_gt_1, …, p_gt_{N-2}, 0_sentinel,
//! 0_counter]` where each `p_gt_i` is the Q15 probability that the
//! symbol exceeds `i` (scaled by 32768, monotonically decreasing).

/// `delta_q_abs` CDF — 4 symbols (`0..=DELTA_Q_SMALL`, with
/// `DELTA_Q_SMALL == 3` in the spec constants table). Spec
/// §9.4 `Default_Delta_Q_Cdf[DELTA_Q_SMALL + 2] =
/// {28160, 32120, 32677, 32768, 0}`. Stored values are
/// `32768 - cdf_spec[i]` (survival function) per the wire convention
/// in [`crate::cdfs`].
pub const DEFAULT_DELTA_Q_CDF: [u16; 5] = [4608, 648, 91, 0, 0];

/// `delta_lf_abs` CDF — 4 symbols, same shape / same default values
/// as the delta-q CDF. Spec §9.4 `Default_Delta_Lf_Cdf[DELTA_LF_SMALL
/// + 2] = {28160, 32120, 32677, 32768, 0}`. Stored as survival values.
pub const DEFAULT_DELTA_LF_CDF: [u16; 5] = [4608, 648, 91, 0, 0];

/// `use_intrabc` CDF — 2-symbol flag. Spec §9.4
/// `Default_Intrabc_Cdf[2 + 1] = {30531, 32768, 0}`. Stored as
/// `32768 - 30531 = 2237` per the wire convention.
pub const DEFAULT_INTRABC_CDF: [u16; 3] = [2237, 0, 0];

/// `use_filter_intra` CDF — 2-symbol flag, indexed by `MiSize` (22
/// block-size entries). Spec §9.4
/// `Default_Filter_Intra_Cdf[BLOCK_SIZES][3]`. Only the 12 eligible
/// block sizes in §5.11.24 are ever consulted (`DC_PRED` on blocks
/// ≤ 32×32); the remaining entries are the spec's "never used" pad
/// rows (index 10..=15 in the outer dim, plus 20..=21 — see
/// spec note).
pub static DEFAULT_USE_FILTER_INTRA_CDF: [&[u16]; 22] = [
    &[28147, 0, 0], // BLOCK_4X4   (32768 - 4621)
    &[26025, 0, 0], // BLOCK_4X8   (32768 - 6743)
    &[26875, 0, 0], // BLOCK_8X4   (32768 - 5893)
    &[24902, 0, 0], // BLOCK_8X8   (32768 - 7866)
    &[20217, 0, 0], // BLOCK_8X16  (32768 - 12551)
    &[23374, 0, 0], // BLOCK_16X8  (32768 - 9394)
    &[20360, 0, 0], // BLOCK_16X16 (32768 - 12408)
    &[18467, 0, 0], // BLOCK_16X32 (32768 - 14301)
    &[20012, 0, 0], // BLOCK_32X16 (32768 - 12756)
    &[10425, 0, 0], // BLOCK_32X32 (32768 - 22343)
    &[16384, 0, 0], // BLOCK_32X64  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X32  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X64  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X128 (never used — pad)
    &[16384, 0, 0], // BLOCK_128X64 (never used — pad)
    &[16384, 0, 0], // BLOCK_128X128(never used — pad)
    &[19998, 0, 0], // BLOCK_4X16  (32768 - 12770)
    &[22400, 0, 0], // BLOCK_16X4  (32768 - 10368)
    &[12539, 0, 0], // BLOCK_8X32  (32768 - 20229)
    &[14667, 0, 0], // BLOCK_32X8  (32768 - 18101)
    &[16384, 0, 0], // BLOCK_16X64  (never used — pad)
    &[16384, 0, 0], // BLOCK_64X16  (never used — pad)
];

/// `filter_intra_mode` CDF — 5 symbols (`FILTER_DC_PRED`,
/// `FILTER_V_PRED`, `FILTER_H_PRED`, `FILTER_D157_PRED`,
/// `FILTER_PAETH_PRED`; spec §6.10.23). Spec §9.4
/// `Default_Filter_Intra_Mode_Cdf[6] =
/// {8949, 12776, 17211, 29558, 32768, 0}` is in cumulative form.
/// Stored as `32768 - cdf_spec[i]` survival values so the wire
/// convention in [`crate::cdfs`] holds.
pub const DEFAULT_FILTER_INTRA_MODE_CDF: [u16; 6] = [23819, 19992, 15557, 3210, 0, 0];

/// `has_palette_y` CDF — §5.11.46 / §9.4
/// `Default_Palette_Y_Mode_Cdf[PALETTE_BLOCK_SIZE_CONTEXTS=7][
/// PALETTE_Y_MODE_CONTEXTS=3][3]`. Indexed as
/// `[bsizeCtx][ctx]` where `bsizeCtx = Mi_Width_Log2[MiSize] +
/// Mi_Height_Log2[MiSize] - 2` runs 0..=6 for the eligible
/// `BLOCK_8X8..BLOCK_64X64` range, and `ctx = (AvailU &&
/// PaletteSizes[0][row-1][col] > 0) + (AvailL &&
/// PaletteSizes[0][row][col-1] > 0)` is 0..=2. Each entry is a
/// 2-symbol `{p_gt_0, 0_sentinel, 0_counter}` tuple. Stored values
/// are `32768 - cdf_spec[0]` (the survival function) so the
/// `decode_symbol` hot loop can index without a subtraction.
pub static DEFAULT_PALETTE_Y_MODE_CDF: [[[u16; 3]; 3]; 7] = [
    [[1092, 0, 0], [29349, 0, 0], [31507, 0, 0]],
    [[856, 0, 0], [29909, 0, 0], [31788, 0, 0]],
    [[945, 0, 0], [29368, 0, 0], [31987, 0, 0]],
    [[738, 0, 0], [29207, 0, 0], [31864, 0, 0]],
    [[459, 0, 0], [25431, 0, 0], [31306, 0, 0]],
    [[503, 0, 0], [28753, 0, 0], [31247, 0, 0]],
    [[318, 0, 0], [24822, 0, 0], [32639, 0, 0]],
];

/// `has_palette_uv` CDF — §5.11.46 / §9.4
/// `Default_Palette_Uv_Mode_Cdf[PALETTE_UV_MODE_CONTEXTS=2][3]`.
/// Indexed as `[ctx]` where `ctx = (PaletteSizeY > 0) ? 1 : 0`.
/// 2-symbol flag per entry; same inversion convention.
pub static DEFAULT_PALETTE_UV_MODE_CDF: [[u16; 3]; 2] = [[307, 0, 0], [11280, 0, 0]];

/// `palette_size_y_minus_2` CDF — §5.11.46 / §9.4
/// `Default_Palette_Y_Size_Cdf[PALETTE_BLOCK_SIZE_CONTEXTS=7][
/// PALETTE_SIZES+1=8]`. `bsizeCtx` indexes the outer dim; the inner
/// is a 7-symbol distribution (palette sizes 2..=8 → values 0..=6).
/// Wire format: `[p_gt_0, p_gt_1, ..., p_gt_5, 0_sentinel,
/// 0_counter]` where each `p_gt_i = 32768 - cdf_spec[i]` is the Q15
/// survival function (decreasing). `decode_symbol` interprets
/// `cdf.len()-1` as the symbol count, so for 7 symbols we keep 8
/// entries total (6 inverted values + sentinel + counter).
pub static DEFAULT_PALETTE_Y_SIZE_CDF: [[u16; 8]; 7] = [
    [24816, 19768, 14619, 11290, 7241, 3527, 0, 0],
    [25629, 21347, 16573, 13224, 9102, 4695, 0, 0],
    [24980, 20027, 15443, 12268, 8453, 4238, 0, 0],
    [24497, 18704, 14522, 11204, 7697, 4235, 0, 0],
    [20043, 13588, 10905, 7929, 5233, 2648, 0, 0],
    [23057, 17880, 15845, 11716, 7107, 4893, 0, 0],
    [17828, 11971, 11090, 8582, 5735, 3769, 0, 0],
];

/// `palette_size_uv_minus_2` CDF — §5.11.46 / §9.4
/// `Default_Palette_Uv_Size_Cdf[PALETTE_BLOCK_SIZE_CONTEXTS=7][
/// PALETTE_SIZES+1=8]`. Same shape as Y; same inversion convention.
pub static DEFAULT_PALETTE_UV_SIZE_CDF: [[u16; 8]; 7] = [
    [24055, 12789, 5640, 3159, 1437, 496, 0, 0],
    [26929, 17195, 9187, 5821, 2920, 1068, 0, 0],
    [28342, 21508, 14769, 11285, 6905, 3338, 0, 0],
    [29540, 23304, 17775, 14679, 10245, 5348, 0, 0],
    [29000, 23882, 19677, 14916, 10273, 5561, 0, 0],
    [30304, 24317, 19907, 11136, 7243, 4213, 0, 0],
    [31499, 27333, 22335, 13805, 11068, 6903, 0, 0],
];

// Palette `palette_color_idx_*` CDFs — §5.11.49 / §9.4. One CDF table
// per `(plane, palette_size)` combo, indexed by `PALETTE_COLOR_CONTEXTS
// = 5`. The decoded symbol value is the rank of the palette colour
// within `ColorOrder[]` (most-likely first), so the inner alphabet is
// `palette_size`-wide. Wire format: `[p_gt_0, ..., p_gt_{N-2},
// 0_sentinel, 0_counter]` where each `p_gt_i = 32768 - cdf_spec[i]`
// is the Q15 survival function. `decode_symbol` computes
// `n = cdf.len() - 1`, so a 2-symbol CDF needs 3 entries total.
pub static DEFAULT_PALETTE_Y_COLOR_SIZE_2_CDF: [[u16; 3]; 5] = [
    [4058, 0, 0],
    [16384, 0, 0],
    [22215, 0, 0],
    [5732, 0, 0],
    [1165, 0, 0],
];

pub static DEFAULT_PALETTE_Y_COLOR_SIZE_3_CDF: [[u16; 4]; 5] = [
    [4891, 2278, 0, 0],
    [21236, 7071, 0, 0],
    [26224, 2534, 0, 0],
    [9750, 4696, 0, 0],
    [853, 383, 0, 0],
];

pub static DEFAULT_PALETTE_Y_COLOR_SIZE_4_CDF: [[u16; 5]; 5] = [
    [7196, 4722, 2723, 0, 0],
    [23290, 11178, 5512, 0, 0],
    [25520, 5931, 2944, 0, 0],
    [13601, 8282, 4419, 0, 0],
    [1368, 943, 518, 0, 0],
];

pub static DEFAULT_PALETTE_Y_COLOR_SIZE_5_CDF: [[u16; 6]; 5] = [
    [7989, 5813, 4192, 2486, 0, 0],
    [24099, 12404, 8695, 4675, 0, 0],
    [28513, 5203, 3391, 1701, 0, 0],
    [12904, 9094, 6052, 3238, 0, 0],
    [1122, 875, 621, 342, 0, 0],
];

pub static DEFAULT_PALETTE_Y_COLOR_SIZE_6_CDF: [[u16; 7]; 5] = [
    [9636, 7361, 5798, 4333, 2695, 0, 0],
    [25325, 15526, 12051, 8006, 4786, 0, 0],
    [26468, 7906, 5824, 3984, 2097, 0, 0],
    [13852, 9873, 7501, 5333, 3116, 0, 0],
    [1498, 1218, 960, 709, 415, 0, 0],
];

pub static DEFAULT_PALETTE_Y_COLOR_SIZE_7_CDF: [[u16; 8]; 5] = [
    [9663, 7569, 6304, 5084, 3837, 2450, 0, 0],
    [25818, 17321, 13816, 10087, 7201, 4205, 0, 0],
    [25208, 9294, 7278, 5565, 3847, 2060, 0, 0],
    [14224, 10395, 8311, 6573, 4649, 2723, 0, 0],
    [1570, 1317, 1098, 886, 645, 377, 0, 0],
];

pub static DEFAULT_PALETTE_Y_COLOR_SIZE_8_CDF: [[u16; 9]; 5] = [
    [11079, 8885, 7605, 6416, 5262, 3941, 2573, 0, 0],
    [25876, 17383, 14928, 11162, 8481, 6015, 3564, 0, 0],
    [27117, 9586, 7726, 6250, 4786, 3376, 1868, 0, 0],
    [13419, 10190, 8350, 6774, 5244, 3737, 2320, 0, 0],
    [1740, 1498, 1264, 1063, 841, 615, 376, 0, 0],
];

pub static DEFAULT_PALETTE_UV_COLOR_SIZE_2_CDF: [[u16; 3]; 5] = [
    [3679, 0, 0],
    [16384, 0, 0],
    [24055, 0, 0],
    [3511, 0, 0],
    [1158, 0, 0],
];

pub static DEFAULT_PALETTE_UV_COLOR_SIZE_3_CDF: [[u16; 4]; 5] = [
    [7511, 3623, 0, 0],
    [20481, 5475, 0, 0],
    [25735, 4808, 0, 0],
    [12623, 7363, 0, 0],
    [2160, 1129, 0, 0],
];

pub static DEFAULT_PALETTE_UV_COLOR_SIZE_4_CDF: [[u16; 5]; 5] = [
    [8558, 5593, 2865, 0, 0],
    [22880, 10382, 5554, 0, 0],
    [26867, 6715, 3475, 0, 0],
    [14450, 10616, 4435, 0, 0],
    [2309, 1632, 842, 0, 0],
];

pub static DEFAULT_PALETTE_UV_COLOR_SIZE_5_CDF: [[u16; 6]; 5] = [
    [9788, 7289, 4987, 2782, 0, 0],
    [24355, 11360, 7909, 3894, 0, 0],
    [30511, 3319, 2174, 1170, 0, 0],
    [13579, 11566, 6853, 4148, 0, 0],
    [924, 724, 487, 250, 0, 0],
];

pub static DEFAULT_PALETTE_UV_COLOR_SIZE_6_CDF: [[u16; 7]; 5] = [
    [10551, 8201, 6131, 4085, 2220, 0, 0],
    [25461, 16362, 13132, 8136, 4344, 0, 0],
    [28327, 7704, 5889, 3826, 1849, 0, 0],
    [15558, 12240, 9449, 6018, 3186, 0, 0],
    [2094, 1815, 1372, 1033, 561, 0, 0],
];

pub static DEFAULT_PALETTE_UV_COLOR_SIZE_7_CDF: [[u16; 8]; 5] = [
    [11529, 9600, 7724, 5806, 4063, 2262, 0, 0],
    [26223, 17756, 14764, 10951, 7265, 4067, 0, 0],
    [29320, 6473, 5331, 4064, 2642, 1326, 0, 0],
    [16879, 14445, 11064, 8070, 5792, 3078, 0, 0],
    [1780, 1564, 1289, 1034, 785, 443, 0, 0],
];

pub static DEFAULT_PALETTE_UV_COLOR_SIZE_8_CDF: [[u16; 9]; 5] = [
    [11326, 9480, 8010, 6522, 5119, 3788, 2205, 0, 0],
    [26905, 17835, 15216, 12100, 9085, 6357, 3495, 0, 0],
    [29353, 6958, 5891, 4778, 3545, 2374, 1150, 0, 0],
    [14803, 12684, 10536, 8794, 6494, 4366, 2378, 0, 0],
    [1578, 1439, 1252, 1089, 943, 742, 446, 0, 0],
];

/// Spec §9.3 / additional tables `Palette_Color_Context[
/// PALETTE_MAX_COLOR_CONTEXT_HASH + 1 ]`. Maps a `ColorContextHash`
/// value (0..=8 — the multiplier sum from `get_palette_color_context`)
/// onto the palette-colour-index CDF context (0..=4). Negative entries
/// are spec-marked "never accessed"; we replace them with 0 so a
/// rare misindex degrades gracefully instead of panicking.
pub static PALETTE_COLOR_CONTEXT: [u8; 9] = [0, 0, 0, 0, 0, 4, 3, 2, 1];

/// Spec §9.3 / additional tables `Palette_Color_Hash_Multipliers[
/// PALETTE_NUM_NEIGHBORS = 3 ]`. Used by `get_palette_color_context`
/// to fold the per-neighbour scores into a single hash.
pub const PALETTE_COLOR_HASH_MULTIPLIERS: [u32; 3] = [1, 2, 2];

/// `txfm_split` CDF — §5.11.17 / §9.4
/// `Default_Txfm_Split_Cdf[TXFM_PARTITION_CONTEXTS=21][3]`. 2-symbol
/// flag per entry, indexed by the §9.4 `ctx` formula:
/// `ctx = (txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 +
/// above + left`. Spec values `{x, 32768, 0}` are cumulative; stored
/// as `32768 - x` (survival) per the wire convention.
pub static DEFAULT_TXFM_SPLIT_CDF: [[u16; 3]; 21] = [
    [4187, 0, 0],  // 32768 - 28581
    [8922, 0, 0],  // 32768 - 23846
    [11921, 0, 0], // 32768 - 20847
    [8453, 0, 0],  // 32768 - 24315
    [14572, 0, 0], // 32768 - 18196
    [20635, 0, 0], // 32768 - 12133
    [13977, 0, 0], // 32768 - 18791
    [21881, 0, 0], // 32768 - 10887
    [21763, 0, 0], // 32768 - 11005
    [5589, 0, 0],  // 32768 - 27179
    [12764, 0, 0], // 32768 - 20004
    [21487, 0, 0], // 32768 - 11281
    [6219, 0, 0],  // 32768 - 26549
    [13460, 0, 0], // 32768 - 19308
    [18544, 0, 0], // 32768 - 14224
    [4753, 0, 0],  // 32768 - 28015
    [11222, 0, 0], // 32768 - 21546
    [18368, 0, 0], // 32768 - 14400
    [4603, 0, 0],  // 32768 - 28165
    [10367, 0, 0], // 32768 - 22401
    [16680, 0, 0], // 32768 - 16088
];
