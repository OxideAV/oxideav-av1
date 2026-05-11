//! AV1 coefficient context derivation — §5.11.39 / §6.10 / §9.4.
//!
//! Round 45 (workspace task #776) graduated this module from the
//! round-3 stub (which derived only the per-position 2D-template
//! offset and assumed all neighbour contexts were 0) to the
//! spec-correct neighbour-aware derivation that consults the
//! per-tile `AboveLevelContext` / `LeftLevelContext` /
//! `AboveDcContext` / `LeftDcContext` arrays.
//!
//! Spec coverage:
//!
//! * `txb_skip_ctx_spec` — §5.11.39 / §9.4 `all_zero` ctx.
//! * `eob_pt_ctx` — §5.11.39 / §9.4 `eob_pt_*`.
//! * `get_coeff_base_ctx` (and `coeff_base_eob_from_base`) —
//!   §5.11.39 with the
//!   `Coeff_Base_Ctx_Offset[txSz][min(row,4)][min(col,4)]` table
//!   and the `Sig_Ref_Diff_Offset[txClass]` 5-position template.
//! * `coeff_br_ctx_spec` — §5.11.39 / §9.4 `coeff_br`.
//! * `dc_sign_ctx_spec` — §5.11.39 ±1 polling across the per-plane
//!   `Above/LeftDcContext` arrays.
//!
//! The arrays are owned by [`super::tile::TileDecoder`] and updated
//! by the residual reconstruction path after each TU.
//!
//! The two legacy round-3 helpers ([`sig_coef_ctx_2d`] and
//! [`level_ctx`]) are retained because the encoder's still-WIP
//! coefficient emitter consumes them.

use crate::transform::TxType;

/// Spec `TX_CLASS` (§5.11.39 / §6.10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxClass {
    /// All transforms whose row and column kernels are both real
    /// (DCT/ADST/IDTX/WHT). Includes DCT_DCT and IDTX_IDTX.
    Class2D = 0,
    /// `H_*` family — horizontal-only.
    ClassHoriz = 1,
    /// `V_*` family — vertical-only.
    ClassVert = 2,
}

impl TxClass {
    /// Spec `get_tx_class()` (§5.11.39).
    pub fn from_tx_type(t: TxType) -> Self {
        match t {
            TxType::VDct | TxType::VAdst | TxType::VFlipAdst => Self::ClassVert,
            TxType::HDct | TxType::HAdst | TxType::HFlipAdst => Self::ClassHoriz,
            _ => Self::Class2D,
        }
    }
}

/// Spec constant `SIG_COEF_CONTEXTS_2D = 26`.
pub const SIG_COEF_CONTEXTS_2D: i32 = 26;
/// Spec constant `SIG_COEF_CONTEXTS = 42`.
pub const SIG_COEF_CONTEXTS: i32 = 42;
/// Spec constant `SIG_COEF_CONTEXTS_EOB = 4`.
pub const SIG_COEF_CONTEXTS_EOB: i32 = 4;

/// Spec constant `NUM_BASE_LEVELS = 2`.
pub const NUM_BASE_LEVELS: i32 = 2;
/// Spec constant `COEFF_BASE_RANGE = 12`.
pub const COEFF_BASE_RANGE: i32 = 12;

/// `Coeff_Base_Ctx_Offset[txSzCtx][min(row,4)][min(col,4)]` for the
/// 5 square buckets the coefficient decoder routes through
/// (`tx_size_idx ∈ 0..=4`).
const COEFF_BASE_CTX_OFFSET_SQ: [[[i32; 5]; 5]; 5] = [
    // TX_4X4: edge entries are zero.
    [
        [0, 1, 6, 6, 0],
        [1, 6, 6, 21, 0],
        [6, 6, 21, 21, 0],
        [6, 21, 21, 21, 0],
        [0, 0, 0, 0, 0],
    ],
    // TX_8X8.
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_16X16.
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_32X32.
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
    // TX_64X64 — Adjusted_Tx_Size collapses to TX_32X32 here.
    [
        [0, 1, 6, 6, 21],
        [1, 6, 6, 21, 21],
        [6, 6, 21, 21, 21],
        [6, 21, 21, 21, 21],
        [21, 21, 21, 21, 21],
    ],
];

/// Spec `Sig_Ref_Diff_Offset[txClass][idx][0..=1]`.
const SIG_REF_DIFF_OFFSET_SPEC: [[(i32, i32); 5]; 3] = [
    // TX_CLASS_2D.
    [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)],
    // TX_CLASS_HORIZ.
    [(0, 1), (1, 0), (0, 2), (0, 3), (0, 4)],
    // TX_CLASS_VERT.
    [(0, 1), (1, 0), (2, 0), (3, 0), (4, 0)],
];

/// Spec `Coeff_Base_Pos_Ctx_Offset[3]`.
const COEFF_BASE_POS_CTX_OFFSET: [i32; 3] = [
    SIG_COEF_CONTEXTS_2D,
    SIG_COEF_CONTEXTS_2D + 5,
    SIG_COEF_CONTEXTS_2D + 10,
];

/// Spec `Mag_Ref_Offset_With_Tx_Class[txClass][3][2]`.
const MAG_REF_OFFSET_WITH_TX_CLASS: [[(i32, i32); 3]; 3] = [
    // TX_CLASS_2D.
    [(0, 1), (1, 0), (1, 1)],
    // TX_CLASS_HORIZ.
    [(0, 1), (1, 0), (0, 2)],
    // TX_CLASS_VERT.
    [(0, 1), (1, 0), (2, 0)],
];

/// `txb_skip_ctx` per AV1 spec §5.11.39 / §9.4 `all_zero`.
pub fn txb_skip_ctx_spec(
    plane: usize,
    top: i32,
    left: i32,
    bw: i32,
    bh: i32,
    w: i32,
    h: i32,
) -> i32 {
    if plane == 0 {
        let top = top.min(255);
        let left = left.min(255);
        if bw == w && bh == h {
            0
        } else if top == 0 && left == 0 {
            1
        } else if top == 0 || left == 0 {
            2 + (top.max(left) > 3) as i32
        } else if top.max(left) <= 3 {
            4
        } else if top.min(left) <= 3 {
            5
        } else {
            6
        }
    } else {
        let mut ctx = (top != 0) as i32 + (left != 0) as i32;
        ctx += 7;
        if bw * bh > w * h {
            ctx += 3;
        }
        ctx
    }
}

/// `dc_sign_ctx` per AV1 spec §5.11.39 / §9.4 `dc_sign`.
pub fn dc_sign_ctx_spec(above_dc_signs: &[u8], left_dc_signs: &[u8]) -> i32 {
    let mut dc_sign: i32 = 0;
    for &s in above_dc_signs {
        if s == 1 {
            dc_sign -= 1;
        } else if s == 2 {
            dc_sign += 1;
        }
    }
    for &s in left_dc_signs {
        if s == 1 {
            dc_sign -= 1;
        } else if s == 2 {
            dc_sign += 1;
        }
    }
    if dc_sign < 0 {
        1
    } else if dc_sign > 0 {
        2
    } else {
        0
    }
}

/// `eob_pt_ctx` per AV1 spec §5.11.39 / §9.4 `eob_pt_*`.
pub fn eob_pt_ctx(tx_class: TxClass) -> i32 {
    if tx_class == TxClass::Class2D {
        0
    } else {
        1
    }
}

/// `get_coeff_base_ctx` per AV1 spec §5.11.39.
#[allow(clippy::too_many_arguments)]
pub fn get_coeff_base_ctx(
    tx_size_idx: usize,
    tx_class: TxClass,
    bwl: u32,
    height: i32,
    quants: &[i32],
    pos: usize,
    scan_idx: usize,
    is_eob: bool,
) -> i32 {
    let width: i32 = 1 << bwl;
    if is_eob {
        if scan_idx == 0 {
            return SIG_COEF_CONTEXTS - 4;
        }
        let area = (height << bwl) as usize;
        if scan_idx <= area / 8 {
            return SIG_COEF_CONTEXTS - 3;
        }
        if scan_idx <= area / 4 {
            return SIG_COEF_CONTEXTS - 2;
        }
        return SIG_COEF_CONTEXTS - 1;
    }
    let row = (pos as i32) >> bwl;
    let col = (pos as i32) - (row << bwl);
    let mut mag: i32 = 0;
    let class_idx = tx_class as usize;
    for &(dr, dc) in &SIG_REF_DIFF_OFFSET_SPEC[class_idx] {
        let rr = row + dr;
        let cc = col + dc;
        if rr >= 0 && cc >= 0 && rr < height && cc < width {
            let idx = ((rr << bwl) + cc) as usize;
            if let Some(&q) = quants.get(idx) {
                let v = q.unsigned_abs() as i32;
                mag += v.min(3);
            }
        }
    }
    let ctx = ((mag + 1) >> 1).min(4);
    if tx_class == TxClass::Class2D {
        if row == 0 && col == 0 {
            return 0;
        }
        let r_idx = row.min(4) as usize;
        let c_idx = col.min(4) as usize;
        return ctx + COEFF_BASE_CTX_OFFSET_SQ[tx_size_idx.min(4)][r_idx][c_idx];
    }
    let pos_idx = if tx_class == TxClass::ClassVert {
        row
    } else {
        col
    };
    ctx + COEFF_BASE_POS_CTX_OFFSET[pos_idx.clamp(0, 2) as usize]
}

/// Reduce a `coeff_base` ctx to the `coeff_base_eob` ctx range.
pub fn coeff_base_eob_from_base(ctx: i32) -> i32 {
    let r = ctx - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB;
    r.clamp(0, SIG_COEF_CONTEXTS_EOB - 1)
}

/// `coeff_br` ctx per AV1 spec §5.11.39 / §9.4 `coeff_br`.
pub fn coeff_br_ctx_spec(
    tx_class: TxClass,
    bwl: u32,
    height: i32,
    quants: &[i32],
    pos: usize,
) -> i32 {
    let width: i32 = 1 << bwl;
    let row = (pos as i32) >> bwl;
    let col = (pos as i32) - (row << bwl);
    let mut mag: i32 = 0;
    let class_idx = tx_class as usize;
    let cap = COEFF_BASE_RANGE + NUM_BASE_LEVELS + 1;
    for &(dr, dc) in &MAG_REF_OFFSET_WITH_TX_CLASS[class_idx] {
        let rr = row + dr;
        let cc = col + dc;
        if rr >= 0 && cc >= 0 && rr < height && cc < width {
            let idx = ((rr << bwl) + cc) as usize;
            if let Some(&q) = quants.get(idx) {
                let v = q.unsigned_abs() as i32;
                mag += v.min(cap);
            }
        }
    }
    let mag = ((mag + 1) >> 1).min(6);
    if pos == 0 {
        return mag;
    }
    match tx_class {
        TxClass::Class2D => {
            if row < 2 && col < 2 {
                mag + 7
            } else {
                mag + 14
            }
        }
        TxClass::ClassHoriz => {
            if col == 0 {
                mag + 7
            } else {
                mag + 14
            }
        }
        TxClass::ClassVert => {
            if row == 0 {
                mag + 7
            } else {
                mag + 14
            }
        }
    }
}

// ===== Legacy round-3 helpers retained for the encoder + tests =====

/// Five neighbor positions sampled for the round-3 2D scan's
/// coefficient context: `(dr, dc)` offsets from the current `(r, c)`.
const TEMPLATE_2D_OFFSETS: [(i32, i32); 5] = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0)];

/// Legacy round-3 `sig_coef_ctx_2d`. Decoder routes through
/// [`get_coeff_base_ctx`].
pub fn sig_coef_ctx_2d(
    r: i32,
    c: i32,
    w: i32,
    h: i32,
    abs_levels: &[i8],
    nz_map_offset: &[i8],
    scan_idx: usize,
) -> i32 {
    if scan_idx == 0 {
        return 0;
    }
    let mut stats: i32 = 0;
    for (dr, dc) in TEMPLATE_2D_OFFSETS {
        let rr = r + dr;
        let cc = c + dc;
        if rr < h && cc < w {
            let idx = (rr * w + cc) as usize;
            let mut v = abs_levels[idx] as i32;
            if v > 3 {
                v = 3;
            }
            stats += v;
        }
    }
    let mut ctx_base = (stats + 1) >> 1;
    if ctx_base > 4 {
        ctx_base = 4;
    }
    ctx_base + nz_map_offset[scan_idx] as i32
}

/// Legacy round-3 `coeff_br_ctx`. Decoder routes through
/// [`coeff_br_ctx_spec`].
pub fn level_ctx(r: i32, c: i32, w: i32, h: i32, abs_levels: &[i8]) -> i32 {
    let mut stats: i32 = 0;
    for (dr, dc) in [(0i32, 1i32), (1, 0), (1, 1)] {
        let rr = r + dr;
        let cc = c + dc;
        if rr < h && cc < w {
            let idx = (rr * w + cc) as usize;
            let mut v = abs_levels[idx] as i32;
            if v > 3 {
                v -= 3;
                if v > 3 {
                    v = 3;
                }
                stats += v;
            }
        }
    }
    let mut ctx = (stats + 1) >> 1;
    if ctx > 3 {
        ctx = 3;
    }
    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tx_class_default_is_2d() {
        assert_eq!(TxClass::from_tx_type(TxType::DctDct), TxClass::Class2D);
        assert_eq!(TxClass::from_tx_type(TxType::IdtIdt), TxClass::Class2D);
        assert_eq!(TxClass::from_tx_type(TxType::Wht), TxClass::Class2D);
        assert_eq!(TxClass::from_tx_type(TxType::HDct), TxClass::ClassHoriz);
        assert_eq!(TxClass::from_tx_type(TxType::VDct), TxClass::ClassVert);
    }

    #[test]
    fn txb_skip_ctx_plane0_block_eq_tx() {
        assert_eq!(txb_skip_ctx_spec(0, 0, 0, 4, 4, 4, 4), 0);
        assert_eq!(txb_skip_ctx_spec(0, 5, 7, 16, 16, 16, 16), 0);
    }

    #[test]
    fn txb_skip_ctx_plane0_zero_neighbors() {
        assert_eq!(txb_skip_ctx_spec(0, 0, 0, 16, 16, 4, 4), 1);
    }

    #[test]
    fn txb_skip_ctx_chroma_zero_neighbors() {
        assert_eq!(txb_skip_ctx_spec(1, 0, 0, 4, 4, 4, 4), 7);
        assert_eq!(txb_skip_ctx_spec(2, 0, 0, 4, 4, 4, 4), 7);
    }

    #[test]
    fn txb_skip_ctx_chroma_bw_bh_bigger() {
        assert_eq!(txb_skip_ctx_spec(1, 0, 0, 8, 8, 4, 4), 10);
    }

    #[test]
    fn dc_sign_ctx_tallies() {
        assert_eq!(dc_sign_ctx_spec(&[], &[]), 0);
        assert_eq!(dc_sign_ctx_spec(&[1], &[]), 1);
        assert_eq!(dc_sign_ctx_spec(&[2], &[]), 2);
        assert_eq!(dc_sign_ctx_spec(&[1, 2], &[]), 0);
        assert_eq!(dc_sign_ctx_spec(&[2, 2], &[1]), 2);
    }

    #[test]
    fn eob_pt_ctx_two_way() {
        assert_eq!(eob_pt_ctx(TxClass::Class2D), 0);
        assert_eq!(eob_pt_ctx(TxClass::ClassHoriz), 1);
        assert_eq!(eob_pt_ctx(TxClass::ClassVert), 1);
    }

    #[test]
    fn coeff_base_ctx_dc_2d_returns_zero() {
        let q = vec![0i32; 16];
        assert_eq!(
            get_coeff_base_ctx(0, TxClass::Class2D, 2, 4, &q, 0, 0, false),
            0
        );
    }

    #[test]
    fn coeff_base_ctx_eob_dc_returns_38() {
        let q = vec![0i32; 16];
        assert_eq!(
            get_coeff_base_ctx(0, TxClass::Class2D, 2, 4, &q, 0, 0, true),
            SIG_COEF_CONTEXTS - 4
        );
    }

    #[test]
    fn coeff_base_eob_from_base_folds() {
        assert_eq!(coeff_base_eob_from_base(SIG_COEF_CONTEXTS - 4), 0);
        assert_eq!(coeff_base_eob_from_base(SIG_COEF_CONTEXTS - 3), 1);
        assert_eq!(coeff_base_eob_from_base(SIG_COEF_CONTEXTS - 2), 2);
        assert_eq!(coeff_base_eob_from_base(SIG_COEF_CONTEXTS - 1), 3);
    }

    #[test]
    fn coeff_br_ctx_dc_returns_mag() {
        let q = vec![0i32; 16];
        assert_eq!(coeff_br_ctx_spec(TxClass::Class2D, 2, 4, &q, 0), 0);
    }

    #[test]
    fn coeff_br_ctx_2d_upper_left_corner_adds_7() {
        let q = vec![0i32; 16];
        assert_eq!(coeff_br_ctx_spec(TxClass::Class2D, 2, 4, &q, 1), 7);
    }

    // ===== Legacy round-3 tests =====

    #[test]
    fn sig_coef_ctx_dc_is_zero() {
        let levels = [0i8; 16];
        let offset = [0i8, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21];
        assert_eq!(sig_coef_ctx_2d(0, 0, 4, 4, &levels, &offset, 0), 0);
    }

    #[test]
    fn sig_coef_ctx_zero_neighbors_uses_position_offset() {
        let levels = [0i8; 16];
        let offset = [0i8, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21];
        assert_eq!(sig_coef_ctx_2d(0, 1, 4, 4, &levels, &offset, 1), 1);
        assert_eq!(sig_coef_ctx_2d(1, 3, 4, 4, &levels, &offset, 7), 21);
    }

    #[test]
    fn sig_coef_ctx_clamps_stats() {
        let mut levels = [0i8; 16];
        levels[5] = 100;
        levels[1] = 100;
        levels[4] = 100;
        let offset = [0i8, 1, 6, 6, 1, 6, 6, 21, 6, 6, 21, 21, 6, 21, 21, 21];
        assert_eq!(sig_coef_ctx_2d(0, 0, 4, 4, &levels, &offset, 1), 5);
    }

    #[test]
    fn level_ctx_returns_zero_for_all_below_base() {
        let levels = [0i8; 16];
        assert_eq!(level_ctx(0, 0, 4, 4, &levels), 0);
    }
}
