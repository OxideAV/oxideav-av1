//! AV1 intra prediction mode taxonomy — §6.4.1 / §5.11.18
//! `intra_mode_info`.
//!
//! Numeric values match the bitstream encoding so they can be decoded
//! from CDFs without translation.
//!
//! This module only defines the enum shape + the context-bucket helper
//! the mode decoder needs. The actual `DC_PRED` / `V_PRED` / `H_PRED`
//! sample-level predictor lives in [`crate::intra`]; full support for
//! the remaining 10 modes is Phase 3+.

/// AV1 per-block intra prediction modes (spec §6.4.1). The low 13
/// values are the primary `IntraMode`; [`UvMode`] extends with the
/// CFL sentinel at 13.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntraMode {
    DcPred = 0,
    VPred = 1,
    HPred = 2,
    D45Pred = 3,
    D135Pred = 4,
    D113Pred = 5,
    D157Pred = 6,
    D203Pred = 7,
    D67Pred = 8,
    SmoothPred = 9,
    SmoothVPred = 10,
    SmoothHPred = 11,
    PaethPred = 12,
    /// Chroma-only CFL sentinel (UV plane only). Signalled as symbol
    /// 13 in the `uv_mode` CDF; never appears in Y.
    CflPred = 13,
}

/// Number of primary intra modes (everything before `CflPred`).
pub const INTRA_MODES: usize = 13;
/// Number of UV-plane modes, including `CflPred`.
pub const UV_MODES: usize = 14;

impl IntraMode {
    /// Convert an already-validated symbol value to the enum. Returns
    /// `None` for out-of-range values (callers surface that as a
    /// bitstream violation).
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::DcPred,
            1 => Self::VPred,
            2 => Self::HPred,
            3 => Self::D45Pred,
            4 => Self::D135Pred,
            5 => Self::D113Pred,
            6 => Self::D157Pred,
            7 => Self::D203Pred,
            8 => Self::D67Pred,
            9 => Self::SmoothPred,
            10 => Self::SmoothVPred,
            11 => Self::SmoothHPred,
            12 => Self::PaethPred,
            13 => Self::CflPred,
            _ => return None,
        })
    }

    /// `true` for the 6 directional modes (`D45`..`D67`) — these take
    /// an `angle_delta` in `{-3..3}` from the bitstream.
    pub fn is_directional(self) -> bool {
        matches!(
            self,
            Self::D45Pred
                | Self::D135Pred
                | Self::D113Pred
                | Self::D157Pred
                | Self::D203Pred
                | Self::D67Pred
        )
    }

    /// Spec mnemonic for the mode (matches libaom names).
    pub fn name(self) -> &'static str {
        match self {
            Self::DcPred => "DC_PRED",
            Self::VPred => "V_PRED",
            Self::HPred => "H_PRED",
            Self::D45Pred => "D45_PRED",
            Self::D135Pred => "D135_PRED",
            Self::D113Pred => "D113_PRED",
            Self::D157Pred => "D157_PRED",
            Self::D203Pred => "D203_PRED",
            Self::D67Pred => "D67_PRED",
            Self::SmoothPred => "SMOOTH_PRED",
            Self::SmoothVPred => "SMOOTH_V_PRED",
            Self::SmoothHPred => "SMOOTH_H_PRED",
            Self::PaethPred => "PAETH_PRED",
            Self::CflPred => "CFL_PRED",
        }
    }
}

/// UV-plane mode. Alias kept for spec-parity — the type is the same
/// as [`IntraMode`] but `CflPred` may appear.
pub type UvMode = IntraMode;

/// 5-bucket mode context used by `Default_Intra_Frame_Y_Mode_Cdf`
/// (§9.4 / spec table 22182).
///
/// Spec `Intra_Mode_Context[INTRA_MODES] = { 0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0 }`
/// indexed by the AV1 intra mode enum, namely:
/// - DC(0)→0, V(1)→1, H(2)→2,
/// - D45(3)→3, D135(4)→4, D113(5)→4, D157(6)→4, D203(7)→4, D67(8)→3,
/// - Smooth(9)→0, SmoothV(10)→1, SmoothH(11)→2, Paeth(12)→0.
pub fn mode_ctx_bucket(m: IntraMode) -> u32 {
    match m {
        IntraMode::DcPred => 0,
        IntraMode::VPred => 1,
        IntraMode::HPred => 2,
        IntraMode::D45Pred | IntraMode::D67Pred => 3,
        IntraMode::D135Pred | IntraMode::D113Pred | IntraMode::D157Pred | IntraMode::D203Pred => 4,
        IntraMode::SmoothPred => 0,
        IntraMode::SmoothVPred => 1,
        IntraMode::SmoothHPred => 2,
        IntraMode::PaethPred => 0,
        IntraMode::CflPred => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_directional_matches_spec() {
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
            assert_eq!(m.is_directional(), *want, "{:?}", m);
        }
    }

    #[test]
    fn mode_ctx_bucket_table() {
        // Spec table 22182: { 0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0 }
        assert_eq!(mode_ctx_bucket(IntraMode::DcPred), 0);
        assert_eq!(mode_ctx_bucket(IntraMode::VPred), 1);
        assert_eq!(mode_ctx_bucket(IntraMode::HPred), 2);
        assert_eq!(mode_ctx_bucket(IntraMode::D45Pred), 3);
        assert_eq!(mode_ctx_bucket(IntraMode::D135Pred), 4);
        assert_eq!(mode_ctx_bucket(IntraMode::D113Pred), 4);
        assert_eq!(mode_ctx_bucket(IntraMode::D157Pred), 4);
        assert_eq!(mode_ctx_bucket(IntraMode::D203Pred), 4);
        assert_eq!(mode_ctx_bucket(IntraMode::D67Pred), 3);
        assert_eq!(mode_ctx_bucket(IntraMode::SmoothPred), 0);
        assert_eq!(mode_ctx_bucket(IntraMode::SmoothVPred), 1);
        assert_eq!(mode_ctx_bucket(IntraMode::SmoothHPred), 2);
        assert_eq!(mode_ctx_bucket(IntraMode::PaethPred), 0);
    }
}
