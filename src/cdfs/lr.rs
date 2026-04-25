//! Loop Restoration per-unit CDFs — spec §5.11.40-.44.
//!
//! Values transcribed from the AV1 spec (§9.4 "Additional tables"):
//!
//! - `Default_Restoration_Type_Cdf = {9413, 22581, 32768, 0}` (3 symbols)
//! - `Default_Use_Wiener_Cdf = {11570, 32768, 0}` (2 symbols)
//! - `Default_Use_Sgrproj_Cdf = {16855, 32768, 0}` (2 symbols)
//!
//! Wire format matches the rest of [`crate::cdfs`]: each non-sentinel
//! entry is the survival function `P(X > i) * 32768 = 32768 -
//! cdf_spec[i]` (monotonically decreasing). Spec values are
//! cumulative `P(X <= i) * 32768`, so we invert them on transcription.
//!
//! - 2-symbol CDFs: `[p_gt_0, 0_sentinel, 0_counter]`.
//! - 3-symbol switchable CDF: `[p_gt_0, p_gt_1, 0_sentinel, 0_counter]`.
//!
//! These CDFs are NOT part of the auto-generated table in
//! [`generated`](super::generated) — the generator does not cover
//! per-unit LR signalling, whereas oxideav implements the full
//! §5.11.40-.44 path.

/// Switchable-type CDF — 3 symbols: 0 = `RESTORE_NONE`, 1 =
/// `RESTORE_WIENER`, 2 = `RESTORE_SGRPROJ`. Read only when the frame
/// header's `FrameRestorationType[plane]` equals `RESTORE_SWITCHABLE`.
/// Spec `{9413, 22581, 32768, 0}` → `{32768-9413, 32768-22581, 0, 0}`.
pub const DEFAULT_SWITCHABLE_RESTORE_CDF: [u16; 4] = [23355, 10187, 0, 0];

/// `use_wiener` flag — 2 symbols (0 = no, 1 = yes). Read only when
/// `FrameRestorationType[plane] == RESTORE_WIENER`. Spec
/// `{11570, 32768, 0}` → `{32768-11570, 0, 0}`.
pub const DEFAULT_WIENER_RESTORE_CDF: [u16; 3] = [21198, 0, 0];

/// `use_sgrproj` flag — 2 symbols. Read only when
/// `FrameRestorationType[plane] == RESTORE_SGRPROJ`. Spec
/// `{16855, 32768, 0}` → `{32768-16855, 0, 0}`.
pub const DEFAULT_SGRPROJ_RESTORE_CDF: [u16; 3] = [15913, 0, 0];
