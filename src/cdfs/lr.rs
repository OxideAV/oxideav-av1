//! Loop Restoration per-unit CDFs — spec §5.11.40-.44.
//!
//! Values hand-copied from libaom `av1/common/entropymode.c` (3.12.1):
//!
//! ```c
//! static const aom_cdf_prob default_switchable_restore_cdf[
//!     CDF_SIZE(RESTORE_SWITCHABLE_TYPES)] = { AOM_CDF3(9413, 22581) };
//! static const aom_cdf_prob default_wiener_restore_cdf[CDF_SIZE(2)] =
//!     { AOM_CDF2(11570) };
//! static const aom_cdf_prob default_sgrproj_restore_cdf[CDF_SIZE(2)] =
//!     { AOM_CDF2(16855) };
//! ```
//!
//! Wire format matches the rest of [`crate::cdfs`]:
//! - 2-symbol CDFs: `[Q15_p_gt_0, 0_sentinel, 0_counter]`.
//! - 3-symbol switchable CDF: `[Q15_p_gt_0, Q15_p_gt_1, 0_sentinel, 0_counter]`.
//!
//! These CDFs are NOT part of the auto-generated table in
//! [`generated`](super::generated) — the generator does not cover
//! per-unit LR signalling, whereas oxideav implements the full
//! §5.11.40-.44 path.

/// Switchable-type CDF — 3 symbols: 0 = `RESTORE_NONE`, 1 =
/// `RESTORE_WIENER`, 2 = `RESTORE_SGRPROJ`. Read only when the frame
/// header's `FrameRestorationType[plane]` equals `RESTORE_SWITCHABLE`.
pub const DEFAULT_SWITCHABLE_RESTORE_CDF: [u16; 4] = [9413, 22581, 0, 0];

/// `use_wiener` flag — 2 symbols (0 = no, 1 = yes). Read only when
/// `FrameRestorationType[plane] == RESTORE_WIENER`.
pub const DEFAULT_WIENER_RESTORE_CDF: [u16; 3] = [11570, 0, 0];

/// `use_sgrproj` flag — 2 symbols. Read only when
/// `FrameRestorationType[plane] == RESTORE_SGRPROJ`.
pub const DEFAULT_SGRPROJ_RESTORE_CDF: [u16; 3] = [16855, 0, 0];
