//! AV1 default CDF tables.
//!
//! The tables come from libaom (`av1/common/entropymode.c`) and are
//! emitted verbatim by the `tools/gen_cdfs` Go program into
//! `generated.rs`. Every entry is already in wire format:
//!
//! - `cdf[0..N-2]` are Q15 `P(X>i) * 32768` values (monotonically
//!   decreasing).
//! - `cdf[N-1]` is the 0 sentinel.
//! - `cdf[N]` is the adaptive-update counter, initialised to 0.
//!
//! To regenerate:
//!
//! ```sh
//! (cd tools/gen_cdfs && go build . && ./oxideav-gen-cdfs) > src/cdfs/generated.rs
//! ```
//!
//! The Loop Restoration per-unit CDFs (§5.11.40-.44) are hand-copied
//! from `libaom/av1/common/entropymode.c` into `lr.rs` — they aren't
//! part of the auto-generated set because the generator doesn't cover
//! per-unit LR signalling.

mod extra;
mod generated;
mod lr;

pub use extra::{
    DEFAULT_DELTA_LF_CDF, DEFAULT_DELTA_Q_CDF, DEFAULT_FILTER_INTRA_MODE_CDF, DEFAULT_INTRABC_CDF,
    DEFAULT_PALETTE_UV_COLOR_SIZE_2_CDF, DEFAULT_PALETTE_UV_COLOR_SIZE_3_CDF,
    DEFAULT_PALETTE_UV_COLOR_SIZE_4_CDF, DEFAULT_PALETTE_UV_COLOR_SIZE_5_CDF,
    DEFAULT_PALETTE_UV_COLOR_SIZE_6_CDF, DEFAULT_PALETTE_UV_COLOR_SIZE_7_CDF,
    DEFAULT_PALETTE_UV_COLOR_SIZE_8_CDF, DEFAULT_PALETTE_UV_MODE_CDF, DEFAULT_PALETTE_UV_SIZE_CDF,
    DEFAULT_PALETTE_Y_COLOR_SIZE_2_CDF, DEFAULT_PALETTE_Y_COLOR_SIZE_3_CDF,
    DEFAULT_PALETTE_Y_COLOR_SIZE_4_CDF, DEFAULT_PALETTE_Y_COLOR_SIZE_5_CDF,
    DEFAULT_PALETTE_Y_COLOR_SIZE_6_CDF, DEFAULT_PALETTE_Y_COLOR_SIZE_7_CDF,
    DEFAULT_PALETTE_Y_COLOR_SIZE_8_CDF, DEFAULT_PALETTE_Y_MODE_CDF, DEFAULT_PALETTE_Y_SIZE_CDF,
    DEFAULT_SEGMENT_ID_PREDICTED_CDF, DEFAULT_TXFM_SPLIT_CDF, DEFAULT_USE_FILTER_INTRA_CDF,
    PALETTE_COLOR_CONTEXT, PALETTE_COLOR_HASH_MULTIPLIERS,
};
pub use generated::*;
pub use lr::{
    DEFAULT_SGRPROJ_RESTORE_CDF, DEFAULT_SWITCHABLE_RESTORE_CDF, DEFAULT_WIENER_RESTORE_CDF,
};
