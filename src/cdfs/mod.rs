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

mod generated;
mod lr;

pub use generated::*;
pub use lr::{
    DEFAULT_SGRPROJ_RESTORE_CDF, DEFAULT_SWITCHABLE_RESTORE_CDF, DEFAULT_WIENER_RESTORE_CDF,
};
