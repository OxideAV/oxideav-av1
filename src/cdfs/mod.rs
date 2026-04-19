//! AV1 default CDF tables.
//!
//! The tables come from libaom (`av1/common/entropymode.c`) and are
//! ported verbatim through the `tools/gen_cdfs` Go program which
//! imports `github.com/KarpelesLab/goavif/av1/entropy/cdfs` and emits
//! `generated.rs`. Every entry is already in wire format:
//!
//! - `cdf[0..N-2]` are Q15 `P(X>i) * 32768` values (monotonically
//!   decreasing).
//! - `cdf[N-1]` is the 0 sentinel.
//! - `cdf[N]` is the adaptive-update counter, initialised to 0.
//!
//! Re-run after updating goavif:
//!
//! ```sh
//! (cd tools/gen_cdfs && go build . && ./oxideav-gen-cdfs) > src/cdfs/generated.rs
//! ```
//!
//! The Loop Restoration per-unit CDFs (§5.11.40-.44) are hand-copied
//! from `libaom/av1/common/entropymode.c` into `lr.rs` — they aren't
//! part of goavif's CDF port because goavif short-circuits LR
//! per-unit signalling.

mod generated;
mod lr;

pub use generated::*;
pub use lr::{
    DEFAULT_SGRPROJ_RESTORE_CDF, DEFAULT_SWITCHABLE_RESTORE_CDF, DEFAULT_WIENER_RESTORE_CDF,
};
