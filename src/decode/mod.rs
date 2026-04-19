//! AV1 tile-decode pipeline — Phase 2 (partition walk + mode decode).
//!
//! Ports `github.com/KarpelesLab/goavif/av1/decoder/*` to pure Rust,
//! minus the pixel-reconstruction branches. This initial chunk lands
//! the pure-data helpers: block-size / partition-type / intra-mode
//! taxonomy, oracle-driven partition walker, coefficient-context
//! helpers (for Phase 3), TX-type set mapping (for Phase 3), and the
//! mutable per-frame `FrameState` holding the MI grid.
//!
//! The tile + superblock decoder (which reads range-coded partition
//! and mode symbols) lands in the next commit.

pub mod block;
pub mod coeff_ctx;
pub mod frame_state;
pub mod modes;
pub mod partition;
pub mod tx_type_map;

pub use block::{BlockSize, PartitionType, SubBlock};
pub use frame_state::{FrameState, ModeInfo};
pub use modes::{IntraMode, UvMode, INTRA_MODES, UV_MODES};
pub use partition::walk_partition;
