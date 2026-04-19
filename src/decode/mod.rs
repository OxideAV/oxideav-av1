//! AV1 tile-decode pipeline — Phase 2 (partition walk + mode decode).
//!
//! This sub-crate ports
//! `github.com/KarpelesLab/goavif/av1/decoder/{block,partition,modes,
//! coeff_ctx,tx_type_map,tile,superblock}.go` to pure Rust, minus the
//! pixel-reconstruction branches. It walks the AV1 partition
//! quadtree (§5.11.4) and decodes every MI-unit's mode info
//! (§5.11.18) — Y mode, UV mode, skip, segment_id, angle_delta, CFL
//! sign/alpha. Coefficient decode (§5.11.39), transforms (§7.7),
//! intra prediction (§7.11.2), and loop filtering are **not**
//! implemented here; the first non-skip leaf returns
//! `Error::Unsupported("av1 coefficient decode pending (§5.11.39)")`.
//!
//! Top-level entry point: [`tile::decode_tile_group`]. The tile
//! walker mutates a [`frame_state::FrameState`] in place.

pub mod block;
pub mod coeff_ctx;
pub mod coeffs;
pub mod frame_state;
pub mod lr_unit;
pub mod modes;
pub mod partition;
pub mod reconstruct;
pub mod superblock;
pub mod tile;
pub mod tx_type_map;

pub use block::{BlockSize, PartitionType, SubBlock};
pub use frame_state::{FrameState, ModeInfo};
pub use modes::{IntraMode, UvMode, INTRA_MODES, UV_MODES};
pub use partition::walk_partition;
pub use tile::{decode_tile_group, finish_frame, TileDecoder};
