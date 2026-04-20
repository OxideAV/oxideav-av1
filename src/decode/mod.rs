//! AV1 tile-decode pipeline — Phase 2 (partition walk + mode decode).
//!
//! Walks the AV1 partition quadtree (§5.11.4) and decodes every
//! MI-unit's mode info (§5.11.18) — Y mode, UV mode, skip, segment_id,
//! angle_delta, CFL sign/alpha. Coefficient decode (§5.11.39),
//! transforms (§7.7), intra prediction (§7.11.2), and loop filtering
//! are **not** implemented here; the first non-skip leaf returns
//! `Error::Unsupported("av1 coefficient decode pending (§5.11.39)")`.
//!
//! Top-level entry point: [`tile::decode_tile_group`]. The tile
//! walker mutates a [`frame_state::FrameState`] in place.

pub mod block;
pub mod coeff_ctx;
pub mod coeffs;
pub mod frame_state;
pub mod inter;
pub mod inter_block;
pub mod lr_unit;
pub mod mc;
pub mod modes;
pub mod mv;
pub mod partition;
pub mod reconstruct;
pub mod superblock;
pub mod tile;
pub mod tx_type_map;

pub use block::{BlockSize, PartitionType, SubBlock};
pub use frame_state::{FrameState, ModeInfo};
pub use inter::{InterDecoder, InterMode};
pub use modes::{IntraMode, UvMode, INTRA_MODES, UV_MODES};
pub use mv::{Mv, MvDecoder, MvJoint};
pub use partition::walk_partition;
pub use tile::{decode_tile_group, finish_frame, TileDecoder};
