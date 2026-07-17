//! Decoder side of the crate — round 224 (arc 18).
//!
//! Arc 18 is the **integration arc** that wires the existing
//! decoder modules (`obu`, `sequence_header`, `frame_header`,
//! `parse_tile_group_obu_body`, the §5.11.39 `coefficients()` reader
//! on [`crate::PartitionWalker`], the §7.13 inverse-transform driver
//! [`crate::transform::inverse_transform_2d`], the §7.12.3 step-1
//! dequantizer [`crate::dequantize_step1`], and the §7.11.2.5
//! DC_PRED leaf [`crate::predict_intra_dc_pred`]) into a single
//! pixel-out entry point: [`decode_av1`], the inverse of
//! [`crate::encoder::encode_intra_frame_yuv`].
//!
//! ## Scope (arc 18)
//!
//! Matches the encoder pixel-driver's hard scope exactly: a 16×16
//! 4:2:0 YUV intra-only frame at `base_q_idx = 0` (lossless WHT
//! arm), BLOCK_4X4 leaves, TX_4X4 DCT_DCT default scan, no
//! segmentation, no QM, no in-loop filters (`enable_cdef = 0`,
//! `enable_restoration = 0`, `loop_filter_level = 0`,
//! `enable_superres = 0`, `apply_grain = 0`). Under this combination
//! the §7.14 / §7.15 / §7.16 / §7.17 / §7.18.3 post-processing
//! passes are all no-ops, so the decoder can skip directly from
//! `compute_prediction` + `residual` reconstruction into the output
//! frame buffer.
//!
//! The `Frame` output ([`Frame::Yuv420_16x16`]) mirrors the
//! encoder's [`crate::encoder::pixel_driver::Yuv420Frame16x16`] in
//! shape. Encoder → `decode_av1` → pixel-equality is the first
//! full encode-decode roundtrip exercise via the public API.
//!
//! ## What this arc does NOT do
//!
//!   * Decode arbitrary AV1 streams on the MIRROR path. Frame sizes
//!     other than 16×16, base_q_idx > 0, partition shapes other than
//!     the BLOCK_16X16 → BLOCK_8X8 → BLOCK_4X4 split tree, inter
//!     frames, multi-tile frames, and the in-loop filter stack are
//!     out of the mirror driver's scope. As of r409 the public
//!     [`decode_av1`] entry no longer REJECTS such streams: it falls
//!     back to the spec-faithful [`decode_av1_spec`] frame driver and
//!     surfaces each shown frame as [`pixel_driver::Frame::Spec`] —
//!     full public-API parity with the internal spec driver (see
//!     [`pixel_driver::decode_av1`]'s path-ordering contract).
//!   * In-loop post-processing. The §7.14 / §7.15 / §7.16 / §7.17 /
//!     §7.18.3 passes are no-ops on the lossless arc-18 frame's
//!     parameter set (`loop_filter_level = 0`, `enable_cdef = 0`,
//!     etc.); subsequent arcs will exercise them.
//!   * Run the §5.11.4 `PartitionWalker::decode_partition_syntax`
//!     driver. The walker's full §5.11.5 dispatch reads many more
//!     side bits (CDEF / delta-q / delta-lf / segment-id / skip-mode
//!     / mode-info ctx walks) than the encoder's arc-15..17 leaf
//!     writer emits — the two surfaces are intentionally minimal
//!     here. The arc-18 driver re-implements the matching minimal
//!     dispatch shape against the encoder's leaf writer. The full
//!     walker is wired for richer streams in a follow-up arc once
//!     the encoder grows the matching side-bit emissions.
//!
//! ## Spec provenance
//!
//! `docs/video/av1/av1-spec.txt`:
//!   * §5.3.1 — Open Bitstream Unit framing.
//!   * §5.5.1 / §5.9.1 / §5.11.1 — SH / FH / TileGroup OBU bodies.
//!   * §5.11.4 — `decode_partition()` recursion (mirrored shape).
//!   * §5.11.5 — `decode_block()` per-leaf reads (intra arm).
//!   * §5.11.11 — `read_skip()`.
//!   * §5.11.22 — `y_mode()` / `uv_mode()`.
//!   * §5.11.39 — `coefficients()`.
//!   * §7.11.2.5 — DC_PRED sample generation.
//!   * §7.12.3 — `dequantize_step1`.
//!   * §7.13 — Inverse transform.

// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod frame_driver;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod pixel_driver;
// internal — exposed for tests/fuzz; not part of the stable API
#[doc(hidden)]
pub mod pixel_driver_dyn;

pub use frame_driver::{decode_av1_spec, SpecFrame};
#[doc(hidden)]
pub use frame_driver::{decode_frame_spec, SpecDecodeSession};
pub use pixel_driver::{decode_av1, Frame};
#[doc(hidden)]
pub use pixel_driver::{decode_temporal_unit, TemporalUnitResult};
