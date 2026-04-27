# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- round 19 — `tests/svtav1_chain_walk.rs` chain-walk diagnostic.
  Walks every Frame OBU in `/tmp/av1_inter.ivf` with the §7.20
  `Dpb::refresh_with_gm` chain wired through and asserts a
  `parsed_ok ≥ 38 / total ≥ 44` floor; on failure it reports the
  first-fail `(packet, frame#)`. The current SVT-AV1 fixture (48
  Frame OBUs across 48 IVF packets) parses 38/48 cleanly; the 10
  remaining frames all fail with `out of bits` inside
  `parse_global_motion_params` for slot 2 (AFFINE) — investigation
  notes captured inline in the test docstring + workspace README.
  Skipped if the fixture is missing so CI without ffmpeg/libsvtav1
  installed remains green.
- round 19 — `frame_header_tail::gm_tests::affine_minimum_bit_count_for_identity_prev`
  locks down the spec read order for AFFINE warps with the identity-
  default `prev_gm_params`: 3 type bits + 6 params × 4-bit minimum
  subexp + 6 trailing IDENTITY slots = exactly 33 consumed bits. Pairs
  with the round-18 ROTZOOM regression guard so any future reordering
  of `read_global_param_with_ref` in `parse_global_motion_params`
  trips a focused unit test instead of only the integration-level
  `svtav1_chain_walk_baseline_38_of_48` assertion.

- round 16 — implement §5.11.38 `Subsampled_Size[][2][2]` table on
  `BlockSize` and route every chroma reconstruction (`reconstruct_inter_chroma_block`,
  `reconstruct_skip_mode_compound_chroma`, `reconstruct_chroma_block`)
  through it via the new `chroma_residual_dims` helper. Eliminates the
  bogus 8×2 / 2×8 chroma TX dispatches that bailed real-encoder
  bitstreams; narrow luma blocks (16×4, 4×16, etc.) now produce the
  spec-correct 8×4 / 4×8 chroma TX.
- round 16 — apply §7.13.3 step-f dequant clip
  (`±(1 << (7 + BitDepth))`) inside `dequant_coeff` and the
  inter-pass `colClampRange = 16` clip inside `inverse_2d`. Stops
  `i32` overflow in `half_btf` on real-encoder coefficient streams
  that the round-15 §8.2.6 fix newly exposed. Promote
  `svtav1_skip_mode_compound_decodes_real_pixels` to a hard
  assertion: SVT-AV1 SkipMode compound frame now decodes end-to-end
  with multi-ref planes.

- round 12 — wire §5.11.10 read_skip_mode + §5.11.19 inter_segment_id
  end-to-end on the inter leaf path (DEFAULT_SEGMENT_ID_PREDICTED_CDF
  added; new `decode_skip_mode` / `decode_seg_id_predicted` /
  `seg_feature_active` helpers; `InterBlockInfo` now carries `skip_mode`
  + `segment_id`; `MiInfo` gained a `skip_mode` field for the §9.4 ctx).
  Inter-leaf path also gained the spec-mandated read_delta_qindex /
  read_delta_lf calls (no-op on aomenc fixtures, but bitstream-correct).
- read_skip ctx now sums above/left Skips[][] neighbours per §9.4
  (was hard-coded to 0).
- ns(1) underflow guard in BitReader — fixes a crash on SVT-AV1 fixtures
  whose tile_info / dim signalling reaches the n=1 branch.

## [0.1.1](https://github.com/OxideAV/oxideav-av1/compare/v0.1.0...v0.1.1) - 2026-04-25

### Fixed

- fix CI clippy + pin release-plz to 0.1.x

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- round 9b — unit tests for delta_q / delta_lf apply arithmetic
- round 9a — filter-intra dispatch + CurrentQIndex/DeltaLF apply
- round 8 — palette_mode_info + var-tx inter + inter-skip ordering
- round 7 — wire delta_q/lf + use_intrabc + filter_intra_mode_info (§5.11.7/.12/.13/.24)
- wire read_block_tx_size() + fix read_skip ordering (§5.11.5/.7/.16)
- document testsrc luma gap root causes (investigation notes)
- spec-correct intra neighbor gathering (§7.11.2.1)
- §7.9 reference scaling scaffolding + scaled luma MC
- retain global-motion params + compound luma MC skeleton (§7.10 / §7.11.3.9)
- round-4 inter MC — §7.11.3.3 MV clamp + §5.11.26 per-block interp filter
- spec-exact per-SB CDEF (§7.15 + §5.11.56)
- wire intra edge filter + upsample into TX-unit predictors
- add intra edge filter + upsample (§7.11.2.9-.12)
- drop goavif origin markers (self-port; no third-party attribution needed)
- add AVIF-still 128×128-SB decode fixture + test
- run intra prediction per TX unit, not per block (§7.11.2)
- split 128×N intra blocks into 64×64 TX units (§5.11.27)
- fix reduced_still_picture_header parsing (§5.9.1)
- release v0.0.4

## [0.1.0](https://github.com/OxideAV/oxideav-av1/compare/v0.0.3...v0.1.0) - 2026-04-19

### Other

- Update Cargo.toml: set release to 0.1.0
- fix debug-mode panic in decode_uniform (§5.11.41)
- rustfmt sweep across Phase 1-7 code
- finish Phase 7 — inter-intra path + gm_params + AVIS test
- land Phase 7 plumbing — interp/mv/mc/inter + DPB + inter leaf path
- add LR + film-grain integration fixtures
- wire per-unit LR signalling + finish_frame LR/grain apply
- add lr + filmgrain modules — Phase 6 primitives
- add 128×128 non-skip fixture integration test
- land Phase 5 — full intra + deblock + CDEF in one sweep
- update crate docs + codec capability tag for Phase 4
- extend TX-size coverage to every rectangular shape + add HBD test
- wire 32/64 TX sizes + tx_type decode + HBD plumbing
- port idct32/idct64/idtx/iwht4/flipadst + wire full 1D dispatch
- update crate-level docs + codec capability tag for Phase 3
- add end-to-end tile-walk decode test against /tmp/av1.ivf
- wire coefficient decoder + intra predict + reconstruct into tile walk
- add decode::coeffs + decode::reconstruct — §5.11.39 / §7.7.4
- rewrite transform as modular 4/8/16 iDCT + iADST dispatch
- add quant module — DC/AC dequantiser tables + per-plane compute
- wire decode::decode_tile_group into Av1Decoder; drop tile_decode stub
- add decode::tile + decode::superblock — partition walker + mode decoder
- add decode module helpers — block/partition/modes/coeff_ctx/tx_type_map/frame_state
- parse uncompressed-header tail — quant through film-grain
- rewrite range-coder symbol decoder against goavif wire-format
- import default CDF tables via gen_cdfs Go-to-Rust generator
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- claim AVI FourCC via oxideav-codec CodecTag registry
- drop vestigial placeholder write in SymbolDecoder::new
- parse tile_info (§5.9.15) + tile_group framing + README rewrite
