# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- round 22 — spec-correct AV1 inverse-transform path landed alongside
  the legacy `inverse_2d`. New entry point `inverse_2d_spec` follows
  §7.13.3 exactly: per-shape `Transform_Row_Shift[TX_SIZES_ALL]` table
  applied between row and column passes, constant `colShift = 4` after
  the column pass, and the rectangular `Round2(T[j] * 2896, 12)`
  per-element pre-row scale fired only for `|log2W - log2H| == 1`
  (the 2:1 aspect shapes — Tx4x8/Tx8x4/Tx8x16/Tx16x8/Tx16x32/Tx32x16/
  Tx32x64/Tx64x32). 1:4 and 4:1 shapes (Tx4x16/Tx16x4/Tx8x32/Tx32x8/
  Tx16x64/Tx64x16) and squares correctly skip it. New
  `transform/idtx_spec.rs` module ships the spec-magnitude IDTX
  kernels (`idtx4_spec`: `Round2(T*5793, 12)` ≈ ×√2;
  `idtx8_spec`: ×2; `idtx16_spec`: `Round2(T*11586, 12)` ≈ ×2√2;
  `idtx32_spec`: ×4) per §7.13.2.11/12/13/14, replacing the
  uniform-`<<= 1` legacy variants on the new path. The new path also
  drops the redundant `flip_1d` wrapper used by `inverse_2d` for
  FLIPADST kernels — `iflipadst*` already reverses its own output, so
  wrapping with pre-flip + post-flip cancelled the kernel's reverse
  and produced `IADST(reverse(input))` instead of the spec-equivalent
  `reverse(IADST(input))`. The legacy `inverse_2d` is preserved
  verbatim because its callers in `decode/superblock.rs` are
  PSNR-calibrated against the legacy IDTX magnitudes and the bucketed
  post-2D `inverse_shift`; switching them over to `inverse_2d_spec`
  needs the per-shape `residual_shift` accounting revised in tandem
  and is deferred to the caller-migration round. New tests pin: the
  full `Transform_Row_Shift` table verbatim
  (`transform_row_shift_matches_spec_table`); every spec-allowed
  TX_TYPE × TX_SIZE pair (159 of 323 — 9 sizes × 16 types in the
  full INTER_1 set + 5 sizes × 2 types in INTER_3 + 5 sizes × 1 type
  in DCTONLY — `inverse_2d_spec_covers_every_spec_allowed_pair`);
  spec-correct IDTX magnitudes per length (`idtx4_spec_unit_and_scale`,
  `idtx16_spec_unit_and_scale`, `idtx32_spec_quadruples_each_sample`);
  the rectangular 2896 trigger gate
  (`inverse_2d_spec_rect_2896_pre_scale_only_2to1`); DC-constant
  reconstruction across all 14 rectangular shapes
  (`inverse_2d_spec_dct_dc_constant_across_all_rectangles`); the
  `iflipadst4 == reverse(iadst4)` invariant
  (`inverse_2d_spec_flipadst_uses_kernel_internal_reverse`); and the
  spec-disallowed kernel rejection set
  (`inverse_2d_spec_rejects_disallowed_kernels`).
- round 21 — fix §5.9.2 inter-branch ORDER: `frame_size()` and
  `render_size()` now run AFTER the `ref_frame_idx[]` loop, not
  before. The previous ordering happened to consume the same total
  number of bits as the spec for the typical override-off /
  superres-off case, but mis-aligned the bitstream by ~13 bits on
  every non-short-signaling inter frame because spec's
  `frame_refs_short_signaling=0` path consumes 21 ref_frame_idx
  bits before render_size. The misalignment caused 10/48 SVT-AV1
  chain frames to mis-interpret tile_group bits as `gm_params`
  type bits — surfacing as `parse_global_motion_params` AFFINE
  overruns at bit 130-131 with only 5 bits remaining for 6 params,
  or `parse_lr_params` `out of bits`. Round 21 also wires §5.9.7
  `frame_size_with_refs` enough to read the 7 `found_ref` bits
  (returning Unsupported when any are set, since
  RefFrameWidth/Height tracking is still pending). New regression
  test `inter_branch_reads_frame_size_after_ref_frame_idx_loop`
  pins the post-fix bit ordering against a synthesised SVT-AV1-
  style payload. `svtav1_chain_walk` baseline raised from 38/48
  to 48/48 (full pass) and renamed to
  `svtav1_chain_walk_round21_full_pass`.
- round 21 — env-gated `AV1_TRACE_BITS=1` instrumentation extended
  with per-section checkpoints inside `parse_loop_filter_params`
  and per-OBU layout dump in the chain-walk test (sequence-header
  config + per-packet OBU types + per-frame payload hex). The
  diagnostics make future bit-account regressions bisectable in
  one trace pass instead of repeated re-builds.

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
