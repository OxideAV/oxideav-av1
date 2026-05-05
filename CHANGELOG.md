# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4](https://github.com/OxideAV/oxideav-av1/compare/v0.1.3...v0.1.4) - 2026-05-04

### Fixed

- *(clippy)* replace redundant smooth_pred closures with direct fn refs

### Other

- rewrite SymbolEncoder as streaming 16-bit live register ([#388](https://github.com/OxideAV/oxideav-av1/pull/388))
- av1 decoder: round 6 follow-up — close #403 OOB + #401 wrong spec dim
- av1 decoder: round 6 #394 — replace run_smooth_padded_* shim with spec/clipped split
- cargo fmt — collapse rust-formatter splits in #393 changes
- av1 decoder: round 6 #393 — extend round-5 six-fix pattern to inter intra-within-inter
- av1 decoder: round 5 — close 6 spec-syntax desync bugs (#386, #387)

### Changed

- **encoder `SymbolEncoder` now streams** (#388) — replaced the round-2
  wide-bigint V-tracking design (`v_low: Vec<u8>` + `bake_pending_shifts`
  + `add_to_v_low` with `Vec::insert` carry handling, all O(N) per
  symbol → O(N²) total) with an O(1)-state streaming encoder built on
  a 16-bit live register (`low: u32`, bits 0..15) plus the spec's R
  internal range, with carry-back propagation through `out: Vec<u8>` /
  the partial `bit_buf`. The first renorm bit (V-position −1, above
  V's MSB) is dropped; the remaining 16 V bits emit at finish from
  `low`'s live region. Public API unchanged. New 1M-symbol roundtrip
  tests (`roundtrip_one_million_symbols_uniform`,
  `roundtrip_one_million_skewed_bools`) pin the streaming property —
  internal state no longer scales with symbol count.

### Fixed

- decoder round 6 follow-up (#403, #401) — close two regressions
  introduced by the round-6 (#394) spec/clipped split:
  - **#403 OOB panic**: `gather_neighbors_*` panicked with
    `index out of bounds` when the SPEC TU walk in
    `reconstruct_luma_block` / `reconstruct_chroma_block` placed a TU
    whose top-left already sits past the frame plane (e.g.
    Block128x128 at y=640 in a 722-tall frame walked at tx_h=16; or
    any block at the right edge of a frame whose width isn't a
    multiple of the SB size). The OOB TUs paste-collapse to a no-op
    but the predictor's neighbor gather still tried to read
    `(uy-1) * stride + sx` past the plane buffer. Fix: skip the
    predict + paste step for fully-OOB TUs (`wr_w == 0 || wr_h == 0`)
    while still consuming residual symbols when `!skip` so the range
    coder stays synchronized with the §5.11.34 SPEC-dim walk.
    Repro: oxideav-avif `decoder_pipes_through_av1_errors_cleanly` on
    `kimono_rotate90`.
  - **#401 `TX 4×32 not in the AV1 set`**: the round-6 helper
    `bs_or_clipped_spec(d)` rounded a clipped block dim UP to the
    next power-of-two ({4, 8, 16, 32, 64, 128}), but for blocks like
    Block16x64 clipped at the right edge to bw=5 the actual SPEC dim
    is 16 — `bs_or_clipped_spec(5)=8` produced the wrong spec shape
    and downstream `chroma_residual_dims(8, 64, 1, 1)` landed on
    `(spec_cw, spec_ch) = (4, 32)` which is not in the AV1 TX set.
    Fix: thread the original `bs: BlockSize` through
    `decode_inter_leaf_block` / `reconstruct_luma_block` /
    `reconstruct_chroma_block` and recover spec dims directly from
    `bs.width()` / `bs.height()`. Removes the now-unused
    `bs_or_clipped_spec` helper. The super-resolution ReportOnly
    fixture now reaches `visible_produced=1/1` (pending the §7.16
    super-res upscale step).

- decoder round 6 (#394) — replace the round-5 `run_smooth_padded_*`
  super-resolution shim with the proper spec/clipped split through
  the `reconstruct_luma_block` / `reconstruct_chroma_block` chain.
  The previous shim rounded up SMOOTH-only block dims to the next
  power-of-two table slot to keep the kernel from panicking on a
  width-5 block at the right frame edge of the `super-resolution`
  ReportOnly fixture, but every other predictor + the residual /
  inverse-transform path still ran at the clipped dim — which broke
  the spec contract that prediction operates at the FULL block size
  and only plane writes are clipped. Result: the SMOOTH path produced
  divergent samples past the clip line, and the residual was
  decoded against an invalid TX shape (e.g. `TX 5×16 not in the AV1
  set`).

  Now `reconstruct_luma_block` / `reconstruct_chroma_block` derive
  `(spec_w, spec_h)` from the input clipped `(bw, bh)` via
  `bs_or_clipped_spec` (rounds 1..=4→4, 5..=8→8, 9..=16→16, etc.),
  walk TX units at the spec dim, and pass `(tx_w, tx_h)` (spec) +
  `(wr_w, wr_h)` (clipped against frame edge) to the per-TU
  reconstruct. The TU reconstruct runs the predictor + IDCT residual
  + clip-add at the SPEC dim into a `tx_w × tx_h` scratch, then
  `paste_block_clipped(_, ..., src_w=tx_w, wr_w, wr_h)` writes only
  the in-frame rectangle.

  `run_smooth_padded_u8 / u16` and `next_smooth_dim` are removed —
  the SMOOTH dispatch now calls `smooth_pred*` directly with the
  spec dim. Two new helpers — `decode_dequant_idct_luma` /
  `_chroma` — factor out the per-TU residual decode (was inlined
  twice in the bit-depth dispatch).

  Test impact: corpus aggregate exact-pixel counts unchanged at
  power-of-two block sizes (the spec rounding is identity there).
  `super-resolution` ReportOnly fixture now reaches the next
  unsupported TX shape (`TX 4×32 not in the AV1 set`) instead of
  panicking on `TX 5×16`. All 471 local tests green.

- decoder round 6 (#393) — extend the round-5 six-fix pattern
  (`HasChroma` / `intra_angle_info_y` / `angle_delta_cdf` index /
  `cfl_signs` / `cfl_alpha_ctx` / `cfl_allowed`) to the
  intra-within-inter branch in `decode_inter_leaf_block`. The
  bitstream-syntax reads already happened on the keyframe-intra path
  after 7d1f297, but `decode_inter_block_syntax` still emitted the
  collapsed-DC defaults (`uv_mode = DcPred`, `angle_delta_*` = 0,
  `cfl_alpha_*` = 0) and the inter leaf reconstruction blindly fed
  those defaults to `reconstruct_chroma_block` — silently dropping
  every chroma-syntax read inside an inter frame's intra-within-inter
  block. Fix:
  1. Read `intra_angle_info_y` (`MiSize >= BLOCK_8X8` + directional).
  2. Compute `HasChroma` per §5.11.5 with `compute_has_chroma` (now
     `pub(crate)`); skip chroma syntax on the "left" / "top" half
     of a shared 4:2:x footprint.
  3. Compute the §9.4 `cfl_allowed` predicate using the spec
     `max(W, H) <= 32` rule (with the `chroma_residual_dims`-equals-
     4×4 lossless override) before reading `uv_mode` so the
     `CflAllowed=1` 14-symbol CDF only fires for eligible blocks.
  4. Read `intra_angle_info_uv` (same MiSize gate).
  5. Read `read_cfl_alphas` via `cfl_signs` + `cfl_alpha_ctx` for
     `uv_mode == CFL_PRED` blocks.
  Carry the new fields (`intra_uv_mode`, `intra_angle_delta_y`,
  `intra_angle_delta_uv`, `intra_cfl_alpha_u/_v`) on `InterBlockInfo`
  and thread them into the per-MI `mi_mut` writes plus the
  `reconstruct_luma_block` / `reconstruct_chroma_block` calls so the
  intra-within-inter branch produces real prediction values for the
  chroma plane instead of always falling back to DC. `chroma_residual_dims`
  and `shared_chroma_footprint` were lifted to `pub(crate)` for the
  shared call site.

  Test impact (corpus PSNR drift, ReportOnly):
  - `obu-with-extension-headers`: 276 → 284 exact pixels (+0.06pp,
    UV max-diff 187 → 162). Two-visible-frame inter clip with
    intra-within-inter blocks.
  - `show-existing-frame`: 616 → 657 exact pixels (+0.05pp, UV
    max-diff 238 → 249). 13-visible-frame mixed intra/inter clip.
  - All other corpus tests unchanged. Full test suite (471 cases)
    green via `CARGO_TARGET_DIR=/tmp/oxideav-av1-target cargo test
    -j 2 -- --test-threads 4`.

- decoder round 5 — close 6 spec-syntax desync bugs uncovered by the
  round-4 spec-correct `update_cdf` evolution and remove the
  `SM_WEIGHTS2` band-aid from the SMOOTH predictor:
  1. `HasChroma` per AV1 §5.11.5 — gate `uv_mode` + chroma residual
     reconstruct on the spec predicate (`bw4==1 && sub_x && (MiCol&1)==0`
     or `bh4==1 && sub_y && (MiRow&1)==0` ⇒ `HasChroma=0`). The
     pre-fix decoder always read `uv_mode` for any non-monochrome
     block, silently consuming the chroma-syntax symbols on narrow
     blocks at even MI cols / rows in 4:2:x and producing a TX-2x8
     panic on the `tile-cols-2-rows-1` corpus fixture (#386).
  2. `intra_angle_info_y / intra_angle_info_uv` per §5.11.42/43 —
     gate `angle_delta_y/uv` reads on `MiSize >= BLOCK_8X8`. Was
     unconditional on `is_directional()`, desyncing the range coder
     on every Block4x4 / 4x8 / 8x4 directional block.
  3. `angle_delta_cdf` index per §9.4 — `mode - V_PRED` (range 2..7),
     not `mode - D45_PRED` (range 0..5). The 8-entry table's first
     two slots are V/H contexts; the wrong index used those for the
     D45/D135 reads.
  4. `Intra_Mode_Context` table per §9.4 / spec line 22182 —
     `{ 0,1,2,3,4,4,4,4,3,0,1,2,0 }` (D135/D113/D157/D203 → 4;
     Smooth/SmoothV/SmoothH → 0/1/2; Paeth → 0). The pre-fix
     `mode_ctx_bucket` mapped every directional mode to 3 and every
     smooth/Paeth mode to 4 — opposite of spec for 8 of the 13
     entries.
  5. `cfl_signs` / `cfl_alpha_ctx` per §5.11.45 / §6.10.14 / §9.4 —
     compute the (signU, signV) pair as `((joint+1)/3, (joint+1)%3)`
     mapped through `{ZERO=0, NEG=1, POS=2}`, and the per-plane
     CDF context as `(signU-1)*3 + signV` (resp.
     `(signV-1)*3 + signU`). The pre-fix table-driven mapping was
     wrong for 7 of 8 joint values, silently zeroing CflAlphaU on a
     sizeable share of CFL-coded blocks.
  6. `cfl_allowed` per §9.4 — gate the 14-symbol UV CDF on
     `max(Block_Width, Block_Height) <= 32` (or 4×4 chroma residual
     under lossless). The pre-fix decoder always read the wider
     CDF, drifting the next symbol on 64-wide non-CFL blocks.

  The chroma reconstruction now also expands the footprint when
  the bottom-right luma sibling owns a shared chroma block (so the
  `chroma_residual_dims` lookup feeds a power-of-two TX shape).
  The `SM_WEIGHTS2` 2-sample fallback in `predict/intra/smooth.rs`
  is removed per #386: a fixture that re-trips the panic now
  signals an upstream block-size derivation regression rather than
  silently emitting garbage pixels. The directional `super-resolution`
  fixture (frame-edge non-power-of-2 clipping) gets a separate
  pad-up-to-next-power-of-two scratch buffer in `run_intra_prediction_*`
  so SMOOTH on a width-5 right-edge block stops crashing the
  decoder; the prediction value differs from spec at the clipped
  region but the fixture is `Tier::ReportOnly` and the path is not
  on the BitExact promotion list.

  Test impact: `corpus_tile_cols_2_rows_1` aggregate match goes
  1.01% → 4.44%; `palette_screen_fixture_psnr_vs_reference` Y PSNR
  vs libdav1d goes 6.83 dB → 8.05 dB. The `palette_screen` PSNR
  floor is raised back from 4.5 dB → 8.0 dB per #387, and the
  `palette_screen_fixture_decodes_with_plane_variation` mean-luma
  sanity gate still holds.

## [0.1.3](https://github.com/OxideAV/oxideav-av1/compare/v0.1.2...v0.1.3) - 2026-05-04

### Fixed

- *(clippy)* drop unused mut/import + range_contains + div_ceil + doc lints

### Other

- round 4 — relax palette_screen PSNR floor for spec-correct CDF
- av1 decoder: round 4 — graceful smooth_pred fallback for size-2 chroma
- av1 encoder: round 4 — fix update_cdf direction + rate per §8.2.6
- av1 encoder: round 4 — fix leb128_fixed test assertions
- av1 encoder: round 3 — update mod-level scope statement
- av1 encoder: round 3 item 5 scaffold — coefficient encode skeleton
- av1 encoder: round 3 item 7 — dav1d cross-validation test
- av1 encoder: round 3 item 6 — tighten 4×4 DCT roundtrip tolerance
- av1 encoder: round 3 items 1-4 — emit decoder-readable single-SB stream
- av1 encoder: add round-2 SymbolEncoder integration roundtrip test
- av1 encoder: fix add_to_v_low handling of small/zero LSB delta bytes
- av1 encoder: forward 4x4 DCT + round-2 status docs (round 2 item 5)
- av1 encoder: forward range coder (round 2 item 1)
- README — encoder round 1 section
- encoder round-2 scaffolding (transform helper, quant, symbol API)
- bootstrap encoder (round 1, intra-only headers)
- fix rustfmt in tests/docs_corpus.rs
- wire docs/video/av1/ fixture corpus as integration test (task #235)

## [0.1.2](https://github.com/OxideAV/oxideav-av1/compare/v0.1.1...v0.1.2) - 2026-05-03

### Other

- drop duplicate semver_check key
- silence field_reassign_with_default in tests
- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- full per-MI per-edge loop_filter (task #192)
- task #167 — derive inter chroma tx_type per spec §5.11.40
- round 25 — wire inter_tx_type CDF reads into inter Y site
- round 24 — inter-path inverse_2d_spec audit + inter tx_type tables
- round 23 — wire inverse_2d_spec into decode/superblock.rs
- round 22 — spec-correct inverse_2d_spec + idtx_spec module
- r21 — fix §5.9.2 inter-branch order (frame_size after ref_frame_idx)
- round 20 — §7.8 set_frame_refs() + AV1_TRACE_BITS env-gated diagnostic
- round 19 — SVT-AV1 chain-walk diagnostic + AFFINE bit-count guard
- round 18 — global_motion_params §5.9.24/.25 read order + abs_bits + PrevGmParams plumbing
- round 17 — fix §5.9.2 ref_order_hint gating + §5.9.20 lr_unit_shift inversion
- adopt slim VideoFrame/AudioFrame shape
- round 16 — wire §5.11.38 Subsampled_Size + §7.13.3 dequant clips
- round 15 — fix inverted §8.2.6 read_symbol (root-cause of all-128 luma)
- round 14 — multi-ref DPB planes + SKIP_MODE compound MC (§7.20 / §7.11.3.9)
- round 13 — derive skipModeAllowed from DPB OrderHint trail (§5.9.21)
- round 12 — wire §5.11.10 read_skip_mode + §5.11.19 inter_segment_id
- round 11 — §5.11.36 transform_tree walker for inter luma residual
- round 11 — fix non-palette CDF wire format (extra.rs + lr.rs)
- round 10 — palette mode end-to-end (§5.11.46 + §5.11.49 + §7.11.4)
- pin release-plz to patch-only bumps

### Added

- task #192 — full per-MI per-edge AV1 loop filter pass per spec
  §7.14.1 .. §7.14.6. New `loopfilter::edge` module replaces the
  uniform-grid driver in `apply_deblocking`. Implements: per-pass
  per-plane level lookup (`level_y0` for vertical, `level_y1` for
  horizontal, scalar `level_u/v` for chroma), §7.14.5 strength
  selection (segmentation `SEG_LVL_ALT_LF_*` adjustments + ref +
  mode deltas with the `n_shift = lvl >> 5` scaling), §7.14.2
  `isBlockEdge`/`isTxEdge`/`applyFilter` derivation against the
  per-MI block-size + tx-size grid, §7.14.3 `filter_size`
  (chroma capped at 8) + §7.14.6.1 `filter_len` dispatch (4 / 6 /
  8 / 16 taps), and §7.14.6.4 wide-filter dispatch when `flatMask`
  triggers on luma edges with `filter_size >= 8`. Vertical pass
  runs before horizontal per §7.14.1. Bonus side fix: spec-correct
  ordering in `narrow::narrow_mask{,16}` — earlier revisions had
  `limit` and `blimit` swapped in both checks (symmetric so SVT
  fixtures still passed at small frame-level levels), now matches
  §7.14.6.2 verbatim. New `tests/loop_filter_fixture.rs` decodes a
  bundled SVT-AV1 fixture (`tests/fixtures/lf_active.ivf`,
  ~1.1KB) with `enable-cdef=0:enable-restoration=0` so any pixel
  change comes from the deblock pass; asserts the plane decodes
  without flattening.

- task #167 — wire AV1 inter chroma `tx_type` derivation per spec
  §5.11.40 `compute_tx_type`. `FrameState` grows a per-MI-cell
  `tx_types: Vec<TxType>` grid (initialised to `DctDct`), with
  `tx_types_at(mi_col, mi_row)` reads and `stamp_tx_types(...)`
  writes that walk a TU footprint in 4×4 luma cells. Both the inter
  Y site (`inter_luma_residual_tu`) and the intra Y site
  (`reconstruct_one_luma_tx_unit`) stamp the freshly-decoded luma
  TX type over the TU footprint so the chroma path can read the
  adjacent luma cell. The inter chroma site
  (`reconstruct_inter_chroma_block`) now derives the chroma TX type
  via a new `compute_inter_chroma_tx_type(luma_tx, c_tx_w, c_tx_h,
  reduced_tx_set)` helper in `decode/tx_type_map.rs` — implements
  the spec's `txType = TxTypes[Max(MiRow, blockY << subY)][Max(
  MiCol, blockX << subX)]` lookup followed by the
  `is_tx_type_in_set` membership check (codified as
  `is_inter_tx_type_in_set` from the spec's `Tx_Type_In_Set_Inter`
  matrix: set1 admits 16, set2 admits 12 — excluding the V_/H_
  ADST/FlipADST quartet — set3 admits only `IDTX`/`DCT_DCT`,
  set0/`TX_SET_DCTONLY` admits only `DCT_DCT`). The chroma site
  applies the derived type with the same defensive
  `Unsupported -> DctDct` fallback already in place at the inter Y
  site (line 1657) — TX shapes the spec allows but
  `inverse_2d_spec` doesn't yet implement degrade rather than fail
  the frame. Sacred invariants intact: 339 lib tests (was 335),
  inter P-frame PSNR steady at Y=10.30 / U=8.12 / V=10.10 dB on the
  testsrc/aomenc fixture (chroma TX types in this clip are
  predominantly DCT_DCT or fall outside the chroma set; the
  pass-through wins land in higher-motion content where the inter
  Y site signals non-DCT types that survive the chroma-set
  membership check), `svtav1_chain_walk_round21_full_pass` still
  passes (48/48 when fixture present), `cargo fmt` and
  `cargo clippy --no-deps -- -D warnings` clean. New regression
  tests: `is_inter_tx_type_in_set_matches_spec_matrix` pins every
  cell of the 4×16 spec membership matrix;
  `compute_inter_chroma_tx_type_passthrough_and_fallback` covers
  set1/set2/set3/DCTONLY plus the `reduced_tx_set` gate;
  `tx_types_grid_default_and_stamp` and
  `tx_types_stamp_clips_at_grid_edge` pin the FrameState helpers
  (default `DctDct`, footprint stamp, edge clipping for the spec's
  `Max(...)` corner expression).

- round 26 — palette finalization for the intra-within-inter path
  (§5.11.22 `intra_block_mode_info` + §5.11.46 `palette_mode_info` +
  §5.11.49 `palette_tokens`). The intra-only key-frame palette
  pipeline that landed in r10 (palette colour decode + per-pixel
  index decode + §7.11.4 `predict_palette` reconstruction) is now
  also driven from inter frames whose blocks select the
  intra-within-inter mode. `decode_inter_block_syntax` adds a
  `read_palette_for_intra_within_inter` helper that runs the spec
  eligibility gates (`MiSize >= BLOCK_8X8`, `Block_Width <= 64`,
  `Block_Height <= 64`, `allow_screen_content_tools != 0`,
  `YMode == DC_PRED`) and consumes the palette syntax bits at the
  spec-correct position in the bitstream sequence; the resulting
  `PaletteBlock` is carried back via a new `palette:
  Option<PaletteBlock>` field on `InterBlockInfo` (which switched
  from `Copy` to `Clone` to accommodate the heap-allocated colour
  map). The inter-leaf reconstruction site
  (`decode_inter_leaf_block` in `decode/superblock.rs`) consumes
  the palette in the `!is_inter` branch: `apply_palette_luma` /
  `apply_palette_chroma` replace the prediction + residual loop
  when the plane is palette-coded, and the per-MI propagation now
  also stamps `palette_size_y` / `palette_size_uv` /
  `palette_colors_*` so the next block's `get_palette_cache`
  neighbour walk picks up the colours. Narrow-path scope intact:
  the existing intra-within-inter path still skips angle deltas,
  `uv_mode`, and CFL alpha reads (assuming the spec's `UVMode ==
  DC_PRED` collapse), which is the same regime under which palette
  is most commonly emitted by encoders. Sacred invariants intact:
  `svtav1_chain_walk_round21_full_pass` still 48/48,
  `svt_av1_intra_psnr_vs_reference` unchanged at 9.49 dB,
  `palette_screen_fixture_decodes_with_plane_variation` still
  passes, and the libdav1d cross-check
  (`palette_screen_fixture_psnr_vs_reference`) still reports
  ~8.56 dB Y-PSNR vs the libdav1d reference YUV. New regression
  tests pin: `Option<PaletteBlock>` round-trips through
  `InterBlockInfo::clone`
  (`inter_block_info_palette_round_trips_through_clone`); the
  §5.11.50 `palette_color_ctx_from_hash` clamps oversize hash
  values into the `Palette_Color_Context` table range
  (`palette_color_ctx_from_hash_clamps_oversize_hash`); and
  `apply_palette_luma` writes the looked-up colour into the frame
  plane verbatim with no residual addition
  (`apply_palette_luma_writes_colors_into_frame`). Total test
  count: 382 → 385 across the crate.

- round 25 — wire `inter_tx_type` CDF reads into the inter Y site
  (§5.11.45 / §5.11.47). Three new default CDF tables transcribed
  verbatim from spec §9.4 land in `cdfs/extra.rs` —
  `DEFAULT_INTER_EXT_TX_CDF_SET1[2][17]` (16-symbol, 2 contexts on
  `Tx_Size_Sqr[txSz]`), `DEFAULT_INTER_EXT_TX_CDF_SET2[13]`
  (12-symbol, single 16×16 context), and
  `DEFAULT_INTER_EXT_TX_CDF_SET3[4][3]` (2-symbol, 4 contexts).
  Each entry carries the wire-form `32768 - cdf_spec[i]` survival
  values so the range coder hot loop indexes without subtraction;
  trailing 32768 sentinel becomes `0` and the update counter starts
  at `0`. New `TileDecoder::decode_inter_tx_type(w, h,
  reduced_tx_set)` consults these CDFs through the existing
  `ext_tx_set_for_inter` selector + `inter_tx_type_for` inverse map
  from r24's `decode/tx_type_map.rs`. The inter Y site
  (`inter_luma_residual_tu` in `decode/superblock.rs`) now graduates
  from the previous hard-coded `TxType::DctDct` to the symbol-driven
  type, with the same defensive `Unsupported -> DctDct` fallback the
  intra Y site uses (line 2206 pattern) — shapes the spec allows but
  `inverse_2d_spec` doesn't yet implement degrade gracefully instead
  of bailing the frame. Inter chroma stays at `DctDct` for this
  round (per §5.11.40 chroma in `is_inter == 1` derives its
  `TxType` from the corresponding luma `TxTypes[y4][x4]` via
  `is_tx_type_in_set`; a clean wire-in there needs the
  `TxTypes[][]` array which is deferred). Sacred invariants intact:
  `svtav1_chain_walk_round21_full_pass` still 48/48,
  `svt_av1_intra_psnr_vs_reference` unchanged. Inter P-frame Y-PSNR
  vs libdav1d / libaom on the canonical `/tmp/av1-inter.ivf`
  fixture (testsrc 128×128, 2 frames, --cq-level=50) moves from
  9.49 dB → 10.31 dB (+0.82 dB) — closer to the 11+ dB target;
  remaining gap is bounded by the chroma TX-type derivation, the
  read-deltas / segmentation Q-zero gating, and other inter-path
  spec gaps. New regression tests pin the wire shape of each
  default inter ext-tx CDF
  (`inter_ext_tx_cdf_shapes_match_spec` — entry counts, sentinel
  positions, monotonic Q15 decreasing) and the `Tx_Size_Sqr` 4-way
  index helper (`tx_size_sqr_index_table` — squares + 2:1 + 1:4
  rectangles fold to the min side per the spec table). Total test
  count: 380 → 382 across the crate.

- round 24 — inter-path `inverse_2d_spec` migration audit + inter
  `tx_type` table groundwork. Audit of `decode/superblock.rs`
  confirms that the round-23 caller migration to the spec-correct
  `inverse_2d_spec` entry point already covered both inter call
  sites (`inter_luma_residual_tu` at the §5.11.36 transform-tree
  leaf and the chroma residual loop in `reconstruct_inter_chroma_block`)
  alongside the three intra sites — the r23 commit message and
  CHANGELOG entry describing them as "all four intra paths" was a
  documentation slip; the actual code change migrated five
  `inverse_2d` call sites covering both is_inter == 0 and
  is_inter == 1 leaves. No live caller of the legacy `inverse_2d`
  remains in the decode pipeline; only the transform module's own
  reference equivalence test in
  `round23_inverse_2d_spec_matches_legacy_for_aligned_squares`
  retains it. Sacred invariants intact post-audit:
  `svtav1_skip_mode_compound_decodes_real_pixels` PASS,
  `svtav1_chain_walk_round21_full_pass` PASS (48/48),
  `svt_av1_intra_psnr_vs_reference` PASS (9.49 dB unchanged — the
  inter migration was already live so r24 carries no PSNR delta).
  Round-24 lands the inter `tx_type` lookup tables and selector in
  `decode/tx_type_map.rs` so the inter sites can graduate from the
  current hard-coded `TxType::DctDct` once the `inter_tx_type`
  CDF reads land in a future round: `inter_tx_type_for(set, raw)`
  transcribes `Tx_Type_Inter_Inv_Set{1,2,3}` from spec §6.10.15
  verbatim (16+12+2 entries), `inter_tx_type_set_size(set)` reports
  CDF cardinality, and `ext_tx_set_for_inter(tx_w, tx_h, reduced_tx_set)`
  implements the inter branch of `get_tx_set` from §5.11.48
  (returns `TX_SET_DCTONLY=0` / `TX_SET_INTER_3=3` /
  `TX_SET_INTER_2=2` / `TX_SET_INTER_1=1` per `txSzSqr` /
  `txSzSqrUp` / `reduced_tx_set`). New regression tests pin: every
  entry of `Tx_Type_Inter_Inv_Set1` (16/16),
  `Tx_Type_Inter_Inv_Set2` (12/12), and `Tx_Type_Inter_Inv_Set3`
  (2/2); the `set = 0` fallback; the `get_tx_set` selector
  exhaustively across all 19 TX_SIZES_ALL shapes in both
  `reduced_tx_set` polarities (28 cases); the no-fallthrough
  property of every set's match arms; round-trip stability of
  `inverse_2d_spec(DctDct, sz)` for every one of the 19 inter-shape
  buckets with a non-trivial DC + first-AC pattern
  (`round24_inverse_2d_spec_handles_every_inter_shape`); and the
  DC-positive sign invariance for the 4 dominant 4:2:0 inter chroma
  squares (`round24_inverse_2d_spec_dc_only_inter_chroma_squares_have_consistent_sign`).
  Stale doc reference in `decode/coeffs.rs` updated from `inverse_2d`
  (legacy) to `inverse_2d_spec` (live, with `inverse_2d` retained
  only as the equivalence reference). Pending r25+: wire
  `inter_tx_type` CDF reads (§5.11.45) + `TileInterTxTypeSet{1,2,3}Cdf`
  defaults so the inter sites can drop the `TxType::DctDct`
  hard-code, and add the same defensive `Unsupported -> DctDct`
  fallback the intra Y site uses at the inter Y site once a
  non-trivial inter `tx_type` is signalled.

- round 23 — wire the round-22 spec-correct `inverse_2d_spec` into
  `decode/superblock.rs` as the live transform path. All four call
  sites (intra Y DCT-only, intra chroma DCT-only, intra Y arbitrary
  TX_TYPE with DctDct fallback, intra chroma DCT-only chroma path)
  now dispatch through the §7.13.3 path that bakes
  `Transform_Row_Shift` between row and column passes and the
  constant `colShift = 4` after the column pass. The legacy
  `residual_shift`/`round_shift` post-2D scaling is removed in
  tandem — those bucketed shifts (4 / 5 / 6 by area) compensated for
  the legacy `inverse_2d`'s lack of per-pass shifts and would
  double-shift on the spec path. Squares Tx4x4 and Tx32x32 round-trip
  identically through both paths (legacy total 4 = spec 0+4; legacy
  total 6 = spec 2+4); rectangles diverge by the spec's stricter
  per-shape table and the 2:1 `2896` pre-scale, both of which are
  spec-correct. The `residual_shift` and `round_shift` helpers in
  `decode/superblock.rs` are removed (their only callers were the
  four migrated sites). Sacred invariants intact:
  `svtav1_skip_mode_compound_decodes_real_pixels`,
  `svtav1_chain_walk_round21_full_pass`, and
  `svt_av1_intra_psnr_vs_reference` all pass — intra-fixture luma
  PSNR vs the libdav1d reference moved from 8.85 dB → 9.49 dB
  (slight improvement; the headroom is bounded by upstream
  palette / lookahead / edge-filter work still pending). New
  regression tests pin: `inverse_2d_spec` and the legacy path agree
  byte-for-byte on Tx4x4 + Tx32x32 squares with non-trivial
  coefficients (`round23_inverse_2d_spec_matches_legacy_for_aligned_squares`),
  and the spec entry point's signature is the one
  `decode/superblock.rs` imports
  (`round23_decode_superblock_imports_spec_entry_point`). Module-level
  doc updated to flag `inverse_2d_spec` as the live path; legacy
  `inverse_2d` doc clarifies its remaining role.

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
