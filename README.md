# oxideav-av1

[![CI](https://github.com/OxideAV/oxideav-av1/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-av1/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-av1.svg)](https://crates.io/crates/oxideav-av1) [![docs.rs](https://docs.rs/oxideav-av1/badge.svg)](https://docs.rs/oxideav-av1) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust AV1 (AOMedia Video 1) codec — a clean-room implementation
built from the public AV1 Bitstream & Decoding Process Specification.
Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework.

## Status

Clean-room rebuild in progress. The bitstream-syntax and header layers
are broadly complete (OBU framing, sequence header, full
uncompressed-frame-header syntax tree, tile info), and — as of r390 —
**every stream in the independent conformance corpus (16 of 16)
decodes to pixels byte-identical to a third-party decoder's output**:
the full intra surface, KEY + P inter, multi-frame GOPs with
`show_existing_frame`, and 10/12-bit Professional-profile 4:2:2. On
top of the r387 inter driver, r390 lands the cross-frame session
state: §8.3.1 `load_cdfs` / §7.20 `save_cdfs` / §8.4
`frame_end_update_cdf` forwarding with the §6.8.21 counter reset,
§7.9 `motion_field_estimation` over §7.19/§7.20-stored motion fields,
§5.9.2 `load_previous()` (loop-filter delta + `PrevGmParams`
forwarding), §5.9.22 skip-mode (`SkipModeFrame[]` derivation +
walker threading), the §7.21 KEY-frame `show_existing_frame` reload,
and the 10/12-bit output surface. Three decode bugs fell out of the
corpus work: the missing §5.11.5 inter `YModes[]` grid-fill (whose
absence starved §7.10.2.8 `has_newmv` and desynchronised the
arithmetic decoder on NEWMV-adjacent blocks), forwarding CDF symbol
counts that §6.8.21 says restart at zero, and a double-subsampled
§7.11.3.5 chroma-warp clamp.

### Conformance-validated decode (r384 intra, r387 inter, r390 session state)

`decoder::decode_av1_spec(ivf_bytes) -> Vec<SpecFrame>` is the
spec-faithful frame driver: IVF + §7.5 OBU walk (including the combined
§5.10 `OBU_FRAME`), §5.9 header-derived state, per-tile §8.2.2/§8.3.1
symbol-decoder + CDF init (with the q-context coefficient-CDF slice),
the §5.11.2 `decode_tile` superblock loop (with the §5.11.57 `read_lr`
interleave and per-tile `begin_tile` resets), and the full §7.4
post-pass chain on mi-grid-padded planes — §7.14 deblock (gated on
nonzero luma filter levels), §7.15 CDEF, §7.16 superres upscaling of
both the CDEF output and the post-deblock frame, §7.17 loop restoration
(Wiener / self-guided / switchable), the §7.18.2 crop, and §7.18.3 film
grain. `tests/fixture_conformance.rs` pins 16 streams byte-exact against
independent-decoder output (fixture corpus under
`docs/video/av1/fixtures/`, used as opaque black-box tools): the full
intra feature surface — lossy quant at every coded TX size (including
the 64-wide compact-`tw` dequant layout), lossless WHT, palette (luma +
chroma, in-walk §5.11.35 `predict_palette`), CfL, filter-intra,
directional prediction including V/H-with-angle-delta and the
§7.11.2.9-12 edge filter + upsample pre-pass, monochrome, 128×128
superblocks, multi-superblock and multi-tile frames, film grain,
superres on a non-mi-aligned width — plus the inter surface: GLOBALMV
and NEWMV motion (mv-stack prediction, `drl_mode`, §5.11.32 mv
coding), var-tx trees with TX_SET_INTER_1/2/3 transform types,
SIMPLE / OBMC / LOCALWARP motion modes (§7.11.3.8 least-squares fit +
§7.11.3.5 warp filter on luma AND ≥8×8 chroma), compound references,
skip-mode blocks, primary-ref CDF forwarding across a 29-header GOP,
§7.9 temporal MV projection, `show_existing_frame` replays (including
the §7.21 KEY reload), and 10/12-bit 4:2:2 output (`yuv422p1{0,2}le`
little-endian packing).

The r384 conformance debugging also fixed five spec deviations that
encoder-mirror round-trips could never catch (both sides shared the
same deviation): the §5.11.39 `all_zero`-before-`transform_type` read
order (encoder writer reordered in lockstep), the §7.12.3
`Quant[i*tw+j]` compact layout for 64-wide transforms, the §7.4 rule
that zero luma filter levels skip deblocking entirely, the
§7.11.2.1 rule that V_PRED / H_PRED with a non-zero angle delta run the
full directional process, and the §5.11.49 palette-cache left gate at a
mid-frame tile's first column.

The intra-only decoder is now reachable through the runtime codec
registry: `register` installs an `oxideav_core::Decoder` factory for
codec id `av1` and claims the container identifiers an AV1 elementary
stream is carried under — the ISOBMFF sample entry `av01` / IVF FourCC
`AV01` and the Matroska / WebM Codec ID `V_AV1`. The wrapper bridges the
existing `decode_av1` driver onto the packet-to-frame trait surface; its
coverage equals that driver's (single-tile intra keyframes), so the
registered surface is partial but resolvable. The historical direct API
(`decode_av1` / `encode_av1`) is unchanged.

### What parses

- **OBU bytestream** (§5.2/§5.3) — low-overhead and length-delimited
  framing, `obu_type` / extension / `temporal_id` / `spatial_id` /
  `obu_size`, LEB128 sizes, per-unit payload slices.
- **Sequence header OBU** (§5.5) — `sequence_header_obu`,
  `color_config`, `timing_info`, `decoder_model_info`,
  `operating_parameters_info`, surfaced as a typed `SequenceHeader`.
- **Uncompressed frame header** (§5.9) — the full intra-path syntax
  tree: frame/render size + superres, tile info, quantization,
  segmentation, delta-Q / delta-LF, loop filter, CDEF, loop
  restoration, TX mode, global motion, skip-mode, and film-grain
  parameter blocks.

### What decodes / encodes (intra pixel pipeline)

`decode_av1(bytes) -> Vec<Frame>` and `encode_av1(pixels, width,
height) -> Vec<u8>` (IVF v0 output) cover a constrained intra-only
profile:

- 4:2:0 8-bit YUV or 8-bit monochrome.
- Intra-only key frames, single tile per frame.
- The 13-mode `INTRA_MODES` luma set plus chroma-from-luma
  (`UV_CFL_PRED`) on the chroma path.
- Palette-coded intra blocks: the §5.11.46 palette-colour reads,
  §5.11.49 `palette_tokens` colour-index map, §7.11.4 per-TU
  `predict_palette` leaf, and the §5.11.35 per-block walker bridge
  (`reconstruct_palette_block_into_curr_frame`) that drives the leaf
  across a block's transform-block grid into `CurrFrame[plane]`.
- Lossless arm (`base_q_idx == 0`, inverse WHT, bit-exact
  encode/decode round-trip) and a lossy inverse-DCT arm
  (`base_q_idx > 0`, encoder/decoder self-consistency).
- In-loop / post passes (loop filter, CDEF, loop restoration) are
  present as modules; on the lossless intra dyn parameter set they are
  no-ops (`loop_filter_level = 0`, `enable_cdef = 0`,
  `enable_restoration = 0`).
- §7.16 **superres** and §7.18.3 **film-grain synthesis** are wired into
  the public dynamic-extent decode path (`decode_frame_dyn` /
  `decode_frame_dyn_y`), running in §7.4 decode order (superres before
  film grain). Both gate on the parsed frame header: `use_superres == 0`
  / `apply_grain == 0` (every encoder-produced fixture) make the passes
  verbatim no-ops, preserving byte-for-byte parity; when active, superres
  upscales each plane horizontally to `upscaled_width` and film grain
  blends §7.18.3 noise into the (post-superres) planes in place.

The inter-prediction reconstruction layer covers the §7.11.3.1 single-
reference translational (SIMPLE), compound (AVERAGE / DISTANCE / WEDGE /
DIFFWTD), and inter-intra arms, plus the §7.11.3.5 **warped-motion**
(LOCALWARP `useWarp == 1` / GLOBAL_GLOBALMV `useWarp == 2`) arm —
`reconstruct_inter_block_warp` and its `PartitionWalker` bridge
(`reconstruct_inter_block_warp_into_curr_frame`) drive `block_warp` into
`CurrFrame[plane]`, and the §5.11.33 frame walk dispatches a decoded
`motion_mode == WARPED_CAUSAL` leaf to the warp path (via the opt-in
`InterModeInfoGrid.warp` context). §7.11.3.9-10 **OBMC** (overlapped
block motion compensation) now also has a reconstruction-surface entry:
`reconstruct_inter_block_obmc` and its `PartitionWalker` bridge
(`reconstruct_inter_block_obmc_into_curr_frame`) drive a decoded
`motion_mode == OBMC` leaf — the block's own §7.11.3.1 prediction plus
the §7.11.3.9 above/left neighbour walk's §7.11.3.10 overlap-blend
contributions — into `CurrFrame[plane]` from a caller-resolved
`ObmcParams` neighbour bundle, the OBMC counterpart of the per-block warp
bridge. As of r378 the §5.11.33 frame walk **dispatches OBMC leaves
automatically**, as it already does for warp: `InterModeInfoGrid` carries
an opt-in `obmc` context (`GridObmcContext`), and `reconstruct_inter_frame`'s
single-reference arm routes a leaf whose per-cell `motion_modes` ordinal is
`OBMC` through a frame-walk `obmc_dispatch_leaf` helper. That helper runs
the §7.11.3.9 outer `(x4, y4, step4, nLimit)` neighbour scan against the
grid's own `mi_sizes` / `ref_frames` / `mvs` slices (above candidate
`(MiRow - 1, x4 | 1)`, left candidate `(y4 | 1, MiCol - 1)`, keeping
`RefFrames[cand][0] > INTRA_FRAME` candidates), resolves each kept
neighbour's MV + per-plane reference buffer into an `ObmcNeighbour`, and
drives `reconstruct_inter_block_obmc` per plane — so a real OBMC leaf
decoded from a bitstream reconstructs its overlap blend end-to-end. The
walker bridge (`reconstruct_inter_frame_into_curr_frame`) threads the
`obmc` context from the walker's persisted `motion_modes` grid plus per-cell
`AvailU` / `AvailL` derived from the tile geometry.

The **encoder** now has a single-reference (P-frame) inter pixel pipeline
(`encoder::inter_predict`). The intra dyn driver builds a leaf's
reconstruction as `recon = pred + Q^-1(Q(T(input - pred)))` where `pred`
is the §7.11.2 intra prediction; the inter arm differs in exactly one
place — `pred` is the §7.11.3.1 motion-compensated reference. The
encode-side primitives supply that one difference and share every
downstream stage verbatim: `predict_inter_block_single` takes the
prediction straight from the **decoder's** `reconstruct_inter_block`, so
the prediction the encoder codes its residual against is bit-identical to
what the decoder reproduces from the same `(RefFrame[0], Mv)` — there is
no second prediction implementation. `encode_inter_block_residual_4x4` is
the §5.11.39 TX_4X4 residual leaf (forward transform + quantize on the
lossless-WHT / lossy-DCT_DCT arm, the matching dequant + inverse, and the
`recon = Clip1(pred + inv_residual)` stitch). Motion estimation is a
deterministic SAD search: `estimate_motion_4x4_full_search` over an
integer-pel window, then `estimate_motion_4x4_subpel` refines through the
half/quarter/eighth-pel MV grid the interpolation filter supports
(steepest-descent diamond, strict-improvement acceptance). Frame-scope
entries `encode_inter_frame_y` / `encode_inter_frame_y_opt` (luma) and
`encode_inter_frame_yuv` (4:2:0; each chroma 4×4 reuses the collocated
luma MV `cand = (mi >> sub) << sub` through the chroma arm so the
§7.11.3.2 chroma MV scaling matches the decoder) produce the per-cell
motion field + running reconstruction. The round-trip is verified
end-to-end against the decoder: feeding the encoder's motion field into
the **independent** `reconstruct_inter_frame` frame walk reproduces the
exact MC prediction the encoder coded against (integer-pel, sub-pel, and
3-plane chroma), and the lossless arm reconstructs every plane
byte-for-byte.

The spec-faithful §5.11 syntax walker (`PartitionWalker`, separate from
the encoder-mirror pixel driver above) now reconstructs **intra pixels**
end-to-end from a real bitstream: every intra transform block runs the
§7.11.2.1 general intra prediction (`predict_intra_into_curr_frame` —
DC / V / H / PAETH / SMOOTH{,_V,_H} / directional, deriving the
`AboveRow[]` / `LeftCol[]` neighbours from the already-reconstructed
`CurrFrame[plane]`) ahead of the §5.11.39 coefficient read + §7.12.3
dequant + §7.13 inverse transform + step-3 residual merge, realising the
§5.11.35 `reconstruct()` body `CurrFrame = Clip1(pred + residual)`. The
new §5.11.2 `decode_tile_syntax` superblock loop drives this across a
whole tile, so after the walk the per-plane `curr_frame` buffers hold
the reconstructed intra tile (pre loop-filter / CDEF / loop-restoration
post passes). As of r363 the **directional** modes additionally run the
§7.11.2.4 step-4 edge pre-pass — the §7.11.2.7 filter corner, the
§7.11.2.9/§7.11.2.12 intra edge filter, and the §7.11.2.10/§7.11.2.11
intra edge upsample — applied to `AboveRow[]` / `LeftCol[]` before the
directional kernel projects them, gated on the frame's
`enable_intra_edge_filter` and the §7.11.2.8 `get_filter_type`
neighbour smooth-mode check. Both planes are covered: the luma check
reads the §6.10.4 `YModes[]` grid; the chroma check reads the §5.11.22
`UVModes[]` grid (now stamped per-block) at the §7.11.2.8 sub-sampled
neighbour coordinates. As of r367 the **chroma-from-luma (CfL)** AC
contribution is also wired: a `UV_CFL_PRED` chroma TU writes the §7.11.2
`DC_PRED` base, then `predict_chroma_from_luma_into_curr_frame` (§7.11.5)
layers the reconstructed-luma high frequencies on top — subsampling
`CurrFrame[0]` into `L[i][j]` with 3 fractional bits, deriving `lumaAvg`,
and rewriting each sample as `Clip1(dc + Round2Signed(CflAlpha{U,V} *
(L - lumaAvg), 6))`, clamped to the §5.11.35 `MaxLumaW` / `MaxLumaH`
per-luma-TU extent (now tracked on the walker). The §5.11.45-decoded
signed alphas thread onto `ResidualContext`, so CfL blocks reconstruct
their full DC + luma-AC prediction rather than DC-only. Also as of r367
the §7.11.2.3 **recursive intra (filter-intra)** luma arm is wired: a
`use_filter_intra == 1` block routes its luma plane through
`predict_intra_recursive` (the §3 `Intra_Filter_Taps` 7-tap kernel + the
`Round2Signed(.., INTRA_FILTER_SCALE_BITS)` per-`4×2`-sub-block walk) as
the §7.11.2.1 first dispatch arm, reusing the head-extended edge
buffers; such blocks now stay on the intra reconstruction path rather
than being skipped. IntraBC and the lossy-quant post-pass chain remain
follow-ups before this path produces validated bit-exact keyframe
pixels.

The §5.11 walker now also reconstructs **inter pixels** at frame scope:
the §5.11.18 → §5.11.23 → §5.11.31 inter-syntax cascade stamps each
single-reference leaf's `IsInters[]` / `RefFrames[]` / `Mvs[]` /
`InterpFilters[]` / `MiSizes[]` grids during the syntax walk, and the new
`reconstruct_inter_frame_into_curr_frame` `PartitionWalker` bridge reads
those grids back out and drives every single-reference translational
(SIMPLE, `RefFrame[1] == NONE`) leaf through the shared
`reconstruct_inter_frame` walk, stitching each leaf's §7.11.3.1
motion-compensated (8-tap sub-pel) prediction into `CurrFrame[plane]`
against a caller-supplied §7.11.3.3 reference-frame store. This closes
the §5.11.33 frame walk on the single-ref path — a real single-reference
inter leaf decoded from a bitstream (the seg-globalmv `GLOBALMV` arm)
reconstructs to validated pixels end-to-end, and multi-leaf frames with
distinct per-leaf sub-pel MVs reconstruct leaf-by-leaf matching the
per-block driver. As of r359 the §5.11.23 inter cascade also stamps the
§5.11.29 / §5.11.28 / §5.11.27 **side-data grids** (`compound_types`,
`compound_wedge_{indices,signs}`, `compound_mask_types`,
`interintra_modes`, `wedge_interintras`, `interintra_wedge_indices`,
`motion_modes`) over each leaf's `bh4 × bw4` footprint, and the frame
bridge feeds them into the `InterModeInfoGrid` — so the frame walk now
dispatches **compound** (AVERAGE / DISTANCE / WEDGE / DIFFWTD) and
**inter-intra** leaves automatically through their §7.11.3 combine arms,
not translationally. The COMPOUND_DISTANCE (`enable_jnt_comp`) arm reads
its §7.11.3.15 order-hint context through the new
`reconstruct_inter_frame_into_curr_frame_with_order_hints` entry (the
no-hint entry delegates with the identity-zero context, correct for
frames with no distance-weighted compound leaves). Warped-causal leaves
remain on the opt-in `InterModeInfoGrid.warp` per-block context; threading
the LOCALWARP fit grid into the frame walk plus reference-frame buffer
management across a GOP remain the follow-ups toward a full inter AV1
frame.

The §5.11 walker also drives the **in-loop filter chain** at frame
scope, in the §7.4 decode order, straight from its persisted decode
grids — no separate filter-state mirror. `loop_filter_frame_from_grid`
(§7.14 deblock) wires the per-mi `Skips[]` / `RefFrames[][][0]` /
`YModes[]` / `SegmentIds[]` / `TxSizes[]` / `InterTxSizes[]` /
`MiSizes[]` grids into the §7.14 edge driver, reconstructing the
§7.14.2 `LoopfilterTxSizes` lookup on the fly (per-mi luma transform
for plane 0, the §5.11.37 `get_tx_size` chroma mapping for planes 1/2);
`cdef_frame_from_idx` (§7.15) and `loop_restore_frame_from_grid`
(§7.17) follow on the `cdef_idx[]` / §5.11.58 unit grids. An
integration test composes all three over one reconstructed
`CurrFrame[plane]` in order (deblock → CDEF → loop-restoration),
verifying the buffer plumbing and the identity case on a flat field.
As of r378 the §7.14.4 `DeltaLFs` term is bridged for **both**
`delta_lf_present` cases: the walker persists a per-mi `DeltaLFs[][][]`
grid (`delta_lfs`), stamped from the §5.11.13 accumulator over each
decoded block's footprint at `decode_delta_lf` (and in the encoder-mirror
`stamp_encoder_block_syntax`). `loop_filter_frame_from_grid` reads it via
`delta_lf_at` with the §7.14.4 `delta_lf_multi` slot indexing, so the
`delta_lf_present == 1` path now deblocks with the correct per-mi strength
rather than refusing.

The public `encode_av1` entry takes the constrained
`[8, 64]`-per-axis lossless case; wider extents, lossy quant, and
monochrome are reachable through the crate-public `encoder::*` driver
functions. Streams outside the supported scope return a typed `Error`
(commonly `Error::PartitionWalkOutOfRange`).

### Not yet supported

- Inter FRAMES through the spec frame driver: reference-frame buffer
  management across a GOP (§7.20/§7.21), the §5.11.18 inter cascade
  driven from real P-frame headers (`InterFrameContext` construction,
  temporal MVs), and `show_existing_frame` — the 3 remaining corpus
  streams. (The §5.11.33 reconstruction arms themselves — single-ref,
  compound, inter-intra, OBMC, warp — are wired at the walker level.)
- Frame-walk reconstruction of warped-causal inter leaves stays on the
  opt-in per-block warp context.
- 10/12-bit and 4:2:2 / 4:4:4 pixel output from the spec driver (the
  walker threads the subsampling; only 8-bit is surfaced).
- Quantizer-matrix (`using_qmatrix == 1`) streams in the spec driver.
- The runtime-registry wrapper still bridges the constrained
  encoder-mirror `decode_av1`, not the spec driver.

## Module layout

`obu`, `sequence_header`, `frame_header`, `tile_info`,
`uncompressed_header_tail`, `symbol_decoder`, `cdf`, `scan`,
`transform`, `qmatrix`, `superres`, `loop_filter`, `loop_restoration`,
`cdef`, `film_grain`, `inter_pred`, and the `decoder` / `encoder`
pipelines.

## Fuzzing

`fuzz/` holds three `cargo fuzz` libFuzzer targets, each driving only
this crate's public Rust API (no external decoder / oracle linked):

- `decode` — attacker bytes through `decode_av1` (IVF → OBU walk →
  headers → tile / partition / reconstruction).
- `obu` — the framing layer in isolation (`parse_leb128`, `parse_obu`,
  `ObuIter`, `parse_sequence_header`).
- `roundtrip` — derives dimensions from input bytes, encodes a YUV
  4:2:0 blob via `encode_av1`, then re-decodes the IVF output.

Run with `cargo +nightly fuzz run decode` from `fuzz/`.

## Clean-room policy

All syntax tables and decoding logic are written from the public AV1
Bitstream & Decoding Process Specification (AOMedia), staged under
`docs/video/av1/`. No third-party AV1 codec source is consulted;
`aomenc` / external decoders are used only as black-box CLI tools to
generate test fixtures.

## License

MIT.
