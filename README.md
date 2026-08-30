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

r394 grows the corpus to **26 byte-exact streams** and closes four
subsystems: (1) §7.12.3 **quantizer-matrix** dequantization — the
§5.9.2 `SegQMLevel[ plane ][ segmentId ]` derivation feeds the
long-landed §9.5.3 table arm on real streams (intra + inter GOPs at
matrix levels 0/1/4); (2) **segmentation-enabled inter frames** — the
§5.11.14 `seg_feature_active` gates are derived per block at the
block's own `segment_id` from the FeatureEnabled/FeatureData tables,
§5.11.19 `predictedSegmentId` is the real §5.11.21 `get_segment_id()`
over `PrevSegmentIds` (§7.20 `SavedSegmentIds` per slot + §5.9.2
`load_previous_segment_ids()`), and §5.9.14 `segmentation_update_data
== 0` reloads the primary reference's saved feature tables
(cyclic-refresh GOPs incl. a `segmentation_temporal_update` frame
decode byte-exact); (3) the §8.3.2 `compound_idx` **distance context**
is per block (`fwd == bck` over the block's own reference pair —
`InterFrameContext` carries `order_hints`, retiring the frame-scope
`dist_equal` bool); (4) four §7.11.3 fixes caught against
highest-effort encoder streams — the frame walk's swapped
per-direction interpolation-filter slots (§7.11.3.4 reads slot 1
horizontally, slot 0 vertically), candidate-cell filters on the
§5.11.33 sub-8×8 chroma stitch and §7.11.3.9 OBMC overlap bands,
`ObmcNeighbour::axis_pos4` (a skipped intra candidate advances the
§7.11.3.9 walk without producing an overlap, so band positions are
not re-derivable from the qualifying list), and clipped `CurrFrame`
stitches for §5.11.4 bottom/right-edge overhanging inter blocks.

r405 lands scaled-reference motion compensation (references of a
DIFFERENT resolution through the §7.11.3.3 scaling process, luma-unit
dimension contract), §7.11.3.1 intra-block-copy prediction, the
§5.11.2/§5.11.7 `ReadDeltas` delta-q lifecycle, and the SIMPLE-GLOBALMV
global-warp arm — 32 streams pinned. r408 closes three
spec-conformance root causes found against textured (mandelbrot /
testsrc) GOP sweeps: (1) the §7.10.2.12 extra-search single-pred tail —
`RefStackMv[ idx ][ 0 ] = GlobalMvs[ 0 ]` for `idx = NumMvFound..2`
without incrementing `NumMvFound` — was omitted entirely, so every
NEARESTMV/NEARMV block with an empty ref-MV stack on a global-motion
frame (top-edge blocks of zooming content) predicted a zero MV instead
of the warp-projected global MV; (2) §7.11.3.5 block-warp rounding uses
the §3 PLAIN `Round2` for `offs` / `intermediate` / `pred` — the
previous `Round2Signed` picked the adjacent warp-filter phase whenever
the shear walked `sx` negative, leaving isolated ±1 sample diffs on
compound GLOBAL_GLOBALMV blocks that propagated through the reference
chain; (3) §5.11.27 `is_scaled( refFrame )` divides by the CODED
`FrameWidth`/`FrameHeight` per the spec body, so a superres inter
frame's references are correctly "scaled" even though every upscaled
extent matches — the upscaled-vs-upscaled shortcut desynchronised the
arithmetic decoder (`motion_mode` read where the encoder wrote
`use_obmc`) on the first superres inter frame. With all three fixed,
full-superres GOPs (every frame coded at denominator 12 with loop
restoration at the §7.17 upscaled extent), resize-mode GOPs, and
default alt-ref-pyramid GOPs over textured content decode byte-exact —
a 54-config black-box encoder sweep (superres fixed/random, resize
fixed/random, global-motion on/off, order-hint off, cq 0-50, cpu-used
1-6, 10/12-bit, 4:4:4 / 4:2:2 / monochrome, screen content + intrabc,
128×128 superblocks, 2×2 tiles, error-resilient, S-frames, film grain,
arnr, three synthetic sources) passes with zero mismatches. The sweep
uncovered and r408 fixed four more root causes: §5.11.2
`clear_above_context()` at every tile entry (multi-tile-ROW frames
desynced their second tile row's coefficient contexts), the
§7.11.3.1 `useWarp = 2` arm on the INTER HALF of inter-intra blends
(GLOBALMV interintra leaves translated where the spec warps), §7.20
film-grain forwarding (`save_grain_params` / the §5.9.30
`update_grain == 0` predicted load / grain on `show_existing_frame`
outputs), and the §7.11.5 CfL luma TU-overhang store (spec
`CurrFrame[ 0 ]` extends past the mi grid; the `MaxLumaW` clamp reads
it). 39 streams pinned.

### Conformance-validated decode (r384 intra, r387 inter, r390 session state, r394 QM / segmentation / edge cases)

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
grain. `tests/fixture_conformance.rs` pins 57 streams byte-exact against
independent-decoder output (the 16-stream corpus staged under
`docs/video/av1/fixtures/` plus 10 r394 validator-produced streams —
QM intra/inter, dual-filter + OBMC, jnt-comp pyramids, cyclic-refresh
segmentation, a segmentation+QM+jnt composition, bottom-edge
overhang; encoder and decoder used as opaque black-box tools): the full
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

The **full inter decoder** is reachable through the runtime codec
registry (r394): `register` installs an `oxideav_core::Decoder` factory
for codec id `av1` and claims the container identifiers an AV1
elementary stream is carried under — the ISOBMFF sample entry `av01` /
IVF FourCC `AV01` and the Matroska / WebM Codec ID `V_AV1`. The wrapper
bridges `decoder::SpecDecodeSession`, which owns the cross-packet
session state (§7.20 reference store, cached sequence header, per-slot
CDF / motion-field / segment-id state), and accepts BOTH packet
framings: a whole IVF buffer (`DKIF` magic) or one §7.5 temporal unit
per packet (the Matroska / ISOBMFF sample framing) — a KEY + INTER GOP
split one-TU-per-packet decodes byte-identical to the same bytes in one
buffer. As of r409 the historical direct API reaches **full parity
with the spec driver**: `decode_av1` tries the encoder-mirror path
first (this crate's own constrained non-conformant intra streams keep
their bit-exact round-trip and historical `Frame` shapes), then falls
back to `decoder::decode_av1_spec` for everything else, surfacing each
shown frame as `Frame::Spec(SpecFrame)`. Per-fixture parity assertions
pin public-API output == spec-driver output across the whole 44-stream
conformance corpus.

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

(HISTORICAL until r428 — the encoder-mirror surface described in this
section's first paragraph was retired in r428: the mirror emit arms,
`decode_av1`'s mirror-acceptance arm, and the historical `Frame`
variants are gone; `decode_av1` rides the spec-faithful driver
exclusively and `encode_av1` has been conformance-grade since r409.
The reconstruction-layer notes below describe decoder modules that
remain fully live.) The retired constrained intra-only profile
covered:

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

The public `encode_av1` entry is, as of r409, the conformance-grade
KEY-frame encoder (`[8, 4096]`-per-axis lossless as of r410; see the
"Conformance-grade encoding" section above). Lossy quant is on
`encoder::encode_key_frame_yuv420_with_q`; monochrome and the
historical mirror drivers stay on the crate-public `encoder::*`
entries. Streams outside the supported scope return a typed `Error`
(commonly `Error::PartitionWalkOutOfRange`).

### Conformance-grade encoding (r409, generalised r410; every §6.4.1 format pairing r427)

`encoder::encode_key_frame_yuv420{,_with_q}` is the
**conformance-grade** encode path: it emits real §5.11 keyframe syntax
through the spec-faithful write side (§5.11.7 `intra_frame_mode_info`
with the neighbour-CDF `intra_frame_y_mode`, §5.11.22 `uv_mode` +
§5.11.45 CFL alphas, §5.11.34 per-TU residual with live §8.3.2
contexts), assembled as IVF → TD + SH + the combined §5.10 `OBU_FRAME`.
r427: the general-format siblings `encoder::encode_key_frame_yuv{,_with_q}`,
`encode_gop_yuv{,_with_q}`, `encode_pyramid_gop_yuv_with_q` and
`encode_adaptive_gop_yuv_with_q` take a `YuvFrame` (`u16` planes +
`bit_depth` + `ChromaFormat`) at **every §6.4.1 (bit depth, chroma
format) pairing** — 8/10/12-bit × 4:2:0 / 4:2:2 / 4:4:4 / monochrome —
with per-pairing §6.4.1 `seq_profile` election and §5.5.2
`color_config` synthesis; the whole encoder pixel pipeline (recon,
reference stores, motion search, CfL, quantiser rows, λ) runs at the
stream depth, the §5.11.38 4:2:2 partition-admissibility rule gates
the RD ladders, and the historical 8-bit 4:2:0 entries route through
the same core byte-identically. 12/12 pairings round-trip KEY + inter
through the in-tree spec driver; the 24-stream matrix (one KEY + one
GOP per pairing) decodes byte-identical in three independent
black-box reference decoders (72/72), and the 22 non-8-bit-4:2:0
streams are pinned in the conformance corpus.
Scope (r410): dims multiples of 8 in [8, 4096] per axis
(multi-superblock beyond 64), **full square partition-tree RD search**
— every in-frame node from BLOCK_64X64 down to BLOCK_8X8 trial-encoded
leaf-vs-split with region/state snapshot-restore (frame-edge nodes take
the §5.11.4 forced-split arms) — **all 13 §6.10.x intra modes** on both
pickers (the directional D-modes run §7.11.2.4 against §7.11.2.1
neighbours built with the real `haveAboveRight`/`haveBelowLeft`
availability off an encoder-side §6.10.3 `BlockDecoded[]` mirror, plus
a full §5.11.42/§5.11.43 `-3..=3` angle-delta search), chroma CFL over
an (αU, αV) grid at any TU size (general §7.11.5 kernel with the
`MaxLumaW/H` clamps; the §8.3.2 lossless-arm `cfl_allowed` gate is
honoured), **TX_MODE_SELECT** on the lossy arm — each leaf's luma
TU grid RD-searched down the §5.11.15 `Split_Tx_Size` ladder from
`Max_Tx_Size_Rect` (TX_4X4…TX_64X64, the 64-wide sizes emitting the
§7.12.3 compact-`tw` coefficient layout; chroma rides §5.11.38
`get_tx_size`, TX_4X4…TX_32X32), a **§5.11.47 per-TU luma
transform-type RD search** over the full §5.11.48 intra sets
(ADST/IDTX/V_DCT/H_DCT arms live), and **§5.11.24 filter-intra** (the
five §7.11.2.3 recursive modes on eligible ≤32×32 blocks). Lossless WHT arm (`q = 0`: decode ==
input bit-exact) and lossy DCT arm at any `base_q_idx` in 1..=255
(decode == encoder reconstruction bit-exact). Validated four ways: the
in-tree spec driver and THREE independent reference decoders (run as
black-box binaries) all produce byte-identical output on a 310-stream
matrix (12 geometries incl. 512×8 / 8×512 extremes and 1280×720 × q ∈
{0, 20, 50, 100, 160, 255} × gradient / noise / mixed /
diagonal-stripe / sharp-stripe content; 1080p and 4K spot-validated);
five self-encoded streams are pinned in the conformance corpus (44
total). Encoder-side conformance root causes found across the two
rounds: §5.3.4 `trailing_bits` placed bit-precisely by the OBU body
writers, the §8.2.4 arithmetic-coder termination
(`SymbolWriter::finish` lands the trailing one-bit exactly at
`trailingBitPosition`), and the §8.3.2 lossless-arm `cfl_allowed`
derivation (subsampled chroma residual must be 4×4 — the lossy
`Max(w,h) <= 32` arm does not apply).

### Conformance-grade inter P-frame GOPs (r411)

`encoder::encode_gop_yuv420{,_with_q}` extends the keyframe driver
into a **conformant KEY + P GOP encoder**: each INTER P-frame predicts
from the previous frame's reconstruction (single reference LAST_FRAME,
every §7.20 slot refreshed per frame) through the REAL §5.11.18
`inter_frame_mode_info()` syntax — the new r411 write arm of
`write_partition_tree_syntax` (§5.11.18 prologue with mirror-derived
§8.3.2 contexts, the §5.11.25 reference cascade, §7.10.2
`find_mv_stack` against the write mirror, the §5.11.24 single-pred
mode cascade + drl loop + §5.11.31 MV write, and the §5.11.22
intra-in-inter composite). P-frame headers ride §5.9.2
`error_resilient_mode` (`PRIMARY_REF_NONE`, per-frame default CDFs),
identity §5.9.24 global motion, `EIGHTTAP`, quarter-pel MVs and no
order hints. The RD search (BLOCK_64X64 down to a BLOCK_8X8 P-frame
leaf floor) trials an INTER leaf — integer motion search plus
half/quarter-pel refinement scored through the decoder's OWN §7.11.3
leaf driver, coding `NEWMV` or zero-vector `GLOBALMV` — against a
§5.11.22 INTRA leaf and the recursive split; inter leaves RD-select a
uniform §5.11.17 `txfm_split` depth (TUs coded in §5.11.36
transform-tree quadtree order) and run the §5.11.47 transform-type
search over the full §5.11.48 INTER sets (all 16 types at 4×4/8×8,
FLIPADST family included via the §7.12.3 step-3 destination remap)
with the §5.11.40 chroma inheritance; `skip = 1` on pred-exact leaves.
Validated four ways: the spec driver and THREE independent reference
decoders decode a 45-config GOP sweep (5 geometries × q ∈ {0, 30, 50,
100, 160, 255} × moving / static / content-cut / noise / half-pel
content + an 8-frame P-chain) byte-identical to the encoder's
per-frame reconstruction — lossless GOPs equal the input exactly.
Three self-encoded GOP streams are pinned in the conformance corpus
(47 total).

### Inter encoder: modes, filters, rect partitions, compound (r412)

r412 works the r411 follow-up ladder to exhaustion. (1)
**NEARESTMV / NEARMV mode selection** through a snapshotable
driver-side §7.10.2 MV-prediction mirror: the RD search owns a
`PartitionWalker` twin of the write-pass mirror (committed leaves
stamped, trials rolled back via a rect snapshot of every stamped
grid), so each leaf trials the full §5.11.24 candidate set — NEWMV at
the searched vector with the `drl_mode` index minimising the §5.11.32
difference bits, NEARESTMV / drl-reachable NEARMV slots straight from
the stack (no MV bits), GLOBALMV at the §7.10.2.1 derivation. (2)
**SWITCHABLE interpolation filters**: `is_filter_switchable = 1`
headers, the per-leaf §5.11.x `interp_filter` S() against the §8.3.2
neighbour ctx, and a per-leaf EIGHTTAP / SMOOTH / SHARP distortion
search through the decoder's own §7.11.3.4 kernel. (3) **HORZ / VERT
rectangular partitions**: `SyntaxNode::{Horz,Vert}` write dispatch +
the whole inter leaf pipeline generalised to rectangular blocks (rect
`Max_Tx_Size_Rect` transforms with the SPLIT-aware §5.11.36/§5.11.17
recursion — 2 children per rect split). (4) **Two-slot reference
rotation + per-block LAST/GOLDEN selection**: frame `k` refreshes
§7.20 slot `(k-1) & 1` with explicit `ref_frame_idx[]`, and the
candidate ladder runs per reference (a flash GOP provably selects
GOLDEN). (5) **COMPOUND_AVERAGE two-reference prediction**:
`reference_select = 1`, the §5.11.25 unidirectional { LAST, GOLDEN }
cascade, compound modes NEAREST_NEARESTMV / NEAR_NEARMV /
GLOBAL_GLOBALMV / NEW_NEWMV with both §5.11.31 MV lists
§5.11.26-checked, and the bit-silent §5.11.29 COMPOUND_AVERAGE
derivation. Validated four ways per feature: dedicated
selection-proving unit tests, the decode-walker syntax round trips,
a 66-config black-box sweep (moving / static / cut / noise / band /
flash / blend content, 5 geometries, q 0-255) byte-exact in THREE
independent reference decoders, and three more self-encoded streams
pinned in the conformance corpus (50 total).

### Inter encoder: order hints, skip mode, segmentation, EXT partitions, temporal MVs (r413)

r413 works the r412 follow-up ladder further down. (1) **Order
hints**: every encoded sequence header carries `enable_order_hint`
(`OrderHintBits = 7`); the §5.9.2 error-resilient `ref_order_hint[]`
block round-trips the TRUE per-slot stored hints through the new
`FrameHeader::ref_order_hints`. (2) **Skip-mode P-frames**: the
§5.9.22 `skip_mode_params()` write twin derives `skipModeAllowed`
from real reference state (also fixing a latent phantom-bit desync in
the pre-r413 writer), and every >= 8×8 inter leaf RD-trials the
§5.11.10 `skip_mode = 1` pure-derivation block — ONE S() coding a
compound NEAREST_NEARESTMV over `SkipModeFrame[]` with no residual
(static content provably selects it). (3) **SEG_LVL_ALT_Q
segmentation** (`encode_gop_yuv420_with_q_seg`): §5.9.14 feature
tables per P-frame header, the §5.11.19/§5.11.20 spatial segment map
with the bit-silent skip-leaf `pred` inheritance, and per-segment
residual quantisation through a deterministic luma-activity policy.
(4) **EXT-alphabet partitions**: `SyntaxNode` + write dispatch +
RD trials for HORZ_A / HORZ_B / VERT_A / VERT_B T-shapes and
HORZ_4 / VERT_4 four-strip shapes (tri-motion content provably
selects a T-shape). (5) **`use_ref_frame_mvs = 1` P-frames**: the
§7.9 motion-field estimation moves into a shared core
(`inter_pred::motion_field_estimation_core`) the decode driver and
the encoder's write mirror both run — the encoder keeps its own §7.20
motion-field store (§7.19-filtered committed mirror grids per
rotation slot) so the §7.10.2.5 temporal scan sees identical
candidates at search, write and decode time; headers drop error
resilience (coded `primary_ref_frame = PRIMARY_REF_NONE`). Validated
per feature by selection-proving unit tests, a 230-config black-box
sweep (5 geometries × 6 q × 7 contents + 20 segmentation configs)
byte-exact in THREE independent reference decoders, and three more
self-encoded streams pinned in the conformance corpus (53 total).

### Inter encoder: B-pyramid GOPs + masked compound (r415)

r415 lands the backward-reference arc. **B-pyramid GOPs**
(`encoder::encode_pyramid_gop_yuv420{,_with_q}`): each mini-GOP of up
to four frames codes OUT OF ORDER as a two-level pyramid — the last
frame first as a decoded-not-shown ALT reference (`show_frame = 0`,
coded `showable_frame = 1`), the midpoint as a not-shown MID
reference predicting forward (LAST) and backward (BWDREF/ALTREF —
§7.8 sign bias 1, §7.9 bidirectional temporal projection), shown B
frames between the anchors with `{ LAST, BWDREF }` / `{ LAST,
ALTREF }` bidirectional COMPOUND_AVERAGE pairs (the §5.11.25
`BIDIR_COMP_REFERENCE` cascade) and §5.9.22 forward/backward skip
mode, and §5.9.2 `show_existing_frame` short headers at each
not-shown frame's display position. Order-hint-tracked three-slot
§7.20 rotation hands the ALT slot to the next mini-GOP as its anchor;
temporal units follow the "exactly one shown frame per unit"
conformance rule (not-shown frames ride the next shown frame's unit).
**Masked compound**: every sequence header now opens
`enable_masked_compound` — compound leaves code the §5.11.29
`comp_group_idx` cascade and the RD ladder trials all 32
COMPOUND_WEDGE `(index, sign)` pairs plus both COMPOUND_DIFFWTD mask
types through the decoder's own §7.11.3.11/§7.11.3.12 mask blends
(wedge-blend content provably commits WEDGE leaves). Validated by
selection-proving witnesses, spec-driver round trips over GOP
lengths 1-9 × the full content/q matrix, a 30-config pyramid
black-box sweep plus P-GOP re-validation byte-exact in THREE
independent reference decoders, and four more self-encoded streams
pinned in the conformance corpus (57 total).

### Inter encoder: jnt-comp + sub-8×8 leaves (r416)

r416 works the r415 follow-up ladder. **Jnt-comp** (§7.11.3.15
distance-weighted compound): every sequence header now opens
`enable_jnt_comp` — compound leaves code the §5.11.29 `compound_idx`
S() (per-block §8.3.2 `fwd == bck` order-hint ctx seed, derived
identically at search, write and decode time) and the RD ladder
trials the COMPOUND_DISTANCE blend (`Quant_Dist_Weight` /
`Quant_Dist_Lookup` over the real frame order-hint deltas) against
the coded-AVERAGE arm; distance-blend content provably commits
DISTANCE leaves. **Sub-8×8 inter leaves**: the partition-search floor
drops from BLOCK_8X8 to BLOCK_4X4 — HORZ / VERT at BLOCK_8X8 (8×4 /
4×8), PARTITION_SPLIT to four BLOCK_4X4 leaves, and the 16×4 / 4×16
HORZ_4 / VERT_4 strip alphabet at BLOCK_16X16. Sub-8 leaves are
single-reference per the §5.11.25 `Min( bw4, bh4 ) >= 2` forcing;
residual coding lands the §5.11.34 `HasChroma` gate (the bottom/right
cell of each 2×2 group codes the WHOLE group's chroma at the §5.11.38
plane residual size, predicted through the decoder's own §5.11.33
per-luma-cell chroma tiling). Selection witnesses pin
4×4-checkerboard motion → BLOCK_4X4 SPLIT leaves and 4-row band
motion → HORZ_4 strips + HORZ 8×4 halves. The black-box sweep matrix
gains `fine` / `bands` content kinds; the 30-config pyramid sweep and
all three r416 self-encoded streams decode byte-exact in THREE
independent reference decoders (corpus 60 total).

### Inter encoder: inter-intra blends + sub-8×8 intra leaves (r417)

r417 works the r416 follow-up ladder. **Inter-intra blends**
(§7.11.3.14): every sequence header now opens
`enable_interintra_compound` — single-reference 8×8..32×32 leaves
code the §5.11.28 cascade, and the RD ladder trials all four
§6.10.27 II modes through the §7.11.3.13 smooth intra-variant mask
plus the 16 §7.11.3.11 wedge masks (where `Wedge_Bits > 0`), the
intra half predicted into the search scratch through a
buffer-parameterised split of the decode walker's own §7.11.2 core
(one code path for decode and search — the r416 "missing piece").
Blend content provably commits inter-intra leaves. **Sub-8×8 intra
leaves in inter frames**: BLOCK_4X4 nodes RD-trial the §5.11.22
intra arm against the searched inter leaf, and committed intra
winners stamp `RefFrame[ 0 ] = INTRA_FRAME` into the driver grids so
the §5.11.33 `someUseIntra` chroma arm (whole-region group chroma at
the inter leaf's own MV) fires identically at search and decode
time; mixed-group content provably commits intra 4×4 leaves beside
inter ones. The sweep matrix gains the `iifade` kind; the 30-config
pyramid sweep and both r417 self-encoded streams decode byte-exact
in THREE independent reference decoders (corpus 62 total).

### Screen-content encoding: palette + intra-block-copy search (r418)

r418 builds the SEARCH side of the screen-content tools (the write
arms landed earlier). **§5.11.46 palette election**: every eligible
square leaf (8×8..64×64, fully on-screen) builds palette candidates —
exact colour lists where a block carries ≤ 8 distinct values, and
(new) k-means-clustered quantised palettes beyond that (weighted 1-D
luma / 2-D joint-(U,V) Lloyd with a size-RD pick of `k ∈ 2..=8` and a
density gate `distinct ≤ samples/8`) — and RD-trials every available
combination (luma / chroma / both) at every §5.11.15 TX shape against
the plain intra leaf, on the lossy and the lossless arm, in KEY
frames and (via the shared leaf encoder) intra leaves inside inter
frames. **§5.11.7 intra-block-copy election** (KEY frames): the
§5.9.20 gate opens content-adaptively (duplicate-64×64-tile scan,
§6.10.24-reachability-checked), and eligible leaves RD-trial a
bounded even-offset DV set filtered by a full §6.10.24 `is_mv_valid`
transcription (raster delay + wavefront), coded on the
`use_intrabc = 1` arm with the `is_inter = 1` residual layout.
Selection witnesses prove palette (exact + clustered, luma + chroma,
KEY + P-frame) and intrabc leaves are committed; the sweep matrix
gains the `screen` kind; the 30-config pyramid sweep, 18 ad-hoc
screen/palette/intrabc streams, and both r418 self-encoded pins
decode byte-exact in THREE independent reference decoders (corpus 64
total).

### Inter encoder: motion-mode election + intra tools in inter frames (r419)

r419 closes the remaining inter-tool ELECTION axes. **§5.11.27
motion-mode election**: every inter frame codes
`is_motion_mode_switchable = 1` and `allow_warped_motion = 1` (the
§5.5.2 `enable_warped_motion` sequence gate opens), so every eligible
single-reference leaf codes the `use_obmc` / 3-way `motion_mode` S();
the leaf search trials — after the mode/MV/filter selection — the
§7.11.3.9-10 **OBMC** overlap blend (per codable filter, through the
decoder's own neighbour-scan dispatch over the committed grids) and,
where the arm-B gates open (`NumSamples > 0` on the §7.10.4 scan,
unscaled reference), the §7.11.3.5 **WARPED_CAUSAL** warp with the
§7.11.3.8 least-squares fit (committed only when `setup_shear`-valid;
committed filters collapse to the reader's bit-silent EIGHTTAP per
`needs_interp_filter( )`). The write arm re-derives the reader's full
§5.11.27 cascade from the write mirror
(`has_overlappable_candidates( )`, `find_warp_samples( )` at the
committed post-`assign_mv` vector) and rejects uncodable commitments;
search/write/decode stamp identical `MotionModes[]` grids, and the
§5.11.5 driver grids join the search's snapshot/rollback discipline
(the OBMC neighbour scan reads committed above/left cells through
them). **Filter-intra + CfL inside inter frames**: the intra-leaf arm
rides the shared leaf encoder, and two witnesses prove reachability
end-to-end — a P-frame region constructed as the §7.11.2.3 prediction
of its own decode-time neighbours commits `use_filter_intra = 1`
leaves, and a fresh region whose chroma tracks the subsampled luma AC
commits `UV_CFL_PRED` leaves. Selection witnesses pin sheared motion →
OBMC leaves and zooming motion (a true affine field) → WARPED_CAUSAL
leaves; measured on the witness contents, warp saves 1.4-3.0% bytes
AND gains 0.3-0.4 dB luma PSNR on affine content, OBMC adds ~0.03 dB
at ~equal rate on shear content, and the always-coded motion-mode
S() costs ≈ 0.4% on translational content. The sweep matrix gains the
`shear` / `zoom` kinds; the 30-config pyramid sweep and three r419
self-encoded pins decode byte-exact in THREE independent reference
decoders (corpus 67 total).

### True bit-accounting rate costs: the search-side rate twin (r421)

r421 replaces every RD ladder's heuristic rate proxy with the real
thing. The encoder now carries a **rate twin** — a shadow of the
tile's live write state (the §8.3.1 working CDFs, the §5.11
neighbour-context mirror, the §8.2.6 arithmetic-coder `range`) that
the search runs candidate symbol sequences through WITHOUT emitting,
reading off each candidate's exact fractional bit cost (1/256-bit
fixed point: renormalisation bits plus the `log2(range)` drift,
deterministic integer arithmetic throughout). The twin re-implements
no syntax: pricing and committing run the SAME
`write_partition_tree_syntax` / `write_block_syntax` / partition-arm
functions the emitting pass runs, only with a counting symbol writer
(identical §8.2.6 range trajectory and §8.3 adaptation, no `low`
accumulator) — so it cannot drift from the writer's arm selection,
and the driver asserts the committed twin equals the writer's CDFs +
coder range after every superblock's real emission (an end-to-end
witness additionally pins the summed per-superblock costs to the
emitted tile payload within the §8.2.4 termination slack). Elections
priced with exact bits: KEY — leaf-vs-split partitions, tx-depth
ladder, palette combos, intra-bc; INTER — the full §5.11.4 shape
election (multi-block shapes thread a running fork so later blocks
are searched and validated under their siblings' committed stamps),
inter-vs-intra, skip-mode, depth ladder, and the §5.11.27 motion-mode
election (SIMPLE / OBMC / WARPED_CAUSAL priced through the writer's
own arm derivation against the current adaptive rows). The twin's
write-path validation also surfaced and fixed two search/header
inconsistencies (compound candidates offered without
`reference_select`; filter trials under a non-SWITCHABLE frame
filter). Measured on the committed A/B matrices (heuristic → twin,
same inputs, joint `SSE + λ·bits` objective never worse): 66-config
inter GOP **−3.06% bytes** at −0.05 dB (twin smaller on 62/66);
30-config pyramid **−4.98% bytes** at −0.19 dB (smaller on 27/30);
315-config intra +0.41% bytes for **+0.15 dB** mean PSNR (smaller on
159/315 — the byte regressions pair with outsized PSNR gains, e.g.
+4.1 dB on q200 noise). The r419 OBMC-at-q60 flag re-judged: on
q60 shear content the twin saves 2.9% bytes AND gains 0.13 dB, and
the OBMC selection witness still commits OBMC leaves under exact
costs. The pre-r421 heuristics stay selectable through hidden
`*_rate_model` entry points as the measurement baseline
(`tests/rate_twin_ab.rs`, env-gated full measurement + always-on
conformance A/B); the full 411-stream twin sweep decodes
byte-identical in THREE independent black-box reference decoders, and
two representative improved streams are pinned in the conformance
corpus — the re-judged q60 shear GOP and the −27% q255 shear pyramid
(corpus 69 total).

### Global warped-motion election (r422)

r422 lands the frame header's last identity-only stub: the §5.9.24
`global_motion_params()` write arm now emits real models. The
§5.9.25 `read_global_param` inverse (recenter forward, §5.9.28
bucket-ladder subexp encoder, both §5.9.27 recenter arms, the
per-type coefficient order with the derived ROTZOOM `[4]/[5]` pair)
is byte-exact against the crate's own parser on synthetic ordinal
sweeps, and a frame-level election feeds it: a coarse per-reference
motion pre-pass (exhaustive half-resolution scan over 2×2-mean
planes — fine-texture aliasing and reference coding blur wash out —
then full-pel + 1/8-pel bilinear refinement), least-squares fits of
TRANSLATION / ROTZOOM / AFFINE, §5.9.25 grid quantization BEFORE
scoring, §7.11.3.6 `setup_shear` validation, and a residual-energy
gate with ratio + absolute-margin class upgrades. The elected
`(GmType, gm_params)` live in ONE shared bundle feeding the §7.10.2.1
`GlobalMvs` derivation and the §7.11.3 global-warp prediction of
search mirror, write pass and decoder alike — the model can only
change which streams the RD ladder prefers, never desync one.
Witnesses prove pan content elects TRANSLATION at the exact coded
vector, zoom and rotation content elect ROTZOOM, and static content
stays IDENTITY (bit-identical stream). Measured on the committed
30-config A/B matrix (identity-only → elected, same inputs): +0.92%
bytes for **+0.53 dB** mean PSNR, warp content decisive —
rotation-64×64-q60 **+1.30 dB** at +6 B, zoom-64×64-q100 +1.42 dB;
all 30 elected-model streams decode byte-identical in THREE
independent black-box reference decoders
(`tests/global_motion_ab.rs` joint-objective smoke + env-gated
matrix). Two streams pinned: `self-gop-64x64-q60-gm-zoom-warp` and
`self-gop-64x64-q60-gm-rotation` (corpus 71 total).

r422 also converts the last big INTER-path heuristic to the twin:
the §5.11.23 mode-cascade candidate rates. The mode + MV prefix
(§5.11.25 reference cascade, four-arm `YMode` dispatch, `drl_mode`
loop, NEWMV `read_mv` differences) is factored into ONE writer body
(`write_inter_mode_mv_prefix`) that both the emitting pass and the
twin's `price_inter_mode` run, so every leaf candidate — NEWMV drl
slot choice included — is priced with exact fractional bits against
the current adaptive CDFs. The refreshed twin-vs-heuristic matrices:
66-config inter GOP **−3.49% bytes** at −0.07 dB (smaller on 63/66,
was −3.06% under the r421 proxy mode rates), 30-config pyramid
−5.02% at −0.17 dB; the full 411-stream twin sweep re-validates
byte-identical in THREE independent black-box reference decoders
(1233/1233 decoder runs).

### Cross-frame state carry: primary-reference election + temporal segment maps (r423)

r423 ends the encoder's per-frame statelessness. P-frames elect
§5.9.2 `primary_ref_frame = 0` (LAST): a §7.20 per-slot carry store
tracks every refreshed frame's end state — the §8.4 `save_cdfs`
frame-end CDF table, `SavedSegmentIds`, `SavedGmParams` — and each
INTER frame starts from its primary slot per the spec loads (§6.8.21
`load_cdfs` with symbol counts zeroed, §7.21 `load_previous()` — the
§5.9.24 subexp coefficients now recenter against the CARRIED
`PrevGmParams` — and `load_previous_segment_ids()`). The §5.9.14
flag triple becomes real coded bits, and the §5.11.19
`segmentation_temporal_update` write arm goes live end-to-end: per
block, the §5.11.21 `get_segment_id()` prediction, the §8.3.2
seg-pred ctx read before the block's own stamp, the
`seg_id_predicted` S() with the §5.11.20 spatial fallback, and both
spec-mandated `SegPredContext[]` stamp arms on the write mirror.
`temporal_update` is elected per frame by EXACT realized bits: the
main pass searches and emits under the spatial arm (trees
bit-identical to the temporal-disabled baseline), the committed
trees replay under the temporal arm from the same frame-start CDFs,
and the smaller tile wins — so the elected stream is
smaller-or-equal per frame by construction. Measured on the
committed 12-config persistent-segment matrix: the carry is worth
**−1.52%** total bytes (12/12 smaller), the temporal election a
further −0.06% (12/12 smaller-or-equal, up to 4/5 P-frames elected).
r423 also fixes a latent skip-leaf invariant bug in the shared
intra-leaf ladder (trial candidates priced with the
constructor-default segment instead of the §5.11.9 forced pred,
hard-erroring segmented encodes). Witnesses + A/B harness in
`tests/temporal_segmentation.rs`; two streams pinned:
`self-gop-128x64-q72-seg-temporal-moving` and
`self-gop-192x128-q72-seg-temporal-static` (corpus 73 total).

### Deep B-pyramids, adaptive mini-GOPs, per-TU twin residue (r424)

r424 deepens the GOP structure end-to-end. The **B-pyramid planner**
generalizes from the fixed two-level mini-GOP to a recursive dyadic
pyramid of arbitrary depth (`encode_pyramid_gop_yuv420*` now codes
mini-GOPs up to 16 frames = four temporal layers): the ALT anchor
codes first decoded-not-shown, midpoints recurse level by level, and
shown non-reference B leaves bottom out gap-2 intervals — with
backward roles drawn from the enclosing-anchor chain (`BWDREF` the
nearest coded future frame, `ALTREF2` the next enclosing anchor,
`ALTREF` the mini-GOP ALT), the matching §5.11.25 BIDIR compound
pairs in the RD ladder, all eight §7.20 slots under a free-list
rotation, and per-layer quantiser offsets. The r423
**primary-reference carry flows through the pyramid** with a
per-frame exact-bytes election: the search runs under the LAST-slot
carry, the committed trees replay bit-exactly under the
nearest-backward anchor's carry and under per-frame defaults, and
the smallest total frame wins — pure rate, identical reconstruction
by construction. **Adaptive mini-GOP sizing**
(`encode_adaptive_gop_yuv420_with_q`): a motion-compensated MAD
probe drives scene-cut detection (cuts are absorbed by flat P steps
— no mini-GOP spans one) and depth classes, with a twin-consistent
trial-encode election at the class boundary (deep chunk vs
half-depth splits over the identical frame span). Measured on the
36-config A/B matrix: deep −4.11% bytes at −0.15 dB vs the two-level
baseline; adaptive −2.85% bytes at **+0.16 dB** (smaller AND better —
the election puts depth only where it pays); the primary election
adopts a carried primary on essentially every coherent frame and
demotes to `PRIMARY_REF_NONE` exactly at post-cut frames. r424 also
lands the **per-TU twin residue** standing since r421: a running
`TuFork` threads the leaf residual chain so every §5.11.47 tx-type
candidate (inter, intra and intrabc arms) prices its ACTUAL §5.11.39
coefficient chain through the writer's own one-TU body against the
fork's running CDF / level-context state, and the §5.11.46 palette
k-means inner ladders surface per-`k` candidates settled by exact
full-leaf twin bits — the last proxy prices inside the residual
chain are gone. The screen-content ladder opens with the §5.11.46
**signed-delta V-plane arm election** (UV-palette leaves price both
V-entry coding arms and commit the exact-bits winner — witnessed on
tight-V-cluster content). Harnesses: `tests/pyramid_deep_ab.rs` +
`tests/screen_content_polish.rs`; three streams pinned:
`self-pyr-64x64-q60-len17-deep`, `self-adaptive-96x80-q60-cut-n13`
and `self-kf-64x64-q60-vdelta` — the first self-encoded signed-delta
V-plane stream on the wire (corpus 76 total).

### Screen-content completion: hash-match intrabc + rect/clipped palette (r425)

r425 closes the screen-content ladder item. **Hash-match DV
search**: a per-frame block-hash index (own design — FNV-1a 8×8 base
tier at every even input position, 16/32/64 tiers composed from
quadrant hashes, flat-block suppression, capped buckets) arms with
the §5.9.20 gate; eligible leaves probe it with their input samples
and exact-match sources seed the §5.11.7 DV search at ARBITRARY even
offsets, nearest-first, ahead of the r418 geometric strides — every
seed still passes the full §6.10.24 transcription, the
reconstruction-space SSD ranking and the exact-twin-bits election.
The frame gate grows a **glyph tier** (16×16 duplicate-cell scan at
§6.10.24-valid lags) for repeated patterns that never align to whole
superblocks. **Rectangular + clipped palette leaves**: the KEY RD
ladder trials `PARTITION_HORZ`/`PARTITION_VERT` with two intra
leaves at `BLOCK_16X16+`, and frame-edge half-straddle nodes elect
the `split_or_horz`/`split_or_vert` single-rect arm — a clipped
HORZ-top / VERT-left block whose §5.11.46 candidates build over the
ACTUAL on-screen sub-rectangle, whose colour maps carry the §5.11.49
off-screen replication fill, and whose residual walk skips
off-screen-origin TUs with clip-aware legs. Measured on the
18-config screen matrix: screen tools code **5.49× smaller than
natural coding** (6.97× on the 11 pixel-exact-luma configs; r418
stood at 4.6×), the hash index alone is worth **−34.9%** on the
repeated-glyph page. Two streams pinned:
`self-kf-256x144-q60-screen-rect` (hash-seeded off-stride DVs, the
corpus's first rectangular AND first clipped palette leaves) and
`self-gop-192x112-q60-screen-scroll` (scrolling page GOP, elected —
not forced — edge partitions), byte-identical in THREE independent
reference decoders (corpus 78 total).

### Per-segment lossless mixing: pixel-exact regions in lossy frames (r426)

r426 lands ladder item 6. A §5.9.14 `SEG_LVL_ALT_Q` segment whose
§7.12.2 `get_qindex` clamps to 0 flips the full §5.9.2
`LosslessArray[]` leaf semantics for its blocks INSIDE an
otherwise-lossy frame — TX_4X4-only §5.11.34 row-major TU walk,
§7.13.2.10 WHT residuals (bit-exact), no tx-size / tx-type symbols —
exactly the decoder's per-block `Lossless = LosslessArray[
segment_id ]` derivation (the spec decode driver was already
per-segment; no decode gap found). The encoder resolves `Lossless`
from each leaf's OWN committed segment across the residual/depth
ladder, mirror stamps, skip-mode guard and intra fallback, with two
§5.11.9 skip-inheritance corners: the bit-silent `segment_id = pred`
short-circuit is frame-type-agnostic (segmented KEY frames too), and
a skip leaf whose pred segment flips its `Lossless` derivation
reverts its tx commitment to the spec-forced default. **Exactness
demand**: `encode_gop_yuv420_with_q_lossless_regions` turns caller
`LosslessRegion` pixel rectangles into an mi-cell mask (2×2-mi group
dilation covers the sub-8×8 `HasChroma` coder + 4:2:0 cositing);
every overlapping leaf is FORCED onto the lossless segment on every
arm, so the region decodes pixel-exact against the INPUT on EVERY
frame — asserted per-sample across aligned/unaligned rects,
multi-superblock frames and lossy-delta ladders. Measured on the
5-config typing-panel matrix: mixed streams run **18–56 % of
full-lossless** while keeping the panel exact, and at 64×64 q60 the
mixed stream (575 B) UNDERCUTS plain lossy (907 B) — exact
references collapse later panel blocks to skips, cross-frame value
the per-leaf greedy election cannot see. The content-driven
`auto_detect` election (synthetic leaves outside the mask trial the
lossless segment on twin bits + distortion) measured honestly inert
on this matrix — ≤8-alphabet panels are already palette-exact
(r425), and the 12-value irregular panel keeps the lossy arm at
q100. Two streams pinned byte-identical in THREE independent
reference decoders: `self-gop-64x64-q60-mixll-typing` and
`self-gop-96x80-q160-mixll-bigpanel` (mi-unaligned 43×30 panel,
su(1+8) `-160` delta; corpus 80 total).

### Segmentation inter overrides: SKIP / GLOBALMV / REF_FRAME pinned (r426)

r426 closes ladder item 8. The encoder codes all three §5.9.14
inter-override features: the writer derives every
§5.11.10/§5.11.11/§5.11.20/§5.11.23/§5.11.25 gate per BLOCK from the
committed segment id (a frame-level any-segment collapse in the
decode-side intra prefix — which mis-forced `skip = 1` on mixed
tables — is fixed with it), `SegIdPreSkip = 1` moves segment-id
coding to the pre-skip arm on every block, and three twin-priced
per-leaf trials elect the segments (pure-derivation SKIP blocks,
mode/ref-silent GLOBALMV blocks over the full depth ladder,
REF_FRAME re-labels of single-LAST winners). Three streams pinned
byte-identical in THREE independent reference decoders —
`self-gop-96x80-q80-seg-skip`, `self-gop-96x80-q72-seg-globalmv`,
`self-gop-64x64-q60-seg-refframe` (corpus 83) — closing the
decoder's last unpinned §5.11 segmentation paths. The GLOBALMV work
surfaced and fixed a latent r422 bug: §7.10.2.1 stores TRANSLATION
models in (row, col) order while the affine projection is x-first —
the estimator packed x-first for every class, so TRANSLATION
GLOBALMV predictions ran on a swapped vector (conformant, never
elected).

### High-precision MVs: the eighth-pel arm (r428)

r428 opens encoder-election ladder item 1. Every conformance-grade
inter driver arms §5.9.2 `allow_high_precision_mv`: the sub-pel
refinement adds an eighth-pel (±1 in 1/8-luma units) pass through
the real §7.11.3.4 kernels, the §7.10.2 stacks run the §7.10.2.10
no-op precision arm, and every §5.11.32 difference component codes
the `mv_hp` cascade at exact twin-priced bits. The header flag is
elected per frame by EXACT realized bits: the committed trees replay
under the quarter-pel arm (self-validating — an odd committed
component or a rounded-away derivation errors the replay out) and
the smaller tile wins, with ties preferring the conservative
quarter-pel wire shape. Measured on the 18-config A/B sweep
(`tests/hp_mv_ab.rs`): **−1.1% bytes at +0.33 dB mean** vs the
quarter-pel baseline. Pinned: `self-gop-96x80-q60-hpmv` — the
corpus's first `allow_high_precision_mv = 1` stream, byte-identical
through three independent black-box reference decoders (corpus 106).

### Per-superblock delta-q: elected adaptive quantisation (r428)

r428 closes encoder-election ladder item 2. Unsegmented lossy inter
frames probe per-superblock source activity into an absolute
§5.9.17 `CurrentQIndex` plan (flats refine, texture coarsens;
`delta_q_res = 3`), run a second full search under it — per-SB λ
tracks the running index, the twin forks thread the §5.11.2
lifecycle so exactly each superblock's first coded block carries the
§5.11.13 symbol, full-SB skip leaves honour the short-circuit arm —
and elect the better arm per frame by exact realized bytes under a
masking-weighted (variance-normalising) joint objective. Measured
(`tests/delta_q_ab.rs`): the banding-prone flat region gains
**+0.72 dB at −3.0% bytes** (128×128 q100) with the plain-PSNR trade
reported honestly; the uniform-spread control stays bit-identical.
Pinned: `self-gop-128x128-q100-deltaq` — the corpus's first
`delta_q_present = 1` stream, byte-identical through three
independent black-box reference decoders (corpus 107). r431 lands
the KEY-frame twin: the shared complexity probe plans per-superblock
`CurrentQIndex` units on unsegmented lossy KEY frames, a second full
search encodes under the plan, and the masking-weighted
exact-realized-bytes election keeps the better arm (corpus 115,
`self-key-128x128-q140-delta-q`). The segmentation × delta-q pairing
stays open.

### Frame-level CDEF election (r428)

r428 closes encoder-election ladder item 3 at frame granularity. The
sequence `enable_cdef` gate is open on every conformance-grade
stream: each lossy frame's committed reconstruction is filtered
through the decoder's own §7.15 driver over the write mirror's
§5.11.56 `cdef_idx[]` anchors and 8×8 skip conjunction, a bounded
strength search scores candidates against the source, and a winner
that beats the unfiltered frame lands in the header and in the
§7.20 reference store (encoder recon stays byte-exact with the
decoder, filter live). `cdef_bits = 0` — zero tile bits, the
election is pure distortion. Hard-gated off on lossless / intrabc /
exactness-demand / auto-lossless configurations (segmented
configurations joined in r436).
Measured (`tests/cdef_ab.rs`): **+0.28 dB at −1.2% bytes** on
ringing-prone edges (96×80 q140). Pinned:
`self-gop-96x80-q140-cdef` — the corpus's first self-encoded stream
with non-zero §5.9.19 strengths, byte-identical through three
independent black-box reference decoders (corpus 108). r429 deepens
both in-loop filters to UNIT granularity: per-64×64 `cdef_bits > 0`
election (multi-set §5.9.19 headers, §5.11.56 per-unit strength-id
literals via tile re-emission; corpus 110) and the §5.9.20/§5.11.57/
§7.17 loop-restoration election — per-unit Wiener (alternating-LS
fit) + self-guided (projection fit over all 16 Sgr sets) mirrored on
the encoder recon path with exact-realized-bytes settlement (+0.32 dB
at ~2 B/frame on detail content; corpus 111). r436 lifts the
segmented-frame gates on the CDEF, delta-q AND loop-restoration
elections (see the segmentation-pairings section).

### Mirror-path retirement (r428)

The historical encoder-mirror surface is GONE: the fixed-16×16 and
dyn-extent intra mirror encoders (whose non-conformant streams only
the matching writer-inverse decode arm could read), `decode_av1`'s
mirror-acceptance arm, and the historical `Frame::Yuv420_16x16` /
`Frame::Yuv420Dyn` / `Frame::YDyn` variants (the enum is
`#[non_exhaustive]`; only `Frame::Spec` remains). Every stream —
including everything `encode_av1` emits — decodes through the
spec-faithful driver. The shared SH/FH scaffolding the mirror module
housed (`Yuv420Frame`, the intra-only header builders,
`sb_grid_origins`) moved to `encoder::yuv_frame`. The public-API
round-trip gate (`tests/encode_decode_pixel_roundtrip.rs`) was
rewritten onto the conformance-grade encoders over the same
dimension / quantiser / content axes.

### Annex B length-delimited bitstream (r428)

The `annexb` module reads and writes the Annex B.2 packing
(`temporal_unit_size` / `frame_unit_size` / `obu_length` nesting):
`decoder::decode_av1_annexb` converts each temporal unit to its §5.2
low-overhead equivalent (enforcing the Annex B.3 size-consistency
and temporal-delimiter-placement rules) and drives the same spec
decode session as the IVF path; `annexb::build_from_temporal_units`
wraps this crate's own streams into Annex B framing (one frame unit
per frame). Both arms are triple-validated: an external `--annexb`
stream decodes byte-exact to the three-decoder reference digest, and
the crate's own repacked KEY / GOP / pyramid streams decode
byte-identically in all three external decoders
(`tests/annexb_conformance.rs`; fixtures `ext-annexb-96x80` +
`self-annexb-96x80-q80`).

### Scalability: operating points + temporal layers (r430)

The §5.3.3 OBU extension headers the walker always parsed are now
HONOURED end to end. Decode side: `SpecDecodeSession::
set_operating_point` / `decode_av1_at_operating_point` select a
§6.7.5 operating point (default: entry 0), `OperatingPointIdc`
re-derives at every sequence-header parse per §5.5.1, and the §5.3.1
`drop_obu()` rule skips every extension-carrying OBU outside the
point's temporal/spatial layer set before any payload parse — a
temporally scalable stream decodes to exactly the shown frames of
the surviving layers (out-of-range selections surface
`Error::OperatingPointOutOfRange`, the §6.7.5 abandon arm). Encode
side: `encoder::encode_temporal_layered_gop_yuv{420,}_with_q` codes
a dyadic 2..4-layer ladder in display order (zero latency, one
temporal unit per frame, layer `t` predicting only from layers
`<= t`, per-layer §7.20 slot policy + LAST-slot CDF carry) with the
§6.7.5 operating-point list on the wire and extension headers on
every frame OBU. A three-layer stream decodes byte-exact in two
independent reference decoders at ALL THREE operating points (and a
third on the full decode); corpus stream 112 pins the full + both
reduced subsets (`tests/temporal_layers.rs`,
`tests/scalability_op.rs`; fixture
`self-gop-64x64-q72-temporal-layers`).

### Large-scale-tile mode: §5.12 tile lists + §7.3 decoding (r430)

AV1's second operating mode (§7.1). The `tile_list` module parses
and writes the §5.12.1/.2 tile-list OBU under the §6.11 conformance
bounds, and `tile_list::decode_tile_list{,_stream}` runs the §7.3.1
ordered steps: per entry, the selected anchor (an externally
supplied decoded frame — §6.11.2 `AnchorFrames`) is installed as
`FrameStore[ ref_frame_idx[0] ]`, the §7.3.2 decode-camera-tile
process decodes the entry's `coded_tile_data` (fresh symbol decoder,
frame-start CDFs, single tile, no post-processing, no reference
update), and the tile lands in the output frame in raster order —
uncovered tiles stay untouched. The whole mode is gated on the
§7.3.1 constraint list (`Error::TileListInvalid` otherwise); general
decode sessions skip tile-list OBUs per the §7.3.1 note. The write
arm (`encoder::encode_camera_frame_yuv420`) produces §7.3-conformant
single-tile `W×64` camera frames through the generic inter driver
under an order-hint-free frozen-CDF shape (`InterFrameConfig::
freeze_cdfs`; the §5.9.3 zero-`OrderHintBits` distance arm and the
sequence-gate-derived `use_ref_frame_mvs` landed with it); a 2×2
output assembled from four camera tiles over four anchors decodes
byte-identical to the per-frame encoder reconstructions, and the
lossless arm reproduces the source exactly
(`tests/large_scale_tile.rs`).

### Multi-tile encoding: the §5.9.15 write arm (r431)

The encoder codes real tile grids. Every conformance-grade encode
path takes a uniform `(TileColsLog2, TileRowsLog2)` layout
(`encode_key_frame_yuv{,420}_with_q_tiles`, `GopTuning::tiles`; r433
extends this to the B-pyramid / adaptive driver via
`PyramidTuning::tiles` and the temporal ladder via
`encode_temporal_layered_gop_yuv{,420}_with_q_tiles`): per-tile §8.2
symbol partitions from the §8.3.1 frame-start CDF state, §5.11.2
`begin_tile` re-scoping (clear_above, tile-scoped availability,
LR-reference resets) on the write driver AND the search context,
§5.11.1 `tile_size_minus_1` fields at the minimal realized §6.8.14
`TileSizeBytes`, tile-0 §8.4 CDF donation, and every post-tile
exact-bytes election (hp / temporal-seg / primary-ref / CDEF / LR)
re-emitting ALL tiles. `(0, 0)` reproduces the single-tile streams
byte for byte. §7.3 camera frames split into tile COLUMNS
(`encode_camera_frame_yuv420_tiles` — one §5.12.2 `coded_tile_data`
run per column for `anchor_tile_col`-addressed tile-list entries).
Corpus stream 113 (`self-gop-192x128-q72-tiles-2x2`) pins the first
multi-tile stream, byte-identical across three independent black-box
reference decoders.

r433 adds the §5.9.15 **non-uniform** arm
(`uniform_tile_spacing_flag = 0`): `TileInfo::explicit_layout` (the
parser-twin derivation under the `maxTileWidthSb` /
recomputed-`maxTileAreaSb` / widest-column `maxTileHeightSb` bounds)
drives `encode_key_frame_yuv{,420}_with_q_tile_layout` and
`encode_gop_yuv{,420}_with_q_tile_layout` — uneven splits no uniform
layout can express (1+2 / 2+1 / 1+3+1 columns, 1+2 rows) round-trip
pixel-exact and decode byte-identical in three independent reference
decoders.

### Multi-tile-group frames (r433)

Both arms of the §5.11.1 tile-group SPLIT wire shape. Decode: the
OBU walk accumulates a frame's tiles across several
`OBU_TILE_GROUP` OBUs under the §5.9.1 `SeenFrameHeader` discipline
— §6.10.1 running-`TileNum` ordering enforced, decode fires only at
`tg_end == NumTiles - 1`, `OBU_REDUNDANT_FRAME_HEADER` accepted
mid-frame iff its §6.8.1 `frame_header_copy` bytes match the
original, mid-frame `OBU_FRAME_HEADER` / flagged `OBU_FRAME` groups /
temporal units ending mid-frame all reject with the typed
`Error::TileGroupInvalid`. Encode:
`encode_key_frame_yuv{,420}_with_q_tile_groups`,
`GopTuning::tile_groups` and `PyramidTuning::tile_groups` emit a
standalone `OBU_FRAME_HEADER` (§5.3.4 trailing bits) plus N
tile-group OBUs with contiguous `tg_start ..= tg_end` slices — the
per-tile entropy payloads are byte-identical to the single-group
stream (framing-only change), and `tile_groups <= 1` reproduces the
§5.10 `OBU_FRAME` packing bit for bit. Every repacked grouping,
native split KEY / GOP / pyramid / layered-ladder stream decodes
byte-identical through three independent black-box reference
decoders.

### Segmentation pairings: delta-q, CDEF and LR (r436)

The r428/r429 scope gates are lifted: §5.9.17 per-superblock
delta-q, §5.9.19/§7.15 CDEF (frame-level + per-unit ids) AND
§5.9.20/§7.17 loop restoration all run on ACTIVELY segmented
frames. A non-zero-segment block of a delta-stepped
superblock quantises at the §7.12.2 step-3 composition
`Clip3(0, 255, CurrentQIndex + FeatureData)` (an encoder-side
per-segment bundle previously baked `base_q_idx + data` — fixed);
tables carrying a lossless segment conservatively stay on the
single-quantiser arm per the §7.12.2 note. Corpus streams 120–122
(`self-gop-128x128-q120-seg-delta-q`,
`self-gop-128x96-q140-seg-cdef`, `self-gop-128x96-q140-seg-lr`) pin
all three pairings byte-identical through three independent
black-box reference decoders.

### `context_update_tile_id` election (r436)

On multi-tile GOPs the §6.8.14 field is ELECTED, not fixed: each
P-frame replays its committed trees from EVERY tile of its primary
frame's frame-end CDF states (§6.8.21 `load_cdfs`) and keeps the
start state realizing the smallest assembled §5.11.1 body — then
the primary frame's already-emitted fixed-width field is patched in
place (nothing else on the wire moves). `GopTuning::ctx_update_elect
= false` keeps the tile-0 donation bit for bit. On a designed
flat-tile/textured-tile GOP, 4 of 5 P-frames elect tile 1 and every
elected frame is strictly smaller (3298 → 3258 total bytes). Corpus
stream 119 (`self-gop-128x64-q80-ctx-update-elect`) pins the first
non-zero `context_update_tile_id` fields — three independent
black-box reference decoders ride the patched donations
byte-identically. The §6.7.5 temporal-ladder driver elects too,
under the multi-consumer discipline: a slot's donor set FREEZES at
its first consumption (several frames may chain off the same slot —
the KEY seeds all eight), so every later consumer replays the
committed donation, and every operating point decodes the patched
stream bit-exact.

r439 closes the election on the two remaining drivers. The
**B-pyramid / adaptive** encoder elects through its out-of-order
refresh graph: the donor election follows the frame's ELECTED
§5.9.2 primary (the r424 ordinal election may commit the
nearest-backward anchor — the replay candidates then come from THAT
carry and the patch lands in THAT frame's field), and both patch
paths of the multi-frame temporal units are live — a donor frame
still pending inside the current chunk's open unit is patched in
its OBU body, a flushed one through the unit's frame-ordinal walk.
The **spatial-SVC** driver elects per layer chain (openers included)
at each layer's own multi-tile layout, patching at `(temporal unit,
spatial-layer ordinal)` wire locations; every §6.7.5 operating
point decodes the patched stream bit-exact.
`PyramidTuning::ctx_update_elect` is the A/B switch (default on;
inert on single-tile layouts). Pinned:
`self-pyr-128x64-q80-tiles-ctx-elect` (the first out-of-order stream
with a patched donation) and `self-svc-128-256-q80-ctx-elect` (the
first spatially scalable one, digested at both operating points) —
byte-identical through three independent black-box reference
decoders (corpus 126 with the two QM pins).

### Quantizer-matrix election (r439)

The §5.9.12 `using_qmatrix` / `qm_y` / `qm_u` / `qm_v` write arm goes
live with a frame-level election — the encoder side of the §9.5.3
dequant tables the conformance corpus has exercised decode-side
since r394. On an unsegmented lossy frame whose luma carries real
high-frequency energy (a mean-absolute-second-difference probe —
smooth gradients skip the arm and stay bit-identical), a second
full search runs under `using_qmatrix = 1` at a quantiser-keyed
§9.5.3 level on all three planes: every quantise/dequantise step of
the residual chain rides the §7.12.3 QM-scaled
`q2 = Round2( q * Quantizer_Matrix[..], 5 )` through the same
`SegQMLevel[ plane ][ 0 ]` row the decoder derives per §5.9.2, and
the per-TU rate twin prices the actual QM-quantised coefficient
chains. The election is the plain joint objective
`D·256 + λ·R_bits256` over exact realized bytes, per frame (KEY and
inter alike), and the winner feeds the §5.9.17 delta-q election —
the arms compose on the wire. The second full search is spent only
inside the MEASURED win regime (`base_q_idx` 88..=176, luma extent
≥ 96×80 — at fine quantisers the election trades bytes for
distortion the flat lattice already serves, at very coarse ones it
essentially never fires): an election-scoping choice, not a
conformance constraint. Default-on for every conformance-grade
driver; the §7.3 camera mode arms it through
`encode_camera_frame_yuv420_tiles_qm` (r450 — §7.3.1's constraint
list never bars `using_qmatrix`, and the §7.3.2 camera-tile decode
dequantizes through the same header, witnessed byte-exact in
`tests/large_scale_tile.rs`); `GopTuning::qm` /
`PyramidTuning::qm` A/B switches, `tests/qm_ab.rs` harness. Measured
on the committed natural-texture matrix: up to **−14.85% bytes at
−0.067 dB** (96×80 q100 GOP) and −5.39% at **+0.20 dB** (128×128
q140); gradients stay bit-identical. Pinned:
`self-kf-128x128-q140-qm` and `self-gop-96x80-q100-qm` — the
corpus's first self-encoded `using_qmatrix = 1` streams,
byte-identical through three independent black-box reference
decoders.

### QM × segmentation + the multi-level QM ladder (r441)

The r439 election's segmented-frame gate is lifted: on an actively
segmented lossy frame the QM arm runs with the per-segment §5.9.2
`SegQMLevel[ plane ][ segment_id ]` derivation (a lossless segment
takes the no-QM sentinel 15), and the per-segment quantiser bundle
carries each block's own row through the residual chain — every
SEG_LVL feature trial prices its QM-quantised coefficients through
the same bundles, and the winner still feeds the §5.9.17 delta-q
election (three-way composition on the wire). r441 also grows the
arm into a **multi-level ladder**: the quantiser-keyed §9.5.3 level
plus its lighter neighbour both run full searches, settled by the
same exact-realized-bytes joint objective (refinement discipline —
the neighbour runs only when the keyed level already won). Pinned:
`self-gop-128x96-q120-seg-qm` — the first stream pairing
`segmentation_enabled = 1` with `using_qmatrix = 1`, byte-identical
through three independent black-box reference decoders.

### Superres election (r441; every driver + the §7.17 pairing r444)

The §5.9.8 write arm goes live. On a lossy single-tile KEY frame
inside the arming window (`base_q_idx` ≥ 60, extent ≥ 96×80, a
horizontal-second-difference content probe — §7.16 resamples columns
only), each legal candidate denominator codes the frame at the
§5.9.8 downscaled width, the reconstruction is upscaled through the
decoder's own §7.16 driver, and the plain joint objective over
ORIGINAL-extent SSE + exact realized bytes keeps the winner. Wire
shape: `frame_size_override_flag = 0` seeds `FrameWidth` from the
sequence maximum (the upscaled width) and §5.9.8 re-derives the
coded width; `allow_intrabc` follows its §5.9.5 gate. A
superres-elected KEY feeds the P chain its upscaled §7.20 reference.
Measured (`tests/superres_ab.rs`): on probe-passing content the arm
wins across the band — 128×96 q100 −9.3 % bytes at +1.09 dB,
192×128 q220 −20 %; horizontal-detail content collapses (18 →
14 dB) and stays probe-gated bit-identical.

r444 closes the r441 scope tails. **LR × superres**: the intra core
runs the §7.4 order end to end — CDEF settles at the coded extent,
the reconstruction and the pre-CDEF stripe snapshot are
§7.16-upscaled, and the §7.17 loop-restoration election fits, prices
and applies at the UPSCALED extent against the original source (the
§5.11.57 write window maps superblock columns through the §5.9.8
denominator ratio; the inner QM / delta-q elections of a superres
arm also score at the upscaled extent). **Every driver**: segmented
GOPs (the segmented P chain's §5.11.19 temporal prediction rides the
extent-checked all-zero `load_previous_segment_ids()` arm), the
B-pyramid / adaptive driver (`PyramidTuning::superres`), the §6.7.5
temporal ladder (the multi-OP repack preserves the elected sequence
gate), and the spatial-SVC driver — a per-layer PRE-PASS election
decides the shared header's `enable_superres` before any layer
codes, and elected openers ride BOTH §5.9.5 arms
(`frame_size_override_flag = 1` base KEY with the DISPLAY width in
the override fields; `INTRA_ONLY` enhancement openers off the
sequence maximum). Pinned: `self-gop-128x96-q180-superres` (r441),
plus the r444 pins `self-gop-128x96-q140-superres-lr` (the first
`use_superres = 1` stream with live §7.17 restoration),
`self-gop-128x96-q180-seg-superres` (the first
segmentation × superres pairing) and `self-svc-96-192-q180-superres`
(the first spatially scalable stream with superres openers, digested
at both operating points) — all byte-identical through three
independent black-box reference decoders.

r447 lifts the single-tile-frame scope: uniform §5.9.15 layouts and
tile-group packaging ride the arm — each candidate re-validates its
layout at the DOWNSCALED width and against the Annex A rule that a
`use_superres = 1` frame's non-rightmost tiles are ≥ 128 luma
samples wide (reference decoders enforce it; layouts violating it
are filtered per denominator). Pinned:
`self-gop-320x96-q180-superres-tiles` — the first
superres × multi-tile stream.

### Film-grain election (r441; AR taps + chroma points r444)

The §5.9.30 write arm goes live with estimated parameters. On an
unsegmented lossy GOP whose content passes the noise probe (real
residual energy against a 3×3-binomial denoised twin, spatially
modelable at lag 1, temporally decorrelated across the first two
frames — static texture and clean content are rejected up front),
the grain arm codes the DENOISED frames with a full §5.9.30 block on
every header (per-frame `grain_seed` schedule, scaling points
calibrated through the decoder's own §7.18.3 synthesis) and
publishes §7.18.3-synthesized output planes while the reference
chain stays pre-grain (§7.20). Elected under a documented
"perceptually-neutral rate" objective — structure fidelity plus
noise-amplitude mismatch penalties against the plain arm's
source-matched SSE — with a strict realized-bytes win required.
r444 grows the parameter surface into a CANDIDATE ladder: the r441
white luma-only shape, plus fitted §7.18.3 causal-neighbourhood AR
taps at lag 1 and lag 2 (each ring a separate candidate — the
parameter bytes ride every header, so depth belongs to the score;
chroma coefficient lists carry the jointly-fitted luma-correlation
tap) and per-plane four-point chroma scaling under identity index
mults, gated by a chroma twin of the luma probe with the §5.9.30
4:2:0 both-or-neither rule enforced; the objective adds chroma
amplitude terms and a luma correlation-match term, and the
strictly-fewer-bytes mandate applies per candidate. Measured
(`tests/film_grain_ab.rs`): −9.4 % bytes at amp-6 q60, −6.1 % at
amp-3 q60, with the honest plain-PSNR trade reported (~1–2 dB vs
the noisy source — definitional for synthetic grain).
`GopTuning::film_grain` (default on). Pinned:
`self-gop-128x96-q60-film-grain` (r441, white) and
`self-gop-128x96-q60-film-grain-ar` (r444 — the corpus's first
self-encoded `ar_coeff_lag > 0` stream), byte-identical through
three independent black-box reference decoders (corpus 133 with the
four r444 pins).

r447 closes the ladder's last §5.9.30 field: each fitted AR shape
offers a `chroma_scaling_from_luma` TWIN — no chroma points, no
mult/offset fields (§7.18.3.4 reads the LUMA points for all three
planes; the blend indexes at the co-located average luma), the
chroma AR lists still coded per the §5.9.30 csfl gates. Offered when
both chroma noise gates fire, settled by the same score +
strictly-fewer-bytes mandate: luma-tracking three-plane noise elects
csfl (670 B vs 763 B for the points shape); chroma-dominant noise
keeps the points shape via the amplitude-mismatch terms. Pinned:
`self-gop-128x96-q60-film-grain-csfl` — the corpus's first
`chroma_scaling_from_luma = 1` stream.

r450 walks the election across the §6.4.1 chroma/bit-depth axis —
no encoder change needed beyond witnesses (the estimation runs in
8-bit-normalized units and the synthesis mirror is the decoder's
own depth-aware §7.18.3 driver): MONOCHROME grain (the §5.9.30
header suppresses the `chroma_scaling_from_luma` bit and the whole
chroma surface; luma-only synthesis; −44 % bytes at amp-12 q40),
10/12-bit grain (`generate_grain` seeds at the `12 − bit_depth`
shift, the §7.18.3.4 LUT interpolates through the `bit_depth − 8`
index split, the blend clips at depth range), and 4:2:2 / 4:4:4
grain with the per-plane chroma gates UNCOUPLED (the
both-or-neither rule binds only 4:2:0), putting a legal
`num_cb_points > 0, num_cr_points == 0` header on the wire.
Witnesses (`tests/film_grain_formats.rs`) cover mono 8/12-bit,
4:2:0 10-bit, 4:2:2 Cb-only and 4:4:4 12-bit, all bit-exact.
Pinned: `self-gop-96x80-q40-mono-film-grain`,
`self-gop-96x80-q60-10bit-film-grain` and
`self-gop-96x80-q60-422-cb-film-grain` — the corpus's first mono,
first >8-bit and first single-chroma-plane grain streams (corpus
140).

r450 also lifts the election's UNSEGMENTED gate for the plain
SEG_LVL_ALT_Q ladder: the grain arm codes the denoised frames under
the same §5.9.14 table, so a P header carries BOTH the feature
table and the full §5.9.30 block (ladders with a lossless segment
keep the plain shape — grain on an exactness-contracted region
would break the pixel contract). The witness also caught a HARNESS
desync: the ref-less header parse mis-derived §5.9.22
`skipModeAllowed` (its two-forward fallback needs true slot hints),
so the wire audit now tracks §7.20 reference state across frames.
Pinned: `self-gop-128x96-q60-seg-film-grain` — the corpus's first
segmentation × grain stream (corpus 151).

### Short reference signaling: the §7.8 `set_frame_refs()` write twin (r452)

`GopTuning::short_ref_signaling` codes plain P-frames with
`frame_refs_short_signaling = 1`: only `last_frame_idx` /
`gold_frame_idx` go on the wire (7 bits instead of the explicit
22) and the decoder derives the whole seven-entry `ref_frame_idx[]`
through §7.8 from its stored `RefOrderHint[]`. The encoder ADOPTS
that derivation as its own reference map — LAST / GOLDEN stay on the
rotation slots (§7.8 seeds them from the explicit indices), the
unsearched ordinals land on the derived slots, and every downstream
twin (§7.8 sign bias, §5.9.22 skip mode, §7.9 projection) runs over
the adopted map. The §5.9.22 skip-mode twin resolves its second
forward reference by ORDINAL order over equal hints, so a derived
ordinal can name an out-of-rotation slot: those slots carry the
all-refresh FLOOR reconstruction, exactly the decoder's §7.20 store.
SWITCH / coded-error-resilient positions keep the explicit shape.
Pinned: `self-gop-96x64-q80-shortrefs` — the corpus's first stream
on the short arm, byte-identical through three black-box reference
decoders (corpus 152).

### Switch frames: the §5.9.2 S-frame cadence (r447)

`GopTuning::s_frame_period` codes every N-th inter frame as a
`frame_type = SWITCH_FRAME` — the spec's chunk-boundary frame:
all eight §7.20 slots overwritten without intra coding, so a
same-geometry stream can splice its §7.5 temporal units in at the
boundary. The wire rides the four §5.9.2 inferred (bit-free) fields
— `error_resilient_mode = 1`, `frame_size_override_flag = 1` (the
frame size codes explicitly), `refresh_frame_flags = allFrames`,
`primary_ref_frame = PRIMARY_REF_NONE` — plus the coded
`ref_order_hint[]` block over the true slot hints and the
error-resilient derivations `use_ref_frame_mvs = 0` and
`allow_warped_motion = 0` on both twins. Every cross-frame decode
dependency except the reference SAMPLES re-anchors at the switch
point (default CDFs; motion-field / segment-id / gm carries reload
from the S-frame's own committed state), and the S-frame still rides
the frame-level elections (hp-mv / delta-q / QM / CDEF / LR, tiles,
film-grain headers). Witnesses (`tests/s_frames.rs`): cadence GOPs
byte-exact through the spec driver, the SWITCH header shape parsed
back off the wire, and a CROSS-RATE SPLICE — q60 units 0..3 + q140
units 3.. decode end-to-end, byte-identical through three
independent black-box reference decoders on the spliced bytes.
Pinned: `self-gop-96x64-q80-sframes` (the corpus's first
`SWITCH_FRAME` stream) and `self-splice-96x64-q60-q140-sframe` (the
switch frame's symbols decoding against reference samples coded at
a different rate) — corpus 135.

### Coded error resilience: mid-GOP re-anchor frames (r450)

`GopTuning::error_resilient_period` codes every N-th inter frame
with the §5.9.2 `error_resilient_mode` f(1) set — the SWITCH
frame's re-anchor semantics WITHOUT the switch shape: the frame
stays a plain INTER frame (ordinary single-slot
`refresh_frame_flags`, references predicting across the boundary,
every frame-level election riding), but every cross-frame DECODE
dependency resets: §5.9.2 infers `primary_ref_frame =
PRIMARY_REF_NONE` (the f(3) is not coded — default CDFs +
`setup_past_independence`), `use_ref_frame_mvs = 0` and
`allow_warped_motion = 0` (neither f(1) coded), the
`ref_order_hint[]` block goes on the wire over the true §7.20 slot
hints (a mismatch marks the slot invalid), and the writer bypasses
`frame_size_with_refs`. Witnesses (`tests/error_resilient.rs`):
cadence GOPs byte-exact (lossy + lossless), the coded shape parsed
back off the wire, composition with the SWITCH cadence and with
tiles/elections. Pinned: `self-gop-96x80-q72-erm` — the corpus's
first CODED `error_resilient_mode = 1` stream (corpus 141).

### Explicit tile spans: the §5.11.1 flag-1 single group (r450)

`GopTuning::tile_spans` closes a read-only wire shape:
`tile_start_and_end_present_flag = 1` on a SINGLE tile group
covering the whole frame (`tg_start = 0`, `tg_end = NumTiles − 1`).
§5.10 requires the flag to be 0 inside an `OBU_FRAME`, so the arm
takes the split packaging — `OBU_FRAME_HEADER` + one
`OBU_TILE_GROUP` coding its span. The per-tile entropy payloads are
byte-identical to the flag-0 twin (witnessed), single-tile frames
and `tile_groups > 1` splits stay bit-identical to their baselines,
and the shape decodes byte-exact (`tests/tile_spans.rs`). Pinned:
`self-gop-128x128-q100-tilespans` (corpus 142).

### Spatial scalability: the SVC write arm (r431)

`encode_spatial_layered_gop_yuv{,420}_with_q` codes 2..=4
INDEPENDENTLY CODED spatial layers under ONE sequence header
(top-layer dimension budget; smaller layers ride §5.9.5
`frame_size_override_flag = 1`, inter frames the §5.9.7 no-found-ref
arm), §5.3.3 `spatial_id` extension headers on every frame OBU, one
shown frame per layer per §7.5 temporal unit, and nested §6.7.5
operating points (every sub-bitstream opens on the layer-0 KEY).
Layer 0 opens with the stream's only KEY frame; each enhancement
layer opens with a §5.9.2 `INTRA_ONLY` frame refreshing only its own
§7.20 slot pair, then predicts LAST-only inside its own rotation
with the §8.3.1 primary-reference CDF chain riding the same pair.
Corpus stream 114 (`self-svc-64-128-q72-spatial-layers`) pins the
first `spatial_id > 0` stream at BOTH operating points against
independent reference decoders' operating-point output.

r436 adds PER-LAYER tile layouts:
`encode_spatial_layered_gop_yuv{,420}_with_q_tiles` codes each
spatial layer under its own §5.9.15 uniform layout (validated
against that layer's legality window) on every frame of the layer,
plus §5.11.1 tile-group packaging clamped per frame to the layer's
realized tile count — split frames ride `OBU_FRAME_HEADER` +
`OBU_TILE_GROUP` OBUs with the §5.3.3 extension header on every
frame-carrying OBU per §7.5. Corpus stream 118
(`self-svc-tiles-128-256-q84`) pins a 2×1-tiled base under a
4×2-tiled enhancement, both split across two tile groups,
byte-identical through three independent black-box reference
decoders at both operating points.

r450 pins the first THREE-operating-point stream:
`self-svc3-64-128-q84` (64×64 / 128×64 / 128×128 spatial layers,
§6.7.5 idc masks `0x701 / 0x301 / 0x101`) — every point's output
byte-identical through two independent black-box reference decoders
AT THAT POINT (full interleave, two-layer prefix, base layer).

### External tool-combination battery (r450)

The r450 decode-tail sweep (`tests/external_sweep.rs`, an env-gated
harness that decodes `<name>.ivf` / `<name>.yuv` pairs through the
public entry and localizes the first diverging frame/plane) ran an
independent-encoder battery over previously untested pairings —
every stream decoded byte-exact on the FIRST run, zero fixes
needed, and all seven are pinned: coded
`error_resilient_mode` inter chains with hidden ALTREFs
(`ext-erm`), frame-parallel CDF freezing
(`disable_frame_end_update_cdf = 1`, `ext-frame-parallel`),
film-grain × reference scaling (`ext-grain-resize`), monochrome
INTER with `show_existing_frame` repeats (`ext-mono-inter`),
reference scaling × §5.9.8 superres on the SAME frame — two
downscales, three live extents, inside the 2× conformance bound
(`ext-resize-ss`), reference scaling × multi-tile
(`ext-resize-tiles`), and a 4:4:4 10-bit INTER chain
(`ext-444-10bit-inter`). With the 3-operating-point SVC pin the
corpus stands at 150.

### Not yet supported

- No black-box cross-check exists for assembled §5.12 tile-list
  output (external decoders cannot take an external anchor array);
  the coded camera tiles themselves ride the corpus-validated §5.11
  walk. (r433: the camera WRITE arm codes full 2-D grids — heights
  in multiples of 64 force one §7.3.1-conformant tile row per
  superblock row, addressed via `anchor_tile_row` /
  `anchor_tile_col`.)
- Delta-q stays off tables carrying a lossless segment
  (conservative §7.12.2-note guard).
- The §5.9.8 superres election stays KEY/opener-scoped (inter
  frames never elect a mid-GOP resize); explicit (non-uniform) tile
  layouts keep flat widths (their per-column superblock widths are
  bound to the full-width geometry), and the §7.3 camera mode is
  SPEC-BARRED from the pairing (§7.3.1 requires
  `enable_superres = 0`).
- The §5.9.30 film-grain election is GOP-scoped (≥ 2 frames; r450
  admits plain SEG_LVL_ALT_Q ladders — feature-extra segmentation,
  lossless regions/segments and the pyramid drivers stay out); the
  AR ladder stops at `ar_coeff_lag = 2` (the lag-3 ring's 24 + 2×25
  coefficient bytes per header never paid on the measured extents).
- Intra-block-copy × superres is SPEC-BARRED (§5.9.2 reads
  `allow_intrabc` only when `UpscaledWidth == FrameWidth`), so the
  composition never appears on either side of the crate.
- Conformance-grade encoding lives on
  `encoder::encode_key_frame_yuv{420,}{,_with_q}` /
  `encoder::encode_gop_yuv{420,}{,_with_q,...}` /
  `encoder::encode_pyramid_gop_yuv{420,}_with_q` /
  `encoder::encode_adaptive_gop_yuv{420,}_with_q` — every §6.4.1
  (bit depth, chroma format) pairing.

## Module layout

`obu`, `sequence_header`, `frame_header`, `tile_info`,
`uncompressed_header_tail`, `symbol_decoder`, `cdf`, `scan`,
`transform`, `qmatrix`, `superres`, `loop_filter`, `loop_restoration`,
`cdef`, `film_grain`, `inter_pred`, and the `decoder` / `encoder`
pipelines.

## Fuzzing

`fuzz/` holds three `cargo fuzz` libFuzzer targets, each driving only
this crate's public Rust API (no external decoder / oracle linked):

- `decode` — attacker bytes through the `SpecDecodeSession` IVF entry
  (IVF → OBU walk → headers → tile / partition / reconstruction),
  capped at a 2^20-luma-sample picture ceiling so the harness's RSS
  limit measures panics rather than legitimate frame storage (the
  library default is `decoder::MAX_PICTURE_SIZE`, Annex A's largest
  defined `MaxPicSize`).
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
