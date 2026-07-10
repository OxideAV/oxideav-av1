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
grain. `tests/fixture_conformance.rs` pins 44 streams byte-exact against
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

The `decode_av1(bytes) -> Vec<Frame>` encoder-MIRROR path (the first
of the public entry's two decode paths; the second is the spec-driver
fallback described above) and the crate-public mirror encoders
(`encoder::encode_intra_frame_yuv_dyn` and friends — NOT the public
`encode_av1`, which is conformance-grade as of r409) cover a
constrained intra-only profile:

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

### Conformance-grade encoding (r409, generalised r410)

`encoder::encode_key_frame_yuv420{,_with_q}` is the
**conformance-grade** encode path: it emits real §5.11 keyframe syntax
through the spec-faithful write side (§5.11.7 `intra_frame_mode_info`
with the neighbour-CDF `intra_frame_y_mode`, §5.11.22 `uv_mode` +
§5.11.45 CFL alphas, §5.11.34 per-TU residual with live §8.3.2
contexts), assembled as IVF → TD + SH + the combined §5.10 `OBU_FRAME`.
Scope (r410): 8-bit 4:2:0, dims multiples of 8 in [8, 4096] per axis
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

### Not yet supported

- `SEG_LVL_REF_FRAME` / `SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV` inter
  overrides are implemented per §5.11.14/§5.11.20/§5.11.25 and
  unit-tested, but no conformance stream pins them — the black-box
  encoder's CLI cannot signal those features (`SEG_LVL_ALT_Q` streams
  are pinned byte-exact).
- The historical intra `encode_av1` mirror paths emit non-conformant
  streams (kept for their bit-exact self round-trip through
  `decode_av1`'s mirror arm); conformance-grade encoding lives on
  `encoder::encode_key_frame_yuv420`. Conformant encoding beyond the
  r410 keyframe scope (palette/intrabc leaves, the asymmetric
  HORZ/VERT partition shapes, true bit-accounting rate costs, inter
  P-frames) is the follow-up ladder.

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
