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
chain are gone. Harness: `tests/pyramid_deep_ab.rs`; two streams
pinned: `self-pyr-64x64-q60-len17-deep` and
`self-adaptive-96x80-q60-cut-n13` (corpus 75 total).

### Not yet supported

- `SEG_LVL_REF_FRAME` / `SEG_LVL_SKIP` / `SEG_LVL_GLOBALMV` inter
  overrides are implemented per §5.11.14/§5.11.20/§5.11.25 and
  unit-tested, but no conformance stream pins them — the black-box
  encoder's CLI cannot signal those features (`SEG_LVL_ALT_Q` streams
  are pinned byte-exact).
- The historical intra `encode_av1` mirror paths emit non-conformant
  streams (kept for their bit-exact self round-trip through
  `decode_av1`'s mirror arm); conformance-grade encoding lives on
  `encoder::encode_key_frame_yuv420` /
  `encoder::encode_gop_yuv420{,_with_q,_with_q_seg}` /
  `encoder::encode_pyramid_gop_yuv420{,_with_q}` /
  `encoder::encode_adaptive_gop_yuv420_with_q`. Conformant encoding
  beyond the r424 scope (per-segment lossless mixing, intrabc
  hash-match DV search + rect / clipped palette leaves, the §5.11.46
  signed-delta V-plane arm — the screen-content polish ladder item)
  is the follow-up ladder.

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
