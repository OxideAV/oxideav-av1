# oxideav-av1

Pure-Rust **AV1** (AOMedia Video 1) parser and partial decoder scaffold.
Parses the full bitstream envelope through `tile_info()`, splits
`OBU_TILE_GROUP` payloads into per-tile byte ranges, and ships the pixel
primitives (range coder, inverse DCT, DC/V/H intra predictors) that the
remaining coefficient-decode work will drive. Pixel reconstruction
itself is not implemented yet. Zero C dependencies.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core  = "0.0"
oxideav-codec = "0.1"
oxideav-av1   = "0.0"
```

## What works

The crate follows the AV1 Bitstream & Decoding Process Specification
(2019-01-08). Sections are cited in both source and error messages.

- **OBU walk (§5.3)** — all 10 OBU types, low-overhead and sized
  framing, temporal+spatial IDs, LEB128 sizes.
- **Sequence header OBU (§5.5)** — profile, reduced still picture
  header, operating points (count + tier + IDC + decoder model), full
  color config (bit depth, subsampling, primaries, transfer, matrix,
  sample position, range), timing info, decoder model info, all
  feature-enable flags.
- **Frame header OBU (§5.9) including tile_info() (§5.9.15)** — frame
  type, show / showable flags, error-resilient mode, screen-content
  tools, frame size (with superres), render size, intrabc, ref frame
  indices, interpolation filter, motion-mode / ref-frame-mvs switches,
  `disable_frame_end_update_cdf`, and the tile grid: uniform or
  non-uniform spacing, `TileCols` / `TileRows`, `MiColStarts[]` /
  `MiRowStarts[]`, `context_update_tile_id`, `TileSizeBytes`.
- **`OBU_FRAME` split (§5.10)** — `parse_frame_obu()` returns the
  parsed frame header plus the byte slice carrying the embedded
  `tile_group_obu()`.
- **Tile group OBU (§5.11)** — `tile_start_and_end_present_flag`,
  `tg_start` / `tg_end`, `byte_alignment()` padding, per-tile
  `tile_size_minus_1` size prefix, last-tile remainder. Returns a
  `Vec<TilePayload>` with `(tile_row, tile_col, offset, len)` pointers
  into the source buffer.
- **`av1C` extradata** — AV1CodecConfigurationRecord as used by MP4
  (`av01` sample entry) and Matroska/WebM (`V_AV1` codec private),
  including the embedded sequence-header config OBU.
- **Registry-side `Decoder`** — `make_decoder()` + `Av1Decoder`
  surface headers via `sequence_header()`, `last_frame_header()`,
  `last_tile_payloads()`. `send_packet()` parses through the tile
  framing then returns `Error::Unsupported` pointing at the exact
  clause that blocks pixel reconstruction.
- **Pixel primitives** — range / arithmetic symbol decoder (§4.10.4 +
  §9.3), 4x4 and 8x8 inverse DCT-DCT (§7.7), DC_PRED / V_PRED / H_PRED
  (§7.11.2).

## Usage

```rust
use oxideav_av1::{parse_frame_obu, parse_sequence_header, iter_obus, ObuType};

let obus: Vec<u8> = /* raw OBU stream, e.g. from IVF */;
let mut sh = None;
for obu in iter_obus(&obus) {
    let obu = obu?;
    match obu.header.obu_type {
        ObuType::SequenceHeader => {
            sh = Some(parse_sequence_header(obu.payload)?);
        }
        ObuType::Frame => {
            let seq = sh.as_ref().expect("seq before frame");
            let (fh, tg_payload) = parse_frame_obu(seq, obu.payload)?;
            let ti = fh.tile_info.as_ref().unwrap();
            println!(
                "frame {}x{} {:?} tiles={}x{} tg_payload={} bytes",
                fh.frame_width, fh.frame_height, fh.frame_type,
                ti.tile_cols, ti.tile_rows, tg_payload.len(),
            );
        }
        _ => {}
    }
}
# Ok::<(), oxideav_core::Error>(())
```

For per-tile byte ranges:

```rust
use oxideav_av1::{parse_tile_group_header, split_tile_payloads};

let tgh = parse_tile_group_header(tg_payload, ti)?;
let tiles = split_tile_payloads(tg_payload, ti, &tgh)?;
for t in &tiles {
    let bytes = &tg_payload[t.offset..t.offset + t.len];
    // `bytes` is the compressed body of tile (t.tile_row, t.tile_col).
}
# Ok::<(), oxideav_core::Error>(())
```

## Palette mode — round 26

Round 26 finalizes the §5.11.46 / §5.11.49 palette path. The
intra-only key-frame palette pipeline that landed in r10 (palette
colour list decode + cache-aware merge, §5.11.50 anti-diagonal
colour-index map decode with the 3-neighbour ColorContextHash, and
§7.11.4 `predict_palette` reconstruction) is now also driven from
the intra-within-inter path inside inter frames. A new
`read_palette_for_intra_within_inter` helper in
`decode/inter_block.rs` runs the spec eligibility gates
(`MiSize >= BLOCK_8X8`, `Block_Width <= 64`, `Block_Height <= 64`,
`allow_screen_content_tools != 0`, `YMode == DC_PRED`) and consumes
the palette syntax bits at the spec-correct position; the decoded
`PaletteBlock` is carried back via a new `palette:
Option<PaletteBlock>` field on `InterBlockInfo` (which switched
from `Copy` to `Clone` to fit the heap-allocated colour map). The
inter-leaf reconstruction site (`decode_inter_leaf_block` in
`decode/superblock.rs`) consumes the palette in the `!is_inter`
branch: `apply_palette_luma` / `apply_palette_chroma` replace the
prediction + residual loop when the plane is palette-coded, and
the per-MI propagation now also stamps `palette_size_y`,
`palette_size_uv`, and `palette_colors_*` so the next block's
`get_palette_cache` neighbour walk picks up the colours correctly.
Sacred invariants intact:
`svtav1_chain_walk_round21_full_pass` still 48/48,
`svt_av1_intra_psnr_vs_reference` unchanged at 9.49 dB,
`palette_screen_fixture_decodes_with_plane_variation` still
passes, and the libdav1d cross-check on `palette_screen.ivf`
reports ~8.56 dB Y-PSNR (well above the 8.0 dB regression floor).

## Partition tree — round 48 (force-split for tiny frames)

Round 48 (workspace task #791) lands two §5.11.4 / §5.11.39 spec-fidelity
fixes uncovered by hand-tracing the lossless 1×1 YUV444 KEY frame in
`y_plane_divergence_match.avif`:

- **§5.11.4 partition force-split** — `decode_partition_node` now
  honours the spec's `hasRows = (r + halfBlock4x4) < MiRows` /
  `hasCols = (c + halfBlock4x4) < MiCols` tests. When both flags are
  false (frames smaller than `halfBlock4x4` luma 4×4 cells in either
  dim) the partition collapses to `PARTITION_SPLIT` WITHOUT a symbol
  read; previously the decoder emitted a phantom multi-symbol
  partition CDF read at every recursion level. The `hasCols`-only
  and `hasRows`-only branches use the §5.11.4 / §9.4 derived
  `split_or_horz` / `split_or_vert` 2-symbol CDFs constructed from
  the per-bsl partition CDF.
- **§5.11.39 sign loop** — moved the Golomb tail extension from the
  reverse `coeff_br` loop into the forward sign loop where the spec
  interleaves it with the per-coefficient `sign_bit` read. No-op for
  TUs that don't saturate; spec-faithful for those that do.

`tests/issue_791_partition_force_split_for_tiny_frames.rs` pins the
post-fix decode output. The §5.11.39 entropy decoder still reads
sign bits for the divergence fixture's luma TU that diverge from
`dav1d`'s reference — closing that gap is the next divergence-closing
work item.

## Sign-bit divergence — round 49 audit (workspace task #796)

Round 49 followed up the round-48 partition fix with a full audit of
the remaining §5.11.39 / §9.4.7 sign-bit divergence on the
`y_plane_divergence_match.avif` luma TU. The luma level magnitudes
(`[4, 0, 0, 0, 3, 0, 0, 0, 6, 1, 0, 0, 1, 1, 0, 0]`) match `dav1d
1.5.3` exactly, the DC sign decodes correctly through the
`dc_sign_cdf[plane=0][ctx=0]` context-coded symbol, but 4 of the 5 AC
`sign_bit L(1)` literal reads come out negative while `dav1d` reads
all 5 as positive. Forcing the AC signs to `+` empirically lifts the
WHT residual at (0, 0) from `2` to `5` and produces the dav1d-matching
`(Y, U, V) = (133, 197, 215)` output via a chroma cascade (chroma
`txb_skip` flips from `1` to `0`).

The audit covered:

- §5.11.39 forward sign loop ordering and `c == 0` gating.
- §9.4.7 `dc_sign_ctx` derivation against `AboveDcContext` /
  `LeftDcContext` neighbour polling.
- `decode_bool(16384)` 50/50 bit read vs the equivalent
  `decode_symbol(&mut [16384, 0, 0])` 2-symbol path (pinned by
  `symbol::tests::decode_bool_and_decode_symbol_two_way_agree`).
- `dc_sign_cdf` wire-format conversion against `Default_Dc_Sign_Cdf`
  forward CDF values.
- §8.2.6 renormalise step including `paddedData ^ (((SV+1)<<bits)-1)`.
- §9.4 CDF adaptation rate `3 + (cdf[N]>15) + (cdf[N]>31) +
  Min(FloorLog2(N), 2)`.
- Default_Scan_4x4 order against §7.16.4 spec table.
- `compute_tx_type` lossless `return DCT_DCT` short-circuit (§5.11.40).
- §5.11.47 `transform_type` `qindex > 0` symbol gating.

None of these surfaced a spec divergence. The remaining divergence is
consistent with an entropy-state delta vs `dav1d` that is too small
to flip any of the 4-way `coeff_base` / `coeff_br` symbols (which sit
on probability mass thresholds far from 0.5) but just large enough to
flip 4 of 5 50/50 literal `sign_bit` reads. Pinning sentinel:
`tests/issue_796_sign_bits_match_dav1d.rs`. Closing this will lift
the round-48 test's Y from `130` to `133` (and U/V from `(128, 128)`
to `(197, 215)`).

## Range-coder trace — round 66 (`rc-trace` feature)

Workspace task #801 added a `rc-trace` cargo feature that lets callers
dump every range-coder operation to JSONL for cross-decoder
comparison. Enable with `--features rc-trace` and point
`OXIDEAV_AV1_RC_TRACE=/tmp/our.jsonl` at the desired sink (omit to
stream to stderr). Each line is `{call_idx, op, rng_in, value_in,
p_or_cdf, result, rng_out, value_out, bit_pos}`. The feature is gated
behind `#[cfg(feature = "rc-trace")]` so the default build pays no
runtime cost. The pinned `divergence.avif` trace lives at
`tests/fixtures/issue_796_rc_trace.jsonl`. Round 66 used it to narrow
the §5.11.39 / §9.4.7 sign-bit divergence to **`call_idx = 27`** —
the first AC `sign_bit L(1)` read entering with `range = 45796,
value = 11884` against the same `range` `dav1d` produces but a
different `value` register; see `tests/issue_796_sign_bits_match_dav1d.rs`
for the three hand-off hypotheses targeting round 67.

## Sign-bit divergence — round 67 audit (workspace task #801)

Round 67 ruled out the three leading round-66 hypotheses by direct
cross-check against the in-tree AV1 spec corpus at
`docs/video/av1/av1-spec.txt`:

- **Hypothesis 1 — `SymbolDecoder::new` `sz` accounting.**
  `split_tile_payloads` emits a 14-byte slice for our fixture
  (frame OBU payload 17 B − uncompressed header 3 B). The
  resulting `SymbolMaxBits = 8 × 14 − 15 = 97` matches §8.2.2
  line 19441 to the bit. The 15-bit init read yields the exact
  `value = 0x7323 = 29475` recorded by the rc-trace `init` line.
  **Falsified — no off-by-one.**
- **Hypothesis 2 — `update_cdf` rate arithmetic at `count == 0`.**
  The spec's forward-form update (lines 19811-19823) was
  analytically transformed to our wire-form (inverse) update and
  cross-verified with two worked examples:
    - `N=4`, `symbol=1`, `rate=5`, forward
      `[10000, 20000, 30000, 32768, 0]` → forward post
      `[9688, 20399, 30086, 32768, 1]`. Wire equivalent matches
      bit-for-bit.
    - `N=2`, `symbol ∈ {0, 1}`, `rate=4` (the AC-sign config) →
      forward post `[17408 | 15360, 32768, 1]`. Wire matches.
  Both checks now ride as
  `symbol::tests::update_cdf_matches_spec_forward_form_worked_example`
  and `update_cdf_2sym_count0_rate4_matches_spec`.
  **Falsified — algorithm is direction-correct, rate-correct.**
- **Hypothesis 3 — `coeff_br_multi` CDF drift at calls 22-25.**
  Each call's `coeff_br_ctx_spec` ctx was recomputed independently
  from the partially-decoded `quants[]` at that scan_idx:
    - Call 22 (scan_idx 2 br#1 at pos=4): ctx=11. ✓
    - Call 23 (scan_idx 1 coeff_base at pos=1): ctx=2. ✓
    - Call 24 (scan_idx 0 coeff_base at pos=0): ctx=0. ✓
    - Call 25 (scan_idx 0 br#1 at pos=0): ctx=2. ✓
  All CDF lookups land on the spec-correct index.
  **Falsified — CDFs are not drifting between us and dav1d.**

Round-67 verdict: the §8.2.6 entropy decoder is spec-compliant.
The ~10.9 k Q15 delta in the value register entering call 27 is
upstream of §8.2.6 — either a spec-table typo not yet bisected
or a context-derivation off-by-one that produces the same chosen
symbols on this fixture (invisible to the rc-trace's symbol-result
gate). Closing the divergence requires **dav1d's internal entropy
trace** for direct call-by-call state comparison; that is the
round-68 plan. Black-box verification:
`dav1d 1.5.3 -i divergence.obu -o /tmp/decoded.yuv` produces
exactly `(133, 197, 215)`, pinned now in
`tests/issue_796_sign_bits_match_dav1d.rs::issue_796_dav1d_reference_yuv_pinned`.

## Sign-bit divergence — round 72 rc-trace tagging (workspace task #801)

Round 72 extended the `rc-trace` feature with a per-call `tag`
field — every wrapper in `CoeffCdfBank` /
`TileDecoder` symbol dispatchers and the bypass-bit sites in
`decode::coeffs::{decode_coefficients, decode_coefficients_spec,
read_golomb}` push a short identifier of the CDF table being
consulted plus its `(q, tx, plane, ctx)` tuple via
`crate::symbol::set_rc_trace_tag(&str)` immediately before the
symbol read. The pinned fixture
`tests/fixtures/issue_796_rc_trace.jsonl` now carries the labelled
sequence, and the regression test asserts that the AC sign reads
(calls 27..=31) plus the filter-intra reads (calls 4..=5) keep
their tags so a future refactor that takes them off the
instrumented path is loud.

Round-72 also confirmed via a one-shot frame-header probe that
every parsed §5.5 / §5.9 flag on this fixture matches a hand-
decoded bitstream walk — `enable_filter_intra=1`,
`disable_cdf_update=0`, `error_resilient_mode=1` (spec-forced for
KEY+show_frame), `tx_mode=Only4x4` (coded_lossless),
`base_q_idx=0`. This falsifies the round-68 hypothesis #3
(frame-header field misparse). The remaining unfalsified attack
surface is a renormalise-step bit-padding divergence between the
two decoders, or a single CDF-entry Q15 typo that shifts the
value register without crossing any symbol-pick boundary on the
high-skew CDFs through call 26. Round 73 needs a
dav1d-side state log for the divergence fixture; nothing in the
in-tree spec corpus distinguishes the two trajectories.

## Inverse transform — round 47 (lossless WHT audit)

Round 47 (workspace task #786) audited the §7.7.4 / §7.13.2.10
lossless WHT residual path against the
`y_plane_divergence_match.avif` 1×1 lossless YUV444 KEY frame.
Findings:

- The `inverse_2d_spec_lossless` dispatcher in `transform/mod.rs`
  applies the §7.7.4 row pass with `shift = 2` and column pass with
  `shift = 0` (§7.13.2.10), `rowShift = colShift = 0` between passes,
  and no `Round2()` step before clip-add — spec-correct.
- The lossless dequantiser is `q = DC8[0] = AC8[0] = 4` from the
  §7.12.2 tables, applied as `level * q` per §7.12.3 step (c)
  with `dqDenom = 1` for TX_4X4.
- For the in-bounds Y TU's dequantised buffer
  `[12, 4, 0, 0, 8, -4, 0, 0, -4, 8, 0, 0, 0, 0, 0, 0]`, the spec
  WHT yields `(0, 0) = 2` (hand-traced in
  `iwht4_2d_divergence_y_tu_matches_spec`), matching runtime output
  exactly. `dav1d 1.5.3` and `avifdec` both decode the same OBU to
  `(Y, U, V) = (133, 197, 215)`; oxideav decodes to
  `(130, 128, 128)`. The 3-LSB Y delta and 69 / 87-LSB chroma
  deltas are upstream of the WHT — the §5.11.39 / §9.4 coefficient
  entropy decoder reads different `level` values for the same
  range coder state, tracked as the next divergence-closing work
  item.

## Inverse transform — round 25

Round 25 wires the `inter_tx_type` symbol read into the inter Y site.
Three default CDFs from spec §9.4 join `cdfs/extra.rs` —
`DEFAULT_INTER_EXT_TX_CDF_SET1[2][17]`, `…SET2[13]`, and
`…SET3[4][3]` — each in the wire-form `32768 - cdf_spec[i]`
survival convention so the range coder hot loop indexes
without a subtraction.  `TileDecoder::decode_inter_tx_type(w, h,
reduced_tx_set)` consults them through the existing
`ext_tx_set_for_inter` selector and `inter_tx_type_for` inverse
map (the r24 groundwork in `decode/tx_type_map.rs`), then
`inter_luma_residual_tu` in `decode/superblock.rs` graduates
from the previous hard-coded `TxType::DctDct` to the
symbol-driven type with the same defensive `Unsupported ->
DctDct` fallback the intra Y site uses.  Inter chroma stays at
`DctDct` for this round — §5.11.40 derives chroma in
`is_inter == 1` from the corresponding luma `TxTypes[y4][x4]`
via `is_tx_type_in_set`, which needs the `TxTypes[][]` array
that isn't currently tracked.  Inter P-frame Y-PSNR vs
libdav1d / libaom on the canonical
`/tmp/av1-inter.ivf` (testsrc 128×128, --cq-level=50) moves
9.49 dB → 10.31 dB (+0.82 dB).  The SVT-AV1 chain stays
48/48 (`svtav1_chain_walk_round21_full_pass`) and the intra
sacred-invariant (`svt_av1_intra_psnr_vs_reference`) is
unchanged.

## Inverse transform — round 24

Round 24 audited the inter-path migration that landed in r23 and
confirmed both inter call sites in `decode/superblock.rs`
(`inter_luma_residual_tu` at the §5.11.36 transform-tree leaf, and
the chroma residual loop in `reconstruct_inter_chroma_block`) are
already dispatching through the spec-correct `inverse_2d_spec` entry
point — the r23 commit message and CHANGELOG describing the
migration as "all four intra paths" was a labelling slip; the code
change actually covered five sites, two of them inter. No live
caller of the legacy `inverse_2d` remains in the decode pipeline;
only the transform module's own
`round23_inverse_2d_spec_matches_legacy_for_aligned_squares`
equivalence test still references it.

Round 24 also lands the inter `tx_type` mapping tables in
`decode/tx_type_map.rs`: `inter_tx_type_for(set, raw)` transcribes
`Tx_Type_Inter_Inv_Set{1,2,3}` from spec §6.10.15 verbatim,
`inter_tx_type_set_size(set)` reports CDF cardinality,
and `ext_tx_set_for_inter(tx_w, tx_h, reduced_tx_set)` implements
the inter branch of `get_tx_set` from §5.11.48. The inter sites
themselves still hard-code `TxType::DctDct` until a future round
wires up the `inter_tx_type` CDF reads (§5.11.45) and
`TileInterTxTypeSet{1,2,3}Cdf` default tables — at which point the
inter Y site will also adopt the same defensive
`Unsupported -> DctDct` fallback the intra Y site already uses.
Sacred invariants intact post-audit:
`svtav1_skip_mode_compound_decodes_real_pixels`,
`svtav1_chain_walk_round21_full_pass`, and
`svt_av1_intra_psnr_vs_reference` (9.49 dB) all pass; r24 carries
no PSNR delta because the inter migration was already live since
r23.

## Inverse transform — round 23

The inverse-2D dispatcher ships **two** entry points:

- `transform::inverse_2d_spec` — the **live** path used by
  `decode/superblock.rs` since round 23. Spec-faithful §7.13.3
  implementation: per-shape `Transform_Row_Shift[TX_SIZES_ALL]` applied
  between row and column passes, constant `colShift = 4` after the
  column pass, and the rectangular `Round2(T[j] * 2896, 12)`
  per-element pre-row scale fired only for `|log2W - log2H| == 1`
  (the 2:1 aspect shapes — Tx4x8/Tx8x4/Tx8x16/Tx16x8/Tx16x32/Tx32x16/
  Tx32x64/Tx64x32). The 1:4 / 4:1 shapes (Tx4x16/Tx16x4/Tx8x32/
  Tx32x8/Tx16x64/Tx64x16) and squares correctly skip the 2896 scale
  per spec. Identity 1-D kernels dispatch through the
  `transform::idtx_spec` module that ships the spec magnitudes
  (`Round2(T*5793, 12)` ≈ ×√2 at length 4, ×2 at 8,
  `Round2(T*11586, 12)` ≈ ×2√2 at 16, ×4 at 32) per
  §7.13.2.11/12/13/14, replacing the uniform-`<<= 1` legacy variants
  on the new path. The path drops the `flip_1d` wrapper used by
  `inverse_2d` for FLIPADST kernels: `iflipadst*` already reverses
  its own output, so wrapping pre-flip + post-flip cancelled the
  kernel's reverse and produced `IADST(reverse(input))` instead of
  the spec-equivalent `reverse(IADST(input))`.
- `transform::inverse_2d` — legacy 2D entry that performs row/column
  passes without per-pass round-shifts. Preserved as a reference
  implementation (used by the `inverse_transform_add` smoke-test
  wrapper and as the equivalence target in
  `round23_inverse_2d_spec_matches_legacy_for_aligned_squares`,
  which pins byte-for-byte agreement with the spec path on Tx4x4
  and Tx32x32 squares).

Round-23 caller migration: all four `decode/superblock.rs` call sites
(intra Y DCT-only path, intra chroma DCT-only path, intra Y arbitrary
TX_TYPE with DctDct fallback, intra chroma DCT-only chroma path) now
dispatch through `inverse_2d_spec`. The legacy
`residual_shift`/`round_shift` post-2D scaling that compensated for
`inverse_2d`'s lack of per-pass shifts is removed in tandem — the
spec path bakes those shifts into the kernel itself, so leaving the
post-call shift in would double-shift and crater PSNR. Sacred
invariants (`svtav1_skip_mode_compound_decodes_real_pixels`,
`svtav1_chain_walk_round21_full_pass`, `svt_av1_intra_psnr_vs_reference`)
all pass post-migration. Intra-fixture luma PSNR vs the libdav1d
reference moved 8.85 dB → 9.49 dB on `tests/fixtures/svt_av1_intra_64.ivf`
(slight improvement; the headroom is bounded by upstream palette /
lookahead / edge-filter work still pending).

The transform module now carries 13 unit tests covering: the
row-shift table verbatim; spec coverage of every TX_TYPE × TX_SIZE
pair the bitstream may carry (159 of 323 — full INTER_1 set on
Sqr_Up≤16, INTER_3 on Sqr_Up=32, DCTONLY on Sqr_Up=64); the
rectangular 2896 trigger gate; DC-constant reconstruction across all
14 rectangular shapes; spec IDTX magnitudes; the
iflipadst-equals-reverse-iadst invariant; the spec-disallowed kernel
rejection set (Adst@32/64, FlipAdst@32/64, Idtx@64, Wht@non-4); and
(round 23) the legacy/spec equivalence on Tx4x4 + Tx32x32 with
non-trivial coefficients plus a compile-time witness pinning the
spec entry-point signature that `decode/superblock.rs` imports.

## SVT-AV1 chain status

Round 21 (commit pending) lands the §5.9.2 inter-branch ORDER fix:
`frame_size()` and `render_size()` are now parsed AFTER the
`ref_frame_idx[]` loop, matching the spec exactly. Previously
they ran before it, which mis-aligned the bitstream by ~13 bits on
every non-short-signaling inter frame (the bit COUNT happened to
match the `frame_size_override=0` / `enable_superres=false` case but
the bit POSITIONS shifted because spec's `frame_refs_short_signaling=0`
path consumes a 21-bit per-slot ref_frame_idx loop before render_size).
The mis-alignment caused 10/48 SVT-AV1 chain frames to mis-interpret
tile_group bits as `gm_params` type bits — surfacing as
`parse_global_motion_params` AFFINE overruns or `parse_lr_params`
`out of bits`. After r21, `svtav1_chain_walk` reaches **48/48** Frame
OBUs (was 38/48 in r20). The `AV1_TRACE_BITS=1` env-gated diagnostic
is retained for future bisects.

## What's missing

`Av1Decoder::receive_frame()` returns `Error::Unsupported`. To produce
pixels the following still need to land (each error message in the
crate names its §ref):

- **Default CDF tables** (§9.4.1 / §9.4.2) — ~9 KB of static probability
  tables required by every symbol decode.
- **Partition quadtree** walk (§5.11.4) and the remaining partition
  shapes beyond `PARTITION_NONE`.
- **`decode_block()` per-symbol syntax** — `skip`, `tx_size`, `tx_type`,
  `intra_y_mode`, `uv_mode`, angle deltas, CFL, coefficient decode
  (§5.11.5 – §5.11.39). Palette is wired since r10 (key-frame intra)
  and r26 (intra-within-inter).
- **Dequantisation + AC/DC scaling** (§7.12).
- **Remaining inverse transforms** — 16x16 / 32x32 / 64x64 sizes, ADST,
  flipped ADST, identity, WHT, and the mixed (V*/H*) combinations
  (§7.7).
- **10 remaining intra predictors** — D45 / D67 / D113 / D135 / D157 /
  D203, SMOOTH / SMOOTH_V / SMOOTH_H, PAETH, plus CFL chroma (§7.11.2).
- **Inter prediction** — MV decode, warped / global motion, compound
  weighted / masked / interintra modes, OBMC (§7.11.3).
- **Quantisation, segmentation, delta-q/lf, loop filter, CDEF, loop
  restoration, tx_mode, frame_reference_mode, skip_mode, global motion,
  film grain** sub-sections of `uncompressed_header()` past
  `tile_info()` (§5.9.16 onwards).
- **Deblock / CDEF / LR / superres / film grain** post-processing
  passes (§7.14 – §7.20).
- **Cross-frame reference state** — `frame_size_with_refs()` (§5.9.7)
  parses its 7 `found_ref` bits per r21 but still returns
  `Error::Unsupported` when any are set, since `RefFrameWidth[]` /
  `RefFrameHeight[]` per-DPB-slot tracking is pending. The found_ref-
  all-zero fall-through (regular `frame_size + render_size`) works.
- **10+ bit-depth decode** — the pixel primitives are `u8`-only; 10 /
  12-bit paths land together with the coefficient decode.

## Encoder — round 1 (intra-only headers)

The `encoder` module ships the AV1 bitstream framing + sequence /
frame-header writers behind an `Av1Encoder` that implements
`oxideav_core::Encoder`. Round 1 scope:

- OBU framing (header + leb128 size prefix, TD + SH + FRAME).
- Sequence Header — profile 0 (8-bit 4:2:0),
  `reduced_still_picture_header = 1`. CDEF / restoration / superres /
  film grain / 128×128 SB all off.
- Frame Header — KEY / show / error_resilient. Single tile, single
  64×64 superblock. Configurable `base_q_idx`.
- Encoder envelope: width / height multiples of 8, ≤ 64×64, Yuv420P.

Tile-group payload is a 16-byte zero stub — the recursive arithmetic
coder + partition / mode / coefficient encoder is round 2+ work.
Round 2 needs to land:

1. Forward range coder (carry-out byte queue + `od_ec_enc_done` tail).
2. Partition / intra-mode / TX-type emit (DC_PRED + DCT_DCT only for
   round 2 first cut).
3. Forward 4×4 DCT pinned against `inverse_2d_spec` for `±1 LSB`
   roundtrip.
4. Coefficient entropy emit (`txb_skip` / `eob_pt` /
   `coeff_base_*` / signs / Golomb-Rice tail).
5. dav1d cross-validation.

Self-roundtrip via the in-tree decoder gets header parse end-to-end;
the tile group decode surfaces `Error::Unsupported` until the entropy
coder ships.

## Encoder — round 3 (transform + coefficient scaffolding)

Round 3 adds the forward-transform stack and coefficient entropy emitter:

- **Forward DCT kernels**: `fdct4`/`fdct8`/`fdct16`/`fdct32` (1-D) and the
  corresponding 2-D `fdct4x4`/`fdct8x8`/`fdct16x16`/`fdct32x32` wrappers plus
  a generic `fdct2d` dispatcher. `fdct4x4` round-trips through
  `inverse_2d_spec(Tx4x4)` within ±1 LSB; `fdct8x8` DC within ±2 LSB.
  The 16×32-point kernels compile and produce correct DC but their
  AC precision is deferred to round 4 (pre-shift / interleave audit needed).
- **`CoeffCdfBankEnc`** — full encoder-side CDF bank (mirrors `CoeffCdfBank`
  field-for-field). All symbol emitters implemented.
- **`encode_coefficients`** — full forward coefficient stream emitter:
  txb_skip, eob, base-level, br-level refinement, Golomb-Rice tail, signs.
  Roundtrip-tested against `decode_coefficients` for 4×4 blocks.

## Encoder — round 40 (coefficient stream wired into tile leaf)

Round 40 lands the deferred round-3 followup: `encode_coefficients` is now
called from the tile-group writer per plane. The new
`tile::write_tile_group_intra_64(seq, base_q_idx)` emits **`skip = 0`** at
the block level and walks the decoder's per-plane coefficient path:

- **Non-lossless** (`base_q_idx > 0`): one 64×64 luma TU (DCT_DCT
  implicit because area > 32×32) plus two 32×32 chroma TUs (one each for
  U / V at 4:2:0).
- **Coded-lossless** (`base_q_idx == 0`): 256 × 4×4 luma TUs (the spec
  pins luma block TxSize to `Tx4x4` in lossless) plus the same 32×32
  chroma TUs (chroma sizing is independent of the block-level TxSize in
  this implementation).

Each TU currently emits an all-zero residual, so every plane sends a
`txb_skip = true` symbol and the reconstructed output equals the round-3
DC_PRED-no-neighbours mid-grey 128 fill. The bitstream is now
`>= round-3` size with the entropy state diverged: the decoder reads three
extra `txb_skip` symbols per block before short-circuiting, exercising the
full non-skip leg. Self-roundtrip via `Av1Decoder` is pinned at 32×32 and
64×64.

Round 40 also fixed a latent `uv_mode_cdf` mis-index that desynced the
encoder's range coder on 64×64 streams (the round-3 writer hard-coded
`cfl_idx = 1`; the decoder picks `cfl_idx = 0` for blocks where
`max(bw, bh) > 32`). `dav1d` would still reject the round-3 stream
end-to-end (multiple downstream symbols absent), but the corrected
`cfl_allowed` gating is a prerequisite for round-41+ production-quality
emit. Round 41 will replace the all-zero residual with a quantised
forward-DCT residual and wire the larger-than-4×4 transform sizes already
shipped in `encoder::transform`.

## Encoder — round r-next (dav1d 1.5.x cross-decode unblock)

Round r-next lands the actual `dav1d` external-decoder green. Before
this round the cross-decode test
(`tests/encode_roundtrip.rs::round3_dav1d_self_decode_64x64_keyframe`)
soft-skipped on every dimension because dav1d returned EINVAL during
frame parse. Two interlocking fixes plus a router change unblock every
single-SB square frame from 8×8 to 64×64 (with `dav1d --strict 0` —
strict-mode level conformance is a separate followup):

- **Per-bsl partition emit (§5.11.4 force-split mirror)** —
  `tile::leaf_bsl_ctx_for_frame` returns the actual `bsl_ctx` where
  the decoder's `decode_partition_node` recursion settles for a
  single-SB frame. For a 64×64 SB at top-left, the spec test
  `(c_mi + half_block_4x4) < MiCols` decides whether the partition
  symbol is read; below 33×33 it force-splits silently and recurses
  down. The encoder previously hard-coded `partition_cdf[12]`
  (Block64x64) regardless of dimensions, desyncing dav1d for every
  frame ≤ 32 × 32.
- **Spec-correct chroma `txb_skip_ctx` (§5.11.39 / §9.4 `all_zero`)**
  — `coeffs::isolated_txb_skip_ctx` returns 0 for luma full-TU
  isolated blocks and 7 for chroma. `encode_coefficients` previously
  hard-coded `ctx = 0` for both planes, producing wire bits that read
  as a different probability bin under the chroma CDF.
- **Production route → round-3 skip path** — `write_keyframe_stream`
  now calls `write_tile_group_skip_intra_64` (block-level `skip = 1`,
  no coefficient symbols). The round-40 non-skip writer
  (`write_tile_group_intra_64`) carries enough additional spec-context
  derivations beyond the chroma `txb_skip_ctx` fix that dav1d still
  rejects its 64×64 / Tx64x64 luma + Tx32x32 chroma walk; that writer
  stays available to standalone tests for follow-up hardening.

The hardened
`tests/encode_roundtrip.rs::round_r_next_dav1d_decodes_every_single_sb_square_size`
sweeps every dimension × `base_q_idx ∈ {0, 100, 200}` through
`dav1d --strict 0 --demuxer section5` and asserts a valid 4:2:0 YUV
output. The pre-existing `round3_dav1d_self_decode_64x64_keyframe`
test now `panic!`s on non-zero dav1d exit instead of soft-skipping.

Round-(r-next + 1) followups:
- Annex-A.3 MinCompressedSize padding so `dav1d --strict 1` (default)
  also accepts. A 64×64 4:2:0 8-bit frame at any seq_level_idx ≤ 6
  needs ≥ 768 bytes per the level table; our 20-byte still streams
  trip the compliance check.
- Restore the round-40 non-skip path: complete the §5.11.39
  context-derivation wiring in `encode_coefficients` so dav1d accepts
  Tx64x64 luma + Tx32x32 chroma walks.
- Force-split / split_or_horz / split_or_vert emits for non-square
  frames (currently `dim_x != dim_y` self-roundtrips but dav1d
  rejects).

## Codec ID

Registered as `"av1"`. The capability tag is `"av1_sw_parse"` —
surfacing in `oxideav list` style output that this build parses the
bitstream envelope but does not yet reconstruct pixels.

## License

MIT — see [LICENSE](LICENSE).
