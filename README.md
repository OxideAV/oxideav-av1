# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec.

## Status — 2026-05-28 (round 213)

Round 213 extends `encoder::coefficients` with the **§5.11.39
per-coefficient base-level chain writers** — `coeff_base_eob` /
`coeff_base` / `coeff_br`. Each is the inverse of one §8.2.6 `S()`
the §5.11.39 reverse-scan loop inside `PartitionWalker::coefficients`
issues per coded coefficient.

`encoder::coefficients` — three new pure, stateless writers
(re-exported from `encoder`):

* **`write_coeff_base_eob(writer, cdfs, sym, tx_sz_ctx, ptype, ctx)`**
  — inverse of the §5.11.39 line-60 `coeff_base_eob S()` against
  `TileCoeffBaseEobCdf[txSzCtx][ptype][ctx]` (§8.3.2 p.376). One
  3-symbol §9.4 alphabet write (`sym ∈ 0..=2`, level = `sym + 1`).
  `ctx` is `0..SIG_COEF_CONTEXTS_EOB = 4` — the §8.3.2 reduction
  `get_coeff_base_ctx(..., is_eob = true) - SIG_COEF_CONTEXTS +
  SIG_COEF_CONTEXTS_EOB` is already exposed as the public helper
  `cdf::get_coeff_base_eob_ctx`.
* **`write_coeff_base(writer, cdfs, sym, tx_sz_ctx, ptype, ctx)`** —
  inverse of the §5.11.39 line-63 `coeff_base S()` against
  `TileCoeffBaseCdf[txSzCtx][ptype][ctx]` (§8.3.2 p.371). One
  4-symbol §9.4 alphabet write (`sym ∈ 0..=3`, level = `sym`). `ctx`
  is `0..SIG_COEF_CONTEXTS = 42` — the public helper
  `cdf::get_coeff_base_ctx(..., is_eob = false)` derives it from the
  running `Quant[]` array (the same array the decoder side updates as
  it walks the reverse-scan loop).
* **`write_coeff_br(writer, cdfs, sym, tx_sz_ctx, ptype, ctx)`** —
  inverse of one §5.11.39 line-67 `coeff_br S()` against
  `TileCoeffBrCdf[Min(txSzCtx, TX_32X32)][ptype][ctx]` (§8.3.2 p.378).
  One `BR_CDF_SIZE = 4`-symbol §9.4 alphabet write (`sym ∈ 0..=3`,
  `sym = BR_CDF_SIZE - 1 = 3` keeps the chain going). `ctx` is
  `0..LEVEL_CONTEXTS = 21` — the public helper `cdf::get_br_ctx`
  supplies it. One write per call; the chain-stacker that bounds at
  `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) = 4` iterations lives in the
  follow-on arc.

Stateless on purpose, same surface contract as r212: every writer takes
its §8.3.2 ctx values as inputs and the §8.3 CDF adaptation runs in
lockstep with the matching `PartitionWalker::coefficients` reader.

14 new lib tests (1201 → 1215), bit-exact round-trips encode →
`SymbolWriter` → decode through small read shims wrapping
`coeff_base_eob_cdf` / `coeff_base_cdf` / `coeff_br_cdf`. Coverage:
per-writer round-trips at the zero / chain-continues sentinel /
upper-ctx edges (`SIG_COEF_CONTEXTS_EOB - 1`, `SIG_COEF_CONTEXTS - 1`,
`LEVEL_CONTEXTS - 1`); the `coeff_br` `Min(txSzCtx, TX_32X32)` clamp
exercised at `tx_sz_ctx = TX_64X64 = 4`; out-of-range guards
(`sym >= alphabet`, `ctx >= bound`) returning
`Error::PartitionWalkOutOfRange`; and a hand-built two-coefficient
driver-shape integration test that drives the §8.3.2 helpers
identically on both sides off the same `Quant[]` (`coeff_base_eob = 0`
at `c = eob - 1`, then `coeff_base = 3` + `coeff_br = 0` at `c = 0`).

Out of scope this arc (next): the `golomb_length_bit` /
`golomb_data_bit` magnitude tail for coefficients above
`NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`; the §5.11.39 driver loop
composing all of the above into the reverse + forward-scan composite;
§5.11.4 partition decision-tree writer; inter-arm mode_info writers
(§5.11.18 dispatcher composite, §5.11.23 `inter_block_mode_info`).

## Status — 2026-05-28 (round 212)

Round 212 lands the **first §5.11.39 coefficient-encode primitives** —
the encoder counterparts to the three opening §8.2.6 `S()` reads
inside `PartitionWalker::coefficients` (`txb_skip` / `eob_pt` /
`dc_sign`). The larger §5.11.39 reader body (per-coefficient
`coeff_base{_eob}` / `coeff_br` chain and the `golomb_*` magnitude
tail) stays deferred to the follow-on arc.

`encoder::coefficients` — three pure, stateless writers (re-exported
from `encoder`):

* **`write_txb_skip(writer, cdfs, all_zero, tx_sz_ctx, ctx)`** —
  inverse of the §5.11.39 line-13 `all_zero S()` against
  `TileTxbSkipCdf[txSzCtx][ctx]` (§8.3.2 p.371). One binary symbol.
* **`write_eob_pt(writer, cdfs, eob, tx_size, tx_class, plane,
  is_inter)`** — inverse of the §5.11.39 lines 19–55 EOB block: the
  `eob_pt_{16,32,64,128,256,512,1024}` selector S() plus the optional
  `eob_extra` S() and the raw `eob_extra_bit` L(1) loop. Solves for
  `eobPt` from the target `eob` via the §5.11.39 line-39 base-eob
  inverse (`eobPt = floor_log2(eob - 1) + 2` for `eob ≥ 3`), then
  emits the `(eobPt - 2)`-bit residue MSB-first against the
  contextual `eob_extra` row + the raw L(1) loop, in lockstep with
  the decoder.
* **`write_dc_sign(writer, cdfs, dc_sign, plane, ctx)`** — inverse of
  the §5.11.39 line-564 `dc_sign S()` against
  `TileDcSignCdf[ptype][ctx]` (§8.3.2 p.377). Caller honours the
  `Quant[0] != 0` gate before invoking.

Stateless on purpose, same surface contract as `block_mode_info`:
every writer takes its §8.3.2 ctx values as inputs. The follow-on arc
lands the `coeff_base{_eob}` (four- / three-symbol §9.4 alphabets) +
`coeff_br` per-coefficient writers and the `golomb_length_bit` /
`golomb_data_bit` magnitude tail for coefficients above
`NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14`.

21 new lib tests (1180 → 1201), all bit-exact round-trips: encode
through `SymbolWriter`, decode back through small `read_*` shims that
mirror the §5.11.39 reader's first three `S()` reads (the full
reader also consumes the coeff_base / coeff_br chain we don't emit
yet). Coverage: `txb_skip ∈ {0, 1}` at `(0, 0)` and the upper grid
`(TX_SIZES - 1, TXB_SKIP_CONTEXTS - 1)`; `eob` ∈ `{1, 2, 3, 8, 16}`
on `TX_4X4` / `TX_8X8` (eobMultisize ∈ {0, 1}; refinement bit-widths
0 → 3); `eob = 5` on `TX_16X16` chroma intra with `TX_CLASS_HORIZ`
(exercising the `ptype = 1` axis + the non-`TX_CLASS_2D` ctx flip);
`eob = 17` on `TX_32X32` inter (eobMultisize = 3); `dc_sign ∈ {0,
1}` on luma `ctx = 0` and chroma `ctx = 2`; plus a sequenced
`txb_skip = 0` → `eob_pt(4)` round-trip confirming §8.3 CDF
adaptation stays in lockstep across multiple writer calls.
Caller-bug guards (`eob = 0`, `tx_size >= TX_SIZES_ALL`,
alphabet-overflow `eob`, out-of-range ctx/symbol) surface as
`Error::PartitionWalkOutOfRange`.

Out of scope this arc (next): per-coefficient `coeff_base{_eob}` +
`coeff_br` writers and the `golomb_*` magnitude tail; §5.11.39
driver loop composing all of the above with `Quant[]`-aware §8.3.2
`get_coeff_base_ctx` / `get_coeff_base_eob_ctx` / `get_br_ctx`
plumbing; §5.11.4 partition decision-tree writer; inter-arm
mode_info writers (§5.11.18 dispatcher composite, §5.11.23
`inter_block_mode_info`).

## Status — 2026-05-28 (round 211)

Round 211 lands the **first per-block syntax writers** on top of the
r210 `tile_group_obu` framing — the encoder counterparts to
`PartitionWalker::decode_skip` / `decode_intra_segment_id` /
`decode_intra_frame_y_mode` / the `y_mode` + `uv_mode` arm of
`decode_intra_block_mode_info`. Intra arm only: no inter mode_info,
no MV encode, no partition split, no coefficient encode (next arcs).

`encoder::block_mode_info` — five pure, stateless writers (re-exported
from `encoder`):

* **`write_skip(writer, cdfs, skip, ctx, seg_skip_active)`** — inverse
  of §5.11.11 `read_skip`. The `seg_skip_active` arm (`SegIdPreSkip &&
  seg_feature_active(SEG_LVL_SKIP)`) emits no bits and demands
  `skip == 1`; otherwise writes one §8.2.6 `S()` against
  `Default_Skip_Cdf[ctx]`. `ctx` is the §8.3.2 sum-of-neighbour
  derivation through `crate::cdf::skip_ctx`.
* **`write_intra_segment_id(writer, cdfs, segment_id, skip, pred, ctx,
  segmentation_enabled, last_active_seg_id)`** — inverse of §5.11.8
  `intra_segment_id` and its §5.11.9 `read_segment_id` dispatch. The
  `segmentation_enabled = false` and `skip != 0` arms emit no bits;
  the active arm solves `neg_deinterleave(diff, pred, max) ==
  segment_id` by O(max) search over the §5.11.9 bijection and writes
  the recovered `diff` against `TileSegmentIdCdf[ctx]`.
* **`write_intra_frame_y_mode(writer, cdfs, y_mode, abovemode_ctx,
  leftmode_ctx)`** — inverse of §5.11.7 line 13 `intra_frame_y_mode`
  S(). The §8.3.2 ctx (`Intra_Mode_Context[neighbour]`) is
  caller-derived via `crate::cdf::intra_mode_ctx`; the writer drives
  the keyframe's `TileIntraFrameYModeCdf[above][left]` row.
* **`write_y_mode(writer, cdfs, y_mode, size_group_ctx)`** — inverse
  of §5.11.22 line 3 `y_mode` S() (the inter-frame intra-block path,
  `Size_Group[MiSize]` ctx).
* **`write_intra_uv_mode(writer, cdfs, y_mode, uv_mode, cfl_allowed)`**
  — inverse of §5.11.22 line 6 `uv_mode` S(). Caller selects the
  CFL-allowed / CFL-not-allowed row via
  `crate::cdf::cfl_allowed_for_uv_mode`.

Stateless on purpose: mirroring `SymbolWriter::write_symbol`'s
caller-supplied CDF slice pattern, every writer takes its §8.3.2
context indices as inputs rather than a `PartitionWalker` reference.
Callers derive ctx through the public `cdf::` helpers exactly as the
decoder side does — neighbour state lives in the caller's own grid
(e.g. a shared `PartitionWalker` instance, or a custom encoder-side
tracker).

13 new lib tests (1167 → 1180), all bit-exact round-trips: encode
through `SymbolWriter`, then decode through the matching
`PartitionWalker::decode_*` methods on a fresh walker + default CDFs.
Coverage:

* `write_skip`: seg-short-circuit no-bit arm, `skip = 0` at origin
  with §8.3 adaptation engaged on both sides, `skip = 1` at origin,
  caller-bug rejection of `skip != 1` on seg-arm (4 tests).
* `write_intra_segment_id`: disabled no-bit arm, §5.11.9 skip
  short-circuit (segmentation enabled but `skip == 1` ⇒ `segment_id
  = pred`, no bits), else-branch `neg_deinterleave` round-trip at
  origin with `pred = 0` (3 tests).
* `write_intra_frame_y_mode`: `DC_PRED` at origin, `V_PRED`
  (directional, non-DC) at origin, caller-bug rejection of `y_mode
  >= INTRA_MODES` (3 tests).
* `write_y_mode` + `write_intra_uv_mode`: composed `(DC_PRED,
  DC_PRED)` on `BLOCK_16X16` through `decode_intra_block_mode_info`
  (CFL-allowed selector, palette + filter-intra gates off); monochrome
  (`has_chroma = false`) y_mode-only path (2 tests).
* Composed `{skip, intra_segment_id, y_mode, uv_mode}` smoke test in
  §5.11.7 order at frame origin on `BLOCK_8X8` (1 test).

Out of scope this arc (next): §5.11.4 partition decision-tree writer;
first §5.11.39 coefficient-encode primitives (`txb_skip` / `eob` /
`coeff_base`) so emitted tile payloads decode to real pixel data;
inter-arm mode_info writers (§5.11.18 dispatcher composite,
§5.11.23 `inter_block_mode_info`); §5.11.42 `intra_angle_info_y`
inverse for directional Y modes; §5.11.46 `palette_mode_info` inverse;
§5.11.24 `filter_intra_mode_info` inverse; §5.11.56 `read_cdef` /
§5.11.12-13 delta inverses.

## Status — 2026-05-28 (round 210)

Round 210 lands the **§5.11.1 `tile_group_obu` framing writer** on top
of the r209 [`SymbolWriter`], plus three **composition wrappers** on
the entropy encoder (`write_literal` / `write_ns` / `write_subexp_bool`)
that mirror the inverse `SymbolDecoder` primitives the §5.11 per-block
syntax will use.

`encoder::tile_group_obu` — §5.11.1 body framing:

* **`TilePayload { bytes }`** — one tile's entropy-coded byte run, the
  raw output of `SymbolWriter::finish` (including the §8.2.4 trailing
  zero pads `init_symbol`/`exit_symbol` consume).
* **`TileGroupObu { num_tiles, tile_cols_log2, tile_rows_log2,
  tile_size_bytes, tg_start, tg_end, start_and_end_present, tiles }`**
  — the descriptor; helpers `TileGroupObu::whole_frame(...)` for the
  §6.10.1 single-tile-group default (flag = 0, `tg_start = 0`, `tg_end
  = NumTiles - 1`).
* **`write_tile_group_obu(&obu) -> Vec<u8>`** — emits the §5.11.1 body
  in spec order:
  `tile_start_and_end_present_flag` (`f(1)`, NumTiles > 1 only),
  `tg_start` / `tg_end` (`f(tileBits)` each, flag = 1 only),
  `byte_alignment()`, then per non-last tile `tile_size_minus_1`
  (`le(TileSizeBytes)`) + tile bytes, then the last tile's bytes (no
  size). Wrappable via `write_obu_with_size(.., OBU_TILE_GROUP, body)`
  which §5.3.1 excludes from the §5.3.4 trailer.
* **`parse_tile_group_obu_body(body, ..)`** — companion walker that
  reads the §5.11.1 body back out (used by the round-trip tests; no
  production caller yet — the parser side will land alongside the
  per-block decode arc).

`encoder::symbol_writer` — three new composition wrappers (inverses of
the §8.2.5 / §4.10.10 / §5.9.28 decoder primitives, all composed of
existing `write_bool` / `write_symbol`):

* **`write_literal(n, value)`** — inverse of §8.2.5 `read_literal(n)`:
  emits the low `n` bits of `value` MSB-first via `write_bool`.
* **`write_ns(n, value)`** — inverse of §4.10.10 `NS(n)`: values
  `0..m` written in `w - 1` literal bits; values `m..n` written as
  `(value + m)` in `w` literal bits where `w = FloorLog2(n) + 1` and
  `m = (1 << w) - n`.
* **`write_subexp_bool(num_syms, k, value)`** — inverse of §5.9.28
  `decode_subexp_bool`: walks the `(i, mk)` rung ladder; for each
  rung whose `a`-wide chunk doesn't cover `value`, emits
  `subexp_more_bools = 1` and advances. Once covered, emits either
  the `NS(num_syms - mk)` uniform tail or the `subexp_more_bools = 0`
  + `L(b2)` fixed-width tail with `value - mk`.

15 new lib tests (1152 → 1167), all bit-exact round-trips:

* `write_literal`: width-1..16 grid + concatenation + `n = 0` no-op
  (3 tests).
* `write_ns`: exhaustive value sweep at `n ∈ {1, 2, 3, 4, 5, 7, 8, 9,
  13, 16, 17}` + `n = 1` no-op (2 tests).
* `write_subexp_bool`: grid covering the §5.9.28 immediate-uniform
  branch + tail uniform + ladder-advance + fixed-width tail paths,
  plus a concatenation test (2 tests).
* `tile_group_obu`: single-tile (no flag/size), multi-tile flag = 0,
  multi-tile flag = 1 (`tg_start` / `tg_end` writes),
  `TileSizeBytes = 2` (byte-order check), `TileSizeBytes = 4`
  (max width), `OBU_TILE_GROUP` wrap-and-walk via
  `write_obu_with_size` + `parse_obu`, zero-padded `SymbolWriter`
  payload size accounting, `whole_frame` constructor defaults
  (8 tests).

Out of scope this arc (next): per-block §5.11 syntax — `skip` /
`mode_info` / the first §5.11.39 coefficient-encode primitives so
emitted tile payloads decode to real pixel data; §5.11.4 partition
decision tree; §5.9.7 `frame_size_with_refs()` inverse for the inter
frame size path; §5.9.24 `read_global_param` signed-subexp inverse
for non-IDENTITY refs.

## Status — 2026-05-28 (round 209)

Round 209 lands the **§8.2 entropy *encoder*** (`SymbolWriter`) — the
inverse of the existing `SymbolDecoder` (§8.2.2 `init_symbol` / §8.2.6
`read_symbol` / §8.2.4 `exit_symbol`). This is the foundation every
§5.11 tile-content encode pass sits on top of: without it nothing
inside `tile_group_obu()` can be written.

`encoder::symbol_writer::SymbolWriter`:

* **`write_symbol(symbol, cdf)`** — inverse of §8.2.6
  `read_symbol(cdf)`. Computes the same `cur(s)` / `prev` boundaries
  the decoder uses (`((range >> 8) * (f >> EC_PROB_SHIFT)) >> (7 -
  EC_PROB_SHIFT) + EC_MIN_PROB * (N - s - 1)`), advances `low` to
  the symbol's u-space interval (`offset = range - prev`), shrinks
  `range`, then renormalises by appending `bits = 15 -
  FloorLog2(range)` zero bits to the accumulated `low`.
* **`write_bool(bit)`** — inverse of §8.2.3 `read_bool()`; uses the
  fresh `[1<<14, 1<<15, 0]` CDF and discards the §8.3 adaptation
  per the spec's note.
* **`finish()`** — emits the accumulated `low` bits MSB-first as the
  encoded bytestream. The decoder's `paddedBuf` is the top 15 bits
  of that emission, which (by construction) lies in the encoder's
  recorded `[low, low + range)` interval for every symbol — so the
  decode round-trip is bit-exact.
* `disable_cdf_update` honoured exactly as on the decoder side: when
  `false`, the §8.3 `update_cdf` runs on every `write_symbol` so
  the encoder and decoder evolve identical CDFs along the same
  symbol stream.

The §8.2 algebra is `u = (SymbolRange - 1) - SymbolValue`: after
init, `u == paddedBuf` (the top 15 bits of input the decoder
consumes); the §8.2.6 update `value -= cur` becomes `u_post = u_pre
- (range_pre - prev)` and the §8.2.6 renorm becomes `u_new = (u_post
<< bits) | paddedData`. The encoder accumulates `low` as the lower
edge of the valid `u` interval; bits aren't carry-propagated because
the entire `low` is kept as a `Vec<bool>` big-endian bigint until
`finish()` packs it into bytes.

11 new lib tests (1141 → 1152), all bit-exact round-trips through
the existing `SymbolDecoder`: single bool 0 / 1, mixed-pattern
bools, multi-symbol N=4 with §8.3 adaptation (encoder/decoder CDFs
remain equal across the run), heavy- and rare-side skewed CDFs,
200-symbol N=8 with §8.3 adaptation, mixed bool + multi-symbol on
one writer, empty-payload sanity, all-ones and all-zeros bool
streams.

Out of scope this arc (next): `read_literal` / `NS` /
`decode_subexp_bool` inverses for the descriptor wrappers; §5.11
`tile_group_obu()` writer skeleton on top of `SymbolWriter`; the
first per-block syntax (skip / mode_info / coefficient encode);
byte-level flushing during encode for multi-megabyte tile payloads
(the current implementation holds the entire accumulated `low` in
RAM until `finish` — fine for current test sizes).

## Status — 2026-05-28 (round 208)

Round 208 closes the **framing layer** of the encoder. The §5.3.4
`trailing_bits()` helper, the §5.3.1 OBU `obu_size` self-counts
wrapper, and the §7.5 temporal-unit aggregator now glue the r206
sequence-header writer and r207 frame-header writer into a complete,
parser-walkable IVF file.

Three layers land on top of arcs 1 + 2:

* **`BitWriter::trailing_bits` / `trailing_bits_to_alignment`** —
  §5.3.4 `trailing_bits(nbBits)` (1 `trailing_one_bit` +
  `nbBits - 1` zero pads). The alignment helper emits a full `0x80`
  byte when the writer is already byte-aligned (the common §5.3.1
  case for a body that finished on a byte boundary).
* **`encoder::obu::write_obu_with_size`** — §5.3.1 composite that
  appends the §5.3.4 trailer for the OBU types the spec wraps (every
  type except `OBU_TILE_GROUP` / `OBU_TILE_LIST` / `OBU_FRAME`),
  computes `obu_size` over the trailer-inclusive body, and writes
  `obu_header` + `leb128(obu_size)` + body in one call. Companion
  helpers: `obu_type_takes_trailing_bits(t)`,
  `ObuFrame { header, body }`,
  `write_temporal_unit(&[ObuFrame])`,
  `build_temporal_unit(seq_payload, &[ObuFrame])` (the §7.5 helper
  that auto-prefixes an `OBU_TEMPORAL_DELIMITER` + optional
  `OBU_SEQUENCE_HEADER`).
* **`encoder::temporal_unit`** — typed
  `TemporalUnitPlan { seq, emit_sequence_header, frames }` aggregator.
  `encode_temporal_unit` produces the complete byte-aligned
  bytestream for one TU; convenience wrappers
  `encode_sequence_header_obu(seq)` and
  `encode_frame_header_obu(fh, seq)` build single OBUs.

End-to-end smoke fixture (`tests/encoder_end_to_end_ivf.rs`): builds
a `SequenceHeader` + `FrameHeader` from the
`tiny-i-only-16x16-prof0` corpus fixture, encodes a single-TU IVF
file via the new aggregator + r206 `IvfWriter`, demuxes it back,
walks the OBUs via the parser-side `ObuIter`, and confirms
`parse_sequence_header` + `parse_frame_header` round-trip the
embedded descriptors. A second test covers multi-TU streams that
ship the `OBU_SEQUENCE_HEADER` only in TU0.

15 new lib tests + 2 end-to-end IVF integration tests (1126 → 1141
lib): 3 BitWriter (`trailing_bits(5)`, byte-aligned full-byte
trailer, partial-byte pad), 7 OBU framer (`write_obu_with_size` ×
SH / Frame / TileGroup / TileList / zero-body,
`write_temporal_unit`, `build_temporal_unit` with + without SH, the
§5.3.1 type-gate table), 4 typed-descriptor wrappers (SH OBU, FH
OBU, TU with SH, TU without SH), 2 integration (single-frame +
multi-frame end-to-end IVF round-trip).

Out of scope this arc (next): §5.11 `tile_group_obu()` writer
(entropy coder + coefficient encode + per-block syntax) so the
emitted file decodes to pixels; §5.9.7 `frame_size_with_refs()`
inverse for the inter frames that override the seq size; §5.9.24
`read_global_param` signed-subexp inverse for non-IDENTITY refs.

## Status — 2026-05-28 (round 207)

Round 207 lands the **`frame_header_obu()` writer** — the encoder
counterpart to `crate::frame_header::parse_frame_header`. Takes a
fully-populated `FrameHeader` + the active `SequenceHeader` and emits
the §5.9.1 `uncompressed_header()` payload that an `OBU_FRAME_HEADER`
/ `OBU_REDUNDANT_FRAME_HEADER` / `OBU_FRAME` should wrap. Round-trip
discipline: every encoder output feeds straight into the parser and
yields a structurally equal `FrameHeader`.

Sub-procedures covered (intra + show-existing-frame + reduced-still
paths, plus the inter shared tail above `disable_frame_end_update_cdf`
for headers where `frame_size_with_refs()` would have bailed):

* §5.9.2 `uncompressed_header()` — show-existing / reduced-still /
  intra (KEY / INTRA_ONLY); inter shared tail.
* §5.9.3 `allow_intrabc` gate, §5.9.5 `frame_size()` + §5.9.6
  `render_size()` + §5.9.8 `superres_params()` + §5.9.9
  `compute_image_size()`.
* §5.9.10 `read_interpolation_filter()` (inter); §5.9.11
  `loop_filter_params()` with ref/mode-delta walks; §5.9.12/13
  `quantization_params()` + `read_delta_q()`.
* §5.9.14 `segmentation_params()` (full per-segment / per-feature
  walk including `SEGMENTATION_FEATURE_SIGNED`-gated `su(1+bits)`).
* §5.9.15 `tile_info()` (uniform + non-uniform); §5.9.17
  `delta_q_params()` + §5.9.18 `delta_lf_params()`; §5.9.19
  `cdef_params()` (with the secondary `4⇒3` invert of the spec's
  `==3 ⇒ +=1`); §5.9.20 `lr_params()` (with the `lr_unit_shift` /
  `lr_unit_extra_shift` / `lr_uv_shift` bit derivations).
* §5.9.21 `read_tx_mode()`; §5.9.22/23 `skip_mode_params()` /
  `frame_reference_mode()` short-circuits; §5.9.24
  `global_motion_params()` intra short-circuit + inter
  `IDENTITY`-only emission; §5.9.30 `film_grain_params()` full path.

Three new `BitWriter` primitives land alongside: `write_uvlc`
(§4.10.3), `write_su(n)` (§4.10.6), `write_ns(n)` (§4.10.7) — the
inverses of the parser's `BitReader::{uvlc,su,ns}` descriptor readers.

18 new tests (1108 → 1126 lib): 5 primitive round-trips through the
reader and 13 frame-header round-trips covering the tiny-i-only
fixture (byte-exact at 72 bits), lossless intra (every short-circuit
path fires), QM-enabled, non-zero delta-q offsets, non-trivial CDEF
strengths (with the `4`-invert path), LR with Wiener on Y, active
ALT_Q segmentation, render-size-different, loop-filter delta updates,
show-existing-frame replay.

Out of scope this arc (next arc): §5.3.1 OBU size-field self-counts
+ §5.3.4 `trailing_bits()` wiring, §5.9.7 `frame_size_with_refs()`
inverse for inter frames with overridden size, §5.9.24
`read_global_param` signed-subexp inverse for non-IDENTITY refs,
§5.9.31 `temporal_point_info()` round-trip.

## Status — 2026-05-28 (round 206)

Round 206 opens the **encoder side** of the crate. Arc 1 lands the
bit-output plumbing only — no encode decisions yet. New
`crate::encoder` module ships four writers:

* **`encoder::bitwriter::BitWriter`** — MSB-first bit-output buffer
  (inverse of `crate::bitreader::BitReader` from §8.1). Primitives:
  `write_bit`, `write_bits(n, value)` (inverse of §4.10.2 `f(n)`),
  `write_leb128(value)` (inverse of §4.10.5 `leb128()`, with the
  same `(1 << 32) - 1` conformance cap), `byte_align`,
  `leb128_size`, `finish`.
* **`encoder::obu::{ObuHeader, ObuExtensionHeader, ObuWriter,
  write_obu}`** — §5.3 OBU framer. One-byte §5.3.2 `obu_header`
  (with `obu_forbidden_bit == 0`, 4-bit `obu_type`,
  `obu_extension_flag`, `obu_has_size_field`, reserved low bit) +
  optional one-byte §5.3.3 `obu_extension_header` (3-bit
  `temporal_id`, 2-bit `spatial_id`, 3 reserved bits) + optional
  §5.3.1 / §4.10.5 `obu_size` leb128 prefix for the §5.2
  low-overhead bytestream format. Multi-OBU temporal units are
  byte-aligned concatenations the parser's `ObuIter` walks.
* **`encoder::sequence_obu::write_sequence_header_obu(&SequenceHeader)
  -> Vec<u8>`** — `sequence_header_obu()` payload writer per §5.5.1
  (with §5.5.2 `color_config`, §5.5.3 `timing_info`, §5.5.4
  `decoder_model_info`, §5.5.5 `operating_parameters_info`).
  Reuses the parser's `crate::sequence_header::SequenceHeader` as
  the source-of-truth descriptor; every output round-trips through
  `parse_sequence_header` byte-for-byte (the decoder is the oracle).
* **`encoder::ivf::IvfWriter<W: Write>`** — IVF v0 container
  writer. Trivial public file format: 32-byte file header (`DKIF`
  magic, `version=0`, `header_length=32`, codec FOURCC,
  `u16 width / height`, `u32 fps_num / fps_den`, `u32 frame_count`,
  4 unused bytes) + per-frame 12-byte header (`u32 size + u64
  pts`). `patch_frame_count` available when the sink is `Seek`.
  File-header bytes match the `tiny-i-only-16x16-prof0/input.ivf`
  fixture exactly.

43 new tests (1065 → 1108 lib): 8 BitWriter (MSB packing, n=0
no-op, byte_align padding, leb128 single/multi-byte +
round-trip-against-parser, 64-bit width), 7 OBU framer (byte-exact
TD, round-trip through parser for every `ObuType`, multi-OBU walk,
extension-header byte layout, reserved-type pass-through, multi-byte
leb128 size), 9 IVF (file-header byte-exact vs fixture, frame-header
byte-exact, single-frame layout, multi-frame pts round-trip,
patch_frame_count, empty-payload frame, FOURCC constants,
`into_inner`), 19 sequence-OBU round-trip (tiny 16×16 profile-0,
monochrome, profile-2 4:2:2 12-bit, reduced-still-picture, timing
info with equal-picture-interval, timing+decoder-model+operating-
parameters, multi-operating-point with seq_level_idx>7 tier bit,
initial-display-delay, color description present, frame-id-numbers,
seq_force_screen_content_tools=0, explicit force_integer_mv,
order-hint disabled, uvlc round-trip across decade-scale values +
sentinel).

No `frame_header_obu` writer, no tile encode, no coefficient
encode yet — those are subsequent arcs. The crate's `encode_av1`
public entry continues to return `Error::NotImplemented`; the
encoder is wired up at the framing layer but the actual encode
decisions still need to land.

## Status — 2026-05-28 (round 205)

Round 205 closes the last documented §7.10.2 driver gap by wiring
**§7.10.2.5 temporal scan** + **§7.10.2.6 temporal sample** into
`find_mv_stack` and surfacing the §7.9 motion-field-estimation
helpers (`get_mv_projection` / `get_block_position`) the §7.10.2.5
read site relies on. The r172 `Error::TemporalMvScanUnsupported`
deferral is retired. A new public `MotionFieldMvs` aggregate
(per-ref 7-frame × `h8` × `w8` × 2 i16 grid sized for `MiRows >> 1
× MiCols >> 1` per §7.9.1) carries the §7.9.2 projection output
into the §5.11.23 / §7.10.2 cascade through `find_mv_stack` and
the §5.11.18 dispatcher (`InterFrameContext` gains a
`motion_field_mvs: &'a MotionFieldMvs` field). When the caller
opts in via `use_ref_frame_mvs == true`, the §7.10.2.5 `scan_blk`
driver walks the per-block 4×4 step grid (`stepW4 = (bw4 >= 16) ?
4 : 2`, `stepH4 = (bh4 >= 16) ? 4 : 2`, capped at `Min(b{w,h}4,
16)`) plus the §7.10.2.5 `allowExtension` corner-offset table
`{{bh4,-2}, {bh4,bw4}, {bh4-2,bw4}}` gated on `BLOCK_8X8 <= block
< BLOCK_64X64` and §7.10.2.5 `check_sb_border`'s 16×16-MI
superblock window. The §7.10.2.6 body reads
`MotionFieldMvs[RefFrame[list]][y8][x8]`, short-circuits on the
`MFMV_INVALID = -1 << 15` sentinel, runs the §7.10.2.10 lower-
precision pass, then dedupes / appends to the stack with weight 2
on a hit (with the §7.10.2.6 centre-cell `ZeroMvContext` adjust
applied per spec — single-pred + compound branches both wired).
New §3 constants `MFMV_STACK_SIZE = 3`, `MAX_OFFSET_WIDTH = 8`,
`MAX_OFFSET_HEIGHT = 0`, `MV_IN_USE_BITS = 14`, `MFMV_INVALID = -1
<< 15`, plus the `DIV_MULT[32]` quantised-inverse table from
av1-spec p.215. New `get_mv_projection(mv, num, denom)` runs the
§7.9.3 `Round2Signed(mv * num * Div_Mult[denom], 14)` clamp into
`±((1 << MV_IN_USE_BITS) - 1)`; new `get_block_position(x8, y8,
dst_sign, projMv, mi_rows, mi_cols)` returns `Option<(u32, u32)>`
matching the §7.9.4 `posValid` outcome. 9 new tests (1056 -> 1065
lib, workspace-wide 1119 -> 1128): MotionFieldMvs `new_invalid`
sentinel-fills + sizes; set / get round-trip + out-of-grid
sentinel; set_all_refs stamps every ref; get_mv_projection identity
when num==denom; get_mv_projection clamp into `MV_IN_USE_BITS`;
get_block_position valid + invalid; temporal_scan single-pred
centre-cell lands a candidate; compound branch drops on
second-ref-sentinel; +1 cascade reshape (`find_mv_stack` with
opt-in + invalid-grid asserts the §7.10.2.6 ZeroMvContext stamp).
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

## Status — 2026-05-28 (round 204)

Round 204 lands the **§9.5.3 Quantizer matrix tables** (av1-spec
p.510-553) and wires them into the §7.12.3 step-1 dequantization
loop. New `crate::qmatrix` module transcribes the normative
`Qm_Offset[ TX_SIZES_ALL ]` (p.510, 19 entries including five
share-with-parent entries: `TX_64X64` / `TX_32X64` / `TX_64X32`
share `TX_32X32`'s 32×32 region at offset 336; `TX_16X64` /
`TX_64X16` share `TX_16X32` / `TX_32X16` at offsets 1680 / 2192)
and the full `Quantizer_Matrix[ 15 ][ 2 ][ QM_TOTAL_SIZE = 3344 ]`
table (~98 KiB of `.rodata`, all values in u8 range 30..=242,
indexed by `SegQMLevel[ plane ][ segment_id ]` ∈ `0..=14` × plane
class `plane > 0 ? 1 : 0` × per-bin offset `Qm_Offset[ txSz ] + i
* tw + j`). Level 15 short-circuits at the §7.12.3 step-1b guard
and is not represented. New constants `QM_TOTAL_SIZE = 3344`,
`QM_LEVELS = 15`, `QM_PLANES = 2`; new helper `qmatrix_value(
seg_qm_level, plane, tx_size, i, j ) -> u8` with the spec's `tw
= min(32, Tx_Width[ txSz ])` / `th = min(32, Tx_Height[ txSz ])`
clamp baked in. The `cdf::dequantize_step1` QM-active arm
(previously fell through with a `debug_assert!`) now evaluates
`q2 = Round2( q * Quantizer_Matrix[ ... ], 5 ) = (q * qm + 16)
>> 5` per §7.12.3 step-1b + §3 p.13 integer Round2. Branch
guards (`using_qmatrix && PlaneTxType < IDTX && SegQMLevel <
15`) preserved verbatim. 15 new tests (1104 -> 1119): 10
qmatrix-module tests (`QM_TOTAL_SIZE` / `QM_LEVELS` / `QM_PLANES`
constants; `Qm_Offset` literal-match + unique-entry sum = 3344;
level-0 luma 4×4 first row + bottom-right; level-0 chroma 4×4
first row + plane=2 mirror; level-14 luma 4×4 first row = 31;
range invariant 30..=242 across all 100320 cells; every
TX_SIZES_ALL covered within `tw`/`th` for both plane classes;
TX_64X64 ↔ TX_32X32 share; TX_16X64 ↔ TX_16X32 and TX_64X16 ↔
TX_32X16 share; Qm_Offset advance = tw*th invariant on the 14
unique sizes) plus 4 new cdf-module integration tests for
`dequantize_step1` (QM-active level-0 luma DC with Round2 form;
chroma `plane > 0` selector across plane=1 and plane=2; IDTX
fall-through guard; AC bin uses the right matrix weight).
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

## Status — 2026-05-28 (round 203)

**Clean-room rebuild, round 24.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
core decoder modules could not be defended against the "no external
library source as reference" rule that governs every crate in this
workspace.

Bitstream parsing currently covers:

* **§5.3 / §4.10.5 — OBU bytestream walker (round 1).**
  `leb128()` (with the `(1 << 32) - 1` conformance cap and the
  8-byte length bound), `obu_header` (forbidden bit, 4-bit
  `obu_type`, extension flag, size flag), `obu_extension_header`
  (`temporal_id`, `spatial_id`, inferred to 0 when no extension —
  §6.2.3), `obu_size` payload framing, and an iterator over a
  concatenation of OBUs in the §5.2 low-overhead format.

* **§5.5 — Sequence header OBU parse (round 2).**
  `sequence_header_obu()` (§5.5.1) plus its nested syntax tables:
  `color_config()` (§5.5.2), `timing_info()` (§5.5.3),
  `decoder_model_info()` (§5.5.4), and
  `operating_parameters_info()` (§5.5.5). Returns a typed
  `SequenceHeader` (profile, still-picture / reduced-still flags,
  timing / decoder-model state, operating-point list, frame-size
  bits, frame-id presence, all `enable_*` capability bits,
  `seq_force_screen_content_tools` / `seq_force_integer_mv`,
  `order_hint_bits`, full `ColorConfig`, `film_grain_params_present`,
  and the bit-count consumed). Backed by a small internal MSB-first
  bit reader implementing §4.10.2 `f(n)` and §4.10.3 `uvlc()` per
  the §8.1 parsing process. Enforces the two §6.4.1
  bitstream-conformance gates (`seq_profile <= 2`,
  `reduced_still_picture_header == 1 ⇒ still_picture == 1`).

* **§5.9.2 — Uncompressed-header prefix parse (round 3).**
  `parse_frame_header()` consumes the leading slice of
  `uncompressed_header()` and returns a typed `FrameHeader`:
  `show_existing_frame` plus the optional `frame_to_show_map_idx`
  and `display_frame_id` for the show-existing replay path,
  `frame_type` (`KEY_FRAME` / `INTER_FRAME` / `INTRA_ONLY_FRAME` /
  `SWITCH_FRAME`), `show_frame`, `showable_frame`,
  `error_resilient_mode`, `disable_cdf_update`,
  `allow_screen_content_tools`, `force_integer_mv` (with the
  §5.9.2 `FrameIsIntra` override applied), `current_frame_id`
  (only when the sequence header opted into frame-id numbering),
  `frame_size_override_flag`, `order_hint` (width derived from
  §5.5.1's `order_hint_bits`), `primary_ref_frame` (with
  `PRIMARY_REF_NONE` for intra / error-resilient frames), and
  `refresh_frame_flags` (with the SWITCH or KEY-with-show_frame
  derivation to `allFrames = 0xff`). The reduced-still-picture
  collapse from §5.9.2 is honoured. Bit-count consumed is reported
  via `FrameHeader::bits_consumed` for the next round.
  `temporal_point_info()` (§5.9.31) call sites — gated by
  `decoder_model_info_present_flag && !equal_picture_interval` —
  are stubbed; the parser returns `Error::TemporalPointInfoUnsupported`
  if it would have to descend, but none of the 16 fixtures
  triggers it.

* **§5.9.5 / §5.9.6 / §5.9.8 / §5.9.9 — Frame-size sub-syntax
  block (round 4).** The same `parse_frame_header()` now drops
  past `refresh_frame_flags` into the four frame-size sub-syntaxes
  and returns a typed `FrameSize`: `frame_width` (post-superres),
  `frame_height`, `render_width`, `render_height`, `superres_denom`
  (in `9..=16` when `use_superres == 1`, otherwise `SUPERRES_NUM
  = 8`), `upscaled_width` (pre-superres), `mi_cols`, `mi_rows`
  (the `MI_SIZE = 4` block grid via `2 * ((dim + 7) >> 3)`), and
  the three sub-syntax-input fields (`use_superres`, `coded_denom`,
  `render_and_frame_size_different`). For super-resolved frames the
  rounded-half-up downscale `FrameWidth = (UpscaledWidth *
  SUPERRES_NUM + SuperresDenom / 2) / SuperresDenom` is applied
  literally per §5.9.8. `FrameHeader::frame_size` is
  `Some(FrameSize)` for every intra (`KEY_FRAME` /
  `INTRA_ONLY_FRAME`) frame and `None` for show-existing-frame
  replays and inter frames (the §5.9.7 `frame_size_with_refs()`
  `found_ref == 1` branch needs ref-frame state — `RefUpscaledWidth[]`
  / `RefFrameHeight[]` / `RefRenderWidth[]` / `RefRenderHeight[]` —
  not yet tracked across calls).

* **§5.9.10 / §5.9.11 / §5.9.12 / §5.9.13 — Uncompressed-header
  tail sub-syntaxes (round 5).** Three standalone parser entry
  points landed in a new `uncompressed_header_tail` module:
  * `parse_interpolation_filter` (§5.9.10) — reads
    `is_filter_switchable` + optional `f(2)` `interpolation_filter`,
    returning a typed `InterpolationFilter` enum (`Eighttap` /
    `EighttapSmooth` / `EighttapSharp` / `Bilinear` / `Switchable`)
    per §6.8.9.
  * `parse_loop_filter_params` (§5.9.11) — honours the
    `CodedLossless || allow_intrabc` short-circuit (no bits read,
    `loop_filter_ref_deltas` reset to the spec's literal defaults
    `INTRA = 1`, `LAST/LAST2/LAST3/BWDREF = 0`, `GOLDEN/ALTREF2/
    ALTREF = -1`), then for the full path reads the four
    `loop_filter_level[]` slots (with the `NumPlanes > 1 &&
    (level[0] || level[1])` gate on the chroma pair), the `f(3)`
    `loop_filter_sharpness`, and the `loop_filter_delta_enabled /
    delta_update / update_ref_delta[i] / update_mode_delta[i]`
    per-slot update walk over `TOTAL_REFS_PER_FRAME = 8`
    ref-deltas + 2 mode-deltas with `su(7)` signed offsets.
  * `parse_quantization_params` (§5.9.12 + §5.9.13) — reads
    `base_q_idx` (`f(8)`), the four `delta_q_*` per-plane offsets
    via `read_delta_q()` (each a `delta_coded` gate followed by a
    `su(7)` signed offset), the `diff_uv_delta` /
    `separate_uv_delta_q` chroma-coupling logic that mirrors V to
    U when `diff_uv_delta == 0`, and the `using_qmatrix` / `qm_y` /
    `qm_u` / `qm_v` quantizer-matrix selection.
  These three calls are **not** wired into the streaming
  `parse_frame_header` walk yet — the intervening §5.9.2 syntax
  (`allow_intrabc`, `disable_frame_end_update_cdf`, `tile_info()`,
  `segmentation_params()`, `delta_q_params()`, `delta_lf_params()`)
  sits between round 4's stop point and these calls. The next
  round can stitch them in as the intervening syntaxes land.

  New types: `InterpolationFilter`, `LoopFilterParams`,
  `QuantizationParams`. New constant: `TOTAL_REFS_PER_FRAME = 8`.
  New bitreader primitive: `BitReader::su(n)` (§4.10.6). 21 new
  unit tests cover all three sub-syntaxes (switchable +
  non-switchable interpolation, short-circuit + full-path
  loop_filter with mono/3-plane gating + delta update walk,
  mono/3-plane quantization with and without `separate_uv_delta_q`
  and with/without `using_qmatrix`) and `su(7)` boundary values.

* **§5.9.3 `allow_intrabc` + §5.9.15 `tile_info()` wired into the
  streaming parser (round 6).** For intra frames whose
  `allow_screen_content_tools && UpscaledWidth == FrameWidth`
  conjunction holds, the `parse_frame_header` walk now consumes
  the §5.9.3 `f(1)` `allow_intrabc` slot; otherwise the §5.9.2
  `allow_intrabc = 0` initialiser stands. The
  `disable_frame_end_update_cdf` bit (gated by
  `!reduced_still_picture_header && !disable_cdf_update`) is
  consumed next, then `tile_info()` per §5.9.15 walks the
  per-frame tile layout: the uniform-spacing path uses
  `increment_tile_cols_log2` / `increment_tile_rows_log2` loops
  capped at `tile_log2(1, min(sbCols, MAX_TILE_COLS))` /
  `tile_log2(1, min(sbRows, MAX_TILE_ROWS))`; the non-uniform
  path uses `ns(maxWidth)` / `ns(maxHeight)` for
  `width_in_sbs_minus_1` / `height_in_sbs_minus_1` via the new
  `BitReader::ns(n)` §4.10.7 primitive. `context_update_tile_id`
  (`f(TileColsLog2 + TileRowsLog2)`) +
  `tile_size_bytes_minus_1` (`f(2)`) are read when at least one
  of the log2 counts is non-zero. New type `TileInfo` exposes
  `uniform_tile_spacing_flag`, `tile_cols`, `tile_rows`,
  `tile_cols_log2`, `tile_rows_log2`, `context_update_tile_id`,
  `tile_size_bytes`, `mi_col_starts` (`MiColStarts[0..=TileCols]`),
  and `mi_row_starts`. New public §3 constants:
  `MAX_TILE_WIDTH = 4096`, `MAX_TILE_AREA = 4096 * 2304`,
  `MAX_TILE_ROWS = MAX_TILE_COLS = 64`. New fields on
  `FrameHeader`: `allow_intrabc`, `disable_frame_end_update_cdf`,
  `tile_info: Option<TileInfo>` (the latter `None` for inter
  frames + show-existing-frame replays). 11 new tests (7 for
  `tile_info` + 3 for `ns(n)` + 2 for the streaming-parser
  integration); the 16-fixture frame-header integration test
  gains 4 new asserted trace columns (`allow_intrabc`,
  `tile_cols`, `tile_rows`, `context_update_tile_id`) plus a
  §6.8.14 `MAX_TILE_COLS` / `MAX_TILE_ROWS` conformance guard.
  The `tile-cols-2-rows-1` fixture exercises a real 2-tile
  layout (`TileColsLog2 = 1`).

* **§5.9.12 `quantization_params()` + §5.9.14
  `segmentation_params()` wired into the streaming parser (round
  7).** After `tile_info()` the parser now consumes the
  quantization_params block (already implemented standalone in
  round 5) — `base_q_idx` (`f(8)`), the per-plane `delta_q_*`
  offsets via `read_delta_q()` (`delta_coded` `f(1)` gate + `su(7)`
  signed offset), the chroma `diff_uv_delta` / V-mirrors-U logic,
  and the `using_qmatrix` / `qm_y` / `qm_u` / `qm_v` quantizer-
  matrix block — and surfaces a typed `QuantizationParams` on
  `FrameHeader::quantization_params`. Then the new
  segmentation_params routine reads `segmentation_enabled` and,
  when enabled, either reads the three update flags or uses the
  §5.9.14 `primary_ref_frame == PRIMARY_REF_NONE` collapse
  (`update_map=1` / `temporal_update=0` / `update_data=1`, no
  bitstream reads). For `update_data=1` the inner loop walks all
  8 segments × 8 features, reading `feature_enabled` and (when
  active) `su(1+bits)` or `f(bits)` per the `Segmentation_Feature_Bits`
  / `Segmentation_Feature_Signed` / `Segmentation_Feature_Max`
  Table 5.9.14 tables. The §5.9.14 trailing `SegIdPreSkip` /
  `LastActiveSegId` derivations are surfaced. New type
  `SegmentationParams` exposing `enabled`, `update_map`,
  `temporal_update`, `update_data`,
  `segment_feature_active: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS]`,
  `segment_feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS]`,
  `seg_id_pre_skip`, `last_active_seg_id`. New public §3 constants:
  `MAX_SEGMENTS = 8`, `SEG_LVL_MAX = 8`, `SEG_LVL_ALT_Q = 0`,
  `SEG_LVL_ALT_LF_Y_V = 1`, `SEG_LVL_ALT_LF_Y_H = 2`,
  `SEG_LVL_ALT_LF_U = 3`, `SEG_LVL_ALT_LF_V = 4`,
  `SEG_LVL_REF_FRAME = 5`, `SEG_LVL_SKIP = 6`,
  `SEG_LVL_GLOBALMV = 7`, `MAX_LOOP_FILTER = 63`. New public Table
  5.9.14 tables: `SEGMENTATION_FEATURE_BITS`,
  `SEGMENTATION_FEATURE_SIGNED`, `SEGMENTATION_FEATURE_MAX`. New
  standalone parser entry point `parse_segmentation_params`. 10
  new tests (9 standalone — disabled / PRIMARY_REF_NONE collapse /
  three-bit update walk / `update_map=0` skips temporal /
  signed `SEG_LVL_ALT_Q` value `-50` / clipped at the `-255` floor
  / unsigned `SEG_LVL_REF_FRAME=6` sets `SegIdPreSkip=1` /
  zero-width `SEG_LVL_SKIP` sets `LastActiveSegId=3` /
  unexpected-end — plus 1 streaming-parser synthetic with
  `SEG_LVL_ALT_Q` active value `-123`). The 16-fixture
  frame-header integration test gains two new asserted trace
  columns (`base_q_idx`, `seg_enabled`) plus a `SegIdPreSkip = 0`
  / `LastActiveSegId = 0` invariant guard.

* **§5.9.17 `delta_q_params()` + §5.9.18 `delta_lf_params()` wired
  into the streaming parser (round 8).** After `segmentation_params()`
  the parser now consumes `delta_q_params()` per §5.9.17 — the
  `delta_q_present` `f(1)` slot is read only when `base_q_idx > 0`
  (otherwise the §5.9.17 `delta_q_present = 0` initialiser stands and
  no bit is consumed), and `delta_q_res` (`f(2)`) follows only when
  `delta_q_present == 1` — then `delta_lf_params()` per §5.9.18: the
  whole block is gated on `delta_q_present`, the `delta_lf_present`
  `f(1)` slot is suppressed when `allow_intrabc == 1`, and
  `delta_lf_res` (`f(2)`) + `delta_lf_multi` (`f(1)`) follow only when
  `delta_lf_present == 1`. New types `DeltaQParams`
  (`delta_q_present`, `delta_q_res`) and `DeltaLfParams`
  (`delta_lf_present`, `delta_lf_res`, `delta_lf_multi`) surface on
  `FrameHeader::delta_q_params` / `FrameHeader::delta_lf_params`
  (`Some` for intra frames, `None` for inter / show-existing-frame
  paths). New standalone parser entry points `parse_delta_q_params` /
  `parse_delta_lf_params`. 9 new unit tests (3 for `delta_q_params` —
  `base_q_idx == 0` no-read / `delta_q_present == 0` 1-bit /
  `delta_q_present == 1` reads `delta_q_res` — plus an
  unexpected-end; 5 for `delta_lf_params` — gated-off when
  `delta_q_present == 0` / `delta_lf_present == 0` 1-bit / full path
  reading `delta_lf_res` + `delta_lf_multi` / suppressed by
  `allow_intrabc` / unexpected-end). The 16-fixture frame-header
  integration test gains two new asserted trace columns
  (`delta_q_present`, `delta_lf_present`) plus `delta_q_res = 0` /
  `delta_lf_res = 0` / `delta_lf_multi = false` invariant guards. The
  `lossless-i-only` fixture (`base_q_idx = 0`) exercises the §5.9.17
  no-read branch; every other fixture reads exactly one
  `delta_q_present` bit (all 0).

* **§5.9.11 `loop_filter_params()` wired into the streaming parser
  (round 9).** After `delta_lf_params()` the parser now derives
  `CodedLossless` per the §5.9.2 lines that scan `LosslessArray[]` over
  the eight per-segment qindexes — `get_qindex(1, segmentId)` (the §8.7
  ignore-delta branch) returns `base_q_idx`, or
  `Clip3(0, 255, base_q_idx + FeatureData[segmentId][SEG_LVL_ALT_Q])`
  when `seg_feature_active_idx(segmentId, SEG_LVL_ALT_Q)` is set; a
  segment is lossless when its qindex is 0 and all five §5.9.12
  `DeltaQ?*` offsets are 0 — and then consumes `loop_filter_params()`
  per §5.9.11. The `CodedLossless || allow_intrabc` short-circuit
  consumes no bits and resets `loop_filter_ref_deltas` to the spec
  defaults; the full path reads the four `loop_filter_level[]` slots
  (chroma `[2]`/`[3]` gated on `NumPlanes > 1 && (level[0] ||
  level[1])`), the `f(3)` `loop_filter_sharpness`, and the
  `loop_filter_delta_enabled` / `delta_update` per-slot update walk. The
  routine itself landed standalone in round 5
  (`parse_loop_filter_params`); this round adds the streaming wire-in
  plus the new `compute_coded_lossless` derivation. New field
  `FrameHeader::loop_filter_params: Option<LoopFilterParams>` (`Some`
  for intra frames, `None` for inter / show-existing replays). 6 new
  unit tests (5 for `compute_coded_lossless` covering the lossless /
  non-lossless / non-zero-delta / `SEG_LVL_ALT_Q`-clamp / seg-disabled
  branches, 1 streaming full-path test with non-zero loop-filter
  levels); the `parses_tiny_key_frame_prefix` bit-count rises 31 → 48.
  The 16-fixture frame-header integration test gains five new asserted
  trace columns (`lf_y`, `lf_uv0`, `lf_uv1`, `lf_sharp`,
  `lf_delta_enabled`) mapped to `loop_filter_level[0, 2, 3]` /
  `loop_filter_sharpness` / `loop_filter_delta_enabled` per §6.8.10. The
  `lossless-i-only` fixture (`base_q_idx = 0`, `lf_delta_enabled = 0`)
  exercises the §5.9.11 short-circuit and validates the `CodedLossless`
  derivation; the other 15 take the full path (several with non-zero
  chroma levels, e.g. `film-grain-on` `lf_y=4 / lf_uv0=14 / lf_uv1=11`).

* **§5.9.19 `cdef_params()` wired into the streaming parser (round
  10).** After `loop_filter_params()` the parser now consumes
  `cdef_params()` per §5.9.19. The `CodedLossless || allow_intrabc ||
  !enable_cdef` short-circuit consumes no bits and leaves `cdef_bits =
  0`, `CdefDamping = 3`, and all four strength arrays zeroed; the full
  path reads `cdef_damping_minus_3` (`f(2)`, `CdefDamping =
  cdef_damping_minus_3 + 3`), `cdef_bits` (`f(2)`), then for each of the
  `1 << cdef_bits` entries the `cdef_y_pri_strength[i]` (`f(4)`) /
  `cdef_y_sec_strength[i]` (`f(2)`) pair plus, when `NumPlanes > 1`, the
  `cdef_uv_pri_strength[i]` (`f(4)`) / `cdef_uv_sec_strength[i]` (`f(2)`)
  pair. The §5.9.19 secondary `== 3 ⇒ += 1` adjustment is applied
  literally (a raw `3` is stored as `4`). New type `CdefParams`
  (`cdef_damping`, `cdef_bits`, the four `cdef_*_strength: [u8; 8]`
  arrays, `short_circuited`). New public §3-derived constant
  `CDEF_MAX_STRENGTHS = 8` (the loop bound `1 << cdef_bits` with
  `cdef_bits` an `f(2)` value is at most 8). New standalone parser entry
  point `parse_cdef_params`. New field `FrameHeader::cdef_params:
  Option<CdefParams>` (`Some` for intra frames, `None` for inter /
  show-existing replays). 8 new unit tests (short-circuit on each of the
  three gate conditions, full-path single-entry 3-plane,
  secondary-`3⇒4` for both Y/UV, monochrome chroma-skip, 8-entry
  `cdef_bits=3` loop bound, unexpected-end); the
  `parses_tiny_key_frame_prefix` bit-count rises 48 → 64. The 16-fixture
  frame-header integration test gains six new asserted trace columns
  (`cdef_damping`, `cdef_bits`, `cdef_y_pri_strength[0]`,
  `cdef_uv_pri_strength[0]`, `cdef_y_sec_strength[0]`,
  `cdef_uv_sec_strength[0]`) sourced from each fixture's `CDEF idx=0`
  trace line, plus a short-circuit invariant. The `CDEF` trace lines were
  empirically confirmed to log the raw pre-adjustment secondary strength
  (a value of `3` appears, which the parser stores as `4`); the test maps
  the raw expectation through the adjustment. `lossless-i-only`
  (CodedLossless) and `screen-content-tools` (`enable_cdef=0`) exercise
  the short-circuit; the other 14 take the full path (e.g.
  `super-resolution` `damping=5 / cdef_y_pri=2 / cdef_uv_pri=14`).

* **§5.9.20 `lr_params()` wired into the streaming parser (round
  11).** After `cdef_params()` the parser now consumes `lr_params()`
  per §5.9.20. `AllLossless = CodedLossless && (FrameWidth ==
  UpscaledWidth)` is derived inline (super-resolution downscaling
  keeps AllLossless 0 even when CodedLossless is 1). The `AllLossless
  || allow_intrabc || !enable_restoration` short-circuit consumes no
  bits and leaves every plane `RESTORE_NONE` with `UsesLr = 0` and
  zero `LoopRestorationSize[]`. The full path reads one `lr_type`
  (`f(2)`) per plane, mapping each through `Remap_Lr_Type[4] = {
  RESTORE_NONE, RESTORE_SWITCHABLE, RESTORE_WIENER, RESTORE_SGRPROJ }`;
  when any plane uses LR, the parser reads `lr_unit_shift` (`f(1)`,
  post-incremented for 128×128 superblocks, otherwise extended by
  `lr_unit_extra_shift` `f(1)` when the first bit is set) and — when
  `subsampling_x && subsampling_y && usesChromaLr` — `lr_uv_shift`
  (`f(1)`). The three `LoopRestorationSize[]` entries derive from
  `RESTORATION_TILESIZE_MAX = 256` via `>> (2 - lr_unit_shift)` for
  luma and `>> lr_uv_shift` for chroma. New types `LrParams`
  (`frame_restoration_type[3]`, `uses_lr`, `uses_chroma_lr`,
  `lr_unit_shift`, `lr_uv_shift`, `loop_restoration_size[3]`,
  `short_circuited`) and `FrameRestorationType` (4-variant enum with
  §6.10.15 symbol discriminants 0..=3, plus `remap(lr_type)`). New
  constant `RESTORATION_TILESIZE_MAX = 256`. New standalone parser
  entry point `parse_lr_params`. New field
  `FrameHeader::lr_params: Option<LrParams>`. 19 new unit tests
  (short-circuit on all three gates, the `Remap_Lr_Type` table, UsesLr=0
  no-shift path, non-128 SB with lr_unit_shift each of {0, 1, 2}, 128 SB
  post-increment with each of {1, 2}, 4:2:0 chroma uv-shift {0, 1},
  subsampling-gated uv-shift skip on 4:4:4 and 4:2:2, monochrome
  one-type-only, distinct types per plane, and two unexpected-end
  cases). The `parses_tiny_key_frame_prefix` bit-count rises 64 → 70.
  The 16-fixture frame-header integration test gains five new asserted
  trace columns (`y_type`, `u_type`, `v_type`, `unit_shift`,
  `uv_shift` from each fixture's `LOOP_RESTORATION idx=0` trace line)
  plus a `UsesLr` cross-check, short-circuit invariant (only
  `lossless-i-only` AllLossless short-circuits), and a
  `LoopRestorationSize[0]` derivation cross-check. Empirically
  confirmed: the trace logger writes the **raw bitstream `lr_type`**
  (not the post-`Remap_Lr_Type` symbol) — the four
  `bits_consumed`-traceable fixtures (`i-only-64x64-prof0`,
  `i-frame-then-p-64x64`, `super-resolution`, `superblocks-128`) all
  decode bit-exactly only when the test routes the trace value through
  `Remap_Lr_Type` before comparing. `super-resolution` exercises
  three-plane Wiener LR with unit_shift=2; `superblocks-128`
  exercises two-plane SgrProj LR with unit_shift=2 from a 128×128
  superblock; `i-only-64x64-prof0` and `i-frame-then-p-64x64` each
  exercise V-plane-only Wiener LR (`usesChromaLr=1` and the 4:2:0
  uv-shift read=0); `lossless-i-only` exercises the AllLossless
  short-circuit (the only fixture that does); the other 10 walk the
  full LR path with all three planes RESTORE_NONE / `UsesLr = 0`.

* **§5.9.21 `read_tx_mode()` wired into the streaming parser (round
  12).** After `lr_params()` the parser now consumes `read_tx_mode()`
  per §5.9.21. When `CodedLossless == 1` the first branch fires:
  no bits are read and `TxMode = ONLY_4X4`. Otherwise a single
  `tx_mode_select` (`f(1)`) bit selects `TX_MODE_SELECT` (`1`) or
  `TX_MODE_LARGEST` (`0`). New type `TxMode` (3-variant enum with
  §6.8.21 symbol discriminants `Only4x4 = 0`, `TxModeLargest = 1`,
  `TxModeSelect = 2`). New constant `TX_MODES = 3`. New standalone
  parser entry point `parse_tx_mode`. New field
  `FrameHeader::tx_mode: Option<TxMode>` (`Some` for intra frames,
  `None` for inter / show-existing replays). 6 new unit tests (the
  §6.8.21 symbol values + count, the CodedLossless no-bits path twice,
  `tx_mode_select` set/clear, unexpected-end); the
  `parses_tiny_key_frame_prefix` bit-count rises 70 → 71. The 16-fixture
  frame-header integration test gains one new asserted trace column
  (`tx_mode` from each fixture's `FRAME_HEADER` trace line) plus a
  `ONLY_4X4 ⇒ CodedLossless` invariant. The corpus exercises all three
  values: `lossless-i-only` is the only `tx_mode = 0` (ONLY_4X4,
  CodedLossless, no-bits path); `tiny-i-only-16x16-prof0`,
  `monochrome-grey-only`, `profile-1-yuv444-8bit`, and
  `profile-2-yuv422-12bit` are `tx_mode = 1` (TX_MODE_LARGEST,
  `tx_mode_select = 0`); the other 11 are `tx_mode = 2`
  (TX_MODE_SELECT, `tx_mode_select = 1`).

* **The §5.9.2 uncompressed-header tail wired into the streaming parser
  (round 13).** After `read_tx_mode()` the intra path now walks to the
  end of `uncompressed_header()`: `frame_reference_mode()` (§5.9.23),
  `skip_mode_params()` (§5.9.22), the `allow_warped_motion` slot,
  `reduced_tx_set` (`f(1)`), `global_motion_params()` (§5.9.24), and
  `film_grain_params()` (§5.9.30). For an intra frame all but
  `reduced_tx_set` (one bit) and `film_grain_params()` collapse without
  reading bits. New types `WarpModelType`, `GlobalMotionParams`
  (`gm_type[8]` / `gm_params[8][6]`, identity short-circuit), and
  `FilmGrainParams` (full §5.9.30 field set); new `FrameHeader` fields
  `reference_select` / `skip_mode_present` / `allow_warped_motion` /
  `reduced_tx_set` / `global_motion_params` / `film_grain_params`. The
  full §5.9.24/§5.9.25 inter global-motion syntax — `read_global_param`
  plus the §5.9.26–§5.9.29 `decode_signed_subexp_with_ref` /
  `decode_subexp` / `inverse_recenter` sub-exponential decoders — is
  implemented and exposed as standalone `parse_global_motion_params`;
  `film_grain_params()` is exposed as `parse_film_grain_params`. 14 new
  unit tests; the integration test embeds the `film-grain-on` fixture's
  full 718-byte FRAME OBU payload to validate the `apply_grain = 1` FGS
  block (14 Y points, 8 Cb + 9 Cr points, `ar_coeff_lag = 2`,
  `seed = 45231`) byte-exact against the fixture trace.

* **The inter-frame `uncompressed_header()` path (round 14).** An
  `INTER_FRAME` / `SWITCH_FRAME` header now parses end-to-end. The §5.9.2
  `else` branch reads `frame_refs_short_signaling`, the explicit
  `ref_frame_idx[]` (or computes them via §7.8 `set_frame_refs()`), the
  §5.9.7 `frame_size_with_refs()` / `frame_size()` + `render_size()` size
  selection, `allow_high_precision_mv`, §5.9.10
  `read_interpolation_filter()`, `is_motion_mode_switchable`,
  `use_ref_frame_mvs`, then the shared tile / quant / segment / delta /
  loop-filter / CDEF / LR / tx-mode tail, the inter
  `frame_reference_mode()` (`reference_select`), §5.9.22
  `skip_mode_params()`, `allow_warped_motion`, `reduced_tx_set`, inter
  `global_motion_params()`, and `film_grain_params()`. New §7.8
  `set_frame_refs()`, §5.9.3 `get_relative_dist()`, §5.9.7
  `frame_size_with_refs()`, and §5.9.22 `skip_mode_params()` are backed by
  a new public `RefInfo` cross-frame reference state (`RefValid[]` /
  `RefOrderHint[]` / `RefFrameId[]` + per-slot `RefUpscaledWidth[]` /
  `RefFrameHeight[]` / `RefRenderWidth[]` / `RefRenderHeight[]`). New
  public API: `parse_frame_header_with_refs()`, `RefInfo`,
  `InterFrameRefs` (surfaced on `FrameHeader::inter_refs`). Verified
  byte-exact against the `i-frame-then-p-64x64` `idx=1` `FRAME_HEADER` +
  `REF_MAP` trace lines (134 uncompressed-header bits; `ref_frame_idx =
  [0; 7]`, `base_q_idx = 120`, `tx_mode = 1`, `reference_select = 0`,
  `allow_warped_motion = 1`). 5 new tests (a byte-exact inter-header
  fixture test, a `RefInfo` default contract test, and three
  `set_frame_refs()` / `get_relative_dist()` unit tests).

* **§8.2 — the symbol (arithmetic / msac) decoder (round 15).** A new
  standalone `SymbolDecoder` implements the AV1 entropy engine that
  every tile-content read will sit on: §8.2.2 `init_symbol(sz)`, §8.2.6
  `read_symbol(cdf)` (the CDF-adaptive multisymbol search with
  `EC_PROB_SHIFT`/`EC_MIN_PROB`, the `prev - cur` range update, and the
  seven-step renormalisation drawing new bits — or §8.2.2 padding zeros
  once `SymbolMaxBits` is exhausted), the §8.3 adaptive-rate CDF update
  (`rate = 3 + (cdf[N]>15) + (cdf[N]>31) + Min(FloorLog2(N), 2)` plus the
  count-to-32 counter), §8.2.3 `read_bool()`, §8.2.5 `read_literal(n)`
  (`L(n)`), `NS(n)` (§4.10.10), the arithmetic-coded
  `decode_subexp_bool(numSyms, k)` (§5.9.28 bool variant), and §8.2.4
  `exit_symbol()` (trailing-bit accounting + byte-alignment advance,
  rejecting the `SymbolMaxBits < -14` conformance violation). The
  decoder shares the existing MSB-first `BitReader` (§8.1 `f(n)`) so its
  position indicator advances the same `get_position()` the OBU walk
  uses. Default CDF tables and the §8.3.2 CDF-selection process are out
  of scope — they land with the tile-content decode that consumes them.
  13 byte-exact unit tests (hand-traced single decodes + term-by-term
  §8.3 update checks + padding-zero / underflow edge cases).

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — intra-frame
  mode / partition subset (round 16).** A new `cdf` module transcribes
  the §9.4 `Default_Intra_Frame_Y_Mode_Cdf` (5×5×14), the five
  `Default_Partition_W{8,16,32,64,128}_Cdf` tables, `Default_Skip_Cdf`,
  and `Default_Segment_Id_Cdf` verbatim from the spec — every row is
  length `N + 1` with `row[N-1] == 1 << 15` and `row[N]` the §8.3
  adaptation counter, exactly as `SymbolDecoder::read_symbol` expects.
  `TileCdfContext::new_from_defaults` performs the §8.3.1 init step
  ("each `Tile*Cdf` array is set equal to a copy of `Default_*_Cdf`"),
  producing a mutable per-tile working set. The §8.3.2 selection
  surfaces a `&mut [u16]` row for each element: `intra_frame_y_mode`
  (`[abovemode][leftmode]`), `partition` (array-by-`bsl`, row-by-`ctx`),
  `skip` (`[ctx]`), `segment_id` (`[ctx]`) — passed straight to
  `read_symbol`. Scalar context helpers `intra_mode_ctx` /
  `partition_ctx` / `skip_ctx` / `segment_id_ctx` compute the index from
  the neighbour inputs the (future) tile walk supplies. The remaining
  ~100 §9.4 tables, the `init_coeff_cdfs` coefficient set, and the
  other §8.3.2 selections (`split_or_horz` / `split_or_vert` /
  `tx_depth` / `txfm_split` / the motion-vector + uv-mode groups) are a
  clear followup. 9 new unit tests: §8.3.1 byte-exact copy + the
  `cdf[N-1] == 32768` / `cdf[N] == 0` invariant on every transcribed
  row; working-copy independence from the immutable defaults;
  `Intra_Mode_Context[]` term-by-term; `partition_ctx` / `skip_ctx` /
  `segment_id_ctx` formulae across all branches; `partition_cdf`
  selected by `bsl` returning the right row lengths and the
  default-row contents; and two end-to-end decodes driving the real
  `SymbolDecoder` through a default-CDF row (a `skip` decode that
  exercises the §8.3 update path, and a `partition` multisymbol decode
  with the update disabled).

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — motion-vector
  component subset (round 17).** Extends `cdf` with the nine
  `Default_Mv_*_Cdf` tables transcribed verbatim from §9.4
  (`Default_Mv_Joint_Cdf`, `Default_Mv_Sign_Cdf`, `Default_Mv_Class_Cdf`,
  `Default_Mv_Class0_Bit_Cdf`, `Default_Mv_Class0_Fr_Cdf`,
  `Default_Mv_Class0_Hp_Cdf`, `Default_Mv_Bit_Cdf`, `Default_Mv_Fr_Cdf`,
  `Default_Mv_Hp_Cdf` — the `216*128` / `136*128` / … fixed-point
  notation expanded). `TileCdfContext::new_from_defaults` performs the
  §8.3.1 init step ("`Mv*Cdf[ i ]` is set equal to a copy of
  `Default_Mv_*_Cdf` for `i = 0..MV_CONTEXTS-1`"), broadcasting the
  per-`comp` flat defaults to both `comp = 0..1` slots. The §8.3.2
  selection surfaces nine new `&mut [u16]` accessors —
  `mv_joint_cdf(MvCtx)`, `mv_sign_cdf(MvCtx, comp)`,
  `mv_class_cdf(MvCtx, comp)`, `mv_class0_bit_cdf(MvCtx, comp)`,
  `mv_class0_fr_cdf(MvCtx, comp, mv_class0_bit)`,
  `mv_class0_hp_cdf(MvCtx, comp)`, `mv_bit_cdf(MvCtx, comp, i)`,
  `mv_fr_cdf(MvCtx, comp)`, `mv_hp_cdf(MvCtx, comp)` — yielding the
  row `SymbolDecoder::read_symbol` consumes. The §5.11.31
  `MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0` derivation is exposed
  as the `mv_ctx` helper. 7 new unit tests: every §9.4 transcribed
  value asserted byte-exact (including the expanded `*128` fixed-point);
  §8.3.1 init copies the default into every `MV_CONTEXTS × MV_COMPS`
  slot with the `cdf[N - 1] == 32768` / `cdf[N] == 0` invariant on
  every row; the §5.11.31 `mv_ctx` derivation matches the spec; §8.3.2
  selectors return the right default row for every
  `(MvCtx, comp, *)` indexing variant; working-copy independence —
  adapting `mv_joint` / `mv_sign` / `mv_class0_fr` / `mv_bit` does not
  mutate the §9.4 source; and two end-to-end decodes driving the real
  `SymbolDecoder` through a default CDF (a 4-value `mv_joint` decode
  that exercises the §8.3 update path and a binary `mv_bit` decode
  with `disable_cdf_update == true`).

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — inter-mode /
  reference-frame subset (round 18).** Extends `cdf` with the 13
  `Default_*_Cdf` tables that drive every inter-block mode and reference
  syntax: `Default_New_Mv_Cdf`, `Default_Zero_Mv_Cdf`,
  `Default_Ref_Mv_Cdf`, `Default_Drl_Mode_Cdf`, `Default_Is_Inter_Cdf`,
  `Default_Comp_Mode_Cdf`, `Default_Skip_Mode_Cdf`,
  `Default_Comp_Ref_Cdf`, `Default_Comp_Bwd_Ref_Cdf`,
  `Default_Single_Ref_Cdf`, `Default_Compound_Mode_Cdf`,
  `Default_Comp_Ref_Type_Cdf`, `Default_Uni_Comp_Ref_Cdf` —
  all transcribed verbatim from §9.4 with the §3 constants
  `NEW_MV_CONTEXTS = 6`, `ZERO_MV_CONTEXTS = 2`, `REF_MV_CONTEXTS = 6`,
  `DRL_MODE_CONTEXTS = 3`, `IS_INTER_CONTEXTS = 4`,
  `COMP_INTER_CONTEXTS = 5`, `SKIP_MODE_CONTEXTS = 3`, `REF_CONTEXTS = 3`,
  `FWD_REFS = 4`, `BWD_REFS = 3`, `SINGLE_REFS = 7`,
  `UNIDIR_COMP_REFS = 4`, `COMP_REF_TYPE_CONTEXTS = 5`,
  `COMPOUND_MODES = 8`, `COMPOUND_MODE_CONTEXTS = 8`, `COMP_NEWMV_CTXS = 5`
  re-exposed at the crate root. `TileCdfContext::new_from_defaults`
  performs the §8.3.1 init step for every new array. The §8.3.2
  selection surfaces 13 `&mut [u16]` accessors —
  `new_mv_cdf` / `zero_mv_cdf` / `ref_mv_cdf` / `drl_mode_cdf` /
  `is_inter_cdf` / `comp_mode_cdf` / `skip_mode_cdf` / `comp_ref_cdf` /
  `comp_bwd_ref_cdf` / `single_ref_cdf` / `compound_mode_cdf` /
  `comp_ref_type_cdf` / `uni_comp_ref_cdf` — feeding straight into
  `SymbolDecoder::read_symbol`. Scalar §8.3.2 context helpers
  `is_inter_ctx(above_intra, left_intra)` (`AvailU && AvailL` /
  `AvailU XOR AvailL` / neither-avail branches per the spec ladder),
  `skip_mode_ctx(above, left)` (sum of neighbour `SkipModes[]`),
  `ref_count_ctx(c0, c1)` (the `<` / `==` / `>` three-branch ladder
  shared by every `single_ref_p*` / `comp_ref` / `comp_bwdref` /
  `uni_comp_ref_p*` paragraph), and `compound_mode_ctx(ref_mv_ctx,
  new_mv_ctx)` (the `Compound_Mode_Ctx_Map[ RefMvContext >> 1 ][
  Min(NewMvContext, COMP_NEWMV_CTXS - 1) ]` lookup, with the
  `COMPOUND_MODE_CTX_MAP` table itself surfaced as a public constant)
  compute each `ctx` from the neighbour-summary inputs the (future)
  tile walk supplies. 10 new unit tests (182 in src/, up from 172):
  table-dimension audit verifying every new `Default_*_Cdf` shape
  matches the spec literal (with the §8.2.6 `cdf[N - 1] == 32768` /
  `cdf[N] == 0` invariant enforced on every row); hand-picked
  byte-exact spot-checks across all 13 tables (every literal that
  appears at a row boundary read back unchanged); §8.3.1 init copies
  every default into the corresponding `Tile*Cdf` slot; §8.3.2
  selectors return the right default row at every hand-picked
  `(frame_type, ctx)` tuple — both extremes of every `ctx` index for
  all 13 syntax elements; working-copy independence — adapting
  `new_mv` / `comp_ref` / `compound_mode` does not mutate the §9.4
  source; §8.3.2 `is_inter_ctx` branch coverage (all 9 above/left
  combinations); `skip_mode_ctx` (the 4 neighbour-flag pairs);
  `ref_count_ctx` (the 3 ordering branches); `compound_mode_ctx` (one
  spot-check from each of the 3 `COMPOUND_MODE_CTX_MAP` rows plus the
  `Min(.., COMP_NEWMV_CTXS - 1)` clamp + the `RefMvContext >> 1`
  saturation); and an end-to-end §8.2 `SymbolDecoder` decode driving
  the `compound_mode` (8-value) default CDF row selected by
  `compound_mode_ctx(4, 4) = 7`.

* **§9.4 default CDF tables + §8.3.1 / §8.3.2 selection — palette /
  filter-intra / CFL subset (round 19).** Extends `cdf` with the
  filter-intra group (`Default_Filter_Intra_Mode_Cdf` 5-value,
  `Default_Filter_Intra_Cdf[ BLOCK_SIZES ]` binary with the §9.4
  "indices 10–15 / 20–21 never used" filler preserved), the palette
  group (`Default_Palette_Y_Mode_Cdf[ 7 ][ 3 ]`,
  `Default_Palette_Uv_Mode_Cdf[ 2 ]`,
  `Default_Palette_{Y,Uv}_Size_Cdf[ 7 ]` 7-value, and the fourteen
  `Default_Palette_Size_{2..8}_{Y,Uv}_Color_Cdf[ 5 ]` colour-index
  tables whose symbol count grows with `PaletteSize`), and the CFL
  group (`Default_Cfl_Sign_Cdf` 8-value,
  `Default_Cfl_Alpha_Cdf[ 6 ][ 17 ]` 16-value) — all transcribed
  verbatim from §9.4 with the §3 constants `BLOCK_SIZES = 22`,
  `INTRA_FILTER_MODES = 5`, `PALETTE_BLOCK_SIZE_CONTEXTS = 7`,
  `PALETTE_{Y,UV}_MODE_CONTEXTS = 3/2`, `PALETTE_SIZES = 7`,
  `PALETTE_COLORS = 8`, `PALETTE_COLOR_CONTEXTS = 5`,
  `CFL_JOINT_SIGNS = 8`, `CFL_ALPHABET_SIZE = 16`,
  `CFL_ALPHA_CONTEXTS = 6` and the `Palette_Color_Context` /
  `Palette_Color_Hash_Multipliers` additional-tables arrays re-exposed
  at the crate root. `new_from_defaults` performs the §8.3.1 init step
  for every array. The §8.3.2 selection surfaces ten `&mut [u16]`
  accessors — `filter_intra_cdf(MiSize)`, `filter_intra_mode_cdf()`,
  `palette_y_mode_cdf(bsizeCtx, ctx)`, `palette_uv_mode_cdf(ctx)`,
  `palette_{y,uv}_size_cdf(bsizeCtx)`,
  `palette_{y,uv}_color_cdf(PaletteSize, ctx)` (size-keyed, `Option`
  for out-of-range), `cfl_sign_cdf()`, `cfl_alpha_cdf(ctx)`. Scalar
  §8.3.2 helpers `palette_y_mode_ctx(above, left)`,
  `palette_uv_mode_ctx(PaletteSizeY)`,
  `palette_color_ctx(ColorContextHash)` (the `Palette_Color_Context`
  lookup returning `None` for the spec's `-1` sentinels), and
  `cfl_alpha_{u,v}_ctx(signU, signV)` (`(signU-1)*3+signV` /
  `(signV-1)*3+signU`, with the §8.3.2 `ctx == cfl_alpha_signs - 2`
  identity checked) compute each `ctx` from scalar neighbour inputs.
  8 new unit tests (190 in src/, up from 182): full dimension audit of
  every palette / filter-intra / CFL table with the §8.2.6
  `cdf[N-1] == 32768` / `cdf[N] == 0` invariant on every row;
  hand-picked byte-exact spot-checks; §8.3.1 init copy; size-keyed
  colour-CDF selection (row length `K+1` for size `K`, `None` outside
  `2..=8`); the palette/CFL context formulas; the
  `Palette_Color_Context` sentinel map; and an end-to-end §8.2
  `SymbolDecoder` decode driving the 16-value `cfl_alpha_u` default
  CDF row selected by `cfl_alpha_u_ctx(1, 0) = 0`.

* **§9.4 default CDF tables + §8.3.2 selection — transform-size
  subset (round 20).** Extends `cdf` with the five transform-size
  default tables (`Default_Tx_8x8_Cdf[ 3 ][ 3 ]`,
  `Default_Tx_16x16_Cdf[ 3 ][ 4 ]`, `Default_Tx_32x32_Cdf[ 3 ][ 4 ]`,
  `Default_Tx_64x64_Cdf[ 3 ][ 4 ]`, `Default_Txfm_Split_Cdf[ 21 ][ 3 ]`)
  — all transcribed verbatim from §9.4 with the §3 constants
  `TX_SIZE_CONTEXTS = 3`, `TX_SIZES = 5`, `MAX_TX_DEPTH = 2`,
  `TXFM_PARTITION_CONTEXTS = 21`. `new_from_defaults` performs the
  §8.3.1 init step for every array. Two `&mut [u16]` accessors
  surface the §8.3.2 selection: `tx_depth_cdf(maxTxDepth, ctx)` picks
  `TileTx{8x8,16x16,32x32,64x64}Cdf[ ctx ]` per the §8.3.2 paragraph's
  four-way `maxTxDepth ∈ {1, 2, 3, 4}` switch (returning `None` when
  `maxTxDepth == 0`, the syntax-not-read case), and
  `txfm_split_cdf(ctx)` picks `TileTxfmSplitCdf[ ctx ]`. Scalar
  context helpers `tx_depth_ctx(aboveW, leftH, maxTxWidth, maxTxHeight)`
  (the `(aboveW >= maxTxWidth) + (leftH >= maxTxHeight)` formula) and
  `txfm_split_ctx(above, left, txSzSqrUp, maxTxSz)` (the
  `(txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 + above + left`
  formula) compute the `ctx` from scalar inputs the §5.11.15 /
  §5.11.16 syntax supplies. 8 new unit tests (198 in src/, up from
  190): every transform-size table's `cdf[N-1] == 32768` /
  `cdf[N] == 0` invariant, dimension audit against the §3 constants;
  byte-anchor spot-checks on every table's first/last entries;
  §8.3.1 init-copy independence; `tx_depth_cdf` four-way selection
  with row-length assertions; the `tx_depth_ctx` formula across all
  four neighbour combinations; the `txfm_split_ctx` formula
  walked term-by-term plus an exhaustive bounds sweep over the
  reachable `(above, left, maxTxSz, txSzSqrUp)` tuples; and two
  end-to-end §8.2 `SymbolDecoder` decodes — one driving the
  3-symbol `TileTx16x16Cdf[ 2 ]` row selected by
  `tx_depth_ctx(16, 16, 16, 16) = 2`, the other driving the
  binary `TileTxfmSplitCdf[ 2 ]` row selected by
  `txfm_split_ctx(true, true, 4, 4) = 2`.

* **§9.4 default CDF tables + §8.3.2 selection — inter-frame
  transform-type subset (round 21).** Extends `cdf` with the three
  inter-frame transform-type default tables
  (`Default_Inter_Tx_Type_Set1_Cdf[ 2 ][ 17 ]` — the 16-symbol full
  set for 4x4 / 8x8 inter blocks reaching `TX_SET_INTER_1`;
  `Default_Inter_Tx_Type_Set2_Cdf[ 13 ]` — the 12-symbol 16x16-only
  set for `TX_SET_INTER_2`; `Default_Inter_Tx_Type_Set3_Cdf[ 4 ][ 3 ]`
  — the 2-symbol `{ IDTX, DCT_DCT }` reduced set for
  `TX_SET_INTER_3`) — all transcribed verbatim from §9.4 with the §3
  constants `TX_TYPES = 16`, `TX_TYPES_SET2 = 12`, `TX_TYPES_SET3 = 2`,
  `INTER_TX_TYPE_SET1_SIZES = 2`, `INTER_TX_TYPE_SET3_SIZES = 4` and
  the §6.10.19 transform-set tag constants `TX_SET_DCTONLY = 0` /
  `TX_SET_INTER_1 = 1` / `TX_SET_INTER_2 = 2` / `TX_SET_INTER_3 = 3`.
  `new_from_defaults` performs the §8.3.1 init step for every array.
  An `&mut [u16]` accessor `inter_tx_type_cdf(set, tx_size_sqr)`
  surfaces the §8.3.2 selection — the three-way
  `TileInterTxTypeSet{1,2,3}Cdf` switch keyed by the §5.11.48 `set`
  return — yielding `None` for `TX_SET_DCTONLY` (where §5.11.47
  forces `TxType = DCT_DCT` and `inter_tx_type` is not read) and for
  any unreachable `(set, tx_size_sqr)` combination. The scalar
  §5.11.48 helper `inter_tx_type_set(tx_sz_sqr, tx_sz_sqr_up,
  reduced_tx_set)` computes the `set ∈ { TX_SET_DCTONLY,
  TX_SET_INTER_1, TX_SET_INTER_2, TX_SET_INTER_3 }` from the
  `Tx_Size_Sqr[ txSz ]` / `Tx_Size_Sqr_Up[ txSz ]` / `reduced_tx_set`
  inputs the surrounding §5.11.47 syntax supplies. 6 new unit tests
  (204 in src/, up from 198): every inter-tx-type table's
  `cdf[N-1] == 32768` / `cdf[N] == 0` invariant, dimension audit
  against the §3 constants; byte-anchor spot-checks on every table's
  first / last entries; §8.3.1 init-copy independence with a
  mutate-doesn't-touch-source assertion; `inter_tx_type_cdf`
  three-way selection with row-length assertions and unreachable-set
  / out-of-range coverage; the `inter_tx_type_set` formula walked
  across every reachable `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)`
  triple (including the rectangular `TX_4X8` / `TX_16X32` cases
  where `tx_sz_sqr != tx_sz_sqr_up`); and one end-to-end §8.2
  `SymbolDecoder` decode driving the 2-symbol
  `TileInterTxTypeSet3Cdf[ 1 ]` row selected by
  `inter_tx_type_set(1, 1, true) = TX_SET_INTER_3`.

* **§9.4 default CDF tables + §8.3.2 selection — intra-frame
  transform-type subset (round 137).** Completes the §6.10.19
  transform-set coverage started in round 21 with the two intra-frame
  default tables (`Default_Intra_Tx_Type_Set1_Cdf[ 2 ][ INTRA_MODES ][ 8 ]`
  — the 7-symbol full intra set for 4x4 / 8x8 intra blocks reaching
  `TX_SET_INTRA_1`; `Default_Intra_Tx_Type_Set2_Cdf[ 3 ][ INTRA_MODES ][ 6 ]`
  — the 5-symbol reduced intra set for 4x4 / 8x8 / 16x16 intra blocks
  reaching `TX_SET_INTRA_2`) — both transcribed verbatim from §9.4 with
  the §3 constants `TX_SET_INTRA_1 = 1`, `TX_SET_INTRA_2 = 2`,
  `TX_TYPES_INTRA_SET1 = 7`, `TX_TYPES_INTRA_SET2 = 5`,
  `INTRA_TX_TYPE_SET1_SIZES = 2`, `INTRA_TX_TYPE_SET2_SIZES = 3`, and
  the §8.3.2 `Filter_Intra_Mode_To_Intra_Dir[ INTRA_FILTER_MODES ]`
  table mapping each filter mode to a directional anchor
  (`{ DC_PRED, V_PRED, H_PRED, D157_PRED, DC_PRED }`).
  `new_from_defaults` performs the §8.3.1 init step for both arrays.
  An `&mut [u16]` accessor `intra_tx_type_cdf(set, tx_size_sqr, intra_dir)`
  surfaces the §8.3.2 selection — the two-way
  `TileIntraTxTypeSet{1,2}Cdf` switch keyed by the §5.11.48 `set`
  return and indexed on the `intraDir` axis — yielding `None` for
  `TX_SET_DCTONLY` and for any unreachable `(set, tx_size_sqr,
  intra_dir)` combination. Two scalar §5.11.48 / §8.3.2 helpers
  complete the path: `intra_tx_type_set(tx_sz_sqr, tx_sz_sqr_up,
  reduced_tx_set)` computes the `set ∈ { TX_SET_DCTONLY,
  TX_SET_INTRA_1, TX_SET_INTRA_2 }` from the surrounding §5.11.47
  syntax (differing from the inter branch in two places:
  `txSzSqrUp == TX_32X32` itself routes to `TX_SET_DCTONLY` rather
  than `TX_SET_INTER_3`, and `txSzSqr == TX_16X16` routes to
  `TX_SET_INTRA_2` rather than `TX_SET_INTER_2`), and
  `intra_dir(use_filter_intra, y_mode, filter_intra_mode)` derives
  the `intraDir` axis from the `use_filter_intra` flag plus
  the `YMode` / `filter_intra_mode` pair. 7 new unit tests:
  every intra-tx-type table's `cdf[N-1] == 32768` / `cdf[N] == 0`
  invariant + dimension audit against the §3 constants; byte-anchor
  spot-checks on every table's first / last entries plus the explicit
  flat-distribution check for `Set2` sizes 0..=1; §8.3.1 init-copy
  independence with a mutate-doesn't-touch-source assertion;
  `intra_tx_type_cdf` two-way selection over every reachable
  `(set, tx_size_sqr, intra_dir)` triple with row-length assertions
  and unreachable-set / out-of-range coverage; the
  `intra_tx_type_set` formula walked across every reachable
  `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)` triple (including the
  rectangular `TX_4X8` / `TX_16X32` cases and the `txSzSqrUp == TX_32X32`
  short-circuit specific to the intra branch); the `intra_dir`
  derivation tested for the pass-through (`use_filter_intra == 0`)
  and the `Filter_Intra_Mode_To_Intra_Dir` lookup branches; and one
  end-to-end §8.2 `SymbolDecoder` decode driving the 5-symbol
  `TileIntraTxTypeSet2Cdf[ 2 ][ 0 ]` row selected by
  `intra_tx_type_set(2, 2, false) = TX_SET_INTRA_2` +
  `intra_dir(false, DC_PRED, _) = DC_PRED`. This completes §9.4's
  transform-type coverage (intra + inter); remaining §9.4 work is
  the `init_coeff_cdfs` coefficient set (`Default_Coeff_Base_Eob_Cdf`
  / `Default_Coeff_Base_Cdf` / `Default_Coeff_Br_Cdf`) and the
  inter-intra group.

Validation: all 16 IVF fixtures under
`docs/video/av1/fixtures/` (`tiny-i-only-16x16-prof0`,
`i-only-64x64-prof0`, `profile-1-yuv444-8bit`,
`profile-2-yuv422-10bit`, `profile-2-yuv422-12bit`,
`monochrome-grey-only`, `super-resolution`, `screen-content-tools`,
`film-grain-on`, `superblocks-128`, `tile-cols-2-rows-1`,
`show-existing-frame`, `lossless-i-only`, `i-frame-then-p-64x64`,
`obu-with-extension-headers`, `profile-0-yuv420-8bit`) round-trip
both the first sequence header bit-exact against the `SEQ_HEADER`
line captured in each fixture's `trace.txt`, and the first frame
OBU's leading uncompressed-header slice bit-exact against the
`FRAME_HEADER idx=0` line in the same trace. Round 4 extends the
trace columns asserted per fixture from 12 to 17 (adding `w`, `h`,
`use_superres`, `coded_denom`, plus a derived assertion ladder
computing `superres_denom` / post-superres `frame_width` /
`mi_cols` / `mi_rows` from the §5.9.5 / §5.9.8 / §5.9.9 formulas).
The `super-resolution` fixture exercises the §5.9.8 downscale
(`UpscaledWidth = 128`, `coded_denom = 3` ⇒ `SuperresDenom = 12`,
post-downscale `FrameWidth = (128 * 8 + 6) / 12 = 85`,
`MiCols = 22`); every other fixture is `use_superres == 0` with
`FrameWidth == UpscaledWidth`.

Both the intra and the inter `uncompressed_header()` are now parsed
end-to-end (through `film_grain_params()`). What remains: the
`set_frame_refs()` short-signaling ordering is implemented and unit-
tested but not yet exercised by a corpus fixture (no short-signaling
bitstream exists in `docs/video/av1/fixtures/`); the
`frame_size_with_refs()` `found_ref == 1` branch is implemented but
likewise unexercised by the corpus; the §5.9.2 OrderHints[] /
RefFrameSignBias[] derivation and the §7.20 reference frame update
process (which would *store* a decoded frame's dimensions / hints back
into `RefInfo` across frames) are session-state concerns left to the
decode pipeline; and tile-content decode (motion vectors, transform /
quantisation, in-loop filters, film-grain synthesis) is unstarted. The
§8.2 symbol (arithmetic) decoder — the engine all tile-content reads run
on — now exists as a standalone, byte-exact `SymbolDecoder` (round 15);
round 16 lands the §9.4 default CDF tables and the §8.3.1 / §8.3.2
selection for a bounded **intra-frame mode / partition** syntax group
(`intra_frame_y_mode` / `partition` / `skip` / `segment_id`); round 17
extends the same `TileCdfContext` shape with the **motion-vector
component** subset (`mv_joint` / `mv_sign` / `mv_class` /
`mv_class0_bit` / `mv_class0_fr` / `mv_class0_hp` / `mv_bit` / `mv_fr`
/ `mv_hp`) and the §5.11.31 `MvCtx` derivation; round 18 extends it
again with the **inter-mode / reference-frame** subset (`new_mv` /
`zero_mv` / `ref_mv` / `drl_mode` / `is_inter` / `comp_mode` /
`skip_mode` / `comp_ref{,_p1,_p2}` / `comp_bwdref{,_p1}` /
`single_ref_p{1..6}` / `compound_mode` / `comp_ref_type` /
`uni_comp_ref{,_p1,_p2}`) plus the §8.3.2 context helpers
`is_inter_ctx` / `skip_mode_ctx` / `ref_count_ctx` / `compound_mode_ctx`.
round 20 extends the same `TileCdfContext` shape with the
**transform-size** subset (`tx_depth` over the four
per-`maxTxDepth` `Default_Tx_{8,16,32,64}x{8,16,32,64}_Cdf` arrays
and `txfm_split` over `Default_Txfm_Split_Cdf`) plus the §8.3.2
`tx_depth_ctx` / `txfm_split_ctx` derivations. Round 21 lands the
**inter-frame transform-type** subset (`inter_tx_type` over
`Default_Inter_Tx_Type_Set{1,2,3}_Cdf`) plus the §5.11.48
`inter_tx_type_set` switch driving the §8.3.2 three-way
`TileInterTxTypeSet{1,2,3}Cdf` selection. Round 22 lands the
**inter-frame interpolation-filter** subset (`interp_filter` over
`Default_Interp_Filter_Cdf` — 16 contexts × 3 cumulative
frequencies) plus the §8.3.2 four-branch `interp_filter_ctx`
formula (the `((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4`
base plus the leftType / aboveType / NONE-match folding). Round 23
lands the **motion-mode** subset (`motion_mode` over
`Default_Motion_Mode_Cdf` — 22 block-size rows × 3 cumulative
frequencies) plus its §8.3.2 selection — a straight
`TileMotionModeCdf[ MiSize ]` index with no neighbour-context
arithmetic; the §6.10.26 enumeration `MOTION_MODES = 3`
(`SIMPLE` / `OBMC` / `LOCALWARP`) is added as a new §3 constant.
Round 24 lands the **compound-prediction** subset — the three
default tables `Default_Comp_Group_Idx_Cdf` /
`Default_Compound_Idx_Cdf` (binary, 6 contexts each) and
`Default_Compound_Type_Cdf` (22 block-size rows × 2 cumulative
frequencies) — plus their §8.3.2 selections:
`comp_group_idx_cdf(ctx)` / `compound_idx_cdf(ctx)` take the
precomputed neighbour-derived context (whose arithmetic stays in
the future tile walk) and `compound_type_cdf(mi_size)` is a
straight `TileCompoundTypeCdf[ MiSize ]` index; the §3 constants
`COMPOUND_TYPES = 2`, `COMP_GROUP_IDX_CONTEXTS = 6` and
`COMPOUND_IDX_CONTEXTS = 6` are added.
Round 134 lands the **inter-frame intra-mode** subset — the three
default tables `Default_Y_Mode_Cdf` (4 block-size-group contexts ×
13 cumulative frequencies), `Default_Uv_Mode_Cfl_Not_Allowed_Cdf`
(13 `YMode` rows × 13) and `Default_Uv_Mode_Cfl_Allowed_Cdf`
(13 `YMode` rows × 14) — plus their §8.3.2 selections:
`y_mode_cdf(ctx)` indexes `TileYModeCdf[ Size_Group[ MiSize ] ]`
(the `Size_Group` table + `size_group()` helper land alongside),
and `uv_mode_cdf(cfl_allowed, y_mode)` picks the cfl-allowed /
cfl-not-allowed variant by the resolved flag (the `Lossless` /
`get_plane_residual_size` / `Max(Block_Width, Block_Height) <= 32`
derivation stays in the future tile walk) then indexes by `YMode`;
the §3 constants `BLOCK_SIZE_GROUPS = 4`,
`UV_INTRA_MODES_CFL_NOT_ALLOWED = 13` and
`UV_INTRA_MODES_CFL_ALLOWED = 14` are added.
Round 135 lands the **angle-delta** subset — the default table
`Default_Angle_Delta_Cdf` (8 directional-mode rows × 7 cumulative
frequencies) — plus its §8.3.2 selection: `angle_delta_cdf(mode)`
indexes `TileAngleDeltaCdf[ mode - V_PRED ]` (the
`TileAngleDeltaCdf[ YMode - V_PRED ]` / `[ UVMode - V_PRED ]`
selection shared by `angle_delta_y` / `angle_delta_uv`), returning
`None` for non-directional modes; the §3 constants
`DIRECTIONAL_MODES = 8`, `MAX_ANGLE_DELTA = 3` and the
directional-mode base `V_PRED = 1` are added.
Round 136 lands the **coefficient-token entry sub-group** — the
`init_coeff_cdfs` gateway to tile-content decode: the
transform-block skip flag `Default_Txb_Skip_Cdf` (4 q-contexts ×
5 transform sizes × 13 skip contexts), the end-of-block position
classes `Default_Eob_Pt_{16,32,64,128,256}_Cdf` (per-plane,
per-`isInter`) plus the no-`isInter`-axis
`Default_Eob_Pt_{512,1024}_Cdf`, the binary `Default_Eob_Extra_Cdf`
(per transform size / plane / 9 EOB contexts) and the binary
`Default_Dc_Sign_Cdf` (per plane / 3 contexts, in the §9.4
`128 * N` fixed-point form), all transcribed verbatim from §9.4.
Unlike the non-coeff CDFs these are reset by the separate
`TileCdfContext::init_coeff_cdfs`, which derives the q-context
`idx` from `base_q_idx` (`coeff_cdf_q_ctx`: `<=20→0`, `<=60→1`,
`<=120→2`, else `3`) and copies `Default_*_Cdf[ idx ]` into the
working arrays; the §8.3.2 selectors `txb_skip_cdf` /
`eob_pt_*_cdf` / `eob_extra_cdf` / `dc_sign_cdf` land alongside,
and the §3 constants `PLANE_TYPES = 2`, `COEFF_CDF_Q_CTXS = 4`,
`TXB_SKIP_CONTEXTS = 13`, `EOB_COEF_CONTEXTS = 9`,
`DC_SIGN_CONTEXTS = 3` are added.
Round 137 completes §9.4's transform-type coverage by adding the
**intra-frame transform-type** subset (`intra_tx_type` over
`Default_Intra_Tx_Type_Set{1,2}_Cdf`) plus the §5.11.48
`intra_tx_type_set` switch driving the §8.3.2 two-way
`TileIntraTxTypeSet{1,2}Cdf` selection and the §8.3.2 `intra_dir`
helper that derives the `intraDir` axis from `use_filter_intra` +
`YMode` / `filter_intra_mode` via the
`Filter_Intra_Mode_To_Intra_Dir` table.
Round 138 lands the first member of the `coeff_base` /
`coeff_base_eob` / `coeff_br` braid — the **`coeff_base_eob`
sub-group** — by adding `Default_Coeff_Base_Eob_Cdf`
(`[COEFF_CDF_Q_CTXS=4][TX_SIZES=5][PLANE_TYPES=2][SIG_COEF_CONTEXTS_EOB=4][4]`)
transcribed verbatim from §9.4. `coeff_base_eob` codes the base
level of the last non-zero coefficient (the base level is
`coeff_base_eob + 1`, restricted to 1, 2, or 3, so only three
symbols are coded). `init_coeff_cdfs` grows the new
`self.coeff_base_eob = DEFAULT_COEFF_BASE_EOB_CDF[ idx ]` copy on
the `base_q_idx`-derived `idx`, and the §8.3.2 selector
`coeff_base_eob_cdf(tx_sz_ctx, ptype, ctx)` surfaces the
`TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]` lookup. The §8.3.2
ctx derivation
(`get_coeff_base_ctx() - SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB`)
belongs to the not-yet-implemented tile-content walk and is
deferred along with `Default_Coeff_Base_Cdf` and
`Default_Coeff_Br_Cdf`; the §3 constant `SIG_COEF_CONTEXTS_EOB = 4`
is added.
Round 139 lands the second member of the braid — the **`coeff_base`
sub-group** — by adding `Default_Coeff_Base_Cdf`
(`[COEFF_CDF_Q_CTXS=4][TX_SIZES=5][PLANE_TYPES=2][SIG_COEF_CONTEXTS=42][5]`,
1680 5-entry rows = 16800 bytes; declared `static` to satisfy
`clippy::large_const_arrays`) transcribed verbatim from §9.4.
`coeff_base` codes the base level of every non-EOB coefficient — the
4-symbol alphabet `0..3`, so each row carries 4 cumulative
frequencies plus the §8.3 adaptation counter. `init_coeff_cdfs`
grows the new `self.coeff_base = DEFAULT_COEFF_BASE_CDF[ idx ]`
copy on the `base_q_idx`-derived `idx`, and the §8.3.2 selector
`coeff_base_cdf(tx_sz_ctx, ptype, ctx)` surfaces the
`TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]` lookup. The §3
constant `SIG_COEF_CONTEXTS = 42` is added (the §3 partition tag
`SIG_COEF_CONTEXTS_2D = 26` splits this range between the
two-dimensional scan prefix and the 1D horizontal- / vertical-only
tails). Just as in r138, the largest `(TX_SIZE = TX_64X64, ptype =
chroma)` slice is the flat `{8192, 16384, 24576, 32768, 0}`
placeholder for every q-context and ctx value — a sentinel for an
unreachable chroma row at the largest TX size — and is locked down
by an exhaustive byte-equality test.
Round 140 lands the LAST member of the braid — the **`coeff_br`
sub-group** — by adding `Default_Coeff_Br_Cdf`
(`[COEFF_CDF_Q_CTXS=4][TX_SIZES=5][PLANE_TYPES=2][LEVEL_CONTEXTS=21][BR_CDF_SIZE + 1 = 5]`,
840 5-entry rows = 8400 bytes; declared `static` to satisfy
`clippy::large_const_arrays`) transcribed verbatim from §9.4. With
this table all three coefficient-CDF braid members are landed.
`coeff_br` codes the per-coefficient base-range increment used to
push a level above `NUM_BASE_LEVELS`: each read codes a value in
`0..BR_CDF_SIZE = 4`, and §5.11.39 stacks
`COEFF_BASE_RANGE / (BR_CDF_SIZE - 1)` such reads per coefficient.
`init_coeff_cdfs` grows the new
`self.coeff_br = DEFAULT_COEFF_BR_CDF[ idx ]` copy on the
`base_q_idx`-derived `idx`, and the §8.3.2 selector
`coeff_br_cdf(tx_sz_ctx, ptype, ctx)` surfaces the
`TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][ ctx ]` lookup
with the `TX_32X32 = 3` clamp built in (so any `txSzCtx` is
accepted; only `ptype` / `ctx` are bounds-checked). New §3
constants `LEVEL_CONTEXTS = 21` and `BR_CDF_SIZE = 4`. Mirroring
r138 / r139, the largest `(TX_SIZE = TX_64X64, ptype = chroma)`
slice is the flat `{8192, 16384, 24576, 32768, 0}` placeholder for
every q-context and ctx value, locked down by an exhaustive
byte-equality test. The next gate is the §8.3.2
`get_coeff_base_ctx()` / `get_br_ctx()` neighbour-derivation helpers,
deferred to a different round because they need tile-content walker
state.
Round 141 lands those **§8.3.2 `get_coeff_base_ctx()` /
`get_br_ctx()` neighbour-derivation helpers** — the per-coefficient
`ctx` computation that feeds the r138–r140 selectors. Both are free
functions that take the coefficient-magnitude array `Quant` plus
scalar transform / position state (`tx_size`, `tx_class`, `pos`,
`c`, `is_eob`) and return the `ctx` index; they own the §8.3.2
neighbour scan only — the tile-content walker that produces `Quant`
and the `compute_tx_type()` derivation is the next gate.
`get_coeff_base_ctx()` scans `Sig_Ref_Diff_Offset`
(`SIG_REF_DIFF_OFFSET_NUM = 5` offsets) accumulating
`Min(Abs(Quant[(refRow<<bwl)+refCol]), 3)` over in-bounds neighbours
(`refRow < height && refCol < width`, `bwl =
Tx_Width_Log2[Adjusted_Tx_Size[txSz]]`), forms `ctx = Min((mag+1)>>1,
4)`, then routes through the 2D `Coeff_Base_Ctx_Offset[txSz][Min(row,
4)][Min(col,4)]` branch (with the `row==0 && col==0 -> 0` early
return) or the 1D `Coeff_Base_Pos_Ctx_Offset[Min(idx,2)]` branch;
the `isEob` path returns the `SIG_COEF_CONTEXTS-{4,3,2,1}` buckets per
`c` thresholds. A `get_coeff_base_eob_ctx()` wrapper applies the
§8.3.2 `- SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB` reduction.
`get_br_ctx()` scans `Mag_Ref_Offset_With_Tx_Class` (3 offsets)
accumulating `Min(Quant[refRow*txw+refCol],
COEFF_BASE_RANGE+NUM_BASE_LEVELS+1)` (no abs, distinct clamp, bound
`refRow < txh && refCol < (1<<bwl)`), forms `mag = Min((mag+1)>>1,
6)`, then `pos==0 -> mag`; 2D `+7/+14` on `(row<2 && col<2)`;
horizontal `+7/+14` on `col==0`; vertical `+7/+14` on `row==0`
(result in `0..LEVEL_CONTEXTS`). Adds the §3 constants
`SIG_COEF_CONTEXTS_2D = 26`, `SIG_REF_DIFF_OFFSET_NUM = 5`,
`NUM_BASE_LEVELS = 2`, `COEFF_BASE_RANGE = 12`, `TX_SIZES_ALL = 19`,
the `TX_CLASS_{2D,HORIZ,VERT}` tags, the `Adjusted_Tx_Size` /
`Tx_Width` / `Tx_Height` / `Tx_Width_Log2` size tables, and the
`Sig_Ref_Diff_Offset` / `Mag_Ref_Offset_With_Tx_Class` /
`Coeff_Base_Ctx_Offset` / `Coeff_Base_Pos_Ctx_Offset` tables, all
transcribed verbatim from the spec; a pure `get_tx_class()` helper
reduces the directional transform-type flags to a class. 12 new unit
tests (266 -> 278) pin each branch with hand-computed `ctx` values.
Round 142 follows up with the **§5.11.40 `compute_tx_type()`
derivation** — `compute_tx_type(plane, tx_sz, lossless, is_inter,
tx_set, mi_row, mi_col, block_x, block_y, subsampling_x,
subsampling_y, uv_mode, tx_types)` implements the full spec function
the tile-content walker reads before kicking off coefficient
decoding. `Lossless || Tx_Size_Sqr_Up[ txSz ] > TX_32X32`
short-circuits to `DCT_DCT`; `plane == 0` returns the
`TxTypes[ blockY ][ blockX ]` luma cache entry; `is_inter` chroma
reads the cache at
`(Max(MiRow, blockY << subsampling_y), Max(MiCol, blockX <<
subsampling_x))` then runs the §5.11.40 `is_tx_type_in_set`
admission filter; intra chroma reads `Mode_To_Txfm[UVMode]` then
runs the same filter. The caller supplies the §5.11.40 `txSet`
(i.e. the already-resolved `inter_tx_type_set` /
`intra_tx_type_set` result) and a closure over `TxTypes[y][x]` so
the helper does not bake in a particular storage shape — a dense 2D
array, a sparse map, or a `MiRow/MiCol`-relative tile-local view
all work. Adds the §6.10.16 size ordinals
`TX_4X4` / `TX_8X8` / `TX_16X16` / `TX_32X32` / `TX_64X64`
(replacing the previously locally-scoped `const TX_*` shadows
inside `inter_tx_type_set` / `intra_tx_type_set`), the §6.10.19
transform-type ordinals `DCT_DCT` through `H_FLIPADST` (16
entries), the `TX_SET_TYPES_INTRA = 3` / `TX_SET_TYPES_INTER = 4`
row-count constants, the `Tx_Size_Sqr_Up[ TX_SIZES_ALL ]`,
`Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]`,
`Tx_Type_In_Set_Intra[ 3 ][ TX_TYPES ]`, and
`Tx_Type_In_Set_Inter[ 4 ][ TX_TYPES ]` tables, all transcribed
verbatim from the spec. 10 new unit tests (278 -> 288) pin each
branch (lossless / `txSzSqrUp > TX_32X32` fallback, luma cache
read, inter-chroma `Max` lift + admission pass/fail, intra-chroma
`Mode_To_Txfm` + admission pass/fail + out-of-range, plus a
selector/admission closed-loop check).
Round 143 lands the **inter-intra group** — the three §9.4 default
CDFs read by the §5.11.28 `read_interintra_mode` syntax:
`Default_Inter_Intra_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ 3 ]`,
`Default_Inter_Intra_Mode_Cdf[ BLOCK_SIZE_GROUPS - 1 ][ INTERINTRA_MODES + 1 ]`,
and `Default_Wedge_Inter_Intra_Cdf[ BLOCK_SIZES ][ 3 ]`. Adds the
`INTERINTRA_MODES = 4` §3 constant (per §6.10.27 — `II_DC_PRED` /
`II_V_PRED` / `II_H_PRED` / `II_SMOOTH_PRED`) and the §8.3.2
`interintra_ctx(mi_size) = Size_Group[ MiSize ] - 1` mapping
(returning `None` for the `Size_Group[ MiSize ] == 0` rows that the
§5.11.28 syntax gate excludes — `MiSize < BLOCK_8X8`). The
`TileCdfContext` grows the `inter_intra` / `inter_intra_mode` /
`wedge_inter_intra` fields and gains the
`inter_intra_cdf(ctx)` / `inter_intra_mode_cdf(ctx)` /
`wedge_inter_intra_cdf(mi_size)` selectors. The wedge table's outer
dimension is transcribed full-width per the §9.4 listing; per its
note only indices `3..=9` (the `BLOCK_8X8`..`BLOCK_32X32` band) are
reachable and the other rows hold the placeholder
`{16384, 32768, 0}` row. 8 new unit tests (288 -> 296) pin the table
shapes / well-formedness / `Size_Group - 1` mapping / per-row
selector return value / working-copy independence, plus end-to-end
`SymbolDecoder` reads through `interintra`, `interintra_mode`, and
`wedge_interintra` rows.
Round 144 lands the **wedge-index CDF** — the §9.4
`Default_Wedge_Index_Cdf[ BLOCK_SIZES ][ WEDGE_TYPES + 1 ]` table
(p.435) and the matching §8.3.2 selection. `wedge_index` is the
16-symbol element read by both §5.11.28 `read_interintra_mode` (the
inter-intra wedge sub-branch, when `wedge_interintra == 1`) and
§5.11.29 `read_compound_type` (the inter-inter `COMPOUND_WEDGE`
branch). Adds the §3 constant `WEDGE_TYPES = 16` (the spec text reads
*"Number of directions for the wedge mask process"*). `TileCdfContext`
grows a `wedge_index` field and gains a
`wedge_index_cdf(mi_size) -> Option<&mut [u16]>` selector (straight
`TileWedgeIndexCdf[ MiSize ]` indexing). The table's outer dimension is
transcribed full-width per the §9.4 listing; per its note (p.436)
indices `0..2`, `10..17`, and `20..21` are never used in the first
dimension (matching the §3 `Wedge_Bits[ MiSize ] == 0` rows) and carry
the placeholder uniform CDF `{ 2048, 4096, …, 30720, 32768, 0 }` (step
`32768 / WEDGE_TYPES`). 6 new unit tests (296 -> 302) pin the §3
constant / table shape and values (cross-checked against the §3
`Wedge_Bits` table) / well-formedness / `init_non_coeff_cdfs` seeding /
selector return value with out-of-range rejection / working-copy
independence, plus an end-to-end `SymbolDecoder` read through a
`wedge_index` row from the reachable band.
Round 145 lands the §8.3.2 **`split_or_horz` / `split_or_vert`
cdf-derivation helpers** (p.362–363) — two pure functions that turn an
already-selected `partition` cdf into a 2-symbol binary cdf by folding
the §9.4 partition probabilities of the splittable plus orthogonal-axis
symbols into a single `psum`. Per the §8.3.2 note the disallowed
orthogonal partition's probability is folded into the split branch
(`split_or_horz` cannot return `PARTITION_VERT`; `split_or_vert` cannot
return `PARTITION_HORZ`). The `b_size != BLOCK_128X128` guard drops the
`PARTITION_*_4` term that the §9.4 `Default_Partition_W128_Cdf` row has
no entry for. Adds the §3 / §6.10.4 partition ordinal constants
(`PARTITION_NONE` through `PARTITION_VERT_4` plus `EXT_PARTITION_TYPES =
10`) and the block-size ordinal `BLOCK_128X128 = 15` that the §8.3.2
formulas reference by name. 10 new unit tests (302 -> 312) pin the
§6.10.4 ordinals against the spec table (p.172), validate the
W{16,32,64,128} row-length budget against the §8.3.2 indexing reach,
re-derive both helpers' `psum` inline against a known
`Default_Partition_W{16,32}_Cdf` row, exercise the `PARTITION_*_4`
omission for both helpers under `b_size == BLOCK_128X128`, sweep §8.2.6
well-formedness across every default partition cdf row, drive the §8.2
`SymbolDecoder` through both derived binary cdfs (`BLOCK_64X64` for
`split_or_horz`, `BLOCK_128X128` for `split_or_vert`), and reject the
disallowed `bsl == 1` W8 row with `None`.
Round 146 lands the §5.11.50 **`get_palette_color_context` derivation**
(p.103) — the function the §5.11.49 diagonal-walk reads at each
`palette_color_idx_*` position to derive the `ColorOrder[ PALETTE_COLORS ]`
permutation + `ColorContextHash` that flow back through
[`palette_color_ctx`] into the §8.3.2 cdf selector. Surface:
`palette_color_context_from_neighbors(left, above_left, above, n) ->
Option<PaletteColorContext>` (pure-scoring core taking the three optional
neighbour palette indices) and `get_palette_color_context(color_map,
stride, r, c, n) -> Option<PaletteColorContext>` (spec-faithful 2-D
entry that applies the §5.11.50 `r > 0` / `c > 0` boundary guards). The
partial selection sort is the §5.11.50 three-iteration loop that
promotes the top-scoring neighbours to the head of `ColorOrder` while
preserving the runners-up's ascending order; the hash is the
`Palette_Color_Hash_Multipliers`-weighted sum of the top three sorted
scores. 11 new unit tests (312 -> 323) cover the spec example (all-same
neighbour, hash 5, ctx 4), distinct left/above (hash 6, ctx 3), the
partial-sort swap (two-of-three neighbours sharing an index, hash 6,
ctx 3), three distinct neighbours (hash 8, ctx 1), the no-neighbour
identity, palette-size / neighbour-value rejection, the full
spec-realisable combinatorial sweep across every `(left, above_left,
above)` at every palette size (every reachable hash maps to a
`Some(_)` ctx; the `-1` entries 0/1/3/4 are unreachable), 2-D
entry-point equivalence (interior + top-left corner + top-row-only +
left-column-only positions), 2-D boundary rejection (zero stride / OOB
column / OOB palette size / OOB row), and an end-to-end `SymbolDecoder`
read through the `palette_color_idx_y` default cdf selected by the
derivation -> `palette_color_ctx` -> `palette_y_color_cdf` chain.
The remaining §8.3.2 selections (the tile-content walker plumbing
that wires `get_palette_color_context` into the §5.11.49 diagonal walk,
plus the corresponding wedge / inter / intra walks for the unwalked
syntax elements …) are a mechanical followup against the same
`TileCdfContext` shape.
Round 147 lands the §5.11.49 **`palette_tokens( )` per-plane diagonal
walker** (p.101–102) — the caller-facing entry that drives the §5.11.50
colour-context derivation across an anti-diagonal walk, decodes one
`palette_color_idx_{y,uv}` per `(i - j, j)` against the cdf row picked
by `palette_color_ctx`, remaps through `ColorOrder[idx]`, and replicates
the on-screen last column / last row into the block's border band.
Surface: `palette_tokens_plane(dec, tile_ctx, plane, palette_size,
block_w, block_h, onscreen_w, onscreen_h, color_index_map, color_map,
stride) -> Result<(), Error>` with `PalettePlane::{Y, Uv}` picking
between `palette_y_color_cdf` and `palette_uv_color_cdf`. The
chroma-subsampling adjustments (`blockWidth >> subsampling_x` and the
`< 4` bump) are the caller's responsibility — they belong to the
§5.11.49 outer-control flow, not the walker. Two new `Error` variants
surface caller-bug preconditions (`InvalidPaletteWalkArgs`) and the
§5.11.50 unreachable hash slots (`PaletteColorContextUnmapped`); the
`SymbolDecoder` underflow path still propagates as `UnexpectedEnd`.
11 new unit tests (323 -> 334) cover every caller-bug rejection, a 2x2
on-screen walk that writes every cell with no border-fill, the
horizontal / vertical / combined border-fill paths on a 2x2 / 4x4
shape, a rectangular shape sweep over every `(onscreen_w, onscreen_h)`
in `1..=4 × 1..=4`, the UV plane adapting only the UV cdf family, the
chroma-subsampled UV / Y shape parity, the `ColorOrder[idx]` remap on
the 2x2 edge positions, the degenerate 1-wide-block shape, and
`read_symbol` underflow propagating as `UnexpectedEnd` (not as a
walker-side caller-bug variant).
Round 148 stages the §9.3 **block-size conversion tables** (av1-spec
p.400–401) that the §5.11.49 caller needs to derive `block_w` / `block_h`
from a `MiSize`. The four `BLOCK_SIZES`-indexed lookup tables
(`MI_WIDTH_LOG2`, `MI_HEIGHT_LOG2`, `NUM_4X4_BLOCKS_WIDE`,
`NUM_4X4_BLOCKS_HIGH`) are transcribed verbatim with the spec
ordering (16 square/rectangular entries `BLOCK_4X4` ..
`BLOCK_128X128` followed by the seven `1:4` / `4:1` aspect-ratio
entries `BLOCK_4X16` .. `BLOCK_64X16`). The §3 constants `MI_SIZE = 4`
and `MI_SIZE_LOG2 = 2` land alongside them so the §9.3
`Block_Width[ x ] = 4 * Num_4x4_Blocks_Wide[ x ]` identity is encoded
as `NUM_4X4_BLOCKS_WIDE[ x ] << MI_SIZE_LOG2` rather than duplicated
as a numeric table. Six new `MiSize`-keyed accessors round-out the
surface: `block_width`, `block_height`, `num_4x4_blocks_wide`,
`num_4x4_blocks_high`, `mi_width_log2`, `mi_height_log2` — each a
`const fn` with a `debug_assert!(mi_size < BLOCK_SIZES)` bound. These
feed the §5.11.49 `palette_tokens_plane` caller staged in r147 and
unblock the wider §5.x reconstruction call sites (`bw4 =
Num_4x4_Blocks_Wide[ MiSize ]`) the parser will surface once
`read_block` is wired. 10 new unit tests (334 -> 344) cover the four
§9.3 tables pinned byte-for-byte at `BLOCK_SIZES = 22`, the §3
`MI_SIZE == 1 << MI_SIZE_LOG2` identity, the `Num_4x4_Blocks_* == 1
<< Mi_*_Log2` identity per §9.3, the canonical 22-entry expected
width/height vectors, the square diagonal `BLOCK_4X4` ..
`BLOCK_128X128` resolving to `n×n` luma sizes, the §5.11.46
`bsizeCtx` derivation staying inside `0..PALETTE_BLOCK_SIZE_CONTEXTS`
for every `MiSize` inside the §5.11.46 palette syntax gate
(`MiSize >= BLOCK_8X8 && Block_Width <= 64 && Block_Height <= 64`),
and a §5.11.49 caller data-flow pin confirming `block_width(mi_size)`
/ `block_height(mi_size)` are inside `8..=64` at the palette-minimum
(`BLOCK_8X8`) and palette-maximum (`BLOCK_64X64`) `MiSize` values
the gate admits.

Round 149 wires the §5.11.49 **caller-side argument derivation**
(av1-spec p.101–102) on top of the r148 tables. The new
`palette_tokens_args(mi_size, mi_row, mi_col, mi_rows, mi_cols, plane,
subsampling_x, subsampling_y) -> Option<PaletteTokensArgs>` helper
returns the four `palette_tokens_plane` size arguments
(`block_w`, `block_h`, `onscreen_w`, `onscreen_h`) for one plane, and
a new `BLOCK_8X8` constant (`3`, the §5.11.46 palette gate) sits
alongside it. Y branch returns the §9.3 dimensions clipped by
`Min(.., (MiRows - MiRow) * MI_SIZE)` / `Min(.., (MiCols - MiCol)
* MI_SIZE)`. UV branch applies the §5.11.49 `>> subsampling_{x,y}`
shift and the `<4`-bump (`block_w += 2; onscreen_w += 2` when post-
shift `block_w < 4`, ditto height); the bump preserves the walker's
`onscreen_* <= block_*` invariant because it adds the same `2` to
both. The helper returns `None` for any §5.11.46 palette-gate
violation (sub-`BLOCK_8X8` size, `block_w > 64`, `block_h > 64`,
out-of-table index), out-of-bounds `mi_row` / `mi_col`, zero mi-grid,
or out-of-range subsampling flag — safe to call defensively from a
not-yet-gated caller. 15 new unit tests (344 -> 359) cover
`BLOCK_8X8`-row pinning; Y-plane fully-on-screen / right-edge /
bottom-edge clipping; UV 4:2:0 minimum block + large block;
UV 4:2:0 width-`<4`-bump (`BLOCK_4X16`) and height-`<4`-bump
(`BLOCK_16X4`); UV 4:2:2 + UV 4:4:4 shape; UV right-edge clip carry-
through; an exhaustive sweep proving `1 <= onscreen_* <= block_* <=
64` over every palette-eligible `(MiSize, sub_x, sub_y, plane)`
combination; palette-gate + caller-bug rejection paths; and an end-
to-end shape test that feeds the helper's output straight into
`palette_tokens_plane` against the §9.4 default palette CDFs,
confirming the walker's `InvalidPaletteWalkArgs` guard never fires
on conformant arguments. This closes the data-flow gap pinned by the
r147 follow-up test and leaves `read_block` clear to call
`palette_tokens` once the parser surfaces the variables.
`decode_av1` and `encode_av1` still return `Error::NotImplemented`.

Round 150 stages the §9.3 **`Partition_Subsize[ 10 ][ BLOCK_SIZES ]`**
lookup (av1-spec p.402–403) plus the §3 enumeration of all 22 named
`BLOCK_*` ordinals and the `BLOCK_INVALID = 22` sentinel from the
§3 constant table (p.7). `PARTITION_SUBSIZE` is transcribed verbatim;
every rectangular `bSize` column carries `BLOCK_INVALID` per the
spec p.401 note "The table will never get accessed for rectangular
block sizes". The typed accessor
`partition_subsize(partition, b_size) -> Option<usize>` folds the
sentinel into `None` so the upcoming §5.11.4 `decode_partition()`
body — which reads
`subSize = Partition_Subsize[ partition ][ bSize ]` and
`splitSize = Partition_Subsize[ PARTITION_SPLIT ][ bSize ]` side
by side, then dispatches `decode_block` / `decode_partition` per
the resolved size — never silently hands the `22` sentinel to its
recursive children. 16 new unit tests (359 -> 375) cover BLOCK_*
ordinal pinning (`BLOCK_4X4 = 0` through `BLOCK_INVALID = 22`);
`PARTITION_TYPES_TOTAL` (`10`) pin; table-shape pin; row-0
(PARTITION_NONE) identity on every square; row-1 / row-2 / row-3
halving / quartering identities; row-4..7 (`_A` / `_B`) row equality
with rows 1 / 2; row-8..9 (`_4`) quarter-splits + `BLOCK_128X128`
drop; `BLOCK_4X4`-only-resolves-for-`PARTITION_NONE` column-0
invariant; exhaustive rectangular-`bSize`-is-invalid coverage; every
resolved subSize in `0..BLOCK_SIZES`; out-of-range
(`partition >= 10`, `b_size >= BLOCK_SIZES`) guard; the §5.11.4
subdivision-shrinks-area invariant; and the §5.11.4 `subSize` +
`splitSize` pair well-formedness for every reachable HORZ_A /
HORZ_B parent block. The full `decode_partition` body remains the
next round's target; this round drops the last lookup it needs.
`decode_av1` and `encode_av1` still return `Error::NotImplemented`.

Round 151 lands the §5.11.4 **`decode_partition()` body** (av1-spec
p.61–62) — the recursive partition-tree walker that stitches every
partition prerequisite landed in rounds 137–150 into one
`PartitionWalker` type. The walker carries the §6.10.4
`MiSizes[r][c]` grid (filled at every leaf via the block's `bh4 *
bw4` footprint) and consults it for the §8.3.2 `partition_ctx`
derivation `above = AvailU && (Mi_Width_Log2[ MiSizes[r-1][c] ] <
bsl)` / `left = AvailL && (Mi_Height_Log2[ MiSizes[r][c-1] ] <
bsl)` on every recursive child (av1-spec p.362). It emits a
`Vec<DecodedBlockRecord>` of `(MiRow, MiCol, MiSize)` leaves in
§5.11.4 syntax order; the actual §5.11.5 `decode_block()` body
(coefficient / motion-vector / reconstruction) is the next round's
target. All four §5.11.4 edge-of-frame branches are honoured —
`r >= MiRows || c >= MiCols` early return, `bSize < BLOCK_8X8`
short-circuit to `PARTITION_NONE` with no symbol read,
`hasCols`-alone `split_or_horz`, `hasRows`-alone `split_or_vert`,
`!hasRows && !hasCols` fall-through to `PARTITION_SPLIT`. All ten
partition arms (`NONE` / `HORZ` / `VERT` / `SPLIT` / `HORZ_A` /
`HORZ_B` / `VERT_A` / `VERT_B` / `HORZ_4` / `VERT_4`) dispatch
the spec's literal `decode_block` / recursive `decode_partition`
calls with the appropriate `subSize` (`Partition_Subsize[
partition ][ bSize ]`) or `splitSize` (`Partition_Subsize[
PARTITION_SPLIT ][ bSize ]`). The §5.11.4 bottom-right edge clip
on the optional `HORZ_4` / `VERT_4` fourth leaf is applied
literally. New `TileGeometry` type carries the four §5.11.1
mi-unit tile bounds for the §5.11.51 `is_inside()` test. New
`Error::PartitionWalkOutOfRange` surfaces caller-bug preconditions
(`bSize` / `partition` / `bsl` / `Partition_Subsize` lookup out of
range). 19 new cdf-module tests (375 -> 394): TileGeometry
boundary cases; §5.11.4 early-return; the `bSize < BLOCK_8X8`
no-symbol-read short-circuit; §6.10.4 grid-fill; the `!hasRows &&
!hasCols` corner-case fallback; forced PARTITION_NONE / HORZ /
VERT / HORZ_4 / VERT_4 / HORZ_A / VERT_B / SPLIT at BLOCK_16X16
via rigged CDFs; forced-HORZ grid-fill; default-CDF W128 smoke
test; partition_ctx derivation at origin + after a wide-neighbour
leaf; construction overflow; `take_blocks` drain semantics.
`decode_av1` and `encode_av1` still return `Error::NotImplemented`
— the next round wires the §5.11.5 `decode_block()` body (block
coefficient decode + motion-vector decode + reconstruction)
behind the now-complete partition walker.

Round 152 lands the §5.11.11 **`read_skip()` syntax element**
(av1-spec p.65) as a new `PartitionWalker::decode_skip` method
plus a `Skips[r][c]` flag grid on the walker (parallel to the
§6.10.4 `MiSizes[]` grid landed in r151). Both spec branches are
honoured: the `SegIdPreSkip && seg_feature_active(SEG_LVL_SKIP)`
short-circuit (no symbol read, `skip = 1`) is taken when the
caller passes `seg_skip_active = true`; otherwise an `S()` symbol
is decoded against `TileSkipCdf[ctx]` with the §8.3.2 ctx `ctx =
AvailU * Skips[MiRow-1][MiCol] + AvailL * Skips[MiRow][MiCol-1]`
(av1-spec p.378). The decoded value is stamped over the block's
`bw4 * bh4` footprint of `Skips[]` per the §5.11.5 footer (clipped
at the frame's `MiRows` / `MiCols` extent). New
`PartitionWalker::skips()` accessor surfaces a row-major view of
the grid for downstream consumers. The combined precondition
`SegIdPreSkip && seg_feature_active(SEG_LVL_SKIP)` is computed
upstream by the frame parser — the walker stays segmentation-state
free. 11 new cdf-module tests (394 -> 405): seg short-circuit
returns 1 with no symbol read; else branch returns the rigged
symbol on a forced binary CDF; seg- + else-branch grid-stamp
invariant; origin ctx = 0; ctx-2 path through two prior
`Skips=1` neighbours; AvailL-false drops the left contribution;
non-zero tile origin clears AvailU / AvailL; right-edge `bw4`
clip; out-of-range `mi_row` / `mi_col` / `sub_size` ⇒
`PartitionWalkOutOfRange`; fresh-walker grid all-zero. The §5.11.5
`decode_block()` body itself (coefficient / motion-vector /
reconstruction) remains the next round's target. `decode_av1` and
`encode_av1` still return `Error::NotImplemented`.

Round 154 lands the §5.11.10 **`read_skip_mode()` syntax element**
(av1-spec p.67) as a new `PartitionWalker::decode_skip_mode` method
plus a `SkipModes[r][c]` flag grid on the walker (parallel to the
r152 `Skips[]` and the r151 `MiSizes[]` grids). The §5.11.10
short-circuit set (any-true ⇒ `skip_mode = 0`, no symbol read) is
honoured: `seg_feature_active( SEG_LVL_SKIP / REF_FRAME / GLOBALMV )`
collapsed into the caller-provided `seg_skip_mode_off`;
`!skip_mode_present` via the §5.9.21 frame-header scalar; and
`Block_Width[MiSize] < 8 || Block_Height[MiSize] < 8` derived
locally from `sub_size` via the §9.3 `block_width` / `block_height`
tables. Otherwise an `S()` symbol is decoded against
`TileSkipModeCdf[ctx]` with `ctx = AvailU *
SkipModes[MiRow-1][MiCol] + AvailL * SkipModes[MiRow][MiCol-1]`
(av1-spec p.378), routed through the existing `skip_mode_ctx`
helper. The §5.11.5 footer stamps the value over the block's
`bw4 * bh4` footprint of `SkipModes[]` (clipped at the frame's
`MiRows` / `MiCols` extent). New `PartitionWalker::skip_modes()`
accessor surfaces a row-major view. `skip_mode` is the
inter-frame compound-reference shortcut read in §5.11.18
`inter_frame_mode_info` before the rest of the inter mode decode
(intra-only frames never call this). 12 new cdf-module tests
(405 -> 417): seg short-circuit; `skip_mode_present` false
short-circuit; `Block_Width < 8` short-circuit (BLOCK_4X8);
`Block_Height < 8` short-circuit (BLOCK_8X4); else-branch S()
returning 0 / 1 on a forced binary CDF; footprint grid-stamp;
ctx-0 selection at the frame origin; ctx-1 single-neighbour and
ctx-2 both-neighbour paths; non-zero tile origin clearing AvailU
/ AvailL; right-edge `bw4` clip; out-of-range guards ⇒
`PartitionWalkOutOfRange`; fresh-walker grid all-zero. The
§5.11.5 `decode_block()` body itself (coefficient / motion-vector
/ reconstruction) remains the next round's target. `decode_av1`
and `encode_av1` still return `Error::NotImplemented`.

Round 174 lands the **§5.11.31 `assign_mv` + §5.11.32 `read_mv_component`
syntax tree** — wiring the per-block motion-vector decode into the
§5.11.23 inter cascade so the §5.11.31 / §5.11.32 leaves run
end-to-end. The r173 `Error::AssignMvUnsupported` stub is lifted; the
reader now short-circuits one step later at the new
`Error::MotionModeUnsupported` (§5.11.27 `read_motion_mode` is the
next-arc target — its body composes `use_obmc` / `motion_mode` S()
reads gated on §7.10.4 `find_warp_samples` + the §5.9.5
`force_integer_mv` / global-motion arms).

The §5.11.31 `assign_mv( isCompound )` body iterates over the active
reference lists (`i = 0..1 + isCompound`) and resolves each per-list
`Mv[ i ]` per the four-arm spec dispatch (av1-spec p.78):
`compMode = get_mode(i)` (the §5.11.30 helper, also lands this round);
`PredMv[ i ] = GlobalMvs[i]` when `compMode == GLOBALMV`, else
`PredMv[ i ] = RefStackMv[ pos ][ i ]` with `pos = (compMode ==
NEARESTMV) ? 0 : RefMvIdx` (forced to `0` when `compMode == NEWMV &&
NumMvFound <= 1`); finally `Mv[ i ] = PredMv[ i ] + diffMv` when
`compMode == NEWMV` (via §5.11.31 `read_mv`) or `Mv[ i ] = PredMv[ i ]`
otherwise (no MV bits read).

The §5.11.31 `read_mv( ref )` body composes one `mv_joint` S()
against `TileMvJointCdf[ MvCtx ]` (the 4-way joint code
`MV_JOINT_ZERO` / `MV_JOINT_HNZVZ` / `MV_JOINT_HZVNZ` /
`MV_JOINT_HNZVNZ`), then conditionally invokes
`read_mv_component( 0 )` and/or `read_mv_component( 1 )` per the
spec body's gating (`HZVNZ || HNZVNZ` for `diffMv[0]`,
`HNZVZ || HNZVNZ` for `diffMv[1]`), and finishes with
`Mv[ ref ][ c ] = PredMv[ ref ][ c ] + diffMv[ c ]`. `MvCtx` is
derived per the §5.11.31 `mv_ctx` helper (`MV_INTRABC_CONTEXT = 1`
when `use_intrabc == 1`, `0` otherwise); the inter caller always
passes `use_intrabc = false`.

The §5.11.32 `read_mv_component( comp )` body composes the full
two-branch sign-magnitude tree (av1-spec p.81-82): one `mv_sign` S()
against `TileMvSignCdf[ MvCtx ][ comp ]`, one `mv_class` S() against
`TileMvClassCdf[ MvCtx ][ comp ]` (one of `MV_CLASS_0..=MV_CLASS_10`),
then either the **MV_CLASS_0 ladder** (`mv_class0_bit` S() +
`mv_class0_fr` S() OR `= 3` when `force_integer_mv == 1` + `mv_class0_hp`
S() OR `= 1` when `allow_high_precision_mv == 0`, then
`mag = ((mv_class0_bit << 3) | (mv_class0_fr << 1) | mv_class0_hp) + 1`)
or the **MV_CLASS_K ladder** for `K >= 1` (per-bit `mv_bit` S() loop
`d |= mv_bit_i << i` for `i = 0..K`, then `mv_fr` / `mv_hp` reads with
the same `force_integer_mv` / `allow_high_precision_mv` gating, and
`mag = (CLASS0_SIZE << (mv_class + 2)) + ((d << 3) | (mv_fr << 1) |
mv_hp) + 1`). The signed return is `mv_sign ? -mag : mag`; the
§6.10.25 `is_mv_valid` conformance bound (`|Mv[ i ][ comp ]| < (1 <<
14)`) is the caller's responsibility.

The `Mvs[r][c][list][comp]` grid (introduced in r172 as a §7.10.2
neighbour-walk feed) is now stamped over the bh4 * bw4 footprint
after every `assign_mv` call. The pre-fill `0` value is replaced by
the decoded per-list Mv (cast `i32` → `i16`; the §6.10.25 bound keeps
the value in i16 range). Subsequent blocks' §7.10.2 spatial scans
therefore observe the decoded motion vectors instead of the
fresh-walker pre-fill.

`DecodedInterBlockModeInfo` gains a `mv: [[i32; 2]; 2]` field
carrying the §5.11.31 `Mv[ 0..2 ]` array. Slot `mv[0]` is always
written; `mv[1]` is only meaningful on compound blocks
(`is_compound == true`) — on single-pred blocks slot 1 stays at the
§5.11.5 pre-fill `[0, 0]`. The aggregate remains observable only on
the `Ok` path; the r174 dispatcher always returns `Err`
(`MotionModeUnsupported`) since the §5.11.27 `read_motion_mode` body
is not yet wired.

New `get_mode(y_mode, ref_list)` helper (§5.11.30) folds a YMode +
reference-list index into the per-list `compMode` used by §5.11.31
(one of `MODE_NEWMV` / `MODE_NEARESTMV` / `MODE_NEARMV` /
`MODE_GLOBALMV`). The single-pred branch (`ref_list == 0` AND
`YMode < NEAREST_NEARESTMV`) is the identity; the compound branch
applies the eight-row §6.10.22 → §5.11.30 mapping table.

Six new §3 / §6.10.27 / §6.10.28 named constants land:
`MV_JOINT_ZERO`, `MV_JOINT_HNZVZ`, `MV_JOINT_HZVNZ`,
`MV_JOINT_HNZVNZ` (the four §6.10.27 mv_joint ordinals) plus
`MV_CLASS_0` (the §5.11.32 small-magnitude class). The remaining
`MV_CLASS_K` for `K = 1..=10` reuses the literal `K` value (the spec
only names class 0 explicitly in the syntax body).

11 new unit tests cover: `get_mode` single-pred identity on
`ref_list = 0` (4 modes); `get_mode` compound table (8 modes ×
2 ref_lists); `assign_mv` skip_mode arm with NEAREST_NEARESTMV
producing zero MV bits read; `assign_mv` seg_globalmv arm using
identity GlobalMvs; `read_mv` NEWMV with `mv_joint = MV_JOINT_ZERO`
yielding zero diff (no per-component reads); `read_mv_component`
direct exercise of the MV_CLASS_0 ladder (three sub-cases: all-sym-0
with allow_hp ⇒ mag=1; all-sym-0 with hp=false fall-through ⇒ mag=2;
all-sym-0 with sign=1 ⇒ mag=-2); `force_integer_mv` short-circuiting
the `mv_class0_fr` read (forces fr=3 ⇒ mag=8); cascade structural
smoke (NEWMV cascade with default CDFs producing in-range Mv per
§6.10.25); and an `assign_mv` defensive guard (rejecting
`use_intrabc = true` in the inter arm). Test count: 661 → 672 (+11).
`decode_av1` / `encode_av1` still return `Error::NotImplemented`.

Round 175 lands the **§5.11.27 `read_motion_mode` reader** — with its
§7.10.3 `has_overlappable_candidates` and §7.10.4 `find_warp_samples`
helpers — wiring the per-block motion-mode decode into the §5.11.23
inter cascade. The r174 `Error::MotionModeUnsupported` stub is lifted;
the §5.11.18 dispatcher now reaches its `Ok(_)` arm and surfaces the
next-arc gap (`read_interintra_mode` / `read_compound_type` /
`read_interpolation_filter`) as `InterBlockModeInfoUnsupported`.

The §5.11.27 body composes the five short-circuit arms in spec order
(av1-spec p.79): `skip_mode == 1` ⇒ SIMPLE; `!is_motion_mode_switchable`
⇒ SIMPLE; `Min(Block_Width, Block_Height) < 8` ⇒ SIMPLE;
`!force_integer_mv && YMode ∈ {GLOBALMV, GLOBAL_GLOBALMV} &&
GmType[RefFrame[0]] > TRANSLATION` ⇒ SIMPLE; `isCompound ||
RefFrame[1] == INTRA_FRAME || !has_overlappable_candidates()` ⇒ SIMPLE.
Otherwise the body invokes §7.10.4 `find_warp_samples()` then either
the **`use_obmc` S()** arm (`force_integer_mv || NumSamples == 0 ||
!allow_warped_motion || is_scaled(RefFrame[0])`) returning
SIMPLE / OBMC, or the **`motion_mode` S()** arm returning
SIMPLE / OBMC / WARPED_CAUSAL directly.

The §7.10.3 `has_overlappable_candidates()` helper walks the 8x8-
granular above + left ref-frame grid (`(x4 | 1)` / `(y4 | 1)` probes
into `RefFrames[][][0]`) returning true on the first non-INTRA_FRAME
slot. The §7.10.4 `find_warp_samples()` walk runs the spec's
above-neighbour + left-neighbour + top-left + top-right `add_sample`
invocations, threading the §7.10.4.2 in-tile / written-cell /
matching-ref / single-list gates and the magnitude-validity check
(`Clip3(16, 112, Max(Block_Width, Block_Height))`-bounded
`|mvDiffRow| + |mvDiffCol|`). The §7.10.4.1 "first large MV kept if no
small one matches" special case is honoured (`NumSamples = 1` when all
scanned samples failed the magnitude bound). The §3
`LEAST_SQUARES_SAMPLES_MAX = 8` cap on `NumSamplesScanned` short-
circuits the walk per the §7.10.4.2 first-line guard. The `CandList`
itself is not surfaced (the §7.11.4 `warp_estimation()` consumer is a
downstream arc; the data is recoverable from the `Mvs[]` / `MiSizes[]`
walker grids on demand).

The default `Default_Use_Obmc_Cdf[BLOCK_SIZES][3]` table (§9.4) is
transcribed verbatim, surfaced through the new
`TileCdfContext::use_obmc` field + `use_obmc_cdf(mi_size)` selector.
The §9.4 note (indices `0..=2` and `16..=17` of the first dimension
unreachable) is honoured by the §5.11.27 prologue's `Min(BW, BH) >= 8`
gate. Three new §6.10.26 named constants land: `MOTION_MODE_SIMPLE`,
`MOTION_MODE_OBMC`, `MOTION_MODE_WARPED_CAUSAL`. Two new §3 constants
land: `LEAST_SQUARES_SAMPLES_MAX = 8`, `REF_SCALE_SHIFT = 14`.

Three new §5.11.27 frame-header scalars thread through
`decode_inter_block_mode_info` (and the §5.11.18 dispatcher):
`is_motion_mode_switchable` (§5.9.2 bit), `allow_warped_motion`
(§5.9.2 bit), `is_scaled_per_ref: [bool; 7]` (caller-precomputed per-
reference `is_scaled` lookup; `is_scaled_per_ref[refFrame - LAST_FRAME]`
mirrors the §5.11.27 `is_scaled(RefFrame[0])` predicate without
threading the frame-size / RefInfo arrays into the walker).

`DecodedInterBlockModeInfo` gains two new fields: `motion_mode: u8`
(carrying one of the three `MOTION_MODE_*` ordinals) and
`num_warp_samples: u32` (the §7.10.4 `NumSamples` snapshot at the
§5.11.27 site, in `0..=LEAST_SQUARES_SAMPLES_MAX = 0..=8`). Observable
through the §5.11.23 dispatcher's `Ok(_)` arm — now reached on every
conformant inter block — even though the §5.11.18 dispatcher still
surfaces `InterBlockModeInfoUnsupported` for the pending
post-`read_motion_mode` cascade.

19 new unit tests cover: `DEFAULT_USE_OBMC_CDF` row-shape sanity
(BLOCK_SIZES rows × 3 cumulative frequencies + counter); the
`use_obmc_cdf(mi_size)` selector against the §9.4 source for every
in-range index plus None for out-of-range; the §6.10.26 ordinal
identities; `has_overlappable_candidates` returning false on a fresh
walker and true after stamping an above (resp. left) inter
neighbour; `find_warp_samples` returning 0 on a fresh walker, ≥1 on a
same-ref + matching-MV above neighbour, and 0 on a mismatched-ref
neighbour; the five §5.11.27 SIMPLE short-circuit arms (skip_mode,
!is_motion_mode_switchable, small block, GLOBALMV + non-TRANSLATION
gm_type, compound, no overlappable candidates) each verified to read
zero S() bits; the `use_obmc` arm consuming the §8.3 `update_cdf` of
the use_obmc row while leaving the motion_mode row untouched; the
`motion_mode` arm consuming the `motion_mode` row while leaving
use_obmc untouched; the `is_scaled = true` routing forcing the
`use_obmc` arm; and an end-to-end §5.11.23 dispatcher exercise
showing `skip_mode = 1` surfacing
`Ok(DecodedInterBlockModeInfo { motion_mode: SIMPLE, .. })`. Test
count: 672 → 691 (+19). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 178 lands the **§5.11.x `read_interpolation_filter` reader** —
the LAST leaf of the §5.11.23 inter cascade. The §5.11.23 dispatcher
now runs the entire `inter_block_mode_info()` body to completion;
the §5.11.18 dispatcher continues to surface
`InterBlockModeInfoUnsupported` from its `Ok(_)` arm pending the
next-arc §5.11.34 `residual()` lift and the follow-on refactor that
lifts the `DecodedInterBlockModeInfo` aggregate into the
`DecodedInterFrameModeInfo` return path.

The §5.11.x body composes two paths: the `interpolation_filter ==
SWITCHABLE` arm runs the per-direction `for dir = 0..(enable_dual_filter
? 2 : 1)` loop with the inner `if ( needs_interp_filter( ) )
interp_filter[ dir ] S() against TileInterpFilterCdf[ ctx ] else
interp_filter[ dir ] = EIGHTTAP` branch, and on the
`!enable_dual_filter` post-loop mirror sets `interp_filter[ 1 ] =
interp_filter[ 0 ]`. The `else` arm forces both slots to the
frame-header value (`interpolation_filter`) with no per-block bits.

`needs_interp_filter( )` (av1-spec p.75) is also lifted: it closes
the gate on `skip_mode || motion_mode == LOCALWARP`, plus the two
large-block GLOBALMV / GLOBAL_GLOBALMV paths consult the
caller-supplied `GmType[ RefFrame[ . ] ] == TRANSLATION` predicate;
the default arm returns `1`.

The §8.3.2 `interp_filter` ctx walk consumes the §8.3.2 paragraph
verbatim through `interp_filter_ctx` (already in place since round
22): a folded `((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4` ctx
seed, then the two neighbour-mismatch / match-and-equal / one-mismatch
arms. The match test consults the neighbour cell's
`RefFrames[..][..][0..2]` against the current block's `RefFrame[0]`;
non-matching neighbours collapse to the `aboveType = leftType = 3`
sentinel (`INTERP_FILTER_NONE`).

The walker grows an `InterpFilters[r][c][dir]` grid (two slots per
cell, pre-fill `EIGHTTAP = 0`), stamped over the bh4 * bw4 footprint
on every inter block decode so the next block's ctx walk observes
the values. The `interp_filters()` accessor surfaces it.

New §6.8.9 named ordinals land: `EIGHTTAP = 0`, `EIGHTTAP_SMOOTH = 1`,
`EIGHTTAP_SHARP = 2`, `BILINEAR = 3`, `SWITCHABLE = 4`. New
`InterpolationFilterReadout` aggregate carries `interp_filter: [u8; 2]`
and `read_from_bitstream: [bool; 2]` (the per-dir "did this slot
fire an S()?" flag). `DecodedInterBlockModeInfo` gains an
`interp_filter: InterpolationFilterReadout` field.
`decode_inter_block_mode_info` and `decode_inter_frame_mode_info` gain
two new caller-supplied scalars: `interpolation_filter: u8` (§5.9.10
frame-header value) and `enable_dual_filter: bool` (§5.5.2
sequence-header bit).

10 new unit tests cover: the §6.8.9 ordinal alignment (and the
`INTERP_FILTERS` boundary against `BILINEAR`); the non-switchable
arm (both slots forced to the frame-header value, no bits read);
the four `needs_interp_filter == 0` short-circuits (`skip_mode`,
`motion_mode == LOCALWARP`, large GLOBALMV + non-TRANSLATION
gm_type) all collapsing to EIGHTTAP with no bits; the
`SWITCHABLE && needs_interp_filter` single-dir path (one S() ⇒ both
slots get the symbol, `read_from_bitstream = [true, false]`); the
dual-dir path (two S()s ⇒ `read_from_bitstream = [true, true]`); the
three caller-bug guards (out-of-range `sub_size` /
`interpolation_filter` / `ref_frame[0]` all surface
`PartitionWalkOutOfRange`); and an end-to-end §5.11.23 cascade test
through the dispatcher witnessing the `InterpolationFilterReadout`
on the SWITCHABLE + skip_mode (needs_interp_filter == 0) path with
the walker's `interp_filters` grid stamped EIGHTTAP over the
BLOCK_16X16 footprint. Test count: 721 → 731 (+10). `decode_av1` /
`encode_av1` still return `Error::NotImplemented`.

Round 197 lands the **§7.17 loop restoration — driver + Wiener arm
+ §7.17.5 coefficient reconstruction + §7.17.6 stripe-aware source
fetch**, sitting on top of the post-§7.15 CDEF samples per av1-spec
p.327-335. The new `loop_restoration` module exposes:

* **§7.17 [`loop_restoration_frame`]** — top-level driver: walks the
  frame in `MI_SIZE` steps over `(y, x) ∈ [0, FrameHeight) ×
  [0, UpscaledWidth)`, copies `UpscaledCdefFrame ↦ LrFrame` up-front,
  short-circuits on `UsesLr == 0`, and otherwise dispatches each
  `(plane, row, col)` whose `FrameRestorationType[plane] !=
  RESTORE_NONE` to [`loop_restore_block`].
* **§7.17.1 [`loop_restore_block`]** — per-block driver: computes the
  `(unitRow, unitCol, x, y, w, h)` geometry, the stripe extents
  `(StripeStartY, StripeEndY)` from `stripeNum = (lumaY + 8) / 64`,
  and the plane extents `(PlaneEndX, PlaneEndY)`; dispatches by
  `rType = LrType[plane][unitRow][unitCol]` to the Wiener / self-guided
  / no-op arm.
* **§7.17.4 [`wiener_filter`]** — two-pass separable 7-tap convolution:
  horizontal pass builds `intermediate[h+6][w]` via `Round2(s,
  InterRound0)` + `Clip3(-offset, limit - offset, v)`; vertical pass
  writes `LrFrame[plane][y + r][x + c] = Clip1(Round2(s, InterRound1))`
  with `(InterRound0, InterRound1)` from the §7.11.3.2 `isCompound = 0`
  schedule (`(3, 11)` for 8/10-bit, `(5, 9)` for 12-bit).
* **§7.17.5 [`wiener_coefficients`]** — `filter[3] = 128`, `filter[i] =
  filter[6 - i] = coeff[i]`, `filter[3] -= 2 * Σ coeff[i]`. Reconstructs
  the symmetric 7-tap filter from 3 transmitted coefficients; the
  symmetric DC gain `Σ filter == 128` (asserted in the test suite).
* **§7.17.6 [`get_source_sample`]** — clamps `(x, y)` to `[0,
  PlaneEndX] × [0, PlaneEndY]`, then routes the fetch by the stripe
  boundary: `y < StripeStartY` ⇒ `UpscaledCurrFrame` with
  `y = max(StripeStartY - 2, y)` (two-line reach); `y > StripeEndY` ⇒
  `UpscaledCurrFrame` with `y = min(StripeEndY + 2, y)`; otherwise ⇒
  `UpscaledCdefFrame`.
* **§7.17.2 [`self_guided_filter`]** — projection-arm driver: runs
  the §7.17.3 box filter twice with `(r0, eps0)` and `(r1, eps1)`
  from `SGR_PARAMS[set]`, then combines the two filtered outputs
  with weights `(w0, w1)` from `LrSgrXqd` (and derived `w2 = (1 <<
  SGRPROJ_PRJ_BITS) - w0 - w1`) against `u = UpscaledCdefFrame <<
  SGRPROJ_RST_BITS`, rounding via `SGRPROJ_RST_BITS +
  SGRPROJ_PRJ_BITS` and clipping to the sample range. The `r == 0`
  branch substitutes `u` for `flt` per av1-spec p.330 lines
  18263-18269.
* **§7.17.3 box filter** (internal `box_filter` helper) — builds the
  `A[]/B[]` integral-style tables across `[-1, h + 1] × [-1, w + 1]`
  via the `(2r + 1)²` sum kernel, applies the `Sgr_Params`-driven
  `m`-table nonlinearity (`s = ((1 << SGRPROJ_MTABLE_BITS) + n2e/2)
  / n2e`; piecewise `a2`), and convolves with the §7.17.3 weighted
  3×3 footprint (pass-0 odd-row-only; pass-1 centre-plus-cross 4 /
  corner 3).
* **[`derive_block_geometry`]** — pure-function geometry derivation that
  bundles all §7.17.1 named locals into a single [`LrBlockGeometry`]
  return value; consumed by both `loop_restore_block` and the
  self-guided arm.

The tables [`WIENER_TAPS_MID`] / [`WIENER_TAPS_MIN`] /
[`WIENER_TAPS_MAX`] / [`WIENER_TAPS_K`] (`{3,-7,15} / {-5,-23,-17} /
{10,8,46} / {1,2,3}`), [`SGRPROJ_XQD_MID`] / [`SGRPROJ_XQD_MIN`] /
[`SGRPROJ_XQD_MAX`] (`{-32,31} / {-96,-32} / {31,95}`), and the full
16-entry [`SGR_PARAMS`] table (av1-spec p.332 lines 18395-18400) are
transcribed verbatim. The §3 / §5.9.20 constants [`LR_FILTER_BITS`]
(= 7), [`WIENER_COEFFS`] (= 3), [`SGRPROJ_PARAMS_BITS`] (= 4),
[`SGRPROJ_PRJ_SUBEXP_K`] (= 4), [`SGRPROJ_PRJ_BITS`] (= 7),
[`SGRPROJ_RST_BITS`] (= 4), [`SGRPROJ_MTABLE_BITS`] (= 20),
[`SGRPROJ_RECIP_BITS`] (= 12), and [`SGRPROJ_SGR_BITS`] (= 8) are
exported. [`count_units_in_frame`] implements the §5.9.20 helper
`Max((frameSize + (unitSize >> 1)) / unitSize, 1)`.

A small standalone [`LoopRestorationFrameContext`] bundles the §5.5 /
§5.9.5 / §5.9.20 frame-header fields plus four `&dyn Fn(...)` closures
the caller hooks into the §5.11.58 decode state (`lr_type`,
`lr_wiener`, `lr_sgr_set`, `lr_sgr_xqd`), letting the driver run
against synthetic predicates. The §5.9.20 [`LrParams`] (r10) and
§5.11.58 per-unit reads will plug straight into these closures.

Round 203 wires the **§7.11.3.1 OBMC motion-mode arm** of the
[`predict_inter`] driver (av1-spec p.258 line 14414 + §7.11.3.9 +
§7.11.3.10) — the r194 driver short-circuit at `motion_mode ==
MOTION_MODE_OBMC` is retired in favour of an end-to-end dispatch that
performs the §7.11.3.9 `overlapped_motion_compensation` mi-grid
neighbour walk and runs the §7.11.3.10 overlap blend against each
qualifying above-row / left-column neighbour's translational MC. The
§7.11.3.9-10 pixel-blend leaves landed in r193 ([`overlap_blending`] /
[`overlap_neighbour_predict_blend`] / [`get_obmc_mask`] / the five
`OBMC_MASK_*` tables); r203 is the driver-side mi-grid walk that
consumes them. **All four motion modes (SIMPLE / WARPED_CAUSAL /
OBMC + compound) are now driver-side complete.**

New public surface:

* **[`ObmcParams<'a, 'n>`]** — caller-resolved mi-grid context:
  * `mi_row` / `mi_col` / `mi_cols` / `mi_rows` — §5.9.5 mi-grid
    coordinates of the current block + frame dimensions. Used for the
    `x4 < Min(MiCols, MiCol + w4)` (above) / `y4 < Min(MiRows, MiRow
    + h4)` (left) loop-bound per av1-spec p.275 lines 15309 / 15332.
  * `mi_width_log2` / `mi_height_log2` — `Mi_{Width,Height}_Log2[
    MiSize ]` driving `nLimit = Min(4, …)` (av1-spec p.275 lines
    15308 / 15331).
  * `avail_u` / `avail_l` — §5.11.18 availability gates for the two
    passes (av1-spec p.275 lines 15301 / 15325).
  * `plane_residual_size_ge_block_8x8` — the additional above-pass
    gate `get_plane_residual_size(MiSize, plane) >= BLOCK_8X8` (av1-
    spec p.275 line 15302). The left-pass is unconditional on this
    per the av1-spec p.275 line 15283 carve-out ("For small blocks,
    only the left neighbor will be used").
  * `above_neighbours` / `left_neighbours` — ordered slices of the
    per-axis qualifying neighbours (caller pre-filters `RefFrames[
    candRow][candCol][0] > INTRA_FRAME` per av1-spec p.275 lines
    15314 / 15337).

* **[`ObmcNeighbour<'a>`]** — per-neighbour bundle: `bundle:
  PredictInterRef` carries the neighbour's `Mvs[..][0]` + ref-frame
  plane resolution; `step4: u8` is the `Clip3(2, 16,
  Num_4x4_Blocks_{Wide,High}[ candSz ])` advance per spec lines
  15313 / 15336. The driver iterates the slice in order, advancing
  `x4 += step4` (above) or `y4 += step4` (left), capped at `nLimit`.

The §7.11.3.9 `predict_overlap` per-candidate steps 1-8 are wired
inside the module-private `obmc_walk_axis` helper:

1. Steps 1-2 — `mv` and `refIdx` are pre-resolved into the neighbour
   `PredictInterRef`.
2. Steps 3-4 — `predX = (x4 * 4) >> subX` / `predY = (y4 * 4) >>
   subY` derive the plane-coordinate top-left of the overlap region.
3. Steps 5-6 — `predict_inter_per_ref` runs `motion_vector_scaling`
   + `block_inter_prediction` against `predW × predH` (per-pass per
   spec lines 15316-15317 / 15339-15340).
4. Step 7 — `clip1_single_ref` on the i32 prediction.
5. Step 8 — `overlap_neighbour_predict_blend` blends the neighbour
   prediction into the per-block `pred_out` buffer (treated as a
   `(h, w)` window of `CurrFrame[plane]` with origin at the block's
   plane coords `(x, y)`).

[`predict_inter`]'s signature gains a trailing `obmc:
Option<&ObmcParams<'_, '_>>` argument before `pred_out`. A caller
that passes `obmc = None` runs the prior translational / compound /
warp paths unchanged (the OBMC post-step is gated on `motion_mode
== MOTION_MODE_OBMC` only). A caller that sets `motion_mode ==
MOTION_MODE_OBMC` without supplying `ObmcParams` returns
`Error::PartitionWalkOutOfRange` (caller bug). The
`Error::PredictInterObmcUnsupported` variant is removed (no longer
reachable). With r203 every motion-mode arm of [`predict_inter`] is
driver-side complete.

Test count: 1033 → 1041 (+8 in lib). New `inter_pred::tests`:

* **`r203_predict_inter_obmc_no_neighbours_matches_simple`** — pins
  the empty-list base case: `AvailU == AvailL == true` with empty
  `above_neighbours` and `left_neighbours` lists yields the SIMPLE
  translational baseline byte-for-byte (no blend invocations).
* **`r203_predict_inter_obmc_unavailable_neighbours_noops`** — pins
  the §5.11.18 outer-gate behaviour: `avail_u = avail_l = false`
  short-circuits both passes even when neighbour slices are
  populated.
* **`r203_predict_inter_obmc_identical_above_neighbour_idempotent`** —
  exercises the above-pass with one identical neighbour (same ref +
  same MV); the §7.11.3.10 identity `Round2(64 * curr, 6) = curr`
  pins that the blend wiring is correct without depending on a
  specific reference-sample trace.
* **`r203_predict_inter_obmc_identical_left_neighbour_idempotent`** —
  same identity property for the left-pass (which uses `mask[j]`
  rather than `mask[i]` per av1-spec p.278 lines 15442-15446).
* **`r203_predict_inter_obmc_above_modifies_top_half_only`** — pins
  the above-pass extent: with `predH = h >> 1 = 4` for an 8×8 block
  and a non-zero neighbour MV, rows 4..8 must equal the SIMPLE
  baseline byte-for-byte while at least one cell in rows 0..4 must
  differ.
* **`r203_predict_inter_obmc_above_gated_by_plane_residual_size`** —
  pins the `get_plane_residual_size(MiSize, plane) >= BLOCK_8X8`
  above-pass-only gate (av1-spec p.275 line 15302).
* **`r203_predict_inter_obmc_n_limit_caps_above_walk`** — pins the
  `nLimit = Min(4, Mi_Width_Log2[ MiSize ])` cap: a 4-wide block
  with `mi_width_log2 = 0` yields `nLimit = 0` and the walk skips
  the neighbour even when supplied.
* **`r203_predict_inter_obmc_multi_above_neighbours_walk_correctly`**
  — exercises the multi-neighbour `x4 += step4` advance with two
  ordered `step4 = 2` neighbours on a 16-wide block (`nLimit = 2`).

Followup candidates (next arcs): §9.5.3 Quantizer matrix; §7.10.2.5
/ §7.10.2.6 temporal-MV scan; §5.11.34 transform-coefficient residual
parser; encoder.

The prior r202 round notes (preserved verbatim for traceability):

Round 202 wires the **§7.11.3.1 WARPED_CAUSAL motion-mode arm** of the
[`predict_inter`] driver (av1-spec p.257-258 lines 14308-14370) — the
r194 driver stub at `motion_mode == MOTION_MODE_WARPED_CAUSAL` is
retired in favour of an end-to-end WARP dispatch that performs the
step-2/3/6/7 `useWarp` derivation per `refList` and, when `useWarp ∈
{1, 2}`, drives the §7.11.3.5 [`block_warp`] kernel through the
step-12 per-8×8 sub-block loop (`i8 ∈ 0..((h-1) >> 3)`, `j8 ∈
0..((w-1) >> 3)`) in lieu of the §7.11.3.4 translational kernel. The
four warp sub-bodies landed in r192 ([`block_warp`] / [`setup_shear`]
/ [`resolve_divisor`] / [`warp_estimation`]); r202 is the driver-side
wiring against them.

New public surface:

* **[`WarpDriverParams`]** — struct plumbing the spec inputs the
  step-2/3/6/7 derivation consumes:
  * `y_mode[refList]` — §5.11.x `YMode` per-ref; the step-7 ◦ 4 gate
    requires `YMode ∈ {GLOBALMV, GLOBAL_GLOBALMV}`.
  * `gm_type[refList]` — §5.9.x `GmType[refFrame]` per-ref; the
    step-7 ◦ 4 gate requires `gm_type > TRANSLATION`.
  * `gm_params[refList][0..6]` — §5.9.x `gm_params[refFrame]` per-ref;
    the `useWarp == 2` arm passes this matrix to [`block_warp`].
  * `local_warp_params[0..6]` — §7.11.3.8 [`warp_estimation`] output;
    the `useWarp == 1` arm passes this matrix to [`block_warp`].
  * `local_valid` — §7.11.3.8 `LocalValid` bit; re-validated against
    [`setup_shear`] internally per step-3 (av1-spec p.257 line 14311).
  * `ref_is_scaled[refList]` — §5.11.27 `is_scaled(refFrame)` per-ref;
    the step-7 ◦ 4 gate requires `false`.
  * `force_integer_mv` — §5.9.x frame flag; step-7 ◦ 2 forces
    `useWarp = 0` when `true`.

The step-7 decision tree is exposed through the module-private
`derive_use_warp` helper covering all five spec arms (`w/h < 8` ⇒ 0;
`force_integer_mv` ⇒ 0; LOCALWARP + LocalValid ⇒ 1; GLOBAL_GLOBALMV +
`GmType > TRANSLATION` + `!is_scaled` + `globalValid` ⇒ 2; else 0).
The driver also performs the step-3 `plane == 0`
`setup_shear(LocalWarpParams)` re-validation internally — a
`LocalWarpParams` with a degenerate `[2] == 0` diagonal demotes the
run to translational fallback even when the caller passes
`local_valid = true`.

[`predict_inter`]'s signature gains a trailing
`warp: Option<&WarpDriverParams>` argument before `pred_out`; a
caller that supplies `warp = None` runs the prior translational /
compound path unchanged. A caller that sets `motion_mode ==
WARPED_CAUSAL` without supplying `WarpDriverParams` returns
`Error::PartitionWalkOutOfRange` (caller bug). The
`Error::PredictInterWarpUnsupported` variant is removed (no longer
reachable). After r202, only `motion_mode == OBMC` still
short-circuits to a dedicated error.

Test count: 1025 → 1033 (+8 in lib). New `inter_pred::tests`:

* **`r202_predict_inter_localwarp_routes_to_block_warp`** — pins the
  step-7 ◦ 3 (`useWarp = 1`) routing: `motion_mode == WARPED_CAUSAL`
  + `local_valid = true` + `WARP_IDENTITY` matrix triggers the
  per-8×8 [`block_warp`] dispatch, and the driver output equals a
  hand-composed [`block_warp`] + [`clip1_single_ref`] call.
* **`r202_predict_inter_global_warp_routes_to_block_warp`** — same
  for step-7 ◦ 4 (`useWarp = 2`): SIMPLE motion mode with
  `YMode == GLOBAL_GLOBALMV` and `gm_type == ROTZOOM` triggers the
  GLOBAL_WARP path with `gm_params[refFrame]`.
* **`r202_predict_inter_use_warp_small_block_falls_back_to_translational`**
  — step-7 ◦ 1: `w < 8` forces translational fallback even with
  `WARPED_CAUSAL` motion mode and a valid `local_warp_params`.
  Output equals the SIMPLE-mode baseline.
* **`r202_predict_inter_force_integer_mv_disables_warp`** — step-7 ◦
  2: `force_integer_mv = true` forces translational fallback.
* **`r202_predict_inter_local_invalid_falls_back_to_translational`** —
  step-7 ◦ 3 negative arm: `local_valid = false` flows through to
  step-7 ◦ 5 (no LOCALWARP, no GLOBAL via `gm_type == IDENTITY`).
* **`r202_predict_inter_global_blocked_by_is_scaled`** — step-7 ◦ 4
  negative arm: flipping `ref_is_scaled[0]` to `true` blocks the
  GLOBAL gate; output equals the SIMPLE baseline.
* **`r202_predict_inter_step3_shear_revalidates_localwarp`** — pins
  the step-3 internal re-validation: a `LocalWarpParams` with
  `[2] == 0` (which `setup_shear` rejects via
  `resolve_divisor(0) ⇒ None`) demotes `effective_local_valid` to
  `false` and the dispatch falls back to translational.
* **`r202_derive_use_warp_truth_table`** — `derive_use_warp` truth
  table covering all five spec cases plus three negative-arm cases of
  step-7 ◦ 4 (gm_type == TRANSLATION; globalValid == false; y_mode
  not in {GLOBALMV, GLOBAL_GLOBALMV}).

Followup candidates (next arcs): wiring the **OBMC** arm
(§7.11.3.9-10 leaves landed in r193; needs the
`overlapped_motion_compensation` mi-grid neighbour walk); §9.5.3
Quantizer matrix; §7.10.2.5 / §7.10.2.6 temporal-MV scan.

The prior r201 round notes (preserved verbatim for traceability):

Round 201 wires the **§7.11.3.1 step-14 compound arm** of the
[`predict_inter`] driver (av1-spec p.258 lines 14381-14412) — the
r194 driver stub at `is_compound == true` is retired in favour of an
end-to-end dispatch that iterates `refList ∈ {0, 1}`, runs the
§7.11.3.3 + §7.11.3.4 pipeline twice (one prediction per ref), and
combines via one of the five `compound_type` mechanisms. The five
§7.11.3.11-15 sample-generation leaves landed in r191; r201 is the
driver-side wiring against them.

New public surface:

* **[`CompoundParams<'a>`]** — enum carrying the per-`compound_type`
  side data the driver dispatches on:
  * `Average` — `Clip1(Round2(preds[0] + preds[1],
    1 + InterPostRound))` per pixel (av1-spec p.258 line 14405).
    No side data.
  * `Distance(DistanceWeights)` — `Clip1(Round2(FwdWeight * preds[0]
    + BckWeight * preds[1], 4 + InterPostRound))` per pixel (av1-spec
    p.258 line 14408); the weights come from
    [`distance_weights`] (§7.11.3.15).
  * `Wedge { mask, mask_stride }` — wedge-blended compound mode
    (av1-spec p.258 line 14386 + §7.11.3.11 + §7.11.3.14). Mask is
    the [`wedge_mask`] output on the luma grid; reused across planes
    via [`mask_blend`]'s `(sub_x, sub_y)` downsampling.
  * `Diffwtd { mask, mask_stride }` — difference-weighted compound
    mode (§7.11.3.12 + §7.11.3.14). Mask is the
    [`difference_weight_mask`] output.
  * `Intra { mask, mask_stride }` — inter-intra-variant blend
    (§7.11.3.13 + §7.11.3.14). Mask is the
    [`intra_mode_variant_mask`] output.

`predict_inter` now takes a `Option<CompoundParams<'_>>` argument:
`Some(_)` on the compound arm, `None` (or rejected) on the single-ref
arm. Per av1-spec p.258 lines 14386-14393, the §7.11.3.11 /
§7.11.3.12 / §7.11.3.13 mask is computed once at `plane == 0` and
reused for chroma planes — the driver does not cache it across
calls; the walker / caller is responsible for supplying the same mask
buffer on subsequent `plane == 1` / `plane == 2` invocations.

Internal refactor: the per-`refList` body (steps 5-13) is factored
into a new internal `predict_inter_per_ref` helper so the compound arm
can run it twice without duplicating the `motion_vector_scaling` +
`block_inter_prediction` + fallible alloc plumbing. The compound
dispatch lives in a new internal `predict_inter_compound_blend`
that pattern-matches on [`CompoundParams`] and routes to the inline
`COMPOUND_AVERAGE` body, [`compound_distance_blend`], or
[`mask_blend`].

Caller-bug guards:

* `is_compound == true` ⇒ `refs.len() >= 2` and `compound.is_some()`.
* `is_compound == false` ⇒ `compound.is_none()` (a spurious
  `Some(_)` is rejected).

The r194 [`Error::PredictInterCompoundUnsupported`] variant is
removed (no longer reachable). The WARP and OBMC stub arms still
surface their dedicated [`Error::PredictInterWarpUnsupported`] /
[`Error::PredictInterObmcUnsupported`] — those are the next-arc
targets (r202 / r203).

Test count: 1019 → 1025 (+6 in lib). New tests:

* **`r201_predict_inter_compound_average_matches_hand_blend`** —
  drives `Some(CompoundParams::Average)` on a `ref0 == ref1` setup
  and verifies the per-pixel output equals a hand-derived
  `Clip1(Round2(p+p, 1+InterPostRound))` against the standalone
  `block_inter_prediction` leaf.
* **`r201_predict_inter_compound_distance_matches_hand_blend`** —
  drives `Some(CompoundParams::Distance(DistanceWeights {
  fwd_weight: 9, bck_weight: 7 }))` and verifies the driver matches
  [`compound_distance_blend`] applied externally.
* **`r201_predict_inter_compound_wedge_routes_to_mask_blend`** —
  feeds a §7.11.3.11 wedge mask through
  `Some(CompoundParams::Wedge { mask, .. })` and verifies the driver
  output matches the external [`mask_blend`] composition.
* **`r201_predict_inter_compound_diffwtd_routes_to_mask_blend`** —
  same shape against a §7.11.3.12 difference-weight mask.
* **`r201_predict_inter_compound_intra_routes_to_mask_blend`** —
  same shape against a §7.11.3.13 `II_SMOOTH_PRED` mask.
* **`r201_predict_inter_compound_rejects_caller_bugs`** —
  `(is_compound, refs.len(), compound)` mismatch matrix:
  `is_compound + refs.len() == 1`, `is_compound + compound == None`,
  `!is_compound + compound == Some`, each returning
  `PartitionWalkOutOfRange` without entering the rounding-variables /
  scaling / convolution pipeline.

Followup candidates (next arcs): wiring the WARP arm into the driver
(§7.11.3.5-8 kernels are landed in r192; the step-2/3/6/7 `useWarp`
derivation + step-12 `(i8, j8)` sub-block loop is the remaining
piece); wiring the OBMC arm (§7.11.3.9-10 leaves landed in r193;
needs the `overlapped_motion_compensation` mi-grid neighbour walk);
§9.5.3 Quantizer matrix; §7.10.2.5 / §7.10.2.6 temporal-MV scan.

The prior r200 round notes (preserved verbatim for traceability):

Round 200 lands the **§7.16 superres upscaling** process — the
horizontal post-CDEF / pre-LR pass that takes the encoded `FrameWidth`-
wide downscaled frame and produces an `UpscaledWidth`-wide frame per
av1-spec p.325-326. With this arc the per-frame post-processing
pipeline (§7.14 LF → §7.15 CDEF → §7.16 superres → §7.17 LR →
§7.18.3 film grain) is now complete end-to-end at the
sample-filtering level. Coverage lands as a new `superres` module:

* **[`upscale_frame`]** — top-level per-plane driver. Iterates
  `plane ∈ [0, NumPlanes)`; on `use_superres == 0` (or
  `FrameWidth == UpscaledWidth`) returns a verbatim copy of the input
  planes per av1-spec p.325 line 17968. Otherwise dispatches each
  plane to [`upscale_plane`].
* **[`upscale_plane`]** — per-plane body. Derives `(subX, subY,
  downscaledPlaneW, upscaledPlaneW, planeH, stepX, err,
  initialSubpelX, miW, minX, maxX)` per av1-spec p.325 lines
  17979-17999, then walks `(y, x) ∈ [0, planeH) × [0,
  upscaledPlaneW)` and writes `outputFrame[plane][y][x] =
  Clip1(Round2(Σ px * Upscale_Filter[srcXSubpel][k], FILTER_BITS))`
  per av1-spec p.325 lines 18000-18013.
* **[`upscale_sample`]** — per-output-pixel inner loop. Performs the
  8-tap polyphase convolution against `UPSCALE_FILTER[srcXSubpel]`
  with the §3 `Clip3(minX, maxX, srcXPx + (k - SUPERRES_FILTER_OFFSET))`
  left/right edge replication.
* **[`UPSCALE_FILTER`]** — the
  `Upscale_Filter[SUPERRES_FILTER_SHIFTS][SUPERRES_FILTER_TAPS]`
  table verbatim from av1-spec p.326 lines 18027-18060 — 64 sub-pixel
  phases × 8 taps, each tap a signed `i32` scaled to the
  `FILTER_BITS = 7` precision (Σ taps per row = 128, pinned by
  `upscale_filter_table_sums_to_128`).
* **§3 named constants surfaced** —
  [`SUPERRES_FILTER_BITS`] / [`SUPERRES_FILTER_SHIFTS`] /
  [`SUPERRES_FILTER_TAPS`] / [`SUPERRES_FILTER_OFFSET`] /
  [`SUPERRES_SCALE_BITS`] / [`SUPERRES_SCALE_MASK`] /
  [`SUPERRES_EXTRA_BITS`] / [`FILTER_BITS`][superres::FILTER_BITS] per
  av1-spec p.32 lines 1395-1412. The §5.9.8 `SUPERRES_NUM` /
  `SUPERRES_DENOM_MIN` / `SUPERRES_DENOM_BITS` continue to live in
  [`crate::frame_header`] where the bitstream-parse path needs them.

The driver takes a small `SuperresFrameContext` bundling
`(use_superres, FrameWidth, UpscaledWidth, FrameHeight, MiCols,
NumPlanes, BitDepth, subsampling_x, subsampling_y)` — straight from
the §5.5.2 / §5.9.5 / §5.9.8 parsed frame header — plus input and
output `PlaneBuffer<'_>` slices. The §7.17 driver's
`UpscaledCdefFrame` input is now genuinely the upscaled version of
`CdefFrame`; the prior placeholder in r197/r198 was a verbatim copy.

Test count: 1070 → 1082 (+12 in lib). New `superres::tests`:

* **`upscale_filter_table_sums_to_128`** — every row of
  `Upscale_Filter` sums to `128 = 1 << FILTER_BITS`, so the
  `Round2(., FILTER_BITS)` reduction preserves average brightness
  on a constant input.
* **`upscale_filter_table_geometry`** — 64 rows × 8 taps; phase 0 is
  the identity kernel `[0,0,0,128,0,0,0,0]`.
* **`upscale_constants_match_spec`** — pins the named-constant table
  relations (`SUPERRES_FILTER_SHIFTS == 1 << SUPERRES_FILTER_BITS`,
  `SUPERRES_EXTRA_BITS == SUPERRES_SCALE_BITS - SUPERRES_FILTER_BITS`,
  `SUPERRES_SCALE_MASK == (1 << SUPERRES_SCALE_BITS) - 1`).
* **`use_superres_false_copies_planes_verbatim`** — av1-spec p.325
  line 17968 short-circuit.
* **`equal_widths_copies_planes_verbatim`** — even with
  `use_superres == true`, `FrameWidth == UpscaledWidth` skips the
  filter (so the FILTER_OFFSET drift doesn't bleed through).
* **`flat_input_yields_flat_output`** — convolving a constant input
  with a unity-sum filter returns the same constant exactly (the
  rounding term `+ 64` lands cleanly when `Σ taps * v = 128 * v`).
* **`upscale_clips_to_bit_depth`** — 8-bit envelope holds on a spiky
  255/0 input through every `(y, x)` of the inner loop.
* **`upscaled_smaller_than_downscaled_is_rejected`** — av1-spec p.326
  line 18063 conformance: `upscaledPlaneW > downscaledPlaneW` when
  `use_superres == 1`.
* **`upscale_sample_phase_zero_is_passthrough`** — phase 0 of the
  helper is the identity kernel for any srcXPx well inside the row.
* **`upscale_sample_edge_replicates_via_clip3`** — Clip3 edge taps
  pull from the first/last sample.
* **`upscale_plane_geometry_mismatch_is_rejected`** — caller-supplied
  buffer extent disagreeing with the ctx-derived plane extent returns
  `SuperresError::PlaneShapeMismatch` rather than panicking.
* **`chroma_subsampling_halves_widths`** — for 4:2:0 the chroma plane
  derivation halves both downscaledPlaneW and upscaledPlaneW per
  `Round2(., subX)` and the flat-input invariant still holds across
  all three planes.

This completes the per-frame post-processing pipeline. Next-arc
candidates: inter driver arming (compound / WARP / OBMC sample paths
are landed but not yet dispatched from `predict_inter`); §9.5.3
Quantizer matrices; or §7.10.2.5 / §7.10.2.6 temporal-MV scan.

The prior r199 round notes (preserved verbatim for traceability):

Round 199 lands the **§7.18.3 film grain synthesis** post-processing
layer — the final stage of the AV1 decode pipeline, applied after the
§7.17 loop restoration pass. Coverage spans the §7.18.3.1 driver,
§7.18.3.2 LFSR random number process, §7.18.3.3 generate-grain (white
noise + AR filter, including the chroma-from-luma cross term),
§7.18.3.4 scaling-lookup initialisation (256-entry per-plane lookup
tables built by piecewise linear interpolation of the §5.9.30
`(point_*_value, point_*_scaling)` pairs), and §7.18.3.5 add-noise
synthesis (per-stripe `noiseStripe`/`noiseImage` construction with
`overlap_flag` X/Y blending, then the per-sample blend via
`scale_lut` with the `cb_mult` / `cb_luma_mult` / `cb_offset` /
`cr_mult` / `cr_luma_mult` / `cr_offset` chroma-from-luma merge and
the `clip_to_restricted_range` ↦ `(minValue, maxLuma, maxChroma)`
clamp). The verbatim 2048-entry `Gaussian_Sequence` from av1-spec
p.411-414 lives in a sibling `film_grain_tables` module. The
top-level [`film_grain_synthesis`] entry takes the parsed
[`FilmGrainParams`] (§5.9.30, already populated by the r10 / r158
bit-reader), frame-level `(bit_depth, num_planes, sub_x, sub_y,
matrix_coefficients)`, and mutable [`crate::loop_filter::PlaneBuffer`]s
for the post-LR `OutY` / `OutU` / `OutV` samples; sub-stage helpers
[`generate_grain`], [`scaling_lookup_init`], [`add_noise_synthesis`],
[`scale_lut`], and the [`RandomRegister`] LFSR are exposed for direct
testing.

Test count: 1058 → 1070 (+12 in lib). New `film_grain::tests`:

* **`lfsr_first_three_draws_for_seed_one`** — pins the §7.18.3.2 LFSR
  tap polynomial against the seed=1 trajectory (`r=1 ⇒ 0x8000 ⇒
  0x4000`; 11-bit results `0x400, 0x200`).
* **`lfsr_eight_bit_draw_matches_top_byte`** — `get_random_number(8)`
  returns the post-shift register's high byte (seed=0xff00 ⇒ 0xff,
  state=0xff80).
* **`scaling_lookup_zero_num_points_is_zero_table`** —
  `numPoints == 0` branch zero-fills all three planes.
* **`scaling_lookup_two_point_ramp_interpolates`** — `(0,0) →
  (255,255)` two-point ramp yields ≈x for all `x ∈ [0, 256)` after
  the §7.18.3.4 `(65536 + deltaX/2) / deltaX` reciprocal.
* **`scaling_lookup_left_plateau_holds`** — entries `[0,
  point_y_value[0])` carry `point_y_scaling[0]`; the right edge
  carries `point_y_scaling[numPoints-1]`.
* **`generate_grain_zero_when_num_y_points_zero`** —
  `num_y_points == 0` ⇒ `LumaGrain` is all-zero; with no chroma
  points and no `chroma_scaling_from_luma`, `CbGrain` / `CrGrain` are
  also all-zero. Chroma dims `(44, 38)` for 4:2:0.
* **`generate_grain_luma_first_sample_matches_round2_of_first_gauss_draw`**
  — first written `LumaGrain[0][0]` equals
  `Round2(Gaussian_Sequence[LFSR.next(11)], 12 - BitDepth +
  grain_scale_shift)`; pins the §7.18.3.3 white-noise pass.
* **`film_grain_synthesis_no_apply_is_noop`** — `apply_grain == 0`
  short-circuits before §7.18.3.3; planes unchanged.
* **`film_grain_synthesis_zero_y_points_leaves_luma_unchanged`** —
  `apply_grain == 1` but `num_y_points == 0` ⇒ ScalingLut is zero;
  the §7.18.3.5 luma blend is gated, output unchanged; chroma blend
  also gated by `num_cb_points == num_cr_points == 0 &&
  !chroma_scaling_from_luma`.
* **`scale_lut_8bit_is_direct_indexing`** — `BitDepth == 8` path is a
  direct `ScalingLut[plane][index]` lookup.
* **`scale_lut_10bit_interpolates`** — `BitDepth == 10` path bridges
  two adjacent entries via `start + Round2((end - start) * rem,
  shift)` (`table[2]=100, table[3]=200, index=9` ⇒ `125`).
* **`film_grain_synthesis_luma_ramp_mutates_samples`** — full
  pipeline smoke test against a 64×64 constant-128 monochrome plane
  with `(point_y_value, point_y_scaling) = (0, 64) → (255, 64)`,
  `grain_seed = 7`; asserts the blend mutates at least some samples
  and all samples stay in `[0, 256)`.

This completes the §7.18 output / film grain stage; §7.16 superres
upscaling is now the lone remaining post-processing layer.

The prior r198 round notes (preserved verbatim for traceability):

Round 198 lands the **§7.17.2 / §7.17.3 self-guided projection arm**
on top of the r197 driver scaffold. [`self_guided_filter`] reads the
per-unit `set = LrSgrSet[plane][unitRow][unitCol]` selector + the two
weights `(w0, w1) = LrSgrXqd[plane][unitRow][unitCol][0..2]`, derives
`w2 = (1 << SGRPROJ_PRJ_BITS) - w0 - w1` per av1-spec p.330 line
18255, invokes the new `box_filter` helper twice (with `(r0, eps0) =
(SGR_PARAMS[set][0], SGR_PARAMS[set][1])` and `(r1, eps1) =
(SGR_PARAMS[set][2], SGR_PARAMS[set][3])`), then combines the two
filtered outputs against `u = UpscaledCdefFrame[plane][y + i][x + j]
<< SGRPROJ_RST_BITS` per the §7.17.2 projection
`v = w1 * u + w0 * (flt0 or u) + w2 * (flt1 or u)`,
`Clip1(Round2(v, SGRPROJ_RST_BITS + SGRPROJ_PRJ_BITS))`. The `r == 0`
branch (sets 10..=13 for `r0`; sets 14..=15 for `r1`) substitutes `u`
for the corresponding `flt` per av1-spec p.330 lines 18263-18269.

`box_filter` implements §7.17.3 verbatim: builds the `A[]/B[]`
tables across the `[-1, h + 1] × [-1, w + 1]` extended neighbourhood
via the `(2r + 1)²` box-sum kernel, normalises by the `n2e = n² · eps`
divisor (`s = ((1 << SGRPROJ_MTABLE_BITS) + n2e/2) / n2e`), applies
the piecewise `a2 ∈ {1, ((z << SGR_BITS) + z/2) / (z + 1), 256}`
nonlinearity, computes `b2 = ((1 << SGR_BITS) - a2) * b * oneOverN`
with `oneOverN = ((1 << SGRPROJ_RECIP_BITS) + n/2) / n`, and rounds
`B = Round2(b2, SGRPROJ_RECIP_BITS)`. The second pass convolves
`(A, B)` with the §7.17.3 weighted 3×3 footprint: pass-0 uses the
odd-row `(5, 6, 5, 0, 0, 0)` weights with `shift = 4` on odd
output rows / `shift = 5` on even rows; pass-1 uses the
centre-plus-cross `4` / corner `3` weighting with uniform `shift =
5`. The final `F = Round2(a * u + b, SGR_BITS + shift - RST_BITS)`
matches av1-spec p.332 line 18383. The 10/12-bit profile widens the
`bd_shift_{a, b} = 2 * (BitDepth - 8) / (BitDepth - 8)` Round2
shifts; intermediates that overflow `i32` (the `b2` product reaches
`2³⁹` at 12-bit) are held in `i64` via a new [`round2_i64`] helper.

Test count: 1053 → 1058 (+5 net in lib; +6 SGR tests minus the
retired `restore_sgrproj_passes_cdef_through` stub-pass-through
assertion). New SGR tests (all in `loop_restoration::tests`):

* **`self_guided_uniform_zero_weights_recovers_source`** — `(w0, w1)
  = (0, 0)` ⇒ `w2 = 128`; on a uniform 222-sample plane the
  projection `128 * flt1 >> 11` recovers the source within ±1 LSB
  across the deep interior `[4, 12)²` patch (boundary samples reach
  through the box-sum's `(2r + 1)²` neighbourhood into the clipped
  edge).
* **`self_guided_w2_zero_recovers_source_directly`** — `(w0, w1) =
  (0, 128)` ⇒ `w2 = 0`; projection collapses to `128 * u`, and the
  `Round2(., 11)` recovers `cdef` exactly without going through the
  box-filter nonlinearity.
* **`self_guided_set_10_r0_zero_uses_u_for_flt0`** —
  `SGR_PARAMS[10] = [0, 0, 1, 5]` triggers the `r == 0` early-exit;
  the projection substitutes `u` for `flt0`. With `(w0, w1) = (128,
  0)` ⇒ `w2 = 0`, the projection becomes `128 * u`, recovering
  `cdef`.
* **`self_guided_set_14_r1_zero_uses_u_for_flt1`** — symmetric
  variant: `SGR_PARAMS[14] = [2, 30, 0, 0]` triggers the `r1 == 0`
  early-exit and substitutes `u` for `flt1`. `(w0, w1) = (0, 128)`
  recovers `cdef` exactly.
* **`box_filter_r0_returns_zeroed_buffer`** — direct test of the
  helper's `r == 0` exit: returns a zero-filled `h * w` buffer (the
  caller's projection substitutes `u` for `flt` in that branch, so
  the zeros are never read on the §7.17.2 path).
* **`self_guided_clips_to_sample_range`** — pathological weights
  `(w0, w1) = (-96, 95)` (`Sgrproj_Xqd_Min[0]` / `Sgrproj_Xqd_Max[1]`)
  with `w2 = 129`; the output sample range stays inside `[0, 255]`
  for an 8-bit decode, exercising the `Clip1(s)` clamp at av1-spec
  p.330 line 18271.

The §7.17 dispatcher's `SgrProj` arm now routes to
[`self_guided_filter`] instead of leaving the pre-copied
`UpscaledCdefFrame` in place; the §7.17.4 / §7.17.5 / §7.17.6 Wiener
path is unchanged from r197.

The prior r197 round notes (preserved verbatim for traceability):

Test count: 1040 → 1053 (+13 in lib). r197 tests (all in
`loop_restoration::tests`):

* **`wiener_taps_tables_match_spec`** — verbatim `[Mid|Min|Max|K]`
  + `Sgrproj_Xqd_[Mid|Min|Max]` constant tables.
* **`sgr_params_table_matches_spec`** — spot-checks `SGR_PARAMS` rows
  0 / 10 / 14 and asserts the `1 << SGRPROJ_PARAMS_BITS` length.
* **`wiener_coefficients_unit_dc_gain`** — exercises four `coeff`
  inputs (zero / mid / min / max), asserts `Σ filter == 128`, the
  `filter[i] == filter[6 - i]` symmetry, and the `coeff[i] == filter[i]`
  transmitted-tap landing.
* **`wiener_coefficients_zero_centre_is_128`** — zero taps ⇒
  `[0,0,0,128,0,0,0]`.
* **`count_units_in_frame_matches_spec`** — three corner cases
  (`(64, 256)`, `(64, 100)`, `(64, 0)`, `(256, 240)`).
* **`loop_restoration_frame_no_op_when_uses_lr_false`** — `UsesLr = 0`
  ⇒ `LrFrame = UpscaledCdefFrame` and the closures are never consulted.
* **`loop_restoration_frame_restore_none_keeps_cdef_copy`** —
  `FrameRestorationType[0] = RESTORE_NONE` ⇒ `LrFrame` holds the CDEF
  copy unchanged even with `UsesLr = 1`.
* **`wiener_filter_identity_filter_recovers_source`** — `coeff =
  [0, 0, 0]` ⇒ identity 7-tap filter scaled by 128; a 64×64 uniform
  input plane round-trips through the two-pass convolution to the same
  uniform value (the `>> InterRound0` and `>> InterRound1` shifts
  cancel: `128 << 7 / 2^(3+11) = 1`).
* **`get_source_sample_inside_stripe_reads_cdef`** — `y ∈
  [StripeStartY, StripeEndY]` ⇒ reads `UpscaledCdefFrame`.
* **`get_source_sample_above_stripe_reads_curr_clamped`** — `y <
  StripeStartY` ⇒ reads `UpscaledCurrFrame` with `y = max(StripeStartY
  - 2, y)`.
* **`get_source_sample_below_stripe_reads_curr`** — `y > StripeEndY`
  ⇒ reads `UpscaledCurrFrame` with `y = min(StripeEndY + 2, y)`.
* **`derive_block_geometry_luma_top_left`** — `(row, col) = (0, 0)`
  on a 64×64 plane: `unit_row = unit_col = 0`, `(x, y, w, h) = (0, 0,
  4, 4)`, `(plane_end_x, plane_end_y) = (63, 63)`, `(stripe_start_y,
  stripe_end_y) = (-8, 55)`.
* **`restore_sgrproj_passes_cdef_through`** — `RESTORE_SGRPROJ` (no-op
  for this arc) ⇒ `LrFrame` holds the pre-copied CDEF content.

The §7.15 CDEF de-ringing pass landed in r196; r197 adds the
Wiener arm of the loop restoration pass that runs immediately after;
r198 completes the §7.17.2 / §7.17.3 self-guided projection arm,
finishing the `reconstruct ↦ deblock ↦ CDEF ↦ LR` sample-flow
(modulo §7.16 superres upscaling which decouples `UpscaledCurrFrame`
from `CurrFrame`). §7.16 upscaling and the §7.20 film-grain
post-pass remain on the roadmap for subsequent arcs. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 196 lands the **§7.15 CDEF (Constrained Directional Enhancement
Filter) — driver + direction search + primary + secondary tap
filter**, sitting on top of the post-§7.14 deblocked `CurrFrame[plane]`
samples per av1-spec p.318-324. The new `cdef` module exposes:

* **§7.15.1 [`cdef_frame`] / [`cdef_block`]** — top-level per-8×8
  walk over the luma `(MiRows, MiCols)` grid with the spec's `step4 =
  Num_4x4_Blocks_Wide[BLOCK_8X8] = 2` stride; per-block driver that
  copies `CurrFrame[plane]` into `CdefFrame[plane]` (the per-plane
  8×8 region with subsampling), short-circuits when `idx == -1` or
  when the four 4×4 cells' `Skips[]` all hold, and otherwise dispatches
  the §7.15.2 direction search + the §7.15.3 primary / secondary
  filter per plane. The luma path applies the `(priStr * (4 +
  varStr) + 8) >> 4` strength rescale per av1-spec p.319 line 17683;
  chroma uses the [`CDEF_UV_DIR`] lookup to translate `yDir` ↦ `dir`
  per the §7.15.1 line 17696 schedule.
* **§7.15.2 [`cdef_direction`]** — 8-direction match against the
  `partial[8][15]` projection sums, scored via the [`DIV_TABLE`]
  constant. The 4 cardinal directions (0 / 2 / 4 / 6) use the spec's
  paired-symmetry accumulator; the 4 oblique directions (1 / 3 / 5
  / 7) use the slab + tail formulation. Returns `(yDir, var)` with
  `var = (bestCost - cost[(yDir + 4) & 7]) >> 10`.
* **§7.15.3 [`cdef_filter_block`]** — per-plane filter that walks
  the `h × w` plane region (`h = 8 >> subY`, `w = 8 >> subX`), and
  for each output sample accumulates the [`CDEF_PRI_TAPS`][`(priStr
  >> coeffShift) & 1`] tap pair along `dir` and the [`CDEF_SEC_TAPS`]
  tap pair along `(dir ± 2) & 7`, both filtered through the
  [`constrain`] damping primitive. The output is
  `Clip3(min, max, x + ((8 + sum - (sum < 0)) >> 4))` per av1-spec
  p.324 line 17892, where `min` / `max` track the actually-consulted
  neighbour samples only (the `is_inside_filter_region` gate per
  §5.11.52 / av1-spec p.103 drops off-region taps).
* **[`constrain`]** — `sign * Clip3(0, |diff|, threshold - (|diff|
  >> dampingAdj))` with `dampingAdj = Max(0, damping -
  FloorLog2(threshold))` and a `threshold == 0 ⇒ 0` short-circuit.

The tables [`CDEF_PRI_TAPS`], [`CDEF_SEC_TAPS`], [`CDEF_DIRECTIONS`],
[`CDEF_SEC_TAPS`], [`DIV_TABLE`], and [`CDEF_UV_DIR`] are transcribed
verbatim from av1-spec p.320-324.

A small standalone [`CdefFrameContext`] bundles the §5.5 / §5.9.5 /
§5.9.19 frame-header fields plus two `&dyn Fn(...)` closures the
caller hooks into the §5.11 decode state (`cdef_idx`, `skip`),
letting the driver run against synthetic predicates. The §5.9.19
`CdefParams` (r10) and §5.11.56 `cdef_idx[]` walker (r157) are now
consumed end-to-end: the parser-produced strength schedule + per-leaf
index drive the sample-modification pass.

Test count: 1026 → 1040 (+14 in lib). New tests (all in
`cdef::tests`):

* **`constrain_returns_zero_for_zero_threshold`** — `threshold == 0`
  ⇒ 0 per av1-spec p.324 line 17920.
* **`constrain_signs_match_diff_sign`** — modest `|diff| ≤
  threshold` returns `±diff` verbatim (sign-preserving identity).
* **`constrain_large_diff_clamped_by_threshold_minus_shifted`** —
  large `|diff|` shifted past `threshold` collapses to 0.
* **`cdef_directions_table_matches_spec`** — sanity-check three rows
  of [`CDEF_DIRECTIONS`].
* **`div_table_matches_spec`** — verbatim [`DIV_TABLE`] match.
* **`cdef_uv_dir_lookup_identity_for_444`** — `subX = subY = 0`
  passes `yDir` through unchanged.
* **`direction_search_uniform_block_returns_zero_var`** — uniform
  100-luma block ⇒ `(yDir = 0, var = 0)`.
* **`direction_search_horizontal_stripes_pick_direction_2`** —
  row-varying input picks direction 2 (taps along x, edge runs
  horizontally).
* **`direction_search_vertical_stripes_pick_direction_6`** —
  column-varying input picks direction 6 (taps along y, edge runs
  vertically).
* **`cdef_block_with_idx_neg1_only_copies`** — `idx == -1` copies
  the 8×8 source region into the destination and returns without
  invoking §7.15.2 / §7.15.3.
* **`cdef_block_with_skip_only_copies`** — every 4×4 `Skip[]` set
  ⇒ filter path skipped; only the copy survives.
* **`cdef_filter_uniform_block_is_idempotent`** — uniform input ⇒
  `p - x = 0` ⇒ `sum = 0` ⇒ `delta = 0` ⇒ identity.
* **`cdef_frame_driver_walks_every_8x8_anchor`** — 32×32 frame ⇒
  4×4 anchors, all `idx == -1`, every cell of the destination
  matches the source after the driver returns.
* **`cdef_frame_chroma_copy_respects_subsampling`** — 16×16 luma +
  8×8 chroma at 4:2:0, the driver copies each plane at its
  sub-sampled stride.

The §7.14 deblock pass landed in r195; r196 adds the de-ringing pass
that runs immediately after, completing the
`reconstruct ↦ deblock ↦ CDEF ↦ (upscale) ↦ (LR)` sample-flow. §7.16
superres upscaling and §7.17 loop restoration remain on the roadmap
for subsequent arcs, as does the §7.20 film-grain post-pass.
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 195 lands the **§7.14 loop filter (deblocking) — driver +
edge-strength + 4/8/14-tap filter bodies**, sitting on top of the
post-§7.12.3 `CurrFrame[plane]` reconstructed samples per av1-spec
p.307-318. The new `loop_filter` module exposes:

* **§7.14.1 [`loop_filter_frame`]** — top-level per-plane × per-pass
  (vertical-then-horizontal) × per-row × per-col walk with the
  `rowStep`/`colStep = (plane == 0) ? 1 : (1 << subY/X)` chroma
  stride. Mutates the per-plane [`PlaneBuffer`]s in place.
* **§7.14.2 [`loop_filter_edge`]** — per-edge driver: derives
  `(dx, dy)`, `(x, y)`, `(xP, yP)`, `(prevRow, prevCol)`, `onScreen`,
  `isBlockEdge`, `isTxEdge`, and `applyFilter` per the spec's exact
  gate ordering; then runs §7.14.3 + §7.14.4 + §7.14.6 against the
  `MI_SIZE`-pixel span of the boundary.
* **§7.14.3 [`filter_size`]** — `Min(plane==0 ? 16 : 8, Min(Tx_*[txSz],
  Tx_*[prevTxSz]))`, with `pass == 0` consulting `Tx_Width[]` and
  `pass == 1` consulting `Tx_Height[]`.
* **§7.14.4 [`adaptive_filter_strength`]** — derives the
  [`FilterStrength`] bundle `(lvl, limit, blimit, thresh)`: `lvl`
  from §7.14.5; `limit` via the `loop_filter_sharpness`-driven
  shift + `Clip3(1, 9 - sharp, lvl >> shift)` / `Max(1, lvl >>
  shift)` branch; `blimit = 2 * (lvl + 2) + limit`; `thresh = lvl
  >> 4`.
* **§7.14.5 [`adaptive_filter_strength_selection`]** — the per-segment
  `SEG_LVL_ALT_LF_Y_V + i` adjustment + the
  `loop_filter_ref_deltas[INTRA_FRAME]` (intra path) / `+ ref + mode`
  (inter path) shifted-by-`lvlSeg >> 5` adjustment, both clipped to
  `[0, MAX_LOOP_FILTER]`.
* **§7.14.6.1 [`sample_filtering`]** — per-pixel dispatch from
  [`filter_mask`] outputs: `filterMask == 0` ⇒ no-op; otherwise
  `filterSize == 4 || !flatMask` ⇒ [`narrow_filter`], else
  `filterSize == 8 || !flatMask2` ⇒ [`wide_filter`] with
  `log2Size = 3`, else `wide_filter` with `log2Size = 4`.
* **§7.14.6.2 [`filter_mask`]** — computes the four masks
  [`FilterMaskOutput`] (`hevMask`, `filterMask`, `flatMask`,
  `flatMask2`) from samples at `(x ± k*dx, y ± k*dy)` for `k ∈
  {0..6}` plus the seventh `p6 / q6` slot. `filterLen` widens with
  `(plane != 0)` (6) / `filterSize == 8` (8) / else 16.
* **§7.14.6.3 [`narrow_filter`]** — the 4-pixel-window edge filter:
  subtract `0x80 << (BitDepth - 8)` from inputs, build `filter`
  from `hevMask ? ps1 - qs1 : 0` plus `3 * (qs0 - ps0)`,
  `filter4_clamp` to `[-(1 << (BitDepth-1)), (1 << (BitDepth-1)) -
  1]`, then `filter1 = (filter + 4) >> 3` / `filter2 = (filter + 3)
  >> 3` ⇒ `oq0 = qs0 - filter1 + half`, `op0 = ps0 + filter2 +
  half`. The `!hevMask` arm additionally writes `oq1` / `op1` via a
  `Round2(filter1, 1)` half-strength tap.
* **§7.14.6.4 [`wide_filter`]** — the low-pass 6/8/14-tap filter for
  detected flat regions: `n ∈ {2, 3, 6}` (chroma-8 / luma-8 / 16
  branches), `n2 ∈ {0, 1}`, with the `Clip3(-(n+1), n, i + j)`
  replicated-edge index. Inner accumulator runs in `i64` so the
  14-tap luma path doesn't overflow at 12-bit content.

A small standalone [`LoopFilterFrameContext`] bundles the §5.9.11 /
§5.9.18 / §5.5 / §5.9.5 frame-header fields the driver needs plus
seven `&dyn Fn(...)` closures the caller hooks into the §5.11 decode
state — `is_intra`, `skip`, `ref_frame`, `mode`, `segment_id`,
`delta_lf`, `seg_feature_active` / `seg_feature_data`, `lf_tx_size`,
`mi_size`. Letting the driver run against synthetic predicates keeps
the §7.14 walk testable without the full §5.11.x walker wired up.

Test count: 1014 → 1026 (+12 in lib). New tests (all in
`loop_filter::tests`):

* **`strength_returns_zero_when_loop_filter_level_zero`** — the
  §7.14.5 short-circuit when `loop_filter_level[i] == 0` and no
  deltas are enabled returns `lvl == 0`, suppressing the edge.
* **`strength_basic_table_lvl_32_sharp_0`** — sanity:
  `lvl=32, sharp=0` ⇒ `limit=32, blimit=100, thresh=2` per the §7.14.4
  formulas.
* **`strength_intra_ref_delta_lifts_lvl`** — with
  `loop_filter_delta_enabled` + `ref_deltas[INTRA_FRAME] = 5`,
  `lvl_seg = 10 + (5 << (10 >> 5))` = 15 (n_shift = 0).
* **`strength_clipped_at_max_loop_filter`** — `lvl=60` + ref_delta `10`
  shifted by `n_shift = 60 >> 5 = 1` (⇒ 80) clips to 63.
* **`filter_size_luma_caps_at_16_chroma_at_8`** — `TX_64X64`
  64-wide ⇒ luma 16 / chroma 8; `TX_4X4` 4-wide ⇒ 4.
* **`narrow_filter_idempotent_on_uniform_input`** — uniform 128 ⇒
  every sample unchanged (filter terms collapse to ±0).
* **`narrow_filter_softens_sharp_step_edge`** — left-100 / right-156
  step ⇒ `p0` rises above 100, `q0` falls below 156, and the move
  is symmetric about the mid-point (128).
* **`wide_filter_log2_3_luma_preserves_uniform_input`** — uniform 200
  ⇒ each tap-summed result equals 200 (DC-gain-1 design).
* **`wide_filter_log2_4_luma_preserves_uniform_input`** — uniform 64
  ⇒ unchanged.
* **`filter_mask_flat_region_allows_filter`** — uniform 100 with
  generous thresholds ⇒ `filter_mask && flat_mask && !hev_mask`.
* **`filter_mask_steep_edge_blocks_filter`** — sharp 50/200 step with
  tight thresholds ⇒ `!filter_mask` (filter vetoed).
* **`driver_iterates_correct_edge_count_for_2x2_mi_grid`** — 2×2
  mi-grid × 2 passes × on-screen gate ⇒ exactly 4 edges visit the
  per-block predicate.

The §5.9.11 `loop_filter_params()` parse landed in r5 + was
stream-wired in r9; r195 turns the parsed parameters into the actual
post-reconstruction sample modifications they govern. The driver does
not yet wire against [`PartitionWalker`] directly — the next arc will
plumb the closure surface into the walker's `y_modes` / `is_inters`
/ `segment_ids` / `skips` accessors so a full decode can apply the
deblock pass without bespoke predicates. §7.15 CDEF and §7.17 LR
remain on the roadmap for subsequent arcs.

Round 194 lands the **§7.11.3.1 [`predict_inter`] driver skeleton** —
the entry point the §5.11.33 dispatcher invokes per inter task. The
driver composes the four r189-r193 leaves into one `w × h` prediction
per the av1-spec p.257-258 process; on this round only the
**single-reference translational SIMPLE arm** is wired end-to-end.
Compound (`is_compound == true`) was a stub on r194 surfacing
`Error::PredictInterCompoundUnsupported`; **r201 retires that variant
and wires the step-14 compound arm** (see below). WARP (`motion_mode
== MOTION_MODE_WARPED_CAUSAL`) and OBMC (`motion_mode ==
MOTION_MODE_OBMC`) still short-circuit to a dedicated
[`Error::PredictInterWarpUnsupported`] /
[`Error::PredictInterObmcUnsupported`], so callers can narrow against
the exact unsupported arm.

The driver body (single-ref translational arm):

1. **Step 1** — [`rounding_variables(bit_depth, is_compound)`] →
   `(InterRound0, InterRound1, InterPostRound)`.
2. **Step 4** — `refList = 0` (single-ref).
3. **Steps 5,8,9** — caller-supplied [`PredictInterRef`] bundles
   (`ref_plane`, `ref_stride`, `ref_upscaled_width`, `ref_width`,
   `ref_height`, `mv`); the §7.11.3.1 step-5/step-8 spec lookups are
   external to the driver since the walker owns the `RefFrames[..]` /
   `Mvs[..]` grids.
4. **Step 10** — [`motion_vector_scaling`] →
   `(startX, startY, stepX, stepY)`.
5. **Step 13** — [`block_inter_prediction`] runs the §7.11.3.4 8-tap
   translational kernel against `(startX, startY, stepX, stepY)`.
6. **Final-clip (single-ref)** — `CurrFrame[plane][y + i][x + j] =
   Clip1(preds[0][i][j])` via [`clip1_single_ref`], written to the
   caller-supplied `pred_out: &mut [u16]` per-block buffer (the merge
   into [`PartitionWalker::curr_frame`] remains the walker's
   responsibility — it owns the `(x, y)` offsetting and the per-plane
   buffer).

The driver's per-task `predict_inter` is the kernel the §5.11.33
inter task list ([`PlanePredictionTask`] entries with
`mode == COMPUTE_PRED_MODE_INTER`) was always going to invoke — r190
emitted the tasks; r194 makes them executable end-to-end on the
SIMPLE single-ref path. With this round, a single-reference P-frame
whose every inter block is `motion_mode == SIMPLE && is_compound ==
false` would produce sample-accurate prediction across the §5.11.5
walker (modulo the still-pending `RefFrames[..]` / `Mvs[..]` /
`InterpFilters[..]` plumbing the dispatcher passes through to its
caller).

Two [`Error`] variants surface the still-stubbed arms (the r194
`PredictInterCompoundUnsupported` variant was retired in r201 once
the step-14 compound combine wiring landed):

* **`Error::PredictInterWarpUnsupported`** — `motion_mode ==
  MOTION_MODE_WARPED_CAUSAL`. The §7.11.3.5-8 [`block_warp`] /
  [`setup_shear`] / [`resolve_divisor`] / [`warp_estimation`]
  kernels landed in r192; step-2/3/6/7 `useWarp` derivation +
  step-12 `(i8, j8)` 8×8 sub-block loop wiring is a next-arc target.
* **`Error::PredictInterObmcUnsupported`** — `motion_mode ==
  MOTION_MODE_OBMC`. The §7.11.3.9-10 [`overlap_blending`] /
  [`overlap_neighbour_predict_blend`] / [`get_obmc_mask`] leaves
  landed in r193; the post-step "overlapped_motion_compensation"
  mi-grid neighbour walk wiring is a next-arc target. Note: this
  arm fires *after* the translational prediction has been written
  to `pred_out` — matching the spec's "post-step" ordering, where
  the overlap blend runs on top of the existing prediction.

Test count: 1011 → 1014 (+3 in lib). New tests:

* **`r194_predict_inter_simple_runs_to_completion_on_translational_path`** —
  the driver output equals the direct composition
  `motion_vector_scaling` → `block_inter_prediction` →
  `clip1_single_ref` against the same inputs on a 4×4 `mv = [0, 0]`
  region.
* **`r194_predict_inter_stub_arms_each_surface_dedicated_error`** —
  each of `is_compound == true` / `WARPED_CAUSAL` / `OBMC` returns
  its dedicated [`Error`] variant, letting the dispatcher narrow
  against the live remaining gap.
* **`r194_predict_inter_rejects_caller_bugs`** — empty `refs`,
  undersized `pred_out`, and `bit_depth ∉ {8, 10, 12}` each return
  `PartitionWalkOutOfRange` without entering the leaves.

The §5.11.33 [`PartitionWalker::compute_prediction`] dispatcher's
`is_inter` task-list output is unchanged from r190 — the walker
continues to emit one [`PlanePredictionTask`] per `(plane, 4x4
sub-block)`; the per-task `predict_inter` body is now an executable
leaf instead of an `Error::ComputePredictionInterUnsupported` stub
on the SIMPLE single-ref path. The variant is retained as a
defensive fallback and still fires when the dispatcher itself would
need to drive compound / WARP / OBMC arms before the per-task
driver is invoked.

Round 193 lifts the **§7.11.3.9-10 OBMC (Overlapped Block Motion
Compensation) overlap-blend leaves** — the last §7.11.3 inter-prediction
sample-generation body. With r193 the entire §7.11.3 layer
(translational MC + WARP MC + OBMC overlap blend + the five compound
mechanisms) is implemented as callable Rust; the only piece still
deferred is the §7.11.3.1 driver entry point that iterates per-block
candidates against `RefFrames[..]` plane buffers and dispatches per
`motion_mode` (next arc).

Three new public bodies in `inter_pred`:

* **§7.11.3.10 [`overlap_blending`]** — the per-pixel blend kernel:
  for `i in 0..predH`, `j in 0..predW`,
  `currentPlane[i][j] = Round2( m * currentPlane[i][j] +
  (64 - m) * obmcPred[i][j], 6 )` with `m = mask[i]` on the
  `OverlapPass::Above` (vertical fall-off, spec `pass = 0`) arm and
  `m = mask[j]` on the `OverlapPass::Left` (horizontal fall-off,
  spec `pass = 1`) arm. The `>> 6` matches the §7.11.3.9 weights being
  in `0..=64`. Implemented in-place against a `u16` current-plane row-
  major buffer with explicit `(pred_x, pred_y, curr_stride)` framing
  so the caller can wire it against `CurrFrame[plane][..][..]` without
  re-allocating.

* **§7.11.3.9 [`get_obmc_mask`]** — the length-to-table dispatch per
  av1-spec p.276 lines 15381-15393: `length ∈ {2, 4, 8, 16}` maps to
  the matching `Obmc_Mask_*` slice; any other length returns
  `Obmc_Mask_32` per the spec's `else` clause.

* **§7.11.3.9 [`overlap_neighbour_predict_blend`]** — the
  `predict_overlap` step-8 wrapper: `get_obmc_mask(mask_length)` →
  [`overlap_blending`]. This is the spec-direct entry point the
  §7.11.3.1 driver will invoke once per qualifying neighbour, after
  the caller has already iterated the above-row / left-column mi-grid
  + run translational MC ([`block_inter_prediction`] +
  [`clip1_single_ref`]) for the neighbour's MV and derived
  `(predX, predY, predW, predH, pass)` per the §7.11.3.9
  `(x4, y4, step4)` walk.

Plus the [`OverlapPass`] enum (`Above` / `Left`, matching spec
`pass ∈ {0, 1}`) and the five transcribed tables (verbatim from
av1-spec.txt p.277 lines 15406-15418):

* **[`OBMC_MASK_2`]**  — `[u8; 2]  = [45, 64]`.
* **[`OBMC_MASK_4`]**  — `[u8; 4]  = [39, 50, 59, 64]`.
* **[`OBMC_MASK_8`]**  — `[u8; 8]  = [36, 42, 48, 53, 57, 61, 64, 64]`.
* **[`OBMC_MASK_16`]** — `[u8; 16] = [34, 37, 40, 43, 46, 49, 52, 54,
                                       56, 58, 60, 61, 64, 64, 64, 64]`.
* **[`OBMC_MASK_32`]** — `[u8; 32]` (rising 33→64 over the first 24
  entries, then 8 ×64 saturated tail).

The §7.11.3.9 outer mi-grid driver (the `while ( nCount < nLimit && x4
< Min(MiCols, MiCol + w4) )` above-row walk + the mirror left-column
walk + per-candidate `(candRow, candCol, step4)` derivation +
`get_plane_residual_size( MiSize, plane ) >= BLOCK_8X8` gate) is *NOT*
landed in r193: it depends on the partially-implemented mi-grid /
`MiSizes[..]` / `RefFrames[..]` / `Mvs[..]` state and is part of the
§7.11.3.1 driver wiring deferred to the next arc. Once the driver
lands, it consumes [`overlap_neighbour_predict_blend`] once per
qualifying neighbour with the §7.11.3.4 MC output for that neighbour's
MV.

Test count: 998 → 1011 (+13 in lib). New tests include
`r193_obmc_mask_table_shapes` (lengths + every entry ≤ 64),
`r193_obmc_mask_first_and_last_values` (literal first/last entries of
all five tables), `r193_get_obmc_mask_dispatches_each_size`,
`r193_get_obmc_mask_else_returns_mask_32` (`length ∈ {0, 1, 7, 33}`
all fall through),
`r193_overlap_blending_identity_when_obmc_equals_curr` (the fixed-
point invariant `m * v + (64 - m) * v = 64 * v` survives `Round2(_, 6)`
exactly because `64 = 1 << 6`),
`r193_overlap_blending_above_uses_row_mask` /
`r193_overlap_blending_left_uses_col_mask` (per-row vs per-col mask
indexing with `curr = 0`, `obmc = 64`, output equals `64 - mask[axis]`),
`r193_overlap_blending_is_convex_combination` (output bounded by
`[min(curr, obmc), max(curr, obmc)]`),
`r193_overlap_blending_above_hand_computed_8tap`
(`Round2(36 * 200 + 28 * 40, 6) = 130` and the last-row
`Obmc_Mask_8[7] = 64` boundary identity),
`r193_overlap_blending_offsets_respected`
(`(pred_x, pred_y) = (2, 3)` leaves rows < 3 + cols < 2 untouched),
`r193_overlap_blending_rejects_caller_bugs` (7 caller-bug cases
including `pred_w == 0`, `obmc_stride < pred_w`,
`curr_stride < pred_x + pred_w`, short `mask`),
`r193_overlap_neighbour_predict_blend_matches_direct`,
`r193_overlap_neighbour_predict_blend_falls_back_to_mask_32`
(`mask_length = 100` hits `OBMC_MASK_32[0] = 33`).

**Deferred to next arc.** §7.11.3.1 driver entry point that
iterates per-block MC candidates against `RefFrames[refList]` plane
buffers and wires the four §7.11.3 sample-generation kernels
(translational, WARP, OBMC overlap-blend, compound blends) per the
§5.11.27-decoded `motion_mode`; §7.14 loop filter; §9.5.3 Quantizer_Matrix
table; §5.11.17 `read_var_tx_size` recursion (inter `TX_MODE_SELECT &&
!skip && !lossless` arm of §5.11.16); §5.11.22 intra-block-in-inter-
frame stub.

Round 192 lifts the **§7.11.3.5-8 WARP motion compensation kernel** —
the affine warp predictor that the §7.11.3.1 driver invokes on the
LOCALWARP (`motion_mode == WARPED_CAUSAL` per §5.11.27) and
GLOBAL_GLOBALMV (`§5.11.x` with `gm_type[refFrame] >= AFFINE`) arms.
With r192 the only inter-prediction MC body still pending is OBMC
(§7.11.3.9-10).

Four new public bodies in `inter_pred`:

* **§7.11.3.5 [`block_warp`]** — the warp kernel. Given a per-block
  8-sample sub-section index `(i8, j8)`, the spec's 6-element
  `warp_params` matrix, and the pre-computed shear coefficients, the
  body projects each (`srcX`, `srcY`) through the affine matrix into
  the reference frame, then applies the spec's two-pass 8-tap horizontal
  + vertical convolution with phase index `Round2(sx,
  WARPEDDIFF_PREC_BITS) + WARPEDPIXEL_PREC_SHIFTS` into the
  `WARPED_FILTERS` table. The intermediate buffer is the spec-literal
  `intermediate[15][8]` (`(i1+7, i2+4)` indexing for `i1 ∈ [-7, 8)`
  and `i2 ∈ [-4, 4)`). The vertical pass clips its `i1` / `i2` upper
  bounds against the residual block area (`Min(4, h - i8*8 - 4)` /
  `Min(4, w - j8*8 - 4)`) per av1-spec p.267 lines 14895-14896. Output
  is the §7.11.3.5 `Round2(s, InterRound1)` result, suitable for the
  §7.11.3.1 compound or single-ref final-clip step.

* **§7.11.3.6 [`setup_shear`]** — derives `(alpha, beta, gamma, delta,
  warpValid)` from the 6-element warp matrix. `alpha0 = clip(
  warpParams[2] - (1<<16))`, `beta0 = clip(warpParams[3])`, `gamma0`
  / `delta0` use the §7.11.3.7 divisor of `warpParams[2]`, then each
  is `Round2Signed(_, WARP_PARAM_REDUCE_BITS=6) << 6`. The final
  bound `4|α| + 7|β| < 2^16` and `4|γ| + 4|δ| < 2^16` drives
  `warpValid`. Returns `None` only on the divisor-fail
  `warpParams[2] == 0` caller-bug input.

* **§7.11.3.7 [`resolve_divisor`]** — the `(divFactor, divShift)`
  fixed-point inverse helper. Computes `n = FloorLog2(|d|)`, `e = |d| -
  (1 << n)`, `f = Round2(e, n - 8)` (or `e << (8 - n)`) into the 257-
  entry `DIV_LUT` table. `divFactor` is negated when `d < 0`.
  `divShift = n + DIV_LUT_PREC_BITS = n + 14`. Returns `None` for
  `d == 0`.

* **§7.11.3.8 [`warp_estimation`]** — least-squares fit of
  `LocalWarpParams` from up to `LEAST_SQUARES_SAMPLES_MAX = 8`
  `(sy, sx, dy, dx)` candidates produced by §7.10.4
  `find_warp_samples`. Builds the symmetric 2×2 matrix `A` and length-2
  arrays `Bx, By` via the spec's `ls_product(a, b) = ((a * b) >> 2) +
  (a + b)` accumulator with the per-iter `+8 / +4` biases. The det = 0
  short-circuit emits `LocalValid = false`. Otherwise the resolved
  divisor of `det` drives `nondiag` / `diag` clamps on the four
  affine entries, and the translation `(LocalWarpParams[0],
  LocalWarpParams[1])` is derived from the current-block MV and
  `(midX, midY)` with `WARPEDMODEL_TRANS_CLAMP = 1<<23` bounds.

Plus the two transcribed tables (verbatim from av1-spec.txt lines
14919-15036 and 15117-15140):

* **[`WARPED_FILTERS`]** — `[i32; 8]; 193` (= `3 * 64 + 1` =
  `WARPEDPIXEL_PREC_SHIFTS * 3 + 1`). Every row sums to 128 (the
  filter normalises to unit response across its 8 taps).

* **[`DIV_LUT`]** — `[i32; 257]`. Monotonically non-increasing from
  `16384 = 1 << DIV_LUT_PREC_BITS` (at index 0) to `8192 = 1 << 13`
  (at index 256).

Plus the supporting constants `WARP_WARPEDMODEL_PREC_BITS = 16`,
`WARP_PARAM_REDUCE_BITS = 6`, `DIV_LUT_PREC_BITS = 14`, `DIV_LUT_BITS
= 8`, `DIV_LUT_NUM = 257`, `LS_MV_MAX = 256`,
`WARPEDMODEL_TRANS_CLAMP = 1<<23`, `WARPEDMODEL_NONDIAGAFFINE_CLAMP =
1<<13`, `WARPEDPIXEL_PREC_SHIFTS = 64`, `WARPEDDIFF_PREC_BITS = 10`,
`USE_WARP_LOCAL = 1`, `USE_WARP_GLOBAL = 2`.

Test count: 983 → 998 (+15 in lib). New tests include
`r192_resolve_divisor_zero_returns_none`,
`r192_resolve_divisor_basic_inverse` (d ∈ {1, 256, 1024} all yield
`Round2Signed(d * divFactor, divShift) ≈ 1`),
`r192_resolve_divisor_negative_flips_sign`,
`r192_setup_shear_identity_is_zero` (`[0, 0, 1<<16, 0, 0, 1<<16]` ⇒
shear quad `(0, 0, 0, 0)` + `warpValid = true`),
`r192_setup_shear_zero_diag_is_caller_bug`,
`r192_setup_shear_small_horizontal_shear` (`warpParams[2] = (1<<16) +
64` ⇒ `alpha = 64`), `r192_setup_shear_rejects_unstable_factorisation`
(`alpha = 16384` ⇒ `4|α| = 1<<16` ⇒ NOT strictly < ⇒ invalid),
`r192_warp_estimation_empty_cands_is_invalid`,
`r192_warp_estimation_single_cand_is_invalid` (single zero-vector
sample with the `+8 / +4` per-iter biases gives det = 48 ≠ 0 ⇒
formally valid; documents the spec's per-sample bias inflation),
`r192_block_warp_rejects_invalid_use_warp` /
`r192_block_warp_rejects_invalid_dims`,
`r192_block_warp_identity_on_constant_ref` (identity warp on a
constant-100 ref plane recovers `pred[i, j] = 100` at every of the
8×8 output cells via the 128-sum filter normalisation chain through
`Round2(128 * 100, 3)` then `Round2(128 * 16 * 100, 11)`),
`r192_warped_filters_table_shape` / `r192_warped_filters_rows_sum_to_128`,
`r192_div_lut_shape` (length 257, head 16384, tail 8192, monotonic).

**Deferred to next arc.** §7.11.3.9-10 OBMC + overlap blending (the
last inter-pred MC body), §7.11.3.1 driver entry point that wires the
five compound bodies + the §7.11.3.4 translational kernel + the new
§7.11.3.5 warp kernel together against `RefFrames[refList]` plane
buffers, §7.14 loop filter, §5.11.17 `read_var_tx_size` recursion
(inter `TX_MODE_SELECT && !skip && !lossless` arm of §5.11.16),
§5.11.22 intra-block-in-inter-frame stub.

Round 191 lifts the **§7.11.3.11-15 compound bodies** — the
bi-prediction blending step that combines two §7.11.3.4-formed
predictions per the §5.11.29-decoded `compound_type`. With r191 the
§7.11.3.1 inter-prediction layer admits all five compound mechanisms:
`COMPOUND_AVERAGE` (already worked via the §7.11.3.1 inline
`Clip1(Round2(p0+p1, 1+InterPostRound))`); `COMPOUND_WEDGE` (now via
§7.11.3.11 + §7.11.3.14); `COMPOUND_DIFFWTD` (now via §7.11.3.12 +
§7.11.3.14); `COMPOUND_INTRA` (now via §7.11.3.13 +
[`mask_blend_interintra`]); `COMPOUND_DISTANCE` (now via §7.11.3.15 +
[`compound_distance_blend`]). The five new public bodies (all in the
`inter_pred` module):

* **§7.11.3.11 `wedge_mask(mi_size, num_4x4_w, num_4x4_h, wedge_index,
  wedge_sign, &mut mask)`** — the COMPOUND_WEDGE per-block-size
  wedge-mask materializer. Uses the spec's generation algorithm: a
  lazily-built `MasterMask[6][64][64]` table (`MASK_MASTER_SIZE = 64`)
  filled from the three 1d driver arrays (`WEDGE_MASTER_OBLIQUE_ODD`,
  `WEDGE_MASTER_OBLIQUE_EVEN`, `WEDGE_MASTER_VERTICAL`, each `64`
  entries with values in `0..=64`) is then sliced at the `(yoff,
  xoff)` offsets the spec computes from `(get_wedge_xoff,
  get_wedge_yoff) * (w, h) >> 3`. The `flipSign` average-of-edge
  adjustment is honoured (`avg < 32` ⇒ stored sense inverted; the
  `wedge_sign` argument selects between the two stored variants). The
  `WEDGE_CODEBOOK[3][16][3]` table maps `(block_shape, wedge_index)`
  to `(direction, xoff, yoff)`; `block_shape` is `0` (tall: h4 > w4),
  `1` (wide), or `2` (square). 9 wedge-eligible block sizes
  (`BLOCK_8X8`, `8x16`, `16x8`, `16x16`, `16x32`, `32x16`, `32x32`,
  `8x32`, `32x8`) per `WEDGE_BITS`.

* **§7.11.3.12 `difference_weight_mask(bit_depth, inter_post_round,
  mask_type, preds0, preds1, w, h, &mut mask)`** — the
  COMPOUND_DIFFWTD per-pixel `m = Clip3(0, 64, 38 +
  Round2(|p0-p1|, (BD-8)+InterPostRound) / 16)` driver. `mask_type ==
  1` (`DIFFWTD_38_INV`) inverts to `64 - m`. Equal-prediction case
  yields the constant `38` (or `26` for inverted).

* **§7.11.3.13 `intra_mode_variant_mask(interintra_mode, w, h, &mut
  mask)`** — the COMPOUND_INTRA smooth-mask driver. The `II_V_PRED` /
  `II_H_PRED` / `II_SMOOTH_PRED` arms index into the 128-entry
  `II_WEIGHTS_1D` table via `i * sizeScale`, `j * sizeScale`, or
  `Min(i,j) * sizeScale`. The `II_DC_PRED` arm emits a uniform `32`
  (50/50 blend). Table is monotonically non-increasing from `60` at
  index `0` to `1` at index `127` (verified by an invariant test).

* **§7.11.3.14 `mask_blend(bit_depth, inter_post_round, sub_x, sub_y,
  preds0, preds1, w, h, mask, mask_stride, &mut out)`** — the
  inter-inter mask-driven blend per `out = Clip1(Round2(m * p0 +
  (64-m) * p1, 6 + InterPostRound))`. Honours the `(sub_x, sub_y)`
  chroma-mask averaging: `(0,0)` reads `Mask[y][x]` directly; `(1,0)`
  averages two adjacent luma-mask values; `(0,1)` averages two
  vertical neighbours; `(1,1)` averages four (`Round2(sum, 1)` /
  `Round2(sum, 2)`). The companion **`mask_blend_interintra(bit_depth,
  inter_post_round, preds0, w, h, mask, &mut dst)`** body handles the
  `IsInterIntra == 1` arm where `pred1 = CurrFrame[plane][...]`
  (already holding the intra prediction); the inter prediction is
  pre-Clip'd via `Clip1(Round2(preds[0], InterPostRound))` and blended
  with the shift-6 path the spec specifies.

* **§7.11.3.15 `distance_weights(order_hint_bits, current_order_hint,
  order_hint_ref0, order_hint_ref1) -> DistanceWeights {fwd_weight,
  bck_weight}`** — the COMPOUND_DISTANCE `(FwdWeight, BckWeight)`
  derivation from the two refs' `OrderHints[]` deltas relative to the
  current frame. Includes the §5.9.3 `get_relative_dist(a, b,
  order_hint_bits)` sign-extending subtraction (returns `0` when
  `order_hint_bits == 0` per `enable_order_hint == 0`). The
  `Quant_Dist_Weight[4][2]` + `Quant_Dist_Lookup[4][2]` tables are
  transcribed verbatim; the early-return `d0 == 0 || d1 == 0` branch
  is honoured (chooses row `[3]`), as is the per-`i` search loop
  with the `order_bool ? d0*c0 > d1*c1 : d0*c0 < d1*c1` break
  condition. `MAX_FRAME_DISTANCE = 31` saturates the per-list
  distances. The two output weights sum to `16` by construction (an
  invariant test asserts `QUANT_DIST_LOOKUP[i][0] +
  QUANT_DIST_LOOKUP[i][1] == 16` for every row). Companion
  **`compound_distance_blend(bit_depth, inter_post_round, weights,
  preds0, preds1, w, h, &mut out)`** applies the spec's `Clip1(
  Round2(FwdWeight * preds[0] + BckWeight * preds[1], 4 +
  InterPostRound))` site (av1-spec p.258 line 14408).

Test count: 954 → 983 (+29 across lib + integration). New
inter_pred tests in lib (29 total): `r191_wedge_master_1d_values_within_0_64`
+ `r191_master_mask_table_values_within_0_64` (every entry in `0..=64`);
`r191_master_mask_horizontal_is_vertical_transpose` +
`r191_master_mask_oblique27_is_oblique63_transpose` (the spec's
derivation symmetries hold across all 4096 entries); `r191_block_shape_truth_table`
(tall / wide / square dispatch); `r191_wedge_mask_block_8x8_values_within_0_64`
(all 32 `(wedge_index, wedge_sign)` combinations of BLOCK_8X8 yield
in-range pixels); `r191_wedge_mask_sign_flip_inverts` (sign flip
toggles `64 - m` per pixel); `r191_wedge_mask_rejects_invalid_inputs`
(six caller-bug guards); `r191_difference_weight_mask_equal_preds_yields_38`
(spec degenerate case = 38 / 26); `r191_difference_weight_mask_saturates_at_64`
(extreme-diff saturation in both directions); `r191_difference_weight_mask_rejects_invalid_inputs`
(five guards); `r191_intra_mode_variant_mask_dc_yields_32` (II_DC_PRED ⇒
uniform `32`); `r191_intra_mode_variant_mask_v_pred_row_uniform` (rows
match `Ii_Weights_1d`); `r191_intra_mode_variant_mask_h_pred_col_uniform`
(columns match `Ii_Weights_1d`); `r191_intra_mode_variant_mask_smooth_pred_diagonal_symmetric`
(`Min(i,j)` symmetry); `r191_ii_weights_1d_monotone_non_increasing`;
`r191_intra_mode_variant_mask_rejects_invalid_inputs`; `r191_mask_blend_full_mask_yields_pred0`
(m=64 ⇒ p0, m=0 ⇒ p1); `r191_mask_blend_chroma_subsample_4x_average`
(4× luma → 1 chroma averaging matches expectation); `r191_mask_blend_interintra_endpoints`
(m=64 ⇒ unchanged intra dst; m=0 ⇒ Clip'd inter pred); `r191_mask_blend_rejects_invalid_inputs`;
`r191_get_relative_dist_truth_table` (sign extension at 7-bit OHB +
the `OHB == 0` short-circuit); `r191_quant_dist_lookup_sums_to_16`;
`r191_distance_weights_equal_distances` (`d0 == d1` ⇒ weights sum to
16); `r191_distance_weights_zero_distance_branch` (early-return
`[3]` row); `r191_distance_weights_no_order_hint` (OHB=0 path);
`r191_distance_weights_asymmetric_far_forward` (loop falls through to
`i = 3`); `r191_compound_distance_blend_symmetric` (8/8 weights ⇒
arithmetic mean); `r191_compound_distance_blend_rejects_invalid_inputs`.

**Deferred to subsequent arcs.** §7.11.3.5-8 warp / shear / divisor
/ warp-estimation (LOCALWARP / GLOBAL_GLOBALMV affine warp),
§7.11.3.9-10 OBMC + overlap blending, §7.14 loop filter, §5.11.17
`read_var_tx_size` recursion (inter `TX_MODE_SELECT && !skip &&
!lossless` arm of §5.11.16), §5.11.22 intra-block-in-inter-frame
stub. The §7.11.3.1 driver itself still awaits a `predict_inter`
entry point that wires the five compound bodies and the §7.11.3.4
single-ref kernel together against `RefFrames[refList]` plane
buffers; today `decode_block_syntax`'s inter arm short-circuits
inside `compute_prediction`.

Round 190 is the **`decode_block_syntax` inter-arm wire-up** —
integration arc that lifts the historical
`Error::DecodeBlockInterFrameUnsupported` short-circuit from the
§5.11.5 walker and threads MV / ref-frame / interp-filter state from
the §5.11.18 dispatcher into the §5.11.33 `compute_prediction` inter
arm (per r189) and the §5.11.34 `residual()` `is_inter && !Lossless
&& !plane` `transform_tree` arm.

The change set:

* `decode_block_syntax` and `decode_partition_syntax` grow an
  `inter_ctx: Option<&InterFrameContext>` last argument. `None`
  preserves the pre-r190 stub on `frame_is_intra == false` (legacy
  callers see no behaviour change); `Some(&ctx)` routes through the
  full §5.11.18 cascade and onward into prediction + residual.
* A new public `InterFrameContext` struct bundles the 25+ per-frame
  scalars `decode_inter_frame_mode_info` consumes (segmentation
  overrides, `skip_mode_present`, `skip_mode_frame[]`,
  `reference_select`, the §5.9.24 `gm_type` / `gm_params` /
  `ref_frame_sign_bias` triple, `allow_high_precision_mv` /
  `force_integer_mv` / `use_ref_frame_mvs`, the §5.11.27
  motion-mode trio `is_motion_mode_switchable` /
  `allow_warped_motion` / `is_scaled_per_ref[]`, the §5.11.28 /
  §5.11.29 compound-blend quad `enable_interintra_compound` /
  `enable_masked_compound` / `enable_jnt_comp` / `dist_equal`, and
  the §5.11.x interp-filter pair `interpolation_filter` /
  `enable_dual_filter`). An `identity_default()` constructor seeds
  every flag off, identity §5.9.24 warp params
  (`gm_params[ref][2] = gm_params[ref][5] = 1 << WARPEDMODEL_PREC_BITS`,
  `gm_type = [GM_TYPE_IDENTITY; 8]`), and `EIGHTTAP` interp filter.
* The §5.11.18 `decode_inter_frame_mode_info` dispatcher's `Ok(_)`
  arm is lifted from `Err(InterBlockModeInfoUnsupported)` to
  `Ok(DecodedInterFrameModeInfo { inter_block: Some(_), .. })`. A
  new `inter_block: Option<DecodedInterBlockModeInfo>` field on
  the aggregate carries the §5.11.23 readout through to the
  §5.11.5 walker.
* An internal `decode_block_syntax_inter_arm` helper bundles the
  §5.11.5 inter-arm composition: §5.11.18 cascade → §5.11.16
  `read_block_tx_size` (the §5.11.17 var-tx-size recursion is a
  next-arc target, the walker uses the §5.11.15 `else` arm
  fallback) → §5.11.33 `compute_prediction` (inter arm) → §5.11.34
  `residual` (the `is_inter && !Lossless && !plane`
  `transform_tree` arm fires on luma; chroma uses the direct
  iteration path).

End-to-end smoke: a P-frame `BLOCK_8X8` block with
`seg_globalmv_active = true` (§5.11.20 third arm forces
`is_inter = 1` without an S() read), `seg_skip_active = false`, no
compound, no interintra, no warp, identity globalmv ⇒ MV = [0, 0],
walks cleanly through every leaf. The §5.11.31 grid stamp records
the MV over the `bw4 * bh4` footprint; the §5.11.18 grid stamp
records `IsInters[r][c] = 1` over the same footprint.

The §5.11.18 `is_inter == 0` arm of an inter frame (the §5.11.22
intra-block stub) is unchanged — that arm still surfaces
`Err(IntraBlockModeInfoUnsupported)` for the next arc.

Test count: 949 → 954 (+5). New tests:
`r190_decode_block_syntax_with_inter_ctx_runs_inter_arm_to_completion`
(end-to-end on the `seg_globalmv_active` path);
`r190_decode_partition_syntax_with_inter_ctx_routes_through_inter_arm`
(the §5.11.4 partition driver propagates the inter context to its
`BLOCK_4X4` leaf);
`r190_decode_block_syntax_inter_arm_without_ctx_keeps_legacy_stub`
(`None` preserves the historical short-circuit);
`r190_inter_frame_context_identity_default_matches_spec_identity_warp`
(`identity_default()` matches the §5.9.24 identity-warp defaults);
`r190_decoded_inter_frame_mode_info_intra_arm_still_stubs` (the
§5.11.22 intra-arm stub is unchanged). Two pre-existing
`decode_inter_frame_mode_info_*` tests were converted from
asserting `Err(InterBlockModeInfoUnsupported)` to asserting
`Ok(_)` with `inter_block: Some(_)` populated.

**Deferred to subsequent arcs.** §7.11.3.5–8 warp / shear /
divisor / warp-estimation (LOCALWARP / GLOBAL_GLOBALMV affine
warp), §7.11.3.9–10 OBMC + overlap blending, §7.11.3.11–15
compound bodies (wedge_mask / diff weight / mask_blend /
distance_weights), §7.14 loop filter. The §5.11.17 `read_var_tx_size`
recursion (inter `TX_MODE_SELECT && !skip && !lossless` arm of
§5.11.16). The §5.11.22 intra-block-in-inter-frame stub.

Round 189 lifts the **§7.11.3 inter prediction process** (av1-spec
p.257-265) — the translational single-reference motion-compensation
sample-generation surface. A new `inter_pred` module bundles three
spec-faithful leaves the future §7.11.3.1 driver invokes on the
`useWarp == 0` branch:

* §7.11.3.2 `rounding_variables(bit_depth, is_compound)` — the
  `(InterRound0, InterRound1, InterPostRound)` rounding-shift
  derivation (six-case spec table verified bit-exact: 8/10-bit
  single-ref → `(3, 11, 0)`, 8/10-bit compound → `(3, 7, 4)`,
  12-bit single-ref → `(5, 9, 0)`, 12-bit compound → `(5, 7, 2)`).
* §7.11.3.3 `motion_vector_scaling(plane, subX, subY, FW, FH,
  RefUpscaledWidth, RefFrameHeight, x, y, mv)` — the
  `(startX, startY, stepX, stepY)` reference-frame sample-location
  derivation in `SCALE_SUBPEL_BITS = 10` fractional bits, including
  the `halfSample = 1 << (SUBPEL_BITS - 1)` centre-of-sample offset,
  the `(2 * mv) >> subX` per-plane MV downshift, and the `off = 1
  << (SCALE_SUBPEL_BITS - SUBPEL_BITS) / 2` tap-selection rounding
  bias.
* §7.11.3.4 `block_inter_prediction(...)` — the 8-tap horizontal +
  vertical convolution kernel with `Clip3(0, lastX/Y, ...)` boundary
  clamp and `Round2(s, R0/R1)` after each pass. Drives the spec's
  `intermediateHeight = ((h-1) * yStep + 1023) >> 10 + 8`
  intermediate-array sizing and the phase index `(p >> 6) &
  SUBPEL_MASK`.

The 6 × 16 × 8 `Subpel_Filters[]` table (av1-spec p.263-265) is
transcribed verbatim into `SUBPEL_FILTERS`: four 8-tap full-size
filters (EIGHTTAP / EIGHTTAP_SMOOTH / EIGHTTAP_SHARP / BILINEAR) and
two 4-tap small-block reductions (`EIGHTTAP_4TAP` = index 4,
`EIGHTTAP_SMOOTH_4TAP` = index 5). Three table invariants verified
across all 96 phase rows: every coefficient is even (av1-spec p.265
note); each row sums to `1 << FILTER_BITS = 128` (sample
conservation); the phase-0 row of every filter is the unit copy
`(0, 0, 0, 128, 0, 0, 0, 0)` so an integer-aligned tap reproduces the
reference sample exactly. The small-block 4-tap rows zero taps 0, 1,
6, 7 at every phase.

`select_interp_filter_small_block(f)` implements the spec's `w <= 4`
/ `h <= 4` per-pass remap (`EIGHTTAP` / `EIGHTTAP_SHARP` →
`EIGHTTAP_4TAP`, `EIGHTTAP_SMOOTH` → `EIGHTTAP_SMOOTH_4TAP`).
`clip1_single_ref(bit_depth, pred, out)` implements the §7.11.3.1
post-prediction `Clip1(preds[0])` clamp callers need for the
single-reference, non-interintra path (av1-spec p.258 line 14402).

The §5.11.33 `compute_prediction` dispatcher's `is_inter == 1` arm
now emits one `PlanePredictionTask` per `(plane, i4, j4)` 4x4
sub-block carrying `mode = COMPUTE_PRED_MODE_INTER`, `log2_w =
log2_h = 2`, `start_x = baseX + j4 * 4`, `start_y = baseY + i4 * 4`
— the per-task `predict_inter` body is the standalone
`block_inter_prediction` leaf callers invoke against a real
`RefFrames[refList]` plane buffer plus per-block `(InterpFilters,
Mvs, RefFrames)` triple. A BLOCK_8X8 luma-only inter block now
produces four tasks at `(0,0)`, `(4,0)`, `(0,4)`, `(4,4)`. The
previous `Error::ComputePredictionInterUnsupported` short-circuit is
retired (the variant is retained for API stability + future defensive
use).

Test count: 935 → 949 (+14). New tests: `rounding_variables`
six-case spec-table verification + invalid-bit-depth guard;
`motion_vector_scaling` identity-case `stepX = stepY = 1024` +
integer-MV origin shift + six caller-bug guards; `SUBPEL_FILTERS`
all-even invariant (768 entries) + phase-0 unit-copy (6 rows) +
phase-rows-sum-to-128 (96 rows) + small-block 4-tap structure
(zeros at taps 0/1/6/7); `select_interp_filter_small_block` remap
truth table; `block_inter_prediction` phase-0 integer-aligned
copy-from-reference (4 × 4 sample-bit-exact) + negative-start
boundary-clamp with constant reference + six caller-bug guards;
`clip1_single_ref` per-bit-depth clamp truth table.

**Scope (sensibly split — translational single-ref MC only).**
Deferred to subsequent arcs: §7.11.3.5 `block_warp` (LOCALWARP /
GLOBAL_GLOBALMV affine warp), §7.11.3.6 `setup_shear`, §7.11.3.7
`resolve_divisor`, §7.11.3.8 `warp_estimation`, §7.11.3.9
`overlapped_motion_compensation` (OBMC), §7.11.3.10
`overlap_blending`, §7.11.3.11 `wedge_mask` (COMPOUND_WEDGE),
§7.11.3.12 `difference_weight_mask` (COMPOUND_DIFFWTD), §7.11.3.13
`intra_mode_variant_mask` (COMPOUND_INTRA), §7.11.3.14 `mask_blend`,
§7.11.3.15 `distance_weights` (COMPOUND_DISTANCE). The §7.11.3.1
driver itself awaits the §5.11.5 walker's inter arm reaching
`predict_inter` (today `decode_block_syntax` short-circuits at
`Error::DecodeBlockInterFrameUnsupported`).

Round 188 lifts the **§7.11.2.4 six non-degenerate directional D-mode
sample-generation leaves** (D45 / D135 / D113 / D157 / D203 / D67) +
the **§7.11.2.{7, 9, 10, 11, 12} intra-edge helpers** (filter corner /
strength selection / upsample selection / 2x upsample / 5-tap filter)
+ the **`Mode_To_Angle[ INTRA_MODES ]`** (av1-spec p.485) and
**`Dr_Intra_Derivative[ 90 ]`** (av1-spec p.487) tables. With r188
the §5.11.33 `compute_prediction` dispatcher's intra arm admits all
**13 §6.10.x Y intra modes** — `Error::
ComputePredictionIntraModeUnsupported` is reachable only via the
caller-bug `UV_CFL_PRED == 13` on the luma plane.

`predict_intra_directional(w, h, p_angle, upsample_above,
upsample_left, above_row, left_col, pred)` transcribes §7.11.2.4
steps 5-9 (av1-spec p.245-248) verbatim. The body fans out on
`p_angle`:

* `p_angle < 90` (step 7 — D45 / D67): per-cell `idx = (i + 1) * dx`,
  `base = (idx >> (6 - upsampleAbove)) + (j << upsampleAbove)`,
  `shift = ((idx << upsampleAbove) >> 1) & 0x1F`, `maxBaseX = (w + h
  - 1) << upsampleAbove`. Output is
  `Round2(AboveRow[base] * (32 - shift) + AboveRow[base + 1] * shift,
  5)` when `base < maxBaseX`, else `AboveRow[maxBaseX]`.
* `90 < p_angle < 180` (step 8 — D113 / D135 / D157, hybrid): per-cell
  `idx_x = (j << 6) - (i + 1) * dx`. When `base_x = idx_x >> (6 -
  upsampleAbove)` is `>= -(1 << upsampleAbove)` the AboveRow arm
  fires (above projection); otherwise the spec re-projects along the
  Y axis: `idx_y = (i << 6) - (j + 1) * dy`, then a LeftCol blend.
* `p_angle > 180` (step 9 — D203, mirror of step 7): per-cell `idx =
  (j + 1) * dy`, `base = (idx >> (6 - upsampleLeft)) + (i <<
  upsampleLeft)`, blend on `LeftCol[base]` / `LeftCol[base + 1]`.

`dx` / `dy` are looked up from `DR_INTRA_DERIVATIVE[]` per §7.11.2.4
step 5 / step 6 (av1-spec p.246). The dispatch is `p_angle < 90 ⇒ dx
= DR_INTRA_DERIVATIVE[ p_angle ]`; `90 < p_angle < 180 ⇒ dx =
DR_INTRA_DERIVATIVE[ 180 - p_angle ]` + `dy = DR_INTRA_DERIVATIVE[
p_angle - 90 ]`; `p_angle > 180 ⇒ dy = DR_INTRA_DERIVATIVE[ 270 -
p_angle ]`. The `dx` value for D45 (`pAngle = 45`) is `64`, for D67
(`pAngle = 67`) is `27`, for D135 (`pAngle = 135`) is `64` (`180 -
135 = 45`), and so on per the table.

`predict_intra_d_mode(mode, angle_delta, w, h, upsample_above,
upsample_left, above_row, left_col, pred)` is the §6.10.x ordinal
wrapper. Looks up `MODE_TO_ANGLE[ mode ] + angle_delta * ANGLE_STEP`
and delegates to `predict_intra_directional`. The r188 dispatcher
passes `angle_delta = 0` (the §5.11.5 walker has not yet stitched in
`intra_angle_info`), so `pAngle == MODE_TO_ANGLE[ mode ]` for every
D-mode: D45 → 45, D135 → 135, D113 → 113, D157 → 157, D203 → 203,
D67 → 67. Each falls into one of the step-7 / step-8 / step-9 arms
naturally.

`MODE_TO_ANGLE[ INTRA_MODES ]` (av1-spec p.485) is transcribed as
`[0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0]`. Seven
directional entries (the V_PRED / H_PRED / six D-modes) and six
non-directional zero entries (DC, three SMOOTH arms, PAETH).
`ANGLE_STEP = 3` per the same page.

`DR_INTRA_DERIVATIVE[ 90 ]` (av1-spec p.487) is transcribed verbatim
with 27 non-zero per-angle derivatives at the angles the §6.10.x
enumeration plus `angle_delta ∈ -3..=3` can produce (3°, 6°, 9°,
…, 87° in multiples of 3°). The non-zero pins:

| index | value | index | value | index | value |
|------:|------:|------:|------:|------:|------:|
| 3 | 1023 | 36 | 90 | 67 | 27 |
| 6 | 547 | 39 | 80 | 70 | 23 |
| 9 | 372 | 42 | 71 | 73 | 19 |
| 14 | 273 | 45 | 64 | 76 | 15 |
| 17 | 215 | 48 | 57 | 81 | 11 |
| 20 | 178 | 51 | 51 | 84 | 7 |
| 23 | 151 | 54 | 45 | 87 | 3 |
| 26 | 132 | 58 | 40 | | |
| 29 | 116 | 61 | 35 | | |
| 32 | 102 | 64 | 31 | | |

`filter_corner(left_col_0, above_left, above_row_0) -> u16`
transcribes §7.11.2.7 (av1-spec p.252) — three-tap filter
`(5, 6, 5) / 16` producing the new value for the top-left corner cell.
Each kernel sums to 16, so a constant input is preserved (`(5*c +
6*c + 5*c + 8) >> 4 == c`).

`intra_edge_filter_strength_selection(w, h, filter_type, delta) ->
u8` and `intra_edge_upsample_selection(w, h, filter_type, delta) ->
u8` transcribe §7.11.2.9 (p.253) and §7.11.2.10 (p.254). The
strength-selection branches on `blkWh = w + h` and `|delta|`; the
two consecutive `<= 12` and `<= 16` `filter_type == 0` arms are
collapsed (identical body per spec — `if d >= 40 strength = 1`). The
upsample-selection short-circuits `|delta| <= 0 || |delta| >= 40`
to `0`.

`intra_edge_upsample(num_px, bit_depth, buf)` transcribes §7.11.2.11
(p.255). 2x in-place upsampling with the 4-tap symmetric kernel
`(-1, 9, 9, -1) / 16` plus `Clip1` to `BitDepth`. The buffer layout
shifts index `-2` / `-1` into offsets `0` / `1`; offset `i + 2`
holds spec index `i`. Constant input is preserved (the inner term
sums to 16).

`intra_edge_filter(sz, strength, buf)` transcribes §7.11.2.12
(p.255-256). 5-tap kernel `Intra_Edge_Kernel[ strength - 1 ]`
applied at `i ∈ 1..sz`, with `Clip3( 0, sz - 1, i - 2 + j )`
boundary handling. Strength `0` is an early-return no-op. Constant
input is preserved across strengths 1 / 2 / 3 (each kernel row sums
to 16).

The §5.11.33 `compute_prediction` dispatcher's intra-mode gate widens
to `0..INTRA_MODES` — every `plane_mode` in `0..13` is admitted.
`Error::ComputePredictionIntraModeUnsupported` is reachable only when
`plane_mode == UV_CFL_PRED == 13` arrives on the luma plane (the
§5.11.7 / §5.11.22 readers cap at the legal per-plane range, so no
conformant stream surfaces this error post-r188). New named §6.10.x
intra-mode ordinals exposed: `D45_PRED = 3`, `D135_PRED = 4`,
`D113_PRED = 5`, `D157_PRED = 6`, `D203_PRED = 7` (`D67_PRED = 8`
already exposed via the r180 boundary marker).

The r188 dispatcher takes the **no-upsample, no-filter** path
through §7.11.2.4 step 4 (the `enable_intra_edge_filter == 0`
short-circuit + `useUpsample == 0` short-circuit from the §7.11.2.10
selection). The §7.11.2.{9, 10, 11, 12} helpers are exposed as
standalone functions so a future arc can stitch them into the
walker; today's path produces the correct shape with un-filtered /
un-upsampled edges (lower quality, but spec-conformant for the
non-`enable_intra_edge_filter` profile).

Test count: 914 → 935 (+21). New tests: `MODE_TO_ANGLE` length +
13-cell value pin; `DR_INTRA_DERIVATIVE` length + 27 non-zero
index/value pins + every-other-cell-is-zero pin;
`INTRA_EDGE_KERNEL` three-row transcription pin + per-row sum-to-16
DC-preserving pin; `filter_corner` constant-input identity +
asymmetric mid-point hand-trace; `intra_edge_upsample_selection`
truth-table; `intra_edge_filter_strength_selection` truth-table
covering both `filter_type` values across all `blkWh` branches;
`intra_edge_filter` strength-0 no-op + constant-input
DC-preservation across strengths 1 / 2 / 3; `intra_edge_upsample`
constant-input DC-preservation; `predict_intra_directional` D45 /
D135 / D203 constant-neighbour identity (the
`32 * c / 32 = c` collapse on every projection step); D45 ramp
diagonal-copy hand-trace (with `dx = DR_INTRA_DERIVATIVE[45] = 64`,
`pred[i][j] = AboveRow[i + 1 + j]` up to the `max_base_x = w + h -
1 = 7` ceiling); `predict_intra_d_mode` ordinal dispatch across all
six D-modes; caller-bug guards on out-of-range mode (`V_PRED` /
`SMOOTH_PRED`) / out-of-range `angle_delta` (`±4`);
`predict_intra_directional` caller-bug guards on degenerate `pAngle
∈ {90, 180}` / out-of-range `pAngle` / `w == 0` / `h > 64` /
upsample flag > 1 / short pred buffer; `intra_edge_filter` /
`intra_edge_upsample` caller-bug guards; dispatcher acceptance for
each of the six D-modes (`3..=8`) on a 3-plane 4:2:0 block;
dispatcher rejection of `UV_CFL_PRED == 13` on luma as
`ComputePredictionIntraModeUnsupported`; dispatcher rejection of
out-of-range mode `== 14` as `PartitionWalkOutOfRange`. `decode_av1`
/ `encode_av1` still return `Error::NotImplemented`.

Round 187 lifts the **§7.11.2.2 PAETH and §7.11.2.6 SMOOTH /
SMOOTH_V / SMOOTH_H sample-generation leaves**, alongside the
**§7.11.2.1 `AboveRow[-1]` / `LeftCol[-1]` corner-sample
derivation** (the corner cell PAETH needs and r186 deferred). The
§5.11.33 `compute_prediction` dispatcher's intra arm now admits
seven of thirteen Y intra modes —
`{DC_PRED, V_PRED, H_PRED, SMOOTH_PRED, SMOOTH_V_PRED,
SMOOTH_H_PRED, PAETH_PRED}` — leaving only the six §7.11.2.4
non-degenerate directional D-modes (D45 / D67 / D113 / D135 / D157
/ D203) for the next arc.

`derive_above_left(have_above, have_left, x, y, curr_frame,
curr_frame_cols, bit_depth) -> u16` transcribes §7.11.2.1 corner
arms 1-4 (av1-spec p.242). Four-arm dispatch on
`(haveAbove, haveLeft)`: `(1, 1)` reads
`CurrFrame[plane][y-1][x-1]` (the up-left diagonal); `(1, 0)`
reads `CurrFrame[plane][y-1][x]`; `(0, 1)` reads
`CurrFrame[plane][y][x-1]`; `(0, 0)` returns `1 << (BitDepth - 1)`
— the symmetric mid-grey value, NOT the `±1`-offset asymmetric
values arms 1 / 2 of `derive_above_row` / `derive_left_col` use.
The spec sets `LeftCol[-1] = AboveRow[-1]`, so a single helper
covers both.

`predict_intra_paeth_pred(w, h, above_row, left_col, above_left,
pred)` transcribes the §7.11.2.2 basic intra prediction process
(av1-spec p.243) verbatim. For each `(i, j)`: `base = AboveRow[j]
+ LeftCol[i] - AboveRow[-1]`; compute `pLeft = |base -
LeftCol[i]|`, `pTop = |base - AboveRow[j]|`, `pTopLeft = |base -
AboveRow[-1]|`; pick `LeftCol[i]` if `pLeft <= pTop && pLeft <=
pTopLeft` (first arm), else `AboveRow[j]` if `pTop <= pTopLeft`
(second arm), else `AboveRow[-1]` (third arm). The intermediates
are computed in `i32` so the subtraction is exact. The first arm
dominates when `AboveRow[j]` and `AboveRow[-1]` are close (PAETH
degenerates to H_PRED); the second arm dominates when `LeftCol[i]`
and `AboveRow[-1]` are close (PAETH degenerates to V_PRED); the
third arm fires when neither edge is close to the corner.

`predict_intra_smooth_pred(log2_w, log2_h, w, h, above_row,
left_col, pred)` transcribes the §7.11.2.6 SMOOTH_PRED arm
(av1-spec p.250). 4-tap bidirectional weighted blend:
`smoothPred = smWeightsY[i] * AboveRow[j] + (256 - smWeightsY[i])
* LeftCol[h-1] + smWeightsX[j] * LeftCol[i] + (256 - smWeightsX[j])
* AboveRow[w-1]`, then `Round2(smoothPred, 9)`. The weight tables
`smWeightsX` / `smWeightsY` are picked from `Sm_Weights_Tx_{NxN}`
(N ∈ {4, 8, 16, 32, 64}) by `log2_w - 2` / `log2_h - 2`. Crucially
the SMOOTH formula references `LeftCol[h-1]` (the bottom-left of
the per-edge neighbour array, the spec's "bottom-left reference
sample") and `AboveRow[w-1]` (top-right reference) — NOT the
corner cell `AboveRow[-1]` (the corner is consumed exclusively by
PAETH).

`predict_intra_smooth_v_pred(log2_h, w, h, above_row, left_col,
pred)` transcribes the §7.11.2.6 SMOOTH_V_PRED arm. Vertical-only
2-tap blend `smWeights[i] * AboveRow[j] + (256 - smWeights[i]) *
LeftCol[h-1]`, then `Round2(_, 8)`. The top row (where
`smWeights[0] = 255`) is anchored to `AboveRow[j]`; the bottom
row (`smWeights[h-1] ∈ {64, 32, 16, 8, 4}` per the per-size
tables) is anchored to `LeftCol[h-1]`.

`predict_intra_smooth_h_pred(log2_w, w, h, above_row, left_col,
pred)` transcribes the §7.11.2.6 SMOOTH_H_PRED arm. Horizontal-
only 2-tap blend `smWeights[j] * LeftCol[i] + (256 -
smWeights[j]) * AboveRow[w-1]`, then `Round2(_, 8)`. Mirror of
SMOOTH_V along the other axis.

The `Sm_Weights_Tx_{4x4, 8x8, 16x16, 32x32, 64x64}` tables are
transcribed verbatim from av1-spec p.508 — five size variants
covering the §3 transform-size set, with monotone-decreasing
weight profiles (`[255, …, 64]` for 4x4, down to `[255, …, 4]`
for 64x64).

The §5.11.33 `compute_prediction` dispatcher's intra-mode gate now
admits `mode ∈ {DC_PRED, V_PRED, H_PRED, SMOOTH_PRED,
SMOOTH_V_PRED, SMOOTH_H_PRED, PAETH_PRED}` (seven of thirteen Y
intra modes). `Error::ComputePredictionIntraModeUnsupported` still
fires for the six remaining D-mode directional ordinals (`D45_PRED`
= 3 through `D67_PRED` = 8) — the §7.11.2.4 non-degenerate body
with `Dr_Intra_Derivative[]` driven sample projection and the
§7.11.2.{10..12} intra-edge upsample / filter pre-passes is the
next-arc target. New named §6.10.x intra-mode ordinals
`SMOOTH_PRED` = 9, `SMOOTH_V_PRED` = 10, `SMOOTH_H_PRED` = 11,
`PAETH_PRED` = 12 exposed as `pub const`.

Test count: 885 → 914 (+29). New tests: SMOOTH / PAETH ordinal
pins; `Sm_Weights_Tx_*` boundary-value transcription pins;
`derive_above_left` four-arm dispatch on top-left / above-only /
left-only / no-neighbour for 8 / 10 / 12-bit `BitDepth`;
out-of-range sample clipping to `[0, (1 << BitDepth) - 1]`; PAETH
on constant-neighbour blocks (first arm always wins); PAETH
H_PRED-degeneracy (`AboveRow[j] == corner` for all `j`); PAETH
V_PRED-degeneracy (`LeftCol[i] == corner` for all `i`); PAETH
first-arm tie-break (`pLeft == pTop` and both `<= pTopLeft`);
PAETH third-arm hand-trace where `pTopLeft = 0 < pLeft = pTop`
fires the corner-cell write; PAETH caller-bug guards;
SMOOTH_PRED constant-neighbour produces constant block (the
`c * 512 / 512 = c` identity); SMOOTH_PRED `pred[0][0]` and
`pred[h-1][w-1]` hand-traces against av1-spec p.250 (the
`(255*100 + 200 + 255*200 + 100 + 256) >> 9 = 150` pin);
SMOOTH_PRED 64×64 max-size 12-bit ceiling round-trip;
SMOOTH_V top-row / bottom-row weighting asymmetry; SMOOTH_H
left-col / right-col weighting asymmetry; all-three SMOOTH leaves'
caller-bug guards; dispatcher acceptance of SMOOTH_PRED /
SMOOTH_V_PRED / SMOOTH_H_PRED / PAETH_PRED with task-list mode
forwarding on a 3-plane 4:2:0 block; dispatcher rejection of
D45 / D135 / D67 at the D-mode-interval boundaries; end-to-end
PAETH and SMOOTH_PRED through the §7.11.2.1 derivation helpers
against a real `CurrFrame[plane]`-shaped buffer. `decode_av1` /
`encode_av1` still return `Error::NotImplemented`.

Round 186 lifts the **§7.11.2.4 V_PRED and H_PRED sample-generation
leaves** alongside the **§7.11.2.1 `AboveRow[]` / `LeftCol[]`
neighbour derivation** that feeds them — the two cheapest non-DC
intra modes, covering the degenerate directional cases where
`pAngle == 90` (V_PRED) and `pAngle == 180` (H_PRED) collapse the
§7.11.2.4 directional process to a row / column broadcast.

`predict_intra_v_pred(w, h, above_row, pred)` (§7.11.2.4 step 10)
fills `pred[i][j] = AboveRow[j]` — every row of the block is a
copy of the top-edge sample at column `j`.
`predict_intra_h_pred(w, h, left_col, pred)` (step 11) is the
mirror: `pred[i][j] = LeftCol[i]`. Both reachable when the
§5.11.5 walker's `decode_intra_block_mode_info` emits the matching
`YMode` and `angleDelta == 0`.

`derive_above_row(have_above, have_left, have_above_right, x, y,
w, h, max_x, curr_frame, curr_frame_cols, bit_depth, above_row)`
transcribes §7.11.2.1 arms 1-3 — reads the r185
`CurrFrame[plane]` buffer at the per-TU top-left to populate the
`AboveRow[0..w+h-1]` neighbour array. Honours the spec's
`haveAboveRight ? 2*w : w` tail extension, the `aboveLimit =
Min(maxX, x + extend - 1)` frame-right-boundary clamp, the
left-only constant-propagation arm, and the asymmetric
all-unavailable `(1 << (BitDepth - 1)) - 1` mid-grey fallback.
`derive_left_col(...)` is the mirror with the asymmetric
`(1 << (BitDepth - 1)) + 1` no-neighbour fill.

The §5.11.33 `compute_prediction` dispatcher's intra arm now
admits `mode ∈ {DC_PRED, V_PRED, H_PRED}` (rebroken-down from
DC_PRED-only at r180). `Error::ComputePredictionIntraModeUnsupported`
still fires for the 10 remaining intra modes (SMOOTH_PRED /
SMOOTH_V_PRED / SMOOTH_H_PRED / PAETH_PRED / D45_PRED /
D135_PRED / D113_PRED / D157_PRED / D203_PRED / D67_PRED) —
next-arc targets. New named ordinals `DC_PRED` and `H_PRED`
exposed as `pub const` alongside the pre-existing `V_PRED`.

The §7.11.2.1 corner sample (`AboveRow[-1]` / `LeftCol[-1]`) is
not derived in r186 — V_PRED and H_PRED do not consult it. It
lands with the next-arc PAETH leaf which is its only reader on
the degenerate-angle subset.

Test count: 859 → 885 (+26). New tests: V_PRED row broadcast on
4×4 / 8×4 / 64×64; H_PRED column broadcast on 4×4 / 4×8 / 64×64;
both leaves ignore neighbour-array slots past `w` / `h` (the
§7.11.2.1 prologue derives `w + h` but step-10 / step-11 only
consume `w` / `h`); caller-bug guards on each leaf;
`derive_above_row` arms 1-3 (top-row read, `haveAboveRight` tail
extension, `max_x` clamp, left-only propagation,
mid-grey-minus-one all-unavailable for 8 / 10 / 12-bit
`BitDepth`); `derive_left_col` arms 1-3 mirror; end-to-end
`derive_above_row` → `predict_intra_v_pred` + `derive_left_col`
→ `predict_intra_h_pred` round trips through a real
`CurrFrame[plane]`-shaped buffer; dispatcher acceptance of
V_PRED / H_PRED with task-list mode forwarding on a 3-plane
4:2:0 block; dispatcher rejection of D45_PRED / SMOOTH_PRED /
PAETH_PRED at the next-arc-gap boundary. `decode_av1` /
`encode_av1` still return `Error::NotImplemented`.

Round 185 lifts the **§7.12.3 step-3 frame-buffer merge** — the
per-TU sample-buffer write that turns the r182 inverse-transform
residual into the actual reconstructed `CurrFrame[plane][y][x]`
samples. With r185 the decoder produces real per-plane sample
output for the first time (subject to the §7.11.x prediction
kernels still being placeholder-zero for all intra modes except
DC_PRED).

`PartitionWalker` grows three lazily-allocated per-plane
`CurrFrame` sample buffers — one for luma (`plane == 0`), one for
each chroma plane (`plane == 1` / `plane == 2`). Per-plane
dimensions follow §5.9.5 / §5.11.34: the luma buffer is `MiRows *
MI_SIZE × MiCols * MI_SIZE` samples; chroma buffers are `>> sub_y`
/ `>> sub_x` subsampled per the §5.11.34 line 19 derivation, so a
4:2:0 stream halves both axes. The buffers are allocated on the
first §7.12.3 step-3 merge call that touches the plane (a stream
with no `eob > 0` TUs on a given plane leaves that buffer at the
`None` sentinel; the new `PartitionWalker::curr_frame(plane)` /
`curr_frame_dims(plane)` accessors return `None` until the first
merge fires). The flat sample buffer is row-major `i32` with a
zero initial fill — matching the spec's `CurrFrame[..] = Clip1(0
+ Residual[i][j])` additive-merge identity before any §7.11.x
prediction-sample writer fires.

The new `curr_frame_step3_merge` helper transcribes av1-spec
p.295 step-3 verbatim: derive `flipUD = 1` iff `PlaneTxType ∈ {
FLIPADST_DCT, FLIPADST_ADST, V_FLIPADST, FLIPADST_FLIPADST }`,
derive `flipLR = 1` iff `PlaneTxType ∈ { DCT_FLIPADST,
ADST_FLIPADST, H_FLIPADST, FLIPADST_FLIPADST }`, then iterate
`for i in 0..h, j in 0..w` with `yy = flipUD ? h-i-1 : i` and `xx
= flipLR ? w-j-1 : j` and apply `CurrFrame[plane][y + yy][x + xx]
= Clip1(CurrFrame[..][..] + Residual[i][j])` where `Clip1(x) =
Clip3(0, (1 << BitDepth) - 1, x)` per §3 `Clip1`. The bit depth
(8 / 10 / 12) is routed from the per-block `ResidualContext`'s
`QuantizerParams::bit_depth`.

`PartitionWalker::transform_block_emit` now invokes
`curr_frame_step3_merge` after the §7.13 inverse transform on the
`!skip && eob > 0` arm — driven directly from the dequantized
residual the r182 inverse transform produced and the r183 §5.11.40
`compute_tx_type` derivation of `PlaneTxType`. The merge fires
before the residual is recorded onto `ResidualReadout::residuals`
(so a caller inspecting the readout sees both the per-TU residual
AND a coherent `CurrFrame` state). The §7.12.3 step-3 task site
fires on every TU emitted by both the per-plane intra / chroma
`for y, for x` direct iteration AND the §5.11.36 `transform_tree`
inter-luma recursion.

`ResidualTuTask` gains a `plane_tx_type: u8` field — the per-TU
§5.11.40 `compute_tx_type` outcome the §7.12.3 step-3 merge
consults for `flipUD` / `flipLR`. The field is back-filled by
the dispatcher after the §5.11.40 derivation on the `!skip` arm;
the `skip == true` arm leaves it at the `DCT_DCT` placeholder
(no transform fired, the step-3 merge for the TU is gated to the
prediction-only write — which is also `DCT_DCT`'s neither-flip
identity, so the bookkeeping stays consistent).

Out-of-buffer overhang (a TU whose `start_x + w` extends past
the right frame edge, possible at the frame boundary per
§5.11.34's chunk loop) is silently clipped; the in-buffer slice
is still merged. Plane-out-of-range and wrong-residual-size
arguments are defensive no-ops.

**Split off explicitly to subsequent arcs (unchanged from r184):**
(a) 12 non-DC intra-prediction modes (V_PRED / H_PRED / SMOOTH_*
/ PAETH_PRED / D45..D203 directional); (b) §7.9 inter prediction
(sub-pixel interpolation, reference-frame sampling, warping);
(c) §9.5.3 `Quantizer_Matrix[15][2][QM_TOTAL_SIZE]` table
transcription; (d) §7.14 deblocking loop filter; (e) §7.15 CDEF;
(f) §7.16 super-resolution; (g) §7.17 loop restoration. The
§7.12.3 step-1 / step-2 / step-3 trio is now complete; the
`CurrFrame[plane][y][x]` output buffer is ready for the §7.11.x
intra prediction kernels and the §7.14 / §7.15 / §7.16 / §7.17
post-processing chain to consume.

16 new tests (843 → 859): unallocated-on-fresh-walker accessor
behaviour; `ensure_curr_frame_plane` per-subsampling sizing
(luma 16×16, 4:2:0 chroma 8×8 on the 4-mi-square fixture);
DCT_DCT identity write across a 4×4 TU (top-left 4×4 matches the
residual, untouched cells stay 0); `Clip1` 8-bit saturation at
both the upper (200 + 200 → 255) and lower (100 - 150 → 0)
envelopes; `Clip1` 10-bit saturation (upper bound 1023);
FLIPADST_DCT row-flip (dst row `r` reads src row `3 - r`);
H_FLIPADST col-flip (dst col `c` reads src col `3 - c`);
FLIPADST_FLIPADST 180° rotation (both flips fire); offset
top-left placement (4×4 TU at `(start_y = 4, start_x = 8)`
writes into rows 4-7, cols 8-11); additive accumulation across
two consecutive merges (two `+3` merges sum to 6); silent
overhang clip (4×4 TU at `start_x = 14` in a 16-wide buffer
lands cells 14, 15 only); plane-out-of-range no-op; wrong-
residual-size no-op; chroma 4:2:0 plane allocation with luma /
V left unallocated; `ResidualTuTask::plane_tx_type` back-fill on
the walker's reachable intra-only path (every TU carries
DCT_DCT after the §5.11.40 derivation on the `YMode = DC_PRED`
fixture); skip-arm `plane_tx_type` stays at the DCT_DCT
placeholder. `decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 184 lifts the **§7.5 / §5.11.41 `get_scan(txSz)` scan-table
dispatcher** — the last placeholder the r183 walker left in the
§5.11.39 coefficients-reader call site.

A new `scan` module (`pub mod scan` in `lib.rs`) transcribes the full
§7.5 scan-table family from av1-spec p.388-399 verbatim: 14 default
scans (`DEFAULT_SCAN_4X4` through `DEFAULT_SCAN_32X8`, covering every
size enumerated by `get_default_scan`), 9 column-major scans
(`MCOL_SCAN_4X4` through `MCOL_SCAN_16X4`), and 9 row-major identity
scans (`MROW_SCAN_4X4` through `MROW_SCAN_16X4`). The Mcol / Mrow
families only enumerate sizes whose `Tx_Size_Sqr_Up <= TX_16X16`
because the §5.11.40 / §5.11.48 set restrictions only admit `H_*` /
`V_*` plane-tx-types for those sizes (the spec's
`return Mrow_Scan_16x4` / `return Mcol_Scan_16x4` tail in
`get_mrow_scan` / `get_mcol_scan` is a defensive fallback on an
unreachable branch — we mirror it directly).

`scan::get_scan(tx_sz, plane_tx_type)` reproduces the §5.11.41
dispatcher verbatim from av1-spec p.94: the three size-driven
short-circuits (`TX_16X64` → `Default_Scan_16x32`, `TX_64X16` →
`Default_Scan_32x16`, `Tx_Size_Sqr_Up[txSz] == TX_64X64` →
`Default_Scan_32x32`) catch every size whose `segEob = 512` cap
(per §5.11.39 on `TX_16X64` / `TX_64X16`) or `segEob = Min(1024,
w*h) = 1024` cap (per the 32×32 coefficient-grid limit on the
larger sizes) makes the smaller scan table cover the reachable
positions; then the `IDTX` arm routes to the per-size default,
the `preferRow` arm (V_DCT / V_ADST / V_FLIPADST) routes to
`get_mrow_scan`, the `preferCol` arm (H_DCT / H_ADST / H_FLIPADST)
routes to `get_mcol_scan`, and symmetric tx types
(DCT_DCT / ADST_DCT / DCT_ADST / ADST_ADST / FLIPADST_DCT /
DCT_FLIPADST / FLIPADST_FLIPADST / ADST_FLIPADST / FLIPADST_ADST)
fall through to `get_default_scan`.

`PartitionWalker::transform_block_emit` now calls
`scan::get_scan(tx_sz, plane_tx_type)` directly and forwards the
returned `&'static [u16]` to `self.coefficients(...)`. The previous
`Vec<u16>` of `0..segEob` per-iteration allocation (and the local
`seg_eob` derivation that fed it) is dropped; the §5.11.39 reader
re-computes `segEob` internally from `tx_sz` for its zero-fill and
bounds check, so the caller never needs it.

**Split off explicitly to subsequent arcs (unchanged from r183):**
(a) §9.5.3 `Quantizer_Matrix[15][2][QM_TOTAL_SIZE]` table
transcription; (b) §7.12.3 step-3 frame-buffer merge
(`CurrFrame[plane][y + yy][x + xx]` write with `flipLR` / `flipUD`
and `Clip1`); (c) 12 non-DC intra-prediction modes; (d) §7.9 inter
prediction.

22 new tests (821 → 843): scan-permutation checks (every default /
Mcol / Mrow table visits every coefficient position exactly once);
Mrow-identity check (every `MROW_SCAN_*[i] == i`); Mcol
column-major formula check (`scan[col*h + row] == row*w + col`);
spec-anchor zig-zag prologue checks (`DEFAULT_SCAN_4X4[..4] ==
[0,1,4,8]`; `DEFAULT_SCAN_8X8[..6] == [0,1,8,16,9,2]`);
`get_default_scan` / `get_mcol_scan` / `get_mrow_scan` per-size
enumeration including spec-tail fallback; `get_scan` arm-by-arm
coverage (`TX_16X64` / `TX_64X16` / `Tx_Size_Sqr_Up == TX_64X64` /
IDTX / preferRow / preferCol / symmetric) under every
representative `PlaneTxType`; and `scan.len() >= segEob` covers
the §5.11.40-admitted `(tx_sz, plane_tx_type)` reachable subset
for every transform size in `TX_SIZES_ALL`. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 183 lifts the **§7.12.2 dequantization-function tables + §7.12.3
step-1 dequantization loop + §5.11.47 `transform_type` per-TU S()
reader** — the three split-offs the r182 walker left as placeholders.

The new §7.12.2 helpers in `cdf` expose `Dc_Qlookup[3][256]` /
`Ac_Qlookup[3][256]` (the per-bit-depth dc / ac quantizer index
tables, verbatim from av1-spec p.289-292) and the `dc_q(bit_depth, b)`
/ `ac_q(bit_depth, b)` lookups with the spec's `(BitDepth-8) >> 1`
row select and `Clip3(0, 255, b)` column clamp. `get_qindex(quant,
ignore_delta_q, seg_id)` reproduces av1-spec p.293 (segmentation
`SEG_LVL_ALT_Q` arm + `delta_q_present` running-`CurrentQIndex`
arm + `base_q_idx` fallback) and feeds `get_dc_quant(quant, plane,
seg_id)` / `get_ac_quant(quant, plane, seg_id)` (per-plane
`DeltaQ?Dc` / `DeltaQ?Ac` routing). The new `QuantizerParams`
struct carries the §5.9.12 `base_q_idx` + four `delta_q_*` slots,
the §5.9.18 `delta_q_present` / `current_q_index`, the §5.9.14
segmentation `SEG_LVL_ALT_Q` `FeatureEnabled[]` / `FeatureData[]`
tables, the §5.9.12 `using_qmatrix` bit, and the §6.4.1 `BitDepth`.

`dequantize_step1(quant_levels, tx_size, plane, seg_id,
plane_tx_type, seg_qm_level, &QuantizerParams)` runs the §7.12.3
step-1 loop verbatim:
`for i = 0..(th-1), j = 0..(tw-1) { q = (i==0 && j==0) ?
get_dc_quant : get_ac_quant; q2 = q; dq = Quant[i*tw+j] * q2;
sign = (dq < 0) ? -1 : 1; dq2 = sign * (|dq| & 0xFFFFFF) /
dqDenom; Dequant[i][j] = Clip3(-(1<<(7+BitDepth)),
(1<<(7+BitDepth)) - 1, dq2); }` — with `tw = Min(32, w)`, `th =
Min(32, h)`, and `dqDenom = 4 / 2 / 1` per av1-spec p.294
(`dequant_denom`). Cells with `i >= th || j >= tw` stay at the
§5.11.39 `Dequant[i][j] = 0` pre-loop default. The §9.5.3
`Quantizer_Matrix[15][2][QM_TOTAL_SIZE]` table is not yet
tabulated; the `using_qmatrix && plane_tx_type < IDTX &&
seg_qm_level < 15` arm therefore falls through to the no-QM
identity (`q2 = q`) path with a `debug_assert!` guard.
`seg_qm_level == 15` (the spec's "skip QM" sentinel) takes the
no-QM path directly.

`PartitionWalker::transform_type(decoder, cdfs, tx_size, is_inter,
intra_dir, reduced_tx_set, segment_id, &QuantizerParams)` reads
the §5.11.47 per-TU luma `TxType` S() against
`TileIntraTxTypeSet{1,2}Cdf` / `TileInterTxTypeSet{1,2,3}Cdf`
(routed via the existing `inter_tx_type_cdf` /
`intra_tx_type_cdf` selectors) and inverts the symbol through
the new spec inversion tables `TX_TYPE_INTRA_INV_SET1` (7-entry),
`TX_TYPE_INTRA_INV_SET2` (5-entry), `TX_TYPE_INTER_INV_SET1`
(16-entry), `TX_TYPE_INTER_INV_SET2` (12-entry),
`TX_TYPE_INTER_INV_SET3` (`{ IDTX, DCT_DCT }`). The spec's `set
> 0 && (segmentation_enabled ? get_qindex(1, segment_id) :
base_q_idx) > 0` guard short-circuits to `DCT_DCT` (no bitstream
read) on the §5.11.48 `TX_SET_DCTONLY` path AND on the
`qindex == 0` path. The result is emitted as a
`TransformTypeReadout { tx_type, w4, h4 }` so the caller can
stamp `TxTypes[ y4 + j ][ x4 + i ] = TxType` over the per-TU
4×4-luma footprint.

A new walker-level `tx_types: Vec<u8>` grid (one cell per
4×4-luma-sample / mi unit) tracks the §5.11.39 / §5.11.47
`TxTypes[][]` luma cache. Exposed via `PartitionWalker::tx_types()`
(read-only view), `tx_type_at(r, c)` (single-cell lookup with
out-of-bounds → `DCT_DCT`), and `stamp_tx_type(x4, y4, w4, h4,
tx_type)` (write the footprint with edge-clipping). The §5.11.40
`compute_tx_type(plane, ...)` helper landed in r142 then reads
`TxTypes[][]` on the chroma path via this grid.

`PartitionWalker::residual` gains a `&ResidualContext` parameter
that bundles the new per-block state (`QuantizerParams`,
`reduced_tx_set`, `intra_dir`, `segment_id`, `seg_qm_level[3]`,
`uv_mode`). The dispatcher's per-TU `transform_block_emit` now,
on `!skip`, runs `transform_type` (luma only), `compute_tx_type`
(per-plane), `get_tx_class(plane_tx_type)`, and `dequantize_step1`
in order — replacing the r182 hard-coded `DCT_DCT` + identity
dequant. The §7.13 `inverse_transform_2d` call now receives the
true per-plane `PlaneTxType` and `BitDepth`. The §5.11.5
`decode_block_syntax` walker synthesises a neutral
`ResidualContext` (`base_q_idx = 0`, no segmentation, `intra_dir
= y_mode`, `uv_mode = DC_PRED`, no QM) so the reachable walker
path exercises the §5.11.47 `q_for_guard == 0 ⇒ DCT_DCT`
short-circuit (every existing fixture in the corpus uses this
arm and remains green).

**Split off explicitly to subsequent arcs:** (a) §9.5.3
`Quantizer_Matrix[15][2][QM_TOTAL_SIZE]` table transcription
(~30 KB; needed for `using_qmatrix == 1 && seg_qm_level < 15`
fixtures); (b) §7.5 `get_scan` table dispatch (still identity);
(c) §7.12.3 step-3 frame-buffer merge (`CurrFrame[plane][y +
yy][x + xx]` write with `flipLR` / `flipUD` and `Clip1`); (d)
12 non-DC intra-prediction modes (V_PRED, H_PRED, SMOOTH, PAETH,
directional); (e) §7.9 inter prediction.

20 new tests (801 → 821): six on the dequant table primitives
(table dimensions + spec-anchor bit-exact values; row select +
`Clip3` clamp; `get_qindex` no-seg / delta-q / segmentation
paths; per-plane delta routing through `get_dc_quant` /
`get_ac_quant`); seven on the §7.12.3 step-1 dequant loop
(DC-only 4×4 at `base_q_idx == 0`, AC + dqDenom, `dqDenom == 4`
on 64×64, `Clip3` band saturation, `seg_qm_level == 15`
QM-skip); five on `transform_type` (`base_q_idx == 0`
short-circuit, `TX_SET_DCTONLY` short-circuit, intra-set1
symbol read inverted to IDTX, the inversion-table verbatim
spot-check, `ResidualContext::neutral` defaults); plus
`stamp_tx_type` grid-footprint + out-of-bounds checks and a
new end-to-end smoke (`residual_dc_only_block_pipeline_smoke`)
that drives the §5.11.34 / §5.11.35 dispatcher with the new
`ResidualContext` on the 4×4 intra-luma path with `all_zero ==
1` rigged — the new pipeline runs without observable
side-effects on the gate-closed path, confirming the wiring is
hot. `decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 182 lifts the **§7.13 inverse transform process** — the
headline §5.11.35 `reconstruct()` body the r181 walker stopped at.
A new `transform` module reproduces the §7.13.2.1 butterflies (`B(a,
b, angle, flip, r)` rotation, `H(a, b, flip, r)` Hadamard, `cos128`
/ `sin128` per the spec's 65-entry quarter-period
`Cos128_Lookup[65]`, `brev` bit-reversal) plus the full 1D
transform stack: §7.13.2.3 inverse DCT for `n in 2..=6` (sizes 4 /
8 / 16 / 32 / 64; 31-step butterfly schedule with `permute` →
n-conditional `B`/`H` stages reproduced verbatim), §7.13.2.6 /
§7.13.2.7 / §7.13.2.8 inverse ADST 4 / 8 / 16 (the ADST4
closed-form `SINPI_*` sum, ADST8 7-step schedule, ADST16 9-step
schedule, all wrapped in §7.13.2.4 input + §7.13.2.5 output
Gray-code permutations), §7.13.2.10 inverse WHT4 (Lossless path),
and §7.13.2.11..§7.13.2.14 inverse identity 4 / 8 / 16 / 32 (the
`5793 / 11586` scaling pairs and `*2 / *4` doublings). The
§7.13.3 2D dispatcher composes row-then-column over the 16-element
`PlaneTxType` matrix (DCT_DCT through H_FLIPADST), including the
`Abs(log2W - log2H) == 1` rectangular `2896` pre-scale, the
`Transform_Row_Shift[TX_SIZES_ALL]` per-size row right-shift, the
between-stage `colClampRange = Max(BitDepth + 6, 16)` clamp, and
the `Lossless` short-circuit through the WHT (shift 2 row / shift
0 col). The FLIPADST destination flip (`flipLR` / `flipUD`) is
documented as a §7.12.3-step-3 frame-buffer-write concern, not a
§7.13.3 transform-pass concern.

Wired into `PartitionWalker::residual` / `transform_block_emit`:
on every TU with `eob > 0` (the §5.11.39 `!all_zero` cascade) the
walker now invokes `inverse_transform_2d` and pushes
`Some(residual)` onto a new `ResidualReadout::residuals: Vec<Option<Vec<i64>>>`;
TUs with `eob == 0` push `None`. The walker returns
`Ok(DecodedBlock)` cleanly on every path the §5.11.5 walker can
reach today — **lifting `Error::ResidualReconstructUnsupported`**
(the variant is retained as a defensive caller-bug sentinel; the
walker no longer surfaces it).

**Split off explicitly to subsequent arcs:** (a) §7.12.3 step-1
true dequantization (`Dequant[i][j] = Quant[pos] * q2 / dqDenom`
with `q2` derived from `get_dc_quant` / `get_ac_quant` /
`using_qmatrix` / `Quantizer_Matrix[ SegQMLevel[plane][segment_id]
][ ... ]`; the walker passes `Quant[]` directly as a placeholder
identity dequant — sufficient to lift the §7.13 boundary, not
sufficient for bit-exact CurrFrame samples); (b) §7.12.3 step-3
frame-buffer merge (`CurrFrame[plane][y + yy][x + xx] =
Clip1(CurrFrame[...] + Residual[i][j])` with `xx = flipLR ? w - j
- 1 : j`, `yy = flipUD ? h - i - 1 : i`; needs a `CurrFrame[][][]`
backing buffer); (c) §5.11.40 / §5.11.41 per-TU `PlaneTxType` read
(the walker still defaults to `DCT_DCT`); (d) §7.5 `get_scan`
table dispatch (still identity scan); (e) `AboveLevelContext` /
`LeftLevelContext` neighbour grids; (f) 10/12-bit `BitDepth` —
hardcoded to `8` for now.

33 new tests (768 → 801): twelve on the 1D primitives (cos128
anchors, sin128 == cos128(a-64) over the full period, brev basic,
round2 sign-handling, DCT permute, ADST input permute n=3, zero
inputs for IDCT 4 / 8 / 16 / 32 / 64, zero inputs for IADST 4 / 8
/ 16, identity 4 / 8 / 16 / 32 scaling, identity dispatcher);
eight on the §7.13.3 2D dispatcher (DCT_DCT 4x4 DC-only bit-exact
expected = `128` per cell, DCT_DCT zero-preserves, IDTX 4x4
DC-only bit-exact = `512` at origin, Lossless zero-preserves, 4x8
rectangular zero-preserves, 8x8 / 16x16 / 32x32 / 64x64
zero-preserves, FLIPADST and V_ / H_ variants zero-preserve);
one IDCT4 linearity check; plus one new walker-integration test
(`residual_no_skip_eob_positive_invokes_inverse_transform`)
asserting that gate-open TUs do produce `Some(Residual)` of length
`Tx_Width * Tx_Height` on the `ResidualReadout`. The seven
existing `decode_block_syntax` integration tests that previously
asserted `Err(ResidualReconstructUnsupported)` now assert
`Ok(DecodedBlock)`. `decode_av1` / `encode_av1` continue to
return `Error::NotImplemented`.

Round 181 lifts the **§5.11.34 `residual()` outer dispatch** — the
final §5.11.5 walker stub between the §5.11.30 prediction pass and the
§7.6 / §7.7 reconstruction kernel. The §5.11.34 driver now walks
`widthChunks` / `heightChunks` per av1-spec p.84-85 (1 chunk per 64-luma
stride, with `Max(1, _)` clamping and the `BLOCK_64X64` `miSizeChunk`
fallback for 128-luma blocks), then for every chunk iterates `for plane
= 0..1 + HasChroma * 2` with the §5.11.37 `get_tx_size(plane, TxSize)` /
§5.11.38 `get_plane_residual_size(miSizeChunk, plane)` per-plane size
derivations. The inter-luma arm (`is_inter && !Lossless && !plane`)
recurses through the §5.11.36 `transform_tree(startX, startY, w, h)`
body — `w <= lumaW && h <= lumaH` leaf-emit (with `find_tx_size(w, h)`
size resolution); `w > h` / `w < h` / `w == h` per-direction splits
operating against the walker's `InterTxSizes[ row ][ col ]` grid (the
§5.11.16 / §5.11.17 stamps the walker already tracks). Every other
arm — non-inter luma or any chroma plane — falls through the §5.11.34
`for y = 0..num4x4H step stepY, for x = 0..num4x4W step stepX
transform_block(plane, ...)` direct iteration.

Per leaf TU the §5.11.35 `transform_block(plane, baseX, baseY, txSz, x,
y)` body runs the §5.11.35 `startX = baseX + 4 * x` /
`startY = baseY + 4 * y` derivation, the `startX >= maxX || startY >=
maxY` early-return, and — on the `!skip` arm — the §5.11.39
`coefficients(plane, startX, startY, txSz)` reader landed in r179.
Each per-TU `CoefficientsReadout` is captured alongside the per-TU task
in a new `ResidualReadout { width_chunks, height_chunks, mi_size_chunk,
num_planes_visited, tasks, coeffs, skip }` aggregate, with one
`ResidualTuTask { plane, start_x, start_y, tx_size, chunk_x, chunk_y,
from_transform_tree }` per emit.

Wired into `decode_block_syntax`, lifting
`Error::DecodeBlockResidualUnsupported`. The walker now reaches
`Error::ResidualReconstructUnsupported` (§5.11.35
`reconstruct(plane, startX, startY, txSz)` — the §7.6 inverse-transform
+ §7.13 dequant pre-pass + §7.7 reconstruction merge) on the first TU
with `eob > 0`; the `all_zero == 1` short-circuit and `skip == true`
arms both complete cleanly and return the per-block `DecodedBlock`.

**Explicitly split off to subsequent arcs (per [`ResidualReadout`] docs):**
(a) §5.11.40 `transform_type` per-luma-TU S() read into `TxTypes[]`
(defaulted to `DCT_DCT`); (b) §5.11.41 `compute_tx_type` full body
(defaulted to `DCT_DCT` alongside §5.11.40); (c) §7.5 `get_scan`
table dispatch over `Default_Scan_*` / `Mrow_Scan_*` / `Mcol_Scan_*`
(defaulted to identity scan `0..segEob`); (d) `AboveLevelContext` /
`LeftLevelContext` / `AboveDcContext` / `LeftDcContext` per-plane
context arrays (passed as `0` clean-state ctx); (e) `predict_intra` /
`predict_palette` / `predict_chroma_from_luma` per-TU intra-prediction
kernels (only DC_PRED has a standalone leaf via r180's
`predict_intra_dc_pred`); (f) §5.11.35 `reconstruct(...)` — the §7.6
inverse-transform kernel itself (the headline next-arc target); (g)
§5.11.35 `LoopfilterTxSizes[]` / `BlockDecoded[]` per-TU stamps; (h)
§7.13 `Dequant[][]` per-`(plane, pos)` dequantization. Each split is
plumbed through the dispatcher's argument list so the next-arc patch
that wires one can do so without re-touching the §5.11.34 driver
itself.

New public surface: `PartitionWalker::residual(...)`, `ResidualReadout`,
`ResidualTuTask`, `get_tx_size(plane, tx_size, mi_size, subsampling_x,
subsampling_y)` per §5.11.37. Three new `Error` variants
(`ResidualReconstructUnsupported` / `ResidualTransformTreeUnsupported`
/ `ResidualCoefficientsTxSizeUnsupported`) — only the first is reachable
from `decode_block_syntax` today; the other two are caller-bug guards
on direct `residual()` invocations.

13 new tests (755 → 768): three on `get_tx_size` (plane-0 pass-through
across all 19 sizes, 4:4:4 chroma residual = luma residual, 4:2:0
BLOCK_64X64 / BLOCK_128X128 size clamp, caller-bug guards); seven on
`residual` (skip-arm task-only emission + zero bits read, no-skip
all_zero per-TU short-circuit completes without `reconstruct` stub,
lossless `TX_4X4` forcing per plane, monochrome luma-only emission,
BLOCK_128X128 4-chunk emission with `BLOCK_64X64` miSizeChunk, six
caller-bug guards, inter-luma `transform_tree` 4-way recursion to
`TX_4X4` leaves); two on `ResidualReadout` Clone/Debug/Eq smoke and
on the `decode_block_syntax` walker reaching the new
`ResidualReconstructUnsupported` short-circuit. The existing
`decode_block_syntax` integration tests are updated to expect
`Error::ResidualReconstructUnsupported`. `decode_av1` / `encode_av1`
continue to return `Error::NotImplemented`.

Round 180 lifts **§5.11.30 / §5.11.33 `compute_prediction()`** — the
other major §5.11.5 walker stub (alongside r179's §5.11.39 inside
§5.11.34 `residual()`). The §5.11.33 per-plane prediction-task
dispatcher now runs to completion, threading through `for plane = 0..1
+ HasChroma * 2` with the spec's §5.11.38 `get_plane_residual_size` +
§5.11.33 `IsInterIntra` / `is_inter` / `someUseIntra` gates honoured
verbatim. Exposed as `PartitionWalker::compute_prediction(...)` and
surfaced through a new `ComputePredictionReadout { is_inter_intra,
is_inter, num_planes_visited, tasks }` aggregate carrying one
`PlanePredictionTask { plane, start_x, start_y, log2_w, log2_h, mode,
have_left, have_above }` per visited plane. Wired into
`decode_block_syntax`, lifting `Error::DecodeBlockComputePredictionUnsupported`
past the §5.11.30 site; the walker now reaches
`Error::DecodeBlockResidualUnsupported` (§5.11.34, the outer dispatch
around r179's already-landed §5.11.39 inner body).

Standalone leaf landed alongside: `predict_intra_dc_pred(have_left,
have_above, log2_w, log2_h, w, h, bit_depth, above_row, left_col,
pred)` per **§7.11.2.5 DC intra prediction process** (av1-spec
p.248-249). The §7.11.2.5 four-arm dispatch on `(haveLeft, haveAbove)`
is transcribed verbatim: union average with `(w+h)/2` rounding term
(arm 1); left-only `Clip1((sum + h/2) >> log2H)` (arm 2); above-only
`Clip1((sum + w/2) >> log2W)` (arm 3); `1 << (BitDepth - 1)` mid-grey
fallback (arm 4). Operates on caller-supplied `u16` neighbour slices +
flat row-major output buffer — no `CurrFrame` allocation by the
walker (the §5.11.5 syntax walker doesn't yet track sample buffers).
This is the first §7.11.2.x sample-generation leaf to land; the
remaining 12 intra modes (V_PRED / H_PRED / SMOOTH variants /
PAETH_PRED / D45..D203_PRED directional, plus CFL chroma) are
next-arc targets.

Supporting infra: `SUBSAMPLED_SIZE[ BLOCK_SIZES ][ 2 ][ 2 ]`
transcribed from av1-spec p.88 (`Subsampled_Size` table — the
§5.11.38 per-plane chroma-down-shift lookup, with the spec's
`BLOCK_INVALID` sentinel for `(MiSize, subsampling_{x,y})` combos
the §6.4.1 conformance gates forbid). `get_plane_residual_size(
subsize, plane, subsampling_x, subsampling_y) -> Option<usize>`
returns `Some(BLOCK_*)` for valid configurations, `None` for the
§3-enumeration sentinel. Three new `Error` variants
(`ComputePredictionInterUnsupported` / `ComputePredictionInterIntraUnsupported`
/ `ComputePredictionIntraModeUnsupported`) surface the §5.11.33
inner-arm next-arc boundaries one-to-one — the §7.9 inter sample
generation (sub-pel interpolation + reference frame sampling), the
§7.11.5 interintra blend, and the §7.11.2.2 / .3 / .4 / .6 / .7+
non-DC intra modes respectively. Each is reachable only when a
caller drives `compute_prediction` directly with the matching
input; the §5.11.5-walker path always lands on the
`is_inter == 0`, `y_mode == DC_PRED`, no-interintra arm and
succeeds.

18 new tests (737 → 755): seven on `compute_prediction` (3-plane
intra DC, luma-only one-task, `is_inter` stub, inter-intra stub,
non-DC_PRED stub, four caller-bug guards, readout Clone/Debug
smoke); eight on `predict_intra_dc_pred` (4-bit / 10-bit / 12-bit
no-neighbour fallback, left-only average, above-only average,
union average, six caller-bug guards combined into one test);
three table / helper sanity tests (`SUBSAMPLED_SIZE` shape
invariant, plane-0 pass-through across all 22 sizes,
`Clip1`-bit-depth saturation at 8 / 10 / 12 bit). The
`decode_block_syntax` integration tests are updated to expect
`Error::DecodeBlockResidualUnsupported` (the walker now advances
past §5.11.30). `decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

**Split off explicitly for the next arc:** (a) the §7.9
`predict_inter` body (motion compensation, sub-pixel interpolation,
reference-frame sampling, warping); (b) the remaining 12 §7.11.2.x
intra sample-generation modes (V_PRED, H_PRED, SMOOTH_PRED /
SMOOTH_V_PRED / SMOOTH_H_PRED, PAETH_PRED, the eight directional
D45..D203_PRED modes); (c) the §7.11.5 inter+intra blend; (d) the
§5.11.34 outer dispatch (the `widthChunks` / `heightChunks` /
`transform_tree` / `transform_block` recursion that drives the
already-landed r179 §5.11.39 inner coefficient reader and r180's
§7.11.2.5 leaf against a real `CurrFrame` buffer). The walker
needs a per-plane sample buffer to feed `predict_intra` /
`predict_inter` / `reconstruct` with actual `CurrFrame[plane][y][x]`
data — landing one is the natural next arc.

Round 179 lands the **§5.11.39 `coeffs( plane, startX, startY, txSz )`
per-TU coefficient reader** — the first body of the §5.11.34
`residual()` cascade, the gate to the entire transform / inverse-
transform / reconstruction pipeline. Exposed as
`PartitionWalker::coefficients(decoder, cdfs, plane, is_inter,
tx_size, tx_class, txb_skip_ctx, dc_sign_ctx, scan, quant)` and
surfaced through the new `CoefficientsReadout { all_zero, eob,
cul_level, dc_category }` aggregate. This reader is callable directly
today; the §5.11.34 outer dispatch (the `widthChunks` / `heightChunks`
/ `transform_tree` / `transform_block` recursion) and the §5.11.30
`compute_prediction()` walker-site that gates it remain
subsequent-arc targets — the §5.11.5 walker
([`PartitionWalker::decode_block_syntax`]) still short-circuits at
`Error::DecodeBlockComputePredictionUnsupported` upstream of the new
reader.

The §5.11.39 body composes seven pieces, all wired:

* **Line 3-7 derived sizes.** `txSzCtx` from `Tx_Size_Sqr` (derived
  inline from `Min(Tx_Width, Tx_Height)`'s `trailing_zeros`) +
  `Tx_Size_Sqr_Up` (the existing `TX_SIZE_SQR_UP` table); `ptype =
  plane > 0`; `segEob` honouring the `TX_16X64 || TX_64X16 ⇒ 512`
  special case and the `Min(1024, tx_w * tx_h)` cap.
* **Line 8-9 Quant pre-loop.** Zero-fills `Quant[ 0..segEob ]` in the
  caller's output buffer.
* **Line 13 `all_zero` S().** Against `TileTxbSkipCdf[ txSzCtx ][
  ctx ]` (caller-supplied ctx). On the `all_zero == 1` short-circuit
  returns `CoefficientsReadout { all_zero: true, eob: 0, cul_level:
  0, dc_category: 0 }`. The §5.11.40 luma `TxTypes[ y4 + j ][ x4 + i
  ] = DCT_DCT` stamp on the short-circuit arm is deferred to the
  §5.11.34 caller (the walker's `TxTypes` grid is not yet tracked).
* **Lines 19-37 EOB position read.** `eobMultisize = Min(
  Tx_Width_Log2, 5 ) + Min( Tx_Height_Log2, 5 ) - 4` selects one of
  the seven `eob_pt_{16, 32, 64, 128, 256, 512, 1024}` S() reads
  against `TileEobPt*Cdf[ ptype ][ ctx ]` (with `ctx = 0` for
  `TX_CLASS_2D`, `ctx = 1` for HORIZ / VERT; the 512 / 1024 tables
  have no ctx axis per §8.3.2). `eobPt = sym + 1`.
* **Lines 39-55 `eob` derivation.** `eob = (eobPt < 2) ? eobPt : ((1
  << (eobPt - 2)) + 1)`; `eobShift = Max(-1, eobPt - 3)`. When
  `eobShift >= 0` reads `eob_extra` S() against `TileEobExtraCdf[
  txSzCtx ][ ptype ][ eobPt - 3 ]` plus the `eob_extra_bit` L(1)
  loop adding `1 << eobShift` per set bit.
* **Lines 56-71 reverse-scan base levels.** Walks `c = eob-1 → 0`:
  at `c == eob-1` reads `coeff_base_eob` S() against
  `TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][ ctx ]` with ctx from the
  existing `get_coeff_base_eob_ctx(quant, tx_size, tx_class, pos,
  c)` helper; otherwise reads `coeff_base` S() against
  `TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]` with ctx from the
  existing `get_coeff_base_ctx(quant, tx_size, tx_class, pos, c,
  false)` helper. `level > NUM_BASE_LEVELS = 2` triggers the
  `coeff_br` chain — up to `COEFF_BASE_RANGE / (BR_CDF_SIZE - 1) =
  4` S() reads against `TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][
  ptype ][ ctx ]` with ctx from `get_br_ctx`; chain terminates on the
  first symbol `< BR_CDF_SIZE - 1 = 3`. Writes `Quant[ pos ] =
  level` (positive magnitude only at this stage).
* **Lines 73-100 forward-scan signs + golomb.** Walks `c = 0..eob`:
  reads `dc_sign` S() at `c == 0` (against `TileDcSignCdf[ ptype ][
  ctx ]` with caller-supplied ctx), otherwise `sign_bit` L(1).
  `Quant[ pos ] > NUM_BASE_LEVELS + COEFF_BASE_RANGE = 14` triggers
  the §5.11.39 golomb chain — `golomb_length_bit` L(1) do-while loop
  for the magnitude exponent, then `golomb_data_bit` L(1) loop for
  the data bits; `Quant[ pos ] = x + COEFF_BASE_RANGE +
  NUM_BASE_LEVELS`. Updates `dc_category` on the `pos == 0 &&
  Quant[ pos ] > 0` arm (`1` for negative DC, `2` for positive),
  applies the `& 0xFFFFF` 20-bit clip and the `culLevel +=
  Quant[ pos ]` accumulation, then sign-applies `Quant[ pos ] =
  -Quant[ pos ]` when needed. Closes with the `culLevel = Min(63,
  culLevel)` clamp.

Caller-supplied state surfaced through the function signature:
`tx_size` (TX_SIZES_ALL ordinal), `tx_class` (the §8.3.2
`get_tx_class(txType)` result, one of TX_CLASS_2D / TX_CLASS_HORIZ /
TX_CLASS_VERT — the §5.11.40 `compute_tx_type` derivation belongs to
the caller), `plane` / `is_inter`, `txb_skip_ctx` (`all_zero` ctx;
the §8.3.2 derivation needs `AboveLevelContext` / `LeftLevelContext`
/ `AboveDcContext` / `LeftDcContext` neighbour state that the
walker does not yet track), `dc_sign_ctx` (same neighbour-state
dependency), `scan` (caller-derived from `txSz` + `txType` per §7.5
`get_scan` — also deferred), and a `quant: &mut [i32]` output buffer
sized `>= tx_w * tx_h`. Caller-bug guards return
`PartitionWalkOutOfRange` for each out-of-range argument
(`tx_size >= TX_SIZES_ALL`, `tx_class > TX_CLASS_VERT`, `plane > 2`,
`is_inter > 1`, `txb_skip_ctx >= TXB_SKIP_CONTEXTS`,
`dc_sign_ctx >= DC_SIGN_CONTEXTS`, undersized `scan` / `quant`).

The §5.11.39 `Dequant[][]` step (true §7.13 dequantization through
the per-plane qmatrix) is **out of scope for this round**; the
reader produces the §5.11.39 raw signed `Quant[]` array — the same
quantity that feeds the §7.13 dequant — and leaves the
dequantization itself for a subsequent arc. The §5.11.34
`AboveLevelContext` / `LeftLevelContext` / `AboveDcContext` /
`LeftDcContext` stamping at the post-§5.11.39 line 104-115 site is
also deferred to the round that adds those context arrays to the
walker (the readout already carries `cul_level` and `dc_category`
for the caller to apply).

6 new unit tests cover: the gate-closed `all_zero == 1`
short-circuit (no symbol / literal bits past `txb_skip`, returned
readout zeroed, output buffer zero-filled); a gate-open smoke test
confirming the cascade produces a non-`all_zero` readout with
`eob ∈ 1..=segEob`, `cul_level ≤ 63`, `dc_category ∈ {0, 1, 2}`,
and decoder position advanced past at least one symbol bit; a CDF
adaptation cross-check confirming the §8.3 `txb_skip` counter slot
increments on a non-`disable_cdf_update` run (proves the reader is
indexing the right CDF row); the seven caller-bug guards all
returning `PartitionWalkOutOfRange`; a `TX_16X64` `segEob = 512`
boundary check (511-element scan rejected, 512-element accepted);
and a square-tx-size sweep across all five `TX_NxN` ordinals
confirming the internal `txSzCtx = (Tx_Size_Sqr + Tx_Size_Sqr_Up +
1) >> 1` derivation stays within `0..TX_SIZES` for every square
size. Test count: 731 → 737 (+6). `decode_av1` / `encode_av1` still
return `Error::NotImplemented`.

Round 177 lands the **§5.11.29 `read_compound_type` reader** —
wiring the inter-inter compound-blending controls into the §5.11.23
inter cascade. The §5.11.23 dispatcher now runs the full §5.11.23
body through line 36 (`read_compound_type( isCompound )`); the
§5.11.18 dispatcher continues to surface `InterBlockModeInfoUnsupported`
for the still-pending `read_interpolation_filter` post-cascade.

The §5.11.29 body composes five paths: (1) the `skip_mode == 1`
short-circuit (`compound_type = COMPOUND_AVERAGE`, defaults
`comp_group_idx = 0` / `compound_idx = 1`, no bits); (2) the
`!isCompound` else-arm (the §5.11.28 outcome alone selects the type
— `interintra && wedge_interintra ? COMPOUND_WEDGE : COMPOUND_INTRA`
or `COMPOUND_AVERAGE`, no bits); (3) the `isCompound &&
enable_masked_compound` arm (`comp_group_idx` S() against
`TileCompGroupIdxCdf[ ctx ]` with ctx from the §8.3.2 paragraph
combining `CompGroupIdxs` neighbour values, `AvailU` / `AvailL`,
`AboveSingle` / `LeftSingle`, and the `AboveRefFrame[0] / LeftRefFrame[0]
== ALTREF_FRAME` tests, clamped at `Min(5, ctx)`); (4) the
`comp_group_idx == 0 && enable_jnt_comp` sub-arm (`compound_idx` S()
against `TileCompoundIdxCdf[ ctx ]` with ctx seeded by `dist_equal
? 3 : 0` and incremented by the same neighbour walk, then
`compound_type = compound_idx ? COMPOUND_AVERAGE : COMPOUND_DISTANCE`);
(5) the `comp_group_idx == 1` sub-arm (`n = Wedge_Bits[ MiSize ];
n == 0 ? COMPOUND_DIFFWTD : compound_type S() against
TileCompoundTypeCdf[ MiSize ]`). The wedge sub-branch reads
`wedge_index` S() against `TileWedgeIndexCdf[ MiSize ]` (the
16-symbol element shared with the §5.11.28 inter-intra wedge arm)
plus `wedge_sign` L(1); the diffwtd sub-branch reads `mask_type`
L(1) selecting `UNIFORM_45` (0) or `UNIFORM_45_INV` (1) per §6.10.24.

The walker grows two grids — `CompGroupIdxs[r][c]` (pre-fill `0`)
and `CompoundIdxs[r][c]` (pre-fill `1`, matching the §5.11.29
default initialiser) — stamped over the bh4 * bw4 footprint on
every block decode so the next block's §8.3.2 ctx walks observe the
values. `comp_group_idxs()` / `compound_idxs()` accessors surface
both grids for tests / external consumers.

New §3 `WEDGE_BITS` table (`[u8; BLOCK_SIZES]`, av1-spec p.470
verbatim — non-zero only on the `BLOCK_8X8..=BLOCK_32X32` band plus
the `BLOCK_8X16` / `BLOCK_16X8` 2:1 rectangles, matching the §9.4
`Default_Wedge_Index_Cdf` non-placeholder rows) + `wedge_bits()`
accessor. New §6.10.24 named constants land: `COMPOUND_WEDGE = 0`,
`COMPOUND_DIFFWTD = 1`, `COMPOUND_AVERAGE = 2`, `COMPOUND_INTRA = 3`,
`COMPOUND_DISTANCE = 4` (the spec note clarifies the last three are
inferred from other syntax elements rather than coded directly via
the §8.3.2 `compound_type` S()); plus the two `mask_type` ordinals
`UNIFORM_45` / `UNIFORM_45_INV`. New §8.3.2 ctx helpers
`comp_group_idx_ctx` / `compound_idx_ctx` accept the neighbour scalar
inputs (closed forms covered by exhaustive small-input audits).

New `CompoundTypeReadout` aggregate carries `comp_group_idx` (u8) +
`compound_idx` (u8) + `compound_type` (u8) plus three `Option<u8>`
companions (`wedge_index` / `wedge_sign` Some on `COMPOUND_WEDGE`;
`mask_type` Some on `COMPOUND_DIFFWTD`). `DecodedInterBlockModeInfo`
gains a `compound_type: CompoundTypeReadout` field.
`decode_inter_block_mode_info` and `decode_inter_frame_mode_info`
gain three new caller-supplied scalars: `enable_masked_compound: bool`
(§5.5.2 sequence-header bit), `enable_jnt_comp: bool` (§5.5.2),
and `dist_equal: bool` (the `Abs(get_relative_dist(OrderHints[
RefFrame[0]], OrderHint)) == Abs(get_relative_dist(OrderHints[
RefFrame[1]], OrderHint))` outcome the caller computes from
frame-header state).

18 new unit tests cover: the `WEDGE_BITS` table verbatim cross-check
+ `wedge_bits()` helper out-of-range fallback; the §6.10.24
ordinal identities (compound-type + mask-type); the §8.3.2
`comp_group_idx_ctx` formula across every closed-form arm (closed-
both, above-only !Single, above-only Single+ALTREF, both-singles
ALTREF clamp at 5, mixed arms); the §8.3.2 `compound_idx_ctx`
formula across the dist_equal seed + every neighbour arm + a
2^7 = 128-trial exhaustive bound-check; six per-arm reader tests
exercising the `skip_mode` short-circuit (no bits), the three
`!isCompound` else-arms (`interintra=0`, `interintra=1 +
wedge_interintra=0`, `interintra=1 + wedge_interintra=1`, no bits
each), the `isCompound && !enable_masked_compound && !enable_jnt_comp`
no-bits arm, the `enable_masked_compound + comp_group_idx S()⇒0 +
!enable_jnt_comp` one-symbol arm (witnessed by CDF row adaptation),
the wedge-sub-branch path (forced `comp_group_idx⇒1` +
`compound_type⇒COMPOUND_WEDGE`; checks `wedge_index` + `wedge_sign`
both Some + the wedge_index row CDF adaptation), the
diffwtd-sub-branch path (forced `comp_group_idx⇒1` +
`compound_type⇒COMPOUND_DIFFWTD`; checks `mask_type` Some + the
wedge_index row stayed untouched), the `n == 0 ⇒ COMPOUND_DIFFWTD`
shortcut on `BLOCK_4X4` (no `compound_type` S() read; CDF row
unchanged), the `compound_idx S()⇒1 ⇒ COMPOUND_AVERAGE` and
`compound_idx S()⇒0 ⇒ COMPOUND_DISTANCE` arms, the `dist_equal=true`
ctx-3 seed witness, the `mi_size = BLOCK_SIZES` caller-bug defensive
fallback (collapses to the `wedge_bits == 0` shortcut without
hitting `wedge_index_cdf`); and an end-to-end §5.11.23 cascade test
through the dispatcher witnessing the `CompoundTypeReadout` on the
single-pred + closed-§5.11.28-gate path with the walker
`comp_group_idxs` / `compound_idxs` grids stamped. Test count: 703
→ 721 (+18). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 176 lands the **§5.11.28 `read_interintra_mode` reader** —
wiring the inter-intra blending triple into the §5.11.23 inter
cascade. The §5.11.23 dispatcher now runs the full §5.11.23 body
through line 35 (`read_interintra_mode( isCompound )`); the §5.11.18
dispatcher continues to surface `InterBlockModeInfoUnsupported` for
the still-pending §5.11.29 `read_compound_type` /
`read_interpolation_filter` post-cascade.

The §5.11.28 body composes the outer gate (`skip_mode == 0 &&
enable_interintra_compound && !isCompound && BLOCK_8X8 <= MiSize <=
BLOCK_32X32`) and the inner four-symbol arm: `interintra` S() against
`TileInterIntraCdf[ Size_Group[ MiSize ] - 1 ]`; on
`interintra == 1` then `interintra_mode` S() against
`TileInterIntraModeCdf[ ctx ]` (one of the four §6.10.27 `II_*`
modes — `II_DC_PRED` / `II_V_PRED` / `II_H_PRED` / `II_SMOOTH_PRED`),
`wedge_interintra` S() against `TileWedgeInterIntraCdf[ MiSize ]`,
and on `wedge_interintra == 1` then `wedge_index` S() against
`TileWedgeIndexCdf[ MiSize ]` (the 16-symbol element shared with
§5.11.29). On the closed-gate path zero bits are consumed and the
returned `InterIntraReadout` has `interintra = 0` with the per-symbol
`Option` fields at `None`.

Inner-arm side-effect: the spec sets `RefFrame[ 1 ] = INTRA_FRAME` on
the inner arm; the §5.11.23 dispatcher restamps the walker's slot-1
`RefFrames[..][..][1]` grid over the bh4 * bw4 block footprint so
subsequent neighbour walks observe the override. The §5.11.27 readers
covered in r175 consult slot 0 only, so the slot-1 override does not
perturb r175 behaviour; the §5.11.29 reader (next-arc) will consult
slot 1 explicitly via the §8.3.2 `comp_group_idx` ctx. The
imperative `AngleDeltaY = AngleDeltaUV = use_filter_intra = 0` writes
the spec spells out are inter-block scalars not currently tracked.

Four new §6.10.27 named constants land: `II_DC_PRED = 0`,
`II_V_PRED = 1`, `II_H_PRED = 2`, `II_SMOOTH_PRED = 3` — spanning the
`INTERINTRA_MODES = 4` axis. New `InterIntraReadout` aggregate carries
`interintra: u8` plus three `Option<u8>` companions; the inner-arm
arms populate `interintra_mode` + `wedge_interintra` always and
`wedge_index` only when `wedge_interintra == 1`.
`DecodedInterBlockModeInfo` gains an `interintra: InterIntraReadout`
field. `decode_inter_block_mode_info` and
`decode_inter_frame_mode_info` gain a new
`enable_interintra_compound: bool` parameter — the §5.5.2 sequence-
header bit caller-supplied.

10 new unit tests cover: the four §6.10.27 II_* ordinal identities;
five outer-gate-closed paths (`skip_mode = 1`, !enable_interintra_-
compound, `is_compound = true`, `MiSize < BLOCK_8X8` via `BLOCK_4X4`,
`MiSize > BLOCK_32X32` via `BLOCK_64X64`) each verified to read zero
S() bits; the `MiSize = BLOCK_SIZES` defensive fallback collapsing to
the closed-gate path; the gate-open + `interintra = 0` arm
(witnessed by `inter_intra` row CDF adaptation alongside an untouched
`inter_intra_mode` row); the four-symbol path reachability
(100-trial property test biased toward the inner arm, witnessing the
`wedge_index` row CDF adaptation); and two end-to-end §5.11.23
cascade tests — `enable_interintra_compound = false` short-circuiting
through the dispatcher with `interintra = 0` + every Option `None`,
and `enable_interintra_compound = true` witnessing the
`inter_intra` row adaptation through the dispatcher with the
conditional slot-1 grid stamp asserted on the inner-arm trials. Test
count: 691 → 703 (+12). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 173 lands the **§5.11.23 post-find_mv_stack reader cascade** —
wiring `find_mv_stack` into the `decode_inter_block_mode_info`
dispatcher so the inter arm of §5.11.18 runs end-to-end through every
bit-consuming leaf in §5.11.23 lines 1-32. The §5.11.23
`Error::FindMvStackUnsupported` stub is lifted; the reader now
short-circuits one step later at the new
`Error::AssignMvUnsupported` (§5.11.31 `assign_mv` is the next-arc
target — its body composes `read_mv` / `read_mv_component` which is
the large §5.11.32 syntax tree).

The cascade implements the four-arm YMode dispatch in spec order
(av1-spec p.74). Arm 1 (`skip_mode == 1`) forces
`YMode = NEAREST_NEARESTMV` with no S() read. Arm 2
(`seg_feature_active(SEG_LVL_SKIP|GLOBALMV)`) forces `YMode =
GLOBALMV` with no S() read. Arm 3 (compound, `RefFrame[1] >
INTRA_FRAME`) reads one `S()` against `TileCompoundModeCdf[ctx]`
where `ctx = Compound_Mode_Ctx_Map[RefMvContext >> 1][Min(NewMvContext,
COMP_NEWMV_CTXS - 1)]` per §8.3.2, then `YMode = NEAREST_NEARESTMV +
compound_mode`. Arm 4 (single-pred fall-through) walks the
`new_mv` ⇒ `zero_mv` ⇒ `ref_mv` ladder: each is one S() against the
matching `TileNewMvCdf[NewMvContext]` / `TileZeroMvCdf[ZeroMvContext]` /
`TileRefMvCdf[RefMvContext]` row from §7.10.2.14, terminating in one
of NEWMV / GLOBALMV / NEARESTMV / NEARMV.

The §5.11.23 RefMvIdx + drl_mode loops then run (av1-spec p.74-75).
On `YMode ∈ {NEWMV, NEW_NEWMV}` the loop iterates `idx = 0, 1`; on
`has_nearmv(YMode)` true (= one of NEARMV / NEAR_NEARMV / NEAR_NEWMV /
NEW_NEARMV) the loop seeds `RefMvIdx = 1` and iterates `idx = 1, 2`.
Each iteration is gated on `NumMvFound > idx + 1`; when fired it
reads one `S()` against `TileDrlModeCdf[DrlCtxStack[idx]]`, breaking
to `RefMvIdx = idx` on symbol 0 or continuing with `RefMvIdx = idx +
1` on symbol 1. On fresh-walker / `NumMvFound = 0` paths the loop
runs zero iterations and `RefMvIdx = 0`.

The §7.10 `find_mv_stack` call is wired in directly via the r172
spatial-only entry. Six new caller scalars are threaded through:
`gm_type[8]` / `gm_params[8][6]` (§7.10.2.1 setup-global-mv),
`ref_frame_sign_bias[8]` (§7.10.2.13 extra-MV negation arm),
`allow_high_precision_mv` / `force_integer_mv` (§7.10.2.10 lower-
precision LSB strip vs 3-bit integer round), and `use_ref_frame_mvs`
(the §7.10.2.5 temporal-scan deferral — when true, the reader
surfaces `Error::TemporalMvScanUnsupported` without partial state
mutation). The §5.11.18 `decode_inter_frame_mode_info` dispatcher
threads the same six scalars to keep the §5.11.18 → §5.11.23 call
chain end-to-end.

`DecodedInterBlockModeInfo` gains six new fields surfaced for caller
verification: `y_mode` (one of `MODE_NEARESTMV..=MODE_NEW_NEWMV =
14..=25`); `ref_mv_idx` (in `0..=2`); and the §7.10.2 stack-summary
snapshot — `num_mv_found`, `new_mv_context`, `ref_mv_context`,
`zero_mv_context`. These are observable only when the dispatcher
returns `Ok`; the r173 method always returns `Err`
(`AssignMvUnsupported`) since the §5.11.31 body is not yet wired —
direct callers can inspect the stamped walker grids
(`ref_frames()` / `is_inters()`) for the conformant prologue state
that committed before the stub fired.

New `has_nearmv(mode)` predicate (av1-spec p.75) returns true for
NEARMV / NEAR_NEARMV / NEAR_NEWMV / NEW_NEARMV (the four NEARMV-
bearing inter Y modes). Joins the existing `has_newmv` predicate
that already covers the six NEWMV-bearing modes.

10 new unit tests cover: Arm 1 `skip_mode = 1` yielding NEAREST_NEARESTMV
with zero bits consumed past §5.11.25; Arm 2 `seg_skip_active`
yielding GLOBALMV with zero bits consumed; Arm 3 compound case with
rigged `comp_mode` / `comp_ref_type` / `uni_comp_ref` cascade
producing `RefFrame = [BWDREF, ALTREF]` and the `compound_mode` S()
consuming bits; Arm 4 single-pred three terminal outcomes
(`new_mv = 0` ⇒ NEWMV; `new_mv = 1, zero_mv = 0` ⇒ GLOBALMV;
`new_mv = 1, zero_mv = 1, ref_mv = 0` ⇒ NEARESTMV); drl_mode loop
short-circuit on `NumMvFound = 0` (CDF counter stays 0 ⇒ no read
fired); `has_nearmv` truth table over the entire inter Y-mode
enumeration; caller-bug guards unchanged with the new r173 args;
and the `use_ref_frame_mvs == true` arm surfacing
`TemporalMvScanUnsupported` ahead of the cascade. Test count: 651 →
661 (+10). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 172 lands the **§7.10 `find_mv_stack` spatial-only path** —
the motion-vector-stack derivation reached from the §5.11.23
`inter_block_mode_info()` body once the §5.11.25 `read_ref_frames`
prologue commits `RefFrame[0..2]`. The new
`PartitionWalker::find_mv_stack(...)` method composes every
sub-process from §7.10.2.1 setup-global-mv through §7.10.2.14
context + clamping, on the `use_ref_frame_mvs == false` path.

The §7.10.2 driver runs steps 1-16 + 18-36 in spec order:
`NumMvFound = 0`, `NewMvCount = 0`, setup-global-mv for slot 0 (and
slot 1 if compound), then the inner-ring scans
(`scan_row(-1)` / `scan_col(-1)` / `scan_point(-1, bw4)` when
`Max(bw4, bh4) ≤ 16`), the §7.10.2 step 15 `WeightStack[0..numNearest]
+= REF_CAT_LEVEL = 640` close-match bonus, the outer-ring scans
(`scan_point(-1, -1)` / `scan_row(-3)` / `scan_col(-3)` /
`scan_row(-5)` if `bh4 > 1` / `scan_col(-5)` if `bw4 > 1`), the
§7.10.2.11 stable descending-by-weight sort over both segments
(`[0..numNearest)` and `[numNearest..NumMvFound)`), the §7.10.2.12
extra-search when `NumMvFound < 2` (two-pass partial-match scan over
the row above + column to the left, then a `combinedMvs`
fill from `RefIdMvs` + `RefDiffMvs` + `GlobalMvs` for compound, or
direct `RefStackMv` extension for single — though single-pred does
NOT increment `NumMvFound` per the §7.10.2.12 note), and the
§7.10.2.14 context + clamping
(`DrlCtxStack[idx]` derivation off the `WeightStack[idx] >=
REF_CAT_LEVEL` / `WeightStack[idx+1] < REF_CAT_LEVEL` partition,
per-list `clamp_mv_row` / `clamp_mv_col` with
`border = MV_BORDER + (bh * 8) | (bw * 8)`, and the
`NewMvContext` / `RefMvContext` three-arm ladder over `CloseMatches`).

The §7.10.2.2 scan_row applies the `Abs(deltaRow) > 1` neighbour
adjustment (`deltaRow += MiRow & 1; deltaCol = 1 - (MiCol & 1)`), the
`useStep16 = bw4 >= 16` 4-step stride, and the per-neighbour
weighted advance (`len = Min(bw4, Num_4x4_Blocks_Wide[MiSizes[mvRow]
[mvCol]])`, then `Max(2, len)` outer-ring + `Max(4, len)` step16
refinements). §7.10.2.3 scan_col is the vertical mirror. §7.10.2.4
scan_point gates on `RefFrames[mvRow][mvCol][0] != INTRA_FRAME = 0`
(matching the spec's "has been written" check; the walker's pre-fill
`[INTRA_FRAME, NONE]` makes the gate the spec-correct identity).

The §7.10.2.7 add_ref_mv_candidate gate
(`IsInters[mvRow][mvCol] == 1`) then dispatches to §7.10.2.8
search-stack (single-pred, for each `candList` where
`RefFrames[mvRow][mvCol][candList] == RefFrame[0]`) or §7.10.2.9
compound-search-stack (compound, when both
`RefFrames[mvRow][mvCol][0..2]` match `RefFrame[0..2]`). The
§7.10.2.10 lower-precision pass (LSB-strip when
`allow_high_precision_mv == 0`, `(|a| + 3) >> 3 << 3` integer round
when `force_integer_mv == 1`) is applied to every candidate MV
before dedupe. The §7.10.2.8 `(candMode in {GLOBALMV,
GLOBAL_GLOBALMV}) && GmType[RefFrame[0]] > TRANSLATION && large == 1`
global-MV substitution arm is a no-op refinement on the conformant
decode path — §5.11.31 `assign_mv` already stamps
`Mvs[..][refList] = GlobalMvs[refList]` for those modes, so the
"use Mvs[..]" fall-through produces the same bytes.

The §7.10.2.5 temporal scan + §7.10.2.6 temporal sample sub-
processes (step 17 of the §7.10.2 driver) are deferred — when the
caller passes `use_ref_frame_mvs == true`, the new method returns
`Error::TemporalMvScanUnsupported` without partial state mutation.
The temporal scan needs a `MotionFieldMvs[ref][y8][x8]` grid that
§7.9 `motion_field_estimation` is responsible for populating from
the §5.9.20 `RefFrameSignBias` chain — that scaffolding lands with
the next arc.

The walker gains a new `Mvs[r][c][list][comp]` grid stored as a flat
`Vec<i16>` with four `i16` slots per `(row, col)` cell (two lists ×
two components). Pre-fill is zero; the §7.10.2.7
`IsInters[mvRow][mvCol] == 0 ⇒ return` gate ensures the pre-fill
is unobservable on the conformant path until §5.11.31 `assign_mv`
(next-arc) writes it. New `mvs()` accessor exposes the grid.

New aggregate `FindMvStackResult` surfaces every §7.10.2 output:
`num_mv_found`, `new_mv_count`, `ref_stack_mv[idx][list][comp]`,
`weight_stack[idx]`, `global_mvs[list]`, `new_mv_context`,
`ref_mv_context`, `zero_mv_context`, `drl_ctx_stack[idx]` plus
the §7.10.2 step 12-14 snapshot values `close_matches`,
`total_matches`, `num_nearest`, `num_new`.

20 new §3 / §6.10.22 / §7.10 constants land:
`MAX_REF_MV_STACK_SIZE = 8`, `REF_CAT_LEVEL = 640`,
`MV_BORDER = 128`, `WARPEDMODEL_PREC_BITS = 16` (cdf-local twin);
GM type discriminants (`IDENTITY`/`TRANSLATION`/`ROTZOOM`/`AFFINE`
= 0..=3); the §6.10.22 inter Y mode ordinal table
(`MODE_NEARESTMV = 14`..`MODE_NEW_NEWMV = 25`) plus the
`has_newmv(mode)` predicate.

The §5.11.23 `decode_inter_block_mode_info` reader continues to
return `Error::FindMvStackUnsupported` because the post-MV-stack
reader cascade (`compound_mode` / `new_mv` / `zero_mv` / `ref_mv` /
`drl_mode` / `assign_mv` / `read_motion_mode` /
`read_interintra_mode` / `read_compound_type` /
`read_interpolation_filter`) remains pending. The wiring of
`find_mv_stack` into the dispatcher follows once those readers land.

24 new unit tests cover: §7.10.2.10 lower-precision matrix
(high-precision passthrough, LSB strip, `force_integer_mv` 3-bit
round); §5.11.53 / §5.11.54 clamp bracket derivations; §7.10.2.11
stable sort (descending by weight + equal-weight stability);
§7.10.2.1 setup-global-mv (INTRA_FRAME ⇒ 0, IDENTITY ⇒ 0,
TRANSLATION shift); empty-walker single-pred (NumMvFound = 0) and
compound (NumMvFound = 2 global-fill) outcomes; caller-bug guards;
temporal-scan deferral; one above-neighbour with NEWMV
(REF_CAT_LEVEL bonus + NewMvContext = 2, RefMvContext = 3); one
left-neighbour with NEARESTMV (numNew = 0, NewMvContext = 3);
both-neighbours case (CloseMatches = 2 ⇒ NewMvContext = 4,
RefMvContext = 5); mismatched-ref close-match exclusion +
extra-search pickup; identical-MV dedupe; single-pred extra-search
NumMvFound non-increment; `has_newmv` truth table; MV-stack
constant values; DrlCtxStack[] derivation. Test count: 627 → 651
(+24). `decode_av1` / `encode_av1` still return
`Error::NotImplemented`.

Round 171 lands the **§5.11.46 palette-entries reader**
(`palette_colors_y[]` / `palette_colors_u[]` /
`palette_colors_v[]`) and the **§5.11.49 `get_palette_cache(plane)`**
two-pointer merge of the above + left neighbour palettes —
lifting the `Error::PaletteEntriesUnsupported` stub the r169
§5.11.22 reader surfaced after `palette_size_*_minus_2`. The
§5.11.22 `PartitionWalker::decode_intra_block_mode_info` reader
now threads a `bit_depth: u8` argument (in {8, 10, 12} per
§5.5.2) and runs every leaf to completion on the conformant path.
The Y plane walks the cache-coded indices loop
(`use_palette_color_cache_y L(1)` per cache slot, copying
`PaletteCache[i]`), the first new entry as `L(BitDepth)`, the
optional `L(2) palette_num_extra_bits_y`, and the delta loop with
`palette_delta_y L(paletteBits)` + the spec's `++` increment +
`Clip1(prev + delta)` + `range = (1 << BitDepth) -
palette_colors_y[idx] - 1` paletteBits refinement, terminating in
the trailing ascending sort. The U plane mirrors Y minus the `++`
step and with `range = (1 << BitDepth) - palette_colors_u[idx]`
(no `- 1`). The V plane dispatches on
`delta_encode_palette_colors_v L(1)`: on 1, a signed-delta path
with `minBits = BitDepth - 4`, per-entry `L(paletteBits)
palette_delta_v` + optional `L(1) palette_delta_sign_bit_v` sign
flip + modular wrap into `[0, maxVal)` + `Clip1`; on 0, direct
`PaletteSizeUV`-many `L(BitDepth)` literals. The V plane is not
sorted either way per §5.11.46.

The walker grows two new grids — `PaletteSizes[plane][r][c]` and
`PaletteColors[plane][r][c][idx]`, packed flat with the plane
outermost — pre-filled at construction time to the spec's "no
palette" identity. Decoded entries are stamped over the block's
`bh4 * bw4` footprint per plane so the next §5.11.46 call's
§5.11.49 neighbour merge observes the propagated values. The
§5.11.49 `(MiRow * MI_SIZE) % 64` superblock-top gate suppresses
the above-neighbour read at 64×64 boundaries; out-of-grid reads
return 0, the spec-correct identity. New public accessors:
`palette_sizes()` and `palette_colors()` (flat slice views) plus
`get_palette_cache(plane, mi_row, mi_col, &mut [u16; 16])` (runs
the merge + dedupe and returns the entry count). Two new free
helpers cover the §4.7 mathematical primitives the §5.11.46
reader needs: `clip1_to_bit_depth(x, bd)` (§4.7 `Clip1`) and
`ceil_log2_av1(x)` (§4.7 `CeilLog2`, defined to return 0 on
`x < 2`).

New on `DecodedIntraBlockModeInfo`: three optional palette-entry
arrays — `palette_colors_y / _u / _v: Option<[u16; PALETTE_COLORS]>`
— carrying the decoded entries (Y and U sorted; V in source order
per spec). `palette_size_y` / `palette_size_uv` now report
`Some(decoded_size)` on the conformant path (previously stayed
`None` because the reader short-circuited before committing).
`Error::PaletteEntriesUnsupported` is retained for ABI stability
but is no longer constructed on the conformant code path.

12 new unit tests cover: the `bit_depth ∈ {8, 10, 12}`
caller-bug guard; an empty-cache fresh-walker read;
`clip1_to_bit_depth` / `ceil_log2_av1` truth tables at the
spec-defined boundaries; `decode_intra_block_mode_info` with a
rigged `has_palette_y == 0` path returning `Ok` with empty
palette state; `read_palette_entries_y` direct-call with
`PaletteSizeY = 2` and a zero bitstream yielding `[0, 1]` (the Y
`delta + 1` step); the U reader yielding `[0, 0]` (no `++`); the
V direct-literal arm yielding `[0, 0]`; hand-stamped palette
grids visible through `get_palette_cache` (left-neighbour-only);
the §5.11.49 above + left merge with a duplicate entry; the
§5.11.49 superblock-top boundary gate; and the
`read_palette_entries_y` cache-coded no-bit-read path via the
`get_palette_cache` accessor. Test count: 615 → 627 (+12).
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 169 lands **§5.11.22 `intra_block_mode_info()`** (av1-spec
p.73) — the per-block intra-mode syntax composite reached from the
§5.11.18 `else` arm of `if (is_inter)`. The new
`PartitionWalker::decode_intra_block_mode_info` reader composes every
§5.11.22 leaf in spec order: `RefFrame[0] = INTRA_FRAME, RefFrame[1] =
NONE` (constant `[0, -1]`); `y_mode` S() against `TileYModeCdf[ctx =
Size_Group[MiSize]]` (decoded value in `0..INTRA_MODES = 13`) + the
§5.11.5 `YModes[r + y][c + x] = YMode` grid-fill over the `bh4 * bw4`
footprint; §5.11.42 `intra_angle_info_y()` (gated on `MiSize >=
BLOCK_8X8 && is_directional(YMode)`, reads against
`TileAngleDeltaCdf[YMode - V_PRED]`, biased into
`-MAX_ANGLE_DELTA..=MAX_ANGLE_DELTA = -3..=3`); the `if (HasChroma)`
arm with `uv_mode` S() (§8.3.2 CFL-allowance branch routes through the
new `cfl_allowed_for_uv_mode` helper: lossless + post-subsampling 4×4
OR !lossless + Max(W,H) ≤ 32 → `TileUVModeCflAllowedCdf[YMode]`, else
`TileUVModeCflNotAllowedCdf[YMode]`); §5.11.45 `read_cfl_alphas()`
(gated on `UVMode == UV_CFL_PRED == 13`, reads `cfl_alpha_signs` S()
against `TileCflSignCdf` then per-axis `cfl_alpha_{u,v}` S() against
`TileCflAlphaCdf[ctx]` with sign-decomposition `signU =
(signs+1)/3, signV = (signs+1)%3` and `CflAlpha{U,V} = ±(1 + raw)` per
sign); §5.11.43 `intra_angle_info_uv()` (mirror of §5.11.42 against
`TileAngleDeltaCdf[UVMode - V_PRED]`); §5.11.46 `palette_mode_info()`
(outer gate `MiSize >= BLOCK_8X8 && Block_W ≤ 64 && Block_H ≤ 64 &&
allow_screen_content_tools`, per-plane reads `has_palette_*` S() then
on 1 reads `palette_size_*_minus_2` S() and surfaces
`Err(PaletteEntriesUnsupported)` — the §5.11.46 `palette_colors_*[]`
literal + delta reads need parser-scope `BitDepth` + `PaletteCache[]`
plumbing the walker doesn't yet thread through); §5.11.24
`filter_intra_mode_info()` (outer gate `enable_filter_intra && YMode
== DC_PRED && PaletteSizeY == 0 && Max(Block_W, Block_H) ≤ 32`, reads
`use_filter_intra` S() against `TileFilterIntraCdf[MiSize]` then on 1
reads `filter_intra_mode` S() against `TileFilterIntraModeCdf`).

Returns `DecodedIntraBlockModeInfo` carrying every decoded value with
`Option<…>` fields for the spec's "not read" arms (uv_mode /
cfl_alpha_* / angle_delta_uv / palette_size_* / use_filter_intra /
filter_intra_mode). Two new public free functions:
`is_directional(mode)` (§5.11.44, `V_PRED ≤ mode ≤ D67_PRED`) and
`cfl_allowed_for_uv_mode(lossless, mi_size, sub_x, sub_y)`
(§8.3.2 derivation). Three new public constants: `D67_PRED = 8`,
`UV_CFL_PRED = 13`, plus one new `Error` variant
`PaletteEntriesUnsupported`. 15 new unit tests cover the bounds
guards, the DC_PRED all-gates-off happy path, the HasChroma-false
short-circuit, the §5.11.42 non-directional / small-block
short-circuits, the §5.11.46 / §5.11.24 outer-gate short-circuits,
the §5.11.22 RefFrame invariant, the §5.11.5 grid-fill (DC and
non-DC), the §5.11.44 truth-table, and the §8.3.2 CFL-allowance
truth-table.

The §5.11.18 `decode_inter_frame_mode_info` dispatcher still
short-circuits at `Error::IntraBlockModeInfoUnsupported` — wiring the
new reader into the dispatcher needs additional sequence-header
arguments (`has_chroma`, `allow_screen_content_tools`,
`enable_filter_intra`, `subsampling_x`, `subsampling_y`,
`above_palette_y`, `left_palette_y`) that the §5.11.18 signature
doesn't yet thread through. Direct callers can invoke
`decode_intra_block_mode_info` with the missing arguments. Test count:
587 → 602 (+15). `decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 170 lands **§5.11.23 `inter_block_mode_info()` prologue +
§5.11.25 `read_ref_frames()`** (av1-spec p.74-77) — the per-block
inter-mode reference-frame syntax tree reached from the §5.11.18
`if (is_inter)` arm. The new
`PartitionWalker::decode_inter_block_mode_info` reader composes:
lines 1-2 (`PaletteSizeY = 0, PaletteSizeUV = 0`); §5.11.25
`read_ref_frames()` four-arm dispatch — `skip_mode` arm
(`RefFrame[0..2] = SkipModeFrame[0..2]`, no `S()`),
`seg_feature_active(SEG_LVL_REF_FRAME)` arm
(`RefFrame[0] = seg_ref_frame_data, RefFrame[1] = NONE`, no `S()`),
`seg_feature_active(SEG_LVL_SKIP | SEG_LVL_GLOBALMV)` arm
(`RefFrame[0] = LAST_FRAME, RefFrame[1] = NONE`, no `S()`), and the
default syntax-tree arm (`comp_mode` `S()` gated on
`reference_select && Min(bw4, bh4) >= 2`; COMPOUND splits into the
UNIDIR_COMP cascade `uni_comp_ref` / `uni_comp_ref_p1` /
`uni_comp_ref_p2` or the BIDIR_COMP cascade `comp_ref` /
`comp_ref_p1` / `comp_ref_p2` + `comp_bwdref` / `comp_bwdref_p1`;
SINGLE runs the `single_ref_p1..p6` cascade); line 4
(`isCompound = RefFrame[1] > INTRA_FRAME`); and the walker grid stamp
`RefFrames[r + y][c + x][0..2]` over the `bh4 * bw4` footprint.

The reader then short-circuits at the §7.10.2 `find_mv_stack(isCompound)`
entry with the new `Error::FindMvStackUnsupported`. The §7.10
motion-vector-stack derivation (scan-row, scan-col, temporal-scan,
sorting, extra-search, context-and-clamping) and every dependent
§5.11.23 reader (`compound_mode`, `new_mv` / `zero_mv` / `ref_mv`,
`drl_mode`, `assign_mv`, `read_motion_mode`, `read_interintra_mode`,
`read_compound_type`, `read_interpolation_filter`) remain
subsequent-arc targets. Every §5.11.25 `S()` read + the grid stamp
commits to state before the stub fires; the §5.11.18 dispatcher's
`if (is_inter)` arm now routes through the new reader (the
`InterBlockModeInfoUnsupported` variant becomes a defensive
caller-bug fallback no longer fired on the conformant path).

Five new §8.3.2 free functions land for the ref-frame ctx walks:
`check_backward(ref)` (`BWDREF_FRAME..=ALTREF_FRAME`),
`is_samedir_ref_pair(ref0, ref1)`, `count_refs(frame_type, …)`,
`comp_mode_ctx(…)` (the av1-spec p.366 nine-arm dispatch), and
`comp_ref_type_ctx(…)` (the av1-spec p.382 three-nested-if
dispatch).

The walker gains a `RefFrames[][][..]` grid as a flat `Vec<i8>`
with two slots per `(row, col)` cell, pre-filled with the §5.11.18
"unavailable neighbour" identity `[INTRA_FRAME = 0, NONE = -1]`.
View accessor `ref_frames()` exposes it for tests / callers; the
§5.11.18 prologue's `Left/AboveRefFrame[..]` derivations now consult
the grid so subsequent inter blocks observe the propagated values.

Eight new ref-frame ordinal constants land: `LAST2_FRAME = 2`,
`LAST3_FRAME = 3`, `GOLDEN_FRAME = 4`, `BWDREF_FRAME = 5`,
`ALTREF2_FRAME = 6` (plus the existing `INTRA_FRAME = 0`,
`LAST_FRAME = 1`, `ALTREF_FRAME = 7`), and the
`SINGLE_REFERENCE = 0`, `COMPOUND_REFERENCE = 1`,
`UNIDIR_COMP_REFERENCE = 0`, `BIDIR_COMP_REFERENCE = 1` enum
constants. New aggregate `DecodedInterBlockModeInfo` carries the
§5.11.25 output + `is_compound` derivation (observable only via the
walker grid since the reader always returns `Err`-path until §7.10
lands). The §5.11.18 dispatcher signature gains four new arguments
(`skip_mode_frame: [i32; 2]`, `seg_skip_active`, `seg_ref_frame_data`,
`reference_select`).

13 new unit tests cover the §8.3.2 ctx helpers (check_backward /
is_samedir_ref_pair / count_refs truth tables, comp_mode_ctx
nine-arm corners), the §5.11.23 caller-bug guards (out-of-range
mi_row / mi_col / sub_size / seg_ref_frame_data / skip_mode_frame),
the four §5.11.25 arm dispatch paths (skip_mode, seg_ref_frame_active,
seg_skip / seg_globalmv, default), the §5.11.25 small-block
`Min(bw4, bh4) < 2` gate, and the walker `ref_frames()` grid
propagation. Test count: 602 → 615 (+13).
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 168 lands **§5.11.17 `read_var_tx_size()`** + **§5.11.18
`inter_frame_mode_info()`** (av1-spec p.70-71) — the two missing
inter-arm composites that bound the §5.11.5 walker's inter side.

The **§5.11.17 reader** is exposed as a new
`PartitionWalker::read_var_tx_size` method (callable standalone). The
spec body is a frame-edge clip + a recursive `txfm_split` symbol read
that bottoms out at `txSz == TX_4X4` or `depth == MAX_VARTX_DEPTH`,
with each terminal-else stamping `InterTxSizes[ row + i ][ col + j ] =
txSz` over the `(h4, w4)` sub-block footprint. The §8.3.2 `txfm_split`
ctx selector inlines the spec's `get_above_tx_width` /
`get_left_tx_height` helpers against the walker's grids (`Skips[]` /
`IsInters[]` / `MiSizes[]` / `InterTxSizes[]`) — the `row == MiRow` /
`col == MiCol` arm gates the `Skips && IsInters` early return, and
the fall-through reads `Tx_Width[ InterTxSizes[ above ] ]` /
`Tx_Height[ InterTxSizes[ left ] ]`. The ctx formula
`ctx = (txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 +
above + left` already had its `txfm_split_ctx` helper from earlier
rounds; this round wires it into the live recursion. The §5.11.16
`read_block_tx_size` inter-arm now enters `read_var_tx_size` instead
of surfacing `Error::ReadVarTxSizeUnsupported` — `TxSizes[]` is
stamped with the last terminal-else's `txSz` over the full block
footprint per §5.11.5.

New free function `find_tx_size( w, h )` (§5.11.36 spec helper) —
the linear scan over `TX_SIZES_ALL` returning the first ordinal whose
`(Tx_Width, Tx_Height)` matches. Used by the §8.3.2 `txfm_split` ctx
selector's `maxTxSz = find_tx_size( size, size )` derivation.

The **§5.11.18 reader** is exposed as a new
`PartitionWalker::decode_inter_frame_mode_info` method composing every
pre-dispatch leaf in spec order: `use_intrabc = 0`; the §5.11.18
`LeftRefFrame[..]` / `AboveRefFrame[..]` / `LeftIntra` / `AboveIntra`
/ `LeftSingle` / `AboveSingle` local derivations (currently fixed at
`[INTRA_FRAME, NONE]` since the walker doesn't yet track
`RefFrames[][][..]` — the §5.11.23 readers' next-round target);
`skip = 0`; §5.11.19 `inter_segment_id(1)` via
`decode_inter_segment_id`; §5.11.10 `read_skip_mode()` via
`decode_skip_mode`; the `if (skip_mode) skip = 1 else read_skip()`
dispatch (the `skip_mode == 1` arm bypasses `decode_skip` and stamps
`Skips[][] = 1` directly per the §5.11.5 grid-fill invariant);
§5.11.19 `inter_segment_id(0)` (the post-skip arm, fired only when
`!SegIdPreSkip`); `Lossless = LosslessArray[segment_id]`; §5.11.56
`read_cdef()`; §5.11.12 `read_delta_qindex()`; §5.11.13
`read_delta_lf()`; §5.11.20 `read_is_inter()` via `decode_is_inter`.

The §5.11.18 terminal `if (is_inter) inter_block_mode_info() else
intra_block_mode_info()` dispatch short-circuits at two new `Error`
variants: `Error::InterBlockModeInfoUnsupported` (the §5.11.23
`inter_block_mode_info()` next-round target — MV stack / ref-frame
readers) and `Error::IntraBlockModeInfoUnsupported` (the §5.11.22
`intra_block_mode_info()` next-round target — per-block intra angle
/ UV mode readers). All pre-dispatch reads commit to the bitstream
/ grids before the stub fires.

New `DecodedInterFrameModeInfo` per-block aggregate carries every
§5.11.18 derived value (`mi_row` / `mi_col` / `mi_size` / `use_intrabc`
/ `avail_u` / `avail_l` / `left_ref_frame` / `above_ref_frame` /
`left_intra` / `above_intra` / `left_single` / `above_single` / `skip`
/ `skip_mode` / `segment_id` / `lossless` / `cdef_idx` /
`current_q_index` / `current_delta_lf` / `is_inter`).

The §5.11.5 `decode_block_syntax` walker is unchanged on the
`frame_is_intra = false` arm — it still short-circuits with
`Error::DecodeBlockInterFrameUnsupported` (the umbrella stub) because
the §5.11.18 reader needs additional segmentation-feature /
skip-mode-present caller state the §5.11.5 driver doesn't yet
thread through. Direct callers of `decode_inter_frame_mode_info`
get the full pre-dispatch walk and the §5.11.22 / §5.11.23
distinction.

11 new integration tests (`tests/decode_block_syntax_walker.rs`)
cover: the §5.11.17 base case (`TX_4X4`) with no S() read; the
depth cap (`MAX_VARTX_DEPTH`) with no S() read; the split path with
all-`1` `txfm_split` returning `TX_4X4` at depth 2; the frame-edge
clip; the caller-bug guards; the inter-arm `read_block_tx_size` now
enters `read_var_tx_size` and returns the input `maxTxSz` on the
no-split path with the `(4, 4)` footprint of `InterTxSizes[]`
stamped; the §5.11.18 baseline path reaches the §5.11.22 intra stub;
the §5.11.20 segment-override arm reaches the §5.11.23 inter stub;
the `skip_mode = 1` arm forces `Skips[][] = 1` + reaches §5.11.23;
the `seg_globalmv_active` arm reaches §5.11.23; the four
caller-bug guards. Plus 1 new unit test for `find_tx_size` on
square and rectangular sizes.

`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 167 lands **§5.11.16 `read_block_tx_size()`** (av1-spec p.70) —
the per-block transform-size syntax-tree read that the r166 §5.11.5
walker hits as the first stub on the intra arm. Exposed as a new
`PartitionWalker::read_block_tx_size` method (callable standalone) and
wired into `decode_block_syntax` so the walker now reaches §5.11.30
`compute_prediction()` instead.

The §5.11.16 reader transcribes the spec body one-to-one:

* `bw4` / `bh4` from `Num_4x4_Blocks_*[ MiSize ]`.
* The outer `TX_MODE_SELECT && MiSize > BLOCK_4X4 && is_inter && !skip
  && !Lossless` gate routes to §5.11.17 `read_var_tx_size` (deferred
  to the next arc — surfaces `Error::ReadVarTxSizeUnsupported`).
* The `else` arm performs §5.11.15 `read_tx_size(!skip || !is_inter)`
  inline: the `Lossless` short-circuit forces `TxSize = TX_4X4` (no
  S() consumed); otherwise `TxSize` starts at
  `Max_Tx_Size_Rect[ MiSize ]` and is further split `tx_depth` times
  via `Split_Tx_Size[]` when `MiSize > BLOCK_4X4 && allowSelect &&
  TxMode == TX_MODE_SELECT`.
* `tx_depth` is an `S()` against the §8.3.2-selected CDF. The
  selector is `TileTx{8x8,16x16,32x32,64x64}Cdf[ ctx ]` indexed by
  `maxTxDepth = Max_Tx_Depth[ MiSize ] ∈ { 1, 2, 3, 4 }`. The ctx
  derivation `ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)`
  walks the §8.3.2 neighbour ladder: when `AvailU &&
  IsInters[above]`, `aboveW = Block_Width[ MiSizes[above] ]`;
  otherwise `aboveW = Tx_Width[ InterTxSizes[above] ]` per the
  `get_above_tx_width` helper. Mirrored for `leftH`.
* The `else`-arm grid-fill loops over `(row, col) ∈ MiRow + 0..bh4,
  MiCol + 0..bw4` and stamps `InterTxSizes[ row ][ col ] = TxSize`.
  The §5.11.5 outer footer additionally stamps
  `TxSizes[ r + y ][ c + x ] = TxSize` over the same footprint —
  both stamps land in the reader.

New `PartitionWalker` grids: `tx_sizes: Vec<u8>` for the §5.11.5
`TxSizes[][]` writes and `inter_tx_sizes: Vec<u8>` for the §5.11.16
/ §5.11.17 `InterTxSizes[][]` writes. Both initialised to `TX_4X4`
(the §8.3.2 ctx-walk identity for an unavailable neighbour:
`Tx_Width[TX_4X4] = 4`). Public accessors `tx_sizes()` /
`inter_tx_sizes()` surface them after the walk.

New spec tables transcribed in `src/cdf.rs`:

* `MAX_TX_SIZE_RECT[ BLOCK_SIZES ]` — `Max_Tx_Size_Rect[ MiSize ]`
  from av1-spec p.402. The square `BLOCK_NxN → TX_NxN` identity for
  the four primary square sizes, `TX_64X64` cap for the 128×*
  blocks, and the matching rectangular `TX_*` entry for every
  rectangular `BLOCK_*`.
* `MAX_TX_DEPTH_TABLE[ BLOCK_SIZES ]` — `Max_Tx_Depth[ MiSize ]`
  from av1-spec p.69. The four-row listing
  `{ 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 3, 3, 4,
  4 }`. Named `MAX_TX_DEPTH_TABLE` to avoid shadowing the existing
  `MAX_TX_DEPTH = 2` symbol-cap constant (per the §3 listing /
  CDF row-length contract).
* `SPLIT_TX_SIZE[ TX_SIZES_ALL ]` — `Split_Tx_Size[ TxSize ]` from
  av1-spec p.404. The recursive-split table that takes a transform
  size to the size obtained by splitting it into four
  sub-transforms; `TX_4X4` is a fixed point.
* `MAX_VARTX_DEPTH = 2` — the §3 constant table value for the
  §5.11.17 recursion depth cap.

New `TX_*` rectangular ordinals (`TX_4X8` through `TX_64X16`, ordinals
5..=18 per §6.10.16), needed for `MAX_TX_SIZE_RECT` and the future
§5.11.17 reader.

`DecodedBlock` gains a `tx_size: u8` field carrying the §5.11.16
return. The struct is the no-stub-path return of the §5.11.5 walker;
it remains publicly constructible.

`decode_block_syntax` gains a `tx_mode_select: bool` parameter
(threaded from the §5.9.21 / §6.8.21 frame-header `TxMode ==
TX_MODE_SELECT` derivation), now invokes `read_block_tx_size` after
the §5.11.49 `palette_tokens()` no-op, and short-circuits with the
new `Error::DecodeBlockComputePredictionUnsupported` (the §5.11.30
target). The §5.11.16 stub variant
`Error::DecodeBlockReadBlockTxSizeUnsupported` is retained for API
stability but no longer reached from the walker.

New `Error::ReadVarTxSizeUnsupported` variant surfaces the §5.11.17
`read_var_tx_size` deferral on the inter `TX_MODE_SELECT && !skip &&
!Lossless` arm. Currently unreachable from `decode_block_syntax`
(the inter arm is stubbed upstream at
`Error::DecodeBlockInterFrameUnsupported`); reachable from direct
`read_block_tx_size` calls with the right (`is_inter = 1`, `skip = 0`,
`lossless = false`, `tx_mode_select = true`) shape.

10 new integration tests
(`tests/decode_block_syntax_walker.rs`): the standalone reader's
`Lossless` short-circuit forces `TX_4X4` with no S() consumed; the
`TX_MODE_LARGEST` arm skips the `tx_depth` read and returns
`maxRectTxSize`; the `BLOCK_4X4` arm skips the read regardless of
TxMode (`MiSize > BLOCK_4X4` is false); `TX_MODE_SELECT` with rigged
`tx_depth = 0` / `1` / `2` walks the `Split_Tx_Size` chain
`TX_16X16 → TX_8X8 → TX_4X4` for a BLOCK_16X16 block; the inter
`TX_MODE_SELECT` arm surfaces the `ReadVarTxSizeUnsupported` stub
without bitstream consumption; the inter `skip` path falls through
the `else` arm with `allowSelect = false` and yields `maxRectTxSize`;
out-of-range guards reject the three caller-bug cases; and the
integrated walker reaches `compute_prediction` after the
`read_block_tx_size` pass stamps `TxSizes[]` / `InterTxSizes[]` over
the BLOCK_16X16 footprint. Plus 5 new unit tests for `MAX_TX_SIZE_RECT`
square-block identity, `MAX_TX_DEPTH_TABLE` spec-listing match,
`SPLIT_TX_SIZE` recursive-split contract, `tx_depth_ctx` combination
table, and `MAX_VARTX_DEPTH` value.

The §5.11.5 calls that remain STUBBED:

* **§5.11.17 `read_var_tx_size()`** — the variable-transform-tree
  recursion the §5.11.16 inter-arm enters; the immediate next-round
  target for the §5.11.16 inter-arm completion (paired with §5.11.18
  inter mode-info).
* **§5.11.18 `inter_frame_mode_info()`** — the inter arm of §5.11.6
  `mode_info()`; the immediate next-round target along the intra arc.
* **§5.11.30 `compute_prediction()`** — reached after §5.11.16. The
  immediate next-round target on the intra arm.
* **§5.11.34 `residual()`** — reachable once §5.11.30 lands.

`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 166 lands the §5.11.5 **`decode_block()` syntax-walker
skeleton** (av1-spec p.63-64) — the missing dispatcher that the
§5.11.4 partition walker recurses into at every leaf. Exposed as a
new `PartitionWalker::decode_block_syntax` method that performs the
§5.11.5 prologue (`MiRow` / `MiCol` / `MiSize` / `bw4` / `bh4` /
`HasChroma` three-arm dispatch / `AvailU` / `AvailL` /
`AvailUChroma` / `AvailLChroma` with the chroma fix-up arms for
sub-sampled 1×1 edge cases) and the §5.11.6 `mode_info()` intra arm
(routed through the implemented `decode_intra_frame_mode_info_prefix`
+ `decode_use_intrabc` + `decode_intra_frame_y_mode` composition,
plus the `use_intrabc == 1` short-circuit's `YMode = DC_PRED` /
`is_inter = 1` fixed assignments). The §5.11.49 `palette_tokens()`
call is a no-op on the no-palette path reachable while
`palette_mode_info()` remains unimplemented (the spec's outer `if
( PaletteSize{Y,UV} )` guard short-circuits). After the mode-info
pass completes the walker emits a `DecodedBlockRecord` leaf (same
as `decode_partition`'s leaf emitter) and short-circuits at the
§5.11.16 `read_block_tx_size()` call with a new
`Error::DecodeBlockReadBlockTxSizeUnsupported` variant — the
immediate next-round target.

Four new `Error` stub variants surface the §5.11.5 next-round
boundaries one-to-one: `DecodeBlockInterFrameUnsupported` (§5.11.18
inter_frame_mode_info, fires on `frame_is_intra = false` with zero
bits consumed); `DecodeBlockReadBlockTxSizeUnsupported` (§5.11.16
read_block_tx_size, the immediate next stub); and the reserved
`DecodeBlockComputePredictionUnsupported` (§5.11.30) and
`DecodeBlockResidualUnsupported` (§5.11.34) variants for the rounds
that land §5.11.16 → §5.11.30 → §5.11.34 in turn.

New `DecodedBlock` per-block aggregate carries the §5.11.5 prologue
+ §5.11.7 mode-info derivations (`mi_row` / `mi_col` / `mi_size` /
`bw4` / `bh4` / `has_chroma` / `avail_u` / `avail_l` /
`avail_u_chroma` / `avail_l_chroma` + the `IntraFrameModeInfoPrefix`
fields + `use_intrabc` / `is_inter` / `y_mode` / `is_compound`).
Constructible publicly so the round that lands §5.11.16 can return
it from the no-stub path.

Partition-walker driver: a new
`PartitionWalker::decode_partition_syntax` method mirrors the
§5.11.4 recursion of `decode_partition` exactly (same partition
symbol reads, same `partition_subsize` dispatch, same edge-of-frame
fall-throughs) but routes every `decode_block( r, c, sz )` leaf
call through `decode_block_syntax` instead of the leaf-only emitter.
On a stub the recursion short-circuits with the stub variant; grid
stamps stamped before the short-circuit remain observable on the
walker's grid accessors.

10 new integration tests
(`tests/decode_block_syntax_walker.rs`): the baseline
keyframe / no-segmentation / no-screen-content path reaches the
§5.11.16 stub after consuming the §5.11.11 skip + §5.11.7
`intra_frame_y_mode` bits + emitting one leaf record with the
expected mi-grid stamps; the §5.11.5 prologue `HasChroma` three-arm
dispatch on `BLOCK_4X4` (subsampling-y / subsampling-x / fall-
through); the §5.11.6 inter-frame arm surfaces the §5.11.18 stub
with zero bits consumed and no leaf record; the §5.11.7
`SegIdPreSkip = true` arm reaches the stub after the pre-skip
`intra_segment_id` + `read_skip` pair fire in spec order;
out-of-range guard symmetry (`mi_row` / `mi_col` / `sub_size`); the
§5.11.4 partition driver routes a `BLOCK_4X4` superblock (the
`< BLOCK_8X8` short-circuit arm) through `decode_block_syntax` and
propagates the stub; the §5.11.4 `r >= MiRows` early return
short-circuits cleanly; `DecodedBlock` public-API constructibility;
`BLOCK_8X16` grid-fill footprint (bw4 = 2, bh4 = 4) at the frame
origin; `BLOCK_16X16` with `cdef_bits = 2` exercises the §5.11.56
literal-bits read path.

The §5.11.5 calls that are STUBBED with `NotImplemented`-class
errors and become the next round's targets:

* **§5.11.16 `read_block_tx_size()`** + its §5.11.15 / §5.11.17
  sub-tree — the immediate next-round target.
* **§5.11.18 `inter_frame_mode_info()`** — the inter arm of
  §5.11.6 `mode_info()`.
* **§5.11.30 `compute_prediction()`** — reachable once §5.11.16
  lands.
* **§5.11.34 `residual()`** — reachable once §5.11.30 lands.

The §5.11.7 follow-on `else`-arm elements (`intra_angle_info_y`,
`uv_mode`, `read_cfl_alphas`, `intra_angle_info_uv`,
`palette_mode_info`, `filter_intra_mode_info`) and the `use_intrabc
== 1` MV-stack / assign-mv body remain bounded leaf targets that
can be slotted into a future round before or alongside §5.11.16.
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 165 lands the §5.11.7 / §5.11.22 **`intra_frame_y_mode` syntax
element** (av1-spec p.65) as a new
`PartitionWalker::decode_intra_frame_y_mode` method — the per-block
luma intra-prediction-mode selector read on the §5.11.7
`intra_frame_mode_info()` `else` arm (`use_intrabc == 0`), immediately
after the `is_inter = 0` assignment that r164's
`decode_use_intrabc` fall-through arm produced. The spec body is
the two-line `intra_frame_y_mode S(); YMode = intra_frame_y_mode`;
the dispatcher reads a single `S()` symbol against the §8.3.2
`intra_frame_y_mode` ctx-selected CDF and stamps the result over the
block's `bw4 * bh4` footprint of a new `YModes[][]` grid. New
`PartitionWalker::y_modes: Vec<u8>` field — a `mi_rows * mi_cols`
row-major buffer covering the §6.10.4 `YModes[ r ][ c ]` grid
(av1-spec p.378 / §5.11.5). Cells are initialised to `0` (=
`DC_PRED`); the initial-zero state matches the §8.3.2 ctx walk's
"neighbour unavailable" arm (`abovemode = Intra_Mode_Context[ AvailU
? YModes[ MiRow - 1 ][ MiCol ] : DC_PRED ]` — an
unavailable-or-pre-write neighbour contributes the same
`Intra_Mode_Context[ DC_PRED ] = 0` weight). New `y_modes()` read
accessor surfaces the grid; a private `y_mode_at` helper performs
the bounds-clipped neighbour lookup. §8.3.2 ctx derivation honours
both §5.11.51 tile-bound predicates (`AvailU` / `AvailL` via
`TileGeometry::is_inside`) and the §8.3.2 `Intra_Mode_Context[]`
mapping (driven through the existing `intra_mode_ctx` helper). The
CDF row is selected via the existing
`TileCdfContext::intra_frame_y_mode_cdf` accessor — no new
`Default_*` table is added (r127 already transcribed the §9.4
`Default_Intra_Frame_Y_Mode_Cdf` verbatim). 7 new cdf-module tests
(539 → 546): fresh-walker `YModes[]` zero-init; three-way
out-of-range guard (`sub_size`, `mi_row`, `mi_col`) with no bits
read; rigged-CDF returns over symbols 0 (`DC_PRED`) and 12
(`PAETH_PRED`, the largest valid `YMode`), each with footprint
grid-stamp verification; (0,0) corner block routes through ctx
`(0, 0)` (both neighbours unavailable ⇒ both map to `DC_PRED`);
neighbour-`YModes`-grid read after a first block stamps `YMode =
V_PRED = 1`, observable via the next block's distinctive-symbol
return; and `Intra_Mode_Context[]` table application (a `YMode = 4`
= `D203_PRED` neighbour maps to ctx 4 per the §8.3.2 table). Closes
the first leaf of the §5.11.7 `else` arm and the first sub-element
of the §5.11.22 `intra_block_mode_info` composite. Remaining
`else`-arm elements (`intra_angle_info_y`, `uv_mode`,
`read_cfl_alphas`, `intra_angle_info_uv`, `palette_mode_info`,
`filter_intra_mode_info`) remain bounded leaf targets for r166+;
the `use_intrabc == 1` short-circuit still awaits the
motion-vector stack walker. `decode_av1` / `encode_av1` continue to
return `Error::NotImplemented`.

Round 164 lands the §5.11.7 **`use_intrabc` syntax element** (av1-spec
p.65) as a new `PartitionWalker::decode_use_intrabc` method —
the per-block intra-block-copy enable bit read on the §5.11.7
`intra_frame_mode_info()` body immediately after the
`RefFrame[ 0..2 ]` assignments r161
`decode_intra_frame_mode_info_prefix` produced. The spec body is the
two-arm `if ( allow_intrabc ) { use_intrabc S() } else { use_intrabc
= 0 }`; the dispatcher routes both arms exactly, with no bit consumed
on the fall-through. New `DEFAULT_INTRABC_CDF: [u16; 3] = [30531,
32768, 0]` verbatim from §9.4 (av1-spec p.430), a single-row binary
CDF with no context index (the §8.3.2 selection text is contextless,
mirroring `Default_Delta_Q_Cdf` / `Default_Delta_Lf_Cdf`). New
`TileCdfContext::intrabc` field + `intrabc_cdf()` selector; constant
re-exported at the crate root. Unlike the §5.11.5
`decode_skip` / `decode_skip_mode` / `decode_is_inter` siblings,
`decode_use_intrabc` writes nothing to the walker's §5.11.5 grids
(`Skips[]` / `SkipModes[]` / `IsInters[]` / `SegmentIds[]`) — AV1
has no per-block `UseIntrabc[][]` map; the value is consumed locally
by the §5.11.7 follow-on arm. 7 new cdf-module tests (532 → 539):
`allow = false` zero-bit fall-through on a hostile `0xFF` buffer;
rigged S() arm forcing symbols 0 / 1; three-way out-of-range guard
(`sub_size`, `mi_row`, `mi_col`); contextless-selection check
across three `(mi_row, mi_col, sub_size)` triples; `DEFAULT_INTRABC_CDF
== [30531, 32768, 0]` layout assertion plus accessor round-trip; and
a no-grid-stamp invariant covering `Skips[]` / `SkipModes[]` /
`IsInters[]` after a `use_intrabc = 1` call. The §5.11.7 follow-on
body now divides into two remaining unblocked arms: (i) the
`use_intrabc == 1` short-circuit (needs the motion-vector
stack walker) and (ii) the `intra_block_mode_info` composite
(`intra_frame_y_mode`, `intra_angle_info_y`, `uv_mode`, etc., each a
bounded leaf, queued for r165+). `decode_av1` / `encode_av1`
continue to return `Error::NotImplemented`.

Round 163 lands the §5.11.21 **`get_segment_id()` predicted-segment-id
helper** (av1-spec p.72) as a new free function
`cdf::get_segment_id` (re-exported at the crate root). The §5.11.21
function is the inter-frame per-block segment-id **prediction**
lookup: it scans the previous frame's `PrevSegmentIds[][]` map over
the on-screen window `xMis = Min(MiCols - MiCol, bw4)` ×
`yMis = Min(MiRows - MiRow, bh4)` and returns the `Min` of the cells
visited, with the §5.11.21 sentinel `seg = 7` (i.e. `MAX_SEGMENTS -
1`) as the initial upper bound. Returns `Some(seg)` with `seg` in
`-1..=7` — the `-1` sentinel surfaces only if a not-yet-decoded
cell of a previous walker falls inside the window, letting callers
detect a malformed reference surface via the existing §5.11.19
`predicted_segment_id > last_active_seg_id` range guard. Returns
`None` for caller-bug arguments: out-of-range `sub_size`, anchor
outside the current frame, previous-frame extent smaller than the
current frame's, or `prev_segment_ids.len() != prev_mi_rows *
prev_mi_cols`. The function is **pure** (no walker state, no
bitreader, no CDF), and complements the r162
`PartitionWalker::decode_inter_segment_id` caller (which takes a
pre-computed `predicted_segment_id: u8` so the walker can stay
inter-frame-state-free) — the §6.10 reference-frame walk now has a
verbatim spec-shaped routine to compute that argument from
`PrevSegmentIds[]`. 12 new cdf-module tests (520 → 532):
uniform-0 / uniform-7 reductions over several block sizes;
Min-over-2x2-window cells with out-of-window decoy values that must
not contribute; frame-edge `xMis`/`yMis` clipping on a `BLOCK_16X16`
anchored at the bottom-right of a 4x4 mi-grid; `-1` sentinel
round-trip; a wider-than-current previous frame exercising the
`prev_mi_cols` row stride; single-cell `BLOCK_4X4` covering exactly
`prev[MiRow][MiCol]` with a neighbour-cell decoy; out-of-range
guards for invalid `sub_size`, anchor past frame extent, previous-
frame extent smaller than current frame's, and length / shape
mismatch; an end-to-end composition test feeding `get_segment_id`'s
result into `decode_inter_segment_id`'s no-read
`!segmentation_update_map` arm and verifying the predicted id is
adopted with zero bit reads on a hostile `0xFF` buffer. The §5.11.18
`inter_frame_mode_info()` top-level dispatcher, §5.11.7 `use_intrabc`
arm, and §5.11.22 `intra_block_mode_info` composite remain the
next round's targets. `decode_av1` / `encode_av1` continue to
return `Error::NotImplemented`.

Round 162 lands the §5.11.19 **`inter_segment_id( preSkip )` syntax
element** (av1-spec p.71) as a new
`PartitionWalker::decode_inter_segment_id` method. `inter_segment_id`
is the §5.11.18 inter-frame variant of the per-block segment-id read,
called twice per block (once with `preSkip = 1` before the §5.11.11
`read_skip()` call and once with `preSkip = 0` after); together with
the §5.9.14 `SegIdPreSkip` derivation, the two calls cover every
combination of "segment-id read before / after skip" the inter-frame
walk encounters. New `SEGMENT_ID_PREDICTED_CONTEXTS = 3` constant and
new `DEFAULT_SEGMENT_ID_PREDICTED_CDF[3][3]` table verbatim from
av1-spec p.442 (each ctx row is the uniform `[128 * 128, 32768, 0]`
binary start). New `TileCdfContext::segment_id_predicted` field plus
`segment_id_predicted_cdf(ctx)` accessor implementing the §8.3.2
selection. New persistent `above_seg_pred_context: Vec<u8>` (length
`mi_cols`) and `left_seg_pred_context: Vec<u8>` (length `mi_rows`)
buffers on `PartitionWalker` per the §8.3.1 tile-entry initialisation
(`AboveSegPredContext[i] = 0`, `LeftSegPredContext[i] = 0`); the
§8.3.2 ctx walk reads `LeftSegPredContext[ MiRow ] +
AboveSegPredContext[ MiCol ]` (each `0..=1`, sum `0..=2`). The
dispatcher routes the full §5.11.19 cascade exactly: outer
`!segmentation_enabled` collapses to `segment_id = 0`; inner
`!segmentation_update_map` adopts `predictedSegmentId` without
reading; the `pre_skip && !SegIdPreSkip` early-exit returns
`segment_id = 0`; the post-skip `skip != 0` arm zeroes the context
arrays then descends into `decode_segment_id` (the §5.11.9 path
short-circuits on `skip`); the `segmentation_temporal_update == 1`
arm reads the binary `seg_id_predicted` symbol, branches to
`predictedSegmentId` adopt or `read_segment_id()`, then stamps the
context arrays; the `temporal_update == 0` fall-through reads
`read_segment_id()` without touching the context arrays. The
`predicted_segment_id` (§5.11.21 `get_segment_id()`) is caller-supplied
from the §6.10 reference-frame walk's `PrevSegmentIds[]` lookup so
the walker stays inter-frame-state-free. Range guards (`sub_size >=
BLOCK_SIZES`, `mi_row`/`mi_col` past extent, `last_active_seg_id >=
MAX_SEGMENTS`, `predicted_segment_id > last_active_seg_id`) fire
up-front. 11 new cdf-module tests (509 → 520): fresh-walker
context-array zeroing; `!segmentation_enabled` no-read on both
pre/post-skip arms; `!segmentation_update_map` adopts predicted-id
without reading; `pre_skip && !SegIdPreSkip` early-exit;
post-skip + `skip != 0` zeroes context arrays then descends into
the §5.11.9 short-circuit (poisoned-context arrays prove the spec
write fires); `temporal_update == 1` + rigged `seg_id_predicted = 1`
adopts `predictedSegmentId` and stamps context arrays to `1`;
`temporal_update == 1` + rigged `seg_id_predicted = 0` descends into
`read_segment_id()` and stamps context arrays to `0`;
`temporal_update == 0` fall-through leaves context arrays untouched;
five-way out-of-range guard; `Default_Segment_Id_Predicted_Cdf`
layout (`[16384, 32768, 0]` per row) and the
`segment_id_predicted_cdf` accessor round-trip. The §5.11.18
`inter_frame_mode_info()` top-level dispatcher (`use_intrabc` arm
+ the `LeftRefFrame`/`AboveRefFrame`/`LeftIntra`/`AboveIntra` /
`LeftSingle`/`AboveSingle` derivations + the §5.11.18 two-call
`inter_segment_id` protocol composing on top of r152
`read_skip()` / r156 `read_cdef()` / r154 `read_delta_qindex()` /
r155 `read_delta_lf()` / r158 `read_is_inter()` /
§5.11.22 `intra_block_mode_info` / §5.11.23 `inter_block_mode_info`)
is the next round's architectural payoff. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 161 lands the §5.11.7 **`intra_frame_mode_info()` prefix
dispatcher** (av1-spec p.64) as a new
`PartitionWalker::decode_intra_frame_mode_info_prefix` method
composing the first 11 lines of the §5.11.7 spec body into a single
walker entry-point. The dispatcher fires the §5.11.7 sequence in
exact spec order: `skip = 0`; conditional pre-skip
`intra_segment_id()` (r160); `skip_mode = 0`; `read_skip()` (r152);
conditional post-skip `intra_segment_id()`; `read_cdef()` (r156);
`read_delta_qindex()` (r154); `read_delta_lf()` (r155); the fixed
`RefFrame[0] = INTRA_FRAME` / `RefFrame[1] = NONE` trailing
assignments. Returns a new public `IntraFrameModeInfoPrefix` struct
carrying every post-call observable: `skip`, `skip_mode`,
`segment_id`, `lossless`, `cdef_idx`, `current_q_index`,
`current_delta_lf`, `ref_frame`. The §5.11.7 `SegIdPreSkip`
conditional routes the §5.11.8 call before or after the §5.11.11
`read_skip()` per the caller-passed `seg_id_pre_skip` boolean (the
§5.9.14 trailing derivation surfaced as
`SegmentationParams::seg_id_pre_skip`). `skip_mode` is fixed at `0`
because the intra-frame walk never calls `decode_skip_mode` —
§5.11.10 short-circuits on `!skip_mode_present` (intra-only frames
have `skip_mode_present == 0` per §5.9.21). The §6.10.4 `ReadDeltas
= 0` assignment on the spec's line 11 is left to the caller (the
walker stays stateless about per-superblock first-block detection),
matching the §6.10.4 pattern existing `decode_delta_qindex` /
`decode_delta_lf` call sites already use. Range guards (`sub_size >=
BLOCK_SIZES`, `mi_row >= MiRows`, `mi_col >= MiCols`,
`last_active_seg_id >= MAX_SEGMENTS`, `cdef_bits > 3`) fire on the
dispatcher level before any inner read so a caller bug never
produces a partial-read. 8 new cdf-module tests (501 → 509):
minimum-bit path (only the §5.11.11 `S()` consumed, every other
field short-circuits); `SegIdPreSkip = true` arm reads `segment_id`
(diff = 2 against rigged cdf) before `skip` (forced 1) and the
§5.11.56 `cdef_idx = -1` sentinel survives the skip short-circuit;
`SegIdPreSkip = false` arm post-skip with `skip = 1` triggers the
§5.11.9 short-circuit (`segment_id = pred = 0`, no bit consumed);
seg-skip-active path forces `skip = 1` with zero bits consumed on a
hostile `0xFF` buffer; `ref_frame = [INTRA_FRAME, NONE] = [0, -1]`
regardless of path taken; `read_deltas = true` + `delta_lf_present
= true` wires through to both delta reads (rigged `delta_q_abs = 0`
/ `delta_lf_abs = 0` ⇒ accumulators unchanged but the `S()` reads
advance the decoder); five-way out-of-range guard on `mi_row`,
`mi_col`, `sub_size`, `last_active_seg_id`, `cdef_bits`;
`skip_mode = 0` on both pre-skip arms (verifies the dispatcher
never calls `decode_skip_mode` and `SkipModes[]` stays untouched).
The §5.11.7 follow-on body (`use_intrabc` arm + the §5.11.22
`intra_block_mode_info` composite — `intra_frame_y_mode`,
`intra_angle_info_y`, `uv_mode`, `intra_angle_info_uv`,
`palette_mode_info`, `filter_intra_mode_info`) and the §5.11.18
`inter_frame_mode_info` / §5.11.19 `inter_segment_id` two-call
protocol remain the next round's targets. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 160 lands the §5.11.8 **`intra_segment_id()` syntax element**
(av1-spec p.66) as a new `PartitionWalker::decode_intra_segment_id`
method built on top of r159's `decode_segment_id`. `intra_segment_id`
is the intra-frame variant of the per-block segment-id read, called
from §5.11.7 `intra_frame_mode_info` on both the `SegIdPreSkip`
pre-skip arm and the `!SegIdPreSkip` post-skip arm. The §5.11.8 spec
body is short — `if (segmentation_enabled) read_segment_id(); else
segment_id = 0; Lossless = LosslessArray[segment_id]` — but the
Lossless lookup is the first place the per-segment §6.8.2
`LosslessArray[]` table reaches the leaf walk. The dispatch is exact:
the `segmentation_enabled = true` arm descends into the r159
implementation (which performs the §5.11.9 neighbour cascade, the
skip / non-skip dispatch, the `S()` read against
`TileSegmentIdCdf[ctx]`, the `neg_deinterleave` mapping, and the
§5.11.5 grid-fill); the `segmentation_enabled = false` arm forces
`segment_id = 0` without reading any bits and stamps the `bh4 * bw4`
footprint to `0` so subsequent §5.11.9 neighbour lookups see a real
zero rather than the `-1` sentinel. Both arms then resolve
`Lossless = lossless_array[segment_id as usize]` from the
caller-supplied `&[bool; MAX_SEGMENTS]` table (the §6.8.2 derivation
the frame-header walk computes from `qindex = get_qindex(1,
segmentId)` plus the five `DeltaQ?Dc` / `DeltaQ?Ac` offsets;
`compute_coded_lossless` in `frame_header.rs` is the frame-wide
conjunction, this round's table is the per-segment data the walker
indexes by `segment_id`). The walker stays segmentation-state-free:
callers pass `segmentation_enabled`, `last_active_seg_id`, and
`lossless_array` per-call, mirroring the r159 pattern. Range guards
(out-of-range `sub_size`, `mi_row` / `mi_col` past extent,
`last_active_seg_id >= MAX_SEGMENTS`) fire on both arms so the
no-symbol path is total over the same input space as the
bitstream-reading path. The §5.11.18 `inter_frame_mode_info`
top-level dispatcher (the §5.11.19 `inter_segment_id(preSkip)`
two-call protocol) remains the next round's target. 7 new cdf-module
tests (494 → 501): `!segmentation_enabled` no-read on a hostile
`0xFF` buffer with `lossless_array[0] = true` and `= false` arms
each; `segmentation_enabled = true, skip = 1` at frame origin
(pred = 0, no `S()` consumed, grid stamped to 0, Lossless from slot
0); `segmentation_enabled = true, skip = 0` reading `diff = 3` on a
rigged CDF (Lossless from slot 3 = true); per-segment Lossless
indexing (rig `diff = 5`, set `lossless_array[5] = false` while
every other slot is `true`, expect `Lossless = false`); bottom-right
edge clip on `BLOCK_16X16 @ (2, 2)` in a 4×4 frame on the
`!segmentation_enabled` arm stamps only the in-grid 2×2 quadrant
and leaves the rest at `-1`; five-way out-of-range guard
(`mi_row >= mi_rows`, `mi_col >= mi_cols`, `sub_size ==
BLOCK_SIZES`, `last_active_seg_id == MAX_SEGMENTS`, and the
`segmentation_enabled = true` path also rejects bad mi-row). The
§5.11.5 `decode_block()` body itself (coefficient / motion-vector /
reconstruction) remains the next round's target. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 159 lands the §5.11.9 **`read_segment_id()` syntax element**
(av1-spec p.66) as a new `PartitionWalker::decode_segment_id`
method on the r158 walker, plus a `SegmentIds[r][c]` grid carried
alongside the r158 `IsInters[]`, r156 `cdef_idx[]`, r154
`SkipModes[]`, r152 `Skips[]`, and the existing §6.10.4 `MiSizes[]`
grids. The grid is pre-filled with the §5.11.9 `-1` sentinel (the
spec's `prevUL = -1` / `prevU = -1` / `prevL = -1` "neighbour
unavailable" marker); cells inside a decoded block's `bh4 * bw4`
footprint then carry the block's `segment_id ∈ 0..MAX_SEGMENTS = 0..8`.
The walker stays segmentation-state-free: the caller passes
`last_active_seg_id` (the §5.9.14 trailing derivation) and the
`skip` value the §5.11.11 `decode_skip` just returned. The §5.11.9
neighbour cascade is honoured exactly as the spec spells it out:
`prevUL` requires both `AvailU` AND `AvailL`; `prevU` and `prevL`
each gate on their own edge; out-of-grid neighbours fall through to
`-1`. The four-arm `pred` derivation (`prevU == -1 ⇒ prevL/0`;
`prevL == -1 ⇒ prevU`; `prevUL == prevU ⇒ prevU`; else `prevL`) is
preserved verbatim — two arms happen to return `prev_u` but the
predicates are semantically distinct ("left neighbour unavailable"
vs. "above-left agrees with above"), and collapsing them would
obscure the spec correspondence. The §5.11.9 dispatch distinguishes
two paths: `skip != 0` ⇒ `segment_id = pred` (zero bits read; the
spatially-predicted-on-skip semantics the spec relies on for
skip-block segment-map continuity); else `diff S()` against
`TileSegmentIdCdf[ctx]` (ctx from the existing `segment_id_ctx`
helper, which already honours the `-1` sentinels), then `segment_id
= neg_deinterleave(diff, pred, last_active_seg_id + 1)`. The
§5.11.5 grid-fill stamps the result over the block's `bw4 * bh4`
footprint, clipped at the frame's extent. A new public module-level
`neg_deinterleave(diff, ref, max)` helper transcribes the §5.11.9
bijection (`diff ∈ 0..max ↔ segment_id ∈ 0..max` biased toward
values near `ref`). `decode_segment_id` is the read inside both
§5.11.8 `intra_segment_id` and §5.11.19 `inter_segment_id` (the
segmentation-enabled inner branch in each); the latter's `preSkip`
machinery and the `segment_id`-aware Arm 2 of §5.11.20 (which
consumes `FeatureData[segment_id][SEG_LVL_REF_FRAME]`) are now both
unblocked at the leaf-walk level. 11 new cdf-module tests (483 →
494): fresh-walker grid all `-1`; skip short-circuit at frame
origin writes `segment_id = pred = 0` (no S() bit consumed on a
hostile `0xFF` byte buffer); skip inherits `prev_u` when `prev_l`
is unavailable; non-skip path with `pred = 0` returns `diff`
unchanged; direct `neg_deinterleave` table exercises for both the
`2 * ref < max` upward branch (`pred = 2`, all eight `diff` values)
and the `2 * ref >= max` downward branch (`pred = 5`, all eight
`diff` values); edge cases (`ref == 0` identity, `ref == max - 1`
inverted, smallest non-trivial alphabet `max = 2`); ctx-0 origin
selection via rigged rows; ctx-2 all-neighbours-match selection
through three walker-stamped seeds; bottom-right edge clip on
`BLOCK_16X16 @ (2, 2)` in a 4×4 frame stamps only the in-grid 2×2
quadrant; four-way out-of-range guard (`mi_row` past extent /
`mi_col` past extent / `sub_size == BLOCK_SIZES` /
`last_active_seg_id >= MAX_SEGMENTS`) ⇒ `PartitionWalkOutOfRange`.
The §5.11.5 `decode_block()` body itself (coefficient /
motion-vector / reconstruction) remains the next round's target.
`decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

Round 158 lands the §5.11.20 **`read_is_inter()` syntax element**
(av1-spec p.71-72) as a new `PartitionWalker::decode_is_inter`
method on the r157 walker, plus an `IsInters[r][c]` flag grid
carried alongside the r156 `cdef_idx[]`, r154 `SkipModes[]`,
r152 `Skips[]`, and the existing §6.10.4 `MiSizes[]` grids
(`IsInters[ r + y ][ c + x ] = is_inter` per the §5.11.5
footer at av1-spec p.65). All four arms of the §5.11.20 dispatch
are honoured in spec order (first match fires, no read on the
short-circuit arms): Arm 1 — `skip_mode == 1` forces
`is_inter = 1` (a compound-reference skip block is by definition
inter); Arm 2 — `seg_feature_active(SEG_LVL_REF_FRAME) == true`
routes through the caller-pre-computed
`FeatureData[segment_id][SEG_LVL_REF_FRAME] != INTRA_FRAME`
boolean (the walker stays segmentation-state-free, identical to
r154's `seg_skip_mode_off` pattern); Arm 3 —
`seg_feature_active(SEG_LVL_GLOBALMV) == true` forces
`is_inter = 1` (global-MV is intrinsically inter); Arm 4 — `S()`
symbol read against `TileIsInterCdf[ctx]` with `ctx` from the
existing `is_inter_ctx(above_intra, left_intra)` helper. Per
§5.11.18 the §8.3.2 neighbour intra-ness is sampled from
`LeftRefFrame[0] / AboveRefFrame[0]` (`LeftIntra = LeftRefFrame[0]
<= INTRA_FRAME`); the walker derives this from the complement of
its `IsInters[]` grid (`intra = !is_inter`), with an unavailable
neighbour treated as intra per §5.11.18 (`LeftRefFrame[0] = AvailL
? RefFrames[..][0] : INTRA_FRAME` ⇒ `None` to `is_inter_ctx`).
The §5.11.5 grid-fill stamps the decoded value over the block's
`bw4 * bh4` footprint, clipped at the frame's `MiRows` / `MiCols`
extent so a leaf straddling the bottom or right edge fills only
the in-grid portion. New `PartitionWalker::is_inters()` accessor
returns a row-major view; the fresh-walker initial state is
all-zero (a pre-write neighbour weights as intra in the §8.3.2
ctx walk, which is the natural identity for the spec's `LeftIntra
= LeftRefFrame[0] <= INTRA_FRAME` reading and matches an
unavailable neighbour gated by `AvailU` / `AvailL`). `is_inter`
is the per-block intra/inter classifier read inside §5.11.18
`inter_frame_mode_info` (after `inter_segment_id` / `read_skip` /
`read_cdef` / `read_delta_qindex` / `read_delta_lf`) that
dispatches between §5.11.22 `intra_block_mode_info` and §5.11.23
`inter_block_mode_info`; intra-only frames never call it. 15 new
cdf-module tests (468 → 483): fresh-walker grid all-zero; Arm 1
skip_mode short-circuit (position-invariant); Arm 2 routing to
intra (`seg_ref_frame_is_inter = false`) and to inter
(`seg_ref_frame_is_inter = true`), both position-invariant;
Arm 3 globalmv short-circuit; Arm 1 takes precedence over both
Arm 2 and Arm 3; Arm 2 takes precedence over Arm 3; else-branch
S() returning 0 / 1 on a forced binary CDF (the
`is_inter = 1` arm verifies the footprint grid-stamp); ctx-0
selection at the frame origin; ctx-3 selection through two prior
intra-stamping seeds; ctx-1 through one intra + one inter
neighbour with both available; ctx-2 through a non-zero-tile-col
origin clearing AvailL with an above-intra seed; bottom-right
edge clip on `BLOCK_16X16 @ (2, 2)` in a 4×4 frame; three-way
out-of-range guard ⇒ `PartitionWalkOutOfRange`. The §5.11.5
`decode_block()` body itself (coefficient / motion-vector /
reconstruction) remains the next round's target. `decode_av1` /
`encode_av1` continue to return `Error::NotImplemented`.

Round 157 lands the §5.11.56 **`read_cdef()` syntax element** plus
the §5.11.55 **`clear_cdef()` reset** (av1-spec p.104) as new
`PartitionWalker::decode_cdef` + `PartitionWalker::clear_cdef`
methods on the r156 walker, alongside a `cdef_idx: Vec<i8>`
row-major grid sized `MiRows × MiCols` (pre-filled with the `-1`
sentinel per §5.11.55, interpreted as "CDEF disabled for that
block" per §6.10.40) with a `cdef_idx()` read accessor. CDEF
operates on 64×64 anchor cells, so `decode_cdef` masks the leaf's
`(MiRow, MiCol)` to the anchor at `(MiRow & cdefMask4, MiCol &
cdefMask4)` (`cdefMask4 = ~(cdefSize4 - 1)`, `cdefSize4 =
Num_4x4_Blocks_Wide[ BLOCK_64X64 ] = 16` so the low four bits are
zeroed). If the anchor still holds the `-1` sentinel, an
`L(cdef_bits)` literal is read (`cdef_bits ∈ 0..=3` per §5.9.19
`f(2)` ⇒ decoded value in `0..=7`) and the grid-fill loop stamps
the value across the leaf's `(w4, h4)` footprint at the `cdefSize4
= 16` stride so super-64 blocks (`BLOCK_128X128`) reach all four
anchor cells while sub-64 blocks touch only their containing
anchor. Subsequent leaves whose `cdefMask4` lands on the same
anchor short-circuit (no read; the anchor already holds the
value). The §5.11.56 short-circuit set is honoured: `skip ||
CodedLossless || !enable_cdef || allow_intrabc` ⇒ no read; the
anchor's current value (sentinel or prior stamp) is returned
unchanged. `clear_cdef( r, c, use_128x128_superblock )` — called
by the §5.11.2 tile walk at each superblock — stamps `-1` at the
one (64×64 superblock) or four (128×128 superblock) anchor cells;
out-of-grid anchors are silently skipped so the bottom/right
superblock can straddle the frame edge without panic. `cdef_bits
== 0` yields `L(0) = 0` (no bit read) and still transitions the
anchor from `-1` to `0`, matching the §5.9.19 single-strength
case. 18 new cdf-module tests (450 → 468): fresh-walker all-`-1`
invariant; `clear_cdef` 64×64 single-anchor stamp; `clear_cdef`
128×128 four-anchor stamp; `clear_cdef` out-of-grid silent skip;
each of the four `skip` / `CodedLossless` / `!enable_cdef` /
`allow_intrabc` short-circuit gates (separately, with `0xFF` byte
buffers proving no bit consumed); first-leaf-reads-literal-
and-stamps-anchor with off-anchor cell stays at sentinel;
second-leaf-in-anchor-no-read; `cdef_bits == 0` zero-bit stamp;
`cdef_bits == 3` upper-bound; anchor-mask routes (10, 13) ⇒ (0,
0); `BLOCK_128X128` stamps all four anchors with off-anchor cells
at sentinel; grid-fill clips at frame edge; short-circuit returns
prior stamp; `clear_cdef` after stamp resets anchor; four-way
out-of-range guard (`mi_row` past extent / `mi_col` past extent /
`sub_size == BLOCK_SIZES` / `cdef_bits > 3`) ⇒
`PartitionWalkOutOfRange`. The §5.11.5 `decode_block()` body
itself (coefficient / motion-vector / reconstruction) remains the
next round's target. `decode_av1` / `encode_av1` continue to
return `Error::NotImplemented`.

Round 156 lands the §5.11.13 **`read_delta_lf()` syntax element**
(av1-spec p.68) as a new `PartitionWalker::decode_delta_lf` method,
structurally parallel to r155's §5.11.12 walker but iterating
`frameLfCount` times over a four-slot `DeltaLF[ i ]` accumulator
and selecting between the §8.3.2 single-LF (`TileDeltaLFCdf`) and
per-edge multi-LF (`TileDeltaLFMultiCdf[ i ]`) CDF rows via the
`delta_lf_multi` argument. Adds a
`current_delta_lf: [i32; FRAME_LF_COUNT]` accumulator on
`PartitionWalker` with `current_delta_lf()` read accessor and
`reset_current_delta_lf()` for the §5.11.2 tile-entry reset.
Honours the §5.11.13 superblock-skip short-circuit (identical to
§5.11.12) and the outer `ReadDeltas && delta_lf_present` gate
(two AND-ed flags — `delta_lf_present` is the §5.9.18 frame-header
bit, accepted as an argument). When the gate passes, `frameLfCount`
is derived locally: `delta_lf_multi == 0 ⇒ 1`;
`delta_lf_multi == 1 && mono_chrome ⇒ FRAME_LF_COUNT - 2 = 2`;
otherwise `FRAME_LF_COUNT = 4`. Each iteration reads
`delta_lf_abs` `S()` against the branch-selected CDF, then either
the literal value or the §5.11.13 escape ladder
(`delta_lf_rem_bits` `L(3)` + post-increment + `delta_lf_abs_bits`
`L(rem_bits + 1)` ⇒ `deltaLfAbs = abs_bits + (1 << n) + 1`); for
non-zero magnitudes reads `delta_lf_sign_bit` `L(1)` and applies
`DeltaLF[ i ] = Clip3(-MAX_LOOP_FILTER, MAX_LOOP_FILTER,
DeltaLF[ i ] + (reducedDeltaLfLevel << delta_lf_res))`. New
constants `DELTA_LF_SMALL = 3`, `FRAME_LF_COUNT = 4`,
`cdf::MAX_LOOP_FILTER = 63i32` (distinct from the §5.9.11
`uncompressed_header_tail::MAX_LOOP_FILTER` `i16` twin); new table
`DEFAULT_DELTA_LF_CDF` transcribed verbatim from §9.4 p.431
(`[28160, 32120, 32677, 32768, 0]`, identical row to
`DEFAULT_DELTA_Q_CDF` per the spec — preserved as two independent
constants so adaptation drift on one does not leak through the
other); new fields `TileCdfContext::delta_lf` +
`TileCdfContext::delta_lf_multi` with `delta_lf_cdf()` /
`delta_lf_multi_cdf(i)` accessors. 17 new cdf-module tests (433 ->
450): default-CDF literal match (incl. §9.4 equality with
`DEFAULT_DELTA_Q_CDF`); init-from-defaults invariant for the
single-LF row and all four multi-LF rows; sb-skip short-circuit
at both `use_128x128_superblock` settings; `ReadDeltas` false
short-circuit; `delta_lf_present` false short-circuit; single-LF
branch writes only `DeltaLF[ 0 ]`; multi-LF colour branch writes
all four slots; multi-LF monochrome branch writes only the two Y
slots; zero-`delta_lf_abs` no-update; literal-positive with
shift; Clip3 upper-bound at `MAX_LOOP_FILTER = 63`; Clip3
lower-bound via hostile seed at `i32::MIN + 1`; `DELTA_LF_SMALL`
escape ladder minimum value; cross-call accumulation;
fresh-walker initial accumulator all-zero + `reset_current_delta_lf`
round-trip; out-of-range guards ⇒ `PartitionWalkOutOfRange`. The
§5.11.5 `decode_block()` body itself (coefficient / motion-vector
/ reconstruction) remains the next round's target. `decode_av1`
/ `encode_av1` continue to return `Error::NotImplemented`.

Round 155 lands the §5.11.12 **`read_delta_qindex()` syntax
element** (av1-spec p.67) as a new
`PartitionWalker::decode_delta_qindex` method on the r154 walker,
plus a `CurrentQIndex` scalar tracked across calls via
`PartitionWalker::current_q_index()` /
`PartitionWalker::set_current_q_index()` accessors. Honours the
§5.11.12 superblock-skip short-circuit (`MiSize == sbSize && skip`
⇒ no symbol read; `sbSize = use_128x128_superblock ?
BLOCK_128X128 : BLOCK_64X64`, the latter via the §5.5.1
sequence-header flag) and the outer `ReadDeltas` (§6.10.4) gate.
Otherwise an `S()` symbol is decoded against `TileDeltaQCdf` —
length `DELTA_Q_SMALL + 2 = 5`, no context index per §8.3.2; the
working CDF is a single row. A decoded value of `DELTA_Q_SMALL`
(`= 3`) triggers the §5.11.12 escape ladder
(`delta_q_rem_bits` `L(3)` + post-increment + `delta_q_abs_bits`
`L(rem_bits + 1)`), reconstructing
`delta_q_abs = delta_q_abs_bits + (1 << n) + 1` over the extended
range `0..=2 ∪ 3..=511`. Non-zero `delta_q_abs` reads
`delta_q_sign_bit` `L(1)` and applies the spec's
`CurrentQIndex = Clip3(1, 255, CurrentQIndex +
(reducedDeltaQIndex << delta_q_res))` update. New constant
`DELTA_Q_SMALL = 3`, new table `DEFAULT_DELTA_Q_CDF` transcribed
verbatim from §9.4 p.431 (`[28160, 32120, 32677, 32768, 0]`), new
field `TileCdfContext::delta_q` + `delta_q_cdf()` accessor. 16
new cdf-module tests (417 -> 433): default-CDF literal match;
init-from-defaults invariant; sb-skip short-circuit for both
`use_128x128_superblock` values; `ReadDeltas` false short-circuit;
zero-`delta_q_abs` no-update; literal-positive no-shift; literal
positive with shift; Clip3 lower-bound via hostile seed; Clip3
upper-bound; DELTA_Q_SMALL escape ladder minimum value; escape
ladder stays in `Clip3(1, 255)` range; cross-call accumulation;
fresh-walker initial `CurrentQIndex = 0`; out-of-range guards ⇒
`PartitionWalkOutOfRange`; arithmetic-decoder zero-byte sign-bit
observation. `decode_av1` / `encode_av1` continue to return
`Error::NotImplemented`.

## Sources consulted (clean-room wall)

* AV1 Bitstream & Decoding Process Specification — AOMedia, copy at
  `docs/video/av1/av1-spec.txt` / `av1-spec.pdf`. Sections cited in
  module documentation:
  * Round 1: §4.10.5, §5.3.1, §5.3.2, §5.3.3, §6.2.1, §6.2.2,
    §6.2.3.
  * Round 2: §3 (constants — `SELECT_SCREEN_CONTENT_TOOLS`,
    `SELECT_INTEGER_MV`, `CP_UNSPECIFIED`, `TC_UNSPECIFIED`,
    `MC_UNSPECIFIED`, `CSP_UNKNOWN`, `CP_BT_709`, `TC_SRGB`,
    `MC_IDENTITY`), §4.10.2 (`f(n)`), §4.10.3 (`uvlc()`), §5.5.1
    (`sequence_header_obu`), §5.5.2 (`color_config`), §5.5.3
    (`timing_info`), §5.5.4 (`decoder_model_info`), §5.5.5
    (`operating_parameters_info`), §6.4.1 / §6.4.2 (semantics +
    conformance), §8.1 (`read_bit`).
  * Round 3: §3 (constants — `NUM_REF_FRAMES`, `PRIMARY_REF_NONE`),
    §5.9.1 (`frame_header_obu` framing), §5.9.2
    (`uncompressed_header` leading slice — `show_existing_frame`
    through `refresh_frame_flags`), §6.8.1 / §6.8.2 (semantics +
    conformance, including the `idLen <= 16` constraint on
    `display_frame_id`).
  * Round 4: §3 (constants — `SUPERRES_NUM`, `SUPERRES_DENOM_MIN`,
    `SUPERRES_DENOM_BITS`, `MI_SIZE`), §5.9.5 (`frame_size`),
    §5.9.6 (`render_size`), §5.9.7 (`frame_size_with_refs` — the
    `found_ref == 0` branch only), §5.9.8 (`superres_params`),
    §5.9.9 (`compute_image_size`), §6.8.4 / §6.8.5 / §6.8.6 /
    §6.8.7 / §6.8.8 (semantics).
  * Round 5: §3 (constants — `TOTAL_REFS_PER_FRAME`,
    `INTRA_FRAME`, `LAST_FRAME`, `GOLDEN_FRAME`, `BWDREF_FRAME`,
    `ALTREF_FRAME`, `ALTREF2_FRAME`), §4.10.6 (`su(n)`), §5.9.10
    (`read_interpolation_filter`), §5.9.11 (`loop_filter_params`),
    §5.9.12 (`quantization_params`), §5.9.13 (`read_delta_q`),
    §6.8.9 / §6.8.10 / §6.8.11 / §6.8.12 (semantics).
  * Round 6: §3 (constants — `MAX_TILE_WIDTH`, `MAX_TILE_AREA`,
    `MAX_TILE_ROWS`, `MAX_TILE_COLS`), §4.7 (`FloorLog2`),
    §4.10.7 (`ns(n)`), §5.9.2 (the `allow_intrabc` +
    `disable_frame_end_update_cdf` + `tile_info()` placement
    inside the `if (FrameIsIntra)` block / `reduced_still_picture
    header || disable_cdf_update` gate), §5.9.15 (`tile_info`),
    §5.9.16 (`tile_log2`), §6.8.14 (semantics + conformance —
    `TileCols <= MAX_TILE_COLS`, `TileRows <= MAX_TILE_ROWS`,
    `context_update_tile_id < TileCols * TileRows`).
  * Round 7: §3 (constants — `MAX_SEGMENTS`, `SEG_LVL_MAX`,
    `SEG_LVL_ALT_Q`, `SEG_LVL_ALT_LF_Y_V`, `SEG_LVL_ALT_LF_Y_H`,
    `SEG_LVL_ALT_LF_U`, `SEG_LVL_ALT_LF_V`, `SEG_LVL_REF_FRAME`,
    `SEG_LVL_SKIP`, `SEG_LVL_GLOBALMV`, `MAX_LOOP_FILTER`),
    §"Conventions" (`Clip3`), §5.9.2 (the `quantization_params()`
    + `segmentation_params()` placement after `tile_info()` in the
    `if (FrameIsIntra)` block), §5.9.14 (`segmentation_params`),
    §6.8.13 (semantics — `SegIdPreSkip` / `LastActiveSegId`
    derivations).
  * Round 8: §5.9.2 (the `delta_q_params()` + `delta_lf_params()`
    placement after `segmentation_params()` in the
    `if (FrameIsIntra)` block), §5.9.17 (`delta_q_params` — the
    `base_q_idx > 0` gate on `delta_q_present`, the `delta_q_present`
    gate on `delta_q_res`), §5.9.18 (`delta_lf_params` — the
    `delta_q_present` gate on the block, the `!allow_intrabc` gate on
    `delta_lf_present`, the `delta_lf_present` gate on `delta_lf_res`
    / `delta_lf_multi`), §6.8.15 (quantizer-index delta semantics),
    §6.8.16 (loop-filter delta semantics).
  * Round 9: §5.9.2 (the `CodedLossless` / `AllLossless` derivation
    lines and the `loop_filter_params()` placement after
    `delta_lf_params()` in the `if (FrameIsIntra)` block), §5.9.11
    (`loop_filter_params` — the `CodedLossless || allow_intrabc`
    short-circuit, the `NumPlanes > 1 && (level[0] || level[1])` gate on
    the chroma levels, the delta-update walk), §8.7 (`get_qindex`
    `ignoreDeltaQ` branch + the `SEG_LVL_ALT_Q` `Clip3(0, 255, ..)`
    clamp), §5.9.14's `seg_feature_active_idx`, §6.8.10 (loop-filter
    level / sharpness semantics).
  * Round 10: §5.9.2 (the `cdef_params()` placement after
    `loop_filter_params()` in the `if (FrameIsIntra)` block), §5.9.19
    (`cdef_params` — the `CodedLossless || allow_intrabc || !enable_cdef`
    short-circuit, the `CdefDamping = cdef_damping_minus_3 + 3`
    derivation, the `1 << cdef_bits` strength loop, the `NumPlanes > 1`
    gate on the chroma strengths, the secondary `== 3 ⇒ += 1`
    adjustment), §5.5.1 (`enable_cdef`), §6.4 (`enable_cdef`
    semantics), §6.10.14 (CDEF params semantics — `cdef_damping_minus_3`
    / `cdef_bits` / `cdef_*_pri_strength` / `cdef_*_sec_strength`).
  * Round 11: §3 (constant — `RESTORATION_TILESIZE_MAX = 256`),
    §5.9.2 (the `AllLossless = CodedLossless && (FrameWidth ==
    UpscaledWidth)` derivation line and the `lr_params()` placement
    after `cdef_params()` in the `if (FrameIsIntra)` block), §5.9.20
    (`lr_params` — the `AllLossless || allow_intrabc ||
    !enable_restoration` short-circuit, the per-plane `lr_type` `f(2)`
    loop with `Remap_Lr_Type[4] = { RESTORE_NONE, RESTORE_SWITCHABLE,
    RESTORE_WIENER, RESTORE_SGRPROJ }`, the `UsesLr` / `usesChromaLr`
    derivation, the 128×128-superblock-gated `lr_unit_shift`
    post-increment vs the non-128 `lr_unit_extra_shift` extension,
    the 4:2:0-chroma-gated `lr_uv_shift`, and the
    `LoopRestorationSize[plane]` derivation —
    `RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift)` for luma and
    `>> lr_uv_shift` for chroma), §5.5.1 (`enable_restoration`,
    `use_128x128_superblock`), §6.10.15 (Loop restoration params
    semantics — the `FrameRestorationType` symbol-value table).
  * Round 12: §3 (constant — `TX_MODES = 3`), §5.9.2 (the
    `read_tx_mode()` placement after `lr_params()` in the
    `if (FrameIsIntra)` block), §5.9.21 (`read_tx_mode` — the
    `CodedLossless == 1 ⇒ TxMode = ONLY_4X4` no-bits branch and the
    `tx_mode_select` `f(1)` ⇒ `TX_MODE_SELECT` / `TX_MODE_LARGEST`
    selection), §6.8.21 (TX mode semantics — the `TxMode` symbol-value
    table `ONLY_4X4 = 0`, `TX_MODE_LARGEST = 1`, `TX_MODE_SELECT = 2`).
  * Round 13: §3 (constants — `WARPEDMODEL_PREC_BITS = 16`, the `GM_*`
    bit / precision constants, `IDENTITY`/`TRANSLATION`/`ROTZOOM`/
    `AFFINE`, `LAST_FRAME`..`ALTREF_FRAME` reference-frame indices),
    §5.9.2 (the tail ordering after `read_tx_mode()`:
    `frame_reference_mode()` / `skip_mode_params()` / the
    `allow_warped_motion` guard / `reduced_tx_set` `f(1)` /
    `global_motion_params()` / `film_grain_params()`), §5.9.22
    (`skip_mode_params` — the `FrameIsIntra` ⇒ `skipModeAllowed = 0`
    branch), §5.9.23 (`frame_reference_mode` — `FrameIsIntra` ⇒
    `reference_select = 0`), §5.9.24 (`global_motion_params` — the
    identity initialiser, the `FrameIsIntra` early return, and the
    per-ref `is_global` / `is_rot_zoom` / `is_translation` type walk),
    §5.9.25 (`read_global_param` — `absBits` / `precBits` / `precDiff` /
    `round` / `sub` derivation), §5.9.26–§5.9.29
    (`decode_signed_subexp_with_ref` / `decode_unsigned_subexp_with_ref`
    / `decode_subexp` / `inverse_recenter`), §5.9.30 (`film_grain_params`
    — the `reset_grain_params()` short-circuits, the
    `apply_grain` / `grain_seed` / `update_grain` reads, the predicted
    `update_grain == 0` path, the Y / Cb / Cr scaling-point loops, the
    chroma-suppression branch, the AR-coefficient loops, and the
    chroma mult/offset triplets), §6.8.18 (global-motion-type symbol
    values), §6.8.20 (film-grain params semantics).
  * Round 15: §3 (constants — `EC_PROB_SHIFT = 6`, `EC_MIN_PROB = 4`),
    §4.7 (`FloorLog2`), §4.10.8 (`L(n)` descriptor), §4.10.10 (`NS(n)`
    descriptor), §5.9.28 (`decode_subexp_bool` — the bool variant of the
    subexponential code), §8.1 (`f(n)` parsing process — shared bit
    reader), §8.2.2 (`init_symbol` — `numBits` / `paddedBuf` /
    `SymbolValue` / `SymbolRange` / `SymbolMaxBits` init), §8.2.3
    (`read_bool` — the fixed `[1<<14, 1<<15, 0]` CDF), §8.2.4
    (`exit_symbol` — `trailingBitPosition` / `paddingEndPosition`
    derivation, the `Max(0, SymbolMaxBits)` advance, the
    `SymbolMaxBits >= -14` conformance gate), §8.2.5 (`read_literal`),
    §8.2.6 (`read_symbol` — the CDF-adaptive symbol search loop + the
    seven-step renormalisation), §8.3 (the adaptive-rate CDF update).
  * Round 16: §3 / §9.3 (constants — `INTRA_MODES = 13`,
    `INTRA_MODE_CONTEXTS = 5`, `PARTITION_CONTEXTS = 4`,
    `SKIP_CONTEXTS = 3`, `SEGMENT_ID_CONTEXTS = 3`, `MAX_SEGMENTS = 8`),
    §8.3.1 (the "set equal to a copy of `Default_*_Cdf`" init step that
    seeds the per-tile `Tile*Cdf` working set), §8.3.2 (selection
    paragraphs for `intra_frame_y_mode` — including the
    `Intra_Mode_Context[ INTRA_MODES ] = { 0, 1, 2, 3, 4, 4, 4, 4, 3, 0,
    1, 2, 0 }` array — `partition` with the `bsl` / `ctx = left * 2 +
    above` derivation, `skip` with the neighbour-`Skips[]` sum, and
    `segment_id` with the `prevUL / prevU / prevL` branch ladder), §9.4
    (default CDF table values for `Default_Intra_Frame_Y_Mode_Cdf`,
    `Default_Partition_W{8,16,32,64,128}_Cdf`, `Default_Skip_Cdf`,
    `Default_Segment_Id_Cdf`).
  * Round 17: §3 (constants — `MV_CONTEXTS = 2`,
    `MV_INTRABC_CONTEXT = 1`, `MV_JOINTS = 4`, `MV_CLASSES = 11`,
    `CLASS0_SIZE = 2`, `MV_OFFSET_BITS = 10`), §5.11.31 (`read_mv()` —
    the `MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0` derivation),
    §5.11.32 (`read_mv_component()` — the per-`comp` walk through
    `mv_sign` / `mv_class` / `mv_class0_bit` / `mv_class0_fr` /
    `mv_class0_hp` / `mv_bit` / `mv_fr` / `mv_hp`), §8.3.1 (the
    per-`i = 0..MV_CONTEXTS-1` / per-`comp = 0..1` "set equal to a copy
    of `Default_Mv_*_Cdf`" init step for the nine `Mv*Cdf` working
    arrays), §8.3.2 (the selection paragraphs — `mv_joint:
    TileMvJointCdf[ MvCtx ]`, `mv_sign: TileMvSignCdf[ MvCtx ][ comp ]`,
    `mv_class: TileMvClassCdf[ MvCtx ][ comp ]`, `mv_class0_bit:
    TileMvClass0BitCdf[ MvCtx ][ comp ]`, `mv_class0_fr:
    TileMvClass0FrCdf[ MvCtx ][ comp ][ mv_class0_bit ]`,
    `mv_class0_hp: TileMvClass0HpCdf[ MvCtx ][ comp ]`, `mv_fr:
    TileMvFrCdf[ MvCtx ][ comp ]`, `mv_hp: TileMvHpCdf[ MvCtx ][ comp ]`,
    `mv_bit: TileMvBitCdf[ MvCtx ][ comp ][ i ]`), §9.4 (default CDF
    table values for `Default_Mv_Joint_Cdf`, `Default_Mv_Sign_Cdf`,
    `Default_Mv_Class_Cdf`, `Default_Mv_Class0_Bit_Cdf`,
    `Default_Mv_Class0_Fr_Cdf`, `Default_Mv_Class0_Hp_Cdf`,
    `Default_Mv_Bit_Cdf`, `Default_Mv_Fr_Cdf`, `Default_Mv_Hp_Cdf`).
  * Round 18: §3 (constants — `NEW_MV_CONTEXTS = 6`,
    `ZERO_MV_CONTEXTS = 2`, `REF_MV_CONTEXTS = 6`, `DRL_MODE_CONTEXTS = 3`,
    `IS_INTER_CONTEXTS = 4`, `COMP_INTER_CONTEXTS = 5`,
    `SKIP_MODE_CONTEXTS = 3`, `REF_CONTEXTS = 3`, `FWD_REFS = 4`,
    `BWD_REFS = 3`, `SINGLE_REFS = 7`, `UNIDIR_COMP_REFS = 4`,
    `COMP_REF_TYPE_CONTEXTS = 5`, `COMPOUND_MODES = 8`,
    `COMPOUND_MODE_CONTEXTS = 8`, `COMP_NEWMV_CTXS = 5`), §8.3.1 (the
    "set to a copy of `Default_*_Cdf`" init step for `NewMvCdf`,
    `ZeroMvCdf`, `RefMvCdf`, `DrlModeCdf`, `IsInterCdf`, `CompModeCdf`,
    `SkipModeCdf`, `CompRefCdf`, `CompBwdRefCdf`, `SingleRefCdf`,
    `CompoundModeCdf`, `CompRefTypeCdf`, `UniCompRefCdf`), §8.3.2 (the
    selection paragraphs — `new_mv: TileNewMvCdf[ NewMvContext ]`,
    `zero_mv: TileZeroMvCdf[ ZeroMvContext ]`,
    `ref_mv: TileRefMvCdf[ RefMvContext ]`,
    `drl_mode: TileDrlModeCdf[ DrlCtxStack[ idx ] ]`, the
    `is_inter` context ladder over `(AvailU, AvailL) × (AboveIntra,
    LeftIntra)`, `comp_mode: TileCompModeCdf[ ctx ]`, `skip_mode:
    TileSkipModeCdf[ ctx ]` with the neighbour `SkipModes[]` sum,
    `comp_ref{,_p1,_p2}: TileCompRefCdf[ ctx ][ 0..2 ]` with
    `ctx = ref_count_ctx(last12Count, last3GoldCount)` / `..(lastCount,
    last2Count)` / `..(last3Count, goldCount)`, `comp_bwdref{,_p1}:
    TileCompBwdRefCdf[ ctx ][ 0..1 ]` with `ctx = ref_count_ctx(
    brfarf2Count, arfCount)` / `..(brfCount, arf2Count)`,
    `single_ref_p{1..6}: TileSingleRefCdf[ ctx ][ 0..5 ]` (the
    cross-referenced `single_ref_p2` ↔ `comp_bwdref` / `_p3` ↔
    `comp_ref` / `_p4` ↔ `comp_ref_p1` / `_p5` ↔ `comp_ref_p2` / `_p6` ↔
    `comp_bwdref_p1` mappings), `compound_mode: TileCompoundModeCdf[ ctx ]`
    with `ctx = Compound_Mode_Ctx_Map[ RefMvContext >> 1 ][ Min(
    NewMvContext, COMP_NEWMV_CTXS - 1) ]`, `comp_ref_type:
    TileCompRefTypeCdf[ ctx ]` (taken as a precomputed index — the
    nine-branch `aboveCompInter` / `leftCompInter` / `is_samedir_ref_pair`
    ladder is the tile walk's responsibility), `uni_comp_ref{,_p1,_p2}:
    TileUniCompRefCdf[ ctx ][ 0..2 ]`), and §9.4 (default CDF table
    values for `Default_New_Mv_Cdf`, `Default_Zero_Mv_Cdf`,
    `Default_Ref_Mv_Cdf`, `Default_Drl_Mode_Cdf`, `Default_Is_Inter_Cdf`,
    `Default_Comp_Mode_Cdf`, `Default_Skip_Mode_Cdf`,
    `Default_Comp_Ref_Cdf`, `Default_Comp_Bwd_Ref_Cdf`,
    `Default_Single_Ref_Cdf`, `Default_Compound_Mode_Cdf`,
    `Default_Comp_Ref_Type_Cdf`, `Default_Uni_Comp_Ref_Cdf`, plus the
    `Compound_Mode_Ctx_Map[ 3 ][ COMP_NEWMV_CTXS ]` table).
  * Round 20: §3 (constants — `TX_SIZE_CONTEXTS = 3`, `TX_SIZES = 5`,
    `MAX_TX_DEPTH = 2`, `TXFM_PARTITION_CONTEXTS = 21`), §5.11.15
    (`read_tx_size` — the `MiSize > BLOCK_4X4 && allowSelect &&
    TxMode == TX_MODE_SELECT` gate on the `tx_depth` `S()` read +
    the `Max_Tx_Depth[ MiSize ]` driver), §5.11.16 (`read_block_tx_size`
    — the `read_var_tx_size` call site that drives `txfm_split`),
    §8.3.1 (the "set equal to a copy of `Default_*_Cdf`" init step
    for `Tx8x8Cdf`, `Tx16x16Cdf`, `Tx32x32Cdf`, `Tx64x64Cdf`,
    `TxfmSplitCdf`), §8.3.2 (the four-way `tx_depth` selection by
    `maxTxDepth` over `TileTx{8x8,16x16,32x32,64x64}Cdf[ ctx ]`, the
    `ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)` formula,
    the `txfm_split: TileTxfmSplitCdf[ ctx ]` selection with
    `ctx = (txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6
    + above + left`), §9.4 (default CDF table values for
    `Default_Tx_8x8_Cdf`, `Default_Tx_16x16_Cdf`,
    `Default_Tx_32x32_Cdf`, `Default_Tx_64x64_Cdf`,
    `Default_Txfm_Split_Cdf`).
  * Round 21: §3 (constants — `TX_TYPES = 16`, `TX_TYPES_SET2 = 12`,
    `TX_TYPES_SET3 = 2`), §5.11.47 (`transform_type` — the `set > 0
    && qindex > 0 && is_inter` gate on the `inter_tx_type` `S()`
    read + the `Tx_Type_Inter_Inv_Set{1,2,3}` inversion tables),
    §5.11.48 (`get_tx_set` — the `is_inter == 1` branch:
    `txSzSqrUp > TX_32X32 ⇒ TX_SET_DCTONLY` /
    `reduced_tx_set || txSzSqrUp == TX_32X32 ⇒ TX_SET_INTER_3` /
    `txSzSqr == TX_16X16 ⇒ TX_SET_INTER_2` / else `TX_SET_INTER_1`),
    §6.10.19 (`set` tag enumeration — `TX_SET_DCTONLY = 0`,
    `TX_SET_INTRA_1 = 1`, `TX_SET_INTRA_2 = 2`, `TX_SET_INTER_1 = 1`,
    `TX_SET_INTER_2 = 2`, `TX_SET_INTER_3 = 3`), §8.3.1 (the "set
    equal to a copy of `Default_Inter_Tx_Type_Set{1,2,3}_Cdf`" init
    step for `InterTxTypeSet{1,2,3}Cdf`), §8.3.2 (the three-way
    `inter_tx_type` selection by `set` over
    `TileInterTxTypeSet1Cdf[ Tx_Size_Sqr[ txSz ] ]` /
    `TileInterTxTypeSet2Cdf` /
    `TileInterTxTypeSet3Cdf[ Tx_Size_Sqr[ txSz ] ]`), §9.4 (default
    CDF table values for `Default_Inter_Tx_Type_Set1_Cdf`,
    `Default_Inter_Tx_Type_Set2_Cdf`,
    `Default_Inter_Tx_Type_Set3_Cdf`).
  * Round 22: §3 (constants — `INTERP_FILTERS = 3`,
    `INTERP_FILTER_CONTEXTS = 16`), §8.3.1 (the "set equal to a copy
    of `Default_Interp_Filter_Cdf`" init step for `InterpFilterCdf`),
    §8.3.2 (the four-branch `interp_filter` ctx formula —
    `ctx = ((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4` base,
    `leftType = aboveType = 3` initialisers, the
    `RefFrames[..][0|1] == RefFrame[0]` neighbour-matching gate that
    promotes the `InterpFilters[..][dir]` entry, and the
    match / left-NONE / above-NONE / distinct branches that fold
    `leftType` / `aboveType` / `3` into the ctx total; the
    `interp_filter: TileInterpFilterCdf[ ctx ]` selection), §9.4
    (default CDF table values for `Default_Interp_Filter_Cdf`).
  * Round 23: §3 (constants — `MOTION_MODES = 3`), §6.10.26
    (`motion_mode` semantics — `SIMPLE = 0`, `OBMC = 1`,
    `LOCALWARP = 2`), §8.3.1 (the "set equal to a copy of
    `Default_Motion_Mode_Cdf`" init step for `MotionModeCdf`),
    §8.3.2 (the `motion_mode: TileMotionModeCdf[ MiSize ]` selection
    — a straight `0..BLOCK_SIZES` index with no neighbour-context
    arithmetic), §9.4 (default CDF table values for
    `Default_Motion_Mode_Cdf` including the §9.4 note that
    first-dimension indices `0..=2` and `16..=17` are never reached
    by the §5.11.x `read_motion_mode` selection but are still
    transcribed full-width).
  * Round 24: §3 (constants — `COMPOUND_TYPES = 2`,
    `COMP_GROUP_IDX_CONTEXTS = 6`, `COMPOUND_IDX_CONTEXTS = 6`),
    §6.10.24 (`comp_group_idx` / `compound_idx` / `compound_type`
    semantics), §8.3.1 (the "set equal to a copy of
    `Default_Comp_Group_Idx_Cdf` / `Default_Compound_Idx_Cdf` /
    `Default_Compound_Type_Cdf`" init steps), §8.3.2 (the
    `comp_group_idx: TileCompGroupIdxCdf[ ctx ]` paragraph with its
    `ctx = Min(5, ctx)` neighbour clamp, the
    `compound_idx: TileCompoundIdxCdf[ ctx ]` paragraph with its
    `get_relative_dist` fwd/bck seed, and the
    `compound_type: TileCompoundTypeCdf[ MiSize ]` straight index),
    §9.4 (default CDF table values for the three tables including the
    §9.4 note that `Default_Compound_Type_Cdf` first-dimension indices
    `0..=2`, `10..=17` and `20..=21` are never used but are still
    transcribed full-width).
  * **Inter-frame intra-mode CDFs** (round 134): §3 (`BLOCK_SIZE_GROUPS`,
    `UV_INTRA_MODES_CFL_NOT_ALLOWED`, `UV_INTRA_MODES_CFL_ALLOWED`
    constant definitions), §8.3.1 (the "set equal to a copy of
    `Default_Y_Mode_Cdf` / `Default_Uv_Mode_Cfl_Not_Allowed_Cdf` /
    `Default_Uv_Mode_Cfl_Allowed_Cdf`" init steps for `YModeCdf` /
    `UVModeCflNotAllowedCdf` / `UVModeCflAllowedCdf`), §8.3.2 (the
    `y_mode: TileYModeCdf[ Size_Group[ MiSize ] ]` paragraph and the
    `uv_mode` paragraph selecting the cfl-allowed / cfl-not-allowed
    variant by the `Lossless` / `get_plane_residual_size` /
    `Max(Block_Width, Block_Height) <= 32` tests, then indexing by
    `YMode`), §8.3.2 `Size_Group[ BLOCK_SIZES ]` table, §9.4 (default
    CDF table values for the three tables).
  * **Angle-delta CDF** (round 135): §3 (`DIRECTIONAL_MODES`,
    `MAX_ANGLE_DELTA`, `V_PRED`), §8.3.1 (the "`AngleDeltaCdf` is set
    to a copy of `Default_Angle_Delta_Cdf`" init step), §8.3.2 (the
    `TileAngleDeltaCdf[ YMode - V_PRED ]` / `[ UVMode - V_PRED ]`
    selections for `angle_delta_y` / `angle_delta_uv`), §9.4 (default
    CDF table values for `Default_Angle_Delta_Cdf`).
  * **Coefficient-token entry CDFs** (round 136): §3 (`PLANE_TYPES`,
    `COEFF_CDF_Q_CTXS`, `TXB_SKIP_CONTEXTS`, `EOB_COEF_CONTEXTS`,
    `DC_SIGN_CONTEXTS` constant definitions), §8.3.1 `init_coeff_cdfs`
    (the `base_q_idx` → `idx` derivation and the "set to a copy of
    `Default_Txb_Skip_Cdf[ idx ]` / `Default_Eob_Pt_*_Cdf[ idx ]` /
    `Default_Eob_Extra_Cdf[ idx ]` / `Default_Dc_Sign_Cdf[ idx ]`"
    reset steps), §9.4 (default CDF table values for
    `Default_Txb_Skip_Cdf`, `Default_Eob_Pt_{16,32,64,128,256,512,
    1024}_Cdf`, `Default_Eob_Extra_Cdf` and `Default_Dc_Sign_Cdf`,
    the last in the `128 * N` fixed-point form).
  * **Intra-frame transform-type CDFs** (round 137): §3 (constants —
    `TX_SET_INTRA_1 = 1`, `TX_SET_INTRA_2 = 2`,
    `TX_TYPES_INTRA_SET1 = 7`, `TX_TYPES_INTRA_SET2 = 5`,
    `INTRA_TX_TYPE_SET1_SIZES = 2`, `INTRA_TX_TYPE_SET2_SIZES = 3`),
    §5.11.48 `get_tx_set()` (the `is_inter == 0` branch routing
    `txSzSqrUp >= TX_32X32` → `TX_SET_DCTONLY`, `reduced_tx_set ||
    txSzSqr == TX_16X16` → `TX_SET_INTRA_2`, else → `TX_SET_INTRA_1`),
    §6.10.19 (the `Tx_Type_Intra_Inv_Set1` 7-entry / `Tx_Type_Intra_Inv_Set2`
    5-entry transform-type enumeration), §8.3.1 (the "set equal to a
    copy of `Default_Intra_Tx_Type_Set1_Cdf` / `Default_Intra_Tx_Type_Set2_Cdf`"
    init steps for `IntraTxTypeSet1Cdf` / `IntraTxTypeSet2Cdf`), §8.3.2
    (the `intra_tx_type: TileIntraTxTypeSet{1,2}Cdf[ Tx_Size_Sqr[ txSz ]
    ][ intraDir ]` two-way switch and the `intraDir` derivation —
    `Filter_Intra_Mode_To_Intra_Dir[ filter_intra_mode ]` when
    `use_filter_intra == 1`, else `YMode`), §8.3.2
    `Filter_Intra_Mode_To_Intra_Dir[ INTRA_FILTER_MODES ]` table,
    §9.4 (default CDF table values for the two tables).
  * **`coeff_base_eob` CDF** (round 138): §3 (`SIG_COEF_CONTEXTS_EOB = 4`
    constant definition), §5.11 `coeff_base_eob` semantics (the
    "base level is `coeff_base_eob + 1`; only base levels 1, 2, or 3
    can be coded" note constraining the symbol set to three values),
    §8.3.1 `init_coeff_cdfs` (the "`CoeffBaseEobCdf` is set to a
    copy of `Default_Coeff_Base_Eob_Cdf[ idx ]`" reset step), §8.3.2
    (the `coeff_base_eob: TileCoeffBaseEobCdf[ txSzCtx ][ ptype ][
    ctx ]` selection plus the deferred-to-tile-walk
    `ctx = get_coeff_base_ctx(...) - SIG_COEF_CONTEXTS +
    SIG_COEF_CONTEXTS_EOB` derivation), §9.4 (default CDF table
    values for `Default_Coeff_Base_Eob_Cdf`).
  * **`coeff_base` CDF** (round 139): §3 (`SIG_COEF_CONTEXTS = 42`
    constant definition; the `SIG_COEF_CONTEXTS_2D = 26` partition
    tag that splits the 2D-scan prefix from the 1D horizontal- /
    vertical-only tails), §5.11 `coeff_base` semantics (the "level
    is `coeff_base`" assignment for non-EOB coefficients),
    §8.3.1 `init_coeff_cdfs` (the "`CoeffBaseCdf` is set to a copy
    of `Default_Coeff_Base_Cdf[ idx ]`" reset step), §8.3.2 (the
    `coeff_base: TileCoeffBaseCdf[ txSzCtx ][ ptype ][ ctx ]`
    selection plus the deferred-to-tile-walk
    `ctx = get_coeff_base_ctx(...)` derivation), §9.4 (default CDF
    table values for `Default_Coeff_Base_Cdf`).
  * **`coeff_br` CDF** (round 140): §3 (`LEVEL_CONTEXTS = 21` and
    `BR_CDF_SIZE = 4` constant definitions), §5.11.39 `coeff_br`
    semantics (the `for idx = 0; idx < COEFF_BASE_RANGE /
    (BR_CDF_SIZE - 1); idx++` per-coefficient stacking loop with
    the `coeff_br < (BR_CDF_SIZE - 1)` early-break),
    §8.3.1 `init_coeff_cdfs` (the "`CoeffBrCdf` is set to a copy of
    `Default_Coeff_Br_Cdf[ idx ]`" reset step), §8.3.2 (the
    `coeff_br: TileCoeffBrCdf[ Min(txSzCtx, TX_32X32) ][ ptype ][
    ctx ]` selection with the `TX_32X32 = 3` clamp, plus the
    deferred-to-tile-walk `ctx = get_br_ctx(...)` derivation),
    §9.4 (default CDF table values for `Default_Coeff_Br_Cdf`).
  * **`get_coeff_base_ctx` / `get_br_ctx` helpers** (round 141):
    §3 (`TX_SIZES_ALL = 19`, `SIG_COEF_CONTEXTS_2D = 26`,
    `SIG_REF_DIFF_OFFSET_NUM = 5`, `NUM_BASE_LEVELS = 2`,
    `COEFF_BASE_RANGE = 12`, `TX_CLASS_{2D, HORIZ, VERT}` tag
    enumeration with values `{ 0, 1, 2 }`, and the `V_DCT = 10`,
    `H_DCT = 11`, `V_ADST = 12`, `H_ADST = 13`, `V_FLIPADST = 14`,
    `H_FLIPADST = 15` transform-type enumeration used by
    `get_tx_class`), §8.3.2 (the `get_coeff_base_ctx()` and
    `get_br_ctx()` function bodies — `isEob` branch with the
    `(height << bwl) / 8` / `/ 4` boundaries, the
    `Sig_Ref_Diff_Offset` neighbour scan with the
    `Min(Abs(Quant[..]), 3)` per-neighbour clamp, the 2D
    `row == 0 && col == 0 -> 0` short-circuit, the
    `Coeff_Base_Ctx_Offset[txSz][Min(row, 4)][Min(col, 4)]` 2D
    offset, the `Coeff_Base_Pos_Ctx_Offset[Min(idx, 2)]` 1D
    branch, the `Mag_Ref_Offset_With_Tx_Class` three-neighbour
    scan with the `Min(Quant[..], COEFF_BASE_RANGE + NUM_BASE_LEVELS
    + 1)` clamp, the `Min((mag + 1) >> 1, 6)` magnitude bucket,
    the `pos == 0` short-circuit, and the per-class `+7` /
    `+14` offsets — plus the `get_tx_class()` function body and
    the `coeff_base_eob` `ctx = get_coeff_base_ctx(.., 1) -
    SIG_COEF_CONTEXTS + SIG_COEF_CONTEXTS_EOB` reduction),
    §"Additional tables" (`Tx_Width[TX_SIZES_ALL]`,
    `Tx_Height[TX_SIZES_ALL]`, `Tx_Width_Log2[TX_SIZES_ALL]`,
    `Adjusted_Tx_Size[TX_SIZES_ALL]`,
    `Sig_Ref_Diff_Offset[3][SIG_REF_DIFF_OFFSET_NUM][2]`,
    `Coeff_Base_Ctx_Offset[TX_SIZES_ALL][5][5]`,
    `Coeff_Base_Pos_Ctx_Offset[3]`,
    `Mag_Ref_Offset_With_Tx_Class[3][3][2]`).
  * **Inter-intra CDFs** (round 143): §3 (`INTERINTRA_MODES = 4`),
    §5.11.28 (`read_interintra_mode` — the `interintra` /
    `interintra_mode` / `wedge_interintra` reads, including the
    `BLOCK_8X8 <= MiSize <= BLOCK_32X32` syntax gate),
    §6.10.27 (`II_DC_PRED` / `II_V_PRED` / `II_H_PRED` /
    `II_SMOOTH_PRED` enumeration; `wedge_interintra` semantics),
    §8.3.1 (`init_non_coeff_cdfs` — `InterIntraCdf` /
    `InterIntraModeCdf` / `WedgeInterIntraCdf` "is set equal to a copy
    of `Default_*`"), §8.3.2 (`interintra` /
    `interintra_mode` / `wedge_interintra` paragraphs — the
    `ctx = Size_Group[ MiSize ] - 1` mapping for the first two and
    the `TileWedgeInterIntraCdf[ MiSize ]` straight index for the
    third), §8.3.2 `Size_Group[ BLOCK_SIZES ]` table (from round
    134), §9.4 listings (`Default_Inter_Intra_Cdf`,
    `Default_Inter_Intra_Mode_Cdf`, `Default_Wedge_Inter_Intra_Cdf`
    on pp.434–436 of the spec PDF — including the latter's note that
    only first-dimension indices 3..=9 are used).
  * **`compute_tx_type` derivation** (round 142): §3 / §6.10.16 (the
    `TX_4X4 = 0`, `TX_8X8 = 1`, `TX_16X16 = 2`, `TX_32X32 = 3`,
    `TX_64X64 = 4` `TxSize` ordinals — previously used only as
    locally-scoped `const TX_*` shadows inside the §5.11.48
    helpers — and the `TX_SET_TYPES_INTRA = 3` /
    `TX_SET_TYPES_INTER = 4` row-count constants for the
    `Tx_Type_In_Set_*` tables), §6.10.19 (the `DCT_DCT = 0` through
    `H_FLIPADST = 15` 16-entry transform-type enumeration —
    previously the `V_DCT..H_FLIPADST` tail only), §5.11.40
    (`compute_tx_type()` function body — the `Lossless ||
    Tx_Size_Sqr_Up[ txSz ] > TX_32X32 -> DCT_DCT` short-circuit, the
    `plane == 0 -> TxTypes[ blockY ][ blockX ]` luma branch, the
    `is_inter` chroma `Max(MiRow, blockY << subsampling_y)` /
    `Max(MiCol, blockX << subsampling_x)` lift into the
    `TxTypes[..]` cache, the `is_tx_type_in_set` admission filter
    with the `!is_tx_type_in_set -> DCT_DCT` fallback, the intra
    chroma `Mode_To_Txfm[ UVMode ]` path with the same filter — and
    the `Tx_Type_In_Set_Intra[ 3 ][ TX_TYPES ]` /
    `Tx_Type_In_Set_Inter[ 4 ][ TX_TYPES ]` admission-flag tables
    transcribed verbatim from the spec listing), §"Additional
    tables" (`Tx_Size_Sqr_Up[ TX_SIZES_ALL ]` — `t -> Max(w, h)`-
    sided square — and `Mode_To_Txfm[ UV_INTRA_MODES_CFL_ALLOWED ]`
    — chroma-mode default-tx-type table).
  * **`intra_segment_id` syntax element** (round 160): §3 (the
    `MAX_SEGMENTS = 8` upper bound that sizes the §6.8.2
    `LosslessArray[]` table the §5.11.8 footer indexes), §5.11.7
    (the §5.11.7 `intra_frame_mode_info` caller context — the two
    `SegIdPreSkip` arms that each call `intra_segment_id()`),
    §5.11.8 (the function body itself — the
    `segmentation_enabled ⇒ read_segment_id() : segment_id = 0`
    dispatch and the `Lossless = LosslessArray[ segment_id ]`
    footer), §5.11.9 (the §5.11.9 `read_segment_id` body the
    `segmentation_enabled = true` arm delegates to, including the
    §5.11.5 grid-fill footer line `SegmentIds[ r + y ][ c + x ] =
    segment_id` — accepted as the r159 implementation), and §6.8.2
    (the `LosslessArray[ segmentId ] = qindex == 0 && DeltaQYDc ==
    0 && DeltaQUAc == 0 && DeltaQUDc == 0 && DeltaQVAc == 0 &&
    DeltaQVDc == 0` per-segment derivation that produces the
    eight-entry table the walker indexes by `segment_id` — accepted
    as a `&[bool; MAX_SEGMENTS]` parameter so the walker stays
    segmentation-state-free).
  * **`read_segment_id` syntax element** (round 159): §3 (the
    `MAX_SEGMENTS = 8` upper bound capping the §5.11.9 alphabet
    and the `SEGMENT_ID_CONTEXTS = 3` constant bounding the §8.3.2
    ctx derivation), §5.9.14 (the `LastActiveSegId` trailing
    derivation that supplies `max = LastActiveSegId + 1` to the
    §5.11.9 `neg_deinterleave` call — accepted as a `u8` parameter
    so the walker stays segmentation-state-free), §5.11.5 (the
    per-block grid-fill footer line `SegmentIds[ r + y ][ c + x ] =
    segment_id` the walker stamps over the `bw4 * bh4` footprint),
    §5.11.8 / §5.11.19 (the two caller contexts: intra-frame /
    inter-frame `*_segment_id` syntax, both of which route through
    `read_segment_id` when `segmentation_enabled`), §5.11.9 (the
    function body itself — the three neighbour reads `prevUL` /
    `prevU` / `prevL`, the four-arm `pred` cascade `prevU == -1 ⇒
    prevL / 0`; `prevL == -1 ⇒ prevU`; `prevUL == prevU ⇒ prevU`;
    else `prevL`, the `skip ⇒ segment_id = pred` short-circuit, the
    `S()` read against `TileSegmentIdCdf[ ctx ]` plus
    `neg_deinterleave( diff, pred, max )` reconstruction), and
    §8.3.2 (the `segment_id` ctx formula `prevUL < 0 ⇒ 0; (prevUL
    == prevU && prevUL == prevL) ⇒ 2; (prevUL == prevU || prevUL
    == prevL || prevU == prevL) ⇒ 1; else 0` — already lifted as
    `segment_id_ctx`).
  * **`read_is_inter` syntax element** (round 158): §3 (the
    `IS_INTER_CONTEXTS = 4` constant bounding the §8.3.2 ctx
    derivation, the implicit `INTRA_FRAME = 0` ordinal under which
    the §5.11.20 Arm 2 comparison `FeatureData[..] != INTRA_FRAME`
    encodes "the segmentation override selected an inter reference",
    the `SEG_LVL_REF_FRAME = 5` / `SEG_LVL_GLOBALMV = 7` segment-
    feature indices that gate Arms 2 and 3), §5.11.5 (the per-block
    grid-fill footer line `IsInters[ r + y ][ c + x ] = is_inter`
    that the walker stamps over the `bw4 * bh4` footprint after the
    arm dispatch), §5.11.18 (the §5.11.20 caller context: the
    `LeftRefFrame[0] = AvailL ? RefFrames[..][0] : INTRA_FRAME` /
    `LeftIntra = LeftRefFrame[0] <= INTRA_FRAME` neighbour-
    derivation that maps onto an `Option<bool>` to `is_inter_ctx`,
    and the `read_is_inter()` call-site placement immediately after
    `read_delta_lf()` and before the `is_inter ?
    inter_block_mode_info() : intra_block_mode_info()` dispatch),
    §5.11.20 (the four-arm dispatch body itself —
    `skip_mode` ⇒ 1; `seg_feature_active(SEG_LVL_REF_FRAME)` ⇒
    `FeatureData != INTRA_FRAME`; `seg_feature_active(SEG_LVL_GLOBALMV)`
    ⇒ 1; else `S()` — and the §8.3.2 ctx formula at av1-spec p.365:
    `AvailU && AvailL ⇒ (LeftIntra && AboveIntra) ? 3 : LeftIntra
    || AboveIntra`; `AvailU ^ AvailL ⇒ 2 * (AvailU ? AboveIntra :
    LeftIntra)`; else 0).
  * **`read_cdef` syntax element + `clear_cdef` reset** (round 157):
    §3 (the `BLOCK_64X64 = 12` ordinal that anchors the §5.11.55
    `cdefSize4 = Num_4x4_Blocks_Wide[ BLOCK_64X64 ] = 16` stride; the
    `Num_4x4_Blocks_Wide[ ]` / `Num_4x4_Blocks_High[ ]` tables used
    by the §5.11.56 grid-fill loop, transcribed in round 148 from
    §9.3 p.400), §5.9.19 (the `cdef_bits` `f(2)` upper bound — so
    `cdef_bits ∈ 0..=3` and the decoded `L(cdef_bits)` value fits in
    `i8`; the `CodedLossless || allow_intrabc || !enable_cdef ⇒
    cdef_bits = 0` short-circuit that pairs with §5.11.56's
    short-circuit set), §5.11.2 (the per-superblock `clear_cdef( r,
    c )` call site at the top of the tile-walk inner loop —
    immediately before `decode_partition`), §5.11.55 (the
    `clear_cdef()` function body — the unconditional `cdef_idx[ r ][
    c ] = -1` stamp plus the `use_128x128_superblock` three-extra-
    anchor stamps), §5.11.56 (the `read_cdef()` function body — the
    four-way short-circuit `skip || CodedLossless || !enable_cdef ||
    allow_intrabc`, the `cdefMask4 = ~(cdefSize4 - 1)` anchor mask,
    the `cdef_idx[ r ][ c ] == -1` first-leaf gate that gates the
    `L(cdef_bits)` literal read, and the `cdefSize4`-strided
    grid-fill loop over the leaf's `(w4, h4)` footprint), §6.10.40
    (the `cdef_idx` semantics — "A value of -1 means that CDEF is
    disabled for that block").
  * **`read_delta_lf` syntax element** (round 156): §3 (the
    `DELTA_LF_SMALL = 3` escape sentinel against which `delta_lf_abs`
    is compared to trigger the `delta_lf_rem_bits` /
    `delta_lf_abs_bits` ladder; the `FRAME_LF_COUNT = 4` upper bound
    on the per-superblock iteration loop and on the size of the
    `DeltaLF[ ]` accumulator; the `MAX_LOOP_FILTER = 63` Clip3
    bound — `cdf::MAX_LOOP_FILTER` is an `i32` twin of the
    pre-existing §5.9.11 `uncompressed_header_tail::MAX_LOOP_FILTER`
    `i16` so the signed `DeltaLF[ i ] + (reducedDeltaLfLevel <<
    delta_lf_res)` arithmetic stays in a single integer type), §5.11.2
    (`for ( i = 0; i < FRAME_LF_COUNT; i++ ) DeltaLF[ i ] = 0` — the
    tile-entry reset honoured by `reset_current_delta_lf`), §5.11.13
    (`read_delta_lf()` function body — the `sbSize`-derived skip
    short-circuit identical to §5.11.12, the
    `ReadDeltas && delta_lf_present` outer gate, the
    `frameLfCount = delta_lf_multi ?
        ((NumPlanes > 1) ? FRAME_LF_COUNT : FRAME_LF_COUNT - 2) : 1`
    derivation, the per-iteration `S()` against `TileDeltaLFCdf` or
    `TileDeltaLFMultiCdf[ i ]`, the `DELTA_LF_SMALL` escape ladder,
    the `delta_lf_sign_bit` sign read, and the `Clip3(-MAX_LOOP_FILTER,
    MAX_LOOP_FILTER, ...)` update), §6.10.4 (the `ReadDeltas`
    derivation as `delta_q_present` AND first-block-of-superblock,
    re-used identically for the §5.11.13 outer gate — accepted from
    the caller per the §5.11.12 convention), §8.3.1 (the
    `init_non_coeff_cdfs` steps "`DeltaLFCdf` is set to a copy of
    `Default_Delta_Lf_Cdf`" and "`DeltaLFMultiCdf[ i ]` is set to a
    copy of `Default_Delta_Lf_Cdf` for `i = 0..FRAME_LF_COUNT-1`"),
    §8.3.2 (the `delta_lf_abs` cdf-selection paragraph — `cdf` is
    `TileDeltaLFCdf` when `delta_lf_multi == 0` and
    `TileDeltaLFMultiCdf[ i ]` when `delta_lf_multi == 1`; no
    context index in either case), §9.4 (default CDF table values
    for `Default_Delta_Lf_Cdf[ DELTA_LF_SMALL + 2 ]` on av1-spec
    p.431).
  * **`read_delta_qindex` syntax element** (round 155): §3 (the
    `DELTA_Q_SMALL = 3` constant — the §5.11.12 escape sentinel
    against which `delta_q_abs` is compared to trigger the
    `delta_q_rem_bits` / `delta_q_abs_bits` ladder), §5.11.12
    (`read_delta_qindex()` function body — the `sbSize =
    use_128x128_superblock ? BLOCK_128X128 : BLOCK_64X64` derivation,
    the `MiSize == sbSize && skip` superblock-skip short-circuit,
    the `ReadDeltas` outer gate, the `delta_q_abs` `S()` read with
    the `DELTA_Q_SMALL` escape ladder
    `delta_q_rem_bits = L(3); delta_q_rem_bits++;
    delta_q_abs_bits = L(delta_q_rem_bits); delta_q_abs =
    delta_q_abs_bits + (1 << delta_q_rem_bits) + 1`, the
    `delta_q_sign_bit = L(1)` sign read, the
    `reducedDeltaQIndex = delta_q_sign_bit ? -delta_q_abs :
    delta_q_abs` derivation, and the
    `CurrentQIndex = Clip3(1, 255, CurrentQIndex +
    (reducedDeltaQIndex << delta_q_res))` update), §6.10.4 (the
    `ReadDeltas` derivation as `delta_q_present` AND
    first-block-of-superblock — accepted from the caller as a
    pre-computed boolean rather than re-derived inside the walker),
    §8.3.1 (the `init_non_coeff_cdfs` step "`DeltaQCdf` is set to
    a copy of `Default_Delta_Q_Cdf`"), §8.3.2 (the `delta_q_abs:
    the cdf is given by TileDeltaQCdf` paragraph — single CDF row
    with no context index), §9.4 (default CDF table values for
    `Default_Delta_Q_Cdf[ DELTA_Q_SMALL + 2 ]` on av1-spec p.431).
* Fixtures under `docs/video/av1/fixtures/` (bitstreams + trace
  files emitted by an AV1_TRACE-patched FFmpeg + libdav1d host;
  treated as opaque ground-truth, no source consulted).

No external library source — libaom, dav1d, libgav1, rav1e, SVT-AV1,
FFmpeg AV1 — was consulted. No third-party crate that wraps or
implements the same format was consulted. No web search was
performed.

## License

MIT. See `LICENSE`.
