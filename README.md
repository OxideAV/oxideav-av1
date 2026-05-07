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

## Codec ID

Registered as `"av1"`. The capability tag is `"av1_sw_parse"` —
surfacing in `oxideav list` style output that this build parses the
bitstream envelope but does not yet reconstruct pixels.

## License

MIT — see [LICENSE](LICENSE).
