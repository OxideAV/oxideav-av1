# oxideav-av1

Pure-Rust AV1 (AOMedia Video 1) codec — a clean-room implementation
built from the public AV1 Bitstream & Decoding Process Specification.
Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework.

## Status

Clean-room rebuild in progress. The bitstream-syntax and header layers
are broadly complete (OBU framing, sequence header, full
uncompressed-frame-header syntax tree, tile info), and an intra-only
decode/encode pixel pipeline is wired end-to-end for a constrained
parameter set. Inter prediction and full-feature reconstruction are
not yet covered.

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
- In-loop / post passes (loop filter, CDEF, loop restoration,
  superres, film grain) are present as modules but disabled under this
  parameter set.

The inter-prediction reconstruction layer covers the §7.11.3.1 single-
reference translational (SIMPLE), compound (AVERAGE / DISTANCE / WEDGE /
DIFFWTD), and inter-intra arms, plus the §7.11.3.5 **warped-motion**
(LOCALWARP `useWarp == 1` / GLOBAL_GLOBALMV `useWarp == 2`) arm —
`reconstruct_inter_block_warp` and its `PartitionWalker` bridge
(`reconstruct_inter_block_warp_into_curr_frame`) drive `block_warp` into
`CurrFrame[plane]`, and the §5.11.33 frame walk dispatches a decoded
`motion_mode == WARPED_CAUSAL` leaf to the warp path (via the opt-in
`InterModeInfoGrid.warp` context). §7.11.3.9-10 OBMC remains a leaf-only
layer not yet wired into the frame walk.

The public `encode_av1` entry takes the constrained
`[8, 64]`-per-axis lossless case; wider extents, lossy quant, and
monochrome are reachable through the crate-public `encoder::*` driver
functions. Streams outside the supported scope return a typed `Error`
(commonly `Error::PartitionWalkOutOfRange`).

### Not yet supported

- Inter prediction / reference-frame management.
- Multi-tile reconstruction beyond the single-tile decode path.
- 10/12-bit and 4:2:2 / 4:4:4 reconstruction.
- Registration as a live codec in the runtime registry.

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
