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

## What's missing

`Av1Decoder::receive_frame()` returns `Error::Unsupported`. To produce
pixels the following still need to land (each error message in the
crate names its §ref):

- **Default CDF tables** (§9.4.1 / §9.4.2) — ~9 KB of static probability
  tables required by every symbol decode.
- **Partition quadtree** walk (§5.11.4) and the remaining partition
  shapes beyond `PARTITION_NONE`.
- **`decode_block()` per-symbol syntax** — `skip`, `tx_size`, `tx_type`,
  `intra_y_mode`, `uv_mode`, angle deltas, CFL, palette, coefficient
  decode (§5.11.5 – §5.11.39).
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
- **Cross-frame reference state** — `frame_size_with_refs()` and the
  whole inter-frame machinery. Key and intra-only frames work as far as
  tile boundaries; inter frames that rely on `frame_size_with_refs()`
  currently return `Error::Unsupported`.
- **10+ bit-depth decode** — the pixel primitives are `u8`-only; 10 /
  12-bit paths land together with the coefficient decode.

Encoder: out of scope. This crate is decoder-side.

## Codec ID

Registered as `"av1"`. The capability tag is `"av1_sw_parse"` —
surfacing in `oxideav list` style output that this build parses the
bitstream envelope but does not yet reconstruct pixels.

## License

MIT — see [LICENSE](LICENSE).
