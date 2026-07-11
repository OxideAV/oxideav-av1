//! `tile_group_obu()` framing **writer** — §5.11.1.
//!
//! Builds the byte-aligned §5.11.1 tile-group OBU body around a
//! caller-supplied list of per-tile entropy payloads. Each payload is
//! the raw output of a [`SymbolWriter::finish`] call for one tile,
//! including the §8.2.4 `exit_symbol`-produced trailing zero pads —
//! the wrapper accounts for those bytes in the `tile_size_minus_1`
//! field per the §6.10.1 note:
//!
//! > This size includes any padding bytes if added by the exit
//! > process for the Symbol decoder. The size does not include the
//! > bytes used for `tile_size_minus_1` or syntax elements sent
//! > before `tile_size_minus_1`. For the last tile in the tile group,
//! > `tileSize` is computed instead of being read and includes the
//! > OBU trailing bits.
//!
//! What this module emits, in order (per §5.11.1):
//!
//!   1. `tile_start_and_end_present_flag` (`f(1)`) — only when
//!      `num_tiles > 1`. Set per [`TileGroupObu::start_and_end_present`].
//!   2. `tg_start` / `tg_end` (`f(tileBits)` each), where `tileBits
//!      = TileColsLog2 + TileRowsLog2` — only when the flag was
//!      written **and** is `1`.
//!   3. `byte_alignment()` (`f(1)` zero pads to the next byte
//!      boundary).
//!   4. For each tile **except the last** of the group: `tile_size_minus_1`
//!      (`le(TileSizeBytes)`), followed by the tile's `tileSize` bytes
//!      of entropy-coded payload.
//!   5. The last tile's payload, with no size field — `tileSize` is
//!      computed by the decoder from the remaining OBU body length.
//!
//! `TileSizeBytes` (1..=4) and `TileColsLog2` / `TileRowsLog2` come
//! from the §5.9.15 tile-info portion of the frame header; the writer
//! takes them as constructor inputs because tile-group OBUs by
//! themselves do not re-derive them.
//!
//! What this module does **not** do:
//!
//!   * Emit any per-block syntax. The caller is responsible for
//!     producing each `TilePayload::bytes` via the entropy encoder
//!     ([`SymbolWriter`]).
//!   * Frame the result inside a §5.3 OBU wrapper. Callers wrap the
//!     bytes in an `OBU_TILE_GROUP` via
//!     [`crate::encoder::obu::write_obu_with_size`] (which knows the
//!     §5.3.1 trailer is skipped for `OBU_TILE_GROUP`).
//!   * Re-implement [`SymbolWriter::finish`]'s zero-pad / `exit_symbol`
//!     accounting. Each `TilePayload::bytes` must already include
//!     those bytes — the wrapper just stitches sizes around them.
//!
//! [`SymbolWriter`]: crate::encoder::symbol_writer::SymbolWriter

use crate::encoder::bitwriter::BitWriter;
use crate::Error;

/// One tile's entropy-coded payload as it appears inside the §5.11.1
/// `tile_group_obu` body. `bytes` is the full output of a
/// [`SymbolWriter::finish`] call for the tile, including any §8.2.4
/// trailing zero padding the §8.2 decoder will consume.
///
/// [`SymbolWriter::finish`]: crate::encoder::symbol_writer::SymbolWriter::finish
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePayload {
    /// Entropy-coded bytes — must be a complete §8.2 partition
    /// (`init_symbol(sz)` over `bytes.len()` would decode it cleanly).
    pub bytes: Vec<u8>,
}

impl TilePayload {
    /// Convenience constructor.
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

/// §5.11.1 tile-group OBU descriptor.
///
/// `num_tiles` is the `TileCols * TileRows` invariant from the active
/// frame header. `tile_cols_log2` / `tile_rows_log2` likewise carry
/// the §5.9.15 tile-info dimensions; together they form `tileBits` for
/// the `tg_start` / `tg_end` `f(tileBits)` writes when the
/// `tile_start_and_end_present_flag` path is taken.
///
/// `tile_size_bytes` is `TileSizeBytes` from §6.8.14
/// (`tile_size_bytes_minus_1 + 1`, 1..=4) — the number of bytes used
/// to encode each non-last `tile_size_minus_1`.
#[derive(Debug, Clone)]
pub struct TileGroupObu {
    /// Total tiles in the frame (`NumTiles = TileCols * TileRows`).
    pub num_tiles: u32,
    /// `TileColsLog2` from §5.9.15. Sum with `tile_rows_log2` gives
    /// `tileBits` per §5.11.1.
    pub tile_cols_log2: u32,
    /// `TileRowsLog2` from §5.9.15.
    pub tile_rows_log2: u32,
    /// `TileSizeBytes` per §6.8.14 — 1..=4.
    pub tile_size_bytes: u32,
    /// `tg_start` per §5.11.1.
    pub tg_start: u32,
    /// `tg_end` per §5.11.1 — must be `>= tg_start`, and on the last
    /// tile group of a frame must equal `num_tiles - 1` per §6.10.1.
    pub tg_end: u32,
    /// `tile_start_and_end_present_flag` per §5.11.1. Honoured only
    /// when `num_tiles > 1`; ignored otherwise (the writer skips the
    /// `f(1)` per the §5.11.1 conditional).
    pub start_and_end_present: bool,
    /// One [`TilePayload`] per tile in `tg_start..=tg_end`. Caller
    /// must supply exactly `tg_end - tg_start + 1` entries; the writer
    /// debug-asserts this.
    pub tiles: Vec<TilePayload>,
}

impl TileGroupObu {
    /// Construct a tile-group OBU that covers the entire frame
    /// (`tg_start = 0`, `tg_end = num_tiles - 1`, flag set to `0` so
    /// the §5.11.1 conditional collapses to "no start/end fields").
    /// This is the §6.10.1 single-tile-group case.
    pub fn whole_frame(
        num_tiles: u32,
        tile_cols_log2: u32,
        tile_rows_log2: u32,
        tile_size_bytes: u32,
        tiles: Vec<TilePayload>,
    ) -> Self {
        debug_assert!(num_tiles >= 1, "§5.9.15 NumTiles >= 1");
        debug_assert!((1..=4).contains(&tile_size_bytes));
        debug_assert!(
            tiles.len() == num_tiles as usize,
            "whole_frame expects num_tiles payloads ({} != {})",
            tiles.len(),
            num_tiles
        );
        Self {
            num_tiles,
            tile_cols_log2,
            tile_rows_log2,
            tile_size_bytes,
            tg_start: 0,
            tg_end: num_tiles - 1,
            start_and_end_present: false,
            tiles,
        }
    }
}

/// `TileGroupObuWriter` — emits a §5.11.1 tile-group body for the
/// supplied [`TileGroupObu`] descriptor.
///
/// The returned `Vec<u8>` is the byte-aligned body the §5.3.1 OBU
/// wrapper consumes; callers pass it through
/// [`crate::encoder::obu::write_obu_with_size`] with
/// [`crate::obu::ObuType::TileGroup`] to produce the full
/// `OBU_TILE_GROUP`.
#[derive(Debug, Default)]
pub struct TileGroupObuWriter;

impl TileGroupObuWriter {
    /// Emit the §5.11.1 body. Returns [`Error::NotImplemented`]-free
    /// on the §5.11.1 paths covered (intra & inter alike; the body
    /// itself is structurally identical).
    ///
    /// Conformance checks (debug-asserted; the spec defers them to
    /// the §6.10.1 / §6.10.2 semantics):
    ///
    ///   * `tg_end >= tg_start` (§6.10.1).
    ///   * `tg_end <= num_tiles - 1` (§6.10.1; the conformance rule
    ///     that the last tile group has `tg_end == num_tiles - 1` is
    ///     up to the caller — we accept any conformant slice).
    ///   * `tiles.len() == tg_end - tg_start + 1`.
    ///   * Each non-last tile's `bytes.len()` fits in
    ///     `tile_size_bytes` (i.e. `bytes.len() - 1 < 1 <<
    ///     (8 * tile_size_bytes)` — `tile_size_minus_1` is the
    ///     written value).
    pub fn write(&self, obu: &TileGroupObu) -> Result<Vec<u8>, Error> {
        debug_assert!(obu.tg_end >= obu.tg_start, "§6.10.1 tg_end >= tg_start");
        debug_assert!(
            obu.tg_end < obu.num_tiles,
            "§6.10.1 tg_end < num_tiles ({} >= {})",
            obu.tg_end,
            obu.num_tiles
        );
        let expected_len = (obu.tg_end - obu.tg_start + 1) as usize;
        debug_assert!(
            obu.tiles.len() == expected_len,
            "tiles len {} does not match tg range [{}, {}] (expected {})",
            obu.tiles.len(),
            obu.tg_start,
            obu.tg_end,
            expected_len
        );
        debug_assert!(
            (1..=4).contains(&obu.tile_size_bytes),
            "§6.8.14 TileSizeBytes is 1..=4"
        );

        let mut bw = BitWriter::new();

        // §5.11.1 header.
        if obu.num_tiles > 1 {
            // tile_start_and_end_present_flag — f(1).
            bw.write_bits(1, u64::from(obu.start_and_end_present));
            if obu.start_and_end_present {
                // tileBits = TileColsLog2 + TileRowsLog2.
                let tile_bits = obu.tile_cols_log2 + obu.tile_rows_log2;
                // tg_start — f(tileBits). tg_end — f(tileBits). Both
                // are debug-asserted to fit.
                debug_assert!(
                    tile_bits == 0 || obu.tg_start < (1u32 << tile_bits),
                    "tg_start {} >= 1 << tileBits {}",
                    obu.tg_start,
                    tile_bits
                );
                debug_assert!(
                    tile_bits == 0 || obu.tg_end < (1u32 << tile_bits),
                    "tg_end {} >= 1 << tileBits {}",
                    obu.tg_end,
                    tile_bits
                );
                bw.write_bits(tile_bits, u64::from(obu.tg_start));
                bw.write_bits(tile_bits, u64::from(obu.tg_end));
            }
        }

        // §5.11.1 byte_alignment() — zero pads to the next byte boundary.
        bw.byte_align();
        debug_assert!(bw.is_byte_aligned());

        let mut out = bw.finish();

        // §5.11.1 per-tile loop. All but the last tile carry
        // tile_size_minus_1 (le(TileSizeBytes)); the last tile is
        // identified by the OBU body's remaining length.
        let last_idx = obu.tiles.len() - 1;
        for (i, tile) in obu.tiles.iter().enumerate() {
            if i != last_idx {
                let size = tile.bytes.len();
                debug_assert!(size >= 1, "non-last tile cannot be empty (§5.11.1)");
                let size_minus_1 = (size - 1) as u64;
                let cap_bits: u32 = obu.tile_size_bytes * 8;
                debug_assert!(
                    cap_bits == 64 || size_minus_1 < (1u64 << cap_bits),
                    "tile_size_minus_1 {} does not fit in {} bytes",
                    size_minus_1,
                    obu.tile_size_bytes
                );
                // le(TileSizeBytes) — write `tile_size_minus_1` as
                // `TileSizeBytes` little-endian bytes (§4.10.4 inverse).
                for b in 0..obu.tile_size_bytes {
                    out.push(((size_minus_1 >> (b * 8)) & 0xff) as u8);
                }
            }
            out.extend_from_slice(&tile.bytes);
        }

        Ok(out)
    }
}

/// Free-function convenience around
/// [`TileGroupObuWriter::write`].
pub fn write_tile_group_obu(obu: &TileGroupObu) -> Result<Vec<u8>, Error> {
    TileGroupObuWriter.write(obu)
}

/// Parser counterpart for round-trip tests: walks a §5.11.1
/// `tile_group_obu` body and surfaces the same fields the writer
/// emitted. Lives next to the writer because no production caller
/// drives a §5.11.1 parse yet — full per-block decode is the next
/// arc on the consumer side.
///
/// Returns the parsed `(tg_start, tg_end, tile_start_and_end_present_flag,
/// tile_payloads)` tuple; each payload is the raw entropy bytes for
/// that tile (the inverse of [`TilePayload::bytes`]).
///
/// `num_tiles`, `tile_cols_log2`, `tile_rows_log2`, `tile_size_bytes`
/// must match the values the encoder used (the spec sources them from
/// the active frame header, which is out-of-band relative to the
/// OBU body).
pub fn parse_tile_group_obu_body(
    body: &[u8],
    num_tiles: u32,
    tile_cols_log2: u32,
    tile_rows_log2: u32,
    tile_size_bytes: u32,
) -> Result<ParsedTileGroup, Error> {
    debug_assert!(num_tiles >= 1);
    debug_assert!((1..=4).contains(&tile_size_bytes));

    // §5.11.1 header reads. Use a fresh BitReader so the §5.11.1
    // `byte_alignment()` advances the bit position correctly.
    let mut br = crate::bitreader::BitReader::new(body);
    let (tg_start, tg_end, flag) = if num_tiles > 1 {
        let flag = br.f(1)? as u32;
        if flag != 0 {
            let tile_bits = tile_cols_log2 + tile_rows_log2;
            let tg_start = br.f(tile_bits)? as u32;
            let tg_end = br.f(tile_bits)? as u32;
            // §6.10.1 conformance: `tg_end >= tg_start` and every tile
            // index is `< NumTiles`. A stream violating either is
            // non-conformant — reject instead of underflowing the
            // `tg_end - tg_start` span below (2026-07-11 scheduled
            // fuzz finding: a crafted `tg_start > tg_end` panicked
            // with `attempt to subtract with overflow`).
            if tg_start > tg_end || tg_end >= num_tiles {
                return Err(Error::UnexpectedEnd);
            }
            (tg_start, tg_end, true)
        } else {
            (0, num_tiles - 1, false)
        }
    } else {
        (0, 0, false)
    };
    // §5.11.1 byte_alignment.
    while br.position() & 7 != 0 {
        let _ = br.f(1)?;
    }
    let header_bytes = br.position() / 8;
    let mut cursor = header_bytes;
    let last_idx = (tg_end - tg_start) as usize;
    let mut tiles: Vec<TilePayload> = Vec::with_capacity(last_idx + 1);
    let mut remaining = body.len() - header_bytes;
    for i in 0..=last_idx {
        let tile_size = if i == last_idx {
            // §5.11.1: last tile takes the residual.
            remaining
        } else {
            if cursor + tile_size_bytes as usize > body.len() {
                return Err(Error::UnexpectedEnd);
            }
            // le(TileSizeBytes) read.
            let mut tile_size_minus_1: u64 = 0;
            for b in 0..tile_size_bytes {
                tile_size_minus_1 |= u64::from(body[cursor + b as usize]) << (b * 8);
            }
            cursor += tile_size_bytes as usize;
            remaining -= tile_size_bytes as usize;
            let s = (tile_size_minus_1 + 1) as usize;
            if s > remaining {
                return Err(Error::UnexpectedEnd);
            }
            s
        };
        if cursor + tile_size > body.len() {
            return Err(Error::UnexpectedEnd);
        }
        tiles.push(TilePayload::new(body[cursor..cursor + tile_size].to_vec()));
        cursor += tile_size;
        remaining -= tile_size;
    }
    Ok(ParsedTileGroup {
        tg_start,
        tg_end,
        tile_start_and_end_present_flag: flag,
        tiles,
    })
}

/// Walked-back §5.11.1 body, surfaced by [`parse_tile_group_obu_body`].
/// Provides byte-equality round-trip checking against the writer's
/// input descriptor for the test harness.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedTileGroup {
    pub tg_start: u32,
    pub tg_end: u32,
    pub tile_start_and_end_present_flag: bool,
    pub tiles: Vec<TilePayload>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::obu::{write_obu_with_size, ObuHeader};
    use crate::encoder::symbol_writer::SymbolWriter;
    use crate::obu::ObuType;

    /// Build one tile's worth of entropy bytes — a `SymbolWriter` with
    /// a few `write_bool` calls, finished. This produces real §8.2
    /// payloads so the size accounting matches what a tile body would
    /// look like in production.
    fn tile_payload_with_bools(bools: &[u32]) -> TilePayload {
        let mut w = SymbolWriter::new(true);
        for &b in bools {
            w.write_bool(b).unwrap();
        }
        TilePayload::new(w.finish())
    }

    /// Single-tile (NumTiles == 1) frame: §5.11.1 collapses to
    /// `byte_alignment()` + the lone tile payload, with no flag and no
    /// size field.
    #[test]
    fn single_tile_omits_flag_and_size() {
        let payload = tile_payload_with_bools(&[1, 0, 1, 1, 0, 0, 1, 0]);
        let obu = TileGroupObu::whole_frame(1, 0, 0, 1, vec![payload.clone()]);
        let body = write_tile_group_obu(&obu).unwrap();
        // No header bits (NumTiles == 1) ⇒ body is exactly the tile
        // payload bytes.
        assert_eq!(body, payload.bytes);
    }

    /// Multi-tile (NumTiles > 1) frame with flag = 0: §5.11.1 emits
    /// just the flag bit, byte-aligns, then concatenates per-tile
    /// `tile_size_minus_1 | bytes` records (last tile has no size).
    #[test]
    fn multi_tile_flag_zero_round_trip() {
        let a = tile_payload_with_bools(&[1, 1, 0, 0]);
        let b = tile_payload_with_bools(&[0, 1, 0, 1, 1, 0]);
        let c = tile_payload_with_bools(&[1, 0]);
        let obu = TileGroupObu::whole_frame(
            3,
            1, // 2 cols
            1, // 2 rows — overshoots NumTiles=3 but tileBits unused here
            1,
            vec![a.clone(), b.clone(), c.clone()],
        );
        let body = write_tile_group_obu(&obu).unwrap();
        // Header: f(1)=0 + byte_align => one byte 0x00.
        assert_eq!(body[0], 0x00);
        // Then a.len()-1, a.bytes, b.len()-1, b.bytes, c.bytes.
        let mut expected = vec![0x00];
        expected.push((a.bytes.len() - 1) as u8);
        expected.extend_from_slice(&a.bytes);
        expected.push((b.bytes.len() - 1) as u8);
        expected.extend_from_slice(&b.bytes);
        expected.extend_from_slice(&c.bytes);
        assert_eq!(body, expected);

        let parsed = parse_tile_group_obu_body(&body, 3, 1, 1, 1).unwrap();
        assert_eq!(parsed.tg_start, 0);
        assert_eq!(parsed.tg_end, 2);
        assert!(!parsed.tile_start_and_end_present_flag);
        assert_eq!(parsed.tiles.len(), 3);
        assert_eq!(parsed.tiles[0], a);
        assert_eq!(parsed.tiles[1], b);
        assert_eq!(parsed.tiles[2], c);
    }

    /// Multi-tile with flag = 1, exercising the `f(tileBits)`
    /// `tg_start` / `tg_end` writes.
    #[test]
    fn multi_tile_flag_one_writes_tg_start_and_end() {
        // 4-tile frame (2x2), tileBits = 2.
        let payloads: Vec<TilePayload> = (0..4)
            .map(|i| tile_payload_with_bools(&[i & 1, (i >> 1) & 1, 1, 0]))
            .collect();
        let obu = TileGroupObu {
            num_tiles: 4,
            tile_cols_log2: 1,
            tile_rows_log2: 1,
            tile_size_bytes: 1,
            tg_start: 1,
            tg_end: 3,
            start_and_end_present: true,
            tiles: payloads[1..].to_vec(),
        };
        let body = write_tile_group_obu(&obu).unwrap();
        let parsed = parse_tile_group_obu_body(&body, 4, 1, 1, 1).unwrap();
        assert_eq!(parsed.tg_start, 1);
        assert_eq!(parsed.tg_end, 3);
        assert!(parsed.tile_start_and_end_present_flag);
        assert_eq!(parsed.tiles.len(), 3);
        assert_eq!(parsed.tiles[0].bytes, payloads[1].bytes);
        assert_eq!(parsed.tiles[1].bytes, payloads[2].bytes);
        assert_eq!(parsed.tiles[2].bytes, payloads[3].bytes);
    }

    /// `TileSizeBytes = 2` — the size field becomes two little-endian
    /// bytes. Verifies the byte order and the round trip.
    #[test]
    fn tile_size_bytes_two_round_trip() {
        let a = TilePayload::new(vec![0xAA; 300]);
        let b = TilePayload::new(vec![0xBB; 200]);
        let obu = TileGroupObu::whole_frame(2, 1, 0, 2, vec![a.clone(), b.clone()]);
        let body = write_tile_group_obu(&obu).unwrap();
        // Header: f(1)=0 (flag=false) + 7 zero pad bits => byte 0x00.
        // Then le(2)=tile_size_minus_1 for tile 0 = 299 = 0x012B
        //   => low byte 0x2B, high byte 0x01.
        assert_eq!(body[0], 0x00);
        assert_eq!(body[1], 0x2B);
        assert_eq!(body[2], 0x01);
        assert_eq!(&body[3..303], &a.bytes[..]);
        assert_eq!(&body[303..], &b.bytes[..]);

        let parsed = parse_tile_group_obu_body(&body, 2, 1, 0, 2).unwrap();
        assert_eq!(parsed.tiles.len(), 2);
        assert_eq!(parsed.tiles[0], a);
        assert_eq!(parsed.tiles[1], b);
    }

    /// `TileSizeBytes = 4` covers the §6.8.14 max width.
    #[test]
    fn tile_size_bytes_four_round_trip() {
        let a = TilePayload::new(vec![0x11; 5]);
        let b = TilePayload::new(vec![0x22; 9]);
        let obu = TileGroupObu::whole_frame(2, 1, 0, 4, vec![a.clone(), b.clone()]);
        let body = write_tile_group_obu(&obu).unwrap();
        // After the 1-byte header: 4-byte le size, then a, then b.
        assert_eq!(body[0], 0x00);
        // tile_size_minus_1(a) = 4 ⇒ 0x04 0x00 0x00 0x00.
        assert_eq!(&body[1..5], &[0x04, 0x00, 0x00, 0x00]);
        assert_eq!(&body[5..10], &a.bytes[..]);
        assert_eq!(&body[10..], &b.bytes[..]);

        let parsed = parse_tile_group_obu_body(&body, 2, 1, 0, 4).unwrap();
        assert_eq!(parsed.tiles[0], a);
        assert_eq!(parsed.tiles[1], b);
    }

    /// Wrap the §5.11.1 body in an `OBU_TILE_GROUP` via
    /// `write_obu_with_size` and confirm the §5.3.1 wrapper does NOT
    /// append a §5.3.4 trailer for `OBU_TILE_GROUP` (per the
    /// `obu_type_takes_trailing_bits` exclusion). Parse the OBU back
    /// out and re-parse the body.
    #[test]
    fn tile_group_obu_in_obu_wrapper_round_trip() {
        let a = tile_payload_with_bools(&[1, 0, 1, 0]);
        let b = tile_payload_with_bools(&[1, 1, 1, 1, 0, 0]);
        let obu = TileGroupObu::whole_frame(2, 1, 0, 1, vec![a.clone(), b.clone()]);
        let body = write_tile_group_obu(&obu).unwrap();

        let mut out = Vec::new();
        write_obu_with_size(&mut out, &ObuHeader::new(ObuType::TileGroup), &body);
        let (desc, _consumed) = crate::obu::parse_obu(&out).unwrap();
        assert_eq!(desc.obu_type, ObuType::TileGroup);
        // OBU_TILE_GROUP: §5.3.1 explicitly skips the trailer, so the
        // payload length equals the body length exactly.
        assert_eq!(desc.payload_len, body.len());
        assert_eq!(desc.payload, &body[..]);

        // Re-parse the body the OBU walker handed us.
        let parsed = parse_tile_group_obu_body(desc.payload, 2, 1, 0, 1).unwrap();
        assert_eq!(parsed.tiles[0], a);
        assert_eq!(parsed.tiles[1], b);
    }

    /// Round-trip with zero-padded payloads (the typical `SymbolWriter`
    /// output where `finish()` packs the last partial byte with 0s).
    /// Verifies the size accounting includes those pad bytes per the
    /// §6.10.1 note.
    #[test]
    fn payload_pad_bytes_counted_in_tile_size() {
        // 18 bool writes => `low_bits.len() = 15 + 18*15 = 285` then
        // padded to a multiple of 8 = 288 bits = 36 bytes — exact byte
        // count doesn't matter, but the wrapper must use the full
        // returned byte length as `tile_size`.
        let mut w = SymbolWriter::new(true);
        for _ in 0..18 {
            w.write_bool(1).unwrap();
        }
        let bytes_a = w.finish();
        let mut w2 = SymbolWriter::new(true);
        for _ in 0..7 {
            w2.write_bool(0).unwrap();
        }
        let bytes_b = w2.finish();
        let len_a = bytes_a.len();
        let len_b = bytes_b.len();

        let obu = TileGroupObu::whole_frame(
            2,
            1,
            0,
            1,
            vec![
                TilePayload::new(bytes_a.clone()),
                TilePayload::new(bytes_b.clone()),
            ],
        );
        let body = write_tile_group_obu(&obu).unwrap();
        // Header (1 byte 0x00) + 1-byte size + len_a + len_b.
        assert_eq!(body.len(), 1 + 1 + len_a + len_b);
        let parsed = parse_tile_group_obu_body(&body, 2, 1, 0, 1).unwrap();
        assert_eq!(parsed.tiles[0].bytes, bytes_a);
        assert_eq!(parsed.tiles[1].bytes, bytes_b);
    }

    /// `whole_frame` constructor sets `start_and_end_present = false`
    /// and `tg_start=0` / `tg_end=num_tiles-1` per the §6.10.1
    /// single-tile-group default.
    #[test]
    fn whole_frame_constructor_defaults() {
        let payloads = vec![TilePayload::new(vec![0x10]), TilePayload::new(vec![0x20])];
        let obu = TileGroupObu::whole_frame(2, 1, 0, 1, payloads);
        assert_eq!(obu.tg_start, 0);
        assert_eq!(obu.tg_end, 1);
        assert!(!obu.start_and_end_present);
    }
}
