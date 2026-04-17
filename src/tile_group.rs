//! AV1 tile group OBU — §5.11 / §6.11.
//!
//! `OBU_TILE_GROUP` carries the compressed coefficients + per-tile syntax
//! for a contiguous range of tiles within a frame. The payload format is:
//!
//! 1. 1-bit `tile_start_and_end_present_flag` (only when `NumTiles > 1`).
//! 2. Optional `tg_start` / `tg_end` (each `tileBits = TileColsLog2 +
//!    TileRowsLog2` bits).
//! 3. `byte_alignment()` padding.
//! 4. For each tile `TileNum in tg_start..=tg_end` except the last: a
//!    little-endian `tile_size_minus_1` of `TileSizeBytes` bytes, followed
//!    by that many bytes of compressed tile data.
//! 5. The last tile's payload fills the remainder of the OBU.
//!
//! This module parses that framing. It does **not** run `decode_tile()` —
//! which requires default CDF tables (§9.4.1 / §9.4.2) + coefficient decode
//! + intra prediction + transforms + deblock + CDEF + loop restoration.
//!
//! Callers receive a `Vec<TilePayload>` with precise byte-range pointers
//! into the source buffer; surfacing an `Error::Unsupported` beyond that
//! point makes it obvious where the decoder stops.

use oxideav_core::{Error, Result};

use crate::bitreader::BitReader;
use crate::tile_info::TileInfo;

/// Parsed header fields of a `tile_group_obu()` (§5.11.1).
#[derive(Clone, Debug)]
pub struct TileGroupHeader {
    pub tile_start_and_end_present_flag: bool,
    pub tg_start: u32,
    pub tg_end: u32,
    /// Byte offset within the OBU payload at which per-tile data starts.
    /// Includes the leading flag bits + byte_alignment() padding.
    pub header_bytes: usize,
}

/// One tile's compressed payload, described as a byte range within the
/// enclosing OBU payload.
#[derive(Clone, Debug)]
pub struct TilePayload {
    pub tile_num: u32,
    pub tile_row: u32,
    pub tile_col: u32,
    pub offset: usize,
    pub len: usize,
}

/// Build the "tile decode not implemented" error with a precise spec
/// reference list. Emitted by decoder entry points that have nothing
/// further to do beyond the byte-boundary extraction performed here.
pub fn tile_decode_unsupported() -> Error {
    Error::unsupported(
        "av1 tile decode: §5.11 tile_group_obu body — default CDF tables \
         (§9.4.1/§9.4.2), coefficient decode (§5.11.39), intra prediction \
         (§7.11.2), transforms (§7.7), deblock (§7.14), CDEF (§7.15) and \
         loop restoration (§7.17) are not implemented. Parse-only build: \
         tile boundaries are extracted; pixel reconstruction is not.",
    )
}

/// Parse the `tile_group_obu()` header (flags + optional tg_start/tg_end +
/// byte_alignment). Returns the decoded header plus the byte offset at
/// which per-tile data begins.
pub fn parse_tile_group_header(payload: &[u8], tile_info: &TileInfo) -> Result<TileGroupHeader> {
    let mut br = BitReader::new(payload);
    let num_tiles = tile_info.num_tiles();
    let tile_start_and_end_present_flag = if num_tiles > 1 { br.bit()? } else { false };
    let (tg_start, tg_end) = if num_tiles == 1 || !tile_start_and_end_present_flag {
        (0u32, num_tiles - 1)
    } else {
        let tile_bits = tile_info.tile_cols_log2 + tile_info.tile_rows_log2;
        let s = br.f(tile_bits)?;
        let e = br.f(tile_bits)?;
        (s, e)
    };
    br.byte_alignment()?;
    let header_bytes = (br.bit_position() / 8) as usize;
    if tg_end < tg_start || tg_end >= num_tiles {
        return Err(Error::invalid(format!(
            "av1 tile_group_obu: tg_start={tg_start} tg_end={tg_end} \
             NumTiles={num_tiles} violates §5.11.1",
        )));
    }
    Ok(TileGroupHeader {
        tile_start_and_end_present_flag,
        tg_start,
        tg_end,
        header_bytes,
    })
}

/// Split a `tile_group_obu` payload into per-tile byte ranges. The returned
/// offsets are relative to the payload (not including the OBU header).
///
/// Uses `TileInfo::tile_size_bytes` for the per-tile `tile_size_minus_1`
/// encoding and the tile geometry to assign `(tile_row, tile_col)` pairs.
/// Single-tile frames (`TileSizeBytes == 0`) skip the size-prefix read; the
/// full remainder of the payload is the single tile.
pub fn split_tile_payloads(
    payload: &[u8],
    tile_info: &TileInfo,
    header: &TileGroupHeader,
) -> Result<Vec<TilePayload>> {
    let mut out = Vec::new();
    let mut sz = payload
        .len()
        .checked_sub(header.header_bytes)
        .ok_or_else(|| Error::invalid("av1 tile_group_obu: header_bytes > payload"))?;
    let mut pos = header.header_bytes;
    let tsb = tile_info.tile_size_bytes as usize;
    for tile_num in header.tg_start..=header.tg_end {
        let last_tile = tile_num == header.tg_end;
        let tile_size = if last_tile || tsb == 0 {
            sz
        } else {
            if pos + tsb > payload.len() {
                return Err(Error::invalid(
                    "av1 tile_group_obu: truncated tile_size_minus_1",
                ));
            }
            let mut v: u64 = 0;
            for i in 0..tsb {
                v |= (payload[pos + i] as u64) << (8 * i);
            }
            pos += tsb;
            sz -= tsb;
            (v as usize)
                .checked_add(1)
                .ok_or_else(|| Error::invalid("av1 tile_group_obu: tile_size overflow"))?
        };
        if pos + tile_size > payload.len() {
            return Err(Error::invalid(format!(
                "av1 tile_group_obu: tile {tile_num} size {tile_size} \
                 exceeds remaining payload {}",
                payload.len() - pos,
            )));
        }
        let tile_row = tile_num / tile_info.tile_cols;
        let tile_col = tile_num % tile_info.tile_cols;
        out.push(TilePayload {
            tile_num,
            tile_row,
            tile_col,
            offset: pos,
            len: tile_size,
        });
        pos += tile_size;
        if !last_tile {
            sz = sz
                .checked_sub(tile_size)
                .ok_or_else(|| Error::invalid("av1 tile_group_obu: sz underflow"))?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_tile_info(mi: u32) -> TileInfo {
        TileInfo {
            tile_cols: 1,
            tile_rows: 1,
            tile_cols_log2: 0,
            tile_rows_log2: 0,
            uniform_tile_spacing_flag: true,
            mi_col_starts: vec![0, mi],
            mi_row_starts: vec![0, mi],
            tile_size_bytes: 0,
            context_update_tile_id: 0,
        }
    }

    #[test]
    fn single_tile_has_no_flag_and_no_size_prefix() {
        let ti = single_tile_info(16);
        // Tile-group body is all compressed bytes. No leading flag.
        let body = [0x01, 0x02, 0x03, 0x04];
        let header = parse_tile_group_header(&body, &ti).unwrap();
        assert_eq!(header.header_bytes, 0);
        assert_eq!(header.tg_start, 0);
        assert_eq!(header.tg_end, 0);
        let tiles = split_tile_payloads(&body, &ti, &header).unwrap();
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].offset, 0);
        assert_eq!(tiles[0].len, 4);
    }

    #[test]
    fn four_tiles_uniform_size_prefixes() {
        let mut ti = single_tile_info(16);
        ti.tile_cols = 2;
        ti.tile_rows = 2;
        ti.tile_cols_log2 = 1;
        ti.tile_rows_log2 = 1;
        ti.tile_size_bytes = 1;
        ti.mi_col_starts = vec![0, 8, 16];
        ti.mi_row_starts = vec![0, 8, 16];

        // Construct a body:
        //   [flag=0][7 bits zero-padding → 1 byte header]
        //   [sz0=5-1=4][tile0 5 bytes][sz1=3-1=2][tile1 3 bytes]
        //   [sz2=2-1=1][tile2 2 bytes][tile3 = rest, 4 bytes]
        // Note: tg_start/tg_end are skipped because flag=0.
        let mut body = Vec::new();
        body.push(0x00);
        body.push(0x04);
        body.extend_from_slice(&[0xA0, 0xA1, 0xA2, 0xA3, 0xA4]);
        body.push(0x02);
        body.extend_from_slice(&[0xB0, 0xB1, 0xB2]);
        body.push(0x01);
        body.extend_from_slice(&[0xC0, 0xC1]);
        body.extend_from_slice(&[0xD0, 0xD1, 0xD2, 0xD3]);

        let header = parse_tile_group_header(&body, &ti).unwrap();
        assert_eq!(header.header_bytes, 1);
        assert_eq!(header.tg_start, 0);
        assert_eq!(header.tg_end, 3);

        let tiles = split_tile_payloads(&body, &ti, &header).unwrap();
        assert_eq!(tiles.len(), 4);
        // Tile geometry:
        assert_eq!((tiles[0].tile_row, tiles[0].tile_col), (0, 0));
        assert_eq!((tiles[1].tile_row, tiles[1].tile_col), (0, 1));
        assert_eq!((tiles[2].tile_row, tiles[2].tile_col), (1, 0));
        assert_eq!((tiles[3].tile_row, tiles[3].tile_col), (1, 1));
        // Sizes + content:
        assert_eq!(
            &body[tiles[0].offset..tiles[0].offset + tiles[0].len],
            &[0xA0, 0xA1, 0xA2, 0xA3, 0xA4]
        );
        assert_eq!(
            &body[tiles[1].offset..tiles[1].offset + tiles[1].len],
            &[0xB0, 0xB1, 0xB2]
        );
        assert_eq!(
            &body[tiles[2].offset..tiles[2].offset + tiles[2].len],
            &[0xC0, 0xC1]
        );
        assert_eq!(
            &body[tiles[3].offset..tiles[3].offset + tiles[3].len],
            &[0xD0, 0xD1, 0xD2, 0xD3]
        );
    }
}
