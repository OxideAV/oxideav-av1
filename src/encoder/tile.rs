//! AV1 tile-group payload writer — round 1 stub.
//!
//! Round 1 emits a placeholder tile group whose body decodes to a
//! single all-skip 64×64 superblock. The full coefficient + entropy
//! coder is a round 2+ task; this stub gets the OBU framing right
//! (single-tile header → byte_alignment() → entropy bytes) so the
//! decoder's `parse_tile_group_header` + `split_tile_payloads` succeed
//! before the symbol decoder is invoked.
//!
//! Since the decoder's symbol decoder requires at least 15 bits of
//! payload (`init_symbol`), we emit 16 zero bytes. A round-2 encoder
//! will replace this with a real partition+mode+coefficient stream.

use crate::encoder::bitwriter::BitWriter;

/// Tile group payload size used by the round-1 stub. Picked so that
/// `SymbolDecoder::new()` does not error on the 15-bit init read.
pub const ROUND1_STUB_TILE_BYTES: usize = 16;

/// Emit the tile_group_obu body for a single-tile frame:
///
/// - `tile_start_and_end_present_flag` is suppressed because
///   `NumTiles == 1`.
/// - `byte_alignment()` is implicit (the writer is empty / aligned).
/// - The tile bytes follow verbatim. Single-tile frames have
///   `TileSizeBytes == 0` so no size prefix is emitted.
pub fn write_tile_group_stub() -> Vec<u8> {
    // For a single-tile frame the tile_group header is empty (no
    // start/end flag is coded since NumTiles == 1; the spec's
    // `byte_alignment()` is a no-op on an empty writer). The spec
    // mandates the per-tile data immediately follows.
    let mut bw = BitWriter::new();
    // tile_group_header bits would land here when NumTiles > 1; for
    // NumTiles == 1 nothing is coded. The buffer is byte-aligned by
    // construction.
    let _ = bw.is_byte_aligned();

    // Append the placeholder tile bytes. A round-2 encoder writes a
    // properly-coded entropy stream here.
    let mut out = bw.finish();
    out.extend(std::iter::repeat(0u8).take(ROUND1_STUB_TILE_BYTES));
    out
}
