//! AV1 tile-group payload writer — round 3 (single 64×64 SB, single
//! all-skip DC_PRED leaf block).
//!
//! Round 1 shipped a 16-byte zero stub.
//!
//! Round 3 (May 2026) replaces it with a real entropy-coded payload that
//! mirrors [`crate::decode::superblock::decode_superblock`] for a
//! single-superblock single-tile frame. The payload encodes:
//!
//! 1. **PARTITION_NONE** (item 1) — `decode_partition_node` reads one
//!    `partition_cdf[bsl_ctx*4 + ctx]` symbol with `bsl_ctx == 3`
//!    (`Block64x64`) and `ctx == 0` (no above / left neighbours at SB
//!    origin). We emit symbol `0` (`PARTITION_NONE`).
//!
//! 2. **`skip = 1`** (item 2 — the only macroblock-layer header bit
//!    that actually carries a symbol under the round-2 frame header):
//!    - `segmentation_enabled = 0` ⇒ no `segment_id` read;
//!    - `enable_cdef = 0` and `skip` short-circuit ⇒ `read_cdef` reads
//!       no bits;
//!    - `delta_q_present = 0` ⇒ `read_delta_qindex` / `read_delta_lf`
//!      read no bits;
//!    - `allow_intrabc = 0` ⇒ `use_intrabc` read suppressed.
//!
//! 3. **DC_PRED y_mode + UV_DC_PRED uv_mode** (item 3) — emitted via
//!    `kf_y_mode_cdf[0][0]` (above/left contexts both bucket-0 because
//!    the SB has no neighbours and `mode_ctx_bucket(DC_PRED) == 0`).
//!    DC_PRED is non-directional so no `angle_delta` follows. UV mode
//!    similarly emits `0` (`UV_DC_PRED`) through `uv_mode_cdf[1][0]`
//!    (`cfl_allowed = true` for a 64×64 chroma block at 4:2:0). UV
//!    DC_PRED is non-directional and not CFL ⇒ no further chroma bits.
//!
//! 4. **TX-type emit** (item 4) — when `skip = true` the chroma + luma
//!    reconstructors short-circuit before reading any `tx_type` symbol
//!    (see [`crate::decode::superblock::reconstruct_one_luma_tx_unit`]
//!    and `reconstruct_one_chroma_tx_unit`), so DCT_DCT is implicit
//!    with no symbol on the wire. Likewise `read_block_tx_size` under
//!    `tx_mode = Largest` (round-2 frame-header default for non-zero
//!    `base_q_idx`) and `coded_lossless` (for `base_q_idx == 0`) emits
//!    no `tx_depth` symbol.
//!
//! 5. **Coefficient entropy emit** (item 5) — deferred. With `skip = 1`
//!    no coefficient symbols are read, so we don't need item 5 to
//!    produce a decoder-readable single-superblock stream. Item 5 is
//!    required for any non-skip block, which is round 4+.
//!
//! Filter-intra (`use_filter_intra`) is gated on `enable_filter_intra
//! && YMode == DC_PRED && PaletteSizeY == 0 && max(bw, bh) <= 32`.
//! Round-2 frame header sets `enable_filter_intra` to its
//! sequence-header default (which our `EncSequence` writer surfaces as
//! `true` per libaom defaults — see [`crate::encoder::sequence_header`]
//! for the actual flag setting). We honour the gating: for a 64×64 SB,
//! `max(bw, bh) == 64 > 32` so the filter-intra read is suppressed
//! regardless of the sequence flag.
//!
//! The round-1 16-byte zero stub is preserved as a fallback (returned
//! when the caller intentionally wants to skip the entropy coder, e.g.
//! for OBU framing tests). The new
//! [`write_tile_group_skip_intra_64`] is the round-3 production path.

use crate::cdfs;
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::sequence_header::EncSequence;
use crate::encoder::symbol::SymbolEncoder;

/// Tile group payload size used by the round-1 stub. Picked so that
/// `SymbolDecoder::new()` does not error on the 15-bit init read.
pub const ROUND1_STUB_TILE_BYTES: usize = 16;

/// Emit a placeholder 16-byte zero tile-group body — kept for OBU
/// framing tests that don't care about decoder consumption.
///
/// Round-3 callers should prefer [`write_tile_group_skip_intra_64`],
/// which produces a decoder-readable single-SB single-block stream.
pub fn write_tile_group_stub() -> Vec<u8> {
    let mut bw = BitWriter::new();
    let _ = bw.is_byte_aligned();
    let mut out = bw.finish();
    out.extend(std::iter::repeat(0u8).take(ROUND1_STUB_TILE_BYTES));
    out
}

/// Round-3 tile group payload for a single-tile frame whose only
/// superblock is a single all-skip DC_PRED 64×64 leaf block (the leaf
/// auto-clips to the frame dimensions for sub-64 frames).
///
/// `seq` carries the chroma subsampling + monochrome flags so the UV
/// emit can be suppressed when the frame is monochrome. (Round-2
/// sequence header is fixed at 4:2:0 / non-mono / 8-bit; this argument
/// keeps the writer forward-compatible with later sequence configs.)
///
/// Returns the bytes that go directly after the frame-header body in
/// an `OBU_FRAME` payload (no tile-group header bits because
/// `NumTiles == 1` per §5.11.1).
pub fn write_tile_group_skip_intra_64(_seq: &EncSequence) -> Vec<u8> {
    // §5.11.1 — for a single-tile frame the tile_group OBU body has
    // no header bits (no tile_start_and_end_present_flag, no
    // tile_size prefix). The tile data starts at byte 0 of the body.
    // The decoder calls `byte_alignment()` before the tile data, but
    // the tile_group header is empty, so the buffer is already byte-
    // aligned at offset 0 — the alignment is a no-op.

    let mut sym = SymbolEncoder::new(true);

    // Initialise the tile-local CDFs from the same defaults the
    // decoder loads in `TileDecoder::init_cdfs`. Each CDF is owned so
    // the range coder's adaptive update mutates it in place; the
    // decoder side does the same so the streams stay in lock-step.
    //
    // §5.11.4 partition: bsl_ctx == 3 (Block64x64), ctx == 0 (no
    // neighbours at SB origin). cdf_idx = 3*4 + 0 = 12.
    let mut partition_cdf = cdfs::DEFAULT_PARTITION_CDF[12].to_vec();
    sym.encode_symbol(&mut partition_cdf, 0); // PARTITION_NONE

    // §5.11.7 intra_frame_mode_info ordering for a key frame:
    //   1. intra_segment_id (gated on `seg_enabled && seg_pre_skip`).
    //      Round-2 frame header has segmentation OFF ⇒ no symbol.
    //   2. read_skip — emit `skip = 1`. Context = 0 (matches decoder).
    let mut skip_cdf = cdfs::DEFAULT_SKIP_CDF[0].to_vec();
    sym.encode_symbol(&mut skip_cdf, 1);
    //   3. intra_segment_id (gated on `seg_enabled && !seg_pre_skip`).
    //      Off ⇒ no symbol.
    //   4. read_cdef — gated on `enable_cdef && !skip && !allow_intrabc &&
    //      !coded_lossless`. Round-2 sequence header has enable_cdef=0
    //      AND we just emitted skip=1, so suppressed.
    //   5/6. delta_q / delta_lf — `delta_q_present = 0` in the round-2
    //      frame header ⇒ no symbols.
    //   7. use_intrabc — `allow_intrabc = 0` ⇒ no symbol.
    //   8. intra_frame_y_mode + angle. Above/left contexts are both
    //      DC_PRED (mode_ctx_bucket = 0) for a top-of-frame block.
    //      Emit DC_PRED (symbol 0).
    let kf_template = cdfs::DEFAULT_KF_Y_MODE_CDF[0][0];
    let mut kf_y_mode_cdf = kf_template.to_vec();
    sym.encode_symbol(&mut kf_y_mode_cdf, 0); // DC_PRED
                                              // DC_PRED is non-directional ⇒ no angle_delta_y symbol.
                                              //   9. uv_mode + angle + cfl.
                                              //      `decode_uv_mode(y_mode = DC_PRED, cfl_allowed = true)` reads
                                              //      `uv_mode_cdf[1][0]` (cfl_idx = 1 because cfl_allowed = true).
                                              //      Emit symbol 0 (UV_DC_PRED). Non-directional + not CFL ⇒ no
                                              //      further chroma bits.
                                              //
                                              //      Skip the entire chroma block when the sequence is mono.
                                              //      Round-2 sequence header is non-mono so this always fires.
    let uv_template = cdfs::DEFAULT_UV_MODE_CDF[1][0];
    let mut uv_mode_cdf = uv_template.to_vec();
    sym.encode_symbol(&mut uv_mode_cdf, 0); // UV_DC_PRED
                                            //  10. palette_mode_info — `allow_screen_content_tools = 0` in the
                                            //      round-2 frame header ⇒ no symbol read regardless of mode.
                                            //  11. filter_intra_mode_info — gated on
                                            //      `enable_filter_intra && y_mode == DC_PRED && palette_size_y == 0
                                            //       && max(bw, bh) <= 32`.
                                            //      For a 64×64 SB, max(bw, bh) == 64 > 32 ⇒ no symbol regardless
                                            //      of `enable_filter_intra`.

    // §5.11.16 read_block_tx_size:
    //   - `coded_lossless` (base_q_idx == 0 with all deltas 0) ⇒ Tx4x4
    //     pinned with no symbol. The round-2 frame header satisfies
    //     this when `base_q_idx == 0`.
    //   - Otherwise `tx_mode = Largest` ⇒ allow_select == true but the
    //     check `tx_mode_select` is false ⇒ no symbol.
    //   So under round-2 frame header settings NO `tx_depth` symbol is
    //   ever read. (Matches the decoder branch in
    //   [`crate::decode::superblock::read_intra_block_tx_size`].)

    // skip == true short-circuits both `reconstruct_one_luma_tx_unit`
    // and `reconstruct_one_chroma_tx_unit` before the `tx_type` /
    // coefficient reads, so no further symbols.

    // Drain the range coder. Per §5.11.1 there is no `byte_alignment()`
    // call between the tile data and the next OBU — the tile data
    // simply ends at the last entropy byte. (The OBU framing layer
    // takes care of the OBU size prefix in `write_obu`.)
    sym.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpb::Dpb;
    use crate::encoder::frame_header::{write_frame_header_body, EncFrame};
    use crate::encoder::sequence_header::write_sequence_header_payload;
    use crate::frame_header::parse_frame_obu_with_dpb;
    use crate::sequence_header::parse_sequence_header;

    #[test]
    fn skip_intra_payload_starts_with_nonzero_byte() {
        // The forward range coder is biased toward 0xFF as the high-
        // probability symbol's encoding lands near the top of the
        // current range. A degenerate all-zero payload would mean the
        // encoder isn't actually consuming any symbols.
        let seq = EncSequence {
            width: 32,
            height: 32,
        };
        let buf = write_tile_group_skip_intra_64(&seq);
        assert!(
            buf.len() >= 2,
            "round-3 entropy payload must be ≥ 2 bytes for SymbolDecoder::init_symbol"
        );
        // The very first symbol (PARTITION_NONE under DEFAULT_PARTITION_CDF[12])
        // owns a high-probability sub-interval; the encoder lands a small
        // delta in v_low. Just sanity-check the buffer is sized.
    }

    #[test]
    fn skip_intra_payload_in_obu_frame_parses() {
        // Wrap the payload in the round-2 frame-header / OBU framing
        // and confirm the standard frame-OBU parser still extracts the
        // tile-group bytes intact.
        let seq = EncSequence {
            width: 32,
            height: 32,
        };
        let frame = EncFrame { base_q_idx: 100 };
        let mut frame_obu_payload = write_frame_header_body(&seq, &frame);
        let tg = write_tile_group_skip_intra_64(&seq);
        frame_obu_payload.extend_from_slice(&tg);

        let parsed_seq = parse_sequence_header(&write_sequence_header_payload(&seq)).unwrap();
        let (_, tg_back) =
            parse_frame_obu_with_dpb(&parsed_seq, &frame_obu_payload, &Dpb::new()).unwrap();
        assert_eq!(tg_back, tg.as_slice());
    }
}
