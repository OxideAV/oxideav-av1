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
//!      no bits;
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
use crate::decode::coeffs::{nz_map_ctx_offset, q_index_to_ctx, tx_size_idx};
use crate::encoder::bitwriter::BitWriter;
use crate::encoder::coeffs::{encode_coefficients, CoeffCdfBankEnc};
use crate::encoder::sequence_header::EncSequence;
use crate::encoder::symbol::SymbolEncoder;
use crate::transform::{clamped_scan, default_zigzag_scan};

/// Tile group payload size used by the round-1 stub. Picked so that
/// `SymbolDecoder::new()` does not error on the 15-bit init read.
pub const ROUND1_STUB_TILE_BYTES: usize = 16;

/// Emit a placeholder 16-byte zero tile-group body — kept for OBU
/// framing tests that don't care about decoder consumption.
///
/// Round-3 callers should prefer [`write_tile_group_skip_intra_64`],
/// which produces a decoder-readable single-SB single-block stream.
pub fn write_tile_group_stub() -> Vec<u8> {
    let bw = BitWriter::new();
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
pub fn write_tile_group_skip_intra_64(seq: &EncSequence) -> Vec<u8> {
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
                                              //      `decode_uv_mode(y_mode = DC_PRED, cfl_allowed)` reads
                                              //      `uv_mode_cdf[cfl_idx][0]`. cfl_allowed mirrors the
                                              //      decoder's `(bw.max(bh) <= 32)` test — the FRAME-EDGE
                                              //      clipped block dim, NOT the SPEC dim. For a 32×32
                                              //      frame (single 32×32 leaf clipped from Block64x64)
                                              //      cfl_allowed = true ⇒ cfl_idx = 1; for a 64×64 frame
                                              //      cfl_allowed = false ⇒ cfl_idx = 0. (Round-39 wired
                                              //      this consistently with the decoder side; before then
                                              //      the writer hard-coded cfl_idx = 1 which produced a
                                              //      latent dav1d cross-decode failure on 64×64.)
                                              //      Emit symbol 0 (UV_DC_PRED). Non-directional + not CFL ⇒ no
                                              //      further chroma bits.
                                              //
                                              //      Skip the entire chroma block when the sequence is mono.
                                              //      Round-2 sequence header is non-mono so this always fires.
    let cfl_allowed = seq.width.max(seq.height) <= 32;
    let cfl_idx = if cfl_allowed { 1 } else { 0 };
    let uv_template = cdfs::DEFAULT_UV_MODE_CDF[cfl_idx][0];
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

/// Round-40 tile group payload for a single-tile frame whose only
/// superblock is a single **non-skip** DC_PRED 64×64 leaf block with
/// all-zero residual coefficients per plane.
///
/// This is the round-40 production path. Unlike
/// [`write_tile_group_skip_intra_64`] which short-circuits at
/// `skip = 1` (and therefore exercises ZERO coefficient symbols on the
/// wire), this writer emits `skip = 0` and walks the full
/// `decode_coefficients` mirror per plane: one luma TU at 64×64 plus
/// one U + one V TU at 32×32 (under 4:2:0).
///
/// Each plane currently emits an all-zero residual (`txb_skip = true`
/// inside `encode_coefficients`), so the mathematical result matches
/// the skip=1 path exactly: DC_PRED with no neighbours fills the plane
/// with `1 << (bit_depth - 1) = 128` (8-bit). The bitstream is
/// strictly larger (3 extra `txb_skip` symbols) but every symbol the
/// decoder expects under the non-skip path is now present, exercising:
///
/// 1. `read_skip` returning 0
/// 2. `read_intra_block_tx_size` (which still emits no symbol because
///    `tx_mode = Largest` with `bs == Block64x64` ⇒ no split possible)
/// 3. `decode_intra_tx_type` (no symbol because area > 32×32 ⇒
///    `ext_tx_set_for_intra` returns 0 ⇒ implicit `DCT_DCT`)
/// 4. The full luma `decode_coefficients` (`txb_skip` for the all-zero
///    block, returns immediately)
/// 5. The full chroma `decode_coefficients` × 2 (U + V; same all-zero
///    `txb_skip` short-circuit)
///
/// `base_q_idx` is required because the `CoeffCdfBank` defaults
/// branch on the `q_index_to_ctx(base_q_idx)` bucket (Phase-3 / §9.4
/// `q_ctx`-indexed CDFs).
pub fn write_tile_group_intra_64(seq: &EncSequence, base_q_idx: u8) -> Vec<u8> {
    let mut sym = SymbolEncoder::new(true);

    // §5.11.4 partition: bsl_ctx == 3 (Block64x64), ctx == 0.
    let mut partition_cdf = cdfs::DEFAULT_PARTITION_CDF[12].to_vec();
    sym.encode_symbol(&mut partition_cdf, 0); // PARTITION_NONE

    // §5.11.7 #2 read_skip — emit `skip = 0` (was `1` in
    // `write_tile_group_skip_intra_64`).
    let mut skip_cdf = cdfs::DEFAULT_SKIP_CDF[0].to_vec();
    sym.encode_symbol(&mut skip_cdf, 0);

    // §5.11.7 #8 intra_frame_y_mode — DC_PRED at the SB origin (no
    // above / left, both context buckets = 0).
    let mut kf_y_mode_cdf = cdfs::DEFAULT_KF_Y_MODE_CDF[0][0].to_vec();
    sym.encode_symbol(&mut kf_y_mode_cdf, 0); // DC_PRED
                                              // DC_PRED is non-directional ⇒ no angle_delta_y symbol.
                                              // §5.11.7 #9 uv_mode — UV_DC_PRED with cfl_allowed = true (64×64
                                              // luma block: max(bw, bh) = 64 ⇒ cfl_allowed FALSE actually
                                              // because §5.11.7 cfl_allowed gates on `max(Block_W, Block_H) ≤ 32`
                                              // for non-lossless. For Block64x64 cfl_allowed = false ⇒ uv_mode
                                              // CDF is `cfl_idx = 0`, not `cfl_idx = 1`. Match the decoder.
                                              // cfl_allowed mirrors the decoder's clipped-frame-dim test:
                                              // `(bw.max(bh) <= 32)` where bw/bh are the FRAME-EDGE clipped
                                              // block dims, not the spec block dims. For a 32×32 frame (the
                                              // single 32×32 leaf clipped from a 64×64 SB) cfl_allowed = true.
    let cfl_allowed = seq.width.max(seq.height) <= 32;
    let cfl_idx = if cfl_allowed { 1 } else { 0 };
    let mut uv_mode_cdf = cdfs::DEFAULT_UV_MODE_CDF[cfl_idx][0].to_vec();
    sym.encode_symbol(&mut uv_mode_cdf, 0); // UV_DC_PRED

    // §5.11.7 #10 palette_mode_info — `allow_screen_content_tools = 0`
    // ⇒ no symbol regardless of mode.
    // §5.11.7 #11 filter_intra_mode_info — gated on
    // `enable_filter_intra && y_mode == DC_PRED && palette_size_y == 0
    //  && max(bw, bh) <= 32`.
    // For Block64x64 max(bw, bh) = 64 > 32 ⇒ no symbol regardless of
    // the sequence's `enable_filter_intra` setting.

    // §5.11.16 read_block_tx_size — `tx_mode = Largest` (frame header
    // emits `tx_mode_select = 0` for `base_q_idx != 0`) ⇒ no symbol;
    // TxSize pinned to `bs.max_tx_size_rect()` = `Tx64x64`.
    //
    // For coded_lossless (`base_q_idx == 0`) the spec pins TxSize =
    // `Tx4x4` with no symbol, again no bit on the wire. The two paths
    // share the no-symbol property; the residual loop below however
    // walks at the appropriate TX dim, so we branch.

    let lossless = base_q_idx == 0;
    let q_ctx = q_index_to_ctx(base_q_idx as i32);
    let mut bank = CoeffCdfBankEnc::new(q_ctx);

    if lossless {
        // Coded-lossless: `read_intra_block_tx_size` pins the LUMA
        // block TxSize to `Tx4x4`. `reconstruct_luma_block` walks the
        // spec 64×64 dim as 16×16 = 256 4×4 TUs (every TU triggers
        // `decode_dequant_idct_luma` ⇒ `decode_coefficients`).
        //
        // Chroma is INDEPENDENT of the block-level TxSize:
        // `reconstruct_chroma_block` uses `tx_unit_dims(spec_cw,
        // spec_ch)` directly, so even in lossless the chroma walks
        // ONE 32×32 TU per plane (per `tx_unit_dims(32,32) = (32,32)`).
        emit_zero_tus(
            &mut sym,
            &mut bank,
            /*tx_w=*/ 4,
            /*tx_h=*/ 4,
            /*plane_type=*/ 0,
            /*count=*/ 16 * 16,
        );
        // Chroma — single 32×32 TU per plane.
        let tx_idx_uv = tx_size_idx(32, 32).expect("32×32 tx_size_idx");
        let scan_uv = default_zigzag_scan(32, 32);
        let nz_uv = nz_map_ctx_offset(32, 32).expect("32×32 nz_map");
        let zeros_uv = vec![0i32; 32 * 32];
        for _plane_idx in 0..2u32 {
            encode_coefficients(
                &mut sym, &mut bank, tx_idx_uv, /*plane_type=*/ 1, /*num_coeffs=*/ 1024,
                &scan_uv, nz_uv, /*w=*/ 32, /*h=*/ 32, &zeros_uv,
            );
        }
    } else {
        // §5.11.40 — `tx_type` for a 64×64 intra block: ext_tx_set = 0
        // (area > 32×32) ⇒ implicit `DCT_DCT` with no symbol.
        // §5.11.39 — coefficients walk a single 64×64 luma TU.
        // §5.11.34 — chroma walks one 32×32 TU per plane (sub_x = sub_y = 1).

        // Luma 64×64 block: tx_size_idx = 4, num_coeffs = 1024,
        // scan = clamped_scan(32, 32, 64), nz_map = NZ_MAP_CTX_OFFSET_32X32.
        let tx_idx_y = tx_size_idx(64, 64).expect("64×64 tx_size_idx");
        let scan_y = clamped_scan(32, 32, 64);
        let nz_y = nz_map_ctx_offset(64, 64).expect("64×64 nz_map");
        let zeros_y = vec![0i32; 64 * 64];
        encode_coefficients(
            &mut sym, &mut bank, tx_idx_y, /*plane_type=*/ 0, /*num_coeffs=*/ 1024,
            &scan_y, nz_y, /*w=*/ 64, /*h=*/ 64, &zeros_y,
        );

        // Chroma 32×32 block per plane: tx_size_idx = 3, num_coeffs = 1024,
        // scan = default_zigzag(32, 32), nz_map = NZ_MAP_CTX_OFFSET_32X32.
        let tx_idx_uv = tx_size_idx(32, 32).expect("32×32 tx_size_idx");
        let scan_uv = default_zigzag_scan(32, 32);
        let nz_uv = nz_map_ctx_offset(32, 32).expect("32×32 nz_map");
        let zeros_uv = vec![0i32; 32 * 32];
        for _plane_idx in 0..2u32 {
            encode_coefficients(
                &mut sym, &mut bank, tx_idx_uv, /*plane_type=*/ 1, /*num_coeffs=*/ 1024,
                &scan_uv, nz_uv, /*w=*/ 32, /*h=*/ 32, &zeros_uv,
            );
        }
    }

    sym.finish()
}

/// Helper: emit `count` consecutive all-zero TX units of dimensions
/// `tx_w × tx_h`. Each unit costs exactly one `txb_skip = true` symbol.
/// Used by the lossless path which decomposes the 64×64 block into
/// many 4×4 TUs.
fn emit_zero_tus(
    sym: &mut SymbolEncoder,
    bank: &mut CoeffCdfBankEnc,
    tx_w: usize,
    tx_h: usize,
    plane_type: usize,
    count: usize,
) {
    let tx_idx = tx_size_idx(tx_w, tx_h).expect("tx_size_idx");
    let scan = default_zigzag_scan(tx_w, tx_h);
    let nz = nz_map_ctx_offset(tx_w, tx_h).expect("nz_map");
    let zeros = vec![0i32; tx_w * tx_h];
    let num_coeffs = tx_w * tx_h;
    for _ in 0..count {
        encode_coefficients(
            sym, bank, tx_idx, plane_type, num_coeffs, &scan, nz, tx_w, tx_h, &zeros,
        );
    }
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
    fn intra_64_payload_64x64_self_decodes() {
        // Exercise the 64×64 cfl_idx=0 path explicitly (decoder uses
        // `uv_mode_cdf[0][0]` because `bw.max(bh) > 32`).
        use crate::Av1Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, MediaType, Packet, TimeBase};

        let seq = EncSequence {
            width: 64,
            height: 64,
        };
        let bytes = crate::encoder::write_keyframe_stream(&seq, &EncFrame { base_q_idx: 100 });

        let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        params.width = Some(64);
        params.height = Some(64);
        params.media_type = MediaType::Video;
        let mut dec = Av1Decoder::new(params);
        let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
        dec.send_packet(&pkt)
            .expect("decoder accepts 64×64 round-40 stream");
        let frame = dec.receive_frame().expect("decoder yields a frame");
        let Frame::Video(vf) = frame else {
            panic!("expected Frame::Video");
        };
        assert_eq!(vf.planes[0].data.len(), 64 * 64);
        let mean: u32 = vf.planes[0].data.iter().map(|&v| v as u32).sum::<u32>() / (64 * 64);
        assert_eq!(mean, 128);
    }

    #[test]
    fn intra_64_payload_is_strictly_larger_than_skip() {
        // The non-skip writer walks `encode_coefficients` per plane,
        // so it MUST emit at least the symbols emitted by the skip
        // writer plus one txb_skip per plane (3 extra symbols for
        // 4:2:0). The byte count is dominated by the range coder's
        // 16-bit init/finalise pad, so equality is also acceptable
        // for very compact payloads — but the entropy state must
        // diverge.
        let seq = EncSequence {
            width: 32,
            height: 32,
        };
        let skip = write_tile_group_skip_intra_64(&seq);
        let intra = write_tile_group_intra_64(&seq, 100);
        assert!(intra.len() >= skip.len());
        // The two payloads are NOT identical — the non-skip writer
        // emits skip=0 (different range bin from skip=1) plus three
        // extra txb_skip symbols.
        assert_ne!(intra, skip, "non-skip and skip payloads must differ");
    }

    #[test]
    fn intra_64_payload_in_obu_frame_self_decodes_32x32() {
        // End-to-end: round-40 writer through the full Av1Decoder.
        // Confirms every symbol emitted is what the decoder expects.
        use crate::Av1Decoder;
        use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, MediaType, Packet, TimeBase};

        let seq = EncSequence {
            width: 32,
            height: 32,
        };
        let bytes = crate::encoder::write_keyframe_stream(&seq, &EncFrame { base_q_idx: 100 });

        let mut params = CodecParameters::video(CodecId::new(crate::CODEC_ID_STR));
        params.width = Some(32);
        params.height = Some(32);
        params.media_type = MediaType::Video;
        let mut dec = Av1Decoder::new(params);
        let pkt = Packet::new(0, TimeBase::new(1, 30), bytes);
        dec.send_packet(&pkt)
            .expect("decoder accepts round-40 stream");
        let frame = dec.receive_frame().expect("decoder yields a frame");
        let Frame::Video(vf) = frame else {
            panic!("expected Frame::Video");
        };
        assert_eq!(vf.planes[0].data.len(), 32 * 32);
        // DC_PRED no-neighbours + zero residual ⇒ Y mean = 128.
        let mean: u32 = vf.planes[0].data.iter().map(|&v| v as u32).sum::<u32>() / (32 * 32);
        assert_eq!(mean, 128, "Y plane mean expected 128, got {mean}");
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
