//! AV1 uncompressed frame header writer — §5.9 inverse, round 1 scope.
//!
//! Round 1 coverage:
//!
//! - `reduced_still_picture_header = 1` — `frame_type=Key`,
//!   `show_frame=1`, `error_resilient_mode=true` are implied; the
//!   bitstream skips the `show_existing_frame / frame_type / show_frame
//!   / showable_frame / error_resilient_mode` triplet entirely.
//! - `disable_cdf_update = 1` — round 1 emits every frame as a fresh
//!   single-frame still, the decoder's CDFs reset per-tile.
//! - `allow_screen_content_tools = 0` — disables the screen-content
//!   bit pile (`force_integer_mv` / `allow_intrabc`).
//! - Single tile, single 64×64 superblock, no tiling bits.
//! - `base_q_idx` from caller; all per-plane / per-segment deltas zero.
//! - `using_qmatrix = 0`.
//! - Segmentation off, delta_q off, delta_lf off.
//! - `loop_filter_level = 0` ⇒ filter disabled (no per-plane bits coded
//!   under that condition).
//! - CDEF + LR skipped because the sequence header disables them.
//! - `tx_mode = Only4x4` (`coded_lossless` true; round 1 does only 4×4
//!   transforms anyway). Round 2 will switch to `tx_mode_select_bit=1`
//!   and `Largest` / `Select` once larger transforms ship.
//! - `reduced_tx_set = 1`.
//!
//! Output is the **frame-header body** (the bits that go after the
//! `OBU_FRAME` size prefix and before the tile group payload). The
//! body terminates with `trailing_bits()` and a `byte_alignment()`
//! per §5.9.1 so the tile group OBU body that follows is byte-aligned.

use crate::encoder::bitwriter::{ceil_log2, BitWriter};
use crate::encoder::sequence_header::EncSequence;

/// Round-1 frame configuration. The encoder enforces these to keep the
/// matched decoder path narrow.
#[derive(Clone, Copy, Debug)]
pub struct EncFrame {
    /// `base_q_idx` (0..=255). 0 ⇒ lossless intent (round 1 still
    /// emits `tx_mode = Only4x4` because `coded_lossless` is derived
    /// from the *quant* state).
    pub base_q_idx: u8,
}

/// Emit the frame-header body. For `OBU_FRAME` callers should follow
/// the returned bytes immediately with the tile-group payload (no
/// extra alignment is needed — the writer leaves the buffer byte-
/// aligned).
pub fn write_frame_header_body(seq: &EncSequence, frame: &EncFrame) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // §5.9.1 — `reduced_still_picture_header == 1` short-circuits the
    // first 4-5 bits. Implicit: frame_type = Key, show_frame = 1,
    // showable_frame = 0, error_resilient_mode = true.

    // disable_cdf_update — 1 bit. Round 1 keeps the per-frame CDF state
    // local; the decoder's TileDecoder re-inits CDFs from defaults on
    // every tile.
    bw.bit(true);

    // allow_screen_content_tools — coded because
    // `seq_force_screen_content_tools = SELECT_SCREEN_CONTENT_TOOLS = 2`
    // under reduced_still. Round 1 sets it to 0 to suppress the
    // `force_integer_mv` / `allow_intrabc` follow-ups.
    bw.f(1, 0);
    // Skip `force_integer_mv` (gated by allow_screen_content_tools != 0).
    // Skip `current_frame_id` (frame_id_numbers_present = 0).
    // Skip `frame_size_override_flag` (reduced_still ⇒ implicit 0).
    // Skip `order_hint` (enable_order_hint = 0).
    // primary_ref_frame is derived as PRIMARY_REF_NONE (intra), no bits.
    // No decoder_model_info_present.

    // refresh_frame_flags — `frame_type=Key && show_frame=1` implies
    // all-frames bitmask, **no bits coded**.

    // No ref_order_hint loop (intra & refresh==all_frames).

    // §5.9.5 frame_size: reduced_still ⇒ frame_size_override_flag=0,
    // dimensions come from sequence header — no bits coded.
    // §superres: enable_superres=0 ⇒ no bits.
    // §5.9.6 render_size: 1 bit `render_and_frame_size_different`.
    bw.bit(false);

    // §5.9.1 allow_intrabc — gated by `allow_screen_content_tools != 0`,
    // which we just zeroed. Suppressed.

    // §5.9.1 disable_frame_end_update_cdf — `seq.reduced_still_picture_
    // header || disable_cdf_update` short-circuits to true with no bit.

    // §5.9.15 tile_info(). Single-tile / single-64×64-SB frame: no bits
    // coded under uniform branch (min_log2 == max_log2 == 0 for both
    // axes), and tile_size_bytes / context_update_tile_id only emitted
    // when `tile_cols_log2 > 0 || tile_rows_log2 > 0`.
    write_tile_info_single_tile(&mut bw, seq);

    // §5.9.12 quantization_params.
    bw.f(8, frame.base_q_idx as u32); // base_q_idx
    write_delta_q(&mut bw, 0); // delta_q_y_dc
                               // num_planes > 1 ⇒ enter chroma block. separate_uv_deltas was 0 in
                               // the sequence header, so we don't code `diff_uv_delta`. We code
                               // delta_q_u_dc + delta_q_u_ac.
    write_delta_q(&mut bw, 0); // delta_q_u_dc
    write_delta_q(&mut bw, 0); // delta_q_u_ac
    bw.bit(false); // using_qmatrix

    // §5.9.14 segmentation_params.
    bw.bit(false); // segmentation_enabled = 0
                   // primary_ref_frame == PRIMARY_REF_NONE forces update_map = 1,
                   // temporal_update = 0, update_data = 1 — but only when enabled.
                   // Disabled ⇒ no further bits.

    // §5.9.16 delta_q_params + delta_lf_params — both gated by
    // `base_q_idx != 0`. With base_q_idx == 0 (lossless intent) the
    // entire delta_q + delta_lf section is implicit.
    if frame.base_q_idx != 0 {
        bw.bit(false); // delta_q_present
                       // delta_lf_present is gated on delta_q_present,
                       // so its bit is also suppressed.
    }

    // §5.9.11 loop_filter_params.
    // `coded_lossless_hint` is true iff base_q_idx == 0 AND every
    // delta is zero (we've kept them zero). Under coded_lossless the
    // decoder takes the `mode_ref_delta_enabled = true` early-return
    // and reads NO bits.
    if frame.base_q_idx != 0 {
        // level_y0
        bw.f(6, 0);
        // level_y1
        bw.f(6, 0);
        // num_planes > 1 && (level_y0 != 0 || level_y1 != 0) — neither
        // is set, so chroma levels skipped.
        bw.f(3, 0); // sharpness
        bw.bit(false); // mode_ref_delta_enabled = 0
                       // mode_ref_delta_update bit suppressed.
    }

    // §5.9.19 cdef_params — `coded_lossless || allow_intrabc ||
    // !seq.enable_cdef` ⇒ no bits. enable_cdef = 0 in sequence header.

    // §5.9.20 lr_params — same gating: enable_restoration = 0 ⇒ no
    // bits.

    // §5.9.21 read_tx_mode. coded_lossless ⇒ Only4x4 with no bit.
    // Otherwise we code `1` bit; 0 = Largest, 1 = Select. Round 1 uses
    // Largest (4×4 fits because TX_MODE_LARGEST = use the largest TX
    // size for each block, and small blocks use 4×4 transforms).
    if frame.base_q_idx != 0 {
        bw.bit(false); // tx_mode select bit = 0 ⇒ Largest
    }

    // §5.9.22 reference_select. frame_is_intra ⇒ 0 with no bit.

    // §5.9.23 skip_mode_present — `skip_mode_allowed` is false for an
    // intra-only frame, so no bit.

    // §5.9.25 allow_warped_motion — gated by !frame_is_intra ⇒ no bit.

    // reduced_tx_set — 1 bit. Round 1 sets it to 1 (smaller TX-type set).
    bw.bit(true);

    // No global_motion_params (intra), no film_grain (sequence header
    // film_grain_params_present == 0).

    // For OBU_FRAME the spec sequence is `uncompressed_header()` then
    // `byte_alignment()` then `tile_group_obu(...)` (§5.10). The
    // alignment is what separates header bits from the tile-group
    // body — `trailing_bits()` is NOT used here because there is no
    // trailing zero region between the header and tile group. We
    // simply zero-pad to the next byte boundary so the tile-group
    // payload starts byte-aligned.
    bw.byte_align();

    bw.finish()
}

/// `tile_info()` for a frame whose width AND height fit in a single
/// 64×64 superblock. No bits beyond `uniform_tile_spacing_flag = 1`.
fn write_tile_info_single_tile(bw: &mut BitWriter, seq: &EncSequence) {
    let sb_size_log2 = 6; // use_128x128_superblock = 0 ⇒ 64-sample SB
    let sb_cols = ((seq.width + (1u32 << sb_size_log2) - 1) >> sb_size_log2).max(1);
    let sb_rows = ((seq.height + (1u32 << sb_size_log2) - 1) >> sb_size_log2).max(1);
    debug_assert_eq!(sb_cols, 1, "round 1 supports single-SB-wide frames only");
    debug_assert_eq!(sb_rows, 1, "round 1 supports single-SB-tall frames only");

    // uniform_tile_spacing_flag = 1.
    bw.bit(true);
    // min_log2_tile_cols == max_log2_tile_cols == 0 ⇒ NO increment
    // bits coded.
    // Same for rows.
    // tile_cols_log2 == tile_rows_log2 == 0 ⇒ NO context_update_tile_id
    // / tile_size_bytes bits coded.
    let _ = ceil_log2; // silence
}

/// `read_delta_q()` inverse — emits a 1-bit presence flag, optionally
/// followed by a 7-bit signed value.
fn write_delta_q(bw: &mut BitWriter, value: i8) {
    if value == 0 {
        bw.bit(false);
    } else {
        bw.bit(true);
        bw.su(7, value as i32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dpb::Dpb;
    use crate::encoder::sequence_header::write_sequence_header_payload;
    use crate::frame_header::{parse_frame_header_with_dpb, FrameType};
    use crate::sequence_header::parse_sequence_header;

    fn parse_pair(seq: EncSequence, frame: EncFrame) -> crate::frame_header::FrameHeader {
        let seq_payload = write_sequence_header_payload(&seq);
        let parsed_seq = parse_sequence_header(&seq_payload).unwrap();
        let body = write_frame_header_body(&seq, &frame);
        parse_frame_header_with_dpb(&parsed_seq, &body, &Dpb::new()).unwrap()
    }

    #[test]
    fn frame_header_roundtrip_64x64_q120() {
        let seq = EncSequence {
            width: 64,
            height: 64,
        };
        let fh = parse_pair(seq, EncFrame { base_q_idx: 120 });
        assert_eq!(fh.frame_type, FrameType::Key);
        assert!(fh.show_frame);
        assert!(fh.error_resilient_mode);
        assert_eq!(fh.frame_width, 64);
        assert_eq!(fh.frame_height, 64);
        assert_eq!(fh.quant.base_q_idx, 120);
        assert!(!fh.allow_intrabc);
        assert_eq!(fh.allow_screen_content_tools, 0);
        let ti = fh.tile_info.as_ref().unwrap();
        assert_eq!(ti.tile_cols, 1);
        assert_eq!(ti.tile_rows, 1);
        assert_eq!(ti.tile_size_bytes, 0);
    }

    #[test]
    fn frame_header_roundtrip_16x16_q100() {
        let seq = EncSequence {
            width: 16,
            height: 16,
        };
        let fh = parse_pair(seq, EncFrame { base_q_idx: 100 });
        assert_eq!(fh.frame_width, 16);
        assert_eq!(fh.frame_height, 16);
        assert_eq!(fh.quant.base_q_idx, 100);
    }

    #[test]
    fn frame_header_roundtrip_lossless_q0() {
        let seq = EncSequence {
            width: 32,
            height: 32,
        };
        let fh = parse_pair(seq, EncFrame { base_q_idx: 0 });
        assert_eq!(fh.quant.base_q_idx, 0);
    }
}
