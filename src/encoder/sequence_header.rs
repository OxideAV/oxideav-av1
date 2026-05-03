//! AV1 sequence header writer — §5.5 / §6.4 inverse.
//!
//! Round 1 scope: profile 0 (Main, 8-bit 4:2:0), still-picture stream
//! with `reduced_still_picture_header=1`. This is the minimum subset
//! the spec defines: a single operating point, the bare minimum tool
//! flags, no frame-id signalling, no decoder-model timing, no
//! superres / CDEF / loop restoration / film grain. Many bits in the
//! full §5.5.1 tree become implicit defaults under this short path.
//!
//! The encoder is intentionally narrow: extending to inter/profile 1+/
//! superres/etc. is a round-2+ exercise once the entropy coder is
//! online.

use crate::encoder::bitwriter::{ceil_log2, BitWriter};

/// Round-1 sequence configuration. Just the dimensions; everything
/// else is held at the spec-minimum defaults so the matching decoder
/// state is fully constrained.
#[derive(Clone, Copy, Debug)]
pub struct EncSequence {
    pub width: u32,
    pub height: u32,
}

/// Emit the Sequence Header OBU body (everything between the OBU
/// `size` prefix and the next OBU header). The byte sequence is
/// terminated with `trailing_bits()` per §5.3.4 and byte-aligned.
///
/// Round 1 payload layout (§5.5.1 with `reduced_still_picture_header=1`):
///
/// ```text
///   seq_profile                      f(3) = 0
///   still_picture                    f(1) = 1
///   reduced_still_picture_header     f(1) = 1
///   seq_level_idx_0                  f(5) = 0   (Level 2.0)
///   frame_width_bits_minus_1         f(4)
///   frame_height_bits_minus_1        f(4)
///   max_frame_width_minus_1          f(frame_width_bits)
///   max_frame_height_minus_1         f(frame_height_bits)
///   use_128x128_superblock           f(1) = 0   (64x64 SB)
///   enable_filter_intra              f(1) = 0
///   enable_intra_edge_filter         f(1) = 0
///   enable_superres                  f(1) = 0
///   enable_cdef                      f(1) = 0
///   enable_restoration               f(1) = 0
///   color_config:
///     high_bitdepth                  f(1) = 0
///     mono_chrome                    f(1) = 0  (profile != 1)
///     color_description_present      f(1) = 0
///     color_range                    f(1) = 0
///     // subsampling_x/y implicit = 1/1 for profile 0
///     chroma_sample_position         f(2) = 0  (CSP_UNKNOWN)
///     separate_uv_deltas             f(1) = 0
///   film_grain_params_present        f(1) = 0
///   trailing_bits()                  1 + zero pad
/// ```
pub fn write_sequence_header_payload(seq: &EncSequence) -> Vec<u8> {
    let mut bw = BitWriter::new();

    // §5.5.1
    bw.f(3, 0); // seq_profile = 0
    bw.bit(true); // still_picture = 1
    bw.bit(true); // reduced_still_picture_header = 1

    // Reduced-still short path.
    bw.f(5, 0); // seq_level_idx[0] = 0 (Level 2.0)

    // Frame size signalling. frame_width_bits == ceil_log2(max_w);
    // frame_width_bits_minus_1 = frame_width_bits - 1 ∈ 0..=15.
    let fw_bits = ceil_log2(seq.width).max(1);
    let fh_bits = ceil_log2(seq.height).max(1);
    bw.f(4, fw_bits - 1);
    bw.f(4, fh_bits - 1);
    bw.f(fw_bits, seq.width - 1);
    bw.f(fh_bits, seq.height - 1);

    // Tool-flag block (the reduced-still path skips most of these by
    // defaulting them, but the bits enable_filter_intra / enable_intra_
    // edge_filter / enable_superres / enable_cdef / enable_restoration
    // are still coded per §5.5.1).
    bw.bit(false); // use_128x128_superblock
    bw.bit(false); // enable_filter_intra
    bw.bit(false); // enable_intra_edge_filter
    bw.bit(false); // enable_superres
    bw.bit(false); // enable_cdef
    bw.bit(false); // enable_restoration

    // color_config — §5.5.2.
    bw.bit(false); // high_bitdepth = 0
                   // seq_profile != 1 ⇒ monochrome bit is coded.
    bw.bit(false); // mono_chrome = 0
    bw.bit(false); // color_description_present_flag = 0
                   // color_range — coded for profile 0 non-mono.
    bw.bit(false); // color_range = 0
                   // Profile 0: subsampling_x = subsampling_y = 1 implicitly.
                   // chroma_sample_position is read when both subsampling flags are 1.
    bw.f(2, 0); // CSP_UNKNOWN
    bw.bit(false); // separate_uv_deltas

    bw.bit(false); // film_grain_params_present

    // Terminating §5.3.4 — write a 1 bit then byte-align with zero pad.
    bw.write_obu_trailing_bits();

    bw.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_header::parse_sequence_header;

    #[test]
    fn round_trip_64x64() {
        let seq = EncSequence {
            width: 64,
            height: 64,
        };
        let payload = write_sequence_header_payload(&seq);
        let parsed = parse_sequence_header(&payload).expect("parse");
        assert_eq!(parsed.seq_profile, 0);
        assert!(parsed.still_picture);
        assert!(parsed.reduced_still_picture_header);
        assert_eq!(parsed.max_frame_width, 64);
        assert_eq!(parsed.max_frame_height, 64);
        assert_eq!(parsed.color_config.bit_depth, 8);
        assert!(!parsed.color_config.mono_chrome);
        assert_eq!(parsed.color_config.num_planes, 3);
        assert!(parsed.color_config.subsampling_x);
        assert!(parsed.color_config.subsampling_y);
        assert!(!parsed.use_128x128_superblock);
        assert!(!parsed.enable_cdef);
        assert!(!parsed.enable_restoration);
        assert!(!parsed.film_grain_params_present);
    }

    #[test]
    fn round_trip_16x16() {
        let seq = EncSequence {
            width: 16,
            height: 16,
        };
        let payload = write_sequence_header_payload(&seq);
        let parsed = parse_sequence_header(&payload).expect("parse");
        assert_eq!(parsed.max_frame_width, 16);
        assert_eq!(parsed.max_frame_height, 16);
    }

    #[test]
    fn round_trip_320x240() {
        let seq = EncSequence {
            width: 320,
            height: 240,
        };
        let payload = write_sequence_header_payload(&seq);
        let parsed = parse_sequence_header(&payload).expect("parse");
        assert_eq!(parsed.max_frame_width, 320);
        assert_eq!(parsed.max_frame_height, 240);
    }
}
