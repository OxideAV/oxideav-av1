//! Round-1 encoder smoke test.
//!
//! Round 1 lands the OBU framing + sequence header + frame header. The
//! tile group payload is a placeholder (round 2+ wires the entropy
//! coder). Self-roundtrip via the existing decoder therefore goes
//! `headers OK → tile group surfaces Unsupported`. We assert the
//! intermediate parse milestones explicitly so the next round has a
//! clear baseline to extend.

use oxideav_av1::dpb::Dpb;
use oxideav_av1::encoder::{write_keyframe_stream, FrameConfig, SequenceConfig};
use oxideav_av1::frame_header::{parse_frame_obu_with_dpb, FrameType};
use oxideav_av1::obu::{iter_obus, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;

#[test]
fn round1_keyframe_obu_framing_parses_64x64() {
    let seq = SequenceConfig {
        width: 64,
        height: 64,
    };
    let frame = FrameConfig { base_q_idx: 100 };
    let bytes = write_keyframe_stream(&seq, &frame);

    let obus: Vec<_> = iter_obus(&bytes).map(|r| r.expect("OBU parse")).collect();
    assert_eq!(obus.len(), 3, "TD + SH + FRAME");
    assert_eq!(obus[0].header.obu_type, ObuType::TemporalDelimiter);
    assert_eq!(obus[1].header.obu_type, ObuType::SequenceHeader);
    assert_eq!(obus[2].header.obu_type, ObuType::Frame);
}

#[test]
fn round1_sequence_header_decodes_to_round1_envelope() {
    let seq = SequenceConfig {
        width: 32,
        height: 32,
    };
    let bytes = write_keyframe_stream(&seq, &FrameConfig { base_q_idx: 80 });
    let obus: Vec<_> = iter_obus(&bytes).map(|r| r.unwrap()).collect();
    let sh = parse_sequence_header(obus[1].payload).expect("seq parse");
    assert_eq!(sh.seq_profile, 0);
    assert!(sh.still_picture);
    assert!(sh.reduced_still_picture_header);
    assert_eq!(sh.max_frame_width, 32);
    assert_eq!(sh.max_frame_height, 32);
    assert_eq!(sh.color_config.bit_depth, 8);
    assert!(!sh.color_config.mono_chrome);
    assert!(sh.color_config.subsampling_x);
    assert!(sh.color_config.subsampling_y);
    assert!(!sh.use_128x128_superblock);
    assert!(!sh.enable_cdef);
    assert!(!sh.enable_restoration);
    assert!(!sh.film_grain_params_present);
}

#[test]
fn round1_frame_header_decodes_to_keyframe_intra_only() {
    let seq = SequenceConfig {
        width: 16,
        height: 16,
    };
    let bytes = write_keyframe_stream(&seq, &FrameConfig { base_q_idx: 100 });
    let obus: Vec<_> = iter_obus(&bytes).map(|r| r.unwrap()).collect();
    let sh = parse_sequence_header(obus[1].payload).unwrap();
    let (fh, tg_payload) =
        parse_frame_obu_with_dpb(&sh, obus[2].payload, &Dpb::new()).expect("frame parse");

    assert_eq!(fh.frame_type, FrameType::Key);
    assert!(fh.show_frame);
    assert!(fh.error_resilient_mode);
    assert_eq!(fh.frame_width, 16);
    assert_eq!(fh.frame_height, 16);
    assert_eq!(fh.quant.base_q_idx, 100);
    assert_eq!(fh.allow_screen_content_tools, 0);
    assert!(!fh.allow_intrabc);

    let ti = fh.tile_info.as_ref().expect("tile_info populated");
    assert_eq!(ti.tile_cols, 1);
    assert_eq!(ti.tile_rows, 1);
    assert_eq!(ti.tile_size_bytes, 0);

    // Round 1: tile group is the 16-byte stub. Round 2+ replaces this
    // with a real entropy-coded payload.
    assert_eq!(
        tg_payload.len(),
        oxideav_av1::encoder::tile::ROUND1_STUB_TILE_BYTES
    );
}
