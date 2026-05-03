//! Round-1 + round-2 encoder smoke tests.
//!
//! Round 1 (May 2026, shipped): OBU framing + sequence header + frame
//! header. Tile group is a 16-byte placeholder; self-roundtrip via the
//! existing decoder gets to `parse_tile_group_header` then surfaces
//! `Error::Unsupported` from the coefficient decoder.
//!
//! Round 2 (May 2026, in progress): forward range coder
//! (`SymbolEncoder`) lands; `decode(encode(.))` roundtrip pinned in
//! the encoder's unit tests. Items 2-6 (partition emit, intra mode
//! emit, TX-type emit, forward DCT pinning, coefficient entropy emit)
//! are documented as round-3 deferrals in `crates/oxideav-av1/src/
//! encoder/mod.rs`.

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

/// Round-2 — the forward range coder ([`oxideav_av1::encoder::symbol::
/// SymbolEncoder`]) is bit-exact pinned against
/// [`oxideav_av1::symbol::SymbolDecoder`] via a `decode(encode(.))`
/// roundtrip. The encoder's unit tests already exercise the same
/// invariant on multiple CDFs and adaptation modes; this integration
/// test pins the public API surface so a later refactor can't drop
/// the roundtrip property without breaking the integration suite too.
#[test]
fn round2_symbol_encoder_decoder_roundtrip_default_partition_cdf() {
    use oxideav_av1::encoder::symbol::SymbolEncoder;
    use oxideav_av1::symbol::SymbolDecoder;

    // Use the real DEFAULT_PARTITION_CDF — exercises the production
    // CDF shape rather than a synthetic uniform.
    let template = oxideav_av1::cdfs::DEFAULT_PARTITION_CDF[0].to_vec();
    let mut enc_cdf = template.clone();
    let mut dec_cdf = template.clone();
    let symbols: Vec<u32> = (0..50).map(|i| ((i * 11 + 3) % 4) as u32).collect();

    let mut enc = SymbolEncoder::new(true);
    for &s in &symbols {
        enc.encode_symbol(&mut enc_cdf, s);
    }
    let buf = enc.finish();
    assert!(
        buf.len() >= 2,
        "round-2 SymbolEncoder must emit ≥ 2 bytes for init_symbol"
    );

    let mut dec = SymbolDecoder::new(&buf, buf.len(), true).expect("init_symbol");
    for (i, &want) in symbols.iter().enumerate() {
        let got = dec.decode_symbol(&mut dec_cdf).expect("decode_symbol");
        assert_eq!(
            got, want,
            "round-2 SymbolEncoder roundtrip mismatch at symbol {i}: \
             want {want}, got {got}"
        );
    }
}
