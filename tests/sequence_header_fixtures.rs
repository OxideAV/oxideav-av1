//! Integration tests for §5.5 `sequence_header_obu` parsing against
//! a clean-room fixture corpus.
//!
//! Each entry below is the verbatim SEQUENCE_HEADER OBU payload
//! (everything after the OBU header / extension / `obu_size` bytes)
//! extracted from an IVF fixture in
//! `docs/video/av1/fixtures/<name>/input.ivf`, paired with the
//! expected field values from the `SEQ_HEADER` line in the same
//! fixture's `trace.txt`. The trace events themselves are documented
//! in `docs/video/av1/av1-fixtures-and-traces.md`.
//!
//! Embedding the payloads here (instead of reading the fixture
//! directories at test time) keeps the integration test
//! self-contained — the crate is published to crates.io as a
//! stand-alone artifact and the `docs/` corpus is workspace-only.

use oxideav_av1::parse_sequence_header;

#[derive(Debug)]
struct Expected {
    profile: u8,
    still_picture: bool,
    reduced_still: bool,
    max_w: u32,
    max_h: u32,
    level0: u8,
    tier0: u8,
    num_ops: u8,
    use_128sb: bool,
    enable_filter_intra: bool,
    enable_intra_edge_filter: bool,
    enable_interintra: bool,
    enable_masked: bool,
    enable_warped: bool,
    enable_dual_filter: bool,
    enable_order_hint: bool,
    order_hint_bits: u8,
    enable_jnt_comp: bool,
    enable_ref_mvs: bool,
    enable_superres: bool,
    enable_cdef: bool,
    enable_restoration: bool,
    monochrome: bool,
    high_bitdepth: bool,
    twelve_bit: bool,
    color_range: bool,
    subsampling_x: bool,
    subsampling_y: bool,
    film_grain_present: bool,
}

struct Fixture {
    name: &'static str,
    payload: &'static [u8],
    expected: Expected,
}

/// All 16 fixtures from `docs/video/av1/fixtures/`. The hex payloads
/// were extracted from each fixture's `input.ivf` (strip the 32-byte
/// IVF file header + 12-byte frame header, walk the first
/// SEQUENCE_HEADER OBU and read its `obu_size` payload bytes). The
/// expected fields were copied from the `SEQ_HEADER` line at the top
/// of each fixture's `trace.txt`.
#[rustfmt::skip]
const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "tiny-i-only-16x16-prof0",
        payload: &[0x00,0x00,0x00,0x01,0x9f,0xfb,0xff,0xf3,0x00,0x80],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 16, max_h: 16, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "i-only-64x64-prof0",
        payload: &[0x00,0x00,0x00,0x02,0xaf,0xff,0xbf,0xff,0x30,0x08],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "i-frame-then-p-64x64",
        payload: &[0x00,0x00,0x00,0x02,0xaf,0xff,0xbf,0xff,0x30,0x08],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "profile-0-yuv420-8bit",
        payload: &[0x00,0x00,0x00,0x02,0xaf,0xff,0xbf,0xff,0x30,0x08],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "profile-1-yuv444-8bit",
        payload: &[0x20,0x00,0x00,0x02,0xaf,0xff,0xbf,0xff,0x30,0x40],
        expected: Expected {
            profile: 1, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: false, subsampling_y: false, film_grain_present: false,
        },
    },
    Fixture {
        name: "profile-2-yuv422-10bit",
        payload: &[0x40,0x00,0x00,0x02,0xaf,0xff,0xbf,0xff,0x38,0x10],
        expected: Expected {
            profile: 2, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: true, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: false, film_grain_present: false,
        },
    },
    Fixture {
        name: "profile-2-yuv422-12bit",
        payload: &[0x40,0x00,0x00,0x02,0xaf,0xff,0xbf,0xff,0x3c,0x44],
        expected: Expected {
            profile: 2, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: true, twelve_bit: true, color_range: false,
            subsampling_x: true, subsampling_y: false, film_grain_present: false,
        },
    },
    Fixture {
        name: "monochrome-grey-only",
        payload: &[0x00,0x00,0x00,0x02,0xaf,0xff,0xbf,0xff,0x35,0x40],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: true,
            high_bitdepth: false, twelve_bit: false, color_range: true,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "super-resolution",
        payload: &[0x18,0x19,0x7f,0xff,0xf8,0x04],
        expected: Expected {
            profile: 0, still_picture: true, reduced_still: true,
            max_w: 128, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: false, enable_masked: false, enable_warped: false,
            enable_dual_filter: false, enable_order_hint: false, order_hint_bits: 0,
            enable_jnt_comp: false, enable_ref_mvs: false, enable_superres: true,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "screen-content-tools",
        payload: &[0x18,0x1d,0xbf,0xff,0xf2,0x01],
        expected: Expected {
            profile: 0, still_picture: true, reduced_still: true,
            max_w: 256, max_h: 128, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: false, enable_masked: false, enable_warped: false,
            enable_dual_filter: false, enable_order_hint: false, order_hint_bits: 0,
            enable_jnt_comp: false, enable_ref_mvs: false, enable_superres: false,
            enable_cdef: false, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "film-grain-on",
        payload: &[0x00,0x00,0x00,0x02,0xaf,0xff,0x9b,0x5f,0x30,0x18],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: false, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: false, enable_masked: true, enable_warped: true,
            enable_dual_filter: false, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: false, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: true,
        },
    },
    Fixture {
        name: "superblocks-128",
        payload: &[0x00,0x00,0x00,0x03,0x37,0xff,0xef,0xff,0xcc,0x02],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 128, max_h: 128, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: true, enable_masked: true, enable_warped: true,
            enable_dual_filter: true, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: true, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "tile-cols-2-rows-1",
        payload: &[0x00,0x00,0x00,0x03,0xaf,0xff,0xe6,0xd7,0xcc,0x02],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 256, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: false, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: false, enable_masked: true, enable_warped: true,
            enable_dual_filter: false, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: false, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "show-existing-frame",
        payload: &[0x00,0x00,0x00,0x02,0xaf,0xff,0x9b,0x5f,0x30,0x08],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: false, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: false, enable_masked: true, enable_warped: true,
            enable_dual_filter: false, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: false, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "lossless-i-only",
        payload: &[0x18,0x15,0x7f,0xff,0xb0,0x08],
        expected: Expected {
            profile: 0, still_picture: true, reduced_still: true,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: true, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: false, enable_masked: false, enable_warped: false,
            enable_dual_filter: false, enable_order_hint: false, order_hint_bits: 0,
            enable_jnt_comp: false, enable_ref_mvs: false, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
    Fixture {
        name: "obu-with-extension-headers",
        payload: &[0x00,0x00,0x00,0x02,0xaf,0xff,0x9b,0x5f,0x30,0x08],
        expected: Expected {
            profile: 0, still_picture: false, reduced_still: false,
            max_w: 64, max_h: 64, level0: 0, tier0: 0, num_ops: 1,
            use_128sb: false, enable_filter_intra: true, enable_intra_edge_filter: true,
            enable_interintra: false, enable_masked: true, enable_warped: true,
            enable_dual_filter: false, enable_order_hint: true, order_hint_bits: 7,
            enable_jnt_comp: false, enable_ref_mvs: true, enable_superres: false,
            enable_cdef: true, enable_restoration: true, monochrome: false,
            high_bitdepth: false, twelve_bit: false, color_range: false,
            subsampling_x: true, subsampling_y: true, film_grain_present: false,
        },
    },
];

#[test]
fn all_corpus_fixtures_round_trip_sequence_header() {
    for fixture in FIXTURES {
        let sh = parse_sequence_header(fixture.payload).unwrap_or_else(|e| {
            panic!("fixture {}: parse failed: {e:?}", fixture.name);
        });
        let e = &fixture.expected;
        let mismatches = collect_mismatches(fixture.name, &sh, e);
        if !mismatches.is_empty() {
            panic!(
                "fixture {} mismatched fields:\n  {}",
                fixture.name,
                mismatches.join("\n  ")
            );
        }
    }
    assert_eq!(
        FIXTURES.len(),
        16,
        "embedded corpus shrank below 16 fixtures"
    );
}

#[allow(clippy::cognitive_complexity)]
fn collect_mismatches(name: &str, sh: &oxideav_av1::SequenceHeader, e: &Expected) -> Vec<String> {
    let mut m: Vec<String> = Vec::new();
    macro_rules! check {
        ($field:ident, $actual:expr) => {
            if e.$field != $actual {
                m.push(format!(
                    "{}: {} expected {:?} got {:?}",
                    name,
                    stringify!($field),
                    e.$field,
                    $actual
                ));
            }
        };
    }
    check!(profile, sh.seq_profile);
    check!(still_picture, sh.still_picture);
    check!(reduced_still, sh.reduced_still_picture_header);
    check!(max_w, sh.max_frame_width_minus_1 + 1);
    check!(max_h, sh.max_frame_height_minus_1 + 1);
    check!(level0, sh.operating_points[0].seq_level_idx);
    check!(tier0, sh.operating_points[0].seq_tier);
    check!(num_ops, sh.operating_points_cnt_minus_1 + 1);
    check!(use_128sb, sh.use_128x128_superblock);
    check!(enable_filter_intra, sh.enable_filter_intra);
    check!(enable_intra_edge_filter, sh.enable_intra_edge_filter);
    check!(enable_interintra, sh.enable_interintra_compound);
    check!(enable_masked, sh.enable_masked_compound);
    check!(enable_warped, sh.enable_warped_motion);
    check!(enable_dual_filter, sh.enable_dual_filter);
    check!(enable_order_hint, sh.enable_order_hint);
    check!(order_hint_bits, sh.order_hint_bits);
    check!(enable_jnt_comp, sh.enable_jnt_comp);
    check!(enable_ref_mvs, sh.enable_ref_frame_mvs);
    check!(enable_superres, sh.enable_superres);
    check!(enable_cdef, sh.enable_cdef);
    check!(enable_restoration, sh.enable_restoration);
    check!(monochrome, sh.color_config.mono_chrome);
    check!(high_bitdepth, sh.color_config.high_bitdepth);
    check!(twelve_bit, sh.color_config.twelve_bit);
    check!(color_range, sh.color_config.color_range);
    check!(subsampling_x, sh.color_config.subsampling_x);
    check!(subsampling_y, sh.color_config.subsampling_y);
    check!(film_grain_present, sh.film_grain_params_present);
    m
}
