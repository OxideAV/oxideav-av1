//! Integration tests for the §5.9.2 `uncompressed_header()` prefix
//! parser against the same clean-room fixture corpus that the
//! sequence-header tests use.
//!
//! Each entry embeds:
//!
//!   * The verbatim SEQUENCE_HEADER OBU payload, so the test parses
//!     it fresh through round 2's `parse_sequence_header`.
//!   * The verbatim FRAME / FRAME_HEADER OBU payload from the first
//!     IVF frame, so the test parses the first per-frame
//!     `uncompressed_header()` prefix through round 3's
//!     `parse_frame_header`.
//!   * The trace fields the parsed [`FrameHeader`] is expected to
//!     match, captured from the `FRAME_HEADER` line at `idx=0` in
//!     each fixture's `trace.txt`.
//!
//! Both payloads were extracted by walking the IVF frame data and
//! decoding OBU headers per §5.3.x — i.e. the same procedure the
//! crate's `parse_obu` walker would perform at runtime, but inlined
//! here as static data so the integration test stays self-contained
//! (the `docs/` corpus is workspace-only and the crate is published
//! independently).
//!
//! Note: the trace's `force_integer_mv` column reports the **raw
//! bitstream bit** before the §5.9.2 `if (FrameIsIntra)` override
//! fires. Every fixture in the corpus is a `KEY_FRAME` (the first
//! frame of each clip), so [`FrameHeader::force_integer_mv`] reports
//! the post-override value of `1`. The test asserts on the
//! post-override value; the raw bit is preserved as documentation
//! only.

use oxideav_av1::{parse_frame_header, parse_sequence_header, FrameType};

#[derive(Debug)]
struct Expected {
    show_existing_frame: bool,
    frame_to_show_map_idx: Option<u8>,
    frame_type: FrameType,
    show_frame: bool,
    showable_frame: bool,
    error_resilient_mode: bool,
    disable_cdf_update: bool,
    allow_screen_content_tools: bool,
    /// Post-override value (raw bit ANDed with `!FrameIsIntra` then
    /// ORed with `FrameIsIntra` — i.e. for an intra frame this is
    /// always `true`).
    force_integer_mv: bool,
    order_hint: u32,
    primary_ref_frame: u8,
    refresh_frame_flags: u8,
}

struct Fixture {
    name: &'static str,
    seq_payload: &'static [u8],
    frame_payload: &'static [u8],
    expected: Expected,
}

/// The 16 fixtures in `docs/video/av1/fixtures/`. Every clip's first
/// frame is a `KEY_FRAME` with `show_frame == 1` *except*
/// `show-existing-frame` whose first IVF frame contains both the
/// (hidden) KEY and the inter-coded P frame as separate OBUs — we
/// test the underlying KEY OBU here (`show_frame == 0`).
#[rustfmt::skip]
const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "tiny-i-only-16x16-prof0",
        seq_payload: &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80],
        frame_payload: &[
            0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f,
            0x67, 0x6c, 0xc7, 0xee, 0x51, 0x80,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: false, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "i-only-64x64-prof0",
        seq_payload: &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x30, 0x08],
        frame_payload: &[
            0x14, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x90, 0x3c, 0x42, 0xa0, 0xdd, 0xe9, 0x5c, 0xfb,
            0x95, 0x31,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "i-frame-then-p-64x64",
        seq_payload: &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x30, 0x08],
        frame_payload: &[
            0x14, 0x00, 0x2b, 0xe0, 0x00, 0x09, 0x42, 0x09, 0x00, 0x2c, 0x2a, 0xe3, 0x30, 0x6d,
            0xea, 0xa6,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "profile-0-yuv420-8bit",
        seq_payload: &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x30, 0x08],
        frame_payload: &[
            0x14, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x90, 0x83, 0x00, 0x80, 0xf1, 0x87, 0x59, 0x8f,
            0x49, 0xd9,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "profile-1-yuv444-8bit",
        seq_payload: &[0x20, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x30, 0x40],
        frame_payload: &[
            0x14, 0x00, 0x2f, 0x00, 0x00, 0x20, 0x82, 0x09, 0x28, 0x0c, 0x00, 0xb4, 0xc9, 0x11,
            0x82, 0x60,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "profile-2-yuv422-10bit",
        seq_payload: &[0x40, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x38, 0x10],
        frame_payload: &[
            0x10, 0x00, 0xbc, 0x00, 0x00, 0x82, 0x0c, 0x24, 0x64, 0x70, 0x20, 0xe8, 0x36, 0x0a,
            0x13, 0xfc,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: false, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "profile-2-yuv422-12bit",
        seq_payload: &[0x40, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x3c, 0x44],
        frame_payload: &[
            0x14, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x90, 0x51, 0x40, 0x00, 0x32, 0xe1, 0x82, 0xd8,
            0x47, 0xd8,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "monochrome-grey-only",
        seq_payload: &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0xbf, 0xff, 0x35, 0x40],
        frame_payload: &[
            0x14, 0x00, 0x2f, 0x00, 0x00, 0x02, 0x41, 0x40, 0x15, 0x6b, 0x4b, 0xd0, 0xb6, 0x75,
            0x80,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "super-resolution",
        seq_payload: &[0x18, 0x19, 0x7f, 0xff, 0xf8, 0x04],
        frame_payload: &[
            0x56, 0xd0, 0x00, 0x61, 0x87, 0x20, 0x28, 0x2b, 0xaa, 0xa8, 0xea, 0xd7, 0xb3, 0x38,
            0x65, 0x52,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "screen-content-tools",
        seq_payload: &[0x18, 0x1d, 0xbf, 0xff, 0xf2, 0x01],
        frame_payload: &[
            0x44, 0xa0, 0x00, 0x00, 0x08, 0x08, 0xfa, 0x05, 0x6c, 0x9a, 0xa3, 0x8a, 0xd0, 0x6b,
            0x9a, 0x06,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "film-grain-on",
        seq_payload: &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x9b, 0x5f, 0x30, 0x18],
        frame_payload: &[
            0x14, 0x00, 0x2f, 0x00, 0x08, 0x21, 0xc5, 0x89, 0x02, 0x88, 0x0b, 0x61, 0x5f, 0xc2,
            0x00, 0x03,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "superblocks-128",
        seq_payload: &[0x00, 0x00, 0x00, 0x03, 0x37, 0xff, 0xef, 0xff, 0xcc, 0x02],
        frame_payload: &[
            0x10, 0x00, 0xc0, 0x00, 0x69, 0x89, 0x9c, 0x29, 0x61, 0xc1, 0xcc, 0xf2, 0x80, 0xc4,
            0xc9, 0xc5,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: false, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "tile-cols-2-rows-1",
        seq_payload: &[0x00, 0x00, 0x00, 0x03, 0xaf, 0xff, 0xe6, 0xd7, 0xcc, 0x02],
        frame_payload: &[
            0x14, 0x00, 0x31, 0x78, 0x00, 0x92, 0x50, 0x1c, 0x4a, 0x10, 0x51, 0x44, 0x04, 0x00,
            0x0e, 0x02,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "show-existing-frame",
        seq_payload: &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x9b, 0x5f, 0x30, 0x08],
        frame_payload: &[
            0x01, 0x00, 0x7f, 0x89, 0x10, 0x02, 0x08, 0x10, 0x82, 0x00, 0x0a, 0x02, 0xdc, 0x85,
            0x28, 0xf5,
        ],
        expected: Expected {
            // First IVF frame's first FRAME OBU: hidden KEY (show_frame=0).
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: false, showable_frame: false,
            error_resilient_mode: false, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "lossless-i-only",
        seq_payload: &[0x18, 0x15, 0x7f, 0xff, 0xb0, 0x08],
        frame_payload: &[
            0x44, 0x00, 0x00, 0xdc, 0xa5, 0x89, 0xe9, 0x7b, 0xc8, 0xd6, 0xfa, 0xb2, 0x3f, 0x01,
            0x74, 0x0b,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
    Fixture {
        name: "obu-with-extension-headers",
        seq_payload: &[0x00, 0x00, 0x00, 0x02, 0xaf, 0xff, 0x9b, 0x5f, 0x30, 0x08],
        frame_payload: &[
            0x14, 0x00, 0x2d, 0x40, 0x08, 0x20, 0xa3, 0x89, 0x0a, 0x88, 0x08, 0xdb, 0x71, 0xc0,
            0xfe, 0x90,
        ],
        expected: Expected {
            show_existing_frame: false, frame_to_show_map_idx: None,
            frame_type: FrameType::Key, show_frame: true, showable_frame: false,
            error_resilient_mode: true, disable_cdf_update: false,
            allow_screen_content_tools: true, force_integer_mv: true,
            order_hint: 0, primary_ref_frame: 7, refresh_frame_flags: 0xff,
        },
    },
];

#[test]
fn all_corpus_fixtures_round_trip_frame_header_prefix() {
    for fx in FIXTURES {
        let seq = parse_sequence_header(fx.seq_payload)
            .unwrap_or_else(|e| panic!("fixture {}: seq parse failed: {e:?}", fx.name));
        let fh = parse_frame_header(fx.frame_payload, &seq)
            .unwrap_or_else(|e| panic!("fixture {}: frame header parse failed: {e:?}", fx.name));
        let mut mismatches: Vec<String> = Vec::new();
        macro_rules! check {
            ($field:ident) => {
                if fx.expected.$field != fh.$field {
                    mismatches.push(format!(
                        "{}: {} expected {:?} got {:?}",
                        fx.name,
                        stringify!($field),
                        fx.expected.$field,
                        fh.$field,
                    ));
                }
            };
        }
        check!(show_existing_frame);
        check!(frame_to_show_map_idx);
        check!(frame_type);
        check!(show_frame);
        check!(showable_frame);
        check!(error_resilient_mode);
        check!(disable_cdf_update);
        check!(allow_screen_content_tools);
        check!(force_integer_mv);
        check!(order_hint);
        check!(primary_ref_frame);
        check!(refresh_frame_flags);
        if !mismatches.is_empty() {
            panic!(
                "fixture {} mismatched fields:\n  {}",
                fx.name,
                mismatches.join("\n  ")
            );
        }
    }
    assert_eq!(
        FIXTURES.len(),
        16,
        "embedded frame-header corpus shrank below 16 fixtures"
    );
}
