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

use oxideav_av1::{parse_frame_header, parse_sequence_header, FrameType, SUPERRES_NUM};

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
    /// `w=` column from the `FRAME_HEADER` trace line.
    ///
    /// Per `docs/video/av1/av1-fixtures-and-traces.md` §"`FRAME_HEADER`"
    /// this is `frame_width_minus_1 + 1` — i.e. the coded
    /// **pre-superres** width that §5.9.8 then assigns to
    /// `UpscaledWidth` before downscaling to the
    /// session-FrameWidth. The asserted value below is
    /// [`FrameSize::upscaled_width`].
    trace_w: u32,
    /// `h=` column from the `FRAME_HEADER` trace line —
    /// `frame_height_minus_1 + 1`. Super-resolution is horizontal
    /// only so this equals [`FrameSize::frame_height`].
    trace_h: u32,
    /// `use_superres` column.
    use_superres: bool,
    /// `coded_denom` column. The implied `SuperresDenom` is
    /// `coded_denom + 9` when `use_superres == 1`, otherwise
    /// `SUPERRES_NUM = 8`.
    coded_denom: u8,
    /// `allow_intrabc` column. Round 6 reads the §5.9.3 `f(1)` slot
    /// when `allow_screen_content_tools && UpscaledWidth ==
    /// FrameWidth`; otherwise the §5.9.2 initialiser `allow_intrabc =
    /// 0` stands.
    allow_intrabc: bool,
    /// `tile_cols` column. The §5.9.15 `TileCols` derivation result.
    tile_cols: u32,
    /// `tile_rows` column.
    tile_rows: u32,
    /// `context_update_tile_id` column. `0` for fixtures that have
    /// only a single tile (the spec `f(TileRowsLog2 + TileColsLog2)`
    /// read collapses to 0 bits, and the parser surfaces the
    /// `tile_size_bytes = 1` default for the trailing field).
    context_update_tile_id: u32,
    /// `base_q_idx` column (§5.9.12). Round 7 wired
    /// `quantization_params()` into the streaming walk so this is
    /// asserted against `FrameHeader::quantization_params.base_q_idx`.
    trace_base_q_idx: u8,
    /// `seg_enabled` column (§5.9.14). Round 7 wired
    /// `segmentation_params()` into the streaming walk. The 16-fixture
    /// corpus uniformly reports `seg_enabled=0`, exercising the
    /// 1-bit short-circuit path.
    trace_seg_enabled: bool,
    /// `delta_q_present` column (§5.9.17). Round 8 wired
    /// `delta_q_params()` into the streaming walk. The corpus uniformly
    /// reports `delta_q_present=0`; the `lossless-i-only` fixture has
    /// `base_q_idx=0` so its `delta_q_present` slot is not even read.
    trace_delta_q_present: bool,
    /// `delta_lf_present` column (§5.9.18). Round 8 wired
    /// `delta_lf_params()` into the streaming walk. The corpus uniformly
    /// reports `delta_lf_present=0`; because every fixture has
    /// `delta_q_present=0` the whole `delta_lf_params()` block is a
    /// no-op (no bits read).
    trace_delta_lf_present: bool,
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
            trace_w: 16, trace_h: 16, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 95, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 128, trace_h: 64, use_superres: true, coded_denom: 3,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 160, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 256, trace_h: 128, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 80, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 128, trace_h: 128, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 128, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 256, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 2, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 120, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 34, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 0, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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
            trace_w: 64, trace_h: 64, use_superres: false, coded_denom: 0,
            allow_intrabc: false, tile_cols: 1, tile_rows: 1, context_update_tile_id: 0,
            trace_base_q_idx: 106, trace_seg_enabled: false,
            trace_delta_q_present: false, trace_delta_lf_present: false,
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

        // Round 4: every fixture's first frame is intra so
        // `frame_size` is `Some`. Each trace's `w=` column is the
        // pre-superres width (i.e. `UpscaledWidth`); `h=` is the
        // height as-is. For the `super-resolution` fixture the
        // post-superres `FrameWidth` is `(128*8 + 6) / 12 = 85` and
        // `MiCols = 2 * ((85+7) >> 3) = 22`; every other fixture's
        // `FrameWidth` equals `UpscaledWidth`.
        let fs = fh
            .frame_size
            .unwrap_or_else(|| panic!("fixture {}: expected frame_size = Some(..)", fx.name));
        if fs.upscaled_width != fx.expected.trace_w {
            mismatches.push(format!(
                "{}: upscaled_width expected {} (trace w) got {}",
                fx.name, fx.expected.trace_w, fs.upscaled_width,
            ));
        }
        if fs.frame_height != fx.expected.trace_h {
            mismatches.push(format!(
                "{}: frame_height expected {} (trace h) got {}",
                fx.name, fx.expected.trace_h, fs.frame_height,
            ));
        }
        if fs.use_superres != fx.expected.use_superres {
            mismatches.push(format!(
                "{}: use_superres expected {} got {}",
                fx.name, fx.expected.use_superres, fs.use_superres,
            ));
        }
        if fs.coded_denom != fx.expected.coded_denom {
            mismatches.push(format!(
                "{}: coded_denom expected {} got {}",
                fx.name, fx.expected.coded_denom, fs.coded_denom,
            ));
        }
        let expected_superres_denom = if fx.expected.use_superres {
            u32::from(fx.expected.coded_denom) + 9
        } else {
            SUPERRES_NUM
        };
        if fs.superres_denom != expected_superres_denom {
            mismatches.push(format!(
                "{}: superres_denom expected {} got {}",
                fx.name, expected_superres_denom, fs.superres_denom,
            ));
        }
        let expected_frame_width = (fx.expected.trace_w * SUPERRES_NUM
            + expected_superres_denom / 2)
            / expected_superres_denom;
        if fs.frame_width != expected_frame_width {
            mismatches.push(format!(
                "{}: frame_width (post-superres) expected {} got {}",
                fx.name, expected_frame_width, fs.frame_width,
            ));
        }
        let expected_mi_cols = 2 * ((expected_frame_width + 7) >> 3);
        let expected_mi_rows = 2 * ((fx.expected.trace_h + 7) >> 3);
        if fs.mi_cols != expected_mi_cols {
            mismatches.push(format!(
                "{}: mi_cols expected {} got {}",
                fx.name, expected_mi_cols, fs.mi_cols,
            ));
        }
        if fs.mi_rows != expected_mi_rows {
            mismatches.push(format!(
                "{}: mi_rows expected {} got {}",
                fx.name, expected_mi_rows, fs.mi_rows,
            ));
        }

        // Round 6: §5.9.3 allow_intrabc + §5.9.15 tile_info.
        if fh.allow_intrabc != fx.expected.allow_intrabc {
            mismatches.push(format!(
                "{}: allow_intrabc expected {} got {}",
                fx.name, fx.expected.allow_intrabc, fh.allow_intrabc,
            ));
        }
        let ti = fh
            .tile_info
            .as_ref()
            .unwrap_or_else(|| panic!("fixture {}: expected tile_info = Some(..)", fx.name));
        if ti.tile_cols != fx.expected.tile_cols {
            mismatches.push(format!(
                "{}: tile_cols expected {} got {}",
                fx.name, fx.expected.tile_cols, ti.tile_cols,
            ));
        }
        if ti.tile_rows != fx.expected.tile_rows {
            mismatches.push(format!(
                "{}: tile_rows expected {} got {}",
                fx.name, fx.expected.tile_rows, ti.tile_rows,
            ));
        }
        if ti.context_update_tile_id != fx.expected.context_update_tile_id {
            mismatches.push(format!(
                "{}: context_update_tile_id expected {} got {}",
                fx.name, fx.expected.context_update_tile_id, ti.context_update_tile_id,
            ));
        }
        // §6.8.14 conformance: TileCols <= MAX_TILE_COLS, TileRows <=
        // MAX_TILE_ROWS, context_update_tile_id < TileCols * TileRows.
        // Surface here so a regression in the parser would trip.
        if ti.tile_cols > oxideav_av1::MAX_TILE_COLS {
            mismatches.push(format!(
                "{}: TileCols ({}) exceeded MAX_TILE_COLS ({})",
                fx.name,
                ti.tile_cols,
                oxideav_av1::MAX_TILE_COLS,
            ));
        }
        if ti.tile_rows > oxideav_av1::MAX_TILE_ROWS {
            mismatches.push(format!(
                "{}: TileRows ({}) exceeded MAX_TILE_ROWS ({})",
                fx.name,
                ti.tile_rows,
                oxideav_av1::MAX_TILE_ROWS,
            ));
        }

        // Round 7: §5.9.12 quantization_params + §5.9.14
        // segmentation_params wired into the streaming walk.
        let qp = fh.quantization_params.as_ref().unwrap_or_else(|| {
            panic!(
                "fixture {}: expected quantization_params = Some(..)",
                fx.name
            )
        });
        if qp.base_q_idx != fx.expected.trace_base_q_idx {
            mismatches.push(format!(
                "{}: base_q_idx expected {} got {}",
                fx.name, fx.expected.trace_base_q_idx, qp.base_q_idx,
            ));
        }
        let sp = fh.segmentation_params.as_ref().unwrap_or_else(|| {
            panic!(
                "fixture {}: expected segmentation_params = Some(..)",
                fx.name
            )
        });
        if sp.enabled != fx.expected.trace_seg_enabled {
            mismatches.push(format!(
                "{}: seg_enabled expected {} got {}",
                fx.name, fx.expected.trace_seg_enabled, sp.enabled,
            ));
        }
        // §6.8.13 trailing derivation: when no feature is active,
        // SegIdPreSkip = 0 and LastActiveSegId = 0. The corpus is
        // uniformly seg_enabled=0, so this MUST hold for every entry.
        if sp.seg_id_pre_skip {
            mismatches.push(format!(
                "{}: seg_id_pre_skip expected false (no active features) got true",
                fx.name,
            ));
        }
        if sp.last_active_seg_id != 0 {
            mismatches.push(format!(
                "{}: last_active_seg_id expected 0 (no active features) got {}",
                fx.name, sp.last_active_seg_id,
            ));
        }

        // Round 8: §5.9.17 delta_q_params + §5.9.18 delta_lf_params
        // wired into the streaming walk.
        let dq = fh
            .delta_q_params
            .as_ref()
            .unwrap_or_else(|| panic!("fixture {}: expected delta_q_params = Some(..)", fx.name));
        if dq.delta_q_present != fx.expected.trace_delta_q_present {
            mismatches.push(format!(
                "{}: delta_q_present expected {} got {}",
                fx.name, fx.expected.trace_delta_q_present, dq.delta_q_present,
            ));
        }
        // §5.9.17: delta_q_res stays 0 whenever delta_q_present == 0.
        // The whole corpus is delta_q_present=0, so this MUST hold.
        if !dq.delta_q_present && dq.delta_q_res != 0 {
            mismatches.push(format!(
                "{}: delta_q_res expected 0 (delta_q_present==0) got {}",
                fx.name, dq.delta_q_res,
            ));
        }
        let dlf = fh
            .delta_lf_params
            .as_ref()
            .unwrap_or_else(|| panic!("fixture {}: expected delta_lf_params = Some(..)", fx.name));
        if dlf.delta_lf_present != fx.expected.trace_delta_lf_present {
            mismatches.push(format!(
                "{}: delta_lf_present expected {} got {}",
                fx.name, fx.expected.trace_delta_lf_present, dlf.delta_lf_present,
            ));
        }
        // §5.9.18: delta_lf_res / delta_lf_multi stay 0 / false when
        // delta_lf_present == 0. The corpus is uniformly
        // delta_lf_present=0 (since delta_q_present=0), so this MUST hold.
        if !dlf.delta_lf_present && (dlf.delta_lf_res != 0 || dlf.delta_lf_multi) {
            mismatches.push(format!(
                "{}: delta_lf_res/multi expected 0/false (delta_lf_present==0) got {}/{}",
                fx.name, dlf.delta_lf_res, dlf.delta_lf_multi,
            ));
        }

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
