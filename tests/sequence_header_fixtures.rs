//! Integration tests for §5.5 `sequence_header_obu` parsing.
//!
//! For every IVF fixture under `docs/video/av1/fixtures/`, this test
//! walks the IVF container manually (32-byte file header + 12-byte
//! frame headers), runs the round-1 OBU walker over the AV1 payload,
//! finds the first SEQUENCE_HEADER OBU, parses it with
//! [`oxideav_av1::parse_sequence_header`], and asserts the fields
//! match the expected values captured in each fixture's
//! `trace.txt` (parsed lazily here from the `SEQ_HEADER ...` line the
//! AV1_TRACE-patched ffmpeg+libdav1d harness emitted at corpus
//! generation time).
//!
//! The trace lines themselves are the contract; the cleanroom wall
//! treats them as static fixture metadata (same status as the bytes
//! of `expected.yuv`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use oxideav_av1::{parse_sequence_header, ObuIter, ObuType};

fn fixtures_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at crates/oxideav-av1/
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .join("..")
        .join("..")
        .join("docs")
        .join("video")
        .join("av1")
        .join("fixtures")
}

/// Strip an IVF container and return the concatenated AV1 OBU stream
/// covering all frames in the file. IVF layout:
///
/// * 32-byte file header (`DKIF` magic, codec FourCC, dimensions, …)
/// * For each frame: 12-byte frame header (`u32` size LE + `u64` pts
///   LE) + size bytes of payload.
fn strip_ivf(bytes: &[u8]) -> Vec<u8> {
    assert!(bytes.len() >= 32, "ivf header truncated");
    assert_eq!(&bytes[..4], b"DKIF", "not an IVF file");
    let mut out = Vec::new();
    let mut cursor = 32usize;
    while cursor + 12 <= bytes.len() {
        let frame_size = u32::from_le_bytes([
            bytes[cursor],
            bytes[cursor + 1],
            bytes[cursor + 2],
            bytes[cursor + 3],
        ]) as usize;
        cursor += 12;
        let end = cursor + frame_size;
        assert!(end <= bytes.len(), "ivf frame truncated");
        out.extend_from_slice(&bytes[cursor..end]);
        cursor = end;
    }
    out
}

/// Parse one trace line of the form
/// `SEQ_HEADER\tprofile=0\tstill_picture=0\t...` into a key→value map.
fn parse_trace_kv(line: &str) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for token in line.split('\t').skip(1) {
        if let Some(eq) = token.find('=') {
            map.insert(token[..eq].to_string(), token[eq + 1..].to_string());
        }
    }
    map
}

fn first_seq_header_trace_line(trace: &str) -> Option<BTreeMap<String, String>> {
    for line in trace.lines() {
        if line.starts_with("SEQ_HEADER\t") {
            return Some(parse_trace_kv(line));
        }
    }
    None
}

fn assert_field<T: std::fmt::Debug + PartialEq>(fixture: &str, key: &str, expected: T, actual: T) {
    assert_eq!(
        actual, expected,
        "fixture {fixture}: field {key} mismatch (expected {expected:?}, got {actual:?})"
    );
}

fn run_fixture(dir: &Path) {
    let name = dir.file_name().unwrap().to_string_lossy().into_owned();
    let ivf = fs::read(dir.join("input.ivf")).expect("read input.ivf");
    let trace = fs::read_to_string(dir.join("trace.txt")).expect("read trace.txt");
    let expected = first_seq_header_trace_line(&trace)
        .unwrap_or_else(|| panic!("fixture {name}: no SEQ_HEADER line in trace"));

    let stream = strip_ivf(&ivf);

    // Walk OBUs and find the first SEQUENCE_HEADER.
    let mut seq_payload: Option<Vec<u8>> = None;
    for obu in ObuIter::new(&stream) {
        let obu = obu.expect("OBU walker failure on conformant fixture");
        if obu.obu_type == ObuType::SequenceHeader {
            seq_payload = Some(obu.payload.to_vec());
            break;
        }
    }
    let seq_payload =
        seq_payload.unwrap_or_else(|| panic!("fixture {name}: no SEQUENCE_HEADER OBU"));

    let sh = parse_sequence_header(&seq_payload)
        .unwrap_or_else(|e| panic!("fixture {name}: parse_sequence_header failed: {e:?}"));

    // Per-field comparison against the trace expectations.
    let get = |k: &str| -> &str {
        expected
            .get(k)
            .unwrap_or_else(|| panic!("fixture {name}: trace missing key {k}"))
            .as_str()
    };
    let getu = |k: &str| -> u64 { get(k).parse().expect("uint") };
    let getb = |k: &str| -> bool { getu(k) != 0 };

    assert_field(&name, "profile", getu("profile") as u8, sh.seq_profile);
    assert_field(
        &name,
        "still_picture",
        getb("still_picture"),
        sh.still_picture,
    );
    assert_field(
        &name,
        "reduced_still",
        getb("reduced_still"),
        sh.reduced_still_picture_header,
    );
    // max_w / max_h are encoded as max_frame_width_minus_1+1.
    assert_field(
        &name,
        "max_w",
        getu("max_w") as u32,
        sh.max_frame_width_minus_1 + 1,
    );
    assert_field(
        &name,
        "max_h",
        getu("max_h") as u32,
        sh.max_frame_height_minus_1 + 1,
    );
    assert_field(
        &name,
        "level0",
        getu("level0") as u8,
        sh.operating_points[0].seq_level_idx,
    );
    assert_field(
        &name,
        "tier0",
        getu("tier0") as u8,
        sh.operating_points[0].seq_tier,
    );
    assert_field(
        &name,
        "num_ops",
        getu("num_ops") as u8,
        sh.operating_points_cnt_minus_1 + 1,
    );
    assert_field(
        &name,
        "use_128sb",
        getb("use_128sb"),
        sh.use_128x128_superblock,
    );
    assert_field(
        &name,
        "enable_filter_intra",
        getb("enable_filter_intra"),
        sh.enable_filter_intra,
    );
    assert_field(
        &name,
        "enable_intra_edge_filter",
        getb("enable_intra_edge_filter"),
        sh.enable_intra_edge_filter,
    );
    assert_field(
        &name,
        "enable_interintra",
        getb("enable_interintra"),
        sh.enable_interintra_compound,
    );
    assert_field(
        &name,
        "enable_masked",
        getb("enable_masked"),
        sh.enable_masked_compound,
    );
    assert_field(
        &name,
        "enable_warped",
        getb("enable_warped"),
        sh.enable_warped_motion,
    );
    assert_field(
        &name,
        "enable_dual_filter",
        getb("enable_dual_filter"),
        sh.enable_dual_filter,
    );
    assert_field(
        &name,
        "enable_order_hint",
        getb("enable_order_hint"),
        sh.enable_order_hint,
    );
    assert_field(
        &name,
        "order_hint_bits",
        getu("order_hint_bits") as u8,
        sh.order_hint_bits,
    );
    assert_field(
        &name,
        "enable_jnt_comp",
        getb("enable_jnt_comp"),
        sh.enable_jnt_comp,
    );
    assert_field(
        &name,
        "enable_ref_mvs",
        getb("enable_ref_mvs"),
        sh.enable_ref_frame_mvs,
    );
    assert_field(
        &name,
        "enable_superres",
        getb("enable_superres"),
        sh.enable_superres,
    );
    assert_field(&name, "enable_cdef", getb("enable_cdef"), sh.enable_cdef);
    assert_field(
        &name,
        "enable_restoration",
        getb("enable_restoration"),
        sh.enable_restoration,
    );
    assert_field(
        &name,
        "monochrome",
        getb("monochrome"),
        sh.color_config.mono_chrome,
    );
    assert_field(
        &name,
        "high_bitdepth",
        getb("high_bitdepth"),
        sh.color_config.high_bitdepth,
    );
    assert_field(
        &name,
        "twelve_bit",
        getb("twelve_bit"),
        sh.color_config.twelve_bit,
    );
    assert_field(
        &name,
        "color_range",
        getb("color_range"),
        sh.color_config.color_range,
    );
    assert_field(
        &name,
        "subsampling_x",
        getb("subsampling_x"),
        sh.color_config.subsampling_x,
    );
    assert_field(
        &name,
        "subsampling_y",
        getb("subsampling_y"),
        sh.color_config.subsampling_y,
    );
    assert_field(
        &name,
        "film_grain_present",
        getb("film_grain_present"),
        sh.film_grain_params_present,
    );
}

#[test]
fn all_fixtures_round_trip_sequence_header() {
    let root = fixtures_root();
    let mut fixtures: Vec<PathBuf> = fs::read_dir(&root)
        .expect("read fixtures dir")
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.is_dir() {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    fixtures.sort();
    assert!(
        !fixtures.is_empty(),
        "no fixture directories under {}",
        root.display()
    );
    let mut tested = 0usize;
    for dir in &fixtures {
        // Skip fixture directories that don't have both an IVF and a trace
        // (defensive — the round-1 corpus has both for every entry).
        if !dir.join("input.ivf").is_file() || !dir.join("trace.txt").is_file() {
            continue;
        }
        run_fixture(dir);
        tested += 1;
    }
    assert!(tested >= 10, "expected >= 10 fixtures, got {tested}");
}
