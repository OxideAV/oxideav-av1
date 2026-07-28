//! r433 — §5.11.1 multi-tile-group FRAMES: a frame whose tiles arrive
//! split across several `OBU_TILE_GROUP` OBUs (each carrying
//! `tile_start_and_end_present_flag = 1` with a `tg_start ..= tg_end`
//! slice), preceded by a standalone `OBU_FRAME_HEADER`.
//!
//! What these tests pin (decode side):
//!
//!   * Accumulation — a 2×2-tiled KEY frame repackaged as
//!     `FH + G tile groups` (every legal grouping of 4 tiles) decodes
//!     to pixels identical to the original single-group `OBU_FRAME`
//!     stream. Per §5.11.1 the frame decodes only once `tg_end ==
//!     NumTiles - 1`.
//!   * §6.10.1 ordering — a `tg_start` that skips ahead, overlaps, or
//!     a temporal unit ending before the last group are rejected
//!     (`Error::TileGroupInvalid`).
//!   * §6.8.1 header discipline — a mid-frame `OBU_FRAME_HEADER` is
//!     rejected; a mid-frame `OBU_REDUNDANT_FRAME_HEADER` carrying
//!     the byte-identical `frame_header_copy` is accepted (and does
//!     not restart the frame); a redundant copy with differing bytes
//!     is rejected; a tile group with no pending header is rejected.
//!   * §6.10.1 `OBU_FRAME` shape — an `OBU_FRAME` whose embedded tile
//!     group codes `tile_start_and_end_present_flag = 1` is rejected
//!     even when the range covers the whole frame.
//!
//! The repacker below is a TEST harness: it re-frames this crate's
//! own single-group output without touching any entropy bytes, so a
//! byte-level mistake in the re-framing (not the decoder) would decode
//! to garbage and fail the pixel comparison loudly.
//!
//! Spec: docs/video/av1/av1-spec.txt §5.9.1, §5.10, §5.11.1, §6.8.1,
//! §6.10.1, §7.5.

use oxideav_av1::decoder::Frame;
use oxideav_av1::encoder::ivf::{build_file_header, build_frame_header, IvfReader};
use oxideav_av1::encoder::obu::{write_obu_with_size, ObuHeader};
use oxideav_av1::encoder::tile_group_obu::{
    parse_tile_group_obu_body, write_tile_group_obu, TileGroupObu, TilePayload,
};
use oxideav_av1::encoder::{encode_key_frame_yuv420_with_q_tiles, Yuv420Frame};
use oxideav_av1::frame_header::parse_frame_header;
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::parse_sequence_header;
use oxideav_av1::Error;

// ---------------------------------------------------------------------
// Content.
// ---------------------------------------------------------------------

/// Deterministic LCG noise — maximal coded symbols per tile, so any
/// mis-sliced tile payload desyncs the §8.2 coder immediately.
fn noise(w: u32, h: u32, seed: u32) -> Yuv420Frame {
    let mut state = seed | 1;
    let mut next = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    };
    let mut f = Yuv420Frame::filled(w, h, 0);
    for v in f.y.iter_mut().chain(f.u.iter_mut()).chain(f.v.iter_mut()) {
        *v = next();
    }
    f
}

// ---------------------------------------------------------------------
// The repacker.
// ---------------------------------------------------------------------

/// The pieces of one `OBU_FRAME`, split at the §5.10 boundary.
struct SplitFrame {
    /// Standalone `OBU_FRAME_HEADER` payload — the §5.9 header bits
    /// re-terminated with §5.3.4 `trailing_bits` (the `OBU_FRAME`
    /// packing ends them with `byte_alignment()` zeros instead).
    fh_payload: Vec<u8>,
    /// Per-tile §8.2 entropy payloads, tile raster order.
    tiles: Vec<TilePayload>,
    /// (NumTiles, TileColsLog2, TileRowsLog2) for `tg_*` re-framing.
    dims: (u32, u32, u32),
}

/// Split an `OBU_FRAME` payload into a standalone header payload and
/// the per-tile payloads. KEY-frame-only (the header parse needs no
/// reference state).
fn split_frame_obu(
    payload: &[u8],
    seq: &oxideav_av1::sequence_header::SequenceHeader,
) -> SplitFrame {
    let fh = parse_frame_header(payload, seq).expect("own KEY header parses");
    let ti = fh.tile_info.clone().expect("KEY header carries tile_info");
    let bits = fh.bits_consumed;
    let header_bytes = bits.div_ceil(8);

    // §5.3.4 trailing_bits for the standalone OBU_FRAME_HEADER: a one
    // bit at position `bits`, zeros to the byte boundary. Inside the
    // OBU_FRAME the same positions carried §5.10 byte_alignment()
    // zeros, so only the one bit differs (or a whole 0x80 byte when
    // the header already ended byte-aligned).
    let mut fh_payload = payload[..header_bytes].to_vec();
    if bits % 8 == 0 {
        fh_payload.push(0x80);
    } else {
        fh_payload[bits / 8] |= 0x80u8 >> (bits % 8);
    }

    let num_tiles = ti.tile_cols * ti.tile_rows;
    let parsed = parse_tile_group_obu_body(
        &payload[header_bytes..],
        num_tiles,
        ti.tile_cols_log2,
        ti.tile_rows_log2,
        u32::from(ti.tile_size_bytes),
    )
    .expect("own tile-group body parses");
    assert_eq!(parsed.tg_start, 0);
    assert_eq!(parsed.tg_end + 1, num_tiles);
    SplitFrame {
        fh_payload,
        tiles: parsed.tiles,
        dims: (num_tiles, ti.tile_cols_log2, ti.tile_rows_log2),
    }
}

/// Build one `OBU_TILE_GROUP` for tiles `start ..= end` (inclusive,
/// frame-scope indices) with `tile_start_and_end_present_flag = 1`.
fn tile_group_obu_for(split: &SplitFrame, start: u32, end: u32) -> Vec<u8> {
    let (num_tiles, cl2, rl2) = split.dims;
    let payloads: Vec<TilePayload> = split.tiles[start as usize..=end as usize].to_vec();
    let tsb = payloads
        .iter()
        .take(payloads.len().saturating_sub(1))
        .map(|p| {
            let mut n = 1u32;
            while (p.bytes.len() as u64 - 1) >= (1u64 << (8 * n)) {
                n += 1;
            }
            n
        })
        .max()
        .unwrap_or(1);
    let body = write_tile_group_obu(&TileGroupObu {
        num_tiles,
        tile_cols_log2: cl2,
        tile_rows_log2: rl2,
        tile_size_bytes: tsb,
        tg_start: start,
        tg_end: end,
        start_and_end_present: true,
        tiles: payloads,
    })
    .expect("re-framed tile group writes");
    let mut out = Vec::new();
    write_obu_with_size(&mut out, &ObuHeader::new(ObuType::TileGroup), &body);
    out
}

/// Extra OBUs to interleave between the re-framed tile groups, keyed
/// by the number of tile groups already emitted.
enum MidFrameObu {
    None,
    /// Insert an `OBU_REDUNDANT_FRAME_HEADER` (§6.8.1
    /// `frame_header_copy`) after the given group, with an optional
    /// byte corruption.
    Redundant {
        after_group: usize,
        corrupt: bool,
    },
    /// Insert a second `OBU_FRAME_HEADER` (identical bytes — still
    /// non-conformant mid-frame per §6.8.1) after the given group.
    FrameHeader {
        after_group: usize,
    },
}

/// Repackage a single-frame IVF stream (TD + SH + OBU_FRAME temporal
/// unit) into `TD + SH + OBU_FRAME_HEADER + tile groups`, splitting
/// the tiles at `groups` (frame-scope inclusive ranges).
fn repack_split(ivf: &[u8], groups: &[(u32, u32)], mid: MidFrameObu) -> Vec<u8> {
    let mut reader = IvfReader::new(ivf).expect("own IVF parses");
    let header = *reader.header();
    let frame = reader
        .read_next_frame()
        .expect("own IVF frame reads")
        .expect("one frame present");

    let mut seq = None;
    let mut tu = Vec::new();
    for desc in ObuIter::new(&frame.payload) {
        let desc = desc.expect("own TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("own SH parses"));
                write_obu_with_size(
                    &mut tu,
                    &ObuHeader::new(ObuType::SequenceHeader),
                    desc.payload,
                );
            }
            ObuType::TemporalDelimiter => {
                write_obu_with_size(&mut tu, &ObuHeader::new(ObuType::TemporalDelimiter), &[]);
            }
            ObuType::Frame => {
                let split = split_frame_obu(desc.payload, seq.as_ref().expect("SH first"));
                write_obu_with_size(
                    &mut tu,
                    &ObuHeader::new(ObuType::FrameHeader),
                    &split.fh_payload,
                );
                for (gi, &(start, end)) in groups.iter().enumerate() {
                    tu.extend_from_slice(&tile_group_obu_for(&split, start, end));
                    match mid {
                        MidFrameObu::Redundant {
                            after_group,
                            corrupt,
                        } if after_group == gi => {
                            let mut copy = split.fh_payload.clone();
                            if corrupt {
                                // Flip a §5.9.2 header bit — the copy
                                // is no longer byte-identical.
                                let last = copy.len() - 1;
                                copy[last] ^= 0x02;
                            }
                            write_obu_with_size(
                                &mut tu,
                                &ObuHeader::new(ObuType::RedundantFrameHeader),
                                &copy,
                            );
                        }
                        MidFrameObu::FrameHeader { after_group } if after_group == gi => {
                            write_obu_with_size(
                                &mut tu,
                                &ObuHeader::new(ObuType::FrameHeader),
                                &split.fh_payload,
                            );
                        }
                        _ => {}
                    }
                }
            }
            other => panic!("unexpected OBU in single-frame TU: {other:?}"),
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(&build_file_header(
        header.fourcc,
        header.width,
        header.height,
        header.fps_num,
        header.fps_den,
        1,
    ));
    out.extend_from_slice(&build_frame_header(tu.len() as u32, frame.pts));
    out.extend_from_slice(&tu);
    out
}

/// Repackage keeping the `OBU_FRAME` shape but rewriting its embedded
/// tile group with `tile_start_and_end_present_flag = 1` over the full
/// range — non-conformant per §6.10.1.
fn repack_frame_obu_flag_one(ivf: &[u8]) -> Vec<u8> {
    let mut reader = IvfReader::new(ivf).expect("own IVF parses");
    let header = *reader.header();
    let frame = reader
        .read_next_frame()
        .expect("own IVF frame reads")
        .expect("one frame present");

    let mut seq = None;
    let mut tu = Vec::new();
    for desc in ObuIter::new(&frame.payload) {
        let desc = desc.expect("own TU walks");
        match desc.obu_type {
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload).expect("own SH parses"));
                write_obu_with_size(
                    &mut tu,
                    &ObuHeader::new(ObuType::SequenceHeader),
                    desc.payload,
                );
            }
            ObuType::TemporalDelimiter => {
                write_obu_with_size(&mut tu, &ObuHeader::new(ObuType::TemporalDelimiter), &[]);
            }
            ObuType::Frame => {
                let fh = parse_frame_header(desc.payload, seq.as_ref().expect("SH first"))
                    .expect("own KEY header parses");
                let header_bytes = fh.bits_consumed.div_ceil(8);
                let split = split_frame_obu(desc.payload, seq.as_ref().expect("SH first"));
                let (num_tiles, ..) = split.dims;
                // Whole-frame range, but with the flag SET.
                let tg = tile_group_obu_for(&split, 0, num_tiles - 1);
                // Strip the OBU wrapper we just added: re-derive the
                // body directly instead.
                let (desc2, _) = oxideav_av1::obu::parse_obu(&tg).expect("own TG parses");
                let mut body = desc.payload[..header_bytes].to_vec();
                body.extend_from_slice(desc2.payload);
                write_obu_with_size(&mut tu, &ObuHeader::new(ObuType::Frame), &body);
            }
            other => panic!("unexpected OBU in single-frame TU: {other:?}"),
        }
    }

    let mut out = Vec::new();
    out.extend_from_slice(&build_file_header(
        header.fourcc,
        header.width,
        header.height,
        header.fps_num,
        header.fps_den,
        1,
    ));
    out.extend_from_slice(&build_frame_header(tu.len() as u32, frame.pts));
    out.extend_from_slice(&tu);
    out
}

fn spec_planes(ivf: &[u8], label: &str) -> Vec<Vec<u8>> {
    let frames = oxideav_av1::decode_av1(ivf)
        .unwrap_or_else(|e| panic!("{label}: decode_av1 rejected stream: {e:?}"));
    assert_eq!(frames.len(), 1, "{label}: shown frame count");
    match frames.into_iter().next().expect("one frame") {
        Frame::Spec(s) => s.planes,
        #[allow(unreachable_patterns)]
        other => panic!("{label}: non-Spec frame variant {other:?}"),
    }
}

fn tiled_key_ivf() -> Vec<u8> {
    let frame = noise(192, 128, 41);
    encode_key_frame_yuv420_with_q_tiles(&frame, 72, 1, 1)
        .expect("2x2 tiled KEY encode")
        .ivf_bytes
}

// ---------------------------------------------------------------------
// Accumulation round trips.
// ---------------------------------------------------------------------

/// Every legal grouping of a 4-tile frame decodes byte-identical to
/// the original single-group stream — one group per tile, pairs, the
/// 3+1 / 1+3 splits, and the single whole-frame group re-framed with
/// `tile_start_and_end_present_flag = 1`.
#[test]
fn split_tile_groups_decode_identical_to_single_group() {
    let ivf = tiled_key_ivf();
    let baseline = spec_planes(&ivf, "single-group baseline");
    let groupings: &[&[(u32, u32)]] = &[
        &[(0, 0), (1, 1), (2, 2), (3, 3)],
        &[(0, 1), (2, 3)],
        &[(0, 2), (3, 3)],
        &[(0, 0), (1, 3)],
        &[(0, 3)],
    ];
    for groups in groupings {
        let repacked = repack_split(&ivf, groups, MidFrameObu::None);
        let planes = spec_planes(&repacked, &format!("groups {groups:?}"));
        assert_eq!(
            planes, baseline,
            "split {groups:?} must decode identical to the single-group stream"
        );
    }
}

/// §6.8.1 `frame_header_copy`: a byte-identical
/// `OBU_REDUNDANT_FRAME_HEADER` between two tile groups is accepted
/// and changes nothing.
#[test]
fn redundant_frame_header_copy_mid_frame_is_accepted() {
    let ivf = tiled_key_ivf();
    let baseline = spec_planes(&ivf, "single-group baseline");
    let repacked = repack_split(
        &ivf,
        &[(0, 1), (2, 3)],
        MidFrameObu::Redundant {
            after_group: 0,
            corrupt: false,
        },
    );
    let planes = spec_planes(&repacked, "redundant copy mid-frame");
    assert_eq!(planes, baseline);
}

// ---------------------------------------------------------------------
// Conformance rejects.
// ---------------------------------------------------------------------

fn assert_rejected(ivf: Vec<u8>, label: &str) {
    match oxideav_av1::decode_av1(&ivf) {
        Err(Error::TileGroupInvalid) => {}
        Err(other) => panic!("{label}: expected TileGroupInvalid, got {other:?}"),
        Ok(_) => panic!("{label}: non-conformant stream decoded"),
    }
}

/// A gap in the tile coverage (`tg_start` skips a tile) violates the
/// §6.10.1 running-`TileNum` rule.
#[test]
fn tile_group_gap_is_rejected() {
    let ivf = tiled_key_ivf();
    assert_rejected(
        repack_split(&ivf, &[(0, 0), (2, 3)], MidFrameObu::None),
        "gap",
    );
}

/// An overlapping `tg_start` (re-sending a tile) is likewise
/// rejected.
#[test]
fn tile_group_overlap_is_rejected() {
    let ivf = tiled_key_ivf();
    assert_rejected(
        repack_split(&ivf, &[(0, 1), (1, 3)], MidFrameObu::None),
        "overlap",
    );
}

/// A temporal unit that ends before `tg_end == NumTiles - 1` leaves
/// the frame undecodable (§7.5 — a frame's OBUs live in one unit).
#[test]
fn incomplete_frame_at_temporal_unit_end_is_rejected() {
    let ivf = tiled_key_ivf();
    assert_rejected(
        repack_split(&ivf, &[(0, 2)], MidFrameObu::None),
        "incomplete",
    );
}

/// A second `OBU_FRAME_HEADER` mid-frame is non-conformant even with
/// identical bytes (§6.8.1 — copies must use the REDUNDANT type).
#[test]
fn mid_frame_frame_header_is_rejected() {
    let ivf = tiled_key_ivf();
    assert_rejected(
        repack_split(
            &ivf,
            &[(0, 1), (2, 3)],
            MidFrameObu::FrameHeader { after_group: 0 },
        ),
        "mid-frame FH",
    );
}

/// An `OBU_REDUNDANT_FRAME_HEADER` whose bytes differ from the
/// original violates the §6.8.1 `frame_header_copy` identity.
#[test]
fn corrupted_redundant_frame_header_is_rejected() {
    let ivf = tiled_key_ivf();
    assert_rejected(
        repack_split(
            &ivf,
            &[(0, 1), (2, 3)],
            MidFrameObu::Redundant {
                after_group: 0,
                corrupt: true,
            },
        ),
        "corrupted redundant FH",
    );
}

/// An `OBU_FRAME` whose embedded tile group sets
/// `tile_start_and_end_present_flag = 1` is rejected per §6.10.1,
/// even though the coded range covers the whole frame.
#[test]
fn frame_obu_with_tg_flag_one_is_rejected() {
    let ivf = tiled_key_ivf();
    assert_rejected(repack_frame_obu_flag_one(&ivf), "OBU_FRAME flag=1");
}

// ---------------------------------------------------------------------
// Encoder-native split emission (r433 write arm).
// ---------------------------------------------------------------------

/// Walk one temporal unit and return the OBU type sequence plus, for
/// every `OBU_TILE_GROUP`, its parsed `(tg_start, tg_end, flag)`.
/// `seq` carries the sequence header across temporal units (P-frame
/// units don't repeat it).
fn audit_tu(
    tu: &[u8],
    seq: &mut Option<oxideav_av1::sequence_header::SequenceHeader>,
) -> (Vec<ObuType>, Vec<(u32, u32, bool)>) {
    let mut types = Vec::new();
    let mut groups = Vec::new();
    let mut ti: Option<oxideav_av1::tile_info::TileInfo> = None;
    for desc in ObuIter::new(tu) {
        let desc = desc.expect("own TU walks");
        types.push(desc.obu_type);
        match desc.obu_type {
            ObuType::SequenceHeader => {
                *seq = Some(parse_sequence_header(desc.payload).expect("own SH parses"));
            }
            ObuType::FrameHeader => {
                let fh = parse_frame_header(desc.payload, seq.as_ref().expect("SH first"))
                    .expect("own FH parses");
                ti = fh.tile_info.clone();
            }
            ObuType::TileGroup => {
                let ti = ti.as_ref().expect("FH precedes tile groups");
                let parsed = parse_tile_group_obu_body(
                    desc.payload,
                    ti.tile_cols * ti.tile_rows,
                    ti.tile_cols_log2,
                    ti.tile_rows_log2,
                    u32::from(ti.tile_size_bytes),
                )
                .expect("own TG parses");
                groups.push((
                    parsed.tg_start,
                    parsed.tg_end,
                    parsed.tile_start_and_end_present_flag,
                ));
            }
            _ => {}
        }
    }
    (types, groups)
}

/// `tile_groups = 1` reproduces the `_tiles` entry byte for byte —
/// the packaging knob at its default is a no-op.
#[test]
fn key_tile_groups_1_is_byte_identical_to_tiles_entry() {
    use oxideav_av1::encoder::encode_key_frame_yuv420_with_q_tile_groups;
    let frame = noise(192, 128, 41);
    let a = encode_key_frame_yuv420_with_q_tiles(&frame, 72, 1, 1).expect("tiles entry");
    let b = encode_key_frame_yuv420_with_q_tile_groups(&frame, 72, 1, 1, 1).expect("groups=1");
    assert_eq!(a.ivf_bytes, b.ivf_bytes);
}

/// Native split emission: a 2×2-tiled KEY at `tile_groups` 2..=4 (and
/// an over-ask 9 that clamps to 4) decodes pixel-exact to the
/// encoder reconstruction, carries the `FH + G tile groups` OBU
/// shape with contiguous `tg` ranges, and its per-tile entropy bytes
/// are BYTE-IDENTICAL to the single-group stream's.
#[test]
fn key_native_split_groups_decode_pixel_exact() {
    use oxideav_av1::encoder::encode_key_frame_yuv420_with_q_tile_groups;
    let frame = noise(192, 128, 41);
    let single = encode_key_frame_yuv420_with_q_tiles(&frame, 72, 1, 1).expect("single-group");
    let single_planes = spec_planes(&single.ivf_bytes, "single");
    // Reference per-tile bytes from the single-group stream.
    let single_tiles = {
        let mut reader = IvfReader::new(&single.ivf_bytes).expect("own IVF parses");
        let f = reader.read_next_frame().expect("reads").expect("one frame");
        let mut seq = None;
        let mut tiles = None;
        for desc in ObuIter::new(&f.payload) {
            let desc = desc.expect("walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seq = Some(parse_sequence_header(desc.payload).expect("SH"));
                }
                ObuType::Frame => {
                    tiles = Some(split_frame_obu(desc.payload, seq.as_ref().expect("SH")).tiles);
                }
                _ => {}
            }
        }
        tiles.expect("OBU_FRAME present")
    };
    for (groups, want_groups) in [(2u32, 2usize), (3, 3), (4, 4), (9, 4)] {
        let enc = encode_key_frame_yuv420_with_q_tile_groups(&frame, 72, 1, 1, groups)
            .unwrap_or_else(|e| panic!("groups={groups}: encode failed: {e:?}"));
        let planes = spec_planes(&enc.ivf_bytes, &format!("groups={groups}"));
        assert_eq!(planes, single_planes, "groups={groups}: pixels");
        assert_eq!(planes[0], enc.recon_y, "groups={groups}: recon luma");
        let (types, tgs) = audit_tu(&enc.temporal_unit_bytes, &mut None);
        assert_eq!(
            types
                .iter()
                .filter(|t| matches!(t, ObuType::TileGroup))
                .count(),
            want_groups,
            "groups={groups}: tile-group OBU count"
        );
        assert!(
            types.contains(&ObuType::FrameHeader) && !types.contains(&ObuType::Frame),
            "groups={groups}: split packaging must use OBU_FRAME_HEADER"
        );
        // Contiguous coverage 0..=3 with the flag set on every group.
        let mut next = 0u32;
        for &(s, e, flag) in &tgs {
            assert!(flag, "groups={groups}: tg flag");
            assert_eq!(s, next, "groups={groups}: tg_start contiguity");
            next = e + 1;
        }
        assert_eq!(next, 4, "groups={groups}: full coverage");
        // Entropy bytes identical to the single-group stream.
        let mut reader = IvfReader::new(&enc.ivf_bytes).expect("own IVF parses");
        let f = reader.read_next_frame().expect("reads").expect("one frame");
        let mut seqh = None;
        let mut ti = None;
        let mut collected: Vec<TilePayload> = Vec::new();
        for desc in ObuIter::new(&f.payload) {
            let desc = desc.expect("walks");
            match desc.obu_type {
                ObuType::SequenceHeader => {
                    seqh = Some(parse_sequence_header(desc.payload).expect("SH"));
                }
                ObuType::FrameHeader => {
                    let fh = parse_frame_header(desc.payload, seqh.as_ref().expect("SH first"))
                        .expect("FH");
                    ti = fh.tile_info.clone();
                }
                ObuType::TileGroup => {
                    let t = ti.as_ref().expect("FH first");
                    let parsed = parse_tile_group_obu_body(
                        desc.payload,
                        t.tile_cols * t.tile_rows,
                        t.tile_cols_log2,
                        t.tile_rows_log2,
                        u32::from(t.tile_size_bytes),
                    )
                    .expect("TG parses");
                    collected.extend(parsed.tiles);
                }
                _ => {}
            }
        }
        assert_eq!(
            collected, single_tiles,
            "groups={groups}: per-tile entropy bytes must be identical"
        );
    }
}

/// GOP-wide native split: every frame of a 2×2-tiled 3-frame GOP
/// (full election set armed) rides `FH + 2 tile groups`, and every
/// frame decodes pixel-exact to the encoder reconstruction.
#[test]
fn gop_native_split_groups_decode_pixel_exact() {
    use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning};
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| noise(192, 128, 100 + t)).collect();
    let tuning = GopTuning {
        tiles: (1, 1),
        tile_groups: 2,
        ..GopTuning::default()
    };
    let enc = encode_gop_yuv420_with_q_seg_tuned(&frames, 72, &[], tuning).expect("split GOP");
    let mut seq_carry = None;
    for (i, tu) in enc.gop.temporal_units.iter().enumerate() {
        let (types, tgs) = audit_tu(tu, &mut seq_carry);
        assert_eq!(
            types
                .iter()
                .filter(|t| matches!(t, ObuType::TileGroup))
                .count(),
            2,
            "frame {i}: tile-group OBU count"
        );
        assert!(
            !types.contains(&ObuType::Frame),
            "frame {i}: split packaging must not use OBU_FRAME"
        );
        assert_eq!(tgs[0].0, 0, "frame {i}: first tg_start");
        assert_eq!(tgs.last().unwrap().1, 3, "frame {i}: last tg_end");
    }
    let decoded = {
        let frames = oxideav_av1::decode_av1(&enc.gop.ivf_bytes).expect("split GOP decodes");
        assert_eq!(frames.len(), 3);
        frames
            .into_iter()
            .map(|f| match f {
                Frame::Spec(s) => s,
                #[allow(unreachable_patterns)]
                other => panic!("non-Spec frame {other:?}"),
            })
            .collect::<Vec<_>>()
    };
    for (i, f) in decoded.iter().enumerate() {
        assert_eq!(f.planes[0], enc.gop.recon[i].y, "frame {i} luma");
        assert_eq!(f.planes[1], enc.gop.recon[i].u, "frame {i} U");
        assert_eq!(f.planes[2], enc.gop.recon[i].v, "frame {i} V");
    }
}

/// GOP `tile_groups = 1` byte identity — the knob's default is a
/// no-op on the tiled GOP stream.
#[test]
fn gop_tile_groups_1_is_byte_identical() {
    use oxideav_av1::encoder::{encode_gop_yuv420_with_q_seg_tuned, GopTuning};
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| noise(192, 128, 200 + t)).collect();
    let a = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        72,
        &[],
        GopTuning {
            tiles: (1, 1),
            ..GopTuning::default()
        },
    )
    .expect("tiles GOP");
    let b = encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        72,
        &[],
        GopTuning {
            tiles: (1, 1),
            tile_groups: 1,
            ..GopTuning::default()
        },
    )
    .expect("groups=1 GOP");
    assert_eq!(a.gop.ivf_bytes, b.gop.ivf_bytes);
}

// ---------------------------------------------------------------------
// Black-box revalidation hook.
// ---------------------------------------------------------------------

/// Env-gated dump for external validation: with `AV1_MTG_DUMP_DIR`
/// set, write the single-group baseline plus every legal repack to
/// that directory so independent reference decoders (run as black-box
/// binaries) can confirm the split-tile-group wire shape decodes
/// identically to the original stream. Inert (and green) otherwise.
#[test]
fn dump_repacked_streams_for_blackbox_validation() {
    let Ok(dir) = std::env::var("AV1_MTG_DUMP_DIR") else {
        return;
    };
    let dir = std::path::Path::new(&dir);
    std::fs::create_dir_all(dir).expect("dump dir");
    let ivf = tiled_key_ivf();
    std::fs::write(dir.join("mtg-baseline.ivf"), &ivf).expect("write baseline");
    let groupings: &[(&str, &[(u32, u32)])] = &[
        ("mtg-split-1each.ivf", &[(0, 0), (1, 1), (2, 2), (3, 3)]),
        ("mtg-split-2x2.ivf", &[(0, 1), (2, 3)]),
        ("mtg-split-3plus1.ivf", &[(0, 2), (3, 3)]),
        ("mtg-split-1plus3.ivf", &[(0, 0), (1, 3)]),
        ("mtg-split-whole-flag1.ivf", &[(0, 3)]),
    ];
    for (name, groups) in groupings {
        let repacked = repack_split(&ivf, groups, MidFrameObu::None);
        std::fs::write(dir.join(name), repacked).expect("write repack");
    }
    let redundant = repack_split(
        &ivf,
        &[(0, 1), (2, 3)],
        MidFrameObu::Redundant {
            after_group: 0,
            corrupt: false,
        },
    );
    std::fs::write(dir.join("mtg-split-redundant-fh.ivf"), redundant).expect("write redundant");
    // Encoder-NATIVE split streams (the r433 write arm).
    let frame = noise(192, 128, 41);
    let native_key =
        oxideav_av1::encoder::encode_key_frame_yuv420_with_q_tile_groups(&frame, 72, 1, 1, 4)
            .expect("native KEY g=4");
    std::fs::write(dir.join("mtg-native-key-g4.ivf"), &native_key.ivf_bytes).expect("write");
    let frames: Vec<Yuv420Frame> = (0..3).map(|t| noise(192, 128, 100 + t)).collect();
    let native_gop = oxideav_av1::encoder::encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        72,
        &[],
        oxideav_av1::encoder::GopTuning {
            tiles: (1, 1),
            tile_groups: 2,
            ..oxideav_av1::encoder::GopTuning::default()
        },
    )
    .expect("native GOP g=2");
    std::fs::write(dir.join("mtg-native-gop-g2.ivf"), &native_gop.gop.ivf_bytes).expect("write");
    // Single-group siblings of the native streams — the black-box
    // digests must match pairwise.
    let single_gop = oxideav_av1::encoder::encode_gop_yuv420_with_q_seg_tuned(
        &frames,
        72,
        &[],
        oxideav_av1::encoder::GopTuning {
            tiles: (1, 1),
            ..oxideav_av1::encoder::GopTuning::default()
        },
    )
    .expect("single GOP");
    std::fs::write(dir.join("mtg-single-gop.ivf"), &single_gop.gop.ivf_bytes).expect("write");
}
