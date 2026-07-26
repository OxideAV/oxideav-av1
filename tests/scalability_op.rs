//! r430 — operating-point selection (§5.5.1 / §6.7.5) and the §5.3.1
//! `drop_obu()` rule, exercised end to end on temporally layered
//! streams.
//!
//! The layered fixtures are built from this crate's own sequential
//! GOP encoder output: every temporal unit is repacked with §5.3.3
//! OBU extension headers carrying a per-frame `temporal_id`, and the
//! sequence header is rewritten with a two-entry operating-point list
//! (`operating_point_idc` masks per §6.7.5 — temporal-layer bits
//! 0..7, spatial-layer bits 8..11). The final GOP frame is placed in
//! temporal layer 1 — it refreshes slots no later frame reads, so the
//! layer-0 subset stays a self-contained prediction chain, exactly
//! the §7.5 scalability discipline.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.3.1 (drop_obu),
//! §5.3.3, §5.5.1 (OperatingPointIdc derivation), §6.7.5
//! (operating_point_idc semantics, choose_operating_point), §7.5
//! (layered-stream ordering constraints).

use oxideav_av1::decoder::{
    decode_av1_spec, decode_av1_spec_at_operating_point, Frame, SpecDecodeSession,
};
use oxideav_av1::encoder::{
    encode_gop_yuv420_with_q, write_obu_with_size, write_sequence_header_obu, IvfWriter,
    ObuExtensionHeader, ObuHeader, Yuv420Frame,
};
use oxideav_av1::obu::{ObuIter, ObuType};
use oxideav_av1::sequence_header::{parse_sequence_header, OperatingPoint};
use oxideav_av1::Error;

/// A little moving-gradient source so successive frames differ.
fn source_frames(n: usize, w: u32, h: u32) -> Vec<Yuv420Frame> {
    (0..n)
        .map(|t| {
            let mut y = vec![0u8; (w * h) as usize];
            for r in 0..h as usize {
                for c in 0..w as usize {
                    y[r * w as usize + c] =
                        ((r as u32 * 3 + c as u32 * 5 + t as u32 * 11) % 224) as u8 + 16;
                }
            }
            let cw = (w / 2) as usize;
            let ch = (h / 2) as usize;
            let u = vec![(96 + 4 * t) as u8; cw * ch];
            let v = vec![(160u8).wrapping_sub(3 * t as u8); cw * ch];
            Yuv420Frame {
                width: w,
                height: h,
                y,
                u,
                v,
            }
        })
        .collect()
}

/// Rewrite `sh_payload` (a sequence_header_obu body) with a two-entry
/// operating-point list: op 0 = temporal layers {0,1} (idc 0x103),
/// op 1 = temporal layer {0} only (idc 0x101). Spatial layer 0 in
/// both (bit 8).
fn seq_payload_with_two_ops(sh_payload: &[u8]) -> Vec<u8> {
    let mut sh = parse_sequence_header(sh_payload).expect("own seq header parses");
    assert_eq!(sh.operating_points.len(), 1, "encoder emits a single op");
    let base = sh.operating_points[0];
    sh.operating_points[0].operating_point_idc = 0x103;
    let op1 = OperatingPoint {
        operating_point_idc: 0x101,
        ..base
    };
    sh.operating_points.push(op1);
    sh.operating_points_cnt_minus_1 = 1;
    write_sequence_header_obu(&sh)
}

/// Repack one temporal unit: frame-carrying OBUs get a §5.3.3
/// extension header with the given `temporal_id`; the sequence-header
/// OBU payload is replaced when `seq_payload` is `Some`; TD OBUs pass
/// through untouched.
fn repack_tu(tu: &[u8], temporal_id: u8, seq_payload: Option<&[u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    for desc in ObuIter::new(tu) {
        let desc = desc.expect("own stream walks");
        match desc.obu_type {
            ObuType::TemporalDelimiter => {
                write_obu_with_size(
                    &mut out,
                    &ObuHeader::new(ObuType::TemporalDelimiter),
                    desc.payload,
                );
            }
            ObuType::SequenceHeader => {
                let body: &[u8] = seq_payload.unwrap_or(desc.payload);
                write_obu_with_size(&mut out, &ObuHeader::new(ObuType::SequenceHeader), body);
            }
            other => {
                // §7.5: every frame header / tile group / frame OBU of
                // a layered stream carries the extension header.
                let header =
                    ObuHeader::new(other).with_extension(ObuExtensionHeader::new(temporal_id, 0));
                write_obu_with_size(&mut out, &header, desc.payload);
            }
        }
    }
    out
}

/// Build the layered fixture: an n-frame sequential GOP whose final
/// frame rides temporal layer 1, everything else layer 0, with the
/// two-op sequence header. Returns (ivf_bytes, n).
fn layered_fixture(n: usize) -> (Vec<u8>, usize) {
    let frames = source_frames(n, 64, 64);
    let gop = encode_gop_yuv420_with_q(&frames, 60).expect("GOP encodes");
    assert_eq!(gop.temporal_units.len(), n);

    // The KEY TU carries the SH; rewrite it with the two-op list.
    let mut tus: Vec<Vec<u8>> = Vec::with_capacity(n);
    for (i, tu) in gop.temporal_units.iter().enumerate() {
        let tid = if i + 1 == n { 1 } else { 0 };
        let seq_override = if i == 0 {
            // Find the original SH payload inside the KEY TU.
            let sh_payload = ObuIter::new(tu)
                .map(|d| d.expect("own stream walks"))
                .find(|d| d.obu_type == ObuType::SequenceHeader)
                .expect("KEY TU carries the SH")
                .payload
                .to_vec();
            Some(seq_payload_with_two_ops(&sh_payload))
        } else {
            None
        };
        tus.push(repack_tu(tu, tid, seq_override.as_deref()));
    }

    let mut ivf = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf);
        let mut iw = IvfWriter::new(cursor, *b"AV01", 64, 64, 25, 1).expect("ivf header");
        for (i, tu) in tus.iter().enumerate() {
            iw.write_frame(tu, i as u64).expect("ivf record");
        }
        iw.patch_frame_count().expect("ivf count patch");
    }
    (ivf, n)
}

#[test]
fn full_operating_point_decodes_every_layer() {
    let (ivf, n) = layered_fixture(4);
    // Default entry (operating point 0 — idc 0x103, both layers).
    let full = decode_av1_spec(&ivf).expect("full decode");
    assert_eq!(full.len(), n, "op 0 must surface every shown frame");
    // Explicit op 0 must be byte-identical.
    let full_explicit = decode_av1_spec_at_operating_point(&ivf, 0).expect("op 0 decode");
    assert_eq!(full, full_explicit);
    // Public API parity.
    let public = oxideav_av1::decode_av1_at_operating_point(&ivf, 0).expect("public op 0");
    assert_eq!(public.len(), n);
    for (pf, sf) in public.iter().zip(full.iter()) {
        let Frame::Spec(s) = pf else {
            panic!("unexpected non-spec frame")
        };
        assert_eq!(s, sf);
    }
}

#[test]
fn reduced_operating_point_yields_the_layer0_frame_subset() {
    let (ivf, n) = layered_fixture(4);
    let full = decode_av1_spec(&ivf).expect("full decode");
    let base = decode_av1_spec_at_operating_point(&ivf, 1).expect("op 1 decode");
    // Operating point 1 excludes temporal layer 1 — exactly the final
    // frame drops; the surviving frames must be byte-identical to the
    // full decode's prefix (the layer-0 chain never references the
    // dropped frame).
    assert_eq!(base.len(), n - 1, "op 1 must drop the layer-1 frame");
    assert_eq!(&full[..n - 1], &base[..]);
}

#[test]
fn out_of_range_operating_point_abandons_decode() {
    let (ivf, _) = layered_fixture(3);
    // The list has two entries — op 2 is out of range (§6.7.5 abandon).
    let err = decode_av1_spec_at_operating_point(&ivf, 2)
        .expect_err("op index beyond operating_points_cnt_minus_1 must reject");
    assert_eq!(err, Error::OperatingPointOutOfRange);

    // On a single-op stream (this crate's default encoder output),
    // op 1 is already out of range.
    let frames = source_frames(2, 64, 64);
    let gop = encode_gop_yuv420_with_q(&frames, 60).expect("GOP encodes");
    let err = decode_av1_spec_at_operating_point(&gop.ivf_bytes, 1)
        .expect_err("single-op stream has no op 1");
    assert_eq!(err, Error::OperatingPointOutOfRange);
}

#[test]
fn session_operating_point_selection_applies_per_temporal_unit() {
    let (ivf, n) = layered_fixture(4);
    let full = decode_av1_spec(&ivf).expect("full decode");

    // Drive the session TU by TU at operating point 1.
    let reader = oxideav_av1::encoder::IvfReader::new(&ivf).expect("ivf parses");
    let records = reader.read_all().expect("records parse");
    let mut session = SpecDecodeSession::new();
    session.set_operating_point(1).expect("no seq cached yet");
    let mut out = Vec::new();
    for r in &records {
        out.extend(
            session
                .decode_temporal_unit(&r.payload)
                .expect("TU decodes"),
        );
    }
    assert_eq!(out.len(), n - 1);
    assert_eq!(&full[..n - 1], &out[..]);
    // §5.5.1: OperatingPointIdc took the op-1 mask.
    assert_eq!(session.operating_point_idc(), 0x101);

    // Re-selecting after the sequence header is cached re-derives
    // immediately — and validates the range.
    session.set_operating_point(0).expect("op 0 exists");
    assert_eq!(session.operating_point_idc(), 0x103);
    assert_eq!(
        session.set_operating_point(5),
        Err(Error::OperatingPointOutOfRange)
    );
}

#[test]
fn nonzero_idc_without_extension_headers_drops_nothing() {
    // §5.3.1: the drop rule fires only on OBUs that carry an
    // extension header. A stream whose selected op has a non-zero idc
    // but whose OBUs have obu_extension_flag == 0 decodes in full
    // (temporal_id / spatial_id are inferred 0 per §6.2.3, but the
    // syntax-table guard tests obu_extension_flag itself).
    let frames = source_frames(3, 64, 64);
    let gop = encode_gop_yuv420_with_q(&frames, 60).expect("GOP encodes");
    let n = gop.temporal_units.len();
    let mut tus: Vec<Vec<u8>> = Vec::new();
    for (i, tu) in gop.temporal_units.iter().enumerate() {
        if i == 0 {
            // Rewrite only the SH (two ops, idc != 0); keep every
            // frame OBU extension-free.
            let mut out = Vec::new();
            for desc in ObuIter::new(tu) {
                let desc = desc.expect("own stream walks");
                let body: Vec<u8> = if desc.obu_type == ObuType::SequenceHeader {
                    seq_payload_with_two_ops(desc.payload)
                } else {
                    desc.payload.to_vec()
                };
                write_obu_with_size(&mut out, &ObuHeader::new(desc.obu_type), &body);
            }
            tus.push(out);
        } else {
            tus.push(tu.clone());
        }
    }
    let mut ivf = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf);
        let mut iw = IvfWriter::new(cursor, *b"AV01", 64, 64, 25, 1).expect("ivf header");
        for (i, tu) in tus.iter().enumerate() {
            iw.write_frame(tu, i as u64).expect("ivf record");
        }
        iw.patch_frame_count().expect("ivf count patch");
    }
    let out = decode_av1_spec_at_operating_point(&ivf, 1).expect("decodes in full");
    assert_eq!(out.len(), n);
}

#[test]
fn operating_point_masks_honour_spatial_bits() {
    // Give the layered fixture's frame OBUs spatial_id = 0 and select
    // an op whose spatial mask EXCLUDES layer 0 (idc with bit 8
    // clear): every extension-carrying OBU drops, so no frame
    // surfaces. This pins the `spatial_id + 8` bit lane of the
    // §5.3.1 rule.
    let frames = source_frames(2, 64, 64);
    let gop = encode_gop_yuv420_with_q(&frames, 60).expect("GOP encodes");
    let mut tus: Vec<Vec<u8>> = Vec::new();
    for (i, tu) in gop.temporal_units.iter().enumerate() {
        let seq_override = if i == 0 {
            let sh_payload = ObuIter::new(tu)
                .map(|d| d.expect("own stream walks"))
                .find(|d| d.obu_type == ObuType::SequenceHeader)
                .expect("KEY TU carries the SH")
                .payload
                .to_vec();
            let mut sh = parse_sequence_header(&sh_payload).expect("seq parses");
            // Single op: temporal layer 0, but spatial mask = layer 1
            // only (bit 9) — a degenerate mask that excludes every
            // spatial_id-0 OBU.
            sh.operating_points[0].operating_point_idc = (1 << 9) | 0x001;
            Some(write_sequence_header_obu(&sh))
        } else {
            None
        };
        tus.push(repack_tu(tu, 0, seq_override.as_deref()));
    }
    let mut ivf = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf);
        let mut iw = IvfWriter::new(cursor, *b"AV01", 64, 64, 25, 1).expect("ivf header");
        for (i, tu) in tus.iter().enumerate() {
            iw.write_frame(tu, i as u64).expect("ivf record");
        }
        iw.patch_frame_count().expect("ivf count patch");
    }
    let out = decode_av1_spec(&ivf).expect("stream walks");
    assert!(
        out.is_empty(),
        "a spatial mask excluding layer 0 must drop every frame OBU"
    );
}

/// Local-only black-box hook: when `AV1_R430_DUMP_DIR` is set, write
/// the layered fixture to disk for external reference-decoder
/// cross-checks (dav1d / aomdec `--oppoint`). Inert in CI (no env).
#[test]
fn dump_layered_fixture_for_blackbox_when_requested() {
    let Some(dir) = std::env::var_os("AV1_R430_DUMP_DIR") else {
        return;
    };
    let (ivf, _) = layered_fixture(4);
    let p = std::path::Path::new(&dir).join("layered4.ivf");
    std::fs::write(&p, &ivf).expect("dump writes");
    // Also dump our own full + op1 decodes as raw planar for diffing.
    let full = decode_av1_spec(&ivf).expect("full decode");
    let base = decode_av1_spec_at_operating_point(&ivf, 1).expect("op1 decode");
    for (name, frames) in [("full.yuv", &full), ("op1.yuv", &base)] {
        let mut buf = Vec::new();
        for f in frames {
            for pl in &f.planes {
                buf.extend_from_slice(pl);
            }
        }
        std::fs::write(std::path::Path::new(&dir).join(name), buf).expect("dump writes");
    }
}
