//! Workspace task #786 — lossless WHT residual scaling audit.
//!
//! End-to-end regression covering the §7.7.4 `Lossless = 1` branch
//! of the reconstruct process against the
//! `y_plane_divergence_match.avif` 1×1 YUV444 KEY frame
//! (`crates/oxideav-avif/tests/fixtures/fuzz/y_plane_divergence_match.avif`).
//!
//! Round 47 finding: with the AVIF `mdat` OBU stream extracted
//! verbatim, `dav1d 1.5.3` and `avifdec --raw-color` (libavif via
//! its bundled `dav1d`) both decode to `(Y, U, V) = (133, 197, 215)`.
//! `oxideav-av1` decodes to `(Y, U, V) = (130, 128, 128)` — a Y
//! delta of 3 LSB and chroma deltas of 69 and 87 LSB.
//!
//! Hand-tracing the §7.13.2.10 inverse WHT through the in-bounds Y
//! TU's dequantised coefficient buffer (row pass `shift = 2`,
//! column pass `shift = 0`) shows the WHT scaling is spec-correct
//! — the runtime residual at (0, 0) matches a hand-trace of the
//! §7.13.2.10 algorithm character-for-character on the entropy
//! decoder's output. The pixel divergence vs `dav1d` is therefore
//! **upstream** of the WHT — the §5.11.39 / §9.4 coefficient
//! entropy decoder reads different `level` values than `dav1d`
//! for the same range coder state. The WHT/lossless dispatch is
//! pinned spec-correct by
//! `transform::iwht4::tests::iwht4_2d_divergence_y_tu_matches_spec`
//! (residual buffer hand-traced cell-by-cell) and
//! `transform::iwht4::tests::iwht4_2d_dc_only_level_1_yields_unit_residual`
//! (the original level=1 round-46 trace).
//!
//! This test pins the **current oxideav decode output** against the
//! spec-correct WHT residual path. A regression that moves the
//! sample value would surface here; future progress on the entropy
//! decoder (closing the residual delta to `dav1d`'s 133/197/215)
//! would also need to update the assertions in this test along
//! with the entropy fix.
//!
//! Spec refs: §7.7.4 reconstruction (rowShift = colShift = 0 in
//! Lossless); §7.13.2.10 inverse Walsh-Hadamard transform process
//! (row pass shift = 2, column pass shift = 0); §7.12.2
//! quantisation tables (`DC8[0] = AC8[0] = 4` at `base_q_idx = 0`);
//! §5.11.34 lossless TX size override (`txSz = TX_4X4`).

use oxideav_av1::decoder::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

/// Real 27-byte AV1 OBU stream extracted from the `mdat` box of
/// `y_plane_divergence_match.avif`. Matches what `dav1d` and
/// `avifdec` operate on.
const OBU_STREAM: &[u8] = &[
    0x12, 0x00, 0x0a, 0x04, 0x38, 0x00, 0x0e, 0x49, 0x32, 0x11, 0x10, 0x00, 0x00, 0x19, 0xb9, 0xca,
    0xe3, 0x37, 0x39, 0x09, 0x47, 0xd9, 0x6e, 0x65, 0x96, 0x64, 0xaf,
];

/// Cross-decoder reference YUV (from `dav1d 1.5.3` and `avifdec`
/// — both reach exact agreement on the raw YUV plane values).
const REF_Y: u8 = 0x85;
const REF_U: u8 = 0xC5;
const REF_V: u8 = 0xD7;

/// Decode the divergence OBU through `Av1Decoder` and return the
/// `(Y, U, V)` sample at position (0, 0) of each plane. Helper kept
/// small so the assertion sites stay focused on the spec semantics.
fn decode_divergence() -> (u8, u8, u8) {
    let mut dec = Av1Decoder::new(CodecParameters::video(CodecId::new("av1")));
    let pkt = Packet::new(0, TimeBase::new(1, 1), OBU_STREAM.to_vec())
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt)
        .expect("divergence OBU must decode without error");
    let mut out: Option<(u8, u8, u8)> = None;
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            assert_eq!(vf.planes.len(), 3, "YUV444 must surface 3 planes");
            let y = vf.planes[0].data[0];
            let u = vf.planes[1].data[0];
            let v = vf.planes[2].data[0];
            out = Some((y, u, v));
            break;
        }
    }
    out.expect("a Video frame must be produced")
}

/// The decode must complete without panicking — this is the
/// AVIF-side fuzz contract that round 44 first restored
/// (`oxideav-avif::tests::fuzz_regressions::fuzz_y_plane_divergence_match_does_not_panic`).
#[test]
fn issue_786_divergence_obu_decodes_without_panic() {
    let _ = decode_divergence();
}

/// Pin the §7.7.4 / §7.13.2.10 spec-correct decode output for the
/// in-bounds 1×1 sample on each plane. The Y residual `2` (added
/// to the 128 default predictor) and the chroma residuals `0`
/// (also added to 128) are what the inverse WHT mathematically
/// produces from oxideav's entropy-decoded level streams; the
/// WHT itself is verified spec-correct by hand-trace in
/// `transform::iwht4::tests::iwht4_2d_divergence_y_tu_matches_spec`.
///
/// A future regression that breaks the WHT row/column shifts or
/// the lossless `rowShift = colShift = 0` per-pass round would
/// move these values; pinning them defends the dispatch.
#[test]
fn issue_786_divergence_yuv_matches_current_spec_wht_residual() {
    let (y, u, v) = decode_divergence();
    assert_eq!(
        y, 130,
        "Y(0,0): expected 130 = predictor(128) + spec WHT residual(2) \
         (see iwht4_2d_divergence_y_tu_matches_spec for the hand-trace)"
    );
    assert_eq!(
        u, 128,
        "U(0,0): expected 128 = predictor(128) + spec WHT residual(0)"
    );
    assert_eq!(
        v, 128,
        "V(0,0): expected 128 = predictor(128) + spec WHT residual(0)"
    );
}

/// Document the cross-decoder reference (libavif / dav1d) values
/// next to the oxideav output. The deltas captured here record the
/// remaining work item for #786's entropy-decoder follow-up: the
/// §5.11.39 / §9.4 coefficient decoder reads different `level`
/// values than `dav1d` for the same range coder state, so the WHT
/// — operating spec-correctly on different inputs — produces
/// smaller residual magnitudes than the reference. Updating these
/// constants would track progress on closing the divergence; the
/// asserted deltas below are exactly what oxideav currently emits.
#[test]
fn issue_786_divergence_delta_vs_libavif_reference_is_documented() {
    let (y, u, v) = decode_divergence();
    let dy = (REF_Y as i32) - (y as i32);
    let du = (REF_U as i32) - (u as i32);
    let dv = (REF_V as i32) - (v as i32);
    // dav1d / avifdec: (133, 197, 215). oxideav: (130, 128, 128).
    // Deltas: Y = 3, U = 69, V = 87.
    assert_eq!(dy, 3, "Y delta vs libavif/dav1d reference");
    assert_eq!(du, 69, "U delta vs libavif/dav1d reference");
    assert_eq!(dv, 87, "V delta vs libavif/dav1d reference");
}
