//! Conformance tests for the spec-faithful frame decode driver
//! ([`oxideav_av1::decoder::decode_av1_spec`]) against **independently
//! produced** AV1 bitstreams and their expected decoded pixels.
//!
//! Unlike the encoder-mirror round-trip suites (which prove
//! encoder/decoder *self*-consistency), these fixtures were produced by
//! an external AV1 encoder, and the `expected` planes are what an
//! independent third-party AV1 decoder reconstructs from the same
//! bitstream (fixture corpus staged under `docs/video/av1/fixtures/`;
//! both binaries used strictly as opaque black-box tools). A byte-exact
//! match therefore validates the §5.11 syntax walk, the §8.2/§8.3
//! symbol decode + CDF adaptation, the §7.11.2 intra prediction, the
//! §7.12.2/§7.12.3 dequantization, and the §7.13 inverse transforms
//! against the AV1 specification itself — not against this crate's own
//! encoder.
//!
//! The fixture bytes are embedded as hex so the test runs in per-crate
//! CI without the docs checkout.

use oxideav_av1::decoder::decode_av1_spec;

fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "hex literal must have even length");
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// Decode `ivf_hex` through the spec driver and compare the
/// concatenated output planes byte-for-byte against `expected_hex`.
fn assert_decodes_byte_exact(name: &str, ivf_hex: &str, expected_hex: &str, frames_hint: usize) {
    let ivf = unhex(ivf_hex);
    let expected = unhex(expected_hex);
    let frames = decode_av1_spec(&ivf)
        .unwrap_or_else(|e| panic!("{name}: spec driver rejected the fixture: {e:?}"));
    assert_eq!(frames.len(), frames_hint, "{name}: frame count");
    let mut got = Vec::with_capacity(expected.len());
    for f in &frames {
        for p in &f.planes {
            got.extend_from_slice(p);
        }
    }
    assert_eq!(got.len(), expected.len(), "{name}: output length");
    if got != expected {
        let n = got
            .iter()
            .zip(expected.iter())
            .filter(|(a, b)| a != b)
            .count();
        panic!(
            "{name}: decoded pixels differ from the independent decoder in {n} of {} bytes",
            expected.len()
        );
    }
}

// ---------------------------------------------------------------------
// `tiny-i-only-16x16-prof0` — the smallest corpus stream: one 16×16
// profile-0 4:2:0 8-bit KEY frame (single BLOCK_16X16 leaf, DC_PRED
// luma + chroma, TX_16X16 under TX_MODE_LARGEST, `base_q_idx = 120`
// lossy quant, deblock levels 0, all-zero CDEF strengths, LR off).
// ---------------------------------------------------------------------

const TINY_I_ONLY_16X16_IVF: &str = concat!(
    "444b494600002000415630311000100019000000010000000100000000000000240000000000",
    "00000000000012000a0a000000019ffbfff3008032141000bc00000240000000789d762f676c",
    "c7ee5180",
);

const TINY_I_ONLY_16X16_EXPECTED_YUV: &str = concat!(
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "5151515151515151515151515151515151515151515151515151515151515151515151515151",
    "515151515151515151515151515151515151515151515151515151515a5a5a5a5a5a5a5a5a5a",
    "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a",
    "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5aefefefefefefefefefefefefefefefefefefefefefef",
    "efefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefefef",
    "efefefef",
);

/// The first external-encoder AV1 keyframe this crate decodes to pixels
/// byte-identical to an independent decoder's output.
#[test]
fn tiny_i_only_16x16_prof0_decodes_byte_exact() {
    assert_decodes_byte_exact(
        "tiny-i-only-16x16-prof0",
        TINY_I_ONLY_16X16_IVF,
        TINY_I_ONLY_16X16_EXPECTED_YUV,
        1,
    );
}

/// Same stream through the driver twice — the decode is deterministic
/// and holds no cross-call state.
#[test]
fn spec_driver_is_deterministic() {
    let ivf = unhex(TINY_I_ONLY_16X16_IVF);
    let a = decode_av1_spec(&ivf).unwrap();
    let b = decode_av1_spec(&ivf).unwrap();
    assert_eq!(a, b);
}

/// Truncated IVF payloads surface a typed error, never a panic. (A cut
/// at exactly the 32-byte IVF file-header boundary is the one
/// well-formed prefix: an IVF with zero frame records decodes to zero
/// frames.)
#[test]
fn spec_driver_rejects_truncated_input() {
    let ivf = unhex(TINY_I_ONLY_16X16_IVF);
    for cut in [0, 10, 31, 44, 60, ivf.len() - 1] {
        assert!(
            decode_av1_spec(&ivf[..cut]).is_err(),
            "truncation at {cut} must be rejected"
        );
    }
    assert_eq!(
        decode_av1_spec(&ivf[..32]).unwrap(),
        vec![],
        "header-only IVF decodes to zero frames"
    );
}
