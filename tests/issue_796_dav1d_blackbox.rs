//! Round-68 black-box smoke test for the §5.11.39 sign-bit divergence
//! fixture (workspace task #801). When `dav1d` is on `PATH`, this
//! test runs the binary against the 27-byte `DIVERGENCE_OBU` and
//! asserts the documented `(Y, U, V) = (133, 197, 215)` reference.
//!
//! The test is skipped (with a stdout note) on machines without
//! `dav1d` installed — CI runners can run it on macOS where `brew
//! install dav1d` is the standard path. This pinning ensures any
//! future dav1d update that perturbs the reference YUV becomes a
//! loud failure instead of a silent reinterpretation of the round-67
//! "(133, 197, 215)" pin in `issue_796_sign_bits_match_dav1d.rs`.
//!
//! Memory `feedback_no_external_libs`: dav1d binary is OK as a
//! black-box validator. No dav1d source files are read or vendored;
//! we only invoke the released CLI and parse its raw-YUV output.
//!
//! See `docs/video/av1/specs/dav1d-range-coder-divergence-call-idx-27.md`
//! for the full round-68 audit and round-69 attack vectors.

use std::io::Write;
use std::process::{Command, Stdio};

/// 27-byte AV1 OBU stream — duplicate of the constant in
/// `tests/issue_796_sign_bits_match_dav1d.rs` so each test file is
/// self-contained.
const DIVERGENCE_OBU: &[u8] = &[
    0x12, 0x00, 0x0a, 0x04, 0x38, 0x00, 0x0e, 0x49, 0x32, 0x11, 0x10, 0x00, 0x00, 0x19, 0xb9, 0xca,
    0xe3, 0x37, 0x39, 0x09, 0x47, 0xd9, 0x6e, 0x65, 0x96, 0x64, 0xaf,
];

fn dav1d_available() -> bool {
    Command::new("dav1d")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Pipe `DIVERGENCE_OBU` through `dav1d -i - -o - --muxer yuv -q`
/// and read the first 3 bytes of the raw YUV output as `(Y, U, V)`.
/// Returns `None` if dav1d is not installed.
fn dav1d_decode(obu: &[u8]) -> Option<(u8, u8, u8)> {
    let tmpdir = std::env::temp_dir();
    let in_path = tmpdir.join("oxideav_av1_issue_796_input.obu");
    let out_path = tmpdir.join("oxideav_av1_issue_796_output.yuv");
    std::fs::write(&in_path, obu).ok()?;
    let _ = std::fs::remove_file(&out_path);
    let status = Command::new("dav1d")
        .args([
            "-i",
            in_path.to_str()?,
            "-o",
            out_path.to_str()?,
            "-q",
            "--muxer",
            "yuv",
        ])
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out_path).ok()?;
    if bytes.len() < 3 {
        return None;
    }
    Some((bytes[0], bytes[1], bytes[2]))
}

/// Pin `dav1d 1.5.3 → (133, 197, 215)` for the round-67/68 fixture.
/// Skipped when dav1d is not installed.
#[test]
fn issue_796_dav1d_blackbox_reference_yuv_pinned() {
    if !dav1d_available() {
        let mut out = std::io::stderr();
        let _ = writeln!(
            out,
            "[issue_796_dav1d_blackbox] skipped — `dav1d` not on PATH; \
             install via `brew install dav1d` to run this test."
        );
        return;
    }
    let yuv =
        dav1d_decode(DIVERGENCE_OBU).expect("dav1d is on PATH but decoding DIVERGENCE_OBU failed");
    assert_eq!(
        yuv,
        (133, 197, 215),
        "dav1d 1.5.x reference for the §5.11.39 sign-bit divergence \
         fixture is pinned at (Y, U, V) = (133, 197, 215). If this \
         changes the round-67 audit + round-68 doc need a refresh — \
         see docs/video/av1/specs/dav1d-range-coder-divergence-call-idx-27.md"
    );
}

/// Round-68 bit-flip witness: flipping bit_pos=46 (byte 18 bit 1 of
/// the OBU, the first bit consumed during the call_idx=27 sign read)
/// shifts dav1d's `U` plane from 197 to 129 while keeping `Y = 133`.
/// This is the strongest single-bit pin we have on which input bit
/// drives the sign-bit divergence: dav1d's chroma decode behaves like
/// our chroma decode (defaulted) when this bit is forced, confirming
/// the divergence is a bit-stream alignment issue and not a
/// chroma-context issue.
///
/// Skipped when dav1d is not on PATH.
#[test]
fn issue_796_dav1d_blackbox_bit_flip_46_perturbs_chroma() {
    if !dav1d_available() {
        let mut out = std::io::stderr();
        let _ = writeln!(
            out,
            "[issue_796_dav1d_blackbox] skipped — `dav1d` not on PATH"
        );
        return;
    }
    let mut perturbed = DIVERGENCE_OBU.to_vec();
    // bit_pos 46 within tile data (which starts at OBU byte 13) =
    // tile-byte 5 = OBU byte 18, bit index 7 - (46 & 7) = 1.
    perturbed[18] ^= 1 << 1;
    // dav1d --strict 0 lets the perturbed stream parse past Annex-A
    // conformance asserts.
    let tmpdir = std::env::temp_dir();
    let in_path = tmpdir.join("oxideav_av1_issue_796_perturbed.obu");
    let out_path = tmpdir.join("oxideav_av1_issue_796_perturbed.yuv");
    std::fs::write(&in_path, &perturbed).expect("write perturbed OBU");
    let _ = std::fs::remove_file(&out_path);
    let status = Command::new("dav1d")
        .args([
            "-i",
            in_path.to_str().unwrap(),
            "-o",
            out_path.to_str().unwrap(),
            "-q",
            "--muxer",
            "yuv",
            "--strict",
            "0",
        ])
        .status()
        .expect("invoke dav1d --strict 0");
    if !status.success() {
        // Some dav1d builds may reject this perturbation entirely; the
        // test is informational, not a hard regression gate.
        eprintln!(
            "[issue_796_dav1d_blackbox] dav1d rejected the bit_pos=46 \
             perturbation; not a hard failure."
        );
        return;
    }
    let bytes = std::fs::read(&out_path).expect("read perturbed YUV");
    if bytes.len() >= 3 {
        let (y, u, v) = (bytes[0], bytes[1], bytes[2]);
        // Round-68 documented witness:
        // - Y must remain 133 (luma decode untouched).
        // - U must drop to 129 (chroma defaulted by the perturbed
        //   sign read).
        // - V must remain 128 (chroma defaulted as well).
        assert_eq!(
            (y, u, v),
            (133, 129, 128),
            "dav1d bit_pos=46 perturbation: round-68 doc records \
             (133, 129, 128); a different result here means dav1d 1.5.x \
             changed its perturbation-handling and the round-68 trace \
             needs a rerun."
        );
    }
}
