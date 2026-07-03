//! Regression corpus for inputs the scheduled `Fuzz` workflow found.
//!
//! Each case is a libFuzzer-minimized adversarial input that once
//! panicked the decoder. The contract (same as the `decode` fuzz
//! target): any byte shape must produce `Ok(..)` or a typed
//! [`oxideav_av1::Error`] — never a panic, overflow, or hang.

/// Decode raw hex into bytes (test-local helper; fixtures embed hex so
/// the suite runs in per-crate CI without a fixture checkout).
fn hex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "hex literal must have even length");
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("valid hex"))
        .collect()
}

/// 2026-07-03 scheduled-Fuzz crash `8f12c488…`: a truncated IVF frame
/// whose coefficient payload ends inside the §5.11.39 golomb chain.
/// The §8.2.2 arithmetic decoder then pads with zero bits forever, so
/// the uncapped `do { length++ } while ( !golomb_length_bit )` loop
/// of §5.11.39 spun ~2^32 iterations and overflowed `length`
/// (`attempt to add with overflow` at the `length += 1`). Fixed by
/// the 30-bit robustness cap surfacing
/// [`oxideav_av1::Error::GolombLengthOverflow`].
#[test]
fn golomb_length_chain_is_bounded_on_truncated_coefficients() {
    let bytes = hex(
        "444b494600002000000020443f4946cccc57cc7acccccccccccccc4b491c0000\
         2000000028cccccccc9e55af0e46095f4cf7ffd1ff46f6001f00097affff0000\
         000000c52206fff7ff286a00",
    );
    // Success and every typed error are both acceptable; only a panic
    // or a hang (caught by the test harness / CI timeout) is a
    // finding. This input specifically must fail fast rather than
    // walk a multi-billion-iteration zero-bit tail.
    let start = std::time::Instant::now();
    let _ = oxideav_av1::decode_av1(&bytes);
    assert!(
        start.elapsed() < std::time::Duration::from_secs(30),
        "decode of a 76-byte adversarial input must not take tens of seconds"
    );
}

/// 2026-07-03 dispatched-Fuzz crash `c055549e…` (surfaced once the
/// golomb cap above unblocked deeper coverage): a §5.9.30
/// `film_grain_params()` whose chroma point-count `f(4)` literal codes
/// a value past the `MAX_NUM_CHROMA_POINTS = 10` conformance bound,
/// indexing past the fixed-size point arrays (`index out of bounds:
/// the len is 10 but the index is 10`). Fixed by the §5.9.30
/// conformance checks surfacing
/// [`oxideav_av1::Error::FilmGrainPointCountOverflow`].
#[test]
fn film_grain_point_counts_are_bounded() {
    let bytes = hex(
        "444b494600002000002900000000000000fff8cccc33cc23cccc0800000000002c\
         000000cc28cccc4a61a04b0e4d095d1e00ff01823b00051f001c0a00003100000e\
         2000f7ffa2a9b61f4400fffff8cce001000204ffff00",
    );
    // Success and every typed error are both acceptable; only a panic
    // is a finding.
    let _ = oxideav_av1::decode_av1(&bytes);
}
