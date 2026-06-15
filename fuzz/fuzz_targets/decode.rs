#![no_main]
//! `decode` fuzz target — full decode pipeline panic-freedom.
//!
//! Drives attacker-controlled bytes straight through the crate's
//! top-level [`oxideav_av1::decode_av1`] entry. That covers the IVF
//! container parse, the §5.2 / §5.3 OBU framing walk, the §5.5 / §5.9
//! sequence + frame header parse, and the §5.11 tile / partition /
//! reconstruction pipeline. Every field on every layer is
//! attacker-chosen.
//!
//! The contract under test: no input shape may panic. A malformed
//! stream must surface a typed [`oxideav_av1::Error`] (or decode), never
//! an out-of-bounds index, an arithmetic overflow, or an `unwrap` on a
//! value the attacker forced to `None` / `Err`.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // We deliberately ignore the result: success and every typed error
    // are both acceptable outcomes. Only a panic (caught by libFuzzer)
    // is a finding.
    let _ = oxideav_av1::decode_av1(data);
});
