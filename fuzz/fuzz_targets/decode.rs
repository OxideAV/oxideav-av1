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

//!
//! The session runs with a 2^20-luma-sample picture ceiling (a
//! 1024 × 1024 frame; the library default is Annex A's largest
//! `MaxPicSize`, 35 651 584): every frame-sized reservation is
//! proportional to that area, and libFuzzer's 2 GiB RSS limit sits
//! well below what the library-default ceiling admits, so the cap
//! keeps the harness measuring panics instead of the runner's memory.

use libfuzzer_sys::fuzz_target;
use oxideav_av1::decoder::SpecDecodeSession;

/// Picture-size ceiling for the harness (luma samples).
const FUZZ_MAX_PICTURE_SIZE: u32 = 1 << 20;

fuzz_target!(|data: &[u8]| {
    // We deliberately ignore the result: success and every typed error
    // are both acceptable outcomes. Only a panic (caught by libFuzzer)
    // is a finding.
    let mut session = SpecDecodeSession::new();
    session.set_max_picture_size(FUZZ_MAX_PICTURE_SIZE);
    let _ = session.decode_ivf(data);
});
