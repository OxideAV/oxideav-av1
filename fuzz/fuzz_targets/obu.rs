#![no_main]
//! `obu` fuzz target — OBU framing layer panic-freedom.
//!
//! Exercises the §5.2 low-overhead OBU framing and the §4.10.5
//! `leb128()` primitive in isolation, below the full decode pipeline:
//!
//! * [`oxideav_av1::parse_leb128`] (§4.10.5) on the raw input.
//! * [`oxideav_av1::parse_obu`] (§5.3.2 header + §5.3.3 extension
//!   header) on the raw input.
//! * [`oxideav_av1::ObuIter`] (§5.2 concatenation walk), and for every
//!   sequence-header OBU it surfaces, a [`oxideav_av1::parse_sequence_header`]
//!   (§5.5) on that descriptor's payload.
//!
//! The contract under test: panic-freedom on every prefix / field
//! shape. Truncated size fields, oversized leb128 values, a set
//! forbidden bit, an extension flag with no extension byte, and payload
//! lengths that overrun the buffer must all surface typed errors.

use libfuzzer_sys::fuzz_target;

use oxideav_av1::{parse_leb128, parse_obu, ObuIter, ObuType};

fuzz_target!(|data: &[u8]| {
    // §4.10.5 leb128 primitive on the raw bytes.
    let _ = parse_leb128(data);

    // §5.3.2 single-OBU parse on the raw bytes.
    let _ = parse_obu(data);

    // §5.2 concatenation walk. Iterate every descriptor; on each
    // sequence-header OBU, re-parse its payload through the §5.5
    // sequence_header_obu grammar so that deeper layer is also driven
    // with attacker bytes.
    for desc in ObuIter::new(data) {
        match desc {
            Ok(d) => {
                if d.obu_type == ObuType::SequenceHeader {
                    let _ = oxideav_av1::parse_sequence_header(d.payload);
                }
            }
            Err(_) => break,
        }
    }
});
