//! AV1 OBU framing — encoder side, inverse of [`crate::obu`].
//!
//! Each emitted OBU is wrapped with a 1-byte header (no extension bits
//! used in round 1) plus a `leb128` size prefix (`obu_has_size_field=1`).
//! The size prefix is computed from the payload and emitted as a
//! variable-length leb128 — fixed-length back-patching is not needed
//! because the payload is fully assembled in a temporary buffer first.
//!
//! Spec references: §5.3.2 (OBU header), §5.3 (OBU framing).

use crate::encoder::bitwriter::BitWriter;
use crate::obu::ObuType;

/// Emit one OBU with `obu_has_size_field=1`, no extension bits, no
/// temporal/spatial id (round 1 produces a single-layer stream).
///
/// Mutates `out` in place. Caller-supplied `payload` becomes the OBU
/// body verbatim; the spec mandates that the payload itself terminates
/// with the relevant `byte_alignment()` / `trailing_bits()` markers
/// (Sequence Header and Frame Header bodies do; Tile Group body and
/// Temporal Delimiter do not).
pub fn write_obu(out: &mut Vec<u8>, obu_type: ObuType, payload: &[u8]) {
    // OBU header byte (§5.3.2):
    //   obu_forbidden_bit  = 0  (1 bit, MSB)
    //   obu_type           = 4 bits
    //   obu_extension_flag = 0
    //   obu_has_size_field = 1
    //   obu_reserved_1bit  = 0
    let header_byte = ((obu_type as u8) & 0x0f) << 3 | 0x02;
    out.push(header_byte);
    // leb128 size prefix.
    let mut bw = BitWriter::new();
    bw.leb128(payload.len() as u64);
    out.extend_from_slice(&bw.finish());
    out.extend_from_slice(payload);
}

/// Convenience entry point for the empty OBU_TEMPORAL_DELIMITER (§7.5).
/// Emits 2 bytes: `0x12` (header byte for type=2 / has_size=1) + `0x00`
/// (leb128 size 0).
pub fn write_temporal_delimiter(out: &mut Vec<u8>) {
    write_obu(out, ObuType::TemporalDelimiter, &[]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::obu::{iter_obus, ObuType};

    #[test]
    fn temporal_delimiter_is_two_bytes() {
        let mut out = Vec::new();
        write_temporal_delimiter(&mut out);
        assert_eq!(out, vec![0x12, 0x00]);
    }

    #[test]
    fn write_then_iter_roundtrip() {
        let mut out = Vec::new();
        write_temporal_delimiter(&mut out);
        write_obu(&mut out, ObuType::SequenceHeader, &[0xAB, 0xCD, 0xEF]);
        write_obu(&mut out, ObuType::Frame, &[0x01, 0x02]);

        let parsed: Vec<_> = iter_obus(&out).map(|r| r.unwrap()).collect();
        assert_eq!(parsed.len(), 3);
        assert_eq!(parsed[0].header.obu_type, ObuType::TemporalDelimiter);
        assert!(parsed[0].payload.is_empty());
        assert_eq!(parsed[1].header.obu_type, ObuType::SequenceHeader);
        assert_eq!(parsed[1].payload, &[0xAB, 0xCD, 0xEF]);
        assert_eq!(parsed[2].header.obu_type, ObuType::Frame);
        assert_eq!(parsed[2].payload, &[0x01, 0x02]);
    }

    #[test]
    fn header_byte_layout() {
        // SequenceHeader = type 1 → high nibble 0x08, with has_size=1
        // bit 1 = 0x0A. Extension = 0, reserved = 0.
        let mut out = Vec::new();
        write_obu(&mut out, ObuType::SequenceHeader, &[]);
        assert_eq!(out[0], 0x0A);
        // Frame = type 6 → 0x30 | 0x02 = 0x32.
        let mut out = Vec::new();
        write_obu(&mut out, ObuType::Frame, &[]);
        assert_eq!(out[0], 0x32);
    }
}
