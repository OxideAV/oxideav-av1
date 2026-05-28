//! §7.5 temporal-unit aggregator — the glue that takes typed
//! [`SequenceHeader`] / [`FrameHeader`] descriptors and produces a
//! complete, parser-walkable byte buffer.
//!
//! Round 208 (encoder arc 3) wires:
//!
//!   * §5.5.1 `sequence_header_obu()` payload (already landed) ⇒
//!     §5.3.1 `OBU_SEQUENCE_HEADER` framing with §5.3.4
//!     `trailing_bits` ([`encode_sequence_header_obu`]).
//!   * §5.9.1 `frame_header_obu()` payload (already landed) ⇒
//!     `OBU_FRAME_HEADER` framing with §5.3.4 `trailing_bits`.
//!   * §7.5 temporal-unit assembly: `OBU_TEMPORAL_DELIMITER`
//!     prefix + optional `OBU_SEQUENCE_HEADER` + the per-frame OBUs
//!     in caller order ([`encode_temporal_unit`]).
//!
//! Until the §5.11 `tile_group_obu()` writer lands (next arc), the
//! per-frame leg emits a stand-alone `OBU_FRAME_HEADER` with no
//! following `OBU_TILE_GROUP`. The §5.3.1 wrapper still accepts that
//! (every emitted OBU passes through `parse_obu`); the parser cannot
//! reconstruct pixels without a tile body, but it can read the
//! `SequenceHeader` + `FrameHeader` round-trip — which is the gold
//! standard for this arc.
//!
//! Spec references:
//!
//!   * §5.3.1 — `open_bitstream_unit()` framing tail (`trailing_bits`
//!     gate).
//!   * §5.3.4 — `trailing_bits(nbBits)` syntax.
//!   * §5.5.1 — General sequence header OBU syntax.
//!   * §5.9.1 / §5.9.2 — `frame_header_obu()` / `uncompressed_header()`.
//!   * §7.5 — Temporal unit decoding process; TD-prefix invariant.

use crate::encoder::frame_obu::write_frame_header_obu;
use crate::encoder::obu::{build_temporal_unit, write_obu_with_size, ObuFrame, ObuHeader};
use crate::encoder::sequence_obu::write_sequence_header_obu;
use crate::frame_header::FrameHeader;
use crate::obu::ObuType;
use crate::sequence_header::SequenceHeader;

/// One frame's worth of plan for [`encode_temporal_unit`].
///
/// Currently carries only the [`FrameHeader`]; the next encoder arc
/// will extend this with the parsed tile-group plan and, eventually,
/// the per-block syntax.
#[derive(Debug, Clone)]
pub struct TemporalUnitPlan<'a> {
    /// Active `SequenceHeader` for this temporal unit. The encoder
    /// emits an `OBU_SEQUENCE_HEADER` iff [`Self::emit_sequence_header`]
    /// is set (typically true at the start of a coded video sequence
    /// and at every key frame for resilience; see §7.5 / §6.4.1).
    pub seq: &'a SequenceHeader,
    /// `true` to prefix the temporal unit with an
    /// `OBU_SEQUENCE_HEADER`. False allows downstream consumers to
    /// reuse the sequence header from a prior temporal unit (a §7.5
    /// inter / B / key-no-resilience case).
    pub emit_sequence_header: bool,
    /// Frame headers to emit, in display / coding order. Each header
    /// produces one `OBU_FRAME_HEADER` with the §5.3.4 trailing_bits
    /// trailer. (The matching `OBU_TILE_GROUP` will land in the next
    /// arc — see module docs.)
    pub frames: &'a [FrameHeader],
}

/// Build an `OBU_SEQUENCE_HEADER` complete with §5.3.4 trailing_bits
/// and the §5.3.1 `obu_size` size field. Returns the bytes the
/// caller can splice into a temporal unit.
pub fn encode_sequence_header_obu(seq: &SequenceHeader) -> Vec<u8> {
    let body = write_sequence_header_obu(seq);
    let mut out = Vec::new();
    write_obu_with_size(&mut out, &ObuHeader::new(ObuType::SequenceHeader), &body);
    out
}

/// Build an `OBU_FRAME_HEADER` complete with §5.3.4 trailing_bits and
/// the §5.3.1 `obu_size` size field.
pub fn encode_frame_header_obu(fh: &FrameHeader, seq: &SequenceHeader) -> Vec<u8> {
    let body = write_frame_header_obu(fh, seq);
    let mut out = Vec::new();
    write_obu_with_size(&mut out, &ObuHeader::new(ObuType::FrameHeader), &body);
    out
}

/// Encode a §7.5 temporal unit: `OBU_TEMPORAL_DELIMITER` prefix +
/// optional `OBU_SEQUENCE_HEADER` + one `OBU_FRAME_HEADER` per
/// [`TemporalUnitPlan::frames`].
///
/// The returned buffer is a complete §5.2 low-overhead bytestream
/// fragment: every OBU walks back through [`crate::obu::ObuIter`] and
/// the embedded sequence/frame headers round-trip through their
/// respective parsers.
pub fn encode_temporal_unit(plan: &TemporalUnitPlan<'_>) -> Vec<u8> {
    // Build per-frame OBU bodies (header writers return byte-aligned
    // payloads — the §5.3.4 trailer is applied by build_temporal_unit
    // ⇒ write_obu_with_size).
    let frame_obus: Vec<ObuFrame> = plan
        .frames
        .iter()
        .map(|fh| ObuFrame::new(ObuType::FrameHeader, write_frame_header_obu(fh, plan.seq)))
        .collect();
    let seq_body = plan
        .emit_sequence_header
        .then(|| write_sequence_header_obu(plan.seq));
    build_temporal_unit(seq_body.as_deref(), &frame_obus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::frame_obu::write_frame_header_obu;
    use crate::encoder::sequence_obu::write_sequence_header_obu;
    use crate::frame_header::parse_frame_header;
    use crate::obu::{ObuIter, ObuType};
    use crate::sequence_header::parse_sequence_header;

    // Tiny-i-only-16x16-prof0 sequence header payload, also used by
    // the parser- and writer-side tests.
    const TINY_SEQ_PAYLOAD: &[u8] = &[0x00, 0x00, 0x00, 0x01, 0x9f, 0xfb, 0xff, 0xf3, 0x00, 0x80];
    // Tiny-i-only-16x16-prof0 frame header payload.
    const TINY_FRAME_PAYLOAD: &[u8] = &[
        0x10, 0x00, 0xbc, 0x00, 0x00, 0x02, 0x40, 0x00, 0x00, 0x00, 0x78, 0x9d, 0x76, 0x2f, 0x67,
        0x6c, 0xc7, 0xee, 0x51, 0x80,
    ];

    fn tiny_seq() -> SequenceHeader {
        parse_sequence_header(TINY_SEQ_PAYLOAD).unwrap()
    }

    fn tiny_fh(seq: &SequenceHeader) -> FrameHeader {
        parse_frame_header(TINY_FRAME_PAYLOAD, seq).unwrap()
    }

    #[test]
    fn encode_sequence_header_obu_round_trips_through_parser() {
        let seq = tiny_seq();
        let bytes = encode_sequence_header_obu(&seq);
        // Walk one OBU back out.
        let mut iter = ObuIter::new(&bytes);
        let desc = iter.next().unwrap().unwrap();
        assert_eq!(desc.obu_type, ObuType::SequenceHeader);
        // SH body parses (we strip the §5.3.4 trailer by handing only
        // the body bytes to the parser — the parser stops at
        // `bits_consumed` and ignores the trailer).
        let reparsed = parse_sequence_header(desc.payload).expect("SH payload parses");
        // Normalise the bit-count field — the parser may report a
        // different `bits_consumed` once trailing-bit padding is in
        // the slice, but the structural fields must match.
        let mut expected = seq.clone();
        expected.bits_consumed = reparsed.bits_consumed;
        assert_eq!(reparsed, expected);
    }

    #[test]
    fn encode_frame_header_obu_round_trips_through_parser() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let bytes = encode_frame_header_obu(&fh, &seq);
        let mut iter = ObuIter::new(&bytes);
        let desc = iter.next().unwrap().unwrap();
        assert_eq!(desc.obu_type, ObuType::FrameHeader);
        let reparsed = parse_frame_header(desc.payload, &seq).expect("FH payload parses");
        let mut expected = fh.clone();
        expected.bits_consumed = reparsed.bits_consumed;
        assert_eq!(reparsed, expected);
    }

    #[test]
    fn encode_temporal_unit_emits_td_sh_fh_in_order() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let frames = [fh.clone()];
        let plan = TemporalUnitPlan {
            seq: &seq,
            emit_sequence_header: true,
            frames: &frames,
        };
        let bytes = encode_temporal_unit(&plan);
        let descs: Vec<_> = ObuIter::new(&bytes).collect::<Result<_, _>>().unwrap();
        assert_eq!(descs.len(), 3);
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[0].payload_len, 0);
        assert_eq!(descs[1].obu_type, ObuType::SequenceHeader);
        // SH body matches what `write_sequence_header_obu` emits, plus
        // the 0x80 §5.3.4 trailer.
        let expected_sh_body = write_sequence_header_obu(&seq);
        assert_eq!(
            &descs[1].payload[..expected_sh_body.len()],
            &expected_sh_body[..]
        );
        assert_eq!(descs[1].payload[expected_sh_body.len()], 0x80);
        assert_eq!(descs[2].obu_type, ObuType::FrameHeader);
        let expected_fh_body = write_frame_header_obu(&fh, &seq);
        assert_eq!(
            &descs[2].payload[..expected_fh_body.len()],
            &expected_fh_body[..]
        );
        assert_eq!(descs[2].payload[expected_fh_body.len()], 0x80);
    }

    #[test]
    fn encode_temporal_unit_without_sh_omits_sequence_header() {
        let seq = tiny_seq();
        let fh = tiny_fh(&seq);
        let frames = [fh];
        let plan = TemporalUnitPlan {
            seq: &seq,
            emit_sequence_header: false,
            frames: &frames,
        };
        let bytes = encode_temporal_unit(&plan);
        let descs: Vec<_> = ObuIter::new(&bytes).collect::<Result<_, _>>().unwrap();
        assert_eq!(descs.len(), 2);
        assert_eq!(descs[0].obu_type, ObuType::TemporalDelimiter);
        assert_eq!(descs[1].obu_type, ObuType::FrameHeader);
    }
}
