//! Annex B length-delimited bitstream format (read + write).
//!
//! §5.2 defines the low-overhead bitstream format this crate has
//! always consumed (every OBU carries `obu_has_size_field = 1` and
//! its own `leb128` size). Annex B defines the alternative
//! length-delimited packing: a `leb128` `temporal_unit_size` prefix
//! per temporal unit, a `frame_unit_size` prefix per frame unit
//! inside it, and an `obu_length` prefix per OBU inside that —
//! `bitstream() → temporal_unit(sz) → frame_unit(sz) →
//! open_bitstream_unit(obu_length)` per the Annex B.2 syntax.
//!
//! Reading ([`split_temporal_units`]) converts each length-delimited
//! temporal unit into the equivalent §5.2 low-overhead byte run the
//! spec decode session already consumes: OBUs that carry their own
//! size field are validated for the Annex B.3 `obu_size`/`obu_length`
//! consistency rule and passed through verbatim; OBUs without one
//! (legal in Annex B — the outer length replaces the size field) are
//! re-emitted with `obu_has_size_field = 1` and a synthesised
//! `leb128(payload length)`. The two Annex B.3 structural rules are
//! enforced: the FIRST OBU of the FIRST frame unit of every temporal
//! unit must be a temporal delimiter, and temporal delimiters must
//! appear nowhere else.
//!
//! Writing ([`build_from_temporal_units`]) walks each low-overhead
//! §7.5 temporal unit this crate's encoders emit and wraps it in the
//! Annex B framing: the temporal delimiter (plus any sequence header
//! / metadata prefix) opens the first frame unit together with the
//! first frame's OBUs, and every subsequent frame-carrying OBU
//! (`OBU_FRAME`, or an `OBU_FRAME_HEADER` .. tile-group run) starts a
//! new frame unit — "all the frame header and tile group OBUs
//! required for decoding a single frame must be within the same
//! frame_unit (and a frame_unit must not contain frame headers for
//! more than one frame)" (Annex B.3).

use crate::obu::{parse_leb128, parse_obu, ObuType};
use crate::Error;

/// Encode one `leb128()` value per §4.10.5 (minimal length).
fn write_leb128(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

/// Split one Annex B length-delimited bitstream into its temporal
/// units, each converted to the equivalent §5.2 low-overhead byte
/// run (every OBU carrying `obu_has_size_field = 1`).
///
/// ## Errors
///
/// * [`Error::UnexpectedEnd`] — a `temporal_unit_size` /
///   `frame_unit_size` / `obu_length` runs past the buffer.
/// * [`Error::AnnexBInvalid`] — nested sizes that do not tile their
///   container exactly, an `obu_size`/`obu_length` inconsistency
///   (Annex B.3 "deemed invalid"), a temporal delimiter anywhere but
///   the first OBU of a temporal unit's first frame unit, or a
///   first OBU that is not a temporal delimiter.
pub fn split_temporal_units(bytes: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut pos = 0usize;
    while pos < bytes.len() {
        // bitstream(): temporal_unit_size leb128()
        let (tu_size, n) = parse_leb128(&bytes[pos..])?;
        pos += n;
        let tu_size = usize::try_from(tu_size).map_err(|_| Error::SizeOverflow)?;
        let tu_end = pos.checked_add(tu_size).ok_or(Error::SizeOverflow)?;
        if tu_end > bytes.len() {
            return Err(Error::UnexpectedEnd);
        }
        let mut low_overhead: Vec<u8> = Vec::with_capacity(tu_size + 8);
        let mut first_fu = true;
        while pos < tu_end {
            // temporal_unit( sz ): frame_unit_size leb128()
            let (fu_size, n) = parse_leb128(&bytes[pos..tu_end])?;
            pos += n;
            let fu_size = usize::try_from(fu_size).map_err(|_| Error::SizeOverflow)?;
            let fu_end = pos.checked_add(fu_size).ok_or(Error::SizeOverflow)?;
            if fu_end > tu_end {
                return Err(Error::AnnexBInvalid);
            }
            let mut first_obu = true;
            while pos < fu_end {
                // frame_unit( sz ): obu_length leb128()
                let (obu_length, n) = parse_leb128(&bytes[pos..fu_end])?;
                pos += n;
                let obu_length = usize::try_from(obu_length).map_err(|_| Error::SizeOverflow)?;
                let obu_end = pos.checked_add(obu_length).ok_or(Error::SizeOverflow)?;
                if obu_end > fu_end {
                    return Err(Error::AnnexBInvalid);
                }
                append_low_overhead_obu(
                    &bytes[pos..obu_end],
                    first_fu && first_obu,
                    &mut low_overhead,
                )?;
                pos = obu_end;
                first_obu = false;
            }
            if pos != fu_end {
                return Err(Error::AnnexBInvalid);
            }
            first_fu = false;
        }
        if pos != tu_end {
            return Err(Error::AnnexBInvalid);
        }
        out.push(low_overhead);
    }
    Ok(out)
}

/// Convert one Annex-B-framed OBU (`obu_bytes` = exactly
/// `obu_length` bytes: header, optional extension byte, optional
/// size field, payload) into its §5.2 low-overhead form, enforcing
/// the Annex B.3 rules. `must_be_td` marks the first OBU of the
/// temporal unit's first frame unit.
fn append_low_overhead_obu(
    obu_bytes: &[u8],
    must_be_td: bool,
    out: &mut Vec<u8>,
) -> Result<(), Error> {
    let first = *obu_bytes.first().ok_or(Error::AnnexBInvalid)?;
    if (first >> 7) & 1 != 0 {
        return Err(Error::ForbiddenBitSet);
    }
    let obu_type = ObuType::from_raw((first >> 3) & 0x0f);
    let extension_flag = (first >> 2) & 1 != 0;
    let has_size_field = (first >> 1) & 1 != 0;
    // Annex B.3: TD first (and only first).
    let is_td = obu_type == ObuType::TemporalDelimiter;
    if must_be_td != is_td {
        return Err(Error::AnnexBInvalid);
    }
    let header_len = 1 + usize::from(extension_flag);
    if obu_bytes.len() < header_len {
        return Err(Error::UnexpectedEnd);
    }
    if has_size_field {
        // Annex B.3: obu_size and obu_length must be consistent —
        // re-parse through the low-overhead walker and demand the
        // OBU tiles its obu_length span exactly.
        let (_, consumed) = parse_obu(obu_bytes)?;
        if consumed != obu_bytes.len() {
            return Err(Error::AnnexBInvalid);
        }
        out.extend_from_slice(obu_bytes);
    } else {
        // Synthesise the size field: set bit 1 of the header byte,
        // keep the extension byte, emit leb128(payload length).
        let payload = &obu_bytes[header_len..];
        out.push(first | 0b10);
        if extension_flag {
            out.push(obu_bytes[1]);
        }
        write_leb128(out, payload.len() as u64);
        out.extend_from_slice(payload);
    }
    Ok(())
}

/// Wrap this crate's low-overhead §7.5 temporal units into one
/// Annex B length-delimited bitstream.
///
/// Frame-unit grouping per Annex B.3: the temporal delimiter (with
/// any sequence-header / metadata prefix) opens the first frame unit
/// together with the first frame's OBUs; every later frame-carrying
/// OBU starts a new frame unit. `OBU_FRAME` carries a whole frame;
/// an `OBU_FRAME_HEADER` opens a frame that its following tile-group
/// OBUs complete.
///
/// ## Errors
///
/// * Any [`crate::obu`] parse failure on the input units.
/// * [`Error::AnnexBInvalid`] — a unit that does not start with a
///   temporal delimiter (the §7.5 grammar this crate's encoders
///   always satisfy).
pub fn build_from_temporal_units(temporal_units: &[Vec<u8>]) -> Result<Vec<u8>, Error> {
    let mut out: Vec<u8> = Vec::new();
    for tu in temporal_units {
        // Walk the unit's OBUs and group them into frame units.
        let mut frame_units: Vec<Vec<u8>> = Vec::new();
        let mut current: Vec<u8> = Vec::new();
        let mut current_has_frame = false;
        let mut pos = 0usize;
        let mut first = true;
        while pos < tu.len() {
            let (desc, consumed) = parse_obu(&tu[pos..])?;
            let obu_bytes = &tu[pos..pos + consumed];
            pos += consumed;
            if first {
                if desc.obu_type != ObuType::TemporalDelimiter {
                    return Err(Error::AnnexBInvalid);
                }
                first = false;
            }
            let starts_frame = matches!(desc.obu_type, ObuType::Frame | ObuType::FrameHeader);
            if starts_frame && current_has_frame {
                frame_units.push(std::mem::take(&mut current));
                current_has_frame = false;
            }
            let mut obu_out: Vec<u8> = Vec::with_capacity(consumed + 4);
            write_leb128(&mut obu_out, consumed as u64);
            obu_out.extend_from_slice(obu_bytes);
            current.extend_from_slice(&obu_out);
            if starts_frame {
                current_has_frame = true;
            }
        }
        if !current.is_empty() {
            frame_units.push(current);
        }
        // temporal_unit( sz ) body: frame_unit_size + frame_unit per
        // unit; then the temporal_unit_size prefix.
        let mut tu_body: Vec<u8> = Vec::new();
        for fu in &frame_units {
            write_leb128(&mut tu_body, fu.len() as u64);
            tu_body.extend_from_slice(fu);
        }
        write_leb128(&mut out, tu_body.len() as u64);
        out.extend_from_slice(&tu_body);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal hand-built round trip: TD + one sized OBU_FRAME-ish
    /// blob survives build → split byte-exact.
    #[test]
    fn build_then_split_round_trips_low_overhead_units() {
        // TD: header 0x12 (type 2, has_size_field=1), size 0.
        // "Frame": header 0x32 (type 6 OBU_FRAME, has_size_field=1),
        // size 3, payload [1,2,3].
        let tu: Vec<u8> = vec![0x12, 0x00, 0x32, 0x03, 1, 2, 3];
        let annexb = build_from_temporal_units(std::slice::from_ref(&tu)).unwrap();
        let back = split_temporal_units(&annexb).unwrap();
        assert_eq!(back, vec![tu]);
    }

    /// An OBU without its own size field gains one on conversion.
    #[test]
    fn sizeless_obu_gains_a_size_field() {
        // Annex B TU: TD (sizeless: header 0x10, obu_length 1) in FU0
        // with a sizeless frame OBU (header 0x30, 3 payload bytes,
        // obu_length 4).
        let fu: Vec<u8> = vec![0x01, 0x10, 0x04, 0x30, 7, 8, 9];
        let mut annexb: Vec<u8> = Vec::new();
        write_leb128(&mut annexb, (fu.len() + 1) as u64);
        write_leb128(&mut annexb, fu.len() as u64);
        annexb.extend_from_slice(&fu);
        let back = split_temporal_units(&annexb).unwrap();
        assert_eq!(back, vec![vec![0x12, 0x00, 0x32, 0x03, 7, 8, 9]]);
    }

    /// Annex B.3: a first OBU that is not a TD is invalid, and a TD
    /// anywhere else is invalid.
    #[test]
    fn td_placement_is_enforced() {
        // First OBU is a frame.
        let fu: Vec<u8> = vec![0x04, 0x30, 7, 8, 9];
        let mut annexb: Vec<u8> = Vec::new();
        write_leb128(&mut annexb, fu.len() as u64);
        annexb.extend_from_slice(&fu);
        assert_eq!(split_temporal_units(&annexb), Err(Error::AnnexBInvalid));

        // TD after the first position.
        let fu: Vec<u8> = vec![0x01, 0x10, 0x01, 0x10];
        let mut annexb: Vec<u8> = Vec::new();
        write_leb128(&mut annexb, fu.len() as u64);
        annexb.extend_from_slice(&fu);
        assert_eq!(split_temporal_units(&annexb), Err(Error::AnnexBInvalid));
    }

    /// Annex B.3: inconsistent `obu_size` vs `obu_length` is invalid.
    #[test]
    fn size_length_inconsistency_is_invalid() {
        // TD sized 0 (2 bytes), then a sized frame OBU whose
        // obu_size (3) disagrees with obu_length (6 bytes of OBU
        // where the size-consistent length would be 5).
        let fu: Vec<u8> = vec![0x02, 0x12, 0x00, 0x06, 0x32, 0x03, 1, 2, 3, 0];
        let mut annexb: Vec<u8> = Vec::new();
        write_leb128(&mut annexb, fu.len() as u64);
        annexb.extend_from_slice(&fu);
        assert_eq!(split_temporal_units(&annexb), Err(Error::AnnexBInvalid));
    }

    /// Truncations surface typed errors.
    #[test]
    fn truncations_error_cleanly() {
        let tu: Vec<u8> = vec![0x12, 0x00, 0x32, 0x03, 1, 2, 3];
        let annexb = build_from_temporal_units(std::slice::from_ref(&tu)).unwrap();
        for cut in 1..annexb.len() {
            assert!(
                split_temporal_units(&annexb[..cut]).is_err(),
                "cut at {cut} must not parse"
            );
        }
    }
}
