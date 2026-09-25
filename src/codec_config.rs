//! The AV1 codec-configuration fields a container carries next to an
//! elementary stream — the `av1C` record every AVIF / HEIF `av01` item
//! and every ISOBMFF `av01` sample entry is required to carry
//! (av1-avif §2.2.1: "the values of the fields in the
//! AV1ItemConfigurationProperty shall match those of the Sequence
//! Header OBU in the AV1 Image Item Data").
//!
//! This module is bitstream-side only: it derives the fields from a
//! parsed [`SequenceHeader`] and (de)serialises the fixed 4-byte
//! record plus the trailing `configOBUs` bytes. It writes no ISOBMFF
//! boxes — the container crates wrap the record into their `av1C`
//! property / box themselves.
//!
//! The record layout (marker / version byte, `seq_profile` +
//! `seq_level_idx_0`, the tier / depth / chroma-layout flag byte, the
//! initial-presentation-delay byte, then `configOBUs`) is the one the
//! AV1 ISOBMFF binding defines; it was cross-checked black-box against
//! the records third-party AVIF producers emit (e.g. `81 00 0c 00` for
//! an 8-bit 4:2:0 Main-profile level-2.0 still).

use crate::sequence_header::SequenceHeader;

/// The `av1C` / `AV1CodecConfigurationRecord` fields (r460).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Av1CodecConfig {
    /// §5.5.1 `seq_profile` (0 Main, 1 High, 2 Professional).
    pub seq_profile: u8,
    /// §5.5.1 `seq_level_idx[ 0 ]` (operating point 0).
    pub seq_level_idx: u8,
    /// §5.5.1 `seq_tier[ 0 ]`.
    pub seq_tier: u8,
    /// §5.5.2 `high_bitdepth`.
    pub high_bitdepth: bool,
    /// §5.5.2 `twelve_bit`.
    pub twelve_bit: bool,
    /// §5.5.2 `mono_chrome`.
    pub monochrome: bool,
    /// §5.5.2 `subsampling_x`.
    pub chroma_subsampling_x: bool,
    /// §5.5.2 `subsampling_y`.
    pub chroma_subsampling_y: bool,
    /// §5.5.2 `chroma_sample_position` (§6.4.2 CSP_* code).
    pub chroma_sample_position: u8,
    /// `initial_presentation_delay_minus_one` when the record signals
    /// one (`initial_presentation_delay_present = 1`); `None`
    /// otherwise. A still picture never signals it.
    pub initial_presentation_delay_minus_one: Option<u8>,
    /// The `configOBUs` tail: zero or more complete OBUs (typically a
    /// Sequence Header OBU and/or Metadata OBUs) in the §5.2
    /// low-overhead format with `obu_has_size_field = 1`. AVIF
    /// recommends leaving this empty for image items (av1-avif
    /// §2.2.1: "Sequence Header OBUs should not be present").
    pub config_obus: Vec<u8>,
}

impl Av1CodecConfig {
    /// Derive the record fields from a parsed sequence header
    /// (operating point 0 for the level / tier pair). `config_obus`
    /// starts empty.
    #[must_use]
    pub fn from_sequence_header(seq: &SequenceHeader) -> Self {
        let op0 = seq.operating_points.first();
        Self {
            seq_profile: seq.seq_profile,
            seq_level_idx: op0.map_or(0, |op| op.seq_level_idx),
            seq_tier: op0.map_or(0, |op| op.seq_tier),
            high_bitdepth: seq.color_config.high_bitdepth,
            twelve_bit: seq.color_config.twelve_bit,
            monochrome: seq.color_config.mono_chrome,
            chroma_subsampling_x: seq.color_config.subsampling_x,
            chroma_subsampling_y: seq.color_config.subsampling_y,
            chroma_sample_position: seq.color_config.chroma_sample_position,
            initial_presentation_delay_minus_one: None,
            config_obus: Vec::new(),
        }
    }

    /// §5.5.2 `BitDepth` implied by the depth flags.
    #[must_use]
    pub fn bit_depth(&self) -> u8 {
        match (self.high_bitdepth, self.twelve_bit) {
            (true, true) => 12,
            (true, false) => 10,
            _ => 8,
        }
    }

    /// Serialise the record: the 4-byte fixed part followed by
    /// `config_obus`.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.config_obus.len());
        out.push(0x80 | 1); // marker = 1, version = 1
        out.push(((self.seq_profile & 7) << 5) | (self.seq_level_idx & 31));
        out.push(
            ((self.seq_tier & 1) << 7)
                | (u8::from(self.high_bitdepth) << 6)
                | (u8::from(self.twelve_bit) << 5)
                | (u8::from(self.monochrome) << 4)
                | (u8::from(self.chroma_subsampling_x) << 3)
                | (u8::from(self.chroma_subsampling_y) << 2)
                | (self.chroma_sample_position & 3),
        );
        out.push(match self.initial_presentation_delay_minus_one {
            Some(d) => 0x10 | (d & 15),
            None => 0,
        });
        out.extend_from_slice(&self.config_obus);
        out
    }

    /// Parse a record (the `av1C` property / box payload). Returns
    /// `None` when the marker / version byte is not the `0x81` a
    /// version-1 record carries or the buffer is shorter than the
    /// fixed part.
    #[must_use]
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 || bytes[0] != 0x81 {
            return None;
        }
        let b1 = bytes[1];
        let b2 = bytes[2];
        let b3 = bytes[3];
        Some(Self {
            seq_profile: b1 >> 5,
            seq_level_idx: b1 & 31,
            seq_tier: b2 >> 7,
            high_bitdepth: (b2 >> 6) & 1 == 1,
            twelve_bit: (b2 >> 5) & 1 == 1,
            monochrome: (b2 >> 4) & 1 == 1,
            chroma_subsampling_x: (b2 >> 3) & 1 == 1,
            chroma_subsampling_y: (b2 >> 2) & 1 == 1,
            chroma_sample_position: b2 & 3,
            initial_presentation_delay_minus_one: if b3 & 0x10 != 0 { Some(b3 & 15) } else { None },
            config_obus: bytes[4..].to_vec(),
        })
    }

    /// Whether the record's fields match `seq` (the av1-avif §2.2.1
    /// "shall match the Sequence Header OBU in the AV1 Image Item
    /// Data" check).
    ///
    /// `chroma_sample_position` is compared only when the header codes
    /// it (4:2:0 — §5.5.2 reads the field under `subsampling_x &&
    /// subsampling_y`); for 4:2:2 / 4:4:4 items third-party producers
    /// write `2` (`CSP_COLOCATED`) into the record while the header
    /// carries no position at all, so the field is ignored there.
    #[must_use]
    pub fn matches_sequence_header(&self, seq: &SequenceHeader) -> bool {
        let mut mine = self.clone();
        mine.config_obus.clear();
        mine.initial_presentation_delay_minus_one = None;
        let mut theirs = Self::from_sequence_header(seq);
        if !(theirs.chroma_subsampling_x && theirs.chroma_subsampling_y) {
            mine.chroma_sample_position = 0;
            theirs.chroma_sample_position = 0;
        }
        mine == theirs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_bytes() {
        let cfg = Av1CodecConfig {
            seq_profile: 2,
            seq_level_idx: 16,
            seq_tier: 1,
            high_bitdepth: true,
            twelve_bit: true,
            monochrome: false,
            chroma_subsampling_x: true,
            chroma_subsampling_y: false,
            chroma_sample_position: 2,
            initial_presentation_delay_minus_one: Some(3),
            config_obus: vec![0x0a, 0x01, 0x00],
        };
        let bytes = cfg.to_bytes();
        assert_eq!(bytes[0], 0x81);
        assert_eq!(Av1CodecConfig::parse(&bytes), Some(cfg));
    }

    #[test]
    fn third_party_main_profile_still_record() {
        // The record a third-party AVIF producer emits for an 8-bit
        // 4:2:0 Main-profile level-2.0 still (`81 00 0c 00`).
        let cfg = Av1CodecConfig::parse(&[0x81, 0x00, 0x0c, 0x00]).expect("valid record");
        assert_eq!(cfg.seq_profile, 0);
        assert_eq!(cfg.seq_level_idx, 0);
        assert!(!cfg.high_bitdepth && !cfg.twelve_bit && !cfg.monochrome);
        assert!(cfg.chroma_subsampling_x && cfg.chroma_subsampling_y);
        assert_eq!(cfg.chroma_sample_position, 0);
        assert_eq!(cfg.initial_presentation_delay_minus_one, None);
        assert!(cfg.config_obus.is_empty());
        assert_eq!(cfg.bit_depth(), 8);
        assert_eq!(cfg.to_bytes(), vec![0x81, 0x00, 0x0c, 0x00]);
    }

    #[test]
    fn non_420_records_ignore_the_uncoded_sample_position() {
        // A 4:4:4 High-profile lossless item from a third-party
        // producer: `81 20 02 00` — the record says CSP_COLOCATED
        // although §5.5.2 codes no position for 4:4:4.
        let cfg = Av1CodecConfig::parse(&[0x81, 0x20, 0x02, 0x00]).expect("valid record");
        let mut seq =
            crate::encoder::build_intra_only_seq_yuv(8, 8, 8, crate::encoder::ChromaFormat::Yuv444)
                .expect("seq");
        seq.operating_points[0].seq_level_idx = 0;
        assert!(cfg.matches_sequence_header(&seq));
        let mut cfg420 = Av1CodecConfig::parse(&[0x81, 0x00, 0x0e, 0x00]).expect("valid record");
        let seq420 =
            crate::encoder::build_intra_only_seq_yuv(8, 8, 8, crate::encoder::ChromaFormat::Yuv420)
                .expect("seq");
        assert!(
            !cfg420.matches_sequence_header(&seq420),
            "4:2:0 compares the position"
        );
        cfg420.chroma_sample_position = 0;
        assert!(cfg420.matches_sequence_header(&seq420));
    }

    #[test]
    fn rejects_short_or_unmarked() {
        assert!(Av1CodecConfig::parse(&[0x81, 0, 0]).is_none());
        assert!(Av1CodecConfig::parse(&[0x01, 0, 0, 0]).is_none());
    }
}
