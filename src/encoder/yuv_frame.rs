//! General YUV input frame for the conformance-grade encoders (r427).
//!
//! [`YuvFrame`] carries any §6.4.1 (bit depth, chroma format) pairing
//! the three AV1 profiles admit — 8 / 10 / 12-bit samples in 4:2:0,
//! 4:2:2, 4:4:4 or monochrome layout — as `u16` planes. The module
//! also owns the §6.4.1 `seq_profile` election and the §5.5.2
//! `color_config` synthesis that map a pairing onto the wire:
//!
//! | `seq_profile` | Bit depth | Monochrome | Chroma subsampling      |
//! |---------------|-----------|------------|-------------------------|
//! | 0             | 8 or 10   | yes        | YUV 4:2:0               |
//! | 1             | 8 or 10   | no         | YUV 4:4:4               |
//! | 2             | 8 or 10   | yes        | YUV 4:2:2               |
//! | 2             | 12        | yes        | 4:2:0 / 4:2:2 / 4:4:4   |
//!
//! (§5.5.2 semantics, "seq_profile" table; monochrome is only
//! signalable on profiles 0 and 2.)
//!
//! The historical 8-bit 4:2:0 entry points keep their
//! [`Yuv420Frame`]-based signatures; [`YuvFrame::from_yuv420_8bit`]
//! widens those inputs into this representation so the whole encoder
//! pipeline runs on one sample type.

use crate::encoder::pixel_driver_dyn::{build_intra_only_yuv420_8bit_seq, Yuv420Frame};
use crate::sequence_header::{ColorConfig, SequenceHeader, CSP_UNKNOWN};
use crate::Error;

/// §6.4.2 chroma layout selector: the `(subsampling_x, subsampling_y,
/// mono_chrome)` triple every geometry derivation keys off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChromaFormat {
    /// Luma only (`mono_chrome = 1`, `NumPlanes = 1`). §5.5.2 infers
    /// `subsampling_x = subsampling_y = 1` on the mono arm.
    Monochrome,
    /// `subsampling_x = subsampling_y = 1` — chroma at half extent on
    /// both axes.
    Yuv420,
    /// `subsampling_x = 1`, `subsampling_y = 0` — chroma at half
    /// horizontal extent, full vertical extent.
    Yuv422,
    /// `subsampling_x = subsampling_y = 0` — chroma at full extent.
    Yuv444,
}

impl ChromaFormat {
    /// §6.4.2 `(subsampling_x, subsampling_y)` for this layout (the
    /// §5.5.2-inferred `(1, 1)` on the monochrome arm).
    #[must_use]
    pub fn subsampling(self) -> (u8, u8) {
        match self {
            ChromaFormat::Monochrome | ChromaFormat::Yuv420 => (1, 1),
            ChromaFormat::Yuv422 => (1, 0),
            ChromaFormat::Yuv444 => (0, 0),
        }
    }

    /// §5.5.2 `NumPlanes` (`1` for monochrome, `3` otherwise).
    #[must_use]
    pub fn num_planes(self) -> u8 {
        match self {
            ChromaFormat::Monochrome => 1,
            _ => 3,
        }
    }
}

/// Dynamic-extent YUV input at any conformant (bit depth, chroma
/// format) pairing. Samples are `u16` regardless of depth; every
/// sample must lie in `[0, (1 << bit_depth) - 1]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct YuvFrame {
    /// Luma width in pixels (multiple of 8).
    pub width: u32,
    /// Luma height in pixels (multiple of 8).
    pub height: u32,
    /// §5.5.2 `BitDepth`: 8, 10 or 12.
    pub bit_depth: u8,
    /// Chroma layout.
    pub format: ChromaFormat,
    /// Luma plane, row-major, length `width * height`.
    pub y: Vec<u16>,
    /// Cb plane at the subsampled extent ([`Self::chroma_width`] ×
    /// [`Self::chroma_height`]); empty on the monochrome arm.
    pub u: Vec<u16>,
    /// Cr plane; same shape as `u`.
    pub v: Vec<u16>,
}

impl YuvFrame {
    /// All-`fill` constant frame (every plane set to `fill`).
    #[must_use]
    pub fn filled(width: u32, height: u32, bit_depth: u8, format: ChromaFormat, fill: u16) -> Self {
        let (ssx, ssy) = format.subsampling();
        let (cw, ch) = if format == ChromaFormat::Monochrome {
            (0, 0)
        } else {
            (width >> ssx, height >> ssy)
        };
        Self {
            width,
            height,
            bit_depth,
            format,
            y: vec![fill; (width * height) as usize],
            u: vec![fill; (cw * ch) as usize],
            v: vec![fill; (cw * ch) as usize],
        }
    }

    /// Widen an 8-bit 4:2:0 [`Yuv420Frame`] into the general
    /// representation (lossless).
    #[must_use]
    pub fn from_yuv420_8bit(input: &Yuv420Frame) -> Self {
        let widen = |p: &[u8]| p.iter().map(|&s| u16::from(s)).collect::<Vec<u16>>();
        Self {
            width: input.width,
            height: input.height,
            bit_depth: 8,
            format: ChromaFormat::Yuv420,
            y: widen(&input.y),
            u: widen(&input.u),
            v: widen(&input.v),
        }
    }

    /// Chroma plane width in samples (`width >> subsampling_x`; `0`
    /// on the monochrome arm).
    #[must_use]
    pub fn chroma_width(&self) -> u32 {
        if self.format == ChromaFormat::Monochrome {
            0
        } else {
            self.width >> self.format.subsampling().0
        }
    }

    /// Chroma plane height in samples (`height >> subsampling_y`; `0`
    /// on the monochrome arm).
    #[must_use]
    pub fn chroma_height(&self) -> u32 {
        if self.format == ChromaFormat::Monochrome {
            0
        } else {
            self.height >> self.format.subsampling().1
        }
    }

    /// Validate shape + sample range: dimensions multiples of 8 in
    /// `[8, 4096]` per axis, `bit_depth ∈ {8, 10, 12}`, plane lengths
    /// consistent with the format (empty chroma on monochrome), every
    /// sample `< (1 << bit_depth)`.
    ///
    /// ## Errors
    ///
    /// [`Error::PartitionWalkOutOfRange`] on any violation.
    pub fn validate(&self) -> Result<(), Error> {
        if !matches!(self.bit_depth, 8 | 10 | 12) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        if self.width < 8
            || self.height < 8
            || self.width > crate::encoder::key_frame::KEY_FRAME_MAX_DIM
            || self.height > crate::encoder::key_frame::KEY_FRAME_MAX_DIM
            || self.width % 8 != 0
            || self.height % 8 != 0
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let expected_y = (self.width * self.height) as usize;
        let expected_uv = (self.chroma_width() * self.chroma_height()) as usize;
        if self.y.len() != expected_y || self.u.len() != expected_uv || self.v.len() != expected_uv
        {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let ceil = 1u16 << self.bit_depth;
        let in_range = |p: &[u16]| p.iter().all(|&s| s < ceil);
        if !in_range(&self.y) || !in_range(&self.u) || !in_range(&self.v) {
            return Err(Error::PartitionWalkOutOfRange);
        }
        Ok(())
    }
}

/// §6.4.1 `seq_profile` election for a (bit depth, chroma format)
/// pairing (see the module-docs table).
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] when no profile admits the
/// pairing: `bit_depth` outside `{8, 10, 12}`, or 8/10-bit 4:4:4
/// monochrome-style mismatches (profile 1 forbids monochrome — but
/// monochrome never carries the 4:4:4 layout here, so the only
/// invalid inputs are bad depths).
pub fn elect_seq_profile(bit_depth: u8, format: ChromaFormat) -> Result<u8, Error> {
    match (bit_depth, format) {
        (12, _) => Ok(2),
        (8 | 10, ChromaFormat::Yuv422) => Ok(2),
        (8 | 10, ChromaFormat::Yuv444) => Ok(1),
        (8 | 10, ChromaFormat::Yuv420 | ChromaFormat::Monochrome) => Ok(0),
        _ => Err(Error::PartitionWalkOutOfRange),
    }
}

/// §5.5.2 `color_config` synthesis for a (bit depth, chroma format)
/// pairing: unspecified colour description, studio range, `CSP
/// unknown`, `separate_uv_delta_q = 0` — only the depth / layout /
/// plane-count fields vary.
#[must_use]
pub fn color_config_for(bit_depth: u8, format: ChromaFormat) -> ColorConfig {
    let (ssx, ssy) = format.subsampling();
    ColorConfig {
        high_bitdepth: bit_depth >= 10,
        twelve_bit: bit_depth == 12,
        bit_depth,
        mono_chrome: format == ChromaFormat::Monochrome,
        num_planes: format.num_planes(),
        color_description_present_flag: false,
        color_primaries: crate::sequence_header::CP_UNSPECIFIED,
        transfer_characteristics: crate::sequence_header::TC_UNSPECIFIED,
        matrix_coefficients: crate::sequence_header::MC_UNSPECIFIED,
        color_range: false,
        subsampling_x: ssx == 1,
        subsampling_y: ssy == 1,
        chroma_sample_position: CSP_UNKNOWN,
        separate_uv_delta_q: false,
    }
}

/// General-format sibling of
/// [`build_intra_only_yuv420_8bit_seq`]: the same
/// minimal intra-capable `SequenceHeader` shape with `seq_profile` +
/// §5.5.2 `color_config` elected from `(bit_depth, format)`.
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] when [`elect_seq_profile`]
/// rejects the pairing.
pub fn build_intra_only_seq_yuv(
    max_width: u32,
    max_height: u32,
    bit_depth: u8,
    format: ChromaFormat,
) -> Result<SequenceHeader, Error> {
    let mut seq = build_intra_only_yuv420_8bit_seq(max_width, max_height);
    seq.seq_profile = elect_seq_profile(bit_depth, format)?;
    seq.color_config = color_config_for(bit_depth, format);
    Ok(seq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_header::parse_sequence_header;

    #[test]
    fn profile_election_matches_the_6_4_1_table() {
        assert_eq!(elect_seq_profile(8, ChromaFormat::Yuv420).unwrap(), 0);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Yuv420).unwrap(), 0);
        assert_eq!(elect_seq_profile(8, ChromaFormat::Monochrome).unwrap(), 0);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Monochrome).unwrap(), 0);
        assert_eq!(elect_seq_profile(8, ChromaFormat::Yuv444).unwrap(), 1);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Yuv444).unwrap(), 1);
        assert_eq!(elect_seq_profile(8, ChromaFormat::Yuv422).unwrap(), 2);
        assert_eq!(elect_seq_profile(10, ChromaFormat::Yuv422).unwrap(), 2);
        for fmt in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            assert_eq!(elect_seq_profile(12, fmt).unwrap(), 2);
        }
        assert!(elect_seq_profile(9, ChromaFormat::Yuv420).is_err());
        assert!(elect_seq_profile(16, ChromaFormat::Yuv444).is_err());
    }

    /// Every admissible pairing's synthesized sequence header must
    /// round-trip through the §5.5 parser with the exact depth /
    /// subsampling / plane-count fields.
    #[test]
    fn general_seq_builder_round_trips_every_pairing() {
        let pairings: &[(u8, ChromaFormat)] = &[
            (8, ChromaFormat::Yuv420),
            (10, ChromaFormat::Yuv420),
            (12, ChromaFormat::Yuv420),
            (8, ChromaFormat::Yuv422),
            (10, ChromaFormat::Yuv422),
            (12, ChromaFormat::Yuv422),
            (8, ChromaFormat::Yuv444),
            (10, ChromaFormat::Yuv444),
            (12, ChromaFormat::Yuv444),
            (8, ChromaFormat::Monochrome),
            (10, ChromaFormat::Monochrome),
            (12, ChromaFormat::Monochrome),
        ];
        for &(bd, fmt) in pairings {
            let seq = build_intra_only_seq_yuv(128, 96, bd, fmt).unwrap();
            let payload = crate::encoder::sequence_obu::write_sequence_header_obu(&seq);
            let parsed = parse_sequence_header(&payload).unwrap();
            let (ssx, ssy) = fmt.subsampling();
            assert_eq!(parsed.seq_profile, elect_seq_profile(bd, fmt).unwrap());
            assert_eq!(parsed.color_config.bit_depth, bd, "{bd} {fmt:?}");
            assert_eq!(
                parsed.color_config.mono_chrome,
                fmt == ChromaFormat::Monochrome
            );
            assert_eq!(parsed.color_config.num_planes, fmt.num_planes());
            assert_eq!(parsed.color_config.subsampling_x, ssx == 1, "{bd} {fmt:?}");
            assert_eq!(parsed.color_config.subsampling_y, ssy == 1, "{bd} {fmt:?}");
        }
    }

    #[test]
    fn yuv_frame_validation_rejects_bad_shapes() {
        // Good frames validate.
        for fmt in [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
            ChromaFormat::Monochrome,
        ] {
            for bd in [8u8, 10, 12] {
                let f = YuvFrame::filled(64, 32, bd, fmt, (1 << bd) - 1);
                f.validate().unwrap_or_else(|e| {
                    panic!("filled({bd}, {fmt:?}) must validate: {e:?}");
                });
            }
        }
        // Sample over range.
        let mut f = YuvFrame::filled(8, 8, 10, ChromaFormat::Yuv444, 0);
        f.y[3] = 1 << 10;
        assert!(f.validate().is_err());
        // Chroma length mismatch (4:2:2 chroma is width/2 × FULL height).
        let mut f = YuvFrame::filled(16, 16, 8, ChromaFormat::Yuv422, 0);
        assert_eq!(f.u.len(), 8 * 16);
        f.u.truncate(8 * 8);
        assert!(f.validate().is_err());
        // Monochrome must carry empty chroma.
        let mut f = YuvFrame::filled(16, 16, 8, ChromaFormat::Monochrome, 0);
        assert!(f.u.is_empty() && f.v.is_empty());
        f.u = vec![0; 64];
        assert!(f.validate().is_err());
        // Bad depth / bad extent.
        let f = YuvFrame::filled(16, 16, 9, ChromaFormat::Yuv420, 0);
        assert!(f.validate().is_err());
        let f = YuvFrame::filled(12, 16, 8, ChromaFormat::Yuv420, 0);
        assert!(f.validate().is_err());
    }

    #[test]
    fn widening_from_yuv420_8bit_is_lossless() {
        let mut base = Yuv420Frame::filled(16, 8, 0);
        for (i, s) in base.y.iter_mut().enumerate() {
            *s = (i % 251) as u8;
        }
        for (i, s) in base.u.iter_mut().enumerate() {
            *s = (i % 249) as u8;
        }
        for (i, s) in base.v.iter_mut().enumerate() {
            *s = (i % 247) as u8;
        }
        let wide = YuvFrame::from_yuv420_8bit(&base);
        wide.validate().unwrap();
        assert_eq!(wide.bit_depth, 8);
        assert_eq!(wide.format, ChromaFormat::Yuv420);
        assert!(wide.y.iter().zip(&base.y).all(|(&w, &b)| w == u16::from(b)));
        assert!(wide.u.iter().zip(&base.u).all(|(&w, &b)| w == u16::from(b)));
        assert!(wide.v.iter().zip(&base.v).all(|(&w, &b)| w == u16::from(b)));
    }
}
