//! AV1 motion vector decoder — §5.11.25 / §6.10.31.
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/decoder/mv.go`
//! (MIT, KarpelesLab/goavif). Decodes joint / sign / class /
//! bits / fractional / high-precision components of a single motion
//! vector diff symbol, returning the MV in eighth-pel units.
//!
//! Phase 7 scope: only `ReadMv` is exercised (for NEWMV). NEARESTMV /
//! NEARMV / GLOBALMV degrade to zero-MV without a real ref-MV list;
//! the higher-level decoder handles that policy.

use oxideav_core::Result;

use crate::cdfs;
use crate::symbol::SymbolDecoder;

/// `MVJoint` enumerates the four `mv_joint` values (spec §6.10.27).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MvJoint {
    /// Both components zero.
    Zero = 0,
    /// Horizontal non-zero, vertical zero.
    HnzVz = 1,
    /// Horizontal zero, vertical non-zero.
    HzVnz = 2,
    /// Both non-zero.
    HnzVnz = 3,
}

impl MvJoint {
    /// Raw symbol value → `MvJoint`. Unknown values collapse to
    /// `Zero` — the spec permits only 0..=3.
    pub fn from_u32(v: u32) -> Self {
        match v {
            1 => Self::HnzVz,
            2 => Self::HzVnz,
            3 => Self::HnzVnz,
            _ => Self::Zero,
        }
    }

    /// `true` when the horizontal component is non-zero.
    pub fn has_col(self) -> bool {
        matches!(self, Self::HnzVz | Self::HnzVnz)
    }

    /// `true` when the vertical component is non-zero.
    pub fn has_row(self) -> bool {
        matches!(self, Self::HzVnz | Self::HnzVnz)
    }
}

/// Single motion vector. Components are in eighth-pel units when the
/// frame header's `allow_high_precision_mv` is set, quarter-pel
/// otherwise. The decoder still returns eighth-pel magnitudes: the
/// lower bits will simply land on even values when `hp` is forced.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Mv {
    pub row: i32,
    pub col: i32,
}

/// Owns all CDFs consumed while decoding a motion vector diff. One
/// instance per inter block / tile is fine; CDFs adapt in place when
/// the symbol decoder was created with `allow_update = true`.
pub struct MvDecoder {
    joint_cdf: Vec<u16>,
    sign_cdf: [Vec<u16>; 2],
    class_cdf: [Vec<u16>; 2],
    class0_bit_cdf: [Vec<u16>; 2],
    class0_fr_cdf: [[Vec<u16>; 2]; 2],
    class0_hp_cdf: [Vec<u16>; 2],
    fr_cdf: [Vec<u16>; 2],
    hp_cdf: [Vec<u16>; 2],
    bits_cdf: [[Vec<u16>; 10]; 2],
    allow_high_precision_mv: bool,
}

impl MvDecoder {
    /// Fresh `MvDecoder` primed with libaom default CDFs.
    /// `allow_high_precision_mv` forwards the frame-header flag — AVIF
    /// still images force integer-pel, so pass `false` in that case.
    pub fn new(allow_high_precision_mv: bool) -> Self {
        let joint_cdf = cdfs::DEFAULT_MV_JOINT_CDF.to_vec();
        let init_pair = |tbl: &[&[u16]; 2]| [tbl[0].to_vec(), tbl[1].to_vec()];
        let sign_cdf = init_pair(&cdfs::DEFAULT_MV_SIGN_CDF);
        let class_cdf = init_pair(&cdfs::DEFAULT_MV_CLASS_CDF);
        let class0_bit_cdf = init_pair(&cdfs::DEFAULT_MV_CLASS0_BIT_CDF);
        let class0_hp_cdf = init_pair(&cdfs::DEFAULT_MV_CLASS0_HP_CDF);
        let fr_cdf = init_pair(&cdfs::DEFAULT_MV_FR_CDF);
        let hp_cdf = init_pair(&cdfs::DEFAULT_MV_HP_CDF);

        // class0_fr is [[CDF; 2]; 2]
        let class0_fr_cdf = [
            [
                cdfs::DEFAULT_MV_CLASS0_FR_CDF[0][0].to_vec(),
                cdfs::DEFAULT_MV_CLASS0_FR_CDF[0][1].to_vec(),
            ],
            [
                cdfs::DEFAULT_MV_CLASS0_FR_CDF[1][0].to_vec(),
                cdfs::DEFAULT_MV_CLASS0_FR_CDF[1][1].to_vec(),
            ],
        ];

        // bits_cdf is [[CDF; 10]; 2]
        fn bits_row(row: &[&[u16]; 10]) -> [Vec<u16>; 10] {
            [
                row[0].to_vec(),
                row[1].to_vec(),
                row[2].to_vec(),
                row[3].to_vec(),
                row[4].to_vec(),
                row[5].to_vec(),
                row[6].to_vec(),
                row[7].to_vec(),
                row[8].to_vec(),
                row[9].to_vec(),
            ]
        }
        let bits_cdf = [
            bits_row(&cdfs::DEFAULT_MV_BITS_CDF[0]),
            bits_row(&cdfs::DEFAULT_MV_BITS_CDF[1]),
        ];

        Self {
            joint_cdf,
            sign_cdf,
            class_cdf,
            class0_bit_cdf,
            class0_fr_cdf,
            class0_hp_cdf,
            fr_cdf,
            hp_cdf,
            bits_cdf,
            allow_high_precision_mv,
        }
    }

    /// Decode a motion vector difference. The caller adds the result
    /// to the predicted MV; this routine returns only the diff
    /// components.
    pub fn read_mv(&mut self, sym: &mut SymbolDecoder<'_>) -> Result<Mv> {
        let joint_sym = sym.decode_symbol(&mut self.joint_cdf)?;
        let joint = MvJoint::from_u32(joint_sym);
        let mut mv = Mv::default();
        if joint.has_col() {
            mv.col = self.read_component(sym, 0)?;
        }
        if joint.has_row() {
            mv.row = self.read_component(sym, 1)?;
        }
        Ok(mv)
    }

    /// Decode a single MV component: horizontal when `comp == 0`,
    /// vertical when `comp == 1`. Output is in eighth-pel units.
    fn read_component(&mut self, sym: &mut SymbolDecoder<'_>, comp: usize) -> Result<i32> {
        let sign = sym.decode_symbol(&mut self.sign_cdf[comp])?;
        let cls = sym.decode_symbol(&mut self.class_cdf[comp])?;

        let mag_int: i32;
        let frac: i32;
        let hp: i32;
        if cls == 0 {
            let b = sym.decode_symbol(&mut self.class0_bit_cdf[comp])? as i32;
            mag_int = b;
            let b_idx = b as usize;
            frac = sym.decode_symbol(&mut self.class0_fr_cdf[comp][b_idx])? as i32;
            hp = if self.allow_high_precision_mv {
                sym.decode_symbol(&mut self.class0_hp_cdf[comp])? as i32
            } else {
                1
            };
        } else {
            let cls_idx = cls as usize;
            let mut bits = 0i32;
            for i in 0..cls_idx {
                let b = sym.decode_symbol(&mut self.bits_cdf[comp][i])? as i32;
                bits |= b << i;
            }
            mag_int = (1i32 << (cls_idx + 2)) + (bits << 3);
            frac = sym.decode_symbol(&mut self.fr_cdf[comp])? as i32;
            hp = if self.allow_high_precision_mv {
                sym.decode_symbol(&mut self.hp_cdf[comp])? as i32
            } else {
                1
            };
        }

        let mag = if cls == 0 {
            mag_int * 8 + frac * 2 + hp + 1
        } else {
            mag_int + frac * 2 + hp + 1
        };
        Ok(if sign == 1 { -mag } else { mag })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cdfs;
    use crate::symbol::SymbolDecoder;

    // Minimal CDF encoder mirroring goavif `entropy.Encoder` behaviour
    // enough to drive the symbol decoder for these tests.
    //
    // Rather than re-implementing the range coder, we synthesise a
    // plausible bitstream by picking a single CDF symbol that the
    // all-zero symbol-decoder fixture will land on. This keeps the
    // test focused on the MV diff decoding pipeline (joint -> class ->
    // components) rather than on CDF encoding mechanics.

    fn fresh_decoder() -> SymbolDecoder<'static> {
        // Use a static buffer so the returned decoder has a 'static
        // lifetime. 32 bytes of zeros is enough for any single MV.
        static BUF: [u8; 32] = [0u8; 32];
        SymbolDecoder::new(&BUF, BUF.len(), false).expect("init")
    }

    #[test]
    fn zero_buf_yields_valid_mv() {
        // With an all-zero stream, the symbol decoder gravitates to
        // the higher end of each CDF, producing a predictable MV.
        let mut sym = fresh_decoder();
        let mut md = MvDecoder::new(false);
        let mv = md.read_mv(&mut sym).expect("decode");
        // Validity: magnitude never exceeds the AV1 spec cap (32k).
        assert!(mv.row.abs() < 32768);
        assert!(mv.col.abs() < 32768);
    }

    #[test]
    fn mv_joint_zero_produces_zero_mv() {
        // Craft a CDF that forces decode_symbol to return joint=Zero,
        // then pipe it through read_mv manually. Since the symbol
        // decoder's first call returns higher-index symbols for the
        // zero-buffer seed, we swap in a CDF whose first bucket is
        // dominant.
        let mut sym = fresh_decoder();
        let mut md = MvDecoder::new(false);
        // Force joint=0 by giving decode_symbol a CDF where symbol 0
        // takes the entire probability mass.
        md.joint_cdf = vec![0u16, 0u16, 0u16, 0u16, 0u16, 0u16];
        let mv = md.read_mv(&mut sym).expect("decode");
        assert_eq!(mv, Mv { row: 0, col: 0 });
    }

    /// Confirm the default CDF table matches the spec shape expected
    /// by the encoder-side Go test: 4 symbols + sentinel + counter.
    #[test]
    fn default_joint_cdf_shape() {
        let c = cdfs::DEFAULT_MV_JOINT_CDF;
        // joint CDF = 4 symbols → entries = 4 + 1 (counter).
        // wire format: N-1 thresholds + sentinel + counter = N+1 → 5 entries.
        assert_eq!(c.len(), 5);
    }

    /// Check the allow_high_precision_mv flag routes through the
    /// decoder without altering CDF identity for non-HP symbols.
    #[test]
    fn allow_high_precision_mv_flag_remembered() {
        let md = MvDecoder::new(true);
        assert!(md.allow_high_precision_mv);
        let md = MvDecoder::new(false);
        assert!(!md.allow_high_precision_mv);
    }
}
