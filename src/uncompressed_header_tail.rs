//! Sub-syntax functions called from the tail of
//! `uncompressed_header()` (§5.9.2) — landed in round 5 of the
//! clean-room rebuild. None of them are wired into the streaming
//! [`crate::parse_frame_header`] walk yet (the intervening
//! `allow_intrabc` / `disable_frame_end_update_cdf` / `tile_info()` /
//! `segmentation_params()` / `delta_q_params()` / `delta_lf_params()`
//! syntax sits between the round-4 stop point and these calls); they
//! are instead exposed as standalone parser entry points that operate
//! on a byte slice. The next round can stitch them into the streaming
//! parser as the intervening syntaxes land.
//!
//! ## Syntax / semantics references (all in `docs/video/av1/`)
//!
//!   * §5.9.10 — `read_interpolation_filter()`
//!   * §5.9.11 — `loop_filter_params()`
//!   * §5.9.12 — `quantization_params()`
//!   * §5.9.13 — `read_delta_q()` (helper for §5.9.12)
//!   * §6.8.9  — Interpolation filter semantics
//!   * §6.8.10 — Loop filter semantics
//!   * §6.8.11 — Quantization params semantics
//!   * §6.8.12 — Delta quantizer semantics
//!   * §4.10.6 — `su(n)` signed-integer descriptor (used for
//!     `loop_filter_ref_deltas` / `loop_filter_mode_deltas` and the
//!     `delta_q` field of `read_delta_q()`).
//!
//! ## §3 constants referenced here
//!
//!   * `TOTAL_REFS_PER_FRAME = 8` — total number of reference-frame
//!     types including the implicit `INTRA_FRAME`. Used as the loop
//!     bound for the `loop_filter_ref_deltas[i]` update walk inside
//!     `loop_filter_params()`.

use crate::bitreader::BitReader;
use crate::Error;

// ---------------------------------------------------------------------
// §3 constants
// ---------------------------------------------------------------------

/// `TOTAL_REFS_PER_FRAME` per §3 — `INTRA_FRAME` plus the seven
/// inter-prediction reference frame types.
pub const TOTAL_REFS_PER_FRAME: usize = 8;

// ---------------------------------------------------------------------
// §5.9.10 read_interpolation_filter
// ---------------------------------------------------------------------

/// `interpolation_filter` per §6.8.9.
///
/// | code | name                |
/// |-----:|:--------------------|
/// |   0  | `EIGHTTAP`          |
/// |   1  | `EIGHTTAP_SMOOTH`   |
/// |   2  | `EIGHTTAP_SHARP`    |
/// |   3  | `BILINEAR`          |
/// |   4  | `SWITCHABLE`        |
///
/// The `SWITCHABLE` code is **not** encoded as `f(2) == 4` in the
/// bitstream — it's instead signalled by `is_filter_switchable == 1`
/// per §5.9.10, in which case the §5.9.10 syntax skips the `f(2)`
/// `interpolation_filter` field entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationFilter {
    Eighttap,
    EighttapSmooth,
    EighttapSharp,
    Bilinear,
    Switchable,
}

impl InterpolationFilter {
    /// Decode the raw `f(2)` `interpolation_filter` value (only valid
    /// for codes 0..=3 — `SWITCHABLE` is never written as `f(2)` in
    /// the bitstream).
    pub fn from_raw(raw: u8) -> Self {
        match raw & 0x3 {
            0 => Self::Eighttap,
            1 => Self::EighttapSmooth,
            2 => Self::EighttapSharp,
            _ => Self::Bilinear,
        }
    }

    /// Numeric code from §6.8.9.
    pub fn as_raw(&self) -> u8 {
        match self {
            Self::Eighttap => 0,
            Self::EighttapSmooth => 1,
            Self::EighttapSharp => 2,
            Self::Bilinear => 3,
            Self::Switchable => 4,
        }
    }

    /// Whether this filter is the §5.9.10 `is_filter_switchable == 1`
    /// catch-all.
    pub fn is_switchable(&self) -> bool {
        matches!(self, Self::Switchable)
    }
}

/// Read `read_interpolation_filter()` (§5.9.10) from `payload`.
///
/// Returns the chosen [`InterpolationFilter`] plus the number of bits
/// consumed. The caller is responsible for positioning `payload` at
/// the right bit before invoking — the standalone entry points in
/// this round always start at bit 0.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`] — payload exhausted mid-read.
pub fn parse_interpolation_filter(payload: &[u8]) -> Result<(InterpolationFilter, usize), Error> {
    let mut br = BitReader::new(payload);
    let filt = read_interpolation_filter(&mut br)?;
    Ok((filt, br.position()))
}

pub(crate) fn read_interpolation_filter(
    br: &mut BitReader<'_>,
) -> Result<InterpolationFilter, Error> {
    let is_filter_switchable = br.f(1)? == 1;
    if is_filter_switchable {
        Ok(InterpolationFilter::Switchable)
    } else {
        let raw = br.f(2)? as u8;
        Ok(InterpolationFilter::from_raw(raw))
    }
}

// ---------------------------------------------------------------------
// §5.9.11 loop_filter_params
// ---------------------------------------------------------------------

/// Defaults applied when the §5.9.11 `(CodedLossless || allow_intrabc)`
/// short-circuit fires. Mirrors the spec's literal initialisers per
/// `loop_filter_ref_deltas[ INTRA_FRAME ] = 1; loop_filter_ref_deltas[
/// LAST_FRAME ] = 0; ...` block. INTRA_FRAME is index 0,
/// LAST_FRAME..ALTREF2_FRAME are indices 1..=7 per the §3 enumeration
/// (the spec lists LAST=1, LAST2=2, LAST3=3, GOLDEN=4, BWDREF=5,
/// ALTREF2=6, ALTREF=7 ordering — but only the values matter here,
/// indexed by the §3 numbering).
pub const LOOP_FILTER_REF_DELTAS_DEFAULT: [i8; TOTAL_REFS_PER_FRAME] = {
    // Spec defaults per §5.9.11 short-circuit path:
    //   INTRA_FRAME    = 1
    //   LAST_FRAME     = 0
    //   LAST2_FRAME    = 0
    //   LAST3_FRAME    = 0
    //   GOLDEN_FRAME   = -1
    //   BWDREF_FRAME   = 0
    //   ALTREF2_FRAME  = -1
    //   ALTREF_FRAME   = -1
    //
    // Per §3's numbering of the inter-prediction enums:
    //   INTRA=0, LAST=1, LAST2=2, LAST3=3, GOLDEN=4, BWDREF=5,
    //   ALTREF2=6, ALTREF=7.
    [1, 0, 0, 0, -1, 0, -1, -1]
};

/// Defaults for `loop_filter_mode_deltas[]` per §5.9.11 short-circuit.
pub const LOOP_FILTER_MODE_DELTAS_DEFAULT: [i8; 2] = [0, 0];

/// Parsed `loop_filter_params()` per §5.9.11.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopFilterParams {
    /// `loop_filter_level[0..=3]`. Indices 2 and 3 are populated only
    /// when `NumPlanes > 1` and at least one of `loop_filter_level[0]`
    /// / `loop_filter_level[1]` is non-zero; otherwise the spec leaves
    /// them at the default 0.
    pub loop_filter_level: [u8; 4],
    /// `loop_filter_sharpness` (`f(3)`). 0 when the §5.9.11
    /// short-circuit fires.
    pub loop_filter_sharpness: u8,
    /// `loop_filter_delta_enabled` (`f(1)`). 0 when the §5.9.11
    /// short-circuit fires.
    pub loop_filter_delta_enabled: bool,
    /// `loop_filter_delta_update` (`f(1)`). Only meaningful when
    /// `loop_filter_delta_enabled == 1`. 0 otherwise.
    pub loop_filter_delta_update: bool,
    /// `loop_filter_ref_deltas[0..=7]` after the §5.9.11 update walk.
    /// In the streaming parser these values "maintain previous value"
    /// (§6.8.10) when `update_ref_delta == 0`; the standalone parser
    /// returns the spec's "no previous value" defaults
    /// ([`LOOP_FILTER_REF_DELTAS_DEFAULT`]) for slots that weren't
    /// updated this frame.
    pub loop_filter_ref_deltas: [i8; TOTAL_REFS_PER_FRAME],
    /// `loop_filter_mode_deltas[0..=1]` after the §5.9.11 update walk.
    /// Same "no previous value" caveat as
    /// [`Self::loop_filter_ref_deltas`].
    pub loop_filter_mode_deltas: [i8; 2],
    /// Whether the §5.9.11 short-circuit (`CodedLossless ||
    /// allow_intrabc`) fired and the parser returned without reading
    /// any bits.
    pub short_circuited: bool,
}

impl LoopFilterParams {
    /// The §5.9.11 short-circuit-path output (no bits consumed).
    pub const fn short_circuit() -> Self {
        Self {
            loop_filter_level: [0; 4],
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: false,
            loop_filter_delta_update: false,
            loop_filter_ref_deltas: LOOP_FILTER_REF_DELTAS_DEFAULT,
            loop_filter_mode_deltas: LOOP_FILTER_MODE_DELTAS_DEFAULT,
            short_circuited: true,
        }
    }
}

/// Parse `loop_filter_params()` per §5.9.11.
///
/// `num_planes` is §5.5.2's `NumPlanes` derived value (1 for
/// monochrome, otherwise 3). `coded_lossless` and `allow_intrabc` are
/// the runtime-derived flags from §5.9.2 that select the
/// short-circuit path; standalone callers that haven't run the
/// preceding `quantization_params()` + intrabc reads can pass
/// `coded_lossless = false, allow_intrabc = false` to exercise the
/// full bitstream path.
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_loop_filter_params(
    payload: &[u8],
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<(LoopFilterParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let lf = read_loop_filter_params(&mut br, num_planes, coded_lossless, allow_intrabc)?;
    Ok((lf, br.position()))
}

pub(crate) fn read_loop_filter_params(
    br: &mut BitReader<'_>,
    num_planes: u8,
    coded_lossless: bool,
    allow_intrabc: bool,
) -> Result<LoopFilterParams, Error> {
    if coded_lossless || allow_intrabc {
        return Ok(LoopFilterParams::short_circuit());
    }

    let mut loop_filter_level = [0u8; 4];
    loop_filter_level[0] = br.f(6)? as u8;
    loop_filter_level[1] = br.f(6)? as u8;
    if num_planes > 1 && (loop_filter_level[0] != 0 || loop_filter_level[1] != 0) {
        loop_filter_level[2] = br.f(6)? as u8;
        loop_filter_level[3] = br.f(6)? as u8;
    }

    let loop_filter_sharpness = br.f(3)? as u8;
    let loop_filter_delta_enabled = br.f(1)? == 1;

    let mut loop_filter_ref_deltas = LOOP_FILTER_REF_DELTAS_DEFAULT;
    let mut loop_filter_mode_deltas = LOOP_FILTER_MODE_DELTAS_DEFAULT;
    let mut loop_filter_delta_update = false;

    if loop_filter_delta_enabled {
        loop_filter_delta_update = br.f(1)? == 1;
        if loop_filter_delta_update {
            for slot in loop_filter_ref_deltas.iter_mut() {
                let update_ref_delta = br.f(1)? == 1;
                if update_ref_delta {
                    *slot = br.su(7)? as i8;
                }
            }
            for slot in loop_filter_mode_deltas.iter_mut() {
                let update_mode_delta = br.f(1)? == 1;
                if update_mode_delta {
                    *slot = br.su(7)? as i8;
                }
            }
        }
    }

    Ok(LoopFilterParams {
        loop_filter_level,
        loop_filter_sharpness,
        loop_filter_delta_enabled,
        loop_filter_delta_update,
        loop_filter_ref_deltas,
        loop_filter_mode_deltas,
        short_circuited: false,
    })
}

// ---------------------------------------------------------------------
// §5.9.12 quantization_params + §5.9.13 read_delta_q
// ---------------------------------------------------------------------

/// `quantization_params()` per §5.9.12 + §6.8.11.
///
/// `base_q_idx` is the unsigned 8-bit base Y AC qindex. The four
/// `delta_q_*` fields hold the per-plane DC / AC offsets relative to
/// `base_q_idx`, decoded through §5.9.13 `read_delta_q()`. For
/// `num_planes == 1` (monochrome) the U/V deltas remain at 0 per
/// §5.9.12 line `DeltaQUDc = 0; DeltaQUAc = 0; DeltaQVDc = 0;
/// DeltaQVAc = 0`. For `num_planes > 1` with
/// `separate_uv_delta_q == 0`, `delta_q_v_*` mirror their
/// `delta_q_u_*` counterparts per §5.9.12.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationParams {
    /// `base_q_idx` (`f(8)`).
    pub base_q_idx: u8,
    /// `DeltaQYDc` — Y DC offset relative to `base_q_idx`.
    pub delta_q_y_dc: i8,
    /// `diff_uv_delta` — read only when `NumPlanes > 1 &&
    /// separate_uv_delta_q == 1`, otherwise 0.
    pub diff_uv_delta: bool,
    /// `DeltaQUDc` — U DC offset.
    pub delta_q_u_dc: i8,
    /// `DeltaQUAc` — U AC offset.
    pub delta_q_u_ac: i8,
    /// `DeltaQVDc` — V DC offset (mirrors `delta_q_u_dc` when
    /// `diff_uv_delta == 0`).
    pub delta_q_v_dc: i8,
    /// `DeltaQVAc` — V AC offset.
    pub delta_q_v_ac: i8,
    /// `using_qmatrix` (`f(1)`).
    pub using_qmatrix: bool,
    /// `qm_y` (`f(4)`) — only meaningful when `using_qmatrix == 1`.
    pub qm_y: u8,
    /// `qm_u` (`f(4)`).
    pub qm_u: u8,
    /// `qm_v` (`f(4)`) — mirrors `qm_u` when `separate_uv_delta_q == 0`.
    pub qm_v: u8,
}

/// Parse `quantization_params()` per §5.9.12.
///
/// `num_planes` is §5.5.2's `NumPlanes`. `separate_uv_delta_q` is
/// §5.5.2's `separate_uv_delta_q` — both are surfaced on
/// [`crate::SequenceHeader::color_config`].
///
/// ## Errors
///   * [`Error::UnexpectedEnd`]
pub fn parse_quantization_params(
    payload: &[u8],
    num_planes: u8,
    separate_uv_delta_q: bool,
) -> Result<(QuantizationParams, usize), Error> {
    let mut br = BitReader::new(payload);
    let q = read_quantization_params(&mut br, num_planes, separate_uv_delta_q)?;
    Ok((q, br.position()))
}

pub(crate) fn read_quantization_params(
    br: &mut BitReader<'_>,
    num_planes: u8,
    separate_uv_delta_q: bool,
) -> Result<QuantizationParams, Error> {
    let base_q_idx = br.f(8)? as u8;
    let delta_q_y_dc = read_delta_q(br)?;

    let mut diff_uv_delta = false;
    let mut delta_q_u_dc = 0i8;
    let mut delta_q_u_ac = 0i8;
    let mut delta_q_v_dc = 0i8;
    let mut delta_q_v_ac = 0i8;

    if num_planes > 1 {
        if separate_uv_delta_q {
            diff_uv_delta = br.f(1)? == 1;
        }
        delta_q_u_dc = read_delta_q(br)?;
        delta_q_u_ac = read_delta_q(br)?;
        if diff_uv_delta {
            delta_q_v_dc = read_delta_q(br)?;
            delta_q_v_ac = read_delta_q(br)?;
        } else {
            delta_q_v_dc = delta_q_u_dc;
            delta_q_v_ac = delta_q_u_ac;
        }
    }

    let using_qmatrix = br.f(1)? == 1;
    let (qm_y, qm_u, qm_v) = if using_qmatrix {
        let qm_y = br.f(4)? as u8;
        let qm_u = br.f(4)? as u8;
        let qm_v = if separate_uv_delta_q {
            br.f(4)? as u8
        } else {
            qm_u
        };
        (qm_y, qm_u, qm_v)
    } else {
        (0, 0, 0)
    };

    Ok(QuantizationParams {
        base_q_idx,
        delta_q_y_dc,
        diff_uv_delta,
        delta_q_u_dc,
        delta_q_u_ac,
        delta_q_v_dc,
        delta_q_v_ac,
        using_qmatrix,
        qm_y,
        qm_u,
        qm_v,
    })
}

/// `read_delta_q()` per §5.9.13: a `delta_coded == 1` prefix bit gates
/// a `su(1+6) = su(7)` signed offset; absent ⇒ `delta_q = 0`.
fn read_delta_q(br: &mut BitReader<'_>) -> Result<i8, Error> {
    let delta_coded = br.f(1)? == 1;
    if delta_coded {
        Ok(br.su(7)? as i8)
    } else {
        Ok(0)
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // §5.9.10 read_interpolation_filter
    // -----------------------------------------------------------------

    #[test]
    fn interpolation_filter_switchable_one_bit() {
        // is_filter_switchable = 1 ⇒ Switchable, 1 bit consumed.
        let payload = [0b1000_0000u8];
        let (filt, n) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::Switchable);
        assert!(filt.is_switchable());
        assert_eq!(n, 1);
    }

    #[test]
    fn interpolation_filter_eighttap_three_bits() {
        // is_filter_switchable = 0, interpolation_filter f(2) = 00 ⇒
        // EIGHTTAP, 3 bits consumed.
        let payload = [0b0000_0000u8];
        let (filt, n) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::Eighttap);
        assert_eq!(n, 3);
    }

    #[test]
    fn interpolation_filter_eighttap_smooth() {
        // is_filter_switchable = 0, f(2) = 01 ⇒ EIGHTTAP_SMOOTH.
        // bits = 0 01 = 001 = 0010_0000 = 0x20.
        let payload = [0b0010_0000u8];
        let (filt, n) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::EighttapSmooth);
        assert_eq!(n, 3);
    }

    #[test]
    fn interpolation_filter_eighttap_sharp() {
        // is_filter_switchable = 0, f(2) = 10 ⇒ EIGHTTAP_SHARP.
        // bits = 0 10 = 010 = 0100_0000 = 0x40.
        let payload = [0b0100_0000u8];
        let (filt, _) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::EighttapSharp);
    }

    #[test]
    fn interpolation_filter_bilinear() {
        // is_filter_switchable = 0, f(2) = 11 ⇒ BILINEAR.
        // bits = 0 11 = 011 = 0110_0000 = 0x60.
        let payload = [0b0110_0000u8];
        let (filt, _) = parse_interpolation_filter(&payload).expect("decodes");
        assert_eq!(filt, InterpolationFilter::Bilinear);
    }

    #[test]
    fn interpolation_filter_raw_round_trip() {
        for raw in 0u8..4u8 {
            let f = InterpolationFilter::from_raw(raw);
            assert_eq!(f.as_raw(), raw);
        }
        assert_eq!(InterpolationFilter::Switchable.as_raw(), 4);
    }

    #[test]
    fn interpolation_filter_unexpected_end() {
        let payload: [u8; 0] = [];
        let err = parse_interpolation_filter(&payload).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }

    // -----------------------------------------------------------------
    // §5.9.11 loop_filter_params
    // -----------------------------------------------------------------

    #[test]
    fn loop_filter_short_circuit_coded_lossless() {
        // CodedLossless = true ⇒ short-circuit, no bits read.
        let payload: [u8; 0] = [];
        let (lf, n) = parse_loop_filter_params(&payload, 3, true, false).expect("decodes");
        assert!(lf.short_circuited);
        assert_eq!(n, 0);
        assert_eq!(lf.loop_filter_level, [0; 4]);
        assert_eq!(lf.loop_filter_sharpness, 0);
        assert!(!lf.loop_filter_delta_enabled);
        assert_eq!(lf.loop_filter_ref_deltas, LOOP_FILTER_REF_DELTAS_DEFAULT);
        assert_eq!(lf.loop_filter_mode_deltas, LOOP_FILTER_MODE_DELTAS_DEFAULT);
        // INTRA_FRAME default per §5.9.11 short-circuit is 1.
        assert_eq!(lf.loop_filter_ref_deltas[0], 1);
        // GOLDEN_FRAME / ALTREF2_FRAME / ALTREF_FRAME defaults = -1.
        assert_eq!(lf.loop_filter_ref_deltas[4], -1);
        assert_eq!(lf.loop_filter_ref_deltas[6], -1);
        assert_eq!(lf.loop_filter_ref_deltas[7], -1);
    }

    #[test]
    fn loop_filter_short_circuit_allow_intrabc() {
        let payload: [u8; 0] = [];
        let (lf, n) = parse_loop_filter_params(&payload, 3, false, true).expect("decodes");
        assert!(lf.short_circuited);
        assert_eq!(n, 0);
    }

    #[test]
    fn loop_filter_full_path_levels_only() {
        // num_planes=3, neither short-circuit. Bits:
        //   loop_filter_level[0] f(6) = 0
        //   loop_filter_level[1] f(6) = 0   (so 2/3 are NOT read)
        //   loop_filter_sharpness f(3) = 0
        //   loop_filter_delta_enabled f(1) = 0
        // Total = 6 + 6 + 3 + 1 = 16 bits, all zero.
        let payload = [0u8; 2];
        let (lf, n) = parse_loop_filter_params(&payload, 3, false, false).expect("decodes");
        assert!(!lf.short_circuited);
        assert_eq!(lf.loop_filter_level, [0; 4]);
        assert_eq!(lf.loop_filter_sharpness, 0);
        assert!(!lf.loop_filter_delta_enabled);
        assert!(!lf.loop_filter_delta_update);
        assert_eq!(n, 16);
    }

    #[test]
    fn loop_filter_levels_with_planes() {
        // Construct loop_filter_level[0]=42, level[1]=17 — both
        // non-zero so num_planes=3 path reads two more 6-bit values.
        // We pack:
        //   level[0]=42 (6 bits) = 101010
        //   level[1]=17 (6 bits) = 010001
        //   level[2]=3  (6 bits) = 000011
        //   level[3]=4  (6 bits) = 000100
        //   sharpness=5 (3 bits) = 101
        //   delta_enabled=0 (1 bit)
        // Total = 6+6+6+6+3+1 = 28 bits.
        let bits: &[u8] = &[
            1, 0, 1, 0, 1, 0, // level[0] = 42
            0, 1, 0, 0, 0, 1, // level[1] = 17
            0, 0, 0, 0, 1, 1, // level[2] = 3
            0, 0, 0, 1, 0, 0, // level[3] = 4
            1, 0, 1, // sharpness = 5
            0, // delta_enabled = 0
        ];
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (lf, n) = parse_loop_filter_params(&payload, 3, false, false).expect("decodes");
        assert_eq!(lf.loop_filter_level, [42, 17, 3, 4]);
        assert_eq!(lf.loop_filter_sharpness, 5);
        assert!(!lf.loop_filter_delta_enabled);
        assert_eq!(n, bits.len());
    }

    #[test]
    fn loop_filter_mono_skips_plane2_plane3() {
        // num_planes=1, levels non-zero, but the `NumPlanes > 1` gate
        // suppresses level[2]/level[3] reads.
        //   level[0]=10 (6) = 001010
        //   level[1]=20 (6) = 010100
        //   sharpness=0 (3) = 000
        //   delta_enabled=0 (1) = 0
        // Total = 6+6+3+1 = 16 bits.
        let bits: &[u8] = &[
            0, 0, 1, 0, 1, 0, // 10
            0, 1, 0, 1, 0, 0, // 20
            0, 0, 0, // sharpness
            0, // delta_enabled
        ];
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (lf, n) = parse_loop_filter_params(&payload, 1, false, false).expect("decodes");
        assert_eq!(lf.loop_filter_level, [10, 20, 0, 0]);
        assert_eq!(n, 16);
    }

    #[test]
    fn loop_filter_delta_update_walks_refs_and_modes() {
        // num_planes=1, level[0]=level[1]=0 to avoid the plane-2/3
        // reads. Then delta_enabled=1 / delta_update=1 with a sparse
        // pattern: only loop_filter_ref_deltas[0] gets updated to -3,
        // and only loop_filter_mode_deltas[1] gets updated to 5.
        let mut bits: Vec<u8> = vec![
            0, 0, 0, 0, 0, 0, // level[0] = 0
            0, 0, 0, 0, 0, 0, // level[1] = 0
            0, 0, 0, // sharpness = 0
            1, // delta_enabled
            1, // delta_update
        ];
        // 8 ref slots — first one updated to -3 (su(7) = -3 in 7-bit
        // two's complement: -3 = 0b1111101 = 0x7D).
        bits.push(1); // update_ref_delta[0]
        for i in (0..7).rev() {
            bits.push(((0x7Du32 >> i) & 1) as u8); // -3
        }
        bits.resize(bits.len() + (TOTAL_REFS_PER_FRAME - 1), 0);
        // 2 mode slots — second one updated to 5.
        bits.push(0); // update_mode_delta[0]
        bits.push(1); // update_mode_delta[1]
        for i in (0..7).rev() {
            bits.push(((5u32 >> i) & 1) as u8); // 5
        }
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (lf, n) = parse_loop_filter_params(&payload, 1, false, false).expect("decodes");
        assert!(lf.loop_filter_delta_enabled);
        assert!(lf.loop_filter_delta_update);
        assert_eq!(lf.loop_filter_ref_deltas[0], -3);
        // Slots 1..=7 were not updated, but the standalone parser
        // surfaces the §5.9.11 short-circuit defaults for any slot it
        // didn't read. The spec says "maintain previous value" — which
        // is "loaded from a previous frame's state" in the streaming
        // decoder. For a standalone trace we report the defaults, but
        // we do NOT assert on slots[1..] here because they aren't
        // bitstream-visible in this synthetic trace.
        assert_eq!(lf.loop_filter_mode_deltas[1], 5);
        assert_eq!(n, bits.len());
    }

    // -----------------------------------------------------------------
    // §5.9.12 quantization_params
    // -----------------------------------------------------------------

    #[test]
    fn quantization_mono_no_qm() {
        // num_planes=1, separate_uv_delta_q=false.
        //   base_q_idx f(8) = 64 (0b0100_0000)
        //   read_delta_q for DeltaQYDc:
        //     delta_coded f(1) = 0 ⇒ delta_q = 0
        //   (mono: U/V deltas skipped)
        //   using_qmatrix f(1) = 0
        // Total = 8 + 1 + 1 = 10 bits = `0100_0000 0 0` packed:
        // bytes = 0100_0000 0000_0000 → 0x40 0x00.
        let payload = [0x40u8, 0x00];
        let (q, n) = parse_quantization_params(&payload, 1, false).expect("decodes");
        assert_eq!(q.base_q_idx, 64);
        assert_eq!(q.delta_q_y_dc, 0);
        assert_eq!(q.delta_q_u_dc, 0);
        assert_eq!(q.delta_q_u_ac, 0);
        assert_eq!(q.delta_q_v_dc, 0);
        assert_eq!(q.delta_q_v_ac, 0);
        assert!(!q.using_qmatrix);
        assert!(!q.diff_uv_delta);
        assert_eq!(n, 10);
    }

    #[test]
    fn quantization_3plane_no_separate_uv() {
        // num_planes=3, separate_uv_delta_q=false:
        //   base_q_idx f(8) = 128 = 1000_0000
        //   DeltaQYDc: delta_coded=1, su(7)= -4 (0b1111100 = 0x7C)
        //   (separate_uv_delta_q=false ⇒ no diff_uv_delta bit)
        //   DeltaQUDc: delta_coded=0
        //   DeltaQUAc: delta_coded=1, su(7)= +3 (0b000_0011)
        //   diff_uv_delta=false ⇒ V mirrors U
        //   using_qmatrix=0
        let bits: Vec<u8> = {
            let mut b = vec![];
            // base_q_idx = 128 = 1000_0000
            for i in (0..8).rev() {
                b.push(((128u32 >> i) & 1) as u8);
            }
            // DeltaQYDc: delta_coded=1, value=-4 (0b1111100)
            b.push(1);
            for i in (0..7).rev() {
                b.push(((0x7Cu32 >> i) & 1) as u8);
            }
            // DeltaQUDc: delta_coded=0
            b.push(0);
            // DeltaQUAc: delta_coded=1, value=+3 (0b0000011)
            b.push(1);
            for i in (0..7).rev() {
                b.push(((3u32 >> i) & 1) as u8);
            }
            // using_qmatrix = 0
            b.push(0);
            b
        };
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (q, n) = parse_quantization_params(&payload, 3, false).expect("decodes");
        assert_eq!(q.base_q_idx, 128);
        assert_eq!(q.delta_q_y_dc, -4);
        assert!(!q.diff_uv_delta);
        assert_eq!(q.delta_q_u_dc, 0);
        assert_eq!(q.delta_q_u_ac, 3);
        // V mirrors U.
        assert_eq!(q.delta_q_v_dc, q.delta_q_u_dc);
        assert_eq!(q.delta_q_v_ac, q.delta_q_u_ac);
        assert!(!q.using_qmatrix);
        assert_eq!(n, bits.len());
    }

    #[test]
    fn quantization_separate_uv_with_diff() {
        // num_planes=3, separate_uv_delta_q=true:
        //   base_q_idx = 100 = 0b0110_0100
        //   DeltaQYDc: delta_coded=0
        //   diff_uv_delta=1
        //   DeltaQUDc: delta_coded=1 value=+1 (0b0000_001)
        //   DeltaQUAc: delta_coded=0
        //   DeltaQVDc: delta_coded=1 value=-1 (0b1111_111)
        //   DeltaQVAc: delta_coded=0
        //   using_qmatrix=1
        //   qm_y=3 (0b0011)
        //   qm_u=4 (0b0100)
        //   (separate_uv_delta_q=true ⇒ qm_v read)
        //   qm_v=5 (0b0101)
        let mut bits = vec![];
        for i in (0..8).rev() {
            bits.push(((100u32 >> i) & 1) as u8);
        }
        bits.push(0); // delta_coded for DeltaQYDc
        bits.push(1); // diff_uv_delta
        bits.push(1); // delta_coded
        for i in (0..7).rev() {
            bits.push(((1u32 >> i) & 1) as u8);
        }
        bits.push(0); // delta_coded for DeltaQUAc
        bits.push(1); // delta_coded
        for i in (0..7).rev() {
            bits.push(((0x7Fu32 >> i) & 1) as u8); // -1
        }
        bits.push(0); // delta_coded for DeltaQVAc
        bits.push(1); // using_qmatrix
        for i in (0..4).rev() {
            bits.push(((3u32 >> i) & 1) as u8); // qm_y
        }
        for i in (0..4).rev() {
            bits.push(((4u32 >> i) & 1) as u8); // qm_u
        }
        for i in (0..4).rev() {
            bits.push(((5u32 >> i) & 1) as u8); // qm_v
        }
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (q, n) = parse_quantization_params(&payload, 3, true).expect("decodes");
        assert_eq!(q.base_q_idx, 100);
        assert_eq!(q.delta_q_y_dc, 0);
        assert!(q.diff_uv_delta);
        assert_eq!(q.delta_q_u_dc, 1);
        assert_eq!(q.delta_q_u_ac, 0);
        assert_eq!(q.delta_q_v_dc, -1);
        assert_eq!(q.delta_q_v_ac, 0);
        assert!(q.using_qmatrix);
        assert_eq!(q.qm_y, 3);
        assert_eq!(q.qm_u, 4);
        assert_eq!(q.qm_v, 5);
        assert_eq!(n, bits.len());
    }

    #[test]
    fn quantization_using_qmatrix_without_separate_uv() {
        // num_planes=3, separate_uv_delta_q=false, using_qmatrix=1
        // ⇒ qm_v = qm_u (no separate read).
        //   base_q_idx = 0
        //   DeltaQYDc: delta_coded=0
        //   (no diff_uv_delta — gated by separate_uv_delta_q)
        //   DeltaQUDc: delta_coded=0
        //   DeltaQUAc: delta_coded=0
        //   (V mirrors U)
        //   using_qmatrix=1
        //   qm_y=7 (0b0111)
        //   qm_u=9 (0b1001)
        // Bits = 8 + 1 + 1 + 1 + 1 + 4 + 4 = 20.
        let mut bits = vec![0u8; 8]; // base_q_idx = 0
        bits.push(0); // DeltaQYDc delta_coded
        bits.push(0); // DeltaQUDc delta_coded
        bits.push(0); // DeltaQUAc delta_coded
        bits.push(1); // using_qmatrix
        for i in (0..4).rev() {
            bits.push(((7u32 >> i) & 1) as u8); // qm_y
        }
        for i in (0..4).rev() {
            bits.push(((9u32 >> i) & 1) as u8); // qm_u
        }
        let mut payload = vec![0u8; bits.len().div_ceil(8)];
        for (i, &b) in bits.iter().enumerate() {
            if b != 0 {
                payload[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        let (q, _) = parse_quantization_params(&payload, 3, false).expect("decodes");
        assert!(q.using_qmatrix);
        assert_eq!(q.qm_y, 7);
        assert_eq!(q.qm_u, 9);
        assert_eq!(q.qm_v, 9, "qm_v mirrors qm_u when !separate_uv_delta_q");
    }

    #[test]
    fn quantization_unexpected_end() {
        let payload: [u8; 0] = [];
        let err = parse_quantization_params(&payload, 1, false).expect_err("must err");
        assert_eq!(err, Error::UnexpectedEnd);
    }
}
