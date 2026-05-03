//! Forward quantisation helpers for the AV1 encoder.
//!
//! Round 1 uses a single `base_q_idx` with no per-plane / per-segment
//! deltas — this matches the encoder's frame-header writer. The
//! decoder side already maps `base_q_idx → (dc_quant, ac_quant)` via
//! [`crate::quant`]; this module provides the forward path.
//!
//! The forward step is the textbook integer divide:
//!
//! ```text
//! q_coeff = round(coeff / dequant)
//!         = (coeff >= 0)
//!             ? (coeff + dequant / 2) / dequant
//!             : -((-coeff + dequant / 2) / dequant)
//! ```
//!
//! When `dequant == 0` (lossless) the encoder MUST NOT call this
//! helper — pass coefficients through unchanged instead.

use oxideav_core::Result;

use crate::quant::{Params, Plane};

/// Per-plane DC + AC quantisers, derived from the frame-header
/// `base_q_idx` via the spec's lookup tables. Identical to the
/// decoder's [`crate::quant::Values`] but materialised with explicit
/// constructor + accessors so the encoder side reads cleanly.
#[derive(Clone, Copy, Debug)]
pub struct PlaneQuant {
    pub dc: u16,
    pub ac: u16,
}

impl PlaneQuant {
    /// Fetch the DC / AC quantisers for `plane` at frame-level
    /// `base_q_idx` and 8-bit depth (round 1 is profile 0 only).
    pub fn for_round1(base_q_idx: u8, plane: Plane) -> Result<Self> {
        let params = Params {
            base_q_idx: base_q_idx as i32,
            delta_q_y_dc: 0,
            delta_q_u_dc: 0,
            delta_q_u_ac: 0,
            delta_q_v_dc: 0,
            delta_q_v_ac: 0,
            bit_depth: 8,
        };
        let v = params.compute(plane)?;
        Ok(Self { dc: v.dc, ac: v.ac })
    }

    /// Quantise a single DC coefficient.
    pub fn quantise_dc(&self, coeff: i32) -> i32 {
        quantise(coeff, self.dc as i32)
    }

    /// Quantise a single AC coefficient.
    pub fn quantise_ac(&self, coeff: i32) -> i32 {
        quantise(coeff, self.ac as i32)
    }
}

/// Forward quantisation: round-half-away-from-zero divide.
///
/// `dequant == 0` ⇒ identity (the lossless path; callers should never
/// invoke this helper in lossless mode but the safe fallthrough avoids
/// a panic).
fn quantise(coeff: i32, dequant: i32) -> i32 {
    if dequant <= 0 {
        return coeff;
    }
    let half = dequant / 2;
    if coeff >= 0 {
        (coeff + half) / dequant
    } else {
        -((-coeff + half) / dequant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Plane;

    #[test]
    fn for_round1_y_q100_within_table_range() {
        let q = PlaneQuant::for_round1(100, Plane::Y).unwrap();
        // Quantiser tables in §7.12.2 are monotonically increasing
        // with q-index. q=100 lands well inside the dynamic range —
        // both DC and AC dequantisers should be in (0, 1024].
        assert!(q.dc > 0 && q.dc < 1024, "dc = {}", q.dc);
        assert!(q.ac > 0 && q.ac < 1024, "ac = {}", q.ac);
    }

    #[test]
    fn for_round1_y_q0_dequantisers_minimal() {
        let q = PlaneQuant::for_round1(0, Plane::Y).unwrap();
        // q=0 is the lossless / minimal-step quantiser — the table
        // entry is 4 for both DC and AC (§7.12.2).
        assert_eq!(q.dc, 4);
        assert_eq!(q.ac, 4);
    }

    #[test]
    fn quantise_round_half_away_from_zero() {
        assert_eq!(quantise(0, 8), 0);
        assert_eq!(quantise(4, 8), 1);
        assert_eq!(quantise(-4, 8), -1);
        assert_eq!(quantise(7, 8), 1);
        assert_eq!(quantise(-7, 8), -1);
        assert_eq!(quantise(12, 8), 2);
        assert_eq!(quantise(-12, 8), -2);
    }

    #[test]
    fn quantise_zero_dequant_passes_through() {
        // Lossless fallback: helper returns the input unchanged when
        // dequant is 0.
        assert_eq!(quantise(123, 0), 123);
    }

    #[test]
    fn quantise_dc_uses_dc_step() {
        let q = PlaneQuant { dc: 16, ac: 32 };
        assert_eq!(q.quantise_dc(64), 4);
        assert_eq!(q.quantise_ac(64), 2);
    }
}
