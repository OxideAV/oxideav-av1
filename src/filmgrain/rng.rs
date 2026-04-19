//! 16-bit linear-feedback shift register for AV1 film grain synthesis
//! (spec §7.20.2).
//!
//! Ported from `github.com/KarpelesLab/goavif/av1/filmgrain/rng.go`
//! (MIT, KarpelesLab/goavif).
//!
//! The generator XORs taps at bits 0, 1, 3 and 12 and shifts the
//! result into bit 15. The spec seeds the RNG from `grain_seed` plus
//! row/column offsets so that the noise is deterministic for any
//! `(frame, position)` pair.

/// 16-bit LFSR. See module docs.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u16,
}

impl Rng {
    /// Construct an RNG primed with `seed`. A zero seed produces a
    /// trivial cycle; callers should ensure the per-position seed
    /// mixes in non-zero bits from `grain_seed`.
    pub fn new(seed: u16) -> Self {
        Self { state: seed }
    }

    /// Reset the RNG to the given state.
    pub fn seed(&mut self, seed: u16) {
        self.state = seed;
    }

    /// Current LFSR value (without advancing).
    pub fn state(&self) -> u16 {
        self.state
    }

    /// Advance one step and return the new state.
    #[allow(clippy::should_implement_trait)] // Not an Iterator — returns u16 not Option<u16>.
    pub fn next(&mut self) -> u16 {
        // Taps: bits 0, 1, 3, 12.
        let bit = (self.state ^ (self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 12)) & 1;
        self.state = (self.state >> 1) | (bit << 15);
        self.state
    }

    /// Return the high byte of the next state, sign-extended into
    /// `i32` (range `[-128, 127]`). Film grain uses this as a signed
    /// 8-bit noise sample.
    pub fn byte(&mut self) -> i32 {
        let v = self.next();
        (v >> 8) as i8 as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let mut a = Rng::new(0x1234);
        let mut b = Rng::new(0x1234);
        for _ in 0..100 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn advances_state() {
        let mut r = Rng::new(0xACE1);
        let mut prev = r.state();
        let mut moved = false;
        for _ in 0..16 {
            let v = r.next();
            if v != prev {
                moved = true;
            }
            prev = v;
        }
        assert!(moved);
    }

    #[test]
    fn byte_in_signed_range() {
        let mut r = Rng::new(0x4321);
        for _ in 0..1000 {
            let b = r.byte();
            assert!((-128..=127).contains(&b));
        }
    }
}
