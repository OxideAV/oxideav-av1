//! AV1 Decoded Picture Buffer (DPB) bookkeeping — §7.20 / §7.21.
//!
//! The bitstream-level DPB holds `NUM_REF_FRAMES = 8` slots; each
//! slot tracks the per-frame state that a subsequent inter frame
//! may consult. For our narrow Phase 7+ decoder we don't materialise
//! the per-slot reconstructed planes (those still live behind the
//! single `Av1Decoder::prev_frame` field), but we DO need the
//! per-slot `OrderHint` so we can derive `skipModeAllowed` /
//! `SkipModeFrame[0..=1]` per §5.9.21 — that derivation is what
//! tells a frame whether the bitstream emits the `skip_mode_present`
//! bit at all (and, if so, which two reference frames a SKIP_MODE
//! block uses for compound prediction).
//!
//! See `frame_header.rs` and `decoder.rs` for the wiring.

use crate::frame_header::NUM_REF_FRAMES;

/// Per-slot DPB state tracked for §5.9.21.
///
/// `OrderHint` follows the spec's `RefOrderHint[]` storage — set when
/// the slot is refreshed by a frame whose `show_frame` (or `showable_
/// frame`) made it visible to consumers. `valid` is `true` iff the
/// slot has ever been refreshed since reset.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefSlot {
    pub order_hint: u32,
    pub valid: bool,
}

/// Decoded Picture Buffer — `NUM_REF_FRAMES` slots, indexed 0..=7.
///
/// Used by the parser to compute `skipModeAllowed` per §5.9.22: the
/// derivation walks `ref_frame_idx[0..REFS_PER_FRAME]` and reads
/// `RefOrderHint[ref_frame_idx[i]]` from these slots.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dpb {
    pub slots: [RefSlot; NUM_REF_FRAMES],
}

impl Dpb {
    /// Empty DPB — every slot starts invalid (no reference frames yet).
    pub const fn new() -> Self {
        Self {
            slots: [RefSlot {
                order_hint: 0,
                valid: false,
            }; NUM_REF_FRAMES],
        }
    }

    /// `true` iff at least one slot has been refreshed since reset
    /// (i.e. the decoder has seen at least one frame whose
    /// `refresh_frame_flags != 0`).
    pub fn any_valid(&self) -> bool {
        self.slots.iter().any(|s| s.valid)
    }

    /// §7.20: refresh selected slots from the just-decoded frame's
    /// `OrderHint`. `mask` is the frame-header `refresh_frame_flags`
    /// 8-bit field — bit `i` set means slot `i` is overwritten.
    pub fn refresh(&mut self, mask: u8, order_hint: u32) {
        for i in 0..NUM_REF_FRAMES {
            if (mask >> i) & 1 == 1 {
                self.slots[i] = RefSlot {
                    order_hint,
                    valid: true,
                };
            }
        }
    }

    /// Reset every slot to invalid. Called on key-frame decode start
    /// (matches §5.9.4 / §6.8 mark-references behaviour).
    pub fn reset(&mut self) {
        for s in &mut self.slots {
            *s = RefSlot::default();
        }
    }
}

/// §5.9.3 `get_relative_dist( a, b )`: signed difference of two
/// `OrderHint` values, modulo `OrderHintBits`. Returns 0 when
/// `enable_order_hint` is false (caller's responsibility — we just
/// expect `order_hint_bits` to be 0 in that case here).
///
/// Pure helper so callers can unit-test the wraparound semantics.
pub fn get_relative_dist(a: u32, b: u32, order_hint_bits: u32) -> i32 {
    if order_hint_bits == 0 {
        return 0;
    }
    let diff = a as i64 - b as i64;
    let m: i64 = 1 << (order_hint_bits - 1);
    let mask: i64 = (1 << order_hint_bits) - 1;
    let wrapped = (diff & mask) - (diff & m) * 2;
    wrapped as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_dpb_has_no_valid_slots() {
        let dpb = Dpb::new();
        assert!(!dpb.any_valid());
    }

    #[test]
    fn refresh_sets_selected_slots() {
        let mut dpb = Dpb::new();
        // Refresh slots 0 and 3 with order hint 5.
        dpb.refresh(0b00001001, 5);
        assert!(dpb.slots[0].valid);
        assert_eq!(dpb.slots[0].order_hint, 5);
        assert!(!dpb.slots[1].valid);
        assert!(dpb.slots[3].valid);
        assert_eq!(dpb.slots[3].order_hint, 5);
        assert!(!dpb.slots[7].valid);
        assert!(dpb.any_valid());
    }

    #[test]
    fn reset_clears_validity() {
        let mut dpb = Dpb::new();
        dpb.refresh(0xFF, 7);
        assert!(dpb.any_valid());
        dpb.reset();
        assert!(!dpb.any_valid());
    }

    #[test]
    fn relative_dist_within_window() {
        // 4 - 2 = 2 with 4-bit order-hint window.
        assert_eq!(get_relative_dist(4, 2, 4), 2);
        assert_eq!(get_relative_dist(2, 4, 4), -2);
    }

    #[test]
    fn relative_dist_handles_wrap() {
        // OrderHintBits = 3 -> values mod 8. 1 vs 7: 1 is "after" 7
        // by +2 (wrap), not -6.
        assert_eq!(get_relative_dist(1, 7, 3), 2);
        assert_eq!(get_relative_dist(7, 1, 3), -2);
    }

    #[test]
    fn relative_dist_zero_when_order_hint_disabled() {
        assert_eq!(get_relative_dist(5, 1, 0), 0);
    }
}
