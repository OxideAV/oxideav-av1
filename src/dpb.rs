//! AV1 Decoded Picture Buffer (DPB) bookkeeping — §7.20 / §7.21.
//!
//! The bitstream-level DPB holds `NUM_REF_FRAMES = 8` slots; each
//! slot tracks the per-frame state that a subsequent inter frame
//! may consult. Round 14 extends the DPB to also carry per-slot
//! reconstructed planes (post-CDEF / post-LR), so SKIP_MODE compound
//! motion compensation (§7.11.3.9) can finally fetch from two
//! independent references named by `SkipModeFrame[0..=1]`. The
//! per-slot `OrderHint` (§5.9.21) is still tracked alongside, since
//! the parser uses it to derive `skipModeAllowed` /
//! `SkipModeFrame[0..=1]` _before_ any block decode.
//!
//! See `frame_header.rs` and `decoder.rs` for the wiring.

use std::sync::Arc;

use crate::decode::FrameState;
use crate::frame_header::NUM_REF_FRAMES;

/// Per-slot DPB state tracked for §5.9.21 + §7.20.
///
/// `OrderHint` follows the spec's `RefOrderHint[]` storage — set when
/// the slot is refreshed by a frame whose `show_frame` (or `showable_
/// frame`) made it visible to consumers. `frame` is the reconstructed
/// frame state (post-CDEF / post-LR), shared via `Arc` so refreshes
/// are O(1). `valid` is `true` iff the slot has ever been refreshed
/// since reset.
#[derive(Clone, Default)]
pub struct RefSlot {
    pub order_hint: u32,
    pub valid: bool,
    /// Reconstructed planes from the frame that last refreshed this
    /// slot (§7.20). `None` when the slot was refreshed for OrderHint
    /// only — the planes path is best-effort and lazily populated.
    pub frame: Option<Arc<FrameState>>,
}

impl std::fmt::Debug for RefSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefSlot")
            .field("order_hint", &self.order_hint)
            .field("valid", &self.valid)
            .field("frame", &self.frame.is_some())
            .finish()
    }
}

/// Decoded Picture Buffer — `NUM_REF_FRAMES` slots, indexed 0..=7.
///
/// Used by the parser to compute `skipModeAllowed` per §5.9.22 (walks
/// `ref_frame_idx[0..REFS_PER_FRAME]` and reads
/// `RefOrderHint[ref_frame_idx[i]]`) AND by the inter-block
/// reconstruction path to source reference samples for compound MC.
#[derive(Clone, Default)]
pub struct Dpb {
    pub slots: [RefSlot; NUM_REF_FRAMES],
}

impl std::fmt::Debug for Dpb {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dpb").field("slots", &self.slots).finish()
    }
}

impl Dpb {
    /// Empty DPB — every slot starts invalid (no reference frames yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` iff at least one slot has been refreshed since reset
    /// (i.e. the decoder has seen at least one frame whose
    /// `refresh_frame_flags != 0`).
    pub fn any_valid(&self) -> bool {
        self.slots.iter().any(|s| s.valid)
    }

    /// §7.20: refresh selected slots from the just-decoded frame's
    /// `OrderHint` only. `mask` is the frame-header
    /// `refresh_frame_flags` 8-bit field — bit `i` set means slot
    /// `i` is overwritten. The reconstructed-planes pointer is
    /// cleared on this path; use [`Self::refresh_with_frame`] to
    /// install both the OrderHint and the reconstructed planes
    /// atomically (the canonical §7.20 behaviour).
    pub fn refresh(&mut self, mask: u8, order_hint: u32) {
        for i in 0..NUM_REF_FRAMES {
            if (mask >> i) & 1 == 1 {
                self.slots[i] = RefSlot {
                    order_hint,
                    valid: true,
                    frame: None,
                };
            }
        }
    }

    /// §7.20 `reference_frame_update_process` — refresh selected
    /// slots from the just-decoded frame, installing both the
    /// `OrderHint` and the reconstructed planes (`Arc<FrameState>`)
    /// so subsequent inter / SKIP_MODE blocks can sample from the
    /// correct slot. Bit `i` of `mask` set means slot `i` is
    /// overwritten.
    pub fn refresh_with_frame(&mut self, mask: u8, order_hint: u32, frame: Arc<FrameState>) {
        for i in 0..NUM_REF_FRAMES {
            if (mask >> i) & 1 == 1 {
                self.slots[i] = RefSlot {
                    order_hint,
                    valid: true,
                    frame: Some(frame.clone()),
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

    /// §7.20 helper: fetch the reconstructed reference planes for
    /// `slot`. Returns `None` when the slot is invalid or planes
    /// haven't been installed.
    pub fn frame_at(&self, slot: usize) -> Option<&Arc<FrameState>> {
        self.slots.get(slot).and_then(|s| s.frame.as_ref())
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

    #[test]
    fn refresh_with_frame_installs_planes() {
        let mut dpb = Dpb::new();
        let fs = std::sync::Arc::new(FrameState::with_bit_depth(8, 8, 1, 1, false, 8));
        // Refresh slots 0 and 5 with a frame at order_hint=12.
        dpb.refresh_with_frame(0b00100001, 12, fs.clone());
        assert_eq!(dpb.slots[0].order_hint, 12);
        assert!(dpb.slots[0].valid);
        assert!(dpb.slots[0].frame.is_some());
        assert!(dpb.frame_at(0).is_some());
        assert!(dpb.frame_at(5).is_some());
        // Untouched slots stay invalid.
        assert!(!dpb.slots[1].valid);
        assert!(dpb.frame_at(1).is_none());
        // OOB slot returns None.
        assert!(dpb.frame_at(99).is_none());
    }

    #[test]
    fn refresh_without_frame_clears_planes() {
        let mut dpb = Dpb::new();
        let fs = std::sync::Arc::new(FrameState::with_bit_depth(8, 8, 1, 1, false, 8));
        // First install planes via the planes-aware refresh.
        dpb.refresh_with_frame(0b00000001, 1, fs);
        assert!(dpb.frame_at(0).is_some());
        // Then a planes-less refresh on the same slot drops the planes
        // pointer (matches the legacy OrderHint-only path).
        dpb.refresh(0b00000001, 2);
        assert!(dpb.frame_at(0).is_none());
        assert_eq!(dpb.slots[0].order_hint, 2);
    }

    #[test]
    fn reset_clears_planes_too() {
        let mut dpb = Dpb::new();
        let fs = std::sync::Arc::new(FrameState::with_bit_depth(8, 8, 1, 1, false, 8));
        dpb.refresh_with_frame(0xFF, 0, fs);
        for s in &dpb.slots {
            assert!(s.frame.is_some());
        }
        dpb.reset();
        for s in &dpb.slots {
            assert!(s.frame.is_none());
            assert!(!s.valid);
        }
    }
}
