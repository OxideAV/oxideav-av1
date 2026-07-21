//! Conformance-grade B-pyramid GOP encoder (r415, deepened r424).
//!
//! Extends the r411-r413 KEY + P GOP driver ([`super::inter_frame`])
//! with OUT-OF-ORDER coding. r415 introduced the fixed two-level
//! mini-GOP (up to four frames: ALT + MID + shown Bs +
//! `show_existing_frame` chains); r424 generalizes the planner to a
//! RECURSIVE DYADIC PYRAMID of arbitrary depth (mini-GOPs up to
//! [`PyramidTuning::max_mini_gop`] frames, four temporal layers at
//! the default 16), adds per-layer quantiser offsets, threads the
//! r423 §5.9.2 primary-reference carry through the pyramid's refresh
//! graph with a per-frame exact-bytes election, and adds the
//! content-adaptive mini-GOP driver
//! ([`encode_adaptive_gop_yuv420_with_q`]).
//!
//! ## The recursive plan
//!
//! A mini-GOP of `l` frames after anchor display position `A`
//! (`hi = A + l`) codes:
//!
//! 1. the **ALT frame** (`hi`) FIRST as a decoded-not-shown reference
//!    (`show_frame = 0`, `showable_frame = 1`), forward-predicted
//!    from the anchor;
//! 2. the interval `(A, hi)` recursively: its **midpoint** as a
//!    decoded-not-shown MID reference predicting forward from the
//!    interval's low anchor (`LAST_FRAME`) and backward from the
//!    interval's high anchor and the enclosing anchors
//!    (`BWDREF_FRAME` / `ALTREF2_FRAME` / `ALTREF_FRAME` — §7.8 sign
//!    bias 1, §6.10 reference roles), then the left half, a
//!    `show_existing_frame` unit for the midpoint, then the right
//!    half;
//! 3. gap-2 intervals bottom out as **shown B leaves** (no refresh),
//!    with bidirectional COMPOUND_AVERAGE pairs in the RD ladder
//!    (§5.11.25 `BIDIR_COMP_REFERENCE` cascade) and §5.9.22
//!    forward/backward skip mode;
//! 4. a final `show_existing_frame` unit displays the ALT frame.
//!
//! Backward reference roles per coded frame come from the recursion's
//! ENCLOSING-ANCHOR CHAIN (nearest first): `BWDREF` reads the nearest
//! coded future frame, `ALTREF2` the next enclosing one, `ALTREF` the
//! mini-GOP's ALT. Reference rotation runs on all eight §7.20 slots
//! with order-hint-tracked state: the anchor slot holds the previous
//! mini-GOP's last frame, one slot per live pyramid level is
//! allocated from a free list, and the ALT slot becomes the next
//! mini-GOP's anchor.
//!
//! ## Per-layer quantisers (r424)
//!
//! With [`PyramidTuning::layer_q_offsets`] on (the default) each
//! role's frame quantiser is offset from the GOP `base_q_idx` by its
//! temporal layer: the ALT anchor codes slightly FINER (more frames
//! predict from it), midpoints slightly coarser per level, and
//! non-reference B leaves coarsest. Offsets are disabled at
//! `base_q_idx == 0` (lossless) and clamped to `1..=255` otherwise —
//! per-frame `base_q_idx` is plain §5.9.12 header state, so every
//! stream stays conformant.
//!
//! ## Primary-reference carry (r424)
//!
//! With [`PyramidTuning::primary_ref`] on, every coded frame elects a
//! §5.9.2 `primary_ref_frame` over the encoder-side §7.20 carry store
//! (frame-end CDFs / segment ids / gm params per slot — the r423
//! [`RefSlotCarry`] state): the search runs under the LAST-slot carry
//! (the interval's low anchor — the frame's most-relevant coded
//! predecessor), and the committed trees are then replayed bit-exactly
//! under the nearest-BACKWARD anchor's carry and under the
//! per-frame-defaults (`PRIMARY_REF_NONE`) state; the smallest total
//! frame wins ([`InterFrameConfig::alt_primaries`] — a pure-rate
//! election, the reconstruction is identical by construction).
//!
//! Every inter frame keeps the r413 session shape —
//! non-error-resilient headers, 7-bit order hints,
//! `use_ref_frame_mvs = 1` (§7.9 temporal projection over the
//! encoder-side §7.20 motion-field store, which tracks refreshes by
//! slot exactly like the decoder's).
//!
//! Display-order structure examples (anchor at display position `A`,
//! coding order left to right; `SEF(x)` = the `show_existing_frame`
//! unit displaying `x`):
//!
//! ```text
//! L = 1:  P(A+1, shown)
//! L = 2:  ALT(A+2), B(A+1, shown), SEF(A+2)
//! L = 4:  ALT(A+4), MID(A+2), B(A+1, shown), SEF(A+2),
//!         B(A+3, shown), SEF(A+4)
//! L = 8:  ALT(A+8), MID1(A+4), MID2(A+2), B(A+1), SEF(A+2), B(A+3),
//!         SEF(A+4), MID2'(A+6), B(A+5), SEF(A+6), B(A+7), SEF(A+8)
//! ```
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.9.2
//! (`show_existing_frame`, `showable_frame`, `refresh_frame_flags`,
//! `ref_frame_idx`, `primary_ref_frame`), §5.9.22 (skip mode),
//! §6.10.24 / §5.11.25 (bidirectional compound reference groups),
//! §6.8.21 (`load_cdfs`), §7.8 (reference frame sign bias), §7.9 /
//! §7.19 / §7.20 (motion fields + reference update), §7.21
//! (`load_previous`).

use std::rc::Rc;

use crate::cdf::QuantizerParams;
use crate::encoder::frame_obu::write_frame_header_obu;
use crate::encoder::inter_frame::{
    encode_inter_frame_generic, EncodedGop, GopFrameRecon, InterFrameConfig, RefSlotCarry,
    SavedMotionField, GOP_MAX_FRAMES,
};
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::key_frame::lambda_for;
use crate::encoder::obu::{build_temporal_unit, ObuFrame};
use crate::encoder::pixel_driver_dyn::{build_intra_only_yuv420_8bit_fh_with_q, Yuv420Frame};
use crate::encoder::rate_twin::RateModel;
use crate::frame_header::{FrameHeader, FrameType, PRIMARY_REF_NONE};
use crate::obu::ObuType;
use crate::sequence_header::SequenceHeader;
use crate::Error;

/// r424 — the B-pyramid encoder's tuning switches, kept public
/// (hidden) so the measurement harnesses can A/B each feature against
/// its baseline on identical inputs. Production entry points use
/// [`PyramidTuning::default`] (everything on, mini-GOPs up to 16).
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct PyramidTuning {
    /// RD-election rate model (r421).
    pub model: RateModel,
    /// Mini-GOP frame cap (`1..=32`). `4` reproduces the r415-r423
    /// two-level pyramid chunking; the r424 default `16` codes up to
    /// four temporal layers per mini-GOP.
    pub max_mini_gop: usize,
    /// r424 — per-layer quantiser offsets (see the module docs).
    /// Inert at `base_q_idx == 0`.
    pub layer_q_offsets: bool,
    /// r424 — §5.9.2 primary-reference carry + per-frame exact-bytes
    /// election through the pyramid. `false` keeps `PRIMARY_REF_NONE`
    /// per-frame default state on every frame (the r423 baseline).
    pub primary_ref: bool,
}

impl Default for PyramidTuning {
    fn default() -> Self {
        PyramidTuning {
            model: RateModel::Twin,
            max_mini_gop: 16,
            layer_q_offsets: true,
            primary_ref: true,
        }
    }
}

/// r424 (hidden) — [`EncodedGop`] plus the pyramid election traces:
/// the committed mini-GOP lengths and, per coded inter frame in
/// CODING order, the elected §5.9.2 `primary_ref_frame` ordinal
/// (`PRIMARY_REF_NONE` when the per-frame-defaults candidate won or
/// the carry was off). [`EncodedGop`] itself stays field-stable.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TunedPyramidGop {
    pub gop: EncodedGop,
    pub chunk_lengths: Vec<usize>,
    pub primary_elections: Vec<(u32, u8)>,
}

/// One coded (non-KEY) pyramid frame's role.
struct Role {
    /// Display position (== §5.9.2 `order_hint`).
    display: usize,
    /// §5.9.2 `show_frame`.
    show: bool,
    /// §7.20 slot this frame refreshes (`None` = `refresh_frame_flags
    /// = 0`, a pure non-reference frame).
    refresh: Option<usize>,
    /// Slot the forward references (`LAST_FRAME..=GOLDEN_FRAME`) read.
    last_slot: usize,
    /// Slot `BWDREF_FRAME` reads (the nearest coded future frame);
    /// `None` = forward-only (maps to `last_slot`).
    bwd_slot: Option<usize>,
    /// Slot `ALTREF2_FRAME` reads (the next enclosing backward
    /// anchor); `None` falls back to `bwd_slot`.
    alt2_slot: Option<usize>,
    /// Slot `ALTREF_FRAME` reads (the mini-GOP's ALT); `None` falls
    /// back to `bwd_slot`.
    alt_ref_slot: Option<usize>,
    /// r424 — signed per-layer quantiser offset (see
    /// [`PyramidTuning::layer_q_offsets`]).
    q_off: i32,
    /// RD ladder (see `PSearchCtx::single_refs` /
    /// `PSearchCtx::compound_pairs`).
    singles: Vec<i8>,
    pairs: Vec<[i8; 2]>,
}

/// One coding-order step of a mini-GOP plan.
enum Step {
    /// Encode a frame.
    Code(Role),
    /// Emit a `show_existing_frame` header displaying the given slot.
    Show(usize),
}

/// A role's backward reference assignment + RD ladder, derived from
/// the enclosing-anchor chain by [`backward_roles`].
struct BackwardRoles {
    bwd: Option<usize>,
    alt2: Option<usize>,
    alt_ref: Option<usize>,
    singles: Vec<i8>,
    pairs: Vec<[i8; 2]>,
}

/// Build a role's backward reference assignment + RD ladder from the
/// enclosing-anchor `chain` (nearest coded future frame first, the
/// mini-GOP ALT last).
fn backward_roles(chain: &[usize]) -> BackwardRoles {
    let bwd = chain.first().copied();
    let alt_ref = if chain.len() >= 2 {
        chain.last().copied()
    } else {
        None
    };
    let alt2 = if chain.len() >= 3 {
        Some(chain[1])
    } else {
        None
    };
    let mut singles: Vec<i8> = vec![1];
    let mut pairs: Vec<[i8; 2]> = Vec::new();
    if bwd.is_some() {
        singles.push(5);
        pairs.push([1, 5]);
    }
    if alt2.is_some() {
        singles.push(6);
        pairs.push([1, 6]);
    }
    if alt_ref.is_some() {
        singles.push(7);
        pairs.push([1, 7]);
    }
    BackwardRoles {
        bwd,
        alt2,
        alt_ref,
        singles,
        pairs,
    }
}

/// r424 — per-layer quantiser offset schedule: the ALT anchor codes
/// finer (every pyramid frame predicts from it), midpoints coarser
/// per level, non-reference B leaves coarsest. `level` is the
/// midpoint recursion depth (1 = the mini-GOP's top interval).
fn mid_q_off(level: u32) -> i32 {
    match level {
        1 => 2,
        2 => 4,
        _ => 6,
    }
}

/// Recursive interval planner: emit the coding-order steps for the
/// frames STRICTLY between display positions `lo` and `hi` (already
/// coded, held in `lo_slot` / `hi_slot`). `chain` is the
/// enclosing-anchor slot chain visible to this interval's frames
/// (nearest first — `chain[0] == hi_slot`); `level` the midpoint
/// recursion depth; `free` the slot free list.
fn plan_interval(
    lo: usize,
    hi: usize,
    lo_slot: usize,
    chain: &[usize],
    level: u32,
    free: &mut Vec<usize>,
    steps: &mut Vec<Step>,
) {
    let gap = hi - lo;
    if gap <= 1 {
        return;
    }
    if gap == 2 {
        // Shown non-reference B leaf.
        let br = backward_roles(chain);
        steps.push(Step::Code(Role {
            display: lo + 1,
            show: true,
            refresh: None,
            last_slot: lo_slot,
            bwd_slot: br.bwd,
            alt2_slot: br.alt2,
            alt_ref_slot: br.alt_ref,
            q_off: 8,
            singles: br.singles,
            pairs: br.pairs,
        }));
        return;
    }
    // Decoded-not-shown midpoint reference.
    let mid = (lo + hi).div_ceil(2);
    let mid_slot = free.pop().expect("slot free list exhausted (depth > 6)");
    let br = backward_roles(chain);
    steps.push(Step::Code(Role {
        display: mid,
        show: false,
        refresh: Some(mid_slot),
        last_slot: lo_slot,
        bwd_slot: br.bwd,
        alt2_slot: br.alt2,
        alt_ref_slot: br.alt_ref,
        q_off: mid_q_off(level),
        singles: br.singles,
        pairs: br.pairs,
    }));
    // Left half sees the midpoint as its nearest backward anchor.
    let mut inner_chain = Vec::with_capacity(chain.len() + 1);
    inner_chain.push(mid_slot);
    inner_chain.extend_from_slice(chain);
    plan_interval(lo, mid, lo_slot, &inner_chain, level + 1, free, steps);
    steps.push(Step::Show(mid_slot));
    // Right half: the midpoint is its LOW anchor; the backward chain
    // is unchanged (nearest future frame is still `hi`).
    plan_interval(mid, hi, mid_slot, chain, level + 1, free, steps);
    free.push(mid_slot);
}

/// The coding-order plan for one mini-GOP of `l` frames starting at
/// display position `pos` (anchor at `pos - 1` in `anchor_slot`).
/// Returns the steps and the ALT slot (the next chunk's anchor).
fn plan_mini_gop(pos: usize, l: usize, anchor_slot: usize) -> (Vec<Step>, usize) {
    let mut free: Vec<usize> = (0..8usize).rev().filter(|&s| s != anchor_slot).collect();
    let alt_slot = free.pop().expect("eight slots, one anchor");
    let mut steps = Vec::new();
    if l == 1 {
        steps.push(Step::Code(Role {
            display: pos,
            show: true,
            refresh: Some(alt_slot),
            last_slot: anchor_slot,
            bwd_slot: None,
            alt2_slot: None,
            alt_ref_slot: None,
            q_off: 0,
            singles: vec![1],
            pairs: vec![],
        }));
        return (steps, alt_slot);
    }
    let hi = pos + l - 1;
    steps.push(Step::Code(Role {
        display: hi,
        show: false,
        refresh: Some(alt_slot),
        last_slot: anchor_slot,
        bwd_slot: None,
        alt2_slot: None,
        alt_ref_slot: None,
        q_off: -3,
        singles: vec![1],
        pairs: vec![],
    }));
    plan_interval(
        pos - 1,
        hi,
        anchor_slot,
        &[alt_slot],
        1,
        &mut free,
        &mut steps,
    );
    steps.push(Step::Show(alt_slot));
    (steps, alt_slot)
}

/// r424 — the pyramid encoder's running session state: the KEY-seeded
/// §7.20 stores (motion fields, r423 carry, order hints, display
/// map), the anchor slot, the display-order reconstructions and the
/// emitted temporal units. `Clone` is cheap enough for the adaptive
/// driver's trial chunks (the carry store is `Rc`-shared).
#[derive(Clone)]
struct PyramidSession {
    seq: SequenceHeader,
    base_q: u8,
    tuning: PyramidTuning,
    mf_store: [SavedMotionField; 8],
    carry_store: [Rc<RefSlotCarry>; 8],
    slot_hints: [u32; 8],
    slot_display: [usize; 8],
    anchor_slot: usize,
    recons: Vec<Option<GopFrameRecon>>,
    temporal_units: Vec<Vec<u8>>,
    chunk_lengths: Vec<usize>,
    primary_elections: Vec<(u32, u8)>,
}

impl PyramidSession {
    /// Encode the KEY frame and seed every §7.20 slot with its state.
    fn new(frames: &[Yuv420Frame], base_q: u8, tuning: PyramidTuning) -> Result<Self, Error> {
        let n = frames.len();
        let (width, height) = (frames[0].width, frames[0].height);
        let (key, key_carry) = crate::encoder::key_frame::encode_key_frame_yuv420_with_q_carry(
            &frames[0],
            base_q,
            tuning.model,
        )?;
        let seq = key.seq.clone();
        let mut recons: Vec<Option<GopFrameRecon>> = (0..n).map(|_| None).collect();
        recons[0] = Some(GopFrameRecon {
            y: key.recon_y,
            u: key.recon_u,
            v: key.recon_v,
        });
        let key_mi = {
            let fh0 = build_intra_only_yuv420_8bit_fh_with_q(&seq, width, height, base_q);
            let fs0 = fh0.frame_size.expect("KEY builder always sizes");
            (fs0.mi_rows, fs0.mi_cols)
        };
        let key_carry = Rc::new(key_carry);
        Ok(PyramidSession {
            seq,
            base_q,
            tuning,
            mf_store: core::array::from_fn(|_| SavedMotionField::intra(key_mi.0, key_mi.1)),
            carry_store: core::array::from_fn(|_| key_carry.clone()),
            slot_hints: [0u32; 8],
            slot_display: [0usize; 8],
            anchor_slot: 0,
            recons,
            temporal_units: vec![key.temporal_unit_bytes],
            chunk_lengths: Vec::new(),
            primary_elections: Vec::new(),
        })
    }

    /// The role's frame quantiser (per-layer offsets, clamped to the
    /// lossy range; inert at `base_q == 0`).
    fn role_q(&self, role: &Role) -> u8 {
        if self.base_q == 0 || !self.tuning.layer_q_offsets {
            return self.base_q;
        }
        (i32::from(self.base_q) + role.q_off).clamp(1, 255) as u8
    }

    /// Encode one mini-GOP of `l` frames starting at display `pos`
    /// (anchor at `pos - 1`).
    fn encode_chunk(&mut self, frames: &[Yuv420Frame], pos: usize, l: usize) -> Result<(), Error> {
        let (steps, alt_slot) = plan_mini_gop(pos, l, self.anchor_slot);
        // §7.5 bitstream conformance: each temporal unit must carry
        // EXACTLY ONE shown frame — decoded-not-shown pyramid frames
        // accumulate here and ride the next shown frame's unit.
        let mut pending: Vec<ObuFrame> = Vec::new();
        for step in steps {
            match step {
                Step::Code(role) => {
                    // §5.9.2 `ref_frame_idx[]` from the role's slots.
                    let mut rfi = [role.last_slot as u8; 7];
                    if let Some(b) = role.bwd_slot {
                        rfi[4] = b as u8;
                        rfi[5] = b as u8;
                    }
                    if let Some(a2) = role.alt2_slot {
                        rfi[5] = a2 as u8;
                    }
                    if let Some(a) = role.alt_ref_slot.or(role.bwd_slot) {
                        rfi[6] = a as u8;
                    }
                    // Distinct reference reconstructions + the full
                    // 8-slot map onto them (each slot resolves to the
                    // display frame it truly holds).
                    let mut displays: Vec<usize> = Vec::new();
                    let mut slot_to_plane = [0usize; 8];
                    for (plane, &d) in slot_to_plane.iter_mut().zip(&self.slot_display) {
                        *plane = displays.iter().position(|&x| x == d).unwrap_or_else(|| {
                            displays.push(d);
                            displays.len() - 1
                        });
                    }
                    let refs: Vec<&GopFrameRecon> = displays
                        .iter()
                        .map(|&d| self.recons[d].as_ref().expect("slot holds a coded frame"))
                        .collect();
                    // r424 — §5.9.2 primary-reference candidates: the
                    // search runs under the LAST-slot carry (the
                    // frame's most-relevant coded predecessor in its
                    // own prediction chain); the nearest-backward
                    // anchor's carry and the per-frame defaults are
                    // exact-bytes replay alternatives.
                    let last_carry;
                    let bwd_carry;
                    let (primary_ref_frame, primary_carry, alt_primaries) =
                        if self.tuning.primary_ref {
                            last_carry = self.carry_store[role.last_slot].clone();
                            let mut alts: Vec<(u8, Option<&RefSlotCarry>)> =
                                vec![(PRIMARY_REF_NONE, None)];
                            match role.bwd_slot {
                                Some(b) if b != role.last_slot => {
                                    bwd_carry = self.carry_store[b].clone();
                                    alts.push((4, Some(&*bwd_carry)));
                                }
                                _ => {}
                            }
                            (0u8, Some(&*last_carry), alts)
                        } else {
                            (PRIMARY_REF_NONE, None, Vec::new())
                        };
                    let cfg = InterFrameConfig {
                        order_hint: role.display as u32,
                        show_frame: role.show,
                        refresh_frame_flags: role.refresh.map_or(0, |s| 1u8 << s),
                        ref_frame_idx: rfi,
                        slot_hints: self.slot_hints,
                        single_refs: role.singles.clone(),
                        compound_pairs: role.pairs.clone(),
                        refs,
                        slot_to_plane,
                        primary_ref_frame,
                        primary_carry,
                        allow_temporal_seg: false,
                        alt_primaries,
                    };
                    let q = self.role_q(&role);
                    let (obu, rc, saved, carry, aux) = encode_inter_frame_generic(
                        &frames[role.display],
                        &self.seq,
                        q,
                        &cfg,
                        &[],
                        &self.mf_store,
                        self.tuning.model,
                    )?;
                    self.primary_elections
                        .push((role.display as u32, aux.primary_ref));
                    pending.push(obu);
                    if role.show {
                        self.temporal_units
                            .push(build_temporal_unit(None, &pending));
                        pending.clear();
                    }
                    self.recons[role.display] = Some(rc);
                    // §7.20 reference frame update.
                    if let Some(s) = role.refresh {
                        self.mf_store[s] = saved;
                        self.carry_store[s] = Rc::new(carry);
                        self.slot_hints[s] = role.display as u32;
                        self.slot_display[s] = role.display;
                    }
                }
                Step::Show(slot) => {
                    pending.push(show_existing_obu(&self.seq, slot as u8));
                    self.temporal_units
                        .push(build_temporal_unit(None, &pending));
                    pending.clear();
                }
            }
        }
        debug_assert!(pending.is_empty(), "every mini-GOP plan ends shown");
        self.anchor_slot = alt_slot;
        self.chunk_lengths.push(l);
        Ok(())
    }

    /// Sum of coded bytes across all emitted temporal units.
    fn total_bytes(&self) -> usize {
        self.temporal_units.iter().map(Vec::len).sum()
    }

    /// Squared-error distortion of the committed reconstructions over
    /// display positions `range` (all three planes).
    fn distortion_over(&self, frames: &[Yuv420Frame], range: core::ops::Range<usize>) -> u64 {
        let mut d = 0u64;
        for pos in range {
            let rc = self.recons[pos].as_ref().expect("range coded");
            let f = &frames[pos];
            for (a, b) in
                rc.y.iter()
                    .zip(&f.y)
                    .chain(rc.u.iter().zip(&f.u))
                    .chain(rc.v.iter().zip(&f.v))
            {
                let diff = i64::from(*a) - i64::from(*b);
                d += (diff * diff) as u64;
            }
        }
        d
    }

    /// IVF v0 wrap + [`EncodedGop`] assembly. Each temporal unit shows
    /// exactly one frame and the units are emitted in display order,
    /// so the record index IS the display PTS.
    fn finish(self, width: u32, height: u32) -> Result<TunedPyramidGop, Error> {
        let mut ivf_bytes: Vec<u8> = Vec::new();
        {
            let cursor = std::io::Cursor::new(&mut ivf_bytes);
            let mut iw = IvfWriter::new(cursor, FOURCC_AV01, width as u16, height as u16, 25, 1)
                .map_err(|_| Error::PartitionWalkOutOfRange)?;
            for (idx, tu) in self.temporal_units.iter().enumerate() {
                iw.write_frame(tu, idx as u64)
                    .map_err(|_| Error::PartitionWalkOutOfRange)?;
            }
            iw.patch_frame_count()
                .map_err(|_| Error::PartitionWalkOutOfRange)?;
        }
        Ok(TunedPyramidGop {
            gop: EncodedGop {
                ivf_bytes,
                temporal_units: self.temporal_units,
                recon: self
                    .recons
                    .into_iter()
                    .map(|r| r.expect("every display position coded"))
                    .collect(),
                seq: self.seq,
            },
            chunk_lengths: self.chunk_lengths,
            primary_elections: self.primary_elections,
        })
    }
}

/// Shared input validation for the GOP drivers.
fn validate_gop_input(frames: &[Yuv420Frame]) -> Result<(u32, u32), Error> {
    if frames.is_empty() || frames.len() > GOP_MAX_FRAMES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let (width, height) = (frames[0].width, frames[0].height);
    if frames
        .iter()
        .any(|f| f.width != width || f.height != height)
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    Ok((width, height))
}

/// Lossless (`base_q_idx = 0`) B-pyramid GOP encode — see
/// [`encode_pyramid_gop_yuv420_with_q`].
pub fn encode_pyramid_gop_yuv420(frames: &[Yuv420Frame]) -> Result<EncodedGop, Error> {
    encode_pyramid_gop_yuv420_with_q(frames, 0)
}

/// Encode a KEY + B-pyramid GOP of 8-bit 4:2:0 frames at `base_q_idx`
/// into a spec-conformant IVF stream (see the module docs for the
/// exact structure).
///
/// The returned [`EncodedGop`]'s `temporal_units` hold one §7.5 unit
/// per DISPLAY frame: each unit carries exactly one shown frame (the
/// bitstream-conformance rule), so decoded-not-shown ALT / MID frames
/// ride the unit of the next shown frame and `show_existing_frame`
/// headers form their own units. `recon` is in DISPLAY order, one
/// entry per input frame — the decoded output equals it byte-for-byte
/// (and the inputs too at `base_q_idx == 0`).
///
/// ## Errors
///
/// * Empty input, more than [`GOP_MAX_FRAMES`] frames, mismatched
///   dimensions across frames, or dimensions outside the KEY-frame
///   rules — [`Error::PartitionWalkOutOfRange`].
pub fn encode_pyramid_gop_yuv420_with_q(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
) -> Result<EncodedGop, Error> {
    encode_pyramid_gop_yuv420_with_q_tuned(frames, base_q_idx, PyramidTuning::default())
        .map(|t| t.gop)
}

/// r421 — [`encode_pyramid_gop_yuv420_with_q`] with an explicit
/// [`RateModel`], kept public (hidden) so the sweep harnesses can A/B
/// the twin-priced elections against the pre-r421 heuristic baseline.
#[doc(hidden)]
pub fn encode_pyramid_gop_yuv420_with_q_rate_model(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    model: RateModel,
) -> Result<EncodedGop, Error> {
    encode_pyramid_gop_yuv420_with_q_tuned(
        frames,
        base_q_idx,
        PyramidTuning {
            model,
            ..PyramidTuning::default()
        },
    )
    .map(|t| t.gop)
}

/// r424 — [`encode_pyramid_gop_yuv420_with_q`] with explicit
/// [`PyramidTuning`] switches (the measurement-harness entry point):
/// fixed maximal chunking at `tuning.max_mini_gop`.
#[doc(hidden)]
pub fn encode_pyramid_gop_yuv420_with_q_tuned(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    tuning: PyramidTuning,
) -> Result<TunedPyramidGop, Error> {
    let (width, height) = validate_gop_input(frames)?;
    if tuning.max_mini_gop == 0 || tuning.max_mini_gop > 32 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let n = frames.len();
    let mut session = PyramidSession::new(frames, base_q_idx, tuning)?;
    let mut pos = 1usize;
    while pos < n {
        let l = (n - pos).min(tuning.max_mini_gop);
        session.encode_chunk(frames, pos, l)?;
        pos += l;
    }
    session.finish(width, height)
}

// ---------------------------------------------------------------------
// r424 — content-adaptive mini-GOP sizing.
// ---------------------------------------------------------------------

/// r424 — the adaptive driver's tuning switches (hidden — the
/// measurement harnesses A/B the election against fixed chunking).
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveTuning {
    pub pyramid: PyramidTuning,
    /// Trial-encode boundary election between the class-picked
    /// mini-GOP shape and its half-depth split (twin-consistent
    /// `D + λ·R` scoring over the identical frame span). `false`
    /// commits the class pick directly.
    pub elect: bool,
}

impl Default for AdaptiveTuning {
    fn default() -> Self {
        AdaptiveTuning {
            pyramid: PyramidTuning::default(),
            elect: true,
        }
    }
}

/// r424 (hidden) — [`EncodedGop`] plus the adaptive election traces.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct TunedAdaptiveGop {
    pub gop: EncodedGop,
    /// Committed mini-GOP lengths, in order.
    pub chunk_lengths: Vec<usize>,
    /// Per-transition motion-compensated mean-absolute-difference
    /// (`mc_mad[k]` = frames `k → k+1`, `k` in `0..n-1`).
    pub mc_mads: Vec<f64>,
    /// Scene-cut flags per transition (same indexing).
    pub cuts: Vec<bool>,
    /// Boundary elections: `(pos, deep_len, split_len, chosen_len)`.
    pub elections: Vec<(usize, usize, usize, usize)>,
}

/// Half-resolution luma (2×2 box filter) for the motion probe.
fn half_res_luma(f: &Yuv420Frame) -> (Vec<u8>, usize, usize) {
    let (w, h) = (f.width as usize, f.height as usize);
    let (hw, hh) = (w / 2, h / 2);
    let mut out = vec![0u8; hw * hh];
    for i in 0..hh {
        for j in 0..hw {
            let s = u32::from(f.y[(2 * i) * w + 2 * j])
                + u32::from(f.y[(2 * i) * w + 2 * j + 1])
                + u32::from(f.y[(2 * i + 1) * w + 2 * j])
                + u32::from(f.y[(2 * i + 1) * w + 2 * j + 1]);
            out[i * hw + j] = ((s + 2) / 4) as u8;
        }
    }
    (out, hw, hh)
}

/// Motion-compensated MAD probe: the minimum mean absolute luma
/// difference between `prev` and `cur` over global integer shifts of
/// up to ±3 half-resolution samples (±6 full-resolution) per axis —
/// a cheap predictability measure that stays small under pure
/// translation and large across scene cuts / noise.
fn mc_mad(prev: &[u8], cur: &[u8], w: usize, h: usize) -> f64 {
    let mut best = f64::INFINITY;
    for dy in -3i32..=3 {
        for dx in -3i32..=3 {
            let y0 = dy.max(0) as usize;
            let y1 = (h as i32 + dy.min(0)) as usize;
            let x0 = dx.max(0) as usize;
            let x1 = (w as i32 + dx.min(0)) as usize;
            if y1 <= y0 || x1 <= x0 {
                continue;
            }
            let mut sum = 0u64;
            for i in y0..y1 {
                let pi = (i as i32 - dy) as usize;
                for j in x0..x1 {
                    let pj = (j as i32 - dx) as usize;
                    sum += u64::from(cur[i * w + j].abs_diff(prev[pi * w + pj]));
                }
            }
            let mad = sum as f64 / ((y1 - y0) * (x1 - x0)) as f64;
            if mad < best {
                best = mad;
            }
        }
    }
    best
}

/// Scene-cut threshold on the motion-compensated MAD probe.
const CUT_MC_MAD: f64 = 20.0;

/// Motion-class mini-GOP target from the window's mean probe value.
fn class_len(mean_mc_mad: f64, cap: usize) -> usize {
    let target = if mean_mc_mad < 1.5 {
        16
    } else if mean_mc_mad < 6.0 {
        8
    } else if mean_mc_mad < 12.0 {
        4
    } else if mean_mc_mad < CUT_MC_MAD {
        2
    } else {
        1
    };
    target.min(cap).max(1)
}

/// Encode a KEY + adaptive mini-GOP stream: content-driven mini-GOP
/// sizing (scene-cut / motion-probe classes choosing between pyramid
/// depths and flat P-runs, with a twin-consistent `D + λ·R`
/// trial-encode election at the class boundary — see the module
/// docs). Same output contract as
/// [`encode_pyramid_gop_yuv420_with_q`].
pub fn encode_adaptive_gop_yuv420_with_q(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
) -> Result<EncodedGop, Error> {
    encode_adaptive_gop_yuv420_with_q_tuned(frames, base_q_idx, AdaptiveTuning::default())
        .map(|t| t.gop)
}

/// r424 — [`encode_adaptive_gop_yuv420_with_q`] with explicit
/// [`AdaptiveTuning`] (the measurement-harness entry point).
#[doc(hidden)]
pub fn encode_adaptive_gop_yuv420_with_q_tuned(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    tuning: AdaptiveTuning,
) -> Result<TunedAdaptiveGop, Error> {
    let (width, height) = validate_gop_input(frames)?;
    if tuning.pyramid.max_mini_gop == 0 || tuning.pyramid.max_mini_gop > 32 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let n = frames.len();

    // Per-transition motion probe (`mc_mads[k]`: frame k -> k+1).
    let halves: Vec<(Vec<u8>, usize, usize)> = frames.iter().map(half_res_luma).collect();
    let mc_mads: Vec<f64> = (0..n.saturating_sub(1))
        .map(|k| {
            let (p, w, h) = &halves[k];
            let (c, _, _) = &halves[k + 1];
            mc_mad(p, c, *w, *h)
        })
        .collect();
    let cuts: Vec<bool> = mc_mads.iter().map(|&m| m > CUT_MC_MAD).collect();

    let lambda = lambda_for(&QuantizerParams::neutral(base_q_idx, 8));
    let mut session = PyramidSession::new(frames, base_q_idx, tuning.pyramid)?;
    let mut elections: Vec<(usize, usize, usize, usize)> = Vec::new();

    let mut pos = 1usize;
    while pos < n {
        let remaining = n - pos;
        // Cut-free window: the chunk `pos..pos+l-1` must not contain a
        // cut transition strictly inside its prediction span
        // (`pos..pos+l-1`); a cut right at the anchor (`cuts[pos-1]`)
        // forces a flat P step that absorbs the discontinuity.
        let mut w = 1usize;
        if !cuts[pos - 1] {
            while w < remaining.min(tuning.pyramid.max_mini_gop) && !cuts[pos + w - 1] {
                w += 1;
            }
        }
        let l_main = if cuts[pos - 1] {
            1
        } else {
            let mean = mc_mads[pos - 1..pos + w - 1].iter().sum::<f64>() / w as f64;
            class_len(mean, w)
        };
        let l_alt = l_main / 2;
        if !tuning.elect || l_alt == 0 || l_alt == l_main {
            session.encode_chunk(frames, pos, l_main)?;
            pos += l_main;
            continue;
        }
        // Twin-consistent boundary election over the IDENTICAL frame
        // span `pos..pos+l_main`: one deep chunk vs. half-depth
        // splits. `D + λ·R` in `score256` units (exact realized bytes
        // — 8·256 rate units per byte); at `base_q_idx == 0` both
        // reconstructions are the input, so the comparison reduces to
        // pure rate.
        let base_bytes = session.total_bytes();
        let mut deep = session.clone();
        deep.encode_chunk(frames, pos, l_main)?;
        let mut split = session.clone();
        let mut covered = 0usize;
        while covered < l_main {
            let step = l_alt.min(l_main - covered);
            split.encode_chunk(frames, pos + covered, step)?;
            covered += step;
        }
        let deep_j = deep.distortion_over(frames, pos..pos + l_main) * 256
            + lambda * ((deep.total_bytes() - base_bytes) as u64 * 8 * 256);
        let split_j = split.distortion_over(frames, pos..pos + l_main) * 256
            + lambda * ((split.total_bytes() - base_bytes) as u64 * 8 * 256);
        let chosen = if deep_j <= split_j { l_main } else { l_alt };
        elections.push((pos, l_main, l_alt, chosen));
        session = if deep_j <= split_j { deep } else { split };
        pos += l_main;
    }

    let tuned = session.finish(width, height)?;
    Ok(TunedAdaptiveGop {
        gop: tuned.gop,
        chunk_lengths: tuned.chunk_lengths,
        mc_mads,
        cuts,
        elections,
    })
}

/// §5.9.2 short-form `show_existing_frame` OBU: an
/// `OBU_FRAME_HEADER` whose body is `show_existing_frame = 1` +
/// `frame_to_show_map_idx` (+ §5.3.4 trailing bits).
fn show_existing_obu(seq: &SequenceHeader, map_idx: u8) -> ObuFrame {
    let fh = FrameHeader {
        show_existing_frame: true,
        frame_to_show_map_idx: Some(map_idx),
        display_frame_id: None,
        frame_type: FrameType::Inter,
        frame_is_intra: false,
        show_frame: true,
        showable_frame: false,
        error_resilient_mode: false,
        disable_cdf_update: false,
        allow_screen_content_tools: false,
        force_integer_mv: false,
        current_frame_id: 0,
        frame_size_override_flag: false,
        order_hint: 0,
        primary_ref_frame: PRIMARY_REF_NONE,
        refresh_frame_flags: 0,
        ref_order_hints: None,
        frame_size: None,
        allow_intrabc: false,
        disable_frame_end_update_cdf: false,
        tile_info: None,
        quantization_params: None,
        segmentation_params: None,
        delta_q_params: None,
        delta_lf_params: None,
        loop_filter_params: None,
        cdef_params: None,
        lr_params: None,
        tx_mode: None,
        reference_select: None,
        skip_mode_present: None,
        skip_mode_frame: None,
        allow_warped_motion: None,
        reduced_tx_set: None,
        global_motion_params: None,
        film_grain_params: None,
        inter_refs: None,
        bits_consumed: 0,
    };
    let body = write_frame_header_obu(&fh, seq);
    ObuFrame::new(ObuType::FrameHeader, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic textured frame with translation `(sy, sx)` — the
    /// shared GOP-test generator.
    fn moving_gradient(w: u32, h: u32, shift_y: usize, shift_x: usize, seed: u32) -> Yuv420Frame {
        let (wu, hu) = (w as usize, h as usize);
        let s = seed as usize;
        let mut f = Yuv420Frame::filled(w, h, 0);
        for i in 0..hu {
            for j in 0..wu {
                let (si, sj) = (i + shift_y, j + shift_x);
                f.y[i * wu + j] = ((si * 5 + sj * 3 + (si / 16) * (sj / 16) + s) % 256) as u8;
            }
        }
        let (cw, ch) = (wu / 2, hu / 2);
        for i in 0..ch {
            for j in 0..cw {
                let (si, sj) = (i + shift_y / 2, j + shift_x / 2);
                f.u[i * cw + j] = ((128 + si * 2 + sj + s) % 256) as u8;
                f.v[i * cw + j] = ((64 + si + sj * 2 + s) % 256) as u8;
            }
        }
        f
    }

    fn assert_pyramid_round_trip(frames: &[Yuv420Frame], q: u8) -> EncodedGop {
        let (w, h) = (frames[0].width, frames[0].height);
        let enc = encode_pyramid_gop_yuv420_with_q(frames, q)
            .unwrap_or_else(|e| panic!("{w}x{h} q={q}: pyramid encode failed: {e:?}"));
        assert_eq!(enc.recon.len(), frames.len());
        let decoded = crate::decoder::decode_av1_spec(&enc.ivf_bytes)
            .unwrap_or_else(|e| panic!("{w}x{h} q={q}: spec driver rejected pyramid GOP: {e:?}"));
        assert_eq!(
            decoded.len(),
            frames.len(),
            "{w}x{h} q={q}: one shown frame per input (display order)"
        );
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!((f.width, f.height), (w, h));
            let rc = &enc.recon[idx];
            assert_eq!(f.planes[0], rc.y, "{w}x{h} q={q} display {idx}: luma");
            assert_eq!(f.planes[1], rc.u, "{w}x{h} q={q} display {idx}: U");
            assert_eq!(f.planes[2], rc.v, "{w}x{h} q={q} display {idx}: V");
            if q == 0 {
                assert_eq!(f.planes[0], frames[idx].y, "lossless {idx}: luma != input");
                assert_eq!(f.planes[1], frames[idx].u, "lossless {idx}: U != input");
                assert_eq!(f.planes[2], frames[idx].v, "lossless {idx}: V != input");
            }
        }
        enc
    }

    /// One full L=4 mini-GOP (5 frames): ALT + MID + two shown Bs +
    /// two show_existing units, lossy.
    #[test]
    fn r415_pyramid_len5_lossy_round_trips() {
        let frames: Vec<Yuv420Frame> = (0..5)
            .map(|k| moving_gradient(64, 64, 2 * k, 3 * k, 11))
            .collect();
        let enc = assert_pyramid_round_trip(&frames, 60);
        // One temporal unit per DISPLAY frame (each unit carries
        // exactly one shown frame; ALT/MID ride their B's unit).
        assert_eq!(enc.temporal_units.len(), 5);
    }

    /// Lossless single-level pyramid (L=2 tail): 3 frames = KEY +
    /// ALT + B + SEF.
    #[test]
    fn r415_pyramid_len3_lossless_round_trips() {
        let frames: Vec<Yuv420Frame> = (0..3)
            .map(|k| moving_gradient(64, 64, 3 * k, 5 * k, 9))
            .collect();
        let enc = assert_pyramid_round_trip(&frames, 0);
        // 1 KEY unit + (L=2: [ALT + B] + [SEF]) = 3 units, one per
        // display frame.
        assert_eq!(enc.temporal_units.len(), 3);
    }

    /// r424: one full three-level mini-GOP (9 frames = KEY + L=8):
    /// ALT + MID1 + two MID2s + four shown B leaves, distinct
    /// BWDREF / ALTREF2 / ALTREF backward roles on the deepest
    /// leaves, per-layer q offsets and the primary-reference
    /// election live.
    #[test]
    fn r424_pyramid_len9_three_levels_round_trips() {
        let frames: Vec<Yuv420Frame> = (0..9)
            .map(|k| moving_gradient(64, 64, 2 * k, 3 * k, 23))
            .collect();
        let enc = assert_pyramid_round_trip(&frames, 60);
        assert_eq!(enc.temporal_units.len(), 9);
    }

    /// r424: the recursive planner reproduces the r415 two-level
    /// shapes at `max_mini_gop = 4` and codes every display position
    /// exactly once at every depth cap.
    #[test]
    fn r424_planner_covers_every_position_once() {
        for max in [1usize, 2, 3, 4, 8, 16] {
            for n in 1..=17usize {
                let mut covered = vec![false; n + 1];
                let mut pos = 1usize;
                let mut anchor = 0usize;
                while pos <= n {
                    let l = (n + 1 - pos).min(max);
                    let (steps, alt) = plan_mini_gop(pos, l, anchor);
                    let mut shows = 0usize;
                    for s in &steps {
                        match s {
                            Step::Code(role) => {
                                assert!(
                                    !covered[role.display],
                                    "max={max} n={n}: display {} coded twice",
                                    role.display
                                );
                                covered[role.display] = true;
                                if let Some(r) = role.refresh {
                                    assert_ne!(r, anchor, "must not clobber the live anchor");
                                }
                                if role.show {
                                    shows += 1;
                                }
                            }
                            Step::Show(_) => shows += 1,
                        }
                    }
                    assert_eq!(shows, l, "max={max} n={n}: one display per chunk frame");
                    pos += l;
                    anchor = alt;
                }
                assert!(
                    covered[1..=n].iter().all(|&c| c),
                    "max={max} n={n}: every position coded"
                );
            }
        }
    }

    /// r424: adaptive driver on scene-cut content — the cut is
    /// detected, no mini-GOP spans it, and the stream round-trips.
    #[test]
    fn r424_adaptive_scene_cut_round_trips() {
        let n = 9usize;
        let frames: Vec<Yuv420Frame> = (0..n)
            .map(|k| {
                if k >= n / 2 {
                    // Different texture family after the cut.
                    moving_gradient(64, 64, 40 + 2 * k, 70 + 3 * k, 901)
                } else {
                    moving_gradient(64, 64, 2 * k, 3 * k, 5)
                }
            })
            .collect();
        let tuned = encode_adaptive_gop_yuv420_with_q_tuned(&frames, 60, AdaptiveTuning::default())
            .expect("adaptive encode");
        assert!(
            tuned.cuts.iter().any(|&c| c),
            "the constructed cut must trip the probe: {:?}",
            tuned.mc_mads
        );
        let decoded = crate::decoder::decode_av1_spec(&tuned.gop.ivf_bytes)
            .expect("spec driver accepts adaptive stream");
        assert_eq!(decoded.len(), n);
        for (idx, f) in decoded.iter().enumerate() {
            assert_eq!(f.planes[0], tuned.gop.recon[idx].y, "display {idx} luma");
        }
    }
}
