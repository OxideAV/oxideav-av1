//! Conformance-grade B-pyramid GOP encoder (r415).
//!
//! Extends the r411-r413 KEY + P GOP driver ([`super::inter_frame`])
//! with OUT-OF-ORDER coding: [`encode_pyramid_gop_yuv420_with_q`]
//! codes each mini-GOP of up to four input frames as a two-level
//! bidirectional pyramid —
//!
//! * an **ALT frame** (the mini-GOP's last frame) coded FIRST as a
//!   decoded-not-shown reference (`show_frame = 0`,
//!   `showable_frame = 1`), forward-predicted from the anchor;
//! * a **MID frame** (the mini-GOP midpoint, chunks of 3-4) coded
//!   second, also not shown, predicting forward from the anchor
//!   (`LAST_FRAME`) and backward from the ALT frame
//!   (`BWDREF_FRAME` / `ALTREF_FRAME` — §7.8 sign bias 1);
//! * **B frames** between the anchors, SHOWN as they are coded, each
//!   predicting forward from its nearest coded past frame and
//!   backward from its nearest coded future frames, with
//!   bidirectional COMPOUND_AVERAGE pairs in the RD ladder
//!   (§5.11.25 `BIDIR_COMP_REFERENCE` cascade) and §5.9.22
//!   forward/backward skip mode;
//! * **`show_existing_frame` headers** (§5.9.2 short form, an
//!   `OBU_FRAME_HEADER`-only temporal unit) emitted at each
//!   not-shown frame's display position.
//!
//! Reference rotation runs on three §7.20 slots with order-hint
//! tracked state: the anchor slot holds the previous mini-GOP's last
//! frame, the ALT / MID slots are refreshed per mini-GOP, and the
//! ALT slot becomes the next mini-GOP's anchor. Every inter frame
//! keeps the r413 session shape — non-error-resilient headers,
//! `PRIMARY_REF_NONE`, 7-bit order hints, `use_ref_frame_mvs = 1`
//! (§7.9 temporal projection over the encoder-side §7.20 motion-field
//! store, which tracks refreshes by slot exactly like the decoder's).
//!
//! Display-order structure per mini-GOP length `L` (anchor at display
//! position `A`, coding order left to right; `SEF(x)` = the
//! `show_existing_frame` unit displaying `x`):
//!
//! ```text
//! L = 1:  P(A+1, shown)
//! L = 2:  ALT(A+2), B(A+1, shown), SEF(A+2)
//! L = 3:  ALT(A+3), MID(A+2), B(A+1, shown), SEF(A+2), SEF(A+3)
//! L = 4:  ALT(A+4), MID(A+2), B(A+1, shown), SEF(A+2),
//!         B(A+3, shown), SEF(A+4)
//! ```
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.9.2
//! (`show_existing_frame`, `showable_frame`, `refresh_frame_flags`,
//! `ref_frame_idx`), §5.9.22 (skip mode), §6.10.24 / §5.11.25
//! (bidirectional compound reference groups), §7.8 (reference frame
//! sign bias), §7.9 / §7.19 / §7.20 (motion fields + reference
//! update).

use crate::encoder::frame_obu::write_frame_header_obu;
use crate::encoder::inter_frame::{
    encode_inter_frame_generic, EncodedGop, GopFrameRecon, InterFrameConfig, SavedMotionField,
    GOP_MAX_FRAMES,
};
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::obu::{build_temporal_unit, ObuFrame};
use crate::encoder::pixel_driver_dyn::{build_intra_only_yuv420_8bit_fh_with_q, Yuv420Frame};
use crate::encoder::rate_twin::RateModel;
use crate::frame_header::{FrameHeader, FrameType, PRIMARY_REF_NONE};
use crate::obu::ObuType;
use crate::sequence_header::SequenceHeader;
use crate::Error;

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
    /// Slot the nearest-backward references (`BWDREF_FRAME` /
    /// `ALTREF2_FRAME`) read; `None` = forward-only (maps to
    /// `last_slot`).
    bwd_slot: Option<usize>,
    /// Slot `ALTREF_FRAME` reads; `None` falls back to `bwd_slot`
    /// (then `last_slot`).
    alt_ref_slot: Option<usize>,
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
    encode_pyramid_gop_yuv420_with_q_rate_model(frames, base_q_idx, RateModel::Twin)
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
    let n = frames.len();

    // Frame 0: the r410 conformance-grade KEY-frame encoder (which
    // also validates the dimension rules). Its `allFrames` refresh
    // fills every §7.20 slot with the KEY (hint 0, intra motion
    // field).
    let key = crate::encoder::key_frame::encode_key_frame_yuv420_with_q_rate_model(
        &frames[0], base_q_idx, model,
    )?;
    let seq = key.seq.clone();
    let mut temporal_units = vec![key.temporal_unit_bytes.clone()];
    let mut recons: Vec<Option<GopFrameRecon>> = (0..n).map(|_| None).collect();
    recons[0] = Some(GopFrameRecon {
        y: key.recon_y,
        u: key.recon_u,
        v: key.recon_v,
    });
    let key_mi = {
        let fh0 = build_intra_only_yuv420_8bit_fh_with_q(&seq, width, height, base_q_idx);
        let fs0 = fh0.frame_size.expect("KEY builder always sizes");
        (fs0.mi_rows, fs0.mi_cols)
    };
    let mut mf_store: [SavedMotionField; 8] =
        core::array::from_fn(|_| SavedMotionField::intra(key_mi.0, key_mi.1));
    let mut slot_hints = [0u32; 8];
    let mut slot_display = [0usize; 8];
    let mut anchor_slot = 0usize;

    let mut pos = 1usize;
    while pos < n {
        let l = (n - pos).min(4);
        // The two non-anchor working slots (all roles live in slots
        // 0..3; the ALT slot becomes the next chunk's anchor).
        let mut others = (0..3usize).filter(|&s| s != anchor_slot);
        let alt_slot = others.next().expect("three working slots");
        let mid_slot = others.next().expect("three working slots");

        let steps = plan_mini_gop(pos, l, anchor_slot, alt_slot, mid_slot);
        // §7.5 bitstream conformance: each temporal unit must carry
        // EXACTLY ONE shown frame — decoded-not-shown pyramid frames
        // accumulate here and ride the next shown frame's unit.
        let mut pending: Vec<ObuFrame> = Vec::new();
        for step in steps {
            match step {
                Step::Code(role) => {
                    // §5.9.2 `ref_frame_idx[]` from the role's slots:
                    // forward references read `last_slot`; BWDREF /
                    // ALTREF2 the nearest-backward slot; ALTREF the
                    // farthest-backward slot.
                    let mut rfi = [role.last_slot as u8; 7];
                    if let Some(b) = role.bwd_slot {
                        rfi[4] = b as u8;
                        rfi[5] = b as u8;
                    }
                    if let Some(a) = role.alt_ref_slot.or(role.bwd_slot) {
                        rfi[6] = a as u8;
                    }
                    // Distinct reference reconstructions + the full
                    // 8-slot map onto them (each slot resolves to the
                    // display frame it truly holds).
                    let mut displays: Vec<usize> = Vec::new();
                    let mut slot_to_plane = [0usize; 8];
                    for s in 0..8 {
                        let d = slot_display[s];
                        let p = displays.iter().position(|&x| x == d).unwrap_or_else(|| {
                            displays.push(d);
                            displays.len() - 1
                        });
                        slot_to_plane[s] = p;
                    }
                    let refs: Vec<&GopFrameRecon> = displays
                        .iter()
                        .map(|&d| recons[d].as_ref().expect("slot holds a coded frame"))
                        .collect();
                    let cfg = InterFrameConfig {
                        order_hint: role.display as u32,
                        show_frame: role.show,
                        refresh_frame_flags: role.refresh.map_or(0, |s| 1u8 << s),
                        ref_frame_idx: rfi,
                        slot_hints,
                        single_refs: role.singles.clone(),
                        compound_pairs: role.pairs.clone(),
                        refs,
                        slot_to_plane,
                        // r423 — the pyramid roles keep per-frame
                        // default state (primary-reference carry
                        // across the B-pyramid's refresh graph is the
                        // ladder-item-4 arc).
                        primary_ref_frame: PRIMARY_REF_NONE,
                        primary_carry: None,
                        allow_temporal_seg: false,
                    };
                    let (obu, rc, saved, _carry, _seg_temporal) = encode_inter_frame_generic(
                        &frames[role.display],
                        &seq,
                        base_q_idx,
                        &cfg,
                        &[],
                        &mf_store,
                        model,
                    )?;
                    pending.push(obu);
                    if role.show {
                        temporal_units.push(build_temporal_unit(None, &pending));
                        pending.clear();
                    }
                    recons[role.display] = Some(rc);
                    // §7.20 reference frame update.
                    if let Some(s) = role.refresh {
                        mf_store[s] = saved;
                        slot_hints[s] = role.display as u32;
                        slot_display[s] = role.display;
                    }
                }
                Step::Show(slot) => {
                    pending.push(show_existing_obu(&seq, slot as u8));
                    temporal_units.push(build_temporal_unit(None, &pending));
                    pending.clear();
                }
            }
        }
        debug_assert!(pending.is_empty(), "every mini-GOP plan ends shown");
        pos += l;
        anchor_slot = alt_slot;
    }

    // IVF v0 wrap. Each temporal unit shows exactly one frame and the
    // units are emitted in display order (a unit's shown frame is
    // always the earliest not-yet-displayed position), so the record
    // index IS the display PTS.
    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(cursor, FOURCC_AV01, width as u16, height as u16, 25, 1)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        for (idx, tu) in temporal_units.iter().enumerate() {
            iw.write_frame(tu, idx as u64)
                .map_err(|_| Error::PartitionWalkOutOfRange)?;
        }
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(EncodedGop {
        ivf_bytes,
        temporal_units,
        recon: recons
            .into_iter()
            .map(|r| r.expect("every display position coded"))
            .collect(),
        seq,
    })
}

/// The coding-order plan for one mini-GOP of `l` frames starting at
/// display position `pos` (anchor at `pos - 1` in `anchor_slot`) —
/// see the module docs for the four structures.
fn plan_mini_gop(
    pos: usize,
    l: usize,
    anchor_slot: usize,
    alt_slot: usize,
    mid_slot: usize,
) -> Vec<Step> {
    // Ladders: `fwd_only` = plain P/ALT shape; `one_bwd` = one
    // distinct backward slot (LAST + BWDREF singles, the
    // { LAST, BWDREF } BIDIR pair); `two_bwd` = distinct BWDREF and
    // ALTREF slots (adds the ALTREF single + { LAST, ALTREF } pair).
    let fwd_only = (vec![1i8], vec![]);
    let one_bwd = (vec![1i8, 5], vec![[1i8, 5]]);
    let two_bwd = (vec![1i8, 5, 7], vec![[1i8, 5], [1i8, 7]]);
    match l {
        1 => vec![Step::Code(Role {
            display: pos,
            show: true,
            refresh: Some(alt_slot),
            last_slot: anchor_slot,
            bwd_slot: None,
            alt_ref_slot: None,
            singles: fwd_only.0,
            pairs: fwd_only.1,
        })],
        2 => vec![
            Step::Code(Role {
                display: pos + 1,
                show: false,
                refresh: Some(alt_slot),
                last_slot: anchor_slot,
                bwd_slot: None,
                alt_ref_slot: None,
                singles: fwd_only.0,
                pairs: fwd_only.1,
            }),
            Step::Code(Role {
                display: pos,
                show: true,
                refresh: None,
                last_slot: anchor_slot,
                bwd_slot: Some(alt_slot),
                alt_ref_slot: None,
                singles: one_bwd.0,
                pairs: one_bwd.1,
            }),
            Step::Show(alt_slot),
        ],
        3 => vec![
            Step::Code(Role {
                display: pos + 2,
                show: false,
                refresh: Some(alt_slot),
                last_slot: anchor_slot,
                bwd_slot: None,
                alt_ref_slot: None,
                singles: fwd_only.0.clone(),
                pairs: fwd_only.1.clone(),
            }),
            Step::Code(Role {
                display: pos + 1,
                show: false,
                refresh: Some(mid_slot),
                last_slot: anchor_slot,
                bwd_slot: Some(alt_slot),
                alt_ref_slot: None,
                singles: one_bwd.0.clone(),
                pairs: one_bwd.1.clone(),
            }),
            Step::Code(Role {
                display: pos,
                show: true,
                refresh: None,
                last_slot: anchor_slot,
                bwd_slot: Some(mid_slot),
                alt_ref_slot: Some(alt_slot),
                singles: two_bwd.0,
                pairs: two_bwd.1,
            }),
            Step::Show(mid_slot),
            Step::Show(alt_slot),
        ],
        _ => vec![
            Step::Code(Role {
                display: pos + 3,
                show: false,
                refresh: Some(alt_slot),
                last_slot: anchor_slot,
                bwd_slot: None,
                alt_ref_slot: None,
                singles: fwd_only.0.clone(),
                pairs: fwd_only.1.clone(),
            }),
            Step::Code(Role {
                display: pos + 1,
                show: false,
                refresh: Some(mid_slot),
                last_slot: anchor_slot,
                bwd_slot: Some(alt_slot),
                alt_ref_slot: None,
                singles: one_bwd.0.clone(),
                pairs: one_bwd.1.clone(),
            }),
            Step::Code(Role {
                display: pos,
                show: true,
                refresh: None,
                last_slot: anchor_slot,
                bwd_slot: Some(mid_slot),
                alt_ref_slot: Some(alt_slot),
                singles: two_bwd.0,
                pairs: two_bwd.1,
            }),
            Step::Show(mid_slot),
            Step::Code(Role {
                display: pos + 2,
                show: true,
                refresh: None,
                last_slot: mid_slot,
                bwd_slot: Some(alt_slot),
                alt_ref_slot: None,
                singles: one_bwd.0,
                pairs: one_bwd.1,
            }),
            Step::Show(alt_slot),
        ],
    }
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
}
