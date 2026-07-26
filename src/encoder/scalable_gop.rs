//! r430 — temporally scalable GOP encoder: §6.7.5 operating points on
//! the wire, §5.3.3 OBU extension headers on every frame OBU.
//!
//! ## Stream shape
//!
//! Display-order, zero-latency dyadic temporal layering over `L`
//! layers (`2..=4`): with the period `P = 1 << (L - 1)`, display
//! position `i` rides temporal layer
//!
//! ```text
//!   tid(i) = 0                                    if i % P == 0
//!   tid(i) = L - 1 - TrailingZeros(i % P)         otherwise
//! ```
//!
//! (the classic dyadic ladder — for `L = 3`: `0 2 1 2 0 2 1 2 …`).
//! Every frame is SHOWN and rides its own §7.5 temporal unit, so each
//! unit is homogeneous in `temporal_id` — the §7.5 layered-stream
//! constraints ("all OBU extension headers in the same temporal unit
//! with the same spatial_id must have the same temporal_id"; "every
//! layer with a coded frame in a temporal unit has exactly one shown
//! frame, the last of that layer in the unit") hold by construction.
//!
//! ## Reference discipline
//!
//! A frame at layer `t` predicts ONLY from frames at layers `<= t`
//! (in fact `< t` for `t > 0`; the previous same-layer frame for
//! `t = 0`): its LAST reference is display position
//! `i - (1 << (L - 1 - tid))`, whose own layer is strictly lower for
//! `tid > 0`. §7.20 slot policy: layer `t < L - 1` refreshes slot `t`
//! exclusively; top-layer frames are non-reference
//! (`refresh_frame_flags = 0`). Dropping any suffix of layers
//! therefore leaves every surviving frame's references (and its
//! §8.3.1 primary-reference CDF chain, which rides the LAST slot)
//! intact — decoding at operating point `k` yields exactly the shown
//! frames with `tid <= L - 1 - k`, byte-identical to their full-
//! stream reconstructions.
//!
//! ## Operating-point signalling (§5.5.1 / §6.7.5)
//!
//! The sequence header codes `operating_points_cnt_minus_1 = L - 1`
//! with, for operating point `k`,
//!
//! ```text
//!   operating_point_idc[ k ] = 0x100 | ((1 << (L - k)) - 1)
//! ```
//!
//! — spatial layer 0 (bit 8) plus temporal layers `0..=L-1-k` (§6.7.5
//! bit lanes: temporal layers in bits 0..7, spatial layers in bits
//! 8..11). Operating point 0 is the full stream (the §6.7.5 preferred
//! entry), the last is the base layer alone. Every
//! `operating_point_idc` is non-zero, satisfying the §6.7.5
//! requirement that `OperatingPointIdc == 0` streams carry no
//! extension headers (ours carries one on every frame OBU, per the
//! §7.5 rule that a sequence with any enhancement-layer OBU carries
//! extension headers on ALL frame header + tile group OBUs).
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.3.3, §5.5.1,
//! §6.7.5, §7.5.

use crate::encoder::inter_frame::{
    encode_inter_frame_generic, narrow_gop_8bit, EncodedGop, EncodedGopYuv, GopFrameReconYuv,
    InterFrameConfig, RefSlotCarry, SavedMotionField,
};
use crate::encoder::ivf::{IvfWriter, FOURCC_AV01};
use crate::encoder::key_frame::encode_key_frame_yuv_seg_carry;
use crate::encoder::obu::{
    build_temporal_unit, write_obu_with_size, ObuExtensionHeader, ObuHeader,
};
use crate::encoder::pyramid_gop::validate_gop_input;
use crate::encoder::rate_twin::RateModel;
use crate::encoder::sequence_obu::write_sequence_header_obu;
use crate::encoder::yuv_frame::{Yuv420Frame, YuvFrame};
use crate::frame_header::PRIMARY_REF_NONE;
use crate::obu::{ObuIter, ObuType};
use crate::sequence_header::OperatingPoint;
use crate::Error;
use std::rc::Rc;

/// [`EncodedGopYuv`] plus the per-display-frame `temporal_id` ladder
/// the stream signals (index = display position; `temporal_ids[0] ==
/// 0` — the KEY frame anchors the base layer).
#[derive(Debug, Clone)]
pub struct TemporalLayeredGopYuv {
    /// The encoded stream (IVF wrap + bare §7.5 temporal units +
    /// per-frame reconstructions + the emitted sequence header, which
    /// carries the §6.7.5 operating-point list).
    pub gop: EncodedGopYuv,
    /// §5.3.3 `temporal_id` per display frame.
    pub temporal_ids: Vec<u8>,
}

/// 8-bit 4:2:0 sibling of [`TemporalLayeredGopYuv`].
#[derive(Debug, Clone)]
pub struct TemporalLayeredGop {
    /// The encoded stream (see [`TemporalLayeredGopYuv::gop`]).
    pub gop: EncodedGop,
    /// §5.3.3 `temporal_id` per display frame.
    pub temporal_ids: Vec<u8>,
}

/// §5.3.3 `temporal_id` of display position `i` in the dyadic
/// `layers`-deep ladder (see the module docs).
#[must_use]
pub fn temporal_layer_of(i: usize, layers: u8) -> u8 {
    let period = 1usize << (layers - 1);
    let m = i % period;
    if m == 0 {
        0
    } else {
        layers - 1 - (m.trailing_zeros() as u8)
    }
}

/// Per-layer quantiser offset — anchors code finest, leaves coarsest
/// (the same ladder philosophy as the B-pyramid's `mid_q_off`).
/// Inert at `base_q_idx == 0`.
fn layer_q_off(tid: u8) -> i32 {
    match tid {
        0 => 0,
        1 => 4,
        2 => 6,
        _ => 8,
    }
}

/// The §6.7.5 operating-point list for `layers` temporal layers over
/// spatial layer 0: entry `k` selects temporal layers `0..=layers-1-k`
/// (entry 0 = the full stream, the preferred §6.7.5 ordering).
fn operating_points_for(base: OperatingPoint, layers: u8) -> Vec<OperatingPoint> {
    (0..layers)
        .map(|k| OperatingPoint {
            operating_point_idc: 0x100 | ((1u16 << (layers - k)) - 1),
            ..base
        })
        .collect()
}

/// Repack a §7.5 temporal unit: replace the sequence-header OBU body
/// with `seq_payload` (when supplied) and stamp a §5.3.3 extension
/// header (`temporal_id`, spatial_id 0) onto every frame-carrying
/// OBU. TD OBUs pass through bare (the §5.3.1 drop rule exempts
/// them; §7.5 requires extension headers on frame header / tile
/// group / frame OBUs only).
fn repack_tu_with_extension(tu: &[u8], temporal_id: u8, seq_payload: Option<&[u8]>) -> Vec<u8> {
    let mut out = Vec::new();
    for desc in ObuIter::new(tu) {
        let desc = desc.expect("own temporal unit walks");
        match desc.obu_type {
            ObuType::TemporalDelimiter => {
                write_obu_with_size(
                    &mut out,
                    &ObuHeader::new(ObuType::TemporalDelimiter),
                    desc.payload,
                );
            }
            ObuType::SequenceHeader => {
                let body: &[u8] = seq_payload.unwrap_or(desc.payload);
                write_obu_with_size(&mut out, &ObuHeader::new(ObuType::SequenceHeader), body);
            }
            other => {
                let header =
                    ObuHeader::new(other).with_extension(ObuExtensionHeader::new(temporal_id, 0));
                write_obu_with_size(&mut out, &header, desc.payload);
            }
        }
    }
    out
}

/// Encode a temporally scalable KEY + inter GOP at any conformant
/// (bit depth, chroma format) pairing: `temporal_layers` dyadic
/// layers (`2..=4`), operating points signalled per §6.7.5, extension
/// headers on every frame OBU per §7.5, display-order zero-latency
/// coding (no hidden frames).
///
/// Decoding the returned stream at operating point `k`
/// ([`crate::decode_av1_at_operating_point`]) yields exactly the
/// shown frames whose `temporal_ids[i] <= temporal_layers - 1 - k`,
/// each byte-identical to `gop.recon[i]`.
///
/// ## Errors
///
/// * `temporal_layers` outside `2..=4`, or any
///   [`crate::encoder::encode_pyramid_gop_yuv_with_q`] input reject
///   (empty / oversized / mismatched GOP) —
///   [`Error::PartitionWalkOutOfRange`].
pub fn encode_temporal_layered_gop_yuv_with_q(
    frames: &[YuvFrame],
    base_q_idx: u8,
    temporal_layers: u8,
) -> Result<TemporalLayeredGopYuv, Error> {
    if !(2..=4).contains(&temporal_layers) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let (width, height) = validate_gop_input(frames)?;
    let n = frames.len();

    // ---- KEY frame + the multi-OP sequence header. ----
    let (key, key_carry) = encode_key_frame_yuv_seg_carry(
        &frames[0],
        base_q_idx,
        RateModel::Twin,
        &[],
        None,
        /* cdef = */ true,
        /* cdef_units = */ true,
        /* lr = */ true,
    )?;
    let mut seq = key.seq.clone();
    seq.operating_points = operating_points_for(seq.operating_points[0], temporal_layers);
    seq.operating_points_cnt_minus_1 = temporal_layers - 1;
    let seq_payload = write_sequence_header_obu(&seq);

    // The KEY temporal unit was emitted with the single-OP header and
    // bare OBU headers — repack it with the §6.7.5 list + the layer-0
    // extension header.
    let key_tu = repack_tu_with_extension(&key.temporal_unit_bytes, 0, Some(&seq_payload));

    // ---- Session state (the §7.20 mirror the pyramid drivers keep). ----
    let key_mi = {
        let fs = key.fh.frame_size.as_ref().expect("KEY builder sizes");
        (fs.mi_rows, fs.mi_cols)
    };
    let key_carry = Rc::new(key_carry);
    let mut mf_store: [SavedMotionField; 8] =
        core::array::from_fn(|_| SavedMotionField::intra(key_mi.0, key_mi.1));
    let mut carry_store: [Rc<RefSlotCarry>; 8] = core::array::from_fn(|_| key_carry.clone());
    let mut slot_hints = [0u32; 8];
    let mut slot_display = [0usize; 8];
    let mut recons: Vec<GopFrameReconYuv> = Vec::with_capacity(n);
    recons.push(GopFrameReconYuv {
        y: key.recon_y,
        u: key.recon_u,
        v: key.recon_v,
    });
    let mut temporal_units: Vec<Vec<u8>> = vec![key_tu];
    let mut temporal_ids: Vec<u8> = vec![0];

    // ---- Display-order layered coding. ----
    #[allow(clippy::needless_range_loop)] // `i` IS the display position / order_hint
    for i in 1..n {
        let tid = temporal_layer_of(i, temporal_layers);
        // LAST = the most recent frame of a layer this frame may
        // reference (see the module docs' reference discipline).
        let last_display = i - (1usize << (temporal_layers - 1 - tid));
        let last_slot = slot_display
            .iter()
            .position(|&d| d == last_display)
            .expect("dyadic ladder keeps every LAST target stored");
        // GOLDEN = the most recent base-layer anchor (slot 0) — a
        // second, longer-range forward reference for the RD ladder.
        let golden_slot = 0usize;
        let mut ref_frame_idx = [last_slot as u8; 7];
        ref_frame_idx[3] = golden_slot as u8;
        let singles: Vec<i8> = if slot_display[golden_slot] == last_display {
            vec![1]
        } else {
            vec![1, 4]
        };

        // Distinct reference reconstructions + the 8-slot map.
        let mut displays: Vec<usize> = Vec::new();
        let mut slot_to_plane = [0usize; 8];
        for (plane, &d) in slot_to_plane.iter_mut().zip(&slot_display) {
            *plane = displays.iter().position(|&x| x == d).unwrap_or_else(|| {
                displays.push(d);
                displays.len() - 1
            });
        }
        let refs: Vec<&GopFrameReconYuv> = displays.iter().map(|&d| &recons[d]).collect();

        // §5.9.2 primary reference: the LAST slot's carry (a lower or
        // equal layer — the CDF chain survives every layer drop), with
        // the per-frame-defaults candidate as exact-bytes alternative.
        let last_carry = carry_store[last_slot].clone();
        let alt_primaries: Vec<(u8, Option<&RefSlotCarry>)> = vec![(PRIMARY_REF_NONE, None)];

        let q = if base_q_idx == 0 {
            0
        } else {
            (i32::from(base_q_idx) + layer_q_off(tid)).clamp(1, 255) as u8
        };
        let cfg = InterFrameConfig {
            order_hint: i as u32,
            show_frame: true,
            refresh_frame_flags: if tid + 1 < temporal_layers {
                1u8 << tid
            } else {
                0
            },
            ref_frame_idx,
            slot_hints,
            single_refs: singles,
            compound_pairs: Vec::new(),
            refs,
            slot_to_plane,
            primary_ref_frame: 0,
            primary_carry: Some(&*last_carry),
            allow_temporal_seg: false,
            alt_primaries,
            exact_mask: None,
            auto_lossless: false,
            seg_extras: None,
            high_precision_mv: true,
            delta_q: true,
            cdef: true,
            cdef_units: true,
            lr: true,
        };
        let (mut obu, recon, saved, carry, _aux) =
            encode_inter_frame_generic(&frames[i], &seq, q, &cfg, &[], &mf_store, RateModel::Twin)?;
        // §5.3.3 / §7.5: the frame OBU carries its layer id.
        obu.header.extension = Some(ObuExtensionHeader::new(tid, 0));
        temporal_units.push(build_temporal_unit(None, &[obu]));
        temporal_ids.push(tid);
        recons.push(recon);
        // §7.20 reference frame update under the slot policy.
        if tid + 1 < temporal_layers {
            let s = usize::from(tid);
            mf_store[s] = saved;
            carry_store[s] = Rc::new(carry);
            slot_hints[s] = i as u32;
            slot_display[s] = i;
        }
    }

    // ---- IVF wrap (one record per temporal unit, display order). ----
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

    Ok(TemporalLayeredGopYuv {
        gop: EncodedGopYuv {
            ivf_bytes,
            temporal_units,
            recon: recons,
            seq,
        },
        temporal_ids,
    })
}

/// 8-bit 4:2:0 entry point of
/// [`encode_temporal_layered_gop_yuv_with_q`].
pub fn encode_temporal_layered_gop_yuv420_with_q(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    temporal_layers: u8,
) -> Result<TemporalLayeredGop, Error> {
    let wide: Vec<YuvFrame> = frames.iter().map(YuvFrame::from_yuv420_8bit).collect();
    let t = encode_temporal_layered_gop_yuv_with_q(&wide, base_q_idx, temporal_layers)?;
    Ok(TemporalLayeredGop {
        gop: narrow_gop_8bit(t.gop),
        temporal_ids: t.temporal_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyadic_ladder_matches_the_module_doc() {
        // L = 3, P = 4: 0 2 1 2 | 0 2 1 2 …
        let expect = [0u8, 2, 1, 2, 0, 2, 1, 2, 0];
        for (i, &e) in expect.iter().enumerate() {
            assert_eq!(temporal_layer_of(i, 3), e, "i = {i}");
        }
        // L = 2, P = 2: 0 1 0 1 …
        for i in 0..8 {
            assert_eq!(temporal_layer_of(i, 2), (i % 2) as u8);
        }
        // L = 4, P = 8: 0 3 2 3 1 3 2 3.
        let expect4 = [0u8, 3, 2, 3, 1, 3, 2, 3];
        for (i, &e) in expect4.iter().enumerate() {
            assert_eq!(temporal_layer_of(i, 4), e, "i = {i}");
        }
    }

    #[test]
    fn operating_point_masks_cover_layer_prefixes() {
        let base = OperatingPoint {
            operating_point_idc: 0,
            seq_level_idx: 0,
            seq_tier: 0,
            decoder_model_present_for_this_op: false,
            operating_parameters_info: None,
            initial_display_delay_present_for_this_op: false,
            initial_display_delay_minus_1: None,
        };
        let ops = operating_points_for(base, 3);
        assert_eq!(ops.len(), 3);
        // op 0 = full: spatial 0 + temporal {0,1,2}.
        assert_eq!(ops[0].operating_point_idc, 0x107);
        assert_eq!(ops[1].operating_point_idc, 0x103);
        assert_eq!(ops[2].operating_point_idc, 0x101);
        // §6.7.5: every idc non-zero (extension headers are coded).
        assert!(ops.iter().all(|o| o.operating_point_idc != 0));
    }
}
