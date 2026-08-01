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
use crate::encoder::key_frame::encode_key_frame_yuv_seg_carry_tiles;
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
    encode_temporal_layered_gop_yuv_with_q_tiles(frames, base_q_idx, temporal_layers, (0, 0), 1)
}

/// r433 — [`encode_temporal_layered_gop_yuv_with_q`] with a §5.9.15
/// uniform tile layout (`(TileColsLog2, TileRowsLog2)`, coded on
/// every frame of the ladder) and §5.11.1 tile-group packaging
/// (`tile_groups > 1` splits each frame's tiles across that many
/// `OBU_TILE_GROUP` OBUs behind a standalone `OBU_FRAME_HEADER`; the
/// §5.3.3 extension header rides EVERY frame OBU either way, per
/// §7.5). `(0, 0)` / `1` reproduce the unlayered-tile stream bit for
/// bit.
pub fn encode_temporal_layered_gop_yuv_with_q_tiles(
    frames: &[YuvFrame],
    base_q_idx: u8,
    temporal_layers: u8,
    tiles: (u32, u32),
    tile_groups: u32,
) -> Result<TemporalLayeredGopYuv, Error> {
    if !(2..=4).contains(&temporal_layers) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let (width, height) = validate_gop_input(frames)?;
    let n = frames.len();
    // r436 — the §6.8.14 donor election arms on multi-tile ladders
    // (single-tile frames carry no `context_update_tile_id` field).
    let donor_armed = tiles != (0, 0);

    // ---- KEY frame + the multi-OP sequence header. ----
    let (key, key_carry) = encode_key_frame_yuv_seg_carry_tiles(
        &frames[0],
        base_q_idx,
        RateModel::Twin,
        &[],
        None,
        /* cdef = */ true,
        /* cdef_units = */ true,
        /* lr = */ true,
        // r433 — the ladder-wide tile layout + tile-group packaging.
        tiles,
        tile_groups,
        None,
        /* delta_q = */ true,
        donor_armed,
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
    // r436 — which temporal unit each slot's carried frame lives in
    // (the donor patch rewrites that unit's header field in place).
    let mut carry_tu: [usize; 8] = [0; 8];
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
            freeze_cdfs: false,
            tiles,
            tile_groups,
            // r436 — collect only on REFERENCE frames (a top-layer
            // frame's donation is never consumed); elect whenever the
            // primary carry offers candidates.
            collect_donor_cdfs: donor_armed && tid + 1 < temporal_layers,
            elect_donor: donor_armed,
            explicit_tiles: None,
        };
        let (mut obus, recon, saved, carry, aux) =
            encode_inter_frame_generic(&frames[i], &seq, q, &cfg, &[], &mf_store, RateModel::Twin)?;
        // r436 — §6.8.14 donor settlement. THE MULTI-CONSUMER RULE:
        // a slot's donor set is frozen at its FIRST consumption —
        // whether or not the election won — because this frame's
        // bytes were committed under the donation as it stood; a
        // LATER consumer re-electing would silently re-start THIS
        // frame's already-emitted CDF chain. On a win the primary
        // frame's already-emitted `context_update_tile_id` is
        // patched in place and its stored carry takes the elected
        // tile's state; either way the donor set clears, and every
        // §7.20 slot holding the same frame's carry (the KEY seeds
        // all eight) is swept by pointer identity.
        if carry_store[last_slot].donor_cdfs.len() > 1 {
            let consumed = carry_store[last_slot].clone();
            let fixed_cdfs = match aux.donor_elected {
                Some(t) => {
                    let span = consumed
                        .ctx_update_span
                        .expect("donor election only fires on multi-tile primaries");
                    crate::encoder::inter_frame::patch_ctx_update_in_tu(
                        &mut temporal_units[carry_tu[last_slot]],
                        span,
                        t,
                    );
                    consumed.donor_cdfs[t as usize].clone()
                }
                None => consumed.cdfs.clone(),
            };
            let fixed = Rc::new(RefSlotCarry {
                cdfs: fixed_cdfs,
                segment_ids: consumed.segment_ids.clone(),
                mi_rows: consumed.mi_rows,
                mi_cols: consumed.mi_cols,
                gm_params: consumed.gm_params,
                donor_cdfs: Vec::new(),
                ctx_update_span: None,
            });
            for slot in carry_store.iter_mut() {
                if Rc::ptr_eq(slot, &consumed) {
                    *slot = fixed.clone();
                }
            }
        }
        // §5.3.3 / §7.5: every OBU of the frame carries its layer id.
        for obu in &mut obus {
            obu.header.extension = Some(ObuExtensionHeader::new(tid, 0));
        }
        temporal_units.push(build_temporal_unit(None, &obus));
        temporal_ids.push(tid);
        recons.push(recon);
        // §7.20 reference frame update under the slot policy.
        if tid + 1 < temporal_layers {
            let s = usize::from(tid);
            mf_store[s] = saved;
            carry_store[s] = Rc::new(carry);
            carry_tu[s] = i;
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

// ---------------------------------------------------------------------
// r431 — SPATIAL scalability: independently-coded spatial layers.
// ---------------------------------------------------------------------

/// r431 — an encoded spatially scalable stream (see
/// [`encode_spatial_layered_gop_yuv_with_q`]).
#[derive(Debug, Clone)]
pub struct SpatialLayeredGopYuv {
    /// Complete IVF v0 file (one record per §7.5 temporal unit).
    pub ivf_bytes: Vec<u8>,
    /// The bare §7.5 temporal units, one per time instant — each
    /// carries every spatial layer's frame OBU in increasing
    /// `spatial_id` order (unit 0 also carries the shared sequence
    /// header).
    pub temporal_units: Vec<Vec<u8>>,
    /// The shared sequence header (top-layer dimension budget +
    /// the §6.7.5 spatial operating-point list).
    pub seq: crate::sequence_header::SequenceHeader,
    /// `layer_recons[ s ][ i ]` — layer `s`'s reconstruction of time
    /// instant `i`, at that layer's own dimensions.
    pub layer_recons: Vec<Vec<GopFrameReconYuv>>,
    /// Per-layer `(width, height)`.
    pub layer_dims: Vec<(u32, u32)>,
}

/// 8-bit 4:2:0 sibling of [`SpatialLayeredGopYuv`].
#[derive(Debug, Clone)]
pub struct SpatialLayeredGop {
    pub ivf_bytes: Vec<u8>,
    pub temporal_units: Vec<Vec<u8>>,
    pub seq: crate::sequence_header::SequenceHeader,
    /// `layer_recons[ s ][ i ]`, 8-bit planes.
    pub layer_recons: Vec<Vec<crate::encoder::inter_frame::GopFrameRecon>>,
    pub layer_dims: Vec<(u32, u32)>,
}

/// The §6.7.5 operating-point list for `s_count` INDEPENDENTLY CODED
/// spatial layers (one temporal layer): entry `k` selects spatial
/// layers `0..=s_count-1-k` — nested prefixes, so every §6.7.5
/// sub-bitstream still begins with the base layer's KEY frame (an
/// operating point excluding layer 0 would start on an
/// `INTRA_ONLY` frame, which cannot begin a coded video sequence).
fn spatial_operating_points_for(base: OperatingPoint, s_count: u8) -> Vec<OperatingPoint> {
    (0..s_count)
        .map(|k| OperatingPoint {
            operating_point_idc: (((1u16 << (s_count - k)) - 1) << 8) | 1,
            ..base
        })
        .collect()
}

/// Encode `layers.len()` INDEPENDENTLY CODED spatial layers (2..=4;
/// `layers[ s ][ i ]` is layer `s`'s frame at time instant `i`, all
/// layers the same length, the LAST layer the largest — its
/// dimensions size the shared sequence header) into one §6.7.5
/// spatially scalable stream:
///
///   * ONE sequence header (top-layer dimension budget); smaller
///     layers code §5.9.5 `frame_size_override_flag = 1` explicit
///     dimensions (inter frames ride the §5.9.7 no-found-ref arm).
///   * §5.3.3 extension headers on every frame OBU (`temporal_id =
///     0`, `spatial_id = s`); each §7.5 temporal unit carries every
///     layer's frame for that instant in increasing `spatial_id`
///     order, one shown frame per layer (the §7.5 layered-stream
///     rules hold by construction).
///   * Layer 0 opens with the ONLY KEY frame (its `allFrames`
///     refresh seeds every §7.20 slot); each enhancement layer opens
///     with a §5.9.2 `INTRA_ONLY` frame refreshing ONLY its own two
///     slots (`0b11 << 2s`) — sibling layers' reference state
///     survives. Inter frames predict LAST-only inside their own
///     layer's slot pair (`2s` / `2s + 1` rotation) with the
///     §8.3.1 primary-reference CDF chain riding the same pair, so
///     layers stay fully independent: dropping any spatial-layer
///     SUFFIX (the §6.7.5 nested operating points) leaves every
///     surviving frame bit-identical.
///
/// Decoding at operating point `k`
/// ([`crate::decode_av1_at_operating_point`]) yields the shown
/// frames of layers `0..=layers.len()-1-k`, interleaved in decode
/// order within each temporal unit, each byte-identical to
/// `layer_recons[ s ][ i ]`.
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] on: layer count outside 2..=4,
/// unequal layer lengths / empty layers / over-`GOP_MAX_FRAMES`,
/// mixed bit depths or chroma formats across layers, a layer
/// exceeding the top layer's dimensions, or any per-frame encoder
/// reject.
pub fn encode_spatial_layered_gop_yuv_with_q(
    layers: &[Vec<YuvFrame>],
    base_q_idx: u8,
) -> Result<SpatialLayeredGopYuv, Error> {
    encode_spatial_layered_gop_yuv_with_q_tiles(layers, base_q_idx, None, 1)
}

/// r436 — [`encode_spatial_layered_gop_yuv_with_q`] with PER-LAYER
/// §5.9.15 uniform tile layouts and §5.11.1 tile-group packaging.
///
/// `layer_tiles[ s ] = (TileColsLog2, TileRowsLog2)` is layer `s`'s
/// OWN tile layout, coded on every frame of that layer (the KEY /
/// `INTRA_ONLY` opener and every inter frame). Each layout must sit
/// inside the §5.9.15 legal window FOR THAT LAYER'S dimensions (the
/// per-layer legality windows: a 64×64 base layer admits only
/// `(0, 0)` while a 256×256 enhancement layer admits up to
/// `(2, 2)` — the layouts are validated independently per layer,
/// exactly like the single-layer drivers validate theirs).
/// `None` (or all-`(0, 0)`) reproduces the untiled spatial stream
/// bit for bit.
///
/// `tile_groups > 1` splits EVERY frame whose realized tile count
/// allows it across that many `OBU_TILE_GROUP` OBUs behind a
/// standalone `OBU_FRAME_HEADER` (clamping per frame to its layer's
/// tile count, so a single-tile base layer keeps the §5.10
/// `OBU_FRAME` packing while a tiled enhancement layer splits). The
/// §5.3.3 extension header rides EVERY frame-carrying OBU of a
/// layer's frame — `OBU_FRAME`, `OBU_FRAME_HEADER` and
/// `OBU_TILE_GROUP` alike — per the §7.5 layered-stream rule.
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] on every
/// [`encode_spatial_layered_gop_yuv_with_q`] reject, on
/// `layer_tiles` whose length differs from `layers.len()`, or on a
/// layout outside its own layer's §5.9.15 legal window.
pub fn encode_spatial_layered_gop_yuv_with_q_tiles(
    layers: &[Vec<YuvFrame>],
    base_q_idx: u8,
    layer_tiles: Option<&[(u32, u32)]>,
    tile_groups: u32,
) -> Result<SpatialLayeredGopYuv, Error> {
    let s_count = layers.len();
    if !(2..=4).contains(&s_count) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    if layer_tiles.is_some_and(|t| t.len() != s_count) {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let tiles_of = |s: usize| layer_tiles.map_or((0, 0), |t| t[s]);
    let n = layers[0].len();
    if n == 0 || n > crate::encoder::inter_frame::GOP_MAX_FRAMES {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let (bit_depth, format) = (layers[0][0].bit_depth, layers[0][0].format);
    for layer in layers {
        if layer.len() != n {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let (w, h) = (layer[0].width, layer[0].height);
        for f in layer {
            if f.width != w || f.height != h || f.bit_depth != bit_depth || f.format != format {
                return Err(Error::PartitionWalkOutOfRange);
            }
            f.validate()?;
        }
    }
    let (top_w, top_h) = (layers[s_count - 1][0].width, layers[s_count - 1][0].height);
    if layers
        .iter()
        .any(|l| l[0].width > top_w || l[0].height > top_h)
    {
        return Err(Error::PartitionWalkOutOfRange);
    }

    // ---- The shared sequence header. ----
    let mut seq =
        crate::encoder::yuv_frame::build_intra_only_seq_yuv(top_w, top_h, bit_depth, format)?;
    // The same tool gates the per-frame drivers assume (see the KEY
    // driver's internal builder).
    seq.enable_filter_intra = true;
    seq.operating_points = spatial_operating_points_for(seq.operating_points[0], s_count as u8);
    seq.operating_points_cnt_minus_1 = (s_count - 1) as u8;
    let seq_payload = write_sequence_header_obu(&seq);

    // ---- Session state: §7.20 slots partitioned two per layer. ----
    let mut layer_recons: Vec<Vec<GopFrameReconYuv>> = vec![Vec::with_capacity(n); s_count];
    let mut mf_store: Vec<SavedMotionField> = Vec::new();
    let mut carry_store: Vec<Option<Rc<RefSlotCarry>>> = vec![None; 8];
    let mut slot_hints = [0u32; 8];
    let mut temporal_units: Vec<Vec<u8>> = Vec::with_capacity(n);

    // ---- Time instant 0: the layer-0 KEY + enhancement INTRA_ONLYs. ----
    let mut tu0: Vec<u8> = Vec::new();
    write_obu_with_size(
        &mut tu0,
        &crate::encoder::obu::ObuHeader::new(ObuType::TemporalDelimiter),
        &[],
    );
    write_obu_with_size(
        &mut tu0,
        &crate::encoder::obu::ObuHeader::new(ObuType::SequenceHeader),
        &seq_payload,
    );
    for (s, layer) in layers.iter().enumerate() {
        let extras = crate::encoder::key_frame::KeyExtras {
            // r436 — the layer's OWN §5.9.15 layout + §5.11.1
            // packaging (validated against ITS dimensions).
            tiles: tiles_of(s),
            tile_groups,
            explicit_tiles: None,
            seq_override: Some(&seq),
            // Layer 0: a true KEY (refreshes ALL slots — §5.9.2
            // derives allFrames); enhancement layers: INTRA_ONLY
            // refreshing only their own pair.
            intra_only_refresh: (s > 0).then_some(0b11u8 << (2 * s)),
            // r431 — the same §5.9.17 delta-q election as every
            // default intra entry.
            delta_q: true,
            delta_plan: None,
            collect_donor_cdfs: false,
        };
        let (k, carry) = crate::encoder::key_frame::encode_key_frame_yuv_full(
            &layer[0],
            base_q_idx,
            RateModel::Twin,
            &[],
            None,
            true,
            true,
            true,
            &extras,
        )?;
        let fs = k.fh.frame_size.as_ref().expect("intra driver sizes");
        let (mi_rows, mi_cols) = (fs.mi_rows, fs.mi_cols);
        let carry = Rc::new(carry);
        if s == 0 {
            // §7.20 allFrames refresh: every slot takes the layer-0
            // payload (the enhancement intras then overwrite theirs).
            mf_store = (0..8)
                .map(|_| SavedMotionField::intra(mi_rows, mi_cols))
                .collect();
            for c in carry_store.iter_mut() {
                *c = Some(carry.clone());
            }
            slot_hints = [0; 8];
        } else {
            for b in 0..2usize {
                mf_store[2 * s + b] = SavedMotionField::intra(mi_rows, mi_cols);
                carry_store[2 * s + b] = Some(carry.clone());
                slot_hints[2 * s + b] = 0;
            }
        }
        // Extract the frame-carrying OBUs from the driver's own
        // temporal unit and re-wrap each with the §5.3.3 extension
        // header (r436: with `tile_groups > 1` the intra driver
        // emits `OBU_FRAME_HEADER` + N `OBU_TILE_GROUP` OBUs instead
        // of one `OBU_FRAME` — §7.5 requires the extension header on
        // ALL of them).
        let mut wrote_frame = false;
        for desc in ObuIter::new(&k.temporal_unit_bytes) {
            let desc = desc.expect("own temporal unit walks");
            match desc.obu_type {
                ObuType::TemporalDelimiter | ObuType::SequenceHeader => {}
                other => {
                    let header = crate::encoder::obu::ObuHeader::new(other)
                        .with_extension(ObuExtensionHeader::new(0, s as u8));
                    write_obu_with_size(&mut tu0, &header, desc.payload);
                    wrote_frame = true;
                }
            }
        }
        if !wrote_frame {
            return Err(Error::PartitionWalkOutOfRange);
        }
        layer_recons[s].push(GopFrameReconYuv {
            y: k.recon_y,
            u: k.recon_u,
            v: k.recon_v,
        });
    }
    temporal_units.push(tu0);

    // ---- Time instants 1..n: per-layer LAST-only inter frames. ----
    for i in 1..n {
        let mut tu: Vec<u8> = Vec::new();
        write_obu_with_size(
            &mut tu,
            &crate::encoder::obu::ObuHeader::new(ObuType::TemporalDelimiter),
            &[],
        );
        for (s, layer) in layers.iter().enumerate() {
            // Own-layer slot rotation: frame i-1 sits in slot
            // `2s + ((i-1) & 1)`; this frame refreshes the other one.
            let last_slot = 2 * s + ((i - 1) & 1);
            let refresh_slot = 2 * s + (i & 1);
            let prev = layer_recons[s].last().expect("instant i-1 encoded").clone();
            let last_carry = carry_store[last_slot]
                .clone()
                .expect("layer slots seeded at instant 0");
            let mf: [SavedMotionField; 8] = core::array::from_fn(|k| mf_store[k].clone());
            let cfg = InterFrameConfig {
                order_hint: i as u32,
                show_frame: true,
                refresh_frame_flags: 1u8 << refresh_slot,
                ref_frame_idx: [last_slot as u8; 7],
                slot_hints,
                single_refs: vec![1],
                compound_pairs: Vec::new(),
                refs: vec![&prev],
                slot_to_plane: [0usize; 8],
                primary_ref_frame: 0,
                primary_carry: Some(&last_carry),
                allow_temporal_seg: false,
                alt_primaries: vec![(PRIMARY_REF_NONE, None)],
                exact_mask: None,
                auto_lossless: false,
                seg_extras: None,
                high_precision_mv: true,
                delta_q: true,
                cdef: true,
                cdef_units: true,
                lr: true,
                freeze_cdfs: false,
                // r436 — the layer's own layout on every inter frame.
                tiles: tiles_of(s),
                tile_groups,
                collect_donor_cdfs: false,
                elect_donor: false,
                explicit_tiles: None,
            };
            let (obus, recon, saved, carry, _aux) = encode_inter_frame_generic(
                &layer[i],
                &seq,
                base_q_idx,
                &cfg,
                &[],
                &mf,
                RateModel::Twin,
            )?;
            for mut obu in obus {
                obu.header.extension = Some(ObuExtensionHeader::new(0, s as u8));
                write_obu_with_size(&mut tu, &obu.header, &obu.body);
            }
            layer_recons[s].push(recon);
            mf_store[refresh_slot] = saved;
            carry_store[refresh_slot] = Some(Rc::new(carry));
            slot_hints[refresh_slot] = i as u32;
        }
        temporal_units.push(tu);
    }

    // ---- IVF wrap (top-layer dimensions). ----
    let mut ivf_bytes: Vec<u8> = Vec::new();
    {
        let cursor = std::io::Cursor::new(&mut ivf_bytes);
        let mut iw = IvfWriter::new(cursor, FOURCC_AV01, top_w as u16, top_h as u16, 25, 1)
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
        for (idx, tu) in temporal_units.iter().enumerate() {
            iw.write_frame(tu, idx as u64)
                .map_err(|_| Error::PartitionWalkOutOfRange)?;
        }
        iw.patch_frame_count()
            .map_err(|_| Error::PartitionWalkOutOfRange)?;
    }

    Ok(SpatialLayeredGopYuv {
        ivf_bytes,
        temporal_units,
        seq,
        layer_recons,
        layer_dims: layers.iter().map(|l| (l[0].width, l[0].height)).collect(),
    })
}

/// 8-bit 4:2:0 entry point of
/// [`encode_spatial_layered_gop_yuv_with_q`].
pub fn encode_spatial_layered_gop_yuv420_with_q(
    layers: &[Vec<Yuv420Frame>],
    base_q_idx: u8,
) -> Result<SpatialLayeredGop, Error> {
    encode_spatial_layered_gop_yuv420_with_q_tiles(layers, base_q_idx, None, 1)
}

/// 8-bit 4:2:0 entry point of
/// [`encode_spatial_layered_gop_yuv_with_q_tiles`].
pub fn encode_spatial_layered_gop_yuv420_with_q_tiles(
    layers: &[Vec<Yuv420Frame>],
    base_q_idx: u8,
    layer_tiles: Option<&[(u32, u32)]>,
    tile_groups: u32,
) -> Result<SpatialLayeredGop, Error> {
    let wide: Vec<Vec<YuvFrame>> = layers
        .iter()
        .map(|l| l.iter().map(YuvFrame::from_yuv420_8bit).collect())
        .collect();
    let s =
        encode_spatial_layered_gop_yuv_with_q_tiles(&wide, base_q_idx, layer_tiles, tile_groups)?;
    let narrow = |p: &[u16]| p.iter().map(|&v| v as u8).collect::<Vec<u8>>();
    Ok(SpatialLayeredGop {
        ivf_bytes: s.ivf_bytes,
        temporal_units: s.temporal_units,
        seq: s.seq,
        layer_recons: s
            .layer_recons
            .iter()
            .map(|lr| {
                lr.iter()
                    .map(|r| crate::encoder::inter_frame::GopFrameRecon {
                        y: narrow(&r.y),
                        u: narrow(&r.u),
                        v: narrow(&r.v),
                    })
                    .collect()
            })
            .collect(),
        layer_dims: s.layer_dims,
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

/// 8-bit 4:2:0 sibling of
/// [`encode_temporal_layered_gop_yuv_with_q_tiles`].
pub fn encode_temporal_layered_gop_yuv420_with_q_tiles(
    frames: &[Yuv420Frame],
    base_q_idx: u8,
    temporal_layers: u8,
    tiles: (u32, u32),
    tile_groups: u32,
) -> Result<TemporalLayeredGop, Error> {
    let wide: Vec<YuvFrame> = frames.iter().map(YuvFrame::from_yuv420_8bit).collect();
    let t = encode_temporal_layered_gop_yuv_with_q_tiles(
        &wide,
        base_q_idx,
        temporal_layers,
        tiles,
        tile_groups,
    )?;
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
