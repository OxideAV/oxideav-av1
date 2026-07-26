//! r430 — §7.3 large-scale-tile CAMERA FRAME write arm.
//!
//! Produces the coded material the §7.3 decode mode consumes: a
//! sequence header inside the §7.3.1 constraint envelope, a camera
//! frame header (INTER, frozen CDFs, `refresh_frame_flags = 0`,
//! LAST-only prediction, zero loop-filter levels, no CDEF / LR /
//! superres / order hints), and the frame's `coded_tile_data` —
//! ready to ride a §5.12 tile-list entry against the anchor it was
//! predicted from.
//!
//! The encode itself is the conformance-grade generic inter driver
//! ([`encode_inter_frame_generic`]) under an order-hint-free
//! sequence: `use_ref_frame_mvs` derives to 0 on both twins, skip
//! mode derives absent, the §7.8 sign biases collapse to 0, and the
//! [`InterFrameConfig::freeze_cdfs`] knob codes the frame with
//! `disable_cdf_update = 1` + `disable_frame_end_update_cdf = 1`.
//! The RD ladder carries LAST alone (`single_refs = [1]`, no
//! compound pairs) so every inter leaf satisfies the §7.3.2
//! `read_ref_frames` conformance requirement (`RefFrame[0] ==
//! LAST_FRAME`, `RefFrame[1] == NONE`; intra leaves stay legal).
//!
//! One tile per frame in this arm: a §7.3-conformant tile is exactly
//! one superblock high with a width that is a whole multiple of the
//! superblock size, so a `W × 64` camera frame (64-pel superblocks,
//! `W <= 4096`) IS its own single tile — the §5.12 tile list then
//! assembles outputs from MANY camera frames / anchors. The
//! multi-tile-per-frame write arm (tile_cols > 1 on the encoder
//! side) is not built yet — see the crate README's frontier notes.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.9.2, §5.12,
//! §6.11, §7.3 (incl. the §7.3.1 constraint list + §7.3.2 decode
//! camera tile process).

use crate::encoder::inter_frame::{
    encode_inter_frame_generic, GopFrameRecon, GopFrameReconYuv, InterFrameConfig, SavedMotionField,
};
use crate::encoder::rate_twin::RateModel;
use crate::encoder::tile_group_obu::parse_tile_group_obu_body;
use crate::encoder::yuv_frame::{
    build_intra_only_yuv420_8bit_fh_with_q, build_intra_only_yuv420_8bit_seq, Yuv420Frame, YuvFrame,
};
use crate::frame_header::{parse_frame_header_with_refs, FrameHeader, RefInfo, PRIMARY_REF_NONE};
use crate::sequence_header::SequenceHeader;
use crate::Error;

/// One encoded §7.3 camera frame.
#[derive(Debug, Clone)]
pub struct CameraFrameEncode {
    /// The §7.3-shaped sequence header the frame was coded under
    /// (identical for every camera frame of the same geometry — the
    /// §7.3 inputs carry ONE sequence header for the whole set).
    pub seq: SequenceHeader,
    /// The camera frame header (identical across same-config camera
    /// frames; the §7.3 inputs carry ONE frame header).
    pub fh: FrameHeader,
    /// The §5.9 frame-header bits as an `OBU_FRAME_HEADER` payload
    /// (§5.3.4 trailer included) — the middle OBU of a §7.3 input
    /// stream.
    pub frame_header_payload: Vec<u8>,
    /// The frame's single tile's coded bytes — a §5.12.2
    /// `coded_tile_data` run.
    pub coded_tile_data: Vec<u8>,
    /// Encoder reconstruction (what the §7.3.2 camera-tile decode
    /// reproduces byte-for-byte; this arm is 8-bit).
    pub recon: GopFrameRecon,
}

/// The §7.3.1-constrained sequence header: 8-bit 4:2:0, 64-pel
/// superblocks, and every sequence-level tool the constraint list
/// bars switched OFF (superres, order hints + their dependents
/// `enable_jnt_comp` / `enable_ref_frame_mvs`, CDEF, loop
/// restoration; film grain / timing / decoder model / initial
/// display delay are already absent from the base builder).
#[must_use]
pub fn build_camera_seq_yuv420_8bit(max_width: u32, max_height: u32) -> SequenceHeader {
    let mut seq = build_intra_only_yuv420_8bit_seq(max_width, max_height);
    // The conformance-grade leaf builders run under the production
    // sequence shape (the KEY driver opens the §5.11.24 filter-intra
    // gate since r410, and intra leaves inside inter frames ride the
    // same picker) — §7.3 does not bar filter intra.
    seq.enable_filter_intra = true;
    seq.enable_order_hint = false;
    seq.order_hint_bits = 0;
    // §5.5.1: both are coded (and inferred 0) only under
    // enable_order_hint == 1.
    seq.enable_jnt_comp = false;
    seq.enable_ref_frame_mvs = false;
    seq.enable_cdef = false;
    seq.enable_restoration = false;
    seq.enable_superres = false;
    seq
}

/// Encode one §7.3 camera frame against `anchor` (the raw anchor
/// pixels — §6.11.2's `AnchorFrames` are "provided by external
/// means", so the prediction source IS the externally agreed anchor
/// plane set, not a decoded stream).
///
/// `input` and `anchor` share one geometry: width a multiple of 64
/// up to 4096, height exactly 64 (one 64-pel superblock — the frame
/// is its own §7.3-conformant tile).
///
/// ## Errors
///
/// [`Error::PartitionWalkOutOfRange`] on a geometry outside the
/// single-tile camera envelope or mismatched input/anchor shapes,
/// plus every generic-inter-driver error surface.
pub fn encode_camera_frame_yuv420(
    input: &Yuv420Frame,
    anchor: &Yuv420Frame,
    base_q_idx: u8,
) -> Result<CameraFrameEncode, Error> {
    if input.width != anchor.width
        || input.height != anchor.height
        || input.height != 64
        || input.width == 0
        || input.width % 64 != 0
        || input.width > 4096
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let wide = YuvFrame::from_yuv420_8bit(input);
    let anchor_recon = GopFrameReconYuv {
        y: anchor.y.iter().map(|&v| u16::from(v)).collect(),
        u: anchor.u.iter().map(|&v| u16::from(v)).collect(),
        v: anchor.v.iter().map(|&v| u16::from(v)).collect(),
    };
    let seq = build_camera_seq_yuv420_8bit(input.width, input.height);
    let mi = {
        let fh0 = build_intra_only_yuv420_8bit_fh_with_q(&seq, input.width, input.height, 0);
        let fs0 = fh0.frame_size.expect("builder always sizes");
        (fs0.mi_rows, fs0.mi_cols)
    };
    let mf_store: [SavedMotionField; 8] =
        core::array::from_fn(|_| SavedMotionField::intra(mi.0, mi.1));
    let cfg = InterFrameConfig {
        order_hint: 0,
        show_frame: true,
        refresh_frame_flags: 0,
        ref_frame_idx: [0u8; 7],
        slot_hints: [0u32; 8],
        single_refs: vec![1],
        compound_pairs: Vec::new(),
        refs: vec![&anchor_recon],
        slot_to_plane: [0usize; 8],
        primary_ref_frame: PRIMARY_REF_NONE,
        primary_carry: None,
        allow_temporal_seg: false,
        alt_primaries: Vec::new(),
        exact_mask: None,
        auto_lossless: false,
        seg_extras: None,
        high_precision_mv: false,
        delta_q: false,
        cdef: false,
        cdef_units: false,
        lr: false,
        freeze_cdfs: true,
    };
    let (obu, recon, _saved, _carry, _aux) = encode_inter_frame_generic(
        &wide,
        &seq,
        base_q_idx,
        &cfg,
        &[],
        &mf_store,
        RateModel::Twin,
    )?;

    // Split the §5.10 OBU_FRAME body back into (frame header, tile
    // bytes): the header ends at the §5.10 byte_alignment(), the tile
    // group body follows, and the single-tile §5.11.1 parse recovers
    // the raw coded_tile_data.
    let fh = parse_frame_header_with_refs(&obu.body, &seq, &RefInfo::default())?;
    let tg_offset = fh.bits_consumed.div_ceil(8);
    if tg_offset > obu.body.len() {
        return Err(Error::UnexpectedEnd);
    }
    let ti = fh
        .tile_info
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let parsed = parse_tile_group_obu_body(
        &obu.body[tg_offset..],
        ti.tile_cols * ti.tile_rows,
        0,
        0,
        u32::from(ti.tile_size_bytes),
    )?;
    let coded_tile_data = parsed
        .tiles
        .first()
        .ok_or(Error::PartitionWalkOutOfRange)?
        .bytes
        .to_vec();
    let frame_header_payload = crate::encoder::frame_obu::write_frame_header_obu(&fh, &seq);
    let narrow = |p: Vec<u16>| p.into_iter().map(|s| s as u8).collect::<Vec<u8>>();
    Ok(CameraFrameEncode {
        seq,
        fh,
        frame_header_payload,
        coded_tile_data,
        recon: GopFrameRecon {
            y: narrow(recon.y),
            u: narrow(recon.u),
            v: narrow(recon.v),
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_seq_satisfies_the_7_3_sequence_constraints() {
        let seq = build_camera_seq_yuv420_8bit(128, 64);
        assert!(!seq.enable_superres);
        assert!(!seq.enable_order_hint);
        assert!(!seq.still_picture);
        assert!(!seq.film_grain_params_present);
        assert!(!seq.timing_info_present_flag);
        assert!(!seq.decoder_model_info_present_flag);
        assert!(!seq.initial_display_delay_present_flag);
        assert!(!seq.enable_restoration);
        assert!(!seq.enable_cdef);
        assert!(!seq.color_config.mono_chrome);
        assert!(!seq.enable_jnt_comp);
        assert!(!seq.enable_ref_frame_mvs);
    }

    #[test]
    fn camera_geometry_envelope_is_enforced() {
        let f = Yuv420Frame::filled(64, 64, 100);
        let tall = Yuv420Frame::filled(64, 128, 100);
        assert!(encode_camera_frame_yuv420(&f, &tall, 60).is_err());
        assert!(encode_camera_frame_yuv420(&tall, &tall, 60).is_err());
    }
}
