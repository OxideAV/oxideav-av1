//! §5.12 tile-list OBU + §7.3 large-scale-tile decoding.
//!
//! AV1's second operating mode (§7.1): instead of a sequence of
//! temporal units, the input is a **tile list OBU** plus side
//! information — the parsed sequence-header and frame-header state
//! and an `AnchorFrames` array "provided by external means"
//! (§6.11.2) — and the output is ONE frame assembled from
//! independently coded camera tiles.
//!
//! * [`parse_tile_list_obu`] — the §5.12.1/§5.12.2 syntax
//!   (`output_frame_width/height_in_tiles_minus_1`,
//!   `tile_count_minus_1`, per-entry `anchor_frame_idx` /
//!   `anchor_tile_row` / `anchor_tile_col` / `coded_tile_data`) with
//!   the §6.11 conformance bounds enforced
//!   (`tile_count_minus_1 <= 511`, `anchor_frame_idx <= 127`).
//! * [`write_tile_list_obu`] — the byte-exact inverse (the OBU body;
//!   §5.3.1 exempts `OBU_TILE_LIST` from `trailing_bits`, so the body
//!   IS the payload).
//! * [`decode_tile_list`] — the §7.3.1 ordered steps: per entry, set
//!   `FrameStore[ ref_frame_idx[ 0 ] ]` to the selected anchor,
//!   invoke the §7.3.2 decode-camera-tile process (a §5.11.2 tile
//!   decode with a fresh symbol decoder over `coded_tile_data`, no
//!   post-processing, no reference update), and write the decoded
//!   tile into the output frame in raster order at
//!   `destX/destY = TileWidth/Height * (tile % / (owidth_in_tiles))`.
//!   The §7.3.1 bitstream-conformance constraint list (superres /
//!   order hints / CDEF / restoration / film grain OFF, INTER frame,
//!   frozen CDFs, superblock-high uniform tiles, zero loop-filter
//!   levels, …) gates the whole mode: inputs outside it surface
//!   [`Error::TileListInvalid`].
//! * [`decode_tile_list_stream`] — convenience walker over a §5.2
//!   OBU concatenation (`OBU_SEQUENCE_HEADER` + `OBU_FRAME_HEADER` +
//!   `OBU_TILE_LIST`), the natural byte-level packaging of the §7.3
//!   inputs.
//!
//! Anchors are supplied as decoded frames ([`SpecFrame`]) — pixels
//! only. §7.3's input list optionally includes CDF tables loaded from
//! a reference frame; a camera frame header naming a primary
//! reference therefore cannot be decoded from these inputs and is
//! rejected ([`Error::TileListInvalid`]).
//!
//! In the GENERAL decoding mode (§7.2), tile-list OBUs are skipped by
//! [`crate::decoder::SpecDecodeSession`] — §7.3.1: "a decoder is
//! recommended to support decoding of tile list OBUs, but this is not
//! a requirement for decoder conformance", and the anchor array only
//! exists by external means.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.12, §6.11, §7.1,
//! §7.3 (incl. §7.3.2 decode camera tile process).

use crate::cdf::{
    FrameInterOrderHints, InterFrameContext, InterWalkPixels, MotionFieldMvs, PartitionWalker,
    QuantizerParams, TileCdfContext, TileDecodeParams, TileGeometry,
};
use crate::decoder::frame_driver::lossless_array;
use crate::decoder::SpecFrame;
use crate::frame_header::{
    parse_frame_header_with_refs, FrameHeader, FrameType, RefInfo, NUM_REF_FRAMES, PRIMARY_REF_NONE,
};
use crate::obu::{ObuIter, ObuType};
use crate::sequence_header::{parse_sequence_header, SequenceHeader};
use crate::symbol_decoder::SymbolDecoder;
use crate::uncompressed_header_tail::{SegmentationParams, MAX_SEGMENTS, SEG_LVL_ALT_Q};
use crate::Error;
use crate::{PlaneRefSpec, RefFrameStoreEntry};

/// One §5.12.2 `tile_list_entry`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileListEntry {
    /// Index into the externally supplied `AnchorFrames` array
    /// (§6.11.2; conformance bound `<= 127`).
    pub anchor_frame_idx: u8,
    /// Tile-grid row of this tile inside the camera frame
    /// (conformance: `< TileRows`).
    pub anchor_tile_row: u8,
    /// Tile-grid column (conformance: `< TileCols`).
    pub anchor_tile_col: u8,
    /// The `tile_data_size_minus_1 + 1` coded tile bytes.
    pub coded_tile_data: Vec<u8>,
}

/// A parsed §5.12.1 `tile_list_obu`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileListObu {
    /// Output-frame width in tile units, minus one.
    pub output_frame_width_in_tiles_minus_1: u8,
    /// Output-frame height in tile units, minus one.
    pub output_frame_height_in_tiles_minus_1: u8,
    /// The `tile_count_minus_1 + 1` entries, in output raster order.
    pub entries: Vec<TileListEntry>,
}

/// Parse a §5.12.1 tile-list OBU body.
///
/// ## Errors
///
/// * [`Error::UnexpectedEnd`] — the payload ends inside a fixed field
///   or inside an entry's declared `coded_tile_data`.
/// * [`Error::TileListInvalid`] — `tile_count_minus_1 > 511` (§6.11.1),
///   `anchor_frame_idx > 127` (§6.11.2), or bytes left over after the
///   last entry (§5.3.1 gives `OBU_TILE_LIST` no trailing bits).
pub fn parse_tile_list_obu(payload: &[u8]) -> Result<TileListObu, Error> {
    if payload.len() < 4 {
        return Err(Error::UnexpectedEnd);
    }
    let output_frame_width_in_tiles_minus_1 = payload[0];
    let output_frame_height_in_tiles_minus_1 = payload[1];
    let tile_count_minus_1 = u16::from_be_bytes([payload[2], payload[3]]);
    if tile_count_minus_1 > 511 {
        return Err(Error::TileListInvalid);
    }
    let mut cursor = 4usize;
    let mut entries = Vec::with_capacity(usize::from(tile_count_minus_1) + 1);
    for _ in 0..=tile_count_minus_1 {
        let header = payload
            .get(cursor..cursor + 5)
            .ok_or(Error::UnexpectedEnd)?;
        let anchor_frame_idx = header[0];
        if anchor_frame_idx > 127 {
            return Err(Error::TileListInvalid);
        }
        let anchor_tile_row = header[1];
        let anchor_tile_col = header[2];
        let n = usize::from(u16::from_be_bytes([header[3], header[4]])) + 1;
        cursor += 5;
        let data = payload
            .get(cursor..cursor + n)
            .ok_or(Error::UnexpectedEnd)?;
        cursor += n;
        entries.push(TileListEntry {
            anchor_frame_idx,
            anchor_tile_row,
            anchor_tile_col,
            coded_tile_data: data.to_vec(),
        });
    }
    if cursor != payload.len() {
        return Err(Error::TileListInvalid);
    }
    Ok(TileListObu {
        output_frame_width_in_tiles_minus_1,
        output_frame_height_in_tiles_minus_1,
        entries,
    })
}

/// Serialize a [`TileListObu`] back to the §5.12.1 body bytes — the
/// byte-exact inverse of [`parse_tile_list_obu`].
///
/// ## Errors
///
/// [`Error::TileListInvalid`] — no entries, more than 512 entries
/// (§6.11.1), `anchor_frame_idx > 127`, or an entry whose
/// `coded_tile_data` is empty or longer than 65536 bytes (the
/// `f(16)` `tile_data_size_minus_1` envelope).
pub fn write_tile_list_obu(tl: &TileListObu) -> Result<Vec<u8>, Error> {
    if tl.entries.is_empty() || tl.entries.len() > 512 {
        return Err(Error::TileListInvalid);
    }
    let mut out = Vec::new();
    out.push(tl.output_frame_width_in_tiles_minus_1);
    out.push(tl.output_frame_height_in_tiles_minus_1);
    out.extend_from_slice(&((tl.entries.len() - 1) as u16).to_be_bytes());
    for e in &tl.entries {
        if e.anchor_frame_idx > 127
            || e.coded_tile_data.is_empty()
            || e.coded_tile_data.len() > 65536
        {
            return Err(Error::TileListInvalid);
        }
        out.push(e.anchor_frame_idx);
        out.push(e.anchor_tile_row);
        out.push(e.anchor_tile_col);
        out.extend_from_slice(&((e.coded_tile_data.len() - 1) as u16).to_be_bytes());
        out.extend_from_slice(&e.coded_tile_data);
    }
    Ok(out)
}

/// The camera frame's uniform tile geometry, validated per §7.3.1.
struct LstGeometry {
    /// Tile width in pixels (identical for every tile column).
    tile_width: u32,
    /// Tile height in pixels (exactly one superblock).
    tile_height: u32,
}

/// Enforce the §7.3.1 bitstream-conformance constraint list on the
/// sequence + frame header pair (the "large-scale-tile gate"), plus
/// the two input-contract requirements of this API (no primary
/// reference — anchors carry pixels only; spatial segment map when
/// segmentation is on).
fn validate_lst_shape(seq: &SequenceHeader, fh: &FrameHeader) -> Result<LstGeometry, Error> {
    // ---- Sequence-level constraints. ----
    if seq.enable_superres
        || seq.enable_order_hint
        || seq.still_picture
        || seq.film_grain_params_present
        || seq.timing_info_present_flag
        || seq.decoder_model_info_present_flag
        || seq.initial_display_delay_present_flag
        || seq.enable_restoration
        || seq.enable_cdef
        || seq.color_config.mono_chrome
    {
        return Err(Error::TileListInvalid);
    }
    // ---- Frame-level constraints. ----
    if fh.show_existing_frame
        || fh.frame_type != FrameType::Inter
        || !fh.show_frame
        || fh.error_resilient_mode
        || !fh.disable_cdf_update
        || !fh.disable_frame_end_update_cdf
        || fh.frame_size_override_flag
        || fh.refresh_frame_flags != 0
        || fh.allow_intrabc
    {
        return Err(Error::TileListInvalid);
    }
    let ir = fh.inter_refs.as_ref().ok_or(Error::TileListInvalid)?;
    if ir.use_ref_frame_mvs || fh.reference_select != Some(false) {
        return Err(Error::TileListInvalid);
    }
    if fh.delta_q_params.map(|d| d.delta_q_present) != Some(false)
        || fh.delta_lf_params.map(|d| d.delta_lf_present) != Some(false)
    {
        return Err(Error::TileListInvalid);
    }
    let lf = fh
        .loop_filter_params
        .as_ref()
        .ok_or(Error::TileListInvalid)?;
    if lf.loop_filter_level[0] != 0 || lf.loop_filter_level[1] != 0 {
        return Err(Error::TileListInvalid);
    }
    if let Some(sp) = fh.segmentation_params.as_ref() {
        if sp.temporal_update {
            return Err(Error::TileListInvalid);
        }
        // Input contract: a predicted segment map needs the previous
        // frame's SegmentIds — not part of the §7.3 inputs.
        if sp.enabled && !sp.update_map {
            return Err(Error::TileListInvalid);
        }
    }
    // Input contract: anchors carry pixels only, so a header that
    // loads a reference's saved CDF state cannot be honoured.
    if fh.primary_ref_frame != PRIMARY_REF_NONE {
        return Err(Error::TileListInvalid);
    }
    // ---- Geometry constraints. ----
    let fs = fh.frame_size.as_ref().ok_or(Error::TileListInvalid)?;
    if fs.use_superres
        || fs.frame_width != fs.mi_cols * 4
        || fs.frame_height != fs.mi_rows * 4
        || fs.upscaled_width != fs.frame_width
    {
        return Err(Error::TileListInvalid);
    }
    let ti = fh.tile_info.as_ref().ok_or(Error::TileListInvalid)?;
    if ti.mi_col_starts.len() != ti.tile_cols as usize + 1
        || ti.mi_row_starts.len() != ti.tile_rows as usize + 1
    {
        return Err(Error::TileListInvalid);
    }
    // TileHeight == one superblock, for ALL tile rows.
    let sb_mi = if seq.use_128x128_superblock {
        32u32
    } else {
        16
    };
    for r in 0..ti.tile_rows as usize {
        if ti.mi_row_starts[r + 1] - ti.mi_row_starts[r] != sb_mi {
            return Err(Error::TileListInvalid);
        }
    }
    // TileWidth identical for all tiles + an integer multiple of
    // TileHeight.
    let tile_w_mi = ti.mi_col_starts[1] - ti.mi_col_starts[0];
    for c in 0..ti.tile_cols as usize {
        if ti.mi_col_starts[c + 1] - ti.mi_col_starts[c] != tile_w_mi {
            return Err(Error::TileListInvalid);
        }
    }
    if tile_w_mi == 0 || tile_w_mi % sb_mi != 0 {
        return Err(Error::TileListInvalid);
    }
    Ok(LstGeometry {
        tile_width: tile_w_mi * 4,
        tile_height: sb_mi * 4,
    })
}

/// An anchor's planes widened to `u16` (the §7.20 `FrameStore`
/// sample type), validated against the camera frame's extents.
struct AnchorPlanes {
    planes: Vec<Vec<u16>>,
    dims: Vec<(u32, u32)>,
}

/// One decoded camera tile: per-plane §7.3.2 `OutY`/`OutU`/`OutV`
/// buffers plus their `(width, height)` extents.
type CameraTilePlanes = (Vec<Vec<i32>>, Vec<(u32, u32)>);

fn anchor_planes(
    anchor: &SpecFrame,
    seq: &SequenceHeader,
    fh: &FrameHeader,
) -> Result<AnchorPlanes, Error> {
    let fs = fh.frame_size.as_ref().ok_or(Error::TileListInvalid)?;
    let cc = &seq.color_config;
    // §7.3.1 steps 6-13 stamp the CURRENT frame's dimensions /
    // subsampling / bit depth onto the anchor's slot — an anchor of
    // any other shape cannot satisfy them.
    if anchor.width != fs.frame_width
        || anchor.height != fs.frame_height
        || anchor.bit_depth != cc.bit_depth
        || anchor.planes.len() != cc.num_planes as usize
    {
        return Err(Error::TileListInvalid);
    }
    let mut planes = Vec::with_capacity(anchor.planes.len());
    for (p, bytes) in anchor.planes.iter().enumerate() {
        let (w, h) = *anchor.plane_dims.get(p).ok_or(Error::TileListInvalid)?;
        let want = (w as usize) * (h as usize);
        let widened: Vec<u16> = if anchor.bit_depth == 8 {
            if bytes.len() != want {
                return Err(Error::TileListInvalid);
            }
            bytes.iter().map(|&v| u16::from(v)).collect()
        } else {
            if bytes.len() != want * 2 {
                return Err(Error::TileListInvalid);
            }
            bytes
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        };
        planes.push(widened);
    }
    Ok(AnchorPlanes {
        planes,
        dims: anchor.plane_dims.clone(),
    })
}

/// §7.3.2 decode camera tile process: decode ONE tile of the camera
/// frame against `anchor` (installed as `FrameStore[ ref_frame_idx[0]
/// ]`), returning the tile's plane buffers at their §7.3.2 `OutY` /
/// `OutU` / `OutV` extents.
fn decode_camera_tile(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    anchor: &AnchorPlanes,
    entry: &TileListEntry,
) -> Result<CameraTilePlanes, Error> {
    let cc = &seq.color_config;
    let fs = fh.frame_size.as_ref().ok_or(Error::TileListInvalid)?;
    let ti = fh.tile_info.as_ref().ok_or(Error::TileListInvalid)?;
    let qp = fh
        .quantization_params
        .as_ref()
        .ok_or(Error::TileListInvalid)?;
    let default_seg = SegmentationParams::disabled();
    let sp = fh.segmentation_params.as_ref().unwrap_or(&default_seg);

    if usize::from(entry.anchor_tile_row) >= ti.tile_rows as usize
        || usize::from(entry.anchor_tile_col) >= ti.tile_cols as usize
    {
        return Err(Error::TileListInvalid);
    }

    // §5.9.2 CodedLossless / LosslessArray + the §7.12.2 quantiser
    // bundle (delta-q is constraint-barred, segmentation ALT_Q not).
    let lossless = lossless_array(qp, sp);
    let coded_lossless = lossless.iter().all(|&l| l);
    let seg_alt_q_active = {
        let mut a = [false; MAX_SEGMENTS];
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = sp.segment_feature_active[i][SEG_LVL_ALT_Q];
        }
        a
    };
    let seg_alt_q_data = {
        let mut a = [0i16; MAX_SEGMENTS];
        for (i, slot) in a.iter_mut().enumerate() {
            *slot = sp.segment_feature_data[i][SEG_LVL_ALT_Q];
        }
        a
    };
    let mut seg_qm_level = [[15u8; MAX_SEGMENTS]; 3];
    if qp.using_qmatrix {
        for segment_id in 0..MAX_SEGMENTS {
            if !lossless[segment_id] {
                seg_qm_level[0][segment_id] = qp.qm_y;
                seg_qm_level[1][segment_id] = qp.qm_u;
                seg_qm_level[2][segment_id] = qp.qm_v;
            }
        }
    }
    let quant = QuantizerParams {
        base_q_idx: qp.base_q_idx,
        delta_q_y_dc: qp.delta_q_y_dc,
        delta_q_u_dc: qp.delta_q_u_dc,
        delta_q_u_ac: qp.delta_q_u_ac,
        delta_q_v_dc: qp.delta_q_v_dc,
        delta_q_v_ac: qp.delta_q_v_ac,
        using_qmatrix: qp.using_qmatrix,
        bit_depth: cc.bit_depth,
        delta_q_present: false,
        current_q_index: qp.base_q_idx,
        segmentation_enabled: sp.enabled,
        seg_alt_q_active,
        seg_alt_q_data,
        seg_qm_level,
    };

    let seg_skip: [bool; MAX_SEGMENTS] = if sp.enabled {
        core::array::from_fn(|s| {
            sp.segment_feature_active[s][crate::uncompressed_header_tail::SEG_LVL_SKIP]
        })
    } else {
        [false; MAX_SEGMENTS]
    };
    let tx_mode_select = matches!(
        fh.tx_mode,
        Some(crate::uncompressed_header_tail::TxMode::TxModeSelect)
    );
    let params = TileDecodeParams {
        frame_is_intra: false,
        subsampling_x: u8::from(cc.subsampling_x),
        subsampling_y: u8::from(cc.subsampling_y),
        num_planes: cc.num_planes,
        seg_id_pre_skip: sp.seg_id_pre_skip,
        segmentation_enabled: sp.enabled,
        seg_skip,
        last_active_seg_id: sp.last_active_seg_id,
        lossless_array: &lossless,
        coded_lossless,
        enable_cdef: false,
        allow_intrabc: false,
        cdef_bits: 0,
        use_128x128_superblock: seq.use_128x128_superblock,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_multi: false,
        mono_chrome: cc.mono_chrome,
        delta_lf_res: 0,
        allow_screen_content_tools: fh.allow_screen_content_tools,
        enable_filter_intra: seq.enable_filter_intra,
        bit_depth: cc.bit_depth,
        tx_mode_select,
        reduced_tx_set: fh.reduced_tx_set.unwrap_or(false),
        enable_intra_edge_filter: seq.enable_intra_edge_filter,
    };

    // ---- §7.3.1 steps 3-13: FrameStore[ last ] = the anchor. ----
    let ir = fh.inter_refs.as_ref().ok_or(Error::TileListInvalid)?;
    let last_slot = usize::from(ir.ref_frame_idx[0]);
    if last_slot >= NUM_REF_FRAMES as usize {
        return Err(Error::TileListInvalid);
    }
    let num_planes = cc.num_planes as usize;
    let sub_x = u8::from(cc.subsampling_x);
    let sub_y = u8::from(cc.subsampling_y);
    // Never-referenced placeholder for the other slots (§7.3.2's
    // read_ref_frames conformance restricts prediction to LAST).
    let dummy_plane: Vec<u16> = vec![0u16; 64];
    let dummy_entry = || RefFrameStoreEntry {
        plane: &dummy_plane,
        stride: 8,
        upscaled_width: 8,
        width: 8,
        height: 8,
    };
    let mut plane_stores: Vec<[RefFrameStoreEntry<'_>; NUM_REF_FRAMES as usize]> = Vec::new();
    for plane in 0..num_planes {
        let arr: [RefFrameStoreEntry<'_>; NUM_REF_FRAMES as usize] = core::array::from_fn(|slot| {
            if slot == last_slot {
                let (w, _h) = anchor.dims[plane];
                let (lw, lh) = anchor.dims[0];
                RefFrameStoreEntry {
                    plane: &anchor.planes[plane],
                    stride: w as usize,
                    upscaled_width: lw,
                    width: lw,
                    height: lh,
                }
            } else {
                dummy_entry()
            }
        });
        plane_stores.push(arr);
    }
    let mut ref_frame_idx = [0u8; 7];
    for (i, slot) in ref_frame_idx.iter_mut().enumerate() {
        *slot = ir.ref_frame_idx[i];
    }
    let order_hints = FrameInterOrderHints {
        order_hint_bits: 0,
        current_order_hint: 0,
        order_hints_by_ref: [0; 8],
    };
    let plane_ref_specs: Vec<PlaneRefSpec<'_>> = (0..num_planes)
        .map(|p| PlaneRefSpec {
            plane: p as u8,
            subsampling_x: if p > 0 { sub_x } else { 0 },
            subsampling_y: if p > 0 { sub_y } else { 0 },
            frame_store: &plane_stores[p],
            frame_width: fs.frame_width,
            frame_height: fs.frame_height,
        })
        .collect();
    let pixels = InterWalkPixels {
        ref_frame_idx,
        bit_depth: cc.bit_depth,
        plane_refs: &plane_ref_specs,
        order_hints,
    };
    // §7.3: use_ref_frame_mvs = 0 — the motion-field grid stays
    // invalid on both sides.
    let mfmvs = MotionFieldMvs::new_invalid(fs.mi_rows, fs.mi_cols);
    let mut ictx = InterFrameContext::identity_default(&mfmvs);
    ictx.segmentation_update_map = sp.update_map;
    ictx.segmentation_temporal_update = false;
    ictx.seg_feature_active = sp.segment_feature_active;
    ictx.seg_feature_data = sp.segment_feature_data;
    ictx.prev_segment_ids = None;
    ictx.skip_mode_present = fh.skip_mode_present.unwrap_or(false);
    ictx.reference_select = false;
    if let Some(g) = fh.global_motion_params.as_ref() {
        for r in 0..8 {
            ictx.gm_type[r] = g.gm_type[r] as i32;
            ictx.gm_params[r] = g.gm_params[r];
        }
    }
    ictx.allow_high_precision_mv = ir.allow_high_precision_mv;
    ictx.force_integer_mv = fh.force_integer_mv;
    ictx.use_ref_frame_mvs = false;
    ictx.is_motion_mode_switchable = ir.is_motion_mode_switchable;
    ictx.allow_warped_motion = fh.allow_warped_motion.unwrap_or(false);
    ictx.enable_interintra_compound = seq.enable_interintra_compound;
    ictx.enable_masked_compound = seq.enable_masked_compound;
    ictx.enable_jnt_comp = seq.enable_jnt_comp;
    ictx.order_hints = order_hints;
    ictx.interpolation_filter = ir.interpolation_filter as u8;
    ictx.enable_dual_filter = seq.enable_dual_filter;
    ictx.pixels = Some(&pixels);

    // ---- §7.3.2: one §5.11.2 tile decode, no post-processing. ----
    let mi_rows = fs.mi_rows;
    let mi_cols = fs.mi_cols;
    let mut walker = PartitionWalker::new(
        mi_rows,
        mi_cols,
        TileGeometry {
            mi_row_start: 0,
            mi_row_end: mi_rows,
            mi_col_start: 0,
            mi_col_end: mi_cols,
        },
    )
    .ok_or(Error::TileListInvalid)?;
    let row = usize::from(entry.anchor_tile_row);
    let col = usize::from(entry.anchor_tile_col);
    let geometry = TileGeometry {
        mi_row_start: ti.mi_row_starts[row],
        mi_row_end: ti.mi_row_starts[row + 1],
        mi_col_start: ti.mi_col_starts[col],
        mi_col_end: ti.mi_col_starts[col + 1],
    };
    walker.begin_tile(geometry);
    // §7.3.2: CurrentQIndex = base_q_idx.
    walker.set_current_q_index(i32::from(qp.base_q_idx));
    // §7.3.2: init_symbol( tile_data_size_minus_1 + 1 ) over the
    // entry's bytes; disable_cdf_update == 1 per the §7.3 gate.
    let mut decoder = SymbolDecoder::init_symbol(
        &entry.coded_tile_data,
        entry.coded_tile_data.len(),
        /* disable_cdf_update = */ true,
    )?;
    // §7.3 inputs: PRIMARY_REF_NONE (validated) — §8.3.1 defaults +
    // the q-selected coefficient CDF load.
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.init_coeff_cdfs(qp.base_q_idx);
    walker.decode_tile_syntax_with_lr(
        &mut decoder,
        &mut cdfs,
        &params,
        /* lr = */ None,
        Some(&ictx),
        &quant,
        /* read_deltas = */ false,
    )?;

    // ---- §7.3.2 OutY / OutU / OutV extraction. ----
    let w = (geometry.mi_col_end - geometry.mi_col_start) * 4;
    let h = (geometry.mi_row_end - geometry.mi_row_start) * 4;
    let x0 = geometry.mi_col_start * 4;
    let y0 = geometry.mi_row_start * 4;
    let mut out_planes: Vec<Vec<i32>> = Vec::with_capacity(num_planes);
    let mut out_dims: Vec<(u32, u32)> = Vec::with_capacity(num_planes);
    for plane in 0..num_planes {
        let src = walker.curr_frame(plane).ok_or(Error::TileListInvalid)?;
        let (rows, cols) = walker
            .curr_frame_dims(plane)
            .ok_or(Error::TileListInvalid)?;
        let (pw, ph, px0, py0) = if plane == 0 {
            (w, h, x0, y0)
        } else {
            (
                (w + u32::from(sub_x)) >> sub_x,
                (h + u32::from(sub_y)) >> sub_y,
                x0 >> sub_x,
                y0 >> sub_y,
            )
        };
        if py0 + ph > rows || px0 + pw > cols {
            return Err(Error::TileListInvalid);
        }
        let mut buf = vec![0i32; (pw as usize) * (ph as usize)];
        for y in 0..ph as usize {
            let src_row = (py0 as usize + y) * cols as usize + px0 as usize;
            buf[y * pw as usize..(y + 1) * pw as usize]
                .copy_from_slice(&src[src_row..src_row + pw as usize]);
        }
        out_planes.push(buf);
        out_dims.push((pw, ph));
    }
    Ok((out_planes, out_dims))
}

/// §7.3.1 large-scale-tile decoding: assemble one output frame from a
/// tile list against an externally supplied anchor set.
///
/// `seq` / `fh` are the parsed sequence + camera frame headers (§7.3
/// inputs); `anchors` is the `AnchorFrames` array (up to 128 decoded
/// frames, pixels only). The output frame is `outputW × outputH`
/// (`(output_frame_width/height_in_tiles_minus_1 + 1) ×
/// TileWidth/Height`) with each entry's decoded tile written in
/// raster order; § 7.3.1: samples outside the decoded tiles are left
/// untouched (zero-initialised here).
///
/// ## Errors
///
/// [`Error::TileListInvalid`] for every §7.3/§6.11 conformance or
/// input-contract violation (see the module docs), or any tile-decode
/// error from the §5.11 walk over an entry's `coded_tile_data`.
pub fn decode_tile_list(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    anchors: &[SpecFrame],
    tl: &TileListObu,
) -> Result<SpecFrame, Error> {
    let geom = validate_lst_shape(seq, fh)?;
    let cc = &seq.color_config;
    let width_in_tiles = u32::from(tl.output_frame_width_in_tiles_minus_1) + 1;
    let height_in_tiles = u32::from(tl.output_frame_height_in_tiles_minus_1) + 1;
    // §7.3.1: tile_count_minus_1 + 1 <= output width * height.
    if tl.entries.is_empty() || tl.entries.len() as u32 > width_in_tiles * height_in_tiles {
        return Err(Error::TileListInvalid);
    }
    let output_w = width_in_tiles * geom.tile_width;
    let output_h = height_in_tiles * geom.tile_height;
    let sub_x = u8::from(cc.subsampling_x);
    let sub_y = u8::from(cc.subsampling_y);
    let num_planes = cc.num_planes as usize;
    let mut out_dims: Vec<(u32, u32)> = Vec::with_capacity(num_planes);
    for plane in 0..num_planes {
        out_dims.push(if plane == 0 {
            (output_w, output_h)
        } else {
            (output_w >> sub_x, output_h >> sub_y)
        });
    }
    let mut out_planes: Vec<Vec<i32>> = out_dims
        .iter()
        .map(|&(w, h)| vec![0i32; (w as usize) * (h as usize)])
        .collect();

    // Anchor conversions are cached per distinct index (a tile list
    // routinely references the same anchor many times).
    let mut anchor_cache: Vec<Option<AnchorPlanes>> = Vec::new();
    anchor_cache.resize_with(anchors.len(), || None);

    for (tile, entry) in tl.entries.iter().enumerate() {
        let a_idx = usize::from(entry.anchor_frame_idx);
        if a_idx >= anchors.len() {
            return Err(Error::TileListInvalid);
        }
        if anchor_cache[a_idx].is_none() {
            anchor_cache[a_idx] = Some(anchor_planes(&anchors[a_idx], seq, fh)?);
        }
        let anchor = anchor_cache[a_idx].as_ref().expect("just filled");
        let (tile_planes, tile_dims) = decode_camera_tile(seq, fh, anchor, entry)?;
        // §7.3.1 raster placement.
        let dest_x = geom.tile_width * (tile as u32 % width_in_tiles);
        let dest_y = geom.tile_height * (tile as u32 / width_in_tiles);
        for plane in 0..num_planes {
            let (tw, th) = tile_dims[plane];
            let (ow, _oh) = out_dims[plane];
            let (dx, dy) = if plane == 0 {
                (dest_x, dest_y)
            } else {
                (dest_x >> sub_x, dest_y >> sub_y)
            };
            let src = &tile_planes[plane];
            let dst = &mut out_planes[plane];
            for y in 0..th as usize {
                let d = (dy as usize + y) * ow as usize + dx as usize;
                dst[d..d + tw as usize]
                    .copy_from_slice(&src[y * tw as usize..(y + 1) * tw as usize]);
            }
        }
    }

    // §7.3.1: "The bitdepth of each output sample is given by
    // BitDepth" — the same output layout contract as [`SpecFrame`].
    let max_val: i32 = (1 << cc.bit_depth) - 1;
    let planes: Vec<Vec<u8>> = out_planes
        .into_iter()
        .map(|buf| {
            if cc.bit_depth == 8 {
                buf.into_iter().map(|v| v.clamp(0, 255) as u8).collect()
            } else {
                let mut out = Vec::with_capacity(buf.len() * 2);
                for v in buf {
                    out.extend_from_slice(&(v.clamp(0, max_val) as u16).to_le_bytes());
                }
                out
            }
        })
        .collect();
    Ok(SpecFrame {
        width: output_w,
        height: output_h,
        planes,
        plane_dims: out_dims,
        bit_depth: cc.bit_depth,
    })
}

/// [`decode_tile_list`] over the natural byte-level packaging: a §5.2
/// low-overhead OBU concatenation carrying `OBU_SEQUENCE_HEADER`,
/// `OBU_FRAME_HEADER` and `OBU_TILE_LIST` (temporal delimiters and
/// padding are skipped; any other OBU type rejects — the §7.3 mode
/// has no tile groups or shown frames of its own).
///
/// ## Errors
///
/// [`Error::TileListInvalid`] when the stream is missing any of the
/// three §7.3 inputs or carries a foreign OBU type, plus every
/// [`decode_tile_list`] error surface.
pub fn decode_tile_list_stream(input: &[u8], anchors: &[SpecFrame]) -> Result<SpecFrame, Error> {
    let mut seq: Option<SequenceHeader> = None;
    let mut fh: Option<FrameHeader> = None;
    let mut out: Option<SpecFrame> = None;
    for desc in ObuIter::new(input) {
        let desc = desc?;
        match desc.obu_type {
            ObuType::TemporalDelimiter | ObuType::Padding => {}
            ObuType::SequenceHeader => {
                seq = Some(parse_sequence_header(desc.payload)?);
            }
            ObuType::FrameHeader => {
                let s = seq.as_ref().ok_or(Error::TileListInvalid)?;
                // §7.3: refresh_frame_flags == 0 and no ref-derived
                // sizes (frame_size_override == 0), so an empty
                // RefInfo satisfies the parse.
                fh = Some(parse_frame_header_with_refs(
                    desc.payload,
                    s,
                    &RefInfo::default(),
                )?);
            }
            ObuType::TileList => {
                let s = seq.as_ref().ok_or(Error::TileListInvalid)?;
                let f = fh.as_ref().ok_or(Error::TileListInvalid)?;
                let tl = parse_tile_list_obu(desc.payload)?;
                out = Some(decode_tile_list(s, f, anchors, &tl)?);
            }
            _ => return Err(Error::TileListInvalid),
        }
    }
    out.ok_or(Error::TileListInvalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tl() -> TileListObu {
        TileListObu {
            output_frame_width_in_tiles_minus_1: 1,
            output_frame_height_in_tiles_minus_1: 0,
            entries: vec![
                TileListEntry {
                    anchor_frame_idx: 0,
                    anchor_tile_row: 0,
                    anchor_tile_col: 0,
                    coded_tile_data: vec![0xAA, 0xBB, 0xCC],
                },
                TileListEntry {
                    anchor_frame_idx: 3,
                    anchor_tile_row: 0,
                    anchor_tile_col: 1,
                    coded_tile_data: vec![0x11],
                },
            ],
        }
    }

    #[test]
    fn tile_list_round_trips_byte_exact() {
        let tl = sample_tl();
        let bytes = write_tile_list_obu(&tl).expect("writes");
        // §5.12.1/.2 layout: 2 + 2 header bytes, then per entry
        // 5 + N bytes.
        assert_eq!(bytes.len(), 4 + 5 + 3 + 5 + 1);
        assert_eq!(&bytes[..4], &[1, 0, 0x00, 0x01]);
        let parsed = parse_tile_list_obu(&bytes).expect("parses");
        assert_eq!(parsed, tl);
    }

    #[test]
    fn parse_rejects_truncated_and_overlong_payloads() {
        let tl = sample_tl();
        let bytes = write_tile_list_obu(&tl).expect("writes");
        // Truncated inside the last entry's tile data.
        assert_eq!(
            parse_tile_list_obu(&bytes[..bytes.len() - 1]),
            Err(Error::UnexpectedEnd)
        );
        // Trailing garbage after the last entry (§5.3.1 gives
        // OBU_TILE_LIST no trailing bits).
        let mut padded = bytes.clone();
        padded.push(0);
        assert_eq!(parse_tile_list_obu(&padded), Err(Error::TileListInvalid));
        // Truncated fixed header.
        assert_eq!(parse_tile_list_obu(&bytes[..3]), Err(Error::UnexpectedEnd));
    }

    #[test]
    fn parse_enforces_the_6_11_conformance_bounds() {
        // tile_count_minus_1 = 512 > 511.
        let bytes = [0u8, 0, 0x02, 0x00];
        assert_eq!(parse_tile_list_obu(&bytes), Err(Error::TileListInvalid));
        // anchor_frame_idx = 128 > 127.
        let bytes = [0u8, 0, 0x00, 0x00, 128, 0, 0, 0x00, 0x00, 0xAA];
        assert_eq!(parse_tile_list_obu(&bytes), Err(Error::TileListInvalid));
    }

    #[test]
    fn writer_rejects_out_of_envelope_lists() {
        let mut tl = sample_tl();
        tl.entries[0].anchor_frame_idx = 128;
        assert_eq!(write_tile_list_obu(&tl), Err(Error::TileListInvalid));
        let mut tl = sample_tl();
        tl.entries[0].coded_tile_data = Vec::new();
        assert_eq!(write_tile_list_obu(&tl), Err(Error::TileListInvalid));
        let tl = TileListObu {
            output_frame_width_in_tiles_minus_1: 0,
            output_frame_height_in_tiles_minus_1: 0,
            entries: Vec::new(),
        };
        assert_eq!(write_tile_list_obu(&tl), Err(Error::TileListInvalid));
    }
}
