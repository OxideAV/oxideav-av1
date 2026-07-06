//! Spec-faithful frame decode driver — §7.4 `decode_frame_wrapup` order
//! over the §5.11 `PartitionWalker` syntax walk.
//!
//! This is the decode path for **real encoder-produced bitstreams** (as
//! opposed to the encoder-mirror drivers in [`super::pixel_driver`] /
//! [`super::pixel_driver_dyn`], which accept only this crate's own
//! constrained encoder output). It wires the crate's spec modules end to
//! end for one frame:
//!
//!   1. §5.9 frame-header derived state → [`TileDecodeParams`] +
//!      [`QuantizerParams`] (including the §5.9.2 `CodedLossless` /
//!      `LosslessArray` derivations and the §5.9.14 segmentation
//!      feature plumbing).
//!   2. §8.2.2 `init_symbol` over the tile bytes + §8.3.1 default CDF
//!      load (`new_from_defaults` + the q-context-selected
//!      `init_coeff_cdfs( base_q_idx )` copy).
//!   3. §5.11.2 [`PartitionWalker::decode_tile_syntax`] — the
//!      superblock loop over §5.11.4 `decode_partition` → §5.11.5
//!      `decode_block` → §5.11.34 `residual`, reconstructing every
//!      intra transform block into `CurrFrame[ plane ]`.
//!   4. The §7.4 in-loop / post passes in decode order over the
//!      mi-grid-padded planes: §7.14 deblock (only when a luma filter
//!      level is nonzero) → §7.15 CDEF → §7.16 superres (both the CDEF
//!      output and the retained post-deblock frame) → §7.17 loop
//!      restoration → the §7.18.2 crop → §7.18.3 film grain.
//!
//! ## Scope
//!
//! * Intra-only frames (KEY / INTRA_ONLY), 8-bit output, single- and
//!   multi-tile layouts.
//! * 4:2:0 / 4:2:2 / 4:4:4 and monochrome layouts (the walker threads
//!   `subsampling_x/y` + `mono_chrome`; only 8-bit output is surfaced).
//! * Inter frames, `show_existing_frame`, and quantizer-matrix streams
//!   return [`Error::PartitionWalkOutOfRange`] (follow-ups).
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.9, §5.10, §5.11,
//! §7.4, §7.12.2, §7.14, §7.15, §7.16, §7.17, §7.18.

use crate::cdf::{
    FrameInterOrderHints, InterFrameContext, InterWalkPixels, MotionFieldMvs, PartitionWalker,
    QuantizerParams, TileCdfContext, TileDecodeParams, TileGeometry,
};
use crate::encoder::ivf::IvfReader;
use crate::encoder::tile_group_obu::parse_tile_group_obu_body;
use crate::film_grain::film_grain_synthesis;
use crate::frame_header::{
    parse_frame_header_with_refs, FrameHeader, RefInfo, NUM_REF_FRAMES, PRIMARY_REF_NONE,
};
use crate::inter_pred::get_relative_dist;
use crate::loop_filter::PlaneBuffer;
use crate::obu::{ObuIter, ObuType};
use crate::sequence_header::{parse_sequence_header, SequenceHeader};
use crate::symbol_decoder::SymbolDecoder;
use crate::uncompressed_header_tail::{
    SegmentationParams, ALTREF_FRAME, LAST_FRAME, MAX_SEGMENTS, SEG_LVL_ALT_Q, SEG_LVL_SKIP,
};
use crate::Error;
use crate::{PlaneRefSpec, RefFrameStoreEntry};

/// One frame decoded by the spec-faithful driver. Planes are surfaced
/// at their §5.9.8 cropped extents (`FrameWidth` × `FrameHeight` for
/// luma, the §5.5.2 subsampled extent for chroma), row-major, 8-bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecFrame {
    /// Luma width in pixels (`FrameWidth`, post-superres if any).
    pub width: u32,
    /// Luma height in pixels (`FrameHeight`).
    pub height: u32,
    /// Decoded planes: `[Y]` (monochrome) or `[Y, U, V]`.
    pub planes: Vec<Vec<u8>>,
    /// `(width, height)` per surfaced plane.
    pub plane_dims: Vec<(u32, u32)>,
}

/// §7.20 per-slot reference store — the decoded frame a later inter
/// frame motion-compensates against, plus the per-mi grids the §7.9
/// temporal projection and the §7.21 `show_existing_frame` output
/// path consume.
#[derive(Debug, Clone)]
struct SpecRefSlot {
    /// §7.20 `FrameStore[ i ][ plane ]` — the post-§7.17 (pre-§7.18.3
    /// film-grain) planes at their §7.18.2 cropped extents
    /// (`UpscaledWidth × FrameHeight`, per-plane subsampled), `u16`
    /// (post-`Clip1`, so the widening from the walker's `i32` is
    /// lossless).
    planes: Vec<Vec<u16>>,
    /// `(width, height)` per stored plane.
    plane_dims: Vec<(u32, u32)>,
    /// §7.20 `SavedMvs[ i ]` — the §7.19 `MfMvs[ row ][ col ][ 0..2 ]`
    /// grid snapshot (2 `i16` per mi cell), the §7.9.2 projection
    /// source.
    mf_mvs: Vec<i16>,
    /// §7.20 `SavedRefFrames[ i ]` — the §7.19 `MfRefFrames[ row ][
    /// col ]` grid snapshot (1 `i8` per mi cell).
    mf_ref_frames: Vec<i8>,
    /// §7.20 `SavedOrderHints[ i ][ ref ]` — the stored frame's
    /// `OrderHints[]` array (what ITS references' output order was),
    /// consumed by the §7.9.1/§7.9.2 `SavedOrderHints[ srcIdx ][ .. ]`
    /// reads.
    saved_order_hints: [i32; ALTREF_FRAME + 1],
    /// The stored frame's mi extent.
    mi_rows: u32,
    mi_cols: u32,
    /// §7.20 `RefFrameType[ i ]` — `FrameIsIntra` of the stored frame
    /// (the §7.9.2 projection skips KEY / INTRA_ONLY sources).
    frame_is_intra: bool,
    /// §7.20 `RefFrameType[ i ] == KEY_FRAME` — the §7.21 trigger.
    frame_type_is_key: bool,
    /// §7.20 `save_cdfs( i )` — the frame-end CDF state (§8.4
    /// `frame_end_update_cdf`), loaded back by §8.3.1 `load_cdfs()`
    /// when a later frame names this slot as its primary reference.
    cdfs: Box<TileCdfContext>,
    /// §7.20 `SavedGmParams[ i ]` — this frame's `gm_params`.
    gm_params: [[i32; 6]; 8],
    /// §7.20 `save_loop_filter_params( i )` — the §5.9.11 running
    /// delta state at the end of this frame's header parse.
    lf_ref_deltas: [i8; 8],
    /// Mode-delta half of `save_loop_filter_params( i )`.
    lf_mode_deltas: [i8; 2],
}

/// Cross-frame decoder session state: the §5.9.2 `RefInfo` arrays the
/// inter `uncompressed_header()` parse consumes plus the §7.20
/// per-slot pixel/grid stores.
#[derive(Debug, Clone)]
struct SpecRefState {
    info: RefInfo,
    slots: [Option<SpecRefSlot>; NUM_REF_FRAMES as usize],
}

impl SpecRefState {
    fn new() -> Self {
        Self {
            info: RefInfo::default(),
            slots: Default::default(),
        }
    }
}

/// One frame decoded by [`decode_frame_spec_full`] — the surfaced
/// [`SpecFrame`] plus the §7.20 reference-update payload.
struct DecodedFrameInternal {
    /// The §7.18 output frame (post-film-grain).
    frame: SpecFrame,
    /// The §7.20 store payload: pre-grain cropped planes (`u16`).
    ref_planes: Vec<Vec<u16>>,
    /// `(width, height)` per `ref_planes` entry.
    ref_plane_dims: Vec<(u32, u32)>,
    /// §7.19 `MfMvs[]` grid (2 `i16` per mi cell).
    mf_mvs: Vec<i16>,
    /// §7.19 `MfRefFrames[]` grid (1 `i8` per mi cell).
    mf_ref_frames: Vec<i8>,
    /// This frame's `OrderHints[]` array (§5.9.2), the §7.20
    /// `SavedOrderHints` payload.
    order_hints_by_ref: [i32; ALTREF_FRAME + 1],
    /// §8.4 frame-end CDF state (`frame_end_update_cdf` output — the
    /// `context_update_tile_id` tile's adapted CDFs, or the frame-start
    /// state under `disable_frame_end_update_cdf == 1`).
    end_cdfs: Box<TileCdfContext>,
    /// The decoded frame's mi extent.
    mi_rows: u32,
    mi_cols: u32,
}

/// §5.9.2 `LosslessArray[ segmentId ]` — `get_qindex( 1, segmentId ) ==
/// 0 && DeltaQYDc == 0 && DeltaQ{U,V}{Ac,Dc} == 0` for every segment.
fn lossless_array(
    qp: &crate::uncompressed_header_tail::QuantizationParams,
    sp: &SegmentationParams,
) -> [bool; MAX_SEGMENTS] {
    let deltas_all_zero = qp.delta_q_y_dc == 0
        && qp.delta_q_u_dc == 0
        && qp.delta_q_u_ac == 0
        && qp.delta_q_v_dc == 0
        && qp.delta_q_v_ac == 0;
    let mut out = [false; MAX_SEGMENTS];
    for (segment_id, slot) in out.iter_mut().enumerate() {
        // §7.12.2 get_qindex( ignoreDeltaQ = 1, segmentId ).
        let qindex = if sp.enabled && sp.segment_feature_active[segment_id][SEG_LVL_ALT_Q] {
            let data = i32::from(sp.segment_feature_data[segment_id][SEG_LVL_ALT_Q]);
            (i32::from(qp.base_q_idx) + data).clamp(0, 255)
        } else {
            i32::from(qp.base_q_idx)
        };
        *slot = qindex == 0 && deltas_all_zero;
    }
    out
}

/// Decode one intra frame through the §5.11 syntax walker + the §7.4
/// post-pass chain, given its already-parsed sequence header, frame
/// header, and the §5.11.1 tile-group OBU body.
pub fn decode_frame_spec(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    tile_group_body: &[u8],
) -> Result<SpecFrame, Error> {
    // The historical intra-only entry: no cross-frame reference state,
    // so inter frames are rejected inside the full driver.
    Ok(decode_frame_spec_full(seq, fh, tile_group_body, None)?.frame)
}

/// [`decode_frame_spec`] with the cross-frame reference state — the
/// full §5.11 + §7.4 decode for one KEY / INTRA_ONLY / INTER frame,
/// returning both the output frame and the §7.20 reference-update
/// payload.
fn decode_frame_spec_full(
    seq: &SequenceHeader,
    fh: &FrameHeader,
    tile_group_body: &[u8],
    refs: Option<&SpecRefState>,
) -> Result<DecodedFrameInternal, Error> {
    if fh.show_existing_frame {
        // §7.21 output-existing path is the caller's (no tile group).
        return Err(Error::PartitionWalkOutOfRange);
    }
    if !fh.frame_is_intra && refs.is_none() {
        // Inter frames are undecodable without the §7.20 store.
        return Err(Error::PartitionWalkOutOfRange);
    }
    let cc = &seq.color_config;
    if cc.bit_depth != 8 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    if fs.use_superres && fs.upscaled_width <= fs.frame_width {
        // §5.9.8 conformance: superres only ever widens.
        return Err(Error::PartitionWalkOutOfRange);
    }
    let ti = fh
        .tile_info
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let qp = fh
        .quantization_params
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    if qp.using_qmatrix {
        // §7.12.3 step-1 QM path not threaded through the walker driver.
        return Err(Error::PartitionWalkOutOfRange);
    }
    let default_seg = SegmentationParams::disabled();
    let sp = fh.segmentation_params.as_ref().unwrap_or(&default_seg);
    let dq = fh.delta_q_params.unwrap_or_default();
    let dlf = fh.delta_lf_params.unwrap_or_default();
    let lr = fh
        .lr_params
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    // §5.11.57 loop-restoration state for the per-superblock `read_lr`
    // interleave (walker-facing shape).
    let lr_walk = crate::cdf::LrParams {
        num_planes: cc.num_planes as usize,
        frame_restoration_type: [
            lr.frame_restoration_type[0] as u8,
            lr.frame_restoration_type[1] as u8,
            lr.frame_restoration_type[2] as u8,
        ],
        loop_restoration_size: lr.loop_restoration_size,
        subsampling_x: u8::from(cc.subsampling_x),
        subsampling_y: u8::from(cc.subsampling_y),
        frame_height: fs.frame_height,
        upscaled_width: fs.upscaled_width,
        use_superres: fs.use_superres,
        superres_denom: fs.superres_denom,
        allow_intrabc: fh.allow_intrabc,
    };

    // §5.9.2 CodedLossless / LosslessArray.
    let lossless = lossless_array(qp, sp);
    let coded_lossless = lossless.iter().all(|&l| l);

    // §7.12.2 quantizer state for the walk. `CurrentQIndex` starts at
    // `base_q_idx` (§5.9.18); the walker's §5.11.13 read_delta_qindex
    // maintains the running value internally.
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
    let quant = QuantizerParams {
        base_q_idx: qp.base_q_idx,
        delta_q_y_dc: qp.delta_q_y_dc,
        delta_q_u_dc: qp.delta_q_u_dc,
        delta_q_u_ac: qp.delta_q_u_ac,
        delta_q_v_dc: qp.delta_q_v_dc,
        delta_q_v_ac: qp.delta_q_v_ac,
        using_qmatrix: qp.using_qmatrix,
        bit_depth: cc.bit_depth,
        delta_q_present: dq.delta_q_present,
        current_q_index: qp.base_q_idx,
        segmentation_enabled: sp.enabled,
        seg_alt_q_active,
        seg_alt_q_data,
    };

    // `seg_skip_active`: any segment with SEG_LVL_SKIP active.
    let seg_skip_active = sp.enabled
        && sp
            .segment_feature_active
            .iter()
            .any(|features| features[SEG_LVL_SKIP]);

    let cdef_bits = fh
        .cdef_params
        .as_ref()
        .map_or(0, |c| u32::from(c.cdef_bits));
    let tx_mode_select = matches!(
        fh.tx_mode,
        Some(crate::uncompressed_header_tail::TxMode::TxModeSelect)
    );

    let params = TileDecodeParams {
        frame_is_intra: fh.frame_is_intra,
        subsampling_x: u8::from(cc.subsampling_x),
        subsampling_y: u8::from(cc.subsampling_y),
        num_planes: cc.num_planes,
        seg_id_pre_skip: sp.seg_id_pre_skip,
        segmentation_enabled: sp.enabled,
        seg_skip_active,
        last_active_seg_id: sp.last_active_seg_id,
        lossless_array: &lossless,
        coded_lossless,
        enable_cdef: seq.enable_cdef,
        allow_intrabc: fh.allow_intrabc,
        cdef_bits,
        use_128x128_superblock: seq.use_128x128_superblock,
        delta_q_res: dq.delta_q_res,
        delta_lf_present: dlf.delta_lf_present,
        delta_lf_multi: dlf.delta_lf_multi,
        mono_chrome: cc.mono_chrome,
        delta_lf_res: dlf.delta_lf_res,
        allow_screen_content_tools: fh.allow_screen_content_tools,
        enable_filter_intra: seq.enable_filter_intra,
        bit_depth: cc.bit_depth,
        tx_mode_select,
        reduced_tx_set: fh.reduced_tx_set.unwrap_or(false),
        enable_intra_edge_filter: seq.enable_intra_edge_filter,
    };

    // §5.11.1 tile-group body → per-tile byte ranges.
    let num_tiles = ti.tile_cols * ti.tile_rows;
    let ceil_log2 = |v: u32| -> u32 {
        if v <= 1 {
            0
        } else {
            32 - (v - 1).leading_zeros()
        }
    };
    if ti.mi_col_starts.len() != (ti.tile_cols as usize) + 1
        || ti.mi_row_starts.len() != (ti.tile_rows as usize) + 1
    {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let parsed = parse_tile_group_obu_body(
        tile_group_body,
        num_tiles,
        ceil_log2(ti.tile_cols),
        ceil_log2(ti.tile_rows),
        u32::from(ti.tile_size_bytes),
    )?;
    if parsed.tiles.len() != num_tiles as usize {
        return Err(Error::PartitionWalkOutOfRange);
    }

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
    .ok_or(Error::PartitionWalkOutOfRange)?;

    // ---- Inter-frame state: §5.11.18 context + §7.11.3 ref pixels. ----
    // Owned buffers first (the `InterFrameContext` borrows them), then
    // the context itself. All of it is inert on intra frames.
    let is_inter_frame = !fh.frame_is_intra;
    let sub_x_u8 = u8::from(cc.subsampling_x);
    let sub_y_u8 = u8::from(cc.subsampling_y);
    let num_planes_usize = cc.num_planes as usize;
    // §7.9 motion-field grid — the temporal-scan source. §5.9.2
    // invokes `motion_field_estimation()` only when `use_ref_frame_mvs
    // == 1`; the projection walks the §7.20-stored `SavedMvs` /
    // `SavedRefFrames` / `SavedOrderHints` of up to four reference
    // frames (LAST backwards, BWDREF / ALTREF2 / ALTREF forwards,
    // LAST2 as the stack filler).
    let mfmvs = if is_inter_frame
        && fh
            .inter_refs
            .as_ref()
            .is_some_and(|ir| ir.use_ref_frame_mvs)
    {
        motion_field_estimation(
            refs.ok_or(Error::PartitionWalkOutOfRange)?,
            fh,
            seq,
            mi_rows,
            mi_cols,
        )?
    } else {
        MotionFieldMvs::new_invalid(mi_rows, mi_cols)
    };
    // Never-referenced placeholder for empty §7.20 slots (a conformant
    // stream only references `RefValid` slots).
    let dummy_plane: Vec<u16> = vec![0u16; 64];
    let dummy_entry = || RefFrameStoreEntry {
        plane: &dummy_plane,
        stride: 8,
        upscaled_width: 8,
        width: 8,
        height: 8,
    };
    // Per-plane `FrameStore[ slot ]` views over the §7.20 store.
    let mut plane_stores: Vec<[RefFrameStoreEntry<'_>; NUM_REF_FRAMES as usize]> = Vec::new();
    let mut ref_frame_idx = [0u8; 7];
    let mut order_hints_by_ref = [0i32; ALTREF_FRAME + 1];
    let mut sign_bias = [0i32; 8];
    let mut is_scaled_per_ref = [false; 7];
    if is_inter_frame {
        let st = refs.ok_or(Error::PartitionWalkOutOfRange)?;
        let ir = fh
            .inter_refs
            .as_ref()
            .ok_or(Error::PartitionWalkOutOfRange)?;
        // Loud follow-up gates for state this milestone does not
        // forward yet (none fire on the corpus arc this lands for).
        if sp.enabled {
            // §5.9.14 segmentation feature state on the §5.11.18 ctx.
            return Err(Error::PartitionWalkOutOfRange);
        }
        for plane in 0..num_planes_usize {
            let arr: [RefFrameStoreEntry<'_>; NUM_REF_FRAMES as usize] =
                core::array::from_fn(|slot| match st.slots[slot].as_ref() {
                    Some(s) if plane < s.planes.len() => {
                        let (w, h) = s.plane_dims[plane];
                        RefFrameStoreEntry {
                            plane: &s.planes[plane],
                            stride: w as usize,
                            upscaled_width: w,
                            width: w,
                            height: h,
                        }
                    }
                    _ => dummy_entry(),
                });
            plane_stores.push(arr);
        }
        for i in 0..7 {
            let slot = ir.ref_frame_idx[i] as usize;
            if slot >= NUM_REF_FRAMES as usize {
                return Err(Error::PartitionWalkOutOfRange);
            }
            ref_frame_idx[i] = slot as u8;
            // §5.9.2 `OrderHints[ LAST_FRAME + i ] =
            // RefOrderHint[ ref_frame_idx[ i ] ]`.
            let hint = st.info.order_hint[slot] as i32;
            order_hints_by_ref[LAST_FRAME + i] = hint;
            // §7.8 `RefFrameSignBias[ refFrame ] =
            // get_relative_dist( hint, OrderHint ) > 0`.
            if seq.enable_order_hint {
                sign_bias[LAST_FRAME + i] = i32::from(
                    get_relative_dist(hint, fh.order_hint as i32, u32::from(seq.order_hint_bits))
                        > 0,
                );
            }
            // §5.11.27 `is_scaled( refFrame )`: the stored dims differ
            // from the current frame's.
            is_scaled_per_ref[i] = st.info.upscaled_width[slot] != fs.upscaled_width
                || st.info.frame_height[slot] != fs.frame_height;
        }
    }
    let order_hints = FrameInterOrderHints {
        order_hint_bits: if seq.enable_order_hint {
            u32::from(seq.order_hint_bits)
        } else {
            0
        },
        current_order_hint: fh.order_hint as i32,
        order_hints_by_ref,
    };
    let plane_ref_specs: Vec<PlaneRefSpec<'_>> = (0..plane_stores.len())
        .map(|p| PlaneRefSpec {
            plane: p as u8,
            subsampling_x: if p > 0 { sub_x_u8 } else { 0 },
            subsampling_y: if p > 0 { sub_y_u8 } else { 0 },
            frame_store: &plane_stores[p],
            frame_width: if p == 0 {
                fs.frame_width
            } else {
                (fs.frame_width + u32::from(sub_x_u8)) >> sub_x_u8
            },
            frame_height: if p == 0 {
                fs.frame_height
            } else {
                (fs.frame_height + u32::from(sub_y_u8)) >> sub_y_u8
            },
        })
        .collect();
    let pixels = InterWalkPixels {
        ref_frame_idx,
        bit_depth: cc.bit_depth,
        plane_refs: &plane_ref_specs,
        order_hints,
    };
    let ictx: Option<InterFrameContext<'_>> = if is_inter_frame {
        let ir = fh
            .inter_refs
            .as_ref()
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let mut c = InterFrameContext::identity_default(&mfmvs);
        // §5.9.22 skip-mode state: the §5.11.10 `read_skip_mode` gate
        // plus the fixed compound reference pair the skip-mode arm
        // predicts from.
        c.skip_mode_present = fh.skip_mode_present.unwrap_or(false);
        if let Some(smf) = fh.skip_mode_frame {
            c.skip_mode_frame = [i32::from(smf[0]), i32::from(smf[1])];
        }
        c.reference_select = fh.reference_select.unwrap_or(false);
        if let Some(g) = fh.global_motion_params.as_ref() {
            for r in 0..8 {
                c.gm_type[r] = g.gm_type[r] as i32;
                c.gm_params[r] = g.gm_params[r];
            }
        }
        c.ref_frame_sign_bias = sign_bias;
        c.allow_high_precision_mv = ir.allow_high_precision_mv;
        c.force_integer_mv = fh.force_integer_mv;
        c.use_ref_frame_mvs = ir.use_ref_frame_mvs;
        c.is_motion_mode_switchable = ir.is_motion_mode_switchable;
        c.allow_warped_motion = fh.allow_warped_motion.unwrap_or(false);
        c.is_scaled_per_ref = is_scaled_per_ref;
        c.enable_interintra_compound = seq.enable_interintra_compound;
        c.enable_masked_compound = seq.enable_masked_compound;
        c.enable_jnt_comp = seq.enable_jnt_comp;
        // §8.3.2 compound-idx `dist_equal` is per-block (the two refs'
        // relative distances); single-ref-only streams never read it.
        c.dist_equal = false;
        c.interpolation_filter = ir.interpolation_filter as u8;
        c.enable_dual_filter = seq.enable_dual_filter;
        c.pixels = Some(&pixels);
        Some(c)
    } else {
        None
    };

    // §8.3.1 frame-start CDF state: `load_cdfs( ref_frame_idx[
    // primary_ref_frame ] )` when a primary reference exists (the
    // §7.20-saved frame-end state of that slot, coefficient CDFs
    // included), otherwise `init_non_coeff_cdfs()` (the §9.4 defaults)
    // + the q-context-selected `init_coeff_cdfs( base_q_idx )`.
    let frame_start_cdfs: Box<TileCdfContext> = if fh.primary_ref_frame != PRIMARY_REF_NONE {
        let st = refs.ok_or(Error::PartitionWalkOutOfRange)?;
        let ir = fh
            .inter_refs
            .as_ref()
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let slot = ir.ref_frame_idx[fh.primary_ref_frame as usize] as usize;
        let slot_state = st
            .slots
            .get(slot)
            .and_then(|s| s.as_ref())
            .ok_or(Error::PartitionWalkOutOfRange)?;
        slot_state.cdfs.clone()
    } else {
        let mut c = TileCdfContext::new_from_defaults();
        c.init_coeff_cdfs(qp.base_q_idx);
        Box::new(c)
    };

    // §5.11.2 decode_tile() per tile, in tile order — each tile gets a
    // fresh §8.2.2 symbol decoder and a fresh copy of the frame-start
    // CDF state (§8.2.2 `clear_above_context` etc. are per-tile inside
    // `begin_tile`), while the walker's frame-scope decode grids
    // accumulate across tiles. The §5.11.57 `read_lr` interleave runs
    // when the frame signals loop restoration. Per §8.2.4
    // `exit_symbol` / §8.4 `frame_end_update_cdf`, the tile numbered
    // `context_update_tile_id` donates its adapted CDFs as the
    // frame-end state (unless `disable_frame_end_update_cdf`, which
    // keeps the frame-start state).
    let mut end_cdfs: Option<Box<TileCdfContext>> = None;
    for (tile_num, tile) in parsed.tiles.iter().enumerate() {
        let tile_row = (tile_num as u32) / ti.tile_cols;
        let tile_col = (tile_num as u32) % ti.tile_cols;
        walker.begin_tile(TileGeometry {
            mi_row_start: ti.mi_row_starts[tile_row as usize],
            mi_row_end: ti.mi_row_starts[tile_row as usize + 1],
            mi_col_start: ti.mi_col_starts[tile_col as usize],
            mi_col_end: ti.mi_col_starts[tile_col as usize + 1],
        });
        let mut decoder =
            SymbolDecoder::init_symbol(&tile.bytes, tile.bytes.len(), fh.disable_cdf_update)?;
        let mut cdfs = frame_start_cdfs.as_ref().clone();
        walker.decode_tile_syntax_with_lr(
            &mut decoder,
            &mut cdfs,
            &params,
            if lr.uses_lr { Some(&lr_walk) } else { None },
            /* inter_ctx = */ ictx.as_ref(),
            &quant,
            /* read_deltas = */ dq.delta_q_present,
        )?;
        if !fh.disable_frame_end_update_cdf && tile_num as u32 == ti.context_update_tile_id {
            end_cdfs = Some(Box::new(cdfs));
        }
    }
    let end_cdfs = end_cdfs.unwrap_or(frame_start_cdfs);

    // ---- Take `CurrFrame[ plane ]` at its FULL mi-grid extent. ----
    // The spec's CurrFrame covers the padded `MiRows x MiCols` grid
    // (frames whose dimensions are not mi-aligned carry decoded padding
    // columns/rows past the crop). The §7.4 in-loop passes read that
    // padding — the §7.14 wide-filter taps, the §7.15 mi-bounded
    // availability region, and especially the §7.16 upscaler's
    // `Clip3(0, miW * MI_SIZE - 1, ..)` source clamp — so the whole
    // chain runs at the padded extent and the §7.18.2 crop to the
    // output extents happens at the very end.
    let sub_x = u8::from(cc.subsampling_x) as u32;
    let sub_y = u8::from(cc.subsampling_y) as u32;
    let num_planes = cc.num_planes as usize;
    let mut plane_bufs: Vec<Vec<i32>> = Vec::with_capacity(num_planes);
    let mut plane_dims: Vec<(u32, u32)> = Vec::with_capacity(num_planes);
    for plane in 0..num_planes {
        let src = walker
            .curr_frame(plane)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let (rows, cols) = walker
            .curr_frame_dims(plane)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let (pw, ph) = if plane == 0 {
            (fs.frame_width, fs.frame_height)
        } else {
            (
                (fs.frame_width + sub_x) >> sub_x,
                (fs.frame_height + sub_y) >> sub_y,
            )
        };
        if rows < ph || cols < pw {
            return Err(Error::PartitionWalkOutOfRange);
        }
        plane_bufs.push(src.to_vec());
        plane_dims.push((cols, rows));
    }

    // ---- §7.4 in-loop passes: §7.14 deblock, then §7.15 CDEF. ----
    // §7.4 step 1: the loop filter is invoked ONLY when
    // `loop_filter_level[ 0 ] != 0 || loop_filter_level[ 1 ] != 0` —
    // with both luma levels zero the frame is NOT deblocked at all,
    // even though the §7.14.4 `loop_filter_delta_enabled` ref-delta
    // path could otherwise lift a per-edge strength above zero.
    if let Some(lf) = fh.loop_filter_params.as_ref() {
        if (lf.loop_filter_level[0] != 0 || lf.loop_filter_level[1] != 0)
            && !coded_lossless
            && !fh.allow_intrabc
        {
            let mut bufs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
            for (buf, &(pw, ph)) in plane_bufs.iter_mut().zip(plane_dims.iter()) {
                bufs.push(PlaneBuffer {
                    rows: ph,
                    cols: pw,
                    samples: buf,
                });
            }
            walker.loop_filter_frame_from_grid(
                lf,
                sp,
                dlf.delta_lf_multi,
                cc.num_planes,
                cc.bit_depth,
                sub_x as u8,
                sub_y as u8,
                fs.frame_width,
                fs.frame_height,
                &mut bufs,
            );
        }
    }
    // §7.4 steps 2-5: CDEF, then loop restoration. `plane_bufs` holds
    // the post-deblock CurrFrame here; §7.17 needs BOTH that frame (the
    // `UpscaledCurrFrame` — no superres on this path) and the CDEF
    // output (`UpscaledCdefFrame`), so keep the pre-CDEF copy around
    // when restoration is active.
    let deblocked: Option<Vec<Vec<i32>>> = if lr.uses_lr {
        Some(plane_bufs.clone())
    } else {
        None
    };
    if let Some(cdef) = fh.cdef_params.as_ref() {
        if !coded_lossless && !fh.allow_intrabc && seq.enable_cdef && !cdef.short_circuited {
            // §7.15 filters from the deblocked frame into a fresh copy.
            let src_bufs_data: Vec<Vec<i32>> = plane_bufs.clone();
            let mut src_owned = src_bufs_data;
            let mut src: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
            for (buf, &(pw, ph)) in src_owned.iter_mut().zip(plane_dims.iter()) {
                src.push(PlaneBuffer {
                    rows: ph,
                    cols: pw,
                    samples: buf,
                });
            }
            let mut dst: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
            for (buf, &(pw, ph)) in plane_bufs.iter_mut().zip(plane_dims.iter()) {
                dst.push(PlaneBuffer {
                    rows: ph,
                    cols: pw,
                    samples: buf,
                });
            }
            walker.cdef_frame_from_idx(
                cdef,
                cc.num_planes,
                cc.bit_depth,
                sub_x as u8,
                sub_y as u8,
                &src,
                &mut dst,
            );
        }
    }
    // §7.4 steps 3-4 / §7.16: horizontal superres upscaling of BOTH the
    // CDEF output (`plane_bufs` → UpscaledCdefFrame) and the post-
    // deblock copy (`deblocked` → UpscaledCurrFrame), ahead of loop
    // restoration. No-op when `use_superres == 0`.
    let mut deblocked = deblocked;
    if fs.use_superres && fs.upscaled_width > fs.frame_width {
        let sr_ctx = crate::superres::SuperresFrameContext {
            use_superres: true,
            frame_width: fs.frame_width,
            upscaled_width: fs.upscaled_width,
            frame_height: fs.frame_height,
            mi_cols: fs.mi_cols,
            num_planes: cc.num_planes,
            bit_depth: cc.bit_depth,
            subsampling_x: u8::from(cc.subsampling_x),
            subsampling_y: u8::from(cc.subsampling_y),
        };
        // §7.16 output extents: `upscaledPlaneW x planeH` — exact
        // (un-padded) per-plane dimensions.
        let mut new_dims: Vec<(u32, u32)> = Vec::with_capacity(num_planes);
        for plane in 0..num_planes {
            let (out_w, out_h) = if plane == 0 {
                (fs.upscaled_width, fs.frame_height)
            } else {
                (
                    (fs.upscaled_width + sub_x) >> sub_x,
                    (fs.frame_height + sub_y) >> sub_y,
                )
            };
            new_dims.push((out_w, out_h));
        }
        let upscale_set = |bufs: &mut Vec<Vec<i32>>| -> Result<(), Error> {
            let mut inputs_owned = std::mem::take(bufs);
            let mut inputs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
            for (buf, &(pw, ph)) in inputs_owned.iter_mut().zip(plane_dims.iter()) {
                inputs.push(PlaneBuffer {
                    rows: ph,
                    cols: pw,
                    samples: buf,
                });
            }
            let mut outputs_owned: Vec<Vec<i32>> = new_dims
                .iter()
                .map(|&(pw, ph)| vec![0i32; (pw as usize) * (ph as usize)])
                .collect();
            {
                let mut outputs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
                for (buf, &(pw, ph)) in outputs_owned.iter_mut().zip(new_dims.iter()) {
                    outputs.push(PlaneBuffer {
                        rows: ph,
                        cols: pw,
                        samples: buf,
                    });
                }
                crate::superres::upscale_frame(&sr_ctx, &inputs, &mut outputs)
                    .map_err(|_| Error::PartitionWalkOutOfRange)?;
            }
            *bufs = outputs_owned;
            Ok(())
        };
        upscale_set(&mut plane_bufs)?;
        if let Some(d) = deblocked.as_mut() {
            upscale_set(d)?;
        }
        plane_dims = new_dims;
    }

    // §7.4 step 5 / §7.17: loop restoration from (UpscaledCurrFrame,
    // UpscaledCdefFrame) into LrFrame. The §5.11.57 units were decoded
    // by the tile walk's `read_lr` interleave above.
    if let Some(mut curr_owned) = deblocked {
        let mut cdef_owned: Vec<Vec<i32>> = plane_bufs.clone();
        let mut curr: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
        for (buf, &(pw, ph)) in curr_owned.iter_mut().zip(plane_dims.iter()) {
            curr.push(PlaneBuffer {
                rows: ph,
                cols: pw,
                samples: buf,
            });
        }
        let mut cdef_bufs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
        for (buf, &(pw, ph)) in cdef_owned.iter_mut().zip(plane_dims.iter()) {
            cdef_bufs.push(PlaneBuffer {
                rows: ph,
                cols: pw,
                samples: buf,
            });
        }
        let mut lr_out: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
        for (buf, &(pw, ph)) in plane_bufs.iter_mut().zip(plane_dims.iter()) {
            lr_out.push(PlaneBuffer {
                rows: ph,
                cols: pw,
                samples: buf,
            });
        }
        walker.loop_restore_frame_from_grid(
            lr,
            cc.num_planes,
            cc.bit_depth,
            sub_x as u8,
            sub_y as u8,
            mi_rows,
            mi_cols,
            fs.frame_height,
            fs.upscaled_width,
            &curr,
            &cdef_bufs,
            &mut lr_out,
        );
    }

    // ---- §7.18.2 intermediate-output crop to the surfaced extents. ----
    // On the superres path the buffers are already exact; otherwise
    // trim the mi-grid padding down to `UpscaledWidth x FrameHeight`
    // (per-plane subsampled). §7.18.3 film grain then applies to the
    // cropped output planes.
    for plane in 0..num_planes {
        let (out_w, out_h) = if plane == 0 {
            (fs.upscaled_width, fs.frame_height)
        } else {
            (
                (fs.upscaled_width + sub_x) >> sub_x,
                (fs.frame_height + sub_y) >> sub_y,
            )
        };
        let (pw, ph) = plane_dims[plane];
        if (pw, ph) == (out_w, out_h) {
            continue;
        }
        if pw < out_w || ph < out_h {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let src = &plane_bufs[plane];
        let mut buf = vec![0i32; (out_w as usize) * (out_h as usize)];
        for y in 0..out_h as usize {
            let row = &src[y * pw as usize..y * pw as usize + out_w as usize];
            buf[y * out_w as usize..(y + 1) * out_w as usize].copy_from_slice(row);
        }
        plane_bufs[plane] = buf;
        plane_dims[plane] = (out_w, out_h);
    }

    // ---- §7.20 store payload: pre-grain cropped planes (u16). ----
    // §7.18.3 film grain applies to the OUTPUT copy only; the §7.20
    // reference store keeps the grain-free frame (the samples later
    // frames motion-compensate against).
    let ref_planes: Vec<Vec<u16>> = plane_bufs
        .iter()
        .map(|buf| buf.iter().map(|&v| v.max(0) as u16).collect())
        .collect();
    let ref_plane_dims: Vec<(u32, u32)> = plane_dims.clone();

    // ---- §7.18.3 film grain. ----
    if let Some(fg) = fh.film_grain_params.as_ref() {
        if fg.apply_grain {
            let mut bufs: Vec<PlaneBuffer<'_>> = Vec::with_capacity(num_planes);
            for (buf, &(pw, ph)) in plane_bufs.iter_mut().zip(plane_dims.iter()) {
                bufs.push(PlaneBuffer {
                    rows: ph,
                    cols: pw,
                    samples: buf,
                });
            }
            film_grain_synthesis(
                fg,
                cc.bit_depth,
                cc.num_planes,
                u8::from(cc.subsampling_x),
                u8::from(cc.subsampling_y),
                cc.matrix_coefficients,
                &mut bufs,
            );
        }
    }

    // ---- Narrow to 8-bit output. ----
    let planes: Vec<Vec<u8>> = plane_bufs
        .into_iter()
        .map(|buf| buf.into_iter().map(|v| v.clamp(0, 255) as u8).collect())
        .collect();

    // ---- §7.19 motion field motion vector storage. ----
    // Filter the §5.11.5 `Mvs[]` / `RefFrames[]` grids down to the
    // `MfMvs[]` / `MfRefFrames[]` payload §7.20 stores: per cell, the
    // LAST candidate list whose reference lies in the past
    // (`get_relative_dist( RefOrderHint[ refIdx ], OrderHint ) < 0`)
    // and whose MV components sit within `REFMVS_LIMIT`.
    const REFMVS_LIMIT: i16 = (1 << 12) - 1;
    let cells = (mi_rows as usize) * (mi_cols as usize);
    let mut mf_ref_frames: Vec<i8> = vec![-1; cells];
    let mut mf_mvs: Vec<i16> = vec![0; cells * 2];
    if is_inter_frame {
        let st = refs.ok_or(Error::PartitionWalkOutOfRange)?;
        let raw_refs = walker.ref_frames();
        let raw_mvs = walker.mvs();
        let hint_bits = if seq.enable_order_hint {
            u32::from(seq.order_hint_bits)
        } else {
            0
        };
        for cell in 0..cells {
            for list in 0..2usize {
                let r = raw_refs[cell * 2 + list];
                if r > 0 {
                    let slot = ref_frame_idx[(r - 1) as usize] as usize;
                    let dist = get_relative_dist(
                        st.info.order_hint[slot] as i32,
                        fh.order_hint as i32,
                        hint_bits,
                    );
                    if dist < 0 {
                        let mv_row = raw_mvs[(cell * 2 + list) * 2];
                        let mv_col = raw_mvs[(cell * 2 + list) * 2 + 1];
                        if mv_row.abs() <= REFMVS_LIMIT && mv_col.abs() <= REFMVS_LIMIT {
                            mf_ref_frames[cell] = r;
                            mf_mvs[cell * 2] = mv_row;
                            mf_mvs[cell * 2 + 1] = mv_col;
                        }
                    }
                }
            }
        }
    }

    Ok(DecodedFrameInternal {
        frame: SpecFrame {
            // Post-§7.16 the surfaced luma width is `UpscaledWidth`
            // (`== FrameWidth` when superres is off).
            width: fs.upscaled_width,
            height: fs.frame_height,
            planes,
            plane_dims,
        },
        ref_planes,
        ref_plane_dims,
        mf_mvs,
        mf_ref_frames,
        order_hints_by_ref,
        end_cdfs,
        mi_rows,
        mi_cols,
    })
}

/// §7.9 `motion_field_estimation()` — project the §7.20-saved motion
/// vectors of up to four reference frames onto the current frame's
/// 8×8 motion-field grid, one projected MV per (destination reference,
/// cell).
fn motion_field_estimation(
    refs: &SpecRefState,
    fh: &FrameHeader,
    seq: &SequenceHeader,
    mi_rows: u32,
    mi_cols: u32,
) -> Result<MotionFieldMvs, Error> {
    // §3 constants scoped to the §7.9 processes.
    const MFMV_STACK_SIZE: i32 = 3;
    const MAX_FRAME_DISTANCE: i32 = 31;
    const MAX_OFFSET_WIDTH: i32 = 8;
    const MAX_OFFSET_HEIGHT: i32 = 0;
    /// §7.9.3 `Div_Mult[ 32 ]`.
    const DIV_MULT: [i32; 32] = [
        0, 16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638, 1489, 1365, 1260, 1170,
        1092, 1024, 963, 910, 862, 819, 780, 744, 712, 682, 655, 630, 606, 585, 564, 546, 528,
    ];
    // Reference ordinals (§6.10.24): LAST=1, LAST2=2, BWDREF=5,
    // ALTREF2=6, ALTREF=7; GOLDEN=4.
    const LAST2_FRAME: usize = 2;
    const GOLDEN_FRAME: usize = 4;
    const BWDREF_FRAME: usize = 5;
    const ALTREF2_FRAME: usize = 6;

    let ir = fh
        .inter_refs
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let hint_bits = if seq.enable_order_hint {
        u32::from(seq.order_hint_bits)
    } else {
        0
    };
    let order_hint = fh.order_hint as i32;
    // §5.9.2 `OrderHints[ LAST_FRAME + i ]`.
    let mut order_hints = [0i32; ALTREF_FRAME + 1];
    for i in 0..7 {
        order_hints[LAST_FRAME + i] = refs.info.order_hint[ir.ref_frame_idx[i] as usize] as i32;
    }

    let mut mfmvs = MotionFieldMvs::new_invalid(mi_rows, mi_cols);
    let w8 = (mi_cols >> 1) as i32;
    let h8 = (mi_rows >> 1) as i32;

    // §7.9.2 projection for one source reference.
    let project_ref = |src: usize, dst_sign: i32, mfmvs: &mut MotionFieldMvs| -> bool {
        let src_idx = ir.ref_frame_idx[src - LAST_FRAME] as usize;
        let Some(slot) = refs.slots[src_idx].as_ref() else {
            return false;
        };
        if slot.mi_rows != mi_rows || slot.mi_cols != mi_cols || slot.frame_is_intra {
            return false;
        }
        for y8 in 0..h8 {
            for x8 in 0..w8 {
                let row = (2 * y8 + 1) as usize;
                let col = (2 * x8 + 1) as usize;
                let cell = row * (mi_cols as usize) + col;
                let src_ref = slot.mf_ref_frames[cell];
                if src_ref <= 0 {
                    continue;
                }
                let ref_to_cur = get_relative_dist(order_hints[src], order_hint, hint_bits);
                let ref_offset = get_relative_dist(
                    order_hints[src],
                    slot.saved_order_hints[src_ref as usize],
                    hint_bits,
                );
                let pos_valid = ref_to_cur.abs() <= MAX_FRAME_DISTANCE
                    && ref_offset.abs() <= MAX_FRAME_DISTANCE
                    && ref_offset > 0;
                if !pos_valid {
                    continue;
                }
                let mv = [
                    i32::from(slot.mf_mvs[cell * 2]),
                    i32::from(slot.mf_mvs[cell * 2 + 1]),
                ];
                // §7.9.3 get_mv_projection.
                let mv_projection = |mv: [i32; 2], numerator: i32, denominator: i32| -> [i32; 2] {
                    let clipped_den = denominator.min(MAX_FRAME_DISTANCE);
                    let clipped_num = numerator.clamp(-MAX_FRAME_DISTANCE, MAX_FRAME_DISTANCE);
                    let mut out = [0i32; 2];
                    for i in 0..2 {
                        let v = mv[i] * clipped_num * DIV_MULT[clipped_den as usize];
                        // Round2Signed(v, 14).
                        let scaled = if v >= 0 {
                            (v + (1 << 13)) >> 14
                        } else {
                            -((-v + (1 << 13)) >> 14)
                        };
                        out[i] = scaled.clamp(-(1 << 14) + 1, (1 << 14) - 1);
                    }
                    out
                };
                let proj_mv = mv_projection(mv, ref_to_cur * dst_sign, ref_offset);
                // §7.9.4 get_block_position.
                let project = |v8: i32, delta: i32, max8: i32, max_off8: i32| -> (i32, bool) {
                    let base8 = (v8 >> 3) << 3;
                    let offset8 = if delta >= 0 {
                        delta >> (3 + 1 + 2)
                    } else {
                        -((-delta) >> (3 + 1 + 2))
                    };
                    let v8 = v8 + dst_sign * offset8;
                    let valid = !(v8 < 0
                        || v8 >= max8
                        || v8 < base8 - max_off8
                        || v8 >= base8 + 8 + max_off8);
                    (v8, valid)
                };
                let (pos_y8, vy) = project(y8, proj_mv[0], h8, MAX_OFFSET_HEIGHT);
                let (pos_x8, vx) = project(x8, proj_mv[1], w8, MAX_OFFSET_WIDTH);
                if !(vy && vx) {
                    continue;
                }
                for (dst, &dst_hint) in order_hints
                    .iter()
                    .enumerate()
                    .take(ALTREF_FRAME + 1)
                    .skip(LAST_FRAME)
                {
                    let ref_to_dst = get_relative_dist(order_hint, dst_hint, hint_bits);
                    let out = mv_projection(mv, ref_to_dst, ref_offset);
                    mfmvs.set(
                        dst,
                        pos_y8 as u32,
                        pos_x8 as u32,
                        [out[0] as i16, out[1] as i16],
                    );
                }
            }
        }
        true
    };

    // §7.9.1 source-selection order.
    let last_idx = ir.ref_frame_idx[0] as usize;
    let cur_gold_order_hint = order_hints[GOLDEN_FRAME];
    let last_alt_order_hint = refs.slots[last_idx]
        .as_ref()
        .map_or(0, |s| s.saved_order_hints[ALTREF_FRAME]);
    let use_last = last_alt_order_hint != cur_gold_order_hint;
    if use_last {
        project_ref(LAST_FRAME, -1, &mut mfmvs);
    }
    let mut ref_stamp = MFMV_STACK_SIZE - 2;
    let use_bwd = get_relative_dist(order_hints[BWDREF_FRAME], order_hint, hint_bits) > 0;
    if use_bwd && project_ref(BWDREF_FRAME, 1, &mut mfmvs) {
        ref_stamp -= 1;
    }
    let use_alt2 = get_relative_dist(order_hints[ALTREF2_FRAME], order_hint, hint_bits) > 0;
    if use_alt2 && project_ref(ALTREF2_FRAME, 1, &mut mfmvs) {
        ref_stamp -= 1;
    }
    let use_alt = get_relative_dist(order_hints[ALTREF_FRAME], order_hint, hint_bits) > 0;
    if use_alt && ref_stamp >= 0 && project_ref(ALTREF_FRAME, 1, &mut mfmvs) {
        ref_stamp -= 1;
    }
    if ref_stamp >= 0 {
        project_ref(LAST2_FRAME, -1, &mut mfmvs);
    }
    Ok(mfmvs)
}

/// Decode an AV1 IVF v0 buffer through the spec-faithful frame driver.
///
/// Walks the IVF frame records, and within each temporal unit walks the
/// §5.2 OBU sequence: `OBU_TEMPORAL_DELIMITER` / `OBU_SEQUENCE_HEADER`
/// / `OBU_FRAME_HEADER` + `OBU_TILE_GROUP` / the combined `OBU_FRAME`
/// (§5.10: frame header + `byte_alignment()` + tile group in one OBU,
/// split via [`FrameHeader::bits_consumed`]). Padding and metadata OBUs
/// are skipped.
pub fn decode_av1_spec(input: &[u8]) -> Result<Vec<SpecFrame>, Error> {
    let reader = IvfReader::new(input).map_err(|_| Error::UnexpectedEnd)?;
    let records = reader.read_all().map_err(|_| Error::UnexpectedEnd)?;
    let mut out = Vec::new();
    let mut seq: Option<SequenceHeader> = None;
    let mut refs = SpecRefState::new();
    for record in records {
        decode_temporal_unit_spec(&record.payload, &mut seq, &mut refs, &mut out)?;
    }
    Ok(out)
}

/// §7.20 `reference_frame_update()` — store the just-decoded frame
/// into every slot `refresh_frame_flags` selects, updating the
/// §5.9.2 `RefInfo` arrays in lockstep.
fn reference_frame_update(
    refs: &mut SpecRefState,
    fh: &FrameHeader,
    decoded: &DecodedFrameInternal,
) -> Result<(), Error> {
    if fh.refresh_frame_flags == 0 {
        return Ok(());
    }
    let fs = fh
        .frame_size
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    // §7.20 per-slot payload: pixels + §7.19 motion-field grids +
    // `SavedOrderHints` + `save_cdfs` + `SavedGmParams` +
    // `save_loop_filter_params`.
    let mut gm_params = crate::uncompressed_header_tail::prev_gm_params_default();
    if let Some(g) = fh.global_motion_params.as_ref() {
        gm_params = g.gm_params;
    }
    let (lf_ref_deltas, lf_mode_deltas) = fh
        .loop_filter_params
        .as_ref()
        .map(|lf| (lf.loop_filter_ref_deltas, lf.loop_filter_mode_deltas))
        .unwrap_or((
            crate::uncompressed_header_tail::LOOP_FILTER_REF_DELTAS_DEFAULT,
            [0i8; 2],
        ));
    let payload = SpecRefSlot {
        planes: decoded.ref_planes.clone(),
        plane_dims: decoded.ref_plane_dims.clone(),
        mf_mvs: decoded.mf_mvs.clone(),
        mf_ref_frames: decoded.mf_ref_frames.clone(),
        saved_order_hints: decoded.order_hints_by_ref,
        mi_rows: decoded.mi_rows,
        mi_cols: decoded.mi_cols,
        frame_is_intra: fh.frame_is_intra,
        frame_type_is_key: matches!(fh.frame_type, crate::frame_header::FrameType::Key),
        cdfs: decoded.end_cdfs.clone(),
        gm_params,
        lf_ref_deltas,
        lf_mode_deltas,
    };
    for i in 0..NUM_REF_FRAMES as usize {
        if (fh.refresh_frame_flags >> i) & 1 != 0 {
            refs.info.valid[i] = true;
            refs.info.order_hint[i] = fh.order_hint;
            refs.info.frame_id[i] = fh.current_frame_id;
            refs.info.upscaled_width[i] = fs.upscaled_width;
            refs.info.frame_height[i] = fs.frame_height;
            refs.info.render_width[i] = fs.render_width;
            refs.info.render_height[i] = fs.render_height;
            refs.info.frame_type_is_key[i] = payload.frame_type_is_key;
            refs.info.saved_gm_params[i] = payload.gm_params;
            refs.info.saved_lf_ref_deltas[i] = payload.lf_ref_deltas;
            refs.info.saved_lf_mode_deltas[i] = payload.lf_mode_deltas;
            refs.slots[i] = Some(payload.clone());
        }
    }
    Ok(())
}

/// §7.21-adjacent `show_existing_frame` output: surface the stored
/// slot's (grain-free) planes as a [`SpecFrame`].
fn output_existing_frame(refs: &SpecRefState, fh: &FrameHeader) -> Result<SpecFrame, Error> {
    let idx = fh
        .frame_to_show_map_idx
        .ok_or(Error::PartitionWalkOutOfRange)? as usize;
    if idx >= NUM_REF_FRAMES as usize {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let slot = refs.slots[idx]
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    let planes: Vec<Vec<u8>> = slot
        .planes
        .iter()
        .map(|p| p.iter().map(|&v| v.min(255) as u8).collect())
        .collect();
    let (w, h) = *slot
        .plane_dims
        .first()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    Ok(SpecFrame {
        width: w,
        height: h,
        planes,
        plane_dims: slot.plane_dims.clone(),
    })
}

/// Decode one §7.5 temporal-unit body, appending every SHOWN frame to
/// `out` (`show_frame == 1` coded frames and `show_existing_frame`
/// outputs — the §7.4 output discipline) and updating the cached
/// sequence header + the §7.20 reference state.
fn decode_temporal_unit_spec(
    payload: &[u8],
    seq: &mut Option<SequenceHeader>,
    refs: &mut SpecRefState,
    out: &mut Vec<SpecFrame>,
) -> Result<(), Error> {
    let mut pending_fh: Option<FrameHeader> = None;
    for desc in ObuIter::new(payload) {
        let desc = desc?;
        match desc.obu_type {
            ObuType::TemporalDelimiter | ObuType::Padding | ObuType::Metadata => {}
            ObuType::SequenceHeader => {
                *seq = Some(parse_sequence_header(desc.payload)?);
            }
            ObuType::FrameHeader => {
                let s = seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let fh = parse_frame_header_with_refs(desc.payload, s, &refs.info)?;
                if fh.show_existing_frame {
                    out.push(output_existing_frame(refs, &fh)?);
                    // §7.4 / §7.21: a shown KEY frame re-loads the
                    // stored frame state and re-stores it into every
                    // slot (`refresh_frame_flags == allFrames` per the
                    // §5.9.2 show_existing arm). The wholesale slot
                    // clone IS the §7.21 load followed by the §7.20
                    // store — pixels, §7.19 grids, `SavedOrderHints`,
                    // CDFs, gm params and loop-filter deltas all ride
                    // the payload.
                    if fh.refresh_frame_flags != 0 {
                        let idx = fh
                            .frame_to_show_map_idx
                            .ok_or(Error::PartitionWalkOutOfRange)?
                            as usize;
                        let payload = refs.slots[idx]
                            .as_ref()
                            .cloned()
                            .ok_or(Error::PartitionWalkOutOfRange)?;
                        let loaded_order_hint = refs.info.order_hint[idx];
                        let loaded_frame_id = refs.info.frame_id[idx];
                        let loaded_uw = refs.info.upscaled_width[idx];
                        let loaded_fh = refs.info.frame_height[idx];
                        let loaded_rw = refs.info.render_width[idx];
                        let loaded_rh = refs.info.render_height[idx];
                        for i in 0..NUM_REF_FRAMES as usize {
                            if (fh.refresh_frame_flags >> i) & 1 != 0 {
                                refs.info.valid[i] = true;
                                refs.info.order_hint[i] = loaded_order_hint;
                                refs.info.frame_id[i] = loaded_frame_id;
                                refs.info.upscaled_width[i] = loaded_uw;
                                refs.info.frame_height[i] = loaded_fh;
                                refs.info.render_width[i] = loaded_rw;
                                refs.info.render_height[i] = loaded_rh;
                                refs.info.frame_type_is_key[i] = payload.frame_type_is_key;
                                refs.info.saved_gm_params[i] = payload.gm_params;
                                refs.info.saved_lf_ref_deltas[i] = payload.lf_ref_deltas;
                                refs.info.saved_lf_mode_deltas[i] = payload.lf_mode_deltas;
                                refs.slots[i] = Some(payload.clone());
                            }
                        }
                    }
                    pending_fh = None;
                } else {
                    pending_fh = Some(fh);
                }
            }
            ObuType::TileGroup => {
                let s = seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let fh = pending_fh.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let decoded = decode_frame_spec_full(s, fh, desc.payload, Some(refs))?;
                reference_frame_update(refs, fh, &decoded)?;
                if fh.show_frame {
                    out.push(decoded.frame);
                }
            }
            ObuType::Frame => {
                // §5.10 frame_obu: frame_header_obu() + byte_alignment()
                // + tile_group_obu(). The tile group starts at the next
                // byte boundary after the frame header.
                let s = seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let fh = parse_frame_header_with_refs(desc.payload, s, &refs.info)?;
                let tg_offset = fh.bits_consumed.div_ceil(8);
                if tg_offset > desc.payload.len() {
                    return Err(Error::UnexpectedEnd);
                }
                let tg_body = &desc.payload[tg_offset..];
                let decoded = decode_frame_spec_full(s, &fh, tg_body, Some(refs))?;
                reference_frame_update(refs, &fh, &decoded)?;
                if fh.show_frame {
                    out.push(decoded.frame);
                }
            }
            _ => return Err(Error::PartitionWalkOutOfRange),
        }
    }
    Ok(())
}
