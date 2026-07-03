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
//!   4. §7.4 in-loop / post passes in decode order: §7.14 deblock →
//!      §7.15 CDEF → §7.18.3 film grain. (§7.16 superres and §7.17
//!      loop-restoration are follow-ups; frames signalling them are
//!      rejected rather than decoded wrongly.)
//!
//! ## Scope
//!
//! * Intra-only frames (KEY / INTRA_ONLY), 8-bit, single tile.
//! * 4:2:0 / 4:2:2 / 4:4:4 and monochrome layouts (the walker threads
//!   `subsampling_x/y` + `mono_chrome`; only 8-bit output is surfaced).
//! * Frames with `use_superres == 1` or an active §5.9.20 loop-
//!   restoration type return [`Error::PartitionWalkOutOfRange`] until
//!   those passes are wired here.
//!
//! Spec provenance: `docs/video/av1/av1-spec.txt` §5.9, §5.11, §7.4,
//! §7.12.2, §7.14, §7.15, §7.18.3.

use crate::cdf::{
    PartitionWalker, QuantizerParams, TileCdfContext, TileDecodeParams, TileGeometry,
};
use crate::encoder::ivf::IvfReader;
use crate::encoder::tile_group_obu::parse_tile_group_obu_body;
use crate::film_grain::film_grain_synthesis;
use crate::frame_header::{parse_frame_header, FrameHeader};
use crate::loop_filter::PlaneBuffer;
use crate::obu::{ObuIter, ObuType};
use crate::sequence_header::{parse_sequence_header, SequenceHeader};
use crate::symbol_decoder::SymbolDecoder;
use crate::uncompressed_header_tail::{
    SegmentationParams, MAX_SEGMENTS, SEG_LVL_ALT_Q, SEG_LVL_SKIP,
};
use crate::Error;

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
    if fh.show_existing_frame || !fh.frame_is_intra {
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
    if fs.use_superres {
        // §7.16 superres is not wired into this driver yet.
        return Err(Error::PartitionWalkOutOfRange);
    }
    let ti = fh
        .tile_info
        .as_ref()
        .ok_or(Error::PartitionWalkOutOfRange)?;
    if ti.tile_cols != 1 || ti.tile_rows != 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
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

    // §5.11.1 tile-group body → single tile's bytes.
    let parsed = parse_tile_group_obu_body(
        tile_group_body,
        /* num_tiles = */ 1,
        /* tile_cols_log2 = */ 0,
        /* tile_rows_log2 = */ 0,
        u32::from(ti.tile_size_bytes),
    )?;
    if parsed.tiles.len() != 1 {
        return Err(Error::PartitionWalkOutOfRange);
    }
    let tile_bytes = &parsed.tiles[0].bytes;

    // §8.2.2 init_symbol + §8.3.1 CDF init (defaults + the q-context
    // coefficient-CDF selection keyed by base_q_idx).
    let mut decoder =
        SymbolDecoder::init_symbol(tile_bytes, tile_bytes.len(), fh.disable_cdf_update)?;
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.init_coeff_cdfs(qp.base_q_idx);

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

    // §5.11.2 decode_tile() — with the §5.11.57 `read_lr` interleave
    // when the frame signals loop restoration.
    walker.decode_tile_syntax_with_lr(
        &mut decoder,
        &mut cdfs,
        &params,
        if lr.uses_lr { Some(&lr_walk) } else { None },
        /* inter_ctx = */ None,
        &quant,
        /* read_deltas = */ dq.delta_q_present,
    )?;

    // ---- Crop `CurrFrame[ plane ]` to the §5.9.8 frame extents. ----
    let sub_x = u8::from(cc.subsampling_x) as u32;
    let sub_y = u8::from(cc.subsampling_y) as u32;
    let num_planes = cc.num_planes as usize;
    let mut plane_bufs: Vec<Vec<i32>> = Vec::with_capacity(num_planes);
    let mut plane_dims: Vec<(u32, u32)> = Vec::with_capacity(num_planes);
    for plane in 0..num_planes {
        let (pw, ph) = if plane == 0 {
            (fs.frame_width, fs.frame_height)
        } else {
            (
                (fs.frame_width + sub_x) >> sub_x,
                (fs.frame_height + sub_y) >> sub_y,
            )
        };
        let src = walker
            .curr_frame(plane)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        let (rows, cols) = walker
            .curr_frame_dims(plane)
            .ok_or(Error::PartitionWalkOutOfRange)?;
        if rows < ph || cols < pw {
            return Err(Error::PartitionWalkOutOfRange);
        }
        let mut buf = vec![0i32; (pw as usize) * (ph as usize)];
        for y in 0..ph as usize {
            let src_row = &src[y * cols as usize..y * cols as usize + pw as usize];
            buf[y * pw as usize..(y + 1) * pw as usize].copy_from_slice(src_row);
        }
        plane_bufs.push(buf);
        plane_dims.push((pw, ph));
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

    Ok(SpecFrame {
        width: fs.frame_width,
        height: fs.frame_height,
        planes,
        plane_dims,
    })
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
    for record in records {
        decode_temporal_unit_spec(&record.payload, &mut seq, &mut out)?;
    }
    Ok(out)
}

/// Decode one §7.5 temporal-unit body, appending every decoded frame to
/// `out` and updating the cached sequence header.
fn decode_temporal_unit_spec(
    payload: &[u8],
    seq: &mut Option<SequenceHeader>,
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
                pending_fh = Some(parse_frame_header(desc.payload, s)?);
            }
            ObuType::TileGroup => {
                let s = seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let fh = pending_fh.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                out.push(decode_frame_spec(s, fh, desc.payload)?);
            }
            ObuType::Frame => {
                // §5.10 frame_obu: frame_header_obu() + byte_alignment()
                // + tile_group_obu(). The tile group starts at the next
                // byte boundary after the frame header.
                let s = seq.as_ref().ok_or(Error::PartitionWalkOutOfRange)?;
                let fh = parse_frame_header(desc.payload, s)?;
                let tg_offset = fh.bits_consumed.div_ceil(8);
                if tg_offset > desc.payload.len() {
                    return Err(Error::UnexpectedEnd);
                }
                let tg_body = &desc.payload[tg_offset..];
                out.push(decode_frame_spec(s, &fh, tg_body)?);
            }
            _ => return Err(Error::PartitionWalkOutOfRange),
        }
    }
    Ok(())
}
