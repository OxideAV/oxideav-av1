//! r460 — §5.9.11 / §7.14 loop-filter (deblocking) LEVEL election for
//! the intra drivers.
//!
//! Every intra frame this crate coded before r460 carried
//! `loop_filter_level = [0, 0, 0, 0]` — the §7.4 deblocking pass never
//! ran — so block-edge discontinuities went straight into CDEF / loop
//! restoration and the output. The election here runs the decoder's
//! own §7.14 filter (the write-side mirror's
//! [`PartitionWalker::loop_filter_frame_from_grid`], the same grids the
//! decoder reads: `MiSizes[]`, `Skips[]`, `TxSizes[]`, `YModes[]`,
//! `RefFrames[]`) on a copy of the post-reconstruction planes at
//! candidate strengths and keeps the level with the lowest SSD against
//! the source over the coded extent — luma first (`loop_filter_level[
//! 0 ] = loop_filter_level[ 1 ]`), then U and V at the elected luma
//! level. No rate term: the level costs a fixed 6 + 6 (+ 6 + 6) header
//! bits whichever value it takes.
//!
//! The search is coarse-to-fine over the §5.9.11 `0..=63` range:
//! `[0, 4, 8, 12, 16, 24, 32, 48]` then `±2`, `±1` around the winner
//! (`effort = 2`), or the coarse ladder alone (`effort = 1`).
//! `loop_filter_sharpness = 0`, `loop_filter_delta_enabled = 0` (the
//! §7.14.4 baseline level applies to every edge — an intra frame has
//! one reference class).

use crate::cdf::PartitionWalker;
use crate::encoder::yuv_frame::YuvFrame;
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::{
    LoopFilterParams, SegmentationParams, LOOP_FILTER_MODE_DELTAS_DEFAULT,
    LOOP_FILTER_REF_DELTAS_DEFAULT,
};

/// Inputs to [`elect_loop_filter`].
pub(crate) struct LfElectInput<'a> {
    /// The write-side mirror after the tile walk (the §7.14 per-mi
    /// grids).
    pub mirror: &'a PartitionWalker,
    /// The source picture (padded to the plane extent).
    pub input: &'a YuvFrame,
    /// Post-reconstruction planes at the (padded) plane extent.
    pub recon_y: &'a [u16],
    pub recon_u: &'a [u16],
    pub recon_v: &'a [u16],
    /// Plane extents.
    pub width: usize,
    pub height: usize,
    pub chroma_w: usize,
    pub chroma_h: usize,
    /// The coded (true) frame extent — the §7.14.2 on-screen check
    /// and the SSD window.
    pub frame_width: u32,
    pub frame_height: u32,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub num_planes: u8,
    /// `1` = coarse ladder, `2` = coarse + refinement.
    pub effort: u8,
}

/// The elected header block plus the deblocked planes.
pub(crate) struct LfElection {
    pub params: LoopFilterParams,
    pub y: Vec<u16>,
    pub u: Vec<u16>,
    pub v: Vec<u16>,
}

const COARSE: [u8; 8] = [0, 4, 8, 12, 16, 24, 32, 48];

fn ssd_plane(a: &[u16], b: &[u16], stride: usize, w: usize, h: usize) -> u64 {
    let mut d = 0u64;
    for y in 0..h {
        let ra = &a[y * stride..y * stride + w];
        let rb = &b[y * stride..y * stride + w];
        for (&x, &s) in ra.iter().zip(rb) {
            let e = i64::from(x) - i64::from(s);
            d += (e * e) as u64;
        }
    }
    d
}

/// Run the §7.14 filter at `levels` over copies of the input planes.
fn filter_at(inp: &LfElectInput<'_>, levels: [u8; 4]) -> (Vec<u16>, Vec<u16>, Vec<u16>) {
    let params = LoopFilterParams {
        loop_filter_level: levels,
        loop_filter_sharpness: 0,
        loop_filter_delta_enabled: false,
        loop_filter_delta_update: false,
        loop_filter_ref_deltas: LOOP_FILTER_REF_DELTAS_DEFAULT,
        loop_filter_mode_deltas: LOOP_FILTER_MODE_DELTAS_DEFAULT,
        short_circuited: false,
    };
    let seg = SegmentationParams::disabled();
    let to_i32 = |p: &[u16]| -> Vec<i32> { p.iter().map(|&v| i32::from(v)).collect() };
    let num_planes = usize::from(inp.num_planes.min(3));
    let mut owned: Vec<Vec<i32>> = vec![to_i32(inp.recon_y)];
    if num_planes > 1 {
        owned.push(to_i32(inp.recon_u));
        owned.push(to_i32(inp.recon_v));
    }
    let dims = [
        (inp.width, inp.height),
        (inp.chroma_w, inp.chroma_h),
        (inp.chroma_w, inp.chroma_h),
    ];
    {
        let mut bufs: Vec<PlaneBuffer<'_>> = owned
            .iter_mut()
            .zip(dims.iter())
            .map(|(buf, &(w, h))| PlaneBuffer {
                rows: h as u32,
                cols: w as u32,
                samples: buf.as_mut_slice(),
            })
            .collect();
        inp.mirror.loop_filter_frame_from_grid(
            &params,
            &seg,
            false,
            inp.num_planes,
            inp.bit_depth,
            inp.subsampling_x,
            inp.subsampling_y,
            inp.frame_width,
            inp.frame_height,
            &mut bufs,
        );
    }
    let back = |p: &[i32]| -> Vec<u16> { p.iter().map(|&v| v.max(0) as u16).collect() };
    let y = back(&owned[0]);
    let (u, v) = if num_planes > 1 {
        (back(&owned[1]), back(&owned[2]))
    } else {
        (Vec::new(), Vec::new())
    };
    (y, u, v)
}

/// Elect the loop-filter levels (see the module docs). Returns `None`
/// when no non-zero level improves on the unfiltered reconstruction.
pub(crate) fn elect_loop_filter(inp: &LfElectInput<'_>) -> Option<LfElection> {
    let fw = inp.frame_width as usize;
    let fh = inp.frame_height as usize;
    let (cfw, cfh) = (
        (fw + usize::from(inp.subsampling_x)) >> inp.subsampling_x,
        (fh + usize::from(inp.subsampling_y)) >> inp.subsampling_y,
    );
    let luma_d = |y: &[u16]| ssd_plane(y, &inp.input.y, inp.width, fw, fh);
    let chroma_d = |u: &[u16], v: &[u16]| {
        ssd_plane(u, &inp.input.u, inp.chroma_w, cfw, cfh)
            + ssd_plane(v, &inp.input.v, inp.chroma_w, cfw, cfh)
    };
    let d0 = luma_d(inp.recon_y);
    let mut tried: Vec<(u8, u64)> = vec![(0, d0)];
    let mut best = (0u8, d0);
    let trial = |lvl: u8, best: &mut (u8, u64), tried: &mut Vec<(u8, u64)>| {
        if tried.iter().any(|&(l, _)| l == lvl) {
            return;
        }
        let (y, _, _) = filter_at(inp, [lvl, lvl, 0, 0]);
        let d = luma_d(&y);
        tried.push((lvl, d));
        if d < best.1 {
            *best = (lvl, d);
        }
    };
    for &l in COARSE.iter().skip(1) {
        trial(l, &mut best, &mut tried);
    }
    if inp.effort >= 2 {
        for step in [2u8, 1] {
            let centre = best.0;
            for cand in [
                centre.saturating_sub(step),
                centre.saturating_add(step).min(63),
            ] {
                trial(cand, &mut best, &mut tried);
            }
        }
    }
    let ly = best.0;
    if ly == 0 {
        return None;
    }
    // Chroma at the elected luma level.
    let mut lc = 0u8;
    if inp.num_planes > 1 {
        let (_, u0, v0) = filter_at(inp, [ly, ly, 0, 0]);
        let mut best_c = (0u8, chroma_d(&u0, &v0));
        let mut tried_c: Vec<u8> = vec![0];
        let ladder: Vec<u8> = if inp.effort >= 2 {
            vec![ly / 2, ly, ly.saturating_add(ly / 2).min(63)]
        } else {
            vec![ly]
        };
        for c in ladder {
            if c == 0 || tried_c.contains(&c) {
                continue;
            }
            tried_c.push(c);
            let (_, u, v) = filter_at(inp, [ly, ly, c, c]);
            let d = chroma_d(&u, &v);
            if d < best_c.1 {
                best_c = (c, d);
            }
        }
        lc = best_c.0;
    }
    let levels = [ly, ly, lc, lc];
    let (y, u, v) = filter_at(inp, levels);
    Some(LfElection {
        params: LoopFilterParams {
            loop_filter_level: levels,
            loop_filter_sharpness: 0,
            loop_filter_delta_enabled: false,
            loop_filter_delta_update: false,
            loop_filter_ref_deltas: LOOP_FILTER_REF_DELTAS_DEFAULT,
            loop_filter_mode_deltas: LOOP_FILTER_MODE_DELTAS_DEFAULT,
            short_circuited: false,
        },
        y,
        u,
        v,
    })
}
