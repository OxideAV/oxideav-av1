//! r460 — §5.9.11 / §7.14 loop-filter (deblocking) LEVEL election for
//! the intra drivers.
//!
//! Every intra frame this crate coded before r460 carried
//! `loop_filter_level = [0, 0, 0, 0]` — the §7.4 deblocking pass never
//! ran — so block-edge discontinuities went straight into CDEF / loop
//! restoration and the output. The election here runs the decoder's
//! own §7.14 filter (the write-side mirror's
//! [`PartitionWalker::loop_filter_pass_window_from_grid`], the same
//! grids the decoder reads: `MiSizes[]`, `Skips[]`, `TxSizes[]`,
//! `YModes[]`, `RefFrames[]`) on the post-reconstruction planes at
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
//!
//! r479 — every trial and the final filter run on ROW WINDOWS of the
//! plane (one superblock band plus its skirts, `i32`, ~1.5 MiB at 12
//! MP) instead of frame-sized `i32` copies of every plane (the r464
//! election converted the whole luma plane per trial: 46 MiB + the
//! 23 MiB `u16` result, eight times). The §7.14.1 order — pass 0
//! (vertical edges) over the whole plane, then pass 1 (horizontal
//! edges) — is kept exactly by a rolling window: pass 0 is applied
//! to rows as they enter the window (an edge of pass 0 touches only
//! its own rows), pass 1 band by band top to bottom (an edge of pass
//! 1 at row `e` reads `e - 7 ..= e + 6` and writes `e - 6 ..= e + 5`,
//! §7.14.6.2 / §7.14.6.4), and a row leaves the window once no later
//! edge can read it. The sampled probe (`effort = 1`) keeps its r464
//! semantics: both passes over the band's edges only, the rows
//! around the band unfiltered.

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

/// The frame geometry the windowed filter needs (the subset of
/// [`LfElectInput`] the final in-place apply shares with the
/// election).
#[derive(Clone, Copy)]
pub(crate) struct LfGeometry {
    pub width: usize,
    pub height: usize,
    pub chroma_w: usize,
    pub chroma_h: usize,
    pub frame_width: u32,
    pub frame_height: u32,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub num_planes: u8,
}

impl LfElectInput<'_> {
    /// The geometry [`apply_loop_filter`] takes.
    pub(crate) fn geometry(&self) -> LfGeometry {
        LfGeometry {
            width: self.width,
            height: self.height,
            chroma_w: self.chroma_w,
            chroma_h: self.chroma_h,
            frame_width: self.frame_width,
            frame_height: self.frame_height,
            bit_depth: self.bit_depth,
            subsampling_x: self.subsampling_x,
            subsampling_y: self.subsampling_y,
            num_planes: self.num_planes,
        }
    }

    fn recon_plane(&self, plane: usize) -> &[u16] {
        match plane {
            0 => self.recon_y,
            1 => self.recon_u,
            _ => self.recon_v,
        }
    }

    fn src_plane(&self, plane: usize) -> &[u16] {
        match plane {
            0 => &self.input.y,
            1 => &self.input.u,
            _ => &self.input.v,
        }
    }
}

const COARSE: [u8; 8] = [0, 4, 8, 12, 16, 24, 32, 48];

/// Band height in mi rows (one 64-sample superblock row).
const BAND_MI: u32 = 16;

/// Window skirt in mi rows: 4 mi rows = 16 luma / 8 chroma (4:2:0)
/// rows — at least the 7-row §7.14.6 reach on every plane, and a
/// multiple of the chroma `rowStep`.
const SKIRT_MI: u32 = 4;

/// The sampled bands the `effort = 1` ladder is scored on: every
/// fourth 64-row superblock band (`(mi_row_start, mi_row_end)` pairs)
/// when the picture has at least eight bands, else the whole frame.
fn probe_bands(mi_rows: u32, effort: u8) -> Vec<(u32, u32)> {
    let bands = mi_rows.div_ceil(BAND_MI);
    if effort >= 2 || bands < 8 {
        return vec![(0, mi_rows)];
    }
    (0..bands)
        .step_by(4)
        .map(|b| (b * BAND_MI, ((b + 1) * BAND_MI).min(mi_rows)))
        .collect()
}

fn lf_params(levels: [u8; 4]) -> LoopFilterParams {
    LoopFilterParams {
        loop_filter_level: levels,
        loop_filter_sharpness: 0,
        loop_filter_delta_enabled: false,
        loop_filter_delta_update: false,
        loop_filter_ref_deltas: LOOP_FILTER_REF_DELTAS_DEFAULT,
        loop_filter_mode_deltas: LOOP_FILTER_MODE_DELTAS_DEFAULT,
        short_circuited: false,
    }
}

/// Per-plane extents: `(plane width, plane height, coded width,
/// coded height, sub_y)`.
fn plane_dims(g: &LfGeometry, plane: usize) -> (usize, usize, usize, usize, u8) {
    let (fw, fh) = (g.frame_width as usize, g.frame_height as usize);
    if plane == 0 {
        (g.width, g.height, fw, fh, 0)
    } else {
        (
            g.chroma_w,
            g.chroma_h,
            (fw + usize::from(g.subsampling_x)) >> g.subsampling_x,
            (fh + usize::from(g.subsampling_y)) >> g.subsampling_y,
            g.subsampling_y,
        )
    }
}

/// One pass of §7.14 on `plane` over the edge rows `[r0, r1)`, the
/// samples in `window` (plane row `window_row0` at index 0).
#[allow(clippy::too_many_arguments)]
fn run_pass(
    mirror: &PartitionWalker,
    g: &LfGeometry,
    params: &LoopFilterParams,
    plane: usize,
    pass: u8,
    window: &mut [i32],
    cols: usize,
    window_row0: usize,
    r0: u32,
    r1: u32,
) {
    let seg = SegmentationParams::disabled();
    let mut pb = PlaneBuffer {
        rows: (window.len() / cols.max(1)) as u32,
        cols: cols as u32,
        samples: window,
    };
    mirror.loop_filter_pass_window_from_grid(
        params,
        &seg,
        false,
        g.num_planes,
        g.bit_depth,
        g.subsampling_x,
        g.subsampling_y,
        g.frame_width,
        g.frame_height,
        plane as u8,
        pass,
        &mut pb,
        window_row0 as u32,
        r0,
        r1,
    );
}

/// SSD of `rows` filtered rows (`i32`, `cols` wide, plane row `y0`
/// at index 0) against the source over the coded width.
fn ssd_rows(
    rows: &[i32],
    cols: usize,
    y0: usize,
    src: &[u16],
    stride: usize,
    cw: usize,
    ch: usize,
) -> u64 {
    let mut d = 0u64;
    for (i, row) in rows.chunks_exact(cols.max(1)).enumerate() {
        let y = y0 + i;
        if y >= ch {
            break;
        }
        let s = &src[y * stride..y * stride + cw];
        for (&a, &b) in row[..cw].iter().zip(s) {
            let e = i64::from(a) - i64::from(b);
            d += (e * e) as u64;
        }
    }
    d
}

/// r464 probe semantics on a window: both passes over the edges of
/// `[r0, r1)` only, the rows around the band unfiltered; returns the
/// SSD over the band's rows.
#[allow(clippy::too_many_arguments)]
fn band_probe_ssd(
    inp: &LfElectInput<'_>,
    g: &LfGeometry,
    params: &LoopFilterParams,
    plane: usize,
    r0: u32,
    r1: u32,
    scratch: &mut Vec<i32>,
) -> u64 {
    let (pw, ph, cw, ch, sub_y) = plane_dims(g, plane);
    let y0 = ((r0 as usize * 4) >> sub_y).min(ph);
    let y1 = ((r1 as usize * 4) >> sub_y).min(ph);
    if y1 <= y0 {
        return 0;
    }
    let skirt = (SKIRT_MI as usize * 4) >> sub_y;
    let w0 = y0.saturating_sub(skirt);
    let w1 = (y1 + skirt).min(ph);
    let recon = inp.recon_plane(plane);
    scratch.clear();
    scratch.extend(recon[w0 * pw..w1 * pw].iter().map(|&v| i32::from(v)));
    run_pass(inp.mirror, g, params, plane, 0, scratch, pw, w0, r0, r1);
    run_pass(inp.mirror, g, params, plane, 1, scratch, pw, w0, r0, r1);
    ssd_rows(
        &scratch[(y0 - w0) * pw..(y1 - w0) * pw],
        pw,
        y0,
        inp.src_plane(plane),
        pw,
        cw,
        ch,
    )
}

/// One step of [`rolling_filter`]: the window asks for plane rows
/// `[y0, y1)` of the UNFILTERED plane to be appended to `dst`, or
/// hands over the final rows from plane row `y` on (`i32`, one or
/// more rows). Loads always lie below every row handed over so far,
/// so an in-place consumer never overwrites a row it still has to
/// load.
pub(crate) enum Step<'a> {
    Load {
        y0: usize,
        y1: usize,
        dst: &'a mut Vec<i32>,
    },
    Emit {
        y: usize,
        rows: &'a [i32],
    },
}

/// The exact §7.14 frame process on `plane` through a rolling window
/// (see the module docs), driven through [`Step`]s.
fn rolling_filter(
    mirror: &PartitionWalker,
    g: &LfGeometry,
    params: &LoopFilterParams,
    plane: usize,
    scratch: &mut Vec<i32>,
    mut step: impl FnMut(Step<'_>),
) {
    let (pw, ph, _, _, sub_y) = plane_dims(g, plane);
    let mi_rows = mirror.mi_rows();
    let row_of = |mi: u32| ((mi as usize * 4) >> sub_y).min(ph);
    // Rows `[buf_y0, buf_y0 + scratch.len() / pw)` live in `scratch`;
    // mi rows below `loaded_mi` have been loaded and pass-0 filtered.
    scratch.clear();
    let mut buf_y0 = 0usize;
    let mut loaded_mi = 0u32;
    let mut emitted = 0usize;
    let mut r0 = 0u32;
    while r0 < mi_rows {
        let r1 = (r0 + BAND_MI).min(mi_rows);
        // Load (and pass-0 filter) the rows this band's pass-1 edges
        // reach: up to `SKIRT_MI` mi rows past the band.
        let need_mi = (r1 + SKIRT_MI).min(mi_rows);
        if need_mi > loaded_mi {
            let (ly0, ly1) = (row_of(loaded_mi), row_of(need_mi));
            step(Step::Load {
                y0: ly0,
                y1: ly1,
                dst: scratch,
            });
            run_pass(
                mirror, g, params, plane, 0, scratch, pw, buf_y0, loaded_mi, need_mi,
            );
            loaded_mi = need_mi;
        }
        run_pass(mirror, g, params, plane, 1, scratch, pw, buf_y0, r0, r1);
        // Rows no later edge can read or write (an edge at mi row >=
        // `r1` reaches 7 rows above its row): final.
        let final_to = if r1 >= mi_rows {
            ph
        } else {
            row_of(r1).saturating_sub(8)
        };
        if final_to > emitted {
            step(Step::Emit {
                y: emitted,
                rows: &scratch[(emitted - buf_y0) * pw..(final_to - buf_y0) * pw],
            });
            emitted = final_to;
            scratch.drain(..(final_to - buf_y0) * pw);
            buf_y0 = final_to;
        }
        r0 = r1;
    }
    if emitted < ph {
        step(Step::Emit {
            y: emitted,
            rows: &scratch[..(ph - buf_y0) * pw],
        });
    }
}

/// SSD of the exact frame process at `params` on `plane` over the
/// coded extent.
fn plane_ssd(
    inp: &LfElectInput<'_>,
    g: &LfGeometry,
    params: &LoopFilterParams,
    plane: usize,
    scratch: &mut Vec<i32>,
) -> u64 {
    let (pw, _, cw, ch, _) = plane_dims(g, plane);
    let src = inp.src_plane(plane);
    let recon = inp.recon_plane(plane);
    let mut d = 0u64;
    rolling_filter(inp.mirror, g, params, plane, scratch, |step| match step {
        Step::Load { y0, y1, dst } => {
            dst.extend(recon[y0 * pw..y1 * pw].iter().map(|&v| i32::from(v)));
        }
        Step::Emit { y, rows } => d += ssd_rows(rows, pw, y, src, pw, cw, ch),
    });
    d
}

/// Distortion of the planes in `which` (`0` = luma, `1` = U + V) at
/// `levels` over `bands`: the sampled probe when `bands` is a sample,
/// the exact frame process on the whole frame.
fn trial_ssd(
    inp: &LfElectInput<'_>,
    g: &LfGeometry,
    levels: [u8; 4],
    which: u8,
    bands: &[(u32, u32)],
    whole: bool,
) -> u64 {
    let params = lf_params(levels);
    let planes: &[usize] = if which == 0 { &[0] } else { &[1, 2] };
    let mut scratch: Vec<i32> = Vec::new();
    let mut d = 0u64;
    for &plane in planes {
        if plane >= inp.num_planes as usize {
            break;
        }
        if whole {
            d += plane_ssd(inp, g, &params, plane, &mut scratch);
        } else {
            for &(r0, r1) in bands {
                d += band_probe_ssd(inp, g, &params, plane, r0, r1, &mut scratch);
            }
        }
    }
    d
}

/// Unfiltered distortion of the planes in `which` over `bands`.
fn base_ssd(inp: &LfElectInput<'_>, g: &LfGeometry, which: u8, bands: &[(u32, u32)]) -> u64 {
    let planes: &[usize] = if which == 0 { &[0] } else { &[1, 2] };
    let mut d = 0u64;
    for &plane in planes {
        if plane >= inp.num_planes as usize {
            break;
        }
        let (pw, ph, cw, ch, sub_y) = plane_dims(g, plane);
        let (rec, src) = (inp.recon_plane(plane), inp.src_plane(plane));
        for &(r0, r1) in bands {
            let y0 = ((r0 as usize * 4) >> sub_y).min(ph);
            let y1 = ((r1 as usize * 4) >> sub_y).min(ph).min(ch);
            for y in y0..y1 {
                for (&a, &b) in rec[y * pw..y * pw + cw]
                    .iter()
                    .zip(&src[y * pw..y * pw + cw])
                {
                    let e = i64::from(a) - i64::from(b);
                    d += (e * e) as u64;
                }
            }
        }
    }
    d
}

/// Elect the loop-filter levels (see the module docs). Returns `None`
/// when no non-zero level improves on the unfiltered reconstruction;
/// the caller applies the elected header through
/// [`apply_loop_filter`].
///
/// r464 — the `effort = 1` ladder is scored on sampled superblock
/// bands ([`probe_bands`]).
pub(crate) fn elect_loop_filter(inp: &LfElectInput<'_>) -> Option<LoopFilterParams> {
    let g = inp.geometry();
    let mi_rows = inp.mirror.mi_rows();
    let bands = probe_bands(mi_rows, inp.effort);
    let whole = bands.len() == 1 && bands[0] == (0, mi_rows);
    let d0 = base_ssd(inp, &g, 0, &bands);
    let mut tried: Vec<(u8, u64)> = vec![(0, d0)];
    let mut best = (0u8, d0);
    let trial = |lvl: u8, best: &mut (u8, u64), tried: &mut Vec<(u8, u64)>| {
        if tried.iter().any(|&(l, _)| l == lvl) {
            return;
        }
        let d = trial_ssd(inp, &g, [lvl, lvl, 0, 0], 0, &bands, whole);
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
        let mut best_c = (0u8, base_ssd(inp, &g, 1, &bands));
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
            let d = trial_ssd(inp, &g, [ly, ly, c, c], 1, &bands, whole);
            if d < best_c.1 {
                best_c = (c, d);
            }
        }
        lc = best_c.0;
    }
    Some(lf_params([ly, ly, lc, lc]))
}

/// Apply the elected §7.14 filter to the planes in place (the exact
/// frame process, luma and chroma in parallel), one rolling window
/// per plane.
pub(crate) fn apply_loop_filter(
    mirror: &PartitionWalker,
    params: &LoopFilterParams,
    g: &LfGeometry,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
) {
    let filter_plane = |plane: usize, buf: &mut [u16]| {
        if plane > 0 && params.loop_filter_level[1 + plane] == 0 {
            return;
        }
        let (pw, _, _, _, _) = plane_dims(g, plane);
        let mut scratch: Vec<i32> = Vec::new();
        rolling_filter(mirror, g, params, plane, &mut scratch, |step| match step {
            Step::Load { y0, y1, dst } => {
                dst.extend(buf[y0 * pw..y1 * pw].iter().map(|&v| i32::from(v)));
            }
            Step::Emit { y, rows } => {
                for (d, &v) in buf[y * pw..y * pw + rows.len()].iter_mut().zip(rows) {
                    *d = v.max(0) as u16;
                }
            }
        });
    };
    if g.num_planes > 1 {
        std::thread::scope(|scope| {
            let h = scope.spawn(|| filter_plane(0, recon_y));
            filter_plane(1, recon_u);
            filter_plane(2, recon_v);
            h.join().expect("luma deblock thread panicked");
        });
    } else {
        filter_plane(0, recon_y);
    }
}
