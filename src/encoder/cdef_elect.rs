//! r428 — encoder-side §5.9.19 / §7.15 CDEF election (encoder ladder
//! item 3, frame-level arm).
//!
//! The decoder's CDEF is corpus-complete; this module mirrors it on
//! the ENCODER's reconstruction path: after the tile is committed the
//! frame's pre-CDEF reconstruction is filtered through the decoder's
//! own §7.15 driver ([`crate::cdef::cdef_frame`]) over the write
//! mirror's committed grids (the §5.11.56 `cdef_idx[]` anchors and
//! the §7.15.1 `Skips[]` conjunction — exactly the state the decoder
//! derives from the emitted tile), a bounded strength search scores
//! each candidate against the SOURCE, and the winner (when it beats
//! the unfiltered frame) is stamped into the header and applied to
//! the reconstruction — so the stored reference planes equal the
//! decoder's §7.20 store byte-for-byte, like every other stage of
//! this encoder.
//!
//! Frame-level scope: `cdef_bits = 0` (one strength set for the whole
//! frame) — the §5.11.56 `cdef_idx` literal is `L(0)`, ZERO tile
//! bits, so the election is pure distortion: filtered-vs-source SSD
//! against unfiltered-vs-source SSD, per plane set. The search
//! strategy (coarse-then-refine primary sweep, secondary set, damping
//! sweep on the winner) is free encoder engineering; every candidate
//! is evaluated through the real §7.15 kernels.

use crate::cdf::PartitionWalker;
use crate::encoder::yuv_frame::YuvFrame;
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::CdefParams;

/// One plane set's strength candidate.
#[derive(Clone, Copy)]
struct Strength {
    pri: u8,
    sec: u8,
}

/// The frame-level CDEF election. `recon_*` are the committed
/// pre-CDEF reconstruction planes (the §7.14 deblock levels this
/// encoder codes are 0, so the reconstruction IS the CDEF input).
/// Returns the elected params — and has applied them to the
/// reconstruction — or `None` when no candidate beat the unfiltered
/// frame (the caller keeps the all-zero header shape).
#[allow(clippy::too_many_arguments)]
pub(crate) fn elect_and_apply_cdef(
    mirror: &PartitionWalker,
    input: &YuvFrame,
    recon_y: &mut [u16],
    recon_u: &mut [u16],
    recon_v: &mut [u16],
    width: usize,
    height: usize,
    chroma_w: usize,
    chroma_h: usize,
    bit_depth: u8,
    subsampling_x: u8,
    subsampling_y: u8,
    num_planes: u8,
) -> Option<CdefParams> {
    // Pre-CDEF source buffers (§7.15 filters CurrFrame → CdefFrame;
    // the driver reads ONLY from `src`, so one immutable copy serves
    // every candidate).
    let planes: Vec<(usize, usize)> = if num_planes > 1 {
        vec![(width, height), (chroma_w, chroma_h), (chroma_w, chroma_h)]
    } else {
        vec![(width, height)]
    };
    let mut src_owned: Vec<Vec<i32>> = Vec::with_capacity(planes.len());
    src_owned.push(recon_y.iter().map(|&v| i32::from(v)).collect());
    if num_planes > 1 {
        src_owned.push(recon_u.iter().map(|&v| i32::from(v)).collect());
        src_owned.push(recon_v.iter().map(|&v| i32::from(v)).collect());
    }
    let mut dst_owned: Vec<Vec<i32>> = src_owned.clone();

    // Per-plane-set SSD vs the source after one §7.15 run under the
    // given params. Luma and chroma filter independently (their
    // strengths never cross planes), so each side searches on its own
    // SSD; `which = 0` scores luma, `1` scores chroma.
    let mut run = |params: &CdefParams, which: u8| -> u64 {
        let src: Vec<PlaneBuffer<'_>> = src_owned
            .iter_mut()
            .zip(planes.iter())
            .map(|(buf, &(pw, ph))| PlaneBuffer {
                rows: ph as u32,
                cols: pw as u32,
                samples: buf,
            })
            .collect();
        let mut dst: Vec<PlaneBuffer<'_>> = dst_owned
            .iter_mut()
            .zip(planes.iter())
            .map(|(buf, &(pw, ph))| PlaneBuffer {
                rows: ph as u32,
                cols: pw as u32,
                samples: buf,
            })
            .collect();
        mirror.cdef_frame_from_idx(
            params,
            num_planes,
            bit_depth,
            subsampling_x,
            subsampling_y,
            &src,
            &mut dst,
        );
        let mut ssd = 0u64;
        if which == 0 {
            for (a, b) in dst_owned[0].iter().zip(input.y.iter()) {
                let d = i64::from(*a) - i64::from(*b);
                ssd += (d * d) as u64;
            }
        } else {
            for (a, b) in dst_owned[1]
                .iter()
                .zip(input.u.iter())
                .chain(dst_owned[2].iter().zip(input.v.iter()))
            {
                let d = i64::from(*a) - i64::from(*b);
                ssd += (d * d) as u64;
            }
        }
        ssd
    };

    let params_for = |damping: u8, y: Strength, uv: Strength| -> CdefParams {
        let mut p = CdefParams::short_circuit();
        p.short_circuited = false;
        p.cdef_damping = damping;
        p.cdef_bits = 0;
        p.cdef_y_pri_strength[0] = y.pri;
        p.cdef_y_sec_strength[0] = y.sec;
        p.cdef_uv_pri_strength[0] = uv.pri;
        p.cdef_uv_sec_strength[0] = uv.sec;
        p
    };
    let zero = Strength { pri: 0, sec: 0 };

    // Baseline: unfiltered SSD per plane set.
    let base_y: u64 = {
        let mut ssd = 0u64;
        for (a, b) in recon_y.iter().zip(input.y.iter()) {
            let d = i64::from(*a) - i64::from(*b);
            ssd += (d * d) as u64;
        }
        ssd
    };
    let base_uv: u64 = if num_planes > 1 {
        let mut ssd = 0u64;
        for (a, b) in recon_u
            .iter()
            .zip(input.u.iter())
            .chain(recon_v.iter().zip(input.v.iter()))
        {
            let d = i64::from(*a) - i64::from(*b);
            ssd += (d * d) as u64;
        }
        ssd
    } else {
        0
    };

    if std::env::var_os("OXIDEAV_AV1_CDEF_DEBUG").is_some() {
        let mut stamped = 0usize;
        let mut total = 0usize;
        let mi_rows = mirror.mi_rows();
        let mi_cols = mirror.mi_cols();
        let grid = mirror.cdef_idx();
        for r in (0..mi_rows).step_by(16) {
            for c in (0..mi_cols).step_by(16) {
                total += 1;
                if grid[(r * mi_cols + c) as usize] != -1 {
                    stamped += 1;
                }
            }
        }
        let probe = params_for(3, Strength { pri: 4, sec: 2 }, Strength { pri: 4, sec: 2 });
        let y4 = run(&probe, 0);
        eprintln!(
            "cdef-debug: anchors {stamped}/{total} stamped, base_y {base_y}, base_uv {base_uv}, y@pri4sec2 {y4}"
        );
    }
    // Coarse-then-refine strength search for one plane set at a fixed
    // damping (the other plane set stays at zero — identity there).
    let mut search_set = |damping: u8, which: u8, base: u64| -> (Strength, u64) {
        let mut best = (zero, base);
        let mut eval = |best: &mut (Strength, u64), s: Strength| {
            if s.pri == 0 && s.sec == 0 {
                return;
            }
            let p = if which == 0 {
                params_for(damping, s, zero)
            } else {
                params_for(damping, zero, s)
            };
            let ssd = run(&p, which);
            if ssd < best.1 {
                *best = (s, ssd);
            }
        };
        for pri in [0u8, 1, 2, 3, 4, 6, 9, 12, 15] {
            for sec in [0u8, 2] {
                eval(&mut best, Strength { pri, sec });
            }
        }
        let center = best.0;
        for pri in center.pri.saturating_sub(1)..=(center.pri + 1).min(15) {
            for sec in [0u8, 1, 2, 4] {
                if (pri, sec) != (center.pri, center.sec) {
                    eval(&mut best, Strength { pri, sec });
                }
            }
        }
        best
    };

    let mut best_total: Option<(u8, Strength, Strength, u64)> = None;
    for damping in [3u8, 5] {
        let (y_s, y_ssd) = search_set(damping, 0, base_y);
        let (uv_s, uv_ssd) = if num_planes > 1 {
            search_set(damping, 1, base_uv)
        } else {
            (zero, 0)
        };
        let total = y_ssd + uv_ssd;
        if best_total.is_none() || total < best_total.as_ref().unwrap().3 {
            best_total = Some((damping, y_s, uv_s, total));
        }
    }
    let (mut damping, y_s, uv_s, mut total) = best_total?;
    if y_s.pri == 0 && y_s.sec == 0 && uv_s.pri == 0 && uv_s.sec == 0 {
        return None;
    }
    // Damping refinement on the winning strengths.
    for d in [4u8, 6] {
        let p = params_for(d, y_s, uv_s);
        let t = run(&p, 0) + if num_planes > 1 { run(&p, 1) } else { 0 };
        if t < total {
            total = t;
            damping = d;
        }
    }
    if total >= base_y + base_uv {
        return None;
    }

    // Apply the winner to the reconstruction (the §7.20 reference
    // store the decoder will hold after decoding this frame). One
    // §7.15 run filters every plane; `which` only selects the SSD
    // readout.
    let winner = params_for(damping, y_s, uv_s);
    let _ = run(&winner, 0);
    for (dst, src) in recon_y.iter_mut().zip(dst_owned[0].iter()) {
        *dst = (*src).max(0) as u16;
    }
    if num_planes > 1 {
        for (dst, src) in recon_u.iter_mut().zip(dst_owned[1].iter()) {
            *dst = (*src).max(0) as u16;
        }
        for (dst, src) in recon_v.iter_mut().zip(dst_owned[2].iter()) {
            *dst = (*src).max(0) as u16;
        }
    }
    Some(winner)
}
