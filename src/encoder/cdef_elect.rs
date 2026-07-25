//! r428/r429 — encoder-side §5.9.19 / §7.15 CDEF election (encoder
//! ladder item 3: r428 frame-level arm, r429 per-64×64-unit arm).
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
//! ## The two arms
//!
//! * **Frame-level** (`cdef_bits = 0`, r428): one strength set for
//!   the whole frame — the §5.11.56 `cdef_idx` literal is `L(0)`,
//!   ZERO tile bits, so the arm is pure distortion.
//! * **Per-unit** (`cdef_bits ∈ 1..=3`, r429): up to `1 << cdef_bits`
//!   §5.9.19 strength sets in the header, each 64×64 unit electing
//!   its id through the `L(cdef_bits)` literal of §5.11.56. §7.15
//!   reads only PRE-CDEF samples (`CurrFrame` in, `CdefFrame` out),
//!   so a unit's filtered output depends on its own id alone and the
//!   election decomposes exactly into per-unit SSD tables. Rate is
//!   exact by construction: an `L(n)` literal through the §8.2.6
//!   bool coder costs exactly `n` bits (equiprobable halving), and
//!   the §5.9.19 header grows by exactly 6 (+6 with chroma) bits per
//!   extra strength set — both priced against λ on the same
//!   1/256-bit scale the twin-priced ladders use. The caller
//!   RE-EMITS the tile with the elected per-leaf ids (the literal
//!   perturbs the arithmetic coder state, so the whole tile is
//!   rewritten from the committed trees — the established
//!   exact-replay machinery).
//!
//! The search strategy (coarse-then-refine primary sweep, secondary
//! set, damping sweep, greedy set-list growth) is free encoder
//! engineering; every candidate is evaluated through the real §7.15
//! kernels.

use crate::cdf::PartitionWalker;
use crate::encoder::yuv_frame::YuvFrame;
use crate::loop_filter::PlaneBuffer;
use crate::uncompressed_header_tail::CdefParams;

/// One plane set's strength candidate.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Strength {
    pri: u8,
    sec: u8,
}

const ZERO: Strength = Strength { pri: 0, sec: 0 };

/// The elected CDEF configuration: header params + the per-64×64-unit
/// §5.11.56 strength ids (raster order over `ceil(MiCols / 16)`
/// units per row; `-1` = the unit codes no idx — every block in it
/// is skip — and the decoder's copy path applies). `d` is the plan's
/// EXACT whole-frame SSD against the source under the real §7.15
/// kernels (the per-unit decomposition is exact — §7.15 reads only
/// pre-CDEF samples).
pub(crate) struct CdefPlan {
    pub params: CdefParams,
    pub unit_idx: Vec<i8>,
    pub d: u64,
}

/// [`elect_cdef`]'s result: the estimate-best plan plus what the
/// caller needs for the FINAL exact-realized-bytes election when
/// `best` is a per-unit plan (the plan-stage rate model prices the
/// `L(cdef_bits)` literals and the §5.9.19 header growth exactly in
/// bits, but the emitted tile and header are byte-aligned — the
/// caller re-emits and settles per-unit vs frame-level vs unfiltered
/// on real byte counts, the same doctrine as the hp / temporal-seg /
/// primary-ref elections).
pub(crate) struct CdefElection {
    /// The `D + λ·R` winner on the plan-stage (exact-bits) scale.
    pub best: CdefPlan,
    /// The frame-level alternative (`cdef_bits = 0`), present iff it
    /// beats the unfiltered frame — the caller's fallback arm when
    /// `best` is per-unit and loses the exact-bytes settlement.
    pub frame_level: Option<CdefPlan>,
    /// The unfiltered frame's SSD (the no-election arm).
    pub base_d: u64,
}

/// Election inputs (see [`elect_cdef`]).
pub(crate) struct CdefElectInput<'a> {
    pub mirror: &'a PartitionWalker,
    pub input: &'a YuvFrame,
    pub recon_y: &'a [u16],
    pub recon_u: &'a [u16],
    pub recon_v: &'a [u16],
    pub width: usize,
    pub height: usize,
    pub chroma_w: usize,
    pub chroma_h: usize,
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub num_planes: u8,
    /// λ on the [`super::rate_twin::score256`] 1/256-bit convention
    /// (the frame quantiser's [`super::key_frame::lambda_for`]).
    pub lambda: u64,
    /// Highest §5.9.19 `cdef_bits` the election may propose
    /// (`0` = frame-level only — the r428 shape; spec cap is 3).
    pub max_bits: u8,
}

/// Per-unit SSD tables for one plane set at one damping: `cands[0]`
/// is always the zero strength (per-unit SSD = the unfiltered base).
struct SetTables {
    cands: Vec<Strength>,
    /// `ssd[cand][unit]`.
    ssd: Vec<Vec<u64>>,
}

impl SetTables {
    fn total(&self, cand: usize) -> u64 {
        self.ssd[cand].iter().sum()
    }
    fn best_by_total(&self) -> usize {
        (0..self.cands.len())
            .min_by_key(|&i| self.total(i))
            .unwrap_or(0)
    }
}

/// The frame-level + per-unit CDEF election. `recon_*` are the
/// committed pre-CDEF reconstruction planes (the §7.14 deblock levels
/// this encoder codes are 0, so the reconstruction IS the CDEF
/// input). Returns the winning plan — NOT yet applied — or `None`
/// when nothing beat the unfiltered frame under `D + λ·R`. The
/// caller applies via [`apply_cdef_plan`] (and, for
/// `params.cdef_bits > 0`, first re-emits the tile with the plan's
/// per-leaf ids and settles the final arm on exact realized bytes —
/// see [`CdefElection`]).
pub(crate) fn elect_cdef(inp: &CdefElectInput<'_>) -> Option<CdefElection> {
    let mi_rows = inp.mirror.mi_rows();
    let mi_cols = inp.mirror.mi_cols();
    let sb_rows = mi_rows.div_ceil(16) as usize;
    let sb_cols = mi_cols.div_ceil(16) as usize;
    let n_units = sb_rows * sb_cols;

    // Which units carry a §5.11.56 idx on the wire: the committed
    // grid stamped the anchor (any non-skip block exists) — `-1`
    // anchors are all-skip and stay on the decoder's copy path.
    let coded: Vec<bool> = {
        let grid = inp.mirror.cdef_idx();
        (0..n_units)
            .map(|k| {
                let (ur, uc) = ((k / sb_cols) as u32, (k % sb_cols) as u32);
                grid[((ur * 16) * mi_cols + uc * 16) as usize] != -1
            })
            .collect()
    };
    let coded_count = coded.iter().filter(|&&c| c).count() as u64;
    if coded_count == 0 {
        return None;
    }

    let planes: Vec<(usize, usize)> = if inp.num_planes > 1 {
        vec![
            (inp.width, inp.height),
            (inp.chroma_w, inp.chroma_h),
            (inp.chroma_w, inp.chroma_h),
        ]
    } else {
        vec![(inp.width, inp.height)]
    };
    let mut src_owned: Vec<Vec<i32>> = Vec::with_capacity(planes.len());
    src_owned.push(inp.recon_y.iter().map(|&v| i32::from(v)).collect());
    if inp.num_planes > 1 {
        src_owned.push(inp.recon_u.iter().map(|&v| i32::from(v)).collect());
        src_owned.push(inp.recon_v.iter().map(|&v| i32::from(v)).collect());
    }
    let mut dst_owned: Vec<Vec<i32>> = src_owned.clone();

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

    // Per-unit SSD of one plane set after one §7.15 run under the
    // given params over the committed (all-id-0) grid. `which = 0`
    // reads luma, `1` chroma (U + V).
    let mut run_units = |params: &CdefParams, which: u8| -> Vec<u64> {
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
        inp.mirror.cdef_frame_from_idx(
            params,
            inp.num_planes,
            inp.bit_depth,
            inp.subsampling_x,
            inp.subsampling_y,
            &src,
            &mut dst,
        );
        let mut ssd = vec![0u64; n_units];
        if which == 0 {
            for (i, (a, b)) in dst_owned[0].iter().zip(inp.input.y.iter()).enumerate() {
                let (y, x) = (i / inp.width, i % inp.width);
                let d = i64::from(*a) - i64::from(*b);
                ssd[(y / 64) * sb_cols + x / 64] += (d * d) as u64;
            }
        } else {
            let (ssx, ssy) = (inp.subsampling_x as usize, inp.subsampling_y as usize);
            for (dst_plane, inp_plane) in dst_owned[1..3].iter().zip([&inp.input.u, &inp.input.v]) {
                for (i, (a, b)) in dst_plane.iter().zip(inp_plane.iter()).enumerate() {
                    let (cy, cx) = (i / inp.chroma_w, i % inp.chroma_w);
                    let d = i64::from(*a) - i64::from(*b);
                    ssd[((cy << ssy) / 64) * sb_cols + (cx << ssx) / 64] += (d * d) as u64;
                }
            }
        }
        ssd
    };

    // Unfiltered per-unit baselines.
    let base_y_units: Vec<u64> = {
        let mut ssd = vec![0u64; n_units];
        for (i, (a, b)) in inp.recon_y.iter().zip(inp.input.y.iter()).enumerate() {
            let (y, x) = (i / inp.width, i % inp.width);
            let d = i64::from(*a) - i64::from(*b);
            ssd[(y / 64) * sb_cols + x / 64] += (d * d) as u64;
        }
        ssd
    };
    let base_uv_units: Vec<u64> = if inp.num_planes > 1 {
        let mut ssd = vec![0u64; n_units];
        let (ssx, ssy) = (inp.subsampling_x as usize, inp.subsampling_y as usize);
        for (rec, src) in [(inp.recon_u, &inp.input.u), (inp.recon_v, &inp.input.v)] {
            for (i, (a, b)) in rec.iter().zip(src.iter()).enumerate() {
                let (cy, cx) = (i / inp.chroma_w, i % inp.chroma_w);
                let d = i64::from(*a) - i64::from(*b);
                ssd[((cy << ssy) / 64) * sb_cols + (cx << ssx) / 64] += (d * d) as u64;
            }
        }
        ssd
    } else {
        vec![0u64; n_units]
    };
    let base_total: u64 = base_y_units.iter().sum::<u64>() + base_uv_units.iter().sum::<u64>();

    // Coarse-then-refine candidate sweep for one plane set at one
    // damping, per-unit tables retained for the per-unit arm.
    let mut sweep = |damping: u8, which: u8, base_units: &[u64]| -> SetTables {
        let mut t = SetTables {
            cands: vec![ZERO],
            ssd: vec![base_units.to_vec()],
        };
        let mut eval = |t: &mut SetTables, s: Strength| {
            if t.cands.contains(&s) {
                return;
            }
            let p = if which == 0 {
                params_for(damping, s, ZERO)
            } else {
                params_for(damping, ZERO, s)
            };
            t.ssd.push(run_units(&p, which));
            t.cands.push(s);
        };
        for pri in [1u8, 2, 3, 4, 6, 9, 12, 15] {
            for sec in [0u8, 2] {
                eval(&mut t, Strength { pri, sec });
            }
        }
        // Secondary-only candidates (legal stored sec ∈ {1, 2, 4}).
        for sec in [1u8, 2, 4] {
            eval(&mut t, Strength { pri: 0, sec });
        }
        let center = t.cands[t.best_by_total()];
        for pri in center.pri.saturating_sub(1)..=(center.pri + 1).min(15) {
            for sec in [0u8, 1, 2, 4] {
                eval(&mut t, Strength { pri, sec });
            }
        }
        t
    };

    // Per-damping tables + the r428 frame-level winner.
    let dampings = [3u8, 5];
    let mut tables: Vec<(SetTables, SetTables)> = Vec::new();
    for &d in &dampings {
        let ty = sweep(d, 0, &base_y_units);
        let tuv = if inp.num_planes > 1 {
            sweep(d, 1, &base_uv_units)
        } else {
            SetTables {
                cands: vec![ZERO],
                ssd: vec![base_uv_units.clone()],
            }
        };
        tables.push((ty, tuv));
    }

    // Frame-level arm: best (y, uv) per damping by totals, then the
    // damping refinement {4, 6} on the winning strengths (full-frame
    // totals — the arm is pure distortion, R = 0).
    let (mut fl_damping, mut fl_y, mut fl_uv, mut fl_total) = (3u8, ZERO, ZERO, u64::MAX);
    for (di, &d) in dampings.iter().enumerate() {
        let (ty, tuv) = &tables[di];
        let (yi, uvi) = (ty.best_by_total(), tuv.best_by_total());
        let total = ty.total(yi) + tuv.total(uvi);
        if total < fl_total {
            (fl_damping, fl_y, fl_uv, fl_total) = (d, ty.cands[yi], tuv.cands[uvi], total);
        }
    }
    if fl_y != ZERO || fl_uv != ZERO {
        for d in [4u8, 6] {
            let p = params_for(d, fl_y, fl_uv);
            let ty = run_units(&p, 0);
            let t: u64 = ty.iter().sum::<u64>()
                + if inp.num_planes > 1 {
                    run_units(&p, 1).iter().sum::<u64>()
                } else {
                    0
                };
            if t < fl_total {
                fl_total = t;
                fl_damping = d;
            }
        }
    }

    // Candidate plans: (D, R256, damping, set list, per-unit set
    // choice). Set = (y index, uv index) into the damping's tables.
    struct Plan {
        d: u64,
        r256: u64,
        damping: u8,
        sets: Vec<(usize, usize)>,
        choice: Vec<usize>,
        bits: u8,
        table: usize,
    }
    let mut plans: Vec<Plan> = Vec::new();

    // The unfiltered baseline and the frame-level arm as plans.
    plans.push(Plan {
        d: base_total,
        r256: 0,
        damping: 3,
        sets: Vec::new(),
        choice: Vec::new(),
        bits: 0,
        table: usize::MAX,
    });
    if fl_total < u64::MAX && (fl_y != ZERO || fl_uv != ZERO) {
        plans.push(Plan {
            d: fl_total,
            r256: 0,
            damping: fl_damping,
            sets: Vec::new(),
            choice: Vec::new(),
            bits: 0,
            table: usize::MAX - 1,
        });
    }

    // Per-unit arms: greedy set-list growth over the (y, uv) product
    // space at each swept damping; assignment = per-unit argmin;
    // exact rate = `cdef_bits` per coded unit + 6 (+6 chroma) header
    // bits per extra set.
    let set_hdr_bits: u64 = if inp.num_planes > 1 { 12 } else { 6 };
    for (di, &d) in dampings.iter().enumerate() {
        if inp.max_bits == 0 {
            break;
        }
        let (ty, tuv) = &tables[di];
        let unit_cost = |set: (usize, usize), k: usize| ty.ssd[set.0][k] + tuv.ssd[set.1][k];
        // Greedy seed: the best single set.
        let mut best_single = (0usize, 0usize);
        let mut best_single_d = u64::MAX;
        for yi in 0..ty.cands.len() {
            for uvi in 0..tuv.cands.len() {
                let t = ty.total(yi) + tuv.total(uvi);
                if t < best_single_d {
                    best_single_d = t;
                    best_single = (yi, uvi);
                }
            }
        }
        let mut sets = vec![best_single];
        let mut cur: Vec<u64> = (0..n_units).map(|k| unit_cost(best_single, k)).collect();
        for bits in 1..=inp.max_bits.min(3) {
            let want = 1usize << bits;
            while sets.len() < want {
                // The set whose addition reduces the assigned total most.
                let mut best_gain = 0u64;
                let mut best_set: Option<(usize, usize)> = None;
                for yi in 0..ty.cands.len() {
                    for uvi in 0..tuv.cands.len() {
                        if sets.contains(&(yi, uvi)) {
                            continue;
                        }
                        let gain: u64 = (0..n_units)
                            .filter(|&k| coded[k])
                            .map(|k| cur[k].saturating_sub(unit_cost((yi, uvi), k)))
                            .sum();
                        if gain > best_gain {
                            best_gain = gain;
                            best_set = Some((yi, uvi));
                        }
                    }
                }
                match best_set {
                    Some(s) => {
                        for (k, c) in cur.iter_mut().enumerate() {
                            if coded[k] {
                                *c = (*c).min(unit_cost(s, k));
                            }
                        }
                        sets.push(s);
                    }
                    // No remaining set helps: pad with the seed (the
                    // duplicate costs header bits and will lose to
                    // the smaller-bits plan on R).
                    None => sets.push(best_single),
                }
            }
            let choice: Vec<usize> = (0..n_units)
                .map(|k| {
                    if !coded[k] {
                        return 0;
                    }
                    (0..sets.len())
                        .min_by_key(|&s| unit_cost(sets[s], k))
                        .unwrap_or(0)
                })
                .collect();
            let dtot: u64 = (0..n_units)
                .map(|k| {
                    if coded[k] {
                        unit_cost(sets[choice[k]], k)
                    } else {
                        base_y_units[k] + base_uv_units[k]
                    }
                })
                .sum();
            let r256 = 256 * (u64::from(bits) * coded_count + ((1u64 << bits) - 1) * set_hdr_bits);
            plans.push(Plan {
                d: dtot,
                r256,
                damping: d,
                sets: sets.clone(),
                choice,
                bits,
                table: di,
            });
        }
    }

    // `D·256 + λ·R256` — the twin ladders' exact scale.
    if std::env::var_os("OXIDEAV_AV1_CDEF_DEBUG").is_some() {
        eprintln!(
            "cdef-elect: units {n_units} ({coded_count} coded), lambda {}, base D {base_total}",
            inp.lambda
        );
        for p in &plans {
            let tag = match p.table {
                usize::MAX => "base".to_string(),
                t if t == usize::MAX - 1 => {
                    format!(
                        "frame-level d{fl_damping} y({},{}) uv({},{})",
                        fl_y.pri, fl_y.sec, fl_uv.pri, fl_uv.sec
                    )
                }
                t => {
                    let (ty, tuv) = &tables[t];
                    let sets: Vec<String> = p
                        .sets
                        .iter()
                        .map(|&(yi, uvi)| {
                            format!(
                                "y({},{})uv({},{})",
                                ty.cands[yi].pri,
                                ty.cands[yi].sec,
                                tuv.cands[uvi].pri,
                                tuv.cands[uvi].sec
                            )
                        })
                        .collect();
                    format!("bits{} d{} [{}]", p.bits, p.damping, sets.join(" "))
                }
            };
            eprintln!(
                "cdef-elect:   D {} R256 {} score {} <- {tag}",
                p.d,
                p.r256,
                crate::encoder::rate_twin::score256(p.d, inp.lambda, p.r256)
            );
        }
    }
    let winner = plans
        .into_iter()
        .min_by_key(|p| crate::encoder::rate_twin::score256(p.d, inp.lambda, p.r256))?;
    if winner.table == usize::MAX {
        return None; // the unfiltered baseline won
    }
    // The frame-level arm as a plan (present iff it beats the
    // unfiltered frame — the r428 election condition).
    let frame_level: Option<CdefPlan> = if (fl_y != ZERO || fl_uv != ZERO) && fl_total < base_total
    {
        Some(CdefPlan {
            params: params_for(fl_damping, fl_y, fl_uv),
            unit_idx: coded.iter().map(|&c| if c { 0 } else { -1 }).collect(),
            d: fl_total,
        })
    } else {
        None
    };
    if winner.table == usize::MAX - 1 {
        // Frame-level arm won outright — no exact-bytes settlement
        // needed (zero tile bits, header size equals the default's).
        return frame_level.map(|best| CdefElection {
            best,
            frame_level: None,
            base_d: base_total,
        });
    }
    // Per-unit arm.
    let (ty, tuv) = &tables[winner.table];
    let mut params = CdefParams::short_circuit();
    params.short_circuited = false;
    params.cdef_damping = winner.damping;
    params.cdef_bits = winner.bits;
    for (i, &(yi, uvi)) in winner.sets.iter().enumerate() {
        params.cdef_y_pri_strength[i] = ty.cands[yi].pri;
        params.cdef_y_sec_strength[i] = ty.cands[yi].sec;
        params.cdef_uv_pri_strength[i] = tuv.cands[uvi].pri;
        params.cdef_uv_sec_strength[i] = tuv.cands[uvi].sec;
    }
    let unit_idx: Vec<i8> = (0..n_units)
        .map(|k| if coded[k] { winner.choice[k] as i8 } else { -1 })
        .collect();
    Some(CdefElection {
        best: CdefPlan {
            params,
            unit_idx,
            d: winner.d,
        },
        frame_level,
        base_d: base_total,
    })
}

/// Apply an elected plan to the reconstruction (the §7.20 reference
/// store the decoder will hold after decoding this frame): one §7.15
/// run through the decoder's own driver over the plan's per-unit
/// grid. The caller must have re-emitted the tile first when
/// `params.cdef_bits > 0` (the write mirror's grid then equals
/// `plan.unit_idx` — asserted by the callers).
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_cdef_plan(
    mirror: &PartitionWalker,
    plan: &CdefPlan,
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
) {
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
    {
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
        mirror.cdef_frame_with_unit_grid(
            &plan.params,
            &plan.unit_idx,
            num_planes,
            bit_depth,
            subsampling_x,
            subsampling_y,
            &src,
            &mut dst,
        );
    }
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
}
