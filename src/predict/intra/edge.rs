//! Intra edge filter + upsample — spec §7.11.2.9, §7.11.2.10,
//! §7.11.2.11, §7.11.2.12.
//!
//! Before a directional intra predictor (§7.11.2.5) projects across
//! its neighbour arrays, the bitstream's `enable_intra_edge_filter`
//! flag may cause the reference edges `AboveRow` and `LeftCol` to be
//! pre-processed in two ways:
//!
//! * **Intra edge filter (§7.11.2.12)** — a 5-tap low-pass convolution
//!   whose strength (0..3) is derived from the block dimension and
//!   how far the predicted angle deviates from 90° / 180° (§7.11.2.9).
//!   Strength 0 is "do nothing".
//! * **Intra edge upsample (§7.11.2.11)** — a 4-tap polyphase that
//!   doubles the density of edge samples, giving the directional
//!   sub-pel interpolator a finer grid. Whether upsampling applies is
//!   chosen by §7.11.2.10 based on block size and angle difference.
//!
//! Both routines operate in place on the gathered edge slices so the
//! caller can run them immediately after
//! [`super::super::super::decode::superblock::gather_neighbors_u8`]
//! populates the reference samples.
//!
//! For the upsample routine the caller must reserve a small amount of
//! headroom in the slice: entries `-2..=2*numPx-2` must be addressable
//! (§7.11.2.11 bullet "When the process completes"). We surface this
//! by requiring the caller pass a slice *starting at index -2* — that
//! is, `buf[0]` maps to spec index `-2`, `buf[2]` maps to `-1`, etc.
//! See [`edge_upsample`] for the exact convention.

/// Output of the filter-strength selector (§7.11.2.9). Range 0..=3.
pub type EdgeFilterStrength = u8;

/// Filter taps from `Intra_Edge_Kernel` (§7.11.2.12). Indexed by
/// `strength - 1` (strength 0 is a no-op so it's not in the table).
/// Each row is 5 taps applied with a Round2(·, 4) rounding.
pub const INTRA_EDGE_KERNEL: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];

/// Number of taps in the intra edge filter kernel — `INTRA_EDGE_TAPS`
/// from §7.11.2.12.
pub const INTRA_EDGE_TAPS: usize = 5;

/// Implements the `intra edge filter strength selection process`
/// from §7.11.2.9. Returns a strength value in `0..=3`.
///
/// * `w`/`h` — transform dimensions in samples.
/// * `filter_type` — 0 or 1, the `filterType` output of the
///   §7.11.2.8 smooth-neighbour test.
/// * `delta` — `p_angle - 90` (above edge) or `p_angle - 180` (left
///   edge), in degrees. Use the raw signed difference; the routine
///   takes `abs(delta)` internally.
// The spec (§7.11.2.9) has separate `blk_wh <= 12` and `blk_wh <= 16`
// branches that happen to assign the same value today. Keeping them as
// two branches mirrors the spec text so future errata only touch one
// arm — allow the identical-blocks lint here.
#[allow(clippy::if_same_then_else)]
pub fn edge_filter_strength(w: u32, h: u32, filter_type: u32, delta: i32) -> EdgeFilterStrength {
    // §7.11.2.9: "The variable d is set equal to Abs( delta )".
    let d = delta.unsigned_abs();
    // blkWh = w + h per spec.
    let blk_wh = w + h;
    let mut strength: u8 = 0;
    if filter_type == 0 {
        if blk_wh <= 8 {
            if d >= 56 {
                strength = 1;
            }
        } else if blk_wh <= 12 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 16 {
            if d >= 40 {
                strength = 1;
            }
        } else if blk_wh <= 24 {
            if d >= 8 {
                strength = 1;
            }
            if d >= 16 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else if blk_wh <= 32 {
            strength = 1;
            if d >= 4 {
                strength = 2;
            }
            if d >= 32 {
                strength = 3;
            }
        } else {
            strength = 3;
        }
    } else if blk_wh <= 8 {
        if d >= 40 {
            strength = 1;
        }
        if d >= 64 {
            strength = 2;
        }
    } else if blk_wh <= 16 {
        if d >= 20 {
            strength = 1;
        }
        if d >= 48 {
            strength = 2;
        }
    } else if blk_wh <= 24 {
        if d >= 4 {
            strength = 3;
        }
    } else {
        strength = 3;
    }
    strength
}

/// Implements the `intra edge upsample selection process` from
/// §7.11.2.10. Returns `true` if the caller should upsample the edge.
pub fn edge_use_upsample(w: u32, h: u32, filter_type: u32, delta: i32) -> bool {
    let d = delta.unsigned_abs();
    let blk_wh = w + h;
    if d == 0 || d >= 40 {
        false
    } else if filter_type == 0 {
        blk_wh <= 16
    } else {
        blk_wh <= 8
    }
}

/// 5-tap low-pass filter applied along a reference edge in place —
/// spec §7.11.2.12.
///
/// * `edge` — the edge samples. In the spec these are indexed
///   `-1..sz-1`; here the input array is `edge[0..sz]` where
///   `edge[0]` maps to spec index `-1` (the corner sample). The first
///   element is not overwritten, matching the spec's "for i = 1..sz-1"
///   loop plus its `edge[k]` derivation.
/// * `strength` — 0..=3 from [`edge_filter_strength`]. 0 is a no-op.
///
/// `edge.len()` must equal `sz`. The spec caps `sz <= 129` (§7.11.2.12
/// opening line).
// We write to `edge[i]` while reading the unmodified `snapshot`. Using
// `enumerate` on a mutable iterator doesn't help — the spec explicitly
// indexes by `i` to build `k = Clip3(0, sz - 1, i - 2 + j)`.
#[allow(clippy::needless_range_loop)]
pub fn edge_filter(edge: &mut [u8], strength: EdgeFilterStrength) {
    if strength == 0 {
        return;
    }
    let sz = edge.len();
    if sz < 2 {
        return;
    }
    // Snapshot unfiltered values so later iterations read the original
    // spec `edge[]` array, not the in-place modifications. This mirrors
    // §7.11.2.12 step "The array edge is derived by setting edge[ i ]
    // equal to ( left ? LeftCol[ i - 1 ] : AboveRow[ i - 1 ] )".
    let snapshot: Vec<i32> = edge.iter().map(|&v| v as i32).collect();
    let kernel = &INTRA_EDGE_KERNEL[(strength as usize) - 1];
    // Per §7.11.2.12: "for i = 1..sz-1". The spec stores the output
    // into `LeftCol[i-1]` / `AboveRow[i-1]`, which in our layout is
    // simply `edge[i]` because our `edge[0]` is the corner.
    for i in 1..sz {
        let mut s: i32 = 0;
        for (j, &tap) in kernel.iter().enumerate() {
            // k = Clip3(0, sz - 1, i - 2 + j)
            let k = (i as i32 - 2 + j as i32).clamp(0, sz as i32 - 1) as usize;
            s += tap * snapshot[k];
        }
        // Round2(s, 4) + Clip1(·). The inputs are u8 so the low-pass
        // result is in [0, 255] already, but we clamp defensively.
        let v = ((s + 8) >> 4).clamp(0, 255) as u8;
        edge[i] = v;
    }
}

/// 10-bit / 12-bit HBD variant of [`edge_filter`]. `bit_depth` is the
/// plane bit depth and is used for the final Clip1 step of §7.11.2.12.
#[allow(clippy::needless_range_loop)]
pub fn edge_filter16(edge: &mut [u16], strength: EdgeFilterStrength, bit_depth: u32) {
    if strength == 0 {
        return;
    }
    let sz = edge.len();
    if sz < 2 {
        return;
    }
    let max_v = (1i32 << bit_depth) - 1;
    let snapshot: Vec<i32> = edge.iter().map(|&v| v as i32).collect();
    let kernel = &INTRA_EDGE_KERNEL[(strength as usize) - 1];
    for i in 1..sz {
        let mut s: i32 = 0;
        for (j, &tap) in kernel.iter().enumerate() {
            let k = (i as i32 - 2 + j as i32).clamp(0, sz as i32 - 1) as usize;
            s += tap * snapshot[k];
        }
        edge[i] = ((s + 8) >> 4).clamp(0, max_v) as u16;
    }
}

/// 4-tap polyphase upsampler that doubles the edge density — spec
/// §7.11.2.11.
///
/// The spec indexes `buf[-1..numPx]` on input and `buf[-2..2*numPx-2]`
/// on output. To stay in safe Rust we adopt the convention that the
/// caller passes `buf` with an offset of `+2` from the spec's index:
/// `buf[0]` is spec index `-2`, `buf[2]` is spec index `0`.
///
/// The caller must therefore supply a slice of length
/// `2 * num_px + 2`, with the valid unfiltered input placed at
/// `buf[1..=num_px + 1]` (that is, spec indices `-1..=numPx - 1`).
/// After the call, `buf[0..2 * num_px + 2]` holds the upsampled
/// stream — `buf[0]` = spec index `-2`, `buf[2*num_px]` = spec index
/// `2*numPx - 2`.
pub fn edge_upsample(buf: &mut [u8], num_px: usize) {
    if num_px == 0 {
        return;
    }
    assert!(
        buf.len() >= 2 * num_px + 2,
        "edge_upsample: buf too small (need {}, got {})",
        2 * num_px + 2,
        buf.len()
    );
    // Build `dup[0..numPx + 3]` by extending `buf` as per spec:
    //   dup[ 0 ]          = buf[-1]        (== input buf[1])
    //   dup[ i + 2 ]      = buf[i]         for i in -1..numPx
    //   dup[ numPx + 2 ]  = buf[numPx - 1] (== input buf[numPx + 1])
    // Our slice is shifted by +2, so buf[-1] maps to buf[1],
    // buf[0] maps to buf[2], ..., buf[numPx - 1] maps to buf[numPx + 1].
    let mut dup = vec![0i32; num_px + 3];
    dup[0] = buf[1] as i32;
    // for i = -1..numPx { dup[i+2] = buf[i] }  →  dup[1..=numPx+1] = buf[1..=numPx+1]
    for i in 0..=num_px {
        dup[i + 1] = buf[i + 1] as i32;
    }
    dup[num_px + 2] = buf[num_px + 1] as i32;

    // buf[-2] = dup[0]  →  spec buf[-2] is our buf[0].
    buf[0] = dup[0].clamp(0, 255) as u8;
    for i in 0..num_px {
        // s = -dup[i] + 9*dup[i+1] + 9*dup[i+2] - dup[i+3]
        // s = Clip1( Round2(s, 4) )
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        let s = ((s + 8) >> 4).clamp(0, 255) as u8;
        // spec "buf[2*i - 1] = s"  → our index (2*i - 1) + 2 = 2*i + 1.
        buf[2 * i + 1] = s;
        // spec "buf[2*i]      = dup[i+2]"  → our index 2*i + 2.
        buf[2 * i + 2] = dup[i + 2].clamp(0, 255) as u8;
    }
}

/// HBD variant of [`edge_upsample`].
pub fn edge_upsample16(buf: &mut [u16], num_px: usize, bit_depth: u32) {
    if num_px == 0 {
        return;
    }
    assert!(
        buf.len() >= 2 * num_px + 2,
        "edge_upsample16: buf too small (need {}, got {})",
        2 * num_px + 2,
        buf.len()
    );
    let max_v = (1i32 << bit_depth) - 1;
    let mut dup = vec![0i32; num_px + 3];
    dup[0] = buf[1] as i32;
    for i in 0..=num_px {
        dup[i + 1] = buf[i + 1] as i32;
    }
    dup[num_px + 2] = buf[num_px + 1] as i32;

    buf[0] = dup[0].clamp(0, max_v) as u16;
    for i in 0..num_px {
        let s = -dup[i] + 9 * dup[i + 1] + 9 * dup[i + 2] - dup[i + 3];
        let s = ((s + 8) >> 4).clamp(0, max_v) as u16;
        buf[2 * i + 1] = s;
        buf[2 * i + 2] = dup[i + 2].clamp(0, max_v) as u16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // §7.11.2.9 — spot-check every size bucket for filter_type == 0.
    #[test]
    fn strength_filter_type_0_buckets() {
        // blkWh <= 8: only d >= 56 triggers s=1.
        assert_eq!(edge_filter_strength(4, 4, 0, 0), 0);
        assert_eq!(edge_filter_strength(4, 4, 0, 55), 0);
        assert_eq!(edge_filter_strength(4, 4, 0, 56), 1);
        assert_eq!(edge_filter_strength(4, 4, 0, -80), 1);
        // blkWh <= 12: d >= 40 → 1.
        assert_eq!(edge_filter_strength(8, 4, 0, 39), 0);
        assert_eq!(edge_filter_strength(8, 4, 0, 40), 1);
        // blkWh <= 16: same 40-threshold.
        assert_eq!(edge_filter_strength(8, 8, 0, 40), 1);
        // blkWh <= 24: three thresholds 8/16/32.
        assert_eq!(edge_filter_strength(16, 8, 0, 7), 0);
        assert_eq!(edge_filter_strength(16, 8, 0, 8), 1);
        assert_eq!(edge_filter_strength(16, 8, 0, 16), 2);
        assert_eq!(edge_filter_strength(16, 8, 0, 32), 3);
        // blkWh <= 32: base 1, d>=4 → 2, d>=32 → 3.
        assert_eq!(edge_filter_strength(16, 16, 0, 0), 1);
        assert_eq!(edge_filter_strength(16, 16, 0, 4), 2);
        assert_eq!(edge_filter_strength(16, 16, 0, 32), 3);
        // blkWh > 32: always 3.
        assert_eq!(edge_filter_strength(32, 32, 0, 0), 3);
        assert_eq!(edge_filter_strength(64, 64, 0, 0), 3);
    }

    // §7.11.2.9 — filter_type == 1 paths.
    #[test]
    fn strength_filter_type_1_buckets() {
        // blkWh <= 8.
        assert_eq!(edge_filter_strength(4, 4, 1, 39), 0);
        assert_eq!(edge_filter_strength(4, 4, 1, 40), 1);
        assert_eq!(edge_filter_strength(4, 4, 1, 64), 2);
        // blkWh <= 16.
        assert_eq!(edge_filter_strength(8, 8, 1, 19), 0);
        assert_eq!(edge_filter_strength(8, 8, 1, 20), 1);
        assert_eq!(edge_filter_strength(8, 8, 1, 48), 2);
        // blkWh <= 24.
        assert_eq!(edge_filter_strength(16, 8, 1, 3), 0);
        assert_eq!(edge_filter_strength(16, 8, 1, 4), 3);
        // blkWh > 24 — always 3.
        assert_eq!(edge_filter_strength(32, 32, 1, 0), 3);
    }

    // §7.11.2.10 — upsample gate.
    #[test]
    fn upsample_gate_spec() {
        // d == 0 or d >= 40 → never upsample.
        assert!(!edge_use_upsample(4, 4, 0, 0));
        assert!(!edge_use_upsample(4, 4, 0, 40));
        assert!(!edge_use_upsample(4, 4, 0, 50));
        // filterType == 0: upsample iff blkWh <= 16.
        assert!(edge_use_upsample(8, 8, 0, 8));
        assert!(!edge_use_upsample(16, 16, 0, 8));
        // filterType == 1: upsample iff blkWh <= 8.
        assert!(edge_use_upsample(4, 4, 1, 8));
        assert!(!edge_use_upsample(8, 8, 1, 8));
        // Negative deltas behave like positive (abs inside).
        assert!(edge_use_upsample(4, 4, 0, -8));
        assert!(!edge_use_upsample(4, 4, 0, -40));
    }

    // §7.11.2.12 — strength 0 is a no-op.
    #[test]
    fn edge_filter_strength0_is_noop() {
        let mut edge = [10u8, 20, 30, 40, 50];
        let before = edge;
        edge_filter(&mut edge, 0);
        assert_eq!(edge, before);
    }

    // §7.11.2.12 — constant input must remain constant (kernel rows sum to 16).
    #[test]
    fn edge_filter_constant_preserved_all_strengths() {
        for s in 1..=3 {
            let mut edge = [100u8; 10];
            edge_filter(&mut edge, s);
            for (i, &v) in edge.iter().enumerate() {
                assert_eq!(v, 100, "strength {s} edge[{i}] drifted to {v}");
            }
            // Kernel sanity.
            let sum: i32 = INTRA_EDGE_KERNEL[(s as usize) - 1].iter().sum();
            assert_eq!(sum, 16, "kernel {s} doesn't sum to 16");
        }
    }

    // §7.11.2.12 — first element is never modified (the spec loop runs i=1..sz-1).
    #[test]
    fn edge_filter_corner_unchanged() {
        let mut edge = [1u8, 50, 100, 150, 200];
        edge_filter(&mut edge, 2);
        assert_eq!(edge[0], 1);
    }

    // §7.11.2.12 — hand-compute for a small case to lock tap ordering.
    //
    // edge = [0, 0, 255, 0, 0], strength 1 → kernel [0,4,8,4,0], Round2(·,4).
    // For i = 2: k at j = 0..4 = Clip3(0..4, i-2+j) = 0..4.
    //   s = 0*edge[0] + 4*edge[1] + 8*edge[2] + 4*edge[3] + 0*edge[4]
    //     = 0 + 0 + 8*255 + 0 + 0 = 2040
    //   edge[2] = (2040 + 8) >> 4 = 2048 >> 4 = 128.
    #[test]
    fn edge_filter_impulse_response_strength1() {
        let mut edge = [0u8, 0, 255, 0, 0];
        edge_filter(&mut edge, 1);
        // Indexes 1, 3: s = 4*255 = 1020 → (1020+8)>>4 = 64.
        assert_eq!(edge[1], 64);
        assert_eq!(edge[2], 128);
        assert_eq!(edge[3], 64);
        // Index 4: k = Clip3(0, 4, 2..6) = {2, 3, 4, 4, 4}; taps 4, 8, 4, 0, 0.
        // dup[2]=edge[2]=255, edge[3]=0, edge[4]=0 → s = 4*0 + 8*0 + 4*255 + 0 + 0
        //   ... wait, clamp positions are for read, not tap. Let me recompute
        //   more carefully: j = 0..4, k = Clip3(0,4,4-2+j) = Clip3(0,4,{2,3,4,5,6}) = {2,3,4,4,4}
        //   s = 0*edge[2] + 4*edge[3] + 8*edge[4] + 4*edge[4] + 0*edge[4]
        //     = 0 + 0 + 0 + 0 + 0 = 0
        assert_eq!(edge[4], 0);
    }

    // §7.11.2.11 — upsample correctness for a flat constant edge.
    // All samples identical → polyphase tap sum is 16*v, Round2 returns v.
    #[test]
    fn edge_upsample_constant_is_constant() {
        let num_px = 4;
        // Slice layout: [spec -2, spec -1, spec 0, spec 1, spec 2, spec 3,
        //                 spec 4, spec 5, spec 6]
        // Input valid range is spec [-1, numPx-1] = [-1, 3] → our [1..=5].
        let mut buf = vec![77u8; 2 * num_px + 2];
        edge_upsample(&mut buf, num_px);
        for (i, &v) in buf.iter().enumerate() {
            assert_eq!(v, 77, "buf[{i}] = {v}, expected 77");
        }
    }

    // §7.11.2.11 — upsample preserves original samples at even output positions.
    // Per the spec: `buf[2*i] = dup[i+2]` → the mid-point of each original
    // input sample lands unchanged at the corresponding even slot.
    #[test]
    fn edge_upsample_original_samples_preserved() {
        let num_px = 4;
        // Build a varied input at our shifted indices 1..=5 (spec -1..=3).
        let mut buf = vec![0u8; 2 * num_px + 2];
        let src = [10u8, 50, 100, 150, 200, 230]; // spec indices -1..=4
        buf[1..1 + src.len()].copy_from_slice(&src);
        edge_upsample(&mut buf, num_px);
        // Per spec, `buf[2*i] = dup[i+2]` for i=0..numPx, and
        // `dup[i+2] = buf[i]` (spec indexing) for i = -1..numPx.
        // With our +2 shift:
        //   spec buf[2*i]  →  our buf[2*i + 2]
        //   dup[i+2]       = original input buf[i] (spec) = our input[i + 1]
        // So for i=0..numPx=4: our buf[2, 4, 6, 8] ==  input at spec [0, 1, 2, 3]
        // = our input[2, 3, 4, 5] = 50, 100, 150, 200.
        assert_eq!(buf[2], 50);
        assert_eq!(buf[4], 100);
        assert_eq!(buf[6], 150);
        assert_eq!(buf[8], 200);
        // Also: buf[0] = dup[0] = spec buf[-1] = our input[1] = 10.
        assert_eq!(buf[0], 10);
    }

    // HBD: filter should use the supplied bit depth's max in Clip1.
    #[test]
    fn edge_filter16_respects_bit_depth() {
        let mut edge = vec![0u16, 0, 1023, 0, 0];
        edge_filter16(&mut edge, 3, 10);
        // With kernel [2,4,4,4,2] the center gets:
        // s = 2*0 + 4*0 + 4*1023 + 4*0 + 2*0 = 4092
        // (4092 + 8) >> 4 = 4100 >> 4 = 256 → within [0, 1023] for 10-bit.
        assert_eq!(edge[2], 256);
    }

    // Upsample HBD constant test.
    #[test]
    fn edge_upsample16_constant_is_constant_10bit() {
        let num_px = 4;
        let mut buf = vec![1000u16; 2 * num_px + 2];
        edge_upsample16(&mut buf, num_px, 10);
        for (i, &v) in buf.iter().enumerate() {
            assert_eq!(v, 1000, "buf[{i}] = {v}, expected 1000");
        }
    }
}
