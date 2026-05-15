//! Workspace task #796 — investigation of the remaining §5.11.39
//! sign-bit divergence between `oxideav-av1` and `dav1d` on the 1×1
//! lossless YUV444 KEY frame in
//! `crates/oxideav-avif/tests/fixtures/fuzz/y_plane_divergence_match.avif`.
//!
//! ## Round 67 audit (2026-05-14) — three hypotheses FALSIFIED
//!
//! All three round-66 hypotheses were verified against the in-tree
//! AV1 spec corpus (`docs/video/av1/av1-spec.txt` §8.2.2-§8.2.6 +
//! §9.4) and ruled out:
//!
//! 1. **`SymbolDecoder::new` `sz` accounting** —
//!    `split_tile_payloads` emits a 14-byte slice for our
//!    `DIVERGENCE_OBU`; `SymbolDecoder::new` is called with
//!    `sz = 14`, giving `SymbolMaxBits = 8 * 14 - 15 = 97`. The spec
//!    formula at line 19441 matches our code at `src/symbol.rs:140`
//!    to the bit. The 15-bit init read of the 14 bytes yields
//!    `paddedBuf = 0x0CDC`, `SymbolValue = 0x7323 = 29475`,
//!    `SymbolRange = 0x8000 = 32768` — exactly what the rc-trace
//!    `init` line records. **No off-by-one in `sz`.**
//!
//! 2. **`update_cdf` rate arithmetic at `count == 0`** — the spec
//!    rate (`3 + (count > 15) + (count > 31) + min(log2(N), 2)`) at
//!    spec line 19814 is mathematically equivalent to our wire-form
//!    update at `src/symbol.rs:413-446`. A worked 4-symbol example
//!    (forward `[10000, 20000, 30000, 32768, 0]`, symbol=1, rate=4)
//!    produced bit-exact identical post-state through the spec's
//!    forward-form algorithm and our inverse-form algorithm:
//!
//!    - Spec forward: `[9375, 20798, 30173, 32768, 1]`
//!    - Our wire inverse: `[23393, 11970, 2595, 0, 1]`
//!    - `32768 - wire_i == forward_i` for `i = 0..N-1`. ✓
//!
//!    **No direction bug; no rate off-by-one at `count == 0`.**
//!
//! 3. **`coeff_br_multi` CDF drift at calls 22-25** — recomputing
//!    `coeff_br_ctx_spec(...)` independently from the partially-
//!    decoded `quants[]` array at each br call confirmed the trace's
//!    CDF indices:
//!    - Call 22 (scan_idx 2 br#1 at pos=4, row=1 col=0): neighbours
//!      at pos 5/8/9 give `mag = 0 + 6 + 1 = 7`, ctx = `((7+1)>>1).min(6) + 7 = 11`.
//!      Matches `DEFAULT_COEFF_BR_MULTI_CDF[0][0][0][11]` →
//!      `cdf[0] = 25700`. ✓
//!    - Call 23 (scan_idx 1 coeff_base at pos=1): falls through to
//!      `DEFAULT_COEFF_BASE_MULTI_CDF[0][0][0][2]` → `cdf[0] = 20172`.
//!      ✓
//!    - Call 24 (scan_idx 0 coeff_base at pos=0): DC special-case
//!      `ctx = 0`. Matches `DEFAULT_COEFF_BASE_MULTI_CDF[0][0][0][0]`
//!      → `cdf[0] = 28734`. ✓
//!    - Call 25 (scan_idx 0 br#1 at pos=0): `mag = 3`, ctx = 2 (DC
//!      special). Matches `DEFAULT_COEFF_BR_MULTI_CDF[0][0][0][2]` →
//!      `cdf[0] = 24056`. ✓
//!
//!    **All CDF lookups are at correct indices per spec.**
//!
//! ## Round 67 verdict
//!
//! The §8.2.6 entropy decoder is spec-compliant to the byte. The
//! ~10.9 k Q15 delta in the `value` register entering call 27 vs
//! dav1d's reference cannot be explained by any of the audited
//! hypotheses. The remaining gap is therefore at a layer **upstream
//! of §8.2.6**: either a CDF-default value typo (spot-checked but
//! not exhaustively diffed against the spec's 26 880-byte
//! `Default_*_Cdf` tables), or a context-derivation off-by-one that
//! produces the same chosen symbols on this fixture (so it's
//! invisible in level/sign reads) but selects a different CDF
//! entry. Closing this divergence requires dav1d's internal
//! entropy trace for direct call-by-call state comparison; an
//! offer to the docs collaborator to commission such a trace is
//! the round-68 plan.
//!
//! ## Round 68 black-box probe (2026-05-15)
//!
//! `dav1d 1.5.3` exposes only YUV decode + MD5 verify; there is no
//! public CLI flag for internal range-coder state. A single-bit-flip
//! sweep across all 110 bits of `tile_data[0..14]` (with
//! `--strict 0`) yielded 16 successful decodes; **none** lift the
//! `(U, V)` tuple from `(128, 128)` toward `(197, 215)`. Bit-positions
//! 24..50 keep `Y = 133` — these are the bits dav1d consumes during
//! luma-coefficient decode, where our entropy reads them identically.
//! Bit_pos=46 perturbs dav1d to `(133, 129, 128)` — pinned in
//! `tests/issue_796_dav1d_blackbox.rs::issue_796_dav1d_blackbox_bit_flip_46_perturbs_chroma`.
//!
//! The full round-68 audit (CDF-table dimensionality survey, dav1d
//! binary surface inventory, three remaining hypotheses for round 69)
//! is in
//! `docs/video/av1/specs/dav1d-range-coder-divergence-call-idx-27.md`.
//!
//! Round-68 net new finding: `DEFAULT_COEFF_BASE_EOB_MULTI_CDF`,
//! `DEFAULT_TXB_SKIP_CDF`, and `DEFAULT_DC_SIGN_CDF` are stored
//! WITHOUT the spec's `COEFF_CDF_Q_CTXS = 4` outer dimension. For
//! this fixture (`base_q_idx = 0` lossless, q_ctx=0) the stored
//! slice is the q_ctx=0 row, so the divergence fixture decodes
//! against spec-correct CDFs and the missing dim is NOT the
//! divergence cause. For `q_ctx > 0` the decoder will silently read
//! the wrong CDF entries — a separate latent bug filed for round 69.
//!
//! ## Investigation summary (round 49, 2026-05-12)
//!
//! After round 48 (`cfae193`) landed the §5.11.4 partition force-split
//! fix, the §5.11.39 coefficient entropy decoder reads luma level
//! magnitudes that match `dav1d 1.5.3` exactly:
//!
//! ```text
//! [4, 0, 0, 0, 3, 0, 0, 0, 6, 1, 0, 0, 1, 1, 0, 0]
//! ```
//!
//! However the 5 AC `sign_bit L(1)` reads (at scan indices 2, 3, 8, 9, 10
//! — positions 4, 8, 9, 12, 13 in row-major terms) diverge:
//!
//! | Scan idx | Row-maj pos | dav1d sign | oxideav sign |
//! |---------:|------------:|-----------:|-------------:|
//! |        0 |           0 |          + |            + |
//! |        2 |           4 |          + |          *-* |
//! |        3 |           8 |          + |            + |
//! |        8 |           9 |          + |          *-* |
//! |        9 |          12 |          + |          *-* |
//! |       10 |          13 |          + |          *-* |
//!
//! The DC sign (at scan_idx 0) is correctly decoded via the `dc_sign`
//! S() symbol with `dc_sign_cdf[plane=0][ctx=0]`. Of the 5 AC sign reads,
//! 4 come out negative (`sign_neg = true` via `decode_bool(16384) != 0`).
//! `dav1d` reads all 5 as positive.
//!
//! ## Forced-positive experiment
//!
//! When the AC sign reads are clamped to positive (i.e. all AC signs
//! treated as `+`), the WHT residual at (0, 0) flips from `2` to `5`
//! and the decoded Y sample becomes `133` — matching `dav1d` /
//! `avifdec --raw-color` exactly. Hand-trace of the §7.13.2.10 2D
//! inverse Walsh-Hadamard with row pass `shift = 2` + column pass
//! `shift = 0` on the all-positive dequantised buffer
//! `[16, 0, 0, 0, 12, 0, 0, 0, 24, 4, 0, 0, 4, 4, 0, 0]` (levels × 4
//! per the lossless DC8[0] = AC8[0] = 4 dequantiser, §7.12.2):
//!
//! Row pass:
//! - `[16, 0, 0, 0]` shift=2 → `[2, 2, 2, 2]`
//! - `[12, 0, 0, 0]` shift=2 → `[2, 1, 1, 1]`
//! - `[24, 4, 0, 0]` shift=2 → `[4, 3, 2, 2]`
//! - `[4, 4, 0, 0]`  shift=2 → `[1, 1, 0, 0]`
//!
//! Column 0 = `[2, 2, 4, 1]` shift=0 → `[5, -1, -2, 1]`.
//! Residual at (0, 0) = 5 → Y = predictor(128) + 5 = 133.
//!
//! ## Root cause status — UNRESOLVED
//!
//! Round 49 audited the §5.11.39 sign loop, the §9.4.7 dc_sign context
//! derivation, the `decode_bool(16384)` 50/50 bit read, the
//! `decode_symbol` 2-way CDF path, the `dc_sign_cdf` wire-format
//! conversion, the §8.2.6 renormalise step, the CDF adaptation rate,
//! the scan order (Default_Scan_4x4), the `compute_tx_type` lossless
//! return value, and the §5.11.47 `transform_type` qindex gating.
//! None of these surfaced a spec divergence.
//!
//! ## Round 66 side-channel (`rc-trace` feature)
//!
//! Round 66 (workspace task #801) landed the `rc-trace` cargo feature
//! on `oxideav-av1`: every `decode_bool` / `decode_symbol` /
//! `SymbolDecoder::new` call emits one JSONL line (`call_idx`,
//! `rng_in`, `value_in`, `p_or_cdf`, `result`, `rng_out`, `value_out`,
//! `bit_pos`) to the path in `OXIDEAV_AV1_RC_TRACE` (or stderr if
//! unset). The pinned `divergence.avif` trace is captured at
//! `tests/fixtures/issue_796_rc_trace.jsonl` for round-67 comparison
//! work.
//!
//! From that trace the divergent calls were narrowed to:
//!
//! * **`call_idx = 27`** — first AC `sign_bit L(1)` read. Our state
//!   entering this call: `range = 45796, value = 11884`; we read
//!   `bit = 1` (negative). For `dav1d` to read `bit = 0` (matching its
//!   all-positive AC-sign pattern) the entering `value` must be
//!   `≥ 22788`. Since both decoders agree on every prior **symbol**
//!   (the 16 luma coefficient magnitudes match exactly,
//!   `[4,0,0,0,3,0,0,0,6,1,0,0,1,1,0,0]`), the `value` divergence is
//!   accumulated through earlier `renormalise()` bit reads without
//!   ever crossing a CDF boundary.
//! * Calls **28-31** (AC signs 2-5) all run on `range = 45576` — the
//!   range never decreases because each `decode_bool(16384)` renorm
//!   exactly doubles + reads one bit, returning the post-renorm range
//!   to 45576. Sign results: `1,0,1,1,1` (we read 4 negatives);
//!   `dav1d` reads `0,0,0,0,0`.
//!
//! ## Suspected bug class — bit-stream offset on a renormalisation
//!
//! Because every symbol we picked matches `dav1d`'s symbol (so the CDF
//! lookup branches are equivalent at the symbol-selection layer), but
//! the `value` register coming into call 27 differs from `dav1d`'s,
//! the leading hypothesis is that one of the earlier renormalisations
//! (most likely between calls 22 and 26 — the post-coeff-base /
//! coeff_br / dc_sign reads — see fixture trace for exact bit-pos
//! deltas) consumes a different number of bits than dav1d, OR reads
//! the same number of bits but at a different offset. The §8.2.6
//! renormalise body is byte-for-byte spec-correct (audited in
//! `symbol::renormalise`), so the more likely culprits are upstream:
//!
//! 1. **`max_bits` accounting** — our `SymbolDecoder::new` sets
//!    `max_bits = sz * 8 - 15`. If `sz` is sub-optimal (e.g. uses the
//!    OBU `payload.len()` rather than the tile-group `tile_size`
//!    field), `max_bits` can go negative one renorm earlier than
//!    dav1d, switching subsequent renorms into "pad with zeros" mode.
//!    Worth grepping for `SymbolDecoder::new` callers in
//!    `decode/tile.rs` + verifying `sz == tile_data.len()` matches
//!    what `split_tile_payloads` produced.
//! 2. **CDF adaptation rate off-by-one for `count == 0`** — every
//!    pre-call CDF on this fixture has count = 0 (this is a KEY
//!    frame's first TU). Round 4 fixed the direction bug but the
//!    `rate = 3 + 0 + 0 + min(log2(N), 2)` arithmetic at count = 0
//!    has not been spec-cross-checked against an independent
//!    `decode_subexp` golden vector. A 1-bit-off rate produces ~50%
//!    of the adapted-CDF entries off by ±1, which is below the noise
//!    floor on multi-CDF symbols but would shift the `cur` threshold
//!    of subsequent 2-symbol CDFs (e.g. `dc_sign_cdf` at call 26)
//!    enough to cross the renormalisation boundary — and that's
//!    exactly the location of the observed `value`-register delta.
//! 3. **`coeff_br_multi` saturation behaviour** — calls 22-25 are
//!    `read_br_level` reads against `[24056, 17717, 13265, 0, 0]`. The
//!    spec §9.4 update for a 4-symbol CDF with rate `3 + 0 + 0 + 2 =
//!    5` nudges the four entries by `(32768 - v) >> 5` / `v >> 5`. If
//!    the adapted CDF at call 23 onward differs from dav1d's by a
//!    handful of Q15 units, the `cur` computation downstream pulls
//!    `value` along by tens of units per renorm — accumulating to the
//!    observed ~11k delta after a dozen renormalisations.
//!
//! Empirical findings:
//! 1. Forcing all AC signs positive yields the dav1d-matching
//!    `(Y, U, V) = (133, 197, 215)` output (a chroma cascade emerges
//!    because the desynced entropy state after the wrong AC sign reads
//!    flips the chroma `txb_skip` decode from `0` to `1`, leaving the
//!    chroma predictor unmodified — `(128, 128)` instead of
//!    `(197, 215)`).
//! 2. `decode_bool(16384)` and `decode_symbol(&mut [16384, 0, 0])`
//!    produce identical post-state (added test
//!    `symbol::tests::decode_bool_and_decode_symbol_two_way_agree`).
//! 3. Reversing the sign loop order (eob-1 → 0) produces a different
//!    but still incorrect sign pattern.
//! 4. Bypassing the dc_sign CDF and using a literal bit instead at
//!    scan_idx 0 changes the post-state and uncovers a different
//!    chroma desync (U = 129).
//!
//! The divergence is consistent with an entropy state delta vs `dav1d`
//! that is *too small to flip any of the 4-way `coeff_base` or
//! `coeff_br` symbols* (which sit on probability mass thresholds far
//! from the 0.5 boundary), yet *just large enough to flip 4 of 5*
//! 50/50 literal `sign_bit` reads. Tracking down the source of that
//! delta requires comparing range-coder state against a dav1d build
//! with internal-state debug logging — a task deferred for round 50.
//!
//! Closing this divergence would lift the test in
//! `tests/issue_791_partition_force_split_for_tiny_frames.rs` from
//! `(Y, U, V) = (130, 128, 128)` to `(133, 197, 215)`. Pinning the
//! current behaviour here gives round-49 work a sentinel.

use oxideav_av1::decoder::Av1Decoder;
use oxideav_core::{CodecId, CodecParameters, Decoder, Frame, Packet, TimeBase};

/// 27-byte AV1 OBU stream extracted from
/// `crates/oxideav-avif/tests/fixtures/fuzz/y_plane_divergence_match.avif`.
/// Identical to the bitstream pinned by `tests/issue_786_*` and
/// `tests/issue_791_*`.
const DIVERGENCE_OBU: &[u8] = &[
    0x12, 0x00, 0x0a, 0x04, 0x38, 0x00, 0x0e, 0x49, 0x32, 0x11, 0x10, 0x00, 0x00, 0x19, 0xb9, 0xca,
    0xe3, 0x37, 0x39, 0x09, 0x47, 0xd9, 0x6e, 0x65, 0x96, 0x64, 0xaf,
];

/// `dav1d 1.5.3` and `avifdec --raw-color` raw YUV reference. These
/// are the target values for any future entropy-decoder fix that
/// closes the §5.11.39 sign divergence.
const REF_Y: u8 = 133;
const REF_U: u8 = 197;
const REF_V: u8 = 215;

fn decode_divergence_yuv() -> (u8, u8, u8) {
    let mut dec = Av1Decoder::new(CodecParameters::video(CodecId::new("av1")));
    let pkt = Packet::new(0, TimeBase::new(1, 1), DIVERGENCE_OBU.to_vec())
        .with_pts(0)
        .with_keyframe(true);
    dec.send_packet(&pkt)
        .expect("divergence OBU must decode without error");
    while let Ok(f) = dec.receive_frame() {
        if let Frame::Video(vf) = f {
            assert_eq!(vf.planes.len(), 3, "YUV444 must surface 3 planes");
            return (
                vf.planes[0].data[0],
                vf.planes[1].data[0],
                vf.planes[2].data[0],
            );
        }
    }
    panic!("a Video frame must be produced");
}

/// Decode must complete without panicking — the AVIF-side fuzz
/// contract restored in round 44.
#[test]
fn issue_796_divergence_obu_decodes_without_panic() {
    let _ = decode_divergence_yuv();
}

/// Pin the current oxideav decode output alongside the `dav1d` /
/// `avifdec` reference. Updating these asserts (specifically — the
/// non-reference column to match `REF_*`) marks the close of the
/// remaining §5.11.39 sign-bit divergence.
#[test]
fn issue_796_yuv_matches_pinned_current_output() {
    let (y, u, v) = decode_divergence_yuv();
    // Current oxideav output. Differs from REF_* by the per-plane
    // entropy-state delta documented in the module docstring.
    assert_eq!(
        y, 130,
        "Y(0,0): currently 130; dav1d reference is {REF_Y} — see module \
         doc for the AC-sign-read divergence root-cause analysis"
    );
    assert_eq!(
        u, 128,
        "U(0,0): currently 128 (chroma TU read as txb_skip=1 — entropy \
         desynced); dav1d reference is {REF_U}"
    );
    assert_eq!(
        v, 128,
        "V(0,0): currently 128 (chroma TU read as txb_skip=1 — entropy \
         desynced); dav1d reference is {REF_V}"
    );
}

/// Document the per-plane delta vs the cross-decoder reference so
/// progress on the entropy divergence is visible in the test output.
#[test]
fn issue_796_delta_vs_dav1d_reference_is_documented() {
    let (y, u, v) = decode_divergence_yuv();
    let dy = (REF_Y as i32) - (y as i32);
    let du = (REF_U as i32) - (u as i32);
    let dv = (REF_V as i32) - (v as i32);
    assert_eq!(dy, 3, "Y delta vs dav1d/avifdec reference");
    assert_eq!(du, 69, "U delta vs dav1d/avifdec reference");
    assert_eq!(dv, 87, "V delta vs dav1d/avifdec reference");
}

/// Round-67 black-box pin: feeding `DIVERGENCE_OBU` to `dav1d 1.5.3 -i -`
/// produces `(Y, U, V) = (133, 197, 215)` exactly. This was verified
/// out-of-band during the round-67 audit (`dav1d -i divergence.obu -o
/// /tmp/decoded.yuv` → `xxd /tmp/decoded.yuv` → `85 c5 d7` = 133, 197,
/// 215). The values are duplicated as `REF_Y`/`REF_U`/`REF_V`
/// above; this test exists so any future change that updates them
/// surfaces an explicit assertion site. Decoupled from the running
/// decoder so it pins the *target*, not the *current* output —
/// `issue_796_yuv_matches_pinned_current_output` above pins the
/// current output for the inevitable day the entropy divergence is
/// closed.
#[test]
fn issue_796_dav1d_reference_yuv_pinned() {
    // dav1d 1.5.3 -i divergence.obu -o decoded.yuv produced:
    //   00000000: 85 c5 d7    (= 133, 197, 215)
    // This is the round-67 black-box reference. Closing the
    // entropy divergence updates the *current-output* test above to
    // these values.
    assert_eq!(
        (REF_Y, REF_U, REF_V),
        (133, 197, 215),
        "dav1d 1.5.3 reference output for DIVERGENCE_OBU is pinned at \
         (Y, U, V) = (133, 197, 215). Update both this assert AND \
         issue_796_yuv_matches_pinned_current_output once the §5.11.39 \
         sign-bit divergence closes."
    );
}

/// Round-66 pinned trace: when the crate is built with the `rc-trace`
/// feature and `OXIDEAV_AV1_RC_TRACE` points to a file, decoding the
/// `divergence.avif` OBU emits exactly 34 range-coder operations (1
/// init + 33 symbol/bool decodes). The pinned trace is checked in at
/// `tests/fixtures/issue_796_rc_trace.jsonl` for round-67 diffing
/// against any future dav1d-side instrumentation.
///
/// Without the `rc-trace` feature this test is a no-op smoke check
/// (the rc-trace counter reset is a no-op at compile time). With the
/// feature on, decode produces the documented call sequence; the
/// fixture file is the round-66 hand-off artefact.
#[test]
fn issue_796_rc_trace_fixture_is_pinned() {
    // Reset the trace counter so repeated decode runs in this test
    // suite start fresh — important when other tests in the same
    // process also exercise the symbol decoder.
    oxideav_av1::symbol::reset_rc_trace_counter();
    let _ = decode_divergence_yuv();
    // The fixture is at `tests/fixtures/issue_796_rc_trace.jsonl`.
    // Round 66 verified it has 34 lines (init + 33 calls); the 5 AC
    // sign reads are calls 27, 28, 29, 30, 31 — all on
    // `decode_bool(16384)` with post-renorm `range = 45576`.
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/issue_796_rc_trace.jsonl");
    if fixture_path.exists() {
        let s = std::fs::read_to_string(&fixture_path).expect("read trace fixture");
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(
            lines.len(),
            34,
            "rc-trace fixture should have 34 lines (1 init + 33 calls); \
             update the fixture if the entropy decode path changes"
        );
        assert!(
            lines[27].contains("\"call_idx\":27") && lines[27].contains("\"op\":\"decode_bool\""),
            "call_idx 27 should be the first AC `sign_bit L(1)` read — \
             the documented divergence point"
        );
    }
}
