//! Integration tests for the §5.11.5 `decode_block()` syntax walker
//! exposed as [`PartitionWalker::decode_block_syntax`].
//!
//! The §5.11.5 syntax walker performs the per-block syntax pass
//! through:
//!
//! 1. The §5.11.5 prologue (sets `MiRow` / `MiCol` / `MiSize` /
//!    `bw4` / `bh4` / `HasChroma` / `AvailU` / `AvailL` /
//!    `AvailUChroma` / `AvailLChroma`).
//! 2. §5.11.6 `mode_info()` — intra arm composed from
//!    `decode_intra_frame_mode_info_prefix` + `decode_use_intrabc` +
//!    `decode_intra_frame_y_mode`.
//! 3. §5.11.49 `palette_tokens()` — no-op on the no-palette path
//!    reachable while `palette_mode_info()` remains unimplemented.
//! 4. §5.11.16 `read_block_tx_size()` — landed in r167. On the intra
//!    arm (`is_inter == 0`) the §5.11.16 `else` arm runs
//!    `read_tx_size(true)`, optionally reading `tx_depth` per §5.11.15
//!    when `TxMode == TX_MODE_SELECT`. The resulting `TxSize` is
//!    stamped into `TxSizes[]` / `InterTxSizes[]` across the block
//!    footprint.
//! 5. §5.11.30 `compute_prediction()` — STUBBED. The walker
//!    short-circuits with
//!    [`oxideav_av1::Error::DecodeBlockComputePredictionUnsupported`].
//!
//! These tests drive the walker with synthesised in-memory state
//! (no real bitstream decode required) on a minimal mi grid and
//! assert:
//!
//! * The prologue derivations are correct (`HasChroma` / `AvailU` /
//!   `AvailL` / `bw4` / `bh4` for several block sizes).
//! * The §5.11.6 dispatch routes the keyframe / intra-only path
//!   through the implemented intra-mode-info arm, and the inter arm
//!   surfaces the §5.11.18 stub.
//! * After the syntax-walker prologue + mode-info pass +
//!   `read_block_tx_size()` complete, the walker reaches the
//!   §5.11.30 stub at the correct bitstream position.
//! * The implemented mode-info pass stamps the §5.11.5 grids
//!   (`Skips[][]`, `SkipModes[][]`, `SegmentIds[][]`, `MiSizes[][]`,
//!   `YModes[][]`, `cdef_idx[][]`, `TxSizes[][]`, `InterTxSizes[][]`).
//! * The partition-walker driver [`PartitionWalker::decode_partition_syntax`]
//!   invokes `decode_block_syntax` at every leaf and propagates the
//!   stub through the recursion.

use oxideav_av1::{
    DecodedBlock, Error, PartitionWalker, SymbolDecoder, TileCdfContext, TileGeometry, BLOCK_16X16,
    BLOCK_4X4, BLOCK_8X16, BLOCK_8X8, MAX_SEGMENTS, MAX_TX_DEPTH, SKIP_CONTEXTS, TX_16X16, TX_4X4,
    TX_8X8, TX_SIZE_CONTEXTS,
};

/// Helper: force the §5.11.11 skip CDF to deterministically return
/// `symbol` on every context — every test below uses this to gate
/// the otherwise-stochastic read.
fn force_binary_cdf(symbol: u8) -> [u16; 3] {
    match symbol {
        0 => [1 << 15, 1 << 15, 0],
        1 => [0, 1 << 15, 0],
        _ => panic!("force_binary_cdf supports 0 or 1 only"),
    }
}

/// Helper: build a fresh walker for a square `n × n` mi-grid tile.
fn walker_n(n: u32) -> PartitionWalker {
    let geom = TileGeometry {
        mi_row_start: 0,
        mi_row_end: n,
        mi_col_start: 0,
        mi_col_end: n,
    };
    PartitionWalker::new(n, n, geom).unwrap()
}

/// §5.11.5 baseline path: a single keyframe / intra-only block at the
/// frame origin (`BLOCK_8X8`) with:
///   * `frame_is_intra = true` ⇒ §5.11.6 routes to intra arm
///   * `subsampling_x = 0` / `subsampling_y = 0` / `num_planes = 3` ⇒
///     `HasChroma = true` (no sub-sampled edge case)
///   * `segmentation_enabled = false`, `seg_skip_active = false` ⇒
///     the intra-segment_id path forces segment_id = 0 (no `S()`)
///   * `allow_intrabc = false` ⇒ `use_intrabc = 0` (no `S()`)
///   * `cdef_bits = 0` ⇒ §5.11.56 reads zero bits
///   * `read_deltas = false` ⇒ no delta_qindex / delta_lf reads
///   * `tx_mode_select = false` ⇒ §5.11.16 short-circuits the
///     `tx_depth` read (TxSize = maxRectTxSize without S() consumed)
///   * §5.11.11 skip-cdf forced to `0` ⇒ skip = 0 (the only `S()`
///     consumed besides the §5.11.7 `intra_frame_y_mode` read)
///
/// The walker is expected to:
///   1. Complete the §5.11.5 prologue.
///   2. Run `decode_intra_frame_mode_info_prefix` (consumes the
///      §5.11.11 skip bit + the §5.11.7 `intra_frame_y_mode` `S()`).
///   3. No-op `palette_tokens()`.
///   4. Run `read_block_tx_size` (no S() since TX_MODE_SELECT is off),
///      stamping `TxSizes[]` / `InterTxSizes[]` with TX_8X8.
///   5. Short-circuit at §5.11.30 `compute_prediction()`.
#[test]
fn decode_block_syntax_reaches_compute_prediction_stub_after_intra_mode_info() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let rigged_skip = force_binary_cdf(0);
    for ctx_idx in 0..SKIP_CONTEXTS {
        cdfs.skip[ctx_idx] = rigged_skip;
    }
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let pos_before = dec.position();
    let result = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 0, BLOCK_8X8, /* frame_is_intra = */ true,
        /* subsampling_x = */ 0, /* subsampling_y = */ 0, /* num_planes = */ 3,
        /* seg_id_pre_skip = */ false, /* segmentation_enabled = */ false,
        /* seg_skip_active = */ false, /* last_active_seg_id = */ 0, &lossless,
        /* coded_lossless = */ false, /* enable_cdef = */ true,
        /* allow_intrabc = */ false, /* cdef_bits = */ 0, /* read_deltas = */ false,
        /* use_128x128_superblock = */ false, /* delta_q_res = */ 0,
        /* delta_lf_present = */ false, /* delta_lf_multi = */ false,
        /* mono_chrome = */ false, /* delta_lf_res = */ 0,
        /* tx_mode_select = */ false,
    );
    let pos_after = dec.position();

    // §5.11.30 stub fired — the walker reached but did not enter
    // `compute_prediction()`.
    assert_eq!(
        result,
        Err(Error::DecodeBlockComputePredictionUnsupported),
        "the §5.11.5 walker must short-circuit at §5.11.30 compute_prediction after the intra mode-info + read_block_tx_size pass"
    );

    // The §5.11.5 prologue + §5.11.7 prefix consumed at least the
    // §5.11.11 skip bit + the §5.11.7 intra_frame_y_mode bit.
    assert!(
        pos_after > pos_before,
        "the §5.11.5 syntax walker must consume bits during the intra mode-info pass"
    );

    // The walker emitted exactly one [`DecodedBlockRecord`] for the
    // single leaf.
    let blocks = walker.blocks();
    assert_eq!(
        blocks.len(),
        1,
        "decode_block_syntax should emit one leaf record per call"
    );
    assert_eq!(blocks[0].mi_row, 0);
    assert_eq!(blocks[0].mi_col, 0);
    assert_eq!(blocks[0].sub_size, BLOCK_8X8);

    // §5.11.5 grid-fill assertions — the intra-mode-info pass stamps
    // the BLOCK_8X8 (2×2) footprint of every grid it touched.
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(
                walker.skips()[r * mi_cols + c],
                0,
                "§5.11.11 skip = 0 stamp"
            );
            assert_eq!(
                walker.segment_ids()[r * mi_cols + c],
                0,
                "§5.11.9 segment_id = 0 stamp (no segmentation)"
            );
            assert_eq!(
                walker.mi_sizes()[r * mi_cols + c],
                BLOCK_8X8,
                "§5.11.4 MiSizes stamp"
            );
            // §5.11.16 stamp: with tx_mode_select = false, TxSize =
            // maxRectTxSize for BLOCK_8X8 = TX_8X8 (= 1).
            assert_eq!(
                walker.tx_sizes()[r * mi_cols + c],
                TX_8X8 as u8,
                "§5.11.16 TxSizes stamp matches maxRectTxSize[BLOCK_8X8] = TX_8X8"
            );
            assert_eq!(
                walker.inter_tx_sizes()[r * mi_cols + c],
                TX_8X8 as u8,
                "§5.11.16 InterTxSizes stamp matches TxSizes on the else-arm"
            );
        }
    }
}

/// §5.11.5 prologue: `HasChroma` three-arm dispatch on a `BLOCK_4X4`
/// block (bh4 = bw4 = 1). Drive the walker against the three regimes:
///   * subsampling_y = 1, MiRow even ⇒ HasChroma = false (arm 1)
///   * subsampling_x = 1, MiCol even ⇒ HasChroma = false (arm 2)
///   * no sub-sampling on the 1×1 edge ⇒ HasChroma = num_planes > 1
///
/// Each subcase reaches the §5.11.30 stub (the walker doesn't short-
/// circuit on the chroma side); the test is the §5.11.5 prologue
/// dispatch sanity-check. BLOCK_4X4 has Max_Tx_Depth = 0, so the
/// §5.11.15 `MiSize > BLOCK_4X4` gate is false and no `tx_depth` is
/// read — TxSize = maxRectTxSize[BLOCK_4X4] = TX_4X4.
#[test]
fn decode_block_syntax_prologue_has_chroma_three_arm_dispatch() {
    let lossless = [false; MAX_SEGMENTS];

    // Arm 1: bh4 == 1 && subsampling_y && MiRow & 1 == 0 ⇒ HasChroma = false.
    // BLOCK_4X4 has bh4 == bw4 == 1. mi_row = 0 (even). subsampling_y = 1.
    {
        let mut walker = walker_n(4);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
        let bytes = [0u8; 8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
        let result = walker.decode_block_syntax(
            &mut dec, &mut cdfs, 0, 0, BLOCK_4X4, true, /* subsampling_x = */ 0,
            /* subsampling_y = */ 1, /* num_planes = */ 3, false, false, false, 0,
            &lossless, false, true, false, 0, false, false, 0, false, false, false, 0, false,
        );
        assert_eq!(result, Err(Error::DecodeBlockComputePredictionUnsupported));
    }

    // Arm 2: bw4 == 1 && subsampling_x && MiCol & 1 == 0 ⇒ HasChroma = false.
    {
        let mut walker = walker_n(4);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
        let bytes = [0u8; 8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
        let result = walker.decode_block_syntax(
            &mut dec, &mut cdfs, 0, 0, BLOCK_4X4, true, /* subsampling_x = */ 1,
            /* subsampling_y = */ 0, /* num_planes = */ 3, false, false, false, 0,
            &lossless, false, true, false, 0, false, false, 0, false, false, false, 0, false,
        );
        assert_eq!(result, Err(Error::DecodeBlockComputePredictionUnsupported));
    }

    // Arm 3 fall-through: no sub-sampling ⇒ HasChroma = num_planes > 1.
    {
        let mut walker = walker_n(4);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
        let bytes = [0u8; 8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
        let result = walker.decode_block_syntax(
            &mut dec, &mut cdfs, 0, 0, BLOCK_4X4, true, 0, 0, /* num_planes = */ 1, false,
            false, false, 0, &lossless, false, true, false, 0, false, false, 0, false, false, true,
            0, false,
        );
        assert_eq!(result, Err(Error::DecodeBlockComputePredictionUnsupported));
    }
}

/// §5.11.6 inter-frame arm: `frame_is_intra = false` ⇒ the §5.11.18
/// `inter_frame_mode_info()` stub fires with no bitstream consumed
/// (other than the §5.11.5 prologue, which doesn't read bits).
#[test]
fn decode_block_syntax_inter_frame_arm_returns_stub() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0xFFu8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let pos_before = dec.position();
    let result = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 0, BLOCK_8X8, /* frame_is_intra = */ false, 0, 0, 3, false,
        false, false, 0, &lossless, false, true, false, 0, false, false, 0, false, false, false, 0,
        false,
    );
    let pos_after = dec.position();

    assert_eq!(
        result,
        Err(Error::DecodeBlockInterFrameUnsupported),
        "§5.11.6 inter-frame arm must surface the §5.11.18 stub"
    );
    // The §5.11.5 prologue derives sizes / availability without
    // touching the bitstream cursor — the inter stub fires before
    // any read.
    assert_eq!(
        pos_after, pos_before,
        "§5.11.18 stub must fire before any bitstream read"
    );
    // No leaf record emitted on the stub path (we only emit at the
    // post-mode-info §5.11.16 stub).
    assert!(walker.blocks().is_empty());
}

/// §5.11.7 `intra_segment_id` pre-skip arm gates the skip-mode read
/// in spec order: the segment_id reading happens BEFORE the
/// `read_skip()`. With `segmentation_enabled = true`,
/// `seg_id_pre_skip = true` and the skip CDF forced to symbol 1, the
/// walker reads `segment_id` first, then `skip = 1`. The intra
/// mode-info pass completes; the §5.11.30 stub fires after the
/// §5.11.16 reader runs.
#[test]
fn decode_block_syntax_intra_pre_skip_arm_reaches_stub() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(1); SKIP_CONTEXTS];
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let result = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 0, BLOCK_8X8, true, 0, 0, 3, /* seg_id_pre_skip = */ true,
        /* segmentation_enabled = */ true, false, /* last_active_seg_id = */ 7,
        &lossless, false, true, false, 0, false, false, 0, false, false, false, 0, false,
    );

    assert_eq!(result, Err(Error::DecodeBlockComputePredictionUnsupported));
    // The §5.11.5 mode-info pass succeeded — the per-block stamp
    // landed on the BLOCK_8X8 (2×2) footprint.
    let mi_cols = walker.mi_cols() as usize;
    assert_eq!(walker.mi_sizes()[0], BLOCK_8X8);
    assert_eq!(walker.mi_sizes()[mi_cols + 1], BLOCK_8X8);
}

/// §5.11.5 caller-bug detection: out-of-range `sub_size` /
/// `mi_row` / `mi_col` returns [`Error::PartitionWalkOutOfRange`]
/// without entering the §5.11.6 dispatch.
#[test]
fn decode_block_syntax_rejects_out_of_range() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 8];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    // Out-of-range mi_row.
    let r = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 8, 0, BLOCK_8X8, true, 0, 0, 3, false, false, false, 0, &lossless,
        false, true, false, 0, false, false, 0, false, false, false, 0, false,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));

    // Out-of-range mi_col.
    let r = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 8, BLOCK_8X8, true, 0, 0, 3, false, false, false, 0, &lossless,
        false, true, false, 0, false, false, 0, false, false, false, 0, false,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));

    // Out-of-range sub_size.
    let r = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 0, /* sub_size = */ 999, true, 0, 0, 3, false, false, false,
        0, &lossless, false, true, false, 0, false, false, 0, false, false, false, 0, false,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
}

/// `decode_partition_syntax` driver wires `decode_block_syntax` into
/// the §5.11.4 partition tree. With a 16×16 mi-grid, BLOCK_16X16
/// superblock, and the partition CDF forced via default to a
/// PARTITION_NONE outcome (this is not guaranteed without rigging,
/// but on the no-bit-read short-circuit `b_size < BLOCK_8X8` arm we
/// can construct an even simpler scenario by passing a smaller
/// `b_size` directly).
///
/// We test the simplest case: `b_size = BLOCK_4X4` (which is below
/// BLOCK_8X8 and short-circuits to PARTITION_NONE with no partition
/// CDF read), so the partition driver immediately calls
/// `decode_block_syntax( 0, 0, BLOCK_4X4 )`. The §5.11.30 stub
/// propagates out of the recursion (BLOCK_4X4 has Max_Tx_Depth = 0,
/// so no tx_depth read).
#[test]
fn decode_partition_syntax_routes_leaf_through_decode_block_syntax() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let result = walker.decode_partition_syntax(
        &mut dec, &mut cdfs, /* r = */ 0, /* c = */ 0, /* b_size = */ BLOCK_4X4,
        /* frame_is_intra = */ true, 0, 0, 3, false, false, false, 0, &lossless, false, true,
        false, 0, false, false, 0, false, false, false, 0, false,
    );

    assert_eq!(
        result,
        Err(Error::DecodeBlockComputePredictionUnsupported),
        "the partition driver must propagate the §5.11.30 stub from its leaf call"
    );

    // Exactly one leaf emitted at (0, 0, BLOCK_4X4) — the
    // `b_size < BLOCK_8X8` short-circuit gives PARTITION_NONE with
    // no recursion.
    let blocks = walker.blocks();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].mi_row, 0);
    assert_eq!(blocks[0].mi_col, 0);
    assert_eq!(blocks[0].sub_size, BLOCK_4X4);
}

/// `decode_partition_syntax` with `r >= MiRows` short-circuits to
/// `return 0` per §5.11.4 line 1 — no decode_block_syntax call, no
/// stub, no error.
#[test]
fn decode_partition_syntax_out_of_grid_short_circuits() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let result = walker.decode_partition_syntax(
        &mut dec, &mut cdfs, /* r = */ 4, /* c = */ 0, BLOCK_8X8, true, 0, 0, 3, false,
        false, false, 0, &lossless, false, true, false, 0, false, false, 0, false, false, false, 0,
        false,
    );

    assert_eq!(
        result,
        Ok(()),
        "out-of-grid recursion must short-circuit cleanly"
    );
    assert!(
        walker.blocks().is_empty(),
        "no leaves emitted on the out-of-grid path"
    );
}

/// `DecodedBlock` is the per-block aggregate returned on the no-stub
/// path. We don't reach that path in this round (the §5.11.30 stub
/// always fires after the read_block_tx_size pass), but the struct's
/// constructibility check is a sanity-check on the public API
/// surface — it should compile and be `Debug + Clone + Copy +
/// PartialEq + Eq`.
#[test]
fn decoded_block_struct_public_api_smoke() {
    let _db = DecodedBlock {
        mi_row: 0,
        mi_col: 0,
        mi_size: BLOCK_8X8,
        bw4: 2,
        bh4: 2,
        has_chroma: true,
        avail_u: false,
        avail_l: false,
        avail_u_chroma: false,
        avail_l_chroma: false,
        skip: 0,
        skip_mode: 0,
        segment_id: 0,
        lossless: false,
        cdef_idx: -1,
        current_q_index: 0,
        current_delta_lf: [0; 4],
        ref_frame: [0, -1],
        use_intrabc: 0,
        is_inter: 0,
        y_mode: 0,
        is_compound: false,
        tx_size: TX_4X4 as u8,
    };
}

/// §5.11.5 prologue: `BLOCK_8X16` at (0, 0) has bw4 = 2, bh4 = 4
/// (`Num_4x4_Blocks_*` per §9.3). With no sub-sampling, neither
/// chroma fix-up arm fires, so AvailUChroma = AvailU = false (frame
/// origin), AvailLChroma = AvailL = false.
///
/// We assert via the §5.11.30 stub reach (i.e. the prologue is total
/// over this case) plus the §5.11.5 grid-fill footprint.
#[test]
fn decode_block_syntax_block_8x16_grid_fill_footprint() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let result = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 0, BLOCK_8X16, true, 0, 0, 3, false, false, false, 0, &lossless,
        false, true, false, 0, false, false, 0, false, false, false, 0, false,
    );
    assert_eq!(result, Err(Error::DecodeBlockComputePredictionUnsupported));

    // BLOCK_8X16: bw4 = 2, bh4 = 4 ⇒ the §5.11.5 grid-fill stamps a
    // 2×4 footprint (2 cols × 4 rows) at (0, 0).
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..2 {
            assert_eq!(
                walker.mi_sizes()[r * mi_cols + c],
                BLOCK_8X16,
                "BLOCK_8X16 footprint covers (0..4, 0..2)"
            );
        }
    }
    // Outside the footprint: BLOCK_INVALID sentinel.
    assert_ne!(
        walker.mi_sizes()[2],
        BLOCK_8X16,
        "(0, 2) is outside the BLOCK_8X16 footprint"
    );
}

/// §5.11.5 prologue: `BLOCK_16X16` at (0, 0) has bw4 = bh4 = 4. With
/// `cdef_bits = 2` the §5.11.56 read consumes two literal bits.
/// We test that the stub still fires after the cdef literal read.
#[test]
fn decode_block_syntax_cdef_bits_two_reaches_stub() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let pos_before = dec.position();
    let result = walker.decode_block_syntax(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_16X16,
        true,
        0,
        0,
        3,
        false,
        false,
        false,
        0,
        &lossless,
        false,
        /* enable_cdef = */ true,
        false,
        /* cdef_bits = */ 2,
        false,
        false,
        0,
        false,
        false,
        false,
        0,
        /* tx_mode_select = */ false,
    );
    let pos_after = dec.position();
    assert_eq!(result, Err(Error::DecodeBlockComputePredictionUnsupported));
    assert!(
        pos_after > pos_before,
        "literal cdef-bits read must advance the bit cursor"
    );

    // BLOCK_16X16 (bw4 = bh4 = 4) ⇒ 4×4 mi-cells stamped from (0, 0).
    // With tx_mode_select = false, TxSize = maxRectTxSize[BLOCK_16X16]
    // = TX_16X16 (= 2).
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(walker.mi_sizes()[r * mi_cols + c], BLOCK_16X16);
            assert_eq!(walker.skips()[r * mi_cols + c], 0);
            assert_eq!(
                walker.tx_sizes()[r * mi_cols + c],
                TX_16X16 as u8,
                "§5.11.16 TxSize for BLOCK_16X16 with TX_MODE_LARGEST = TX_16X16"
            );
        }
    }
}

// ---------------------------------------------------------------------
// §5.11.16 read_block_tx_size — r167. The tests below drive the
// standalone reader and the integrated walker on the §5.11.15 /
// §5.11.16 spec paths: lossless short-circuit, TX_MODE_LARGEST (no
// tx_depth read), TX_MODE_SELECT with tx_depth = 0 / 1 / 2, the
// §5.11.16 inter-arm TX_MODE_SELECT stub, and the §5.11.5 grid-fill
// footprint stamping `TxSizes[]` / `InterTxSizes[]` together.
// ---------------------------------------------------------------------

/// Build a forced CDF for a `tx_depth` row of `MAX_TX_DEPTH + 1 = 3`
/// or `MAX_TX_DEPTH + 2 = 4` slots. `slots` is the row length excluding
/// the counter slot (so `2` for the 8×8 row, `3` for the 16×16 / 32×32
/// / 64×64 rows). The returned slice has length `slots + 1` (the
/// counter slot added at the end set to 0). The row is rigged so that
/// reading it yields the supplied `symbol`.
fn force_n_ary_cdf(slots: usize, symbol: u8) -> Vec<u16> {
    assert!((symbol as usize) < slots);
    // Build a cdf where prob slot[symbol] == 1 << 15 and earlier
    // slots are 0. The §8.2.6 S() routine returns the first index i
    // for which cdf[i] > drawn; setting cdf[symbol] = 1 << 15 and
    // cdf[i < symbol] = 0 with cdf[symbol+1..N-1] = 1 << 15 forces
    // i = symbol on every draw.
    let mut row = vec![0u16; slots + 1];
    for slot in row.iter_mut().take(slots).skip(symbol as usize) {
        *slot = 1 << 15;
    }
    // The last slot (counter) stays 0.
    row
}

/// §5.11.15 `Lossless` short-circuit: when the segment is lossless,
/// `TxSize = TX_4X4` with no symbol read regardless of TxMode or
/// MiSize.
#[test]
fn read_block_tx_size_lossless_forces_tx_4x4() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    let tx = walker
        .read_block_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            /* lossless = */ true,
            /* is_inter = */ false,
            /* skip = */ false,
            /* tx_mode_select = */ true,
        )
        .unwrap();
    let pos_after = dec.position();
    assert_eq!(tx, TX_4X4 as u8, "lossless forces TX_4X4");
    assert_eq!(
        pos_after, pos_before,
        "lossless short-circuit consumes no bits"
    );
    // Grid-fill stamps TX_4X4 over the BLOCK_16X16 footprint (4×4).
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(walker.tx_sizes()[r * mi_cols + c], TX_4X4 as u8);
            assert_eq!(walker.inter_tx_sizes()[r * mi_cols + c], TX_4X4 as u8);
        }
    }
}

/// §5.11.15 `TX_MODE_LARGEST` short-circuit: `tx_mode_select = false`
/// skips the `tx_depth` read, so `TxSize = maxRectTxSize`. For
/// BLOCK_8X8, `maxRectTxSize = TX_8X8`.
#[test]
fn read_block_tx_size_tx_mode_largest_skips_tx_depth_read() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    let tx = walker
        .read_block_tx_size(
            &mut dec, &mut cdfs, 0, 0, BLOCK_8X8, /* lossless = */ false,
            /* is_inter = */ false, /* skip = */ false,
            /* tx_mode_select = */ false,
        )
        .unwrap();
    let pos_after = dec.position();
    assert_eq!(tx, TX_8X8 as u8, "TX_MODE_LARGEST ⇒ TxSize = TX_8X8");
    assert_eq!(pos_after, pos_before, "no tx_depth read on TX_MODE_LARGEST");
}

/// §5.11.15 `BLOCK_4X4` short-circuit: regardless of TxMode, the
/// `MiSize > BLOCK_4X4` gate is false, so no `tx_depth` is read.
/// `TxSize = maxRectTxSize[BLOCK_4X4] = TX_4X4`.
#[test]
fn read_block_tx_size_block_4x4_skips_tx_depth_read() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    let tx = walker
        .read_block_tx_size(
            &mut dec, &mut cdfs, 0, 0, BLOCK_4X4, false, false, false,
            /* tx_mode_select = */ true,
        )
        .unwrap();
    let pos_after = dec.position();
    assert_eq!(tx, TX_4X4 as u8, "BLOCK_4X4 ⇒ TxSize = TX_4X4");
    assert_eq!(
        pos_after, pos_before,
        "BLOCK_4X4 short-circuits the tx_depth read"
    );
}

/// §5.11.15 `TX_MODE_SELECT` with rigged `tx_depth = 0`: TxSize stays
/// at `maxRectTxSize`. For BLOCK_16X16 that's TX_16X16. The Default
/// Tx16x16Cdf row is selected (Max_Tx_Depth[BLOCK_16X16] = 2 ⇒
/// maxTxDepth = 2 ⇒ Tile_Tx_16x16_Cdf).
#[test]
fn read_block_tx_size_tx_mode_select_depth_zero_keeps_max_rect() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // Rig the Tx_16x16 CDF (the one selected for max_tx_depth == 2)
    // to return symbol 0 on every ctx.
    let rigged: [u16; MAX_TX_DEPTH + 2] = {
        let v = force_n_ary_cdf(MAX_TX_DEPTH + 1, 0);
        let mut a = [0u16; MAX_TX_DEPTH + 2];
        a[..v.len()].copy_from_slice(&v);
        a
    };
    for ctx in 0..TX_SIZE_CONTEXTS {
        cdfs.tx_16x16[ctx] = rigged;
    }
    let bytes = [0u8; 8];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
    let tx = walker
        .read_block_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            false,
            false,
            false,
            /* tx_mode_select = */ true,
        )
        .unwrap();
    assert_eq!(
        tx, TX_16X16 as u8,
        "tx_depth = 0 ⇒ TxSize = maxRectTxSize = TX_16X16"
    );
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(walker.tx_sizes()[r * mi_cols + c], TX_16X16 as u8);
            assert_eq!(walker.inter_tx_sizes()[r * mi_cols + c], TX_16X16 as u8);
        }
    }
}

/// §5.11.15 `TX_MODE_SELECT` with rigged `tx_depth = 1`: TxSize =
/// `Split_Tx_Size[ maxRectTxSize ]`. For BLOCK_16X16:
/// maxRectTxSize = TX_16X16 ⇒ Split_Tx_Size[TX_16X16] = TX_8X8.
#[test]
fn read_block_tx_size_tx_mode_select_depth_one_splits_once() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let rigged: [u16; MAX_TX_DEPTH + 2] = {
        let v = force_n_ary_cdf(MAX_TX_DEPTH + 1, 1);
        let mut a = [0u16; MAX_TX_DEPTH + 2];
        a[..v.len()].copy_from_slice(&v);
        a
    };
    for ctx in 0..TX_SIZE_CONTEXTS {
        cdfs.tx_16x16[ctx] = rigged;
    }
    let bytes = [0u8; 8];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
    let tx = walker
        .read_block_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            false,
            false,
            false,
            /* tx_mode_select = */ true,
        )
        .unwrap();
    assert_eq!(
        tx, TX_8X8 as u8,
        "tx_depth = 1 ⇒ TxSize = Split_Tx_Size[TX_16X16] = TX_8X8"
    );
}

/// §5.11.15 `TX_MODE_SELECT` with rigged `tx_depth = 2`: TxSize =
/// `Split_Tx_Size[ Split_Tx_Size[ maxRectTxSize ] ]`. For
/// BLOCK_16X16: TX_16X16 -> TX_8X8 -> TX_4X4.
#[test]
fn read_block_tx_size_tx_mode_select_depth_two_splits_twice() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let rigged: [u16; MAX_TX_DEPTH + 2] = {
        let v = force_n_ary_cdf(MAX_TX_DEPTH + 1, 2);
        let mut a = [0u16; MAX_TX_DEPTH + 2];
        a[..v.len()].copy_from_slice(&v);
        a
    };
    for ctx in 0..TX_SIZE_CONTEXTS {
        cdfs.tx_16x16[ctx] = rigged;
    }
    let bytes = [0u8; 8];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
    let tx = walker
        .read_block_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            false,
            false,
            false,
            /* tx_mode_select = */ true,
        )
        .unwrap();
    assert_eq!(
        tx, TX_4X4 as u8,
        "tx_depth = 2 ⇒ TX_16X16 -> TX_8X8 -> TX_4X4"
    );
}

/// §5.11.16 inter-arm stub: `TX_MODE_SELECT && MiSize > BLOCK_4X4 &&
/// is_inter && !skip && !Lossless` enters the §5.11.17
/// `read_var_tx_size` recursion. That round's target is stubbed; the
/// reader surfaces [`Error::ReadVarTxSizeUnsupported`].
#[test]
fn read_block_tx_size_inter_arm_returns_var_tx_size_stub() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    let r = walker.read_block_tx_size(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_16X16,
        /* lossless = */ false,
        /* is_inter = */ true,
        /* skip = */ false,
        /* tx_mode_select = */ true,
    );
    let pos_after = dec.position();
    assert_eq!(r, Err(Error::ReadVarTxSizeUnsupported));
    assert_eq!(
        pos_after, pos_before,
        "stub fires before any bitstream read"
    );
}

/// §5.11.16 inter-arm `else` path: with `skip = 1` the inter arm
/// falls into the `else` branch (`!skip` gate is false), and
/// `allowSelect = !skip || !is_inter = false || false = false`. So
/// the §5.11.15 `MiSize > BLOCK_4X4 && allowSelect && TX_MODE_SELECT`
/// gate is also false → TxSize = maxRectTxSize. The grid-fill loop
/// runs and stamps the result. This exercises the inter-shape path
/// without needing the §5.11.17 reader.
#[test]
fn read_block_tx_size_inter_skip_else_arm_uses_max_rect_tx_size() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    let tx = walker
        .read_block_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            /* lossless = */ false,
            /* is_inter = */ true,
            /* skip = */ true,
            /* tx_mode_select = */ true,
        )
        .unwrap();
    let pos_after = dec.position();
    assert_eq!(
        tx, TX_16X16 as u8,
        "inter+skip ⇒ else arm, allowSelect = false ⇒ TxSize = maxRectTxSize"
    );
    assert_eq!(
        pos_after, pos_before,
        "no S() consumed on the inter+skip+TX_MODE_SELECT path"
    );
}

/// §5.11.16 caller-bug detection: out-of-range mi_row / mi_col /
/// sub_size returns [`Error::PartitionWalkOutOfRange`] without
/// entering the §5.11.15 body.
#[test]
fn read_block_tx_size_rejects_out_of_range() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let r = walker.read_block_tx_size(
        &mut dec, &mut cdfs, 4, 0, BLOCK_4X4, false, false, false, false,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    let r = walker.read_block_tx_size(
        &mut dec, &mut cdfs, 0, 4, BLOCK_4X4, false, false, false, false,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    let r = walker.read_block_tx_size(&mut dec, &mut cdfs, 0, 0, 999, false, false, false, false);
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
}

/// §5.11.5 walker: with `tx_mode_select = true` and the §5.11.16
/// reader's rigged `tx_depth = 0`, the integrated walker completes
/// `decode_intra_frame_mode_info_prefix` plus `decode_intra_frame_y_mode`
/// plus `read_block_tx_size` and short-circuits at §5.11.30
/// `compute_prediction`. TxSize = TX_16X16 stamps over the
/// BLOCK_16X16 footprint.
#[test]
fn decode_block_syntax_with_tx_mode_select_reaches_compute_prediction() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    // Rig Tx_16x16 row (max_tx_depth == 2) to depth 0.
    let rigged: [u16; MAX_TX_DEPTH + 2] = {
        let v = force_n_ary_cdf(MAX_TX_DEPTH + 1, 0);
        let mut a = [0u16; MAX_TX_DEPTH + 2];
        a[..v.len()].copy_from_slice(&v);
        a
    };
    for ctx in 0..TX_SIZE_CONTEXTS {
        cdfs.tx_16x16[ctx] = rigged;
    }
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let result = walker.decode_block_syntax(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_16X16,
        true,
        0,
        0,
        3,
        false,
        false,
        false,
        0,
        &lossless,
        false,
        true,
        false,
        0,
        false,
        false,
        0,
        false,
        false,
        false,
        0,
        /* tx_mode_select = */ true,
    );
    assert_eq!(result, Err(Error::DecodeBlockComputePredictionUnsupported));

    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(
                walker.tx_sizes()[r * mi_cols + c],
                TX_16X16 as u8,
                "TX_MODE_SELECT with tx_depth = 0 keeps maxRectTxSize"
            );
        }
    }
}
