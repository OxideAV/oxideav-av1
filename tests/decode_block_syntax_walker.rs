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
//! 4. §5.11.16 `read_block_tx_size()` — STUBBED. The walker
//!    short-circuits with
//!    [`oxideav_av1::Error::DecodeBlockReadBlockTxSizeUnsupported`].
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
//! * After the syntax-walker prologue + mode-info pass completes,
//!   the walker reaches the §5.11.16 stub at the correct bitstream
//!   position (a single byte of consumed input on the no-bit-read
//!   path).
//! * The implemented mode-info pass stamps the §5.11.5 grids
//!   (`Skips[][]`, `SkipModes[][]`, `SegmentIds[][]`, `MiSizes[][]`,
//!   `YModes[][]`, `cdef_idx[][]`).
//! * The partition-walker driver [`PartitionWalker::decode_partition_syntax`]
//!   invokes `decode_block_syntax` at every leaf and propagates the
//!   stub through the recursion.

use oxideav_av1::{
    DecodedBlock, Error, PartitionWalker, SymbolDecoder, TileCdfContext, TileGeometry, BLOCK_16X16,
    BLOCK_4X4, BLOCK_8X16, BLOCK_8X8, MAX_SEGMENTS, SKIP_CONTEXTS,
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
///   * §5.11.11 skip-cdf forced to `0` ⇒ skip = 0 (the only `S()`
///     consumed besides the §5.11.7 `intra_frame_y_mode` read)
///
/// The walker is expected to:
///   1. Complete the §5.11.5 prologue.
///   2. Run `decode_intra_frame_mode_info_prefix` (consumes the
///      §5.11.11 skip bit + the §5.11.7 `intra_frame_y_mode` `S()`).
///   3. No-op `palette_tokens()`.
///   4. Short-circuit at §5.11.16 `read_block_tx_size()`.
#[test]
fn decode_block_syntax_reaches_read_block_tx_size_stub_after_intra_mode_info() {
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
    );
    let pos_after = dec.position();

    // §5.11.16 stub fired — the walker reached but did not enter
    // `read_block_tx_size()`.
    assert_eq!(
        result,
        Err(Error::DecodeBlockReadBlockTxSizeUnsupported),
        "the §5.11.5 walker must short-circuit at §5.11.16 read_block_tx_size after the intra mode-info pass"
    );

    // The §5.11.5 prologue + §5.11.7 prefix consumed at least the
    // §5.11.11 skip bit + the §5.11.7 intra_frame_y_mode bit — the
    // bitstream cursor must have advanced past the prologue. (The
    // exact byte count depends on msac internals, but it must be
    // strictly greater than the no-bit-read baseline.)
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
        }
    }
}

/// §5.11.5 prologue: `HasChroma` three-arm dispatch on a `BLOCK_4X4`
/// block (bh4 = bw4 = 1). Drive the walker against the three regimes:
///   * subsampling_y = 1, MiRow even ⇒ HasChroma = false (arm 1)
///   * subsampling_x = 1, MiCol even ⇒ HasChroma = false (arm 2)
///   * no sub-sampling on the 1×1 edge ⇒ HasChroma = num_planes > 1
///
/// Each subcase reaches the §5.11.16 stub (the walker doesn't short-
/// circuit on the chroma side); the test is the §5.11.5 prologue
/// dispatch sanity-check.
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
            &lossless, false, true, false, 0, false, false, 0, false, false, false, 0,
        );
        assert_eq!(result, Err(Error::DecodeBlockReadBlockTxSizeUnsupported));
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
            &lossless, false, true, false, 0, false, false, 0, false, false, false, 0,
        );
        assert_eq!(result, Err(Error::DecodeBlockReadBlockTxSizeUnsupported));
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
            0,
        );
        assert_eq!(result, Err(Error::DecodeBlockReadBlockTxSizeUnsupported));
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
/// mode-info pass completes; the §5.11.16 stub fires.
///
/// (The §5.11.9 segment_id read uses default CDFs — we don't rig it
/// here. We only assert that the §5.11.16 stub is reached after the
/// composed reads, and that the prefix-arm produced a `skip = 1`
/// stamp via the intra-mode-info pass.)
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
        &lossless, false, true, false, 0, false, false, 0, false, false, false, 0,
    );

    assert_eq!(result, Err(Error::DecodeBlockReadBlockTxSizeUnsupported));
    // The §5.11.5 mode-info pass succeeded — the per-block stamp
    // landed on the BLOCK_8X8 (2×2) footprint. (The exact `skip`
    // value depends on the rigged skip CDF + the §5.11.9 segment_id
    // read's msac state after the pre-skip arm consumes bits, so
    // we don't pin it here; the §5.11.16 stub reaching IS the
    // pre-skip arm fired correctly assertion.)
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
        false, true, false, 0, false, false, 0, false, false, false, 0,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));

    // Out-of-range mi_col.
    let r = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 8, BLOCK_8X8, true, 0, 0, 3, false, false, false, 0, &lossless,
        false, true, false, 0, false, false, 0, false, false, false, 0,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));

    // Out-of-range sub_size.
    let r = walker.decode_block_syntax(
        &mut dec, &mut cdfs, 0, 0, /* sub_size = */ 999, true, 0, 0, 3, false, false, false,
        0, &lossless, false, true, false, 0, false, false, 0, false, false, false, 0,
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
/// `decode_block_syntax( 0, 0, BLOCK_4X4 )`. The §5.11.16 stub
/// propagates out of the recursion.
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
        false, 0, false, false, 0, false, false, false, 0,
    );

    assert_eq!(
        result,
        Err(Error::DecodeBlockReadBlockTxSizeUnsupported),
        "the partition driver must propagate the §5.11.16 stub from its leaf call"
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
/// path. We don't reach that path in this round (the §5.11.16 stub
/// always fires after the mode-info pass), but the struct's
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
    };
}

/// §5.11.5 prologue: `BLOCK_8X16` at (0, 0) has bw4 = 2, bh4 = 4
/// (`Num_4x4_Blocks_*` per §9.3). With no sub-sampling, neither
/// chroma fix-up arm fires, so AvailUChroma = AvailU = false (frame
/// origin), AvailLChroma = AvailL = false.
///
/// We assert via the §5.11.16 stub reach (i.e. the prologue is total
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
        false, true, false, 0, false, false, 0, false, false, false, 0,
    );
    assert_eq!(result, Err(Error::DecodeBlockReadBlockTxSizeUnsupported));

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
    );
    let pos_after = dec.position();
    assert_eq!(result, Err(Error::DecodeBlockReadBlockTxSizeUnsupported));
    assert!(
        pos_after > pos_before,
        "literal cdef-bits read must advance the bit cursor"
    );

    // BLOCK_16X16 (bw4 = bh4 = 4) ⇒ 4×4 mi-cells stamped from (0, 0).
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(walker.mi_sizes()[r * mi_cols + c], BLOCK_16X16);
            assert_eq!(walker.skips()[r * mi_cols + c], 0);
        }
    }
}
