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
//! 5. §5.11.30 / §5.11.33 `compute_prediction()` — LANDED in r180.
//!    The walker runs the §5.11.33 per-plane dispatcher (one
//!    DC_PRED [`oxideav_av1::PlanePredictionTask`] per visited plane)
//!    and advances past §5.11.30; the §7.11.2.5 DC-PRED sample-
//!    generation leaf is exposed standalone as
//!    [`oxideav_av1::predict_intra_dc_pred`].
//! 6. §5.11.34 `residual()` — WIRED THROUGH (r181). The walker
//!    invokes `PartitionWalker::residual` which drives the §5.11.34
//!    outer dispatch + §5.11.36 transform_tree recursion + per-TU
//!    §5.11.39 `PartitionWalker::coefficients` bitstream read. On the
//!    `!skip` walker path the §5.11.39 reader will, with high
//!    probability against an unrigged bitstream, decode at least one
//!    non-zero TU.
//! 7. §5.11.35 `reconstruct()` — WIRED THROUGH (r182). For every TU
//!    with `eob > 0` the walker invokes
//!    [`oxideav_av1::inverse_transform_2d`] (§7.13.3 2D inverse
//!    transform) on the §5.11.39 `Quant[]` levels (passed through as
//!    a placeholder identity dequant — the §7.12.3 step-1
//!    quantization-matrix derivation is the next-arc target). The
//!    per-TU `Residual[][]` buffer is recorded on the readout. The
//!    walker now returns `Ok(DecodedBlock)` on both the `skip == 1`
//!    and the `!skip` cleanly-reconstructed paths.
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

use oxideav_av1::encoder::block_mode_info::{
    write_intra_frame_else_arm, write_intra_frame_intrabc_arm, write_skip, IntrabcArmInputs,
};
use oxideav_av1::encoder::symbol_writer::SymbolWriter;
use oxideav_av1::{
    get_palette_color_context, interintra_ctx, palette_color_ctx, BILINEAR, INTERINTRA_MODES,
    PALETTE_COLORS, V_PRED,
};
use oxideav_av1::{
    DecodedBlock, DecodedInterFrameModeInfo, DecodedPaletteMap, Error, InterFrameContext,
    MotionFieldMvs, PartitionWalker, QuantizerParams, SymbolDecoder, TileCdfContext,
    TileDecodeParams, TileGeometry, BLOCK_16X16, BLOCK_4X4, BLOCK_8X16, BLOCK_8X8,
    GM_TYPE_IDENTITY, MAX_SEGMENTS, MAX_TX_DEPTH, MAX_VARTX_DEPTH, SKIP_CONTEXTS, TX_16X16, TX_4X4,
    TX_8X8, TX_SIZE_CONTEXTS, WARPEDMODEL_PREC_BITS,
};

/// Helper for r173 `decode_inter_frame_mode_info` tests: builds the
/// §5.9.24 identity-default `gm_params` table
/// (`gm_params[ref][2] = gm_params[ref][5] = 1 <<
/// WARPEDMODEL_PREC_BITS`, other slots = 0). Pair with
/// `[GM_TYPE_IDENTITY; 8]` for the §7.10.2.1 identity short-circuit.
fn identity_gm_params() -> [[i32; 6]; 8] {
    let mut params = [[0i32; 6]; 8];
    for p in &mut params {
        p[2] = 1 << WARPEDMODEL_PREC_BITS;
        p[5] = 1 << WARPEDMODEL_PREC_BITS;
    }
    params
}

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
///   5. Run §5.11.33 `compute_prediction()` (intra-DC dispatcher;
///      emits one [`oxideav_av1::PlanePredictionTask`] per visited
///      plane). Post-r180 advances past §5.11.30.
///   6. Short-circuit at §5.11.34 `residual()`.
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
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        /* frame_is_intra = */ true,
        /* subsampling_x = */ 0,
        /* subsampling_y = */ 0,
        /* num_planes = */ 3,
        /* seg_id_pre_skip = */ false,
        /* segmentation_enabled = */ false,
        /* seg_skip_active = */ false,
        /* last_active_seg_id = */ 0,
        &lossless,
        /* coded_lossless = */ false,
        /* enable_cdef = */ true,
        /* allow_intrabc = */ false,
        /* cdef_bits = */ 0,
        /* read_deltas = */ false,
        /* use_128x128_superblock = */ false,
        /* delta_q_res = */ 0,
        /* delta_lf_present = */ false,
        /* delta_lf_multi = */ false,
        /* mono_chrome = */ false,
        /* delta_lf_res = */ 0,
        /* allow_screen_content_tools = */ false,
        /* enable_filter_intra = */ false,
        /* bit_depth = */ 8,
        /* tx_mode_select = */ false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    let pos_after = dec.position();

    // r182: the §5.11.34 dispatcher advances cleanly past §5.11.35
    // `reconstruct()` — the §7.13 inverse transform is now wired
    // through [`oxideav_av1::inverse_transform_2d`] and produces a
    // `Residual[][]` buffer for every TU with `eob > 0`. The walker
    // returns `Ok(DecodedBlock)` on this path.
    let block = result.expect(
        "post-r182 the §5.11.5 walker must return Ok(DecodedBlock) after running §5.11.34 residual() + per-TU §5.11.39 coefficients() reads + §7.13 inverse_transform_2d",
    );

    // r325: this non-palette intra block read no §5.11.49
    // `palette_tokens()` (`PaletteSize{Y,UV} == 0`), so the surfaced
    // colour-index maps must both be absent.
    assert_eq!(block.palette_size_y, 0);
    assert_eq!(block.palette_size_uv, 0);
    assert!(
        block.color_map_y.is_none(),
        "no luma palette ⇒ ColorMapY must be None"
    );
    assert!(
        block.color_map_uv.is_none(),
        "no chroma palette ⇒ ColorMapUV must be None"
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
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_4X4,
            true,
            /* subsampling_x = */ 0,
            /* subsampling_y = */ 1,
            /* num_planes = */ 3,
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
            false,
            false,
            8,
            false,
            /* inter_ctx = */ None,
            /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        );
        // r182: §5.11.34 cleanly returns Ok after §7.13 inverse transform.
        assert!(result.is_ok(), "arm 1 must return Ok(DecodedBlock)");
    }

    // Arm 2: bw4 == 1 && subsampling_x && MiCol & 1 == 0 ⇒ HasChroma = false.
    {
        let mut walker = walker_n(4);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
        let bytes = [0u8; 8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
        let result = walker.decode_block_syntax(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_4X4,
            true,
            /* subsampling_x = */ 1,
            /* subsampling_y = */ 0,
            /* num_planes = */ 3,
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
            false,
            false,
            8,
            false,
            /* inter_ctx = */ None,
            /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        );
        assert!(result.is_ok(), "arm 2 must return Ok(DecodedBlock)");
    }

    // Arm 3 fall-through: no sub-sampling ⇒ HasChroma = num_planes > 1.
    {
        let mut walker = walker_n(4);
        let mut cdfs = TileCdfContext::new_from_defaults();
        cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
        let bytes = [0u8; 8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
        let result = walker.decode_block_syntax(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_4X4,
            true,
            0,
            0,
            /* num_planes = */ 1,
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
            true,
            0,
            false,
            false,
            8,
            false,
            /* inter_ctx = */ None,
            /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        );
        assert!(result.is_ok(), "arm 3 must return Ok(DecodedBlock)");
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
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        /* frame_is_intra = */ false,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    let pos_after = dec.position();

    assert_eq!(
        result,
        Err(Error::DecodeBlockInterFrameUnsupported),
        "§5.11.6 inter-frame arm with `inter_ctx = None` keeps the historical §5.11.18 stub"
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
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        true,
        0,
        0,
        3,
        /* seg_id_pre_skip = */ true,
        /* segmentation_enabled = */ true,
        false,
        /* last_active_seg_id = */ 7,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );

    // r182: §5.11.34 returns Ok after §7.13 inverse transform fires.
    assert!(result.is_ok(), "pre-skip arm must reach Ok(DecodedBlock)");
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
        &mut dec,
        &mut cdfs,
        8,
        0,
        BLOCK_8X8,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));

    // Out-of-range mi_col.
    let r = walker.decode_block_syntax(
        &mut dec,
        &mut cdfs,
        0,
        8,
        BLOCK_8X8,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));

    // Out-of-range sub_size.
    let r = walker.decode_block_syntax(
        &mut dec,
        &mut cdfs,
        0,
        0,
        /* sub_size = */ 999,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
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
        &mut dec,
        &mut cdfs,
        /* r = */ 0,
        /* c = */ 0,
        /* b_size = */ BLOCK_4X4,
        /* frame_is_intra = */ true,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );

    // r182: §5.11.34 returns Ok after §7.13 inverse transform fires.
    assert_eq!(
        result,
        Ok(()),
        "the partition driver must propagate the §5.11.34 Ok-return from its leaf call"
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
        &mut dec,
        &mut cdfs,
        /* r = */ 4,
        /* c = */ 0,
        BLOCK_8X8,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
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

/// r342: `decode_tile_syntax` drives the §5.11.2 superblock loop across
/// a whole tile, reconstructing every leaf's intra prediction + residual
/// into `CurrFrame[plane]`. On a 16×16 intra keyframe (one 64×64
/// superblock clipped to the 4×4 mi grid) with skip forced to 1 every
/// block is a no-residual intra DC_PRED block, so the loop populates all
/// three plane buffers with the §7.11.2.5 DC default (128, no
/// neighbours at the frame origin, propagated rightward / downward by
/// each block's prediction reading the reconstructed neighbours).
#[test]
fn decode_tile_syntax_reconstructs_intra_tile_into_curr_frame() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // Force §5.11.11 skip = 1 on every context so each leaf is a
    // no-residual intra block (deterministic, no coeff reads).
    cdfs.skip = [force_binary_cdf(1); SKIP_CONTEXTS];
    let bytes = [0u8; 64];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 64, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];
    let quant = QuantizerParams::neutral(0, 8);
    let params = TileDecodeParams {
        frame_is_intra: true,
        subsampling_x: 1,
        subsampling_y: 1,
        num_planes: 3,
        seg_id_pre_skip: false,
        segmentation_enabled: false,
        seg_skip_active: false,
        last_active_seg_id: 0,
        lossless_array: &lossless,
        coded_lossless: false,
        enable_cdef: false,
        allow_intrabc: false,
        cdef_bits: 0,
        use_128x128_superblock: false,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_multi: false,
        mono_chrome: false,
        delta_lf_res: 0,
        allow_screen_content_tools: false,
        enable_filter_intra: false,
        bit_depth: 8,
        tx_mode_select: false,
        reduced_tx_set: false,
    };

    let result = walker.decode_tile_syntax(
        &mut dec, &mut cdfs, &params, /* inter_ctx = */ None, &quant,
        /* read_deltas = */ false,
    );
    assert_eq!(
        result,
        Ok(()),
        "the §5.11.2 tile loop must reconstruct cleanly"
    );

    // Every plane buffer is allocated + reconstructed. The luma plane
    // is 16×16 (4 mi * MI_SIZE), chroma planes 8×8 (>> subsampling).
    assert_eq!(walker.curr_frame_dims(0), Some((16, 16)));
    assert_eq!(walker.curr_frame_dims(1), Some((8, 8)));
    assert_eq!(walker.curr_frame_dims(2), Some((8, 8)));
    // The top-left luma sample is the §7.11.2.5 no-neighbour DC default.
    let y = walker.curr_frame(0).unwrap();
    assert_eq!(
        y[0], 128,
        "top-left luma is the DC_PRED no-neighbour default"
    );
    // Every reconstructed sample is in the valid 8-bit range (the
    // §7.12.3 step-3 Clip1 envelope holds across the whole tile).
    for &s in walker.curr_frame(0).unwrap() {
        assert!((0..=255).contains(&s), "luma sample {s} in Clip1 range");
    }
}

/// Helper: walk one 16×16 intra keyframe tile (skip-forced, no
/// residual) so the walker's §7.12.3 `CurrFrame[plane]` buffers + the
/// §5.11 per-mi grids the §7.14 bridge reads are all populated. Returns
/// the walker (with grids + CurrFrame) for the deblock-bridge tests
/// below.
fn walk_flat_intra_16x16_tile() -> PartitionWalker {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(1); SKIP_CONTEXTS];
    let bytes = [0u8; 64];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 64, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];
    let quant = QuantizerParams::neutral(0, 8);
    let params = TileDecodeParams {
        frame_is_intra: true,
        subsampling_x: 1,
        subsampling_y: 1,
        num_planes: 3,
        seg_id_pre_skip: false,
        segmentation_enabled: false,
        seg_skip_active: false,
        last_active_seg_id: 0,
        lossless_array: &lossless,
        coded_lossless: false,
        enable_cdef: false,
        allow_intrabc: false,
        cdef_bits: 0,
        use_128x128_superblock: false,
        delta_q_res: 0,
        delta_lf_present: false,
        delta_lf_multi: false,
        mono_chrome: false,
        delta_lf_res: 0,
        allow_screen_content_tools: false,
        enable_filter_intra: false,
        bit_depth: 8,
        tx_mode_select: false,
        reduced_tx_set: false,
    };
    walker
        .decode_tile_syntax(&mut dec, &mut cdfs, &params, None, &quant, false)
        .expect("§5.11.2 tile loop must reconstruct cleanly");
    walker
}

/// Extract the walker's §7.12.3 `CurrFrame[plane]` buffers into owned
/// `(rows, cols, Vec<i32>)` triples the §7.14 [`oxideav_av1::loop_filter::PlaneBuffer`]
/// driver can borrow mutably.
fn extract_curr_planes(walker: &PartitionWalker) -> Vec<(u32, u32, Vec<i32>)> {
    (0..3)
        .map(|p| {
            let (rows, cols) = walker.curr_frame_dims(p).unwrap();
            (rows, cols, walker.curr_frame(p).unwrap().to_vec())
        })
        .collect()
}

/// §7.14 deblock bridge — `loop_filter_level == 0` is a frame-scope
/// identity (av1-spec p.307 line 16961 / line 11423 `If
/// loop_filter_level[0] != 0 || loop_filter_level[1] != 0`). The bridge
/// must leave every `CurrFrame[plane]` sample byte-for-byte unchanged.
#[test]
fn loop_filter_bridge_level_zero_is_identity() {
    let walker = walk_flat_intra_16x16_tile();
    let mut planes = extract_curr_planes(&walker);
    let before: Vec<Vec<i32>> = planes.iter().map(|(_, _, s)| s.clone()).collect();

    let mut bufs: Vec<oxideav_av1::loop_filter::PlaneBuffer<'_>> = planes
        .iter_mut()
        .map(|(rows, cols, s)| oxideav_av1::loop_filter::PlaneBuffer {
            rows: *rows,
            cols: *cols,
            samples: s.as_mut_slice(),
        })
        .collect();

    let lf = oxideav_av1::uncompressed_header_tail::LoopFilterParams::short_circuit();
    let seg = oxideav_av1::uncompressed_header_tail::SegmentationParams::disabled();
    walker.loop_filter_frame_from_grid(
        &lf, &seg, /* delta_lf_present = */ false, 3, 8, 1, 1, 16, 16, &mut bufs,
    );

    for (p, (_, _, s)) in planes.iter().enumerate() {
        assert_eq!(
            s, &before[p],
            "plane {p}: level-0 deblock must be a pure identity"
        );
    }
}

/// §7.14 deblock bridge on a flat (DC = 128 everywhere) frame — a
/// non-zero `loop_filter_level` still produces no sample change because
/// the §7.14.6 filter mask never fires when every neighbour pair is
/// equal (the §7.14.6 `filterMask` derivation reads zero gradient).
/// This exercises the full §7.14.1 `plane × pass × row × col` edge
/// walk + the §7.14.4/§7.14.5 strength derivation off the walker's
/// real grids, and proves the bridge keeps the flat field invariant.
#[test]
fn loop_filter_bridge_flat_frame_invariant_under_nonzero_level() {
    let walker = walk_flat_intra_16x16_tile();
    // Confirm the precondition: the reconstructed tile is flat DC.
    for &s in walker.curr_frame(0).unwrap() {
        assert_eq!(
            s, 128,
            "flat-tile precondition: every luma sample is DC 128"
        );
    }
    let mut planes = extract_curr_planes(&walker);
    let before: Vec<Vec<i32>> = planes.iter().map(|(_, _, s)| s.clone()).collect();

    let mut bufs: Vec<oxideav_av1::loop_filter::PlaneBuffer<'_>> = planes
        .iter_mut()
        .map(|(rows, cols, s)| oxideav_av1::loop_filter::PlaneBuffer {
            rows: *rows,
            cols: *cols,
            samples: s.as_mut_slice(),
        })
        .collect();

    // A real, in-range §5.9.11 strength schedule (luma V/H + both
    // chroma planes at level 32, sharpness 1).
    let lf = oxideav_av1::uncompressed_header_tail::LoopFilterParams {
        loop_filter_level: [32, 32, 32, 32],
        loop_filter_sharpness: 1,
        loop_filter_delta_enabled: false,
        loop_filter_delta_update: false,
        loop_filter_ref_deltas:
            oxideav_av1::uncompressed_header_tail::LOOP_FILTER_REF_DELTAS_DEFAULT,
        loop_filter_mode_deltas:
            oxideav_av1::uncompressed_header_tail::LOOP_FILTER_MODE_DELTAS_DEFAULT,
        short_circuited: false,
    };
    let seg = oxideav_av1::uncompressed_header_tail::SegmentationParams::disabled();
    walker.loop_filter_frame_from_grid(&lf, &seg, false, 3, 8, 1, 1, 16, 16, &mut bufs);

    for (p, (_, _, s)) in planes.iter().enumerate() {
        assert_eq!(
            s, &before[p],
            "plane {p}: a flat field has no §7.14.6 edge gradient, so deblock is invariant"
        );
        for &v in s {
            assert!(
                (0..=255).contains(&v),
                "plane {p} sample {v} stays in Clip1 range"
            );
        }
    }
}

/// §7.14 deblock bridge `delta_lf_present == 1` guard — the walker does
/// not persist a per-mi `DeltaLFs[][][]` snapshot (only the running
/// §5.11.13 accumulator), so the bridge conservatively returns the
/// pre-deblock `CurrFrame[plane]` unchanged rather than apply a wrong
/// §7.14.4 strength. Verify it is a no-op even with a non-zero level.
#[test]
fn loop_filter_bridge_delta_lf_present_is_noop_guard() {
    let walker = walk_flat_intra_16x16_tile();
    let mut planes = extract_curr_planes(&walker);
    let before: Vec<Vec<i32>> = planes.iter().map(|(_, _, s)| s.clone()).collect();

    let mut bufs: Vec<oxideav_av1::loop_filter::PlaneBuffer<'_>> = planes
        .iter_mut()
        .map(|(rows, cols, s)| oxideav_av1::loop_filter::PlaneBuffer {
            rows: *rows,
            cols: *cols,
            samples: s.as_mut_slice(),
        })
        .collect();

    let lf = oxideav_av1::uncompressed_header_tail::LoopFilterParams {
        loop_filter_level: [32, 32, 32, 32],
        ..oxideav_av1::uncompressed_header_tail::LoopFilterParams::short_circuit()
    };
    let seg = oxideav_av1::uncompressed_header_tail::SegmentationParams::disabled();
    walker.loop_filter_frame_from_grid(
        &lf, &seg, /* delta_lf_present = */ true, 3, 8, 1, 1, 16, 16, &mut bufs,
    );

    for (p, (_, _, s)) in planes.iter().enumerate() {
        assert_eq!(
            s, &before[p],
            "plane {p}: delta_lf_present guard must leave planes untouched"
        );
    }
}

/// `DecodedBlock` is the per-block aggregate returned on the no-stub
/// path. The struct's constructibility check is a sanity-check on the
/// public API surface — it should compile and be `Debug + Clone +
/// PartialEq + Eq` (r325 dropped `Copy` when the §5.11.49
/// colour-index maps were surfaced as heap-allocated fields).
#[test]
fn decoded_block_struct_public_api_smoke() {
    let db = DecodedBlock {
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
        uv_mode: None,
        angle_delta_y: 0,
        angle_delta_uv: None,
        cfl_alpha_u: None,
        cfl_alpha_v: None,
        palette_size_y: 0,
        palette_size_uv: 0,
        use_filter_intra: None,
        filter_intra_mode: None,
        mv: [[0, 0], [0, 0]],
        is_compound: false,
        is_inter_intra: false,
        tx_size: TX_4X4 as u8,
        // §5.11.49 colour-index maps — `None` on a non-palette block.
        color_map_y: None,
        color_map_uv: None,
    };
    // r325: `Clone` (no longer `Copy`) + `PartialEq` round-trip.
    let cloned = db.clone();
    assert_eq!(db, cloned);
    assert!(db.color_map_y.is_none());
    assert!(db.color_map_uv.is_none());
}

/// `DecodedPaletteMap` is the §5.11.49 colour-index aggregate the
/// §5.11.5 walker surfaces on a palette block (r325). The struct must
/// be constructible from the public API and round-trip through `Clone`
/// / `PartialEq`; its geometry invariants (`data.len() == block_w *
/// block_h`, `stride == block_w`, on-screen ⊆ full block) mirror the
/// [`palette_tokens_args`] dimensions the reader consumed.
#[test]
fn decoded_palette_map_public_api_smoke() {
    // A fully-on-screen 8×8 luma palette block (the §5.11.46 minimum
    // palette-eligible size): block == on-screen, stride == width.
    let map = DecodedPaletteMap {
        data: vec![0u8; 8 * 8],
        stride: 8,
        block_w: 8,
        block_h: 8,
        onscreen_w: 8,
        onscreen_h: 8,
    };
    assert_eq!(map.data.len(), map.block_w * map.block_h);
    assert_eq!(map.stride, map.block_w);
    assert!(map.onscreen_w <= map.block_w && map.onscreen_h <= map.block_h);
    assert_eq!(map, map.clone());
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
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X16,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    // r182: §5.11.34 returns Ok after §7.13 inverse transform fires.
    assert!(
        result.is_ok(),
        "BLOCK_8X16 walker must return Ok(DecodedBlock)"
    );

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
        false,
        false,
        8,
        /* tx_mode_select = */ false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    let pos_after = dec.position();
    // r182: §5.11.34 returns Ok after §7.13 inverse transform fires.
    assert!(
        result.is_ok(),
        "cdef_bits walker must return Ok(DecodedBlock)"
    );
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

/// §5.11.16 inter-arm: `TX_MODE_SELECT && MiSize > BLOCK_4X4 &&
/// is_inter && !skip && !Lossless` enters the §5.11.17
/// `read_var_tx_size` recursion (landed in r168). With both
/// `txfm_split_cdf` rows rigged to symbol 0 (no split), the recursion
/// bottoms out at depth 0 with `txfm_split = 0`, the terminal else
/// arm stamps `InterTxSizes[]` over the `(h4, w4)` sub-block, and
/// returns the input `txSz`. For BLOCK_16X16 with maxTxSz =
/// TX_16X16, txW4 = txH4 = 4, the outer loop fires once, and the
/// reader returns TX_16X16.
#[test]
fn read_block_tx_size_inter_arm_no_split_returns_max_tx_size() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // Rig txfm_split CDF rows to all return symbol 0 (no split).
    let rigged = force_binary_cdf(0);
    for row in cdfs.txfm_split.iter_mut() {
        *row = rigged;
    }
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
            /* skip = */ false,
            /* tx_mode_select = */ true,
        )
        .unwrap();
    let pos_after = dec.position();
    assert_eq!(
        tx, TX_16X16 as u8,
        "no-split: read_var_tx_size returns the input txSz = max_tx_sz"
    );
    assert!(
        pos_after > pos_before,
        "the §5.11.17 reader must consume the txfm_split bit"
    );
    // Verify the §5.11.17 terminal-else stamp landed over the BLOCK_16X16
    // 4×4 footprint of `InterTxSizes[]`.
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(
                walker.inter_tx_sizes()[r * mi_cols + c],
                TX_16X16 as u8,
                "§5.11.17 terminal-else stamp at ({r}, {c})"
            );
        }
    }
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
        false,
        false,
        8,
        /* tx_mode_select = */ true,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    // r182: §5.11.34 returns Ok after §7.13 inverse transform fires.
    assert!(
        result.is_ok(),
        "TX_MODE_SELECT walker must return Ok(DecodedBlock)"
    );

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

// ===================================================================
// §5.11.17 read_var_tx_size — direct-call coverage.
//
// The §5.11.17 reader was implemented in r168. It implements the
// variable-transform-tree syntax recursion the §5.11.16 inter-arm
// enters when `TX_MODE_SELECT && MiSize > BLOCK_4X4 && is_inter &&
// !skip && !Lossless`. The recursion bottoms out at `txSz == TX_4X4`
// or `depth == MAX_VARTX_DEPTH`, each terminal-else stamps
// `InterTxSizes[]` over the `(h4, w4)` sub-block footprint.
// ===================================================================

/// §5.11.17 base case: `txSz == TX_4X4` short-circuits the
/// `txfm_split` read (the spec body sets `txfm_split = 0` directly,
/// no S() read) and stamps `InterTxSizes[]` at the anchor cell.
#[test]
fn read_var_tx_size_tx_4x4_no_read() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0xFFu8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    let tx = walker
        .read_var_tx_size(
            &mut dec, &mut cdfs, /* mi_row_b = */ 0, /* mi_col_b = */ 0, BLOCK_8X8,
            /* row = */ 0, /* col = */ 0, TX_4X4, /* depth = */ 0,
            /* is_inter = */ true,
        )
        .unwrap();
    let pos_after = dec.position();
    assert_eq!(tx, TX_4X4 as u8);
    assert_eq!(
        pos_after, pos_before,
        "TX_4X4 base case must not consume bits"
    );
    assert_eq!(
        walker.inter_tx_sizes()[0],
        TX_4X4 as u8,
        "terminal-else stamp at (0, 0)"
    );
}

/// §5.11.17 depth cap: `depth == MAX_VARTX_DEPTH` forces
/// `txfm_split = 0` (no S() read), stamps `InterTxSizes[]` and
/// returns the input `txSz`.
#[test]
fn read_var_tx_size_max_depth_no_read() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0xFFu8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    let tx = walker
        .read_var_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            0,
            0,
            TX_16X16,
            /* depth = */ MAX_VARTX_DEPTH,
            true,
        )
        .unwrap();
    let pos_after = dec.position();
    assert_eq!(tx, TX_16X16 as u8);
    assert_eq!(pos_after, pos_before, "depth cap must not consume bits");
    // §5.11.17 terminal-else stamp covers the TX_16X16 footprint (4×4 mi cells).
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(
                walker.inter_tx_sizes()[r * mi_cols + c],
                TX_16X16 as u8,
                "§5.11.17 stamp at ({r}, {c})"
            );
        }
    }
}

/// §5.11.17 split path: forcing `txfm_split = 1` enters the
/// recursion which calls itself with `subTxSz = Split_Tx_Size[ txSz ]`
/// and `depth + 1`. With both txfm_split CDF rows forcing symbol 1,
/// the recursion descends to `MAX_VARTX_DEPTH = 2` then stamps the
/// sub-block footprint. For BLOCK_16X16 with `tx_sz = TX_16X16` at
/// depth 0: TX_16X16 → split → 4× TX_8X8 → each split → 4× TX_4X4
/// (depth 2, the cap), each stamps 1×1 mi cells.
#[test]
fn read_var_tx_size_split_to_max_depth_stamps_tx_4x4() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    for row in cdfs.txfm_split.iter_mut() {
        *row = force_binary_cdf(1);
    }
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let pos_before = dec.position();
    let tx = walker
        .read_var_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            0,
            0,
            TX_16X16,
            0,
            true,
        )
        .unwrap();
    let _ = pos_before;
    // Terminal `txSz` returned. At depth 2 the recursion stops with
    // the spec's `txfm_split = 0` short-circuit and stamps TX_4X4.
    assert_eq!(
        tx, TX_4X4 as u8,
        "last terminal-else assigns txSz = TX_4X4 at depth MAX_VARTX_DEPTH"
    );
    // The §5.11.17 stamp covers every 1×1 mi cell of the TX_16X16 footprint.
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(
                walker.inter_tx_sizes()[r * mi_cols + c],
                TX_4X4 as u8,
                "§5.11.17 deepest-split stamp at ({r}, {c})"
            );
        }
    }
}

/// §5.11.17 frame-edge clip: `row >= MiRows || col >= MiCols`
/// short-circuits with no read, no stamp, returning the input `txSz`.
#[test]
fn read_var_tx_size_out_of_frame_short_circuits() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0xFFu8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    let pos_before = dec.position();
    // row >= MiRows.
    let tx = walker
        .read_var_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            /* row = */ 4,
            0,
            TX_16X16,
            0,
            true,
        )
        .unwrap();
    assert_eq!(tx, TX_16X16 as u8);
    assert_eq!(dec.position(), pos_before);
    // col >= MiCols.
    let tx = walker
        .read_var_tx_size(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_16X16,
            0,
            /* col = */ 4,
            TX_16X16,
            0,
            true,
        )
        .unwrap();
    assert_eq!(tx, TX_16X16 as u8);
    assert_eq!(dec.position(), pos_before);
}

/// §5.11.17 caller-bug detection: out-of-range arguments return
/// `PartitionWalkOutOfRange` before any bit is read.
#[test]
fn read_var_tx_size_rejects_out_of_range() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 4];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
    // out-of-range sub_size.
    let r = walker.read_var_tx_size(&mut dec, &mut cdfs, 0, 0, 999, 0, 0, TX_16X16, 0, true);
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    // out-of-range tx_sz.
    let r = walker.read_var_tx_size(&mut dec, &mut cdfs, 0, 0, BLOCK_16X16, 0, 0, 999, 0, true);
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    // out-of-range depth.
    let r = walker.read_var_tx_size(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_16X16,
        0,
        0,
        TX_16X16,
        MAX_VARTX_DEPTH + 1,
        true,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    // out-of-range mi_row_b.
    let r = walker.read_var_tx_size(
        &mut dec,
        &mut cdfs,
        4,
        0,
        BLOCK_16X16,
        0,
        0,
        TX_16X16,
        0,
        true,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
}

// ===================================================================
// §5.11.18 decode_inter_frame_mode_info — direct-call coverage.
//
// The §5.11.18 reader was implemented in r168. It composes every
// pre-dispatch leaf (`inter_segment_id`, `read_skip_mode`,
// `read_skip`, `read_cdef`, `read_delta_qindex`, `read_delta_lf`,
// `read_is_inter`) and short-circuits at the terminal
// `if ( is_inter )` dispatch into §5.11.22 / §5.11.23 (both
// next-round targets).
// ===================================================================

/// Baseline §5.11.18 path: segmentation off, skip_mode_present off,
/// allow_intrabc off, cdef_bits = 0, read_deltas off, no segmentation
/// overrides. With the skip CDF rigged to symbol 0 and the is_inter
/// CDF rigged to symbol 0, the walker reaches the §5.11.22 intra
/// stub.
#[test]
fn decode_inter_frame_mode_info_reaches_intra_block_stub() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // Skip CDF forces symbol 0 (skip = 0).
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    // is_inter CDF forces symbol 0 (is_inter = 0 → intra arm).
    for row in cdfs.is_inter.iter_mut() {
        *row = force_binary_cdf(0);
    }
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let pos_before = dec.position();
    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let result = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        /* mi_row = */ 0,
        /* mi_col = */ 0,
        BLOCK_8X8,
        /* seg_id_pre_skip = */ false,
        /* segmentation_enabled = */ false,
        /* segmentation_update_map = */ false,
        /* segmentation_temporal_update = */ false,
        /* predicted_segment_id = */ 0,
        /* last_active_seg_id = */ 0,
        &lossless,
        /* seg_skip_mode_off = */ false,
        /* seg_ref_frame_active = */ false,
        /* seg_ref_frame_is_inter = */ false,
        /* seg_globalmv_active = */ false,
        /* skip_mode_present = */ false,
        /* coded_lossless = */ false,
        /* enable_cdef = */ true,
        /* allow_intrabc = */ false,
        /* cdef_bits = */ 0,
        /* read_deltas = */ false,
        /* use_128x128_superblock = */ false,
        /* delta_q_res = */ 0,
        /* delta_lf_present = */ false,
        /* delta_lf_multi = */ false,
        /* mono_chrome = */ false,
        /* delta_lf_res = */ 0,
        // r170 §5.11.25 args — the intra arm ignores them.
        /* skip_mode_frame = */
        [0, -1],
        /* seg_skip_active = */ false,
        /* seg_ref_frame_data = */ 0,
        /* reference_select = */ false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    let pos_after = dec.position();
    assert_eq!(
        result,
        Err(Error::IntraBlockModeInfoUnsupported),
        "is_inter = 0 ⇒ §5.11.22 intra_block_mode_info stub"
    );
    assert!(
        pos_after > pos_before,
        "the §5.11.18 walker must consume at least the skip + is_inter S() bits"
    );
    // §5.11.18 grids stamped: Skips[0..2][0..2] = 0, IsInters[0..2][0..2] = 0.
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(walker.skips()[r * mi_cols + c], 0);
            assert_eq!(walker.is_inters()[r * mi_cols + c], 0);
        }
    }
}

/// §5.11.18 inter arm via §5.11.20 segment-override: with
/// `seg_ref_frame_active = true` and `seg_ref_frame_is_inter = true`
/// the §5.11.20 arm 2 forces `is_inter = 1` without S() — driving
/// the §5.11.18 terminal dispatch into the §5.11.23
/// `InterBlockModeInfoUnsupported` stub. The walker stamps
/// `IsInters[][] = 1` over the footprint before the stub fires.
#[test]
fn decode_inter_frame_mode_info_reaches_inter_block_stub() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let result = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        false,
        false,
        false,
        false,
        0,
        0,
        &lossless,
        false,
        /* seg_ref_frame_active = */ true,
        /* seg_ref_frame_is_inter = */ true,
        false,
        false,
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
        // r170 §5.11.25 args. seg_ref_frame_active = true puts the
        // §5.11.25 reader in the "RefFrame[0] = seg_ref_frame_data,
        // RefFrame[1] = NONE" arm — no S() reads — so the §5.11.23
        // reader then immediately hits the FindMvStackUnsupported
        // stub at the §7.10 entry. Pass seg_ref_frame_data = 1
        // (LAST_FRAME) so the value is a conformant ref.
        /* skip_mode_frame = */
        [0, -1],
        /* seg_skip_active = */ false,
        /* seg_ref_frame_data = */ 1,
        /* reference_select = */ false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    // r190 — the §5.11.18 dispatcher's `Ok(_)` arm now lifts the
    // historical `InterBlockModeInfoUnsupported` stub and surfaces
    // the §5.11.23 inter-block aggregate through
    // `DecodedInterFrameModeInfo::inter_block`. With
    // `seg_ref_frame_active = true` + `is_inter = 1` the §5.11.23
    // cascade runs to completion (`read_ref_frames` /
    // `find_mv_stack` / `assign_mv` / `read_motion_mode` /
    // `read_interintra_mode` / `read_compound_type` /
    // `read_interpolation_filter`).
    let info = result.expect("§5.11.18 inter cascade should now complete (r190 wire-up)");
    assert_eq!(info.is_inter, 1);
    let inter = info
        .inter_block
        .expect("§5.11.18 `is_inter == 1` ⇒ inter_block populated");
    // §5.11.25 single-ref arm — RefFrame[0] = seg_ref_frame_data,
    // RefFrame[1] = NONE.
    assert_eq!(inter.ref_frame, [1, -1]);
    assert!(!inter.is_compound);
    // r170: §5.11.25 stamps RefFrames[0..2][0..2][0..2] over the
    // BLOCK_8X8 footprint = [LAST_FRAME = 1, NONE = -1].
    let mi_cols_grid = walker.mi_cols() as usize;
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(walker.ref_frames()[(r * mi_cols_grid + c) * 2], 1);
            assert_eq!(walker.ref_frames()[(r * mi_cols_grid + c) * 2 + 1], -1);
        }
    }
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..2 {
        for c in 0..2 {
            assert_eq!(walker.is_inters()[r * mi_cols + c], 1);
        }
    }
}

/// §5.11.18 lines 12-14: with `skip_mode_present = true` and a CDF
/// forcing `skip_mode = 1`, the walker takes the `skip_mode → skip = 1`
/// shortcut (no read_skip S()), stamps `Skips[][] = 1`, and the
/// §5.11.20 read_is_inter consumes the `skip_mode → is_inter = 1`
/// first arm (no S() either). The §5.11.23 stub fires.
#[test]
fn decode_inter_frame_mode_info_skip_mode_forces_skip_and_inter() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // Skip-mode CDF forces symbol 1 (skip_mode = 1).
    for row in cdfs.skip_mode.iter_mut() {
        *row = force_binary_cdf(1);
    }
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let result = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_16X16, // ≥ 8x8 — small-block short-circuit doesn't fire
        false,
        false,
        false,
        false,
        0,
        0,
        &lossless,
        /* seg_skip_mode_off = */ false,
        false,
        false,
        false,
        /* skip_mode_present = */ true,
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
        // r170 §5.11.25 args. skip_mode = 1 puts the §5.11.25 reader
        // in the "RefFrame[0..2] = SkipModeFrame[0..2]" arm. Pass
        // [LAST_FRAME, ALTREF_FRAME] so the result is a conformant
        // compound pair (so the post-stamp grid view is non-trivial).
        /* skip_mode_frame = */
        [1, 7],
        /* seg_skip_active = */ false,
        /* seg_ref_frame_data = */ 0,
        /* reference_select = */ false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    // r190: the inter cascade now runs to completion through the
    // §5.11.18 dispatcher's Ok-arm; the §5.11.23 aggregate is
    // surfaced via `DecodedInterFrameModeInfo::inter_block`.
    let info = result.expect("§5.11.18 inter cascade with skip_mode = 1 must complete");
    assert_eq!(info.skip, 1, "skip_mode → skip = 1");
    assert_eq!(info.skip_mode, 1);
    assert_eq!(info.is_inter, 1, "skip_mode → is_inter = 1");
    let inter = info
        .inter_block
        .expect("§5.11.18 `is_inter == 1` ⇒ inter_block populated");
    // §5.11.25 skip-mode arm — RefFrame[0..2] = SkipModeFrame[0..2].
    assert_eq!(inter.ref_frame, [1, 7]);
    assert!(inter.is_compound, "[LAST_FRAME, ALTREF_FRAME] ⇒ compound");
    // §5.11.18 grid-fill: Skips[][] stamped to 1 over the 4×4 footprint.
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..4 {
        for c in 0..4 {
            assert_eq!(walker.skips()[r * mi_cols + c], 1);
            assert_eq!(walker.skip_modes()[r * mi_cols + c], 1);
            assert_eq!(walker.is_inters()[r * mi_cols + c], 1);
            // r170: RefFrames[][] stamped to [1, 7].
            assert_eq!(walker.ref_frames()[(r * mi_cols + c) * 2], 1);
            assert_eq!(walker.ref_frames()[(r * mi_cols + c) * 2 + 1], 7);
        }
    }
}

/// §5.11.18 segmentation override: with `seg_globalmv_active = true`
/// the §5.11.20 `read_is_inter` third arm fires (forces is_inter = 1
/// without S() read).
#[test]
fn decode_inter_frame_mode_info_seg_globalmv_forces_inter() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    let bytes = [0u8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let result = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        false,
        false,
        false,
        false,
        0,
        0,
        &lossless,
        false,
        false,
        false,
        /* seg_globalmv_active = */ true,
        false,
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
        // r170: seg_globalmv_active = true puts §5.11.25 in the
        // "RefFrame[0] = LAST_FRAME, RefFrame[1] = NONE" arm — no
        // S() — then §7.10 stub.
        /* skip_mode_frame = */
        [0, -1],
        /* seg_skip_active = */ false,
        /* seg_ref_frame_data = */ 0,
        /* reference_select = */ false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    // r190: the inter cascade now runs to completion. The §5.11.20
    // `read_is_inter` third arm fires (`seg_globalmv_active = true`
    // ⇒ is_inter = 1 without S()).
    let info = result.expect("§5.11.18 inter cascade with seg_globalmv override must complete");
    assert_eq!(info.is_inter, 1);
    let inter = info
        .inter_block
        .expect("§5.11.18 `is_inter == 1` ⇒ inter_block populated");
    assert_eq!(
        inter.ref_frame[0], 1,
        "§5.11.25 globalmv arm sets RefFrame[0] = LAST_FRAME"
    );
    assert_eq!(
        inter.ref_frame[1], -1,
        "§5.11.25 globalmv arm leaves RefFrame[1] = NONE"
    );
}

/// §5.11.18 caller-bug detection: out-of-range arguments surface
/// `PartitionWalkOutOfRange` before any bit is read.
#[test]
fn decode_inter_frame_mode_info_rejects_out_of_range() {
    let mut walker = walker_n(8);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0u8; 8];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 8, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    // Out-of-range mi_row.
    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let r = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        8,
        0,
        BLOCK_8X8,
        false,
        false,
        false,
        false,
        0,
        0,
        &lossless,
        false,
        false,
        false,
        false,
        false,
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
        [0, -1],
        false,
        0,
        false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    // Out-of-range mi_col.
    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let r = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        0,
        8,
        BLOCK_8X8,
        false,
        false,
        false,
        false,
        0,
        0,
        &lossless,
        false,
        false,
        false,
        false,
        false,
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
        [0, -1],
        false,
        0,
        false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    // Out-of-range sub_size.
    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let r = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        0,
        0,
        999,
        false,
        false,
        false,
        false,
        0,
        0,
        &lossless,
        false,
        false,
        false,
        false,
        false,
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
        [0, -1],
        false,
        0,
        false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
    // Out-of-range last_active_seg_id.
    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let r = walker.decode_inter_frame_mode_info(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        false,
        false,
        false,
        false,
        0,
        /* last_active_seg_id = */ MAX_SEGMENTS as u8,
        &lossless,
        false,
        false,
        false,
        false,
        false,
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
        [0, -1],
        false,
        0,
        false,
        /* gm_type = */ [GM_TYPE_IDENTITY; 8],
        /* gm_params = */ identity_gm_params(),
        /* ref_frame_sign_bias = */ [0; 8],
        /* allow_high_precision_mv = */ false,
        /* force_integer_mv = */ false,
        /* use_ref_frame_mvs = */ false,
        /* is_motion_mode_switchable = */ false,
        /* allow_warped_motion = */ false,
        /* is_scaled_per_ref = */ [false; 7],
        /* enable_interintra_compound = */ false,
        /* enable_masked_compound = */ false,
        /* enable_jnt_comp = */ false,
        /* dist_equal = */ false,
        /* interpolation_filter = */ 0,
        /* enable_dual_filter = */ false,
        &mfmvs,
    );
    assert_eq!(r, Err(Error::PartitionWalkOutOfRange));
}

/// `DecodedInterFrameModeInfo` smoke: the struct is publicly
/// constructible and matches `Debug + Clone + Copy + PartialEq + Eq`.
#[test]
fn decoded_inter_frame_mode_info_struct_public_api_smoke() {
    let info = DecodedInterFrameModeInfo {
        mi_row: 0,
        mi_col: 0,
        mi_size: BLOCK_8X8,
        use_intrabc: 0,
        avail_u: false,
        avail_l: false,
        left_ref_frame: [0, -1],
        above_ref_frame: [0, -1],
        left_intra: true,
        above_intra: true,
        left_single: true,
        above_single: true,
        skip: 0,
        skip_mode: 0,
        segment_id: 0,
        lossless: false,
        cdef_idx: -1,
        current_q_index: 0,
        current_delta_lf: [0; 4],
        is_inter: 0,
        inter_block: None,
    };
    let copy = info;
    assert_eq!(info, copy);
    let _ = format!("{info:?}");
}

// ----------------------------------------------------------------
// Round 180 — §5.11.33 `compute_prediction()` dispatcher tests +
// §7.11.2.5 DC_PRED standalone leaf tests.
// ----------------------------------------------------------------

use oxideav_av1::{
    get_plane_residual_size, predict_intra_dc_pred, ComputePredictionReadout, PlanePredictionTask,
    BLOCK_16X8, BLOCK_4X8,
};

/// §5.11.38 `get_plane_residual_size` table sanity — plane 0 is a
/// pass-through (luma never sub-samples), and a few §3 spot-checks for
/// the (subsampling=1,1) chroma down-shift on common sub-blocks.
#[test]
fn get_plane_residual_size_table_spot_checks() {
    // Plane 0 (luma) is a pass-through for every subsampling.
    for bs in 0..22 {
        for sx in 0..=1 {
            for sy in 0..=1 {
                assert_eq!(
                    get_plane_residual_size(bs, 0, sx, sy),
                    Some(bs),
                    "plane 0 must pass through bs={bs}",
                );
            }
        }
    }
    // BLOCK_8X8 with full 420 subsampling → BLOCK_4X4 chroma residual.
    assert_eq!(get_plane_residual_size(BLOCK_8X8, 1, 1, 1), Some(BLOCK_4X4));
    // BLOCK_16X16 with 420 → BLOCK_8X8 chroma residual.
    assert_eq!(
        get_plane_residual_size(BLOCK_16X16, 1, 1, 1),
        Some(BLOCK_8X8)
    );
    // BLOCK_8X16 with 422 (subx=1, suby=0) → §3 forbids the chroma
    // residual (returns None / BLOCK_INVALID).
    assert_eq!(get_plane_residual_size(BLOCK_8X16, 1, 1, 0), None);
    // BLOCK_8X16 with 420 (1,1) → BLOCK_4X8 chroma residual.
    assert_eq!(
        get_plane_residual_size(BLOCK_8X16, 1, 1, 1),
        Some(BLOCK_4X8)
    );
    // BLOCK_16X8 with 422 (subx=1, suby=0) → BLOCK_8X8 chroma.
    assert_eq!(
        get_plane_residual_size(BLOCK_16X8, 1, 1, 0),
        Some(BLOCK_8X8)
    );
}

/// Out-of-range guards on `get_plane_residual_size`.
#[test]
fn get_plane_residual_size_guards() {
    assert_eq!(get_plane_residual_size(22, 0, 0, 0), None);
    assert_eq!(get_plane_residual_size(100, 1, 1, 1), None);
}

/// §5.11.33 `compute_prediction` on the §5.11.5-reachable intra path:
/// `is_inter == 0`, `y_mode == DC_PRED`, `has_chroma == true`. Emits
/// one DC_PRED task per plane (plane 0/1/2) on a `BLOCK_8X8` block
/// with 420 subsampling (chroma plane residual is `BLOCK_4X4`).
#[test]
fn compute_prediction_intra_dc_pred_3_plane_path() {
    let mut walker = walker_n(16);
    let result = walker.compute_prediction(
        /* mi_row = */ 0, /* mi_col = */ 0, /* mi_size = */ BLOCK_8X8,
        /* has_chroma = */ true, /* avail_u = */ false, /* avail_l = */ false,
        /* avail_u_chroma = */ false, /* avail_l_chroma = */ false,
        /* subsampling_x = */ 1, /* subsampling_y = */ 1, /* is_inter = */ false,
        /* y_mode = */ 0, // DC_PRED
        /* uv_mode = */ 0, // DC_PRED
        /* ref_frame_1_is_intra = */ false, /* interintra_mode = */ 0,
    );
    let r = result.expect("3-plane intra DC dispatch must succeed");
    assert!(!r.is_inter);
    assert!(!r.is_inter_intra);
    assert_eq!(r.num_planes_visited, 3);
    assert_eq!(r.tasks.len(), 3);

    // Plane 0 (luma) — BLOCK_8X8, log2W=3, log2H=3, baseX=baseY=0.
    let t0 = &r.tasks[0];
    assert_eq!(t0.plane, 0);
    assert_eq!(t0.start_x, 0);
    assert_eq!(t0.start_y, 0);
    assert_eq!(t0.log2_w, 3);
    assert_eq!(t0.log2_h, 3);
    assert_eq!(t0.mode, 0);
    assert!(!t0.have_above);
    assert!(!t0.have_left);

    // Plane 1 (Cb) — chroma residual is BLOCK_4X4, log2W=log2H=2.
    let t1 = &r.tasks[1];
    assert_eq!(t1.plane, 1);
    assert_eq!(t1.log2_w, 2);
    assert_eq!(t1.log2_h, 2);
    // baseX = (0 >> 1) * 4 = 0; baseY same.
    assert_eq!(t1.start_x, 0);
    assert_eq!(t1.start_y, 0);

    // Plane 2 (Cr) — mirror of plane 1's per-plane derivation.
    let t2 = &r.tasks[2];
    assert_eq!(t2.plane, 2);
    assert_eq!(t2.log2_w, 2);
    assert_eq!(t2.log2_h, 2);
}

/// §5.11.33 luma-only path: `has_chroma == false` → only plane 0
/// visited.
#[test]
fn compute_prediction_luma_only_one_task() {
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, false, true, true, false, false, 0, 0, false, 0, 0, false, 0,
        )
        .unwrap();
    assert_eq!(r.num_planes_visited, 1);
    assert_eq!(r.tasks.len(), 1);
    assert_eq!(r.tasks[0].plane, 0);
    assert!(r.tasks[0].have_above);
    assert!(r.tasks[0].have_left);
}

/// r189: §5.11.33 `is_inter == true` arm — the dispatcher now emits
/// one [`oxideav_av1::PlanePredictionTask`] per `(plane, i4, j4)` 4x4
/// sub-block carrying the §7.11.3.1 `predict_inter` arguments
/// (`mode = COMPUTE_PRED_MODE_INTER`). For a BLOCK_8X8 luma-only
/// block this gives `2 x 2 = 4` tasks at (start_x, start_y) ∈
/// {(0,0), (4,0), (0,4), (4,4)}.
#[test]
fn compute_prediction_inter_arm_emits_per_subblock_tasks() {
    use oxideav_av1::COMPUTE_PRED_MODE_INTER;
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, false, false, false, false, false, 0, 0, /* is_inter = */ true,
            0, 0, false, 0,
        )
        .unwrap();
    assert!(r.is_inter);
    assert!(!r.is_inter_intra);
    assert_eq!(r.num_planes_visited, 1);
    // 8x8 luma block at MI_SIZE=4 / num4x4 = 8 / 4 = 2 ⇒ 2 × 2 = 4 tasks.
    assert_eq!(r.tasks.len(), 4);
    let expected_starts: [(u32, u32); 4] = [(0, 0), (4, 0), (0, 4), (4, 4)];
    for (task, (ex, ey)) in r.tasks.iter().zip(expected_starts.iter()) {
        assert_eq!(task.plane, 0);
        assert_eq!(task.mode, COMPUTE_PRED_MODE_INTER);
        assert_eq!(task.log2_w, 2);
        assert_eq!(task.log2_h, 2);
        assert_eq!(task.start_x, *ex);
        assert_eq!(task.start_y, *ey);
    }
}

/// r300: §5.11.33 inter-intra arm — the dispatcher now emits, per
/// plane, one intra `PlanePredictionTask` (the §7.11.3.1 inter-intra
/// blend's intra half) AHEAD of the `is_inter` arm's per-4x4 inter
/// tasks. For a luma-only `BLOCK_8X8` inter-intra block this gives
/// `1` intra task at `(0, 0)` (`log2_w == log2_h == 3`, mode derived
/// from `interintra_mode`) followed by `4` inter tasks (the `2 × 2`
/// 4x4 sub-block grid carrying `COMPUTE_PRED_MODE_INTER`).
#[test]
fn compute_prediction_inter_intra_arm_emits_intra_then_inter_tasks() {
    use oxideav_av1::COMPUTE_PRED_MODE_INTER;
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, false, false, false, false, false, 0, 0, /* is_inter = */ true,
            0, 0, /* ref_frame_1_is_intra = */ true,
            /* interintra_mode = II_DC_PRED */ 0,
        )
        .expect("inter-intra dispatch must now succeed");
    assert!(r.is_inter);
    assert!(r.is_inter_intra);
    assert_eq!(r.num_planes_visited, 1);
    // 1 intra half-task + 4 inter sub-block tasks.
    assert_eq!(r.tasks.len(), 5);

    // The §5.11.33 `IsInterIntra` arm emits FIRST: the intra half.
    let intra = &r.tasks[0];
    assert_eq!(intra.plane, 0);
    assert_eq!(intra.start_x, 0);
    assert_eq!(intra.start_y, 0);
    assert_eq!(intra.log2_w, 3); // whole BLOCK_8X8 region in one call
    assert_eq!(intra.log2_h, 3);
    assert_eq!(intra.mode, 0); // II_DC_PRED → DC_PRED == 0

    // Then the four §7.11.3.1 inter sub-block tasks.
    let expected_starts: [(u32, u32); 4] = [(0, 0), (4, 0), (0, 4), (4, 4)];
    for (task, (ex, ey)) in r.tasks[1..].iter().zip(expected_starts.iter()) {
        assert_eq!(task.plane, 0);
        assert_eq!(task.mode, COMPUTE_PRED_MODE_INTER);
        assert_eq!(task.log2_w, 2);
        assert_eq!(task.log2_h, 2);
        assert_eq!(task.start_x, *ex);
        assert_eq!(task.start_y, *ey);
    }
}

/// r300: §5.11.33 inter-intra `interintra_mode → mode` translation
/// (av1-spec p.82 lines 5142-5145): `II_V_PRED → V_PRED`,
/// `II_H_PRED → H_PRED`, `II_SMOOTH_PRED → SMOOTH_PRED`. The intra
/// half-task's `mode` reflects the translation; the inter tasks are
/// unaffected.
#[test]
fn compute_prediction_inter_intra_mode_translation() {
    // (interintra_mode ordinal, expected intra-half mode ordinal).
    // II_DC_PRED=0→DC_PRED=0, II_V_PRED=1→V_PRED=1, II_H_PRED=2→
    // H_PRED=2, II_SMOOTH_PRED=3→SMOOTH_PRED=9.
    let cases: [(u8, u8); 4] = [(0, 0), (1, 1), (2, 2), (3, 9)];
    for (ii_mode, expected) in cases {
        let mut walker = walker_n(16);
        let r = walker
            .compute_prediction(
                0, 0, BLOCK_8X8, false, false, false, false, false, 0, 0,
                /* is_inter = */ true, 0, 0, /* ref_frame_1_is_intra = */ true, ii_mode,
            )
            .expect("inter-intra dispatch must succeed");
        assert!(r.is_inter_intra);
        assert_eq!(
            r.tasks[0].mode, expected,
            "interintra_mode {ii_mode} must translate to intra mode {expected}",
        );
    }
}

/// r300: §5.11.33 inter-intra arm with chroma — the `IsInterIntra`
/// intra half emits one task per plane (plane 0/1/2), each ahead of
/// that plane's inter sub-block tasks. For a `BLOCK_8X8` 420 block
/// (chroma residual `BLOCK_4X4`): plane 0 = 1 intra + 4 inter,
/// planes 1/2 = 1 intra + 1 inter each.
#[test]
fn compute_prediction_inter_intra_3_plane() {
    use oxideav_av1::COMPUTE_PRED_MODE_INTER;
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, /* has_chroma = */ true, false, false, false, false,
            /* subsampling = */ 1, 1, /* is_inter = */ true, 0, 0,
            /* ref_frame_1_is_intra = */ true, /* II_DC_PRED */ 0,
        )
        .expect("3-plane inter-intra dispatch must succeed");
    assert!(r.is_inter_intra);
    assert_eq!(r.num_planes_visited, 3);
    // plane0: 1 intra + 4 inter = 5; plane1: 1 + 1; plane2: 1 + 1 = 9.
    assert_eq!(r.tasks.len(), 9);
    // First task of each plane is the intra half.
    assert_eq!(r.tasks[0].plane, 0);
    assert_eq!(r.tasks[0].mode, 0);
    assert_eq!(r.tasks[0].log2_w, 3);
    assert_eq!(r.tasks[5].plane, 1);
    assert_eq!(r.tasks[5].mode, 0);
    assert_eq!(r.tasks[5].log2_w, 2); // chroma BLOCK_4X4
    assert_eq!(r.tasks[6].plane, 1);
    assert_eq!(r.tasks[6].mode, COMPUTE_PRED_MODE_INTER);
    assert_eq!(r.tasks[7].plane, 2);
    assert_eq!(r.tasks[7].mode, 0);
    assert_eq!(r.tasks[8].plane, 2);
    assert_eq!(r.tasks[8].mode, COMPUTE_PRED_MODE_INTER);
}

/// r300: §5.11.33 caller-bug guard — an `interintra_mode` outside
/// `0..INTERINTRA_MODES` on the inter-intra arm surfaces
/// `PartitionWalkOutOfRange`. The same out-of-range value on the
/// non-inter-intra arm is ignored (the mode is unread there).
#[test]
fn compute_prediction_inter_intra_mode_out_of_range_guard() {
    let mut walker = walker_n(16);
    let bad = walker.compute_prediction(
        0, 0, BLOCK_8X8, false, false, false, false, false, 0, 0, /* is_inter = */ true, 0, 0,
        /* ref_frame_1_is_intra = */ true, /* interintra_mode = */ 4,
    );
    assert_eq!(bad, Err(Error::PartitionWalkOutOfRange));

    // Non-inter-intra: out-of-range interintra_mode is inert.
    let mut walker2 = walker_n(16);
    let ok = walker2.compute_prediction(
        0, 0, BLOCK_8X8, false, false, false, false, false, 0, 0, /* is_inter = */ false, 0,
        0, /* ref_frame_1_is_intra = */ false, /* interintra_mode = */ 99,
    );
    assert!(ok.is_ok());
}

/// r186: §5.11.33 V_PRED admitted on the dispatcher's intra arm —
/// `y_mode == V_PRED == 1` produces a per-plane task with the
/// V_PRED mode ordinal forwarded into the task. The dispatcher
/// itself does not invoke the §7.11.2.4 step-10 leaf
/// ([`predict_intra_v_pred`]); it surfaces the per-plane prediction
/// task list for the caller to drive against `CurrFrame[plane]`.
#[test]
fn compute_prediction_v_pred_dispatches_one_task_per_plane() {
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, true, true, true, true, true, 1, 1, false,
            /* y_mode = */ 1, // V_PRED
            /* uv_mode = */ 1, // V_PRED
            false, 0,
        )
        .expect("V_PRED intra arm must dispatch post-r186");
    assert_eq!(r.num_planes_visited, 3);
    assert_eq!(r.tasks.len(), 3);
    assert_eq!(r.tasks[0].mode, 1, "plane 0 V_PRED forwarded");
    assert_eq!(r.tasks[1].mode, 1, "plane 1 V_PRED forwarded");
    assert_eq!(r.tasks[2].mode, 1, "plane 2 V_PRED forwarded");
}

/// r186: §5.11.33 H_PRED admitted on the dispatcher's intra arm —
/// mirror of the V_PRED test with `y_mode == H_PRED == 2`.
#[test]
fn compute_prediction_h_pred_dispatches_one_task_per_plane() {
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, true, true, true, true, true, 1, 1, false,
            /* y_mode = */ 2, // H_PRED
            /* uv_mode = */ 2, // H_PRED
            false, 0,
        )
        .expect("H_PRED intra arm must dispatch post-r186");
    assert_eq!(r.num_planes_visited, 3);
    assert_eq!(r.tasks.len(), 3);
    assert_eq!(r.tasks[0].mode, 2);
    assert_eq!(r.tasks[1].mode, 2);
    assert_eq!(r.tasks[2].mode, 2);
}

/// r188: §5.11.33 D-mode intra-mode ordinals (`3..=8`) admitted on
/// the dispatcher's intra arm. Mirrors the r186 / r187 dispatcher
/// acceptance tests — all six §7.11.2.4 non-degenerate directional
/// arms (D45 / D135 / D113 / D157 / D203 / D67) produce a per-plane
/// task with the D-mode ordinal forwarded into the task.
#[test]
fn compute_prediction_d_mode_intra_modes_dispatch_one_task_per_plane() {
    let mut walker = walker_n(16);
    for &mode in &[3u8, 4, 5, 6, 7, 8] {
        let r = walker
            .compute_prediction(
                0, 0, BLOCK_8X8, true, true, true, true, true, 1, 1, false, mode, mode, false, 0,
            )
            .unwrap_or_else(|e| panic!("D-mode {mode} must dispatch post-r188 (got {e:?})"));
        assert_eq!(r.num_planes_visited, 3);
        assert_eq!(r.tasks.len(), 3);
        for t in r.tasks {
            assert_eq!(t.mode, mode);
        }
    }
}

/// r188: §5.11.33 dispatcher rejects `UV_CFL_PRED == 13` when it
/// reaches the luma plane (plane 0). The mode-bounds guard
/// (`plane_mode >= INTRA_MODES && plane_mode != UV_CFL_PRED`) lets
/// `13` pass because chroma legitimately uses it; the post-guard
/// supported check then rejects with
/// `ComputePredictionIntraModeUnsupported`. With r188 admitting all
/// `0..INTRA_MODES` Y intra modes, this is the only remaining path
/// that reaches `ComputePredictionIntraModeUnsupported` on the luma
/// plane.
#[test]
fn compute_prediction_uv_cfl_on_luma_rejected() {
    let mut walker = walker_n(16);
    let result = walker.compute_prediction(
        0, 0, BLOCK_8X8, false, true, true, false, false, 0, 0, false, /* y_mode = */ 13, 0,
        false, 0,
    );
    assert_eq!(result, Err(Error::ComputePredictionIntraModeUnsupported));
}

/// r188: §5.11.33 dispatcher rejects out-of-range Y intra modes
/// (`mode >= INTRA_MODES + 1`) as a caller-bug. The §5.11.7 /
/// §5.11.22 readers cap at INTRA_MODES (or `INTRA_MODES + 1` for
/// UV_CFL_PRED on chroma), so this is the §6.4.1 conformance gate.
#[test]
fn compute_prediction_above_uv_cfl_rejected_as_caller_bug() {
    let mut walker = walker_n(16);
    // mode = 14 (one past UV_CFL_PRED) — out of every conformant
    // §5.11.7 / §5.11.22 reader emission.
    let result = walker.compute_prediction(
        0, 0, BLOCK_8X8, false, true, true, false, false, 0, 0, false, /* y_mode = */ 14, 0,
        false, 0,
    );
    assert_eq!(result, Err(Error::PartitionWalkOutOfRange));
}

/// r187: §5.11.33 SMOOTH_PRED (= 9) now admitted on the dispatcher's
/// intra arm. Mirror of the r186 V_PRED / H_PRED dispatcher acceptance
/// tests.
#[test]
fn compute_prediction_smooth_pred_dispatches_one_task_per_plane() {
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, true, true, true, true, true, 1, 1, false,
            /* y_mode = */ 9, // SMOOTH_PRED
            /* uv_mode = */ 9, false, 0,
        )
        .expect("SMOOTH_PRED intra arm must dispatch post-r187");
    assert_eq!(r.num_planes_visited, 3);
    assert_eq!(r.tasks.len(), 3);
    assert_eq!(r.tasks[0].mode, 9);
    assert_eq!(r.tasks[1].mode, 9);
    assert_eq!(r.tasks[2].mode, 9);
}

/// r187: §5.11.33 SMOOTH_V_PRED (= 10) admitted on the dispatcher's
/// intra arm.
#[test]
fn compute_prediction_smooth_v_pred_dispatches_one_task_per_plane() {
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, true, true, true, true, true, 1, 1, false,
            /* y_mode = */ 10, // SMOOTH_V_PRED
            /* uv_mode = */ 10, false, 0,
        )
        .expect("SMOOTH_V_PRED intra arm must dispatch post-r187");
    assert_eq!(r.num_planes_visited, 3);
    assert_eq!(r.tasks[0].mode, 10);
}

/// r187: §5.11.33 SMOOTH_H_PRED (= 11) admitted on the dispatcher's
/// intra arm.
#[test]
fn compute_prediction_smooth_h_pred_dispatches_one_task_per_plane() {
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, true, true, true, true, true, 1, 1, false,
            /* y_mode = */ 11, // SMOOTH_H_PRED
            /* uv_mode = */ 11, false, 0,
        )
        .expect("SMOOTH_H_PRED intra arm must dispatch post-r187");
    assert_eq!(r.num_planes_visited, 3);
    assert_eq!(r.tasks[0].mode, 11);
}

/// r187: §5.11.33 PAETH_PRED (= 12) admitted on the dispatcher's
/// intra arm.
#[test]
fn compute_prediction_paeth_pred_dispatches_one_task_per_plane() {
    let mut walker = walker_n(16);
    let r = walker
        .compute_prediction(
            0, 0, BLOCK_8X8, true, true, true, true, true, 1, 1, false,
            /* y_mode = */ 12, // PAETH_PRED
            /* uv_mode = */ 12, false, 0,
        )
        .expect("PAETH_PRED intra arm must dispatch post-r187");
    assert_eq!(r.num_planes_visited, 3);
    assert_eq!(r.tasks[0].mode, 12);
}

/// §5.11.33 caller-bug guards.
#[test]
fn compute_prediction_caller_bug_guards() {
    let mut walker = walker_n(16);
    // mi_row out of range.
    assert_eq!(
        walker.compute_prediction(
            999, 0, BLOCK_8X8, false, false, false, false, false, 0, 0, false, 0, 0, false, 0,
        ),
        Err(Error::PartitionWalkOutOfRange)
    );
    // mi_col out of range.
    assert_eq!(
        walker.compute_prediction(
            0, 999, BLOCK_8X8, false, false, false, false, false, 0, 0, false, 0, 0, false, 0,
        ),
        Err(Error::PartitionWalkOutOfRange)
    );
    // mi_size out of range.
    assert_eq!(
        walker.compute_prediction(
            0, 0, 99, false, false, false, false, false, 0, 0, false, 0, 0, false, 0,
        ),
        Err(Error::PartitionWalkOutOfRange)
    );
    // subsampling out of range.
    assert_eq!(
        walker.compute_prediction(
            0, 0, BLOCK_8X8, false, false, false, false, false, 2, 0, false, 0, 0, false, 0,
        ),
        Err(Error::PartitionWalkOutOfRange)
    );
}

/// §7.11.2.5 DC_PRED leaf — `(haveLeft, haveAbove) = (0, 0)` four-arm
/// fallback writes `1 << (BitDepth - 1)` (= 128 for 8-bit) into every
/// cell of an 8×8 prediction region.
#[test]
fn predict_intra_dc_pred_no_neighbours_fills_mid_grey() {
    let mut pred = [0u16; 8 * 8];
    let above = [0u16; 0];
    let left = [0u16; 0];
    predict_intra_dc_pred(0, 0, 3, 3, 8, 8, 8, &above, &left, &mut pred).unwrap();
    for cell in &pred {
        assert_eq!(*cell, 128);
    }
}

/// §7.11.2.5 DC_PRED — 10-bit fallback writes `512` (= `1 << 9`).
#[test]
fn predict_intra_dc_pred_no_neighbours_10bit() {
    let mut pred = [0u16; 4 * 4];
    let none: [u16; 0] = [];
    predict_intra_dc_pred(0, 0, 2, 2, 4, 4, 10, &none, &none, &mut pred).unwrap();
    assert!(pred.iter().all(|&v| v == 512));
}

/// §7.11.2.5 DC_PRED — `(haveLeft=1, haveAbove=0)` left-only average
/// with `Clip1`.
#[test]
fn predict_intra_dc_pred_left_only_average() {
    let mut pred = [0u16; 4 * 4];
    // 4 left samples summing to 100: 25 each. (h>>1)=2. (100+2)>>2=25.
    let left = [25u16, 25, 25, 25];
    let above: [u16; 0] = [];
    predict_intra_dc_pred(1, 0, 2, 2, 4, 4, 8, &above, &left, &mut pred).unwrap();
    assert!(pred.iter().all(|&v| v == 25));
}

/// §7.11.2.5 DC_PRED — `(haveLeft=0, haveAbove=1)` above-only average.
#[test]
fn predict_intra_dc_pred_above_only_average() {
    let mut pred = [0u16; 4 * 4];
    let above = [40u16, 40, 40, 40];
    let left: [u16; 0] = [];
    predict_intra_dc_pred(0, 1, 2, 2, 4, 4, 8, &above, &left, &mut pred).unwrap();
    assert!(pred.iter().all(|&v| v == 40));
}

/// §7.11.2.5 DC_PRED — union arm averages left + above. Both filled
/// with 50 → output is 50. `(w+h)/2 = 4` rounding term added before
/// integer division by `w + h = 8`: (50*4 + 50*4 + 4)/8 = (400 + 4)/8
/// = 404/8 = 50 (truncating division).
#[test]
fn predict_intra_dc_pred_union_average() {
    let mut pred = [0u16; 4 * 4];
    let above = [50u16, 50, 50, 50];
    let left = [50u16, 50, 50, 50];
    predict_intra_dc_pred(1, 1, 2, 2, 4, 4, 8, &above, &left, &mut pred).unwrap();
    assert!(pred.iter().all(|&v| v == 50));
}

/// §7.11.2.5 DC_PRED — caller-bug guards.
#[test]
fn predict_intra_dc_pred_caller_bug_guards() {
    let mut pred = [0u16; 16];
    let row: [u16; 8] = [0; 8];
    // Bad bit_depth.
    assert_eq!(
        predict_intra_dc_pred(0, 0, 2, 2, 4, 4, 9, &row, &row, &mut pred),
        Err(Error::PartitionWalkOutOfRange)
    );
    // Mismatched log2/dim.
    assert_eq!(
        predict_intra_dc_pred(0, 0, 2, 2, 5, 4, 8, &row, &row, &mut pred),
        Err(Error::PartitionWalkOutOfRange)
    );
    // Output buffer too short.
    let mut short = [0u16; 8];
    assert_eq!(
        predict_intra_dc_pred(0, 0, 2, 2, 4, 4, 8, &row, &row, &mut short),
        Err(Error::PartitionWalkOutOfRange)
    );
    // above_row too short when have_above=1.
    let too_short = [0u16; 2];
    assert_eq!(
        predict_intra_dc_pred(0, 1, 2, 2, 4, 4, 8, &too_short, &row, &mut pred),
        Err(Error::PartitionWalkOutOfRange)
    );
    // log2 out of range.
    assert_eq!(
        predict_intra_dc_pred(0, 0, 7, 2, 128, 4, 8, &row, &row, &mut pred),
        Err(Error::PartitionWalkOutOfRange)
    );
}

/// `PlanePredictionTask` / `ComputePredictionReadout` `Clone` +
/// `Debug` smoke — keeps the public API constructable.
#[test]
fn compute_prediction_readout_clone_debug_smoke() {
    let task = PlanePredictionTask {
        plane: 0,
        start_x: 0,
        start_y: 0,
        log2_w: 3,
        log2_h: 3,
        mode: 0,
        have_left: false,
        have_above: false,
    };
    let readout = ComputePredictionReadout {
        is_inter_intra: false,
        is_inter: false,
        num_planes_visited: 1,
        tasks: vec![task],
    };
    let cloned = readout.clone();
    assert_eq!(readout, cloned);
    let _ = format!("{readout:?}");
}

// =====================================================================
// r190 — `decode_block_syntax` wire-up tests.
//
// The §5.11.5 [`PartitionWalker::decode_block_syntax`] dispatcher now
// accepts an `Option<&InterFrameContext>` last argument. The pre-r190
// short-circuit (`Err(Error::DecodeBlockInterFrameUnsupported)` on
// `frame_is_intra == false`) is preserved when the caller passes
// `None` (covered by `decode_block_syntax_inter_frame_arm_returns_stub`
// above). When the caller passes `Some(&ctx)` the §5.11.18 inter
// dispatcher runs, the §5.11.23 inter-block cascade completes, and the
// walker threads MV / ref-frame / interp-filter state into
// [`PartitionWalker::compute_prediction`] (the §5.11.33 inter arm
// per r189) + [`PartitionWalker::residual`] (the §5.11.34 `is_inter &&
// !Lossless && !plane` transform_tree arm).
// =====================================================================

/// r190 wire-up: `decode_block_syntax(frame_is_intra = false,
/// Some(&ctx))` lifts the pre-r190
/// `Err(Error::DecodeBlockInterFrameUnsupported)` short-circuit and
/// runs the §5.11.18 inter cascade + §5.11.33 inter-arm
/// `compute_prediction` + §5.11.34 inter-arm `residual` to completion.
///
/// Test fixture: `BLOCK_8X8` inter block, intra-only sequence header
/// surrogate (`subsampling_x = subsampling_y = 0`, `num_planes = 3`),
/// segmentation disabled, `seg_globalmv_active = true` so the §5.11.20
/// `read_is_inter` third arm forces `is_inter = 1` without an S()
/// read, identity `InterFrameContext` defaults for every other inter
/// scalar. The §5.11.25 `read_ref_frames` reader hits the
/// `seg_skip_active || seg_globalmv_active` arm which sets
/// `RefFrame[0] = LAST_FRAME`, `RefFrame[1] = NONE` (no S() bits
/// read). The §5.11.23 YMode dispatch falls into the `seg_skip_active
/// || seg_globalmv_active` arm (no S()) and resolves `YMode =
/// GLOBALMV`. The §5.11.31 `assign_mv(GLOBALMV)` arm reads
/// `GlobalMvs[0]` (identity → [0, 0]) — no S() bits. Motion mode
/// short-circuits to SIMPLE (no S() bits) because
/// `is_motion_mode_switchable = false`. `read_interintra_mode` /
/// `read_compound_type` / `read_interpolation_filter` all
/// short-circuit on the closed outer gates. The walker then advances
/// through §5.11.33 (one per-4x4 inter task per plane) + §5.11.34
/// (transform_tree on the luma plane, direct iteration on chroma).
#[test]
fn r190_decode_block_syntax_with_inter_ctx_runs_inter_arm_to_completion() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // Skip CDF deterministically returns 0 so the §5.11.11 read_skip
    // takes the "no-skip" arm — §5.11.34 will visit per-TU
    // coefficient reads.
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    let bytes = [0u8; 64];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 64, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let mut ctx = InterFrameContext::identity_default(&mfmvs);
    // §5.11.20 `read_is_inter` third arm: seg_globalmv_active forces
    // is_inter = 1 with no S() read.
    ctx.seg_globalmv_active = true;

    let result = walker.decode_block_syntax(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        /* frame_is_intra = */ false,
        /* subsampling_x = */ 0,
        /* subsampling_y = */ 0,
        /* num_planes = */ 3,
        /* seg_id_pre_skip = */ false,
        /* segmentation_enabled = */ false,
        /* seg_skip_active = */ false,
        /* last_active_seg_id = */ 0,
        &lossless,
        /* coded_lossless = */ false,
        /* enable_cdef = */ true,
        /* allow_intrabc = */ false,
        /* cdef_bits = */ 0,
        /* read_deltas = */ false,
        /* use_128x128_superblock = */ false,
        /* delta_q_res = */ 0,
        /* delta_lf_present = */ false,
        /* delta_lf_multi = */ false,
        /* mono_chrome = */ false,
        /* delta_lf_res = */ 0,
        /* allow_screen_content_tools = */ false,
        /* enable_filter_intra = */ false,
        /* bit_depth = */ 8,
        /* tx_mode_select = */ false,
        /* inter_ctx = */ Some(&ctx),
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );

    let db = result.expect(
        "r190 wire-up: §5.11.18 inter arm + §5.11.33 compute_prediction (inter) + §5.11.34 residual (inter transform_tree) should run to completion on the globalmv path",
    );
    assert_eq!(
        db.is_inter, 1,
        "r190: §5.11.18 `is_inter == 1` after seg_globalmv_active override"
    );
    assert_eq!(
        db.ref_frame,
        [1, -1],
        "r190: §5.11.25 globalmv arm sets RefFrame = [LAST_FRAME, NONE]"
    );
    assert!(!db.is_compound, "single-ref: slot 1 = NONE ⇒ !isCompound");
    // §5.11.33 `IsInterIntra = ( is_inter && RefFrame[1] ==
    // INTRA_FRAME )`. On the globalmv arm `RefFrame[1] = NONE = -1 ≠
    // INTRA_FRAME = 0`, so the §5.11.33 gate stays closed even though
    // `is_inter == 1`. The flag is now read back out of the
    // `compute_prediction()` readout and surfaced on `DecodedBlock`.
    assert!(
        !db.is_inter_intra,
        "globalmv block: RefFrame[1] = NONE ⇒ IsInterIntra == false"
    );
    assert_eq!(
        db.mi_row, 0,
        "r190: §5.11.5 prologue records the per-block coordinates"
    );
    assert_eq!(db.mi_col, 0);
    assert_eq!(db.mi_size, BLOCK_8X8);
    // r189: the §5.11.31 `assign_mv(GLOBALMV)` arm adopts the
    // identity-default GlobalMvs[0] = [0, 0] (no bits read). The
    // grid stamp over the bw4 * bh4 footprint records the zero MV
    // for downstream §7.10 neighbour walks.
    let mi_cols = walker.mi_cols() as usize;
    // §5.11.5 / §5.11.31 grid stamp: `Mvs[r + y][c + x][refList]`.
    // `[(cell * 2 + list) * 2 + comp]` per-list / per-component
    // layout (mirrors the `decode_inter_block_mode_info` stamp).
    let cell: usize = 0; // (mi_row * mi_cols + mi_col) at (0, 0).
    assert_eq!(
        walker.mvs()[(cell * 2) * 2],
        0,
        "identity globalmv ⇒ row component = 0"
    );
    assert_eq!(
        walker.mvs()[(cell * 2) * 2 + 1],
        0,
        "identity globalmv ⇒ col component = 0"
    );
    // §5.11.5 / §5.11.18 grid stamps: `IsInters[r][c] = is_inter`
    // over the 2×2 footprint.
    for r in 0..2usize {
        for c in 0..2usize {
            assert_eq!(walker.is_inters()[r * mi_cols + c], 1);
        }
    }
}

/// r346 end-to-end two-pass inter flow: decode a real single-reference
/// inter leaf through the §5.11 syntax walk
/// ([`PartitionWalker::decode_block_syntax`], the seg_globalmv path that
/// resolves `RefFrame = [LAST_FRAME, NONE]` + zero `GlobalMvs[0]`), then
/// run the §5.11.33 frame-scope reconstruction bridge
/// ([`PartitionWalker::reconstruct_inter_frame_into_curr_frame`]) to
/// motion-compensate the decoded leaf's pixels into `CurrFrame[0]`. This
/// exercises the whole inter arc end-to-end: bitstream syntax → grids →
/// reconstructed inter pixels. With a zero MV (`GLOBALMV` identity) and
/// `EIGHTTAP` filters at an integer position, the §7.11.3.1 MC copies the
/// reference block verbatim, so the luma plane must equal the reference
/// frame's top-left 8×8 window.
#[test]
fn r346_two_pass_inter_decode_reconstructs_pixels_from_bitstream() {
    use oxideav_av1::{PlaneRefSpec, RefFrameStoreEntry, LAST_FRAME};

    // 2×2 mi walker → an 8×8 luma frame; one BLOCK_8X8 inter leaf covers it.
    let mut walker = walker_n(2);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // §5.11.11 skip = 1 ⇒ no residual reads; CurrFrame[0] stays zero
    // before the inter pass, so the bridge result is pure MC.
    cdfs.skip = [force_binary_cdf(1); SKIP_CONTEXTS];
    let bytes = [0u8; 64];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 64, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let mut ctx = InterFrameContext::identity_default(&mfmvs);
    // §5.11.20 third arm: seg_globalmv_active forces is_inter = 1 (no S()),
    // §5.11.25 sets RefFrame = [LAST_FRAME, NONE], §5.11.23 YMode = GLOBALMV
    // → §5.11.31 assign_mv adopts identity GlobalMvs[0] = [0, 0].
    ctx.seg_globalmv_active = true;

    let db = walker
        .decode_block_syntax(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_8X8,
            /* frame_is_intra = */ false,
            /* subsampling_x = */ 0,
            /* subsampling_y = */ 0,
            /* num_planes = */ 3,
            /* seg_id_pre_skip = */ false,
            /* segmentation_enabled = */ false,
            /* seg_skip_active = */ false,
            /* last_active_seg_id = */ 0,
            &lossless,
            /* coded_lossless = */ false,
            /* enable_cdef = */ true,
            /* allow_intrabc = */ false,
            /* cdef_bits = */ 0,
            /* read_deltas = */ false,
            /* use_128x128_superblock = */ false,
            /* delta_q_res = */ 0,
            /* delta_lf_present = */ false,
            /* delta_lf_multi = */ false,
            /* mono_chrome = */ false,
            /* delta_lf_res = */ 0,
            /* allow_screen_content_tools = */ false,
            /* enable_filter_intra = */ false,
            /* bit_depth = */ 8,
            /* tx_mode_select = */ false,
            /* inter_ctx = */ Some(&ctx),
            /* quant = */ &QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        )
        .expect("§5.11 inter syntax walk should decode the globalmv leaf");
    assert_eq!(db.is_inter, 1);
    assert_eq!(db.ref_frame, [LAST_FRAME as i8, -1]);
    assert_eq!(db.mv, [[0, 0], [0, 0]], "identity GLOBALMV ⇒ zero MV");

    // §7.11.3.3 reference frame (8×8 luma minimum; the MC reads only the
    // 8×8 window at the zero-MV integer position). A distinct per-sample
    // ramp so the copy is observable.
    let ref_w: usize = 8;
    let ref_h: usize = 8;
    let mut refp = vec![0u16; ref_w * ref_h];
    for r in 0..ref_h {
        for c in 0..ref_w {
            refp[r * ref_w + c] = (10 + r * 8 + c) as u16;
        }
    }
    let entry = RefFrameStoreEntry {
        plane: &refp[..],
        stride: ref_w,
        upscaled_width: ref_w as u32,
        width: ref_w as u32,
        height: ref_h as u32,
    };
    // `ref_frame_idx[ LAST_FRAME - LAST_FRAME = 0 ]` = slot 0.
    let store = [entry, entry, entry, entry, entry, entry, entry];
    let ref_frame_idx: [u8; 7] = [0, 1, 2, 3, 4, 5, 6];

    let ref_spec = PlaneRefSpec {
        plane: 0,
        subsampling_x: 0,
        subsampling_y: 0,
        frame_store: &store,
        frame_width: ref_w as u32,
        frame_height: ref_h as u32,
    };

    walker
        .reconstruct_inter_frame_into_curr_frame(&ref_frame_idx, 8, &[ref_spec])
        .expect("frame-scope inter reconstruction from the decoded grids");

    // Zero-MV EIGHTTAP MC at an integer position copies the reference 8×8
    // window verbatim into CurrFrame[0].
    let (rows, cols) = walker.curr_frame_dims(0).unwrap();
    assert_eq!((rows, cols), (8, 8));
    let got: Vec<u16> = walker
        .curr_frame(0)
        .unwrap()
        .iter()
        .map(|&s| s as u16)
        .collect();
    assert_eq!(
        got, refp,
        "two-pass inter decode: zero-MV MC must copy the reference 8×8 window into CurrFrame[0]"
    );
}

/// Helper: force an `N`-symbol CDF (length `N + 1`, last slot = adapt
/// count) to deterministically decode symbol `0`. The §8.2.6 search
/// breaks at the first `symbol` whose `cur` falls at/below
/// `SymbolValue`; setting `cdf[0] = 1 << 15` (so `f = (1<<15) -
/// cdf[0] = 0` ⇒ `cur` collapses to the `EC_MIN_PROB` tail) makes
/// symbol 0 the break point regardless of the arithmetic state.
fn force_cdf_symbol_zero<const N1: usize>() -> [u16; N1] {
    let mut row = [1u16 << 15; N1];
    // Last slot is the §8.3 adaptation counter, initialised to 0.
    row[N1 - 1] = 0;
    row
}

/// r302 wire-up: `decode_block_syntax(frame_is_intra = false,
/// Some(&ctx))` surfaces the §5.11.33 `IsInterIntra` verdict on the
/// returned [`DecodedBlock`]. This test rigs the §5.11.28
/// `read_interintra_mode` inner arm to fire so the §5.11.5 inter
/// walker produces a real inter-intra block end-to-end and the new
/// `db.is_inter_intra` flag reads `true`.
///
/// Fixture: `BLOCK_8X8` (inside the §5.11.28 `BLOCK_8X8..=BLOCK_32X32`
/// band) single-ref inter block on the same `seg_globalmv_active`
/// no-S() cascade as the previous test (so `RefFrame = [LAST_FRAME,
/// NONE]`, `!isCompound`, motion-mode SIMPLE). `enable_interintra_
/// compound = true` opens the §5.11.28 outer gate; the three §5.11.28
/// CDF rows are rigged so `interintra → 1`, `interintra_mode →
/// II_DC_PRED` and `wedge_interintra → 0` (no wedge-index read). The
/// §5.11.28 inner arm then stamps `RefFrame[1] = INTRA_FRAME = 0`,
/// the §5.11.33 `IsInterIntra = ( is_inter && RefFrame[1] ==
/// INTRA_FRAME )` gate fires, and the inter walker reads the verdict
/// back out of the `compute_prediction()` readout.
#[test]
fn r302_decode_block_syntax_inter_intra_block_surfaces_is_inter_intra_flag() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];

    // §5.11.28 CDF rigging. `interintra_ctx(BLOCK_8X8)` selects the
    // Size_Group-1 row consumed by the `interintra` / `interintra_mode`
    // S(); `wedge_inter_intra` / nothing else is indexed straight by
    // MiSize.
    let ii_ctx = interintra_ctx(BLOCK_8X8).expect("BLOCK_8X8 is in the §5.11.28 band");
    cdfs.inter_intra[ii_ctx] = force_binary_cdf(1);
    cdfs.inter_intra_mode[ii_ctx] = force_cdf_symbol_zero::<{ INTERINTRA_MODES + 1 }>();
    cdfs.wedge_inter_intra[BLOCK_8X8] = force_binary_cdf(0);

    let bytes = [0xffu8; 64];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 64, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let mut ctx = InterFrameContext::identity_default(&mfmvs);
    ctx.seg_globalmv_active = true;
    // §5.5.2 sequence-header bit — opens the §5.11.28 outer gate.
    ctx.enable_interintra_compound = true;

    let db = walker
        .decode_block_syntax(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_8X8,
            /* frame_is_intra = */ false,
            /* subsampling_x = */ 0,
            /* subsampling_y = */ 0,
            /* num_planes = */ 3,
            /* seg_id_pre_skip = */ false,
            /* segmentation_enabled = */ false,
            /* seg_skip_active = */ false,
            /* last_active_seg_id = */ 0,
            &lossless,
            /* coded_lossless = */ false,
            /* enable_cdef = */ true,
            /* allow_intrabc = */ false,
            /* cdef_bits = */ 0,
            /* read_deltas = */ false,
            /* use_128x128_superblock = */ false,
            /* delta_q_res = */ 0,
            /* delta_lf_present = */ false,
            /* delta_lf_multi = */ false,
            /* mono_chrome = */ false,
            /* delta_lf_res = */ 0,
            /* allow_screen_content_tools = */ false,
            /* enable_filter_intra = */ false,
            /* bit_depth = */ 8,
            /* tx_mode_select = */ false,
            /* inter_ctx = */ Some(&ctx),
            /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        )
        .expect("§5.11.28 inner arm + §5.11.33 inter-intra gate must run to completion");

    assert_eq!(db.is_inter, 1, "seg_globalmv_active ⇒ is_inter == 1");
    // §5.11.28 inner arm overrode the §5.11.25 slot-1 NONE to
    // INTRA_FRAME = 0, so the per-block aggregate now carries
    // RefFrame[1] = INTRA_FRAME.
    assert_eq!(
        db.ref_frame,
        [1, 0],
        "§5.11.28 inner arm sets RefFrame = [LAST_FRAME, INTRA_FRAME]"
    );
    // §5.11.5 `IsCompound = RefFrame[1] > INTRA_FRAME` — INTRA_FRAME
    // (0) is NOT > INTRA_FRAME, so inter-intra is single-pred.
    assert!(
        !db.is_compound,
        "inter-intra: RefFrame[1] = INTRA_FRAME ⇒ !isCompound"
    );
    // The r302 payload: the §5.11.33 verdict is surfaced.
    assert!(
        db.is_inter_intra,
        "RefFrame[1] = INTRA_FRAME && is_inter ⇒ IsInterIntra == true (surfaced on DecodedBlock)"
    );
}

/// r190 wire-up: `decode_partition_syntax(frame_is_intra = false,
/// Some(&ctx))` propagates the inter context through the §5.11.4
/// partition-tree recursion. With `BLOCK_4X4` (below `BLOCK_8X8` ⇒
/// PARTITION_NONE short-circuit) the partition driver invokes
/// `decode_block_syntax` exactly once at `(0, 0)` and the §5.11.18
/// inter cascade runs through the same `seg_globalmv_active` arm as
/// the previous test. CDF rigging mirrors the previous test.
#[test]
fn r190_decode_partition_syntax_with_inter_ctx_routes_through_inter_arm() {
    let mut walker = walker_n(4);
    let mut cdfs = TileCdfContext::new_from_defaults();
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    let bytes = [0u8; 64];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 64, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let mut ctx = InterFrameContext::identity_default(&mfmvs);
    ctx.seg_globalmv_active = true;

    let result = walker.decode_partition_syntax(
        &mut dec,
        &mut cdfs,
        /* r = */ 0,
        /* c = */ 0,
        /* b_size = */ BLOCK_4X4,
        /* frame_is_intra = */ false,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ Some(&ctx),
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    assert_eq!(
        result,
        Ok(()),
        "r190: §5.11.4 partition driver propagates the §5.11.5 walker's Ok(_) from the inter arm"
    );
    // Exactly one leaf block was emitted at (0, 0, BLOCK_4X4).
    let blocks = walker.blocks();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].sub_size, BLOCK_4X4);
}

/// r190: caller passing `frame_is_intra = false, inter_ctx = None`
/// preserves the pre-r190 short-circuit semantics
/// (`Err(Error::DecodeBlockInterFrameUnsupported)`). This guards the
/// back-compat contract for callers that haven't yet built an
/// `InterFrameContext` (e.g. tests that pre-date r190 or callers
/// staying on the intra-only path).
#[test]
fn r190_decode_block_syntax_inter_arm_without_ctx_keeps_legacy_stub() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let bytes = [0xFFu8; 16];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 16, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let pos_before = dec.position();
    let result = walker.decode_block_syntax(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        /* frame_is_intra = */ false,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ None,
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    assert_eq!(
        result,
        Err(Error::DecodeBlockInterFrameUnsupported),
        "r190: `None` inter_ctx ⇒ legacy short-circuit preserved"
    );
    assert_eq!(
        dec.position(),
        pos_before,
        "legacy short-circuit fires before any bitstream read"
    );
}

/// r190: `InterFrameContext::identity_default` smoke — every field
/// initialised to the §5.9.24 identity defaults
/// (`gm_params[ref][2] = gm_params[ref][5] = 1 << WARPEDMODEL_PREC_BITS`,
/// other slots = 0, `gm_type = [GM_TYPE_IDENTITY; 8]`, every flag off,
/// `interpolation_filter = EIGHTTAP`).
#[test]
fn r190_inter_frame_context_identity_default_matches_spec_identity_warp() {
    let mfmvs = MotionFieldMvs::new_invalid(16, 16);
    let ctx = InterFrameContext::identity_default(&mfmvs);
    for ref_idx in 0..8 {
        assert_eq!(
            ctx.gm_params[ref_idx][2],
            1 << WARPEDMODEL_PREC_BITS,
            "§5.9.24 identity warp: slot 2 = 1 << WARPEDMODEL_PREC_BITS"
        );
        assert_eq!(
            ctx.gm_params[ref_idx][5],
            1 << WARPEDMODEL_PREC_BITS,
            "§5.9.24 identity warp: slot 5 = 1 << WARPEDMODEL_PREC_BITS"
        );
        assert_eq!(ctx.gm_params[ref_idx][0], 0);
        assert_eq!(ctx.gm_params[ref_idx][1], 0);
        assert_eq!(ctx.gm_params[ref_idx][3], 0);
        assert_eq!(ctx.gm_params[ref_idx][4], 0);
        assert_eq!(ctx.gm_type[ref_idx], GM_TYPE_IDENTITY);
    }
    assert!(!ctx.segmentation_update_map);
    assert!(!ctx.seg_skip_mode_off);
    assert!(!ctx.seg_globalmv_active);
    assert_eq!(ctx.skip_mode_frame, [0, -1]);
    assert!(!ctx.allow_high_precision_mv);
    assert!(!ctx.enable_interintra_compound);
    assert_eq!(ctx.interpolation_filter, 0 /* EIGHTTAP */);
}

/// r190: `DecodedInterFrameModeInfo::inter_block` is `None` on the
/// §5.11.22 intra-arm path (when `is_inter == 0` inside an inter
/// frame). Test the §5.11.18 dispatcher returns
/// `Err(IntraBlockModeInfoUnsupported)` on the `is_inter == 0` arm
/// (unchanged by r190 since the §5.11.22 stub is still upstream).
#[test]
fn r190_decoded_inter_frame_mode_info_intra_arm_still_stubs() {
    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    // Skip CDF off; segmentation off; is_inter forced to 0 via the
    // CDF surface: the §5.11.20 read_is_inter S() reads against
    // is_inter_cdf — we want it to return 0. Set the CDF to force
    // 0 deterministically.
    cdfs.skip = [force_binary_cdf(0); SKIP_CONTEXTS];
    for row in cdfs.is_inter.iter_mut() {
        *row = force_binary_cdf(0);
    }
    let bytes = [0u8; 64];
    let mut dec = SymbolDecoder::init_symbol(&bytes, 64, true).unwrap();
    let lossless = [false; MAX_SEGMENTS];

    let mfmvs = MotionFieldMvs::new_invalid(walker.mi_rows(), walker.mi_cols());
    let ctx = InterFrameContext::identity_default(&mfmvs);
    let result = walker.decode_block_syntax(
        &mut dec,
        &mut cdfs,
        0,
        0,
        BLOCK_8X8,
        /* frame_is_intra = */ false,
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
        false,
        false,
        8,
        false,
        /* inter_ctx = */ Some(&ctx),
        /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
        /* reduced_tx_set = */ false,
    );
    assert_eq!(
        result,
        Err(Error::IntraBlockModeInfoUnsupported),
        "r190: §5.11.18 `is_inter == 0` arm still stubs at §5.11.22 (next-arc target)"
    );
}

// ===================================================================
// r281 — full §5.11.7 composition threaded into decode_block_syntax:
// the `else` arm runs the r280 composite (intra_frame_y_mode +
// intra_angle_info_y + the HasChroma arm + §5.11.46 palette_mode_info
// + §5.11.24 filter_intra_mode_info), the `use_intrabc == 1` arm runs
// §7.10.2 find_mv_stack(0) + the §5.11.26 assign_mv(0) intra-block-
// copy body, and §5.11.49 palette_tokens() consumes its NS() + S()
// reads on palette blocks. Each test writes the block with the
// encoder-side §5.11.7 writers, appends an 8-bit sync sentinel, and
// asserts the walker leaves the decoder positioned exactly at the
// sentinel (a missing or extra read desynchronises the arithmetic
// coder and fails the literal assertion).
// ===================================================================

/// r281: §5.11.7 `else` arm full composition — directional luma +
/// chroma (V_PRED + AngleDeltaY = +2, UVMode = V_PRED + AngleDeltaUV
/// = -1) round-trips through `decode_block_syntax`, which previously
/// stopped after `intra_frame_y_mode`.
#[test]
fn r281_decode_block_syntax_runs_full_else_arm_composition() {
    // Write side: §5.11.11 skip = 1 (origin ctx 0), then the §5.11.7
    // else-arm body, then the sentinel.
    let walker_w = walker_n(16);
    let mut enc_cdfs = TileCdfContext::new_from_defaults();
    let mut writer = SymbolWriter::new(false);
    let dummy = [0u16; PALETTE_COLORS];
    write_skip(&mut writer, &mut enc_cdfs, 1, 0, false).unwrap();
    write_intra_frame_else_arm(
        &mut writer,
        &mut enc_cdfs,
        BLOCK_8X8,
        V_PRED as u8,
        Some(V_PRED as u8),
        /* angle_delta_y = */ 2,
        /* angle_delta_uv = */ -1,
        // §8.3.2 CFL allowance: !Lossless && Max(Block_Width,
        // Block_Height) = 8 <= 32 ⇒ allowed.
        /* cfl_allowed = */
        true,
        /* has_chroma = */ true,
        /* allow_screen_content_tools = */ false,
        /* enable_filter_intra = */ false,
        0,
        None,
        false,
        false,
        8,
        0,
        0,
        &dummy,
        0,
        0,
        &dummy,
        &dummy,
        false,
        None,
        None,
        &walker_w,
        0,
        0,
        /* abovemode_ctx = */ 0,
        /* leftmode_ctx = */ 0,
    )
    .unwrap();
    writer.write_literal(8, 0xA5).unwrap();
    let bytes = writer.finish();

    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
    let lossless = [false; MAX_SEGMENTS];
    let db = walker
        .decode_block_syntax(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_8X8,
            /* frame_is_intra = */ true,
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
            /* allow_intrabc = */ false,
            0,
            false,
            false,
            0,
            false,
            false,
            false,
            0,
            /* allow_screen_content_tools = */ false,
            /* enable_filter_intra = */ false,
            /* bit_depth = */ 8,
            /* tx_mode_select = */ false,
            /* inter_ctx = */ None,
            /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        )
        .expect("r281: the full §5.11.7 else-arm composition must run to completion");

    assert_eq!(db.skip, 1, "§5.11.11 skip = 1");
    assert_eq!(db.use_intrabc, 0, "§5.11.7 use_intrabc = 0 on the else arm");
    assert_eq!(db.is_inter, 0, "§5.11.7 else arm: is_inter = 0");
    assert_eq!(db.y_mode, V_PRED as u8, "§5.11.7 intra_frame_y_mode");
    assert_eq!(db.angle_delta_y, 2, "§5.11.42 AngleDeltaY");
    assert_eq!(db.uv_mode, Some(V_PRED as u8), "§5.11.22 uv_mode");
    assert_eq!(db.angle_delta_uv, Some(-1), "§5.11.43 AngleDeltaUV");
    assert_eq!(db.palette_size_y, 0, "§5.11.22 line 11: PaletteSizeY = 0");
    assert_eq!(db.palette_size_uv, 0, "§5.11.22 line 12: PaletteSizeUV = 0");
    assert_eq!(db.use_filter_intra, None, "§5.11.24 outer gate closed");
    assert_eq!(db.mv, [[0, 0], [0, 0]], "no MV coded on the else arm");

    // §8.3 adaptation lockstep on the arm-distinguishing CDF row.
    assert_eq!(
        enc_cdfs.intra_frame_y_mode_cdf(0, 0).to_vec(),
        cdfs.intra_frame_y_mode_cdf(0, 0).to_vec(),
        "TileIntraFrameYModeCdf[0][0] adaptation must match"
    );

    // Sync sentinel: the walker consumed exactly the §5.11.5 reads.
    assert_eq!(
        dec.read_literal(8).unwrap(),
        0xA5,
        "decoder must be positioned at the sentinel after the block"
    );
}

/// r281: §5.11.7 `use_intrabc == 1` arm — `decode_block_syntax` now
/// runs §7.10.2 `find_mv_stack( 0 )` + the §5.11.26 `assign_mv( 0 )`
/// intra-block-copy body (previously the MV bits were not consumed).
/// At the frame origin the MV stack is empty, so the §5.11.26
/// fallback chain yields `PredMv[ 0 ] = [ 0, -(sbSize4 * MI_SIZE +
/// INTRABC_DELAY_PIXELS) * 8 ] = [ 0, -2560 ]` (64×64 superblocks);
/// the coded block vector equals the predictor (`MV_JOINT_ZERO`).
#[test]
fn r281_decode_block_syntax_intrabc_arm_runs_mv_chain() {
    // Write side mirrors the §5.11.7 region: skip = 1, then
    // `use_intrabc` S() + the §5.11.26 / §5.11.31 MV write.
    let walker_w = walker_n(16);
    let mfmvs = MotionFieldMvs::new_invalid(16, 16);
    let stack = walker_w
        .find_mv_stack(
            0,
            0,
            BLOCK_8X8,
            /* ref_frame = */ [0, -1],
            /* is_compound = */ false,
            /* use_ref_frame_mvs = */ false,
            [GM_TYPE_IDENTITY; 8],
            identity_gm_params(),
            [0; 8],
            /* allow_high_precision_mv = */ false,
            /* force_integer_mv = */ true,
            &mfmvs,
        )
        .unwrap();
    let mv = [0, -2560];
    let mut enc_cdfs = TileCdfContext::new_from_defaults();
    let mut writer = SymbolWriter::new(false);
    write_skip(&mut writer, &mut enc_cdfs, 1, 0, false).unwrap();
    let inputs = IntrabcArmInputs {
        mv,
        mv_stack: &stack,
        use_128x128_superblock: false,
        mi_row: 0,
        mi_row_start: 0,
    };
    let info = write_intra_frame_intrabc_arm(&mut writer, &mut enc_cdfs, true, Some(&inputs))
        .unwrap()
        .expect("intrabc arm must fire");
    assert_eq!(info.pred_mv, [0, -2560], "§5.11.26 fallback predictor");
    writer.write_literal(8, 0x5A).unwrap();
    let bytes = writer.finish();

    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
    let lossless = [false; MAX_SEGMENTS];
    let db = walker
        .decode_block_syntax(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_8X8,
            /* frame_is_intra = */ true,
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
            /* allow_intrabc = */ true,
            0,
            false,
            false,
            0,
            false,
            false,
            false,
            0,
            false,
            false,
            8,
            /* tx_mode_select = */ false,
            /* inter_ctx = */ None,
            /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        )
        .expect("r281: the §5.11.7 intrabc arm must run to completion");

    assert_eq!(db.use_intrabc, 1, "§5.11.7 use_intrabc = 1");
    assert_eq!(db.is_inter, 1, "§5.11.7 intrabc arm: is_inter = 1");
    assert_eq!(db.y_mode, 0, "§5.11.7 intrabc arm: YMode = DC_PRED");
    assert_eq!(db.uv_mode, Some(0), "§5.11.7 intrabc arm: UVMode = DC_PRED");
    assert_eq!(
        db.mv[0],
        [0, -2560],
        "§5.11.26 Mv[0] = PredMv[0] + zero diff"
    );
    assert_eq!(db.ref_frame, [0, -1], "RefFrame = [INTRA_FRAME, NONE]");
    assert!(!db.is_compound, "slot 1 = NONE ⇒ !isCompound");

    // §5.11.5 grid stamps over the 2×2 footprint.
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..2usize {
        for c in 0..2usize {
            let cell = r * mi_cols + c;
            assert_eq!(walker.is_inters()[cell], 1, "IsInters stamp");
            assert_eq!(walker.y_modes()[cell], 0, "YModes = DC_PRED stamp");
            assert_eq!(walker.ref_frames()[cell * 2], 0, "RefFrames[0] stamp");
            assert_eq!(walker.ref_frames()[cell * 2 + 1], -1, "RefFrames[1] stamp");
            assert_eq!(
                walker.interp_filters()[cell * 2],
                BILINEAR,
                "InterpFilters[0] = BILINEAR stamp"
            );
            assert_eq!(
                walker.interp_filters()[cell * 2 + 1],
                BILINEAR,
                "InterpFilters[1] = BILINEAR stamp"
            );
            assert_eq!(walker.mvs()[(cell * 2) * 2], 0, "Mvs row component");
            assert_eq!(walker.mvs()[(cell * 2) * 2 + 1], -2560, "Mvs col component");
        }
    }

    assert_eq!(
        dec.read_literal(8).unwrap(),
        0x5A,
        "decoder must be positioned at the sentinel after the block"
    );
}

/// r281: §5.11.49 `palette_tokens()` wiring — a DC_PRED luma-palette
/// block (`PaletteSizeY = 2`, entries `[10, 200]`) makes
/// `decode_block_syntax` consume the `color_index_map_y`
/// `NS(PaletteSizeY)` literal plus the 63 anti-diagonal
/// `palette_color_idx_y` S() reads of the 8×8 block. The write side
/// mirrors the §5.11.49 walk with the same §5.11.50 colour-context
/// derivation.
#[test]
fn r281_decode_block_syntax_palette_block_consumes_palette_tokens() {
    let walker_w = walker_n(16);
    let mut enc_cdfs = TileCdfContext::new_from_defaults();
    let mut writer = SymbolWriter::new(false);
    let mut colors_y = [0u16; PALETTE_COLORS];
    colors_y[0] = 10;
    colors_y[1] = 200;
    let dummy = [0u16; PALETTE_COLORS];
    write_skip(&mut writer, &mut enc_cdfs, 1, 0, false).unwrap();
    write_intra_frame_else_arm(
        &mut writer,
        &mut enc_cdfs,
        BLOCK_8X8,
        /* y_mode = DC_PRED */ 0,
        /* uv_mode = DC_PRED */ Some(0),
        0,
        0,
        /* cfl_allowed = */ true,
        /* has_chroma = */ true,
        /* allow_screen_content_tools = */ true,
        /* enable_filter_intra = */ false,
        0,
        None,
        /* above_palette_y = */ false,
        /* left_palette_y = */ false,
        /* bit_depth = */ 8,
        /* has_palette_y = */ 1,
        /* palette_size_y = */ 2,
        &colors_y,
        /* has_palette_uv = */ 0,
        0,
        &dummy,
        &dummy,
        false,
        None,
        None,
        &walker_w,
        0,
        0,
        0,
        0,
    )
    .unwrap();

    // §5.11.49 palette_tokens mirror: `color_index_map_y NS(2) = 1`,
    // then the anti-diagonal walk with every other sample = palette
    // index 0. The write side maintains the same ColorMapY the reader
    // reconstructs and derives the identical §5.11.50 colour context
    // per position.
    writer.write_ns(2, 1).unwrap();
    let (bw, bh) = (8usize, 8usize);
    let mut color_map = vec![0u8; bw * bh];
    color_map[0] = 1;
    for i in 1..(bh + bw - 1) {
        let j_hi = i.min(bw - 1);
        let j_lo = i.saturating_sub(bh - 1);
        let mut j = j_hi;
        loop {
            let (r, c) = (i - j, j);
            let pcc = get_palette_color_context(&color_map, bw, r, c, 2).unwrap();
            let ctx = palette_color_ctx(pcc.color_context_hash).unwrap();
            let desired = 0u8;
            let idx = pcc
                .color_order
                .iter()
                .take(2)
                .position(|&v| v == desired)
                .expect("desired index must appear in ColorOrder");
            let row = enc_cdfs.palette_y_color_cdf(2, ctx).unwrap();
            writer.write_symbol(idx as u32, row).unwrap();
            color_map[r * bw + c] = desired;
            if j == j_lo {
                break;
            }
            j -= 1;
        }
    }
    writer.write_literal(8, 0xC3).unwrap();
    let bytes = writer.finish();

    let mut walker = walker_n(16);
    let mut cdfs = TileCdfContext::new_from_defaults();
    let mut dec = SymbolDecoder::init_symbol(&bytes, bytes.len(), false).unwrap();
    let lossless = [false; MAX_SEGMENTS];
    let db = walker
        .decode_block_syntax(
            &mut dec,
            &mut cdfs,
            0,
            0,
            BLOCK_8X8,
            /* frame_is_intra = */ true,
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
            /* allow_intrabc = */ false,
            0,
            false,
            false,
            0,
            false,
            false,
            false,
            0,
            /* allow_screen_content_tools = */ true,
            /* enable_filter_intra = */ false,
            /* bit_depth = */ 8,
            /* tx_mode_select = */ false,
            /* inter_ctx = */ None,
            /* quant = */ &oxideav_av1::QuantizerParams::neutral(0, 8),
            /* reduced_tx_set = */ false,
        )
        .expect("r281: the palette block must run §5.11.49 palette_tokens to completion");

    assert_eq!(db.y_mode, 0, "DC_PRED luma");
    assert_eq!(db.palette_size_y, 2, "§5.11.46 PaletteSizeY = 2");
    assert_eq!(db.palette_size_uv, 0, "§5.11.46 has_palette_uv = 0 arm");
    assert_eq!(
        db.use_filter_intra, None,
        "§5.11.24 outer gate closed by PaletteSizeY > 0 (and the disabled sequence bit)"
    );

    // §5.11.46 / §5.11.49 grid stamps over the 2×2 footprint.
    let mi_cols = walker.mi_cols() as usize;
    for r in 0..2usize {
        for c in 0..2usize {
            let cell = r * mi_cols + c;
            assert_eq!(walker.palette_sizes()[cell], 2, "PaletteSizes[0] stamp");
            assert_eq!(
                walker.palette_colors()[cell * PALETTE_COLORS],
                10,
                "PaletteColors[0][..][0] stamp"
            );
            assert_eq!(
                walker.palette_colors()[cell * PALETTE_COLORS + 1],
                200,
                "PaletteColors[0][..][1] stamp"
            );
        }
    }

    assert_eq!(
        dec.read_literal(8).unwrap(),
        0xC3,
        "decoder must be positioned at the sentinel after the palette walk"
    );
}

// ===========================================================================
// §5.11.3 clear_block_decoded_flags / §6.10.3 BlockDecoded queries
// ===========================================================================
//
// `BlockDecoded[ plane ][ y ][ x ]` (av1-spec p.60 / p.85 / p.405) is the
// per-superblock availability grid the §5.11.35 `predict_intra` invocation
// reads for above-right / below-left reference-sample selection. r318 wires
// the §5.11.3 reset + §5.11.35 per-TU stamp into the walker; these tests
// pin the reset's border-vs-interior pattern across the SB-size /
// num-planes / subsampling / frame-edge axes that drive the §5.11.3 body.

/// §5.11.3 base case: a 64×64 superblock (`sbSize4 = 16`) at the frame
/// origin with three planes, no subsampling, the SB strictly inside the
/// tile. The top border (`y == -1`) and left border (`x == -1`) are `1`
/// over the SB span; the interior and the off-SB tails are `0`; the
/// below-left corner (`[sbSize4][-1]`) is forced back to `0`.
#[test]
fn clear_block_decoded_flags_64x64_three_plane_no_subsampling() {
    // 32×32 mi tile (two 64×64 superblocks per axis) so the origin SB is
    // strictly interior (sbWidth4 / sbHeight4 both 32 > sbSize4 = 16).
    let mut walker = walker_n(32);
    let sb_size4 = 16u32;
    walker.clear_block_decoded_flags(0, 0, sb_size4, /* num_planes = */ 3, 0, 0);

    for plane in 0..3usize {
        // Top border (y == -1): `1` for x in 0..sbWidth4 (= 32 here), so
        // every in-SB column 0..=15 is available.
        for x in 0..=15i32 {
            assert!(
                walker.block_decoded(plane, -1, x),
                "plane {plane}: top border [-1][{x}] must be 1"
            );
        }
        // Left border (x == -1): `1` for y in 0..sbHeight4.
        for y in 0..=15i32 {
            assert!(
                walker.block_decoded(plane, y, -1),
                "plane {plane}: left border [{y}][-1] must be 1"
            );
        }
        // Top-left corner [-1][-1]: y < 0 && x (= -1) < sbWidth4 ⇒ 1.
        assert!(
            walker.block_decoded(plane, -1, -1),
            "plane {plane}: corner [-1][-1] must be 1"
        );
        // Interior must be 0 (nothing decoded yet).
        for y in 0..=15i32 {
            for x in 0..=15i32 {
                assert!(
                    !walker.block_decoded(plane, y, x),
                    "plane {plane}: interior [{y}][{x}] must be 0"
                );
            }
        }
        // §5.11.3 final line: below-left corner [sbSize4][-1] forced 0.
        assert!(
            !walker.block_decoded(plane, sb_size4 as i32, -1),
            "plane {plane}: below-left corner [{sb_size4}][-1] must be 0"
        );
        // The bottom border row [sbSize4][x] for x >= 0 is also 0.
        assert!(
            !walker.block_decoded(plane, sb_size4 as i32, 0),
            "plane {plane}: bottom border [{sb_size4}][0] must be 0"
        );
    }
}

/// §5.11.3 monochrome: `num_planes = 1` ⇒ only plane 0 is reset. Planes 1
/// and 2 are never touched (stay at their construction-time `0`).
#[test]
fn clear_block_decoded_flags_monochrome_only_luma() {
    let mut walker = walker_n(32);
    walker.clear_block_decoded_flags(0, 0, 16, /* num_planes = */ 1, 0, 0);
    // Plane 0 border set.
    assert!(walker.block_decoded(0, -1, 0), "luma top border must be 1");
    // Planes 1 / 2 untouched (the §5.11.3 loop bound `NumPlanes = 1`
    // never enters the chroma iterations).
    assert!(
        !walker.block_decoded(1, -1, 0),
        "chroma plane 1 must stay 0 in monochrome"
    );
    assert!(
        !walker.block_decoded(2, -1, 0),
        "chroma plane 2 must stay 0 in monochrome"
    );
}

/// §5.11.3 chroma subsampling: with `subsampling_x = subsampling_y = 1`
/// (4:2:0) the chroma planes' `y_max` / `x_max` are `sbSize4 >> 1 = 8`
/// and their border extents (`sbWidth4 >> 1` / `sbHeight4 >> 1`) shrink
/// accordingly. Luma (plane 0) is unaffected by subsampling.
#[test]
fn clear_block_decoded_flags_420_chroma_subsampled_extent() {
    let mut walker = walker_n(32);
    walker.clear_block_decoded_flags(0, 0, 16, 3, /* sub_x = */ 1, /* sub_y = */ 1);

    // Luma border spans 0..=15 (subX = 0 on plane 0).
    assert!(walker.block_decoded(0, -1, 15), "luma top border [-1][15]");
    // Chroma top border spans only 0..=7 (the SB is 8 chroma-4x4 wide).
    for x in 0..=7i32 {
        assert!(
            walker.block_decoded(1, -1, x),
            "chroma top border [-1][{x}] must be 1"
        );
    }
    // Chroma below-left corner [sbSize4 >> subY = 8][-1] forced 0.
    assert!(
        !walker.block_decoded(1, 8, -1),
        "chroma below-left corner [8][-1] must be 0"
    );
    // Chroma left border spans 0..=7.
    assert!(walker.block_decoded(2, 7, -1), "chroma left border [7][-1]");
}

/// §5.11.3 128×128 superblock: `sbSize4 = 32`. The luma grid spans the
/// full SB (border at -1, interior 0..=31, below-left corner forced 0).
/// Exercises the upper bound of the fixed `BD_STRIDE = 34` storage.
#[test]
fn clear_block_decoded_flags_128x128_full_span() {
    // 64×64 mi tile (one 128×128 superblock per axis edge plus room).
    let mut walker = walker_n(64);
    let sb_size4 = 32u32;
    walker.clear_block_decoded_flags(0, 0, sb_size4, 3, 0, 0);

    assert!(walker.block_decoded(0, -1, 31), "luma top border [-1][31]");
    assert!(walker.block_decoded(0, 31, -1), "luma left border [31][-1]");
    assert!(
        !walker.block_decoded(0, 31, 31),
        "luma interior far corner [31][31] must be 0"
    );
    assert!(
        !walker.block_decoded(0, sb_size4 as i32, -1),
        "luma below-left corner [32][-1] must be 0"
    );
}

/// §5.11.3 partial superblock at the frame's bottom-right edge: when the
/// SB straddles `MiColEnd` / `MiRowEnd`, `sbWidth4` / `sbHeight4` are
/// smaller than `sbSize4`, so the top / left border `1`-run is truncated
/// to the in-frame extent. With a 24×24 tile and a 64×64 SB at (0,0),
/// `sbWidth4 = sbHeight4 = 24` but `sbSize4 = 16`, so the border still
/// fully covers the SB (24 > 16) — instead place the SB so only part of
/// the row is in-tile by using an 8×8 tile (sbWidth4 = 8 < sbSize4 = 16).
#[test]
fn clear_block_decoded_flags_partial_sb_truncates_border() {
    // 8×8 mi tile, 64×64 superblock (sbSize4 = 16) at the origin: the SB
    // is larger than the tile, so sbWidth4 = sbHeight4 = 8.
    let mut walker = walker_n(8);
    walker.clear_block_decoded_flags(0, 0, 16, 3, 0, 0);

    // Top border `1` only for x in 0..sbWidth4 = 0..8.
    for x in 0..=7i32 {
        assert!(
            walker.block_decoded(0, -1, x),
            "in-tile top border [-1][{x}] must be 1"
        );
    }
    // Beyond sbWidth4 the top border falls to the `else` arm ⇒ 0.
    for x in 8..=15i32 {
        assert!(
            !walker.block_decoded(0, -1, x),
            "off-tile top border [-1][{x}] must be 0"
        );
    }
    // Left border `1` only for y in 0..sbHeight4 = 0..8.
    for y in 0..=7i32 {
        assert!(
            walker.block_decoded(0, y, -1),
            "in-tile left border [{y}][-1] must be 1"
        );
    }
    for y in 8..=15i32 {
        assert!(
            !walker.block_decoded(0, y, -1),
            "off-tile left border [{y}][-1] must be 0"
        );
    }
}

/// §6.10.3 query bounds: an out-of-range `(plane, y, x)` returns `false`
/// rather than panicking (the §5.11.35 derivation never queries outside
/// the SB + one-cell border, but the accessor is defensive).
#[test]
fn block_decoded_out_of_range_returns_false() {
    let walker = walker_n(16);
    assert!(!walker.block_decoded(3, 0, 0), "plane 3 out of range");
    assert!(!walker.block_decoded(0, -2, 0), "y = -2 below border");
    assert!(!walker.block_decoded(0, 0, -2), "x = -2 below border");
    assert!(!walker.block_decoded(0, 33, 0), "y = 33 past BD_STRIDE");
    assert!(!walker.block_decoded(0, 0, 33), "x = 33 past BD_STRIDE");
}
