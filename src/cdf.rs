//! Default CDF tables (§9.4) and the §8.3.1 / §8.3.2 CDF-selection
//! process for a bounded **intra-frame mode / partition** syntax group.
//!
//! The §8.2 [`crate::symbol_decoder::SymbolDecoder`] decodes a symbol
//! against a *caller-supplied* CDF array. The bytes of that array — and
//! the rule that maps a syntax-element name to the right array slot — are
//! the subject of this module:
//!
//!   * **§9.4 default tables.** The spec lists, in the "Additional
//!     tables" annex, the initial cumulative-distribution values copied
//!     into each `Tile*Cdf` array at the start of tile parsing
//!     (§8.3.1). Each array is stored with a trailing entry that
//!     [`crate::symbol_decoder::SymbolDecoder::read_symbol`] uses as the
//!     §8.3 adaptation counter (it starts at `0`), so a row of length
//!     `N + 1` codes a symbol with `N` possible values, with
//!     `row[N-1] == 1 << 15` and `row[N]` the counter.
//!
//!   * **§8.3.1 init-from-defaults.** At tile start every `Tile*Cdf`
//!     array "is set equal to a copy of" its `Default_*_Cdf` table.
//!     [`TileCdfContext::new_from_defaults`] performs exactly that copy,
//!     producing a per-tile, mutable working set the symbol decoder
//!     adapts in place.
//!
//!   * **§8.3.2 selection.** Given a syntax element and the surrounding
//!     block context, §8.3.2 derives which row of which `Tile*Cdf` array
//!     is the `cdf` passed to `read_symbol`. This module implements the
//!     selection for the subset it carries: `intra_frame_y_mode`,
//!     `partition`, `skip`, and `segment_id`.
//!
//! ## Scope (bounded subset)
//!
//! Two §9.4 groups currently land here:
//!
//!   * **Intra-frame mode / partition** (round 16):
//!       * `Default_Intra_Frame_Y_Mode_Cdf` (`intra_frame_y_mode`)
//!       * `Default_Partition_W8/W16/W32/W64/W128_Cdf` (`partition`)
//!       * `Default_Skip_Cdf` (`skip`)
//!       * `Default_Segment_Id_Cdf` (`segment_id`)
//!
//!   * **Motion-vector component** (round 17):
//!       * `Default_Mv_Joint_Cdf` (`mv_joint`)
//!       * `Default_Mv_Sign_Cdf` (`mv_sign`)
//!       * `Default_Mv_Class_Cdf` (`mv_class`)
//!       * `Default_Mv_Class0_Bit_Cdf` (`mv_class0_bit`)
//!       * `Default_Mv_Class0_Fr_Cdf` (`mv_class0_fr`)
//!       * `Default_Mv_Class0_Hp_Cdf` (`mv_class0_hp`)
//!       * `Default_Mv_Bit_Cdf` (`mv_bit`)
//!       * `Default_Mv_Fr_Cdf` (`mv_fr`)
//!       * `Default_Mv_Hp_Cdf` (`mv_hp`)
//!
//!   * **Inter-mode / reference-frame** (round 18): `new_mv`,
//!     `zero_mv`, `ref_mv`, `drl_mode`, `is_inter`, `comp_mode`,
//!     `skip_mode`, `comp_ref`, `comp_bwdref`, `single_ref`,
//!     `compound_mode`, `comp_ref_type`, `uni_comp_ref`.
//!
//!   * **Palette / filter-intra / CFL** (round 19):
//!       * `Default_Filter_Intra_Mode_Cdf` (`filter_intra_mode`)
//!       * `Default_Filter_Intra_Cdf` (`use_filter_intra`)
//!       * `Default_Palette_Y_Mode_Cdf` (`has_palette_y`)
//!       * `Default_Palette_Uv_Mode_Cdf` (`has_palette_uv`)
//!       * `Default_Palette_Y_Size_Cdf` (`palette_size_y_minus_2`)
//!       * `Default_Palette_Uv_Size_Cdf` (`palette_size_uv_minus_2`)
//!       * `Default_Palette_Size_{2..8}_Y_Color_Cdf` (`palette_color_idx_y`)
//!       * `Default_Palette_Size_{2..8}_Uv_Color_Cdf` (`palette_color_idx_uv`)
//!       * `Default_Cfl_Sign_Cdf` (`cfl_alpha_signs`)
//!       * `Default_Cfl_Alpha_Cdf` (`cfl_alpha_u` / `cfl_alpha_v`)
//!
//!   * **Transform-size** (round 20):
//!       * `Default_Tx_8x8_Cdf` (`tx_depth`, `maxTxDepth == 1`)
//!       * `Default_Tx_16x16_Cdf` (`tx_depth`, `maxTxDepth == 2`)
//!       * `Default_Tx_32x32_Cdf` (`tx_depth`, `maxTxDepth == 3`)
//!       * `Default_Tx_64x64_Cdf` (`tx_depth`, `maxTxDepth == 4`)
//!       * `Default_Txfm_Split_Cdf` (`txfm_split`)
//!
//!   * **Inter-frame transform-type** (round 21):
//!       * `Default_Inter_Tx_Type_Set1_Cdf` (`inter_tx_type`,
//!         `TX_SET_INTER_1`, 4x4 / 8x8 inter blocks)
//!       * `Default_Inter_Tx_Type_Set2_Cdf` (`inter_tx_type`,
//!         `TX_SET_INTER_2`, 16x16 inter blocks)
//!       * `Default_Inter_Tx_Type_Set3_Cdf` (`inter_tx_type`,
//!         `TX_SET_INTER_3`, 4x4..32x32 reduced-set inter blocks)
//!
//! The remaining `Default_*_Cdf` arrays of §9.4 (the y_mode,
//! uv_mode, angle-delta, intra transform-type (`intra_tx_type`,
//! `Default_Intra_Tx_Type_Set{1,2}_Cdf`), coefficient,
//! interpolation, motion-mode, … groups), the `init_coeff_cdfs`
//! coefficient tables, and the §8.3.2 `split_or_horz` /
//! `split_or_vert` / `intra_tx_type` / … selections are a clear
//! followup: each is a mechanical transcription of one §9.4 table
//! plus its §8.3.2 paragraph, slotted into the same
//! [`TileCdfContext`] shape used here.
//!
//! All values are transcribed directly from `docs/video/av1/av1-spec`
//! §8.3 and §9.4 — no external source consulted.

// ---------------------------------------------------------------------
// §3 / §9.3 symbol constants used to dimension the tables below.
// ---------------------------------------------------------------------

/// `INTRA_MODES` (§9.3) — number of values for `y_mode` (and the first /
/// second dimension index range of intra-mode CDFs).
pub const INTRA_MODES: usize = 13;

/// `INTRA_MODE_CONTEXTS` (§9.3) — number of each of the left and above
/// contexts for `intra_frame_y_mode`.
pub const INTRA_MODE_CONTEXTS: usize = 5;

/// `PARTITION_CONTEXTS` (§9.3) — number of contexts when decoding
/// `partition`.
pub const PARTITION_CONTEXTS: usize = 4;

/// `SKIP_CONTEXTS` (§9.3) — number of contexts for decoding `skip`.
pub const SKIP_CONTEXTS: usize = 3;

/// `SEGMENT_ID_CONTEXTS` (§9.3) — number of contexts for `segment_id`.
pub const SEGMENT_ID_CONTEXTS: usize = 3;

/// `MAX_SEGMENTS` (§9.3) — number of segments allowed in the
/// segmentation map (number of `segment_id` symbol values).
pub const MAX_SEGMENTS: usize = 8;

/// `Intra_Mode_Context[ INTRA_MODES ]` (§8.3.2) — maps a neighbouring
/// block's `YMode` to the above/left context index used to select the
/// `intra_frame_y_mode` CDF.
pub const INTRA_MODE_CONTEXT: [usize; INTRA_MODES] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

// ---------------------------------------------------------------------
// §3 motion-vector constants (round 17).
// ---------------------------------------------------------------------

/// `MV_CONTEXTS` (§3) — number of contexts for decoding motion vectors.
/// The §5.11.31 `read_mv()` derivation sets `MvCtx = MV_INTRABC_CONTEXT`
/// for intra-block-copy use and `MvCtx = 0` otherwise, so an MvCtx value
/// addresses one of `0..MV_CONTEXTS` (with `MV_INTRABC_CONTEXT = 1`
/// hitting the second slot).
pub const MV_CONTEXTS: usize = 2;

/// `MV_INTRABC_CONTEXT` (§3) — motion-vector context used by §5.11.31
/// `read_mv()` when `use_intrabc == 1`.
pub const MV_INTRABC_CONTEXT: usize = 1;

/// `MV_JOINTS` (§3) — number of values for `mv_joint`
/// (`MV_JOINT_ZERO`, `MV_JOINT_HNZVZ`, `MV_JOINT_HZVNZ`, `MV_JOINT_HNZVNZ`).
pub const MV_JOINTS: usize = 4;

/// `MV_CLASSES` (§3) — number of values for `mv_class`.
pub const MV_CLASSES: usize = 11;

/// `CLASS0_SIZE` (§3) — number of values for `mv_class0_bit`. Also the
/// inner dimension of `Default_Mv_Class0_Fr_Cdf`.
pub const CLASS0_SIZE: usize = 2;

/// `MV_OFFSET_BITS` (§3) — maximum number of `mv_bit` slots read by
/// `read_mv_component()` (one per `i = 0..mv_class-1`).
pub const MV_OFFSET_BITS: usize = 10;

/// Number of distinct mv components per call: the §5.11.31 motion vector
/// has a horizontal and vertical component (`comp = 0..1`).
pub const MV_COMPS: usize = 2;

// ---------------------------------------------------------------------
// §3 inter-mode / reference-frame constants (round 18).
// ---------------------------------------------------------------------

/// `NEW_MV_CONTEXTS` (§3) — number of contexts for `new_mv`.
pub const NEW_MV_CONTEXTS: usize = 6;

/// `ZERO_MV_CONTEXTS` (§3) — number of contexts for `zero_mv`.
pub const ZERO_MV_CONTEXTS: usize = 2;

/// `REF_MV_CONTEXTS` (§3) — number of contexts for `ref_mv`.
pub const REF_MV_CONTEXTS: usize = 6;

/// `DRL_MODE_CONTEXTS` (§3) — number of contexts for `drl_mode`.
pub const DRL_MODE_CONTEXTS: usize = 3;

/// `IS_INTER_CONTEXTS` (§3) — number of contexts for `is_inter`.
pub const IS_INTER_CONTEXTS: usize = 4;

/// `COMP_INTER_CONTEXTS` (§3) — number of contexts for `comp_mode`.
pub const COMP_INTER_CONTEXTS: usize = 5;

/// `SKIP_MODE_CONTEXTS` (§3) — number of contexts for `skip_mode`.
pub const SKIP_MODE_CONTEXTS: usize = 3;

/// `REF_CONTEXTS` (§3) — number of contexts for `single_ref`, `comp_ref`,
/// `comp_bwdref`, and `uni_comp_ref`.
pub const REF_CONTEXTS: usize = 3;

/// `FWD_REFS` (§3) — number of forward reference syntax elements (the
/// `Default_Comp_Ref_Cdf` second dimension is `FWD_REFS - 1`).
pub const FWD_REFS: usize = 4;

/// `BWD_REFS` (§3) — number of backward reference syntax elements (the
/// `Default_Comp_Bwd_Ref_Cdf` second dimension is `BWD_REFS - 1`).
pub const BWD_REFS: usize = 3;

/// `SINGLE_REFS` (§3) — number of single-reference syntax elements (the
/// `Default_Single_Ref_Cdf` second dimension is `SINGLE_REFS - 1`).
pub const SINGLE_REFS: usize = 7;

/// `UNIDIR_COMP_REFS` (§3) — number of unidirectional-compound reference
/// syntax elements (the `Default_Uni_Comp_Ref_Cdf` second dimension is
/// `UNIDIR_COMP_REFS - 1`).
pub const UNIDIR_COMP_REFS: usize = 4;

/// `COMP_REF_TYPE_CONTEXTS` (§3) — number of contexts for `comp_ref_type`.
pub const COMP_REF_TYPE_CONTEXTS: usize = 5;

/// `COMPOUND_MODES` (§3) — number of values for `compound_mode`.
pub const COMPOUND_MODES: usize = 8;

/// `COMPOUND_MODE_CONTEXTS` (§3) — number of contexts for `compound_mode`.
pub const COMPOUND_MODE_CONTEXTS: usize = 8;

/// `COMP_NEWMV_CTXS` (§3) — number of new-mv values used when
/// constructing the compound-mode context (the second axis of
/// `Compound_Mode_Ctx_Map`).
pub const COMP_NEWMV_CTXS: usize = 5;

/// `Compound_Mode_Ctx_Map[ 3 ][ COMP_NEWMV_CTXS ]` (§8.3.2) — maps the
/// `(RefMvContext >> 1, Min(NewMvContext, COMP_NEWMV_CTXS - 1))` pair to
/// the `compound_mode` context index used to select the
/// `TileCompoundModeCdf` row.
pub const COMPOUND_MODE_CTX_MAP: [[usize; COMP_NEWMV_CTXS]; 3] =
    [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]];

// ---------------------------------------------------------------------
// §3 palette / filter-intra / CFL constants (round 19).
// ---------------------------------------------------------------------

/// `BLOCK_SIZES` (§3) — number of different block sizes. Also the first
/// dimension of `Default_Filter_Intra_Cdf` (indexed by `MiSize`). Per the
/// §9.4 note, first-dimension indices 10..=15 and 20..=21 are never
/// reached by the `use_filter_intra` selection but the table is still
/// transcribed full-width.
pub const BLOCK_SIZES: usize = 22;

/// `INTRA_FILTER_MODES` (§3) — number of values for `filter_intra_mode`
/// (the length of `Default_Filter_Intra_Mode_Cdf` is this + 1).
pub const INTRA_FILTER_MODES: usize = 5;

/// `PALETTE_BLOCK_SIZE_CONTEXTS` (§3) — number of `bsizeCtx` values for
/// palette block size. `bsizeCtx = Mi_Width_Log2[ MiSize ] +
/// Mi_Height_Log2[ MiSize ] - 2` (§5.11.46), in `0..PALETTE_BLOCK_SIZE_CONTEXTS`.
pub const PALETTE_BLOCK_SIZE_CONTEXTS: usize = 7;

/// `PALETTE_Y_MODE_CONTEXTS` (§3) — number of `has_palette_y` contexts.
pub const PALETTE_Y_MODE_CONTEXTS: usize = 3;

/// `PALETTE_UV_MODE_CONTEXTS` (§3) — number of `has_palette_uv` contexts.
pub const PALETTE_UV_MODE_CONTEXTS: usize = 2;

/// `PALETTE_SIZES` (§3) — number of values for `palette_size`
/// (`palette_size_y_minus_2` / `palette_size_uv_minus_2` code this many
/// symbols; the cumulative-array length is this + 1).
pub const PALETTE_SIZES: usize = 7;

/// `PALETTE_COLORS` (§3) — maximum palette size (the largest
/// `palette_color_idx_*` symbol count). The per-size colour-index CDFs
/// run from size 2 to size `PALETTE_COLORS` inclusive.
pub const PALETTE_COLORS: usize = 8;

/// `PALETTE_COLOR_CONTEXTS` (§3) — number of `ctx` values for
/// `palette_color_idx_*` (the colour-index CDFs' first dimension).
pub const PALETTE_COLOR_CONTEXTS: usize = 5;

/// `PALETTE_NUM_NEIGHBORS` (§3) — number of neighbours considered when
/// computing the palette colour-context hash.
pub const PALETTE_NUM_NEIGHBORS: usize = 3;

/// `PALETTE_MAX_COLOR_CONTEXT_HASH` (§3) — number of distinct colour
/// context hash values (the `Palette_Color_Context` map is indexed by
/// `0..=PALETTE_MAX_COLOR_CONTEXT_HASH`).
pub const PALETTE_MAX_COLOR_CONTEXT_HASH: usize = 8;

/// `Palette_Color_Context[ PALETTE_MAX_COLOR_CONTEXT_HASH + 1 ]`
/// (§9.4 additional tables) — maps a `ColorContextHash` to the
/// `palette_color_idx_*` context. The `-1` entries are hashes that the
/// §5.11.50 derivation never produces, so they are never accessed; they
/// are stored as `i8` to carry the spec's sentinel verbatim.
pub const PALETTE_COLOR_CONTEXT: [i8; PALETTE_MAX_COLOR_CONTEXT_HASH + 1] =
    [-1, -1, 0, -1, -1, 4, 3, 2, 1];

/// `Palette_Color_Hash_Multipliers[ PALETTE_NUM_NEIGHBORS ]`
/// (§9.4 additional tables) — the per-neighbour weights summed into the
/// `ColorContextHash`.
pub const PALETTE_COLOR_HASH_MULTIPLIERS: [u32; PALETTE_NUM_NEIGHBORS] = [1, 2, 2];

/// `CFL_JOINT_SIGNS` (§3) — number of values for `cfl_alpha_signs` (the
/// `Default_Cfl_Sign_Cdf` cumulative-array length is this + 1).
pub const CFL_JOINT_SIGNS: usize = 8;

/// `CFL_ALPHABET_SIZE` (§3) — number of values for `cfl_alpha_u` /
/// `cfl_alpha_v` (the `Default_Cfl_Alpha_Cdf` inner length is this + 1).
pub const CFL_ALPHABET_SIZE: usize = 16;

/// `CFL_ALPHA_CONTEXTS` (§3) — number of `ctx` values for `cfl_alpha_u` /
/// `cfl_alpha_v` (the `Default_Cfl_Alpha_Cdf` first dimension).
pub const CFL_ALPHA_CONTEXTS: usize = 6;

/// `TX_SIZE_CONTEXTS` (§3) — number of contexts when decoding
/// `tx_depth`. Drives the first dimension of every
/// `Default_Tx_{8,16,32,64}x{8,16,32,64}_Cdf` table (§9.4) and is the
/// number of values returned by the §8.3.2 `tx_depth` context formula.
pub const TX_SIZE_CONTEXTS: usize = 3;

/// `TX_SIZES` (§3) — number of square transform sizes (`TX_4X4`,
/// `TX_8X8`, `TX_16X16`, `TX_32X32`, `TX_64X64`).
pub const TX_SIZES: usize = 5;

/// `MAX_TX_DEPTH` (§3) — maximum number of times the transform can
/// be split. Drives the §9.4 row widths for the per-`maxRectTxSize`
/// `tx_depth` CDFs: `Default_Tx_8x8_Cdf` rows are length
/// `MAX_TX_DEPTH + 1` (one symbol of 2 values, since the spec
/// constraint `tx_depth in 0..=2` collapses to two choices for the
/// `maxTxDepth == 1` block-size group), while every other
/// `Default_Tx_*Cdf` row is `MAX_TX_DEPTH + 2` (three choices) per
/// the §9.4 source.
pub const MAX_TX_DEPTH: usize = 2;

/// `TXFM_PARTITION_CONTEXTS` (§3) — number of contexts when decoding
/// `txfm_split`. Drives the first dimension of
/// `Default_Txfm_Split_Cdf` (§9.4).
pub const TXFM_PARTITION_CONTEXTS: usize = 21;

/// `TX_TYPES` (§3) — total number of inverse transform types in the
/// `Tx_Type_Inter_Inv_Set1` enumeration (the §6.10.19 full set). Drives
/// the symbol-count of [`DEFAULT_INTER_TX_TYPE_SET1_CDF`] (16 cumulative
/// frequencies + one trailing adaptation counter).
pub const TX_TYPES: usize = 16;

/// `TX_TYPES_SET2` — symbol count for §6.10.19 `TX_SET_INTER_2`
/// (`Tx_Type_Inter_Inv_Set2` has 12 entries). Drives the row width of
/// [`DEFAULT_INTER_TX_TYPE_SET2_CDF`].
pub const TX_TYPES_SET2: usize = 12;

/// `TX_TYPES_SET3` — symbol count for §6.10.19 `TX_SET_INTER_3`
/// (`Tx_Type_Inter_Inv_Set3 = { IDTX, DCT_DCT }`, two entries). Drives
/// the row width of [`DEFAULT_INTER_TX_TYPE_SET3_CDF`].
pub const TX_TYPES_SET3: usize = 2;

/// `TX_SET_DCTONLY` (§6.10.19) — the trivial transform-set returned by
/// §5.11.48 `get_tx_set()` when the block is too large or
/// `reduced_tx_set` collapses everything to DCT_DCT. Listed for
/// completeness alongside the inter-tx-type sets; no CDF row is read in
/// this case (§5.11.47 forces `TxType = DCT_DCT`).
pub const TX_SET_DCTONLY: u32 = 0;

/// `TX_SET_INTER_1` (§6.10.19) — the full-inter transform set
/// (`Tx_Type_Inter_Inv_Set1`, 16 symbols), selected when neither
/// `reduced_tx_set` nor `txSzSqrUp == TX_32X32` apply and
/// `txSzSqr != TX_16X16`. Returned by [`inter_tx_type_set`].
pub const TX_SET_INTER_1: u32 = 1;

/// `TX_SET_INTER_2` (§6.10.19) — the 16x16 inter transform set
/// (`Tx_Type_Inter_Inv_Set2`, 12 symbols), selected when
/// `txSzSqr == TX_16X16` and `reduced_tx_set == 0`. Returned by
/// [`inter_tx_type_set`].
pub const TX_SET_INTER_2: u32 = 2;

/// `TX_SET_INTER_3` (§6.10.19) — the reduced inter transform set
/// (`Tx_Type_Inter_Inv_Set3 = { IDTX, DCT_DCT }`), selected when
/// `reduced_tx_set == 1` or `txSzSqrUp == TX_32X32`. Returned by
/// [`inter_tx_type_set`].
pub const TX_SET_INTER_3: u32 = 3;

/// First-axis length of [`DEFAULT_INTER_TX_TYPE_SET1_CDF`] — two entries
/// for `Tx_Size_Sqr[ txSz ] ∈ { TX_4X4, TX_8X8 }`, the only sizes that
/// reach `TX_SET_INTER_1` (per §5.11.48 `get_tx_set()`: the function
/// already returns `TX_SET_INTER_2` for `txSzSqr == TX_16X16` and
/// `TX_SET_INTER_3` for `txSzSqrUp == TX_32X32`).
pub const INTER_TX_TYPE_SET1_SIZES: usize = 2;

/// First-axis length of [`DEFAULT_INTER_TX_TYPE_SET3_CDF`] — four
/// entries for `Tx_Size_Sqr[ txSz ] ∈ { TX_4X4, TX_8X8, TX_16X16,
/// TX_32X32 }`, the reachable sizes for the reduced inter-tx-type set
/// (per §5.11.48: `txSzSqrUp > TX_32X32` already returns
/// `TX_SET_DCTONLY`).
pub const INTER_TX_TYPE_SET3_SIZES: usize = 4;

/// `INTERP_FILTERS` per §3. Number of values for `interp_filter` (the
/// three switchable filters: `EIGHTTAP`, `EIGHTTAP_SMOOTH`,
/// `EIGHTTAP_SHARP` — `BILINEAR` is reachable only when the frame
/// header's `interpolation_filter` is `BILINEAR` and the §5.11.x
/// switchable-filter path is not entered).
pub const INTERP_FILTERS: usize = 3;

/// `INTERP_FILTER_CONTEXTS` per §3. Number of contexts for
/// `interp_filter`. The §8.3.2 ctx formula (`((dir & 1) * 2 +
/// (RefFrame[1] > INTRA_FRAME)) * 4 + ...`) ranges across
/// `0..INTERP_FILTER_CONTEXTS`.
pub const INTERP_FILTER_CONTEXTS: usize = 16;

// ---------------------------------------------------------------------
// §9.4 default CDF tables (the intra-frame mode / partition subset).
//
// Each innermost array has length `N + 1`: `N` cumulative frequencies
// (the last of which is `1 << 15 == 32768`) followed by the §8.3
// adaptation counter, which starts at 0.
// ---------------------------------------------------------------------

/// `Default_Intra_Frame_Y_Mode_Cdf[ INTRA_MODE_CONTEXTS ][ INTRA_MODE_CONTEXTS ][ INTRA_MODES + 1 ]`
/// (§9.4). Indexed `[abovemode][leftmode]` (§8.3.2).
pub const DEFAULT_INTRA_FRAME_Y_MODE_CDF: [[[u16; INTRA_MODES + 1]; INTRA_MODE_CONTEXTS];
    INTRA_MODE_CONTEXTS] = [
    [
        [
            15588, 17027, 19338, 20218, 20682, 21110, 21825, 23244, 24189, 28165, 29093, 30466,
            32768, 0,
        ],
        [
            12016, 18066, 19516, 20303, 20719, 21444, 21888, 23032, 24434, 28658, 30172, 31409,
            32768, 0,
        ],
        [
            10052, 10771, 22296, 22788, 23055, 23239, 24133, 25620, 26160, 29336, 29929, 31567,
            32768, 0,
        ],
        [
            14091, 15406, 16442, 18808, 19136, 19546, 19998, 22096, 24746, 29585, 30958, 32462,
            32768, 0,
        ],
        [
            12122, 13265, 15603, 16501, 18609, 20033, 22391, 25583, 26437, 30261, 31073, 32475,
            32768, 0,
        ],
    ],
    [
        [
            10023, 19585, 20848, 21440, 21832, 22760, 23089, 24023, 25381, 29014, 30482, 31436,
            32768, 0,
        ],
        [
            5983, 24099, 24560, 24886, 25066, 25795, 25913, 26423, 27610, 29905, 31276, 31794,
            32768, 0,
        ],
        [
            7444, 12781, 20177, 20728, 21077, 21607, 22170, 23405, 24469, 27915, 29090, 30492,
            32768, 0,
        ],
        [
            8537, 14689, 15432, 17087, 17408, 18172, 18408, 19825, 24649, 29153, 31096, 32210,
            32768, 0,
        ],
        [
            7543, 14231, 15496, 16195, 17905, 20717, 21984, 24516, 26001, 29675, 30981, 31994,
            32768, 0,
        ],
    ],
    [
        [
            12613, 13591, 21383, 22004, 22312, 22577, 23401, 25055, 25729, 29538, 30305, 32077,
            32768, 0,
        ],
        [
            9687, 13470, 18506, 19230, 19604, 20147, 20695, 22062, 23219, 27743, 29211, 30907,
            32768, 0,
        ],
        [
            6183, 6505, 26024, 26252, 26366, 26434, 27082, 28354, 28555, 30467, 30794, 32086,
            32768, 0,
        ],
        [
            10718, 11734, 14954, 17224, 17565, 17924, 18561, 21523, 23878, 28975, 30287, 32252,
            32768, 0,
        ],
        [
            9194, 9858, 16501, 17263, 18424, 19171, 21563, 25961, 26561, 30072, 30737, 32463,
            32768, 0,
        ],
    ],
    [
        [
            12602, 14399, 15488, 18381, 18778, 19315, 19724, 21419, 25060, 29696, 30917, 32409,
            32768, 0,
        ],
        [
            8203, 13821, 14524, 17105, 17439, 18131, 18404, 19468, 25225, 29485, 31158, 32342,
            32768, 0,
        ],
        [
            8451, 9731, 15004, 17643, 18012, 18425, 19070, 21538, 24605, 29118, 30078, 32018,
            32768, 0,
        ],
        [
            7714, 9048, 9516, 16667, 16817, 16994, 17153, 18767, 26743, 30389, 31536, 32528, 32768,
            0,
        ],
        [
            8843, 10280, 11496, 15317, 16652, 17943, 19108, 22718, 25769, 29953, 30983, 32485,
            32768, 0,
        ],
    ],
    [
        [
            12578, 13671, 15979, 16834, 19075, 20913, 22989, 25449, 26219, 30214, 31150, 32477,
            32768, 0,
        ],
        [
            9563, 13626, 15080, 15892, 17756, 20863, 22207, 24236, 25380, 29653, 31143, 32277,
            32768, 0,
        ],
        [
            8356, 8901, 17616, 18256, 19350, 20106, 22598, 25947, 26466, 29900, 30523, 32261,
            32768, 0,
        ],
        [
            10835, 11815, 13124, 16042, 17018, 18039, 18947, 22753, 24615, 29489, 30883, 32482,
            32768, 0,
        ],
        [
            7618, 8288, 9859, 10509, 15386, 18657, 22903, 28776, 29180, 31355, 31802, 32593, 32768,
            0,
        ],
    ],
];

/// `Default_Partition_W8_Cdf[ PARTITION_CONTEXTS ][ 5 ]` (§9.4). Codes a
/// 4-value symbol (`PARTITION_NONE/HORZ/VERT/SPLIT`).
pub const DEFAULT_PARTITION_W8_CDF: [[u16; 5]; PARTITION_CONTEXTS] = [
    [19132, 25510, 30392, 32768, 0],
    [13928, 19855, 28540, 32768, 0],
    [12522, 23679, 28629, 32768, 0],
    [9896, 18783, 25853, 32768, 0],
];

/// `Default_Partition_W16_Cdf[ PARTITION_CONTEXTS ][ 11 ]` (§9.4). Codes
/// a 10-value symbol (the full `EXT_PARTITION_TYPES`).
pub const DEFAULT_PARTITION_W16_CDF: [[u16; 11]; PARTITION_CONTEXTS] = [
    [
        15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902, 32768, 0,
    ],
    [
        7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834, 32768, 0,
    ],
    [
        5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117, 32768, 0,
    ],
    [
        2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171, 32768, 0,
    ],
];

/// `Default_Partition_W32_Cdf[ PARTITION_CONTEXTS ][ 11 ]` (§9.4).
pub const DEFAULT_PARTITION_W32_CDF: [[u16; 11]; PARTITION_CONTEXTS] = [
    [
        18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544, 32768, 0,
    ],
    [
        7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052, 32768, 0,
    ],
    [
        6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047, 32768, 0,
    ],
    [
        1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899, 32768, 0,
    ],
];

/// `Default_Partition_W64_Cdf[ PARTITION_CONTEXTS ][ 11 ]` (§9.4).
pub const DEFAULT_PARTITION_W64_CDF: [[u16; 11]; PARTITION_CONTEXTS] = [
    [
        20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724, 32768, 0,
    ],
    [
        6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104, 32768, 0,
    ],
    [
        5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238, 32768, 0,
    ],
    [
        870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332, 32768, 0,
    ],
];

/// `Default_Partition_W128_Cdf[ PARTITION_CONTEXTS ][ 9 ]` (§9.4). The
/// 128×128 superblock omits the two `*_4` partitions, so the symbol has
/// 8 values.
pub const DEFAULT_PARTITION_W128_CDF: [[u16; 9]; PARTITION_CONTEXTS] = [
    [27899, 28219, 28529, 32484, 32539, 32619, 32639, 32768, 0],
    [6607, 6990, 8268, 32060, 32219, 32338, 32371, 32768, 0],
    [5429, 6676, 7122, 32027, 32227, 32531, 32582, 32768, 0],
    [711, 966, 1172, 32448, 32538, 32617, 32664, 32768, 0],
];

/// `Default_Skip_Cdf[ SKIP_CONTEXTS ][ 3 ]` (§9.4). A binary symbol.
pub const DEFAULT_SKIP_CDF: [[u16; 3]; SKIP_CONTEXTS] =
    [[31671, 32768, 0], [16515, 32768, 0], [4576, 32768, 0]];

/// `Default_Segment_Id_Cdf[ SEGMENT_ID_CONTEXTS ][ MAX_SEGMENTS + 1 ]`
/// (§9.4). Codes the `segment_id` (`MAX_SEGMENTS == 8` values).
pub const DEFAULT_SEGMENT_ID_CDF: [[u16; MAX_SEGMENTS + 1]; SEGMENT_ID_CONTEXTS] = [
    [5622, 7893, 16093, 18233, 27809, 28373, 32533, 32768, 0],
    [14274, 18230, 22557, 24935, 29980, 30851, 32344, 32768, 0],
    [27527, 28487, 28723, 28890, 32397, 32647, 32679, 32768, 0],
];

// ---------------------------------------------------------------------
// §9.4 motion-vector default CDF tables (round 17).
//
// Per §8.3.1 every per-tile `Mv*Cdf[ i ]` array (`i = 0..MV_CONTEXTS-1`)
// is "set equal to a copy of" the corresponding `Default_Mv_*_Cdf`. The
// per-component (`comp = 0..1`) decomposition for `MvSign`/`MvBit`/
// `MvHp`/`MvClass0Bit`/`MvClass0Hp` similarly broadcasts the same flat
// default row to both components; `MvClassCdf`/`MvClass0FrCdf`/
// `MvFrCdf` carry distinct per-component rows in the source default
// (the inner `2` in the spec dimension is the `comp` axis).
//
// Each innermost array has length `N + 1`: `N` cumulative frequencies
// (the last `1 << 15 == 32768`) followed by the §8.3 adaptation
// counter, which starts at 0.
// ---------------------------------------------------------------------

/// `Default_Mv_Joint_Cdf[ MV_JOINTS + 1 ]` (§9.4). The spec uses
/// `MV_JOINTS + 1` as both the symbol count and the cumulative-array
/// length (the row holds 4 frequencies + 1 counter).
pub const DEFAULT_MV_JOINT_CDF: [u16; MV_JOINTS + 1] = [4096, 11264, 19328, 32768, 0];

/// `Default_Mv_Sign_Cdf[ 3 ]` (§9.4). Binary symbol; the cumulative
/// value `128*128 = 16384` is transcribed expanded.
pub const DEFAULT_MV_SIGN_CDF: [u16; 3] = [128 * 128, 32768, 0];

/// `Default_Mv_Class_Cdf[ 2 ][ MV_CLASSES + 1 ]` (§9.4). The leading `2`
/// is the `comp = 0..1` axis (both rows are identical per spec).
pub const DEFAULT_MV_CLASS_CDF: [[u16; MV_CLASSES + 1]; MV_COMPS] = [
    [
        28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767, 32768, 0,
    ],
    [
        28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767, 32768, 0,
    ],
];

/// `Default_Mv_Class0_Bit_Cdf[ 3 ]` (§9.4). Binary symbol; broadcast to
/// every `[comp]` slot at §8.3.1 init.
pub const DEFAULT_MV_CLASS0_BIT_CDF: [u16; 3] = [216 * 128, 32768, 0];

/// `Default_Mv_Class0_Fr_Cdf[ 2 ][ CLASS0_SIZE ][ MV_JOINTS + 1 ]`
/// (§9.4). The leading `2` is the `comp = 0..1` axis; the middle
/// dimension is `mv_class0_bit = 0..1` (the literal §5.11.32 dispatch
/// `[ MvCtx ][ comp ][ mv_class0_bit ]`).
pub const DEFAULT_MV_CLASS0_FR_CDF: [[[u16; MV_JOINTS + 1]; CLASS0_SIZE]; MV_COMPS] = [
    [
        [16384, 24576, 26624, 32768, 0],
        [12288, 21248, 24128, 32768, 0],
    ],
    [
        [16384, 24576, 26624, 32768, 0],
        [12288, 21248, 24128, 32768, 0],
    ],
];

/// `Default_Mv_Class0_Hp_Cdf[ 3 ]` (§9.4). Binary symbol.
pub const DEFAULT_MV_CLASS0_HP_CDF: [u16; 3] = [160 * 128, 32768, 0];

/// `Default_Mv_Bit_Cdf[ MV_OFFSET_BITS ][ 3 ]` (§9.4). One binary
/// distribution per offset-bit position `i = 0..MV_OFFSET_BITS-1`. The
/// `*128` factor expands the `8.7`-style fixed-point notation used in
/// the spec.
pub const DEFAULT_MV_BIT_CDF: [[u16; 3]; MV_OFFSET_BITS] = [
    [136 * 128, 32768, 0],
    [140 * 128, 32768, 0],
    [148 * 128, 32768, 0],
    [160 * 128, 32768, 0],
    [176 * 128, 32768, 0],
    [192 * 128, 32768, 0],
    [224 * 128, 32768, 0],
    [234 * 128, 32768, 0],
    [234 * 128, 32768, 0],
    [240 * 128, 32768, 0],
];

/// `Default_Mv_Fr_Cdf[ 2 ][ MV_JOINTS + 1 ]` (§9.4). The leading `2` is
/// the `comp = 0..1` axis. Both rows are identical per spec; the inner
/// `MV_JOINTS + 1` matches the 4-value `mv_fr` symbol (4 frequencies +
/// 1 counter).
pub const DEFAULT_MV_FR_CDF: [[u16; MV_JOINTS + 1]; MV_COMPS] = [
    [8192, 17408, 21248, 32768, 0],
    [8192, 17408, 21248, 32768, 0],
];

/// `Default_Mv_Hp_Cdf[ 3 ]` (§9.4). Binary symbol.
pub const DEFAULT_MV_HP_CDF: [u16; 3] = [128 * 128, 32768, 0];

// ---------------------------------------------------------------------
// §9.4 inter-mode / reference-frame default CDF tables (round 18).
//
// Per §8.3.1 each of `NewMvCdf`, `ZeroMvCdf`, `RefMvCdf`, `DrlModeCdf`,
// `IsInterCdf`, `CompModeCdf`, `SkipModeCdf`, `CompRefCdf`,
// `CompBwdRefCdf`, `SingleRefCdf`, `CompoundModeCdf`, `CompRefTypeCdf`,
// `UniCompRefCdf` "is set to a copy of" the corresponding `Default_*_Cdf`.
//
// Each innermost array has length `N + 1`: `N` cumulative frequencies
// (the last `1 << 15 == 32768`) followed by the §8.3 adaptation
// counter, which starts at 0.
// ---------------------------------------------------------------------

/// `Default_New_Mv_Cdf[ NEW_MV_CONTEXTS ][ 3 ]` (§9.4). Binary; codes
/// `new_mv` ("is the predicted mv a NEWMV?") per §8.3.2 selection
/// `TileNewMvCdf[ NewMvContext ]`.
pub const DEFAULT_NEW_MV_CDF: [[u16; 3]; NEW_MV_CONTEXTS] = [
    [24035, 32768, 0],
    [16630, 32768, 0],
    [15339, 32768, 0],
    [8386, 32768, 0],
    [12222, 32768, 0],
    [4676, 32768, 0],
];

/// `Default_Zero_Mv_Cdf[ ZERO_MV_CONTEXTS ][ 3 ]` (§9.4). Binary; codes
/// `zero_mv` per §8.3.2 `TileZeroMvCdf[ ZeroMvContext ]`.
pub const DEFAULT_ZERO_MV_CDF: [[u16; 3]; ZERO_MV_CONTEXTS] = [[2175, 32768, 0], [1054, 32768, 0]];

/// `Default_Ref_Mv_Cdf[ REF_MV_CONTEXTS ][ 3 ]` (§9.4). Binary; codes
/// `ref_mv` per §8.3.2 `TileRefMvCdf[ RefMvContext ]`.
pub const DEFAULT_REF_MV_CDF: [[u16; 3]; REF_MV_CONTEXTS] = [
    [23974, 32768, 0],
    [24188, 32768, 0],
    [17848, 32768, 0],
    [28622, 32768, 0],
    [24312, 32768, 0],
    [19923, 32768, 0],
];

/// `Default_Drl_Mode_Cdf[ DRL_MODE_CONTEXTS ][ 3 ]` (§9.4). Binary; codes
/// `drl_mode` per §8.3.2 `TileDrlModeCdf[ DrlCtxStack[ idx ] ]`.
pub const DEFAULT_DRL_MODE_CDF: [[u16; 3]; DRL_MODE_CONTEXTS] =
    [[13104, 32768, 0], [24560, 32768, 0], [18945, 32768, 0]];

/// `Default_Is_Inter_Cdf[ IS_INTER_CONTEXTS ][ 3 ]` (§9.4). Binary; codes
/// `is_inter` per §8.3.2 `TileIsInterCdf[ ctx ]`.
pub const DEFAULT_IS_INTER_CDF: [[u16; 3]; IS_INTER_CONTEXTS] = [
    [806, 32768, 0],
    [16662, 32768, 0],
    [20186, 32768, 0],
    [26538, 32768, 0],
];

/// `Default_Comp_Mode_Cdf[ COMP_INTER_CONTEXTS ][ 3 ]` (§9.4). Binary;
/// codes `comp_mode` per §8.3.2 `TileCompModeCdf[ ctx ]`.
pub const DEFAULT_COMP_MODE_CDF: [[u16; 3]; COMP_INTER_CONTEXTS] = [
    [26828, 32768, 0],
    [24035, 32768, 0],
    [12031, 32768, 0],
    [10640, 32768, 0],
    [2901, 32768, 0],
];

/// `Default_Skip_Mode_Cdf[ SKIP_MODE_CONTEXTS ][ 3 ]` (§9.4). Binary;
/// codes `skip_mode` per §8.3.2 `TileSkipModeCdf[ ctx ]`.
pub const DEFAULT_SKIP_MODE_CDF: [[u16; 3]; SKIP_MODE_CONTEXTS] =
    [[32621, 32768, 0], [20708, 32768, 0], [8127, 32768, 0]];

/// `Default_Comp_Ref_Cdf[ REF_CONTEXTS ][ FWD_REFS - 1 ][ 3 ]` (§9.4).
/// Binary; codes `comp_ref` / `comp_ref_p1` / `comp_ref_p2` per §8.3.2
/// `TileCompRefCdf[ ctx ][ 0..2 ]`.
pub const DEFAULT_COMP_REF_CDF: [[[u16; 3]; FWD_REFS - 1]; REF_CONTEXTS] = [
    [[4946, 32768, 0], [9468, 32768, 0], [1503, 32768, 0]],
    [[19891, 32768, 0], [22441, 32768, 0], [15160, 32768, 0]],
    [[30731, 32768, 0], [31059, 32768, 0], [27544, 32768, 0]],
];

/// `Default_Comp_Bwd_Ref_Cdf[ REF_CONTEXTS ][ BWD_REFS - 1 ][ 3 ]`
/// (§9.4). Binary; codes `comp_bwdref` / `comp_bwdref_p1` per §8.3.2
/// `TileCompBwdRefCdf[ ctx ][ 0..1 ]`.
pub const DEFAULT_COMP_BWD_REF_CDF: [[[u16; 3]; BWD_REFS - 1]; REF_CONTEXTS] = [
    [[2235, 32768, 0], [1423, 32768, 0]],
    [[17182, 32768, 0], [15175, 32768, 0]],
    [[30606, 32768, 0], [30489, 32768, 0]],
];

/// `Default_Single_Ref_Cdf[ REF_CONTEXTS ][ SINGLE_REFS - 1 ][ 3 ]`
/// (§9.4). Binary; codes `single_ref_p1` .. `single_ref_p6` per §8.3.2
/// `TileSingleRefCdf[ ctx ][ 0..5 ]` (the 6th `single_ref_p*` slot
/// `[5]` is `single_ref_p6` per the §8.3.2 list).
pub const DEFAULT_SINGLE_REF_CDF: [[[u16; 3]; SINGLE_REFS - 1]; REF_CONTEXTS] = [
    [
        [4897, 32768, 0],
        [1555, 32768, 0],
        [4236, 32768, 0],
        [8650, 32768, 0],
        [904, 32768, 0],
        [1444, 32768, 0],
    ],
    [
        [16973, 32768, 0],
        [16751, 32768, 0],
        [19647, 32768, 0],
        [24773, 32768, 0],
        [11014, 32768, 0],
        [15087, 32768, 0],
    ],
    [
        [29744, 32768, 0],
        [30279, 32768, 0],
        [31194, 32768, 0],
        [31895, 32768, 0],
        [26875, 32768, 0],
        [30304, 32768, 0],
    ],
];

/// `Default_Compound_Mode_Cdf[ COMPOUND_MODE_CONTEXTS ][ COMPOUND_MODES + 1 ]`
/// (§9.4). Codes the 8-value `compound_mode` per §8.3.2
/// `TileCompoundModeCdf[ ctx ]`, where `ctx` is taken from
/// `Compound_Mode_Ctx_Map[ RefMvContext >> 1 ][ Min(NewMvContext,
/// COMP_NEWMV_CTXS - 1) ]`.
pub const DEFAULT_COMPOUND_MODE_CDF: [[u16; COMPOUND_MODES + 1]; COMPOUND_MODE_CONTEXTS] = [
    [7760, 13823, 15808, 17641, 19156, 20666, 26891, 32768, 0],
    [10730, 19452, 21145, 22749, 24039, 25131, 28724, 32768, 0],
    [10664, 20221, 21588, 22906, 24295, 25387, 28436, 32768, 0],
    [13298, 16984, 20471, 24182, 25067, 25736, 26422, 32768, 0],
    [18904, 23325, 25242, 27432, 27898, 28258, 30758, 32768, 0],
    [10725, 17454, 20124, 22820, 24195, 25168, 26046, 32768, 0],
    [17125, 24273, 25814, 27492, 28214, 28704, 30592, 32768, 0],
    [13046, 23214, 24505, 25942, 27435, 28442, 29330, 32768, 0],
];

/// `Default_Comp_Ref_Type_Cdf[ COMP_REF_TYPE_CONTEXTS ][ 3 ]` (§9.4).
/// Binary; codes `comp_ref_type` per §8.3.2 `TileCompRefTypeCdf[ ctx ]`.
pub const DEFAULT_COMP_REF_TYPE_CDF: [[u16; 3]; COMP_REF_TYPE_CONTEXTS] = [
    [1198, 32768, 0],
    [2070, 32768, 0],
    [9166, 32768, 0],
    [7499, 32768, 0],
    [22475, 32768, 0],
];

/// `Default_Uni_Comp_Ref_Cdf[ REF_CONTEXTS ][ UNIDIR_COMP_REFS - 1 ][ 3 ]`
/// (§9.4). Binary; codes `uni_comp_ref` / `uni_comp_ref_p1` /
/// `uni_comp_ref_p2` per §8.3.2 `TileUniCompRefCdf[ ctx ][ 0..2 ]`.
pub const DEFAULT_UNI_COMP_REF_CDF: [[[u16; 3]; UNIDIR_COMP_REFS - 1]; REF_CONTEXTS] = [
    [[5284, 32768, 0], [3865, 32768, 0], [3128, 32768, 0]],
    [[23152, 32768, 0], [14173, 32768, 0], [15270, 32768, 0]],
    [[31774, 32768, 0], [25120, 32768, 0], [26710, 32768, 0]],
];

// ---------------------------------------------------------------------
// §9.4 palette / filter-intra / CFL default CDF tables (round 19).
//
// Per §8.3.1 each working `*Cdf` array "is set to a copy of" its
// `Default_*_Cdf` (no per-context broadcast for this group). Each
// innermost array has length `N + 1`: `N` cumulative frequencies (the
// last `1 << 15 == 32768`) followed by the §8.3 adaptation counter,
// which starts at 0.
// ---------------------------------------------------------------------

/// `Default_Filter_Intra_Mode_Cdf[ INTRA_FILTER_MODES + 1 ]` (§9.4).
/// Codes the 5-value `filter_intra_mode`.
pub const DEFAULT_FILTER_INTRA_MODE_CDF: [u16; INTRA_FILTER_MODES + 1] =
    [8949, 12776, 17211, 29558, 32768, 0];

/// `Default_Filter_Intra_Cdf[ BLOCK_SIZES ][ 3 ]` (§9.4). Binary; codes
/// `use_filter_intra` per §8.3.2 `TileFilterIntraCdf[ MiSize ]`. Per the
/// §9.4 note, first-dimension indices 10..=15 and 20..=21 are never
/// reached but are transcribed verbatim.
pub const DEFAULT_FILTER_INTRA_CDF: [[u16; 3]; BLOCK_SIZES] = [
    [4621, 32768, 0],
    [6743, 32768, 0],
    [5893, 32768, 0],
    [7866, 32768, 0],
    [12551, 32768, 0],
    [9394, 32768, 0],
    [12408, 32768, 0],
    [14301, 32768, 0],
    [12756, 32768, 0],
    [22343, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
    [12770, 32768, 0],
    [10368, 32768, 0],
    [20229, 32768, 0],
    [18101, 32768, 0],
    [16384, 32768, 0],
    [16384, 32768, 0],
];

/// `Default_Palette_Y_Mode_Cdf[ PALETTE_BLOCK_SIZE_CONTEXTS ][ PALETTE_Y_MODE_CONTEXTS ][ 3 ]`
/// (§9.4). Binary; codes `has_palette_y` per §8.3.2
/// `TilePaletteYModeCdf[ bsizeCtx ][ ctx ]`.
pub const DEFAULT_PALETTE_Y_MODE_CDF: [[[u16; 3]; PALETTE_Y_MODE_CONTEXTS];
    PALETTE_BLOCK_SIZE_CONTEXTS] = [
    [[31676, 32768, 0], [3419, 32768, 0], [1261, 32768, 0]],
    [[31912, 32768, 0], [2859, 32768, 0], [980, 32768, 0]],
    [[31823, 32768, 0], [3400, 32768, 0], [781, 32768, 0]],
    [[32030, 32768, 0], [3561, 32768, 0], [904, 32768, 0]],
    [[32309, 32768, 0], [7337, 32768, 0], [1462, 32768, 0]],
    [[32265, 32768, 0], [4015, 32768, 0], [1521, 32768, 0]],
    [[32450, 32768, 0], [7946, 32768, 0], [129, 32768, 0]],
];

/// `Default_Palette_Uv_Mode_Cdf[ PALETTE_UV_MODE_CONTEXTS ][ 3 ]` (§9.4).
/// Binary; codes `has_palette_uv` per §8.3.2 `TilePaletteUVModeCdf[ ctx ]`.
pub const DEFAULT_PALETTE_UV_MODE_CDF: [[u16; 3]; PALETTE_UV_MODE_CONTEXTS] =
    [[32461, 32768, 0], [21488, 32768, 0]];

/// `Default_Palette_Y_Size_Cdf[ PALETTE_BLOCK_SIZE_CONTEXTS ][ PALETTE_SIZES + 1 ]`
/// (§9.4). Codes the 7-value `palette_size_y_minus_2` per §8.3.2
/// `TilePaletteYSizeCdf[ bsizeCtx ]`.
pub const DEFAULT_PALETTE_Y_SIZE_CDF: [[u16; PALETTE_SIZES + 1]; PALETTE_BLOCK_SIZE_CONTEXTS] = [
    [7952, 13000, 18149, 21478, 25527, 29241, 32768, 0],
    [7139, 11421, 16195, 19544, 23666, 28073, 32768, 0],
    [7788, 12741, 17325, 20500, 24315, 28530, 32768, 0],
    [8271, 14064, 18246, 21564, 25071, 28533, 32768, 0],
    [12725, 19180, 21863, 24839, 27535, 30120, 32768, 0],
    [9711, 14888, 16923, 21052, 25661, 27875, 32768, 0],
    [14940, 20797, 21678, 24186, 27033, 28999, 32768, 0],
];

/// `Default_Palette_Uv_Size_Cdf[ PALETTE_BLOCK_SIZE_CONTEXTS ][ PALETTE_SIZES + 1 ]`
/// (§9.4). Codes the 7-value `palette_size_uv_minus_2` per §8.3.2
/// `TilePaletteUVSizeCdf[ bsizeCtx ]`.
pub const DEFAULT_PALETTE_UV_SIZE_CDF: [[u16; PALETTE_SIZES + 1]; PALETTE_BLOCK_SIZE_CONTEXTS] = [
    [8713, 19979, 27128, 29609, 31331, 32272, 32768, 0],
    [5839, 15573, 23581, 26947, 29848, 31700, 32768, 0],
    [4426, 11260, 17999, 21483, 25863, 29430, 32768, 0],
    [3228, 9464, 14993, 18089, 22523, 27420, 32768, 0],
    [3768, 8886, 13091, 17852, 22495, 27207, 32768, 0],
    [2464, 8451, 12861, 21632, 25525, 28555, 32768, 0],
    [1269, 5435, 10433, 18963, 21700, 25865, 32768, 0],
];

/// `Default_Palette_Size_2_Y_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 3 ]`
/// (§9.4). Codes the 2-value `palette_color_idx_y` for `PaletteSizeY == 2`.
pub const DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF: [[u16; 3]; PALETTE_COLOR_CONTEXTS] = [
    [28710, 32768, 0],
    [16384, 32768, 0],
    [10553, 32768, 0],
    [27036, 32768, 0],
    [31603, 32768, 0],
];

/// `Default_Palette_Size_3_Y_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 4 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF: [[u16; 4]; PALETTE_COLOR_CONTEXTS] = [
    [27877, 30490, 32768, 0],
    [11532, 25697, 32768, 0],
    [6544, 30234, 32768, 0],
    [23018, 28072, 32768, 0],
    [31915, 32385, 32768, 0],
];

/// `Default_Palette_Size_4_Y_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 5 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF: [[u16; 5]; PALETTE_COLOR_CONTEXTS] = [
    [25572, 28046, 30045, 32768, 0],
    [9478, 21590, 27256, 32768, 0],
    [7248, 26837, 29824, 32768, 0],
    [19167, 24486, 28349, 32768, 0],
    [31400, 31825, 32250, 32768, 0],
];

/// `Default_Palette_Size_5_Y_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 6 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF: [[u16; 6]; PALETTE_COLOR_CONTEXTS] = [
    [24779, 26955, 28576, 30282, 32768, 0],
    [8669, 20364, 24073, 28093, 32768, 0],
    [4255, 27565, 29377, 31067, 32768, 0],
    [19864, 23674, 26716, 29530, 32768, 0],
    [31646, 31893, 32147, 32426, 32768, 0],
];

/// `Default_Palette_Size_6_Y_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 7 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF: [[u16; 7]; PALETTE_COLOR_CONTEXTS] = [
    [23132, 25407, 26970, 28435, 30073, 32768, 0],
    [7443, 17242, 20717, 24762, 27982, 32768, 0],
    [6300, 24862, 26944, 28784, 30671, 32768, 0],
    [18916, 22895, 25267, 27435, 29652, 32768, 0],
    [31270, 31550, 31808, 32059, 32353, 32768, 0],
];

/// `Default_Palette_Size_7_Y_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 8 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF: [[u16; 8]; PALETTE_COLOR_CONTEXTS] = [
    [23105, 25199, 26464, 27684, 28931, 30318, 32768, 0],
    [6950, 15447, 18952, 22681, 25567, 28563, 32768, 0],
    [7560, 23474, 25490, 27203, 28921, 30708, 32768, 0],
    [18544, 22373, 24457, 26195, 28119, 30045, 32768, 0],
    [31198, 31451, 31670, 31882, 32123, 32391, 32768, 0],
];

/// `Default_Palette_Size_8_Y_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 9 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF: [[u16; 9]; PALETTE_COLOR_CONTEXTS] = [
    [21689, 23883, 25163, 26352, 27506, 28827, 30195, 32768, 0],
    [6892, 15385, 17840, 21606, 24287, 26753, 29204, 32768, 0],
    [5651, 23182, 25042, 26518, 27982, 29392, 30900, 32768, 0],
    [19349, 22578, 24418, 25994, 27524, 29031, 30448, 32768, 0],
    [31028, 31270, 31504, 31705, 31927, 32153, 32392, 32768, 0],
];

/// `Default_Palette_Size_2_Uv_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 3 ]`
/// (§9.4). Codes the 2-value `palette_color_idx_uv` for `PaletteSizeUV == 2`.
pub const DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF: [[u16; 3]; PALETTE_COLOR_CONTEXTS] = [
    [29089, 32768, 0],
    [16384, 32768, 0],
    [8713, 32768, 0],
    [29257, 32768, 0],
    [31610, 32768, 0],
];

/// `Default_Palette_Size_3_Uv_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 4 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF: [[u16; 4]; PALETTE_COLOR_CONTEXTS] = [
    [25257, 29145, 32768, 0],
    [12287, 27293, 32768, 0],
    [7033, 27960, 32768, 0],
    [20145, 25405, 32768, 0],
    [30608, 31639, 32768, 0],
];

/// `Default_Palette_Size_4_Uv_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 5 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF: [[u16; 5]; PALETTE_COLOR_CONTEXTS] = [
    [24210, 27175, 29903, 32768, 0],
    [9888, 22386, 27214, 32768, 0],
    [5901, 26053, 29293, 32768, 0],
    [18318, 22152, 28333, 32768, 0],
    [30459, 31136, 31926, 32768, 0],
];

/// `Default_Palette_Size_5_Uv_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 6 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF: [[u16; 6]; PALETTE_COLOR_CONTEXTS] = [
    [22980, 25479, 27781, 29986, 32768, 0],
    [8413, 21408, 24859, 28874, 32768, 0],
    [2257, 29449, 30594, 31598, 32768, 0],
    [19189, 21202, 25915, 28620, 32768, 0],
    [31844, 32044, 32281, 32518, 32768, 0],
];

/// `Default_Palette_Size_6_Uv_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 7 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF: [[u16; 7]; PALETTE_COLOR_CONTEXTS] = [
    [22217, 24567, 26637, 28683, 30548, 32768, 0],
    [7307, 16406, 19636, 24632, 28424, 32768, 0],
    [4441, 25064, 26879, 28942, 30919, 32768, 0],
    [17210, 20528, 23319, 26750, 29582, 32768, 0],
    [30674, 30953, 31396, 31735, 32207, 32768, 0],
];

/// `Default_Palette_Size_7_Uv_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 8 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF: [[u16; 8]; PALETTE_COLOR_CONTEXTS] = [
    [21239, 23168, 25044, 26962, 28705, 30506, 32768, 0],
    [6545, 15012, 18004, 21817, 25503, 28701, 32768, 0],
    [3448, 26295, 27437, 28704, 30126, 31442, 32768, 0],
    [15889, 18323, 21704, 24698, 26976, 29690, 32768, 0],
    [30988, 31204, 31479, 31734, 31983, 32325, 32768, 0],
];

/// `Default_Palette_Size_8_Uv_Color_Cdf[ PALETTE_COLOR_CONTEXTS ][ 9 ]` (§9.4).
pub const DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF: [[u16; 9]; PALETTE_COLOR_CONTEXTS] = [
    [21442, 23288, 24758, 26246, 27649, 28980, 30563, 32768, 0],
    [5863, 14933, 17552, 20668, 23683, 26411, 29273, 32768, 0],
    [3415, 25810, 26877, 27990, 29223, 30394, 31618, 32768, 0],
    [17965, 20084, 22232, 23974, 26274, 28402, 30390, 32768, 0],
    [31190, 31329, 31516, 31679, 31825, 32026, 32322, 32768, 0],
];

/// `Default_Cfl_Sign_Cdf[ CFL_JOINT_SIGNS + 1 ]` (§9.4). Codes the
/// 8-value `cfl_alpha_signs` per §8.3.2 `TileCflSignCdf`.
pub const DEFAULT_CFL_SIGN_CDF: [u16; CFL_JOINT_SIGNS + 1] =
    [1418, 2123, 13340, 18405, 26972, 28343, 32294, 32768, 0];

/// `Default_Cfl_Alpha_Cdf[ CFL_ALPHA_CONTEXTS ][ CFL_ALPHABET_SIZE + 1 ]`
/// (§9.4). Codes the 16-value `cfl_alpha_u` / `cfl_alpha_v` per §8.3.2
/// `TileCflAlphaCdf[ ctx ]`.
pub const DEFAULT_CFL_ALPHA_CDF: [[u16; CFL_ALPHABET_SIZE + 1]; CFL_ALPHA_CONTEXTS] = [
    [
        7637, 20719, 31401, 32481, 32657, 32688, 32692, 32696, 32700, 32704, 32708, 32712, 32716,
        32720, 32724, 32768, 0,
    ],
    [
        14365, 23603, 28135, 31168, 32167, 32395, 32487, 32573, 32620, 32647, 32668, 32672, 32676,
        32680, 32684, 32768, 0,
    ],
    [
        11532, 22380, 28445, 31360, 32349, 32523, 32584, 32649, 32673, 32677, 32681, 32685, 32689,
        32693, 32697, 32768, 0,
    ],
    [
        26990, 31402, 32282, 32571, 32692, 32696, 32700, 32704, 32708, 32712, 32716, 32720, 32724,
        32728, 32732, 32768, 0,
    ],
    [
        17248, 26058, 28904, 30608, 31305, 31877, 32126, 32321, 32394, 32464, 32516, 32560, 32576,
        32593, 32622, 32768, 0,
    ],
    [
        14738, 21678, 25779, 27901, 29024, 30302, 30980, 31843, 32144, 32413, 32520, 32594, 32622,
        32656, 32660, 32768, 0,
    ],
];

// ---------------------------------------------------------------------
// Round 20 — transform-size group (§9.4): `Default_Tx_*_Cdf` (four
// per-`maxRectTxSize` arrays codes `tx_depth`) + `Default_Txfm_Split_Cdf`
// (codes the binary `txfm_split`).
// ---------------------------------------------------------------------

/// `Default_Tx_8x8_Cdf[ TX_SIZE_CONTEXTS ][ MAX_TX_DEPTH + 1 ]` (§9.4).
/// Selected by §8.3.2 when `maxTxDepth == 1` (the small block-size
/// `tx_depth` group, two symbol values).
pub const DEFAULT_TX_8X8_CDF: [[u16; MAX_TX_DEPTH + 1]; TX_SIZE_CONTEXTS] =
    [[19968, 32768, 0], [19968, 32768, 0], [24320, 32768, 0]];

/// `Default_Tx_16x16_Cdf[ TX_SIZE_CONTEXTS ][ MAX_TX_DEPTH + 2 ]` (§9.4).
/// Selected by §8.3.2 when `maxTxDepth == 2` (three symbol values).
pub const DEFAULT_TX_16X16_CDF: [[u16; MAX_TX_DEPTH + 2]; TX_SIZE_CONTEXTS] = [
    [12272, 30172, 32768, 0],
    [12272, 30172, 32768, 0],
    [18677, 30848, 32768, 0],
];

/// `Default_Tx_32x32_Cdf[ TX_SIZE_CONTEXTS ][ MAX_TX_DEPTH + 2 ]` (§9.4).
/// Selected by §8.3.2 when `maxTxDepth == 3`.
pub const DEFAULT_TX_32X32_CDF: [[u16; MAX_TX_DEPTH + 2]; TX_SIZE_CONTEXTS] = [
    [12986, 15180, 32768, 0],
    [12986, 15180, 32768, 0],
    [24302, 25602, 32768, 0],
];

/// `Default_Tx_64x64_Cdf[ TX_SIZE_CONTEXTS ][ MAX_TX_DEPTH + 2 ]` (§9.4).
/// Selected by §8.3.2 when `maxTxDepth == 4`.
pub const DEFAULT_TX_64X64_CDF: [[u16; MAX_TX_DEPTH + 2]; TX_SIZE_CONTEXTS] = [
    [5782, 11475, 32768, 0],
    [5782, 11475, 32768, 0],
    [16803, 22759, 32768, 0],
];

/// `Default_Txfm_Split_Cdf[ TXFM_PARTITION_CONTEXTS ][ 3 ]` (§9.4).
/// Codes the binary `txfm_split` syntax element per §8.3.2.
pub const DEFAULT_TXFM_SPLIT_CDF: [[u16; 3]; TXFM_PARTITION_CONTEXTS] = [
    [28581, 32768, 0],
    [23846, 32768, 0],
    [20847, 32768, 0],
    [24315, 32768, 0],
    [18196, 32768, 0],
    [12133, 32768, 0],
    [18791, 32768, 0],
    [10887, 32768, 0],
    [11005, 32768, 0],
    [27179, 32768, 0],
    [20004, 32768, 0],
    [11281, 32768, 0],
    [26549, 32768, 0],
    [19308, 32768, 0],
    [14224, 32768, 0],
    [28015, 32768, 0],
    [21546, 32768, 0],
    [14400, 32768, 0],
    [28165, 32768, 0],
    [22401, 32768, 0],
    [16088, 32768, 0],
];

// ---------------------------------------------------------------------
// Round 21 — inter-frame transform-type group (§9.4):
// `Default_Inter_Tx_Type_Set{1,2,3}_Cdf`. Codes `inter_tx_type` per
// §5.11.47 / §8.3.2 against the §6.10.19 transform-set returned by
// §5.11.48 `get_tx_set()`. The intra counterparts
// (`Default_Intra_Tx_Type_Set{1,2}_Cdf`) are a follow-up — they carry
// an extra `intraDir` axis and are handled in a separate round.
// ---------------------------------------------------------------------

/// `Default_Inter_Tx_Type_Set1_Cdf[ 2 ][ TX_TYPES + 1 ]` (§9.4).
///
/// Selected by §8.3.2 when `set == TX_SET_INTER_1` (the full inter
/// transform set, `Tx_Type_Inter_Inv_Set1` with 16 entries). The first
/// axis is `Tx_Size_Sqr[ txSz ] ∈ { TX_4X4, TX_8X8 }` — the only sizes
/// that reach `TX_SET_INTER_1` per §5.11.48 (`txSzSqr == TX_16X16`
/// would already have been routed to `TX_SET_INTER_2`).
pub const DEFAULT_INTER_TX_TYPE_SET1_CDF: [[u16; TX_TYPES + 1]; INTER_TX_TYPE_SET1_SIZES] = [
    [
        4458, 5560, 7695, 9709, 13330, 14789, 17537, 20266, 21504, 22848, 23934, 25474, 27727,
        28915, 30631, 32768, 0,
    ],
    [
        1645, 2573, 4778, 5711, 7807, 8622, 10522, 15357, 17674, 20408, 22517, 25010, 27116, 28856,
        30749, 32768, 0,
    ],
];

/// `Default_Inter_Tx_Type_Set2_Cdf[ TX_TYPES_SET2 + 1 ]` (§9.4).
///
/// Selected by §8.3.2 when `set == TX_SET_INTER_2` (the 16x16-only
/// inter transform set, `Tx_Type_Inter_Inv_Set2` with 12 entries).
/// `TX_SET_INTER_2` is reached only when `txSzSqr == TX_16X16` and
/// `reduced_tx_set == 0`, so there is no per-`Tx_Size_Sqr` first axis.
pub const DEFAULT_INTER_TX_TYPE_SET2_CDF: [u16; TX_TYPES_SET2 + 1] = [
    770, 2421, 5225, 12907, 15819, 18927, 21561, 24089, 26595, 28526, 30529, 32768, 0,
];

/// `Default_Inter_Tx_Type_Set3_Cdf[ 4 ][ TX_TYPES_SET3 + 1 ]` (§9.4).
///
/// Selected by §8.3.2 when `set == TX_SET_INTER_3` (the reduced inter
/// transform set, `Tx_Type_Inter_Inv_Set3 = { IDTX, DCT_DCT }`, two
/// entries). The first axis is `Tx_Size_Sqr[ txSz ] ∈ { TX_4X4, TX_8X8,
/// TX_16X16, TX_32X32 }` — `txSzSqrUp > TX_32X32` would already have
/// been routed to `TX_SET_DCTONLY` per §5.11.48.
pub const DEFAULT_INTER_TX_TYPE_SET3_CDF: [[u16; TX_TYPES_SET3 + 1]; INTER_TX_TYPE_SET3_SIZES] = [
    [16384, 32768, 0],
    [4167, 32768, 0],
    [1998, 32768, 0],
    [748, 32768, 0],
];

// ---------------------------------------------------------------------
// Round 22 — inter-frame interpolation-filter group (§9.4):
// `Default_Interp_Filter_Cdf`. Codes `interp_filter` per §5.11.x and
// the §8.3.2 selection. The §8.3.2 `ctx` formula folds two scalar
// neighbour inputs (`aboveType` / `leftType`) plus the §5.11.x scope
// `dir` / `RefFrame[1]` into a single `0..INTERP_FILTER_CONTEXTS`
// index — see [`interp_filter_ctx`].
// ---------------------------------------------------------------------

/// `Default_Interp_Filter_Cdf[ INTERP_FILTER_CONTEXTS ][ INTERP_FILTERS + 1 ]`
/// (§9.4). Indexed by the §8.3.2 `interp_filter` ctx (in
/// `0..INTERP_FILTER_CONTEXTS`); the innermost row codes the
/// `interp_filter ∈ { EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP_SHARP }`
/// symbol (`INTERP_FILTERS = 3` cumulative frequencies plus the
/// §8.3 adaptation counter that starts at 0).
pub const DEFAULT_INTERP_FILTER_CDF: [[u16; INTERP_FILTERS + 1]; INTERP_FILTER_CONTEXTS] = [
    [31935, 32720, 32768, 0],
    [5568, 32719, 32768, 0],
    [422, 2938, 32768, 0],
    [28244, 32608, 32768, 0],
    [31206, 31953, 32768, 0],
    [4862, 32121, 32768, 0],
    [770, 1152, 32768, 0],
    [20889, 25637, 32768, 0],
    [31910, 32724, 32768, 0],
    [4120, 32712, 32768, 0],
    [305, 2247, 32768, 0],
    [27403, 32636, 32768, 0],
    [31022, 32009, 32768, 0],
    [2963, 32093, 32768, 0],
    [601, 943, 32768, 0],
    [14969, 21398, 32768, 0],
];

// ---------------------------------------------------------------------
// §8.3.1 init-from-defaults: the per-tile working CDF set.
// ---------------------------------------------------------------------

/// The per-tile working set of CDF arrays for the intra-frame mode /
/// partition subset, as set up by §8.3.1 ("each `Tile*Cdf` array is set
/// equal to a copy of `Default_*_Cdf`").
///
/// Field names mirror the spec's `Tile*Cdf` arrays with the `Tile`
/// prefix dropped (the prefix only distinguishes the per-tile copy from
/// the immutable `Default_*` source). Each array is mutated in place by
/// [`crate::symbol_decoder::SymbolDecoder::read_symbol`] as it adapts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileCdfContext {
    /// `TileIntraFrameYModeCdf` (§8.3.1).
    pub intra_frame_y_mode: [[[u16; INTRA_MODES + 1]; INTRA_MODE_CONTEXTS]; INTRA_MODE_CONTEXTS],
    /// `TilePartitionW8Cdf` (§8.3.1).
    pub partition_w8: [[u16; 5]; PARTITION_CONTEXTS],
    /// `TilePartitionW16Cdf` (§8.3.1).
    pub partition_w16: [[u16; 11]; PARTITION_CONTEXTS],
    /// `TilePartitionW32Cdf` (§8.3.1).
    pub partition_w32: [[u16; 11]; PARTITION_CONTEXTS],
    /// `TilePartitionW64Cdf` (§8.3.1).
    pub partition_w64: [[u16; 11]; PARTITION_CONTEXTS],
    /// `TilePartitionW128Cdf` (§8.3.1).
    pub partition_w128: [[u16; 9]; PARTITION_CONTEXTS],
    /// `TileSkipCdf` (§8.3.1).
    pub skip: [[u16; 3]; SKIP_CONTEXTS],
    /// `TileSegmentIdCdf` (§8.3.1).
    pub segment_id: [[u16; MAX_SEGMENTS + 1]; SEGMENT_ID_CONTEXTS],

    // -----------------------------------------------------------------
    // Round 17 — motion-vector working CDFs. §8.3.1 enumerates each as
    // "`Mv*Cdf[ i ]` is set equal to a copy of `Default_Mv_*_Cdf` for
    // `i = 0..MV_CONTEXTS - 1`" (with the inner `comp = 0..1` axis
    // either replicated or carried by the source default).
    // -----------------------------------------------------------------
    /// `TileMvJointCdf[ MV_CONTEXTS ]` (§8.3.1).
    pub mv_joint: [[u16; MV_JOINTS + 1]; MV_CONTEXTS],
    /// `TileMvSignCdf[ MV_CONTEXTS ][ 2 ]` (§8.3.1). The `2` is
    /// `comp = 0..1`.
    pub mv_sign: [[[u16; 3]; MV_COMPS]; MV_CONTEXTS],
    /// `TileMvClassCdf[ MV_CONTEXTS ][ 2 ]` (§8.3.1).
    pub mv_class: [[[u16; MV_CLASSES + 1]; MV_COMPS]; MV_CONTEXTS],
    /// `TileMvClass0BitCdf[ MV_CONTEXTS ][ 2 ]` (§8.3.1).
    pub mv_class0_bit: [[[u16; 3]; MV_COMPS]; MV_CONTEXTS],
    /// `TileMvClass0FrCdf[ MV_CONTEXTS ][ 2 ][ CLASS0_SIZE ]` (§8.3.1).
    /// The §5.11.32 selection indexes by `[ MvCtx ][ comp ][ mv_class0_bit ]`.
    pub mv_class0_fr: [[[[u16; MV_JOINTS + 1]; CLASS0_SIZE]; MV_COMPS]; MV_CONTEXTS],
    /// `TileMvClass0HpCdf[ MV_CONTEXTS ][ 2 ]` (§8.3.1).
    pub mv_class0_hp: [[[u16; 3]; MV_COMPS]; MV_CONTEXTS],
    /// `TileMvBitCdf[ MV_CONTEXTS ][ 2 ][ MV_OFFSET_BITS ]` (§8.3.1).
    /// Selection: `[ MvCtx ][ comp ][ i ]`.
    pub mv_bit: [[[[u16; 3]; MV_OFFSET_BITS]; MV_COMPS]; MV_CONTEXTS],
    /// `TileMvFrCdf[ MV_CONTEXTS ][ 2 ]` (§8.3.1).
    pub mv_fr: [[[u16; MV_JOINTS + 1]; MV_COMPS]; MV_CONTEXTS],
    /// `TileMvHpCdf[ MV_CONTEXTS ][ 2 ]` (§8.3.1).
    pub mv_hp: [[[u16; 3]; MV_COMPS]; MV_CONTEXTS],

    // -----------------------------------------------------------------
    // Round 18 — inter-mode / reference-frame working CDFs. §8.3.1
    // enumerates each as "`*Cdf` is set to a copy of `Default_*_Cdf`".
    // -----------------------------------------------------------------
    /// `TileNewMvCdf[ NEW_MV_CONTEXTS ]` (§8.3.1).
    pub new_mv: [[u16; 3]; NEW_MV_CONTEXTS],
    /// `TileZeroMvCdf[ ZERO_MV_CONTEXTS ]` (§8.3.1).
    pub zero_mv: [[u16; 3]; ZERO_MV_CONTEXTS],
    /// `TileRefMvCdf[ REF_MV_CONTEXTS ]` (§8.3.1).
    pub ref_mv: [[u16; 3]; REF_MV_CONTEXTS],
    /// `TileDrlModeCdf[ DRL_MODE_CONTEXTS ]` (§8.3.1).
    pub drl_mode: [[u16; 3]; DRL_MODE_CONTEXTS],
    /// `TileIsInterCdf[ IS_INTER_CONTEXTS ]` (§8.3.1).
    pub is_inter: [[u16; 3]; IS_INTER_CONTEXTS],
    /// `TileCompModeCdf[ COMP_INTER_CONTEXTS ]` (§8.3.1).
    pub comp_mode: [[u16; 3]; COMP_INTER_CONTEXTS],
    /// `TileSkipModeCdf[ SKIP_MODE_CONTEXTS ]` (§8.3.1).
    pub skip_mode: [[u16; 3]; SKIP_MODE_CONTEXTS],
    /// `TileCompRefCdf[ REF_CONTEXTS ][ FWD_REFS - 1 ]` (§8.3.1).
    pub comp_ref: [[[u16; 3]; FWD_REFS - 1]; REF_CONTEXTS],
    /// `TileCompBwdRefCdf[ REF_CONTEXTS ][ BWD_REFS - 1 ]` (§8.3.1).
    pub comp_bwd_ref: [[[u16; 3]; BWD_REFS - 1]; REF_CONTEXTS],
    /// `TileSingleRefCdf[ REF_CONTEXTS ][ SINGLE_REFS - 1 ]` (§8.3.1).
    pub single_ref: [[[u16; 3]; SINGLE_REFS - 1]; REF_CONTEXTS],
    /// `TileCompoundModeCdf[ COMPOUND_MODE_CONTEXTS ]` (§8.3.1).
    pub compound_mode: [[u16; COMPOUND_MODES + 1]; COMPOUND_MODE_CONTEXTS],
    /// `TileCompRefTypeCdf[ COMP_REF_TYPE_CONTEXTS ]` (§8.3.1).
    pub comp_ref_type: [[u16; 3]; COMP_REF_TYPE_CONTEXTS],
    /// `TileUniCompRefCdf[ REF_CONTEXTS ][ UNIDIR_COMP_REFS - 1 ]`
    /// (§8.3.1).
    pub uni_comp_ref: [[[u16; 3]; UNIDIR_COMP_REFS - 1]; REF_CONTEXTS],

    // -----------------------------------------------------------------
    // Round 19 — palette / filter-intra / CFL working CDFs. §8.3.1
    // enumerates each as "`*Cdf` is set to a copy of `Default_*_Cdf`"
    // (no per-context broadcast for this group).
    // -----------------------------------------------------------------
    /// `TileFilterIntraModeCdf` (§8.3.1).
    pub filter_intra_mode: [u16; INTRA_FILTER_MODES + 1],
    /// `TileFilterIntraCdf[ BLOCK_SIZES ]` (§8.3.1).
    pub filter_intra: [[u16; 3]; BLOCK_SIZES],
    /// `TilePaletteYModeCdf[ PALETTE_BLOCK_SIZE_CONTEXTS ][ PALETTE_Y_MODE_CONTEXTS ]`
    /// (§8.3.1).
    pub palette_y_mode: [[[u16; 3]; PALETTE_Y_MODE_CONTEXTS]; PALETTE_BLOCK_SIZE_CONTEXTS],
    /// `TilePaletteUVModeCdf[ PALETTE_UV_MODE_CONTEXTS ]` (§8.3.1).
    pub palette_uv_mode: [[u16; 3]; PALETTE_UV_MODE_CONTEXTS],
    /// `TilePaletteYSizeCdf[ PALETTE_BLOCK_SIZE_CONTEXTS ]` (§8.3.1).
    pub palette_y_size: [[u16; PALETTE_SIZES + 1]; PALETTE_BLOCK_SIZE_CONTEXTS],
    /// `TilePaletteUVSizeCdf[ PALETTE_BLOCK_SIZE_CONTEXTS ]` (§8.3.1).
    pub palette_uv_size: [[u16; PALETTE_SIZES + 1]; PALETTE_BLOCK_SIZE_CONTEXTS],
    /// `TilePaletteSize2YColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_2_y_color: [[u16; 3]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize3YColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_3_y_color: [[u16; 4]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize4YColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_4_y_color: [[u16; 5]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize5YColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_5_y_color: [[u16; 6]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize6YColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_6_y_color: [[u16; 7]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize7YColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_7_y_color: [[u16; 8]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize8YColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_8_y_color: [[u16; 9]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize2UVColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_2_uv_color: [[u16; 3]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize3UVColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_3_uv_color: [[u16; 4]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize4UVColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_4_uv_color: [[u16; 5]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize5UVColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_5_uv_color: [[u16; 6]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize6UVColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_6_uv_color: [[u16; 7]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize7UVColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_7_uv_color: [[u16; 8]; PALETTE_COLOR_CONTEXTS],
    /// `TilePaletteSize8UVColorCdf[ PALETTE_COLOR_CONTEXTS ]` (§8.3.1).
    pub palette_size_8_uv_color: [[u16; 9]; PALETTE_COLOR_CONTEXTS],
    /// `TileCflSignCdf` (§8.3.1).
    pub cfl_sign: [u16; CFL_JOINT_SIGNS + 1],
    /// `TileCflAlphaCdf[ CFL_ALPHA_CONTEXTS ]` (§8.3.1).
    pub cfl_alpha: [[u16; CFL_ALPHABET_SIZE + 1]; CFL_ALPHA_CONTEXTS],

    // -----------------------------------------------------------------
    // Round 20 — transform-size working CDFs. §8.3.1 lists each as
    // "`Tx*Cdf` is set equal to a copy of `Default_Tx_*_Cdf`" and
    // "`TxfmSplitCdf` is set equal to a copy of
    // `Default_Txfm_Split_Cdf`".
    // -----------------------------------------------------------------
    /// `TileTx8x8Cdf[ TX_SIZE_CONTEXTS ]` (§8.3.1). Selected for
    /// `tx_depth` when `maxTxDepth == 1`.
    pub tx_8x8: [[u16; MAX_TX_DEPTH + 1]; TX_SIZE_CONTEXTS],
    /// `TileTx16x16Cdf[ TX_SIZE_CONTEXTS ]` (§8.3.1). Selected for
    /// `tx_depth` when `maxTxDepth == 2`.
    pub tx_16x16: [[u16; MAX_TX_DEPTH + 2]; TX_SIZE_CONTEXTS],
    /// `TileTx32x32Cdf[ TX_SIZE_CONTEXTS ]` (§8.3.1). Selected for
    /// `tx_depth` when `maxTxDepth == 3`.
    pub tx_32x32: [[u16; MAX_TX_DEPTH + 2]; TX_SIZE_CONTEXTS],
    /// `TileTx64x64Cdf[ TX_SIZE_CONTEXTS ]` (§8.3.1). Selected for
    /// `tx_depth` when `maxTxDepth == 4`.
    pub tx_64x64: [[u16; MAX_TX_DEPTH + 2]; TX_SIZE_CONTEXTS],
    /// `TileTxfmSplitCdf[ TXFM_PARTITION_CONTEXTS ]` (§8.3.1). Codes
    /// `txfm_split`.
    pub txfm_split: [[u16; 3]; TXFM_PARTITION_CONTEXTS],

    // -----------------------------------------------------------------
    // Round 21 — inter-frame transform-type group. §8.3.1 enumerates
    // each as "`InterTxTypeSet{n}Cdf` is set equal to a copy of
    // `Default_Inter_Tx_Type_Set{n}_Cdf`".
    // -----------------------------------------------------------------
    /// `TileInterTxTypeSet1Cdf[ INTER_TX_TYPE_SET1_SIZES ]` (§8.3.1).
    /// Selected for `inter_tx_type` when `set == TX_SET_INTER_1`.
    pub inter_tx_type_set1: [[u16; TX_TYPES + 1]; INTER_TX_TYPE_SET1_SIZES],
    /// `TileInterTxTypeSet2Cdf` (§8.3.1). Selected for `inter_tx_type`
    /// when `set == TX_SET_INTER_2`. Single row — no per-`Tx_Size_Sqr`
    /// axis (only `TX_16X16` reaches this set).
    pub inter_tx_type_set2: [u16; TX_TYPES_SET2 + 1],
    /// `TileInterTxTypeSet3Cdf[ INTER_TX_TYPE_SET3_SIZES ]` (§8.3.1).
    /// Selected for `inter_tx_type` when `set == TX_SET_INTER_3`.
    pub inter_tx_type_set3: [[u16; TX_TYPES_SET3 + 1]; INTER_TX_TYPE_SET3_SIZES],

    // -----------------------------------------------------------------
    // Round 22 — inter-frame interpolation-filter group. §8.3.1
    // enumerates as "`InterpFilterCdf` is set equal to a copy of
    // `Default_Interp_Filter_Cdf`".
    // -----------------------------------------------------------------
    /// `TileInterpFilterCdf[ INTERP_FILTER_CONTEXTS ]` (§8.3.1). Codes
    /// `interp_filter` (the switchable-filter selection).
    pub interp_filter: [[u16; INTERP_FILTERS + 1]; INTERP_FILTER_CONTEXTS],
}

impl TileCdfContext {
    /// §8.3.1: initialise every `Tile*Cdf` array from its `Default_*`
    /// table. Called at the start of tile parsing (and again when
    /// `init_non_coeff_cdfs()` is invoked per §7.4 / §5.11.4).
    ///
    /// The returned context is independent of [`DEFAULT_INTRA_FRAME_Y_MODE_CDF`]
    /// et al. (it is a value copy), so adapting it leaves the defaults
    /// untouched for the next tile's `new_from_defaults`.
    pub fn new_from_defaults() -> Self {
        // Per §8.3.1 the flat (per-comp, per-bit) defaults are
        // broadcast into a [MV_CONTEXTS][..] / [MV_CONTEXTS][2][..]
        // working set; expand them once here.
        let mv_sign_row: [[u16; 3]; MV_COMPS] = [DEFAULT_MV_SIGN_CDF, DEFAULT_MV_SIGN_CDF];
        let mv_class0_bit_row: [[u16; 3]; MV_COMPS] =
            [DEFAULT_MV_CLASS0_BIT_CDF, DEFAULT_MV_CLASS0_BIT_CDF];
        let mv_class0_hp_row: [[u16; 3]; MV_COMPS] =
            [DEFAULT_MV_CLASS0_HP_CDF, DEFAULT_MV_CLASS0_HP_CDF];
        let mv_hp_row: [[u16; 3]; MV_COMPS] = [DEFAULT_MV_HP_CDF, DEFAULT_MV_HP_CDF];
        let mv_bit_row: [[[u16; 3]; MV_OFFSET_BITS]; MV_COMPS] =
            [DEFAULT_MV_BIT_CDF, DEFAULT_MV_BIT_CDF];

        Self {
            intra_frame_y_mode: DEFAULT_INTRA_FRAME_Y_MODE_CDF,
            partition_w8: DEFAULT_PARTITION_W8_CDF,
            partition_w16: DEFAULT_PARTITION_W16_CDF,
            partition_w32: DEFAULT_PARTITION_W32_CDF,
            partition_w64: DEFAULT_PARTITION_W64_CDF,
            partition_w128: DEFAULT_PARTITION_W128_CDF,
            skip: DEFAULT_SKIP_CDF,
            segment_id: DEFAULT_SEGMENT_ID_CDF,

            mv_joint: [DEFAULT_MV_JOINT_CDF; MV_CONTEXTS],
            mv_sign: [mv_sign_row; MV_CONTEXTS],
            mv_class: [DEFAULT_MV_CLASS_CDF; MV_CONTEXTS],
            mv_class0_bit: [mv_class0_bit_row; MV_CONTEXTS],
            mv_class0_fr: [DEFAULT_MV_CLASS0_FR_CDF; MV_CONTEXTS],
            mv_class0_hp: [mv_class0_hp_row; MV_CONTEXTS],
            mv_bit: [mv_bit_row; MV_CONTEXTS],
            mv_fr: [DEFAULT_MV_FR_CDF; MV_CONTEXTS],
            mv_hp: [mv_hp_row; MV_CONTEXTS],

            // Round 18 — inter-mode / reference-frame group.
            new_mv: DEFAULT_NEW_MV_CDF,
            zero_mv: DEFAULT_ZERO_MV_CDF,
            ref_mv: DEFAULT_REF_MV_CDF,
            drl_mode: DEFAULT_DRL_MODE_CDF,
            is_inter: DEFAULT_IS_INTER_CDF,
            comp_mode: DEFAULT_COMP_MODE_CDF,
            skip_mode: DEFAULT_SKIP_MODE_CDF,
            comp_ref: DEFAULT_COMP_REF_CDF,
            comp_bwd_ref: DEFAULT_COMP_BWD_REF_CDF,
            single_ref: DEFAULT_SINGLE_REF_CDF,
            compound_mode: DEFAULT_COMPOUND_MODE_CDF,
            comp_ref_type: DEFAULT_COMP_REF_TYPE_CDF,
            uni_comp_ref: DEFAULT_UNI_COMP_REF_CDF,

            // Round 19 — palette / filter-intra / CFL group.
            filter_intra_mode: DEFAULT_FILTER_INTRA_MODE_CDF,
            filter_intra: DEFAULT_FILTER_INTRA_CDF,
            palette_y_mode: DEFAULT_PALETTE_Y_MODE_CDF,
            palette_uv_mode: DEFAULT_PALETTE_UV_MODE_CDF,
            palette_y_size: DEFAULT_PALETTE_Y_SIZE_CDF,
            palette_uv_size: DEFAULT_PALETTE_UV_SIZE_CDF,
            palette_size_2_y_color: DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF,
            palette_size_3_y_color: DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF,
            palette_size_4_y_color: DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF,
            palette_size_5_y_color: DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF,
            palette_size_6_y_color: DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF,
            palette_size_7_y_color: DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF,
            palette_size_8_y_color: DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF,
            palette_size_2_uv_color: DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF,
            palette_size_3_uv_color: DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF,
            palette_size_4_uv_color: DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF,
            palette_size_5_uv_color: DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF,
            palette_size_6_uv_color: DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF,
            palette_size_7_uv_color: DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF,
            palette_size_8_uv_color: DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF,
            cfl_sign: DEFAULT_CFL_SIGN_CDF,
            cfl_alpha: DEFAULT_CFL_ALPHA_CDF,

            // Round 20 — transform-size group.
            tx_8x8: DEFAULT_TX_8X8_CDF,
            tx_16x16: DEFAULT_TX_16X16_CDF,
            tx_32x32: DEFAULT_TX_32X32_CDF,
            tx_64x64: DEFAULT_TX_64X64_CDF,
            txfm_split: DEFAULT_TXFM_SPLIT_CDF,

            // Round 21 — inter-frame transform-type group.
            inter_tx_type_set1: DEFAULT_INTER_TX_TYPE_SET1_CDF,
            inter_tx_type_set2: DEFAULT_INTER_TX_TYPE_SET2_CDF,
            inter_tx_type_set3: DEFAULT_INTER_TX_TYPE_SET3_CDF,

            // Round 22 — inter-frame interpolation-filter group.
            interp_filter: DEFAULT_INTERP_FILTER_CDF,
        }
    }

    // -----------------------------------------------------------------
    // §8.3.2 selection: a syntax-element name + its block context maps
    // to a mutable reference to the right CDF row. The caller passes the
    // returned `&mut [u16]` straight to `SymbolDecoder::read_symbol`.
    // -----------------------------------------------------------------

    /// §8.3.2 `intra_frame_y_mode`: the cdf is
    /// `TileIntraFrameYModeCdf[ abovemode ][ leftmode ]`, where
    /// `abovemode` / `leftmode` are the [`INTRA_MODE_CONTEXT`]-mapped
    /// intra modes of the blocks immediately above / to the left.
    ///
    /// The caller supplies the already-mapped context indices (each in
    /// `0..INTRA_MODE_CONTEXTS`), since the neighbour-availability +
    /// `YModes[]` lookup belongs to the (not-yet-implemented) tile walk.
    /// [`intra_mode_ctx`] is provided for the mapping step.
    pub fn intra_frame_y_mode_cdf(&mut self, abovemode: usize, leftmode: usize) -> &mut [u16] {
        &mut self.intra_frame_y_mode[abovemode][leftmode]
    }

    /// §8.3.2 `partition`: select the `TilePartitionW{8,16,32,64,128}Cdf`
    /// array by `bsl` (= `Mi_Width_Log2[ bSize ]`, in `1..=5`) and index
    /// it by `ctx` (= `left * 2 + above`, in `0..PARTITION_CONTEXTS`).
    ///
    /// Returns `None` for a `bsl` outside `1..=5` (a caller bug — the
    /// partition syntax is never reached for other block sizes).
    pub fn partition_cdf(&mut self, bsl: u32, ctx: usize) -> Option<&mut [u16]> {
        match bsl {
            1 => Some(&mut self.partition_w8[ctx]),
            2 => Some(&mut self.partition_w16[ctx]),
            3 => Some(&mut self.partition_w32[ctx]),
            4 => Some(&mut self.partition_w64[ctx]),
            5 => Some(&mut self.partition_w128[ctx]),
            _ => None,
        }
    }

    /// §8.3.2 `skip`: the cdf is `TileSkipCdf[ ctx ]` where `ctx` is the
    /// sum of the above and left blocks' `Skips[]` (in
    /// `0..SKIP_CONTEXTS`).
    pub fn skip_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.skip[ctx]
    }

    /// §8.3.2 `segment_id`: the cdf is `TileSegmentIdCdf[ ctx ]` where
    /// `ctx` is computed from the neighbouring segment ids (in
    /// `0..SEGMENT_ID_CONTEXTS`); see [`segment_id_ctx`].
    pub fn segment_id_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.segment_id[ctx]
    }

    // -----------------------------------------------------------------
    // Round 17 — motion-vector §8.3.2 selectors. The shared `MvCtx`
    // input is derived from §5.11.31 `read_mv()`:
    //
    //   MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0
    //
    // See [`mv_ctx`].
    // -----------------------------------------------------------------

    /// §8.3.2 `mv_joint`: the cdf is `TileMvJointCdf[ MvCtx ]`.
    pub fn mv_joint_cdf(&mut self, mv_ctx: usize) -> &mut [u16] {
        &mut self.mv_joint[mv_ctx]
    }

    /// §8.3.2 `mv_sign`: the cdf is `TileMvSignCdf[ MvCtx ][ comp ]`,
    /// with `comp = 0` for the horizontal component and `comp = 1` for
    /// the vertical.
    pub fn mv_sign_cdf(&mut self, mv_ctx: usize, comp: usize) -> &mut [u16] {
        &mut self.mv_sign[mv_ctx][comp]
    }

    /// §8.3.2 `mv_class`: the cdf is `TileMvClassCdf[ MvCtx ][ comp ]`.
    pub fn mv_class_cdf(&mut self, mv_ctx: usize, comp: usize) -> &mut [u16] {
        &mut self.mv_class[mv_ctx][comp]
    }

    /// §8.3.2 `mv_class0_bit`: the cdf is
    /// `TileMvClass0BitCdf[ MvCtx ][ comp ]`. Only reached when
    /// §5.11.32 `read_mv_component()` saw `mv_class == MV_CLASS_0`.
    pub fn mv_class0_bit_cdf(&mut self, mv_ctx: usize, comp: usize) -> &mut [u16] {
        &mut self.mv_class0_bit[mv_ctx][comp]
    }

    /// §8.3.2 `mv_class0_fr`: the cdf is
    /// `TileMvClass0FrCdf[ MvCtx ][ comp ][ mv_class0_bit ]`. The
    /// caller supplies the already-decoded `mv_class0_bit` (in
    /// `0..CLASS0_SIZE`).
    pub fn mv_class0_fr_cdf(
        &mut self,
        mv_ctx: usize,
        comp: usize,
        mv_class0_bit: usize,
    ) -> &mut [u16] {
        &mut self.mv_class0_fr[mv_ctx][comp][mv_class0_bit]
    }

    /// §8.3.2 `mv_class0_hp`: the cdf is
    /// `TileMvClass0HpCdf[ MvCtx ][ comp ]`. Only reached when
    /// `allow_high_precision_mv == 1`.
    pub fn mv_class0_hp_cdf(&mut self, mv_ctx: usize, comp: usize) -> &mut [u16] {
        &mut self.mv_class0_hp[mv_ctx][comp]
    }

    /// §8.3.2 `mv_bit`: the cdf is `TileMvBitCdf[ MvCtx ][ comp ][ i ]`
    /// where `i` is the bit position currently being read by §5.11.32
    /// (`i = 0..mv_class - 1`, bounded above by `MV_OFFSET_BITS`).
    pub fn mv_bit_cdf(&mut self, mv_ctx: usize, comp: usize, i: usize) -> &mut [u16] {
        &mut self.mv_bit[mv_ctx][comp][i]
    }

    /// §8.3.2 `mv_fr`: the cdf is `TileMvFrCdf[ MvCtx ][ comp ]`. Only
    /// reached when `force_integer_mv == 0` and `mv_class != MV_CLASS_0`.
    pub fn mv_fr_cdf(&mut self, mv_ctx: usize, comp: usize) -> &mut [u16] {
        &mut self.mv_fr[mv_ctx][comp]
    }

    /// §8.3.2 `mv_hp`: the cdf is `TileMvHpCdf[ MvCtx ][ comp ]`. Only
    /// reached when `allow_high_precision_mv == 1` and
    /// `mv_class != MV_CLASS_0`.
    pub fn mv_hp_cdf(&mut self, mv_ctx: usize, comp: usize) -> &mut [u16] {
        &mut self.mv_hp[mv_ctx][comp]
    }

    // -----------------------------------------------------------------
    // Round 18 — inter-mode / reference-frame §8.3.2 selectors. The
    // caller pre-computes the §8.3.2 `ctx` (the spec has explicit
    // formulas for each — see [`is_inter_ctx`], [`skip_mode_ctx`],
    // [`ref_count_ctx`], [`compound_mode_ctx`]; the `comp_mode` and
    // `comp_ref_type` formulas need full tile-walk neighbour state and
    // are deferred) and passes the index straight to the array lookup.
    // -----------------------------------------------------------------

    /// §8.3.2 `new_mv`: the cdf is `TileNewMvCdf[ NewMvContext ]`. The
    /// `NewMvContext` is supplied by `find_mv_stack()` (in `0..NEW_MV_CONTEXTS`).
    pub fn new_mv_cdf(&mut self, new_mv_context: usize) -> &mut [u16] {
        &mut self.new_mv[new_mv_context]
    }

    /// §8.3.2 `zero_mv`: the cdf is `TileZeroMvCdf[ ZeroMvContext ]`.
    /// `ZeroMvContext` in `0..ZERO_MV_CONTEXTS`.
    pub fn zero_mv_cdf(&mut self, zero_mv_context: usize) -> &mut [u16] {
        &mut self.zero_mv[zero_mv_context]
    }

    /// §8.3.2 `ref_mv`: the cdf is `TileRefMvCdf[ RefMvContext ]`.
    /// `RefMvContext` in `0..REF_MV_CONTEXTS`.
    pub fn ref_mv_cdf(&mut self, ref_mv_context: usize) -> &mut [u16] {
        &mut self.ref_mv[ref_mv_context]
    }

    /// §8.3.2 `drl_mode`: the cdf is `TileDrlModeCdf[ DrlCtxStack[ idx ] ]`.
    /// The caller supplies the `DrlCtxStack[ idx ]` value in
    /// `0..DRL_MODE_CONTEXTS`.
    pub fn drl_mode_cdf(&mut self, drl_ctx: usize) -> &mut [u16] {
        &mut self.drl_mode[drl_ctx]
    }

    /// §8.3.2 `is_inter`: the cdf is `TileIsInterCdf[ ctx ]`, with `ctx`
    /// computed by [`is_inter_ctx`] (in `0..IS_INTER_CONTEXTS`).
    pub fn is_inter_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.is_inter[ctx]
    }

    /// §8.3.2 `comp_mode`: the cdf is `TileCompModeCdf[ ctx ]`. `ctx` is
    /// supplied in `0..COMP_INTER_CONTEXTS`; its §8.3.2 derivation needs
    /// `AvailU` / `AvailL` / `AboveSingle` / `LeftSingle` /
    /// `AboveRefFrame[0]` / `LeftRefFrame[0]` / `AboveIntra` /
    /// `LeftIntra` from the tile walk plus the spec's `check_backward(ref)`
    /// `(ref >= BWDREF_FRAME) && (ref <= ALTREF_FRAME)` predicate, so the
    /// branch ladder lives in the (future) tile-walk crate rather than as
    /// a standalone helper here.
    pub fn comp_mode_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.comp_mode[ctx]
    }

    /// §8.3.2 `skip_mode`: the cdf is `TileSkipModeCdf[ ctx ]`, with
    /// `ctx` computed by [`skip_mode_ctx`] (in `0..SKIP_MODE_CONTEXTS`).
    pub fn skip_mode_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.skip_mode[ctx]
    }

    /// §8.3.2 `comp_ref`: the cdf is `TileCompRefCdf[ ctx ][ p ]`, with
    /// `p ∈ {0, 1, 2}` selecting `comp_ref` / `comp_ref_p1` /
    /// `comp_ref_p2` (§8.3.2). `ctx` is `ref_count_ctx(..)` per the
    /// matching §8.3.2 paragraph.
    pub fn comp_ref_cdf(&mut self, ctx: usize, p: usize) -> &mut [u16] {
        &mut self.comp_ref[ctx][p]
    }

    /// §8.3.2 `comp_bwdref`: the cdf is `TileCompBwdRefCdf[ ctx ][ p ]`,
    /// with `p ∈ {0, 1}` selecting `comp_bwdref` / `comp_bwdref_p1`.
    pub fn comp_bwd_ref_cdf(&mut self, ctx: usize, p: usize) -> &mut [u16] {
        &mut self.comp_bwd_ref[ctx][p]
    }

    /// §8.3.2 `single_ref_p{1..6}`: the cdf is
    /// `TileSingleRefCdf[ ctx ][ p ]`, with `p ∈ {0..5}` selecting
    /// `single_ref_p1` .. `single_ref_p6` (the §8.3.2 list maps each
    /// to the `comp_*` paragraph that defines its `ctx`).
    pub fn single_ref_cdf(&mut self, ctx: usize, p: usize) -> &mut [u16] {
        &mut self.single_ref[ctx][p]
    }

    /// §8.3.2 `compound_mode`: the cdf is `TileCompoundModeCdf[ ctx ]`,
    /// with `ctx = Compound_Mode_Ctx_Map[ RefMvContext >> 1 ]
    /// [ Min(NewMvContext, COMP_NEWMV_CTXS - 1) ]`. See
    /// [`compound_mode_ctx`].
    pub fn compound_mode_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.compound_mode[ctx]
    }

    /// §8.3.2 `comp_ref_type`: the cdf is `TileCompRefTypeCdf[ ctx ]`,
    /// with `ctx` computed by the multi-branch paragraph in §8.3.2 (in
    /// `0..COMP_REF_TYPE_CONTEXTS`). The branch evaluator belongs in
    /// the (future) tile walk; this selector takes the already-computed
    /// index.
    pub fn comp_ref_type_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.comp_ref_type[ctx]
    }

    /// §8.3.2 `uni_comp_ref{,_p1,_p2}`: the cdf is
    /// `TileUniCompRefCdf[ ctx ][ p ]`, with `p ∈ {0, 1, 2}` selecting
    /// `uni_comp_ref` / `uni_comp_ref_p1` / `uni_comp_ref_p2`.
    pub fn uni_comp_ref_cdf(&mut self, ctx: usize, p: usize) -> &mut [u16] {
        &mut self.uni_comp_ref[ctx][p]
    }

    // -----------------------------------------------------------------
    // Round 19 — palette / filter-intra / CFL §8.3.2 selectors. The
    // caller pre-computes each `ctx` from the §5.11.x neighbour state
    // (helpers [`palette_y_mode_ctx`], [`palette_uv_mode_ctx`],
    // [`palette_color_ctx`], [`cfl_alpha_u_ctx`], [`cfl_alpha_v_ctx`]
    // cover the parts that need only scalar inputs; `bsizeCtx` is the
    // §5.11.46 block-size mapping the tile walk supplies).
    // -----------------------------------------------------------------

    /// §8.3.2 `use_filter_intra`: the cdf is `TileFilterIntraCdf[ MiSize ]`.
    pub fn filter_intra_cdf(&mut self, mi_size: usize) -> &mut [u16] {
        &mut self.filter_intra[mi_size]
    }

    /// §8.3.2 `filter_intra_mode`: the cdf is `TileFilterIntraModeCdf`
    /// (a single context-free row).
    pub fn filter_intra_mode_cdf(&mut self) -> &mut [u16] {
        &mut self.filter_intra_mode
    }

    /// §8.3.2 `has_palette_y`: the cdf is
    /// `TilePaletteYModeCdf[ bsizeCtx ][ ctx ]`. `bsizeCtx` is the
    /// §5.11.46 block-size class (in `0..PALETTE_BLOCK_SIZE_CONTEXTS`);
    /// `ctx` is the neighbour-palette count (see [`palette_y_mode_ctx`],
    /// in `0..PALETTE_Y_MODE_CONTEXTS`).
    pub fn palette_y_mode_cdf(&mut self, bsize_ctx: usize, ctx: usize) -> &mut [u16] {
        &mut self.palette_y_mode[bsize_ctx][ctx]
    }

    /// §8.3.2 `has_palette_uv`: the cdf is `TilePaletteUVModeCdf[ ctx ]`,
    /// with `ctx = (PaletteSizeY > 0) ? 1 : 0` (see [`palette_uv_mode_ctx`]).
    pub fn palette_uv_mode_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.palette_uv_mode[ctx]
    }

    /// §8.3.2 `palette_size_y_minus_2`: the cdf is
    /// `TilePaletteYSizeCdf[ bsizeCtx ]`.
    pub fn palette_y_size_cdf(&mut self, bsize_ctx: usize) -> &mut [u16] {
        &mut self.palette_y_size[bsize_ctx]
    }

    /// §8.3.2 `palette_size_uv_minus_2`: the cdf is
    /// `TilePaletteUVSizeCdf[ bsizeCtx ]`.
    pub fn palette_uv_size_cdf(&mut self, bsize_ctx: usize) -> &mut [u16] {
        &mut self.palette_uv_size[bsize_ctx]
    }

    /// §8.3.2 `palette_color_idx_y`: the cdf is
    /// `TilePaletteSize{2..8}YColorCdf[ ctx ]`, selected by `PaletteSizeY`
    /// (in `2..=PALETTE_COLORS`). `ctx = Palette_Color_Context[ ColorContextHash ]`
    /// (see [`palette_color_ctx`], in `0..PALETTE_COLOR_CONTEXTS`).
    ///
    /// Returns `None` for a `palette_size_y` outside `2..=PALETTE_COLORS`
    /// (a caller bug — the colour-index syntax is never reached otherwise).
    pub fn palette_y_color_cdf(&mut self, palette_size_y: usize, ctx: usize) -> Option<&mut [u16]> {
        match palette_size_y {
            2 => Some(&mut self.palette_size_2_y_color[ctx]),
            3 => Some(&mut self.palette_size_3_y_color[ctx]),
            4 => Some(&mut self.palette_size_4_y_color[ctx]),
            5 => Some(&mut self.palette_size_5_y_color[ctx]),
            6 => Some(&mut self.palette_size_6_y_color[ctx]),
            7 => Some(&mut self.palette_size_7_y_color[ctx]),
            8 => Some(&mut self.palette_size_8_y_color[ctx]),
            _ => None,
        }
    }

    /// §8.3.2 `palette_color_idx_uv`: the cdf is
    /// `TilePaletteSize{2..8}UVColorCdf[ ctx ]`, selected by `PaletteSizeUV`
    /// (in `2..=PALETTE_COLORS`). `ctx = Palette_Color_Context[ ColorContextHash ]`.
    ///
    /// Returns `None` for a `palette_size_uv` outside `2..=PALETTE_COLORS`.
    pub fn palette_uv_color_cdf(
        &mut self,
        palette_size_uv: usize,
        ctx: usize,
    ) -> Option<&mut [u16]> {
        match palette_size_uv {
            2 => Some(&mut self.palette_size_2_uv_color[ctx]),
            3 => Some(&mut self.palette_size_3_uv_color[ctx]),
            4 => Some(&mut self.palette_size_4_uv_color[ctx]),
            5 => Some(&mut self.palette_size_5_uv_color[ctx]),
            6 => Some(&mut self.palette_size_6_uv_color[ctx]),
            7 => Some(&mut self.palette_size_7_uv_color[ctx]),
            8 => Some(&mut self.palette_size_8_uv_color[ctx]),
            _ => None,
        }
    }

    /// §8.3.2 `cfl_alpha_signs`: the cdf is `TileCflSignCdf` (a single
    /// context-free row).
    pub fn cfl_sign_cdf(&mut self) -> &mut [u16] {
        &mut self.cfl_sign
    }

    /// §8.3.2 `cfl_alpha_u` / `cfl_alpha_v`: the cdf is
    /// `TileCflAlphaCdf[ ctx ]`, with `ctx` from [`cfl_alpha_u_ctx`] /
    /// [`cfl_alpha_v_ctx`] (in `0..CFL_ALPHA_CONTEXTS`).
    pub fn cfl_alpha_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.cfl_alpha[ctx]
    }

    /// §8.3.2 `tx_depth`: the cdf depends on the value of `maxTxDepth`
    /// and `ctx`. Per the §8.3.2 paragraph:
    ///
    /// * `TileTx64x64Cdf[ ctx ]` if `maxTxDepth == 4`
    /// * `TileTx32x32Cdf[ ctx ]` if `maxTxDepth == 3`
    /// * `TileTx16x16Cdf[ ctx ]` if `maxTxDepth == 2`
    /// * `TileTx8x8Cdf[ ctx ]` otherwise (`maxTxDepth == 1`)
    ///
    /// `ctx` is the [`tx_depth_ctx`] result (in `0..TX_SIZE_CONTEXTS`);
    /// `max_tx_depth` is the `Max_Tx_Depth[ MiSize ]` value from §5.11.15.
    /// Returns `None` when `max_tx_depth == 0` (no `tx_depth` is read in
    /// that case — §5.11.15 forces `TxSize = maxRectTxSize`).
    pub fn tx_depth_cdf(&mut self, max_tx_depth: u32, ctx: usize) -> Option<&mut [u16]> {
        match max_tx_depth {
            1 => Some(&mut self.tx_8x8[ctx]),
            2 => Some(&mut self.tx_16x16[ctx]),
            3 => Some(&mut self.tx_32x32[ctx]),
            4 => Some(&mut self.tx_64x64[ctx]),
            _ => None,
        }
    }

    /// §8.3.2 `txfm_split`: the cdf is `TileTxfmSplitCdf[ ctx ]`, where
    /// `ctx` is the [`txfm_split_ctx`] result (in
    /// `0..TXFM_PARTITION_CONTEXTS`).
    pub fn txfm_split_cdf(&mut self, ctx: usize) -> &mut [u16] {
        &mut self.txfm_split[ctx]
    }

    /// §8.3.2 `inter_tx_type`: the cdf depends on the §5.11.48
    /// `get_tx_set()` value as follows (spec §8.3.2, "inter_tx_type"):
    ///
    /// * `TX_SET_INTER_1` ⇒ `TileInterTxTypeSet1Cdf[ Tx_Size_Sqr[ txSz ] ]`
    /// * `TX_SET_INTER_2` ⇒ `TileInterTxTypeSet2Cdf` (no `txSz` axis)
    /// * `TX_SET_INTER_3` ⇒ `TileInterTxTypeSet3Cdf[ Tx_Size_Sqr[ txSz ] ]`
    ///
    /// `set` is the [`inter_tx_type_set`] result; `tx_size_sqr` is the
    /// `Tx_Size_Sqr[ txSz ]` value supplied by §5.11.47 (in
    /// `0..TX_SIZES`, but only the first 2 / 4 entries of Set1 / Set3
    /// are addressable per the §5.11.48 routing). Returns `None` for
    /// the §5.11.47 `set == TX_SET_DCTONLY` path (where `inter_tx_type`
    /// is not read — `TxType = DCT_DCT` is forced), and for any
    /// out-of-range `set` / `tx_size_sqr` combination that §5.11.48
    /// would not actually produce.
    pub fn inter_tx_type_cdf(&mut self, set: u32, tx_size_sqr: u32) -> Option<&mut [u16]> {
        let sqr = tx_size_sqr as usize;
        match set {
            TX_SET_INTER_1 if sqr < INTER_TX_TYPE_SET1_SIZES => {
                Some(&mut self.inter_tx_type_set1[sqr])
            }
            TX_SET_INTER_2 => Some(&mut self.inter_tx_type_set2),
            TX_SET_INTER_3 if sqr < INTER_TX_TYPE_SET3_SIZES => {
                Some(&mut self.inter_tx_type_set3[sqr])
            }
            _ => None,
        }
    }

    /// §8.3.2 `interp_filter`: the cdf is `TileInterpFilterCdf[ ctx ]`,
    /// with `ctx` in `0..INTERP_FILTER_CONTEXTS` (see
    /// [`interp_filter_ctx`]). Returns `None` if `ctx` is out of range
    /// (a caller bug — the §8.3.2 formula bounds the result).
    pub fn interp_filter_cdf(&mut self, ctx: usize) -> Option<&mut [u16]> {
        if ctx < INTERP_FILTER_CONTEXTS {
            Some(&mut self.interp_filter[ctx])
        } else {
            None
        }
    }
}

impl Default for TileCdfContext {
    fn default() -> Self {
        Self::new_from_defaults()
    }
}

// ---------------------------------------------------------------------
// §8.3.2 context-derivation helpers (the parts that need only scalar
// neighbour inputs — the full neighbour lookups live in the tile walk).
// ---------------------------------------------------------------------

/// §8.3.2 `intra_frame_y_mode` context mapping:
/// `Intra_Mode_Context[ mode ]`. `mode` is a neighbour's `YMode` (or
/// `DC_PRED == 0` when that neighbour is unavailable).
pub fn intra_mode_ctx(mode: usize) -> usize {
    INTRA_MODE_CONTEXT[mode]
}

/// §8.3.2 `partition` context:
///
/// ```text
///   ctx = left * 2 + above
/// ```
///
/// where `above` / `left` are the booleans
/// `AvailU && (Mi_Width_Log2[..] < bsl)` /
/// `AvailL && (Mi_Height_Log2[..] < bsl)` evaluated by the tile walk.
pub fn partition_ctx(above: bool, left: bool) -> usize {
    (left as usize) * 2 + (above as usize)
}

/// §8.3.2 `skip` context:
///
/// ```text
///   ctx = 0
///   if ( AvailU ) ctx += Skips[ MiRow - 1 ][ MiCol ]
///   if ( AvailL ) ctx += Skips[ MiRow ][ MiCol - 1 ]
/// ```
///
/// `above_skip` / `left_skip` are the neighbour `Skips[]` values (0 or
/// 1), already gated on `AvailU` / `AvailL` by the caller (an
/// unavailable neighbour contributes 0).
pub fn skip_ctx(above_skip: u8, left_skip: u8) -> usize {
    (above_skip + left_skip) as usize
}

/// §8.3.2 `segment_id` context:
///
/// ```text
///   if ( prevUL < 0 )                                        ctx = 0
///   else if ( (prevUL == prevU) && (prevUL == prevL) )        ctx = 2
///   else if ( (prevUL == prevU) || (prevUL == prevL)
///                                || (prevU  == prevL) )        ctx = 1
///   else                                                     ctx = 0
/// ```
///
/// `prevUL` / `prevU` / `prevL` are the above-left / above / left
/// neighbour segment ids; an unavailable neighbour is signalled with
/// [`None`] (the spec's negative sentinel). When `prev_ul` is `None`
/// the result is `0` regardless of the others.
pub fn segment_id_ctx(prev_ul: Option<i32>, prev_u: Option<i32>, prev_l: Option<i32>) -> usize {
    // prevUL < 0 (unavailable) ⇒ ctx = 0 unconditionally.
    let ul = match prev_ul {
        Some(v) => v,
        None => return 0,
    };
    // A missing U or L neighbour is the spec's negative sentinel and so
    // cannot equal anything (not even another missing neighbour — the
    // spec compares concrete segment ids, not the sentinel).
    let ul_eq_u = prev_u == Some(ul);
    let ul_eq_l = prev_l == Some(ul);
    let u_eq_l = match (prev_u, prev_l) {
        (Some(u), Some(l)) => u == l,
        _ => false,
    };
    if ul_eq_u && ul_eq_l {
        2
    } else if ul_eq_u || ul_eq_l || u_eq_l {
        1
    } else {
        0
    }
}

/// §5.11.31 `read_mv()` `MvCtx` derivation. Returns
/// [`MV_INTRABC_CONTEXT`] when `use_intrabc == 1` and `0` otherwise;
/// the result is the first index into every `Mv*Cdf` selector above.
pub fn mv_ctx(use_intrabc: bool) -> usize {
    if use_intrabc {
        MV_INTRABC_CONTEXT
    } else {
        0
    }
}

// ---------------------------------------------------------------------
// Round 18 — inter-mode / reference-frame §8.3.2 context helpers. Each
// directly transcribes one §8.3.2 paragraph (the scalar fragment that
// needs only neighbour-summary inputs; the AvailU/AvailL gating and the
// neighbour ref-frame lookups belong to the tile walk).
// ---------------------------------------------------------------------

/// §8.3.2 `is_inter` context:
///
/// ```text
///   if ( AvailU && AvailL )
///        ctx = (LeftIntra && AboveIntra) ? 3 : LeftIntra || AboveIntra
///   else if ( AvailU || AvailL )
///        ctx = 2 * (AvailU ? AboveIntra : LeftIntra)
///   else
///        ctx = 0
/// ```
///
/// Each input is `Some(true)` (neighbour is intra) / `Some(false)`
/// (neighbour is inter), or `None` if the neighbour is unavailable.
pub fn is_inter_ctx(above_intra: Option<bool>, left_intra: Option<bool>) -> usize {
    match (above_intra, left_intra) {
        (Some(a), Some(l)) => {
            if a && l {
                3
            } else {
                (a || l) as usize
            }
        }
        (Some(a), None) => 2 * (a as usize),
        (None, Some(l)) => 2 * (l as usize),
        (None, None) => 0,
    }
}

/// §8.3.2 `skip_mode` context: sum of the neighbour `SkipModes[]` flags
/// (0 if the neighbour is unavailable).
///
/// ```text
///   ctx = 0
///   if ( AvailU ) ctx += SkipModes[ MiRow - 1 ][ MiCol ]
///   if ( AvailL ) ctx += SkipModes[ MiRow ][ MiCol - 1 ]
/// ```
///
/// Result in `0..SKIP_MODE_CONTEXTS`.
pub fn skip_mode_ctx(above_skip_mode: u8, left_skip_mode: u8) -> usize {
    (above_skip_mode + left_skip_mode) as usize
}

/// §8.3.2 `ref_count_ctx` (the inner helper used by every
/// `single_ref_p*` / `comp_ref` / `comp_bwdref` / `uni_comp_ref_p*`
/// selection):
///
/// ```text
///   if ( counts0 < counts1 )       return 0
///   else if ( counts0 == counts1 ) return 1
///   else                           return 2
/// ```
///
/// Result in `0..REF_CONTEXTS`.
pub fn ref_count_ctx(counts0: u32, counts1: u32) -> usize {
    use core::cmp::Ordering;
    match counts0.cmp(&counts1) {
        Ordering::Less => 0,
        Ordering::Equal => 1,
        Ordering::Greater => 2,
    }
}

/// §8.3.2 `compound_mode` context:
///
/// ```text
///   ctx = Compound_Mode_Ctx_Map[ RefMvContext >> 1 ]
///                              [ Min(NewMvContext, COMP_NEWMV_CTXS - 1) ]
/// ```
///
/// `ref_mv_context` is the §8.3.2 `RefMvContext`; `new_mv_context` is the
/// §8.3.2 `NewMvContext`. Returns a value in `0..COMPOUND_MODE_CONTEXTS`.
///
/// The `RefMvContext >> 1` selects one of the three [`COMPOUND_MODE_CTX_MAP`]
/// rows; the `Min(.., COMP_NEWMV_CTXS - 1)` clamps the `NewMvContext`
/// to the map's 5-wide second axis.
pub fn compound_mode_ctx(ref_mv_context: usize, new_mv_context: usize) -> usize {
    let row = (ref_mv_context >> 1).min(COMPOUND_MODE_CTX_MAP.len() - 1);
    let col = new_mv_context.min(COMP_NEWMV_CTXS - 1);
    COMPOUND_MODE_CTX_MAP[row][col]
}

// ---------------------------------------------------------------------
// Round 19 — palette / filter-intra / CFL §8.3.2 context helpers. Each
// computes a `ctx` from the scalar neighbour inputs the §5.11.x syntax
// supplies (the §5.11.46 `bsizeCtx` itself comes from the block-size
// tables in the tile walk and is passed through directly).
// ---------------------------------------------------------------------

/// §8.3.2 `has_palette_y` context:
///
/// ```text
///   ctx = 0
///   if ( AvailU && PaletteSizes[ 0 ][ MiRow - 1 ][ MiCol ] > 0 ) ctx += 1
///   if ( AvailL && PaletteSizes[ 0 ][ MiRow ][ MiCol - 1 ] > 0 ) ctx += 1
/// ```
///
/// The caller passes whether the above / left neighbour (when available)
/// carries a non-empty Y palette. Returns a value in
/// `0..PALETTE_Y_MODE_CONTEXTS`.
pub fn palette_y_mode_ctx(above_has_palette: bool, left_has_palette: bool) -> usize {
    above_has_palette as usize + left_has_palette as usize
}

/// §8.3.2 `has_palette_uv` context: `ctx = (PaletteSizeY > 0) ? 1 : 0`.
/// Returns a value in `0..PALETTE_UV_MODE_CONTEXTS`.
pub fn palette_uv_mode_ctx(palette_size_y: usize) -> usize {
    (palette_size_y > 0) as usize
}

/// §8.3.2 `palette_color_idx_y` / `palette_color_idx_uv` context:
/// `ctx = Palette_Color_Context[ ColorContextHash ]`.
///
/// The §5.11.50 colour-context derivation only ever produces a
/// `ColorContextHash` for which [`PALETTE_COLOR_CONTEXT`] holds a
/// non-negative entry; this helper returns `None` for the spec's `-1`
/// sentinels (which "will never be accessed") and for an out-of-range
/// hash. On the valid path the result is in `0..PALETTE_COLOR_CONTEXTS`.
pub fn palette_color_ctx(color_context_hash: usize) -> Option<usize> {
    let mapped = *PALETTE_COLOR_CONTEXT.get(color_context_hash)?;
    if mapped < 0 {
        None
    } else {
        Some(mapped as usize)
    }
}

/// §8.3.2 `cfl_alpha_u` context: `ctx = (signU - 1) * 3 + signV`, which
/// the spec notes equals `cfl_alpha_signs - 2`. `sign_u` / `sign_v` are
/// the §5.11.45 joint-sign components (`CFL_SIGN_ZERO == 0`,
/// `CFL_SIGN_NEG == 1`, `CFL_SIGN_POS == 2`). Only reached when
/// `sign_u != CFL_SIGN_ZERO`, so the result is in `0..CFL_ALPHA_CONTEXTS`.
pub fn cfl_alpha_u_ctx(sign_u: usize, sign_v: usize) -> usize {
    (sign_u - 1) * 3 + sign_v
}

/// §8.3.2 `cfl_alpha_v` context: `ctx = (signV - 1) * 3 + signU`. Only
/// reached when `sign_v != CFL_SIGN_ZERO`, so the result is in
/// `0..CFL_ALPHA_CONTEXTS`.
pub fn cfl_alpha_v_ctx(sign_u: usize, sign_v: usize) -> usize {
    (sign_v - 1) * 3 + sign_u
}

// ---------------------------------------------------------------------
// Round 20 — transform-size §8.3.2 context helpers. Each computes a
// `ctx` from scalar inputs the caller derives from the §5.11.15 /
// §5.11.16 syntax + the local neighbour state. The full neighbour
// `get_above_tx_width` / `get_left_tx_height` walks live in the tile
// walk (none of those tables are yet tracked here); the helpers below
// take the already-computed `aboveW` / `leftH` width-and-height
// neighbours and the `maxTxWidth` / `maxTxHeight` from the spec's
// `Tx_Width` / `Tx_Height` tables.
// ---------------------------------------------------------------------

/// §8.3.2 `tx_depth` context formula:
///
/// ```text
///   ctx = (aboveW >= maxTxWidth) + (leftH >= maxTxHeight)
/// ```
///
/// `above_w` and `left_h` are the §8.3.2-defined neighbour widths /
/// heights (`0` when the neighbour is unavailable; the
/// `Block_Width[ MiSizes[..] ]` / `get_above_tx_width(..)` /
/// `get_left_tx_height(..)` ladders for the present-neighbour case
/// happen in the tile walk). `max_tx_width` / `max_tx_height` are the
/// `Tx_Width[ maxRectTxSize ]` / `Tx_Height[ maxRectTxSize ]` values.
/// Result is in `0..TX_SIZE_CONTEXTS`.
pub fn tx_depth_ctx(above_w: u32, left_h: u32, max_tx_width: u32, max_tx_height: u32) -> usize {
    (above_w >= max_tx_width) as usize + (left_h >= max_tx_height) as usize
}

/// §8.3.2 `txfm_split` context formula:
///
/// ```text
///   above = get_above_tx_width( row, col ) < Tx_Width[ txSz ]
///   left  = get_left_tx_height( row, col ) < Tx_Height[ txSz ]
///   size  = Min( 64, Max( Block_Width[ MiSize ], Block_Height[ MiSize ] ) )
///   maxTxSz = find_tx_size( size, size )
///   txSzSqrUp = Tx_Size_Sqr_Up[ txSz ]
///   ctx = (txSzSqrUp != maxTxSz) * 3
///       + (TX_SIZES - 1 - maxTxSz) * 6
///       + above + left
/// ```
///
/// `above` / `left` are the two booleans the spec computes from
/// `get_above_tx_width` / `get_left_tx_height` against `Tx_Width[ txSz ]`
/// / `Tx_Height[ txSz ]`; `tx_sz_sqr_up` is `Tx_Size_Sqr_Up[ txSz ]` and
/// `max_tx_sz` is `find_tx_size( size, size )` for `size = Min( 64,
/// Max( Block_Width, Block_Height ) )` — both `TX_4X4..TX_64X64`
/// indices.
///
/// Returns `None` if the computed ctx would land outside
/// `0..TXFM_PARTITION_CONTEXTS` (a caller-input bug; the spec's
/// reachable combinations stay in range).
pub fn txfm_split_ctx(above: bool, left: bool, tx_sz_sqr_up: u32, max_tx_sz: u32) -> Option<usize> {
    if max_tx_sz as usize >= TX_SIZES {
        return None;
    }
    let split = (tx_sz_sqr_up != max_tx_sz) as usize;
    let size_term = (TX_SIZES - 1 - max_tx_sz as usize).checked_mul(6)?;
    let ctx = split * 3 + size_term + (above as usize) + (left as usize);
    if ctx < TXFM_PARTITION_CONTEXTS {
        Some(ctx)
    } else {
        None
    }
}

// ---------------------------------------------------------------------
// Round 21 — inter-frame transform-type §8.3.2 helpers. Mirrors the
// `is_inter == 1` branch of §5.11.48 `get_tx_set()`.
// ---------------------------------------------------------------------

/// §5.11.48 `get_tx_set()` (the `is_inter == 1` branch). Maps the
/// `txSzSqr` / `txSzSqrUp` pair (and the frame's `reduced_tx_set`) to
/// the inter transform-set index `set ∈ { TX_SET_DCTONLY,
/// TX_SET_INTER_1, TX_SET_INTER_2, TX_SET_INTER_3 }`.
///
/// ```text
///   if ( txSzSqrUp > TX_32X32 )                              return TX_SET_DCTONLY
///   if ( reduced_tx_set || txSzSqrUp == TX_32X32 )           return TX_SET_INTER_3
///   else if ( txSzSqr == TX_16X16 )                          return TX_SET_INTER_2
///   return TX_SET_INTER_1
/// ```
///
/// `tx_sz_sqr` and `tx_sz_sqr_up` are `Tx_Size_Sqr[ txSz ]` and
/// `Tx_Size_Sqr_Up[ txSz ]` (`TX_4X4 = 0..TX_64X64 = 4`). The intra
/// branch is left for a follow-up round (it carries an `intraDir`
/// axis and selects a different default-cdf family).
pub fn inter_tx_type_set(tx_sz_sqr: u32, tx_sz_sqr_up: u32, reduced_tx_set: bool) -> u32 {
    // §3 `TX_32X32 = 3` per the spec's per-`TX_*` constant table.
    const TX_16X16: u32 = 2;
    const TX_32X32: u32 = 3;
    if tx_sz_sqr_up > TX_32X32 {
        TX_SET_DCTONLY
    } else if reduced_tx_set || tx_sz_sqr_up == TX_32X32 {
        TX_SET_INTER_3
    } else if tx_sz_sqr == TX_16X16 {
        TX_SET_INTER_2
    } else {
        TX_SET_INTER_1
    }
}

// ---------------------------------------------------------------------
// Round 22 — inter-frame interpolation-filter §8.3.2 helper. The
// §8.3.2 ctx formula folds the §5.11.x `dir` / `RefFrame[1]` scope
// inputs with the two `(above|left)Type` neighbour-filter inputs into
// a single `0..INTERP_FILTER_CONTEXTS` index.
// ---------------------------------------------------------------------

/// Sentinel returned by the §8.3.2 `aboveType` / `leftType` neighbour
/// reads when the neighbour is unavailable or carries a different
/// reference frame (so its `InterpFilters` entry doesn't count as a
/// matching neighbour). Per §8.3.2 the spec's literal initialiser
/// `aboveType = 3; leftType = 3` matches `INTERP_FILTERS`, i.e. one
/// past the highest reachable filter index.
pub const INTERP_FILTER_NONE: usize = INTERP_FILTERS;

/// §8.3.2 `interp_filter` context. Mirrors the spec block:
///
/// ```text
///   ctx = ((dir & 1) * 2 + (RefFrame[1] > INTRA_FRAME)) * 4
///   leftType = 3
///   aboveType = 3
///
///   if (AvailL) {
///       if (RefFrames[ MiRow ][ MiCol - 1 ][ 0 ] == RefFrame[ 0 ] ||
///           RefFrames[ MiRow ][ MiCol - 1 ][ 1 ] == RefFrame[ 0 ])
///           leftType = InterpFilters[ MiRow ][ MiCol - 1 ][ dir ]
///   }
///   if (AvailU) {
///       if (RefFrames[ MiRow - 1 ][ MiCol ][ 0 ] == RefFrame[ 0 ] ||
///           RefFrames[ MiRow - 1 ][ MiCol ][ 1 ] == RefFrame[ 0 ])
///           aboveType = InterpFilters[ MiRow - 1 ][ MiCol ][ dir ]
///   }
///
///   if (leftType == aboveType)        ctx += leftType
///   else if (leftType == 3)           ctx += aboveType
///   else if (aboveType == 3)          ctx += leftType
///   else                              ctx += 3
/// ```
///
/// The caller supplies the already-resolved `above_type` / `left_type`
/// in `0..INTERP_FILTERS` for a matching neighbour, or
/// [`INTERP_FILTER_NONE`] (== `INTERP_FILTERS`, the spec's literal `3`)
/// for an unavailable / mismatched neighbour — the §5.11.x neighbour
/// walk owns that resolution. `dir` is 0 (horizontal) or 1 (vertical),
/// and `is_compound` is the §5.11.27 `isCompound = RefFrame[1] >
/// INTRA_FRAME` derivation.
///
/// Returns `Some(ctx)` in `0..INTERP_FILTER_CONTEXTS`, or `None` if
/// any input is out of the spec-bounded range (a caller bug —
/// `above_type` / `left_type` strictly bounded to `0..=INTERP_FILTERS`,
/// `dir <= 1`).
pub fn interp_filter_ctx(
    above_type: usize,
    left_type: usize,
    dir: u32,
    is_compound: bool,
) -> Option<usize> {
    if dir > 1 {
        return None;
    }
    if above_type > INTERP_FILTERS || left_type > INTERP_FILTERS {
        return None;
    }
    let mut ctx = (((dir & 1) as usize) * 2 + (is_compound as usize)) * 4;
    if left_type == above_type {
        ctx += left_type;
    } else if left_type == INTERP_FILTER_NONE {
        ctx += above_type;
    } else if above_type == INTERP_FILTER_NONE {
        ctx += left_type;
    } else {
        ctx += INTERP_FILTERS;
    }
    if ctx < INTERP_FILTER_CONTEXTS {
        Some(ctx)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbol_decoder::SymbolDecoder;

    /// §8.3.1: a fresh context is a verbatim copy of the §9.4 defaults,
    /// and the well-formedness invariants the §8.2.6 decoder relies on
    /// hold for every row: the second-to-last entry is `1 << 15` and the
    /// last (counter) entry is 0.
    #[test]
    fn init_from_defaults_copies_tables() {
        let ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.intra_frame_y_mode, DEFAULT_INTRA_FRAME_Y_MODE_CDF);
        assert_eq!(ctx.partition_w8, DEFAULT_PARTITION_W8_CDF);
        assert_eq!(ctx.partition_w16, DEFAULT_PARTITION_W16_CDF);
        assert_eq!(ctx.partition_w32, DEFAULT_PARTITION_W32_CDF);
        assert_eq!(ctx.partition_w64, DEFAULT_PARTITION_W64_CDF);
        assert_eq!(ctx.partition_w128, DEFAULT_PARTITION_W128_CDF);
        assert_eq!(ctx.skip, DEFAULT_SKIP_CDF);
        assert_eq!(ctx.segment_id, DEFAULT_SEGMENT_ID_CDF);

        // §8.2.6 contract checks on every transcribed row.
        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for a in &DEFAULT_INTRA_FRAME_Y_MODE_CDF {
            for r in a {
                check(r);
            }
        }
        for r in &DEFAULT_PARTITION_W8_CDF {
            check(r);
        }
        for r in &DEFAULT_PARTITION_W16_CDF {
            check(r);
        }
        for r in &DEFAULT_PARTITION_W32_CDF {
            check(r);
        }
        for r in &DEFAULT_PARTITION_W64_CDF {
            check(r);
        }
        for r in &DEFAULT_PARTITION_W128_CDF {
            check(r);
        }
        for r in &DEFAULT_SKIP_CDF {
            check(r);
        }
        for r in &DEFAULT_SEGMENT_ID_CDF {
            check(r);
        }
    }

    /// §8.3.1 independence: adapting the working copy must not mutate the
    /// `Default_*` source (the next tile re-inits from it).
    #[test]
    fn working_copy_is_independent_of_defaults() {
        let mut ctx = TileCdfContext::new_from_defaults();
        ctx.skip_cdf(0)[0] = 12345;
        assert_ne!(ctx.skip[0][0], DEFAULT_SKIP_CDF[0][0]);
        // The immutable source is untouched.
        assert_eq!(DEFAULT_SKIP_CDF[0][0], 31671);
    }

    /// §8.3.2 `Intra_Mode_Context[]` mapping, term by term.
    #[test]
    fn intra_mode_context_maps_per_spec() {
        let expected = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];
        for (mode, &want) in expected.iter().enumerate() {
            assert_eq!(intra_mode_ctx(mode), want);
        }
    }

    /// §8.3.2 `partition` ctx = `left * 2 + above`.
    #[test]
    fn partition_context_formula() {
        assert_eq!(partition_ctx(false, false), 0);
        assert_eq!(partition_ctx(true, false), 1);
        assert_eq!(partition_ctx(false, true), 2);
        assert_eq!(partition_ctx(true, true), 3);
    }

    /// §8.3.2 `partition` array selection by `bsl`.
    #[test]
    fn partition_cdf_selected_by_bsl() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.partition_cdf(1, 0).unwrap().len(), 5); // W8
        assert_eq!(ctx.partition_cdf(2, 0).unwrap().len(), 11); // W16
        assert_eq!(ctx.partition_cdf(3, 0).unwrap().len(), 11); // W32
        assert_eq!(ctx.partition_cdf(4, 0).unwrap().len(), 11); // W64
        assert_eq!(ctx.partition_cdf(5, 0).unwrap().len(), 9); // W128
        assert!(ctx.partition_cdf(0, 0).is_none());
        assert!(ctx.partition_cdf(6, 0).is_none());
        // The selected row matches the §9.4 default for that ctx.
        assert_eq!(
            ctx.partition_cdf(2, 3).unwrap(),
            &DEFAULT_PARTITION_W16_CDF[3]
        );
    }

    /// §8.3.2 `skip` ctx = sum of neighbour `Skips[]`.
    #[test]
    fn skip_context_sum() {
        assert_eq!(skip_ctx(0, 0), 0);
        assert_eq!(skip_ctx(1, 0), 1);
        assert_eq!(skip_ctx(0, 1), 1);
        assert_eq!(skip_ctx(1, 1), 2);
    }

    /// §8.3.2 `segment_id` ctx derivation across the four branches.
    #[test]
    fn segment_id_context_branches() {
        // prevUL < 0 (unavailable) ⇒ 0.
        assert_eq!(segment_id_ctx(None, Some(1), Some(1)), 0);
        // all three equal ⇒ 2.
        assert_eq!(segment_id_ctx(Some(3), Some(3), Some(3)), 2);
        // exactly one pair equal ⇒ 1.
        assert_eq!(segment_id_ctx(Some(3), Some(3), Some(5)), 1); // UL==U
        assert_eq!(segment_id_ctx(Some(3), Some(5), Some(3)), 1); // UL==L
        assert_eq!(segment_id_ctx(Some(3), Some(5), Some(5)), 1); // U==L
                                                                  // all distinct ⇒ 0.
        assert_eq!(segment_id_ctx(Some(3), Some(5), Some(7)), 0);
        // A missing U/L cannot equal a present UL, so it falls through.
        assert_eq!(segment_id_ctx(Some(3), None, Some(3)), 1); // UL==L
        assert_eq!(segment_id_ctx(Some(3), None, None), 0);
    }

    /// End-to-end: decode a `skip` symbol through a default CDF selected
    /// by §8.3.2, driving the real §8.2 `SymbolDecoder`.
    ///
    /// We pick `ctx = 2` (`Default_Skip_Cdf[2] = {4576, 32768, 0}`, a
    /// strongly-toward-1 distribution) and a window whose `SymbolValue`
    /// lands in the high (symbol-1) region, then assert both the decoded
    /// value and that the §8.3 update mutated the working copy while the
    /// §9.4 source stayed put.
    #[test]
    fn decode_skip_through_default_cdf() {
        // sz = 2 ⇒ numBits = 15. bytes = 0xFF 0xFE ⇒ top 15 bits =
        // 111111111111111 = 0x7FFF; SymbolValue = 0x7FFF ^ 0x7FFF = 0.
        // A SymbolValue of 0 is below every `cur` boundary, so the
        // §8.2.6 search returns the LAST symbol (here symbol 1).
        let bytes = [0xFFu8, 0xFEu8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 2, false).unwrap();
        assert_eq!(dec.symbol_value(), 0);

        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.skip;
        let cdf = ctx.skip_cdf(2);
        let sym = dec.read_symbol(cdf).unwrap();
        assert_eq!(sym, 1, "SymbolValue 0 selects the final (skip) symbol");

        // §8.3 update ran (disable_cdf_update == false at init): the
        // counter advanced and the row changed.
        assert_ne!(ctx.skip, before, "read_symbol must adapt the working CDF");
        assert_eq!(ctx.skip[2][2], 1, "§8.3 counter incremented to 1");
        // The §9.4 source is immutable.
        assert_eq!(DEFAULT_SKIP_CDF[2], [4576, 32768, 0]);
    }

    /// End-to-end through a multisymbol partition CDF, confirming the
    /// §8.3.2-selected row drives a valid §8.2.6 decode in range.
    #[test]
    fn decode_partition_through_default_cdf() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        // bsl = 2 (W16), ctx = partition_ctx(above=true, left=false) = 1.
        let pctx = partition_ctx(true, false);
        assert_eq!(pctx, 1);
        let cdf = ctx.partition_cdf(2, pctx).unwrap();
        let sym = dec.read_symbol(cdf).unwrap();
        // W16 codes a 10-value symbol.
        assert!(sym < 10, "partition symbol in 0..10, got {sym}");
        // disable_cdf_update was true ⇒ the row is untouched.
        assert_eq!(ctx.partition_w16[1], DEFAULT_PARTITION_W16_CDF[1]);
    }

    // -----------------------------------------------------------------
    // Round 17 — motion-vector default CDF tests.
    // -----------------------------------------------------------------

    /// §9.4 verbatim values for the small / flat MV defaults: the
    /// `216*128`-style fixed-point expansions land as the bytes
    /// `SymbolDecoder::read_symbol` will see.
    #[test]
    fn mv_default_byte_exact_values() {
        // Default_Mv_Joint_Cdf = { 4096, 11264, 19328, 32768, 0 }
        assert_eq!(DEFAULT_MV_JOINT_CDF, [4096, 11264, 19328, 32768, 0]);
        // Default_Mv_Sign_Cdf = { 128*128, 32768, 0 } = { 16384, 32768, 0 }
        assert_eq!(DEFAULT_MV_SIGN_CDF, [16384, 32768, 0]);
        // Default_Mv_Hp_Cdf = { 128*128, 32768, 0 }
        assert_eq!(DEFAULT_MV_HP_CDF, [16384, 32768, 0]);
        // Default_Mv_Class0_Bit_Cdf = { 216*128, 32768, 0 } = { 27648, ... }
        assert_eq!(DEFAULT_MV_CLASS0_BIT_CDF, [27648, 32768, 0]);
        // Default_Mv_Class0_Hp_Cdf = { 160*128, 32768, 0 } = { 20480, ... }
        assert_eq!(DEFAULT_MV_CLASS0_HP_CDF, [20480, 32768, 0]);
        // Default_Mv_Bit_Cdf[ MV_OFFSET_BITS ][ 3 ] — every multiplier
        // verbatim from the spec.
        let expected: [[u16; 3]; MV_OFFSET_BITS] = [
            [136 * 128, 32768, 0],
            [140 * 128, 32768, 0],
            [148 * 128, 32768, 0],
            [160 * 128, 32768, 0],
            [176 * 128, 32768, 0],
            [192 * 128, 32768, 0],
            [224 * 128, 32768, 0],
            [234 * 128, 32768, 0],
            [234 * 128, 32768, 0],
            [240 * 128, 32768, 0],
        ];
        assert_eq!(DEFAULT_MV_BIT_CDF, expected);
        // Default_Mv_Class_Cdf — first row, by literal §9.4 listing.
        assert_eq!(
            DEFAULT_MV_CLASS_CDF[0],
            [28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767, 32768, 0]
        );
        // The leading `2` axis carries identical rows per spec.
        assert_eq!(DEFAULT_MV_CLASS_CDF[0], DEFAULT_MV_CLASS_CDF[1]);
        // Default_Mv_Class0_Fr_Cdf — both comp rows, both mv_class0_bit
        // sub-rows.
        assert_eq!(
            DEFAULT_MV_CLASS0_FR_CDF[0][0],
            [16384, 24576, 26624, 32768, 0]
        );
        assert_eq!(
            DEFAULT_MV_CLASS0_FR_CDF[0][1],
            [12288, 21248, 24128, 32768, 0]
        );
        assert_eq!(DEFAULT_MV_CLASS0_FR_CDF[1], DEFAULT_MV_CLASS0_FR_CDF[0]);
        // Default_Mv_Fr_Cdf — both comp rows identical.
        assert_eq!(DEFAULT_MV_FR_CDF[0], [8192, 17408, 21248, 32768, 0]);
        assert_eq!(DEFAULT_MV_FR_CDF[1], DEFAULT_MV_FR_CDF[0]);
    }

    /// §8.3.1 init step for the MV group: every working row matches the
    /// transcribed §9.4 default, broadcast to `MV_CONTEXTS` slots (and
    /// to `MV_COMPS` slots for the flat per-component defaults). The
    /// §8.2.6 well-formedness invariants hold on every row.
    #[test]
    fn init_from_defaults_copies_mv_tables() {
        let ctx = TileCdfContext::new_from_defaults();

        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };

        for i in 0..MV_CONTEXTS {
            assert_eq!(ctx.mv_joint[i], DEFAULT_MV_JOINT_CDF);
            check(&ctx.mv_joint[i]);

            for comp in 0..MV_COMPS {
                assert_eq!(ctx.mv_sign[i][comp], DEFAULT_MV_SIGN_CDF);
                check(&ctx.mv_sign[i][comp]);
                assert_eq!(ctx.mv_class[i][comp], DEFAULT_MV_CLASS_CDF[comp]);
                check(&ctx.mv_class[i][comp]);
                assert_eq!(ctx.mv_class0_bit[i][comp], DEFAULT_MV_CLASS0_BIT_CDF);
                check(&ctx.mv_class0_bit[i][comp]);
                assert_eq!(ctx.mv_class0_hp[i][comp], DEFAULT_MV_CLASS0_HP_CDF);
                check(&ctx.mv_class0_hp[i][comp]);
                assert_eq!(ctx.mv_hp[i][comp], DEFAULT_MV_HP_CDF);
                check(&ctx.mv_hp[i][comp]);
                assert_eq!(ctx.mv_fr[i][comp], DEFAULT_MV_FR_CDF[comp]);
                check(&ctx.mv_fr[i][comp]);

                for (bit, default_row) in DEFAULT_MV_CLASS0_FR_CDF[comp].iter().enumerate() {
                    assert_eq!(ctx.mv_class0_fr[i][comp][bit], *default_row);
                    check(&ctx.mv_class0_fr[i][comp][bit]);
                }
                for (off, default_row) in DEFAULT_MV_BIT_CDF.iter().enumerate() {
                    assert_eq!(ctx.mv_bit[i][comp][off], *default_row);
                    check(&ctx.mv_bit[i][comp][off]);
                }
            }
        }
    }

    /// §5.11.31 `MvCtx = use_intrabc ? MV_INTRABC_CONTEXT : 0`.
    #[test]
    fn mv_ctx_derivation_per_spec() {
        assert_eq!(mv_ctx(false), 0);
        assert_eq!(mv_ctx(true), MV_INTRABC_CONTEXT);
        assert_eq!(MV_INTRABC_CONTEXT, 1);
        // The §5.11.31 result must always be a valid MvCtx index.
        assert!(mv_ctx(false) < MV_CONTEXTS);
        assert!(mv_ctx(true) < MV_CONTEXTS);
    }

    /// §8.3.2 MV selectors all return the §9.4 default row (and the
    /// `MvCtx + comp + mv_class0_bit / i` indexing matches the spec's
    /// `[ MvCtx ][ comp ][ ... ]` literal).
    #[test]
    fn mv_selectors_return_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();

        for i in 0..MV_CONTEXTS {
            assert_eq!(ctx.mv_joint_cdf(i), &DEFAULT_MV_JOINT_CDF);
            for comp in 0..MV_COMPS {
                assert_eq!(ctx.mv_sign_cdf(i, comp), &DEFAULT_MV_SIGN_CDF);
                assert_eq!(ctx.mv_class_cdf(i, comp), &DEFAULT_MV_CLASS_CDF[comp]);
                assert_eq!(ctx.mv_class0_bit_cdf(i, comp), &DEFAULT_MV_CLASS0_BIT_CDF);
                assert_eq!(ctx.mv_class0_hp_cdf(i, comp), &DEFAULT_MV_CLASS0_HP_CDF);
                assert_eq!(ctx.mv_hp_cdf(i, comp), &DEFAULT_MV_HP_CDF);
                assert_eq!(ctx.mv_fr_cdf(i, comp), &DEFAULT_MV_FR_CDF[comp]);
                for (bit, default_row) in DEFAULT_MV_CLASS0_FR_CDF[comp].iter().enumerate() {
                    assert_eq!(ctx.mv_class0_fr_cdf(i, comp, bit), default_row);
                }
                for (off, default_row) in DEFAULT_MV_BIT_CDF.iter().enumerate() {
                    assert_eq!(ctx.mv_bit_cdf(i, comp, off), default_row);
                }
            }
        }
    }

    /// §8.3.1 independence for the MV group: adapting the working copy
    /// must not mutate the §9.4 source.
    #[test]
    fn mv_working_copy_is_independent_of_defaults() {
        let mut ctx = TileCdfContext::new_from_defaults();
        ctx.mv_joint_cdf(0)[0] = 17;
        ctx.mv_sign_cdf(1, 1)[0] = 33;
        ctx.mv_class0_fr_cdf(0, 1, 0)[2] = 99;
        ctx.mv_bit_cdf(1, 0, 3)[0] = 41;

        assert_ne!(ctx.mv_joint[0][0], DEFAULT_MV_JOINT_CDF[0]);
        assert_ne!(ctx.mv_sign[1][1][0], DEFAULT_MV_SIGN_CDF[0]);
        assert_ne!(
            ctx.mv_class0_fr[0][1][0][2],
            DEFAULT_MV_CLASS0_FR_CDF[1][0][2]
        );
        assert_ne!(ctx.mv_bit[1][0][3][0], DEFAULT_MV_BIT_CDF[3][0]);

        // §9.4 sources untouched.
        assert_eq!(DEFAULT_MV_JOINT_CDF, [4096, 11264, 19328, 32768, 0]);
        assert_eq!(DEFAULT_MV_SIGN_CDF, [16384, 32768, 0]);
        assert_eq!(
            DEFAULT_MV_CLASS0_FR_CDF[1][0],
            [16384, 24576, 26624, 32768, 0]
        );
        assert_eq!(DEFAULT_MV_BIT_CDF[3], [160 * 128, 32768, 0]);
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a
    /// `mv_joint` (4-value) default CDF selected by §8.3.2, and assert
    /// the §8.3 update path actually mutated the working row + counter.
    #[test]
    fn decode_mv_joint_through_default_cdf() {
        // sz = 2 ⇒ numBits = 15. bytes = 0xFF 0xFE ⇒ top 15 bits =
        // 0x7FFF; SymbolValue = 0x7FFF ^ 0x7FFF = 0. SymbolValue 0 is
        // below every `cur` boundary, so §8.2.6 returns the LAST symbol
        // (here symbol 3 = MV_JOINT_HNZVNZ).
        let bytes = [0xFFu8, 0xFEu8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 2, false).unwrap();
        assert_eq!(dec.symbol_value(), 0);

        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.mv_joint;
        let mctx = mv_ctx(false);
        assert_eq!(mctx, 0);
        let cdf = ctx.mv_joint_cdf(mctx);
        let sym = dec.read_symbol(cdf).unwrap();
        assert_eq!(sym, 3, "SymbolValue 0 selects MV_JOINT_HNZVNZ");

        // §8.3 update ran: counter advanced and the working row changed.
        assert_ne!(ctx.mv_joint, before);
        assert_eq!(ctx.mv_joint[0][4], 1, "§8.3 counter incremented to 1");
        // §9.4 source is immutable.
        assert_eq!(DEFAULT_MV_JOINT_CDF, [4096, 11264, 19328, 32768, 0]);
    }

    /// End-to-end through a binary `mv_bit` default CDF with
    /// `disable_cdf_update == true`, confirming the §8.3.2 selector
    /// drives a valid §8.2.6 decode in range and the working row stays
    /// untouched in the non-adaptive path.
    #[test]
    fn decode_mv_bit_through_default_cdf_no_update() {
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();

        // §5.11.32 inputs: MvCtx for a non-intrabc inter MV; comp = 1
        // (vertical); offset bit position i = 3.
        let mctx = mv_ctx(false);
        let cdf = ctx.mv_bit_cdf(mctx, 1, 3);
        let sym = dec.read_symbol(cdf).unwrap();
        assert!(sym < 2, "mv_bit is binary; got {sym}");

        // disable_cdf_update was true ⇒ the row is untouched.
        assert_eq!(ctx.mv_bit[mctx][1][3], DEFAULT_MV_BIT_CDF[3]);
    }

    // -----------------------------------------------------------------
    // Round 18 — inter-mode / reference-frame default CDF tests.
    // -----------------------------------------------------------------

    /// (a) Table dimensions match §9.4 verbatim for every new
    /// inter-mode / ref-frame array.
    #[test]
    fn inter_default_cdf_dimensions_match_spec() {
        assert_eq!(DEFAULT_NEW_MV_CDF.len(), NEW_MV_CONTEXTS);
        assert_eq!(DEFAULT_ZERO_MV_CDF.len(), ZERO_MV_CONTEXTS);
        assert_eq!(DEFAULT_REF_MV_CDF.len(), REF_MV_CONTEXTS);
        assert_eq!(DEFAULT_DRL_MODE_CDF.len(), DRL_MODE_CONTEXTS);
        assert_eq!(DEFAULT_IS_INTER_CDF.len(), IS_INTER_CONTEXTS);
        assert_eq!(DEFAULT_COMP_MODE_CDF.len(), COMP_INTER_CONTEXTS);
        assert_eq!(DEFAULT_SKIP_MODE_CDF.len(), SKIP_MODE_CONTEXTS);
        assert_eq!(DEFAULT_COMP_REF_TYPE_CDF.len(), COMP_REF_TYPE_CONTEXTS);

        // Binary CDFs all have 3 slots = N + 1 with N = 2.
        for row in &DEFAULT_NEW_MV_CDF {
            assert_eq!(row.len(), 3);
        }
        for row in &DEFAULT_IS_INTER_CDF {
            assert_eq!(row.len(), 3);
        }

        // Three-axis ref CDFs.
        assert_eq!(DEFAULT_COMP_REF_CDF.len(), REF_CONTEXTS);
        for row in &DEFAULT_COMP_REF_CDF {
            assert_eq!(row.len(), FWD_REFS - 1);
        }
        assert_eq!(DEFAULT_COMP_BWD_REF_CDF.len(), REF_CONTEXTS);
        for row in &DEFAULT_COMP_BWD_REF_CDF {
            assert_eq!(row.len(), BWD_REFS - 1);
        }
        assert_eq!(DEFAULT_SINGLE_REF_CDF.len(), REF_CONTEXTS);
        for row in &DEFAULT_SINGLE_REF_CDF {
            assert_eq!(row.len(), SINGLE_REFS - 1);
        }
        assert_eq!(DEFAULT_UNI_COMP_REF_CDF.len(), REF_CONTEXTS);
        for row in &DEFAULT_UNI_COMP_REF_CDF {
            assert_eq!(row.len(), UNIDIR_COMP_REFS - 1);
        }

        // Compound-mode CDF: 8 ctxs × (8 + 1) cumulatives.
        assert_eq!(DEFAULT_COMPOUND_MODE_CDF.len(), COMPOUND_MODE_CONTEXTS);
        for row in &DEFAULT_COMPOUND_MODE_CDF {
            assert_eq!(row.len(), COMPOUND_MODES + 1);
        }

        // §8.2.6 invariants on every transcribed row.
        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for r in &DEFAULT_NEW_MV_CDF {
            check(r);
        }
        for r in &DEFAULT_ZERO_MV_CDF {
            check(r);
        }
        for r in &DEFAULT_REF_MV_CDF {
            check(r);
        }
        for r in &DEFAULT_DRL_MODE_CDF {
            check(r);
        }
        for r in &DEFAULT_IS_INTER_CDF {
            check(r);
        }
        for r in &DEFAULT_COMP_MODE_CDF {
            check(r);
        }
        for r in &DEFAULT_SKIP_MODE_CDF {
            check(r);
        }
        for a in &DEFAULT_COMP_REF_CDF {
            for r in a {
                check(r);
            }
        }
        for a in &DEFAULT_COMP_BWD_REF_CDF {
            for r in a {
                check(r);
            }
        }
        for a in &DEFAULT_SINGLE_REF_CDF {
            for r in a {
                check(r);
            }
        }
        for r in &DEFAULT_COMPOUND_MODE_CDF {
            check(r);
        }
        for r in &DEFAULT_COMP_REF_TYPE_CDF {
            check(r);
        }
        for a in &DEFAULT_UNI_COMP_REF_CDF {
            for r in a {
                check(r);
            }
        }
    }

    /// (a) Byte-exact §9.4 verbatim values for hand-picked rows of every
    /// new inter-mode / ref-frame default array — every literal that
    /// appears in the spec table is read back unchanged.
    #[test]
    fn inter_default_cdf_byte_exact_values() {
        assert_eq!(DEFAULT_NEW_MV_CDF[0], [24035, 32768, 0]);
        assert_eq!(DEFAULT_NEW_MV_CDF[5], [4676, 32768, 0]);
        assert_eq!(DEFAULT_ZERO_MV_CDF[0], [2175, 32768, 0]);
        assert_eq!(DEFAULT_ZERO_MV_CDF[1], [1054, 32768, 0]);
        assert_eq!(DEFAULT_REF_MV_CDF[3], [28622, 32768, 0]);
        assert_eq!(DEFAULT_DRL_MODE_CDF[1], [24560, 32768, 0]);
        assert_eq!(DEFAULT_IS_INTER_CDF[0], [806, 32768, 0]);
        assert_eq!(DEFAULT_IS_INTER_CDF[3], [26538, 32768, 0]);
        assert_eq!(DEFAULT_COMP_MODE_CDF[2], [12031, 32768, 0]);
        assert_eq!(DEFAULT_SKIP_MODE_CDF[0], [32621, 32768, 0]);
        assert_eq!(DEFAULT_COMP_REF_CDF[0][0], [4946, 32768, 0]);
        assert_eq!(DEFAULT_COMP_REF_CDF[2][2], [27544, 32768, 0]);
        assert_eq!(DEFAULT_COMP_BWD_REF_CDF[0][0], [2235, 32768, 0]);
        assert_eq!(DEFAULT_COMP_BWD_REF_CDF[2][1], [30489, 32768, 0]);
        assert_eq!(DEFAULT_SINGLE_REF_CDF[0][0], [4897, 32768, 0]);
        assert_eq!(DEFAULT_SINGLE_REF_CDF[1][3], [24773, 32768, 0]);
        assert_eq!(DEFAULT_SINGLE_REF_CDF[2][5], [30304, 32768, 0]);
        assert_eq!(
            DEFAULT_COMPOUND_MODE_CDF[0],
            [7760, 13823, 15808, 17641, 19156, 20666, 26891, 32768, 0]
        );
        assert_eq!(
            DEFAULT_COMPOUND_MODE_CDF[7],
            [13046, 23214, 24505, 25942, 27435, 28442, 29330, 32768, 0]
        );
        assert_eq!(DEFAULT_COMP_REF_TYPE_CDF[0], [1198, 32768, 0]);
        assert_eq!(DEFAULT_COMP_REF_TYPE_CDF[4], [22475, 32768, 0]);
        assert_eq!(DEFAULT_UNI_COMP_REF_CDF[1][2], [15270, 32768, 0]);
    }

    /// (b) §8.3.1 init places every §9.4 row into the corresponding
    /// `Tile*Cdf` working slot of the freshly-constructed
    /// `TileCdfContext`.
    #[test]
    fn init_from_defaults_copies_inter_tables() {
        let ctx = TileCdfContext::new_from_defaults();

        assert_eq!(ctx.new_mv, DEFAULT_NEW_MV_CDF);
        assert_eq!(ctx.zero_mv, DEFAULT_ZERO_MV_CDF);
        assert_eq!(ctx.ref_mv, DEFAULT_REF_MV_CDF);
        assert_eq!(ctx.drl_mode, DEFAULT_DRL_MODE_CDF);
        assert_eq!(ctx.is_inter, DEFAULT_IS_INTER_CDF);
        assert_eq!(ctx.comp_mode, DEFAULT_COMP_MODE_CDF);
        assert_eq!(ctx.skip_mode, DEFAULT_SKIP_MODE_CDF);
        assert_eq!(ctx.comp_ref, DEFAULT_COMP_REF_CDF);
        assert_eq!(ctx.comp_bwd_ref, DEFAULT_COMP_BWD_REF_CDF);
        assert_eq!(ctx.single_ref, DEFAULT_SINGLE_REF_CDF);
        assert_eq!(ctx.compound_mode, DEFAULT_COMPOUND_MODE_CDF);
        assert_eq!(ctx.comp_ref_type, DEFAULT_COMP_REF_TYPE_CDF);
        assert_eq!(ctx.uni_comp_ref, DEFAULT_UNI_COMP_REF_CDF);
    }

    /// (b) §8.3.1 init + §8.3.2 selection: at a hand-picked
    /// `(frame_type, ctx)` tuple — here the intra-only "tile init" path
    /// for the `comp_ref` syntax — the selected row is exactly the
    /// `Default_Comp_Ref_Cdf[ ctx ][ p ]` value from §9.4.
    ///
    /// AV1 §8.3.1 always seeds `Tile*Cdf` from the same `Default_*_Cdf`
    /// regardless of `frame_type` (the only `frame_type`-keyed variant
    /// is `intra_frame_y_mode`, which uses a separate `Y_Mode_Cdf` table
    /// for non-intra frames — out of this round's scope). So we exercise
    /// every § member here with both extremes of its `ctx` index.
    #[test]
    fn inter_selectors_return_default_rows_at_hand_picked_ctx() {
        let mut ctx = TileCdfContext::new_from_defaults();

        // (frame_type=key-or-intra-only, new_mv, ctx=0).
        assert_eq!(ctx.new_mv_cdf(0), &DEFAULT_NEW_MV_CDF[0]);
        assert_eq!(
            ctx.new_mv_cdf(NEW_MV_CONTEXTS - 1),
            &DEFAULT_NEW_MV_CDF[NEW_MV_CONTEXTS - 1]
        );
        assert_eq!(ctx.zero_mv_cdf(0), &DEFAULT_ZERO_MV_CDF[0]);
        assert_eq!(ctx.zero_mv_cdf(1), &DEFAULT_ZERO_MV_CDF[1]);
        assert_eq!(ctx.ref_mv_cdf(0), &DEFAULT_REF_MV_CDF[0]);
        assert_eq!(ctx.ref_mv_cdf(5), &DEFAULT_REF_MV_CDF[5]);
        assert_eq!(ctx.drl_mode_cdf(0), &DEFAULT_DRL_MODE_CDF[0]);
        assert_eq!(ctx.drl_mode_cdf(2), &DEFAULT_DRL_MODE_CDF[2]);
        assert_eq!(ctx.is_inter_cdf(0), &DEFAULT_IS_INTER_CDF[0]);
        assert_eq!(ctx.is_inter_cdf(3), &DEFAULT_IS_INTER_CDF[3]);
        assert_eq!(ctx.comp_mode_cdf(0), &DEFAULT_COMP_MODE_CDF[0]);
        assert_eq!(ctx.comp_mode_cdf(4), &DEFAULT_COMP_MODE_CDF[4]);
        assert_eq!(ctx.skip_mode_cdf(0), &DEFAULT_SKIP_MODE_CDF[0]);
        assert_eq!(ctx.skip_mode_cdf(2), &DEFAULT_SKIP_MODE_CDF[2]);
        assert_eq!(ctx.comp_ref_cdf(0, 0), &DEFAULT_COMP_REF_CDF[0][0]);
        assert_eq!(ctx.comp_ref_cdf(2, 2), &DEFAULT_COMP_REF_CDF[2][2]);
        assert_eq!(ctx.comp_bwd_ref_cdf(0, 0), &DEFAULT_COMP_BWD_REF_CDF[0][0]);
        assert_eq!(ctx.comp_bwd_ref_cdf(2, 1), &DEFAULT_COMP_BWD_REF_CDF[2][1]);
        assert_eq!(ctx.single_ref_cdf(0, 0), &DEFAULT_SINGLE_REF_CDF[0][0]);
        assert_eq!(ctx.single_ref_cdf(2, 5), &DEFAULT_SINGLE_REF_CDF[2][5]);
        assert_eq!(ctx.compound_mode_cdf(0), &DEFAULT_COMPOUND_MODE_CDF[0]);
        assert_eq!(ctx.compound_mode_cdf(7), &DEFAULT_COMPOUND_MODE_CDF[7]);
        assert_eq!(ctx.comp_ref_type_cdf(0), &DEFAULT_COMP_REF_TYPE_CDF[0]);
        assert_eq!(ctx.comp_ref_type_cdf(4), &DEFAULT_COMP_REF_TYPE_CDF[4]);
        assert_eq!(ctx.uni_comp_ref_cdf(0, 0), &DEFAULT_UNI_COMP_REF_CDF[0][0]);
        assert_eq!(ctx.uni_comp_ref_cdf(2, 2), &DEFAULT_UNI_COMP_REF_CDF[2][2]);
    }

    /// §8.3.1 independence for the inter group: adapting the working
    /// copy must not mutate the §9.4 source (the next tile re-inits
    /// from it).
    #[test]
    fn inter_working_copy_is_independent_of_defaults() {
        let mut ctx = TileCdfContext::new_from_defaults();
        ctx.new_mv_cdf(0)[0] = 7;
        ctx.comp_ref_cdf(1, 2)[0] = 13;
        ctx.compound_mode_cdf(3)[5] = 999;

        assert_ne!(ctx.new_mv[0][0], DEFAULT_NEW_MV_CDF[0][0]);
        assert_ne!(ctx.comp_ref[1][2][0], DEFAULT_COMP_REF_CDF[1][2][0]);
        assert_ne!(ctx.compound_mode[3][5], DEFAULT_COMPOUND_MODE_CDF[3][5]);

        // §9.4 sources untouched.
        assert_eq!(DEFAULT_NEW_MV_CDF[0], [24035, 32768, 0]);
        assert_eq!(DEFAULT_COMP_REF_CDF[1][2], [15160, 32768, 0]);
        assert_eq!(DEFAULT_COMPOUND_MODE_CDF[3][5], 25736);
    }

    /// §8.3.2 `is_inter` context branches — every neighbour-availability
    /// + intra-flag combination per the §8.3.2 paragraph.
    #[test]
    fn is_inter_context_branches() {
        // AvailU && AvailL.
        assert_eq!(is_inter_ctx(Some(false), Some(false)), 0); // both inter
        assert_eq!(is_inter_ctx(Some(true), Some(false)), 1); // exactly one intra
        assert_eq!(is_inter_ctx(Some(false), Some(true)), 1);
        assert_eq!(is_inter_ctx(Some(true), Some(true)), 3); // both intra

        // AvailU XOR AvailL.
        assert_eq!(is_inter_ctx(Some(true), None), 2);
        assert_eq!(is_inter_ctx(Some(false), None), 0);
        assert_eq!(is_inter_ctx(None, Some(true)), 2);
        assert_eq!(is_inter_ctx(None, Some(false)), 0);

        // Neither available.
        assert_eq!(is_inter_ctx(None, None), 0);

        // Every result is a valid IS_INTER context index.
        for above in [None, Some(false), Some(true)] {
            for left in [None, Some(false), Some(true)] {
                assert!(is_inter_ctx(above, left) < IS_INTER_CONTEXTS);
            }
        }
    }

    /// §8.3.2 `skip_mode` context: sum of neighbour `SkipModes[]`.
    #[test]
    fn skip_mode_context_sum() {
        assert_eq!(skip_mode_ctx(0, 0), 0);
        assert_eq!(skip_mode_ctx(1, 0), 1);
        assert_eq!(skip_mode_ctx(0, 1), 1);
        assert_eq!(skip_mode_ctx(1, 1), 2);
        for a in 0..=1 {
            for l in 0..=1 {
                assert!(skip_mode_ctx(a, l) < SKIP_MODE_CONTEXTS);
            }
        }
    }

    /// §8.3.2 `ref_count_ctx` three-branch ladder.
    #[test]
    fn ref_count_context_branches() {
        assert_eq!(ref_count_ctx(0, 1), 0);
        assert_eq!(ref_count_ctx(3, 4), 0);
        assert_eq!(ref_count_ctx(0, 0), 1);
        assert_eq!(ref_count_ctx(7, 7), 1);
        assert_eq!(ref_count_ctx(2, 1), 2);
        assert_eq!(ref_count_ctx(99, 0), 2);
        // Every result is a valid REF_CONTEXTS index.
        for c0 in 0..3 {
            for c1 in 0..3 {
                assert!(ref_count_ctx(c0, c1) < REF_CONTEXTS);
            }
        }
    }

    /// §8.3.2 `compound_mode` context: the `Compound_Mode_Ctx_Map`
    /// lookup with the `RefMvContext >> 1` / `Min(NewMvContext,
    /// COMP_NEWMV_CTXS - 1)` indexing — three hand-picked entries from
    /// each of the three map rows.
    #[test]
    fn compound_mode_context_map_lookup() {
        // Row 0 (RefMvContext >> 1 == 0): map = { 0, 1, 1, 1, 1 }.
        assert_eq!(compound_mode_ctx(0, 0), 0);
        assert_eq!(compound_mode_ctx(1, 1), 1); // 1 >> 1 == 0
        assert_eq!(compound_mode_ctx(0, 4), 1);

        // Row 1 (RefMvContext >> 1 == 1): map = { 1, 2, 3, 4, 4 }.
        assert_eq!(compound_mode_ctx(2, 0), 1);
        assert_eq!(compound_mode_ctx(2, 2), 3);
        assert_eq!(compound_mode_ctx(3, 3), 4); // 3 >> 1 == 1

        // Row 2 (RefMvContext >> 1 == 2): map = { 4, 4, 5, 6, 7 }.
        assert_eq!(compound_mode_ctx(4, 0), 4);
        assert_eq!(compound_mode_ctx(4, 4), 7);
        assert_eq!(compound_mode_ctx(5, 2), 5); // 5 >> 1 == 2

        // The `Min(.., COMP_NEWMV_CTXS - 1)` clamp: anything ≥ 4 is
        // treated as 4.
        assert_eq!(compound_mode_ctx(4, 99), 7);
        // The map has only 3 rows; anything beyond row 2 saturates at
        // row 2 (the §8.3.2 spec doesn't define a fourth row — every
        // valid `RefMvContext` value yields `RefMvContext >> 1 < 3`,
        // since `RefMvContext < REF_MV_CONTEXTS == 6`).
        assert_eq!(compound_mode_ctx(REF_MV_CONTEXTS - 1, 0), 4);

        // Every result is a valid COMPOUND_MODE_CONTEXTS index.
        for r in 0..REF_MV_CONTEXTS {
            for n in 0..NEW_MV_CONTEXTS {
                assert!(compound_mode_ctx(r, n) < COMPOUND_MODE_CONTEXTS);
            }
        }
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a
    /// `compound_mode` (8-value) default CDF selected by §8.3.2
    /// `compound_mode_ctx`, confirming the §8.3.2-selected row matches
    /// the §9.4 default and a valid §8.2.6 decode lands in range.
    #[test]
    fn decode_compound_mode_through_default_cdf() {
        // Pick a hand-picked (RefMvContext, NewMvContext) pair so that
        // `compound_mode_ctx` lands on row 7 (= map[2][4]).
        let cm_ctx = compound_mode_ctx(4, 4);
        assert_eq!(cm_ctx, 7);

        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, true).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();

        // §8.3.2 selection: `TileCompoundModeCdf[ cm_ctx ]` equals
        // `Default_Compound_Mode_Cdf[ 7 ]` since we just init'd.
        let row = ctx.compound_mode_cdf(cm_ctx);
        assert_eq!(row, &DEFAULT_COMPOUND_MODE_CDF[7]);

        let sym = dec.read_symbol(row).unwrap();
        assert!(sym < COMPOUND_MODES as u32, "compound_mode is in 0..8");
        // disable_cdf_update was true ⇒ row untouched.
        assert_eq!(ctx.compound_mode[7], DEFAULT_COMPOUND_MODE_CDF[7]);
    }

    // -----------------------------------------------------------------
    // Round 19 — palette / filter-intra / CFL default CDF tests.
    // -----------------------------------------------------------------

    /// §9.4 dimensions for every palette / filter-intra / CFL table,
    /// matching the spec's declared array shapes, and the §8.2.6
    /// well-formedness invariant (`cdf[N-1] == 32768`, `cdf[N] == 0`) on
    /// every transcribed row.
    #[test]
    fn palette_group_table_dimensions_and_invariants() {
        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };

        // Filter-intra.
        assert_eq!(DEFAULT_FILTER_INTRA_MODE_CDF.len(), INTRA_FILTER_MODES + 1);
        check(&DEFAULT_FILTER_INTRA_MODE_CDF);
        assert_eq!(DEFAULT_FILTER_INTRA_CDF.len(), BLOCK_SIZES);
        for r in &DEFAULT_FILTER_INTRA_CDF {
            assert_eq!(r.len(), 3);
            check(r);
        }

        // Palette mode / size.
        assert_eq!(
            DEFAULT_PALETTE_Y_MODE_CDF.len(),
            PALETTE_BLOCK_SIZE_CONTEXTS
        );
        for plane in &DEFAULT_PALETTE_Y_MODE_CDF {
            assert_eq!(plane.len(), PALETTE_Y_MODE_CONTEXTS);
            for r in plane {
                assert_eq!(r.len(), 3);
                check(r);
            }
        }
        assert_eq!(DEFAULT_PALETTE_UV_MODE_CDF.len(), PALETTE_UV_MODE_CONTEXTS);
        for r in &DEFAULT_PALETTE_UV_MODE_CDF {
            check(r);
        }
        assert_eq!(
            DEFAULT_PALETTE_Y_SIZE_CDF.len(),
            PALETTE_BLOCK_SIZE_CONTEXTS
        );
        assert_eq!(
            DEFAULT_PALETTE_UV_SIZE_CDF.len(),
            PALETTE_BLOCK_SIZE_CONTEXTS
        );
        for r in &DEFAULT_PALETTE_Y_SIZE_CDF {
            assert_eq!(r.len(), PALETTE_SIZES + 1);
            check(r);
        }
        for r in &DEFAULT_PALETTE_UV_SIZE_CDF {
            assert_eq!(r.len(), PALETTE_SIZES + 1);
            check(r);
        }

        // Palette colour-index: a size-K table codes K symbols (row len
        // K + 1), with PALETTE_COLOR_CONTEXTS rows each.
        macro_rules! check_color {
            ($tab:expr, $size:expr) => {{
                assert_eq!($tab.len(), PALETTE_COLOR_CONTEXTS);
                for r in &$tab {
                    assert_eq!(r.len(), $size + 1, "size-{} table row len", $size);
                    check(r);
                }
            }};
        }
        check_color!(DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF, 2);
        check_color!(DEFAULT_PALETTE_SIZE_3_Y_COLOR_CDF, 3);
        check_color!(DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF, 4);
        check_color!(DEFAULT_PALETTE_SIZE_5_Y_COLOR_CDF, 5);
        check_color!(DEFAULT_PALETTE_SIZE_6_Y_COLOR_CDF, 6);
        check_color!(DEFAULT_PALETTE_SIZE_7_Y_COLOR_CDF, 7);
        check_color!(DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF, 8);
        check_color!(DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF, 2);
        check_color!(DEFAULT_PALETTE_SIZE_3_UV_COLOR_CDF, 3);
        check_color!(DEFAULT_PALETTE_SIZE_4_UV_COLOR_CDF, 4);
        check_color!(DEFAULT_PALETTE_SIZE_5_UV_COLOR_CDF, 5);
        check_color!(DEFAULT_PALETTE_SIZE_6_UV_COLOR_CDF, 6);
        check_color!(DEFAULT_PALETTE_SIZE_7_UV_COLOR_CDF, 7);
        check_color!(DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF, 8);

        // CFL.
        assert_eq!(DEFAULT_CFL_SIGN_CDF.len(), CFL_JOINT_SIGNS + 1);
        check(&DEFAULT_CFL_SIGN_CDF);
        assert_eq!(DEFAULT_CFL_ALPHA_CDF.len(), CFL_ALPHA_CONTEXTS);
        for r in &DEFAULT_CFL_ALPHA_CDF {
            assert_eq!(r.len(), CFL_ALPHABET_SIZE + 1);
            check(r);
        }
    }

    /// §9.4 hand-picked byte-exact values, spot-checking each table type
    /// at a row the spec lists explicitly.
    #[test]
    fn palette_group_byte_exact_values() {
        assert_eq!(
            DEFAULT_FILTER_INTRA_MODE_CDF,
            [8949, 12776, 17211, 29558, 32768, 0]
        );
        // Filter-intra MiSize 0 and the "never used" filler at index 10.
        assert_eq!(DEFAULT_FILTER_INTRA_CDF[0], [4621, 32768, 0]);
        assert_eq!(DEFAULT_FILTER_INTRA_CDF[10], [16384, 32768, 0]);
        // Palette Y-mode last block-size context, third ctx (the
        // distinctive `{ 129, .. }` row).
        assert_eq!(DEFAULT_PALETTE_Y_MODE_CDF[6][2], [129, 32768, 0]);
        assert_eq!(DEFAULT_PALETTE_UV_MODE_CDF[1], [21488, 32768, 0]);
        assert_eq!(
            DEFAULT_PALETTE_Y_SIZE_CDF[0],
            [7952, 13000, 18149, 21478, 25527, 29241, 32768, 0]
        );
        assert_eq!(
            DEFAULT_PALETTE_UV_SIZE_CDF[6],
            [1269, 5435, 10433, 18963, 21700, 25865, 32768, 0]
        );
        // Largest Y colour table, last ctx.
        assert_eq!(
            DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF[4],
            [31028, 31270, 31504, 31705, 31927, 32153, 32392, 32768, 0]
        );
        // Smallest UV colour table.
        assert_eq!(DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF[2], [8713, 32768, 0]);
        assert_eq!(
            DEFAULT_CFL_SIGN_CDF,
            [1418, 2123, 13340, 18405, 26972, 28343, 32294, 32768, 0]
        );
        assert_eq!(
            DEFAULT_CFL_ALPHA_CDF[0],
            [
                7637, 20719, 31401, 32481, 32657, 32688, 32692, 32696, 32700, 32704, 32708, 32712,
                32716, 32720, 32724, 32768, 0
            ]
        );
    }

    /// §8.3.1 init step for the palette / filter-intra / CFL group: every
    /// working array matches its §9.4 default verbatim.
    #[test]
    fn init_from_defaults_copies_palette_group() {
        let ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.filter_intra_mode, DEFAULT_FILTER_INTRA_MODE_CDF);
        assert_eq!(ctx.filter_intra, DEFAULT_FILTER_INTRA_CDF);
        assert_eq!(ctx.palette_y_mode, DEFAULT_PALETTE_Y_MODE_CDF);
        assert_eq!(ctx.palette_uv_mode, DEFAULT_PALETTE_UV_MODE_CDF);
        assert_eq!(ctx.palette_y_size, DEFAULT_PALETTE_Y_SIZE_CDF);
        assert_eq!(ctx.palette_uv_size, DEFAULT_PALETTE_UV_SIZE_CDF);
        assert_eq!(
            ctx.palette_size_2_y_color,
            DEFAULT_PALETTE_SIZE_2_Y_COLOR_CDF
        );
        assert_eq!(
            ctx.palette_size_8_y_color,
            DEFAULT_PALETTE_SIZE_8_Y_COLOR_CDF
        );
        assert_eq!(
            ctx.palette_size_2_uv_color,
            DEFAULT_PALETTE_SIZE_2_UV_COLOR_CDF
        );
        assert_eq!(
            ctx.palette_size_8_uv_color,
            DEFAULT_PALETTE_SIZE_8_UV_COLOR_CDF
        );
        assert_eq!(ctx.cfl_sign, DEFAULT_CFL_SIGN_CDF);
        assert_eq!(ctx.cfl_alpha, DEFAULT_CFL_ALPHA_CDF);
    }

    /// §8.3.2 `palette_color_idx_*` array selection by `PaletteSize`:
    /// a size-K palette selects the size-K colour table (row length
    /// `K + 1`), and out-of-range sizes return `None`.
    #[test]
    fn palette_color_cdf_selected_by_size() {
        let mut ctx = TileCdfContext::new_from_defaults();
        for size in 2..=PALETTE_COLORS {
            assert_eq!(
                ctx.palette_y_color_cdf(size, 0).unwrap().len(),
                size + 1,
                "Y size-{size} colour row length"
            );
            assert_eq!(
                ctx.palette_uv_color_cdf(size, 0).unwrap().len(),
                size + 1,
                "UV size-{size} colour row length"
            );
        }
        assert!(ctx.palette_y_color_cdf(1, 0).is_none());
        assert!(ctx.palette_y_color_cdf(9, 0).is_none());
        assert!(ctx.palette_uv_color_cdf(1, 0).is_none());
        assert!(ctx.palette_uv_color_cdf(9, 0).is_none());
        // The selected row matches the §9.4 default for that ctx.
        assert_eq!(
            ctx.palette_y_color_cdf(4, 3).unwrap(),
            &DEFAULT_PALETTE_SIZE_4_Y_COLOR_CDF[3]
        );
    }

    /// §8.3.2 `has_palette_y` / `has_palette_uv` context formulas.
    #[test]
    fn palette_mode_contexts() {
        assert_eq!(palette_y_mode_ctx(false, false), 0);
        assert_eq!(palette_y_mode_ctx(true, false), 1);
        assert_eq!(palette_y_mode_ctx(false, true), 1);
        assert_eq!(palette_y_mode_ctx(true, true), 2);
        assert_eq!(palette_uv_mode_ctx(0), 0);
        assert_eq!(palette_uv_mode_ctx(4), 1);
    }

    /// §8.3.2 `Palette_Color_Context[ ColorContextHash ]` mapping: the
    /// `-1` sentinels yield `None`, the rest map per the spec table.
    #[test]
    fn palette_color_context_map() {
        // { -1, -1, 0, -1, -1, 4, 3, 2, 1 }
        assert_eq!(palette_color_ctx(0), None);
        assert_eq!(palette_color_ctx(1), None);
        assert_eq!(palette_color_ctx(2), Some(0));
        assert_eq!(palette_color_ctx(3), None);
        assert_eq!(palette_color_ctx(4), None);
        assert_eq!(palette_color_ctx(5), Some(4));
        assert_eq!(palette_color_ctx(6), Some(3));
        assert_eq!(palette_color_ctx(7), Some(2));
        assert_eq!(palette_color_ctx(8), Some(1));
        // Out of range.
        assert_eq!(palette_color_ctx(9), None);
        // Every valid result is a PALETTE_COLOR_CONTEXTS index.
        for h in 0..=PALETTE_MAX_COLOR_CONTEXT_HASH {
            if let Some(c) = palette_color_ctx(h) {
                assert!(c < PALETTE_COLOR_CONTEXTS);
            }
        }
        assert_eq!(PALETTE_COLOR_HASH_MULTIPLIERS, [1, 2, 2]);
    }

    /// §8.3.2 `cfl_alpha_u` / `cfl_alpha_v` context formulas. The spec
    /// notes `cfl_alpha_u` ctx == `cfl_alpha_signs - 2`; check that
    /// identity across every joint-sign value that decodes a U component
    /// (`signU != CFL_SIGN_ZERO`), with the §5.11.45 joint decomposition
    /// `signU = (cfl_alpha_signs + 1) / 3`, `signV = (cfl_alpha_signs + 1) % 3`.
    #[test]
    fn cfl_alpha_contexts() {
        // Spot-check both formulas against the §8.3.2 tables.
        // cfl_alpha_u: (signU - 1) * 3 + signV.
        assert_eq!(cfl_alpha_u_ctx(1, 0), 0);
        assert_eq!(cfl_alpha_u_ctx(2, 2), 5);
        // cfl_alpha_v: (signV - 1) * 3 + signU.
        assert_eq!(cfl_alpha_v_ctx(0, 1), 0);
        assert_eq!(cfl_alpha_v_ctx(2, 2), 5);

        // The §8.3.2 note: for cfl_alpha_u, ctx == cfl_alpha_signs - 2.
        for joint in 0..CFL_JOINT_SIGNS {
            let sign_u = (joint + 1) / 3;
            let sign_v = (joint + 1) % 3;
            if sign_u != 0 {
                let ctx = cfl_alpha_u_ctx(sign_u, sign_v);
                assert_eq!(ctx, joint - 2, "cfl_alpha_u ctx == cfl_alpha_signs - 2");
                assert!(ctx < CFL_ALPHA_CONTEXTS);
            }
            if sign_v != 0 {
                assert!(cfl_alpha_v_ctx(sign_u, sign_v) < CFL_ALPHA_CONTEXTS);
            }
        }
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a
    /// `cfl_alpha_u` (16-value) default CDF selected by §8.3.2
    /// `cfl_alpha_u_ctx`, confirming the selected row matches the §9.4
    /// default and a valid §8.2.6 decode lands in range and adapts.
    #[test]
    fn decode_cfl_alpha_through_default_cdf() {
        // signU = 1, signV = 0 ⇒ ctx = 0.
        let ctx_idx = cfl_alpha_u_ctx(1, 0);
        assert_eq!(ctx_idx, 0);

        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.cfl_alpha;

        let row = ctx.cfl_alpha_cdf(ctx_idx);
        assert_eq!(row, &DEFAULT_CFL_ALPHA_CDF[0]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(sym < CFL_ALPHABET_SIZE as u32, "cfl_alpha is in 0..16");

        // §8.3 update ran (disable_cdf_update == false): the row changed
        // but the §9.4 source is immutable.
        assert_ne!(ctx.cfl_alpha, before, "read_symbol must adapt the CDF");
        assert_eq!(DEFAULT_CFL_ALPHA_CDF[0][0], 7637);
    }

    // -----------------------------------------------------------------
    // Round 20 — transform-size group tests.
    // -----------------------------------------------------------------

    /// §9.4 default tables: every row terminates with `1 << 15` and a
    /// zero adaptation counter (the §8.2.6 contract). Locks all five
    /// transcribed transform-size tables row-by-row.
    #[test]
    fn tx_size_default_tables_well_formed() {
        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for r in &DEFAULT_TX_8X8_CDF {
            check(r);
        }
        for r in &DEFAULT_TX_16X16_CDF {
            check(r);
        }
        for r in &DEFAULT_TX_32X32_CDF {
            check(r);
        }
        for r in &DEFAULT_TX_64X64_CDF {
            check(r);
        }
        for r in &DEFAULT_TXFM_SPLIT_CDF {
            check(r);
        }

        // §3 dimensions held by every default table.
        assert_eq!(DEFAULT_TX_8X8_CDF.len(), TX_SIZE_CONTEXTS);
        assert_eq!(DEFAULT_TX_16X16_CDF.len(), TX_SIZE_CONTEXTS);
        assert_eq!(DEFAULT_TX_32X32_CDF.len(), TX_SIZE_CONTEXTS);
        assert_eq!(DEFAULT_TX_64X64_CDF.len(), TX_SIZE_CONTEXTS);
        assert_eq!(DEFAULT_TXFM_SPLIT_CDF.len(), TXFM_PARTITION_CONTEXTS);
        assert_eq!(DEFAULT_TX_8X8_CDF[0].len(), MAX_TX_DEPTH + 1);
        assert_eq!(DEFAULT_TX_16X16_CDF[0].len(), MAX_TX_DEPTH + 2);
        assert_eq!(DEFAULT_TX_32X32_CDF[0].len(), MAX_TX_DEPTH + 2);
        assert_eq!(DEFAULT_TX_64X64_CDF[0].len(), MAX_TX_DEPTH + 2);
        assert_eq!(DEFAULT_TXFM_SPLIT_CDF[0].len(), 3);
    }

    /// Spot-check the first / last byte of each §9.4 transcribed
    /// transform-size table. If a digit was mis-keyed during the
    /// transcription, the equality breaks.
    #[test]
    fn tx_size_table_byte_anchors() {
        // First / last cumulative-frequency in each table.
        assert_eq!(DEFAULT_TX_8X8_CDF[0][0], 19968);
        assert_eq!(DEFAULT_TX_8X8_CDF[2][0], 24320);
        assert_eq!(DEFAULT_TX_16X16_CDF[0][0], 12272);
        assert_eq!(DEFAULT_TX_16X16_CDF[0][1], 30172);
        assert_eq!(DEFAULT_TX_16X16_CDF[2][1], 30848);
        assert_eq!(DEFAULT_TX_32X32_CDF[0][0], 12986);
        assert_eq!(DEFAULT_TX_32X32_CDF[2][1], 25602);
        assert_eq!(DEFAULT_TX_64X64_CDF[0][0], 5782);
        assert_eq!(DEFAULT_TX_64X64_CDF[2][1], 22759);

        // First / last row of Default_Txfm_Split_Cdf.
        assert_eq!(DEFAULT_TXFM_SPLIT_CDF[0][0], 28581);
        assert_eq!(
            DEFAULT_TXFM_SPLIT_CDF[TXFM_PARTITION_CONTEXTS - 1][0],
            16088
        );
    }

    /// §8.3.1: a fresh context copies every transform-size default
    /// in (independent of the source — the §9.4 array is not aliased).
    #[test]
    fn tx_size_init_from_defaults_copies_tables() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.tx_8x8, DEFAULT_TX_8X8_CDF);
        assert_eq!(ctx.tx_16x16, DEFAULT_TX_16X16_CDF);
        assert_eq!(ctx.tx_32x32, DEFAULT_TX_32X32_CDF);
        assert_eq!(ctx.tx_64x64, DEFAULT_TX_64X64_CDF);
        assert_eq!(ctx.txfm_split, DEFAULT_TXFM_SPLIT_CDF);

        // Working-copy independence: mutating the context must not
        // touch the §9.4 source.
        ctx.txfm_split_cdf(0)[0] = 12345;
        assert_ne!(ctx.txfm_split[0][0], DEFAULT_TXFM_SPLIT_CDF[0][0]);
        assert_eq!(DEFAULT_TXFM_SPLIT_CDF[0][0], 28581);
    }

    /// §8.3.2 `tx_depth` selection: each `max_tx_depth` value picks the
    /// expected `Default_Tx_*_Cdf` row and the row length matches the
    /// spec's per-table symbol count (`MAX_TX_DEPTH + 1` for the
    /// `maxTxDepth == 1` group, `MAX_TX_DEPTH + 2` for every other).
    #[test]
    fn tx_depth_cdf_selected_by_max_tx_depth() {
        let mut ctx = TileCdfContext::new_from_defaults();

        // maxTxDepth == 1 ⇒ TileTx8x8Cdf, row width MAX_TX_DEPTH + 1.
        let row1 = ctx.tx_depth_cdf(1, 1).unwrap();
        assert_eq!(row1.len(), MAX_TX_DEPTH + 1);
        assert_eq!(row1, &DEFAULT_TX_8X8_CDF[1]);

        // maxTxDepth == 2 ⇒ TileTx16x16Cdf.
        let row2 = ctx.tx_depth_cdf(2, 2).unwrap();
        assert_eq!(row2.len(), MAX_TX_DEPTH + 2);
        assert_eq!(row2, &DEFAULT_TX_16X16_CDF[2]);

        // maxTxDepth == 3 ⇒ TileTx32x32Cdf.
        let row3 = ctx.tx_depth_cdf(3, 0).unwrap();
        assert_eq!(row3, &DEFAULT_TX_32X32_CDF[0]);

        // maxTxDepth == 4 ⇒ TileTx64x64Cdf.
        let row4 = ctx.tx_depth_cdf(4, 2).unwrap();
        assert_eq!(row4, &DEFAULT_TX_64X64_CDF[2]);

        // maxTxDepth == 0 ⇒ tx_depth is not read; selection returns None.
        assert!(ctx.tx_depth_cdf(0, 0).is_none());
        // out-of-range max_tx_depth ⇒ None.
        assert!(ctx.tx_depth_cdf(5, 0).is_none());
    }

    /// §8.3.2 `tx_depth` context formula:
    /// `ctx = (above_w >= max_tx_width) + (left_h >= max_tx_height)`.
    /// Cover all four neighbour-vs-max combinations and check the
    /// strict-less-than semantics ("strictly less" doesn't count).
    #[test]
    fn tx_depth_context_formula() {
        // both neighbours strictly smaller ⇒ 0.
        assert_eq!(tx_depth_ctx(0, 0, 16, 16), 0);
        // above hits the threshold ⇒ 1 (>= is the spec's relation).
        assert_eq!(tx_depth_ctx(16, 0, 16, 16), 1);
        // left hits ⇒ 1.
        assert_eq!(tx_depth_ctx(0, 16, 16, 16), 1);
        // both hit ⇒ 2.
        assert_eq!(tx_depth_ctx(16, 16, 16, 16), 2);
        // both strictly greater ⇒ also 2 (the >= still holds).
        assert_eq!(tx_depth_ctx(32, 64, 16, 16), 2);
        // Result is bounded by TX_SIZE_CONTEXTS.
        for aw in [0u32, 8, 16, 32, 64] {
            for lh in [0u32, 8, 16, 32, 64] {
                assert!(tx_depth_ctx(aw, lh, 16, 16) < TX_SIZE_CONTEXTS);
            }
        }
    }

    /// §8.3.2 `txfm_split` context:
    /// `ctx = (txSzSqrUp != maxTxSz) * 3 + (TX_SIZES - 1 - maxTxSz) * 6 + above + left`.
    /// Walk the spec formula term-by-term for several reachable
    /// `(above, left, txSzSqrUp, maxTxSz)` tuples, then check the
    /// out-of-range and overflow guards.
    #[test]
    fn txfm_split_context_formula() {
        // maxTxSz == TX_4X4 (= 0), tx_sz_sqr_up == 0, no neighbours ⇒
        // ctx = 0 + (5 - 1 - 0) * 6 + 0 + 0 = 24.
        assert_eq!(txfm_split_ctx(false, false, 0, 0), None);
        // ^ 24 >= TXFM_PARTITION_CONTEXTS (21), so the helper returns
        // None — only reachable tuples should land in-range.

        // A reachable in-range tuple: maxTxSz == TX_64X64 (= 4),
        // txSzSqrUp == 0 (split), no neighbours ⇒
        // ctx = 1 * 3 + (5 - 1 - 4) * 6 + 0 + 0 = 3.
        assert_eq!(txfm_split_ctx(false, false, 0, 4), Some(3));
        // Same shape but both neighbours present ⇒ ctx = 3 + 2 = 5.
        assert_eq!(txfm_split_ctx(true, true, 0, 4), Some(5));
        // maxTxSz == TX_64X64, tx_sz_sqr_up == 4 (no split) ⇒ ctx = 0.
        assert_eq!(txfm_split_ctx(false, false, 4, 4), Some(0));
        // maxTxSz == TX_64X64, tx_sz_sqr_up == 4 (no split), both nbrs ⇒
        // ctx = 0 + 0 + 1 + 1 = 2.
        assert_eq!(txfm_split_ctx(true, true, 4, 4), Some(2));

        // maxTxSz == TX_32X32 (= 3), split ⇒
        // ctx = 3 + (5 - 1 - 3) * 6 + 0 + 0 = 9.
        assert_eq!(txfm_split_ctx(false, false, 0, 3), Some(9));
        // maxTxSz == TX_32X32, no split, both nbrs ⇒
        // ctx = 0 + 6 + 1 + 1 = 8.
        assert_eq!(txfm_split_ctx(true, true, 3, 3), Some(8));

        // Out-of-range max_tx_sz.
        assert_eq!(txfm_split_ctx(false, false, 0, TX_SIZES as u32), None);

        // Every reachable in-range result is < TXFM_PARTITION_CONTEXTS.
        for &max_tx_sz in &[0u32, 1, 2, 3, 4] {
            for &split in &[0u32, 1] {
                let sqr_up = if split == 1 { 0 } else { max_tx_sz };
                for above in [false, true] {
                    for left in [false, true] {
                        if let Some(ctx) = txfm_split_ctx(above, left, sqr_up, max_tx_sz) {
                            assert!(
                                ctx < TXFM_PARTITION_CONTEXTS,
                                "ctx {ctx} out of range for max={max_tx_sz} split={split}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through a
    /// `tx_depth` 3-value default CDF selected by §8.3.2 `tx_depth_ctx`,
    /// confirming the selected row matches the §9.4 default and a valid
    /// §8.2.6 decode lands in range and adapts.
    #[test]
    fn decode_tx_depth_through_default_cdf() {
        // aboveW == 16, leftH == 16, both equal maxTxWidth/Height = 16
        // ⇒ ctx = 2 (both >= threshold).
        let ctx_idx = tx_depth_ctx(16, 16, 16, 16);
        assert_eq!(ctx_idx, 2);

        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.tx_16x16;

        // maxTxDepth == 2 ⇒ TileTx16x16Cdf[ ctx_idx ].
        let row = ctx.tx_depth_cdf(2, ctx_idx).unwrap();
        assert_eq!(row, &DEFAULT_TX_16X16_CDF[2]);
        let sym = dec.read_symbol(row).unwrap();
        // tx_depth in 0..MAX_TX_DEPTH + 1 (the row carries 3 symbols).
        assert!(sym <= MAX_TX_DEPTH as u32, "tx_depth in 0..=MAX_TX_DEPTH");

        // §8.3 update ran (the working copy mutated) but the §9.4
        // source is immutable.
        assert_ne!(ctx.tx_16x16, before, "read_symbol must adapt the CDF");
        assert_eq!(DEFAULT_TX_16X16_CDF[2][0], 18677);
    }

    /// End-to-end: drive the §8.2 `SymbolDecoder` through a `txfm_split`
    /// (binary, 3 entries with the trailing counter) default CDF row
    /// selected by §8.3.2 `txfm_split_ctx`, confirming the row matches
    /// the §9.4 default and the symbol is 0 or 1.
    #[test]
    fn decode_txfm_split_through_default_cdf() {
        // maxTxSz = TX_64X64, no-split, both neighbours present ⇒ ctx = 2.
        let ctx_idx = txfm_split_ctx(true, true, 4, 4).unwrap();
        assert_eq!(ctx_idx, 2);

        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.txfm_split;

        let row = ctx.txfm_split_cdf(ctx_idx);
        assert_eq!(row, &DEFAULT_TXFM_SPLIT_CDF[2]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(sym <= 1, "txfm_split is binary");

        assert_ne!(ctx.txfm_split, before, "read_symbol must adapt the CDF");
        assert_eq!(DEFAULT_TXFM_SPLIT_CDF[2][0], 20847);
    }

    // -----------------------------------------------------------------
    // Round 21 — inter-frame transform-type group tests.
    // -----------------------------------------------------------------

    /// §9.4 default tables: every row terminates with `1 << 15` and a
    /// zero adaptation counter (§8.2.6 contract). Locks each
    /// transcribed inter-tx-type default.
    #[test]
    fn inter_tx_type_default_tables_well_formed() {
        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for r in &DEFAULT_INTER_TX_TYPE_SET1_CDF {
            check(r);
        }
        check(&DEFAULT_INTER_TX_TYPE_SET2_CDF);
        for r in &DEFAULT_INTER_TX_TYPE_SET3_CDF {
            check(r);
        }

        // §3 dimensions.
        assert_eq!(
            DEFAULT_INTER_TX_TYPE_SET1_CDF.len(),
            INTER_TX_TYPE_SET1_SIZES
        );
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET1_CDF[0].len(), TX_TYPES + 1);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET2_CDF.len(), TX_TYPES_SET2 + 1);
        assert_eq!(
            DEFAULT_INTER_TX_TYPE_SET3_CDF.len(),
            INTER_TX_TYPE_SET3_SIZES
        );
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET3_CDF[0].len(), TX_TYPES_SET3 + 1);
    }

    /// Spot-check the first / last cumulative frequency in each §9.4
    /// inter-tx-type transcribed table. A mis-keyed digit during
    /// transcription breaks the equality.
    #[test]
    fn inter_tx_type_table_byte_anchors() {
        // Default_Inter_Tx_Type_Set1_Cdf.
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET1_CDF[0][0], 4458);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET1_CDF[0][14], 30631);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET1_CDF[1][0], 1645);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET1_CDF[1][14], 30749);

        // Default_Inter_Tx_Type_Set2_Cdf.
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET2_CDF[0], 770);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET2_CDF[10], 30529);

        // Default_Inter_Tx_Type_Set3_Cdf.
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET3_CDF[0][0], 16384);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET3_CDF[1][0], 4167);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET3_CDF[2][0], 1998);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET3_CDF[3][0], 748);
    }

    /// §8.3.1: a fresh context copies every inter-tx-type default in
    /// (independent of the source — the §9.4 array is not aliased).
    #[test]
    fn inter_tx_type_init_from_defaults_copies_tables() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.inter_tx_type_set1, DEFAULT_INTER_TX_TYPE_SET1_CDF);
        assert_eq!(ctx.inter_tx_type_set2, DEFAULT_INTER_TX_TYPE_SET2_CDF);
        assert_eq!(ctx.inter_tx_type_set3, DEFAULT_INTER_TX_TYPE_SET3_CDF);

        // Working-copy independence: mutating the context must not
        // touch the §9.4 source.
        ctx.inter_tx_type_cdf(TX_SET_INTER_2, 0).unwrap()[0] = 12345;
        assert_ne!(ctx.inter_tx_type_set2[0], DEFAULT_INTER_TX_TYPE_SET2_CDF[0]);
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET2_CDF[0], 770);
    }

    /// §8.3.2 `inter_tx_type` selection: each `(set, tx_size_sqr)`
    /// picks the expected `Default_Inter_Tx_Type_Set*_Cdf` row, the
    /// row length matches the spec's per-set symbol count, and the
    /// unreachable / `TX_SET_DCTONLY` paths return `None`.
    #[test]
    fn inter_tx_type_cdf_selected_by_set() {
        let mut ctx = TileCdfContext::new_from_defaults();

        // TX_SET_INTER_1, Tx_Size_Sqr = TX_4X4 = 0.
        let row1 = ctx.inter_tx_type_cdf(TX_SET_INTER_1, 0).unwrap();
        assert_eq!(row1.len(), TX_TYPES + 1);
        assert_eq!(row1, &DEFAULT_INTER_TX_TYPE_SET1_CDF[0]);
        // TX_SET_INTER_1, Tx_Size_Sqr = TX_8X8 = 1.
        let row1b = ctx.inter_tx_type_cdf(TX_SET_INTER_1, 1).unwrap();
        assert_eq!(row1b, &DEFAULT_INTER_TX_TYPE_SET1_CDF[1]);

        // TX_SET_INTER_2: tx_size_sqr is ignored (single row).
        let row2 = ctx.inter_tx_type_cdf(TX_SET_INTER_2, 0).unwrap();
        assert_eq!(row2.len(), TX_TYPES_SET2 + 1);
        assert_eq!(row2, &DEFAULT_INTER_TX_TYPE_SET2_CDF);
        // Same row regardless of tx_size_sqr.
        let row2b = ctx.inter_tx_type_cdf(TX_SET_INTER_2, 4).unwrap();
        assert_eq!(row2b, &DEFAULT_INTER_TX_TYPE_SET2_CDF);

        // TX_SET_INTER_3, all four reachable Tx_Size_Sqr values.
        for sqr in 0..INTER_TX_TYPE_SET3_SIZES as u32 {
            let row = ctx.inter_tx_type_cdf(TX_SET_INTER_3, sqr).unwrap();
            assert_eq!(row.len(), TX_TYPES_SET3 + 1);
            assert_eq!(row, &DEFAULT_INTER_TX_TYPE_SET3_CDF[sqr as usize]);
        }

        // TX_SET_DCTONLY: §5.11.47 forces TxType = DCT_DCT and skips
        // the inter_tx_type read; the selector returns None.
        assert!(ctx.inter_tx_type_cdf(TX_SET_DCTONLY, 0).is_none());

        // Out-of-range tx_size_sqr for Set1 / Set3.
        assert!(ctx
            .inter_tx_type_cdf(TX_SET_INTER_1, INTER_TX_TYPE_SET1_SIZES as u32)
            .is_none());
        assert!(ctx
            .inter_tx_type_cdf(TX_SET_INTER_3, INTER_TX_TYPE_SET3_SIZES as u32)
            .is_none());

        // Out-of-range set.
        assert!(ctx.inter_tx_type_cdf(99, 0).is_none());
    }

    /// §5.11.48 `get_tx_set()` (inter branch). Walk every reachable
    /// `(tx_sz_sqr, tx_sz_sqr_up, reduced_tx_set)` and confirm the
    /// helper returns the spec-prescribed set.
    #[test]
    fn inter_tx_type_set_get_tx_set_inter_branch() {
        // TX_4X4 / TX_8X8 with txSzSqrUp == txSzSqr, !reduced ⇒
        // TX_SET_INTER_1.
        assert_eq!(inter_tx_type_set(0, 0, false), TX_SET_INTER_1);
        assert_eq!(inter_tx_type_set(1, 1, false), TX_SET_INTER_1);

        // TX_16X16 (txSzSqr == 2), !reduced ⇒ TX_SET_INTER_2.
        assert_eq!(inter_tx_type_set(2, 2, false), TX_SET_INTER_2);

        // TX_32X32 (txSzSqrUp == 3), !reduced ⇒ TX_SET_INTER_3 (the
        // `txSzSqrUp == TX_32X32` branch fires before the
        // `txSzSqr == TX_16X16` branch).
        assert_eq!(inter_tx_type_set(3, 3, false), TX_SET_INTER_3);

        // Reduced-tx-set forces TX_SET_INTER_3 for any sqrUp <= 32x32.
        assert_eq!(inter_tx_type_set(0, 0, true), TX_SET_INTER_3);
        assert_eq!(inter_tx_type_set(2, 2, true), TX_SET_INTER_3);
        assert_eq!(inter_tx_type_set(3, 3, true), TX_SET_INTER_3);

        // txSzSqrUp > TX_32X32 ⇒ TX_SET_DCTONLY regardless of
        // reduced_tx_set / txSzSqr.
        assert_eq!(inter_tx_type_set(0, 4, false), TX_SET_DCTONLY);
        assert_eq!(inter_tx_type_set(3, 4, true), TX_SET_DCTONLY);

        // The rectangular-tx case (txSzSqr != txSzSqrUp): e.g.
        // a TX_4X8 block has txSzSqr = TX_4X4 (0), txSzSqrUp = TX_8X8 (1).
        // No reduced_tx_set ⇒ TX_SET_INTER_1.
        assert_eq!(inter_tx_type_set(0, 1, false), TX_SET_INTER_1);
        // A TX_16X32 / TX_32X16: txSzSqr = TX_16X16 (2),
        // txSzSqrUp = TX_32X32 (3) ⇒ TX_SET_INTER_3 (sqrUp == 32x32
        // wins over sqr == 16x16).
        assert_eq!(inter_tx_type_set(2, 3, false), TX_SET_INTER_3);
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through an
    /// `inter_tx_type` default CDF selected by the §8.3.2 selection,
    /// confirming the chosen row matches the §9.4 source, the decode
    /// lands in range and the working copy adapts.
    #[test]
    fn decode_inter_tx_type_through_default_cdf() {
        // Pick TX_SET_INTER_3 (2-symbol row, easiest to bound the
        // decoded symbol). Use a TX_8X8 block (Tx_Size_Sqr = 1).
        let set = inter_tx_type_set(1, 1, true);
        assert_eq!(set, TX_SET_INTER_3);

        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut ctx = TileCdfContext::new_from_defaults();
        let before = ctx.inter_tx_type_set3;

        let row = ctx.inter_tx_type_cdf(set, 1).unwrap();
        assert_eq!(row, &DEFAULT_INTER_TX_TYPE_SET3_CDF[1]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            sym <= 1,
            "inter_tx_type via Set3 codes one of {{ IDTX, DCT_DCT }}"
        );

        assert_ne!(
            ctx.inter_tx_type_set3, before,
            "read_symbol must adapt the CDF"
        );
        assert_eq!(DEFAULT_INTER_TX_TYPE_SET3_CDF[1][0], 4167);
    }

    // Round 22 — inter-frame interpolation-filter group tests.

    /// §8.3.1 / §9.4: the `Default_Interp_Filter_Cdf` table well-formed.
    /// Each row is `[INTERP_FILTERS + 1]` (3 cumulative frequencies +
    /// adaptation counter), the second-to-last entry is `1 << 15`, and
    /// the last entry is 0. Outer dim matches `INTERP_FILTER_CONTEXTS`.
    #[test]
    fn interp_filter_default_table_well_formed() {
        assert_eq!(DEFAULT_INTERP_FILTER_CDF.len(), INTERP_FILTER_CONTEXTS);
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[0].len(), INTERP_FILTERS + 1);

        let check = |row: &[u16]| {
            let n = row.len() - 1;
            assert_eq!(row[n - 1], 1 << 15, "cdf[N-1] must be 32768");
            assert_eq!(row[n], 0, "fresh adaptation counter must be 0");
        };
        for r in &DEFAULT_INTERP_FILTER_CDF {
            check(r);
        }
    }

    /// Spot-check the §9.4 `Default_Interp_Filter_Cdf` values byte-for-byte.
    /// A mis-keyed digit during transcription breaks the equality.
    #[test]
    fn interp_filter_default_byte_exact_values() {
        // Row 0: { 31935, 32720, 32768, 0 } — strongest bias to filter 0.
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[0][0], 31935);
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[0][1], 32720);
        // Row 2: { 422, 2938, 32768, 0 } — strong bias to filter 2.
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[2][0], 422);
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[2][1], 2938);
        // Row 7: { 20889, 25637, 32768, 0 } — mixed-neighbour anchor.
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[7][0], 20889);
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[7][1], 25637);
        // Row 8: { 31910, 32724, 32768, 0 } — vertical-dir row 0.
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[8][0], 31910);
        // Row 14: { 601, 943, 32768, 0 } — anchor near the end.
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[14][0], 601);
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[14][1], 943);
        // Row 15: { 14969, 21398, 32768, 0 } — last row.
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[15][0], 14969);
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[15][1], 21398);
    }

    /// §8.3.1: a fresh context copies the interp-filter default in
    /// (the §9.4 source is not aliased).
    #[test]
    fn interp_filter_init_from_defaults_copies_table() {
        let mut ctx = TileCdfContext::new_from_defaults();
        assert_eq!(ctx.interp_filter, DEFAULT_INTERP_FILTER_CDF);

        // Working-copy independence: mutating the context must not
        // touch the §9.4 source.
        ctx.interp_filter_cdf(0).unwrap()[0] = 12345;
        assert_ne!(ctx.interp_filter[0][0], DEFAULT_INTERP_FILTER_CDF[0][0]);
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[0][0], 31935);
    }

    /// §8.3.2 `interp_filter` ctx formula: walk the four §8.3.2
    /// branches against §5.11.x scope inputs and assert the returned
    /// `ctx` matches the spec literal exactly.
    #[test]
    fn interp_filter_context_formula() {
        // Base (dir=0, single-ref): ((0 & 1) * 2 + 0) * 4 = 0.
        // leftType == aboveType branch: ctx += leftType.
        assert_eq!(interp_filter_ctx(0, 0, 0, false).unwrap(), 0);
        assert_eq!(interp_filter_ctx(1, 1, 0, false).unwrap(), 1);
        assert_eq!(interp_filter_ctx(2, 2, 0, false).unwrap(), 2);

        // leftType == 3 (NONE) branch: ctx += aboveType.
        assert_eq!(
            interp_filter_ctx(0, INTERP_FILTER_NONE, 0, false).unwrap(),
            0,
        );
        assert_eq!(
            interp_filter_ctx(2, INTERP_FILTER_NONE, 0, false).unwrap(),
            2,
        );

        // aboveType == 3 (NONE) branch: ctx += leftType.
        assert_eq!(
            interp_filter_ctx(INTERP_FILTER_NONE, 1, 0, false).unwrap(),
            1,
        );

        // Distinct, both available: ctx += 3.
        assert_eq!(interp_filter_ctx(0, 1, 0, false).unwrap(), 3);
        assert_eq!(interp_filter_ctx(2, 0, 0, false).unwrap(), 3);

        // dir=1, single-ref: ((1 & 1) * 2 + 0) * 4 = 8.
        assert_eq!(interp_filter_ctx(0, 0, 1, false).unwrap(), 8);
        assert_eq!(interp_filter_ctx(1, 1, 1, false).unwrap(), 9);
        assert_eq!(interp_filter_ctx(0, 1, 1, false).unwrap(), 11); // ctx += 3
        assert_eq!(
            interp_filter_ctx(INTERP_FILTER_NONE, INTERP_FILTER_NONE, 1, false).unwrap(),
            11,
        );

        // dir=0, compound (RefFrame[1] > INTRA_FRAME):
        // ((0 & 1) * 2 + 1) * 4 = 4.
        assert_eq!(interp_filter_ctx(0, 0, 0, true).unwrap(), 4);
        assert_eq!(interp_filter_ctx(0, 1, 0, true).unwrap(), 7);

        // dir=1, compound: ((1 & 1) * 2 + 1) * 4 = 12. Max bucket.
        assert_eq!(interp_filter_ctx(0, 0, 1, true).unwrap(), 12);
        assert_eq!(interp_filter_ctx(2, 2, 1, true).unwrap(), 14);
        assert_eq!(interp_filter_ctx(0, 1, 1, true).unwrap(), 15);

        // Out-of-range: dir > 1, type > INTERP_FILTERS.
        assert_eq!(interp_filter_ctx(0, 0, 2, false), None);
        assert_eq!(interp_filter_ctx(INTERP_FILTERS + 1, 0, 0, false), None);
        assert_eq!(interp_filter_ctx(0, INTERP_FILTERS + 1, 0, false), None);
    }

    /// §8.3.2 `interp_filter` ctx coverage: every reachable ctx in
    /// `0..INTERP_FILTER_CONTEXTS` is hit by some valid input.
    #[test]
    fn interp_filter_context_full_coverage() {
        let mut hit = [false; INTERP_FILTER_CONTEXTS];
        for dir in 0..=1 {
            for is_comp in [false, true] {
                for above in 0..=INTERP_FILTERS {
                    for left in 0..=INTERP_FILTERS {
                        if let Some(ctx) = interp_filter_ctx(above, left, dir, is_comp) {
                            hit[ctx] = true;
                        }
                    }
                }
            }
        }
        for (i, h) in hit.iter().enumerate() {
            assert!(h, "ctx {i} unreachable");
        }
    }

    /// §8.3.2 `interp_filter` selector returns the right §9.4 row for
    /// each `ctx`, with the row length matching the spec
    /// (`INTERP_FILTERS + 1`).
    #[test]
    fn interp_filter_selector_returns_default_rows() {
        let mut ctx = TileCdfContext::new_from_defaults();
        for (i, want) in DEFAULT_INTERP_FILTER_CDF.iter().enumerate() {
            let row = ctx.interp_filter_cdf(i).unwrap();
            assert_eq!(row.len(), INTERP_FILTERS + 1);
            assert_eq!(row, want);
        }
        // Out-of-range ctx returns None.
        assert!(ctx.interp_filter_cdf(INTERP_FILTER_CONTEXTS).is_none());
    }

    /// End-to-end: drive the real §8.2 `SymbolDecoder` through an
    /// `interp_filter` default CDF selected by the §8.3.2 selection,
    /// confirming the chosen row matches the §9.4 source, the decode
    /// lands in range and the working copy adapts.
    #[test]
    fn decode_interp_filter_through_default_cdf() {
        // ctx=2 (Default_Interp_Filter_Cdf[2] = {422, 2938, 32768, 0}):
        // a strongly-toward-filter-2 row. Drive the decoder with a
        // window that lands in the high-symbol region, then assert
        // the working copy mutated and the §9.4 source stayed put.
        let bytes = [0x10u8, 0x80u8, 0x00u8, 0x00u8];
        let mut dec = SymbolDecoder::init_symbol(&bytes, 4, false).unwrap();
        let mut tile_ctx = TileCdfContext::new_from_defaults();
        let before = tile_ctx.interp_filter;

        let chosen = interp_filter_ctx(2, 2, 0, false).unwrap();
        assert_eq!(chosen, 2, "leftType==aboveType, base=0");
        let row = tile_ctx.interp_filter_cdf(chosen).unwrap();
        assert_eq!(row, &DEFAULT_INTERP_FILTER_CDF[2]);
        let sym = dec.read_symbol(row).unwrap();
        assert!(
            (sym as usize) < INTERP_FILTERS,
            "interp_filter must code a symbol in 0..INTERP_FILTERS"
        );

        assert_ne!(
            tile_ctx.interp_filter, before,
            "read_symbol must adapt the working CDF"
        );
        assert_eq!(DEFAULT_INTERP_FILTER_CDF[2], [422, 2938, 32768, 0]);
    }
}
